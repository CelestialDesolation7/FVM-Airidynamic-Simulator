#include "stream_server.h"
#include <rtc/rtcpnackresponder.hpp>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <map>

StreamServer::~StreamServer() { shutdown(); }

bool StreamServer::initialize(const Config &cfg, int captureW, int captureH) {
    config_ = cfg;
    // 宽高对齐到偶数：H.264 YUV 4:2:0色度采样要求宽高均为偶数
    captureW = std::max(64, captureW & ~1);
    captureH = std::max(64, captureH & ~1);

    // 获取当前 GL 线程上活跃的 CUDA 上下文
    // 编码器和捕获器必须与渲染器共享同一 CUDA 上下文，才能在同一 GPU 显存空间直接传递像素数据
    cuCtxGetCurrent(&cuCtx_);
    if (!cuCtx_) {
        std::cerr << "[StreamServer] No CUDA context on current thread\n";
        return false;
    }

    // 初始化帧捕获器：创建双缓冲 FBO + CUDA-GL 互操作资源 + CUDA 暂存缓冲
    if (!capture_.initialize(captureW, captureH)) {
        std::cerr << "[StreamServer] Frame capture init failed\n";
        return false;
    }

    // 初始化 NVENC 编码器：加载驱动、打开编码会话、配置 CBR/超低延迟参数
    NvencEncoder::Config encCfg;
    encCfg.width       = captureW;
    encCfg.height      = captureH;
    encCfg.fps         = cfg.fps;
    encCfg.bitrateMbps = cfg.bitrateMbps;
    if (!encoder_.initialize(cuCtx_, encCfg)) {
        std::cerr << "[StreamServer] NVENC encoder init failed\n";
        return false;
    }

    running_ = true;
    streamStartTime_ = std::chrono::steady_clock::now();

    // 在指定端口启动 HTTP 服务器（独立线程），为浏览器提供静态文件和信令 API
    startHttpServer();
    // 在独立线程中运行编码循环，等待 GL 线程通过 encodeCV_ 发出帧就绪信号
    encodeThread_ = std::thread(&StreamServer::encodeLoop, this);

    std::cout << "[StreamServer] Started on http://0.0.0.0:" << cfg.httpPort << "\n";
    return true;
}

void StreamServer::shutdown() {
    if (!running_) return;

    // 步骤 1: 标记停止并唤醒编码线程退出循环
    // encodeLoop 内部每次循环都检查 running_；当前可能阻塞在 encodeCV_.wait()，
    // 调用 notify_all 后它会检测到 !running_ 并正常退出
    running_ = false;
    encodeCV_.notify_all();
    if (encodeThread_.joinable()) encodeThread_.join();  // 等待编码线程完全退出

    // 步骤 2: 关闭 WebRTC 连接（此时编码线程已完全退出，无并发访问 shared_ptr）
    // pc_->close() 将发送 DTLS 关闭杂 WebRTC 消息，浏览器收到后会进入 "closed" 状态
    // 浏览器将在 3 秒后尝试重连，但服务器已关闭，/api/offer 不再响应（AbortController 5s 超时）
    if (pc_) { pc_->close(); pc_.reset(); }
    videoTrack_.reset();
    dataChannel_.reset();
    rtpConfig_.reset();

    // 步骤 3: 停止 HTTP 服务器（阻塞直到所有待处理请求完成）
    if (httpServer_) httpServer_->stop();
    if (httpThread_.joinable()) httpThread_.join();

    // 步骤 4: 在 pipelineMutex_ 保护下清理 GPU 资源
    // 此时编码线程已结束，pipelineMutex_ 必然可获取，不会死锁
    // cleanup 顺序：编码器先清理（注销 NVENC 资源），捕获器后清理（注销 CUDA-GL 互操作资源）
    {
        std::lock_guard<std::mutex> pipelineLock(pipelineMutex_);
        encoder_.cleanup();
        capture_.cleanup();
    }

    std::cout << "[StreamServer] Shutdown complete\n";
}

void StreamServer::onFrameReady(int fbW, int fbH) {
    // 快速路径检查：未运行或无客户端时不做任何操作，保证零开销
    // 此函数在 GL 线程主循环中每帧调用，必须迅速返回以不阻塞渲染
    if (!running_ || !hasClient_) return;

    // 尝试应用待调大小的分辨率变化（若编码线程正并发执行则延迟到下帧）
    applyPendingResize();

    // 背压检查：如果编码线程尚未消费上一帧的捕获数据，跳过本帧捕获。
    // 这保证 GL 线程不会覆写编码线程正在读取的暂存缓冲，消除数据竞争。
    // NVENC 编码通常 < 1ms（1080p），因此绝大多数帧不会触发此跳过。
    if (frameInFlight_.load(std::memory_order_acquire)) return;

    // 执行帧捕获：传入实际窗口 FB 尺寸作为 blit 源矩形
    // 当 fbW/fbH ≠ FBO 尺寸时（如服务器显示器分辨率 < 浏览器请求分辨率），
    // glBlitFramebuffer 自动以 GL_LINEAR 缩放
    capture_.capture(fbW, fbH);

    // 向编码线程发出帧就绪信号
    // 编码线程在 encodeCV_.wait() 中阻塞等待，此后将被唤醒开始处理
    // GL 线程到此返回，不等待编码完成（异步处理）
    {
        std::lock_guard<std::mutex> lock(encodeMutex_);
        frameAvailable_ = true;
    }
    encodeCV_.notify_one();
    // 标记帧已提交：在编码线程完成处理前，GL 线程不再执行新的 capture()
    frameInFlight_.store(true, std::memory_order_release);
}

void StreamServer::requestResize(int w, int h) {
    // 只做原子写，立即返回，绥毫不阻塞 GL 线程
    // 真正的 resize（需要 GL 上下文 + 重建编码器）延迟到 applyPendingResize() 中执行
    // 宽高对齐加固：H.264 YUV 4:2:0 要求宽高均为偶数
    pendingW_.store(std::max(64, w & ~1));
    pendingH_.store(std::max(64, h & ~1));
}

void StreamServer::applyPendingResize() {
    int w = pendingW_.load(), h = pendingH_.load();
    if (w <= 0 || h <= 0) return;                                   // 无待处理的分辨率
    if (w == capture_.getWidth() && h == capture_.getHeight()) return; // 分辨率未变

    // 尝试非阻塞获取流水线锁
    // 如果编码线程正在执行 encodeFrame（持有 pipelineMutex_），则跳过本帧，下帧再试
    // 这确保 GL 线程对 pipelineMutex_ 的控制永不会被编码线程阻塞
    std::unique_lock<std::mutex> pipelineLock(pipelineMutex_, std::try_to_lock);
    if (!pipelineLock.owns_lock()) return; // 编码进行中，延迟 resize，下帧再试

    if (!capture_.resize(w, h)) {
        std::cerr << "[StreamServer] Frame capture resize failed: " << w << "x" << h << "\n";
        return;
    }

    // NVENC 不支持热修改分辨率，必须完整重初始化
    encoder_.cleanup();

    NvencEncoder::Config encCfg;
    encCfg.width       = w;
    encCfg.height      = h;
    encCfg.fps         = config_.fps;
    encCfg.bitrateMbps = config_.bitrateMbps;
    if (!encoder_.initialize(cuCtx_, encCfg)) {
        std::cerr << "[StreamServer] NVENC reinitialize failed after resize\n";
        return;
    }
    // 强制下一帧为 IDR，使浏览器在新分辨率下可立即解码（旧帧的 DPB 已失效）
    encoder_.forceKeyframe();
}

// ---- 编码线程 ----
// 运行在独立线程中，持续循环等待帧就绪信号，然后调用 NVENC 并通过 WebRTC 发送数据。
// 与 GL 线程的并发关系：
//   - 通过 encodeMutex_/encodeCV_ 同步帧就绪信号（GL 线程写，编码线程读）
//   - 通过 pipelineMutex_ try_lock 避免与 GL 线程的 applyPendingResize 同时操作捕获器/编码器
//   - 通过 sendMutex_ 保护 videoTrack_/rtpConfig_ 的读写互斥

void StreamServer::encodeLoop() {
    cuCtxSetCurrent(cuCtx_);  // 在编码线程上绑定同一个 CUDA 上下文（NVENC API 调用要求）

    while (running_) {
        {
            // 阻塞等待 GL 线程发出帧就绪信号（零 CPU 占用的条件变量等待）
            // 条件：frameAvailable_ == true 或 running_ == false（关机路径）
            std::unique_lock<std::mutex> lock(encodeMutex_);
            encodeCV_.wait(lock, [&] { return frameAvailable_ || !running_; });
            if (!running_) break;    // shutdown() 调用 notify_all 后退出循环
            frameAvailable_ = false; // 消费信号（每个信号对应一帧）
        }

        {
            // 尝试非阻塞获取流水线锁：如果 GL 线程正在执行 applyPendingResize，则跳过本帧
            // 这确保捕获器和编码器的 resize 操作与编码读取操作互斥
            std::unique_lock<std::mutex> pipelineLock(pipelineMutex_, std::try_to_lock);
            if (pipelineLock.owns_lock()) {
                // getDevicePtr(): 返回上一次 capture() 写入的暂存缓冲设备指针
                // 索引翻转已在 capture() 内完成，此处仅做简单读取
                void *devPtr = capture_.getDevicePtr();
                if (devPtr) {
                    encoder_.encodeFrame(devPtr, capture_.getPitch(),
                        [this](const uint8_t *data, size_t size, bool /*isKey*/) {
                        // 在 sendMutex_ 保护下将 videoTrack_/rtpConfig_/streamStartTime_
                        // 复制到局部变量，防止信令线程在编码过程中替换指针导致的悬空引用。
                        std::shared_ptr<rtc::Track> track;
                        std::shared_ptr<rtc::RtpPacketizationConfig> rtp;
                        std::chrono::steady_clock::time_point startTime;
                        {
                            std::lock_guard<std::mutex> lk(sendMutex_);
                            track = videoTrack_;
                            rtp   = rtpConfig_;
                            startTime = streamStartTime_;
                        }
                        if (!track || !track->isOpen() || !rtp) return;

                        // 计算 RTP 时间戳：单位为 90000Hz H.264 时钟
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                            now - startTime).count();
                        rtp->timestamp = rtp->startTimestamp +
                            uint32_t(elapsed * 90000ULL / 1000000ULL);

                        try {
                            track->send(
                                reinterpret_cast<const std::byte *>(data), size);
                            framesEncoded_.fetch_add(1, std::memory_order_relaxed);
                            bytesSent_.fetch_add(size, std::memory_order_relaxed);
                        } catch (const std::exception &e) {
                            std::cerr << "[WebRTC] send error: " << e.what() << std::endl;
                        }
                    });
                }
            }
        }

        // 释放背压：编码完成（或跳过），GL 线程可安全地向另一个缓冲执行下一帧捕获。
        // cuMemcpy2D 从 capture staging buffer 到 encoder staging buffer 已完成，
        // capture 的缓冲不再被读取。
        frameInFlight_.store(false, std::memory_order_release);
    }
}

// ---- SDP 解析辅助函数 ----
// SDP（Session Description Protocol）是 WebRTC 信令的核心格式，描述双方的媒体能力和网络信息。
// 这组函数从浏览器 Offer SDP 中提取 H.264 编解码器参数，选出与 NVENC 输出最匹配的一项。

// 从 SDP 中提取指定媒体类型（如 "video"）的 a=mid: 属性值
// mid（媒体标识符）用于媒体多路复用时（BUNDLE）区分不同媒体流
static std::string extractMid(const std::string &sdp, const std::string &mediaType) {
    std::istringstream ss(sdp);
    std::string line;
    bool inSection = false;
    std::string prefix = "m=" + mediaType;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.compare(0, prefix.size(), prefix) == 0) inSection = true;
        else if (line.compare(0, 2, "m=") == 0) inSection = false;
        if (inSection && line.compare(0, 6, "a=mid:") == 0)
            return line.substr(6);
    }
    return "";
}

struct H264CodecInfo {
    int payloadType = -1;
    std::string fmtp;
};

// 从 SDP Offer 的 video 段中解析所有 H.264 编码条目（rtpmap + fmtp）
// SDP 示例：
//   a=rtpmap:97 H264/90000
//   a=fmtp:97 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
static std::vector<H264CodecInfo> findAllH264(const std::string &sdp) {
    std::istringstream ss(sdp);
    std::string line;
    bool inVideo = false;

    std::vector<int> h264pts;
    std::map<int, std::string> fmtpMap;

    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.compare(0, 7, "m=video") == 0) inVideo = true;
        else if (line.compare(0, 2, "m=") == 0) inVideo = false;
        if (!inVideo) continue;

        if (line.compare(0, 9, "a=rtpmap:") == 0) {
            auto rest = line.substr(9);
            auto sp = rest.find(' ');
            if (sp != std::string::npos) {
                int pt = std::stoi(rest.substr(0, sp));
                auto codec = rest.substr(sp + 1);
                if (codec.find("H264") == 0 || codec.find("h264") == 0)
                    h264pts.push_back(pt);
            }
        }
        if (line.compare(0, 7, "a=fmtp:") == 0) {
            auto rest = line.substr(7);
            auto sp = rest.find(' ');
            if (sp != std::string::npos) {
                int pt = std::stoi(rest.substr(0, sp));
                fmtpMap[pt] = rest.substr(sp + 1);
            }
        }
    }

    std::vector<H264CodecInfo> result;
    for (int pt : h264pts) {
        H264CodecInfo info;
        info.payloadType = pt;
        auto it = fmtpMap.find(pt);
        if (it != fmtpMap.end()) info.fmtp = it->second;
        result.push_back(info);
    }
    return result;
}

// 从候选 H.264 编解码器中挑选与编码器输出最匹配的一项。
// 评分依据：
//   packetization-mode=1  (+100分): 允许 RTP 将超 MTU 的 NALU 切分为 FU-A 分片
//                                       与 NVENC 输出的大 NALU 格式完全匹配
//   profile-level-id 42e0 (+50分): 约束基线 Profile，硬解兼容性最广
static H264CodecInfo pickBestH264(const std::vector<H264CodecInfo> &codecs) {
    H264CodecInfo best;
    int bestScore = -1;
    for (auto &c : codecs) {
        int score = 0;
        if (c.fmtp.find("packetization-mode=1") != std::string::npos) score += 100;
        if (c.fmtp.find("42e0") != std::string::npos) score += 50;
        else if (c.fmtp.find("42c0") != std::string::npos) score += 40;
        else if (c.fmtp.find("4200") != std::string::npos) score += 30;
        else if (c.fmtp.find("4d00") != std::string::npos) score += 20;
        else if (c.fmtp.find("6400") != std::string::npos) score += 10;
        if (score > bestScore) { bestScore = score; best = c; }
    }
    return best;
}

// ---- HTTP 服务器初始化 ----

void StreamServer::startHttpServer() {
    httpServer_ = std::make_unique<httplib::Server>();

    httpServer_->set_mount_point("/", config_.webRoot);

    httpServer_->Get("/api/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    httpServer_->Get("/api/diag", [this](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::ostringstream ss;
        ss << "{\"connected\":" << (hasClient_.load() ? "true" : "false")
           << ",\"framesEncoded\":" << framesEncoded_.load()
           << ",\"bytesSent\":" << bytesSent_.load()
           << ",\"captureW\":" << capture_.getWidth()
           << ",\"captureH\":" << capture_.getHeight()
           << ",\"trackOpen\":" << (videoTrack_ && videoTrack_->isOpen() ? "true" : "false")
           << "}";
        res.set_content(ss.str(), "application/json");
    });

    httpServer_->Post("/api/browser-stats", [this](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        {
            std::lock_guard<std::mutex> lk(browserStatsMutex_);
            lastBrowserStats_ = req.body;
        }
        res.set_content("{\"ok\":true}", "application/json");
    });

    httpServer_->Get("/api/browser-stats", [this](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::lock_guard<std::mutex> lk(browserStatsMutex_);
        res.set_content(lastBrowserStats_, "application/json");
    });

    httpServer_->Options("/(.*)", [](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    httpServer_->Post("/api/offer", [this](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        std::cout << "[WebRTC] Received offer (" << req.body.size() << " bytes)\n";

        // 串行化连接建立流程：同一时刻只允许一条活跃连接。
        // 若第二个浏览器窗口同时发送 Offer，将在此阻塞，等待前一个 createPeerConnection
        // 完成并拆除旧连接后，再建立新连接——实现“新窗口踢旧窗口”语义。
        std::lock_guard<std::mutex> offerLk(offerMutex_);

        std::mutex mtx;
        std::condition_variable cv;
        bool gatheringDone = false;

        createPeerConnection(req.body, mtx, cv, gatheringDone);

        {
            std::unique_lock<std::mutex> lk(mtx);
            if (!gatheringDone) {
                cv.wait_for(lk, std::chrono::seconds(5), [&] { return gatheringDone; });
            }
        }

        if (pc_ && pc_->localDescription()) {
            std::string sdp = pc_->localDescription()->generateSdp();
            std::cout << "[WebRTC] Sending answer (" << sdp.size() << " bytes)\n";
            res.set_content(sdp, "application/sdp");
        } else {
            std::cerr << "[WebRTC] Failed to generate answer\n";
            res.status = 503;
        }
    });

    httpThread_ = std::thread([this] {
        httpServer_->listen("0.0.0.0", config_.httpPort);
    });
}

// ---- WebRTC 对等连接建立 ----

void StreamServer::createPeerConnection(const std::string &offerSdp,
                                         std::mutex &gatherMtx,
                                         std::condition_variable &gatherCV,
                                         bool &gatheringDone) {
    if (pc_) {
        hasClient_ = false;
        pc_->close();
        pc_.reset();
        {
            std::lock_guard<std::mutex> lk(sendMutex_);
            videoTrack_.reset();
            rtpConfig_.reset();
        }
        dataChannel_.reset();
    }

    rtc::Configuration rtcConfig;
    pc_ = std::make_shared<rtc::PeerConnection>(rtcConfig);

    pc_->onGatheringStateChange(
        [&gatherMtx, &gatherCV, &gatheringDone](rtc::PeerConnection::GatheringState state) {
            if (state == rtc::PeerConnection::GatheringState::Complete) {
                std::lock_guard<std::mutex> lk(gatherMtx);
                gatheringDone = true;
                gatherCV.notify_one();
            }
        });

    pc_->onStateChange([this](rtc::PeerConnection::State state) {
        const char *names[] = {"New", "Connecting", "Connected", "Disconnected", "Failed", "Closed"};
        int idx = static_cast<int>(state);
        std::cout << "[WebRTC] State: " << (idx < 6 ? names[idx] : "?") << "\n";
        if (state == rtc::PeerConnection::State::Connected) {
            hasClient_ = true;
            {
                // streamStartTime_ 在 encodeLoop 回调中被读取（编码线程），
                // 此处在 libdatachannel 内部线程中写入，需 sendMutex_ 保护
                std::lock_guard<std::mutex> lk(sendMutex_);
                streamStartTime_ = std::chrono::steady_clock::now();
            }
            encoder_.forceKeyframe();
        } else if (state == rtc::PeerConnection::State::Disconnected ||
                   state == rtc::PeerConnection::State::Failed ||
                   state == rtc::PeerConnection::State::Closed) {
            hasClient_ = false;
        }
    });

    // 解析浏览器 Offer SDP，从其网络能力列表中挑选最匹配的 H.264 编解码参数
    std::string videoMid = extractMid(offerSdp, "video");
    if (videoMid.empty()) videoMid = "0";  // 如果 SDP 中无 a=mid，默认使用 "0"

    auto allH264 = findAllH264(offerSdp);
    auto chosen = pickBestH264(allH264);

    int h264pt = chosen.payloadType > 0 ? chosen.payloadType : 109;
    std::string fmtp = chosen.fmtp.empty()
        ? "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"
        : chosen.fmtp;

    std::cout << "[WebRTC] Negotiated: mid=" << videoMid
              << " PT=" << h264pt << " fmtp=" << fmtp << "\n";

    // 用协商所得的 Payload Type 和 fmtp 参数构造视频 Track，方向为 SendOnly（仅服务器推流）
    rtc::Description::Video media(videoMid, rtc::Description::Direction::SendOnly);
    media.addH264Codec(h264pt, fmtp);
    media.addSSRC(rtpSsrc_, "video-stream");  // SSRC：同步源标识符，唯一标识这个视频流
    auto newTrack = pc_->addTrack(media);

    // Media handler chain: H264RtpPacketizer → RtcpSrReporter → NackResponder → PliHandler
    // 每层只处理自己关心的信号，其余透传：
    //   H264RtpPacketizer: 将 NALU 切分为 RTP 包（加序列号/时间戳）
    //   RtcpSrReporter:    定期发送 RTCP SR（发送者报告），供浏览器校准播放时钟
    //   RtcpNackResponder: 缓存已发 RTP 包，收到 NACK 时重发丢失包
    //   PliHandler:        收到 PLI（画面丢失指示）时调用 forceKeyframe()
    auto newRtpConfig = std::make_shared<rtc::RtpPacketizationConfig>(
        rtpSsrc_, "video-stream", uint8_t(h264pt),
        rtc::H264RtpPacketizer::ClockRate);  // ClockRate = 90000 Hz，H.264 标准时钟频率

    auto packetizer = std::make_shared<rtc::H264RtpPacketizer>(
        rtc::H264RtpPacketizer::Separator::StartSequence, newRtpConfig);
    // Separator::StartSequence: 以 H.264 起始码（0x00 00 00 01）为 NALU 分隔符
    // NVENC 输出的比特流使用起始码格式（Annex B），与此 Separator 一致

    auto srReporter = std::make_shared<rtc::RtcpSrReporter>(newRtpConfig);
    packetizer->addToChain(srReporter);

    auto nackResponder = std::make_shared<rtc::RtcpNackResponder>();
    srReporter->addToChain(nackResponder);

    auto pliHandler = std::make_shared<rtc::PliHandler>([this]() {
        // PLI（Picture Loss Indication）：浏览器解码失败时发送，请求一个新的关键帧
        // 强制 IDR 后，浏览器就能恢复解码
        encoder_.forceKeyframe();
    });
    nackResponder->addToChain(pliHandler);

    newTrack->setMediaHandler(packetizer);  // 将责任链根节点绑定到 Track

    newTrack->onOpen([this] {
        std::cout << "[WebRTC] Video track opened\n";
        // Track 就绪即强制一个 IDR，使浏览器可从第一帧开始正常解码
        encoder_.forceKeyframe();
    });

    // 在 sendMutex_ 保护下原子性地发布 track 和 rtpConfig_，
    // 确保编码线程看到的永远是一对完整有效（或同为 null）的指针。
    {
        std::lock_guard<std::mutex> lk(sendMutex_);
        videoTrack_ = newTrack;
        rtpConfig_ = newRtpConfig;
    }

    pc_->onDataChannel([this](std::shared_ptr<rtc::DataChannel> dc) {
        std::cout << "[WebRTC] DataChannel opened: " << dc->label() << "\n";
        dataChannel_ = dc;
        dc->onMessage([this](auto msg) {
            if (std::holds_alternative<std::string>(msg)) {
                inputHandler_.onMessage(std::get<std::string>(msg));
            }
        });
    });

    pc_->setRemoteDescription(rtc::Description(offerSdp, rtc::Description::Type::Offer));
}
