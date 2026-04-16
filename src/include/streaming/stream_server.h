#pragma once

#include "frame_capture.h"
#include "nvenc_encoder.h"
#include "input_handler.h"

#include <rtc/rtc.hpp>
#include <rtc/rtppacketizationconfig.hpp>
#include <rtc/h264rtppacketizer.hpp>
#include <rtc/rtcpsrreporter.hpp>
#include <rtc/plihandler.hpp>
#include <httplib.h>

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <chrono>

struct CUctx_st;
typedef struct CUctx_st *CUcontext;

class StreamServer {
public:
    StreamServer() = default;
    ~StreamServer();

    struct Config {
        int httpPort = 8080;
        int wsPort = 8081;
        std::string webRoot = "web";
        int bitrateMbps = 15;
        int fps = 60;
    };

    bool initialize(const Config &cfg, int captureW, int captureH);
    void shutdown();

    void onFrameReady();

    InputHandler &inputHandler() { return inputHandler_; }

    bool isStreaming() const { return hasClient_.load(); }
    bool isRunning() const { return running_.load(); }

    void requestResize(int w, int h);

private:
    Config config_{};
    std::atomic<bool> running_{false};
    std::atomic<bool> hasClient_{false};

    FrameCapture capture_;

    NvencEncoder encoder_;
    CUcontext cuCtx_ = nullptr;
    std::mutex pipelineMutex_;
    std::mutex offerMutex_;   // 串行化 /api/offer 处理，确保同一时刻只有一条活跃连接
    std::mutex sendMutex_;    // 保护 videoTrack_/rtpConfig_：编码线程读、信令线程写，二者需互斥
    std::atomic<int> pendingW_{0};
    std::atomic<int> pendingH_{0};

    InputHandler inputHandler_;

    std::shared_ptr<rtc::PeerConnection> pc_;
    std::shared_ptr<rtc::Track> videoTrack_;
    std::shared_ptr<rtc::DataChannel> dataChannel_;
    std::shared_ptr<rtc::RtpPacketizationConfig> rtpConfig_;

    std::unique_ptr<httplib::Server> httpServer_;
    std::thread httpThread_;

    std::thread encodeThread_;
    std::mutex encodeMutex_;
    std::condition_variable encodeCV_;
    bool frameAvailable_ = false;

    uint32_t rtpSsrc_ = 1;
    std::chrono::steady_clock::time_point streamStartTime_;

    std::atomic<uint64_t> framesEncoded_{0};
    std::atomic<uint64_t> bytesSent_{0};

    std::mutex browserStatsMutex_;
    std::string lastBrowserStats_ = "{}";

    void encodeLoop();
    void applyPendingResize();
    void startHttpServer();
    void createPeerConnection(const std::string &offerSdp,
                              std::mutex &gatherMtx,
                              std::condition_variable &gatherCV,
                              bool &gatheringDone);
};
