#include "nvenc_encoder.h"
#include <cuda.h>
#include <iostream>
#include <cstring>

#define NVENC_CALL(call)                                                       \
    do {                                                                       \
        NVENCSTATUS s = (call);                                                \
        if (s != NV_ENC_SUCCESS) {                                             \
            std::cerr << "[NVENC] Error " << s << " at " << #call << "\n";     \
            return false;                                                      \
        }                                                                      \
    } while (0)

NvencEncoder::~NvencEncoder() { cleanup(); }

bool NvencEncoder::initialize(CUcontext cuCtx, const Config &cfg) {
    cuCtx_ = cuCtx;
    config_ = cfg;

    // --- 步骤 A: 动态加载 NVENC 驱动库，填充函数指针表 ---
    // NVENC API 通过运行时加载实现（nvEncodeAPI64.dll），
    // fn_ 是包含所有 NVENC 函数指针的结构体，后续所有编码调用均通过 fn_.xxx() 完成
    if (!loadNvEncApi(fn_)) {
        std::cerr << "[NVENC] Failed to load NVENC driver library\n";
        return false;
    }

    // --- 步骤 B: 打开 NVENC 编码会话，绑定当前 CUDA 上下文 ---
    // 编码会话绑定了 GPU 设备和 CUDA 上下文，后续所有 NVENC 调用均在此会话上进行
    // deviceType = CUDA: 使用 CUDA 设备作为数据源（对应 cuMemcpy2D 输入路径）
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessParams{};
    sessParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sessParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sessParams.device = cuCtx_;          // 绑定与 GL/FrameCapture 相同的 CUDA 上下文
    sessParams.apiVersion = NVENCAPI_VERSION;
    NVENC_CALL(fn_.nvEncOpenEncodeSessionEx(&sessParams, &encoder_));

    // --- 步骤 C: 获取超低延迟预设配置（P3 + ULTRA_LOW_LATENCY）---
    // P3 预设在质量和性能之间取得良好平衡
    // ULTRA_LOW_LATENCY 调优：禁用多帧看前/参考缓冲，编码完成立即输出
    NV_ENC_PRESET_CONFIG presetConfig{};
    presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
    presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;
    NVENC_CALL(fn_.nvEncGetEncodePresetConfigEx(
        encoder_, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P3_GUID,
        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &presetConfig));

    // --- 步骤 D: 在预设基础上覆盖关键参数 ---
    NV_ENC_CONFIG encConfig = presetConfig.presetCfg;
    encConfig.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
    // Baseline Profile: 无 B 帧、无 CABAC，解码复杂度最低，浏览器硬解兼容性最广
    encConfig.gopLength = cfg.gopLength;
    // gopLength = 120（60fps 下每 2 秒一个 IDR 帧）
    // IDR 帧是完全独立的关键帧，之后的帧可从此处开始解码；间隔越长，平均码率越低
    encConfig.frameIntervalP = 1;
    // frameIntervalP = 1: 相邻参考帧间距为 1，即每帧均为 P 帧（参考前一帧），无 B 帧
    // B 帧需要缓存后续帧才能编码，会引入数帧延迟，与超低延迟目标冲突
    encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    // CBR (Constant Bit Rate): 恒定码率，确保网络带宽占用稳定可预测
    encConfig.rcParams.averageBitRate = cfg.bitrateMbps * 1000000u;
    encConfig.rcParams.maxBitRate     = cfg.bitrateMbps * 1000000u;
    encConfig.rcParams.vbvBufferSize  = cfg.bitrateMbps * 1000000u / cfg.fps;
    // VBV 缓冲大小 = 1 帧的码量（bitrate / fps），使编码器以帧为单位严格控制码率
    // VBV (Video Buffering Verifier) 是 H.264 码率控制模型中的虚拟解码缓冲区
    encConfig.rcParams.vbvInitialDelay = encConfig.rcParams.vbvBufferSize;
    encConfig.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
    // 禁用多轮编码（用于超低延迟：单轮编码避免额外延迟）
    encConfig.rcParams.zeroReorderDelay = 1;
    // zeroReorderDelay = 1: 编码器不等待后续帧参考，每帧编码完立即输出 NAL 数据
    // 与 frameIntervalP=1 协同工作，确保每帧延迟 = 1 帧编码时间
    encConfig.encodeCodecConfig.h264Config.idrPeriod    = cfg.gopLength;
    encConfig.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    // repeatSPSPPS = 1: 每个 IDR 帧前附加 SPS（序列参数集）和 PPS（图像参数集）
    // 使网络中途加入的浏览器或从任意 IDR 帧开始解码时无需获取初始化信息
    encConfig.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
    // CAVLC (Context-Adaptive Variable-Length Coding): 相比 CABAC 解码更快
    // CABAC 虽然压缩率更高，但解码延迟更大，且不在 Baseline Profile 中
    encConfig.encodeCodecConfig.h264Config.sliceMode     = 0;
    encConfig.encodeCodecConfig.h264Config.sliceModeData = 0;
    // 单 slice 模式（整帧一个 slice），简化 RTP 打包逻辑

    // --- 步骤 E: 填充初始化参数并初始化编码器 ---
    NV_ENC_INITIALIZE_PARAMS initParams{};
    initParams.version       = NV_ENC_INITIALIZE_PARAMS_VER;
    initParams.encodeGUID    = NV_ENC_CODEC_H264_GUID;    // 使用 H.264 编码标准
    initParams.presetGUID    = NV_ENC_PRESET_P3_GUID;     // P3 质量预设
    initParams.encodeWidth   = cfg.width;
    initParams.encodeHeight  = cfg.height;
    initParams.darWidth      = cfg.width;   // DAR（显示宽高比），1:1 像素比
    initParams.darHeight     = cfg.height;
    initParams.frameRateNum  = cfg.fps;     // 帧率分子
    initParams.frameRateDen  = 1;           // 帧率分母（fps = frameRateNum / frameRateDen）
    initParams.enablePTD     = 1;           // 启用 Picture Type Decision，自动决定帧类型
    initParams.encodeConfig  = &encConfig;
    initParams.maxEncodeWidth  = 3840;      // 预留最大分辨率（支持动态 resize 到 4K）
    initParams.maxEncodeHeight = 2160;
    initParams.tuningInfo    = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    NVENC_CALL(fn_.nvEncInitializeEncoder(encoder_, &initParams));

    // --- 步骤 F: 创建输出比特流缓冲区 ---
    // NVENC 将编码后的 H.264 NAL 数据写入此缓冲区（由 SDK 在驱动侧分配和管理）
    // 访问数据需通过 nvEncLockBitstream / nvEncUnlockBitstream 对进行
    NV_ENC_CREATE_BITSTREAM_BUFFER bsBuf{};
    bsBuf.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    NVENC_CALL(fn_.nvEncCreateBitstreamBuffer(encoder_, &bsBuf));
    bitstreamBuffer_ = bsBuf.bitstreamBuffer;

    // --- 步骤 G: 分配 CUDA 输入暂存缓冲区 ---
    // NVENC 注册资源时需要一个已分配的 CUDA 设备指针作为输入
    // 行步长 = width * 4（RGBA 每像素 4 字节，无对齐填充）
    stagingPitch_ = cfg.width * 4;
    cudaMalloc(&stagingBuffer_, stagingPitch_ * cfg.height);

    // --- 步骤 H: 将暂存缓冲区注册为 NVENC 输入资源 ---
    // 注册后 NVENC SDK 内部维护对该内存的描述符（registeredRes_）
    // 每次编码时通过 nvEncMapInputResource 将其映射为当次编码的输入句柄
    // bufferFormat = ABGR：对应 OpenGL/CUDA 侧的 RGBA（字节序名称差异来自端序约定）
    NV_ENC_REGISTER_RESOURCE regRes{};
    regRes.version           = NV_ENC_REGISTER_RESOURCE_VER;
    regRes.resourceType      = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    regRes.width             = cfg.width;
    regRes.height            = cfg.height;
    regRes.pitch             = (uint32_t)stagingPitch_;
    regRes.resourceToRegister = stagingBuffer_;
    regRes.bufferFormat      = NV_ENC_BUFFER_FORMAT_ABGR;
    NVENC_CALL(fn_.nvEncRegisterResource(encoder_, &regRes));
    registeredRes_ = regRes.registeredResource;

    frameIndex_ = 0;
    forceIdr_ = true;
    std::cout << "[NVENC] Encoder initialized: " << cfg.width << "x" << cfg.height
              << " @ " << cfg.fps << "fps, " << cfg.bitrateMbps << "Mbps\n";
    return true;
}

void NvencEncoder::cleanup() {
    if (!encoder_) return;

    if (registeredRes_) {
        fn_.nvEncUnregisterResource(encoder_, registeredRes_);
        registeredRes_ = nullptr;
    }
    if (bitstreamBuffer_) {
        fn_.nvEncDestroyBitstreamBuffer(encoder_, bitstreamBuffer_);
        bitstreamBuffer_ = nullptr;
    }
    if (stagingBuffer_) {
        cudaFree(stagingBuffer_);
        stagingBuffer_ = nullptr;
    }
    fn_.nvEncDestroyEncoder(encoder_);
    encoder_ = nullptr;
}

bool NvencEncoder::encodeFrame(void *devicePtr, size_t srcPitch, const OnEncodedCallback &cb) {
    // === 步骤 1: GPU 内部 2D 拷贝（FrameCapture 暂存 → NVENC 注册暂存）===
    // FrameCapture 的 stagingBuffer_（getDevicePtr 返回）是捕获器自己管理的内存，
    // NVENC 只接受通过 nvEncRegisterResource 注册过的指针，故需要此次拷贝。
    // cuMemcpy2D 正确处理行步长差异：srcPitch 来自捕获器，dstPitch 由编码器自己分配。
    // 全程在 GPU 显存内进行（DeviceToDevice），不经 PCIe 总线，约 0.1ms（1080p）。
    CUDA_MEMCPY2D copyParam{};
    copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParam.srcDevice     = (CUdeviceptr)devicePtr;
    copyParam.srcPitch      = srcPitch;
    copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParam.dstDevice     = (CUdeviceptr)stagingBuffer_;
    copyParam.dstPitch      = stagingPitch_;
    copyParam.WidthInBytes  = config_.width * 4;   // 每行有效数据字节数（RGBA）
    copyParam.Height        = config_.height;
    CUresult cr = cuMemcpy2D(&copyParam);
    if (cr != CUDA_SUCCESS) {
        std::cerr << "[NVENC] cuMemcpy2D failed: " << cr << "\n";
        return false;
    }

    // === 步骤 2: 映射注册资源为本次编码的输入句柄 ===
    // nvEncMapInputResource 将 registeredRes_（常驻描述符）转为一次性使用的 mappedResource
    // 在 Unmap 之前，NVENC SDK 持有该资源的使用权，其他操作不得修改该内存
    NV_ENC_MAP_INPUT_RESOURCE mapRes{};
    mapRes.version            = NV_ENC_MAP_INPUT_RESOURCE_VER;
    mapRes.registeredResource = registeredRes_;
    NVENC_CALL(fn_.nvEncMapInputResource(encoder_, &mapRes));

    // === 步骤 3: 提交编码请求 ===
    // nvEncEncodePicture 是异步提交（在超低延迟模式下为同步，因为 zeroReorderDelay=1）
    // NV_ENC_ERR_NEED_MORE_INPUT: 编码器正在建立内部流水线（通常只在首帧出现），
    //   此时无输出，是正常状态；返回 NV_ENC_SUCCESS 时则有数据可读取。
    NV_ENC_PIC_PARAMS picParams{};
    picParams.version       = NV_ENC_PIC_PARAMS_VER;
    picParams.inputWidth    = config_.width;
    picParams.inputHeight   = config_.height;
    picParams.inputPitch    = (uint32_t)stagingPitch_;
    picParams.inputBuffer   = mapRes.mappedResource;  // 本帧输入像素数据
    picParams.outputBitstream = bitstreamBuffer_;     // 输出 NAL 写入此缓冲
    picParams.bufferFmt     = mapRes.mappedBufferFmt; // 格式由 Map 操作确定（ABGR）
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME; // 逐行扫描帧（非隔行）
    picParams.frameIdx      = frameIndex_++;
    if (forceIdr_) {
        picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
        // FORCEIDR: 强制本帧为 IDR 帧（完全独立，不参考任何前帧）
        // 在新连接建立（浏览器无解码上下文）或 PLI 后立即调用，使对端可从此帧开始解码
        forceIdr_ = false;
    }

    NVENCSTATUS encStatus = fn_.nvEncEncodePicture(encoder_, &picParams);
    fn_.nvEncUnmapInputResource(encoder_, mapRes.mappedResource);  // 立即释放映射

    if (encStatus != NV_ENC_SUCCESS && encStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
        std::cerr << "[NVENC] nvEncEncodePicture failed: " << encStatus << "\n";
        return false;
    }

    // === 步骤 4: 取出编码结果，调用回调 ===
    // 只有 NV_ENC_SUCCESS 时才有数据输出（NEED_MORE_INPUT 时无输出）
    if (encStatus == NV_ENC_SUCCESS) {
        return processOutput(cb);
    }
    return true;
}

bool NvencEncoder::flush(const OnEncodedCallback &cb) {
    NV_ENC_PIC_PARAMS picParams{};
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    fn_.nvEncEncodePicture(encoder_, &picParams);
    return processOutput(cb);
}

bool NvencEncoder::processOutput(const OnEncodedCallback &cb) {
    // 锁定输出比特流缓冲区，获取 NAL 数据的 CPU 可读指针和字节数
    // Lock/Unlock 是 NVENC SDK 的访问协议，Lock 期间驱动不得写入该缓冲
    NV_ENC_LOCK_BITSTREAM lockBs{};
    lockBs.version         = NV_ENC_LOCK_BITSTREAM_VER;
    lockBs.outputBitstream = bitstreamBuffer_;
    NVENC_CALL(fn_.nvEncLockBitstream(encoder_, &lockBs));

    // 判断是否为关键帧（IDR 或 I 帧）
    // IDR: Instantaneous Decoding Refresh，完全独立帧，重置解码器参考帧列表
    // I 帧: 帧内预测帧，不依赖其他帧但不重置参考列表（本配置下 IDR=I）
    bool isKey = (lockBs.pictureType == NV_ENC_PIC_TYPE_IDR ||
                  lockBs.pictureType == NV_ENC_PIC_TYPE_I);
    // 调用回调：将 NAL 字节流传给 stream_server.cpp 中的 encodeLoop lambda
    // 回调内会计算 RTP 时间戳并调用 track->send()
    cb(static_cast<const uint8_t *>(lockBs.bitstreamBufferPtr),
       lockBs.bitstreamSizeInBytes, isKey);

    fn_.nvEncUnlockBitstream(encoder_, bitstreamBuffer_);  // 归还缓冲区给驱动
    return true;
}

bool NvencEncoder::reconfigure(const Config &cfg) {
    // 分辨率变化需要完整重新初始化（NVENC 热重配置仅支持码率等参数，不支持尺寸变化）
    if (cfg.width != config_.width || cfg.height != config_.height) {
        return false;
    }
    config_ = cfg;
    return true;
}
