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

    if (!loadNvEncApi(fn_)) {
        std::cerr << "[NVENC] Failed to load NVENC driver library\n";
        return false;
    }

    // 打开 NVENC 编础会话，绑定当前 CUDA 上下文
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessParams{};
    sessParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sessParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sessParams.device = cuCtx_;
    sessParams.apiVersion = NVENCAPI_VERSION;
    NVENC_CALL(fn_.nvEncOpenEncodeSessionEx(&sessParams, &encoder_));

    // 获取超低延迟 H.264 预设配置（P3 预设 + ULTRA_LOW_LATENCY 调优档位）
    // P3 提供较好的质量/性能平衡，超低延迟调优禁用多帧缓冲和 B 帧
    NV_ENC_PRESET_CONFIG presetConfig{};
    presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
    presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;
    NVENC_CALL(fn_.nvEncGetEncodePresetConfigEx(
        encoder_, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P3_GUID,
        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &presetConfig));

    NV_ENC_CONFIG encConfig = presetConfig.presetCfg;
    encConfig.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
    encConfig.gopLength = cfg.gopLength;
    encConfig.frameIntervalP = 1; // no B-frames for low latency
    encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    encConfig.rcParams.averageBitRate = cfg.bitrateMbps * 1000000u;
    encConfig.rcParams.maxBitRate = cfg.bitrateMbps * 1000000u;
    encConfig.rcParams.vbvBufferSize = cfg.bitrateMbps * 1000000u / cfg.fps;
    encConfig.rcParams.vbvInitialDelay = encConfig.rcParams.vbvBufferSize;
    encConfig.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
    encConfig.rcParams.zeroReorderDelay = 1;
    encConfig.encodeCodecConfig.h264Config.idrPeriod = cfg.gopLength;
    encConfig.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    encConfig.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
    encConfig.encodeCodecConfig.h264Config.sliceMode = 0;
    encConfig.encodeCodecConfig.h264Config.sliceModeData = 0;

    NV_ENC_INITIALIZE_PARAMS initParams{};
    initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
    initParams.presetGUID = NV_ENC_PRESET_P3_GUID;
    initParams.encodeWidth = cfg.width;
    initParams.encodeHeight = cfg.height;
    initParams.darWidth = cfg.width;
    initParams.darHeight = cfg.height;
    initParams.frameRateNum = cfg.fps;
    initParams.frameRateDen = 1;
    initParams.enablePTD = 1;
    initParams.encodeConfig = &encConfig;
    initParams.maxEncodeWidth = 3840;
    initParams.maxEncodeHeight = 2160;
    initParams.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    NVENC_CALL(fn_.nvEncInitializeEncoder(encoder_, &initParams));

    // 创建比特流输出缓冲区（NVENC 将编础后的 NAL 数据写入此缓冲）
    NV_ENC_CREATE_BITSTREAM_BUFFER bsBuf{};
    bsBuf.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    NVENC_CALL(fn_.nvEncCreateBitstreamBuffer(encoder_, &bsBuf));
    bitstreamBuffer_ = bsBuf.bitstreamBuffer;

    // 分配 CUDA 线性暂存缓冲区（RGBA 格式，行宽 = width * 4 字节）
    // NVENC 要求输入资源必须是其通过 nvEncRegisterResource 注册过的指针
    stagingPitch_ = cfg.width * 4;
    cudaMalloc(&stagingBuffer_, stagingPitch_ * cfg.height);

    // 将暂存缓冲区注册为 NVENC 输入资源
    // 格式 ABGR 对应 OpenGL/CUDA 侧的 RGBA（字节顺序取决于平台，此处与 FBO 输出匹配）
    NV_ENC_REGISTER_RESOURCE regRes{};
    regRes.version = NV_ENC_REGISTER_RESOURCE_VER;
    regRes.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    regRes.width = cfg.width;
    regRes.height = cfg.height;
    regRes.pitch = (uint32_t)stagingPitch_;
    regRes.resourceToRegister = stagingBuffer_;
    regRes.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
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
    // 将来源线性缓冲（FrameCapture 的暂存缓冲）通过 cuMemcpy2D 拷贝到 NVENC 注册暂存缓冲
    // 使用二维拷贝以正确处理不同的行步长（pitch）
    CUDA_MEMCPY2D copyParam{};
    copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParam.srcDevice = (CUdeviceptr)devicePtr;
    copyParam.srcPitch = srcPitch;
    copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParam.dstDevice = (CUdeviceptr)stagingBuffer_;
    copyParam.dstPitch = stagingPitch_;
    copyParam.WidthInBytes = config_.width * 4;
    copyParam.Height = config_.height;
    CUresult cr = cuMemcpy2D(&copyParam);
    if (cr != CUDA_SUCCESS) {
        std::cerr << "[NVENC] cuMemcpy2D failed: " << cr << "\n";
        return false;
    }

    // 将注册的 CUDA 资源映射为 NVENC 可用的输入缓冲句柄
    NV_ENC_MAP_INPUT_RESOURCE mapRes{};
    mapRes.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    mapRes.registeredResource = registeredRes_;
    NVENC_CALL(fn_.nvEncMapInputResource(encoder_, &mapRes));

    // 提交编础请求：填充 NV_ENC_PIC_PARAMS 并调用 nvEncEncodePicture
    // NV_ENC_ERR_NEED_MORE_INPUT 表示编础器正在缓冲（流水线未复），无可输出帧，此为正常状态
    NV_ENC_PIC_PARAMS picParams{};
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.inputWidth = config_.width;
    picParams.inputHeight = config_.height;
    picParams.inputPitch = (uint32_t)stagingPitch_;
    picParams.inputBuffer = mapRes.mappedResource;
    picParams.outputBitstream = bitstreamBuffer_;
    picParams.bufferFmt = mapRes.mappedBufferFmt;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picParams.frameIdx = frameIndex_++;
    if (forceIdr_) {
        picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
        forceIdr_ = false;
    }

    NVENCSTATUS encStatus = fn_.nvEncEncodePicture(encoder_, &picParams);
    fn_.nvEncUnmapInputResource(encoder_, mapRes.mappedResource);

    if (encStatus != NV_ENC_SUCCESS && encStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
        std::cerr << "[NVENC] nvEncEncodePicture failed: " << encStatus << "\n";
        return false;
    }

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
    NV_ENC_LOCK_BITSTREAM lockBs{};
    lockBs.version = NV_ENC_LOCK_BITSTREAM_VER;
    lockBs.outputBitstream = bitstreamBuffer_;
    NVENC_CALL(fn_.nvEncLockBitstream(encoder_, &lockBs));

    bool isKey = (lockBs.pictureType == NV_ENC_PIC_TYPE_IDR ||
                  lockBs.pictureType == NV_ENC_PIC_TYPE_I);
    cb(static_cast<const uint8_t *>(lockBs.bitstreamBufferPtr),
       lockBs.bitstreamSizeInBytes, isKey);

    fn_.nvEncUnlockBitstream(encoder_, bitstreamBuffer_);
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
