#pragma once

#include "nvenc_api.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <functional>

struct CUctx_st;
typedef struct CUctx_st *CUcontext;

// 基于 NVIDIA GPU 专用硬件编码单元（NVENC ASIC）的 H.264 硬件编码器。
// 输入：来自 FrameCapture 的线性 CUDA 设备指针（RGBA 格式）。
// 输出：H.264 NAL 单元字节流，通过回调传递给 WebRTC 打包器。
class NvencEncoder {
public:
    NvencEncoder() = default;
    ~NvencEncoder();

    struct Config {
        int width = 1920;
        int height = 1080;
        int fps = 60;
        int bitrateMbps = 15;
        int gopLength = 120;        // 关键帧间隔（帧数），即 IDR 帧周期
    };

    bool initialize(CUcontext cuCtx, const Config &cfg);
    void cleanup();

    // 编码一帧：输入为线性 CUDA 设备指针（RGBA，行优先存储）。
    // 编码完成后通过回调 cb 传出 H.264 NAL 数据和是否为关键帧的标志。
    using OnEncodedCallback = std::function<void(const uint8_t *data, size_t size, bool isKeyframe)>;
    bool encodeFrame(void *devicePtr, size_t srcPitch, const OnEncodedCallback &cb);

    // 冲刷编码器：排空编码器内部延迟队列，输出所有待输出帧（流结束时调用）
    bool flush(const OnEncodedCallback &cb);

    // 强制下一帧编础 IDR 关键帧。
    // 在新连接建立或浏览器发送 PLI（画面丢失指示）时调用，使对端可立即解码新帧。
    void forceKeyframe() { forceIdr_ = true; }

    // 动态重配置编础器分辨率和码率。
    // 若分辨率发生变化（NVENC 不支持热更改），返回 false，调用方需重新初始化。
    bool reconfigure(const Config &cfg);

    int getWidth() const { return config_.width; }
    int getHeight() const { return config_.height; }

private:
    NV_ENCODE_API_FUNCTION_LIST fn_{};
    void *encoder_ = nullptr;
    CUcontext cuCtx_ = nullptr;
    Config config_{};
    bool forceIdr_ = true; // 首帧强制为 IDR，确保接收端开始解码时可立即渲染

    // 已向 NVENC 注册的 CUDA 资源句柄（与暂存缓冲绑定，生命周期与编础器一致）
    NV_ENC_REGISTERED_PTR registeredRes_ = nullptr;

    // NVENC 输出比特流缓冲区（由 SDK 内部管理，需通过 Lock/Unlock 才能访问数据）
    NV_ENC_OUTPUT_PTR bitstreamBuffer_ = nullptr;

    // CUDA 暂存缓冲区：将外部线性像素数据拷贝到 NVENC 注册输入资源
    void *stagingBuffer_ = nullptr;
    size_t stagingPitch_ = 0;

    uint32_t frameIndex_ = 0;

    bool processOutput(const OnEncodedCallback &cb);
};
