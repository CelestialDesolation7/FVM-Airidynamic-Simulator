#pragma once

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cstdint>

struct cudaGraphicsResource;

// 基于 OpenGL FBO 双缓冲 + CUDA 互操作的 GPU 侧帧捕获器。
// 工作流：将默认帧缓冲（OpenGL 主渲染目标）Blit 到专用 FBO，
// 随后在 GL 线程上通过 CUDA-GL 互操作将纹理数据拷贝到线性 CUDA 设备缓冲（暂存缓冲）。
// 编码线程可直接读取暂存缓冲，无需持有 GL 上下文。
// 双缓冲（fbo_[2] / texture_[2] / stagingBuffer_[2]）保证捕获与编码可流水线化并发执行。
class FrameCapture {
public:
    FrameCapture() = default;
    ~FrameCapture();

    bool initialize(int width, int height);
    void cleanup();

    bool resize(int width, int height);

    // 捕获一帧：Blit 默认帧缓冲 → FBO → CUDA 暂存缓冲（必须在持有 GL 上下文的线程调用）
    // srcW/srcH: 默认帧缓冲（GLFW 窗口 FB）的实际尺寸，可能与 FBO 尺寸不同
    // 当 srcW/srcH ≠ width_/height_ 时，glBlitFramebuffer 执行 GL_LINEAR 缩放
    void capture(int srcW, int srcH);

    // 返回上一帧的 CUDA 设备指针（线性 RGBA 布局，行步长为 width * 4 字节）。
    // 同时翻转双缓冲写索引：下一次 capture() 将写入另一个缓冲，
    // 使编码线程读取当前帧的同时，GL 线程可安全捕获下一帧，实现零拷贝流水线化。
    void *getDevicePtr();

    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    size_t getPitch() const { return pitch_; }

private:
    int width_ = 0;
    int height_ = 0;
    size_t pitch_ = 0;

    int writeIndex_ = 0;
    GLuint fbo_[2] = {};
    GLuint texture_[2] = {};
    cudaGraphicsResource *cudaResource_[2] = {};

    void *stagingBuffer_[2] = {};

    bool createFBO(int idx, int w, int h);
    void destroyFBO(int idx);
};
