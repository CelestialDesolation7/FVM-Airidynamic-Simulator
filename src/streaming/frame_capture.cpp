#include "frame_capture.h"
#include <cuda_gl_interop.h>
#include <iostream>

#define GL_CHECK(msg)                                         \
    do {                                                      \
        GLenum e = glGetError();                              \
        if (e != GL_NO_ERROR)                                 \
            std::cerr << "[FrameCapture] GL error " << e      \
                      << " at " << msg << std::endl;          \
    } while (0)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[FrameCapture] CUDA error: "                       \
                      << cudaGetErrorString(err) << " (" << #call << ")\n";  \
            return false;                                                    \
        }                                                                    \
    } while (0)

FrameCapture::~FrameCapture() { cleanup(); }

bool FrameCapture::createFBO(int idx, int w, int h) {
    // --- 第一步：创建 RGBA8 纹理（FBO 的颜色附件）---
    // GL_RGBA8 = 每通道 8 位无符号整数，总计 32 位/像素，与 NVENC 输入格式匹配
    glGenTextures(1, &texture_[idx]);
    glBindTexture(GL_TEXTURE_2D, texture_[idx]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // --- 第二步：创建 FBO 并将纹理挂载为颜色附件 ---
    // FBO（帧缓冲对象）是 OpenGL 的离屏渲染目标
    // 将纹理挂载为 COLOR_ATTACHMENT0 后，glBlitFramebuffer 可将数据写入该纹理
    glGenFramebuffers(1, &fbo_[idx]);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_[idx]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_[idx], 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[FrameCapture] FBO incomplete: " << status << std::endl;
        return false;
    }

    // --- 第三步：向 CUDA 注册该 OpenGL 纹理（CUDA-GL 互操作）---
    // ReadOnly 标志：CUDA 只读取该纹理，不会写入，允许 OpenGL 渲染时持有所有权
    // 注册后可通过 cudaGraphicsMapResources 在 CUDA 代码中访问 OpenGL 显存
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cudaResource_[idx], texture_[idx], GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsReadOnly));

    // --- 第四步：分配线性 CUDA 设备内存（暂存缓冲）---
    // 这块内存是编码线程实际读取的数据来源
    // 行步长（pitch_）= width * 4 字节（RGBA 每像素 4 字节，线性排列无填充）
    // NVENC 要求输入为线性布局；CUDA 纹理数组为 swizzle 布局，因此需要这次额外拷贝
    CUDA_CHECK(cudaMalloc(&stagingBuffer_[idx], (size_t)w * h * 4));

    return true;
}

void FrameCapture::destroyFBO(int idx) {
    // 注销顺序需与创建顺序相反，且必须在 CUDA-GL 互操作上下文仍然有效时调用
    // 若先销毁 GL 上下文再调用此函数，cudaGraphicsUnregisterResource 会崩溃
    if (cudaResource_[idx]) {
        cudaGraphicsUnregisterResource(cudaResource_[idx]);  // 解除 CUDA 对 GL 纹理的引用
        cudaResource_[idx] = nullptr;
    }
    if (stagingBuffer_[idx]) {
        cudaFree(stagingBuffer_[idx]);  // 释放 CUDA 线性暂存缓冲
        stagingBuffer_[idx] = nullptr;
    }
    if (fbo_[idx]) { glDeleteFramebuffers(1, &fbo_[idx]); fbo_[idx] = 0; }
    if (texture_[idx]) { glDeleteTextures(1, &texture_[idx]); texture_[idx] = 0; }
}

bool FrameCapture::initialize(int width, int height) {
    width_ = width;
    height_ = height;
    pitch_ = width * 4;  // 每行字节数：RGBA 每像素 4 字节，无填充（tight packing）

    // 创建两套独立的 FBO + 暂存缓冲，实现双缓冲流水线
    // fbo_[0]/stagingBuffer_[0] 与 fbo_[1]/stagingBuffer_[1] 交替使用：
    //   - GL 线程写 fbo_[writeIndex_] → stagingBuffer_[writeIndex_]
    //   - 编码线程读 stagingBuffer_[1-writeIndex_]（上一帧）
    for (int i = 0; i < 2; ++i) {
        if (!createFBO(i, width, height)) return false;
    }
    writeIndex_ = 0;  // 初始写入缓冲 0
    return true;
}

void FrameCapture::cleanup() {
    for (int i = 0; i < 2; ++i) destroyFBO(i);
    width_ = height_ = 0;
}

bool FrameCapture::resize(int width, int height) {
    if (width == width_ && height == height_) return true;
    cleanup();
    return initialize(width, height);
}

void FrameCapture::capture(int srcW, int srcH) {
    // 【关键】将 writeIndex_ 缓存到局部变量，防止编码线程在 capture() 执行过程中
    // 通过 getDevicePtr() 修改 writeIndex_，导致 blit/map/copy/unmap 操作使用不同的缓冲索引，
    // 造成数据错乱（闪回）或资源泄漏（map buffer[0] 却 unmap buffer[1]）
    int idx = writeIndex_;

    // 第一步：将默认帧缓冲（前台渲染结果）Blit 到捕获专用 FBO。
    // 源矩形 (0, 0, srcW, srcH)：GLFW 窗口 FB 的实际尺寸
    // 目标矩形 (0, height_, width_, 0)：FBO 尺寸（= 浏览器请求的物理像素分辨率）
    //   Y 方向翻转补偿 OpenGL 纹理坐标从左下角起点的特性
    // 当 srcW/srcH ≠ width_/height_ 时（如窗口被显示器截断），GL_LINEAR 自动完成缩放
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_[idx]);
    glBlitFramebuffer(0, 0, srcW, srcH,
                      0, height_, width_, 0,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    // 第二步：映射 CUDA 资源，将 FBO 纹理数据拷贝到线性暂存缓冲，再解除映射。
    // 全部操作必须在持有 GL 上下文的 GL 线程上执行（CUDA-GL 互操作约束）。
    // 拷贝完成后编码线程即可安全读取暂存缓冲，无需等待 GL 上下文。
    cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_[idx], 0);
    if (err != cudaSuccess) return;

    cudaArray_t array = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource_[idx], 0, 0);
    if (err == cudaSuccess && array) {
        cudaMemcpy2DFromArray(
            stagingBuffer_[idx], pitch_,
            array, 0, 0,
            pitch_, height_,
            cudaMemcpyDeviceToDevice);
    }

    cudaGraphicsUnmapResources(1, &cudaResource_[idx], 0);

    // 捕获完成：记录本次写入的缓冲索引，翻转 writeIndex_ 供下次 capture() 使用
    // 翻转放在 GL 线程内（而非 getDevicePtr 的编码线程），消除跨线程数据竞争
    lastCapturedIdx_ = idx;
    writeIndex_ = 1 - idx;
}

void *FrameCapture::getDevicePtr() {
    // 返回最近一次 capture() 写入的暂存缓冲设备指针
    // 索引翻转已在 capture() 内完成，此函数仅做简单读取
    // 线程安全保证：调用方（encodeLoop）通过 encodeMutex_ 与 GL 线程的 onFrameReady()
    // 建立 happens-before 关系，确保看到 capture() 写入的最新 lastCapturedIdx_
    return stagingBuffer_[lastCapturedIdx_];
}
