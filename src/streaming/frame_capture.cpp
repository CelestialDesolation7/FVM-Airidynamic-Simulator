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
    glGenTextures(1, &texture_[idx]);
    glBindTexture(GL_TEXTURE_2D, texture_[idx]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &fbo_[idx]);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_[idx]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_[idx], 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[FrameCapture] FBO incomplete: " << status << std::endl;
        return false;
    }

    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cudaResource_[idx], texture_[idx], GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsReadOnly));

    CUDA_CHECK(cudaMalloc(&stagingBuffer_[idx], (size_t)w * h * 4));

    return true;
}

void FrameCapture::destroyFBO(int idx) {
    if (cudaResource_[idx]) {
        cudaGraphicsUnregisterResource(cudaResource_[idx]);
        cudaResource_[idx] = nullptr;
    }
    if (stagingBuffer_[idx]) {
        cudaFree(stagingBuffer_[idx]);
        stagingBuffer_[idx] = nullptr;
    }
    if (fbo_[idx]) { glDeleteFramebuffers(1, &fbo_[idx]); fbo_[idx] = 0; }
    if (texture_[idx]) { glDeleteTextures(1, &texture_[idx]); texture_[idx] = 0; }
}

bool FrameCapture::initialize(int width, int height) {
    width_ = width;
    height_ = height;
    pitch_ = width * 4;

    for (int i = 0; i < 2; ++i) {
        if (!createFBO(i, width, height)) return false;
    }
    writeIndex_ = 0;
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

void FrameCapture::capture() {
    // 第一步：将默认帧缓冲（前台渲染结果）Blit 到捕获专用 FBO。
    // 目标坐标 (0, height_, width_, 0) 在 Y 轴方向翻转，补偿 OpenGL 纹理坐标从左下角起点的特性，
    // 使最终传到浏览器的画面方向与屏幕显示一致（上下不颠倒）。
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_[writeIndex_]);
    glBlitFramebuffer(0, 0, width_, height_,
                      0, height_, width_, 0,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    // 第二步：映射 CUDA 资源，将 FBO 纹理数据拷贝到线性暂存缓冲，再解除映射。
    // 全部操作必须在持有 GL 上下文的 GL 线程上执行（CUDA-GL 互操作约束）。
    // 拷贝完成后编码线程即可安全读取暂存缓冲，无需等待 GL 上下文。
    cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_[writeIndex_], 0);
    if (err != cudaSuccess) return;

    cudaArray_t array = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource_[writeIndex_], 0, 0);
    if (err == cudaSuccess && array) {
        cudaMemcpy2DFromArray(
            stagingBuffer_[writeIndex_], pitch_,
            array, 0, 0,
            pitch_, height_,
            cudaMemcpyDeviceToDevice);
    }

    cudaGraphicsUnmapResources(1, &cudaResource_[writeIndex_], 0);
}

void *FrameCapture::getDevicePtr() {
    int readIdx = writeIndex_;
    writeIndex_ = 1 - writeIndex_;
    return stagingBuffer_[readIdx];
}
