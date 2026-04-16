#pragma once

#include <GLFW/glfw3.h>
#include <string>
#include <mutex>
#include <queue>
#include <functional>

// 浏览器输入事件翻译器：将经 WebRTC DataChannel 传入的 JSON 格式输入消息
// 解析为 GLFW 兼容事件，并入队等待主线程（GL 线程）消费。
// 实现浏览器鼠标/键盘/滚轮/窗口尺寸变化 → 本地 GLFW 仿真事件的完整映射。
class InputHandler {
public:
    // 已解析的远端输入事件结构体
    struct Event {
        enum Type { MOUSE_MOVE, MOUSE_DOWN, MOUSE_UP, SCROLL, KEY_DOWN, KEY_UP, RESIZE };
        Type type;
        double x = 0, y = 0;    // 鼠标事件：归一化坐标 [0,1]；RESIZE：像素尺寸；SCROLL：滚动增量
        int button = 0;         // 鼠标按键编号（GLFW 常量，如 GLFW_MOUSE_BUTTON_LEFT）
        int key = 0;            // GLFW 键码（由 browserKeyToGlfw 从 KeyCode 字符串转换）
        int mods = 0;           // GLFW 修饰键位掘码（Shift/Ctrl/Alt/Super）
    };

    // 解析来自 DataChannel 的 JSON 消息，并将转换后的事件加入队列。
    // 可在任意线程调用（内部持锁），通常由 WebRTC 回调线程调用。
    void onMessage(const std::string &json);

    // 在主线程（GL 线程）上消费队列中的全部待处理事件。
    // window：GLFW 窗口句柄；winW/winH：当前窗口像素尺寸（用于坐标反归一化）。
    // onResize：RESIZE 事件到来时的回调，用于通知上层更新捕获器/编础器分辨率。
    void processEvents(GLFWwindow *window, int winW, int winH,
                       std::function<void(int, int)> onResize = nullptr);

    bool hasPendingResize(int &outW, int &outH);

private:
    std::mutex mutex_;
    std::queue<Event> queue_;
    int pendingResizeW_ = 0, pendingResizeH_ = 0;
    bool resizePending_ = false;

    static int browserKeyToGlfw(const std::string &code);
    static int parseMods(bool shift, bool ctrl, bool alt, bool meta);
};
