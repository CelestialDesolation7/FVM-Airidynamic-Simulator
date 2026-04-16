#include "input_handler.h"
#include <imgui.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

// 极简 JSON 字段提取器（避免引入 JSON 库依赖）。
// 仅支持扁平对象（无嵌套），可提取字符串、数字、布尔三种值类型。
// 实现原理：直接字符串搜索键名，随后定位冒号后的值起始位置解析。
namespace {

std::string jsonStr(const std::string &json, const char *key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

double jsonNum(const std::string &json, const char *key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return 0.0;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return 0.0;
    while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ')) ++pos;
    return std::atof(json.c_str() + pos);
}

bool jsonBool(const std::string &json, const char *key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    return json.find("true", pos) == pos + 1 || json.find("true", pos) == pos + 2;
}

ImGuiKey glfwKeyToImGuiKey(int key) {
    if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z)
        return (ImGuiKey)(ImGuiKey_A + (key - GLFW_KEY_A));
    if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
        return (ImGuiKey)(ImGuiKey_0 + (key - GLFW_KEY_0));
    if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F12)
        return (ImGuiKey)(ImGuiKey_F1 + (key - GLFW_KEY_F1));
    switch (key) {
    case GLFW_KEY_TAB: return ImGuiKey_Tab;
    case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
    case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
    case GLFW_KEY_UP: return ImGuiKey_UpArrow;
    case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
    case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
    case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
    case GLFW_KEY_HOME: return ImGuiKey_Home;
    case GLFW_KEY_END: return ImGuiKey_End;
    case GLFW_KEY_INSERT: return ImGuiKey_Insert;
    case GLFW_KEY_DELETE: return ImGuiKey_Delete;
    case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
    case GLFW_KEY_SPACE: return ImGuiKey_Space;
    case GLFW_KEY_ENTER: return ImGuiKey_Enter;
    case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
    case GLFW_KEY_APOSTROPHE: return ImGuiKey_Apostrophe;
    case GLFW_KEY_COMMA: return ImGuiKey_Comma;
    case GLFW_KEY_MINUS: return ImGuiKey_Minus;
    case GLFW_KEY_PERIOD: return ImGuiKey_Period;
    case GLFW_KEY_SLASH: return ImGuiKey_Slash;
    case GLFW_KEY_SEMICOLON: return ImGuiKey_Semicolon;
    case GLFW_KEY_EQUAL: return ImGuiKey_Equal;
    case GLFW_KEY_LEFT_BRACKET: return ImGuiKey_LeftBracket;
    case GLFW_KEY_BACKSLASH: return ImGuiKey_Backslash;
    case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
    case GLFW_KEY_GRAVE_ACCENT: return ImGuiKey_GraveAccent;
    case GLFW_KEY_CAPS_LOCK: return ImGuiKey_CapsLock;
    case GLFW_KEY_NUM_LOCK: return ImGuiKey_NumLock;
    case GLFW_KEY_LEFT_SHIFT: return ImGuiKey_LeftShift;
    case GLFW_KEY_LEFT_CONTROL: return ImGuiKey_LeftCtrl;
    case GLFW_KEY_LEFT_ALT: return ImGuiKey_LeftAlt;
    case GLFW_KEY_LEFT_SUPER: return ImGuiKey_LeftSuper;
    case GLFW_KEY_RIGHT_SHIFT: return ImGuiKey_RightShift;
    case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
    case GLFW_KEY_RIGHT_ALT: return ImGuiKey_RightAlt;
    case GLFW_KEY_RIGHT_SUPER: return ImGuiKey_RightSuper;
    default: return ImGuiKey_None;
    }
}

} // namespace

int InputHandler::browserKeyToGlfw(const std::string &code) {
    static const std::unordered_map<std::string, int> map = {
        {"Space", GLFW_KEY_SPACE}, {"Enter", GLFW_KEY_ENTER}, {"Escape", GLFW_KEY_ESCAPE},
        {"Backspace", GLFW_KEY_BACKSPACE}, {"Tab", GLFW_KEY_TAB},
        {"ArrowUp", GLFW_KEY_UP}, {"ArrowDown", GLFW_KEY_DOWN},
        {"ArrowLeft", GLFW_KEY_LEFT}, {"ArrowRight", GLFW_KEY_RIGHT},
        {"ShiftLeft", GLFW_KEY_LEFT_SHIFT}, {"ShiftRight", GLFW_KEY_RIGHT_SHIFT},
        {"ControlLeft", GLFW_KEY_LEFT_CONTROL}, {"ControlRight", GLFW_KEY_RIGHT_CONTROL},
        {"AltLeft", GLFW_KEY_LEFT_ALT}, {"AltRight", GLFW_KEY_RIGHT_ALT},
        {"MetaLeft", GLFW_KEY_LEFT_SUPER}, {"MetaRight", GLFW_KEY_RIGHT_SUPER},
        {"Delete", GLFW_KEY_DELETE}, {"Insert", GLFW_KEY_INSERT},
        {"Home", GLFW_KEY_HOME}, {"End", GLFW_KEY_END},
        {"PageUp", GLFW_KEY_PAGE_UP}, {"PageDown", GLFW_KEY_PAGE_DOWN},
        {"F1", GLFW_KEY_F1}, {"F2", GLFW_KEY_F2}, {"F3", GLFW_KEY_F3},
        {"F4", GLFW_KEY_F4}, {"F5", GLFW_KEY_F5}, {"F6", GLFW_KEY_F6},
        {"F7", GLFW_KEY_F7}, {"F8", GLFW_KEY_F8}, {"F9", GLFW_KEY_F9},
        {"F10", GLFW_KEY_F10}, {"F11", GLFW_KEY_F11}, {"F12", GLFW_KEY_F12},
        {"Minus", GLFW_KEY_MINUS}, {"Equal", GLFW_KEY_EQUAL},
        {"BracketLeft", GLFW_KEY_LEFT_BRACKET}, {"BracketRight", GLFW_KEY_RIGHT_BRACKET},
        {"Backslash", GLFW_KEY_BACKSLASH}, {"Semicolon", GLFW_KEY_SEMICOLON},
        {"Quote", GLFW_KEY_APOSTROPHE}, {"Comma", GLFW_KEY_COMMA},
        {"Period", GLFW_KEY_PERIOD}, {"Slash", GLFW_KEY_SLASH},
        {"Backquote", GLFW_KEY_GRAVE_ACCENT}, {"CapsLock", GLFW_KEY_CAPS_LOCK},
        {"NumLock", GLFW_KEY_NUM_LOCK},
        // KeyA-KeyZ
        {"KeyA", GLFW_KEY_A}, {"KeyB", GLFW_KEY_B}, {"KeyC", GLFW_KEY_C},
        {"KeyD", GLFW_KEY_D}, {"KeyE", GLFW_KEY_E}, {"KeyF", GLFW_KEY_F},
        {"KeyG", GLFW_KEY_G}, {"KeyH", GLFW_KEY_H}, {"KeyI", GLFW_KEY_I},
        {"KeyJ", GLFW_KEY_J}, {"KeyK", GLFW_KEY_K}, {"KeyL", GLFW_KEY_L},
        {"KeyM", GLFW_KEY_M}, {"KeyN", GLFW_KEY_N}, {"KeyO", GLFW_KEY_O},
        {"KeyP", GLFW_KEY_P}, {"KeyQ", GLFW_KEY_Q}, {"KeyR", GLFW_KEY_R},
        {"KeyS", GLFW_KEY_S}, {"KeyT", GLFW_KEY_T}, {"KeyU", GLFW_KEY_U},
        {"KeyV", GLFW_KEY_V}, {"KeyW", GLFW_KEY_W}, {"KeyX", GLFW_KEY_X},
        {"KeyY", GLFW_KEY_Y}, {"KeyZ", GLFW_KEY_Z},
        // Digit0-Digit9
        {"Digit0", GLFW_KEY_0}, {"Digit1", GLFW_KEY_1}, {"Digit2", GLFW_KEY_2},
        {"Digit3", GLFW_KEY_3}, {"Digit4", GLFW_KEY_4}, {"Digit5", GLFW_KEY_5},
        {"Digit6", GLFW_KEY_6}, {"Digit7", GLFW_KEY_7}, {"Digit8", GLFW_KEY_8},
        {"Digit9", GLFW_KEY_9},
    };
    auto it = map.find(code);
    return (it != map.end()) ? it->second : GLFW_KEY_UNKNOWN;
}

int InputHandler::parseMods(bool shift, bool ctrl, bool alt, bool meta) {
    int m = 0;
    if (shift) m |= GLFW_MOD_SHIFT;
    if (ctrl) m |= GLFW_MOD_CONTROL;
    if (alt) m |= GLFW_MOD_ALT;
    if (meta) m |= GLFW_MOD_SUPER;
    return m;
}

void InputHandler::onMessage(const std::string &json) {
    // 此函数在 libdatachannel 的内部网络线程中被回调，与 GL 线程并发运行
    // 内部不执行任何 OpenGL 调用，只做解析和入队
    std::string type = jsonStr(json, "type");
    Event ev{};

    if (type == "mousemove") {
        ev.type = Event::MOUSE_MOVE;
        ev.x = jsonNum(json, "x");  // 已经在浏览器端归一化为 [0,1]
        ev.y = jsonNum(json, "y");
    } else if (type == "mousedown") {
        ev.type = Event::MOUSE_DOWN;
        ev.x = jsonNum(json, "x");
        ev.y = jsonNum(json, "y");
        ev.button = (int)jsonNum(json, "button");  // 浏览器按键编号: 0=左, 1=中, 2=右
    } else if (type == "mouseup") {
        ev.type = Event::MOUSE_UP;
        ev.x = jsonNum(json, "x");
        ev.y = jsonNum(json, "y");
        ev.button = (int)jsonNum(json, "button");
    } else if (type == "wheel") {
        ev.type = Event::SCROLL;
        ev.x = jsonNum(json, "dx");  // 水平滚动量（对应 WheelEvent.deltaX）
        ev.y = jsonNum(json, "dy");  // 垂直滚动量（对应 WheelEvent.deltaY）
    } else if (type == "keydown") {
        ev.type = Event::KEY_DOWN;
        // 浏览器发送 KeyCode 字符串（如 "KeyA", "ArrowLeft"），转换为 GLFW 键码
        ev.key  = browserKeyToGlfw(jsonStr(json, "code"));
        ev.mods = parseMods(jsonBool(json, "shift"), jsonBool(json, "ctrl"),
                            jsonBool(json, "alt"),   jsonBool(json, "meta"));
    } else if (type == "keyup") {
        ev.type = Event::KEY_UP;
        ev.key  = browserKeyToGlfw(jsonStr(json, "code"));
        ev.mods = parseMods(jsonBool(json, "shift"), jsonBool(json, "ctrl"),
                            jsonBool(json, "alt"),   jsonBool(json, "meta"));
    } else if (type == "resize") {
        ev.type = Event::RESIZE;
        ev.x = jsonNum(json, "width");   // 实物理像素宽度（浏览器已乘以 DPR）
        ev.y = jsonNum(json, "height");
    } else {
        return; // 未知事件类型，丢弃
    }

    std::lock_guard<std::mutex> lock(mutex_);  // 保护对队列的并发写入
    queue_.push(ev);
    if (ev.type == Event::RESIZE) {
        // 额外记录最新的待处理 resize 尺寸，支持 hasPendingResize() 快速查询
        pendingResizeW_ = (int)ev.x;
        pendingResizeH_ = (int)ev.y;
        resizePending_ = true;
    }
}

bool InputHandler::hasPendingResize(int &outW, int &outH) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!resizePending_) return false;
    outW = pendingResizeW_;
    outH = pendingResizeH_;
    resizePending_ = false;
    return true;
}

void InputHandler::processEvents(GLFWwindow *window, int winW, int winH,
                                 std::function<void(int, int)> onResize) {
    // 快照模式：一次性将队列中的所有事件转移到局部快照
    // 优对：持锁时间极短（只做 swap），不会长时间阻塞网络线程的 onMessage 入队
    std::queue<Event> snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::swap(snapshot, queue_);  // O(1)，无拷贝
    }
    // 此后无锁处理 snapshot，网络线程可并发入队新事件

    ImGuiIO &io = ImGui::GetIO();

    while (!snapshot.empty()) {
        Event ev = snapshot.front();
        snapshot.pop();

        switch (ev.type) {
        case Event::MOUSE_MOVE: {
            // 将归一化坐标 [0,1] 反归一化为像素坐标，注入 ImGui 鼠标位置
            double px = ev.x * winW;
            double py = ev.y * winH;
            io.AddMousePosEvent((float)px, (float)py);
            break;
        }
        case Event::MOUSE_DOWN: {
            double px = ev.x * winW;
            double py = ev.y * winH;
            io.AddMousePosEvent((float)px, (float)py);
            int btn = (ev.button == 2) ? 2 : (ev.button == 1) ? 2 : ev.button;
            // 浏览器鼠标键编号：0=左键, 1=中键, 2=右键；ImGui 鼠标键编号：0=左键, 1=右键, 2=中键
            if (ev.button == 0) btn = 0;
            else if (ev.button == 1) btn = 2;
            else if (ev.button == 2) btn = 1;
            io.AddMouseButtonEvent(btn, true);
            break;
        }
        case Event::MOUSE_UP: {
            int btn = ev.button;
            if (ev.button == 0) btn = 0;
            else if (ev.button == 1) btn = 2;
            else if (ev.button == 2) btn = 1;
            io.AddMouseButtonEvent(btn, false);
            break;
        }
        case Event::SCROLL:
            // 浏览器 WheelEvent.deltaY 向下为正，ImGui Y 轴向上为正，此处还除以 120（标准一格 = 120）
            io.AddMouseWheelEvent((float)ev.x, (float)(-ev.y / 120.0));
            break;
        case Event::KEY_DOWN:
            if (ev.key != GLFW_KEY_UNKNOWN) {
                // 路径一：注入 ImGui 键盘事件（UI 控件响应）
                io.AddKeyEvent(glfwKeyToImGuiKey(ev.key), true);
                // 路径二：手动触发 GLFW 键盘回调（应用逻辑响应，如 ESC 退出、Space 暂停）
                // 取出→装回→调用技巧：glfwSetKeyCallback 在设置新值时返回旧值。
                // 用 nullptr 作为临时新值取出旧回调，立即装回，再手动调用。
                // 这样既不需要全局存储回调指针，也不改变已注册的回调。
                auto keyCb = glfwSetKeyCallback(window, nullptr);
                glfwSetKeyCallback(window, keyCb);
                if (keyCb) keyCb(window, ev.key, 0, GLFW_PRESS, ev.mods);
            }
            break;
        case Event::KEY_UP:
            if (ev.key != GLFW_KEY_UNKNOWN) {
                io.AddKeyEvent(glfwKeyToImGuiKey(ev.key), false);
                auto keyCb = glfwSetKeyCallback(window, nullptr);
                glfwSetKeyCallback(window, keyCb);
                if (keyCb) keyCb(window, ev.key, 0, GLFW_RELEASE, ev.mods);
            }
            break;
        case Event::RESIZE:
            // 触发 StreamServer::requestResize()，将目标分辨率存入原子变量，由 applyPendingResize() 延迟执行
            if (onResize) onResize((int)ev.x, (int)ev.y);
            break;
        }
    }
}
