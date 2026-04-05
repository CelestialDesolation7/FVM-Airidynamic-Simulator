#pragma once
#include <vector>
#include <string>
#include <cstdint>

// =========================================================================
// Mongoose 封装层 - 负责提供 HTTP(网站页面) 和 WebSocket(像素流双向通信)
// 这就是典型的“C/S架构云游戏”的后端封装。它不依赖外部庞大的网络库。
// =========================================================================

// 初始化 Web 服务器并在后台独立线程中运行
// 参数 port 是监听的端口号，比如 18080
void WebServer_Init(int port);

// 提取一次底层非阻塞的事件轮询
// 这是因为 Mongoose 是基于 poll/select 等机制的异步 IO 库，必须主动叫它。
// 这里我们把轮询任务和主渲染线程绑定在一起，这样就不会有多线程锁的乱局。
void WebServer_PollEvents();

// 广播帧画面（把截取下来的 JPEG 图片喂给全部网页连接者）
void WebServer_BroadcastFrame(const std::vector<uint8_t>& jpeg_data);

// 检查当前是不是空跑（没有客户端时没必要截频压缩性能浪费）
bool WebServer_HasClients();

// 获取前端网页扔过来的鼠标或键盘事件（JSON 文本，需要被解析后塞给 ImGui 引擎）
// 每次只能弹出一个，返回空字符串代表目前事件队列被清空了
std::string WebServer_PopEvent();

// 优雅停止并释放占用端口及内存
void WebServer_Stop();

// =========================================================================
// 核心：一键截取当前 OpenGL 可视化帧缓冲，并将其强行编码成 JPG 图片
// 这样每次只传几十 KB 的超清画面，而不需要传几 MB 的数组让前端再去用 WebGL 画。
// =========================================================================
std::vector<uint8_t> WebServer_CaptureFrame(int width, int height);