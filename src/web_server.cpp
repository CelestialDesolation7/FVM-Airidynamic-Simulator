#include "web_server.h"

// STB_IMAGE_WRITE_IMPLEMENTATION 代表把这个头文件当成源文件去编译一次所有函数
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <mongoose.h>    // 我们用 Mongoose，因为它只要一个单文件就能当一个服务器，不会对你现有的 CMake 设置产生破坏
#include <glad/glad.h>   // 需要用 OpenGL 的 glReadPixels 从显卡上扒图片数据下来
#include <vector>
#include <queue>
#include <iostream>

// --- 以下全是只存在于本文件内部的静态变量，防止污染项目的全局作用域 ---
static struct mg_mgr s_mgr;                   // Mongoose 的全局上下文管家（所有网络连接和事件循环的持有者）
static std::queue<std::string> s_eventQueue;   // 负责存放网页传来的字符串（鼠标点击、拖拽的 JSON），像一个快递中转站
static int s_wsClientCount = 0;               // 记一下有多少个游览器连着，没人连就别截图压图，不然你的 CPU/GPU 会哭

// ----------------------------------------------------------------------------------------------------------
// 这是最核心的回调函数！不管是你访问网页还是推流，全部通过这个函数触发。
// 换句话说它就是服务端的大门，谁来敲门都会被它接待。
// ----------------------------------------------------------------------------------------------------------
static void ServerEventHandler(struct mg_connection *c, int ev, void *ev_data) {
// 如果有人发了一个基础的 HTTP 网页请求（比如你在浏览器输入地址按下回车）
    if (ev == MG_EV_HTTP_MSG) {
        // hm 里装满了关于他想访问什么的一堆细致参数，我们要看看里面说啥
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;
        
        // 1. 如果他要访问的是 "/ws" 这个网址，说明他想把普通的网页 HTTP 连接【升级】成可以随时双向推送二进制视频流的 [WebSocket] 连接
        if (mg_match(hm->uri, mg_str("/ws"), NULL)) {
            mg_ws_upgrade(c, hm, NULL);  // 当老手，允许它变为 WebSocket
        
        // 2. 如果他访问的就是什么都没加的纯根目录 "/" ，我们就在这里发一个内置好了的一大段 HTML5 代码
        } else if (mg_match(hm->uri, mg_str("/"), NULL)) {
            // 这个变量里的文字叫做 "字面量字符串"(C++11 新特性 R"()") 把整块 HTML 直接写进 C++。
            // 这是极其简单的实现，好处是程序生成的可执行文件可以直接拷贝到别的系统跑，不需要附带一堆资源文件。
            const char* html = R"html(
<!DOCTYPE html>
<html>
<title>云游戏 - FVM 模拟器像素流架构</title>
<!-- 让画布填充整个背景，就像游戏窗口一样满屏展现 -->
<style>
    body { background: #222; margin: 0; display: flex; align-items: center; justify-content: center; height: 100vh; overflow: hidden; }
    canvas { box-shadow: 0 0 20px black; cursor: crosshair; }
</style>
<body oncontextmenu="return false;">  <!-- 屏蔽网页的右键菜单，防止点菜单时蹦出网页提示烂风景 -->
    <canvas id="view"></canvas>
    
    <!-- 这里就是接收 C++ 服务端丢过来的图片流，在浏览器实时重绘播放的核心大脑 JS 脚本 -->
    <script>
        const canvas = document.getElementById('view');
        // 取出一张只负责 2D 绘图的操作句柄画笔（ctx）
        const ctx = canvas.getContext('2d');
        
        // 我们刚刚在这儿写了 /ws, 一旦这个脚本被网页执行，它会自动连接那个端口。
        const ws = new WebSocket('ws://' + location.host + '/ws');
        ws.binaryType = 'blob';  // 告诉浏览器：我要收的是一大泡乱码（Blob）而不是普通人能看懂的字串

        // 每次一旦 C++ 服务器（WebServer_BroadcastFrame）发推流，这里就立刻跑一遍。
        // （这会把 CPU 和显卡跑得很快，这就算你在玩云游戏的过程）
        ws.onmessage = async (e) => {
            if (e.data instanceof Blob) {  
                const img = new Image();  
                img.onload = () => {    // 因为图片就算到了浏览器也需要一点零点几毫秒解压加载
                    // 照片加载好的瞬间，如果是第一帧画面，调整画板的长宽。这里也就是为什么能在 0 代码缩放下自适应你的 Windows 窗口拉伸。
                    if (canvas.width !== img.width || canvas.height !== img.height) {
                        canvas.width = img.width;
                        canvas.height = img.height;
                    }
                    // 一键全部贴到面板上。这就是一帧的内容了。如果有 60FPS 它就能看成连续的视频。
                    ctx.drawImage(img, 0, 0); 
                    // 处理完就要清理垃圾回收掉，防止这个大块乱码图越堆越多撑爆电脑（尤其是内存很宝贵的手机设备）
                    URL.revokeObjectURL(img.src);
                };
                img.src = URL.createObjectURL(e.data); // 创建一段指向内存里视频流最新 JPEG 的乱码 URI 链接，塞给这个无形的假 img。
            }
        };

        // ----------------------------------------------------
        // 这些其实在所有的 JS 和 HTML 里都很基础。就是鼠标点击坐标收集器。
        // 这些监听事件的意思是让浏览器发现你动鼠标了或者点下去了就把动作包装成 {type: 'mousedown', x: 0.11, y: 0.22, button: 0} 给 C++ 程序
        // 这里特别使用除以宽/高（比例坐标），也是最精髓的部分！避免了屏幕分辨率不同的映射头疼。
        // ----------------------------------------------------
        function sendEvent(type, e) {
            if (ws.readyState === WebSocket.OPEN) {
                // 这个 getBoundingClientRect 仅仅是帮你拿掉空白边距边框（如果有的话）找到真实的图像本体范围。
                const rect = canvas.getBoundingClientRect();
                // We send coordinates proportional to canvas width/height
                // So it works if scaled.
                const px = (e.clientX - rect.left) / rect.width;
                const py = (e.clientY - rect.top) / rect.height;
                // 这行 JSON 打包发送就是逆向从你发往服务端的东西。
                ws.send(JSON.stringify({ type: type, x: px, y: py, button: e.button }));
            }
        }
        canvas.addEventListener('mousedown', e => sendEvent('mousedown', e)); // 0:左键 1:滚轮中键 2:右键
        canvas.addEventListener('mouseup', e => sendEvent('mouseup', e));
        canvas.addEventListener('mousemove', e => sendEvent('mousemove', e));
    </script>
</body>
</html>
)html";
            mg_http_reply(c, 200, "Content-Type: text/html\r\n", "%s", html);
        } else {
            mg_http_reply(c, 404, "", "Not found\n");
        }
    } 
    else if (ev == MG_EV_WS_OPEN) {
        c->data[0] = 'W'; // Marker
        s_wsClientCount++;
        std::cout << "[Web] 客户端已连接. 总连接数: " << s_wsClientCount << std::endl;
    }
    else if (ev == MG_EV_WS_MSG) {
        struct mg_ws_message *wm = (struct mg_ws_message *)ev_data;
        std::string msg(wm->data.ptr, wm->data.len);
        s_eventQueue.push(msg);
    }
    else if (ev == MG_EV_CLOSE) {
        if (c->data[0] == 'W') {
            s_wsClientCount--;
            std::cout << "[Web] 客户端已断开. 总连接数: " << s_wsClientCount << std::endl;
        }
    }
}

void WebServer_Init(int port) {
    mg_mgr_init(&s_mgr);
    std::string url = "http://0.0.0.0:" + std::to_string(port);
    std::cout << "[Web] 服务器监听开始: " << url << std::endl;
    mg_http_listen(&s_mgr, url.c_str(), ServerEventHandler, &s_mgr);
}

void WebServer_PollEvents() {
    mg_mgr_poll(&s_mgr, 0); // Non-blocking poll
}

void WebServer_Stop() {
    mg_mgr_free(&s_mgr);
}

bool WebServer_HasClients() {
    return s_wsClientCount > 0;
}

void WebServer_BroadcastFrame(const std::vector<uint8_t>& jpeg_data) {
    for (struct mg_connection* c = s_mgr.conns; c != NULL; c = c->next) {
        if (c->data[0] == 'W') {
            mg_ws_send(c, (const void*)jpeg_data.data(), jpeg_data.size(), WEBSOCKET_OP_BINARY);
        }
    }
}

std::string WebServer_PopEvent() {
    if (!s_eventQueue.empty()) {
        std::string evt = s_eventQueue.front();
        s_eventQueue.pop();
        return evt;
    }
    return "";
}

// STB 写入回调
static void StbWriteVectorCallback(void *context, void *data, int size) {
    std::vector<uint8_t>* vec = static_cast<std::vector<uint8_t>*>(context);
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), ptr, ptr + size);
}

std::vector<uint8_t> WebServer_CaptureFrame(int width, int height) {
    std::vector<uint8_t> pixels(width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    std::vector<uint8_t> flipped_pixels(width * height * 4);
    int row_bytes = width * 4;
    for (int y = 0; y < height; ++y) {
        memcpy(&flipped_pixels[y * row_bytes],
               &pixels[(height - 1 - y) * row_bytes],
               row_bytes);
    }

    std::vector<uint8_t> jpeg_buffer;
    stbi_write_jpg_to_func(StbWriteVectorCallback, &jpeg_buffer, width, height, 4, flipped_pixels.data(), 80);

    return jpeg_buffer;
}
