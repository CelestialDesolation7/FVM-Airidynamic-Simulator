"use strict";

// ======================================================================
// 全局状态变量
// ======================================================================
const video = document.getElementById("video");    // 接收视频流的 <video> 元素
const overlay = document.getElementById("overlay"); // 右上角 WebRTC 统计信息覆盖层
const statusEl = document.getElementById("status"); // 中央状态提示文字（连接中/断开）

let pc = null;           // RTCPeerConnection 实例（当前活跃连接，null 表示未连接）
let dc = null;           // RTCDataChannel 实例（用于向服务器回传输入事件）
let reconnectTimer = null; // 重连定时器句柄（防止重复调度多个定时器）
let statsTimer = null;     // WebRTC 统计信息轮询定时器句柄
let lastResizeW = 0;       // 上一次发送的宽度（去重用：避免重复发送相同尺寸）
let lastResizeH = 0;       // 上一次发送的高度

const SIGNALING_URL = `${location.origin}/api/offer`; // 信令服务器地址
const RECONNECT_DELAY_MS = 3000;  // 断开后重连延迟（毫秒）

// ======================================================================
// 建立 WebRTC 连接
// ======================================================================
async function connect() {
    showStatus("正在连接...");
    cleanup();  // 清理旧连接（若有），重置 lastResizeW/H 使后续 sendResize 一定执行

    // ① 创建新的 RTCPeerConnection
    // iceServers: [] 表示不使用 STUN/TURN，仅局域网直连（host candidates）
    pc = new RTCPeerConnection({ iceServers: [] });

    // ② 注册视频轨道回调：服务器发来 RTP 视频流时触发
    pc.ontrack = (ev) => {
        console.log("[WebRTC] ontrack fired, streams:", ev.streams.length);
        if (ev.streams && ev.streams[0]) {
            video.srcObject = ev.streams[0];  // 将 MediaStream 绑定到 <video> 元素
        } else {
            // 极少数浏览器不附带 streams，手动构造 MediaStream
            const ms = new MediaStream([ev.track]);
            video.srcObject = ms;
        }
        video.play().catch(e => console.warn("video.play() error:", e));
        hideStatus();
    };

    // ③ 注册 ICE 连接状态回调
    // ICE 状态机：new → checking → connected/completed → disconnected/failed/closed
    pc.oniceconnectionstatechange = () => {
        const s = pc.iceConnectionState;
        console.log("[WebRTC] ICE state:", s);
        if (s === "disconnected" || s === "failed" || s === "closed") {
            showStatus("连接断开，正在重连...");
            scheduleReconnect();  // 延迟 3 秒后重新调用 connect()
        }
    };

    pc.onconnectionstatechange = () => {
        console.log("[WebRTC] Connection state:", pc.connectionState);
    };

    // ④ 创建 DataChannel（有序传输，用于向服务器回传鼠标/键盘/尺寸事件）
    // ordered: true 保证事件按发送顺序到达（避免鼠标坐标乱序）
    dc = pc.createDataChannel("input", { ordered: true });
    dc.onopen  = () => {
        console.log("[DC] open");
        sendResize();  // DataChannel 就绪后立即同步当前窗口尺寸给服务器
    };
    dc.onclose = () => console.log("[DC] closed");

    // ⑤ 声明接收视频（recvonly：浏览器只接收，不发送视频）
    // 必须在 createOffer() 之前添加 transceiver，确保 Offer SDP 中含有 video 媒体描述
    pc.addTransceiver("video", { direction: "recvonly" });

    try {
        // ⑥ 生成 Offer SDP（浏览器宣告自己支持的编解码器、ICE 参数等）
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // ⑦ 等待 ICE 候选收集完成（最多 3 秒）
        // 确保 localDescription.sdp 中包含本机 IP/端口（host candidates）
        await waitGathering(pc, 3000);

        // ⑧ 将 Offer SDP POST 到服务器，等待 Answer SDP（5 秒 AbortController 超时）
        // AbortController 防止服务器未启动时无限挂起
        const controller = new AbortController();
        const fetchTimeout = setTimeout(() => controller.abort(), 5000);
        const resp = await fetch(SIGNALING_URL, {
            method: "POST",
            headers: { "Content-Type": "application/sdp" },
            body: pc.localDescription.sdp,  // Offer SDP 文本
            signal: controller.signal,
        });
        clearTimeout(fetchTimeout);

        if (!resp.ok) throw new Error(`Signaling HTTP ${resp.status}`);

        // ⑨ 将服务器返回的 Answer SDP 设为远端描述，触发 ICE 协商
        const answerSdp = await resp.text();
        console.log("[WebRTC] Answer SDP length:", answerSdp.length);
        await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
        // 至此：ICE 候选开始连通性检测 → DTLS 握手 → ontrack 触发 → 视频开始播放

        startStats();  // 启动每秒一次的 WebRTC 统计信息采集
    } catch (e) {
        console.error("[connect]", e);
        showStatus("连接失败，正在重试...");
        scheduleReconnect();
    }
}

// 等待 ICE 候选收集完成的 Promise 包装器
// 收集完成（iceGatheringState === "complete"）或超时后 resolve
function waitGathering(pc, timeoutMs) {
    return new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") return resolve();
        const t = setTimeout(resolve, timeoutMs);  // 超时保底
        pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") { clearTimeout(t); resolve(); }
        };
    });
}

// 清理当前连接的所有状态，为下一次 connect() 做准备
function cleanup() {
    if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
    if (pc) { pc.close(); pc = null; }
    dc = null;
    video.srcObject = null;
    // 关键：重置上一次发送的尺寸为 0，确保重连后 sendResize() 一定发送新尺寸
    // 若不重置，当尺寸未变化时 sendResize() 的去重逻辑会跳过发送，
    // 导致服务器维持旧分辨率（与新连接不一致时会产生黑边）
    lastResizeW = 0;
    lastResizeH = 0;
}

// 调度重连（防止多次调用重复创建定时器）
function scheduleReconnect() {
    if (reconnectTimer) return;  // 已有定时器在等待，不重复调度
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, RECONNECT_DELAY_MS);
}

function showStatus(msg) { statusEl.textContent = msg; statusEl.classList.remove("hidden"); }
function hideStatus() { statusEl.classList.add("hidden"); }

// ======================================================================
// WebRTC 统计信息采集（每秒一次）
// ======================================================================
function startStats() {
    if (statsTimer) clearInterval(statsTimer);
    let lastBytes = 0, lastTime = performance.now();

    statsTimer = setInterval(async () => {
        if (!pc) return;
        try {
            // pc.getStats() 返回所有统计条目（RTCStatsReport）
            const stats = await pc.getStats();
            let fps = 0, bytesNow = 0, jitter = 0;
            let framesReceived = 0, framesDecoded = 0, framesDropped = 0;
            let keyFramesDecoded = 0, packetsReceived = 0, packetsLost = 0;
            let nackCount = 0, pliCount = 0, decoderImpl = "", codecId = "";
            stats.forEach((s) => {
                // 只处理视频的入站 RTP 统计条目
                if (s.type === "inbound-rtp" && s.kind === "video") {
                    fps             = s.framesPerSecond || 0;
                    bytesNow        = s.bytesReceived   || 0;
                    jitter          = (s.jitter || 0) * 1000;  // 转换为毫秒
                    framesReceived  = s.framesReceived  || 0;
                    framesDecoded   = s.framesDecoded   || 0;
                    framesDropped   = s.framesDropped   || 0;
                    keyFramesDecoded = s.keyFramesDecoded || 0;
                    packetsReceived = s.packetsReceived  || 0;
                    packetsLost     = s.packetsLost      || 0;
                    nackCount       = s.nackCount        || 0;  // 浏览器请求重传的次数
                    pliCount        = s.pliCount         || 0;  // 浏览器请求关键帧的次数
                    decoderImpl     = s.decoderImplementation || ""; // 解码器实现（如 "ExternalDecoder"）
                    codecId         = s.codecId || "";
                }
            });
            const width  = video.videoWidth  || 0;
            const height = video.videoHeight || 0;
            // 计算实时码率（当前周期内接收字节数 / 时间差）
            const now    = performance.now();
            const dt     = (now - lastTime) / 1000;
            const bitrate = dt > 0 ? ((bytesNow - lastBytes) * 8 / dt / 1e6).toFixed(1) : "?";
            lastBytes = bytesNow; lastTime = now;

            // 更新覆盖层显示
            overlay.textContent =
                `${width}x${height}  ${fps} fps  ${bitrate} Mbps  jitter ${jitter.toFixed(1)}ms  dec:${decoderImpl || "none"}`;

            // 将统计数据上报给服务器（供 /api/browser-stats GET 接口查询）
            fetch(`${location.origin}/api/browser-stats`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    framesReceived, framesDecoded, framesDropped, keyFramesDecoded,
                    frameWidth: width, frameHeight: height,
                    fps, bytesReceived: bytesNow, bitrateMbps: parseFloat(bitrate) || 0,
                    jitterMs: jitter, connectionState: pc.connectionState,
                    iceState: pc.iceConnectionState,
                    packetsReceived, packetsLost, nackCount, pliCount,
                    decoderImpl, codecId
                })
            }).catch(() => {});
        } catch (_) {}
    }, 1000);
}

// ======================================================================
// DataChannel 消息发送辅助函数
// ======================================================================

// 发送 JSON 对象到服务器（通过 DataChannel）
// 只有 DataChannel 处于 open 状态才发送，避免因连接未就绪导致的异常
function send(obj) {
    if (dc && dc.readyState === "open") dc.send(JSON.stringify(obj));
}

// 获取视频元素的屏幕矩形（用于计算归一化坐标）
function videoRect() {
    return video.getBoundingClientRect();
}

// 将鼠标事件的屏幕坐标归一化为 [0,1] 相对坐标
// 服务器端将其乘以窗口宽高还原为像素坐标，与分辨率无关
function normalizePos(e) {
    const r = videoRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}

// 提取键盘修饰键状态
function modifiers(e) {
    return { shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey, meta: e.metaKey };
}

// ======================================================================
// 输入事件监听（鼠标/键盘/滚轮）
// ======================================================================

// 鼠标移动：发送归一化坐标，用于服务器端模拟 GLFW 鼠标位置
video.addEventListener("mousemove", (e) => {
    send({ type: "mousemove", ...normalizePos(e) });
});

// 鼠标按下：发送归一化坐标 + 按键编号（0=左, 1=中, 2=右）
video.addEventListener("mousedown", (e) => {
    e.preventDefault();
    send({ type: "mousedown", ...normalizePos(e), button: e.button });
});

video.addEventListener("mouseup", (e) => {
    e.preventDefault();
    send({ type: "mouseup", ...normalizePos(e), button: e.button });
});

// 滚轮事件：发送 deltaX/deltaY（单位：像素，通常一格 = 120）
// passive: false 允许调用 preventDefault() 阻止页面滚动
video.addEventListener("wheel", (e) => {
    e.preventDefault();
    send({ type: "wheel", dx: e.deltaX, dy: e.deltaY });
}, { passive: false });

video.addEventListener("contextmenu", (e) => e.preventDefault());  // 屏蔽右键菜单

// 键盘事件绑定在 document 上（而非 video），确保即使 video 未获焦点也能捕获
// F11/F12 保留给浏览器自身使用（全屏/开发者工具）
document.addEventListener("keydown", (e) => {
    if (e.code === "F11" || e.code === "F12") return;
    e.preventDefault();  // 阻止浏览器默认行为（如空格滚动页面）
    // code: 物理键位标识符（如 "KeyA", "ArrowLeft"），不受输入法影响
    send({ type: "keydown", code: e.code, ...modifiers(e) });
});

document.addEventListener("keyup", (e) => {
    e.preventDefault();
    send({ type: "keyup", code: e.code, ...modifiers(e) });
});

// ======================================================================
// 分辨率同步（浏览器窗口尺寸 → 服务器捕获/编码分辨率）
// ======================================================================

// 发送当前浏览器窗口的物理像素尺寸给服务器
// 目的：使服务器以与浏览器显示区域完全匹配的分辨率捕获和编码，避免缩放导致的模糊
function sendResize() {
    const dpr = window.devicePixelRatio || 1;  // HiDPI 缩放因子（Retina 屏为 2.0）
    const vp   = window.visualViewport;
    const cssW = vp ? vp.width  : window.innerWidth;   // CSS 逻辑像素宽度
    const cssH = vp ? vp.height : window.innerHeight;   // CSS 逻辑像素高度

    // 转换为物理像素，并对齐到偶数（H.264 YUV 4:2:0 要求宽高均为偶数）
    const w = Math.max(64, (Math.round(cssW * dpr)) & ~1);
    const h = Math.max(64, (Math.round(cssH * dpr)) & ~1);

    // 去重：尺寸未变则跳过（避免不必要的服务器端 resize 重建 FBO/编码器）
    if (w === lastResizeW && h === lastResizeH) return;
    lastResizeW = w;
    lastResizeH = h;
    send({ type: "resize", width: w, height: h });
}

// 窗口尺寸变化时触发 resize 同步（防抖 300ms，避免拖动窗口时频繁触发）
window.addEventListener("resize", () => {
    clearTimeout(window._resizeDebounce);
    window._resizeDebounce = setTimeout(sendResize, 300);
});

// 页面加载完成后立即尝试建立连接
connect();
