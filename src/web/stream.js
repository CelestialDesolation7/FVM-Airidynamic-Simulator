"use strict";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");

let pc = null;
let dc = null;
let reconnectTimer = null;
let statsTimer = null;
let lastResizeW = 0;
let lastResizeH = 0;

const SIGNALING_URL = `${location.origin}/api/offer`;
const RECONNECT_DELAY_MS = 3000;

async function connect() {
    showStatus("正在连接...");
    cleanup();

    pc = new RTCPeerConnection({ iceServers: [] });

    pc.ontrack = (ev) => {
        console.log("[WebRTC] ontrack fired, streams:", ev.streams.length);
        if (ev.streams && ev.streams[0]) {
            video.srcObject = ev.streams[0];
        } else {
            const ms = new MediaStream([ev.track]);
            video.srcObject = ms;
        }
        video.play().catch(e => console.warn("video.play() error:", e));
        hideStatus();
    };

    pc.oniceconnectionstatechange = () => {
        const s = pc.iceConnectionState;
        console.log("[WebRTC] ICE state:", s);
        if (s === "disconnected" || s === "failed" || s === "closed") {
            showStatus("连接断开，正在重连...");
            scheduleReconnect();
        }
    };

    pc.onconnectionstatechange = () => {
        console.log("[WebRTC] Connection state:", pc.connectionState);
    };

    dc = pc.createDataChannel("input", { ordered: true });
    dc.onopen = () => {
        console.log("[DC] open");
        sendResize();
    };
    dc.onclose = () => console.log("[DC] closed");

    pc.addTransceiver("video", { direction: "recvonly" });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitGathering(pc, 3000);

        const controller = new AbortController();
        const fetchTimeout = setTimeout(() => controller.abort(), 5000);
        const resp = await fetch(SIGNALING_URL, {
            method: "POST",
            headers: { "Content-Type": "application/sdp" },
            body: pc.localDescription.sdp,
            signal: controller.signal,
        });
        clearTimeout(fetchTimeout);

        if (!resp.ok) throw new Error(`Signaling HTTP ${resp.status}`);

        const answerSdp = await resp.text();
        console.log("[WebRTC] Answer SDP length:", answerSdp.length);
        await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

        startStats();
    } catch (e) {
        console.error("[connect]", e);
        showStatus("连接失败，正在重试...");
        scheduleReconnect();
    }
}

function waitGathering(pc, timeoutMs) {
    return new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") return resolve();
        const t = setTimeout(resolve, timeoutMs);
        pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") { clearTimeout(t); resolve(); }
        };
    });
}

function cleanup() {
    if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
    if (pc) { pc.close(); pc = null; }
    dc = null;
    video.srcObject = null;
    lastResizeW = 0;
    lastResizeH = 0;
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, RECONNECT_DELAY_MS);
}

function showStatus(msg) { statusEl.textContent = msg; statusEl.classList.remove("hidden"); }
function hideStatus() { statusEl.classList.add("hidden"); }

function startStats() {
    if (statsTimer) clearInterval(statsTimer);
    let lastBytes = 0, lastTime = performance.now();

    statsTimer = setInterval(async () => {
        if (!pc) return;
        try {
            const stats = await pc.getStats();
            let fps = 0, bytesNow = 0, jitter = 0;
            let framesReceived = 0, framesDecoded = 0, framesDropped = 0;
            let keyFramesDecoded = 0, packetsReceived = 0, packetsLost = 0;
            let nackCount = 0, pliCount = 0, decoderImpl = "", codecId = "";
            stats.forEach((s) => {
                if (s.type === "inbound-rtp" && s.kind === "video") {
                    fps = s.framesPerSecond || 0;
                    bytesNow = s.bytesReceived || 0;
                    jitter = (s.jitter || 0) * 1000;
                    framesReceived = s.framesReceived || 0;
                    framesDecoded = s.framesDecoded || 0;
                    framesDropped = s.framesDropped || 0;
                    keyFramesDecoded = s.keyFramesDecoded || 0;
                    packetsReceived = s.packetsReceived || 0;
                    packetsLost = s.packetsLost || 0;
                    nackCount = s.nackCount || 0;
                    pliCount = s.pliCount || 0;
                    decoderImpl = s.decoderImplementation || "";
                    codecId = s.codecId || "";
                }
            });
            const width = video.videoWidth || 0;
            const height = video.videoHeight || 0;
            const now = performance.now();
            const dt = (now - lastTime) / 1000;
            const bitrate = dt > 0 ? ((bytesNow - lastBytes) * 8 / dt / 1e6).toFixed(1) : "?";
            lastBytes = bytesNow; lastTime = now;

            overlay.textContent =
                `${width}x${height}  ${fps} fps  ${bitrate} Mbps  jitter ${jitter.toFixed(1)}ms  dec:${decoderImpl || "none"}`;

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

function send(obj) {
    if (dc && dc.readyState === "open") dc.send(JSON.stringify(obj));
}

function videoRect() {
    return video.getBoundingClientRect();
}

function normalizePos(e) {
    const r = videoRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}

function modifiers(e) {
    return { shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey, meta: e.metaKey };
}

video.addEventListener("mousemove", (e) => {
    send({ type: "mousemove", ...normalizePos(e) });
});

video.addEventListener("mousedown", (e) => {
    e.preventDefault();
    send({ type: "mousedown", ...normalizePos(e), button: e.button });
});

video.addEventListener("mouseup", (e) => {
    e.preventDefault();
    send({ type: "mouseup", ...normalizePos(e), button: e.button });
});

video.addEventListener("wheel", (e) => {
    e.preventDefault();
    send({ type: "wheel", dx: e.deltaX, dy: e.deltaY });
}, { passive: false });

video.addEventListener("contextmenu", (e) => e.preventDefault());

document.addEventListener("keydown", (e) => {
    if (e.code === "F11" || e.code === "F12") return;
    e.preventDefault();
    send({ type: "keydown", code: e.code, ...modifiers(e) });
});

document.addEventListener("keyup", (e) => {
    e.preventDefault();
    send({ type: "keyup", code: e.code, ...modifiers(e) });
});

function sendResize() {
    const dpr = window.devicePixelRatio || 1;
    const vp = window.visualViewport;
    const cssW = vp ? vp.width : window.innerWidth;
    const cssH = vp ? vp.height : window.innerHeight;

    const w = Math.max(64, (Math.round(cssW * dpr)) & ~1);
    const h = Math.max(64, (Math.round(cssH * dpr)) & ~1);
    if (w === lastResizeW && h === lastResizeH) return;

    lastResizeW = w;
    lastResizeH = h;
    send({ type: "resize", width: w, height: h });
}

window.addEventListener("resize", () => {
    clearTimeout(window._resizeDebounce);
    window._resizeDebounce = setTimeout(sendResize, 300);
});

connect();
