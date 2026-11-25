
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

model_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(model_path)

if detector.empty():
    raise RuntimeError("Yuz tespit modeli yuklenemedi!")

HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>School Face Detection</title>
<style>
body {
    background: #101827;
    color: white;
    text-align: center;
    font-family: Arial, sans-serif;
    padding: 20px;
}
button {
    padding: 14px 24px;
    background: #22c55e;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
    margin: 8px;
}
#stop { background: #ef4444; color: white; }
#screen {
    position: relative;
    width: 100%;
    max-width: 720px;
    margin: 20px auto;
}
video {
    width: 100%;
    display: block;
}
canvas {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
}
</style>
</head>
<body>

<h1>School Face Detection System</h1>
<p id="status">Kamera henüz başlatılmadı.</p>

<button onclick="startCamera()">Kamerayı Başlat</button>
<button id="stop" onclick="stopCamera()">Durdur</button>

<div id="screen">
    <video id="video" autoplay muted playsinline></video>
    <canvas id="overlay"></canvas>
</div>

<h2 id="count">Tespit edilen yüz: 0</h2>

<script>
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusText = document.getElementById("status");
const countText = document.getElementById("count");

const capture = document.createElement("canvas");
capture.width = 640;
capture.height = 480;

let stream = null;
let running = false;
let busy = false;

async function startCamera() {
    if (running) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;

        running = true;
        statusText.textContent = "Kamera aktif";
        detectLoop();

    } catch (error) {
        statusText.textContent = "Kamera hatası: " + error.message;
    }
}

function stopCamera() {
    running = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    countText.textContent = "Tespit edilen yüz: 0";
    statusText.textContent = "Kamera durduruldu";
}

async function detectLoop() {
    if (!running || busy) return;
    busy = true;

    try {
        const captureCtx = capture.getContext("2d");
        captureCtx.drawImage(
            video, 0, 0,
            capture.width, capture.height
        );

        const blob = await new Promise(resolve =>
            capture.toBlob(resolve, "image/jpeg", 0.75)
        );

        if (!blob) throw new Error("Görüntü oluşturulamadı");

        const response = await fetch("/detect", {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: blob
        });

        if (!response.ok) throw new Error("Sunucu hatası");

        const data = await response.json();

        if (running) {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            const sx = overlay.width / capture.width;
            const sy = overlay.height / capture.height;

            for (const face of data.faces) {
                ctx.strokeStyle = "#22c55e";
                ctx.lineWidth = 3;
                ctx.strokeRect(
                    face.x * sx,
                    face.y * sy,
                    face.w * sx,
                    face.h * sy
                );
            }

            countText.textContent =
                "Tespit edilen yüz: " + data.faces.length;
        }
    } catch (error) {
        if (running) statusText.textContent = error.message;
    } finally {
        busy = false;
        if (running) setTimeout(detectLoop, 350);
    }
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return HTML

@app.route("/detect", methods=["POST"])
def detect():
    data = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Gecersiz goruntu"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    results = [
        {"x": int(x), "y": int(y),
         "w": int(w), "h": int(h)}
        for x, y, w, h in faces
    ]

    return jsonify({"faces": results})

if __name__ == "__main__":
    print("Face Detection System baslatiliyor...")
    app.run(host="0.0.0.0", port=5000, debug=False)
