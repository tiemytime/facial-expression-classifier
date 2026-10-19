/**
 * Captures periodic snapshots from the webcam and posts them to the backend
 * /predict endpoint. This is a deliberate choice over true video streaming
 * (WebRTC/websockets): the assessment's expected flow is a single
 * input -> single prediction round trip, and polling snapshots keeps the
 * frontend a plain static page with no streaming server of its own, which
 * is the whole reason this app is a trivial, free, single-container
 * deploy. See README.md "Known limitations" for the latency trade-off.
 */
(() => {
  const apiBaseUrl = (window.APP_CONFIG && window.APP_CONFIG.apiBaseUrl) || "";

  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const intervalSelect = document.getElementById("interval");

  const apiStatusEl = document.getElementById("apiStatus");
  const faceStatusEl = document.getElementById("faceStatus");
  const gazeStatusEl = document.getElementById("gazeStatus");
  const expressionStatusEl = document.getElementById("expressionStatus");
  const timingStatusEl = document.getElementById("timingStatus");
  const errorBanner = document.getElementById("errorBanner");

  let stream = null;
  let captureTimer = null;
  let inFlight = false;

  function apiUrl(path) {
    return `${apiBaseUrl}${path}`;
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
  }

  function setBadge(el, ok, text) {
    el.textContent = text;
    el.className = `badge ${ok ? "badge-ok" : "badge-bad"}`;
  }

  async function checkHealth() {
    try {
      const res = await fetch(apiUrl("/health"), { cache: "no-store" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      setBadge(apiStatusEl, true, "online");
    } catch (err) {
      setBadge(apiStatusEl, false, "unreachable");
      showError(`Cannot reach the API at "${apiBaseUrl}". Is the backend running and is apiBaseUrl configured correctly?`);
    }
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    } catch (err) {
      showError(`Camera access failed: ${err.message}. This page must be served over HTTPS or localhost.`);
      return;
    }
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    scheduleCapture();
  }

  function stopCamera() {
    if (captureTimer) clearTimeout(captureTimer);
    captureTimer = null;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  function scheduleCapture() {
    const delay = Number(intervalSelect.value);
    captureTimer = setTimeout(captureAndPredict, delay);
  }

  async function captureAndPredict() {
    if (!stream) return;
    if (inFlight) {
      scheduleCapture();
      return;
    }
    inFlight = true;

    const snapshot = document.createElement("canvas");
    snapshot.width = video.videoWidth;
    snapshot.height = video.videoHeight;
    snapshot.getContext("2d").drawImage(video, 0, 0);

    snapshot.toBlob(
      async (blob) => {
        try {
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");
          const res = await fetch(apiUrl("/predict"), { method: "POST", body: formData });
          const body = await res.json();
          if (!res.ok) {
            throw new Error(body.detail || `request failed with status ${res.status}`);
          }
          clearError();
          setBadge(apiStatusEl, true, "online");
          renderPrediction(body);
        } catch (err) {
          setBadge(apiStatusEl, false, "error");
          showError(`Prediction failed: ${err.message}`);
        } finally {
          inFlight = false;
          if (stream) scheduleCapture();
        }
      },
      "image/jpeg",
      0.85
    );
  }

  function renderPrediction(body) {
    faceStatusEl.textContent = body.face_detected ? "yes" : "no";
    gazeStatusEl.textContent = body.gaze ? `${body.gaze.direction} (${(body.gaze.confidence * 100).toFixed(0)}%)` : "—";
    expressionStatusEl.textContent = body.expression
      ? `${body.expression.label} (${(body.expression.confidence * 100).toFixed(0)}%)`
      : "—";
    timingStatusEl.textContent = `${body.inference_time_ms.toFixed(1)} ms`;

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = 4;
    ctx.strokeStyle = body.face_detected ? "#22c55e" : "#ef4444";
    ctx.strokeRect(2, 2, overlay.width - 4, overlay.height - 4);

    if (body.face_detected) {
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "16px sans-serif";
      const label = [body.gaze && `gaze: ${body.gaze.direction}`, body.expression && `expression: ${body.expression.label}`]
        .filter(Boolean)
        .join("  |  ");
      ctx.fillText(label, 10, overlay.height - 12);
    }
  }

  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);

  checkHealth();
})();
