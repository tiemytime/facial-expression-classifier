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

  const logsToggle = document.getElementById("logsToggle");
  const logsPanel = document.getElementById("logsPanel");
  const logsList = document.getElementById("logsList");
  const logsCountEl = document.getElementById("logsCount");
  const logsClearBtn = document.getElementById("logsClearBtn");

  let stream = null;
  let captureTimer = null;
  let inFlight = false;

  // Request log: a client-side view of traffic this page sends to the API,
  // gated behind a toggle so nothing is recorded (or rendered) unless the
  // user opts in. Capped to avoid unbounded memory growth on a long session.
  const MAX_LOG_ENTRIES = 50;
  let logsEnabled = false;
  try {
    logsEnabled = localStorage.getItem("logsEnabled") === "true";
  } catch (err) {}
  let logEntries = [];

  function applyLogsEnabled(enabled) {
    logsEnabled = enabled;
    logsToggle.checked = enabled;
    logsPanel.hidden = !enabled;
    try {
      localStorage.setItem("logsEnabled", String(enabled));
    } catch (err) {}
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderLogs() {
    logsCountEl.textContent = `${logEntries.length} request${logEntries.length === 1 ? "" : "s"}`;
    if (logEntries.length === 0) {
      logsList.innerHTML = '<li class="logs-empty">No requests logged yet.</li>';
      return;
    }
    logsList.innerHTML = logEntries
      .map((entry) => {
        const statusClass = entry.ok ? "status-ok" : "status-bad";
        return `<li class="log-entry">
          <span class="log-time">${entry.time}</span>
          <span class="log-method">${entry.method}</span>
          <span class="log-path">${entry.path}</span>
          <span class="log-status ${statusClass}">${entry.status}</span>
          <span class="log-time-ms">${entry.ms}ms</span>
          <span class="log-summary">${escapeHtml(entry.summary || "")}</span>
        </li>`;
      })
      .join("");
  }

  function addLogEntry({ method, path, status, ms, ok, summary }) {
    if (!logsEnabled) return;
    const now = new Date();
    const time = now.toLocaleTimeString([], { hour12: false }) + "." + String(now.getMilliseconds()).padStart(3, "0");
    logEntries.unshift({ time, method, path, status, ms, ok, summary });
    if (logEntries.length > MAX_LOG_ENTRIES) logEntries.length = MAX_LOG_ENTRIES;
    renderLogs();
  }

  function summarizePrediction(body) {
    if (!body.face_detected) return "no face detected";
    const parts = [];
    if (body.gaze) parts.push(`gaze=${body.gaze.direction}`);
    if (body.expression) parts.push(`expression=${body.expression.label}`);
    parts.push(`${body.inference_time_ms.toFixed(0)}ms inference`);
    return parts.join(" ");
  }

  applyLogsEnabled(logsEnabled);
  logsToggle.addEventListener("change", () => applyLogsEnabled(logsToggle.checked));
  logsClearBtn.addEventListener("click", () => {
    logEntries = [];
    renderLogs();
  });

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
    const start = performance.now();
    try {
      const res = await fetch(apiUrl("/health"), { cache: "no-store" });
      const ms = Math.round(performance.now() - start);
      if (!res.ok) throw new Error(`status ${res.status}`);
      setBadge(apiStatusEl, true, "online");
      addLogEntry({ method: "GET", path: "/health", status: res.status, ms, ok: true, summary: "ok" });
    } catch (err) {
      const ms = Math.round(performance.now() - start);
      setBadge(apiStatusEl, false, "unreachable");
      showError(`Cannot reach the API at "${apiBaseUrl}". Is the backend running and is apiBaseUrl configured correctly?`);
      addLogEntry({ method: "GET", path: "/health", status: "ERR", ms, ok: false, summary: err.message });
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
        const start = performance.now();
        let status = "ERR";
        let ok = false;
        let summary = "";
        try {
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");
          const res = await fetch(apiUrl("/predict"), { method: "POST", body: formData });
          status = res.status;
          const body = await res.json();
          if (!res.ok) {
            summary = body.detail || `request failed with status ${res.status}`;
            throw new Error(summary);
          }
          ok = true;
          clearError();
          setBadge(apiStatusEl, true, "online");
          renderPrediction(body);
          summary = summarizePrediction(body);
        } catch (err) {
          setBadge(apiStatusEl, false, "error");
          showError(`Prediction failed: ${err.message}`);
          if (!summary) summary = err.message;
        } finally {
          const ms = Math.round(performance.now() - start);
          addLogEntry({ method: "POST", path: "/predict", status, ms, ok, summary });
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
