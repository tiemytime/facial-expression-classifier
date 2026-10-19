/**
 * Minimal "try it" API explorer: one button per endpoint, firing a real
 * request against the same backend the tracker page talks to.
 */
(() => {
  const apiBaseUrl = (window.APP_CONFIG && window.APP_CONFIG.apiBaseUrl) || "";

  function apiUrl(path) {
    return `${apiBaseUrl}${path}`;
  }

  function showResponse(endpoint, status, elapsedMs, body) {
    const block = document.querySelector(`.response-block[data-response="${endpoint}"]`);
    const statusEl = block.querySelector(".response-status");
    const timeEl = block.querySelector(".response-time");
    const codeEl = block.querySelector("code");

    statusEl.textContent = status === null ? "network error" : `${status}`;
    statusEl.className = `response-status ${status !== null && status < 400 ? "status-ok" : "status-bad"}`;
    timeEl.textContent = elapsedMs === null ? "" : `${elapsedMs.toFixed(0)} ms`;
    codeEl.textContent = typeof body === "string" ? body : JSON.stringify(body, null, 2);
    block.hidden = false;
  }

  async function sendGet(endpoint, path) {
    const start = performance.now();
    try {
      const res = await fetch(apiUrl(path), { cache: "no-store" });
      const elapsed = performance.now() - start;
      const body = await res.json().catch(() => ({}));
      showResponse(endpoint, res.status, elapsed, body);
    } catch (err) {
      showResponse(endpoint, null, null, `Request failed: ${err.message}`);
    }
  }

  async function sendPredict(file) {
    const start = performance.now();
    try {
      const formData = new FormData();
      formData.append("file", file, file.name || "frame.jpg");
      const res = await fetch(apiUrl("/predict"), { method: "POST", body: formData });
      const elapsed = performance.now() - start;
      const body = await res.json().catch(() => ({}));
      showResponse("predict", res.status, elapsed, body);
    } catch (err) {
      showResponse("predict", null, null, `Request failed: ${err.message}`);
    }
  }

  document.querySelectorAll(".try-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const endpoint = btn.dataset.endpoint;
      btn.disabled = true;
      const originalText = btn.textContent;
      btn.textContent = "Sending…";

      if (endpoint === "health") await sendGet("health", "/health");
      else if (endpoint === "ready") await sendGet("ready", "/ready");
      else if (endpoint === "predict") {
        const file = document.getElementById("predictFile").files[0];
        if (file) await sendPredict(file);
      }

      btn.textContent = originalText;
      btn.disabled = endpoint === "predict" ? !document.getElementById("predictFile").files[0] : false;
    });
  });

  const predictFile = document.getElementById("predictFile");
  const predictBtn = document.querySelector('.try-btn[data-endpoint="predict"]');
  predictFile.addEventListener("change", () => {
    predictBtn.disabled = !predictFile.files[0];
  });
})();
