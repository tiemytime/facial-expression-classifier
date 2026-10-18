// Default for running the frontend directly (e.g. `python -m http.server`)
// without Docker. Overwritten at container start from config.template.js in
// both the standalone and combined Docker images - do not rely on this file
// being used in a deployed environment.
window.APP_CONFIG = {
  apiBaseUrl: "http://localhost:8000",
};
