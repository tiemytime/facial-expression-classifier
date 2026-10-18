// Templated at container start by docker-entrypoint.sh (envsubst) in the
// standalone frontend image. In the combined single-origin deployment mode
// (see deploy/huggingface/), apiBaseUrl is left empty so requests go to the
// same origin the page was served from, and no CORS is involved at all.
window.APP_CONFIG = {
  apiBaseUrl: "${API_BASE_URL}",
};
