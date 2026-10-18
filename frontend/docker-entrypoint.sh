#!/bin/sh
# Runs automatically before nginx starts (nginx's base image executes every
# *.sh script in /docker-entrypoint.d/). Renders config.js from the env var
# at container start, so API_BASE_URL is a genuine runtime setting, not
# something baked into the image at build time.
set -eu
: "${API_BASE_URL:=}"
envsubst '${API_BASE_URL}' < /usr/share/nginx/html/config.template.js > /usr/share/nginx/html/config.js
