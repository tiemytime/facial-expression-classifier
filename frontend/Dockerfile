# nginx-unprivileged runs as a non-root user and listens on 8080 by default,
# satisfying the "run as non-root" guidance without extra Dockerfile plumbing.
FROM nginxinc/nginx-unprivileged:1.27-alpine

USER root
RUN apk add --no-cache gettext curl
COPY . /usr/share/nginx/html
COPY docker-entrypoint.sh /docker-entrypoint.d/40-render-config.sh
RUN chmod +x /docker-entrypoint.d/40-render-config.sh \
    && rm -f /usr/share/nginx/html/Dockerfile /usr/share/nginx/html/docker-entrypoint.sh
USER nginx

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1
