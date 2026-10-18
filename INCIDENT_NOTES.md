# Incident notes - troubleshooting exercise

**Incident.** A new release was deployed. The frontend loads, but
predictions now fail intermittently. CI passed. Container logs show
occasional connection errors from the frontend to the API, some requests
return HTTP 500, and the API container restarts after deployment. The
previous release worked.

## 1. First checks and evidence to collect

- `docker ps` / platform dashboard - confirm the API container's restart
  count and current status; `docker logs api --since 30m` for the exact
  error and stack trace around each 500 and each restart.
- `docker inspect api` - check the exit code and whether it was
  `OOMKilled: true`, and compare the configured memory limit against the
  previous release's.
- Call `GET /health` and `GET /ready` **directly on the API**, bypassing
  the frontend, to separate "API process is unhealthy" from "frontend
  can't reach a healthy API."
- `docker exec api env` (or the platform's equivalent) - confirm the
  environment variables actually present in the running container match
  what was intended for this release, especially anything related to
  networking (`CORS_ORIGINS`, internal host/port names).
- Diff the new release against the previous one: image tag, Dockerfile
  changes, env var changes, and any config file changes, via the CI/CD
  deployment history.
- `docker stats` / host metrics - CPU and memory pressure at the time of
  the 500s and restarts.
- Browser dev tools network tab on the frontend - is the connection error
  DNS failure, connection refused, CORS rejection, or a timeout? Each
  points to a different layer.

## 2. Plausible causes across categories (not settling on one)

- **Configuration**: the frontend's `API_BASE_URL` (or equivalent) was
  changed in this release to a value only resolvable *inside* the
  deployment network (e.g. a Docker Compose service name or internal
  hostname) - but that value is resolved by the **user's browser**, not by
  a container, so it fails intermittently depending on client DNS/cache
  behaviour and looks exactly like "occasional connection errors." This is
  a real trap in this codebase - see the callout in `docker-compose.yml`.
- **Networking**: a changed exposed port, a reverse proxy/load balancer
  route that wasn't updated for the new release, or a health-check path
  that doesn't match the new release's actual health endpoint, causing the
  LB to intermittently route to a container it thinks is healthy but isn't
  yet.
- **Container health and resources**: a lowered or unchanged memory limit
  combined with increased memory usage in the new release (e.g. a bigger
  model, more workers, a per-request leak) causing periodic OOM kills -
  "restarts after deployment" plus "intermittent" is a classic
  crash-loop-then-recover pattern, where requests landing during the
  restart window get connection errors or 500s while requests between
  restarts succeed.
- **Application logs**: an uncaught exception on a specific input shape
  introduced by the new release (e.g. a new validation path, an edge case
  in image decoding) crashing the worker process rather than returning a
  clean 4xx/5xx, which would manifest as "works most of the time, 500s and
  restarts occasionally" depending on which inputs happen to trigger it.
- **Model loading**: if the new release lazily loads the model on first
  request instead of at startup (or loads it per-request), the *first*
  request after every restart pays a cold-load cost or fails outright if
  the model artifact isn't present/reachable in the image - again
  intermittent, and specifically correlated with the restarts.

No single cause is assumed; the checks in (1) are chosen specifically to
distinguish between these before deciding on a fix.

## 3. Restoring service safely

- If the API is actively flapping (restarting repeatedly) or user impact
  is ongoing and the cause isn't yet certain: **roll back immediately** to
  the previous known-good image/release rather than attempting a live
  forward-fix under pressure. Rollback is the fastest path to a stable
  baseline and buys time to investigate calmly.
- Only forward-fix without rolling back if the cause is already certain
  *and* trivially safe (e.g., one obviously wrong environment variable)
  *and* it's been verified in staging first.
- After rolling back, confirm recovery with evidence, not assumption:
  restart count returns to zero, `/health` and `/ready` are consistently
  `200`, and the 500 rate on `/predict` drops to baseline.
- Preserve the faulty image, its logs, and the diff against the previous
  release for the postmortem - don't let a "fix forward later" attempt
  overwrite the evidence.

## 4. Changes to reduce likelihood/impact of a recurrence

1. **Automated post-deploy smoke test that gates the rollout.** After
   deploy, have CI/CD call `/health`, `/ready`, and one real `/predict`
   request against the new release before it's considered successful;
   auto-rollback on failure. This would have caught a crash-looping
   container or a broken `API_BASE_URL` before any real user hit it,
   instead of relying on someone noticing intermittent failures.
2. **Alert on restart count and 5xx rate immediately after a deploy.**
   Container restarts and a 500-rate spike are both already visible in
   this app's logs/health signals (see README "Observability"); wiring a
   simple threshold alert on "restart count > 0 in the 10 minutes after a
   deploy" would surface this incident automatically instead of via user
   reports, and would localize it to the deploy that caused it.
