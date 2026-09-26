# One-command Web Deployment

This repository includes a broker-aware, single-GPU deployment with:

- automatic Hugging Face checkpoint download before the GPU lease is acquired;
- automatic local bootstrap venv and pinned Cloudflared container installation;
- pinned checkpoint revision, byte-size validation, and SHA-256 verification;
- a token-protected upload API with a one-job GPU queue;
- a Vite-minified Three.js GLB viewer;
- ephemeral Cloudflare Quick Tunnel and named-tunnel modes;
- a supervised systemd user service that holds the scoped GPU lease for the
  exact lifetime of the containers.

## Start

```bash
./run.sh
```

The first start creates `.venv`, installs the lightweight checkpoint tooling,
downloads the 4.63 GB model, builds the image and minified frontend, installs
the pinned Cloudflared container, and starts the user service. The command waits
for the CUDA model and tunnel before printing non-secret access details. Retrieve
the token only when provisioning a trusted client:

```bash
./run.sh url
./run.sh token
./run.sh verify
```

If a token is exposed, rotate it without printing the replacement, then copy
the new value into trusted clients and restart the deployment:

```bash
./run.sh rotate-token
```

The app is also bound locally at `http://127.0.0.1:8080`. The Cloudflare URL
serves only the web app; the host port is never bound to a public interface.

## Configuration

On first use, `run.sh` creates the venv and `scripts/deploy.sh` copies
`.env.example` to `.env`, generates a
48-character random API token, and sets file permissions to `0600`.

The default checkpoint is reproducibly pinned to:

```text
repo:     robbyant/lingbot-map
file:     lingbot-map.pt
revision: 204754b72bb24f561f8d7e7e1e4e4cd9e809adf9
sha256:   ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72
size:     4632303465 bytes
```

To use another checkpoint, change the repository, filename, revision, checksum,
and size together. Set `LINGBOT_MODEL_PATH` only when a checkpoint is already
mounted in the app container.

The deploy script runs `docker gpu discover` on every start. To constrain the
service to the host's GPU 1, set `LINGBOT_GPU_INDEX=1`; the script resolves that
entry to its current UUID, verifies that the broker selected it, and passes only
that UUID to both `docker gpu run` and the container. `LINGBOT_GPU_UUID` remains
available for an explicit UUID constraint, but the index and UUID selectors are
mutually exclusive. Leave both blank (or set the index to `auto`) to select the
broker-eligible GPU with the most current headroom. The app also calls `prepare`
before each inference burst and `ready` after CUDA memory stabilizes.
Every lease records owner `lingbot-map`, the NOCLIP reconstruction purpose, and
the operator-configurable `LINGBOT_EXPECTED_DURATION` (default 86400 seconds).

## Cloudflare Modes

With `CLOUDFLARE_TUNNEL_TOKEN` blank, the service starts a Quick Tunnel and its
temporary `trycloudflare.com` URL can be read with `./run.sh url`.
Quick Tunnels are convenient for ad-hoc access but have no stable hostname or
availability guarantee.

For a stable hostname, create a named tunnel in Cloudflare and put both its
token and public route in `.env`. The URL is required so startup and verification
can test the configured hostname:

```dotenv
CLOUDFLARE_TUNNEL_TOKEN=replace-with-the-named-tunnel-token
LINGBOT_PUBLIC_URL=https://map.example.com
```

The application bearer token remains required in both modes. Cloudflare Access
can be added in front of the named tunnel for another authentication layer.

## NOCLIP Internal Mode

For `api.noclip.org`, disable the public tunnel and keep the worker on its
loopback bind:

```dotenv
LINGBOT_DEPLOYMENT_MODE=internal
LINGBOT_PORT=7410
```

Then run the ordinary supervised deployment:

```bash
./run.sh start
./run.sh verify
```

`verify` checks the checkpoint and resident-model readiness and reports the
tunnel gate as skipped. `./run.sh url` returns the loopback URL. Copy the value
from `./run.sh token` into the backend's `LINGBOT_MAP_SERVICE_TOKEN`; never put
it in a URL or expose port 7410 on a public interface. Internal mode still runs
`docker gpu discover`, reserves one exact UUID with `docker gpu run`, calls
`prepare` before inference growth, and returns the lease to ready state after
cleanup.

The NOCLIP contract endpoints are:

- `POST /v1/reconstructions`
- `GET|DELETE /v1/reconstructions/{jobId}`
- `GET /v1/reconstructions/{jobId}/result`
- `GET /v1/reconstructions/{jobId}/artifacts/{fileName}`

All require `Authorization: Bearer <LINGBOT_API_TOKEN>`; the deprecated
`X-API-Token` form is rejected. Submission also requires an `Idempotency-Key`.
The key maps durably to one provider job, same-key/same-request replay returns
that job, and conflicting reuse is rejected.

Status responses report the durable stage and progress plus queue position,
queue size, active/available capacity, and cancellation timestamps. Queued work
can cancel immediately. Running work remains `cancelling` while an upstream
model call is in flight, becomes `cancelled` only after CUDA cleanup, and has
all partial result output removed before that terminal status is visible.
Decoding, image preprocessing, GPU admission, postprocessing, export, and
artifact finalization have cooperative checkpoints. The bounded upstream
`inference_streaming` call itself cannot safely stop mid-call and is reported as
such; cancellation is checked immediately when it returns.

Completed NOCLIP results contain a normalized GLB, bounded binary-PLY
point-cloud LOD, solved trajectory, per-frame intrinsics, confidence/quality
diagnostics, and `noclip.lingbot.reconstruction/1.0` manifest. They all declare
the `exported_lingbot_model` frame and the manifest records the exact transform
convention and SHA-256/size metadata. The LOD is capped by
`LINGBOT_POINT_CLOUD_MAX_POINTS` (default 250000). Confidence filtering remains
enabled; sky masking remains disabled until a representative outdoor field
corpus demonstrates an accuracy benefit.

Video decoding retains a presentation timestamp for every sampled frame. The
trajectory maps decoded PTS into the capture monotonic clock and reports the
time source, uncertainty, and nearest sensor-sample delta per frame. Invalid or
non-monotonic decoder PTS fall back to source-FPS timing with explicit
one-frame uncertainty. Downstream alignment remains responsible for bounded
interpolation and for rejecting evidence outside its validity window.

Persisted completed/failed/cancelled job records are reloaded after restart;
staging/queued/running records become explicit restart failures, while an
interrupted cancellation becomes acknowledged. Upload and decoded-frame
temporaries are removed at terminal completion; retained durable artifacts are
bounded by `LINGBOT_RETAIN_JOBS`.

## Operations

```bash
# Service and broker state
./run.sh status

# Follow app and tunnel output
./run.sh logs

# Stop service and containers; keep checkpoint/results
./run.sh stop

# Low-level container teardown (the user service must already be stopped)
scripts/deploy.sh down
```

Persistent data lives under `data/models/` and `data/jobs/`. Container teardown
does not delete either directory.

## Verification Gates

`./run.sh verify` checks three expected outcomes:

1. the configured checkpoint downloads or resolves and passes integrity checks;
2. `/readyz` confirms the CUDA model is resident;
3. a Quick Tunnel URL or configured named-tunnel URL is available.

For a full authenticated NOCLIP acceptance using a short MP4 generated from the
bundled loop scene, including GLB and point-cloud contract validation:

```bash
./run.sh smoke
```

For source-level validation before launch:

```bash
./run.sh test
```

The direct source gates are `pytest -q`, `python3 -m pytest -q`,
`python3 -m compileall -q webapp tests`, and `bash -n run.sh scripts/*.sh`.
Plain `pytest` resolves the repository packages without a manual `PYTHONPATH`.

## Video workflow

The browser accepts one video (`MP4`, `MOV`, `M4V`, `AVI`, `WebM`, or `MKV`) or
an ordered image sequence. For video, the service samples frames at the chosen
rate, caps the job before inference, serializes GPU work through a one-worker
queue, and reports each phase to the site. When the GLB export is complete, the
same page loads it into an orbit/zoom viewer without exposing an unauthenticated
artifact URL.

The defaults are intentionally conservative for a tunneled endpoint: 90 MiB per
upload, 96 sampled frames, four queued jobs, and 20 retained results. Adjust the
`LINGBOT_MAX_*` values in `.env` before restarting with `./run.sh restart`.
