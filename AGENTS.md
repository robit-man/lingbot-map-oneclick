# AGENTS.md — LingBot Map One-Click Contract

Keep `README.md` and `DEPLOYMENT.md` synchronized with runtime API, deployment,
checkpoint, security, or ownership changes.

## CUDA and deployment

- Before changing or starting this CUDA workload, run `docker gpu discover`.
- Launch only through the repository's scoped `docker gpu run --gpu GPU_UUID`
  workflow. The container must receive exactly the reserved UUID and lease.
- `LINGBOT_GPU_INDEX` is an operator convenience only. Resolve it against the
  ordered `docker gpu discover` list, require broker eligibility, and pass the
  resulting exact UUID to the lease and child. Never pass the numeric index to
  CUDA or Compose. Reject simultaneous index and UUID constraints.
- Call broker `prepare` before inference growth and `ready` after CUDA cleanup.
  Do not add static Compose GPU counts or anonymous CUDA allocation.
- `LINGBOT_DEPLOYMENT_MODE=internal` is the NOCLIP production mode: loopback
  application only, no Cloudflared service. Public mode remains available for
  the standalone Maproom.

## NOCLIP API invariants

- Require the bearer token on every reconstruction/status/result/artifact or
  cancellation route. Health/readiness alone are public loopback probes.
- Accept only `noclip.lingbot.request/1.0` with exact WGS84, ENU, OpenCV camera
  axes, `xyzw` quaternion order, and meter units.
- The trajectory and GLB must use the same scene-alignment transform. Preserve
  the media sequence, synchronized monotonic time, and sensor sample mapping.
- Persist terminal job state atomically. Reload terminal jobs on restart and
  convert interrupted queued/running jobs to an explicit failure; never let an
  upstream caller poll a lost in-memory job forever.
- NOCLIP backend owns account auth, holdings quotas, media retention,
  publication, altitude datum, and placement. Do not duplicate or bypass those
  decisions in this internal GPU worker.

## Verification

Run `python3 -m pytest -q`, `python3 -m compileall -q webapp tests`,
`bash -n run.sh scripts/*.sh`, and the broker-aware `./run.sh verify` before a
production handoff. A full GPU acceptance uses `./run.sh smoke`.
