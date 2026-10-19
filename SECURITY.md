# Security Policy

AutoVision is a self-hosted training/export framework. We take reports about
the framework code, model-loading paths, and the reference inference runtime
seriously.

## Supported versions

| Version | Status | Security fixes |
|---|---|---|
| `0.3.x` | Current Beta | Yes |
| `0.2.x` | Previous line | Critical issues, best effort |
| `< 0.2` | Unsupported | No |

The policy for LTS branches, patch windows and backports is defined in
`docs/lts_policy.md`.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Use one of these private channels:

- GitHub Security Advisories for this repository (Security → Report a
  vulnerability), or
- the private support ticket / contact named in your enterprise agreement.

Please include, when possible:

- affected version (`autovision.__version__` or the installed distribution),
- a minimal reproduction (command or Python snippet),
- the impact you believe is reachable (code execution, data disclosure,
  denial of service, model/weight integrity, licensing bypass),
- whether the issue affects training, export, or the reference `serve`
  endpoint.

Never include production credentials, private keys, customer data, or
restricted-license weight files in a report. Share a hash or a synthetic
reproduction instead.

## Response targets

| Severity | Acknowledgement | Target fix |
|---|---|---|
| Critical (RCE, arbitrary file write, credential disclosure) | 1 business day | 7 days |
| High (model integrity bypass, auth bypass in the reference server) | 3 business days | Next PATCH |
| Medium / Low | 5 business days | Next MINOR/PATCH |

These are targets, not contractual SLAs; enterprise SLAs are governed by the
support agreement.

## Scope

In scope:

- framework code under `src/autovision/`,
- checkpoint/export loading and the `weights_only` allowlist,
- weight/dataset download and sha256 verification,
- the reference FastAPI serving app (`autovision.serving`),
- release/build scripts and dependency declarations.

Out of scope:

- vulnerabilities in third-party SDKs/runtimes (PyTorch, TensorRT, OpenVINO,
  CUDA drivers, vendor edge toolchains) — report those upstream,
- customer datasets and model weights,
- deployments that disabled documented guards (e.g. unauthenticated `serve`
  exposed to the public internet, or `weights_only=False` paths used on
  untrusted files).

## Security-relevant defaults

- Checkpoints and exports are loaded with `torch.load(..., weights_only=True)`
  and an AutoVision-only class allowlist.
- Weight downloads verify a pinned sha256; restricted-license weights are not
  downloaded by default. EoMT hub checkpoints are fail-closed when unpinned;
  `AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1` is an explicit trusted-mirror opt-in.
- Legacy teacher checkpoints that need the full unpickler are refused unless
  `AUTOVISION_TRUST_TEACHER_CHECKPOINT=1` is set for a trusted local artifact.
- Telemetry is opt-in (`AUTOVISION_TELEMETRY=0` by default).
- The reference serving app supports an opt-in API key, request-body and
  image-pixel limits, a per-process token-bucket rate limit, and runs inference
  off the event loop. `/metrics` and `/drift` follow the API key
  (`protect_observability_endpoints`); `/healthz`/`/readyz` stay public for
  probes. `serve(...)` adds TLS, multi-worker launch and a graceful-drain
  timeout; a prediction-distribution drift endpoint and A/B/canary routing are
  available.
  `trust_proxy_headers=true` must only be used behind a trusted reverse proxy
  (otherwise clients can spoof `X-Forwarded-For` to bypass per-IP throttling).
  Global quotas, alerting and WAF enforcement remain deployment responsibilities.

See `docs/known_limits.md` for the complete boundary.
