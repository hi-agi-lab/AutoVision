# Kubernetes reference manifests

These manifests are a **reference deployment**, not a hardened gateway. They
assume:

- a built/published image (the repository's `Dockerfile` is training-oriented;
  build a serving image that installs `autovision[serving,video]`),
- an RWX/ROX volume containing the registry `prod/` tree (or an exported
  artifact + sidecars),
- a Secret `autovision-serving` with the `api-key` value.

```bash
kubectl create secret generic autovision-serving \
  --from-literal=api-key="$(openssl rand -hex 32)"
kubectl apply -k deploy/k8s
```

## What the manifests cover

- Deployment with non-root user, read-only root filesystem, dropped
  capabilities, resource requests/limits, rolling updates and a 60 s
  termination grace period.
- `/healthz` liveness and `/readyz` readiness probes.
- Service, HPA (CPU 70%), PDB (`minAvailable: 1`) and a model PVC.

## Known boundaries

- **Metrics and rate limits are per worker.** The in-process Prometheus
  registry and token bucket are not aggregated across uvicorn workers. Keep
  `workers=1` and scale with replicas (this is what the HPA does), or put a
  gateway/prometheus multiprocess collector in front for aggregate quotas.
- **`/metrics` and `/drift` are API-key protected** when an API key is set
  (`protect_observability_endpoints=True`). Configure Prometheus with the
  same key, or point scraping at an internal-only listener; `/healthz` and
  `/readyz` stay unauthenticated for probes.
- **No model hot reload.** A new version is rolled out with
  `kubectl rollout restart deployment/autovision-serving` (the registry
  `current` pointer selects the version at process start). In-flight requests
  drain for `timeout_graceful_shutdown=45` (CLI arg) within the pod's 60 s
  grace period.
- **TLS termination** is expected at the ingress/mesh. The reference server
  also supports `ssl_keyfile`/`ssl_certfile` for direct TLS deployments.
- The PVC's `ReadOnlyMany` mode requires a storage class that supports it;
  adjust to your cluster's RWX class.
