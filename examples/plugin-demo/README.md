# AutoVision demo plugin

A minimal, installable plugin package. It registers two factories through
the `autovision.plugins` entry-point group:

| name | kind | factory |
|---|---|---|
| `autovision-demo-metric` | `metric` | `DemoMetric` (top-k accuracy) |
| `autovision-demo-dataset` | `dataset` | `DemoDataset` |

## Install and discover

```bash
pip install -e examples/plugin-demo
autovision-cli plugins action=list
autovision-cli plugins action=check
```

The framework does **not** auto-load plugins at import time; `plugins list`
(or an explicit `autovision.discover_plugins()`) is the trust boundary.

## Use a factory

```python
from autovision.plugins import get_plugin

metric = get_plugin("metric", "autovision-demo-metric").factory()
dataset = get_plugin("dataset", "autovision-demo-dataset").factory(rows=[])
```

## Model-source plugins

A `model_source` factory must return an
`autovision.encoders.package.ModelSource` (or `CatalogModelSource`) instance;
registration delegates to `register_source`, so `autovision.detection(...)` /
`get_wrapped_model(...)` can resolve the new package name. See
`autovision.encoders.torchvision` for a reference implementation.

## Publishing

Build with `python -m build`; the entry point travels in the wheel metadata.
Keep the plugin API version in mind (`PLUGIN_API_VERSION = 1`); a version
mismatch is rejected instead of silently loaded.
