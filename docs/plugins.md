# 插件协议

AutoVision 通过 `autovision.plugins` entry-point 组发现插件。**导入
`autovision` 不会自动执行插件代码**；显式调用 `discover_plugins()` 或
`autovision-cli plugins` 才是信任边界。

## 插件种类

| kind | 用途 | 是否自动接入核心 |
|---|---|---|
| `model_source` | 新 backbone 包（`ModelSource` / `CatalogModelSource`） | ✅ 注册到编码器目录 |
| `dataset` | 数据集工厂 | 存取用（调用方显式取用） |
| `metric` | 指标工厂 | 存取用 |
| `loss` | 损失工厂 | 存取用 |
| `task` | 任务构建器 | 存取用（高级） |
| `postprocess` | 后处理解码器 | 存取用 |

`model_source` 工厂必须返回 `autovision.encoders.package.ModelSource` 实例；
注册等价于 `register_source`，因此 `autovision.detection(...)` /
`get_wrapped_model(...)` 能解析新的包名。

## 写一个插件

```python
from autovision.plugins import PluginInfo

def plugin():
    return PluginInfo(
        name="my-metric", version="0.1.0", kind="metric",
        factory=MyMetric, description="My metric",
    )
```

`pyproject.toml`：

```toml
[project.entry-points."autovision.plugins"]
my_plugin = "my_package:plugin"
```

完整可安装模板：`examples/plugin-demo/`（metric + dataset 两个工厂）。
安装后运行：

```bash
pip install -e examples/plugin-demo
autovision-cli plugins action=list
autovision-cli plugins action=check
```

## API 版本

`PLUGIN_API_VERSION = 1`。插件必须声明相同版本；不匹配会被拒绝，而不是
猜测加载。发现失败默认收集到 `DiscoveryResult.errors`；`strict=True` 时
首个失败即抛 `PluginError`。

## 边界

- 插件是本地代码，等同依赖；不要加载不可信来源。
- 插件不得绕过权重许可门控 / 溯源门禁。
- `dataset` / `metric` / `loss` / `task` 的工厂目前是“可发现、可调用”，
  与具体训练入口的深度接线按需求推进；`model_source` 已完成目录接入。
