# AutoVision 行业方案

基于 AutoVision 开源底座的行业视频分析方案族：**一套通用范式 + 八个场景
方案 + 一个交付平台**。每个方案把行业口径、合规红线与误报治理工程化成
可验收的系统，而不是"一个模型识别一切"。

![方案族全景](https://raw.githubusercontent.com/hi-agi-lab/ai-traffic/main/images/hero_banner.jpg)

## 通用范式：三模型流水线

所有方案共享同一条判定链——**发现、确认、裁决分离**，每层只回答自己的问题：

| 模型 | 职责 | 回答的问题 |
| --- | --- | --- |
| **AI-Detect** | 开放词汇检测，L1 发现 | "画面里有可判定的对象/事件吗？" |
| **AI-Recognize** | 时序身份追踪 + 区域语义，L2 确认 | "还是之前那个目标吗？相对基线变了吗？出现在哪些点位？" |
| **AI-Reference** | 多模态 grounding 裁决（Qwen3-VL） | "那真的是目标事件——还是光影、雾霭、道具或规范工况？" |

判定链之外的四条统一纪律：

- **范围先行**：每个方案的开篇都是"范围外声明"——系统只输出"疑似/线索"，
  认定权在行业主管（交管/公安/防火条例），措辞纪律有单测保证。
- **人审是主流程**：AI 产出线索，默认进入复核队列；告警预算与台账治理
  控制"狼来了"效应。
- **误报治理结构化**：白名单台账、豁免区域、作业窗口、持续确认窗、
  双源融合——用结构消歧，不靠阈值硬调。
- **fail-loud 与降敏**：断流告警、相机健康门（冻结/散焦/遮挡/移位）、
  夜间与恶劣天气按天气分桶降级声明。

## 方案总览

<div class="grid cards" markdown>

- :material-traffic-cone: **[ai-traffic](https://github.com/hi-agi-lab/ai-traffic)**

    ---

    交通事件三域分析

    遗撒散落物 · 事故复合评分分级 · 拥堵监测

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-traffic)

- :material-fire: **[ai-wildfire](https://github.com/hi-agi-lab/ai-wildfire)**

    ---

    林草火险 + 防汛防台

    早期烟柱 · 水位尺读数 · 星空地三源合并

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-wildfire)

- :material-factory: **[ai-illop](https://github.com/hi-agi-lab/ai-illop)**

    ---

    违规操作视频监管

    SOP 分解为动作清单 · 注册表驱动 · 逐项独立验收

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-illop)

- :material-alert-octagon: **[ai-conflict](https://github.com/hi-agi-lab/ai-conflict)**

    ---

    治安冲突视频分析

    打架/斗殴 T3 长尾 · 疑似告警 + 强制人审 · 告警预算

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-conflict)

- :material-fence: **[ai-perimeter](https://github.com/hi-agi-lab/ai-perimeter)**

    ---

    周界防护视频分析

    越界拌线 · 禁区闯入 · 徘徊 · 离岗

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-perimeter)

- :material-knife: **[ai-security](https://github.com/hi-agi-lab/ai-security)**

    ---

    持械/管制器具监管

    卡口实时告警 · 以图搜械检索 · 定性权在公安

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-security)

- :material-home-city: **[ai-life-line](https://github.com/hi-agi-lab/ai-life-line)**

    ---

    城市生命线视频分析

    井盖缺失 · 路面积水 · 路灯故障 · 双源融合派单

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-life-line)

- :material-city: **[ai-urban](https://github.com/hi-agi-lab/ai-urban)**

    ---

    城市治理视频分析

    九类市容事件 · 检出、复核、工单闭环

    [:octicons-arrow-right-24: 方案仓](https://github.com/hi-agi-lab/ai-urban)

</div>

交付平台：[hi-agi-lab/hi-ai-vision](https://github.com/hi-agi-lab/hi-ai-vision)
（私有）——多点位编排、告警工单、台账治理与复核工作台。

---

## ai-traffic · 交通事件三域分析

![ai-traffic 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-traffic/main/images/demo_montage.jpg)

**遗撒散落物 + 事故复合评分分级 + 拥堵监测。疑似/确认两级口径，认定权在交管。**

- **事故复合评分**：①异常停车 且 ②碎屑 → 疑似（≤60s 抢响应）；加 ③倒地人
  → 较可能，且③命中即升 risk4 一票否决，不等确认窗；3 分钟持续确认窗
  自动升级/关闭，事件内幂等合并。
- **遗撒双路线**：车道内新增 + 静止 ≥10s 承诺达标精度；行驶掉落走 120s
  观测窗静态确认，以图搜车串联取证。
- **消歧三层生效**：车道多边形过滤、白名单台账（公交站/施工区/锥桶）、
  拥堵排队消歧（排队中的静止不计异常停车）。
- **提示词工程量化**：同一批 130 图，通用提示词检出 0%，领域提示词把
  散落物拉到 79%、事故特征 50%——提示词随代码产线化交付。

### 技术架构

![ai-traffic 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-traffic/main/images/architecture_zh.png)

## ai-wildfire · 林草火险 + 防汛防台

![ai-wildfire 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-wildfire/main/images/demo_montage.jpg)

**瞭望塔双光谱早期烟柱 · 水位尺读数 · 星空地三源合并。零漏报纪律与误报
预算全部工程化。**

- **早期烟窗口**：可见色度 + 热像温升双通道发现；升起/增长/持续三特征
  齐备才确认，早期烟（<10min）是扑救窗口，漏报的代价是灾损。
- **误报裁决**：云、雾、晨霭、扬尘、工业白烟是烟柱的光学近亲——
  AI-Reference 逐条裁决"真的从林冠线升起的烟，还是光学近亲"。
- **法定报告链**：火情报告制度是人工职责——系统不替代，只提速；夜间/
  浓雾能力边界写入降级声明。
- **水位尺读数**：防汛水位自动判读与预警联动，星空地三源在分钟桶配对。

### 技术架构

![ai-wildfire 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-wildfire/main/images/architecture_zh.png)

## ai-illop · 违规操作视频监管

![ai-illop 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-illop/main/images/demo_montage.jpg)

**把企业安全规程（SOP）分解成可训练、可验收的动作清单。注册表驱动，
逐项独立验收，合规红线全部代码化。**

- **第一原则**：引擎不认识"违规操作"，只认识注册表里的 N 个独立任务——
  违规定义来自每家企业的规程，强训"一个大模型"必然不可验收。
- **8 种规则件**：ppe_absence（未戴防护）、region_absence（警示牌缺失）、
  class_in_anchor（人/车锚点关系）等 + 消抖/豁免区域/作业窗口/台账对账。
- **排除台账同等重要**：对"为什么 AI 做不了这条"给出书面答复，
  期望管理前置。
- **KPI 口径**：只有清单级"达标率"（达到各自 SLO 的动作项数/已启用项数），
  永远不存在"违规识别整体准确率"。涉员工监控的制度依据/告知/用途限定
  是上线前置条件。

### 技术架构

![ai-illop 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-illop/main/images/architecture_zh.png)

## ai-conflict · 治安冲突分析

![ai-conflict 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-conflict/main/images/demo_montage.jpg)

**打架/斗殴 T3 长尾专项：疑似肢体冲突告警 + 强制人审 + 告警预算 +
以图搜人取证。**

- **承诺降级**：不承诺"识别打架"（"打"是意图不是像素），承诺"剧烈肢体
  冲突动作"的疑似告警；全链路措辞"疑似肢体冲突"，确认前不出现定性词
  （措辞纪律有单测）。
- **特征三取二**：接触交互（<0.5m ∧ 相对速度 >1.5m/s 持续 ≥2s）+ 动作
  模式（挥击/踢击/拉拽/缠斗骨架命中）+ 场景激化（聚集/倒地）；单特征
  不触发推送。
- **告警预算制**：每点位 ≤8 条/夜，超预算自动降敏（仅紧急级透传），
  次日恢复；打闹消歧三特征把嬉闹从告警里摘出去。
- **人审即标注**：每条推送必经 10s 短片人审，reject 原因枚举直接成为
  飞轮负样本；星光级相机是点位准入。

### 技术架构

![ai-conflict 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-conflict/main/images/architecture_zh.png)

## ai-perimeter · 周界防护

![ai-perimeter 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-perimeter/main/images/demo_montage.jpg)

**越界/拌线 · 禁区闯入 · 徘徊逗留 · 离岗，四类事件一套底座。地面几何
判定，fail-safe 不静默，告警零丢失。**

- **地面几何判定**：透视下画面画框不可靠——脚点轨迹投影到地面坐标再做
  拌线交叉/多边形包含，同一个框近处远处不再代表不同区域。
- **相机健康门**：冻结、过暗、散焦、遮挡、移位、断流六类问题逐类型
  记账；检测器连续异常同样降级——不拿"看不见"当"没有"。
- **告警可靠性**：平台抖动不丢告警，网络分区恢复后补投；撤防窗口零
  事件（连规则状态机都不推进），布防按站点时区跨午夜归属。
- **点位准入**：夜间/红外/远景小人衰减写入点位画像，准入后才承诺指标。

### 技术架构

![ai-perimeter 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-perimeter/main/images/architecture_zh.png)

## ai-security · 持械/管制器具监管

![ai-security 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-security/main/images/demo_montage.jpg)

**双形态：卡口实时告警 + 以图搜械检索。合规红线全部代码化。**

- **范围声明**：通用点位的实时持械检测不承诺、不建设——远距/遮挡/小
  目标掉点严重；玩具刀与真刀光学同构（87 crop 实测天花板 74%），
  纯视觉原理上判不了"真实性"。
- **形态一 · 卡口告警**：专用安检口近距受控通过，器械 ≥64px 硬准入；
  开放词汇检测 + 负提示消歧 + 持握累积 + AI-Reference 三态裁决。
- **形态二 · 以图搜械**：低频轮巡把手持"器械状物"crop 入索引；案件
  驱动，双人授权查询 Top-N + 出现时间线——历史视频从死资产变可检索。
- **定性权在公安**：系统只判"器械状物"，告警措辞不含定性词。

### 技术架构

![ai-security 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-security/main/images/architecture_zh.png)

## ai-life-line · 城市生命线

![ai-life-line 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-life-line/main/images/demo_montage.jpg)

**复用存量摄像头的城市生命线视频分析：井盖缺失、路面积水、路灯故障的
自动发现与派单。**

- **存量复用**：不加装传感器，轮巡式分析（复用既有 GA-T1400 视频接入）
  替代逐帧解码，覆盖巡查盲区。
- **三场景统一运行时**：井盖变化（登记式基线）、路面积水（物理分割 +
  干态基线门控）、路灯故障（台账 + 亮度基线）共享同一轮巡/推送/复核链路。
- **双源融合定级**：视频线索与 IoT 传感、市民上报在分钟桶融合总线配对，
  双源命中即高置信直派——单源不可信时不硬判。
- **气象联动**：暴雨预警主备双源触发，积水点位自动升频，预警解除回落。

### 技术架构

![ai-life-line 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-life-line/main/images/architecture_cn.png)

## ai-urban · 城市治理

![ai-urban 实景](https://raw.githubusercontent.com/hi-agi-lab/ai-urban/main/images/demo_montage.jpg)

**复用存量摄像头的九类市容事件视频分析：检出、复核、工单闭环。**

- **九类事件统一注册**：占道经营、流动摊贩、暴露垃圾、渣土遗撒、道路
  破损、绿化损毁、违规户外广告、违建、乱倒污水——提示词库、规则参数、
  验收目标、复核问句、工单默认均在单一注册表声明，新增类目仅需一行。
- **轮巡规模**：A/B 档点位 1 帧/30s，万路规模约 8–11 张 L4（tile 2×2 口径）；
  违建快照每日 1 帧另计。
- **两阶段判定**：AI-Detect + 规则引擎保召回、抑制瞬态；AI-Reference 仅
  对被标记线索做语义裁决（单张约 1–2s、5GB 显存）。
- **线索不立案**：AI 仅产出线索，默认进人工复核；疏导点白名单必配；
  证据帧带"AI 生成线索"水印，行政处罚须现场重新取证。

### 技术架构

![ai-urban 技术架构](https://raw.githubusercontent.com/hi-agi-lab/ai-urban/main/images/architecture.png)

---

## 底座：AutoVision 开源框架

方案引擎统一通过 `pip install autovision-framework`（Apache-2.0）消费
底座稳定接缝，不做重复建设：

| 稳定接缝 | 方案侧用途 |
| --- | --- |
| `InferenceResult` 事件契约 | 检出对象 → 平台事件的标准载体（mask 默认 RLE） |
| `PolygonZone` / `LineZone` | 车道多边形、禁区、拌线、岗位区域判定 |
| `run_stream` / `multi_stream` | 断流告警、重连退避、多点位并发轮巡 |
| `registry` + model card | 底座权重版本固定与交付审计（sha256 钉住） |
| `export`（ONNX/TensorRT + parity 门禁） | 边缘盒子部署，数值可对拍 |
| `serving` 参考 FastAPI | 平台接入原型对齐 |

框架能力细节见[架构总览](framework_design.md)、
[插件协议](plugins.md)与[导出部署](export.md)。

<details>
<summary><strong>生态治理（面向维护者）</strong></summary>

### Open-Core 边界

| 资产 | 授权 |
| --- | --- |
| 框架全链（训练/评测/导出/流/参考服务） | Apache-2.0，PyPI 发布 |
| 场景评分引擎、领域提示词、SOP 清单 | Commercial（场景仓展示，代码私有） |
| `hi-ai-vision` 平台 | 商业闭源 |

### 许可红线

1. 场景引擎只依赖 PyPI 公开包，禁止 import 未发布代码。
2. `ultralytics`（AGPL-3.0）仅经框架 extra 隔离加载，不进核心依赖。
3. 场景仓引用框架文档用链接，不复制粘贴。
4. "AutoVision" 名称按 "Powered by" 形式使用，商业背书需维护者同意。

### 实施阶段

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 互链 | 徽章/回链/topics/README 小节 | 已完成（2026-10-19） |
| P1 试点 | ai-traffic 五项改造（契约化/车道/断流/权重/导出） | 进行中 |
| P2 上浮 | 确认窗状态机、台账治理进框架（0.4.0） | 待启动 |
| P3 推广 | 其余场景仓按接缝接入 | 待启动 |
| P4 引流 | 行业方案页 SEO 与案例数据回填 | 持续 |

详细接缝契约与验收标准由框架维护者按 `docs/lts_policy.md` 与
`CHANGELOG.md` 同步管理。
</details>
