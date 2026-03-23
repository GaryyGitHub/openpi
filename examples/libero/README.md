# LIBERO Benchmark
## 多维鲁棒性评估框架（扩展版）

本仓库提供了一个受 **LIBERO-PROLIBERO-Plus** 启发的**多维扰动框架**，用于系统地评测机器人模型在多个维度上的鲁棒性，**仅需使用单一预训练checkpoint（无需重新训练）**。

### 快速开始

#### 30秒快速启动

```bash
# 1. 生成所有扰动（3-5 分钟）
uv run scripts/build_libero_multiperturbation.py

# 2. 运行演示
python demo_multiperturbation.py

# 3. 输出位置
ls -lh examples/libero/demo_output/
ls -lh third_party/libero/libero/libero/bddl_files_multiperturbation/libero_spatial/
```

---

### 1. 语义扰动类型与具体方式

- `none`：不做任何改写，使用原始指令。
- `paraphrase`（同义改写）：
	- 按任务结构做改写（如 `pick-and-place`、`put`、`open/close/turn on/off`、`stack`）；
	- 采用模板 + 同义词池生成，不是固定前后缀；
	- 同一条指令会稳定生成同一改写（确定性），不同任务通常得到不同表达。
- `constraint`（约束增强）：
	- 在保持原目标不变的前提下，增加过程约束；
	- 约束会尽量引用当前任务中的主对象（例如"仅与目标物体交互""避免移动其他物体"）。
- `reference`（指代/上下文增强）：
	- 引入场景上下文描述（如"在当前布局下"）；
	- 结合主对象或目标关系，形成更自然的指代式表达。
- `noisy`（噪声口语化）：
	- 增加礼貌词、提醒词等与任务无关但常见的自然语言噪声；
	- 保持核心动作语义不变。

### 2. 位置(Position)扰动类型与具体方式

位置扰动通过修改BDDL中的 `(:regions ...)` 部分，改变任务的空间布局，评估机器人对空间变化的鲁棒性。

#### 实现方式

修改目标：通过BDDL的 `(:regions ...)` 块中的 `(:ranges ...)` 坐标范围，对整组任务区域应用统一的 `shift` 平移变换：

- `shift`（平移）：
  - 将每个region沿X/Y轴整体平移（当前实现为 ±10% 工作空间）
  - 保留每个region的形状和大小，只改变其在桌面平面上的位置
  - 同一任务内的多个region会一起平移，从而整体重排场景布局
  - 典型用途：评估机器人对工作台布局重新安排的鲁棒性

**开发者注意：** 所有坐标修改都保证不会产生碰撞（与固定物体的interactions维持原有语义）。确定性选择使用SHA256哈希，同一任务总是产生相同的扰动。

示例修改：
```
原始 (:ranges ((0.05 0.19 0.07 0.21)))
平移 (:ranges ((0.15 0.29 0.07 0.21)))  # x所有坐标+0.10
```

### 3. 物体(Object)扰动类型与具体方式

物体扰动通过修改BDDL中的 `(:objects ...)` 部分和 `(:init ...)` 初始条件，改变场景中的物体构成。

#### 实现方式

修改目标：BDDL中的 `(:objects ...)` 列表和 `(:init ...)` 初始化条件。

- `add_distractor`（添加干扰物体）：
	- 在 `(:objects ...)` 中添加一个额外的同类干扰物体（非目标物体）
	- 在 `(:init ...)` 中为新对象添加初始位置约束（放在不影响任务的位置）
	- 例如：如果任务是"把黑碗放到盘子上"，可添加第三个碗作为干扰项
	- 典型用途：评估机器人在多物体场景中的注意力与目标物体识别能力

	**开发者注意：** 干扰物体选择使用确定性哈希，同一任务总是添加相同的干扰物体。

示例修改：
```
原始 objects:  
  akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl  
  cookies_1 - cookies  
  plate_1 - plate  

添加干扰:  
  akita_black_bowl_1 akita_black_bowl_2 akita_black_bowl_3 - akita_black_bowl  
  cookies_1 - cookies  
  plate_1 - plate    
```

### 4. 目标(Goal)扰动类型与具体方式

目标扰动通过修改BDDL中的 `(:goal ...)` 部分，改变机器人需要达成的终态条件。

#### 实现方式

修改目标：BDDL中的 `(:goal ...)` 区块。

- `change_target`（改变目标表面）：
	- 修改目标对象的最终支撑面，例如从"放在盘子上"改为"放在餐桌中心"
	- 从 `(On akita_black_bowl_1 plate_1)` 改为 `(On akita_black_bowl_1 main_table_center)`
	- 可选的目标表面: `plate_1`, `main_table`, `main_table_center`, `wooden_cabinet_1`, `flat_stove_1` 等
	- 保持涉及的对象不变，仅改变目标位置
	- 典型用途：评估机器人的目标表示泛化（对同一物体放在不同位置的理解）

- `add_constraint`（增加约束）：
	- 在目标中添加额外的约束（如顺序约束、避免约束等）
	- 当前版本支持：目标顺序提示（多个对象时的处理顺序）
	- 例如：保持原有 `(On bowl plate)` 但隐含"先开抓手再放下"
	- 注意：显式的时序约束需要扩展BDDL语法，当前版本主要记录意图
	- 典型用途：评估机器人对复杂多步任务的理解

**开发者注意：** 目标修改保证不改变任务的本质（同一物体、同一交互类型）。改变的仅是支撑表面或约束顺序。使用确定性哈希选择替代目标，同一任务总是相同修改。

示例修改：
```
原始 goal: (And (On akita_black_bowl_1 plate_1))
改变目标: (And (On akita_black_bowl_1 main_table_center))
增加约束: (And (On akita_black_bowl_1 plate_1) (Holding_before_release))  # 隐含约束
```

---
### 生成逻辑说明

#### 单一扰动维度

- 优先使用对应的 `*_variants.json` manifest 中已存在的改写结果（如 `instruction_variants.json` 用于语义扰动）；
- 若某项在 manifest 中没有对应变体，则回退到规则改写；
- BDDL 修改方式因扰动维度而异：
	- **语义扰动**：仅修改 `(:language ...)` 行
	- **位置扰动**：仅修改 `(:regions ...)` 中的坐标范围
	- **物体扰动**：修改 `(:objects ...)` 和 `(:init ...)` 中的物体列表与初始条件
	- **目标扰动**：仅修改 `(:goal ...)` 条件
- 其他BDDL字段（`:fixtures`, `:obj_of_interest`等）保持不变，确保任务本质不变。

#### 多维扰动组合

多个扰动维度可以**独立组合**使用：
- 位置 + 物体 + 目标：评估三维鲁棒性
- 语义 + 位置：评估语言理解 + 空间适应能力
- 等等

每个维度的扰动类型通过命令行参数独立指定，生成的BDDL文件以 `{task_name}__{perturbation_type}_{variant_name}.bddl` 命名。

#### 可选 LLM 精修阶段

- 当前仅语义扰动支持 Gemini 精修
- 位置、物体、目标扰动为结构型 BDDL 改写，保持确定性生成
- LLM 调用失败或验证未通过时，自动回退规则结果

---

### 两阶段改写逻辑（语义扰动详细）

#### 第一阶段：规则改写（`semantic_utils.py`）

- 输入：原始任务指令（来自 BDDL 的 `(:language ...)`）。
- 结构识别：先用正则识别任务模式，例如：
	- `pick-and-place`
	- `put`
	- `stack`
	- `open/close/turn on/turn off`
- 变体生成：按 `paraphraseconstraintreferencenoisy` 使用不同模板与同义词池。
- 确定性：同一条原始指令总是得到同一条规则改写（使用哈希做稳定选择），便于复现实验。

#### 第二阶段：LLM 精修（`gemini_rewriter.py`）

- 启用条件：
	- `--enable-llm-stage`
	- 且存在 `GEMINI_API_KEY`（或 `--gemini-api-key`）。
- 调用方式：
	- 请求地址：`{api_base}/v1beta/models/{model}:generateContent?key=...`
	- 默认推荐：`https://api2.xcodecli.com/`
	- 若配置了 `https://api.xcodecli.com/` 且遇到 `403/1010`，会自动回退到 `api2`。
- Prompt 约束（核心思想）：
	- 仅输出一行英文改写；
	- 保持原任务目标与语义不变；
	- 保留关键对象与空间关系；
	- 禁止引入新子目标、否定翻转或额外歧义；
	- 输入同时包含 `Original instruction` 和 `Stage-1 rewrite`。

#### 结果判定与回退

- LLM 返回后会经过保真检查（关键词覆盖等）。
- 通过：`variant_source = "llm"`。
- 不通过或请求失败：回退第一阶段结果，并标记如：
	- `rule(api_error)`：接口调用失败
	- `rule(rejected)`：保真检查未通过
	- `rule(empty)`：返回为空

#### 如何读取 manifest 结果

- 关注 `metadata.llm_stage.status_counts`：查看 `accepted/api_error/rejected/...`。
- 关注 `metadata.source_counts` 与 `metadata.variant_source_counts`：看各变体最终有多少来自 LLM。
- 关注每个任务的 `variant_source`：定位具体哪条指令走了回退。

示例（你当前一次运行）：

- `accepted: 156`
- `api_error: 4`
- 说明：160 条变体里有 4 条请求失败，已自动回退到规则结果，整体流程仍可完成。

### 使用方法

#### A. 生成单一维度的扰动

##### 1. 语义扰动（Semantic Perturbation）

仅规则改写（默认，推荐）：

```bash
uv run scripts/build_libero_semantic_bddl.py
```

生成产物：
- `third_party/libero/libero/libero/bddl_files_semantic/*`
- `libero_data/libero_semantic_enhanced/instruction_variants.json`

带LLM精修（可选）：

```bash
export GEMINI_API_KEY="your_api_key"
export GEMINI_API_BASE="https://api2.xcodecli.com/"
uv run scripts/build_libero_semantic_bddl.py \
	--enable-llm-stage \
	--gemini-model gemini-2.5-flash
```

使用 `build_libero_multiperturbation.py` 兼容替代旧语义脚本（推荐迁移方式）：

```bash
uv run scripts/build_libero_multiperturbation.py \
  --perturbation-types semantic \
  --semantic-variants paraphrase constraint reference noisy \
  --output-root third_party/libero/libero/libero/bddl_files_semantic \
  --legacy-semantic-manifest-out libero_data/libero_semantic_enhanced/instruction_variants.json
```

兼容行为：
- 会同时生成 `task__paraphrase.bddl`（旧命名）和 `task__semantic_paraphrase.bddl`（新命名）；
- 会写出带 `by_instruction` 的 manifest，可直接被现有 `semantic_utils.load_semantic_manifest` 读取；
- 因此在该兼容模式下，可逐步替换 `build_libero_semantic_bddl.py`。

##### 2. 多维扰动（Position + Object + Goal）

一次生成所有维度的扰动：

```bash
uv run scripts/build_libero_multiperturbation.py \
	--perturbation-types semantic position object goal \
	--suites libero_spatial libero_object libero_goal libero_10
```

可选参数：

```bash
uv run scripts/build_libero_multiperturbation.py \
	--perturbation-types semantic position object goal \
	--suites libero_spatial \
	--enable-llm-stage \
	--gemini-api-key "your_api_key" \
	--gemini-api-base "https://api2.xcodecli.com/" \
	--gemini-model gemini-2.5-flash \
	--log-progress-every 10 \
	--log-verbose
```

生成产物：
- `third_party/libero/libero/libero/bddl_files_multiperturbation/{suite}/{task_name}__{perturb_type}_{variant}.bddl`
- `libero_data/libero_multiperturbation_variants.json`（包含所有维度的统计元数据）

#### B. 在评测中使用扰动

##### 语义扰动评测

```bash
python examples/libero/main.py \
  --perturbation-type semantic \
  --perturbation-variant paraphrase \
	--semantic-variant paraphrase \
	--semantic-manifest libero_data/libero_semantic_enhanced/instruction_variants.json \
	--bddl-root-override third_party/libero/libero/libero/bddl_files_semantic
```

常用 `--perturbation-variant` 值：`none``paraphrase``constraint``reference``noisy`

兼容性说明：`--semantic-variant` 仍然保留，旧命令可以继续使用；新实现优先读取 `--perturbation-variant`。

##### 其他维度评测

现在 positionobjectgoal 也可以直接通过同一个评测入口切换，无需手动拼接 BDDL 文件名。

```bash
# 评测位置平移扰动
python examples/libero/main.py \
  --perturbation-type position \
  --perturbation-variant shift \
  --bddl-root-override third_party/libero/libero/libero/bddl_files_multiperturbation

# 评测物体干扰扰动
python examples/libero/main.py \
  --perturbation-type object \
  --perturbation-variant add_distractor \
  --bddl-root-override third_party/libero/libero/libero/bddl_files_multiperturbation

# 评测目标变动扰动
python examples/libero/main.py \
  --perturbation-type goal \
  --perturbation-variant change_target \
  --bddl-root-override third_party/libero/libero/libero/bddl_files_multiperturbation
```

对应关系如下：

- `--perturbation-type semantic` 对应 `__paraphrase.bddl` 或 `__semantic_paraphrase.bddl`
- `--perturbation-type position` 对应 `__position_shift.bddl`
- `--perturbation-type goal` 对应 `__goal_change_target.bddl``__goal_add_constraint.bddl`

如果你想在自定义脚本中动态选择BDDL文件，也可以直接按文件名筛选：

```python
import random
from pathlib import Path

bddl_root = Path('/path/to/bddl_files_multiperturbation/libero_spatial')
position_bddls = list(bddl_root.glob('*__position_shift.bddl'))

# 随机选择一个位置扰动的BDDL文件
selected = random.choice(position_bddls)
```

#### C. 数据转换（可选）

在训练数据转换中写入改写后的任务指令：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py \
	--data_dir /path/to/your/data \
	--semantic_variant constraint \
	--semantic_manifest libero_data/libero_semantic_enhanced/instruction_variants.json
```

---

### 实际使用指南与最佳实践

#### 完整工作流程示例

完整的扰动生成与评测流程：

```bash
# 第1步：生成所有维度的扰动（语义 + 位置 + 物体 + 目标）
uv run scripts/build_libero_multiperturbation.py \
  --perturbation-types semantic position object goal \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --log-progress-every 5 \
  --log-verbose

# 第2步：选择单一维度进行评测
# 评测位置扰动 (shift)
for bddl_file in third_party/libero/libero/libero/bddl_files_multiperturbation/libero_spatial/*__position_shift.bddl; do
  echo "Testing: $bddl_file"
  # 调用你的评测脚本，传入 BDDL_FILE 环境变量或参数
done

# 第3步：在评测脚本中使用各维度扰动 
# 见下文"评测脚本集成"
```

#### 评测脚本集成示例

在你的评测脚本中加载并使用不同维度的扰动：

```python
import json
from pathlib import Path
import random

class LiberoMultiPerturbationEvaluator:
    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.bddl_root = Path('third_party/libero/libero/libero/bddl_files_multiperturbation')
    
    def get_bddl_for_variant(self, suite: str, task_name: str, perturbation_type: str, variant: str) -> str:
        """获取指定维度变体的BDDL文件内容"""
        bddl_file = self.bddl_rootsuitef"{task_name}__{perturbation_type}_{variant}.bddl"
        with open(bddl_file) as f:
            return f.read()
    
    def evaluate_task_across_perturbations(self, suite: str, task_name: str, model_eval_fn):
        """在多维扰动下评测单个任务"""
        results = {}
        
        for perturb_type in ['semantic', 'position', 'object', 'goal']:
            results[perturb_type] = {}
            
            for variant in ['none', 'variant1', 'variant2', ...]:  # 根据实际维度调整
                try:
                    bddl_content = self.get_bddl_for_variant(suite, task_name, perturb_type, variant)
                    score = model_eval_fn(bddl_content)
                    results[perturb_type][variant] = score
                except Exception as e:
                    results[perturb_type][variant] = None
        
        return results

# 使用示例
evaluator = LiberoMultiPerturbationEvaluator('libero_data/libero_multiperturbation_variants.json')

# 评测单个任务
task_results = evaluator.evaluate_task_across_perturbations(
    suite='libero_spatial',
    task_name='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate',
    model_eval_fn=my_forward_pass_function
)

# 分析健壮性 (Robustness Analysis)
def compute_robustness_metrics(results):
    """计算各维度下的鲁棒性指标"""
    metrics = {}
    
    for perturb_type, variants in results.items():
        baseline = variants.get('none')
        if baseline is None:
            continue
        
        perturbed_scores = [s for k, s in variants.items() if k != 'none' and s is not None]
        
        if perturbed_scores:
            mean_drop = baseline - sum(perturbed_scores)len(perturbed_scores)
            max_drop = baseline - min(perturbed_scores)
            metrics[perturb_type] = {
                'baseline': baseline,
                'mean_drop': mean_drop,
                'max_drop': max_drop,
                'n_variants': len(perturbed_scores),
            }
    
    return metrics
```

#### 推荐的评测策略

1. **基线（Baseline）**：所有维度均为 `none`，获取原始模型性能

2. **单维度评测**：逐个维度评测，分离各自的影响
   - Position → 空间泛化能力
   - Object → 物体识别鲁棒性
   - Goal → 多目标理解能力
   - Semantic → 语言理解泛化

3. **跨维度评测**：组合多个维度，评测复合鲁棒性
   - Position + Semantic：空间 + 语言变化
   - All dimensions：全面鲁棒性评估

4. **统计分析**：计算各维度下的性能下降（drop）和方差

#### 预期的鲁棒性表现

- **好的表现（>90% baseline）**：
  - 语义扰动（paraphrase/constraint）：模型理解稳定
  - 位置平移扰动（shift）：良好的空间泛化

- **可接受的表现（80-90% baseline）**：
  - 目标位置改变：需要理解任务目标的灵活性

- **需要改进（<80% baseline）**：
  - 多维组合扰动：模型可能过度拟合单一场景
  - 大幅位置平移（shift）：模型可能需要增加环境多样性训练

---

#### `build_libero_multiperturbation.py` 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--libero-root` | `third_party/libero/libero/libero` | LIBERO库路径 |
| `--output-root` | `bddl_files_multiperturbation` | 输出目录 |
| `--manifest-path` | `libero_data/libero_multiperturbation_variants.json` | manifest文件路径 |
| `--suites` | 所有suite | 要处理的LIBERO suite列表 |
| `--perturbation-types` | 所有类型 | 要生成的扰动维度（semantic/position/object/goal） |
| `--semantic-variants` | `paraphrase constraint reference noisy` | semantic维度要生成的变体列表 |
| `--enable-llm-stage` | False | 是否启用LLM精修阶段 |
| `--llm-variants` | 与semantic_variants相同 | 指定哪些semantic变体走LLM精修 |
| `--gemini-api-key` | 从GEMINI_API_KEY环境变量读取 | Gemini API密钥 |
| `--gemini-api-base` | `https://api2.xcodecli.com/` | Gemini API基础URL |
| `--gemini-model` | `gemini-2.5-flash` | 使用的Gemini模型 |
| `--gemini-temperature` | `0.2` | LLM温度 |
| `--gemini-max-output-tokens` | `128` | LLM最大输出token |
| `--gemini-timeout-sec` | `20` | LLM请求超时（秒） |
| `--write-semantic-legacy-filenames` | True | semantic是否额外写入旧命名文件（`__paraphrase.bddl`） |
| `--legacy-semantic-manifest-out` | `libero_data/libero_semantic_enhanced/instruction_variants.json` | 兼容旧语义流程的manifest输出路径 |
| `--log-progress-every` | 5 | 每N个任务打印进度 |
| `--log-verbose` | False | 启用详细日志 |

#### Gemini API 端点说明

- **推荐**：`https://api2.xcodecli.com/`（稳定、高可用）
- **备用**：`https://api.xcodecli.com/`（如果遇到403/1010错误，自动回退）

自动回退逻辑已内置在 `gemini_rewriter.py`。

---

### 输出文件结构

若生成了semantic、position、object、goal四种扰动，最终输出如下：

```
bddl_files_multiperturbation/
├── libero_spatial/
│   ├── pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate__semantic_paraphrase.bddl
│   ├── pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate__semantic_constraint.bddl
│   ├── ...
│   ├── pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate__position_shift.bddl
│   ├── ...
│   ├── pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate__object_add_distractor.bddl
│   ├── ...
│   ├── pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate__goal_change_target.bddl
│   └── ...
├── libero_object/
│   └── ...
├── libero_goal/
│   └── ...
└── libero_10/
    └── ...

libero_data/
└── libero_multiperturbation_variants.json
```

manifest文件结构示例：

```json
{
  "metadata": {
    "llm_stage": {
      "enabled": true,
      "model": "gemini-2.5-flash",
      "status_counts": {
        "accepted": 156,
        "api_error": 4
      }
    },
    "perturbation_types": ["semantic", "position", "object", "goal"],
    "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
  },
  "by_suite": {
    "libero_spatial": {
      "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": {
        "perturbation_counts": {
          "semantic": 4,
          "position": 3,
          "object": 2,
          "goal": 2
        },
        "total_variants": 11
      }
    }
  }
}
```
