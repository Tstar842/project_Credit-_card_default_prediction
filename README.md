# project_Credit-_card_default_prediction

项目简介：
这个项目是一个面向信贷风控场景的违约风险预测项目，目标是基于客户历史行为、信用特征或其他匿名化特征，判断客户未来发生违约的概率，并进一步将模型结果转化为可供审批流程使用的风险等级、审批建议和监控指标。项目最初以 notebook 形式完成了数据清洗、缺失值处理、特征筛选、多模型训练、模型评估、阈值优化和稳定性分析；随后又抽离出工程化 Python 模块，使模型能够以更规范、更可复用的方式完成训练和批量评分。

从业务角度看，本项目解决的是典型的二分类信贷风险问题。数据中的 y 字段代表客户是否违约，其中 y = 0 表示未违约或好客户，y = 1 表示违约或坏客户；user_id 用于标识客户；其余 x_001 到 x_199 等字段是经过匿名化处理的建模特征。模型的核心任务是学习这些特征与违约标签之间的关系，并对新客户输出违约概率。违约概率越高，表示客户风险越高，后续可据此进行自动通过、人工复核或拒绝等审批动作。

项目原始 notebook 主要承担实验复现和模型探索功能。它首先读取原始样本数据，对缺失率较高的字段进行剔除，对部分中高缺失率字段使用随机森林回归模型进行填补，并对剩余缺失值进行数值填充。随后，项目从清洗后的数据中筛选出最终入模字段，并采用训练集和测试集划分方式进行模型验证。考虑到信贷违约样本通常少于正常样本，项目还引入了 SMOTETomek 方法对训练集进行类别不平衡处理，以提高模型对坏客户的识别能力。

在模型层面，项目对比了多种常见机器学习算法，包括逻辑回归、L1 正则逻辑回归、决策树、随机森林、梯度提升树、HistGradientBoosting、LightGBM 和 XGBoost。模型评估不只关注准确率，还综合使用了 Precision、Recall、F1、AUC、AP、KS 和 PSI 等指标。其中，AUC 用于衡量模型整体排序能力，KS 是风控建模中常用的区分度指标，AP 对不平衡样本下的模型表现更敏感，PSI 则用于观察训练集和测试集之间的分数分布稳定性。

项目进一步进行了阈值优化。传统二分类模型通常默认使用 0.5 作为分类阈值，但在信贷风控场景中，这个阈值并不一定符合业务目标。项目原始流程中已经加入了基于 KS 的阈值搜索，用于寻找更适合区分好坏客户的决策阈值。在工程化版本中，阈值优化被进一步扩展为业务成本矩阵驱动的策略选择，不再只追求统计指标最优，而是可以根据放款给好客户的收益、放过坏客户的损失、错拒好客户的机会成本等业务假设，选择预期收益更高或风险约束下通过率更合理的阈值。

根据关键逻辑，该项目被拆分为多个 Python 模块。config.py 用于管理模型配置、业务成本矩阵、风险等级规则和监控阈值；preprocessing.py 负责无数据泄漏的前处理流程，确保缺失值规则、分箱规则和字段筛选规则只在训练集上学习，并被一致地应用到测试集和未来评分数据；business.py 负责业务阈值搜索、风险分映射和审批输出；modeling.py 负责模型训练、评估、模型包封装和批量评分；monitoring.py 负责 PSI、特征漂移、缺失率变化、特征重要性和原因码生成；train_pipeline.py 和 score_batch.py 分别作为训练入口和批量评分入口。

此项目采用“先划分训练集和测试集，再只在训练集上学习前处理规则”的方式，保证测试集和未来评分数据只调用已经固化的规则。这使得模型评估结果更接近真实上线后的表现，也使未来批量评分时的处理口径与训练阶段保持一致。

项目最终以更接近信贷审批实际需要的形式输出。训练完成后，系统会生成完整的模型包 model_package.pkl，其中包含前处理规则、训练好的模型、业务阈值、评分等级规则、监控基准和特征解释信息。批量评分时，系统可以对新客户数据输出 approval_decision_output.csv，其中包含客户 ID、预测违约概率、风险分、风险等级、审批建议、阈值判断结果和原因码。这使模型输出能够更直接地服务于审批流程，而不仅仅停留在实验指标层面。

此外，项目还引入了监控能力。通过 monitoring_report.csv，用户可以观察新评分批次与训练样本之间是否存在明显分布漂移，例如模型分数 PSI、特征 PSI、缺失率变化等。如果监控结果出现 warning 或 alert，说明当前评分数据可能与训练数据存在差异，需要进一步排查数据口径、客群变化或模型失效风险。这为模型上线后的持续维护和风险管理提供了基础。

整体来看，本项目从一个信贷违约预测的建模复现实验，逐步扩展为一个具备工程化雏形的风控评分系统。它覆盖了从原始数据处理、模型训练、多模型评估、业务阈值优化，到批量评分、审批建议输出、原因码解释和稳定性监控的完整链路。项目当前仍属于轻量级工程化版本，适合作为信贷风控模型原型、课程项目、研究复现或内部验证工具；如果后续继续推进，可以进一步增加模型校准、SHAP 解释、API 服务、定时评分任务、模型版本管理和线上监控看板，从而演进为更完整的生产级风控建模系统。

项目说明：本文档说明如何使用当前项目中的工程化 Python 版本完成模型训练、批量评分、审批结果输出和监控报告生成。

工程化代码位于：

```text
risk_engineering/
```

该目录不依赖原 notebook 执行。原有的 `credit_default_repro.ipynb` 可以继续用于实验复现和可视化分析，但工程化训练与评分建议使用本文档中的命令。

---

## 1. 项目目录结构

推荐的项目结构如下：

```text
credit_default_repro/
  sample.csv
  new_apply_batch.csv
  risk_engineering/
    __init__.py
    config.py
    preprocessing.py
    business.py
    modeling.py
    monitoring.py
    train_pipeline.py
    score_batch.py
    usage_notes.py
  USAGE.md
```

其中：

```text
sample.csv
```

是训练数据。

```text
new_apply_batch.csv
```

是未来需要评分的新客户数据，文件名可以自定义。

```text
risk_engineering/
```

是工程化代码目录。

---

## 2. 运行环境要求

建议使用 Python 3.10 或更高版本。

工程化代码主要依赖：

```text
pandas
numpy
scikit-learn
joblib
imbalanced-learn
lightgbm
xgboost
```

其中：

| 依赖 | 用途 |
|---|---|
| `pandas` | 读取 CSV、处理表格数据 |
| `numpy` | 数值计算 |
| `scikit-learn` | 机器学习模型、指标、数据划分 |
| `joblib` | 保存和加载模型包 |
| `imbalanced-learn` | `SMOTETomek` 类别不平衡处理 |
| `lightgbm` | LightGBM 模型 |
| `xgboost` | XGBoost 模型 |

如果 `lightgbm` 或 `xgboost` 没有安装，工程代码会跳过对应模型，但建议完整安装，方便复现多模型对比结果。

---

## 3. 数据应该放在哪里

最简单的方式是把数据放在项目根目录，也就是和 `risk_engineering/` 同级的位置。

例如：

```text
D:\jupyter_files\FinTech\credit_default_repro\
  sample.csv
  new_apply_batch.csv
  risk_engineering\
```

然后在项目根目录运行命令：

```powershell
cd D:\jupyter_files\FinTech\credit_default_repro
```

训练时：

```powershell
python -m risk_engineering.train_pipeline --input sample.csv --output-dir risk_engineering_outputs
```

评分时：

```powershell
python -m risk_engineering.score_batch --input new_apply_batch.csv --model-package risk_engineering_outputs/model_package.pkl --output-dir risk_engineering_scoring_outputs
```

数据也可以放在其他目录，只要通过 `--input` 参数指定正确路径即可。

例如：

```powershell
python -m risk_engineering.train_pipeline --input D:\data\sample.csv --output-dir D:\model_outputs\train
```

---

## 4. 训练数据要求

训练数据必须是 CSV 文件。

训练数据必须包含：

```text
user_id
y
x_001, x_002, x_003, ...
```

### 4.1 字段说明

| 字段 | 是否必须 | 含义 |
|---|---|---|
| `user_id` | 是 | 客户 ID 或样本 ID |
| `y` | 是 | 违约标签 |
| `x_***` | 是 | 模型特征 |

其中 `y` 必须是二分类标签：

| y | 含义 |
|---:|---|
| `0` | 好客户 / 未违约 |
| `1` | 坏客户 / 违约 |

### 4.2 训练数据示例

```csv
user_id,y,x_001,x_002,x_003,x_004,x_005,x_006
A00002,0,0,32,0,0,0,0
A00005,0,0,29,0,0,0,0
A00006,0,0,31,0,0,0,0
A00010,1,1,45,0,2,1,0
```

实际数据中会有更多特征列。

### 4.3 默认入模字段

工程化版本默认沿用原 notebook 中筛选出的最终字段。

默认最终字段共 57 列：

```text
user_id, y,
x_001, x_002, x_003, x_004, x_005, x_006,
x_019, x_020, x_021, x_027,
x_033, x_034, x_035, x_036, x_037, x_038,
x_041, x_042, x_044, x_045, x_048, x_049,
x_052, x_054, x_055, x_056,
x_074, x_075, x_077, x_078,
x_088, x_089,
x_121, x_122, x_124, x_125,
x_131, x_132, x_134, x_137,
x_142, x_143, x_144, x_149,
x_154, x_155, x_157, x_159, x_162,
x_188, x_189, x_190,
x_196, x_197, x_198
```

其中 `user_id` 和 `y` 不是模型特征，真正进入模型的是其余 55 个 `x_***` 字段。

### 4.4 数据可以包含额外字段吗

可以。

例如原始 `sample.csv` 中可能包含 `x_001` 到 `x_199` 的更多字段。工程化代码会根据配置中的默认字段选择需要的列。

额外字段不会影响默认训练。

### 4.5 数据可以有缺失值吗

可以。

工程化流程会处理缺失值：

1. 只在训练集上学习缺失处理规则；
2. 对高缺失字段按训练集规则处理；
3. 对中高缺失字段使用随机森林填补；
4. 对普通缺失值使用训练集上的中位数填补；
5. 将同一套规则应用到测试集和未来评分数据。

这样可以避免测试集或未来数据参与训练阶段规则学习，从而降低数据泄漏风险。

### 4.6 字段类型要求

模型特征建议为数值型。

如果某些 `x_***` 字段是字符串、类别文本或包含无法解析的字符，可能导致训练失败或效果异常。建议在输入工程化 pipeline 前，将特征处理为数值型。

---

## 5. 评分数据要求

评分数据是未来要预测违约风险的新客户数据。

评分数据必须是 CSV 文件。

评分数据通常必须包含：

```text
user_id
x_001, x_002, x_003, ...
```

评分数据通常不需要包含 `y`，因为新客户在评分时还没有真实违约标签。

### 5.1 评分数据示例

```csv
user_id,x_001,x_002,x_003,x_004,x_005,x_006
B00001,0,28,0,0,0,0
B00002,1,36,0,1,0,0
B00003,0,41,1,0,1,0
```

### 5.2 评分数据可以包含 y 吗

可以。

如果评分数据包含 `y`，输出文件中会包含 `y_true`，适合做回测、验证或模型效果复盘。

如果评分数据不包含 `y`，程序会正常输出风险概率、风险等级和审批建议。

### 5.3 评分数据缺少部分特征怎么办

当前代码会尝试使用训练阶段保存的中位数对缺失字段进行补齐。

也就是说，少量缺字段时，程序仍可能运行。

但是从业务可靠性角度，不建议长期依赖自动补齐缺字段。更推荐保证训练数据和评分数据的字段口径一致。

---

## 6. 第一次训练模型

进入项目根目录：

```powershell
cd D:\jupyter_files\FinTech\credit_default_repro
```

运行训练命令：

```powershell
python -m risk_engineering.train_pipeline --input sample.csv --output-dir risk_engineering_outputs
```

参数说明：

| 参数 | 示例 | 含义 |
|---|---|---|
| `--input` | `sample.csv` | 训练数据路径 |
| `--output-dir` | `risk_engineering_outputs` | 训练结果输出目录 |

训练完成后，会生成：

```text
risk_engineering_outputs/
  model_package.pkl
  metrics_summary_engineered.csv
  business_threshold_optimized_metrics.csv
  approval_decision_output.csv
  global_feature_importance.csv
  monitoring_report.csv
  test_predictions.csv
```

---

## 7. 训练输出文件说明

### 7.1 model_package.pkl

```text
model_package.pkl
```

这是最重要的文件。

它是完整模型包，包含：

| 内容 | 说明 |
|---|---|
| 前处理规则 | 删除列、缺失填补、分箱边界 |
| 训练好的模型 | 最终选中的模型 |
| 业务阈值 | 按业务成本矩阵选择的阈值 |
| 评分等级规则 | A/B/C/D/E 风险等级 |
| 监控基准 | 训练集分布、分数分布 |
| 特征重要性 | 用于解释和原因码 |

后续批量评分必须使用这个文件。

### 7.2 metrics_summary_engineered.csv

传统模型指标汇总表。

常见字段：

| 字段 | 含义 |
|---|---|
| `model` | 模型名称 |
| `Accuracy` | 准确率 |
| `Precision` | 精确率 |
| `Recall` | 召回率 |
| `F1` | F1 值 |
| `AUC` | ROC AUC |
| `AP` | Average Precision |
| `KS` | KS 指标 |
| `PSI` | 训练集和测试集分数稳定性 |

### 7.3 business_threshold_optimized_metrics.csv

业务阈值优化结果。

当前工程化版本不只看 KS，而是引入业务成本矩阵。

常见字段：

| 字段 | 含义 |
|---|---|
| `model` | 模型名称 |
| `threshold` | 业务阈值 |
| `approval_rate` | 通过率 |
| `reject_rate` | 拒绝率 |
| `bad_approval_rate` | 通过客户中的坏客户比例 |
| `expected_profit` | 预期总收益 |
| `expected_profit_per_user` | 人均预期收益 |
| `Recall` | 坏客户识别召回率 |
| `Precision` | 拒绝客户中的坏客户比例 |
| `TP` | 坏客户被拒绝数量 |
| `FP` | 好客户被拒绝数量 |
| `TN` | 好客户被通过数量 |
| `FN` | 坏客户被通过数量 |

### 7.4 approval_decision_output.csv

测试集上的审批输出样例。

这张表的格式与未来批量评分输出一致，可以直接作为审批策略对接的参考。

常见字段：

| 字段 | 含义 |
|---|---|
| `user_id` | 客户 ID |
| `y_true` | 真实标签，如果存在 |
| `pd_score` | 预测违约概率，越高风险越高 |
| `risk_score` | 风险分，通常越高越安全 |
| `risk_grade` | 风险等级，如 A/B/C/D/E |
| `decision` | 按等级规则给出的审批建议 |
| `threshold_decision` | 按业务阈值给出的 approve/reject |
| `threshold` | 当前使用的阈值 |
| `model_name` | 使用的模型 |
| `reason_code_1` | 原因码 1 |
| `reason_code_2` | 原因码 2 |
| `reason_code_3` | 原因码 3 |

### 7.5 global_feature_importance.csv

全局特征重要性。

常见字段：

| 字段 | 含义 |
|---|---|
| `model` | 模型名称 |
| `feature` | 特征名 |
| `importance` | 重要性 |
| `rank` | 排名 |

### 7.6 monitoring_report.csv

训练后生成的测试集监控报告，用于观察测试集与训练集之间是否存在明显漂移。

常见字段：

| 字段 | 含义 |
|---|---|
| `metric_type` | 监控类型 |
| `metric_name` | 监控对象 |
| `value` | 监控值 |
| `status` | 状态 |
| `threshold` | 告警阈值 |

`status` 可能为：

| 状态 | 含义 |
|---|---|
| `stable` | 稳定 |
| `warning` | 有轻微漂移，需要观察 |
| `alert` | 明显漂移，需要排查 |

---

## 8. 对新客户数据批量评分

假设已经训练完成，并且存在：

```text
risk_engineering_outputs/model_package.pkl
```

假设新客户数据为：

```text
new_apply_batch.csv
```

运行：

```powershell
python -m risk_engineering.score_batch --input new_apply_batch.csv --model-package risk_engineering_outputs/model_package.pkl --output-dir risk_engineering_scoring_outputs
```

参数说明：

| 参数 | 示例 | 含义 |
|---|---|---|
| `--input` | `new_apply_batch.csv` | 待评分数据 |
| `--model-package` | `risk_engineering_outputs/model_package.pkl` | 训练好的模型包 |
| `--output-dir` | `risk_engineering_scoring_outputs` | 评分结果输出目录 |

评分完成后会生成：

```text
risk_engineering_scoring_outputs/
  approval_decision_output.csv
  monitoring_report.csv
```

---

## 9. 批量评分输出说明

### 9.1 approval_decision_output.csv

这是批量评分的核心输出。

示例结构：

```csv
user_id,pd_score,risk_score,risk_grade,decision,threshold_decision,threshold,model_name,reason_code_1,reason_code_2,reason_code_3
B00001,0.0832,721,A,auto_approve,approve,0.42,RandomForest,x_041 <= train_P25,x_088 <= train_P25,x_121 >= train_P75
B00002,0.2765,612,C,manual_review,approve,0.42,RandomForest,x_154 >= train_P75,x_021 >= train_P75,x_188 <= train_P25
B00003,0.6811,503,E,reject,reject,0.42,RandomForest,x_044 >= train_P75,x_089 >= train_P75,x_196 <= train_P25
```

字段解释：

| 字段 | 含义 |
|---|---|
| `user_id` | 客户 ID |
| `pd_score` | 预测违约概率 |
| `risk_score` | 风险分 |
| `risk_grade` | 风险等级 |
| `decision` | 风险等级对应的审批建议 |
| `threshold_decision` | 阈值判断结果 |
| `threshold` | 当前模型使用的业务阈值 |
| `model_name` | 当前使用的模型 |
| `reason_code_1/2/3` | 简单原因码 |

### 9.2 decision 与 threshold_decision 的区别

`decision` 来自风险等级规则。

例如：

| 风险等级 | decision |
|---|---|
| A | `auto_approve` |
| B | `auto_approve` |
| C | `manual_review` |
| D | `manual_review` |
| E | `reject` |

`threshold_decision` 来自业务最优阈值。

例如：

```text
pd_score >= threshold -> reject
pd_score < threshold  -> approve
```

如果需要接入真实审批流程，通常建议以 `decision` 作为三段式策略：

```text
auto_approve
manual_review
reject
```

而 `threshold_decision` 可以作为二分类风控参考。

---

## 10. 业务成本矩阵在哪里配置

业务成本矩阵位于：

```text
risk_engineering/config.py
```

对应类：

```python
BusinessConfig
```

默认配置类似：

```python
profit_good_approved = 1000.0
loss_bad_approved = 5000.0
opportunity_loss_good_rejected = 300.0
benefit_bad_rejected = 0.0
min_recall = 0.55
min_approval_rate = 0.30
max_approval_rate = 0.85
max_bad_approval_rate = 0.20
optimize_mode = "max_profit"
```

含义：

| 参数 | 含义 |
|---|---|
| `profit_good_approved` | 放款给好客户的收益 |
| `loss_bad_approved` | 放款给坏客户的损失 |
| `opportunity_loss_good_rejected` | 错拒好客户的机会损失 |
| `benefit_bad_rejected` | 拒绝坏客户的收益或避免损失 |
| `min_recall` | 最低坏客户召回率 |
| `min_approval_rate` | 最低通过率 |
| `max_approval_rate` | 最高通过率 |
| `max_bad_approval_rate` | 通过客户中坏客户比例上限 |
| `optimize_mode` | 阈值优化方式 |

当前支持两种阈值优化模式：

| optimize_mode | 含义 |
|---|---|
| `max_profit` | 在约束条件下最大化预期收益 |
| `max_approval_under_risk` | 在风险约束下最大化通过率 |

---

## 11. 风险等级规则在哪里配置

风险等级规则位于：

```text
risk_engineering/config.py
```

对应类：

```python
ScoreConfig
```

默认按预测违约概率分层：

| 风险等级 | 概率区间 | 审批建议 |
|---|---|---|
| A | `[0.00, 0.10)` | `auto_approve` |
| B | `[0.10, 0.20)` | `auto_approve` |
| C | `[0.20, 0.35)` | `manual_review` |
| D | `[0.35, 0.50)` | `manual_review` |
| E | `[0.50, 1.00]` | `reject` |

可以根据业务策略调整区间。

例如，如果风控更保守，可以把拒绝阈值提前：

```python
{"grade": "E", "min_prob": 0.40, "max_prob": 1.01, "decision": "reject"}
```

---

## 12. 监控规则在哪里配置

监控规则位于：

```text
risk_engineering/config.py
```

对应类：

```python
MonitoringConfig
```

默认配置：

```python
psi_stable = 0.10
psi_warning = 0.25
bins = 10
missing_rate_warning_delta = 0.10
```

一般解释：

| PSI | 状态 | 含义 |
|---:|---|---|
| `< 0.10` | `stable` | 分布稳定 |
| `0.10 - 0.25` | `warning` | 有一定漂移，需要观察 |
| `>= 0.25` | `alert` | 明显漂移，需要排查或重训 |

---

## 13. 迁移到其他文件夹如何使用

如果你把工程化代码移动到别的地方，只要保持 `risk_engineering/` 是一个完整 Python 包即可。

推荐结构：

```text
D:\new_project\
  sample.csv
  new_apply_batch.csv
  risk_engineering\
    __init__.py
    config.py
    preprocessing.py
    business.py
    modeling.py
    monitoring.py
    train_pipeline.py
    score_batch.py
```

然后在 `D:\new_project` 下运行：

```powershell
cd D:\new_project
```

训练：

```powershell
python -m risk_engineering.train_pipeline --input sample.csv --output-dir risk_engineering_outputs
```

评分：

```powershell
python -m risk_engineering.score_batch --input new_apply_batch.csv --model-package risk_engineering_outputs/model_package.pkl --output-dir risk_engineering_scoring_outputs
```

关键点：

1. 当前运行目录下要能找到 `risk_engineering/`；
2. `--input` 指向训练或评分 CSV；
3. `--model-package` 指向训练阶段生成的 `model_package.pkl`；
4. `--output-dir` 指向结果输出目录。

---

## 14. 推荐的数据目录结构

如果后续数据越来越多，建议使用更清晰的目录结构：

```text
credit_risk_project/
  data/
    train/
      sample.csv
    score/
      new_apply_batch_202604.csv
  outputs/
    train/
    score/
  risk_engineering/
```

训练命令：

```powershell
cd D:\credit_risk_project

python -m risk_engineering.train_pipeline --input data/train/sample.csv --output-dir outputs/train
```

评分命令：

```powershell
python -m risk_engineering.score_batch --input data/score/new_apply_batch_202604.csv --model-package outputs/train/model_package.pkl --output-dir outputs/score
```

这样数据、代码、输出会更加清晰。

---

## 15. 常见问题

### 15.1 是否必须运行 notebook

不需要。

工程化版本可以直接通过 `.py` 文件训练和评分。

notebook 仍然可以用于展示、画图、复现实验过程，但不是工程化流程的必要步骤。

### 15.2 是否必须把数据放进 risk_engineering 文件夹

不需要，也不建议。

`risk_engineering/` 是代码目录，数据建议放在它的同级目录或单独的 `data/` 目录中。

推荐：

```text
project/
  sample.csv
  risk_engineering/
```

或者：

```text
project/
  data/
    sample.csv
  risk_engineering/
```

### 15.3 为什么用 python -m 运行

因为 `risk_engineering` 是一个 Python 包。

使用：

```powershell
python -m risk_engineering.train_pipeline
```

可以保证包内的相对导入正常工作。

不推荐直接运行：

```powershell
python risk_engineering/train_pipeline.py
```

因为直接运行单个文件时，包内相对导入可能失败。

### 15.4 训练很慢怎么办

当前默认会训练多个模型：

```text
LogisticRegression
LogisticRegression_L1
DecisionTree
RandomForest
GradientBoosting
HistGradientBoosting
LightGBM
XGBoost
```

如果后续希望加快训练，可以在 `risk_engineering/modeling.py` 的 `build_models` 函数中暂时减少模型数量。

例如只保留：

```text
RandomForest
XGBoost
LightGBM
```

### 15.5 评分时报字段缺失怎么办

先检查评分数据是否包含 `user_id` 和主要 `x_***` 特征。

当前代码会对部分缺失字段使用训练阶段中位数补齐，但如果关键字段大面积缺失，模型结果会不可靠。

建议对评分数据做字段检查：

```text
字段名是否一致
是否多了空格
是否大小写不同
是否把 x_001 写成了 x001
是否把数值列读成了文本
```

### 15.6 评分输出中的 pd_score 是什么

`pd_score` 是预测违约概率。

数值越高，表示模型认为该客户违约风险越高。

例如：

| pd_score | 含义 |
|---:|---|
| `0.05` | 低风险 |
| `0.25` | 中等风险 |
| `0.65` | 高风险 |

### 15.7 risk_score 是什么

`risk_score` 是由违约概率映射出的风险分。

通常：

```text
risk_score 越高，风险越低
risk_score 越低，风险越高
```

### 15.8 reason_code 是否等同于 SHAP 解释

不是。

当前 `reason_code_1/2/3` 是轻量原因码，基于全局特征重要性和样本特征相对训练集分位数生成。

它适合作为初版审批解释字段，但不是严格的 SHAP 局部解释。

如果未来需要更强解释性，可以加入 SHAP。

### 15.9 什么情况下需要重新训练

建议在以下情况考虑重新训练：

1. `monitoring_report.csv` 中大量特征出现 `alert`；
2. 模型分数 PSI 明显超过 `0.25`；
3. 新客户客群和训练样本客群发生明显变化；
4. 业务策略、产品利率、授信政策发生变化；
5. 积累了足够多的新标签数据；
6. 模型线上坏账率明显偏离预期。

---

## 16. 完整流程示例

### 16.1 项目结构

```text
D:\credit_risk_project\
  data\
    sample.csv
    new_apply_batch.csv
  risk_engineering\
  outputs\
```

### 16.2 训练

```powershell
cd D:\credit_risk_project

python -m risk_engineering.train_pipeline --input data/sample.csv --output-dir outputs/train
```

训练完成后检查：

```text
outputs/train/model_package.pkl
outputs/train/metrics_summary_engineered.csv
outputs/train/business_threshold_optimized_metrics.csv
```

### 16.3 评分

```powershell
python -m risk_engineering.score_batch --input data/new_apply_batch.csv --model-package outputs/train/model_package.pkl --output-dir outputs/score
```

评分完成后查看：

```text
outputs/score/approval_decision_output.csv
outputs/score/monitoring_report.csv
```

### 16.4 审批使用

业务审批可以重点使用：

```text
approval_decision_output.csv
```

重点字段：

```text
user_id
pd_score
risk_score
risk_grade
decision
threshold_decision
reason_code_1
reason_code_2
reason_code_3
```

建议审批策略：

| decision | 处理方式 |
|---|---|
| `auto_approve` | 自动通过 |
| `manual_review` | 人工复核 |
| `reject` | 拒绝 |

---

## 17. 最简命令速查

在项目根目录训练：

```powershell
python -m risk_engineering.train_pipeline --input sample.csv --output-dir risk_engineering_outputs
```

在项目根目录评分：

```powershell
python -m risk_engineering.score_batch --input new_apply_batch.csv --model-package risk_engineering_outputs/model_package.pkl --output-dir risk_engineering_scoring_outputs
```

迁移到新目录后，只要目录结构类似：

```text
new_project/
  sample.csv
  risk_engineering/
```

就可以在 `new_project/` 下运行：

```powershell
python -m risk_engineering.train_pipeline --input sample.csv --output-dir risk_engineering_outputs
```

