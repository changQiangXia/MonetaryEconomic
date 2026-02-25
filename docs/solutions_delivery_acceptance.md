# `solutions.docx` 任务交付与验收清单

## 1) 原任务要点（从 `solutions.docx` 提炼）
1. 目标：为每次沟通事件生成两个指数：`Imp`（货币政策）、`Ieo`（经济形势）。  
2. Step1：将长句切为短句，并给每个短句打标签。  
3. 特征提取：使用 `jieba` 分词，抽取连续 N-gram（2 到 37 字），并输出短语分布 Excel。  
4. Step2：对短语做分类（货币政策类 / 经济形势类），并给倾向标签。  
5. Step3：统计短语在各标签中的出现概率，概率 > 50% 的短语进入最终词典。  
6. Step4：基于词典按“概率 × 次数”计算每次沟通事件的倾向概率，再得到 `Imp` 与 `Ieo`。  

## 2) 已完成内容（逐条对照）
1. 事件级双指数已完成  
证据：`outputs/event_indices.xlsx`（含 `Imp`、`Ieo` 及各倾向概率列）。

2. 短句切分与短句标签已完成  
证据：`outputs/sentence_labels.xlsx`  
说明：包含 2702 条短句（`sentence_labels` sheet），含短句主题标签、政策倾向标签、经济倾向标签。

3. 分词与 2~37 字短语提取已完成  
证据：`outputs/phrase_distribution.xlsx`（`phrases_all`、`length_summary`、`length_bins`）。

4. 机器学习短语标签已完成  
证据：`outputs/ml_labeling_report.xlsx`  
说明：包含短语主题模型、倾向模型输出，以及 `model_metrics`（4 个模型均可用）。

5. 概率筛选词典（>50%）已完成  
证据：`outputs/realtime_dictionary.xlsx`  
说明：词典条目均满足 `tendency_prob > 0.5`。

6. 参数实验与图表化解释已完成  
证据：`outputs/experiments/experiment_summary.xlsx` 与 `outputs/experiments/figures/*.png`。

## 3) 如何验收（最省时）
### A. 一键验收（推荐）
在项目根目录执行：

```bash
python -m src.validate_delivery
```

通过标准：输出 `[OK] 所有验收项通过`。

### B. 手工验收（抽查）
1. 打开 `outputs/sentence_labels.xlsx`：确认有 `sentence_labels` 与 `summary` 两个 sheet。  
2. 打开 `outputs/ml_labeling_report.xlsx`：确认有 `phrase_theme_ml`、`phrase_tendency_ml`、`model_metrics`。  
3. 打开 `outputs/realtime_dictionary.xlsx`：确认 `tendency_prob` 全部 > 0.5。  
4. 打开 `outputs/event_indices.xlsx`：确认存在 `Imp`、`Ieo`。  
5. 打开 `outputs/experiments/figures/`：确认 4 张 PNG 图都存在。  

## 4) 当前交付版本关键结果（本次实跑）
- 事件数：221  
- 短句数：2702  
- 词典条数：1527（货币政策 584，经济形势 943）  
- 最优实验配置：`0.50 / 2 / 3 / 0.30`（`threshold/support/min_dict_chars/max_event_coverage`）  

## 5) 复现实跑命令
```bash
python -m src.run_pipeline
python -m src.run_experiments --save-details
python -m src.validate_delivery
```

## 6) 后续增量优化（在Word文档中提出方法的基础上）
本节为增量优化构思，目的不是替代当前产物，而是在保持既有结果可验收的前提下，新增更强解释力和鲁棒性。

### 6.1 不改变现有成果的约束
1. 不修改当前主流程命令：python -m src.run_pipeline。
2. 不修改当前交付输出：outputs/*.xlsx 与 outputs/experiments/* 继续保留。
3. 所有优化试验输出写入 outputs_opt/，并通过新脚本触发。

### 6.2 可落地优化包（建议顺序）
1. 可解释增强包
- 事件贡献分解：输出每个事件的 top-k 驱动短语及贡献值。
- 词典稳定性：输出词条跨年份波动指标和异常词清单。
- 句子置信度：输出每条短句标签的置信度分位数。

2. 建模增强包
- 分层主题-倾向建模，降低跨主题误判。
- 多任务句子模型（主题+倾向联合学习），规则结果作为兜底。
- 年份滚动评估，跟踪词义漂移与性能衰减。

3. 鲁棒性增强包
- 指数 Bootstrap 区间估计。
- 近优参数带报告（而非单点最优）。
- 对外部宏观指标的一致性诊断图。

### 6.3 新增验收维度（在不影响现有验收基础上）
1. 新输出目录是否与 outputs/ 完全隔离。
2. 新增解释文件是否可追溯到事件/短句/短语三级。
3. 近优参数带是否与当前最优参数结论一致。
4. 运行新增优化脚本后，src.validate_delivery 仍然通过。

## 7) 本轮增强交付（不覆盖原交付）
本轮新增输出全部位于 `outputs_opt/`，原交付 `outputs/` 未覆盖。

### 7.1 新增可交付物
1. 增强版主流程输出：`outputs_opt/events_clean.xlsx`、`outputs_opt/phrase_distribution.xlsx`、`outputs_opt/realtime_dictionary.xlsx`、`outputs_opt/event_indices.xlsx`、`outputs_opt/sentence_labels.xlsx`、`outputs_opt/ml_labeling_report.xlsx`。  
2. 增强版实验输出：`outputs_opt/experiments/experiment_summary.xlsx`、`outputs_opt/experiments/experiment_summary.csv`、`outputs_opt/experiments/figures/*.png`。  
3. 前后对比输出：`outputs_opt/comparison/before_after_summary.xlsx`、`outputs_opt/comparison/figures/*.png`。  

### 7.2 本轮增强最优参数（实跑）
- `dict_threshold=0.50`
- `min_support=2`
- `min_dict_chars=3`
- `max_event_coverage=0.30`
- `dict_prob_smoothing_alpha=0.20`
- `dict_ml_blend_weight=0.35`
- `use_ml_prior=1`

对应最优 run：`run_006_thr0.50_sup2_len3_cov0.30_ml1_mg0.00_sa0.20_bw0.35`（见 `outputs_opt/experiments/experiment_summary.xlsx`）。

### 7.3 优化前后关键指标（Top1）
| metric | before (`outputs/experiments`) | after (`outputs_opt/experiments`) | delta |
|---|---:|---:|---:|
| `score_primary` | 0.766742 | 0.783494 | +0.016752 |
| `policy_label_match_rate` | 0.764706 | 0.778281 | +0.013575 |
| `policy_event_active_ratio` | 0.895928 | 0.927602 | +0.031674 |
| `econ_event_active_ratio` | 0.841629 | 0.846154 | +0.004525 |
| `dict_total_size` | 1585 | 1749 | +164 |

结论：增强版在不改动原交付文件的前提下，提升了匹配率与激活率。

### 7.4 图表证据
核心指标前后对比：  
![Before After Core Metrics](../outputs_opt/comparison/figures/before_after_core_metrics.png)

词典规模与主评分前后对比：  
![Before After Size Score](../outputs_opt/comparison/figures/before_after_size_score.png)

Top-K 得分曲线对比：  
![Before After TopK](../outputs_opt/comparison/figures/before_after_topk_score_curve.png)

最佳配置月度 Imp/Ieo 对比：  
![Before After Imp Ieo](../outputs_opt/comparison/figures/before_after_imp_ieo_monthly_overlay.png)

## 8) 验收步骤（新增增强版）
1. 先验收原交付（`solutions.docx` 对应）：`python -m src.validate_delivery`。  
2. 再看增强交付：检查 `outputs_opt/experiments/experiment_summary.xlsx` 与 `outputs_opt/comparison/before_after_summary.xlsx`。  
3. 打开 `outputs_opt/comparison/figures/`，核对 4 张对比图是否齐全。  
4. 若你只看一个总览文件，优先看：`outputs_opt/comparison/before_after_summary.xlsx` 的 `metric_comparison` sheet。  
