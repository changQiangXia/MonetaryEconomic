# 参数推荐（基于网格实验）

## 推荐默认参数
- `dict_threshold = 0.50`
- `min_support = 2`
- `min_dict_chars = 3`
- `max_dict_event_coverage = 0.30`

对应命令：

```bash
python -m src.run_pipeline \
  --dict-threshold 0.50 \
  --min-support 2 \
  --min-dict-chars 3 \
  --max-dict-event-coverage 0.30
```

## 依据
来自 `outputs/experiments/experiment_summary.xlsx` 当前实验结果（36组）：

- 该组在 `policy_label_match_rate`、`policy_event_active_ratio`、`econ_event_active_ratio` 的综合表现最优。
- 与更宽松设置相比，词典规模已显著收缩（1585 条），同时保持较高事件激活率。

关键指标（推荐组）：
- `policy_dict_size = 627`
- `econ_dict_size = 958`
- `dict_total_size = 1585`
- `policy_event_active_ratio = 0.8959`
- `econ_event_active_ratio = 0.8416`
- `policy_label_match_rate = 0.7647`

## 说明
- 当前版本已启用“词典质量增强”：
  - 人名/来源词黑名单过滤；
  - 主题词命中门槛；
  - 短语边界与噪声规则过滤；
  - 覆盖率上限过滤（过度高频低信息短语会被剔除）。
- 如果后续你补充人工标注，可在此基础上进一步把 `dict_threshold` 提到 `0.55` 做高置信词典版本。
