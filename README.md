# 社交机器人检测稳定性实验代码与结果整理包

本目录为本次 **固定 8 条推文单视图社交机器人检测稳定性研究**（TwiBot-20 `train` + `dev`，官方 `test` 集全程未打开、未评分）的完整代码、配置、统计脚本与结果文档归档包。

> 注：本目录为只读整理副本，原始可运行工作区保留在 `D:\大数据分析\work\antigravity_robust_mh5\`，未被移动或修改。

---

## 目录结构说明

### `01_core_experiments/` — 核心训练与缓存代码
- `build_cache_k4.py`：提取 MiniLM 最后 4 层输入（`layer[-4]`）并缓存至 `cache_k4/`（仅处理 `train` 与 `dev`，`test_opened=False`，逐 shard 验证 fp16 重放余弦相似度 `> 0.99999`）。
- `run_robust_exp.py`：`K=1` 架构与多子包训练实现，包含：
  - `RobustSharedHeads`（标准多头、`SetAttn` 熵正则集合注意力、`ResGate` 自洽残差门控）
  - `sample_subbags_from_pool`（同账号 16 条推文零额外编码开销抽取 `S=8` 个 8-tweet 子包）
  - `subbag_objective`（`SubBag-ProbVar` 边界概率方差损失、`CVaR` 消融项、`anti-collapse` 约束）
- `run_robust_k4.py`：`K=4` 局部解冻训练与推理主程序（`LastKBlock` 封装最后 4 层 MiniLM，支持 `mh5_kl` 对照组与 `subbag_probvar`）。

### `02_configs/` — 全量实验配置文件
- **核心 `K=4` 组**：
  - `k4_mh5_kl01.yaml`：`K4-MH5-KL01`（`K=4` 纯对照组，`AP=0.92805`）
  - `k4_mh5_subbag.yaml`：`K4-MH5-SubBag`（`K=4` + `SubBag-ProbVar`，最高 `AP=0.92986`，`SpVar -42.3%`）
  - `k4_mh5_setattn.yaml`：`K4-MH5-SetAttn`（`K=4` + `encoder_lr=1.2e-5` + `SetAttn`，压低 `TrVar` 至 `0.2457`）
- **`K=1` 消融组**：
  - `mh5_subbag_probvar.yaml` / `mh5_subbag_clean.yaml` / `mh5_setattn_clean.yaml` / `mh5_setattn_subbag.yaml` / `mh5_resgate_subbag.yaml`
- **`base/`**：被上述配置继承的统一超参基座（`common.yaml`、`partial_unfreeze_kl.yaml`）。

### `03_evaluation_and_stats/` — 统一评估、方差分解与统计检验
- `analyze_robust.py`：在 4 种阈值校准模式（`cal_v0_botf1`、`cal_matched_human_fp`、`dev_iso_human_stable_fp`、`dev_iso_human_mean_fp`）下统一评估 12 组模型并执行联合筛选。
- `bootstrap_compare.py`：向量化配对账号簇 Bootstrap（`B=1000`, Holm 多重检验校正），数值与 `statistical_analysis.py` 逐小数一致。
- `common_eval.py`、`evaluate_stability.py`、`decompose_variance.py`、`statistical_analysis.py`、`experiment_common.py`、`run_experiment.py`：复用的项目标准评测基座（指标计算、交叉随机效应方差分解与基线脚本）。

### `04_results_and_docs/` — 实验结果表与理论证明文档
- `robust_summary.csv`：全部 12 组模型 × 4 种校准模式的汇总指标表（可直接用 Excel/Python 画图）。
- `robust_analysis.json`：完整评估结果与逐 seed 联合筛选明细（`test_scored: false`）。
- `bootstrap_summary.json`：`K=4` 核心模型 vs `EXP-0` / `EXP-8-MH5-KL` 的 95% 置信区间与 Holm 校正 p 值。
- `robust_ablation_report.md`：完整消融分析报告。
- `classifier_control_properties.docx`（及 `.md`）：分类器控制与稳定性理论证明（含原生 Word 公式，对应 PPT 定理 1 & 定理 2）。

---

## 复现命令参考（在 `D:\大数据分析` 根目录执行）

```powershell
$PY = "D:\local_experiments\twibot22_local_repro\twibot22_local_repro\.conda\python.exe"

# 1. 构建 K=4 缓存（仅 train + dev，不碰 test）
& $PY -s -B work\antigravity_robust_mh5\build_cache_k4.py

# 2. 运行 K=4 全量 5-seed × 30-view 实验
& $PY -s -B work\antigravity_robust_mh5\run_robust_k4.py --config work\antigravity_robust_mh5\configs\k4_mh5_kl01.yaml --stage full
& $PY -s -B work\antigravity_robust_mh5\run_robust_k4.py --config work\antigravity_robust_mh5\configs\k4_mh5_subbag.yaml --stage full
& $PY -s -B work\antigravity_robust_mh5\run_robust_k4.py --config work\antigravity_robust_mh5\configs\k4_mh5_setattn.yaml --stage full

# 3. 统一多校准评估与配对账号簇 Bootstrap（B=1000）
& $PY -s -B work\antigravity_robust_mh5\analyze_robust.py
& $PY -s -B work\antigravity_robust_mh5\bootstrap_compare.py
```
