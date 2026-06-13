# 中文欺诈文本检测系统 — ChiFraud 实验平台

> 大数据原理与技术 — 期末项目  
> 基于 ChiFraud 数据集的 10 分类欺诈检测 | 8 个模型体系 | 8 种对抗攻击策略

---

## 项目概述

本平台对中文短信/社交文本进行 **10 分类** 欺诈检测（1 类正常 + 9 类欺诈），包含三大模型体系：

| 体系 | 模型 | 核心方法 |
|:---|:---|:---|
| **论文 Baseline（5 个）** | Word2Vec/Doc2Vec + LR/GBDT、GAS(GCN) | 静态词/字嵌入 + 传统分类器 |
| **SOTA 字符 N-gram（8 个变体）** | 字符 1~5 gram + LinearSVC/SGD/LR | 绕过分词，直接字符 n-gram 特征 |
| **SOTA Transformer** | MacBERT / RoBERTa 微调 | 全词遮罩预训练 + GPU 微调 |
| **★ 模型集成** | 6 模型加权 + per-class 校正因子 | 跨范式融合，错误模式互不重叠 |

### 类别映射

| Label | 含义 | Label | 含义 |
|:-----:|------|:-----:|------|
| 0 | 正常 | 5 | 违禁药品交易 |
| 1 | 赌博博彩 | 6 | 违规提现 |
| 2 | 招嫖色情 | 7 | 虚假证明 |
| 3 | 办假证 | 8 | 虚假手机卡 |
| 4 | 虚假办卡 | 9 | 地下黑贷 |

---

## 项目结构

```
Final_hw/
│
├── README.md                          ← 本文件
├── requirements.txt                   ← Python 依赖
├── config.py                          ← 全局配置
├── data_processor.py                  ← 数据加载/清洗/分词/TF-IDF
│
├── models/                            ← 模型包
│   ├── __init__.py                    │   导出接口
│   ├── base.py                        │   BaseModel 基类
│   ├── baselines.py                   │   5 个论文 Baseline
│   └── evaluation.py                  │   评估工具(含置信度阈值指标)
│
├── run_baselines.py                   ← 5 个 Baseline 统一运行脚本
├── run_sota.py                        ← ★ 创新方法：字符 N-gram SOTA（8 组配置）
├── run_transformer_sota.py            ← ★ 创新方法：Transformer 微调（MacBERT/RoBERTa）
├── run_ensemble_sota.py               ← ★ 创新方法：多模型加权集成
├── run_all.py                         ← ★ 一键全流程运行（Python 主控）
├── run_all.ps1                        ← ★ 一键全流程运行（PowerShell）
├── run_all.sh                         ← ★ 一键全流程运行（Bash）
│
├── make_adversarial_dataset.py        ← 对抗测试集生成
├── visualize.py                       ← 可视化（混淆矩阵/模型对比/学习曲线）
├── postprocess_binary.py              ← 十分类→二分类后处理
│
├── dataset/                           ← 数据目录
│   ├── ChiFraud_train.csv             │   训练集
│   ├── ChiFraud_t2022.csv             │   验证集
│   ├── ChiFraud_t2023.csv             │   测试集
│   └── adversarial_test.csv           │   对抗测试集(脚本生成)
│
└── output/                            ← 输出目录
    ├── {model_name}/                  │   每个模型独立子目录
    │   ├── test_results.csv           │   测试集预测(text/true/pred/confidence)
    │   ├── test_metrics.csv           │   测试集指标
    │   ├── adversarial_results.csv    │   对抗集预测
    │   └── adversarial_metrics.csv    │   对抗集指标
    ├── sota_results.csv               │   字符 N-gram 指标汇总
    ├── transformer_results.csv        │   Transformer 指标汇总
    ├── ensemble_results.csv           │   集成指标汇总
    ├── baselines_summary.csv          │   Baseline 对比汇总
    ├── binary_classification_summary.csv│ 二分类转换结果
    └── figures/                       │   可视化图表彰
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据（确保 dataset/ 下有 3 个 CSV 文件）

### 3. 一键运行（推荐）

使用 `run_all.py` / `run_all.ps1` / `run_all.sh` 一次性跑完所有模型：

```bash
# ── Python 主控（功能最全，跨平台）──
python run_all.py                                           # 首次完整训练（baseline→sota→transformer→ensemble）
python run_all.py --load                                    # 仅加载已保存模型评估
python run_all.py --adv --save-predictions --ensemble       # 含对抗评估 + 预测文件 + 集成
python run_all.py --stages baseline sota                    # 只跑指定阶段
python run_all.py --dry-run                                 # 仅打印要执行的命令，不实际运行

# 并行运行（Windows/Linux 均可，注意 GPU 显存）
python run_all.py --parallel

# 自定义 Transformer 配置
python run_all.py --tf-epochs 3 --tf-class-weight balanced  # 覆盖所有 TF 配置的参数
python run_all.py --tf-configs 1 3                          # 只跑第1和第3个 TF 配置
python run_all.py --tf-list-configs                         # 列出所有 TF 配置

# 自定义 Ensemble
python run_all.py --ensemble --ens-auto-tune --ens-name my_ensemble  # 自动调优权重
python run_all.py --ensemble --ens-name ensemble_auto                # 加载已保存配置

# ── PowerShell（Windows）──
.\run_all.ps1                                               # 完整训练
.\run_all.ps1 -Load                                         # 仅评估
.\run_all.ps1 -Adv -SavePred -Ensemble                      # 对抗+预测+集成
.\run_all.ps1 -Parallel                                     # 并行运行

# ── Bash（Linux/macOS）──
bash run_all.sh                                             # 完整训练
bash run_all.sh --load                                      # 仅评估
bash run_all.sh --adv --save-pred --ensemble                # 对抗+预测+集成
bash run_all.sh --parallel                                  # 并行运行
```

### 4. 单独运行每个模型

如果需要更细粒度的控制，可以单独运行：

#### Baseline（5 个，CPU 可跑）

```bash
python run_baselines.py                                    # 全部 5 个，各 30k 子集
python run_baselines.py --models w2v_c gas                 # 指定模型
python run_baselines.py --full                             # 全量训练集
python run_baselines.py --adv                              # 含对抗评估
python run_baselines.py --load                             # 加载已保存模型直接评估
python run_baselines.py --load --models w2v_c              # 加载指定已保存模型
python run_baselines.py --adv --save-predictions           # 对抗评估 + 保存预测文件
```

#### 字符 N-gram SOTA（CPU）

```bash
python run_sota.py --experiments all                       # 全部 8 组配置
python run_sota.py --experiments char15_svc_c1             # 单个实验
python run_sota.py --experiments all --save-predictions --adv  # 保存预测 + 对抗
python run_sota.py --experiments all --train-with-val      # 使用 train+val 训练
python run_sota.py --experiments all --load                # 加载已保存模型评估
```

#### Transformer 微调（需 GPU）

```bash
# 基础训练
python run_transformer_sota.py --train-with-val --epochs 3 --save-predictions --adv

# 不同训练方式自动保存到不同目录（run_id 含关键参数）
python run_transformer_sota.py --epochs 3 --class-weight balanced --loss-type focal
python run_transformer_sota.py --train-with-val --epochs 3 --class-weight balanced
python run_transformer_sota.py --train-with-val --epochs 3 --sampler-weight-power 0.5
python run_transformer_sota.py --train-with-val --epochs 3 --augment-minority 2

# RoBERTa 变体
python run_transformer_sota.py --model-name hfl/chinese-roberta-wwm-ext --run-name roberta_base --train-with-val --epochs 3

# 不保存每 epoch checkpoint（仅最佳模型）
python run_transformer_sota.py --train-with-val --epochs 3 --no-save-every-epoch

# 加载已保存模型评估
python run_transformer_sota.py --load --run-name macbert_base_+val --save-predictions --adv
python run_transformer_sota.py --load --run-name macbert_base_+val_cwbalanced_focal1.5 --save-predictions --adv
```

#### 模型集成

```bash
# 自动发现模型 + 调优权重（调优后自动保存配置到 saved_models/）
python run_ensemble_sota.py --discover --auto-tune --save-predictions --name ensemble_auto

# 加载已保存配置直接评估（无需重新调优）
python run_ensemble_sota.py --load-config --name ensemble_auto --save-predictions

# 含对抗评估
python run_ensemble_sota.py --load-config --name ensemble_auto --save-predictions --adv

# 自定义调优目标
python run_ensemble_sota.py --discover --auto-tune --objective balanced_pr --name ensemble_balanced
python run_ensemble_sota.py --discover --auto-tune --objective f1_macro --name ensemble_f1

# 手动指定模型和权重
python run_ensemble_sota.py --models model1_test model2_test --weights 1.0 0.5 --save-predictions
```

### 5. 可视化与后处理

```bash
# ===== 可视化（混淆矩阵/模型对比/学习曲线）=====
python visualize.py

# ===== 二分类后处理（10分类→2分类）=====
python postprocess_binary.py

# ===== 生成对抗数据集 =====
python make_adversarial_dataset.py
```

---

## 输出格式说明

### test_results.csv（每样本预测）

| 列名 | 含义 |
|:---|:---|
| `text` | 原始文本（截取前 200 字） |
| `true_label` | 真实标签 (0-9) |
| `pred_label` | 预测标签 (0-9) |
| `confidence` | 预测置信度（最大类别概率） |

### test_metrics.csv（指标汇总）

| 指标 | 说明 |
|:---|:---|
| `accuracy` | 总体准确率 |
| `precision_macro` | 宏平均 Precision |
| `recall_macro` | 宏平均 Recall |
| `f1_macro` | 宏平均 F1 |
| `f1_weighted` | 加权 F1 |
| `recall@90`, `precision@90`, `f1@90`, `coverage@90` | 置信度 ≥ 90分位数阈值时的指标 |
| `recall@95`, `precision@95`, `f1@95`, `coverage@95` | 置信度 ≥ 95分位数阈值时的指标 |
| `per_class_precision_json` | 各类别 Precision（JSON） |
| `per_class_f1_json` | 各类别 F1（JSON） |
| `per_class_recall_json` | 各类别 Recall（JSON） |

---

## 8 种对抗攻击策略

| # | 策略 | 示例 |
|:--:|------|------|
| ① | 形近字替换 | 证→証, 药→薬 |
| ② | 随机删字 | 删除 10% 字符 |
| ③ | 插入无关字符 | 插入"的/了/空格" |
| ④ | 关键词掩码 | 用 \*#X 替换字符 |
| ⑤ | 正常文本伪装 | 头尾插入正常文本 |
| ⑥ | 数字全角混淆 | 5→５ |
| ⑦ | 伪装 URL 添加 | 追加虚假链接 |
| ⑧ | 拼音简写 | 微信→vx, 加→+ |

---

## 核心设计思想

### 为什么字符级 N-gram 优于词级 TF-IDF

传统方案用 jieba 分词：`"办假证"` 可能被切成 `"办/假/证"` 三个独立词，丢失完整模式。字符 N-gram 直接保留 `"办假证"` 作为 3-gram 特征，不需要分词决策。词表从 3 万提升到 40 万，信息量提升 13 倍。

### 为什么 MacBERT 优于标准 BERT

MacBERT 使用全词遮罩（Whole Word Masking）+ 纠错预训练，对中文短语的语义理解更好，对形近字/同音字有天然鲁棒性。

### 为什么集成优于单模型

被集成的 6 个模型跨越了传统特征工程（n-gram SVC/SGD）和深度学习（MacBERT/RoBERTa）的范式鸿沟，错误模式不重叠 → 投票矫正。

### 置信度阈值指标的意义

`recall@90`/`f1@90` 表示：只保留模型最自信的 90% 样本时的性能。高 `coverage@90` + 高 `f1@90` 意味着模型不仅能预测准，而且知道哪些预测不确定（可以拒绝回答）。

---

## 依赖

```
python>=3.8
pandas, numpy, scikit-learn, scipy
jieba, gensim
matplotlib, seaborn
torch, transformers
pypinyin, Pillow, tqdm
```
## 当前实验结果

评估集为 `dataset/ChiFraud_t2023.csv`。当前最佳提交候选为 `ensemble_gpu_sampler_f1`，由字符级模型、MacBERT、RoBERTa full fine-tuning、weighted-sampler RoBERTa 进行 10 分类概率集成，并启用 train+2022 的 exact text fallback。

### 置信度阈值指标

| 模型 | Recall@90 | Precision@90 | F1@90 | Coverage@90 | Recall@95 | Precision@95 | F1@95 | Coverage@95 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ensemble_gpu_sampler_f1` | 0.9304 | 0.9475 | 0.9379 | 0.9000 | 0.9220 | 0.9385 | 0.9292 | 0.9500 |

### 完整结果表

下面表格整理自 `output/ensemble_results.csv`、`output/transformer_results.csv`、`output/sota_results.csv` 中的全部标量指标列。`config_json` 和逐类 JSON 字段保留在原始 CSV 中；当前最佳模型的逐类 Precision/Recall/F1 已在下一节展开。

#### 集成模型结果

| experiment | split | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted | recall@90 | precision@90 | f1@90 | coverage@90 | recall@95 | precision@95 | f1@95 | coverage@95 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ensemble_gpu_sampler_f1 | test | 0.974386 | 0.900104 | 0.854417 | 0.874238 | 0.974288 | 0.930400 | 0.947500 | 0.937900 | 0.900000 | 0.922000 | 0.938500 | 0.929200 | 0.950000 |
| ensemble_v2_exact | test | 0.974072 | 0.900860 | 0.850811 | 0.873435 | 0.973819 | 0.927900 | 0.951900 | 0.938600 | 0.900000 | 0.918800 | 0.941700 | 0.929300 | 0.950000 |
| ensemble_recheck_default | test | 0.973993 | 0.900490 | 0.850995 | 0.873355 | 0.973745 | 0.928100 | 0.951600 | 0.938600 | 0.900000 | 0.919000 | 0.941200 | 0.929200 | 0.950000 |
| ensemble_v2_six_model | test | 0.973993 | 0.900490 | 0.850995 | 0.873355 | 0.973745 | - | - | - | - | - | - | - | - |
| ensemble_gpu_roberta_f1 | test | 0.974089 | 0.899342 | 0.852934 | 0.873138 | 0.974004 | 0.929300 | 0.946200 | 0.936700 | 0.900000 | 0.922800 | 0.938500 | 0.929700 | 0.950000 |
| ensemble_auto_sota | test | 0.973749 | 0.905813 | 0.845721 | 0.872708 | 0.973487 | 0.927900 | 0.954200 | 0.939800 | 0.900000 | 0.919500 | 0.944300 | 0.930900 | 0.950000 |
| ensemble_v1_five_model | test | 0.972360 | 0.894514 | 0.849944 | 0.869223 | 0.972144 | - | - | - | - | - | - | - | - |
| ensemble_gpu_roberta_balanced_pr | test | 0.972814 | 0.870092 | 0.869415 | 0.868575 | 0.973106 | 0.929200 | 0.922400 | 0.925300 | 0.900000 | 0.921600 | 0.912700 | 0.916600 | 0.950000 |
| ensemble_gpu_sampler_balanced_pr | test | 0.973233 | 0.870083 | 0.869221 | 0.868272 | 0.973553 | 0.932700 | 0.929700 | 0.930700 | 0.900000 | 0.924700 | 0.919900 | 0.921800 | 0.950000 |
| ensemble_auto_balanced_pr | test | 0.971226 | 0.869489 | 0.868609 | 0.867294 | 0.971706 | 0.927300 | 0.921600 | 0.923800 | 0.900000 | 0.919800 | 0.913000 | 0.915500 | 0.950000 |

#### Transformer 结果

| experiment | split | epoch | protocol | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted | recall@90 | precision@90 | f1@90 | coverage@90 | recall@95 | precision@95 | f1@95 | coverage@95 | exact_matches | n_samples | train_time_s | predict_time_s |
|:---|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macbert_smoke | val | 1 | train->val/test exact_fallback=True | 0.933000 | 0.251751 | 0.278467 | 0.262230 | 0.913462 | - | - | - | - | - | - | - | - | 8 | 1000 | 27.330016 | 1.952682 |
| macbert_smoke | test | 1 | train->val/test exact_fallback=True | 0.881000 | 0.143710 | 0.177440 | 0.157475 | 0.844079 | - | - | - | - | - | - | - | - | 11 | 1000 | 27.330016 | 2.230537 |
| macbert_50k_bal | val | 1 | train->val/test exact_fallback=True | 0.990500 | 0.927915 | 0.927997 | 0.927175 | 0.990540 | - | - | - | - | - | - | - | - | 429 | 20000 | 406.251579 | 35.018368 |
| macbert_50k_bal | test | 1 | train->val/test exact_fallback=True | 0.939850 | 0.833024 | 0.682646 | 0.719857 | 0.929658 | - | - | - | - | - | - | - | - | 662 | 20000 | 406.251579 | 35.231467 |
| macbert_50k_bal | val | 2 | train->val/test exact_fallback=True | 0.992600 | 0.937766 | 0.940045 | 0.938618 | 0.992630 | - | - | - | - | - | - | - | - | 429 | 20000 | 891.394832 | 35.287406 |
| macbert_50k_bal | test | 2 | train->val/test exact_fallback=True | 0.927500 | 0.844214 | 0.666719 | 0.721606 | 0.915124 | - | - | - | - | - | - | - | - | 662 | 20000 | 891.394832 | 35.559025 |
| macbert_50k_plus2022_bal_b32 | test | 1 | train+val->test exact_fallback=True | 0.929836 | 0.846536 | 0.706892 | 0.743735 | 0.917441 | - | - | - | - | - | - | - | - | 7141 | 114546 | 870.849809 | 194.643542 |
| macbert_full_plus2022_bal_b64 | test | 1 | train+val->test exact_fallback=True | 0.931896 | 0.850902 | 0.713083 | 0.746101 | 0.918748 | - | - | - | - | - | - | - | - | 11470 | 114546 | 1528.295210 | 194.561295 |
| macbert_full_plus2022_sqrt_b64 | test | 1 | train+val->test exact_fallback=True | 0.921027 | 0.870724 | 0.670640 | 0.723609 | 0.904717 | - | - | - | - | - | - | - | - | 11470 | 114546 | 1523.353207 | 196.001566 |
| roberta_50k_plus2022_bal_b64 | test | 1 | train+val->test exact_fallback=True | 0.946441 | 0.806375 | 0.771601 | 0.769757 | 0.940000 | - | - | - | - | - | - | - | - | 7141 | 114546 | 763.565601 | 195.223373 |
| macbert_full_plus2022_bal_len192_b48 | test | 1 | train+val->test exact_fallback=True | 0.930028 | 0.852457 | 0.703138 | 0.739729 | 0.915855 | - | - | - | - | - | - | - | - | 11470 | 114546 | 2313.742137 | 273.619119 |
| roberta_full_plus2022_bal_seed2026_b64 | test | 1 | train+val->test | 0.933300 | 0.850300 | 0.746400 | 0.771300 | 0.922100 | 0.850000 | 0.940000 | 0.884200 | 0.900000 | 0.797900 | 0.906200 | 0.828400 | 0.950000 | - | 114546 | 1374.200000 | - |
| roberta_full_plus2022_sampler05_sqrt_seed2030_b64 | test | 1 | train+val->test | 0.942900 | 0.799000 | 0.776800 | 0.765300 | 0.935500 | 0.874200 | 0.942400 | 0.894500 | 0.900000 | 0.841900 | 0.909900 | 0.859100 | 0.950000 | - | 114546 | 1375.900000 | - |

#### 字符级 SOTA 结果

| experiment | split | protocol | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted | exact_matches | n_samples | train_time_s | predict_time_s |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| char15_svc_c1 | val | train->val/test exact_fallback=True | 0.985000 | 0.962298 | 0.867547 | 0.907728 | 0.984464 | 4 | 1000 | 4.069947 | 0.774190 |
| char15_svc_c1 | test | train->val/test exact_fallback=True | 0.910000 | 0.813930 | 0.486218 | 0.543068 | 0.890171 | 7 | 1000 | 4.069947 | 0.959124 |
| char13_svc_120k | val | train->val/test exact_fallback=True | 0.993852 | 0.954981 | 0.946724 | 0.950377 | 0.993856 | 2847 | 96289 | 320.028376 | 61.096533 |
| char13_svc_120k | test | train->val/test exact_fallback=True | 0.923201 | 0.888023 | 0.666456 | 0.738306 | 0.911015 | 8674 | 114546 | 320.028376 | 70.175714 |
| char13_svc_120k | test | train+val->test exact_fallback=True | 0.923786 | 0.889166 | 0.675287 | 0.745333 | 0.911934 | 11470 | 114546 | 423.544642 | 67.678020 |
| hash_char14_sgd_log | test | train+val->test exact_fallback=True | 0.918548 | 0.920510 | 0.624021 | 0.715041 | 0.905279 | 11470 | 114546 | 68.218905 | 41.157353 |

### 当前最佳逐类指标

| Label | 类别 | Precision | Recall | F1 |
|:---:|---|---:|---:|---:|
| 0 | 正常 | 0.988441 | 0.988245 | 0.988343 |
| 1 | 赌博博彩 | 0.851997 | 0.897209 | 0.874019 |
| 2 | 招嫖色情 | 0.944776 | 0.950804 | 0.947780 |
| 3 | 办假证 | 0.880893 | 0.692008 | 0.775109 |
| 4 | 虚假办卡 | 0.875371 | 0.766234 | 0.817175 |
| 5 | 违禁药品交易 | 0.884035 | 0.843501 | 0.863293 |
| 6 | 违规提现 | 0.900901 | 0.920810 | 0.910747 |
| 7 | 虚假证明 | 0.901468 | 0.907173 | 0.904311 |
| 8 | 虚假手机卡 | 0.884273 | 0.898944 | 0.891548 |
| 9 | 地下黑贷 | 0.888889 | 0.679245 | 0.770053 |

## 作者

大数据原理与技术 — 期末项目
