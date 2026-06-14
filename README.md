# 中文欺诈文本检测系统 — ChiFraud 实验平台

> 大数据原理与技术 — 期末项目  
> 基于 ChiFraud 数据集的 10 分类欺诈检测 | 8 个传统模型 + 8 个 Transformer 模型 + 跨范式集成

---

## 项目概述

本平台对中文短信/社交文本进行 **10 分类** 欺诈检测（1 类正常 + 9 类欺诈），包含三大模型体系：

| 体系 | 模型数 | 核心方法 |
|:---|:---:|:---|
| **Baseline（5 个）** | 5 | Word2Vec/Doc2Vec + LR/GBDT、GAS(GCN) — 静态词/字嵌入 + 传统分类器 |
| **SOTA 字符 N-gram（8 个）** | 8 | 字符 1~5 gram + LinearSVC/SGD/LR — 绕过分词，直接字符 n-gram 特征 |
| **Transformer 微调（8 个变体 × 2 epoch）** | 16 | MacBERT / RoBERTa 全词遮罩预训练 + GPU 微调 |
| **★ 跨范式集成** | 2 | ensemble_auto（仅 Transformer）+ ensemble_cross（Transformer + 字符 N-gram） |

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
├── run_sota.py                        ← ★ 字符 N-gram SOTA（8 组配置）
├── run_transformer_sota.py            ← ★ Transformer 微调（MacBERT/RoBERTa）
├── run_ensemble_sota.py               ← ★ 跨范式多模型加权集成
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
    ├── ensemble_results.csv           │   集成指标汇总
    ├── baselines_summary.csv          │   Baseline 对比汇总
    └── binary_classification_summary.csv│ 二分类转换结果
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
python run_all.py --ensemble --ens-auto-tune --ens-name ensemble_cross  # 自动调优权重
python run_all.py --ensemble --ens-name ensemble_cross                  # 加载已保存配置

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

#### Baseline（5 个，CPU 可跑）
```bash
python run_baselines.py                                    # 全部 5 个，各 30k 子集
python run_baselines.py --models w2v_c gas                 # 指定模型
python run_baselines.py --full                             # 全量训练集
python run_baselines.py --adv                              # 含对抗评估
python run_baselines.py --load                             # 加载已保存模型直接评估
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
python run_transformer_sota.py --train-with-val --epochs 2 --save-predictions --adv

# 不同训练方式自动保存到不同目录
python run_transformer_sota.py --epochs 2 --class-weight balanced --loss-type focal
python run_transformer_sota.py --train-with-val --epochs 2 --class-weight balanced
python run_transformer_sota.py --train-with-val --epochs 2 --sampler-weight-power 0.5
python run_transformer_sota.py --train-with-val --epochs 2 --augment-minority 2

# RoBERTa 变体
python run_transformer_sota.py --model-name hfl/chinese-roberta-wwm-ext --run-name roberta_base --train-with-val --epochs 2

# 加载已保存模型评估
python run_transformer_sota.py --load --run-name macbert_base_+val --save-predictions --adv
```

#### 跨范式模型集成
```bash
# 自动发现模型 + 调优权重（调优后自动保存配置到 saved_models/）
python run_ensemble_sota.py --discover --auto-tune --save-predictions --name ensemble_cross

# 加载已保存配置直接评估（无需重新调优）
python run_ensemble_sota.py --load-config --name ensemble_cross --save-predictions

# 含对抗评估
python run_ensemble_sota.py --load-config --name ensemble_cross --save-predictions --adv

# 自定义调优目标
python run_ensemble_sota.py --discover --auto-tune --objective balanced_pr --name ensemble_balanced
python run_ensemble_sota.py --discover --auto-tune --objective f1_macro --name ensemble_f1
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

### test_metrics.csv / metrics.csv（指标汇总）

| 指标 | 说明 |
|:---|:---|
| `accuracy` | 总体准确率 |
| `precision_macro` | 宏平均 Precision |
| `recall_macro` | 宏平均 Recall |
| `f1_macro` | 宏平均 F1 |
| `f1_weighted` | 加权 F1 |
| `recall@90`, `precision@90`, `f1@90`, `coverage@90` | 置信度 ≥ 90分位数阈值时的指标 |
| `recall@95`, `precision@95`, `f1@95`, `coverage@95` | 置信度 ≥ 95分位数阈值时的指标 |
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

传统方案用 jieba 分词：`"办假证"` 可能被切成 `"办/假/证"` 三个独立词，丢失完整模式。字符 N-gram 直接保留 `"办假证"` 作为 3-gram 特征，不需要分词决策。词表从 3 万提升到 30~40 万，信息量提升 10 倍以上。

### 为什么 MacBERT 优于标准 BERT

MacBERT 使用全词遮罩（Whole Word Masking）+ 纠错预训练，对中文短语的语义理解更好，对形近字/同音字有天然鲁棒性。

### 为什么跨范式集成优于单模型

**ensemble_cross** 集成 22 个模型（8 个字符 N-gram SVC/SGD/LR + 14 个 MacBERT/RoBERTa epoch checkpoints），跨越了传统特征工程和深度学习的范式鸿沟。两种范式错误模式互补——字符 N-gram 对局部黑话变体敏感，Transformer 对全局语义理解强。per-class 校正因子大幅提升少数类的决策权重，配合 auto-tune 自动搜索最优权重配置。

### 置信度阈值指标的意义

`f1@90`/`f1@95` 表示：只保留模型最自信的 90%/95% 样本时的性能。高 F1@90 + 高 Coverage@90 意味着模型不仅能预测准，而且知道哪些预测不确定（可以拒绝回答），这对实际部署至关重要。

---

## 当前实验结果

评估集为 `dataset/ChiFraud_t2023.csv`（114,546 条），对抗集为 `dataset/adversarial_test.csv`（14,800 条）。

### 十分类核心指标对比

| 模型 | Accuracy | F1-macro | F1-weighted | Recall-macro | Precision-macro | F1@90 | F1@95 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | | | | | | | |
| Word2Vec-w + LR | 0.8339 | 0.4089 | 0.8572 | 0.5929 | 0.3624 | 0.6002 | 0.4382 |
| Word2Vec-c + LR | 0.8546 | 0.4707 | 0.8706 | 0.6797 | 0.4105 | 0.8625 | 0.8557 |
| Word2Vec-c + GBDT | 0.9237 | 0.6262 | 0.9160 | 0.5787 | 0.7038 | 0.6508 | 0.8333 |
| Doc2Vec-c + GBDT | 0.8581 | 0.3516 | 0.8419 | 0.2924 | 0.5110 | 0.1836 | 0.1420 |
| GAS (GCN) | 0.9221 | 0.6879 | 0.9126 | 0.6321 | 0.8155 | 0.1392 | 0.1392 |
| **字符 N-gram SOTA** | | | | | | | |
| char13_svc_120k | 0.9232 | 0.7383 | 0.9110 | 0.6664 | 0.8880 | 0.8225 | 0.7707 |
| char14_svc_160k | 0.9242 | 0.7412 | 0.9125 | 0.6651 | 0.8979 | 0.8174 | 0.7729 |
| char15_svc_c1 | 0.9290 | 0.7508 | 0.9194 | 0.6736 | 0.9021 | 0.8218 | 0.7901 |
| char15_svc_c2 | 0.9253 | 0.7373 | 0.9141 | 0.6591 | 0.9041 | 0.8111 | 0.7745 |
| char25_svc_c1 | 0.9195 | 0.7049 | 0.9062 | 0.6253 | 0.8915 | 0.7943 | 0.7439 |
| hash_char14_sgd_log | 0.9160 | 0.7002 | 0.9015 | 0.6083 | 0.9192 | 0.6810 | 0.6926 |
| **char15_sgd_log ★** | **0.9404** | **0.7870** | **0.9345** | **0.7122** | **0.9057** | **0.7634** | **0.8172** |
| char15_lr_saga | 0.9294 | 0.6981 | 0.9292 | 0.7419 | 0.7366 | 0.8073 | 0.7582 |
| **Transformer 微调** | | | | | | | |
| macbert_base_+val (ep1) | 0.9198 | 0.7238 | 0.9016 | 0.6684 | 0.8757 | 0.8129 | 0.7748 |
| macbert_base_+val_cwbalanced (ep1) | 0.9289 | 0.7595 | 0.9183 | 0.6908 | 0.8919 | 0.8187 | 0.8015 |
| macbert_base_+val_cwnone_sp0.5 (ep1) | 0.9375 | 0.7594 | 0.9306 | 0.7481 | 0.8304 | 0.8442 | 0.8122 |
| roberta_base_+val (ep1) | 0.9325 | 0.7621 | 0.9240 | 0.7428 | 0.8298 | 0.8615 | 0.8198 |
| **★ 跨范式集成 ★** | | | | | | | |
| ensemble_auto | 0.9577 | 0.8504 | 0.9581 | 0.8243 | 0.8843 | 0.9303 | 0.9101 |
| **ensemble_cross ★** | **0.9712** | **0.8680** | **0.9711** | **0.8430** | **0.9028** | **0.9389** | **0.9297** |

### 逐类别 F1 对比（集成 vs Baseline 最佳）

| 类别 | GAS (最佳Baseline) | char15_sgd_log | ensemble_auto | ensemble_cross | 最大提升 |
|:---|---:|---:|---:|---:|:---:|
| 0-正常 | 0.9592 | 0.9678 | 0.9789 | **0.9868** | +2.9% |
| 1-赌博博彩 | 0.6313 | 0.6510 | 0.7381 | **0.8571** | +35.8% |
| 2-招嫖色情 | 0.7574 | 0.8265 | 0.9315 | **0.9387** | +23.9% |
| 3-办假证 | 0.5614 | 0.6301 | 0.7495 | **0.7782** | +38.6% |
| 4-虚假办卡 | 0.6720 | 0.7023 | 0.7978 | **0.8132** | +21.0% |
| 5-违禁药品 | 0.5038 | 0.7555 | 0.8258 | **0.8452** | +67.8% |
| 6-违规提现 | 0.7017 | 0.8431 | 0.8938 | **0.8949** | +27.5% |
| 7-虚假证明 | 0.8150 | 0.8504 | 0.8800 | **0.8912** | +9.3% |
| 8-虚假手机卡 | 0.5864 | 0.9037 | 0.9101 | **0.9068** | +54.6% |
| 9-地下黑贷 | 0.6909 | 0.7399 | 0.7989 | **0.7684** | +15.6% |

### 二分类指标对比（Normal vs Spam）

| 模型 | Binary Acc | Binary Prec | Binary Rec | Binary F1 | Binary AUC |
|:---|---:|---:|---:|---:|---:|
| Word2Vec-w+LR | 0.8730 | 0.5798 | 0.8022 | 0.6731 | 0.8763 |
| Word2Vec-c+LR | 0.8789 | 0.5979 | 0.7850 | 0.6788 | 0.9019 |
| Word2Vec-c+GBDT | 0.9386 | 0.9182 | 0.6841 | 0.7840 | 0.9749 |
| Doc2Vec-c+GBDT | 0.8781 | 0.6981 | 0.4441 | 0.5429 | 0.8648 |
| GAS (GCN) | 0.9292 | 0.9513 | 0.5964 | 0.7331 | 0.9022 |
| char15_sgd_log | 0.9444 | 0.9734 | 0.6775 | 0.7989 | 0.9885 |
| char15_lr_saga | 0.9573 | 0.8599 | 0.8816 | 0.8706 | 0.9819 |
| ensemble_auto | 0.9647 | 0.8878 | 0.8965 | 0.8921 | 0.9875 |
| **ensemble_cross ★** | **0.9779** | **0.9277** | **0.9374** | **0.9325** | 0.8971 |

### 对抗鲁棒性对比（十分类 Accuracy）

| 模型 | 干净测试 Acc | 对抗测试 Acc | 鲁棒性 |
|:---|---:|---:|---:|
| char13_svc_120k | 0.9232 | 0.5725 | 62.0% |
| char14_svc_160k | 0.9242 | 0.5718 | 61.9% |
| char15_svc_c1 | 0.9290 | 0.5769 | 62.1% |
| char15_svc_c2 | 0.9253 | 0.5634 | 60.9% |
| char25_svc_c1 | 0.9195 | 0.5274 | 57.4% |
| hash_char14_sgd_log | 0.9160 | 0.5013 | 54.7% |
| char15_sgd_log | 0.9404 | 0.6116 | 65.0% |
| **char15_lr_saga ★** | **0.9294** | **0.6916** | **74.4%** |

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

## 作者

大数据原理与技术 — 期末项目
