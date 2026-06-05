# 中文欺诈文本检测系统

> 大数据原理与技术 — 期末项目  
> 基于 ChiFraud 数据集的垃圾/欺诈文本多分类检测与对抗鲁棒性评估

---

## 项目概述

对中文短信/社交文本进行 **11 分类** 欺诈检测（正常 + 10 类欺诈），实现 **11 种模型**（5 个传统 ML、5 个论文 Baseline、1 个创新模型），并设计 **7 种中文对抗攻击策略** 进行鲁棒性评估。

### 类别映射

| Label | 含义 | Label | 含义 |
|:-----:|------|:-----:|------|
| 0 | 正常 | 6 | 违规提现 |
| 1 | 赌博博彩 | 7 | 虚假证明 |
| 2 | 招嫖色情 | 8 | 虚假手机卡 |
| 3 | 办假证 | 9 | 地下黑贷 |
| 4 | 虚假办卡 | 10 | 新类型（训练集未见） |
| 5 | 违禁药品交易 | | |

### 数据集

| 数据集 | 文件 | 用途 |
|--------|------|------|
| 训练集 | `ChiFraud_train.csv` | 模型训练 |
| 验证集(2022) | `ChiFraud_t2022.csv` | 超参调优 |
| 测试集(2023) | `ChiFraud_t2023.csv` | 最终评估（包含标签10"新类型"） |
| 对抗测试集 | `adversarial_test.csv` | 对抗鲁棒性评估（脚本生成） |

---

## 项目结构

```
Final_hw/
│
├── README.md                    ← 本文件
├── requirements.txt             ← Python 依赖清单
├── config.py                    ← 全局配置（路径、超参、类别映射）
│
├── data_processor.py            ← 数据层：加载 / 清洗 / jieba分词 / TF-IDF向量化
│
├── models/                      ← ★ 模型包
│   ├── __init__.py              │   统一导出接口
│   ├── base.py                  │   BaseModel 基类
│   ├── traditional_ml.py        │   TF-IDF + LR / SVC / RF / NB
│   ├── neural_net.py            │   TF-IDF + MLP (PyTorch)
│   ├── bert_model.py            │   BERT 微调 (bert-base-chinese)
│   ├── pdf_baselines.py         │   论文 5 个 Baseline (Word2Vec/Doc2Vec/GAS)
│   ├── gca_net.py               │   ★ GCA-Net 创新模型（字形对比对齐）
│   └── evaluation.py            │   评估工具（evaluate_model / 置信度阈值）
│
├── adversarial.py               ← 7 种中文对抗攻击策略 + 鲁棒性评估
├── visualize.py                 ← 4 类可视化图表（混淆矩阵/对比/分布/对抗）
│
├── main.py                      ← ★ 主流水线（9 步骤编排）
├── test_run.py                  ← 快速模块测试（不训练模型）
├── make_adversarial_dataset.py  ← 对抗测试集生成脚本
├── eval_adversarial.py          ← 对抗测试集独立评估脚本
│
├── dataset/                     ← 数据目录
│   ├── ChiFraud_train.csv       │   训练集（45 MB）
│   ├── ChiFraud_t2022.csv       │   验证集（23 MB）
│   ├── ChiFraud_t2023.csv       │   测试集（27 MB）
│   ├── class.txt                │   类别定义文件
│   ├── metadata.json            │   数据集元信息
│   ├── adversarial_test.csv     │   对抗测试集（脚本生成）
│   └── adversarial_test_vectorized.npz  │ 对抗测试集 TF-IDF（脚本生成）
│
└── output/                      ← 输出目录（自动创建）
    ├── final_results.csv        │   基础指标汇总
    ├── marked_dataset_threshold.csv   │ 表1：标记数据集阈值指标
    ├── adversarial_dataset_threshold.csv│ 表2：对抗数据集阈值指标
    ├── adv_testset_threshold.csv      │ 表3：对抗测试集阈值指标
    ├── adversarial_testset_overall.csv │ 对抗测试集总体
    ├── adversarial_testset_by_attack.csv│ 对抗测试集按攻击分解
    └── *.png                    │   可视化图表
```

---

## 文件说明

### 核心流水线

| 文件 | 说明 |
|------|------|
| `main.py` | **主入口**。执行 9 步流程：加载→预处理→训练11种模型→对抗测试集评估→对抗鲁棒性评估→置信度阈值指标→GCA-Net→汇总报告→可视化 |
| `config.py` | 全局常量：路径、LABEL_MAP(11类中文名)、TF-IDF参数、BERT模型名、随机种子等 |
| `data_processor.py` | 数据层：`load_data()` 加载 TSV、`clean_text()` 去 URL/空白、`tokenize()` jieba 分词、`build_tfidf_vectorizer()` 构建 30000 词表 TF-IDF |

### 模型包 (`models/`)

| 文件 | 模型 | 输入 | 分类器 | 说明 |
|------|------|------|--------|------|
| `traditional_ml.py` | TF-IDF+LR | TF-IDF 矩阵 | LogisticRegression | 快速基线 |
| | TF-IDF+SVC | TF-IDF 矩阵 | LinearSVC | 高维稀疏强项 |
| | TF-IDF+RF | TF-IDF 矩阵 | RandomForest | 非线性 |
| | TF-IDF+MNB | TF-IDF 矩阵 | MultinomialNB | 文本经典 |
| `neural_net.py` | TF-IDF+MLP | TF-IDF 矩阵 | 3层PyTorch网络 | 深度学习 |
| `bert_model.py` | BERT | 原始文本 | bert-base-chinese | SOTA |
| `pdf_baselines.py` | Word2Vec-w+LR | jieba分词→Word2Vec | LogisticRegression | 论文Baseline |
| | Word2Vec-c+LR | 字符切分→Word2Vec | LogisticRegression | 论文Baseline |
| | Word2Vec-c+GBDT | 字符切分→Word2Vec | GradientBoosting | 论文Baseline |
| | Doc2Vec-c+GBDT | 字符切分→Doc2Vec | GradientBoosting | 论文Baseline |
| | GAS(GCN) | TF-IDF→2层FC网络 | Softmax | 图卷积简化 |
| `gca_net.py` | **GCA-Net** | TF-IDF + 字形嵌入 | 融合网络 | ★ 创新方法 |
| `evaluation.py` | — | — | — | `evaluate_model`、`evaluate_with_confidence_threshold`、`compare_models` |

### 对抗与评估

| 文件 | 说明 |
|------|------|
| `adversarial.py` | 7 种中文对抗攻击：形近字替换 / 随机删字 / 插入无关字符 / 关键词掩码 / 正常文本伪装 / 数字全角混淆 / 伪装URL添加 |
| `make_adversarial_dataset.py` | 对抗测试集生成脚本，从测试集采样并按攻击策略生成变体，保存为 `adversarial_test.csv` |
| `eval_adversarial.py` | 独立评估脚本，加载预生成的对抗测试集，批量评估所有模型，输出对比报告 |
| `visualize.py` | 4 类图表：混淆矩阵 / 模型对比 / 标签分布 / 对抗鲁棒性 |

---

## 11 种模型一览

| # | 模型名称 | 类型 | 代码位置 |
|:--:|---------|------|----------|
| 1 | TF-IDF + LogisticRegression | 传统ML | `models/traditional_ml.py` |
| 2 | TF-IDF + LinearSVC | 传统ML | `models/traditional_ml.py` |
| 3 | TF-IDF + RandomForest | 传统ML | `models/traditional_ml.py` |
| 4 | TF-IDF + MultinomialNB | 传统ML | `models/traditional_ml.py` |
| 5 | Word2Vec-w + LR | 论文Baseline | `models/pdf_baselines.py` |
| 6 | Word2Vec-c + LR | 论文Baseline | `models/pdf_baselines.py` |
| 7 | Word2Vec-c + GBDT | 论文Baseline | `models/pdf_baselines.py` |
| 8 | Doc2Vec-c + GBDT | 论文Baseline | `models/pdf_baselines.py` |
| 9 | GAS (GCN) | 论文Baseline | `models/pdf_baselines.py` |
| 10 | BERT (bert-base-chinese) | 深度学习 | `models/bert_model.py` |
| 11 | **GCA-Net** | ★ 创新方法 | `models/gca_net.py` |

---

## 6 项评估指标

| 评估维度 | 指标 | 对应输出 |
|----------|------|----------|
| 基础性能 | Accuracy / Precision(macro) / Recall(macro) / F1(macro) / F1(weighted) | `final_results.csv` |
| 标记数据集@阈值 | Precision@90%, Recall@90%, F1@90%, Precision@95%, Recall@95%, F1@95% | `marked_dataset_threshold.csv` (表1) |
| 对抗数据集@阈值 | Recall@90%, Recall@95% | `adversarial_dataset_threshold.csv` (表2) |
| 对抗测试集@阈值 | Recall@90%, Recall@95% | `adv_testset_threshold.csv` (表3) |
| 对抗测试集基础 | Accuracy / F1 / 按攻击类型分解 | `adversarial_testset_overall.csv` |
| 可视化 | 混淆矩阵 / 模型对比 / 标签分布 / 对抗鲁棒性 | `output/*.png` |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成对抗测试集

```bash
python make_adversarial_dataset.py          # 默认每类200条
```

### 3. 运行完整流水线

```bash
python main.py
```

### 4. 模块测试（可选）

```bash
python test_run.py                          # 测试数据加载/预处理/对抗
```

### 5. 独立对抗评估（可选）

```bash
python eval_adversarial.py                  # 只评估已训练模型
python eval_adversarial.py --export         # 导出详细CSV
```

---

## 创新方法：GCA-Net

**Glyph-Contrastive Alignment Network（字形对比对齐网络）**

核心思想：中文字符是有结构的视觉符号，不应被降维为无意义的 token ID。

```
传统方法:  '证' → token_3847
          '証' → token_9123
          → 模型认为两个完全不同的字！

GCA-Net:   '证' → 部首[讠+正] + 拼音[zheng] + 结构[左右] + 笔画[7]
          '証' → 部首[言+正] + 拼音[zheng] + 结构[左右] + 笔画[12]
          → 共享部件'正'、同音、同结构 → 字形空间中相近！
```

三个关键设计：
1. **四维字符拆解**：部首、拼音、结构类型、笔画数
2. **GlyphEmbedding**：将四维特征融合为 128 维字形嵌入
3. **对比学习**：主动拉近形近/音近字，推远无关字

预期在对抗鲁棒性上显著优于所有非 BERT 基线。

---

## 依赖

```
python>=3.8
pandas, numpy, scikit-learn, scipy
jieba, gensim
matplotlib, seaborn
torch, transformers
```

---

## 作者

大数据原理与技术 — 期末项目
