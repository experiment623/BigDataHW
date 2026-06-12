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

### 3. 运行模型

```bash
# ===== Baseline（5 个，CPU 可跑）=====
python run_baselines.py                             # 全部 5 个，各 30k 子集
python run_baselines.py --models w2v_c gas          # 指定模型
python run_baselines.py --full                      # 全量训练集
python run_baselines.py --adv                       # 含对抗评估

# ===== 字符 N-gram SOTA（CPU）=====
python run_sota.py --experiments all --save-predictions --adv
python run_sota.py --experiments char15_svc_c1 --train-with-val

# ===== Transformer 微调（需 GPU）=====
python run_transformer_sota.py --train-with-val --epochs 2 --save-predictions --adv

# ===== 模型集成（需先跑 run_sota 和 run_transformer 生成预测）=====
python run_ensemble_sota.py --save-predictions

# ===== 可视化 =====
python visualize.py

# ===== 二分类后处理 =====
python postprocess_binary.py

# ===== 对抗数据集生成 =====
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

## 作者

大数据原理与技术 — 期末项目
