# 中文欺诈文本检测系统

> 大数据原理与技术 — 期末项目  
> 基于 ChiFraud 数据集的垃圾/欺诈文本 10 分类检测与对抗鲁棒性评估

---

## 项目概述

对中文短信/社交文本进行 **10 分类** 欺诈检测（正常 + 9 类欺诈），实现 **11 种模型**（4 传统 ML + 5 论文 Baseline + 1 深度学习 + 1 创新模型），并设计 **8 种中文对抗攻击策略** 进行鲁棒性评估。

### 类别映射

| Label | 含义 | Label | 含义 |
|:-----:|------|:-----:|------|
| 0 | 正常 | 5 | 违规提现 |
| 1 | 赌博博彩 | 6 | 虚假证明 |
| 2 | 招嫖色情 | 7 | 虚假手机卡 |
| 3 | 办假证 | 8 | 地下黑贷 |
| 4 | 违禁药品交易 | 9 | 虚假办卡 |

### 数据集

| 数据集 | 文件 | 用途 |
|--------|------|------|
| 训练集 | `ChiFraud_train.csv` | 模型训练 |
| 验证集(2022) | `ChiFraud_t2022.csv` | 超参调优 |
| 测试集(2023) | `ChiFraud_t2023.csv` | 最终评估 |
| 对抗测试集 | `adversarial_test.csv` | 对抗鲁棒性评估（脚本生成） |

---

## 项目结构

```
Final_hw/
│
├── README.md                      ← 本文件
├── requirements.txt                ← Python 依赖清单
├── config.py                       ← 全局配置（路径/超参/类别映射）
│
├── pipeline_base.py                ← ★ 共享基础模块（数据加载/预处理/评估缓存）
│
├── data_processor.py               ← 数据层：加载/清洗/jieba分词/TF-IDF/分层采样
│
├── models/                         ← ★ 模型包
│   ├── __init__.py                 │   统一导出接口
│   ├── base.py                     │   BaseModel 基类 (含 input_type)
│   ├── traditional_ml.py           │   TF-IDF + LR/SVC/RF/NB
│   ├── neural_net.py               │   TF-IDF + MLP (PyTorch)
│   ├── bert_model.py               │   BERT 微调 (bert-base-chinese)
│   ├── pdf_baselines.py            │   论文 5 个 Baseline (Word2Vec/Doc2Vec/GAS)
│   │                               │   使用 HistGradientBoosting 加速 GBDT
│   ├── gca_net.py                  │   ★ GCA-Net v5：字形-字音-语义三模态对抗预训练
│   └── evaluation.py               │   评估工具（evaluate/cross_validate/置信度阈值）
│
├── adversarial.py                  ← 8 种中文对抗攻击策略 + 鲁棒性评估
├── visualize.py                    ← 4 类可视化图表
│
├── main.py                         ← ★ 主流水线（一键跑全部模型+对比+可视化）
├── run_single.py                   ← ★ 单模型独立运行（12 个模型任选，互不阻塞）
│
├── make_adversarial_dataset.py     ← 对抗测试集生成脚本
├── eval_adversarial.py             ← 对抗测试集独立评估脚本
├── test_run.py                     ← 快速模块测试（不训练模型）
│
├── clean_label10.py                ← 工具：删除 label=10
├── show_label_dist.py              ← 工具：展示标签分布
│
├── dataset/                        ← 数据目录
│   ├── ChiFraud_train.csv          │   训练集
│   ├── ChiFraud_t2022.csv          │   验证集
│   ├── ChiFraud_t2023.csv          │   测试集
│   ├── class.txt                   │   类别定义文件
│   ├── adversarial_test.csv        │   对抗测试集（脚本生成）
│   └── adversarial_test_vectorized.npz│ 对抗测试集 TF-IDF（脚本生成）
│
└── output/                         ← 输出目录（自动创建）
    ├── result_*.csv                 │   每个模型的独立结果
    ├── final_results.csv            │   基础指标汇总
    ├── adversarial_testset_overall.csv│ 对抗测试集总体
    ├── adversarial_testset_by_attack.csv│ 对抗测试集按攻击分解
    └── *.png                        │   可视化图表
```

---

## 11 种模型一览

| # | 模型名称 | 类型 | 运行命令 |
|:--:|---------|------|----------|
| 1 | TF-IDF + LogisticRegression | 传统ML | `python run_single.py --model lr` |
| 2 | TF-IDF + LinearSVC | 传统ML | `python run_single.py --model svc` |
| 3 | TF-IDF + RandomForest | 传统ML | `python run_single.py --model rf` |
| 4 | TF-IDF + MultinomialNB | 传统ML | `python run_single.py --model nb` |
| 5 | Word2Vec-w + LR | 论文Baseline | `python run_single.py --model w2v_w` |
| 6 | Word2Vec-c + LR | 论文Baseline | `python run_single.py --model w2v_c` |
| 7 | Word2Vec-c + GBDT | 论文Baseline | `python run_single.py --model w2v_gbdt` |
| 8 | Doc2Vec-c + GBDT | 论文Baseline | `python run_single.py --model d2v_gbdt` |
| 9 | GAS (GCN) | 论文Baseline | `python run_single.py --model gas` |
| 10 | TF-IDF + MLP | 深度学习 | `python run_single.py --model mlp` |
| 11 | **GCA-Net (三模态)** | ★ 创新方法 | `python run_single.py --model gca` |

BETR 也可单独运行：`python run_single.py --model bert`

---

## 创新方法：GCA-Net v5 — 字形-字音-语义三模态联合对抗预训练

#### 核心思想

传统方法将中文视为无结构的 token 序列。GCA-Net 通过**三个独立模态编码器**理解每个汉字，并通过**对抗变体预训练**主动学习鲁棒性。

#### 三模态编码

| 模态 | 输入 | 编码器 | 输出维度 |
|:---|:---|:---|:---:|
| 字形流 (Glyph) | 32×32 字体渲染灰度图 | 4层CNN | 512 |
| 字音流 (Phonetic) | 带声调拼音序列 | 1D-CNN + BiLSTM | 512 |
| 语义流 (Semantic) | 可学习字符嵌入(方案C) | MLP | 512 |

三模态拼接 → 768维 → 3层Transformer → 上下文编码

#### 四损失联合预训练

```
L_total = L_tri + 0.5·L_inv_sent + 0.2·L_sem_confl + 1.0·L_disc

L_tri:       三模态锚点对齐 — 每个模态预测另外两模态的均值
L_inv_sent:  对抗不变性 — 原文与对抗变体句表示必须接近
L_sem_confl: 语义冲突推远 — 故意替换的冲突字对(如惠/慧)要远离
L_disc:      跨模态字符判别 — 从被替换位置推理原始字符
```

#### 微调

冻结预训练主干，仅训练 2 层 MLP 分类器（TF-IDF + 上下文表示 → 10类）。

---

## 8 种对抗攻击策略

| # | 策略 | 攻击维度 | 说明 |
|:--:|------|:---|------|
| ① | 形近字替换 | 视觉 | 证→証, 药→薬 |
| ② | 随机删字 | 完整性 | 删除10%字符 |
| ③ | 插入无关字符 | 密度 | 插入的/了/空格 |
| ④ | 关键词掩码 | 完整性 | 用*#X替换字符 |
| ⑤ | 正常文本伪装 | 上下文 | 头部/尾部插入正常文本 |
| ⑥ | 数字全角混淆 | 编码 | 5→５, 1→１ |
| ⑦ | 伪装URL添加 | 上下文 | 追加伪装链接 |
| ⑧ | 拼音字母简写 | 符号约定 | 微→v, 加→+, 微信→vx |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成对抗测试集

```bash
python clean_label10.py                    # 清理 label=10 (仅首次)
python make_adversarial_dataset.py          # 生成 adversarial_test.csv
python show_label_dist.py                   # 查看标签分布
```

### 3. 单独运行模型

```bash
# 快速验证 (30秒)
python run_single.py --model lr

# 传统ML全部
python run_single.py --model lr
python run_single.py --model svc
python run_single.py --model rf
python run_single.py --model nb

# 论文Baseline (各2-3分钟)
python run_single.py --model w2v_w
python run_single.py --model w2v_c
python run_single.py --model w2v_gbdt
python run_single.py --model d2v_gbdt
python run_single.py --model gas

# 深度学习
python run_single.py --model mlp            # 2分钟
python run_single.py --model bert           # 10-20分钟

# 创新模型
python run_single.py --model gca            # 15分钟

# 含交叉验证 + 对抗评估
python run_single.py --model lr --cv --all
```

### 4. 一键运行全部 (完整流水线)

```bash
python main.py
```

### 5. 工具脚本

```bash
python test_run.py                          # 快速模块测试
python show_label_dist.py                   # 查看数据集标签分布
python clean_label10.py                     # 删除 label=10 行
python eval_adversarial.py                  # 独立对抗评估
python eval_adversarial.py --export         # 导出详细CSV
```

---

## 输出文件

每个模型运行后在 `output/` 生成独立 CSV：

```
output/
├── result_TF-IDF_+_LogisticRegression.csv
├── result_TF-IDF_+_LinearSVC.csv
├── result_TF-IDF_+_RandomForest.csv
├── result_TF-IDF_+_MultinomialNB.csv
├── result_TF-IDF_+_MLP.csv
├── result_BERTbert-base-chinese.csv
├── result_Word2Vec-w+LR.csv
├── result_Word2Vec-c+LR.csv
├── result_Word2Vec-c+GBDT.csv
├── result_Doc2Vec-c+GBDT.csv
├── result_GAS_GCN.csv
├── result_GCA-Net_Glyph+Phonetic+Semantic.csv
└── *.png (可视化图表)
```

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

---

## 作者

大数据原理与技术 — 期末项目
