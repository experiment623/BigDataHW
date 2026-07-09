# 中文欺诈文本检测系统

> 大数据原理与技术 · 期末项目

## 这个项目做什么

一句话概括：给定一段中文短信或社交文本，判断它是不是欺诈信息，如果是的话，属于哪种欺诈——赌博、色情、假证、黑贷等 9 种类型之一，加上正常文本共 10 个类别。

用的是 ChiFraud 数据集，训练集约 8 万条，测试集 11 万多条。最大的挑战不是模型选型，而是两个根本问题：第一，欺诈文本用黑话和变体字（"办假证"写成"办假証"），传统 jieba 分词会把关键信息切碎；第二，类别极度不平衡，正常文本占了 83%，有的欺诈类别只占 0.3%。

## 做了什么

整个项目可以理解为四步走：

**第一步，跑通 Baseline。** 实现 5 个经典方法——Word2Vec 词级/字符级 + LR/GBDT、GAS（一个简化版 GCN 文本分类器）。这批模型的最佳 F1-macro 是 0.6879（GAS），但少数类的 F1 惨不忍睹，"违禁药品"只有 0.50，"办假证"只有 0.56。问题出在哪？分词。jieba 把"办假证"切成"办 / 假 / 证"，特征完全散了。

**第二步，字符级 N-gram 绕过分词。** 这是整个项目性价比最高的改进。既然分词是瓶颈，那就直接不做了——用 TfidfVectorizer 的 `analyzer='char'` 在字符序列上直接滑 n-gram 窗口，`ngram_range=(1,5)`，词表从 3 万涨到 30 万。`char15_sgd_log` 这个配置在 CPU 上就能跑出 F1-macro 0.787，比 Baseline 最佳高了 14.4%。而且对"违禁药品"的 F1 从 0.50 拉到了 0.76，说明绕过分词确实抓住了之前被切碎的关键模式。

**第三步，引入 MacBERT 做语义理解。** 字符 N-gram 虽然局部模式抓得好，但缺乏全局语义理解。MacBERT 的全词遮罩预训练天然对形近字有鲁棒性——它见过大量"証"和"证"出现在相同上下文的语料。同时试了 7 种应对类别不平衡的方法：class_weight、Focal Loss、WeightedRandomSampler、数据增强等。最优单模型 macbert_cwb 的 F1-macro 0.7712，比 char15_sgd_log 的 0.787 还略低，但 Recall 更高（0.738 vs 0.712），说明 Transformer 更"敢判"欺诈。

**第四步，跨范式集成，也是最终的核心创新。** 前面的实验让我意识到，字符 N-gram 和 Transformer 的错误模式不重叠——N-gram 对局部字符变体敏感，Transformer 对全局语义强但可能在形近字上翻车。于是把 22 个模型（8 个字符 N-gram 变体 + 14 个 MacBERT/RoBERTa epoch checkpoint）统一纳入集成框架，加权投票 + per-class 校正因子 + auto-tune 自动搜权重。最终 **ensemble_cross** 的 F1-macro 达到 **0.868**，"违禁药品"F1 从 Baseline 的 0.50 飙到 0.85（+67.8%），"办假证"从 0.56 到 0.78（+38.6%）。

## 实验结果一览

评估都在 ChiFraud_t2023.csv（114,546 条）上做，下面是最关键的几个数字：

| | Accuracy | F1-macro | Recall-macro | F1@90 |
|:---|:---:|:---:|:---:|:---:|
| Baseline 最佳（GAS） | 0.9221 | 0.6879 | 0.6321 | 0.1392 |
| 字符 N-gram 最佳（char15_sgd_log） | 0.9404 | 0.7870 | 0.7122 | 0.7634 |
| Transformer 最佳（macbert_cwb） | 0.9352 | 0.7712 | 0.7380 | 0.8611 |
| Transformer 集成（ensemble_auto） | 0.9577 | 0.8504 | 0.8243 | 0.9303 |
| **跨范式集成（ensemble_cross）** | **0.9712** | **0.8680** | **0.8430** | **0.9389** |

几个值得注意的点：

1. **字符 N-gram 性价比最高**。CPU 上就能跑，F1 比 Baseline 提升 14%，甚至比 MacBERT 单模型还高。如果只做一个改进，做这个。

2. **置信度校准很关键**。GAS 的 F1@90 只有 0.14，说明它的错误预测自信得很，这在风控里是不可接受的——你不知道什么时候该相信模型的判断。ensemble_cross 的 F1@90 是 0.94，可以安心拒绝 10% 不确定的样本。

3. **对抗鲁棒性差距大**。char15_lr_saga 的对抗 F1 只比干净 F1 低 2.8%，而其他模型普遍跌 17%-21%。Logistic Regression 的 L2 正则化似乎在分布偏移下更稳。

4. **集成的质变来自跨范式**。只用 Transformer 做集成（ensemble_auto，14 个模型）的 F1-macro 是 0.850，加上 8 个字符 N-gram 模型（ensemble_cross，22 个模型）涨到 0.868。字符 N-gram 贡献了额外的 +1.8%，验证了两类模型错误互补的假设。

## 怎么跑

### 装依赖

```bash
pip install -r requirements.txt
```

### 一键全流程

```bash
python run_all.py                    # 从头训练所有模型
python run_all.py --load             # 加载已有模型评估
python run_all.py --adv --ensemble   # 含对抗测试和集成
```

### 分步跑

```bash
# Baseline（CPU 可跑，5 个模型）
python run_baselines.py

# 字符 N-gram SOTA（CPU，8 组配置）
python run_sota.py --experiments all

# Transformer 微调（需要 GPU）
python run_transformer_sota.py --train-with-val --epochs 2

# 跨范式集成（自动发现模型 + 搜权重）
python run_ensemble_sota.py --discover --auto-tune --name ensemble_cross
```

### 可视化

```bash
python visualize.py              # 混淆矩阵、模型对比、学习曲线
python app.py                    # 启动 Web 对比系统（http://127.0.0.1:5000）
```

## 项目结构

```
Final_hw/
├── config.py                    全局配置
├── data_processor.py            数据加载/清洗/分词/TF-IDF
├── models/
│   ├── base.py                  BaseModel 基类
│   ├── baselines.py             5 个 Baseline 实现
│   └── evaluation.py            评估工具（含置信度阈值指标）
├── run_baselines.py             Baseline 运行脚本
├── run_sota.py                  字符 N-gram SOTA 运行脚本
├── run_transformer_sota.py      Transformer 微调脚本
├── run_ensemble_sota.py         跨范式集成脚本
├── run_all.py                   一键全流程主控
├── make_adversarial_dataset.py  8 种对抗攻击数据集生成
├── visualize.py                 可视化
├── postprocess_binary.py        十分类 → 二分类
├── app.py + templates/          Flask Web 多模型对比系统
├── dataset/                     数据目录
└── output/                      模型输出和指标
```

## 依赖

核心是 `torch`、`transformers`、`scikit-learn`、`jieba`、`gensim`。具体版本见 `requirements.txt`。

---

详细的算法设计、代码说明和完整实验数据见 `实验报告.md`。
