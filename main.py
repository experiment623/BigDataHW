"""
主程序入口：垃圾文本检测器
流程：
  1. 加载 & 预处理数据
  2. 训练 TF-IDF + 传统 ML Baseline (LR/SVC/RF/NB)
  3. 训练 PDF 论文 5 个经典 Baseline (Word2Vec/Doc2Vec/GAS)
  4. 训练 MLP / BERT 深度学习模型
  5. 训练 GCA-Net 创新模型
  6. 对抗测试集评估 (加载 adversarial_test.csv)
  7. 对抗样本动态生成并测试鲁棒性
  8. 置信度阈值指标 (标记数据集 + 对抗数据集)
  9. 汇总对比结果 + 可视化
"""
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import load_npz

warnings.filterwarnings('ignore')

from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_DIR, MODEL_DIR, DATASET_DIR,
    RANDOM_SEED, LABEL_MAP, NUM_CLASSES, BERT_MODEL_NAME, N_FOLDS,
    SUBSET_TRAIN_SIZE, SUBSET_BERT_TRAIN, SUBSET_BERT_VAL, SUBSET_BERT_TEST,
    GLYPH_WEIGHT, CONTRAST_WEIGHT, GCA_EPOCHS,
)
from data_processor import (
    load_data, clean_text, tokenize, preprocess_texts,
    build_tfidf_vectorizer, load_vectorizer
)
from adversarial import (
    build_adversarial_dataset, evaluate_on_adversarial
)
from visualize import generate_all_visualizations

# 模型模块
from models import (
    TfidfLogisticRegression, TfidfLinearSVC,
    TfidfRandomForest, TfidfNaiveBayes,
    SimpleMLP, BERTClassifier,
    Word2VecWordLR, Word2VecCharLR,
    Word2VecCharGBDT, Doc2VecCharGBDT, GAS,
    GCANet,
    evaluate_model, compare_models, evaluate_with_confidence_threshold,
    cross_validate,
)
from config import CV_MODELS, CV_SAMPLE_SIZE

np.random.seed(RANDOM_SEED)


# ========== 辅助函数：消除重复代码 ==========

def _model_predict(model, X_tfidf=None, texts=None):
    """统一预测接口：根据 model.input_type 自动选择输入格式"""
    if model.input_type == 'text' and texts is not None:
        return np.array(model.predict(texts))
    return np.array(model.predict(X_tfidf))


def _model_predict_proba(model, X_tfidf=None, texts=None):
    """统一概率预测接口"""
    if model.input_type == 'text' and texts is not None:
        return model.predict_proba(texts)
    return model.predict_proba(X_tfidf)


def _prepare_subset(texts, labels, X_tfidf, n_samples):
    """分层采样子集（保持类别分布）"""
    from data_processor import stratified_sample_indices
    if n_samples is None or n_samples >= len(texts):
        return texts, labels, X_tfidf
    
    indices = stratified_sample_indices(labels, n_samples, RANDOM_SEED)
    sub_texts = [texts[i] for i in indices]
    sub_labels = labels[indices]
    sub_X = X_tfidf[indices] if hasattr(X_tfidf, '__getitem__') else X_tfidf
    return sub_texts, sub_labels, sub_X


def step1_load_data():
    """步骤1：加载原始数据"""
    print('\n' + '='*60)
    print('  [步骤1] 加载数据集')
    print('='*60)

    train_texts, train_labels = load_data(TRAIN_PATH)
    val_texts, val_labels = load_data(VAL_PATH)
    test_texts, test_labels = load_data(TEST_PATH)

    print(f'  训练集: {len(train_texts)} 条')
    print(f'    标签分布: {pd.Series(train_labels).value_counts().sort_index().to_dict()}')
    print(f'  验证集(2022): {len(val_texts)} 条')
    print(f'    标签分布: {pd.Series(val_labels).value_counts().sort_index().to_dict()}')
    print(f'  测试集(2023): {len(test_texts)} 条')
    print(f'    标签分布: {pd.Series(test_labels).value_counts().sort_index().to_dict()}')

    # 检查标签对齐
    train_labels_set = set(train_labels)
    test_labels_set = set(test_labels)
    new_in_test = test_labels_set - train_labels_set
    if new_in_test:
        print(f'\n  ⚠ 训练集未见标签出现在测试集中: {sorted(new_in_test)} -> { {l: LABEL_MAP.get(l) for l in new_in_test} }')
        print(f'    这模拟了真实场景中的"新类型欺诈"检测挑战')

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def step2_preprocess(raw_data):
    """步骤2：数据预处理（清洗+分词+TF-IDF）"""
    print('\n' + '='*60)
    print('  [步骤2] 数据预处理')
    print('='*60)

    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = raw_data

    # 清洗+分词
    print('  文本清洗 & jieba分词...')
    train_tokens = preprocess_texts(train_texts)
    val_tokens = preprocess_texts(val_texts)
    test_tokens = preprocess_texts(test_texts)

    # 构建 TF-IDF 向量化器
    print('  构建 TF-IDF 向量化器...')
    vectorizer = build_tfidf_vectorizer(
        train_tokens,
        save_path=os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    )
    print(f'  词汇表大小: {len(vectorizer.vocabulary_)}')

    # 向量化
    print('  向量化中...')
    X_train = vectorizer.transform(train_tokens)
    X_val = vectorizer.transform(val_tokens)
    X_test = vectorizer.transform(test_tokens)
    print(f'  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

    return {
        'X_train': X_train, 'y_train': np.array(train_labels),
        'X_val': X_val, 'y_val': np.array(val_labels),
        'X_test': X_test, 'y_test': np.array(test_labels),
        'vectorizer': vectorizer,
        'train_texts': train_texts, 'val_texts': val_texts, 'test_texts': test_texts,
    }


def step3_train_traditional_ml(proc_data):
    """步骤3：训练 TF-IDF + 传统 ML 模型"""
    print('\n' + '='*60)
    print('  [步骤3] 训练传统 ML Baseline 模型')
    print('='*60)

    X_train, y_train = proc_data['X_train'], proc_data['y_train']
    X_val, y_val = proc_data['X_val'], proc_data['y_val']
    X_test, y_test = proc_data['X_test'], proc_data['y_test']

    models_cfg = [
        TfidfLogisticRegression(),
        TfidfLinearSVC(),
        TfidfRandomForest(),
        TfidfNaiveBayes(),
    ]

    results_all = {'val': [], 'test': []}
    trained_models = {}

    for model in models_cfg:
        print(f'\n--- 训练: {model.name} ---')
        t0 = time.time()
        model.fit(X_train, y_train)
        t1 = time.time()
        print(f'  训练耗时: {t1-t0:.1f}s')

        # 验证集
        r_val = evaluate_model(model, X_val, y_val)
        results_all['val'].append(r_val)

        # 测试集
        r_test = evaluate_model(model, X_test, y_test)
        results_all['test'].append(r_test)

        trained_models[model.name] = model

        # 保存
        save_path = os.path.join(MODEL_DIR, model.name.replace(' ', '_') + '.pkl')
        model.save(save_path)

    # 汇总
    compare_models(results_all['val'], '验证集(2022)')
    compare_models(results_all['test'], '测试集(2023)')

    return trained_models, results_all


def step4_train_pdf_baselines(proc_data, ml_models, ml_results):
    """
    步骤4：训练 PDF 论文中的 5 个经典 Baseline
    1. Word2Vec-w+LR  → 词级 Word2Vec + 逻辑回归
    2. Word2Vec-c+LR  → 字符级 Word2Vec + 逻辑回归
    3. Word2Vec-c+GBDT → 字符级 Word2Vec + GBDT
    4. Doc2Vec-c+GBDT  → 字符级 Doc2Vec + GBDT
    5. GAS (GCN)       → 图卷积网络简化版
    """
    print('\n' + '='*60)
    print('  [步骤4] 训练 PDF 论文 5 个经典 Baseline')
    print('  Word2Vec/Doc2Vec/GAS')
    print('='*60)

    X_train = proc_data['X_train']
    y_train = proc_data['y_train']
    X_val = proc_data['X_val']
    y_val = proc_data['y_val']
    X_test = proc_data['X_test']
    y_test = proc_data['y_test']

    train_texts = proc_data['train_texts']
    val_texts = proc_data['val_texts']
    test_texts = proc_data['test_texts']

    # 对 Word2Vec/Doc2Vec 模型使用分层采样子集（保持类别分布）
    sub_texts, sub_y, sub_X = _prepare_subset(
        train_texts, y_train, X_train, SUBSET_TRAIN_SIZE
    )
    print(f'  子集大小: {len(sub_texts)}, 标签分布: {dict(pd.Series(sub_y).value_counts().sort_index())}')

    pdf_models = [
        Word2VecWordLR(vec_dim=200),       # 1. Word2Vec-w+LR
        Word2VecCharLR(vec_dim=200),       # 2. Word2Vec-c+LR
        Word2VecCharGBDT(vec_dim=200),     # 3. Word2Vec-c+GBDT
        Doc2VecCharGBDT(vec_dim=200),      # 4. Doc2Vec-c+GBDT
        GAS(hidden_dim=128),               # 5. GAS (GCN)
    ]

    for model in pdf_models:
        print(f'\n--- 训练: {model.name} ---')
        t0 = time.time()

        try:
            # Word2Vec/Doc2Vec 模型用文本输入，GAS 用 TF-IDF 矩阵
            if 'GAS' in model.name:
                model.fit(sub_X, sub_y)
                r_val = evaluate_model(model, X_val, y_val)
                r_test = evaluate_model(model, X_test, y_test)
            elif 'Word2Vec' in model.name or 'Doc2Vec' in model.name:
                model.fit(sub_texts, sub_y)
                r_val = evaluate_model(model, val_texts, y_val)
                r_test = evaluate_model(model, test_texts, y_test)
            else:
                model.fit(sub_X, sub_y)
                r_val = evaluate_model(model, X_val, y_val)
                r_test = evaluate_model(model, X_test, y_test)

            t1 = time.time()
            print(f'  训练耗时: {t1-t0:.1f}s')

            ml_results['val'].append(r_val)
            ml_results['test'].append(r_test)
            ml_models[model.name] = model

            # 保存
            save_path = os.path.join(MODEL_DIR, model.name.replace(' ', '_') + '.pkl')
            model.save(save_path)

        except Exception as e:
            print(f'  [跳过] {model.name} 训练失败: {e}')
            import traceback
            traceback.print_exc()
            continue

    return ml_models, ml_results


def step4_train_mlp(proc_data):
    """步骤4：训练 TF-IDF + MLP 神经网络"""
    print('\n' + '='*60)
    print('  [步骤4] 训练 MLP Baseline')
    print('='*60)

    X_train, y_train = proc_data['X_train'], proc_data['y_train']
    X_val, y_val = proc_data['X_val'], proc_data['y_val']
    X_test, y_test = proc_data['X_test'], proc_data['y_test']

    input_dim = X_train.shape[1]
    mlp = SimpleMLP(input_dim=input_dim, num_classes=NUM_CLASSES)
    print(f'  MLP 输入维度: {input_dim}')

    t0 = time.time()
    mlp.fit(X_train, y_train, epochs=10, batch_size=64)
    t1 = time.time()
    print(f'  训练耗时: {t1-t0:.1f}s')

    r_val = evaluate_model(mlp, X_val, y_val)
    r_test = evaluate_model(mlp, X_test, y_test)

    return mlp, r_val, r_test


def step5_train_bert(proc_data):
    """步骤5：训练 BERT 微调模型（可选，耗时较长）"""
    print('\n' + '='*60)
    print('  [步骤5] 训练 BERT 微调模型')
    print('  ⚠ 此步骤耗时较长，需要 GPU 或耐心等待...')
    print('='*60)

    train_texts = proc_data['train_texts']
    train_labels = proc_data['y_train']
    val_texts = proc_data['val_texts']
    val_labels = proc_data['y_val']
    test_texts = proc_data['test_texts']
    test_labels = proc_data['y_test']

    # 使用分层采样子集（保持类别分布，避免少数类丢失）
    sub_train_texts, sub_train_labels, _ = _prepare_subset(
        train_texts, train_labels, None, SUBSET_BERT_TRAIN
    )
    sub_val_texts, sub_val_labels, _ = _prepare_subset(
        val_texts, val_labels, None, SUBSET_BERT_VAL
    )
    sub_test_texts, sub_test_labels, _ = _prepare_subset(
        test_texts, test_labels, None, SUBSET_BERT_TEST
    )
    print(f'  BERT 子集: train={len(sub_train_texts)}, val={len(sub_val_texts)}, test={len(sub_test_texts)}')
    print(f'    训练集标签分布: {dict(pd.Series(sub_train_labels).value_counts().sort_index())}')

    bert = BERTClassifier(model_name=BERT_MODEL_NAME, num_classes=NUM_CLASSES)

    t0 = time.time()
    bert.fit(sub_train_texts, sub_train_labels, epochs=3, lr=2e-5)
    t1 = time.time()
    print(f'  训练耗时: {t1-t0:.1f}s')

    r_val = evaluate_model(bert, sub_val_texts, sub_val_labels)
    r_test = evaluate_model(bert, sub_test_texts, sub_test_labels)

    # 保存
    bert.save(os.path.join(MODEL_DIR, 'bert_model'))

    return bert, r_val, r_test, (sub_val_texts, sub_val_labels, sub_test_texts, sub_test_labels)


# ==================== 对抗测试集评估 ====================

def step_adv_testset_eval(trained_models, proc_data):
    """
    加载预生成的对抗测试集 (adversarial_test.csv)，
    对所有模型计算 accuracy / precision / recall / f1 指标

    返回: adv_test_results = {
        'overall': DataFrame,   # 各模型总体指标
        'by_attack': DataFrame, # 各攻击类型 F1 对比
    }
    """
    print('\n' + '='*60)
    print('  [对抗测试集评估] 加载预生成对抗数据集')
    print('='*60)

    adv_csv = os.path.join(DATASET_DIR, 'adversarial_test.csv')
    adv_vec = os.path.join(DATASET_DIR, 'adversarial_test_vectorized.npz')

    if not os.path.exists(adv_csv):
        print(f'  [跳过] 对抗测试集不存在: {adv_csv}')
        print(f'         请先运行: python make_adversarial_dataset.py')
        return None, None, []

    # 加载数据
    adv_df = pd.read_csv(adv_csv, encoding='utf-8-sig')
    y_adv = np.array(adv_df['label'].values)
    print(f'  加载对抗测试集: {len(adv_df)} 条')

    # 加载向量化版本
    if os.path.exists(adv_vec):
        X_adv_vec = load_npz(adv_vec)
        print(f'  向量化维度: {X_adv_vec.shape}')
    else:
        X_adv_vec = None

    # ---- 评估每个模型 ----
    print(f'\n  评估 所有模型...')
    overall_results = []
    attack_detail = {}
    adv_texts_list = adv_df['adv_text'].tolist()

    for name, model in trained_models.items():
        # 根据 model.input_type 自动选择输入格式
        if model.input_type == 'text':
            y_pred = _model_predict(model, texts=adv_texts_list)
        else:
            if X_adv_vec is None:
                print(f'    [跳过] {name}: 无向量化数据')
                continue
            y_pred = _model_predict(model, X_tfidf=X_adv_vec)

        # 总体指标
        overall_results.append({
            'Model': name,
            'accuracy': round(accuracy_score(y_adv, y_pred), 4),
            'precision_macro': round(precision_score(y_adv, y_pred, average='macro', zero_division=0), 4),
            'recall_macro': round(recall_score(y_adv, y_pred, average='macro', zero_division=0), 4),
            'f1_macro': round(f1_score(y_adv, y_pred, average='macro', zero_division=0), 4),
            'f1_weighted': round(f1_score(y_adv, y_pred, average='weighted', zero_division=0), 4),
        })

        # 按攻击类型分组
        atk_scores = {}
        for atk_idx in sorted(adv_df['attack_idx'].unique()):
            mask = adv_df['attack_idx'] == atk_idx
            if mask.sum() == 0:
                continue
            atk_name = adv_df.loc[adv_df['attack_idx'] == atk_idx, 'attack_method'].iloc[0]
            y_t = y_adv[mask]
            y_p = y_pred[mask]
            atk_scores[atk_name] = round(f1_score(y_t, y_p, average='macro', zero_division=0), 4)

        attack_detail[name] = atk_scores

    # ---- 打印报告 ----
    df_overall = pd.DataFrame(overall_results)
    df_overall = df_overall.sort_values('f1_macro', ascending=False)

    print(f'\n{"="*70}')
    print(f'  对抗测试集 - 总体指标')
    print(f'{"="*70}')
    print(df_overall.to_string(index=False))

    # 按攻击类型 F1 对比表
    if attack_detail:
        df_attack = pd.DataFrame(attack_detail).T
        df_attack.index.name = 'Model'
        print(f'\n{"="*70}')
        print(f'  对抗测试集 - 各攻击类型 F1(macro) 对比')
        print(f'{"="*70}')
        print(df_attack.round(4).to_string())

    # 保存
    df_overall.to_csv(os.path.join(OUTPUT_DIR, 'adversarial_testset_overall.csv'), index=False, encoding='utf-8-sig')
    if 'df_attack' in locals():
        df_attack.to_csv(os.path.join(OUTPUT_DIR, 'adversarial_testset_by_attack.csv'), encoding='utf-8-sig')

    # ---- 置信度阈值指标 (对抗测试集) ----
    print(f'\n  计算对抗测试集置信度阈值指标...')
    adv_test_threshold = []
    for name, model in trained_models.items():
        if model.input_type == 'text':
            r90 = evaluate_with_confidence_threshold(model, adv_texts_list, y_adv, 0.90)
            r95 = evaluate_with_confidence_threshold(model, adv_texts_list, y_adv, 0.95)
        elif X_adv_vec is not None:
            r90 = evaluate_with_confidence_threshold(model, X_adv_vec, y_adv, 0.90)
            r95 = evaluate_with_confidence_threshold(model, X_adv_vec, y_adv, 0.95)
        else:
            continue
        adv_test_threshold.append({
            'Model': name,
            f'recall@90%': r90.get(f'recall@90%', 0.0),
            f'recall@95%': r95.get(f'recall@95%', 0.0),
        })

    # 打印
    if adv_test_threshold:
        th_df = pd.DataFrame(adv_test_threshold)
        cols_th = ['Model', 'recall@90%', 'recall@95%']
        available_th = [c for c in cols_th if c in th_df.columns]
        th_df = th_df[available_th + [c for c in th_df.columns if c not in available_th]]
        print(f'\n  对抗测试集 置信度阈值 Recall:')
        print(th_df.to_string(index=False))

    return df_overall, attack_detail, adv_test_threshold


# ==================== 对抗样本动态评估 ====================

def step6_adversarial_eval(trained_models, proc_data):
    """步骤6：对抗样本评估"""
    print('\n' + '='*60)
    print('  [步骤6] 对抗样本鲁棒性评估')
    print('='*60)

    test_texts = proc_data['test_texts']
    test_labels = proc_data['y_test']
    X_test = proc_data['X_test']

    # 从测试集生成对抗样本
    print('  生成对抗样本...')
    adv_texts, adv_labels, adv_attack_types = build_adversarial_dataset(
        test_texts, test_labels, samples_per_class=100
    )

    # 为 TF-IDF 模型准备向量化后的对抗样本
    vectorizer = proc_data['vectorizer']
    adv_tokenized = preprocess_texts(adv_texts)
    X_adv = vectorizer.transform(adv_tokenized)

    adv_results = {}

    for name, model in trained_models.items():
        if model.input_type == 'text':
            df, acc = evaluate_on_adversarial(model, adv_texts, adv_labels, adv_attack_types)
        else:
            df, acc = evaluate_on_adversarial(model, X_adv, adv_labels, adv_attack_types)
        adv_results[name] = {'accuracy': acc, 'detail': df}

    # 汇总对抗评估
    print(f'\n{"="*60}')
    print(f'  对抗鲁棒性对比')
    print(f'{"="*60}')
    summary = [(name, info['accuracy']) for name, info in adv_results.items()]
    summary.sort(key=lambda x: x[1], reverse=True)
    for name, acc in summary:
        bar = '█' * int(acc * 40)
        print(f'  {name:<35s} {acc:.4f} {bar}')

    return adv_results, adv_texts, adv_labels, X_adv, adv_attack_types


def step7_confidence_threshold_eval(trained_models, proc_data, adv_texts, adv_labels, X_adv):
    """步骤7：置信度阈值指标评估 (@90% 和 @95%)"""
    print('\n' + '='*60)
    print('  [步骤7] 置信度阈值指标计算')
    print('='*60)

    X_test = proc_data['X_test']
    y_test = proc_data['y_test']
    test_texts = proc_data['test_texts']

    # ---- 标记数据集 @90% 和 @95% ----
    print('\n--- 标记数据集 置信度阈值指标 ---')
    mark_results = []
    for name, model in trained_models.items():
        print(f'  评估: {name}')
        if model.input_type == 'text':
            r90 = evaluate_with_confidence_threshold(model, test_texts, y_test, threshold=0.90)
            r95 = evaluate_with_confidence_threshold(model, test_texts, y_test, threshold=0.95)
        else:
            r90 = evaluate_with_confidence_threshold(model, X_test, y_test, threshold=0.90)
            r95 = evaluate_with_confidence_threshold(model, X_test, y_test, threshold=0.95)

        mark_results.append({
            'Model': name,
            **r90,
            **r95
        })

    # ---- 对抗数据集 @90% 和 @95% 召回率 ----
    print('\n--- 对抗数据集 置信度阈值 Recall ---')
    adv_threshold_results = []
    for name, model in trained_models.items():
        print(f'  评估: {name}')

        if model.input_type == 'text':
            r90 = evaluate_with_confidence_threshold(model, adv_texts, adv_labels, threshold=0.90)
            r95 = evaluate_with_confidence_threshold(model, adv_texts, adv_labels, threshold=0.95)
        else:
            r90 = evaluate_with_confidence_threshold(model, X_adv, adv_labels, threshold=0.90)
            r95 = evaluate_with_confidence_threshold(model, X_adv, adv_labels, threshold=0.95)

        adv_threshold_results.append({
            'Model': name,
            f'recall@90%': r90.get(f'recall@90%', 0.0),
            f'recall@95%': r95.get(f'recall@95%', 0.0),
        })

    return mark_results, adv_threshold_results


def print_final_report(mark_results, adv_threshold_results, adv_testset_threshold=None):
    """打印最终报告表格
    mark_results: 标记数据集(测试集2023)的置信度阈值指标
    adv_threshold_results: 动态生成对抗数据集的置信度阈值指标
    adv_testset_threshold: 预生成对抗测试集的置信度阈值指标
    """
    print('\n' + '='*80)
    print('  ★ 最终评估报告')
    print('='*80)

    # ---- 表1: 标记数据集 ----
    print(f'\n{"="*80}')
    print('  表1: 标记数据集 置信度阈值指标')
    print(f'{"="*80}')
    mark_df = pd.DataFrame(mark_results)
    cols = ['Model',
            'precision@90%', 'recall@90%', 'f1@90%', 'coverage',
            'precision@95%', 'recall@95%', 'f1@95%']
    available_cols = [c for c in cols if c in mark_df.columns]
    mark_df = mark_df[available_cols + [c for c in mark_df.columns if c not in available_cols]]
    print(mark_df.to_string(index=False))
    mark_df.to_csv(os.path.join(OUTPUT_DIR, 'marked_dataset_threshold.csv'), index=False, encoding='utf-8-sig')

    # ---- 表2: 对抗数据集 (动态生成) ----
    print(f'\n{"="*80}')
    print('  表2: 对抗数据集(动态) 置信度阈值 Recall')
    print(f'{"="*80}')
    adv_df = pd.DataFrame(adv_threshold_results)
    cols_adv = ['Model', 'recall@90%', 'recall@95%']
    available_adv = [c for c in cols_adv if c in adv_df.columns]
    adv_df = adv_df[available_adv + [c for c in adv_df.columns if c not in available_adv]]
    print(adv_df.to_string(index=False))
    adv_df.to_csv(os.path.join(OUTPUT_DIR, 'adversarial_dataset_threshold.csv'), index=False, encoding='utf-8-sig')

    # ---- 表3: 对抗测试集 (预生成 CSV) ----
    if adv_testset_threshold is not None and len(adv_testset_threshold) > 0:
        print(f'\n{"="*80}')
        print('  表3: 对抗测试集(预生成) 置信度阈值 Recall')
        print(f'{"="*80}')
        adv_ts_df = pd.DataFrame(adv_testset_threshold)
        available_ts = [c for c in cols_adv if c in adv_ts_df.columns]
        adv_ts_df = adv_ts_df[available_ts + [c for c in adv_ts_df.columns if c not in available_ts]]
        print(adv_ts_df.to_string(index=False))
        adv_ts_df.to_csv(os.path.join(OUTPUT_DIR, 'adv_testset_threshold.csv'), index=False, encoding='utf-8-sig')

    return mark_df, adv_df


def step8_train_gca(proc_data):
    """
    步骤8: GCA-Net v5 — 字形-字音-语义三模态联合对抗预训练

    四个预训练损失:
      L_tri       — 三模态锚点对齐 (拉近同字g/p/s)
      L_inv_sent  — 原文-变体全局对比
      L_sem_confl — 语义冲突推远 (惠/慧 等)
      L_disc      — 跨模态字符判别 (从变体推理原字)

    微调: 冻结主干 + 2层MLP分类器
    """
    print('\n' + '='*60)
    print('  [步骤8] 训练 GCA-Net (v5 三模态)')
    print('  ★ 字形流: 32×32渲染 → CNN → 512维')
    print('  ★ 字音流: pinyin序列 → 1D-CNN+BiLSTM → 512维')
    print('  ★ 语义流: 可学习嵌入 → MLP → 512维')
    print('  ★ 预训练: 对抗变体生成 + 四损失联合优化')
    print('='*60)

    X_train, y_train = proc_data['X_train'], proc_data['y_train']
    X_val, y_val = proc_data['X_val'], proc_data['y_val']
    X_test, y_test = proc_data['X_test'], proc_data['y_test']
    train_texts = proc_data['train_texts']

    # 分层采样子集
    sub_texts, sub_y, sub_X = _prepare_subset(
        train_texts, y_train, X_train, SUBSET_TRAIN_SIZE
    )
    print(f'  子集: {len(sub_texts)} 条')

    gca = GCANet(name='GCA-Net (Glyph+Phonetic+Semantic)', glyph_weight=GLYPH_WEIGHT)
    t0 = time.time()

    # Phase I: 三模态联合对抗预训练
    print('\n--- Phase I: 三模态对抗预训练 ---')
    gca.pretrain(texts=sub_texts, epochs=10, batch_size=32, lr=5e-4)

    # Phase II: 微调
    print('\n--- Phase II: 冻结主干 + 微调分类器 ---')
    gca.fit(
        X_tfidf=sub_X, y=sub_y, texts=sub_texts,
        epochs=min(GCA_EPOCHS, 10), batch_size=32, lr=1e-3,
        glyph_weight=GLYPH_WEIGHT, contrast_weight=CONTRAST_WEIGHT
    )
    print(f'  总训练耗时: {time.time()-t0:.1f}s')

    r_val = evaluate_model(gca, X_val, y_val)
    r_test = evaluate_model(gca, X_test, y_test)
    gca.save(os.path.join(MODEL_DIR, 'gca_net_model.pt'))

    return gca, r_val, r_test


def step8b_gca_adversarial_eval(gca, proc_data, adv_texts, adv_labels, X_adv, adv_attack_types):
    """GCA-Net 对抗鲁棒性评估"""
    from adversarial import evaluate_on_adversarial
    
    print(f'\n--- GCA-Net 对抗鲁棒性 ---')
    df, acc = evaluate_on_adversarial(gca, X_adv, adv_labels, adv_attack_types)
    
    # 也用带文本的方式评估（如果有原文本的话，可以捕获字形信息）
    # 对抗样本中需要用到 predict_proba，GCA-Net 支持
    print(f'  GCA-Net 对抗样本总体准确率: {acc:.4f}')
    
    return {'accuracy': acc, 'detail': df}


def step8c_gca_confidence_eval(gca, proc_data, adv_texts, adv_labels, X_adv):
    """GCA-Net 置信度阈值指标"""
    from models.evaluation import evaluate_with_confidence_threshold
    
    X_test = proc_data['X_test']
    y_test = proc_data['y_test']
    test_texts = proc_data['test_texts']
    
    # 标记数据集
    r90 = evaluate_with_confidence_threshold(gca, X_test, y_test, threshold=0.90)
    r95 = evaluate_with_confidence_threshold(gca, X_test, y_test, threshold=0.95)
    
    mark_result = {
        'Model': 'GCA-Net',
        **r90,
        **r95
    }
    
    # 对抗数据集
    r90_adv = evaluate_with_confidence_threshold(gca, X_adv, adv_labels, threshold=0.90)
    r95_adv = evaluate_with_confidence_threshold(gca, X_adv, adv_labels, threshold=0.95)
    
    adv_result = {
        'Model': 'GCA-Net',
        f'recall@90%': r90_adv.get(f'recall@90%', 0.0),
        f'recall@95%': r95_adv.get(f'recall@95%', 0.0),
    }
    
    return mark_result, adv_result


def main():
    """主流程"""
    print('='*60)
    print('  垃圾文本检测器 - 多模型对比实验')
    print(f'  ChiFraud 中文欺诈文本数据集')
    print('='*60)

    # Step 1: 加载数据
    raw_data = step1_load_data()

    # Step 2: 预处理
    proc_data = step2_preprocess(raw_data)

    # Step 3: 传统 ML Baseline (TF-IDF+LR/SVC/RF/NB)
    ml_models, ml_results = step3_train_traditional_ml(proc_data)

    # ===== Step 3.5: 交叉验证（增强实验可信度）=====
    cv_results = {}
    X_train = proc_data['X_train']
    y_train = proc_data['y_train']
    X_cv = X_train[:CV_SAMPLE_SIZE]
    y_cv = y_train[:CV_SAMPLE_SIZE]

    for model_name in CV_MODELS:
        if model_name in ml_models:
            print(f'\n{"="*60}')
            print(f'  [交叉验证] {model_name} ({N_FOLDS}-Fold)')
            print(f'{"="*60}')
            factory_map = {
                'TF-IDF + LogisticRegression': TfidfLogisticRegression,
                'TF-IDF + LinearSVC': TfidfLinearSVC,
            }
            if model_name in factory_map:
                cv_summary = cross_validate(
                    lambda: factory_map[model_name](), X_cv, y_cv
                )
                cv_results[model_name] = cv_summary

    # Step 4: PDF 论文 5 个经典 Baseline (Word2Vec/Doc2Vec/GAS)
    ml_models, ml_results = step4_train_pdf_baselines(proc_data, ml_models, ml_results)

    # Step 5: MLP 神经网络
    mlp_model, mlp_val, mlp_test = step4_train_mlp(proc_data)
    ml_models[mlp_model.name] = mlp_model

    # Step 6: BERT 微调（可选）
    bert_skip = False
    try:
        bert_model, bert_val, bert_test, bert_data = step5_train_bert(proc_data)
        ml_models[bert_model.name] = bert_model
    except Exception as e:
        print(f'\n[警告] BERT 训练跳过: {e}')
        bert_skip = True

    # Step 6: 对抗样本评估
    adv_results, adv_texts, adv_labels, X_adv, adv_attack_types = step6_adversarial_eval(ml_models, proc_data)

    # Step 7: 置信度阈值指标 (标记数据集 + 对抗数据集)
    mark_threshold, adv_threshold = step7_confidence_threshold_eval(
        ml_models, proc_data, adv_texts, adv_labels, X_adv
    )

    # ===================== Step 8: 创新模型 GCA-Net =====================
    gca_skip = False
    try:
        gca_model, gca_val, gca_test = step8_train_gca(proc_data)
        ml_models[gca_model.name] = gca_model
        
        # GCA-Net 对抗评估
        gca_adv_result = step8b_gca_adversarial_eval(
            gca_model, proc_data, adv_texts, adv_labels, X_adv, adv_attack_types
        )
        adv_results[gca_model.name] = gca_adv_result
        
        # GCA-Net 置信度阈值评估
        gca_mark, gca_adv_thresh = step8c_gca_confidence_eval(gca_model, proc_data, adv_texts, adv_labels, X_adv)
        mark_threshold.append(gca_mark)
        adv_threshold.append(gca_adv_thresh)
        
    except Exception as e:
        print(f'\n[警告] GCA-Net 训练跳过: {e}')
        import traceback
        traceback.print_exc()
        gca_skip = True

    # ===================== 对抗测试集评估 =====================
    adv_test_overall, adv_test_detail, adv_testset_threshold = step_adv_testset_eval(ml_models, proc_data)

    # ===================== 最终结果汇总 =====================
    print('\n' + '='*70)
    print('  最终结果汇总')
    print('='*70)

    # 汇总所有模型的测试集结果
    all_results = ml_results['test'].copy()
    all_results.append(mlp_test)
    bert_results_tuple = None
    if not bert_skip:
        all_results.append(bert_test)
        bert_results_tuple = (bert_val, bert_test, bert_data)
    if not gca_skip:
        all_results.append(gca_test)

    final_df = pd.DataFrame(all_results)
    final_df = final_df.sort_values('f1_macro', ascending=False)
    print(final_df.to_string(index=False))

    # 打印对抗测试集对比摘要
    if adv_test_overall is not None:
        print(f'\n  [对抗测试集 Top-3]:')
        for i, row in adv_test_overall.head(3).iterrows():
            print(f'    {row["Model"]:<40s} F1={row["f1_macro"]:.4f}')

    # 保存结果
    final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_results.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  结果已保存至: {os.path.join(OUTPUT_DIR, "final_results.csv")}')

    # ===================== 最终报告 =====================
    print_final_report(mark_threshold, adv_threshold, adv_testset_threshold)

    # ===================== 可视化 =====================
    generate_all_visualizations(
            raw_data=raw_data,
            proc_data=proc_data,
            ml_models=ml_models,
            ml_results=ml_results,
            mlp_val=mlp_val,
            mlp_test=mlp_test,
            adv_results=adv_results,
            bert_results=bert_results_tuple
        )

    print('\n' + '='*60)
    print('  实验完成！')
    print('='*60)


if __name__ == '__main__':
    main()
