"""
单模型独立运行脚本 (带进度条)
=============================
每个模型一条命令, 互不阻塞。

用法:
  python run_single.py --model lr         # TF-IDF + LogisticRegression
  python run_single.py --model svc        # TF-IDF + LinearSVC
  python run_single.py --model rf         # TF-IDF + RandomForest
  python run_single.py --model nb         # TF-IDF + MultinomialNB
  python run_single.py --model mlp        # TF-IDF + MLP
  python run_single.py --model bert       # BERT(bert-base-chinese)
  python run_single.py --model w2v_w      # Word2Vec-w+LR
  python run_single.py --model w2v_c      # Word2Vec-c+LR
  python run_single.py --model w2v_gbdt   # Word2Vec-c+GBDT
  python run_single.py --model d2v_gbdt   # Doc2Vec-c+GBDT
  python run_single.py --model gas        # GAS (GCN)
  python run_single.py --model gca        # GCA-Net (三模态)

前置条件:
  不需要。首次运行自动加载数据、构建TF-IDF词表并缓存。
  如需对抗评估(--all), 需先运行一次: python make_adversarial_dataset.py

可选:
  --all    含对抗测试集评估 (需预生成 adversarial_test.csv)
  --cv     含 K-Fold 交叉验证
  --load   加载已保存模型 (跳过训练, 如 GCA: models/gca_net_model.pt)
"""
import os, sys, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OUTPUT_DIR, MODEL_DIR, SUBSET_TRAIN_SIZE, SUBSET_BERT_TRAIN,
    SUBSET_BERT_VAL, SUBSET_BERT_TEST, GLYPH_WEIGHT, CONTRAST_WEIGHT,
    GCA_EPOCHS, NUM_CLASSES, BERT_MODEL_NAME, N_FOLDS, RANDOM_SEED,
    DATASET_DIR,
)
from pipeline_base import (
    get_prepared_data, load_adversarial_testset,
    evaluate_on_adversarial_testset, evaluate_thresholds,
    print_single_model_report, subset_data,
)
import numpy as np; np.random.seed(RANDOM_SEED)

# ── 进度条 ──
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw):
        for i, item in enumerate(iterable):
            if i % max(1, len(iterable)//5) == 0:
                print(f'    [{i+1}/{len(iterable)}]', flush=True)
            yield item

# ===================================================================
# 前置条件检查
# ===================================================================

def check_prerequisites(do_full=False):
    """检查运行所需文件是否存在, 打印提示"""
    issues = []

    # 数据文件
    for path, desc in [
        (os.path.join(DATASET_DIR, 'ChiFraud_train.csv'), '训练集'),
        (os.path.join(DATASET_DIR, 'ChiFraud_t2022.csv'), '验证集'),
        (os.path.join(DATASET_DIR, 'ChiFraud_t2023.csv'), '测试集'),
    ]:
        if not os.path.exists(path):
            issues.append(f'  ✗ 缺少{desc}: {path}')

    # 对抗数据
    if do_full:
        adv_path = os.path.join(DATASET_DIR, 'adversarial_test.csv')
        if not os.path.exists(adv_path):
            issues.append(f'  ✗ 缺少对抗测试集, 请先运行: python make_adversarial_dataset.py')

    if issues:
        print('\n[前置检查] 发现问题:')
        for i in issues:
            print(i)
        return False

    print('[前置检查] ✓ 数据文件就绪')
    return True

# ===================================================================
# 模型注册表
# ===================================================================

def _build_registry():
    from models import (
        TfidfLogisticRegression, TfidfLinearSVC,
        TfidfRandomForest, TfidfNaiveBayes,
        SimpleMLP, BERTClassifier,
        Word2VecWordLR, Word2VecCharLR,
        Word2VecCharGBDT, Doc2VecCharGBDT, GAS,
        GCANet, evaluate_model,
    )
    from models.evaluation import cross_validate

    return {
        'lr':       {'name':'TF-IDF + LogisticRegression', 'factory':lambda: TfidfLogisticRegression(),         'type':'tfidf','subset':False},
        'svc':      {'name':'TF-IDF + LinearSVC',          'factory':lambda: TfidfLinearSVC(),                  'type':'tfidf','subset':False},
        'rf':       {'name':'TF-IDF + RandomForest',       'factory':lambda: TfidfRandomForest(),               'type':'tfidf','subset':False},
        'nb':       {'name':'TF-IDF + MultinomialNB',       'factory':lambda: TfidfNaiveBayes(),                 'type':'tfidf','subset':False},
        'mlp':      {'name':'TF-IDF + MLP',                'factory':lambda: SimpleMLP(input_dim=0,num_classes=NUM_CLASSES), 'type':'tfidf','subset':False},
        'bert':     {'name':'BERT(bert-base-chinese)',      'factory':lambda: BERTClassifier(model_name=BERT_MODEL_NAME,num_classes=NUM_CLASSES), 'type':'text','subset':True},
        'w2v_w':    {'name':'Word2Vec-w+LR',               'factory':lambda: Word2VecWordLR(vec_dim=200),       'type':'text','subset':True},
        'w2v_c':    {'name':'Word2Vec-c+LR',               'factory':lambda: Word2VecCharLR(vec_dim=200),       'type':'text','subset':True},
        'w2v_gbdt': {'name':'Word2Vec-c+GBDT',             'factory':lambda: Word2VecCharGBDT(vec_dim=200),     'type':'text','subset':True},
        'd2v_gbdt': {'name':'Doc2Vec-c+GBDT',              'factory':lambda: Doc2VecCharGBDT(vec_dim=200),      'type':'text','subset':True},
        'gas':      {'name':'GAS (GCN)',                   'factory':lambda: GAS(hidden_dim=128),               'type':'tfidf','subset':True},
        'gca':      {'name':'GCA-Net (Glyph+Phonetic+Semantic)', 'factory':lambda: GCANet(name='GCA-Net',glyph_weight=GLYPH_WEIGHT), 'type':'tfidf','subset':True,'is_gca':True},
    }, evaluate_model, cross_validate

REGISTRY, evaluate_model, cross_validate = _build_registry()

# ===================================================================
# 训练 + 评估
# ===================================================================

def run_model(model_key, proc, do_cv=False, do_full=False, is_binary=False, do_plot=False, do_load=False):
    info = REGISTRY[model_key]
    bar = '█' * 50
    num_cls = 2 if is_binary else 10

    print(f'\n{"="*60}')
    print(f'  {info["name"]}' + (' [二分类]' if is_binary else ' [10分类]'))
    print(f'{"="*60}')

    X_train, y_train = proc['X_train'], proc['y_train']
    X_val, y_val     = proc['X_val'], proc['y_val']
    X_test, y_test   = proc['X_test'], proc['y_test']
    train_texts, val_texts = proc['train_texts'], proc['val_texts']
    test_texts = proc['test_texts']

    model = info['factory']()

    if model_key == 'mlp':
        model.input_dim = X_train.shape[1]
        model.num_classes = num_cls  # 覆盖类别数
        model._build_model()
    elif model_key == 'bert':
        model.num_classes = num_cls   # 覆盖类别数
    elif model_key == 'gca':
        model.num_classes = num_cls

    # 子集采样
    if model_key == 'bert':
        print('  分层采样中...')
        sub_texts, sub_y, _ = subset_data(train_texts, y_train, None, SUBSET_BERT_TRAIN)
        sub_val_t, sub_val_y, _ = subset_data(val_texts, y_val, None, SUBSET_BERT_VAL)
        sub_test_t, sub_test_y, _ = subset_data(test_texts, y_test, None, SUBSET_BERT_TEST)
        sub_X = None
        print(f'  BERT 子集: train={len(sub_texts)} val={len(sub_val_t)} test={len(sub_test_t)}')
    elif info['subset']:
        print('  分层采样中...')
        sub_texts, sub_y, sub_X = subset_data(train_texts, y_train, X_train, SUBSET_TRAIN_SIZE)
        print(f'  子集: {len(sub_texts)} 条')
    else:
        sub_texts, sub_y, sub_X = train_texts, y_train, X_train

    # ── 训练 (带进度条) ──
    print(f'\n  [训练] {info["name"]}')
    t0 = time.time()

    if info.get('is_gca'):
        gca_save_path = os.path.join(MODEL_DIR, 'gca_net_model.pt')
        if do_load and os.path.exists(gca_save_path):
            print('  [加载] 跳过训练, 直接从磁盘加载模型...')
            model.load(gca_save_path)
            t0 = time.time()  # 加载几乎不耗时, 但保持结构一致
        else:
            if do_load and not os.path.exists(gca_save_path):
                print(f'  [提示] 未找到已保存模型 {gca_save_path}, 将重新训练')
            print('  Phase I: 三模态对抗预训练 (进度见下方 epoch 输出)...')
            model.pretrain(texts=sub_texts, epochs=10, batch_size=32, lr=5e-4)
            print('  Phase II: 微调分类器 (进度见下方 epoch 输出)...')
            model.fit(X_tfidf=sub_X, y=sub_y, texts=sub_texts,
                      epochs=min(GCA_EPOCHS, 10), batch_size=32, lr=1e-3,
                      glyph_weight=GLYPH_WEIGHT, contrast_weight=CONTRAST_WEIGHT)

    elif info['type'] == 'text':
        # Word2Vec/Doc2Vec/BERT (内部有epoch打印, 外层显示进度)
        print('    训练中...')
        if model_key == 'bert':
            model.fit(sub_texts, sub_y, epochs=3, lr=2e-5)
        else:
            model.fit(sub_texts, sub_y)

    else:
        # sklearn 模型: 一次性fit
        print('    拟合中 (sklearn 单步, 无迭代进度)...')
        model.fit(sub_X, sub_y)

    elapsed = time.time() - t0
    if info.get('is_gca') and do_load and os.path.exists(gca_save_path):
        print(f'  ✓ 模型已加载, 耗时: {elapsed:.1f}s')
    else:
        print(f'  ✓ 训练完成, 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')

    # ── 测试集评估 ──
    print(f'\n  [评估] 测试集')
    if model_key == 'bert':
        r_test = evaluate_model(model, sub_test_t, sub_test_y)
    elif info['type'] == 'text' and not info.get('is_gca'):
        r_test = evaluate_model(model, test_texts, y_test)
    else:
        r_test = evaluate_model(model, X_test, y_test)

    # ── 保存 (加载时跳过, 不覆盖已有模型) ──
    save_name = model.name.replace(' ', '_').replace('(', '').replace(')', '')
    if info.get('is_gca'):
        save_path = os.path.join(MODEL_DIR, 'gca_net_model.pt')
        if do_load and os.path.exists(save_path):
            pass  # 加载模式下不重复保存
        else:
            model.save(save_path)
    elif model_key == 'bert':
        save_path = os.path.join(MODEL_DIR, 'bert_model')
        model.save(save_path)
    else:
        save_path = os.path.join(MODEL_DIR, save_name + '.pkl')
        model.save(save_path)

    # ── 交叉验证 (可选, 文本模型跳过) ──
    cv_summary = None
    if do_cv and not info.get('is_gca') and model_key != 'bert' and X_train is not None:
        X_cv = sub_X[:min(10000, X_train.shape[0])]
        y_cv = sub_y[:min(10000, len(y_train))]
        print(f'\n  [交叉验证] {N_FOLDS}-Fold')
        cv_summary = cross_validate(info['factory'], X_cv, y_cv, n_folds=N_FOLDS)

    # ── 对抗评估 (可选) ──
    adv_result = None
    if do_full:
        adv_df, y_adv, X_adv = load_adversarial_testset()
        if adv_df is not None:
            # 二分类模式: 对抗数据集标签也转二分类
            if is_binary:
                y_adv = (y_adv > 0).astype(int)
                adv_df['label'] = (adv_df['label'] > 0).astype(int)
            print(f'\n  [对抗评估] 计算中...{"(二分类)" if is_binary else ""}')
            adv_texts_list = adv_df['adv_text'].tolist() if info['type'] == 'text' else None
            adv_result = evaluate_on_adversarial_testset(model, adv_df, y_adv, X_adv, adv_texts_list)

    # ── 置信度阈值 ──
    y_for_thr = sub_test_y if model_key == 'bert' else y_test
    X_for_thr = None if (model_key == 'bert' or (info['type'] == 'text' and not info.get('is_gca'))) else X_test
    texts_for_thr = sub_test_t if model_key == 'bert' else (test_texts if info['type'] == 'text' and not info.get('is_gca') else None)
    thr = evaluate_thresholds(model, X_for_thr, y_for_thr, texts=texts_for_thr)

    # ── 报告 ──
    print_single_model_report(model, r_test, adv_result, thr)

    # ── 保存 CSV ──
    import pandas as pd
    row = {**r_test}
    if adv_result: row.update(adv_result['overall'])
    if cv_summary:
        row['cv_f1_mean'] = cv_summary.get('mean_f1_macro', 0)
        row['cv_f1_std'] = cv_summary.get('std_f1_macro', 0)
    # 阈值指标 (r90/r95 只 coverage 重名, 单独处理)
    row.update(thr.get('r90', {}))
    row.update({k if k != 'coverage' else 'coverage_95': v for k, v in thr.get('r95', {}).items()})
    row['train_time_s'] = round(elapsed, 1)
    csv_path = os.path.join(OUTPUT_DIR, f'result_{save_name}.csv')
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  📄 结果已保存: {csv_path}')

    # ── 生成图像 (可选) ──
    if do_plot or do_full:
        print('\n  [可视化] 生成混淆矩阵...')
        from visualize import plot_confusion_matrix
        if model_key == 'bert':
            y_pred_img = model.predict(sub_test_t)
            plot_confusion_matrix(sub_test_y, y_pred_img, model.name)
        elif info['type'] == 'text' and not info.get('is_gca'):
            y_pred_img = model.predict(test_texts)
            plot_confusion_matrix(y_test, y_pred_img, model.name)
        else:
            y_pred_img = model.predict(X_test)
            plot_confusion_matrix(y_test, y_pred_img, model.name)
        print(f'  📊 图像已保存至: {OUTPUT_DIR}')

    # ── 进度条概览 ──
    print(f'\n  {bar}')
    print(f'  ✓ {info["name"]} 完成!')
    print(f'  {bar}\n')

    return r_test


# ===================================================================
# 入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='单模型独立运行')
    parser.add_argument('--model', '-m', required=True,
                        choices=list(REGISTRY.keys()),
                        help='模型: ' + ', '.join(REGISTRY.keys()))
    parser.add_argument('--all', action='store_true', help='含对抗测试集评估')
    parser.add_argument('--cv', action='store_true', help='含交叉验证')
    parser.add_argument('--load', action='store_true', help='加载已保存的模型（跳过训练，仅对 GCA/BERT 等有 save/load 的模型有效）')
    parser.add_argument('--plot', action='store_true', help='生成混淆矩阵图')
    parser.add_argument('--binary', action='store_true', help='二分类模式 (label 0=正常, >0=垃圾)')
    args = parser.parse_args()

    # ── 前置检查 ──
    if not check_prerequisites(do_full=args.all):
        sys.exit(1)

    # ── 加载数据 (文本类模型跳过TF-IDF, 加速加载) ──
    need_tfidf = REGISTRY[args.model]['type'] == 'tfidf'
    print(f'\n[数据] 加载中... ({"含TF-IDF" if need_tfidf else "仅文本"})')
    proc = get_prepared_data(need_tfidf=need_tfidf)

    # ── 二分类模式: 标签映射 ──
    if args.binary:
        proc['y_train'] = (proc['y_train'] > 0).astype(int)
        proc['y_val']   = (proc['y_val'] > 0).astype(int)
        proc['y_test']  = (proc['y_test'] > 0).astype(int)
        print(f'  [二分类] 标签已映射: 0=正常, 1=垃圾')
        print(f'    训练集: 正常 {sum(proc["y_train"]==0)} 条, 垃圾 {sum(proc["y_train"]==1)} 条')

    print(f'  训练集: {len(proc["train_texts"])} 条, 测试集: {len(proc["test_texts"])} 条')

    # ── 运行 ──
    try:
        run_model(args.model, proc, do_cv=args.cv, do_full=args.all, do_load=args.load, is_binary=args.binary, do_plot=args.plot)
    except Exception as e:
        print(f'\n✗ [{REGISTRY[args.model]["name"]}] 运行失败: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
