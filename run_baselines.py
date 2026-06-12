"""
论文 5 个 Baseline 统一运行脚本
===============================
用法:
  python run_baselines.py                        # 训练全部5个(子集)
  python run_baselines.py --models w2v_c gas     # 指定模型
  python run_baselines.py --full                 # 全量训练集
  python run_baselines.py --adv                  # 含对抗评估
"""
from __future__ import annotations
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from config import (
    DATASET_DIR, OUTPUT_DIR, RANDOM_SEED, NUM_CLASSES, LABEL_MAP,
    SUBSET_BASELINE_TRAIN,
)
from data_processor import load_data, stratified_sample, load_adversarial_data, build_tfidf_vectorizer, preprocess_texts
from models import (
    Word2VecWordLR, Word2VecCharLR, Word2VecCharGBDT, Doc2VecCharGBDT, GAS,
    compute_metrics_full, save_predictions_csv, save_metrics_csv,
)
np.random.seed(RANDOM_SEED)

MODELS_MAP = {
    'w2v_w':    ('Word2Vec-w+LR',     lambda: Word2VecWordLR(vec_dim=200)),
    'w2v_c':    ('Word2Vec-c+LR',     lambda: Word2VecCharLR(vec_dim=200)),
    'w2v_gbdt': ('Word2Vec-c+GBDT',   lambda: Word2VecCharGBDT(vec_dim=200)),
    'd2v_gbdt': ('Doc2Vec-c+GBDT',    lambda: Doc2VecCharGBDT(vec_dim=200)),
    'gas':      ('GAS (GCN)',         lambda: GAS(hidden_dim=128)),
}


def evaluate_and_save(model, model_key, test_texts, y_test, adv_df, adv_texts, y_adv,
                      tfidf_vec, out_dir, train_time):
    """统一评估：测试集 + 对抗集"""
    is_gas = (model_key == 'gas')

    results = {}

    # ── 测试集评估 ──
    if is_gas:
        X_test_t = tfidf_vec.transform(preprocess_texts(test_texts))
        y_pred = model.predict(X_test_t)
        proba = model.predict_proba(X_test_t)
    else:
        y_pred = model.predict(test_texts)
        proba = model.predict_proba(test_texts)
    conf = np.max(proba, axis=1)

    metrics = compute_metrics_full(model.name, y_test, y_pred, proba, train_time_s=train_time)
    save_predictions_csv(test_texts, y_test, y_pred, conf,
                         os.path.join(out_dir, 'test_results.csv'))
    save_metrics_csv(metrics, os.path.join(out_dir, 'test_metrics.csv'))
    results['test'] = metrics

    # ── 对抗集评估 ──
    if adv_df is not None and len(adv_texts) > 0:
        if is_gas:
            X_adv_t = tfidf_vec.transform(preprocess_texts(adv_texts))
            y_adv_pred = model.predict(X_adv_t)
            adv_proba = model.predict_proba(X_adv_t)
        else:
            y_adv_pred = model.predict(adv_texts)
            adv_proba = model.predict_proba(adv_texts)
        adv_conf = np.max(adv_proba, axis=1)

        adv_metrics = compute_metrics_full(model.name, y_adv, y_adv_pred, adv_proba)
        save_predictions_csv(adv_texts, y_adv, y_adv_pred, adv_conf,
                             os.path.join(out_dir, 'adversarial_results.csv'))
        save_metrics_csv(adv_metrics, os.path.join(out_dir, 'adversarial_metrics.csv'))
        results['adversarial'] = adv_metrics

    return results


def run_baseline(model_key, train_texts, y_train, val_texts, y_val,
                 test_texts, y_test, adv_df, adv_texts, y_adv,
                 use_full=False, adv_eval=False):
    display_name, factory = MODELS_MAP[model_key]
    out_dir = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)

    is_gas = (model_key == 'gas')

    # 子集采样 (GAS 也采样加速)
    if not use_full:
        sub_texts, sub_y = stratified_sample(train_texts, y_train, SUBSET_BASELINE_TRAIN)
        print(f'\n[{display_name}] 子集采样: {len(sub_texts)} 条')
    else:
        sub_texts, sub_y = train_texts, y_train
        print(f'\n[{display_name}] 全量训练: {len(sub_texts)} 条')

    # 构建 TF-IDF (GAS 需要)
    tfidf_vec = None
    if is_gas:
        print(f'  构建 TF-IDF...')
        tfidf_vec = build_tfidf_vectorizer(preprocess_texts(sub_texts),
                                           save_path=os.path.join(out_dir, 'tfidf_vec.pkl'))
        sub_X = tfidf_vec.transform(preprocess_texts(sub_texts))

    model = factory()
    print(f'  训练中...')
    t0 = time.time()

    try:
        if is_gas:
            model.fit(sub_X, sub_y, epochs=50, lr=1e-2)
        else:
            model.fit(sub_texts, sub_y)
    except Exception as e:
        print(f'  [失败] {e}')
        import traceback; traceback.print_exc()
        return None

    train_time = time.time() - t0
    print(f'  训练耗时: {train_time:.1f}s')

    # 评估
    if not adv_eval:
        adv_df, adv_texts, y_adv = None, [], np.array([])
    results = evaluate_and_save(model, model_key, test_texts, y_test,
                                adv_df, adv_texts, y_adv,
                                tfidf_vec, out_dir, train_time)

    # 打印摘要
    m = results['test']
    print(f'  测试集 -> Acc={m["accuracy"]:.4f} F1={m["f1_macro"]:.4f} '
          f'F1@90={m["f1@90"]:.4f} F1@95={m["f1@95"]:.4f}')
    if 'adversarial' in results:
        am = results['adversarial']
        print(f'  对抗集 -> Acc={am["accuracy"]:.4f} F1={am["f1_macro"]:.4f}')

    return results


def main():
    parser = argparse.ArgumentParser(description='运行论文 5 个 Baseline')
    parser.add_argument('--models', nargs='+', default=['all'],
                        choices=['all', 'w2v_w', 'w2v_c', 'w2v_gbdt', 'd2v_gbdt', 'gas'])
    parser.add_argument('--full', action='store_true', help='使用全量训练集')
    parser.add_argument('--adv', action='store_true', help='含对抗评估')
    args = parser.parse_args()

    models_to_run = list(MODELS_MAP.keys()) if args.models == ['all'] else args.models

    print('=' * 60)
    print(f'  Baseline 模型训练 [{len(models_to_run)} 个]')
    print('=' * 60)

    # 加载数据
    train_texts, y_train = load_data(os.path.join(DATASET_DIR, 'ChiFraud_train.csv'))
    val_texts, y_val = load_data(os.path.join(DATASET_DIR, 'ChiFraud_t2022.csv'))
    test_texts, y_test = load_data(os.path.join(DATASET_DIR, 'ChiFraud_t2023.csv'))
    print(f'数据: Train={len(train_texts)} Val={len(val_texts)} Test={len(test_texts)}')

    adv_df, adv_texts, y_adv = None, [], np.array([])
    if args.adv:
        adv_df, adv_texts, y_adv = load_adversarial_data()

    all_test_metrics = []

    for mkey in models_to_run:
        res = run_baseline(mkey, train_texts, y_train, val_texts, y_val,
                          test_texts, y_test, adv_df, adv_texts, y_adv,
                          use_full=args.full, adv_eval=args.adv)
        if res:
            all_test_metrics.append(res['test'])

    # 汇总
    if all_test_metrics:
        print('\n' + '=' * 60)
        print('  Baseline 模型对比汇总')
        print('=' * 60)
        df = pd.DataFrame(all_test_metrics)
        cols = ['model', 'accuracy', 'f1_macro', 'f1@90', 'coverage@90', 'f1@95', 'coverage@95']
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        df.to_csv(os.path.join(OUTPUT_DIR, 'baselines_summary.csv'), index=False, encoding='utf-8-sig')
        print(f'\n汇总已保存: output/baselines_summary.csv')


if __name__ == '__main__':
    main()
