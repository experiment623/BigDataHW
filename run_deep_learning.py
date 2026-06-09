"""
深度学习模型训练 (MLP / BERT)
============================
用法: python run_deep_learning.py              # 训练MLP
      python run_deep_learning.py --bert        # 训练BERT
      python run_deep_learning.py --all         # 训练MLP+BERT
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OUTPUT_DIR, MODEL_DIR, NUM_CLASSES, BERT_MODEL_NAME,
    SUBSET_BERT_TRAIN, SUBSET_BERT_VAL, SUBSET_BERT_TEST,
)
from pipeline_base import (
    get_prepared_data, load_adversarial_testset,
    evaluate_on_adversarial_testset, evaluate_thresholds,
    print_single_model_report, subset_data,
)
from models import (
    SimpleMLP, BERTClassifier,
    evaluate_model,
)
import numpy as np


def train_mlp(proc, do_full=False):
    X_train, y_train = proc['X_train'], proc['y_train']
    X_test, y_test   = proc['X_test'], proc['y_test']

    input_dim = X_train.shape[1]
    mlp = SimpleMLP(input_dim=input_dim, num_classes=NUM_CLASSES)
    print(f'\n--- 训练 {mlp.name} (输入维度: {input_dim}) ---')
    t0 = time.time()
    mlp.fit(X_train, y_train, epochs=10, batch_size=64)
    print(f'  耗时: {time.time()-t0:.1f}s')

    r_test = evaluate_model(mlp, X_test, y_test)
    mlp.save(os.path.join(MODEL_DIR, 'TF-IDF_+_MLP.pkl'))

    adv_result = None
    if do_full:
        adv_df, y_adv, X_adv = load_adversarial_testset()
        if adv_df is not None:
            adv_result = evaluate_on_adversarial_testset(mlp, adv_df, y_adv, X_adv)

    thr = evaluate_thresholds(mlp, X_test, y_test)
    print_single_model_report(mlp, r_test, adv_result, thr)

    import pandas as pd
    row = {**r_test}
    if adv_result: row.update(adv_result['overall'])
    pd.DataFrame([row]).to_csv(
        os.path.join(OUTPUT_DIR, 'result_mlp.csv'), index=False, encoding='utf-8-sig'
    )
    return mlp, r_test


def train_bert(proc, do_full=False):
    y_train, y_val, y_test = proc['y_train'], proc['y_val'], proc['y_test']
    train_texts, val_texts, test_texts = (
        proc['train_texts'], proc['val_texts'], proc['test_texts']
    )

    # 分层采样 BERT 子集
    sub_train, sub_train_y, _ = subset_data(
        train_texts, y_train, None, SUBSET_BERT_TRAIN)
    sub_val, sub_val_y, _ = subset_data(
        val_texts, y_val, None, SUBSET_BERT_VAL)
    sub_test, sub_test_y, _ = subset_data(
        test_texts, y_test, None, SUBSET_BERT_TEST)
    print(f'  BERT 子集: train={len(sub_train)} val={len(sub_val)} test={len(sub_test)}')

    bert = BERTClassifier(model_name=BERT_MODEL_NAME, num_classes=NUM_CLASSES)
    print(f'\n--- 训练 {bert.name} ---')
    t0 = time.time()
    bert.fit(sub_train, sub_train_y, epochs=3, lr=2e-5)
    print(f'  耗时: {time.time()-t0:.1f}s')

    r_test = evaluate_model(bert, sub_test, sub_test_y)
    bert.save(os.path.join(MODEL_DIR, 'bert_model'))

    adv_result = None
    if do_full:
        adv_df, y_adv, X_adv = load_adversarial_testset()
        if adv_df is not None:
            adv_texts_list = adv_df['adv_text'].tolist()
            adv_result = evaluate_on_adversarial_testset(
                bert, adv_df, y_adv, X_adv, adv_texts_list)

    thr = evaluate_thresholds(bert, X_test=None, y=sub_test_y, texts=sub_test)
    print_single_model_report(bert, r_test, adv_result, thr)

    import pandas as pd
    row = {**r_test}
    if adv_result: row.update(adv_result['overall'])
    pd.DataFrame([row]).to_csv(
        os.path.join(OUTPUT_DIR, 'result_bert.csv'), index=False, encoding='utf-8-sig'
    )
    return bert, r_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp',  action='store_true')
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--all',  action='store_true', help='含对抗评估')
    args = parser.parse_args()

    proc = get_prepared_data()

    do_mlp = args.mlp or (not args.bert)
    do_bert = args.bert

    if do_mlp:
        train_mlp(proc, do_full=args.all)
    if do_bert:
        train_bert(proc, do_full=args.all)


if __name__ == '__main__':
    main()
