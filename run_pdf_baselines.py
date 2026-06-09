"""
PDF 论文 Baseline 模型训练 (Word2Vec / Doc2Vec / GAS)
=====================================================
用法: python run_pdf_baselines.py               # 训练全部5个
      python run_pdf_baselines.py --w2v_w        # 仅Word2Vec-w+LR
      python run_pdf_baselines.py --gas          # 仅GAS
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, MODEL_DIR, SUBSET_TRAIN_SIZE
from pipeline_base import (
    get_prepared_data, load_adversarial_testset,
    evaluate_on_adversarial_testset, evaluate_thresholds,
    print_single_model_report, subset_data,
)
from models import (
    Word2VecWordLR, Word2VecCharLR,
    Word2VecCharGBDT, Doc2VecCharGBDT, GAS,
    evaluate_model,
)
import numpy as np

MODELS_MAP = {
    'w2v_w':    Word2VecWordLR(vec_dim=200),
    'w2v_c':    Word2VecCharLR(vec_dim=200),
    'w2v_gbdt': Word2VecCharGBDT(vec_dim=200),
    'd2v_gbdt': Doc2VecCharGBDT(vec_dim=200),
    'gas':      GAS(hidden_dim=128),
}


def train_and_eval(model, proc, sub_texts, sub_y, sub_X, do_full=False):
    train_texts, val_texts = proc['train_texts'], proc['val_texts']
    y_val = proc['y_val']
    X_val, y_test = proc['X_val'], proc['y_test']
    X_test, test_texts = proc['X_test'], proc['test_texts']

    print(f'\n--- 训练 {model.name} ---')
    t0 = time.time()

    try:
        if 'GAS' in model.name:
            model.fit(sub_X, sub_y)
            r_val = evaluate_model(model, X_val, y_val)
            r_test = evaluate_model(model, X_test, y_test)
        else:
            model.fit(sub_texts, sub_y)
            r_val = evaluate_model(model, val_texts, y_val)
            r_test = evaluate_model(model, test_texts, y_test)

        print(f'  耗时: {time.time()-t0:.1f}s')

        model.save(os.path.join(MODEL_DIR, model.name.replace(' ', '_') + '.pkl'))

        adv_result = None
        if do_full:
            adv_df, y_adv, X_adv = load_adversarial_testset()
            if adv_df is not None:
                adv_texts_list = adv_df['adv_text'].tolist()
                adv_result = evaluate_on_adversarial_testset(
                    model, adv_df, y_adv, X_adv, adv_texts_list)

        thr = evaluate_thresholds(model, X_test, y_test, test_texts)
        print_single_model_report(model, r_test, adv_result, thr)

        import pandas as pd
        row = {**r_test}
        if adv_result: row.update(adv_result['overall'])
        pd.DataFrame([row]).to_csv(
            os.path.join(OUTPUT_DIR, f'result_{model.name.replace(" ","_")}.csv'),
            index=False, encoding='utf-8-sig'
        )
        return r_test

    except Exception as e:
        print(f'  [失败] {model.name}: {e}')
        import traceback; traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    for k in MODELS_MAP: parser.add_argument(f'--{k}', action='store_true')
    parser.add_argument('--all', action='store_true', help='含对抗评估')
    args = parser.parse_args()

    proc = get_prepared_data()
    sub_texts, sub_y, sub_X = subset_data(
        proc['train_texts'], proc['y_train'], proc['X_train'], SUBSET_TRAIN_SIZE
    )
    print(f'  子集: {len(sub_texts)} 条')

    any_flag = any(getattr(args, k) for k in MODELS_MAP)
    if any_flag:
        for k in MODELS_MAP:
            if getattr(args, k):
                train_and_eval(MODELS_MAP[k], proc, sub_texts, sub_y, sub_X, args.all)
    else:
        for k in MODELS_MAP:
            train_and_eval(MODELS_MAP[k], proc, sub_texts, sub_y, sub_X, args.all)


if __name__ == '__main__':
    main()
