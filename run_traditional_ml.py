"""
传统 ML 模型训练 (LR / SVC / RF / NB)
=======================================
用法: python run_traditional_ml.py           # 训练全部4个
      python run_traditional_ml.py --lr      # 仅逻辑回归
      python run_traditional_ml.py --all     # 完整评估(含对抗+阈值)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, MODEL_DIR, RANDOM_SEED
from pipeline_base import (
    get_prepared_data, load_adversarial_testset,
    evaluate_on_adversarial_testset, evaluate_thresholds,
    print_single_model_report,
)
from models import (
    TfidfLogisticRegression, TfidfLinearSVC,
    TfidfRandomForest, TfidfNaiveBayes,
    evaluate_model,
)
import numpy as np; np.random.seed(RANDOM_SEED)

# ── 模型注册 ──
MODELS = {
    'lr':  TfidfLogisticRegression(),
    'svc': TfidfLinearSVC(),
    'rf':  TfidfRandomForest(),
    'nb':  TfidfNaiveBayes(),
}


def train_and_eval(model, proc, do_full=False):
    X_train, y_train = proc['X_train'], proc['y_train']
    X_test, y_test   = proc['X_test'], proc['y_test']

    print(f'\n--- 训练 {model.name} ---')
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f'  耗时: {time.time()-t0:.1f}s')

    # 测试集评估
    r_test = evaluate_model(model, X_test, y_test)

    # 保存
    model.save(os.path.join(MODEL_DIR, model.name.replace(' ', '_') + '.pkl'))

    # ── 完整评估 (可选) ──
    adv_result = None
    if do_full:
        adv_df, y_adv, X_adv = load_adversarial_testset()
        if adv_df is not None:
            adv_result = evaluate_on_adversarial_testset(model, adv_df, y_adv, X_adv)
        thr = evaluate_thresholds(model, X_test, y_test)
    else:
        thr = evaluate_thresholds(model, X_test, y_test) if do_full else None

    print_single_model_report(model, r_test, adv_result, thr)

    # 保存结果CSV
    row = {**r_test}
    if adv_result: row.update(adv_result['overall'])
    pd.DataFrame([row]).to_csv(
        os.path.join(OUTPUT_DIR, f'result_{model.name.replace(" ","_")}.csv'),
        index=False, encoding='utf-8-sig'
    )
    return r_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',   action='store_true')
    parser.add_argument('--svc',  action='store_true')
    parser.add_argument('--rf',   action='store_true')
    parser.add_argument('--nb',   action='store_true')
    parser.add_argument('--all',  action='store_true', help='含对抗评估')
    args = parser.parse_args()

    any_flag = args.lr or args.svc or args.rf or args.nb
    targets = [k for k, v in
               [('lr', args.lr), ('svc', args.svc), ('rf', args.rf), ('nb', args.nb)]
               if v]

    proc = get_prepared_data()
    
    if any_flag:
        for k in targets:
            train_and_eval(MODELS[k], proc, do_full=args.all)
    else:
        for k in ['lr', 'svc', 'rf', 'nb']:
            train_and_eval(MODELS[k], proc, do_full=args.all)

if __name__ == '__main__':
    import pandas as pd  # for CSV
    main()
