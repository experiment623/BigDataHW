"""
GCA-Net (v5) 三模态联合对抗预训练
=================================
用法: python run_gca_net.py                    # 预训练+微调
      python run_gca_net.py --pretrain-only     # 仅预训练 (无标签)
      python run_gca_net.py --skip-pretrain     # 跳过预训练, 仅微调
      python run_gca_net.py --all               # 含对抗评估
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OUTPUT_DIR, MODEL_DIR, SUBSET_TRAIN_SIZE,
    GLYPH_WEIGHT, CONTRAST_WEIGHT, GCA_EPOCHS,
)
from pipeline_base import (
    get_prepared_data, load_adversarial_testset,
    evaluate_on_adversarial_testset, evaluate_thresholds,
    print_single_model_report, subset_data,
)
from models import GCANet, evaluate_model
import numpy as np


def train_gca(proc, skip_pretrain=False, pretrain_only=False, do_full=False):
    X_train, y_train = proc['X_train'], proc['y_train']
    X_test,  y_test  = proc['X_test'],  proc['y_test']
    train_texts = proc['train_texts']

    # 分层采样子集
    sub_texts, sub_y, sub_X = subset_data(
        train_texts, y_train, X_train, SUBSET_TRAIN_SIZE
    )
    print(f'  子集: {len(sub_texts)} 条')

    gca = GCANet(name='GCA-Net (Glyph+Phonetic+Semantic)', glyph_weight=GLYPH_WEIGHT)
    t0 = time.time()

    # Phase I: 三模态联合对抗预训练 (无标签)
    if not skip_pretrain:
        print(f'\n{"="*60}')
        print('  Phase I: 三模态联合对抗预训练')
        print(f'  L_tri + L_inv_sent + L_sem_confl + L_disc')
        print('='*60)
        gca.pretrain(
            texts=sub_texts,
            epochs=10,
            batch_size=32,
            lr=5e-4,
        )
        if pretrain_only:
            print(f'  预训练完成, 耗时: {time.time()-t0:.1f}s')
            return gca, None
    else:
        print('\n[跳过] 预训练阶段')

    # Phase II: 微调 (冻结主干 + 分类器)
    print(f'\n{"="*60}')
    print('  Phase II: 微调分类器 (冻结主干)')
    print('='*60)
    gca.fit(
        X_tfidf=sub_X, y=sub_y, texts=sub_texts,
        epochs=GCA_EPOCHS, batch_size=32, lr=1e-3,
        glyph_weight=GLYPH_WEIGHT, contrast_weight=CONTRAST_WEIGHT
    )
    print(f'  总耗时: {time.time()-t0:.1f}s')

    # 测试集评估
    r_test = evaluate_model(gca, X_test, y_test)
    gca.save(os.path.join(MODEL_DIR, 'gca_net_model.pt'))

    # 对抗评估 (可选)
    adv_result = None
    if do_full:
        adv_df, y_adv, X_adv = load_adversarial_testset()
        if adv_df is not None:
            adv_result = evaluate_on_adversarial_testset(gca, adv_df, y_adv, X_adv)

    thr = evaluate_thresholds(gca, X_test, y_test)
    print_single_model_report(gca, r_test, adv_result, thr)

    import pandas as pd
    row = {**r_test}
    if adv_result: row.update(adv_result['overall'])
    pd.DataFrame([row]).to_csv(
        os.path.join(OUTPUT_DIR, 'result_gca_net.csv'), index=False, encoding='utf-8-sig'
    )

    return gca, r_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain-only', action='store_true', help='仅预训练, 不做微调')
    parser.add_argument('--skip-pretrain', action='store_true', help='跳过预训练, 仅微调')
    parser.add_argument('--all', action='store_true', help='含对抗测试集评估')
    args = parser.parse_args()

    proc = get_prepared_data()
    train_gca(proc, skip_pretrain=args.skip_pretrain,
              pretrain_only=args.pretrain_only, do_full=args.all)


if __name__ == '__main__':
    main()
