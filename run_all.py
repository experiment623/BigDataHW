"""
ChiFraud 全流程一键运行脚本
============================
顺序运行 Baseline → SOTA → Transformer → Ensemble，
每个阶段生成标准测试集 + 对抗数据集预测，自动保存模型权重和评估指标。

用法:
  # 完整训练（推荐首次运行）
  python run_all.py

  # 跳过训练，仅用已保存模型评估
  python run_all.py --load

  # 并行运行（Windows）
  python run_all.py --parallel

  # 自定义 Transformer 配置
  python run_all.py --tf-epochs 3 --tf-train-with-val --tf-class-weight balanced

  # 运行指定阶段
  python run_all.py --stages baseline sota

  # 完整训练 + 对抗评估 + 预测文件 + ensemble
  python run_all.py --adv --save-predictions --ensemble
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# 各 runner 脚本路径
BASELINE_SCRIPT = ROOT / "run_baselines.py"
SOTA_SCRIPT = ROOT / "run_sota.py"
TRANSFORMER_SCRIPT = ROOT / "run_transformer_sota.py"
ENSEMBLE_SCRIPT = ROOT / "run_ensemble_sota.py"

# 默认 Transformer 多配置：不同训练方式产生不同模型
DEFAULT_TF_CONFIGS = [
    {
        "desc": "macbert_train_sqrt",
        "args": ["--epochs", "2", "--class-weight", "sqrt"],
    },
    {
        "desc": "macbert_+val_sqrt",
        "args": ["--train-with-val", "--epochs", "2", "--class-weight", "sqrt"],
    },
    {
        "desc": "macbert_+val_balanced",
        "args": ["--train-with-val", "--epochs", "2", "--class-weight", "balanced"],
    },
    {
        "desc": "macbert_+val_focal",
        "args": ["--train-with-val", "--epochs", "2", "--class-weight", "balanced",
                 "--loss-type", "focal", "--focal-gamma", "1.5"],
    },
    {
        "desc": "macbert_+val_aug",
        "args": ["--train-with-val", "--epochs", "2", "--class-weight", "sqrt",
                 "--augment-minority", "2"],
    },
    {
        "desc": "macbert_+val_sampler",
        "args": ["--train-with-val", "--epochs", "2", "--class-weight", "none",
                 "--sampler-weight-power", "0.5"],
    },
    # RoBERTa 变体
    {
        "desc": "roberta_+val_sqrt",
        "args": ["--model-name", "hfl/chinese-roberta-wwm-ext",
                 "--run-name", "roberta_base",
                 "--train-with-val", "--epochs", "2", "--class-weight", "sqrt"],
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description="ChiFraud 全流程一键运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all.py                                          # 首次完整训练
  python run_all.py --load                                   # 仅评估已保存模型
  python run_all.py --parallel                               # 并行运行各阶段
  python run_all.py --stages baseline sota                   # 只跑指定阶段
  python run_all.py --tf-epochs 3 --tf-configs 1 3           # 只跑指定 TF 配置
  python run_all.py --adv --save-predictions --ensemble      # 含对抗+预测+集成
        """,
    )
    # 阶段控制
    p.add_argument("--stages", nargs="+",
                   choices=["baseline", "sota", "transformer", "ensemble"],
                   default=["baseline", "sota", "transformer", "ensemble"],
                   help="要运行的阶段（默认全部）")
    p.add_argument("--load", action="store_true",
                   help="跳过训练，加载已保存模型直接评估")

    # 运行模式
    p.add_argument("--parallel", action="store_true",
                   help="并行运行 baseline/sota/transformer（注意 GPU 显存冲突）")
    p.add_argument("--dry-run", action="store_true",
                   help="仅打印要执行的命令，不实际运行")

    # 公共开关
    p.add_argument("--adv", action="store_true",
                   help="所有阶段含对抗数据集评估")
    p.add_argument("--save-predictions", action="store_true",
                   help="所有阶段生成预测结果文件")
    p.add_argument("--ensemble", action="store_true",
                   help="在最后运行 ensemble 集成")

    # Baseline 参数
    p.add_argument("--bl-full", action="store_true",
                   help="Baseline 使用全量训练集（否则用 30000 子集）")

    # SOTA 参数
    p.add_argument("--sota-train-with-val", action="store_true",
                   help="SOTA 使用 train+val 训练")

    # Transformer 参数
    p.add_argument("--tf-configs", nargs="+", type=int, default=None,
                   help="指定 Transformer 配置索引（从1开始），默认全部。见脚本内 DEFAULT_TF_CONFIGS")
    p.add_argument("--tf-list-configs", action="store_true",
                   help="列出所有 Transformer 配置并退出")
    p.add_argument("--tf-epochs", type=int, default=None,
                   help="覆盖所有 TF 配置的 epochs")
    p.add_argument("--tf-train-with-val", action="store_true", default=None,
                   help="覆盖所有 TF 配置的 --train-with-val")
    p.add_argument("--tf-class-weight", type=str, default=None,
                   choices=["none", "sqrt", "balanced", "effective"],
                   help="覆盖所有 TF 配置的 --class-weight")
    p.add_argument("--tf-loss-type", type=str, default=None,
                   choices=["ce", "focal"],
                   help="覆盖所有 TF 配置的 --loss-type")

    # Ensemble 参数
    p.add_argument("--ens-name", default="ensemble_auto",
                   help="Ensemble 名称（默认 ensemble_auto）")
    p.add_argument("--ens-auto-tune", action="store_true",
                   help="Ensemble 自动调优权重和因子")
    p.add_argument("--ens-objective", choices=["sota_margin", "balanced_pr", "f1_macro"],
                   default="sota_margin")
    p.add_argument("--ens-search-rounds", type=int, default=3)
    p.add_argument("--ens-random-trials", type=int, default=400)

    # 输出
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录（默认 output/）")
    return p.parse_args()


def list_tf_configs():
    """列出所有 Transformer 配置"""
    print("\n可用的 Transformer 配置:")
    print("-" * 70)
    for i, cfg in enumerate(DEFAULT_TF_CONFIGS, 1):
        print(f"  [{i}] {cfg['desc']}")
        print(f"      {' '.join(cfg['args'])}")
    print()


def get_tf_configs(args) -> list[dict]:
    """获取要运行的 Transformer 配置列表，支持覆盖参数"""
    if args.tf_list_configs:
        list_tf_configs()
        sys.exit(0)

    indices = args.tf_configs
    if indices is None:
        configs = DEFAULT_TF_CONFIGS
    else:
        configs = [DEFAULT_TF_CONFIGS[i - 1] for i in indices if 1 <= i <= len(DEFAULT_TF_CONFIGS)]

    # 应用覆盖参数
    for cfg in configs:
        cfg_args = cfg["args"]
        if args.tf_epochs is not None:
            # 找到 --epochs 并替换
            for i, a in enumerate(cfg_args):
                if a == "--epochs" and i + 1 < len(cfg_args):
                    cfg_args[i + 1] = str(args.tf_epochs)
                    break
            else:
                cfg_args.extend(["--epochs", str(args.tf_epochs)])
        if args.tf_train_with_val is not None:
            if args.tf_train_with_val and "--train-with-val" not in cfg_args:
                cfg_args.append("--train-with-val")
            elif not args.tf_train_with_val:
                cfg_args = [a for a in cfg_args if a != "--train-with-val"]
        if args.tf_class_weight is not None:
            for i, a in enumerate(cfg_args):
                if a == "--class-weight" and i + 1 < len(cfg_args):
                    cfg_args[i + 1] = args.tf_class_weight
                    break
        if args.tf_loss_type is not None:
            for i, a in enumerate(cfg_args):
                if a == "--loss-type" and i + 1 < len(cfg_args):
                    cfg_args[i + 1] = args.tf_loss_type
                    break
        cfg["args"] = cfg_args

    return configs


def build_common_args(args) -> list[str]:
    """构建所有脚本的公共参数"""
    common = []
    if args.adv:
        common.append("--adv")
    if args.save_predictions:
        common.append("--save-predictions")
    return common


def run_cmd(cmd: list[str], desc: str, dry_run: bool = False) -> int:
    """运行一条命令，返回退出码"""
    print(f"\n{'=' * 70}")
    print(f"  {desc}")
    print(f"{'=' * 70}")
    print(f"  命令: {' '.join(cmd)}")
    if dry_run:
        print("  [DRY-RUN] 跳过执行")
        return 0
    print()
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n  [失败] 退出码={result.returncode}: {desc}")
    else:
        print(f"\n  [完成] {desc}")
    return result.returncode


def run_baselines(args) -> int:
    """运行 Baseline 模型（不支持 --save-predictions，仅传 --adv）"""
    cmd = [PYTHON, str(BASELINE_SCRIPT)]
    if args.load:
        cmd.append("--load")
    else:
        cmd.extend(["--models", "all"])
        if args.bl_full:
            cmd.append("--full")
    if args.adv:
        cmd.append("--adv")
    return run_cmd(cmd, "Baseline 模型 (w2v_w, w2v_c, w2v_gbdt, d2v_gbdt, gas)", args.dry_run)


def run_sota(args) -> int:
    """运行 SOTA 模型"""
    common = build_common_args(args)
    cmd = [PYTHON, str(SOTA_SCRIPT)]
    if args.load:
        cmd.append("--load")
    cmd.extend(["--experiments", "all"])
    if args.sota_train_with_val:
        cmd.append("--train-with-val")
    cmd.extend(common)
    return run_cmd(cmd, "SOTA 模型 (8 个 sklearn Pipeline)", args.dry_run)


def run_transformer(args) -> int:
    """运行 Transformer 模型（多个配置）"""
    configs = get_tf_configs(args)
    if not configs:
        print("  [跳过] 没有 Transformer 配置")
        return 0

    common = build_common_args(args)
    exit_code = 0

    for i, cfg in enumerate(configs):
        cmd = [PYTHON, str(TRANSFORMER_SCRIPT)]
        if args.load:
            cmd.append("--load")
        cmd.extend(cfg["args"])
        cmd.extend(common)

        desc = f"Transformer [{i + 1}/{len(configs)}] {cfg['desc']}"
        rc = run_cmd(cmd, desc, args.dry_run)
        if rc != 0:
            exit_code = rc

    return exit_code


def run_ensemble(args) -> int:
    """运行 Ensemble 集成"""
    cmd = [PYTHON, str(ENSEMBLE_SCRIPT)]
    cmd.extend(["--discover", "--name", args.ens_name])
    if args.ens_auto_tune:
        cmd.extend(["--auto-tune", "--objective", args.ens_objective,
                    "--search-rounds", str(args.ens_search_rounds),
                    "--random-trials", str(args.ens_random_trials)])
    else:
        cmd.append("--load-config")
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.adv:
        cmd.append("--adv")

    desc = f"Ensemble 集成 ({args.ens_name})" + (" [auto-tune]" if args.ens_auto_tune else " [load-config]")
    return run_cmd(cmd, desc, args.dry_run)


def print_summary(start_time: float, results: dict[str, int]):
    """打印运行总结"""
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  运行总结 (总耗时: {elapsed:.1f}s)")
    print(f"{'=' * 70}")
    for stage, code in results.items():
        status = "✓ 成功" if code == 0 else f"✗ 失败(code={code})"
        print(f"  {stage:<20} {status}")
    print(f"\n  输出目录: {ROOT / 'output'}/")
    print(f"  模型目录: {ROOT / 'saved_models'}/")
    print(f"{'=' * 70}")


def main():
    args = parse_args()

    if args.tf_list_configs:
        list_tf_configs()
        return

    start_time = time.time()
    results: dict[str, int] = {}

    # 打印运行配置
    print("=" * 70)
    print("  ChiFraud 全流程运行")
    print("=" * 70)
    print(f"  模式: {'加载已有模型' if args.load else '训练新模型'}")
    print(f"  阶段: {', '.join(args.stages)}")
    print(f"  对抗评估: {'是' if args.adv else '否'}")
    print(f"  保存预测: {'是' if args.save_predictions else '否'}")
    print(f"  Ensemble: {'是' if args.ensemble else '否'}")
    print(f"  并行: {'是' if args.parallel else '否'}")
    if args.dry_run:
        print(f"  DRY-RUN: 仅打印命令")
    print()

    # ── 并行模式 ──
    if args.parallel and not args.load:
        import concurrent.futures

        def _run_with_return(fn, *a, **kw):
            return fn(*a, **kw)

        tasks = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            if "baseline" in args.stages:
                tasks["baseline"] = executor.submit(_run_with_return, run_baselines, args)
            if "sota" in args.stages:
                tasks["sota"] = executor.submit(_run_with_return, run_sota, args)
            if "transformer" in args.stages:
                tasks["transformer"] = executor.submit(_run_with_return, run_transformer, args)

            for name, future in tasks.items():
                results[name] = future.result()

        # 并行结束后跑 ensemble
        if args.ensemble and "ensemble" in args.stages:
            results["ensemble"] = run_ensemble(args)

    # ── 顺序模式 ──
    else:
        stage_runners = {
            "baseline": run_baselines,
            "sota": run_sota,
            "transformer": run_transformer,
            "ensemble": run_ensemble if args.ensemble else lambda _: 0,
        }

        for stage in args.stages:
            if stage == "ensemble" and not args.ensemble:
                continue
            results[stage] = stage_runners[stage](args)

    print_summary(start_time, results)


if __name__ == "__main__":
    main()
