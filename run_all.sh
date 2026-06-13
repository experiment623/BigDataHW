#!/usr/bin/env bash
# ============================================================
# ChiFraud 全流程 Bash 脚本
# ============================================================
# 用法:
#   bash run_all.sh                          # 首次完整训练
#   bash run_all.sh --load                   # 仅评估已保存模型
#   bash run_all.sh --adv --save-pred        # 含对抗评估 + 预测文件
#   bash run_all.sh --parallel               # 并行运行 baseline/sota/transformer
#   bash run_all.sh --stages baseline sota   # 只跑指定阶段
#   bash run_all.sh --tf-epochs 3            # 自定义 transformer epochs
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

# ── 默认参数 ──
LOAD=false
ADV=false
SAVE_PRED=false
ENSEMBLE=false
AUTO_TUNE=false
PARALLEL=false
DRY_RUN=false
FULL=false
SOTA_WITH_VAL=false
TF_EPOCHS=2
TF_CLASS_WEIGHT=""
TF_LOSS_TYPE=""
ENS_NAME="ensemble_auto"
STAGES=("baseline" "sota" "transformer" "ensemble")

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --load|--skip-train)     LOAD=true ;;
        --adv)                   ADV=true ;;
        --save-pred|--save-predictions) SAVE_PRED=true ;;
        --ensemble)              ENSEMBLE=true ;;
        --auto-tune)             AUTO_TUNE=true ;;
        --parallel)              PARALLEL=true ;;
        --dry-run)               DRY_RUN=true ;;
        --full)                  FULL=true ;;
        --sota-with-val)         SOTA_WITH_VAL=true ;;
        --tf-epochs)             TF_EPOCHS="$2"; shift ;;
        --tf-class-weight)       TF_CLASS_WEIGHT="$2"; shift ;;
        --tf-loss-type)          TF_LOSS_TYPE="$2"; shift ;;
        --ens-name)              ENS_NAME="$2"; shift ;;
        --stages)                shift; STAGES=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do STAGES+=("$1"); shift; done; continue ;;
        -h|--help)
            echo "用法: bash run_all.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --load                 跳过训练，加载已保存模型"
            echo "  --adv                  含对抗数据集评估"
            echo "  --save-pred            生成预测结果文件"
            echo "  --ensemble             运行 ensemble 集成"
            echo "  --auto-tune            ensemble 自动调优权重"
            echo "  --parallel             并行运行 baseline/sota/transformer"
            echo "  --dry-run              仅打印命令不执行"
            echo "  --full                 baseline 全量训练"
            echo "  --sota-with-val        SOTA 使用 train+val"
            echo "  --tf-epochs N          Transformer epochs (默认 2)"
            echo "  --tf-class-weight W    Transformer class-weight 覆盖"
            echo "  --tf-loss-type L       Transformer loss-type 覆盖"
            echo "  --ens-name NAME        Ensemble 名称 (默认 ensemble_auto)"
            echo "  --stages S1 S2 ...     指定运行阶段"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# ── 公共参数 ──
COMMON=()
[[ "$ADV" == true ]] && COMMON+=("--adv")
[[ "$SAVE_PRED" == true ]] && COMMON+=("--save-predictions")

LOAD_FLAG=""
[[ "$LOAD" == true ]] && LOAD_FLAG="--load"

START_TIME=$(date +%s)

banner() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}

run_step() {
    local desc="$1"
    shift
    banner "$desc"
    echo "  命令: $PYTHON $*"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY-RUN] 跳过执行"
        return 0
    fi
    $PYTHON "$@" || {
        echo ""
        echo "  [失败] $desc"
        return 1
    }
    echo ""
    echo "  [完成] $desc"
}

# ============================================================
# 并行执行函数
# ============================================================
run_parallel() {
    local pids=()
    local results=()

    if in_stages "baseline"; then
        run_baselines &
        pids+=($!)
        results+=("baseline")
    fi
    if in_stages "sota"; then
        run_sota &
        pids+=($!)
        results+=("sota")
    fi
    if in_stages "transformer"; then
        run_transformer &
        pids+=($!)
        results+=("transformer")
    fi

    local all_ok=true
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "  [并行失败] ${results[$i]}"
            all_ok=false
        fi
    done
    $all_ok
}

in_stages() {
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$1" ]] && return 0
    done
    return 1
}

# ============================================================
# 1. Baseline
# ============================================================
run_baselines() {
    local cmd=("run_baselines.py")
    if [[ -n "$LOAD_FLAG" ]]; then
        cmd+=("$LOAD_FLAG")
    else
        cmd+=("--models" "all")
        [[ "$FULL" == true ]] && cmd+=("--full")
    fi
    cmd+=("${COMMON[@]}")
    run_step "Baseline 模型 (w2v_w, w2v_c, w2v_gbdt, d2v_gbdt, gas)" "${cmd[@]}"
}

# ============================================================
# 2. SOTA
# ============================================================
run_sota() {
    local cmd=("run_sota.py")
    if [[ -n "$LOAD_FLAG" ]]; then
        cmd+=("$LOAD_FLAG")
    fi
    cmd+=("--experiments" "all")
    [[ "$SOTA_WITH_VAL" == true ]] && cmd+=("--train-with-val")
    cmd+=("${COMMON[@]}")
    run_step "SOTA 模型 (8 个 sklearn Pipeline)" "${cmd[@]}"
}

# ============================================================
# 3. Transformer (多个配置)
# ============================================================
run_transformer() {
    # 定义 Transformer 配置列表
    local TF_CONFIGS=(
        "macbert_train_sqrt|--epochs $TF_EPOCHS --class-weight sqrt"
        "macbert_+val_sqrt|--train-with-val --epochs $TF_EPOCHS --class-weight sqrt"
        "macbert_+val_balanced|--train-with-val --epochs $TF_EPOCHS --class-weight balanced"
        "macbert_+val_focal|--train-with-val --epochs $TF_EPOCHS --class-weight balanced --loss-type focal --focal-gamma 1.5"
        "macbert_+val_aug|--train-with-val --epochs $TF_EPOCHS --class-weight sqrt --augment-minority 2"
        "macbert_+val_sampler|--train-with-val --epochs $TF_EPOCHS --class-weight none --sampler-weight-power 0.5"
        "roberta_+val_sqrt|--model-name hfl/chinese-roberta-wwm-ext --run-name roberta_base --train-with-val --epochs $TF_EPOCHS --class-weight sqrt"
    )

    local idx=0
    for cfg_line in "${TF_CONFIGS[@]}"; do
        idx=$((idx + 1))
        local desc="${cfg_line%%|*}"
        local cfg_args="${cfg_line#*|}"

        # 参数覆盖
        if [[ -n "$TF_CLASS_WEIGHT" ]]; then
            cfg_args=$(echo "$cfg_args" | sed "s/--class-weight [a-z]*/--class-weight $TF_CLASS_WEIGHT/")
        fi
        if [[ -n "$TF_LOSS_TYPE" ]]; then
            cfg_args=$(echo "$cfg_args" | sed "s/--loss-type [a-z]*/--loss-type $TF_LOSS_TYPE/")
        fi

        # 将参数字符串转为数组
        local cmd=("run_transformer_sota.py")
        [[ -n "$LOAD_FLAG" ]] && cmd+=("$LOAD_FLAG")
        read -ra cfg_arr <<< "$cfg_args"
        cmd+=("${cfg_arr[@]}")
        cmd+=("${COMMON[@]}")

        local total=${#TF_CONFIGS[@]}
        run_step "Transformer [$idx/$total] $desc" "${cmd[@]}" || return 1
    done
}

# ============================================================
# 4. Ensemble
# ============================================================
run_ensemble() {
    local cmd=("run_ensemble_sota.py" "--discover" "--name" "$ENS_NAME")
    if [[ "$AUTO_TUNE" == true ]]; then
        cmd+=("--auto-tune")
    else
        cmd+=("--load-config")
    fi
    [[ "$SAVE_PRED" == true ]] && cmd+=("--save-predictions")
    [[ "$ADV" == true ]] && cmd+=("--adv")
    run_step "Ensemble 集成 ($ENS_NAME)" "${cmd[@]}"
}

# ============================================================
# 主流程
# ============================================================
echo "======================================================================"
echo "  ChiFraud 全流程运行"
echo "======================================================================"
echo "  模式: $( [[ "$LOAD" == true ]] && echo '加载已有模型' || echo '训练新模型' )"
echo "  阶段: ${STAGES[*]}"
echo "  对抗评估: $( [[ "$ADV" == true ]] && echo '是' || echo '否' )"
echo "  保存预测: $( [[ "$SAVE_PRED" == true ]] && echo '是' || echo '否' )"
echo "  Ensemble: $( [[ "$ENSEMBLE" == true ]] && echo '是' || echo '否' )"
echo "  并行: $( [[ "$PARALLEL" == true ]] && echo '是' || echo '否' )"
[[ "$DRY_RUN" == true ]] && echo "  DRY-RUN: 仅打印命令"
echo ""

if [[ "$PARALLEL" == true && "$LOAD" != true ]]; then
    run_parallel
else
    in_stages "baseline"     && run_baselines
    in_stages "sota"         && run_sota
    in_stages "transformer"  && run_transformer
    in_stages "ensemble" && [[ "$ENSEMBLE" == true ]] && run_ensemble
fi

# ── 总结 ──
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
banner "运行完成 (总耗时: ${ELAPSED}s)"
echo "  输出目录: $ROOT/output/"
echo "  模型目录: $ROOT/saved_models/"
echo ""
