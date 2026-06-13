# ============================================================
# ChiFraud 全流程 PowerShell 脚本
# ============================================================
# 用法:
#   .\run_all.ps1                    # 首次完整训练
#   .\run_all.ps1 -Load              # 仅评估已保存模型
#   .\run_all.ps1 -Adv -SavePred     # 含对抗评估 + 预测文件
#   .\run_all.ps1 -SkipTrain         # 跳过训练（等同于 -Load）
# ============================================================
param(
    [switch]$Load,              # 跳过训练，加载已保存模型
    [switch]$SkipTrain,         # 同 -Load
    [switch]$Adv,               # 含对抗评估
    [switch]$SavePred,          # 生成预测文件
    [switch]$Ensemble,          # 运行 ensemble
    [switch]$AutoTune,          # ensemble 自动调优
    [switch]$Parallel,          # 并行运行 baseline/sota/transformer
    [switch]$DryRun,            # 仅打印命令
    [switch]$Full,              # baseline 全量训练
    [switch]$SotaWithVal,       # SOTA 使用 train+val
    [int]$TfEpochs = 2,         # Transformer epochs
    [string]$TfClassWeight = "",# Transformer class-weight 覆盖
    [string]$TfLossType = "",   # Transformer loss-type 覆盖
    [string[]]$Stages = @("baseline", "sota", "transformer", "ensemble"), # 运行阶段
    [string]$EnsName = "ensemble_auto"  # Ensemble 名称
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PYTHON) {
    $PYTHON = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $PYTHON) {
    Write-Error "未找到 Python，请确保已安装并加入 PATH"
    exit 1
}

$START_TIME = Get-Date

# 公共参数
$COMMON = @()
if ($Adv)   { $COMMON += "--adv" }
if ($SavePred) { $COMMON += "--save-predictions" }

$LOAD_FLAG = if ($Load -or $SkipTrain) { "--load" } else { "" }

function Write-Banner($msg) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "  $msg"
    Write-Host ("=" * 70)
}

function Run-Step($desc, $cmd) {
    Write-Banner $desc
    Write-Host "  命令: $($cmd -join ' ')"
    if ($DryRun) {
        Write-Host "  [DRY-RUN] 跳过执行"
        return 0
    }
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $cmd -NoNewWindow -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Host "`n  [失败] 退出码=$($proc.ExitCode): $desc" -ForegroundColor Red
    } else {
        Write-Host "`n  [完成] $desc" -ForegroundColor Green
    }
    return $proc.ExitCode
}

# ============================================================
# 1. Baseline
# ============================================================
if ($Stages -contains "baseline") {
    $cmd = @($BASELINE_SCRIPT ?? "run_baselines.py")
    if ($LOAD_FLAG) { $cmd += $LOAD_FLAG }
    else {
        $cmd += "--models", "all"
        if ($Full) { $cmd += "--full" }
    }
    $cmd += $COMMON
    $null = Run-Step "Baseline 模型 (w2v_w, w2v_c, w2v_gbdt, d2v_gbdt, gas)" $cmd
}

# ============================================================
# 2. SOTA
# ============================================================
if ($Stages -contains "sota") {
    $cmd = @("run_sota.py")
    if ($LOAD_FLAG) { $cmd += $LOAD_FLAG }
    $cmd += "--experiments", "all"
    if ($SotaWithVal) { $cmd += "--train-with-val" }
    $cmd += $COMMON
    $null = Run-Step "SOTA 模型 (8 个 sklearn Pipeline)" $cmd
}

# ============================================================
# 3. Transformer (多个配置)
# ============================================================
if ($Stages -contains "transformer") {
    # 定义 Transformer 配置列表
    $TF_CONFIGS = @(
        @{desc="macbert_train_sqrt";       args=@("--epochs","$TfEpochs","--class-weight","sqrt")},
        @{desc="macbert_+val_sqrt";        args=@("--train-with-val","--epochs","$TfEpochs","--class-weight","sqrt")},
        @{desc="macbert_+val_balanced";    args=@("--train-with-val","--epochs","$TfEpochs","--class-weight","balanced")},
        @{desc="macbert_+val_focal";       args=@("--train-with-val","--epochs","$TfEpochs","--class-weight","balanced","--loss-type","focal","--focal-gamma","1.5")},
        @{desc="macbert_+val_aug";         args=@("--train-with-val","--epochs","$TfEpochs","--class-weight","sqrt","--augment-minority","2")},
        @{desc="macbert_+val_sampler";     args=@("--train-with-val","--epochs","$TfEpochs","--class-weight","none","--sampler-weight-power","0.5")},
        @{desc="roberta_+val_sqrt";        args=@("--model-name","hfl/chinese-roberta-wwm-ext","--run-name","roberta_base","--train-with-val","--epochs","$TfEpochs","--class-weight","sqrt")}
    )

    # 参数覆盖
    if ($TfClassWeight) {
        $TF_CONFIGS = $TF_CONFIGS | ForEach-Object {
            $cfg = $_
            for ($i = 0; $i -lt $cfg.args.Count - 1; $i++) {
                if ($cfg.args[$i] -eq "--class-weight") {
                    $cfg.args[$i + 1] = $TfClassWeight
                }
            }
            $cfg
        }
    }
    if ($TfLossType) {
        $TF_CONFIGS = $TF_CONFIGS | ForEach-Object {
            $cfg = $_
            for ($i = 0; $i -lt $cfg.args.Count - 1; $i++) {
                if ($cfg.args[$i] -eq "--loss-type") {
                    $cfg.args[$i + 1] = $TfLossType
                }
            }
            $cfg
        }
    }

    $idx = 0
    foreach ($cfg in $TF_CONFIGS) {
        $idx++
        $cmd = @("run_transformer_sota.py")
        if ($LOAD_FLAG) { $cmd += $LOAD_FLAG }
        $cmd += $cfg.args
        $cmd += $COMMON
        $null = Run-Step "Transformer [$idx/$($TF_CONFIGS.Count)] $($cfg.desc)" $cmd
    }
}

# ============================================================
# 4. Ensemble
# ============================================================
if ($Ensemble -and ($Stages -contains "ensemble")) {
    $cmd = @("run_ensemble_sota.py", "--discover", "--name", $EnsName)
    if ($AutoTune) {
        $cmd += "--auto-tune"
    } else {
        $cmd += "--load-config"
    }
    if ($SavePred) { $cmd += "--save-predictions" }
    if ($Adv) { $cmd += "--adv" }
    $desc = "Ensemble 集成 ($EnsName)" + $(if ($AutoTune) { " [auto-tune]" } else { " [load-config]" })
    $null = Run-Step $desc $cmd
}

# ============================================================
# 总结
# ============================================================
$elapsed = [math]::Round(((Get-Date) - $START_TIME).TotalSeconds, 1)
Write-Banner "运行完成 (总耗时: ${elapsed}s)"
Write-Host "  输出目录: $ROOT\output\"
Write-Host "  模型目录: $ROOT\saved_models\"
Write-Host ""
