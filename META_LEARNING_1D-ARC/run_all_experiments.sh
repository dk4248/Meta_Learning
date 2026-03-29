#!/bin/bash
# Master script to run ALL meta-learning experiments across 8 A100 GPUs
set +e  # Don't exit on individual experiment failure
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "META-LEARNING EXPERIMENTS - PARALLEL LAUNCH"
echo "Start time: $(date)"
echo "Using 8 A100 GPUs"
echo "============================================================"

PIDS=()
NAMES=()

launch() {
    local gpu=$1
    local script=$2
    local args=$3
    local name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[GPU $gpu] Launching $name → $logfile"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/$script" $args > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
}

# =============================================
# ARC EXPERIMENTS (GPUs 0-4)
# =============================================

# GPU 0: Reptile variants
launch 0 arc_reptile.py      "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --inner_steps 10 --save_dir $SCRIPT_DIR/checkpoints_arc_reptile --gpu 0"    "arc_reptile"
launch 0 arc_reptile_is20.py "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 150 --inner_steps 20 --save_dir $SCRIPT_DIR/checkpoints_arc_reptile_is20 --gpu 0" "arc_reptile_is20"

# GPU 1: FOMAML + ANIL
launch 1 arc_fomaml.py "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --inner_steps 5 --save_dir $SCRIPT_DIR/checkpoints_arc_fomaml --gpu 0" "arc_fomaml"
launch 1 arc_anil.py   "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --inner_steps 5 --save_dir $SCRIPT_DIR/checkpoints_arc_anil --gpu 0"   "arc_anil"

# GPU 2: ProtoNet + CNP
launch 2 arc_protonet.py "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_arc_protonet --gpu 0" "arc_protonet"
launch 2 arc_cnp.py      "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_arc_cnp --gpu 0"      "arc_cnp"

# GPU 3: Matching + Transformer MAML
launch 3 arc_matching.py         "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_arc_matching --gpu 0"    "arc_matching"
launch 3 arc_transformer_maml.py "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --inner_steps 5 --save_dir $SCRIPT_DIR/checkpoints_arc_transformer --gpu 0" "arc_transformer_maml"

# GPU 4: MAML
launch 4 arc_1d_maml.py "--data_dir $SCRIPT_DIR/1D-ARC/dataset --epochs 100 --inner_steps 5 --batch_size 4 --save_dir $SCRIPT_DIR/checkpoints_arc --gpu 0" "arc_1d_maml"

# =============================================
# NUMIN EXPERIMENTS (GPUs 4-7)
# =============================================

# GPU 4: MAML (shares with ARC MAML)
launch 4 numin_maml.py "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 50 --inner_steps 3 --save_dir $SCRIPT_DIR/checkpoints_numin --gpu 0" "numin_maml"

# GPU 5: Reptile family
launch 5 numin_reptile.py            "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_reptile --gpu 0"    "numin_reptile"
launch 5 numin_reptile_aggressive.py "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_aggressive --gpu 0" "numin_reptile_aggressive"
launch 5 numin_reptile_augmented.py  "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_augmented --gpu 0"  "numin_reptile_augmented"

# GPU 6: FOMAML + ANIL + CNP
launch 6 numin_fomaml.py "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --inner_steps 5 --save_dir $SCRIPT_DIR/checkpoints_numin_fomaml --gpu 0" "numin_fomaml"
launch 6 numin_anil.py   "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 50 --inner_steps 3 --save_dir $SCRIPT_DIR/checkpoints_numin_anil --gpu 0"   "numin_anil"
launch 6 numin_cnp.py    "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_cnp --gpu 0"                     "numin_cnp"

# GPU 7: ProtoNet + Ensemble + Ensemble Seeds
launch 7 numin_protonet.py       "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_protonet --gpu 0"        "numin_protonet"
launch 7 numin_ensemble.py       "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_ensemble --gpu 0"        "numin_ensemble"
launch 7 numin_ensemble_seeds.py "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 100 --save_dir $SCRIPT_DIR/checkpoints_numin_ensemble_seeds --gpu 0"  "numin_ensemble_seeds"

# GPU 2: Attention + Transformer (share with ARC ProtoNet/CNP)
launch 2 numin_attention.py   "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 50 --inner_steps 3 --save_dir $SCRIPT_DIR/checkpoints_numin_attention --gpu 0"   "numin_attention"
launch 2 numin_transformer.py "--data_path $SCRIPT_DIR/numin_sample.parquet --epochs 50 --inner_steps 3 --save_dir $SCRIPT_DIR/checkpoints_numin_transformer --gpu 0" "numin_transformer"

echo ""
echo "============================================================"
echo "All ${#PIDS[@]} experiments launched. Waiting for completion..."
echo "Monitor with: tail -f logs/<experiment>.log"
echo "============================================================"
echo ""

# Wait for all processes and report results
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    if wait $pid; then
        echo "[DONE] $name (PID $pid) - SUCCESS"
    else
        echo "[FAIL] $name (PID $pid) - exit code: $?"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "All experiments finished. $FAILED failures out of ${#PIDS[@]}."
echo "End time: $(date)"
echo "============================================================"

# Generate plots and report
echo ""
echo "Generating plots and report..."
cd "$SCRIPT_DIR"
python plot_arc_curves.py
python plot_numin_curves.py
python generate_report.py
python per_task_analysis.py --gpu 0

echo ""
echo "============================================================"
echo "ALL DONE!"
echo "============================================================"
