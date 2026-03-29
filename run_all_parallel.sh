#!/bin/bash

# Run all experiments in parallel across 8 GPUs

source ~/anaconda3/etc/profile.d/conda.sh
conda activate metalearning312

echo "Starting all experiments at $(date)"
echo "========================================"

# Create output directories
mkdir -p logs checkpoints_arc_cnn checkpoints_arc_transformer checkpoints_arc_protonet checkpoints_numin_lstm checkpoints_numin_transformer

# GPU 0: ARC CNN MAML (original)
echo "GPU 0: ARC CNN MAML"
CUDA_VISIBLE_DEVICES=0 python arc_1d_maml.py \
    --epochs 100 --batch_size 8 --inner_lr 0.01 --outer_lr 0.001 \
    --hidden_dim 256 --embed_dim 128 --inner_steps 5 \
    --save_dir checkpoints_arc_cnn > logs/arc_cnn.log 2>&1 &

# GPU 1: ARC Transformer MAML
echo "GPU 1: ARC Transformer MAML"
CUDA_VISIBLE_DEVICES=1 python arc_transformer_maml.py \
    --epochs 100 --inner_lr 0.01 --outer_lr 0.0005 \
    --hidden_dim 256 --embed_dim 128 --num_layers 4 --inner_steps 5 \
    --save_dir checkpoints_arc_transformer > logs/arc_transformer.log 2>&1 &

# GPU 2: ARC ProtoNet
echo "GPU 2: ARC ProtoNet"
CUDA_VISIBLE_DEVICES=2 python arc_protonet.py \
    --epochs 100 --lr 0.001 --hidden_dim 256 --embed_dim 128 \
    --save_dir checkpoints_arc_protonet > logs/arc_protonet.log 2>&1 &

# GPU 3: ARC CNN MAML with different hyperparams
echo "GPU 3: ARC CNN MAML (large)"
CUDA_VISIBLE_DEVICES=3 python arc_1d_maml.py \
    --epochs 100 --batch_size 4 --inner_lr 0.005 --outer_lr 0.0005 \
    --hidden_dim 512 --embed_dim 256 --inner_steps 10 \
    --save_dir checkpoints_arc_cnn_large > logs/arc_cnn_large.log 2>&1 &

# GPU 4: Numin LSTM MAML (original)
echo "GPU 4: Numin LSTM MAML"
CUDA_VISIBLE_DEVICES=4 python numin_maml.py \
    --epochs 100 --inner_lr 0.01 --outer_lr 0.0005 \
    --hidden_dim 256 --window_size 50 --inner_steps 5 \
    --save_dir checkpoints_numin_lstm > logs/numin_lstm.log 2>&1 &

# GPU 5: Numin Transformer MAML
echo "GPU 5: Numin Transformer MAML"
CUDA_VISIBLE_DEVICES=5 python numin_transformer.py \
    --epochs 100 --inner_lr 0.01 --outer_lr 0.0005 \
    --hidden_dim 256 --inner_steps 5 \
    --save_dir checkpoints_numin_transformer > logs/numin_transformer.log 2>&1 &

# GPU 6: Numin LSTM with different window
echo "GPU 6: Numin LSTM (window=30)"
CUDA_VISIBLE_DEVICES=6 python numin_maml.py \
    --epochs 100 --inner_lr 0.01 --outer_lr 0.0005 \
    --hidden_dim 256 --window_size 30 --inner_steps 5 \
    --save_dir checkpoints_numin_lstm_w30 > logs/numin_lstm_w30.log 2>&1 &

# GPU 7: Numin LSTM with more inner steps
echo "GPU 7: Numin LSTM (inner_steps=10)"
CUDA_VISIBLE_DEVICES=7 python numin_maml.py \
    --epochs 100 --inner_lr 0.005 --outer_lr 0.0005 \
    --hidden_dim 256 --window_size 50 --inner_steps 10 \
    --save_dir checkpoints_numin_lstm_is10 > logs/numin_lstm_is10.log 2>&1 &

echo ""
echo "All 8 experiments launched!"
echo "Monitor with: tail -f logs/*.log"
echo "Check GPU usage: nvidia-smi"
echo ""

# Wait for all background jobs
wait

echo "========================================"
echo "All experiments completed at $(date)"
