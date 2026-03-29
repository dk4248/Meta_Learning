"""
Main experiment runner for Meta-Learning Course Project.

Runs both 1D-ARC and Numin2 experiments with multi-GPU support.
"""

import subprocess
import os
import sys
import argparse
from datetime import datetime

def run_command(cmd, env=None):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=full_env
    )

    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description='Run Meta-Learning Experiments')
    parser.add_argument('--experiment', type=str, choices=['arc', 'numin', 'both'],
                       default='both', help='Which experiment to run')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                       help='Comma-separated GPU IDs to use')
    parser.add_argument('--epochs_arc', type=int, default=100, help='Epochs for ARC')
    parser.add_argument('--epochs_numin', type=int, default=50, help='Epochs for Numin')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Starting experiments at {timestamp}")
    print(f"Using GPUs: {args.gpus}")

    # Conda activation prefix
    conda_prefix = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate metalearning312 &&"

    if args.experiment in ['arc', 'both']:
        print("\n" + "="*60)
        print("EXPERIMENT 1: 1D-ARC with MAML + Hebbian TTA")
        print("="*60)

        arc_save_dir = os.path.join(base_dir, f"checkpoints_arc_{timestamp}")
        arc_cmd = f"""
        {conda_prefix} CUDA_VISIBLE_DEVICES={args.gpus.split(',')[0]} python {base_dir}/arc_1d_maml.py \\
            --data_dir {base_dir}/1D-ARC/dataset \\
            --epochs {args.epochs_arc} \\
            --batch_size 8 \\
            --inner_lr 0.01 \\
            --outer_lr 0.001 \\
            --inner_steps 5 \\
            --hidden_dim 256 \\
            --embed_dim 128 \\
            --eval_interval 10 \\
            --save_dir {arc_save_dir}
        """

        ret = run_command(arc_cmd)
        if ret != 0:
            print(f"ARC experiment failed with code {ret}")
        else:
            print(f"ARC experiment completed! Results saved to {arc_save_dir}")

    if args.experiment in ['numin', 'both']:
        print("\n" + "="*60)
        print("EXPERIMENT 2: Numin2 with MAML")
        print("="*60)

        numin_save_dir = os.path.join(base_dir, f"checkpoints_numin_{timestamp}")
        numin_cmd = f"""
        {conda_prefix} CUDA_VISIBLE_DEVICES={args.gpus.split(',')[1] if len(args.gpus.split(',')) > 1 else args.gpus.split(',')[0]} python {base_dir}/numin_maml.py \\
            --data_path {base_dir}/numin_sample.parquet \\
            --epochs {args.epochs_numin} \\
            --inner_lr 0.01 \\
            --outer_lr 0.0005 \\
            --inner_steps 5 \\
            --hidden_dim 256 \\
            --window_size 50 \\
            --support_days 5 \\
            --eval_interval 5 \\
            --save_dir {numin_save_dir}
        """

        ret = run_command(numin_cmd)
        if ret != 0:
            print(f"Numin experiment failed with code {ret}")
        else:
            print(f"Numin experiment completed! Results saved to {numin_save_dir}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
