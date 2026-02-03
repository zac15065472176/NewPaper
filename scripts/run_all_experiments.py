"""
批量运行所有实验
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'


def run_experiment(script_name:  str):
    """运行单个实验脚本"""
    script_path = EXPERIMENTS_DIR / script_name
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys. executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """运行所有实验"""
    print("="*60)
    print("AdaPrune - Run All Experiments")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    experiments = [
        'exp01_main_comparison. py',
        'exp02_overfitting_scenarios.py',
        'exp03_ablation_study.py',
    ]
    
    results = {}
    
    for exp in experiments:
        if (EXPERIMENTS_DIR / exp).exists():
            results[exp] = run_experiment(exp)
        else: 
            print(f"⚠ {exp} not found, skipping...")
            results[exp] = None
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for exp, success in results.items():
        if success is None:
            status = "⚠ SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else: 
            status = "✗ FAILED"
        print(f"  {exp}: {status}")
    
    print("\nResults saved to: results/")


if __name__ == "__main__":
    main()