"""
Run all experiments.
"""

from experiments.exp01_main_comparison import main as exp01
from experiments.exp02_overfitting_scenarios import main as exp02
from experiments.exp03_ablation_study import main as exp03
from experiments.exp04_sensitivity_analysis import main as exp04
from experiments.exp05_efficiency_analysis import main as exp05


def main():
    exp01()
    exp02()
    exp03()
    exp04()
    exp05()


if __name__ == "__main__":
    main()