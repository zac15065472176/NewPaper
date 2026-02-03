"""
AdaPrune 项目安装配置
"""

from setuptools import setup, find_packages

setup(
    name="adaprune",
    version="0.1.0",
    author="Your Name",
    author_email="your. email@example.com",
    description="Adaptive Pruning for Gradient Boosting Decision Trees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaprune",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "openml>=0.14.0",
        "optuna>=3.4.0",
    ],
)