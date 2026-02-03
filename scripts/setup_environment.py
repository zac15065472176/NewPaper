"""
环境检查脚本
"""

import sys
import platform


def check_environment():
    """检查运行环境"""
    print("="*60)
    print("AdaPrune Environment Check")
    print("="*60)
    
    # Python版本
    print(f"\nPython: {sys.version}")
    
    # 系统信息
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    
    # 检查依赖
    dependencies = [
        'numpy', 'pandas', 'scipy', 'sklearn',
        'xgboost', 'lightgbm', 'matplotlib', 'seaborn',
        'tqdm', 'joblib', 'openml'
    ]
    
    print("\nDependencies:")
    missing = []
    
    for dep in dependencies:
        try: 
            if dep == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {dep}: {version}")
        except ImportError: 
            print(f"  ✗ {dep}: NOT INSTALLED")
            missing.append(dep)
    
    # GPU检查
    print("\nGPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA:  {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        else: 
            print("  ⚠ CUDA not available")
    except ImportError: 
        print("  ⚠ PyTorch not installed")
    
    # 结果
    print("\n" + "="*60)
    if missing:
        print(f"⚠ Missing:  {', '.join(missing)}")
        print(f"  Install:  pip install {' '.join(missing)}")
    else:
        print("✓ All dependencies installed!")
    print("="*60)
    
    return len(missing) == 0


if __name__ == "__main__":
    check_environment()