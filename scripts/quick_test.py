"""快速测试脚本"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """测试导入"""
    print("Testing imports...")
    
    try:
        from adaprune import AdaPruneGBDT
        from adaprune.core import DataProfileAnalyzer, StateMonitor, PruningController
        print("  ✓ Core modules imported")
        return True
    except ImportError as e: 
        print(f"  ✗ Import error: {e}")
        return False


def test_synthetic_training():
    """测试合成数据训练"""
    print("\nTesting synthetic data training...")
    
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from adaprune import AdaPruneGBDT
        
        # 生成数据
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"  Data:  X_train={X_train.shape}, X_test={X_test.shape}")
        
        # 测试各策略
        strategies = ['gap_based', 'trend_based', 'data_aware', 'hybrid']
        
        for strategy in strategies:
            model = AdaPruneGBDT(
                n_estimators=30,
                strategy=strategy,
                adaptation_frequency=5,
                random_state=42,
                verbose=0
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"  ✓ {strategy}:  accuracy={acc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error:  {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptation_history():
    """测试自适应历史"""
    print("\nTesting adaptation history...")
    
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from adaprune import AdaPruneGBDT
        
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        
        model = AdaPruneGBDT(
            n_estimators=30,
            strategy='hybrid',
            adaptation_frequency=5,
            verbose=0
        )
        
        model.fit(X, y)
        
        history = model.get_adaptation_history()
        
        assert 'param_history' in history, "Missing param_history"
        assert 'train_scores' in history, "Missing train_scores"
        assert 'val_scores' in history, "Missing val_scores"
        assert len(history['param_history']) > 0, "Empty param_history"
        
        print(f"  ✓ History recorded:  {len(history['param_history'])} iterations")
        print(f"  ✓ Final depth: {history['param_history'][-1]['max_depth']}")
        
        return True
        
    except Exception as e: 
        print(f"  ✗ Error: {e}")
        import traceback
        traceback. print_exc()
        return False


def test_data_loader():
    """测试数据加载"""
    print("\nTesting data loader...")
    
    try:
        from adaprune. utils import list_datasets, load_dataset
        
        datasets = list_datasets()
        
        if len(datasets) == 0:
            print("  ⚠ No datasets found.  Run download_datasets.py first.")
            return True  # 不算失败
        
        print(f"  ✓ Found {len(datasets)} datasets")
        
        # 尝试加载第一个
        X, y, meta = load_dataset(datasets[0])
        print(f"  ✓ Loaded {datasets[0]}:  X={X.shape}")
        
        return True
        
    except Exception as e: 
        print(f"  ✗ Error: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("AdaPrune Quick Test")
    print("=" * 60)
    
    results = {}
    
    # 测试1: 导入
    results["Imports"] = test_imports()
    
    # 只有导入成功才继续其他测试
    if results["Imports"]:
        results["Synthetic Training"] = test_synthetic_training()
        results["Adaptation History"] = test_adaptation_history()
        results["Data Loader"] = test_data_loader()
    else:
        results["Synthetic Training"] = False
        results["Adaptation History"] = False
        results["Data Loader"] = False
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed!  AdaPrune is ready to use.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    return all(results.values())


if __name__ == "__main__": 
    success = run_all_tests()
    sys.exit(0 if success else 1)