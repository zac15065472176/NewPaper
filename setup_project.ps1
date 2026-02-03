# AdaPrune 项目初始化脚本 (PowerShell)
# 使用方法:  .\setup_project.ps1

Write-Host "=" * 60
Write-Host "AdaPrune Project Setup"
Write-Host "=" * 60

# 创建目录结构
$dirs = @(
    "adaprune",
    "adaprune\core",
    "adaprune\strategies",
    "adaprune\utils",
    "configs",
    "data",
    "data\raw",
    "data\processed",
    "experiments",
    "notebooks",
    "results",
    "results\tables",
    "results\figures",
    "scripts",
    "tests"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir"
    }
}

# 创建 __init__.py 文件
$init_dirs = @(
    "adaprune",
    "adaprune\core",
    "adaprune\strategies",
    "adaprune\utils"
)

foreach ($dir in $init_dirs) {
    $init_file = Join-Path $dir "__init__.py"
    if (-not (Test-Path $init_file)) {
        "# AdaPrune" | Out-File -FilePath $init_file -Encoding utf8
        Write-Host "  Created:  $init_file"
    }
}

Write-Host ""
Write-Host "Project structure created!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy the Python files to their respective directories"
Write-Host "  2. Run:  pip install -r requirements.txt"
Write-Host "  3. Run: python scripts/download_datasets.py"
Write-Host "  4. Run: python scripts/quick_test.py"