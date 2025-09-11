#!/bin/bash

# 多Agent数据分析系统安装脚本

echo "🤖 多Agent数据分析系统安装脚本"
echo "================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if (( $(echo "$python_version >= $required_version" | bc -l) )); then
    echo "✅ Python版本检查通过: $python_version"
else
    echo "❌ Python版本过低，需要3.8或更高版本"
    echo "当前版本: $python_version"
    exit 1
fi

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ 虚拟环境创建完成"
fi

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip

# 安装Python依赖
echo "📦 安装Python依赖包..."
pip install -r requirements.txt

# 检查tesseract（OCR功能）
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract已安装"
else
    echo "⚠️  Tesseract未安装，OCR功能将不可用"
    echo "Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim"
    echo "CentOS/RHEL: sudo yum install tesseract tesseract-langpack-chi_sim"
    echo "macOS: brew install tesseract tesseract-lang"
fi

# 创建必要的目录
echo "📁 创建目录结构..."
mkdir -p charts
mkdir -p reports  
mkdir -p generated_code
mkdir -p logs
mkdir -p uploads
mkdir -p temp

# 设置权限
chmod +x *.py
chmod +x *.sh

echo ""
echo "✅ 安装完成！"
echo ""
echo "🚀 启动方法:"
echo "1. 命令行模式: python orchestrator.py <文件路径>"
echo "2. Web界面模式: streamlit run app.py"
echo ""
echo "📚 使用说明:"
echo "- 支持文件格式: PDF, Word, Excel, CSV, 图片, JSON等"
echo "- 自动生成: 数据分析、可视化图表、分析报告、Python代码"
echo "- 输出目录: charts/ (图表), reports/ (报告), generated_code/ (代码)"
echo ""
echo "🔧 可选配置:"
echo "- 安装中文字体以支持更好的中文显示"
echo "- 配置Tesseract支持更多语言的OCR"
echo "- 根据需要调整各Agent的参数"