#!/bin/bash

# 多Agent数据分析系统启动脚本

echo "🤖 多Agent数据分析系统"
echo "======================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查依赖是否安装
if ! python3 -c "import pandas, streamlit" 2>/dev/null; then
    echo "⚠️  依赖包未安装，正在安装..."
    pip install -r requirements.txt
fi

echo ""
echo "请选择启动模式:"
echo "1) 🌐 Web界面模式 (推荐)"
echo "2) 💻 命令行模式" 
echo "3) 🎯 演示模式"
echo "4) ❌ 退出"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "🌐 启动Web界面..."
        echo "浏览器将自动打开 http://localhost:8501"
        streamlit run app.py
        ;;
    2)
        echo "💻 命令行模式"
        echo "用法: python orchestrator.py <文件路径>"
        echo ""
        read -p "请输入要分析的文件路径: " filepath
        if [ -f "$filepath" ]; then
            python orchestrator.py "$filepath"
        else
            echo "❌ 文件不存在: $filepath"
        fi
        ;;
    3)
        echo "🎯 启动演示模式..."
        python demo.py --auto
        ;;
    4)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac