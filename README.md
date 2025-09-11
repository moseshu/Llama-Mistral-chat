# 🤖 多Agent数据分析系统

一个基于多Agent架构的智能数据分析系统，支持多种文件格式，能够自动生成数据洞察、可视化图表和分析报告。

## ✨ 核心特性

### 🔧 多Agent架构
- **文件处理Agent**: 支持PDF、Word、Excel、CSV、图片、JSON等多种格式
- **数据分析Agent**: 自动识别数据类型、评估数据质量、发现数据模式
- **代码生成Agent**: 根据数据特征动态生成Python分析代码
- **可视化Agent**: 自适应生成柱状图、饼图、折线图、散点图等多种图表
- **报告生成Agent**: 整合所有结果生成Markdown、HTML、JSON格式的综合报告

### 📊 智能分析能力
- **自动数据类型识别**: 数值型、分类型、时间序列、文本等
- **数据质量评估**: 缺失值检测、重复数据识别、异常值发现
- **模式识别**: 相关性分析、趋势识别、分布特征
- **自适应可视化**: 根据数据特征自动选择最适合的图表类型

### 🎯 通用性设计
- **跨行业适用**: 金融、医疗、教育、电商、制造业等各行各业
- **多格式支持**: 无需预处理，直接上传各种格式文件
- **自动化流程**: 从文件上传到报告生成全流程自动化
- **可扩展架构**: 易于添加新的Agent和功能模块

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- 8GB+ RAM (推荐)
- 支持的操作系统: Linux, macOS, Windows

### 2. 安装方式

#### 自动安装 (推荐)
```bash
# 克隆项目
git clone <repository-url>
cd multi-agent-data-analysis

# 运行安装脚本
./install.sh
```

#### 手动安装
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 创建目录
mkdir -p {charts,reports,generated_code,logs,uploads,temp}
```

### 3. 启动系统

#### Web界面模式 (推荐)
```bash
streamlit run app.py
```
然后在浏览器中访问 `http://localhost:8501`

#### 命令行模式
```bash
python orchestrator.py <文件路径>
```

## 📁 项目结构

```
multi-agent-data-analysis/
├── agents/                          # Agent模块
│   ├── file_processor_agent.py      # 文件处理Agent
│   ├── data_analyzer_agent.py       # 数据分析Agent
│   ├── code_generator_agent.py      # 代码生成Agent
│   ├── visualization_agent.py       # 可视化Agent
│   └── report_generator_agent.py    # 报告生成Agent
├── orchestrator.py                  # 多Agent协调器
├── app.py                          # Web应用主程序
├── requirements.txt                # Python依赖
├── install.sh                      # 安装脚本
├── README.md                       # 项目说明
├── charts/                         # 生成的图表
├── reports/                        # 生成的报告
├── generated_code/                 # 生成的代码
└── logs/                          # 日志文件
```

## 🎯 使用场景

### 📈 业务数据分析
- **销售数据**: 分析销售趋势、客户行为、产品性能
- **财务数据**: 收入分析、成本控制、盈利能力评估
- **运营数据**: KPI监控、效率分析、资源配置优化

### 🔬 科研数据处理
- **实验数据**: 统计分析、假设检验、结果可视化
- **调研数据**: 问卷分析、相关性研究、报告生成
- **监测数据**: 时间序列分析、异常检测、趋势预测

### 📊 市场研究
- **用户调研**: 用户画像、行为分析、满意度评估
- **竞品分析**: 市场对比、优势识别、策略建议
- **行业分析**: 趋势识别、机会发现、风险评估

## 🛠️ 高级功能

### 自定义分析流程
```python
from orchestrator import MultiAgentOrchestrator

# 创建协调器实例
orchestrator = MultiAgentOrchestrator()

# 自定义分析选项
options = {
    'generate_code': True,           # 生成分析代码
    'create_visualizations': True,  # 创建可视化
    'generate_report': True,        # 生成报告
    'save_intermediate': False      # 保存中间结果
}

# 执行分析
result = orchestrator.analyze_file('data.xlsx', options)
```

### 扩展Agent功能
```python
# 继承现有Agent并添加自定义功能
from agents.data_analyzer_agent import DataAnalyzerAgent

class CustomAnalyzerAgent(DataAnalyzerAgent):
    def custom_analysis_method(self, data):
        # 添加自定义分析逻辑
        pass
```

## 📋 支持的文件格式

| 格式类型 | 支持的扩展名 | 处理能力 |
|---------|-------------|---------|
| **表格数据** | .csv, .xlsx, .xls | 完整的数据分析和可视化 |
| **文档** | .pdf, .docx, .txt | 文本提取、表格识别 |
| **图片** | .png, .jpg, .jpeg | OCR文字识别、表格提取 |
| **数据** | .json | 结构化数据解析 |

## 🎨 可视化类型

- **分布图**: 直方图、密度图、箱线图
- **关系图**: 散点图、相关性热力图
- **分类图**: 柱状图、饼图、堆叠图
- **时间序列**: 折线图、面积图
- **文本分析**: 词云图、词频图

## 📄 输出格式

### 分析报告
- **Markdown**: 适合文档编辑和版本控制
- **HTML**: 包含样式的完整报告，支持交互
- **JSON**: 结构化数据，便于程序处理

### 可视化
- **静态图片**: PNG格式，高分辨率
- **交互式图表**: HTML格式，支持缩放和筛选
- **仪表板**: 集成所有图表的综合视图

### 分析代码
- **Python脚本**: 可重复执行的完整分析流程
- **模块化代码**: 按功能分段的代码结构
- **使用说明**: 详细的代码使用指南

## ⚙️ 配置说明

### 系统配置
```python
# 在orchestrator.py中调整日志级别
logging.basicConfig(level=logging.INFO)

# 在各个Agent中调整参数
class DataAnalyzerAgent:
    def __init__(self):
        self.correlation_threshold = 0.7  # 相关性阈值
        self.outlier_method = 'IQR'       # 异常值检测方法
```

### 可视化配置
```python
# 在visualization_agent.py中设置图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300  # 图片分辨率
```

## 🔧 故障排除

### 常见问题

1. **OCR功能不可用**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
   
   # macOS
   brew install tesseract tesseract-lang
   ```

2. **中文显示问题**
   - 下载并安装中文字体文件
   - 修改matplotlib配置

3. **内存不足**
   - 处理大文件时可能需要更多内存
   - 考虑分批处理或使用更强配置的机器

### 日志调试
```bash
# 查看详细日志
tail -f analysis.log

# 启用调试模式
export PYTHONPATH=$PYTHONPATH:.
python -m pdb orchestrator.py data.xlsx
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- 感谢所有开源库的贡献者
- 特别感谢pandas、matplotlib、streamlit等优秀项目
- 感谢社区用户的反馈和建议

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [your-email@example.com]

---

**🤖 让数据分析变得简单智能！**