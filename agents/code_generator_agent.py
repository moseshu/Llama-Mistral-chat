"""
代码生成Agent
根据数据特征动态生成分析代码
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import textwrap

class CodeGeneratorAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.code_templates = {
            'data_loading': self._generate_data_loading_code,
            'data_cleaning': self._generate_data_cleaning_code,
            'statistical_analysis': self._generate_statistical_code,
            'visualization': self._generate_visualization_code,
            'correlation_analysis': self._generate_correlation_code,
            'time_series': self._generate_time_series_code
        }
    
    def generate_analysis_code(self, analysis_result: Dict[str, Any], 
                             original_file_path: str = None) -> Dict[str, Any]:
        """
        根据分析结果生成完整的数据分析代码
        """
        try:
            if not analysis_result.get('success', False):
                return {
                    'success': False,
                    'error': '分析结果无效',
                    'code': ''
                }
            
            analysis_type = analysis_result.get('analysis_type', '')
            
            if analysis_type == 'single_structured':
                return self._generate_structured_analysis_code(
                    analysis_result.get('analysis', {}), 
                    original_file_path
                )
            elif analysis_type == 'multi_sheet_structured':
                return self._generate_multi_sheet_code(
                    analysis_result.get('main_analysis', {}),
                    analysis_result.get('recommended_sheet', 'Sheet1'),
                    original_file_path
                )
            elif analysis_type == 'document_with_tables':
                return self._generate_document_table_code(
                    analysis_result.get('table_analyses', []),
                    original_file_path
                )
            elif analysis_type in ['document', 'ocr_document']:
                return self._generate_text_analysis_code(original_file_path)
            else:
                return {
                    'success': False,
                    'error': f'不支持的分析类型: {analysis_type}',
                    'code': ''
                }
                
        except Exception as e:
            self.logger.error(f"代码生成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': ''
            }
    
    def _generate_structured_analysis_code(self, analysis: Dict[str, Any], 
                                         file_path: str = None) -> Dict[str, Any]:
        """生成结构化数据分析代码"""
        if 'error' in analysis:
            return {'success': False, 'error': analysis['error'], 'code': ''}
        
        basic_info = analysis.get('basic_info', {})
        column_analysis = analysis.get('column_analysis', {})
        patterns = analysis.get('patterns', {})
        
        # 生成代码段
        code_sections = []
        
        # 1. 导入库
        imports = self._generate_imports()
        code_sections.append(("导入必要的库", imports))
        
        # 2. 数据加载
        data_loading = self._generate_data_loading_code(file_path, basic_info)
        code_sections.append(("数据加载", data_loading))
        
        # 3. 数据探索
        data_exploration = self._generate_data_exploration_code(basic_info, column_analysis)
        code_sections.append(("数据探索", data_exploration))
        
        # 4. 数据清洗
        data_cleaning = self._generate_data_cleaning_code(analysis)
        if data_cleaning:
            code_sections.append(("数据清洗", data_cleaning))
        
        # 5. 统计分析
        statistical_analysis = self._generate_statistical_code(column_analysis)
        code_sections.append(("统计分析", statistical_analysis))
        
        # 6. 相关性分析
        correlations = patterns.get('correlations', [])
        if correlations:
            correlation_code = self._generate_correlation_code(column_analysis)
            code_sections.append(("相关性分析", correlation_code))
        
        # 7. 可视化
        visualization_code = self._generate_visualization_code(column_analysis)
        code_sections.append(("数据可视化", visualization_code))
        
        # 8. 报告生成
        report_code = self._generate_report_code(analysis)
        code_sections.append(("生成分析报告", report_code))
        
        # 组合所有代码
        full_code = self._combine_code_sections(code_sections)
        
        return {
            'success': True,
            'code': full_code,
            'sections': [section[0] for section in code_sections],
            'description': '完整的数据分析代码，包含数据加载、清洗、分析和可视化'
        }
    
    def _generate_imports(self) -> str:
        """生成导入语句"""
        return textwrap.dedent("""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        from datetime import datetime
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        warnings.filterwarnings('ignore')
        
        print("📊 数据分析环境初始化完成")
        """).strip()
    
    def _generate_data_loading_code(self, file_path: str, basic_info: Dict[str, Any]) -> str:
        """生成数据加载代码"""
        if not file_path:
            file_path = "your_data_file.csv"  # 默认文件名
        
        file_ext = file_path.split('.')[-1].lower() if '.' in file_path else 'csv'
        
        if file_ext in ['xlsx', 'xls']:
            load_code = f"""
# 加载Excel文件
df = pd.read_excel('{file_path}')
print(f"✅ 数据加载成功: {{df.shape[0]}} 行, {{df.shape[1]}} 列")
"""
        elif file_ext == 'csv':
            load_code = f"""
# 加载CSV文件
try:
    df = pd.read_csv('{file_path}', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('{file_path}', encoding='gbk')
print(f"✅ 数据加载成功: {{df.shape[0]}} 行, {{df.shape[1]}} 列")
"""
        else:
            load_code = f"""
# 加载数据文件
df = pd.read_csv('{file_path}')  # 根据实际文件格式调整
print(f"✅ 数据加载成功: {{df.shape[0]}} 行, {{df.shape[1]}} 列")
"""
        
        return textwrap.dedent(load_code).strip()
    
    def _generate_data_exploration_code(self, basic_info: Dict[str, Any], 
                                      column_analysis: Dict[str, Any]) -> str:
        """生成数据探索代码"""
        code = """
# 数据基本信息
print("\\n" + "="*50)
print("📋 数据基本信息")
print("="*50)
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print("\\n数据类型:")
print(df.dtypes)

print("\\n" + "="*50)
print("🔍 数据预览")
print("="*50)
print("前5行数据:")
print(df.head())

print("\\n数据描述性统计:")
print(df.describe())

print("\\n缺失值统计:")
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    print(missing_data)
else:
    print("✅ 没有发现缺失值")
"""
        return textwrap.dedent(code).strip()
    
    def _generate_data_cleaning_code(self, analysis: Dict[str, Any]) -> str:
        """生成数据清洗代码"""
        data_quality = analysis.get('data_quality', {})
        missing_percentage = data_quality.get('missing_percentage', 0)
        duplicate_rows = data_quality.get('duplicate_rows', 0)
        
        if missing_percentage == 0 and duplicate_rows == 0:
            return ""  # 数据质量良好，无需清洗
        
        code_parts = ["""
print("\\n" + "="*50)
print("🧹 数据清洗")
print("="*50)
"""]
        
        if missing_percentage > 0:
            code_parts.append("""
# 处理缺失值
print("处理缺失值...")
# 对于数值列，用中位数填充
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# 对于分类列，用众数填充
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
""")
        
        if duplicate_rows > 0:
            code_parts.append("""
# 删除重复行
print("删除重复行...")
df_before = len(df)
df = df.drop_duplicates()
df_after = len(df)
print(f"删除了 {df_before - df_after} 行重复数据")
""")
        
        code_parts.append("""
print("✅ 数据清洗完成")
""")
        
        return textwrap.dedent('\n'.join(code_parts)).strip()
    
    def _generate_statistical_code(self, column_analysis: Dict[str, Any]) -> str:
        """生成统计分析代码"""
        numeric_cols = [col for col, info in column_analysis.items() 
                       if info.get('type') in ['integer', 'float']]
        categorical_cols = [col for col, info in column_analysis.items() 
                           if info.get('type') == 'categorical']
        
        code_parts = ["""
print("\\n" + "="*50)
print("📊 统计分析")
print("="*50)
"""]
        
        if numeric_cols:
            code_parts.append(f"""
# 数值型变量统计分析
numeric_cols = {numeric_cols}
print("数值型变量统计摘要:")
for col in numeric_cols:
    if col in df.columns:
        print(f"\\n{col}:")
        print(f"  均值: {{df[col].mean():.2f}}")
        print(f"  中位数: {{df[col].median():.2f}}")
        print(f"  标准差: {{df[col].std():.2f}}")
        print(f"  最小值: {{df[col].min():.2f}}")
        print(f"  最大值: {{df[col].max():.2f}}")
""")
        
        if categorical_cols:
            code_parts.append(f"""
# 分类变量统计分析
categorical_cols = {categorical_cols}
print("\\n分类变量分布:")
for col in categorical_cols:
    if col in df.columns:
        print(f"\\n{col} 分布:")
        value_counts = df[col].value_counts()
        print(value_counts.head(10))
        print(f"唯一值数量: {{df[col].nunique()}}")
""")
        
        return textwrap.dedent('\n'.join(code_parts)).strip()
    
    def _generate_correlation_code(self, column_analysis: Dict[str, Any]) -> str:
        """生成相关性分析代码"""
        numeric_cols = [col for col, info in column_analysis.items() 
                       if info.get('type') in ['integer', 'float']]
        
        if len(numeric_cols) < 2:
            return ""
        
        code = f"""
print("\\n" + "="*50)
print("🔗 相关性分析")
print("="*50)

# 计算相关性矩阵
numeric_cols = {numeric_cols}
numeric_data = df[numeric_cols]
correlation_matrix = numeric_data.corr()

print("相关性矩阵:")
print(correlation_matrix)

# 找出强相关性（绝对值>0.7）
print("\\n强相关性变量对:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            print(f"{col1} - {col2}: {corr_val:.3f}")
"""
        return textwrap.dedent(code).strip()
    
    def _generate_visualization_code(self, column_analysis: Dict[str, Any]) -> str:
        """生成可视化代码"""
        numeric_cols = [col for col, info in column_analysis.items() 
                       if info.get('type') in ['integer', 'float']]
        categorical_cols = [col for col, info in column_analysis.items() 
                           if info.get('type') == 'categorical']
        
        code_parts = ["""
print("\\n" + "="*50)
print("📈 数据可视化")
print("="*50)

# 创建图形保存目录
import os
if not os.path.exists('charts'):
    os.makedirs('charts')
"""]
        
        # 数值分布直方图
        if numeric_cols:
            code_parts.append(f"""
# 1. 数值分布直方图
numeric_cols = {numeric_cols[:4]}  # 最多显示4个
if len(numeric_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        if i < 4 and col in df.columns:
            df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue')
            axes[i].set_title(f'{col} 分布')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('频次')
    
    # 隐藏多余的子图
    for i in range(len(numeric_cols), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('charts/numeric_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 数值分布图已保存到 charts/numeric_distribution.png")
""")
        
        # 分类数据饼图
        if categorical_cols:
            code_parts.append(f"""
# 2. 分类数据饼图
categorical_cols = {categorical_cols[:2]}  # 最多显示2个
for i, col in enumerate(categorical_cols):
    if col in df.columns and df[col].nunique() <= 10:
        plt.figure(figsize=(10, 8))
        value_counts = df[col].value_counts()
        
        # 饼图
        plt.subplot(1, 2, 1)
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'{col} 分布饼图')
        
        # 柱状图
        plt.subplot(1, 2, 2)
        value_counts.plot(kind='bar', color='lightcoral')
        plt.title(f'{col} 分布柱状图')
        plt.xticks(rotation=45)
        plt.ylabel('数量')
        
        plt.tight_layout()
        plt.savefig(f'charts/{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ {col} 分布图已保存到 charts/{col}_distribution.png")
""")
        
        # 相关性热力图
        if len(numeric_cols) >= 2:
            code_parts.append(f"""
# 3. 相关性热力图
if len({numeric_cols}) >= 2:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[{numeric_cols}].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={{"shrink": .8}})
    plt.title('变量相关性热力图', fontsize=16)
    plt.tight_layout()
    plt.savefig('charts/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 相关性热力图已保存到 charts/correlation_heatmap.png")
""")
        
        # 散点图
        if len(numeric_cols) >= 2:
            code_parts.append(f"""
# 4. 散点图矩阵
if len({numeric_cols}) >= 2:
    # 选择前4个数值列
    cols_for_scatter = {numeric_cols[:4]}
    if len(cols_for_scatter) >= 2:
        scatter_data = df[cols_for_scatter]
        fig = px.scatter_matrix(scatter_data, 
                               title="变量关系散点图矩阵",
                               width=800, height=800)
        fig.write_html('charts/scatter_matrix.html')
        fig.show()
        print("✅ 散点图矩阵已保存到 charts/scatter_matrix.html")
""")
        
        # 箱线图（异常值检测）
        if numeric_cols:
            code_parts.append(f"""
# 5. 箱线图（异常值检测）
numeric_cols_box = {numeric_cols[:3]}  # 最多显示3个
if len(numeric_cols_box) > 0:
    plt.figure(figsize=(15, 6))
    
    for i, col in enumerate(numeric_cols_box):
        if col in df.columns:
            plt.subplot(1, len(numeric_cols_box), i+1)
            df.boxplot(column=col, ax=plt.gca())
            plt.title(f'{col} 箱线图')
            plt.ylabel('数值')
    
    plt.tight_layout()
    plt.savefig('charts/boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 箱线图已保存到 charts/boxplots.png")
""")
        
        return textwrap.dedent('\n'.join(code_parts)).strip()
    
    def _generate_report_code(self, analysis: Dict[str, Any]) -> str:
        """生成报告生成代码"""
        insights = analysis.get('insights', [])
        data_quality = analysis.get('data_quality', {})
        
        code = f"""
print("\\n" + "="*50)
print("📝 生成分析报告")
print("="*50)

# 生成分析报告
report = []
report.append("# 数据分析报告")
report.append(f"\\n生成时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
report.append("\\n## 数据概况")
report.append(f"- 数据行数: {{df.shape[0]:,}}")
report.append(f"- 数据列数: {{df.shape[1]:,}}")
report.append(f"- 数据质量评分: {data_quality.get('quality_score', 0):.1f}/100")

report.append("\\n## 主要发现")
insights = {insights}
for i, insight in enumerate(insights, 1):
    report.append(f"{i}. {{insight}}")

report.append("\\n## 数据质量评估")
missing_pct = {data_quality.get('missing_percentage', 0):.1f}
duplicate_rows = {data_quality.get('duplicate_rows', 0)}
report.append(f"- 缺失值比例: {{missing_pct}}%")
report.append(f"- 重复行数: {{duplicate_rows}}")

report.append("\\n## 可视化图表")
report.append("本次分析生成的图表文件:")
import os
if os.path.exists('charts'):
    chart_files = [f for f in os.listdir('charts') if f.endswith(('.png', '.html'))]
    for chart_file in chart_files:
        report.append(f"- {{chart_file}}")

# 保存报告
report_content = "\\n".join(report)
with open('analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✅ 分析报告已保存到 analysis_report.md")
print("\\n报告内容预览:")
print("-" * 50)
print(report_content)
"""
        return textwrap.dedent(code).strip()
    
    def _combine_code_sections(self, sections: List[tuple]) -> str:
        """组合所有代码段"""
        combined = []
        
        for section_name, code in sections:
            if code:  # 只添加非空代码段
                combined.append(f"# {section_name}")
                combined.append(code)
                combined.append("")  # 添加空行分隔
        
        return "\n".join(combined)
    
    def _generate_multi_sheet_code(self, main_analysis: Dict[str, Any], 
                                 sheet_name: str, file_path: str) -> Dict[str, Any]:
        """生成多工作表Excel文件的分析代码"""
        # 修改数据加载部分以指定工作表
        result = self._generate_structured_analysis_code(main_analysis, file_path)
        
        if result.get('success'):
            # 替换数据加载代码以包含工作表名称
            code = result['code']
            if file_path and file_path.endswith(('.xlsx', '.xls')):
                old_load = f"df = pd.read_excel('{file_path}')"
                new_load = f"df = pd.read_excel('{file_path}', sheet_name='{sheet_name}')"
                code = code.replace(old_load, new_load)
                result['code'] = code
        
        return result
    
    def _generate_document_table_code(self, table_analyses: List[Dict], 
                                    file_path: str) -> Dict[str, Any]:
        """生成文档中表格的分析代码"""
        if not table_analyses:
            return {
                'success': False,
                'error': '没有找到可分析的表格',
                'code': ''
            }
        
        # 选择第一个表格进行分析
        main_table = table_analyses[0]
        analysis = main_table.get('analysis', {})
        
        # 生成基础分析代码，但修改数据加载部分
        result = self._generate_structured_analysis_code(analysis, file_path)
        
        if result.get('success'):
            # 添加表格提取的说明
            code = result['code']
            table_note = f"""
# 注意: 此代码假设表格数据已从文档中提取
# 原始文档: {file_path}
# 表格索引: {main_table.get('table_index', 0)}
# 请根据实际提取的表格数据调整数据加载部分

"""
            result['code'] = table_note + code
        
        return result
    
    def _generate_text_analysis_code(self, file_path: str) -> Dict[str, Any]:
        """生成文本分析代码"""
        code = f"""
# 文本分析代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import jieba  # 中文分词

print("📊 文本分析环境初始化完成")

# 读取文本文件
with open('{file_path}', 'r', encoding='utf-8') as f:
    text_content = f.read()

print(f"✅ 文本加载成功: {{len(text_content)}} 个字符")

# 文本基本统计
print("\\n" + "="*50)
print("📋 文本基本信息")
print("="*50)
print(f"字符数: {{len(text_content):,}}")
print(f"词语数: {{len(text_content.split()):,}}")
print(f"行数: {{len(text_content.splitlines()):,}}")

# 中文分词和词频统计
print("\\n进行中文分词...")
words = jieba.lcut(text_content)
word_freq = Counter(words)

# 过滤停用词和短词
filtered_words = [word for word in words if len(word) > 1 and word.strip()]
filtered_freq = Counter(filtered_words)

print("\\n最常见的词语 (Top 20):")
for word, freq in filtered_freq.most_common(20):
    print(f"{{word}}: {{freq}}")

# 生成词云
print("\\n生成词云图...")
if not os.path.exists('charts'):
    os.makedirs('charts')

# 词云图
wordcloud = WordCloud(
    font_path='SimHei.ttf',  # 需要中文字体文件
    width=800, 
    height=600,
    background_color='white',
    max_words=100
).generate(' '.join(filtered_words))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('词云图', fontsize=16)
plt.tight_layout()
plt.savefig('charts/wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# 词频柱状图
plt.figure(figsize=(12, 8))
top_words = dict(filtered_freq.most_common(15))
plt.bar(range(len(top_words)), list(top_words.values()), color='skyblue')
plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
plt.title('词频统计 (Top 15)')
plt.ylabel('出现次数')
plt.tight_layout()
plt.savefig('charts/word_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 文本分析完成，图表已保存到 charts/ 目录")

# 生成文本分析报告
report = []
report.append("# 文本分析报告")
report.append(f"\\n生成时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
report.append(f"\\n原始文件: {file_path}")
report.append("\\n## 文本统计")
report.append(f"- 总字符数: {{len(text_content):,}}")
report.append(f"- 总词语数: {{len(words):,}}")
report.append(f"- 有效词语数: {{len(filtered_words):,}}")
report.append(f"- 唯一词语数: {{len(set(filtered_words)):,}}")

report.append("\\n## 高频词语")
for i, (word, freq) in enumerate(filtered_freq.most_common(10), 1):
    report.append(f"{{i}}. {{word}}: {{freq}} 次")

report_content = "\\n".join(report)
with open('text_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✅ 文本分析报告已保存到 text_analysis_report.md")
"""
        
        return {
            'success': True,
            'code': textwrap.dedent(code).strip(),
            'sections': ['文本加载', '基本统计', '分词处理', '词频分析', '可视化', '报告生成'],
            'description': '完整的文本分析代码，包含分词、词频统计和可视化'
        }

if __name__ == "__main__":
    # 测试代码
    agent = CodeGeneratorAgent()
    print("代码生成Agent初始化完成")
    print(f"支持的代码模板: {list(agent.code_templates.keys())}")