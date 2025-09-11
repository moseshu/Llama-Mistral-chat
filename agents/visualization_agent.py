"""
可视化Agent
自适应生成各种类型的图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from datetime import datetime
import base64
from io import BytesIO

class VisualizationAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chart_types = {
            'histogram': self._create_histogram,
            'bar': self._create_bar_chart,
            'pie': self._create_pie_chart,
            'line': self._create_line_chart,
            'scatter': self._create_scatter_plot,
            'boxplot': self._create_boxplot,
            'heatmap': self._create_heatmap,
            'violin': self._create_violin_plot,
            'area': self._create_area_chart,
            'wordcloud': self._create_wordcloud
        }
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        warnings.filterwarnings('ignore')
        
        # 创建图表保存目录
        if not os.path.exists('charts'):
            os.makedirs('charts')
    
    def create_visualizations(self, data: pd.DataFrame, 
                            suggestions: List[Dict[str, Any]],
                            analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        根据可视化建议创建图表
        """
        try:
            if data is None or data.empty:
                return {
                    'success': False,
                    'error': '数据为空',
                    'charts': []
                }
            
            created_charts = []
            failed_charts = []
            
            # 按优先级排序
            sorted_suggestions = sorted(suggestions, 
                                      key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(x.get('priority', 'low'), 1), 
                                      reverse=True)
            
            for suggestion in sorted_suggestions:
                try:
                    chart_result = self._create_chart(data, suggestion)
                    if chart_result['success']:
                        created_charts.append(chart_result)
                    else:
                        failed_charts.append({
                            'suggestion': suggestion,
                            'error': chart_result['error']
                        })
                except Exception as e:
                    self.logger.error(f"创建图表失败: {e}")
                    failed_charts.append({
                        'suggestion': suggestion,
                        'error': str(e)
                    })
            
            return {
                'success': True,
                'charts': created_charts,
                'failed_charts': failed_charts,
                'total_created': len(created_charts),
                'total_failed': len(failed_charts)
            }
            
        except Exception as e:
            self.logger.error(f"可视化创建失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'charts': []
            }
    
    def _create_chart(self, data: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建单个图表"""
        chart_type = suggestion.get('type', '')
        columns = suggestion.get('columns', [])
        title = suggestion.get('title', f'{chart_type} 图表')
        
        if chart_type not in self.chart_types:
            return {
                'success': False,
                'error': f'不支持的图表类型: {chart_type}'
            }
        
        # 验证列是否存在
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            return {
                'success': False,
                'error': f'列不存在: {missing_cols}'
            }
        
        # 调用对应的图表创建函数
        chart_func = self.chart_types[chart_type]
        return chart_func(data, columns, title, suggestion)
    
    def _create_histogram(self, data: pd.DataFrame, columns: List[str], 
                         title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建直方图"""
        try:
            fig, axes = plt.subplots(1, len(columns), figsize=(6*len(columns), 6))
            if len(columns) == 1:
                axes = [axes]
            
            for i, col in enumerate(columns):
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col].dropna().hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'{col} 分布')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('频次')
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].text(0.5, 0.5, f'{col} 不是数值类型', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{col} (非数值)')
            
            plt.tight_layout()
            filename = f'charts/histogram_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'histogram',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '数值分布直方图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_bar_chart(self, data: pd.DataFrame, columns: List[str], 
                         title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建柱状图"""
        try:
            col = columns[0]
            
            # 获取值计数
            if pd.api.types.is_numeric_dtype(data[col]):
                # 数值型数据分箱
                bins = pd.cut(data[col], bins=10)
                value_counts = bins.value_counts().sort_index()
            else:
                # 分类数据
                value_counts = data[col].value_counts().head(15)  # 显示前15个
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(value_counts)), value_counts.values, 
                          color='lightcoral', alpha=0.8, edgecolor='black')
            
            # 设置x轴标签
            plt.xticks(range(len(value_counts)), 
                      [str(x) for x in value_counts.index], 
                      rotation=45, ha='right')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel(col, fontsize=12)
            plt.ylabel('数量', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, value in zip(bars, value_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                        f'{value}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            filename = f'charts/bar_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'bar',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '分布柱状图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_pie_chart(self, data: pd.DataFrame, columns: List[str], 
                         title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建饼图"""
        try:
            col = columns[0]
            value_counts = data[col].value_counts().head(10)  # 最多显示10个类别
            
            # 如果类别太多，合并小类别为"其他"
            if len(data[col].value_counts()) > 10:
                others_count = data[col].value_counts().iloc[10:].sum()
                if others_count > 0:
                    value_counts['其他'] = others_count
            
            plt.figure(figsize=(10, 8))
            
            # 生成颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            
            wedges, texts, autotexts = plt.pie(value_counts.values, 
                                              labels=value_counts.index,
                                              autopct='%1.1f%%', 
                                              startangle=90,
                                              colors=colors,
                                              explode=[0.05] * len(value_counts))
            
            plt.title(title, fontsize=16, fontweight='bold')
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.axis('equal')
            filename = f'charts/pie_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'pie',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '分布饼图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_line_chart(self, data: pd.DataFrame, columns: List[str], 
                          title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建折线图"""
        try:
            if len(columns) < 2:
                return {'success': False, 'error': '折线图需要至少2列数据'}
            
            x_col, y_col = columns[0], columns[1]
            
            # 如果x轴是日期时间类型，进行排序
            if pd.api.types.is_datetime64_any_dtype(data[x_col]):
                plot_data = data.sort_values(x_col)
            else:
                plot_data = data.copy()
            
            plt.figure(figsize=(12, 8))
            plt.plot(plot_data[x_col], plot_data[y_col], 
                    marker='o', linewidth=2, markersize=4, alpha=0.8, color='steelblue')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 旋转x轴标签如果太长
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            filename = f'charts/line_{x_col}_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'line',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '趋势折线图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_scatter_plot(self, data: pd.DataFrame, columns: List[str], 
                           title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建散点图"""
        try:
            if len(columns) < 2:
                return {'success': False, 'error': '散点图需要至少2列数据'}
            
            x_col, y_col = columns[0], columns[1]
            
            plt.figure(figsize=(10, 8))
            
            # 添加颜色编码（如果有第三列）
            if len(columns) > 2 and columns[2] in data.columns:
                c_col = columns[2]
                scatter = plt.scatter(data[x_col], data[y_col], 
                                    c=data[c_col], cmap='viridis', 
                                    alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, label=c_col)
            else:
                plt.scatter(data[x_col], data[y_col], 
                          alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
            
            # 添加趋势线
            if pd.api.types.is_numeric_dtype(data[x_col]) and pd.api.types.is_numeric_dtype(data[y_col]):
                z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
                p = np.poly1d(z)
                plt.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, linewidth=2, label='趋势线')
                plt.legend()
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'charts/scatter_{x_col}_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'scatter',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '散点图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_boxplot(self, data: pd.DataFrame, columns: List[str], 
                       title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建箱线图"""
        try:
            # 只选择数值列
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if not numeric_cols:
                return {'success': False, 'error': '没有数值列可用于箱线图'}
            
            plt.figure(figsize=(12, 8))
            
            # 创建箱线图
            box_data = [data[col].dropna() for col in numeric_cols]
            box_plot = plt.boxplot(box_data, labels=numeric_cols, patch_artist=True)
            
            # 美化箱线图
            colors = plt.cm.Set3(np.linspace(0, 1, len(numeric_cols)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.ylabel('数值', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            filename = f'charts/boxplot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'boxplot',
                'title': title,
                'filename': filename,
                'columns': numeric_cols,
                'description': suggestion.get('description', '箱线图（异常值检测）')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_heatmap(self, data: pd.DataFrame, columns: List[str], 
                       title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建热力图"""
        try:
            # 只选择数值列
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if len(numeric_cols) < 2:
                return {'success': False, 'error': '热力图需要至少2个数值列'}
            
            # 计算相关性矩阵
            corr_matrix = data[numeric_cols].corr()
            
            plt.figure(figsize=(12, 10))
            
            # 创建热力图
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True, 
                       fmt='.2f',
                       mask=mask,
                       cbar_kws={"shrink": .8})
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = f'charts/heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'heatmap',
                'title': title,
                'filename': filename,
                'columns': numeric_cols,
                'description': suggestion.get('description', '相关性热力图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_violin_plot(self, data: pd.DataFrame, columns: List[str], 
                           title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建小提琴图"""
        try:
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if not numeric_cols:
                return {'success': False, 'error': '没有数值列可用于小提琴图'}
            
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            plot_data = []
            labels = []
            for col in numeric_cols:
                plot_data.append(data[col].dropna())
                labels.append(col)
            
            # 创建小提琴图
            violin_parts = plt.violinplot(plot_data, positions=range(1, len(plot_data)+1), showmeans=True)
            
            # 美化
            for pc in violin_parts['bodies']:
                pc.set_alpha(0.7)
                pc.set_facecolor('lightblue')
            
            plt.xticks(range(1, len(labels)+1), labels, rotation=45)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.ylabel('数值', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            filename = f'charts/violin_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'violin',
                'title': title,
                'filename': filename,
                'columns': numeric_cols,
                'description': suggestion.get('description', '小提琴图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_area_chart(self, data: pd.DataFrame, columns: List[str], 
                          title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建面积图"""
        try:
            if len(columns) < 2:
                return {'success': False, 'error': '面积图需要至少2列数据'}
            
            x_col = columns[0]
            y_cols = columns[1:]
            
            plt.figure(figsize=(12, 8))
            
            for i, y_col in enumerate(y_cols):
                if pd.api.types.is_numeric_dtype(data[y_col]):
                    plt.fill_between(data[x_col], data[y_col], alpha=0.6, label=y_col)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel('数值', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            filename = f'charts/area_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'area',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '面积图')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_wordcloud(self, data: pd.DataFrame, columns: List[str], 
                         title: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """创建词云图"""
        try:
            from wordcloud import WordCloud
            import jieba
            
            if not columns or columns[0] not in data.columns:
                return {'success': False, 'error': '词云图需要文本列'}
            
            text_col = columns[0]
            
            # 合并所有文本
            text_data = data[text_col].dropna().astype(str)
            full_text = ' '.join(text_data)
            
            # 中文分词
            words = jieba.lcut(full_text)
            filtered_words = [word for word in words if len(word) > 1 and word.strip()]
            text_for_cloud = ' '.join(filtered_words)
            
            # 创建词云
            wordcloud = WordCloud(
                width=800, 
                height=600,
                background_color='white',
                max_words=100,
                colormap='viridis',
                font_path=None  # 需要设置中文字体路径
            ).generate(text_for_cloud)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            filename = f'charts/wordcloud_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'chart_type': 'wordcloud',
                'title': title,
                'filename': filename,
                'columns': columns,
                'description': suggestion.get('description', '词云图')
            }
            
        except ImportError:
            return {'success': False, 'error': '需要安装wordcloud和jieba库'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_dashboard(self, charts: List[Dict[str, Any]], 
                        analysis_summary: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建仪表板汇总所有图表"""
        try:
            if not charts:
                return {'success': False, 'error': '没有图表可用于创建仪表板'}
            
            # 创建HTML仪表板
            html_content = self._generate_dashboard_html(charts, analysis_summary)
            
            dashboard_filename = f'charts/dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            with open(dashboard_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'success': True,
                'dashboard_filename': dashboard_filename,
                'chart_count': len(charts),
                'description': '包含所有图表的交互式仪表板'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_dashboard_html(self, charts: List[Dict[str, Any]], 
                               analysis_summary: Dict[str, Any] = None) -> str:
        """生成仪表板HTML内容"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>数据分析仪表板</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .summary {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .chart-item {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .chart-description {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 15px;
        }
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 数据分析仪表板</h1>
        <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
"""
        
        # 添加分析摘要
        if analysis_summary:
            html += """
    <div class="summary">
        <h2>📋 分析摘要</h2>
        <p>本次分析共生成 """ + str(len(charts)) + """ 个图表，涵盖数据的各个维度。</p>
    </div>
"""
        
        # 添加图表网格
        html += """
    <div class="chart-grid">
"""
        
        for chart in charts:
            if chart.get('success', False):
                html += f"""
        <div class="chart-item">
            <div class="chart-title">{chart.get('title', '图表')}</div>
            <div class="chart-description">{chart.get('description', '')}</div>
            <img src="{chart.get('filename', '')}" alt="{chart.get('title', '图表')}" class="chart-image">
        </div>
"""
        
        html += """
    </div>
    
    <div class="footer">
        <p>🤖 由多Agent数据分析系统自动生成</p>
    </div>
</body>
</html>
"""
        
        return html

if __name__ == "__main__":
    # 测试代码
    agent = VisualizationAgent()
    print("可视化Agent初始化完成")
    print(f"支持的图表类型: {list(agent.chart_types.keys())}")