"""
报告生成Agent
整合分析结果生成完整的分析报告
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import base64

class ReportGeneratorAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.report_templates = {
            'executive_summary': self._generate_executive_summary,
            'data_overview': self._generate_data_overview,
            'analysis_findings': self._generate_analysis_findings,
            'visualizations': self._generate_visualization_section,
            'recommendations': self._generate_recommendations,
            'technical_details': self._generate_technical_details
        }
    
    def generate_comprehensive_report(self, 
                                    file_info: Dict[str, Any],
                                    analysis_result: Dict[str, Any],
                                    visualization_result: Dict[str, Any],
                                    generated_code: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成综合分析报告
        """
        try:
            # 收集所有信息
            report_data = {
                'file_info': file_info,
                'analysis_result': analysis_result,
                'visualization_result': visualization_result,
                'generated_code': generated_code,
                'generation_time': datetime.now(),
                'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # 生成不同格式的报告
            reports = {}
            
            # 1. Markdown报告
            markdown_report = self._generate_markdown_report(report_data)
            reports['markdown'] = markdown_report
            
            # 2. HTML报告
            html_report = self._generate_html_report(report_data)
            reports['html'] = html_report
            
            # 3. JSON报告（结构化数据）
            json_report = self._generate_json_report(report_data)
            reports['json'] = json_report
            
            # 4. 执行摘要
            executive_summary = self._generate_executive_summary(report_data)
            reports['executive_summary'] = executive_summary
            
            return {
                'success': True,
                'reports': reports,
                'report_id': report_data['report_id'],
                'generation_time': report_data['generation_time'].isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成Markdown格式报告"""
        try:
            report_id = report_data['report_id']
            generation_time = report_data['generation_time']
            file_info = report_data.get('file_info', {})
            analysis_result = report_data.get('analysis_result', {})
            visualization_result = report_data.get('visualization_result', {})
            
            markdown_content = []
            
            # 标题和元信息
            markdown_content.extend([
                f"# 数据分析报告",
                f"",
                f"**报告ID:** {report_id}",
                f"**生成时间:** {generation_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**原始文件:** {file_info.get('filename', '未知')}",
                f"",
                "---",
                ""
            ])
            
            # 执行摘要
            executive_summary = self._generate_executive_summary(report_data)
            markdown_content.extend([
                "## 📊 执行摘要",
                "",
                executive_summary.get('content', ''),
                "",
                "---",
                ""
            ])
            
            # 数据概览
            data_overview = self._generate_data_overview(report_data)
            markdown_content.extend([
                "## 📋 数据概览",
                "",
                data_overview.get('content', ''),
                "",
                "---",
                ""
            ])
            
            # 分析发现
            analysis_findings = self._generate_analysis_findings(report_data)
            markdown_content.extend([
                "## 🔍 关键发现",
                "",
                analysis_findings.get('content', ''),
                "",
                "---",
                ""
            ])
            
            # 可视化结果
            if visualization_result.get('success', False):
                viz_section = self._generate_visualization_section(report_data)
                markdown_content.extend([
                    "## 📈 数据可视化",
                    "",
                    viz_section.get('content', ''),
                    "",
                    "---",
                    ""
                ])
            
            # 建议和结论
            recommendations = self._generate_recommendations(report_data)
            markdown_content.extend([
                "## 💡 建议与结论",
                "",
                recommendations.get('content', ''),
                "",
                "---",
                ""
            ])
            
            # 技术细节
            technical_details = self._generate_technical_details(report_data)
            markdown_content.extend([
                "## 🔧 技术细节",
                "",
                technical_details.get('content', ''),
                ""
            ])
            
            # 保存文件
            filename = f"reports/{report_id}.md"
            os.makedirs('reports', exist_ok=True)
            
            full_content = "\n".join(markdown_content)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            return {
                'success': True,
                'filename': filename,
                'content': full_content,
                'format': 'markdown'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成HTML格式报告"""
        try:
            report_id = report_data['report_id']
            generation_time = report_data['generation_time']
            
            # 获取各个部分的内容
            executive_summary = self._generate_executive_summary(report_data)
            data_overview = self._generate_data_overview(report_data)
            analysis_findings = self._generate_analysis_findings(report_data)
            viz_section = self._generate_visualization_section(report_data)
            recommendations = self._generate_recommendations(report_data)
            technical_details = self._generate_technical_details(report_data)
            
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析报告 - {report_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .section {{
            background: white;
            margin-bottom: 25px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        .section-title {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 24px;
        }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px;
            margin: 10px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-image {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .insight-box {{
            background: #e8f6f3;
            border-left: 4px solid #1abc9c;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .warning-box {{
            background: #fdf2e9;
            border-left: 4px solid #e67e22;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
            font-size: 14px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
        }}
        .code-block {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 数据分析报告</h1>
            <p>报告ID: {report_id}</p>
            <p>生成时间: {generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 执行摘要</h2>
            {self._format_html_content(executive_summary.get('content', ''))}
        </div>
        
        <div class="section">
            <h2 class="section-title">📋 数据概览</h2>
            {self._format_html_content(data_overview.get('content', ''))}
        </div>
        
        <div class="section">
            <h2 class="section-title">🔍 关键发现</h2>
            {self._format_html_content(analysis_findings.get('content', ''))}
        </div>
        
        <div class="section">
            <h2 class="section-title">📈 数据可视化</h2>
            {self._format_html_content(viz_section.get('content', ''))}
        </div>
        
        <div class="section">
            <h2 class="section-title">💡 建议与结论</h2>
            {self._format_html_content(recommendations.get('content', ''))}
        </div>
        
        <div class="section">
            <h2 class="section-title">🔧 技术细节</h2>
            {self._format_html_content(technical_details.get('content', ''))}
        </div>
        
        <div class="footer">
            <p>🤖 本报告由多Agent数据分析系统自动生成</p>
        </div>
    </div>
</body>
</html>
"""
            
            # 保存文件
            filename = f"reports/{report_id}.html"
            os.makedirs('reports', exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'success': True,
                'filename': filename,
                'content': html_content,
                'format': 'html'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_json_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成JSON格式报告"""
        try:
            report_id = report_data['report_id']
            
            # 构建结构化数据
            json_data = {
                'report_metadata': {
                    'report_id': report_id,
                    'generation_time': report_data['generation_time'].isoformat(),
                    'version': '1.0',
                    'generator': 'Multi-Agent Data Analysis System'
                },
                'file_information': report_data.get('file_info', {}),
                'analysis_summary': self._extract_analysis_summary(report_data),
                'visualizations': self._extract_visualization_info(report_data),
                'insights': self._extract_insights(report_data),
                'recommendations': self._extract_recommendations(report_data),
                'technical_metadata': self._extract_technical_metadata(report_data)
            }
            
            # 保存文件
            filename = f"reports/{report_id}.json"
            os.makedirs('reports', exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            return {
                'success': True,
                'filename': filename,
                'content': json_data,
                'format': 'json'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_executive_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要"""
        file_info = report_data.get('file_info', {})
        analysis_result = report_data.get('analysis_result', {})
        visualization_result = report_data.get('visualization_result', {})
        
        content_parts = []
        
        # 基本信息
        filename = file_info.get('filename', '未知文件')
        file_type = file_info.get('file_type', '未知')
        content_parts.append(f"本报告分析了文件 **{filename}** ({file_type}格式)。")
        
        # 数据规模
        if analysis_result.get('success', False):
            analysis_type = analysis_result.get('analysis_type', '')
            if analysis_type in ['single_structured', 'multi_sheet_structured']:
                main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
                basic_info = main_analysis.get('basic_info', {})
                if basic_info:
                    rows = basic_info.get('rows', 0)
                    cols = basic_info.get('columns', 0)
                    content_parts.append(f"数据集包含 **{rows:,}** 行记录和 **{cols}** 个字段。")
        
        # 数据质量评估
        if analysis_result.get('success', False):
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            data_quality = main_analysis.get('data_quality', {})
            if data_quality:
                quality_score = data_quality.get('quality_score', 0)
                if quality_score >= 90:
                    content_parts.append("数据质量**优秀**，适合进行深入分析。")
                elif quality_score >= 70:
                    content_parts.append("数据质量**良好**，存在少量需要处理的问题。")
                else:
                    content_parts.append("数据质量**需要改善**，建议先进行数据清洗。")
        
        # 可视化成果
        if visualization_result.get('success', False):
            chart_count = visualization_result.get('total_created', 0)
            if chart_count > 0:
                content_parts.append(f"本次分析生成了 **{chart_count}** 个可视化图表，全面展示数据特征。")
        
        # 主要洞察
        if analysis_result.get('success', False):
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            insights = main_analysis.get('insights', [])
            if insights:
                content_parts.append("\\n**主要洞察：**")
                for i, insight in enumerate(insights[:3], 1):  # 只显示前3个
                    content_parts.append(f"{i}. {insight}")
        
        return {
            'content': '\\n\\n'.join(content_parts),
            'type': 'executive_summary'
        }
    
    def _generate_data_overview(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据概览"""
        analysis_result = report_data.get('analysis_result', {})
        
        content_parts = []
        
        if not analysis_result.get('success', False):
            return {'content': '数据分析失败，无法生成概览。', 'type': 'data_overview'}
        
        analysis_type = analysis_result.get('analysis_type', '')
        
        if analysis_type in ['single_structured', 'multi_sheet_structured']:
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            
            # 基本信息
            basic_info = main_analysis.get('basic_info', {})
            if basic_info:
                content_parts.extend([
                    "### 📊 数据基本信息",
                    f"- **数据行数：** {basic_info.get('rows', 0):,}",
                    f"- **数据列数：** {basic_info.get('columns', 0)}",
                    f"- **列名：** {', '.join(basic_info.get('column_names', []))}"
                ])
            
            # 数据类型分布
            column_analysis = main_analysis.get('column_analysis', {})
            if column_analysis:
                type_counts = {}
                for col_info in column_analysis.values():
                    col_type = col_info.get('type', 'unknown')
                    type_counts[col_type] = type_counts.get(col_type, 0) + 1
                
                content_parts.extend([
                    "",
                    "### 📋 数据类型分布",
                ])
                for data_type, count in type_counts.items():
                    content_parts.append(f"- **{data_type}：** {count} 列")
            
            # 数据质量
            data_quality = main_analysis.get('data_quality', {})
            if data_quality:
                content_parts.extend([
                    "",
                    "### 🔍 数据质量评估",
                    f"- **质量评分：** {data_quality.get('quality_score', 0):.1f}/100",
                    f"- **缺失值比例：** {data_quality.get('missing_percentage', 0):.1f}%",
                    f"- **重复行数：** {data_quality.get('duplicate_rows', 0)}"
                ])
                
                issues = data_quality.get('issues', [])
                if issues:
                    content_parts.append("- **数据问题：**")
                    for issue in issues:
                        content_parts.append(f"  - {issue}")
        
        elif analysis_type == 'document_with_tables':
            table_analyses = analysis_result.get('table_analyses', [])
            content_parts.extend([
                "### 📄 文档信息",
                f"- **文档类型：** 包含表格的文档",
                f"- **表格数量：** {len(table_analyses)}",
            ])
            
            if table_analyses:
                main_table = table_analyses[0].get('analysis', {})
                basic_info = main_table.get('basic_info', {})
                if basic_info:
                    content_parts.extend([
                        "",
                        "### 📊 主要表格信息",
                        f"- **行数：** {basic_info.get('rows', 0):,}",
                        f"- **列数：** {basic_info.get('columns', 0)}",
                    ])
        
        elif analysis_type in ['document', 'ocr_document']:
            content_parts.extend([
                "### 📄 文档信息",
                "- **类型：** 纯文本文档",
                "- **建议：** 使用文本分析工具进行进一步处理"
            ])
        
        return {
            'content': '\\n'.join(content_parts),
            'type': 'data_overview'
        }
    
    def _generate_analysis_findings(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析发现"""
        analysis_result = report_data.get('analysis_result', {})
        
        content_parts = []
        
        if not analysis_result.get('success', False):
            return {'content': '分析过程中出现错误，无法生成发现。', 'type': 'analysis_findings'}
        
        # 获取主要分析结果
        main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
        
        # 洞察发现
        insights = main_analysis.get('insights', [])
        if insights:
            content_parts.extend([
                "### 🔍 数据洞察",
                ""
            ])
            for i, insight in enumerate(insights, 1):
                content_parts.append(f"{i}. {insight}")
        
        # 模式识别
        patterns = main_analysis.get('patterns', {})
        if patterns:
            # 相关性发现
            correlations = patterns.get('correlations', [])
            if correlations:
                content_parts.extend([
                    "",
                    "### 🔗 相关性分析",
                    ""
                ])
                for corr in correlations[:5]:  # 最多显示5个
                    col1 = corr.get('col1', '')
                    col2 = corr.get('col2', '')
                    corr_val = corr.get('correlation', 0)
                    strength = "强" if abs(corr_val) > 0.8 else "中等"
                    direction = "正" if corr_val > 0 else "负"
                    content_parts.append(f"- **{col1}** 与 **{col2}** 存在{strength}{direction}相关性 (r={corr_val:.3f})")
            
            # 异常值发现
            outliers = patterns.get('outliers', {})
            if outliers:
                content_parts.extend([
                    "",
                    "### ⚠️ 异常值检测",
                    ""
                ])
                for col, count in outliers.items():
                    content_parts.append(f"- **{col}** 列发现 {count} 个异常值")
        
        # 列级别的重要发现
        column_analysis = main_analysis.get('column_analysis', {})
        if column_analysis:
            interesting_findings = []
            
            for col, info in column_analysis.items():
                col_type = info.get('type', '')
                
                if col_type in ['integer', 'float']:
                    # 数值列的有趣发现
                    if info.get('std', 0) == 0:
                        interesting_findings.append(f"**{col}** 所有值相同，无变异性")
                    elif info.get('null_percentage', 0) > 50:
                        interesting_findings.append(f"**{col}** 超过50%的值为空")
                
                elif col_type == 'categorical':
                    # 分类列的有趣发现
                    unique_count = info.get('unique_count', 0)
                    total_count = main_analysis.get('basic_info', {}).get('rows', 1)
                    if unique_count == total_count:
                        interesting_findings.append(f"**{col}** 每个值都是唯一的，可能是标识符")
                    elif unique_count == 1:
                        interesting_findings.append(f"**{col}** 只有一个唯一值")
            
            if interesting_findings:
                content_parts.extend([
                    "",
                    "### 💡 特殊发现",
                    ""
                ])
                for finding in interesting_findings:
                    content_parts.append(f"- {finding}")
        
        if not content_parts:
            content_parts = ["暂未发现特殊模式或异常情况，数据表现正常。"]
        
        return {
            'content': '\\n'.join(content_parts),
            'type': 'analysis_findings'
        }
    
    def _generate_visualization_section(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成可视化部分"""
        visualization_result = report_data.get('visualization_result', {})
        
        content_parts = []
        
        if not visualization_result.get('success', False):
            return {'content': '可视化生成失败。', 'type': 'visualization'}
        
        charts = visualization_result.get('charts', [])
        total_created = visualization_result.get('total_created', 0)
        total_failed = visualization_result.get('total_failed', 0)
        
        content_parts.extend([
            f"本次分析共生成了 **{total_created}** 个图表",
            f"其中 {total_failed} 个图表生成失败。" if total_failed > 0 else "",
            ""
        ])
        
        if charts:
            content_parts.append("### 📈 生成的图表")
            content_parts.append("")
            
            for i, chart in enumerate(charts, 1):
                if chart.get('success', False):
                    chart_type = chart.get('chart_type', '未知')
                    title = chart.get('title', '无标题')
                    description = chart.get('description', '')
                    filename = chart.get('filename', '')
                    
                    content_parts.extend([
                        f"**{i}. {title}**",
                        f"- 类型: {chart_type}",
                        f"- 描述: {description}",
                        f"- 文件: {filename}",
                        ""
                    ])
        
        # 可视化建议
        analysis_result = report_data.get('analysis_result', {})
        if analysis_result.get('success', False):
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            viz_suggestions = main_analysis.get('visualization_suggestions', [])
            
            if viz_suggestions:
                content_parts.extend([
                    "### 💡 其他可视化建议",
                    ""
                ])
                
                for suggestion in viz_suggestions:
                    viz_type = suggestion.get('type', '')
                    title = suggestion.get('title', '')
                    description = suggestion.get('description', '')
                    priority = suggestion.get('priority', 'medium')
                    
                    priority_icon = {'high': '🔥', 'medium': '⭐', 'low': '💡'}.get(priority, '⭐')
                    content_parts.append(f"- {priority_icon} **{title}** ({viz_type}): {description}")
        
        return {
            'content': '\\n'.join(content_parts),
            'type': 'visualization'
        }
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成建议和结论"""
        analysis_result = report_data.get('analysis_result', {})
        visualization_result = report_data.get('visualization_result', {})
        
        content_parts = []
        
        # 数据质量建议
        if analysis_result.get('success', False):
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            data_quality = main_analysis.get('data_quality', {})
            
            if data_quality:
                quality_score = data_quality.get('quality_score', 0)
                missing_percentage = data_quality.get('missing_percentage', 0)
                duplicate_rows = data_quality.get('duplicate_rows', 0)
                
                content_parts.append("### 📊 数据质量建议")
                content_parts.append("")
                
                if quality_score < 70:
                    content_parts.append("- 🔧 **数据清洗**: 当前数据质量较低，建议进行全面的数据清洗")
                
                if missing_percentage > 10:
                    content_parts.append("- 🔍 **缺失值处理**: 缺失值比例较高，建议分析缺失模式并选择合适的填充策略")
                
                if duplicate_rows > 0:
                    content_parts.append("- 🔄 **去重处理**: 发现重复数据，建议去除重复行以提高分析准确性")
        
        # 分析建议
        content_parts.extend([
            "",
            "### 🔍 深入分析建议",
            ""
        ])
        
        if analysis_result.get('success', False):
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            column_analysis = main_analysis.get('column_analysis', {})
            
            # 基于数据类型的建议
            numeric_cols = [col for col, info in column_analysis.items() if info.get('type') in ['integer', 'float']]
            categorical_cols = [col for col, info in column_analysis.items() if info.get('type') == 'categorical']
            datetime_cols = [col for col, info in column_analysis.items() if info.get('type') == 'datetime']
            
            if len(numeric_cols) >= 2:
                content_parts.append("- 📈 **回归分析**: 数值变量较多，可考虑进行回归分析探索变量间关系")
            
            if categorical_cols and numeric_cols:
                content_parts.append("- 📊 **分组分析**: 可按分类变量对数值变量进行分组比较")
            
            if datetime_cols:
                content_parts.append("- ⏰ **时间序列分析**: 发现时间变量，建议进行趋势分析和预测")
            
            # 相关性建议
            patterns = main_analysis.get('patterns', {})
            correlations = patterns.get('correlations', [])
            if correlations:
                content_parts.append("- 🔗 **因果关系探索**: 发现强相关变量，建议深入分析是否存在因果关系")
        
        # 业务应用建议
        content_parts.extend([
            "",
            "### 💼 业务应用建议",
            "",
            "- 📋 **制作仪表板**: 将关键指标制作成实时仪表板，便于日常监控",
            "- 🔄 **定期更新**: 建立数据更新机制，确保分析结果的时效性",
            "- 👥 **团队分享**: 将分析结果与相关团队分享，促进数据驱动决策",
            "- 📈 **持续优化**: 根据业务反馈不断优化分析模型和指标"
        ])
        
        # 技术改进建议
        content_parts.extend([
            "",
            "### 🔧 技术改进建议",
            "",
            "- 🤖 **自动化流程**: 将数据处理和分析流程自动化，提高效率",
            "- 📊 **扩展分析**: 考虑引入机器学习模型进行预测分析",
            "- 🔐 **数据安全**: 确保数据处理过程符合安全和隐私要求",
            "- 📚 **文档记录**: 完善数据字典和分析文档，便于后续维护"
        ])
        
        return {
            'content': '\\n'.join(content_parts),
            'type': 'recommendations'
        }
    
    def _generate_technical_details(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成技术细节"""
        content_parts = []
        
        # 系统信息
        content_parts.extend([
            "### 🔧 分析环境",
            "",
            "- **系统**: 多Agent数据分析系统",
            "- **Python版本**: 3.8+",
            "- **主要库**: pandas, numpy, matplotlib, seaborn, plotly",
            f"- **报告生成时间**: {report_data['generation_time'].strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # 处理流程
        content_parts.extend([
            "### 📋 处理流程",
            "",
            "1. **文件解析**: 自动识别文件格式并提取数据",
            "2. **数据分析**: 识别数据类型、质量评估、模式识别",
            "3. **代码生成**: 根据数据特征生成定制化分析代码",
            "4. **可视化**: 自适应生成多种类型图表",
            "5. **报告生成**: 整合所有结果生成综合报告",
            ""
        ])
        
        # 文件信息
        file_info = report_data.get('file_info', {})
        if file_info:
            content_parts.extend([
                "### 📁 原始文件信息",
                "",
                f"- **文件名**: {file_info.get('filename', '未知')}",
                f"- **文件大小**: {file_info.get('file_size', 0):,} 字节",
                f"- **文件类型**: {file_info.get('file_type', '未知')}",
                f"- **处理时间**: {file_info.get('processed_at', '未知')}",
                ""
            ])
        
        # 生成的代码信息
        generated_code = report_data.get('generated_code', {})
        if generated_code and generated_code.get('success', False):
            sections = generated_code.get('sections', [])
            content_parts.extend([
                "### 💻 生成的代码模块",
                "",
            ])
            for section in sections:
                content_parts.append(f"- {section}")
            content_parts.append("")
        
        # 输出文件列表
        content_parts.extend([
            "### 📄 输出文件",
            "",
            "本次分析生成的文件包括:",
            "",
            "- **报告文件**:",
            f"  - Markdown报告: reports/{report_data['report_id']}.md",
            f"  - HTML报告: reports/{report_data['report_id']}.html",
            f"  - JSON数据: reports/{report_data['report_id']}.json",
            "",
            "- **图表文件**: charts/ 目录下的所有图片文件",
            "",
            "- **代码文件**: 如果生成了分析代码，将保存在相应位置"
        ])
        
        return {
            'content': '\\n'.join(content_parts),
            'type': 'technical_details'
        }
    
    def _format_html_content(self, content: str) -> str:
        """将Markdown内容转换为HTML格式"""
        if not content:
            return ""
        
        # 简单的Markdown到HTML转换
        lines = content.split('\\n')
        html_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                html_lines.append("<br>")
                continue
            
            # 标题
            if line.startswith('### '):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith('## '):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('# '):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            # 列表项
            elif line.startswith('- '):
                if not html_lines or not html_lines[-1].startswith('<ul>'):
                    html_lines.append("<ul>")
                html_lines.append(f"<li>{line[2:]}</li>")
            # 普通段落
            else:
                if html_lines and html_lines[-1].startswith('<ul>'):
                    html_lines.append("</ul>")
                html_lines.append(f"<p>{line}</p>")
        
        # 关闭未闭合的列表
        if html_lines and html_lines[-1].startswith('<ul>'):
            html_lines.append("</ul>")
        
        return '\\n'.join(html_lines)
    
    def _extract_analysis_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取分析摘要信息"""
        analysis_result = report_data.get('analysis_result', {})
        
        if not analysis_result.get('success', False):
            return {}
        
        main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
        
        return {
            'data_shape': main_analysis.get('basic_info', {}),
            'data_quality': main_analysis.get('data_quality', {}),
            'column_types': {col: info.get('type') for col, info in main_analysis.get('column_analysis', {}).items()},
            'patterns': main_analysis.get('patterns', {}),
            'insights': main_analysis.get('insights', [])
        }
    
    def _extract_visualization_info(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取可视化信息"""
        visualization_result = report_data.get('visualization_result', {})
        
        if not visualization_result.get('success', False):
            return {}
        
        return {
            'total_charts': visualization_result.get('total_created', 0),
            'failed_charts': visualization_result.get('total_failed', 0),
            'chart_list': [
                {
                    'type': chart.get('chart_type'),
                    'title': chart.get('title'),
                    'filename': chart.get('filename'),
                    'columns': chart.get('columns', [])
                }
                for chart in visualization_result.get('charts', [])
                if chart.get('success', False)
            ]
        }
    
    def _extract_insights(self, report_data: Dict[str, Any]) -> List[str]:
        """提取洞察信息"""
        analysis_result = report_data.get('analysis_result', {})
        
        if not analysis_result.get('success', False):
            return []
        
        main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
        return main_analysis.get('insights', [])
    
    def _extract_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """提取建议信息"""
        recommendations = self._generate_recommendations(report_data)
        content = recommendations.get('content', '')
        
        # 简单提取建议项
        lines = content.split('\\n')
        rec_list = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                rec_list.append(line[2:])
        
        return rec_list
    
    def _extract_technical_metadata(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取技术元数据"""
        return {
            'system': 'Multi-Agent Data Analysis System',
            'generation_time': report_data['generation_time'].isoformat(),
            'report_id': report_data['report_id'],
            'file_info': report_data.get('file_info', {}),
            'processing_agents': [
                'FileProcessorAgent',
                'DataAnalyzerAgent', 
                'CodeGeneratorAgent',
                'VisualizationAgent',
                'ReportGeneratorAgent'
            ]
        }

if __name__ == "__main__":
    # 测试代码
    agent = ReportGeneratorAgent()
    print("报告生成Agent初始化完成")
    print(f"支持的报告模板: {list(agent.report_templates.keys())}")