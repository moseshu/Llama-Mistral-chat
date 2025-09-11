"""
数据分析Agent
自动识别数据类型和结构，生成分析策略
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import re
from collections import Counter

class DataAnalyzerAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_strategies = {
            'numerical': self._analyze_numerical_data,
            'categorical': self._analyze_categorical_data,
            'temporal': self._analyze_temporal_data,
            'text': self._analyze_text_data,
            'mixed': self._analyze_mixed_data
        }
    
    def analyze_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析处理后的数据，生成分析策略和建议
        """
        try:
            if not processed_data.get('success', False):
                return {
                    'success': False,
                    'error': '输入数据处理失败',
                    'analysis_type': 'error'
                }
            
            data_type = processed_data.get('data_type', '')
            data = processed_data.get('data', {})
            
            # 根据数据类型选择分析策略
            if data_type == 'structured_data':
                return self._analyze_structured_data(data)
            elif data_type in ['document_with_tables', 'ocr_document_with_tables']:
                return self._analyze_document_with_tables(data)
            elif data_type in ['document', 'ocr_document']:
                return self._analyze_document(data)
            elif data_type == 'json_data':
                return self._analyze_json_data(data)
            else:
                return {
                    'success': False,
                    'error': f'不支持的数据类型: {data_type}',
                    'analysis_type': 'error'
                }
                
        except Exception as e:
            self.logger.error(f"数据分析失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': 'error'
            }
    
    def _analyze_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析结构化数据（Excel, CSV等）"""
        try:
            # 如果是Excel文件，可能有多个工作表
            if isinstance(data, dict) and any(isinstance(v, dict) and 'data' in v for v in data.values()):
                # Excel多工作表格式
                sheet_analyses = {}
                for sheet_name, sheet_data in data.items():
                    if isinstance(sheet_data, dict) and 'data' in sheet_data:
                        sheet_analyses[sheet_name] = self._analyze_single_dataset(sheet_data)
                
                # 选择最有价值的工作表作为主要分析对象
                main_sheet = self._select_main_sheet(sheet_analyses)
                
                return {
                    'success': True,
                    'analysis_type': 'multi_sheet_structured',
                    'main_analysis': sheet_analyses.get(main_sheet, {}),
                    'all_sheets': sheet_analyses,
                    'recommended_sheet': main_sheet,
                    'visualization_suggestions': self._generate_visualization_suggestions(sheet_analyses.get(main_sheet, {}))
                }
            else:
                # 单一数据集格式
                analysis = self._analyze_single_dataset(data)
                return {
                    'success': True,
                    'analysis_type': 'single_structured',
                    'analysis': analysis,
                    'visualization_suggestions': self._generate_visualization_suggestions(analysis)
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'analysis_type': 'error'}
    
    def _analyze_single_dataset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个数据集"""
        df_data = data.get('data', [])
        columns = data.get('columns', [])
        dtypes = data.get('dtypes', {})
        
        if not df_data or not columns:
            return {'error': '数据为空'}
        
        # 创建DataFrame进行分析
        df = pd.DataFrame(df_data)
        
        analysis = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': columns,
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': dtypes
            },
            'column_analysis': {},
            'data_quality': self._assess_data_quality(df),
            'patterns': self._identify_patterns(df),
            'insights': []
        }
        
        # 分析每一列
        for col in df.columns:
            analysis['column_analysis'][col] = self._analyze_column(df[col])
        
        # 生成洞察
        analysis['insights'] = self._generate_insights(df, analysis)
        
        return analysis
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """分析单列数据"""
        col_analysis = {
            'type': self._detect_column_type(series),
            'unique_count': series.nunique(),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100
        }
        
        if pd.api.types.is_numeric_dtype(series):
            col_analysis.update({
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'quartiles': series.quantile([0.25, 0.5, 0.75]).to_dict()
            })
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            value_counts = series.value_counts().head(10)
            col_analysis.update({
                'top_values': value_counts.to_dict(),
                'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                'frequency_of_most': value_counts.iloc[0] if not value_counts.empty else None
            })
        
        return col_analysis
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """检测列的数据类型"""
        # 检查是否为日期时间
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # 尝试转换为日期时间
        if series.dtype == 'object':
            sample = series.dropna().head(100)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{4}/\d{2}/\d{2}',
                r'\d{2}-\d{2}-\d{4}'
            ]
            
            for pattern in date_patterns:
                if sample.str.match(pattern).sum() > len(sample) * 0.5:
                    return 'datetime'
        
        # 检查数值类型
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                return 'integer'
            else:
                return 'float'
        
        # 检查分类数据
        if series.nunique() < len(series) * 0.1 and series.nunique() < 50:
            return 'categorical'
        
        # 检查文本数据
        if series.dtype == 'object':
            avg_length = series.str.len().mean()
            if avg_length > 50:
                return 'text'
            else:
                return 'categorical'
        
        return 'other'
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        quality_score = max(0, 100 - (missing_cells / total_cells * 100))
        
        issues = []
        if missing_cells > 0:
            issues.append(f"发现 {missing_cells} 个缺失值")
        
        # 检查重复行
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"发现 {duplicates} 行重复数据")
            quality_score -= duplicates / len(df) * 20
        
        return {
            'quality_score': round(quality_score, 2),
            'missing_percentage': round(missing_cells / total_cells * 100, 2),
            'duplicate_rows': duplicates,
            'issues': issues
        }
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """识别数据模式"""
        patterns = {
            'correlations': {},
            'trends': {},
            'outliers': {}
        }
        
        # 计算数值列之间的相关性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # 找出强相关性（>0.7或<-0.7）
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': round(corr_val, 3)
                        })
            patterns['correlations'] = strong_corr
        
        # 检测异常值
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                patterns['outliers'][col] = len(outliers)
        
        return patterns
    
    def _generate_insights(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """生成数据洞察"""
        insights = []
        
        # 数据规模洞察
        rows, cols = df.shape
        insights.append(f"数据集包含 {rows} 行记录和 {cols} 个字段")
        
        # 数据质量洞察
        quality = analysis['data_quality']
        if quality['quality_score'] > 90:
            insights.append("数据质量优秀，缺失值很少")
        elif quality['quality_score'] > 70:
            insights.append("数据质量良好，但存在一些缺失值需要处理")
        else:
            insights.append("数据质量需要改善，存在较多缺失值或重复数据")
        
        # 列类型分布洞察
        col_types = [analysis['column_analysis'][col]['type'] for col in df.columns]
        type_counts = Counter(col_types)
        
        if type_counts['integer'] + type_counts.get('float', 0) > len(df.columns) * 0.6:
            insights.append("数据集以数值型数据为主，适合进行统计分析和趋势分析")
        elif type_counts.get('categorical', 0) > len(df.columns) * 0.5:
            insights.append("数据集包含大量分类数据，适合进行分组分析和对比研究")
        
        # 相关性洞察
        correlations = analysis['patterns']['correlations']
        if correlations:
            strong_corr = [c for c in correlations if abs(c['correlation']) > 0.8]
            if strong_corr:
                insights.append(f"发现 {len(strong_corr)} 对变量存在强相关性，可能存在因果关系")
        
        # 异常值洞察
        outliers = analysis['patterns']['outliers']
        if outliers:
            total_outliers = sum(outliers.values())
            insights.append(f"检测到 {total_outliers} 个异常值，需要进一步分析其原因")
        
        return insights
    
    def _generate_visualization_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成可视化建议"""
        if 'error' in analysis:
            return []
        
        suggestions = []
        col_analysis = analysis.get('column_analysis', {})
        
        # 基于列类型生成建议
        numeric_cols = [col for col, info in col_analysis.items() if info['type'] in ['integer', 'float']]
        categorical_cols = [col for col, info in col_analysis.items() if info['type'] == 'categorical']
        datetime_cols = [col for col, info in col_analysis.items() if info['type'] == 'datetime']
        
        # 数值分布图
        if numeric_cols:
            suggestions.append({
                'type': 'histogram',
                'title': '数值分布直方图',
                'columns': numeric_cols[:3],  # 最多显示3个
                'priority': 'high',
                'description': '展示数值型数据的分布情况'
            })
            
            # 箱线图检测异常值
            suggestions.append({
                'type': 'boxplot',
                'title': '箱线图（异常值检测）',
                'columns': numeric_cols[:3],
                'priority': 'medium',
                'description': '识别数值数据中的异常值'
            })
        
        # 分类数据饼图/柱状图
        if categorical_cols:
            for col in categorical_cols[:2]:  # 最多2个分类变量
                unique_count = col_analysis[col]['unique_count']
                if unique_count <= 10:
                    suggestions.append({
                        'type': 'pie',
                        'title': f'{col} 分布饼图',
                        'columns': [col],
                        'priority': 'high',
                        'description': f'展示 {col} 的分布比例'
                    })
                else:
                    suggestions.append({
                        'type': 'bar',
                        'title': f'{col} 分布柱状图',
                        'columns': [col],
                        'priority': 'medium',
                        'description': f'展示 {col} 的分布情况（前10项）'
                    })
        
        # 时间序列图
        if datetime_cols and numeric_cols:
            suggestions.append({
                'type': 'line',
                'title': '时间序列趋势图',
                'columns': [datetime_cols[0], numeric_cols[0]],
                'priority': 'high',
                'description': '展示数值随时间的变化趋势'
            })
        
        # 相关性热力图
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'heatmap',
                'title': '相关性热力图',
                'columns': numeric_cols,
                'priority': 'medium',
                'description': '展示数值变量之间的相关性'
            })
        
        # 散点图（双变量关系）
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'title': '散点图（变量关系）',
                'columns': numeric_cols[:2],
                'priority': 'medium',
                'description': '展示两个数值变量之间的关系'
            })
        
        return suggestions
    
    def _select_main_sheet(self, sheet_analyses: Dict[str, Dict]) -> str:
        """选择最有价值的工作表"""
        if not sheet_analyses:
            return None
        
        # 评分标准：数据行数、列数、数据质量
        best_sheet = None
        best_score = -1
        
        for sheet_name, analysis in sheet_analyses.items():
            if 'error' in analysis:
                continue
            
            basic_info = analysis.get('basic_info', {})
            data_quality = analysis.get('data_quality', {})
            
            # 计算评分
            score = 0
            score += min(basic_info.get('rows', 0) / 1000, 10)  # 行数评分（最高10分）
            score += min(basic_info.get('columns', 0), 10)     # 列数评分（最高10分）
            score += data_quality.get('quality_score', 0) / 10  # 质量评分（最高10分）
            
            if score > best_score:
                best_score = score
                best_sheet = sheet_name
        
        return best_sheet
    
    def _analyze_document_with_tables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析包含表格的文档"""
        try:
            tables = data.get('tables', [])
            text = data.get('text', '')
            
            if not tables:
                return self._analyze_document(data)
            
            # 分析每个表格
            table_analyses = []
            for i, table in enumerate(tables):
                if len(table) > 1:  # 至少有标题行和数据行
                    df = pd.DataFrame(table[1:], columns=table[0])
                    table_analysis = self._analyze_single_dataset({
                        'data': df.to_dict('records'),
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict()
                    })
                    table_analyses.append({
                        'table_index': i,
                        'analysis': table_analysis,
                        'visualization_suggestions': self._generate_visualization_suggestions(table_analysis)
                    })
            
            return {
                'success': True,
                'analysis_type': 'document_with_tables',
                'text_length': len(text),
                'table_count': len(tables),
                'table_analyses': table_analyses,
                'recommended_table': 0 if table_analyses else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'analysis_type': 'error'}
    
    def _analyze_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析纯文档"""
        text = data.get('text', '')
        
        return {
            'success': True,
            'analysis_type': 'document',
            'text_length': len(text),
            'word_count': len(text.split()),
            'insights': [
                f"文档包含 {len(text)} 个字符",
                f"约 {len(text.split())} 个词语",
                "建议使用文本分析工具进行进一步处理"
            ],
            'visualization_suggestions': [
                {
                    'type': 'wordcloud',
                    'title': '词云图',
                    'columns': ['text'],
                    'priority': 'medium',
                    'description': '展示文档中的关键词'
                }
            ]
        }
    
    def _analyze_json_data(self, data: Any) -> Dict[str, Any]:
        """分析JSON数据"""
        try:
            if isinstance(data, list):
                return {
                    'success': True,
                    'analysis_type': 'json_array',
                    'length': len(data),
                    'insights': [f"JSON数组包含 {len(data)} 个元素"],
                    'visualization_suggestions': []
                }
            elif isinstance(data, dict):
                return {
                    'success': True,
                    'analysis_type': 'json_object',
                    'keys': list(data.keys()),
                    'insights': [f"JSON对象包含 {len(data)} 个键"],
                    'visualization_suggestions': []
                }
            else:
                return {
                    'success': True,
                    'analysis_type': 'json_primitive',
                    'value_type': type(data).__name__,
                    'insights': [f"JSON数据类型: {type(data).__name__}"],
                    'visualization_suggestions': []
                }
        except Exception as e:
            return {'success': False, 'error': str(e), 'analysis_type': 'error'}

if __name__ == "__main__":
    # 测试代码
    agent = DataAnalyzerAgent()
    print("数据分析Agent初始化完成")
    print(f"支持的分析策略: {list(agent.analysis_strategies.keys())}")