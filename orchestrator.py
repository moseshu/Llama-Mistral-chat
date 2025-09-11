"""
多Agent协调器
管理文件处理、数据分析、代码生成、可视化和报告生成的整个流程
"""
import os
import sys
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback
import json

# 添加agents目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from file_processor_agent import FileProcessorAgent
from data_analyzer_agent import DataAnalyzerAgent
from code_generator_agent import CodeGeneratorAgent
from visualization_agent import VisualizationAgent
from report_generator_agent import ReportGeneratorAgent

class MultiAgentOrchestrator:
    def __init__(self):
        """初始化多Agent协调器"""
        self.logger = self._setup_logging()
        
        # 初始化各个Agent
        try:
            self.file_processor = FileProcessorAgent()
            self.data_analyzer = DataAnalyzerAgent()
            self.code_generator = CodeGeneratorAgent()
            self.visualization_agent = VisualizationAgent()
            self.report_generator = ReportGeneratorAgent()
            
            self.logger.info("所有Agent初始化成功")
        except Exception as e:
            self.logger.error(f"Agent初始化失败: {e}")
            raise
        
        # 创建输出目录
        self._create_output_directories()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_output_directories(self):
        """创建输出目录"""
        directories = ['charts', 'reports', 'generated_code', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def analyze_file(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        完整的文件分析流程
        
        Args:
            file_path: 要分析的文件路径
            options: 分析选项
                - generate_code: 是否生成分析代码 (默认True)
                - create_visualizations: 是否创建可视化 (默认True)
                - generate_report: 是否生成报告 (默认True)
                - save_intermediate: 是否保存中间结果 (默认False)
        
        Returns:
            包含所有分析结果的字典
        """
        if options is None:
            options = {}
        
        # 默认选项
        default_options = {
            'generate_code': True,
            'create_visualizations': True,
            'generate_report': True,
            'save_intermediate': False
        }
        default_options.update(options)
        options = default_options
        
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"开始分析文件: {file_path} (ID: {analysis_id})")
        
        try:
            # 阶段1: 文件处理
            self.logger.info("阶段1: 文件处理")
            file_result = self._process_file(file_path)
            
            if not file_result.get('success', False):
                return self._create_error_result("文件处理失败", file_result.get('error', ''))
            
            # 阶段2: 数据分析
            self.logger.info("阶段2: 数据分析")
            analysis_result = self._analyze_data(file_result)
            
            if not analysis_result.get('success', False):
                return self._create_error_result("数据分析失败", analysis_result.get('error', ''))
            
            # 阶段3: 代码生成 (可选)
            code_result = None
            if options.get('generate_code', True):
                self.logger.info("阶段3: 代码生成")
                code_result = self._generate_code(analysis_result, file_path)
            
            # 阶段4: 可视化 (可选)
            visualization_result = None
            if options.get('create_visualizations', True):
                self.logger.info("阶段4: 创建可视化")
                visualization_result = self._create_visualizations(file_result, analysis_result)
            
            # 阶段5: 报告生成 (可选)
            report_result = None
            if options.get('generate_report', True):
                self.logger.info("阶段5: 生成报告")
                report_result = self._generate_report(
                    file_result, analysis_result, visualization_result, code_result
                )
            
            # 保存中间结果 (可选)
            if options.get('save_intermediate', False):
                self._save_intermediate_results(
                    analysis_id, file_result, analysis_result, 
                    code_result, visualization_result
                )
            
            # 汇总结果
            final_result = {
                'success': True,
                'analysis_id': analysis_id,
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'stages': {
                    'file_processing': file_result,
                    'data_analysis': analysis_result,
                    'code_generation': code_result,
                    'visualization': visualization_result,
                    'report_generation': report_result
                },
                'summary': self._create_summary(
                    file_result, analysis_result, visualization_result, report_result
                )
            }
            
            self.logger.info(f"分析完成 (ID: {analysis_id})")
            return final_result
            
        except Exception as e:
            error_msg = f"分析过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return self._create_error_result("系统错误", error_msg)
    
    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """处理文件"""
        try:
            result = self.file_processor.process_file(file_path)
            self.logger.info(f"文件处理完成: {result.get('data_type', 'unknown')}")
            return result
        except Exception as e:
            self.logger.error(f"文件处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_data(self, file_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析数据"""
        try:
            result = self.data_analyzer.analyze_data(file_result)
            
            if result.get('success', False):
                analysis_type = result.get('analysis_type', 'unknown')
                self.logger.info(f"数据分析完成: {analysis_type}")
                
                # 记录主要发现
                if analysis_type in ['single_structured', 'multi_sheet_structured']:
                    main_analysis = result.get('analysis', {}) or result.get('main_analysis', {})
                    insights = main_analysis.get('insights', [])
                    self.logger.info(f"发现 {len(insights)} 个洞察")
            
            return result
        except Exception as e:
            self.logger.error(f"数据分析失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_code(self, analysis_result: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """生成分析代码"""
        try:
            result = self.code_generator.generate_analysis_code(analysis_result, file_path)
            
            if result.get('success', False):
                # 保存生成的代码
                code_content = result.get('code', '')
                if code_content:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    code_filename = f"generated_code/analysis_{timestamp}.py"
                    
                    with open(code_filename, 'w', encoding='utf-8') as f:
                        f.write(code_content)
                    
                    result['code_filename'] = code_filename
                    self.logger.info(f"分析代码已保存: {code_filename}")
            
            return result
        except Exception as e:
            self.logger.error(f"代码生成失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_visualizations(self, file_result: Dict[str, Any], 
                             analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建可视化"""
        try:
            # 准备数据
            data = self._prepare_data_for_visualization(file_result, analysis_result)
            
            if data is None or data.empty:
                return {'success': False, 'error': '没有可用于可视化的数据'}
            
            # 获取可视化建议
            suggestions = self._get_visualization_suggestions(analysis_result)
            
            if not suggestions:
                return {'success': False, 'error': '没有可视化建议'}
            
            # 创建可视化
            result = self.visualization_agent.create_visualizations(data, suggestions, analysis_result)
            
            if result.get('success', False):
                chart_count = result.get('total_created', 0)
                self.logger.info(f"创建了 {chart_count} 个图表")
                
                # 创建仪表板
                if chart_count > 0:
                    charts = result.get('charts', [])
                    dashboard_result = self.visualization_agent.create_dashboard(charts, analysis_result)
                    if dashboard_result.get('success', False):
                        result['dashboard'] = dashboard_result
                        self.logger.info(f"仪表板已创建: {dashboard_result.get('dashboard_filename')}")
            
            return result
        except Exception as e:
            self.logger.error(f"可视化创建失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_data_for_visualization(self, file_result: Dict[str, Any], 
                                      analysis_result: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """为可视化准备数据"""
        try:
            data_type = file_result.get('data_type', '')
            data = file_result.get('data', {})
            
            if data_type == 'structured_data':
                # 单一结构化数据
                if 'data' in data and 'columns' in data:
                    df = pd.DataFrame(data['data'])
                    return df
                # Excel多工作表
                elif isinstance(data, dict):
                    # 选择推荐的工作表
                    recommended_sheet = analysis_result.get('recommended_sheet')
                    if recommended_sheet and recommended_sheet in data:
                        sheet_data = data[recommended_sheet]
                        if 'data' in sheet_data:
                            df = pd.DataFrame(sheet_data['data'])
                            return df
                    # 选择第一个有效工作表
                    for sheet_name, sheet_data in data.items():
                        if isinstance(sheet_data, dict) and 'data' in sheet_data:
                            df = pd.DataFrame(sheet_data['data'])
                            return df
            
            elif data_type in ['document_with_tables', 'ocr_document_with_tables']:
                # 文档中的表格
                tables = data.get('tables', [])
                if tables and len(tables[0]) > 1:
                    # 使用第一个表格
                    table = tables[0]
                    df = pd.DataFrame(table[1:], columns=table[0])
                    return df
            
            return None
        except Exception as e:
            self.logger.error(f"数据准备失败: {e}")
            return None
    
    def _get_visualization_suggestions(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取可视化建议"""
        try:
            analysis_type = analysis_result.get('analysis_type', '')
            
            if analysis_type in ['single_structured']:
                analysis = analysis_result.get('analysis', {})
                return analysis.get('visualization_suggestions', [])
            elif analysis_type == 'multi_sheet_structured':
                main_analysis = analysis_result.get('main_analysis', {})
                return main_analysis.get('visualization_suggestions', [])
            elif analysis_type == 'document_with_tables':
                table_analyses = analysis_result.get('table_analyses', [])
                if table_analyses:
                    return table_analyses[0].get('visualization_suggestions', [])
            
            return []
        except Exception as e:
            self.logger.error(f"获取可视化建议失败: {e}")
            return []
    
    def _generate_report(self, file_result: Dict[str, Any], 
                        analysis_result: Dict[str, Any],
                        visualization_result: Dict[str, Any],
                        code_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        try:
            # 准备文件信息
            file_info = {
                'filename': file_result.get('metadata', {}).get('filename', '未知'),
                'file_size': file_result.get('metadata', {}).get('file_size', 0),
                'file_type': file_result.get('metadata', {}).get('file_type', '未知'),
                'processed_at': file_result.get('metadata', {}).get('processed_at', '未知')
            }
            
            result = self.report_generator.generate_comprehensive_report(
                file_info, analysis_result, visualization_result, code_result
            )
            
            if result.get('success', False):
                report_count = len(result.get('reports', {}))
                self.logger.info(f"生成了 {report_count} 种格式的报告")
            
            return result
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_intermediate_results(self, analysis_id: str, *results):
        """保存中间结果"""
        try:
            intermediate_data = {
                'analysis_id': analysis_id,
                'timestamp': datetime.now().isoformat(),
                'results': {
                    'file_processing': results[0],
                    'data_analysis': results[1],
                    'code_generation': results[2],
                    'visualization': results[3]
                }
            }
            
            filename = f"logs/{analysis_id}_intermediate.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"中间结果已保存: {filename}")
        except Exception as e:
            self.logger.error(f"保存中间结果失败: {e}")
    
    def _create_summary(self, file_result: Dict[str, Any], 
                       analysis_result: Dict[str, Any],
                       visualization_result: Dict[str, Any],
                       report_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建分析摘要"""
        summary = {
            'file_processed': file_result.get('success', False),
            'data_analyzed': analysis_result.get('success', False),
            'visualizations_created': 0,
            'reports_generated': 0,
            'key_findings': [],
            'output_files': []
        }
        
        # 可视化统计
        if visualization_result and visualization_result.get('success', False):
            summary['visualizations_created'] = visualization_result.get('total_created', 0)
            
            # 收集图表文件
            charts = visualization_result.get('charts', [])
            for chart in charts:
                if chart.get('success', False):
                    filename = chart.get('filename', '')
                    if filename:
                        summary['output_files'].append(filename)
            
            # 仪表板文件
            dashboard = visualization_result.get('dashboard', {})
            if dashboard.get('success', False):
                dashboard_file = dashboard.get('dashboard_filename', '')
                if dashboard_file:
                    summary['output_files'].append(dashboard_file)
        
        # 报告统计
        if report_result and report_result.get('success', False):
            reports = report_result.get('reports', {})
            summary['reports_generated'] = len(reports)
            
            # 收集报告文件
            for format_name, report_info in reports.items():
                if report_info.get('success', False):
                    filename = report_info.get('filename', '')
                    if filename:
                        summary['output_files'].append(filename)
        
        # 主要发现
        if analysis_result and analysis_result.get('success', False):
            main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
            insights = main_analysis.get('insights', [])
            summary['key_findings'] = insights[:5]  # 前5个发现
        
        return summary
    
    def _create_error_result(self, stage: str, error: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'success': False,
            'error_stage': stage,
            'error_message': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """获取分析状态（如果有异步处理的话）"""
        # 这里可以实现分析状态查询逻辑
        # 目前是同步处理，所以返回简单状态
        return {
            'analysis_id': analysis_id,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def list_output_files(self, analysis_id: str = None) -> List[str]:
        """列出输出文件"""
        output_files = []
        
        # 图表文件
        if os.path.exists('charts'):
            chart_files = [f"charts/{f}" for f in os.listdir('charts') if f.endswith(('.png', '.html'))]
            output_files.extend(chart_files)
        
        # 报告文件
        if os.path.exists('reports'):
            report_files = [f"reports/{f}" for f in os.listdir('reports') if f.endswith(('.md', '.html', '.json'))]
            output_files.extend(report_files)
        
        # 代码文件
        if os.path.exists('generated_code'):
            code_files = [f"generated_code/{f}" for f in os.listdir('generated_code') if f.endswith('.py')]
            output_files.extend(code_files)
        
        return output_files
    
    def cleanup_old_files(self, days: int = 7):
        """清理旧文件"""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        directories = ['charts', 'reports', 'generated_code', 'logs']
        cleaned_files = 0
        
        for directory in directories:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        file_time = os.path.getmtime(filepath)
                        if file_time < cutoff_time:
                            try:
                                os.remove(filepath)
                                cleaned_files += 1
                                self.logger.info(f"删除旧文件: {filepath}")
                            except Exception as e:
                                self.logger.error(f"删除文件失败 {filepath}: {e}")
        
        self.logger.info(f"清理完成，删除了 {cleaned_files} 个旧文件")
        return cleaned_files

if __name__ == "__main__":
    # 测试代码
    orchestrator = MultiAgentOrchestrator()
    print("🤖 多Agent协调器初始化完成")
    print("支持的文件格式: PDF, Word, Excel, CSV, 图片, JSON等")
    print("分析流程: 文件处理 → 数据分析 → 代码生成 → 可视化 → 报告生成")
    
    # 示例用法
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\\n开始分析文件: {file_path}")
        
        result = orchestrator.analyze_file(file_path)
        
        if result.get('success', False):
            print("\\n✅ 分析完成！")
            print(f"分析ID: {result.get('analysis_id')}")
            
            summary = result.get('summary', {})
            print(f"创建图表: {summary.get('visualizations_created', 0)} 个")
            print(f"生成报告: {summary.get('reports_generated', 0)} 种格式")
            print(f"输出文件: {len(summary.get('output_files', []))} 个")
            
            print("\\n主要发现:")
            for i, finding in enumerate(summary.get('key_findings', []), 1):
                print(f"{i}. {finding}")
        else:
            print(f"\\n❌ 分析失败: {result.get('error_message', '未知错误')}")
    else:
        print("\\n使用方法: python orchestrator.py <文件路径>")