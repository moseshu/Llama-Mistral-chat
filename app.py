"""
多Agent数据分析系统 - 主应用程序
提供Web界面和文件上传功能
"""
import streamlit as st
import os
import sys
import pandas as pd
import json
from datetime import datetime
import traceback
import zipfile
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from orchestrator import MultiAgentOrchestrator

# 页面配置
st.set_page_config(
    page_title="多Agent数据分析系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.agent-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 1rem;
    color: #155724;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 5px;
    padding: 1rem;
    color: #721c24;
    margin: 1rem 0;
}

.info-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 5px;
    padding: 1rem;
    color: #0c5460;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """初始化session state"""
    if 'orchestrator' not in st.session_state:
        try:
            st.session_state.orchestrator = MultiAgentOrchestrator()
            st.session_state.orchestrator_ready = True
        except Exception as e:
            st.session_state.orchestrator = None
            st.session_state.orchestrator_ready = False
            st.session_state.init_error = str(e)
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def display_header():
    """显示页面头部"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 多Agent数据分析系统</h1>
        <p>智能化数据分析，支持多种文件格式，自动生成洞察和可视化</p>
    </div>
    """, unsafe_allow_html=True)

def display_agent_status():
    """显示Agent状态"""
    st.sidebar.markdown("### 🔧 系统状态")
    
    if st.session_state.get('orchestrator_ready', False):
        st.sidebar.success("✅ 系统就绪")
        
        agents = [
            ("📁", "文件处理Agent", "支持PDF、Word、Excel、图片等"),
            ("🔍", "数据分析Agent", "自动识别数据类型和结构"),
            ("💻", "代码生成Agent", "生成定制化分析代码"),
            ("📊", "可视化Agent", "创建各种类型图表"),
            ("📝", "报告生成Agent", "生成综合分析报告")
        ]
        
        for icon, name, desc in agents:
            st.sidebar.markdown(f"""
            <div class="agent-card">
                <strong>{icon} {name}</strong><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ 系统初始化失败")
        if 'init_error' in st.session_state:
            st.sidebar.error(f"错误: {st.session_state.init_error}")

def file_upload_section():
    """文件上传部分"""
    st.markdown("## 📁 文件上传")
    
    # 支持的文件类型
    supported_types = [
        "csv", "xlsx", "xls", "pdf", "docx", "txt", "json",
        "png", "jpg", "jpeg"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "选择要分析的文件",
            type=supported_types,
            help="支持CSV、Excel、PDF、Word、图片等多种格式"
        )
    
    with col2:
        st.markdown("### 支持的文件类型")
        st.markdown("""
        - **表格数据**: CSV, Excel (.xlsx, .xls)
        - **文档**: PDF, Word (.docx)
        - **图片**: PNG, JPG (OCR提取)
        - **其他**: TXT, JSON
        """)
    
    return uploaded_file

def analysis_options_section():
    """分析选项部分"""
    st.markdown("## ⚙️ 分析选项")
    
    col1, col2 = st.columns(2)
    
    with col1:
        generate_code = st.checkbox("生成分析代码", value=True, 
                                  help="生成可重复执行的Python分析代码")
        create_visualizations = st.checkbox("创建可视化图表", value=True,
                                          help="自动生成柱状图、饼图、折线图等")
    
    with col2:
        generate_report = st.checkbox("生成分析报告", value=True,
                                    help="生成Markdown和HTML格式的综合报告")
        save_intermediate = st.checkbox("保存中间结果", value=False,
                                      help="保存各阶段的处理结果，便于调试")
    
    return {
        'generate_code': generate_code,
        'create_visualizations': create_visualizations,
        'generate_report': generate_report,
        'save_intermediate': save_intermediate
    }

def run_analysis(uploaded_file, options):
    """运行分析"""
    if not st.session_state.get('orchestrator_ready', False):
        st.error("系统未就绪，无法进行分析")
        return None
    
    # 保存上传的文件
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 显示进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 开始分析
        status_text.text("🔄 正在初始化分析...")
        progress_bar.progress(10)
        
        orchestrator = st.session_state.orchestrator
        
        # 分阶段显示进度
        status_text.text("📁 正在处理文件...")
        progress_bar.progress(20)
        
        # 运行完整分析
        result = orchestrator.analyze_file(file_path, options)
        
        if result.get('success', False):
            progress_bar.progress(100)
            status_text.text("✅ 分析完成！")
            
            # 保存结果到session state
            st.session_state.current_analysis = result
            st.session_state.analysis_results.append(result)
            
            return result
        else:
            st.error(f"分析失败: {result.get('error_message', '未知错误')}")
            return None
            
    except Exception as e:
        st.error(f"分析过程中发生错误: {str(e)}")
        st.error(traceback.format_exc())
        return None
    finally:
        # 清理上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)

def display_analysis_results(result):
    """显示分析结果"""
    if not result or not result.get('success', False):
        return
    
    st.markdown("## 📊 分析结果")
    
    # 基本信息
    col1, col2, col3, col4 = st.columns(4)
    
    summary = result.get('summary', {})
    
    with col1:
        st.metric("文件处理", "成功" if summary.get('file_processed', False) else "失败")
    
    with col2:
        st.metric("数据分析", "成功" if summary.get('data_analyzed', False) else "失败")
    
    with col3:
        st.metric("创建图表", summary.get('visualizations_created', 0))
    
    with col4:
        st.metric("生成报告", summary.get('reports_generated', 0))
    
    # 主要发现
    key_findings = summary.get('key_findings', [])
    if key_findings:
        st.markdown("### 🔍 主要发现")
        for i, finding in enumerate(key_findings, 1):
            st.markdown(f"{i}. {finding}")
    
    # 详细结果展示
    tabs = st.tabs(["📋 数据概览", "📈 可视化", "📝 报告", "💻 代码", "🔧 技术细节"])
    
    with tabs[0]:
        display_data_overview(result)
    
    with tabs[1]:
        display_visualizations(result)
    
    with tabs[2]:
        display_reports(result)
    
    with tabs[3]:
        display_generated_code(result)
    
    with tabs[4]:
        display_technical_details(result)

def display_data_overview(result):
    """显示数据概览"""
    stages = result.get('stages', {})
    analysis_result = stages.get('data_analysis', {})
    
    if not analysis_result or not analysis_result.get('success', False):
        st.warning("数据分析阶段失败，无法显示概览")
        return
    
    analysis_type = analysis_result.get('analysis_type', '')
    
    if analysis_type in ['single_structured', 'multi_sheet_structured']:
        main_analysis = analysis_result.get('analysis', {}) or analysis_result.get('main_analysis', {})
        
        # 基本信息
        basic_info = main_analysis.get('basic_info', {})
        if basic_info:
            st.markdown("#### 📊 数据基本信息")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("数据行数", f"{basic_info.get('rows', 0):,}")
            with col2:
                st.metric("数据列数", basic_info.get('columns', 0))
            with col3:
                data_quality = main_analysis.get('data_quality', {})
                quality_score = data_quality.get('quality_score', 0)
                st.metric("数据质量", f"{quality_score:.1f}/100")
        
        # 列信息
        column_analysis = main_analysis.get('column_analysis', {})
        if column_analysis:
            st.markdown("#### 📋 列信息")
            
            col_data = []
            for col_name, col_info in column_analysis.items():
                col_data.append({
                    '列名': col_name,
                    '数据类型': col_info.get('type', '未知'),
                    '唯一值数': col_info.get('unique_count', 0),
                    '缺失值数': col_info.get('null_count', 0),
                    '缺失率(%)': f"{col_info.get('null_percentage', 0):.1f}"
                })
            
            if col_data:
                df_cols = pd.DataFrame(col_data)
                st.dataframe(df_cols, use_container_width=True)
        
        # 数据质量
        data_quality = main_analysis.get('data_quality', {})
        if data_quality:
            st.markdown("#### 🔍 数据质量评估")
            
            issues = data_quality.get('issues', [])
            if issues:
                for issue in issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ 未发现数据质量问题")

def display_visualizations(result):
    """显示可视化结果"""
    stages = result.get('stages', {})
    viz_result = stages.get('visualization', {})
    
    if not viz_result or not viz_result.get('success', False):
        st.warning("可视化创建失败或未启用")
        return
    
    charts = viz_result.get('charts', [])
    
    if not charts:
        st.info("没有生成图表")
        return
    
    st.markdown(f"#### 📈 生成的图表 ({len(charts)} 个)")
    
    # 显示图表
    for i, chart in enumerate(charts):
        if chart.get('success', False):
            st.markdown(f"##### {chart.get('title', f'图表 {i+1}')}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                filename = chart.get('filename', '')
                if filename and os.path.exists(filename):
                    st.image(filename, use_column_width=True)
                else:
                    st.error(f"图表文件不存在: {filename}")
            
            with col2:
                st.markdown(f"**类型**: {chart.get('chart_type', '未知')}")
                st.markdown(f"**描述**: {chart.get('description', '无描述')}")
                
                # 下载按钮
                if filename and os.path.exists(filename):
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="下载图表",
                            data=f.read(),
                            file_name=os.path.basename(filename),
                            mime="image/png"
                        )
    
    # 仪表板
    dashboard = viz_result.get('dashboard', {})
    if dashboard and dashboard.get('success', False):
        st.markdown("#### 📊 交互式仪表板")
        dashboard_file = dashboard.get('dashboard_filename', '')
        
        if dashboard_file and os.path.exists(dashboard_file):
            # 读取HTML内容并显示
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 显示HTML（注意：streamlit有安全限制）
            st.markdown("仪表板已生成，请下载查看完整版本。")
            
            # 提供下载
            with open(dashboard_file, "rb") as f:
                st.download_button(
                    label="下载仪表板 (HTML)",
                    data=f.read(),
                    file_name=os.path.basename(dashboard_file),
                    mime="text/html"
                )

def display_reports(result):
    """显示报告"""
    stages = result.get('stages', {})
    report_result = stages.get('report_generation', {})
    
    if not report_result or not report_result.get('success', False):
        st.warning("报告生成失败或未启用")
        return
    
    reports = report_result.get('reports', {})
    
    if not reports:
        st.info("没有生成报告")
        return
    
    st.markdown("#### 📝 生成的报告")
    
    # 显示不同格式的报告
    for format_name, report_info in reports.items():
        if report_info.get('success', False):
            st.markdown(f"##### {format_name.upper()} 格式")
            
            filename = report_info.get('filename', '')
            
            if format_name == 'markdown':
                # 直接显示Markdown内容
                content = report_info.get('content', '')
                if content:
                    with st.expander("查看Markdown报告", expanded=False):
                        st.markdown(content)
                
                # 下载按钮
                if filename and os.path.exists(filename):
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="下载Markdown报告",
                            data=f.read(),
                            file_name=os.path.basename(filename),
                            mime="text/markdown"
                        )
            
            elif format_name == 'html':
                st.markdown("HTML报告已生成，包含完整的样式和交互功能。")
                
                # 下载按钮
                if filename and os.path.exists(filename):
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="下载HTML报告",
                            data=f.read(),
                            file_name=os.path.basename(filename),
                            mime="text/html"
                        )
            
            elif format_name == 'json':
                # 显示JSON数据
                content = report_info.get('content', {})
                if content:
                    with st.expander("查看JSON数据", expanded=False):
                        st.json(content)
                
                # 下载按钮
                if filename and os.path.exists(filename):
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="下载JSON报告",
                            data=f.read(),
                            file_name=os.path.basename(filename),
                            mime="application/json"
                        )

def display_generated_code(result):
    """显示生成的代码"""
    stages = result.get('stages', {})
    code_result = stages.get('code_generation', {})
    
    if not code_result or not code_result.get('success', False):
        st.warning("代码生成失败或未启用")
        return
    
    code_content = code_result.get('code', '')
    code_filename = code_result.get('code_filename', '')
    
    if not code_content:
        st.info("没有生成代码")
        return
    
    st.markdown("#### 💻 生成的分析代码")
    
    # 代码信息
    sections = code_result.get('sections', [])
    if sections:
        st.markdown("**包含的代码模块:**")
        for section in sections:
            st.markdown(f"- {section}")
    
    # 显示代码
    with st.expander("查看完整代码", expanded=False):
        st.code(code_content, language='python')
    
    # 下载按钮
    if code_filename and os.path.exists(code_filename):
        with open(code_filename, "rb") as f:
            st.download_button(
                label="下载Python代码",
                data=f.read(),
                file_name=os.path.basename(code_filename),
                mime="text/x-python"
            )
    
    # 使用说明
    st.markdown("""
    **使用说明:**
    1. 下载生成的Python代码
    2. 确保安装了所需的Python库
    3. 将数据文件放在代码同一目录下
    4. 运行代码即可重现分析结果
    """)

def display_technical_details(result):
    """显示技术细节"""
    st.markdown("#### 🔧 技术信息")
    
    # 分析ID和时间
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**分析ID**: {result.get('analysis_id', '未知')}")
    with col2:
        timestamp = result.get('timestamp', '')
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            st.markdown(f"**分析时间**: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 各阶段状态
    st.markdown("##### 📋 处理阶段状态")
    stages = result.get('stages', {})
    
    stage_info = [
        ('file_processing', '📁 文件处理'),
        ('data_analysis', '🔍 数据分析'),
        ('code_generation', '💻 代码生成'),
        ('visualization', '📈 可视化'),
        ('report_generation', '📝 报告生成')
    ]
    
    for stage_key, stage_name in stage_info:
        stage_result = stages.get(stage_key, {})
        if stage_result:
            success = stage_result.get('success', False)
            status = "✅ 成功" if success else "❌ 失败"
            st.markdown(f"- **{stage_name}**: {status}")
            
            if not success and 'error' in stage_result:
                st.markdown(f"  - 错误: {stage_result['error']}")
    
    # 输出文件
    summary = result.get('summary', {})
    output_files = summary.get('output_files', [])
    
    if output_files:
        st.markdown("##### 📄 生成的文件")
        for file_path in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                st.markdown(f"- {file_path} ({file_size:,} 字节)")
            else:
                st.markdown(f"- {file_path} (文件不存在)")

def create_download_package():
    """创建下载包"""
    if not st.session_state.current_analysis:
        return None
    
    # 收集所有输出文件
    summary = st.session_state.current_analysis.get('summary', {})
    output_files = summary.get('output_files', [])
    
    if not output_files:
        return None
    
    # 创建ZIP文件
    analysis_id = st.session_state.current_analysis.get('analysis_id', 'unknown')
    zip_filename = f"analysis_results_{analysis_id}.zip"
    zip_path = os.path.join("temp", zip_filename)
    
    os.makedirs("temp", exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in output_files:
                if os.path.exists(file_path):
                    # 保持目录结构
                    arcname = file_path
                    zipf.write(file_path, arcname)
        
        return zip_path
    except Exception as e:
        st.error(f"创建下载包失败: {e}")
        return None

def sidebar_history():
    """侧边栏历史记录"""
    st.sidebar.markdown("### 📚 分析历史")
    
    if not st.session_state.analysis_results:
        st.sidebar.info("暂无分析历史")
        return
    
    for i, result in enumerate(reversed(st.session_state.analysis_results)):
        analysis_id = result.get('analysis_id', f'分析-{i+1}')
        timestamp = result.get('timestamp', '')
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%m-%d %H:%M')
            except:
                time_str = timestamp[:16]
        else:
            time_str = '未知时间'
        
        if st.sidebar.button(f"{analysis_id[:15]}...\n{time_str}", key=f"history_{i}"):
            st.session_state.current_analysis = result
            st.experimental_rerun()

def main():
    """主函数"""
    # 初始化
    init_session_state()
    
    # 显示页面头部
    display_header()
    
    # 侧边栏
    display_agent_status()
    sidebar_history()
    
    # 主要内容区域
    if not st.session_state.get('orchestrator_ready', False):
        st.error("系统初始化失败，请检查环境配置")
        if st.button("重新初始化"):
            st.session_state.clear()
            st.experimental_rerun()
        return
    
    # 文件上传
    uploaded_file = file_upload_section()
    
    # 分析选项
    options = analysis_options_section()
    
    # 分析按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("正在分析，请稍候..."):
                    result = run_analysis(uploaded_file, options)
                    if result:
                        st.success("分析完成！")
                        st.experimental_rerun()
            else:
                st.warning("请先上传文件")
    
    # 显示结果
    if st.session_state.current_analysis:
        display_analysis_results(st.session_state.current_analysis)
        
        # 下载所有结果
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📦 打包下载所有结果"):
                with st.spinner("正在创建下载包..."):
                    zip_path = create_download_package()
                    if zip_path and os.path.exists(zip_path):
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="下载分析结果包",
                                data=f.read(),
                                file_name=os.path.basename(zip_path),
                                mime="application/zip"
                            )
                        # 清理临时文件
                        os.remove(zip_path)
                    else:
                        st.error("创建下载包失败")
        
        with col2:
            if st.button("🗑️ 清理临时文件"):
                # 清理旧文件
                if hasattr(st.session_state.orchestrator, 'cleanup_old_files'):
                    cleaned = st.session_state.orchestrator.cleanup_old_files(days=0)
                    st.success(f"已清理 {cleaned} 个文件")
    
    # 页面底部信息
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🤖 多Agent数据分析系统 | 
        支持多种文件格式 | 
        智能化分析和可视化 | 
        自动生成报告和代码
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()