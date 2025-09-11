#!/usr/bin/env python3
"""
多Agent数据分析系统演示脚本
创建示例数据并演示系统功能
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json
from orchestrator import MultiAgentOrchestrator

def create_demo_data():
    """创建演示数据"""
    print("📊 创建演示数据...")
    
    # 创建示例目录
    os.makedirs('demo_data', exist_ok=True)
    
    # 1. 销售数据 (Excel)
    print("  - 创建销售数据 (sales_data.xlsx)")
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    sales_data = []
    products = ['产品A', '产品B', '产品C', '产品D', '产品E']
    regions = ['北京', '上海', '广州', '深圳', '杭州']
    
    for date in dates[:100]:  # 前100天的数据
        for _ in range(np.random.randint(3, 8)):  # 每天3-7条记录
            sales_data.append({
                '日期': date,
                '产品名称': np.random.choice(products),
                '销售区域': np.random.choice(regions),
                '销售数量': np.random.randint(10, 200),
                '单价': round(np.random.uniform(50, 500), 2),
                '销售额': 0,  # 将在后面计算
                '销售员': f'员工{np.random.randint(1, 21):02d}',
                '客户类型': np.random.choice(['个人', '企业', '政府']),
                '折扣率': round(np.random.uniform(0, 0.3), 2)
            })
    
    df_sales = pd.DataFrame(sales_data)
    df_sales['销售额'] = df_sales['销售数量'] * df_sales['单价'] * (1 - df_sales['折扣率'])
    df_sales['销售额'] = df_sales['销售额'].round(2)
    
    # 保存到Excel
    with pd.ExcelWriter('demo_data/sales_data.xlsx', engine='openpyxl') as writer:
        df_sales.to_excel(writer, sheet_name='销售明细', index=False)
        
        # 创建汇总表
        summary = df_sales.groupby('产品名称').agg({
            '销售数量': 'sum',
            '销售额': 'sum'
        }).round(2)
        summary.to_excel(writer, sheet_name='产品汇总')
        
        # 区域汇总
        region_summary = df_sales.groupby('销售区域').agg({
            '销售数量': 'sum',
            '销售额': 'sum'
        }).round(2)
        region_summary.to_excel(writer, sheet_name='区域汇总')
    
    # 2. 用户行为数据 (CSV)
    print("  - 创建用户行为数据 (user_behavior.csv)")
    user_data = []
    for i in range(1000):
        user_data.append({
            '用户ID': f'U{i+1:04d}',
            '年龄': np.random.randint(18, 65),
            '性别': np.random.choice(['男', '女']),
            '城市': np.random.choice(['北京', '上海', '广州', '深圳', '成都', '西安', '武汉']),
            '注册时间': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            '活跃天数': np.random.randint(1, 100),
            '消费金额': round(np.random.exponential(500), 2),
            '订单数量': np.random.randint(1, 50),
            '评分': round(np.random.uniform(3.0, 5.0), 1),
            '会员等级': np.random.choice(['普通', '银卡', '金卡', '钻石'])
        })
    
    df_users = pd.DataFrame(user_data)
    df_users.to_csv('demo_data/user_behavior.csv', index=False, encoding='utf-8-sig')
    
    # 3. 财务数据 (JSON)
    print("  - 创建财务数据 (financial_data.json)")
    financial_data = {
        "company_info": {
            "name": "示例科技有限公司",
            "year": 2023,
            "currency": "CNY"
        },
        "quarterly_data": [
            {
                "quarter": "Q1",
                "revenue": 12500000,
                "costs": 8300000,
                "profit": 4200000,
                "employees": 150
            },
            {
                "quarter": "Q2", 
                "revenue": 15800000,
                "costs": 9200000,
                "profit": 6600000,
                "employees": 165
            },
            {
                "quarter": "Q3",
                "revenue": 18200000,
                "costs": 10500000,
                "profit": 7700000,
                "employees": 180
            },
            {
                "quarter": "Q4",
                "revenue": 21000000,
                "costs": 11800000,
                "profit": 9200000,
                "employees": 195
            }
        ],
        "monthly_expenses": [
            {"month": "01", "rent": 250000, "salaries": 3200000, "marketing": 800000, "other": 450000},
            {"month": "02", "rent": 250000, "salaries": 3300000, "marketing": 750000, "other": 480000},
            {"month": "03", "rent": 250000, "salaries": 3400000, "marketing": 900000, "other": 520000},
            {"month": "04", "rent": 280000, "salaries": 3600000, "marketing": 1100000, "other": 550000},
            {"month": "05", "rent": 280000, "salaries": 3700000, "marketing": 950000, "other": 580000},
            {"month": "06", "rent": 280000, "salaries": 3800000, "marketing": 1200000, "other": 600000}
        ]
    }
    
    with open('demo_data/financial_data.json', 'w', encoding='utf-8') as f:
        json.dump(financial_data, f, ensure_ascii=False, indent=2)
    
    # 4. 文本报告 (TXT)
    print("  - 创建市场研究报告 (market_research.txt)")
    report_text = """
2023年度市场研究报告

一、市场概况
本年度市场整体呈现稳步增长态势，同比增长15.2%。主要驱动因素包括：
1. 消费者需求持续增长
2. 技术创新推动产品升级
3. 政策支持力度加大

二、竞争分析
市场主要参与者分析：
- 公司A：市场份额32%，优势在于品牌知名度
- 公司B：市场份额28%，技术实力强
- 公司C：市场份额18%，成本控制能力突出
- 其他公司：合计22%

三、用户画像
目标用户主要特征：
- 年龄分布：25-45岁占70%
- 地域分布：一二线城市占80%
- 消费能力：中高收入群体为主
- 偏好特点：注重品质和服务

四、趋势预测
未来发展趋势：
1. 数字化转型加速
2. 个性化需求增长
3. 可持续发展重要性提升
4. 跨界合作增多

五、建议
1. 加强技术研发投入
2. 优化产品结构
3. 拓展新兴市场
4. 提升客户体验

数据来源：市场调研、用户访谈、行业报告
报告日期：2023年12月
    """.strip()
    
    with open('demo_data/market_research.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✅ 演示数据创建完成！")
    print(f"📁 数据文件保存在: {os.path.abspath('demo_data')}")
    return [
        'demo_data/sales_data.xlsx',
        'demo_data/user_behavior.csv', 
        'demo_data/financial_data.json',
        'demo_data/market_research.txt'
    ]

def run_demo_analysis(file_paths):
    """运行演示分析"""
    print("\n🚀 开始演示分析...")
    
    # 初始化协调器
    orchestrator = MultiAgentOrchestrator()
    
    # 分析每个文件
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n📊 分析文件 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
        print("-" * 50)
        
        try:
            # 运行分析
            result = orchestrator.analyze_file(file_path)
            
            if result.get('success', False):
                print("✅ 分析完成！")
                
                # 显示摘要
                summary = result.get('summary', {})
                print(f"📈 创建图表: {summary.get('visualizations_created', 0)} 个")
                print(f"📝 生成报告: {summary.get('reports_generated', 0)} 种格式")
                
                # 显示主要发现
                key_findings = summary.get('key_findings', [])
                if key_findings:
                    print("🔍 主要发现:")
                    for j, finding in enumerate(key_findings[:3], 1):
                        print(f"  {j}. {finding}")
                
                # 显示输出文件
                output_files = summary.get('output_files', [])
                if output_files:
                    print("📄 生成文件:")
                    for file in output_files[:5]:  # 显示前5个
                        print(f"  - {file}")
                    if len(output_files) > 5:
                        print(f"  ... 还有 {len(output_files) - 5} 个文件")
                
            else:
                print(f"❌ 分析失败: {result.get('error_message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 分析异常: {e}")
    
    print(f"\n🎉 演示完成！")
    print("📁 查看生成的文件:")
    print("  - charts/     : 可视化图表")
    print("  - reports/    : 分析报告") 
    print("  - generated_code/ : Python代码")

def main():
    """主函数"""
    print("🤖 多Agent数据分析系统演示")
    print("=" * 40)
    
    try:
        # 创建演示数据
        demo_files = create_demo_data()
        
        # 询问是否运行分析
        print(f"\n📋 已创建 {len(demo_files)} 个演示文件")
        
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--auto':
            run_analysis = True
        else:
            response = input("\n是否立即运行演示分析？(y/n): ").lower()
            run_analysis = response in ['y', 'yes', '是']
        
        if run_analysis:
            run_demo_analysis(demo_files)
        else:
            print("\n💡 您可以:")
            print("1. 运行 'python demo.py --auto' 自动运行完整演示")
            print("2. 运行 'streamlit run app.py' 启动Web界面")
            print("3. 运行 'python orchestrator.py demo_data/sales_data.xlsx' 分析单个文件")
            
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()