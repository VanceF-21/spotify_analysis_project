import pandas as pd

# =================================================================
# 🚩 1. 配置区域
# =================================================================

# --- 文件路径 ---
input_file_path = '/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/data/data_clean_all.csv' 
output_file_path = '/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/data/data_sampled_5000.csv'

# --- 抽样参数 ---
desired_sample_size = 5000

# --- 抽样方式控制 ---
# 更改这个参数的值来选择抽样方式
# 选项: 'random'    (随机抽取)
#       'sequential' (按顺序抽取)
SAMPLING_METHOD = 'sequential' 

# =================================================================

print(f"准备加载文件: {input_file_path}")

try:
    # 1. 加载数据
    df = pd.read_csv(input_file_path, delimiter=';')
    
    total_rows = len(df)
    print(f"✓ 文件加载成功. 总行数: {total_rows}")

    # 2. 检查行数是否足够
    if total_rows < desired_sample_size:
        print(f"警告: 您请求 {desired_sample_size} 行, 但文件只有 {total_rows} 行。")
        print(f"将改为选择全部 {total_rows} 行。")
        actual_sample_size = total_rows
    else:
        actual_sample_size = desired_sample_size

    # 3. 根据 SAMPLING_METHOD 参数执行抽样
    
    sampled_df = None # 初始化一个变量来存储抽样结果

    if SAMPLING_METHOD == 'random':
        print(f"\n--- 正在执行: 随机抽样 ---")
        print(f"正在随机抽取 {actual_sample_size} 行...")
        # random_state=42 确保您每次运行都能得到相同的随机结果
        sampled_df = df.sample(n=actual_sample_size, random_state=42)
        print(f"✓ 随机抽样完成.")
        
    elif SAMPLING_METHOD == 'sequential':
        print(f"\n--- 正在执行: 顺序抽样 ---")
        print(f"正在按顺序抽取前 {actual_sample_size} 行...")
        # 使用 .head() 来获取前 N 行
        sampled_df = df.head(actual_sample_size)
        print(f"✓ 顺序抽样完成.")
        
    else:
        # 如果设置了无效的方法，则抛出错误
        raise ValueError(f"错误: 无效的 SAMPLING_METHOD。请选择 'random' 或 'sequential'。")

    # 4. 保存到新文件
    if sampled_df is not None:
        print(f"新数据集形状: {sampled_df.shape}")
        # 使用分号作为分隔符，并设置 index=False
        sampled_df.to_csv(output_file_path, index=False, sep=';')
        
        print(f"\n🎉 成功! 抽样数据已保存到: {output_file_path}")
    
except FileNotFoundError:
    print(f"错误: 文件未找到。")
    print(f"请检查路径是否正确: {input_file_path}")
except Exception as e:
    print(f"发生了一个错误: {e}")