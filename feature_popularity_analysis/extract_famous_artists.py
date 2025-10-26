import pandas as pd
import numpy as np
from datetime import datetime
import re # 导入正则表达式库

"""
预处理脚本：提取著名歌手特征 (按频率, v3)

功能：
1. (不变) 计算每个歌手在 'Artists' 列中的总出现次数。
2. (不变) 选出Top N名最高频歌手。
3. (新) 为每首歌创建 "Famous_Artist" 特征：
   - 规则1: 如果作者中有Top N歌手，则返回 *第一个* 匹配到的著名歌手 (按原始顺序)。
   - 规则2: 如果没有Top N歌手，则返回 *第一个* 列出的歌手。
4. (不变) 移除 'Artist_Canon' 列。
"""

print("="*70)
print("著名歌手特征提取工具 (按频率, v3 规则)")
print("="*70)

# =================================================================
# 配置区域
# =================================================================
INPUT_FILE = '/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/data/data_clean_all.csv'
OUTPUT_FILE = '/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/data/data_with_famous_artist.csv'
TOP_N = 100  # 选择前N名著名歌手

# 定义用于拆分 'Artists' 列的分隔符 (正则表达式)
# 涵盖: , ; / & feat. ft. featuring
SEPARATORS_REGEX = r'[;,/&]| feat. | ft. | featuring '

print(f"\n配置:")
print(f"  输入文件: {INPUT_FILE}")
print(f"  输出文件: {OUTPUT_FILE}")
print(f"  著名歌手数量: Top {TOP_N} (按频率)")
print()

# =================================================================
# 1. 加载数据
# =================================================================
print("步骤1: 加载数据...")
df = pd.read_csv(INPUT_FILE, delimiter=';')
print(f"✓ 数据加载完成! Shape: {df.shape}")

# 检查必要的列
if 'Artists' not in df.columns:
    print(f"❌ 错误: 缺少 'Artists' 列")
    exit(1)

# **修改：根据要求移除 'Artist_Canon' 列**
if 'Artist_Canon' in df.columns:
    df = df.drop(columns=['Artist_Canon'])
    print(f"✓ 已移除 'Artist_Canon' 列")
else:
    print(f"⚠️  'Artist_Canon' 列未找到，跳过移除。")
    
print(f"  处理后列名: {df.columns.tolist()}")


# =================================================================
# 2. 计算每个歌手的出现频率 (基于 'Artists' 列)
# =================================================================
print("\n步骤2: 计算每个歌手的出现频率 (基于 'Artists' 列)...")

# (此部分逻辑不变)
artist_counts = df['Artists'].str.split(SEPARATORS_REGEX).explode().str.strip()
artist_stats = artist_counts.value_counts().reset_index()
artist_stats.columns = ['Artist', 'Frequency'] 

print(f"✓ 共有 {len(artist_stats)} 位独特歌手")
print(f"\nTop 10 歌手预览 (按频率):")
print(artist_stats.head(10).to_string(index=False))

# =================================================================
# 3. 选出Top N著名歌手
# =================================================================
print(f"\n步骤3: 选出Top {TOP_N} 著名歌手 (按频率)...")

top_artists_list = artist_stats.head(TOP_N)['Artist'].tolist()
top_artists_set = set(top_artists_list)  # 用于快速查找

print(f"✓ 已选出 {len(top_artists_list)} 位著名歌手")
print(f"\n著名歌手列表 (前20位):")
for i, row in artist_stats.head(20).iterrows():
    print(f"  {i+1:2d}. {row['Artist']:30s} | 出现次数: {row['Frequency']:5d}")

# =================================================================
# 4. 创建"著名歌手"特征 (*** 核心逻辑修改 ***)
# =================================================================
print(f"\n步骤4: 创建'Famous_Artist'特征...")

# 预编译正则表达式以提高速度
separators_regex_compiled = re.compile(SEPARATORS_REGEX)

def extract_artist_by_rule(artists_string):
    """
    从 'Artists' 字符串中按新规则提取歌手
    
    规则:
    1. 拆分 'Artists' 字符串，保持原始顺序。
    2. 遍历拆分后的列表：
       - 如果某个歌手 *在* Top N 集合中，立即返回该歌手。
    3. 如果遍历完所有歌手，都没有找到 Top N 歌手：
       - 返回列表中的 *第一个* 歌手。
    """
    if pd.isna(artists_string):
        return 'Unknown'
    
    # 拆分艺术家列表，并清理首尾空格 (同时移除空字符串)
    artists = [a.strip() for a in separators_regex_compiled.split(artists_string) if a.strip()]
    
    if not artists:
        return 'Unknown' # 如果拆分后为空

    # 规则 1 & 2: 查找第一个在 Top N 列表中的歌手
    for artist in artists:
        if artist in top_artists_set:
            return artist  # 返回第一个匹配到的著名歌手
    
    # 规则 3: 如果没有找到著名歌手，返回列表中的第一个歌手
    return artists[0]


# 应用函数创建新特征
df['Famous_Artist'] = df['Artists'].apply(extract_artist_by_rule)

print(f"✓ 'Famous_Artist' 特征创建完成")

# 统计新特征的分布
famous_artist_counts = df['Famous_Artist'].value_counts()
top_n_in_new_col = sum(df['Famous_Artist'].isin(top_artists_set))
non_top_n_in_new_col = len(df) - top_n_in_new_col

print(f"\n新特征统计:")
print(f"  独特的'Famous_Artist'值数量: {df['Famous_Artist'].nunique()} (这会 > {TOP_N})")
print(f"  行数被归类为Top {TOP_N}歌手: {top_n_in_new_col}")
print(f"  行数被归类为'第一作者' (非著名): {non_top_n_in_new_col}")

print(f"\n'Famous_Artist' Top 10 出现频率:")
print(famous_artist_counts.head(10).to_string())

# =================================================================
# 5. 数据验证
# =================================================================
print(f"\n步骤5: 数据验证...")

# 检查是否有缺失值
missing_count = df['Famous_Artist'].isna().sum()
if missing_count > 0:
    print(f"⚠️  发现 {missing_count} 个缺失值")
else:
    print(f"✓ 无缺失值")

# 对比示例 (移除了 Artist_Canon)
print(f"\n数据对比示例 (前10行):")
comparison_df = df[['Title', 'Artists', 'Famous_Artist', 'Pop_points_total']].head(10)
print(comparison_df.to_string(index=False))

# =================================================================
# 6. 保存处理后的数据
# =================================================================
print(f"\n步骤6: 保存数据...")

df.to_csv(OUTPUT_FILE, sep=';', index=False)
print(f"✓ 数据已保存到: {OUTPUT_FILE}")
print(f"  保存形状: {df.shape}")

# =================================================================
# 7. 生成报告
# =================================================================
report_file = OUTPUT_FILE.replace('.csv', '_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("著名歌手特征提取报告 (按频率, v3规则)\n")
    f.write("="*70 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"输入文件: {INPUT_FILE}\n")
    f.write(f"输出文件: {OUTPUT_FILE}\n")
    f.write(f"数据量: {len(df)} 行\n")
    f.write(f"'Artist_Canon' 列已移除\n\n")
    
    f.write("-"*70 + "\n")
    f.write(f"Top {TOP_N} 著名歌手列表 (按出现频率)\n")
    f.write("-"*70 + "\n\n")
    
    top_artists_stats = artist_stats.head(TOP_N)
    for i, row in top_artists_stats.iterrows():
        f.write(f"{i+1:3d}. {row['Artist']:40s} | 出现次数: {row['Frequency']:5d}\n")
    
    f.write("\n\n")
    f.write("-"*70 + "\n")
    f.write("'Famous_Artist' 特征分布\n")
    f.write("-"*70 + "\n\n")
    f.write(f"独特值数量: {df['Famous_Artist'].nunique()}\n\n")
    
    f.write(f"Top {TOP_N} 著名歌手覆盖率: {top_n_in_new_col / len(df) * 100:.2f}% ({top_n_in_new_col} 行)\n")
    f.write(f"非著名(取第一作者)覆盖率: {non_top_n_in_new_col / len(df) * 100:.2f}% ({non_top_n_in_new_col} 行)\n\n")
    
    f.write("Top 20 出现频率 (在 Famous_Artist 列中):\n")
    for artist, count in famous_artist_counts.head(20).items():
        pct = count / len(df) * 100
        f.write(f"  {artist:40s}: {count:6d} ({pct:5.2f}%)\n")

print(f"✓ 报告已保存到: {report_file}")

print("\n" + "="*70)
print("✓ 预处理完成!")
print("="*70)
print(f"\n下一步:")
print(f"  1. 检查输出文件: {OUTPUT_FILE}")
print(f"  2. 查看详细报告: {report_file}")
print(f"  3. 使用新文件进行模型训练 (新特征列为 'Famous_Artist')")
print("="*70)