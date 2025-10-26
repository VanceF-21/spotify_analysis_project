import pandas as pd
import sqlite3
df = pd.read_csv(
    r"C:\Users\LENOVO\Desktop\硕士学习\AML\Mini-Project\spotify_data\Spotify_Dataset_V3.csv",
    sep=';',          # ✅ 正确分隔符
    engine='python',  # 更稳定处理特殊字符
)

# 2. 连接 / 创建同文件夹下的 database.sqlite
##conn = sqlite3.connect(r"C:\Users\LENOVO\Desktop\硕士学习\AML\Mini-Project\spotify_data\spotify_database.db")
##
### 3. 将 df 存入数据库里的表（表名可自定义）
##df.to_sql('spotify_data', conn, if_exists='replace', index=False)
##
### 4. 关闭数据库连接
##conn.close()

print("✅ 数据已成功保存到 spotify_database.db 中的 'spotify_data' 表！")
print(df.dtypes)

# 更直观的人类可读类型：number / string / datetime / boolean
import pandas.api.types as ptypes
col_types = df.apply(
    lambda s: 'number'   if ptypes.is_numeric_dtype(s) else
              'datetime' if ptypes.is_datetime64_any_dtype(s) else
              'boolean'  if ptypes.is_bool_dtype(s) else
              'string'
)
print(col_types)




# 1) 日期转为 datetime（英式日/月/年）
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# 2) 为 “# of Artist / # of Nationality” 提取序号为整数（可空 Int64）
df['ArtistOrder'] = (
    df['# of Artist'].str.extract(r'(\d+)').astype('Int64')
)
df['NationalityOrder'] = (
    df['# of Nationality'].str.extract(r'(\d+)').astype('Int64')
)

# 3) 可选：把典型分类列转为 category（节省内存/加快分组）
cat_cols = ['Title','Artists','Artist (Ind.)','Nationality','Continent',
            '# of Artist','# of Nationality','id','Song URL']
for c in cat_cols:
    df[c] = df[c].astype('category')

# 4) （可选）确保数值列为数值类型
num_cols = ['Danceability','Energy','Loudness','Speechiness','Acousticness',
            'Instrumentalness','Valence','Points (Total)','Points (Ind for each Artist/Nat)']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')



print(df.dtypes)

# 更直观的人类可读类型：number / string / datetime / boolean
import pandas.api.types as ptypes
col_types = df.apply(
    lambda s: 'number'   if ptypes.is_numeric_dtype(s) else
              'datetime' if ptypes.is_datetime64_any_dtype(s) else
              'boolean'  if ptypes.is_bool_dtype(s) else
              'string'
)
print(col_types)



import os, sqlite3

db_path = r"C:\Users\LENOVO\Desktop\硕士学习\AML\Mini-Project\spotify_data\spotify_database.db"
# 1) 删掉坏库（或改名）
if os.path.exists(db_path):
    os.remove(db_path)

# 2) 重建并写入
conn = sqlite3.connect(db_path)
df.to_sql('spotify_data', conn, if_exists='replace', index=False)
conn.close()
