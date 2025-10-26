import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import time

# 导入各种模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

# 文件操作
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# 🚀 加速配置区域 - 可根据需要调整
# =================================================================
USE_SAMPLING = True          # 是否使用数据采样（大幅加速）
SAMPLE_SIZE = 100000         # 采样数据量（例如从65万采样到10万）
SAMPLE_RATIO = None          # 或使用比例采样（例如 0.3 表示30%）

ENABLE_TUNING = True         # 是否启用自动调参
TOP_N_MODELS = 3             # 选择前N个模型进行调参
CV_FOLDS = 3                 # 交叉验证折数
PATIENCE = 3                 # 容忍多少次不提升后停止（早停策略）
MIN_IMPROVEMENT = 0.0001     # 最小提升阈值（小于此值视为无提升）

# 选择要测试的模型
MODELS_TO_TEST = {
    'Decision Tree': True,
    'Random Forest': True,
    'Extra Trees': True,
    'Gradient Boosting': True,
    'HistGradient Boosting': True,
    'AdaBoost': True,
}

# =================================================================

# --- 准备工作：创建带时间戳的输出文件夹 ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/feature_popularity_analysis/results/experiment_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"创建输出文件夹: {output_dir}")
print(f"\n⚙️  配置:")
print(f"  - 数据采样: {'启用' if USE_SAMPLING else '禁用'}")
if USE_SAMPLING:
    print(f"  - 采样大小: {SAMPLE_SIZE if SAMPLE_SIZE else f'{SAMPLE_RATIO*100:.0f}%'}")
print(f"  - 智能调参: {'启用' if ENABLE_TUNING else '禁用'}")
if ENABLE_TUNING:
    print(f"  - 调参模型数: Top {TOP_N_MODELS}")
    print(f"  - 交叉验证: {CV_FOLDS} 折")
    print(f"  - 早停策略: {PATIENCE} 次不提升则停止")
print()

# --- 主代码开始 ---

# -----------------------------------------------------------------
# 1. Load the ORIGINAL data
# -----------------------------------------------------------------
file_name = '/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/data/data_with_famous_artist.csv'
print(f"正在加载数据文件: {file_name}")
start_time = time.time()
df = pd.read_csv(file_name, delimiter=';')
load_time = time.time() - start_time
print(f"✓ 数据加载完成! Shape: {df.shape} (耗时: {load_time:.2f}秒)\n")

# -----------------------------------------------------------------
# 🚀 加速策略1: 数据采样
# -----------------------------------------------------------------
if USE_SAMPLING:
    original_size = len(df)
    if SAMPLE_SIZE:
        sample_n = min(SAMPLE_SIZE, original_size)
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    elif SAMPLE_RATIO:
        df = df.sample(frac=SAMPLE_RATIO, random_state=42).reset_index(drop=True)
    
    print(f"🚀 数据采样: {original_size} -> {len(df)} 行 (减少 {(1-len(df)/original_size)*100:.1f}%)")
    print(f"   预计训练速度提升: {(original_size/len(df)):.1f}x\n")

# -----------------------------------------------------------------
# 2. 特征工程 (Feature Engineering)
# -----------------------------------------------------------------
print("--- 2. 特征工程 ---")
df['Mood_Score'] = df['Valence'] + df['Energy']
df['Acoustic_vs_Electronic'] = df['Acousticness'] - df['Instrumentalness']
print("已创建新特征: 'Mood_Score' 和 'Acoustic_vs_Electronic'")

# -----------------------------------------------------------------
# 3. 定义特征 (X) 和目标 (y) - 使用 Famous_Artist 替代 Artist_Canon
# -----------------------------------------------------------------
print("\n--- 3. 定义特征与目标 ---")

numerical_features = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 
    'Acousticness', 'Instrumentalness', 'Valence',
    'Mood_Score', 'Acoustic_vs_Electronic'
]

# ✅ 使用 Famous_Artist 替代 Artist_Canon
categorical_features = ['Continent', 'Nationality', 'Famous_Artist']

X = df[numerical_features + categorical_features]
y = df['Pop_points_total']

print(f"数值特征: {len(numerical_features)} 个")
print(f"分类特征: {len(categorical_features)} 个 ({', '.join(categorical_features)})")
print(f"✓ 使用'Famous_Artist'特征 (Top 100著名歌手)")
print(f"  独特的Famous_Artist数量: {df['Famous_Artist'].nunique()}")

# 处理 y 中的 NaNs
if y.isnull().any():
    nan_indices = y.index[y.isnull()]
    X = X.drop(index=nan_indices).reset_index(drop=True)
    y = y.drop(index=nan_indices).reset_index(drop=True)

print(f"\nShape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")

# -----------------------------------------------------------------
# 4. 拆分数据
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# -----------------------------------------------------------------
# 5. 创建预处理 Pipeline
# -----------------------------------------------------------------

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# -----------------------------------------------------------------
# 6. 第一阶段：基准模型测试
# -----------------------------------------------------------------
print("\n" + "="*60)
print("第一阶段：基准模型快速评估")
print("="*60)

all_baseline_models = {
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'needs_dense': False
    },
    'Random Forest': {
        'model': RandomForestRegressor(
            random_state=42, 
            n_estimators=100,
            n_jobs=-1
        ),
        'needs_dense': False
    },
    'Extra Trees': {
        'model': ExtraTreesRegressor(
            random_state=42, 
            n_estimators=100,
            n_jobs=-1
        ),
        'needs_dense': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(
            random_state=42, 
            n_estimators=100,
            max_depth=5
        ),
        'needs_dense': False
    },
    'HistGradient Boosting': {
        'model': HistGradientBoostingRegressor(
            random_state=42, 
            max_iter=100
        ),
        'needs_dense': True
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(
            random_state=42, 
            n_estimators=50
        ),
        'needs_dense': False
    },
}

# 根据配置过滤模型
baseline_models = {}
models_need_dense = set()

for name, enabled in MODELS_TO_TEST.items():
    if enabled and name in all_baseline_models:
        model_info = all_baseline_models[name]
        baseline_models[name] = model_info['model']
        if model_info['needs_dense']:
            models_need_dense.add(name)

print(f"已选择 {len(baseline_models)} 个模型进行测试")
if models_need_dense:
    print(f"注意: {models_need_dense} 需要密集矩阵，将自动转换")

baseline_results = {}
baseline_pipelines = {}
baseline_times = {}

total_start = time.time()

for i, (name, model) in enumerate(baseline_models.items(), 1):
    print(f"\n[{i}/{len(baseline_models)}] Training {name}...")
    
    model_start = time.time()
    
    # 对于需要密集矩阵的模型，创建特殊的 pipeline
    if name in models_need_dense:
        categorical_transformer_dense = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor_dense = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer_dense, categorical_features)
            ],
            remainder='passthrough'
        )
        
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor_dense),
            ('regressor', model)
        ])
    else:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    model_time = time.time() - model_start
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    baseline_results[name] = {'R2': r2, 'MSE': mse, 'RMSE': rmse}
    baseline_pipelines[name] = pipe
    baseline_times[name] = model_time
    
    print(f"  R2: {r2:.4f} | RMSE: {rmse:.4f} | 耗时: {model_time:.2f}秒")

total_baseline_time = time.time() - total_start

baseline_df = pd.DataFrame(baseline_results).T.sort_values(by='R2', ascending=False)
print("\n--- 基准模型结果 (按 R2 排序) ---")
print(baseline_df.to_markdown(floatfmt=".4f"))
print(f"\n第一阶段总耗时: {total_baseline_time:.2f}秒")

# -----------------------------------------------------------------
# 7. 第二阶段：智能自适应超参数调优
# -----------------------------------------------------------------

def greedy_tune_parameter(pipe, X_train, y_train, param_name, param_values, cv_folds, patience, min_improvement):
    """
    贪心搜索单个参数的最佳值，直到性能不再提升
    
    返回: (最佳值, 最佳分数, 搜索历史)
    """
    best_score = -np.inf
    best_value = None
    history = []
    no_improvement_count = 0
    
    print(f"\n  调优参数: {param_name}")
    print(f"  候选值: {param_values}")
    
    for value in param_values:
        # 设置参数
        pipe.set_params(**{param_name: value})
        
        # 交叉验证评估
        scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, 
                                scoring='r2', n_jobs=-1)
        mean_score = scores.mean()
        
        history.append({'value': value, 'score': mean_score})
        
        improvement = mean_score - best_score
        
        print(f"    {param_name}={value}: R2={mean_score:.4f} (提升: {improvement:+.4f})")
        
        # 检查是否有显著提升
        if improvement > min_improvement:
            best_score = mean_score
            best_value = value
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
            # 早停：连续多次无提升
            if no_improvement_count >= patience:
                print(f"    ⚠️ 连续{patience}次无显著提升，提前停止搜索")
                break
    
    print(f"  ✓ 最佳 {param_name}={best_value} (R2={best_score:.4f})")
    return best_value, best_score, history


tuned_results = {}
tuned_pipelines = {}
best_params_dict = {}
tuning_times = {}
tuning_histories = {}

if ENABLE_TUNING and len(baseline_models) > 0:
    # 选择表现最好的前N个模型进行调参
    top_n = min(TOP_N_MODELS, len(baseline_df))
    top_models = baseline_df.head(top_n).index.tolist()
    print(f"\n选择表现最好的{top_n}个模型进行智能调参: {top_models}")
    
    print("\n" + "="*60)
    print("第二阶段：智能自适应超参数调优")
    print("="*60)
    print(f"策略: 贪心搜索 + 早停 (patience={PATIENCE}, min_improvement={MIN_IMPROVEMENT})")
    
    # 定义参数搜索顺序和候选值（从粗到细）
    param_search_configs = {
        'Decision Tree': [
            ('regressor__max_depth', [10, 20, 30, 50, 100, None]),
            ('regressor__min_samples_leaf', [1, 2, 5, 10, 20, 30]),
            ('regressor__min_samples_split', [2, 5, 10, 20, 30]),
        ],
        'Random Forest': [
            ('regressor__n_estimators', [50, 100, 150, 200, 300, 400, 500]),
            ('regressor__max_depth', [10, 15, 20, 25, 30, None]),
            ('regressor__min_samples_leaf', [1, 2, 5, 10, 15, 20]),
        ],
        'Extra Trees': [
            ('regressor__n_estimators', [50, 100, 150, 200, 300, 400, 500]),
            ('regressor__max_depth', [10, 15, 20, 25, 30, None]),
            ('regressor__min_samples_leaf', [1, 2, 5, 10, 15, 20]),
        ],
        'Gradient Boosting': [
            ('regressor__n_estimators', [50, 100, 150, 200, 300, 400]),
            ('regressor__learning_rate', [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]),
            ('regressor__max_depth', [3, 4, 5, 6, 7, 8]),
        ],
        'HistGradient Boosting': [
            ('regressor__max_iter', [50, 100, 150, 200, 300, 400]),
            ('regressor__learning_rate', [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]),
            ('regressor__max_depth', [5, 10, 15, 20, 25, 30, None]),
        ],
        'AdaBoost': [
            ('regressor__n_estimators', [30, 50, 100, 150, 200, 300]),
            ('regressor__learning_rate', [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]),
        ]
    }
    
    tuning_start = time.time()
    
    for i, model_name in enumerate(top_models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(top_models)}] 智能调优: {model_name}")
        print(f"{'='*60}")
        
        model_tuning_start = time.time()
        
        if model_name not in baseline_models:
            print(f"警告: {model_name} 不在 baseline_models 中, 跳过.")
            continue
        
        # 获取基准模型和 pipeline
        base_model = baseline_models[model_name]
        
        # 创建 pipeline
        if model_name in models_need_dense:
            categorical_transformer_dense = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor_for_tuning = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer_dense, categorical_features)
                ],
                remainder='passthrough'
            )
        else:
            preprocessor_for_tuning = preprocessor
        
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor_for_tuning),
            ('regressor', base_model)
        ])
        
        # 获取参数搜索配置
        param_configs = param_search_configs.get(model_name, [])
        
        if not param_configs:
            print(f"⚠️  {model_name} 没有定义参数搜索配置，跳过调优")
            continue
        
        # 逐个参数进行贪心搜索
        best_params = {}
        search_history = {}
        
        for param_name, param_values in param_configs:
            best_value, best_score, history = greedy_tune_parameter(
                pipe, X_train, y_train, param_name, param_values,
                CV_FOLDS, PATIENCE, MIN_IMPROVEMENT
            )
            best_params[param_name] = best_value
            search_history[param_name] = history
        
        # 使用最佳参数组合在测试集上评估
        pipe.set_params(**best_params)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        model_tuning_time = time.time() - model_tuning_start
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        tuned_results[f"{model_name} (Tuned)"] = {'R2': r2, 'MSE': mse, 'RMSE': rmse}
        tuned_pipelines[f"{model_name} (Tuned)"] = pipe
        best_params_dict[model_name] = best_params
        tuning_times[model_name] = model_tuning_time
        tuning_histories[model_name] = search_history
        
        improvement_r2 = r2 - baseline_results[model_name]['R2']
        improvement_rmse = baseline_results[model_name]['RMSE'] - rmse
        
        print(f"\n{'='*60}")
        print(f"✓ {model_name} 调优完成")
        print(f"{'='*60}")
        print(f"最佳参数组合:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\n测试集性能:")
        print(f"  R2: {r2:.4f} (提升: {improvement_r2:+.4f})")
        print(f"  RMSE: {rmse:.4f} (改善: {improvement_rmse:+.4f})")
        print(f"  耗时: {model_tuning_time:.2f}秒")
    
    total_tuning_time = time.time() - tuning_start
    print(f"\n第二阶段总耗时: {total_tuning_time:.2f}秒")

# -----------------------------------------------------------------
# 8. 合并所有结果
# -----------------------------------------------------------------
print("\n" + "="*60)
print("最终结果汇总")
print("="*60)

all_results = {**baseline_results, **tuned_results}
results_df = pd.DataFrame(all_results).T.sort_values(by='R2', ascending=False)

print("\n--- 所有模型性能对比 (按 R2 排序) ---")
print(results_df.to_markdown(floatfmt=".4f"))

all_pipelines = {**baseline_pipelines, **tuned_pipelines}

# -----------------------------------------------------------------
# 9. 最佳模型的特征重要性
# -----------------------------------------------------------------

best_model_name = results_df.index[0]
best_pipeline = all_pipelines[best_model_name]
print(f"\n\n--- 最佳模型: '{best_model_name}' ---")

importance_available = False
current_model_name = best_model_name
current_pipeline = best_pipeline

for idx in range(len(results_df)):
    current_model_name = results_df.index[idx]
    if current_model_name not in all_pipelines:
        continue
    current_pipeline = all_pipelines[current_model_name]
    
    if 'regressor' not in current_pipeline.named_steps:
        continue
        
    if hasattr(current_pipeline.named_steps['regressor'], 'feature_importances_'):
        importance_available = True
        break

if importance_available:
    feature_names = current_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = current_pipeline.named_steps['regressor'].feature_importances_
    
    if current_model_name != best_model_name:
        print(f"注意: '{best_model_name}' 不支持特征重要性。")
        print(f"改用 '{current_model_name}' 的特征重要性进行可视化。")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(f"\nTotal features after encoding: {len(feature_names)}")
    print("\nTop 15 most important features:")
    print(importance_df.head(15).to_markdown(index=False, floatfmt=".4f"))
    
    # 绘图并保存为 PDF
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=importance_df.head(15), 
        x='Importance', 
        y='Feature', 
        palette='viridis'
    )
    plt.title(f'Top 15 Feature Importances\n(from {current_model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    pdf_filename = os.path.join(output_dir, 'feature_importances.pdf')
    plt.savefig(pdf_filename, format='pdf', dpi=300)
    print(f"\n✓ 特征重要性图表已保存: {pdf_filename}")
else:
    print(f"\n无法生成特征重要性图表: 候选模型均不支持 feature_importances_")

# -----------------------------------------------------------------
# 10. 保存详细结果到 TXT 文件
# -----------------------------------------------------------------

txt_filename = os.path.join(output_dir, 'model_experiment_results.txt')
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("机器学习模型实验结果汇总 (智能自适应调参版)\n")
    f.write("="*70 + "\n\n")
    f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据文件: {file_name}\n")
    f.write(f"数据形状: {df.shape}\n")
    f.write(f"训练集大小: {X_train.shape[0]}\n")
    f.write(f"测试集大小: {X_test.shape[0]}\n")
    f.write(f"特征配置: 使用'Famous_Artist'特征 (Top 100著名歌手)\n")
    f.write(f"  Famous_Artist独特值数: {df['Famous_Artist'].nunique()}\n\n")
    
    f.write("配置:\n")
    f.write(f"  - 数据采样: {'启用' if USE_SAMPLING else '禁用'}\n")
    if USE_SAMPLING:
        f.write(f"  - 采样大小: {len(df)}\n")
    f.write(f"  - 智能调参: {'启用' if ENABLE_TUNING else '禁用'}\n")
    if ENABLE_TUNING:
        f.write(f"  - 调参模型数: Top {TOP_N_MODELS}\n")
        f.write(f"  - 早停策略: patience={PATIENCE}, min_improvement={MIN_IMPROVEMENT}\n")
    f.write("\n")
    
    f.write("-"*70 + "\n")
    f.write("第一阶段: 基准模型结果\n")
    f.write("-"*70 + "\n\n")
    f.write(baseline_df.to_string())
    f.write("\n\n训练时间:\n")
    for name, t in baseline_times.items():
        f.write(f"  {name}: {t:.2f}秒\n")
    f.write("\n")
    
    if ENABLE_TUNING and best_params_dict:
        f.write("-"*70 + "\n")
        f.write("第二阶段: 智能调优结果\n")
        f.write("-"*70 + "\n\n")
        for model_name, params in best_params_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  最佳参数:\n")
            for param, value in params.items():
                f.write(f"    {param}: {value}\n")
            
            if model_name in tuning_histories:
                f.write(f"\n  参数搜索历史:\n")
                for param_name, history in tuning_histories[model_name].items():
                    f.write(f"    {param_name}:\n")
                    for record in history:
                        f.write(f"      {record['value']}: R2={record['score']:.4f}\n")
            
            if model_name in tuning_times:
                f.write(f"  调优耗时: {tuning_times[model_name]:.2f}秒\n")
        f.write("\n")
    
    f.write("-"*70 + "\n")
    f.write("最终所有模型性能对比 (按 R2 排序)\n")
    f.write("-"*70 + "\n\n")
    f.write(results_df.to_string())
    f.write("\n\n")
    
    if importance_available:
        f.write("-"*70 + "\n")
        f.write(f"Top 20 特征重要性 (来自 {current_model_name})\n")
        f.write("-"*70 + "\n\n")
        f.write(importance_df.head(20).to_string())
    
    f.write("\n\n")
    f.write("="*70 + "\n")
    f.write(f"最佳模型: {best_model_name}\n")
    f.write(f"R2 Score: {results_df.loc[best_model_name, 'R2']:.4f}\n")
    f.write(f"RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}\n")
    f.write("="*70 + "\n")

# 计算总耗时
total_time = time.time() - start_time
minutes = int(total_time // 60)
seconds = total_time % 60

print(f"\n{'='*60}")
print(f"--- 实验完成 ---")
print(f"{'='*60}")
print(f"总耗时: {minutes}分{seconds:.1f}秒")
print(f"输出文件夹: {output_dir}")
print(f"  ✓ 实验结果摘要: model_experiment_results.txt")
if importance_available:
    print(f"  ✓ 特征重要性图: feature_importances.pdf")
print(f"{'='*60}")

# 显示最终性能对比
print("\n=== 最终模型性能排名 (Top 5) ===")
print(results_df.head().to_markdown(floatfmt=".4f"))

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   R2 Score: {results_df.loc[best_model_name, 'R2']:.4f}")
print(f"   RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}")

# 显示性能提升摘要
if ENABLE_TUNING and best_params_dict:
    print(f"\n📊 调优效果摘要:")
    for model_name in best_params_dict.keys():
        baseline_r2 = baseline_results[model_name]['R2']
        tuned_r2 = tuned_results[f"{model_name} (Tuned)"]['R2']
        improvement = tuned_r2 - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100
        print(f"   {model_name}: R2提升 {improvement:+.4f} ({improvement_pct:+.2f}%)")