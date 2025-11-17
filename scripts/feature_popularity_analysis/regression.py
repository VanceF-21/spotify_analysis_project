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
import argparse

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Spotify Popularity Regression with Intelligent Hyperparameter Tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--data_path', type=str,
                        default='data/task2/data_with_famous_artist.csv',
                        help='Path to the Spotify dataset CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='scripts/feature_popularity_analysis/results/reg',
                        help='Base directory for output files')
    
    # Sampling configuration
    parser.add_argument('--use_sampling', action='store_true',
                        help='Enable data sampling for faster training')
    parser.add_argument('--sample_size', type=int, default=100000,
                        help='Number of samples to use (if use_sampling enabled)')
    parser.add_argument('--sample_ratio', type=float, default=None,
                        help='Proportion of data to sample (alternative to sample_size)')
    
    # Model configuration
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of test set (0-1)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        default=['Linear Regression', 'Decision Tree', 'Random Forest', 'Extra Trees',
                                'Gradient Boosting', 'HistGradient Boosting', 'AdaBoost'],
                        choices=['Linear Regression', 'Decision Tree', 'Random Forest', 'Extra Trees',
                                'Gradient Boosting', 'HistGradient Boosting', 'AdaBoost'],
                        help='Models to test')
    
    # Hyperparameter tuning
    parser.add_argument('--enable_tuning', action='store_true',
                        help='Enable intelligent hyperparameter tuning')
    parser.add_argument('--top_n_models', type=int, default=3,
                        help='Number of top models to tune')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of cross-validation folds')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience for tuning')
    parser.add_argument('--min_improvement', type=float, default=0.0001,
                        help='Minimum improvement threshold for early stopping')
    
    # Random Forest parameters
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    
    # Gradient Boosting parameters
    parser.add_argument('--gb_n_estimators', type=int, default=100,
                        help='Number of boosting stages')
    parser.add_argument('--gb_max_depth', type=int, default=5,
                        help='Maximum depth of Gradient Boosting trees')
    
    # Performance
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    
    return parser.parse_args()


args = parse_arguments()

# Model selection dictionary
MODELS_TO_TEST = {model: True for model in args.models}

# Create timestamped output folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(args.output_dir, f'experiment_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
print(f"Created output folder: {output_dir}")
print(f"\nConfiguration:")
print(f"  - Data sampling: {'Enabled' if args.use_sampling else 'Disabled'}")
if args.use_sampling:
    print(f"  - Sample size: {args.sample_size if args.sample_size else f'{args.sample_ratio*100:.0f}%'}")
print(f"  - Intelligent tuning: {'Enabled' if args.enable_tuning else 'Disabled'}")
if args.enable_tuning:
    print(f"  - Models to tune: Top {args.top_n_models}")
    print(f"  - Cross-validation: {args.cv_folds} folds")
    print(f"  - Early stopping: {args.patience} iterations without improvement")
print()


# Load data
file_name = args.data_path
print(f"Loading data file: {file_name}")
start_time = time.time()
df = pd.read_csv(file_name, delimiter=',')
load_time = time.time() - start_time
print(f"Data loading completed! Shape: {df.shape} (Time: {load_time:.2f}s)\n")


# Data sampling
if args.use_sampling:
    original_size = len(df)
    if args.sample_size:
        sample_n = min(args.sample_size, original_size)
        df = df.sample(n=sample_n, random_state=args.random_state).reset_index(drop=True)
    elif args.sample_ratio:
        df = df.sample(frac=args.sample_ratio, random_state=args.random_state).reset_index(drop=True)
    
    print(f"Data sampling: {original_size} -> {len(df)} rows (reduced {(1-len(df)/original_size)*100:.1f}%)")
    print(f"   Expected training speedup: {(original_size/len(df)):.1f}x\n")


# Define features and target
print("\nDefining Features and Target")

# Numerical features
numerical_features = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 
    'Acousticness', 'Instrumentalness', 'Valence'
]

# Categorical features
categorical_features = [
    'Nationality', 'Continent', 'is_weekend', 
    'season', 'popularity_class', 'is_top10_artist'
]

X = df[numerical_features + categorical_features]
y = df['Pop_points_total']

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)} ({', '.join(categorical_features)})")
print(f"Using feature set with is_top10_artist, season, etc.")
print(f"  Unique 'is_top10_artist' values: {df['is_top10_artist'].nunique()}")
print(f"  Unique 'season' values: {df['season'].nunique()}")


# Handle NaN values in target
if y.isnull().any():
    nan_indices = y.index[y.isnull()]
    X = X.drop(index=nan_indices).reset_index(drop=True)
    y = y.drop(index=nan_indices).reset_index(drop=True)

print(f"\nShape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


# Create preprocessing pipeline
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


# Baseline model testing
print("\n" + "="*60)
print("Phase 1: Baseline Model Quick Evaluation")
print("="*60)

all_baseline_models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'needs_dense': False
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=args.random_state),
        'needs_dense': False
    },
    'Random Forest': {
        'model': RandomForestRegressor(
            random_state=args.random_state, 
            n_estimators=args.rf_n_estimators,
            n_jobs=args.n_jobs
        ),
        'needs_dense': False
    },
    'Extra Trees': {
        'model': ExtraTreesRegressor(
            random_state=args.random_state, 
            n_estimators=args.rf_n_estimators,
            n_jobs=args.n_jobs
        ),
        'needs_dense': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(
            random_state=args.random_state, 
            n_estimators=args.gb_n_estimators,
            max_depth=args.gb_max_depth
        ),
        'needs_dense': False
    },
    'HistGradient Boosting': {
        'model': HistGradientBoostingRegressor(
            random_state=args.random_state, 
            max_iter=args.gb_n_estimators
        ),
        'needs_dense': True
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(
            random_state=args.random_state, 
            n_estimators=50
        ),
        'needs_dense': False
    },
}

# Filter models based on configuration
baseline_models = {}
models_need_dense = set()

for name, enabled in MODELS_TO_TEST.items():
    if enabled and name in all_baseline_models:
        model_info = all_baseline_models[name]
        baseline_models[name] = model_info['model']
        if model_info['needs_dense']:
            models_need_dense.add(name)

print(f"Selected {len(baseline_models)} models for testing")
if models_need_dense:
    print(f"Note: {models_need_dense} require dense matrices, will be converted automatically")

baseline_results = {}
baseline_pipelines = {}
baseline_times = {}

total_start = time.time()

for i, (name, model) in enumerate(baseline_models.items(), 1):
    print(f"\n[{i}/{len(baseline_models)}] Training {name}...")
    
    model_start = time.time()
    
    # Create special pipeline for models requiring dense matrices
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
    
    print(f"  R2: {r2:.4f} | RMSE: {rmse:.4f} | Time: {model_time:.2f}s")

total_baseline_time = time.time() - total_start

baseline_df = pd.DataFrame(baseline_results).T.sort_values(by='R2', ascending=False)
print("\nBaseline Model Results (Sorted by R2)")
print(baseline_df.to_markdown(floatfmt=".4f"))
print(f"\nPhase 1 total time: {total_baseline_time:.2f}s")


# Intelligent adaptive hyperparameter tuning
def greedy_tune_parameter(pipe, X_train, y_train, param_name, param_values, cv_folds, patience, min_improvement):
    """
    Greedy search for optimal parameter value with early stopping
    
    Returns: (best_value, best_score, search_history)
    """
    best_score = -np.inf
    best_value = None
    history = []
    no_improvement_count = 0
    
    print(f"\n  Tuning parameter: {param_name}")
    print(f"  Candidate values: {param_values}")
    
    for value in param_values:
        # Set parameter
        pipe.set_params(**{param_name: value})
        
        # Cross-validation evaluation
        scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, 
                                scoring='r2', n_jobs=-1)
        mean_score = scores.mean()
        
        history.append({'value': value, 'score': mean_score})
        
        improvement = mean_score - best_score
        
        print(f"    {param_name}={value}: R2={mean_score:.4f} (improvement: {improvement:+.4f})")
        
        # Check for significant improvement
        if improvement > min_improvement:
            best_score = mean_score
            best_value = value
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
            # Early stopping if no improvement
            if no_improvement_count >= patience:
                print(f"    Early stopping: {patience} iterations without improvement")
                break
    
    print(f"  Best {param_name}={best_value} (R2={best_score:.4f})")
    return best_value, best_score, history


tuned_results = {}
tuned_pipelines = {}
best_params_dict = {}
tuning_times = {}
tuning_histories = {}

if args.enable_tuning and len(baseline_models) > 0:
    # Select top N models for tuning
    top_n = min(args.top_n_models, len(baseline_df))
    top_models = baseline_df.head(top_n).index.tolist()
    print(f"\nSelected top {top_n} models for intelligent tuning: {top_models}")
    
    print("\n" + "="*60)
    print("Phase 2: Intelligent Adaptive Hyperparameter Tuning")
    print("="*60)
    print(f"Strategy: Greedy search + early stopping (patience={args.patience}, min_improvement={args.min_improvement})")
    
    # Define parameter search configurations (from coarse to fine)
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
        print(f"[{i}/{len(top_models)}] Intelligent Tuning: {model_name}")
        print(f"{'='*60}")
        
        model_tuning_start = time.time()
        
        if model_name not in baseline_models:
            print(f"Warning: {model_name} not in baseline_models, skipping.")
            continue
        
        # Get baseline model and pipeline
        base_model = baseline_models[model_name]
        
        # Create pipeline
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
        
        # Get parameter search configuration
        param_configs = param_search_configs.get(model_name, [])
        
        if not param_configs:
            print(f"{model_name} has no parameter search configuration defined, skipping tuning")
            continue
        
        # Greedy search for each parameter
        best_params = {}
        search_history = {}
        
        for param_name, param_values in param_configs:
            best_value, best_score, history = greedy_tune_parameter(
                pipe, X_train, y_train, param_name, param_values,
                args.cv_folds, args.patience, args.min_improvement
            )
            best_params[param_name] = best_value
            search_history[param_name] = history
        
        # Evaluate on test set with best parameters
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
        print(f"{model_name} tuning completed")
        print(f"{'='*60}")
        print(f"Best parameter combination:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nTest set performance:")
        print(f"  R2: {r2:.4f} (improvement: {improvement_r2:+.4f})")
        print(f"  RMSE: {rmse:.4f} (improvement: {improvement_rmse:+.4f})")
        print(f"  Time: {model_tuning_time:.2f}s")
    
    total_tuning_time = time.time() - tuning_start
    print(f"\nPhase 2 total time: {total_tuning_time:.2f}s")


# Merge all results
print("\n" + "="*60)
print("Final Results Summary")
print("="*60)

all_results = {**baseline_results, **tuned_results}
results_df = pd.DataFrame(all_results).T.sort_values(by='R2', ascending=False)

print("\nAll Model Performance Comparison (Sorted by R2)")
print(results_df.to_markdown(floatfmt=".4f"))

all_pipelines = {**baseline_pipelines, **tuned_pipelines}


# Feature importance of best model
best_model_name = results_df.index[0]
best_pipeline = all_pipelines[best_model_name]
print(f"\n\nBest Model: '{best_model_name}'")

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
    
    # Check for 'feature_importances_' (tree models) or 'coef_' (linear models)
    if hasattr(current_pipeline.named_steps['regressor'], 'feature_importances_') or \
       hasattr(current_pipeline.named_steps['regressor'], 'coef_'):
        importance_available = True
        break

if importance_available:
    feature_names = current_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    if hasattr(current_pipeline.named_steps['regressor'], 'feature_importances_'):
        # Tree models
        importances = current_pipeline.named_steps['regressor'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        plot_title = f'Top 15 Feature Importances\n(from {current_model_name})'
        plot_xlabel = 'Importance Score'
        
    elif hasattr(current_pipeline.named_steps['regressor'], 'coef_'):
        # Linear models
        importances = current_pipeline.named_steps['regressor'].coef_
        # Use absolute value for linear model coefficients
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Abs_Importance': np.abs(importances)
        }).sort_values(by='Abs_Importance', ascending=False)
        plot_title = f'Top 15 Feature Coefficients (Absolute Value)\n(from {current_model_name})'
        plot_xlabel = 'Coefficient (Absolute Value)'
        
        
    if current_model_name != best_model_name:
        print(f"Note: '{best_model_name}' does not support feature importance/coefficients.")
        print(f"Using '{current_model_name}' feature importance/coefficients for visualization.")
    
    
    print(f"\nTotal features after encoding: {len(feature_names)}")
    print("\nTop 15 most important features (by magnitude):")
    print(importance_df.head(15).to_markdown(index=False, floatfmt=".4f"))
    
    # Plot and save as PDF
    plt.figure(figsize=(12, 8))
    
    # Choose plot data based on model type
    if 'Abs_Importance' in importance_df.columns:
        plot_data = importance_df.head(15)
        sns.barplot(
            data=plot_data, 
            x='Abs_Importance', 
            y='Feature', 
            palette='viridis'
        )
    else:
        plot_data = importance_df.head(15)
        sns.barplot(
            data=plot_data, 
            x='Importance', 
            y='Feature', 
            palette='viridis'
        )

    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(plot_xlabel, fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    pdf_filename = os.path.join(output_dir, 'feature_importances.pdf')
    plt.savefig(pdf_filename, format='pdf', dpi=300)
    print(f"\nFeature importance plot saved: {pdf_filename}")
else:
    print(f"\nCannot generate feature importance plot: No candidate models support feature_importances_ or coef_")


# Save detailed results to TXT file
txt_filename = os.path.join(output_dir, 'model_experiment_results.txt')
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("Machine Learning Model Experiment Results Summary (Intelligent Adaptive Tuning)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Experiment time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Data file: {file_name}\n")
    f.write(f"Data shape: {df.shape}\n")
    f.write(f"Training set size: {X_train.shape[0]}\n")
    f.write(f"Test set size: {X_test.shape[0]}\n")
    
    f.write(f"Feature configuration: Using 'is_top10_artist', 'season', 'popularity_class' and other features\n")
    f.write(f"  'is_top10_artist' unique values: {df['is_top10_artist'].nunique()}\n")
    f.write(f"  'season' unique values: {df['season'].nunique()}\n")
    f.write(f"  'popularity_class' unique values: {df['popularity_class'].nunique()}\n\n")

    
    f.write("Configuration:\n")
    f.write(f"  - Data sampling: {'Enabled' if args.use_sampling else 'Disabled'}\n")
    if args.use_sampling:
        f.write(f"  - Sample size: {len(df)}\n")
    f.write(f"  - Intelligent tuning: {'Enabled' if args.enable_tuning else 'Disabled'}\n")
    if args.enable_tuning:
        f.write(f"  - Models to tune: Top {args.top_n_models}\n")
        f.write(f"  - Early stopping: patience={args.patience}, min_improvement={args.min_improvement}\n")
    f.write("\n")
    
    f.write("-"*70 + "\n")
    f.write("Phase 1: Baseline Model Results\n")
    f.write("-"*70 + "\n\n")
    f.write(baseline_df.to_string())
    f.write("\n\nTraining times:\n")
    for name, t in baseline_times.items():
        f.write(f"  {name}: {t:.2f}s\n")
    f.write("\n")
    
    if args.enable_tuning and best_params_dict:
        f.write("-"*70 + "\n")
        f.write("Phase 2: Intelligent Tuning Results\n")
        f.write("-"*70 + "\n\n")
        for model_name, params in best_params_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Best parameters:\n")
            for param, value in params.items():
                f.write(f"    {param}: {value}\n")
            
            if model_name in tuning_histories:
                f.write(f"\n  Parameter search history:\n")
                for param_name, history in tuning_histories[model_name].items():
                    f.write(f"    {param_name}:\n")
                    for record in history:
                        f.write(f"      {record['value']}: R2={record['score']:.4f}\n")
            
            if model_name in tuning_times:
                f.write(f"  Tuning time: {tuning_times[model_name]:.2f}s\n")
        f.write("\n")
    
    f.write("-"*70 + "\n")
    f.write("Final All Model Performance Comparison (Sorted by R2)\n")
    f.write("-"*70 + "\n\n")
    f.write(results_df.to_string())
    f.write("\n\n")
    
    if importance_available:
        f.write("-"*70 + "\n")
        f.write(f"Top 20 Feature Importance (from {current_model_name})\n")
        f.write("-"*70 + "\n\n")
        f.write(importance_df.head(20).to_string())
    
    f.write("\n\n")
    f.write("="*70 + "\n")
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"R2 Score: {results_df.loc[best_model_name, 'R2']:.4f}\n")
    f.write(f"RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}\n")
    f.write("="*70 + "\n")

# Calculate total time
total_time = time.time() - start_time
minutes = int(total_time // 60)
seconds = total_time % 60

print(f"\n{'='*60}")
print(f"Experiment Completed")
print(f"{'='*60}")
print(f"Total time: {minutes}min {seconds:.1f}s")
print(f"Output folder: {output_dir}")
print(f"  - Experiment results: model_experiment_results.txt")
if importance_available:
    print(f"  - Feature importance plot: feature_importances.pdf")
print(f"{'='*60}")

# Display final performance comparison
print("\nFinal Model Performance Ranking (Top 5)")
print(results_df.head().to_markdown(floatfmt=".4f"))

print(f"\nBest model: {best_model_name}")
print(f"   R2 Score: {results_df.loc[best_model_name, 'R2']:.4f}")
print(f"   RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}")

# Display tuning improvement summary
if args.enable_tuning and best_params_dict:
    print(f"\nTuning improvement summary:")
    for model_name in best_params_dict.keys():
        baseline_r2 = baseline_results[model_name]['R2']
        tuned_r2 = tuned_results[f"{model_name} (Tuned)"]['R2']
        improvement = tuned_r2 - baseline_r2
        # Avoid division by zero or negative baseline_r2
        if baseline_r2 > 0:
            improvement_pct = (improvement / baseline_r2) * 100
            print(f"   {model_name}: R2 improvement {improvement:+.4f} ({improvement_pct:+.2f}%)")
        else:
            print(f"   {model_name}: R2 improvement {improvement:+.4f}")