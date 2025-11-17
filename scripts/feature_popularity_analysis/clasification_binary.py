import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier 

import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_arguments():
    """
    Parse command-line arguments for model configuration
    
    Returns:
        argparse.Namespace: Parsed arguments containing all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description='Spotify Popularity Prediction with Cluster-based Features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--data_path', type=str, 
                        default='data/task2/spotify_dataset.csv',
                        help='Path to the Spotify dataset CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='scripts/feature_popularity_analysis/results/cls',
                        help='Base directory for output files')
    
    # Model configuration
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of test set (0-1)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        default=['Decision Tree', 'Random Forest', 'Extra Trees', 
                                'Gradient Boosting', 'HistGradient Boosting', 'AdaBoost'],
                        choices=['Decision Tree', 'Random Forest', 'Extra Trees', 
                                'Gradient Boosting', 'HistGradient Boosting', 'AdaBoost'],
                        help='Models to test')
    
    # Random Forest parameters
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--rf_max_depth', type=int, default=None,
                        help='Maximum depth of Random Forest trees')
    
    # Gradient Boosting parameters
    parser.add_argument('--gb_n_estimators', type=int, default=100,
                        help='Number of boosting stages in Gradient Boosting')
    parser.add_argument('--gb_max_depth', type=int, default=5,
                        help='Maximum depth of Gradient Boosting trees')
    parser.add_argument('--gb_learning_rate', type=float, default=0.1,
                        help='Learning rate for Gradient Boosting')
    
    # AdaBoost parameters
    parser.add_argument('--ada_n_estimators', type=int, default=50,
                        help='Number of estimators in AdaBoost')
    
    # Performance
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    
    return parser.parse_args()


# Parse command-line arguments
args = parse_arguments()

# Create model selection dictionary from arguments
MODELS_TO_TEST = {model: True for model in args.models}

# Create timestamped output folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(args.output_dir, f'experiment_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
print(f"Created output folder: {output_dir}")
print(f"\nConfiguration:")
print(f"  - Strategy: Weighted features + Balanced data")
print(f"  - Using: 8 weighted combination features (based on Table 5)")
print(f"  - Random State: {args.random_state}")
print(f"  - Test Size: {args.test_size}")
print(f"  - Models: {', '.join(args.models)}")
print()

# Load data
file_name = args.data_path
print(f"Loading data file: {file_name}")
start_time = time.time()
df = pd.read_csv(file_name, delimiter=',')
load_time = time.time() - start_time
print(f"Data loading completed! Shape: {df.shape} (Time: {load_time:.2f}s)\n")

print("\nLoading Table 5 Cluster Weights")

# Define cluster weights from Table 5
# Each cluster represents a distinct musical style with normalized feature weights
cluster_weights = {
    'C0': {
        'Danceability': 0.3728, 'Loudness': 0.5887, 'Speechiness': 0.2229,
        'Acousticness': 0.1514, 'Instrumentalness': 0.0275, 'Valence': 0.3278
    },
    'C1': {
        'Danceability': 0.8205, 'Loudness': 0.5382, 'Speechiness': 0.2842,
        'Acousticness': 0.1674, 'Instrumentalness': 0.0264, 'Valence': 0.3314
    },
    'C2': {
        'Danceability': 0.4008, 'Loudness': 0.3521, 'Speechiness': 0.2489,
        'Acousticness': 0.4459, 'Instrumentalness': 0.8862, 'Valence': 0.2975
    },
    'C3': {
        'Danceability': 0.7512, 'Loudness': 0.5452, 'Speechiness': 0.7963,
        'Acousticness': 0.2382, 'Instrumentalness': 0.0239, 'Valence': 0.5276
    },
    'C4': {
        'Danceability': 0.4084, 'Loudness': 0.3136, 'Speechiness': 0.2159,
        'Acousticness': 0.8204, 'Instrumentalness': 0.0316, 'Valence': 0.3461
    },
    'C5': {
        'Danceability': 0.7113, 'Loudness': 0.9307, 'Speechiness': 0.2964,
        'Acousticness': 0.2660, 'Instrumentalness': 0.0262, 'Valence': 0.5573
    },
    'C6': {
        'Danceability': 0.5818, 'Loudness': 0.4682, 'Speechiness': 0.2600,
        'Acousticness': 0.3421, 'Instrumentalness': 0.4321, 'Valence': 0.3056
    },
    'C7': {
        'Danceability': 0.7973, 'Loudness': 0.6129, 'Speechiness': 0.2559,
        'Acousticness': 0.2271, 'Instrumentalness': 0.0251, 'Valence': 0.8452
    }
}

# Human-readable descriptions for each cluster
cluster_descriptions = {
    'C0': 'Low-energy Emotional',
    'C1': 'High-energy Party', 
    'C2': 'Instrumental Ambient',
    'C3': 'High-speech Rap',
    'C4': 'Acoustic Quiet',
    'C5': 'Powerful Electronic',
    'C6': 'Medium Instrumental',
    'C7': 'Positive Dance'
}

print(f"Loaded {len(cluster_weights)} cluster weights")

print("\nFeature Engineering: Cluster-weighted Combination Features")

# Define audio features used for clustering
feature_names_for_cluster = ['Danceability', 'Loudness', 'Speechiness', 
                             'Acousticness', 'Instrumentalness', 'Valence']

# Normalize features to [0,1] range to match Table 5 weights
df_normalized = df[feature_names_for_cluster].copy()
for col in feature_names_for_cluster:
    min_val = df_normalized[col].min()
    max_val = df_normalized[col].max()
    if max_val > min_val:
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    else:
        df_normalized[col] = 0

# Create weighted score features for each cluster
# Each weighted score represents how similar a song is to that cluster's style
for cluster_id, weights in cluster_weights.items():
    weighted_score = 0
    for feature, weight in weights.items():
        if feature in df.columns:
            weighted_score += weight * df[feature]
    
    df[f'weighted_{cluster_id}'] = weighted_score

# Calculate Euclidean distance to each cluster
# This helps determine which cluster a song is most similar to
cluster_distances = {}
for cluster_id, weights in cluster_weights.items():
    diff_squared = 0
    for feature in feature_names_for_cluster:
        if feature in weights:
            diff_squared += (df_normalized[feature] - weights[feature]) ** 2
    
    cluster_distances[cluster_id] = np.sqrt(diff_squared)

# Assign each song to its closest cluster
distance_df = pd.DataFrame(cluster_distances)
df['closest_cluster'] = distance_df.idxmin(axis=1)

print("\nTarget Variable Engineering: Creating Balanced Dataset")

# Use median split to create balanced binary target
# Songs above median popularity are labeled as "popular" (1)
median_popularity = df['popularity_class'].median()
print(f"Popularity median: {median_popularity}")

df['is_hit_balanced'] = (df['popularity_class'] >= median_popularity).astype(int)

# Verify class balance
class_dist = df['is_hit_balanced'].value_counts(normalize=True).sort_index()
print(f"\nData balanced: {class_dist[0]:.2%} vs {class_dist[1]:.2%}")

# Define feature sets
cluster_weighted_features = [f'weighted_{c}' for c in cluster_weights.keys()]
numerical_features = cluster_weighted_features
categorical_features = [
    'Nationality', 'Continent', 'is_weekend', 'season', 'closest_cluster'
]

print(f"\nFeature Configuration:")
print(f"  - Weighted combination features: {len(cluster_weighted_features)}")
print(f"  - Categorical features: {len(categorical_features)}")
print(f"  - Total numerical features: {len(numerical_features)}")

# Create feature matrix and target vector
X = df[numerical_features + categorical_features]
y = df['is_hit_balanced']

print(f"\nShape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)
print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Create preprocessing pipeline
# Numerical features: standardize to zero mean and unit variance
# Categorical features: one-hot encode
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
print("Baseline Model Evaluation (Balanced Data)")
print("="*60)

# Define all available models with their configurations
all_baseline_models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=args.random_state),
        'needs_dense': False  # Can handle sparse matrices
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=args.random_state, 
            n_estimators=args.rf_n_estimators, 
            max_depth=args.rf_max_depth,
            n_jobs=args.n_jobs
        ),
        'needs_dense': False
    },
    'Extra Trees': {
        'model': ExtraTreesClassifier(
            random_state=args.random_state, 
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            n_jobs=args.n_jobs
        ),
        'needs_dense': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            random_state=args.random_state, 
            n_estimators=args.gb_n_estimators, 
            max_depth=args.gb_max_depth,
            learning_rate=args.gb_learning_rate
        ),
        'needs_dense': False
    },
    'HistGradient Boosting': {
        'model': HistGradientBoostingClassifier(
            random_state=args.random_state, 
            max_iter=args.gb_n_estimators
        ),
        'needs_dense': True  # Requires dense matrices
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(
            random_state=args.random_state, 
            n_estimators=args.ada_n_estimators
        ),
        'needs_dense': False
    },
}

# Filter models based on user selection
baseline_models = {}
models_need_dense = set()

for name, enabled in MODELS_TO_TEST.items():
    if enabled and name in all_baseline_models:
        model_info = all_baseline_models[name]
        baseline_models[name] = model_info['model']
        if model_info['needs_dense']:
            models_need_dense.add(name)

print(f"Selected {len(baseline_models)} models for testing")

# Initialize result storage
baseline_results = {}
baseline_pipelines = {}
baseline_times = {}

total_start = time.time()

# Train and evaluate each model
for i, (name, model) in enumerate(baseline_models.items(), 1):
    print(f"\n[{i}/{len(baseline_models)}] Training {name}...")
    
    model_start = time.time()
    
    # Create appropriate pipeline based on model requirements
    if name in models_need_dense:
        # Create dense matrix pipeline for models that require it
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
        # Use sparse matrix pipeline for efficiency
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    # Train model
    pipe.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipe.predict(X_test)
    
    model_time = time.time() - model_start
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Store results
    baseline_results[name] = {'Accuracy': acc, 'F1 (Binary)': f1}
    baseline_pipelines[name] = pipe
    baseline_times[name] = model_time
    
    print(f"  Accuracy: {acc:.4f} | F1 (Binary): {f1:.4f} | Time: {model_time:.2f}s")

total_baseline_time = time.time() - total_start

# Create results dataframe sorted by F1 score
baseline_df = pd.DataFrame(baseline_results).T.sort_values(by='F1 (Binary)', ascending=False)
print("\nBaseline Model Results (Sorted by F1)")
print(baseline_df.to_markdown(floatfmt=".4f"))
print(f"\nTotal time: {total_baseline_time:.2f}s")

# Select best model based on F1 score
best_model_name = baseline_df.index[0]
best_pipeline = baseline_pipelines[best_model_name]

print(f"\nBest Model: {best_model_name}")

# Generate predictions with best model
y_pred = best_pipeline.predict(X_test)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Less Popular', 'Popular']))

# Generate and save feature importance plot
print("\nGenerating Feature Importance Plot...")

if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    # Extract feature names after preprocessing
    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Get feature importances from the model
    importances = best_pipeline.named_steps['regressor'].feature_importances_
    
    # Create importance dataframe and select top 20
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['Importance'].values, 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'].values, fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top 20 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'), dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved: feature_importance.pdf")
    plt.close()
else:
    print("Best model does not support feature importance")
    importance_df = None

# Save detailed results to text file
txt_filename = os.path.join(output_dir, 'experiment_results.txt')
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("Spotify Popularity Prediction Experiment - Combined Features + Balanced Data\n")
    f.write("="*70 + "\n\n")
    
    # Experiment metadata
    f.write(f"Experiment time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Data size: {df.shape}\n")
    f.write(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}\n\n")
    
    # Feature configuration
    f.write("Feature Configuration:\n")
    f.write(f"  - Weighted combination features: {len(cluster_weighted_features)}\n")
    f.write(f"    {', '.join(cluster_weighted_features)}\n")
    f.write(f"  - Total numerical features: {len(numerical_features)}\n")
    f.write(f"  - Categorical features: {len(categorical_features)}\n")
    f.write(f"    {', '.join(categorical_features)}\n\n")
    
    # Cluster information
    f.write("Cluster Weights Source: Table 5\n")
    f.write(f"  - Popular clusters: C1 (Party), C5 (Electronic), C7 (Dance)\n")
    f.write(f"  - Less popular clusters: C0 (Low-energy), C2 (Instrumental), C4 (Acoustic)\n")
    f.write(f"  - Neutral clusters: C3 (Rap), C6 (Medium Instrumental)\n\n")
    
    # Target variable information
    f.write("Target Variable:\n")
    f.write(f"  - Strategy: Median split for 50%-50% balance\n")
    f.write(f"  - Distribution: {class_dist[0]:.2%} vs {class_dist[1]:.2%}\n\n")
    
    # Model results
    f.write("Baseline Model Results\n")
    f.write("-"*70 + "\n\n")
    f.write(baseline_df.to_string())
    f.write("\n\n")
    
    # Best model details
    f.write(f"Best Model: {best_model_name}\n")
    f.write("-"*70 + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Less Popular', 'Popular']))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")
    
    # Feature importance
    if importance_df is not None:
        f.write("Top 20 Feature Importance\n")
        f.write("-"*70 + "\n\n")
        f.write(importance_df.to_string())

print(f"\nExperiment results saved: {txt_filename}")

# Display final summary
total_time = time.time() - start_time
minutes = int(total_time // 60)
seconds = total_time % 60

print(f"\n{'='*60}")
print(f"Experiment Completed")
print(f"{'='*60}")
print(f"Total time: {minutes}min {seconds:.1f}s")
print(f"Output folder: {output_dir}")
print(f"  - Feature importance plot: feature_importance.pdf")
print(f"  - Experiment results: experiment_results.txt")
print(f"{'='*60}")
print(f"\nBest Model: {best_model_name}")
print(f"   Accuracy:  {baseline_results[best_model_name]['Accuracy']:.4f}")
print(f"   F1 Score:  {baseline_results[best_model_name]['F1 (Binary)']:.4f}")