import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Model Imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# XGBoost is not available in this environment.

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

try:
    # -----------------------------------------------------------------
    # 1. Load the ORIGINAL data
    # -----------------------------------------------------------------
    file_name = '/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/data/data_clean_all.csv'
    df = pd.read_csv(file_name, delimiter=';')
    print(f"Loaded '{file_name}'. Shape: {df.shape}")
    
    # -----------------------------------------------------------------
    # 2. Define Features (X) and Target (y)
    #    *** 这是您要求的修改 ***
    # -----------------------------------------------------------------
    
    # Select numerical features for the model (不变)
    numerical_features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness', 
        'Acousticness', 'Instrumentalness', 'Valence'
    ]
    
    # Select categorical features (修改)
    # 移除了 'Month', 加入了 'Nationality'
    categorical_features = ['Continent', 'Nationality', 'Artist_Canon']
    
    # Define X and y (不变)
    X = df[numerical_features + categorical_features]
    y = df['Pop_points_total']

    # Check for NaNs in the target variable
    if y.isnull().any():
        print("Target variable 'Pop_points_total' contains NaNs. Dropping them.")
        nan_indices = y[y.isnull()].index
        X = X.drop(index=nan_indices).reset_index(drop=True)
        y = y.drop(index=nan_indices).reset_index(drop=True)

    print(f"\nShape of X (features): {X.shape}")
    print(f"Shape of y (target): {y.shape}")
    # 打印新的特征信息
    print(f"Unique artists ('Artist_Canon') to be encoded: {df['Artist_Canon'].nunique()}")
    print(f"Unique nationalities ('Nationality') to be encoded: {df['Nationality'].nunique()}")
    print(f"Unique continents ('Continent') to be encoded: {df['Continent'].nunique()}")

    # -----------------------------------------------------------------
    # 3. Split the data
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # -----------------------------------------------------------------
    # 4. Create Preprocessing Pipeline (不变)
    #    预处理器会自动处理新的 categorical_features 列表
    # -----------------------------------------------------------------
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        # OneHotEncoder 会自动处理 'Continent', 'Nationality', 和 'Artist_Canon'
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
    # 5. Define, Train, and Evaluate Models (不变)
    # -----------------------------------------------------------------
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest (RM/RF)': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    results = {}
    fitted_pipelines = {}

    print("\n--- Starting Model Training and Evaluation (with Nationality, without Month) ---")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results[name] = {'R2': r2, 'MSE': mse, 'RMSE': rmse}
        fitted_pipelines[name] = pipe
        
        print(f"--- {name} Results ---")
        print(f"  R-squared (R2): {r2:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("\n--- Model Comparison (R-squared) ---")
    results_df = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)
    print(results_df.to_markdown(floatfmt=".4f"))

    # -----------------------------------------------------------------
    # 6. Feature Importance (from Random Forest)
    # -----------------------------------------------------------------
    
    print("\n--- Feature Importances (from Random Forest) ---")
    
    try:
        rf_pipeline = fitted_pipelines['Random Forest (RM/RF)']
        feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = rf_pipeline.named_steps['regressor'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print(f"Total features created after One-Hot Encoding: {len(feature_names)}")
        print("Top 15 most important features:")
        # 打印15个最重要的特征，看看 Nationality 是否上榜
        print(importance_df.head(15).to_markdown(index=False, floatfmt=".4f"))
        
        # Plot feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df.head(15), 
            x='Importance', 
            y='Feature', 
            palette='viridis'
        )
        plt.title('Top 15 Feature Importances (RF - Artist + Nationality)')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('/Users/vancefeng/Desktop/ords/AML/spotify_analysis_project/feature_popularity_analysis/first_trial/feature_importances_rf_artist_nationality.png')
        print("\nFeature importance plot saved as 'feature_importances_rf_artist_nationality.png'")

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")