import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = """# Football Goal Prediction - Kaggle Competition
### ⚽ Predicting if a player will score during a football match

**Objective:** Leverage data analytics, machine learning, and statistical modeling to predict whether a player will score during a match (`scored_flag`).

**Evaluation Metric:** Average Precision (AP)

---
## 1. Import Required Libraries"""

code_1 = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import warnings
import time

warnings.filterwarnings('ignore')
%matplotlib inline"""

markdown_2 = """## 2. Load the Dataset"""

code_2 = """# Assuming datasets are in the same directory as the notebook
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("Data loaded successfully!")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
except FileNotFoundError:
    print("Warning: train.csv or test.csv not found in the current directory.")
    # Create dummy data for illustration if missing
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()"""

markdown_3 = """## 3. Exploratory Data Analysis (EDA)
Let's explore the target distribution."""

code_3 = """if not train_df.empty:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train_df, x='scored_flag')
    plt.title('Distribution of Target (scored_flag)')
    plt.show()
    
    print("Target distribution counts:")
    print(train_df['scored_flag'].value_counts())
    print("\nTarget distribution (%):")
    print(train_df['scored_flag'].value_counts(normalize=True) * 100)"""

markdown_4 = """## 4. Data Preprocessing & Feature Engineering
We will handle missing values, encode categorical variables, convert boolean-like flags, and engineer domain features."""

code_4 = """def preprocess_and_engineer(df, is_train=True, train_medians=None, train_encoders=None):
    df_processed = df.copy()
    
    # 1. Fill missing values for advanced metrics with 0
    adv_metrics = ['avg_xG', 'avg_xA', 'avg_shots', 'avg_key_passes', 
                   'avg_xGChain', 'avg_xGBuildup', 'avg_npxG', 'goal_per_cap', 'xG_to_xA_ratio']
    for col in adv_metrics:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0)
            
    # 2. Convert boolean-like columns to float
    bool_cols = ['has_understat', 'full_match_flag', 'starter_flag', 'substitute_flag', 
                 'card_flag', 'prime_age_flag', 'veteran_flag', 'has_national_team_experience', 
                 'finisher_flag', 'creative_player_flag', 'analytics_coverage_flag', 
                 'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_attacker']
    for col in bool_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(float)
            
    # 3. Fill numerical columns with median
    num_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    
    if is_train:
        train_medians = {}
        for col in num_cols:
            median_val = df_processed[col].median()
            if pd.isna(median_val):
                median_val = 0
            train_medians[col] = median_val
            df_processed[col] = df_processed[col].fillna(median_val)
    else:
        for col in num_cols:
            if col in train_medians:
                df_processed[col] = df_processed[col].fillna(train_medians[col])
            else:
                df_processed[col] = df_processed[col].fillna(0)
                
    # 4. Categorical Label Encoding
    cat_cols = ['position', 'sub_position', 'foot', 'market_value_tier', 
                'competition_type', 'home_away', 'name_x', 'confederation', 'age_bucket']
    
    if is_train:
        train_encoders = {}
        for col in cat_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).fillna('Unknown')
                unique_vals = sorted(df_processed[col].unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                train_encoders[col] = mapping
                df_processed[col] = df_processed[col].map(mapping)
    else:
        for col in cat_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).fillna('Unknown')
                mapping = train_encoders[col]
                unseen_val = len(mapping)
                df_processed[col] = df_processed[col].map(lambda x: mapping.get(x, unseen_val))
                
    # 5. Feature Engineering
    if 'avg_xG' in df_processed.columns and 'avg_xA' in df_processed.columns:
        df_processed['xG_plus_xA'] = df_processed['avg_xG'] + df_processed['avg_xA']
        
    if 'minutes_played' in df_processed.columns and 'age' in df_processed.columns:
        df_processed['minutes_per_age'] = df_processed['minutes_played'] / (df_processed['age'] + 1)
        
    if 'avg_xG' in df_processed.columns and 'minutes_played' in df_processed.columns:
        df_processed['expected_goals_match'] = df_processed['avg_xG'] * df_processed['minutes_played']
        
    if 'avg_shots' in df_processed.columns and 'minutes_played' in df_processed.columns:
        df_processed['expected_shots_match'] = df_processed['avg_shots'] * df_processed['minutes_played']
        
    if 'market_value_before_match' in df_processed.columns and 'age' in df_processed.columns:
        df_processed['value_per_age'] = df_processed['market_value_before_match'] / (df_processed['age'] + 1)
        
    # Drop columns that are IDs, dates, or non-predictive text
    cols_to_drop = ['date', 'player_name', 'home_club_name', 'away_club_name', 
                    'stadium', 'referee', 'country_name', 'country_of_citizenship', 'name_y']
    
    df_processed = df_processed.drop([c for c in cols_to_drop if c in df_processed.columns], axis=1)
    
    if is_train:
        return df_processed, train_medians, train_encoders
    else:
        return df_processed

if not train_df.empty:
    print("Preprocessing train and test data...")
    train_processed, train_medians, train_encoders = preprocess_and_engineer(train_df, is_train=True)
    test_processed = preprocess_and_engineer(test_df, is_train=False, train_medians=train_medians, train_encoders=train_encoders)
    print("Preprocessing completed!")"""

markdown_5 = """## 5. Model Development (LightGBM)
We train a LightGBM Classifier with Stratified 5-Fold Cross-Validation, handling class imbalance natively."""

code_5 = """if not train_df.empty:
    X = train_processed.drop(['appearance_id', 'scored_flag'], axis=1, errors='ignore')
    y = train_processed['scored_flag']
    X_test = test_processed.drop(['appearance_id'], axis=1, errors='ignore')
    
    features = X.columns.tolist()
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # Calculate scale_pos_weight
    num_neg = len(y[y == 0])
    num_pos = len(y[y == 1])
    scale_pos_weight = num_neg / max(num_pos, 1)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'subsample': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_estimators': 1500,
        'n_jobs': -1,
        'verbose': -1
    }
    
    feature_importance_df = pd.DataFrame()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold + 1} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        fold_ap = average_precision_score(y_val, val_preds)
        print(f"Fold {fold + 1} Average Precision: {fold_ap:.4f}")
        
        if len(X_test) > 0:
            test_preds += model.predict_proba(X_test[features])[:, 1] / skf.n_splits
            
        fold_importance_df = pd.DataFrame({
            'Feature': features,
            'importance': model.feature_importances_,
            'fold': fold + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    overall_ap = average_precision_score(y, oof_preds)
    print(f"\\nOverall Out-Of-Fold Average Precision: {overall_ap:.4f}")"""

markdown_6 = """## 6. Feature Importance Visualization"""

code_6 = """if not train_df.empty:
    plt.figure(figsize=(10, 8))
    mean_importance = feature_importance_df.groupby('Feature')['importance'].mean().reset_index()
    sns.barplot(x="importance", y="Feature", 
                data=mean_importance.sort_values(by="importance", ascending=False).head(20))
    plt.title('Top 20 Features by LightGBM Importance (Average across Folds)')
    plt.tight_layout()
    plt.show()"""

markdown_7 = """## 7. Generate Submission File"""

code_7 = """if not test_df.empty:
    submission = pd.DataFrame({
        'appearance_id': test_df['appearance_id'],
        'scored_flag': test_preds
    })
    
    submission.to_csv('solution.csv', index=False)
    print("Submission saved to 'solution.csv'")
    print(submission.head())"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(markdown_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(markdown_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(markdown_7),
    nbf.v4.new_code_cell(code_7)
]

with open('Football_Goal_Prediction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook updated successfully!")
