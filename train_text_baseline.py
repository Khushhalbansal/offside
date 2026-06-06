import pandas as pd
import numpy as np
import os
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def extract_features(df):
    df = df.copy()
    # Fill NaN
    df['catalog_content'] = df['catalog_content'].fillna("")
    
    # Text length features
    df['text_len'] = df['catalog_content'].apply(len)
    df['word_count'] = df['catalog_content'].apply(lambda x: len(x.split()))
    
    # Extract numbers (e.g., quantities, weights)
    def extract_numbers(text):
        nums = re.findall(r'\d+\.?\d*', text)
        if len(nums) == 0:
            return 0.0
        return float(nums[0])
    
    df['first_num'] = df['catalog_content'].apply(extract_numbers)
    return df

def main():
    train_path = r'C:\Users\khush\Downloads\student_resource\dataset\train.csv'
    test_path = r'C:\Users\khush\Downloads\student_resource\dataset\test.csv'
    
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    print("Extracting basic features...")
    train = extract_features(train)
    test = extract_features(test)
    
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
    train_tfidf = tfidf.fit_transform(train['catalog_content'])
    test_tfidf = tfidf.transform(test['catalog_content'])
    
    print("Reducing dimensionality with SVD...")
    svd = TruncatedSVD(n_components=128, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    
    # Create feature dataframes
    svd_cols = [f'svd_{i}' for i in range(128)]
    train_svd_df = pd.DataFrame(train_svd, columns=svd_cols)
    test_svd_df = pd.DataFrame(test_svd, columns=svd_cols)
    
    X_train = pd.concat([train[['text_len', 'word_count', 'first_num']], train_svd_df], axis=1)
    y_train = train['price']
    
    X_test = pd.concat([test[['text_len', 'word_count', 'first_num']], test_svd_df], axis=1)
    
    # Log transform target because SMAPE cares about relative error
    # and prices often have a long right tail
    y_train_log = np.log1p(y_train)
    
    print("Training LightGBM models with 5-Fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    lgb_params = {
        'objective': 'regression_l1', # MAE on log(price) closely approximates MAPE/SMAPE
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'subsample': 0.8,
        'verbose': -1,
        'n_estimators': 1500,
        'random_state': 42
    }
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"--- Fold {fold+1} ---")
        X_tr, y_tr = X_train.iloc[trn_idx], y_train_log.iloc[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train_log.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds_log = model.predict(X_va)
        val_preds = np.expm1(val_preds_log)
        oof_preds[val_idx] = val_preds
        
        fold_smape = smape(y_train.iloc[val_idx].values, val_preds)
        print(f"Fold {fold+1} SMAPE: {fold_smape:.4f}")
        
        test_preds_log = model.predict(X_test)
        test_preds += np.expm1(test_preds_log) / kf.n_splits
        
    overall_smape = smape(y_train.values, oof_preds)
    print(f"\nOverall Out-of-Fold SMAPE: {overall_smape:.4f}")
    
    print("Generating submission...")
    submission = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': test_preds
    })
    
    out_path = r'C:\Users\khush\Downloads\student_resource\dataset\test_out.csv'
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    print(submission.head())

if __name__ == '__main__':
    main()
