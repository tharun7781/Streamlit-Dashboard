import pandas as pd
import numpy as np

def load_and_preprocess(path="application_train.csv"):
    """
    Loads the raw data and performs a series of preprocessing steps
    including feature engineering, missing value imputation, and
    outlier handling.

    Args:
        path (str): The file path to the raw application data.

    Returns:
        pd.DataFrame: A cleaned and preprocessed DataFrame ready for analysis.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        raise

    # --- Feature Engineering & Cleaning ---
    
    # 1. Age and Employment Conversions
    # Convert DAYS_BIRTH to AGE_YEARS
    df['AGE_YEARS'] = (-df['DAYS_BIRTH']) / 365.25
    # DAYS_EMPLOYED: special code 365243 -> treat as NaN
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    # Convert DAYS_EMPLOYED to EMPLOYMENT_YEARS
    df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED']) / 365.25

    # 2. Ratio Features (Affordability and Financial Ratios)
    # DTI (Debt-to-Income) Ratio: AMT_ANNUITY / AMT_INCOME_TOTAL
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    df['DTI'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + epsilon)
    # LTI (Loan-to-Income) Ratio: AMT_CREDIT / AMT_INCOME_TOTAL
    df['LTI'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + epsilon)
    # ANNUITY_TO_CREDIT Ratio: AMT_ANNUITY / AMT_CREDIT
    df['ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + epsilon)
    
    # 3. Create Income Bracket feature for better segmentation
    df['INCOME_BRACKET'] = pd.qcut(
        df['AMT_INCOME_TOTAL'],
        q=[0, 0.25, 0.5, 0.75, 1],
        labels=['Low', 'Mid', 'High', 'Very High'],
        duplicates='drop'
    )
    
    # --- Missing Value Handling ---
    
    # 1. Drop columns with high missing percentage
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    cols_to_drop = missing_pct[missing_pct > 0.60].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # 2. Imputation for remaining missing values
    # Simple imputation: median for numerics, mode for categoricals
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].mode()[0])

    # --- Rare Category Handling (newly added) ---
    for col in cat_cols:
        # Check if the column exists after dropping high missing-value columns
        if col in df.columns:
            # Get value counts and find categories with a share < 1%
            value_counts = df[col].value_counts(normalize=True)
            rare_cats = value_counts[value_counts < 0.01].index.tolist()
            # Replace rare categories with 'Other'
            df[col] = df[col].apply(lambda x: 'Other' if x in rare_cats else x)
            
    # --- Outlier Handling (Simple capping) ---
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
    return df
