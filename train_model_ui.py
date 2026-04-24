import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib


def extract_primary_material(material_str):
    if pd.isna(material_str):
        return 'Other'
    first_part = str(material_str).split(',')[0].strip()
    if len(first_part) == 0:
        return 'Other'
    return first_part


def count_sizes(sizes_str):
    if pd.isna(sizes_str) or len(str(sizes_str).strip()) == 0:
        return 1
    items = [s.strip() for s in str(sizes_str).split(',') if s.strip()]
    return len(items) if len(items) > 0 else 1


def build_dataset(path='cleaned_ethnic_dress_dataset_for_deployment.csv'):
    df = pd.read_csv(path)
    df = df.copy()

    df['Length_Code'] = df['Length'].map({
        'Midi': 1,
        'Maxi': 2,
        'Knee Length': 3,
        'Above Knee': 4,
        'Three-Quarter Sleeves': 5
    }).fillna(0)

    df['Neck_Code'] = df['Neck'].map({
        'V-Neck': 1,
        'Round Neck': 2,
        'Shirt Collar': 3,
        'Mandarin Collar': 4,
        'Square Neck': 5,
        'Sweetheart Neck': 6,
        'Boat Neck': 7,
        'Tie-Up Neck': 8,
        'Keyhole Neck': 9,
        'Halter Neck': 10,
        'Shoulder Straps': 11,
        'Strapless': 12
    }).fillna(0)

    df['Shape_Code'] = df['Shape'].map({
        'A-Line': 1,
        'Maxi': 2
    }).fillna(0)

    df['Sizes_Count'] = df['Sizes_Available'].apply(count_sizes)
    df['Primary_Material'] = df['Material_&_Care'].apply(extract_primary_material)

    top_materials = df['Primary_Material'].value_counts().head(10).index.tolist()
    df['Material_Encoded'] = df['Primary_Material'].apply(lambda x: x if x in top_materials else 'Other')

    top_categories = df['Category'].value_counts().head(10).index.tolist()
    df['Category_Encoded'] = df['Category'].apply(lambda x: x if x in top_categories else 'Other')

    df['Sleeve_Length'] = df['Sleeve_Length'].fillna('Unknown')

    return df


def train_models(df):
    features = [
        'Length_Code',
        'Neck_Code',
        'Shape_Code',
        'Sizes_Count'
    ]

    categorical_features = ['Sleeve_Length', 'Material_Encoded', 'Category_Encoded']
    target = 'Price'

    X = df[features + categorical_features].fillna(0)
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train_processed, y_train)
        preds = model.predict(X_test_processed)
        results[name] = {
            'model': model,
            'MSE': mean_squared_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'MAE': mean_absolute_error(y_test, preds),
            'R2': r2_score(y_test, preds)
        }

    return preprocessor, results


if __name__ == '__main__':
    df = build_dataset()
    preprocessor, results = train_models(df)

    joblib.dump(preprocessor, 'ui_preprocessor.pkl')
    for model_name, payload in results.items():
        filename = model_name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(payload['model'], filename)

    summary = {
        'models': {
            name: {
                'rmse': stats['RMSE'],
                'mae': stats['MAE'],
                'r2': stats['R2']
            }
            for name, stats in results.items()
        }
    }
    with open('ui_model_metrics.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print('Saved models:')
    for name in results:
        print('-', name)
    print('Preprocessor saved to ui_preprocessor.pkl')
    print('Metrics saved to ui_model_metrics.json')
