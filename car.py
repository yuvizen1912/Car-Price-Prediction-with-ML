import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_theme(style="whitegrid")


# 1. LOAD DATASET

try:
    df = pd.read_csv("car_data.csv")
    print("✅ Dataset successfully loaded!")
except FileNotFoundError:
    print("❌ 'car_data.csv' not found. Please ensure the file name matches your CSV.")
    exit()

print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
print(df.info())

# 2. ADVANCED FEATURE ENGINEERING

current_year = 2026 
df['Age'] = current_year - df['Year']

df['km_per_year'] = df['Driven_kms'] / (df['Age'] + 1)

X = df.drop(columns=['Selling_Price', 'Car_Name', 'Year'])
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. PIPELINE & TRANSFORMATION PIPELINE

numeric_features = ['Present_Price', 'Driven_kms', 'Owner', 'Age', 'km_per_year']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. MULTI-MODEL BENCHMARKING

models = {
    'Ridge Regression': Ridge(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

print("\n--- Training & Evaluating Base Performance Metrics ---")
for name, model in models.items():
    # Construct distinct execution pipelines to prevent data leakage
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    
    # Evaluate predictions
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name:28} -> R² Score: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

# ==========================================
# 5. HYPERPARAMETER OPTIMIZATION (Grid Search)

print("\n--- Optimizing Gradient Boosting via 5-Fold Cross Validation ---")

optimization_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('gb', GradientBoostingRegressor(random_state=42))
])


param_grid = {
    'gb__n_estimators': [100, 200, 300],
    'gb__learning_rate': [0.03, 0.05, 0.1],
    'gb__max_depth': [3, 4, 5]
}

# Execute Search
grid_search = GridSearchCV(optimization_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Optimized Parameters: {grid_search.best_params_}")


final_preds = best_model.predict(X_test)
final_r2 = r2_score(y_test, final_preds)
print(f"Final Tuned Gradient Boosting R² Score: {final_r2:.4f}")

# ==========================================
# 6. PRODUCTION DIAGNOSTIC VISUALIZATIONS

plt.figure(figsize=(15, 5))


plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=final_preds, alpha=0.7, color='dodgerblue', edgecolor='w', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Perfect Prediction')
plt.title('Actual Selling Price vs. Predicted Selling Price', fontsize=12, pad=10)
plt.xlabel('Actual Selling Price (Lakhs)', fontsize=10)
plt.ylabel('Predicted Selling Price (Lakhs)', fontsize=10)
plt.legend()


plt.subplot(1, 2, 2)
residuals = y_test - final_preds
# Changed the color to a clean, standard 'seagreen'
sns.histplot(residuals, kde=True, color='seagreen', bins=20)
plt.axvline(0, color='crimson', linestyle='--', lw=2)
plt.title('Distribution of Model Errors (Residuals)', fontsize=12, pad=10)
plt.xlabel('Prediction Error Magnitude', fontsize=10)

plt.tight_layout()
plt.show()