"""
Stock Price Analysis & Prediction - Data Science Project
Domain: Finance - Using CSV file from Kaggle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ========== LOAD CSV FILE ==========
# CHANGE THIS FILENAME to match your downloaded CSV
CSV_FILENAME = 'data/aapl_historical_prices.csv'  # <-- UPDATE THIS

print("📊 Loading stock data from CSV file...")
stock_data = pd.read_csv(CSV_FILENAME)
print(f"✅ Loaded {len(stock_data)} rows from {CSV_FILENAME}")
print(f"📋 Columns: {list(stock_data.columns)}")

# Check if 'Date' column exists (if not, rename appropriate column)
if 'Date' not in stock_data.columns:
    # Try common column names
    if 'date' in stock_data.columns:
        stock_data.rename(columns={'date': 'Date'}, inplace=True)
    elif 'datetime' in stock_data.columns:
        stock_data.rename(columns={'datetime': 'Date'}, inplace=True)

# Ensure Date is datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Sort by date
stock_data = stock_data.sort_values('Date')

# 1. DATA UNDERSTANDING
print("\n" + "="*60)
print("📈 DATA SCIENCE PROJECT: STOCK PRICE ANALYSIS")
print("="*60)

print("\n📋 First 5 rows:")
print(stock_data.head())

print("\n📊 Dataset Info:")
print(stock_data.info())

print("\n📈 Statistical Summary:")
print(stock_data.describe())

print(f"\n🔍 Missing Values:\n{stock_data.isnull().sum()}")

# 2. FEATURE ENGINEERING
print("\n" + "="*60)
print("🔧 FEATURE ENGINEERING")
print("="*60)

stock_data['Day'] = pd.to_datetime(stock_data['Date']).dt.day
stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.month
stock_data['Year'] = pd.to_datetime(stock_data['Date']).dt.year
stock_data['DayOfWeek'] = pd.to_datetime(stock_data['Date']).dt.dayofweek

# Price-based features (use column names from your CSV)
# Common column names: 'Open', 'High', 'Low', 'Close', 'Volume'
# Update if your CSV uses different names
open_col = 'Open' if 'Open' in stock_data.columns else 'open'
high_col = 'High' if 'High' in stock_data.columns else 'high'
low_col = 'Low' if 'Low' in stock_data.columns else 'low'
close_col = 'Close' if 'Close' in stock_data.columns else 'close'
volume_col = 'Volume' if 'Volume' in stock_data.columns else 'volume'

stock_data['Price_Range'] = stock_data[high_col] - stock_data[low_col]
stock_data['Price_Change'] = stock_data[close_col] - stock_data[open_col]
stock_data['Return_Pct'] = (stock_data[close_col] - stock_data[open_col]) / stock_data[open_col] * 100

# Moving averages
stock_data['MA_7'] = stock_data[close_col].rolling(window=7).mean()
stock_data['MA_30'] = stock_data[close_col].rolling(window=30).mean()

# Volatility
stock_data['Volatility'] = stock_data['Return_Pct'].rolling(window=7).std()

print("✅ Feature engineering completed!")

# 3. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*60)
print("📊 EXPLORATORY DATA ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Closing price over time
axes[0, 0].plot(stock_data['Date'], stock_data[close_col], color='blue', linewidth=1)
axes[0, 0].set_title('Stock Closing Price Over Time', fontsize=12)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Close Price ($)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Volume over time
if volume_col in stock_data.columns:
    axes[0, 1].bar(stock_data['Date'], stock_data[volume_col], color='green', alpha=0.7, width=0.8)
    axes[0, 1].set_title('Trading Volume Over Time', fontsize=12)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of daily returns
axes[1, 0].hist(stock_data['Return_Pct'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Daily Returns (%)', fontsize=12)
axes[1, 0].set_xlabel('Return %')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)

# Plot 4: Moving averages
axes[1, 1].plot(stock_data['Date'], stock_data[close_col], label='Close Price', alpha=0.5, linewidth=1)
axes[1, 1].plot(stock_data['Date'], stock_data['MA_7'], label='7-Day MA', linewidth=1.5)
axes[1, 1].plot(stock_data['Date'], stock_data['MA_30'], label='30-Day MA', linewidth=1.5)
axes[1, 1].set_title('Moving Averages', fontsize=12)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plots saved as 'stock_analysis_plots.png'")

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = [open_col, high_col, low_col, close_col, 'Price_Range', 'Price_Change', 'Return_Pct']
if volume_col in stock_data.columns:
    numeric_cols.append(volume_col)
numeric_cols = [col for col in numeric_cols if col in stock_data.columns]
corr_matrix = stock_data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Stock Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")

# 4. PREDICTION MODEL
print("\n" + "="*60)
print("🤖 PREDICTION MODEL: Linear Regression")
print("="*60)

# Prepare features
features = ['Day', 'Month', 'DayOfWeek', 'Price_Range']
if volume_col in stock_data.columns:
    features.append(volume_col)

target = close_col

# Drop NaN values
model_data = stock_data[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"   Mean Absolute Error (MAE):  ${mae:.2f}")
print(f"   Root Mean Squared Error:    ${rmse:.2f}")
print(f"   R² Score:                   {r2:.4f}")
print(f"   Accuracy (approx):          {(1 - mae/y_test.mean()) * 100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print(f"\n🔑 Feature Importance:")
for _, row in feature_importance.iterrows():
    print(f"   {row['Feature']}: {row['Coefficient']:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted Stock Prices')
plt.grid(True, alpha=0.3)
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")
print("📁 Files generated: stock_analysis_plots.png, correlation_heatmap.png, actual_vs_predicted.png")