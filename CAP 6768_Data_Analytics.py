# ==============================================================================
# CAP6768 Final Project: Retail Store Analytics
# Classification & Forecasting
# Team Members: Taiwo, Kayla, Fehmida, Vadym, Grace
# ==============================================================================

# ==========================
# Load Required Libraries
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# ==============================================================================
# 1. DATA LOADING AND EXPLORATION
# ==============================================================================
# Load dataset
retail = pd.read_csv("data_analytics_retail.csv")

# Overview
print("=== DATA OVERVIEW ===")
print(retail.info())
print("\n=== FIRST 6 ROWS ===")
print(retail.head())
print("\n=== MISSING VALUES ===")
print(retail.isna().sum())

# Data cleaning and feature engineering
retail['date'] = pd.to_datetime(retail['date'])
retail['day_of_week'] = pd.Categorical(retail['day_of_week'],
                                      categories=["Monday", "Tuesday", "Wednesday",
                                                  "Thursday", "Friday", "Saturday", "Sunday"])
retail['month'] = pd.Categorical(retail['month'], categories=["Jun", "Jul", "Aug"])
retail['day_type'] = np.where(retail['weekend'], 'Weekend', 'Weekday')

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALS
# ==============================================================================
# Create output directory for plots
plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

# 2.1 Time Series Plot of Daily Revenue
plt.figure(figsize=(12,6))
sns.lineplot(data=retail, x='date', y='daily_revenue', color='steelblue')
sns.scatterplot(data=retail, x='date', y='daily_revenue', hue='day_type',
                palette={'Weekday':'darkorange','Weekend':'purple'})
sns.regplot(data=retail, x=retail['date'].map(datetime.toordinal),
            y='daily_revenue', scatter=False, color='red', lowess=True)
plt.title("Daily Revenue Over Time with Trend Line")
plt.xlabel("Date")
plt.ylabel("Daily Revenue ($)")
plt.legend(title="Day Type")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "1_time_series_revenue.png"))
plt.show()
plt.close()

# 2.2 Revenue Distribution by Day Type and Day of Week
fig, axes = plt.subplots(1,2, figsize=(14,6))

# Boxplot: Weekend vs Weekday
sns.boxplot(x='day_type', y='daily_revenue', hue='day_type', data=retail,
            palette={'Weekday':'lightblue','Weekend':'lightcoral'})
sns.pointplot(x='day_type', y='daily_revenue', data=retail,
              color='red', markers='D', estimator=np.mean, ax=axes[0])
axes[0].set_title("Revenue Distribution: Weekend vs Weekday")

# Boxplot: Day of Week
sns.boxplot(x='day_of_week', y='daily_revenue', data=retail, ax=axes[1])
axes[1].set_title("Revenue by Day of Week")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "2_revenue_by_day_type.png"))
plt.show()
plt.close()

# 2.3 Promotion Impact on Revenue & Temperature vs Revenue
fig, axes = plt.subplots(1,2, figsize=(14,6))

# Promotion Impact
retail['promotion_label'] = retail['promotion'].map({False:'No Promotion', True:'Promotion'})
sns.boxplot(x='promotion_label', y='daily_revenue', data=retail,
            palette={'No Promotion':'lightgreen','Promotion':'orange'}, ax=axes[0])
sns.pointplot(x='promotion_label', y='daily_revenue', data=retail,
              color='red', markers='D', estimator=np.mean, ax=axes[0])
axes[0].set_title("Impact of Promotions on Daily Revenue")

# Temperature vs Revenue
retail_temp = retail.dropna(subset=['temperature'])
sns.scatterplot(x='temperature', y='daily_revenue', hue='day_type', data=retail_temp,
                palette={'Weekday':'blue','Weekend':'red'}, ax=axes[1])
sns.regplot(x='temperature', y='daily_revenue', data=retail_temp, scatter=False,
            color='red', ax=axes[1])
axes[1].set_title("Temperature vs Daily Revenue")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "3_promotion_temperature_analysis.png"))
plt.show()
plt.close()

# 2.4 Customer Behavior Analysis
fig, axes = plt.subplots(1,2, figsize=(14,6))

# Daily Customers vs Revenue
sns.scatterplot(x='daily_customers', y='daily_revenue', hue='day_type', data=retail,
                palette={'Weekday':'navy','Weekend':'darkred'}, ax=axes[0])
sns.regplot(x='daily_customers', y='daily_revenue', data=retail, scatter=False, color='red', ax=axes[0])
axes[0].set_title("Daily Customers vs Revenue")

# Avg Transaction vs Revenue
sns.scatterplot(x='avg_transaction', y='daily_revenue', hue='day_type', data=retail,
                palette={'Weekday':'navy','Weekend':'darkred'}, ax=axes[1])
sns.regplot(x='avg_transaction', y='daily_revenue', data=retail, scatter=False, color='red', ax=axes[1])
axes[1].set_title("Average Transaction vs Revenue")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "4_customer_behavior.png"))
plt.show()
plt.close()



# ==============================================================================
# 3. DATA PREPARATION FOR MODELING
# ==============================================================================
# Handle missing temperature values
retail['temperature'] = retail['temperature'].fillna(retail['temperature'].mean())

# Binary classification target
median_revenue = retail['daily_revenue'].median()
retail['high_revenue'] = np.where(retail['daily_revenue'] > median_revenue, 1, 0)
print("=== CLASSIFICATION TARGET SUMMARY ===")
print(retail['high_revenue'].value_counts())
print("Median Revenue:", median_revenue)

# Feature engineering for classification
retail_class = retail.copy()
retail_class['day_of_week_num'] = retail_class['day_of_week'].cat.codes + 1
retail_class['month_num'] = retail_class['month'].cat.codes + 1
retail_class['week_num'] = retail_class['date'].dt.isocalendar().week

features = ['daily_customers', 'avg_transaction', 'temperature', 'promotion',
            'weekend', 'day_of_week_num', 'month_num', 'week_num']
X = retail_class[features]
y = retail_class['high_revenue']

# Time-based train/test split
train_idx = slice(0, 75)   # first 75 rows
test_idx = slice(75, 90)   # rows 76 to 90
X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Train models
rf_model = RandomForestClassifier(n_estimators=500)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

print("=== TRAIN/TEST SPLIT ===")
print("Training days:", len(X_train))
print("Testing days:", len(X_test))

# ==============================================================================
# 2.5 Random Forest Feature Importance (example)
rf_imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=rf_imp_df, palette='Blues_d')
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_feature_importance.png"))
plt.show()
plt.close()

print("All plots saved in:", plot_dir)
# ==============================================================================
# 4. CLASSIFICATION MODELING
# ==============================================================================
# Logistic Regression
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)
logit_pred_class = logit_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=500)
rf_model.fit(X_train, y_train)
rf_pred_class = rf_model.predict(X_test)
rf_pred_prob = rf_model.predict_proba(X_test)[:,1]

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred_class = xgb_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred)
    }

metrics = pd.DataFrame([
    evaluate_model(y_test, logit_pred_class),
    evaluate_model(y_test, rf_pred_class),
    evaluate_model(y_test, xgb_pred_class)
], index=['Logistic Regression','Random Forest','XGBoost'])

print("=== CLASSIFICATION MODEL PERFORMANCE ===")
print(metrics)

# Random Forest Feature Importance
rf_imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=rf_imp_df, palette='Blues_d')
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("plots/5_feature_importance.png")
plt.show()
plt.close()

# ======================================================================
# 5. FEATURE IMPORTANCE: Random Forest + XGBoost
# ======================================================================
# XGBoost feature importance
xgb_imp = xgb_model.get_booster().get_score(importance_type='weight')
xgb_imp_df = pd.DataFrame({
    'Feature': list(xgb_imp.keys()),
    'Importance': list(xgb_imp.values())
}).sort_values(by='Importance', ascending=False)

# Combined plot
fig, axes = plt.subplots(1,2, figsize=(14,6))

# Random Forest Importance
sns.barplot(x='Importance', y='Feature', data=rf_imp_df, palette='Blues_d', ax=axes[0])
axes[0].set_title("Random Forest Feature Importance")

# XGBoost Importance
sns.barplot(x='Importance', y='Feature', data=xgb_imp_df, palette='Oranges_d', ax=axes[1])
axes[1].set_title("XGBoost Feature Importance")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "6_feature_importance_comparison.png"))
plt.show()
plt.close()

# ======================================================================
# 6. MODEL PERFORMANCE PLOT
# ======================================================================
# Create long-format DataFrame for plotting
metrics_long = metrics.reset_index().melt(id_vars='index',
                                         value_vars=['Accuracy','Precision','Recall','F1_Score'],
                                         var_name='Metric', value_name='Score')
metrics_long.rename(columns={'index':'Model'}, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_long, palette='Set2')
plt.title("Classification Model Performance")
plt.ylim(0,1)
plt.legend(title='Metric', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "7_model_performance.png"))
plt.show()
plt.close()

import os
print("Current working directory:", os.getcwd())