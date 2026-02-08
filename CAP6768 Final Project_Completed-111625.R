# ==============================================================================
# CAP6768 Final Project: Retail Store Analytics
# Classification & Forecasting 
# Team Members: Taiwo, Kayla, Vadym, Fehmida, Grace
# ==============================================================================

# ==========================
# Load Required Libraries
# ==========================
# Visualizations :)
library(ggplot2)
library(gridExtra)

# Data manipulation and transformation
library(dplyr)
library(lubridate)
library(tidyverse)

# Modeling and performance testing
library(caret)
library(ranger)
library(pROC)
library(xgboost)

# Time series forecasting
library(forecast)
library(prophet)
library(Metrics)

# ==============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION_T.O.
# ==============================================================================
# Load the csv data file for data_analytics
retail <- read.csv("data_analytics_retail.csv")

# Basic overview to show the structure of the dataset
cat("=== DATA OVERVIEW ===\n")
str(retail)
cat("\n=== FIRST 6 ROWS ===\n")# Showing the first 6 rows
head(retail)
cat("\n=== MISSING VALUES ===\n")# Checking for missing values in each column
colSums(is.na(retail))

# Data cleaning and feature engineering
# Convert date and create additional features (new categorical variables)
retail <- retail %>%
  mutate(
    date = as.Date(date),# Convert our date string to date type
    day_of_week = factor(day_of_week, 
                         levels = c("Monday", "Tuesday", "Wednesday", 
                                    "Thursday", "Friday", "Saturday", "Sunday")),
    month = factor(month, levels = c("Jun", "Jul", "Aug")),
    day_type = ifelse(weekend, "Weekend", "Weekday")
  )

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALS_T.O.
# ==============================================================================

# Create output directory for saving our plots
if(!dir.exists("plots")) dir.create("plots")

# 2.1 Time Series Plot of Daily Revenue aka daily revenue over time
p1 <- ggplot(retail, aes(x = date, y = daily_revenue)) +
  geom_line(color = "steelblue", size = 0.8) +
  geom_point(aes(color = day_type), size = 2, alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "darkorange", "Weekend" = "purple")) +
  labs(title = "Daily Revenue Over Time with Trend Line",
       subtitle = "Red line shows overall trend, Colors indicate weekend/weekday",
       x = "Date", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal() +
  theme(legend.position = "top")# Line plot showing revenue trend and daily variation


ggsave("plots/1_time_series_revenue.png", p1, width = 12, height = 6)

# 2.2 Revenue Distribution by Day Type and Day of the Week
p2 <- ggplot(retail, aes(x = day_type, y = daily_revenue, fill = day_type)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "red") +
  scale_fill_manual(values = c("Weekday" = "lightblue", "Weekend" = "lightcoral")) +
  labs(title = "Revenue Distribution: Weekend vs Weekday",
       subtitle = "Red diamond shows mean revenue",
       x = "Day Type", y = "Daily Revenue ($)") +
  theme_minimal() # Boxplot comparing weekday vs weekend revenue


p3 <- ggplot(retail, aes(x = day_of_week, y = daily_revenue, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Revenue by Day of Week",
       x = "Day of Week", y = "Daily Revenue ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")# Boxplot showing revenue patterns by day of the week


# Combined both plots into one figure
grid_plot <- grid.arrange(p2, p3, ncol = 2)
ggsave("plots/2_revenue_by_day_type.png", grid_plot, width = 14, height = 6)

# 2.3 Promotion Impact on Revenue Analysis
p4 <- ggplot(retail, aes(x = factor(promotion, labels = c("No Promotion", "Promotion")), 
                         y = daily_revenue, fill = factor(promotion))) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "red") +
  scale_fill_manual(values = c("FALSE" = "lightgreen", "TRUE" = "orange")) +
  labs(title = "Impact of Promotions on Daily Revenue",
       subtitle = "Red diamond shows mean revenue",
       x = "Promotion Status", y = "Daily Revenue ($)") +
  theme_minimal() +
  theme(legend.position = "none")

# 2.4 Temperature vs Revenue (with the missing values accounted for)
retail_temp <- retail %>% filter(!is.na(temperature))

p5 <- ggplot(retail_temp, aes(x = temperature, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  scale_color_manual(values = c("Weekday" = "blue", "Weekend" = "red")) +
  labs(title = "Temperature vs Daily Revenue",
       subtitle = "Colored by day type, Red line shows linear relationship",
       x = "Temperature (°F)", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal()

ggsave("plots/3_promotion_temperature_analysis.png", 
       grid.arrange(p4, p5, ncol = 2), width = 14, height = 6)

# 2.5 Customer Behavior Analysis
p6 <- ggplot(retail, aes(x = daily_customers, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "navy", "Weekend" = "darkred")) +
  labs(title = "Daily Customers vs Revenue",
       subtitle = "Strong positive correlation expected",
       x = "Number of Customers", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal() # Scatter plot: number of customers vs revenue


p7 <- ggplot(retail, aes(x = avg_transaction, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "navy", "Weekend" = "darkred")) +
  labs(title = "Average Transaction vs Revenue",
       x = "Average Transaction Amount ($)", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal() # Scatter plot: average transaction vs revenue


ggsave("plots/4_customer_behavior.png", 
       grid.arrange(p6, p7, ncol = 2), width = 14, height = 6)

# ==============================================================================
# 3. DATA PREPARATION FOR MODELING_T.O.
# ==============================================================================

# 3.1 Handles the  missing values (temperature)
retail_clean <- retail %>%
  mutate(
    temperature = ifelse(is.na(temperature), mean(temperature, na.rm = TRUE), temperature)
  )

# 3.2 Creates the binary classification target (High Revenue = 1, Low = 0)
median_revenue <- median(retail_clean$daily_revenue)
retail_clean$high_revenue <- ifelse(retail_clean$daily_revenue > median_revenue, 1, 0)

cat("=== CLASSIFICATION TARGET SUMMARY ===\n")
table(retail_clean$high_revenue)
cat("Median Revenue:", median_revenue, "\n")

# 3.3 Creates features for classification
retail_class <- retail_clean %>%
  mutate(
    day_of_week_num = as.numeric(day_of_week),
    month_num = as.numeric(month),
    week_num = week(date)
  ) %>%
  select(high_revenue, daily_customers, avg_transaction, temperature, promotion, 
         weekend, day_of_week_num, month_num, week_num, day_type)

# 3.4 Preparing time series data for forecasting
ts_data <- retail_clean %>%
  select(date, daily_revenue, daily_customers, weekend, promotion) %>%
  arrange(date) 

# ==============================================================================
# 4. CLASSIFICATION MODELING_G.G.
# ==============================================================================

# 4.1 Split test and train data
set.seed(123)
train_size <- floor(0.75 * nrow(retail_class))
train_data <- retail_class[1:train_size, ]
test_data <- retail_class[(train_size + 1):nrow(retail_class), ]

# 4.2 Verify values in each split set
cat("Training set size:", nrow(train_data), "days\n")
cat("Testing set size:", nrow(test_data), "days\n")
cat("High revenue in training:", sum(train_data$high_revenue), "\n")
cat("High revenue in testing:", sum(test_data$high_revenue), "\n")

# 4.3 Define metric calculation function
calculate_metrics <- function(conf_matrix) {
  TP <- conf_matrix["1","1"]
  TN <- conf_matrix["0","0"]
  FP <- conf_matrix["1","0"]
  FN <- conf_matrix["0","1"]
  
  accuracy  <- (TP + TN) / sum(conf_matrix)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1        <- 2 * (precision * recall) / (precision + recall)
  
  cat("Accuracy :", round(accuracy*100,1), "%\n")
  cat("Precision:", round(precision*100,1), "%\n")
  cat("Recall   :", round(recall*100,1), "%\n")
  cat("F1 Score :", round(f1*100,1), "%\n")
  
  tibble(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1)
}

# 4.4-A Logistic regression model setup and training (core features only)
log_model <- glm(high_revenue ~ daily_customers + promotion + weekend,
             data = train_data,
             family = "binomial")
summary(log_model)

# 4.4-B Predict probabilities on test data and evaluate performance
log_probs <- predict(log_model, newdata = test_data, type = "response")
log_preds <- ifelse(log_probs > 0.5, 1, 0)

log_conf_matrix <- table(Predicted = log_preds, Actual = test_data$high_revenue)
log_metrics <- calculate_metrics(log_conf_matrix)

# 4.4-C Visualize logistic model prediction distribution
log_results <- tibble(
  probability = log_probs,
  actual = factor(test_data$high_revenue, labels = c("Low", "High"))
)

ggplot(log_results, aes(x = probability, fill = actual)) +
  geom_histogram(alpha = 0.7, bins = 15, position = "identity") + 
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "black") +
  scale_fill_manual(values = c("Low" = "purple", "High" = "orange")) +
  labs(
    title = "Predicted Probability Distribution (Logistic Regression)",
    x = "Predicted Probability of High Revenue",
    y = "Count",
    fill = "Actual Class"
  ) +
  theme_minimal() +
  theme(text = element_text(size = 12))
ggsave("plots/5_logistic_probability_distribution.png", width = 10, height = 6)

# 4.5-A Random forest model setup and training
rf_fit <- ranger(
  formula = factor(high_revenue) ~ .,  
  data = train_data,                  
  importance = "impurity",             
  probability = TRUE,                  
  seed = 123                           
)

print(rf_fit)

# 4.5-B Predict probabilities on test data and evaluate performance
rf_pred <- predict(rf_fit, data = test_data)
rf_pred_class <- ifelse(rf_pred$predictions[,2] > 0.5, 1, 0)

rf_conf_matrix <- table(Predicted = rf_pred_class, Actual = test_data$high_revenue)
rf_metrics <- calculate_metrics(rf_conf_matrix)

# 4.4-C Visualize random forest model variable importance
rf_imp <- tibble(
  feature = names(rf_fit$variable.importance),
  importance = rf_fit$variable.importance
)

rf_imp %>%
  arrange(desc(importance)) %>%
  ggplot(aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(
    title = "Feature Importance (Random Forest)",
    x = NULL,
    y = "Importance"
  ) +
  theme_minimal() +
  theme(text = element_text(size = 12))
ggsave("plots/6_rf_feature_importance.png", width = 10, height = 6)

# 4.5-A XGBoost model setup and training
train_matrix <- model.matrix(high_revenue ~ . -1, data = train_data)
train_label  <- train_data$high_revenue

test_matrix  <- model.matrix(high_revenue ~ . -1, data = test_data)
test_label   <- test_data$high_revenue

xgb_fit <- xgboost(
  data = train_matrix,
  label = train_label,
  objective = "binary:logistic",
  nrounds = 100,
  verbose = 0
)

print(xgb_fit)

# 4.5-B Predict probabilities on test data and evaluate performance
xgb_probs <- predict(xgb_fit, test_matrix)
xgb_preds <- ifelse(xgb_probs > 0.5, 1, 0)

xgb_conf_matrix <- table(Predicted = xgb_preds, Actual = test_label)
xgb_metrics <- calculate_metrics(xgb_conf_matrix)

# 4.5-C Visualize XGBoost model ROC curve
roc_obj <- pROC::roc(test_label, xgb_probs)

ggroc(roc_obj, color = "purple", size = 1.2) +
  geom_abline(linetype = "dashed", color = "orange") +
  labs(
    title = "ROC Curve (XGBoost)",
    subtitle = paste("AUC =", round(pROC::auc(roc_obj), 3)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(text = element_text(size = 12))
ggsave("plots/7_xgboost_roc_curve.png", width = 10, height = 6)

# 4.6-A Model comparison with ranking
comparison_table <- tibble(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy = c(log_metrics$Accuracy, 
               rf_metrics$Accuracy, 
               xgb_metrics$Accuracy),
  Precision = c(log_metrics$Precision, 
                rf_metrics$Precision, 
                xgb_metrics$Precision),
  Recall = c(log_metrics$Recall, 
             rf_metrics$Recall, 
             xgb_metrics$Recall),
  F1 = c(log_metrics$F1, 
         rf_metrics$F1, 
         xgb_metrics$F1)
) %>%
  mutate(across(where(is.numeric), ~round(.x * 100, 1))) %>%
  arrange(desc(Accuracy))

cat("\n=== CLASSIFICATION MODEL COMPARISON ===\n")
print(comparison_table)

# 4.6-B Visualize model comparison
comparison_long <- comparison_table %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Model, y = Value, fill = Model)) +
  geom_col() +
  geom_text(aes(label = paste0(Value, "%")), vjust = -0.5, size = 3) +
  facet_wrap(~Metric, scales = "free_y") +
  scale_fill_manual(values = c("Logistic Regression" = "steelblue", 
                               "Random Forest" = "purple", 
                               "XGBoost" = "orange")) +
  labs(title = "Classification Model Performance Comparison",
       x = NULL, y = "Score (%)") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("plots/8_classification_model_comparison.png", width = 12, height = 6)

# 4.6-C Identify best model
best_model <- comparison_table$Model[1]
cat("\nBest Overall Model (Highest Accuracy):", best_model, "\n")

# ==============================================================================
# 5. TIME SERIES FORECASTING_G.G.
# ==============================================================================

# 5.1 Split test and train data
set.seed(123)
train_ts <- ts_data[1:(nrow(ts_data) - 7), ]
test_ts <- ts_data[(nrow(ts_data) - 6):nrow(ts_data), ]

# 5.2 Define forecast metric calculation function
calculate_forecast_metrics <- function(actual, predicted) {
  rmse_val <- sqrt(mean((actual - predicted)^2))
  mae_val <- mean(abs(actual - predicted))
  mape_val <- mean(abs((actual - predicted) / actual)) * 100
  
  tibble(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val)
}

# 5.3-A SARIMA model setup and preprocessing
revenue_ts <- ts(train_ts$daily_revenue, frequency = 7)

sarima_model <- auto.arima(
  revenue_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

summary(sarima_model)

# 5.3-B Generate forecast prediction and evaluate performance
sarima_forecast <- forecast(sarima_model, h = 7)

sarima_predictions <- tibble(
  date = test_ts$date,
  actual = test_ts$daily_revenue,
  forecast = as.numeric(sarima_forecast$mean),
  lower_95 = as.numeric(sarima_forecast$lower[,2]),
  upper_95 = as.numeric(sarima_forecast$upper[,2])
)

sarima_metrics <- calculate_forecast_metrics(
  sarima_predictions$actual, 
  sarima_predictions$forecast
)

# 5.3-C Visualize SARIMA model forecast vs actual
ggplot(sarima_predictions, aes(x = date)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), fill = "purple", alpha = 0.2) +
  geom_line(aes(y = forecast, color = "SARIMA Forecast"), size = 1) +
  geom_line(aes(y = actual, color = "Actual Revenue"), size = 1) +
  geom_point(aes(y = actual, color = "Actual Revenue"), size = 3) +
  scale_color_manual(values = c("SARIMA Forecast" = "purple", "Actual Revenue" = "orange")) +
  labs(
    title = "SARIMA: 7-Day Revenue Forecast",
    subtitle = paste("RMSE:", round(sarima_metrics$RMSE, 2), "| MAE:", round(sarima_metrics$MAE, 2)),
    x = "Date",
    y = "Daily Revenue ($)",
    color = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "top", text = element_text(size = 12))
ggsave("plots/9_sarima_forecast.png", width = 12, height = 6)

# 5.4-A Prophet model setup and preprocessing
prophet_train <- train_ts %>%
  select(ds = date, y = daily_revenue, weekend, promotion)

prophet_model <- prophet(
  yearly.seasonality = FALSE,
  weekly.seasonality = TRUE,
  daily.seasonality = FALSE,
  seasonality.mode = "additive"
)

prophet_model <- add_regressor(prophet_model, "weekend")
prophet_model <- add_regressor(prophet_model, "promotion")
prophet_model <- fit.prophet(prophet_model, prophet_train)

# 5.4-B Generate forecast prediction and evaluate performance
future <- make_future_dataframe(prophet_model, periods = nrow(test_ts), freq = "day")

future <- future %>%
  mutate(
    weekend = wday(ds) %in% c(1, 7),
    promotion = FALSE
  )

prophet_forecast <- predict(prophet_model, future)
prophet_forecast$ds <- as.Date(prophet_forecast$ds)

prophet_predictions <- prophet_forecast %>%
  filter(ds %in% test_ts$date) %>%
  select(date = ds, forecast = yhat, lower_95 = yhat_lower, upper_95 = yhat_upper) %>%
  left_join(test_ts %>% select(date, actual = daily_revenue), by = "date")

prophet_metrics <- calculate_forecast_metrics(
  prophet_predictions$actual,
  prophet_predictions$forecast
)

# 5.4-C Visualize Prophet model forecast vs actual
ggplot(prophet_predictions, aes(x = date)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), fill = "steelblue", alpha = 0.2) +
  geom_line(aes(y = forecast, color = "Prophet Forecast"), size = 1) +
  geom_line(aes(y = actual, color = "Actual Revenue"), size = 1) +
  geom_point(aes(y = actual, color = "Actual Revenue"), size = 3) +
  scale_color_manual(values = c("Prophet Forecast" = "steelblue", "Actual Revenue" = "orange")) +
  labs(
    title = "Prophet: 7-Day Revenue Forecast",
    subtitle = paste("RMSE:", round(prophet_metrics$RMSE, 2), "| MAE:", round(prophet_metrics$MAE, 2)),
    x = "Date",
    y = "Daily Revenue ($)",
    color = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "top", text = element_text(size = 12))

prophet_plot_components(prophet_model, prophet_forecast)
ggsave("plots/10_prophet_forecast.png", width = 12, height = 6)

# 5.5-A Model comparison with ranking
forecast_comparison <- tibble(
  Model = c("SARIMA", "Prophet"),
  RMSE = c(sarima_metrics$RMSE, prophet_metrics$RMSE),
  MAE = c(sarima_metrics$MAE, prophet_metrics$MAE),
  MAPE = c(sarima_metrics$MAPE, prophet_metrics$MAPE)
) %>%
  mutate(
    RMSE_rank = rank(RMSE),
    MAE_rank = rank(MAE),
    MAPE_rank = rank(MAPE),
    Total_Rank = RMSE_rank + MAE_rank + MAPE_rank
  ) %>%
  arrange(Total_Rank) %>%
  mutate(
    RMSE = round(RMSE, 2),
    MAE = round(MAE, 2),
    MAPE = round(MAPE, 2)
  )
print(forecast_comparison)

# 5.5-B Visualize model performance 
comparison_long <- forecast_comparison %>%
  pivot_longer(cols = c(RMSE, MAE), names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Model, y = Value, fill = Model)) +
  geom_col() +
  geom_text(aes(label = Value), vjust = -0.5, size = 4) +
  facet_wrap(~Metric, scales = "free_y") +
  scale_fill_manual(values = c("SARIMA" = "purple", "Prophet" = "steelblue")) +
  labs(
    title = "Forecast Model Performance Comparison",
    subtitle = "Lower values indicate better performance",
    x = NULL,
    y = "Error Value"
  ) +
  theme_minimal() +
  theme(legend.position = "none", text = element_text(size = 12))
ggsave("plots/11_forecast_model_comparison.png", width = 12, height = 6)

# 5.5-C Visualize combined forecasts
combined_forecasts <- tibble(
  date = test_ts$date,
  Actual = test_ts$daily_revenue,
  SARIMA = sarima_predictions$forecast,
  Prophet = prophet_predictions$forecast
) %>%
  pivot_longer(cols = c(Actual, SARIMA, Prophet), names_to = "Type", values_to = "Revenue")

ggplot(combined_forecasts, aes(x = date, y = Revenue, color = Type)) +
  geom_line(size = 1) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Actual" = "orange", "SARIMA" = "purple", "Prophet" = "steelblue")) +
  labs(
    title = "7-Day Revenue Forecast: Model Comparison",
    x = "Date",
    y = "Daily Revenue ($)",
    color = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "top", text = element_text(size = 12))
ggsave("plots/12_combined_forecasts.png", width = 12, height = 6)

# 5.5-D Identify best model
best_forecast_model <- forecast_comparison$Model[1]
cat("\nBest Overall Model (Overall Ranking):", best_forecast_model, "\n")

# 5.6-A Refit SARIMA on full dataset
full_revenue_ts <- ts(ts_data$daily_revenue, frequency = 7)

sarima_final <- auto.arima(
  full_revenue_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

# 5.6-B Generate SARIMA forecast for actual next week
next_week_forecast <- forecast(sarima_final, h = 7)

next_week_predictions <- tibble(
  date = seq(max(ts_data$date) + 1, by = "day", length.out = 7),
  forecast = as.numeric(next_week_forecast$mean),
  lower_95 = as.numeric(next_week_forecast$lower[,2]),
  upper_95 = as.numeric(next_week_forecast$upper[,2])
) %>%
  mutate(
    day_of_week = wday(date, label = TRUE),
    forecast = round(forecast, 2),
    lower_95 = round(lower_95, 2),
    upper_95 = round(upper_95, 2)
  )

print(next_week_predictions)

# ==============================================================================
# 6. BUSINESS INSIGHTS AND RECOMMENDATIONS
# ==============================================================================
# 6.1 Identify key revenue patterns
# Calculate average revenue by day type (Weekday vs Weekend)
revenue_by_day_type <- retail_clean %>%
  group_by(day_type) %>%
  summarise(
    Avg_Revenue = mean(daily_revenue),
    Median_Revenue = median(daily_revenue),
    Total_Revenue = sum(daily_revenue),
    Count_Days = n()
  )

cat("\n=== KEY REVENUE PATTERNS BY DAY TYPE ===\n")
print(revenue_by_day_type)

# Calculate average revenue by day of week
revenue_by_dow <- retail_clean %>%
  group_by(day_of_week) %>%
  summarise(
    Avg_Revenue = mean(daily_revenue),
    Median_Revenue = median(daily_revenue),
    Total_Revenue = sum(daily_revenue),
    Count_Days = n()
  ) %>%
  arrange(desc(Avg_Revenue))

cat("\n=== KEY REVENUE PATTERNS BY DAY OF WEEK ===\n")
print(revenue_by_dow)

# Statistical test: t-test for revenue difference between weekdays and weekends
weekday_revenue <- retail_clean$daily_revenue[retail_clean$day_type == "Weekday"]
weekend_revenue <- retail_clean$daily_revenue[retail_clean$day_type == "Weekend"]
t_test_day_type <- t.test(weekday_revenue, weekend_revenue)

cat("\n=== T-TEST: WEEKDAY VS WEEKEND REVENUE ===\n")
print(t_test_day_type)

# Insight: Weekends typically show higher revenue, suggesting increased marketing or inventory focus on weekends.

# 6.2 Evaluate promotion effectiveness
# Calculate average revenue with and without promotion
revenue_by_promo <- retail_clean %>%
  group_by(promotion) %>%
  summarise(
    Avg_Revenue = mean(daily_revenue),
    Median_Revenue = median(daily_revenue),
    Total_Revenue = sum(daily_revenue),
    Count_Days = n()
  )

cat("\n=== REVENUE BY PROMOTION STATUS ===\n")
print(revenue_by_promo)

# Statistical test: t-test for revenue difference with/without promotion
no_promo_revenue <- retail_clean$daily_revenue[!retail_clean$promotion]
promo_revenue <- retail_clean$daily_revenue[retail_clean$promotion]
t_test_promo <- t.test(no_promo_revenue, promo_revenue)

cat("\n=== T-TEST: PROMOTION IMPACT ON REVENUE ===\n")
print(t_test_promo)

# Calculate lift from promotions
promo_lift <- (mean(promo_revenue) - mean(no_promo_revenue)) / mean(no_promo_revenue) * 100
cat("\nPromotion Revenue Lift: ", round(promo_lift, 2), "%\n")

# Insight: Promotions significantly increase revenue. Recommend scheduling promotions on lower-performing days (e.g., mid-week) to balance revenue.

# 6.3 Recommend staffing strategies based on customer flow
# Calculate average customers by day type
customers_by_day_type <- retail_clean %>%
  group_by(day_type) %>%
  summarise(
    Avg_Customers = mean(daily_customers),
    Median_Customers = median(daily_customers),
    Avg_Transaction = mean(avg_transaction)
  )

cat("\n=== CUSTOMER FLOW BY DAY TYPE ===\n")
print(customers_by_day_type)

# Calculate average customers by day of week
customers_by_dow <- retail_clean %>%
  group_by(day_of_week) %>%
  summarise(
    Avg_Customers = mean(daily_customers),
    Median_Customers = median(daily_customers),
    Avg_Transaction = mean(avg_transaction)
  ) %>%
  arrange(desc(Avg_Customers))

cat("\n=== CUSTOMER FLOW BY DAY OF WEEK ===\n")
print(customers_by_dow)

# Correlation between customers and revenue
cor_customers_revenue <- cor(retail_clean$daily_customers, retail_clean$daily_revenue)
cat("\nCorrelation between Daily Customers and Revenue: ", round(cor_customers_revenue, 3), "\n")

# Insight: Higher customer traffic on weekends correlates strongly with revenue. 
# Recommend increasing staff by 20-30% on weekends and promotions days to handle peak flow and improve service quality.

# Overall Recommendations:
# - Focus marketing efforts on weekends and promotion periods to maximize revenue.
# - Use temperature data for seasonal adjustments; higher temperatures may correlate with increased revenue.
# - Monitor average transaction values to identify upselling opportunities during high-traffic days.

# ==============================================================================
# 7. FINAL RESULTS SUMMARY
# ==============================================================================

# 7.1 Classification Results Summary
cat("\n=== CLASSIFICATION RESULTS SUMMARY ===\n")
print(comparison_table)
cat("\nBest Classification Model:", best_model, "with Accuracy:", round(comparison_table$Accuracy[1], 1), "%\n")

# 7.2 Forecasting Results Summary
cat("\n=== FORECASTING RESULTS SUMMARY ===\n")
print(forecast_comparison)
cat("\nBest Forecasting Model:", best_forecast_model, "with RMSE:", round(forecast_comparison$RMSE[1], 2), "\n")

# Overall Project Summary:
# - Classification models accurately predict high/low revenue days, with Logistic Regression performing best.
# - Forecasting models provide reliable short-term predictions, with SARIMA edging out Prophet.
# - Business recommendations focus on leveraging promotions and optimizing staffing for peak periods.

# ==============================================================================
# APPENDIX A: Use rolling window cross-validation for forecasting accuracy
# ==============================================================================

# A.1 Split data into rolling windows
n_windows <- 5
window_size <- floor(nrow(ts_data) * 0.8)
forecast_horizon <- 7

rolling_metrics <- list()

for (i in 1:n_windows) {
  train_end <- window_size + (i - 1) * forecast_horizon
  test_start <- train_end + 1
  test_end <- test_start + forecast_horizon - 1
  
  if (test_end > nrow(ts_data)) break
  
  train_window <- ts_data[1:train_end, ]
  test_window <- ts_data[test_start:test_end, ]
  
# A.2 Fit Prophet on window (example; can do for SARIMA similarly)
  prophet_train_window <- train_window %>%
    select(ds = date, y = daily_revenue, weekend, promotion)
  
  prophet_model_window <- prophet(
    yearly.seasonality = FALSE,
    weekly.seasonality = TRUE,
    daily.seasonality = FALSE
  )
  prophet_model_window <- add_regressor(prophet_model_window, "weekend")
  prophet_model_window <- add_regressor(prophet_model_window, "promotion")
  prophet_model_window <- fit.prophet(prophet_model_window, prophet_train_window)
  
  future_window <- make_future_dataframe(prophet_model_window, periods = forecast_horizon, freq = "day")
  future_window <- future_window %>%
    mutate(
      weekend = wday(ds) %in% c(1, 7),
      promotion = FALSE
    )
  
  prophet_forecast_window <- predict(prophet_model_window, future_window)
  predictions <- tail(prophet_forecast_window$yhat, forecast_horizon)
  
  window_metrics <- calculate_forecast_metrics(test_window$daily_revenue, predictions)
  rolling_metrics[[i]] <- window_metrics
}

# A.3 Average rolling metrics
avg_rolling_rmse <- mean(sapply(rolling_metrics, function(m) m$RMSE))
avg_rolling_mae <- mean(sapply(rolling_metrics, function(m) m$MAE))
avg_rolling_mape <- mean(sapply(rolling_metrics, function(m) m$MAPE))

cat("\n=== ROLLING WINDOW CV AVERAGE METRICS (Prophet) ===\n")
cat("Avg RMSE:", round(avg_rolling_rmse, 2), "| Avg MAE:", round(avg_rolling_mae, 2), "| Avg MAPE:", round(avg_rolling_mape, 2), "\n")

# Insight: Rolling window CV better simulates real-world forecasting, showing consistent performance across time periods.

# ==============================================================================
# APPENDIX B: Predictive classification without same-day data
# ==============================================================================

# B.1 Create lagged features (previous day data)
retail_predictive <- retail_clean %>%
  arrange(date) %>%
  mutate(
    prev_day_customers = lag(daily_customers, 1),
    prev_day_revenue = lag(daily_revenue, 1),
    prev_day_avg_transaction = lag(avg_transaction, 1),
    avg_customers_7day = slider::slide_dbl(daily_customers, mean, 
                                           .before = 7, .after = -1, 
                                           .complete = TRUE),
    day_of_week_num = as.numeric(day_of_week)
  ) %>%
  drop_na()

median_revenue <- median(retail_predictive$daily_revenue)
retail_predictive$high_revenue <- ifelse(retail_predictive$daily_revenue > median_revenue, 1, 0)

# B.2 Select predictive features only
retail_predictive_features <- retail_predictive %>%
  select(high_revenue, weekend, promotion, day_of_week_num, temperature,
    prev_day_customers,prev_day_revenue,avg_customers_7day)

# B.3 Split data using same approach as main models
set.seed(123)
train_size <- floor(0.75 * nrow(retail_predictive_features))
train_pred <- retail_predictive_features[1:train_size, ]
test_pred <- retail_predictive_features[(train_size + 1):nrow(retail_predictive_features), ]

# B.4 Predictive classification model setup and training
pred_log_model <- glm(
  high_revenue ~ weekend + promotion + prev_day_customers + avg_customers_7day,
  data = train_pred,
  family = "binomial"
)
summary(pred_log_model)

# B.5 Evaluate predictive model performance
pred_log_probs <- predict(pred_log_model, newdata = test_pred, type = "response")
pred_log_preds <- ifelse(pred_log_probs > 0.5, 1, 0)

pred_log_conf <- table(Predicted = pred_log_preds, Actual = test_pred$high_revenue)
pred_log_metrics <- calculate_metrics(pred_log_conf)

# B.6 Compare original vs predictive logistic regression
performance_comparison <- tibble(
  Model = c("Original Logistic Regression", "Predictive Logistic Regression"),
  Accuracy = c(log_metrics$Accuracy, pred_log_metrics$Accuracy),
  Precision = c(log_metrics$Precision, pred_log_metrics$Precision),
  Recall = c(log_metrics$Recall, pred_log_metrics$Recall),
  F1 = c(log_metrics$F1, pred_log_metrics$F1),
  Features_Used = c("Same-day data (customers, promotion, weekend)", 
                    "Previous day data (lagged customers, 7-day avg)")
) %>%
  mutate(across(where(is.numeric), ~round(.x * 100, 1)))

print(performance_comparison)

# B.8 Feature coefficient comparison
coef_comparison <- tibble(
  Model = c(rep("Original", 3), rep("Predictive", 4)),
  Feature = c("daily_customers", "promotion", "weekend",
              "weekend", "promotion", "prev_day_customers", "avg_customers_7day"),
  Coefficient = c(coef(log_model)[-1], coef(pred_log_model)[-1]),
  Std_Error = c(summary(log_model)$coefficients[-1, 2], 
                summary(pred_log_model)$coefficients[-1, 2]),
  P_Value = c(summary(log_model)$coefficients[-1, 4],
              summary(pred_log_model)$coefficients[-1, 4])
) %>%
  mutate(
    Coefficient = round(Coefficient, 4),
    Std_Error = round(Std_Error, 4),
    P_Value = round(P_Value, 4),
    Significance = case_when(
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ ""))

print(coef_comparison)

# Insight: Without same-day customer data, the model shifts to structural factors.
# Promotions matter more, weekends are strong predictors, and previous day customers have litte impact.

# B.9 Practical application example
tomorrow_data <- tibble(
  weekend = TRUE,
  promotion = FALSE,
  prev_day_customers = tail(retail_clean$daily_customers, 1),
  avg_customers_7day = mean(tail(retail_clean$daily_customers, 7))
)

tomorrow_prob <- predict(pred_log_model, newdata = tomorrow_data, type = "response")
tomorrow_class <- ifelse(tomorrow_prob > 0.5, "High Revenue", "Low Revenue")

recommendation <- tibble(
  Scenario = "Tomorrow Prediction",
  Weekend = tomorrow_data$weekend,
  Promotion = tomorrow_data$promotion,
  Prev_Day_Customers = tomorrow_data$prev_day_customers,
  Predicted_Class = tomorrow_class,
  Probability = paste0(round(tomorrow_prob * 100, 1), "%"),
  Staffing_Recommendation = ifelse(
    tomorrow_prob > 0.7, "Full staff + overtime available",
    ifelse(tomorrow_prob > 0.5, "Full staff scheduled",
           ifelse(tomorrow_prob > 0.3, "Regular staff levels",
                  "Reduced staff acceptable"))
  )
)

print(recommendation)






