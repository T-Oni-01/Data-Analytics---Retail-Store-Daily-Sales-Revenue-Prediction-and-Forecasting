# Load libraries
library(ggplot2)
library(dplyr)
library(lubridate)
library(summarytools)

# Read the data
retail <- read.csv("data_analytics_retail.csv")

# Basic overview
str(retail)
head(retail)

# Check for missing values
print("Missing values:")
colSums(is.na(retail))

# Convert date to proper format
retail$date <- as.Date(retail$date)

# Basic statistics
summary(retail)

# More detailed summary
view(dfSummary(retail))

# Create binary classification target (High Revenue = 1, Low = 0)
median_revenue <- median(retail$daily_revenue, na.rm = TRUE)
retail$high_revenue <- ifelse(retail$daily_revenue > median_revenue, 1, 0)

# Time series plot of daily revenue
ggplot(retail, aes(x = date, y = daily_revenue)) +
  geom_line(color = "steelblue") +
  geom_point(aes(color = factor(high_revenue)), size = 1) +
  geom_hline(yintercept = median_revenue, linetype = "dashed", color = "red") +
  labs(title = "Daily Revenue Over Time",
       subtitle = "Red line = Median Revenue",
       x = "Date", y = "Daily Revenue") +
  theme_minimal()

# Revenue by weekend vs weekday
ggplot(retail, aes(x = factor(weekend), y = daily_revenue, fill = factor(weekend))) +
  geom_boxplot() +
  labs(title = "Revenue Distribution: Weekend vs Weekday",
       x = "Weekend", y = "Daily Revenue") +
  theme_minimal()

# Revenue by promotion
ggplot(retail, aes(x = factor(promotion), y = daily_revenue, fill = factor(promotion))) +
  geom_boxplot() +
  labs(title = "Revenue Distribution: Promotion vs No Promotion",
       x = "Promotion", y = "Daily Revenue") +
  theme_minimal()

# Correlation between temperature and revenue
ggplot(retail, aes(x = temperature, y = daily_revenue)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Temperature vs Daily Revenue",
       x = "Temperature (°F)", y = "Daily Revenue") +
  theme_minimal()

# Revenue by day of week
ggplot(retail, aes(x = day_of_week, y = daily_revenue, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Revenue by Day of Week",
       x = "Day of Week", y = "Daily Revenue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Summary statistics by high revenue category
retail %>%
  group_by(high_revenue) %>%
  summarise(
    count = n(),
    avg_temperature = mean(temperature, na.rm = TRUE),
    avg_customers = mean(daily_customers),
    promotion_rate = mean(promotion),
    weekend_rate = mean(weekend)
  )

