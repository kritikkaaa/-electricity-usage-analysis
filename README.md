# -electricity-usage-analysis
This project simulates and analyzes electricity consumption for 10 residential homes over 30 days. It's built using NumPy and includes realistic usage trends, billing simulation, and anomaly detection.

import numpy as np

# Seed for reproducibility
np.random.seed(7)

# Define homes and days
homes = [f"Home_{i}" for i in range(1, 11)]
days = np.arange(1, 31)

# Simulate usage: base usage + daily increase trend
usage_matrix = np.random.normal(loc=18, scale=5, size=(len(homes), len(days))) + np.linspace(0, 5, len(days))

# Clamp values to realistic range (5 to 40 kWh)
usage_matrix = np.clip(usage_matrix, 5, 40)

# 1. Average daily usage per home
print("Average Daily Usage per Home:")
home_avgs = np.mean(usage_matrix, axis=1)
for i, avg in enumerate(home_avgs):
    print(f"{homes[i]}: {avg:.2f} kWh")

# 2. Average usage per day across all homes
print("\nAverage Usage per Day (Community):")
daily_avg = np.mean(usage_matrix, axis=0)
for day, avg in zip(days, daily_avg):
    print(f"Day {day:02}: {avg:.2f} kWh")

# 3. High usage days (community average > 30 kWh)
print("\nHigh Usage Detection (Community Days > 30 kWh avg):")
high_days = np.where(daily_avg > 30)[0]
for d in high_days:
    print(f"Day {d+1:02} → Avg Usage: {daily_avg[d]:.2f} kWh")

# 4. Monthly bill calculation per home
rate_per_kwh = 7  # in INR
total_bills = np.sum(usage_matrix, axis=1) * rate_per_kwh
print("\nMonthly Bill Summary:")
for i in range(len(homes)):
    print(f"{homes[i]}: ₹{total_bills[i]:.2f}")

# 5. Detect usage spikes > 35 kWh
print("\nUsage Anomalies (days with spike > 35 kWh):")
for i in range(len(homes)):
    spikes = np.where(usage_matrix[i] > 35)[0]
    if len(spikes) > 0:
        print(f"{homes[i]} → Spike days: {[int(d+1) for d in spikes]}")

# 6. Normalized usage (0–1) — optional preprocessing for ML/plotting
norm_usage = (usage_matrix - usage_matrix.min()) / (usage_matrix.max() - usage_matrix.min())
print("\nNormalized Usage Sample (Home_1):")
print(np.round(norm_usage[0], 2))

