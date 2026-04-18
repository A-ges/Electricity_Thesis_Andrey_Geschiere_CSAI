import pandas as pd
import matplotlib.pyplot as plt
"""
This file reads the EPEX price file from the 2024 OpenSTEF project. It then gets the mean price for every hour, 
these values will be used as baseline price in the price generator formula

EPEX SPOT is a European market where electricity is bought and sold at a country specific-price.
The price (in MWh) is the market price of electricity for each hour. It is the price paid by DSO's to get their energy for that hour from the central distributor,
they then provide it to consumers and add their own costs. Non-dynamic consumer prices in the Netherlands are often around ~0,26 euro per kWh 
Price is set by supply and demand: when electricity is scarce, the price goes up; when there is plenty available, the price goes down.

All prices are finalized thourgh dividing by ten to get smaller numbers

NOTE: EPEX prices are in MWh, my model (and Williams et al. (2025)) returns KW. Because I will only work with ratios and relative differences when it comes to pricing,
I will keep these amounts as they are and view them as unnamed pricing units for every KW.

"""
url = "https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark/resolve/main/EPEX.parquet"
df_epex = pd.read_parquet(url)

df_epex["timestamp"] = pd.to_datetime(df_epex["timestamp"])
df_epex.set_index("timestamp", inplace=True)

df_epex["hour"] = df_epex.index.hour

#Mean over all hours per hour in dataset
baseline_prices = df_epex.groupby("hour")["EPEX_NL"].mean().values
baseline_prices = (baseline_prices/10).round(3)
baseline_list = baseline_prices.tolist()
print(baseline_list)

#Plotting
hours = range(24)

plt.figure(figsize=(10, 5))
plt.plot(hours, baseline_prices/10, marker="o", linewidth=2)

plt.title("Mean EPEX Price per Hour (Baseline)", fontsize=14)
plt.xlabel("Uur van de dag")
plt.ylabel("Price (€/MWh) / 10")
plt.xticks(hours) 
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
