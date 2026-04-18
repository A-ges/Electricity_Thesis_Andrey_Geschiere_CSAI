import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import HfFileSystem

"""
This file reads all 5 solar data files from the 2024 OpenSTEF project. It then gets the mean generation for each location combined per hour, returning
solar_elasticities for all 24 hours [0-23]. These elasticities will be used in the price estimator.

A low elasticity: A great increase in demand, causes a small increase in price (around 12:00, a lot of surplus supply is sold for a relatively cheap price)
A high elasticity: A great increase in demand, causes a big increase in price (in the evening/night, the low green energy supply leads to extra costs for the DSO)
"""

#boundaries for elasticity, leading to plausible in/decreases in prices
elasticity_min = 0.1
elasticity_max = 0.6

#Loading data from OpenSTEF, for dataset details: https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark

fs = HfFileSystem()

solar_folder = "datasets/OpenSTEF/liander2024-energy-forecasting-benchmark/load_measurements/solar_park"

files = fs.glob(f"{solar_folder}/*.parquet")  #Get the list of all 5 solar park measurement files

dataframes = []

for file_path in files:
    solar_path = f"hf://{file_path}"

    solar_df = pd.read_parquet(solar_path)  #read the pq file

    solar_df["timestamp"] = pd.to_datetime(solar_df["timestamp"])
    solar_df = solar_df.set_index("timestamp")

    solar_df["generation"] = solar_df["load"] * -1  #flips the sign, higher load = higher value rather than original

    dataframes.append(solar_df)

#Combine all Dutch solar parks
df_all = pd.concat(dataframes)

solar_mean_per_hour = df_all.groupby(df_all.index.hour)["generation"].mean()

solar_normalized = solar_mean_per_hour / solar_mean_per_hour.max()

solar_elasticity = elasticity_max - ((elasticity_max - elasticity_min) * solar_normalized)

solar_elasticities = solar_elasticity.tolist()

print(solar_elasticities)  #To copy for calculating prices

plt.show()
