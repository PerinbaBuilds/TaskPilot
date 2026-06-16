"""
Verifies that all required dataset files load correctly.
Usage: python check_datasets.py
"""
import pandas as pd

try:
    tasks_L = pd.read_csv("dataset_rl/task_15min_L.csv")
    print(f"tasks_L     : {tasks_L.shape}")
except FileNotFoundError:
    print("tasks_L     : NOT FOUND (dataset_rl/task_15min_L.csv)")

try:
    tasks_S = pd.read_csv("dataset_rl/task_15min_S.csv")
    print(f"tasks_S     : {tasks_S.shape}")
except FileNotFoundError:
    print("tasks_S     : NOT FOUND (dataset_rl/task_15min_S.csv)")

try:
    server_L = pd.read_excel("dataset_rl/Server_L.xlsx")
    print(f"server_L    : {server_L.shape}")
except FileNotFoundError:
    print("server_L    : NOT FOUND (dataset_rl/Server_L.xlsx)")

try:
    server_S = pd.read_excel("dataset_rl/Server_S.xlsx")
    print(f"server_S    : {server_S.shape}")
except FileNotFoundError:
    print("server_S    : NOT FOUND (dataset_rl/Server_S.xlsx)")

try:
    price_df = pd.read_csv("dataset_rl/price.csv")
    print(f"price       : {price_df.shape}")
except FileNotFoundError:
    print("price       : NOT FOUND (dataset_rl/price.csv)")

try:
    steel = pd.read_csv("steel_industry_data.csv")
    print(f"steel       : {steel.shape}")
except FileNotFoundError:
    print("steel       : NOT FOUND (steel_industry_data.csv)")
