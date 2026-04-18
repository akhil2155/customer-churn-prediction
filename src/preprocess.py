import pandas as pd

def load_data():
    df = pd.read_csv("data/customer_churn.csv")
    return df