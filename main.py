import pandas as pd

df = pd.read_parquet("hf://datasets/SantiagoPG/customer_service_chatbot/0000.parquet")

x=df.drop("conversation",axis=1)
x
y=df["conversation"]
y