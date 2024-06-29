from generate_dataset import generate_data

dataset = generate_data(2)
dataset.to_csv("testing.csv")
