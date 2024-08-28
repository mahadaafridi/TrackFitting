from generate_dataset import generate_data

if __name__ == "__main__":

    dataset = generate_data(10)
    dataset.to_csv("data/testing.csv")
