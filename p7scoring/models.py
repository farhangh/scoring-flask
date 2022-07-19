import pickle
import pandas as pd

def read_data(path="data/p7_data.csv"):
    return pd.read_csv(path).reset_index(drop=True)

def load_model(path="data/best_lr_t.pkl"):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)

def get_model_params(path="data/best_lr_t.pkl"):
    model = load_model(path)
    return model.get_params()