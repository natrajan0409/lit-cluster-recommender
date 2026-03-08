import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_audible_catlog(self):
        file_path = os.path.join(self.data_dir, "Audible_Catlog.csv")
        return pd.read_csv(file_path)

    def load_audible_catlog_adv(self):
        file_path = os.path.join(self.data_dir, "Audible_Catlog_Advanced_Features.csv")
        return pd.read_csv(file_path)

def audible_catlog(data_dir):
    loader = DataLoader(data_dir)
    return loader.load_audible_catlog()

def audible_catlog_adv(data_dir):
    loader = DataLoader(data_dir)
    return loader.load_audible_catlog_adv()
