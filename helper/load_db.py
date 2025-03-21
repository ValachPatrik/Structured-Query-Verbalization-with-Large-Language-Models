import os
import pandas as pd
from helper.preprocess_lc_quad import *

def load_or_combine_dataset(dataset_name):
    if os.path.exists(dataset_name):
        # Load from the saved CSV file
        combined_df = pd.read_csv(dataset_name)
        print("Loaded combined dataset from saved file.")
        return combined_df
    
    preprocess_lc_quad()
    
    if os.path.exists(dataset_name):
        # Load from the saved CSV file
        combined_df = pd.read_csv(dataset_name)
        print("Loaded combined dataset from saved file.")
        return combined_df
    raise ValueError("Dataset not generated")