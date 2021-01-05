import argparse
import xgboost as xgb
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory as tdf
from sklearn.metrics import accuracy
import joblib
import os


def main():
    
    parser = argparse.ArgumentParser()
    
    # TODO: change
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    data_path = 'https://raw.githubusercontent.com/tybyers/AZMLND_projects/capstone/capstone/data/train_1_10k.csv'
    dataset = tdf.from_delimited_files(path=data_path)
    x_df = dataset.to_pandas_dataframe()
    y_df = x_df.pop("HasDetections")
    
    
    
    


if __name__ == '__main__':
    main()