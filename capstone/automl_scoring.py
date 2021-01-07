import json
import pandas as pd
import os
import joblib
from azureml.core.model import Model
import azureml.train.automl


def init():
    global model
    model_path = Model.get_model_path('best_automl')
    model = joblib.load(model_path)

def run(data):
    try:
        df = pd.DataFrame.from_dict(json.loads(data)['data'])
        df.drop(['MachineIdentifier', 'HasDetections'], axis=1, inplace=True)
        result = model.predict(df)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error