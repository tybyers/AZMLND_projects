import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory as tdf
import joblib
import os

LABEL_COLS = ['ProductName', 'EngineVersion', 'AppVersion', 'AvSigVersion', 'RtpStateBitfield', 'DefaultBrowsersIdentifier', 
              'CityIdentifier', 'OrganizationIdentifier', 'Platform', 'Processor', 'OsVer', 'OsPlatformSubRelease', 
              'OsBuildLab', 'SkuEdition', 'PuaMode', 'SMode', 'IeVerIdentifier', 'SmartScreen', 'UacLuaenable', 
              'Census_MDC2FormFactor', 'Census_DeviceFamily', 'Census_OEMNameIdentifier', 'Census_OEMModelIdentifier', 
              'Census_ProcessorClass', 'Census_PrimaryDiskTypeName', 'Census_TotalPhysicalRAM', 'Census_ChassisTypeName', 
              'Census_InternalPrimaryDiagonalDisplaySizeInInches', 'Census_InternalPrimaryDisplayResolutionHorizontal', 
              'Census_InternalPrimaryDisplayResolutionVertical', 'Census_PowerPlatformRoleName', 'Census_InternalBatteryType', 
              'Census_InternalBatteryNumberOfCharges', 'Census_OSVersion', 'Census_OSArchitecture', 'Census_OSBranch', 
              'Census_OSEdition', 'Census_OSSkuName', 'Census_OSInstallTypeName', 'Census_OSInstallLanguageIdentifier', 
              'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName', 'Census_ActivationChannel', 'Census_IsFlightingInternal',
              'Census_IsFlightsDisabled', 'Census_FlightRing', 'Census_ThresholdOptIn', 'Census_FirmwareManufacturerIdentifier', 
              'Census_FirmwareVersionIdentifier', 'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice', 'Wdft_IsGamer', 
              'Wdft_RegionIdentifier']

DATA_PATH = 'https://raw.githubusercontent.com/tybyers/AZMLND_projects/capstone/capstone/data/train_1_10k.csv'

class RunXGBHyperDrive:

    def __init__(self, data_path = None):
        
        self.lab_encs = {}
        self.model = None
        if data_path is None:
            self.data_path = DATA_PATH
        else:
            self.data_path = data_path

    def load_data_tdf(self, data_path = None):

        if data_path is None:
            data_path = self.data_path
        dataset = tdf.from_delimited_files(path=data_path)
        df = dataset.to_pandas_dataframe()

        return df

    def load_data_csv(self, data_path = None):

        if data_path is None:
            data_path = self.data_path
        df = pd.read_csv(data_path)
    
        return df

    def _clean_data(self, df):

        lab_encs = {}
        for label in LABEL_COLS:
            try:
                df[label] = df[label].astype(str)
                le = LabelEncoder().fit(df[label])
                df[label] = le.transform(df[label])
                lab_encs[label] = le
            except ValueError:
                df.drop(label, axis=1, inplace=True)
                print('Dropping column {}'.format(label))
                lab_encs[label] = None

        self.lab_encs = lab_encs
        X = df.copy()
        y = X.pop("HasDetections")

        return X, y

    def run_cv(self, df, max_depth=6, n_estimators=100, learning_rate=0.3):

        model = xgb.XGBClassifier(max_depth = max_depth, n_estimators=n_estimators, 
                                    learning_rate=learning_rate)

        X, y = self._clean_data(df)

        kfold = KFold(n_splits=5)
        results = cross_val_score(model, X.values, y.values, cv=kfold)

        self.model = model
        return results


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_depth', type=int, default=6, 
                        help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.")
    parser.add_argument('--n_estimators', type=int, default=100, 
                        help="Number of boosting rounds. Trains faster when smaller; too large can overfit.")
    parser.add_argument('--learning_rate', type=float, default=0.3, 
                        help='Step size shrinkage used in update to prevents overfitting. Range: [0, 1]')
    parser.add_argument('--from_csv', type=str, default='False', 
                        help='Load from CSV. Use this if using offline (testing purposes mostly).')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to data')

    args = parser.parse_args()    

    modelrun = RunXGBHyperDrive(data_path=args.data_path)

    if args.from_csv == 'True':  # probably running locally
        print('max_depth: {}'.format(np.int(args.max_depth)))
        print('n_estimators: {}'.format(np.int(args.n_estimators)))
        print('learning_rate: {}'.format(np.float(args.learning_rate)))
        df = modelrun.load_data_csv()
    else:
        run = Run.get_context()
        run.log('max_depth:', np.int(args.max_depth))
        run.log('n_estimators:', np.int(args.n_estimators))
        run.log('learning_rate:', np.float(args.learning_rate))
        df = modelrun.load_data_tdf()

    results = modelrun.run_cv(df, args.max_depth, args.n_estimators, args.learning_rate)

    if args.from_csv == 'True':
        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    else:
        run.log("Accuracy", np.float(results.mean()))

if __name__ == '__main__':
    main()