import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pandas.plotting import scatter_matrix
import seaborn as sns
#import matplotlib as mpl
#from matplotlib import cm
import os
import sys
import datetime
from datetime import timedelta
import random
from scipy.stats import uniform, randint

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,f1_score, recall_score
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib #https://joblib.readthedocs.io/en/latest/
from imblearn.under_sampling import TomekLinks
from sklearn.neighbors import NearestNeighbors

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours

#MLflow libraries
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

path = './'
# path = '/content/drive/MyDrive/Colab Notebooks/Industria_4/Analitica_Datos/'
name = 'household_power_consumption.csv'

def load_preprocess_data():
    df = pd.read_csv(path+name, sep = ';',
                parse_dates={'Fecha':['Date','Time']},
                infer_datetime_format=True,
                low_memory=False, na_values=['nan','?']
                )

    df.sort_values(by='Fecha')
    
    #Data imputation
    for i in df.columns[1:]:
        median = df[i].median() # option 3
        df[i].fillna(median, inplace=True)
    
    #select just the columns needed
    df3 = df.loc[:int(len(df)/2), ['Global_active_power', 'Global_reactive_power', 
                               'Voltage',	'Global_intensity',	'Sub_metering_2',
                               'Sub_metering_3']]
    
    #assign categorical tags to the data using the folowing rules
    conditions = [
    (df3['Sub_metering_2'] == 0) & (df3['Sub_metering_3'] == 0), ## Etiqueta 0: No hay consumo
    df3['Sub_metering_2'] > df3['Sub_metering_3'], ## Etiqueta 1: Mayor consumo Zona 2
    df3['Sub_metering_2'] < df3['Sub_metering_3'], ## Etiqueta 2: Mayor consumo Zona 3
    df3['Sub_metering_2'] == df3['Sub_metering_3']] ## Etiqueta 3: Consumo en ambas zonas por igual
    choices = [0, 1, 2, 3]
    y = np.select(conditions, choices, default=4)
    df3.drop(columns=['Sub_metering_2',	'Sub_metering_3'], inplace=True)

    counter = Counter(y)
    print('Distribución de las clases:', counter)

    df3['label'] = y
    Xdata = df3.copy()

    return Xdata, y

def split_balance_data(Xdata, y):
    # Tamaño Xtrain 70%, Tamaño Xtest 30%
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, y, test_size=0.3, random_state=2)
    Xprod, Xtest, yprod, ytest = train_test_split(Xtest, ytest, test_size=0.5, random_state=2)
    Xtest1 = Xtest.copy()
    print('-Train (70%):', Xtrain.shape, '\n-Test (15%):', Xtest.shape, '\n-Production (15%):', Xprod.shape)
    
    # summarize class distribution
    counter = Counter(ytrain)
    print('Sin balancear las clases:', counter)

    # transform the dataset
    undersample = EditedNearestNeighbours(n_neighbors=7, sampling_strategy = 'not minority', kind_sel='all')
    # transform the dataset
    X1, y1 = undersample.fit_resample(Xtrain, ytrain)
    # summarize the new class distribution
    counter = Counter(y1)
    print('Balanceo con Edited:', counter)

    # define the undersampling method
    under = TomekLinks()
    X2, y2 = under.fit_resample(X1, y1)
    # summarize the new class distribution
    counter = Counter(y2)
    print('Balanceo con TomeLinks:', counter)
    return X2, Xtest, Xprod, y2, ytest, yprod

if __name__ == "__main__":
    Xdata, y = load_preprocess_data()
    X2, Xtest, Xprod, y2, ytest, yprod = split_balance_data(Xdata, y)

    #read user input parameters
    nneighb = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    with mlflow.start_run():
        #train the model
        sca = StandardScaler()
        X2 = sca.fit_transform(X2)
        model = KNeighborsClassifier(n_neighbors=nneighb)
        model.fit(X2,y2)

        #test the model
        Xtest = sca.fit_transform(Xtest)
        ypred = model.predict(Xtest)
        acc = accuracy_score(ytest, ypred)
        rec = recall_score(ytest, ypred, average='weighted')
        f1 = f1_score(ytest, ypred, average='weighted')

        #show and log results
        print("N_neighbors: %s" % nneighb)
        print("  Accuracy: %s" % acc)
        print("  F1-Score: %s" % f1)
        print("  Recall: %s" % rec)
        #log parameters and metrics to MLflow
        mlflow.log_param("N_neighbors", nneighb)
        mlflow.log_metric("Recall", rec)
        mlflow.log_metric("F1-Score", f1)
        mlflow.log_metric("Accuracy", acc)

        #store current model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(model, "model")