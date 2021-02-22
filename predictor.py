import pandas as pd
import numpy as np
import json, csv
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from keras.models import load_model

with open('mappings.json') as f: MAPPINGS = json.load(f)

FLAG          = MAPPINGS['flag']
SERVICE       = MAPPINGS['service']
PROTOCOL_TYPE = MAPPINGS['protocol_type']
THRESHOLD     = 0.065

class Predictor():
    def __init__(self):
        self.X, self.Y = self.readData()
        self.pca       = self.getPCAObject(5)
        self.model     = load_model('t2tmodel.h5')
    
    def readData(self):
        data_init             = pd.read_csv('kdd_train.csv')
        data                  = data_init.loc[data_init['labels'] == 'normal']
        data['protocol_type'] = data['protocol_type'].map(PROTOCOL_TYPE)
        data['flag']          = data['flag'].map(FLAG)
        data['service']       = data['service'].map(SERVICE)
        features              = data.columns.tolist()[:-1]
        X_train, Y_train      = data[features], data['labels']

        return X_train, Y_train
    
    def getPCAObject(self, n):
        X_normalized, norms = normalize(self.X, norm='l2', return_norm=True)
        pca                 = PCA().fit(X_normalized)
        pca                 = PCA(n_components=n) # n=5

        pca.fit(X_normalized)

        return pca

    def predict(self,doc):
        test_data                  = pd.read_csv('kdd_test.csv')
        test_data = test_data.append(doc,ignore_index=True)

        test_data['protocol_type'] = test_data['protocol_type'].map(PROTOCOL_TYPE)
        test_data['flag']          = test_data['flag'].map(FLAG)
        test_data['service']       = test_data['service'].map(SERVICE)
        features                   = test_data.columns.tolist()[:-1]
        X_test                     = test_data[features]

        lis                        = [("column" + str(i)) for i in range(1,6)]
        X_test_normalized, norms   = normalize(X_test, norm='l2', return_norm=True)
        X_test_reduced             = self.pca.transform(X_test_normalized)
        X_test_reduced             = pd.DataFrame(data = X_test_reduced,columns = lis)
        X_test_pred                = self.model.predict(np.array(X_test_reduced))
        
        
        X_test_diff                = np.abs(X_test_pred-X_test_reduced)
        ans_test_df                = pd.DataFrame(X_test_diff, columns=lis)
        ans_test_means             = ans_test_df.mean(axis=1)
        
        curr_data                  = ans_test_means.tail(1)
        if(curr_data.item() < THRESHOLD):
            return 'normal'
        else:
            return 'anamoly'

if __name__ == "__main__":
    # For testing the script
    obj = Predictor()
    with open('kdd_test.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(obj.predict(row))
            break