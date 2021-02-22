import pandas as pd
import numpy as np
import json, csv
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import load_model

with open('mappings.json') as f: MAPPINGS = json.load(f)

DOS_lis=['back','land','neptune','pod','smurf','teardrop','mailbomb','processtable','udpstorm','apache2','worm']
Probe_lis=['ipsweep','portsweep','nmap','satan','nmap','mscan','saint']
R2l_lis=['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
U2R_lis=['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']
other_lis=['warezclient','spy']

FLAG          = MAPPINGS['flag']
SERVICE       = MAPPINGS['service']
PROTOCOL_TYPE = MAPPINGS['protocol_type']
#THRESHOLD     = 0.065

class Anomaly():
    def __init__(self):
        self.X = self.readData()
        self.pca       = self.getPCAObject(5)
        self.model     = load_model('network2.h5')
    
    def readData(self):
        # DOS_lis=['back','land','neptune','pod','smurf','teardrop','mailbomb','processtable','udpstorm','apache2','worm']
        # Probe_lis=['ipsweep','portsweep','nmap','satan','nmap','mscan','saint']
        # R2l_lis=['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
        # U2R_lis=['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']
        # other_lis=['warezclient','spy']
        train_data=pd.read_csv('kdd_train.csv')
        train_data['protocol_type'] = train_data['protocol_type'].map(PROTOCOL_TYPE)
        train_data['flag']          = train_data['flag'].map(FLAG)
        train_data['service']       = train_data['service'].map(SERVICE)
        features = train_data.columns.tolist()[:-1]
        X_train=train_data[features]
        return X_train
    
    def getPCAObject(self, n):
        X_normalized, norms = normalize(self.X, norm='l2', return_norm=True)
        pca                 = PCA().fit(X_normalized)
        pca                 = PCA(n_components=n) # n=5

        pca.fit(X_normalized)

        return pca

    def anomaly(self,doc):
        test_data=pd.read_csv('kdd_test.csv')
        test_data = test_data.append(doc,ignore_index=True)
        test_data['protocol_type'] = test_data['protocol_type'].map(PROTOCOL_TYPE)
        test_data['flag']          = test_data['flag'].map(FLAG)
        test_data['service']       = test_data['service'].map(SERVICE)
        
        d = test_data['labels'].isin(DOS_lis)
        p=test_data['labels'].isin(Probe_lis)
        r=test_data['labels'].isin(R2l_lis)
        u=test_data['labels'].isin(U2R_lis)
        o=test_data['labels'].isin(other_lis)
        test_data['labels']=test_data['labels'].mask(d,'DOS')
        test_data['labels']=test_data['labels'].mask(p,'Probe')
        test_data['labels']=test_data['labels'].mask(r,'R2L')
        test_data['labels']=test_data['labels'].mask(u,'U2R')
        test_data['labels']=test_data['labels'].mask(o,'Other')
        features = test_data.columns.tolist()[:-1]
        anomaly_test_data=test_data.loc[test_data['labels']!='normal']
        X_test_anomaly=anomaly_test_data[features]
        Xt_normalized_anomaly, norms = normalize(X_test_anomaly, norm='l2', return_norm=True)
        Xt_reduced_anomaly=self.pca.transform(Xt_normalized_anomaly)
        Yt=anomaly_test_data['labels']
        Y_test=pd.get_dummies(Yt).values
        #curr_data                  = ans_test_means.tail(1)
        #if(curr_data.item() < THRESHOLD):
        #    return 'normal'
        #else:
        #    return 'anamoly'

        yt_pred=self.model.predict(Xt_reduced_anomaly)
        #print(yt_pred)
        y_test_class=np.argmax(Y_test,axis=1)
        yt_pred_class=np.argmax(yt_pred,axis=1)
        #print(len(yt_pred_class))
        curr_anomaly                 = yt_pred_class[-1]
        #print(y_test_class)
        if curr_anomaly == 0:
            return 'DOS'
        elif curr_anomaly == 1:
            return 'Probe'
        elif curr_anomaly == 2:
            return 'R2L'
        elif curr_anomaly == 3:
            return 'U2R'
        else:
            return 'Other'
        
if __name__ == "__main__":
    # For testing the script
    obj = Predictor()
    with open('kdd_test.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(obj.predict(row))
            break