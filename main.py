from elasticsearch import Elasticsearch
import csv, time
from datetime import datetime
from predictor import Predictor
from anomaly_type import Anomaly

REALTIME_INDEX = 'realtime-data-index'

def addDocument(doc):
    try:
        es.index(REALTIME_INDEX, doc, pipeline='timestamp')
        return
    except:
        # Ideally you would have a log file that would store the failure to send data
        print("Failed to send data at", datetime.now())
        return

if __name__ == "__main__":
    '''
        Ideally, we would receive realtime data in a file (maybe csv or json).
        The data would be read from the file, stored to ES and deleted from the file.
    '''
    # Reading from test file for testing purposes
    es   = Elasticsearch()
    file = 'kdd_test.csv'
    pred = Predictor()
    with open(file) as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            res = pred.predict(row)
            row['labels'] = res
            
            if res == 'anamoly':
                anam = Anomaly()
                anomaly_type = anam.anomaly(row)
                # if anomaly_type != 'DOS':
                # print(row,"\n",res)
                # print("Anomaly Type = ",anomaly_type)
                row['labels'] = anomaly_type
            print(row,"\n",res)
            start = time.time()
            while((time.time() - start ) < 10):
                pass

            addDocument(row)
            i += 1
            # if i == 10:
            #     break


    
