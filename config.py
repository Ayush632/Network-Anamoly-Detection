from   elasticsearch import helpers, Elasticsearch
import elasticsearch
import csv

# Other configuration related functions can be defined here.
# The config functions can be used in the main  module in future increments to the pipeline.

TRAIN_DATA     = 'kdd_train.csv'
TEST_DATA      = 'kdd_test.csv'
TRAIN_INDEX    = 'train-data-index'
TEST_INDEX     = 'test-data-index'
REALTIME_INDEX = 'realtime-data-index'

def createIndex(index_name):
    try:
        res = es.indices.create(index=index_name, ignore=400)
        if res.get('acknowledged') == True:
            print('Index {} created successfully'.format(index_name))
            return 
        if res.get('status') == 400:
            print('Index {} already exists'.format(index_name))
            return
    except Exception as e:
        print(e)
        return

def deleteIndex(index_name):
    try:
        res = es.indices.delete(index=index_name, ignore=[400,404])
        if res.get('acknowledged') == True:
            print('Index {} deleted successfully'.format(index_name))
            return 
        if res.get('status') == 404:
            print('Index {} does not exist'.format(index_name))
            return
    except Exception as e:
        print(e)
        return

def createTimestampPipeline():
    body = {
        "description": "Creates a timestamp when a document is initially indexed",
        "processors": [
            {
                "set": {
                    "field": "_source.timestamp",
                    "value": "{{_ingest.timestamp}}"
                }
            }
        ]
    }
    res = elasticsearch.client.IngestClient(es).put_pipeline("timestamp", body)
    print(res)
    return

def addBulkDocuments(file_path,index_name):
    # with open(file_path) as f:
    #     reader = csv.DictReader(f)
    try:
        with open(file_path) as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index=index_name, pipeline='timestamp')
        return 
    except:
        print("Failed to add data from", file_path,"to",index_name)


if __name__ == "__main__":
    es = Elasticsearch()
    # Index name can be an env variable. Currently have stored them as constants.
    
    createIndex(TRAIN_INDEX)
    createIndex(TEST_INDEX)
    createIndex(REALTIME_INDEX)
    createTimestampPipeline()
    
    addBulkDocuments(TRAIN_DATA,TRAIN_INDEX)
    addBulkDocuments(TEST_DATA,TEST_INDEX)
    