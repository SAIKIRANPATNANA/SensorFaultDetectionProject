from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = 'mongodb+srv://saikiranpatnana:MAYYA143@saikiran.bdu0jbl.mongodb.net/'
client = MongoClient(uri)
database_name = 'saikiranpatnana'
collection_name = 'waterfault'
df = pd.read_csv('/Users/saikiranpatnana/Documents/SensorFaultDetection/notebooks/wafers.csv')
df = df.drop('Unnamed: 0',axis=1)
json_record = list(json.loads(df.T.to_json()).values())
client[database_name][collection_name].insert_many(json_record)



