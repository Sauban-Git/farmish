import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler


model = joblib.load('SFS2.pkl')
scaler = joblib.load('scaler2.pkl')
label_encoder = joblib.load('label_encoder.pkl')


input_data = {
    'N': [90],  
    'P': [42],  
    'K': [43],   
    'temperature': [20.87974371], 
    'humidity': [82.00274423], 
    'ph': [6.502985292000001] 
}

featur = pd.DataFrame(input_data)
features = scaler.transform(featur)

probabilities = model.predict_proba(features)
predictions = model.predict(features)

cropn = label_encoder.inverse_transform(predictions.astype(int))

crop_names = model.classes_
maxx = probabilities.argmax()
cropp = crop_names[maxx]
confi = probabilities[0][maxx] * 100
print(f"crop: {cropn[0]} with confidence: {confi:.2f}%")

df = pd.read_csv('meancrop.csv')
def getrow (crop_name):
    row = df[df['crop'] == crop_name]
    if not row.empty:
        return row
    else:
        return "Value Error"

print(f"Most suitable proportion for {cropn[0]} is:")
result = getrow(cropn[0])
print(result)