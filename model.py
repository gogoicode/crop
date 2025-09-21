import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\Users\aayan\OneDrive\Desktop\MCA1_Code\pjt\crop\Crop_recommendation_ds.csv")

x = data.iloc[:,:-1] #features
y = data.iloc[:,-1] #labels

X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, Y_train)

pickle.dump(model, open("model.pkl", "wb"))