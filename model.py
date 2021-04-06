import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("Iris.csv")

from sklearn.preprocessing import LabelEncoder
lblencoder = LabelEncoder()
data['Species'] = lblencoder.fit_transform(data['Species'])

data.drop("Id",axis=1,inplace=True)
X = data.iloc[:,[0,1,2,3]]
y = data['Species']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)
from sklearn.metrics import accuracy_score
print('Accuracy Score:', accuracy_score(y_test, y_pred))

pickle.dump(knn, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.3, 1.6, 2.6, 2.4]]))