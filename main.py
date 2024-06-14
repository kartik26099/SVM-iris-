import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\classification\IRIS.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
classifier=SVC(kernel="linear",random_state=0)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))
