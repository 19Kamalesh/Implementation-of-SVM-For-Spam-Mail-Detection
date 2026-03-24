# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess data: Import data, inspect it, and handle missing values if any.

2.Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.

3.Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features.

4.Assign cluster labels to each data point.

5.Plot data points in a scatter plot, color-coded by cluster assignments for interpretation
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Kamaleshwaran S
RegisterNumber:  212225040165
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
<img width="1001" height="458" alt="image" src="https://github.com/user-attachments/assets/739dc58f-f823-4c0a-9208-1f320e4e16ba" />
<img width="235" height="55" alt="image" src="https://github.com/user-attachments/assets/dd20eebb-9fec-4509-96af-bc8d571f3232" />
<img width="215" height="47" alt="image" src="https://github.com/user-attachments/assets/57413fc2-06f2-4a72-ae58-0ede3ea4f880" />
<img width="219" height="27" alt="image" src="https://github.com/user-attachments/assets/744224c4-8456-4f32-af95-e81da1adfefa" />
<img width="870" height="326" alt="image" src="https://github.com/user-attachments/assets/85a05d7e-83d5-4d4f-8299-262538e4ef30" />
<img width="198" height="31" alt="image" src="https://github.com/user-attachments/assets/67c369ab-f946-4cfc-9b04-aa950705900c" />
<img width="820" height="51" alt="image" src="https://github.com/user-attachments/assets/993b6283-8e21-4c2d-b536-0b0bbbbf8f8a" />
<img width="287" height="59" alt="image" src="https://github.com/user-attachments/assets/7b90ab9e-4474-4142-80f4-8da4cd56e24e" />
<img width="206" height="66" alt="image" src="https://github.com/user-attachments/assets/5d122c79-fbe2-4383-9d20-94079432e533" />
<img width="638" height="204" alt="image" src="https://github.com/user-attachments/assets/ca6ff15b-c5a1-462d-be40-62a868fbe9fd" />









## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
