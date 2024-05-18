import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 


#Loading the dataset to pd dataframe
diabetes_dataset = pd.read_csv('diabetes\diabetes.csv') 

# printing the first 5 rows of the dataset
print("HEAD:")
print(diabetes_dataset.head())

# number of rows and Columns in this dataset
print("SHAPE:")
print(diabetes_dataset.shape)

# getting the statistical measures of the data
print("DESCRIBE:")
print(diabetes_dataset.describe())


# gives the number of 0 and 1s
print(diabetes_dataset['Outcome'].value_counts())


# separating the data and labels
x = diabetes_dataset.drop(columns = 'Outcome', axis=1)


#Data Standardization

scalermodel=StandardScaler()
scalermodel.fit(x)
standard_data=scalermodel.transform(x)

x=standard_data
y = diabetes_dataset['Outcome']

#Train Test Split

xtrain,xtest,ytrain,ytest=train_test_split(x,y, test_size=0.2,random_state=2)
print(x.shape, xtrain.shape, xtest.shape)


#training the model

svmmodel=svm.SVC()
svmmodel.fit(xtrain,ytrain)

trainpredict=svmmodel.predict(xtrain)
trainaccuracy=accuracy_score(trainpredict,ytrain)
print("Train Accuracy:",trainaccuracy)


testpredict=svmmodel.predict(xtest)
testaccuracy=accuracy_score(testpredict,ytest)
print("Test Accuracy:",testaccuracy)


s=(5,166,72,19,175,25.8,0.587,51)
sarray=np.asarray(s)
sre=sarray.reshape(1,-1)
sdata=scalermodel.transform(sre)

prediction=svmmodel.predict(sdata)
print(prediction)


if prediction[0]==0:
    print("The patient is not diabetic")
else:
    print("death")



