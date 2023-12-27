import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import chardet
#Data collection and Preproccessing
#loading data from csv file to a pandas dataframe
file1 ='dataset1.csv'
with open(file1, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
df1= pd.read_csv(file1)
print(df1)
df1 = df1.drop('Unnamed: 0', axis=1)
df1 = df1.drop('label_num', axis=1)
print(df1)
#df.sample(5)
df1.shape
##Steps
#1. Cleaning the data
#2. Exploratory data analysis
#3. Text preproccessing
#4. Model building
#5. Evaluation
df1.info()
df1.sample(5)
df1.rename(columns={'v1':'label','v2':'text'},inplace=True)
df1.sample(5)
encoder=LabelEncoder()
df1.loc[df1['label']=='spam','label',]=0
df1.loc[df1['label']=='ham','label',]=1
df1.sample(5)
#seperating the data as texts and label
X = df1['text']
Y = df1['label']

#Splitting the data into training data & test data
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X,Y,test_size=0.2,random_state=3)


#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

#we will transform the text data to feature vectors that can be used as input to the Logistic Regression Model
feature_extraction1 = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features1 = feature_extraction1.fit_transform(X_train1)
X_test_features1 = feature_extraction1.transform(X_test1)
#convert Y_train and Y_test values as integers

Y_train1 = Y_train1.astype('int')
Y_test1 = Y_test1.astype('int')


# print(X_train_features)

# # Training the model

# # Logistic Regression

model1 = LogisticRegression()

#training the Logistic Regression model with the training data
model1.fit(X_train_features1, Y_train1)


# # Evaluating the trained model


# prediction on training data
prediction_on_training_data1 = model1.predict_proba(X_train_features1)
prediction1 = [1 if np.any(i > 0.1) else 0 for i in prediction_on_training_data1]
accuracy_on_training_data1 = accuracy_score(Y_train1, prediction1)

print('Accuracy on training data : ',accuracy_on_training_data1)

# prediction on test data
prediction_on_test_data1 = model1.predict(X_test_features1)
accuracy_on_test_data1 = accuracy_score(Y_test1, prediction_on_test_data1)


print('Accuracy on test data : ',accuracy_on_test_data1)


# FOR DATASET2

file2 ='dataset2.csv'
with open(file2, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
df2= pd.read_csv(file2)
print(df2)
print(df2)
#df.sample(5)
df2.shape
##Steps
#1. Cleaning the data
#2. Exploratory data analysis
#3. Text preproccessing
#4. Model building
#5. Evaluation
df2.info()
df2.sample(5)
df2.rename(columns={'v1':'text','v2':'label'},inplace=True)
df2.sample(5)
en2oder=LabelEncoder()
df2.loc[df2['label']=='spam','label',]=0
df2.loc[df2['label']=='ham','label',]=1
df2.sample(5)
#seperating the data as texts and label
X = df2['text']
Y = df2['label']

#Splitting the data into training data & test data
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X,Y,test_size=0.2,random_state=3)


#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

#we will transform the text data to feature vectors that can be used as input to the Logistic Regression Model
feature_extraction2 = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features2 = feature_extraction2.fit_transform(X_train2)
X_test_features2= feature_extraction2.transform(X_test2)
#convert Y_train and Y_test values as integers

Y_train2 = Y_train2.astype('int')
Y_test2 = Y_test2.astype('int')


# print(X_train_features)

# # Training the model

# # Logistic Regression

model2 = LogisticRegression()

#training the Logistic Regression model with the training data
model2.fit(X_train_features2, Y_train2)


# # Evaluating the trained model


# prediction on training data
prediction_on_training_data2 = model2.predict(X_train_features2)
accuracy_on_training_data2 = accuracy_score(Y_train2, prediction_on_training_data2)



print('Accuracy on training data : ',accuracy_on_training_data2)

# prediction on test data
prediction_on_test_data2 = model2.predict(X_test_features2)
accuracy_on_test_data2 = accuracy_score(Y_test2, prediction_on_test_data2)


print('Accuracy on test data : ',accuracy_on_test_data2)




# # Building a Predictive System 


# input_mail=["Free entry in 2 a wkly comp to win."]
# #converting text to feature vectors
# input_data_features = feature_extraction.transform(input_mail)
# #making prediction
# prediction = model.predict(input_data_features)
# print(prediction)

def lr_predict_spam(mail,dsNumber):
    if(dsNumber==1):
        input_data_features = feature_extraction1.transform(mail)
        prediction = model1.predict(input_data_features)
        if prediction == 1 :
            return "HAM",accuracy_on_test_data1, accuracy_on_training_data1
        return "SPAM",accuracy_on_test_data1, accuracy_on_training_data1   

    elif(dsNumber==2):
        input_data_features = feature_extraction2.transform(mail)
        prediction = model2.predict(input_data_features)
        if prediction == 1 :
            return "HAM",accuracy_on_test_data2, accuracy_on_training_data2
        return "SPAM",accuracy_on_test_data2, accuracy_on_training_data2
    else:
        return "-1", None , None 
    
  
    


