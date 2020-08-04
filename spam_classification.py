# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:42:10 2019

@author: nlakhotia
"""
# Import necessary libraries
# Dataset download from- https://archive.ics.uci.edu/ml/machine-learning-databases/00228/

import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
# for different classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from nltk.classify.scikitlearn import SklearnClassifier


# Load the dataset

df=pd.read_table('SMSSpamCollection',header=None,encoding='utf-8')
#print(df.head(),"\n")
#print(df.info())
classes=df[0]
#print(classes.value_counts())

''' Pre-processing '''
# Convert the classes into binary format
# 0="ham"  ,  1="spam"
le=LabelEncoder()
Y=le.fit_transform(classes)  #new class variable
#print(Y[0:10])
text_messages=df[1]
#print(text_messages[:10])
# Use Regular Expressions to replace email addresses,symbols,phone nos,urls etc

# replacing email addresses with emailaddr

processed=text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddr')
# replacing urls with webaddress
processed=processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
# replace money symbols with moneysymb
processed=processed.str.replace(r'£|\$','moneysymb')
# replace phone numbers with phonenumb
processed=processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
# replace normal numbers with numbr
processed=processed.str.replace(r'\d+(\.\d+)?','numbr')
# remove punctuation
processed=processed.str.replace(r'[^\w\d\s]', ' ')
# remove whitespace with a single space
processed=processed.str.replace(r'\s+', ' ')
# remove leading and trailing whitespace
processed=processed.str.replace(r'^\s+|\s+?$','')
#convert text to lowercase
processed=processed.str.lower()


# Remove Stopwords
stop_words=set(stopwords.words('english'))
processed=processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# Remove word stems-Stemming procedure using Porter stemmer
ls = WordNetLemmatizer()
#ps=nltk.PorterStemmer()
processed=processed.apply(lambda x: ' '.join(ls.lemmatize(term) for term in x.split()))
#print(processed)

''' Feature extration '''
# Create tokens and bag-of-words
all_words=[]
for message in processed:
    words=word_tokenize(message)
    for w in words:
        all_words.append(w)
all_words=nltk.FreqDist(all_words)    # this is the bag-of-words
    
#print('No of words:{}'.format(len(all_words)))
#print('Most common words:{}'.format(all_words.most_common(15)))
# we use the 1500 most common words as features
word_features=list(all_words.keys())[:1500]
#print(word_features)
# define a find_feature function
def find_features(message):
    words=word_tokenize(message)
    features={}
    for word in word_features:
        features[word]=(word in words)
    return features

messages=zip(processed,Y)
seed=1
#call find_features function for each sms message
# In the form of X,y
featuresets=[(find_features(text),label) for (text,label) in messages]

# split training,testing sets using sklearn
training,testing=train_test_split(featuresets,test_size=0.25,random_state=seed)

# Define models to train
names=['K Nearest Neighbors','Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Naive Bayes','SVM Linear']
classifiers=[
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        LogisticRegression(),
        SGDClassifier(max_iter=100,tol=None),
        MultinomialNB(),
        SVC(kernel='linear')
        ]
models=zip(names,classifiers)
# wrap models in NLTK
for name,model in models:
    nltk_model=SklearnClassifier(model)
    nltk_model.train(training)
    accuracy=nltk.classify.accuracy(nltk_model,testing)*100
    print('{}: Accuracy: {}'.format(name,accuracy))
    print()
# make class label predictions for testing set
# unzip the testing data
txt_features,labels=zip(*testing)
prediction=nltk_model.classify_many(txt_features)
print(classification_report(labels,prediction))
info=pd.DataFrame(
        confusion_matrix(labels,prediction),
        index=[['actual','actual'],['ham','spam']],
        columns=[['predicted','predicted'],['ham','spam']],
        )
print(info)