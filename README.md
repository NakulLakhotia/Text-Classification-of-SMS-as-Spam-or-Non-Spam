# NLP-Projects
## Project 3 - Text Classification of SMS as Spam or Non-Spam- Bag of Words Model

Python Libraires used: Pandas, re (for Regular Expressoins) , nltk (Natural Language Toolkit), Scikit-Learn
**) The project involves a large dataset due to which the output maybe delayed

Project Workflow:

1) Import all the necessary libraries

2) Load the dataset file using pandas

3) Pre-Processing of data
      
      3.1) Converted the class label 'spam' to 1 & 'ham' to 0 using LabelEncoder
      
      3.2) Using Regular Expressions replaced email-address with 'emailaddr' & URLs with 'webaddress' & Money Symbols with 'moneysymb'
           & 10 digit phone-no with 'phonenumbr' & numbers with 'numbr'
           
      3.3) Remove all punctuation marks, white-spaces 
      
      3.4) Convert all the text to lower-case
      
      3.5) Remove all stopwords from each text message
      
      3.6) Lemmatize each word of the text messages

4) Feature Extraction
     
      4.1) Tokenize each text message and create the bag-of-words
      
      4.2) Find the 15 most common words from the bag-of-words
      
      4.3) We use 1500 most common words as features
      
      4.4) Define a find_feature function which returns all the words from the text message (passed as an argument) which is
           present among the 1500 most common words
           
      4.5) The feature_set stores the features for each text message along with the 'label'
 
 5) Training & Classification
      
      5.1) Split training,testing sets using sklearn
      
      5.2) Define models to train
      
      5.3) Classifiers used= 'K Nearest Neighbors','Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Naive Bayes','SVM Linear'
      
      5.4) Wrap the above different models in NLTK
      
      5.5) Using Sklearn Classifier we train the training data
      
      5.6) Find the accuracy of each of the model
      
      5.7) Make the class label predictions for the testing data
      
  
 6) Results
     
     6.1) Print the classification_report
      
      6.2) Print the confusion matrix
      
      6.3) In Non-spam 1200 text messages were correctly classified
      
      6.4) In Spam 173 text messages were correctly classified
      
      6.5) 8 Non-spam messages were misclassified as spam
      
      6.6) 12 Spam messages were misclassified as Non-spam
      

      
           

