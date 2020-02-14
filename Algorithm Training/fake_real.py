# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:05:30 2019

@author: bhanu
"""
# importing pandas for dataframe modification
import pandas as pd

#importing text extraction vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

#importing the test train spliter for making the test and train split
from sklearn.cross_validation import train_test_split

# importing the pickle file
import pickle

# import numpy
import numpy as np
#Creating a pickle file
pickle_out = open("trained_classifiers.pickle", "wb")




# reading a text file from the local host
#df  = pd.read_csv("fake-news/train.csv")
#dftest = pd.read_csv("fake-news/test.csv")

df = pd.read_csv("fake_or_real_news.csv")
# encoding the fake value with 0
df.loc[df["label"]=='FAKE', "label"] = 0
# encoding the  real value with 1
df.loc[df["label"]=='REAL', "label"] = 1


# creating new dataframe of dependent variables
y = df.label

# creating the X_train which is independent  for training
# creating the X_test which is dependent  for testing
# creating the y_train which is independent  for training
# creating the y_test which is dependent  for testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=53)

pickle.dump(X_train, pickle_out)



# count vectorizer which will convert the words into the mathematical dataset
cv = CountVectorizer(stop_words='english')
x_traincv = cv.fit_transform(X_train)
a = x_traincv.toarray()
#inversed array
cv_inversed = cv.inverse_transform(a[0])


# tfidf vectorizer which will convert the words into the mathematical dataset
td = TfidfVectorizer(stop_words='english', max_df=0.7)
x_traintd = td.fit_transform(X_train)
atd = x_traintd.toarray()
#inversed array
td_inversed = td.inverse_transform(atd[0])

# hv vectorizer which will convert the words into the mathematical dataset
hv = HashingVectorizer(stop_words='english', non_negative=True)
x_trainhv = hv.fit_transform(X_train)


# creating a tfidf transformed x_test
x_testtd = td.transform(X_test)
# creating a count vectorizer transformed x_test
x_testcv = cv.transform(X_test)

# dumping the x_testcv
pickle.dump(X_test,pickle_out)
pickle.dump(y_test, pickle_out)


# creating a hash vectorizer transformed x_test
x_testhv = hv.transform(X_test)

# converting a string dependent variable to a int
y_train = y_train.astype('int')



# machine learning algorithm for the multinomial Naive Bayes Classifier using the tfid Vectorizer
from sklearn.naive_bayes import MultinomialNB
#mnb = MultinomialNB()
#mnb.fit(x_traintd,y_train)


# machine learning algorithm for the multinomial Naive Bayes Classifier using the count vectorizer
mnb2 = MultinomialNB().fit(x_traincv,y_train)

# dumping mnb2
pickle.dump(mnb2, pickle_out)


# machine learning algorithm for the multinomial Naive Bayes Classifier using the hash vectorizer
#mnb3 = MultinomialNB()
#mnb3.fit(x_trainhv,y_train)


# machine learning algorithm Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, random_state = 42).fit(x_traincv,y_train)


# dumping random forest
pickle.dump(rf, pickle_out)


# machine learning algorithm Decision Tree
from sklearn.tree import DecisionTreeClassifier  
decision = DecisionTreeClassifier().fit(x_traincv, y_train)  
pickle.dump(decision, pickle_out)

# machine learning algorithm Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression().fit(x_traincv,y_train)

# dumping logistic regression



# machine learning algorithm support vector machine
from sklearn import svm
clf = svm.SVC(gamma=0.001).fit(x_traincv, y_train)

# dumping svm
pickle.dump(clf, pickle_out)

# machine learning algorithm kneighbours
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3).fit(x_traincv, y_train)


# dumping knn
pickle.dump(neigh, pickle_out)


# preddiction
#predmnbtd = mnb.predict(x_testtd)
#predmnbhv = mnb3.predict(x_testhv)


predmnbcv = mnb2.predict(x_testcv)
predrfcv = rf.predict(x_testcv)
preddecisioncv = decision.predict(x_testcv)
predlgcv = logistic.predict(x_testcv)
predsvmcv = clf.predict(x_testcv)
predneighcv = neigh.predict(x_testcv)







# creating a compound machine learning algorithm storing in the final
final = []
for i in range(1267):
    c = 0
    c = predmnbcv[i]+predrfcv[i]+preddecisioncv[i]+predsvmcv[i]+predneighcv[i]

# based on the vote we will opt for the most
    if c>=3:
        final.append(1)
    else:
        final.append(0)


#  importing a confusion matrix
from sklearn.metrics import confusion_matrix
# importing a accuracy score
from sklearn.metrics import accuracy_score





# confusion matrix for the tfidf mnb 
#cmmnbtd = confusion_matrix(y_test,predmnbtd)
# accuracy score for the tfidf mnb
#accmnbtd = accuracy_score(y_test,predmnbtd)

# confusion matrix for the cv mnb 
cmmnbcv = confusion_matrix(y_test,predmnbcv)
# accuracy score for the cv mnb
accmnbcv = accuracy_score(y_test,predmnbcv)

# confusion matrix for the cv rf 
cmrfcv = confusion_matrix(y_test,predrfcv)
# accuracy score for the cv rf
accrfcv = accuracy_score(y_test,predrfcv)

# confusion matrix for the cv decisiontree
cmdecisioncv = confusion_matrix(y_test,preddecisioncv)
# accuracy score for the cv decisiontree
accdecisioncv = accuracy_score(y_test,preddecisioncv)

# confusion matrix for the cv lg
cmlgcv = confusion_matrix(y_test,predlgcv)
# accuracy score for the cv lg
acclgcv = accuracy_score(y_test,predlgcv)

# confusion matrix for the cv svm
cmsvmcv = confusion_matrix(y_test,predsvmcv)
# accuracy score for the cv svm
accsvmcv = accuracy_score(y_test,predsvmcv)

# confusion matrix for the cv neighbour
cmcvneigh = confusion_matrix(y_test,predneighcv)
# accuracy score for the cv neighbour
acccvneigh = accuracy_score(y_test,predneighcv)

# confusion matrix for the cv final set
cmcvf = confusion_matrix(y_test,final)
# accuracy score for the cv final
acccvf = accuracy_score(y_test,final)



import matplotlib.pyplot as plt
 
# Data to plot
labels = ['positive', 'false neg', 'false pos', 'negative']
colors = ['green', 'red', 'orange', 'purple']
explode = (0, 0, 0, 0) 
font_size = 15


# naive bayes
cmmnbcvpie = cmmnbcv.tolist()
sizes = []
for i in cmmnbcv:
    for j in i:
        sizes.append(j)
 # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': font_size})
plt.title("Naive bayes confusion matrix")
plt.axis('equal')
plt.show()


# random Forest 
cmrfcvpie = cmrfcv.tolist()
sizes = []
for i in cmrfcvpie:
    for j in i:
        sizes.append(j)
 # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': font_size})
plt.title("Random Forest confusion matrix")
plt.axis('equal')
plt.show()

# Decision tree
cmdecisioncvpie = cmdecisioncv.tolist()
sizes = []
for i in cmdecisioncvpie:
    for j in i:
        sizes.append(j)
 # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': font_size})
plt.title("Decision Tree confusion matrix")
plt.axis('equal')
plt.show()


# svm
cmsvmcvpie = cmsvmcv.tolist()
sizes = []
for i in cmsvmcvpie:
    for j in i:
        sizes.append(j)
 # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': font_size})
plt.title("SVM confusion matrix")
plt.axis('equal')
plt.show()


# Knn
cmcvneighpie = cmcvneigh.tolist()
sizes = []
for i in cmcvneighpie:
    for j in i:
        sizes.append(j)
 # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': font_size})
plt.title("kNN confusion matrix")
plt.axis('equal')
plt.show()


# multinomial voting algorithm
cmcvfpie = cmcvf.tolist()
sizes = []
for i in cmcvfpie:
    for j in i:
        sizes.append(j)
 # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': font_size})
plt.title("Final confusion matrix")
plt.axis('equal')
plt.show()




label = ['Naive Bayes', 'Random Forest', 'Decision Tree', 'SVM', 'KNN', 'multinomial voting']
accuracy_score = [89.6, 86.2, 83, 89.1, 82, 92.1]

index = np.arange(len(label))
plt.bar(index, accuracy_score)
plt.xlabel('Algorithm', fontsize=font_size)
plt.ylabel('Accuracy_Score', fontsize=font_size)
plt.xticks(index, label, fontsize=font_size, rotation=30)
plt.title('algorithm vs accuracy')
plt.show()


PRF = [[526,87, 44, 610], [547, 66, 108, 546], [495, 118, 97, 557], [575, 38, 99, 555], [521, 92, 136, 518], [570, 43, 57, 597]]
precision = []
recall = []
f1_score = []

for i in PRF:
    p = (i[0])/(i[0]+i[2])
    precision.append(p)
    r = (i[0])/(i[0]+i[1])
    recall.append(r)
    f1 = 2*((p*r)/(p+r))
    f1_score.append(f1)
    
    
    
plt.bar(index, precision)
plt.xlabel('Algorithm', fontsize=font_size)
plt.ylabel('precision', fontsize=font_size)
plt.xticks(index, label, fontsize=font_size, rotation=30)
plt.title('algorithm vs precision')
plt.show()
    

plt.bar(index, recall)
plt.xlabel('Algorithm', fontsize=font_size)
plt.ylabel('recall', fontsize=font_size)
plt.xticks(index, label, fontsize=font_size, rotation=30)
plt.title('algorithm vs recall')
plt.show()


plt.bar(index, f1_score)
plt.xlabel('Algorithm', fontsize=font_size)
plt.ylabel('f1_score', fontsize=font_size)
plt.xticks(index, label, fontsize=font_size, rotation=30)
plt.title('algorithm vs f1_score')
plt.show() 
    
    
    
    