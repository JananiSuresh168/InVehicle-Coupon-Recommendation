# Load libraries
import pandas as pd
import numpy as np
import csv
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import random
import seaborn as sn
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from dtreeviz.trees import *
from sklearn import tree


# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay


def combine_rows(df):
    df['education'] = df['education'].replace(['Graduate degree (Masters or Doctorate)', 'Some college - no degree','Some High School'], 'master/no degree/high school')
    df['occupation'] = df['occupation'].replace(['Architecture & Engineering', 'Arts Design Entertainment Sports & Media', 'Office & Administrative Support'],'Arts and support')
    df['occupation'] = df['occupation'].replace(['Building & Grounds Cleaning & Maintenance','Business & Financial','Computer & Mathematical','Farming Fishing & Forestry','Food Preparation & Serving Related','Installation Maintenance & Repair','Life Physical Social Science','Personal Care & Service','Sales & Related','Transportation & Material Moving'],'building/sales')
    df['occupation'] = df['occupation'].replace(['Community & Social Services','Protective Service','Unemployed'],'Community/Unemployed')
    df['occupation'] = df['occupation'].replace(['Construction & Extraction','Education&Training&Library','Legal','Production Occupations','Student'],'construction/education')
    df['occupation'] = df['occupation'].replace(['Healthcare Practitioners & Technical','Retired'],'healthcare/retired')
    df['occupation'] = df['occupation'].replace(['Healthcare Support','Management'],'health support/management') 

                
# Load dataset
url = "C:\\Users\\vanam\\Downloads\\in-vehicle-coupon-recommendation.csv"
dataset = read_csv(url)

# printing the dataset summary 
pd.options.display.max_columns = None
dataset.info()
print(dataset.describe())

# correlation matrix
corrMatrix = dataset.corr()
sn.heatmap(corrMatrix, annot=True)
pyplot.show()

# drop the attribute 'car' and remove instances with null values
dataset.drop('car', axis=1, inplace = True)
dataset.drop('toCoupon_GEQ5min', axis=1, inplace = True)
dataset.drop('direction_opp', axis=1, inplace = True)
df = dataset.dropna()


#randomize the data 
df = df.sample(frac = 1)


# combine data after exploratory data analysis (weight of evidence and information value) 
combine_rows(df)


df.to_csv("C:\\Users\\vanam\\Downloads\\in-vehicle-coupon-recommendation-cleaned.csv",index=False)



# Load dataset
url = "C:\\Users\\vanam\\Downloads\\in-vehicle-coupon-recommendation-cleaned.csv"
dataset = read_csv(url)



# shape
pd.options.display.max_columns = None
# print("Dataset shape(instances, attributes): {}".format(dataset.shape))


#encoding data
#creating labelEncoder
le = preprocessing.LabelEncoder()
obj_df = dataset.select_dtypes(include=['object']).copy()

# one hot encoding the categorial data
he = pd.get_dummies(obj_df, columns = obj_df.columns)

# converting the data type to int64
for column in he.columns:
    he[column] = he[column].astype(np.int64)

#combining the numerical and hot encoded data
num_data = dataset.select_dtypes(include=['int64'])


# adding column names to list
cols = []
for col in he.columns:
    cols.append(col)
for col in num_data.columns:
    cols.append(col)

# creating dataframe combining all the numerical values
df = pd.concat([he,num_data], ignore_index=True, axis=1)
df.columns = cols


# print(df.head(10))

# split X and y into training and testing sets
X = df.drop(['Y'], axis=1)
y = df['Y']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.30, random_state=1)
# print(Y_train.shape, Y_validation.shape)
# print(X_train.shape, X_validation.shape)


# Random forest

# Create a Random forest Classifier
clf=RandomForestClassifier(n_estimators= 1000,
 min_samples_split= 2,
 min_samples_leaf= 1,
 max_features= 'sqrt',
 max_depth= 110,
 bootstrap= True)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,Y_train)

y_pred = clf.predict(X_validation)

# Accuracy
print("Random Forest accuracy:",accuracy_score(Y_validation, y_pred))

# classification report - precision, recall, f1, support
print("Random Forest classification report - \n", classification_report(Y_validation,y_pred))

# confusion matrix
cm = confusion_matrix(Y_validation, y_pred)
plt.figure(figsize=(5,5))
sn.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues',fmt='g')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Random Forest Accuracy Score: {0}'.format(clf.score(X_validation, Y_validation))
plt.title(all_sample_title, size = 15)
plt.show()

# ROC area under curve calculation
y_proba = clf.predict_proba(X_validation)[:,1]
print("Random Forest Roc AUC:", roc_auc_score(Y_validation, clf.predict_proba(X_validation)[:,1],average='macro'))
fpr, tpr, thresholds = roc_curve(Y_validation, y_proba)
plt.plot(fpr, tpr,label='Random Forest Classifier')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Random Forest ROC curve')
plt.legend(loc='best')
plt.savefig('1.png')
plt.show()


# decision tree

# Create decision tree classifier
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=50, random_state=42)
dtree.fit(X_train,Y_train)

# Predicting the values of test data
y_pred = dtree.predict(X_validation)

# Accuracy
print("Decision Tree accuracy:",accuracy_score(Y_validation, y_pred))

# classification report - precision, recall, f1, support
print("Decision Tree classification report - \n", classification_report(Y_validation,y_pred))

# confusion matrix
cm = confusion_matrix(Y_validation, y_pred)
plt.figure(figsize=(5,5))
sn.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues',fmt='g')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Decision Tree Accuracy Score: {0}'.format(dtree.score(X_validation, Y_validation))
plt.title(all_sample_title, size = 15)
plt.show()

# ROC area under curve calculation
y_proba = dtree.predict_proba(X_validation)[:,1]
print("Roc AUC:", roc_auc_score(Y_validation, dtree.predict_proba(X_validation)[:,1],average='macro'))
fpr, tpr, thresholds = roc_curve(Y_validation, y_proba)
plt.plot(fpr, tpr,label='Decision Tree Classifier')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Decision Tree ROC curve')
plt.legend(loc='best')
plt.savefig('2.png')
plt.show()

# Visualising the graph 
# tree.plot_tree(dtree)


# Logistic Regression
classifier = LogisticRegression(C= 1.0,
 class_weight= None,
 dual= False,
 fit_intercept= True,
 intercept_scaling= 1,
 l1_ratio= None,
 max_iter= 1000,
 multi_class= 'auto',
 n_jobs= None,
 penalty= 'l2',
 random_state= None,
 solver= 'lbfgs',
 tol= 0.0001,
 verbose= 0,
 warm_start= False)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_validation)

# Accuracy
print("Logistic Regression accuracy:",accuracy_score(Y_validation, y_pred))

# classification report - precision, recall, f1, support
print("Logistic Regression classification report - \n", classification_report(Y_validation,y_pred))

# confusion matrix
cm = confusion_matrix(Y_validation, y_pred)
plt.figure(figsize=(5,5))
sn.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues',fmt='g')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Logistic Regression Accuracy Score: {0}'.format(classifier.score(X_validation, Y_validation))
plt.title(all_sample_title, size = 15)
plt.show()

# ROC area under curve calculation
y_proba = classifier.predict_proba(X_validation)[:,1]
print("Roc AUC:", roc_auc_score(Y_validation, classifier.predict_proba(X_validation)[:,1],average='macro'))
fpr, tpr, thresholds = roc_curve(Y_validation, y_proba)
plt.plot(fpr, tpr,label='Logistic Regression Classifier')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic Regression ROC curve')
plt.legend(loc='best')
plt.savefig('3.png')
plt.show()
