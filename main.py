# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:26:30 2022

@author: vadiv
"""
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

header = st.container()
dataset = st.container()
features = st.container()
viz = st.container()
model_training = st.container()

import os
cwd = os.getcwd()

inp_data = pd.read_csv(os.path.join(cwd,"titanic.csv"))


inp_data=inp_data[pd.notnull(inp_data['fare'])]
inp_data=inp_data[pd.notnull(inp_data['embarked'])]

inp_data['label']='Others'
for i in ['Mr.','Mrs.','Master.','Miss.','Dr.']:
           res = inp_data['name'].str.contains(i)
           inp_data.loc[res,'label'] = i
           
X = inp_data[['pclass','sibsp','parch','fare','sex','embarked','label']]
X = pd.get_dummies(X) #one hot encoding for all categorical variables in train_X
Y = pd.DataFrame(inp_data['survived'])

train_x, test_x, train_y, test_y = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state=6)

with header:
    st.title("1. Welcome to my Project")
    st.text("In this project, we are going to build a binary classification model")
    
with dataset:
    st.header("2. Titanic Dataset")
    st.text("We are going to use titanic dataset which can be found in the below link:")
    st.write("[Kaggle_titanic_data](https://www.kaggle.com/c/titanic/data)")
    st.text("We will be predicting the survival of passengers on Titanic ship")
    st.text("Dependent variable: Survived")
    st.text("Independent variables: Age, Passenger class, Sex, embarked location, etc")
    st.subheader("2.1 Available Features")
    st.write(inp_data.columns.values.tolist())
    st.subheader("2.2 Few sample rows : ")
    st.write(inp_data.head(5))
    st.subheader("2.3 Distribution of numerical features :")
    st.write(inp_data.describe())
    st.subheader("Creating derived variable from 'name' column")
    st.write(inp_data['label'].value_counts())
    

with features:
    st.header("3. Exploratory data analysis of features")
    feat1 = st.selectbox("choose variable for exploratory analysis", options=["pclass","sex","sibsp","parch","embarked","label"], index=0)
    st.subheader("Average survival rate by variable")
    st.write(inp_data[[feat1, 'survived']].groupby([feat1], as_index=False).mean().sort_values(by='survived', ascending=False))
    st.subheader("Distribution bar chart : ")
    pclass_dist = inp_data[feat1].value_counts()
    st.bar_chart(pclass_dist)

with viz:
    st.header("4. Visualization")
    st.subheader("Histogram of Age : ")
    g = sns.FacetGrid(inp_data, col='survived')
    g.map(plt.hist, 'age', bins=20)
    st.pyplot(g)
    st.subheader("Correlating numerical and categorical features : ")
    grid = sns.FacetGrid(inp_data, col='survived', row='sex', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'age', alpha=.5, bins=20)
    grid.add_legend();
    st.pyplot(grid)
    grid1 = sns.FacetGrid(inp_data, row='survived', col='sex', size=2.2, aspect=1.6)
    grid1.map(sns.barplot, 'embarked', 'fare', alpha=.5, ci=None)
    grid1.add_legend()
    st.pyplot(grid1)
    
    
with model_training:
    st.header("5. Model Training details")
    algo = st.selectbox("choose the machine learning algorithm", options=["Random Forest","Decision Tree","Logistic Regression","KNN"], index=0)
    st.text("Here you can choose the hyper parameters and see how model performance changes")
    sel_col, disp_col = st.columns(2)
    if (algo=="Random Forest"):
        maxdepth = sel_col.slider("What should be the max depth of the model?",1,10,5)
        minsamplesleaf = sel_col.selectbox("min number of samples in leaf node", options=[4,6,8,10], index=0)
        nestimators = sel_col.selectbox("how many trees should be there", options=[100,200,300], index=2)   
        clf = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth, criterion='entropy', min_samples_leaf=minsamplesleaf, verbose=2)
    
    if (algo=="Decision Tree"):
        maxdepth = sel_col.slider("What should be the max depth of the model?",1,10,3)
        minsamplesleaf = sel_col.selectbox("min number of samples in leaf node", options=[4,6,8,10], index=0)
        clf = DecisionTreeClassifier(max_depth=maxdepth, criterion='entropy', min_samples_leaf=minsamplesleaf)

    if (algo=="Logistic Regression"):
        clf = LogisticRegression()
        
    if (algo=="KNN"):    
        n = sel_col.slider("number of neighbors?",1,10,3)
        clf = KNeighborsClassifier(n_neighbors = n)
        
    clf.fit(train_x,train_y['survived'])
    test_pred=clf.predict(test_x)
    train_pred=clf.predict(train_x)
    disp_col.subheader("Train Accuracy : ")
    disp_col.write(accuracy_score(train_y, train_pred))
    disp_col.subheader("Test Accuracy : ")
    disp_col.write(accuracy_score(test_y, test_pred))