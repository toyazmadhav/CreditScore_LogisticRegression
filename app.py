import wget
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
import math

from xverse.transformer import MonotonicBinning,WOE
from sklearn.metrics import classification_report

def loadData():
  url = 'https://cdn.iisc.talentsprint.com/CDS/MiniProjects/GiveMeSomeCredit.csv'
  filename = wget.download(url)
  df = pd.read_csv(filename)
  return df

def plot_for_all_cols(df: pd.DataFrame, plot_kind = 'box'):
    '''
    Plots box or hist plot for all cols of given dataframe, assuming all cols are numeric
    '''
    plt_num_cols = 2
    plt_num_rows = int(np.ceil(len(df.columns) / plt_num_cols))
    # https://discuss.streamlit.io/t/matplotlib-plots-are-blurry/1224
    fig, axes = plt.subplots(nrows = plt_num_rows, ncols = plt_num_cols, figsize = (40,40))
    plt.subplots_adjust(hspace = 0.4)

    for idx, col in enumerate(df.columns):
        plt_row_idx = int(np.floor(idx / plt_num_cols))
        plt_col_idx = idx % plt_num_cols
        axis = axes[plt_row_idx][plt_col_idx]
        if plot_kind == 'box':
            sns.boxplot(data = df[col], ax = axis)
        else:
            sns.histplot(data = df[col], bins = 100, ax = axis)
        axis.set_title(col)

    st.pyplot(fig)
  
def train_test_split_helper(df, outputCol):
  X = df[df.columns.drop(outputCol)]
  y = df[[outputCol]]
  return train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 14)

def handle_outliers(X_train, X_test, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    # q1 = df.describe()[colm].loc["25%"]
    # q3 = df.describe()[colm].loc["75%"]
    # iqr = q3 - q1
    # lower_bound = q1 - (2 * iqr)
    # upper_bound = q3 + (2 * iqr)

    upper_bound=X_train[colm].mean() + 2*X_train[colm].std()
    lower_bound=X_train[colm].mean() - 2*X_train[colm].std()

    X_train.loc[X_train[colm] < lower_bound,colm] = lower_bound
    X_train.loc[X_train[colm] > upper_bound,colm] = upper_bound

    X_test.loc[X_test[colm] < lower_bound,colm] = lower_bound
    X_test.loc[X_test[colm] > upper_bound,colm] = upper_bound
    # for i in range(len(df)):
    #     if np.isnan(df.loc[i,colm]) or df.loc[i,colm] == None:
    #         continue
    #     if df.loc[i,colm] > upper_bound:
    #         df.loc[i,colm]= upper_bound
    #     if df.loc[i,colm] < lower_bound:
    #         df.loc[i,colm]= lower_bound
    return X_train, X_test

# https://www.analyticsvidhya.com/blog/2022/02/implementing-logistic-regression-from-scratch-using-python/

from numpy import log,dot,exp,shape

class LogisticRegressionClassifier:
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    def fit(self,X,y,alpha=0.001,iter=400):
        weights,X = self.initialize(X)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -(( cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha * dot(X.T, self.sigmoid(dot(X, weights)) - np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis

def train_using_custom_logistic(X_train, y_train):
  lg_clf = LogisticRegressionClassifier()
  lg_clf.fit(X_train.to_numpy(),y_train.to_numpy(), alpha = 0.0001, iter = 1000)
  return lg_clf

def train_using_logistic(X_train, y_train, isBalanced = False, class_weights = {0.5 ,0.5}):
  balanced = None
  if isBalanced:
    balanced = 'balanced'
  lg_clf = LogisticRegression(random_state=0, class_weight = balanced)
  
  model = lg_clf.fit(X_train ,y_train)
  return model

def main():
  df = loadData()
  print(df.head())
  df = df.drop(columns=['Unnamed: 0'])
  st.write('# Histogram Plot')
  plot_for_all_cols(df, 'hist')

  st.write('# Box Plot')
  plot_for_all_cols(df, 'box')

  X_train, X_test, y_train, y_test = train_test_split_helper(df, 'SeriousDlqin2yrs')

  outlier_cols = X_train.columns
  # outlier_cols =['age', 'MonthlyIncome']
  for col in outlier_cols:
      X_train, X_test = handle_outliers(X_train, X_test, col)

  monthlyIncomeImputeVal = X_train['MonthlyIncome'].mean()
  numDependentsImputeVal = X_train['NumberOfDependents'].mode()[0]
  X_train.fillna(value = {'MonthlyIncome': monthlyIncomeImputeVal, 'NumberOfDependents': numDependentsImputeVal}, inplace = True)
  X_test.fillna(value = {'MonthlyIncome': monthlyIncomeImputeVal, 'NumberOfDependents': numDependentsImputeVal}, inplace = True)

  st.write('# Heat Map')
  fig, axes = plt.subplots(figsize = (30,30))
  sns.heatmap(X_train.corr(), annot=True, linewidth = 0.5, center = 0, ax = axes)
  st.pyplot(fig)

  st.write('# WOE and IV')
  clf = WOE()
  X_train_transformed = clf.fit_transform(X_train, y_train.T.squeeze())
  X_test_transformed = clf.transform(X_test)
  st.write(clf.iv_df)
  selected_features_df = clf.iv_df[clf.iv_df['Information_Value'] > 0.15]

  st.write('# Custom Logisitic Regression Classification Report')
  st.write('## Using Normal data')
  lg_custom_clf = train_using_custom_logistic(X_train, y_train)
  y_pred = lg_custom_clf.predict(X_test)
#   https://discuss.streamlit.io/t/classification-score-visualization/48637/3
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

#   st.write('## Using WOE transformed data')
#   lg_custom_clf = train_using_custom_logistic(X_train_transformed, y_train)
#   y_pred = lg_custom_clf.predict(X_test_transformed)
#   st.table(st.dataframe(pd.DataFrame(st.write(classification_report(y_test, y_pred, output_dict=True)))))

  st.write('# Logisitic Regression(sklearn) Classification Report')
  st.write('## Using Normal data')
  st.write('### Unbalanced Model')
  lg_clf = train_using_logistic(X_train, y_train)
  y_pred = lg_clf.predict(X_test)
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

  st.write('### Balanced Model')
  lg_clf = train_using_logistic(X_train, y_train, isBalanced = True)
  y_pred = lg_clf.predict(X_test)
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

  st.write('## Using Transformed data')
  st.write('### Unbalanced Model')
  lg_clf = train_using_logistic(X_train_transformed, y_train, isBalanced = False)
  y_pred = lg_clf.predict(X_test_transformed)
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

  st.write('### Unbalanced Model using only few features with IV > 0.15')
  X_trian_sel_features_trans = X_train_transformed[selected_features_df['Variable_Name']]
  X_test_sel_features_trans = X_test_transformed[selected_features_df['Variable_Name']]
  lg_clf = train_using_logistic(X_trian_sel_features_trans, y_train, isBalanced = False)
  y_pred = lg_clf.predict(X_test_sel_features_trans)
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

  st.write('### Balanced Model')
  lg_clf = train_using_logistic(X_train_transformed, y_train, isBalanced = True)
  y_pred = lg_clf.predict(X_test_transformed)
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
  
  st.write('### Balanced Model using only few features with IV > 0.15')
  X_trian_sel_features_trans = X_train_transformed[selected_features_df['Variable_Name']]
  X_test_sel_features_trans = X_test_transformed[selected_features_df['Variable_Name']]
  lg_clf = train_using_logistic(X_trian_sel_features_trans, y_train, isBalanced = True)
  y_pred = lg_clf.predict(X_test_sel_features_trans)
  st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

main()
