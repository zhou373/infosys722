# -*- coding: utf-8 -*-
# 01-BU

# 02-DU
# Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/nillljiang/Desktop/uoa/s2/722/I3/cardio.xlsx')
# df = pd.read_csv('/Users/nillljiang/Desktop/uoa/s2/722/I3/cardio_train_origin.csv', delimiter=';')

# Explore Data
print(df.head())

df.info()

df.isna().sum()
# missing value columns: height (2/60078) and mock 55%(33109 / 60078)


df_desc = round(df[['age','height','weight','ap_hi','ap_lo', 'cholesterol', 'gluc']].describe(), 2)

df_desc_t = df_desc.transpose()

print(df['gender'].value_counts())
print(df['smoke'].value_counts())
print(df['alco'].value_counts())
print(df['active'].value_counts())
print(df['mock'].value_counts())
print(df['cardio'].value_counts())


df_corr = df[['age','height','weight','ap_hi','ap_lo', 'cholesterol', 'gluc', 'cardio']].corr()

round(df[['age', 'height', 'weight', 'ap_hi','ap_lo', 'cholesterol', 'gluc', 'cardio']].groupby('cardio').agg('mean').transpose(), 2)

round(df[['age', 'height', 'weight', 'ap_hi','ap_lo', 'cholesterol', 'gluc', 'cardio']].groupby('cardio').describe().transpose(), 2)



df['cardio_label'] = df['cardio'].map({0: 'Negative', 1: 'Positive'})

# visualization
import seaborn as sns

# for numerical data
for c in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
    sns.boxplot(data=df, x=c, y = 'cardio_label')
    sns.displot(data=df, x=c, hue='cardio_label', kind="kde")
    plt.show()

# for categorical data
for c in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
    sns.barplot(data=df[[c, 'cardio_label']].groupby(['cardio_label', c]).size().reset_index(name = 'counts'),x = c, y='counts', hue='cardio_label')
    plt.show()


# 3.1 Data Selection
# drop the column with high percentage of missing value
df.pop('mock')


# 3.2 Cleaning the data
# 3.2.1 Handling missing values
# First, we deal with missing value
# replace missing value with average value
df['height'].fillna(df['height'].mean(), inplace = True)

# 3.2.2 Handling outliers and extremes
# Through the box plots, we can see that there are some outliers in age, height, weight, 
# ap_hi, ap_lo
# use the IQR method to find the lower fence and upper fence
def get_iqr_fence(df_array):
    gap = np.percentile(list(df_array), [25, 75])
    iqr = gap[1] - gap[0]
    lower_fence = gap[0] - 1.5 * iqr
    upper_fence = gap[1] + 1.5 * iqr
    return lower_fence, upper_fence

for c in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
    print(c, get_iqr_fence(df[c]))

# the range make sense
# then filter the outliers
df_filter = df.copy(deep = True)
for c in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
    lower_fence, upper_fence = get_iqr_fence(df_filter[c])
    df_filter = df_filter[(df_filter[c] >= lower_fence) & (df_filter[c] <= upper_fence)]
    df_filter.reset_index(inplace = True, drop = True)

# plot again
# for prepocessing numerical data
for c in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
    sns.boxplot(data=df_filter, x=c, y = 'cardio_label')
    sns.displot(data=df_filter, x=c, hue='cardio_label', kind="kde")
    plt.show()

# 3.3 Constructing/Deriving a New Feature
# age constructing
df_filter['age'] = df_filter['age'] / 365

# age categorize
def get_age_category(x):
    if x < 25:
        return 'Young'
    elif x < 35:
        return 'Matured_Young'
    elif x < 55:
        return 'Middle_Age'
    elif x >= 55:
        return 'Old_Age'
    else:
        return 'None'
    
df_filter['age_refeature'] = df_filter['age'].apply(lambda x: get_age_category(x))
print(df_filter['age_refeature'].value_counts())

# reclassify gender
# then convert gender: 0: women, 1: man
df_filter['gender'] = df_filter['gender'].map({'women': 1, 'men': 2})


# construct new feature BMI
df_filter['bmi'] = df_filter['weight'] * 10000 / (df_filter['height'] * df_filter['height'])

# construct new feature ap_gap and ap_mul
print(df_filter[['ap_hi', 'ap_lo']].corr())
print(round(df_filter[['ap_hi','ap_lo', 'cardio']].groupby('cardio').describe().transpose(), 2))
df_filter['ap_gap'] = df_filter['ap_hi'] - df_filter['ap_lo']
df_filter['ap_mul'] = df_filter['ap_hi'] * df_filter['ap_lo']
print(round(df_filter[['ap_gap', 'ap_mul', 'cardio']].groupby('cardio').describe().transpose(), 2))
for c in ['ap_gap', 'ap_mul']:
    sns.boxplot(data=df_filter, x=c, y = 'cardio_label')
    sns.displot(data=df_filter, x=c, hue='cardio_label', kind="kde")
    plt.show()

# consttruct new feature bmi_aphi
print(df_filter[['bmi', 'ap_hi']].corr())
print(round(df_filter[['bmi', 'ap_hi', 'cardio']].groupby('cardio').describe().transpose(), 2))
df_filter['bmi_aphi'] = (df_filter['bmi'] - df_filter['ap_hi']) ** 2
print(round(df_filter[['bmi_aphi', 'cardio']].groupby('cardio').describe().transpose(), 2))
for c in ['bmi_aphi']:
    sns.boxplot(data=df_filter, x=c, y = 'cardio_label')
    sns.displot(data=df_filter, x=c, hue='cardio_label', kind="kde")
    plt.show()




print(df_filter.isna().sum())
# high percentage of missing value, and no significant difference between gender. drop gender.
#df_filter['gender'].fillna(3, inplace = True)
#df_filter.pop('gender')

# 3.4Data Integration
# merge dataset
df_creff = pd.read_excel('/Users/leslie_pun/Documents/temp/creff.xlsx')
df_filter = df_filter.merge(df_creff, left_on = 'id', right_on = 'id', how = 'inner')
df_filter.head()

# deal with outliers
lower_fence, upper_fence = get_iqr_fence(df_filter['creff'])
df_filter = df_filter[(df_filter['creff'] >= lower_fence) & (df_filter['creff'] <= upper_fence)]
df_filter.reset_index(inplace = True, drop = True)


# 4.Data Transformation
# 4.1Data Reduction
from sklearn.ensemble import ExtraTreesClassifier

col_raw = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'creff', 'age_refeature', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender', 'ap_gap', 'ap_mul', 'bmi_aphi']
clf = ExtraTreesClassifier(n_estimators=50)
df_filter_dummy = pd.get_dummies(df_filter[col_raw])
clf = clf.fit(df_filter_dummy, df_filter['cardio'])
print('Feature Importance')
df_feature_importance = pd.DataFrame([{'col': y,'importance': x} for x, y in zip(clf.feature_importances_, df_filter_dummy.columns)])
df_feature_importance.sort_values(by = 'importance', ascending = False, inplace = True)
print(df_feature_importance)             
feature_cols = ['age', 'ap_hi', 'creff', 'bmi', 'weight', 'height', 'ap_lo', 'cholesterol', 'gluc', 'active', 'smoke', 'alco', 'gender', 'ap_gap', 'ap_mul', 'bmi_aphi']

# 4.2 Data Projection
# standardization
standardization_col = ['age', 'weight', 'bmi', 'creff']
from sklearn import preprocessing
df_filter[standardization_col] = preprocessing.StandardScaler().fit(df_filter[standardization_col]).transform(df_filter_dummy[standardization_col])
for c in standardization_col:
    sns.boxplot(data=df_filter, x=c, y = 'cardio_label')
    sns.displot(data=df_filter, x=c, hue='cardio_label', kind="kde")
    plt.show()


# Boost the imbalance of target label
# then deal with the imbalance of target 
# use the sklearn smote methods
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)

x_res, y_res = sm.fit_resample(df_filter[feature_cols], df_filter['cardio'])

df_res = x_res
df_res['cardio'] = y_res

df_res['cardio'].value_counts()
# resample done!!!

# 6. Data-Mining Algorithms Selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_res[feature_cols], df_res['cardio'], test_size = 0.1)

# 6.1.1 Data Mining Objective: CART algorithm
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
clf1 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 5)
clf1 = clf1.fit(X_train, y_train)
print('recall: %.2f'%recall_score(y_test, clf1.predict(X_test)), 
      'accuracy: %.2f'%accuracy_score(y_test, clf1.predict(X_test)),
      'f1: %.2f'%f1_score(y_test, clf1.predict(X_test)))
print(tree.export_text(clf1, feature_names=feature_cols))
#tree.plot_tree(clf)


# 6.1.2 Data Mining Objective: ID3 algorithm
clf2 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
clf2 = clf2.fit(X_train, y_train)
print('recall: %.2f'%recall_score(y_test, clf2.predict(X_test)), 
      'accuracy: %.2f'%accuracy_score(y_test, clf2.predict(X_test)),
      'f1: %.2f'%f1_score(y_test, clf2.predict(X_test)))
print(tree.export_text(clf2, feature_names=feature_cols))

# 6.1.3 Data Mining Objective: Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression()
clf3 = clf3.fit(X_train, y_train)
print('recall: %.2f'%recall_score(y_test, clf3.predict(X_test)), 
      'accuracy: %.2f'%accuracy_score(y_test, clf3.predict(X_test)),
      'f1: %.2f'%f1_score(y_test, clf3.predict(X_test)))

# 6.1.4 Data Mining Objective: Random Forest
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(n_estimators=10)
clf4 = clf4.fit(X_train, y_train)
print('recall: %.2f'%recall_score(y_test, clf4.predict(X_test)), 
      'accuracy: %.2f'%accuracy_score(y_test, clf4.predict(X_test)),
      'f1: %.2f'%f1_score(y_test, clf4.predict(X_test)))

# 6.1.5 Data Mining Objective: XGBoost
from xgboost import XGBClassifier
clf5 = XGBClassifier(n_estimators = 18, max_depth = 4)
clf5.fit(X_train, y_train)
print('recall: %.2f'%recall_score(y_test, clf5.predict(X_test)), 
      'accuracy: %.2f'%accuracy_score(y_test, clf5.predict(X_test)),
      'f1: %.2f'%f1_score(y_test, clf5.predict(X_test)))

# 6.3 Build/Select Model with Algorithm/Model Parameters


# 7.Data Mining
# 7.1 Creating Logical Tests
#for f in range(30):
#    for i in range(10, 26):
#        for j in range(3, 15):
#            X_train, X_test, y_train, y_test = train_test_split(df_res[feature_cols], df_res['cardio'], random_state=42, test_size = 0.05 + f*0.01)
        
            #clf2 = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf=20 * (i + 1), max_depth = j)
            #clf2 = clf2.fit(X_train, y_train)
#            clf5 = XGBClassifier(n_estimators = i, max_depth = j)
#            clf5.fit(X_train, y_train)
            #print(tree.export_text(clf2, feature_names=feature_cols))
#            print(f, i, j,
#                'recall: %.2f'%recall_score(y_test, clf5.predict(X_test)), 
#                  'accuracy: %.2f'%accuracy_score(y_test, clf5.predict(X_test)),
#                  'f1: %.2f'%f1_score(y_test, clf5.predict(X_test)),
#                  'auc: %.2f'%roc_auc_score(y_test, clf5.predict(X_test)))
# n_estimators = 18, max_depth = 4
# recall 0.7 accuracy 0.73 f1 0.72 auc 0.73

# 
clf_pick = XGBClassifier(n_estimators = 18, max_depth = 4)
clf_pick.fit(X_train, y_train)
print('recall: %.2f'%recall_score(y_test, clf_pick.predict(X_test)), 
      'accuracy: %.2f'%accuracy_score(y_test, clf_pick.predict(X_test)),
      'f1: %.2f'%f1_score(y_test, clf_pick.predict(X_test)),
      'auc: %.2f'%roc_auc_score(y_test, clf_pick.predict(X_test)))

plt.barh(feature_cols, clf_pick.feature_importances_)
plt.show()

clf_pick.get_booster().dump_model('xgb_model.txt', with_stats=True)
# read the contents of the file

print('auc score: %.4f'%roc_auc_score(y_test, clf2.predict(X_test)))

if __name__ == '__main__':
    print('done')