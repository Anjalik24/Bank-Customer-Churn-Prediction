import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
warnings.filterwarnings('ignore')
df=pd.read_csv("Bank Customer Churn Prediction.csv")
df
df.info()
df.describe().round(2)
df.drop(['customer_id'],axis=1,inplace=True)
df
df.isnull().sum()
sns.histplot(
    data=df,
    x="active_member",
    hue="churn")
    sns.set(style="darkgrid")

sns.distplot( a=df["credit_score"],hist=True, kde=False, rug=False )
sns.histplot(
    data=df,
    x="age",
    y="balance",
    hue="churn",
)
plt.show()
df.country.unique()
df['country'] = df['country'].map({'France': 0, 'Spain' : 1,'Germany':2})
df['gender'] = df['gender'].map({'Male': 0, 'Female' : 1})
df['balance']=df['balance'].astype(int)
df['estimated_salary']=df['estimated_salary'].astype(int)
df
correlation_matrix = df.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
sns.kdeplot(
    data=df,
    x="balance",
    y="country",
    cmap="viridis")
sns.scatterplot(
    data=df,
    x="balance",
    y="country",
    hue="churn" 
)
sns.boxplot(
    x="products_number",
    y="balance",
    showmeans=True,
    data=df
)
sns.histplot(
    data=df,
    x="country",
    hue="churn")
df['gender'] = df['gender'].astype(str)
df['churn'] = df['churn'].astype(str)
sns.countplot(x="churn",hue="gender", data=df)
plt.show()
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15,20))
axs=axs.flat
for i in range(len(df.columns)-1):
    sns.histplot(data=df, x=df.columns[i],hue="churn",ax=axs[i])
df
x=df.drop(columns='churn') 
y=df['churn']
x.shape
y.shape
x
y
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.20)
x_train.shape
y_train.shape
print(x_train.shape , x_test.shape , y_train.shape , y_test.shape)
sc= StandardScaler()
x_train_rescaled = sc.fit_transform(x_train)
x_test_rescaled = sc.transform(x_test)
x_train_rescaled
x_test_rescaled
classifier = RandomForestClassifier( n_estimators=100,criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred)*100)
