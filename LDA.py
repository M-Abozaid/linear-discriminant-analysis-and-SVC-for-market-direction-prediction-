
# coding: utf-8

# In[26]:

import pandas as pd 
import numpy as np
import datetime
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')


# In[12]:

def make_Xy(coin):
    data = pd.read_csv('{}.csv'.format(coin.upper()))
    data['date'] = data.date.apply(datetime.datetime.fromtimestamp)
    data.set_index('date', inplace=True)

    price = ['open','close','high','low']
    data['price'] = data[price].mean(axis=1)
    data['price_change'] = data.price.pct_change()
    data['spread']  = data.open - data.close

    data['next'] = data.price.shift(1)
    data['change'] = (data.next - data.price)
    data.drop(price,axis=1,inplace=True)

    pos = np.percentile(data.change.dropna(),50 + 25)
    neg = np.percentile(data.change.dropna(),50- 25)
    
    def up_down(row):
        """returns if the next movement or up or down"""
        if row > pos:
            return 1
        elif row < neg:
            return -1
        else:
            return 0   

    data['up_down'] = data.change.apply(up_down)
    ### creating the SMAs
    data['sma_5'] = data['price'].rolling(5).mean()
    data['sma_10'] = data['price'].rolling(10).mean()

    data['mean_spread3'] = data['spread'].rolling(3).mean()
    data['mean_spread6'] = data['spread'].rolling(5).mean()


    data['mean_quote_volume3'] = data['quoteVolume'].rolling(3).mean()
    data['mean_quote_volume5'] = data['quoteVolume'].rolling(5).mean()

    data['pc_ch_5'] = data.sma_5.pct_change()
    data['pc_ch_10'] = data.sma_10.pct_change()

    data['sma5_ask_diff'] = (data.sma_5  - data.price)
    data['sma10_ask_diff'] = (data.sma_10  - data.price)

    data.dropna(inplace=True)


    X = data.drop(['change','up_down','next','price','sma_5','sma_10'],axis=1)
    X = X.values
    y = data.up_down.values
    return X,y 


# In[16]:

X , y = make_Xy('xem')


# In[18]:

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[19]:

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)


# In[69]:

from sklearn.lda import LDA


# In[78]:

lda=LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y_train)
X_test_lda = lda.transform(X_test_std)
lda.n_components


# In[79]:

plt.figure(figsize=(12,7))

colors = ['indianred', 'darkblue', 'darkgreen']
for l, c in zip(np.unique(y_train), colors):
     plt.scatter(X_train_lda[y_train==l, 0],
         X_train_lda[y_train==l, 1],
         c=c, label=l,alpha=0.3)
        
# for l, c in zip(np.unique(y_test), colors):
#      plt.scatter(X_test_lda[y_test==l, 0],
#          X_test_lda[y_test==l, 1],
#          c=c, label=l,alpha=0.1,marker='X')

plt.xlim(-15,15)
plt.ylim(-2,10)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()


# In[80]:

lda.explained_variance_ratio_


# In[81]:

model = RandomForestClassifier(n_estimators=1000).fit(X_train_lda,y_train)


# In[84]:

y_pred = model.predict(X_test_lda)
confusion_matrix(y_test, y_pred)


# In[85]:

accuracy_score(y_test,y_pred)


# In[86]:

import xgboost


# In[87]:

xg = xgboost.XGBClassifier(n_estimators=1000)


# In[88]:

xg.fit(X_train_lda,y_train)


# In[89]:

y_pred = xg.predict(X_test_lda)
confusion_matrix(y_test, y_pred)


# In[90]:

accuracy_score(y_test,y_pred)


# In[91]:

xg = xgboost.XGBClassifier(n_estimators=1000)
xg.fit(X_train,y_train)
y_pred = xg.predict(X_train)
confusion_matrix(y_test, y_pred)


# In[ ]:

from sklearn.svm import SVC
svc = SVC()

