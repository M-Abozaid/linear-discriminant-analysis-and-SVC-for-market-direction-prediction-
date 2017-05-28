import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def make_Xy():
    data = pd.read_csv('GBPUSD60.csv', names=['date', 'time', 'open', 'high',
                                              'low', 'close', 'quoteVolume'])
    a = data.date + ' ' + data.time
    data.set_index(pd.to_datetime(a), inplace=True)
    data.drop(['date', 'time'], axis=1, inplace=True)

    price = ['open', 'close', 'high', 'low']
    data['price'] = data[price].mean(axis=1)
    data['price_change'] = data.price.pct_change()
    data['spread'] = data.open - data.close

    data['next'] = data.price.shift(1)
    data['change'] = (data.next - data.price)
    data.drop(price, axis=1, inplace=True)

    pos = np.percentile(data.change.dropna(), 50 + 25)
    neg = np.percentile(data.change.dropna(), 50 - 25)

    def up_down(row):
        """returns if the next movement or up or down"""
        if row > pos:
            return 1
        elif row < neg:
            return -1
        else:
            return 0

    data['up_down'] = data.change.apply(up_down)
    # creating the SMAs
    data['sma_5'] = data['price'].rolling(5).mean()
    data['sma_10'] = data['price'].rolling(10).mean()

    data['mean_spread3'] = data['spread'].rolling(3).mean()
    data['mean_spread6'] = data['spread'].rolling(5).mean()

    data['mean_quote_volume3'] = data['quoteVolume'].rolling(3).mean()
    data['mean_quote_volume5'] = data['quoteVolume'].rolling(5).mean()

    data['pc_ch_5'] = data.sma_5.pct_change()
    data['pc_ch_10'] = data.sma_10.pct_change()

    data['sma5_ask_diff'] = (data.sma_5 - data.price)
    data['sma10_ask_diff'] = (data.sma_10 - data.price)

    data.dropna(inplace=True)
    X = data.drop(['change', 'up_down', 'next', 'price', 'sma_5', 'sma_10'],
                  axis=1)
    X = X.values
    y = data.up_down.values
    return X, y

X, y = make_Xy()

# splitting the data to train and test-sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# making a pipline
model = Pipeline([('ss', StandardScaler()),
                  ('lda', LinearDiscriminantAnalysis(n_components=4)),
                  ('svc', SVC())])

model.fit(X_train, y_train)
cvs = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
print('the mean is {} and the std is {}'.format(np.mean(cvs), np.std(cvs)))




# learning curves
train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                        X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1, 20),
                                                        cv=10, n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy',alpha=0.5)

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy',alpha=0.5)

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# validation curves


param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=model,
                X=X_train,
                y=y_train,
                param_name='svc__C',
                param_range=param_range,
                cv=10,
                n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')


plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')




param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=model,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
'''
0.892071148628
{'svc__gamma': 0.01, 'svc__C': 1000.0, 'svc__kernel': 'rbf'}
'''

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))