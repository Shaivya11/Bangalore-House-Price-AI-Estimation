import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) ==2:
        return ((float(tokens[0])+float(tokens[1]))/2)
    try:
        return float(x)
    except:
        return None
    
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
                
    return df.drop(exclude_indices,axis='index')



def find_best_model(X, y):
    models={
            'linear_regression':{
                "model": LinearRegression(),
                "params":{
                    "fit_intercept":[True, False]}
                    },
            'lasso':{
                "model": Lasso(),
                "params":{
                    "alpha": [1,2],
                    "selection":['random', "cyclic"]}
                    },
            'decision tree':{
                "model": DecisionTreeRegressor(),
                "params":{
                    "criterion":['mse', 'friedman_mse'],
                    "splitter": ['best', 'random']}
                    }
            }
    
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for model, mp in models.items():
        clf = GridSearchCV(mp["model"], mp['params'], cv=cv, return_train_score=False)
        clf.fit(X, y)
        scores.append({"model": model, "best_score": clf.best_score_, "best_param": clf.best_params_})
    
    return pd.DataFrame(scores, columns=["model", "best_score", "best_params"])


def predict_price(location,sqft, bath, bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >0:
        x[loc_index] = 1
    
    return linear_model.predict([x])[0]


df = pd.read_csv(r"C:\Users\mshai\Documents\house_price_ML_project\Bengaluru_House_Data.csv")
print(df.shape)
print(df.describe())
print(df.head())

df = df.drop(['area_type','society', 'balcony', 'availability'], axis="columns")
print(df.head())

df= df.dropna()

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

df2 = df.copy()

df2['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

df2['price_per_sqft'] = df2['price']*100000/df2['total_sqft']

print(len(df2['location'].unique()))

df2['location'] = df2['location'].apply(lambda x: x.strip())
location_stats = df2.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

location_stats_lass_than_10 = location_stats[location_stats <=10]
df2['location'] = df2['location'].apply(lambda x: 'other' if x in location_stats_lass_than_10 else x)

df2 = df2[~(df2['total_sqft']/df2['bhk']<300)]

print(df2["price_per_sqft"].describe())

df2 = df2[(df2["price_per_sqft"] > (df2["price_per_sqft"].mean() - df2["price_per_sqft"].std())) & (df2["price_per_sqft"] <= (df2["price_per_sqft"].mean() + df2["price_per_sqft"].std()))]

df3 = remove_bhk_outliers(df2)
print(df3.shape)

plt.hist(df3["price_per_sqft"], rwidth=0.8)
plt.xlabel("price_per_sqft")
plt.ylabel("count")
plt.show()

plt.hist(df3["bath"], rwidth=0.8)
plt.xlabel("bath")
plt.ylabel("count")
plt.show()

df3 = df3[df3['bath'] < df3["bhk"]+2]

df4 = df3.drop(['size', 'price_per_sqft'], axis='columns')

dummies = pd.get_dummies(df4["location"])

df5 = pd.concat([df4, dummies.drop(['other'], axis='columns')], axis="columns")
df5 = df5.drop(['location'], axis='columns')

X = df5.drop(["price"], axis="columns")
y = df5["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print(linear_model.score(X_test, y_test))


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores = cross_val_score(LinearRegression(), X, y,cv=cv)
print(scores)

best_scores = find_best_model(X, y)
print(best_scores)

print(predict_price("Whitefield", "2000", "5", "3"))

with open("bangalore_home_price_model.pickle", "wb") as f:
    pickle.dump(linear_model, f)
    
columns = {"data_columns":[col.lower() for col in X.columns]}
with open ("columns.json", "w") as file:
    file.write(json.dumps(columns))