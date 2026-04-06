import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score

df = pd.read_csv("../preprocess/output/spotify_preprocessed.csv")

#each row is one track, which is my objects.

features = [
    'danceability', 'energy', 'acousticness',
    'instrumentalness', 'valence', 'tempo',
    'speechiness', 'track_popularity', 'track_duration_ms',
    'artist_popularity', 'artist_followers'
]

X = df[features].copy()
y = df['genre_fixed'].copy()

mask = X.notna().all(axis=1)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#hyperparamters: num of trees , max depth, min samples leaf, max features

n = len(X_train)

n_estimators_values = [50, 100, 200, 300]
max_depth_values = [5, 10, 20, None]
min_leaf_values = [1, 5, 10, 20]
max_features_values = ['sqrt', 'log2', 0.5]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for n_est in n_estimators_values: #takes 4-5 minutes
    for depth in max_depth_values:
        for min_leaf in min_leaf_values:
            for max_feat in max_features_values:
                rf = RandomForestClassifier(n_estimators=n_est,max_depth=depth,min_samples_leaf=min_leaf,max_features=max_feat,random_state=42, n_jobs=-1)
                score = cross_val_score(rf, X_train, y_train,cv=cv, scoring='f1_macro', n_jobs=-1).mean()
                results.append({'n_estimators': n_est,'max_depth': depth,'min_samples_leaf': min_leaf,'max_features': max_feat,'cv': score})

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['cv'].idxmax()]

print("Best Hyperparameters:\n")
print(best_row)

best_n_est = int(best_row['n_estimators'])
best_depth = best_row['max_depth']
if best_depth != best_depth:  #NaN checker
    best_depth = None
else:
    best_depth = int(best_depth)
best_leaf = best_row['min_samples_leaf']
best_feat_val = best_row['max_features']

#train

final_rf = RandomForestClassifier(n_estimators=int(best_row['n_estimators']),max_depth=best_depth,min_samples_leaf=int(best_row['min_samples_leaf']),max_features=best_row['max_features'],random_state=42,n_jobs=-1)

final_rf.fit(X_train, y_train)

#evaluate
y_pred = final_rf.predict(X_test)
print(classification_report(y_test, y_pred))
