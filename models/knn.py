import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../preprocess/output/spotify_preprocessed.csv")

#each row is one track, which is my objects.

features = [
    'danceability', 'energy', 'acousticness', 
    'instrumentalness', 'valence', 'tempo', 
    'speechiness', 'track_popularity', 'track_duration_ms'
]

X = df[features].copy()
y = df['genre_fixed'].copy()

mask = X.notna().all(axis=1)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#hyperparameters: k, distance, weights

n = len(X_train)
sqrt_n = np.sqrt(n)

metrics = ['euclidean', 'manhattan']
weights = ['uniform', 'distance']

k_values = (np.arange(1, sqrt_n, 2).astype(int)).tolist()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for metric in metrics:
    for weight in weights:
        cv_scores = []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
            score = cross_val_score(knn, X_train_scaled, y_train, cv=cv,scoring='f1_macro', n_jobs=-1).mean()
            cv_scores.append(score)
            results.append({'k': k, 'metric': metric,'weights': weight, 'cv': score})

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['cv'].idxmax()]

plt.figure(figsize=(10, 6))

for metric in metrics:
    for weight in weights:
        subset = results_df[ (results_df['metric'] == metric) & (results_df['weights'] == weight) ]
        label = f"{metric}, {weight}"
        plt.plot(subset['k'], subset['cv'], marker='o', label=label)

plt.xlabel("k (Number of Neighbors)")
plt.ylabel("CV F1 Macro Score")
plt.title("KNN Hyperparameter Tuning")
plt.legend()
plt.grid(True)
plt.show()

best_knn = KNeighborsClassifier( n_neighbors=int(best_row['k']), metric=best_row['metric'], weights=best_row['weights'])

#final

best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
