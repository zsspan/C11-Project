import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

df = pd.read_csv("../preprocess/output/spotify_preprocessed.csv")

NUM_CLASS = len(df['genre_fixed'].unique())

str_features = [
    'track_name', 'artist_name', 'genre', 'album_id', 'album_name', 'album_type'
]

non_str_features = [ 
    'track_number', 'track_popularity', 'track_duration_ms', 
    'explicit', 'artist_popularity', 'artist_followers',
    'album_release_date', 'album_total_tracks', 
    'danceability', 'energy', 'acousticness', 'instrumentalness', 'valence',
    'tempo', 'speechiness'
]

features = str_features + non_str_features

# encode non-numeric values to numeric
for ft in str_features:
    le = LabelEncoder()
    le.fit(df[ft])
    df[ft] = le.transform(df[ft])

le = LabelEncoder()
le.fit(df['genre_fixed'])
df['genre_fixed'] = le.transform(df['genre_fixed'])

# split test & train set with 20-80 split
X_train, X_test, y_train, y_test = train_test_split(
                                    df[features], df['genre_fixed'], test_size=.2, 
                                    random_state=1)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

base_params = {
    'max_depth': 6, # default
    'learning_rate': 0.1, # try values
    'subsample': 0.8, # <1 to prevent overfit: randomly sample 0.8 of the training data prior to growing trees. >= 0.5 for good results.
    'colsample_bytree': 0.8, # subsample ratio of columns when constructing each tree
    'seed': 1, 
    'verbosity': 0, # 0 (silent), 1 (warning), 2 (info), and 3 (debug)
    'num_class': NUM_CLASS
}

def focal_loss_multiclass(y_pred, dtrain, gamma):
    y_true = dtrain.get_label().astype(int)
    n_samples = len(y_true)

    # reshape predictions to (n_samples, num_class)
    y_pred = y_pred.reshape(n_samples, -1)
    # softmax
    y_pred = np.exp(y_pred)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    # one-hot encode y_true, 1 if true
    y_onehot = np.zeros_like(y_pred)
    y_onehot[np.arange(n_samples), y_true] = 1
    # focal weight per sample, probability of true/correct class
    pt = (y_pred * y_onehot).sum(axis=1, keepdims=True)
    focal_weight = (1 - pt) ** gamma
    
    # gradient (first deriv) and hessian (second deriv)
    # first deriv is (1-pt)^gamma (pt-y)
    # second deriv is (1-pt)^gamma 2pt (1-pt)
    grad = focal_weight * (y_pred - y_onehot)
    hess = focal_weight * y_pred * (1 - y_pred) * 2
    return grad.flatten(), hess.flatten()

def get_predictions(model, dtest, num_class, objective):
    raw = model.predict(dtest)

    if objective == 'softmax':
        y_pred = raw.astype(int)
        y_prob = None
    else:
        y_prob = raw if raw.ndim == 2 else raw.reshape(-1, num_class)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        y_pred = np.argmax(y_prob, axis=1)

    return y_prob, y_pred

def evaluate(model, dtest, y_test, name, num_class, objective):
    y_prob, y_pred = get_predictions(model, dtest, num_class, objective)

    metrics = {'Model': name}
    metrics['Accuracy'] = round(accuracy_score(y_test, y_pred), 4)
    metrics['Balanced Acc'] = round(balanced_accuracy_score(y_test, y_pred), 4)
    metrics['F1 Macro'] = round(f1_score(y_test, y_pred, average='macro'), 4)
    metrics['F1 Weighted'] = round(f1_score(y_test, y_pred, average='weighted'), 4)

    if y_prob is not None:
        metrics['AUC-ROC'] = round(roc_auc_score(y_test, y_prob, multi_class='ovr'), 4)
    else:
        metrics['AUC-ROC'] = 'N/A'
    return metrics

# Model 1 : uses softprob as objective function (cross-entropy, outputs probabilities)
model_softprob = xgb.train(
    {**base_params, 'objective': 'multi:softprob'},
    dtrain, num_boost_round=100,
    evals=[(dtest, 'test')], verbose_eval=False
)

params_softprob = {**base_params, 'objective': 'multi:softprob'}

# Model 2 : uses softmax as objective function (outputs class label directly)
model_softmax = xgb.train(
    {**base_params, 'objective': 'multi:softmax'},
    dtrain, num_boost_round=100,
    evals=[(dtest, 'test')], verbose_eval=False
)

# Model 3 : uses custom focal loss (for multiclass) as objective function

def focal_wrapper(y_pred, dtrain):
    return focal_loss_multiclass(y_pred, dtrain, gamma=2.0)
focal_params = {
    **base_params,
    'num_output_group': NUM_CLASS,
}

model_focal = xgb.train(
    {
        **base_params,
        'objective': 'multi:softprob', # output shape
        'num_output_group': NUM_CLASS,
        'disable_default_eval_metric': 1,
    },
    dtrain, num_boost_round=100,
    obj=focal_wrapper,
    evals=[(dtest, 'test')], verbose_eval=False
)

results = [
    evaluate(model_softprob, dtest, y_test, 'Softprob', NUM_CLASS, 'softprob'),
    evaluate(model_softmax, dtest, y_test, 'Softmax', NUM_CLASS, 'softmax'),
    evaluate(model_focal, dtest, y_test, 'Focal Loss', NUM_CLASS, 'focal'),
]

print(results)

result_df = pd.DataFrame(results)

result_df['AUC-ROC'] = pd.to_numeric(result_df['AUC-ROC'], errors='coerce') # handle N/A
result_df.set_index('Model', inplace=True)

# bar chart
result_df.plot(kind='bar', figsize=(10, 6))

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


# auc-roc only
plt.figure(figsize=(6, 4))

auc_df = result_df[['AUC-ROC']].dropna()
plt.bar(auc_df.index, auc_df['AUC-ROC'])

plt.title("AUC-ROC Comparison")
plt.ylabel("AUC-ROC")
plt.tight_layout()
plt.show()
