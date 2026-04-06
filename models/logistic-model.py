import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder, Normalizer, RobustScaler
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse


def train_lr_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_values: list[int],
) -> tuple[float, list[float], list[float]]:
    
    precision = []
    accuracy = []
    balanced_accuracy = []

    for c in C_values:
        lr = LogisticRegression(C=c, random_state=42, class_weight='balanced', l1_ratio=0.75, max_iter=1000, penalty='elasticnet', solver='saga', verbose=2)
        lr.fit(X_train, y_train)

        pred = lr.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0.0)
        precision.append(report["macro avg"]["precision"])
        accuracy.append(report["accuracy"])
        balanced_accuracy.append(balanced_accuracy_score(y_test, pred))

        
    optimal_c = C_values[np.argmax(precision)]
    
    print(f"\nOptimal c: {optimal_c}")
    print(f"\nOptimal c precision: {precision[np.argmax(precision)]}")
    print(f"\nOptimal c accuracy: {accuracy[np.argmax(precision)]}")
    print(f"Optimal c balanced accuracy: {balanced_accuracy[np.argmax(precision)]}")

    return optimal_c, precision, accuracy, precision[np.argmax(precision)], balanced_accuracy

def grid_lr(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_jobs: int,
):
    C_values = np.logspace(-3, 3, 20)
    param = {
        "penalty": ["elasticnet"],
        "C": C_values,
        "solver": ["saga"],
        "max_iter": [100, 200, 400, 800, 1000],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "class_weight": ["balanced"]
    }  
    model = LogisticRegression(verbose=1)
    model.fit(X_train, y_train)
    gs = GridSearchCV(model, param_grid=param, cv=5, verbose=2, n_jobs=n_jobs, scoring='average_precision')
    gs.fit(X_train, y_train)
    print(gs.best_estimator_)
    y_pred = gs.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
    print(report)
    print(report["macro avg"]["precision"])

def visualize_results(
    values: list[float],
    scores: list[list[float]], 
    optimal: float,    
) -> None:

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xscale('log')

    plt.plot(values, scores[0], label="precision")
    plt.plot(values, scores[1], label="accuracy")
    plt.plot(values, scores[2], label="balanced accuracy")
    plt.axvline(x=optimal, color='blue', linestyle='--', alpha=0.7,
                label=f'Optimal value={optimal}')

    plt.xlabel('C value')
    plt.ylabel('Average Precision')
    plt.title('Average Precision vs Hyperparameter')
    
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_bar(
    values,
    scores,
):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bar = plt.bar(values, scores)
    ax. bar_label(bar, padding=2)
    plt.xlabel("")
    plt.ylabel("Precision")
    plt.margins(y=0.1)
    plt.show()

def preprocessData(
        df,
        i
    ):
    features_initial = [
    "danceability", "energy", "acousticness", "instrumentalness", "valence", "tempo", "speechiness", 
    "track_popularity", "artist_popularity", "artist_followers", "track_duration_ms", "explicit", "album_total_tracks", "album_release_date", 
    "text_combined"
    ]
    X = df[features_initial]
    genres = df["genre_fixed"].unique()
    y = df["genre_fixed"]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(binary=True, stop_words="english")
    train_text_binary = vectorizer.fit_transform(X_train["text_combined"])
    test_text_binary = vectorizer.transform(X_test["text_combined"])


    X_train = X_train.drop("text_combined", axis=1)
    X_test = X_test.drop("text_combined", axis=1)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    X_train = sparse.hstack([X_train, train_text_binary])
    X_test = sparse.hstack([X_test, test_text_binary])

    scaler = MaxAbsScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_train, y_train, scaled_X_test, y_test

def showScore(
        X_train,
        y_train,
        X_test,
        y_test,
):
    C_values = np.logspace(-3, 3, 20)
    c, prec, acc, prec_score, bal_acc = train_lr_model(X_train, y_train, X_test, y_test, C_values)
    scores = [prec, acc, bal_acc]
    visualize_results(C_values, scores, c)

def gridSearch(
        X_train,
        y_train,
        X_test,
        y_test,
    ):
    grid_lr(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    df = pd.read_csv("../preprocess/output/spotify_preprocessed.csv")
    df = df.dropna()
    scaled_X_train, y_train, scaled_X_test, y_test = preprocessData(df, 0)

    # showScore(scaled_X_train, y_train, scaled_X_test, y_test)

    # grid_lr(scaled_X_train, y_train, scaled_X_test, y_test, 8)

