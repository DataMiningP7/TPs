# -*- coding: UTF-8 -*-

# To add ../TP2 to the import path so that we can import functions from the
# previous TP:
#   import sys
#   sys.path.insert(0, "../TP2")

# for the Ex1
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# for the Ex3
from sklearn.cluster import AgglomerativeClustering
# for the Ex4.1
from sklearn.model_selection import train_test_split
# for the Ex4.2
from sklearn.neighbors import KNeighborsClassifier

# for the Ex4.5
from sklearn.model_selection import StratifiedKFold


# Ex1
def kmeans(X, k=10):
    """ Run a K-Means clustering on X.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
        k: the number of clusters we want (default: 10).
    Returns:
        A KMeans model trained on X.
    """
    model = KMeans(k).fit(X)
    return model

def evaluate_kmeans(X, model):
    """ Evaluate a K-Means model that has been trained on X using the
     Silhouette score.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
        model: the KMeans model trained on X.
    Returns:
        A double that corresponds to the Silhouette score of the model.
    """
    return silhouette_score(X, model.labels_)


# Ex2
def try_kmeans(X):
    """ Run the K-Means algorithm on X with different values of K, and return
     the one that gives the best score.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
    """
    best_k = 1
    best_score = -1

    for k in range(2, 20+1):
        model = KMeans(k)
        model.fit(X)
        labels = model.predict(X)
        score = silhouette_score(model.transform(X), labels)

        print(k, "->", score)
        if score > best_score:
            best_k = k
            best_score = score

    print("The best K is", best_k)
    return best_k


# Ex3
def agglomerative_clustering(X, k=10):
    """ Run an agglomerative clustering on X.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
        k: the number of clusters we want (default: 10).
    Returns:
        An AgglomerativeClustering model trained on X.
    """
    model = AgglomerativeClustering(k).fit(X)

    # Note all the other functions are the same except we use
    # 'AgglomerativeClustering' instead of 'KMeans'.
    return model


# Ex4.1
def get_train_test_sets(X, y):
    """ Split X and y into a train and a test sets.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
        y: a binary vector where the i-th value indicates whether the i-th is a
           spam or a ham.
    Returns:
        X_train: train subset of X
        X_test: test subset of X
        y_train: train subset of y
        y_test: test subset of y
    """
    return train_test_split(X, y)

# Ex4.2, 4.3, 4.4
def test_kneighbors_k1_3(X, y):
    """ Test the KNeighborsClassifier on X and y with k=1 and k=3 and return
     the best value.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
        y: a binary vector where the i-th value indicates whether the i-th is a
           spam or a ham.
    Returns:
        An int indicating the best value for k.
    """
    X_train, X_test, y_train, y_test = get_train_test_sets(X, y)

    knn_k1 = KNeighborsClassifier(1)
    knn_k1.fit(X_train, y_train)

    # score = accuracy = % good observations
    score_k1 = knn_k1.score1(X_test, y_test)
    print("KNeighbors with k=1:", score_k1)

    # Ex4.3
    knn_k3 = KNeighborsClassifier(3)
    knn_k3.fit(X_train, y_train)

    # score = accuracy = % good observations
    score_k3 = knn_k3.score3(X_test, y_test)
    print("KNeighbors with k=3:", score_k3)

    if score_k1 > score_k3:
        print("The best K is 1")
        return 1

    print("The best K is 3")
    return 3

# Ex4.4
# We can't modify our algorithm based on its results on our test set, otherwise
# we'll overfit. We need to use cross-validation to find the best K.
# See the 6th lecture, slides 29 and 44-45.

# Ex4.5
def find_best_k_for_kneighbors(X, y):
    """ Test the KNeighborsClassifier on X and y with various values of k and
     return the best one.

    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text() from the TP2.
        y: a binary vector where the i-th value indicates whether the i-th is a
           spam or a ham.
    Returns:
        An int indicating the best value for k.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    n_splits = 5  # arbitrary
    skf = StratifiedKFold(n_splits=n_splits)

    best_k = 1
    best_score = 0

    # range(1, 100, 2) ~ [1, 3, 5, ..., 99]
    # you could reduce this to e.g. 10 if you don't want to wait one day for
    # your function to terminate.
    for k in range(1, 100, 2):
        score_sum = 0.0

        for train_idx, test_idx in skf.split(X_train, y_train):
            X_subtrain, X_subtest = X[train_idx], X[test_idx]
            y_subtrain, y_subtest = y[train_idx], y[test_idx]

            knn = KNeighborsClassifier(k)
            knn.fit(X_subtrain, y_subtrain)
            score = knn.score(X_subtest, y_subtest)

            score_sum += score

        # get the average across splits
        score_sum /= n_splits

        print("With K", k, "the score is", score)
        if score_sum > best_score:
            best_score = score_sum
            best_k = k

    # re-launch on the whole dataset
    knn = KNeighborsClassifier(best_k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)

    # for example "Best K is 1, with a score of 0.939..."
    print("Best K is", best_k, "with a score of", score)

    return best_k


# Ex5
# The solution is roughly the same as the Ex4.4 but with DecisionTreeClassifier
# then RandomForestClassifier, and different parameters. But the idea is the
# same: try the algorithm with various configurations then take the one that
# gives the best score.
