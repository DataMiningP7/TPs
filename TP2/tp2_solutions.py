# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Ex 1.1
def read_dataset(filename):
    """ Reads the file at the given path that should contain one type and one
     text separated by a tab on each line, and returns pairs of type/text.

    Args:
        filename: a file path.
    Returns:
        a list of (type, text) tuples. Each type is either 0 or 1.
    """
    lines = []

    f = open(filename)

    for line in f.readlines():
        spam, text = line.split("\t", 1)
        text_type = 1 if spam == "spam" else 0
        lines.append( (text_type, text) )

    f.close()

    return lines


# Ex 1.2
def spams_count(texts):
    """ Returns the number of spams from a list of (type, text) tuples.

    Args:
        texts: a list of (type, text) tuples.
    Returns:
        an integer representing the number of spams.
    """
    # spams have the type 1 so we just sum all types to get their number
    return sum([t for (t, _) in texts])


# Ex 1.3
def transform_text(pairs):
    """ Transforms the pair data into a matrix X containing tf-idf values
     for the messages and a vector y containing 0s and 1s (for hams and spams
     respectively).
     Row i in X corresponds to the i-th element of y.

    Args:
        pairs: a list of (type, message) tuples.
    Returns:
        X: a sparse TF-IDF matrix where each row represents a message and each
        column represents a word.
        Y: a vector whose i-th element is 0 if the i-th message is a ham, else
        1.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    types, texts = zip(*pairs)

    X = tfidf.fit_transform(texts)
    # Convert the list to a Numpy array because some sklearn objects don't
    # accept lists.
    y = np.array(types)

    return X, y

# Ex 2
def ex2_kmeans(X, y):
    """ Applies the KMeans algorithm on X, y using K=10 and print the
    silhouette score of this model. X and y are returned by transform_text
    above.
    """
    model = KMeans(10).fit(X, y)
    print "Silhouette score: %f" % metrics.silhouette_score(X, model.labels_)

# Ex 3
def ex3_kmeans(X, y):
    """ Tries to find the best value for K when applying the KMeans algorithm
    on X, y. There are multiple ways to score a model but here we count what is
    the ratio of clusters with a negative Silhouette score and try to minimize
    it, for K from 2 to 20.

    Returns:
        best_k: the value of K that gives the best score.
        best_score: the score associated with best_k.
    """
    best_k = 1
    best_score = -1

    for k in range(2, 20+1):
        model = KMeans(k).fit(X, y)

        scores = metrics.silhouette_samples(X, model.labels_)
        negative_scores_count = len([x for x in scores if x < 0])
        model_score = negative_scores_count / float(len(scores))

        print "K=%d, score=%f" % (k, model_score)

        if model_score > best_score:
            best_score = model_score
            best_k = k

    # Unsurprisingly the best K is usually 2 because we have two classes of
    # messages: spams and hams.
    return best_k, best_score


# Ex 4
def ex4_agglomerative_clustering(X, y):
    """ This does the same thing as ex2_kmeans but with an agglomerative
    clustering and K=2.
    """
    # AgglomerativeClustering needs a non-spare matrix
    X = X.toarray()

    k = 2
    model = AgglomerativeClustering(k).fit(X, y)

    print "Silhouette score: %f" % metrics.silhouette_score(X, model.labels_)


# Ex 5
def ex5_knn(X, y):
    # Ex 5.1
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Ex 5.2
    k = 1
    model = KNeighborsClassifier(k).fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print "K=%d, score=%f" % (k, score)

    # Ex 5.3 (this is exactly like 5.2 but with k=3)
    k = 3
    model = KNeighborsClassifier(k).fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print "K=%d, score=%f" % (k, score)

    # Ex 5.4: Steps 5.2 and 5.3 are forbidden because we can't use the test set
    #         to choose our model. The solution is to use cross-validation.

    # Ex 5.5
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    best_k = 1
    best_score = 0

    # Remember than range(a, b, 2) yields a, a+2, a+4, ... . So starting at 1
    # with step 2 ensures we test only odd numbers.
    for k in range(1, 100+1, 2):

        # We compute the sum of all scores in order to get an average score
        score_sum = 0.0

        for train_idx, test_idx in skf.split(X_train, y_train):
            X_sub_train, X_sub_test = X[train_idx], X[test_idx]
            y_sub_train, y_sub_test = y[train_idx], y[test_idx]

            model = KNeighborsClassifier(k).fit(X_sub_train, y_sub_train)
            score = model.score(X_sub_test, y_sub_test)

            score_sum += score

        score_average = score_sum / n_splits

        print "K=%d, score=%f" % (k, score_average)
        if score_average > best_score:
            best_score = score_average
            best_k = k

    # Now that we know the best K, re-launch on the whole dataset
    knn = KNeighborsClassifier(best_k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)

    # e.g. K=1, score = ~0.94
    print "Best: K=%d, score=%f" % (best_k, score)

    return best_k, score

# Ex 6
# We simplified the code here to use the same parameters (and the same code!)
# for both the decision tree and the random forest.
def ex6(X, y):
    print "==> Decision Tree"
    tree_score = ex6_generic_tree(X, y, DecisionTreeClassifier)

    print "==> Random Forest"
    forest_score = ex6_generic_tree(X, y, RandomForestClassifier)

    if tree_score > forest_score:
        print "The decision tree is better"
    elif tree_score < forest_score:
        print "The random forest is better"
    else:
        print "The decision tree and the random forest perform the same"

def ex6_generic_tree(X, y, cls):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    # We test with two parameters here: min_samples_split and max_depth
    best_min_samples_split = 2
    best_max_depth = None
    best_score = 0

    # We test with max_depth set to 1, 2, ..., 20, 100, and None (= unlimited)
    # You may want to test with other values.
    max_depths = range(1, 20+1)
    max_depths.append(100)
    max_depths.append(None)

    # We test with min_samples_split set to 2, 3, ..., 20
    for min_samples_split in range(2, 20+1):

        for max_depth in max_depths:
            score_sum = 0.0

            for train_idx, test_idx in skf.split(X_train, y_train):
                X_sub_train, X_sub_test = X[train_idx], X[test_idx]
                y_sub_train, y_sub_test = y[train_idx], y[test_idx]

                # cls is either DecisionTreeClassifier or
                # RandomForestClassifier here.
                model = cls(min_samples_split=min_samples_split, max_depth=max_depth)

                model.fit(X_sub_train, y_sub_train)
                score = model.score(X_sub_test, y_sub_test)

                score_sum += score

            score_average = score_sum / n_splits

            print "min_samples_split=%d, max_depth=%s, score=%f" % \
                    (min_samples_split, max_depth, score_average)

            if score_average > best_score:
                best_score = score_average
                best_min_samples_split = min_samples_split
                best_max_depth = max_depth

    # re-launch on the whole dataset
    m = cls(min_samples_split=best_min_samples_split, max_depth=best_max_depth)
    m.fit(X_train, y_train)
    score = m.score(X_test, y_test)

    # The result might vary if you run the code multiple times.
    # e.g. min_samples_split = 8, max_depth = None, score = ~ 0.97
    print "Best min_samples_split=%d, max_depth=%s, score=%f" % \
            (best_min_samples_split, best_max_depth, score)

    return score
