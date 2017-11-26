# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind

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
        label, text = line.split("\t", 1)
        text_type = 1 if label == "spam" else 0
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
    spams_count = 0
    for t, _ in texts:
        # t=1 if it's a spam, 0 if not
        spams_count += t

    return spams_count


# Ex 1.3
# See http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# for the parameters
def make_tfidf(min_df=0.0005):
    return TfidfVectorizer(stop_words="english", min_df=min_df)


# Ex 1.4
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
    tfidf = make_tfidf()

    types = []
    texts = []
    for label, text in pairs:
        types.append(label)
        texts.append(text)

    X = tfidf.fit_transform(texts)
    # Convert the list to a Numpy array because some sklearn objects don't
    # accept lists.
    y = np.array(types)

    return X, y


# Ex 1.5
def test_word_means(X, y, word_index):
    """ Performs a two-means t-test on the tf-idf values of a given word
     represented by its index in the matrix X. The test checks whether the word
     is over-represented in spammy messages and returns its p-value. The
     smaller the p-value, the more over-represented the word is within spams
     compared to hams.


    Args:
        X: the TF-IDF matrix where each line represents a document and each
           column represents a word, typically obtained by running
           transform_text().
        y: a binary vector where the i-th value indicates whether the i-th
           document is a spam, typically obtained by running transform_text().
        word_index: an int representing a column number in X.
    Returns:
        A double that corresponds to the p-value of the test (the probability
        that the word is NOT over-represented in the spams).
   """
    # get a full matrice instead of a sparse one
    X = X.todense()

    x0 = X[ y == 0, word_index ]
    x1 = X[ y == 1, word_index ]

    #  t < 0 means x0 < x1
    t, p = ttest_ind(x0, x1)
    return p
