# -*- coding: UTF-8 -*-

# Ex 1.1
def average(lst):
    """ Computes the average of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float that represents the average of the values in lst.
    """
    # ...


# Ex 1.2
def median(lst):
    """ Computes the median of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float that represents the median of the values in lst.
    """
    # ...


# Ex 1.3
def occurrences(lst):
    """ Computes the occurrences of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A dictionary where each key is a value in lst and each value is the
        number of times the key appears in lst.
    """
    # ...


# Ex 1.4
def unique(lst):
    """ Returns the unique values of a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A list containing the unique values in lst.
    """
    # ...


# Ex 2.1
def squares(lst):
    """ Squares the values of a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A list containing the squared elements of lst, in the same order.
    """
    # ...


# Ex 2.2
def stddev(lst):
    """ Returns the standard deviation of a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float representing the empirical standard deviation of elements in
        lst.
    """
    # ...


# Ex 2.3 (optional)
def quicksort(lst):
    """ Returns a sorted version of lst.
    Args:
        lst: a list of integers.
    Returns:
        A list of the integers from lst, in increasing order.
    """
    # ...


# Ex 3.1, 3.2, 3.3
def uniform():
    """ Returns a discrete uniform variable between 0 and 1, i.e., it returns 0
    with probability 0.5 and 1 with probability 0.5.
    """
    # ...


# Ex 3.4
def exam_success(n, p):
    """ Returns the number of exams passed by a student.
    Args:
        n: number of independent exams the students sits for.
        p: probability of passing one exam.
    Returns:
        An integer representing the total number of exams passed by the
        student.
    """
    # ...


# Ex 4.1
def monty_hall(change):
    """ Simulates a Monty Hall game: a candidate is asked to choose between
    three doors with only one of them containing a reward. Once the candidate
    has chosen a door, the anchorman opens one of the other doors that does not
    contain the reward. The candidate may now change doors for the remaining
    one. This function returns 1 if the candidate found the reward and 0
    otherwise.
    Args:
        change: a boolean representing whether the candidate changes doors.
    Returns:
        1 if the candidate found the reward at the end of the game, 0
        otherwise.
    """
    # ...


# Ex 4.2
def monty_hall_simulation(n):
    """ Simulates n Monty Hall games with the candidate changing doors and n
    Monty Hall games with the candidate not changing doors. Returns the
    frequency with which the candidate found the reward in each case.
    Args:
        n: the number of games to simulate.
    Returns:
        A tuple (p1, p2) where p1 (resp. p2) is the frequency with which the
        candidate found the reward when always (resp. never) changing doors.
    """
    # ...
