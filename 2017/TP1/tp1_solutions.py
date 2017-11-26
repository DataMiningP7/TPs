# -*- coding: UTF-8 -*-

# for the Ex 2.2
import math
# for the Ex3
from random import random, randint

# Ex 1.1
def average(lst):
    """ Computes the average of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float that represents the average of the values in lst.
    """
    # easy solution:
    # sum(lst)/len(lst)

    # solution without sum nor len:
    s = 0
    length = 0
    for e in lst:
        s += e
        length += 1

    return s/length


# Ex 1.2
def median(lst):
    """ Computes the median of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float that represents the median of the values in lst.
    """
    lst = sorted(lst)
    l = len(lst)

    middle = l // 2

    if l % 2 == 1:
        return lst[ middle ]

    e1 = lst[ middle - 1 ]
    e2 = lst[ middle ]
    return ( e1 + e2 ) / 2


# Ex 1.3
def occurrences(lst):
    """ Computes the occurrences of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A dictionary where each key is a value in lst and each value is the
        number of times the key appears in lst.
    """
    d = {}

    for e in lst:
        if e not in d:
            d[e] = 0

        d[e] += 1

    return d


# Ex 1.4
def unique(lst):
    """ Returns the unique values of a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A list containing the unique values in lst.
    """
    lst2 = []
    dejavu = set()

    for e in lst:
        if e not in dejavu:
            lst2.append(e)
            dejavu.add(e)

    return lst2


# Ex 2.1
def squares(lst):
    """ Squares the values of a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A list containing the squared elements of lst, in the same order.
    """
    return [e ** 2 for e in lst]


# Ex 2.2
def stddev(lst):
    """ Returns the standard deviation of a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float representing the empirical standard deviation of elements in
        lst.
    """
    avg = average(lst)
    n = 0
    for e in lst:
        n += (e - avg) ** 2

    return math.sqrt(n/len(lst))


# Ex 2.3 (optional)
def quicksort(lst):
    """ Returns a sorted version of lst.
    Args:
        lst: a list of integers.
    Returns:
        A list of the integers from lst, in increasing order.
    """
    if len(lst) <= 1:
        return lst

    pivot = lst[0]
    lower = []
    higher = []

    for i, e in enumerate(lst):
        if i == 0:
            # pivot
            continue

        if e <= pivot:
            lower.append(e)
        else:
            higher.append(e)

    final_lst = quicksort(lower) + [pivot] + quicksort(higher)
    return final_lst


# Ex 3.1
def uniform_1():
    """ Returns a discrete uniform variable between 0 and 1, i.e., it returns 0
     with probability 0.5 and 1 with probability 0.5.
    """
    return int(random() * 2)

# Ex 3.1, question 2
def test_uniform_1():
    tries = [uniform() for _ in range(1000)]
    return occurrences(tries)

# Ex 3.2
def uniform_2():
    """ Returns a discrete uniform variable between 0 and 5, i.e., the
     probability of obtaining 0, 1, 2, 3, 4 or 5 is always 1/6.
    """
    return int(random() * 6)

# Ex 3.3
def uniform(n):
    """ Returns a discrete uniform variable between 0 and n-1, i.e., it returns
    every number between 0 and n-1 with the same probability.
    Args:
        n: a positive integer
    """
    return int(random() * n)


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
    return p ** n


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
    chosen = randint(1, 3)   # chosen door
    winning = randint(1, 3)  # winning door

    if not change:
        # If we don't change, the only way to win is to have chosen the winning
        # door:
        return chosen == winning

    # Otherwise if we change, the only way to win is NOT to have chosen the
    # winning door in the first place.
    return chosen != winning

    # One-line solution:
    # return (chosen != winning) == change


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
    tries_change = 0
    tries_no_change = 0

    for _ in range(n):
        if monty_hall(True):
            tries_change += 1

        if monty_hall(False):
            tries_no_change += 1

    return (tries_change/n,
            tries_no_change/n)
