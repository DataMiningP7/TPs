# -*- coding: UTF-8 -*-

import numpy as np
import pylab as plt


def ex1():
    x = np.random.random(100)
    y = np.random.random(100)
    # See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    # red line: "r"
    # red dots: "r."
    # red squares: "rs"
    # green triangles: "g^"
    plt.plot(x, y, "g^", linewidth=2)

    plt.show()


def ex2():
    x = np.linspace(-10, 10)

    # "--" = dashed line
    plt.plot(x, np.sin(x), "--", label="sinus")
    plt.plot(x, np.cos(x), label="cosinus")

    # Show the legend using the labels above
    plt.legend()

    # The trick here is we have to make another plot on top of the two others.
    pi2 = np.pi/2

    # Find B such that (-B * pi/2) >= -10 > ((-B-1) * pi/2), i.e. the
    # first multiple of pi/2 higher than -10.
    b = pi2*int(-10.0/pi2)

    # x2 is all multiples of pi/2 between -10 and 10.
    x2 = np.arange(b, 10, pi2)

    # "b." = blue dots
    plt.plot(x2, np.sin(x2), "b.")
    plt.show()


def ex3():
    x = np.random.randn(1000)

    # Boxplot
    #plt.boxplot(x)

    # Histogram
    plt.hist(x)

    plt.title("Mon Titre")

    plt.show()
