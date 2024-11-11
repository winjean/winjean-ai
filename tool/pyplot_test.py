import matplotlib.pyplot as plt
import numpy as np


def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('value')
    plt.show()


if __name__ == '__main__':
    x = np.linspace(-10, 10, 21)
    y = x**3 + x**2 - 2
    print(x)
    print(y)
    plot(x, y)
