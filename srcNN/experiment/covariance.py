import numpy as np

def y(m, xs, noise):
    return m*xs + noise

def cov(x, y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    return np.sum((x - x_bar)*(y - y_bar))

def cov_for(has_noise):
    xs = np.linspace(-1.0, 1.0, num=10000)
    if has_noise:
        noise = np.random.normal(scale=0.2, size=10000)
    else:
        noise = np.zeros(10000)
    ys = y(1.0, xs, noise)
    return cov(xs, ys)

print cov_for(True)
print cov_for(False)
