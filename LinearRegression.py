import numpy as np
import scipy.optimize as op


def cost(theta, x, y):
    h = np.dot(x, theta)
    m = h.shape[0]
    return (1 / (2 * m)) * np.sum(np.square((h - y)))


def cost_grad(theta, x, y):
    h = np.dot(x, theta)
    m = h.shape[0]
    return ((1 / m) * np.dot(np.transpose(x),(h - y))).flatten()


data = np.loadtxt('./data/x01.txt')

x = data[:, 1:data.shape[1] - 1]
y = data[:, data.shape[1] - 1]

m = x.shape[0]  # number of training examples
n = data.shape[1] - 2  # number of features

x = x.reshape(m, n)
y = y.reshape(m, 1)

#x_ones = np.ones((m, 1))
#x = np.hstack((x_ones, x))

#theta = np.random.random((n + 1, 1))
theta = np.zeros((n, 1))
h = np.dot(x, theta)

print(theta)
print(theta.shape)
print(cost(theta, x, y))
print(cost(theta, x, y).shape)
print(cost_grad(theta, x, y))
print(cost_grad(theta, x, y).shape)

#theta_opt = op.fmin_bfgs(cost, theta, args=(x, y))
#res_opt = op.minimize(cost, theta, args=(x,y), method='BFGS', jac=cost_grad, options={'disp': True})
#res_opt = op.minimize(cost, theta, args=(x,y), method='BFGS', options={'disp': True})
res_opt = op.minimize(cost, theta, args=(x,y), method='nelder-mead', options={'xtol': 1e-8, 'disp':True})
theta_opt = res_opt.x
print(theta_opt)

x_test = np.array([3.385])
x_test = np.transpose(x_test)
print(np.dot(theta_opt, x_test))
