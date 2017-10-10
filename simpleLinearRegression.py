import numpy as np
import scipy.optimize as op

def cost(theta, x, y):
    h = np.dot(x, theta)
    m = h.shape[0]
    return (1 / (2 * m)) * np.sum(np.square((h - y)))

def cost_grad(theta, x, y):
    h = np.dot(x, theta)
    m = h.shape[0]
    return (1 / m) * np.dot(np.transpose(x),(h - y))

def gradient_descent(grad_func, theta_init, x, y, learning_rate, num_iter):
    theta = theta_init - (learning_rate * grad_func(theta_init, x, y))
    print('iteration:', 1, 'cost:', cost(theta, x, y))
    for i in range(num_iter-1):
        theta = theta - (learning_rate * grad_func(theta, x, y))
        print('iteration:', i+2, 'cost:', cost(theta, x, y))

    return theta

x = np.arange(50)
x = x.reshape(50,1)

y = 79 + (2*x)


m = x.shape[0]
n = x.shape[1]

x_ones = np.ones((m, 1))
x = np.hstack((x_ones, x))

theta = np.zeros((n+1, 1))

res_opt = op.minimize(cost, theta, args=(x,y), method='nelder-mead', options={'xtol': 1e-8, 'disp':True})

print(res_opt.x)

#theta = np.array([[79],[2]])
print(cost(theta, x, y))
print(gradient_descent(cost_grad, theta, x, y, 0.001, 40000))


