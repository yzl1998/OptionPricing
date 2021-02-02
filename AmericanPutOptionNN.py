import math
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

# legendre polynomial functions
def l_0(val_arg):
    return math.exp(-val_arg / 2)


def l_1(val_arg):
    return math.exp(-val_arg / 2) * (1 - val_arg)


def l_2(val_arg):
    return math.exp(-val_arg / 2) * (1 - 2 * val_arg + 1 / 2 * (val_arg ** 2))


def l_3(val_arg):
    return math.exp(-val_arg / 2) * (1 - 3 * val_arg + 3 / 2 * (val_arg ** 2) - 1 / 6 * (val_arg ** 3))


def l_4(val_arg):
    return math.exp(-val_arg / 2) * (1 - 4 * val_arg + 3 * (val_arg ** 2)
                                     - 3 / 4 * (val_arg ** 3) + 1 / 24 * (val_arg ** 4))


def l_5(val_arg):
    return math.exp(-val_arg / 2) * (1 - 5 * val_arg + 5 * (val_arg ** 2) - 10 / 6 * (val_arg ** 3)
                                     + 25 / 120 * (val_arg ** 4) - 1 / 120 * (val_arg ** 5))


def l_6(val_arg):
    return math.exp(-val_arg / 2) * (1 - 6 * val_arg + 540 / 72 * (val_arg ** 2)
                                     - 240 / 72 * (val_arg ** 3) + 45 / 72 * (val_arg ** 4)
                                     - 1 / 20 * (val_arg ** 5) + 1 / 720 * (val_arg ** 6))


# check whether to exercise or continue
def whether_exercise(S, K, k, coefficients, test_functions):
    continue_val = 0
    n = len(test_functions)
    for i in range(0, n):
        continue_val += coefficients[k][i] * test_functions[i](S)
    return max(K - S, 0.0) > continue_val


def smooth_digit(S, K):
    return -1.0 / (1.0 + math.exp(-(K - S) / 0.005))


def smooth_diac(S, K):
    return math.exp(-(K-S)/0.005) / 0.005 / ((1.0+math.exp(-(K-S)/0.005))*(1.0+math.exp(-(K-S)/0.005)))


# compare the continuation value and the current payoff
def determine(continue_val, exercise_val, cashflow, m):
    for i in range(0, m):
        if continue_val[i] < exercise_val[i]:
            cashflow[i] = exercise_val[i]


# evaluate the payoff
def payoff(exercise_val, path, i, K, m):
    for j in range(0, m):
        exercise_val[j] = max(0.0, K - path[j][i])


# regression function
def regression(x_regression, y_regression, size_regression, coefficients, k, test_functions):
    size_1 = size_regression
    size_2 = len(test_functions)
    y_regression_2 = np.zeros(size_regression)

    # construct the matrix phi
    d_phi = np.zeros((size_1, size_2))
    for i in range(0, size_1):
        y_regression_2[i] = y_regression[i]
        for j in range(0, size_2):
            d_phi[i][j] = test_functions[j](x_regression[i])

"""
# construct A = phi^T * phi, b = phi^T * Y
    d_phi_transpose = d_phi.transpose()
    A = np.dot(d_phi_transpose, d_phi)
    b = np.dot(d_phi_transpose, y_regression_2)
    # solve linear equations Ax = b
    try:
        x = np.linalg.solve(A, b)
        for i in range(0, size_2):
            coefficients[k][i] = x[i]
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            raise np.linalg.LinAlgError("Singular Matrix Error!")
        else:
            raise np.linalg.LinAlgError("Unexpected Error!")
"""

# evaluate the continuation value
def projection(continue_val, coefficients, k, test_functions, path, m):
    n = len(test_functions)
    for i in range(0, m):
        continue_val[i] = 0
        for j in range(0, n):
            continue_val[i] += coefficients[k][j] * test_functions[j](path[i][k])


# generate the paths and do the regression steps backwards
def euler_path(m, n, S0, sigma, r, K, T, dW, coefficients, test_functions):
    path = np.zeros((m, n))
    dt = T / n
    S = S0

    # generate all paths
    for j in range(0, m):
        for i in range(0, n):
            S += r * S * dt + sigma[i] * S * math.sqrt(dt) * dW[j][i]
            path[j][i] = S
        S = S0

    # do the regression steps backwards
    continue_val = np.zeros(m)
    exercise_val = np.zeros(m)
    cashflow = np.zeros(m)

    for i in range(n - 1, 0, -1):
        projection(continue_val, coefficients, i, test_functions, path, m)
        payoff(exercise_val, path, i, K, m)
        determine(continue_val, exercise_val, cashflow, m)

        x_regression = np.zeros(m)
        y_regression = np.zeros(m)
        size_regression = 0

        for j in range(0, m):
            cashflow[j] *= math.exp(-r * dt)
            if path[j][i] < K:
                x_regression[size_regression] = path[j][i - 1]
                y_regression[size_regression] = cashflow[j]
                size_regression += 1

        print("i & size_regression ")
        print(i)
        print(size_regression)
        """
        regression(x_regression, y_regression, size_regression, coefficients, i - 1, test_functions)
        """

        # construct the matrix phi

        d_phi = np.zeros((size_regression, len(test_functions)))
        y_regression_2 = np.zeros(size_regression)
        for m in range(0, size_regression):
            y_regression_2[m] = y_regression[m]
            for n in range(0, len(test_functions)):
                d_phi[m][n] = test_functions[n](x_regression[m])

        x_train = d_phi
        print("actual")
        print(y_regression_2)
        """
        print(len(x_train))
        print(x_train)
        """
        y_train = y_regression_2
        """
        print(len(y_train))
        print(y_train)
        """

        mean_label = y_train.mean(axis=0)
        std_label = y_train.std(axis=0)
        mean_feat = x_train.mean(axis=0)
        std_feat = x_train.std(axis=0)
        x_train = (x_train-mean_feat)/std_feat
        y_train = (y_train-mean_label)/std_label

        linear_model = SimpleLinearRegression('zeros')
        linear_model.train(x_train, y_train, learning_rate=0.1, epochs=50)
        pred = linear_model.predict(x_train)
        pred *= std_label
        pred += mean_label

        print("predict")
        print(pred)

        sum = 0.0
        for k in range(0, size_regression):
            sum += (pred[k] - y_regression_2[k])**2
            print("sum += ")
            print((pred[k] - y_regression_2[k])**2)
        print("squares sum:")
        print(sum)
        # print(linear_model.m.numpy)
        coefficients[i] = linear_model.m.read_value()

        print(coefficients[99])
        print(coefficients[101])


    projection(continue_val, coefficients, 0, test_functions, path, m)
    payoff(exercise_val, path, 0, K, m)
    determine(continue_val, exercise_val, cashflow, m)
    print(continue_val)
    V = 0

    # evaluate the option
    for j in range(0, m):
        V += math.exp(-r * dt) * cashflow[j]
    return V


class AmericanPutOption(object):

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.S0 = 1
        self.T = 1
        self.K = 1
        self.r = 0.04
        self.sigma = np.full(n, 0.2)
        self.vega = np.full(n, 0)
        self.vanna = np.full(n, 0)
        self.dW = np.zeros((m, n))
        self.S = self.S0
        self.delta = 1
        self.gamma = 0
        self.vegaBS = 0
        self.vannaBS = 0
        self.V = 0

    # the pricing function using tangent method
    def tangent_method(self, coefficients, test_functions):
        sum, sumt, sumd, sumtt, sumtd = 0, 0, 0, 0, 0
        S0 = self.S
        delta0, vegaBS0, gamma0, vannaBS0= self.delta, self.vegaBS, self.gamma, self.vannaBS
        dt = self.T/self.n

        for j in range(0, self.m):
            t = 0
            for i in range(0, self.n):
                self.vannaBS += self.r*self.vannaBS*dt + self.sigma[i]*self.vannaBS*math.sqrt(dt)*self.dW[j][i] \
                                + self.delta*math.sqrt(dt)*self.dW[j][i]
                self.vegaBS += self.r*self.vegaBS*dt + self.sigma[i]*self.vegaBS*math.sqrt(dt)*self.dW[j][i] \
                               + self.S*math.sqrt(dt)*self.dW[j][i]
                self.delta += self.r*self.delta*dt + self.sigma[i]*self.delta*math.sqrt(dt)*self.dW[j][i]
                self.S += self.r*self.S*dt + self.sigma[i]*self.S*math.sqrt(dt)*self.dW[j][i]
                t = dt*(i+1)
                if whether_exercise(self.S, self.K, i, coefficients, test_functions):
                    sum += max(self.K - self.S, 0.0)*math.exp(- self.r*t)
                    sumt += smooth_digit(self.S, self.K)*math.exp(- self.r*t)*self.delta
                    sumd += smooth_digit(self.S, self.K)*math.exp(- self.r*t)*self.vegaBS
                    sumtt += (self.delta ** 2)*math.exp(- self.r*self.T)*smooth_diac(self.S, self.K)
                    sumtd += math.exp(- self.r*t)*smooth_diac(self.S, self.K)*self.delta*self.vegaBS \
                             + smooth_digit(self.S, self.K)*math.exp(- self.r*t)*self.vannaBS
                    break

            self.S = S0
            self.delta = delta0
            self.gamma = gamma0
            self.vegaBS = vegaBS0
            self.vannaBS = vannaBS0

        self.S = sum/self.m
        self.delta = sumt/self.m
        self.gamma = sumtt/self.m
        self.vegaBS = sumd/self.m
        self.vannaBS = sumtd/self.m


def main(m, n):
    ao = AmericanPutOption(m, n)

    np.random.seed(4)
    ao.dW = np.random.normal(0,1,(m,n))

    test_functions = [l_0, l_1, l_2, l_3, l_4, l_5, l_6]
    coefficients = np.full((n, len(test_functions)), 0.0)

    # generate the paths and calculate the regression coefficients
    ao.V = euler_path(m, n, ao.S0, ao.sigma, ao.r, ao.K, ao.T, ao.dW, coefficients, test_functions)

    # print(coefficients)
    print("V = {}".format(ao.V))

    np.random.seed(2)
    ao = AmericanPutOption(2*m, n)
    ao.dW = np.random.normal(0,1,(2*m,n))

    # calculate the price of the American option
    ao.tangent_method(coefficients,test_functions)

    print("V_lower = {}".format(ao.S))
    # dV/dS0
    print("delta = {}".format(ao.delta))
    # dV2/dS02
    print("gamma = {}".format(ao.gamma))
    # dV/dsigma
    print("vegaBS = {}".format(ao.vegaBS))
    # dV2/dsigma dS0
    print("vannaBS = {}".format(ao.vannaBS))

class SimpleLinearRegression:
    def __init__(self, initializer='random'):
        if initializer=='ones':
            self.var = 1.
        elif initializer=='zeros':
            self.var = 0.
        elif initializer=='random':
            self.var = tf.random.uniform(shape=[], minval=0., maxval=1.)

        self.m = tf.Variable(1., shape=tf.TensorShape(None))
        self.b = tf.Variable(self.var)

    def predict(self, x):
        return tf.reduce_sum(self.m * x, 1) + self.b

    def mse(self, true, predicted):
        return tf.reduce_mean(tf.square(true-predicted))

    def update(self, X, y, learning_rate):
        with tf.GradientTape(persistent=True) as g:
            loss = self.mse(y, self.predict(X))

        print("Loss: ", loss)

        dy_dm = g.gradient(loss, self.m)
        dy_db = g.gradient(loss, self.b)

        self.m.assign_sub(learning_rate * dy_dm)
        self.b.assign_sub(learning_rate * dy_db)

    def train(self, X, y, learning_rate=0.01, epochs=5):

        if len(X.shape)==1:
            X=tf.reshape(X,[X.shape[0],1])

        self.m.assign([self.var]*X.shape[-1])

        for i in range(epochs):
            """
            print("Epoch: ", i)
            """
            self.update(X, y, learning_rate)

t0 = time()
main(10000, 100)
t1 = time()
d1 = t1 - t0
print ("Duration in Seconds %6.3f" % d1)
