import math
import numpy as np
from time import time
import tensorflow as tf

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


def smooth_digit(S, K):
    return -1.0 / (1.0 + math.exp(-(K - S) / 0.005))


def smooth_diac(S, K):
    return math.exp(-(K-S)/0.005) / 0.005 / ((1.0+math.exp(-(K-S)/0.005))*(1.0+math.exp(-(K-S)/0.005)))


# compare the continuation value and the current payoff
def determine(continue_val, exercise_val, cashflow, m, j, exercise_boundary):
    for i in range(0, m):
        if continue_val[i] < exercise_val[i]:
            cashflow[i] = exercise_val[i]
            exercise_boundary[i] = j


# evaluate the payoff
def payoff(exercise_val, path, i, K, m):
    for j in range(0, m):
        exercise_val[j] = max(0.0, K - path[j][i])


# generate paths
def generate_paths(m, n, S0, sigma, r, dt, dW, path):
    S = S0
    for j in range(0, m):
        for i in range(0, n):
            S += r * S * dt + sigma[i] * S * math.sqrt(dt) * dW[j][i]
            path[j][i] = S
        S = S0

# generate the paths and do the regression steps backwards
def euler_path(m, n, S0, sigma, r, K, T, dW, test_functions, exercise_boundary):
    path = np.zeros((m, n))
    dt = T / n

    generate_paths(m, n, S0, sigma, r, dt, dW, path)

    continue_val = np.zeros(m)
    exercise_val = np.zeros(m)
    cashflow = np.zeros(m)

    for i in range(n - 1, 0, -1):
        # EXERCISE VALUE[0 ... m] = MAX(0.0, K - PATH[0 ... m][FROM 98 TO 1])
        payoff(exercise_val, path, i, K, m)
        # IF EXERCISE[0 ... m] > CONTINUATION[0 ... m] CASHFLOW[0 ... m] = EXERCISE[0 ... m]
        determine(continue_val, exercise_val, cashflow, m, i, exercise_boundary)

        x_train = np.zeros((m, len(test_functions)))
        y_train = np.zeros(m)
        size_regression = 0

        x = np.zeros((m, len(test_functions)))
        y = np.zeros(m)

        print("i")
        print(i)

        for j in range(0, m):
            cashflow[j] *= math.exp(-r * dt)
            for n in range(0, len(test_functions)):
                x[j][n] = test_functions[n](path[j][i - 1])
            y[j] = cashflow[j]
            if path[j][i] < K:
                for n in range(0, len(test_functions)):
                    x_train[size_regression][n] = test_functions[n](path[j][i-1])
                y_train[size_regression] = cashflow[j]
                size_regression += 1

        x_train = x_train[:size_regression]
        y_train = y_train[:size_regression]

        # NN
        mean_label = y_train.mean(axis=0)
        std_label = y_train.std(axis=0)
        mean_feat = x_train.mean(axis=0)
        std_feat = x_train.std(axis=0)
        x_train = (x_train-mean_feat)/std_feat
        y_train = (y_train-mean_label)/std_label

        linear_model = SimpleLinearRegression('zeros')
        linear_model.train(x_train, y_train, learning_rate=0.1, epochs=50)

        """
        mean_label = y.mean(axis=0)
        std_label = y.std(axis=0)
        mean_feat = x.mean(axis=0)
        std_feat = x.std(axis=0)
        """

        x = (x-mean_feat)/std_feat
        continue_val = linear_model.predict(x)
        continue_val *= std_label
        continue_val += mean_label

        """
        sum = 0.0
        for k in range(0, m):
            sum += (continue_val[k] - y[k])**2
        print("squares sum:")
        print(sum)
        """

    payoff(exercise_val, path, 0, K, m)
    determine(continue_val, exercise_val, cashflow, m, 0, exercise_boundary)

    V = 0
    for j in range(0, m):
        V += math.exp(-r * dt) * cashflow[j]
    return V / m


class AmericanPutOption(object):

    def __init__(self, m, n, S0, T, K, r, sigma):
        self.m = m
        self.n = n
        self.S0 = S0
        self.T = T
        self.K = K
        self.r = r
        self.sigma = np.full(n, sigma)
        self.vega = np.full(n, 0)
        self.vanna = np.full(n, 0)
        self.dW = np.zeros((m, n))
        self.S = S0
        self.delta = 1
        self.vegaBS = 0
        self.gamma = 0
        self.vannaBS = 0
        self.V = 0

    # the pricing function using tangent method
    def tangent_method(self, test_functions, exercise_boundary):
        sum, sumt, sumd, sumtt, sumtd = 0, 0, 0, 0, 0
        S0 = self.S
        delta0, vegaBS0, gamma0, vannaBS0= 1, 0, 0, 0
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
                if exercise_boundary[j] == i:
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


def main(m, n, S0, T, K, r, sigma, seed_1, seed_2):
    ao = AmericanPutOption(m, n, S0, T, K, r, sigma)

    np.random.seed(seed_1)
    ao.dW = np.random.normal(0,1,(m,n))

    test_functions = [l_0, l_1, l_2, l_3, l_4, l_5, l_6]
    exercise_boundary = np.full(m, 100)

    # generate the paths and calculate the exercise boundaries
    ao.V = euler_path(m, n, ao.S0, ao.sigma, ao.r, ao.K, ao.T, ao.dW, test_functions, exercise_boundary)

    print("V = {}".format(ao.V))

    np.random.seed(seed_2)
    ao = AmericanPutOption(m, n, S0, T, K, r, sigma)
    ao.dW = np.random.normal(0,1,(m,n))

    # calculate the price of the American option
    ao.tangent_method(test_functions, exercise_boundary)

    print("V_lower = {}".format(ao.S))
    print("Delta = {}".format(ao.delta))
    print("VegaBS = {}".format(ao.vegaBS))
    print("Gamma = {}".format(ao.gamma))
    print("VannaBS = {}".format(ao.vannaBS))

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

        # print("Loss: ", loss)
        dy_dm = g.gradient(loss, self.m)
        dy_db = g.gradient(loss, self.b)

        self.m.assign_sub(learning_rate * dy_dm)
        self.b.assign_sub(learning_rate * dy_db)

    def train(self, X, y, learning_rate=0.01, epochs=5):

        if len(X.shape)==1:
            X=tf.reshape(X,[X.shape[0],1])

        self.m.assign([self.var]*X.shape[-1])

        for i in range(epochs):
            # print("Epoch: ", i)
            self.update(X, y, learning_rate)

t0 = time()
#      m    n   S0  T  K    r  sigma seed_1 seed_2
main(10000, 100, 1, 1, 1, 0.04, 0.2, 4, 4)
t1 = time()
d1 = t1 - t0
print ("Duration in Seconds %6.3f" % d1)
