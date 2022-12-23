from numpy import *
from sklearn.metrics import mean_squared_error

train_in = array([[1, 0.22, 0.19, 0.85, 0.3], [1, 0.25, 0.2, 0.72, 0.3], [1, 0.23, 0.21, 0.9, 0.4], [1, 0.25, 0.21, 0.8, 0.3], [1, 0.22, 0.21, 0.93, 0.3], [1, 0.24, 0.21, 0.81, 0.4], [1, 0.22, 0.2, 0.92, 0.3], [1, 0.24, 0.2, 0.79, 0.3], [1, 0.21, 0.2, 0.93, 0.3], [1, 0.2, 0.17, 0.86, 0.3], [1, 0.17, 0.15, 0.9, 0.3], [1, 0.18, 0.13, 0.73, 0.4]])
train_sol = array([[0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.8, 1.1, 1.1, 1]]).T

random.seed(1)
w = 2 * random.random((5, 1)) - 1

l_rate = 1
generation = 20000

for i in range(generation):
    print("\n i= ", i, " w=")
    print(w)
    
    train_out = 1 / (1 + exp(-((dot(train_in, w)))))
    print("train_out =")
    print(train_out)
    
    w += l_rate * dot(train_in.T, (train_sol - train_out) * train_out * (1 - train_out))

print('\n Learning rate is', l_rate)
print(' Generation is', generation)
print(' Train Mean squared error is',mean_squared_error(train_sol, train_out))

test_in = array([1, 0.16, 0.11, 0.73, 0.4])
test_sol = array([[0.2]])
test_out = 1 / (1 + exp(-((dot(test_in, w)))))


print(' Test Mean squared error is',mean_squared_error(test_sol, test_out))
print(' Result = ', test_out)