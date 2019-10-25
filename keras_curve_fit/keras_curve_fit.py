import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU


x_train = 1.5*np.random.rand(100000) - 0.25
y_train = x_train**4 + x_train**3 - x_train
x_train = x_train.reshape(len(x_train), 1)

x_test = np.linspace(0, 1, 1000)
y_test = x_test**4 + x_test**3 - x_test
x_test = x_test.reshape(len(x_test), 1)

model = Sequential()
model.add(Dense(units=60, input_dim=1))
model.add(LeakyReLU(0.01))
model.add(Dense(units=1))
print(model)

model.compile(
        loss='mean_squared_error',
        optimizer='sgd')

model.fit(x_train, y_train, epochs=10, batch_size=40, verbose=1)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1000)

classes = model.predict(x_test, batch_size=1)

test = x_test.reshape(-1)
plt.plot(test, y_test, 'k:')
plt.plot(test, classes)
plt.show()
