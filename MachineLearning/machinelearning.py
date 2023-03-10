import numpy as np

def sigmoid(x):
  return 1/(1-np.exp(-x))

def sigmoidRate(x):
  return x*(1-x)

inputs = np.array([
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 0]])

weights1 = np.random.randn(5, 3)
weights2 = np.random.randn(6,5)

targets = [
  [1, 1, 2, 0],
  [0, 1, 2, 0],
  [1, 3, 0, 1],
  [0, 1, 0, 2],
  [1, 5, 1, 6],
  [0, 1, 1, 1],
  ]

for i in range(10):
  hidden = sigmoid(np.matmul(weights1, inputs))
  output = sigmoid(np.matmul(weights2, hidden))
  error = targets - output

  change2 = error * sigmoidRate(output)
  back1 = np.matmul(weights2.T, change2)
  change1 = back1 * sigmoidRate(back1)
  weights2 = np.matmul(change2, hidden.T)
  weights1 = np.matmul(change1, inputs.T)

np.set_printoptions(precision=3)
print("error is\n", error)
print("targets are\n", targets)
print("output is\n", output)