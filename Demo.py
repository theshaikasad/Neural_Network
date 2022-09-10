import numpy as np 
inputs = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1], [0,0,0]])
outputs = np.array([[1],[0],[1],[0]])
np.random.seed(1)
weights = 2 * np.random.random((3, 1))
def sigmoid(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))
print('training...')
for i in range(60000):
	pred = sigmoid(np.dot(inputs, weights))
	error = pred- outputs
	adjustments = np.dot(inputs.T, error* sigmoid(pred, deriv=True))
	weights -= adjustments
print('Your neural network has finished training!. Let us test this...')
print('Considering New situation ->> [1, 1, 0]')
output_for_test = sigmoid(np.dot(np.array([1,1,0]), weights))
print('predicting..')
print(output_for_test) 
