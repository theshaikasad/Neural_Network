#Import dependencies
import tensorflow as tf 
import numpy as np 
#Data
x_data = np.array([[1., 0., 1.], [0., 0., 1.], [1., 1., 1.], [0.,0.,0.]])
y_data = np.array([[1.],[0.],[1.],[0.]])
weights = tf.Variable(tf.random_uniform([3, 1], -1, 1)) 
X = tf.placeholder(tf.float32, shape = (4,3))
y = tf.nn.sigmoid(tf.matmul(X, weights))
y_ = tf.placeholder(tf.float32, shape=(4, 1))
loss = tf.reduce_mean(tf.square(y-y_)) #Error
#Gradient descent to get the cost corrected!
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.Session() #Launch the graph
sess.run(tf.initialize_all_variables())
for i in range(2000):
	sess.run(optimizer, feed_dict={X:x_data, y_:y_data})
	print("Loss\n", sess.run(loss, feed_dict={X:x_data, y_:y_data}))
	print("Predictions\n", sess.run(y, feed_dict ={X:x_data}))
