import tensorflow as tf

#Building computational graph
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

#Launching the graph in a session
sess = tf.Session()

#Visualizing the graph
File_Writer = tf.summary.FileWriter('graph_1', sess.graph)

#Evaluating tensor 'f'
print(sess.run(f))