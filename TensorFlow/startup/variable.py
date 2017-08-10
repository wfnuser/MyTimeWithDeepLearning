import tensorflow as tf
import numpy as np

state = tf.Variable(0, name = 'counter')
one = tf.constant(1)

new_state = tf.add(state, one)
update = tf.assign(state, new_state)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update)
        print(sess.run(state))
