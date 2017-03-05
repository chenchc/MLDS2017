
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.0001
training_iters = 20000000
batch_size = 100
display_step = 5

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    

    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

pred_number = tf.argmax(pred, 1)

global_step = tf.Variable(0, name='global_step', trainable=False)
#train_op = optimizer.minimize(loss, global_step=global_step)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))





# Initializing the variables
init = tf.global_variables_initializer()


saver01 = tf.train.Saver()


config = tf.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth=True

# Launch the graph
with tf.Session(config=config) as sess:

    sess.run(init)
    try:

        #saver01 = tf.train.import_meta_graph('./my-model-20490.meta')
        #saver01.restore(sess,'./my-model-20490')
        print("loaded previous weights")
        
    except Exception as e:
        print(e) 
    
    
    
    # TEST NETWORK
    if(False): 
        prediction_final, pred, correct_pred= sess.run([pred_number,pred, correct_pred], feed_dict={x: mnist.test.images[:9999],
                                      y: mnist.test.labels[:9999],
                                      keep_prob: 1.})
        to_count=0
        missed=0

        f = open('myfile.csv', 'w')
        for i in range(0,9999):

            f.write("%d" % i)
            f.write(',')
            f.write("%d" % prediction_final[i])


            '''
            f.write('  ')

            cur_pred=pred[i]

            norm=np.linalg.norm(cur_pred, ord=1)
            cur_pred=cur_pred/norm


            cur_pred.sort()
            cur_pred=cur_pred[::-1]

            certainty=abs(cur_pred[0]-cur_pred[1])/abs(cur_pred[0])


            print(cur_pred)

            f.write("%f" % certainty)

            if(certainty<=0.4):
                    to_count=to_count+1

            if(not correct_pred[i]):
                
                if(certainty>0.4):
                    f.write('########')
                    missed=missed+1

            '''

            f.write('\n')  

        f.close() 

        '''
        print("Check= " + \
                      "{:d}".format(to_count) + ", Miss= " + \
                      "{:d}".format(missed))
        '''

        batch_test_x, batch_test_y = mnist.train.next_batch(3000)
                # Calculate batch loss and accuracy
        loss, acc, prediction = sess.run([cost, accuracy, pred_number], feed_dict={x: batch_test_x,
                                                                  y: batch_test_y,
                                                                  keep_prob: 1.})
        print("Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    # TRAIN NETWORK
    else:

        
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)next_batch
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:

                ## Generates MetaGraphDef.
                saver01.save(sess,'./saved_networks/my-model', global_step=global_step)


                batch_test_x, batch_test_y = mnist.train.next_batch(3000)
                # Calculate batch loss and accuracy
                loss, acc, prediction = sess.run([cost, accuracy, pred_number], feed_dict={x: batch_test_x,
                                                                  y: batch_test_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                

            step += 1



        print("Optimization Finished! Final prediction: ")
        


    
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
    