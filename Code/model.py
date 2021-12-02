import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers import layer_norm
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
import numpy as np

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output

def resBlock(x,name_scope,channels=64,kernel_size=[3,3],scale=1):

    with tf.variable_scope(name_scope):
        tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
        tmp = tf.nn.relu(tmp)
        tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
        tmp *= scale
    return x + tmp

def Net1(input_i):
    with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input_i,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        up8 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'g_up_2')
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv10=slim.conv2d(conv9,3,[3,3], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)
    return R_out
#     final net
def Net2(input_r,beta,dims = 32):
    with tf.variable_scope('reconnet', reuse=tf.AUTO_REUSE):
        b_conv1 = slim.conv2d(beta, dims, [3, 3], rate=1, activation_fn=lrelu,scope='b_conv1')
        b_pool1 = slim.max_pool2d(b_conv1, [2, 2], padding='SAME')

        b_conv2 = slim.conv2d(b_pool1, dims * 2, [3, 3], rate=1, activation_fn=lrelu,scope='b_conv2')
        b_pool2 = slim.max_pool2d(b_conv2, [2, 2], padding='SAME')

        b_conv3 = slim.conv2d(b_pool2, dims * 4, [3, 3], rate=1, activation_fn=lrelu,scope='b_conv3')
        b_pool3 = slim.max_pool2d(b_conv3, [2, 2], padding='SAME')

        b_conv4 = slim.conv2d(b_pool3, dims * 8, [3, 3], rate=1, activation_fn=lrelu,scope='b_conv4')
        b_pool4 = slim.max_pool2d(b_conv4, [2, 2], padding='SAME')
        b_conv5 = slim.conv2d(b_pool4, dims * 16, [3, 3], rate=1, activation_fn=lrelu,scope='b_conv5')


        conv1=slim.conv2d(input_r,dims,[3,3], rate=1, activation_fn=lrelu,scope='r_conv1_1')
        conv1=slim.conv2d(tf.concat([conv1,b_conv1],3),dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm, scope='r_conv1_2')
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

        conv2=slim.conv2d(pool1,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv2_1')
        conv2=slim.conv2d(tf.concat([conv2,b_conv2],3),dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv2_2')
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

        conv3=slim.conv2d(pool2,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv3_1')
        conv3=slim.conv2d(tf.concat([conv3,b_conv3],3),dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv3_2')
        pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

        conv4=slim.conv2d(pool3,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv4_1')
        conv4=slim.conv2d(tf.concat([conv4,b_conv4],3),dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv4_2')
        pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

        conv5=slim.conv2d(pool4,dims*16,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv5_1')
        conv5=slim.conv2d(tf.concat([conv5,b_conv5],3),dims*16,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv5_2')

        net = conv5
        for i in range(4):
            temp = net
            net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu,normalizer_fn=layer_norm, scope='r_res%d_conv1'%i)
            net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='r_res%d_conv2'%i)
            net = net + temp

        net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='r_res')
        conv5 = net + conv5

        up6 =  upsample_and_concat( conv5, conv4, dims*8, dims*16  ,'up_6')
        conv6=slim.conv2d(up6,  dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv6_1')
        conv6=slim.conv2d(conv6,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, dims*4, dims*8  ,'up_7')
        conv7=slim.conv2d(up7,  dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv7_1')
        conv7=slim.conv2d(conv7,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, dims*2, dims*4 ,'up_8')
        conv8=slim.conv2d(up8,  dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv8_1')
        conv8=slim.conv2d(conv8,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, dims, dims*2 ,'up_9')
        conv9=slim.conv2d(up9,  dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv9_1')
        conv9=slim.conv2d(conv9,dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='r_conv9_2')

        conv10 = slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='g_conv10')

        out = tf.sigmoid(conv10)
    return out
