# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)

input_low_s = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_s')
input_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_r')
input_beta = tf.placeholder(tf.float32, [None, None, None, 3], name='input_beta')

#Net1
beta = Net1(input_low_s)
output_r = tf.pow(input_low_s+0.01,beta)
#Net2
output_r2 = Net2(input_r,input_beta)


var_net1 = [var for var in tf.trainable_variables() if 'Restoration_net' in var.name]
var_net2 = [var for var in tf.trainable_variables() if 'reconnet' in var.name]

saver_net1 = tf.train.Saver(var_list=var_net1)
saver_net2 = tf.train.Saver(var_list=var_net2)


checkpoint_dir_net1 = './checkpoint/Net1/'
ckpt=tf.train.get_checkpoint_state(checkpoint_dir_net1)
if ckpt:
    print('loaded '+ ckpt.model_checkpoint_path)
    saver_net1.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No pre model!")
    
checkpoint_dir_net2 = './checkpoint/Net2/'
ckpt=tf.train.get_checkpoint_state(checkpoint_dir_net2)
if ckpt:
    print('loaded '+ ckpt.model_checkpoint_path)
    saver_net2.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No pre model!")

#load eval data
eval_low_data = []
eval_img_name =[]
#eval data dir
eval_low_data_name = glob('./testdata/low/*')
eval_low_data_name.sort()

#evaluate
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
    print(eval_low_im.shape)

#save dir
sample_dir = './results/data/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("Start evalating!")

for idx in range(len(eval_low_data)):
    print(idx)
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    h, w, _ = input_low.shape
    input_low2 = np.zeros((h+20,w+20,3))
    input_low2[:,:,0] = np.pad(input_low[:,:,0],((10,10),(10,10)),"edge")
    input_low2[:,:,1] = np.pad(input_low[:,:,1],((10,10),(10,10)),"edge")
    input_low2[:,:,2] = np.pad(input_low[:,:,2],((10,10),(10,10)),"edge")
    input_low_eval = np.expand_dims(input_low2, axis=0)
    
    out_beta = sess.run(beta, feed_dict={input_low_s: input_low_eval})
    restoration_r = (input_low_eval + 0.01) ** out_beta
    output_final = sess.run(output_r2, feed_dict={input_r: restoration_r,input_beta:out_beta})

    restoration_r = restoration_r[:, 10:10 + h, 10:10 + w, :]
    output_final = output_final[:, 10:10 + h, 10:10 + w, :]

    #save
    save_images(os.path.join(sample_dir, '%s_net1.png' % (name)), restoration_r)
    save_images(os.path.join(sample_dir, '%s_output_final_net2.png' % (name)), output_final)