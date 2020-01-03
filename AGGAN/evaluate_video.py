
from datetime import datetime
import json
import numpy as np
import os
import random
from scipy.misc import imsave
import cv2

import argparse
import tensorflow as tf


def test():
    """Test Function."""
    print("Testing the results")
    cap = cv2.VideoCapture('E:/CycleGAN-tensorflow-xhujoy/datasets/video/winter/winter_snow_drive_high_street_fair_1080p.mp4')
    x = 340
    # left top corner
    h = 800
    w = 800
    y = 150

    #model_setup()
    out = cv2.VideoWriter('output.avi', -1, 25.0, (256, 256))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        _checkpoint_dir='C:/Users/qxy9300/Documents/MA/02_Results/AGGAN_logs/exp_014/20191218-100450_switch40_thres_0.1'
        chkpt_fname = tf.train.latest_checkpoint(_checkpoint_dir)
        print('------------------->  ', chkpt_fname)
        # saver = tf.train.Saver()
        # tf.train.Saver().restore(sess, chkpt_fname)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                # cv2.imshow('Frame',frame)
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (256, 256))
                cv2.imshow('CroppedFrame', resized_img)

                out_img = save_images_bis(sess, crop_img)

                # write the flipped frame
                out.write(out_img)


        coord.request_stop()
        coord.join(threads)

def save_images_bis(sess, crop_img):
    """
    Saves input and output images.

    :param sess: The session.
    :param epoch: Currnt epoch.
    """

    names = ['input_A_', 'mask_A_', 'masked_inputA_', 'fakeB_',
             'input_B_', 'mask_B_', 'masked_inputB_', 'fakeA_']

    fake_A_temp, fake_B_temp, masks, masked_ims = sess.run([crop_img,
        crop_img,
        crop_img,
        crop_img
    ], feed_dict={
        input_a: crop_img,
        input_b: crop_img,
        transition_rate: 0.1
    })
    tensors = [inputs['images_i'], masks[0], masked_ims[0], fake_B_temp,
               inputs['images_j'], masks[1], masked_ims[1], fake_A_temp]

    for name, tensor in zip(names, tensors):
        #image_name = name + str(i) + ".jpg"

        if 'mask_' in name:
            out_img = np.squeeze(tensor[0])
        else:
            out_img = ((np.squeeze(tensor[0]) + 1) * 127.5).astype(np.uint8)

    return out_img

test()