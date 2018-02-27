from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import scipy
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
import cv2
import matplotlib.pyplot as plt

from eval import compute_map

# import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    N = features["x"].shape[0]
    # input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    with tf.variable_scope("input"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            crop_layer = [tf.image.random_flip_left_right(
                tf.image.random_flip_up_down(
                    tf.random_crop(features["x"][0, :, :, :],
                                   [224, 224, 3])
                ))]
            for i in range(1, N):
                cur_im = tf.image.random_flip_left_right(
                    tf.image.random_flip_up_down(
                        tf.random_crop(features["x"][i, :, :, :],
                                       [224, 224, 3])
                    ))
                crop_layer = tf.concat([crop_layer, [cur_im]], 0)
            # crop_layer = tf.image.resize_images(crop_layer, [256, 256])
            tf.summary.image('training_images', crop_layer)
        else:
            crop_layer = features["x"]
            # crop_layer = tf.image.resize_images(features["x"], [256, 256])
            # crop_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    ##########################
    # 1
    with tf.variable_scope('vgg_16') as scope:
        with tf.variable_scope("conv1"):
            conv1_1 = tf.layers.conv2d(
                name="conv1_1",
                inputs=crop_layer,
                kernel_size=[3, 3],
                strides=1,
                filters=64,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 2
            conv1_2 = tf.layers.conv2d(
                name="conv1_2",
                inputs=conv1_1,
                kernel_size=[3, 3],
                strides=1,
                filters=64,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # max 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

        ##########################
        # 3
        with tf.variable_scope("conv2"):
            conv2_1 = tf.layers.conv2d(
                name="conv2_1",
                inputs=pool1,
                kernel_size=[3, 3],
                strides=1,
                filters=128,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 4
            conv2_2 = tf.layers.conv2d(
                name="conv2_2",
                inputs=conv2_1,
                kernel_size=[3, 3],
                strides=1,
                filters=128,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # max 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

        ##########################
        # 5
        with tf.variable_scope("conv3"):
            conv3_1 = tf.layers.conv2d(
                name="conv3_1",
                inputs=pool2,
                kernel_size=[3, 3],
                strides=1,
                filters=256,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 6
            conv3_2 = tf.layers.conv2d(
                name="conv3_2",
                inputs=conv3_1,
                kernel_size=[3, 3],
                strides=1,
                filters=256,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 7
            conv3_3 = tf.layers.conv2d(
                name="conv3_3",
                inputs=conv3_2,
                kernel_size=[3, 3],
                strides=1,
                filters=256,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # max 2
            pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

        ##########################
        # 8
        with tf.variable_scope("conv4"):
            conv4_1 = tf.layers.conv2d(
                name="conv4_1",
                inputs=pool3,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 9
            conv4_2 = tf.layers.conv2d(
                name="conv4_2",
                inputs=conv4_1,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 10
            conv4_3 = tf.layers.conv2d(
                name="conv4_3",
                inputs=conv4_2,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # max 2
            pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

        ##########################
        # 11
        with tf.variable_scope("conv5"):
            conv5_1 = tf.layers.conv2d(
                name="conv5_1",
                inputs=pool4,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 12
            conv5_2 = tf.layers.conv2d(
                name="conv5_2",
                inputs=conv5_1,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # 13
            conv5_3 = tf.layers.conv2d(
                name="conv5_3",
                inputs=conv5_2,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                # bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
            # max 2
            pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

        # flatten()
        with tf.variable_scope("flatten"):
            pool3_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
        # pool3_flat = tf.reshape(pool3, [int((labels.shape)[0]), -1])

        # fully_connected(4096)
        # relu()
        with tf.variable_scope("fc6"):
            dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                                     activation=tf.nn.relu, name="fc6")

        # dropout(0.5)
        with tf.variable_scope("dropout1"):
            dropout1 = tf.layers.dropout(
                inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # fully_connected(4096)
        # relu()
        with tf.variable_scope("fc7"):
            dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                                     activation=tf.nn.relu, name="fc7")

        # dropout(0.5)
        with tf.variable_scope("dropout2"):
            dropout2 = tf.layers.dropout(
                inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # fully_connected(20)
        # Logits Layer
        with tf.variable_scope("fc8"):
            logits = tf.layers.dense(kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    inputs=dropout2, units=20, name="fc8")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    onehot_labels = labels
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=onehot_labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        tf.summary.scalar('training_loss', loss)
        decay_LR = tf.train.exponential_decay(0.0001, global_step,
                                              1000, 0.5, staircase=True)
        tf.summary.scalar('decay_LR', decay_LR)
        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_LR,
                                               momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)

        # plot histogram of gradients
        train_summary = []
        grads_and_vars = optimizer.compute_gradients(loss)
        # tf.summary.histogram("grad_histogram",grads_and_vars)
        for g, v in grads_and_vars:
            if g is not None:
                # print(format(v.name))
                grad_hist_summary = tf.summary.histogram("grad_histogram".format(v.name)
                                                         , g)
                train_summary.append(grad_hist_summary)
                # sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name)
                #                                        , tf.nn.zero_fraction(g))
                # train_summary.append(sparsity_summary)
        tf.summary.merge(train_summary)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[RestoreHook()])

    # EVAL mode
    tf.summary.scalar('eval_loss', loss)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    # return tf.estimator.EstimatorSpec(
    #     mode=mode, loss=loss)


def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    # Write this function
    H = 256
    W = 256
    crop_px = 224
    fp = data_dir + "/ImageSets/Main/" + split + ".txt"
    with open(fp) as f:
        f_list = f.readlines()
    f_list = [x.strip('\n') for x in f_list]
    N = len(f_list)
    N = 800
    # read images

    EVAL_STEP = 1
    if split != 'test':
        images = np.zeros([N, H, W, 3], np.float32)
        labels = np.zeros([N, 20]).astype(int)
        weights = np.ones([N, 20]).astype(int)
        for i in range(N):
            images[i, :, :, :] = Image.open(data_dir + '/JPEGImages/' + f_list[i]
                                            + '.jpg').resize((W, H), Image.ANTIALIAS)
    else:
        images = np.zeros([int(N / EVAL_STEP), 224, 224, 3], np.float32)
        labels = np.zeros([int(N / EVAL_STEP), 20]).astype(int)
        weights = np.ones([int(N / EVAL_STEP), 20]).astype(int)
        for i in range(int(N / EVAL_STEP)):
            print(str(i) + "/" + str(N))
            images[i, :, :, :] = Image.open(data_dir +'/JPEGImages/'+f_list[i]
                                         +'.jpg').resize((W, H), Image.ANTIALIAS)\
                                                .crop((15,15,239,239))

    # read class labels
    for c_i in range(20):
        class_fp = data_dir + "/ImageSets/Main/" \
                   + CLASS_NAMES[c_i] + "_" + split + ".txt"
        with open(class_fp) as f:
            cls_list = f.readlines()
        cls_list = [x.split() for x in cls_list]
        if split != 'test':
            for im_i in range(N):
                labels[im_i, c_i] = int(int(cls_list[im_i][1]) == 1)
                weights[im_i, c_i] = int(int(cls_list[im_i][1]) != 0)
        else:
            for im_i in range(int(N / EVAL_STEP)):
                labels[im_i, c_i] = int(int(cls_list[im_i][1]) == 1)
                weights[im_i, c_i] = int(int(cls_list[im_i][1]) != 0)

    return images, labels, weights


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


class RestoreHook(tf.train.SessionRunHook):
    # def __init__(self, init_fn):
    #     self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        # if session.run(tf.train.get_or_create_global_step()) == 0:
        #     self.init_fn(session)
        # if session.run(tf.train.get_or_create_global_step()) == 0:
            # self.init_fn(session)
        model_path = "vgg_16.ckpt"
        # restore data
        layers_to_restore = tf.trainable_variables()
        layers_to_restore = layers_to_restore[:-2]

        scopes = [layer.name for layer in layers_to_restore]
        tf.train.init_from_checkpoint(model_path, {s.replace("kernel", "weights") + '/': s
                                                                     + '/' for s in scopes})


def main():
    BATCH_SIZE = 10
    PASCAL_MODEL_DIR = "/tmp/vgg_model_finetune"

    args = parse_args()
    # Load training and eval data
    print("load eval data")
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')
    print("load train data")
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir=PASCAL_MODEL_DIR)
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    print("session is:")
    global sess
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # draw
    total_iters = 4000
    iter = 100
    NUM_ITERS = int(total_iters / iter)
    mAP_writer = tf.summary.FileWriter(PASCAL_MODEL_DIR + '/train', sess.graph)
    # x = np.multiply(range(iter+1),50.0)
    acc_arr = np.multiply(range(iter + 1), 0.0)

    print("start training")
    for i in range(iter):
        pascal_classifier.train(
            steps=NUM_ITERS,
            hooks=[logging_hook],
            input_fn=train_input_fn)
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))

        # draw graph
        summary = tf.Summary(value=[tf.Summary.Value(tag='mean_AP',
                                                     simple_value=np.mean(AP))])
        mAP_writer.add_summary(summary, i)

        # todo: add test loss
        ev = pascal_classifier.evaluate(input_fn=eval_input_fn)
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss',
                                                     simple_value=ev["loss"])])
        mAP_writer.add_summary(summary, i)

        # acc_arr[i+1] = np.mean(AP)

        print("accuracy is: ")
        print(np.mean(AP))

        # plt.clf()
        # fig = plt.figure(1)
        # plt.plot(x, acc_arr)
        # plt.pause(0.0001)
        # fig.savefig("acc_task1_2.png")


if __name__ == "__main__":
    main()
