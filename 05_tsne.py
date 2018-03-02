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
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
import matplotlib.patches as mpatches

from eval import compute_map

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


def weight_2_grid(kernel):

    n = kernel.get_shape().as_list()[3]
    grid_Y = int(np.ceil(sqrt(n)))
    grid_X = int(np.ceil(n/grid_Y))

    print ('grid: %d = (%d, %d)' % (n, grid_Y, grid_X))

    # scaling
    min_val = tf.reduce_min(kernel)
    max_val = tf.reduce_max(kernel)
    kernel = (kernel-min_val)/(max_val-min_val)

    Y = kernel.get_shape().as_list()[0]
    X = kernel.get_shape().as_list()[1]

    n_chan = 3

    x = kernel
    for i in range(grid_X*grid_Y-n):
        x = tf.concat([x,tf.transpose(
                            [tf.zeros([Y,X,n_chan])], (1, 2, 3, 0))],3)

    x = tf.transpose(x, (3, 0, 1, 2))
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, n_chan]))

    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, n_chan]))

    x = tf.transpose(x, (2, 1, 3, 0))
    x = tf.transpose(x, (3, 0, 1, 2))

    return x

def vgg_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    N = features["x"].shape[0]

    crop_layer = features["x"]

    # build model
    with tf.variable_scope('vgg_16'):
        with tf.variable_scope("conv1"):
            conv1_1 = tf.layers.conv2d(
                name="conv1_1",
                inputs=crop_layer,
                kernel_size=[3, 3],
                strides=1,
                filters=64,
                padding="same",
                activation=tf.nn.relu)
            conv1_2 = tf.layers.conv2d(
                name="conv1_2",
                inputs=conv1_1,
                kernel_size=[3, 3],
                strides=1,
                filters=64,
                padding="same",
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

        with tf.variable_scope("conv2"):
            conv2_1 = tf.layers.conv2d(
                name="conv2_1",
                inputs=pool1,
                kernel_size=[3, 3],
                strides=1,
                filters=128,
                padding="same",
                activation=tf.nn.relu)
            conv2_2 = tf.layers.conv2d(
                name="conv2_2",
                inputs=conv2_1,
                kernel_size=[3, 3],
                strides=1,
                filters=128,
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

        with tf.variable_scope("conv3"):
            conv3_1 = tf.layers.conv2d(
                name="conv3_1",
                inputs=pool2,
                kernel_size=[3, 3],
                strides=1,
                filters=256,
                padding="same",
                activation=tf.nn.relu)
            conv3_2 = tf.layers.conv2d(
                name="conv3_2",
                inputs=conv3_1,
                kernel_size=[3, 3],
                strides=1,
                filters=256,
                padding="same",
                activation=tf.nn.relu)
            conv3_3 = tf.layers.conv2d(
                name="conv3_3",
                inputs=conv3_2,
                kernel_size=[3, 3],
                strides=1,
                filters=256,
                padding="same",
                activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

        with tf.variable_scope("conv4"):
            conv4_1 = tf.layers.conv2d(
                name="conv4_1",
                inputs=pool3,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                activation=tf.nn.relu)
            conv4_2 = tf.layers.conv2d(
                name="conv4_2",
                inputs=conv4_1,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                activation=tf.nn.relu)
            conv4_3 = tf.layers.conv2d(
                name="conv4_3",
                inputs=conv4_2,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

        with tf.variable_scope("conv5"):
            conv5_1 = tf.layers.conv2d(
                name="conv5_1",
                inputs=pool4,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                activation=tf.nn.relu)
            conv5_2 = tf.layers.conv2d(
                name="conv5_2",
                inputs=conv5_1,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                activation=tf.nn.relu)
            conv5_3 = tf.layers.conv2d(
                name="conv5_3",
                inputs=conv5_2,
                kernel_size=[3, 3],
                strides=1,
                filters=512,
                padding="same",
                activation=tf.nn.relu)
            pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

        with tf.variable_scope("flatten"):
            pool3_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])

        fc6 = tf.layers.dense(inputs=pool3_flat, units=4096,
                                     activation=tf.nn.relu, name="fc6")

        with tf.variable_scope("dropout1"):
            dropout1 = tf.layers.dropout(
                inputs=fc6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        fc7 = tf.layers.dense(inputs=dropout1, units=4096,
                                     activation=tf.nn.relu, name="fc7")

        with tf.variable_scope("dropout2"):
            dropout2 = tf.layers.dropout(
                inputs=fc7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        fc8 = tf.layers.dense(kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    inputs=dropout2, units=20, name="fc8")

    # output & loss
    predictions = {
        "classes": tf.argmax(input=fc8, axis=1),
        "probabilities": tf.sigmoid(fc8, name="sigmoid_tensor"),
        "fc7": fc7,
        "pool5": pool5
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = labels
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=onehot_labels, logits=fc8), name='loss')

    # EVAL mode
    tf.summary.scalar('eval_loss', loss)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def read_mean_rgb(model_dir):
    mean_rgb = tf.train.NewCheckpointReader(model_dir)\
                        .get_tensor("vgg_16/mean_rgb")
    return mean_rgb

def alex_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    N = features["x"].shape[0]

    # build model
    crop_layer = tf.image.resize_images(features["x"], [224,224])

    conv1 = tf.layers.conv2d(
        inputs=crop_layer,
        kernel_size=[11, 11],
        strides=4,
        filters=96,
        padding="valid",
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        kernel_size=[5, 5],
        strides=1,
        filters=256,
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        kernel_size=[3, 3],
        strides=1,
        filters=384,
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        kernel_size=[3, 3],
        strides=1,
        filters=384,
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        kernel_size=[3, 3],
        strides=1,
        filters=256,
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 5*5*256])

    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                        activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=20)

    # output & loss
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor"),
        "fc7":dense2,
        "pool5":pool3
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = labels
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=onehot_labels, logits=logits), name='loss')

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def load_pascal(data_dir, split='test'):

    # Write this function
    H = 256
    W = 256
    crop_px = 224
    fp = data_dir + "/ImageSets/Main/" + split + ".txt"
    with open(fp) as f:
        f_list = f.readlines()
    f_list = [x.strip('\n') for x in f_list]
    N = len(f_list)

    images = np.zeros([1000, 224, 224, 3], np.float32)
    labels = np.zeros([1000, 20]).astype(int)
    weights = np.ones([1000, 20]).astype(int)

    for i in range(int(1000)):
        print(str(i*3) + "/" + str(N))
        images[i, :, :, :] = Image.open(data_dir +'/JPEGImages/'+f_list[i*3]
                                     +'.jpg').resize((crop_px, crop_px), Image.ANTIALIAS)
    for c_i in range(20):
        class_fp = data_dir + "/ImageSets/Main/" \
                   + CLASS_NAMES[c_i] + "_" + split + ".txt"
        with open(class_fp) as f:
            cls_list = f.readlines()
        cls_list = [x.split() for x in cls_list]

        for im_i in range(int(1000)):
            labels[im_i, c_i] = int(int(cls_list[im_i*3][1]) == 1)
            weights[im_i, c_i] = int(int(cls_list[im_i*3][1]) != 0)

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

def main():
    VGG_DIR = "/tmp/05_nn_vgg"
    ALEX_DIR = "/tmp/05_nn_alex"
    model_path = "vgg_16.ckpt"

    mean_rgb = read_mean_rgb(model_path)

    color_gen = lambda: random.randint(0, 255)
    colors= np.zeros((20,3)).astype(int)
    for i in range(20):
        colors[i, 0] = color_gen()
        colors[i, 1] = color_gen()
        colors[i, 2] = color_gen()


    args = parse_args()
    # Load training and eval data
    print("load eval data")
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')
    eval_data_vgg = eval_data - mean_rgb

    alex_classifier = tf.estimator.Estimator(
        model_fn=partial(alex_fn,
                         num_classes=20),
        model_dir=ALEX_DIR)
    vgg_classifier = tf.estimator.Estimator(
        model_fn=partial(vgg_fn,
                         num_classes=20),
        model_dir=VGG_DIR)

    eval_input_fn_alex = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_input_fn_vgg = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data_vgg, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    N = eval_labels.shape[0]

    # ALEX eval
    pred_alex = list(alex_classifier.predict(input_fn=eval_input_fn_alex))
    X_alex = [sample['fc7'] for sample in pred_alex]

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(X_alex)

    print('ALEX variation (PCA): {}'
          .format(np.sum(pca_50.explained_variance_ratio_)))

    X_alex = TSNE(n_components=2,
                      verbose=1, perplexity=40)\
                        .fit_transform(pca_result_50)

    # VGG eval
    pred_vgg = list(vgg_classifier.predict(input_fn=eval_input_fn_vgg))
    X_vgg = [sample['fc7'] for sample in pred_vgg]
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(X_vgg)

    print('VGG variation (PCA): {}'
          .format(np.sum(pca_50.explained_variance_ratio_)))

    X_vgg_pca = TSNE(n_components=2,
                      verbose=1, perplexity=40)\
                        .fit_transform(pca_result_50)
    X_vgg = TSNE(n_components=2,
                      verbose=1, perplexity=40)\
                        .fit_transform(X_vgg)


    c_cls = np.array([ colors[cls.astype(bool),:].mean(axis=0) for cls in eval_labels])
    alex_fig = plt.figure(1)
    alex_fig.suptitle("alexnet with pca-50", fontsize=16)
    vgg_fig = plt.figure(2)
    vgg_fig.suptitle("vgg without pca-50", fontsize=16)
    vgg_pca_fig = plt.figure(3)
    vgg_pca_fig.suptitle("vgg with pca-50", fontsize=16)
    for i in range(N):
        i
        plt.figure(1)
        plt.scatter(X_alex[i,0], X_alex[i,1],
                c=('#%02X' % int(c_cls[i,0]))
                  +('%02X' % int(c_cls[i,1]))
                  +('%02X' % int(c_cls[i,2])))
        # plt.legend((a), (CLASS_NAMES[i]))

        plt.figure(2)
        plt.scatter(X_vgg[i,0], X_vgg[i,1],
                c=('#%02X' % int(c_cls[i,0]))
                  +('%02X' % int(c_cls[i,1]))
                  +('%02X' % int(c_cls[i,2])))
        # plt.legend((b), (CLASS_NAMES[i]))

        plt.figure(3)
        plt.scatter(X_vgg_pca[i, 0], X_vgg_pca[i, 1],
                    c=('#%02X' % int(c_cls[i, 0]))
                      + ('%02X' % int(c_cls[i, 1]))
                      + ('%02X' % int(c_cls[i, 2])))
        # plt.legend((c), (CLASS_NAMES[i]))

    # legend = []
    # for i in range(20):
    #     legend = np.concatenate((legend,
    #                  [mpatches.Patch(color=('#%02X' % int(c_cls[i,0]))
    #               +('%02X' % int(c_cls[i,1]))
    #               +('%02X' % int(c_cls[i,2])), label=CLASS_NAMES[i])]))
    classes = CLASS_NAMES
    class_colours = c_cls/255
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))

    plt.figure(1)
    plt.legend(recs, classes, loc=4)
    plt.show()
    plt.figure(2)
    plt.legend(recs, classes, loc=4)
    plt.show()
    plt.figure(3)
    plt.legend(recs, classes, loc=4)
    plt.show()

    return
if __name__ == "__main__":
    main()