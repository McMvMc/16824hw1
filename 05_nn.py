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
    # with tf.variable_scope('pool3') as scope:
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
        # scope.reuse_variables()
        # weights = tf.get_variable('/weights')
        # tf.summary.image('pool3/weights', weight_2_grid(pool3))
    # tf.summary.image('pool3/weights', pool3)


    pool3_flat = tf.reshape(pool3, [-1, 5*5*256])

    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # with tf.variable_scope('dense2') as scope:
    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                        activation=tf.nn.relu)
        # scope.reuse_variables()
        # weights = tf.get_variable('weights')
        # tf.summary.image('dense2/weights', weight_2_grid(weights))
    # tf.summary.image('dense2/weights', dense2)

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

    if split != 'test':
        N = 10

    images = np.zeros([int(N), 224, 224, 3], np.float32)
    labels = np.zeros([int(N), 20]).astype(int)
    weights = np.ones([int(N), 20]).astype(int)

    if split == 'test':
        for i in range(int(N)):
            print(str(i) + "/" + str(N))
            images[i, :, :, :] = Image.open(data_dir +'/JPEGImages/'+f_list[i]
                                         +'.jpg').resize((crop_px, crop_px), Image.ANTIALIAS)
        for c_i in range(20):
            class_fp = data_dir + "/ImageSets/Main/" \
                       + CLASS_NAMES[c_i] + "_" + split + ".txt"
            with open(class_fp) as f:
                cls_list = f.readlines()
            cls_list = [x.split() for x in cls_list]

            for im_i in range(int(N)):
                labels[im_i, c_i] = int(int(cls_list[im_i][1]) == 1)
                weights[im_i, c_i] = int(int(cls_list[im_i][1]) != 0)

        return images, labels, weights

    # read class labels
    for c_i in range(10):
        class_fp = data_dir + "/ImageSets/Main/" \
                   + CLASS_NAMES[c_i] + "_" + split + ".txt"
        with open(class_fp) as f:
            cls_list = f.readlines()
        cls_list = [x.split() for x in cls_list]

        for im_i in range(len(cls_list)):
            if int(cls_list[im_i][1]) == 1:
                print(CLASS_NAMES[c_i]+":"+cls_list[im_i][0] + "/" + str(N))
                images[c_i, :, :, :] = Image.open(data_dir + '/JPEGImages/' + cls_list[im_i][0]
                                        + '.jpg').resize((crop_px, crop_px), Image.ANTIALIAS)
                labels[c_i, c_i] = int(int(cls_list[im_i][1]) == 1)
                weights[c_i, c_i] = int(int(cls_list[im_i][1]) != 0)
                break

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

def read_mean_rgb(model_dir):
    mean_rgb = tf.train.NewCheckpointReader(model_dir)\
                        .get_tensor("vgg_16/mean_rgb")
    return mean_rgb

def activation_2_grid(activations, px, r, c):
    image = np.zeros((px*r,px*c),dtype=np.float32)
    max = np.amax(activations)
    for y in range(r):
        for x in range(c):
            image[y*px:(y+1)*px ,x*px:(x+1)*px] = activations[:,:,x+y*c]

    return image/max

def main():
    VGG_DIR = "/tmp/05_nn_vgg"
    ALEX_DIR = "/tmp/05_nn_alex"
    model_path = "vgg_16.ckpt"


    args = parse_args()
    # Load training and eval data
    print("load eval data")
    ref_data, ref_labels, ref_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    eval_input_fn_ref = tf.estimator.inputs.numpy_input_fn(
        x={"x": ref_data, "w": ref_weights},
        y=ref_labels,
        num_epochs=1,
        shuffle=False)

    vgg_classifier = tf.estimator.Estimator(
        model_fn=partial(vgg_fn,
                         num_classes=20),
        model_dir=VGG_DIR)
    alex_classifier = tf.estimator.Estimator(
        model_fn=partial(alex_fn,
                         num_classes=20),
        model_dir=ALEX_DIR)

    mean_rgb = read_mean_rgb(model_path)
    eval_data_vgg = eval_data-mean_rgb

    eval_input_fn_vgg = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data_vgg, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_input_fn_alex = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    N = eval_labels.shape[0]

    # VGG eval
    ref_vgg = list(vgg_classifier.predict(input_fn=eval_input_fn_ref))
    pred_vgg = list(vgg_classifier.predict(input_fn=eval_input_fn_vgg))
    nn_vgg_fc = np.zeros([10]).astype(int)
    nn_vgg_pool = np.zeros([10]).astype(int)
    inf = float('inf')
    for ref_i in range(10):
        min_fc = inf
        min_pool = inf
        cur_ref_fc7 = ref_vgg[ref_i]['fc7']
        cur_ref_pool = ref_vgg[ref_i]['pool5']
        for i in range(1,N):
            cur_fc7 = pred_vgg[i]['fc7']
            cur_dist_fc = np.linalg.norm(cur_ref_fc7-cur_fc7)
            if cur_dist_fc < min_fc:
                nn_vgg_fc[ref_i] = i
                min_fc = cur_dist_fc
            cur_pool = pred_vgg[i]['pool5']
            cur_dist_pool = np.linalg.norm(cur_ref_pool - cur_pool)
            if cur_dist_pool < min_pool:
                nn_vgg_pool[ref_i] = i
                min_pool = cur_dist_pool

    # w = 10
    # h = 10
    fc_fig = plt.figure(1)
    fc_fig.suptitle("vgg fc+reference response", fontsize=16)
    plt.show()
    pool_fig = plt.figure(2)
    pool_fig.suptitle("vgg pooling+reference response", fontsize=16)
    plt.show()
    nn_fig = plt.figure(3)
    nn_fig.suptitle("vgg nearest neighbors+reference", fontsize=16)
    plt.show()
    columns = 8
    rows = 3
    for i in range(10):
        plt.figure(1)
        nn_fc = nn_vgg_fc[i]
        nn_pool = nn_vgg_pool[i]
        fc_img = Image.fromarray(np.reshape(
            pred_vgg[nn_fc]['fc7']/np.amax(pred_vgg[nn_fc]['fc7'])*255,
                                            (64, 64)))
        fc_img_ref = Image.fromarray(np.reshape(
            ref_vgg[i]['fc7']/np.amax(ref_vgg[i]['fc7'])*255,
                                            (64, 64)))
        ax = fc_fig.add_subplot(rows, columns, i*2+1)
        ax.set_title(CLASS_NAMES[i])
        plt.imshow(fc_img)
        ax = fc_fig.add_subplot(rows, columns, i*2+2)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(fc_img_ref)

        plt.figure(2)
        pool_img = activation_2_grid(pred_vgg[nn_pool]['pool5'], 7, 16, 32)
        pool_img_ref = activation_2_grid(ref_vgg[i]['pool5'], 7, 16, 32)
        ax = pool_fig.add_subplot(rows, columns, i*2+1)
        ax.set_title(CLASS_NAMES[i])
        plt.imshow(pool_img)
        ax = pool_fig.add_subplot(rows, columns, i*2+2)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(pool_img_ref)

        plt.figure(3)
        ax = nn_fig.add_subplot(4, 10, i*2 + 1)
        ax.set_title("FC: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(eval_data[nn_vgg_fc[i]].astype(np.uint8)))
        ax = nn_fig.add_subplot(4, 10, i*2 + 2)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(ref_data[i].astype(np.uint8)))

        ax = nn_fig.add_subplot(4, 10, i*2 + 1 + 20)
        ax.set_title("POOLING: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(eval_data[nn_vgg_pool[i]].astype(np.uint8)))
        ax = nn_fig.add_subplot(4, 10, i*2 + 2 + 20)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(ref_data[i].astype(np.uint8)))


    ALEX eval
    ref_alex = list(alex_classifier.predict(input_fn=eval_input_fn_ref))
    pred_alex = list(alex_classifier.predict(input_fn=eval_input_fn_alex))
    nn_alex_fc = np.zeros([10]).astype(int)
    nn_alex_pool = np.zeros([10]).astype(int)
    inf = float('inf')
    for ref_i in range(10):
        min_fc = inf
        min_pool = inf
        cur_ref_fc7 = ref_alex[ref_i]['fc7']
        cur_ref_pool = ref_alex[ref_i]['pool5']
        for i in range(1,N):
            cur_fc7 = pred_alex[i]['fc7']
            cur_dist_fc = np.linalg.norm(cur_ref_fc7-cur_fc7)
            if cur_dist_fc < min_fc:
                nn_alex_fc[ref_i] = i
                min_fc = cur_dist_fc
            cur_pool = pred_alex[i]['pool5']
            cur_dist_pool = np.linalg.norm(cur_ref_pool - cur_pool)
            if cur_dist_pool < min_pool:
                nn_alex_pool[ref_i] = i
                min_pool = cur_dist_pool

    fc_fig = plt.figure(4)
    fc_fig.suptitle("alex fc+reference response", fontsize=16)
    plt.show()
    pool_fig = plt.figure(5)
    pool_fig.suptitle("alex pooling+reference response", fontsize=16)
    plt.show()
    nn_fig = plt.figure(6)
    nn_fig.suptitle("alex nearest neighbors+reference", fontsize=16)
    plt.show()
    columns = 8
    rows = 3
    for i in range(10):
        plt.figure(4)
        nn_fc = nn_alex_fc[i]
        nn_pool = nn_alex_pool[i]
        fc_img = Image.fromarray(np.reshape(
            pred_alex[nn_fc]['fc7']/np.amax(pred_alex[nn_fc]['fc7'])*255,
                                            (64, 64)))
        fc_img_ref = Image.fromarray(np.reshape(
            ref_alex[i]['fc7']/np.amax(ref_alex[i]['fc7'])*255,
                                            (64, 64)))
        ax = fc_fig.add_subplot(rows, columns, i*2+1)
        ax.set_title(CLASS_NAMES[i])
        plt.imshow(fc_img)
        ax = fc_fig.add_subplot(rows, columns, i*2+2)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(fc_img_ref)

        plt.figure(5)
        pool_img = activation_2_grid(pred_alex[nn_pool]['pool5'], 5, 16, 16)
        pool_img_ref = activation_2_grid(ref_alex[i]['pool5'], 5, 16,16)
        ax = pool_fig.add_subplot(rows, columns, i*2+1)
        ax.set_title(CLASS_NAMES[i])
        plt.imshow(pool_img)
        ax = pool_fig.add_subplot(rows, columns, i*2+2)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(pool_img_ref)

        plt.figure(6)
        ax = nn_fig.add_subplot(4, 10, i*2 + 1)
        ax.set_title("FC: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(eval_data[nn_alex_fc[i]].astype(np.uint8)))
        ax = nn_fig.add_subplot(4, 10, i*2 + 2)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(ref_data[i].astype(np.uint8)))

        ax = nn_fig.add_subplot(4, 10, i*2 + 1 + 20)
        ax.set_title("POOLING: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(eval_data[nn_alex_pool[i]].astype(np.uint8)))
        ax = nn_fig.add_subplot(4, 10, i*2 + 2 + 20)
        ax.set_title("REF: " + CLASS_NAMES[i])
        plt.imshow(Image.fromarray(ref_data[i].astype(np.uint8)))

    return
if __name__ == "__main__":
    main()