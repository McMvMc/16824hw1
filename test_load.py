import tensorflow as tf

with tf.variable_scope("conv1"):
    conv1 = tf.layers.conv2d(
        inputs=crop_layer,
        kernel_size=[3, 3],
        strides=1,
        filters=64,
        padding="same",
        # kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        activation=tf.nn.relu)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "vgg_16.ckpt")



print("conv1 : %s" % conv1.eval())