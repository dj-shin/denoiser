import tensorflow as tf
import numpy as np

from segmentation import segmentation_model, divide_image, merge_image


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
    output = segmentation_model(x, False)
    out_mask = tf.nn.sigmoid(output)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    saver.restore(sess, './segmentation-crop-128.ckpt-44')

    images = list()
    for i in range(4):
        fname = './cut{}.jpg'.format(i + 1)     # test images
        img = tf.read_file(fname)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img /= 255.0

        image = sess.run(img)
        print(image.shape)
        cropped = divide_image(image)
        print(cropped.shape)
        result = sess.run(out_mask, feed_dict={x: np.stack(cropped, axis=0)})
        print(result.shape)
        np.save('cut{}.npy'.format(i + 1), merge_image(result, image.shape))
