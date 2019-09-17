import tensorflow as tf
import numpy as np
from glob import glob


INPUT_SIZE = 128

def conv_block(x, filters, kernel_size, is_training):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding='same')(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.elu(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding='same')(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.elu(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, activation=None, padding='same')(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.elu(x)

    return x


def deconv_block(x, filters, kernel_size, inter, is_training):
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation=tf.nn.elu, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, inter])

    x = conv_block(x, filters=filters, kernel_size=kernel_size, is_training=is_training)
    return x


def segmentation_model(input_tensor, is_training):
    x = input_tensor

    x = conv_block(x, 32, 3, is_training)
    inter_1 = x
    x = tf.keras.layers.MaxPool2D()(x)

    x = conv_block(x, 64, 3, is_training)
    inter_2 = x
    x = tf.keras.layers.MaxPool2D()(x)

    x = conv_block(x, 128, 5, is_training)
    inter_3 = x
    x = tf.keras.layers.MaxPool2D()(x)

    x = conv_block(x, 256, 5, is_training)
    inter_4 = x
    x = tf.keras.layers.MaxPool2D()(x)

    x = conv_block(x, 512, 7, is_training)

    x = deconv_block(x, 256, 5, inter_4, is_training)
    x = deconv_block(x, 128, 5, inter_3, is_training)
    x = deconv_block(x, 64, 3, inter_2, is_training)
    x = deconv_block(x, 32, 3, inter_1, is_training)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding='same')(x)
    return x


def data_loader(batch_size):
    x_paths = list(sorted(glob('./original-cropped/*.jpg')))
    y_paths = list(sorted(glob('./xor_shift/*.png')))

    def preprocess_image(img, channels):
        img = tf.image.decode_image(img, channels=channels)
        # img = tf.image.pad_to_bounding_box(img, 0, 0, 512, 512)
        img = tf.cast(img, tf.float32)
        img /= 255.0
        return img

    def load_x(path):
        img = tf.read_file(path)
        img = preprocess_image(img, 3)
        return img

    def load_y(path):
        img = tf.read_file(path)
        return preprocess_image(img, 1)

    def random_crop(x, y):
        combined = tf.concat([x, y], axis=-1)
        crop = tf.image.random_crop(combined, [INPUT_SIZE, INPUT_SIZE, 4])
        return (crop[:, :, :3], crop[:, :, 3:])

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    assert len(x_paths) == len(y_paths)
    image_count = len(x_paths)

    train_count = int(image_count * 0.9)

    # print(x_paths[train_count:])
    # print(y_paths[train_count:])

    x_path_ds = tf.data.Dataset.from_tensor_slices(x_paths)
    x_images = x_path_ds.map(load_x, num_parallel_calls=AUTOTUNE)
    y_path_ds = tf.data.Dataset.from_tensor_slices(y_paths)
    y_images = y_path_ds.map(load_y, num_parallel_calls=AUTOTUNE)

    ds = tf.data.Dataset.zip((x_images, y_images))
    # ds = ds.shuffle()
    train_ds = ds.take(train_count)
    test_ds = ds.skip(train_count)

    train_ds = train_ds.map(random_crop, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=train_count))
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    # test_ds = test_ds.map(random_crop, num_parallel_calls=AUTOTUNE)
    # test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, test_ds


def divide_image(x_image):
    xs = list()

    pad_width = ((INPUT_SIZE // 2) - (x_image.shape[0] % (INPUT_SIZE // 2))) % (INPUT_SIZE // 2)
    pad_height = ((INPUT_SIZE // 2) - (x_image.shape[1] % (INPUT_SIZE // 2))) % (INPUT_SIZE // 2)

    x_image = np.pad(x_image, ((0, pad_width), (0, pad_height), (0, 0)), 'constant', constant_values=0)

    for i in range(0, x_image.shape[0] - INPUT_SIZE // 2, INPUT_SIZE // 2):
        for j in range(0, x_image.shape[1] - INPUT_SIZE // 2, INPUT_SIZE // 2):
            x_crop = x_image[i:i + INPUT_SIZE, j:j + INPUT_SIZE, :]
            xs.append(x_crop)
    return np.stack(xs)


def merge_image(images, shape):
    width = ((INPUT_SIZE // 2) - (shape[0] % (INPUT_SIZE // 2))) % (INPUT_SIZE // 2)
    height = ((INPUT_SIZE // 2) - (shape[1] % (INPUT_SIZE // 2))) % (INPUT_SIZE // 2)
    mask = np.zeros(shape=(shape[0] + width, shape[1] + height, 1))
    
    idx = 0
    for i in range(0, shape[0] - INPUT_SIZE // 2, INPUT_SIZE // 2):
        for j in range(0, shape[1] - INPUT_SIZE // 2, INPUT_SIZE // 2):
            image = images[idx]
            image = image / (np.amax(image) - np.amin(image))
            mask[i:i + INPUT_SIZE, j:j + INPUT_SIZE, :] += image
            idx += 1
    return (mask[:shape[0], :shape[1], :] - np.amin(mask)) / (np.amax(mask) - np.amin(mask))


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 1])
    is_training = tf.placeholder(dtype=tf.bool, shape=())
    lr_ph = tf.placeholder(dtype=tf.float32, shape=())

    output = segmentation_model(x, is_training)
    out_mask = tf.nn.sigmoid(output)
    
    focus = 15.0
    mse = tf.losses.mean_squared_error(labels=y, predictions=out_mask, reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(tf.multiply(y * (focus - 1.0) + 1.0, mse))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_ph)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = optimizer.minimize(loss)
    train_step = tf.group([train_step, update_ops])

    batch_size = 8
    train_ds, test_ds = data_loader(batch_size)
    train_ds_iter = train_ds.make_one_shot_iterator().get_next()
    test_ds_iter = test_ds.make_initializable_iterator()

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())

    train_count = int(945 * 0.9)

    best_loss = 1000.

    for epoch in range(50):
        if epoch < 15:
            lr_value = 1e-2
        elif epoch < 40:
            lr_value = 1e-3
        else:
            lr_value = 1e-4

        print('Epoch {}'.format(epoch + 1))

        train_preds = list()
        train_loss = list()
        for _ in range(train_count // batch_size):
            x_batch, y_batch = sess.run(train_ds_iter)
            loss_value, out_img, _ = sess.run([loss, out_mask, train_step], feed_dict={x: x_batch, y: y_batch, is_training: True, lr_ph: lr_value})
            train_loss.append(loss_value)
            train_preds.append(out_img)

        test_loss = list()
        out_images = list()
        label_images = list()

        sess.run(test_ds_iter.initializer)
        test_batch = test_ds_iter.get_next()
        while True:
            try:
                x_image, y_image = sess.run(test_batch)
                x_batch, y_batch = divide_image(x_image), divide_image(y_image)
                loss_value, out_img = sess.run([loss, out_mask], feed_dict={x: x_batch, y: y_batch, is_training: False})
                test_loss.append(loss_value)
                # out_images.append(merge_image(out_img, x_image.shape))
                # label_images.append(y_batch)
            except tf.errors.OutOfRangeError:
                print('Train loss: {}\tTest loss :{}'.format(np.mean(train_loss), np.mean(test_loss)))
                # np.save('output-{}.npy'.format(epoch), np.concatenate(out_images, axis=0))
                if np.mean(test_loss) < best_loss:
                    best_loss = np.mean(test_loss)
                    saver.save(sess, 'segmentation-crop-{}.ckpt'.format(INPUT_SIZE), global_step=epoch)
                break
