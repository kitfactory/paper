import tensorflow as tf
import tfrecord_util as TFUtil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_model():
    input_layer = tf.keras.layers.Input(shape=(96,96,3,))
    layers = tf.keras.layers.Conv2D(64,3,strides=(2,2),name='conv2d-1',padding='same')(input_layer) # 24*24*6
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2D(16,3,strides=(2,2),name='conv2d-2',padding='same')(layers) # 24*24*6
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2D(8,3,strides=(2,2),name='conv2d-3',padding='same')(layers) # 24*24*6
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2DTranspose(16,3,strides=(2,2),name='transpose-1',padding='same')(layers) #48*48*12
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2DTranspose(32,3,strides=(2,2),name='transpose-2',padding='same')(layers) #48*48*12
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2DTranspose(16,3,strides=(2,2),name='transpose-3',padding='same')(layers) #48*48*12
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2D(6,3,strides=(1,1),name='pre-final',padding='same')(layers)
    final_layer = tf.keras.layers.Conv2D(3,3,strides=(1,1),name='final',padding='same')(layers)
    # final_layer = tf.keras.layers.LeakyReLU()(layers)

    """
    layers = tf.keras.layers.Conv2D(6,3,strides=(2,2),name='conv2d-2')(layers) # 24*24*6
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2D(3,3,strides=(2,2),name='conv2d-3')(layers) # 12*12*3
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2DTranspose(6,3,strides=(2,2),name='transpose-1')(layers) #24*24*3
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2DTranspose(12,3,strides=(2,2),name='transpose-2')(layers) #48*48*12
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2DTranspose(3,3,strides=(2,2),name='transpose-3')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    final_layer = tf.keras.layers.Conv2D(3,3,strides=(1,1),name='final')(layers)
    """
    model = tf.keras.models.Model(input_layer, final_layer)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model
    

#    layers = tf.keras.layers.Flatten()(layers)
#    layers = tf.keras.layers.Dense(32)

def image_fn(image,label):
    image = tf.reshape(image,(1,96,96,3))
    # tf.Print(image,[image],"size of image:")
    return image,image

def main():
    loader = TFUtil.TFRecordLoader()
    dataset = loader.load_dataset('./dogs.tfrecord',96,3)
    dataset = dataset.map(image_fn)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(12500)
    model = create_model()
    

    iterator = dataset.skip(1).take(20).make_initializable_iterator()
    next_obj = iterator.get_next()
    i = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        model.load_weights('dogs_autoencorder.h5')
        while True:
            image, _ = sess.run(next_obj)
            p = model.predict(image)
            filename = "figure{}.png".format(i)
            i += 1
            img = Image.fromarray(np.uint8(p[0] * 255))
            img.save(filename)
            # plt.imshow((p[0] * 255).astype(np.uint8))
            # plt.show()
    model.load_weights('dogs_autoencorder.h5')
    """
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath='dogs_autoencorder.h5', monitor='loss', save_best_only=True,save_weights_only=True,verbose=1)
        # tf.keras.callbacks.ProgbarLogger()
    ]
    model.fit(dataset,callbacks=callbacks,steps_per_epoch=250,epochs=2000)
    """


if __name__ == '__main__':
    main()
