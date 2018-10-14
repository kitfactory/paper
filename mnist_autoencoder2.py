import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

class MnistAutoEncoder():

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32')
        self.x_train = self.x_train / 255
        self.x_test = self.x_test.astype('float32')
        self.x_test = self.x_test / 255
        print(self.x_train.shape)
        print(self.y_test.shape)

    def convert_train_data(self, x):
        x = tf.reshape(x,[1,784])
        # s = tf.shape(x)
        # tf.Print(s,[s],"shape:")
        return x,x

    def convert_to_dataset(self):
        x_const = tf.constant(self.x_train,tf.float32)
        x_train = tf.data.Dataset.from_tensor_slices(x_const)
        self.dataset = x_train.map(map_func=self.convert_train_data)

        x_valid = tf.constant(self.x_test,tf.float32)
        x_valid = tf.data.Dataset.from_tensor_slices(x_valid)
        self.validation = x_valid.map(map_func=self.convert_train_data)
    

    def create_autoencoder_model(self):
        self.model = Sequential()
        # self.model.add(InputLayer(input_shape=(784,)))
        self.model.add(Dense(256,activation='relu',input_dim=784))
        self.model.add(Dense(784,activation='sigmoid'))
        # self.model.compile(optimizer=self.optimizer,loss='binary_crossentropy')
        self.model.compile(optimizer=self.optimizer,loss='mean_absolute_error')

    def create_callback(self):
        self.callback = [
            TensorBoard(),
            ModelCheckpoint('./autoencoder.h5',verbose=1,save_best_only=True,save_weights_only=True),
            # EarlyStopping(patience=50,verbose=1)
        ]
    
    def create_optimizer(self):
        # self.optimizer = tf.keras.optimizers.SGD(lr=0.001)
        # self.optimizer = tf.keras.optimizers.Adadelta()
        self.optimizer = tf.keras.optimizers.Adam()


    def start_training(self):
        dataset = self.dataset.repeat(-1)
        validation = self.validation.repeat(-1)
        self.model.fit(x=dataset,validation_data=validation, verbose=1, steps_per_epoch=1200,validation_steps=200, epochs=800,callbacks=self.callback)

    def load_model_weight(self):
        self.model.load_weights('./autoencoder.h5')

    def show_validation(self):
        iterator = self.validation.shuffle(buffer_size=10000).take(5).make_initializable_iterator()
        next_obj = iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            try:
                while True:
                    x,_ = sess.run(next_obj)
                    y = self.model.predict(x)
                    x = x * 255
                    x = x.astype(np.uint8)
                    x = x.reshape(28,28)
                    plt.imshow(x)
                    plt.gray()
                    plt.show()
                    y = y * 255
                    y = y.astype(np.uint8)
                    y = y.reshape(28,28)
                    plt.imshow(y)
                    plt.gray()
                    plt.show()
            except tf.errors.OutOfRangeError:
                print('end')

if __name__ == '__main__':
    mnist_test = MnistAutoEncoder()
    mnist_test.load_data()
    mnist_test.convert_to_dataset()
    mnist_test.create_optimizer()
    mnist_test.create_autoencoder_model()
    mnist_test.create_callback()
    # mnist_test.load_model_weight()
    # mnist_test.start_training()

    mnist_test.load_model_weight()
    mnist_test.show_validation()
