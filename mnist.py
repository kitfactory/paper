import tensorflow as tf

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


class MnistClassifyTest():

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32')
        self.x_train = self.x_train / 255
        self.x_test = self.x_test.astype('float32')
        self.x_test = self.x_test / 255
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)
        print(self.y_test.dtype)

    def convert_one_hot(self, x, y):
        label = tf.one_hot(y,10)
        label = tf.reshape(label,[1,10])
        x = tf.reshape(x, [1,28,28,1])
        return x,label

    def convert_to_dataset(self):
        x_const = tf.constant(self.x_train,tf.float32)
        y_const = tf.constant(self.y_train,tf.uint8)
        x_train = tf.data.Dataset.from_tensor_slices(x_const)
        y_train = tf.data.Dataset.from_tensor_slices(y_const)
        zipped = tf.data.Dataset.zip((x_train,y_train))
        self.dataset = zipped.map(map_func=self.convert_one_hot)

        x_valid = tf.constant(self.x_test,tf.float32)
        y_valid = tf.constant(self.y_test,tf.uint8)
        x_valid = tf.data.Dataset.from_tensor_slices(x_valid)
        y_valid = tf.data.Dataset.from_tensor_slices(y_valid)
        valid_zipped = tf.data.Dataset.zip((x_valid,y_valid))
        self.validation = valid_zipped.map(map_func=self.convert_one_hot)
    
    def create_classify_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

    def create_callback(self):
        self.callback = [
            TensorBoard(),
            ModelCheckpoint('mnist.-{val_loss:.5f}.hdf5'),
            EarlyStopping(patience=10,verbose=1)
        ]
    
    def create_optimizer(self):
        # self.optimizer = tf.keras.optimizers.SGD(lr=0.001)
        self.optimizer = tf.keras.optimizers.Adam()

    def start_classify_training(self):
        dataset = self.dataset.repeat(-1)
        validation = self.validation.repeat(-1)
        self.model.compile(optimizer=self.optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=dataset,validation_data=validation, verbose=1, steps_per_epoch=1200,validation_steps=200, epochs=1000,callbacks=self.callback)

    def run_dataset(self):
        iterator = self.dataset.make_initializable_iterator()
        next_obj = iterator.get_next()
        with tf.Session() as sess:
            sess.run(iterator.initializer)
            try:
                obj = sess.run(next_obj)
            except tf.errors.OutOfRangeError:
                print('end')

if __name__ == '__main__':
    mnist_test = MnistClassifyTest()
    mnist_test.load_data()
    mnist_test.convert_to_dataset()
    mnist_test.create_classify_model()
    mnist_test.create_callback()
    mnist_test.create_optimizer()
    mnist_test.start_classify_training()

