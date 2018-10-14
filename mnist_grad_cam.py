import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

class MnistCnnGradCam():

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
    
    def create_cnn_model(self):
        input_layer = tf.keras.layers.Input(shape=(28,28,1),name='input')
        model = tf.keras.layers.Conv2D(16,(3,3),1,activation='elu',padding='same',name='conv2d-1')(input_layer)
        model = tf.keras.layers.BatchNormalization(name='bn-1')(model)
        model = tf.keras.layers.Conv2D(16,(3,3),1,activation='elu',padding='same',name='conv2d-2')(model)
        model = tf.keras.layers.BatchNormalization(name='bn-2')(model)
        model = tf.keras.layers.Conv2D(16,(3,3),1,activation='elu',padding='same',name='conv2d-3')(model)
        model = tf.keras.layers.BatchNormalization(name='bn-3')(model)
        model = tf.keras.layers.Conv2D(16,(3,3),1,activation='elu',padding='same',name='conv2d-4')(model)
        model = tf.keras.layers.BatchNormalization(name='bn-4')(model)
        model = tf.keras.layers.Flatten(name='flatten')(model)
        output_layer = tf.keras.layers.Dense(10, name='dense', activation='softmax')(model)
        # self.model.compile(optimizer=self.optimizer,loss='binary_crossentropy')
        keras_model = tf.keras.Model(input_layer, output_layer)
        keras_model.summary()
        self.model = keras_model
        self.model.compile(optimizer=self.optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

    def create_callback(self):
        self.callback = [
            TensorBoard(),
            ModelCheckpoint('./autoencoder_cnn_gradcam.h5',verbose=1,save_best_only=True,save_weights_only=True, monitor='val_acc'),
            # EarlyStopping(patience=50,verbose=1)
        ]
    
    def create_optimizer(self):
        # self.optimizer = tf.keras.optimizers.SGD(lr=0.001)
        # self.optimizer = tf.keras.optimizers.Adadelta()
        self.optimizer = tf.keras.optimizers.Adam()


    def start_training(self):
        dataset = self.dataset.repeat(-1)
        validation = self.validation.repeat(-1)
        self.model.fit(x=dataset,validation_data=validation, verbose=1, steps_per_epoch=1200,validation_steps=200, epochs=2000,callbacks=self.callback)

    def load_model_weight(self):
        self.model.load_weights('./autoencoder_cnn_gradcam.h5')

    def classify(self):
        iterator = self.validation.take(1).make_initializable_iterator()
        next_obj = iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.model.load_weights('./autoencoder_cnn_gradcam.h5')

            try:
                while True:
                    x,gt = sess.run(next_obj)
                    gty = np.argmax(gt, axis=1)
                    y = self.model.predict(x)
                    yy = np.argmax(y, axis=1)
                    print(gty)
                    print(yy)
                    classified_out = self.model.output[:,7]
                    print(classified_out.shape)
                    last_cnn = self.model.get_layer('conv2d-4')
                    gradients = tf.keras.backend.gradients(classified_out, last_cnn.output)[0]
                    print( gradients.shape )
                    pooled_grads = tf.keras.backend.mean( gradients , axis=(0,1,2)) # チャンネルごとの傾きの平均値を出す。
                    print(pooled_grads.shape)

                    iterate = tf.keras.backend.function([self.model.input],[pooled_grads,last_cnn.output[0]])
                    pooled_grads_val, cnn_val = iterate([x]) # 
                    for i in range( 16 ):
                        cnn_val[:,:,i] *= pooled_grads_val[i]
                    heatmap = np.mean(cnn_val, axis=-1) # クラスごとの活性化値を全体で平均
                    heatmap = np.maximum(heatmap, 0)
                    heatmap /= np.max(heatmap)
                    print(heatmap.shape)
                    plt.imshow(heatmap)
                    plt.show()
            except tf.errors.OutOfRangeError:
                print('end')
    
if __name__ == '__main__':
    mnist_test = MnistCnnGradCam()
    mnist_test.load_data()
    mnist_test.convert_to_dataset()
    mnist_test.create_optimizer()
    mnist_test.create_cnn_model()
    mnist_test.create_callback()
    # mnist_test.load_model_weight()
    # mnist_test.start_training()

    # mnist_test.load_model_weight()
    mnist_test.classify()


# 最終層となる、CNNに画像を認識させたときの出力に、
# 傾きの大きさをかけ合わせてヒートマップを作成する。
