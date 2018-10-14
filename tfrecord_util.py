import sys
import os
import argparse
import tensorflow as tf

IMAGE_SIZE = 28

## ディレクトリごとの画像をTFRecordに作成する。
class TFRecordCreator:


    def __init__(self):
        self.size = 0
        self.label_depth = 0
        self.const_label = -1

    def __get_label(self, filename):
        split_string = tf.string_split([filename],"/")
        label_str = split_string.values[self.label_depth]
        label_num = tf.string_to_number(label_str)
        label_num = tf.cast(label_num, tf.int64 )
        label = tf.reshape(label_num, [])
        # tf.Print( label_num, [label_num], "split values: ")
        return label

    def __load_image(self, filename):
        # 指定のファイルサイズに変換
        content = tf.read_file(filename)
        image = tf.image.decode_image(content, name="load_image_file")
        image_shape = tf.shape(image, name="get_image_shape")
        read_hight = image_shape[0]
        read_width = image_shape[1]
        square_image = tf.cond(
            read_hight >= read_width,
            lambda: tf.image.resize_image_with_crop_or_pad(image,read_hight,read_hight),
            lambda: tf.image.resize_image_with_crop_or_pad(image,read_width,read_width),
        )
        resized_image = tf.image.resize_images(square_image,[self.size,self.size])
        float_arrayed_image = tf.reshape(resized_image,[-1])
        float_arrayed_image /= 255.0
        return float_arrayed_image

    def __prepare_example_data(self, filename):
        if self.const_label == -1:
            label = self.__get_label(filename)
        else:
            label = self.const_label
        image = self.__load_image(filename)
        return (label,image)

    def craete_tfrecord(self, filepattern, dest, size, const_label=-1, label_depth=1):
        self.size = size
        self.label_depth = label_depth
        self.const_label = const_label
        files = tf.data.Dataset.list_files(filepattern)
        dataset = files.map(self.__prepare_example_data , num_parallel_calls=2)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        init_op = iterator.initializer
        writer = tf.python_io.TFRecordWriter(dest)
        i = 0
        with tf.Session() as sess:
            sess.run(init_op)
            try:
                while True:
                    (label,image) = sess.run(next_element)
                    features = tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.size])),
                        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.size])),
                        'image': tf.train.Feature(float_list=tf.train.FloatList(value=image))
                    })
                    example = tf.train.Example(features=features)
                    writer.write(example.SerializeToString())
                    i = i + 1
                    if( i % 1000 == 0):
                        print(i)
            except tf.errors.OutOfRangeError:
                writer.close()
                print("TFRecord done!!! {}".format(i))

class TFRecordLoader:

    def __init__(self):
        print("TFRecordLoader")

    def __get_example(self, example_proto):
        features = {
            "label": tf.FixedLenFeature([], tf.int64, default_value=0),
            "width": tf.FixedLenFeature([], tf.int64, default_value=0),
            "height": tf.FixedLenFeature([], tf.int64, default_value=0),
            "image": tf.FixedLenFeature([self.size * self.size * self.channel], dtype=tf.float32 )
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.reshape(parsed_features["image"],[self.size, self.size, self.channel])
        return image, parsed_features["label"]

    def load_dataset(self, path, size, channel):
        dataset = tf.data.TFRecordDataset(path)
        self.size = size
        self.channel = channel
        dataset = dataset.map(map_func=self.__get_example)
        return dataset
        

        """        
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        init_op = iterator.initializer
        i = 0
        with tf.Session() as sess:
            sess.run(init_op)
            try:
                while True:
                    example = sess.run(next_element)
                    i = i + 1
                    if( i % 1000 == 0):
                        print(i)
            except tf.errors.OutOfRangeError:
                print("End dog cat!!!")        
        """
    

def create_model(self, size, channel, classnum):
    input_layer = tf.keras.layers.Input(shape=(size,size,channel),name='input')
    layer = tf.keras.layers.Conv2D(filters=5,padding="same",kernel_size=3,name='conv1')(input_layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.Conv2D(filters=5,padding="same",kernel_size=3,name='conv2')(input_layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.Conv2D(filters=5,padding="same",kernel_size=3,name='conv2')(input_layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dense(size,activation='sigmoid')(layer)
    output_layer = tf.keras.layers.Dense(classnum,activation='softmax')(layer)
    model = tf.keras.models.Model(input_layer,output_layer)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="dirctory of src images")
    parser.add_argument("-o", '--out', help='destination TFRecord file : default out.tfrecord')
    parser.add_argument('-s', '--size' ,help='size of the image (px) : default 96 ', type=int)
    parser.add_argument('-l', '--label', help='fixed label', type=int)
    parser.add_argument("-pl", '--parse_level', help='parse directory level', type=int)
    args = parser.parse_args()

    if args.src is None:
        args.src = 'img'+os.path.sep+'*.jpg'
    else:
        args.src = args.src + os.path.sep +'*.jpg'

    if args.out is None:
        args.out = 'out.tfrecord'
    
    if args.label is None:
        args.label = 0

    if args.size is None:
        args.size = 96

    if 'help' in args:
        sys.exit()

    print(args)

    creator = TFRecordCreator()
    creator.craete_tfrecord(args.src, args.out, args.size, args.label, args.parse_level )

# loader = TFRecordLoader()
# loader.load_dataset("mnist.tfrecord",size=28,channel=1)
# loader.load_dataset("dog_cat.tfrecord",size=64,channel=3)


