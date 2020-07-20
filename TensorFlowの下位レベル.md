# 逆引き：TensorFlowオペレータマスター



## 1.テンソルで四則演算したい。

###  要素ごととの演算

```
t = 1
s = 2
t + s
```

## Tensor演算の性質2:ブロードキャスト

要素の大きさに応じて演算が適用される。

[1,2,3] + [2,3,4] = [3,5,7]

## テンソルの積

要素ごとの積　＊
内積          dot()


## 2. Tensorを変形したい。

### tf.reshape()

テンソルの変形

tf.reshape

```
t = [[1,2,3],[4,5,6]]
r = tf.reshape([6])

[1,2,3,4,5,6]
```

### tf.transpose() 

転置をします。

```
t = [[1,2,3],[4,5,6]]
tr = tf.transpose()
[[1,4],[2,5],[3,6]]
```

## Tensorの部分を取り出したい。

### tf.slice()

指定の添え字領域部分を取り出す。

```
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])

tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]

tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                   #   [4, 4, 4]]]

tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                   #  [[5, 5, 5]]]

```

### tf.split()

指定の軸に沿って、指定のサイズに分割をする。

```
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0)  # [5, 4]
tf.shape(split1)  # [5, 15]
tf.shape(split2)  # [5, 11]

# 次元1に関して、3つに分割
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0)  # [5, 10]
```

## Tensorを結合したい。

## tf.stack()

Tensorを積む。

```
x = tf.constant([1, 4]) 
y = tf.constant([2, 5]) 
z = tf.constant([3, 6]) 
stack = tf.stack([x, y, z]) 

[[1,4],[2,5],[3,6]]
```





## tf.concat(values,axis,name)

__ある次元に沿ってテンソルを連結する。__

values: Tensorオブジェクトのリスト、または単一のTensor。
axis: 0-D int32 Tensor. 連結する次元。rank(values), rank(values))の範囲内でなければなりません。Pythonと同様に，axisのインデックスは0ベースです．(0-rank(値))の範囲内の正の軸は、軸目の次元を参照します。
また、負の軸は軸＋rank(値)-番目の次元を参照します。
name: 操作の名前（オプション）。

戻り値:入力テンソルの連結から得られるテンソル。

例

0次元目で結合

```
t1 = [[1, 2, 3], [4, 5, 6]] 
t2 = [[7, 8, 9], [10, 11, 12]] 
concat([t1, t2], 0) 

[[1,2,3],
 [4,5,6],
 [7, 8, 9], 
 [10, 11, 12]]

```
1次元目で結合

```
t1 = [[1, 2, 3], [4, 5, 6]] 
t2 = [[7, 8, 9], [10, 11, 12]] 
concat([t1, t2], 1) 

[[1,2,3,4,5,6],
 [7, 8, 9,10, 11, 12]]
```



## Tensorの値を組み替え

### tf.scatter_nd(indices, updates, shape) 

shape型のTensorのindicesに指定された場所に、valuesを混ぜたしたTensorを返却する。

```
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)
```


```
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)
```

![scatter_add2](https://www.tensorflow.org/images/ScatterNd2.png)



## tf.gather / tf.gather_nd

paramsからindicesに従ってデータを集約します。scatterの逆の操作になります。

![gather](https://www.tensorflow.org/images/Gather.png)

tf.gatherではindicesがparamsの1次元目のスライスを定義していますが、tf.gather_ndではindicesがparamsの最初のN次元目のスライスを定義しており、N = indices.shape[-1]となります。

## Graph演算をきれいにする

## control_dependencies

control_dependencies(
    control_inputs
)

__何かを実行した後に実行するよう、制御依存関係を指定したコンテキストマネージャを返します。__

with キーワードを使用して、コンテキスト内で構築されたすべての操作が control_inputs への制御依存関係を持つように指定します。例えば、以下のようになります。

```
with g.control_dependencies([a, b, c]).
  # d` と `e` は、`a`, `b`, `c` が実行された後にのみ実行されます。
  d = ...
  e = ...
```

## tf.group()

複数のオペレーションをグループ化します。



## Keras 独自レイヤーを作ってみよう。

状態のあるレイヤーはtf.keras.layers.Layerサブクラスである必要。

必要メソッドの実装内容

|メソッド|内容|
|:--|:--|
|build(input_shape)| これは重みを定義するメソッドです．このメソッドは，self.built = Trueをセットしなければいけません，これはsuper([Layer], self).build()を呼び出しでできます．|
|call(x)| ここではレイヤーのロジックを記述します．オリジナルのレイヤーでマスキングをサポートしない限り，第1引数である入力テンソルがcallに渡されることに気を付けてください．|
|compute_output_shape(input_shape)| 作成したレイヤーの内部で入力のshapeを変更する場合には，ここでshape変換のロジックを指定する必要があります．こうすることでKerasが自動的にshapeを推定します．|


```
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```



## Keras 独自Lossを作るには？

(y_true, y_pred)を引数とする関数を作ればよい。
ただし、y_true,y_predは同一シェイプとする。


## Kerasのレイヤー作成

Kerasのカスタムレイヤーはtf.keras.layers.Lambdaを使用することで、


## Keras Optimizerの作成

これは少し難しそう。。。。
