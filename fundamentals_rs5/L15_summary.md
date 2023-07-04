# TF2 API의 개요
## 1. TensorFlow2로 API 모델 구성하기
TF2에서 모델을 작성하는 방법에는 크게 3가지가 있는데 각각을 `Sequential`, `Functional`, `Model Subclassing`이라고 한다.</br>

---

### Sequential Model 예시
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(example_layer1)
model.add(example_layer2)
model.add(example_layer3)

model.fit(x, y, epochs=10, batch_size=32)
```
==장점==</br>
1. 손쉽게 딥러닝 모델을 쌓을 수 있다.
2. 딥러닝 모델을 학습하는 초보자가 접근하기에 쉽다.</br>

==단점==</br>
1. 모델의 입력과 출력이 여러 개인 경우 적합하지 않은 모델링 방식이다.

[TF2 Example of the Sequential Modeling method](https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko)</br>

---

### Functional Model 예시
```python
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(example_input_shape))
x = keras.layers.example_layer1(ex_params1)(input)
x = keras.layers.example_layer2(ex_params2)(x)
outputs = keras.layers.example_layer3(ex_params3)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.fit(x,y, epochs=10, batch_size=32)
```
딥러닝 모델 구성에서의 일반적인 접근법</br>
**입력과 출력을 규정**함으로써 모델 전체를 규정한다.</br>
```text
Functional Model에서 입력을 규정할 때, Input을 통해 입력을 규정한다.
Input이 될 수 있는 텐서는 여러 개가 될 수도 있다.
```
Functional Modeling에서는 **다중 입력/출력을 가지는 모델을 구성할 수 있다.**</br>
[TF2 Example of the Functional Modeling method](https://www.tensorflow.org/guide/keras/functional?hl=ko)</br>

---

### Subclassing Model 에시
```python
import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.example_layer1()
        self.example_layer2()
        self.example_layer3()
    
    def call(self, x):
        x = self.example_layer4(x)
        x = self.example_layer5(x)
        x = self.example_layer6(x)
        
        return x
    
model = CustomModel()
model.fit(x,y, epochs=10, batch_size=32)
```
모델링 방법 중 제일 자유로운 모델링을 진행할 수 있다.</br>
Subclassing 모델링의 경우 각 레이어에 대한 깊은 이해가 필요하며 초심자에게 의도치 않은 버그를 유발할 수 있다.</br>
[TF2 Example of the Subclassing Modeling method](https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko)</br>

---

## 2. TF2 API로 모델 작성하기 - MINIST with Sequential API
Code Example1
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 데이터 구성부분
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.expand_dims(x_train, axis = 3)
x_test = tf.expand_dims(x_test, axis = 3)

print(len(x_train), len(x_test))
print(f"데이터의 크기는 : {x_train.shape} 입니다.")
```
결과예시1
```shell
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
11501568/11490434 [==============================] - 0s 0us/step
60000 10000
데이터의 크기는 : (60000, 28, 28, 1) 입니다.
```

---

Code Example2
```python
# Sequential Model을 구성해주세요.
"""
Spec:
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu를 갖고 있으며 input_shape은 데이터 크기로 하는 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""
#[[YOUR CODE]]
# Declare Sequential Model
model = tf.keras.Sequential()

# 위 예시에서 보여준 데이터의 크기
example = (28, 28, 1)
# 또는 다음과 같이 Array shape를 변환해주자; batch 이미지 중 하나
# example = x_train.shape[1:]

# 32개 채널, 커널 크기 3, activation func 'relu', input shape은 데이터 크기, Conv2D 레이어
model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=example))
## MaxPooling을 사용해주어야 행렬의 손실이 없다.
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
### 해당 노드에서는 Pooling을 할 필요가 없는것 같다.

# 64개 채널, 커널 크기 3, activation func 'relu', Conv2D 레이어
## input_shape를 생략해도 이미 convolution_1st에서 작업한 내용을 알고 있기 때문에 문제가 없다. (in TF)
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
## MaxPooling을 사용해주어야 행렬의 손실이 없다.
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
## 해당 노드에서는 Pooling을 할 필요가 없는것 같다.

# Flatten 레이어
## 여기서도 input_shape가 필요하지만, 생략할 경우 이전 input_shape의 값을 입력하게된다.
model.add(tf.keras.layers.Flatten())

# 128개 아웃풋 노드, activation func 'relu' Fully-Connected Layer(Dense)
model.add(tf.keras.layers.Dense(128, activation='relu'))

# 데이터셋의 클래스 개수에 맞는 아웃풋 노드, activation func is 'softmax'인 Fully-Connected Layer(Dense)
## MINIST는 이미(by handwritten) 0 ~ 9의 분류로 classification이 되어있기 때문에 output node가 10
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 여기에 모델을 구성해주세요
model.summary()
```
결과예시2
```shell
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               204928    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 225,034
Trainable params: 225,034
Non-trainable params: 0
_________________________________________________________________
```

---

Code Example3
```python
# 모델 학습 설정
# 2 epochs 학습에 10분 정도 소요됩니다.
# 잠시 스트레칭하고 휴식을 취해보아요~
# (빠르게 동작 여부만 확인하고 싶으시면 epochs 값을 줄여주세요.)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)

model.evaluate(x_test,  y_test, verbose=2)
```
결과예시3
```shell
Epoch 1/2
1875/1875 [==============================] - 73s 37ms/step - loss: 0.1370 - accuracy: 0.9567
Epoch 2/2
1875/1875 [==============================] - 73s 39ms/step - loss: 0.0457 - accuracy: 0.9854
313/313 - 3s - loss: 0.0376 - accuracy: 0.9869

[0.03761504963040352, 0.9868999719619751]
```

---

## 3. TF2 API로 모델 작성하기 - MINIST with Functional API
Code Example1
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 데이터 구성부분
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.expand_dims(x_train, axis = 3)
x_test = tf.expand_dims(x_test, axis = 3)
```
결과예시1
```shell
60000 10000
```

---

Code Example2
```python
"""
Spec:
0. 데이터 크기로로 정의된 Input
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)

"""

#[[YOUR CODE]]
## 기존 x_train의 batch size중 하나를 가져온다; (28, 28, 1)
example = x_train.shape[1:]
# 데이터 크기로 정의된 Input
inputs = tf.keras.Input(shape=example)

# 32개 채널, 커널 크기 3, activation func 'relu', Conv2D 레이어
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
## maxpooling 빼보자

# 64개 채널, 커널 크기 3, activation func 'relu', Conv2D 레이어
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
## maxpooling 빼보기

# Flatten 레이어
x = tf.keras.layers.Flatten()(x)

# 128개 output 노드, activation func 'relu', Fully-Connected Layer(Dense)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# 데이터셋의 클래스 개수에 맞는 output 노드, activation func 'softmax', Fully-Connected Layer(Dense)
## MINIST의 classfication은 10으로 고정
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create the Model
model = keras.Model(inputs=inputs, outputs=predictions)
model.summary()
```
결과예시1
```shell
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
flatten_2 (Flatten)          (None, 36864)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               4718720   
_________________________________________________________________
dense_5 (Dense)              (None, 10)                1290      
=================================================================
Total params: 4,738,826
Trainable params: 4,738,826
Non-trainable params: 0
_________________________________________________________________
```

---

Code Example3
```python
# 모델 학습 설정
# 2 epochs 학습에 10분 정도 소요됩니다.
# 잠시 스트레칭하고 휴식을 취해보아요~
# (빠르게 동작 여부만 확인하고 싶으시면 epochs 값을 줄여주세요.)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)

model.evaluate(x_test,  y_test, verbose=2)
```
결과예시3
```shell
Epoch 1/2
1875/1875 [==============================] - 255s 135ms/step - loss: 0.1069 - accuracy: 0.9675
Epoch 2/2
1875/1875 [==============================] - 250s 133ms/step - loss: 0.0348 - accuracy: 0.9889
313/313 - 9s - loss: 0.0385 - accuracy: 0.9874

[0.03853471204638481, 0.9873999953269958]
```

---

## 4. TF2 API로 모델 작성하기 - MINIST with Subclassing API
Code Example1
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 데이터 구성부분
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.expand_dims(x_train, axis = 3)
x_test = tf.expand_dims(x_test, axis = 3)

print(len(x_train), len(x_test))
```
결과예시1
```shell
60000 10000
```

---

Code Example2
```python
# Subclassing을 활용한 Model을 구성해주세요.


# 여기에 모델을 구성해주세요
class CustomModel(keras.Model):
    ##[[YOUR CODE]]
    """
    Spec:
    0. keras.Model 을 상속받았으며, __init__()와 call() 메서드를 가진 모델 클래스
    1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
    2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
    3. Flatten 레이어
    4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
    5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
    6. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
    """
    # keras.Model 상속받았으며, __init__()과 call()을 가진 모델 클래스
    def __init__(self):
        ## keras.Model 상속부분
        super(CustomModel, self).__init__()
        
        # 32개 채널, 커널 크기 3, activation func 'relu', Conv2D
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')

        # 64개 채널, 커널 크기 3, activation func 'relu', Conv2D
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

        # Flatten 레이어
        self.flatten = tf.keras.layers.Flatten()

        # 128개 output 노드, activation func 'relu', Fully-Connected Layer (Dense)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')

        # MINIST의 클래스 개수(10개)에 맞는 아웃풋 노드, activation func 'softmax', Fully-Connected Layer (Dense)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def summary(self):
        # x_train 데이터를 기반으로 input_shape 생성
        example = x_train.shape[1:]
        # print(example)

        x = keras.Input(shape = (example))
        # 기존 코드
        """
        model = Model(inputs=[x], outputs = self.call(x))
        """
        # Functional Model을 선언할 때, tf.keras.Model() Class에서 가져오게끔 작성하여야 한다.
        model = tf.keras.Model(inputs=[x], outputs = self.call(x))
        
        return model.summary()


model = CustomModel()

model.summary()
```
결과예시2
```shell
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_12 (InputLayer)        [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 24, 24, 64)        18496     
_________________________________________________________________
flatten_12 (Flatten)         (None, 36864)             0         
_________________________________________________________________
dense_24 (Dense)             (None, 128)               4718720   
_________________________________________________________________
dense_25 (Dense)             (None, 10)                1290      
=================================================================
Total params: 4,738,826
Trainable params: 4,738,826
Non-trainable params: 0
_________________________________________________________________
```

---

Code Example3
```python
# 모델 학습 설정
# 2 epochs 학습에 10분 정도 소요됩니다.
# 잠시 스트레칭하고 휴식을 취해보아요~
# (빠르게 동작 여부만 확인하고 싶으시면 epochs 값을 줄여주세요.)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)

model.evaluate(x_test,  y_test, verbose=2)
```
결과예시3
```shell
Epoch 1/2
1875/1875 [==============================] - 252s 134ms/step - loss: 0.1073 - accuracy: 0.9671
Epoch 2/2
1875/1875 [==============================] - 245s 131ms/step - loss: 0.0346 - accuracy: 0.9892
313/313 - 9s - loss: 0.0379 - accuracy: 0.9881

[0.03790771961212158, 0.988099992275238]
```

---

## 5. TF2 API로 모델 작성하기 - CIFAR100 with Sequential API
<!-- 추가 작성 필요 -->
## 6. TF2 API로 모델 작성하기 - CIFAR100 with Functional API

## 7. TF2 API로 모델 작성하기 - CIFAR100 with SubClassing API

## 8. GradientTape의 활용
딥러닝 모델 학습 부분 예시
```python
# 모델 학습 설정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
```

---

Numpy만으로 구현한 딥러닝의 모델 훈련 과정 요약
1. Forward Propagation 수행 및 중간 레이어값 저장
2. Loss 값 계산
3. 중간 레이어값 및 Loss를 활용한 체인룰(Chain Rule) 방식의 역전파(Backward Propagation) 수행
4. 학습 파라미터 업데이트

위 과정이 TF2 API에서는 `model.fit()` 메서드 안에 모두 추상화되어 있다.</br>

TF2 API에서는 **GradientTape** 기능을 `tf.GradientTape` 메서드를 통해 제공한다.</br>
> GradientTape의 기능 수행 과정
1. 순전파(forward pass)로 진행된 모든 연산의 중간 레이어 값을 **Tape**에 기록
2. 기록된 tape을 이용하여 **gradient**를 계산한 후 **tape를 폐기**

<center>CIFAR100 데이터 기반의 gradientTape 코드 예시</center></br>

Code Example1
```python
import tensorflow as tf
from tensorflow import keras

# 데이터 구성부분
cifar100 = keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(len(x_train), len(x_test))

# 모델 구성부분
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D((2,2))
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D((2,2))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(100, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = CustomModel()
```
결과예시1
```shell
50000 10000
```

---

Code Example2
```python
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
가 실질적으로는 어떤 과정을 통해 수행되는지 보여주는 예시 코드
"""

# 1. tape.gradient()를 통해 매 스탭 학습시 발생하는 gradient를 추출
# 2. optimizer.appy_gradient()를 통해 gradient 발생
# 3. 2번 결과물(gradient)로 업데이트할 파라미터를 지정; model.trainable_variables 사용


loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# tf.GradientTape()를 활용한 train_step
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

---

Code Example3
```python
"""
model.fit(x_train, y_train, epochs=5, batch_size=32)
가 실질적으로는 어떤 과정을 통해 수행되는지 보여주는 예시 코드
"""

import numpy as np
import time

def train_model(batch_size=32):
    start = time.time()
    for epoch in range(5):
        x_batch = []
        y_batch = []
        for step, (x, y) in enumerate(zip(x_train, y_train)):
            x_batch.append(x)
            y_batch.append(y)
            if step % batch_size == batch_size-1:
                loss = train_step(np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32))
                x_batch = []
                y_batch = []
        print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    print("It took {} seconds".format(time.time() - start))

train_model()
```
결과예시3
```shell
Epoch 0: last batch loss = 3.1631
Epoch 1: last batch loss = 2.6685
Epoch 2: last batch loss = 2.5448
Epoch 3: last batch loss = 2.4732
Epoch 4: last batch loss = 2.4410
It took 338.94916319847107 seconds
```

---

Code Example4
```python
"""
Evaluation 단계에서는 gradient 활용의 필요가 없어서 기존 방식을 사용함
"""

# evaluation
prediction = model.predict(x_test, batch_size=x_test.shape[0], verbose=1)
temp = sum(np.squeeze(y_test) == np.argmax(prediction, axis=1))
temp/len(y_test)  # Accuracy
```
결과예시4
```shell
1/1 [==============================] - 3s 3s/step

0.3494
```

---

`tf.GradientTape()`의 `train_step` 구성 순서</br>
1. feature 입력 후 모델 예측하여 예측값 획득
2. label과 예측값의 차이를 가지고 loss 값 구하기 (loss function 사용)
3. model에 있는 **trainable parameter**와 loss를 이용해 gradient 추출
4. gradient와 `optimizer()`를 사용해 model에 있는 학습가능한 파라미터 지정