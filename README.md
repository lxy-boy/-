# -import tensorflow as tf
import os
from tensorflow import keras
class SingleNN(object):
    model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(130,activation=tf.nn.relu),
        keras.layers.Dense(60,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    def __init__(self):
        (self.x_true,self.y_true),(self.x_test,self.y_test)=keras.datasets.fashion_mnist.load_data()
        self.x_true=self.x_true/255
        self.x_test=self.x_test/255

    def singlenn_compile(self):
        SingleNN.model.compile(optimizer=keras.optimizers.SGD(0.01),
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=["accuracy"])

    def singleen_fit(self):
        SingleNN.model.fit(self.x_true,self.y_true,epochs=5)

    def singlenn_eva(self):
        test_loss,test_acc=SingleNN.model.evaluate(self.x_test,self.y_test)
        print(test_acc,test_loss)
if __name__ == '__main__':
    ss=SingleNN()
    ss.singlenn_compile()
    ss.singleen_fit()
    ss.singlenn_eva()
