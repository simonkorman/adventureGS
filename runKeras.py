from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Merge, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

def addConvBNReLU(model, nOutputPlane, kw, kh, do_size = False, do_relu = True):
    if do_size:
        model.add(Convolution2D(nOutputPlane, kw, kh,input_shape = (3, 19, 19)))
    else:
        model.add(Convolution2D(nOutputPlane, kw, kh))  #
    model.add(BatchNormalization(epsilon=0.001))
    if do_relu:
        model.add(Activation('relu'))

base = Sequential()
addConvBNReLU(base,64,3,3,do_size=True) #1
addConvBNReLU(base,64,3,3) #2
addConvBNReLU(base,64,3,3) #3
addConvBNReLU(base,64,3,3) #4
addConvBNReLU(base,64,3,3) #5
addConvBNReLU(base,64,3,3) #6
addConvBNReLU(base,64,3,3) #7
addConvBNReLU(base,64,3,3) #8
addConvBNReLU(base,64,3,3,do_relu = False) #9
base.add(Flatten())

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

processed_a = base(input_a)
processed_b = base(input_b)

distance = Merge([processed_a, processed_b], mode = 'dot')


