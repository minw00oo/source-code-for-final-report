from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, Add, SeparableConv2D
from tensorflow.keras import Model, optimizers, callbacks, backend
from tensorflow.keras.models import load_model

def conv_block(input, num_filters, kernel_size):
    x = SeparableConv2D(num_filters, kernel_size, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    return x

def multi_dense_block(input, num_filters):
    x1 = conv_block(input, num_filters, 2)
    x2 = conv_block(input, num_filters, 4)
    x3 = conv_block(input, num_filters, 8)
    
    a = Add()([x1, x2, x3])
    a = Concatenate()([a, input])
    
    y1 = conv_block(a, num_filters, 2)
    y2 = conv_block(a, num_filters, 4)
    y3 = conv_block(a, num_filters, 8)
    
    b = Add()([y1, y2, y3])
    b = Concatenate()([a, b, input])
    
    c = conv_block(b, num_filters, 1)
    return c

def encoder_block(inputs, num_filters):
    x = multi_dense_block(inputs, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (8, 8), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = multi_dense_block(x, num_filters)
    return x

inputs = Input((64, 64, 3))

s1, p1 = encoder_block(inputs, 32//3)
s2, p2 = encoder_block(p1, 32//3)
s3, p3 = encoder_block(p2, 64//3)
s4, p4 = encoder_block(p3, 128//3)

b1 = multi_dense_block(p4, 256//3)

d1 = decoder_block(b1, s4, 128//3)
d2 = decoder_block(d1, s3, 64//3)
d3 = decoder_block(d2, s2, 32//3)
d4 = decoder_block(d3, s1, 32//3)

outputs = Conv2D(1, 1, padding="same", activation="linear")(d4)
model = Model(inputs=[inputs], outputs=[outputs])
#%%
adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam,
              loss="mse",
              metrics=["mse"])
model.summary()
#%%
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt

data2 = np.load('dataset2.npy')
target2 = np.load('target2.npy')

train_data = data2[:5000]
test_data = data2[5000:]

target2 = target2.reshape(6000, 64, 64, 1)
train_fem = target2[:5000]
test_fem = target2[5000:]
#%%
results = model.fit(train_data, train_fem, validation_split=0.2, batch_size=200, epochs=1000)
#%%
pred_fem = model.predict(test_data)

test_fem = test_fem.reshape(1000, 64, 64)
pred_fem = pred_fem.reshape(1000, 64, 64)
