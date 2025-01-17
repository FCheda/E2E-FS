from tensorflow.keras.models import Model
from tensorflow.keras import backend as K, optimizers, layers
from tensorflow.keras.regularizers import l1, l2
from .layers.mask import Mask


def fcnn(nfeatures, nclasses=2, layer_dims=None, bn=True, kernel_initializer='he_normal',
         dropout=0.0, lasso=0.0, regularization=0.0, momentum=0.9):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input = layers.Input(shape=nfeatures, sparse=False)
    if layer_dims is None:
        layer_dims = [150, 100, 50]
    x = input
    if lasso > 0:
        x = Mask(kernel_regularizer=l1(lasso))(x)

    for layer_dim in layer_dims:
        x = layers.Dense(layer_dim, use_bias=not bn, kernel_initializer=kernel_initializer,
                         kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
        if bn:
            x = layers.BatchNormalization(axis=channel_axis, momentum=momentum, epsilon=1e-5, gamma_initializer='ones')(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        x = layers.Activation('relu')(x)

    x = layers.Dense(nclasses, use_bias=True, kernel_initializer=kernel_initializer,
                     kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
    output = layers.Activation('softmax')(x)

    model = Model(input, output)

    optimizer = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model


def three_layer_nn(nfeatures, nclasses=2, bn=True, kernel_initializer='he_normal',
                   dropout=0.0, lasso=0.0, regularization=5e-4, momentum=0.9):

    return fcnn(nfeatures, nclasses, layer_dims=[50, 25, 10], bn=bn,
                kernel_initializer=kernel_initializer, dropout=dropout, lasso=lasso,
                regularization=regularization, momentum=momentum)

