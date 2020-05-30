from keras import backend as K, layers, models, initializers, regularizers
import numpy as np
import tensorflow as tf


class E2EFS(layers.Layer):

    def __init__(self, units,
                 kernel_initializer='truncated_normal',
                 kernel_constraint=None,
                 kernel_activation=None,
                 kernel_regularizer=None,
                 heatmap_momentum=.99999,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(E2EFS, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.kernel_activation = kernel_activation
        self.kernel_regularizer = kernel_regularizer
        self.supports_masking = True
        self.kernel = None
        self.heatmap_momentum = heatmap_momentum

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = np.prod(input_shape[1:])
        kernel_shape = (input_dim, )
        self.reduce_func = None
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.trainable)
        self.moving_heatmap = self.add_weight(shape=(input_dim, ),
                                              name='heatmap',
                                              initializer='ones',
                                              trainable=False)
        self.e2efs_kernel = self.kernel if self.kernel_activation is None else self.kernel_activation(self.kernel)
        self.stateful = False
        self.built = True

    def call(self, inputs, training=None, **kwargs):

        kernel = self.kernel
        if self.kernel_activation is not None:
            kernel = self.kernel_activation(kernel)
        if self.reduce_func is not None:
            kernel = self.reduce_func(kernel, axis=-1)
        kernel_clipped = K.reshape(kernel, shape=inputs.shape[1:])

        output = inputs * kernel_clipped

        return output

    def add_to_model(self, model, input_shape, activation=None):
        input = layers.Input(shape=input_shape)
        x = self(input)
        if activation is not None:
            x = layers.Activation(activation=activation)(x)
        output = model(x)
        model = models.Model(input, output)
        model.fs_kernel = self.e2efs_kernel
        model.heatmap = self.moving_heatmap
        return model

    def compute_output_shape(self, input_shape):
        return input_shape


class E2EFSHard(E2EFS):

    def __init__(self, units,
                 kernel_initializer='ones',
                 l1=1.,
                 l2=1.,
                 dropout=.1,
                 **kwargs):

        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        super(E2EFSHard, self).__init__(units=units,
                                        kernel_initializer=kernel_initializer,
                                        **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.moving_units = self.add_weight(shape=(),
                                            name='moving_units',
                                            initializer=initializers.constant(self.units),
                                            trainable=False)

        def apply_dropout(x):
            if 0. < self.dropout < 1.:
                def dropped_inputs():
                    x_shape = K.int_shape(x)
                    noise = K.random_uniform(x_shape)
                    return K.switch(K.less(noise, self.dropout), -1000. * K.ones_like(x), x)
                return K.in_train_phase(dropped_inputs, x)
            return x

        def kernel_activation():
            @tf.custom_gradient
            def func(x):
                x_shape = K.int_shape(x)
                x = apply_dropout(x)
                _, top_k = tf.nn.top_k(x, k=x_shape[0])
                _, ranks = tf.nn.top_k(-top_k, k=x_shape[0])
                mask = K.cast(K.cast(ranks, K.floatx()) < self.moving_units, K.floatx())
                h = mask * K.relu(x)

                def grad(dy, variables=None):
                    return mask * dy * K.relu(K.sign(x)), [K.ones_like(v) for v in variables]

                return h, grad

            return func
        self.kernel_activation = kernel_activation()

        def kernel_regularizer(x):
            return self.l1 * K.mean(K.abs(x)) + self.l2 * K.mean(K.square(x))

        self.kernel_regularizer = kernel_regularizer

        super(E2EFSHard, self).build(input_shape)


class E2EFSSoft(E2EFS):

    def __init__(self, units,
                 dropout=.1,
                 decay_factor=.75,
                 kernel_regularizer=regularizers.l2(1e-2),
                 init_value=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.dropout = dropout
        self.decay_factor = decay_factor
        self.init_value = init_value
        super(E2EFSSoft, self).__init__(units=units,
                                        kernel_regularizer=kernel_regularizer,
                                        **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = int(np.prod(input_shape[1:]))

        self.moving_units = self.add_weight(shape=(),
                                            name='moving_units',
                                            initializer=initializers.constant(self.units),
                                            trainable=False)
        self.moving_factor = self.add_weight(shape=(3, ),
                                             name='moving_factor',
                                             initializer=initializers.constant([1., .1, .5]),
                                             trainable=False)
        self.cont = self.add_weight(shape=(),
                                    name='cont',
                                    initializer='ones',
                                    trainable=False)

        init_value = self.init_value if self.init_value is not None else max(.05, self.units / input_dim)
        self.kernel_initializer = initializers.constant(init_value)

        def apply_dropout(x,  rate, refactor=False):
            if 0. < self.dropout < 1.:
                def dropped_inputs():
                    x_shape = K.int_shape(x)
                    noise = K.random_uniform(x_shape)
                    factor = 1. / (1. - rate) if refactor else 1.
                    return K.switch(K.less(noise, self.dropout), K.zeros_like(x), factor * x)
                return K.in_train_phase(dropped_inputs, x)
            return x

        def kernel_activation(x):
            x = apply_dropout(x, self.dropout, False)
            t = x / K.max(K.abs(x))
            s = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), K.clip(x, 0., 1.))
            s /= K.stop_gradient(K.max(s))
            return s

        self.kernel_activation = kernel_activation

        def loss_units(x):
            t = x / K.max(K.abs(x))
            x = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            m = K.sum(K.cast(K.greater(x, 0.), K.floatx()))
            sum_x = K.sum(x)
            moving_units = K.switch(K.less_equal(m, self.units), m,
                                    (1. - self.decay_factor) * self.moving_units)
            epsilon_minus = K.switch(K.less_equal(m, self.units), 0., self.moving_factor[2])
            epsilon_plus = K.switch(K.less_equal(m, self.units), self.moving_units, self.moving_factor[2])
            return K.relu(moving_units - sum_x - epsilon_minus) + K.relu(sum_x - moving_units - epsilon_plus)

        super(E2EFSSoft, self).build(input_shape)

        def regularization(x):
            l_units = loss_units(x)
            t = x / K.max(K.abs(x))
            x = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            p = K.clip(x, 0., 1.)
            cost = K.cast_to_floatx(0.)
            cost += K.sum(p * (1. - p)) + l_units
            cost += K.sum(K.relu(x - 1.))

            return cost, K.sum(p * (1. - p)), K.sum(p), -K.sum(K.square(p))

        self.regularization_loss = regularization(self.kernel)