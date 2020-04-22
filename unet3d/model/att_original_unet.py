from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D
from keras.engine import Model
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate, get_up_convolution
from ..metrics import dice_coefficient_loss, dice_coefficient
from .att_isensee import gating_signal, expend_as, attention_block


def att_original_unet_model(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001,
                            deconvolution=False,depth=4, n_base_filters=32, metrics=dice_coefficient,
                            batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)

    current_layer = inputs

    levels = list()
    level_filters = list()
    for level_number in range(depth):  # depth=4; [0,1,2,3]

        n_level_filters = list()
        n_level_filters.append((2**level_number) * n_base_filters)  # 2^level_num * 16
        n_level_filters.append(n_level_filters[0] * 2)
        level_filters.append(n_level_filters)

        # add levels with max pooling
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_level_filters[0],
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_level_filters[1],
                                          batch_normalization=batch_normalization)
        if level_number < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:  # 最下一层
            current_layer = layer2
            levels.append([layer1, layer2])

    for level_number in range(depth - 2, -1, -1):  # [2,1,0]

        gating = gating_signal(levels[level_number + 1][1], level_filters[level_number][1], False)
        att = attention_block(levels[level_number][1], gating, level_filters[level_number][1])

        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=level_filters[level_number + 1][1])(current_layer)
        concat = concatenate([up_convolution, att], axis=1)
        current_layer = create_convolution_block(n_filters=level_filters[level_number][1],
                                                 input_layer=concat,
                                                 batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=level_filters[level_number][1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    # sigmoid激活输出
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss)
    # 返回模型
    return model



