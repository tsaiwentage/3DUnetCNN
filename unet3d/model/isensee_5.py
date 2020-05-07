from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, \
    BatchNormalization, Conv3DTranspose, multiply, Lambda, Reshape
from keras.engine import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss

from keras import backend as K

# 携带部分参数生成一个新函数
create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee_5_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                    n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                    loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    # 每个深度的输出
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):  # depth=5; [0,1,2,3,4]
        # Encoder结构构建 (残差)

        n_level_filters = (2**level_number) * n_base_filters  # 2^level_num * 16
        level_filters.append(n_level_filters)

        # convolution_block：继承自原始unet，卷积->(正则)->激活(ReLU/LeakyReLU)
        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)  # 第一层通道数*4，尺寸不变
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2)) # 通道数*2，尺寸减半

        # 残差单元：conv_block->dropout->conv_block
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        # 残差模块：残差单元+in_conv
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):  # [3,2,1,0]
        # attention
        gating = gating_signal(level_output_layers[level_number + 1], level_filters[level_number], True)
        att = attention_block(level_output_layers[level_number], gating, level_filters[level_number])

        # 上采样模块
        # 上采样放大一倍，卷积减少一半通道->conv_block
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        # concat：skip connection
        concatenation_layer = concatenate([att, up_sampling], axis=1)
        # concat后两次卷积channel减半
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:  # 3
            # 记录层
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)
    activation_block = Activation(activation_name)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    # 返回模型
    return model


def create_localization_module(input_layer, n_filters):
    # concat后的两次卷积
    # channel减半
    convolution1 = create_convolution_block(input_layer, n_filters)
    # channel不变
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    # 上采样单元：上采样 (放大一倍)->卷积->(正则)->激活(ReLU/LeakyReLU)
    up_sample = UpSampling3D(size=size)(input_layer)
    # 通道减少一半
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    # 残差单元：conv_block->dropout->conv_block
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv3D(out_size, (1, 1, 1), padding='same')(input)
    if batch_norm:
        # x = BatchNormalization()(x)
        x = InstanceNormalization(axis=1)(x)
    # x = Activation('relu')(x)
    x = LeakyReLU()(x)
    return x


def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1), arguments={'repnum': rep})(tensor)


def attention_block(x, gating, inter_shape, res=False):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv3D(inter_shape, (1, 1, 1), padding='same')(gating)
    upsample_g = Conv3DTranspose(inter_shape, (3, 3, 3),
                                 strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3],
                                          shape_theta_x[4] // shape_g[4]),
                                 padding='same')(phi_g)

    concat_xg = Add()([upsample_g, theta_x])
    # act_xg = Activation('relu')(concat_xg)
    act_xg = LeakyReLU()(concat_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling3D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3],
                                      shape_x[4] // shape_sigmoid[4]))(sigmoid_xg)
    upsample_psi = expend_as(upsample_psi, shape_x[1])

    y = multiply([upsample_psi, x])
    if res:
        y = Add()([y, x])
        y = LeakyReLU()(y)
    result = Conv3D(shape_x[1], (1, 1, 1), padding='same')(y)
    # result_bn = BatchNormalization()(result)
    # return result_bn
    result_in = InstanceNormalization(axis=1)(result)
    result_act = LeakyReLU()(result_in)
    return result_act

