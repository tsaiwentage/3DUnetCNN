from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss

# 携带部分参数生成一个新函数
create_convolution_block = partial(create_convolution_block, instance_normalization=False)


def isensee_1_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
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
        level_output_layers.append(context_output_layer)
        current_layer = context_output_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):  # [3,2,1,0]
        # 上采样模块
        # 上采样放大一倍，卷积减少一半通道->conv_block
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        # concat：skip connection
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        # concat后两次卷积channel减半
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output

    output_layer = Conv3D(n_labels, (1, 1, 1))(current_layer)
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
    convolution2 = create_convolution_block(input_layer=convolution1, n_filters=n_level_filters)
    return convolution2



