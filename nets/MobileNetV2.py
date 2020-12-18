import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# gpu設定
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def MobileNetV2(input_tensor):
    # ----------------------------主干特征提取网络开始---------------------------#
    # SSD结构,net字典
    net = {}
    # Input
    net['input_1'] = input_tensor

    # Conv2D
    # 300, 300, 3 -> 150, 150, 32
    net['Conv1'] = Conv2D(filters=32,
                          kernel_size=(3, 3),
                          strides=(2, 2),
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name='Conv1')(net['input_1'])
    net['bn_Conv1'] = BatchNormalization(name='bn_Conv1',
                                         epsilon=1e-3,
                                         momentum=0.999)(net['Conv1'])
    net['Conv1_relu'] = ReLU(max_value=6,
                             name='Conv1_relu')(net['bn_Conv1'])
    ## bottleneck_1
    net['expanded_conv_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                     padding='same',
                                                     strides=(1, 1),
                                                     use_bias=False,
                                                     activation=None,
                                                     name='expanded_conv_depthwise')(net['Conv1_relu'])
    net['expanded_conv_depthwise_BN'] = BatchNormalization(name='expanded_conv_depthwise_BN',
                                                           epsilon=1e-3,
                                                           momentum=0.999)(net['expanded_conv_depthwise'])
    net['expanded_conv_depthwise_relu'] = ReLU(max_value=6,
                                               name='expanded_conv_depthwise_relu')(net['expanded_conv_depthwise_BN'])
    # 150, 150, 32 -> 150, 150, 16
    net['expanded_conv_project'] = Conv2D(filters=16,
                                          kernel_size=(1, 1),
                                          padding='same',
                                          use_bias=False,
                                          activation=None,
                                          name='expanded_conv_project')(net['expanded_conv_depthwise_relu'])
    net['expanded_conv_project_BN'] = BatchNormalization(name='BatchNormalization',
                                                         epsilon=1e-3,
                                                         momentum=0.999)(net['expanded_conv_project'])

    ## bottleneck_2
    # block 1
    # 150, 150, 16 -> 150, 150, 96 (16*6=96)
    net['block_1_expand'] = Conv2D(filters=96,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_1_expand')(net['expanded_conv_project_BN'])
    net['block_1_expand_BN'] = BatchNormalization(name='block_1_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_1_expand'])
    net['block_1_expand_relu'] = ReLU(max_value=6,
                                      name='block_1_expand_relu')(net['block_1_expand_BN'])
    # 150, 150, 96 -> 75, 75, 96
    net['block_1_pad'] = ZeroPadding2D(padding=imagenet_utils.correct_pad(net['block_1_expand_relu'], 3),
                                              name='block_1_pad')(net['block_1_expand_relu'])
    net['block_1_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='valid',
                                               strides=(2, 2),
                                               use_bias=False,
                                               activation=None,
                                               name='block_1_depthwise')(net['block_1_pad'])
    net['block_1_depthwise_BN'] = BatchNormalization(name='block_1_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_1_depthwise'])
    net['block_1_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_1_depthwise_relu')(net['block_1_depthwise_BN'])
    # 75, 75, 96 -> 75, 75, 24
    net['block_1_project'] = Conv2D(filters=24,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_1_project')(net['block_1_depthwise_relu'])
    net['block_1_project_BN'] = BatchNormalization(name='block_1_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_1_project'])
    ## bottleneck_2
    # block 2
    # 75, 75, 24 -> 75, 75, 144 (24*6=144)
    net['block_2_expand'] = Conv2D(filters=144,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_2_expand')(net['block_1_project_BN'])
    net['block_2_expand_BN'] = BatchNormalization(name='block_2_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_2_expand'])
    net['block_2_expand_relu'] = ReLU(max_value=6,
                                      name='block_2_expand_relu')(net['block_2_expand_BN'])
    net['block_2_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='same',
                                               strides=(1, 1),
                                               use_bias=False,
                                               activation=None,
                                               name='block_2_depthwise')(net['block_2_expand_relu'])
    net['block_2_depthwise_BN'] = BatchNormalization(name='block_2_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_2_depthwise'])
    net['block_2_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_2_depthwise_relu')(net['block_2_depthwise_BN'])
    # 75, 75, 144 -> 75, 75, 24
    net['block_2_project'] = Conv2D(filters=24,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_2_project')(net['block_2_depthwise_relu'])
    net['block_2_project_BN'] = BatchNormalization(name='block_2_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_2_project'])
    net['block_2_add'] = Add(name='blick_2_add')([net['block_1_project_BN'], net['block_2_project_BN']])
    # block 3
    # 75, 75, 24 -> 75, 75, 144 (24*6=144)
    net['block_3_expand'] = Conv2D(filters=144,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_3_expand')(net['block_2_add'])
    net['block_3_expand_BN'] = BatchNormalization(name='block_3_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_3_expand'])
    net['block_3_expand_relu'] = ReLU(max_value=6,
                                      name='block_3_expand_relu')(net['block_3_expand_BN'])
    # 75, 75, 144 -> 38, 38, 144
    net['block_3_pad'] = ZeroPadding2D(padding=imagenet_utils.correct_pad(net['block_3_expand_relu'], 3),
                                              name='block_3_pad')(net['block_3_expand_relu'])
    net['block_3_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='valid',
                                               strides=(2, 2),
                                               use_bias=False,
                                               activation=None,
                                               name='block_3_depthwise')(net['block_3_pad'])
    net['block_3_depthwise_BN'] = BatchNormalization(name='block_3_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_3_depthwise'])
    net['block_3_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_3_depthwise_relu')(net['block_3_depthwise_BN'])
    # 38, 38, 144 -> 38, 38, 32
    net['block_3_project'] = Conv2D(filters=32,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_3_project')(net['block_3_depthwise_relu'])
    net['block_3_project_BN'] = BatchNormalization(name='block_3_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_3_project'])
    ## bottleneck_3
    # block 4
    # 38, 38, 32 -> 38, 38, 192 (32*6=192)
    net['block_4_expand'] = Conv2D(filters=192,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_4_expand')(net['block_3_project_BN'])
    net['block_4_expand_BN'] = BatchNormalization(name='block_4_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_4_expand'])
    net['block_4_expand_relu'] = ReLU(max_value=6,
                                      name='block_4_expand_relu')(net['block_4_expand_BN'])
    net['block_4_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='same',
                                               strides=(1, 1),
                                               use_bias=False,
                                               activation=None,
                                               name='block_4_depthwise')(net['block_4_expand_relu'])
    net['block_4_depthwise_BN'] = BatchNormalization(name='block_4_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_4_depthwise'])
    net['block_4_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_4_depthwise_relu')(net['block_4_depthwise_BN'])
    # 38, 38, 192 -> 38, 38, 32
    net['block_4_project'] = Conv2D(filters=32,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_4_project')(net['block_4_depthwise_relu'])
    net['block_4_project_BN'] = BatchNormalization(name='block_4_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_4_project'])
    net['block_4_add'] = Add(name='blick_4_add')([net['block_3_project_BN'], net['block_4_project_BN']])

    # block 5
    # 38, 38, 32 -> 38, 38, 192 (38*6=192)
    net['block_5_expand'] = Conv2D(filters=192,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_5_expand')(net['block_4_add'])
    net['block_5_expand_BN'] = BatchNormalization(name='block_5_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_5_expand'])
    net['block_5_expand_relu'] = ReLU(max_value=6,
                                      name='block_5_expand_relu')(net['block_5_expand_BN'])
    net['block_5_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='same',
                                               strides=(1, 1),
                                               use_bias=False,
                                               activation=None,
                                               name='block_5_depthwise')(net['block_5_expand_relu'])
    net['block_5_depthwise_BN'] = BatchNormalization(name='block_5_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_5_depthwise'])
    net['block_5_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_5_depthwise_relu')(net['block_5_depthwise_BN'])
    # 38, 38, 192 -> 38, 38, 32
    net['block_5_project'] = Conv2D(filters=32,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_5_project')(net['block_5_depthwise_relu'])
    net['block_5_project_BN'] = BatchNormalization(name='block_5_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_5_project'])
    net['block_5_add'] = Add(name='block_5_add')([net['block_4_add'], net['block_5_project_BN']])
    # block 6
    # 38, 38, 32 -> 38, 38, 192 (38*6=192)
    net['block_6_expand'] = Conv2D(filters=192,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_6_expand')(net['block_5_add'])
    net['block_6_expand_BN'] = BatchNormalization(name='block_6_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_6_expand'])
    net['block_6_expand_relu'] = ReLU(max_value=6,
                                      name='block_6_expand_relu')(net['block_6_expand_BN'])
    # 38, 38, 192 -> 19, 19, 192
    net['block_6_pad'] = ZeroPadding2D(padding=imagenet_utils.correct_pad(net['block_6_expand_relu'], 3),
                                       name='block_6_pad')(net['block_6_expand_relu'])
    net['block_6_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='valid',
                                               strides=(2, 2),
                                               use_bias=False,
                                               activation=None,
                                               name='block_6_depthwise')(net['block_6_pad'])
    net['block_6_depthwise_BN'] = BatchNormalization(name='block_6_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_6_depthwise'])
    net['block_6_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_6_depthwise_relu')(net['block_6_depthwise_BN'])
    # 19, 19, 192 -> 19, 19, 64
    net['block_6_project'] = Conv2D(filters=64,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_6_project')(net['block_6_depthwise_relu'])
    net['block_6_project_BN'] = BatchNormalization(name='block_6_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_6_project'])
    ## bottleneck_4
    # block 7
    # 19, 19, 64 -> 19, 19, 384 (64*6=384)
    net['block_7_expand'] = Conv2D(filters=384,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_7_expand')(net['block_6_project_BN'])
    net['block_7_expand_BN'] = BatchNormalization(name='block_7_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_7_expand'])
    net['block_7_expand_relu'] = ReLU(max_value=6,
                                      name='block_7_expand_relu')(net['block_7_expand_BN'])
    net['block_7_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='same',
                                               strides=(1, 1),
                                               use_bias=False,
                                               activation=None,
                                               name='block_7_depthwise')(net['block_7_expand_relu'])
    net['block_7_depthwise_BN'] = BatchNormalization(name='block_7_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_7_depthwise'])
    net['block_7_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_7_depthwise_relu')(net['block_7_depthwise_BN'])
    net['block_7_project'] = Conv2D(filters=64,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_7_project')(net['block_7_depthwise_relu'])
    net['block_7_project_BN'] = BatchNormalization(name='block_7_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_7_project'])
    net['block_7_add'] = Add(name='block_7_add')([net['block_6_project_BN'], net['block_7_project_BN']])
    # block 8
    # 19, 19, 64 -> 19, 19, 384 (64*6=384)
    net['block_8_expand'] = Conv2D(filters=384,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_8_expand')(net['block_7_add'])
    net['block_8_expand_BN'] = BatchNormalization(name='block_8_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_8_expand'])
    net['block_8_expand_relu'] = ReLU(max_value=6,
                                      name='block_8_expand_relu')(net['block_8_expand_BN'])
    net['block_8_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='same',
                                               strides=(1, 1),
                                               use_bias=False,
                                               activation=None,
                                               name='block_8_depthwise')(net['block_8_expand_relu'])
    net['block_8_depthwise_BN'] = BatchNormalization(name='block_8_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_8_depthwise'])
    net['block_8_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_8_depthwise_relu')(net['block_8_depthwise_BN'])
    net['block_8_project'] = Conv2D(filters=64,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_8_project')(net['block_8_depthwise_relu'])
    net['block_8_project_BN'] = BatchNormalization(name='block_8_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_8_project'])
    net['block_8_add'] = Add(name='block_8_add')([net['block_7_add'], net['block_8_project_BN']])
    # block 9
    # 19, 19, 64 -> 19, 19, 384 (64*6=384)
    net['block_9_expand'] = Conv2D(filters=384,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name='block_9_expand')(net['block_8_add'])
    net['block_9_expand_BN'] = BatchNormalization(name='block_9_expand_BN',
                                                  epsilon=1e-3,
                                                  momentum=0.999)(net['block_9_expand'])
    net['block_9_expand_relu'] = ReLU(max_value=6,
                                      name='block_9_expand_relu')(net['block_9_expand_BN'])
    net['block_9_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                               padding='same',
                                               strides=(1, 1),
                                               use_bias=False,
                                               activation=None,
                                               name='block_9_depthwise')(net['block_9_expand_relu'])
    net['block_9_depthwise_BN'] = BatchNormalization(name='block_9_depthwise_BN',
                                                     epsilon=1e-3,
                                                     momentum=0.999)(net['block_9_depthwise'])
    net['block_9_depthwise_relu'] = ReLU(max_value=6,
                                         name='block_9_depthwise_relu')(net['block_9_depthwise_BN'])
    # 19, 19, 384 -> 19, 19, 64
    net['block_9_project'] = Conv2D(filters=64,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_9_project')(net['block_9_depthwise_relu'])
    net['block_9_project_BN'] = BatchNormalization(name='block_9_project_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_9_project'])
    net['block_9_add'] = Add(name='block_9_add')([net['block_8_add'], net['block_9_project_BN']])
    # block 10
    # 19, 19, 64 -> 19, 19, 384 (64*6=384)
    net['block_10_expand'] = Conv2D(filters=384,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_10_expand')(net['block_9_add'])
    net['block_10_expand_BN'] = BatchNormalization(name='block_10_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_10_expand'])
    net['block_10_expand_relu'] = ReLU(max_value=6,
                                       name='block_10_expand_relu')(net['block_10_expand_BN'])
    net['block_10_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='same',
                                                strides=(1, 1),
                                                use_bias=False,
                                                activation=None,
                                                name='block_10_depthwise')(net['block_10_expand_relu'])
    net['block_10_depthwise_BN'] = BatchNormalization(name='block_10_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_10_depthwise'])
    net['block_10_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_10_depthwise_relu')(net['block_10_depthwise_BN'])
    # 19, 19, 384 -> 19, 19, 96
    net['block_10_project'] = Conv2D(filters=96,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_10_project')(net['block_10_depthwise_relu'])
    net['block_10_project_BN'] = BatchNormalization(name='block_10_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_10_project'])
    ## bottleneck_5
    # block 11
    # 19, 19, 96 -> 19, 19, 576 (96*6=576)
    net['block_11_expand'] = Conv2D(filters=576,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_11_expand')(net['block_10_project_BN'])
    net['block_11_expand_BN'] = BatchNormalization(name='block_11_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_11_expand'])
    net['block_11_expand_relu'] = ReLU(max_value=6,
                                       name='block_11_expand_relu')(net['block_11_expand_BN'])
    net['block_11_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='same',
                                                strides=(1, 1),
                                                use_bias=False,
                                                activation=None,
                                                name='block_11_depthwise')(net['block_11_expand_relu'])
    net['block_11_depthwise_BN'] = BatchNormalization(name='block_11_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_11_depthwise'])
    net['block_11_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_11_depthwise_relu')(net['block_11_depthwise_BN'])
    # 19, 19, 576 -> 19, 19, 96
    net['block_11_project'] = Conv2D(filters=96,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_11_project')(net['block_11_depthwise_relu'])
    net['block_11_project_BN'] = BatchNormalization(name='block_11_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_11_project'])
    net['block_11_add'] = Add(name='block_11_add')([net['block_10_project_BN'], net['block_11_project_BN']])
    # block 12
    # 19, 19, 96 -> 19, 19, 576 (96*6=576)
    net['block_12_expand'] = Conv2D(filters=576,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_12_expand')(net['block_11_add'])
    net['block_12_expand_BN'] = BatchNormalization(name='block_12_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_12_expand'])
    net['block_12_expand_relu'] = ReLU(max_value=6,
                                       name='block_12_expand_relu')(net['block_12_expand_BN'])
    net['block_12_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='same',
                                                strides=(1, 1),
                                                use_bias=False,
                                                activation=None,
                                                name='block_12_depthwise')(net['block_12_expand_relu'])
    net['block_12_depthwise_BN'] = BatchNormalization(name='block_12_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_12_depthwise'])
    net['block_12_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_12_depthwise_relu')(net['block_12_depthwise_BN'])
    # 19, 19, 576 -> 19, 19, 96
    net['block_12_project'] = Conv2D(filters=96,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_12_project')(net['block_12_depthwise_relu'])
    net['block_12_project_BN'] = BatchNormalization(name='block_12_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_12_project'])
    net['block_12_add'] = Add(name='block_12_add')([net['block_11_add'], net['block_12_project_BN']])
    # block 13
    # 19, 19, 96 -> 19, 19, 576 (96*6=576)
    net['block_13_expand'] = Conv2D(filters=576,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_13_expand')(net['block_12_add'])
    net['block_13_expand_BN'] = BatchNormalization(name='block_13_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_13_expand'])
    net['block_13_expand_relu'] = ReLU(max_value=6,
                                       name='block_13_expand_relu')(net['block_13_expand_BN'])
    # 19, 19, 576 -> 10, 10, 576
    net['block_13_pad'] = ZeroPadding2D(padding=imagenet_utils.correct_pad(net['block_13_expand_relu'], 3),
                                       name='block_13_pad')(net['block_13_expand_relu'])
    net['block_13_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='valid',
                                                strides=(2, 2),
                                                use_bias=False,
                                                activation=None,
                                                name='block_13_depthwise')(net['block_13_pad'])
    net['block_13_depthwise_BN'] = BatchNormalization(name='block_13_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_13_depthwise'])
    net['block_13_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_13_depthwise_relu')(net['block_13_depthwise_BN'])
    # 10, 10, 576 -> 10, 10, 160
    net['block_13_project'] = Conv2D(filters=160,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_13_project')(net['block_13_depthwise_relu'])
    net['block_13_project_BN'] = BatchNormalization(name='block_13_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_13_project'])
    ## bottleneck_6
    # block 14
    # 10, 10, 160 -> 10, 10, 960 (160*6=960)
    net['block_14_expand'] = Conv2D(filters=960,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_14_expand')(net['block_13_project_BN'])
    net['block_14_expand_BN'] = BatchNormalization(name='block_14_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_14_expand'])
    net['block_14_expand_relu'] = ReLU(max_value=6,
                                       name='block_14_expand_relu')(net['block_14_expand_BN'])
    net['block_14_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='same',
                                                strides=(1, 1),
                                                use_bias=False,
                                                activation=None,
                                                name='block_14_depthwise')(net['block_14_expand_relu'])
    net['block_14_depthwise_BN'] = BatchNormalization(name='block_14_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_14_depthwise'])
    net['block_14_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_14_depthwise_relu')(net['block_14_depthwise_BN'])
    # 10, 10, 960 -> 10, 10, 160
    net['block_14_project'] = Conv2D(filters=160,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_14_project')(net['block_14_depthwise_relu'])
    net['block_14_project_BN'] = BatchNormalization(name='block_14_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_14_project'])
    net['block_14_add'] = Add(name='block_14_add')([net['block_13_project_BN'], net['block_14_project_BN']])
    # block 15
    # 10, 10, 160 -> 10, 10, 960 (160*6=960)
    net['block_15_expand'] = Conv2D(filters=960,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_15_expand')(net['block_14_add'])
    net['block_15_expand_BN'] = BatchNormalization(name='block_15_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_15_expand'])
    net['block_15_expand_relu'] = ReLU(max_value=6,
                                       name='block_15_expand_relu')(net['block_15_expand_BN'])
    net['block_15_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='same',
                                                strides=(1, 1),
                                                use_bias=False,
                                                activation=None,
                                                name='block_15_depthwise')(net['block_15_expand_relu'])
    net['block_15_depthwise_BN'] = BatchNormalization(name='block_15_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_15_depthwise'])
    net['block_15_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_15_depthwise_relu')(net['block_15_depthwise_BN'])
    # 10, 10, 960 -> 10, 10, 160
    net['block_15_project'] = Conv2D(filters=160,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_15_project')(net['block_15_depthwise_relu'])
    net['block_15_project_BN'] = BatchNormalization(name='block_15_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_15_project'])
    net['block_15_add'] = Add(name='block_15_add')([net['block_14_add'], net['block_15_project_BN']])
    # block 16
    # 10, 10, 160 -> 10, 10, 960 (160*6=960)
    net['block_16_expand'] = Conv2D(filters=960,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='block_16_expand')(net['block_15_add'])
    net['block_16_expand_BN'] = BatchNormalization(name='block_16_expand_BN',
                                                   epsilon=1e-3,
                                                   momentum=0.999)(net['block_16_expand'])
    net['block_16_expand_relu'] = ReLU(max_value=6,
                                       name='block_16_expand_relu')(net['block_16_expand_BN'])
    net['block_16_depthwise'] = DepthwiseConv2D(kernel_size=(3, 3),
                                                padding='same',
                                                strides=(1, 1),
                                                use_bias=False,
                                                activation=None,
                                                name='block_16_depthwise')(net['block_16_expand_relu'])
    net['block_16_depthwise_BN'] = BatchNormalization(name='block_16_depthwise_BN',
                                                      epsilon=1e-3,
                                                      momentum=0.999)(net['block_16_depthwise'])
    net['block_16_depthwise_relu'] = ReLU(max_value=6,
                                          name='block_16_depthwise_relu')(net['block_16_depthwise_BN'])
    # 10, 10, 960 -> 10, 10, 320
    net['block_16_project'] = Conv2D(filters=320,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation=None,
                                     name='block_16_project')(net['block_16_depthwise_relu'])
    net['block_16_project_BN'] = BatchNormalization(name='block_16_project_BN',
                                                    epsilon=1e-3,
                                                    momentum=0.999)(net['block_16_project'])
    # 10, 10, 320 -> 10, 10, 1024
    net['Conv_1'] = Conv2D(filters=1024,
                           kernel_size=(1, 1),
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name='Conv_1')(net['block_16_project_BN'])
    net['Conv_1_bn'] = BatchNormalization(name='Conv_1_bn',
                                          epsilon=1e-3,
                                          momentum=0.999)(net['Conv_1'])
    net['Conv_1_relu'] = ReLU(name='Conv_1_relu')(net['Conv_1_bn'])

    # 接續架構
    # 10, 10, 1024 -> 5, 5, 512
    net['conv_3'] = Conv2D(filters=512,
                           kernel_size=(3, 3),
                           strides=(2, 2),
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name='conv_3')(net['Conv_1_relu'])
    net['conv_3_bn'] = BatchNormalization(name='conv_3_bn',
                                          epsilon=1e-3,
                                          momentum=0.999)(net['conv_3'])
    net['conv_3_relu'] = ReLU(name='conv_3_relu')(net['conv_3_bn'])
    # Block 8
    # 5, 5, 512 -> 3, 3, 256
    net['conv_4'] = Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=(2, 2),
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name='conv_4')(net['conv_3_relu'])
    net['conv_4_bn'] = BatchNormalization(name='conv_4_bn',
                                          epsilon=1e-3,
                                          momentum=0.999)(net['conv_4'])
    net['conv_4_relu'] = ReLU(name='conv_4_relu')(net['conv_4_bn'])

    # Block 9
    # 3, 3, 256 -> 2, 2, 256
    net['conv_5'] = Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=(2, 2),
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name='conv_5')(net['conv_4_relu'])
    net['conv_5_bn'] = BatchNormalization(name='conv_5_bn',
                                          epsilon=1e-3,
                                          momentum=0.999)(net['conv_5'])
    net['conv_5_relu'] = ReLU(name='conv_5_relu')(net['conv_5_bn'])

    # Block 10
    # 2, 2, 256 -> 1, 1, 128
    net['conv_6'] = Conv2D(filters=128,
                           kernel_size=(3, 3),
                           strides=(2, 2),
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name='conv_6')(net['conv_5_relu'])
    net['conv_6_bn'] = BatchNormalization(name='conv_6_bn',
                                          epsilon=1e-3,
                                          momentum=0.999)(net['conv_6'])
    net['conv_6_relu'] = ReLU(name='conv_6_relu')(net['conv_6_bn'])



    # ----------------------------主干特征提取网络结束---------------------------#
    return net


if __name__ == '__main__':
    import os
    # from getdata import get_data
    input_shape = (300, 300)
    input_tensor = Input(shape=input_shape + (3,))
    img_size = (input_shape[1], input_shape[0])
    net1 = MobileNetV2(input_tensor)
    model = Model(net1['input_1'], net1['conv_6_relu'])
    plot_model(model, to_file='test_model.png')
    print(model.summary())
    # data = get_data(dataname='cifar10', batch_size=16, reshape=input_shape + (3,))
    # data.run()
    # train_data = data.train_data
    # valid_data = data.valid_data
    # test_data = data.test_data
    # print('已取得 train_data、valid_data、test_data')
    #
    # # base_model = MobileNetV2(include_top=True, weights='imagenet', pooling='avg', input_shape=input_shape + (3,))
    # # base_model.save_weights('myNet.h5')
    # # del base_model
    # # model.load_weights('myNet.h5', by_name=True, skip_mismatch=True)
    #
    # model.load_weights('myNet.h5', by_name=True, skip_mismatch=True)
    #
    # # 設定路徑與建立資料夾
    # model_dir = 'cifar10/myNet'
    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)
    # model.compile(keras.optimizers.Adam(),
    #               loss=keras.losses.CategoricalCrossentropy(),
    #               metrics=[keras.metrics.CategoricalAccuracy()])
    # # model.fit(train_data, epochs=10)
    # log_dir = os.path.join('cifar10', 'myNet')
    # model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50, update_freq='epoch')
    # model_mckp = keras.callbacks.ModelCheckpoint(model_dir + "/myNet.h5", monitor='val_categorical_accuracy', mode='max')
    # history = model.fit(train_data, initial_epoch=0, epochs=50, validation_data=valid_data, callbacks=[model_cbk, model_mckp])
    # model.evaluate(train_data)
