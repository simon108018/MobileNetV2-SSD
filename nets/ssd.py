import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
# from keras.layers import AtrousConvolution2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ZeroPadding2D, Concatenate
from tensorflow.keras.models import Model
from nets.MobileNetV2 import MobileNetV2
from nets.ssd_layers import Normalize
from nets.ssd_layers import PriorBox


def SSD300(input_shape, num_classes=21):
    # 300,300,3
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    # SSD结构,net字典
    net = MobileNetV2(input_tensor)
    # -----------------------将提取到的主干特征进行处理---------------------------#
    # 对block_12_add进行处理
    # net["block_12_add_norm"] = Normalize(20, name='block_12_add_norm')(net['block_12_add'])
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['block_13_expand_relu_mbox_loc'] = Conv2D(filters=num_priors * 4,
                                              kernel_size=(3, 3),
                                              padding='same',
                                              name='block_13_expand_relu_mbox_loc')(net['block_13_expand_relu'])
    net['block_13_expand_relu_mbox_loc_flat'] = Flatten(name='block_13_expand_relu_mbox_loc_flat')(net['block_13_expand_relu_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['block_13_expand_relu_mbox_conf'] = Conv2D(filters=num_priors * num_classes,
                                               kernel_size=(3, 3),
                                               padding='same',
                                               name='block_13_expand_relu_mbox_conf')(net['block_13_expand_relu'])
    net['block_13_expand_relu_mbox_conf_flat'] = Flatten(name='block_13_expand_relu_mbox_conf_flat')(net['block_13_expand_relu_mbox_conf'])
    # priorbox的設定
    net['block_13_expand_relu_mbox_priorbox'] = PriorBox(img_size=img_size,
                                                      min_size=30.0,
                                                      max_size=60.0,
                                                      aspect_ratios=[2],
                                                      variances=[0.1, 0.1, 0.2, 0.2],
                                                      name='block_13_expand_relu_mbox_priorbox')(net['block_13_expand_relu'])

    # 对Conv_1_relu进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['Conv_1_relu_mbox_loc'] = Conv2D(filters=num_priors * 4,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         name='Conv_1_relu_mbox_loc')(net['Conv_1_relu'])
    net['Conv_1_relu_mbox_loc_flat'] = Flatten(name='Conv_1_relu_mbox_loc_flat')(net['Conv_1_relu_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['Conv_1_relu_mbox_conf'] = Conv2D(filters=num_priors * num_classes,
                                          kernel_size=(3, 3),
                                          padding='same',
                                          name='Conv_1_relu_mbox_conf')(net['Conv_1_relu'])
    net['Conv_1_relu_mbox_conf_flat'] = Flatten(name='Conv_1_relu_mbox_conf_flat')(net['Conv_1_relu_mbox_conf'])
    # priorbox的設定
    net['Conv_1_relu_mbox_priorbox'] = PriorBox(img_size=img_size,
                                                min_size=60.0,
                                                max_size=111.0,
                                                aspect_ratios=[2, 3],
                                                variances=[0.1, 0.1, 0.2, 0.2],
                                                name='Conv_1_relu_mbox_priorbox')(net['Conv_1_relu'])

    # 对conv_3_relu进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv_3_relu_mbox_loc'] = Conv2D(filters=num_priors * 4,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         name='conv_3_relu_mbox_loc')(net['conv_3_relu'])
    net['conv_3_relu_mbox_loc_flat'] = Flatten(name='conv_3_relu_mbox_loc_flat')(net['conv_3_relu_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv_3_relu_mbox_conf'] = Conv2D(filters=num_priors * num_classes,
                                          kernel_size=(3, 3),
                                          padding='same',
                                          name='conv_3_relu_mbox_conf')(net['conv_3_relu'])
    net['conv_3_relu_mbox_conf_flat'] = Flatten(name='conv_3_relu_mbox_conf_flat')(net['conv_3_relu_mbox_conf'])
    # priorbox的設定
    net['conv_3_relu_mbox_priorbox'] = PriorBox(img_size=img_size,
                                                min_size=111.0,
                                                max_size=162.0,
                                                aspect_ratios=[2, 3],
                                                variances=[0.1, 0.1, 0.2, 0.2],
                                                name='conv_3_relu_mbox_priorbox')(net['conv_3_relu'])

    # 对conv_4_relu进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv_4_relu_mbox_loc'] = Conv2D(filters=num_priors * 4,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         name='conv_4_relu_mbox_loc')(net['conv_4_relu'])
    net['conv_4_relu_mbox_loc_flat'] = Flatten(name='conv_4_relu_mbox_loc_flat')(net['conv_4_relu_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv_4_relu_mbox_conf'] = Conv2D(filters=num_priors * num_classes,
                                          kernel_size=(3, 3),
                                          padding='same',
                                          name='conv_4_relu_mbox_conf')(net['conv_4_relu'])
    net['conv_4_relu_mbox_conf_flat'] = Flatten(name='conv_4_relu_mbox_conf_flat')(net['conv_4_relu_mbox_conf'])
    # priorbox的設定
    net['conv_4_relu_mbox_priorbox'] = PriorBox(img_size=img_size,
                                                min_size=162.0,
                                                max_size=213.0,
                                                aspect_ratios=[2, 3],
                                                variances=[0.1, 0.1, 0.2, 0.2],
                                                name='conv_4_relu_mbox_priorbox')(net['conv_4_relu'])

    # 对conv_5_relu进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv_5_relu_mbox_loc'] = Conv2D(filters=num_priors * 4,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         name='conv_5_relu_mbox_loc')(net['conv_5_relu'])
    net['conv_5_relu_mbox_loc_flat'] = Flatten(name='conv_5_relu_mbox_loc_flat')(net['conv_5_relu_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv_5_relu_mbox_conf'] = Conv2D(filters=num_priors * num_classes,
                                          kernel_size=(3, 3),
                                          padding='same',
                                          name='conv_5_relu_mbox_conf')(net['conv_5_relu'])
    net['conv_5_relu_mbox_conf_flat'] = Flatten(name='conv_5_relu_mbox_conf_flat')(net['conv_5_relu_mbox_conf'])
    # priorbox的設定
    net['conv_5_relu_mbox_priorbox']  = PriorBox(img_size=img_size,
                                                 min_size=213.0,
                                                 max_size=264.0,
                                                 aspect_ratios=[2, 3],
                                                 variances=[0.1, 0.1, 0.2, 0.2],
                                                 name='conv_5_relu_mbox_priorbox')(net['conv_5_relu'])
    # 对conv_6_relu进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv_6_relu_mbox_loc'] = Conv2D(filters=num_priors * 4,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         name='conv_6_relu_mbox_loc')(net['conv_6_relu'])
    net['conv_6_relu_mbox_loc_flat'] = Flatten(name='conv_6_relu_mbox_loc_flat')(net['conv_6_relu_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv_6_relu_mbox_conf'] = Conv2D(filters=num_priors * num_classes,
                                          kernel_size=(3, 3),
                                          padding='same',
                                          name='conv_6_relu_mbox_conf')(net['conv_6_relu'])
    net['conv_6_relu_mbox_conf_flat'] = Flatten(name='conv_6_relu_mbox_conf_flat')(net['conv_6_relu_mbox_conf'])
    # priorbox的設定
    net['conv_6_relu_mbox_priorbox'] = PriorBox(img_size=img_size,
                                                min_size=264.0,
                                                max_size=315.0,
                                                aspect_ratios=[2, 3],
                                                variances=[0.1, 0.1, 0.2, 0.2],
                                                name='conv_6_relu_mbox_priorbox')(net['conv_6_relu'])

    # 将所有结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['block_13_expand_relu_mbox_loc_flat'],
                                                            net['Conv_1_relu_mbox_loc_flat'],
                                                            net['conv_3_relu_mbox_loc_flat'],
                                                            net['conv_4_relu_mbox_loc_flat'],
                                                            net['conv_5_relu_mbox_loc_flat'],
                                                            net['conv_6_relu_mbox_loc_flat']])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['block_13_expand_relu_mbox_conf_flat'],
                                                              net['Conv_1_relu_mbox_conf_flat'],
                                                              net['conv_3_relu_mbox_conf_flat'],
                                                              net['conv_4_relu_mbox_conf_flat'],
                                                              net['conv_5_relu_mbox_conf_flat'],
                                                              net['conv_6_relu_mbox_conf_flat']])

    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([net['block_13_expand_relu_mbox_priorbox'],
                                                                      net['Conv_1_relu_mbox_priorbox'],
                                                                      net['conv_3_relu_mbox_priorbox'],
                                                                      net['conv_4_relu_mbox_priorbox'],
                                                                      net['conv_5_relu_mbox_priorbox'],
                                                                      net['conv_6_relu_mbox_priorbox']])



    # 2194,4
    net['mbox_loc'] = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 2194,21
    net['mbox_conf'] = Reshape((-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = Concatenate(axis=2, name='predictions')([net['mbox_loc'],
                                                                  net['mbox_conf'],
                                                                  net['mbox_priorbox']])
    model = Model(net['input_1'], net['predictions'])
    return model


if __name__ == '__main__':
    import os
    from tensorflow.keras.utils import plot_model

    input_shape = (300, 300, 3)
    model = SSD300(input_shape)
    plot_model(model, to_file='test_model.png')
    print(model.summary())

