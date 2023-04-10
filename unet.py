import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input
from tensorflow.keras.models import Model

print(tf.config.list_physical_devices('GPU'))


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)

    return x, p


def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 518)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(
        d4)  # "sigmoid" for binary segmentation problems

    model = Model(inputs, outputs)

    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_unet(input_shape, 12)
    model.summary()
