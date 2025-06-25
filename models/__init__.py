import tensorflow as tf
from tensorflow.keras import layers

MODEL_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

@register_model("slowfast")
class SlowFast(tf.keras.Model):
    def __init__(self, num_classes=2, alpha=4, tau=32, **kwargs):
        """
        SlowFast model for video classification.
        alpha: temporal stride ratio (e.g., 4)
        tau: number of frames in fast path (e.g., 32)
        """
        super(SlowFast, self).__init__()
        self.alpha = alpha
        self.tau = tau

        self.slow_conv = tf.keras.Sequential([
            layers.Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', activation='relu'),
            layers.MaxPooling3D((1, 2, 2)),
            layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
            layers.GlobalAveragePooling3D()
        ])

        self.fast_conv = tf.keras.Sequential([
            layers.Conv3D(16, (3, 3, 3), strides=(1, 2, 2), padding='same', activation='relu'),
            layers.MaxPooling3D((1, 2, 2)),
            layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu'),
            layers.GlobalAveragePooling3D()
        ])

        self.fc = tf.keras.Sequential([
            layers.Concatenate(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=False):
        shape = tf.shape(inputs)
        length = shape[1]
        
        # Generate indices: [0, 4, 8, ..., length)
        indices = tf.range(0, length, delta=4)
        
        # Gather along time axis (axis=1)
        slow = tf.gather(inputs, indices, axis=1)

        # slow = inputs[:, ::self.alpha, :, :, :] Not compatible with TPU
        fast = inputs
        slow_feat = self.slow_conv(slow)
        fast_feat = self.fast_conv(fast)
        return self.fc([slow_feat, fast_feat])
