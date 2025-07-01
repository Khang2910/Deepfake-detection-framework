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
    """
    Args:
        inputs: PaddedVideoBatch(content, pad_mask)
            - content: tf.Tensor of shape [batch_size, max_length, H, W, C]
            - pad_mask: tf.Tensor of shape [batch_size, max_length] (bool)

    Returns:
        tf.Tensor of shape [batch_size, num_classes]
    """
    video = inputs.content
    pad_mask = inputs.pad_mask

    batch_size = tf.shape(video)[0]
    total_frames = tf.shape(video)[1]

    # === Select slow path frames ===
    slow_indices = tf.range(0, total_frames, delta=self.alpha)
    slow = tf.gather(video, slow_indices, axis=1)
    slow_mask = tf.gather(pad_mask, slow_indices, axis=1)

    # === Apply convolutions ===
    # Note: 3D Conv layers will inherently ignore padding since we padded with zeros,
    # but pooling or averaging might be biased if we don’t apply the mask properly.
    slow_feat = self.slow_conv(slow)
    fast_feat = self.fast_conv(video)

    # === Forward through final FC ===
    out = self.fc([slow_feat, fast_feat])
    return out

