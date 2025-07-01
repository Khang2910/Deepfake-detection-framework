import tensorflow as tf
from tensorflow.keras import layers
from collections import namedtuple

PaddedVideoBatch = namedtuple("PaddedVideoBatch", ["content", "pad_mask"])

MODEL_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def masked_avg_pool(inputs, mask):
    """
    Masked average over time axis (axis=1).

    Args:
        inputs: Tensor of shape [B, T, ...]
        mask: Bool tensor of shape [B, T]

    Returns:
        Tensor of shape [B, ...] (aggregated over T)
    """
    mask = tf.cast(mask, dtype=inputs.dtype)  # [B, T]
    mask = mask[..., tf.newaxis, tf.newaxis, tf.newaxis]

    masked_inputs = inputs * mask  # [B, T, H, W, C]
    sum_feat = tf.reduce_sum(masked_inputs, axis=1)  # [B, H, W, C]
    valid_counts = tf.reduce_sum(mask, axis=1)  # [B, 1, 1, 1]

    return sum_feat / (valid_counts + 1e-6)

@register_model("slowfast")
class SlowFast(tf.keras.Model):
    def __init__(self, num_classes=2, alpha=4, **kwargs):
        """
        Mask-aware SlowFast model for video classification.
        alpha: temporal stride ratio (e.g., 4)
        """
        super(SlowFast, self).__init__()
        self.alpha = alpha

        # Slow Path
        self.slow_conv = tf.keras.Sequential([
            layers.Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', activation='relu'),
            layers.MaxPooling3D((1, 2, 2), padding='same'),
            layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
        ])

        # Fast Path
        self.fast_conv = tf.keras.Sequential([
            layers.Conv3D(16, (3, 3, 3), strides=(1, 2, 2), padding='same', activation='relu'),
            layers.MaxPooling3D((1, 2, 2), padding='same'),
            layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu'),
        ])

        # Fully Connected
        self.fc = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs: PaddedVideoBatch, training=False):
        video = inputs.content  # [B, T, H, W, C]
        pad_mask = inputs.pad_mask  # [B, T]

        # Compute slow path (sample every alpha-th frame)
        total_frames = tf.shape(video)[1]
        slow_indices = tf.range(0, total_frames, delta=self.alpha)
        slow_video = tf.gather(video, slow_indices, axis=1)
        slow_mask = tf.gather(pad_mask, slow_indices, axis=1)

        # Conv3D expects [B, T, H, W, C]
        slow_feat = self.slow_conv(slow_video)  # [B, T', H, W, C]
        fast_feat = self.fast_conv(video)

        # Masked average pooling over time
        slow_pooled = masked_avg_pool(slow_feat, slow_mask)  # [B, H, W, C]
        fast_pooled = masked_avg_pool(fast_feat, pad_mask)

        # Global spatial pooling
        slow_global = tf.reduce_mean(slow_pooled, axis=[1, 2])  # [B, C]
        fast_global = tf.reduce_mean(fast_pooled, axis=[1, 2])  # [B, C]

        # Final FC head
        features = tf.concat([slow_global, fast_global], axis=-1)
        return self.fc(features)
