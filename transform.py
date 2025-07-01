import tensorflow as tf
from collections import namedtuple

# Named tuple to return padded sample
PaddedVideo = namedtuple("PaddedVideo", ["content", "pad_mask"])

def pad(video_tensor, max_length):
    """
    Pad a single video tensor (unbatched) to fixed length.

    Args:
        video_tensor: tf.Tensor of shape [num_frames, height, width, channels]
        max_length: int, the length to pad/truncate to

    Returns:
        PaddedVideo namedtuple with:
            - content: tf.Tensor of shape [max_length, height, width, channels]
            - pad_mask: tf.Tensor of shape [max_length], 1 for valid, 0 for padded
    """
    num_frames = tf.shape(video_tensor)[0]
    height = tf.shape(video_tensor)[1]
    width = tf.shape(video_tensor)[2]
    channels = tf.shape(video_tensor)[3]

    pad_len = tf.maximum(0, max_length - num_frames)
    truncated_video = video_tensor[:max_length]
    
    padding = tf.zeros([pad_len, height, width, channels], dtype=video_tensor.dtype)
    padded_video = tf.concat([truncated_video, padding], axis=0)

    valid_len = tf.minimum(num_frames, max_length)
    pad_mask = tf.concat([
        tf.ones([valid_len], dtype=tf.bool),
        tf.zeros([pad_len], dtype=tf.bool)
    ], axis=0)

    padded_video.set_shape([max_length, None, None, None])
    pad_mask.set_shape([max_length])

    return PaddedVideo(content=padded_video, pad_mask=pad_mask)
