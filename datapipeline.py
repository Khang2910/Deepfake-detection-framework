import tensorflow as tf
import os

# ==== Config ====
VIDEO_SIZE = (224, 224)
FRAME_COUNT = 32
BATCH_SIZE = 8
NUM_EPOCHS = 10
AUTO = tf.data.AUTOTUNE

# ==== Global Path Variable (can be changed dynamically) ====
TFRECORD_DIR = "gs://deepfake-detection"  # or "./data" for local

def get_tfrecord_files(folder, split):
    """Return list of .tfrecord files in folder for train/test split"""
    pattern = f"dfdc_*_{split}.tfrecord"
    full_pattern = tf.io.gfile.join(folder, pattern)
    files = tf.io.gfile.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for split={split} in {folder}")
    print(f"[INFO] Found {len(files)} files for {split} in {folder}")
    return files

# ==== TFRecord Parsing ====
def _parse_tfrecord(example_proto):
    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    video = tf.io.decode_raw(example['video'], tf.uint8)

    # Assume shape: [T, H, W, 3] is flattened
    # Try to infer actual number of frames
    raw_size = tf.size(video)
    single_frame_size = 224 * 224 * 3
    total_frames = raw_size // single_frame_size

    video = tf.reshape(video, [total_frames, 224, 224, 3])
    video = tf.cast(video, tf.float32) / 255.0

    video = temporal_resize(video, target_frames=32)

    label = tf.cast(example['label'], tf.int32)
    return video, label

def temporal_resize(video, target_frames):
    """Uniformly sample or pad video to target number of frames."""
    num_frames = tf.shape(video)[0]

    def sample():
        indices = tf.linspace(0.0, tf.cast(num_frames - 1, tf.float32), target_frames)
        indices = tf.cast(indices, tf.int32)
        return tf.gather(video, indices)

    def pad():
        pad_len = target_frames - num_frames
        padding = tf.tile(video[-1:], [pad_len, 1, 1, 1])  # repeat last frame
        return tf.concat([video, padding], axis=0)

    return tf.cond(num_frames >= target_frames, sample, pad)

# ==== Dataset Loader ====
def load_dataset(file_list, batch_size, is_training=True):
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset

# ==== Main Loader Wrappers ====
def get_train_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="train")
    return load_dataset(files, BATCH_SIZE, is_training=True)

def get_test_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="test")
    return load_dataset(files, BATCH_SIZE, is_training=False)
