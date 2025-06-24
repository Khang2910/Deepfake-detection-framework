import tensorflow as tf
import os

# ==== Config ====
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE

TFRECORD_DIR = "gs://deepfake-detection"  # or "./data" for local


def get_tfrecord_files(folder, split):
    pattern = f"dfdc_*_{split}.tfrecord"
    full_pattern = tf.io.gfile.join(folder, pattern)
    files = tf.io.gfile.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for split={split} in {folder}")
    print(f"[INFO] Found {len(files)} files for {split} in {folder}")
    return files


def _parse_tfrecord(example_proto):
    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'frames': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Read shape info
    frames = tf.cast(example['frames'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    height = tf.cast(example['height'], tf.int32)

    # Decode and reshape video
    video = tf.io.decode_raw(example['video'], tf.uint8)
    video = tf.reshape(video, [frames, height, width, 3])
    video = tf.cast(video, tf.float32) / 255.0

    # Resize all frames to fixed spatial shape
    video = tf.image.resize(video, [TARGET_HEIGHT, TARGET_WIDTH])

    label = tf.cast(example['label'], tf.int32)
    return video, label


def load_dataset(file_list, batch_size, is_training=True):
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset


def get_train_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="train")
    return load_dataset(files, BATCH_SIZE, is_training=True)


def get_test_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="test")
    return load_dataset(files, BATCH_SIZE, is_training=False)
