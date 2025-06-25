import tensorflow as tf
import numpy as np
import tempfile
import cv2
import os

# ==== Config ====
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
BATCH_SIZE = 8
NUM_EPOCHS = 10
AUTO = tf.data.AUTOTUNE

# TFRECORD_DIR = "gs://deepfake-detection/metadata_parts"  # or "./data" for local
TFRECORD_DIR = '../Deepfake-detection'


def get_tfrecord_files(folder, split):
    pattern = f"dfdc_*_{split}_meta.tfrecord"
    full_pattern = tf.io.gfile.join(folder, pattern)
    files = tf.io.gfile.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for split={split} in {folder}")
    print(f"[INFO] Found {len(files)} files for {split} in {folder}")
    return files

def decode_video_py(video_bytes):

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes.numpy())
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        os.remove(tmp.name)

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    return np.stack(frames).astype(np.float32) / 255.0

def decode_video_tf(video_bytes):
    video = tf.py_function(decode_video_py, [video_bytes], tf.float32)
    video.set_shape([None, 224, 224, 3])  # allow variable-length videos
    return video

def _parse_tfrecord(example_proto):
    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'num_frames': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode and reshape raw video
    video_bytes = example["video"]
    video = decode_video_tf(video_bytes)
    label = tf.cast(example['label'], tf.int32)
    return video, label


def load_dataset(file_list, batch_size, is_training=True):
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(128)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset


def get_train_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="train")
    return load_dataset(files, BATCH_SIZE, is_training=True)


def get_test_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="test")
    return load_dataset(files, BATCH_SIZE, is_training=False)
