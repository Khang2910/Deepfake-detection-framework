"""
import tensorflow as tf
import numpy as np
import tempfile
import cv2
import os

from transform import pad, PaddedVideo

# ==== Config ====
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
BATCH_SIZE = 2
NUM_EPOCHS = 10
MAX_FRAME = 128
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
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTO).map(lambda video, label: (pad(video, MAX_FRAME), label), num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(4)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTO)
    print(f"Dataset shape is: {dataset.element_spec}")
    return dataset

def get_input():
    content = tf.keras.Input((MAX_FRAME, TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=tf.float32)
    mask = tf.keras.Input((MAX_FRAME,), dtype='bool')
    return PaddedVideo(content, mask)


def get_train_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="train")
    return load_dataset(files, BATCH_SIZE, is_training=True)


def get_test_dataset(folder=TFRECORD_DIR):
    files = get_tfrecord_files(folder, split="test")
    return load_dataset(files, BATCH_SIZE, is_training=False)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tempfile
import cv2
import os
import re

from transform import pad, PaddedVideo

# ==== Config ====
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
BATCH_SIZE = 2
NUM_EPOCHS = 10
MAX_FRAME = 128
AUTO = tf.data.AUTOTUNE

PARQUET_DIR = "../Deepfake-detection"


def get_parquet_files(folder, split):
    pattern = re.compile(rf"{split}_\d+\.parquet")
    all_files = tf.io.gfile.listdir(folder)
    matched_files = [os.path.join(folder, f) for f in all_files if pattern.match(f)]
    if not matched_files:
        raise FileNotFoundError(f"No Parquet files found for split={split} in {folder}")
    print(f"[INFO] Found {len(matched_files)} {split} files")
    return matched_files


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
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            frames.append(frame)
        cap.release()
        os.remove(tmp.name)

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    return np.stack(frames).astype(np.float32) / 255.0


def decode_video_tf(video_bytes):
    video = tf.py_function(decode_video_py, [video_bytes], tf.float32)
    video.set_shape([None, TARGET_HEIGHT, TARGET_WIDTH, 3])
    return video


def parquet_generator(file_list):
    for file_path in file_list:
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            video_bytes = row["video"]
            label = row["label"]
            yield video_bytes, label


def load_dataset_parquet(file_list, batch_size, is_training=True):
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),  # raw video bytes
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: parquet_generator(file_list),
        output_signature=output_signature
    )

    dataset = dataset.map(
        lambda video_bytes, label: (decode_video_tf(video_bytes), tf.cast(label, tf.int32)),
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(lambda video, label: (pad(video, MAX_FRAME), label), num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(4).repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTO)
    print(f"Dataset shape is: {dataset.element_spec}")
    return dataset


def get_train_dataset(folder=PARQUET_DIR):
    files = get_parquet_files(folder, split="train")
    return load_dataset_parquet(files, BATCH_SIZE, is_training=True)


def get_test_dataset(folder=PARQUET_DIR):
    files = get_parquet_files(folder, split="test")
    return load_dataset_parquet(files, BATCH_SIZE, is_training=False)


def get_input():
    content = tf.keras.Input((MAX_FRAME, TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=tf.float32)
    mask = tf.keras.Input((MAX_FRAME,), dtype='bool')
    return PaddedVideo(content, mask)
