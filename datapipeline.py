import tensorflow as tf
import os

# ==== Configuration Parameters ====
GCS_BUCKET = 'gs://your-bucket-name'
TFRECORD_PATH = f'{GCS_BUCKET}/data/train_records/*.tfrecord'  # or replace with parquet pipeline
VIDEO_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_EPOCHS = 10
AUTO = tf.data.AUTOTUNE
IS_TRAINING = True

# ==== TFRecord Parsing Function ====
def _parse_tfrecord_fn(example_proto):
    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),  # 0: real, 1: fake
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    video_raw = tf.io.decode_raw(example['video'], tf.uint8)
    # Suppose fixed-size [frames, H, W, C]
    video = tf.reshape(video_raw, [16, *VIDEO_SIZE, 3])  # Adjust shape as needed
    video = tf.cast(video, tf.float32) / 255.0
    label = tf.cast(example['label'], tf.int32)
    return video, label

# ==== Data Loader ====
def load_dataset(tfrecord_path, batch_size, is_training=True):
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_path))
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=AUTO)
    
    if is_training:
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

# ==== For Parquet (Optional Alternative Loader) ====
def load_parquet_dataset(parquet_path, batch_size):
    import pandas as pd
    import tensorflow_io as tfio  # If needed
    # You’ll need to use TF I/O or preprocess offline to tfrecord
    raise NotImplementedError("Parquet loading not implemented yet, consider converting to TFRecord.")

# ==== Main Entrypoint ====
def main():
    print("Loading dataset...")
    train_dataset = load_dataset(TFRECORD_PATH, BATCH_SIZE, is_training=IS_TRAINING)
    for video, label in train_dataset.take(1):
        print("Video shape:", video.shape)
        print("Label:", label.numpy())

if __name__ == "__main__":
    main()
