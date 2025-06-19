import tensorflow as tf
import os

# ==== Configuration ====
BUCKET_PATH = 'gs://deepfake-detection'
TRAIN_PATTERN = os.path.join(BUCKET_PATH, 'dfdc_*_train.tfrecord')
TEST_PATTERN  = os.path.join(BUCKET_PATH, 'dfdc_*_test.tfrecord')
VIDEO_SIZE = (224, 224)
FRAME_COUNT = 32
BATCH_SIZE = 8
NUM_EPOCHS = 10
AUTO = tf.data.AUTOTUNE

# ==== TFRecord Parsing ====
def _parse_tfrecord(example_proto):
    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    video = tf.io.decode_raw(example['video'], tf.uint8)
    video = tf.reshape(video, [FRAME_COUNT, *VIDEO_SIZE, 3])
    video = tf.cast(video, tf.float32) / 255.0
    label = tf.cast(example['label'], tf.int32)
    return video, label

# ==== Dataset Loader ====
def load_dataset(file_pattern, batch_size, is_training=True):
    file_list = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset

# Shortcuts for entry point
def get_train_dataset():
    return load_dataset(TRAIN_PATTERN, BATCH_SIZE, is_training=True)

def get_test_dataset():
    return load_dataset(TEST_PATTERN, BATCH_SIZE, is_training=False)
