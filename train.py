import argparse
import time
import os
from datetime import datetime
import tensorflow as tf
from models import MODEL_REGISTRY

from datapipeline import get_train_dataset, get_test_dataset, NUM_EPOCHS

OUTPUT_EVAL_FILE = "evaluation.csv"

def log(msg, sep=False):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if sep:
        print("=" * 60)
    print(f"{timestamp} {msg}")
    if sep:
        print("=" * 60)

def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver('local')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        log("Using TPU")
        return tf.distribute.TPUStrategy(tpu), True
    except:
        log("Using CPU/GPU")
        return tf.distribute.MirroredStrategy(), False

def reset_tpu():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.tpu.experimental.shutdown_tpu_system()
        tf.tpu.experimental.initialize_tpu_system(resolver)
        log("TPU reset completed")
    except Exception as e:
        log(f"No TPU to reset or reset failed: {e}")

def print_model_config(name, kwargs):
    log(f"Initializing model: {name}")
    for key, val in kwargs.items():
        log(f"  {key}: {val}")
    print()

def train_model(model_name, use_tpu, model_kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    if use_tpu:
        reset_tpu()

    strategy, _ = get_strategy()

    with strategy.scope():
        model_class = MODEL_REGISTRY[model_name]
        print_model_config(model_name, model_kwargs)
        model = model_class(**model_kwargs)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    log(f"Preparing dataset for {model_name}...", sep=True)
    train_dataset = get_train_dataset(folder=args.tfrecord_dir)
    val_dataset = get_test_dataset(folder=args.tfrecord_dir)

    log(f"Training model: {model_name}")
    start_time = time.time()
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=100)
    elapsed = time.time() - start_time

    log(f"Training completed in {elapsed:.2f} seconds")

    log("Running evaluation...")
    results = model.evaluate(val_dataset, steps=10)
    acc = results[1]
    log(f"Evaluation complete: accuracy = {acc:.4f}")

    log(f"Logging results to {OUTPUT_EVAL_FILE}")
    with open(OUTPUT_EVAL_FILE, "a") as f:
        f.write(f"{model_name}\t{acc:.4f}\t{elapsed:.2f}\n")

    tf.keras.backend.clear_session()
    log(f"Finished training for model: {model_name}", sep=True)

def parse_model_kwargs(args):
    return {k: v for k, v in {
        'num_classes': args.num_classes,
        'alpha': args.alpha,
        'tau': args.tau
    }.items() if v is not None}

def main(args):
    os.makedirs(os.path.dirname(OUTPUT_EVAL_FILE) or ".", exist_ok=True)
    _, is_tpu = get_strategy()  # just to check what's available

    model_kwargs = parse_model_kwargs(args)

    for model_name in args.models:
        try:
            train_model(model_name, use_tpu=is_tpu, model_kwargs=model_kwargs)
        except Exception as e:
            log(f"Exception occurred during training model {model_name}: {e}", sep=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to train.")
    parser.add_argument("--num_classes", type=int, help="Number of output classes.")
    parser.add_argument("--alpha", type=int, help="Temporal stride between fast/slow.")
    parser.add_argument("--tau", type=int, help="Number of frames for fast path.")
    parser.add_argument("--tfrecord_dir", type=str,
        default="gs://deepfake-detection", help="Path to folder containing TFRecords")

    args = parser.parse_args()
    main(args)
