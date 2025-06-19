import argparse
import time
import os
import tensorflow as tf
from models import MODEL_REGISTRY

from datapipeline import get_train_dataset, get_test_dataset, NUM_EPOCHS

OUTPUT_EVAL_FILE = "evaluation.csv"

def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print("Using TPU")
        return tf.distribute.TPUStrategy(tpu), True
    except:
        print("Using CPU/GPU")
        return tf.distribute.MirroredStrategy(), False

def reset_tpu():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.tpu.experimental.shutdown_tpu_system()
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU reset completed")
    except Exception as e:
        print(f"No TPU to reset or reset failed: {e}")

def train_model(model_name, use_tpu, model_kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    if use_tpu:
        reset_tpu()
    strategy, _ = get_strategy()

    with strategy.scope():
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(**model_kwargs)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    print(f"Loading dataset for {model_name}...")
    train_dataset = load_dataset(TFRECORD_PATH, BATCH_SIZE, is_training=True)
    val_dataset = load_dataset(TFRECORD_PATH, BATCH_SIZE, is_training=False)

    print(f"Training model {model_name}...")
    start = time.time()
    model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=100)
    elapsed = time.time() - start

    print(f"Evaluating model {model_name}...")
    results = model.evaluate(val_dataset, steps=10)
    acc = results[1]

    print(f"{model_name}: accuracy={acc:.4f}, time={elapsed:.2f}s")

    with open(OUTPUT_EVAL_FILE, "a") as f:
        f.write(f"{model_name}\t{acc:.4f}\t{elapsed:.2f}\n")

    tf.keras.backend.clear_session()

def parse_model_kwargs(args):
    # Only include keys that aren't None
    return {k: v for k, v in {
        'num_classes': args.num_classes,
        'alpha': args.alpha,
        'tau': args.tau
    }.items() if v is not None}

def main(args):
    os.makedirs(os.path.dirname(OUTPUT_EVAL_FILE) or ".", exist_ok=True)
    _, is_tpu = get_strategy()  # for checking

    model_kwargs = parse_model_kwargs(args)

    for model_name in args.models:
        train_model(model_name, use_tpu=is_tpu, model_kwargs=model_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to train.")
    parser.add_argument("--num_classes", type=int, help="Number of output classes.")
    parser.add_argument("--alpha", type=int, help="Temporal stride between fast/slow.")
    parser.add_argument("--tau", type=int, help="Number of frames for fast path.")
    args = parser.parse_args()
    main(args)
