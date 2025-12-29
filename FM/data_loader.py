import os
import functools
from typing import Literal, Tuple, Optional

import numpy as np
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(rng, config):
    """Build train and validation iterators for the configured dataset."""
    _validate_batch_sizes(config)
    _validate_binarization_support(config)

    input_dtype = tf.float32
    local_train_bs = config.train.batch_size // jax.process_count()
    local_eval_bs = config.evaluation.eval_batch_size // jax.process_count()
    shuffle_seed = None
    if config.data.shuffle:
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        shuffle_seed = int(jax.random.randint(rng, (), 0, 1_000_000))

    train_ds, valid_ds = _load_raw_datasets(config, input_dtype)

    train_iter = _build_iterator(
        train_ds,
        local_bs=local_train_bs,
        config=config,
        shuffle_seed=shuffle_seed,
        is_train=True,
    )
    valid_iter = (
        _build_iterator(
            valid_ds,
            local_bs=local_eval_bs,
            config=config,
            shuffle_seed=None,
            is_train=False,
        )
        if valid_ds is not None
        else None
    )

    return train_iter, valid_iter


def _validate_batch_sizes(config):
    process_count = jax.process_count()
    if config.train.batch_size % process_count != 0:
        raise ValueError("Train batch size must be divisible by the number of devices")
    if config.evaluation.eval_batch_size % process_count != 0:
        raise ValueError(
            "Eval batch size must be divisible by the number of devices"
        )


def _validate_binarization_support(config):
    if config.data.binarized and config.data.name not in ["mnist", "omniglot"]:
        raise ValueError(
            "Binarized datasets are only supported for MNIST and Omniglot."
        )


def _load_raw_datasets(
    config, input_dtype
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    """Return unsharded train/valid tf.data.Datasets."""
    if config.data.data_dir:
        return _load_image_datasets(
            data_dir=config.data.data_dir,
            input_dtype=input_dtype,
            config=config,
        )
    return _load_tfds_datasets(
        dataset_name=config.data.name,
        input_dtype=input_dtype,
        config=config,
    )


def _load_tfds_datasets(
    dataset_name: Literal["flowers", "cifar10", "mnist", "omniglot", "celeba64"],
    input_dtype=tf.float32,
    config=None,
):
    """Load train/valid splits from TFDS."""
    tfds_name, (train_split, valid_split) = _resolve_tfds_spec(dataset_name, config)
    builder = tfds.builder(tfds_name)
    builder.download_and_prepare()

    if valid_split not in builder.info.splits:
        valid_split = None

    preprocess = functools.partial(
        _preprocess_tfds_example, config=config, input_dtype=input_dtype
    )

    train_ds = builder.as_dataset(split=train_split).enumerate()
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    valid_ds = (
        builder.as_dataset(split=valid_split).enumerate() if valid_split else None
    )
    if valid_ds is not None:
        valid_ds = valid_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, valid_ds


def _resolve_tfds_spec(dataset_name, config):
    """Return the TFDS dataset name and (train_split, valid_split)."""
    train_split, valid_split = "train", "test"

    if dataset_name == "flowers":
        tfds_name = "oxford_flowers102"
        valid_split = "validation"
    elif (
        config.data.binarized
        and not config.data.dyna_binarized
        and dataset_name == "mnist"
    ):
        tfds_name = "binarized_mnist"
    else:
        tfds_name = dataset_name

    # Omniglot keeps the same name but uses test split for validation.
    if dataset_name == "omniglot" and config.data.binarized:
        valid_split = "test"

    return tfds_name, (train_split, valid_split)


def _load_image_datasets(
    data_dir,
    train_pattern="train/all/*.png",
    valid_pattern="valid/all/*.png",
    input_dtype=tf.float32,
    config=None,
):
    """Load datasets stored on disk as image files."""
    train_files = tf.io.gfile.glob(os.path.join(data_dir, train_pattern))
    valid_files = tf.io.gfile.glob(os.path.join(data_dir, valid_pattern))

    preprocess = functools.partial(
        _load_and_preprocess_image, config=config, input_dtype=input_dtype
    )

    train_ds = tf.data.Dataset.from_tensor_slices(train_files).enumerate()
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    valid_ds = (
        tf.data.Dataset.from_tensor_slices(valid_files).enumerate()
        if valid_files
        else None
    )
    if valid_ds is not None:
        valid_ds = valid_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, valid_ds


def _preprocess_tfds_example(idx, data, config, input_dtype):
    """Preprocess a TFDS example."""
    img = data["image"]
    label = data.get("label", -1)

    if config.data.binarized:
        img = tf.image.convert_image_dtype(img, input_dtype)
        if config.data.dyna_binarized:
            img = tf.cast(
                tf.random.uniform(shape=tf.shape(img)) < img, dtype=input_dtype
            )
    else:
        img = tf.image.convert_image_dtype(img, input_dtype)
        if config.data.augment:
            img = tf.image.flip_left_right(img)

    return {"image": img, "label": label, "idx": idx}


def _load_and_preprocess_image(idx, file_path, config, input_dtype):
    """Load and preprocess an image from disk; labels are not provided."""
    assert config.data.name not in ["mnist", "omniglot"]
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = _crop_resize(img, config.data.image_size)
    if config.data.augment:
        img = tf.image.flip_left_right(img)
    img = tf.image.convert_image_dtype(img, input_dtype)
    return {"image": img, "idx": idx}


def _crop_resize(img, image_size):
    return tf.image.resize(img, [image_size, image_size], method="bilinear")


def _build_iterator(ds, local_bs, config, shuffle_seed=None, is_train=True):
    """Shard, shuffle, batch, and prepare for device placement."""
    ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())
    if is_train and config.data.shuffle:
        ds = ds.shuffle(buffer_size=16 * local_bs, seed=shuffle_seed)

    ds = ds.cache()
    if is_train:
        ds = ds.repeat()
    ds = ds.batch(local_bs, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    def gen():
        for batch in ds:
            yield _prepare_for_device(batch, config)

    return gen()


def _prepare_for_device(batch, config):
    """
    Convert batch from TF Tensors to NumPy, normalize, and reshape for pmap.
    Only for the images but not alter the labels or indices.
    """
    local_device_count = jax.local_device_count()

    def _normalize_and_reshape(x):
        x = x._numpy()
        should_normalize = (
            not config.data.binarized
            or getattr(config.model, "decoder", None) == "gaussian"
        )
        if should_normalize:
            x = _normalize_to_neg_one_to_one(x)
        # (bs, h, w, c) -> (local_devices, device_bs, h, w, c)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree.map(_normalize_and_reshape, batch)


def _normalize_to_neg_one_to_one(x):
    """Normalize floats to [-1, 1]; leave integer types unchanged."""
    if np.issubdtype(x.dtype, np.integer):
        return x
    else:
        return (x - 0.5) * 2.0
