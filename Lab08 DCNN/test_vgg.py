import json
import os

from tensorflow import keras
from tensorflow.keras import layers

# VGG layouts
VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
VGG_SMALL = [(32, 2), (64, 2), (128, 3), (256, 3)]
VGG_TINY = [(32, 2), (64, 2), (128, 2)]


def vgg_block(model, filters, n_conv, drop_rate=0.15):
    """Conv2D + ReLU without BatchNorm (prevents validation variance spikes), followed by MaxPool + SpatialDropout."""
    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
    
    model.add(layers.MaxPooling2D(2))
    if drop_rate > 0:
        model.add(layers.SpatialDropout2D(drop_rate))


def _compile(model, num_classes, lr=1e-4):
    """AdamW with stable learning rate and weight decay to keep train & val curves aligned."""
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _head(x, num_classes):
    """Shared classifier head used by the pretrained model (Functional API)."""
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.4)(x)
    return layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax",
        bias_initializer="zeros",
    )(x)


def build_model(input_shape, num_classes, blocks=VGG_SMALL):
    """VGG-style network trained from scratch."""
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.RandomFlip("horizontal"))

    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv, drop_rate=0.15)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax",
        bias_initializer="zeros",
    ))

    _compile(model, num_classes, lr=2e-4)
    return model


def build_pretrained(input_shape, num_classes):
    """Real VGG16 with ImageNet weights, frozen backbone, plus a regularized head."""
    base = keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = base(x)
    outputs = _head(x, num_classes)

    model = keras.Model(inputs, outputs)
    _compile(model, num_classes, lr=1e-3)
    return model


def _callbacks(patience=6):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


class _History:
    """Minimal stand-in so plot_history(history, ...) keeps working."""
    def __init__(self, history):
        self.history = history


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=50, batch_size=32, blocks=VGG_SMALL,
                pretrained=False):

    if pretrained:
        model = build_pretrained(X_train.shape[1:], num_classes)
        patience = 5
    else:
        model = build_model(X_train.shape[1:], num_classes, blocks)
        patience = 7
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_callbacks(patience),
        verbose=1,
    )
    combined = {k: list(v) for k, v in history.history.items()}

    if pretrained:
        print("\nFine-tuning last VGG block...")
        base = model.get_layer("vgg16")
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")
        _compile(model, num_classes, lr=1e-5)

        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=batch_size,
            callbacks=_callbacks(4),
            verbose=1,
        )
        for k, v in history2.history.items():
            combined.setdefault(k, []).extend(v)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in combined.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, _History(combined)


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)
    return probabilities.argmax(axis=1)