import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes):
    """Convolutional neural network: 3 conv blocks -> dense head."""

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        
        layers.Rescaling(1.0 / 255),

        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),       
        layers.RandomZoom(0.2),           
        layers.RandomTranslation(0.1, 0.1), 
        layers.RandomContrast(0.3),       

       
        layers.Conv2D(16, 3, padding="same", activation="relu", 
                      kernel_regularizer=keras.regularizers.l2(1e-3)),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(1e-3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(1e-3)),
        layers.MaxPooling2D(),

        # Pooling keeps the dense head small
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Reduced dense units from 128 to 64
        layers.Dense(64, activation="relu", 
                     kernel_regularizer=keras.regularizers.l2(1e-3)),
        layers.Dropout(0.5),

        # 1 sigmoid output for 2 classes, softmax otherwise
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",    
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=50, batch_size=32): # Smaller batch size for small datasets
    """Build, train and save the model. Returns (model, history)."""

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        # Strict patience: Stop immediately when validation loss climbs
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        # Removed learning rate decay to prevent micro-overfitting
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "cnn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'cnn_model.keras')}")

    return model, history

def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)