import os
# Note: In newer TensorFlow versions, use: tensorflow.keras.utils.image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_path, img_size=128, batch_size=32):
    """
    Creates a streaming data generator that loads images from disk on-the-fly.
    This uses almost zero RAM and won't bloat your C or D drive.
    """
    # 1. Initialize the generator configurations (rescales pixel values 0-255 to 0-1)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # 2. Set up the training data stream
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    # 3. Set up the validation data stream
    val_generator = datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, val_generator