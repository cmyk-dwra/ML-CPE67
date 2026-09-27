## Lab07 CNN

CNN image classifier for recognizing **Cheetahs vs Hyenas** using TensorFlow/Keras.

## Dataset Structure

[DOWNLOAD DATASET HERE](https://www.kaggle.com/datasets/singhdatascientist/for-image-classification-of-cheetah-vs-hyena)

```text
train/
├── cheetah/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
└── hyena/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

## Structure

```text
m-project/
├── train/
│   ├── cheetah/
│   └── hyena/
├── validation/
│   ├── cheetah/
│   └── hyena/
├── main.py
├── cnn_model.py
├── data_loader.py
├── preprocessing.py
├── split_data.py
├── evaluate.py
└── test_cnn.py
```
## Model

- 3 convolutional blocks: `32 → 64 → 128` filters
- Batch normalization
- Max pooling
- Data augmentation
- Global average pooling
- Dropout
- Dense classification layer
- Adam optimizer
- Binary cross-entropy loss

## Training

- **Training data:** used to update model weights.
- **Validation data:** unseen during training and used to measure generalization.

## Output

- `cnn_model.keras` - trained model
- `history.json` - training history
- `confusion_matrix.png` - classification results
- `.npy` files - processed datasets and labels

## Run

    python main.py

To test the trained model:

    python test_cnn.py
