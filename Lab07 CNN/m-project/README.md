## Lab07 CNN

CNN image classifier for recognizing **Cheetahs vs Hyenas** using TensorFlow/Keras.

## Structure

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
