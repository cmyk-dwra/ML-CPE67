## Lab08 DCNN

Deep CNN image classifier for recognizing Cheetahs vs Hyenas using a regularized VGG-style architecture in TensorFlow/Keras.

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
├── vgg_model.py
├── data_loader.py
├── preprocessing.py
├── split_data.py
├── evaluate.py
└── test_vgg.py
```
## Model

- 4 convolutional blocks: (32, 2) → (64, 2) → (128, 3) → (256, 3)
- Spatial dropout (0.15) after max pooling for feature-map regularization
- Max pooling 
- Data augmentation 
- Global average pooling 
- Dropout 
- Dense sigmoid classification layer with zero-bias initialization
- AdamW optimizer 
- Binary cross-entropy loss 

## Training

- **Training data:** Used to update network weights via mini-batch gradient descent.
- **Validation data:** Evaluated every epoch to monitor generalization and guide learning rate scheduling.

## Output

- `vgg_model.keras` - trained model 
- `training_history.png` - Plots of training vs. validation accuracy and loss
- `confusion_matrix.png` - Final test set classification results
- `.npy` files - processed datasets and labels 

## Run

    python main.py
