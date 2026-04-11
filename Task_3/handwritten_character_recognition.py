# Handwritten Character Recognition
# Objective: Identify handwritten characters or alphabets.
# Approach: Image processing + deep learning using CNN
# Model: Convolutional Neural Networks (CNN), extendable to CRNN
# Datasets: MNIST (digits), EMNIST (characters)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist

from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Options: 'mnist' | 'emnist_letters' | 'emnist_balanced'
DATASET      = 'mnist'
IMG_SIZE     = 28          # 28x28 pixels
BATCH_SIZE   = 64
EPOCHS       = 10
NUM_CLASSES  = 10          # 10 for MNIST digits; 26 for EMNIST letters


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────

def load_dataset(dataset=DATASET):
    """
    Load MNIST or EMNIST dataset.
    - MNIST        : built into Keras (digits 0-9)
    - EMNIST       : install via `pip install emnist`
                     or download from https://www.nist.gov/itl/products-and-services/emnist-dataset
    """
    if dataset == 'mnist':
        print("Loading MNIST dataset (digits 0–9)...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        class_names = [str(i) for i in range(10)]
        num_classes = 10

    elif dataset in ('emnist_letters', 'emnist_balanced'):
        print(f"Loading {dataset} dataset...")
        try:
            from emnist import extract_training_samples, extract_test_samples
            split = 'letters' if dataset == 'emnist_letters' else 'balanced'
            X_train, y_train = extract_training_samples(split)
            X_test,  y_test  = extract_test_samples(split)

            if split == 'letters':
                # EMNIST letters: labels 1-26 → shift to 0-25
                y_train -= 1
                y_test  -= 1
                class_names = [chr(i + ord('A')) for i in range(26)]
                num_classes = 26
            else:
                num_classes = len(np.unique(y_train))
                class_names = [str(i) for i in range(num_classes)]

        except ImportError:
            print("emnist package not found. Falling back to MNIST.")
            print("Install with: pip install emnist")
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            class_names = [str(i) for i in range(10)]
            num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Image size       : {X_train.shape[1:]}  ")
    print(f"Classes          : {num_classes}\n")

    return X_train, X_test, y_train, y_test, class_names, num_classes


# ─────────────────────────────────────────────
# 2. Explore & Visualize Data
# ─────────────────────────────────────────────

def explore_data(X_train, y_train, class_names):
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i], cmap='gray')
        ax.set_title(class_names[y_train[i]], fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample images saved as 'sample_images.png'")

    # Class distribution
    plt.figure(figsize=(12, 4))
    unique, counts = np.unique(y_train, return_counts=True)
    plt.bar([class_names[u] for u in unique], counts,
            color='#378ADD', alpha=0.85, edgecolor='white')
    plt.title('Class Distribution (Training Set)', fontsize=13)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────

def preprocess(X_train, X_test, y_train, y_test, num_classes):
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    # Reshape to (samples, height, width, channels)
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test  = X_test.reshape(-1,  IMG_SIZE, IMG_SIZE, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train_cat.shape}\n")

    return X_train, X_test, y_train_cat, y_test_cat


# ─────────────────────────────────────────────
# 4. Data Augmentation
# ─────────────────────────────────────────────

def get_data_augmentation():
    """
    Augment training images to improve generalization.
    Handwriting has slight rotations, shifts, and zoom variations.
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )


# ─────────────────────────────────────────────
# 5. Model Architectures
# ─────────────────────────────────────────────

def build_simple_cnn(input_shape, num_classes):
    """
    Simple CNN — fast training, good baseline.
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name='Simple_CNN')
    return model


def build_deep_cnn(input_shape, num_classes):
    """
    Deeper CNN — higher accuracy for complex character sets.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name='Deep_CNN')
    return model


def build_crnn(input_shape, num_classes):
    """
    CNN + RNN (CRNN) — best for word/sentence-level recognition.
    Treats rows of the image as a time sequence fed into LSTM.
    """
    from tensorflow.keras.layers import Reshape, LSTM, Bidirectional

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Collapse width into channels → treat rows as time steps
        Reshape((-1, 64)),

        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name='CRNN_Model')
    return model


# ─────────────────────────────────────────────
# 6. Train Model
# ─────────────────────────────────────────────

def train_model(model, X_train, X_test, y_train, y_test,
                model_name, use_augmentation=True, epochs=EPOCHS):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1, min_lr=1e-6),
        ModelCheckpoint(f'{model_name}_best.keras',
                        monitor='val_accuracy', save_best_only=True, verbose=0)
    ]

    print(f"\nTraining {model_name}...")

    if use_augmentation:
        datagen = get_data_augmentation()
        datagen.fit(X_train)
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{model_name} — Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

    return history, acc


# ─────────────────────────────────────────────
# 7. Visualize Results
# ─────────────────────────────────────────────

def plot_training_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train', color='#378ADD', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='#1D9E75', linewidth=2, linestyle='--')
    axes[0].set_title(f'{model_name} — Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='#D85A30', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val',   color='#BA7517', linewidth=2, linestyle='--')
    axes[1].set_title(f'{model_name} — Loss', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(model, X_test, y_test_cat, class_names, model_name, top_n=10):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    # Show only top_n classes if there are many
    selected = list(range(min(top_n, len(class_names))))
    mask     = np.isin(y_true, selected)
    y_true_s = y_true[mask]
    y_pred_s = y_pred[mask]

    cm = confusion_matrix(y_true_s, y_pred_s, labels=selected)
    labels = [class_names[i] for i in selected]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nClassification Report — {model_name}")
    print(classification_report(y_true, y_pred, target_names=class_names))


def visualize_predictions(model, X_test, y_test_cat, class_names, n=16):
    """Show sample predictions with correct/incorrect labels."""
    y_pred = np.argmax(model.predict(X_test[:n], verbose=0), axis=1)
    y_true = np.argmax(y_test_cat[:n], axis=1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle('Sample Predictions  (green = correct, red = wrong)', fontsize=13)

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        color = '#1D9E75' if y_pred[i] == y_true[i] else '#D85A30'
        ax.set_title(f'P:{class_names[y_pred[i]]}\nT:{class_names[y_true[i]]}',
                     fontsize=8, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Prediction samples saved as 'predictions.png'")


# ─────────────────────────────────────────────
# 8. Predict on a New Image
# ─────────────────────────────────────────────

def predict_image(model, image_path, class_names):
    """
    Predict the character in a new image file.
    Image should be a grayscale handwritten character.
    """
    from PIL import Image

    img = Image.open(image_path).convert('L')       # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Invert if background is white (MNIST has black background)
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx    = np.argmax(predictions)
    confidence  = predictions[pred_idx]

    print("\n" + "=" * 40)
    print("  Handwriting Prediction")
    print("=" * 40)
    print(f"  Predicted Character : {class_names[pred_idx]}")
    print(f"  Confidence          : {confidence:.1%}")
    print("\n  Top 3 predictions:")
    top3 = np.argsort(predictions)[::-1][:3]
    for idx in top3:
        bar = '█' * int(predictions[idx] * 30)
        print(f"    {class_names[idx]}  {predictions[idx]:.1%}  {bar}")
    print("=" * 40)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_array.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f'Predicted: {class_names[pred_idx]} ({confidence:.1%})', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()

    return class_names[pred_idx], confidence


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("  Handwritten Character Recognition")
    print("=" * 55)

    # 1. Load dataset
    X_train, X_test, y_train, y_test, class_names, num_classes = load_dataset(DATASET)

    # 2. Explore
    explore_data(X_train, y_train, class_names)

    # 3. Preprocess
    X_train_p, X_test_p, y_train_cat, y_test_cat = preprocess(
        X_train, X_test, y_train, y_test, num_classes
    )

    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    # 4. Build models
    simple_cnn = build_simple_cnn(input_shape, num_classes)
    deep_cnn   = build_deep_cnn(input_shape, num_classes)
    crnn       = build_crnn(input_shape, num_classes)

    # 5. Train models
    histories = {}

    hist_simple, acc_simple = train_model(
        simple_cnn, X_train_p, X_test_p, y_train_cat, y_test_cat,
        'Simple_CNN', use_augmentation=True
    )
    histories['Simple CNN'] = (hist_simple, acc_simple)

    hist_deep, acc_deep = train_model(
        deep_cnn, X_train_p, X_test_p, y_train_cat, y_test_cat,
        'Deep_CNN', use_augmentation=True
    )
    histories['Deep CNN'] = (hist_deep, acc_deep)

    hist_crnn, acc_crnn = train_model(
        crnn, X_train_p, X_test_p, y_train_cat, y_test_cat,
        'CRNN', use_augmentation=False
    )
    histories['CRNN'] = (hist_crnn, acc_crnn)

    # 6. Results summary
    print("\n" + "=" * 45)
    print("  Final Model Accuracy Summary")
    print("=" * 45)
    for name, (_, acc) in sorted(histories.items(), key=lambda x: -x[1][1]):
        print(f"  {name:<15} {acc:.4f}")

    # 7. Plots & confusion matrix for best model
    best_name   = max(histories, key=lambda n: histories[n][1])
    best_hist   = histories[best_name][0]
    best_model  = {'Simple CNN': simple_cnn,
                   'Deep CNN':   deep_cnn,
                   'CRNN':       crnn}[best_name]

    plot_training_history(best_hist, best_name)
    plot_confusion_matrix(best_model, X_test_p, y_test_cat, class_names, best_name)
    visualize_predictions(best_model, X_test_p, y_test_cat, class_names)

    # 8. Save best model
    best_model.save('best_character_model.keras')
    print(f"\nBest model ({best_name}) saved as 'best_character_model.keras'")

    # 9. Predict on a new image (if available)
    import os
    sample_img = './my_handwriting.png'
    if os.path.exists(sample_img):
        predict_image(best_model, sample_img, class_names)
    else:
        print("\nTo predict on your own handwriting, save the image as 'my_handwriting.png'")
        print("and call: predict_image(best_model, 'my_handwriting.png', class_names)")

    print("\nAll done!")