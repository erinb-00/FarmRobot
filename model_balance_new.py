import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import os
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def unpickle(file):
    """Unpickle CIFAR data files"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_rosbot_images(folder='rosbot_images', target_size=(32, 32)):
    """Load ROSbot images from folder as RGB array"""
    image_list = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    files.sort()
    if len(files) == 0:
        print(f"No images found in {folder} (extensions: {valid_extensions})")
        return np.array([])
    for filename in files:
        path = os.path.join(folder, filename)
        img = Image.open(path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        image_list.append(img_array)
    print(f"Loaded {len(image_list)} ROSbot images from '{folder}'")
    return np.array(image_list, dtype=np.uint8)

def augment_rosbot_images(images, augment_count=200, batch_size=16):
    """Generate augmented ROSbot images to increase training samples"""

    if images.size == 0:
        return np.array([])

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    image_array_norm = images.astype(np.float32) / 255.0

    aug_iter = datagen.flow(image_array_norm, batch_size=batch_size, shuffle=True)

    augmented_images = []
    total_generated = 0

    while total_generated < augment_count:
        batch = next(aug_iter)
        augmented_images.extend(batch)
        total_generated += len(batch)

    augmented_images = np.array(augmented_images[:augment_count]) * 255.0
    augmented_images = augmented_images.astype(np.uint8)

    print(f"Generated {len(augmented_images)} augmented ROSbot images")
    return augmented_images

def load_and_balance_data():
    """Load CIFAR-100 data and create balanced predator/non-predator dataset"""
    # Load CIFAR-100 data
    metadata_path = './archive-2/meta'
    metadata = unpickle(metadata_path)
    # Get both fine and coarse label names
    fine_label_names = [name.decode('utf-8') for name in metadata[b'fine_label_names']]
    coarse_label_names = [name.decode('utf-8') for name in metadata[b'coarse_label_names']]
    data_pre_path = './archive-2/'
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    # Read data
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)
    # Get data (using fine_labels for more specific classification)
    data_train = data_train_dict[b'data']
    label_train = np.array(data_train_dict[b'fine_labels'])
    data_test = data_test_dict[b'data']
    label_test = np.array(data_test_dict[b'fine_labels'])
    # Reshape data
    x_train = data_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = data_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # Define predator classes based on fine labels
    predator_classes = []
    predator_keywords = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'mouse', 'shrew']
    print("Available fine-grained classes:")
    for i, name in enumerate(fine_label_names):
        count_train = np.sum(label_train == i)
        if count_train > 0:
            print(f"  {i}: {name} ({count_train} training samples)")
            if any(keyword in name.lower() for keyword in predator_keywords):
                predator_classes.append(i)
                print(f"    -> Classified as PREDATOR")
    print(f"\nIdentified predator classes: {predator_classes}")
    print(f"Predator class names: {[fine_label_names[i] for i in predator_classes]}")
    # Create binary labels
    y_train_binary = np.isin(label_train, predator_classes).astype(int)
    y_test_binary = np.isin(label_test, predator_classes).astype(int)

    print(f"\nClass distribution:")
    print(f"  Training - Predators: {np.sum(y_train_binary)} ({np.mean(y_train_binary)*100:.2f}%)")
    print(f"  Training - Non-predators: {len(y_train_binary) - np.sum(y_train_binary)} ({(1-np.mean(y_train_binary))*100:.2f}%)")
    print(f"  Test - Predators: {np.sum(y_test_binary)} ({np.mean(y_test_binary)*100:.2f}%)")
    print(f"  Test - Non-predators: {len(y_test_binary) - np.sum(y_test_binary)} ({(1-np.mean(y_test_binary))*100:.2f}%)")

    return x_train, y_train_binary, x_test, y_test_binary, fine_label_names, predator_classes

def balance_dataset(X, y, method='undersample', random_state=42):
    """Balance the dataset using undersampling or oversampling"""
    np.random.seed(random_state)
    counter = Counter(y)
    print(f"Before balancing: {counter}")
    if method == 'undersample':
        min_count = min(counter.values())
        balanced_X, balanced_y = [], []
        for class_label in counter.keys():
            class_indices = np.where(y == class_label)[0]
            if len(class_indices) > min_count:
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
            else:
                selected_indices = class_indices
            balanced_X.extend(X[selected_indices])
            balanced_y.extend(y[selected_indices])
        return np.array(balanced_X), np.array(balanced_y)
    elif method == 'oversample':
        max_count = max(counter.values())
        balanced_X, balanced_y = [], []
        for class_label in counter.keys():
            class_indices = np.where(y == class_label)[0]
            if len(class_indices) < max_count:
                selected_indices = np.random.choice(class_indices, max_count, replace=True)
            else:
                selected_indices = class_indices
            balanced_X.extend(X[selected_indices])
            balanced_y.extend(y[selected_indices])
        return np.array(balanced_X), np.array(balanced_y)
    elif method == 'hybrid':
        counts = list(counter.values())
        target_count = int(np.mean(counts))
        balanced_X, balanced_y = [], []
        for class_label in counter.keys():
            class_indices = np.where(y == class_label)[0]
            current_count = len(class_indices)
            if current_count < target_count:
                selected_indices = np.random.choice(class_indices, target_count, replace=True)
            else:
                selected_indices = np.random.choice(class_indices, min(current_count, target_count), replace=False)
            balanced_X.extend(X[selected_indices])
            balanced_y.extend(y[selected_indices])
        return np.array(balanced_X), np.array(balanced_y)

def create_advanced_model(input_shape=(32, 32, 3), dropout_rate=0.3):
    """Create an advanced CNN model with residual connections"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Initial conv layer
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # First residual block
    residual = x
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    # Second block with more filters
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    residual = x
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    # Third block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dense layers with regularization
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate + 0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """Create data generators with enhanced augmentation"""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    return train_generator, val_generator

def plot_enhanced_results(history, y_test, y_pred, y_pred_prob):
    """Create comprehensive result visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Training history plots
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    # ROC Curve
    fpr, tpr, _= roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axes[0, 2].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 2].set_xlim([0.0, 1.0])
    axes[0, 2].set_ylim([0.0, 1.05])
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Non-Predator', 'Predator'],
                yticklabels=['Non-Predator', 'Predator'])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    # Prediction Distribution
    axes[1, 1].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.7,
                    label='Non-Predator', color='blue', density=True)
    axes[1, 1].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.7,
                    label='Predator', color='red', density=True)
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')
    axes[1, 1].set_xlabel('Prediction Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    # Precision-Recall if available in history
    if 'precision' in history.history:
        axes[1, 2].plot(history.history['precision'], label='Training Precision')
        axes[1, 2].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 2].plot(history.history['recall'], label='Training Recall')
        axes[1, 2].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 2].set_title('Precision & Recall')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Precision/Recall\nNot Available',
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Precision & Recall')
    plt.tight_layout()
    plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    return roc_auc

def train_improved_model():
    """Main training function with improvements including ROSbot data split and augmentation"""

    print("Loading and preparing CIFAR-100 data...")
    X_train, y_train, X_test, y_test, class_names, predator_classes = load_and_balance_data()

    print("\nLoading ROSbot images...")
    rosbot_original_images = load_rosbot_images(folder='rosbot_images', target_size=(32, 32))

    # Split original ROSbot images into train, val, test (~60% train, 20% val, 20% test)
    if rosbot_original_images.size > 0:
        r_y = np.ones(len(rosbot_original_images), dtype=int)  # label = predator (1)

        X_ros_train, X_ros_temp, y_ros_train, y_ros_temp = train_test_split(
            rosbot_original_images, r_y,
            test_size=0.4,
            random_state=42,
            stratify=r_y
        )
        X_ros_val, X_ros_test, y_ros_val, y_ros_test = train_test_split(
            X_ros_temp, y_ros_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_ros_temp
        )

        print(f"ROSbot dataset split: {len(X_ros_train)} train, {len(X_ros_val)} val, {len(X_ros_test)} test images.")

        # Augment train ROSbot images to enrich training data
        augmented_rosbot_train = augment_rosbot_images(X_ros_train, augment_count=200, batch_size=16)

        # Prepare labels
        augmented_rosbot_train_labels = np.ones(len(augmented_rosbot_train), dtype=int)  # predator

        # Combine CIFAR train + augmented ROSbot train
        X_train = np.concatenate((X_train, augmented_rosbot_train), axis=0)
        y_train = np.concatenate((y_train, augmented_rosbot_train_labels), axis=0)

        # Combine CIFAR val + ROSbot val
        X_val = np.concatenate((X_ros_val.astype(np.uint8),), axis=0)
        y_val = np.concatenate((y_ros_val,), axis=0)

        # Combine CIFAR test + ROSbot test
        X_test = np.concatenate((X_test, X_ros_test.astype(np.uint8)), axis=0)
        y_test = np.concatenate((y_test, y_ros_test), axis=0)

        print(f"After appending ROSbot: Train size={len(X_train)}, Val size={len(X_val)}, Test size={len(X_test)}")
    else:
        # If no ROSbot images found, just split CIFAR train into train/val
        print("No ROSbot images found, using CIFAR train/val split only.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            stratify=y_train,
            random_state=42
        )

    # Now balance the training data (only training set)
    print("\nBalancing training data (hybrid method)...")
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method='hybrid')
    counter_balanced = Counter(y_train_balanced)
    print(f"After balancing training data: {counter_balanced}")

    # If ROSbot val not created (no rosbot), split training val
    if 'X_val' not in locals():
        print("Creating validation split from training data")
        X_train_balanced, X_val, y_train_balanced, y_val = train_test_split(
            X_train_balanced, y_train_balanced,
            test_size=0.2,
            stratify=y_train_balanced,
            random_state=42
        )

    # Create data generators
    train_gen, val_gen = create_data_generators(X_train_balanced, y_train_balanced, X_val, y_val)

    # Create advanced model
    print("\nCreating advanced CNN model...")
    model = create_advanced_model()

    # Compile with metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print("\nModel summary:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_predator_model_v2.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Class weights for balanced training
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_balanced),
        y=y_train_balanced
    )
    class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Normalize test images for evaluation
    X_test_normalized = X_test.astype('float32') / 255.0

    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test_normalized, y_test, verbose=0)
    print(f"\nTest Results:")
    for i, metric_name in enumerate(model.metrics_names):
        print(f"  {metric_name.capitalize()}: {test_results[i]:.4f}")

    y_pred_prob = model.predict(X_test_normalized).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Predator', 'Predator']))

    roc_auc = plot_enhanced_results(history, y_test, y_pred, y_pred_prob)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # Save final model
    model.save('predator_classifier_improved.h5')
    print("\nModel saved as 'predator_classifier_improved.h5'")

    return model, history, (X_test_normalized, y_test, y_pred_prob)

def analyze_predictions(model_path='predator_classifier_improved.h5', test_data=None):
    """Analyze model predictions in detail"""
    model = tf.keras.models.load_model(model_path)
    if test_data is not None:
        X_test, y_test, y_pred_prob = test_data
        confident_correct = np.where((np.abs(y_pred_prob - 0.5) > 0.4) &
                                    ((y_pred_prob > 0.5) == y_test.astype(bool)))[0]
        confident_wrong = np.where((np.abs(y_pred_prob - 0.5) > 0.4) &
                                  ((y_pred_prob > 0.5) != y_test.astype(bool)))[0]
        uncertain = np.where(np.abs(y_pred_prob - 0.5) < 0.1)[0]
        print(f"\nPrediction Analysis:")
        print(f"  Confident correct predictions: {len(confident_correct)}")
        print(f"  Confident wrong predictions: {len(confident_wrong)}")
        print(f"  Uncertain predictions (0.4-0.6): {len(uncertain)}")
        if len(confident_wrong) > 0:
            print(f"  Most confident wrong predictions:")
            wrong_indices = confident_wrong[np.argsort(np.abs(y_pred_prob[confident_wrong] - 0.5))[-5:]]
            for idx in wrong_indices:
                print(f"    Index {idx}: True={y_test[idx]}, Pred={y_pred_prob[idx]:.3f}")
    # Synthetic image tests
    print(f"\nSynthetic Image Tests:")
    synthetic_tests = {
        'Black image': np.zeros((1, 32, 32, 3), dtype=np.float32),
        'White image': np.ones((1, 32, 32, 3), dtype=np.float32),
        'Gray image': np.full((1, 32, 32, 3), 0.5, dtype=np.float32),
        'Random noise': np.random.random((1, 32, 32, 3)).astype(np.float32)
    }
    for name, image in synthetic_tests.items():
        pred = model.predict(image, verbose=0)[0][0]
        label = 'Predator' if pred > 0.5 else 'Non-predator'
        confidence = abs(pred - 0.5) * 2
        print(f"  {name}: {pred:.4f} ({label}, confidence: {confidence:.3f})")

if __name__ == "__main__":
    #np.random.seed(42)
    #tf.random.set_seed(42)

    # Train the improved model (with ROSbot splits & augmentation)
    model, history, test_data = train_improved_model()

    # Analyze predictions
    analyze_predictions('best_predator_model_v2.h5', test_data)