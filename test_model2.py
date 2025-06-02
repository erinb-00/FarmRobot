import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import random
import time

# CIFAR-100 Data Loading Functions
def unpickle(file):
    """Unpickle CIFAR data files"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
    label_train = np.array(data_train_dict[b'fine_labels'])  # Changed to fine labels
    data_test = data_test_dict[b'data']
    label_test = np.array(data_test_dict[b'fine_labels'])    # Changed to fine labels
    
    # Reshape data
    x_train = data_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = data_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Define predator classes based on fine labels (more accurate mapping)
    predator_classes = []
    predator_keywords = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'mouse', 'shrew']
    
    print("Available fine-grained classes:")
    for i, name in enumerate(fine_label_names):
        count_train = np.sum(label_train == i)
        if count_train > 0:
            print(f"  {i}: {name} ({count_train} training samples)")
            # Check if this class represents a predator
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

# Model Loading
def load_model(model_path):
    """Load the trained model"""
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Random Image Generation Functions
def generate_random_images(num_images=10, shape=(32, 32, 3)):
    """Generate random test images"""
    # Generate random images with values between 0 and 1
    images = np.random.rand(num_images, *shape).astype(np.float32)
    return images

def generate_normalized_random_images(num_images=10, shape=(32, 32, 3)):
    """Generate random images with proper normalization (0-255 then normalized)"""
    # Generate random integers 0-255, then normalize to 0-1
    images = np.random.randint(0, 256, size=(num_images, *shape), dtype=np.uint8)
    images = images.astype(np.float32) / 255.0
    return images

# Prediction and Analysis Functions
def test_model_predictions(model, test_images, test_name=""):
    """Test model with images and measure performance"""
    print(f"\nTesting model with {len(test_images)} {test_name} images...")
    
    # Time the prediction
    start_time = time.time()
    predictions = model.predict(test_images, verbose=0)
    end_time = time.time()
    
    print(f"Prediction time: {end_time - start_time:.4f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(test_images):.4f} seconds")
    
    return predictions

def analyze_predictions(predictions, class_names=None):
    """Analyze and display prediction statistics"""
    print(f"\nPrediction Analysis:")
    print(f"Predictions shape: {predictions.shape}")
    
    if len(predictions.shape) == 2 and predictions.shape[1] > 1:  # Multi-class classification
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        print(f"Predicted classes: {predicted_classes}")
        print(f"Confidence scores: {confidence_scores}")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        print(f"Min confidence: {np.min(confidence_scores):.4f}")
        print(f"Max confidence: {np.max(confidence_scores):.4f}")
        
        # Class distribution
        unique_classes, counts = np.unique(predicted_classes, return_counts=True)
        print(f"\nClass distribution:")
        for cls, count in zip(unique_classes, counts):
            class_name = class_names[cls] if class_names else f"Class {cls}"
            print(f"  {class_name}: {count} predictions")
            
    elif len(predictions.shape) == 1 or predictions.shape[1] == 1:  # Binary classification
        predictions_flat = predictions.flatten()
        binary_predictions = (predictions_flat > 0.5).astype(int)
        
        print(f"Raw predictions (0-1): {predictions_flat}")
        print(f"Binary predictions: {binary_predictions}")
        print(f"Average prediction: {np.mean(predictions_flat):.4f}")
        print(f"Predictions > 0.5: {np.sum(binary_predictions)} out of {len(binary_predictions)}")

def visualize_test_images(images, predictions, title_prefix="", num_display=6):
    """Visualize test images with their predictions"""
    num_display = min(num_display, len(images))
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(num_display):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        # Format prediction for title
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            pred_class = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            title = f"{title_prefix}Class: {pred_class}\nConf: {confidence:.3f}"
        else:
            pred_value = predictions[i] if len(predictions.shape) == 1 else predictions[i][0]
            title = f"{title_prefix}Pred: {pred_value:.3f}"
            
        axes[i].set_title(title)
    
    # Hide unused subplots
    for i in range(num_display, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# CIFAR-100 Specific Testing Functions
def get_sample_images(X, y, original_labels, fine_label_names, predator_classes, 
                     num_predator=10, num_non_predator=10, random_state=42):
    """Get sample images from both predator and non-predator classes"""
    
    np.random.seed(random_state)
    
    # Get predator samples
    predator_indices = np.where(y == 1)[0]
    if len(predator_indices) >= num_predator:
        selected_predator_indices = np.random.choice(predator_indices, num_predator, replace=False)
    else:
        selected_predator_indices = predator_indices
        print(f"Warning: Only {len(predator_indices)} predator images available")
    
    # Get non-predator samples
    non_predator_indices = np.where(y == 0)[0]
    if len(non_predator_indices) >= num_non_predator:
        selected_non_predator_indices = np.random.choice(non_predator_indices, num_non_predator, replace=False)
    else:
        selected_non_predator_indices = non_predator_indices[:num_non_predator]
    
    # Combine samples
    all_indices = np.concatenate([selected_predator_indices, selected_non_predator_indices])
    sample_images = X[all_indices]
    sample_labels = y[all_indices]
    sample_original_labels = original_labels[all_indices]
    
    # Create detailed information for each sample
    sample_info = []
    for i, (idx, binary_label, orig_label) in enumerate(zip(all_indices, sample_labels, sample_original_labels)):
        class_name = fine_label_names[orig_label]
        is_predator = binary_label == 1
        sample_info.append({
            'index': i,
            'original_index': idx,
            'class_name': class_name,
            'original_label': orig_label,
            'is_predator': is_predator,
            'binary_label': binary_label
        })
    
    return sample_images, sample_labels, sample_info

def test_model_on_samples(model, sample_images, sample_info):
    """Test the model on sample images and analyze results"""
    
    # Normalize images (assuming they're in 0-255 range)
    normalized_images = sample_images.astype(np.float32) / 255.0
    
    # Get predictions
    predictions = model.predict(normalized_images, verbose=0)
    
    # Analyze results
    results = []
    correct_predictions = 0
    
    for i, info in enumerate(sample_info):
        pred_value = predictions[i][0] if len(predictions.shape) > 1 else predictions[i]
        pred_binary = 1 if pred_value > 0.5 else 0
        is_correct = pred_binary == info['binary_label']
        confidence = pred_value if pred_binary == 1 else (1 - pred_value)
        
        if is_correct:
            correct_predictions += 1
        
        result = {
            'index': i,
            'class_name': info['class_name'],
            'true_label': info['binary_label'],
            'predicted_prob': pred_value,
            'predicted_binary': pred_binary,
            'is_correct': is_correct,
            'confidence': confidence
        }
        results.append(result)
    
    accuracy = correct_predictions / len(sample_info)
    
    return results, accuracy, predictions

def visualize_cifar_predictions(sample_images, sample_info, results, num_cols=5):
    """Visualize CIFAR sample images with their predictions"""
    
    num_images = len(sample_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        ax = axes[row, col]
        ax.imshow(sample_images[i])
        ax.axis('off')
        
        result = results[i]
        info = sample_info[i]
        
        # Create title with class name, true label, and prediction
        true_class = "Predator" if info['is_predator'] else "Non-Predator"
        pred_class = "Predator" if result['predicted_binary'] == 1 else "Non-Predator"
        correct_symbol = "✓" if result['is_correct'] else "✗"
        
        title = f"{info['class_name']}\nTrue: {true_class}\nPred: {pred_class} ({result['predicted_prob']:.3f}) {correct_symbol}"
        ax.set_title(title, fontsize=8)
    
    # Hide unused subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_results_by_class(results, sample_info):
    """Analyze results grouped by original class"""
    
    print("\n" + "="*60)
    print("DETAILED RESULTS BY CLASS")
    print("="*60)
    
    # Group by class name
    class_results = {}
    for result, info in zip(results, sample_info):
        class_name = info['class_name']
        if class_name not in class_results:
            class_results[class_name] = []
        class_results[class_name].append((result, info))
    
    # Analyze each class
    for class_name, class_data in class_results.items():
        print(f"\nClass: {class_name}")
        print("-" * 40)
        
        correct_count = sum(1 for result, _ in class_data if result['is_correct'])
        total_count = len(class_data)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        true_label = class_data[0][1]['is_predator']
        true_class = "Predator" if true_label else "Non-Predator"
        
        avg_confidence = np.mean([result['confidence'] for result, _ in class_data])
        avg_prob = np.mean([result['predicted_prob'] for result, _ in class_data])
        
        print(f"  True class: {true_class}")
        print(f"  Samples: {total_count}")
        print(f"  Correct: {correct_count}/{total_count} ({accuracy*100:.1f}%)")
        print(f"  Average prediction probability: {avg_prob:.3f}")
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        # Show individual predictions for this class
        for j, (result, info) in enumerate(class_data):
            pred_class = "Predator" if result['predicted_binary'] == 1 else "Non-Predator"
            status = "✓" if result['is_correct'] else "✗"
            print(f"    Sample {j+1}: {pred_class} ({result['predicted_prob']:.3f}) {status}")

# Main Testing Functions
def run_random_image_tests(model):
    """Run tests with random images"""
    print("\n" + "="*70)
    print("RANDOM IMAGE TESTING")
    print("="*70)
    
    # Test with different types of random images
    print("\n" + "="*50)
    print("TESTING WITH UNIFORM RANDOM IMAGES (0-1)")
    print("="*50)
    
    test_images_1 = generate_random_images(num_images=20)
    predictions_1 = test_model_predictions(model, test_images_1, "uniform random")
    analyze_predictions(predictions_1)
    
    print("\n" + "="*50)
    print("TESTING WITH NORMALIZED RANDOM IMAGES (0-255 -> 0-1)")
    print("="*50)
    
    test_images_2 = generate_normalized_random_images(num_images=20)
    predictions_2 = test_model_predictions(model, test_images_2, "normalized random")
    analyze_predictions(predictions_2)
    
    # Visualize some test images
    print("\nGenerating random image visualization...")
    visualize_test_images(test_images_2[:6], predictions_2[:6], "Random: ")
    
    # Test model robustness with edge cases
    print("\n" + "="*50)
    print("TESTING EDGE CASES")
    print("="*50)
    
    # All zeros
    zeros_image = np.zeros((1, 32, 32, 3), dtype=np.float32)
    zeros_pred = model.predict(zeros_image, verbose=0)
    print(f"All zeros prediction: {zeros_pred[0]}")
    
    # All ones
    ones_image = np.ones((1, 32, 32, 3), dtype=np.float32)
    ones_pred = model.predict(ones_image, verbose=0)
    print(f"All ones prediction: {ones_pred[0]}")
    
    # Test batch prediction performance
    print("\n" + "="*50)
    print("BATCH PERFORMANCE TEST")
    print("="*50)
    
    batch_sizes = [1, 10, 50, 100]
    for batch_size in batch_sizes:
        test_batch = generate_random_images(num_images=batch_size)
        start_time = time.time()
        batch_predictions = model.predict(test_batch, verbose=0)
        end_time = time.time()
        
        total_time = end_time - start_time
        per_image_time = total_time / batch_size
        
        print(f"Batch size {batch_size:3d}: {total_time:.4f}s total, {per_image_time:.6f}s per image")

def run_cifar_tests(model):
    """Run tests with CIFAR-100 data"""
    print("\n" + "="*70)
    print("CIFAR-100 DATA TESTING")
    print("="*70)
    
    print("Loading CIFAR-100 data...")
    try:
        x_train, y_train, x_test, y_test, fine_label_names, predator_classes = load_and_balance_data()
        
        # Get original labels for detailed analysis
        metadata_path = './archive-2/meta'
        metadata = unpickle(metadata_path)
        
        data_pre_path = './archive-2/'
        data_test_path = data_pre_path + 'test'
        data_test_dict = unpickle(data_test_path)
        original_test_labels = np.array(data_test_dict[b'fine_labels'])
        
    except Exception as e:
        print(f"Error loading CIFAR-100 data: {e}")
        print("Make sure the './archive-2/' directory exists with CIFAR-100 data files")
        return False
    
    print("\nGetting sample images...")
    sample_images, sample_labels, sample_info = get_sample_images(
        x_test, y_test, original_test_labels, fine_label_names, predator_classes,
        num_predator=15, num_non_predator=15
    )
    
    print(f"Selected {len(sample_images)} images for testing")
    print(f"Predator samples: {sum(1 for info in sample_info if info['is_predator'])}")
    print(f"Non-predator samples: {sum(1 for info in sample_info if not info['is_predator'])}")
    
    print("\nTesting model on CIFAR-100 sample images...")
    results, accuracy, predictions = test_model_on_samples(model, sample_images, sample_info)
    
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Analyze results by true class
    predator_results = [r for r, info in zip(results, sample_info) if info['is_predator']]
    non_predator_results = [r for r, info in zip(results, sample_info) if not info['is_predator']]
    
    if predator_results:
        predator_accuracy = sum(1 for r in predator_results if r['is_correct']) / len(predator_results)
        print(f"Predator Accuracy: {predator_accuracy*100:.2f}% ({sum(1 for r in predator_results if r['is_correct'])}/{len(predator_results)})")
    
    if non_predator_results:
        non_predator_accuracy = sum(1 for r in non_predator_results if r['is_correct']) / len(non_predator_results)
        print(f"Non-Predator Accuracy: {non_predator_accuracy*100:.2f}% ({sum(1 for r in non_predator_results if r['is_correct'])}/{len(non_predator_results)})")
    
    # Show detailed analysis
    analyze_results_by_class(results, sample_info)
    
    # Visualize results
    print(f"\nGenerating CIFAR-100 visualization...")
    visualize_cifar_predictions(sample_images, sample_info, results)
    
    return True

def run_comprehensive_test():
    """Run comprehensive model testing with both random and CIFAR-100 data"""
    model_path = "best_predator_model_v2.h5"
    
    print("="*70)
    print("COMPREHENSIVE MODEL TESTING SUITE")
    print("="*70)
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Run random image tests
    run_random_image_tests(model)
    
    # Run CIFAR-100 tests
    cifar_success = run_cifar_tests(model)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    if cifar_success:
        print("✓ Random image testing completed")
        print("✓ CIFAR-100 testing completed")
        print("\nBoth test suites ran successfully!")
    else:
        print("✓ Random image testing completed")
        print("✗ CIFAR-100 testing failed (data not available)")
        print("\nRandom image testing completed. CIFAR-100 data not found.")

if __name__ == "__main__":
    # Run the comprehensive test suite
    run_comprehensive_test()
    
    # You can also run individual test components:
    model = load_model("best_predator_model_v2.h5")
    run_random_image_tests(model)
    run_cifar_tests(model)