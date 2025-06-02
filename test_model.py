import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def debug_existing_model(model_path= './best_predator_model_v2.h5'):
    """Debug an existing trained model"""
    
    print("=== DEBUGGING EXISTING MODEL ===")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Model architecture analysis
    print("\n1. MODEL ARCHITECTURE:")
    model.summary()
    
    # Check final layer
    final_layer = model.layers[-1]
    print(f"\nFinal layer: {final_layer.name}")
    print(f"Units: {final_layer.units}")
    
    # Check activation more robustly
    try:
        activation_name = final_layer.activation.__name__
    except:
        activation_name = str(final_layer.activation)
    
    print(f"Activation: {activation_name}")
    
    if final_layer.units == 1 and ('sigmoid' in activation_name.lower()):
        print("✓ Binary classification setup looks correct")
    else:
        print("⚠️  Final layer configuration may be incorrect for binary classification")
        if 'sigmoid' not in activation_name.lower():
            print(f"   Expected 'sigmoid' activation, got '{activation_name}'")
        if final_layer.units != 1:
            print(f"   Expected 1 output unit, got {final_layer.units}")
    
    return model

def test_model_with_synthetic_data(model):
    """Test model with controlled synthetic data"""
    
    print("\n2. SYNTHETIC DATA TESTS:")
    
    # Create test images with known patterns
    img_size = 32  # Adjust based on your model
    
    # Test case 1: All black image (should be non-predator)
    black_img = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
    pred_black = model.predict(black_img, verbose=0)[0][0]
    print(f"Black image prediction: {pred_black:.4f} ({'Predator' if pred_black > 0.5 else 'Non-predator'})")
    
    # Test case 2: All white image (should be non-predator) 
    white_img = np.ones((1, img_size, img_size, 3), dtype=np.float32)
    pred_white = model.predict(white_img, verbose=0)[0][0]
    print(f"White image prediction: {pred_white:.4f} ({'Predator' if pred_white > 0.5 else 'Non-predator'})")
    
    # Test case 3: Random noise
    random_imgs = np.random.random((10, img_size, img_size, 3)).astype(np.float32)
    pred_random = model.predict(random_imgs, verbose=0).flatten()
    print(f"Random images predictions: min={pred_random.min():.4f}, max={pred_random.max():.4f}, mean={pred_random.mean():.4f}")
    print(f"Random predictions: {['P' if p > 0.5 else 'N' for p in pred_random]}")
    
    # Check if model always predicts same class
    all_preds = np.concatenate([[pred_black, pred_white], pred_random])
    unique_classes = len(np.unique((all_preds > 0.5).astype(int)))
    
    if unique_classes == 1:
        print("⚠️  WARNING: Model always predicts the same class!")
        print("   This indicates severe overfitting or training data bias")
        return False
    else:
        print("✓ Model shows variation in predictions")
        return True

def analyze_training_data_distribution():
    """Analyze the distribution of training data"""
    
    print("\n3. TRAINING DATA ANALYSIS:")
    
    # Load CIFAR-100 to check predator class distribution
    (x_train, y_train), _ = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    
    predator_classes = [70, 82]  # Adjust these IDs
    predator_mask = np.isin(y_train.flatten(), predator_classes)
    
    total_samples = len(y_train)
    predator_count = np.sum(predator_mask)
    predator_ratio = predator_count / total_samples
    
    print(f"CIFAR-100 training data:")
    print(f"  Total samples: {total_samples}")
    print(f"  Predator samples: {predator_count} ({predator_ratio*100:.2f}%)")
    print(f"  Non-predator samples: {total_samples - predator_count} ({(1-predator_ratio)*100:.2f}%)")
    
    if predator_ratio < 0.05:
        print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print("   With <5% predators, model will likely always predict non-predator")
        print("   Solution: Balance your dataset or use class weights")
    
    return predator_ratio

def recommend_fixes(model_varies, predator_ratio):
    """Provide recommendations based on analysis"""
    
    print("\n4. RECOMMENDATIONS:")
    
    if not model_varies:
        print("🔧 CRITICAL: Model always predicts same class")
        print("   - Check your training data labels")
        print("   - Balance your dataset (undersample/oversample)")
        print("   - Use class weights in model.fit()")
        print("   - Reduce model complexity to prevent overfitting")
    
    if predator_ratio < 0.1:
        print("🔧 Severe class imbalance detected")
        print("   - Use balanced datasets")
        print("   - Apply class weights:")
        print(f"     class_weight={{0: 1.0, 1: {(1-predator_ratio)/predator_ratio:.1f}}}")
        print("   - Use stratified sampling")
        print("   - Consider SMOTE for oversampling")
    
    print("\n🔧 General improvements:")
    print("   - Add validation metrics: precision, recall, F1-score")
    print("   - Use early stopping with validation loss")
    print("   - Implement proper train/val/test splits")
    print("   - Add data augmentation")
    print("   - Monitor training curves for overfitting")

def create_balanced_training_example():
    """Show how to create balanced training data"""
    
    print("\n5. BALANCED TRAINING EXAMPLE:")
    
    code_example = '''
# Example of balanced training with class weights
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train with class weights
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weight_dict,  # This balances the classes
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3)
    ]
)

# Alternative: Manual balancing
def balance_dataset(X, y):
    from collections import Counter
    counter = Counter(y)
    min_count = min(counter.values())
    
    balanced_X, balanced_y = [], []
    for class_label in counter.keys():
        class_indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(class_indices, min_count, replace=False)
        balanced_X.extend(X[selected_indices])
        balanced_y.extend(y[selected_indices])
    
    return np.array(balanced_X), np.array(balanced_y)
'''
    
    print(code_example)

def main_debug():
    """Main debugging function"""
    # Load your model
    model = tf.keras.models.load_model('./best_predator_model_v2.h5')

    # Create some test images
    test_images = np.random.random((5, 64, 64, 3)).astype(np.float32)

    # Make predictions
    predictions = model.predict(test_images)
    print("Raw predictions:", predictions.flatten())
    print("Binary predictions:", (predictions > 0.5).astype(int).flatten())
        
    # 1. Debug existing model architecture
    model = debug_existing_model()
    
    if model is None:
        print("Cannot proceed without a valid model")
        return
    
    # 2. Test model behavior
    model_varies = test_model_with_synthetic_data(model)
    
    # 3. Check training data distribution  
    predator_ratio = analyze_training_data_distribution()
    
    # 4. Provide recommendations
    recommend_fixes(model_varies, predator_ratio)
    
    # 5. Show balanced training example
    create_balanced_training_example()
    
    print("\n" + "="*50)
    print("DEBUGGING COMPLETE")
    print("Run the fixed training script to resolve issues")
    print("="*50)

if __name__ == "__main__":
    main_debug()
    
