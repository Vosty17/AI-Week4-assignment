import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
#import joblib

def load_and_preprocess_images(folder_path, label, target_size=(128, 128)):
    """
    Load images from folder, convert to grayscale, preprocess, and return flattened arrays with labels
    """
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            
            # Read image (always convert to grayscale for training)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {filename}")
                continue
            
            try:
                # Resize and normalize pixel values to [0,1]
                img = cv2.resize(img, target_size) / 255.0
                images.append(img.flatten())
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return np.array(images), np.array(labels)

def preprocess_for_prediction(image_path, target_size=(128, 128)):
    """
    Preprocess image for prediction (handles both color and grayscale)
    Returns flattened array and original image
    """
    # Read image (try as color first)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        # If color read fails, try grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store original for display
    original_img = img.copy()
    
    # Resize and normalize
    img = cv2.resize(img, target_size) / 255.0
    
    return img.flatten(), original_img

def main():
    # Configuration
    TARGET_SIZE = (128, 128)
    
    # Define paths
    BASE_DIR = "complete_set"
    TRAIN_DIR = os.path.join(BASE_DIR, "training_set")
    TEST_DIR = os.path.join(BASE_DIR, "testing_set")
    
    BENIGN_DIR = os.path.join(TRAIN_DIR, "benign")
    MALIGNANT_DIR = os.path.join(TRAIN_DIR, "malignant")
    
    # Verify directory structure exists
    required_dirs = [BASE_DIR, TRAIN_DIR, TEST_DIR, BENIGN_DIR, MALIGNANT_DIR]
    for directory in required_dirs:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
    
    print("Loading and preprocessing training data (grayscale)...")
    
    # Load images (always grayscale for training)
    benign_images, benign_labels = load_and_preprocess_images(BENIGN_DIR, 0, TARGET_SIZE)
    malignant_images, malignant_labels = load_and_preprocess_images(MALIGNANT_DIR, 1, TARGET_SIZE)
    
    print(f"Loaded {len(benign_images)} benign and {len(malignant_images)} malignant images")
    
    # Combine datasets
    X = np.vstack((benign_images, malignant_images))
    y = np.concatenate((benign_labels, malignant_labels))
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    print("\nValidation Results:")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.2%}")
    print(classification_report(y_val, val_preds, target_names=['benign', 'malignant']))
    
    # Save model
    #model_path = "breast_cancer_detector_grayscale.pkl"
    #joblib.dump(model, model_path)
    #print(f"\nModel saved to {model_path}")
    
    # Process test images (can handle both color and grayscale)
    print("\nProcessing test images...")
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(TEST_DIR, filename)
            
            try:
                # Preprocess image (automatically handles color/grayscale)
                features, original_img = preprocess_for_prediction(img_path, TARGET_SIZE)
                
                # Make prediction
                features = features.reshape(1, -1)
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                
                # Display results
                print(f"\nTest Image: {filename}")
                print(f"Original size: {original_img.shape}")
                print(f"Processed size: {int(np.sqrt(len(features[0])))}x{int(np.sqrt(len(features[0])))} grayscale")
                print(f"Prediction: {'MALIGNANT' if pred else 'BENIGN'}")
                print(f"Confidence: {proba[pred]:.2%}")
                print(f"Probabilities: [Benign: {proba[0]:.2%}, Malignant: {proba[1]:.2%}]")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
