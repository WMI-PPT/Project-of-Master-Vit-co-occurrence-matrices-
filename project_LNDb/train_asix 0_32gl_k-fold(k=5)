import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Read the CSV file
csv_path = r"E:\LNDb\cube_nii\output_32gl\glcm_index_with_label.csv"  # Ensure the path is correct
data = pd.read_csv(csv_path)

# Preprocess data: Load 3D co-occurrence matrix and slice into 2D images
def load_and_process_data(data):
    images = []
    labels = []
    for _, row in data.iterrows():
        label = row["label"]
        glcm_path = row["glcm_path"]
        glcm = np.load(glcm_path)  # Load 3D GLCM with shape (13, 32, 32)
        for i in range(glcm.shape[0]):  # Slice into 32x32 2D images along the first dimension
            images.append(glcm[i])
            labels.append(label)
    return np.array(images), np.array(labels)

images, labels_raw = load_and_process_data(data)
images = images[..., np.newaxis]  # Add channel dimension, resulting in shape (N, 32, 32, 1)
labels_cat = to_categorical(labels_raw)  # One-hot encode labels

# Perform 5-fold cross-validation using StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1
all_train_accuracies = []
all_test_accuracies = []

for train_val_idx, test_idx in skf.split(images, labels_raw):
    print(f"\n=== Fold {fold_no} ===")

    # Split into training+validation set and test set
    X_train_val, X_test = images[train_val_idx], images[test_idx]
    y_train_val, y_test = labels_cat[train_val_idx], labels_cat[test_idx]
    y_train_val_raw = labels_raw[train_val_idx]  # For stratified validation split

    # Use StratifiedShuffleSplit for 75% training, 25% validation within the training+validation set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(sss.split(X_train_val, y_train_val_raw))
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # Define CNN model with input size 32x32x1
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    model = Sequential([
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(labels_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=64,
                        verbose=2)

    # Predict on test and training sets
    test_preds = np.argmax(model.predict(X_test), axis=1)
    test_labels_decoded = np.argmax(y_test, axis=1)

    train_preds = np.argmax(model.predict(X_train), axis=1)
    train_labels_decoded = np.argmax(y_train, axis=1)

    # Generate classification report and accuracy
    test_report = classification_report(test_labels_decoded, test_preds,
                                        target_names=[f'Class {i}' for i in range(labels_cat.shape[1])])
    test_accuracy = accuracy_score(test_labels_decoded, test_preds)

    train_report = classification_report(train_labels_decoded, train_preds,
                                         target_names=[f'Class {i}' for i in range(labels_cat.shape[1])])
    train_accuracy = accuracy_score(train_labels_decoded, train_preds)

    # Record accuracies
    all_train_accuracies.append(train_accuracy)
    all_test_accuracies.append(test_accuracy)

    # Print reports
    print(f"Training Accuracy: {train_accuracy}")
    print("Training Classification Report:\n", train_report)

    print(f"Test Accuracy: {test_accuracy}")
    print("Test Classification Report:\n", test_report)

    # Save reports
    with open(f"training_report_fold_{fold_no}.txt", "w") as train_file:
        train_file.write(f"Training Accuracy: {train_accuracy}\n")
        train_file.write("Training Classification Report:\n")
        train_file.write(train_report)

    with open(f"testing_report_fold_{fold_no}.txt", "w") as test_file:
        test_file.write(f"Test Accuracy: {test_accuracy}\n")
        test_file.write("Test Classification Report:\n")
        test_file.write(test_report)

    fold_no += 1

# Output average accuracy across all folds
print("\n=== Summary ===")
print(f"Average Training Accuracy: {np.mean(all_train_accuracies):.4f}")
print(f"Average Test Accuracy: {np.mean(all_test_accuracies):.4f}")
