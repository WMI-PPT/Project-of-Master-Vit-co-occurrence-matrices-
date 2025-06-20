import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Set output directory
output_dir = r"C:\Users\maine\PycharmProjects\PythonProject4\k-fold"
os.makedirs(output_dir, exist_ok=True)

# Read CSV file
csv_path = r"E:\LNDb\cube_nii\output_32gl\filtered_glcm_index_with_label.csv"  # Please confirm the path is correct
data = pd.read_csv(csv_path)

# Preprocess labels before splitting the dataset
data['nodule_id'] = data['id']  # Use id as unique identifier
nodule_stats = data.groupby('nodule_id')['label'].agg(['count', 'sum'])
nodule_stats['dominant_label'] = np.where(
    nodule_stats['sum'] / nodule_stats['count'] >= 0.5, 1, 0
)

# Perform 5-fold cross-validation using StratifiedKFold
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_no = 1
all_train_accuracies = []
all_test_accuracies = []
nodule_train_reports = []
nodule_test_reports = []

for trainval_idx, test_idx in skf.split(nodule_stats.index, nodule_stats['dominant_label']):
    print(f"\n=== Fold {fold_no} ===")

    # Get training+validation set and test set
    trainval_ids = nodule_stats.index[trainval_idx]
    test_ids = nodule_stats.index[test_idx]

    trainval_data = data[data['nodule_id'].isin(trainval_ids)]
    test_data = data[data['nodule_id'].isin(test_ids)]

    # Function to convert 3D to 2D
    def slice_3d_to_2d(data_df):
        images = []
        labels = []
        ids = []
        for _, row in data_df.iterrows():
            label = row['label']
            glcm = np.load(row['glcm_path'])
            slices = np.moveaxis(glcm, 0, 0)
            for slice_ in slices:
                images.append(slice_)
                labels.append(label)
                ids.append(row['nodule_id'])
        return np.array(images), np.array(labels), np.array(ids)

    # Split dataset
    X_train_val, y_train_val, trainval_ids_slices = slice_3d_to_2d(trainval_data)
    X_test, y_test_raw, test_ids_slices = slice_3d_to_2d(test_data)

    X_train_val = X_train_val[..., np.newaxis]  # Add channel dimension
    X_test = X_test[..., np.newaxis]
    y_train_val = to_categorical(y_train_val)
    y_test = to_categorical(y_test_raw)

    # Further split into training and validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(sss.split(X_train_val, np.argmax(y_train_val, axis=1)))
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # Define model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=64,
                        verbose=1)
    model.save(os.path.join(output_dir, f"fold_{fold_no}_model.h5"))

    # Predict and calculate metrics
    train_preds = np.argmax(model.predict(X_train), axis=1)
    train_labels_decoded = np.argmax(y_train, axis=1)
    test_preds = np.argmax(model.predict(X_test), axis=1)
    test_labels_decoded = np.argmax(y_test, axis=1)

    train_accuracy = accuracy_score(train_labels_decoded, train_preds)
    test_accuracy = accuracy_score(test_labels_decoded, test_preds)

    train_report = classification_report(train_labels_decoded, train_preds)
    test_report = classification_report(test_labels_decoded, test_preds)

    # New: Save and print slice-level reports
    with open(os.path.join(output_dir, f"fold_{fold_no}_slice_train_report.txt"), "w") as f:
        f.write(train_report)
    with open(os.path.join(output_dir, f"fold_{fold_no}_slice_test_report.txt"), "w") as f:
        f.write(test_report)
    print(f"Slice-Level Training Report:\n{train_report}")
    print(f"Slice-Level Test Report:\n{test_report}")

    # Nodule-level prediction
    def aggregate_nodule_predictions(ids, slice_preds, true_labels):
        nodule_results = []
        grouped = pd.DataFrame({'nodule_id': ids, 'slice_preds': slice_preds}).groupby('nodule_id')
        for nodule_id, group in grouped:
            benign_slices = (group['slice_preds'] == 0).sum()
            malignant_slices = (group['slice_preds'] == 1).sum()
            predicted_label = 1 if malignant_slices > benign_slices else 0
            true_label = true_labels[group.index[0]]
            is_correct = (predicted_label == true_label)
            nodule_results.append([
                nodule_id, len(group), benign_slices, malignant_slices,
                true_label, predicted_label, is_correct
            ])
        return nodule_results

    train_nodule_results = aggregate_nodule_predictions(trainval_ids_slices[train_idx], train_preds, train_labels_decoded)
    test_nodule_results = aggregate_nodule_predictions(test_ids_slices, test_preds, test_labels_decoded)

    # Save nodule-level results
    nodule_train_df = pd.DataFrame(train_nodule_results, columns=[
        'nodule_id', 'num_slices', 'Benign(0) slice_predictions', 'Malignant(1) slice_predictions',
        'true_label', 'predicted_label', 'is_correct'
    ])
    nodule_test_df = pd.DataFrame(test_nodule_results, columns=[
        'nodule_id', 'num_slices', 'Benign(0) slice_predictions', 'Malignant(1) slice_predictions',
        'true_label', 'predicted_label', 'is_correct'
    ])

    nodule_train_df.to_csv(os.path.join(output_dir, f"fold_{fold_no}_nodule_train_results.csv"), index=False)
    nodule_test_df.to_csv(os.path.join(output_dir, f"fold_{fold_no}_nodule_test_results.csv"), index=False)

    # Generate nodule-level reports
    train_nodule_report = classification_report(nodule_train_df['true_label'], nodule_train_df['predicted_label'])
    test_nodule_report = classification_report(nodule_test_df['true_label'], nodule_test_df['predicted_label'])

    with open(os.path.join(output_dir, f"fold_{fold_no}_nodule_train_report.txt"), "w") as f:
        f.write(train_nodule_report)

    with open(os.path.join(output_dir, f"fold_{fold_no}_nodule_test_report.txt"), "w") as f:
        f.write(test_nodule_report)

    print(f"Nodule-Level Training Report:\n{train_nodule_report}")
    print(f"Nodule-Level Test Report:\n{test_nodule_report}")

    nodule_train_reports.append(train_nodule_report)
    nodule_test_reports.append(test_nodule_report)

    fold_no += 1
