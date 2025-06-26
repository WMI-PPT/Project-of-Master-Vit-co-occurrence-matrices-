import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 设置输出目录
output_dir = r"/k-fold( creat vit)"
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件
csv_path = r"E:\LNDb\cube_nii\output_32gl\filtered_glcm_index_with_label.csv"
data = pd.read_csv(csv_path)

# 划分数据集前，预处理标签
data['nodule_id'] = data['id']  # 使用 id 作为唯一标识
nodule_stats = data.groupby('nodule_id')['label'].agg(['count', 'sum'])
nodule_stats['dominant_label'] = np.where(
    nodule_stats['sum'] / nodule_stats['count'] >= 0.5, 1, 0
)

# 用StratifiedGroupKFold做5折交叉验证
k_folds = 5
sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_no = 1
all_train_accuracies = []
all_test_accuracies = []

# 新增：用于存储所有fold的指标
slice_train_metrics = []
slice_test_metrics = []
nodule_train_metrics = []
nodule_test_metrics = []

# 新增：用于存储ROC AUC数据
slice_train_roc_data = []
slice_test_roc_data = []
nodule_train_roc_data = []
nodule_test_roc_data = []

for trainval_idx, test_idx in sgkf.split(nodule_stats.index, nodule_stats['dominant_label'], groups=nodule_stats.index):
    # 为当前fold创建子目录
    fold_dir = os.path.join(output_dir, f"fold_{fold_no}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n=== Fold {fold_no} ===")
    print(f"当前fold输出目录: {fold_dir}")

    # 获取训练+验证集和测试集
    trainval_ids = nodule_stats.index[trainval_idx]
    test_ids = nodule_stats.index[test_idx]

    trainval_data = data[data['nodule_id'].isin(trainval_ids)]
    test_data = data[data['nodule_id'].isin(test_ids)]

    # 统计训练+验证集结节级别样本数和0、1数量
    trainval_labels = nodule_stats.loc[trainval_ids, 'dominant_label']
    print(
        f"Train+Val nodules: total={len(trainval_ids)}, benign={np.sum(trainval_labels == 0)}, malignant={np.sum(trainval_labels == 1)}")

    # 统计测试集结节级别样本数和0、1数量
    test_labels = nodule_stats.loc[test_ids, 'dominant_label']
    print(
        f"Test nodules: total={len(test_ids)}, benign={np.sum(test_labels == 0)}, malignant={np.sum(test_labels == 1)}")


    # 3D转2D函数
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


    # 切割数据集
    X_train_val, y_train_val, trainval_ids_slices = slice_3d_to_2d(trainval_data)
    X_test, y_test_raw, test_ids_slices = slice_3d_to_2d(test_data)

    X_train_val = X_train_val[..., np.newaxis]  # 添加通道维度
    X_test = X_test[..., np.newaxis]
    y_train_val = to_categorical(y_train_val)
    y_test = to_categorical(y_test_raw)

    # 再次划分训练集和验证集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(sss.split(X_train_val, np.argmax(y_train_val, axis=1)))
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # 打印切片级别训练集和验证集样本数量及类别分布
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    print(
        f"Slice-Level Train samples: total={len(y_train_labels)}, benign={(y_train_labels == 0).sum()}, malignant={(y_train_labels == 1).sum()}")
    print(
        f"Slice-Level Val samples: total={len(y_val_labels)}, benign={(y_val_labels == 0).sum()}, malignant={(y_val_labels == 1).sum()}")

    # 定义模型
    # 替换原来的CNN模型定义部分（从model = Sequential()开始）
    # 定义轻量级ViT模型
    def create_vit_model(input_shape=(32, 32, 1), patch_size=4, num_classes=2):
        inputs = tf.keras.layers.Input(shape=input_shape)

        # 1. 创建patches
        patches = tf.keras.layers.Conv2D(
            filters=64,  # 较小的嵌入维度以保持轻量
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding='valid',
            name='patch_embedding'
        )(inputs)

        # 获取patch数量
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        patch_dims = patches.shape[-1]

        # 展平patches
        patches = tf.keras.layers.Reshape((num_patches, patch_dims))(patches)

        # 2. 添加可学习的位置嵌入
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches,
            output_dim=patch_dims,
            name='position_embedding'
        )(positions)

        # 3. 合并patch和位置嵌入
        x = patches + position_embedding

        # 4. 添加Transformer编码器层 (使用少量层保持轻量)
        for _ in range(4):  # 仅使用4层Transformer
            # Layer normalization 1
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # 多头注意力 (使用较少的头数)
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=4,  # 减少头数
                key_dim=patch_dims // 4,  # 减小key维度
                dropout=0.1
            )(x1, x1)

            # Skip connection 1
            x2 = tf.keras.layers.Add()([attention_output, x])

            # Layer normalization 2
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

            # MLP
            x3 = tf.keras.layers.Dense(patch_dims * 2, activation='gelu')(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            x3 = tf.keras.layers.Dense(patch_dims, activation='gelu')(x3)

            # Skip connection 2
            x = tf.keras.layers.Add()([x3, x2])

        # 5. 全局平均池化
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # 6. 分类头
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        # 创建模型
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


    # 创建模型
    model = create_vit_model(input_shape=(32, 32, 1), patch_size=4, num_classes=y_train.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=64,
                        verbose=1)

    # 保存模型到当前fold目录
    model.save(os.path.join(fold_dir, "model.h5"))

    # 预测并计算指标
    train_preds = np.argmax(model.predict(X_train), axis=1)
    train_labels_decoded = np.argmax(y_train, axis=1)
    test_preds = np.argmax(model.predict(X_test), axis=1)
    test_labels_decoded = np.argmax(y_test, axis=1)

    train_accuracy = accuracy_score(train_labels_decoded, train_preds)
    test_accuracy = accuracy_score(test_labels_decoded, test_preds)

    train_report = classification_report(train_labels_decoded, train_preds)
    test_report = classification_report(test_labels_decoded, test_preds)


    # 计算切片级别的详细指标
    def calculate_metrics(y_true, y_pred, y_prob=None):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        accuracy = accuracy_score(y_true, y_pred)  # 新增计算accuracy

        metrics = {
            'precision_benign': precision[0],
            'precision_malignant': precision[1],
            'recall_benign': recall[0],
            'recall_malignant': recall[1],
            'f1_benign': f1[0],
            'f1_malignant': f1[1],
            'specificity_benign': specificity,
            'specificity_malignant': recall[1],  # Sensitivity = Recall for class 1
            'accuracy': accuracy  # 新增accuracy
        }

        if y_prob is not None:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            metrics['roc_auc'] = roc_auc
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        return metrics


    # 获取预测概率用于ROC曲线
    train_probs = model.predict(X_train)
    test_probs = model.predict(X_test)

    # 切片级别指标
    slice_train_metrics_fold = calculate_metrics(train_labels_decoded, train_preds, train_probs)
    slice_test_metrics_fold = calculate_metrics(test_labels_decoded, test_preds, test_probs)

    # 保存ROC数据
    slice_train_roc_data.append({
        'fpr': slice_train_metrics_fold['fpr'],
        'tpr': slice_train_metrics_fold['tpr'],
        'auc': slice_train_metrics_fold['roc_auc'],
        'fold': fold_no
    })
    slice_test_roc_data.append({
        'fpr': slice_test_metrics_fold['fpr'],
        'tpr': slice_test_metrics_fold['tpr'],
        'auc': slice_test_metrics_fold['roc_auc'],
        'fold': fold_no
    })

    # 保存指标
    slice_train_metrics.append(slice_train_metrics_fold)
    slice_test_metrics.append(slice_test_metrics_fold)

    # 保存并打印切片级别报告和详细指标（保存到当前fold目录）
    with open(os.path.join(fold_dir, "slice_train_report.txt"), "w") as f:
        f.write(train_report)
        f.write("\n\nDetailed Metrics:\n")
        for k, v in slice_train_metrics_fold.items():
            if k not in ['fpr', 'tpr']:
                f.write(f"{k}: {v:.4f}\n")

    with open(os.path.join(fold_dir, "slice_test_report.txt"), "w") as f:
        f.write(test_report)
        f.write("\n\nDetailed Metrics:\n")
        for k, v in slice_test_metrics_fold.items():
            if k not in ['fpr', 'tpr']:
                f.write(f"{k}: {v:.4f}\n")

    print(f"Slice-Level Training Report:\n{train_report}")
    print(f"Slice-Level Test Report:\n{test_report}")


    # 绘制切片级别混淆矩阵（保存到当前fold目录）
    def plot_confusion_matrix(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(fold_dir, filename))
        plt.close()


    plot_confusion_matrix(train_labels_decoded, train_preds,
                          f"Fold {fold_no} Slice-Level Training Confusion Matrix",
                          "slice_train_cm.png")
    plot_confusion_matrix(test_labels_decoded, test_preds,
                          f"Fold {fold_no} Slice-Level Test Confusion Matrix",
                          "slice_test_cm.png")


    # 结节级别预测
    def aggregate_nodule_predictions(ids, slice_preds, true_labels, slice_probs=None):
        nodule_results = []
        grouped = pd.DataFrame({'nodule_id': ids, 'slice_preds': slice_preds,
                                'true_label': true_labels})
        if slice_probs is not None:
            grouped['slice_probs'] = slice_probs[:, 1]

        nodule_grouped = grouped.groupby('nodule_id')
        for nodule_id, group in nodule_grouped:
            benign_slices = (group['slice_preds'] == 0).sum()
            malignant_slices = (group['slice_preds'] == 1).sum()
            predicted_label = 1 if malignant_slices > benign_slices else 0
            true_label = group['true_label'].iloc[0]  # All slices have same true label

            if slice_probs is not None:
                avg_prob = group['slice_probs'].mean()
            else:
                avg_prob = None

            is_correct = (predicted_label == true_label)
            nodule_results.append([
                nodule_id, len(group), benign_slices, malignant_slices,
                true_label, predicted_label, is_correct, avg_prob
            ])
        return nodule_results


    train_nodule_results = aggregate_nodule_predictions(
        trainval_ids_slices[train_idx], train_preds, train_labels_decoded, train_probs)
    test_nodule_results = aggregate_nodule_predictions(
        test_ids_slices, test_preds, test_labels_decoded, test_probs)

    # 保存nodule级别结果到当前fold目录
    nodule_train_df = pd.DataFrame(train_nodule_results, columns=[
        'nodule_id', 'num_slices', 'Benign(0) slice_predictions', 'Malignant(1) slice_predictions',
        'true_label', 'predicted_label', 'is_correct', 'avg_prob'
    ])
    nodule_test_df = pd.DataFrame(test_nodule_results, columns=[
        'nodule_id', 'num_slices', 'Benign(0) slice_predictions', 'Malignant(1) slice_predictions',
        'true_label', 'predicted_label', 'is_correct', 'avg_prob'
    ])

    nodule_train_df.to_csv(os.path.join(fold_dir, "nodule_train_results.csv"), index=False)
    nodule_test_df.to_csv(os.path.join(fold_dir, "nodule_test_results.csv"), index=False)

    # 生成nodule级别报告
    train_nodule_report = classification_report(nodule_train_df['true_label'], nodule_train_df['predicted_label'])
    test_nodule_report = classification_report(nodule_test_df['true_label'], nodule_test_df['predicted_label'])

    # 结节级别指标
    nodule_train_metrics_fold = calculate_metrics(
        nodule_train_df['true_label'], nodule_train_df['predicted_label'],
        np.column_stack(
            [1 - nodule_train_df['avg_prob'], nodule_train_df['avg_prob']]) if 'avg_prob' in nodule_train_df else None)
    nodule_test_metrics_fold = calculate_metrics(
        nodule_test_df['true_label'], nodule_test_df['predicted_label'],
        np.column_stack(
            [1 - nodule_test_df['avg_prob'], nodule_test_df['avg_prob']]) if 'avg_prob' in nodule_test_df else None)

    # 保存ROC数据
    if 'roc_auc' in nodule_train_metrics_fold:
        nodule_train_roc_data.append({
            'fpr': nodule_train_metrics_fold['fpr'],
            'tpr': nodule_train_metrics_fold['tpr'],
            'auc': nodule_train_metrics_fold['roc_auc'],
            'fold': fold_no
        })
    if 'roc_auc' in nodule_test_metrics_fold:
        nodule_test_roc_data.append({
            'fpr': nodule_test_metrics_fold['fpr'],
            'tpr': nodule_test_metrics_fold['tpr'],
            'auc': nodule_test_metrics_fold['roc_auc'],
            'fold': fold_no
        })

    # 保存指标
    nodule_train_metrics.append(nodule_train_metrics_fold)
    nodule_test_metrics.append(nodule_test_metrics_fold)

    # 保存nodule级别报告到当前fold目录
    with open(os.path.join(fold_dir, "nodule_train_report.txt"), "w") as f:
        f.write(train_nodule_report)
        f.write("\n\nDetailed Metrics:\n")
        for k, v in nodule_train_metrics_fold.items():
            if k not in ['fpr', 'tpr']:
                f.write(f"{k}: {v:.4f}\n")

    with open(os.path.join(fold_dir, "nodule_test_report.txt"), "w") as f:
        f.write(test_nodule_report)
        f.write("\n\nDetailed Metrics:\n")
        for k, v in nodule_test_metrics_fold.items():
            if k not in ['fpr', 'tpr']:
                f.write(f"{k}: {v:.4f}\n")

    print(f"Nodule-Level Training Report:\n{train_nodule_report}")
    print(f"Nodule-Level Test Report:\n{test_nodule_report}")

    # 绘制结节级别混淆矩阵到当前fold目录
    plot_confusion_matrix(nodule_train_df['true_label'], nodule_train_df['predicted_label'],
                          f"Fold {fold_no} Nodule-Level Training Confusion Matrix",
                          "nodule_train_cm.png")
    plot_confusion_matrix(nodule_test_df['true_label'], nodule_test_df['predicted_label'],
                          f"Fold {fold_no} Nodule-Level Test Confusion Matrix",
                          "nodule_test_cm.png")

    fold_no += 1


# 计算所有fold的平均指标和置信区间（保存到根目录）
# [Previous code remains exactly the same until the calculate_overall_metrics function]

# 计算所有fold的平均指标和置信区间（保存到根目录）
def calculate_overall_metrics(metrics_list, name):
    # 提取所有fold的指标
    metrics_df = pd.DataFrame(metrics_list)

    # 1. 保存每个fold的详细指标
    fold_metrics = []
    for i, fold_metrics_dict in enumerate(metrics_list, 1):
        fold_data = {'fold': i}
        for k, v in fold_metrics_dict.items():
            if k not in ['fpr', 'tpr']:
                fold_data[k] = v
        fold_metrics.append(fold_data)

    pd.DataFrame(fold_metrics).to_csv(os.path.join(output_dir, f"per_fold_{name}_metrics.csv"), index=False)

    # 2. 创建符合要求的10个metrics表格格式
    table_data = {
        'Metric': [],
        'Fold 1': [], 'Fold 2': [], 'Fold 3': [], 'Fold 4': [], 'Fold 5': [],
        'Average ± sd': []
    }

    # 定义要显示的10个指标及其显示名称
    metric_mapping = {
        'precision_benign': 'Precision (0)',
        'precision_malignant': 'Precision (1)',
        'recall_benign': 'Recall (0)',
        'recall_malignant': 'Recall (1)',
        'f1_benign': 'F1 (0)',
        'f1_malignant': 'F1 (1)',
        'specificity_benign': 'Specificity (0)',
        'specificity_malignant': 'Specificity (1)',
        'roc_auc': 'ROC AUC',
        'accuracy': 'Accuracy'
    }

    # 填充表格数据
    for metric_key, display_name in metric_mapping.items():
        if metric_key in metrics_df.columns:
            table_data['Metric'].append(display_name)
            for fold in range(1, 6):
                val = metrics_df.iloc[fold - 1][metric_key]
                table_data[f'Fold {fold}'].append(f"{val:.2f}" if not pd.isna(val) else "N/A")

            mean = metrics_df[metric_key].mean()
            std = metrics_df[metric_key].std()
            table_data['Average ± sd'].append(f"{mean:.2f} \u00B1 {std:.2f}")  # \u00B1 是 ± 的 Unicode 编码

    # 创建DataFrame并保存
    summary_table = pd.DataFrame(table_data)

    # 按特定顺序排列指标（与示例保持一致）
    preferred_order = [
        'Precision (0)', 'Precision (1)',
        'Recall (0)', 'Recall (1)',
        'F1 (0)', 'F1 (1)',
        'Specificity (0)', 'Specificity (1)',
        'ROC AUC', 'Accuracy'
    ]
    summary_table = summary_table.set_index('Metric').loc[preferred_order].reset_index()

    summary_table.to_csv(
        os.path.join(output_dir, f"table_format_{name}_summary.csv"),
        index=False,
        encoding='utf-8-sig'  # 确保编码正确
    )

    # 同时保存原始统计量（保持原有功能）
    summary = metrics_df.describe().loc[['mean', 'std']].T
    for col in metrics_df.columns:
        if col not in ['fpr', 'tpr']:
            mean = metrics_df[col].mean()
            std = metrics_df[col].std()
            ci = stats.t.interval(0.95, len(metrics_df[col]) - 1,
                                  loc=mean, scale=std / np.sqrt(len(metrics_df[col])))
            summary.loc[col, 'ci_low'] = ci[0]
            summary.loc[col, 'ci_high'] = ci[1]

    summary.to_csv(os.path.join(output_dir, f"overall_{name}_metrics_summary.csv"))

    return summary_table



# [Rest of the code remains exactly the same]


# 计算所有级别的总体指标
slice_train_summary = calculate_overall_metrics(slice_train_metrics, "slice_train")
slice_test_summary = calculate_overall_metrics(slice_test_metrics, "slice_test")
nodule_train_summary = calculate_overall_metrics(nodule_train_metrics, "nodule_train")
nodule_test_summary = calculate_overall_metrics(nodule_test_metrics, "nodule_test")


# 绘制ROC曲线（保存到根目录）
def plot_roc_curves(roc_data, title, filename):
    plt.figure(figsize=(8, 6))

    # 绘制每个fold的ROC曲线
    for data in roc_data:
        plt.plot(data['fpr'], data['tpr'],
                 label=f"Fold {data['fold']} (AUC = {data['auc']:.2f})",
                 alpha=0.3)

    # 计算平均ROC曲线
    all_fpr = np.unique(np.concatenate([data['fpr'] for data in roc_data]))
    mean_tpr = np.zeros_like(all_fpr)
    for data in roc_data:
        mean_tpr += np.interp(all_fpr, data['fpr'], data['tpr'])
    mean_tpr /= len(roc_data)
    mean_auc = np.mean([data['auc'] for data in roc_data])

    plt.plot(all_fpr, mean_tpr, 'k-',
             label=f'Mean ROC (AUC = {mean_auc:.2f})',
             linewidth=2)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# 绘制所有级别的ROC曲线到根目录
if slice_train_roc_data:
    plot_roc_curves(slice_train_roc_data,
                    'Slice-Level Training ROC Curves',
                    'slice_train_roc_curves.png')
if slice_test_roc_data:
    plot_roc_curves(slice_test_roc_data,
                    'Slice-Level Test ROC Curves',
                    'slice_test_roc_curves.png')
if nodule_train_roc_data:
    plot_roc_curves(nodule_train_roc_data,
                    'Nodule-Level Training ROC Curves',
                    'nodule_train_roc_curves.png')
if nodule_test_roc_data:
    plot_roc_curves(nodule_test_roc_data,
                    'Nodule-Level Test ROC Curves',
                    'nodule_test_roc_curves.png')
