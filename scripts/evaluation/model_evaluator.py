import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, BatchNormalization

# Define the ResidualBlock class
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, use_activation=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_activation = use_activation
        self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        if self.use_activation:
            x = layers.Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return layers.Add()([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_activation': self.use_activation
        })
        return config

# Nucleotide mapping for one-hot encoding
NUCLEOTIDE_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoding."""
    return np.array([NUCLEOTIDE_MAP[nt] for nt in sequence])

def load_sequences_from_folder(folder_path, label, sequence_length=600):
    """
    Load sequences from a folder with flexible sequence length support.
    
    Args:
        folder_path: Path to folder containing .txt files
        label: Label to assign to all sequences in this folder
        sequence_length: Expected sequence length (default: 600)
    
    Returns:
        tuple: (sequences_array, labels_array)
    """
    data, labels = [], []
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == sequence_length:
                            data.append(one_hot_encode(line))
                            labels.append(label)
    else:
        print(f"Warning: Folder not found - {folder_path}")
    return np.array(data), np.array(labels)

def load_test_data_three_class(base_path, show_progress=False, sequence_length=600):
    """
    Load test data for 3-class system:
    0: Acceptor (ACC/CAN + ACC/NC)
    1: Donor (DON/CAN + DON/NC) 
    2: No Splice Site (NEG/ACC + NEG/DON)
    
    Args:
        base_path: Path to test data directory
        show_progress: Whether to show progress bars
        sequence_length: Expected sequence length
    
    Returns:
        tuple: (X_test, y_test)
    """
    pos_path = os.path.join(base_path, 'POS')
    neg_path = os.path.join(base_path, 'NEG')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    all_data = []
    all_labels = []

    # Load Acceptor sequences (Label 0)
    print("[INFO] Loading Acceptor test sequences...")
    acc_can_data, acc_can_labels = load_sequences_from_folder(
        os.path.join(acc_path, 'CAN'), 0, sequence_length
    )
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(
        os.path.join(acc_path, 'NC'), 0, sequence_length
    )
    
    if len(acc_can_data) > 0:
        all_data.append(acc_can_data)
        all_labels.append(acc_can_labels)
    if len(acc_nc_data) > 0:
        all_data.append(acc_nc_data)
        all_labels.append(acc_nc_labels)

    # Load Donor sequences (Label 1)
    print("[INFO] Loading Donor test sequences...")
    don_can_data, don_can_labels = load_sequences_from_folder(
        os.path.join(don_path, 'CAN'), 1, sequence_length
    )
    don_nc_data, don_nc_labels = load_sequences_from_folder(
        os.path.join(don_path, 'NC'), 1, sequence_length
    )
    
    if len(don_can_data) > 0:
        all_data.append(don_can_data)
        all_labels.append(don_can_labels)
    if len(don_nc_data) > 0:
        all_data.append(don_nc_data)
        all_labels.append(don_nc_labels)

    # Load Negative sequences (Label 2)
    print("[INFO] Loading Negative test sequences...")
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(
        os.path.join(neg_path, 'ACC'), 2, sequence_length
    )
    neg_don_data, neg_don_labels = load_sequences_from_folder(
        os.path.join(neg_path, 'DON'), 2, sequence_length
    )
    
    if len(neg_acc_data) > 0:
        all_data.append(neg_acc_data)
        all_labels.append(neg_acc_labels)
    if len(neg_don_data) > 0:
        all_data.append(neg_don_data)
        all_labels.append(neg_don_labels)

    # Combine all data
    if all_data:
        data = np.concatenate(all_data)
        labels = np.concatenate(all_labels)
    else:
        data = np.array([])
        labels = np.array([])

    print(f"[INFO] Test data loaded - shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Test class distribution: {np.bincount(labels)}")
    print("[INFO] Test class mapping: 0=Acceptor, 1=Donor, 2=No Splice Site")

    return data, labels

def load_test_data_with_canonical_info(base_path, show_progress=False, sequence_length=600):
    """
    Load test data with canonical/non-canonical information for detailed misclassification analysis.
    
    Args:
        base_path: Path to test data directory
        show_progress: Whether to show progress bars
        sequence_length: Expected sequence length
    
    Returns:
        tuple: (X_test, y_test, canonical_info)
        canonical_info: dict with keys 'acceptor_canonical', 'acceptor_noncanonical', 
                      'donor_canonical', 'donor_noncanonical' containing indices
    """
    pos_path = os.path.join(base_path, 'POS')
    neg_path = os.path.join(base_path, 'NEG')
    acc_path = os.path.join(pos_path, 'ACC')
    don_path = os.path.join(pos_path, 'DON')

    all_data = []
    all_labels = []
    canonical_info = {
        'acceptor_canonical': [],
        'acceptor_noncanonical': [],
        'donor_canonical': [],
        'donor_noncanonical': []
    }
    
    current_index = 0

    # Load Acceptor sequences (Label 0)
    print("[INFO] Loading Acceptor test sequences...")
    acc_can_data, acc_can_labels = load_sequences_from_folder(
        os.path.join(acc_path, 'CAN'), 0, sequence_length
    )
    acc_nc_data, acc_nc_labels = load_sequences_from_folder(
        os.path.join(acc_path, 'NC'), 0, sequence_length
    )
    
    if len(acc_can_data) > 0:
        all_data.append(acc_can_data)
        all_labels.append(acc_can_labels)
        canonical_info['acceptor_canonical'] = list(range(current_index, current_index + len(acc_can_data)))
        current_index += len(acc_can_data)
    if len(acc_nc_data) > 0:
        all_data.append(acc_nc_data)
        all_labels.append(acc_nc_labels)
        canonical_info['acceptor_noncanonical'] = list(range(current_index, current_index + len(acc_nc_data)))
        current_index += len(acc_nc_data)

    # Load Donor sequences (Label 1)
    print("[INFO] Loading Donor test sequences...")
    don_can_data, don_can_labels = load_sequences_from_folder(
        os.path.join(don_path, 'CAN'), 1, sequence_length
    )
    don_nc_data, don_nc_labels = load_sequences_from_folder(
        os.path.join(don_path, 'NC'), 1, sequence_length
    )
    
    if len(don_can_data) > 0:
        all_data.append(don_can_data)
        all_labels.append(don_can_labels)
        canonical_info['donor_canonical'] = list(range(current_index, current_index + len(don_can_data)))
        current_index += len(don_can_data)
    if len(don_nc_data) > 0:
        all_data.append(don_nc_data)
        all_labels.append(don_nc_labels)
        canonical_info['donor_noncanonical'] = list(range(current_index, current_index + len(don_nc_data)))
        current_index += len(don_nc_data)

    # Load Negative sequences (Label 2)
    print("[INFO] Loading Negative test sequences...")
    neg_acc_data, neg_acc_labels = load_sequences_from_folder(
        os.path.join(neg_path, 'ACC'), 2, sequence_length
    )
    neg_don_data, neg_don_labels = load_sequences_from_folder(
        os.path.join(neg_path, 'DON'), 2, sequence_length
    )
    
    if len(neg_acc_data) > 0:
        all_data.append(neg_acc_data)
        all_labels.append(neg_acc_labels)
    if len(neg_don_data) > 0:
        all_data.append(neg_don_data)
        all_labels.append(neg_don_labels)

    # Combine all data
    if all_data:
        data = np.concatenate(all_data)
        labels = np.concatenate(all_labels)
    else:
        data = np.array([])
        labels = np.array([])

    print(f"[INFO] Test data loaded - shapes: {data.shape}, {labels.shape}")
    print(f"[INFO] Test class distribution: {np.bincount(labels)}")
    print("[INFO] Test class mapping: 0=Acceptor, 1=Donor, 2=No Splice Site")
    
    # Print canonical info summary
    print(f"[INFO] Acceptor Canonical: {len(canonical_info['acceptor_canonical'])} sequences")
    print(f"[INFO] Acceptor Non-canonical: {len(canonical_info['acceptor_noncanonical'])} sequences")
    print(f"[INFO] Donor Canonical: {len(canonical_info['donor_canonical'])} sequences")
    print(f"[INFO] Donor Non-canonical: {len(canonical_info['donor_noncanonical'])} sequences")

    return data, labels, canonical_info

def evaluate_model_three_class(model_path, test_data, test_labels):
    """
    Evaluate model with 3-class system: 
    0=Acceptor, 1=Donor, 2=No Splice Site
    
    Args:
        model_path: Path to trained model (.h5 file)
        test_data: Test sequences (one-hot encoded)
        test_labels: Test labels (0, 1, 2)
    
    Returns:
        tuple: (accuracy, f1_score, precision, recall, classification_report)
    """
    # Load the model
    model = load_model(model_path, custom_objects={"ResidualBlock": ResidualBlock})
    
    # Debug: Check model output shape
    print(f"Model output shape: {model.output_shape}")
    print(f"Model expects {model.output_shape[-1]} classes")

    # Make predictions
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Debug: Check classes
    print(f"Unique classes in test_labels: {np.unique(test_labels)}")
    print(f"Unique classes in predicted_classes: {np.unique(predicted_classes)}")

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes, average='weighted')
    precision = precision_score(test_labels, predicted_classes, average='weighted')
    recall = recall_score(test_labels, predicted_classes, average='weighted')
    
    # Generate classification report with 4 decimal places
    class_report = classification_report(
        test_labels, predicted_classes,
        labels=[0, 1, 2],  # 3-class labels
        target_names=['Acceptor', 'Donor', 'No Splice Site'],
        zero_division=0,  # Handle missing classes gracefully
        digits=4  # Show 4 decimal places
    )
    
    # Format the accuracy row to show 4 decimal places
    lines = class_report.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('accuracy'):
            # Format accuracy row with 4 decimal places
            parts = line.split()
            if len(parts) >= 4:
                accuracy_val = float(parts[3])
                formatted_line = f"{parts[0]:>20} {parts[1]:>10} {parts[2]:>10} {accuracy_val:.4f} {parts[4]:>10}"
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    formatted_report = '\n'.join(formatted_lines)
    
    return accuracy, f1, precision, recall, formatted_report

def evaluate_model_with_canonical_analysis(model_path, test_data, test_labels, canonical_info):
    """
    Evaluate model with detailed canonical/non-canonical misclassification analysis.
    
    Args:
        model_path: Path to trained model (.h5 file)
        test_data: Test sequences (one-hot encoded)
        test_labels: Test labels (0, 1, 2)
        canonical_info: Dictionary with canonical/non-canonical indices
    
    Returns:
        tuple: (accuracy, f1, precision, recall, classification_report, canonical_analysis)
    """
    # Load the model
    model = load_model(model_path, custom_objects={"ResidualBlock": ResidualBlock})
    
    # Make predictions
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate basic metrics
    accuracy = accuracy_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes, average='weighted')
    precision = precision_score(test_labels, predicted_classes, average='weighted')
    recall = recall_score(test_labels, predicted_classes, average='weighted')
    
    # Generate classification report with 4 decimal places
    class_report = classification_report(
        test_labels, predicted_classes,
        labels=[0, 1, 2],  # 3-class labels
        target_names=['Acceptor', 'Donor', 'No Splice Site'],
        zero_division=0,  # Handle missing classes gracefully
        digits=4  # Show 4 decimal places
    )
    
    # Format the accuracy row to show 4 decimal places
    lines = class_report.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('accuracy'):
            # Format accuracy row with 4 decimal places
            parts = line.split()
            if len(parts) >= 4:
                accuracy_val = float(parts[3])
                formatted_line = f"{parts[0]:>20} {parts[1]:>10} {parts[2]:>10} {accuracy_val:.4f} {parts[4]:>10}"
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    formatted_report = '\n'.join(formatted_lines)
    
    # Calculate canonical/non-canonical misclassification rates
    canonical_analysis = {}
    
    # Acceptor Non-canonical Analysis
    if canonical_info['acceptor_noncanonical']:
        acc_nc_indices = np.array(canonical_info['acceptor_noncanonical'])
        acc_nc_true = test_labels[acc_nc_indices]
        acc_nc_pred = predicted_classes[acc_nc_indices]
        acc_nc_total = len(acc_nc_indices)
        acc_nc_correct = np.sum(acc_nc_true == acc_nc_pred)
        acc_nc_accuracy = acc_nc_correct / acc_nc_total if acc_nc_total > 0 else 0.0
        acc_nc_misclassified = np.sum(acc_nc_pred != 0)
        acc_nc_misclassification_rate = acc_nc_misclassified / acc_nc_total if acc_nc_total > 0 else 0.0
        canonical_analysis['acceptor_noncanonical'] = {
            'total': acc_nc_total,
            'correct': acc_nc_correct,
            'accuracy': acc_nc_accuracy,
            'misclassification_rate': acc_nc_misclassification_rate,
            'misclassified': int(acc_nc_misclassified)
        }

    # Acceptor Canonical Analysis
    if canonical_info['acceptor_canonical']:
        acc_can_indices = np.array(canonical_info['acceptor_canonical'])
        acc_can_true = test_labels[acc_can_indices]
        acc_can_pred = predicted_classes[acc_can_indices]
        acc_can_total = len(acc_can_indices)
        acc_can_correct = np.sum(acc_can_true == acc_can_pred)
        acc_can_accuracy = acc_can_correct / acc_can_total if acc_can_total > 0 else 0.0
        acc_can_misclassified = np.sum(acc_can_pred != 0)
        acc_can_misclassification_rate = acc_can_misclassified / acc_can_total if acc_can_total > 0 else 0.0
        canonical_analysis['acceptor_canonical'] = {
            'total': acc_can_total,
            'correct': acc_can_correct,
            'accuracy': acc_can_accuracy,
            'misclassification_rate': acc_can_misclassification_rate,
            'misclassified': int(acc_can_misclassified)
        }

    # Donor Non-canonical Analysis
    if canonical_info['donor_noncanonical']:
        don_nc_indices = np.array(canonical_info['donor_noncanonical'])
        don_nc_true = test_labels[don_nc_indices]
        don_nc_pred = predicted_classes[don_nc_indices]
        don_nc_total = len(don_nc_indices)
        don_nc_correct = np.sum(don_nc_true == don_nc_pred)
        don_nc_accuracy = don_nc_correct / don_nc_total if don_nc_total > 0 else 0.0
        don_nc_misclassified = np.sum(don_nc_pred != 1)
        don_nc_misclassification_rate = don_nc_misclassified / don_nc_total if don_nc_total > 0 else 0.0
        canonical_analysis['donor_noncanonical'] = {
            'total': don_nc_total,
            'correct': don_nc_correct,
            'accuracy': don_nc_accuracy,
            'misclassification_rate': don_nc_misclassification_rate,
            'misclassified': int(don_nc_misclassified)
        }

    # Donor Canonical Analysis
    if canonical_info['donor_canonical']:
        don_can_indices = np.array(canonical_info['donor_canonical'])
        don_can_true = test_labels[don_can_indices]
        don_can_pred = predicted_classes[don_can_indices]
        don_can_total = len(don_can_indices)
        don_can_correct = np.sum(don_can_true == don_can_pred)
        don_can_accuracy = don_can_correct / don_can_total if don_can_total > 0 else 0.0
        don_can_misclassified = np.sum(don_can_pred != 1)
        don_can_misclassification_rate = don_can_misclassified / don_can_total if don_can_total > 0 else 0.0
        canonical_analysis['donor_canonical'] = {
            'total': don_can_total,
            'correct': don_can_correct,
            'accuracy': don_can_accuracy,
            'misclassification_rate': don_can_misclassification_rate,
            'misclassified': int(don_can_misclassified)
        }
    
    return accuracy, f1, precision, recall, formatted_report, canonical_analysis