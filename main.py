import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from dataset import PNGDataset  # Ensure this module is correctly implemented
import torch.optim as optim
import copy
import os

# ============================================================
# 1. Define Custom Transformations
# ============================================================

class CustomTransform:
    def __init__(self, resize_size=(256, 256)):
        self.resize_size = resize_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.transform(image)

# ============================================================
# 2. Define Custom CNN Model
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # Output: 32x256x256
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x128x128

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Output: 64x128x128
        self.bn2 = nn.BatchNorm2d(64)
        # After pooling: 64x64x64

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Output: 128x64x64
        self.bn3 = nn.BatchNorm2d(128)
        # After pooling: 128x32x32

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional Layers with ReLU and Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x128x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128x32x32

        # Flatten
        x = x.view(x.size(0), -1)  # 128*32*32

        # Fully Connected Layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation here (CrossEntropyLoss applies Softmax)

        return x

# ============================================================
# 3. Define Plotting Functions
# ============================================================

def plot_loss_curve(train_losses, test_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(test_losses, label="Validation Loss", marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_curve(train_accuracies, test_accuracies, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(test_accuracies, label="Validation Accuracy", marker='o')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(y_true, y_pred, class_names, save_path):
    labels = unique_labels(y_true, y_pred)
    columns = [f"Predicted {class_names[label]}" for label in labels]
    indices = [f"Actual {class_names[label]}" for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=columns, index=indices)

    plt.figure(figsize=(8, 6))
    sns.heatmap(table, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")
    plt.savefig(save_path)
    plt.close()

def plot_ROC_curve(y_true, y_pred_probs, num_classes, class_names, save_path):
    y_bin_true = label_binarize(y_true, classes=list(range(num_classes)))

    FPR = dict()
    TPR = dict()
    ROC_AUC = dict()

    for i in range(num_classes):
        FPR[i], TPR[i], _ = roc_curve(y_bin_true[:, i], y_pred_probs[:, i])
        ROC_AUC[i] = auc(FPR[i], TPR[i])

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        plt.plot(FPR[i], TPR[i],
                 label=f"Class {class_names[i]} (AUC = {ROC_AUC[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ============================================================
# 4. Define Checkpoint Functions
# ============================================================

def save_checkpoint(state, checkpoint_dir, filename='best_model_checkpoint.pth'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'. Starting from scratch.")
        return 0, 0.0, [], [], [], []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    test_accuracies = checkpoint.get('test_accuracies', [])

    print(f"Checkpoint loaded. Resuming from epoch {epoch}. Best Validation Accuracy: {best_acc:.2f}%")
    return epoch, best_acc, train_losses, test_losses, train_accuracies, test_accuracies

# ============================================================
# 5. Main Execution
# ============================================================

if __name__ == "__main__":
    # Define directories
    root_directory = r"C:\Users\Jasso\OneDrive\桌面\pytorch\100"
    checkpoint_dir = r"C:\Users\Jasso\OneDrive\桌面\pytorch\checkpoints"
    result_images_dir = r"C:\Users\Jasso\OneDrive\桌面\pytorch\result_images"
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Define class labels and names
    class_labels = {"F": 0, "N": 1, "Q": 2, "V": 3}
    num_classes = len(class_labels)
    class_names = ["F", "N", "Q", "V"]

    # Initialize dataset
    dataset = PNGDataset(root_dir=root_directory, class_labels=class_labels, transform=CustomTransform())
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 32

    # Calculate label distribution in the training set
    train_labels = [dataset[i][1] for i in train_dataset.indices]
    class_counts = [0] * num_classes
    for label in train_labels:
        if 0 <= label < num_classes:
            class_counts[label] += 1

    print(f"Training set class counts: {class_counts}")

    # Calculate weights for each class to handle class imbalance
    class_weights = 1. / np.array(class_counts, dtype=np.float32)
    class_weights = torch.tensor(class_weights).to(torch.float32)
    sample_weights = np.array([class_weights[label].item() for label in train_labels])

    # Create sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = CustomCNN(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize training history lists
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Load saved model weights (optional)
    try:
        start_epoch, best_acc, train_losses, test_losses, train_accuracies, test_accuracies = load_checkpoint(
            best_model_path, model, optimizer
        )
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        # If checkpoint loading fails, start from scratch
        start_epoch = 0
        best_acc = 0.0

    # Define training parameters
    num_epochs = 25
    patience = 5  # For early stopping
    counter = 0

    print(f"There are {dataset_size} samples in the dataset, "
          f"with {train_size} for training and {test_size} for testing.")

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        n_correct = 0
        n_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / train_size
        epoch_acc = 100.0 * n_correct / n_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        running_loss_val = 0.0
        n_correct_val = 0
        n_samples_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                n_samples_val += labels.size(0)
                n_correct_val += (predicted == labels).sum().item()

        epoch_loss_val = running_loss_val / test_size
        epoch_acc_val = 100.0 * n_correct_val / n_samples_val
        test_losses.append(epoch_loss_val)
        test_accuracies.append(epoch_acc_val)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
              f'Val Loss: {epoch_loss_val:.4f}, Val Acc: {epoch_acc_val:.2f}%')

        # Check if this is the best model
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }, checkpoint_dir, filename='best_model_checkpoint.pth')
            print(f"Best model updated and saved at epoch {epoch + 1} with validation accuracy {epoch_acc_val:.2f}%.")

            # Reset early stopping counter
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

        # Adjust learning rate
        scheduler.step()

    print("Finished Training.")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the best model's state_dict
    final_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Best model saved at {final_model_path}")

    # Save training and validation curves
    plot_loss_curve(train_losses, test_losses, save_path=os.path.join(result_images_dir, "loss_curve.png"))
    plot_accuracy_curve(train_accuracies, test_accuracies, save_path=os.path.join(result_images_dir, "accuracy_curve.png"))
    print("Loss curve and accuracy curve have been saved.")

    # ============================================================
    # 6. Evaluation and Plotting
    # ============================================================

    # Initialize evaluation metrics
    running_loss = 0.0
    n_correct = 0
    n_samples = 0.0
    all_true_labels = []
    all_pred_probs = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss = running_loss / test_size
    test_accuracy = 100.0 * n_correct / n_samples
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Calculate per-class accuracy
    n_class_correct = [0 for _ in range(num_classes)]
    n_class_samples = [0 for _ in range(num_classes)]

    y_pred = np.array(all_preds)
    y_true = np.array(all_true_labels)

    for i in range(len(y_true)):
        label = y_true[i]
        pred = y_pred[i]
        if label == pred:
            n_class_correct[label] += 1
        n_class_samples[label] += 1

    for i in range(num_classes):
        if n_class_samples[i] > 0:
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Class {class_names[i]} Accuracy: {acc:.2f}%")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix heatmap
    plot_heatmap(y_true, y_pred, class_names, save_path=os.path.join(result_images_dir, "heatmap.png"))
    print("Confusion matrix has been saved.")

    # Plot ROC curve
    all_pred_probs = np.array(all_pred_probs)
    plot_ROC_curve(y_true, all_pred_probs, num_classes, class_names, save_path=os.path.join(result_images_dir, "ROC_curve.png"))
    print("ROC curve has been saved.")

    print("Evaluation complete, task finished.")
