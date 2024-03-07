import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

# Define the path to your dataset folder
dataset_dir = 'EuroSAT_RGB'

# List all class folders in the dataset directory
class_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Create lists to store image paths and corresponding labels
image_paths = []
labels = []

# Iterate through class folders and collect image paths and labels
for class_folder in class_folders:
    class_path = os.path.join(dataset_dir, class_folder)
    class_label = class_folder
    class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
    image_paths.extend(class_images)
    labels.extend([class_label] * len(class_images))

print("Loading Data...")
# Split the data into train, validation, and test sets using a random generator seed
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=37)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=37)

# Define label changes for multi-label classification
mlb = MultiLabelBinarizer(classes=class_folders)

# Fit the MultiLabelBinarizer with the labels
mlb.fit([y.split('_') for y in labels])

# Binarize string labels using scikit-learn's MultiLabelBinarizer
y_train = mlb.transform([y.split('_') for y in y_train])
y_val = mlb.transform([y.split('_') for y in y_val])
y_test = mlb.transform([y.split('_') for y in y_test])

# Apply label changes as specified
y_train[:, class_folders.index('AnnualCrop')] = y_train[:, class_folders.index('PermanentCrop')]
y_train[:, class_folders.index('PermanentCrop')] = y_train[:, class_folders.index('AnnualCrop')]

y_val[:, class_folders.index('AnnualCrop')] = y_val[:, class_folders.index('PermanentCrop')]
y_val[:, class_folders.index('PermanentCrop')] = y_val[:, class_folders.index('AnnualCrop')]

y_test[:, class_folders.index('AnnualCrop')] = y_test[:, class_folders.index('PermanentCrop')]
y_test[:, class_folders.index('PermanentCrop')] = y_test[:, class_folders.index('AnnualCrop')]

y_train[:, class_folders.index('HerbaceousVegetation')] = y_train[:, class_folders.index('Forest')]
y_val[:, class_folders.index('HerbaceousVegetation')] = y_val[:, class_folders.index('Forest')]
y_test[:, class_folders.index('HerbaceousVegetation')] = y_test[:, class_folders.index('Forest')]

class AAI3001Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the custom dataset.

        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of labels (multi-label format).
            transform (callable, optional): Optional data transformations (e.g., resizing, data augmentation).
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        # Ensure the image has 3 color channels (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply data transformations if provided
        if self.transform:
            image = self.transform(image)

        # Convert the image and label to PyTorch tensors
        image = torch.Tensor(np.array(image) / 255.0)  # Normalize without transposing
        label = torch.Tensor(label)

        return image, label

# Define data transformations for train, validation, and test sets
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(16),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(16),
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = val_transform  # Test set transformation can be similar to the validation set


# Create DataLoader instances for training, validation, and test sets
batch_size = 64  #  adjust this based on system's capabilities
train_dataset = AAI3001Dataset(X_train, y_train, transform=train_transform)
val_dataset = AAI3001Dataset(X_val, y_val, transform=val_transform)
test_dataset = AAI3001Dataset(X_test, y_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class MultiLabelResNet18(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelResNet18, self).__init__()  # Pass the class name and self
        self.resnet18 = models.resnet18(pretrained=True)  # Load the pre-trained ResNet-18 model
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


# Initialize the ResNet-18 model
num_classes = len(class_folders)
model = MultiLabelResNet18(num_classes)

# Define the loss function (use Binary Cross-Entropy for multi-label classification)
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.0004)

print("TRAINING MODEL....")

# Calculate the number of batches for each epoch
num_batches = len(train_loader)
print(f"Number of Batches per Epoch: {num_batches}")

# Training loop with loss tracking
num_epochs = 5  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

train_losses = []  # Track training loss per epoch
val_losses = []    # Track validation loss per epoch

for epoch in range(num_epochs):
    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for batch_num, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log batch number and loss
        print(f"Batch: {batch_num}, Loss: {loss.item():.4f}")

    # Calculate average training loss per epoch
    average_train_loss = running_loss / len(train_loader)
    train_losses.append(average_train_loss)

    # Print a message at the end of each epoch
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {average_train_loss:.4f}")

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {average_train_loss:.4f}, "
          f"Validation Loss: {average_val_loss:.4f}")
    


# Evaluate on the test set
def evaluate(loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y_true.append(labels.cpu().numpy())
            y_pred.append(torch.sigmoid(outputs).cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return y_true, y_pred

#Evaluate on the validation set
val_y_true, val_y_pred = evaluate(val_loader)

# Initialize lists to store class-wise metrics for validation
val_class_ap = []  # List for average precision on validation
val_class_accuracies = []  # List for accuracy on validation

for i in range(num_classes):
    # Calculate average precision and accuracy for each class on the validation set
    val_ap = average_precision_score(val_y_true[:, i], val_y_pred[:, i])
    val_accuracy = accuracy_score(val_y_true[:, i], (val_y_pred[:, i] > 0.5).astype(int))

    val_class_ap.append(val_ap)
    val_class_accuracies.append(val_accuracy)

    # Print the results for the validation set
    print(f"Validation Metrics for Class {class_folders[i]}:")
    print(f"  Average Precision: {val_ap:.2%}")
    print(f"  Accuracy: {val_accuracy:.2%}")

# Mean Average Precision (mAP) on the validation set
val_class_ap = []
for i in range(num_classes):
    val_ap = average_precision_score(val_y_true[:, i], val_y_pred[:, i])
    val_class_ap.append(val_ap)

val_mean_ap = sum(val_class_ap) / num_classes
val_mean_ap_percentage = val_mean_ap * 100
print(f"Mean Average Precision (mAP) on Test Set: {val_mean_ap_percentage:.2f}%")

# Calculate the mean accuracy over all classes on the validation set
val_mean_acc = sum(val_class_accuracies) / num_classes
val_mean_acc_percentage = val_mean_acc * 100
print(f"Mean Accuracy over All Classes on Test Set: {val_mean_acc_percentage:.2f}%")

#Evaluate on the test set
test_y_true, test_y_pred = evaluate(test_loader)

# Initialize lists to store class-wise metrics
class_ap = []  # List for average precision
class_accuracies = []  # List for accuracy

for i in range(num_classes):
    # Calculate average precision and accuracy for each class
    ap = average_precision_score(test_y_true[:, i], test_y_pred[:, i])
    accuracy = accuracy_score(test_y_true[:, i], (test_y_pred[:, i] > 0.5).astype(int))

    class_ap.append(ap)
    class_accuracies.append(accuracy)

 # Print the results in a nicely formatted way
    print(f"Test Metrics for Class {class_folders[i]}:")
    print(f"  Average Precision: {ap:.2%}")
    print(f"  Accuracy: {accuracy:.2%}")

# Mean Average Precision (mAP)
class_ap = []
for i in range(num_classes):
    ap = average_precision_score(test_y_true[:, i], test_y_pred[:, i])
    class_ap.append(ap)

mean_ap = sum(class_ap) / num_classes
mean_ap_percentage = mean_ap * 100
print(f"Mean Average Precision (mAP) on Test Set: {mean_ap_percentage:.2f}%")

# Calculate the mean accuracy over all classes on the test set
test_mean_acc = sum(class_accuracies) / num_classes
test_mean_acc_percentage = test_mean_acc * 100
print(f"Mean Accuracy over All Classes on Test Set: {test_mean_acc_percentage:.2f}%")



# Plot the loss curves
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
