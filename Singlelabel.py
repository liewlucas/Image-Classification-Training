import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, accuracy_score,
    precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import logging  # Import the logging module
import warnings#prevent from printing warning messages
warnings.filterwarnings("ignore")


# Step 1: Data Splitting

#main_dir="cv\\small" #declare main dir
#data_dir = main_dir+"\\EuroSAT_RGB" #define data dir
data_dirc = "EuroSAT_RGB"
X, y = [],[]
cName=[] 
label_names = os.listdir(data_dir)
label_map = {label_name: index for index, label_name in enumerate(label_names)}

# Set up logging
log_dir = main_dir+"\\logs"  # define log directory
os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist
log_file = os.path.join(log_dir, 'T1Logs.txt')  # Define the log file path

logging.basicConfig(
    filename=log_file,  # Specify the log file
    level=logging.INFO,  # Set the logging level (you can change this)
    format='%(message)s'
)

# Log all print statements to the log file
def log_print(text):
    logging.info(text)
    print(text)

#declare code started
log_print("TASK 1 START")

#process all images
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        cName.append(class_name)
        class_label = label_map[class_name]
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            X.append(image)
            y.append(class_label)
    log_print(f"[Done processing {class_name}]")
log_print("[Data processing COMPLETED]")

# Split the data into training, validation, and test sets (i am using 80-20 split)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)# 80% training, 20% temporary
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 50% temporary -> 10% validation, 10% test


# Step 2: Custom Dataset Classes

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Step 3: Data Loading and Preprocessing

# Custom transformation function for sharpening and contrast enhancement (not used)
def enhance_image(image):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)

    # Apply sharpening using a kernel
    sharpening_kernel = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=np.float32)
    sharpened_image_cv = cv2.filter2D(image_cv, -1, sharpening_kernel)

    # Increase contrast (you can adjust the alpha and beta values)
    alpha = 1.4  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    enhanced_image_cv = cv2.convertScaleAbs(sharpened_image_cv, alpha=alpha, beta=beta)

    # Convert the OpenCV image back to a PIL image
    enhanced_image = Image.fromarray(enhanced_image_cv)

    return enhanced_image


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.FiveCrop(size=(16, 16)),
    #transforms.RandomCrop(size=(32, 32)), 
    transforms.ColorJitter(brightness=.1,contrast=.3,saturation=2, hue=.5),  # Increase the saturation
    #transforms.Lambda(enhance_image),  # Apply custom sharpening and contrast enhancement
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#apply transformation
X_train = [transform(Image.fromarray(image)) for image in X_train]
X_val = [transform(Image.fromarray(image)) for image in X_val]
X_test = [transform(Image.fromarray(image)) for image in X_test]

binarized_y_train = label_binarize(y_train, classes=np.unique(y_train))
binarized_y_val = label_binarize(y_val, classes=np.unique(y_val))
binarized_y_test = label_binarize(y_test, classes=np.unique(y_test))

# Create custom dataset objects
train_dataset = CustomDataset(X_train, binarized_y_train)
val_dataset = CustomDataset(X_val, binarized_y_val)
test_dataset = CustomDataset(X_test, binarized_y_test)

# Create DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Step 4: Model Training
model = torchvision.models.resnet18(pretrained=True)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

num_epochs = 5


train_losses = []  # To store training losses per epoch
val_losses = []  # To store validation losses per epoch

# Lists for storing class-specific metrics
class_avg_precisions = {i: [] for i in range(num_classes)}
class_accuracies = {i: [] for i in range(num_classes)}

for epoch in range(num_epochs):
    log_print(f"Epoch [{epoch+1}/{num_epochs}]:")

    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #print and let me know the progress
        if batch_idx % 10 == 9:
            log_print(f'  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')

    # Append the training loss for the entire epoch
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0.0

    # Store all predicted scores and labels for each class
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            val_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = torch.argmax(labels, dim=1).cpu().numpy()

            for class_idx in range(num_classes):
                class_true_labels = (true_labels == class_idx)
                class_predicted = (predicted == class_idx)
                class_accuracy = accuracy_score(class_true_labels, class_predicted)
                class_accuracies[class_idx].append(class_accuracy)


            all_scores.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))


    # Calculate average precision for each class
    all_scores = np.vstack(all_scores)
    all_labels = np.vstack(all_labels)

    for class_idx in range(num_classes):
        class_labels = all_labels[:, class_idx]
        class_scores = all_scores[:, class_idx]
        avg_precision_class = average_precision_score(class_labels, class_scores)
        class_avg_precisions[class_idx].append(avg_precision_class)


#to store the curve graph
os.makedirs(main_dir+"\\curve", exist_ok=True)  # Create the folder if it doesn't exist
plot_file_path = os.path.join(main_dir+"\\curve", 'T1Curve.png')

# Report loss curves
plt.figure()
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
# Save the plot as an image
plt.savefig(plot_file_path)
#close the plot window, you can use plt.close()
plt.close()
log_print(f"Loss curves saved to {plot_file_path}")

# Calculate mean average precision for each class on the validation set
mean_avg_precisions = {i: np.mean(class_avg_precisions[i]) for i in range(num_classes)}

# Calculate mean accuracy for each class on the validation set
mean_accuracies = {i: np.mean(class_accuracies[i]) for i in range(num_classes)}
total_precision=0
total_accuracy=0
log_print("==============================================================")
log_print("Mean Average Precision for Each Class (Validation Set):")
for i, avg_precision_class in mean_avg_precisions.items():
    log_print(f"{cName[i]}: {avg_precision_class:.4f}")
    total_precision+=avg_precision_class

mean_avg_precision = total_precision/len(mean_avg_precisions)
log_print(f"Mean Average Precision over all classes:: {mean_avg_precision:.4f}")
log_print("==============================================================")
log_print("Mean Accuracy for Each Class (Validation Set):")
for i, class_accuracy in mean_accuracies.items():
    log_print(f"{cName[i]}: {class_accuracy:.4f}")
    total_accuracy+=class_accuracy

mean_accuracy = total_accuracy/len(mean_accuracies)
log_print(f"Mean Average Precision over all classes:: {mean_accuracy:.4f}")

log_print("==============================================================")

# Lists to store class-specific metrics for test set
test_class_average_precisions = np.zeros(num_classes)
test_class_accuracies = np.zeros(num_classes)

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        
        true_labels = torch.argmax(labels, dim=1)
        
        # Compute class-specific Average Precision (AP) and accuracy
        for i in range(num_classes):
            # Calculate precision-recall curve and AP
            precision, recall, _ = precision_recall_curve(true_labels == i, outputs[:, i])
            ap = auc(recall, precision)
            test_class_average_precisions[i] = ap
            
            # Calculate accuracy
            accuracy_i = accuracy_score(true_labels == i, predicted == i)
            test_class_accuracies[i] = accuracy_i

# Calculate the mean Average Precision (mAP) and accuracy for all classes
test_mean_average_precision = np.mean(test_class_average_precisions)
test_mean_accuracy = np.mean(test_class_accuracies)

# log_print the results for each class
log_print("Test Metrics for Each Class (Test Set):")
for i in range(num_classes):
    log_print(f"{cName[i]}:")
    log_print(f"  Average Precision: {test_class_average_precisions[i] * 100:.2f}%")
    log_print(f"  Accuracy: {test_class_accuracies[i] * 100:.2f}%")

# log_print the overall mean Average Precision (mAP) and accuracy
log_print("Test Metrics (Overall Mean):")
log_print(f"  Mean Average Precision (mAP): {test_mean_average_precision * 100:.2f}%")
log_print(f"  Mean Accuracy: {test_mean_accuracy * 100:.2f}%")
