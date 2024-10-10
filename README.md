# MonReader Short Video Classification

### Background:

Our company develops innovative Artificial Intelligence and Computer Vision solutions that revolutionize industries. Machines that can see: We pack our solutions in small yet intelligent devices that can be easily integrated to your existing data flow. Computer vision for everyone: Our devices can recognize faces, estimate age and gender, classify clothing types and colors, identify everyday objects and detect motion. Technical consultancy: We help you identify use cases of artificial intelligence and computer vision in your industry. Artificial intelligence is the technology of today, not the future.

MonReader is a new mobile document digitization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and MonReader handles everything: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

### Goal(s): 
- Predict if the page is being flipped using a single image.
- Success Metrics: Evaluate model performance based on F1 score, the higher the better.
- Bonus(es): Predict if a given sequence of images contains an action of flipping.

### Data Description:  

We collected page flipping video from smart phones and labelled them as flipping and not flipping.  We clipped the videos as short videos and labelled them as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

### Methodology:

1.    Data Loading and Preprocessing: Load the video frames and preprocess them. The dataset is organized into two folders, training and testing, each containing subfolders flip (for flipping frames) and not_flip (for non-flipping frames). Each image is preprocessed by:

* Resizing the images to 128x128 pixels.
* Normalizing pixel values using ImageNet’s mean and standard deviation.
* Converting images to tensors for model input.
  
```
# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 pixels, numbers dependent on model selection
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize images dependents on the pre-train model
])
```
 * Plot the Transform Images
   
   <img width="724" alt="Screenshot 2024-10-09 at 11 43 36 PM" src="https://github.com/user-attachments/assets/496031b1-6e7b-4d5b-8896-8c1f62eb3cf1">

2.    Define a CNN Model: A basic CNN architecture will be implemented. The CNN is designed to extract spatial features from each frame. It consists of three convolutional layers, each followed by ReLU activation and max pooling. After the convolutional layers, a fully connected layer is used to classify the frames as either flipping or not flipping.
```
import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN model
class FlipCNN(nn.Module):
    def __init__(self):
        super(FlipCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  #combination

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust based on input image size
        self.fc2 = nn.Linear(512, 2)  # Output layer (2 classes: flipping or not flipping)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Classifier

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 16 * 16)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Instantiate the model, loss function, and optimizer
model = FlipCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
3.    Train the CNN: Train the CNN to classify frames as flipping or not flipping. The CNN is trained using CrossEntropyLoss and the Adam optimizer. During each epoch, the model processes batches of images, computes the loss, backpropagates the error, and updates the weights to minimize the classification error.
```
# Create train & test loop functions

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  train_loss = 0
  correct = 0
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

  train_loss /= len(dataloader)
  train_accuracy = correct / size

  train_losses.append(train_loss)
  train_accuracies.append(train_accuracy)


def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  test_accuracy = correct / size
  print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

  test_losses.append(test_loss)
  test_accuracies.append(test_accuracy)
```
4.    Evaluate the Model: After training, the model is evaluated on the test set using accuracy and loss as performance metrics. The model’s predictions are compared to the ground truth, and accuracy is calculated.
![image](https://github.com/user-attachments/assets/cc9c9cf9-7382-4c73-8d52-981679e4f0fc)
![image](https://github.com/user-attachments/assets/0bb97c67-711f-456f-a821-ccb57c253871)

### Conclusion:

Comparing with the Loss Curve:

* In the Loss Curve, both training and testing losses decreased significantly over the epochs, suggesting that the model is improving its performance with minimal errors.
* The Accuracy Curve complements this by showing that as the loss reduces, the model’s accuracy increases, reaching near-perfect classification by the final epoch.

Performance Summary:

* No Overfitting: Both training and testing accuracies are very close to each other throughout the epochs. This indicates that the model is not overfitting to the training data and generalizes well to the test set.
* Convergence: The accuracy and loss curves indicate that the model is converging well after a few epochs (around epoch 4). The model achieves high accuracy with minimal further improvement beyond this point.
* High Accuracy: The model consistently achieves over 90% accuracy on both the training and testing datasets, showing that it effectively differentiates between flipping and not flipping images.

Bonus: 

Predict that a given short video(sequence of images) contains an action of flipping. Apply the same procedures by embedding the LSMT technique and training the CNN + LSTM model using a sequence of images (video frames) as input. Each sequence corresponds to either a “flipping” action or a “not flipping” action. The rapid decrease in both training and testing loss in the early epochs shows that the model is learning well from the data and making significant improvements in its predictions. The training and testing losses follow a similar trend and remain close throughout all epochs. If the training loss kept decreasing while the testing loss started increasing, it would suggest overfitting. However, the model generalizes well to both the training and test data.
