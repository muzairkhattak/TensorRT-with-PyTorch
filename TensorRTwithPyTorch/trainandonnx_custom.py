from __future__ import print_function, division
import itertools

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import ImageFile,Image
import  glob
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'arranged_data_final1'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(len(class_names))

with open('labels.txt', 'w') as filehandle:
    for listitem in class_names:
        filehandle.write('%s\n' % listitem)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#now making of the network
class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
    self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
    self.fc1=nn.Linear(in_features=12*13*13,out_features=120)
    self.fc2=nn.Linear(in_features=120,out_features=60)
    self.out=nn.Linear(in_features=60,out_features=2)

  def forward(self,t):
    #input layer
    t=t
    #conv1 layer
    t=self.conv1(t)
    t=F.relu(t)
    t=F.max_pool2d(t,kernel_size=(2,2),stride=2)
    #2nd conv2d layer
    t=self.conv2(t)
    t=F.relu(t)
    t=F.max_pool2d(t,kernel_size=(2,2),stride=2)
    #now linear layer implementation
    #first flatten the conv layer output 
    t=t.reshape(-1,12*13*13)
    t=self.fc1(t)
    t=F.relu(t)
    #next linear
    t=self.fc2(t)
    t=F.relu(t)
    #final layer
    t=self.out(t)
#     t=torch.sigmoid(t)
    
    return t
model_ft = Network()

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(),lr=0.001)



# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def image_loader(loader, image):
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image





def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward propagation.
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [train_loss, train_acc, val_loss, val_acc]


model_ft, info = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
torch.save(model_ft.state_dict(), './weights/resnet50custom_f.pth')

#Now converting into ONNX as well
model_ft = torch.nn.Sequential(model_ft, torch.nn.Softmax(1))
input=image_loader(data_transforms['val'], Image.open("turkish_coffee.jpg")).cuda()

ONNX_FILE_PATH = "./weights/custom_f.onnx"

torch.onnx.export(model_ft, input, ONNX_FILE_PATH, input_names=["input"],
                      verbose=False,output_names=["output"], export_params=True)

    #onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    #onnx.checker.check_model(onnx_model)

print("Model was successfully converted to ONNX format.")
print("It was saved to", ONNX_FILE_PATH)



