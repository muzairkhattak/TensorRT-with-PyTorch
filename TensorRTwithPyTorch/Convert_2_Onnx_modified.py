import cv2
import onnx
import torch
# from albumentations import (Compose,Resize,)
# from albumentations.augmentations.transforms import Normalize
# from albumentations.pytorch.transforms import ToTensor
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image


# class_names = image_datasets['train'].classes
class_names=open('./labels.txt')
class_names=class_names.read()
class_names=class_names.split('\n')
class_names.remove('')
print('Number of classes ',len(class_names))

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

def image_loader(loader, image):
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


# def preprocess_image(img_path):
#     # transformations for the input data
#     transformss = transforms.Compose([
# #         transforms.Resize(64, 64, interpolation=cv2.INTER_NEAREST),
#         transforms.Resize(64, 64),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         transforms.ToTensor(),
#     ])

#     # read input image
#     input_img = Image.open(img_path)
#     # do transformations
# #     input_data = transformss(image=input_img)["image"]
#     input_data = transformss(input_img)
#     # prepare batch
#     batch_data = torch.unsqueeze(input_data, 0)

#     return batch_data



def main():
    # load pre-trained model -------------------------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    model.load_state_dict(torch.load('./weights/resnet50_f.pth'))
    model = torch.nn.Sequential(model, torch.nn.Softmax(1))
    print('model')
    # preprocessing stage ----------------------------------------------------------------------------------------------
    input=image_loader(data_transforms['val'], Image.open("turkish_coffee.jpg")).cuda()
#     input = preprocess_image("turkish_coffee.jpg").cuda()

    print(input.shape)


    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "./weights/resnet50_f.onnx"

    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"],
                      verbose=False,output_names=["output"], export_params=True)

    #onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    #onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)


if __name__ == '__main__':
    main()
