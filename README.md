# TensorRT-with-PyTorch
In this repository, you will find Complete code files, with proper documentation on how to use PyTorch Models with TensorRT optimization for PCs, Jetson TX2 and Jetson Nano.

This repository have used some help from various sources including Nvidia sample code files for TensorRT and https://www.learnopencv.com/ .

By using the performence metrics like accuracy,total inference time and FPS we will compare the Pytorch NN model of resnet50 and the same Pytorch model optimzed with TensorRT (TensorRT is a Nivida's deep learning optimization library which optimizes the Neural Networks written and trained in differnt frameworks like ONNX,TensorFlow,Pytorch etc) . We will see that the performece interms of speed and FPS is enhanced after optimizing the network using TensorRT but there is small decrease in accuracy (but thats not a big deal because when using AI in production, inference time matter more !).

So before using the code files and getting help from this repository, it is necessary to have a Jeston device which has been fully set-up and one must have following things installed and ready:

-->You must have a Jetson device flashed (means you have Linux running up in your Jetson).For flashing of Nano, we simply use a imagefile (download link given        below) and burn it on the SD-Card using some image writing software like etcher. You can download the imagefile for jetson Nano (its basically the program        called JetPack which install OS and other AI tools in the Jetson Device) from https://developer.nvidia.com/jetson-nano-sd-card-image. Then simply insert the      SD-Card into Jetson Nano and you will see your Linux running up fine.
   For other Jetson Devices like Jetson TX2, you will need to flash it to by having ON-BOARD flashing procedure which is explained pretty good by the tutorials of    JetsonHacks, link for this is https://www.youtube.com/watch?v=s1QDsa6SzuQ&t=309s .

-->After flashing, first thing to install is the pip3 tool, which will be used to install Python Pakages later on (Use this command in terminal "sudo apt-get        install python3-pip" ).

-->By having your device flashed, you will have installed the  linux, TensorRT and some other related libraries for CV as well (like OPENCV, PIL etc),               unfortunately Pytorch and torchvision doesn't come preinstalled with flashing. So we will first need to download and install the Pytorch specifically for JETSON   DEVICES ( please don't use the normal installation method that we use in normal PC's for Pytorch). We will need to follow the steps provided by Nvidia offical     Developers (yes its DUSTIN :P ).
Remember to install that particular version of Pytorch which is compatible with the version of TensorRT (which is already installed during flashing), so for this lookup at the version of TensorRT your Jetson have and then install the recommended Pytorch (you will find the recommended Pytorch version which needs to be installed in the corresponding TensorRT version docmumentation, link is https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html ) version using this link 
https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048 

--> Also you will need to install some other libraries like PyCuda etc . For Pycuda use the following commands and run those in terminal 
 export PATH=/usr/local/cuda/bin:${PATH}            .
 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}               .
 pip3 install 'pycuda>=2019.1.1'                        .
 
 -------------------------------------------------------------------------
 So Now we have full enviroment ready to run the scripts in this repository and then compare the results !!
 
 The model which we have used is resNet50 (you can use any other model or custom model as well).
 Dataset used in this project is CFAR10.
 
 
 The basic steps that needs to be followed are as follows: 
 
 1) Preprocessing of dataset and converting it into IMAGENET dataset style:
   The script for that is included in the "trainonCOLAB" Notebook which will be running in COLAB.
   (IMPORTANT: upload the cfarutil.py in the COLAB before  running the notebook, as it will be required to load certain functions for the data preprocessing          part.)
   
 2) Training Phase:
     We will be using the Jetson Nano only for inference (although I have provided the script for training the model using Jetson Nano, but it will take alot of        time because of limit resources in edge devices). So first of all, use the notebook "trainonCOLAB" and run it on google COLAB to prepare the dataset ( it          will download the dataset and preprecess it as well ) and use it for training the model as well. Just donwload the weights file (.pth extension) ,as it will      be used further in the process.
   
  3) Performence metrics results on COLAB:
   The very same notebook will also do the inference on the test dataset and the relevent code is also included to show the results in the notebook using COLAB.
  4)Download the preprocessed dataset (or you can reuse the downloading/preprocessing code portion from this notebook and run it in the Jetson Nano) and the          trained model weights file from COLAB manually.
  
  Now as we have trained the model, swtich to te jetson device now:
  
  5)Finally evaulate the perfomence of the unoptimzied pytorch model by running the test.py script (do download the preprocessed dataset and weights file and put    them into the relevent location so that the test.py can access the dataset and weights file.
  
  --NOTE--YOU DONOT NEED TO ADD/CHANGE ANY LINE OF THIS "trainonCOLAB" notebook, JUST RUN THE CELLS.
  
 ------------------------------------------------------------------- 
 CONVERTING PYTORCH MODEL INTO OPTIMIZED TENSORRT ENGINE:
 
  So now the training part is completed, we have the preprocessed dataset and the trained model weights.
  So in order to convert the Pytorch Model into TensorRT engine we need to perfrom the following steps.
  
 6) Converting the Model into ONNX format:
   Run the python script Convert_2_Onnx_modified.py . (If necessary, change the directory path in the code to location where the weights have been saved in the      jetson device). It will save the the converted ONNX model for that pytorch model.
   
 7) Using ONNX parser of TensorRT to parse the converted ONNX model and build the engine:
   Run the python script buildig_trt_engine.py 
   It will save the optimzed tensorRT engine which will be used to inference now.
   
 -----------------------------------------------------------------------
 
 INFERENCE WITH TENSORRT
 
 8) Run the python script, trt_inference.py using terminal. 
   It will do the inferecne on same validation/test dataset and will show the results of performence metrics.
   
   
 ---------------------------
For using this repo with your own model and dataset, remember that the model weights should be from Pytorch trained network, and the dataset must be in the ImageNet style format.
 --------------------------------------------------------------------------
 TO USE THIS REPOSITORY WITH YOUR PC, TensorRT compatible with the CUDA and Pytorch version must be installed manually.
 --------------------------------------------------------------------------
 To install Tensorrt on desktop computers, use the link https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
 --------------------------------------------------------------------------
 JUST CLONE THE WHOLE REPOSITORY TO YOUR JETSON DEVICE AND Move your data into the "arranged_data_final folder", trained weights into the "weights" folder.
  
--------------
RESULTS/COMPARISION
--------
In the following results images, the first one shows the performence metrics with unoptimzed normal Pytorch Model on Jetson Nano (4GB), while the second image shows the results acheived using TensorRT with same PyTorch model.

![alt text](https://github.com/Uzair-Khattak/TensorRT-with-PyTorch/blob/main/part1.jpg)

![alt text](https://github.com/Uzair-Khattak/TensorRT-with-PyTorch/blob/main/engine_results.jpg)




