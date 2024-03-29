import common
import os
import tensorrt as trt
import numpy as np
from PIL import Image
import glob
import itertools
import time
import cv2

class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 64,64 )
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_normalized_image(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):

        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.

    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image
# def load_normalized_image(filename,pagelocked_buffer):
#     image = cv2.imread(filename)
#     image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_cv = cv2.resize(image_cv, (64, 64))
#     miu = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img_np = np.array(image_cv, dtype=float) / 255.
#     r = (img_np[:, :, 0] - miu[0]) / std[0]
#     g = (img_np[:, :, 1] - miu[1]) / std[1]
#     b = (img_np[:, :, 2] - miu[2]) / std[2]
#     img_np_t = np.array([r, g, b])
# #     img_np_nchw = np.expand_dims(img_np_t, axis=0)
#     np.copyto(pagelocked_buffer, img_np_t.ravel())
#     return filename

def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "data")
    # Get test images, models and labels.

    labels_file='./labels.txt'
    labels = open(labels_file, 'r').read().split('\n')
    onnx_model_file='./weights/resnet50_f.onnx'

    classes_folder = glob.glob('./arranged_data_final1/val/*')
    all_images = [glob.glob(classes_folder[i] + '/*') for i in range(len(classes_folder))]
    merged_images = list(itertools.chain.from_iterable(all_images))

    with open('./weights/resnet50.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    with engine.create_execution_context() as context:
        # Load a normalized test case into the host input page-locked buffer.
        start_time = time.time()
        count = 0
        total_correct=0
        for test_image in merged_images:
            test_case = load_normalized_image(test_image, inputs[0].host)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(test_case)
            print(trt_outputs[0])
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            pred = labels[np.argmax(trt_outputs[0])]

            if "_".join(pred.split()) in test_case.split('/'):
                print("Correctly recognized " + test_case + " as " + pred)
                total_correct +=1
            else:
                print("Incorrectly recognized " + test_case + " as " + pred)
            count=count+1


        end_time = time.time()
        print('Total time=', end_time - start_time)
        print('Total images processed=', count)
        print('Frames Per Seconds with Tensorrt Engine =', count / (end_time - start_time))
        print('Test Data accuracy with Tensorrt Model =', (total_correct/count)*100)



if __name__ == '__main__':
    main()
