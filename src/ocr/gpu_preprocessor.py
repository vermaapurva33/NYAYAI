# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

import cv2
from pathlib import Path
from src.common.config import USE_GPU


def preprocess_image(input_path: Path, output_path: Path) -> Path:
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError("Failed to load image")

    if USE_GPU and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_img = cv2.cuda_GpuMat() # create GPU matrix which will live in GPU memory   
        gpu_img.upload(img) # upload image to GPU memory
        gpu_img = cv2.cuda.fastNlMeansDenoising(gpu_img) # perform denoising on GPU
        img = gpu_img.download() # download result back to system memory
    else:
        print("[INFO] during preprocessing, GPU not available, running CPU denoising")
        img = cv2.fastNlMeansDenoising(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img) # save the processed image to the specified path

    return output_path
