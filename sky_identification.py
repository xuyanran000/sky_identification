import gradio as gr
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import medfilt

def convert_to_grayscale_and_blur(image):
    # converts an image to grayscale and then applies Gaussian blurring
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.blur(img_gray, (9, 3))
    return img_blurred

def calculate_gradient(img_blurred):
    # calculates the Laplacian gradient of a blurred image to detect edges
    laplacian = cv2.Laplacian(img_blurred, cv2.CV_8U)
    gradient_mask = (laplacian < 6).astype(np.uint8)
    return gradient_mask

def refine_skyline(mask):
    # refines the sky detection using median filtering and morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    eroded_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    skyline_mask = cal_skyline(eroded_mask)
    return skyline_mask

def cal_skyline(mask):
    # adjusts the skyline in the mask based on median filtering to isolate the sky
    h, w = mask.shape
    for i in range(w):
        column = mask[:, i]
        after_median = medfilt(column, kernel_size=19)
        try:
            first_white_index = np.where(after_median == 1)[0][0]
            first_black_index = np.where(after_median == 0)[0][0]
            if first_black_index > first_white_index:
                mask[:first_black_index, i] = 1
                mask[first_black_index:, i] = 0
        except IndexError:
            continue
    return mask

def get_sky_region(image, mask):
    # applies the mask to the original image to extract the sky region
    sky_region = cv2.bitwise_and(image, image, mask=mask)
    return sky_region



def sky_detection(input_img):
    # Process the image and return the final output
    img_blurred = convert_to_grayscale_and_blur(input_img)
    gradient_mask = calculate_gradient(img_blurred)
    skyline_mask = refine_skyline(gradient_mask)
    sky_region = get_sky_region(input_img, skyline_mask)
    return sky_region

demo = gr.Interface(fn=sky_detection, inputs=gr.Image(), outputs="image")
demo.launch(share=True)

