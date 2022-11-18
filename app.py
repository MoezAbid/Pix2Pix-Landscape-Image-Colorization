# Imports
import gradio as gr
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
from PIL import Image
from keras.utils import img_to_array
from tensorflow_addons.layers import InstanceNormalization

# Loading Model
MODEL_PATH = "./model/gen0.h5"
model = keras.models.load_model(MODEL_PATH,custom_objects={'InstanceNormalization': InstanceNormalization}, compile=False)

def colorize(inp):
    img = cv2.cvtColor(cv2.imread(inp), cv2.COLOR_BGR2RGB)
    img_array = img_to_array(Image.fromarray(cv2.resize(img,(128,128))))
    transformed_img = (img_array/127.5) - 1
    expanded_img = np.expand_dims(transformed_img, 0)
    colorized_image = model(expanded_img)[0]
    return inp, colorized_image

def display(bw_image, gen_image):
    plt.figure(figsize = (5, 5))
    plt.imshow((gen_image + 1.0) / 2.0)
    plt.title('Generated Color Image',fontsize = 20)
    plt.axis('off')
    plt.savefig("./demo/result.jpg")


def predict(inp):
    original_input, colorized_image = colorize(inp)
    display(original_input, colorized_image)
    colorization_result = "./demo/result.jpg"
    return colorization_result

def get_example_images():
    examples = os.listdir("./examples")
    for i in range(len(examples)):
        examples[i] = "./examples/" + examples[i]
    return examples

gr.Interface(fn=predict,
title="Black & White Image Colorization",
             inputs = gr.Image(type="filepath"),
             outputs = gr.Image(shape=(3,3), label="Colorized Image").style(full_width=True, height=250, width=725), 
             examples = get_example_images()
             ).launch()