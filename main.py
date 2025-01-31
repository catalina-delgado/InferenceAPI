from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from layers.transformer import Transformer
import matplotlib as mpl
import numpy as np
from PIL import Image
import uvicorn
import requests
import io
import base64
import os


app = FastAPI()
app.mount("/layers", StaticFiles(directory="layers"), name="layers")
app.mount("/models", StaticFiles(directory="models"), name="models")


def __Tanh3(x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3

# Cargar modelos al iniciar
custom_objects = {
    '__Tanh3':__Tanh3,
    'transformer': Transformer
}
model = tf.keras.models.load_model(os.path.join("models/", "CVT.hdf5"), custom_objects=custom_objects)
LAYER_NAME = 'conv2d_27'

# Funcion para leer imagen
def read_imagefile(img_url):
    if img_url.startswith('data:image'):
        base64_data = img_url.split(",")[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
    else:
        response = requests.get(img_url)
        print('response', response)
        response.raise_for_status()  
        image = Image.open(io.BytesIO(response.content))
    
    image = image.resize((256,256))   
    if image.mode == 'RGBA' or image.mode == 'RGB':
        image = image.convert('L')
    img = keras.utils.img_to_array(image)
    return img

# Función para aplicar Grad-CAM
def generate_gradcam(model, layer_name, img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    if len(img_array.shape) == 2:  # Grayscale image
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
            
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Funcion para decodificar predicciones
def decode_predictions(preds):
    class_labels = {0: "cover", 1: "stego"}
    if len(preds.shape) == 1: 
        idx = int(np.argmax(preds)) 
        confidence = preds[idx]
    else:  
        idx = int(np.argmax(preds, axis=1)[0])  
        confidence = preds[0][idx]
    
    label = class_labels[idx]
    return idx, label, confidence

# Endpoint principal
@app.post("/generate_gradCam/")
async def predict(image: str = Form(...)):
  
    # Leer imagen
    img_array = read_imagefile(image)
    heatmap = generate_gradcam(model, LAYER_NAME, img_array)

    # Aplicar Grad-CAM sobre la imagen
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * 0.4 + img_array
    superimposed_img = keras.utils.array_to_img(superimposed_img)
        
    # Convertir imagen a Base64
    buffered = io.BytesIO()
    superimposed_img.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    # Preprocesar imagen
    input_image = Image.fromarray(img_array.squeeze()).resize((256, 256))
    input_image = keras.utils.img_to_array(input_image)
    if len(input_image.shape) == 2:  # Grayscale image
        input_image = np.expand_dims(input_image, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)

    # Obtener predicción
    predictions = model.predict(input_image)
    idx, predicted_class, confidence = decode_predictions(predictions)

    response_data = {
        "model_name": 'CVT',
        "layer_name": LAYER_NAME,
        "prediction_percentage": float(confidence),
        "predicted_class": predicted_class,
        "image": image_base64,
    }

    response = JSONResponse(content=response_data)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# API WELCOME
@app.get("/", response_class=HTMLResponse)
def read_root():
    return {
        "message": "Bienvenido a la API de Clasificación de Imágenes",
        "instructions": {
            "endpoint": "/generate_gradCam/",
            "method": "POST",
            "description": "Envía una imagen para obtener una predicción y un mapa de calor Grad-CAM.",
            "example": {
                "curl": 'curl -X POST -F "image=@ruta/a/tu/imagen.jpg" https://huggingface.co/spaces/MarilineDelgado/stegoapi/generate_gradCam/'
            }
        }
    }
        
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

