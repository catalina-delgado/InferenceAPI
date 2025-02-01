import tensorflow as tf
from tensorflow import keras
from layers.transformer import Transformer
import matplotlib as mpl
import numpy as np
from PIL import Image
import requests
import io
import base64
    
class CVTInference:
    def __init__(self, image, model, layer_name):
        self.image = image
        self.model = model
        self.layer_name = layer_name

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
        img = tf.keras.utils.img_to_array(image)
        return img

    # Funcion para preprocesar la imagen antes de la inferencia
    def preprocess_image(img_array):
        input_image = Image.fromarray(img_array.squeeze()).resize((256, 256))
        input_image = tf.keras.utils.img_to_array(input_image)
        if len(input_image.shape) == 2:  # Grayscale image
            input_image = np.expand_dims(input_image, axis=-1)
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image
    
    # Función para generar Grad-CAM
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

    # Funcion para aplicar Grad-CAM sobre la imagen
    def apply_gradcam(heatmap, img_array):
        heatmap = np.uint8(255 * heatmap)
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * 0.4 + img_array
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    
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
   
    # Funcion para inferir
    def call(self):
        # Leer imagen
        img_array = read_imagefile(self.image)
        # Generar Grad-CAM
        heatmap = generate_gradcam(self.model, self.layer_name, img_array)
        # Aplicar Grad-CAM sobre la imagen
        superimposed_img = apply_gradcam(heatmap, img_array)            
        # Convertir imagen a Base64
        buffered = io.BytesIO()
        superimposed_img.save(buffered, format="PNG")
        buffered.seek(0)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        # Preprocesar imagen
        input_image = preprocess_image(img_array)
        # Obtener predicción
        predictions = self.model.predict(input_image)
        idx, predicted_class, confidence = decode_predictions(predictions)
        
        return confidence, predicted_class, image_base64
