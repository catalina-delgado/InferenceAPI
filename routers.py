from fastapi import APIRouter, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from inference.cvt_inference import CVTInference


router = APIRouter(prefix="/routers", tags=["routers"])

# Endpoint principal
@router.post("/predict")
def predict(image: str = Form(...)):
    
    # Inferencia
    confidence, predicted_class, image_base64 = CVTInference(image, model, LAYER_NAME)
    
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
@router.get("/", response_class=HTMLResponse)
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
 