from fastapi import APIRouter, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from inference import Inference
from dependencies import get_model, get_layer_name

router = APIRouter(prefix="/routers", tags=["routers"])

# Endpoint cvt
@router.post("/predict-cvt")
def predict_cvt(image: str = Form(...), model=Depends(lambda: get_model('cvt')), LAYER_NAME=Depends(lambda: get_layer_name('cvt'))):
    
    if model is None or LAYER_NAME is None:
        raise HTTPException(status_code=500, detail="Model or layer name not loaded")
    
    # Inferencia
    confidence, predicted_class, image_base64 = Inference(image, model, LAYER_NAME).call()
    
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

# Endpoint swint
@router.post("/predict-swint")
def predict_swint(image: str = Form(...), model=Depends(lambda: get_model('swint')), LAYER_NAME=Depends(lambda: get_layer_name('swint'))):
    
    if model is None or LAYER_NAME is None:
        raise HTTPException(status_code=500, detail="Model or layer name not loaded")
    
    # Inferencia
    confidence, predicted_class, image_base64 = Inference(image, model, LAYER_NAME).call()
    
    response_data = {
        "model_name": 'SWINT',
        "layer_name": LAYER_NAME,
        "prediction_percentage": float(confidence),
        "predicted_class": predicted_class,
        "image": image_base64,
    }

    response = JSONResponse(content=response_data)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response
