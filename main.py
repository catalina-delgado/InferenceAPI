from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from routers import router
import uvicorn
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "https://www.boxia.site", "https://stego-app-catalina-delgados-projects.vercel.app"],  # Permitir todos los orígenes
    allow_credentials=True,  # Permitir cookies/autenticación
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

app.include_router(router)

# API WELCOME
@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a la API de Clasificación de Imágenes",
        "instructions": {
            "endpoints": ["/routers/predict-cvt", "/routers/predict-swint"],
            "method": "POST",
            "description": "Envía una imagen para obtener una predicción y un mapa de calor Grad-CAM.",
        }
    }
       
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

