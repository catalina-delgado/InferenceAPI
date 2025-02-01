from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware
from models.cvt import DownloadModel
import routers
import uvicorn
import requests

from azure.storage.blob import BlobServiceClient

# Configuración
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mariline;AccountKey=2Oavrz+lzLK794gkGFINoBqY4MwO7VsUmqVaRMi2FMNkMNTctl0aScOuDdylho0O6zR7WLYy8rAB+AStcaPXdA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "model"
BLOB_NAME = "cvt-model"

# Cargar modelo
LAYER_NAME, modelcvt = DownloadModel(CONNECTION_STRING, CONTAINER_NAME, BLOB_NAME).load_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,  # Permitir cookies/autenticación
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

app.include_router(routers.router)

       
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

