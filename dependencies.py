from fastapi import Depends
from azure.storage.blob import BlobServiceClient
from models.cvt import DownloadModelCVT
from models.swint import DownloadModelSWINT


# Configuración
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mariline;AccountKey=2Oavrz+lzLK794gkGFINoBqY4MwO7VsUmqVaRMi2FMNkMNTctl0aScOuDdylho0O6zR7WLYy8rAB+AStcaPXdA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME_CVT = "cvt-model"
BLOB_NAME_CVT = "CVT.hdf5"

CONTAINER_NAME_SWINT = "swint-model"
BLOB_NAME_SWINT = "SwintBlocks.hdf5"

# Cargar modelo
# Cargar modelo
try:
    print("Descargando modelo SWINT...")
    modelswint, LAYER_NAME_SWINT = DownloadModelSWINT(CONNECTION_STRING, CONTAINER_NAME_SWINT, BLOB_NAME_SWINT).load_model()
    print("Modelo SWINT descargado correctamente.")
except Exception as e:
    print(f"Error al descargar el modelo SWINT: {e}")
    LAYER_NAME_SWINT, modelswint = None, None

try:
    print("Descargando modelo CVT...")
    modelcvt, LAYER_NAME_CVT = DownloadModelCVT(CONNECTION_STRING, CONTAINER_NAME_CVT, BLOB_NAME_CVT).load_model()
    print("Modelo CVT descargado correctamente.")
except Exception as e:
    print(f"Error al descargar el modelo CVT: {e}")
    LAYER_NAME_CVT, modelcvt = None, None
    

def get_model(name: str):
    if name == 'swint':
        return modelswint
    elif name == 'cvt':
        return modelcvt

def get_layer_name(name: str):
    if name == 'swint':
        return LAYER_NAME_SWINT
    elif name == 'cvt':
        return LAYER_NAME_CVT