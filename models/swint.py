import tensorflow as tf
from tensorflow.keras import backend as K
from layers.cnn import SEBlock
from layers.swint import SwinTransformer, PatchMerging, PatchEmbedding, ReshapeLayer, SwinTBlock
from azure.storage.blob import BlobServiceClient

class DownloadModelSWINT:
    def __init__(self, CONNECTION_STRING, CONTAINER_NAME, BLOB_NAME):
        self.CONNECTION_STRING = CONNECTION_STRING
        self.CONTAINER_NAME = CONTAINER_NAME
        self.BLOB_NAME = BLOB_NAME
        self.LAYER_NAME = 'conv2d_4'
        
        # Ruta en Azure Web Apps (no persistente)
        self.local_path = "SWINT_local.hdf5" 
    
    @staticmethod    
    def __Tanh3(x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3

    def load_model(self):
        blob_service = BlobServiceClient.from_connection_string(self.CONNECTION_STRING)
        blob_client = blob_service.get_blob_client(self.CONTAINER_NAME, self.BLOB_NAME)
        
        with open(self.local_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
            
        custom_objects = {
        '__Tanh3':self.__Tanh3,
        'SEBlock': SEBlock,
        'SwinTransformer': SwinTransformer,
        'PatchMerging': PatchMerging,
        'PatchEmbedding': PatchEmbedding,
        'ReshapeLayer': ReshapeLayer,
        'SwinTBlock': SwinTBlock
        }
        

        model = tf.keras.models.load_model(self.local_path, custom_objects=custom_objects)
        
        return model, self.LAYER_NAME

