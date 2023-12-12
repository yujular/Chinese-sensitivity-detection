from . import bert
from . import cnnmodel
from . import model_load
from . import transfer

BertBaseModel = bert.BertBaseModel
TransferNet = transfer.TransferNet
CNNAdapter = cnnmodel.CNNAdapter

Model = model_load.Model
