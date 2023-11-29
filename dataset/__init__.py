from . import COLD
from . import OLID
from . import load_data

load_dataset = load_data.load_dataset
get_dataloader = load_data.get_dataloader
COLDataset = COLD.COLDataset
OLIDataset = OLID.OLIDataset
