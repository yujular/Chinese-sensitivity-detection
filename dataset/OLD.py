import abc
import os

from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class OLDBase(Dataset):
    DATA_FOLDER = abc.ABCMeta

    def __init__(self, root_path, datatype, model_name_or_path, class_num=2, max_length=128):
        self.root_path = root_path
        self.path = os.path.join(self.root_path, self.DATA_FOLDER)
        self.class_num = class_num
        self.datatype = datatype
        self.max_length = max_length
        self.model = model_name_or_path

        # 加载tokenizer, 自动添加CLS, SEP
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model,
                                                           add_special_tokens=True,
                                                           do_lower_case=True,
                                                           do_basic_tokenize=True)

        self.data = {}
        self.load_data(self.datatype)

    @abc.abstractmethod
    def load_data(self, datatype):
        pass
