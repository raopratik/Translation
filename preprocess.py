import sys
sys.path.append(".")

from transformers import XLMRobertaTokenizer
from torch.utils import data
from dataloader import Translation

class Preprocess:
    def __init__(self, language_code="aze-eng"):
        self.language_code = language_code
        self.translation_data = {

            "train": "data/" + self.language_code + "/ted-train.orig." + self.language_code,
            "dev": "data/" + self.language_code + "/ted-dev.orig." + self.language_code,
            "test": "data/" + self.language_code + "/ted-test.orig." + self.language_code
        }
        self.translation_tokenization = {
            "train": {
                "hrl": None,
                "lrl": None
            },
            "dev": {
                "hrl": None,
                "lrl": None
            },
            "test": {
                "hrl": None,
                "lrl": None
            }

        }
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base',
                                                             do_lowercase_and_remove_accent=True)
        self.train_loaders = None
        self.valid_loaders = None
        self.test_loaders = None

    def load_file(self, path):
        with open(path, 'r') as fh:
            clean_data = [x.rstrip().strip() for x in fh.readlines()]

        hrl = []
        lrl = []
        for text in clean_data:
            split_text = text.split("|||")
            lrl.append(split_text[0].strip())
            hrl.append(split_text[1].strip())

        return {
            "hrl": hrl,
            "lrl": lrl
        }

    def load_data(self):
        for key in self.translation_data:
            self.translation_data[key] = self.load_file(self.translation_data[key])

    def tokenize_data(self):
        for split in self.translation_data:
            for key in self.translation_data[split]:
                self.translation_tokenization[split][key] = \
                    self.tokenize_language(self.translation_data[split][key])

    def tokenize_language(self, data):
        return self.tokenizer(data, return_tensors="pt", padding="max_length",
                              max_length=256)

    def get_loaders(self):
        train_dataset = Translation(input_dict=self.translation_tokenization["train"])
        dev_dataset = Translation(input_dict=self.translation_tokenization["dev"])
        test_dataset = Translation(input_dict=self.translation_tokenization["test"])

        loader_args = dict(shuffle=True, batch_size=16, num_workers=8,
                           pin_memory=True)

        self.train_loaders = data.DataLoader(train_dataset, **loader_args)
        self.valid_loaders = data.DataLoader(dev_dataset, **loader_args)
        self.test_loaders = data.DataLoader(test_dataset, **loader_args)

    def setup(self):
        self.load_data()
        self.tokenize_data()
        self.get_loaders()

if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.setup()

