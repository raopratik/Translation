# from transformers import RobertaTokenizer, RobertaModel


class Preprocess:
    def __init__(self, language_code="aze-eng"):
        self.language_code = language_code
        self.translation_data = {

            "train": "data/" + self.language_code + "/ted-train.orig." + self.language_code,
            "dev": "data/" + self.language_code + "/ted-dev.orig." + self.language_code,
            "test": "data/" + self.language_code + "/ted-test.orig." + self.language_code
        }

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
    

    def setup(self):
        self.load_data()
        print(self.translation_data)


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.setup()

