import os
from torchtext import data


class RecommendDataset(data.Dataset):

    MethodBody_FILE = "body.txt"
    LABELS_FILE = "prefix.txt"

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        fields = [('text', text_field), ('label', label_field)]
        examples = []
        print(path)

        with open(os.path.join(path, self.MethodBody_FILE)) as c_file, \
                open(os.path.join(path, self.LABELS_FILE)) as l_file:
            for body, label in zip(c_file, l_file):
                example = data.Example.fromlist([body, label], fields)
                examples.append(example)

        super(RecommendDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, path, train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(RecommendDataset, cls).splits(
            path=path, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)
