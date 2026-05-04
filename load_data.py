from datasets import load_dataset, concatenate_datasets, disable_progress_bars, ClassLabel


class AugmentedDataLoader:
    def __init__(self, corpus):
        self.corpus = corpus
        self.opts = {
            'spa': 'CONLL02_spa',
            'nld': 'CONLL02_nld',
            'deu': 'GermEval_deu',
            'fin': 'TurkuNLP_fin',
            'zul': 'MasakhaNER_2_zul',
            'twt': 'Tweebank_eng',
            'wnut': 'WNUT17_eng'
        }
        self.path = None
        self.config = None
        self.revision = None
        self.train = None
        self.dev = None
        self.test = None
        self.aug_train = None
        self.lower_test = None
        self.upper_test = None
        self.labels = None
        self.id2label = None
        self.label2id = None

        disable_progress_bars()
        self._load_data()
        self._augment_data()

    def _lower(self, example):
        example["tokens"] = [token.lower() for token in example["tokens"]]
        return example

    def _upper(self, example):
        example["tokens"] = [token.upper() for token in example["tokens"]]
        return example

    def _swap_index(self, example, index_map):
        example['ner_tags'] = [index_map[i] if i in index_map else i for i in example['ner_tags']]
        return example

    def _load_data(self):
        if self.corpus == 'eng':
            self.path = 'eriktks/conll2003'
            self.revision = 'convert/parquet'
        else:
            self.path = 'bltlab/open-ner-core-types'
            self.config = self.opts[self.corpus]
        data = load_dataset(path=self.path, name=self.config, revision=self.revision)

        self.train = data['train']
        self.test = data['test']
        if self.corpus == 'eng':
            self.dev = data['validation']
        else:
            self.dev = data['dev']

        if self.corpus in ['twt', 'wnut']:
            self._load_socmed_data()

        self.labels = self.test.features['ner_tags'].feature.names
        self.id2label = {id: label for id, label in enumerate(self.labels)}
        self.label2id = {label: id for id, label in enumerate(self.labels)}

    def _load_socmed_data(self):
        base_data = load_dataset(path='eriktks/conll2003', revision='convert/parquet')
        index_map = {1: 5, 2: 6, 5: 1, 6: 2, 7: 0, 8: 0}
        self.train = base_data['train'].map(self._swap_index, fn_kwargs={'index_map': index_map})
        self.dev = base_data['validation'].map(self._swap_index, fn_kwargs={'index_map': index_map})

    def _augment_data(self):
        aug = load_dataset(path=self.path, name=self.config, split='train', revision=self.revision)
        aug = aug.train_test_split(test_size=0.6, seed=42)
        aug_alter = aug['test'].train_test_split(test_size=0.5, seed=42)
        aug_lower = aug_alter['train'].map(self._lower)
        aug_upper = aug_alter['test'].map(self._upper)
        self.aug_train = concatenate_datasets([aug['train'], aug_lower, aug_upper])
        self.lower_test = load_dataset(path=self.path, name=self.config, split='test', revision=self.revision).map(self._lower)
        self.upper_test = load_dataset(path=self.path, name=self.config, split='test', revision=self.revision).map(self._upper)