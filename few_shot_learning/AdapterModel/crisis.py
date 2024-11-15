import os
import csv
import random

import datasets
from sklearn.model_selection import train_test_split
from datasets.utils import logging

logger = logging.get_logger(__name__)

_CITATION = """
"""

_DESCRIPTION = """
"""

_DATA_URL = None


class AMI18(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{type}-{form}'[:-1], version=datasets.Version("1.0.0"))
        for type in ['8', '9', '13',]
        for form in ['', 'contrastive-']
    ]

    def _info(self):
        type = self.config.name.split('-')[0]
        is_contrastive = True if 'contrastive' in self.config.name else False
        type_labels = {'8': ['informing statement', 'challenge', 'request', 'rejection',
                             'appreciation', 'acceptance', 'question', 'apology', ''
                             ],
                       '9': ['informing statement', 'challenge', 'accusation', 'request',
                             'appreciation', 'rejection', 'acceptance', 'question', 'apology', ''
                             ],
                       '13': ['informing statement', 'challenge', 'accusation', 'appreciation',
                              'request', 'evaluation', 'rejection', 'question', 'acceptance',
                              'proposal', 'denial', 'admission', 'apology', '']
                       }
        if not is_contrastive:
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=type_labels.get(type, '')
                    ),
                    'second-label': datasets.features.ClassLabel(
                        names=type_labels.get(type, '')
                    ),
                    'third-label': datasets.features.ClassLabel(
                        names=type_labels.get(type, '')
                    ),
                }
            )
        else:
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=type_labels.get(type, '')
                    ),
                    'match': datasets.features.ClassLabel(
                        names=[
                            'no',
                            'yes',
                        ]
                    ),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = os.getenv("CRISIS_URL")
        assert dl_dir
        # dl_dir = dl_manager.download_and_extract(dl_dir)
        type = self.config.name.split('-')[0]
        if type == '8':
            train_filename = 'train_dev_8classes_2.csv'
            eval_filename = 'test_8classes_2.csv'
            test_filename = 'test_8classes_2.csv'
        elif type == '9':
            train_filename = 'train_dev_9classes_2.csv'
            eval_filename = 'test_9classes_2.csv'
            test_filename = 'test_9classes_2.csv'
        else:
            train_filename = 'train_dev_13classes.csv'
            eval_filename = 'test_13classes.csv'
            test_filename = 'test_13classes.csv'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        train_filename
                    ),
                    'split': 'train'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        eval_filename
                    ),
                    'split': 'validation'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        test_filename
                    ),
                    'split': 'test'
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        data = list()
        contrastive_data = dict()
        type = self.config.name.split('-')[0]
        is_contrastive = True if 'contrastive' in self.config.name else False
        type_labels = {'8': ['informing statement', 'challenge', 'request', 'rejection',
                             'appreciation', 'acceptance', 'question', 'apology', ''
                             ],
                       '9': ['informing statement', 'challenge', 'accusation', 'request',
                             'appreciation', 'rejection', 'acceptance', 'question', 'apology', ''
                             ],
                       '13': ['informing statement', 'challenge', 'accusation', 'appreciation',
                              'request', 'evaluation', 'rejection', 'question', 'acceptance',
                              'proposal', 'denial', 'admission', 'apology', '']
                       }

        with open(filepath, encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )


            next(csv_reader)

            for row in csv_reader:
                if len(row) == 6:
                    row = row[1:]
                id, text, label, second_label, third_label = row
                if (second_label != '' and second_label not in type_labels[type])\
                    or (third_label != '' and third_label not in type_labels):
                    continue

                text = text.replace('\n', ' ')

                data.append({
                    'id': id,
                    'text': text,
                    'label': label,
                    'second-label': second_label,
                    'third-label': third_label,
                })
                if is_contrastive:
                    d = list()
                    for i, l in enumerate(type_labels[type]):
                        d.append({
                            'id': id,
                            'text': text,
                            'label': l,
                            'match': 'yes' if label == l else 'no'
                        })
                    contrastive_data[id] = d


        if is_contrastive:
            data = [j for i in data for j in contrastive_data[i['id']]]
            random.seed(0)
            random.shuffle(data)

        for idx, item in enumerate(data):
            yield idx, item
