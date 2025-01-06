from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from torch import nn
import torch as pt
from torch.nn import TransformerEncoder
from torch.utils.data import StackDataset, ConcatDataset, TensorDataset, Dataset, DataLoader, Subset
from typing import Dict, Set, Tuple, List, Optional
import collections
import io
import itertools
import json
import numpy as np
import sklearn.model_selection
import torch
# import torchinfo
from .vector_db import EmbeddingDb, ClassificationItemId

Label = str


class ClassificationItemDataset(Dataset):
    def __init__(self, db: EmbeddingDb):
        self.db = db

    def __getitem__(self, item: ClassificationItemId) -> pt.Tensor:
        self.db.db.execute(
            '''
            SELECT tensor_id
            FROM classification_feature
            WHERE classification_item_id = ?
            ''',
            (item,)
        )
        tensor_ids = self.db.db.fetch_df()
        return self.db.fetch_tensors(tensor_ids['tensor_id'])

def get_query_items(db: EmbeddingDb, query: List[str]) -> Set[ClassificationItemId]:
    db.db.execute(
        '''
        SELECT classification_item_id
        FROM classification_item
        WHERE (metadata->>'$.query') in ?
        ''',
        (query,)
    )
    return db.db.fetch_df()['classification_item_id']

def get_classification_features(db: EmbeddingDb, query:str):

    db.db.execute(
        '''
select
  classification_item.classification_item_id,
  classification_feature.tensor_id,
  classification_item.metadata->>'$.query' as query,
  classification_item.metadata->>'$.passage' as passage,
  classification_feature.metadata->>'$.prompt_class' as prompt_class,
  classification_feature.metadata->>'$.test_bank' as test_bank_id
from classification_item
inner join classification_feature on classification_feature.classification_item_id = classification_item.classification_item_id
where classification_item.metadata->>'$.query' = ?
;
        ''',
        (query,)
    )
    return db.db.fetch_df()



def main():
    embedding_db = EmbeddingDb(Path("embedding_db_try2/exam_grading"))
    dataset = ClassificationItemDataset(db=embedding_db)

    ci_ids = get_query_items(embedding_db, ['118440', '121171'])

    features = get_classification_features(embedding_db, '118440')
    print(embedding_db.fetch_tensors(features['tensor_id']))

    tensors = ([dataset[i] for i in ci_ids])
    #print(item)

if __name__ == '__main__':
    main()
