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
import pandas as pd

# import torchinfo
from .vector_db import Align, EmbeddingDb, ClassificationItemId

Label = str


class ClassificationItemDataset(Dataset):
    def __init__(self, db: EmbeddingDb, ):
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


def get_queries(db:EmbeddingDb)-> Set[str]:
    db.db.execute(
        '''
        SELECT DISTINCT classification_item.metadata->>'$.query' as query
        FROM classification_item
        '''
    )
    return db.db.fetch_df()['query']



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

def lookup_queries_paragraphs_judgments(db:EmbeddingDb, classification_item_id: List[ClassificationItemId]):
    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index
    db.db.execute('''
        SELECT needles.i,
                   classification_item.metadata->>'$.query' as query,
                   classification_item.metadata->>'$.passage' as passage,
                   label_assignment.true_labels

        FROM classification_item
        INNER JOIN (SELECT * FROM classification_item_ids_df) AS needles ON classification_item.classification_item_id = needles.classification_item_id
        INNER JOIN label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
        ORDER BY needles.i ASC;
    ''')
    return db.db.df()


def get_classification_features(db: EmbeddingDb, query:str):

    db.db.execute(
        '''
select
  classification_item.classification_item_id,
  classification_feature.tensor_id,
  classification_item.metadata->>'$.query' as query,
  classification_item.metadata->>'$.passage' as passage,
  classification_feature.metadata->>'$.prompt_class' as prompt_class,
  classification_feature.metadata->>'$.test_bank' as test_bank_id,
  label_assignment.true_labels
from classification_item
inner join classification_feature on classification_feature.classification_item_id = classification_item.classification_item_id
inner join label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
where classification_item.metadata->>'$.query' = ?
;
        ''',
        (query,)
    )
    return db.db.fetch_df()


# def get_tensor_ids(db:EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class:str):
#     db.db.execute(
#         '''
# select tensor_id
# from classification_feature
# where classification_item_id in `classification_item_id` classification_item.metadata->>'$.query' = ?
# ;
#         ''',
#         (query,)
#     )
#     return db.db.fetch_df()

def get_tensor_ids(db:EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class:str):
    # classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    # classification_item_ids_df['i'] = classification_item_ids_df.index
    # db.db.execute('''
    #     SELECT needles.i,
    #            tensor_id,
    #            classification_feature.classification_item_id,
    #            classification_feature.metadata->>'$.test_bank' as test_bank_id
    #     FROM classification_feature
    #     INNER JOIN (SELECT * FROM classification_item_ids_df) AS needles ON classification_feature.classification_item_id = needles.classification_item_id
    #     WHERE classification_feature.metadata->>'$.prompt_class' = ?
    #     GROUP BY 
    #         needles.i, tensor_id, classification_feature.metadata->>'$.test_bank'
    #     ORDER BY needles.i ASC;
    # ''', (prompt_class,))
    # return db.db.df()


    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index
    db.db.execute('''
    SELECT 
        needles.i,
        STRING_AGG(tensor_id, ', ') AS tensor_ids,
        STRING_AGG(classification_feature.metadata->>'$.test_bank', ', ') AS test_bank_ids
    FROM classification_feature
    INNER JOIN 
        (SELECT * FROM classification_item_ids_df) AS needles 
    ON 
        classification_feature.classification_item_id = needles.classification_item_id
    WHERE 
        classification_feature.metadata->>'$.prompt_class' = ?
    GROUP BY 
        needles.i
    ORDER BY 
        needles.i ASC;
    ''', (prompt_class,))
    return db.db.df()


def main():
    embedding_db = EmbeddingDb(Path("embedding_db_try2/exam_grading"))
    dataset = ClassificationItemDataset(db=embedding_db)

    ci_ids = get_query_items(embedding_db, ['118440', '121171'])
    print("ci_ids", ci_ids)
    for cid in ci_ids:
        print("classification_item", cid)

    features = get_classification_features(embedding_db, '118440')

    # random_train_items = 

    for f in features[:10].itertuples():
        print("feature",f)
        print("query",f.query)
        tensor = embedding_db.fetch_tensors([f.tensor_id], token_length=10)
        print("tensor", tensor.shape, tensor)
        # f.


    # tensors = ([dataset[i] for i in ci_ids])
    # print(tensors)

    queries = get_queries(db=embedding_db)[:10]
    classification_items_train = [ example for query in queries 
                                           for example in get_query_items(db=embedding_db, query=[query])[:10] 
                                 ]
        
    print(classification_items_train)

    classification_data = lookup_queries_paragraphs_judgments(embedding_db, classification_items_train)
    print(classification_data)
    # for index, row in classification_data.iterrows():
    #     print(f"classification_data {index}: {row.to_dict()}")

    tensor_ids = get_tensor_ids(embedding_db, classification_items_train, prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    print(tensor_ids)

    # print("tensor_ids", embedding_db.fetch_tensors(features['tensor_id'], token_length=10, align=Align.ALIGN_END))
    # train_tensors = embedding_db.fetch_tensors(tensor_ids, token_length=10)



if __name__ == '__main__':
    main()
