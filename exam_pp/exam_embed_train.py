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
    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index
    db.db.execute('''
    SELECT 
        needles.i,
        array_agg(tensor_id) AS tensor_ids,
        array_agg(classification_feature.metadata->>'$.test_bank') AS test_bank_ids
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



def lookup_classification_tensors(db:EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class:str):
    # Get tensor metadata
    metadata_df = db.get_tensor_metadata(classification_item_id, prompt_class)

    # Fetch tensors and add them as a new column to the metadata DataFrame
    metadata_df = db.fetch_tensors_from_metadata(metadata_df, token_length=128, align=Align.ALIGN_BEGIN)

    # Access the computed tensors
    # for _, row in metadata_df.iterrows():
    #     print(f"Index: {row['i']}, Tensor: {row['pt_tensor']}")
    # return metadata_df

def main():
    embedding_db = EmbeddingDb(Path("embedding_db_try2/exam_grading"))
    # dataset = ClassificationItemDataset(db=embedding_db)

    # ci_ids = get_query_items(embedding_db, ['118440', '121171'])
    # print("ci_ids", ci_ids)
    # for cid in ci_ids:
    #     print("classification_item", cid)

    # features = get_classification_features(embedding_db, '118440')

    # # random_train_items = 

    # for f in features[:10].itertuples():
    #     print("feature",f)
    #     print("query",f.query)
    #     tensor = embedding_db.fetch_tensors([f.tensor_id], token_length=10)
    #     print("tensor", tensor.shape, tensor)
    #     # f.


    # tensors = ([dataset[i] for i in ci_ids])
    # print(tensors)

    queries = get_queries(db=embedding_db)[:10]
    classification_items_train = [ example for query in queries 
                                           for example in get_query_items(db=embedding_db, query=[query])[:10] 
                                 ]
        
    print("train items", classification_items_train)

    classification_data = lookup_queries_paragraphs_judgments(embedding_db, classification_items_train)
    print(classification_data)
    # for index, row in classification_data.iterrows():
    #     print(f"classification_data {index}: {row.to_dict()}")

    train_tensors:pd.DataFrame = get_tensor_ids(embedding_db, classification_item_id=classification_items_train, prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    
    class ClassificationItemDataset(Dataset):
        def __init__(self, train_tensor_df:pd.DataFrame, db:EmbeddingDb):
            self.tensor_df = train_tensor_df
            self.db = db


        def __getitem__(self, index) -> pt.Tensor:
            tensor_ids = train_tensors.loc[index,"tensor_ids"]
            # print(f"{index}: {row.to_dict()}")
            tensor = self.db.fetch_tensors(tensor_ids=tensor_ids, token_length=10)
            print(tensor.shape)
            print("-------")
            return tensor

        

        # def __getitems__(self, indices: List) -> List[T_co]:
        # Not implemented to prevent false-positives in fetcher check in
        # torch.utils.data._utils.fetch._MapDatasetFetcher

        def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
            return ConcatDataset([self, other])

        

    dataset = ClassificationItemDataset(db=embedding_db, train_tensor_df=train_tensors)

    print(dataset[3].shape)


    
    # print("train_tensors", train_tensors)
    # for tuple in train_tensors[:1].itertuples():
    #     # print(f"{index}: {row.to_dict()}")
    #     tensors = embedding_db.fetch_tensors(tensor_ids=tuple.tensor_ids, token_length=2)
    #     print(tuple.i, tensors.shape)
    #     print("-------")



    tensor_ids = lookup_classification_tensors(embedding_db, classification_items_train, prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    # print(tensor_ids[0:1])
    # for index, row in tensor_ids.iterrows():
        # print(f"{index}: {row.to_dict()}")

    # tensor = tensor_ids.iloc[0]['pt_tensor']     

    # print(tensor_ids.keys())

    # print("tensor_ids", embedding_db.fetch_tensors(features['tensor_id'], token_length=10, align=Align.ALIGN_END))
    # train_tensors = embedding_db.fetch_tensors(tensor_ids, token_length=10)



if __name__ == '__main__':
    main()
