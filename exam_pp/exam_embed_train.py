from pathlib import Path
import sys
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from torch import nn
import torch as pt
from torch.nn import TransformerEncoder
from torch.utils.data import StackDataset, ConcatDataset, TensorDataset, Dataset, DataLoader, Subset
from typing import Any, Dict, Set, Tuple, List, Optional
import collections
import io
import itertools
import json
import numpy as np
import sklearn.model_selection
import torch
import pandas as pd

from . test_bank_prompts import get_prompt_classes
from . import attention_classify

# import torchinfo
from .vector_db import Align, EmbeddingDb, ClassificationItemId

Label = str


class ClassificationItemDataset(Dataset):
    def __init__(self, db: EmbeddingDb, ):
        self.db = db

    def __getitem__(self, item: ClassificationItemId) -> pt.Tensor:
        self.db.db.execute(
            '''--sql
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
        '''--sql
        SELECT DISTINCT classification_item.metadata->>'$.query' as query
        FROM classification_item
        '''
    )
    return db.db.fetch_df()['query']



def get_query_items(db: EmbeddingDb, query: List[str]) -> Set[ClassificationItemId]:
    db.db.execute(
        '''--sql
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
    db.db.execute(
        '''--sql
        SELECT needles.i,
                   classification_item.metadata->>'$.query' as query,
                   classification_item.metadata->>'$.passage' as passage,
                   label_assignment.true_labels,
                   classification_item.classification_item_id as classification_item_id

        FROM classification_item
        INNER JOIN (SELECT * FROM classification_item_ids_df) AS needles ON classification_item.classification_item_id = needles.classification_item_id
        INNER JOIN label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
        ORDER BY needles.i ASC;
    ''')
    return db.db.df()


def get_classification_features(db: EmbeddingDb, query:str):

    db.db.execute(
        '''--sql
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
    db.db.execute(
        '''--sql
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

def balanced_training_data(embedding_db: EmbeddingDb) -> pd.DataFrame:
    embedding_db.db.execute(
        '''--sql
        SELECT classification_item.metadata->>'$.query' as query,
        classification_item.metadata->>'$.passage' as passage,
               label_assignment.true_labels as true_labels,
               classification_item.classification_item_id as classification_item_id
        FROM classification_item
        INNER JOIN label_assignment on label_assignment.classification_item_id = classification_item.classification_item_id
        WHERE len(label_assignment.true_labels) = 1
        '''
    )
    df = embedding_db.db.fetch_df()
    df['positive'] = df['true_labels'].apply(lambda x: '0' in x)
    by_label = df.groupby('positive')
    counts = by_label['classification_item_id'].count()
    n = counts.min()
    sampled = by_label.sample(n)
    sampled.drop(columns=['positive'], inplace=True)
    return sampled


def create_dataset(embedding_db:EmbeddingDb,  prompt_class:str )->Tuple[Dataset, Dataset, Set[int]]:
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

    queries = list(get_queries(db=embedding_db))[:5]

    
        
    classification_items_train:List[ClassificationItemId]
    classification_items_train =[ example for query in queries 
                                           for example in list(get_query_items(db=embedding_db, query=[query]))[::2] 
                                 ]
    classification_items_test:List[ClassificationItemId]
    classification_items_test = [ example for query in queries 
                                           for example in list(get_query_items(db=embedding_db, query=[query]))[1::2] 
                                 ]
    

        
    # print("test items", classification_items_test)


    def balance_majority_class(df):

        # Convert each array to a tuple (this makes it hashable)
        df['true_labels'] = df['true_labels'].apply(tuple)

        # Filter
        df_zero = df[df['true_labels'] == ('0',)]
        df_not_zero = df[df['true_labels'] != ('0',)]

        print("df_zero shape:", df_zero.shape)
        print(df_zero)

        print("df_not_zero shape:", df_not_zero.shape)
        print(df_not_zero)

        # Downsample to balance
        num_samples = min(len(df_zero), len(df_not_zero))
        print("num_samples", num_samples)

        df_not_zero_sampled = df_not_zero.sample(n=num_samples, random_state=42)

        balanced_df = pd.concat([df_zero, df_not_zero_sampled])
        print("\nBalanced df shape:", balanced_df.shape)
        return balanced_df

    # query, passage, labels
    classification_data_train:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_train)
    classification_data_test:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_test)

    classification_items_train = classification_data_train["classification_item_id"].to_list()
    classification_items_test = classification_data_test["classification_item_id"].to_list()

    train_tensors:pd.DataFrame = get_tensor_ids(embedding_db
                                                , classification_item_id=classification_items_train
                                                , prompt_class= prompt_class)
    test_tensors:pd.DataFrame = get_tensor_ids(embedding_db
                                               , classification_item_id=classification_items_test
                                               , prompt_class= prompt_class)
    # print("test_tensors", test_tensors)

    class ClassificationItemDataset(Dataset):
        def __init__(self, tensor_df:pd.DataFrame, db:EmbeddingDb):
            self.tensor_df = tensor_df
            self.db = db


        def __getitem__(self, index) -> pt.Tensor:
            tensor_ids = self.tensor_df.loc[index,"tensor_ids"]
            # print(f"{index}: {row.to_dict()}")
            tensor = self.db.fetch_tensors_single(tensor_ids=tensor_ids, token_length=384)

            # the real deal:
            # tensor = self.db.fetch_tensors(tensor_ids=tensor_ids, token_length=10)

            # print(tensor.shape)
            # print("-------")
            return tensor

        def __len__(self) -> int:
            return len(self.tensor_df)
        

        # def __getitems__(self, indices: List) -> List[T_co]:
        # Not implemented to prevent false-positives in fetcher check in
        # torch.utils.data._utils.fetch._MapDatasetFetcher

        def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
            return ConcatDataset([self, other])

    class EmbeddingStackDataset(torch.utils.data.Dataset):
        def __init__(self, embedding, label_one_hot, label_id):
            # Check that all datasets have the same first dimension
            if not (len(embedding) == label_one_hot.size(0) == label_id.size(0)):
                raise ValueError(f"Size mismatch between datasets:  ({len(embedding)} == {label_one_hot.size(0)} == {label_id.size(0)})")
            self.embedding = embedding
            self.label_one_hot = label_one_hot
            self.label_id = label_id

        def __len__(self):
            return len(self.embedding)

        def __getitem__(self, idx):
            return {
                "embedding": self.embedding[idx],
                "label_one_hot": self.label_one_hot[idx],
                "label_id": self.label_id[idx],
            }
        
        

    dataset_embedding_train = ClassificationItemDataset(db=embedding_db, tensor_df=train_tensors)
    dataset_embedding_test = ClassificationItemDataset(db=embedding_db, tensor_df=test_tensors)
    # print("len(dataset_embedding_test)", len(dataset_embedding_test))


    example_label_list_train = [d.true_labels[0] for d in  classification_data_train.itertuples() ]
    example_label_list_test = [d.true_labels[0] for d in  classification_data_test.itertuples() ]
    # print(example_label_list_test)

    classes = sorted(list(set(example_label_list_train)))
    label_idx = {c:i for i,c in enumerate(classes)}

    example_label_id_list_train = [label_idx[label] 
                                   for label in example_label_list_train]
    example_label_id_list_test = [label_idx[label] 
                                   if label_idx.get(label) is not None else label_idx['0'] # no training data for this label
                                   for label in example_label_list_test]

    example_label_id_train = torch.tensor(example_label_id_list_train, dtype=torch.long)  # [num_examples]
    example_label_id_test = torch.tensor(example_label_id_list_test, dtype=torch.long)  # [num_examples]
    # Generate one-hot encoding for labels
    example_label_one_hot_train = nn.functional.one_hot(example_label_id_train, num_classes=len(classes)).to(torch.float32)
    example_label_one_hot_test = nn.functional.one_hot(example_label_id_test, num_classes=len(classes)).to(torch.float32)

    print(f'train embedding={len(dataset_embedding_train)}, label_one_hot={example_label_one_hot_train.shape}, label_id={example_label_id_train.shape}')
    print(f'test embedding={len(dataset_embedding_test)}, label_one_hot={example_label_one_hot_test.shape}, label_id={example_label_id_test.shape}')
    print(f'train labels: example_label_id_train',example_label_id_train)
    print(f'test labels: example_label_id_train',example_label_id_test)

    train_ds = EmbeddingStackDataset(embedding=dataset_embedding_train
                                     , label_one_hot=example_label_one_hot_train
                                     , label_id=example_label_id_train)
    test_ds = EmbeddingStackDataset(embedding=dataset_embedding_test
                                    , label_one_hot=example_label_one_hot_test
                                    , label_id=example_label_id_test)


    # print("train_ds", train_ds[3])
    # print("test_ds", test_ds[3])

    return (train_ds, test_ds, classes)
    tensor_ids = lookup_classification_tensors(embedding_db, classification_items_train, prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    

def main(cmdargs=None) -> None:
    import argparse

    sys.stdout.reconfigure(line_buffering=True)


    print("EXAM embed train")
    desc = f'''EXAM Embed Train
             '''
    

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--embedding-db', type=str, metavar='PATH', help='Path for the database directory for recording embedding vectors')

    parser.add_argument('-o', '--root', type=str, metavar="FILE", help='Directory to write training output to', default=Path("./attention_classify"))
    parser.add_argument('--device', type=str, metavar="FILE", help='Device to run on, cuda:0 or cpu', default=Path("cuda:0"))
    parser.add_argument('--prompt-class', type=str, required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
 
    args = parser.parse_args(args = cmdargs) 
 
    root = Path(args.root)
    embedding_db = EmbeddingDb(Path(args.embedding_db))
    # embedding_db = EmbeddingDb(Path("embedding_db_classify/exam_grading"))
    (train_ds, test_ds, classes) = create_dataset(embedding_db, prompt_class=args.prompt_class)
    # x = balanced_training_data(embedding_db)
    # print(x)
    attention_classify.run(root
                           , model_type='multi_label_packed'
                           , train_ds=train_ds
                           , test_ds=test_ds
                           , classes=classes
                           , device_str=args.device)  

if __name__ == '__main__':
    main()
