from pathlib import Path
import sys
from enum import Enum, auto
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from torch import nn
import torch as pt
from torch.nn import TransformerEncoder
from torch.utils.data import StackDataset, ConcatDataset, TensorDataset, Dataset, DataLoader, Subset
from typing import Any, Dict, Set, Tuple, List, Optional

import torch.profiler as ptp
import collections
import io
import itertools
import json
import numpy as np
import sklearn.model_selection
import torch
import pandas as pd
import time

from . multi_seq_class_grade_model import ProblemType

from .rubric_db import *

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



def annotate_with_grades(embedding_db:EmbeddingDb, data_tensors:pd.DataFrame) -> pd.DataFrame:
    embedding_db.db.execute(
        '''--sql
        SELECT needles.tensor_id, exam_grade.self_rating, exam_grade.test_bank_id
        FROM (SELECT of data_tensors.tensor_id FROM data_tensors) AS needles
        INNER JOIN classification_feature AS cf
        ON cf.tensor_id = needles.tensor_id
        INNER JOIN classification_item AS ci
        ON ci.classification_item_id = cf.classification_item_id
        INNER JOIN rubric.relevance_item AS ri
        ON ri.query_id = ci.metadata->>'$.query'
            AND ri.paragraph_id = ci.metadata->>'$.paragraph'
        INNER JOIN exam_grade
        ON exam_grade.relevance_item_id = ri.relevance_item_id
            AND exam_grade.test_bank_id = cf.metadata->>'$.test_bank'
        '''
    )
    grade_df = embedding_db.db.fetch_df()
    return data_tensors.join(grade_df, on="tensor_id")




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


# def annotate_with_grades(embedding_db:EmbeddingDb, data_tensors:pd.DataFrame) -> pd.DataFrame:
#     embedding_db.db.execute(
#         '''--sql
#         SELECT needles.tensor_id, exam_grade.self_rating, exam_grade.test_bank_id
#         FROM (SELECT of data_tensors.tensor_id FROM data_tensors) AS needles
#         INNER JOIN classification_feature AS cf
#         ON cf.tensor_id = needles.tensor_id
#         INNER JOIN classification_item AS ci
#         ON ci.classification_item_id = cf.classification_item_id
#         INNER JOIN relevance_item AS ri
#         ON ri.query_id = ci.metadata->>'$.query'
#             AND ri.paragraph_id = ci.metadata->>'$.paragraph'
#         INNER JOIN exam_grade
#         ON exam_grade.relevance_item_id = ri.relevance_item_id
#             AND exam_grade.test_bank_id = cf.metadata->>'$.test_bank'
#         '''
#     )
#     grade_df = embedding_db.db.fetch_df()
#     return data_tensors.join(grade_df, on="tensor_id")

def get_tensor_ids_with_grades(db: EmbeddingDb, classification_item_id: List[ClassificationItemId], prompt_class: str):
    """
    For each classification_item_id, fetch:
      - array of tensor_ids
      - array of test_bank_ids
      - array of exam_grade fields (e.g., self_rating, is_correct)

        # df_with_grades columns might look like:
        # i | tensor_ids         | test_bank_ids       | self_ratings        | is_corrects
        # --|--------------------|---------------------|---------------------|-------------
        # 0 | [t1, t2, ...]      | ['tb1', 'tb1', ...] | [2, 3, ...]         | [False, True, ...]
        # 1 | [t17, t18, ...]    | ['tb2', ...]        | [1, 4, ...]         | [True, False, ...]
        # ...

    """    

    classification_item_ids_df = pd.DataFrame(data={'classification_item_id': classification_item_id})
    classification_item_ids_df['i'] = classification_item_ids_df.index

    # Register the DataFrame as a temporary table for use in SQL
    db.db.register('classification_item_ids_df', classification_item_ids_df)

    # Execute the query
    db.db.execute(
        '''--sql
        SELECT 
            needles.i,
            array_agg(cf.tensor_id) AS tensor_ids,
            array_agg(cf.metadata->>'$.test_bank') AS test_bank_ids,
            array_agg(eg.self_rating) AS self_ratings,
            array_agg(eg.is_correct) AS correctness,
            la.true_labels AS judgment,
            ci.classification_item_id
        FROM classification_feature AS cf
        INNER JOIN classification_item_ids_df AS needles 
            ON cf.classification_item_id = needles.classification_item_id
        INNER JOIN classification_item AS ci
            ON ci.classification_item_id = cf.classification_item_id
        INNER JOIN rubric.relevance_item AS ri
            ON ri.query_id = (ci.metadata->>'$.query')
            AND ri.paragraph_id = (ci.metadata->>'$.passage')
        INNER JOIN rubric.exam_grade AS eg
            ON eg.relevance_item_id = ri.relevance_item_id
            AND eg.test_bank_id = (cf.metadata->>'$.test_bank')
        INNER JOIN label_assignment AS la
            ON la.classification_item_id = ci.classification_item_id
        WHERE 
            (cf.metadata->>'$.prompt_class') = ?
        GROUP BY 
            needles.i, ci.classification_item_id, judgment
        ORDER BY 
            needles.i ASC;
        ''', (prompt_class,))
    
    return db.db.df()



class SequenceMode(Enum):
    single_sequence = auto()
    concat_sequence = auto()
    multi_sequence = auto()

    @staticmethod
    def from_string(arg:str):
        try:
            return SequenceMode[arg]
        except KeyError:
            import argparse   
            raise argparse.ArgumentTypeError("Invalid ClassificationModel choice: %s" % arg)
        

class ClassificationItemDataset(Dataset):
    def __init__(self, tensor_df:pd.DataFrame, db:EmbeddingDb, sequence_mode:SequenceMode, max_token_len:int):
        self.tensor_df = tensor_df
        self.db = db
        self.sequence_mode = sequence_mode
        self.max_token_len = max_token_len


    def __getitem__(self, index) -> pt.Tensor:
        tensor_ids = self.tensor_df.loc[index,"tensor_ids"]
        tensor=None
        if self.sequence_mode == SequenceMode.single_sequence:
            tensor = self.db.fetch_tensors_single(tensor_ids=tensor_ids, token_length=self.max_token_len)
        elif self.sequence_mode == SequenceMode.multi_sequence:
                        # self.fetch_tensors(tensor_ids=tensor_ids, token_length=token_length, align=align)
            tensor = self.db.fetch_tensors(tensor_ids=tensor_ids, token_length=self.max_token_len)
        elif self.sequence_mode == SequenceMode.concat_sequence:
            tensor = self.db.fetch_tensors_concat(tensor_ids=tensor_ids, token_length=self.max_token_len)
        else:
            raise RuntimeError(f"sequence mode {self.sequence_mode} is not defined.")

        return tensor

    def __len__(self) -> int:
        return len(self.tensor_df)
    
    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])

class EmbeddingStackDataset(torch.utils.data.Dataset):
    def __init__(self, embedding, label_one_hot, label_id, grades_one_hot, grades_id):
        # Check that all datasets have the same first dimension
        if not (len(embedding) == label_one_hot.size(0) == label_id.size(0)):
            raise ValueError(f"Size mismatch between datasets:  ({len(embedding)} == {label_one_hot.size(0)} == {label_id.size(0)})")
        self.embedding = embedding
        self.label_one_hot = label_one_hot
        self.label_id = label_id
        self.grades_one_hot = grades_one_hot
        self.grades_id = grades_id

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return {
            "embedding": self.embedding[idx],
            "label_one_hot": self.label_one_hot[idx],
            "label_id": self.label_id[idx],
            "grades_one_hot": self.grades_one_hot[idx],
            "grades_id":self.grades_id[idx]
        }
    

def conv_class(example_label_list:list[Any], classes:list[int], label_idx:Dict[Any,int]):
        def default_label(label, d):
            l = label_idx.get(label)
            if l is None:
                print(f"Warning: Dataset contains label {label}, which is not in the set of training labels: {label_idx.keys()}")
                return d
            return l
        
        example_label_id_list = [default_label(label,0)
                                        for label in example_label_list]

        example_label_id = torch.tensor(example_label_id_list, dtype=torch.long)  # [num_examples]
        # Generate one-hot encoding for labels
        example_label_one_hot = nn.functional.one_hot(example_label_id, num_classes=len(classes)).to(torch.float32)

        # print(f'label_one_hot={example_label_one_hot.shape}, label_id={example_label_id.shape}')
        # print(f'labels: example_label_id',example_label_id)
        return example_label_one_hot, example_label_id



def conv_grades(example_grades_list:list[list[Any]], grades:list[int], grade_idx:Dict[Any,int]):
        def default_grade(grade, d):
            g = grade_idx.get(grade)
            if g is None:
                print(f"Warning: Dataset contains grade {grade}, which is not in the set of training grades: {grade_idx.keys()}")
                return d
            return g
        
        # print("example_grades_list", example_grades_list)
        
        example_label_id_list = [ [ default_grade(grade,1) for grade in grade_list]
                                        for grade_list in example_grades_list]

        example_label_id = torch.tensor(example_label_id_list, dtype=torch.long)  # [num_examples, num_seq]
        # Generate one-hot encoding for labels
        example_label_one_hot = nn.functional.one_hot(example_label_id, num_classes=len(grades)).to(torch.float32)

        # print(f'grade_one_hot={example_label_one_hot.shape}, label_id={example_label_id.shape}')
        # print(f'grades: example_grades_id',example_label_id)
        return example_label_one_hot, example_label_id


def create_dataset(embedding_db:EmbeddingDb
                   , prompt_class:str
                   , query_list: Optional[List[str]]
                   , max_queries:Optional[int]
                   , max_paragraphs:Optional[int]
                   , max_token_len:Optional[int]
                   , sequence_mode:SequenceMode
                   , split_same_query: bool = False
                   )->Tuple[Dataset, Dataset, List[int], List[int]]:

    queries = None
    if query_list:
        # print("queries_list", query_list)
        queries_fromdb:Set[str] = get_queries(db=embedding_db)
        queries = queries_fromdb[queries_fromdb.isin(query_list)]
    else:
        queries = list(get_queries(db=embedding_db))[:max_queries]

        
    classification_items_train:List[ClassificationItemId]
    classification_items_test:List[ClassificationItemId]

    if split_same_query:
        classification_items_train =[ example for query in queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[:max_paragraphs:2] 
                                    ]
        classification_items_test = [ example for query in queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[1:max_paragraphs:2] 
                                    ]
        print(f"queries: {list(queries)}")
    else:
        train_queries = queries[::2]
        test_queries = queries[1::2]
        classification_items_train =[ example for query in train_queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[:max_paragraphs] 
                                    ]
        classification_items_test = [ example for query in test_queries 
                                            for example in list(get_query_items(db=embedding_db, query=[query]))[:max_paragraphs] 
                                    ]

        print(f"train_queries: {list(train_queries)}")
        print(f"test_queries: {list(test_queries)}")

        


    # query, passage, labels
    classification_data_train:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_train)
    classification_data_test:pd.DataFrame = lookup_queries_paragraphs_judgments(embedding_db, classification_items_test)

    classification_items_train = classification_data_train["classification_item_id"].to_list()
    classification_items_test = classification_data_test["classification_item_id"].to_list()


    train_tensors_with_grades_labels:pd.DataFrame = get_tensor_ids_with_grades(embedding_db
                                                , classification_item_id=classification_items_train
                                                , prompt_class= prompt_class)
    test_tensors_with_grades_labels:pd.DataFrame = get_tensor_ids_with_grades(embedding_db
                                               , classification_item_id=classification_items_test
                                               , prompt_class= prompt_class)
    # print("test_tensors", test_tensors)

    # train_tensors_with_grade:pd.DataFrame = annotate_with_grades(embedding_db=embedding_db, data_tensors=train_tensors)
    # test_tensors_with_grade:pd.DataFrame = annotate_with_grades(embedding_db=embedding_db, data_tensors=test_tensors)

    # print("train_tensors_with_grade", train_tensors)
    # print("len(dataset_embedding_test)", len(dataset_embedding_test))


    example_label_list_train = [d.true_labels[0] for d in  classification_data_train.itertuples() ]
    example_label_list_test = [d.true_labels[0] for d in  classification_data_test.itertuples() ]
    classes = sorted(list(set(example_label_list_train)))
    label_idx = {c:i for i,c in enumerate(classes)}

    # fake some grade info
    # print("classification_data_train", classification_data_train[0:2])
    # print("train_tensors_with_grades", train_tensors_with_grades[0:2])
    # print("dataset_embedding_train", dataset_embedding_train[0:2])
    
    # print("train_tensors_with_grades_labels", train_tensors_with_grades_labels)


    dataset_embedding_train = ClassificationItemDataset(db=embedding_db, tensor_df=train_tensors_with_grades_labels, sequence_mode=sequence_mode, max_token_len=max_token_len)
    dataset_embedding_test = ClassificationItemDataset(db=embedding_db, tensor_df=test_tensors_with_grades_labels, sequence_mode=sequence_mode, max_token_len=max_token_len)

    def filtered_grade(self_rating:int, judgment:List[str])->int:
        label = int(judgment[0])
        if label <= 0:
            return 0
        elif self_rating>=4:
            return self_rating
        else:
            return -1
        
    def plain_grade(self_rating:int, judgment: List[str])->int:
        return self_rating

    example_grades_list_train = [[plain_grade(r, tup.judgment) 
                                    for r in tup.self_ratings] 
                                        for tup in train_tensors_with_grades_labels.itertuples()]
    example_grades_list_test = [[plain_grade(r, tup.judgment)  
                                    for r in tup.self_ratings] 
                                    for tup in test_tensors_with_grades_labels.itertuples()]
    # example_grades_list_test = [tup.self_ratings for tup in test_tensors_with_grades_labels.itertuples()]

    grade_idx = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5} # Adding -1 as "Missing"
    grades = list(grade_idx.values())
    sorted(grades)
    # print("example_grades_list_train", example_grades_list_train)
    # print("grades", grades)

  

    # print(example_label_list_test)


    # example_label_id_list_train = [label_idx[label] 
    #                                for label in example_label_list_train]
    # example_label_id_list_test = [label_idx[label] 
    #                                if label_idx.get(label) is not None else label_idx['0'] # no training data for this label
    #                                for label in example_label_list_test]

    # example_label_id_train = torch.tensor(example_label_id_list_train, dtype=torch.long)  # [num_examples]
    # example_label_id_test = torch.tensor(example_label_id_list_test, dtype=torch.long)  # [num_examples]
    # # Generate one-hot encoding for labels
    # example_label_one_hot_train = nn.functional.one_hot(example_label_id_train, num_classes=len(classes)).to(torch.float32)
    # example_label_one_hot_test = nn.functional.one_hot(example_label_id_test, num_classes=len(classes)).to(torch.float32)

    # print(f'train embedding={len(dataset_embedding_train)}, label_one_hot={example_label_one_hot_train.shape}, label_id={example_label_id_train.shape}')
    # print(f'test embedding={len(dataset_embedding_test)}, label_one_hot={example_label_one_hot_test.shape}, label_id={example_label_id_test.shape}')
    # print(f'train labels: example_label_id_train',example_label_id_train)
    # print(f'test labels: example_label_id_train',example_label_id_test)
    example_label_one_hot_train, example_label_id_train = conv_class(example_label_list_train, classes=classes, label_idx=label_idx)
    example_label_one_hot_test, example_label_id_test = conv_class(example_label_list_test, classes=classes, label_idx=label_idx)
    example_grades_one_hot_train, example_grades_id_train = conv_grades(example_grades_list_train, grades=grades, grade_idx=grade_idx)
    example_grades_one_hot_test, example_grades_id_test = conv_grades(example_grades_list_test, grades=grades, grade_idx=grade_idx)

    
    train_ds = EmbeddingStackDataset(embedding=dataset_embedding_train
                                     , label_one_hot=example_label_one_hot_train
                                     , label_id=example_label_id_train
                                     , grades_one_hot=example_grades_one_hot_train
                                     , grades_id=example_grades_id_train)
    test_ds = EmbeddingStackDataset(embedding=dataset_embedding_test
                                    , label_one_hot=example_label_one_hot_test
                                    , label_id=example_label_id_test
                                    , grades_one_hot=example_grades_one_hot_test
                                    , grades_id=example_grades_id_test)

    # print("train_ds", train_ds[3])
    # print("test_ds", test_ds[3])

    return (train_ds, test_ds, [label_idx[c] for c in classes], [grade_idx[g] for g in grades])
    tensor_ids = lookup_classification_tensors(embedding_db, classification_items_train, prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    
class CachingDataset(Dataset):
    def __init__(self, cachee:Dataset):
        self.data:Dict[int,Any] = dict()
        self.cachee:Dataset = cachee
        self.length:int = len(cachee)

    def __getitem__(self, index):
        item = self.data.get(index)
        if item is None:
            item = self.cachee[index]
            self.data[index]=item
        return item

    def __len__(self):
        return self.length

class PreloadedDataset(Dataset):
    def __init__(self, cachee:Dataset):
        self.data = []
        for i in range(0,len(cachee)):
            data_item = cachee[i]
            self.data.append(data_item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class TrainingTimer:
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        print(f"{self.message}: {TrainingTimer.elapsed_time_str(self.end_time-self.start_time)}")

    @staticmethod
    def elapsed_time_str(elapsed_time:int)-> str:
        days = int(elapsed_time // (24 * 3600))
        hours = int((elapsed_time % (24 * 3600)) // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


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
    parser.add_argument('--rubric-db', type=str, metavar='PATH', help='Path for the database directory for rubric paragraph data')

    parser.add_argument('-o', '--root', type=str, metavar="FILE", help='Directory to write training output to', default=Path("./attention_classify"))
    parser.add_argument('--device', type=str, metavar="FILE", help='Device to run on, cuda:0 or cpu', default=Path("cuda:0"))
    parser.add_argument('--epochs', type=int, metavar="T", help="How many epochs to run training for", default=30)
    parser.add_argument('--batch-size', type=int, metavar="S", help="Batchsize for training", default=128)

    parser.add_argument('--snapshots-every', type=int, metavar="T", help="Take a model shapshort every T epochs")
    parser.add_argument('--snapshots-best-after', type=int, metavar="T", help="Take a model shapshort when target metric is improved (but only after the T'th epoch, and only during evaluation epochs)")
    parser.add_argument('--snapshots-target-metric', type=str, default="roc_auc", metavar="METRIC", help="Target evaluation metric for --snapshots-best-after, such as 'roc_auc'")
    parser.add_argument('--eval-every', type=int, metavar="T", help="Take a model shapshort every T epochs", default=1)

    parser.add_argument('--inner-dim', type=int, metavar="DIM", help="Use DIM as hidden dimension", default=64)
    parser.add_argument('--nhead', type=int, metavar="N", help="Use transformer with N heads", default=1)

    parser.add_argument('--label-problem-type', type=ProblemType.from_string, required=True, choices=list(ProblemType), metavar="MODEL"
                        , help="The classification problem to use for label prediction. Choices: "+", ".join(list(x.name for x in ProblemType)))
    parser.add_argument('--grade-problem-type', type=ProblemType.from_string, required=True, choices=list(ProblemType), metavar="MODEL"
                        , help="The classification problem to use for grade prediction. Choices: "+", ".join(list(x.name for x in ProblemType)))
    parser.add_argument('--no-transformers', dest="use_transformers", action="store_false", help='If set, replaces the transformer layer with an mean pooling', default=True)

    parser.add_argument('--prompt-class', type=str, required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--class-model', type=attention_classify.ClassificationModel.from_string, required=True, choices=list(attention_classify.ClassificationModel), metavar="MODEL"
                        , help="The classification model to use. Choices: "+", ".join(list(x.name for x in attention_classify.ClassificationModel)))
    parser.add_argument('--overwrite', action="store_true", help='will automatically replace the output directory')
    parser.add_argument('--dry-run', action="store_true", help='will automatically replace the output directory')

    parser.add_argument('--max-queries', type=int, metavar="N", help="Use up to N queries")
    parser.add_argument('--queries', type=str, metavar="query_id", nargs='+', help="Use queries with these IDS")
    parser.add_argument('--max-paragraphs', type=int, metavar="N", help="Use to a total of N paragraphs across train and test")
    parser.add_argument('--split-same-query',action="store_true", help='Train/test split on same queries, but different paragraphs.')

    parser.add_argument('--max-token-len', type=int, metavar="N", help="Use up to N embedding tokens")
    parser.add_argument('--sequence-mode',  type=SequenceMode.from_string, required=True, choices=list(SequenceMode), metavar="MODE", help=f'Select how to handle multiple sequences for classification. Choices: {list(SequenceMode)}')
    parser.add_argument('--caching', action="store_true", help='Dataset: build in-memory cache as needed')
    parser.add_argument('--preloaded', action="store_true", help='Dataset: preload into memory')



    
 
    args = parser.parse_args(args = cmdargs) 

    root = Path(args.root)
    train_ds = None
    test_ds = None
    class_list = None

    with TrainingTimer("Data Loading"):

        embedding_db = EmbeddingDb(Path(args.embedding_db))

        embedding_db.db.execute(f'''--sql
                                ATTACH '{args.rubric_db}' as rubric (READ_ONLY)
                                ''')

        # embedding_db = EmbeddingDb(Path("embedding_db_classify/exam_grading"))
        (train_ds, test_ds, class_list, grades_list) = create_dataset(embedding_db
                                                         , prompt_class=args.prompt_class
                                                         , query_list = args.queries
                                                         , max_queries=args.max_queries
                                                         , max_paragraphs=args.max_paragraphs
                                                         , max_token_len=args.max_token_len
                                                         , sequence_mode=args.sequence_mode
                                                         , split_same_query=args.split_same_query
                                                         )

        if args.caching:
            train_ds = CachingDataset(train_ds)
            test_ds = CachingDataset(test_ds)
        elif args.preloaded:
            train_ds = PreloadedDataset(train_ds)
            test_ds = PreloadedDataset(test_ds)


        print(f"Device: {args.device}")


        print(f"Train data: {len(train_ds)}")
        print(f"Test data: {len(test_ds)}")


    # print(f"Data loading took {elapsed_time_str(end_time - start_time)}.")

    with TrainingTimer("Training"):
        # with ptp.profile(activities=[ptp.ProfilerActivity.CPU, ptp.ProfilerActivity.CUDA], with_stack=True) as prof:
        # attention_classify.run(root = root
        #                 , overwrite=args.overwrite
        #                 , model_type=args.class_model
        #                 , train_ds=train_ds
        #                 , test_ds=test_ds
        #                 , class_list=class_list
        #                 , batch_size=args.batch_size
        #                 , snapshot_every=args.snapshots_every
        #                 , eval_every=args.eval_every
        #                 , n_epochs=args.epochs
        #                 , device_str=args.device
        #                 , inner_dim=args.inner_dim
        #                 , nhead=args.nhead
        #                 , epoch_timer = TrainingTimer("Epoch")
        #                 , snapshot_best_after= args.snapshots_best_after
        #                 , target_metric= args.snapshots_target_metric
        #                 )

        attention_classify.run_num_seqs(root = root
                        , overwrite=args.overwrite
                        , model_type=args.class_model
                        , train_ds=train_ds
                        , test_ds=test_ds
                        , class_list=class_list
                        , grades_list=grades_list
                        , batch_size=args.batch_size
                        , snapshot_every=args.snapshots_every
                        , eval_every=args.eval_every
                        , n_epochs=args.epochs
                        , device_str=args.device
                        , inner_dim=args.inner_dim
                        , nhead=args.nhead
                        , epoch_timer = TrainingTimer("Epoch")
                        , snapshot_best_after= args.snapshots_best_after
                        , target_metric= args.snapshots_target_metric
                        , label_problem_type=args.label_problem_type
                        , grade_problem_type=args.grade_problem_type
                        , use_transformer=args.use_transformers
                        )


        # prof.export_chrome_trace('profile.json')

if __name__ == '__main__':
    main()
