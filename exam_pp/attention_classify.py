import argparse
from contextlib import AbstractContextManager
from enum import Enum, auto
from pathlib import Path
import sys
import typing
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from torch import nn
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
# import torchinfo

Label = str


class ElemAt(nn.Module):
    def __init__(self, idx: int=0):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        # x shape = (batch_sz, seq_len, llm_dim)
        # output (batch_sz, llm_dim)
        return x[:,self.idx,:]


class Slice(nn.Module):
    def __init__(self, s: slice):
        super().__init__()
        self.s = s

    def forward(self, x):
        # x shape = (batch_sz, seq_len, llm_dim)
        # output (batch_sz, slice_len, llm_dim)
        return x[:,self.s,:]


class Proj(nn.Module):
    '''Apply a linear projection to each token'''
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # project each token with the same linear function
        self.pos_linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        # self.class_heads = nn.ModuleList([nn.Linear(llm_dim, 1) for _ in range(n_classes)])

    def forward(self, tok_seq):
        """
        tok_seq: (batch_sz, tok_len, llm_dim)
            - Each token corresponds to a specific class.
        Returns:
        logits: (batch_sz, tok_len)
            - Projection for each token.
        """

        projs = torch.cat(
            [self.pos_linear(tok_seq[:, i, :]) for i, _elem in enumerate(tok_seq)], dim=1
        )  # (batch_sz, n_classes)
        return projs

class ClassHead(nn.Module):
    def __init__(self, n_classes: int, llm_dim: int):
        super().__init__()
        # One binary classifier per class
        self.class_heads = nn.ModuleList([nn.Linear(llm_dim, 1) for _ in range(n_classes)])

    def forward(self, class_tokens):
        """
        class_tokens: (batch_sz, n_classes, llm_dim)
            - Each token corresponds to a specific class.
        Returns:
        logits: (batch_sz, n_classes)
            - Logits for each class.
        """
        # Apply each head to its corresponding token
        logits = torch.cat(
            [head(class_tokens[:, i, :]) for i, head in enumerate(self.class_heads)], dim=1
        )  # (batch_sz, n_classes)
        return logits


class PackedClassHead(nn.Module):
    def __init__(self, llm_dim: int, n_classes: int):
        super().__init__()
        # Single linear layer with output size equal to the number of classes
        self.class_head = nn.Linear(llm_dim, n_classes)

    def forward(self, class_tokens):
        """
        class_tokens: (batch_sz, n_classes, llm_dim)
            - Each token corresponds to a specific class.
        Returns:
        logits: (batch_sz, n_classes)
            - Logits for each class.
        """
        # Apply the shared linear layer
        logits = self.class_head(class_tokens).squeeze(-1)  # (batch_sz, n_classes)
        return logits


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Vmap(nn.Module):
    def __init__(self, module: nn.Module, in_dims: List[int], randomness: str):
        super().__init__()
        self.add_module("module", module)
        self.randomness = randomness
        self.in_dims = in_dims
        self.module = module

    def forward(self, x):
        return torch.vmap(self.module, in_dims=self.in_dims, randomness=self.randomness)(x)


def mlp(layer_dims: List[int]) -> nn.Module:
    """
    input:  (batch_sz, n)
    output: (batch_sz, layer_dims[-1])
    """
    layers = [
            l
            for dim in layer_dims
            for l in [nn.LazyLinear(dim), nn.ReLU()]
            ]
    return nn.Sequential(*layers)

def build_model_multi_class_classifier_mlp(
        layer_dims: List[int],
        n_classes: int,
        llm_dim: int,
        ):

    layers = []

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, seq_len, llm_dim)
    layers += [Vmap(mlp(layer_dims), in_dims=2, randomness='same')]

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, seq_len*llm_dim)
    layers += [nn.Flatten(1,2)]

    # output layer
    # input:  (batch_sz, layer_dims[-1])
    # output: (batch_sz, n_classes)
    layers += [nn.LazyLinear(n_classes)]

    model = nn.Sequential(*layers)
    loss_fn = nn.CrossEntropyLoss()
    return (model, loss_fn)


def build_model_multi_class_classifier(
        n_classes: int,
        llm_dim: int,
        ff_dim: Optional[int]=None,
        nhead: int=1,
        ):
    ff_dim = ff_dim or 4*llm_dim

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, seq_len, llm_dim)
    transformer = nn.TransformerEncoderLayer(
        d_model=llm_dim,
        nhead=nhead,
        dim_feedforward=ff_dim,
        dropout=0.1,
        batch_first=True,
    )

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, llm_dim)
    cls_token = ElemAt(0)

    # input:  (batch_sz, llm_dim)
    # output: (batch_sz, n_classes)
    linear = nn.Linear(llm_dim, n_classes)

    # model:
    #   input:  (batch_sz, llm_dim)
    #   output: (batch_sz, n_classes)
    model = nn.Sequential(transformer, cls_token, linear)

    loss_fn = nn.CrossEntropyLoss()

    return (model, loss_fn)


def build_model_multi_label_embedding_classifier(
        n_classes: int,
        class_weights:torch.Tensor,
        llm_dim: int,
        ff_dim: Optional[int]=None,
        nhead: int=1,
        ):
    ff_dim = ff_dim or 4*llm_dim

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, seq_len, llm_dim)
    transformer = nn.TransformerEncoderLayer(
        d_model=llm_dim,
        nhead=nhead,
        dim_feedforward=ff_dim,
        dropout=0.1,
        batch_first=True,
    )

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, n_classes, llm_dim)
    cls_tokens = Slice(slice(0, n_classes))

    #   input: (batch_size, n_classes, llm_dim)
    #   output: (batch_size, n_classes)
    class_heads = ClassHead(n_classes=n_classes, llm_dim=llm_dim)

    # model:
    #   input:  (batch_sz, llm_dim)
    #   output: (batch_sz, n_classes)
    model = nn.Sequential(transformer, cls_tokens, class_heads)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    return (model, loss_fn)


def build_model_multi_label_embedding_classifier_packed(
        n_classes: int,
        class_weights:torch.Tensor,
        llm_dim: int,
        ff_dim: Optional[int]=None,
        nhead: int=1,
        ):
    ff_dim = ff_dim or 4*llm_dim

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, seq_len, llm_dim)
    transformer = nn.TransformerEncoderLayer(
        d_model=llm_dim,
        nhead=nhead,
        dim_feedforward=ff_dim,
        dropout=0.1,
        batch_first=True,
    )

    # input:  (batch_sz, seq_len, llm_dim)
    # output: (batch_sz, llm_dim)
    cls_tokens = ElemAt(0)

    #   input: (batch_size, n_classes, llm_dim)
    #   output: (batch_size, n_classes)
    class_heads = PackedClassHead(n_classes=n_classes, llm_dim=llm_dim)

    # model:
    #   input:  (batch_sz, llm_dim)
    #   output: (batch_sz, n_classes)
    model = nn.Sequential(transformer, cls_tokens, class_heads)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    return (model, loss_fn)


def build_model_multi_label_embedding_classifier_proj_packed(
        n_classes: int,
        class_weights:torch.Tensor,
        llm_dim: int,
        inner_dim: int,
        ff_dim: Optional[int]=None,
        nhead: int=1,
        ):
    
    proj = nn.Linear(in_features=llm_dim, out_features=inner_dim, bias=True)

    ff_dim = ff_dim or 4*inner_dim

    # input:  (batch_sz, seq_len, inner_dim)
    # output: (batch_sz, seq_len, inner_dim)
    transformer = nn.TransformerEncoderLayer(
        d_model=inner_dim,
        nhead=nhead,    
        dim_feedforward=ff_dim,
        dropout=0.1,
        batch_first=True,
    )

    # input:  (batch_sz, seq_len, inner_dim)
    # output: (batch_sz, inner_dim)
    cls_tokens = ElemAt(0)

    #   input: (batch_size, n_classes, inner_dim)
    #   output: (batch_size, n_classes)
    class_heads = PackedClassHead(n_classes=n_classes, llm_dim=inner_dim)

    # model:
    #   input:  (batch_sz, inner_dim)
    #   output: (batch_sz, n_classes)
    model = nn.Sequential(proj, transformer, cls_tokens, class_heads)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    return (model, loss_fn)


# K-sequence label model, which builds on the proj_packed model, but uses `k` sequences (and classification heads) that are aggregated.
# 
# 1. as inputs the model accepts k sequences (all of same seq_len and  llm_dim) 
# 2. for each sequence, predicts a packed classification head and (with linear classification) predicts a vector of classification logits
# 3. across all sequences, aggregates these logits via argmax (or similar function) to predict the final logits
#
# The loss function acts on the final logits.
# The variable seq_logits represents the classification logits predicted for each sequence in the batch. Let’s break it down step by step:
#
#                ----------------- 
# Input Context:
#
# The input to the model is a tensor of shape (batch_size, k, seq_len, llm_dim), where:
#   *  batch_size is the number of examples in the batch.
#   *  k is the number of sequences per example.
#   *  seq_len is the length of each sequence.
#   *  llm_dim is the dimensionality of the input embeddings.
#
# The model processes each sequence independently, and seq_logits is the intermediate output representing the logits for each sequence.
# Flow of seq_logits:
#
#                ----------------- 
#     Reshaping the Input:
#
# inputs = inputs.view(batch_size * k, seq_len, llm_dim)
#
# The input is reshaped to process each sequence independently. After reshaping:
#
#     The new shape is (batch_size * k, seq_len, llm_dim).
#     Each sequence is now treated as a separate input in the batch.
#
# Passing Through the Model:
#
#     Projection: The input is projected to a lower-dimensional space using a Linear layer:
#
# x = self.proj(inputs)  # Shape: (batch_size * k, seq_len, inner_dim)
#
# Transformer: The projected embeddings are passed through a transformer encoder:
#
# x = self.transformer(x)  # Shape: (batch_size * k, seq_len, inner_dim)
#
# CLS Token Extraction: The model uses the embedding of the [CLS] token (assumed to be at position 0) as a summary of the sequence:
#
#     cls_token = x[:, 0, :]  # Shape: (batch_size * k, inner_dim)
#
# Generating Sequence-level Logits:
#
#     The extracted [CLS] token is passed through the classification head:
#
#     seq_logits = self.class_heads(cls_token)  # Shape: (batch_size * k, n_classes)
#
#     Each sequence produces a vector of logits (one for each class). The total number of logits corresponds to (batch_size * k, n_classes).
#
# Reshaping Back for Aggregation: The logits are reshaped to group them by batch and sequence:
#
# seq_logits = seq_logits.view(batch_size, k, -1)  # Shape: (batch_size, k, n_classes)
#
#     Here, seq_logits contains the predicted logits for all k sequences for each example in the batch.
#     The shape (batch_size, k, n_classes) means:
#         For each example in the batch, there are k sequences.
#         For each sequence, the model predicts a vector of size n_classes.
#
# Final Aggregation: The logits from all k sequences are aggregated to produce a single logits vector for each example in the batch:
#
#     final_logits = self.aggregate(seq_logits)  # Shape: (batch_size, n_classes)
#
#     The aggregation reduces the sequence dimension k by combining the sequence-level logits into a final logits vector using methods like max, mean, or argmax.
#
# Summary of seq_logits:
#
#     Purpose: Represents the classification logits for each individual sequence in the input.
#     Shape: (batch_size, k, n_classes).
#     Aggregation: These logits are combined across the k sequences to produce the final logits for the entire example in the batch.
#
# By first generating seq_logits, the model ensures that each sequence is processed independently before combining their results in a meaningful way for the final prediction.

class AggregateAndClassify(nn.Module):
    def __init__(self, 
                 n_classes: int, 
                 class_weights: torch.Tensor, 
                 llm_dim: int, 
                 inner_dim: int, 
                 ff_dim: Optional[int] = None, 
                 nhead: int = 1, 
                 k: int = 1, 
                 aggregation: str = 'max'):
        super().__init__()
        self.k = k
        self.aggregation = aggregation

        # Projection layer
        self.proj = nn.Linear(in_features=llm_dim, out_features=inner_dim, bias=True)

        # Transformer Encoder Layer
        ff_dim = ff_dim or 4 * inner_dim
        self.transformer = nn.TransformerEncoderLayer(
            d_model=inner_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )

        # Classification Head for each sequence
        self.class_heads = PackedClassHead(n_classes=n_classes, llm_dim=inner_dim)

        # Aggregation function (default to max)
        if aggregation == 'max':
            self.aggregate = lambda logits: logits.max(dim=1)[0]
        elif aggregation == 'mean':
            self.aggregate = lambda logits: logits.mean(dim=1)
        elif aggregation == 'argmax':
            self.aggregate = lambda logits: logits.argmax(dim=1)
        else:
            raise ValueError("Invalid aggregation method. Choose from 'max', 'mean', or 'argmax'.")

        # Loss function

    def forward(self, inputs: torch.Tensor):
        """
        Inputs:
            - inputs: Tensor of shape (batch_size, k, seq_len, llm_dim)
        Outputs:
            - final_logits: Tensor of shape (batch_size, n_classes)
        """
        batch_size, k, seq_len, llm_dim = inputs.size()

        # Reshape to process each sequence independently
        inputs = inputs.view(batch_size * k, seq_len, llm_dim)

        # Pass through projection and transformer
        x = self.proj(inputs)  # (batch_size * k, seq_len, inner_dim)
        x = self.transformer(x)  # (batch_size * k, seq_len, inner_dim)

        # Extract classification token (cls token assumed to be at position 0)
        cls_token = x[:, 0, :]  # (batch_size * k, inner_dim)

        # Sequence-level logits
        seq_logits = self.class_heads(cls_token)  # (batch_size * k, n_classes)

        # Reshape back to group by batch
        seq_logits = seq_logits.view(batch_size, k, -1)  # (batch_size, k, n_classes)

        # Aggregate across sequences
        final_logits = self.aggregate(seq_logits)  # (batch_size, n_classes)


        return final_logits


def build_model_multi_label_multi_seq_embedding_classifier_proj_packed(n_classes: int
        ,class_weights:torch.Tensor
        ,llm_dim: int
        ,inner_dim: int
        ,num_seqs: int
        ,ff_dim: Optional[int]=None
        ,nhead: int=1
        ):
    # n_classes = 10
    # class_weights = torch.ones(n_classes)
    # llm_dim = 768
    # inner_dim = 128
    # seq_len = 32
    # k = 5
    # batch_size = 16

    # Model initialization
    model = AggregateAndClassify( n_classes=n_classes
                                , class_weights=class_weights
                                , llm_dim=llm_dim
                                , inner_dim=inner_dim
                                , k=num_seqs
                                , nhead=nhead
                                , aggregation="max"
                                )

    # # Dummy inputs
    # inputs = torch.randn(batch_size, k, seq_len, llm_dim)
    # targets = torch.randint(0, 2, (batch_size, n_classes)).float()

    # # Forward pass
    # final_logits = model(inputs)

    # Compute loss
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    return (model, loss_fn)



# ======================

def label_idx_to_class_list(label_idx: Dict[Label,int]) -> List[int]:
    class_list = list(label_idx.values())
    sorted(class_list)
    return class_list

def read_embeddings_synthetic() -> Tuple[Dataset, Dataset, List[int]]:
    class_list=[0,1,2,3]
    seq_len=5
    #iclass_embedding = torch.rand(size=[len(classes), 512])
    class_embedding = torch.tensor(
            [[1,0,0,0,0],
             [0,1,0,0,0],
             [0,0,1,0,0],
             [0,0,0,1,1]],
            dtype=torch.float32) / 2
    embed_dim = class_embedding.shape[1]
    #class_embedding_seq = torch.tensor([[class_embedding[c, :] for i in range(seq_len)] for c in classes])  # (num_classes, seq_len, embed_dimension)
    class_embedding_seq = class_embedding[:,None,:].expand(len(class_list), seq_len, embed_dim)

    label_idx={0:0, 1:1, 2:2, 3:3}

    # Example label IDs
    example_label_id_list = [0, 1, 2, 3] * 6  # Repeat for balanced examples


    # Create tensor of example label IDs
    example_label_id = torch.tensor(example_label_id_list, dtype=torch.long)  # [num_examples]

    # Generate sequence embeddings for each example
    # Each example corresponds to `seq_len` embeddings (e.g., a sequence)
    embeddings = torch.stack(
        [class_embedding_seq[label_id] for label_id in example_label_id], dim=0
    )  # [num_examples, embedding_dim]

    # Generate one-hot encoding for labels
    example_label_one_hot = nn.functional.one_hot(example_label_id, num_classes=len(class_list)).to(torch.float32)

    train_ds = StackDataset(embedding=embeddings, label_one_hot=example_label_one_hot, label_id=example_label_id)
    test_ds = StackDataset(embedding=embeddings, label_one_hot=example_label_one_hot, label_id=example_label_id)

    print("test set size:", len(test_ds))
    print("train set size:", len(train_ds))
    return train_ds, test_ds, class_list


def read_embeddings(path: Path, n_parts: int) -> Tuple[Dataset, Dataset, List[int]]:
    test_frac = 0.1
    MIN_LABEL_EXAMPLES = 10

    parts = [ torch.load(path / f'{i:04d}.pt') for i in range(n_parts) ]

    # flatten parts
    embeddings = torch.concat([xs['embeddings'] for xs in parts]) # (n_examples, sequence_length, embedding_dim)
    multilabels: List[List[Label]] = [labels for xs in parts for labels in xs['labels']] # len(multilabels) == n_examples

    # Assign single label to each example
    example_label: List[Label] # (n_examples,)
    example_label = [lbls[0] for lbls in multilabels]

    # Determine label set
    label_counts: Dict[Label, int] = collections.Counter(example_label)
    all_labels: Set[Label] = { lbl for lbl, n in label_counts.items() if n > MIN_LABEL_EXAMPLES }
    label_idx: Dict[Label, int] = { lbl: idx for idx, lbl in enumerate(all_labels) }

    # Drop examples whose labels have all been dropped
    example_mask = torch.tensor([label in all_labels for label in example_label])
    embeddings = Subset(embeddings, example_mask.nonzero()[:,0])
    example_label_id: Tensor = torch.tensor([label_idx[label] for label in example_label if label in all_labels], dtype=torch.long)
    example_label_one_hot: Tensor = nn.functional.one_hot(example_label_id, len(label_idx))

    ds = StackDataset(
            embedding=embeddings,
            label_one_hot=example_label_one_hot.to(torch.float32),
            label_id=example_label_id,
        )

    # Test/train split
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idxs, test_idxs = [x for x in sss.split(np.zeros(len(ds)), example_label_one_hot)][0]
    train_ds = Subset(ds, train_idxs)
    test_ds = Subset(ds, test_idxs)
    print('test set size', len(test_ds))
    print('train set size', len(train_ds))
    return (train_ds, test_ds, label_idx_to_class_list(label_idx))


def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, class_list: List[int], device: torch.device):
    '''Computes the avg per-example loss and eval metrics on the whole training set, not just one batch. '''
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['embedding'].to(device)
            labels = batch['label_one_hot'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_labels.append(batch['label_one_hot'])

    y_true = torch.cat(all_labels)
    metrics = classification_metrics(y_pred_logits=torch.cat(all_preds), y_true_one_hot=y_true, class_list=class_list)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def classification_metrics(y_pred_logits, y_true_one_hot, class_list: List[int]) -> Dict[str, object]:
    """
    y_pred_logits: predicted logits
    y_true_one_hot: true one-hot
    class_list: ordered list of classes (as int-labels) to evaluate
    """
    y_true_label = y_true_one_hot.argmax(dim=1)
    y_pred_label = y_pred_logits.argmax(dim=1)
    y_pred_prob = torch.nn.functional.softmax(y_pred_logits, dim=1)

    conf_matrix = confusion_matrix(y_true=y_true_label, y_pred=y_pred_label, labels=np.arange(len(class_list))) #labels=np.array(list(classes)))
    true_positives = np.diagonal(conf_matrix).sum()

    # majority class count
    true_class_counts = conf_matrix.sum(axis=1)
    majority_class_count = np.max(true_class_counts)
    majority_class_index = np.argmax(true_class_counts)
    majority_class_predictions = conf_matrix[:, majority_class_index]  # Extract the column of majority class prediction
    majority_class_false = majority_class_predictions[np.arange(len(majority_class_predictions)) != majority_class_index].sum()

    metrics = {
            'f1_micro': f1_score(y_true_label, y_pred_label, average='micro'),
            'roc_auc': roc_auc_score(y_true_one_hot, y_pred_prob, multi_class='ovo', labels=np.array(class_list)),
            'true_pos': int(true_positives), # (confusion * np.eye(n_classes)).sum(),
            # The sum of predictions where the predicted class is the majority class but the true class is not the majority class.
            'majority_class_false': int(majority_class_false),
            'accuracy': accuracy_score(y_true_label, y_pred_label),
            }
    return metrics


class Reporter:
    def __init__(self, file: io.TextIOBase):
        self.file = file

    def report(self, epoch_n: int, test_metrics, train_metrics):
        x = {
                'epoch_n': epoch_n,
                'test':  test_metrics,
                'train':  train_metrics,
        }
        json.dump(x, self.file)
        print('\n', file=self.file)
        self.file.flush()
        print(epoch_n, "test", test_metrics)
        print(epoch_n, "train", train_metrics)


class Noop(AbstractContextManager):
    def __enter__(self):
        pass
        return self

    def __exit__(self, *args):
        pass
 


class ClassificationModel(Enum):
    multi_class = auto()
    multi_label = auto()
    multi_label_packed = auto()
    multi_label_proj_packed = auto()
    multi_label_multi_seq_proj_packed = auto()
    mlp = auto()

    @staticmethod
    def from_string(arg:str):
        try:
            return ClassificationModel[arg]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid ClassificationModel choice: %s" % arg)
        

def run(root: Path
        , model_type: ClassificationModel
        ,train_ds:Dataset
        ,test_ds:Dataset
        ,class_list:List[int]
        ,out_dir: Optional[Path]=None
        ,overwrite:bool=False
        ,batch_size: int=128
        ,n_epochs: int=30
        ,inner_dim: int=64
        ,nhead: int=1
        ,device_str:str = 'cuda'
        ,snapshot_every:Optional[int]=None
        ,eval_every:Optional[int]=None
        ,epoch_timer:typing.ContextManager = Noop()
        ,target_metric:str = 'roc_auc'
        ,snapshot_best_after:Optional[int] = None
        ):
    if out_dir is None:
        out_dir = root / Path('runs') / str(model_type)
    out_dir.mkdir(parents=True, exist_ok=overwrite)

    seq_len, llm_dim = train_ds[0]['embedding'].shape
    print('llm_dim', llm_dim)
    print('context_sz', seq_len)
    print('classes', len(class_list))
    prev_highest: float = sys.float_info.min



    device = torch.device(device=device_str)
    if model_type == ClassificationModel.multi_class:
        model, loss_fn = build_model_multi_class_classifier(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead)
    elif model_type == ClassificationModel.multi_label:
        model, loss_fn = build_model_multi_label_embedding_classifier(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.multi_label_packed:
        model, loss_fn = build_model_multi_label_embedding_classifier_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.multi_label_proj_packed:
        model, loss_fn = build_model_multi_label_embedding_classifier_proj_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, inner_dim=inner_dim, class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.multi_label_multi_seq_proj_packed:
        raise ValueError(f"{model_type} needs to be called with run_num_seqs")
        # model, loss_fn = build_model_multi_label_multi_seq_embedding_classifier_proj_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, inner_dim=inner_dim, num_seqs=num_seqs,  class_weights=torch.ones(len(class_list)).to(device))
    elif model_type == ClassificationModel.mlp:
        model, loss_fn = build_model_multi_class_classifier_mlp(layer_dims=[llm_dim//2**i for i in range(4)], n_classes=len(class_list), llm_dim=llm_dim)
    else:
        raise ValueError(f'unknown model {model_type}')

    model = model.to(device)
    # torchinfo.summary(model, input_size=(batch_size, seq_len, llm_dim), device=device)

    print('Training...')
    reporter = Reporter((out_dir / 'history-log.json').open('w'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch_t in range(n_epochs):
        with epoch_timer:
            print(f'epoch {epoch_t}...')
            model.train()
            for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
                optimizer.zero_grad()
                output = model(batch['embedding'].to(device))
                loss = loss_fn(output, batch['label_one_hot'].to(device))
                loss.backward()
                optimizer.step()

        # Save model checkpoint
        if snapshot_every and epoch_t % snapshot_every == 1 :
            torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch_t}.pt")


        if eval_every is None or epoch_t % eval_every == 1 :
            # Evaluate loss
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            test_metrics = evaluate(model, test_loader, loss_fn, class_list, device)

            train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            train_metrics = evaluate(model, train_eval_loader, loss_fn, class_list, device)
            reporter.report(epoch_t, test_metrics, train_metrics)

            # Save model checkpoint on better validation metric
            if test_metrics.get(target_metric) is None:
                raise RuntimeError(f"Snapshot target metric {target_metric} not available. Choices: {test_metrics.keys()}")
            target=test_metrics[target_metric]
            if target > prev_highest:
                print(f"Epoch {epoch_t}: Best {target_metric} on test: {target} (was: {prev_highest}).")
                prev_highest = target
                if snapshot_best_after and epoch_t >= snapshot_best_after :
                    print("Epoch {epoch_t}: Saving best snapshot")
                    torch.save(model.state_dict(), out_dir / f"model_best_epoch_{epoch_t}.pt")


    torch.save(model.state_dict(), out_dir / f"model_final.pt")



def run_num_seqs(root: Path
        , model_type: ClassificationModel
        ,train_ds:Dataset
        ,test_ds:Dataset
        ,class_list:List[int]
        ,out_dir: Optional[Path]=None
        ,overwrite:bool=False
        ,batch_size: int=128
        ,n_epochs: int=30
        ,inner_dim: int=64
        ,nhead: int=1
        ,device_str:str = 'cuda'
        ,snapshot_every:Optional[int]=None
        ,eval_every:Optional[int]=None
        ,epoch_timer:typing.ContextManager = Noop()
        ,target_metric:str = 'roc_auc'
        ,snapshot_best_after:Optional[int] = None
        ):
    if out_dir is None:
        out_dir = root / Path('runs') / str(model_type)
    out_dir.mkdir(parents=True, exist_ok=overwrite)

    num_seqs, seq_len, llm_dim = train_ds[0]['embedding'].shape
    print('llm_dim', llm_dim)
    print('seq_len', seq_len)
    print('num_seqs', num_seqs)
    print('classes', len(class_list))
    prev_highest: float = sys.float_info.min



    device = torch.device(device=device_str)
    if model_type == ClassificationModel.multi_label_multi_seq_proj_packed:
        model, loss_fn = build_model_multi_label_multi_seq_embedding_classifier_proj_packed(n_classes=len(class_list), llm_dim=llm_dim, nhead = nhead, inner_dim=inner_dim, num_seqs=num_seqs,  class_weights=torch.ones(len(class_list)).to(device))
    else:
        raise ValueError(f'unknown model {model_type}')

    model = model.to(device)
    # torchinfo.summary(model, input_size=(batch_size, seq_len, llm_dim), device=device)

    print('Training...')
    reporter = Reporter((out_dir / 'history-log.json').open('w'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch_t in range(n_epochs):
        with epoch_timer:
            print(f'epoch {epoch_t}...')
            model.train()
            for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
                optimizer.zero_grad()
                output = model(batch['embedding'].to(device))
                loss = loss_fn(output, batch['label_one_hot'].to(device))
                loss.backward()
                optimizer.step()

        # Save model checkpoint
        if snapshot_every and epoch_t % snapshot_every == 0 :
            torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch_t}.pt")

        if  eval_every is None or epoch_t % eval_every == 0 :
            # Evaluate loss
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            test_metrics = evaluate(model, test_loader, loss_fn, class_list, device)

            train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            train_metrics = evaluate(model, train_eval_loader, loss_fn, class_list, device)
            reporter.report(epoch_t, test_metrics, train_metrics)

            # Save model checkpoint on better validation metric
            if test_metrics.get(target_metric) is None:
                raise RuntimeError(f"Snapshot target metric {target_metric} not available. Choices: {test_metrics.keys()}")
            target=test_metrics[target_metric]
            if target > prev_highest:
                print(f"Epoch {epoch_t}: Best {target_metric} on test: {target} (was: {prev_highest}).")
                prev_highest = target
                if snapshot_best_after and epoch_t >= snapshot_best_after :
                    print("Epoch {epoch_t}: Saving best snapshot")
                    torch.save(model.state_dict(), out_dir / f"model_best_epoch_{epoch_t}.pt")


    torch.save(model.state_dict(), out_dir / f"model_final.pt")



def main() -> None:
    root = Path('out2')
    # train_ds, test_ds, class_list = read_embeddings(root, 95)
    train_ds, test_ds, class_list = read_embeddings_synthetic()

    cmd_args = {"snapshots":2, "n_epochs":10, "train_ds":train_ds, "test_ds":test_ds, "class_list":class_list, "device_str":"cpu", "inner_dim": 64, "nhead":1}

    # run(root=root, model_type='multi_class', **cmd_args)
    run(root=root, model_type=ClassificationModel.multi_label_proj_packed, **cmd_args)
    # run(root=root, model_type='multi_label', **cmd_args)
    # run(root=root, model_type='mlp', **cmd_args)


if __name__ == '__main__':
    main()

