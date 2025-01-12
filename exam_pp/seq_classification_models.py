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
                 n_grades: int,
                 grade_weights: torch.Tensor,
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

        # Projection for sequence-level logits to grade logits
        self.grade_proj = nn.Linear(n_classes, n_grades)

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
        self.loss_fn_class = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        # self.loss_fn_class = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.loss_fn_grade = nn.CrossEntropyLoss(weight=grade_weights)

    def forward(self, inputs: torch.Tensor):
        """
        Inputs:
            - inputs: Tensor of shape (batch_size, k, seq_len, llm_dim)
        Outputs:
            - class_logits: Tensor of shape (batch_size, n_classes)
            - grade_logits: Tensor of shape (batch_size, k, n_grades)
        """
        batch_size, k, seq_len, llm_dim = inputs.size()

        # Reshape to process each sequence independently
        inputs = inputs.view(batch_size * k, seq_len, llm_dim)

        # Pass through projection and transformer
        x = self.proj(inputs)  # (batch_size * k, seq_len, inner_dim)
        x = self.transformer(x)  # (batch_size * k, seq_len, inner_dim)

        # Extract classification token (cls token assumed to be at position 0)
        cls_token = x[:, 0, :]  # (batch_size * k, inner_dim)

        # Sequence-level logits for classes
        seq_logits_classes = self.class_heads(cls_token)  # (batch_size * k, n_classes)

        # Sequence-level logits for grades
        self.seq_logits_grades = self.grade_proj(seq_logits_classes)  # (batch_size * k, n_grades)

        # Reshape back to group by batch
        seq_logits_classes = seq_logits_classes.view(batch_size, k, -1)  # (batch_size, k, n_classes)
        self.seq_logits_grades = self.seq_logits_grades.view(batch_size, k, -1)  # (batch_size, k, n_grades)


        # Aggregate sequence-level logits for classes
        final_logits = self.aggregate(seq_logits_classes)  # (batch_size, n_classes)


        return final_logits, self.seq_logits_grades

    def convert_multi_label_to_multi_class_loss_inputs(self,seq_logits_grades:torch.Tensor,grade_targets:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        '''According to the PyTorch documentation for torch.nn.CrossEntropyLoss, the inputs (logits) and targets need to be shaped as follows 
          for a multi-class classification problem:
            Requirements:

            Logits (input):
                The shape should be (minibatch, C, ...), where:
                    minibatch is the batch size.
                    C is the number of classes (e.g., 6 in your case).
                    The rest (...) depends on the task (e.g., additional dimensions for sequences or images).

            Targets (target):
                The shape should be (minibatch, ...), where:
                    minibatch matches the first dimension of input.
                    ... matches the rest of the dimensions of input, excluding C.
                target must contain class indices (integer values in the range [0, C-1]).
                One-hot encoded targets (like [10, 10, 6] in your case) are not supported directly.
        '''
        return (seq_logits_grades.permute(0, 2, 1), 
                torch.argmax(grade_targets, dim=-1)
        )

    def compute_loss(self, final_logits: torch.Tensor
                     , class_targets: torch.Tensor
                     , seq_logits_grades: Optional[torch.Tensor]=None
                     , grade_targets: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute the total loss:
            - Classification loss on final logits (n_classes)
            - Grade loss on sequence-level logits (n_grades)
        """
        # # Compute class-level loss
        class_loss = self.loss_fn_class(final_logits, class_targets)
        
        grade_loss = 0.0
        if (seq_logits_grades is not None) and ( grade_targets is not None):
            # must rearrage for multi-class loss
            seq_logits_grades_mc, grade_targets_mc = self.convert_multi_label_to_multi_class_loss_inputs(seq_logits_grades, grade_targets)
            grade_loss = self.loss_fn_grade(seq_logits_grades_mc, grade_targets_mc)

        # Total loss is a weighted sum (can modify weights as needed)
        total_loss = class_loss + grade_loss
        return total_loss, class_loss, grade_loss

def build_model_multi_label_multi_seq_embedding_classifier_proj_packed(n_classes: int
        ,class_weights:torch.Tensor
        , n_grades:int
        , grade_weights:torch.Tensor
        ,llm_dim: int
        ,inner_dim: int
        ,num_seqs: int
        ,ff_dim: Optional[int]=None
        ,nhead: int=1
        , aggregation:str= "max"
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
                                , n_grades=n_grades
                                , grade_weights=grade_weights
                                , llm_dim=llm_dim
                                , ff_dim= ff_dim
                                , inner_dim=inner_dim
                                , k=num_seqs
                                , nhead=nhead
                                , aggregation=aggregation
                                )

    # # Dummy inputs
    # inputs = torch.randn(batch_size, k, seq_len, llm_dim)
    # targets = torch.randint(0, 2, (batch_size, n_classes)).float()

    # # Forward pass
    # final_logits = model(inputs)



    return (model, model.loss_fn_class, model.loss_fn_grade)



# =========================


import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple

from einops import rearrange

class ReshapeEinops(nn.Module):
    def __init__(self, pattern: str, **kwargs):
        """
        Args:
            pattern: The einops pattern for rearranging tensors.
            kwargs: Additional dimensions required for the pattern.
        """
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        """
        Args:
            x: Input tensor to be reshaped.
        Returns:
            Reshaped tensor.
        """
        return rearrange(x, self.pattern, **self.kwargs)

class ReshapeView(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(*self.new_shape)


class Aggregate(nn.Module):
    """
    Aggregates across the sequence dimension k using max, mean, or argmax.
    """
    def __init__(self, aggregation: str = "max"):
        super().__init__()
        if aggregation == "max":
            self.aggregate = lambda logits: logits.max(dim=1)[0]
        elif aggregation == "mean":
            self.aggregate = lambda logits: logits.mean(dim=1)
        elif aggregation == "argmax":
            self.aggregate = lambda logits: logits.argmax(dim=1)
        else:
            raise ValueError("Invalid aggregation method. Choose from 'max', 'mean', or 'argmax'.")

    def forward(self, x):
        return self.aggregate(x)


class MultiSequenceLabelGradeClassifier(nn.Module):
    """
    Applies a PackedClassHead for class logits and an additional projection
    for grade logits from the class logits.
    """
    def __init__(self, n_classes: int, inner_dim: int, n_grades: int):
        super().__init__()
        self.class_head = PackedClassHead(llm_dim=inner_dim, n_classes=n_classes)
        self.grade_proj = nn.Linear(n_classes, n_grades)

    def forward(self, cls_tokens):
        seq_logits_classes = self.class_head(cls_tokens)  # (batch_size * k, n_classes)
        seq_logits_grades = self.grade_proj(seq_logits_classes)  # (batch_size * k, n_grades)
        return seq_logits_classes, seq_logits_grades


class PrevMultiLabelMultiSeqEmbeddingClassifier(nn.Sequential):
    """
    A model class that:
    1. Flattens (batch_size, k) into a single dimension so each sequence is processed independently.
    2. Applies a projection -> Transformer -> [CLS] extraction -> MultiSequenceClassifier.
    3. Regroups by batch (restoring k).
    4. Aggregates the per-sequence class logits into final_logits.
    5. Retains seq_logits_grades for optional per-sequence grade classification.
    6. Provides compute_loss and utility for multi-label to multi-class transformation.
    """

    def __init__(self,
                 n_classes: int,
                 class_weights: torch.Tensor,
                 n_grades: int,
                 grade_weights: torch.Tensor,
                 llm_dim: int,
                 inner_dim: int,
                 num_seqs: int,
                 ff_dim: Optional[int] = None,
                 nhead: int = 1,
                 aggregation: str = "max"):
        # 1) Initialize base Sequential
        super().__init__()

        ff_dim = ff_dim or 4 * inner_dim

        # Store these for the compute_loss function
        self.loss_fn_class = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.loss_fn_grade = nn.CrossEntropyLoss(weight=grade_weights)

        # We’ll build the sequence of layers in a list, then extend with add_module
        # so that the forward pass can still function as a normal nn.Sequential chain.
        self.add_module("flatten_sequences", Reshape('b k l d -> (b k) l d'))
        self.add_module("proj", nn.Linear(llm_dim, inner_dim, bias=True))
        self.add_module("transformer",
            nn.TransformerEncoderLayer(
                d_model=inner_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=0.1,
                batch_first=True,
            )
        )
        self.add_module("cls_token", ElemAt(idx=0))
        self.add_module("multi_seq_classifier", MultiSequenceClassifier(
            n_classes=n_classes,
            inner_dim=inner_dim,
            n_grades=n_grades
        ))
        # The last steps need special handling (we want two outputs from multi_seq_classifier)
        # so we won't add them directly as further modules in nn.Sequential.
        # We'll handle them in our custom forward method.

        # aggregator to combine sequence-level class logits
        self.aggregator = Aggregate(method=aggregation)

        # We'll keep track of num_seqs for regrouping
        self.num_seqs = num_seqs
        self.n_classes = n_classes

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that yields:
        final_logits: (batch_size, n_classes)
        seq_logits_grades: (batch_size, num_seqs, n_grades)
        """
        # Step 1 to 5: pass through all sub-modules in order (nn.Sequential style).
        # However, the "multi_seq_classifier" returns two outputs (seq_logits_classes, seq_logits_grades),
        # so we must split after that module and apply aggregator ourselves.

        # Pass inputs through layers up to "multi_seq_classifier"
        # nn.Sequential will pass the output from one layer to the next,
        # but multi_seq_classifier returns a tuple. We can intercept that.
        x = inputs
        for name, module in self.named_children():
            if name == "multi_seq_classifier":
                # The module that returns two outputs
                seq_logits_classes, seq_logits_grades = module(x)  # x is the [CLS] embeddings
                # break out of the loop because now we have the two separate outputs
                break
            else:
                x = module(x)  # normal pass

        # Now `seq_logits_classes`: (batch_size * k, n_classes)
        # and `seq_logits_grades`: (batch_size * k, n_grades)

        # We still need to pass the data after "multi_seq_classifier"
        # through the rest of the layers (if there were any).
        # But in this design, aggregator is the only next step for the class logits
        # plus we need to reshape them first.

        # Step 6: Reshape class logits back to (batch_size, num_seqs, n_classes)
        seq_logits_classes = rearrange(
            seq_logits_classes, 
            '(b k) c -> b k c', 
            b=inputs.size(0), 
            k=self.num_seqs
        )

        # Step 7: Aggregate across k to get the final class logits per batch
        final_logits = self.aggregator(seq_logits_classes)  # (batch_size, n_classes)

        # Meanwhile, for grades, we can also reshape them to (batch_size, num_seqs, n_grades)
        seq_logits_grades = rearrange(
            seq_logits_grades,
            '(b k) g -> b k g',
            b=inputs.size(0),
            k=self.num_seqs
        )

        return final_logits, seq_logits_grades

    def convert_multi_label_to_multi_class_loss_inputs(
        self,
        seq_logits_grades: torch.Tensor,
        grade_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        According to PyTorch docs for CrossEntropyLoss, shape requirements:
          - logits (input): (minibatch, C, ...)
          - targets (target): (minibatch, ...)
        
        We do two things:
          1) permute seq_logits_grades to (batch_size, n_grades, k) [or similar]
          2) convert one-hot targets via argmax
        """
        # seq_logits_grades: (batch_size, k, n_grades)
        # reorder to (batch_size, n_grades, k)
        seq_logits_grades_mc = seq_logits_grades.permute(0, 2, 1)  
        # If `grade_targets` is one-hot (batch_size, k, n_grades), convert to integer indices
        grade_targets_mc = torch.argmax(grade_targets, dim=-1)  # (batch_size, k)
        return seq_logits_grades_mc, grade_targets_mc

    def compute_loss(
        self,
        final_logits: torch.Tensor,
        class_targets: torch.Tensor,
        seq_logits_grades: Optional[torch.Tensor] = None,
        grade_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute the total loss:
          - Classification loss on final logits (n_classes)
          - Grade loss on sequence-level logits (n_grades), if provided
        """
        # (A) Class-level loss
        class_loss = self.loss_fn_class(final_logits, class_targets)

        # (B) Grade-level loss (optional)
        grade_loss = 0.0
        if (seq_logits_grades is not None) and (grade_targets is not None):
            seq_logits_grades_mc, grade_targets_mc = self.convert_multi_label_to_multi_class_loss_inputs(
                seq_logits_grades, grade_targets
            )
            grade_loss = self.loss_fn_grade(seq_logits_grades_mc, grade_targets_mc)

        # Total loss is a sum (you may introduce weighting factors if desired)
        total_loss = class_loss + grade_loss
        return total_loss, class_loss, grade_loss


# class MultiLabelMultiSeqModel(nn.Sequential):
#     def __init__(self, 
#                  n_classes: int,
#                  class_weights: torch.Tensor,
#                  n_grades: int,
#                  grade_weights: torch.Tensor,
#                  llm_dim: int,
#                  inner_dim: int,
#                  num_seqs: int,
#                  ff_dim: Optional[int] = None,
#                  nhead: int = 1,
#                  aggregation: str = "max"):
#         ff_dim = ff_dim or 4 * inner_dim

#         # Components of the model
#         layers = [
#             # Step 1: Flatten sequences for independent processing
#             ReshapeEinops('b k l d -> (b k) l d'),  # Shape: (batch_size * num_seqs, seq_len, llm_dim)

#             # Step 2: Linear projection to reduce dimensionality
#             nn.Linear(in_features=llm_dim, out_features=inner_dim, bias=True),  # Shape: (batch_size * num_seqs, seq_len, inner_dim)

#             # Step 3: Transformer encoder for contextual sequence processing
#             nn.TransformerEncoderLayer(
#                 d_model=inner_dim,
#                 nhead=nhead,
#                 dim_feedforward=ff_dim,
#                 dropout=0.1,
#                 batch_first=True,
#             ),  # Shape: (batch_size * num_seqs, seq_len, inner_dim)

#             # Step 4: Extract CLS token for each sequence
#             ElemAt(idx=0),  # Shape: (batch_size * num_seqs, inner_dim)

#             # Step 5: Sequence-level classification for labels and grades
#             MultiSequenceLabelGradeClassifier(n_classes=n_classes, inner_dim=inner_dim, n_grades=n_grades),  # Outputs:
#             # - seq_logits_classes: (batch_size * num_seqs, n_classes)
#             # - seq_logits_grades: (batch_size * num_seqs, n_grades)

#             # Step 6: Reshape logits back to group by batch
#             ReshapeEinops('(b k) c -> b k c', k=num_seqs),  # Shape: (batch_size, num_seqs, n_classes)

#             # Step 7: Aggregate sequence-level logits for final predictions
#             Aggregate(aggregation=aggregation),  # Shape: (batch_size, n_classes)
#         ]

#         # Initialize the parent nn.Sequential class with these layers
#         super().__init__(*layers)

#         # Loss functions
#         self.loss_fn_class = nn.BCEWithLogitsLoss(pos_weight=class_weights)
#         self.loss_fn_grade = nn.CrossEntropyLoss(weight=grade_weights)

    # def convert_multi_label_to_multi_class_loss_inputs(self,
    #                                                    seq_logits_grades: torch.Tensor,
    #                                                    grade_targets: torch.Tensor
    #                                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Prepares logits and targets for CrossEntropyLoss:
    #         - Logits: Shape (batch_size, n_grades, num_seqs).
    #         - Targets: Shape (batch_size, num_seqs).
    #     """
    #     return (rearrange(seq_logits_grades, 'b k g -> b g k'),  # Permute dimensions for CrossEntropyLoss
    #             torch.argmax(grade_targets, dim=-1))  # Convert one-hot to class indices

    # def compute_loss(self,
    #                  final_logits: torch.Tensor,
    #                  class_targets: torch.Tensor,
    #                  seq_logits_grades: Optional[torch.Tensor] = None,
    #                  grade_targets: Optional[torch.Tensor] = None
    #                  ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    #     """
    #     Computes the total loss combining:
    #         - Class-level multi-label loss.
    #         - Grade-level multi-class loss.
    #     """
    #     # Compute class-level loss
    #     class_loss = self.loss_fn_class(final_logits, class_targets)

    #     grade_loss = 0.0
    #     if seq_logits_grades is not None and grade_targets is not None:
    #         # Rearrange logits and targets for grade-level loss
    #         seq_logits_grades_mc, grade_targets_mc = self.convert_multi_label_to_multi_class_loss_inputs(
    #             seq_logits_grades, grade_targets)
    #         grade_loss = self.loss_fn_grade(seq_logits_grades_mc, grade_targets_mc)

    #     # Total loss as sum of class and grade losses
    #     total_loss = class_loss + grade_loss
    #     return total_loss, class_loss, grade_loss
