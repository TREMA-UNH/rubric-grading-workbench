from collections import OrderedDict
from typing import Tuple, Optional
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




class ProblemType(Enum):
    multi_class = auto()
    multi_label = auto()

    @staticmethod
    def from_string(arg:str):
        import argparse
        try:
            return ProblemType[arg]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid ClassificationModel choice: %s" % arg)
        

###############################################################################
# 1) Basic Primitives
###############################################################################

class Reshape(nn.Module):
    """
    Wraps einops.rearrange into an nn.Module for readability.
    """
    def __init__(self, pattern: str, **kwargs):
        """
        Args:
            pattern: einops rearrange pattern.
            kwargs: dimension variables for the einops pattern.
        """
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply einops.rearrange with the stored pattern and kwargs.
        """
        return rearrange(x, self.pattern, **self.kwargs)


class ElemAt(nn.Module):
    """
    Takes the embedding at index `idx` along dimension=1 (e.g., the [CLS] token).
    """
    def __init__(self, idx: int = 0):
        super().__init__()
        self.idx = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size*k, seq_len, hidden_dim)
        returns: (batch_size*k, hidden_dim)
        """
        return x[:, self.idx, :]


class PackedClassHead(nn.Module):
    """
    A single linear layer that produces class logits of size `n_classes`.
    """
    def __init__(self, llm_dim: int, n_classes: int):
        super().__init__()
        self.class_head = nn.Linear(llm_dim, n_classes)

    def forward(self, class_tokens: torch.Tensor) -> torch.Tensor:
        """
        class_tokens: (batch_size*k, hidden_dim)
        returns: (batch_size*k, n_classes)
        """
        return self.class_head(class_tokens)


###############################################################################
# 2) MultiSequenceClassifier + Aggregation
###############################################################################

class MultiSequenceClassifier(nn.Module):
    """
    A module that:
      - Takes [CLS] embeddings of shape (batch_size*k, hidden_dim).
      - Produces:
          seq_logits_classes: (batch_size*k, n_classes)
          seq_logits_grades:  (batch_size*k, n_grades)
    """
    def __init__(self, n_classes: int, inner_dim: int, n_grades: int):
        super().__init__()
        self.class_head = PackedClassHead(llm_dim=inner_dim, n_classes=n_classes)
        # Project from class logits (n_classes) to grade logits (n_grades).
        self.grade_proj = nn.Linear(n_classes, n_grades)

    def forward(self, cls_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          seq_logits_classes: (batch_size*k, n_classes)
          seq_logits_grades:  (batch_size*k, n_grades)
        """
        seq_logits_classes = self.class_head(cls_tokens)          # (bk, n_classes)
        seq_logits_grades  = self.grade_proj(seq_logits_classes)  # (bk, n_grades)
        return seq_logits_classes, seq_logits_grades


class Aggregate(nn.Module):
    """
    Aggregates across the sequence dimension k using max, mean, or argmax.
    """
    def __init__(self, method: str = "max"):
        super().__init__()
        if method == "max":
            self.aggregate = lambda logits: logits.max(dim=1)[0]
        elif method == "mean":
            self.aggregate = lambda logits: logits.mean(dim=1)
        elif method == "argmax":
            self.aggregate = lambda logits: logits.argmax(dim=1)
        else:
            raise ValueError("Invalid aggregation method. Choose from 'max', 'mean', or 'argmax'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, k, n_classes)
        returns: (batch_size, n_classes) if method in [max, mean],
                 or (batch_size,) if argmax
        """
        return self.aggregate(x)


class MeanPoolAsCLS(nn.Module):
    """
    Replaces the token at position 0 with the mean
    across all tokens along dim=1.
    
    After this layer, calling ElemAt(0) yields the mean-pooled embedding
    for each sequence.

    This is a simple replacement for the transformer layer.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size*k, seq_len, hidden_dim)
        returns: same shape (batch_size*k, seq_len, hidden_dim),
                 but x[:, 0, :] is replaced with the mean across seq_len.
        """
        # mean_pool: (batch_size*k, hidden_dim)
        mean_pool = x.mean(dim=1)  # average over seq_len
        
        # We'll clone x if we don't want to modify it in-place
        x = x.clone()
        x[:, 0, :] = mean_pool  # store the mean at position 0
        return x


###############################################################################
# 3) Single-Layer Primitive for Class & Grade Prediction
###############################################################################

class ClassAndGradePredictor(nn.Module):
    """
    A single module that:
      1) Invokes a MultiSequenceClassifier.
      2) Reshapes (batch_size*k) -> (batch_size, k, ...) for both class & grade logits.
      3) Aggregates class logits to produce final_logits of shape (batch_size, n_classes).
      4) Returns (final_logits, seq_logits_grades).
    """
    def __init__(self,
                 multi_seq_classifier: nn.Module,
                 aggregator: nn.Module,
                 num_seqs: int):
        """
        Args:
            multi_seq_classifier:
                Takes shape (batch_size*k, hidden_dim) -> two outputs
                (seq_logits_classes, seq_logits_grades).
            aggregator:
                Aggregates across k dimension for class logits.
            num_seqs: number of sequences (k) per example.
        """
        super().__init__()
        self.multi_seq_classifier = multi_seq_classifier
        self.aggregator = aggregator
        self.num_seqs = num_seqs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size*k, hidden_dim)

        Returns:
          final_logits: (batch_size, n_classes)
          seq_logits_grades: (batch_size, k, n_grades)
        """
        seq_logits_classes, seq_logits_grades = self.multi_seq_classifier(x)
        # (bk, n_classes), (bk, n_grades)

        batch_size = x.size(0) // self.num_seqs

        # Reshape class logits from (bk, n_classes) -> (b, k, n_classes)
        seq_logits_classes = rearrange(
            seq_logits_classes,
            '(b k) c -> b k c',
            b=batch_size, k=self.num_seqs
        )

        # Aggregate over k dimension => (b, n_classes)
        final_logits = self.aggregator(seq_logits_classes)

        # Reshape grade logits from (bk, n_grades) -> (b, k, n_grades)
        seq_logits_grades = rearrange(
            seq_logits_grades,
            '(b k) g -> b k g',
            b=batch_size, k=self.num_seqs
        )

        return final_logits, seq_logits_grades


###############################################################################
# 4) Model Class with Overridden Forward + Loss Logic
###############################################################################

class MultiLabelMultiSeqEmbeddingClassifier(nn.Module):
    """
    A model that:
      1. Accepts (batch_size, k, seq_len, llm_dim).
      2. Flattens (b, k, seq_len, d) -> (b*k, seq_len, d), applies a projection & Transformer,
         then extracts the [CLS] token.
      3. Calls a single ClassAndGradePredictor module, which returns
         (final_logits, seq_logits_grades).
      4. compute_loss() uses either BCEWithLogits or CrossEntropy for final logits (labels)
         and for seq_logits_grades (grades), depending on label_problem_type and
         grade_problem_type.
    Defaults to multi-class for both label and grade classification.
    """

    def __init__(self,
                 n_classes: int,
                 class_weights_mc: torch.Tensor,
                 class_weights_ml: torch.Tensor,
                 n_grades: int,
                 grade_weights_mc: torch.Tensor,
                 grade_weights_ml: torch.Tensor,
                 llm_dim: int,
                 inner_dim: int,
                 num_seqs: int,
                 ff_dim: Optional[int] = None,
                 nhead: int = 1,
                 aggregation: str = "max",
                 label_problem_type: ProblemType = ProblemType.multi_class,
                 grade_problem_type: ProblemType = ProblemType.multi_class,
                 use_transformer: bool = True,
                 use_inner_proj: bool = True
                 ):
        super().__init__()

        self.num_seqs = num_seqs
        self.n_classes = n_classes
        self.n_grades = n_grades

        self.label_problem_type = label_problem_type
        self.grade_problem_type = grade_problem_type
        self.use_transformer = use_transformer
        self.use_inner_proj = use_inner_proj

        # -------------------------------
        # 1) A small sequential pipeline
        # -------------------------------
        # This pipeline outputs shape (b*k, inner_dim) after flatten, projection, transform, and elemAt.
        self.pipeline = nn.Sequential()
        self.pipeline.add_module("reshape_seqs", Reshape('b k l d -> (b k) l d'))
        if self.use_inner_proj:
            self.pipeline.add_module("proj", nn.Linear(llm_dim, inner_dim, bias=True))
            self.inner_dim = inner_dim
        else:
            self.inner_dim = llm_dim
        if use_transformer:
            ff_dim = ff_dim or 4 * self.inner_dim
            self.pipeline.add_module ("transformer1", nn.TransformerEncoderLayer(
                    d_model=self.inner_dim,
                    nhead=nhead,
                    dim_feedforward=ff_dim,
                    dropout=0.1,
                    batch_first=True,
                ))
            self.pipeline.add_module ("transformer2", nn.TransformerEncoderLayer(
                    d_model=self.inner_dim,
                    nhead=nhead,
                    dim_feedforward=ff_dim,
                    dropout=0.1,
                    batch_first=True,
                ))
        else:
            self.pipeline.add_module ("token_pool", MeanPoolAsCLS())

        self.pipeline.add_module("cls", ElemAt(0))  # pick [CLS] token => shape (b*k, inner_dim)


        # 2) Single-layer predictor that returns (final_logits, seq_logits_grades).
        multi_seq_classifier = MultiSequenceClassifier(n_classes, self.inner_dim, n_grades)
        aggregator = Aggregate(method=aggregation)
        self.predictor = ClassAndGradePredictor(multi_seq_classifier, aggregator, num_seqs)


        # 3) Define the loss functions
        self.loss_fn_class: nn.Module
        if self.label_problem_type == ProblemType.multi_label:
            self.loss_fn_class = nn.BCEWithLogitsLoss(pos_weight=class_weights_ml)
        elif self.label_problem_type == ProblemType.multi_class:
            self.loss_fn_class = nn.CrossEntropyLoss(weight=class_weights_mc)
        else:
            raise ValueError("label_problem_type must be 'ProblemType.multi_label' or 'ProblemType.multi_class'")

        self.loss_fn_grade: nn.Module
        if self.grade_problem_type == ProblemType.multi_label:
            self.loss_fn_grade = nn.BCEWithLogitsLoss(pos_weight=grade_weights_ml)
        elif self.grade_problem_type == ProblemType.multi_class:
            self.loss_fn_grade = nn.CrossEntropyLoss(weight=grade_weights_mc)
        else:
            raise ValueError("grade_problem_type must be 'ProblemType.multi_label' or 'ProblemType.multi_class'")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that yields:
          final_logits: (batch_size, n_classes)
          seq_logits_grades: (batch_size, k, n_grades)
        """
        # 1) Pass input through the pipeline
        out = self.pipeline(x)  # shape: (b*k, inner_dim)

        # 2) Produce final_logits & seq_logits_grades
        final_logits, seq_logits_grades = self.predictor(out)

        return final_logits, seq_logits_grades

    @staticmethod
    def convert_one_hot_to_indices(y: torch.Tensor) -> torch.Tensor:
        """
        If your data is one-hot but you want integer class indices (for multi-class),
        do argmax along the last dimension.
        """
        return torch.argmax(y, dim=-1)

    @staticmethod
    def convert_indices_to_one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        If your data is integer-coded for multi-class but you want one-hot for multi-label,
        do one_hot + float conversion.
        """
        return F.one_hot(y.long(), num_classes=num_classes).float()

    def compute_loss(
        self,
        final_logits: torch.Tensor,
        class_targets: torch.Tensor,
        seq_logits_grades: Optional[torch.Tensor] = None,
        grade_targets: Optional[torch.Tensor] = None,
        grade_valid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, class_loss, grade_loss).
        Expects that `class_targets` and `grade_targets` are already in
        the correct shape/format for the chosen classification mode.
        """
        # A) Class-level loss
        if self.label_problem_type == ProblemType.multi_label:
            assert class_targets.dim() == 2, f"Multi-label class: Expected class_targets to have 2 dimensions (b, n_classes), got {class_targets.shape}"
            assert class_targets.dtype == torch.float, f"Multi-label class: Expected class_targets dtype to be float for multi-label, got {class_targets.dtype}"

            # print(f"loss multi-label label:  true {class_targets}, predict {final_logits}")

            # final_logits: (b, n_classes), class_targets: (b, n_classes)
            class_loss = self.loss_fn_class(final_logits, class_targets)
        else:  # multi_class
            assert class_targets.dim() == 1, f"Multi-class class: Expected class_targets to have 1 dimension (b,), got {class_targets.shape}"
            assert class_targets.dtype == torch.long, f"Multi-class class: Expected class_targets dtype to be long for multi-class, got {class_targets.dtype}"

            # print(f"loss multi-class label:  true {class_targets}, predict {final_logits}")

            # final_logits: (b, n_classes), class_targets: (b,)
            class_loss = self.loss_fn_class(final_logits, class_targets.long())

        # B) Grade-level loss
        grade_loss = torch.tensor(0.0, device=final_logits.device)
        if seq_logits_grades is not None and grade_targets is not None:
            # Check if a grade_valid mask is provided
            if grade_valid is None:
                grade_valid = torch.ones_like(grade_targets, dtype=torch.bool)  # Default: all valid
                num_valid_grades = -1
            else:
                num_valid_grades = grade_valid.sum().item()
                
            if num_valid_grades > 0:
                if self.grade_problem_type == ProblemType.multi_label:
                    # Ensure seq_logits_grades and grade_targets match shape
                    assert seq_logits_grades.shape == grade_targets.shape, (
                        f"Shape mismatch: seq_logits_grades={seq_logits_grades.shape}, "
                        f"grade_targets={grade_targets.shape}"
                    )

                    # Dynamically ensure grade_valid has the correct shape
                    if grade_valid.dim() < seq_logits_grades.dim():
                        expanded_mask = grade_valid.unsqueeze(-1).expand_as(seq_logits_grades)  # Add singleton dimension
                    elif grade_valid.dim() == seq_logits_grades.dim():
                        expanded_mask = grade_valid.expand_as(seq_logits_grades)  # Already aligned, just expand
                    else:
                        raise ValueError(
                            f"Unexpected dimensions for grade_valid: {grade_valid.dim()} "
                            f"with seq_logits_grades: {seq_logits_grades.dim()}"
                        )

                    # Mask logits and targets
                    masked_logits = seq_logits_grades[expanded_mask]
                    masked_targets = grade_targets[expanded_mask]

                    # Check if all entries are invalid
                    if masked_logits.numel() == 0:
                        # print("loss: no valid grades")
                        grade_loss = torch.tensor(0.0, device=seq_logits_grades.device)  # No valid entries, grade loss is 0
                    else:
                        # Reshape masked tensors to (N, n_grades)
                        masked_logits = masked_logits.view(-1, seq_logits_grades.size(-1))
                        masked_targets = masked_targets.view(-1, seq_logits_grades.size(-1))

                        # Compute grade loss
                        grade_loss = self.loss_fn_grade(masked_logits, masked_targets)
                        # print(f"multi-label loss: grade_loss = {grade_loss}")
                else:  # multi_class
                    # Reshape logits and targets
                    # print("loss: mask non-zero:",num_valid_grades)
                    seq_logits_grades_2d = rearrange(seq_logits_grades, 'b k g -> (b k) g')
                    grade_targets_1d = rearrange(grade_targets, 'b k -> (b k)')  
                    grade_valid_1d = rearrange(grade_valid, 'b k -> (b k)')  # Flatten valid mask

                    # Mask logits and targets
                    masked_logits = seq_logits_grades_2d[grade_valid_1d]
                    masked_targets = grade_targets_1d[grade_valid_1d]
                    # print("valid target_grades:", len(masked_targets), "mask non-zero:",num_valid_grades)
                    grade_loss = self.loss_fn_grade(masked_logits, masked_targets.long())
                    # print(f"multi-class loss: grade_loss = {grade_loss}")
       


        total_loss = class_loss + grade_loss
        # print(f"loss: {total_loss}, grade_loss={grade_loss}")
        return total_loss, class_loss, grade_loss


###############################################################################
# 5) Build Function
###############################################################################

def build_better_model_multi_label_multi_seq_embedding_classifier_proj_packed(
    n_classes: int,
    class_weights_mc: torch.Tensor,
    class_weights_ml: torch.Tensor,
    n_grades: int,
    grade_weights_mc: torch.Tensor,
    grade_weights_ml: torch.Tensor,
    llm_dim: int,
    inner_dim: int,
    num_seqs: int,
    ff_dim: Optional[int] = None,
    nhead: int = 1,
    aggregation: str = "max",
    label_problem_type: ProblemType = ProblemType.multi_class,
    grade_problem_type: ProblemType = ProblemType.multi_class, 
    use_transformer: bool = True,
    use_inner_proj: bool = True
    ) -> MultiLabelMultiSeqEmbeddingClassifier:
    """
    Convenience factory function for building the model with desired hyperparams.
    """
    return MultiLabelMultiSeqEmbeddingClassifier(
        n_classes=n_classes,
        class_weights_mc=class_weights_mc,
        class_weights_ml=class_weights_ml,
        n_grades=n_grades,
        grade_weights_mc=grade_weights_mc,
        grade_weights_ml=grade_weights_ml,        
        llm_dim=llm_dim,
        inner_dim=inner_dim,
        num_seqs=num_seqs,
        ff_dim=ff_dim,
        nhead=nhead,
        aggregation=aggregation,
        label_problem_type=label_problem_type,
        grade_problem_type=grade_problem_type,
        use_transformer=use_transformer,
        use_inner_proj=use_inner_proj
    )


###############################################################################
# 6) Example Main
###############################################################################

if __name__ == "__main__":
    # Example hyperparameters
    batch_size = 2
    k = 3
    seq_len = 5
    llm_dim = 16
    inner_dim = 8
    n_classes = 4
    n_grades = 3

    # Example weights (all ones for demonstration)
    class_weights = torch.ones(n_classes)
    grade_weights = torch.ones(n_grades)

    # Build the model
    model = build_better_model_multi_label_multi_seq_embedding_classifier_proj_packed(
        n_classes=n_classes,
        class_weights=class_weights,
        n_grades=n_grades,
        grade_weights=grade_weights,
        llm_dim=llm_dim,
        inner_dim=inner_dim,
        num_seqs=k,
        ff_dim=None,
        nhead=1,
        aggregation="mean",
        label_problem_type=ProblemType.multi_class,  # or "multi_label"
        grade_problem_type=ProblemType.multi_class   # or "multi_label"
    )

    # Example input shape: (batch_size=2, k=3, seq_len=5, llm_dim=16)
    x = torch.randn(batch_size, k, seq_len, llm_dim)

    # Forward pass -> (final_logits, seq_logits_grades)
    final_logits, seq_logits_grades = model(x)
    print("final_logits.shape =", final_logits.shape)          # [2, n_classes] => [2, 4]
    print("seq_logits_grades.shape =", seq_logits_grades.shape)  # [2, k, n_grades] => [2, 3, 3]

    # Prepare targets
    # For multi_class labels => (batch_size,) with integer labels
    class_targets = torch.randint(0, n_classes, (batch_size,))
    # For multi_class grades => (batch_size, k)
    grade_targets = torch.randint(0, n_grades, (batch_size, k))

    # If multi_label, we would convert to shape [b, n_classes] or [b, k, n_grades]
    # e.g.: class_targets = model.convert_indices_to_one_hot(class_targets, n_classes)

    # Compute loss
    total_loss, class_loss, grade_loss = model.compute_loss(
        final_logits, class_targets, seq_logits_grades, grade_targets
    )
    print("total_loss =", total_loss.item())
    print("class_loss =", class_loss.item())
    print("grade_loss =", grade_loss.item())
