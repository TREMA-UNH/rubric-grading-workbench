from pathlib import Path
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


def read_embeddings_synthetic() -> Tuple[Dataset, Dataset, Dict[Label, int]]:
    classes=[0,1,2,3]
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
    class_embedding_seq = class_embedding[:,None,:].expand(len(classes), seq_len, embed_dim)

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
    example_label_one_hot = nn.functional.one_hot(example_label_id, num_classes=len(classes)).to(torch.float32)

    train_ds = StackDataset(embedding=embeddings, label_one_hot=example_label_one_hot, label_id=example_label_id)
    test_ds = StackDataset(embedding=embeddings, label_one_hot=example_label_one_hot, label_id=example_label_id)

    print("test set size:", len(test_ds))
    print("train set size:", len(train_ds))
    return train_ds, test_ds, classes


def read_embeddings(path: Path, n_parts: int) -> Tuple[Dataset, Dataset, Dict[Label, int]]:
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
    return (train_ds, test_ds, label_idx)


def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, classes: Set[int], device: torch.device):
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
    metrics = classification_metrics(y_pred_logits=torch.cat(all_preds), y_true_one_hot=y_true, classes=classes)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def classification_metrics(y_pred_logits, y_true_one_hot, classes: Set[int]) -> Dict[str, object]:
    """
    y_pred_logits: predicted logits
    y_true_one_hot: true one-hot
    classes: classes to evaluate
    """
    y_true_label = y_true_one_hot.argmax(dim=1)
    y_pred_label = y_pred_logits.argmax(dim=1)
    y_pred_prob = torch.nn.functional.softmax(y_pred_logits, dim=1)

    conf_matrix = confusion_matrix(y_true=y_true_label, y_pred=y_pred_label, labels=np.arange(len(classes))) #labels=np.array(list(classes)))
    true_positives = np.diagonal(conf_matrix).sum()

    # majority class count
    true_class_counts = conf_matrix.sum(axis=1)
    majority_class_count = np.max(true_class_counts)
    majority_class_index = np.argmax(true_class_counts)
    majority_class_predictions = conf_matrix[:, majority_class_index]  # Extract the column of majority class prediction
    majority_class_false = majority_class_predictions[np.arange(len(majority_class_predictions)) != majority_class_index].sum()

    metrics = {
            'f1_micro': f1_score(y_true_label, y_pred_label, average='micro'),
            'roc_auc': roc_auc_score(y_true_one_hot, y_pred_prob, multi_class='ovo', labels=np.array(list(classes))),
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
        print(epoch_n, test_metrics['loss'])
        print(train_metrics)


def run(root: Path,
         model_type: str,
        train_ds:Dataset,
        test_ds:Dataset,
        classes:Set[int],
        out_dir: Optional[Path]=None,
        batch_size: int=128,
        n_epochs: int=30,
        device_str:str = 'cuda'
        ):
    if out_dir is None:
        out_dir = root / Path('runs') / model_type
    out_dir.mkdir(parents=True)

    seq_len, llm_dim = train_ds[0]['embedding'].shape
    print('llm_dim', llm_dim)
    print('context_sz', seq_len)
    print('classes', len(classes))

    device = torch.device(device=device_str)
    if model_type == 'multi_class':
        model, loss_fn = build_model_multi_class_classifier(n_classes=len(classes), llm_dim=llm_dim)
    elif model_type == 'multi_label':
        model, loss_fn = build_model_multi_label_embedding_classifier(n_classes=len(classes), llm_dim=llm_dim, class_weights=torch.ones(len(classes)).to(device))
    elif model_type == 'multi_label_packed':
        model, loss_fn = build_model_multi_label_embedding_classifier_packed(n_classes=len(classes), llm_dim=llm_dim, class_weights=torch.ones(len(classes)).to(device))
    elif model_type == 'mlp':
        model, loss_fn = build_model_multi_class_classifier_mlp(layer_dims=[llm_dim//2**i for i in range(4)], n_classes=len(classes), llm_dim=llm_dim)
    else:
        raise ValueError(f'unknown model {model_type}')

    model = model.to(device)
    # torchinfo.summary(model, input_size=(batch_size, seq_len, llm_dim), device=device)

    print('Training...')
    reporter = Reporter((out_dir / 'history-log.json').open('w'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch_n in range(n_epochs):
        print(f'epoch {epoch_n}...')
        model.train()
        for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            output = model(batch['embedding'].to(device))
            loss = loss_fn(output, batch['label_one_hot'].to(device))
            loss.backward()
            optimizer.step()

        # Save model checkpoint
        torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch_n}.pt")

        # Evaluate loss
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        test_metrics = evaluate(model, test_loader, loss_fn, classes, device)

        train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        train_metrics = evaluate(model, train_eval_loader, loss_fn, classes, device)
        reporter.report(epoch_n, test_metrics, train_metrics)


def main() -> None:
    root = Path('out')
    train_ds, test_ds, classes = read_embeddings(root, 95)
    #train_ds, test_ds, classes = read_embeddings_synthetic()

    # run(root, model_type='multi_class', train_ds=train_ds, test_ds=test_ds, classes=classes)
    run(root, model_type='multi_label_packed', train_ds=train_ds, test_ds=test_ds, classes=classes)
    #run(root, model_type='multi_label', train_ds=train_ds, test_ds=test_ds, classes=classes)
    # run(root, model_type='mlp', train_ds=train_ds, test_ds=test_ds, classes=classes)


if __name__ == '__main__':
    main()

