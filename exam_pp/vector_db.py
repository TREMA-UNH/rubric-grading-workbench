import torch as pt
import pandas as pd
import numpy as np
from typing import NewType, List, Optional
from pathlib import Path
import duckdb

VectorId = NewType('VectorId', int)
ClassificationItemId = NewType('ClassificationItemId', int)

SCHEMA = '''
CREATE SEQUENCE tensor_storage_id_seq START 1;
CREATE SEQUENCE tensor_id_seq START 1;
CREATE TABLE tensor (
    tensor_id INTEGER PRIMARY KEY DEFAULT nextval('tensor_id_seq'),
    tensor_storage_id integer,
    index_n integer NOT NULL,   -- index within backing store
);

CREATE SEQUENCE classification_item_id_seq START 1;
CREATE TABLE classification_item (
    classification_item_id INTEGER PRIMARY KEY DEFAULT nextval('classification_item_id_seq'),
    metadata JSON,
);

CREATE SEQUENCE classification_feature_id_seq START 1;
CREATE TABLE classification_feature (
    classification_feature_id INTEGER PRIMARY KEY DEFAULT nextval('classification_feature_id_seq'),
    classification_item_id integer references classification_item(classification_item_id),
    tensor_id integer references tensor(tensor_id),
    metadata JSON,
);

CREATE SEQUENCE label_assignment_id_seq START 1;
CREATE TABLE label_assignment (
    label_assignment_id INTEGER PRIMARY KEY DEFAULT nextval('label_assignment_id_seq'),
    classification_item_id integer references classification_item(classification_item_id),
    true_labels text[],
);
'''

class EmbeddingDb:
    def __init__(self, path: Path):
        self.tensor_dir = path / "tensors"
        needs_init = not path.exists()
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(path / 'embeddings.duckdb')
        if needs_init:
            self.db.execute(SCHEMA)
        self.storage_cache = {}


    def _storage_path(self, tsid: int) -> Path:
        return self.tensor_dir / f'{tsid:08x}.pt'


    def _insert_tensors(self, tensors: pt.Tensor) -> List[VectorId]:
        self.db.begin()
        self.db.execute("SELECT nextval('tensor_storage_id_seq');")
        tsid, = self.db.fetchone()
        path = self._storage_path(tsid)
        try:
            pt.save(tensors, path)
            df = pd.DataFrame(data={
                'index_n': np.arange(tensors.shape[0]),
                'tensor_storage_id': tsid,
            })
            self.db.execute(
                    '''
                    INSERT INTO tensor (tensor_storage_id, index_n)
                    (SELECT tensor_storage_id, index_n FROM df)
                    RETURNING tensor_id;
                    ''')
            tensor_id0, = self.db.fetchone()
            self.db.commit()
            tensor_ids = list(range(tensor_id0, tensor_id0 + tensors.shape[0]))
            return tensor_ids
        except Exception as e:
            path.unlink()
            self.db.rollback()
            raise e


    def fetch_tensors(self, tensor_ids: List[VectorId], token_length:Optional[int]) -> pt.Tensor:
        tensor_ids = pd.DataFrame(data={'tensor_id': tensor_ids})
        tensor_ids['i'] = tensor_ids.index
        self.db.execute('''
            SELECT needles.i, tensor_storage_id, index_n
            FROM tensor
            INNER JOIN (SELECT * FROM tensor_ids) AS needles ON tensor.tensor_id = needles.tensor_id
            ORDER BY needles.i ASC;
        ''')
        vs = self.db.df()
        out = None
        for v in vs.itertuples():
            tsid = v.tensor_storage_id
            if tsid not in self.storage_cache:
                self.storage_cache[tsid] = pt.load(self._storage_path(tsid), weights_only=True)

            t = self.storage_cache[tsid][v.index_n]
            if out is None:
                size = (len(tensor_ids),) + t.shape
                out = pt.empty(size=size, dtype=pt.float)
            else:
                assert t.shape == out.shape[1:]

            out[v.i] = t

        assert out is not None
        return out


    def add_embeddings(self,
                       # entries marked with "# <-" mark the set that denotes a unique entry.
            query_id: str, # <-
            passage_id: str, # <-
            prompt_class: str, # <- 
            prompt_texts: List[str],
            test_bank_ids: List[str], # <-
            answers: List[str],
            embeddings: pt.Tensor,
            true_labels: Optional[List[str]]
            ):
        #raise RuntimeError('uh oh') # XXX
        print(f"Recording {query_id} {passage_id} embeddings:{embeddings.shape}")
        tensor_ids = self._insert_tensors(embeddings)

        metadata = {
            'query': query_id,
            'passage': passage_id,
        }
        self.db.execute(
            '''
            INSERT INTO classification_item (metadata)
            VALUES (?)
            RETURNING classification_item_id;
            ''',
            (metadata,))
        ci_id, = self.db.fetchone()
        # print('hello', ci_id, metadata)

        self.db.executemany(
            '''
            INSERT INTO classification_feature
            (classification_item_id, tensor_id, metadata)
            VALUES (?, ?, ?);
            ''',
            [(ci_id, tid, {
                'text': text,
                'prompt_class': prompt_class,
                'test_bank': test_bank,
                'answer': answer,
              })
             for tid, text, test_bank, answer in zip(tensor_ids, prompt_texts, test_bank_ids, answers)
            ])

        if true_labels is not None:
            self.db.execute(
                '''
                INSERT INTO label_assignment
                (classification_item_id, true_labels)
                VALUES (?,?)
                ''',
                (ci_id, true_labels))


def main() -> None:
    p = Path('test.vecdb')
    dims = (512,1024)
    N = 64

    db = EmbeddingDb(p)
    accum = []
    for i in range(8):
        t = pt.rand((N,) + dims)
        t[:,:,:] = pt.arange(N)[:,None,None] + i*N
        print(t.shape)
        tensor_ids = db._insert_tensors(t)
        print(tensor_ids)
        accum += tensor_ids

    xs = db.fetch_tensors(accum)
    print(xs)

if __name__ == '__main__':
    main()
