import ir_datasets
dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>
    print(query)

    # sad trombone:  TREC DL organizers moved files around, and ir_datasets would need to update paths first.
