from pathlib import Path
from typing import Dict, List
from exam_pp.exam_cover_metric import compute_exam_cover_scores, write_exam_results, ExamCoverScorerFactory
from exam_pp.parse_qrels_runs_with_text import GradeFilter, QueryWithFullParagraphList, parseQueryWithFullParagraphs

# Read Graded Exam files, compute EXAM Cover Evaluation Metric

exam_factory = ExamCoverScorerFactory(GradeFilter.noFilter(), min_self_rating=None)

query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(Path("exam-graded.jsonl.gz"))

resultsPerMethod = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory, rank_cut_off=20)

print(f'plain EXAM: {resultsPerMethod["my_method"].examScore}')
print(f'normalized EXAM:  {resultsPerMethod["my_method"].nExamScore}')

examEvaluationPerQuery:Dict[str,float] = resultsPerMethod['my_method'].examCoverPerQuery

write_exam_results("exam-eval.jsonl.gz", resultsPerMethod)