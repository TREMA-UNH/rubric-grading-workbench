from pathlib import Path
from typing import Dict, List
from exam_pp.exam_cover_metric import compute_exam_cover_scores, write_exam_results, ExamCoverScorerFactory
from exam_pp.parse_qrels_runs_with_text import GradeFilter, QueryWithFullParagraphList, parseQueryWithFullParagraphs

# Read Graded Exam files, compute EXAM Cover Evaluation Metric
query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(Path("exam-graded.jsonl.gz"))
exam_factory = ExamCoverScorerFactory(GradeFilter.noFilter(), min_self_rating=None)
resultsPerMethod = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory, rank_cut_off=20)


# Print Exam Cover Evaluation Scores
for examEval in resultsPerMethod.values():
    print(f'{examEval.method}  exam={examEval.examScore:0.2f}+/-{examEval.examScoreStd:0.2f} \t  n-exam={examEval.nExamScore:0.2f}')

examEvaluationPerQuery:Dict[str,float] = resultsPerMethod['my_method'].examCoverPerQuery

# Export Exam Cover Evaluation Scores
write_exam_results("exam-eval.jsonl.gz", resultsPerMethod)