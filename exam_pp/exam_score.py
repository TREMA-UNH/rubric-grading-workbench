# exam_pp/exam_score.py


from question_types import QuestionPromptWithChoices
from question_types import *
from t5_qa import *
from parse_qrels_runs_with_text import *
import tqa_loader
from typing import *


def fix_car_query_id(input:List[Tuple[str,List[QuestionPromptWithChoices]]]) -> List[Tuple[str,List[QuestionPromptWithChoices]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]


def main():
    """Score paragraphs by number of questions that are correctly answered."""

    import argparse

    desc = f'''EXAM grading, verifying which paragraphs answer which questions with a Q/A system. \n
              The input and output file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             '''
    
    parser = argparse.ArgumentParser(description="EXAM grading"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paragraph_file', type=str, metavar='xxx.jsonl.gz'
                        , help='json file with paragraph to grade with exam questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('-o', '--out-file', type=str, metavar='exam-xxx.jsonl.gz', help='Output file name where paragraphs with exam grade annotations will be written to')
    parser.add_argument('--max-queries', type=int, metavar='INT', default=None, help='limit the number of queries that will be processed (for debugging)')
    parser.add_argument('--max-paragraphs', type=int, metavar='INT', default=None, help='limit the number of paragraphs that will be processed (for debugging)')

    # Parse the arguments
    args = parser.parse_args()  

    lesson_questions:Dict[str,List[QuestionPromptWithChoices]] = dict(fix_car_query_id(tqa_loader.load_all_tqa_data()))
    # print('question bank query ids', lesson_questions.keys())
    batchPipe = BatchingPipeline(BATCH_SIZE)

    with gzip.open(args.out_file, 'wt', encoding='utf-8') as out_file:

        query_paragraphs = parseQueryWithFullParagraphs(args.paragraph_file)
        qa = McqaPipeline()

        for queryWithFullParagraphList in itertools.islice(query_paragraphs, args.max_queries):
            query_id = queryWithFullParagraphList.queryId
            questions = lesson_questions.get(query_id)
            if questions is None:
                
                print(f'No exam question for query Id {query_id} available. skipping.')
                continue

            paragraphs = queryWithFullParagraphList.paragraphs
            for para in itertools.islice(paragraphs, args.max_paragraphs):
                paragraph_id = para.paragraph_id
                paragraph_txt = para.text

                # promptGenerator=lambda qpc: qpc.generate_prompt_with_context_no_choices(paragraph_txt, model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)
                promptGeneratorQC=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(paragraph_txt, model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)
                # promptGenerator=lambda qpc: qpc.generate_prompt_with_context(paragraph_txt)
                answerTuples = batchPipe.chunkingBatchAnswerQuestions(questions, qa, promptGeneratorQC)
                correctQs = [(qpc.question_id, answer) for qpc,answer in answerTuples if qpc.check_answer(answer)]
                numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
                numAll = len(answerTuples)
                if numAll > 0: # can't provide exam when no questions are answered.
                    print(f"{query_id}, {paragraph_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}. {correctQs}")

                    # adding exam data to the JSON file
                    exam_grades = ExamGrades( correctAnswered=[qpc.question_id for qpc,answer in answerTuples if qpc.check_answer(answer)]
                                            , wrongAnswered=[qpc.question_id for qpc,answer in answerTuples if not qpc.check_answer(answer)]
                                            , answers = [(qpc.question_id, answer) for qpc,answer in answerTuples ]
                                            , exam_ratio = ((1.0 * numRight) / (1.0*  numAll))
                                            , llm = qa.exp_modelName()
                                            , llm_options={"prompt_template":"generate_prompt_with_context_QC_no_choices", "answer_match":"lowercase, stemmed, fuzz > 0.8"}
                                    ) 
                    if para.exam_grades is None:
                        para.exam_grades = list()
                    para.exam_grades.append(exam_grades)

                else:
                    print(f'no exam score generated for paragraph {paragraph_id} as numAll=0')
            out_file.write(dumpQueryWithFullParagraphList(queryWithFullParagraphList))
            out_file.write('\n')
            out_file.flush()

        out_file.close()
if __name__ == "__main__":
    main()
