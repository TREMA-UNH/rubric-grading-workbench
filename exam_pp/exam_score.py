# exam_pp/exam_score.py


from question_types import *
from t5_qa import *
from parse_qrels_runs_with_text import *
import tqa_loader


def fix_car_query_id(input:List[Tuple[str,List[QuestionPromptWithChoices]]]) -> List[Tuple[str,List[QuestionPromptWithChoices]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]


def main():
    """Entry point for the module."""
    lesson_questions:Dict[str,List[QuestionPromptWithChoices]] = dict(fix_car_query_id(tqa_loader.load_all_tqa_data()))
    # print('question bank query ids', lesson_questions.keys())
    batchPipe = BatchingPipeline(BATCH_SIZE)

    with gzip.open("output.jsonl.gz", 'wt', encoding='utf-8') as out_file:

        query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs("./benchmarkY3test-qrels-with-text.jsonl.gz")
        qa = McqaPipeline()

        for queryWithFullParagraphList in query_paragraphs:
            query_id = queryWithFullParagraphList.queryId
            questions = lesson_questions.get(query_id)
            if questions is None:
                print(f'No exam question for query Id {query_id} available. skipping.')
                continue

            paragraphs = queryWithFullParagraphList.paragraphs
            for para in paragraphs:
                paragraph_id = para.paragraph_id
                paragraph_txt = para.text

                promptGenerator=lambda qpc: qpc.generate_prompt_with_context_no_choices(paragraph_txt, model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)
                # promptGenerator=lambda qpc: qpc.generate_prompt_with_context(paragraph_txt)
                answerTuples = batchPipe.chunkingBatchAnswerQuestions(questions, qa, promptGenerator)
                correctQs = [(qpc.question_id, answer) for qpc,answer in answerTuples if qpc.check_answer(answer)]
                numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
                numAll = len(answerTuples)
                print(f"{query_id}, {paragraph_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}. {correctQs}")

                # adding exam data to the JSON file
                exam_grades = ExamGrades( correctAnswered=[qpc.question_id for qpc,answer in answerTuples if qpc.check_answer(answer)]
                                        , wrongAnswered=[qpc.question_id for qpc,answer in answerTuples if not qpc.check_answer(answer)]
                                        , answers = [(qpc.question_id, answer) for qpc,answer in answerTuples ]
                                        , exam_ratio = ((1.0 * numRight) / (1.0*  numAll))
                                        , llm = qa.modelName
                                        , llm_options={"prompt_template":"generate_prompt_with_context_no_choices", "answer_match":"lowercase, stemmed, fuzz > 0.8"}
                                ) 
                if para.exam_grades is None:
                    para.exam_grades = list()
                para.exam_grades.append(exam_grades)
            out_file.write(dumpQueryWithFullParagraphList(queryWithFullParagraphList))
            out_file.write('\n')
            out_file.flush()

        out_file.close()
if __name__ == "__main__":
    main()
