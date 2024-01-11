

import question_loader
from question_types import QuestionPrompt, get_prompt_classes
from question_types import *
from t5_qa import *
from parse_qrels_runs_with_text import *
import tqa_loader
from typing import *


def fix_car_query_id(input:List[Tuple[str,List[QuestionPrompt]]]) -> List[Tuple[str,List[QuestionPrompt]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]


def runSquadQA(qa, questions, paragraph_txt, model_tokenizer, max_token_len):
    # promptGenerator=lambda qpc: qpc.generate_prompt_with_context_no_choices(paragraph_txt, model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)
    promptGeneratorQC=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(paragraph_txt, model_tokenizer = model_tokenizer, max_token_len = max_token_len)
    # promptGenerator=lambda qpc: qpc.generate_prompt_with_context(paragraph_txt)
    answerTuples = qa.chunkingBatchAnswerQuestions(questions, paragraph_txt)
    return answerTuples

def runT2TQA(qa, questions, paragraph_txt, model_tokenizer, max_token_len):
    promptGenerator=lambda qpc: qpc.generate_prompt_with_context_no_choices(paragraph_txt, model_tokenizer = model_tokenizer, max_token_len = max_token_len)
    # promptGeneratorQC=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(paragraph_txt, model_tokenizer = model_tokenizer, max_token_len = max_token_len)
    # promptGenerator=lambda qpc: qpc.generate_prompt_with_context(paragraph_txt)
    answerTuples = qa.chunkingBatchAnswerQuestions(questions, paragraph_txt=paragraph_txt)
    return answerTuples


def noodle(qaPipeline, question_set, paragraph_file, out_file, max_queries, max_paragraphs):
    with gzip.open(out_file, 'wt', encoding='utf-8') as out_file:

        query_paragraphs = parseQueryWithFullParagraphs(paragraph_file)



        for queryWithFullParagraphList in itertools.islice(query_paragraphs, max_queries):
            query_id = queryWithFullParagraphList.queryId
            questions = question_set.get(query_id)
            if questions is None or len(questions)==0:
                print(f'No exam question for query Id {query_id} available. skipping.')
                continue
            anyQpc = questions[0]

            paragraphs = queryWithFullParagraphList.paragraphs
            for para in itertools.islice(paragraphs, max_paragraphs):
                paragraph_id = para.paragraph_id
                paragraph_txt = para.text

                answerTuples = qaPipeline.chunkingBatchAnswerQuestions(questions, paragraph_txt=paragraph_txt)
                # for q,a in answerTuples:
                    # print(f'{a} -  {q.question}\n{paragraph_txt}\n')

                ratedQs: Optional[List[SelfRating]]
                ratedQs = [SelfRating(question_id=qpc.question_id
                                    , self_rating=qpc.check_answer_rating(answer)) 
                                    for qpc,answer in answerTuples
                                    if qpc.has_rating()]
                
                if len(ratedQs)==0:
                    ratedQs = None


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
                                            , llm = qaPipeline.exp_modelName()
                                            , llm_options={"prompt_template":"generate_prompt_with_context_QC_no_choices", "answer_match":"lowercase, stemmed, fuzz > 0.8"}
                                            , prompt_info = anyQpc.prompt_info()
                                            , self_ratings = ratedQs
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

    modelPipelineOpts = {'text2text': lambda model_name:  Text2TextPipeline(model_name)
                ,'question-answering': lambda model_name:  QaPipeline(model_name)
                ,'text-generation': lambda model_name:  TextGenerationPipeline(model_name) 
                }

    parser.add_argument('-o', '--out-file', type=str, metavar='exam-xxx.jsonl.gz', help='Output file name where paragraphs with exam grade annotations will be written to')
    parser.add_argument('--question-path', type=str, metavar='PATH', help='Path to read exam questions from (can be tqa directory or file)')
    parser.add_argument('--question-type', type=str, choices=['tqa','naghmeh'], required=True, metavar='PATH', help='Question type to read from question-path')
    

    parser.add_argument('--max-queries', type=int, metavar='INT', default=None, help='limit the number of queries that will be processed (for debugging)')
    parser.add_argument('--max-paragraphs', type=int, metavar='INT', default=None, help='limit the number of paragraphs that will be processed (for debugging)')
    parser.add_argument('--model-pipeline', type=str, choices=modelPipelineOpts.keys(), required=True, metavar='MODEL', help='the huggingface pipeline used to answer questions. For example, \'sjrhuschlee/flan-t5-large-squad2\' is designed for the question-answering pipeline, where \'google/flan-t5-large\' is designed for the text2text-generation pipeline. Choices: '+", ".join(modelPipelineOpts.keys()))
    parser.add_argument('--model-name', type=str, metavar='MODEL', help='the huggingface model used to answer questions')

    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))

    # Parse the arguments
    args = parser.parse_args()  

    question_set:Dict[str,List[QuestionPrompt]]
    if args.question_type == "tqa":
        question_set = dict(fix_car_query_id(tqa_loader.load_all_tqa_data(Path(args.question_path), prompt_class=args.prompt_class)))
    elif args.question_type == 'naghmeh':
        question_set = dict(question_loader.load_naghmehs_questions(args.question_path, prompt_class=args.prompt_class))
    else:
        raise f"args.question_type \'{args.question_type}\' undefined"
    
    qaPipeline = modelPipelineOpts[args.model_pipeline](args.model_name)

    noodle(qaPipeline=qaPipeline
           , question_set=question_set
           , paragraph_file= args.paragraph_file
           , out_file = args.out_file
           , max_queries = args.max_queries
           , max_paragraphs = args.max_paragraphs
           )

if __name__ == "__main__":
    main()
