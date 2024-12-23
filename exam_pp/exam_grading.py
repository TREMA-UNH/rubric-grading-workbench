import asyncio
import gzip
import concurrent.futures

import itertools
from pathlib import Path
from typing import *

from .query_loader import direct_grading_prompts, json_query_loader

from . import question_bank_loader 
from . import question_loader
from .test_bank_prompts import Prompt, QuestionPrompt, NuggetPrompt, get_prompt_classes
from .test_bank_prompts import *
from .t5_qa import *
from .data_model import ExamGrades, FullParagraphData, Grades, SelfRating, dumpQueryWithFullParagraphList, parseQueryWithFullParagraphs
from . import tqa_loader

executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)

def fix_car_query_id(input:List[Tuple[str,List[Prompt]]]) -> List[Tuple[str,List[Prompt]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]


def self_ratings_from_prompt(prompt:Prompt, answer)->SelfRating:
    if prompt.prompt_type() == QuestionPrompt.my_prompt_type:
        return SelfRating(question_id=prompt.prompt_id()
                            , self_rating=prompt.check_answer_rating(answer)
                            ) 
    elif prompt.prompt_type()==NuggetPrompt.my_prompt_type:
        return SelfRating(nugget_id=prompt.prompt_id()
                         , question_id=None
                         , self_rating=prompt.check_answer_rating(answer)
                         ) 
    elif prompt.prompt_type()==DirectGradingPrompt.my_prompt_type:
        return SelfRating(nugget_id=None
                         , question_id=None
                         , self_rating=prompt.check_answer_rating(answer)
                         ) 

    else:
        raise RuntimeError(f"Unknown self rating prompt: {prompt}. \n Prompt-type:{prompt.prompt_type()}")


async def noodle_grading_rubric(queryWithFullParagraphList, grading_prompts:List[Prompt], qaPipeline, max_paragraphs:Optional[int]=None)->None:
    '''Will modify `queryWithFullParagraphList` in place with exam grade annotations from `qaPipeline` on the `questions` set '''

    query_id = queryWithFullParagraphList.queryId
    anyPrompt = grading_prompts[0]

    paragraphs = queryWithFullParagraphList.paragraphs

    async def noodle_one_paragraph(para:FullParagraphData):

            paragraph_id = para.paragraph_id
            paragraph_txt = para.text

            answerTuples = await qaPipeline.grade_paragraph(grading_prompts, paragraph_txt=paragraph_txt)

            for p,answer in answerTuples:
                if isinstance(answer, LlmResponseError):
                    raise RuntimeError("Obtained LlmresponseError {answer}")
                    # todo handle this case

            # for q,a in answerTuples:
                # print(f'{a} -  {q.question}\n{paragraph_txt}\n')

            ratedQs: Optional[List[SelfRating]]
            ratedQs = [self_ratings_from_prompt(prompt=prompt, answer=answer)         
                                for prompt,answer in answerTuples
                                if prompt.has_rating()]
            
            if len(ratedQs)==0:
                ratedQs = None


            correctQs = [(qpc.prompt_id(), answer) for qpc,answer in answerTuples if qpc.check_answer(answer)]
            numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
            numAll = len(answerTuples)
            if numAll > 0: # can't provide exam when no questions are answered.
                print(f"{query_id}, {paragraph_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}. {correctQs}")

                # adding exam data to the JSON file
                exam_grades = ExamGrades( correctAnswered=[qpc.prompt_id() for qpc,answer in answerTuples if qpc.check_answer(answer)]
                                        , wrongAnswered=[qpc.prompt_id() for qpc,answer in answerTuples if not qpc.check_answer(answer)]
                                        , answers = [(qpc.prompt_id(), answer) for qpc,answer in answerTuples ]
                                        , exam_ratio = ((1.0 * numRight) / (1.0*  numAll))
                                        , llm = qaPipeline.exp_modelName()
                                        , llm_options={}
                                        , prompt_type= anyPrompt.prompt_type()
                                        , prompt_info = anyPrompt.prompt_info()
                                        , self_ratings = ratedQs
                                ) 
                if para.exam_grades is None:
                    para.exam_grades = list()
                para.exam_grades.append(exam_grades)

            else:
                print(f'no exam score generated for paragraph {paragraph_id} as numAll=0')
        
    await asyncio.gather( *(noodle_one_paragraph(para) for para in  itertools.islice(paragraphs, max_paragraphs)))
    ## loop = asyncio.get_running_loop()
    # for para in  itertools.islice(paragraphs, max_paragraphs):
    #     ## await loop.run_in_executor(None, noodle_one_paragraph(para))
    #     noodle_one_paragraph(para)


async def noodle_one_query_direct_grading(queryWithFullParagraphList, grading_prompt:Prompt, qaPipeline:LlmPipeline, max_paragraphs:Optional[int]=None)->None:
    '''Will modify `queryWithFullParagraphList` in place with grade annotations from `qaPipeline`  '''

    paragraphs = queryWithFullParagraphList.paragraphs


    async  def noodle_one_paragraph(para):
        paragraph_txt = para.text

        answerTuples = qaPipeline.grade_paragraph([grading_prompt], paragraph_txt=paragraph_txt)
        
        (_, answer) = answerTuples[0]

        if isinstance(answer, LlmResponseError):
            raise RuntimeError("Obtained LlmresponseError {answer}")
        
        else:

            grade_obj = Grades(correctAnswered= grading_prompt.check_answer(answer)
                            , answer=answer
                            , self_ratings= grading_prompt.check_answer_rating(answer)
                            , llm = qaPipeline.exp_modelName()
                            , llm_options={}
                            , prompt_type= grading_prompt.prompt_type()
                            , prompt_info = grading_prompt.prompt_info()
                            )
        
        if para.grades is None:
            para.grades = list()
        para.grades.append(grade_obj)

    await asyncio.gather( *(noodle_one_paragraph(para) for para in itertools.islice(paragraphs, max_paragraphs) ))
    # for para in itertools.islice(paragraphs, max_paragraphs):
    #     noodle_one_paragraph(para)


async def noodle(llmPipeline:LlmPipeline, question_set:Dict[str,List[Prompt]], paragraph_file:Path, out_file:Path, max_queries:Optional[int]=None, max_paragraphs:Optional[int]=None
            , restart_previous_paragraph_file:Optional[Path]=None, restart_from_query:Optional[str]=None
            ):
    with gzip.open(out_file, 'wt', encoding='utf-8') as file:

        query_paragraphs = parseQueryWithFullParagraphs(paragraph_file)

        # restart logic
        restart_previous_query_paragraphs = None
        take_previous_paragraphs = False
        previousQueryWithFullParagraphList = None
        if restart_previous_paragraph_file is not None:
            restart_previous_query_paragraphs = parseQueryWithFullParagraphs(restart_previous_paragraph_file)
            restart_previous_query_paragraphs_iter = itertools.islice(restart_previous_query_paragraphs, max_queries)
            take_previous_paragraphs = True

        for queryWithFullParagraphList in itertools.islice(query_paragraphs, max_queries):
            query_id = queryWithFullParagraphList.queryId
            grading_prompts = question_set.get(query_id)
            if grading_prompts is None or len(grading_prompts)==0:
                print(f'No exam question for query Id {query_id} available. skipping.')
                continue

            # restart logic: check whether we are done copying
            if query_id == restart_from_query:  
                print(f"Restart Logic:  encountered restart query {query_id}. Stopping copying and starting exam grading")
                take_previous_paragraphs = False


            # restart logic: fetch previous
            if take_previous_paragraphs and restart_previous_query_paragraphs is not None:
                previousQueryWithFullParagraphList = next(restart_previous_query_paragraphs_iter, None)
                if previousQueryWithFullParagraphList is None:
                    if restart_from_query is not None:
                        print(f"Restart Logic: restart_previous_query_paragraphs_iter exhausted before we reached restart_query={restart_from_query}), starting to run pipeline" )
                    else:
                        print(f"Restart Logic: restart_previous_query_paragraphs_iter exhausted, starting to run pipeline" )
                    take_previous_paragraphs = False
                elif not previousQueryWithFullParagraphList.queryId == query_id:
                    raise RuntimeError(f"Restart Logic: Query ids out of sequence, obtained {previousQueryWithFullParagraphList.queryId}, but was expecting {query_id}")




            if take_previous_paragraphs and previousQueryWithFullParagraphList is not None: # if restart
                # copy paragraph
                print(f"Restart Logic:  copy query {query_id}")
                queryWithFullParagraphList = previousQueryWithFullParagraphList
                pass
            else:

                any_prompt = grading_prompts[0]
                if any_prompt.prompt_type() == QuestionPrompt.my_prompt_type or any_prompt.prompt_type() == NuggetPrompt.my_prompt_type:
                    # Regular path
                    await  noodle_grading_rubric(queryWithFullParagraphList, grading_prompts, llmPipeline, max_paragraphs)
                elif any_prompt.prompt_type() == DirectGradingPrompt.my_prompt_type:
                    for grading_prompt in grading_prompts: # we expect there to be only one
                        await noodle_one_query_direct_grading(queryWithFullParagraphList, grading_prompt, llmPipeline, max_paragraphs)
                else:
                    raise RuntimeError(f"unknown grading prompt type {any_prompt.prompt_type()}  not matching any of these: {DirectGradingPrompt.my_prompt_type}, {QuestionPrompt.my_prompt_type}, {NuggetPrompt.my_prompt_type}")

            file.write(dumpQueryWithFullParagraphList(queryWithFullParagraphList))
            # out_file.write('\n')
            file.flush()

        llmPipeline.finish()
        file.close()


async def main(cmdargs=None):
    """Score paragraphs by number of questions that are correctly answered."""

    import argparse

    desc = f'''EXAM grading, verifying which paragraphs answer which questions or contain nuggets via LLMs. 
    
            \n
The entries of the given RUBRIC input file will be augmented with exam grades, to be written to a new file
1. Create a RUBRIC inputfile as *JSONL.GZ file. Info about JSON schema with --help-schema
2. Load RUBRIC grading questions via  --question-path $file 
3. Set prompt template via --prompt-class $class
4. Configure the LLM via --name-model $hf_model (as named on huggingface)
5. Different LLM backends and Huggingface pipelines are supported via --model-pipeline these may require additional configuration            
\n
* For vLLM you need to set the url via `export VLLM_URL=http://127.0.0.1:8000/v1`  (also works with ssh port tunnels)
\n
* For OpenAI you need to set the token via `export OPENAI_API_KEY=...`
\n
* For the other pipelines you may need to set the huggingface token via `export HF_TOKEN=...`
             '''
      
    help_schema=f'''The input and output file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             \n
             Create a compatible file with 
             exam_pp.data_model.writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList])
             '''
    
    parser = argparse.ArgumentParser(description="EXAM grading"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('paragraph_file', type=str, metavar='xxx.jsonl.gz'
                        , help='json file with paragraph to grade with exam questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    # MAX_TOKEN_LEN=512
    # MAX_OUT_TOKENS=512

    modelPipelineOpts = {'text2text': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  Text2TextPipeline(model_name, max_token_len=MAX_TOKEN_LEN)
                ,'question-answering': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  QaPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS)
                ,'text-generation': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  TextGenerationPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                , 'llama': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS: LlamaTextGenerationPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS)
                ,'vLLM': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  VllmPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                ,'OpenAI': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  OpenAIPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                }

    parser.add_argument('-o', '--out-file', type=str, metavar='exam-xxx.jsonl.gz', help='Output file name where paragraphs with exam grade annotations will be written to')
    parser.add_argument('--question-path', type=str, metavar='PATH', help='Path to read grading rubric exam questions/nuggets from (can be tqa directory or file)')
    parser.add_argument('--use-nuggets', action='store_true', help="if set, assumed --question-path contains nuggets instead of questions")
    parser.add_argument('--question-type', type=str, choices=['question-bank','direct', 'tqa','genq'], default="question-bank", metavar='PATH', help='Grading rubric file format for reading from --question-path')
    

    parser.add_argument('--model-pipeline', type=str, choices=modelPipelineOpts.keys(), required=True, metavar='MODEL', help='the huggingface pipeline used to answer questions. For example, \'sjrhuschlee/flan-t5-large-squad2\' is designed for the question-answering pipeline, where \'google/flan-t5-large\' is designed for the text2text-generation pipeline. Choices: '+", ".join(modelPipelineOpts.keys()))
    parser.add_argument('--model-name', type=str, metavar='MODEL', help='the huggingface model used to answer questions')
    parser.add_argument('--max-tokens', type=int, metavar="N", default=512, help="total number of tokens for input+output (for generative LLMs, just input)")
    parser.add_argument('--max-out-tokens', type=int, metavar="N", default=512, help="total number of tokens for generated output (not used by some HF pipelines)")


    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))

    parser.add_argument('--custom-prompt', type=str,required=False, metavar="PROMPT_TEXT"
                        , help="Custom question prompt text. Variables {question} and {context} will automatically be filled.")

    parser.add_argument('--custom-prompt-name', type=str,required=False, metavar="NAME"
                        , help="Name for the custom prompt. This name will be used instead of --prompt-class during post-processing and leaderboard evaluation")


    parser.add_argument('--max-queries', type=int, metavar="n", default=-1, help="Limit number of queries to be processed")
    parser.add_argument('--max-paragraphs', type=int, metavar="n", default=-1, help="Limit number of paragraphs to be processed")

    parser.add_argument('--restart-paragraphs-file', type=str, metavar='exam-xxx.jsonl.gz', help='Restart logic: Input file name with partial exam grade annotations that we want to copy from. Copies while queries are defined (unless --restart-from-query is set)')
    parser.add_argument('--restart-from-query', type=str, metavar='QUERY_ID', help='Restart logic: Once we encounter Query Id, we stop copying and start re-running the pipeline (Must also set --restart-paragraphs-file)')
 
    parser.add_argument('--help-schema', action='store_true', help="Additional info on required JSON.GZ input format")


    # Parse the arguments
    args = parser.parse_args(args = cmdargs) 
 
    if args.help_schema:
        print(help_schema)
        sys.exit()

    question_set:Dict[str,List[Prompt]]
    if args.question_type == "tqa":
        question_set = dict(fix_car_query_id( tqa_loader.load_all_tqa_data(tqa_path=Path(args.question_path)
                                                                          , prompt_class=args.prompt_class
                                                                          , self_rater_tolerant = (args.model_pipeline=="llama")
                                                                          )))
    elif args.question_type == 'genq':
        question_set = dict(question_loader.load_naghmehs_question_prompts(args.question_path, prompt_class=args.prompt_class))
    elif args.question_type == 'question-bank':
        question_set = dict(question_bank_loader.load_prompts_from_test_bank(args.question_path
                                                                             , prompt_class=args.prompt_class
                                                                             , use_nuggets=args.use_nuggets
                                                                             , self_rater_tolerant = (args.model_pipeline=="llama")
                                                                             , custom_prompt = args.custom_prompt
                                                                             , custom_prompt_name = args.custom_prompt_name
                                                                             ))
    elif args.question_type == 'direct':
        question_set = direct_grading_prompts(queries=json_query_loader(query_json=args.question_path)
                                              , prompt_class=args.prompt_class
                                              , max_queries=None
                                              , self_rater_tolerant = (args.model_pipeline=="llama")
                                              )
    else:
        raise f"args.question_type \'{args.question_type}\' undefined"
    
    llmPipeline = modelPipelineOpts[args.model_pipeline](args.model_name, args.max_tokens, args.max_out_tokens)

    await noodle(
             llmPipeline=llmPipeline
           , question_set=question_set
           , paragraph_file= args.paragraph_file
           , out_file = args.out_file
           , max_queries = args.max_queries
           , max_paragraphs = args.max_paragraphs
           # Restart logic
           , restart_previous_paragraph_file=args.restart_paragraphs_file, restart_from_query=args.restart_from_query
           )

if __name__ == "__main__":
    asyncio.run(main())
    # main()
