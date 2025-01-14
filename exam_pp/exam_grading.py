import asyncio
import gzip
import concurrent.futures

import itertools
from pathlib import Path
from typing import *

from .vector_db import EmbeddingDb

from .query_loader import direct_grading_prompts, json_query_loader

from . import question_bank_loader 
from . import question_loader
from .test_bank_prompts import Prompt, QuestionPrompt, NuggetPrompt, get_prompt_classes
from .test_bank_prompts import *
from .t5_qa import *
from .data_model import ExamGrades, FullParagraphData, Grades, QueryWithFullParagraphList, SelfRating, dumpQueryWithFullParagraphList, parseQueryWithFullParagraphs
from . import tqa_loader
from .async_utils import *

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


def noodle_grading_rubric(queryWithFullParagraphList:QueryWithFullParagraphList, grading_prompts:List[Prompt]
                                , llmPipeline:LlmPipeline, max_paragraphs:Optional[int]=None 
                                , keep_going_on_llm_parse_error:bool=False
                                , system_message:Optional[str]=None
                                , embedding_db:Optional[EmbeddingDb]=None
                                , **kwargs
                                )->None:
    '''Will modify `queryWithFullParagraphList` in place with exam grade annotations from `qaPipeline` on the `questions` set '''

    query_id = queryWithFullParagraphList.queryId
    anyPrompt = grading_prompts[0]
    prompt_class=anyPrompt.prompt_info()["prompt_class"]

    paragraphs = queryWithFullParagraphList.paragraphs


    async def noodle_one_paragraph(para:FullParagraphData) -> None:
        paragraph_id = para.paragraph_id
        paragraph_txt = para.text

        print(f"Query: {query_id} / Para: {para.paragraph_id}")

        def record_embeddings(prompts:List[str], embeddings:torch.Tensor, answers:List[str]):
            if embedding_db is not None:
                # print(f"Recording Embedding")
                true_relevance = None
                judg = para.get_any_judgment()
                if judg is not None:
                    true_relevance = [f"{judg.relevance}"]
                embedding_db.add_embeddings(query_id=query_id
                                            , passage_id=paragraph_id
                                            , prompt_class=prompt_class
                                            , test_bank_ids = [p.prompt_id() for p in grading_prompts]
                                            , prompt_texts=prompts
                                            , embeddings=embeddings
                                            , answers=answers
                                            , true_labels=true_relevance)



        answer_or_error_tuples = await llmPipeline.grade_paragraph(grading_prompts
                                                                   , paragraph_txt=paragraph_txt
                                                                   , full_paragraph=para
                                                                   , system_message=system_message
                                                                   , record_embeddings=record_embeddings
                                                                   , **kwargs)
        

        # print(answer_or_error_tuples)
        answerTuples:List[Tuple[Prompt,str]] = [ (p,cast(str,a))  for p,a in answer_or_error_tuples if not isinstance(a, LlmResponseError)]
        just_answer_errors:List[Tuple[Prompt,LlmResponseError]]  = [ (p,cast(LlmResponseError,a))  for p,a in answer_or_error_tuples if  isinstance(a,LlmResponseError)]

        # print(f"Received {len(answer_or_error_tuples)} answers from LLM")            

        if not keep_going_on_llm_parse_error:
            for p,llm_error in just_answer_errors:
                raise RuntimeError(f"Obtained LlmResponseError for paragraph id {paragraph_id}:  {llm_error}",llm_error)
        else:
            for p,llm_error in just_answer_errors:
                print(f"Obtained LlmResponseError for paragraph id {paragraph_id}:  {llm_error}",llm_error)

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
        numAll = len(answer_or_error_tuples)
        if numAll > 0: # can't provide exam when no questions are answered.
            print(f"{query_id}, {paragraph_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}. {correctQs}")

            correct_answered = [qpc.prompt_id() for qpc,answer in answerTuples if qpc.check_answer(answer)]
            # adding exam data to the JSON file
            exam_grades = ExamGrades( correctAnswered=correct_answered
                                    , wrongAnswered=[qpc.prompt_id() for qpc,answer in answerTuples if not qpc.check_answer(answer)]
                                    , answers = [(qpc.prompt_id(), answer) for qpc,answer in answerTuples]
                                    , llm_response_errors= {qpc.prompt_id(): llm_error for qpc,llm_error in just_answer_errors}
                                    , exam_ratio = ((1.0 * numRight) / (1.0*  numAll))
                                    , llm = llmPipeline.exp_modelName()
                                    , llm_options={}
                                    , prompt_type= anyPrompt.prompt_type()
                                    , prompt_info = anyPrompt.prompt_info()
                                    , self_ratings = ratedQs
                                    ) 
            if para.exam_grades is None:
                para.exam_grades = list()
            
            if ratedQs is not None:
                exam_grades.relevance_label = max([rating.self_rating for rating in ratedQs])
            else:
                exam_grades.relevance_label = len(correct_answered)

            para.exam_grades.append(exam_grades)

        else:
            print(f'no exam score generated for paragraph {paragraph_id} as no prompt is generated')

    # async def run_all_paras():
    #         for para in (itertools.islice(paragraphs, max_paragraphs) if max_paragraphs > 0 else paragraphs):
    #             tg.create_task(noodle_one_paragraph(para))

    asyncio.run ( apply_concurrently( noodle_one_paragraph, (itertools.islice(paragraphs, max_paragraphs) if max_paragraphs > 0 else paragraphs), n_workers = 20))
    

    # for para in (itertools.islice(paragraphs, max_paragraphs) if max_paragraphs > 0 else paragraphs):
    #         await noodle_one_paragraph(para)


def noodle_one_query_direct_grading(queryWithFullParagraphList, grading_prompt:Prompt
                                          , llmPipeline:LlmPipeline, max_paragraphs:Optional[int]=None
                                          , keep_going_on_llm_parse_error:bool=False
                                          , system_message:Optional[str]=None
                                          , embedding_db:Optional[EmbeddingDb]=None
                                          , **kwargs
                                          )->None:
    query_id = queryWithFullParagraphList.queryId
    paragraphs = queryWithFullParagraphList.paragraphs

    async def noodle_one_paragraph(para)-> None:
        paragraph_id = para.paragraph_id
        paragraph_txt = para.text

        print(f"Query: {query_id} / Para: {para.paragraph_id}")

        def record_embeddings(prompts:List[str], embeddings:torch.Tensor, answers:List[str]):
            if embedding_db is not None:
                true_relevance = None
                judg = para.get_any_judgment()
                if judg is not None:
                    true_relevance = [f"{judg.relevance}"]
                embedding_db.add_embeddings(query_id=queryWithFullParagraphList.queryId
                                            , passage_id=para.paragraph_id
                                            , prompt_class=grading_prompt.prompt_info()["prompt_class"]
                                            , test_bank_ids = [grading_prompt.prompt_id()]
                                            , prompt_texts=prompts
                                            , embeddings=embeddings
                                            , answers=answers
                                            , true_labels=true_relevance)



        answer_or_error_tuples = await llmPipeline.grade_paragraph([grading_prompt]
                                                        , paragraph_txt=paragraph_txt
                                                        , full_paragraph=para
                                                        , system_message=system_message
                                                        , record_embeddings=record_embeddings
                                                        , **kwargs)
        
        # print(answer_or_error_tuples)
        answerTuples:List[Tuple[Prompt,str]] = [ (p,cast(str,a))  for p,a in answer_or_error_tuples if not isinstance(a, LlmResponseError)]
        just_answer_errors:List[Tuple[Prompt,LlmResponseError]]  = [ (p,cast(LlmResponseError,a))  for p,a in answer_or_error_tuples if  isinstance(a,LlmResponseError)]
        just_answer_error = just_answer_errors[0][1] if len(just_answer_errors)>0 else None



        if not keep_going_on_llm_parse_error:
            for p,llm_error in just_answer_errors:
                raise RuntimeError(f"Obtained LlmResponseError for paragraph id {paragraph_id}:  {llm_error}",llm_error)
        else:
            for p,llm_error in just_answer_errors:
                print(f"Obtained LlmResponseError for paragraph id {paragraph_id}:  {llm_error}",llm_error)



        (_, answer) = answerTuples[0] if len(answerTuples)>0 else (None, "")

        # grade_obj = None
        # if not keep_going_on_llm_parse_error:
        #     if isinstance(answer, LlmResponseError):
        #         raise RuntimeError("Obtained LlmresponseError {answer}")
        
        # else:
        grade_obj = Grades(correctAnswered= grading_prompt.check_answer(answer)
                        , answer=answer
                        # , llm_response_error= (cast(LlmResponseError,answer).failure_reason) if isinstance(answer, LlmResponseError) else None
                        , llm_response_error= just_answer_error
                        # , llm_response_errors= {qpc.prompt_id(): llm_error for qpc,llm_error in just_answer_errors}
                        , self_ratings= grading_prompt.check_answer_rating(answer)
                        , llm = llmPipeline.exp_modelName()
                        , llm_options={}
                        , prompt_type= grading_prompt.prompt_type()
                        , prompt_info = grading_prompt.prompt_info()
                        )
        
        if grade_obj.self_ratings is not None:
            grade_obj.relevance_label = grade_obj.self_ratings
        else:
            grade_obj.relevance_label = 1 if grade_obj.correctAnswered else 0



        if para.grades is None:
            para.grades = list()
        para.grades.append(grade_obj)

    # async with asyncio.TaskGroup() as tg:
    #     for para in itertools.islice(paragraphs, max_paragraphs):
    #         tg.create_task(noodle_one_paragraph(para))

    # async def run_all_paras():
    #     async with asyncio.TaskGroup() as tg:
    #         for para in (itertools.islice(paragraphs, max_paragraphs) if max_paragraphs > 0 else paragraphs):
    #             tg.create_task(noodle_one_paragraph(para))
    
    asyncio.run ( apply_concurrently( noodle_one_paragraph, (itertools.islice(paragraphs, max_paragraphs) if max_paragraphs > 0 else paragraphs), n_workers = 20))


    # asyncio.run (run_all_paras())


def noodle(llmPipeline:LlmPipeline, question_set:Dict[str,List[Prompt]], paragraph_file:Path, out_file:Path, max_queries:Optional[int]=None, max_paragraphs:Optional[int]=None
            , restart_previous_paragraph_file:Optional[Path]=None, restart_from_query:Optional[str]=None
            , keep_going_on_llm_parse_error:bool=False
            , system_message:Optional[str]=None
            , embedding_db:Optional[EmbeddingDb]=None
            , **kwargs
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

        for queryWithFullParagraphList in (itertools.islice(query_paragraphs, max_queries) if max_queries>0 else query_paragraphs):
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
                    noodle_grading_rubric( queryWithFullParagraphList= queryWithFullParagraphList
                                                , grading_prompts= grading_prompts
                                                , llmPipeline= llmPipeline
                                                , max_paragraphs= max_paragraphs
                                                , keep_going_on_llm_parse_error=keep_going_on_llm_parse_error
                                                , system_message=system_message
                                                , embedding_db=embedding_db
                                                , **kwargs)
                elif any_prompt.prompt_type() == DirectGradingPrompt.my_prompt_type:
                    for grading_prompt in grading_prompts: # we expect there to be only one
                        noodle_one_query_direct_grading(queryWithFullParagraphList= queryWithFullParagraphList
                                                            , grading_prompt= grading_prompt
                                                            , llmPipeline= llmPipeline
                                                            , max_paragraphs= max_paragraphs
                                                            , keep_going_on_llm_parse_error=keep_going_on_llm_parse_error
                                                            , system_message=system_message
                                                            , embedding_db = embedding_db
                                                            , **kwargs)
                else:
                    raise RuntimeError(f"unknown grading prompt type {any_prompt.prompt_type()}  not matching any of these: {DirectGradingPrompt.my_prompt_type}, {QuestionPrompt.my_prompt_type}, {NuggetPrompt.my_prompt_type}")

            file.write(dumpQueryWithFullParagraphList(queryWithFullParagraphList))
            # out_file.write('\n')
            file.flush()

        llmPipeline.finish()
        file.close()


def main(cmdargs=None):
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

    parser.add_argument('--llm-api-key', type=str, metavar='KEY'
                        , help='Set API key for LLM backend'
                        , required=False
                        )
    parser.add_argument('--llm-base-url', type=str, metavar='URL'
                        , required=False
                        , help='URL of the LLM backend. Must be an endpoint for a Chat Completions protocol.'
                        )
    parser.add_argument('--llm-temperature', type=float, metavar='t'
                        , required=False
                        , help='Temperature passed to LLM backend.'
                        )
    parser.add_argument('--llm-stop-tokens', nargs='+', type=str, metavar='STR'
                        , required=False
                        , help='One (or more) stop tokens'
                        )
    # MAX_TOKEN_LEN=512
    # MAX_OUT_TOKENS=512

    
    modelPipelineOpts = {'text2text': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs:  Text2TextPipeline(model_name, max_token_len=MAX_TOKEN_LEN)
                ,'question-answering': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs:  QaPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS)
                ,'text-generation': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs:  TextGenerationPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                , 'llama': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs: LlamaTextGenerationPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS)
                ,'vLLM': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs:  VllmPipelineOld(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                ,'OpenAI': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs:  OpenAIPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                , 'embed-text2text': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs:  EmbeddingText2TextPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                , 'chat-completions': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS, **kwargs: 
                     ChatCompletionsPipeline(model_name, max_token_len=MAX_TOKEN_LEN,**kwargs)  # pass in additional config paremeters as **kwargs
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
    parser.add_argument('--embedding-db', type=str, metavar='PATH', help='Path for the database directory for recording embedding vectors')


    parser.add_argument('--max-queries', type=int, metavar="n", default=-1, help="Limit number of queries to be processed")
    parser.add_argument('--max-paragraphs', type=int, metavar="n", default=-1, help="Limit number of paragraphs to be processed")

    parser.add_argument('--restart-paragraphs-file', type=str, metavar='exam-xxx.jsonl.gz', help='Restart logic: Input file name with partial exam grade annotations that we want to copy from. Copies while queries are defined (unless --restart-from-query is set)')
    parser.add_argument('--restart-from-query', type=str, metavar='QUERY_ID', help='Restart logic: Once we encounter Query Id, we stop copying and start re-running the pipeline (Must also set --restart-paragraphs-file)')
    parser.add_argument('-k','--keep-going-on-llm-parse-error', action='store_true', help="Keep going even when parsing of LLM-responses fail. Errors will be logged in ExamGrades/Grades object, but the program will not stop with a raised LlmResponseError")
 
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
    

    pipeline_args = {}
    if args.llm_api_key is not None or args.llm_base_url is not None:
        # chat_completions_client = openai_interface.default_openai_client()
        # chat_completions_client = openai_interface.createOpenAIClient(api_key=args.llm_api_key, base_url=args.llm_base_url)
        chat_completions_client = openai.AsyncOpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
        # openai_interface.createOpenAIClient(api_key=args.llm_api_key, base_url=args.llm_base_url)
        pipeline_args["client"]=chat_completions_client

        model_params = dict()
        model_params["max_completion_tokens"]=args.max_out_tokens
        model_params["temperature"]=args.llm_temperature
        print("stop tokens:", ", ".join(args.llm_stop_tokens))
        model_params["stop"] = args.llm_stop_tokens # e.g. for llama models:  ["<|eot_id|>","<|eom_id|>"]

        pipeline_args["model_params"] = model_params

    llmPipeline = modelPipelineOpts[args.model_pipeline](args.model_name, args.max_tokens, args.max_out_tokens, **pipeline_args)
    
    embedding_db = None
    if args.embedding_db is not None:
        embedding_db = EmbeddingDb(Path(args.embedding_db),write=True)


    noodle(
             llmPipeline=llmPipeline
           , question_set=question_set
           , paragraph_file= args.paragraph_file
           , out_file = args.out_file
           , max_queries = args.max_queries
           , max_paragraphs = args.max_paragraphs
           # Restart logic
           , restart_previous_paragraph_file=args.restart_paragraphs_file, restart_from_query=args.restart_from_query
           , keep_going_on_llm_parse_error=args.keep_going_on_llm_parse_error
           , embedding_db=embedding_db
           )

if __name__ == "__main__":
    # asyncio.run(main())
    main()
