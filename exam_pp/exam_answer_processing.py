
import asyncio
import gzip
import concurrent.futures

import itertools
from pathlib import Path
from typing import *

from .exam_grading import fix_car_query_id


from .query_loader import direct_grading_prompts, json_query_loader

from . import question_bank_loader 
from . import question_loader
from .test_bank_prompts import Prompt, QuestionPrompt, NuggetPrompt, get_prompt_classes
from .test_bank_prompts import *
from .t5_qa import *
from .data_model import ExamGrades, FullParagraphData, GradeFilter, Grades, QueryWithFullParagraphList, SelfRating, dumpQueryWithFullParagraphList, parseQueryWithFullParagraphs, writeQueryWithFullParagraphs
from . import tqa_loader
from .async_utils import *


class AnswerProcessor(ABC):
    
    def want_to_process(self, grade: Union[ExamGrades, Grades]) -> bool:
        ''''Returns True on ExamGrades or Grades objects it can process an answer'''
        return False
    
    def convert_grades(self, grade: Grades, paragraph:FullParagraphData) -> Optional[Grades]:
        '''Process and convert answers of a Grades object'''
        return None

    def convert_exam_grades(self, grade: ExamGrades, paragraph:FullParagraphData) -> Optional[ExamGrades]:
        '''Process and convert the list of answers in an ExamGrades object'''
        for i in range(len(grade.answers)):
            (question_id, answer) = grade.answers[i]

            opt_result = self.convert_exam_grade_entry(answer=answer, paragraph=paragraph)
            if opt_result is not None:
                is_correct, self_rating = opt_result
                if grade.self_ratings is not None:
                    for r in grade.self_ratings:
                        if r.question_id == question_id:
                            r.self_rating = self_rating


                grade.correctAnswered.remove(question_id)
                grade.wrongAnswered.remove(question_id)
                if is_correct:
                    grade.correctAnswered.append(question_id)
                else:
                    grade.wrongAnswered.append(question_id)

        self.append_prompt_info(grade=grade)
        return grade

    def convert_exam_grade_entry(self, answer:str, paragraph:FullParagraphData)-> Optional[Tuple[bool,int]]:
        '''Process and convert one answer inside an ExamGrades object.'''
        return None

    def append_prompt_info(self, grade: Union[ExamGrades,Grades]):
        if grade.prompt_info is None:
            grade.prompt_info = dict()
        grade.prompt_info["answer_post_processing"]=self.__class__.__name__

class Llama3yesNoAnswerProcessor(AnswerProcessor):
    def __init__(self, grade_filters:List[GradeFilter]):
        self.grade_filters = grade_filters        

    def want_to_process(self, grade: Union[ExamGrades, Grades]) -> bool:
        return any(grade_filter.filter(grade) for grade_filter in self.grade_filters)


    def convert_grades(self, grade: Grades, paragraph:FullParagraphData) -> Optional[Grades]:
        if grade.answer.lower().startswith("yes"):
            grade.correctAnswered = True
            grade.self_ratings = 1
            self.append_prompt_info(grade=grade)
            return grade  

        elif grade.answer.lower().startswith("no"):
            grade.correctAnswered = False
            grade.self_ratings = 0
            self.append_prompt_info(grade=grade)
            return grade  

        else: 
            return None
    
    def convert_exam_grade_entry(self, answer:str, paragraph:FullParagraphData)-> Optional[Tuple[bool,int]]:
        if answer.lower().startswith("yes"):
            correctAnswered = True
            self_ratings = 1
            return (correctAnswered,self_ratings)

        if answer.lower().startswith("no"):
            correctAnswered = False
            self_ratings = 0
            return (correctAnswered,self_ratings)

        return None



def noodle_grades(query_paragraphs:List[QueryWithFullParagraphList], answer_processor:AnswerProcessor)->List[QueryWithFullParagraphList]:
    '''This method may modify the contents'''
    for qf in query_paragraphs:
        print(f"Query: {qf.queryId}")
        for para in qf.paragraphs:
            # print(f"Paragraph: {para.paragraph_id}")

            if para.grades is not None:                
                for grade in para.grades:
                    if answer_processor.want_to_process(grade=grade):
                        grade_opt = answer_processor.convert_grades(grade=grade, paragraph=para)
                        if grade_opt is not None:
                            grade= grade_opt

            if para.exam_grades is not None:
                for exam_grade in para.exam_grades:
                    if answer_processor.want_to_process(grade=exam_grade):
                        exam_grade_opt = answer_processor.convert_exam_grades(grade=exam_grade, paragraph=para)
                        if exam_grade_opt is not None:
                            exam_grade= exam_grade_opt

    return query_paragraphs


def answer_processing_main(cmdargs=None):
    """Post-process answers from an LLM."""

    import argparse

    desc = f'''Post-process answers from an LLM. 
             '''
      
    
    parser = argparse.ArgumentParser(description="EXAM answer post processig"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('paragraph_file', type=str, metavar='xxx.jsonl.gz'
                        , help='json file with paragraph to grade with exam questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )



    parser.add_argument('-o', '--out-file', type=str, metavar='exam-xxx.jsonl.gz', help='Output file name where paragraphs with exam grade annotations will be written to')
    parser.add_argument('--model', type=str, metavar='MODEL', help='the huggingface model used for grading')


    parser.add_argument('--prompt-class', nargs="+", type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--dont-check-prompt-class',action='store_true',  help='If set, will allow any prompt_class to be used that is in the data, without any verification. Any data errors are your own fault!')
    prompt_type_choices=[QuestionPrompt.my_prompt_type, NuggetPrompt.my_prompt_type, DirectGradingPrompt.my_prompt_type]
    parser.add_argument('--prompt-type', type=str, choices=prompt_type_choices, required=False,  metavar="PROMPT_TYPE", help=f"Manually set the prompt_type when setting --dont-check-prompt-class (it will otherwise be automatically set based on known prompt_classes). Choices: {prompt_type_choices}")

    parser.add_argument('--max-queries', type=int, metavar="n", default=-1, help="Limit number of queries to be processed")
    parser.add_argument('--max-paragraphs', type=int, metavar="n", default=-1, help="Limit number of paragraphs to be processed")


    # Parse the arguments
    args = parser.parse_args(args = cmdargs) 
 

    if not args.dont_check_prompt_class:
        if isinstance(args.prompt_class, list):
            for prompt_class in args.prompt_class:
                if prompt_class not in get_prompt_classes():
                    raise RuntimeError(f"Unknown promptclass {args.prompt_class}. Valid choices: {get_prompt_classes()}. You can disable the check with \'--dont-check-prompt-class\'")
        else:
            if args.prompt_class not in get_prompt_classes():
                raise RuntimeError(f"Unknown promptclass {args.prompt_class}. Valid choices: {get_prompt_classes()}. You can disable the check with \'--dont-check-prompt-class\'")
    if args.dont_check_prompt_class:
        #  make sure that prompt_type argument is specified
        if args.prompt_type is None:
            raise RuntimeError(f"Since --dont-check-prompt-class is set, you must also specify --prompt-type PROMPT_TYPE!")

    def get_prompt_type():
        if args.dont_check_prompt_class:
            return args.prompt_type
        else:
            return get_prompt_type_from_prompt_class(args.prompt_class)

    question_set:Dict[str,List[Prompt]]
    # if args.question_type == "tqa":
    #     question_set = dict(fix_car_query_id( tqa_loader.load_all_tqa_data(tqa_path=Path(args.question_path)
    #                                                                       , prompt_class=args.prompt_class
    #                                                                       , self_rater_tolerant = (args.model_pipeline=="llama")
    #                                                                       )))
    # elif args.question_type == 'genq':
    #     question_set = dict(question_loader.load_naghmehs_question_prompts(args.question_path, prompt_class=args.prompt_class))
    # elif args.question_type == 'question-bank':
    #     question_set = dict(question_bank_loader.load_prompts_from_test_bank(args.question_path
    #                                                                          , prompt_class=args.prompt_class
    #                                                                          , use_nuggets=args.use_nuggets
    #                                                                          , self_rater_tolerant = (args.model_pipeline=="llama")
    #                                                                          , custom_prompt = args.custom_prompt
    #                                                                          , custom_prompt_name = args.custom_prompt_name
    #                                                                          ))
    # elif args.question_type == 'direct':
    #     question_set = direct_grading_prompts(queries=json_query_loader(query_json=args.question_path)
    #                                           , prompt_class=args.prompt_class
    #                                           , max_queries=None
    #                                           , self_rater_tolerant = (args.model_pipeline=="llama")
    #                                           )
    # else:
    #     raise f"args.question_type \'{args.question_type}\' undefined"
    


    grade_filters =  [ GradeFilter(model_name=args.model
                               , prompt_class = prompt_class
                               , is_self_rated=None
                               , min_self_rating=None
                               , question_set=None # args.question_set
                               , prompt_type=get_prompt_type_from_prompt_class(prompt_class)
                               , data_set=None)
                            for  prompt_class in args.prompt_class]  if isinstance(args.prompt_class, list) else \
                          [GradeFilter(model_name=args.model
                               , prompt_class = args.prompt_class
                               , is_self_rated=None
                               , min_self_rating=None
                               , question_set=None # args.question_set
                               , prompt_type=get_prompt_type_from_prompt_class(args.prompt_class)
                               , data_set=None)]
    print("grade_filters", grade_filters)
    answer_processor = Llama3yesNoAnswerProcessor(grade_filters=grade_filters)

    qfs = parseQueryWithFullParagraphs(args.paragraph_file)

    qfs = noodle_grades(query_paragraphs= qfs, answer_processor=answer_processor)
                
    writeQueryWithFullParagraphs(file_path=args.out_file, queryWithFullParagraphList=qfs)
    print("Export complete")

if __name__ == "__main__":
    # asyncio.run(main())
    answer_processing_main()
