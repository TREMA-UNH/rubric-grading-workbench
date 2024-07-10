from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Sequence, cast, Dict, List, Tuple


from . import tqa_loader
from .test_bank_prompts import get_prompt_classes, get_prompt_types
from . data_model import QueryWithFullParagraphList, GradeFilter, parseQueryWithFullParagraphs
from . question_bank_loader import QueryQuestionBank, ExamQuestion, QueryTestBank
from . import question_bank_loader

def verify_grade_extraction(graded:List[QueryWithFullParagraphList], question_bank:Sequence[QueryTestBank], rate_grade_filter: GradeFilter, answer_grade_filter: GradeFilter):
    # graded_dict={q.queryId: q for q in graded}
    questions_dict= {q.query_id:cast(QueryQuestionBank, q)  for q in question_bank } 

    question:ExamQuestion
    for entry in graded:
        question_bank_entry = questions_dict.get(entry.queryId)
        if question_bank_entry is not None:
            for question in question_bank_entry.get_questions():
                qid = question.question_id
                question_text = question.question_text
                print(f"\n{qid}\n{question_text}")

                answer_pairs: List[Tuple[int,str]] = list()

                for paragraph in entry.paragraphs:
                    myanswers = list()
                    # examGrade_tmp = paragraph.retrieve_exam_grade_all(grade_filter=grade_filter)
                    rate = -1
                    extracted_anwer = []
                    for exam_grade in paragraph.retrieve_exam_grade_all(grade_filter=rate_grade_filter):
                        rate = max ([rate.self_rating for rate in exam_grade.self_ratings_as_iterable() if rate.get_id() == qid])
                        ans = [answer for qid_,answer in exam_grade.answers if qid_ == qid]
                        myanswers.extend(ans)

                    for exam_grade in paragraph.retrieve_exam_grade_all(grade_filter=answer_grade_filter):
                        ans = [answer for qid_,answer in exam_grade.answers if qid_ == qid]
                        extracted_anwer = ans
                        myanswers.extend(ans)

                    if extracted_anwer:
                        myanswer_str = "\t".join(extracted_anwer)
                        # print(f"{question_text}\t {paragraph.paragraph_id}\t {myanswer_str}")
                        info_str = f"{question_text}\t {paragraph.paragraph_id}\t {rate} \t {myanswer_str}"
                        answer_pairs.append(  (rate, info_str) ) 
                
                answer_pairs = sorted(answer_pairs, key= lambda x: x[0], reverse=True ) # most confident answers first
                print('\n'.join( (answer for _rate, answer in answer_pairs ) ))
                print("")
        print("")

def questions_to_dict(question_bank:Sequence[QueryTestBank])->dict[str,ExamQuestion]:
    question_by_id:Dict[str,ExamQuestion] = dict() 
    for bank in question_bank:
        if bank is not None:
            for question in cast(QueryQuestionBank, bank).get_questions():
                qid = question.question_id
                question_by_id[qid] = question
    return question_by_id


def identify_uncovered_passages(graded:List[QueryWithFullParagraphList], question_bank:Sequence[QueryTestBank], grade_filter:GradeFilter, min_judgment:int=1, min_rating:int=4):
    print("relevant passages without any high self-ratings:")
    for entry in graded:
        print(f"query_id: {entry.queryId}")
        print("paragraphId worst_judgment best_rating text")
        for paragraph in entry.paragraphs:
            judgments = paragraph.paragraph_data.judgments
            if (judgments is not None):
                judgments_ = [j.relevance  for j in judgments if j is not None]
                worst_judgment = min ( judgments_ ) if judgments_ else 0
                if worst_judgment >= min_judgment:  # we have positive judgments

                    for exam_grade in paragraph.retrieve_exam_grade_all(grade_filter=grade_filter):
                        best_rating = max( (rating.self_rating for rating in exam_grade.self_ratings_as_iterable()) )

                        if best_rating < min_rating: # not a single decent self-rating
                            # no good self-rating
                            print(f"{paragraph.paragraph_id}\t{worst_judgment}\t{best_rating}\t{paragraph.text}")



def identify_bad_question(graded:List[QueryWithFullParagraphList], question_bank:Sequence[QueryTestBank], grade_filter:GradeFilter, min_judgment:int=1, min_rating:int=4):
    questions_dict= questions_to_dict(question_bank=question_bank)

    question:ExamQuestion
    for entry in graded:
        bad_questions:Dict[str, int] = defaultdict(lambda:0)
        bad_nuggets:Dict[str, int] = defaultdict(lambda:0)
        for paragraph in entry.paragraphs:
            judgments = paragraph.paragraph_data.judgments
            if (judgments is not None):
               if any ([j.relevance < min_judgment for j in judgments if j is not None]): # we have negative judgments
                for exam_grade in paragraph.retrieve_exam_grade_all(grade_filter=grade_filter):
                    for rating in exam_grade.self_ratings_as_iterable():
                        if rating.self_rating >= min_rating:
                            if rating.nugget_id is not None:
                                bad_nuggets[rating.nugget_id]+=1
                            if rating.question_id is not None:
                                bad_questions[rating.question_id]+=1

        ranked_nuggets = sorted(bad_nuggets.items(), key=lambda item: item[1], reverse=True)
        ranked_questions = sorted(bad_questions.items(), key=lambda item: item[1], reverse=True)

        print("20 nuggets most frequently associated with paragraphs that are judged negative, but have high self-ratings:")
        for nug,count in itertools.islice(ranked_nuggets, 20):
            print(f"{nug}\t {count}")

        print("20 questions most frequently associated with paragraphs that are judged negative, but have high self-ratings:")
        for q,count in itertools.islice(ranked_questions, 20):
            if q in questions_dict:
                question = questions_dict[q]
                print(f"{q}\t {count}\t{question.question_text}")
            else:
                print(f"{q}\t {count}")

        print("")



@dataclass
class DisplayEntry():
    query_id:str
    question_id:str
    question_text:str
    paragraph_id:str
    paragraph_text:str
    self_rating:int
    extracted_answer:str


def grid_display(graded:List[QueryWithFullParagraphList]
                 , question_bank:Sequence[QueryTestBank]
                 , file_path:Path
                 , rate_grade_filter: GradeFilter
                 , answer_grade_filter: GradeFilter):

    import csv


    # Open the file in write mode
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, dialect=csv.excel)
            
        # graded_dict={q.queryId: q for q in graded}
        # questions_dict= {q.query_id:cast(QueryQuestionBank, q)  for q in question_bank } 
        
        questions_dict:Dict[str, List[ExamQuestion]] = defaultdict(list)
        question_text_dict:Dict[str, str] = dict() # question Id -> question text
        for q in question_bank:
            questions_dict[q.query_id].extend(cast(QueryQuestionBank, q).get_questions())
            for qq in cast(QueryQuestionBank, q).get_questions():
                question_text_dict[qq.question_id]=qq.question_text


        question:ExamQuestion
        for entry in graded:
            question_bank_entry = questions_dict.get(entry.queryId)
            if question_bank_entry is None:
                print(f"Can't identify question bank entry for {entry.queryId}. Available Keys: {questions_dict.keys()}")
            else:
                display_entries:List[List[DisplayEntry]] = list()
                question_ids = [q.question_id for q in question_bank_entry]
                # question_texts = {q.question_id:q.question_text for q in question_bank_entry} 

                for question in question_bank_entry:
                    question_id = question.question_id
                    print(f"{entry.queryId} -- {question_id} -- {question.question_text}")

                for paragraph in entry.paragraphs:
                    paragraph_display_entries: List[DisplayEntry] = list()

                    for question in question_bank_entry:
                        question_id = question.question_id
                        # print(f"{entry.queryId} -- {question_id} -- {question.question_text}")

                        answer_pairs: List[Tuple[int,str]] = list()

                        myanswers = list()
                        # examGrade_tmp = paragraph.retrieve_exam_grade_all(grade_filter=grade_filter)
                        rate = -1
                        extracted_anwer = []
                        for exam_grade in paragraph.retrieve_exam_grade_all(grade_filter=rate_grade_filter):
                            selected_ratings = [rate.self_rating for rate in exam_grade.self_ratings_as_iterable() if rate.get_id() == question_id]
                            if len(selected_ratings):
                                rate_ = max (selected_ratings)
                                rate = max(rate_, rate)
                            ans = [answer for qid_,answer in exam_grade.answers if qid_ == question_id]
                            myanswers.extend(ans)

                        for exam_grade in paragraph.retrieve_exam_grade_all(grade_filter=answer_grade_filter):
                            ans = [answer for qid_,answer in exam_grade.answers if qid_ == question_id]
                            extracted_anwer = ans
                            myanswers.extend(ans)

                        # if extracted_anwer:
                        myanswer_str = "\t".join(extracted_anwer)
                        # print(f"{question_text}\t {paragraph.paragraph_id}\t {myanswer_str}")
                        # info_str = f"{question_text}\t {paragraph.paragraph_id}\t {rate} \t {myanswer_str}"
                        # answer_pairs.append(  (rate, info_str) ) 
                        dp =  DisplayEntry(query_id=entry.queryId
                                            , question_id=question.question_id
                                            , question_text= question.question_text
                                            , paragraph_id= paragraph.paragraph_id
                                            , paragraph_text= paragraph.text
                                            , self_rating=rate
                                            , extracted_answer=myanswer_str 
                                            )  
                        paragraph_display_entries.append(dp)

                    if paragraph_display_entries:
                        display_entries.append(paragraph_display_entries)        


                def rankscore(entries:List[DisplayEntry])->int:
                    return sum( (e.self_rating for e in entries) )


                if display_entries:
                    display_entries = sorted(display_entries, key= lambda x: rankscore(x), reverse=True ) # most confident answers first


                    writer.writerow([])
                    query_id = display_entries[0][0].query_id
                    writer.writerow(['queryid',query_id, 'query_text=?'])
                    
                    # Write the header row
                    writer.writerow(flatten([query_id,'',''] ,[[question_id, ''] for question_id in question_ids]))  # Empty string for the top-left corner cell
                    writer.writerow(flatten([query_id,'',''],  [[question_text_dict[question_id], ''] for question_id in question_ids]))  # Empty string for the top-left corner cell

                    # Write each row
                    for paragraph_display_entries  in display_entries:
                        if len(paragraph_display_entries):
                            p = paragraph_display_entries[0]
                            row = flatten([query_id, p.paragraph_id, p.paragraph_text],   [[display_entry.self_rating, display_entry.extracted_answer] for display_entry in paragraph_display_entries])
                            writer.writerow(row)


        print(f"output written to {file_path}")

def flatten(lst, lst_of_lsts):
    return list(itertools.chain(lst, itertools.chain.from_iterable(lst_of_lsts)))


def main(cmdargs=None):
    import argparse

    desc = r'''Three analyses for manual verification and supervision:   
      --verify-grading: display all extracted answers, grouped by test question/nugget
           If need to be, grading prompts need to be adjusted, or differnt llm should be chosen
      --uncovered-passages: display relevant passages that do not cover any questions/nuggets
           New test bank entries should be created.
      --bad-question: display questions/nuggets frequently covered by non-relevant passages.
           These should be removed from the test bank.
    '''

    parser = argparse.ArgumentParser(description="EXAM Verification"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('exam_graded_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of answerable questions. The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=False, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--prompt-class-answer', type=str, choices=get_prompt_classes(), required=False, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--prompt-type', type=str, choices=get_prompt_types(), required=False, default="question", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--question-path', type=str, metavar='PATH', help='Path to read exam questions from (can be tqa directory or file)')
    parser.add_argument('--question-type', type=str, choices=['question-bank','tqa'], required=False, metavar='PATH', help='Question type to read from question-path')
    parser.add_argument('--use-nuggets', action='store_true', help="if set uses nuggets instead of questions")

    parser.add_argument('--verify-grading', action='store_true', help="If set, will verify that extracted answers correlate with self-ratings.")
    parser.add_argument('--uncovered-passages', action='store_true', help="If set, will verify which relevant passages are not covered by any questions/nuggets")
    parser.add_argument('--bad-question', action='store_true', help="If set, will identify questions/nuggets that are not indicating relevance")
    parser.add_argument('--min-judgment', type=int, required=False, metavar='LEVEL', help='Minimum judgment level for a paragraph to be judged relevant')
    parser.add_argument('--min-rating', type=int, required=False, metavar='LEVEL', help='Minimum self-rating level for a paragraph to be predicted relevant')
    parser.add_argument('--grid-display', type=str, metavar='CSV', help="CSV file to export passage/question grades and answers to.", default=None)



    # Parse the arguments
    if cmdargs is not None:
        args = parser.parse_args(args=cmdargs)    
    else:
        args = parser.parse_args()



    if args.question_type == 'question-bank':
        question_set = question_bank_loader.parseTestBank(args.question_path,  use_nuggets=args.use_nuggets)
    #  Todo TQA!        
    elif args.question_type == "tqa":
        # question_set = fix_car_query_id(tqa_loader.parseTestBank(Path(args.question_path))
        question_set = tqa_loader.parseTestBank_all(Path(args.question_path), fix_query_id=tqa_loader.fix_tqa_car_query_id)
    else:
        raise f"args.question_type \'{args.question_type}\' undefined"

    graded = parseQueryWithFullParagraphs(args.exam_graded_file)

    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_path, prompt_type=None)
    
    if args.verify_grading:
        grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_path, prompt_type=None)
        answer_grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class_answer, is_self_rated=None, min_self_rating=None, question_set=args.question_path, prompt_type=None)
        verify_grade_extraction(graded= graded, question_bank= question_set, rate_grade_filter= grade_filter, answer_grade_filter=answer_grade_filter)
    if args.grid_display:
        grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_path, prompt_type=None)
        answer_grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class_answer, is_self_rated=None, min_self_rating=None, question_set=args.question_path, prompt_type=None)
        grid_display(graded= graded, question_bank= question_set, file_path=Path(args.grid_display),  rate_grade_filter= grade_filter, answer_grade_filter=answer_grade_filter)
    if args.uncovered_passages:
        identify_uncovered_passages(graded=graded, question_bank= question_set, min_judgment= args.min_judgment, min_rating= args.min_rating, grade_filter= grade_filter)
    if args.bad_question:
        identify_bad_question(graded= graded, question_bank= question_set, min_judgment= args.min_judgment, min_rating= args.min_rating, grade_filter= grade_filter)

if __name__ == "__main__":
    main()
