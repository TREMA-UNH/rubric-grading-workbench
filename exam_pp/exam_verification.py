from typing import Sequence
from .test_bank_prompts import get_prompt_classes
from .data_model import *
from .question_bank_loader import *
from . import question_bank_loader

def verify_grade_extraction(graded:List[QueryWithFullParagraphList], question_bank:Sequence[QueryTestBank]):
    # graded_dict={q.queryId: q for q in graded}
    questions_dict= {q.query_id:cast(QueryQuestionBank, q)  for q in question_bank } 

    question:ExamQuestion
    for entry in graded:
        question_bank = questions_dict.get(entry.queryId)
        if question_bank is not None:
            for question in question_bank.get_questions():
                qid = question.question_id
                question_text = question.question_text
                print("\n{qid}\n{question_text}")

                for paragraph in entry.paragraphs:
                    myanswers = list()
                    for exam_grade in paragraph.exam_grades_iterable():
                        ans = [answer for qid_,answer in exam_grade.answers if qid_ == qid]
                        myanswers.extend(ans)

                    myanswer_str = "\t".join(myanswers)
                    print(f"{question_text}\t {paragraph.paragraph_id}\t {myanswer_str}")
        print("")

def questions_to_dict(question_bank:Sequence[QueryTestBank])->dict[str,ExamQuestion]:
    question_by_id:Dict[str,ExamQuestion] = dict() 
    for bank in question_bank:
        if bank is not None:
            for question in cast(QueryQuestionBank, bank).get_questions():
                qid = question.question_id
                question_by_id[qid] = question
    return question_by_id


def identify_uncovered_passages(graded:List[QueryWithFullParagraphList], question_bank:Sequence[QueryTestBank], min_judgment:int=1, min_rating:int=4):
    print("relevant passages without any high self-ratings:")
    for entry in graded:
        print(f"query_id: {entry.queryId}")
        for paragraph in entry.paragraphs:
            judgments = paragraph.paragraph_data.judgments
            if (judgments is not None):
               if any ([j.relevance >= min_judgment for j in judgments if j is not None]): # we have positive judgments

                for exam_grade in paragraph.exam_grades_iterable():
                    if any ( [ rating.self_rating < min_rating for rating in exam_grade.self_ratings_as_iterable()]):
                        # no good self-rating
                        print(f"{paragraph.paragraph_id}\t{paragraph.text}")
                        



def identify_bad_question(graded:List[QueryWithFullParagraphList], question_bank:Sequence[QueryTestBank], min_judgment:int=1, min_rating:int=4):
    questions_dict= questions_to_dict(question_bank=question_bank)

    question:ExamQuestion
    for entry in graded:
        bad_questions:Dict[str, int] = defaultdict(lambda:0)
        bad_nuggets:Dict[str, int] = defaultdict(lambda:0)
        for paragraph in entry.paragraphs:
            judgments = paragraph.paragraph_data.judgments
            if (judgments is not None):
               if any ([j.relevance < min_judgment for j in judgments if j is not None]): # we have negative judgments
                for exam_grade in paragraph.exam_grades_iterable():
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





def main():
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
    parser.add_argument('--question-path', type=str, metavar='PATH', help='Path to read exam questions from (can be tqa directory or file)')
    parser.add_argument('--question-type', type=str, choices=['question-bank'], required=False, metavar='PATH', help='Question type to read from question-path')
    parser.add_argument('--use-nuggets', action='store_true', help="if set uses nuggets instead of questions")

    parser.add_argument('--verify-grading', action='store_true', help="If set, will verify that extracted answers correlate with self-ratings.")
    parser.add_argument('--uncovered-passages', action='store_true', help="If set, will verify which relevant passages are not covered by any questions/nuggets")
    parser.add_argument('--bad-question', action='store_true', help="If set, will identify questions/nuggets that are not indicating relevance")
    parser.add_argument('--min-judgment', type=int, required=False, metavar='PATH', help='Minimum judgment level for a paragraph to be judged relevant')
    parser.add_argument('--min-rating', type=int, required=False, metavar='PATH', help='Minimum self-rating level for a paragraph to be predicted relevant')



    # Parse the arguments
    args = parser.parse_args()           

    if args.question_type == 'question-bank':
        question_set = question_bank_loader.parseTestBank(args.question_path,  use_nuggets=args.use_nuggets)
    else:
        raise f"args.question_type \'{args.question_type}\' undefined"

    graded = parseQueryWithFullParagraphs(args.exam_graded_file)
    
    if args.verify_grading:
        verify_grade_extraction(graded, question_set)
    if args.uncovered_passages:
        identify_uncovered_passages(graded, question_set, args.min_judgment, args.min_rating)
    if args.bad_question:
        identify_bad_question(graded, question_set, args.min_judgment, args.min_rating)

if __name__ == "__main__":
    main()
