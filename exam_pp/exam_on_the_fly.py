from exam_grading import *
from question_types import QuestionPromptWithChoices
from question_types import *
from t5_qa import *
from parse_qrels_runs_with_text import *
import tqa_loader
from typing import *
from exam_cover_metric import *

def main():
    # load QA pipeline
    qaPipeline = Text2TextPipeline('google/flan-t5-large') 
    model_name = qaPipeline.modelName


    # load  exam questions
    exam_questions: List[QuestionPromptWithChoices] = [
        QuestionPromptWithChoices( question_id="?1"
        ,question = "How many people live in this text?"
        ,choices = {"correct":"zero"}
        ,correct = "zero"
        ,correctKey="correct"
        ,query_id=""
        ,query_text=""
       ),
        QuestionPromptWithChoices( question_id="?2"
        ,question = "Which languages are spoken in Switzerland?"
        ,choices = {"correct":"German, French, Italian and Romansh"}
        ,correct = "German, French, Italian and Romansh"
        ,correctKey="correct"
        ,query_id=""
        ,query_text=""
       )
      ]

    # load system responses for this query (dummy example)

    paragraphs:List[FullParagraphData] = [FullParagraphData(paragraph_id="1", text='''
                                                            It has four main linguistic and cultural regions: German, French, Italian and Romansh. 
                                                            Although most Swiss are German-speaking, national identity is fairly cohesive, being 
                                                            rooted in a common historical background, shared values such as federalism and direct 
                                                            democracy,[22][page needed] and Alpine symbolism.[23][24] Swiss identity transcends 
                                                            language, ethnicity, and religion, leading to Switzerland being described as a 
                                                            Willensnation (\"nation of volition\") rather than a nation state.
                                                            '''
                                                            , paragraph=None, paragraph_data = ParagraphData(judgments=list(), rankings=list()), exam_grades = list())]
    

    # grade with QA system (populates the exam_grades field)

    for para in paragraphs:
        paragraph_txt = para.text
        answerTuples = qaPipeline.chunkingBatchAnswerQuestions(exam_questions, paragraph_txt=paragraph_txt)
        # correctQs = [(qpc.question_id, answer) for qpc,answer in answerTuples if qpc.check_answer(answer)]

        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        if numAll > 0: # can't provide exam when no questions are answered.
            print(f"{para.paragraph_id}: {numRight} of {numAll} answers are correct.")

            # adding exam data to the JSON file
            exam_grades = ExamGrades( correctAnswered=[qpc.question_id for qpc,answer in answerTuples if qpc.check_answer(answer)]
                                    , wrongAnswered=[qpc.question_id for qpc,answer in answerTuples if not qpc.check_answer(answer)]
                                    , answers = [(qpc.question_id, answer) for qpc,answer in answerTuples ]  #  not needed, but for debugging
                                    , exam_ratio = 0.0   #  don't use this  ((1.0 * numRight) / (1.0*  numAll))
                                    , llm = qaPipeline.exp_modelName()
                                    , llm_options={}
                            ) 
            # annotate paragraph with the exam grade
            if para.exam_grades is None:
                para.exam_grades = list()
            para.exam_grades.append(exam_grades)

        else:
            print(f'no exam score generated for paragraph {para.paragraph_id} as numAll=0')

    # compute exam cover metric

    overallExamScore = plainExamCoverageScore(paragraphs, model_name=model_name)
    print(f'Exam Score = {overallExamScore}')

    





    # out_file = args.out_file

if __name__ == "__main__":
    main()
