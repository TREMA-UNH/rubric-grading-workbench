

from os import system
import sys
from exam_pp import exam_grading, exam_post_pipeline, exam_leaderboard_analysis, exam_verification
from exam_pp.test_bank_prompts import QuestionPrompt


#python -m exam_pp.exam_grading  -o result.jsonl.gz  ./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz--model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConcisePromptWithAnswerKey tqa --question-path ./tqa_train_val_test --question-type tqa

print("\n\n\n\n\n\n")

# exam_grading.main(cmdargs=["./dl19-qrels-with-text.jsonl.gz"
#                            ,"--run-dir","./run-dl"
                           # ,"-o","result.jsonl.gz"
#                            ,"--model-pipeline", "text2text"
#                            ,"--model-name","google/flan-t5-base"
#                            ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices"
#                            ,"--max-queries","1","--max-paragraphs","1","--question-type"
#                            ,"question-bank","--question-path","./dl19-questions.jsonl.gz"
#                            ])


dl_ungraded_file = "trecDL2020-qrels-runs-with-text.jsonl.gz"


car_graded_file = "./t5-rating-naghmehs-tqa-exam-qrel-runs-result-T0050.jsonl.gz"
dl_graded_file = "dl19-exam-qrels-with-text.jsonl.gz"



# exam_verification.main(cmdargs=[dl_graded_file
#                               , "--uncovered-passages"
#                               , "--model","google/flan-t5-large"
#                               ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices" 
#                               , "--question-type", "question-bank"
#                               , "--question-path", "dl20-questions.jsonl.gz"
#                               , "--min-judgment", "2"
#                               , "--min-rating", "1"
#                                ])


# exam_verification.main(cmdargs=[dl_graded_file
#                               , "--bad-question"
#                               , "--model","google/flan-t5-large"
#                               ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices" 
#                               , "--question-type", "question-bank"
#                               , "--question-path", "dl19-questions.jsonl.gz"
#                               , "--min-judgment", "2"
#                               , "--min-rating", "1"
#                                ])



exam_verification.main(cmdargs=[dl_graded_file
                              , "--verify-grading"
                              , "--model","google/flan-t5-large"
                              ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices" 
                              ,"--prompt-class-answer","QuestionCompleteConciseUnanswerablePromptWithChoices" 
                              ,"--prompt-type",QuestionPrompt.my_prompt_type
                              , "--question-type", "question-bank"
                              , "--question-path", "dl19-questions.jsonl.gz"
                              # , "--min-judgment", "2"
                              # , "--min-rating", "1"
                               ])
sys.exit(1)
# direct grading

dl_output ="graded-"+dl_ungraded_file




exam_leaderboard_analysis.main(cmdargs=[dl_graded_file
                               , "--qrel-analysis-out", "analysis-out.tsv"         
                              # ,"-q","out.qrel"
                              ,"--run-dir","./run-dl","--qrel-query-facets", "--use-ratings"
                              , "--model","google/flan-t5-large"
                              ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices","QuestionSelfRatedUnanswerablePromptWithChoices" 
                              , "--question-set", "question-bank"
                            #  , "--leaderboards-out", "dl.cover.tsv"
                            #  , "--qrel-leaderboard-out", "dl.qrel.tsv"
                            #   , "--testset","dl19"
                              ,"--use-ratings"
                              , "--min-relevant-judgment", "2"
                              , '--official-leaderboard', 'faux-leaderboard.json'
                              , "--trec-eval-metric", "map", "P.20", "ndcg_cut.10"
                              ])



exam_grading.main([dl_ungraded_file
                           ,"-o",dl_output
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","Thomas"
                           ,"--max-queries","1","--max-paragraphs","100"
                           ,"--question-type","question-bank"
                           ,"--use-nuggets"
                           ,"--question-path","dl20-nuggets.jsonl.gz"])


exam_post_pipeline.main(cmdargs=[dl_output,
                                 "--inter-annotator-out","out.correlation.tex"
                                 ,"--model","google/flan-t5-large","--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices"
                              , "--question-set", "question-bank"
                              , "--testset","dl20"
                              , "--min-relevant-judgment", "2"
                              ,"--use-ratings"
                                ])


# Other

exam_post_pipeline.main(cmdargs=[dl_graded_file,
                                 "--inter-annotator-out","out.correlation.tex"
                                 ,"--model","google/flan-t5-large","--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices"
                              , "--question-set", "question-bank"
                              , "--testset","dl19"
                              , "--min-relevant-judgment", "2"
                              ,"--use-ratings"
                              , '--official-leaderboard', 'faux-leaderboard.json'
                                ])

exam_post_pipeline.main(cmdargs=[dl_graded_file
                              ,"-q","out.qrel"
                              ,"--run-dir","./run-dl","--qrel-query-facets", "--use-ratings"
                              , "--model","google/flan-t5-large","--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices" 
                              , "--question-set", "question-bank"
                             , "--leaderboard-out", "dl.cover.tsv"
                             , "--qrel-leaderboard-out", "dl.qrel.tsv"
                            #   , "--testset","dl19"
                              ,"--use-ratings"
                              , "--min-relevant-judgment", "2"
                              , '--official-leaderboard', 'faux-leaderboard.json'
                              ])


# exam_post_pipeline.main(cmdargs=[dl_graded_file 
#                                  ,"--leaderboard-out","out.leaderboard.tsv"
#                                  ,"--model","google/flan-t5-large","--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices"
#                               , "--question-set", "question-bank"
#                               , "-r", "--min-self-rating","4"
#                               , "--testset","dl19"
#                               , '--official-leaderboard', 'faux-leaderboard.json'
#                               ])

# exam_post_pipeline.main(cmdargs= [car_graded_file ,"--leaderboard-out","out.leaderboard.tsv","--model","google/flan-t5-large","--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices", 
#                                "--question-set", "tqa", "-r", "--min-self-rating","4"])

print("\n\n\nNuggets\n\n\n")


exam_grading.main(["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","NuggetSelfRatedPrompt"
                           ,"--max-queries","1","--max-paragraphs","1"
                           ,"--question-type","question-bank"
                           ,"--use-nuggets"
                           ,"--question-path","car-nuggets.jsonl.gz"])

print("\n\n\n\n\n\n")


exam_grading.main(["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","NuggetExtractionPrompt"
                           ,"--max-queries","1","--max-paragraphs","1"
                           ,"--question-type","question-bank"
                           ,"--use-nuggets"
                           ,"--question-path","car-nuggets.jsonl.gz"])




print("\n\n\nQuestionbank Questions\n\n\n")


exam_grading.main(["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","QuestionAnswerablePromptWithChoices"
                           ,"--max-queries","1","--max-paragraphs","1"
                           ,"--question-type","question-bank"
                           ,"--question-path","car-questions.jsonl.gz"])


print("\n\n\n\n\n\n")


print("\n\n\n\n\n\n")


exam_grading.main(["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","QuestionCompleteConciseUnanswerablePromptWithChoices"
                           ,"--max-queries","1","--max-paragraphs","1"
                           ,"--question-type","question-bank"
                           ,"--question-path","car-questions.jsonl.gz"])


print("\n\n\n\n\n\n")


exam_grading.main(cmdargs=["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices"
                           ,"--max-queries","1","--max-paragraphs","1"
                           ,"--question-type","question-bank"
                           ,"--question-path","car-questions.jsonl.gz"])


print("\n\n\nTQA \n\n\n")




exam_grading.main(cmdargs=["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","QuestionCompleteConcisePromptWithAnswerKey2"
                           ,"--question-path","./tqa_train_val_test"
                           ,"--max-queries","1","--max-paragraphs","1","--question-type", "tqa"
                           ])



print("\n\n\n\n\n\n")
exam_grading.main(cmdargs=["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","QuestionCompleteConcisePromptWithAnswerKey"
                           ,"--question-path","./tqa_train_val_test"
                           ,"--max-queries","1","--max-paragraphs","1","--question-type", "tqa"
                           ])



print("\n\n\n\n\n\n")
exam_grading.main(cmdargs=["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                           ,"-o","result.jsonl.gz"
                           ,"--model-pipeline", "text2text"
                           ,"--model-name","google/flan-t5-base"
                           ,"--prompt-class","QuestionPromptWithChoices"
                           ,"--question-path","./tqa_train_val_test"
                           ,"--max-queries","1","--max-paragraphs","1","--question-type", "tqa"
                           ])




print("\n\n\ngenQ \n\n\n")


exam_grading.main(cmdargs=["./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
                     ,"-o","result.jsonl.gz"
                     ,"--model-pipeline", "text2text"
                     ,"--model-name","google/flan-t5-base"
                     ,"--prompt-class","QuestionSelfRatedUnanswerablePromptWithChoices"
                     ,"--max-queries","1","--max-paragraphs","1"
                     ,"--question-type","genq","--question-path","naghmeh-questions.json"
                     ])


#  result_trecDL2019-qrels-with-text__new_subqueries_v2__large.jsonl.gz