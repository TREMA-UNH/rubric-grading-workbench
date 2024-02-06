#!/bin/bash

#echo "\n\n\Generate CAR Nuggets"

#python -O -m exam_pp.question_generation -c benchmarkY3test.cbor-outlines.cbor -o car-nuggets.jsonl.gz --use-nuggets --description "A new set of generated nuggets for CAR"

#echo "\n\n\Generate CAR Questions"

#python -O -m exam_pp.question_generation -c benchmarkY3test.cbor-outlines.cbor -o car-questions.jsonl.gz --description "A new set of generated questions for CAR"

echo "\n\n\self-rated CAR nuggets"

ungraded="benchmarkY3test-qrels-with-text.jsonl.gz"
#ungraded="benchmarkY3test-qrels-runs-with-text.jsonl.gz"
withrate="nuggets-rate--exam--benchmarkY3test-exam-qrels-10q.jsonl.gz"
withrateextract="nuggets-rate-explain--exam--benchmarkY3test-exam-qrels-10q.jsonl.gz"

# grade nuggets

#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetSelfRatedPrompt --question-path car-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  --max-queries 10

echo "\n\n\ Explained CAR Nuggets"

#python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetExtractionPrompt --question-path car-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  --max-queries 10

# grade questions

ungraded="$withrateextract"
withrate="questions-nuggets-rate--exam--benchmarkY3test-exam-qrels-10q.jsonl.gz"
withrateextract="questions-nuggets-rate-explain--exam--benchmarkY3test-exam-qrels-10q.jsonl.gz"

withtqa="tqa-rate-questions-nuggets-rate-explain--exam--benchmarkY3test-exam-qrels-10q.jsonl.gz"
withtqaexplain="tqa-questions-nuggets-rate-explain--exam--benchmarkY3test-exam-qrels-10q.jsonl.gz"

#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path car-questions.jsonl.gz  --question-type question-bank --max-queries 10

echo "\n\n\ Explained CAR Nuggets"



python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConciseUnanswerablePromptWithChoices --question-path car-questions.jsonl.gz  --question-type question-bank --max-queries 10


python -O -m exam_pp.exam_grading  $withrateextract -o $withtqa --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path tqa_train_val_test  --question-type tqa --max-queries 10
python -O -m exam_pp.exam_grading  $withtqa -o $withtqaexplain --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConcisePromptWithAnswerKey2 --question-path tqa_train_val_test  --question-type tqa --max-queries 10


