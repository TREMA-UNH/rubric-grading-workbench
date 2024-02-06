#!/bin/bash

echo "\n\n\nGenerate DL19 Nuggets"

#python -O -m exam_pp.question_generation -q dl19-queries.json -o dl19-nuggets.jsonl.gz --use-nuggets --description "A new set of generated nuggets for DL19"

echo "\n\n\Generate DL19 Questions"

#python -O -m exam_pp.question_generation -q dl19-queries.json -o dl19-questions.jsonl.gz --description "A new set of generated questions for DL19"

echo "\n\n\nself-rated DL19 nuggets"

ungraded="trecDL2019-qrels-runs-with-text.jsonl.gz"

echo "Grading ${ungraded}. Number of queries:"
zcat $ungraded | wc -l

#ungraded="davinci-page-only.jsonl.gz"
#ungraded="chatgpt-page-only.jsonl.gz"
#ungraded="benchmarkY3test-qrels-runs-with-text.jsonl.gz"
withrate="nuggets-rate--10q-${ungraded}"
withrateextract="nuggets-explain--${withrate}"

# grade nuggets

#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetSelfRatedPrompt --question-path dl19-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  --max-queries 10

echo "\n\n\ Explained DL19 Nuggets"

#python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetExtractionPrompt --question-path dl19-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  --max-queries 10

# grade questions

echo "\n\n\ Rated DL19 Questions"
ungraded="$withrateextract"
withrate="questions-rate--${ungraded}"
withrateextract="questions-explain--${withrate}"

withtqa="tqa-rate--${withrateextract}"
withtqaexplain="tqa-explain-${withtqa}"

#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path dl19-questions.jsonl.gz  --question-type question-bank --max-queries 10



echo "\n\n\ Explained DL19 Questions"

#python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConciseUnanswerablePromptWithChoices --question-path dl19-questions.jsonl.gz  --question-type question-bank --max-queries 10


#does not apply to dl19
#echo "\n\n\ Rated TQA"

#python -O -m exam_pp.exam_grading  $withrateextract -o $withtqa --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path tqa_train_val_test  --question-type tqa --max-queries 10


#echo "\n\n\ Answer-verified  TQA"
#python -O -m exam_pp.exam_grading  $withtqa -o $withtqaexplain --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConcisePromptWithAnswerKey2 --question-path tqa_train_val_test  --question-type tqa --max-queries 10
#
final=$withrateextract


for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt; do

# autograde-qrels
#python -O -m exam_pp.exam_post_pipeline $final --testset dl19 --question-set question-bank --prompt-class $promptclass -q dl19-exam-$promptclass.qrel --qrel-leaderboard-out dl-qrel-leaderboard-$promptclass.tsv --run-dir ./dl19runs 

# autograde-cover leaderboard
python -O -m exam_pp.exam_post_pipeline $final --testset dl19 --question-set question-bank --prompt-class $promptclass --min-self-rating 4 --leaderboard-out dl-autograde-cover-$promptclass.tsv

# inter-annotator agreement with judgments
#python -O -m exam_pp.exam_post_pipeline $final --testset dl19 --question-set question-bank --prompt-class $promptclass --min-self-rating 4 --correlation-out dl-correlation-$promptclass.tex

done
