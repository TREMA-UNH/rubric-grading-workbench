#!/bin/bash



echo "\n\n\nGenerate DL20 Nuggets"

#python -O -m exam_pp.question_generation -q dl20-queries.json -o dl20-nuggets.jsonl.gz --use-nuggets --description "A new set of generated nuggets for DL20"

echo "\n\n\Generate DL20 Questions"

#python -O -m exam_pp.question_generation -q dl20-queries.json -o dl20-questions.jsonl.gz --description "A new set of generated questions for DL20"

echo "\n\n\nself-rated DL20 nuggets"

ungraded="trecDL2020-qrels-runs-with-text.jsonl.gz"

echo "Grading ${ungraded}. Number of queries:"
zcat $ungraded | wc -l

withrate="nuggets-rate--all-${ungraded}"
withrateextract="nuggets-explain--${withrate}"

# grade nuggets

#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetSelfRatedPrompt --question-path dl20-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  

echo "\n\n\ Explained DL20 Nuggets"

#python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetExtractionPrompt --question-path dl20-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  

# grade questions

echo "\n\n\ Rated DL20 Questions"
ungraded="$withrateextract"
withrate="questions-rate--${ungraded}"
withrateextract="questions-explain--${withrate}"

withtqa="tqa-rate--${withrateextract}"
withtqaexplain="tqa-explain-${withtqa}"

#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path dl20-questions.jsonl.gz  --question-type question-bank 



echo "\n\n\ Explained DL20 Questions"

#python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConciseUnanswerablePromptWithChoices --question-path dl20-questions.jsonl.gz  --question-type question-bank 


final=$withrateextract


for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt; do
	echo $promptclass

	# autograde-qrels
	#python -O -m exam_pp.exam_post_pipeline $final --testset dl20 --question-set question-bank --prompt-class $promptclass -q dl20-exam-$promptclass.qrel --qrel-leaderboard-out dl20-qrel-leaderboard-$promptclass.tsv --run-dir ./dl20runs 

	for minrating in 3 4 5; do
	# autograde-cover leaderboard
		#python -O -m exam_pp.exam_post_pipeline $final --testset dl20 --question-set question-bank --prompt-class $promptclass --min-self-rating ${minrating} --leaderboard-out dl20-autograde-cover-$promptclass-minrating-${minrating}.tsv
		echo ""
	done

	# inter-annotator agreement with judgments
	#python -O -m exam_pp.exam_post_pipeline $final --testset dl20 --question-set question-bank --prompt-class $promptclass --min-self-rating 4 --correlation-out dl20-correlation-$promptclass.tex

	# just the leaderboard - no analysis
	python -O -m exam_pp.exam_evaluation $final --question-set question-bank --prompt-class $promptclass --min-self-rating 4 --leaderboard-out dl20-autograde-cover-$promptclass-minrating-4.solo.tsv 
	python -O -m exam_pp.exam_evaluation $final --question-set question-bank --prompt-class $promptclass -q dl20-autograde-qrels-$promptclass-minrating-4.solo.qrels  --min-self-rating 4 --qrel-leaderboard-out dl20-autograde-qrels-$promptclass-minrating-4.solo.tsv --run-dir ./dl20runs 

done
