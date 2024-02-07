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


#python -O -m exam_pp.exam_grading $ungraded -o $withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path dl20-questions.jsonl.gz  --question-type question-bank 



echo "\n\n\ Explained DL20 Questions"

#python -O -m exam_pp.exam_grading $withrate  -o $withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConciseUnanswerablePromptWithChoices --question-path dl20-questions.jsonl.gz  --question-type question-bank 


final=$withrateextract

echo "Graded: $final"


# Phase 4: evaluation
#
for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt; do
	echo $promptclass


	#python -O -m exam_pp.exam_evaluation $final --question-set question-bank --prompt-class $promptclass --min-self-rating 4 --leaderboard-out dl20-autograde-cover-leaderboard-$promptclass-minrating-4.solo.tsv 
	#python -O -m exam_pp.exam_evaluation $final --question-set question-bank --prompt-class $promptclass -q dl20-autograde-qrels-leaderboard-$promptclass-minrating-4.solo.qrels  --min-self-rating 4 --qrel-leaderboard-out dl20-autograde-qrels-$promptclass-minrating-4.solo.tsv --run-dir ./dl20runs 

done

# Analyses

for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt; do
	echo $promptclass



	for minrating in 3 4 5; do
		# autograde-qrels
		# qrel leaderboard correlation
		python -O -m exam_pp.exam_post_pipeline $final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings --min-trec-eval-level ${minrating} -q dl20-exam-$promptclass.qrel --qrel-leaderboard-out dl20-autograde-qrels-leaderboard-$promptclass-minlevel-$minrating.correlation.tsv --run-dir ./dl20runs --official-leaderboard official_dl20_leaderboard.json 
	
		# autograde-cover 
		#python -O -m exam_pp.exam_post_pipeline $final --testset dl20 --question-set question-bank --prompt-class $promptclass --use-ratings --min-self-rating ${minrating} --leaderboard-out dl20-autograde-cover-$promptclass-minrating-${minrating}.tsv
		# cover leaderboard correlation
		python -O -m exam_pp.exam_post_pipeline $final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings --min-self-rating ${minrating} --leaderboard-out dl20-autograde-cover-leaderboard-$promptclass-minlevel-$minrating.correlation.tsv  --official-leaderboard official_dl20_leaderboard.json
	
		echo ""
	done



	# inter-annotator agreement
	python -O -m exam_pp.exam_post_pipeline $final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings  --inter-annotator-out dl20-autograde-inter-annotator-$promptclass.tex
done
