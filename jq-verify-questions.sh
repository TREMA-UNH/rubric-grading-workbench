exam_graded=$1
question_id=$2
min_rate=$3

zcat "${exam_graded}" | jq ".[1][] | select(any(.exam_grades[].self_ratings[]?; .question_id == \"${question_id}\" and .self_rating == ${min_rate})) |  {
  text: .text, 
  exam_grades_answers: [
    .exam_grades[]?
    | .answers[]                                                                                                                
    | select(.[0] == \"${question_id}\")
  ]                                                                        
}" | less

