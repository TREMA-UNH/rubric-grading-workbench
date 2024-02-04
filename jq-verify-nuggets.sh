exam_graded=$1
nugget_id=$2
min_rate=$3

zcat "${exam_graded}" | jq ".[1][] | select(any(.exam_grades[].self_ratings[]?; .nugget_id == \"${nugget_id}\" and .self_rating == ${min_rate})) |  {
  text: .text, 
  exam_grades_answers: [
    .exam_grades[]?
    | .answers[]                                                                                                                
    | select(.[0] == \"${nugget_id}\")
  ]                                                                        
}" | less

