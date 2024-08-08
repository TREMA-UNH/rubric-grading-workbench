.maxwidth 200

create type QueryId as varchar;
create type QuestionId as varchar;
create type ParagraphId as varchar;
create type Answer as struct(question_id QuestionId, answer varchar);

create table input as
    select
        x.x[1] :: QueryId as query_id,
        x.x[2] as paragraphs
    from (
        select *
        from read_json_auto('/home/dietz/jelly-home/peanut-jupyter/exampp/data/dl20/questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2020-qrels-runs-with-text.jsonl.gz') as x(x)
    ) as x;

--select * from exam_grades as
--    select * from input as x, x.exam_grades;

create table queries as
    select
        query_id :: QueryId as query_id,
        info,
        query_text,
        hash,
        items as questions,
    from (
        select *
        from read_json_auto('/home/dietz/jelly-home/peanut-jupyter/exampp/data/dl20/dl20-questions.jsonl.gz') as x
    ) as x
;

create table questions as
    select
        query_id :: QueryId as query_id,
        (question->>'$.question_id') :: QuestionId as question_id,
        (question->>'$.question_text') as text,
    from (
        select *, unnest(questions->'$[*]') as question
        from queries
    )
;

create table query_paragraphs as
select 
    x.query_id :: QueryId as query_id,
    (x.x->>'$.paragraph_id') :: ParagraphId as paragraph_id,
    x.x->>'$.text' as text,
    x.x->>'$.paragraph_data' as paragraph_data,
    x.x->>'$.exam_grades' as exam_grades
from (
    select
      input.query_id as query_id,
      unnest(input.paragraphs->'$[*]') as x
    from input
) as x;

create table judgements as
select 
    x.query_id,
    x.paragraph_id,
    (x.judgment->'$.relevance') :: integer as relevance,
from (
    select
      *,
      unnest(paragraph_data->'$.judgments[*]') as judgment
    from query_paragraphs
) as x;

create table rankings as
select
    (x.query_id :: QueryId) as query_id,
    (x.paragraph_id :: ParagraphId) as paragraph_id,
    (x.ranking->>'$.method') as method,
    (x.ranking->'$.rank') :: integer as rank,
    (x.ranking->'$.score') :: real as score,
from (
    select
      *,
      unnest(paragraph_data->'$.rankings[*]') as ranking
    from query_paragraphs
) as x;

create table exam_grades as
select 
    x.query_id,
    x.paragraph_id,
    x.exam_grade->>'$.llm' as llm,
    x.exam_grade->'$.llm_options' as llm_options,
    x.exam_grade->>'$.prompt_type' as prompt_type,
    x.exam_grade->'$.prompt_info' as prompt_info,
    x.exam_grade->>'$.prompt_info.prompt_class' as prompt_class,
    (x.exam_grade->'$.correctAnswered') :: QuestionId[] as correct_answered,
    (x.exam_grade->'$.answers') as answers,
    (x.exam_grade->'$.self_ratings') as self_ratings
from (
    select
        query_id,
        paragraph_id,
        unnest(exam_grades->'$[*]') as exam_grade
    from query_paragraphs
) as x;

create table self_ratings as
select
    x.query_id,
    x.paragraph_id,
    (x.self_rating->>'$.question_id') :: QuestionId as question_id,
    (x.self_rating->>'$.self_rating') :: integer as rating
from (
    select
        *,
        unnest(self_ratings->'$[*]') as self_rating
    from exam_grades
) as x
where x.self_rating->>'$.question_id' is not null
;

select * from exam_grades
where prompt_class = 'NuggetSelfRatedPrompt'
;

-- For each question id and prompt class:
-- Looking at self_rating, how often do we have a negative judgement and positive self-rating?
-- output: question id, frequency, question text

prepare bad_questions as
select
    x.*,
    questions.text
from (
    select
        self_ratings.question_id as question_id,
        exam_grades.prompt_class as prompt_class,
        sum(1) as frequency
    from self_ratings
    inner join exam_grades
        on exam_grades.paragraph_id = self_ratings.paragraph_id
        and exam_grades.query_id = self_ratings.query_id
    inner join judgements
        on judgements.paragraph_id = self_ratings.paragraph_id
        and judgements.query_id = self_ratings.query_id
    where judgements.relevance = $min_relevance
      and self_ratings.rating >= $min_pos_rating
    group by self_ratings.question_id, exam_grades.prompt_class
) as x
inner join questions
    on questions.question_id = x.question_id
where x.prompt_class = $prompt_class
    and query_id like '940547%'
order by x.frequency desc
;

execute bad_questions(
    min_relevance := 0,
    min_pos_rating := 4,
    prompt_class := 'QuestionSelfRatedUnanswerablePromptWithChoices'
);

prepare uncovered_passages as
select
    x.*,
from (
    select
        exam_grades.prompt_class,
        exam_grades.query_id,
        exam_grades.paragraph_id,
        query_paragraphs.text,
    from exam_grades
    inner join judgements
        on judgements.paragraph_id = exam_grades.paragraph_id
        and judgements.query_id = exam_grades.query_id
    inner join query_paragraphs
        on query_paragraphs.paragraph_id = exam_grades.paragraph_id
        and query_paragraphs.query_id = exam_grades.query_id
    where judgements.relevance >= $min_relevance
      and not exists (
        select *
        from self_ratings
        where self_ratings.rating > $min_pos_rating
        and self_ratings.paragraph_id = exam_grades.paragraph_id
        and self_ratings.query_id = exam_grades.query_id
      and (exam_grades.prompt_info->'$.is_self_rated') :: boolean
    )
    order by exam_grades.query_id
) as x
;

execute uncovered_passages(
    min_relevance := 0,
    min_pos_rating := 4
);

