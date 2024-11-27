import abc
import string
from typing import *
import re
from abc import abstractmethod
from dataclasses import dataclass

import hashlib

def get_md5_hash(input_string: str) -> str:
    # Convert the string to bytes
    input_bytes = input_string.encode('utf-8')

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the bytes
    md5_hash.update(input_bytes)

    # Get the hexadecimal digest of the hash
    hex_digest = md5_hash.hexdigest()

    return hex_digest



def get_prompt_classes()->List[str]:
    return ['QuestionPromptWithChoices'
            , 'QuestionAnswerablePromptWithChoices'
            , 'QuestionCompleteConciseUnanswerablePromptWithChoices'
            , 'QuestionCompleteConcisePromptWithAnswerKey'
            , 'QuestionCompleteConcisePromptWithAnswerKey2'
            , 'QuestionSelfRatedUnanswerablePromptWithChoices'
            , 'QuestionSelfRatedExplainPrompt'
            , 'QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2'
            # Nugget prompts
            , 'NuggetSelfRatedPrompt'
            , 'NuggetExtractionPrompt'
            # relevance grade prompts
            , "FagB", "FagB_few", "HELM", "Sun", "Sun_few", "Thomas"
# 
            , 'CustomQuestionSelfRatedPrompt'
            ]

# def get_prompt_type_from_prompt_class(prompt_class:str)->Optional[str]:
#     if prompt_class in {'QuestionPromptWithChoices'
#             , 'QuestionAnswerablePromptWithChoices'
#             , 'QuestionCompleteConciseUnanswerablePromptWithChoices'
#             , 'QuestionCompleteConcisePromptWithAnswerKey'
#             , 'QuestionCompleteConcisePromptWithAnswerKey2'
#             , 'QuestionSelfRatedUnanswerablePromptWithChoices'
#             , 'QuestionSelfRatedExplainPrompt'
#             , 'QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2'}:
#             return QuestionPrompt.my_prompt_type
#     if prompt_class in {'NuggetSelfRatedPrompt'
#                         , 'NuggetExtractionPrompt'}:
#         return NuggetPrompt.my_prompt_type
#     if prompt_class in {"FagB", "FagB_few", "HELM", "Sun", "Sun_few", "Thomas"}:
#         return DirectGradingPrompt.my_prompt_type
#     return None

    
def get_prompt_type_from_prompt_class(prompt_class:str)->Optional[str]:
    clzz = globals().get(prompt_class)
    if clzz:
        return clzz.my_prompt_type
    else:
        return None

def get_prompt_types()->List[str]:
    set_of_types:Set[str] =  {get_prompt_type_from_prompt_class(prompt_class) for prompt_class in get_prompt_classes() 
                                    if get_prompt_type_from_prompt_class(prompt_class) is not None}
    return list(set_of_types)


# -------   helpers --------------- 


class NltkInitializer():
    import nltk
    import nltk.stem #  import PorterStemmer
    import fuzzywuzzy.fuzz #import fuzz
    import nltk.corpus # import stopwords
    import nltk.corpus # import stopwords
    import nltk.tokenize # word_tokenize

    x = nltk.download('stopwords')
    y = nltk.download('punkt')  
    stemmer = nltk.stem.PorterStemmer()

    word_tokenize = nltk.tokenize.word_tokenize
    stopwords = nltk.corpus.stopwords
    fuzzratio = fuzzywuzzy.fuzz.ratio
    # def fuzz(*kwargs):
    #     return fuzzywuzzy.fuzz(kwargs)



class QuestionPromptNormalizer():

    def normalize_answer(self, answer:str)->str:
        # Lowercase, Perform other normalization like removing punctuation, if necessary
        # Stem the answer
        return NltkInitializer.stemmer.stem(answer.lower())



class QuestionStemmedChecker():

    def __init__(self, correct_answers:Set[str]):
        self.question_prompt_normalizer = QuestionPromptNormalizer()
        self.correct_answers = correct_answers
        self.normalized_correct_answers = {normalize_answer(answer) for answer in correct_answers}

    def normalize_answer(self, answer:str)->str:
        return self.question_prompt_normalizer.normalize_answer(answer)

    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer) or self.check_answer_stemmed(answer)
        # return self.check_answer_stemmed(answer)


    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers


    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return NltkInitializer.fuzzratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_correct_answers)

        return is_match

    def answer_match_info(self)->str:
        return "lowercase, stopped, stemmed, removed trailing period, fuzz > 0.8 / true-false special handling"

class TrueFalseMatcher():

    def __init__(self,correct:Set[str]):
        self.correct_answers:Set[str] = correct

        if len(correct)==1:
            add_false = False
            add_true = False
            for correct_answer in correct:
                if correct_answer.lower().strip() =="false":
                    add_false=True
                if correct_answer.lower().strip() =="true":
                    add_true=True

            if add_false:
                    self.correct_answers.add("no")
                    self.correct_answers.add("incorrect")
                    self.correct_answers.add("wrong")
            if add_true:
                    self.correct_answers.add("yes")
                    self.correct_answers.add("correct")


    def is_match(self, answer:str)->bool:
        return answer.lower().strip() in self.correct_answers
    
    def check_answer(self, answer:str)->bool:
        return self.is_match(answer)

class TrueFalseMatcher2():
    def check_true_false(self, correct:Set[str], answer:str)->Optional[bool]:
        FALSE_answers = {"no", "incorrect","false"}
        TRUE_answers = {"yes", "correct","true"}
        answer_ = {correct_answer.lower().strip() for correct_answer in correct}

        if answer_ == {"false"}:
            if answer.lower() in FALSE_answers:
                return True
            else:
                return False

        if answer_ == {"true"}:
            if answer.lower() in TRUE_answers:
                return True
            else:
                return False
            
        return None
            
class UnanswerableMatcher():
    unanswerable_expressions:Set[str]

    def __init__(self, unanswerable_expressions:Set[str]):
        self.unanswerable_expressions = unanswerable_expressions | {"unanswerable"
                                                                        ,"no"
                                                                        ,"no answer",
                                                                        "not enough information"
                                                                        ,"unknown"
                                                                        ,"it is not possible to tell"
                                                                        ,"it does not say"
                                                                        ,"no relevant information"
                                                                        # ,"[iv]","(iv)","[ii]"
                                                                        }
        self.normalized_unanswerable_expressions = {normalize_answer(zonk) for zonk in self.unanswerable_expressions}

    # inverse logic!  we are scanning for non-answers!!!
    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer) and self.check_answer_stemmed(answer)
    

    def check_answer_simple(self,answer:str)->bool:
        return not (answer in self.unanswerable_expressions)
    
    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return NltkInitializer.fuzzratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        if len(stemmed_answer)<1:
            return False
        
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_unanswerable_expressions)

        return not is_match



    def answer_match_info(self)->str:
        return "Check for unanswerable expression, all else is deemed correct"




class UnanswerableMatcher2():
    def __init__(self, unanswerable_expressions:Set[str]):
        self.unanswerable_expressions = unanswerable_expressions.union( {"unanswerable"
                                                                        ,"no"
                                                                        ,"no answer",
                                                                        "not enough information"
                                                                        ,"unknown"
                                                                        ,"it is not possible to tell"
                                                                        ,"it does not say"
                                                                        ,"no relevant information"
                                                                        # ,"[iv]","(iv)","[ii]"
                                                                        })
        self.normalized_unanswerable_expressions = {normalize_answer(zonk) for zonk in self.unanswerable_expressions}


    # inverse logic!  we are scanning for non-answers!!!
    def check_unanswer(self,answer:str)->bool:
        '''Return false if expressions of inability to answer'''
        return self.check_unanswer_simple(answer) and self.check_unanswer_stemmed(answer)



    def check_unanswer_simple(self,answer:str)->bool:
        return not (answer in self.unanswerable_expressions)


    def check_unanswer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return NltkInitializer.fuzzratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        if len(stemmed_answer)<1:
            return False
        
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_unanswerable_expressions)

        return not is_match


class SelfRaterStrict():
    def __init__(self, unanswerable_matcher2, max_rating:int=5):
        self.unanswerable_matcher2 = unanswerable_matcher2
        self.max_rating = max_rating
    
    # self-rated logic. We are scanning for 0-5. 
    # other answers get rating 1
    # unanserable expressions get rating 0
        
    def check_answer_rating(self,answer:str)->int:
        rating:int 
        # Regex to match a string with only one digit (0-5), and possibly other non-letter characters
        match = re.fullmatch(r'[^\w]*([0-9])[^\w]*', answer)
        if match:
            rating = int(match.group(1))
            if rating > self.max_rating: # outside the legal range
                if self.unanswerable_matcher2.check_unanswer(answer):   
                    rating = 0
                else:
                    rating = 1

        elif self.unanswerable_matcher2.check_unanswer(answer):   
            rating = 0
        else:
            rating = 1

        return rating


    def check_answer(self, answer:str)->bool:
        rating = self.check_answer_rating(answer=answer)
        return rating > 0


class SelfRaterTolerant():
    def __init__(self, unanswerable_matcher2, max_rating:int=5):
        self.unanswerable_matcher2 = unanswerable_matcher2
        self.max_rating = max_rating
    
    # self-rated logic. We are scanning for 0-5. 
    # other answers get rating 1
    # unanserable expressions get rating 0
        
    def check_answer_rating(self,answer:str)->int:
        rating:int 
        # Regex to match a string with only one digit (0-5), and possibly other non-letter characters
        match = re.fullmatch(r'[^\w]*([0-9])[^\w]*\s.*', answer)
        if match:
            rating = int(match.group(1))
            if rating > self.max_rating: # outside the legal range
                if self.unanswerable_matcher2.check_unanswer(answer):   
                    rating = 0
                else:
                    rating = 1

        elif self.unanswerable_matcher2.check_unanswer(answer):   
            rating = 0
        else:
            rating = 1

        return rating


    def check_answer(self, answer:str)->bool:
        rating = self.check_answer_rating(answer=answer)
        return rating > 0





class AnswerKey2Verifier():
    import nltk
    nltk.download('punkt_tab')
    question_prompt_normalizer = QuestionPromptNormalizer()
    true_false_matcher = TrueFalseMatcher2()

    def __init__(self, correct:Set[str]):
        self.correct={self.strip_trailing_period(correct_answer) for correct_answer in correct} 

        self.correct_answers = self.correct # we don't give choices:, f"{self.correctKey})", self.correctKey}

        self.normalized_correct_answers = {normalize_answer(gold) for gold in self.correct_answers}
        self.stop_stemmed_correct_answers = {self.stop_stem_normalize_answer(gold) for gold in self.correct_answers}


    def strip_trailing_period(self, answer):
        answer_ =answer.strip()
        if answer_.endswith('.'):
            answer_ = answer_[:-1]
        return answer_


    def stop_stem_normalize_answer(self, text:str)->str:
        # Convert text to lowercase
        text = text.lower().strip()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize text
        tokens = NltkInitializer.word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in NltkInitializer.stopwords.words('english')]

        # Stemming
        stemmed_tokens = [NltkInitializer.stemmer.stem(word) for word in tokens]

        # Rejoin words
        normalized_text = ' '.join(stemmed_tokens)

        return normalized_text

    def check_answer(self,answer:str)->bool:
        answer_=self.strip_trailing_period(answer)

        checkTF = self.check_true_false({answer_})
        if checkTF is not None:
            return checkTF
        
        stemmedAnswer = self.check_answer_stemmed(answer_)
        if stemmedAnswer is not None:
            return stemmedAnswer
        return self.check_answer_simple(answer_)

    def check_true_false(self, answer):
        return self.true_false_matcher.check_true_false(self.correct, answer)


    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers


    def check_answer_stemmed(self,answer:str)->Optional[bool]:
        stemmed_answer = self.stop_stem_normalize_answer(answer)

        if len(stemmed_answer) >=2:
            is_match = stemmed_answer in self.stop_stemmed_correct_answers
            if is_match: 
                return is_match

        if len(stemmed_answer) >=4:
            is_fuzzy = any (NltkInitializer.fuzzratio(stemmed_answer, stemmed_gold) > 80 for stemmed_gold in self.stop_stemmed_correct_answers)
            return is_fuzzy
        return None
    

    def answer_match_info(self):
        return "AnswerKey2Verifier: use fuzzy on normalized answers >80;   normalize by lowercasing, remove punctuation,  tokenize, stemming using NLTK"
    

# ------------ prompt classes ------------

@dataclass
class Prompt(abc.ABC):

    @abstractmethod
    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "is_self_rated":self.has_rating() 
                }

    @abstractmethod
    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        pass

    @abstractmethod
    def check_answer(self, answer):
        return True
    
    @abstractmethod
    def has_rating(self):
        return True
    
    def check_answer_rating(self,answer:str)->int:
        if self.check_answer(answer=answer):
            return 1
        else:
            return 0    
    
    @abstractmethod
    def prompt_id(self)->str:
        return ""

    @abstractmethod
    def prompt_type(self)->str:
        return "undefined"

    @abstractmethod
    def prompt_style(self)->str:
        return "Is this text relevant for ..."


@dataclass
class NuggetPrompt(Prompt):
    my_prompt_type="nugget"
    nugget_id:str
    
    def prompt_id(self)->str:
        return self.nugget_id

    def prompt_type(self)->str:
        return NuggetPrompt.my_prompt_type




@dataclass
class QuestionPrompt(Prompt):
    my_prompt_type="question"
    question_id:str

    @abstractmethod
    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        pass

    def answer_match_info(self):
        return ""
    
    def prompt_id(self)->str:
        return self.question_id

    def prompt_type(self)->str:
        return QuestionPrompt.my_prompt_type


def normalize_answer(answer:str)->str:
    # Lowercase, Perform other normalization like removing punctuation, if necessary
    # Stem the answer
    return QuestionPromptNormalizer().normalize_answer(answer)


# ------ prompt truncation ----

class PromptTruncater():

    @staticmethod
    def truncate_context(tokenizer, template,  context:str, max_length:int)->str:

        # Tokenize the question
        prompt_tokens = tokenizer.encode(template, add_special_tokens=False)

        # Calculate the number of tokens available for the context
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        available_tokens_for_context = max_length - len(prompt_tokens) - num_special_tokens -5 # 5 for good measure

        # Tokenize and truncate the context
        truncated_context_tokens = tokenizer.encode(context, add_special_tokens=False, max_length = available_tokens_for_context, truncation=True)

        return tokenizer.decode(truncated_context_tokens)


    @staticmethod
    def truncate_context_question_prompt(tokenizer, context:str, question:str, max_length:int)->str:

        # Tokenize the question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)

        # Calculate the number of tokens available for the context
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        available_tokens_for_context = max_length - len(question_tokens) - num_special_tokens -5 # 5 for good measure

        # Tokenize and truncate the context
        truncated_context_tokens = tokenizer.encode(context, add_special_tokens=False, max_length = available_tokens_for_context, truncation=True)

        # Combine truncated context with the full question
        combined_tokens = question_tokens + truncated_context_tokens
        prompt = tokenizer.decode(combined_tokens)

        return prompt

    @staticmethod
    def truncate_context_question_prompt_QC(tokenizer, context:str, question:str, max_length:int):
        # Tokenize the question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)

        # Calculate the number of tokens available for the context
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        available_tokens_for_context = max_length - len(question_tokens) - num_special_tokens -5  #5 for good measure

        # Tokenize and truncate the context
        context_tokens = tokenizer.encode(context, add_special_tokens=False, max_length = available_tokens_for_context, truncation=True)
        

        # Combine truncated context with the full question

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt



#  ----------- Prompts ---------------

@dataclass
class QuestionPromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:Set[str]
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    question_prompt_normalizer = QuestionPromptNormalizer()
    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.true_false_matcher = TrueFalseMatcher(correct=self.correct)
        self.question_stemmed_checker = QuestionStemmedChecker(correct_answers={self.correct})

        self.correct_answers = self.true_false_matcher.correct_answers.union(self.question_stemmed_checker.correct_answers)

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": True
                , "check_unanswerable": False
                , "check_answer_key": True
                , "is_self_rated":self.has_rating()
                }
    def prompt_style(self)->str:
        return  "context: question:"
    

    def has_rating(self):
        return False


    def check_answer_rating(self, answer: str) -> int:
        return super().check_answer_rating(answer)
    


    @staticmethod
    def int_key_to_str(i:int)->str:
        return f"chr(65+i)"



    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        question = self.question
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=f"context: {context};", question=f" question: {question}", max_length=max_token_len)
        return prompt
    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question = self.question
        prompt = self.prompt_truncater.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=f"context: {context}", question=f" question: {question}", max_length=max_token_len)
        return prompt

    def check_answer(self,answer:str)->bool:
        return self.true_false_matcher.check_answer(answer) or  self.question_stemmed_checker.check_answer(answer)


@dataclass
class QuestionAnswerablePromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]

    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher=UnanswerableMatcher(self.unanswerable_expressions)
        self.unanswerable_expressions = self.unanswerable_matcher.unanswerable_expressions 

    def check_answer_rating(self, answer: str) -> int:
        return super().check_answer_rating(answer)
    

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                }
    def prompt_style(self)->str:
        return  "How does this text answer this question:"
    

    def has_rating(self):
        return False


    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        question_prompt =  f' question: How does this text answer this question: {self.question}'
        context_prompt = f"context: {context};"
        # question =  f'Is this question answerable: {self.question}'
        # question =  f'Is this question answerable: {self.question}'
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f' question: How does this text answer this question: {self.question}'
        context_prompt = f"context: {context};"
        # question =  f'Is this question answerable: {self.question}'
        prompt = self.prompt_truncater.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    # inverse logic!  we are scanning for non-answers!!!
    def check_answer(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer(answer)



    def check_answer_simple(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer_simple(answer)



    def check_answer_stemmed(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer_stemmed(answer)


    def answer_match_info(self):
        return self.unanswerable_matcher.answer_match_info()

@dataclass
class QuestionCompleteConciseUnanswerablePromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]
    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher = UnanswerableMatcher(self.unanswerable_expressions)
        self.unanswerable_expressions = self.unanswerable_matcher.unanswerable_expressions 


    def check_answer_rating(self, answer: str) -> int:
        return super().check_answer_rating(answer)
    
    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                }
    def prompt_style(self)->str:
        return  "provide a complete and concise answer to the question based on the context."
    

    def has_rating(self):
        return False




    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        # f'''provide a complete and concise answer to the question based on the context. Question: {question}\nContext: {context}'''
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}'
        context_prompt = f"Context: {context}"

        # question =  f'Is this question answerable: {self.question}'
        prompt = self.prompt_truncater.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    # inverse logic!  we are scanning for non-answers!!!
    def check_answer(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer(answer)

    def check_answer_simple(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer_simple(answer)



    def check_answer_stemmed(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer_stemmed(answer)
        


    def answer_match_info(self)->str:
        return self.unanswerable_matcher.answer_match_info()



@dataclass
class QuestionCompleteConcisePromptWithAnswerKey(QuestionPrompt):
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:Set[str]
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    question_prompt_normalizer = QuestionPromptNormalizer()
    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.true_false_matcher = TrueFalseMatcher(self.correct)
        self.question_stemmed_checker = QuestionStemmedChecker(set(self.correct))

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": False
                , "check_answer_key": True
                , "is_self_rated":self.has_rating()
                }
    def prompt_style(self)->str:
        return  "provide a complete and concise answer to the question based on the context"
    

    def has_rating(self):
        return False


    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        # f'''provide a complete and concise answer to the question based on the context. Question: {question}\nContext: {context}'''
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}'
        context_prompt = f"Context: {context}"
        prompt = self.prompt_truncater.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    def check_answer(self,answer:str)->bool:
        return self.true_false_matcher.check_answer(answer) or self.question_stemmed_checker.check_answer(answer)

    def check_answer_simple(self,answer:str)->bool:
        return self.question_stemmed_checker.check_answer_simple(answer)


    def check_answer_stemmed(self,answer:str)->bool:
        return self.question_stemmed_checker.check_answer_stemmed(answer)

    def answer_match_info(self)->str:
        return self.question_stemmed_checker.answer_match_info

@dataclass
class QuestionCompleteConcisePromptWithAnswerKey2(QuestionCompleteConcisePromptWithAnswerKey):
    def __post_init__(self):
        QuestionCompleteConcisePromptWithAnswerKey.__post_init__(self)
        self.answer_key2_verifier = AnswerKey2Verifier(self.correct)

    
    def check_answer(self,answer:str)->bool:
        return self.answer_key2_verifier.check_answer(answer)


    def answer_match_info(self):
        return self.answer_key2_verifier.answer_match_info()
    

@dataclass
class QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2(QuestionCompleteConcisePromptWithAnswerKey):
    '''This is an answer-verifier to be used to regrade any QA prompt with explicit answers.'''
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:Set[str]
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        def old_prompt(key, default):
            if old_prompt_info is None:
                return default
            return old_prompt_info.get(key, default)
        
        info =  {"prompt_class": self.__class__.__name__
                , "orig_prompt_class": old_prompt("prompt_class", "")
                , "prompt_style":  old_prompt("prompt_style", "question-answering prompt")
                , "context_first": old_prompt("context_first", False)
                , "check_unanswerable": False
                , "check_answer_key": True
                , "is_self_rated":self.has_rating()
                }
        return info

    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        raise RuntimeError("This prompt is only for re-grading of previous answers.")


    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        raise RuntimeError("This prompt is only for re-grading of previous answers.")

    def prompt_style(self)->str:
        raise RuntimeError("This prompt is only for re-grading of previous answers.")
    


    def check_answer(self,answer:str)->bool:
        return False

    def check_answer_rating(self,answer:str)->int:
        if self.check_answer(answer=answer):
            return 1
        else:
            return 0

    def has_rating(self):
        return False
    
    def answer_match_info(self):
        return "Using FLAN-T5-Large with this prompt: For the question \"{question}\" the correct answer is \"{correct_answer}\". Is \"{answer}\" an equally correct response to this question? Answer yes or no."
    



@dataclass
class QuestionCompleteConcisePromptT5Checked(QuestionPrompt):
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:str
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    def __post_init__(self):
        self.correct_answers = {self.correct} # we don't give choices:, f"{self.correctKey})", self.correctKey}
            
        self.normalized_correct_answers = {normalize_answer(gold) for gold in self.correct_answers}
        self.stop_stemmed_correct_answers = {self.stop_stem_normalize_answer(gold) for gold in self.correct_answers}

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": False
                , "check_answer_key": True
                , "is_self_rated":self.has_rating()
                }
    def prompt_style(self)->str:
        return  "provide a complete and concise answer to the question based on the context"
    

    def has_rating(self):
        return False


    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        raise RuntimeError("This is a post-hoc answer checker")

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        raise RuntimeError("This is a post-hoc answer checker")


    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers



    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer)








@dataclass
class QuestionSelfRatedUnanswerablePromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]
    self_rater_tolerant:bool


    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher2=UnanswerableMatcher2(unanswerable_expressions=set())
        if self.self_rater_tolerant:
            self.self_rater = SelfRaterTolerant(self.unanswerable_matcher2)
        else:
            self.self_rater = SelfRaterStrict(self.unanswerable_matcher2)

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"orig_prompt_class": "unknown"
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                , "is_self_rater_tolerant": self.self_rater_tolerant
                }
    def prompt_style(self)->str:
        return  "Can the question be answered based on the available context? Choose one"
    

    def has_rating(self):
        return True

    pretext ='''Can the question be answered based on the available context? choose one:
        - 5: The answer is highly relevant, complete, and accurate.
        - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
        - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
        - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
        - 1: The answer is minimally relevant or complete, with substantial shortcomings.
        - 0: The answer is not relevant or complete at all.
        '''


    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:

        question_prompt =  f'{QuestionSelfRatedUnanswerablePromptWithChoices.pretext}\n Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        # question =  f'Is this question answerable: {self.question}'
        # question =  f'Is this question answerable: {self.question}'
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'{QuestionSelfRatedUnanswerablePromptWithChoices.pretext}\n Question: {self.question}'
        context_prompt = f"Context: {context}"

        # question =  f'Is this question answerable: {self.question}'
        prompt = self.prompt_truncater.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    def check_answer(self, answer:str)->bool:
        return self.self_rater.check_answer(answer)

    def check_answer_rating(self,answer:str)->int:
        return self.self_rater.check_answer_rating(answer)
    




@dataclass
class CustomQuestionSelfRatedPrompt(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]
    self_rater_tolerant:bool

    # prompt_text:str
    # prompt_hash:str
    # prompt_name:str
    # prompt_style_str:str

    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher2=UnanswerableMatcher2(unanswerable_expressions=set())
        if self.self_rater_tolerant:
            self.self_rater = SelfRaterTolerant(self.unanswerable_matcher2)
        else:
            self.self_rater = SelfRaterStrict(self.unanswerable_matcher2)

    def set_custom_prompt(self, prompt_name:str, prompt_text:str, prompt_style:str):
        self.prompt_text = prompt_text
        self.prompt_name = prompt_name
        self.prompt_style_str = prompt_style
        self.prompt_hash = get_md5_hash(prompt_name+prompt_text)


    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.prompt_name # self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                , "is_self_rater_tolerant": self.self_rater_tolerant
                , "prompt_hash": self.prompt_hash
                }
    
    def prompt_style(self)->str:
        return self.prompt_style_str
    
    def has_rating(self):
        return True

    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        filled_prompt_text = self.prompt_text.format(question=self.question,context=context)
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=filled_prompt_text, question="", max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        filled_prompt_text = self.prompt_text.format(question=self.question,context=context)
        context_prompt = f"Context: {context}"

        # question =  f'Is this question answerable: {self.question}'
        prompt = self.prompt_truncater.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=filled_prompt_text, max_length=max_token_len)
        return prompt

    def check_answer(self, answer:str)->bool:
        return self.self_rater.check_answer(answer)

    def check_answer_rating(self,answer:str)->int:
        return self.self_rater.check_answer_rating(answer)
    

@dataclass
class NuggetSelfRatedPrompt(NuggetPrompt):
    nugget_id:str
    nugget_text:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]
    self_rater_tolerant:bool


    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher2=UnanswerableMatcher2(unanswerable_expressions=self.unanswerable_expressions)
        if self.self_rater_tolerant:
            self.self_rater = SelfRaterTolerant(self.unanswerable_matcher2)
        else:
            self.self_rater = SelfRaterStrict(self.unanswerable_matcher2)


    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                , "self_rater_tolerant": self.self_rater_tolerant
                }

    def prompt_style(self)->str:
        return  "Is the nugget addressed..."
    

    def has_rating(self):
        return True

 

    pretext ='''Given the context, evaluate the coverage of the specified key fact (nugget). Use this scale:
        - 5: Detailed, clear coverage.
        - 4: Sufficient coverage, minor omissions.
        - 3: Mentioned, some inaccuracies or lacks detail.
        - 2: Briefly mentioned, significant omissions or inaccuracies.
        - 1: Minimally mentioned, largely inaccurate.
        - 0: Not mentioned at all.
        '''


    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        question_prompt =  f'{NuggetSelfRatedPrompt.pretext}\n Key fact: {self.nugget_text}\n'
        context_prompt = f"Context: {context}"
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    def check_answer(self, answer:str)->bool:
        return self.self_rater.check_answer(answer)

    def check_answer_rating(self,answer:str)->int:
        return self.self_rater.check_answer_rating(answer)
    




@dataclass
class NuggetExtractionPrompt(NuggetPrompt):
    nugget_id:str
    nugget_text:str
    query_id:str
    facet_id:Optional[str]
    query_text:str

    unanswerable_expressions:Set[str]

    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher=UnanswerableMatcher(self.unanswerable_expressions)
        self.unanswerable_expressions = self.unanswerable_matcher.unanswerable_expressions 


    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                }
    def prompt_style(self)->str:
        return  "Is the nugget addressed..."
    

    def has_rating(self):
        return False

    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        # f'''provide a complete and concise answer to the question based on the context. Question: {question}\nContext: {context}'''
        question_prompt =  f'Extract the passage from the text that best relates to the key fact (nugget), ensuring relevance and clarity. Key Fact: {self.nugget_text}\n'
        context_prompt = f"Context: {context}"
        prompt = self.prompt_truncater.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt



    # inverse logic!  we are scanning for non-answers!!!
    def check_answer(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer(answer)

    def check_answer_simple(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer_simple(answer)



    def check_answer_stemmed(self,answer:str)->bool:
        return self.unanswerable_matcher.check_answer_stemmed(answer)
        


## --------------
@dataclass
class DirectGradingPrompt(Prompt):
    query_id:str
    query_text:str
    facet_id:Optional[str]
    facet_text:Optional[str]

    my_prompt_type="direct_grading"
    prompt_truncater = PromptTruncater()

    def __post_init__(self):
        self.unanswerable_matcher2=UnanswerableMatcher2(unanswerable_expressions=set())
        self.true_false_matcher = TrueFalseMatcher2()

    def prompt_id(self)->str:
        return "direct_grading"

    def prompt_type(self)->str:
        return DirectGradingPrompt.my_prompt_type


    @abstractmethod
    def prompt_template(self, context:str)->str:
        pass

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": self.prompt_style()
                , "context_first": False
                , "check_unanswerable": False
                , "check_answer_key": False
                , "is_self_rated":self.has_rating() 
                }
    def prompt_style(self)->str:
        return  "Is this passage relevant?"
    

    def generate_prompt(self,context:str, model_tokenizer, max_token_len) -> str:
        empty_prompt =  self.prompt_template(context="")  # measure tokens in prompt template (without context)
        truncated_context = self.prompt_truncater.truncate_context(tokenizer=model_tokenizer, template=empty_prompt, context=context, max_length=max_token_len)
        prompt =  str.format(self.prompt_template(context=truncated_context))
        return prompt
    

    def check_answer(self, answer)->bool:
        if self.unanswerable_matcher2.check_unanswer(answer)==False:
            return False
        else:
            match_result = self.true_false_matcher.check_true_false({"True"}, answer=answer)
            if match_result is not None:
                return match_result
            else:
                return True
    

    def has_rating(self):
        return False
    
    def check_answer_rating(self,answer:str)->int:
        if self.check_answer(answer=answer):
            return 1
        else:
            return 0    


class SelfRatingDirectGradingPrompt(DirectGradingPrompt):
    def __post_init__(self):
        super().__post_init__()
        self.self_rater = SelfRaterStrict(self.unanswerable_matcher2)


    def has_rating(self):
        return True
    

    @abstractmethod
    def max_valid_rating(self)->int:
        '''maximum rating that is valid for the prompt, e.g. if ratings are 0,1,2 then this function should return 2.'''
        pass


    def check_answer_rating(self,answer:str)->int:
        rating = self.self_rater.check_answer_rating(answer)
        if rating > 2:
            return 0
        else:
            return rating

    def check_answer(self,answer:str)->bool:
        return self.check_answer_rating(answer=answer) > 0
        

@dataclass
class FagB(DirectGradingPrompt):
    def prompt_template(self, context:str)->str:
        return f'''Instruction: Indicate if the passage is relevant for the question. Respond with 'Yes' or 'No'.

Question: {self.query_text}
Passage: {context}
Answer:
'''


class FagB_few(DirectGradingPrompt):
    def prompt_template(self, context:str)->str:
        return f'''Instruction: Indicate if the passage is relevant for the question. Respond with 'Yes' or 'No'.

Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20 years ago to make 20 drops = 1mL. The viscosity of most things is temperature dependent, so this would be at room temperature. Hope this helps.
Question: how many eye drops per ml 
Answer: Yes 

Passage: RE: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 
drops per day.In the past other pharmacies have given me 3 10-ml bottles for 100 days.E: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles 
should last me 100 days but I run out way before that time when I am using 4 drops per day. 
Question: how many eye drops per ml 
Answer: No 

Passage: : You can transfer money to your checking account from other Wells Fargo. accounts through Wells Fargo Mobile Banking with the mobile app, online, at any. Wells Fargo ATM, or at a 
Wells Fargo branch. 1 Money in — deposits. 
Question: can you open a wells fargo account online 
Answer: No 

Passage: You can open a Wells Fargo banking account from your home or even online. It is really easy to do, provided you have all of the appropriate documentation. Wells Fargo has so many b
ank account options that you will be sure to find one that works for you. They offer free checking accounts with free online banking. 
Question: can you open a wells fargo account online 
Answer: Yes

Passage: {context}
Question: {self.query_text}
Answer:
'''

@dataclass
class HELM(DirectGradingPrompt):
    def prompt_template(self, context:str)->str:
        return f'''Instruction: Does the passage answer the que
        ry? Respond with 'Yes' or 'No'.

Query: {self.query_text}
Passage: {context}
Answer:
'''


@dataclass
class Sun(DirectGradingPrompt):
    def prompt_template(self, context:str)->str:
        return f'''Instruction: Given a passage and a query, predict whether the passage includes an answer to the query by producing either "Yes" or "No".
Query: {self.query_text}
Passage: {context}
Answer:
'''
    
@dataclass
class Sun_few(DirectGradingPrompt):
    def prompt_template(self, context:str)->str:
        return f'''Instruction: Given a passage and a query, predict whether the passage includes an answer to the query by producing either "Yes" or "No".

Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20 years ago to make 20 drops = 1mL. The viscosity of most things is temperature dependent, so this would be at room temperature. Hope this helps.
Query: how many eye drops per ml
Does the passage answer the query?
Answer: Yes

Passage: RE: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.In the past other pharmacies have given me 3 10-ml bottles for 100 days.E: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.
Query: how many eye drops per ml
Does the passage answer the query?
Answer: No

Passage: : You can transfer money to your checking account from other Wells Fargo. accounts through Wells Fargo Mobile Banking with the mobile app, online, at any. Wells Fargo ATM, or at a Wells Fargo branch. 1 Money in — deposits.
Query: can you open a wells fargo account online
Does the passage answer the query?
Answer: No

Passage: You can open a Wells Fargo banking account from your home or even online. It is really easy to do, provided you have all of the appropriate documentation. Wells Fargo has so many bank account options that you will be sure to find one that works for you. They offer free checking accounts with free online banking. 
Query: can you open a wells fargo account online
Does the passage answer the query?
Answer: Yes

Passage: {context}
Query: {self.query_text}

Does the passage answer the query?
Answer:
'''



@dataclass
class Thomas(SelfRatingDirectGradingPrompt):
    
    def prompt_template(self, context:str)->str:
        return f'''Instruction: You are a search quality rater evaluating the relevance of passages. Given a query and a passage, you must provide a score on an integer scale of 0 to 2 with the following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain other irrelevant content
0 = not relevant, should never be shown for this query

Query: {self.query_text}
Passage: {context}
Answer:
'''
    def max_valid_rating(self)->int:
        return 2

    # def check_answer_rating(self,answer:str)->int:
    #     rating = self.self_rater.check_answer_rating(answer)
    #     if rating > 2:
    #         return 0
    #     else:
    #         return rating

    # def check_answer(self,answer:str)->bool:
    #     return self.check_answer_rating(answer=answer) > 0
        
