import abc
import string
from typing import *
import re
from abc import abstractmethod
from dataclasses import dataclass

from nltk.stem import PorterStemmer
import nltk
from fuzzywuzzy import fuzz

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')  

def get_prompt_classes()->List[str]:
    return ['QuestionPromptWithChoices'
            , 'QuestionAnswerablePromptWithChoices'
            , 'QuestionCompleteConciseUnanswerablePromptWithChoices'
            , 'QuestionCompleteConcisePromptWithAnswerKey'
            , 'QuestionCompleteConcisePromptWithAnswerKey2'
            , 'QuestionSelfRatedUnanswerablePromptWithChoices'
            , 'QuestionSelfRatedExplainPrompt'
            , 'QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2'
            ]



@dataclass
class QuestionPrompt(abc.ABC):
    question_id:str

    @abstractmethod
    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        pass

    @abstractmethod
    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        pass

    @abstractmethod
    def check_answer(self,answer:str)->bool:
        pass

    def check_answer_rating(self,answer:str)->int:
        if self.check_answer(answer=answer):
            return 1
        else:
            return 0

    @abstractmethod
    def has_rating(self):
        return False
    
    def answer_match_info(self):
        return ""

def normalize_answer(answer:str)->str:
    # Lowercase, Perform other normalization like removing punctuation, if necessary
    # Stem the answer
    return QuestionPromptWithChoices.stemmer.stem(answer.lower())


@dataclass
class QuestionPromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:str
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    stemmer = PorterStemmer()

    def __post_init__(self):
        correct_answers = {self.correct} # we don't give choices:, f"{self.correctKey})", self.correctKey}
        if self.correct.lower().strip() =="false":
            self.correct_answers += "no"
            self.correct_answers += "incorrect"
            self.correct_answers += "wrong"
        if self.correct.lower().strip() =="true":
            self.correct_answers += "yes"
            self.correct_answers += "correct"
            
        self.normalized_correct_answers = {normalize_answer(gold) for gold in correct_answers}
        self.correct_answers = correct_answers

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": "context: question:"
                , "context_first": True
                , "check_unanswerable": False
                , "check_answer_key": True
                , "is_self_rated":self.has_rating()
                }

    def has_rating(self):
        return False


    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

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
        # prompt = tokenizer.decode(combined_tokens)

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt

    def check_answer_rating(self, answer: str) -> int:
        return super().check_answer_rating(answer)
    


    @staticmethod
    def int_key_to_str(i:int)->str:
        return f"chr(65+i)"



    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        question = self.question
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=f"context: {context};", question=f" question: {question}", max_length=max_token_len)
        return prompt
    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question = self.question
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=f"context: {context}", question=f" question: {question}", max_length=max_token_len)
        return prompt


    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer) or self.check_answer_stemmed(answer)
        # return self.check_answer_stemmed(answer)


    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers


    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_correct_answers)

        return is_match




@dataclass
class QuestionAnswerablePromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]

    stemmer = PorterStemmer()

    def __post_init__(self):
        self.unanswerable_expressions = self.unanswerable_expressions | {"unanswerable"
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

    def check_answer_rating(self, answer: str) -> int:
        return super().check_answer_rating(answer)
    

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": "How does this text answer this question:"
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                }

    def has_rating(self):
        return False

    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

        # Tokenize the question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)

        # Calculate the number of tokens available for the context
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        available_tokens_for_context = max_length - len(question_tokens) - num_special_tokens -5 # 5 for good measure

        # Tokenize and truncate the context
        truncated_context_tokens = tokenizer.encode(context, add_special_tokens=False, max_length = available_tokens_for_context, truncation=True)

        # Combine truncated context with the full question
        combined_tokens = truncated_context_tokens + question_tokens
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

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt



    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        question_prompt =  f' question: How does this text answer this question: {self.question}'
        context_prompt = f"context: {context};"
        # question =  f'Is this question answerable: {self.question}'
        # question =  f'Is this question answerable: {self.question}'
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f' question: How does this text answer this question: {self.question}'
        context_prompt = f"context: {context};"
        # question =  f'Is this question answerable: {self.question}'
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    # inverse logic!  we are scanning for non-answers!!!
    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer) and self.check_answer_stemmed(answer)



    def check_answer_simple(self,answer:str)->bool:
        return not (answer in self.unanswerable_expressions)



    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        if len(stemmed_answer)<1:
            return False
        
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_unanswerable_expressions)

        return not is_match





@dataclass
class QuestionCompleteConciseUnanswerablePromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]

    stemmer = PorterStemmer()

    def __post_init__(self):
        self.unanswerable_expressions = self.unanswerable_expressions | {"unanswerable"
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


    def check_answer_rating(self, answer: str) -> int:
        return super().check_answer_rating(answer)
    
    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": "provide a complete and concise answer to the question based on the context."
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                }

    def has_rating(self):
        return False


    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

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
        # prompt = tokenizer.decode(combined_tokens)

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt



    # def generate_prompt(self,model_tokenizer, max_token_len)  -> str:
    #     prompt = f"question: {self.question} choices: " + " ; ".join([f"{i}) {choice}" for i, choice in self.choices.items()])
    #     return model_tokenizer.decode(model_tokenizer.encode(prompt, add_special_tokens=False, truncation = True, max_length = max_token_len-1))

    # def generate_prompt_with_context(self,context:str) -> str:
    #     return f"context: {context}; question: {self.question}; choices: " + " ; ".join([f"{i}) {choice}" for i, choice in self.choices.items()])

    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        # f'''provide a complete and concise answer to the question based on the context. Question: {question}\nContext: {context}'''
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        # question =  f'Is this question answerable: {self.question}'
        # question =  f'Is this question answerable: {self.question}'
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}'
        context_prompt = f"Context: {context}"

        # question =  f'Is this question answerable: {self.question}'
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    # inverse logic!  we are scanning for non-answers!!!
    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer) and self.check_answer_stemmed(answer)



    def check_answer_simple(self,answer:str)->bool:
        return not (answer in self.unanswerable_expressions)


    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        if len(stemmed_answer)<1:
            return False
        
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_unanswerable_expressions)

        return not is_match







@dataclass
class QuestionCompleteConcisePromptWithAnswerKey(QuestionPrompt):
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:str
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    stemmer = PorterStemmer()

    def __post_init__(self):
        self.correct_answers = {self.correct} # we don't give choices:, f"{self.correctKey})", self.correctKey}
        if self.correct.lower().strip() =="false":
            self.correct_answers.add("no")
            self.correct_answers.add("incorrect")
            self.correct_answers.add("wrong")
        if self.correct.lower().strip() =="true":
            self.correct_answers.add("yes")
            self.correct_answers.add("correct")
            
        self.normalized_correct_answers = {normalize_answer(gold) for gold in self.correct_answers}

    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"prompt_style": "provide a complete and concise answer to the question based on the context."
                , "context_first": False
                , "check_unanswerable": False
                , "check_answer_key": True
                , "is_self_rated":self.has_rating()
                }

    def has_rating(self):
        return False


    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

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
        # prompt = tokenizer.decode(combined_tokens)

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt


    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        # f'''provide a complete and concise answer to the question based on the context. Question: {question}\nContext: {context}'''
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}'
        context_prompt = f"Context: {context}"
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    def check_answer(self,answer:str)->bool:
        return self.check_answer_simple(answer) or self.check_answer_stemmed(answer)
        # return self.check_answer_stemmed(answer)


    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers


    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_correct_answers)

        return is_match


@dataclass
class QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2(QuestionPrompt):
    '''This is an answer-verifier to be used to regrade any QA prompt with explicit answers.'''
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:str
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

    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        return ""


    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        return {}


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
class QuestionCompleteConcisePromptWithAnswerKey2(QuestionPrompt):
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:str
    correctKey:Optional[str]
    query_id:str
    facet_id:Optional[str]
    query_text:str

    stemmer = PorterStemmer()

    def __post_init__(self):
        self.correct=self.strip_trailing_period(self.correct)
        if self.correct.endswith('.'):
            self.correct = self.correct[:-1]

        self.correct_answers = {self.correct} # we don't give choices:, f"{self.correctKey})", self.correctKey}

        self.normalized_correct_answers = {normalize_answer(gold) for gold in self.correct_answers}
        self.stop_stemmed_correct_answers = {self.stop_stem_normalize_answer(gold) for gold in self.correct_answers}

    def answer_match_info(self)->str:
        return "lowercase, stopped, stemmed, removed trailing period, fuzz > 0.8 / true-false special handling"

    def strip_trailing_period(self, answer):
        answer_ =answer.strip()
        if answer_.endswith('.'):
            answer_ = answer_[:-1]
        return answer_
        
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

    def has_rating(self):
        return False


    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

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
        context_tokens = tokenizer.encode(context, add_special_tokens=False, max_length = available_tokens_for_context, truncation=True)

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt


    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        # f'''provide a complete and concise answer to the question based on the context. Question: {question}\nContext: {context}'''
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'provide a complete and concise answer to the question based on the context. Question: {self.question}'
        context_prompt = f"Context: {context}"
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt



    def stop_stem_normalize_answer(self, text:str)->str:
        # Convert text to lowercase
        text = text.lower().strip()


        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]

        # Rejoin words
        normalized_text = ' '.join(stemmed_tokens)

        return normalized_text

    def check_answer(self,answer:str)->bool:
        answer_=self.strip_trailing_period(answer)

        checkTF = self.check_true_false(answer_)
        if checkTF is not None:
            return checkTF
        
        stemmedAnswer = self.check_answer_stemmed(answer_)
        if stemmedAnswer is not None:
            return stemmedAnswer
        return self.check_answer_simple(answer_)


    def check_true_false(self, answer:str)->Optional[bool]:
        FALSE_answers = {"no", "incorrect","false"}
        TRUE_answers = {"yes", "correct","true"}
        answer_ = self.correct.lower().strip()

        if answer_ == "false":
            if answer.lower() in FALSE_answers:
                return True
            else:
                return False

        if answer_ == "true":
            if answer.lower() in TRUE_answers:
                return True
            else:
                return False
            
        return None
            
                

    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers


    def check_answer_stemmed(self,answer:str)->Optional[bool]:
        # def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
        #         return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = self.stop_stem_normalize_answer(answer)
        # is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
        #                            for stemmed_gold in self.normalized_correct_answers)

        if len(stemmed_answer) >=2:
            is_match = stemmed_answer in self.stop_stemmed_correct_answers
            if is_match: 
                return is_match

        if len(stemmed_answer) >=4:
            is_fuzzy = any (fuzz.ratio(stemmed_answer, stemmed_gold) > 80 for stemmed_gold in self.stop_stemmed_correct_answers)
            return is_fuzzy
        return None




# @dataclass
# class QuestionCompleteConcisePromptT5Checked(QuestionPrompt):
#     question_id:str
#     question:str
#     choices:Dict[str,str]
#     correct:str
#     correctKey:Optional[str]
#     query_id:str
#     facet_id:Optional[str]
#     query_text:str

#     stemmer = PorterStemmer()

#     def __post_init__(self):
#         self.correct_answers = {self.correct} # we don't give choices:, f"{self.correctKey})", self.correctKey}
            
#         self.normalized_correct_answers = {normalize_answer(gold) for gold in self.correct_answers}
#         self.stop_stemmed_correct_answers = {self.stop_stem_normalize_answer(gold) for gold in self.correct_answers}

#     def prompt_info(self)-> Dict[str, Any]:
#         return {"prompt_class": self.__class__.__name__
#                 ,"prompt_style": "provide a complete and concise answer to the question based on the context."
#                 , "context_first": False
#                 , "check_unanswerable": False
#                 , "check_answer_key": True
#                 , "is_self_rated":self.has_rating()
#                 }

#     def has_rating(self):
#         return False


#     def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
#         raise RuntimeError("This is a post-hoc answer checker")

#     def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
#         raise RuntimeError("This is a post-hoc answer checker")




#     def check_answer(self,answer:str)->bool:
#         return self.check_answer_simple(answer)








@dataclass
class QuestionSelfRatedUnanswerablePromptWithChoices(QuestionPrompt):
    question_id:str
    question:str
    query_id:str
    facet_id:Optional[str]
    query_text:str
    unanswerable_expressions:Set[str]

    stemmer = PorterStemmer()

    def __post_init__(self):
        self.unanswerable_expressions = self.unanswerable_expressions.union( {"unanswerable"
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


    def prompt_info(self, old_prompt_info:Optional[Dict[str,Any]]=None)-> Dict[str, Any]:
        return {"prompt_class": self.__class__.__name__
                ,"orig_prompt_class": "unknown"
                ,"prompt_style": "N/A  (this prompt re-verifies answers produced by \"orig_prompt_class\")"
                , "context_first": False
                , "check_unanswerable": True
                , "check_answer_key": False
                , "is_self_rated":self.has_rating()
                }

    def has_rating(self):
        return True

    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

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
        # prompt = tokenizer.decode(combined_tokens)

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        return prompt


    pretext ='''Can the question be answered based on the available context? choose one:
        - 5: The answer is highly relevant, complete, and accurate.
        - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
        - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
        - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
        - 1: The answer is minimally relevant or complete, with substantial shortcomings.
        - 0: The answer is not relevant or complete at all.
        '''


    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:


        question_prompt =  f'{QuestionSelfRatedUnanswerablePromptWithChoices.pretext}\n Question: {self.question}\n'
        context_prompt = f"Context: {context}"
        # question =  f'Is this question answerable: {self.question}'
        # question =  f'Is this question answerable: {self.question}'
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt

    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        question_prompt =  f'{QuestionSelfRatedUnanswerablePromptWithChoices.pretext}\n Question: {self.question}'
        context_prompt = f"Context: {context}"

        # question =  f'Is this question answerable: {self.question}'
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=context_prompt, question=question_prompt, max_length=max_token_len)
        return prompt


    def check_answer(self, answer:str)->bool:
        rating = self.check_answer_rating(answer=answer)
        return rating > 0

    
    # self-rated logic. We are scanning for 0-5. 
    # other answers get rating 1
    # unanserable expressions get rating 0
    def check_answer_rating(self,answer:str)->int:
        rating:int 
        # Regex to match a string with only one digit (0-5), and possibly other non-letter characters
        match = re.fullmatch(r'[^\w]*([0-5])[^\w]*', answer)
        if match:
            rating = int(match.group(1))
        elif self.check_unanswer(answer):
            rating = 0
        else:
            rating = 1

        return rating


    # inverse logic!  we are scanning for non-answers!!!
    def check_unanswer(self,answer:str)->bool:
        return self.check_unanswer_simple(answer) and self.check_unanswer_stemmed(answer)



    def check_unanswer_simple(self,answer:str)->bool:
        return not (answer in self.unanswerable_expressions)


    def check_unanswer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = normalize_answer(answer)
        if len(stemmed_answer)<1:
            return False
        
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_unanswerable_expressions)

        return not is_match




