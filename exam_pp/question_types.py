from itertools import islice
# import dspy
import json
import torch
from dataclasses import dataclass
from typing import *

from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz



@dataclass
class QuestionPromptWithChoices():
    question_id:str
    question:str
    choices:Dict[str,str]
    correct:str
    correctKey:str
    query_id:str
    query_text:str

    stemmer = PorterStemmer()

    def __post_init__(self):
        correct_answers = {self.correct, f"{self.correctKey})", self.correctKey}
        self.normalized_correct_answers = {QuestionPromptWithChoices.normalize_answer(gold) for gold in correct_answers}
        self.correct_answers = correct_answers

    @staticmethod
    def truncate_context_question_prompt(tokenizer, context, question, max_length):

        # Tokenize the question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)

        # Calculate the number of tokens available for the context
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        available_tokens_for_context = max_length - len(question_tokens) - num_special_tokens

        # Tokenize and truncate the context
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        truncated_context_tokens = context_tokens[:available_tokens_for_context]

        # Combine truncated context with the full question
        combined_tokens = truncated_context_tokens + question_tokens
        prompt = tokenizer.decode(combined_tokens)

        return prompt

    @staticmethod
    def truncate_context_question_prompt_QC(tokenizer, context:str, question:str, max_length:int):
        # print("check_truncation")
        # Tokenize the question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)

        # Calculate the number of tokens available for the context
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        available_tokens_for_context = max_length - len(question_tokens) - num_special_tokens -5  #5 for good measure

        # Tokenize and truncate the context
        # context_tokens = tokenizer.encode(context, add_special_tokens=False)
        # truncated_context_tokens = context_tokens[:available_tokens_for_context]
        context_tokens = tokenizer.encode(context, add_special_tokens=False, max_length = available_tokens_for_context, truncation=True)
        

        # Combine truncated context with the full question
        # combined_tokens = truncated_context_tokens + question_tokens
        # prompt = tokenizer.decode(combined_tokens)

        prompt = {
            'question': f'{tokenizer.cls_token}{question}',  # '<cls>Where do I live?'
            'context': tokenizer.decode(context_tokens)
        }
        # if available_tokens_for_context < len(context_tokens):
        #     print(f'truncating context of {len(context_tokens)} to {len(truncated_context_tokens)} prompt:\n{prompt}')

        # print("check_truncation")
        return prompt



    @staticmethod
    def int_key_to_str(i:int)->str:
        return f"chr(65+i)"

# to fix
    def generate_prompt(self,model_tokenizer, max_token_len)  -> str:
        prompt = f"question: {self.question} choices: " + " ; ".join([f"{i}) {choice}" for i, choice in self.choices.items()])
        return model_tokenizer.decode(model_tokenizer.encode(prompt, add_special_tokens=False, truncation = True, max_length = max_token_len-1))

    # def generate_prompt_with_context(self,context:str) -> str:
    #     return f"context: {context}; question: {self.question}; choices: " + " ; ".join([f"{i}) {choice}" for i, choice in self.choices.items()])

    def generate_prompt_with_context_no_choices(self,context:str, model_tokenizer, max_token_len) -> str:
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt(tokenizer=model_tokenizer, context=f"context: {context};", question=f" question: {self.question}", max_length=max_token_len)
        return prompt
    def generate_prompt_with_context_QC_no_choices(self,context:str, model_tokenizer, max_token_len) -> Dict[str,str]:
        prompt = QuestionPromptWithChoices.truncate_context_question_prompt_QC(tokenizer=model_tokenizer, context=f"context: {context}", question=f" question: {self.question}", max_length=max_token_len)
        return prompt


    def check_answer(self,answer:str)->bool:
        # return self.check_answer_simple(answer) or self.check_answer_stemmed(answer)
        return self.check_answer_stemmed(answer)


    def check_answer_simple(self,answer:str)->bool:
        return answer in self.correct_answers


    @staticmethod
    def normalize_answer(answer:str)->str:
        # Lowercase, Perform other normalization like removing punctuation, if necessary
        # Stem the answer
        return QuestionPromptWithChoices.stemmer.stem(answer.lower())

    def check_answer_stemmed(self,answer:str)->bool:
        def is_fuzzy_match(stemmed_answer:str, stemmed_gold:str)->bool:
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = QuestionPromptWithChoices.normalize_answer(answer)
        is_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_correct_answers)

        return is_match



