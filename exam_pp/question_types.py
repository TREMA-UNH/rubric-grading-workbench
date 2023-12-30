from itertools import islice
# import dspy
import json
import torch
from dataclasses import dataclass

from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz


from typing import Dict, List, Any


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
    def int_key_to_str(i:int)->str:
        return f"chr(65+i)"

    def generate_prompt(self) -> str:
        return f"question: {self.question} choices: " + " ; ".join([f"{i}) {choice}" for i, choice in self.choices.items()])

    def generate_prompt_with_context(self,context:str) -> str:
        return f"context: {context}; question: {self.question}; choices: " + " ; ".join([f"{i}) {choice}" for i, choice in self.choices.items()])

    def generate_prompt_with_context_no_choices(self,context:str) -> str:
        return f"context: {context}; question: {self.question}"


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
        def is_fuzzy_match(stemmed_answer, stemmed_gold):
                return fuzz.ratio(stemmed_answer, stemmed_gold) > 80
        
        stemmed_answer = QuestionPromptWithChoices.normalize_answer(answer)
        is_fuzzy_match = any (is_fuzzy_match(stemmed_answer, stemmed_gold) 
                                   for stemmed_gold in self.normalized_correct_answers)

        return is_fuzzy_match



