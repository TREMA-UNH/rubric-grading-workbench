# exam_pp/score_psgs_with_questions.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz
    

import parse_qrels_runs_with_text


def my_function():
    """Provide a short description of what this function does."""
    pass

class MyClass:
    """Provide a short description of what this class represents."""

    def __init__(self, attribute1, attribute2):
        self.attribute1 = attribute1
        self.attribute2 = attribute2

    def my_method(self):
        """Provide a short description of what this method does."""
        pass


def loadTQA():
    import tqa_loader
    tqa_loader.load_all_tqa_data()

    # file = open('tqa_train_val_test/train/tqa_v1_train.json')
    # for lesson in islice(json.load(file), 2):
    #     query = lesson['lessonName']
    #     results = [{'text':'This is a text'}] 
    #     # results = rm(query, k=1)
    #     passages = [x['text'] for x in results]
    #     print('#', query)
    #     print(passages[0])
    #     print()

    #     for qid, q in islice(lesson['questions']['nonDiagramQuestions'].items(), 10):
    #         question = q['beingAsked']['processedText']
    #         answer = q['answerChoices'].get(q['correctAnswer']['processedText'])
    #         if answer is None:
    #             print('bad question', q)
    #             continue

    #         choices = {
    #             f'({n})': choice['processedText']
    #             for n, choice in q['answerChoices'].items()
    #         }
            
    #         context = ' '.join(islice(passages[0].split(' '),150))
    #         context += '?' + ', '.join(
    #             f" {k} {v}" for k,v in choices.items()
    #         )

    #         print("question", question)
    #         print("context", context)
            
    #         resp = t5_qa(context=context, question=question)

    #         choice = choices.get(resp['answer'])
            

            


    #         print('Q:  ', question)
    #         print('C:  ', context)
    #         # print('Cs: ', ', '.join(x['processedText'] for x in q['answerChoices'].values()))
    #         print('A*: ', answer['processedText'])
    #         print('A:  ', choice, resp['answer'])
    #         print()





def main():
    """Entry point for the module."""
    # You can add code here that will be executed if the module is run as a script.
    x = parse_qrels_runs_with_text.parseQueryWithFullParagraphs("./benchmarkY3test-qrels-with-text.jsonl.gz")
    print(x[0])

if __name__ == "__main__":
    main()
