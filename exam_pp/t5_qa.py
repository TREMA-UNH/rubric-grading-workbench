import itertools
import math
import os
from pathlib import Path
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig,AutoModelForQuestionAnswering,AutoTokenizer
from question_types import QuestionPromptWithChoices


os.environ["DSP_NOTEBOOK_CACHEDIR"] = str((Path(".") / "cache").resolve())
device:Optional[int] = None
deviceStr = os.environ.get("GPU_DEVICE")
if deviceStr is not None:
    try:
        device = int(deviceStr)
    except ValueError:
        print(f'Cant parse device number from \"{device}\"')
        device = None

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_TOKEN_LEN = 512
print(f'Device = {device}; BATCH_SIZE = {BATCH_SIZE}')


PromptGenerator = Callable[[QuestionPromptWithChoices],str]
PromptGeneratorQC = Callable[[QuestionPromptWithChoices],Dict[str,str]]



def computeMaxBatchSize(modelConfig:PretrainedConfig)-> int:
    gpu_memory = 45634    # A40
    # Constants
    memory_for_activations_mib = gpu_memory / 2  # Half of the total GPU memory
    d_model = modelConfig.d_model  # 1024 Model dimension
    token_length = MAX_TOKEN_LEN   # 512 Maximum token length
    bytes_per_parameter = 4  # FP32 precision

    # Calculating the maximum batch size
    # Memory required per token in a batch (in MiB)
    memory_per_token_mib = d_model**2 * bytes_per_parameter / (1024**2)

    # Total memory required for one batch of size 1
    total_memory_per_batch_mib = token_length * memory_per_token_mib

    # Estimate the maximum batch size
    max_batch_size = memory_for_activations_mib / total_memory_per_batch_mib
    return math.floor(max_batch_size)



class McqaPipeline():
    """QA Pipeline for multi-choice question answering"""


    # def __init__(self):
    #     """promptGenerator for a particular question. 
    #        Example usages: 
    #           * `promptGenerator=lambda qpc: qpc.generate_prompt()`
    #           * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
    #        """
    #     # Initialize the tokenizer and model
    #     self.modelName = 'google/flan-t5-large'
    #     self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
    #     self.model = T5ForConditionalGeneration.from_pretrained(self.modelName)
    #     print(f"T5 model config: { self.model.config}")
    #     # self.promptGenerator = promptGenerator

    #     # Create a Hugging Face pipeline
    #     self.t5_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE)


    def __init__(self):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """
        # Initialize the tokenizer and model
        self.modelName = 'sjrhuschlee/flan-t5-large-squad2'
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelName)
        # self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"T5 model config: { self.model.config}")
        # self.promptGenerator = promptGenerator

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE)

    def exp_modelName(self)->str:
        return self.modelName

    # def answer_multiple_choice_question(self, qpc:QuestionPromptWithChoices, promptGenerator:PromptGenerator)->str:
    #     prompt = promptGenerator(qpc)

    #     # print("prompt",prompt)
    #     outputs = self.t5_pipeline(prompt, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
    #     return outputs[0]['generated_text']
        

    # def batch_answer_multiple_choice_questions(self, qpcs:List[QuestionPromptWithChoices], promptGenerator:PromptGenerator)->List[Tuple[QuestionPromptWithChoices, str]]:
    #     """Prepare a batch for question answering, tuple it up with the answers"""
    #     prompts = [promptGenerator(qpc) for qpc in qpcs]
        
    #     outputs = self.t5_pipeline(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
    #     answers = [output['generated_text'] for output in outputs]
    #     return zip(qpcs, answers)
        

    def batch_answer_multiple_choice_questions_QC(self, qpcs:List[QuestionPromptWithChoices], promptGenerator:PromptGeneratorQC)->Iterable[Tuple[QuestionPromptWithChoices, str]]:
        """Prepare a batch for question answering, tuple it up with the answers"""
        prompts = [promptGenerator(qpc) for qpc in qpcs]
        
        outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
        answers:List[str] = [output['answer'] for output in outputs]
        return zip(qpcs, answers)
        

class BatchingPipeline():
    def __init__(self, batchSize:int):
        self.batchSize = 100 # batchSize
    

    # def answerQuestions(self, questions: List[QuestionPromptWithChoices], pipeline:McqaPipeline, promptGenerator:PromptGenerator):
    #     for qpc in questions:
    #         answer = pipeline.answer_multiple_choice_question(qpc,promptGenerator)
    #         print("answer", answer)
    #         print("correct?", qpc.check_answer(answer))


    def batchAnswerQuestions(self, questions: List[QuestionPromptWithChoices], pipeline:McqaPipeline, promptGenerator:PromptGeneratorQC)->List[Tuple[QuestionPromptWithChoices, str]]:
        """runs question answering over a single batch, and tuples it up with answers"""
        answerTuples = list(pipeline.batch_answer_multiple_choice_questions_QC(questions, promptGenerator))
        # print('correct list?', [qpc.check_answer(answer) for qpc,answer in answerTuples])
        return answerTuples

    # def batchAnswerQuestions(self, questions: List[QuestionPromptWithChoices], pipeline:McqaPipeline, promptGenerator:PromptGenerator)->List[Tuple[QuestionPromptWithChoices, str]]:
    #     """runs question answering over a single batch, and tuples it up with answers"""
    #     answerTuples = list(pipeline.batch_answer_multiple_choice_questions(questions, promptGenerator))
    #     # print('correct list?', [qpc.check_answer(answer) for qpc,answer in answerTuples])
    #     return answerTuples

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.batchSize))
            if not batch or len(batch)<1:
                break
            yield batch


    def chunkingBatchAnswerQuestions(self, questions:List[QuestionPromptWithChoices], pipeline:McqaPipeline, promptGenerator:PromptGeneratorQC)->List[Tuple[QuestionPromptWithChoices, str]]:
            """Run question answering over batches of questions, and tuples it up with the answers"""
            return list(itertools.chain.from_iterable(
                        (self.batchAnswerQuestions(batch, pipeline, promptGenerator) for batch in self.batchChunker(questions)) 
                        )) 

def main():
    import tqa_loader
    """Entry point for the module."""
    lesson_questions = tqa_loader.load_all_tqa_data()[0:2]
    
    
    qa = McqaPipeline()
    batchPipe = BatchingPipeline(BATCH_SIZE)

    promptGenerator=lambda qpc: qpc.generate_prompt(model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = batchPipe.chunkingBatchAnswerQuestions(questions, qa, promptGenerator)
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")

        


if __name__ == "__main__":
    main()

