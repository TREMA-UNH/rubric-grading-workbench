import asyncio

import itertools
import math
import os
from pathlib import Path
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
import typing
import torch
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig,AutoModelForQuestionAnswering,AutoTokenizer

from .test_bank_prompts import Prompt, QuestionPromptWithChoices,QuestionPrompt
from .batched_worker import BatchedWorker

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


PromptGenerator = Callable[[Prompt],str]
PromptGeneratorQC = Callable[[Prompt],Dict[str,str]]



def computeMaxBatchSize(modelConfig:PretrainedConfig)-> int:
    '''Estimates the batch size possible with a given model and given GPU memory constraints'''
    # TODO: make this its own script


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



class QaPipeline():
    """QA Pipeline for squad question answering"""

    def __init__(self, model_name:str):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """
        self.question_batchSize = 100 # batchSize
    
        # Initialize the tokenizer and model
        # self.modelName = 'sjrhuschlee/flan-t5-large-squad2'
        self.modelName = model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelName)
        # self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"QaPipeline model config: { self.model.config}")
        # print("maxBatchSize",computeMaxBatchSize(self.model.config))
        # self.promptGenerator = promptGenerator
        self.max_token_len = 512

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)

    def exp_modelName(self)->str:
        return self.modelName


    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch


    def chunkingBatchAnswerQuestions(self, questions:List[Prompt],  paragraph_txt:str)->List[Tuple[Prompt, str]]:
            """Run question answering over batches of questions, and tuples it up with the answers"""
            promptGenerator=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

            def processBatch(qpcs:List[Prompt])->Iterable[Tuple[Prompt, str]]:
                """Prepare a batch for question answering, tuple it up with the answers"""
                prompts = [promptGenerator(qpc) for qpc in qpcs]
                
                outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
                answers:List[str] = [output['answer'] for output in outputs]
                return zip(qpcs, answers, strict=True)

            return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(questions)) 
                        )) 

class Text2TextPipeline():
    """QA Pipeline for text2text based question answering"""

    def __init__(self, model_name:str):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """

        self.question_batchSize = 100 # batchSize
        self.batcher: Optional[BatchedWorker] = None

        # Initialize the tokenizer and model
        # self.modelName = 'google/flan-t5-large'
        self.modelName = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.modelName)
        # self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"Text2Text model config: { self.model.config}")
        # print("maxBatchSize",computeMaxBatchSize(self.model.config))
        # self.promptGenerator = promptGenerator
        self.max_token_len = 512

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)


    def call_pipeline(self, prompts: List[str]) -> List[str]:
        resps = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
        return [resp['generated_text'] for resp in resps]

    def finish(self) -> None:
        if self.batcher is not None:
            self.batcher.finish()


    def exp_modelName(self)->str:
        return self.modelName


    async def chunkingBatchAnswerQuestions_async(self, prompts:List[Prompt],  paragraph_txt:str) -> List[Tuple[Prompt, str]]:
        if self.batcher is None:
            self.batcher = BatchedWorker[str, str](func = self.call_pipeline, batch_size = self.question_batchSize)

        promptGenerator=lambda prompt: prompt.generate_prompt(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)
        return [(prompt, await self.batcher.run(promptGenerator(prompt))) for prompt in prompts]


    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch

    def chunkingBatchAnswerQuestions(self, prompts:List[Prompt],  paragraph_txt:str)->List[Tuple[Prompt, str]]:
            """Run question answering over batches of questions, and tuples it up with the answers"""
            promptGenerator=lambda prompt: prompt.generate_prompt(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

            def processBatch(prompt_batch:List[Prompt])->Iterable[Tuple[Prompt, str]]:
                """Prepare a batch for question answering, tuple it up with the answers"""
                prompts = [promptGenerator(prompt) for prompt in prompt_batch]
                
                outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
                answers:List[str] = [output['generated_text']  for output in outputs]
                return zip(prompt_batch, answers, strict=True)

            return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(prompts)) 
                        )) 


class LlamaTextGenerationPipeline():
    """Llama Text Generation Pipeline for text-generation based question answering"""

    def __init__(self, model_name:str):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """
        self.question_batchSize = 100 # batchSize 
    
        # Initialize the tokenizer and model
        self.modelName = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"Text generation model config: { self.model.config}")
        # print("maxBatchSize",computeMaxBatchSize(self.model.config))
        self.max_token_len = 512

        # in order to support batching in Llama
        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side ='left'

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('text-generation'
                                       , model=self.model
                                       , tokenizer=self.tokenizer
                                       , device=device
                                       , batch_size=BATCH_SIZE
                                       , use_fast=True
                                       , model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}}
                                    #    , device_map="auto"
                                       )


    def exp_modelName(self)->str:
        return self.modelName


    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch

    def chunkingBatchAnswerQuestions(self, questions:List[Prompt],  paragraph_txt:str)->List[Tuple[Prompt, str]]:
            """Run question answering over batches of questions, and tuples it up with the answers"""
            promptGenerator=lambda qpc: qpc.generate_prompt(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

            def processBatch(qpcs:List[Prompt])->Iterable[Tuple[Prompt, str]]:
                """Prepare a batch for question answering, tuple it up with the answers"""
                #prompts = [[{"role":"user", "content": promptGenerator(qpc)}] for qpc in qpcs]
                # prompts = [[promptGenerator(qpc)+" must be a value between 0-5!\n Answer: "] for qpc in qpcs]
                prompts = [(promptGenerator(qpc)+" Rate how well the passage answers the question by responding with a code between 0 and 5.\n Answer:") for qpc in qpcs]

                # messages = [
                #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                #     {"role": "user", "content": "Who are you?"},
                # ]

                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                
                answers:List[str] = list()

                output = self.t5_pipeline_qa(prompts, max_new_tokens=100 #, max_length=MAX_TOKEN_LEN, 
                                                 , eos_token_id=terminators
                                                 , pad_token_id = self.tokenizer.pad_token_id
                                                 , do_sample=True
                                                 , temperature=0.6
                                                 , top_p=0.9)
                for index, prompt in enumerate(prompts):
                    # print("Llama output\n", output)
                    raw_answer = output[index][-1]['generated_text']
                    answer = raw_answer[len(prompt):].strip()

                    

                    # print("--\n"+answer)
                    answers.append(answer)

                    # print("Llama pipeline outputs:\n", output)
                # answers:List[str] = [output['generated_text'][-1]  for output in outputs]
                return zip(qpcs, answers, strict=True)

            return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(questions)) 
                        ))     

class TextGenerationPipeline():
    """QA Pipeline for text-generation based question answering"""

    def __init__(self, model_name:str):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """
        self.question_batchSize = 100 # batchSize
    
        # Initialize the tokenizer and model
        # self.modelName = 'mistralai/Mistral-7B-v0.1'
        # self.modelName = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        # self.modelName = 'gpt2-large'
        self.modelName = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        # self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"Text generation model config: { self.model.config}")
        # print("maxBatchSize",computeMaxBatchSize(self.model.config))
        # self.promptGenerator = promptGenerator
        self.max_token_len = 512

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)


    def exp_modelName(self)->str:
        return self.modelName


    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch


    def chunkingBatchAnswerQuestions(self, questions:List[Prompt],  paragraph_txt:str)->List[Tuple[Prompt, str]]:
            """Run question answering over batches of questions, and tuples it up with the answers"""
            promptGenerator=lambda qpc: qpc.generate_prompt(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

            def processBatch(qpcs:List[Prompt])->Iterable[Tuple[Prompt, str]]:
                """Prepare a batch for question answering, tuple it up with the answers"""
                prompts = [promptGenerator(qpc) for qpc in qpcs]
                
                outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
                answers:List[str] = [output['generated_text']  for output in outputs]
                return zip(qpcs, answers, strict=True)

            return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(questions)) 
                        )) 



def mainQA():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data(self_rater_tolerant=False)[0:2]
    
    
    qa = QaPipeline('sjrhuschlee/flan-t5-large-squad2')

    # promptGenerator=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(context='', model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = qa.chunkingBatchAnswerQuestions(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")

        

def mainT2T():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data()[0:2]
    
    
    # qa = Text2TextPipeline('google/flan-t5-large')
    qa = Text2TextPipeline('google/flan-t5-small')
    # promptGenerator=lambda qpc: qpc.generate_prompt(context = '', model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = qa.chunkingBatchAnswerQuestions(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")




if __name__ == "__main__":
    mainT2T()

