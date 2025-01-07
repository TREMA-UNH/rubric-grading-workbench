import argparse
import asyncio

from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable, Union
import typing
import openai
import torch
import torch.nn.functional as F
from transformers import pipeline, T5ForConditionalGeneration, GPT2TokenizerFast, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig,AutoModelForQuestionAnswering,AutoTokenizer

from json import JSONDecodeError

import transformers

from .data_model import FullParagraphData, ParagraphData
from .exam_llm import *
from . import openai_interface
from .openai_interface import query_gpt_batch_with_rate_limiting, OpenAIRateLimiter, FetchGptJson


from .test_bank_prompts import Prompt, QuestionPromptWithChoices,QuestionPrompt
from .batched_worker import BatchedWorker

os.environ["DSP_NOTEBOOK_CACHEDIR"] = str((Path(".") / "cache").resolve())

device:torch.device = torch.device("cpu")
deviceStr = os.environ.get("GPU_DEVICE")
if deviceStr is not None:
    try:
        device = torch.device(deviceStr)
    except ValueError:
        print(f'Cant parse device string from \"{device}\"')
        print("Cuda available? ",torch.cuda.is_available())
        device = torch.device("cpu")



BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_TOKEN_LEN = 200
print(f'Device = {device}; BATCH_SIZE = {BATCH_SIZE}')


def clean_hf_kwargs(kwargs):
    hf_kwargs = kwargs.copy()
    hf_kwargs.pop("record_embeddings", None)
    return hf_kwargs


PromptGenerator = Callable[[Prompt],str]
PromptGeneratorQC = Callable[[Prompt],Dict[str,str]]


def create_gpt_client()->openai.OpenAI:
    return openai_interface.default_openai_client()

def create_vllm_client(base_url:str|None=os.getenv('VLLM_URL'))->openai.OpenAI:
    if base_url is None and os.getenv('VLLM_URL') is None:
        raise RuntimeError ("Must set environment variable \"VLLM_URL\". For localhost use \'http://[::0]:8000/v1\' ")

    return openai_interface.createOpenAIClient(api_key="NONE", base_url=base_url)

class FetchGptGrade(FetchGptJson):
    def __init__(self, gpt_model:str, max_tokens:int, client:openai.OpenAI, use_chat_protocol:True):
        super().__init__(gpt_model=gpt_model, max_tokens=max_tokens, client=client, use_chat_protocol=use_chat_protocol)


        json_instruction= r'''
Give the response in the following JSON format:
```json
{ "grade": int }
```'''
        self.set_json_instruction(json_instruction, field_name="grade")



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


from enum import Enum, auto
from abc import ABC, abstractmethod

        
class HfPipeline(Enum):
    text2text = auto()
    textgeneration = auto()
    llama = auto()
    qa = auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(arg:str):
        try:
            return HfPipeline[arg.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid HfPipeline choice: %s" % arg)
        
class PromptRunner(ABC):
    @abstractmethod
    async def run_prompts(self, prompts: List[Prompt], context:str, full_paragraph:FullParagraphData, system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        pass
    

    @abstractmethod
    async def call_pipeline(self, prompts: List[str]
                            , system_message:Optional[str]=None
                            # , record_embeddings:Optional[Callable[[List[str], torch.Tensor, List[str]], None]]=None
                            , **kwargs) -> List[Union[str, LlmResponseError]]:
        pass
    
    @abstractmethod
    def get_tokenizer(self)-> AutoTokenizer:
        pass

    @abstractmethod
    def finish(self):
        pass

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch



class HfTransformersQaPromptRunner(PromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, MAX_TOKEN_LEN:int, tokenizer:AutoTokenizer):
        self.hf_pipeline:transformers.Pipeline =pipeline
        self.max_token_len = MAX_TOKEN_LEN
        self.tokenizer = tokenizer
        self.question_batchSize=100

    async def run_prompts(self, prompts: List[Prompt], context:str, full_paragraph:FullParagraphData, system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        converted_prompts = [prompt.generate_prompt_with_context_QC_no_choices(context=context, full_paragraph=full_paragraph, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await list(iterable=self.call_dict_pipeline(dict_prompts=converted_prompts, **clean_hf_kwargs(kwargs)))


    async def call_dict_pipeline(self, dict_prompts: List[Dict[str,str]], **kwargs) -> List[str]:
        def processBatch(prompts):
            resps = self.hf_pipeline(prompts, max_length=self.max_token_len, num_beams=5, early_stopping=True, **clean_hf_kwargs(kwargs))
            return [resp['answer'] for resp in resps]

        return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(dict_prompts)) 
                        )) 

    async def call_pipeline(self, prompts: List[str], system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        raise RuntimeError("QA pipeline only supports Dict-prompts")



    def get_tokenizer(self):
        return self.tokenizer
    
    def finish(self):
        pass

class HfTransformersPromptRunner(PromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, MAX_TOKEN_LEN:int, tokenizer:AutoTokenizer, question_batch_size:int, max_output_tokens:int=-1, **kwargs):
        self.hf_pipeline:transformers.Pipeline =pipeline
        self.max_token_len = MAX_TOKEN_LEN
        self.max_new_tokens = max_output_tokens if max_output_tokens >0 else MAX_TOKEN_LEN
        self.tokenizer = tokenizer
        self.question_batchSize=question_batch_size


    async def run_prompts(self, prompts: List[Prompt], context:str, full_paragraph:FullParagraphData, system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        converted_prompts = [prompt.generate_prompt(context=context, full_paragraph=full_paragraph, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts, system_message=system_message, **kwargs)


    async def call_pipeline(self, prompts: List[str], system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        def processBatch(prompts):
            resps = self.hf_pipeline(prompts, max_length=self.max_token_len, num_beams=5, early_stopping=True, **clean_hf_kwargs(kwargs))
            return [resp['generated_text'] for resp in resps]

        return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(prompts)) 
                        )) 

    def get_tokenizer(self):
        return self.tokenizer
    
    def finish(self):
        pass



class HfLlamaTransformersPromptRunner(HfTransformersPromptRunner):
    def __init__(self, model, MAX_TOKEN_LEN:int, tokenizer:AutoTokenizer, max_output_tokens:int=-1):
        self.model=model
        self.tokenizer=tokenizer
        # in order to support batching in Llama
        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side ='left'

        # Create a Hugging Face pipeline
        self.hf_pipeline = pipeline('text-generation'
                                       , model=self.model
                                       , tokenizer=self.tokenizer
                                       , device=device
                                       , batch_size=BATCH_SIZE
                                       , use_fast=True
                                       , model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}}
                                       )
        super().__init__(pipeline=pipeline, MAX_TOKEN_LEN=MAX_TOKEN_LEN,tokenizer=tokenizer, max_output_tokens=max_output_tokens, question_batch_size=self.question_batchSize)

        self.terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

    async def call_pipeline(self, prompts: List[str], system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        def processBatch(prompts):
            answers=list()
            resps = self.hf_pipeline(prompts, system_message=system_message
                                    , max_new_tokens=self.max_new_tokens #, max_length=MAX_TOKEN_LEN, 
                                    , eos_token_id=self.terminators
                                    , pad_token_id = self.tokenizer.pad_token_id
                                    , do_sample=True
                                    , temperature=0.6
                                    , top_p=0.9
                                    , **clean_hf_kwargs(kwargs))

            for index, prompt in enumerate(prompts):
                # print("Llama output\n", output)
                raw_answer = resps[index][-1]['generated_text']
                answer = raw_answer[len(prompt):].strip()

                answers.append(answer)

            return zip(prompts, answers, strict=True)

        return list(itertools.chain.from_iterable(
                    (processBatch(batch) for batch in self.batchChunker(prompts)) 
                    ))    


            

# class HfTransformersAsyncPromptRunner(PromptRunner):
#     def __init__(self, pipeline:transformers.Pipeline, MAX_TOKEN_LEN:int, tokenizer:AutoTokenizer):
#         self.batcher: Optional[BatchedWorker] = None
#         self.hf_pipeline:transformers.Pipeline =pipeline
#         self.max_token_len = MAX_TOKEN_LEN
#         self.tokenizer = tokenizer

#     async def call_pipeline(self, prompts: List[str], **kwargs) -> List[str]:
#         resps = self.hf_pipeline(prompts, max_length=self.max_token_len, num_beams=5, early_stopping=True, **clean_hf_kwargs(kwargs))
#         return [resp['generated_text'] for resp in resps]

#     def get_tokenizer(self):
#         return self.tokenizer
    
#     def finish(self):
#         self.batcher.finish()

    # def batchChunker(self, iterable):
    #     iterator = iter(iterable)
    #     while True:
    #         batch = list(itertools.islice(iterator, self.question_batchSize))
    #         if not batch or len(batch)<1:
    #             break
    #         yield batch


    # async def call_qa_pipeline(self, prompts: List[Dict[str,str]]) -> List[str]:
    #     resps:List[str] = await self.prompt_runner.call_qa_pipeline(prompts)
    #     return resps

    # async def call_pipeline(self, prompts: List[str]) -> List[str]:
    #     resps:List[str] = await self.prompt_runner.call_pipeline(prompts)
    #     return resps



class OpenAIPromptRunner(PromptRunner):
    def __init__(self, fetcher:FetchGptGrade, tokenizer:AutoTokenizer, max_token_len:int=2000, max_output_tokens:int=2000):
        self.openai_fetcher = fetcher
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.max_output_tokens = max_output_tokens  # todo pass this down to VLLM

    async def run_prompts(self, prompts: List[Prompt], context:str, full_paragraph:FullParagraphData, system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        anyprompt=prompts[0]
        # anyprompt.configure_json_gpt_fetcher(self.openai_fetcher)
        self.openai_fetcher.set_json_instruction(json_instruction=anyprompt.gpt_json_prompt()[0], field_name=anyprompt.gpt_json_prompt()[1])

        converted_prompts = [prompt.generate_prompt(context=context, full_paragraph=full_paragraph, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts, system_message=system_message, **kwargs)


    async def call_pipeline(self
                            , prompts: List[str]
                            , system_message:Optional[str]=None
                            , **kwargs) -> List[Union[str, LlmResponseError]]:
        responses:list[Union[str, LlmResponseError]] =   [await self.openai_fetcher.generate_request(prompt, openai_interface.global_rate_limiter, system_message=system_message, **kwargs) for prompt in prompts]
        for p,resp in zip(prompts, responses):
            # if resp is None:  # We don't return None's anymore
            #     raise RuntimeError(f"Obtained None, but should have recevied an LlmResponseError. Prompt {p}")
                # sys.stderr.write(f"Could not obtain OpenAI response for prompt {p}")
            if isinstance(resp, LlmResponseError):
                sys.stderr.write(f"OpenAIPromptRunner.call_pipeline: Stumbled upon LlmResponse error {resp}")

        # return list(filter(None, responses_might_be_none))
        return responses


    def get_tokenizer(self):
        return self.tokenizer

    def finish(self):
        pass

    
class VllmPromptRunner(PromptRunner):
    def __init__(self, fetcher:FetchGptGrade, tokenizer:AutoTokenizer, max_token_len:int, max_output_tokens:int):
        print(fetcher.client.base_url)
        self.vllm_fetcher = fetcher
        self.rate_limiter = OpenAIRateLimiter(max_requests_per_minute= 1000000,max_tokens_per_minute=1000000 )
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.max_output_tokens = max_output_tokens  # todo pass this down to VLLM

    async def run_prompts(self, prompts: List[Prompt], context:str, full_paragraph:FullParagraphData, system_message:Optional[str]=None, **kwargs) -> List[Union[str, LlmResponseError]]:
        anyprompt=prompts[0]
        self.vllm_fetcher.set_json_instruction(json_instruction=anyprompt.gpt_json_prompt()[0], field_name=anyprompt.gpt_json_prompt()[1])

        converted_prompts = [prompt.generate_prompt(context=context, full_paragraph=full_paragraph, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts, system_message=system_message, **kwargs)



    async def call_pipeline(self
                            , prompts: List[str]
                            , system_message:Optional[str]=None
                            , **kwargs) -> List[Union[str, LlmResponseError]]:
        responses =   [await self.vllm_fetcher.generate_request(prompt, self.rate_limiter, system_message=system_message, **kwargs) for prompt in prompts]

        for p,resp in zip(prompts, responses):
            if resp is None:
                raise RuntimeError(f"Obtained None, but should have recevied an LlmResponseError. Prompt {p}")
        #         sys.stderr.write(f"Could not obtain VLLM response for prompt {p}, reason {resp}. Rater limiter: {self.rate_limiter}")
            if isinstance(resp, LlmResponseError):
                sys.stderr.write(f"OpenAIPromptRunner.call_pipeline: Stumbled upon LlmResponse error {resp}")

        return responses

    def get_tokenizer(self):
        return self.tokenizer
        
    def finish(self):
        pass


class LlmPipeline():
    def __init__(self, model_name:str, max_token_len:int=512, max_output_tokens:int=512, question_batchSize:int =100):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """

        self.modelName = model_name
        self.max_token_len = max_token_len
        self.max_output_tokens = max_output_tokens
        self.question_batchSize = question_batchSize
        self.prompt_runner:PromptRunner

    def exp_modelName(self)->str:
        return self.modelName


    def finish(self):
        self.prompt_runner.finish()



    async def grade_paragraph(self, prompts:List[Prompt],  paragraph_txt:str, full_paragraph:FullParagraphData, system_message:Optional[str]=None, **kwargs)->List[Tuple[Prompt, Union[str, LlmResponseError]]]:
        """Run question answering over batches of questions, and tuples it up with the answers"""
        answers:List[Union[str, LlmResponseError]] = await self.prompt_runner.run_prompts(prompts=prompts, context=paragraph_txt, full_paragraph=full_paragraph, system_message=system_message, **kwargs)

        if len(answers) != len(prompts):
            raise RuntimeError("Missing prompt response\mPrompts: {prompts}\n Answers: {answers}")
        
        return list(zip(prompts, answers, strict=True))
        # todo Catch errors

class VllmPipeline(LlmPipeline):
    """Pipeline for vLLM"""

    def __init__(self, model_name:str,  max_token_len:int, max_output_tokens:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_output_tokens)
        # Start VLLM with:  HF_TOKEN="<token>" tmp/bin/vllm serve meta-llama/Llama-3.3-70B-Instruct  --max-model-len 500 --device=cuda --tensor-parallel-size 2

        self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")  # todo use tiktoken
        vllm_fetcher = FetchGptGrade(gpt_model=self.modelName, max_tokens=self.max_output_tokens, client=create_vllm_client(base_url=os.getenv('VLLM_URL')), use_chat_protocol=True)
        self.prompt_runner = VllmPromptRunner(fetcher=vllm_fetcher, tokenizer = self.tokenizer, max_token_len=self.max_token_len, max_output_tokens=self.max_output_tokens)

class OpenAIPipeline(LlmPipeline):
    """Pipeline for OpenAI"""

    def __init__(self, model_name:str, max_token_len:int, max_output_tokens:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_output_tokens)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2") # todo use tiktoken
        open_ai_fetcher = FetchGptGrade(gpt_model=self.modelName, max_tokens=self.max_output_tokens, client=create_gpt_client(), use_chat_protocol=True)
        self.prompt_runner = OpenAIPromptRunner(fetcher=open_ai_fetcher, tokenizer = self.tokenizer, max_token_len=self.max_token_len, max_output_tokens=self.max_output_tokens)



class Text2TextPipeline(LlmPipeline):
    """Pipeline for text2text"""

    def __init__(self, model_name:str, max_token_len:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_token_len)

        self.model = T5ForConditionalGeneration.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        hf_pipeline = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)
        self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline, MAX_TOKEN_LEN=self.max_token_len, tokenizer=self.tokenizer, max_output_tokens=self.max_output_tokens, question_batch_size=self.question_batchSize)
        print(f"Text2Text model config: { self.model.config}")



class EmbeddingText2TextPipeline(LlmPipeline):
    """Pipeline that records embeddings of text2text models"""

    def __init__(self, model_name:str, max_token_len:int,  max_output_tokens:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_output_tokens)

        self.model:T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(self.modelName)
        self.model = self.model.to(device)
        self.model_set = {}
        self.model_set[device] = self.model
        if device.type != "cpu":
            self.model_set[torch.device("cpu")] = T5ForConditionalGeneration.from_pretrained(self.modelName)

        self.tokenizer:T5TokenizerFast = T5TokenizerFast.from_pretrained(self.modelName)

        # hf_pipeline = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)
        # self.prompt_runner = EmbeddingRecordingHfTransformersPromptRunner(pipeline=hf_pipeline, MAX_TOKEN_LEN=self.max_token_len, tokenizer=self.tokenizer, max_output_tokens=self.max_output_tokens, question_batch_size=self.question_batchSize)
        print(f"Embedding Text2Text model config: { self.model.config}")



    @staticmethod
    def export_encoder_hidden_state(prefix_token_len:int, batch_sz: int, out: transformers.generation.utils.GenerateBeamEncoderDecoderOutput, layer_no:int=-1) -> torch.Tensor:
        '''
        Given out.encoder_hidden_states and the number of prefix tokens
        produce a tensor of encoder hidden states, omitting the tokens of the prefix/system message
        Out: [batch_sz][token_len][d_model]'''

        states = out.encoder_hidden_states[-1].cpu()
        return states[:,prefix_token_len:,:]
        
    @staticmethod
    def export_decoder_hidden_state(batch_sz: int, out: transformers.generation.utils.GenerateBeamEncoderDecoderOutput, layer_no:int=-1) -> torch.Tensor:
        '''
        Given out.decoder_hidden_states and out.beam_indices, 
        produce a tensor of decoder hidden states that represent the best beam.
        Precondition: num_return_sequences = 1
        Out: [batch_sz][token_len][d_model]'''

        # Relevant Part of the Object documentation
        # sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
        #     The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
        #     if all batches finished early due to the `eos_token_id`.
        # 
        #  beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True`):
        #     Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
        #    `(batch_size*num_return_sequences, sequence_length)`.
        #
        # encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        #     Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        #     shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        #
        # decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
        #     Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
        #     `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.

        batch_sz, _ = out.sequences.shape
        
        beam_indices = out.beam_indices.cpu()
        _, seq_len1 = beam_indices.shape
        seq_len2 = len(out.decoder_hidden_states)
        seq_len = min(seq_len1,seq_len2)

        # Note: it is possible that beam_indices indicate a shorter seq_len than the decoder_hidden states.

        if seq_len1 > seq_len2:
            print(f"====== Record Embeddings \n seq_len's don't match.\n _, seq_len1 = beam_indices.shape= {beam_indices.shape}\n seq_len2 = len(out.decoder_hidden_states)={len(out.decoder_hidden_states)}")
            print("beam_indices", beam_indices)
            print("size of out.decoder_hidden_states:",len(out.decoder_hidden_states), len(out.decoder_hidden_states[0]), out.decoder_hidden_states[0][0].shape)
                  

        d_model = out.decoder_hidden_states[0][0].shape[2]
        best_hidden_states = torch.empty(size=(batch_sz, seq_len, d_model))
        done = torch.zeros((batch_sz,), dtype=torch.bool)

        for tok_idx in range(seq_len):
            # beam_beam: shape=(batch_sz,)
            best_beam = beam_indices[:,tok_idx]
            done |= best_beam == -1
            #print(best_beam)

            # hidden_states: shape=(batch_sz*num_beams, seq_len, d_model)
            hidden_states = out.decoder_hidden_states[tok_idx][layer_no].cpu()
            best_hidden_states[:,tok_idx,:] = hidden_states[best_beam, 0, :]
            best_hidden_states[done,tok_idx,:] = 0 # pad_token

            # print(best_hidden_states[:,tok_idx,:])

        return best_hidden_states


    @staticmethod
    def cat_pad_tensors(tensors: list[torch.Tensor], dim:int)->torch.Tensor:
        max_dim1 = max(tensor.size(1) for tensor in tensors)
        padded_tensors = [
           F.pad(tensor, (0, 0, 0, max_dim1 - tensor.size(1))) for tensor in tensors
        ]
        result = torch.cat(padded_tensors, dim=dim)
        return result

    def embed_and_record(self, prompts:List[str]
                        , record_embeddings:Callable[[List[str], torch.Tensor, List[str]], None]
                        , prompt_prefix_len:int
                        ,  max_len, num_beams
                        , early_stopping, **kwargs)->List[str]:

        try:
            # Attempt 1: run full batch on GPU
            print(f"Attempt 1: Run full batch on {device}")

            prompts, embeddings, answers= self.embed_model(device=device, prompts = prompts, prompt_prefix_len=prompt_prefix_len, max_len=max_len, num_beams=num_beams, early_stopping=early_stopping, **kwargs)
            record_embeddings(prompts, embeddings, answers)
            return answers
        except torch.OutOfMemoryError:
            # Attempt 2: smaller batches
            
            batch_step = 1
            print(f"torch.OutOfMemory error: Attempt 2: retrying batches of {batch_step}")

            pivots=list(range(0,len(prompts),2))
            pivots.append(len(prompts))
            prompt_segments = [prompts[pivots[i]:pivots[i+1]] for i in range(len(pivots) - 1)] 
            answers_list:List[List[str]]

            embeddings_list = list()
            answers_list = list()
            for prompt_set in prompt_segments:
                try:
                    _, embeddings_entry, answers_entry = self.embed_model(device=device, prompts = prompt_set, prompt_prefix_len=prompt_prefix_len, max_len=max_len, num_beams=num_beams, early_stopping=early_stopping, **kwargs) 
                    embeddings_list.append(embeddings_entry)
                    answers_list.append(answers_entry)
                    # _,embeddings_list, answers_list = map(list, zip(*[]))


                except torch.OutOfMemoryError as ex:
                    # Attempt 3: run on cpu

                    # cpu_device = torch.device("cpu")
                    # print(f"torch.OutOfMemory error: Attempt 3: retrying on {cpu_device}")

                    # _, embeddings_entry, answers_entry = self.embed_model(device=cpu_device, prompts = prompt_set, prompt_prefix_len=prompt_prefix_len, max_len=max_len, num_beams=num_beams, early_stopping=early_stopping, **kwargs) 
                    # embeddings_list.append(embeddings_entry)
                    # answers_list.append(answers_entry)
                    raise ex

            answers:List[str] = list(itertools.chain(*answers_list))
            embeddings = EmbeddingText2TextPipeline.cat_pad_tensors(tensors=embeddings_list, dim=0)


            # store embedding in database
            record_embeddings(prompts, embeddings, answers)
            return answers


    def embed_model(self, device:torch.device, prompts:List[str]
                    # , record_embeddings:Callable[[List[str], torch.Tensor, List[str]], None]
                    , prompt_prefix_len:int
                    ,  max_len, num_beams
                    , early_stopping, **kwargs)->Tuple[List[str], torch.Tensor, List[str]]:
        # model=transformers.AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
        # tokenizer=transformers.AutoTokenizer.from_pretrained('google/flan-t5-large')

        # num_beams = 1
        with torch.no_grad():
            tokens = self.tokenizer(prompts, return_tensors='pt', max_length=max_len, padding=True, truncation=True)
            tokens = tokens.to(device)
            try:
                #out = model(**t, decoder_input_ids=t['input_ids'])
                out = self.model_set[device].generate(
                    **tokens,
                    num_beams=num_beams,
                    max_length=self.max_output_tokens,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_scores=True,
                    early_stopping=early_stopping
                )

                answers = self.tokenizer.batch_decode(out.sequences.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # batch_size = tokens['input_ids'].shape[0]
                # final_encoder_hidden_states = out.encoder_hidden_states[-1].cpu()

                final_encoder_hidden_states = EmbeddingText2TextPipeline.export_encoder_hidden_state(out=out, prefix_token_len=prompt_prefix_len, batch_sz=len(prompts))
                final_decoder_hidden_states = EmbeddingText2TextPipeline.export_decoder_hidden_state(out=out, batch_sz=len(prompts))
                embeddings = torch.cat(tensors=(final_encoder_hidden_states, final_decoder_hidden_states), dim=1)


                # hidden_answers = self.tokenizer.batch_decode(torch.argmax(self.model.lm_head(batch_hidden_state_seq), dim=2))
                # print(hidden_answers)

                return prompts, embeddings, answers
            except torch.OutOfMemoryError as ex:
                del tokens
                del out
                torch.cuda.empty_cache()
                print(f"Allocated: {torch.cuda.memory_allocated()}")
                print(f"Reserved: {torch.cuda.memory_reserved()}")
                raise ex



    async def run_prompts(self, prompts: List[Prompt], context:str, full_paragraph:FullParagraphData
                          , record_embeddings:Optional[Callable[[List[str], torch.Tensor, List[str]], None]]=None
                          , system_message:Optional[str]=None
                          , **kwargs) -> List[Union[str, LlmResponseError]]:
        converted_prompts = [prompt.generate_prompt(context=context, full_paragraph=full_paragraph, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        prompt_prefix_len = prompts[0].prompt_prefix_len(model_tokenizer=self.tokenizer, max_token_len=self.max_token_len)
        return await self.call_pipeline(prompts=converted_prompts, system_message=system_message, record_embeddings=record_embeddings, prompt_prefix_len=prompt_prefix_len, **kwargs)


    async def call_pipeline(self, prompts: List[str]
                            , prompt_prefix_len:int
                            , system_message:Optional[str]=None
                            , record_embeddings:Optional[Callable[[List[str], torch.Tensor, List[str]], None]]=None
                            , **kwargs) -> List[Union[str, LlmResponseError]]:
        if record_embeddings is None:
            raise RuntimeError(f"To record embeddings, {self.__class__}.call_pipeline() must be called with  `record_embeddings:Callable[[List[str], torch.Tensor, List[str]], None]`")
        else:
            try:
                resps = self.embed_and_record(prompts, record_embeddings=record_embeddings, prompt_prefix_len=prompt_prefix_len, max_len=self.max_token_len, num_beams=5, early_stopping=True, **kwargs)
                return resps #[resp['generated_text'] for resp in resps]
            except torch.OutOfMemoryError as ex:
                return [LlmResponseError(failure_reason="torch.OutOfMemoryError", caught_exception=ex.message(), prompt=prompt, response="") for prompt in prompts]


    def exp_modelName(self)->str:
        return self.modelName


    def finish(self):
        pass
        # self.prompt_runner.finish()


    async def grade_paragraph(self, prompts:List[Prompt],  paragraph_txt:str, full_paragraph:FullParagraphData
                              , system_message:Optional[str]=None
                              , record_embeddings:Optional[Callable[[List[str], torch.Tensor, List[str]], None]]=None
                              ,  **kwargs
                              )->List[Tuple[Prompt, Union[str, LlmResponseError]]]:
        """Run question answering over batches of questions, and tuples it up with the answers"""

        answers:List[Union[str, LlmResponseError]] = await self.run_prompts(prompts=prompts
                                                            , context=paragraph_txt
                                                            , full_paragraph=full_paragraph
                                                            , system_message=system_message
                                                            , record_embeddings = record_embeddings
                                                            , **kwargs)

        if len(answers) != len(prompts):
            raise RuntimeError("Missing prompt response\mPrompts: {prompts}\n Answers: {answers}")
        
        return list(zip(prompts, answers, strict=True))
        # todo Catch errors


class TextGenerationPipeline(LlmPipeline):
    """Pipeline for text-generation"""
    
    def __init__(self, model_name:str, max_token_len:int, max_output_tokens:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_output_tokens) 
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        hf_pipeline = pipeline('text-generation'
                                , model=self.model
                                , tokenizer=self.tokenizer
                                , device=device
                                , batch_size=BATCH_SIZE
                                , use_fast=True
                                )
        self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline , MAX_TOKEN_LEN=self.max_token_len, tokenizer=self.tokenizer, max_output_tokens=self.max_output_tokens, question_batch_size=self.question_batchSize)
        print(f"TextGeneration model config: { self.model.config}")




class LlamaTextGenerationPipeline(LlmPipeline):
    """Pipeline for llama text-generation"""

    def __init__(self, model_name:str, max_token_len:int, max_output_tokens:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_output_tokens) 
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        self.prompt_runner = HfLlamaTransformersPromptRunner(model=self.model, MAX_TOKEN_LEN=self.max_token_len, tokenizer = self.tokenizer, max_output_tokens=self.max_output_tokens)
        print(f"Llama model config: { self.model.config}")



class QaPipeline(LlmPipeline):
    """QA Pipeline for text2text-based question answering"""

    def __init__(self, model_name:str, max_token_len:int, max_output_tokens:int):
        super().__init__(model_name=model_name, max_token_len=max_token_len, max_output_tokens=max_output_tokens)

        # Initialize the tokenizer and model
        # self.modelName = 'sjrhuschlee/flan-t5-large-squad2'
        self.modelName = model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"QaPipeline model config: { self.model.config}")

        # Create a Hugging Face pipeline
        qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)
        self.prompt_runner = HfTransformersQaPromptRunner(pipeline=qa_pipeline, MAX_TOKEN_LEN=self.max_token_len, tokenizer=self.tokenizer)

        print(f"Qa model config: { self.model.config}")



def mainQA():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data(self_rater_tolerant=False)[0:2]
    
    
    qa = QaPipeline('sjrhuschlee/flan-t5-large-squad2')

    # promptGenerator=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(context='', model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = qa.grade_paragraph(questions, "", FullParagraphData.empty())
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
        answerTuples = qa.grade_paragraph(questions, "", ParagraphData.empty())
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")




if __name__ == "__main__":
    mainT2T()

