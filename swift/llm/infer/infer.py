# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from swift.llm import (
    HfDataset, InferArguments, Messages, Pipeline, Template, get_template, load_dataset, merge_lora
)
from swift.tuners import Swift
from swift.utils import append_to_jsonl, get_logger, get_main, read_multi_line, seed_everything
from .infer_engine import InferEngine, InferRequest, RequestConfig

logger = get_logger()



@dataclass
class InferCliState:
    # None: use default-system. '': not use system.
    system: Optional[str] = None
    messages: Messages = field(default_factory=list)  # not including system

    images: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    multiline_mode: bool = False
    input_system: bool = False
    result_path: Optional[str] = None

    def clear(self):
        self.messages = []
        self.images = []
        self.audios = []
        self.videos = []

    def copy(self):
        return InferCliState(self.system, deepcopy(self.messages), self.images.copy(), self.audios.copy(),
                             self.videos.copy(), self.multiline_mode, self.input_system)

    def to_infer_request(self) -> InferRequest:
        infer_state = self.copy()
        if infer_state.system is not None:
            infer_state.messages.insert(0, {'role': 'system', 'content': infer_state.system})
        return InferRequest(infer_state.messages, infer_state.images, infer_state.audios, infer_state.videos)

    def add_query(self, query: str) -> None:
        self.messages.append({'role': 'user', 'content': query})

    def add_response(self, response: str) -> None:
        self.messages.append({'role': 'assistant', 'content': response})

    def to_dict(self):
        infer_state = self.copy()
        return {
            'messages': infer_state.messages,
            'images': infer_state.images,
            'audios': infer_state.audios,
            'videos': infer_state.videos
        }


class InferPipeline(Pipeline):
    args_class = InferArguments

    def __init__(self, args: Union[List[str], InferArguments, None] = None) -> None:
        self.args: InferArguments = self.parse_args(args)
        if args.merge_lora:
            merge_lora(args, device_map=args.merge_device_map)
        self.infer_engine = self.get_infer_engine()
        self.template = self.get_template(self.infer_engine.tokenizer)

    def get_infer_engine(self) -> InferEngine:
        args = self.args
        kwargs = {
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
            'use_hf': args.use_hf,
        }
        if args.infer_backend == 'pt':
            from .infer_engine import PtEngine
            infer_engine_cls = PtEngine
            kwargs.update({
                'attn_impl': args.attn_impl,
                'device_map': args.device_map_config,
                'quantization_config': args.quantization_config,
            })
        elif args.infer_backend == 'vllm':
            from .infer_engine import VllmEngine
            infer_engine_cls = VllmEngine
            kwargs.update({
                'gpu_memory_utilization': args.gpu_memory_utilization,
                'tensor_parallel_size': args.tensor_parallel_size,
                'max_num_seqs': args.max_num_seqs,
                'max_model_len': args.max_model_len,
                'disable_custom_all_reduce': args.disable_custom_all_reduce,
                'enforce_eager': args.enforce_eager,
                'limit_mm_per_prompt': args.limit_mm_per_prompt,
                'enable_lora': args.enable_lora,
                'max_loras': args.max_loras,
                'max_lora_rank': args.max_lora_rank
            })
        else:
            from .infer_engine import LmdeployEngine
            infer_engine_cls = LmdeployEngine
            kwargs.update({
                'tp': args.tp,
                'cache_max_entry_count': args.cache_max_entry_count,
                'quant_policy': args.quant_policy,
                'vision_batch_size': args.vision_batch_size
            })

        return infer_engine_cls(**kwargs)


    def run(self) -> None:
        args = self.args


    @staticmethod
    def _input_mm_data(infer_state: InferCliState) -> None:

        def _input_mm_file(mm_type: Literal['image', 'video', 'audio']) -> str:
            a_an = 'an' if mm_type[0] in {'i', 'a'} else 'a'
            return input(f'Input {a_an} {mm_type} path or URL <<< ')

        mm_types = ['image', 'video', 'audio']
        query = infer_state.messages[-1]['content']
        mm_tags = re.findall('|'.join(f'<({mm_type})>' for mm_type in mm_types), query)
        # mm_tag -> mm_type/mm_key
        mm_mapping = {f'<{mm_type}>': (mm_type, f'{mm_type}s') for mm_type in mm_types}
        for mm_tag in mm_tags:
            mm_type, mm_key = mm_mapping[mm_tag]
            mm_val = getattr(infer_state, mm_key)
            mm_val.append(_input_mm_file(mm_type))

    def _prepare_save_result(self, args: InferArguments) -> str:
        if args.result_dir is not None:
            result_dir = args.result_dir
        else:
            if args.ckpt_dir is not None:
                result_dir = args.ckpt_dir
            else:
                result_dir = self.model_dir
            result_dir = os.path.join(result_dir, 'infer_result')
        result_dir = os.path.abspath(os.path.expanduser(result_dir))
        os.makedirs(result_dir, exist_ok=True)
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        return os.path.join(result_dir, f'{time}.jsonl')

    @staticmethod
    def _input_multiline(prompt: str) -> str:
        query = ''
        stop_words = '#\n'
        while True:
            text = f'{input(prompt)}\n'
            prompt = ''
            if text.endswith(stop_words):
                query += text[:len(stop_words)]
                break
            query += text
        return query

    @staticmethod
    def _input_text(multiline_mode: bool, input_system: bool) -> str:
        if multiline_mode:
            addi_prompt = 'MS' if input_system else 'M'
            text = InferEngine._input_multiline(f'<<<[{addi_prompt}] ')
        else:
            addi_prompt = 'S' if input_system else ''
            text = input(f'<<<[{addi_prompt}] ')
        return text

    @staticmethod
    def _check_query(infer_state: InferCliState, query: str) -> Optional[str]:
        query_std = query.strip().lower()
        if infer_state.input_system:
            if query == 'default-system':
                infer_state.system = None
            else:
                infer_state.system = query
            infer_state.input_system = False
            query_std = 'clear'
        if query_std == 'clear':
            infer_state.clear()
            return
        if query_std == '':
            return
        if query_std == 'reset-system':
            infer_state.input_system = True
            return
        if query_std == 'multi-line':
            infer_state.multiline_mode = True
            logger.info('End multi-line input with `#`.')
            logger.info('Input `single-line` to switch to single-line input mode.')
            return
        if query_std == 'single-line':
            infer_state.multiline_mode = False
            return
        return query

    @staticmethod
    def _prepare_request_config(args: InferArguments) -> RequestConfig:
        temperature = args.temperature
        if not args.do_sample:
            temperature = 0
        return RequestConfig(
            max_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            stop=args.stop_words,
            stream=args.stream,
            repetition_penalty=args.repetition_penalty)

    def get_template(self, tokenizer) -> Template:
        args = self.args
        template = get_template(
            args.template_type,
            tokenizer,
            args.system,
            args.max_length,
            truncation_strategy=args.truncation_strategy,
            loss_scale=args.loss_scale_config,
            max_pixels=args.max_pixels,
            sequence_parallel_size=args.sequence_parallel_size,
            tools_prompt=args.tools_prompt)
        logger.info(f'default_system: {template.default_system}')
        return template

    def infer_cli(self, args: InferArguments) -> List[Dict[str, Any]]:
        template = self.prepare_template(args)
        result_path = None
        if args.save_result:
            result_path = self._prepare_save_result(args)
        request_config = self._prepare_request_config(args)

        result = []
        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        if template.support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')

        infer_state = InferCliState()
        while True:
            if not template.support_multi_round:
                infer_state.clear()
            query = self._input_text(infer_state.multiline_mode, infer_state.input_system)
            if query.strip().lower() in {'exit', 'quit'}:
                break
            query = self._check_query(infer_state, query)
            if query is None:
                continue
            infer_state.add_query(query)
            self._input_mm_data(infer_state)
            infer_request = infer_state.to_infer_request()
            res_or_gen = self.infer(template, [infer_request], request_config, use_tqdm=False)
            if request_config.stream:
                response = ''
                for res in res_or_gen:
                    delta = res[0].choices[0].delta
                    print(delta, end='', flush=True)
                    response += delta
                print()
            else:
                response = res_or_gen[0].choices[0].message.content
            infer_state.add_response(response)

            data = infer_state.to_dict()
            result.append(data)
            if result_path is not None:
                append_to_jsonl(result_path, data, strict=False)

        return result

    def prepare_dataset(self, args: InferArguments) -> HfDataset:
        load_dataset(args.val_dataset, args.split_dataset_ratio)
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(args.val_dataset, 1.0, **dataset_kwargs)
        else:
            _, val_dataset = load_dataset(args.dataset, args.dataset_test_ratio, **dataset_kwargs)

    def infer_dataset(self, args: InferArguments):
        template = self.prepare_template(args)
        result_path = None
        if args.save_result:
            result_path = self._prepare_save_result(args)
        request_config = self._prepare_request_config(args)



