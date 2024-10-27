# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from swift.utils import get_logger, is_merge_kit_available

logger = get_logger()


@dataclass
class MergeArguments:
    merge_lora: bool = False
    merge_device_map: Optional[str] = None
    use_merge_kit: bool = False
    instruct_model_id_or_path: Optional[str] = None
    instruct_model_revision: Optional[str] = None

    def __post_init__(self):
        if self.use_merge_kit:
            assert is_merge_kit_available(), ('please install mergekit by pip install '
                                              'git+https://github.com/arcee-ai/mergekit.git')
            logger.info('Important: You are using mergekit, please remember '
                        'the LoRA should be trained against the base model,'
                        'and pass its instruct model by --instruct_model xxx when merging')
            assert self.instruct_model_id_or_path, 'Please pass in the instruct model'

            self.merge_yaml = ('models:'
                               '  - model: {merged_model}'
                               '    parameters:'
                               '      weight: 1'
                               '  }'
                               '  - model: {instruct_model}'
                               '    parameters:'
                               '      weight: 1'
                               '  }'
                               'merge_method: ties'
                               'base_model: {base_model}'
                               'parameters:'
                               '  normalize: true'
                               '  int8_mask: true'
                               'dtype: bfloat16')