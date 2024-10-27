# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass


@dataclass
class WebUIArguments:
    host: str = '127.0.0.1'
    port: int = 7860
    share: bool = False
    lang: str = 'zh'