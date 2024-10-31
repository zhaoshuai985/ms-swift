# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from swift.llm import InferArguments


@dataclass
class WebUIArguments(InferArguments):
    """
    WebUIArguments is a dataclass that inherits from InferArguments.

    Args:
        host (str): The hostname or IP address to bind the web UI server to. Default is '0.0.0.0'.
        port (int): The port number to bind the web UI server to. Default is 7860.
        share (bool): A flag indicating whether to share the web UI publicly. Default is False.
        lang (str): The language setting for the web UI. Default is 'zh'.
    """
    host: str = '0.0.0.0'
    port: int = 7860
    share: bool = False
    lang: str = 'zh'
