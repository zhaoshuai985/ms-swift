import hashlib
import os
import shutil
from typing import Any, Dict, Literal, Optional, Union, Callable

import numpy as np
from modelscope.hub.utils.utils import get_cache_dir

from swift.utils import get_logger

logger = get_logger()


class MediaProcessor:

    grounding_prompts = {
        'ref_grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')]
        },
        'grounding_caption': {
            'en': [
                ('<bbox>', '<ref-object>'),
                ('The object at position <bbox>', '<ref-object>'),
                ('This <bbox> is', '<ref-object>'),
                ('What is the object at <bbox>', '<ref-object>'),
                ('Describe <bbox>', '<ref-object>'),
                ('<bbox> is', '<ref-object>'),
                ('The bounding box coordinate <bbox> contains', '<ref-object>'),
            ],
            'zh': [
                ('<bbox>', '<ref-object>'),
                ('<bbox>是什么', '<ref-object>'),
                ('<bbox>的位置包含', '<ref-object>'),
                ('描述<bbox>', '<ref-object>'),
                ('<bbox>中是', '<ref-object>'),
                ('坐标<bbox>描述了什么', '<ref-object>'),
                ('描述<bbox>中的事物', '<ref-object>'),
            ]
        },
    }

    standard_tags = {
        'image': '<image>',
        'audio': '<audio>',
        'video': '<video>',
    }

    media_keys = {
        'audio': 'audios',
        'image': 'images',
        'video': 'videos',
    }

    def __init__(self,
                 media_type: Optional[Literal['image', 'audio', 'video']],
                 media_tag=None,
                 task_type: Literal['caption_with_grounding', 'ref_grounding', 'grounding_caption', 'ocr',
                                    'vqa'] = 'vqa'):
        self.media_type = media_type
        self.task_type = task_type
        self.media_tag = media_tag or '<unused_tag>'

    def construct_grounding_prompt(self):
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
        prompts = self.grounding_prompts[self.task_type][lang]
        query, response = prompts[np.random.choice(range(len(prompts)))]
        return query, response

    def replace_standard_tag(self, query, response, history, medias):
        media_cnt = len(medias) if isinstance(medias, (tuple, list)) else 1 if medias else 0
        # like <image>, etc
        standard_tag = self.standard_tags[self.media_type]

        all_queries = ''.join([h[0] for h in history]) + query
        if self.media_tag in all_queries:
            assert all_queries.count(self.media_tag) == media_cnt
            for h in history:
                h[0] = h[0].replace(self.media_tag, standard_tag)

            query = query.replace(self.media_tag, standard_tag)
        return query, response, history

    def preprocess_media_prompts(self, d: Dict[str, Any], medias: Union[tuple, list]) -> None:
        """Format the query/response/history with medias:
        1. Construct the ref_grounding/grounding_caption

        Args:
            d: A dict contains history/query/response, this dict will be inplace changed
            medias: A list of medias(one round, multiple medias),
                    a single media(one round, one media), or a tuple of media list(multiple rounds)
        """
        if not self.media_type:
            return

        history = d.get('history') or []
        query = d.get('query')
        response = d.get('response')

        if self.task_type in ('ref_grounding', 'grounding_caption'):
            query, response = self.construct_grounding_prompt()

        query, response, history = self.replace_standard_tag(query, response, history, medias)

        if 'history' in d:
            d['history'] = history
        d['query'] = query
        d['response'] = response


class MediaResource:

    cache_dir = os.path.join(get_cache_dir(), 'media_resources')
    lock_dir = os.path.join(get_cache_dir(), 'lockers')

    media_type_urls = {
        'llava', 'coco', 'sam', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2', 'share_textvqa', 'web-celebrity',
        'web-landmark', 'wikiart'
    }

    URL_PREFIX = 'https://www.modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision=master&FilePath='

    @staticmethod
    def get_url(media_type):
        is_ocr_vqa = (media_type == 'ocr_vqa')
        extension = 'tar' if is_ocr_vqa else 'zip'
        return f'{MediaResource.URL_PREFIX}{media_type}.{extension}'

    @staticmethod
    def download(media_type_or_url: str, local_alias: Optional[str] = None):
        """Download and extract a resource from a http link.

        Args:
            media_type_or_url: `str`, Either belongs to the `media_type_urls` listed in the class field, or a
                remote url to download and extract. Be aware that, this media type or url
                needs to contain a zip or tar file.
            local_alias: `Options[str]`, The local alias name for the `media_type_or_url`. If the first arg is a
            media_type listed in this class, local_alias can leave None. else please pass in a name for the url.
            The local dir contains the extracted files will be: {cache_dir}/{local_alias}

        Returns:
            The local dir contains the extracted files.
        """
        from swift.utils import safe_ddp_context
        from datasets.utils.filelock import FileLock
        file_path = hashlib.md5(media_type_or_url.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(MediaResource.lock_dir, file_path)
        os.makedirs(MediaResource.lock_dir, exist_ok=True)
        with safe_ddp_context():
            with FileLock(file_path):
                return MediaResource._safe_download(media_type=media_type_or_url, media_name=local_alias)

    @staticmethod
    def _safe_download(media_type, media_name=None):
        media_name = media_name or media_type
        if media_type in MediaResource.media_type_urls:
            media_type = MediaResource.get_url(media_type)

        from datasets.download.download_manager import DownloadManager, DownloadConfig
        final_folder = os.path.join(MediaResource.cache_dir, media_name)
        if os.path.exists(final_folder):
            return final_folder

        logger.info('# #################Resource downloading#################')
        logger.info('Downloading necessary resources...')
        logger.info(f'Resource package: {media_type}')
        logger.info(f'Extracting to local dir: {final_folder}')
        logger.info('If the downloading fails or lasts a long time, '
                    'you can manually download the resources and extracting to the local dir.')
        logger.info('Now begin.')
        local_dirs = DownloadManager(download_config=DownloadConfig(
            cache_dir=MediaResource.cache_dir)).download_and_extract(media_type)
        shutil.move(str(local_dirs), final_folder)
        logger.info('# #################Resource downloading finished#################')
        return final_folder

    @staticmethod
    def safe_save(image, file_name, folder, format='JPEG'):
        folder = os.path.join(MediaResource.cache_dir, folder)
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, file_name)
        if os.path.exists(file):
            return file
        image.save(file, format=format)
        return file


class MediaMixin:

    def __init__(self,
                 media_key: Union[str, Callable] = 'image',
                 media_tag: str = '<image>',
                 media_type: Literal['image', 'audio', 'video'] = None):
        self.media_key = media_key
        self.media_tag = media_tag
        self.media_type = media_type
        self.media_processor = MediaProcessor(media_type, media_tag)

    @property
    def media_name(self):
        if not self.media_type:
            return None
        return self.media_processor.media_keys[self.media_type]

    def parse_media_from_row(self, d: Dict[str, Any]):
        media_key = self.media_key
        if isinstance(media_key, str):
            if media_key in d:
                medias = d[media_key]
            else:
                medias = None
        elif media_key:  # function
            medias = media_key(d)
        else:
            medias = None
        return medias

    @property
    def empty_row(self):
        empty_row = {
            'query': None,
            'response': None,
            'tools': None,
            'system': None,
            'history': None,
        }
        if self.media_type and not isinstance(self.media_key, str):
            empty_row[self.media_name] = None
        return empty_row