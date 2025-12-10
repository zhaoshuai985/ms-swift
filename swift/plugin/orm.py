import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import json

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        # pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        # pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>\s*<plane>.*?</plane>\s*<modality>.*?</modality>\s*<title>.*?</title>\s*<caption>.*?</caption>(?![\s\S])'
        pattern = r'^<think>.*?</think>\s*<plane>.*?</plane>\s*<modality>.*?</modality>\s*<title>.*?</title>\s*<caption>.*?</caption>\s*<answer>.*?</answer>(?![\s\S])'
        # pattern = r'^<plane>.*?</plane>\s*<modality>.*?</modality>\s*<title>.*?</title>\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class SmartAccuracy(ORM):
    """
    Simple and reliable accuracy reward function for chart/visual QA tasks.
    Extracts answers from <answer></answer> tags and performs direct string comparison.
    Perfect for numerical answers from chart analysis tasks.
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        
        for content, sol in zip(completions, solution):
            reward = 0.0
            
            try:
                # Extract answer from solution if it has <answer></answer> tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has <answer></answer> tags  
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Case-insensitive string comparison - handles variations like "Yes" vs "yes"
                if student_answer.lower() == ground_truth.lower():
                    reward = 1.0
                    
            except Exception:
                reward = 0.0  # Keep as 0.0 if extraction fails
                    
            rewards.append(reward)
        return rewards


class AnswerMatchString(ORM):
    """
    Exact string-based answer matching for VQA tasks.
    Matches model-generated answers against ground truth answers with case-insensitive comparison.
    Supports extracting answers from <answer></answer> tags in model completions.
    
    Best for: Closed-set answers (yes/no, fixed vocabularies)
    Method: Direct string comparison after normalization
    """

    def __call__(self, completions, answer, **kwargs) -> List[float]:
        rewards = []
        
        for content, ground_truth in zip(completions, answer):
            reward = 0.0
            
            try:
                # Ground truth answer is directly from the dataset annotation
                ground_truth = ground_truth.strip()

                # Extract answer from completion if it has <answer></answer> tags  
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                predicted_answer = content_match.group(1).strip() if content_match else content.strip()

                # Case-insensitive string comparison - handles variations like "Yes" vs "yes"
                if predicted_answer.lower() == ground_truth.lower():
                    reward = 1.0
                    
            except Exception:
                reward = 0.0  # Keep as 0.0 if extraction fails
                    
            rewards.append(reward)
        return rewards


class AnswerMatchCosine(ORM):
    """
    BERT-based semantic answer matching for VQA tasks using cosine similarity.
    
    Handles both closed-set (yes/no, medical terms) and open-set answers 
    using Sentence-BERT semantic similarity with normalized text comparison.
    
    Key features:
    - Normalizes answers (lowercase + strip) before comparison
    - Handles case-insensitive matching (Yes/YES/yes all treated equally)
    - Supports both exact and semantic matches
    - Smooth reward function for gradual feedback
    
    Example:
        prediction: "Yes"
        ground_truth: "yes"
        → normalized: "yes" vs "yes"
        → similarity: 1.0 > 0.80 threshold → reward = 1.0
        
        prediction: "multiple small infarcts in mca"
        ground_truth: "multiple infarcts showing mca"
        → normalized and compared
        → similarity: ~0.88 > 0.80 threshold → reward = 1.0
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 threshold: Optional[float] = None,
                 smooth_reward: Optional[bool] = True,
                 answer_match_cosine_model_name: Optional[str] = None,
                 answer_match_cosine_threshold: Optional[float] = None):
        """
        Initialize AnswerMatchCosine reward function.
        
        Args:
            model_name: SentenceTransformer model name (can be overridden by answer_match_cosine_model_name)
              - "sentence-transformers/all-mpnet-base-v2" (default, 通用强)
              - "all-MiniLM-L6-v2" (lightweight, general purpose)
              - "BAAI/bge-large-en-v1.5" (通用+医学平衡)
            threshold: Cosine similarity threshold (0-1, can be overridden by answer_match_cosine_threshold)
              - 0.75: More lenient, accepts more paraphrases
              - 0.80: Balanced (default, recommended for medical VQA)
              - 0.85: Conservative, stricter semantic matching
            smooth_reward: Use smooth reward function
              - True: reward = max(0, (similarity - threshold) * 10), capped at 1.0
              - False: reward = 1.0 if similarity > threshold else 0.0
            answer_match_cosine_model_name: Override model_name from command line (--answer_match_cosine_model_name)
            answer_match_cosine_threshold: Override threshold from command line (--answer_match_cosine_threshold)
        
        Recommended Models (for medical VQA tasks, ranked by recommendation):
        ======================================================================
        1. BAAI/bge-large-en-v1.5 - 通用+医学平衡，检索优化，MTEB排名靠前，适合问答匹配
        2. sentence-transformers/all-mpnet-base-v2 - 通用强，社区支持最好，STS基准表现优异
        3. dmis-lab/biobert-base-cased - 医学专有名词优化，PubMed数据训练，医学术语敏感
        4. intfloat/e5-large-v2 - 科学文献优化，指令式嵌入，BEIR基准表现好
        5. sentence-transformers/all-roberta-large-v1 - 精度最高，适合高精度要求场景
        6. BAAI/bge-base-en-v1.5 - 轻量级替代large版本，性价比高
        
        Recommended Threshold (based on industry best practices):
        ==========================================================
        - 0.70-0.75: Balanced, suitable for most medical VQA tasks (recommended starting point for experiments)
        - 0.75-0.80: More strict, higher precision, lower recall
        - 0.65-0.70: More lenient, higher recall, lower precision
        - Current default: 0.50 (kept for variable control in experiments, can be adjusted via --answer_match_cosine_threshold)
        - Note: With smooth_reward=True, threshold can be set 0.05-0.10 lower than hard threshold
        """
        # Debug: Log received parameters
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"AnswerMatchCosine __init__ received -> "
            f"answer_match_cosine_model_name={answer_match_cosine_model_name}, "
            f"answer_match_cosine_threshold={answer_match_cosine_threshold}, "
            f"model_name={model_name}, threshold={threshold}"
        )
        
        # Ignore any generic model_name/threshold from args; use specific overrides or defaults
        model_name = None
        threshold = None

        # Priority: CLI override > default
        if answer_match_cosine_model_name is not None and str(answer_match_cosine_model_name).strip() != "":
            model_name = str(answer_match_cosine_model_name).strip()
        else:
            model_name = "pritamdeka/S-BioBERT-snli-multinli-stsb"

        if answer_match_cosine_threshold is not None:
            threshold = answer_match_cosine_threshold
        else:
            threshold = 0.50
        
        # Default smooth_reward to True when not provided (preserve legacy default)
        if smooth_reward is None:
            smooth_reward = True

        # Log final selection
        logger.info(
            f"AnswerMatchCosine init -> model_name={model_name}, threshold={threshold}, smooth_reward={smooth_reward}"
        )
        self.model_name = model_name
        self.threshold = threshold
        self.smooth_reward = smooth_reward
        self.model = None
    
    def _load_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            import logging
            logger = logging.getLogger(__name__)
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Validate model was loaded successfully
                if self.model is None:
                    raise RuntimeError(f"Model {self.model_name} loaded but is None")
                
                # Validate tokenizer exists (required for encoding)
                if not hasattr(self.model, 'tokenizer') or self.model.tokenizer is None:
                    raise RuntimeError(
                        f"Model {self.model_name} loaded but tokenizer is None. "
                        f"This may indicate an incomplete model installation."
                    )
                
                logger.info(f"Successfully loaded model {self.model_name}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                error_msg = (
                    f"Failed to load model {self.model_name}: {e}. "
                    f"Check if model name is correct, download it first, "
                    f"or verify network connection if downloading from HuggingFace."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    def _normalize_answer(self, text: str) -> str:
        """
        Normalize answer text: lowercase + strip whitespace.
        
        This ensures case-insensitive comparison while preserving semantic meaning.
        """
        return text.lower().strip()
    
    def __call__(self, completions, answer, **kwargs) -> List[float]:
        """
        Calculate semantic answer matching rewards using cosine similarity.
        
        Extracts model-generated answer from <answer></answer> tags in completions,
        normalizes both predicted and ground-truth answers, then compares using 
        BERT embeddings and cosine similarity.
        
        Args:
            completions: Model-generated completions (List[str])
            answer: Ground-truth answers (List[str])
            **kwargs: Additional arguments for compatibility
        
        Returns:
            rewards: Reward scores (List[float], range 0-1)
        """
        rewards = []
        
        if not completions or not answer:
            return [0.0] * len(completions)
        
        try:
            # Lazy load model on first call
            self._load_model()
            
            # Validate model is ready before use
            if self.model is None:
                import logging
                logging.error(
                    f"AnswerMatchCosine: Model {self.model_name} is None after loading. "
                    f"Returning zero rewards."
                )
                return [0.0] * len(completions)
            
            if not hasattr(self.model, 'encode'):
                import logging
                logging.error(
                    f"AnswerMatchCosine: Model {self.model_name} does not have encode method. "
                    f"Returning zero rewards."
                )
                return [0.0] * len(completions)
            
            # Extract and normalize answers from <answer></answer> tags in completions
            extracted_answers = []
            for content in completions:
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if answer_match:
                    extracted_answers.append(
                        self._normalize_answer(answer_match.group(1))
                    )
                else:
                    # Fallback: use entire completion if no answer tags found
                    extracted_answers.append(self._normalize_answer(content))
            
            # Normalize ground truth answers
            normalized_gts = [self._normalize_answer(ans) for ans in answer]
            
            # Batch encode both extracted and ground-truth answers
            answer_embeddings = self.model.encode(
                extracted_answers,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            gt_embeddings = self.model.encode(
                normalized_gts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Calculate cosine similarity
            # For each prediction, calculate similarity with corresponding ground truth
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for pred_emb, gt_emb in zip(answer_embeddings, gt_embeddings):
                # Reshape to 2D for cosine_similarity
                sim = cosine_similarity([pred_emb], [gt_emb])[0, 0]
                similarities.append(sim)
            
            # Convert similarities to rewards
            for sim in similarities:
                sim_float = float(sim)
                
                if self.smooth_reward:
                    # Smooth reward: gradual increase around threshold
                    reward = max(0.0, (sim_float - self.threshold) * 5.0)
                    reward = min(1.0, reward)  # Cap at 1.0
                else:
                    # Hard reward: 0 or 1
                    reward = 1.0 if sim_float > self.threshold else 0.0
                
                rewards.append(reward)
        
        except Exception as e:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            error_details = traceback.format_exc()
            logger.error(
                f"AnswerMatchCosine calculation failed for model {self.model_name}: {e}\n"
                f"Error details: {error_details}\n"
                f"Returning zero rewards."
            )
            return [0.0] * len(completions)
        
        return rewards


class PlaneMatchString(ORM):
    """
    Exact string-based image plane matching for medical imaging tasks.
    Matches model-identified image planes against ground truth image planes with case-insensitive comparison.
    Supports extracting plane from <plane></plane> tags in model completions.
    
    Best for: Closed-set medical plane vocabulary (axial, pa, sagittal, etc.)
    Method: Direct string comparison after normalization
    """

    def __call__(self, completions, image_plane, **kwargs) -> List[float]:
        rewards = []
        
        for content, ground_truth in zip(completions, image_plane):
            reward = 0.0
            
            try:
                # Ground truth plane is directly from the dataset annotation
                ground_truth = ground_truth.strip()

                # Extract plane from completion if it has <plane></plane> tags  
                content_match = re.search(r'<plane>(.*?)</plane>', content)
                predicted_plane = content_match.group(1).strip() if content_match else ''

                # Case-insensitive string comparison for plane matching
                if predicted_plane and predicted_plane.lower() == ground_truth.lower():
                    reward = 1.0
                    
            except Exception:
                reward = 0.0  # Keep as 0.0 if extraction fails
                    
            rewards.append(reward)
        return rewards


class ModalityMatchString(ORM):
    """
    Exact string-based imaging modality matching for medical imaging tasks.
    Matches model-identified modalities against ground truth modalities with case-insensitive comparison.
    Supports extracting modality from <modality></modality> tags in model completions.
    
    Best for: Closed-set medical modality vocabulary (CT, X-ray, MRI, DWI, T1, T2, FLAIR, etc.)
    Method: Direct string comparison after normalization
    """

    def __call__(self, completions, image_modality, **kwargs) -> List[float]:
        rewards = []
        
        for content, ground_truth in zip(completions, image_modality):
            reward = 0.0
            
            try:
                # Ground truth modality is directly from the dataset annotation
                ground_truth = ground_truth.strip()

                # Extract modality from completion if it has <modality></modality> tags  
                content_match = re.search(r'<modality>(.*?)</modality>', content)
                predicted_modality = content_match.group(1).strip() if content_match else ''

                # Case-insensitive string comparison for modality matching
                if predicted_modality and predicted_modality.lower() == ground_truth.lower():
                    reward = 1.0
                    
            except Exception:
                reward = 0.0  # Keep as 0.0 if extraction fails
                    
            rewards.append(reward)
        return rewards


class CaptionMatchCosine(ORM):
    """
    BERT-based semantic caption matching using cosine similarity.
    
    Measures semantic similarity between model-generated completion and 
    ground-truth image caption using cosine similarity of sentence embeddings.
    
    Supports multiple embedding models for flexible experimentation.
    Best for: Open-set caption descriptions, semantic matching
    Method: BERT embeddings + cosine similarity
    
    Example:
        completion: "Multiple small infarcts in the MCA territory"
        caption: "Multiple small infarcts showing reduced diffusion..."
        similarity: 0.85 > 0.70 threshold → reward = 1.0
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 threshold: Optional[float] = None,
                 smooth_reward: Optional[bool] = True,
                 caption_match_cosine_model_name: Optional[str] = None,
                 caption_match_cosine_threshold: Optional[float] = None):
        """
        Initialize CaptionAlignment reward function.
        
        Args:
            model_name: SentenceTransformer model name (can be overridden by caption_match_cosine_model_name)
              - "sentence-transformers/all-mpnet-base-v2" (default, 通用强)
              - "all-MiniLM-L6-v2" (22M, lightweight)
              - "BAAI/bge-large-en-v1.5" (通用+医学平衡)
              - "pritamdeka/S-BioBERT-snli-multinli-stsb" (medical-specific)
              - "dmis-lab/biobert-base-cased" (medical BioBERT)
              - "allenai/scibert-base-uncased" (scientific papers)
              - "allenai/specter" (academic citations)
              - Others available for experimentation
            threshold: Similarity threshold (0-1, can be overridden by caption_match_cosine_threshold)
              - 0.65: Aggressive, easier to get reward
              - 0.70: Balanced (default, recommended)
              - 0.75: Conservative, strict requirement
            smooth_reward: Use smooth reward function
              - True: reward = max(0, (similarity - threshold) * 2.0)
              - False: reward = 1.0 if similarity > threshold else 0.0
            caption_match_cosine_model_name: Override model_name from command line (--caption_match_cosine_model_name)
            caption_match_cosine_threshold: Override threshold from command line (--caption_match_cosine_threshold)
        
        Recommended Models (for medical VQA tasks, ranked by recommendation):
        ======================================================================
        1. BAAI/bge-large-en-v1.5 - 通用+医学平衡，检索优化，MTEB排名靠前，适合问答匹配
        2. sentence-transformers/all-mpnet-base-v2 - 通用强，社区支持最好，STS基准表现优异
        3. dmis-lab/biobert-base-cased - 医学专有名词优化，PubMed数据训练，医学术语敏感
        4. intfloat/e5-large-v2 - 科学文献优化，指令式嵌入，BEIR基准表现好
        5. sentence-transformers/all-roberta-large-v1 - 精度最高，适合高精度要求场景
        6. BAAI/bge-base-en-v1.5 - 轻量级替代large版本，性价比高
        
        Recommended Threshold (based on industry best practices):
        ==========================================================
        - 0.60-0.65: Balanced for caption matching (recommended starting point for experiments)
        - 0.65-0.70: More strict, higher precision for longer descriptive text
        - 0.55-0.60: More lenient, higher recall for diverse caption styles
        - Current default: 0.30 (kept for variable control in experiments, can be adjusted via --caption_match_cosine_threshold)
        - Note: Caption text is typically longer than answers, so threshold can be lower than answer matching
        """
        # Ignore any generic model_name/threshold from args; use specific overrides or defaults
        model_name = None
        threshold = None

        # Priority: CLI override > default
        if caption_match_cosine_model_name is not None and str(caption_match_cosine_model_name).strip() != "":
            model_name = str(caption_match_cosine_model_name).strip()
        else:
            model_name = "pritamdeka/S-BioBERT-snli-multinli-stsb"

        if caption_match_cosine_threshold is not None:
            threshold = caption_match_cosine_threshold
        else:
            threshold = 0.30
        
        # Default smooth_reward to True when not provided (preserve legacy default)
        if smooth_reward is None:
            smooth_reward = True

        import logging
        logging.getLogger(__name__).info(
            f"CaptionMatchCosine init -> model_name={model_name}, threshold={threshold}, smooth_reward={smooth_reward}"
        )
        self.model_name = model_name
        self.threshold = threshold
        self.smooth_reward = smooth_reward
        self.model = None

    def _load_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            import logging
            logger = logging.getLogger(__name__)
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Validate model was loaded successfully
                if self.model is None:
                    raise RuntimeError(f"Model {self.model_name} loaded but is None")
                
                # Validate tokenizer exists (required for encoding)
                if not hasattr(self.model, 'tokenizer') or self.model.tokenizer is None:
                    raise RuntimeError(
                        f"Model {self.model_name} loaded but tokenizer is None. "
                        f"This may indicate an incomplete model installation."
                    )
                
                logger.info(f"Successfully loaded model {self.model_name}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                error_msg = (
                    f"Failed to load model {self.model_name}: {e}. "
                    f"Check if model name is correct, download it first, "
                    f"or verify network connection if downloading from HuggingFace."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    def __call__(self, completions, image_caption, **kwargs) -> List[float]:
        """
        Calculate caption alignment rewards.
        
        Extracts model-generated caption from <caption></caption> tags in completions,
        then compares with ground-truth image_caption using cosine similarity.
        
        Args:
            completions: Model-generated completions (List[str])
            image_caption: Ground-truth image captions (List[str])
            **kwargs: Additional arguments for compatibility
        
        Returns:
            rewards: Reward scores (List[float], range 0-1)
        """
        rewards = []
        
        if not completions or not image_caption:
            return [0.0] * len(completions)
        
        try:
            # Lazy load model on first call
            self._load_model()
            
            # Validate model is ready before use
            if self.model is None:
                import logging
                logging.error(
                    f"CaptionMatchCosine: Model {self.model_name} is None after loading. "
                    f"Returning zero rewards."
                )
                return [0.0] * len(completions)
            
            if not hasattr(self.model, 'encode'):
                import logging
                logging.error(
                    f"CaptionMatchCosine: Model {self.model_name} does not have encode method. "
                    f"Returning zero rewards."
                )
                return [0.0] * len(completions)

            # Extract captions from <caption></caption> tags in completions
            # If no tags found, use the entire completion as fallback
            extracted_captions = []
            for content in completions:
                caption_match = re.search(r'<caption>(.*?)</caption>', content, re.DOTALL)
                if caption_match:
                    extracted_captions.append(caption_match.group(1).strip())
                else:
                    # Fallback: use entire completion if no caption tags found
                    extracted_captions.append(content.strip())
            
            # Batch encode both extracted captions and ground-truth captions
            caption_embeddings = self.model.encode(
                extracted_captions,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            gt_caption_embeddings = self.model.encode(
                image_caption,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Calculate cosine similarity
            # For each prediction, calculate similarity with corresponding ground truth
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for pred_emb, gt_emb in zip(caption_embeddings, gt_caption_embeddings):
                # Reshape to 2D for cosine_similarity
                sim = cosine_similarity([pred_emb], [gt_emb])[0, 0]
                similarities.append(sim)
            
            # Convert similarities to rewards
            for sim in similarities:
                sim_float = float(sim)
                
                if self.smooth_reward:
                    # Smooth reward: gradual increase around threshold
                    reward = max(0.0, (sim_float - self.threshold) * 2.0)
                    reward = min(1.0, reward)  # Cap at 1.0
                else:
                    # Hard reward: 0 or 1
                    reward = 1.0 if sim_float > self.threshold else 0.0
                
                rewards.append(reward)
        
        except Exception as e:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            error_details = traceback.format_exc()
            logger.error(
                f"CaptionMatchCosine calculation failed for model {self.model_name}: {e}\n"
                f"Error details: {error_details}\n"
                f"Returning zero rewards."
            )
            return [0.0] * len(completions)
        
        return rewards


class TitleMatchCosine(ORM):
    """
    BERT-based semantic title matching using cosine similarity.
    
    Measures semantic similarity between model-generated completion and 
    ground-truth image title using cosine similarity of sentence embeddings.
    
    Supports extracting title from <title></title> tags in model completions.
    
    Best for: Medical diagnosis matching, high-level summary
    Method: BERT embeddings + cosine similarity
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 threshold: Optional[float] = None,
                 smooth_reward: Optional[bool] = True,
                 title_match_cosine_model_name: Optional[str] = None,
                 title_match_cosine_threshold: Optional[float] = None):
        """
        Initialize TitleMatchCosine reward function.
        
        Args:
            model_name: SentenceTransformer model name (can be overridden by title_match_cosine_model_name)
              - "sentence-transformers/all-mpnet-base-v2" (default, 通用强)
            threshold: Similarity threshold (0-1, can be overridden by title_match_cosine_threshold)
            smooth_reward: Use smooth reward function
            title_match_cosine_model_name: Override model_name from command line (--title_match_cosine_model_name)
            title_match_cosine_threshold: Override threshold from command line (--title_match_cosine_threshold)
        
        Recommended Models (for medical VQA tasks, ranked by recommendation):
        ======================================================================
        1. BAAI/bge-large-en-v1.5 - 通用+医学平衡，检索优化，MTEB排名靠前，适合问答匹配
        2. sentence-transformers/all-mpnet-base-v2 - 通用强，社区支持最好，STS基准表现优异
        3. dmis-lab/biobert-base-cased - 医学专有名词优化，PubMed数据训练，医学术语敏感
        4. intfloat/e5-large-v2 - 科学文献优化，指令式嵌入，BEIR基准表现好
        5. sentence-transformers/all-roberta-large-v1 - 精度最高，适合高精度要求场景
        6. BAAI/bge-base-en-v1.5 - 轻量级替代large版本，性价比高
        
        Recommended Threshold (based on industry best practices):
        ==========================================================
        - 0.65-0.70: Balanced for title matching (recommended starting point for experiments)
        - 0.70-0.75: More strict, higher precision for short title text
        - 0.60-0.65: More lenient, higher recall for diverse title formats
        - Current default: 0.30 (kept for variable control in experiments, can be adjusted via --title_match_cosine_threshold)
        - Note: Title text is typically shorter and more semantic, threshold can be between answer and caption
        """
        # Ignore any generic model_name/threshold from args; use specific overrides or defaults
        model_name = None
        threshold = None

        # Priority: CLI override > default
        if title_match_cosine_model_name is not None and str(title_match_cosine_model_name).strip() != "":
            model_name = str(title_match_cosine_model_name).strip()
        else:
            model_name = "pritamdeka/S-BioBERT-snli-multinli-stsb"

        if title_match_cosine_threshold is not None:
            threshold = title_match_cosine_threshold
        else:
            threshold = 0.30
        
        # Default smooth_reward to True when not provided (preserve legacy default)
        if smooth_reward is None:
            smooth_reward = True

        import logging
        logging.getLogger(__name__).info(
            f"TitleMatchCosine init -> model_name={model_name}, threshold={threshold}, smooth_reward={smooth_reward}"
        )
        self.model_name = model_name
        self.threshold = threshold
        self.smooth_reward = smooth_reward
        self.model = None

    def _load_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            import logging
            logger = logging.getLogger(__name__)
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Validate model was loaded successfully
                if self.model is None:
                    raise RuntimeError(f"Model {self.model_name} loaded but is None")
                
                # Validate tokenizer exists (required for encoding)
                if not hasattr(self.model, 'tokenizer') or self.model.tokenizer is None:
                    raise RuntimeError(
                        f"Model {self.model_name} loaded but tokenizer is None. "
                        f"This may indicate an incomplete model installation."
                    )
                
                logger.info(f"Successfully loaded model {self.model_name}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                error_msg = (
                    f"Failed to load model {self.model_name}: {e}. "
                    f"Check if model name is correct, download it first, "
                    f"or verify network connection if downloading from HuggingFace."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    def __call__(self, completions, image_title, **kwargs) -> List[float]:
        """
        Calculate title alignment rewards.
        
        Extracts model-generated title from <title></title> tags in completions,
        then compares with ground-truth image_title using cosine similarity.
        """
        rewards = []
        
        if not completions or not image_title:
            return [0.0] * len(completions)
        
        try:
            # Lazy load model on first call
            self._load_model()
            
            # Validate model is ready before use
            if self.model is None:
                import logging
                logging.error(
                    f"TitleMatchCosine: Model {self.model_name} is None after loading. "
                    f"Returning zero rewards."
                )
                return [0.0] * len(completions)
            
            if not hasattr(self.model, 'encode'):
                import logging
                logging.error(
                    f"TitleMatchCosine: Model {self.model_name} does not have encode method. "
                    f"Returning zero rewards."
                )
                return [0.0] * len(completions)

            # Extract titles from <title></title> tags in completions
            # If no tags found, use the entire completion as fallback
            extracted_titles = []
            for content in completions:
                title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
                if title_match:
                    extracted_titles.append(title_match.group(1).strip())
                else:
                    # Fallback: use entire completion if no title tags found
                    extracted_titles.append(content.strip())
            
            # Batch encode both extracted titles and ground-truth titles
            title_embeddings = self.model.encode(
                extracted_titles,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            gt_title_embeddings = self.model.encode(
                image_title,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for pred_emb, gt_emb in zip(title_embeddings, gt_title_embeddings):
                sim = cosine_similarity([pred_emb], [gt_emb])[0, 0]
                similarities.append(sim)
            
            # Convert similarities to rewards
            for sim in similarities:
                sim_float = float(sim)
                
                if self.smooth_reward:
                    # Smooth reward: gradual increase around threshold
                    reward = max(0.0, (sim_float - self.threshold) * 5.0) # Steeper slope for titles
                    reward = min(1.0, reward)
                else:
                    reward = 1.0 if sim_float > self.threshold else 0.0
                
                rewards.append(reward)
        
        except Exception as e:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            error_details = traceback.format_exc()
            logger.error(
                f"TitleMatchCosine calculation failed for model {self.model_name}: {e}\n"
                f"Error details: {error_details}\n"
                f"Returning zero rewards."
            )
            return [0.0] * len(completions)
        
        return rewards


class ReasoningConsistencyNLI(ORM):
    """
    基于NLI模型的推理-答案一致性检查奖励函数 (V2 改进版)
    
    改进点:
    1. 将原始问题纳入NLI判断，构造更准确的premise
    2. 只有预测答案正确 + 推理一致时才给正奖励
    3. 矛盾和中性情况统一返回0，避免NLI误判带来的惩罚
    
    检测 <think> 中的推理过程是否与 <answer> 中的答案一致。
    使用预训练的NLI模型判断推理是否蕴含(entail)答案。
    
    推荐模型（按推荐度降序排列，基于实际测试结果）:
    
    ⭐⭐⭐⭐⭐ 强烈推荐:
    - cross-encoder/nli-MiniLM2-L6-H768 (~200MB, 最佳性价比)
      测试表现: 1/2正确判断，平均Ent概率0.5561，对否定句式理解好
      说明: 基于知识蒸馏从RoBERTa-Large提取，在SNLI+MultiNLI联合训练，体积小性能好
    - facebook/bart-large-mnli (~1.5GB, 最高准确度)
      测试表现: 1/2正确判断，平均Ent概率0.5726，适合医学领域
      说明: BART大型版，在明确一致的样本上表现最佳（Ent概率94.41%）
    - cross-encoder/nli-roberta-base (~500MB, 平衡选择)
      测试表现: 1/2正确判断，平均Ent概率0.5249，性能稳定
      说明: RoBERTa基础版，在明确一致的样本上表现优秀（Ent概率93.45%）
    
    ⭐⭐⭐⭐ 推荐:
    - roberta-large-mnli (~1.4GB, 高准确度)
      测试表现: 1/2正确判断，平均Ent概率0.5002
    - typeform/distilbert-base-uncased-mnli (~250MB, 轻量级)
      测试表现: 1/2正确判断，平均Ent概率0.5077，但表现不稳定
    
    ⭐⭐⭐ 可考虑:
    - cross-encoder/nli-distilroberta-base (~400MB)
      测试表现: 0/2正确判断，平均Ent概率0.1571，不推荐
    - cross-encoder/nli-deberta-v3-small (~300MB)
      测试表现: 0/2正确判断，平均Ent概率0.1697，不推荐
    - cross-encoder/nli-deberta-v3-base (~800MB)
      测试表现: 0/2正确判断，平均Ent概率0.0079，不推荐
    - typeform/mobilebert-uncased-mnli (~100MB, 超轻量)
      测试表现: 0/2正确判断，平均Ent概率0.2370，表现一般
    
    ⚠️  不推荐（无NLI分类头）:
    - dmis-lab/biobert-base-cased-v1.1, monologg/biobert_v1.1_pubmed (BioBERT基础模型)
    - allenai/scibert_scivocab_uncased (SciBERT基础模型)
      说明: 这些模型不是专门的NLI模型，没有NLI分类头，无法准确判断
    
    性能说明:
    - MiniLM2系列通过知识蒸馏从大模型提取，在保持小体积的同时保留大部分性能
    - Cross-encoder架构比sentence-transformers的双编码器更准确，但推理更慢
    - 测试发现: 对于不确定性推理（如"could indicate"），大部分模型判断为neutral
    - 测试发现: 对于明确否定推理（如"no visible evidence"），多个模型能正确判断为entailment
    
    概率阈值方案风险分析:
    - 当前使用argmax判断（选择概率最大的类别）
    - 20%阈值方案风险: 可能误判Neutral为Entailment（如Neutral>60%但Ent>20%）
    - 改进建议: 使用阈值+排名检查+排除高Contradiction
      条件: entailment_prob > 0.2 AND entailment_prob >= neutral_prob AND contradiction_prob < 0.5
    
    Example:
        问题: "Is there oral contrast in the colon?"
        推理: "The image shows contrast material properly distributed in the colon"
        答案: "Yes"
        标签: "Yes"
        → 预测正确 ✓ + NLI=entailment ✓ → 奖励 1.0
        
        问题: "Is there oral contrast in the colon?"
        推理: "The image shows no obvious contrast in the colon"
        答案: "No"
        标签: "Yes"
        → 预测错误 ✗ → 奖励 0.0 (即使推理和答案一致)
    """
    
    def __init__(self,
                 model_name: str = "cross-encoder/nli-deberta-v3-base",
                 reward_entailment: float = 1.0,       # 蕴含(一致)且预测正确的奖励
                 penalty_contradiction: float = 0.0,   # 矛盾时不惩罚(避免NLI误判)
                 reward_neutral: float = 0.0,          # 中性时不奖励
                 use_gpu: bool = True,                 # 是否使用GPU
                 max_reasoning_length: int = 1000,     # 推理文本最大长度
                 require_correct_answer: bool = True): # 是否要求预测答案正确
        """
        初始化 ReasoningConsistencyNLI 奖励函数
        
        Args:
            model_name: NLI模型名称，根据显存选择合适的模型
            reward_entailment: 推理与答案一致且预测正确时的奖励值
            penalty_contradiction: 推理与答案矛盾时的惩罚值(建议设为0)
            reward_neutral: 推理与答案无明确关系时的奖励值
            use_gpu: 是否使用GPU进行推理
            max_reasoning_length: 推理文本的最大字符长度（避免OOM）
            require_correct_answer: 是否要求预测答案必须正确才给奖励
        """
        self.model_name = model_name
        self.reward_entailment = reward_entailment
        self.penalty_contradiction = penalty_contradiction
        self.reward_neutral = reward_neutral
        self.use_gpu = use_gpu
        self.max_reasoning_length = max_reasoning_length
        self.require_correct_answer = require_correct_answer
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 不同模型的标签顺序映射
        self.label_maps = {
            'cross-encoder': ['contradiction', 'entailment', 'neutral'],
            'facebook/bart': ['contradiction', 'neutral', 'entailment'],
            'microsoft/deberta': ['entailment', 'neutral', 'contradiction'],
            'roberta': ['contradiction', 'neutral', 'entailment'],
            'default': ['contradiction', 'entailment', 'neutral']
        }
    
    def _get_label_order(self) -> List[str]:
        """根据模型名称获取正确的标签顺序"""
        model_lower = self.model_name.lower()
        if 'cross-encoder' in model_lower:
            return self.label_maps['cross-encoder']
        elif 'bart' in model_lower:
            return self.label_maps['facebook/bart']
        elif 'deberta' in model_lower and 'cross-encoder' not in model_lower:
            return self.label_maps['microsoft/deberta']
        elif 'roberta' in model_lower:
            return self.label_maps['roberta']
        else:
            return self.label_maps['default']
    
    def _load_model(self):
        """懒加载NLI模型"""
        if self.model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.eval()
                
                # 设置设备
                if self.use_gpu and torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
                
                self.model.to(self.device)
                
                import logging
                logging.info(f"ReasoningConsistencyNLI: Loaded {self.model_name} on {self.device}")
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load NLI model {self.model_name}: {e}. "
                    f"Try a smaller model like 'cross-encoder/nli-MiniLM2-L6-H768'"
                )
    
    def _extract_question_from_messages(self, messages: List[Dict]) -> str:
        """
        从messages中提取用户的原始问题
        
        Args:
            messages: [{"role": "user", "content": "<image>Are regions of the brain infarcted?"}]
            
        Returns:
            question: "Are regions of the brain infarcted?"
        """
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # 移除 <image> 标签
                question = re.sub(r'<image>', '', content).strip()
                return question
        return ""
    
    def _safe_capitalize(self, text: str) -> str:
        """
        安全的首字母大写，保留医学缩写（如CT、MRI、DWI等）
        
        Args:
            text: "the CT scan" 或 "MRI image"
        
        Returns:
            "The CT scan" 或 "MRI image" (保留缩写大小写)
        """
        if not text:
            return text
        # 只大写第一个字符，保留其余原样
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    def _construct_direct_hypothesis(self, question: str, answer: str) -> str:
        """
        将问题和答案转换为直接陈述句 (NLI hypothesis)
        
        全面覆盖各种问题类型:
        - Is there X / Are there X
        - Is X / Are X
        - Does X / Do X
        - Was X / Were X
        - Has X / Have X
        - Can X / Could X / Should X / Would X
        - Contains "any"
        
        Args:
            question: "Is there oral contrast in the colon?"
            answer: "Yes" 或 "No"
        
        Returns:
            hypothesis: "There is oral contrast in the colon." 或
                      "There is no oral contrast in the colon."
        """
        # 移除问号并标准化
        q = question.rstrip('?').strip()
        q_lower = q.lower()
        answer_lower = answer.lower().strip()
        is_positive = answer_lower in ['yes', 'true']
        
        # 1. Is there X / Are there X
        if q_lower.startswith('is there '):
            subject = q[9:]  # 移除 "Is there "
            if is_positive:
                return f"There is {subject}."
            else:
                return f"There is no {subject}."
        
        elif q_lower.startswith('are there '):
            subject = q[10:]  # 移除 "Are there " (10个字符)
            if is_positive:
                return f"There are {subject}."
            else:
                return f"There are no {subject}."
        
        # 2. Is X / Are X (主语 + 谓语)
        elif q_lower.startswith('is ') and not q_lower.startswith('is there'):
            rest = q[3:].strip()  # 移除 "Is "
            if is_positive:
                # "Is the heart enlarged?" → "The heart is enlarged."
                # 如果rest中已有"is"，直接使用
                if ' is ' in rest.lower() or rest.lower().startswith('this ') or rest.lower().startswith('that '):
                    return f"{rest}."
                else:
                    # 假设最后一个词是形容词/谓语，前面都是主语
                    # "the CT scan normal" → subject="the CT scan", predicate="normal"
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(' '.join(words[:-1]))  # 除最后一个词外都是主语
                        predicate = words[-1]  # 最后一个词是谓语
                        return f"{subject} is {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} is true."
            else:
                # "Is the heart enlarged?" → "The heart is not enlarged."
                if ' is ' in rest.lower():
                    return rest.replace(' is ', ' is not ', 1) + "."
                elif ' is not ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(' '.join(words[:-1]))
                        predicate = words[-1]
                        return f"{subject} is not {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} is not true."
        
        elif q_lower.startswith('are ') and not q_lower.startswith('are there'):
            rest = q[4:].strip()  # 移除 "Are "
            if is_positive:
                # "Are the lungs normal?" → "The lungs are normal."
                if ' are ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(' '.join(words[:-1]))
                        predicate = words[-1]
                        return f"{subject} are {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} are true."
            else:
                # "Are the lungs normal?" → "The lungs are not normal."
                if ' are ' in rest.lower():
                    return rest.replace(' are ', ' are not ', 1) + "."
                elif ' are not ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(' '.join(words[:-1]))
                        predicate = words[-1]
                        return f"{subject} are not {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} are not true."
        
        # 3. Does X / Do X
        elif q_lower.startswith('does '):
            rest = q[5:].strip()  # 移除 "Does "
            words = rest.split()
            if is_positive:
                # "Does the heart appear enlarged?" → "The heart appears enlarged."
                if len(words) >= 3:
                    subject = self._safe_capitalize(' '.join(words[:2]))
                    verb = words[2]
                    predicate = ' '.join(words[3:]) if len(words) > 3 else ""
                    # 将动词改为第三人称单数
                    if not verb.endswith('s') and not verb.endswith('ed'):
                        verb = verb + 's'
                    result = f"{subject} {verb}"
                    if predicate:
                        result += f" {predicate}"
                    return result + "."
                else:
                    # 回退到安全模式
                    return f"The answer to the question is yes."
            else:
                # "Does the heart appear enlarged?" → "The heart does not appear enlarged."
                if len(words) >= 3:
                    subject = self._safe_capitalize(' '.join(words[:2]))
                    predicate = ' '.join(words[2:])
                    return f"{subject} does not {predicate}."
                else:
                    return f"The answer to the question is no."
        
        elif q_lower.startswith('do '):
            rest = q[3:].strip()  # 移除 "Do "
            words = rest.split()
            if is_positive:
                # "Do the lungs extend?" → "The lungs extend."
                return f"{self._safe_capitalize(rest)}."
            else:
                # "Do the lungs extend?" → "The lungs do not extend."
                if len(words) >= 2:
                    subject = self._safe_capitalize(' '.join(words[:2]))
                    predicate = ' '.join(words[2:]) if len(words) > 2 else words[1]
                    return f"{subject} do not {predicate}."
                else:
                    return f"The answer to the question is no."
        
        # 4. Was X / Were X (过去时)
        elif q_lower.startswith('was '):
            rest = q[4:].strip()  # 移除 "Was "
            if is_positive:
                # "Was contrast used?" → "Contrast was used."
                if ' was ' in rest.lower():
                    return f"{self._safe_capitalize(rest)}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(words[0])
                        predicate = ' '.join(words[1:])
                        return f"{subject} was {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} was true."
            else:
                # "Was contrast used?" → "Contrast was not used."
                if ' was ' in rest.lower():
                    return self._safe_capitalize(rest.replace(' was ', ' was not ', 1)) + "."
                elif ' was not ' in rest.lower():
                    return f"{self._safe_capitalize(rest)}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(words[0])
                        predicate = ' '.join(words[1:])
                        return f"{subject} was not {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} was not true."
        
        elif q_lower.startswith('were '):
            rest = q[5:].strip()  # 移除 "Were "
            if is_positive:
                # "Were the lungs normal?" → "The lungs were normal."
                if ' were ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        return f"{words[0]} were {' '.join(words[1:])}."
                    else:
                        return f"{rest} were true."
            else:
                # "Were the lungs normal?" → "The lungs were not normal."
                if ' were ' in rest.lower():
                    return rest.replace(' were ', ' were not ', 1) + "."
                elif ' were not ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        return f"{words[0]} were not {' '.join(words[1:])}."
                    else:
                        return f"{rest} were not true."
        
        # 5. Has X / Have X (完成时)
        elif q_lower.startswith('has '):
            rest = q[4:].strip()  # 移除 "Has "
            if is_positive:
                # "Has the bowel perforated?" → "The bowel has perforated."
                if ' has ' in rest.lower():
                    return f"{self._safe_capitalize(rest)}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(' '.join(words[:2]))
                        predicate = ' '.join(words[2:]) if len(words) > 2 else words[1]
                        return f"{subject} has {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} has occurred."
            else:
                # "Has the bowel perforated?" → "The bowel has not perforated."
                if ' has ' in rest.lower():
                    return self._safe_capitalize(rest.replace(' has ', ' has not ', 1)) + "."
                elif ' has not ' in rest.lower():
                    return f"{self._safe_capitalize(rest)}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        subject = self._safe_capitalize(' '.join(words[:2]))
                        predicate = ' '.join(words[2:]) if len(words) > 2 else words[1]
                        return f"{subject} has not {predicate}."
                    else:
                        return f"{self._safe_capitalize(rest)} has not occurred."
        
        elif q_lower.startswith('have '):
            rest = q[5:].strip()  # 移除 "Have "
            if is_positive:
                # "Have brain structures crossed?" → "Brain structures have crossed."
                if ' have ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        return f"{words[0]} have {' '.join(words[1:])}."
                    else:
                        return f"{rest} have occurred."
            else:
                # "Have brain structures crossed?" → "Brain structures have not crossed."
                if ' have ' in rest.lower():
                    return rest.replace(' have ', ' have not ', 1) + "."
                elif ' have not ' in rest.lower():
                    return f"{rest}."
                else:
                    words = rest.split()
                    if len(words) >= 2:
                        return f"{words[0]} have not {' '.join(words[1:])}."
                    else:
                        return f"{rest} have not occurred."
        
        # 6. Can X / Could X / Should X / Would X (情态动词)
        elif q_lower.startswith('can '):
            rest = q[4:]  # 移除 "Can "
            if is_positive:
                # "Can you see X?" → "X can be seen." 或 "You can see X."
                if rest.lower().startswith('you '):
                    return f"{rest}."
                else:
                    return f"{rest} can be observed."
            else:
                if rest.lower().startswith('you '):
                    return rest.replace('you ', 'you cannot ', 1) + "."
                else:
                    return f"{rest} cannot be observed."
        
        elif q_lower.startswith('could '):
            rest = q[6:]  # 移除 "Could "
            if is_positive:
                return f"{rest}."
            else:
                return rest.replace(' ', ' could not ', 1) + "."
        
        elif q_lower.startswith('should '):
            rest = q[7:]  # 移除 "Should "
            if is_positive:
                return f"{rest}."
            else:
                return rest.replace(' ', ' should not ', 1) + "."
        
        elif q_lower.startswith('would '):
            rest = q[6:]  # 移除 "Would "
            if is_positive:
                return f"{rest}."
            else:
                return rest.replace(' ', ' would not ', 1) + "."
        
        # 7. Contains "any" (特殊处理)
        elif 'any ' in q_lower:
            # "any observed degenerative changes?" → "There are observed degenerative changes."
            # "any abnormal findings in the lower lung fields?" → "There are abnormal findings in the lower lung fields."
            # 移除 "any" 并转换为陈述句
            q_without_any = re.sub(r'\bany\b', '', q, flags=re.IGNORECASE).strip()
            # 检查是否包含复数名词（findings, changes, signs等）
            plural_keywords = ['findings', 'changes', 'signs', 'evidence', 'observed', 'abnormal findings', 'lesions', 'masses']
            is_plural = any(keyword in q_without_any.lower() for keyword in plural_keywords)
            
            if is_positive:
                # 使用 "There are" 对于复数，否则 "There is"
                if is_plural:
                    return f"There are {q_without_any}."
                else:
                    return f"There is {q_without_any}."
            else:
                if is_plural:
                    return f"There are no {q_without_any}."
                else:
                    return f"There is no {q_without_any}."
        
        # 默认处理：对于未覆盖的问题类型
        else:
            if is_positive:
                # 尝试将问题转换为陈述句
                # "What is X?" → "X is present." (但What类型通常不是Yes/No)
                return f"The answer to '{question}' is yes."
            else:
                return f"The answer to '{question}' is no."
    
    def _batch_nli_prediction(self, premises: List[str], hypotheses: List[str]) -> List[str]:
        """
        批量NLI预测
        
        Args:
            premises: 前提列表（问题 + 推理过程）
            hypotheses: 假设列表（答案陈述）
            
        Returns:
            预测结果列表: 'entailment', 'contradiction', 或 'neutral'
        """
        import torch
        
        if not premises:
            return []
        
        labels = self._get_label_order()
        results = []
        
        # 批量编码
        inputs = self.tokenizer(
            premises, 
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
        
        for pred_idx in predictions:
            results.append(labels[pred_idx])
        
        return results
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        计算推理-答案一致性奖励
        
        改进的工作流程:
        1. 从kwargs中提取原始问题（messages字段）和标签答案（answer字段）
        2. 从completion中提取推理过程和预测答案
        3. 检查预测答案是否正确（如果require_correct_answer=True）
        4. 构造NLI输入: premise = "Question: {q}\nReasoning: {r}"
        5. 只有预测正确 + NLI=entailment时才给奖励
        
        Args:
            completions: 模型生成的完整回复列表
            **kwargs: 包含 messages（原始问题）和 answer（标签答案）
            
        Returns:
            rewards: 奖励值列表
        """
        # 懒加载模型
        self._load_model()
        
        rewards = [0.0] * len(completions)
        premises = []
        hypotheses = []
        valid_indices = []
        answer_correct_flags = []  # 记录预测答案是否正确
        
        # 从kwargs获取messages和answer
        # kwargs格式: {'messages': [...], 'answer': [...], ...}
        messages_list = kwargs.get('messages', [])
        ground_truth_list = kwargs.get('answer', [])
        
        # 确保列表长度匹配
        if not isinstance(messages_list, list):
            messages_list = [messages_list] * len(completions)
        if not isinstance(ground_truth_list, list):
            ground_truth_list = [ground_truth_list] * len(completions)
        
        for i, content in enumerate(completions):
            try:
                # 提取 <think> 和 <answer>
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                
                if think_match and answer_match:
                    reasoning = think_match.group(1).strip()
                    predicted_answer = answer_match.group(1).strip()
                    
                    # 只处理 Yes/No 类型的答案
                    answer_lower = predicted_answer.lower()
                    if answer_lower in ['yes', 'no', 'true', 'false']:
                        # 提取原始问题
                        question = ""
                        if i < len(messages_list) and messages_list[i]:
                            msgs = messages_list[i]
                            if isinstance(msgs, list):
                                question = self._extract_question_from_messages(msgs)
                            elif isinstance(msgs, str):
                                question = re.sub(r'<image>', '', msgs).strip()
                        
                        # 检查预测答案是否正确
                        is_correct = True
                        if self.require_correct_answer and i < len(ground_truth_list):
                            gt = ground_truth_list[i]
                            if gt:
                                gt_lower = str(gt).strip().lower()
                                is_correct = (answer_lower == gt_lower)
                        
                        # 截断过长的推理文本以避免OOM
                        reasoning_truncated = reasoning[:self.max_reasoning_length]
                        
                        # 构造包含问题的premise (改进点1)
                        if question:
                            premise = f"Question: {question}\nReasoning: {reasoning_truncated}"
                        else:
                            premise = f"Reasoning: {reasoning_truncated}"
                        
                        # 使用直接hypothesis构造方式 (方案A)
                        hypothesis = self._construct_direct_hypothesis(question if question else "", predicted_answer)
                        
                        premises.append(premise)
                        hypotheses.append(hypothesis)
                        valid_indices.append(i)
                        answer_correct_flags.append(is_correct)
                        
            except Exception as e:
                import logging
                logging.debug(f"ReasoningConsistencyNLI: Failed to process completion {i}: {e}")
                continue
        
        # 批量预测
        if premises:
            try:
                nli_results = self._batch_nli_prediction(premises, hypotheses)
                
                for idx, result, is_correct in zip(valid_indices, nli_results, answer_correct_flags):
                    if result == 'entailment':
                        # 改进点4: 只有预测正确且一致时才给奖励
                        if self.require_correct_answer:
                            rewards[idx] = self.reward_entailment if is_correct else 0.0
                        else:
                            rewards[idx] = self.reward_entailment
                    elif result == 'contradiction':
                        # 改进点2&3: 矛盾时不惩罚
                        rewards[idx] = self.penalty_contradiction
                    else:  # neutral
                        rewards[idx] = self.reward_neutral
                        
            except Exception as e:
                import logging
                logging.warning(f"ReasoningConsistencyNLI batch prediction failed: {e}")
                # 发生错误时保持默认的0.0奖励
        
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'smart_accuracy': SmartAccuracy,
    'answer_match_string': AnswerMatchString,
    # Changed to class to support command line parameters (--answer_match_cosine_model_name, --answer_match_cosine_threshold)
    # Default values: model_name="pritamdeka/S-BioBERT-snli-multinli-stsb", threshold=0.50
    'answer_match_cosine': AnswerMatchCosine,
    'plane_match_string': PlaneMatchString,
    'modality_match_string': ModalityMatchString,
    # Changed to class to support command line parameters (--caption_match_cosine_model_name, --caption_match_cosine_threshold)
    # Default values: model_name="pritamdeka/S-BioBERT-snli-multinli-stsb", threshold=0.30
    'caption_match_cosine': CaptionMatchCosine,
    # Changed to class to support command line parameters (--title_match_cosine_model_name, --title_match_cosine_threshold)
    # Default values: model_name="pritamdeka/S-BioBERT-snli-multinli-stsb", threshold=0.30
    'title_match_cosine': TitleMatchCosine,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    # 推理-答案一致性奖励函数 V2 (按显存需求从小到大)
    # 改进: 1.问题纳入NLI 2.矛盾不惩罚 3.预测正确+一致才奖励
    'reasoning_consistency_nli_mini': ReasoningConsistencyNLI(model_name="cross-encoder/nli-MiniLM2-L6-H768", reward_entailment=1.0, penalty_contradiction=0.0, require_correct_answer=True),
    'reasoning_consistency_nli_large': ReasoningConsistencyNLI(model_name="facebook/bart-large-mnli", reward_entailment=1.0, penalty_contradiction=0.0, require_correct_answer=True),
}
