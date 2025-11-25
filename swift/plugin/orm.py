import os
import re
from typing import TYPE_CHECKING, Dict, List, Union

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
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
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
        # pattern = r'^<think>.*?</think>\s*<plane>.*?</plane>\s*<modality>.*?</modality>\s*<title>.*?</title>\s*<caption>.*?</caption>\s*<answer>.*?</answer>(?![\s\S])'
        pattern = r'^<plane>.*?</plane>\s*<modality>.*?</modality>\s*<title>.*?</title>\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        
        rewards = []
        for i, match in enumerate(matches):
            if not match:
                rewards.append(0.0)
                continue
            
            # Format check passed, now check consistency between Final Answer and <answer>
            content = completions[i]
            
            # Extract <think> content
            reasoning_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if not reasoning_match:
                rewards.append(0.0)
                continue
            
            reasoning_text = reasoning_match.group(1)
            
            # Extract "Final Answer:" text (take the last occurrence if multiple)
            # Match "Final Answer:" followed by text until newline or end of string
            final_answer_pattern = r'Final Answer:\s*(.+?)(?=\n|$)'
            final_answer_matches = re.findall(final_answer_pattern, reasoning_text, re.IGNORECASE | re.MULTILINE)
            if not final_answer_matches:
                rewards.append(0.0)
                continue
            
            # Take the last occurrence
            final_answer_text = final_answer_matches[-1].strip()
            
            # Extract <answer> tag content
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if not answer_match:
                rewards.append(0.0)
                continue
            
            answer_text = answer_match.group(1).strip()
            
            # Normalize both texts: handle whitespace/hyphen variations, lowercase, remove trailing punctuation
            def normalize_text(text):
                text = text.strip()
                # Normalize hyphens to spaces: all hyphens (with or without surrounding spaces) become single space
                # This makes "XR - Plain Film", "XR-Plain Film", and "XR Plain Film" all equivalent
                # First, normalize hyphen with spaces around it: " - " -> " "
                text = re.sub(r'\s*-\s*', ' ', text)
                # Normalize multiple consecutive whitespace to single space
                text = re.sub(r'\s+', ' ', text)
                text = text.lower()
                # Remove trailing punctuation (.,;:!?)
                text = re.sub(r'[.,;:!?]+$', '', text)
                return text.strip()
            
            normalized_final_answer = normalize_text(final_answer_text)
            normalized_answer = normalize_text(answer_text)
            
            # Check consistency
            if normalized_final_answer == normalized_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards


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
                 model_name: str = "pritamdeka/S-BioBERT-snli-multinli-stsb",
                 threshold: float = 0.70,
                 smooth_reward: bool = True):
        """
        Initialize AnswerMatchCosine reward function.
        
        Args:
            model_name: SentenceTransformer model name
              - "pritamdeka/S-BioBERT-snli-multinli-stsb" (medical-specific, default)
              - "all-MiniLM-L6-v2" (lightweight, general purpose)
              - "all-mpnet-base-v2" (high quality, general purpose)
            threshold: Cosine similarity threshold (0-1)
              - 0.75: More lenient, accepts more paraphrases
              - 0.80: Balanced (default, recommended for medical VQA)
              - 0.85: Conservative, stricter semantic matching
            smooth_reward: Use smooth reward function
              - True: reward = max(0, (similarity - threshold) * 10), capped at 1.0
              - False: reward = 1.0 if similarity > threshold else 0.0
        """
        self.model_name = model_name
        self.threshold = threshold
        self.smooth_reward = smooth_reward
        self.model = None
    
    def _load_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model {self.model_name}: {e}. "
                    f"Check if model name is correct or download it first."
                )
    
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
                    reward = max(0.0, (sim_float - self.threshold) * 10)
                    reward = min(1.0, reward)  # Cap at 1.0
                else:
                    # Hard reward: 0 or 1
                    reward = 1.0 if sim_float > self.threshold else 0.0
                
                rewards.append(reward)
        
        except Exception as e:
            import logging
            logging.warning(
                f"AnswerMatchCosine calculation failed: {e}. "
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
                 model_name: str = "pritamdeka/S-BioBERT-snli-multinli-stsb",
                 threshold: float = 0.40,
                 smooth_reward: bool = True):
        """
        Initialize CaptionAlignment reward function.
        
        Args:
            model_name: SentenceTransformer model name
              - "all-MiniLM-L6-v2" (default, 22M, lightweight)
              - "all-mpnet-base-v2" (109M, high quality)
              - "pritamdeka/S-BioBERT-snli-multinli-stsb" (medical-specific)
              - "dmis-lab/biobert-base-cased" (medical BioBERT)
              - "allenai/scibert-base-uncased" (scientific papers)
              - "allenai/specter" (academic citations)
              - Others available for experimentation
            threshold: Similarity threshold (0-1)
              - 0.65: Aggressive, easier to get reward
              - 0.70: Balanced (default, recommended)
              - 0.75: Conservative, strict requirement
            smooth_reward: Use smooth reward function
              - True: reward = max(0, (similarity - threshold) * 2.0)
              - False: reward = 1.0 if similarity > threshold else 0.0
        """
        self.model_name = model_name
        self.threshold = threshold
        self.smooth_reward = smooth_reward
        self.model = None

    def _load_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model {self.model_name}: {e}. "
                    f"Check if model name is correct or download it first."
                )
    
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
            logging.warning(
                f"CaptionAlignment calculation failed: {e}. "
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
                 model_name: str = "pritamdeka/S-BioBERT-snli-multinli-stsb",
                 threshold: float = 0.60,
                 smooth_reward: bool = True):
        """
        Initialize TitleMatchCosine reward function.
        
        Args:
            model_name: SentenceTransformer model name
            threshold: Similarity threshold (0-1)
            smooth_reward: Use smooth reward function
        """
        self.model_name = model_name
        self.threshold = threshold
        self.smooth_reward = smooth_reward
        self.model = None

    def _load_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model {self.model_name}: {e}. "
                    f"Check if model name is correct or download it first."
                )
    
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
            logging.warning(
                f"TitleMatchCosine calculation failed: {e}. "
                f"Returning zero rewards."
            )
            return [0.0] * len(completions)
        
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'smart_accuracy': SmartAccuracy,
    'answer_match_string': AnswerMatchString,
    'answer_match_cosine': AnswerMatchCosine(model_name="pritamdeka/S-BioBERT-snli-multinli-stsb", threshold=0.70, smooth_reward=True),
    'plane_match_string': PlaneMatchString,
    'modality_match_string': ModalityMatchString,
    'caption_match_cosine': CaptionMatchCosine(model_name="pritamdeka/S-BioBERT-snli-multinli-stsb", threshold=0.40, smooth_reward=True),
    'title_match_cosine': TitleMatchCosine(model_name="pritamdeka/S-BioBERT-snli-multinli-stsb", threshold=0.40, smooth_reward=True),
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
