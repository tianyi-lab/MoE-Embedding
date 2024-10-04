from typing import Dict, List, Union, cast

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig


import sys
from models.modeling_deepseek import DeepseekForCausalLM
from models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from models.modeling_olmoe import OlmoeForCausalLM

import os
from sklearn.decomposition import PCA

def load_pretrained_model(base_model, model_type) -> tuple:
    """ Loads a pretrained model from HuggingFace.

    Args:
        base_model (str): name of model (e.g. "mistralai/Mistral-7B-v0.1")
        model_type (str): Type of model to load ("deepseek-moe", "Qwen", "OLMoE")

    Returns:
        tuple(model, tokenizer): Loaded model and tokenizer
    """
    
    # Configuration for 4-bit quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = "left"

    # Load the model based on the specified model type
    if model_type == 'deepseek-moe':
        model = DeepseekForCausalLM.from_pretrained(
            base_model,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            quantization_config=nf4_config,
            use_cache=False,
            trust_remote_code=True,
        )
    elif model_type == 'Qwen':
        model = Qwen2MoeForCausalLM.from_pretrained(
            base_model,
            quantization_config=nf4_config
        )
    elif model_type == 'OLMoE':
        model = OlmoeForCausalLM.from_pretrained(
            base_model,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            quantization_config=nf4_config,
            use_cache=False,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()

    return model, tokenizer

class MOEE(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        mode: str = 'unified', # One of ['unified', 'embedding', 'generative']        
        pooling_method: str = 'mean', # One of ['cls', 'lasttoken', 'mean', 'weightedmean']
        normalized: bool = True,
        projection: int = None,
        is_inference: bool = True,
        embed_eos: str = "",
        attn: str = 'bbcc',
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_4bit: bool = True,
        nf4_config = None,
        **kwargs, # Passed to the model, e.g. `attn_implementation`, `torch_dtype` etc.
    ) -> None:
        super().__init__()
        
        self.base_model = model_name_or_path

        if 'deepseek-moe' in model_name_or_path:
            self.model, self.tokenizer = load_pretrained_model(model_name_or_path, 'deepseek-moe')
        elif 'Qwen' in model_name_or_path:
            self.model, self.tokenizer = load_pretrained_model(model_name_or_path, 'Qwen')
        elif 'OLMoE' in model_name_or_path:
            self.model, self.tokenizer = load_pretrained_model(model_name_or_path, 'OLMoE')
            
        print('self.model: ', self.model)

        if hasattr(self.model, 'model'): # LLama2 & Mistral
            self.embedding_attr = 'model'
        elif hasattr(self.model, 'transformer'): # GPT-Neo & GPT-J
            self.embedding_attr = 'transformer'
        else: 
            raise ValueError("Could not find attribute to use for embedding: ", self.model)

        self.projection = torch.nn.Linear(
            in_features=self.model.config.hidden_size, 
            out_features=int(projection),
            dtype=self.model.dtype
        ) if projection is not None else None
        self.normalized = normalized
        self.pooling_method = pooling_method
        self.device = device
        self.num_gpus = 1
        self.embed_eos = embed_eos
        self.attn = attn
        if (self.attn is not None) and self.attn not in ['bbcc', 'cccc', 'bb', 'cc']:
            raise ValueError(f"Mixed attention no longer supported: {self.attn}. Only bbcc, cccc, bb, cc are supported")

    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        embed_instruction: bool = False,
        get_cache: bool = False,
        convert_to_tensor: bool = False,
        recast: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> np.ndarray:

        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings, all_kv_caches = [], []
        all_default, all_moe_rw = [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            sentences_batch = [
                instruction + s + self.embed_eos for s in sentences[start_index:start_index + batch_size]
            ]

            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)
            
            if kwargs['embed_method'] == 'prompteol':
                prompt_templates = ["This sentence : \"*sent 0*\" means in one word:\""]
            elif kwargs['embed_method'] == 'none':
                prompt_templates = ["\"*sent 0*\""]
            elif kwargs['embed_method'] == 'style':
                prompt_templates = ["In one word, describe the style of the following sentence - \"*sent 0*\" :\""]
            elif kwargs['embed_method'] == 'sentiment':
                prompt_templates = ["In one word, describe the sentiment of the following sentence (positive, neutral, or negative) - \"*sent 0*\" :"]
            elif kwargs['embed_method'] == 'tone':
                prompt_templates = ["In one word, describe the tone of the following sentence - \"*sent 0*\" (e.g., formal, informal, humorous, serious):"]
            elif kwargs['embed_method'] == 'intent':
                prompt_templates = ["In one word, describe the intent behind the following sentence (e.g., request, suggestion, command) - \"*sent 0*\" :"]
            elif kwargs['embed_method'] == 'complexity':
                prompt_templates = ["In one word, rate the complexity of the following sentence (simple, moderate, complex) - \"*sent 0*\" :"]
            elif kwargs['embed_method'] == 'subjectivity':
                prompt_templates = ["In one word, describe whether the following sentence is subjective or objective - \"*sent 0*\" :"]
            elif kwargs['embed_method'] == 'language_style':
                prompt_templates = ["In one word, describe the language style of the following sentence (e.g., academic, conversational, literary) - \"*sent 0*\" :"]
            elif kwargs['embed_method'] == 'grammar_structure':
                prompt_templates = ["In one word, describe the grammatical structure of the following sentence (simple, compound, complex) - \"*sent 0*\" :"]
                
            sentences_batch = [s.split(' ') for s in sentences_batch]

            if len(sentences_batch) > 0 and len(sentences_batch[0]) > 0 and isinstance(sentences_batch[0][0], bytes):
                sentences_batch = [[word.decode('utf-8') for word in s] for s in sentences_batch]

            prompts = []
            for sent in sentences_batch:
                sent = ' '.join(sent) if sent != [] else '.'
                if len(sent) > 0 and sent[-1] not in '.?!"\'': 
                        sent += '.'
                sent = sent.replace('"', '\'')
                    
                for prompt in prompt_templates:
                    prompts.append(prompt.replace('*sent 0*', sent).replace('_', ' ').strip())
            
            prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")

            if get_cache:
                inputs['use_cache'] = True               
                
            outputs, sent_emb = self.model(**prompts, output_hidden_states=True, return_dict=True)                    
            sent_emb = sent_emb.cpu()
            
            lst_token_emb = outputs.hidden_states[-1][:, -1, :].cpu()
            
            lst_token_rw = sent_emb[:, :, -1, :]
            lst_token_rw = torch.cat([lst_token_rw[:, i, :] for i in range(lst_token_rw.shape[1])], dim=1)
            
            if kwargs['emb_info'] == 'HS':
                embeddings = lst_token_emb
            elif kwargs['emb_info'] == 'RW':
                embeddings = lst_token_rw
            elif kwargs['emb_info'] == 'MoEE':
                embeddings = torch.cat([lst_token_emb, lst_token_rw], dim=1)
                
            if kwargs['emb_info'] != 'HS':  
                mean = embeddings.mean(dim=0, keepdim=True)
                std = embeddings.std(dim=0, keepdim=True)
                embeddings = (embeddings - mean) / (std + 1e-8)
                
                mean = lst_token_emb.mean(dim=0, keepdim=True)
                std = lst_token_emb.std(dim=0, keepdim=True)
                lst_token_emb = (lst_token_emb - mean) / (std + 1e-8)
                
                mean = lst_token_rw.mean(dim=0, keepdim=True)
                std = lst_token_rw.std(dim=0, keepdim=True)
                lst_token_rw = (lst_token_rw - mean) / (std + 1e-8)

            all_embeddings.append(embeddings.cpu().numpy())
            all_default.append(lst_token_emb.cpu().numpy())
            all_moe_rw.append(lst_token_rw.cpu().numpy())

        try:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_default = torch.cat(all_default, dim=0)
            all_moe_rw = torch.cat(all_moe_rw, dim=0)
        except:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_default = np.concatenate(all_default, axis=0)
            all_moe_rw = np.concatenate(all_moe_rw, axis=0)
    
        if input_was_string:
            all_embeddings = all_embeddings[0]
        if get_cache:
            return all_embeddings, all_kv_caches
        
        # Ensure all variables are PyTorch tensors before using torch operations
        if isinstance(all_embeddings, np.ndarray):
            all_embeddings = torch.from_numpy(all_embeddings)

        if isinstance(all_default, np.ndarray):
            all_default = torch.from_numpy(all_default)

        if isinstance(all_moe_rw, np.ndarray):
            all_moe_rw = torch.from_numpy(all_moe_rw)

        all_embeddings = torch.where(torch.isnan(all_embeddings), torch.zeros_like(all_embeddings), all_embeddings)
        all_default = torch.where(torch.isnan(all_default), torch.zeros_like(all_default), all_default)
        all_moe_rw = torch.where(torch.isnan(all_moe_rw), torch.zeros_like(all_moe_rw), all_moe_rw)
        
        all_embeddings = all_embeddings.cpu().numpy()
        all_default = all_default.cpu().numpy()
        all_moe_rw = all_moe_rw.cpu().numpy()
        
        return all_embeddings, all_default, all_moe_rw
