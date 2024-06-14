############################################################################
##
## Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
############################################################################

import torch
from typing import Optional,Tuple,List,Union, Dict

from tokenizers import AddedToken
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .tabular_tokenizer import TabularTokenizer


class HFTabularTokenizer(TabularTokenizer, PreTrainedTokenizerBase):
    def __init__(self, coder, special_tokens=None, delimiter=',', *args, **kwargs):
        TabularTokenizer.__init__(self, coder=coder, special_tokens=None, delimiter=',')
        PreTrainedTokenizerBase.__init__(self)
        self.pad_token = self.special_tokens_decoder[self.vocab_size - 1]
        self.convert_ids_to_tokens = self.decode
        self.convert_tokens_to_ids = self.encode
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

    def get_added_vocab(self, *args, **kwargs):
        pass

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Union[Tuple, Tuple[str]]:
        return ()

    def added_tokens_encoder(self) -> Dict[int, AddedToken]:
        return {}

    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        return {}

    def __call__(self, text_or_batch: Union[List[str], str], return_tensors=False, device=None, **kwargs):
        """creates a similar API to HF tokenizers. 
        ex.HF API:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
        """
        if isinstance(text_or_batch, str):
            token_ids = self.tokenize(text_or_batch)
        else:
            token_ids = []
            for item in text_or_batch:
                token_ids.append(self.tokenize(item))
            
        if return_tensors == 'pt':
            if not device:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            input_ids = torch.tensor([token_ids], device=device)
            if len(input_ids.shape) == 3:
                input_ids = input_ids.squeeze()
                
            return {'input_ids': input_ids,
                    'attention_mask': torch.ones_like(input_ids, device=device)
                    }
        elif return_tensors:
            raise ValueError('return_tensors parameter only accepts "pt" or Falsey input')
        return token_ids
