# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabLSTM(nn.Module):
    def __init__(self, hidden_size=128):
        super(TabLSTM, self).__init__()
        
        # Explicitly specified for simplicity
        cat_cards = [9, 32, 2, 2, 2, 2, 2, 2, 2, 24, 892, 13429, 100343, 224, 60, 13, 10, 3, 30, 27322]    # cardinality of the SORTED categorical columns
        numer_dims = 1      # only 1 numeric column - amount
        
        self.num_cats = len(cat_cards)
        self.hidden_size = hidden_size
        self.embeddings = nn.ModuleList([nn.Embedding(cat_card, min(50, cat_card//2 + 1)) for
                                        cat_card in cat_cards])
        
        total_embed_dims = sum(i.embedding_dim for i in self.embeddings)
        input_size = total_embed_dims + numer_dims
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        

    def forward(self, cat_inputs, numer_inputs):
        
        inputs = torch.cat([self.embeddings[col](cat_inputs[:, col]) 
                            for col in range(self.num_cats)] + [numer_inputs], dim=1)
        batch, d = inputs.shape
        
        x, _ = self.lstm(inputs.reshape(batch, 1, d))
        x = self.linear(x.reshape(batch, self.hidden_size))

        return x
