############################################################################
##
## Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
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
import torch.nn as nn

class binaryClassification(nn.Module):
    def __init__(self, cat_cards, numer_dims=10):
        """
        cat_cards (list): list of integers, where each integer is the cardinality of the SORTED column names
        numer_dims (int): number of numerical dimensions
        """
        super(binaryClassification, self).__init__()
        self.num_cats = len(cat_cards)
        
#         Uncomment lines below to enable categorical embeddings as well as line in forward method
#         self.embeddings = nn.ModuleList([nn.Embedding(cat_card, min(50, cat_card//2 + 1)) for 
#                                          cat_card in cat_cards])
        
#         total_embed_dims = sum(i.embedding_dim for i in self.embeddings)
        
        # Number of input features is X_train.shape[1].
#         self.layer_1 = nn.Linear(numer_dims + total_embed_dims, 512)

        self.layer_1 = nn.Linear(self.num_cats +  numer_dims, 512)
        self.layer_2 = nn.Linear(512, 512)
        self.layer_3 = nn.Linear(512, 512)
        self.layer_4 = nn.Linear(512, 512)
        self.layer_5 = nn.Linear(512, 512)
        self.layer_out = nn.Linear(512, 1) 
        
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)


        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(512)
        
    def forward(self, cat_inputs, numer_inputs):
                
#         inputs = torch.cat([self.embeddings[col](cat_inputs[:, col]) for col in range(self.num_cats)]+[numer_inputs], dim=1)
        
        inputs = torch.cat([cat_inputs, numer_inputs], dim=1)
        
        x = self.prelu1(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.prelu2(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.prelu3(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x = self.prelu4(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.prelu5(self.layer_5(x))
        x = self.batchnorm5(x)
        x = self.layer_out(x)
        return x
