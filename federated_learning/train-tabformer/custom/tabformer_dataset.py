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

import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class TabformerDataset(Dataset):
    
    def __init__(self, fp, client_id, trainset=True):
    
        # read client dataframe
        if trainset:
          df = pd.read_csv(os.path.join(fp, f"{client_id}.csv"))
        else:
          df = pd.read_csv(os.path.join(fp, f"val-{client_id}.csv"))
        
        # read metadata dictionary file
        with open(os.path.join(fp, 'meta_data.pickle'), 'rb') as handle:
            self.meta_data = pickle.load(handle)
            
        amt_scaler = StandardScaler()
        df['amount'] = amt_scaler.fit_transform(df[['amount']])
        
        # cast error columns to uint8 instead of int64 - might not make a difference
        error_cols = ['errors_Bad CVV', 'errors_Bad Card Number', 'errors_Bad Expiration', 'errors_Bad PIN',
                   'errors_Bad Zipcode', 'errors_Insufficient Balance', 'errors_Technical Glitch']
        df[error_cols] = df[error_cols].astype('uint8')
        
        x_cat = df[self.meta_data['cat_cols']].values
        x_cont = df[self.meta_data['num_cols']].values
        y = df[self.meta_data['target_col']].values
        
        self.xcat = torch.tensor(x_cat, dtype=torch.long)
        self.xcont = torch.tensor(x_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
  
    def __getitem__(self,idx):
        return self.xcat[idx], self.xcont[idx], self.y[idx]
