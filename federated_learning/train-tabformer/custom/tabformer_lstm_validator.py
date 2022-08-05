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
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from lstm_network import TabLSTM
from tabformer_dataset import TabformerDataset


class TabformerValidator(Executor):
    
    def __init__(self, dataset_base_dir, batch_size = 1024, validate_task_name=AppConstants.TASK_VALIDATION):
        super(TabformerValidator, self).__init__()

        self.dataset_base_dir = dataset_base_dir
        self._batch_size = batch_size
        self._validate_task_name = validate_task_name

    def _initialize_validator(self, fl_ctx: FLContext):

        # In order to use two different local datasets in POC mode we use the client_id to figure out which dataset needs to be trained on
        self.client_id = fl_ctx.get_identity_name()   #e.g. "site-1"

        # Data
        self._val_ds = TabformerDataset(self.dataset_base_dir, self.client_id, trainset=False)
        self._val_loader = DataLoader(self._val_ds, batch_size=self._batch_size, shuffle=False, drop_last=True)
        self._n_iterations = len(self._val_loader)

        self.model = TabLSTM()

        # Warning: this is specifically for POC mode with 2 clients training on 2 different GPUs on the same machine
        # Modify this section if you want to change GPU training behavior
        gpu_id = f'cuda:{int(self.client_id.split("-")[1]) - 1}'   #e.g. if client_id = "site-1" --> gpu_id = "cuda:0"
        self.device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _terminate_validator(self):
        # collect threads, close files here
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # the start and end of a run - only happen once
        if event_type == EventType.START_RUN:
            try:
                self._initialize_validator(fl_ctx)
            except BaseException as e:
                error_msg = f"Exception in _initialize_trainer: {e}"
                self.log_exception(fl_ctx, error_msg)
                self.system_panic(error_msg, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._terminate_validator()


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_acc': val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()

        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self._val_loader):
                if abort_signal.triggered:
                    return 0

                x_cat, x_cont, target = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                logits = self.model(x_cat, x_cont)

                pred_prob = torch.sigmoid(logits)
                pred_label = (pred_prob > 0.5).float()*1

                correct += (pred_label == target).sum().item()
                total += target.size()[0]

            metric = correct/float(total)

        return metric
