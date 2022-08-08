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
import os.path

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.event_type import EventType
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants
#from simple_network import SimpleNetwork

from lstm_network import TabLSTM
from tabformer_dataset import TabformerDataset


class TabformerTrainer(Executor):

    def __init__(self, dataset_base_dir, lr=0.001, epochs=5, batch_size=1024, train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None):
        """Trainer handles train and submit_model tasks. During train_task, it trains a
        PyTorch network on TabFormer dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            dataset_base_dir (str): Path to where local datasets are stored. Expect two files inside: "site-1.csv" and "site-2.csv".
            lr (float, optional): Learning rate. Defaults to 0.001
            epochs (int, optional): Local epochs for each client. Defaults to 5.
            batch_size (int, optional): Training batch size. Defaults to 1024.
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
        """
        super(TabformerTrainer, self).__init__()

        self.dataset_base_dir = dataset_base_dir
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars


    def _initialize_trainer(self, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # In order to use two different local datasets in POC mode we use the client_id to figure out which dataset needs to be trained on
        self.client_id = fl_ctx.get_identity_name()   #e.g. "site-1"

        # Data
        self._train_ds = TabformerDataset(self.dataset_base_dir, self.client_id)
        self._train_loader = DataLoader(self._train_ds, batch_size=self._batch_size, shuffle=True, drop_last=True)
        self._n_iterations = len(self._train_loader)

        # Training setup
        self.model = TabLSTM()

        # Warning: this is specifically for POC mode with 2 clients training on 2 different GPUs
        gpu_id = f'cuda:{int(self.client_id.split("-")[1]) - 1}'
        self.device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss and Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self._lr)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)
    
    def _terminate_trainer(self):
        # collect threads, close files here
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # the start and end of a run - only happen once
        if event_type == EventType.START_RUN:
            try:
                self._initialize_trainer(fl_ctx)
            except BaseException as e:
                error_msg = f"Exception in _initialize_trainer: {e}"
                self.log_exception(fl_ctx, error_msg)
                self.system_panic(error_msg, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._terminate_trainer()


    def local_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, batch in enumerate(self._train_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                x_cat, x_cont, target = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(x_cat, x_cont)
                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    self.log_info(fl_ctx, f"Epoch: {epoch+1}/{self._epochs}, Iteration: {i}, "
                                          f"Loss: {running_loss/2000:4f}")
                    running_loss = 0.0

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
