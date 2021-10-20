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

###########################################################################
# Usage: python3 scripts/multi-gpu-linkpred.py --num-epochs=5 --num-workers=12 --gpu=0,1,2,3

###########################################################################
import os
import cudf
import numpy as np

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import auroc
from torch.utils.dlpack import from_dlpack
import argparse
import time
import dgl.multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm 
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
import dgl.nn as dglnn

    
BASE_DIR = "./basedir"
processed_path = os.path.join(BASE_DIR, "processed_data_mgpu")

# Ensure BASE_DIR exists
os.makedirs(BASE_DIR, exist_ok=True)

#####################################################################################

# The GNN we will be using is basic 2-layer GNN GraphSage model and since we have a heterogenous graph we will use the `HeterGraphConv` layer in DGL to implement a Heterogenous GraphSage. We use the mean aggregation in GraphSage. Other supported aggregations include `max`, `lstm` and `gcn`. We also use dropout after the first GNN layer and define the GNN as follows: 
class HeteroGraphSage(nn.Module):
    def __init__(self, n_input, n_hidden, n_out, n_layers, rel_names, agg='mean', dout=0.5, activation=F.relu):
        super().__init__()
      
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.agg = agg
        self.dout = dout
  
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(self.n_input, self.n_hidden,
                                aggregator_type=self.agg,
                                feat_drop=self.dout,
                                activation=self.activation)
            for rel in rel_names}))

            
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.SAGEConv(n_hidden, n_hidden,
                                    aggregator_type=self.agg,
                                    feat_drop=self.dout,
                                    activation=self.activation)
                for rel in rel_names}))

        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(self.n_hidden, self.n_out,
                                aggregator_type=self.agg)
            for rel in rel_names}))

    def forward(self, blocks, x):
        for layer, block in (zip(self.layers, blocks)):
            x = layer(block, x)
        return x
    
##################################################################################   

# To compute the score for a potential edge we take the dot product of the representations of its incident nodes.  

# This can be easily accomplished with `apply_edges` method.

class ScorePredictor(nn.Module):

    def __init__(self, score_etype):
        super().__init__()
        self.score_etype = score_etype
        
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            edge_subgraph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=self.score_etype)
            score = edge_subgraph.edata['score'][self.score_etype].squeeze(1)
            return score
        
#####################################################################################
# We now define the `LinkPredictor` model made up of GNN to compute the node representations and score predictor to score th edges.
class LinkPredictor(nn.Module):

    def __init__(self, gnn, link_pred_etype):
        super().__init__()
        self.gnn = gnn
        self.pred = ScorePredictor(link_pred_etype)

    def forward(self, positive_graph, negative_graph, blocks, x):
        h = self.gnn(blocks, x)
        pos_scores = self.pred(positive_graph, h)
        neg_scores = self.pred(negative_graph, h)
        return pos_scores, neg_scores
    
# #################################################################################### GNNs require node features as input to generate the output node representations. In this dataset while we do have some merchant attributes like `mcc`, `city`, `state`, etc., we don't have any features for `card` nodes. So we will pretend we don't have features for either node type and juste learn the node features from scratch as part of the GNN through the [`dgl.nn.sparse_emb.NodeEmbedding`](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html?highlight=embedding#dgl.nn.pytorch.sparse_emb.NodeEmbedding) class. 
# ​ The class is optimized for training large-scale node embeddings. It updates the embedding in a sparse way and can scale to graphs with millions of nodes. It also supports partitioning to multiple GPUs (on a single machine) for more acceleration. 
# ​ Currently, DGL provides two optimizers that work with this NodeEmbedding class: `SparseAdagrad` and `SparseAdam`.
# ​ The implementation is based on torch.distributed package. It depends on the pytorch default distributed process group to collect multi-process information and uses `torch.distributed.TCPStore` to share meta-data information across multiple gpu processes.  


def initializer(emb):
    nn.init.xavier_uniform_(emb)
    return emb


class DGLNodeEmbed(nn.Module):
    def __init__(self, num_nodes_dict, embed_size, device):
        super(DGLNodeEmbed, self).__init__()
        self.embed_size = embed_size
        self.node_embeds = {}
        self.device = device
        for ntype in num_nodes_dict:
            node_embed = dglnn.NodeEmbedding(num_embeddings=num_nodes_dict[ntype],
                                              embedding_dim=self.embed_size,
                                              name=str(ntype)+"_embed",
                                              init_func=initializer,
                                              device=self.device)
            self.node_embeds[ntype] = node_embed
    
    @property
    def dgl_embds(self):
        embds = [emb for emb in self.node_embeds.values()]
        return embds
    
    
    def forward(self, node_ids_dict):
        embeds = {}
        for ntype in node_ids_dict.keys():
            embeds[ntype] = self.node_embeds[ntype](node_ids_dict[ntype], device=self.device)
        return embeds
    
###################################################################################
# Since we are trying to solve binary classification task we can use the binary cross-entropy loss along with AUC (Area under ROC Curve) as the evaluation metric. Let's define those two.    
    
def bce_loss(pos_scores, neg_scores, labels):
    logits = torch.cat([pos_scores, neg_scores])
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def compute_acc(pos_score, neg_score, labels, thresh=0.5):
    probs = torch.sigmoid(torch.cat([pos_score, neg_score]))
    preds = (probs > thresh).int()
    acc = accuracy(preds, labels)
    return acc

def compute_auc(pos_scores, neg_scores, labels):
    logits = torch.cat([pos_scores, neg_scores])
    preds = torch.sigmoid(logits)
    return auroc(preds, labels, pos_label=1)

#####################################################################################
def evaluate(model, embed, dataloader, device):

    epoch_pos_scores = []
    epoch_neg_scores = []


    for input_nodes, pos_graph, neg_graph, blocks in tqdm(dataloader):
        blocks = [b.to(device) for b in blocks]
        for ntype, node_ids in input_nodes.items():
             input_nodes[ntype] = node_ids.to(device)
        input_feats = embed(input_nodes)             
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)
        pos_score, neg_score = model(pos_graph, neg_graph, blocks, input_feats)

        pos_score, neg_score = pos_score.detach(), neg_score.detach()
        epoch_pos_scores.append(pos_score)
        epoch_neg_scores.append(neg_score)

    epoch_pos_scores = torch.cat(epoch_pos_scores)
    epoch_neg_scores = torch.cat(epoch_neg_scores)
    epoch_labels = torch.cat([torch.ones_like(epoch_pos_scores, device=device),
                                torch.zeros_like(epoch_neg_scores, device=device)]).int()
    epoch_auc = compute_auc(epoch_pos_scores, epoch_neg_scores, epoch_labels).item()
    
    return epoch_auc

#####################################################################################
# The following training function performs the forward pass each minibatch, performs the backward pass, keeps a tally of the all the positive scores, negative scores to calculate the epoch AUC and then finally returns the epoch loss and epoch AUC.

def train(model, embed, optimizers, loss_func, dataloader, device):
    
    running_loss = 0
    tot_n_edges = 0
    epoch_pos_scores = []
    epoch_neg_scores = []
    model_lr = 1e-2
    opt = torch.optim.SGD(model.parameters(), lr=model_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt,
                                                  base_lr=1e-2,
                                                  max_lr=7e-1,
                                                  mode='triangular')

    for input_nodes, pos_graph, neg_graph, blocks in tqdm(dataloader):
        input_feats = embed(input_nodes)             
        pos_score, neg_score = model(pos_graph, neg_graph, blocks, input_feats)

        batch_labels = torch.cat([torch.ones_like(pos_score, device=device),
                                  torch.zeros_like(neg_score, device=device)])
        loss = loss_func(pos_score, neg_score, batch_labels)

        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()
        scheduler.step()
        
        batch_size = len(pos_score) + len(neg_score)
        running_loss += loss.item() * batch_size
        tot_n_edges += batch_size

        pos_score, neg_score = pos_score.detach(), neg_score.detach()
        epoch_pos_scores.append(pos_score)
        epoch_neg_scores.append(neg_score)
    
    epoch_loss = running_loss / tot_n_edges 
    epoch_pos_scores = torch.cat(epoch_pos_scores)
    epoch_neg_scores = torch.cat(epoch_neg_scores)
    epoch_labels = torch.cat([torch.ones_like(epoch_pos_scores, device=device),
                                torch.zeros_like(epoch_neg_scores, device=device)]).int()
    epoch_auc = compute_auc(epoch_pos_scores, epoch_neg_scores, epoch_labels).item()
    
    return epoch_loss, epoch_auc
#####################################################################################

# To setup multi-GPU training, we implement a few differences than the single GPU version of this example. First, we have to start up distributed training. We do this initializing a distributed context. 
def run(proc_id, n_gpus, args, devices):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    device = torch.device('cuda:{:d}'.format(proc_id))
    torch.cuda.set_device(device)

    # Load graph to device
    g = dgl.load_graphs('./link_pred_graph.bin')[0][0]
    g.to(device)
    
    SEED = 42
    rng = np.random.default_rng(seed=SEED)
    num_edges = g.num_edges(etype='transaction')
    rand_idx = rng.permutation(num_edges)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    train_edges, val_edges, test_edges = np.split(rand_idx,
                                                  [int(train_ratio * num_edges), int((train_ratio + val_ratio) * num_edges)])
    # g_train only includes train edges
    g_train = dgl.edge_subgraph(graph=g,
                                edges={'transaction' : train_edges,
                                      'transaction-rev': train_edges},
                                relabel_nodes=False)



    train_val_edges = np.concatenate((train_edges, val_edges))

    # g_train_val includes train and val edges
    g_train_val = dgl.edge_subgraph(graph=g,
                                edges={'transaction' : train_val_edges,
                                      'transaction-rev': train_val_edges},
                                relabel_nodes=False)


    median_card_in_deg = torch.median(g.in_degrees(etype='transaction-rev')).item()

    median_merch_in_deg = torch.median(g.in_degrees(etype='transaction')).item()

    num_layers = 2
    neigh_sampler = dgl.dataloading.MultiLayerNeighborSampler([
         {('card', 'transaction', 'merchant'): median_merch_in_deg,
         ('merchant', 'transaction-rev', 'card'): median_card_in_deg}] * num_layers)

    neg_sampling_k = 1
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(neg_sampling_k)

    # specify batch size
    batch_size = 8192
    
# Once we've initialized the distributed context, we need to wrap our model with Distributed Data Parallel. We should also add enable ddp to work with our dataloader. To do this we set use_ddp=n_gpus > 1 below.
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        g=g_train, 
        eids={'transaction' : torch.arange(g_train.num_edges(etype='transaction'))},
        block_sampler=neigh_sampler,
        exclude='reverse_types',
        reverse_etypes={'transaction': 'transaction-rev',
                        'transaction-rev': 'transaction'},
        negative_sampler=neg_sampler,
        batch_size=batch_size,
        device=device,
        use_ddp=n_gpus > 1,
        shuffle=True, # pytorch dataloader arguments
        drop_last=False, #
        num_workers=args.num_workers) #
    
    val_eid_dict = {'transaction' : val_edges}

    val_dataloader = dgl.dataloading.EdgeDataLoader(
        g=g,
        eids=val_eid_dict,
        block_sampler=neigh_sampler,
        g_sampling=g_train,
        negative_sampler=neg_sampler,
        batch_size=batch_size,
        device=device,
        use_ddp=n_gpus > 1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)
    
    test_eid_dict = {'transaction' : test_edges}
    full_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers=num_layers)
    test_dataloader = dgl.dataloading.EdgeDataLoader(
        g=g,
        eids=test_eid_dict,
        block_sampler=full_sampler,
        device=device,
        g_sampling=g_train_val,
        negative_sampler=neg_sampler,
        batch_size=batch_size,
        use_ddp=n_gpus > 1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)
    
    # Define model and optimizer
    input_feat_dim = 64
    hidden_dim = 64
    out_dim = 64
    num_nodes_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    emb_lr = 1e-2
    num_nodes_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    embed = DGLNodeEmbed(num_nodes_dict, input_feat_dim, device=device)
    emb_opt = dgl.optim.SparseAdam(embed.dgl_embds, lr=emb_lr)
    embed.node_embeds['card'].weight
    embed.node_embeds['merchant'].weight
    
    gnn = HeteroGraphSage(input_feat_dim, hidden_dim, out_dim, num_layers, g.etypes)
    # we only perform link pred on transaction etype 
    link_pred_etype = ('card', 'transaction', 'merchant')
    model = LinkPredictor(gnn, link_pred_etype)
    # send model to device
    model = model.to(device)
    # set model intial learning rate
    model_lr = 1e-2
    opt = torch.optim.SGD(model.parameters(), lr=model_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt,
                                                  base_lr=1e-2,
                                                  max_lr=7e-1,
                                                  mode='triangular')
    
    #num_epochs = 5
    optimizers = [opt, emb_opt]
  

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        if n_gpus > 1:
            train_dataloader.set_epoch(epoch)
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        model.train()
        train_loss, train_auc = train(model, embed, optimizers, bce_loss, train_dataloader, device) 

        model.eval()
        with torch.no_grad():
            val_auc = evaluate(model, embed, val_dataloader, device)
            print('Epoch {:05d} | Loss {:.4f} | Train Auc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(epoch, train_loss, train_auc, np.mean(iter_tput[3:]), torch.cuda.max_memory_allocated() / 1000000))

        if n_gpus > 1:
            torch.distributed.barrier()

        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                if n_gpus == 1:
                    eval_acc = evaluate(model, embed, val_dataloader, devices[0])
                    test_acc = evaluate(model, embed, test_dataloader, devices[0])
                else:
                    eval_acc = evaluate(model, embed, val_dataloader)
                    test_acc = evaluate(model, embed, test_dataloader)
                print('Eval Acc {:.4f}'.format(eval_acc))
                print('Test Acc: {:.4f}'.format(test_acc))
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # Construct graph
     # Load graph to device
    g = dgl.load_graphs('./link_pred_graph.bin')[0][0]
    
    # Normally, DGL maintains only one sparse matrix representation (usually COO) for each graph, and will create new formats when some APIs are called for efficiency.
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    
    g.create_formats_()
    
    
# DGL takes advantage of the native multiprocessing module in Python. So we spawn multiple subprocesses this way.

# For our example, we decided to spawn the multiple processes in such a way that forks the main process and shares the same graph object in the training process. This avoids duplicate copies of the graph on each GPU which helps us reduce the memory consumption. Once we're able to have the graph on the GPU in DGL, this should be really helpful.

    if n_gpus == 1:
        run(0, n_gpus, args, devices)
    else:
        procs = []
        for proc_id in range(n_gpus):
#             mp.set_start_method('spawn')
            ctx = mp.get_context('spawn')
            p = ctx.Process(target=run, args=(proc_id, n_gpus, args,devices))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        
