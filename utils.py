import itertools
import json
import os
import random
import numpy as np
from typing import List
import wandb
import torch
def multiply_head_with_embedding(prediction_head, embeddings):
    return prediction_head.matmul(embeddings.transpose(-1, -2))
def bce_loss(pos_logits:torch.Tensor, neg_logits:torch.Tensor, mask:torch.Tensor, epsilon=1e-10):
    loss = torch.log(1. + torch.exp(-pos_logits) + epsilon) + torch.log(1. + torch.exp(neg_logits) + epsilon).mean(-1, keepdim=True)
    return (loss * mask.unsqueeze(-1)).sum() / mask.sum().clamp(0.0000005)
def _uniform_negatives(num_items, shape):
    return np.random.randint(1, num_items+1, shape)

def _uniform_negatives_session_rejected(num_items, shape, in_session_items):
        negatives = []
        for _ in range(np.prod(shape)):
            negative = np.random.randint(1, num_items+1)
            while negative in in_session_items:
                negative = np.random.randint(1, num_items+1)
            negatives.append(negative)
        return np.array(negatives).reshape(shape)
def sample_uniform(num_items, shape, in_session_items, is_session_item_excluded):
    if is_session_item_excluded:
        return _uniform_negatives_session_rejected(num_items, shape, in_session_items)
    else: 
        return _uniform_negatives(num_items, shape)
def _infer_shape(session_len, num_uniform_negatives, sampling_style):
    if sampling_style=="eventwise":
        return [session_len, num_uniform_negatives]
    elif sampling_style=="sessionwise":
        return [num_uniform_negatives]
    else:
        return []
def sample_uniform_negatives_with_shape(pos, num_items, session_len, num_uniform_negatives, sampling_style, reject_session_items):
    """
        Only used for eventwise and sessionwise cases.
        For batchwise, call sample uniform directly since positives are taken for the whole batch,
        which is only known during collating
    """
    shape = _infer_shape(session_len, num_uniform_negatives, sampling_style)
    if shape:
        # Avoid having to create set(clicks) without using it
        if reject_session_items:
            negatives = sample_uniform(num_items, shape, set(pos), reject_session_items)
        else:
            negatives= sample_uniform(num_items, shape, None, reject_session_items)
    else: 
        negatives = np.array([])
    return negatives
def sample_in_batch_negatives(batch_positives, num_in_batch_negatives, batch_session_len:List[int], reject_session_items):
    """
    
    """
    in_batch_negatives = []
    positive_indices = itertools.accumulate(batch_session_len)
    positive_indices = [0] + [p for p in positive_indices]
    if reject_session_items:
        for i in range(len(positive_indices)-1):
            candidate_positives = batch_positives[:positive_indices[i]] + batch_positives[
                positive_indices[i + 1]:]
            in_batch_negatives.append(random.sample(candidate_positives, num_in_batch_negatives))
    else:
        for i in range(len(batch_session_len)):
            in_batch_negatives.append(random.sample(batch_positives, num_in_batch_negatives))
    return in_batch_negatives
from typing import List,Tuple
def evaluate_recall(groundtruth:List[Tuple[int]], predicted:List[List[Tuple[int]]], k:int):
    recall=0
    length=len(groundtruth)
    assert length!=0
    assert length==len(predicted)
    for i in range(length):
        if groundtruth[i] in predicted[i][:k]:
            recall+=1
    return recall/length
def evaluate_MRR(groundtruth:List[Tuple[int]], predicted:List[List[Tuple[int]]], k:int):
    length=len(groundtruth)
    assert length!=0
    assert length==len(predicted)
    for i in range(length):
        if groundtruth[i] in predicted[i][:k]:
            return 1/i
    return 0
def  calculate_ranks(logits, labels, cutoffs):
    num_logits = logits.shape[-1]
    k = min(num_logits, torch.max(cutoffs).item())
    # Indices represent k item IDs with the highest possibilty in DESC
    _,indices  = torch.topk(logits, k=int(k), dim=-1)
    # Indices represent k item IDs with the highest possibilty in ASC
    indices = torch.flip(indices, dims=[-1])
    # There could be only 1 hit since all item IDs are all uniques
    hits = indices == labels.unsqueeze(dim=-1)
    # Cum sum with sum returns the rank NO. with first one is 0
    ranks = torch.sum(torch.cumsum(hits, -1), -1) - 1.
    ranks[ranks == -1] = float('inf')
    return ranks

def pointwise_recall(ranks, cutoffs, mask):
    res = ranks < cutoffs.unsqueeze(-1).unsqueeze(-1)
    return res.float() * mask
def pointwise_mrr(ranks, cutoffs, mask):
    res = torch.where(ranks < cutoffs.unsqueeze(-1).unsqueeze(-1), ranks, float('inf'))
    return (1 / (res + 1)) * mask
def mean_metric(pointwise_metric, mask):
    hits = torch.sum(pointwise_metric, dim=(2, 1))
    # avoid division by 0
    return hits / torch.sum(mask).clamp(0.0000005)
def getNumBatch(dataset_len:int,batch_size:int,max_iter:int=-1):
    num_batch=int(dataset_len/batch_size)
    if max_iter==-1:
        return num_batch
    if num_batch<max_iter:
        print("Num_batch is smaller than args.max_iter. Using num_batch in place of max_iter")
    else:
        num_batch=max_iter
    return num_batch
def saveModel(model:torch.nn.Module,optimizer,train_dir:str,epoch,loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(train_dir,"last.pth")
    )
    print(f"Epoch {epoch} saved----------------")
def tryRestoreStateDict(model:torch.nn.Module,optimizer:torch.optim.Optimizer,train_dir:str,state_dict_path:str):
    epoch = 1
    loss=1
    print("train_dir",train_dir)
    if train_dir is not None:
        try:
            checkpoint = torch.load(os.path.join(train_dir,"last.pth"))
            print(1)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(2)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(3)
            epoch = checkpoint["epoch"]+1
            print(4)
            loss = checkpoint["loss"]
        except: 
            print('failed loading train_dir, pls check file path: ', end="")
            print(train_dir)
            raise NotImplementedError
        # finally:
        #     # return model,optimizer,epoch,loss
        #     pass
    else:
        print('no train_dir provided')
    return model,optimizer,epoch,loss
def jsonl_sample_func(result_queue,dataset,batch_size):
    def _sample():
        item=dataset[0]
        max_seqlen=dataset.max_seqlen
        pad=[0]*(max_seqlen-item["session_len"])
        e=pad+item["events"]
        l=pad+item["labels"]
        n=pad+[e[0] for e in item["negatives"]]
        return np.array(e),np.array(l),np.array(n)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(_sample())
        result_queue.put(zip(*one_batch))
