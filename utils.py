import itertools
import json
import os
import random
import numpy as np
from typing import List

import torch
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
def calculate_ranks(logits, labels, cutoffs):
    num_logits = logits.shape[-1]
    k = min(num_logits, torch.max(cutoffs).item())
    indices,_  = torch.topk(logits, k=int(k), dim=-1)
    indices = torch.flip(indices, dims=[-1])
    hits = indices == labels.unsqueeze(dim=-1)
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
def saveModel(model:torch.nn.Module,epoch,train_dir:str):
    with open(os.path.join(train_dir, 'save.json'), 'w') as f:
        f.write(json.dumps({"epoch":epoch}))
    torch.save(model.state_dict(), os.path.join(train_dir, "latest.pth"))
    print(f"Epoch {epoch} saved----------------")
def tryRestoreStateDict(model:torch.nn.Module,device:str,train_dir:str,state_dict_path:str):
    model.to(device)
    model.train()
    epoch_start_idx = 1
    print("state_dict_path",state_dict_path)
    if state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))
            epoch=1
            with open(os.path.join(train_dir, 'save.json'),"r") as f:
                epoch=int(json.loads(next(f))["epoch"])
            epoch_start_idx= epoch
        except: 
            print('failed loading state_dicts, pls check file path: ', end="")
            print(state_dict_path)
        finally:
            # in case of jupyter notebook => will train model as new.
            return model,epoch_start_idx
    else:
        print('no state_dict_path provided')
    return model,epoch_start_idx
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