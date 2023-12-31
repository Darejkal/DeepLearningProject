from typing import Any, Dict
import torch
from sasrec import DynamicPositionEmbedding
import torch.nn.functional as F
from utils import *
import itertools
from typing import Iterator
class OffsetScale(torch.nn.Module):
    def __init__(self, dim, heads = 1,device="cpu"):
        super(OffsetScale,self).__init__()
        self.device=device
        self.gamma = torch.nn.Parameter(torch.ones(heads, dim,device=device))
        self.beta = torch.nn.Parameter(torch.zeros(heads, dim,device=device))
        torch.nn.init.normal_(self.gamma, std = 0.02)
        self.to(device)
    def forward(self, x):
        out = torch.einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)
def _projection_on_C(y:torch.Tensor,beta:float):
    return (y-beta).clamp(0,1)
def _equation_on_C(y:torch.Tensor,beta:float,B=0.6):
    return (y-beta).clamp(0,1).sum()-B
def _find_beta_on_C(y:torch.Tensor,init_beta:float=1,init_steps=1,delta=1e-5):
    val=_equation_on_C(y,init_beta)
    if val>0:
        pos_beta=init_beta
        lv=val
        while val>0:
            init_beta+=init_steps
            val=_equation_on_C(y,init_beta)
            init_steps*=2
        neg_beta=init_beta
        rv=val
    else:
        neg_beta=init_beta
        rv=val
        while val<0:
            init_beta-=init_steps
            val=_equation_on_C(y,init_beta)
            init_steps*=2
        pos_beta=init_beta
        lv=val
    while(lv-rv>delta):
        mid_beta=(pos_beta+neg_beta)/2
        mv=_equation_on_C(y,mid_beta)
        if (mv>0):
            lv=mv 
            pos_beta=mid_beta
        elif (mv<0):
            rv=mv 
            neg_beta=mid_beta    
        else:
            return mid_beta
    return pos_beta
# class ProjectedGD(torch.optim.Optimizer):
#     def __init__(self, params: params_t,projection_func=_projection_on_C) -> None:
#         defaults={"projection_func":projection_func}
#         super(ProjectedGD,self).__init__(params, defaults)
#     def step():
        
# activation functions

class ReLUSquared(torch.nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2
def sample_gumbel(shape,device="cpu", eps=1e-20):
    U = torch.rand(shape,device=device)
    return -torch.log(-torch.log(U + eps) + eps)
# class SparseAttentionMaskFowardFunc(torch.autograd.Function):
#     """Both forward and backward are static methods."""
#     @staticmethod
#     def forward(ctx, input:torch.Tensor, weights:torch.Tensor):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input, weights)
#         mask=torch.sigmoid((torch.log(weights/(1-weights))+sample_gumbel(weights.size()))/0.1)
#         return mask*input
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the inputs: here input and weights
#         """
#         input, weights = ctx.saved_tensors
#         grad_input = weights.clone()*grad_output
#         grad_weights = input.clone()*grad_output
#         return grad_input, grad_weights
class SparseAttentionMask(torch.nn.Module):
    def __init__(self,max_len:int,hidden_size:int,device="cpu"):
        super(SparseAttentionMask,self).__init__()
        self.device=device
        _weights=torch.ones(size=(max_len,hidden_size),dtype=torch.float,device=device)/2
        self.weights=torch.nn.Parameter(_weights)
        # self.fn=SparseAttentionMaskFowardFunc.apply
    def forward(self,attention_head:torch.Tensor,temperature:float=0.2):
        # return self.fn(attention_head,self.weights)
        mask=torch.sigmoid((torch.log(self.weights/(1-self.weights))+sample_gumbel(self.weights.size()))/temperature)
        return mask*attention_head
        
class DenoisedSasrec(torch.nn.Module):
    def __init__(self, item_num,max_len,hidden_size,dropout_rate,num_layers,sampling_style,device="cpu",share_embeddings=True,topk_sampling=False,topk_sampling_k=1000):
        # dropout rate is ignored   
        super(DenoisedSasrec, self).__init__()  
        self.device=device
        self.item_num=item_num
        self.max_len=max_len
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.sampling_style=sampling_style
        self.topk_sampling=topk_sampling
        self.topk_sampling_k=topk_sampling_k
        self.device=device
        self.share_embeddings=share_embeddings
        self.loss=bce_loss
        self.item_emb=torch.nn.Embedding(item_num+1,hidden_size,device=device)
        self.pos_emb = DynamicPositionEmbedding(max_len,hidden_size,device)
        self.X_to_Z=torch.nn.Sequential(
            torch.nn.Linear(max_len,max_len,bias=False,device=device),
            torch.nn.SiLU()
        )
        self.X_to_V=torch.nn.Sequential(
            torch.nn.Linear(max_len,max_len,bias=False,device=device),
            torch.nn.SiLU()
        )
        self.Z_to_Q=torch.nn.Sequential(
            torch.nn.Linear(max_len,max_len,bias=False,device=device),
            OffsetScale(max_len,num_layers,device)
        )
        self.Z_to_K=torch.nn.Sequential(
            torch.nn.Linear(max_len,max_len,bias=False,device=device),
            OffsetScale(max_len,num_layers,device)
        )
        self.sparse_mask=SparseAttentionMask(max_len,hidden_size)
        self.relu_squared=ReLUSquared()
        self.future_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1)
        self.register_buffer('future_mask_const', self.future_mask)
        self.register_buffer('seq_diag_const', ~torch.diag(torch.ones(max_len, dtype=torch.bool)))
        self.final_activation=torch.nn.Identity()
        self.merge_attn_mask = True
        if share_embeddings:
            self.output_emb = self.item_emb
        else: self.output_emb=torch.nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
    def notmask_parameters(self):
        layers=[self.item_emb,self.pos_emb,self.X_to_Z,self.X_to_V,self.Z_to_Q,self.Z_to_K,self.relu_squared,self.final_activation]
        if not self.share_embeddings:
            layers.append(self.output_emb)
        def getParameters(x:torch.nn.Module):
            return x.parameters()
        result= itertools.chain(*list(map(getParameters,layers)))
        return result
    def ismask_parameters(self):
        return self.sparse_mask.parameters()
    def optimize_sparse_mask(self,loss_grad,lr=0.1):
        y=self.sparse_mask.weights-lr*loss_grad
        alpha=max(0,_find_beta_on_C(torch.flatten(y)))
        self.sparse_mask.weights=torch.clamp(y-alpha,0,1)
    def merge_attn_masks(self, padding_mask):
        """
        padding_mask: 0 if padded and 1 if comes from the source sequence
        
        Returns a mask of size (batch,maxseq,maxseq) where True means masked (not allowed to attend) and False means otherwise.
        """
        batch_size = padding_mask.shape[0]
        seq_len = padding_mask.shape[1]

        if not self.merge_attn_mask:
            return self.future_mask_const[:seq_len, :seq_len]

        padding_mask_broadcast = ~padding_mask.bool().unsqueeze(1)
        future_masks = torch.tile(self.future_mask_const[:seq_len, :seq_len], (batch_size, 1, 1))
        merged_masks = torch.logical_or(padding_mask_broadcast, future_masks)
        diag_masks = torch.tile(self.seq_diag_const[:seq_len, :seq_len], (batch_size, 1, 1))
        return torch.logical_and(diag_masks, merged_masks)
    def encoder(self,x:torch.Tensor,attn_mask:torch.Tensor):
        z=self.X_to_Z(x)
        a=self.Z_to_Q(z)[0]@torch.transpose(self.Z_to_K(z)[0],-2,-1)+attn_mask
        a=self.sparse_mask(a*attn_mask.logical_not().int())
        a=self.relu_squared(a)/(self.max_len*self.hidden_size)
        return a@self.X_to_V(x)
    def forward(self, positives, mask): # for training   
        """
        mask: padding mask of 0 and 1
        returns attention_head
        """ 
        attn_mask = self.merge_attn_masks(mask)    
        x = self.item_emb(positives)
        x = self.pos_emb(x)
        prediction_head = self.encoder(x, attn_mask)
        return prediction_head
    def _getTrainHeadAndLoss(self,batch):
        prediction_head = self.forward(batch["positives"],batch["mask"])
        
        pos = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                   self.output_emb(batch["labels"]).unsqueeze(-2)).squeeze(-1)

        if self.sampling_style == "eventwise":
            uniform_negative_logits = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                                self.output_emb(batch["uniform_negatives"])).squeeze(-2)
        else:
            uniform_negative_logits = multiply_head_with_embedding(prediction_head, self.output_emb(batch["uniform_negatives"]))

        in_batch_negative_logits = multiply_head_with_embedding(prediction_head, self.output_emb(batch["in_batch_negatives"]))
        neg = torch.concat([uniform_negative_logits, in_batch_negative_logits], dim=-1)
        if self.topk_sampling:
            neg, _ = torch.topk(neg, k=self.topk_sampling_k, dim=-1)
        pos_scores, neg_scores = self.final_activation(pos), self.final_activation(neg)
        loss=self.loss(pos_scores,neg_scores,batch["mask"])
        return prediction_head,loss
    def train_step(self, batch, iteration,optimizer:torch.optim.Optimizer,logger):
        optimizer.zero_grad()
        _,loss=self._getTrainHeadAndLoss(batch)
        logger.log("TRAIN",f"i: {iteration}, train_loss: {loss}", )
        loss.retain_grad()
        loss.backward()
        loss_grad=loss.grad
        assert loss_grad!=None
        optimizer.step()
        self.optimize_sparse_mask(loss_grad)
        return loss

    def validate_step(self, batch, iteration,logger):
        prediction_head,loss=self._getTrainHeadAndLoss(batch)
        # score:
        cut_offs = torch.tensor([5, 10, 20], device=self.device)
        recalls,mrrs=[],[]
        for t in range(prediction_head.shape[1]):
            mask = batch['mask'][:, t]
            positives = batch['labels'][:, t]
            logits = multiply_head_with_embedding(prediction_head[:, t], self.output_emb.weight)
            logits[:, 0] = -torch.inf  # set score for padding item to -inf
            ranks = calculate_ranks(logits, positives, cut_offs)
            pw_rec = pointwise_recall(ranks, cut_offs, mask)
            recalls.append(pw_rec.squeeze(dim=1))
            pw_mrr = pointwise_mrr(ranks, cut_offs, mask)
            mrrs.append(pw_mrr.squeeze(dim=1))
        pw_rec = torch.stack(recalls, dim=2)
        pw_mrr = torch.stack(mrrs, dim=2)
        recall,mrr= mean_metric(pw_rec, batch["mask"]), mean_metric(pw_mrr, batch["mask"])
        count=0
        for i, k in enumerate(cut_offs.tolist()):
            logger.log(f"EVALUATE_{iteration}",f'recall_cutoff_{k}= {recall[i]}',True )
            logger.log(f"EVALUATE_{iteration}",f'mrr_cutoff_{k}={mrr[i]}',True)
            count+=1
        logger.log(f"EVALUATE_LOSS_{iteration}",f"loss={loss}")
        return torch.sum(recall)/count,torch.sum(mrr)/count,loss
