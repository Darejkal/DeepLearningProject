import torch
from sasrec import DynamicPositionEmbedding
from utils import *
class SparseAttentionMaskFowardFunc(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input:torch.Tensor, weights:torch.Tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        attention_head=input
        ctx.save_for_backward(input, weights)
        return weights*attention_head
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs: here input and weights
        """
        input, weights = ctx.saved_tensors
        grad_input = weights.clone()*grad_output
        grad_weights = input.clone()*grad_output
        return grad_input, grad_weights
class SparseAttentionMask(torch.nn.Module):
    def __init__(self,max_len:int,hidden_size:int):
        self.weights=torch.ones(size=(max_len,hidden_size),dtype=torch.int)
        self.fn=SparseAttentionMaskFowardFunc.apply
        super().__init__()
    def foward(self,attention_head:torch.Tensor):
        return self.fn(attention_head,self.weights)

class DenoisedSasrec(torch.nn.Module):
    def __init__(self, item_num,max_len,hidden_size,dropout_rate,num_layers,sampling_style,device="cpu",share_embeddings=True,topk_sampling=False,topk_sampling_k=1000):
        super(DenoisedSasrec, self).__init__()
        self.hidden_size=hidden_size
        self.item_num = item_num
        self.share_embeddings=share_embeddings
        self.sampling_style=sampling_style
        self.topk_sampling=topk_sampling
        self.topk_sampling_k=topk_sampling_k
        self.merge_attn_mask = True
        self.device=device
        self.loss=bce_loss
        self.future_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1)
        self.register_buffer('future_mask_const', self.future_mask)
        self.register_buffer('seq_diag_const', ~torch.diag(torch.ones(max_len, dtype=torch.bool)))
        self.item_emb = torch.nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        if share_embeddings:
            self.output_emb = self.item_emb
        self.pos_emb = DynamicPositionEmbedding(max_len,hidden_size,device)
        self.input_dropout = torch.nn.Dropout(p=dropout_rate)
        self.last_layernorm = torch.nn.LayerNorm(hidden_size)
        encoder_layer=torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=1,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout_rate,
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder=torch.nn.TransformerEncoder(encoder_layer,num_layers,self.last_layernorm)
        # self.final_activation = torch.nn.ELU(0.5)
        self.final_activation = torch.nn.Identity()
        self.attnhead_mask = SparseAttentionMask(max_len=max_len,hidden_size=hidden_size)
        torch.nn.init.xavier_uniform_(self.item_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.pos_emb.embedding.weight.data)

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
    def forward(self, positives, mask): # for training   
        """
        mask: padding mask of 0 and 1
        returns attention_head
        """ 
        att_mask = self.merge_attn_masks(mask)    
        x = self.item_emb(positives)
        x = self.pos_emb(x)
        prediction_head = self.encoder(self.input_dropout(x), att_mask)
        return prediction_head
    def get_head_and_loss(self,batch):
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
        _,loss=self.get_head_and_loss(batch)
        logger.log("TRAIN",f"i: {iteration}, train_loss: {loss}", )
        loss.backward()
        optimizer.step()
        return loss

    def validate_step(self, batch, iteration,logger):
        prediction_head,loss=self.get_head_and_loss(batch)
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