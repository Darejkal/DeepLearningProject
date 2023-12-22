import torch
import numpy as np
from utils import bce_loss, calculate_ranks, mean_metric, multiply_head_with_embedding, pointwise_mrr, pointwise_recall
def apply_rotary_position_embeddings(sinusoidal_pos, layers):
        # sin [batch_size, sequence_length, embed_size//2]
        # cos [batch_size, sequence_length, embed_size//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.repeat_interleave(sin, 2, dim=-1)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.repeat_interleave(cos, 2, dim=-1)
        outlayers=[]
        for layer in layers:
            # rotate_half_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
            rotate_half_layer = torch.stack([-layer[..., 1::2], layer[..., ::2]], dim=-1).reshape_as(
                layer
            )
            outlayers.append(layer * cos_pos + rotate_half_layer * sin_pos)
        return outlayers
class DynamicPositionEmbeddingWithRotaty(torch.nn.Module):

    def __init__(self, max_len, hidden_size,device='cpu'):
        super(DynamicPositionEmbeddingWithRotaty, self).__init__()
        self.device=device
        self.hidden_size=hidden_size
        self.max_len = max_len
        self.embedding = torch.nn.Embedding(max_len, hidden_size,device=device)
        self.pos_indices = torch.arange(0, self.max_len, dtype=torch.int,device=device)
        self.register_buffer('pos_indices_const', self.pos_indices)
    def forward(self, x:torch.Tensor,times):
        seq_len = x.shape[1]
        assert(seq_len%2==0)
        batch_size = x.shape[0]
        time_intervals=torch.concat([torch.zeros((batch_size,1)),torch.diff(times)],dim=-1).reshape((batch_size,int(seq_len/2),2))
        time_intervals=torch.mean(time_intervals,-1)
        return self.embedding(self.pos_indices_const[-seq_len:])+apply_rotary_position_embeddings(time_intervals,[x])[0] + x* np.sqrt(self.hidden_size)
class CoSasrec(torch.nn.Module):
    def __init__(self, item_num,max_len,hidden_size,dropout_rate,num_layers,sampling_style,device="cpu",share_embeddings=True,topk_sampling=False,topk_sampling_k=1000):
        super(CoSasrec, self).__init__()
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
        self.pos_emb = DynamicPositionEmbeddingWithRotaty(max_len,hidden_size,device)
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
            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers
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
    def forward(self, positives, mask,types,times): # for training   
        """
        mask: padding mask of 0 and 1
        returns attention_head
        """ 
        att_mask = self.merge_attn_masks(mask)    
        x = self.item_emb(positives)
        x = self.pos_emb(x,times)
        prediction_head = self.encoder(self.input_dropout(x), att_mask)
        return prediction_head
    def train_step(self, batch, iteration,optimizer,logger):
        optimizer.zero_grad()
        prediction_head = self.forward(batch["positives"],batch["mask"],batch["features"]["positives"]["types"],batch["features"]["positives"]["times"])
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
        logger.log("TRAIN",f"i: {iteration}, train_loss: {loss}", )
        loss.backward()
        optimizer.step()
        return loss

    def validate_step(self, batch, iteration,logger):
        prediction_head = self.forward(batch["positives"],batch["mask"],batch["features"]["positives"]["types"],batch["features"]["positives"]["times"])
        # loss:
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