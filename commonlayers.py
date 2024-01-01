import torch
import torch.nn.functional as F
class DynamicPositionEmbedding(torch.nn.Module):

    def __init__(self, max_len, dimension,device='cpu'):
        super(DynamicPositionEmbedding, self).__init__()
        self.device=device
        self.max_len = max_len
        self.embedding = torch.nn.Embedding(max_len, dimension,device=device)
        self.pos_indices = torch.arange(0, self.max_len, dtype=torch.int,device=device)
        self.register_buffer('pos_indices_const', self.pos_indices)
    def forward(self, x:torch.Tensor):
        seq_len = x.shape[1]
        return self.embedding(self.pos_indices_const[-seq_len:]) + x
def _projection_on_C(y:torch.Tensor,beta:float):
    return (y-beta).clamp(0,1)
def _equation_on_C(y:torch.Tensor,beta:float,B):
    return (y-beta).clamp(0,1).sum()-B
def _find_beta_on_C(y:torch.Tensor,B,init_beta:float=1,init_steps=1,delta=1e-5,maxiter=1000):
    val=_equation_on_C(y,init_beta,B)
    if val>0:
        pos_beta=init_beta
        while val>0:
            init_beta+=init_steps
            val=_equation_on_C(y,init_beta,B)
            init_steps*=2
        neg_beta=init_beta
    else:
        neg_beta=init_beta
        while val<0:
            init_beta-=init_steps
            val=_equation_on_C(y,init_beta,B)
            init_steps*=2
        pos_beta=init_beta
    mv=pos_beta
    i=0
    while(i<maxiter and neg_beta-pos_beta>delta):
        mid_beta=(pos_beta+neg_beta)/2
        mv=_equation_on_C(y,mid_beta,B)
        if (mv>0):
            pos_beta=mid_beta
        elif (mv<0):
            neg_beta=mid_beta    
        else:
            return mid_beta
        i+=1
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
#     def forward(ctx, input:torch.Tensor, weights:torch.Tensor,sample_num:int,temperature:int):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input, weights,sample_num,temperature)
#         mask=torch.bernoulli(weights)
#         return mask*input
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the inputs: here input and weights
#         """
#         input, weights,sample_num,temperature = ctx.saved_tensors
#         s_pre=torch.log(weights/(1-weights))
#         mask=0
#         gumbels=sample_gumbel(weights.size()+(2,sample_num))
#         for i in range(sample_num):
#             gumbel_delta=gumbels[...,1,i]-gumbels[...,0,i]
#             mask+=torch.sigmoid((s_pre+gumbel_delta)/temperature)
            
#         mask=mask/sample_num
#         grad_input = weights.clone()*grad_output
#         grad_weights = input.clone()*grad_output
#         return grad_input, grad_weights,None,None
class SparseAttentionMask(torch.nn.Module):
    def __init__(self,max_len:int,hidden_size:int,temperature:float=0.8,sample_num=10,device="cpu"):
        super(SparseAttentionMask,self).__init__()
        self.device=device
        _weights=torch.ones(size=(max_len,hidden_size),dtype=torch.float,device=device)/2
        self.weights=torch.nn.Parameter(_weights)
        self.weights.retain_grad()
        self.sample_num=sample_num
        self.temperature=temperature
        # self.fn=SparseAttentionMaskFowardFunc.apply
    def getPolicyGradient(self):
        return self.weights.grad
    def forward(self,attention_head:torch.Tensor,):
        # return self.fn(attention_head,self.weights)
        s_pre=torch.log(self.weights/(1-self.weights))
        mask=0
        gumbels=sample_gumbel(self.weights.size()+(2,self.sample_num),device=self.device)
        for i in range(self.sample_num):
            gumbel_delta=gumbels[...,1,i]-gumbels[...,0,i]
            mask+=torch.sigmoid((s_pre+gumbel_delta)/self.temperature)
        return (mask/self.sample_num)*attention_head