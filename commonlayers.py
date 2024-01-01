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