  
import torch.nn as nn

class SelfAttention_v1(nn.Module):
  
      def __init__(self, d_in, d_out):
        
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
      def forward(self, x):
        
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value
            attn_scores = queries @ keys.T # omega
            attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
            context_vec = attn_weights @ values
            return context_vec
        
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

this results in a matrix storing the six context vectors:
tensor([[0.2996, 0.8053],
[0.3061, 0.8210],
[0.3058, 0.8203],
[0.2948, 0.7939],
[0.2927, 0.7891],
[0.2990, 0.8040]], grad_fn=<MmBackward0>)
