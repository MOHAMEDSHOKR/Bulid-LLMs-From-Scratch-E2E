class SelfAttention_v2(nn.Module):
  
      def __init__(self, d_in, d_out, qkv_bias=False):
        
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
      def forward(self, x):
        
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
            context_vec = attn_weights @ values
            return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

The output is
tensor([[-0.0739, 0.0713],
[-0.0748, 0.0703],
[-0.0749, 0.0702],
[-0.0760, 0.0685],
[-0.0763, 0.0679],
[-0.0754, 0.0693]], grad_fn=<MmBackward0>)
