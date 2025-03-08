
class MultiHeadAttention(nn.Module):
  
      def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        
            super().__init__()
            assert (d_out % num_heads == 0), \ "d_out must be divisible by num_heads"
            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
  
      def forward(self, x):
        
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
        
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)
        
            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
        
            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)
        
            return context_vec



torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# The results show that the output dimension is directly controlled by the d_out argument:

tensor([[[0.3190, 0.4858],
          [0.2943, 0.3897],
          [0.2856, 0.3593],
          [0.2693, 0.3873],
          [0.2639, 0.3928],
          [0.2575, 0.4028]],
        
          [[0.3190, 0.4858],
          [0.2943, 0.3897],
          [0.2856, 0.3593],
          [0.2693, 0.3873],
          [0.2639, 0.3928],
          [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)

context_vecs.shape: torch.Size([2, 6, 2])
