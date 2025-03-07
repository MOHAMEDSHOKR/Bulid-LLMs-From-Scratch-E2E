class CausalAttention(nn.Module):

          
        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
          
                  super().__init__()
                  self.d_out = d_out
                  self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
                  self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
                  self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
                  self.dropout = nn.Dropout(dropout)  # Compared to the previous SelfAttention_v1 class, we added a dropout layer.
                  self.register_buffer(
                  'mask',
                  torch.triu(torch.ones(context_length, context_length),
                  diagonal=1)) # The register_buffer call is also a new additionm(more information is provided in the following text).
          
        def forward(self, x):
          
                  b, num_tokens, d_in = x.shape
                  keys = self.W_key(x)
                  queries = self.W_query(x)
                  values = self.W_value(x)
                  attn_scores = queries @ keys.transpose(1, 2) # We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
                  attn_scores.masked_fill_(
                  self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
                  attn_weights = torch.softmax(
                  attn_scores / keys.shape[-1]**0.5, dim=-1)
                  attn_weights = self.dropout(attn_weights)
                  context_vec = attn_weights @ values
          
                  return context_vec


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# The resulting context vector is a three-dimensional tensor where each token is now represented by a two-dimensional embedding: 
  
context_vecs.shape: torch.Size([2, 6, 2])
