#%%
import multihead_attention as mha
import torch 

default_args = {'embed_dim': 1280,
 'num_heads': 20,
 'kdim': 1280,
 'vdim': 1280,
 'dropout': 0.0,
 'add_zero_attn': False,
 'self_attention': False,
 'encoder_decoder_attention': False,
 "bias" :True,
 "add_bias_kv" : False,
 "use_rotary_embeddings" : True
 }


sim_batch = torch.randn(1,32,1280).to("cuda")
mha_default = mha.MultiheadAttention(**default_args).to("cuda")

# %%
output_default = mha_default(sim_batch,sim_batch,sim_batch)
# %%
