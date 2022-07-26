# BERT language model params
embed_dim = 256
num_hidden = 128
num_attn_layers = 8
num_attn_heads = 8
enc_maxlen = 256
pos_dropout_rate = 0.1
enc_conv1d_dropout_rate = 0.2
enc_conv1d_layers = 3
enc_conv1d_kernel_size = 5
enc_ffn_dropout_rate = 0.1
self_att_dropout_rate = 0.1
self_att_block_res_dropout = 0.1

# Optimizer params
lr = 0.0005
adam_beta1 = 0.9        # [0.0-1.0]
adam_beta2 = 0.999      # [0.0-1.0]
adam_weight_decay_rate = 0.01
epsilon = 1e-6          # [> 0.0]
mlm_clip_grad_norm = 1.0
clip_grad_norm = True
warmup = 0.1

# Trainer params
epochs = 20
log_freq = 1000
save_train_loss = 2000
save_valid_loss = 10000
save_model = 10000
save_checkpoint = 10000
save_runs = 500
batch_size = 32
train_dataset_ratio = 0.95
graph_batch_size = 4
