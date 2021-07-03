sample_rate = 8000
n_mels = 80
frame_rate = 80
frame_length = 0.05
hop_length = int(sample_rate/frame_rate) 
win_length = int(sample_rate*frame_length)

scaling_factor = 4
min_db_level = -100

bce_weights = 7

embed_dims = 512
hidden_dims = 256 
heads = 4
forward_expansion = 4
num_layers = 1
dropout = 0.15
max_len = 1024
pad_idx = 0

data_file_path = './data/SpeechCommands/speech_commands_v0.02'

checkpoint_path = 'model/'
model_name = 'checkpoint'
restore_step = 1000
save_step = 1000

batch_size = 2
epochs = 40
learning_rate = 3e-4
weight_decay = 0.0001
warmup_steps = 0.2
log_interval = 1
