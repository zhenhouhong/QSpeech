data_file_path = '../../data/SpeechCommands/speech_commands_v0.02'

checkpoint_path = 'model/'
model_name = 'checkpoint'
restore_step = 1000
save_step = 1000

batch_size = 256
epochs = 1
learning_rate = 3e-4
weight_decay = 0.0001
warmup_steps = 0.2
log_interval = 1
