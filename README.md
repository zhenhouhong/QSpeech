# QSpeech
A library tool about quantum neural network in speech application.
Here we implement the Quantum M5(QM5), Quantum Tacotron(QTacotron) and Quantum Transformer-TTS(QTransformer-TTS).

# Install
- pip install torch
- pip install pennylane
- pip install librosa==0.7.2
- pip install numba==0.48.0

# Basic framework
- QCircuit: the variational quantum circuit(VQC) and hybrid VQC.
- QLayer: the qlstm, qgru, qattention, qconv.
- QModels: qm5, qtransformer, qtacotron

# How to use

## Download the datasets
- LJSpeech1.1
- SpeechCommandV0.02

## QM5
- `cd ./Examples/QM5`
- Modify the config.py, like the path of dataset
- `python3 speech-command-recognition.py`

## QTacotron
- `cd ./Examples/QTacotron`
- Modify the config.py, like the path of dataset
- `python3 train.py --batch_size 2`

## QTransformer-TTS
- `cd ./Examples/QTransformerTTS`
- Modify the config.py, like the path of dataset
- `python3 train.py`
