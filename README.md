# QSpeech

>**QSpeech: Low-Qubit Quantum Speech Application Toolkit**

## Introduction
This repository is the official implementation of [QSpeech: Low-Qubit Quantum Speech Application Toolkit]().

<table><tbody>
    <tr>
        <th>Low-qubit VQC</th>
        <th>QSpeech Framework</th>
    </tr>
    <tr>
        <td><div align=center><img src="https://github.com/zhenhouhong/QSpeech/blob/main/low-qubit-cricuit.png"></div></td>
        <td><div align=center><img src="https://github.com/zhenhouhong/QSpeech/blob/main/qspeech-framework.png"></div></td>
    </tr>
</table>

A library tool about quantum neural network in speech application. 
Here we implement the Low-qubit Circuit, Quantum M5(QM5), Quantum Tacotron(QTacotron) and Quantum Transformer-TTS(QTransformer-TTS).


## Requirements
- Linux (Test on Ubuntu18.04)
- Python3.6+ (Test on Python3.6.8)
- PyTorch
- Pennylane
- Librosa (version 0.7.2)
- Numba (version 0.48.0)

## Basic framework
- QCircuit: the variational quantum circuit(VQC) and low-qubit VQC.
- QLayer: the qlstm, qgru, qattention, qconv.
- QModels: qm5, qtransformer, qtacotron

## How to use

### Download the datasets
- LJSpeech1.1
- SpeechCommandV0.02

### QM5
- `cd ./Examples/QM5`
- Modify the config.py, like the path of dataset
- `python3 speech-command-recognition.py`

### QTacotron
- `cd ./Examples/QTacotron`
- Modify the hyperparams.py, like the path of dataset
- `python3 train.py --batch_size 2`

### QTransformer-TTS
- `cd ./Examples/QTransformerTTS`
- Modify the config.py, like the path of dataset
- `python3 train.py`
