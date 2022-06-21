# QSpeech

>**QSpeech: Low-Qubit Quantum Speech Application Toolkit**
>
>Accepted by the International Joint Conference on Neural Networks (IJCNN) 2022

## Introduction
This repository is the official implementation of [QSpeech: Low-Qubit Quantum Speech Application Toolkit](https://arxiv.org/abs/2205.13221). 

It proposes:
- The low-qubit variational quantum circuit (VQC).
- A library for the rapid prototyping of hybrid quantum-classical neural networks in speech applications.

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

For the hybrid quantum-classical neural networks in speech applications, we implement the Quantum M5(QM5), Quantum Tacotron(QTacotron) and Quantum Transformer-TTS(QTransformer-TTS).


## Requirements
- Linux (Test on Ubuntu18.04)
- Python3.6+ (Test on Python3.6.8)
- PyTorch
- PennyLane
- Librosa (version 0.7.2)
- Numba (version 0.48.0)

## Basic framework
- QCircuit: the variational quantum circuit(VQC) and low-qubit VQC.
- QLayer: the qlstm, qgru, qattention, qconv.
- QModels: the qm5, qtransformer, qtacotron.

### Notes
The code of qlstm and qattention is based on these two projects as follow: 
- [qlstm](https://github.com/rdisipio/qlstm)
- [qtransformer](https://github.com/rdisipio/qtransformer)

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

## Citation
If you find QSpeech useful in your research, please consider citing:

    @inproceedings{hong2022qspeech,
      title={QSpeech: Low-Qubit Quantum Speech Application Toolkit},
      author={Hong, Zhenhou and Wang, Jianzong and Qu, Xiaoyang and Zhao, Chendong and Tao, Wei and Xiao, Jing},
      booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
      pages={1--8},
      year={2022},
      organization={IEEE}
    }
