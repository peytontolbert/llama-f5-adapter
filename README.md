# LLaMA-to-Speech Adapter

A neural adapter that converts LLaMA embeddings directly into speech using F5-TTS. This project enables text-to-speech generation by bridging large language models with voice synthesis.

## Overview

This project implements an adapter architecture that:
1. Takes embeddings from LLaMA language model
2. Processes them through a specialized transformer-based adapter
3. Generates mel spectrograms compatible with F5-TTS
4. Synthesizes high-quality speech output

## Features

- Direct conversion from LLaMA embeddings to speech
- Enhanced transformer architecture with DiT-style blocks
- Multi-scale feature matching and temporal consistency
- Voice cloning capabilities through F5-TTS integration
- Continuous learning support for voice profiles

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- torchaudio
- sounddevice
- pydub

## Installation


```bash
Clone the repository
git clone https://github.com/peytontolbert/llama-to-speech-adapter
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download model weights
(Instructions for obtaining F5-TTS weights and LLaMA model)

## Usage

### Training the Adapter
```bash
python train.py
```

This will:
1. Generate a dataset of LLaMA embeddings paired with mel spectrograms
2. Train the adapter model
3. Save checkpoints during training

### Generating Speech
```bash
python run.py
```

This will:
1. Load the trained adapter
2. Generate speech from text using the LLaMA-to-Speech pipeline

## Project Structure
```files
├── adapter/
│ └── adapter.py # Enhanced embedding adapter implementation
├── f5/
│ ├── f5tts.py # F5-TTS service and voice profile management
│ ├── dit.py # DiT model implementation
│ ├── cfm.py # CFM model implementation
│ ├── modules.py # F5-TTS modules
│ ├── utils_infer.py # TTS utilities for inference
│ └── utils.py # TTS utilities
├── scripts/
│ └── generate_dataset.py # Script to generate dataset
├── train.py # Training script
├── run.py # Inference script
├── requirements.txt # Dependencies
└── README.md
```


## Model Architecture

The adapter uses a transformer-based architecture with:
- DiT-style blocks for processing embeddings
- Relative positional embeddings
- Multi-scale feature matching
- Temporal consistency preservation
- Mel spectrogram range normalization

## Training

The model is trained using multiple loss components:
- MSE Loss for basic reconstruction
- Spectral Convergence Loss
- Log-magnitude Loss
- Feature Matching Loss
- Temporal Consistency Loss

## Acknowledgments

- F5-TTS for the text-to-speech foundation
- Meta AI for the LLaMA language model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.