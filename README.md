# Kyrgyz Speech-to-Text Model Fine-tuning

This project fine-tunes the "AkylAI-STT-small" model for Kyrgyz language speech recognition. It uses the "Simonlob/Kany_dataset_mk4_Base" dataset from Hugging Face to train a specialized Kyrgyz speech-to-text (STT) model.

## Features

- Fine-tunes a pre-trained STT model for the Kyrgyz language
- Processes and prepares audio data for training
- Includes evaluation metrics using Word Error Rate (WER)
- Saves training progress and loss plots
- Provides a testing function to verify model performance

## Requirements

```
datasets
scipy
soundfile
numpy
tqdm
torch
transformers
evaluate
jiwer
matplotlib
```

## Setup and Installation

1. Install the required packages:
   ```bash
   pip install transformers datasets evaluate torch torchaudio soundfile scipy matplotlib jiwer
   sudo apt-get install libsndfile1
   pip install -q jiwer evaluate soundfile
   ```

2. Ensure you have enough disk space for the dataset and model.

3. GPU is recommended for faster training (the code will use CUDA if available).

