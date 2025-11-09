# OV-InstructTTS: Towards Open-Vocabulary Instruct Text-to-Speech

<p align="center">
        &nbsp&nbsp🖥️ <a href="https://y-ren16.github.io/OV-InstructTTS">Demo</a> | 🤗 <a href="https://huggingface.co/datasets/y-ren16/OVSpeech">Datasets</a>&nbsp&nbsp | 🤗 <a href="https://huggingface.co/y-ren16/OV-InstructTTS">Checkpoints</a>&nbsp&nbsp 
        <!-- |&nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2510.00000">Paper</a>&nbsp&nbsp -->
<br>

<!-- ## 🔥🔥🔥 News!! -->
## 🔥 News ! 
* Sep 18, 2025: 👋 We release the [OVSpeech](https://huggingface.co/datasets/y-ren16/OVSpeech) datasets on huggingface. 
* Nov 09, 2025: 👋 We release the [OV-InstructTTS-TEP](https://huggingface.co/y-ren16/OV-InstructTTS) checkpoints on huggingface. 

## 1. Introduction

Instruct Text-to-Speech (InstructTTS) leverages natural language descriptions as style prompts to guide speech synthesis. However, existing InstructTTS methods mainly rely on a direct combination of audio-related labels or their diverse rephrasings, making it difficult to handle flexible, high-level instructions. Such rigid control is insufficient for users such as content creators who wish to steer generation with descriptive instructions. To address these constraints, we introduce **OV-InstructTTS**, a new paradigm for open-vocabulary InstructTTS. We propose a comprehensive solution comprising a newly curated dataset, OV-Speech, and a novel reasoning-driven framework. The OV-Speech dataset pairs speech with open-vocabulary instructions, each augmented with a reasoning process that connects high-level instructions to acoustic features. The reasoning-driven framework infers emotional, acoustic, and paralinguistic information from open-vocabulary instructions before synthesizing speech. Evaluations show that this reasoning-driven approach significantly improves instruction-following fidelity and speech expressiveness. We believe this work can inspire the next user-friendly InstructTTS systems with stronger generalization and real-world applicability. The dataset and demos are publicly available on our project page.

- **Paradigm**: This paper proposes OV-InstructTTS, a novel paradigm that shifts instructTTS beyond its dependency on rephrased audio attributes, pushing controllable speech synthesis towards more flexible and user-friendly real-world applications.
- **Dataset**: We construct OV-Speech, a large-scale dataset providing a foundation for this paradigm. It features open-vocabulary instructions derived from narrative context, reasoning chains that connect instructions to acoustics, and transcriptions enriched with paralinguistic tags.
- **Method**: This paper proposes OV-InstructTTS-TEP, a novel reasoning-driven OV-InstructTTS framework based on LALM. Our method is designed to interpret open-ended instructions through a reasoning process to generate highly expressive speech that is consistent with the user's intent.
- **Experiments**: Extensive experiments and ablation studies demonstrate the value of our dataset and the effectiveness of the OV-InstructTTS-TEP framework. LLM-as-a-judge and subjective evaluations confirm the consistency of synthesized speech with open-ended instructions.

<div align="center">
  <img src="assets/images/ov.png" alt="Architecture" width="800" />
</div>

## 2. Code Usage

### 🗂️ 2.1 Dataset and Pretrained Model Preparation

#### 2.1.1 ContextSpeech Dataset

Download the ContextSpeech dataset from [here](https://huggingface.co/datasets/Insects/ContextSpeech) and place it in the `dataset/ContextSpeech` directory.

#### 2.1.1 OV-Speech Dataset

Download the OV-Speech dataset from [here](https://huggingface.co/datasets/y-ren16/OVSpeech) and place it in the `dataset/OVSpeech` directory.

#### 2.1.3 pretrained model

Download the OV-InstructTTS-TEP checkpoints from [here](https://huggingface.co/y-ren16/OV-InstructTTS) and place it in the `checkpoints/OV-InstructTTS` directory.

### 🔧  2.2  Dependencies and Installation

- Python >= 3.10
- [PyTorch >= 2.3-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
conda create -n ovtts python=3.10
conda activate ovtts
pip install torch torchvision torchaudio
pip install transformers==4.49.0 librosa onnxruntime s3tokenizer diffusers hyperpyyaml
```
###  🚀 2.3 Inference Scripts

To run inference with OV-InstructTTS-TEP, use the following command:

```bash
python inf_ov_instructtts_examples.py
```

## 3. Acknowledgement

This repo is based on [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2). We highly appreciate their contributions to this community.