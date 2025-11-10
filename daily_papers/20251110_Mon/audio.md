# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] Robust Neural Audio Fingerprinting using Music Foundation Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文面向鲁棒音频指纹识别任务，针对短视频平台中音乐被篡改的问题，提出基于音乐基础模型与增强数据训练的神经指纹方法，显著提升对失真音频的识别与定位能力。**

- **链接: [http://arxiv.org/pdf/2511.05399v1](http://arxiv.org/pdf/2511.05399v1)**

> **作者:** Shubhr Singh; Kiran Bhat; Xavier Riley; Benjamin Resnick; John Thickstun; Walter De Brouwer
>
> **摘要:** The proliferation of distorted, compressed, and manipulated music on modern media platforms like TikTok motivates the development of more robust audio fingerprinting techniques to identify the sources of musical recordings. In this paper, we develop and evaluate new neural audio fingerprinting techniques with the aim of improving their robustness. We make two contributions to neural fingerprinting methodology: (1) we use a pretrained music foundation model as the backbone of the neural architecture and (2) we expand the use of data augmentation to train fingerprinting models under a wide variety of audio manipulations, including time streching, pitch modulation, compression, and filtering. We systematically evaluate our methods in comparison to two state-of-the-art neural fingerprinting models: NAFP and GraFPrint. Results show that fingerprints extracted with music foundation models (e.g., MuQ, MERT) consistently outperform models trained from scratch or pretrained on non-musical audio. Segment-level evaluation further reveals their capability to accurately localize fingerprint matches, an important practical feature for catalog management.
>
---
#### [new 002] EMO100DB: An Open Dataset of Improvised Songs with Emotion Data
- **分类: cs.SD; cs.IR; cs.MM**

- **简介: 该论文构建了EMO100DB开源数据集，收录20名参与者基于Russell情绪环模型报告情绪后即兴创作的歌曲，包含音频、旋律MIDI与歌词，旨在支持音乐与情绪关系的多维研究。**

- **链接: [http://arxiv.org/pdf/2511.04755v1](http://arxiv.org/pdf/2511.04755v1)**

> **作者:** Daeun Hwang; Saebyul Park
>
> **备注:** 4 pages, 6 figures, International Conference on Music Perception and Cognition
>
> **摘要:** In this study, we introduce Emo100DB: a dataset consisting of improvised songs that were recorded and transcribed with emotion data based on Russell's circumplex model of emotion. The dataset was developed by collecting improvised songs that consist of melody, lyrics, and an instrumental accompaniment played, sung, and recorded by 20 young adults. Before recording each song, the participants were asked to report their emotional state, with the axes representing arousal and valence based on Russell's circumplex model of emotions. The dataset is organized into four emotion quadrants, and it includes the lyrics text and MIDI file of the melody extracted from the participant recordings, along with the original audio in WAV format. By providing an integrated composition of data and analysis, this study aims to offer a comprehensive dataset that allows for a diverse exploration of the relationship between music and emotion.
>
---
#### [new 003] MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages
- **分类: cs.SD; cs.AI**

- **简介: MERaLiON-SER是一款面向英语与东南亚语言的语音情感识别模型，通过联合离散与维度情感损失，提升跨语言情感理解精度，超越开源编码器与Audio-LLMs，推动情感感知在智能音频系统中的应用。**

- **链接: [http://arxiv.org/pdf/2511.04914v1](http://arxiv.org/pdf/2511.04914v1)**

> **作者:** Hardik B. Sailor; Aw Ai Ti; Chen Fang Yih Nancy; Chiu Ying Lay; Ding Yang; He Yingxu; Jiang Ridong; Li Jingtao; Liao Jingyi; Liu Zhuohan; Lu Yanfeng; Ma Yi; Manas Gupta; Muhammad Huzaifah Bin Md Shahrin; Nabilah Binte Md Johan; Nattadaporn Lertcheva; Pan Chunlei; Pham Minh Duc; Siti Maryam Binte Ahmad Subaidi; Siti Umairah Binte Mohammad Salleh; Sun Shuo; Tarun Kumar Vangani; Wang Qiongqiong; Won Cheng Yi Lewis; Wong Heng Meng Jeremy; Wu Jinyang; Zhang Huayun; Zhang Longyin; Zou Xunlong
>
> **备注:** https://huggingface.co/MERaLiON/MERaLiON-SER-v1
>
> **摘要:** We present MERaLiON-SER, a robust speech emotion recognition model de- signed for English and Southeast Asian languages. The model is trained using a hybrid objective combining weighted categorical cross-entropy and Concordance Correlation Coefficient (CCC) losses for joint discrete and dimensional emotion modelling. This dual approach enables the model to capture both the distinct categories of emotion (like happy or angry) and the fine-grained, such as arousal (intensity), valence (positivity/negativity), and dominance (sense of control), lead- ing to a more comprehensive and robust representation of human affect. Extensive evaluations across multilingual Singaporean languages (English, Chinese, Malay, and Tamil ) and other public benchmarks show that MERaLiON-SER consistently surpasses both open-source speech encoders and large Audio-LLMs. These results underscore the importance of specialised speech-only models for accurate paralin- guistic understanding and cross-lingual generalisation. Furthermore, the proposed framework provides a foundation for integrating emotion-aware perception into future agentic audio systems, enabling more empathetic and contextually adaptive multimodal reasoning.
>
---
#### [new 004] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出一种噪声增强自编码器，通过感知损失使音乐表征形成感知层次结构，提升音高惊讶度估计与脑电响应预测性能，解决传统编码器忽视感知重要性的问题。**

- **链接: [http://arxiv.org/pdf/2511.05350v1](http://arxiv.org/pdf/2511.05350v1)**

> **作者:** Mathias Rose Bjare; Giorgia Cantisani; Marco Pasini; Stefan Lattner; Gerhard Widmer
>
> **备注:** Accepted at NeurIPS 2025 - AI for Music Workshop, 11 pages, 5 figures, 1 table
>
> **摘要:** We argue that training autoencoders to reconstruct inputs from noised versions of their encodings, when combined with perceptual losses, yields encodings that are structured according to a perceptual hierarchy. We demonstrate the emergence of this hierarchical structure by showing that, after training an audio autoencoder in this manner, perceptually salient information is captured in coarser representation structures than with conventional training. Furthermore, we show that such perceptual hierarchies improve latent diffusion decoding in the context of estimating surprisal in music pitches and predicting EEG-brain responses to music listening. Pretrained weights are available on github.com/CPJKU/pa-audioic.
>
---
#### [new 005] Synthesizing speech with selected perceptual voice qualities - A case study with creaky voice
- **分类: eess.AS**

- **简介: 该论文研究语音合成中音质调控任务，旨在解决非持久性嗓音（如喉音）难以精准合成的问题。通过引入归一化流模块，实现对喉音的全局操控，无需依赖不可靠的帧级预测，实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2511.05143v1](http://arxiv.org/pdf/2511.05143v1)**

> **作者:** Frederik Rautenberg; Fritz Seebauer; Jana Wiechmann; Michael Kuhlmann; Petra Wagner; Reinhold Haeb-Umbach
>
> **备注:** Proceedings of Interspeech
>
> **摘要:** The control of perceptual voice qualities in a text-to-speech (TTS) system is of interest for applications where unmanipu- lated and manipulated speech probes can serve to illustrate pho- netic concepts that are otherwise difficult to grasp. Here, we show that a TTS system, that is augmented with a global speaker attribute manipulation block based on normalizing flows1 , is capable of correctly manipulating the non-persistent, localized quality of creaky voice, thus avoiding the necessity of a, typi- cally unreliable, frame-wise creak predictor. Subjective listen- ing tests confirm successful creak manipulation at a slightly re- duced MOS score compared to the original recording.
>
---
#### [new 006] Passive Acoustic Monitoring of Noisy Coral Reefs
- **分类: cs.SD**

- **简介: 该论文研究利用被动声学监测受噪声干扰的珊瑚礁生态，通过训练CNN去噪器提升信号质量，揭示声学指标与珊瑚健康状况的关联，验证去噪后声学数据在珊瑚礁监测中的有效性。**

- **链接: [http://arxiv.org/pdf/2511.05349v1](http://arxiv.org/pdf/2511.05349v1)**

> **作者:** Hari Vishnu; Yuen Min Too; Mandar Chitre; Danwei Huang; Teong Beng Koay; Sudhanshi S. Jain
>
> **摘要:** Passive acoustic monitoring offers the potential to enable long-term, spatially extensive assessments of coral reefs. To explore this approach, we deployed underwater acoustic recorders at ten coral reef sites around Singapore waters over two years. To mitigate the persistent biological noise masking the low-frequency reef soundscape, we trained a convolutional neural network denoiser. Analysis of the acoustic data reveals distinct morning and evening choruses. Though the correlation with environmental variates was obscured in the low-frequency part of the noisy recordings, the denoised data showed correlations of acoustic activity indices such as sound pressure level and acoustic complexity index with diver-based assessments of reef health such as live coral richness and cover, and algal cover. Furthermore, the shrimp snap rate, computed from the high-frequency acoustic band, is robustly correlated with the reef parameters, both temporally and spatially. This study demonstrates that passive acoustics holds valuable information that can help with reef monitoring, provided the data is effectively denoised and interpreted. This methodology can be extended to other marine environments where acoustic monitoring is hindered by persistent noise.
>
---
#### [new 007] A Penny for Your Thoughts: Decoding Speech from Inexpensive Brain Signals
- **分类: cs.SD; cs.AI; cs.CL; cs.HC; eess.AS; q-bio.NC**

- **简介: 该论文研究从低成本脑电图（EEG）信号中解码语音，属于脑机接口中的脑到语音解码任务。通过改进Meta的EEG解码器，引入个性化注意力与双路径RNN，提升语音重建准确率，验证个性化架构的有效性。**

- **链接: [http://arxiv.org/pdf/2511.04691v1](http://arxiv.org/pdf/2511.04691v1)**

> **作者:** Quentin Auster; Kateryna Shapovalenko; Chuang Ma; Demaio Sun
>
> **摘要:** We explore whether neural networks can decode brain activity into speech by mapping EEG recordings to audio representations. Using EEG data recorded as subjects listened to natural speech, we train a model with a contrastive CLIP loss to align EEG-derived embeddings with embeddings from a pre-trained transformer-based speech model. Building on the state-of-the-art EEG decoder from Meta, we introduce three architectural modifications: (i) subject-specific attention layers (+0.15% WER improvement), (ii) personalized spatial attention (+0.45%), and (iii) a dual-path RNN with attention (-1.87%). Two of the three modifications improved performance, highlighting the promise of personalized architectures for brain-to-speech decoding and applications in brain-computer interfaces.
>
---
#### [new 008] Model Merging Improves Zero-Shot Generalization in Bioacoustic Foundation Models
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于生物声学基础模型的零样本泛化任务，解决NatureLM在多指令请求下性能下降问题。通过融合其与基础语言模型，恢复指令遵循能力，显著提升零样本泛化性能，实现超200%的相对提升。**

- **链接: [http://arxiv.org/pdf/2511.05171v1](http://arxiv.org/pdf/2511.05171v1)**

> **作者:** Davide Marincione; Donato Crisostomi; Roberto Dessi; Emanuele Rodolà; Emanuele Rossi
>
> **摘要:** Foundation models capable of generalizing across species and tasks represent a promising new frontier in bioacoustics, with NatureLM being one of the most prominent examples. While its domain-specific fine-tuning yields strong performance on bioacoustic benchmarks, we observe that it also introduces trade-offs in instruction-following flexibility. For instance, NatureLM achieves high accuracy when prompted for either the common or scientific name individually, but its accuracy drops significantly when both are requested in a single prompt. We address this by applying a simple model merging strategy that interpolates NatureLM with its base language model, recovering instruction-following capabilities with minimal loss of domain expertise. Finally, we show that the merged model exhibits markedly stronger zero-shot generalization, achieving over a 200% relative improvement and setting a new state-of-the-art in closed-set zero-shot classification of unseen species.
>
---
## 更新

#### [replaced 001] Preserving Speaker Information in Direct Speech-to-Speech Translation with Non-Autoregressive Generation and Pretraining
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.07316v3](http://arxiv.org/pdf/2412.07316v3)**

> **作者:** Rui Zhou; Akinori Ito; Takashi Nose
>
> **摘要:** Speech-to-Speech Translation (S2ST) refers to the conversion of speech in one language into semantically equivalent speech in another language, facilitating communication between speakers of different languages. Speech-to-Discrete Unit Translation (S2UT), a mainstream approach for end-to-end S2ST, addresses challenges such as error propagation across modules and slow inference speed often encountered in traditional cascade systems. However, as discrete units primarily capture content information, conventional S2UT methods fail to retain speaker-specific characteristics from the source. Our previous work, SC-S2UT, introduced a speaker adapter and a unit-to-mel structure, enabling the preservation of speaker information and non-autoregressive speech generation. Building on this foundation, this study proposes a self-supervised pretraining method to enrich the information extracted by both the speaker adapter and the unit-to-mel structure. Additionally, we investigate different feature fusion strategies to further improve the integration of speaker and content features. Experiments conducted on the CVSS-T dataset for ES-EN and FR-EN tasks demonstrate that our proposed method achieves a BLEU score improvement of 1.14 compared to SC-S2UT, along with significant enhancements in MOS and speaker similarity. Furthermore, our approach achieves translation quality comparable to traditional S2UT, with only a minimal increase of 0.04s per utterance in inference time, while maintaining high speaker similarity. These results validate the effectiveness of the proposed method.
>
---
#### [replaced 002] Comparative Study on Noise-Augmented Training and its Effect on Adversarial Robustness in ASR Systems
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.01813v4](http://arxiv.org/pdf/2409.01813v4)**

> **作者:** Karla Pizzi; Matías Pizarro; Asja Fischer
>
> **摘要:** In this study, we investigate whether noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different ASR architectures, each trained under three different augmentation conditions: (1) background noise, speed variations, and reverberations; (2) speed variations only; (3) no data augmentation. We then evaluate the robustness of all resulting models against attacks with white-box or black-box adversarial examples. Our results demonstrate that noise augmentation not only enhances model performance on noisy speech but also improves the model's robustness to adversarial attacks.
>
---
