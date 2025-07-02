# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Leveraging Large Language Models for Spontaneous Speech-Based Suicide Risk Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于自杀风险检测任务，旨在通过语音识别青少年自杀风险。利用大语言模型提取特征，结合传统声学与语义特征，提升检测准确率。**

- **链接: [http://arxiv.org/pdf/2507.00693v1](http://arxiv.org/pdf/2507.00693v1)**

> **作者:** Yifan Gao; Jiao Fu; Long Guo; Hong Liu
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Early identification of suicide risk is crucial for preventing suicidal behaviors. As a result, the identification and study of patterns and markers related to suicide risk have become a key focus of current research. In this paper, we present the results of our work in the 1st SpeechWellness Challenge (SW1), which aims to explore speech as a non-invasive and easily accessible mental health indicator for identifying adolescents at risk of suicide.Our approach leverages large language model (LLM) as the primary tool for feature extraction, alongside conventional acoustic and semantic features. The proposed method achieves an accuracy of 74\% on the test set, ranking first in the SW1 challenge. These findings demonstrate the potential of LLM-based methods for analyzing speech in the context of suicide risk assessment.
>
---
#### [new 002] MambAttention: Mamba with Multi-Head Attention for Generalizable Single-Channel Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于单通道语音增强任务，旨在解决序列模型泛化能力不足的问题。提出MambAttention架构，结合Mamba与多头注意力机制，提升模型在不同数据集上的表现。**

- **链接: [http://arxiv.org/pdf/2507.00966v1](http://arxiv.org/pdf/2507.00966v1)**

> **作者:** Nikolai Lund Kühne; Jesper Jensen; Jan Østergaard; Zheng-Hua Tan
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing for possible publication
>
> **摘要:** With the advent of new sequence models like Mamba and xLSTM, several studies have shown that these models match or outperform state-of-the-art models in single-channel speech enhancement, automatic speech recognition, and self-supervised audio representation learning. However, prior research has demonstrated that sequence models like LSTM and Mamba tend to overfit to the training set. To address this issue, previous works have shown that adding self-attention to LSTMs substantially improves generalization performance for single-channel speech enhancement. Nevertheless, neither the concept of hybrid Mamba and time-frequency attention models nor their generalization performance have been explored for speech enhancement. In this paper, we propose a novel hybrid architecture, MambAttention, which combines Mamba and shared time- and frequency-multi-head attention modules for generalizable single-channel speech enhancement. To train our model, we introduce VoiceBank+Demand Extended (VB-DemandEx), a dataset inspired by VoiceBank+Demand but with more challenging noise types and lower signal-to-noise ratios. Trained on VB-DemandEx, our proposed MambAttention model significantly outperforms existing state-of-the-art LSTM-, xLSTM-, Mamba-, and Conformer-based systems of similar complexity across all reported metrics on two out-of-domain datasets: DNS 2020 and EARS-WHAM_v2, while matching their performance on the in-domain dataset VB-DemandEx. Ablation studies highlight the role of weight sharing between the time- and frequency-multi-head attention modules for generalization performance. Finally, we explore integrating the shared time- and frequency-multi-head attention modules with LSTM and xLSTM, which yields a notable performance improvement on the out-of-domain datasets. However, our MambAttention model remains superior on both out-of-domain datasets across all reported evaluation metrics.
>
---
#### [new 003] A High-Fidelity Speech Super Resolution Network using a Complex Global Attention Module with Spectro-Temporal Loss
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音超分辨率任务，旨在提升语音的采样率。通过重建幅度和相位，解决传统方法忽视相位的问题，提出CTFT-Net模型提升语音质量。**

- **链接: [http://arxiv.org/pdf/2507.00229v1](http://arxiv.org/pdf/2507.00229v1)**

> **作者:** Tarikul Islam Tamiti; Biraj Joshi; Rida Hasan; Rashedul Hasan; Taieba Athay; Nursad Mamun; Anomadarshi Barua
>
> **摘要:** Speech super-resolution (SSR) enhances low-resolution speech by increasing the sampling rate. While most SSR methods focus on magnitude reconstruction, recent research highlights the importance of phase reconstruction for improved perceptual quality. Therefore, we introduce CTFT-Net, a Complex Time-Frequency Transformation Network that reconstructs both magnitude and phase in complex domains for improved SSR tasks. It incorporates a complex global attention block to model inter-phoneme and inter-frequency dependencies and a complex conformer to capture long-range and local features, improving frequency reconstruction and noise robustness. CTFT-Net employs time-domain and multi-resolution frequency-domain loss functions for better generalization. Experiments show CTFT-Net outperforms state-of-the-art models (NU-Wave, WSRGlow, NVSR, AERO) on the VCTK dataset, particularly for extreme upsampling (2 kHz to 48 kHz), reconstructing high frequencies effectively without noisy artifacts.
>
---
#### [new 004] Multi-interaction TTS toward professional recording reproduction
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决合成语音风格难以精细调整的问题。通过多轮交互机制，允许用户逐步优化语音风格。**

- **链接: [http://arxiv.org/pdf/2507.00808v1](http://arxiv.org/pdf/2507.00808v1)**

> **作者:** Hiroki Kanagawa; Kenichi Fujita; Aya Watanabe; Yusuke Ijima
>
> **备注:** 7 pages,6 figures, Accepted to Speech Synthesis Workshop 2025 (SSW13)
>
> **摘要:** Voice directors often iteratively refine voice actors' performances by providing feedback to achieve the desired outcome. While this iterative feedback-based refinement process is important in actual recordings, it has been overlooked in text-to-speech synthesis (TTS). As a result, fine-grained style refinement after the initial synthesis is not possible, even though the synthesized speech often deviates from the user's intended style. To address this issue, we propose a TTS method with multi-step interaction that allows users to intuitively and rapidly refine synthetized speech. Our approach models the interaction between the TTS model and its user to emulate the relationship between voice actors and voice directors. Experiments show that the proposed model with its corresponding dataset enable iterative style refinements in accordance with users' directions, thus demonstrating its multi-interaction capability. Sample audios are available: https://ntt-hilab-gensp. github.io/ssw13multiinteraction_tts/
>
---
#### [new 005] MuteSwap: Silent Face-based Voice Conversion
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文提出MuteSwap，解决无声视频中的语音转换任务（SFVC），通过视觉信息生成目标说话人语音，克服音频不可用的挑战。**

- **链接: [http://arxiv.org/pdf/2507.00498v1](http://arxiv.org/pdf/2507.00498v1)**

> **作者:** Yifan Liu; Yu Fang; Zhouhan Lin
>
> **摘要:** Conventional voice conversion modifies voice characteristics from a source speaker to a target speaker, relying on audio input from both sides. However, this process becomes infeasible when clean audio is unavailable, such as in silent videos or noisy environments. In this work, we focus on the task of Silent Face-based Voice Conversion (SFVC), which does voice conversion entirely from visual inputs. i.e., given images of a target speaker and a silent video of a source speaker containing lip motion, SFVC generates speech aligning the identity of the target speaker while preserving the speech content in the source silent video. As this task requires generating intelligible speech and converting identity using only visual cues, it is particularly challenging. To address this, we introduce MuteSwap, a novel framework that employs contrastive learning to align cross-modality identities and minimize mutual information to separate shared visual features. Experimental results show that MuteSwap achieves impressive performance in both speech synthesis and identity conversion, especially under noisy conditions where methods dependent on audio input fail to produce intelligible results, demonstrating both the effectiveness of our training approach and the feasibility of SFVC.
>
---
#### [new 006] AudioBERTScore: Objective Evaluation of Environmental Sound Synthesis Based on Similarity of Audio embedding Sequences
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到音频任务，旨在解决合成音频客观评估不足的问题。提出AudioBERTScore，通过计算音频嵌入序列的相似性提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2507.00475v1](http://arxiv.org/pdf/2507.00475v1)**

> **作者:** Minoru Kishi; Ryosuke Sakai; Shinnosuke Takamichi; Yusuke Kanamori; Yuki Okamoto
>
> **摘要:** We propose a novel objective evaluation metric for synthesized audio in text-to-audio (TTA), aiming to improve the performance of TTA models. In TTA, subjective evaluation of the synthesized sound is an important, but its implementation requires monetary costs. Therefore, objective evaluation such as mel-cepstral distortion are used, but the correlation between these objective metrics and subjective evaluation values is weak. Our proposed objective evaluation metric, AudioBERTScore, calculates the similarity between embedding of the synthesized and reference sounds. The method is based not only on the max-norm used in conventional BERTScore but also on the $p$-norm to reflect the non-local nature of environmental sounds. Experimental results show that scores obtained by the proposed method have a higher correlation with subjective evaluation values than conventional metrics.
>
---
#### [new 007] Beat and Downbeat Tracking in Performance MIDI Using an End-to-End Transformer Architecture
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于音乐节奏分析任务，解决性能MIDI中的节拍与强拍跟踪问题。提出基于Transformer的端到端模型，提升准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00466v1](http://arxiv.org/pdf/2507.00466v1)**

> **作者:** Sebastian Murgul; Michael Heizmann
>
> **备注:** Accepted to the 22nd Sound and Music Computing Conference (SMC), 2025
>
> **摘要:** Beat tracking in musical performance MIDI is a challenging and important task for notation-level music transcription and rhythmical analysis, yet existing methods primarily focus on audio-based approaches. This paper proposes an end-to-end transformer-based model for beat and downbeat tracking in performance MIDI, leveraging an encoder-decoder architecture for sequence-to-sequence translation of MIDI input to beat annotations. Our approach introduces novel data preprocessing techniques, including dynamic augmentation and optimized tokenization strategies, to improve accuracy and generalizability across different datasets. We conduct extensive experiments using the A-MAPS, ASAP, GuitarSet, and Leduc datasets, comparing our model against state-of-the-art hidden Markov models (HMMs) and deep learning-based beat tracking methods. The results demonstrate that our model outperforms existing symbolic music beat tracking approaches, achieving competitive F1-scores across various musical styles and instruments. Our findings highlight the potential of transformer architectures for symbolic beat tracking and suggest future integration with automatic music transcription systems for enhanced music analysis and score generation.
>
---
#### [new 008] Musical Source Separation of Brazilian Percussion
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音乐源分离任务，旨在解决非西方乐器分离问题。利用巴西萨马鼓数据训练U-Net模型，成功分离出低音鼓声部。**

- **链接: [http://arxiv.org/pdf/2503.04995v1](http://arxiv.org/pdf/2503.04995v1)**

> **作者:** Richa Namballa; Giovana Morais; Magdalena Fuentes
>
> **备注:** 2 pages + references, 1 figure, 1 table, Extended Abstracts for the Late-Breaking Demo Session of the 25th International Society for Music Information Retrieval Conference
>
> **摘要:** Musical source separation (MSS) has recently seen a big breakthrough in separating instruments from a mixture in the context of Western music, but research on non-Western instruments is still limited due to a lack of data. In this demo, we use an existing dataset of Brazilian sama percussion to create artificial mixtures for training a U-Net model to separate the surdo drum, a traditional instrument in samba. Despite limited training data, the model effectively isolates the surdo, given the drum's repetitive patterns and its characteristic low-pitched timbre. These results suggest that MSS systems can be successfully harnessed to work in more culturally-inclusive scenarios without the need of collecting extensive amounts of data.
>
---
#### [new 009] Biorthogonal Tunable Wavelet Unit with Lifting Scheme in Convolutional Neural Network
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 该论文属于图像分类与异常检测任务，旨在提升CNN性能。通过引入可调双正交小波单元，优化卷积与下采样操作，提高分类与检测精度。**

- **链接: [http://arxiv.org/pdf/2507.00739v1](http://arxiv.org/pdf/2507.00739v1)**

> **作者:** An Le; Hung Nguyen; Sungbal Seo; You-Suk Bae; Truong Nguyen
>
> **摘要:** This work introduces a novel biorthogonal tunable wavelet unit constructed using a lifting scheme that relaxes both the orthogonality and equal filter length constraints, providing greater flexibility in filter design. The proposed unit enhances convolution, pooling, and downsampling operations, leading to improved image classification and anomaly detection in convolutional neural networks (CNN). When integrated into an 18-layer residual neural network (ResNet-18), the approach improved classification accuracy on CIFAR-10 by 2.12% and on the Describable Textures Dataset (DTD) by 9.73%, demonstrating its effectiveness in capturing fine-grained details. Similar improvements were observed in ResNet-34. For anomaly detection in the hazelnut category of the MVTec Anomaly Detection dataset, the proposed method achieved competitive and wellbalanced performance in both segmentation and detection tasks, outperforming existing approaches in terms of accuracy and robustness.
>
---
#### [new 010] Tunable Wavelet Unit based Convolutional Neural Network in Optical Coherence Tomography Analysis Enhancement for Classifying Type of Epiretinal Membrane Surgery
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 该论文属于医学图像分类任务，旨在准确区分视网膜手术类型。通过改进的CNN模型和可调小波单元提升分类性能。**

- **链接: [http://arxiv.org/pdf/2507.00743v1](http://arxiv.org/pdf/2507.00743v1)**

> **作者:** An Le; Nehal Mehta; William Freeman; Ines Nagel; Melanie Tran; Anna Heinke; Akshay Agnihotri; Lingyun Cheng; Dirk-Uwe Bartsch; Hung Nguyen; Truong Nguyen; Cheolhong An
>
> **摘要:** In this study, we developed deep learning-based method to classify the type of surgery performed for epiretinal membrane (ERM) removal, either internal limiting membrane (ILM) removal or ERM-alone removal. Our model, based on the ResNet18 convolutional neural network (CNN) architecture, utilizes postoperative optical coherence tomography (OCT) center scans as inputs. We evaluated the model using both original scans and scans preprocessed with energy crop and wavelet denoising, achieving 72% accuracy on preprocessed inputs, outperforming the 66% accuracy achieved on original scans. To further improve accuracy, we integrated tunable wavelet units with two key adaptations: Orthogonal Lattice-based Wavelet Units (OrthLatt-UwU) and Perfect Reconstruction Relaxation-based Wavelet Units (PR-Relax-UwU). These units allowed the model to automatically adjust filter coefficients during training and were incorporated into downsampling, stride-two convolution, and pooling layers, enhancing its ability to distinguish between ERM-ILM removal and ERM-alone removal, with OrthLattUwU boosting accuracy to 76% and PR-Relax-UwU increasing performance to 78%. Performance comparisons showed that our AI model outperformed a trained human grader, who achieved only 50% accuracy in classifying the removal surgery types from postoperative OCT scans. These findings highlight the potential of CNN based models to improve clinical decision-making by providing more accurate and reliable classifications. To the best of our knowledge, this is the first work to employ tunable wavelets for classifying different types of ERM removal surgery.
>
---
#### [new 011] Mitigating Language Mismatch in SSL-Based Speaker Anonymization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音匿名化任务，解决跨语言适应问题。通过微调自监督学习模型，提升不同语言下的语音匿名化效果。**

- **链接: [http://arxiv.org/pdf/2507.00458v1](http://arxiv.org/pdf/2507.00458v1)**

> **作者:** Zhe Zhang; Wen-Chin Huang; Xin Wang; Xiaoxiao Miao; Junichi Yamagishi
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Speaker anonymization aims to protect speaker identity while preserving content information and the intelligibility of speech. However, most speaker anonymization systems (SASs) are developed and evaluated using only English, resulting in degraded utility for other languages. This paper investigates language mismatch in SASs for Japanese and Mandarin speech. First, we fine-tune a self-supervised learning (SSL)-based content encoder with Japanese speech to verify effective language adaptation. Then, we propose fine-tuning a multilingual SSL model with Japanese speech and evaluating the SAS in Japanese and Mandarin. Downstream experiments show that fine-tuning an English-only SSL model with the target language enhances intelligibility while maintaining privacy and that multilingual SSL further extends SASs' utility across different languages. These findings highlight the importance of language adaptation and multilingual pre-training of SSLs for robust multilingual speaker anonymization.
>
---
#### [new 012] Leveraging Unlabeled Audio-Visual Data in Speech Emotion Recognition using Knowledge Distillation
- **分类: cs.LG; cs.HC; cs.MM; eess.AS; eess.IV; eess.SP**

- **简介: 该论文属于语音情感识别任务，旨在减少对大量标注数据的依赖。通过知识蒸馏方法，利用未标注的音视频数据提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00055v1](http://arxiv.org/pdf/2507.00055v1)**

> **作者:** Varsha Pendyala; Pedro Morgado; William Sethares
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Voice interfaces integral to the human-computer interaction systems can benefit from speech emotion recognition (SER) to customize responses based on user emotions. Since humans convey emotions through multi-modal audio-visual cues, developing SER systems using both the modalities is beneficial. However, collecting a vast amount of labeled data for their development is expensive. This paper proposes a knowledge distillation framework called LightweightSER (LiSER) that leverages unlabeled audio-visual data for SER, using large teacher models built on advanced speech and face representation models. LiSER transfers knowledge regarding speech emotions and facial expressions from the teacher models to lightweight student models. Experiments conducted on two benchmark datasets, RAVDESS and CREMA-D, demonstrate that LiSER can reduce the dependence on extensive labeled datasets for SER tasks.
>
---
#### [new 013] Do Music Source Separation Models Preserve Spatial Information in Binaural Audio?
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音乐源分离任务，研究现有模型在双耳音频中保留空间信息的能力。通过实验发现传统模型无法有效保持沉浸式音频的空间特性。**

- **链接: [http://arxiv.org/pdf/2507.00155v1](http://arxiv.org/pdf/2507.00155v1)**

> **作者:** Richa Namballa; Agnieszka Roginska; Magdalena Fuentes
>
> **备注:** 6 pages + references, 4 figures, 2 tables, 26th International Society for Music Information Retrieval (ISMIR) Conference
>
> **摘要:** Binaural audio remains underexplored within the music information retrieval community. Motivated by the rising popularity of virtual and augmented reality experiences as well as potential applications to accessibility, we investigate how well existing music source separation (MSS) models perform on binaural audio. Although these models process two-channel inputs, it is unclear how effectively they retain spatial information. In this work, we evaluate how several popular MSS models preserve spatial information on both standard stereo and novel binaural datasets. Our binaural data is synthesized using stems from MUSDB18-HQ and open-source head-related transfer functions by positioning instrument sources randomly along the horizontal plane. We then assess the spatial quality of the separated stems using signal processing and interaural cue-based metrics. Our results show that stereo MSS models fail to preserve the spatial information critical for maintaining the immersive quality of binaural audio, and that the degradation depends on model architecture as well as the target instrument. Finally, we highlight valuable opportunities for future work at the intersection of MSS and immersive audio.
>
---
#### [new 014] LearnAFE: Circuit-Algorithm Co-design Framework for Learnable Audio Analog Front-End
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于音频信号分类任务，解决AFE与分类器分离设计的问题，通过联合优化AFE滤波器和分类器实现系统最优。**

- **链接: [http://arxiv.org/pdf/2507.00755v1](http://arxiv.org/pdf/2507.00755v1)**

> **作者:** Jinhai Hu; Zhongyi Zhang; Cong Sheng Leow; Wang Ling Goh; Yuan Gao
>
> **备注:** 11 pages, 15 figures, accepted for publication on IEEE Transactions on Circuits and Systems I: Regular Papers
>
> **摘要:** This paper presents a circuit-algorithm co-design framework for learnable analog front-end (AFE) in audio signal classification. Designing AFE and backend classifiers separately is a common practice but non-ideal, as shown in this paper. Instead, this paper proposes a joint optimization of the backend classifier with the AFE's transfer function to achieve system-level optimum. More specifically, the transfer function parameters of an analog bandpass filter (BPF) bank are tuned in a signal-to-noise ratio (SNR)-aware training loop for the classifier. Using a co-design loss function LBPF, this work shows superior optimization of both the filter bank and the classifier. Implemented in open-source SKY130 130nm CMOS process, the optimized design achieved 90.5%-94.2% accuracy for 10-keyword classification task across a wide range of input signal SNR from 5 dB to 20 dB, with only 22k classifier parameters. Compared to conventional approach, the proposed audio AFE achieves 8.7% and 12.9% reduction in power and capacitor area respectively.
>
---
## 更新

#### [replaced 001] Generative AI-based data augmentation for improved bioacoustic classification in noisy environments
- **分类: cs.SD; eess.AS; stat.AP**

- **链接: [http://arxiv.org/pdf/2412.01530v2](http://arxiv.org/pdf/2412.01530v2)**

> **作者:** Anthony Gibbons; Emma King; Ian Donohue; Andrew Parnell
>
> **备注:** 20 pages, 3 tables, 6 figures
>
> **摘要:** 1. Obtaining data to train robust artificial intelligence (AI)-based models for species classification can be challenging, particularly for rare species. Data augmentation can boost classification accuracy by increasing the diversity of training data and is cheaper to obtain than expert-labelled data. However, many classic image-based augmentation techniques are not suitable for audio spectrograms. 2. We investigate two generative AI models as data augmentation tools to synthesise spectrograms and supplement audio data: Auxiliary Classifier Generative Adversarial Networks (ACGAN) and Denoising Diffusion Probabilistic Models (DDPMs). The latter performed particularly well in terms of both realism of generated spectrograms and accuracy in a resulting classification task. 3. Alongside these new approaches, we present a new audio data set of 640 hours of bird calls from wind farm sites in Ireland, approximately 800 samples of which have been labelled by experts. Wind farm data are particularly challenging for classification models given the background wind and turbine noise. 4. Training an ensemble of classification models on real and synthetic data combined gave 92.6% accuracy (and 90.5% with just the real data) when compared with highly confident BirdNET predictions. 5. Our approach can be used to augment acoustic signals for more species and other land-use types, and has the potential to bring about a step-change in our capacity to develop reliable AI-based detection of rare species. Our code is available at https://github.com/gibbona1/SpectrogramGenAI.
>
---
#### [replaced 002] ETTA: Elucidating the Design Space of Text-to-Audio Models
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.19351v2](http://arxiv.org/pdf/2412.19351v2)**

> **作者:** Sang-gil Lee; Zhifeng Kong; Arushi Goel; Sungwon Kim; Rafael Valle; Bryan Catanzaro
>
> **备注:** ICML 2025. Demo: https://research.nvidia.com/labs/adlr/ETTA/ Code: https://github.com/NVIDIA/elucidated-text-to-audio
>
> **摘要:** Recent years have seen significant progress in Text-To-Audio (TTA) synthesis, enabling users to enrich their creative workflows with synthetic audio generated from natural language prompts. Despite this progress, the effects of data, model architecture, training objective functions, and sampling strategies on target benchmarks are not well understood. With the purpose of providing a holistic understanding of the design space of TTA models, we set up a large-scale empirical experiment focused on diffusion and flow matching models. Our contributions include: 1) AF-Synthetic, a large dataset of high quality synthetic captions obtained from an audio understanding model; 2) a systematic comparison of different architectural, training, and inference design choices for TTA models; 3) an analysis of sampling methods and their Pareto curves with respect to generation quality and inference speed. We leverage the knowledge obtained from this extensive analysis to propose our best model dubbed Elucidated Text-To-Audio (ETTA). When evaluated on AudioCaps and MusicCaps, ETTA provides improvements over the baselines trained on publicly available data, while being competitive with models trained on proprietary data. Finally, we show ETTA's improved ability to generate creative audio following complex and imaginative captions -- a task that is more challenging than current benchmarks.
>
---
#### [replaced 003] Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14518v2](http://arxiv.org/pdf/2505.14518v2)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025. Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation.
>
---
#### [replaced 004] ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling
- **分类: cs.CL; eess.SP; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2412.14373v2](http://arxiv.org/pdf/2412.14373v2)**

> **作者:** William Han; Chaojing Duan; Michael A. Rosenberg; Emerson Liu; Ding Zhao
>
> **备注:** 38 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional versatility across domains, including applications to electrocardiograms (ECGs). A growing body of work focuses on generating text from multi-channeled ECG signals and corresponding textual prompts. Existing approaches often involve a two-stage process: pretraining an ECG-specific encoder with a self-supervised learning (SSL) objective, followed by finetuning an LLM for natural language generation (NLG) using encoder-derived features. However, these methods face two key limitations: inefficiency due to multi-stage training and challenges in interpreting encoder-generated features. To overcome these issues, we propose ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs. ECG-Byte compresses and encodes ECG signals into tokens, enabling direct end-to-end LLM training by combining ECG and text tokens. This approach enhances interpretability, as ECG tokens can be directly mapped back to the original signals. Leveraging ECG-Byte, we achieve competitive NLG performance while training 3 times faster and using just 48\% of the data required by traditional two-stage methods.
>
---
#### [replaced 005] Contrastive Conditional Latent Diffusion for Audio-visual Segmentation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2307.16579v2](http://arxiv.org/pdf/2307.16579v2)**

> **作者:** Yuxin Mao; Jing Zhang; Mochu Xiang; Yunqiu Lv; Dong Li; Yiran Zhong; Yuchao Dai
>
> **摘要:** We propose a contrastive conditional latent diffusion model for audio-visual segmentation (AVS) to thoroughly investigate the impact of audio, where the correlation between audio and the final segmentation map is modeled to guarantee the strong correlation between them. To achieve semantic-correlated representation learning, our framework incorporates a latent diffusion model. The diffusion model learns the conditional generation process of the ground-truth segmentation map, resulting in ground-truth aware inference during the denoising process at the test stage. As our model is conditional, it is vital to ensure that the conditional variable contributes to the model output. We thus extensively model the contribution of the audio signal by minimizing the density ratio between the conditional probability of the multimodal data, e.g. conditioned on the audio-visual data, and that of the unimodal data, e.g. conditioned on the audio data only. In this way, our latent diffusion model via density ratio optimization explicitly maximizes the contribution of audio for AVS, which can then be achieved with contrastive learning as a constraint, where the diffusion part serves as the main objective to achieve maximum likelihood estimation, and the density ratio optimization part imposes the constraint. By adopting this latent diffusion model via contrastive learning, we effectively enhance the contribution of audio for AVS. The effectiveness of our solution is validated through experimental results on the benchmark dataset. Code and results are online via our project page: https://github.com/OpenNLPLab/DiffusionAVS.
>
---
#### [replaced 006] ELGAR: Expressive Cello Performance Motion Generation for Audio Rendition
- **分类: cs.GR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04203v2](http://arxiv.org/pdf/2505.04203v2)**

> **作者:** Zhiping Qiu; Yitong Jin; Yuan Wang; Yi Shi; Chongwu Wang; Chao Tan; Xiaobing Li; Feng Yu; Tao Yu; Qionghai Dai
>
> **摘要:** The art of instrument performance stands as a vivid manifestation of human creativity and emotion. Nonetheless, generating instrument performance motions is a highly challenging task, as it requires not only capturing intricate movements but also reconstructing the complex dynamics of the performer-instrument interaction. While existing works primarily focus on modeling partial body motions, we propose Expressive ceLlo performance motion Generation for Audio Rendition (ELGAR), a state-of-the-art diffusion-based framework for whole-body fine-grained instrument performance motion generation solely from audio. To emphasize the interactive nature of the instrument performance, we introduce Hand Interactive Contact Loss (HICL) and Bow Interactive Contact Loss (BICL), which effectively guarantee the authenticity of the interplay. Moreover, to better evaluate whether the generated motions align with the semantic context of the music audio, we design novel metrics specifically for string instrument performance motion generation, including finger-contact distance, bow-string distance, and bowing score. Extensive evaluations and ablation studies are conducted to validate the efficacy of the proposed methods. In addition, we put forward a motion generation dataset SPD-GEN, collated and normalized from the MoCap dataset SPD. As demonstrated, ELGAR has shown great potential in generating instrument performance motions with complicated and fast interactions, which will promote further development in areas such as animation, music education, interactive art creation, etc.
>
---
#### [replaced 007] StreamFlow: Streaming Flow Matching with Block-wise Guided Attention Mask for Speech Token Decoding
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.23986v2](http://arxiv.org/pdf/2506.23986v2)**

> **作者:** Dake Guo; Jixun Yao; Linhan Ma; He Wang; Lei Xie
>
> **摘要:** Recent advancements in discrete token-based speech generation have highlighted the importance of token-to-waveform generation for audio quality, particularly in real-time interactions. Traditional frameworks integrating semantic tokens with flow matching (FM) struggle with streaming capabilities due to their reliance on a global receptive field. Additionally, directly implementing token-by-token streaming speech generation often results in degraded audio quality. To address these challenges, we propose StreamFlow, a novel neural architecture that facilitates streaming flow matching with diffusion transformers (DiT). To mitigate the long-sequence extrapolation issues arising from lengthy historical dependencies, we design a local block-wise receptive field strategy. Specifically, the sequence is first segmented into blocks, and we introduce block-wise attention masks that enable the current block to receive information from the previous or subsequent block. These attention masks are combined hierarchically across different DiT-blocks to regulate the receptive field of DiTs. Both subjective and objective experimental results demonstrate that our approach achieves performance comparable to non-streaming methods while surpassing other streaming methods in terms of speech quality, all the while effectively managing inference time during long-sequence generation. Furthermore, our method achieves a notable first-packet latency of only 180 ms.\footnote{Speech samples: https://dukguo.github.io/StreamFlow/}
>
---
#### [replaced 008] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.16211v2](http://arxiv.org/pdf/2505.16211v2)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Zhe Wang; Xingjian Du; Shun Zhang; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Xiaojun Jia; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Haoyang Li; Yiming Li; Xiaobin Zhuang; Yang Liu; Haibo Hu; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; XiaoFeng Wang; Wenyuan Xu; Wei Dong; Xinfeng Li
>
> **备注:** Technical Report
>
> **摘要:** The rapid advancement and expanding applications of Audio Large Language Models (ALLMs) demand a rigorous understanding of their trustworthiness. However, systematic research on evaluating these models, particularly concerning risks unique to the audio modality, remains largely unexplored. Existing evaluation frameworks primarily focus on the text modality or address only a restricted set of safety dimensions, failing to adequately account for the unique characteristics and application scenarios inherent to the audio modality. We introduce AudioTrust-the first multifaceted trustworthiness evaluation framework and benchmark specifically designed for ALLMs. AudioTrust facilitates assessments across six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. To comprehensively evaluate these dimensions, AudioTrust is structured around 18 distinct experimental setups. Its core is a meticulously constructed dataset of over 4,420 audio/text samples, drawn from real-world scenarios (e.g., daily conversations, emergency calls, voice assistant interactions), specifically designed to probe the multifaceted trustworthiness of ALLMs. For assessment, the benchmark carefully designs 9 audio-specific evaluation metrics, and we employ a large-scale automated pipeline for objective and scalable scoring of model outputs. Experimental results reveal the trustworthiness boundaries and limitations of current state-of-the-art open-source and closed-source ALLMs when confronted with various high-risk audio scenarios, offering valuable insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are available at https://github.com/JusperLee/AudioTrust.
>
---
