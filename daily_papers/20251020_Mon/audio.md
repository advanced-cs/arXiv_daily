# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] SpikeVox: Towards Energy-Efficient Speech Therapy Framework with Spike-driven Generative Language Models
- **分类: cs.SD; cs.AI; cs.NE**

- **简介: 该论文提出SpikeVox框架，属语音康复任务，旨在解决现有方法能耗高、缺乏治疗反馈的问题。工作包括：结合脉冲生成语言模型实现低功耗语音识别与障碍检测，生成个性化训练内容，并提供发音指导，支持移动端部署。**

- **链接: [http://arxiv.org/pdf/2510.15566v1](http://arxiv.org/pdf/2510.15566v1)**

> **作者:** Rachmad Vidya Wicaksana Putra; Aadithyan Rajesh Nair; Muhammad Shafique
>
> **备注:** Accepted at the IEEE Biomedical Circuits and Systems Conference (BioCAS) 2025, Abu Dhabi, UAE
>
> **摘要:** Speech disorders can significantly affect the patients capability to communicate, learn, and socialize. However, existing speech therapy solutions (e.g., therapist or tools) are still limited and costly, hence such solutions remain inadequate for serving millions of patients worldwide. To address this, state-of-the-art methods employ neural network (NN) algorithms to help accurately detecting speech disorders. However, these methods do not provide therapy recommendation as feedback, hence providing partial solution for patients. Moreover, these methods incur high energy consumption due to their complex and resource-intensive NN processing, hence hindering their deployments on low-power/energy platforms (e.g., smartphones). Toward this, we propose SpikeVox, a novel framework for enabling energy-efficient speech therapy solutions through spike-driven generative language model. Specifically, SpikeVox employs a speech recognition module to perform highly accurate speech-to-text conversion; leverages a spike-driven generative language model to efficiently perform pattern analysis for speech disorder detection and generates suitable exercises for therapy; provides guidance on correct pronunciation as feedback; as well as utilizes the REST API to enable seamless interaction for users. Experimental results demonstrate that SpikeVox achieves 88% confidence level on average in speech disorder recognition, while providing a complete feedback for therapy exercises. Therefore, SpikeVox provides a comprehensive framework for energy-efficient speech therapy solutions, and potentially addresses the significant global speech therapy access gap.
>
---
#### [new 002] LDCodec: A high quality neural audio codec with low-complexity decoder
- **分类: eess.AS**

- **简介: 该论文提出LDCodec，旨在解决神经音频编解码器解码复杂度高的问题。面向低码率音频压缩任务，设计了低复杂度解码器，结合残差单元、LSRVQ量化与感知损失，在6kbps下音质优于Opus 12kbps。**

- **链接: [http://arxiv.org/pdf/2510.15364v1](http://arxiv.org/pdf/2510.15364v1)**

> **作者:** Jiawei Jiang; Linping Xu; Dejun Zhang; Qingbo Huang; Xianjun Xia; Yijian Xiao
>
> **摘要:** Neural audio coding has been shown to outperform classical audio coding at extremely low bitrates. However, the practical application of neural audio codecs is still limited by their elevated complexity. To address this challenge, we have developed a high-quality neural audio codec with a low-complexity decoder, named LDCodec (Low-complexity Decoder Neural Audio Codec), specifically designed for on-demand streaming media clients, such as smartphones. Specifically, we introduced a novel residual unit combined with Long-term and Short-term Residual Vector Quantization (LSRVQ), subband-fullband frequency discriminators, and perceptual loss functions. This combination results in high-quality audio reconstruction with lower complexity. Both our subjective and objective tests demonstrated that our proposed LDCodec at 6kbps outperforms Opus at 12kbps.
>
---
#### [new 003] Quantization-Based Score Calibration for Few-Shot Keyword Spotting with Dynamic Time Warping in Noisy Environments
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究少样本关键词检测中的阈值选择问题，提出基于量化的分数校准方法，通过量化嵌入并利用量化误差归一化检测分数，提升噪声环境下的检测性能。**

- **链接: [http://arxiv.org/pdf/2510.15432v1](http://arxiv.org/pdf/2510.15432v1)**

> **作者:** Kevin Wilkinghoff; Alessia Cornaggia-Urrigshardt; Zheng-Hua Tan
>
> **摘要:** Detecting occurrences of keywords with keyword spotting (KWS) systems requires thresholding continuous detection scores. Selecting appropriate thresholds is a non-trivial task, typically relying on optimizing the performance on a validation dataset. However, such greedy threshold selection often leads to suboptimal performance on unseen data, particularly in varying or noisy acoustic environments or few-shot settings. In this work, we investigate detection threshold estimation for template-based open-set few-shot KWS using dynamic time warping on noisy speech data. To mitigate the performance degradation caused by suboptimal thresholds, we propose a score calibration approach consisting of two different steps: quantizing embeddings and normalizing detection scores using the quantization error prior to thresholding. Experiments on KWS-DailyTalk with simulated high frequency radio channels show that the proposed calibration approach simplifies the choice of detection thresholds and significantly improves the resulting performance.
>
---
#### [new 004] MC-LExt: Multi-Channel Target Speaker Extraction with Onset-Prompted Speaker Conditioning Mechanism
- **分类: eess.AS**

- **简介: 该论文研究多通道目标说话人提取（MC-TSE），旨在解决现有方法依赖声源方向或易受噪声混响影响的问题。提出MC-LExt框架，通过在混合信号各通道前拼接目标说话人注册语音，实现端到端的说话人与空间线索联合建模，提升提取性能。**

- **链接: [http://arxiv.org/pdf/2510.15437v1](http://arxiv.org/pdf/2510.15437v1)**

> **作者:** Tongtao Ling; Shulin He; Pengjie Shen; Zhong-Qiu Wang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Multi-channel target speaker extraction (MC-TSE) aims to extract a target speaker's voice from multi-speaker signals captured by multiple microphones. Existing methods often rely on auxiliary clues such as direction-of-arrival (DOA) or speaker embeddings. However, DOA-based approaches depend on explicit direction estimation and are sensitive to microphone array geometry, while methods based on speaker embeddings model speaker identity in an implicit manner and may degrade in noisy-reverberant conditions. To address these limitations, we propose multi-channel listen to extract (MC-LExt), a simple but highly-effective framework for MC-TSE. Our key idea is to prepend a short enrollment utterance of the target speaker to each channel of the multi-channel mixture, providing an onset-prompted conditioning signal that can guide TSE. This design allows the deep neural network (DNN) to learn spatial and speaker identity cues jointly in a fully end-to-end manner. Experiments on noisy-reverberant benchmarks, including WHAMR! and MC-Libri2Mix, demonstrate the effectiveness of MC-TSE.
>
---
#### [new 005] Magnitude and Phase-based Feature Fusion Using Co-attention Mechanism for Speaker recognition
- **分类: eess.AS**

- **简介: 该论文属说话人识别任务，旨在解决传统特征融合方法忽略幅度与相位特征互补性的问题。提出基于共注意力机制的特征级融合框架，分别提取幅度和相位高阶特征，通过共注意力动态加权融合，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2510.15659v1](http://arxiv.org/pdf/2510.15659v1)**

> **作者:** Rongfeng Su; Mengjie Du; Xiaokang Liu; Lan Wang; Nan Yan
>
> **摘要:** Phase-based features related to vocal source characteristics can be incorporated into magnitude-based speaker recognition systems to improve the system performance. However, traditional feature-level fusion methods typically ignore the unique contributions of speaker semantics in the magnitude and phase domains. To address this issue, this paper proposed a feature-level fusion framework using the co-attention mechanism for speaker recognition. The framework consists of two separate sub-networks for the magnitude and phase domains respectively. Then, the intermediate high-level outputs of both domains are fused by the co-attention mechanism before a pooling layer. A correlation matrix from the co-attention module is supposed to re-assign the weights for dynamically scaling contributions in the magnitude and phase domains according to different pronunciations. Experiments on VoxCeleb showed that the proposed feature-level fusion strategy using the co-attention mechanism gave the Top-1 accuracy of 97.20%, outperforming the state-of-the-art system with 0.82% absolutely, and obtained EER reduction of 0.45% compared to single feature system using FBank.
>
---
#### [new 006] DroneAudioset: An Audio Dataset for Drone-based Search and Rescue
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文针对无人机搜救中音频感知受自身噪声干扰的问题，构建了真实、多样的无人机音频数据集DroneAudioset，涵盖多种机型、环境与信噪比，支持噪声抑制和人类存在检测方法的研究，推动无人机听觉系统的开发与评估。**

- **链接: [http://arxiv.org/pdf/2510.15383v1](http://arxiv.org/pdf/2510.15383v1)**

> **作者:** Chitralekha Gupta; Soundarya Ramesh; Praveen Sasikumar; Kian Peen Yeo; Suranga Nanayakkara
>
> **备注:** Accepted in Neurips (Datasets and Benchmarks Track) 2025. The first two authors are equal contributors
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) or drones, are increasingly used in search and rescue missions to detect human presence. Existing systems primarily leverage vision-based methods which are prone to fail under low-visibility or occlusion. Drone-based audio perception offers promise but suffers from extreme ego-noise that masks sounds indicating human presence. Existing datasets are either limited in diversity or synthetic, lacking real acoustic interactions, and there are no standardized setups for drone audition. To this end, we present DroneAudioset (The dataset is publicly available at https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/ under the MIT license), a comprehensive drone audition dataset featuring 23.5 hours of annotated recordings, covering a wide range of signal-to-noise ratios (SNRs) from -57.2 dB to -2.5 dB, across various drone types, throttles, microphone configurations as well as environments. The dataset enables development and systematic evaluation of noise suppression and classification methods for human-presence detection under challenging conditions, while also informing practical design considerations for drone audition systems, such as microphone placement trade-offs, and development of drone noise-aware audio processing. This dataset is an important step towards enabling design and deployment of drone-audition systems.
>
---
#### [new 007] Towards Blind Data Cleaning: A Case Study in Music Source Separation
- **分类: eess.AS**

- **简介: 该论文研究音乐源分离中的盲数据清洗，旨在解决训练数据含未知噪声导致性能下降的问题。提出两种无需知晓噪声类型的清洗方法：基于遗忘的数据归因和Fréchet音频距离，有效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.15409v1](http://arxiv.org/pdf/2510.15409v1)**

> **作者:** Azalea Gui; Woosung Choi; Junghyun Koo; Kazuki Shimada; Takashi Shibuya; Joan Serrà; Wei-Hsiang Liao; Yuki Mitsufuji
>
> **备注:** Submitted to IEEE ICASSP 2026
>
> **摘要:** The performance of deep learning models for music source separation heavily depends on training data quality. However, datasets are often corrupted by difficult-to-detect artifacts such as audio bleeding and label noise. Since the type and extent of contamination are typically unknown, cleaning methods targeting specific corruptions are often impractical. This paper proposes and evaluates two distinct, noise-agnostic data cleaning methods to address this challenge. The first approach uses data attribution via unlearning to identify and filter out training samples that contribute the least to producing clean outputs. The second leverages the Fr\'echet Audio Distance to measure and remove samples that are perceptually dissimilar to a small and trusted clean reference set. On a dataset contaminated with a simulated distribution of real-world noise, our unlearning-based methods produced a cleaned dataset and a corresponding model that outperforms both the original contaminated data and the small clean reference set used for cleaning. This result closes approximately 66.7\% of the performance gap between the contaminated baseline and a model trained on the same dataset without any contamination. Unlike methods tailored for specific artifacts, our noise-agnostic approaches offer a more generic and broadly applicable solution for curating high-quality training data.
>
---
#### [new 008] LongCat-Audio-Codec: An Audio Tokenizer and Detokenizer Solution Designed for Speech Large Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出LongCat-Audio-Codec，面向语音大模型的音频编解码任务，解决低比特率下高质量语音合成与语义建模的平衡问题。采用解耦架构与多阶段训练，实现低延迟、高保真、极低码率（0.43–0.87 kbps）的音频编码与解码。**

- **链接: [http://arxiv.org/pdf/2510.15227v1](http://arxiv.org/pdf/2510.15227v1)**

> **作者:** Xiaohan Zhao; Hongyu Xiang; Shengze Ye; Song Li; Zhengkun Tian; Guanyu Chen; Ke Ding; Guanglu Wan
>
> **摘要:** This paper presents LongCat-Audio-Codec, an audio tokenizer and detokenizer solution designed for industrial grade end-to-end speech large language models. By leveraging a decoupled model architecture and a multistage training strategy, LongCat-Audio-Codec exhibits robust semantic modeling capabilities, flexible acoustic feature extraction capabilities, and low-latency streaming synthesis capabilities. It encodes speech at an ultra-low frame rate of 16.67 Hz, with a minimum bitrate of 0.43 kbps and a maximum bitrate of 0.87 kbps. Evaluation results demonstrate that LongCat-Audio-Codec achieves strong speech intelligibility and is capable of synthesizing highquality speech at low bitrate, thus effectively balancing coding efficiency and decoding quality. The inference code and model checkpoints of LongCat-Audio-Codec are available at: https://github.com/meituan-longcat/LongCat-Audio-Codec.
>
---
#### [new 009] Sound Clouds: Exploring ambient intelligence in public spaces to elicit deep human experience of awe, wonder, and beauty
- **分类: cs.HC; cs.MM; cs.SD**

- **简介: 该论文探讨如何通过环境智能（AmI）在公共空间中激发人类对敬畏、惊奇与美的深层体验。作者设计并实现了“声音云”艺术装置，以互动球体生成实时音乐，探索具有情感唤起能力的公共AmI系统。**

- **链接: [http://arxiv.org/pdf/2510.15865v1](http://arxiv.org/pdf/2510.15865v1)**

> **作者:** Chengzhi Zhang; Dashiel Carrera; Daksh Kapoor; Jasmine Kaur; Jisu Kim; Brian Magerko
>
> **备注:** 4 pages, Artwork accepted by NeurIPS Creative AI Track 2025
>
> **摘要:** While the ambient intelligence (AmI) systems we encounter in our daily lives, including security monitoring and energy-saving systems, typically serve pragmatic purposes, we wonder how we can design and implement ambient artificial intelligence experiences in public spaces that elicit deep human feelings of awe, wonder, and beauty. As a manifestation, we introduce Sound Clouds, an immersive art installation that generates live music based on participants' interaction with several human-height spheres. Our installation serves as a provocation into future ambient intelligence that provokes, not limits, the future possibilities of AmI.
>
---
#### [new 010] Extending Audio Context for Long-Form Understanding in Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文研究长音频理解任务，解决大音频语言模型因音频上下文短导致的长时理解受限问题。提出无需训练的Partial YaRN及训练策略VLAT，分别通过调整音频位置编码和模拟多长度音频增强长上下文能力。**

- **链接: [http://arxiv.org/pdf/2510.15231v1](http://arxiv.org/pdf/2510.15231v1)**

> **作者:** Yuatyong Chaichana; Pittawat Taveekitworachai; Warit Sirichotedumrong; Potsawee Manakul; Kunat Pipatanakul
>
> **摘要:** Large Audio-Language Models (LALMs) are often constrained by short audio context windows, even when their text backbones support long contexts, limiting long-form audio understanding. Prior work has introduced context-extension methods (e.g. YaRN) on unimodal LLMs, yet their application to LALMs remains unexplored. First, building on RoPE-based context extension, we introduce Partial YaRN, a training-free, audio-only extension method that modifies only audio token positions, leaving text positions intact to preserve the base LLM's text capabilities. Second, we propose Virtual Longform Audio Training (VLAT), a training strategy that extends Partial YaRN into a training-time positional augmentation. VLAT simulates diverse audio lengths during training, enabling generalization to inputs far longer than those seen in training and improving robustness for long-context audio understanding. Our experiments on SALMONN and Qwen2-Audio show that Partial YaRN outperforms the original models across wide range of settings, and VLAT training strategy provides substantial improvement, achieving strong performance on long audio of unseen lengths.
>
---
## 更新

#### [replaced 001] Beat Tracking as Object Detection
- **分类: cs.SD; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.14391v2](http://arxiv.org/pdf/2510.14391v2)**

> **作者:** Jaehoon Ahn; Moon-Ryul Jung
>
> **备注:** 11 pages, 4 figures, 5 tables
>
> **摘要:** Recent beat and downbeat tracking models (e.g., RNNs, TCNs, Transformers) output frame-level activations. We propose reframing this task as object detection, where beats and downbeats are modeled as temporal "objects." Adapting the FCOS detector from computer vision to 1D audio, we replace its original backbone with WaveBeat's temporal feature extractor and add a Feature Pyramid Network to capture multi-scale temporal patterns. The model predicts overlapping beat/downbeat intervals with confidence scores, followed by non-maximum suppression (NMS) to select final predictions. This NMS step serves a similar role to DBNs in traditional trackers, but is simpler and less heuristic. Evaluated on standard music datasets, our approach achieves competitive results, showing that object detection techniques can effectively model musical beats with minimal adaptation.
>
---
#### [replaced 002] Summarizing Speech: A Comprehensive Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.08024v3](http://arxiv.org/pdf/2504.08024v3)**

> **作者:** Fabian Retkowski; Maike Züfle; Andreas Sudmann; Dinah Pfau; Shinji Watanabe; Jan Niehues; Alexander Waibel
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Speech summarization has become an essential tool for efficiently managing and accessing the growing volume of spoken and audiovisual content. However, despite its increasing importance, speech summarization remains loosely defined. The field intersects with several research areas, including speech recognition, text summarization, and specific applications like meeting summarization. This survey not only examines existing datasets and evaluation protocols, which are crucial for assessing the quality of summarization approaches, but also synthesizes recent developments in the field, highlighting the shift from traditional systems to advanced models like fine-tuned cascaded architectures and end-to-end solutions. In doing so, we surface the ongoing challenges, such as the need for realistic evaluation benchmarks, multilingual datasets, and long-context handling.
>
---
#### [replaced 003] DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.19687v2](http://arxiv.org/pdf/2505.19687v2)**

> **作者:** Deok-Hyeon Cho; Hyung-Seok Oh; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Cross-speaker emotion transfer in speech synthesis relies on extracting speaker-independent emotion embeddings for accurate emotion modeling without retaining speaker traits. However, existing timbre compression methods fail to fully separate speaker and emotion characteristics, causing speaker leakage and degraded synthesis quality. To address this, we propose DiEmo-TTS, a self-supervised distillation method to minimize emotional information loss and preserve speaker identity. We introduce cluster-driven sampling and information perturbation to preserve emotion while removing irrelevant factors. To facilitate this process, we propose an emotion clustering and matching approach using emotional attribute prediction and speaker embeddings, enabling generalization to unlabeled data. Additionally, we designed a dual conditioning transformer to integrate style features better. Experimental results confirm the effectiveness of our method in learning speaker-irrelevant emotion embeddings.
>
---
#### [replaced 004] MRSAudio: A Large-Scale Multimodal Recorded Spatial Audio Dataset with Refined Annotations
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.10396v3](http://arxiv.org/pdf/2510.10396v3)**

> **作者:** Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Xintong Hu; Yu Zhang; Li Tang; Rui Yang; Han Wang; Zongbao Zhang; Yuhan Wang; Yixuan Chen; Hankun Xu; Ke Xu; Pengfei Fan; Zhetao Chen; Yanhao Yu; Qiange Huang; Fei Wu; Zhou Zhao
>
> **备注:** 24 pages
>
> **摘要:** Humans rely on multisensory integration to perceive spatial environments, where auditory cues enable sound source localization in three-dimensional space. Despite the critical role of spatial audio in immersive technologies such as VR/AR, most existing multimodal datasets provide only monaural audio, which limits the development of spatial audio generation and understanding. To address these challenges, we introduce MRSAudio, a large-scale multimodal spatial audio dataset designed to advance research in spatial audio understanding and generation. MRSAudio spans four distinct components: MRSLife, MRSSpeech, MRSMusic, and MRSSing, covering diverse real-world scenarios. The dataset includes synchronized binaural and ambisonic audio, exocentric and egocentric video, motion trajectories, and fine-grained annotations such as transcripts, phoneme boundaries, lyrics, scores, and prompts. To demonstrate the utility and versatility of MRSAudio, we establish five foundational tasks: audio spatialization, and spatial text to speech, spatial singing voice synthesis, spatial music generation and sound event localization and detection. Results show that MRSAudio enables high-quality spatial modeling and supports a broad range of spatial audio research. Demos and dataset access are available at https://mrsaudio.github.io.
>
---
#### [replaced 005] Improving Inference-Time Optimisation for Vocal Effects Style Transfer with a Gaussian Prior
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11315v2](http://arxiv.org/pdf/2505.11315v2)**

> **作者:** Chin-Yun Yu; Marco A. Martínez-Ramírez; Junghyun Koo; Wei-Hsiang Liao; Yuki Mitsufuji; György Fazekas
>
> **备注:** Published at WASPAA 2025
>
> **摘要:** Style Transfer with Inference-Time Optimisation (ST-ITO) is a recent approach for transferring the applied effects of a reference audio to an audio track. It optimises the effect parameters to minimise the distance between the style embeddings of the processed audio and the reference. However, this method treats all possible configurations equally and relies solely on the embedding space, which can result in unrealistic configurations or biased outcomes. We address this pitfall by introducing a Gaussian prior derived from the DiffVox vocal preset dataset over the parameter space. The resulting optimisation is equivalent to maximum-a-posteriori estimation. Evaluations on vocal effects transfer on the MedleyDB dataset show significant improvements across metrics compared to baselines, including a blind audio effects estimator, nearest-neighbour approaches, and uncalibrated ST-ITO. The proposed calibration reduces the parameter mean squared error by up to 33% and more closely matches the reference style. Subjective evaluations with 16 participants confirm the superiority of our method in limited data regimes. This work demonstrates how incorporating prior knowledge at inference time enhances audio effects transfer, paving the way for more effective and realistic audio processing systems.
>
---
#### [replaced 006] A New Time Series Similarity Measure and Its Smart Grid Applications
- **分类: eess.SP; eess.AS**

- **链接: [http://arxiv.org/pdf/2310.12399v2](http://arxiv.org/pdf/2310.12399v2)**

> **作者:** Rui Yuan; Hossein Ranjbar; S. Ali Pourmousavi; Wen L. Soong; Andrew J. Black; Jon A. R. Liisberg; Julian Lemos-Vinasco
>
> **备注:** 6 pages, 5 figures conference
>
> **摘要:** Many smart grid applications involve data mining, clustering, classification, identification, and anomaly detection, among others. These applications primarily depend on the measurement of similarity, which is the distance between different time series or subsequences of a time series. The commonly used time series distance measures, namely Euclidean Distance (ED) and Dynamic Time Warping (DTW), do not quantify the flexible nature of electricity usage data in terms of temporal dynamics. As a result, there is a need for a new distance measure that can quantify both the amplitude and temporal changes of electricity time series for smart grid applications, e.g., demand response and load profiling. This paper introduces a novel distance measure to compare electricity usage patterns. The method consists of two phases that quantify the effort required to reshape one time series into another, considering both amplitude and temporal changes. The proposed method is evaluated against ED and DTW using real-world data in three smart grid applications. Overall, the proposed measure outperforms ED and DTW in accurately identifying the best load scheduling strategy, anomalous days with irregular electricity usage, and determining electricity users' behind-the-meter (BTM) equipment.
>
---
#### [replaced 007] EmoSphere-SER: Enhancing Speech Emotion Recognition Through Spherical Representation with Auxiliary Classification
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.19693v2](http://arxiv.org/pdf/2505.19693v2)**

> **作者:** Deok-Hyeon Cho; Hyung-Seok Oh; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Speech emotion recognition predicts a speaker's emotional state from speech signals using discrete labels or continuous dimensions such as arousal, valence, and dominance (VAD). We propose EmoSphere-SER, a joint model that integrates spherical VAD region classification to guide VAD regression for improved emotion prediction. In our framework, VAD values are transformed into spherical coordinates that are divided into multiple spherical regions, and an auxiliary classification task predicts which spherical region each point belongs to, guiding the regression process. Additionally, we incorporate a dynamic weighting scheme and a style pooling layer with multi-head self-attention to capture spectral and temporal dynamics, further boosting performance. This combined training strategy reinforces structured learning and improves prediction consistency. Experimental results show that our approach exceeds baseline methods, confirming the validity of the proposed framework.
>
---
#### [replaced 008] BandCondiNet: Parallel Transformers-based Conditional Popular Music Generation with Multi-View Features
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.10462v2](http://arxiv.org/pdf/2407.10462v2)**

> **作者:** Jing Luo; Xinyu Yang; Dorien Herremans
>
> **备注:** To appear in ESWA. Demo page: https://chinglohsiu.github.io/files/bandcondinet.html
>
> **摘要:** Conditional music generation offers significant advantages in terms of user convenience and control, presenting great potential in AI-generated content research. However, building conditional generative systems for multitrack popular songs presents three primary challenges: insufficient fidelity of input conditions, poor structural modeling, and inadequate inter-track harmony learning in generative models. To address these issues, we propose BandCondiNet, a conditional model based on parallel Transformers, designed to process the multiple music sequences and generate high-quality multitrack samples. Specifically, we propose multi-view features across time and instruments as high-fidelity conditions. Moreover, we propose two specialized modules for BandCondiNet: Structure Enhanced Attention (SEA) to strengthen the musical structure, and Cross-Track Transformer (CTT) to enhance inter-track harmony. We conducted both objective and subjective evaluations on two popular music datasets with different sequence lengths. Objective results on the shorter dataset show that BandCondiNet outperforms other conditional models in 9 out of 10 metrics related to fidelity and inference speed, with the exception of Chord Accuracy. On the longer dataset, BandCondiNet surpasses all conditional models across all 10 metrics. Subjective evaluations across four criteria reveal that BandCondiNet trained on the shorter dataset performs best in Richness and performs comparably to state-of-the-art models in the other three criteria, while significantly outperforming them across all criteria when trained on the longer dataset. To further expand the application scope of BandCondiNet, future work should focus on developing an advanced conditional model capable of adapting to more user-friendly input conditions and supporting flexible instrumentation.
>
---
#### [replaced 009] Benchmarking Fake Voice Detection in the Fake Voice Generation Arms Race
- **分类: cs.SD; cs.CR; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.06544v2](http://arxiv.org/pdf/2510.06544v2)**

> **作者:** Xutao Mao; Ke Li; Cameron Baird; Ezra Xuanru Tao; Dan Lin
>
> **摘要:** The rapid advancement of fake voice generation technology has ignited a race with detection systems, creating an urgent need to secure the audio ecosystem. However, existing benchmarks suffer from a critical limitation: they typically aggregate diverse fake voice samples into a single dataset for evaluation. This practice masks method-specific artifacts and obscures the varying performance of detectors against different generation paradigms, preventing a nuanced understanding of their true vulnerabilities. To address this gap, we introduce the first ecosystem-level benchmark that systematically evaluates the interplay between 17 state-of-the-art fake voice generators and 8 leading detectors through a novel one-to-one evaluation protocol. This fine-grained analysis exposes previously hidden vulnerabilities and sensitivities that are missed by traditional aggregated testing. We also propose unified scoring systems to quantify both the evasiveness of generators and the robustness of detectors, enabling fair and direct comparisons. Our extensive cross-domain evaluation reveals that modern generators, particularly those based on neural audio codecs and flow matching, consistently evade top-tier detectors. We found that no single detector is universally robust; their effectiveness varies dramatically depending on the generator's architecture, highlighting a significant generalization gap in current defenses. This work provides a more realistic assessment of the threat landscape and offers actionable insights for building the next generation of detection systems.
>
---
