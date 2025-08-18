# 音频 cs.SD;  eess.SP

- **最新发布 13 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Motive-level Analysis of Form-functions Association in Korean Folk song
- **分类: cs.SD; cs.CY**

- **简介: 论文提出基于微调模型的自动动机分割方法，利用歌词与动机边界数据提取结构特征，分析不同社会功能下的结构差异，提供可扩展的定量分析工具。**

- **链接: [http://arxiv.org/pdf/2508.10472v1](http://arxiv.org/pdf/2508.10472v1)**

> **作者:** Danbinaerin Han; Dasaem Jeong; Juhan Nam
>
> **摘要:** Computational analysis of folk song audio is challenging due to structural irregularities and the need for manual annotation. We propose a method for automatic motive segmentation in Korean folk songs by fine-tuning a speech transcription model on audio lyric with motif boundary annotation. Applying this to 856 songs, we extracted motif count and duration entropy as structural features. Statistical analysis revealed that these features vary systematically according to the social function of the songs. Songs associated with collective labor, for instance, showed different structural patterns from those for entertainment or personal settings. This work offers a scalable approach for quantitative structural analysis of oral music traditions.
>
---
#### [new 002] Fake Speech Wild: Detecting Deepfake Speech on Social Media Platform
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出Fake Speech Wild（FSW）数据集，针对深度伪造语音在社交平台的检测问题，评估跨域场景下CMs性能，通过数据增强优化检测效果，实现平均EER 3.54%。**

- **链接: [http://arxiv.org/pdf/2508.10559v1](http://arxiv.org/pdf/2508.10559v1)**

> **作者:** Yuankun Xie; Ruibo Fu; Xiaopeng Wang; Zhiyong Wang; Ya Li; Zhengqi Wen; Haonnan Cheng; Long Ye
>
> **摘要:** The rapid advancement of speech generation technology has led to the widespread proliferation of deepfake speech across social media platforms. While deepfake audio countermeasures (CMs) achieve promising results on public datasets, their performance degrades significantly in cross-domain scenarios. To advance CMs for real-world deepfake detection, we first propose the Fake Speech Wild (FSW) dataset, which includes 254 hours of real and deepfake audio from four different media platforms, focusing on social media. As CMs, we establish a benchmark using public datasets and advanced selfsupervised learning (SSL)-based CMs to evaluate current CMs in real-world scenarios. We also assess the effectiveness of data augmentation strategies in enhancing CM robustness for detecting deepfake speech on social media. Finally, by augmenting public datasets and incorporating the FSW training set, we significantly advanced real-world deepfake audio detection performance, achieving an average equal error rate (EER) of 3.54% across all evaluation sets.
>
---
#### [new 003] Advances in Speech Separation: Techniques, Challenges, and Future Trends
- **分类: cs.SD; eess.AS**

- **简介: 本文综述语音分离技术，解决嘈杂环境下的多说话人分离问题，系统分析DNN架构、训练范式及挑战，提出跨领域技术趋势与评估方法，为研究者提供全面参考。**

- **链接: [http://arxiv.org/pdf/2508.10830v1](http://arxiv.org/pdf/2508.10830v1)**

> **作者:** Kai Li; Guo Chen; Wendi Sang; Yi Luo; Zhuo Chen; Shuai Wang; Shulin He; Zhong-Qiu Wang; Andong Li; Zhiyong Wu; Xiaolin Hu
>
> **备注:** 34 pages, 10 figures
>
> **摘要:** The field of speech separation, addressing the "cocktail party problem", has seen revolutionary advances with DNNs. Speech separation enhances clarity in complex acoustic environments and serves as crucial pre-processing for speech recognition and speaker recognition. However, current literature focuses narrowly on specific architectures or isolated approaches, creating fragmented understanding. This survey addresses this gap by providing systematic examination of DNN-based speech separation techniques. Our work differentiates itself through: (I) Comprehensive perspective: We systematically investigate learning paradigms, separation scenarios with known/unknown speakers, comparative analysis of supervised/self-supervised/unsupervised frameworks, and architectural components from encoders to estimation strategies. (II) Timeliness: Coverage of cutting-edge developments ensures access to current innovations and benchmarks. (III) Unique insights: Beyond summarization, we evaluate technological trajectories, identify emerging patterns, and highlight promising directions including domain-robust frameworks, efficient architectures, multimodal integration, and novel self-supervised paradigms. (IV) Fair evaluation: We provide quantitative evaluations on standard datasets, revealing true capabilities and limitations of different methods. This comprehensive survey serves as an accessible reference for experienced researchers and newcomers navigating speech separation's complex landscape.
>
---
#### [new 004] Facilitating Personalized TTS for Dysarthric Speakers Using Knowledge Anchoring and Curriculum Learning
- **分类: cs.SD**

- **简介: 论文提出一种基于知识锚定与课程学习的个性化TTS方法，解决失语者语音数据不足及合成质量差问题，通过音频增强提升语音自然性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.10412v1](http://arxiv.org/pdf/2508.10412v1)**

> **作者:** Yejin Jeon; Solee Im; Youngjae Kim; Gary Geunbae Lee
>
> **备注:** Interspeech 2025
>
> **摘要:** Dysarthric speakers experience substantial communication challenges due to impaired motor control of the speech apparatus, which leads to reduced speech intelligibility. This creates significant obstacles in dataset curation since actual recording of long, articulate sentences for the objective of training personalized TTS models becomes infeasible. Thus, the limited availability of audio data, in addition to the articulation errors that are present within the audio, complicates personalized speech synthesis for target dysarthric speaker adaptation. To address this, we frame the issue as a domain transfer task and introduce a knowledge anchoring framework that leverages a teacher-student model, enhanced by curriculum learning through audio augmentation. Experimental results show that the proposed zero-shot multi-speaker TTS model effectively generates synthetic speech with markedly reduced articulation errors and high speaker fidelity, while maintaining prosodic naturalness.
>
---
#### [new 005] Alternating Approach-Putt Models for Multi-Stage Speech Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出PuttNet模型，通过交替使用语音增强与Putt模型缓解伪影，提升多阶段语音质量，采用PESQ等指标验证效果。**

- **链接: [http://arxiv.org/pdf/2508.10436v1](http://arxiv.org/pdf/2508.10436v1)**

> **作者:** Iksoon Jeong; Kyung-Joong Kim; Kang-Hun Ahn
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Speech enhancement using artificial neural networks aims to remove noise from noisy speech signals while preserving the speech content. However, speech enhancement networks often introduce distortions to the speech signal, referred to as artifacts, which can degrade audio quality. In this work, we propose a post-processing neural network designed to mitigate artifacts introduced by speech enhancement models. Inspired by the analogy of making a `Putt' after an `Approach' in golf, we name our model PuttNet. We demonstrate that alternating between a speech enhancement model and the proposed Putt model leads to improved speech quality, as measured by perceptual quality scores (PESQ), objective intelligibility (STOI), and background noise intrusiveness (CBAK) scores. Furthermore, we illustrate with graphical analysis why this alternating Approach outperforms repeated application of either model alone.
>
---
#### [new 006] No Free Lunch from Audio Pretraining in Bioacoustics: A Benchmark Study of Embeddings
- **分类: cs.SD; cs.AI**

- **简介: 论文探讨生物声学中音频预训练模型嵌入向量的性能，揭示未经微调模型在任务中表现不佳，通过对比实验指出微调必要性，并强调对嵌入向量的后续检查。**

- **链接: [http://arxiv.org/pdf/2508.10230v1](http://arxiv.org/pdf/2508.10230v1)**

> **作者:** Chenggang Chen; Zhiyu Yang
>
> **摘要:** Bioacoustics, the study of animal sounds, offers a non-invasive method to monitor ecosystems. Extracting embeddings from audio-pretrained deep learning (DL) models without fine-tuning has become popular for obtaining bioacoustic features for tasks. However, a recent benchmark study reveals that while fine-tuned audio-pretrained VGG and transformer models achieve state-of-the-art performance in some tasks, they fail in others. This study benchmarks 11 DL models on the same tasks by reducing their learned embeddings' dimensionality and evaluating them through clustering. We found that audio-pretrained DL models 1) without fine-tuning even underperform fine-tuned AlexNet, 2) both with and without fine-tuning fail to separate the background from labeled sounds, but ResNet does, and 3) outperform other models when fewer background sounds are included during fine-tuning. This study underscores the necessity of fine-tuning audio-pretrained models and checking the embeddings after fine-tuning. Our codes are available: https://github.com/NeuroscienceAI/Audio\_Embeddings
>
---
#### [new 007] Dynamic Synchronization and Resonance as a Universal Origin of 1/f Fluctuations -- Amplitude Modulation Across Music and Nature
- **分类: cs.SD**

- **简介: 论文提出动态同步与共振机制解释1/f波动，通过幅度调制与解调揭示其普遍性，结合Kuramoto框架与频谱积累模型，验证跨领域（音乐、地震、天体）的共性，强调解调为生成1/f波动的通用路径。**

- **链接: [http://arxiv.org/pdf/2508.10049v1](http://arxiv.org/pdf/2508.10049v1)**

> **作者:** Akika Nakamichi; Izumi Uesaka; Masahiro Morikawa
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** We propose a universal physical mechanism for the emergence of 1/f fluctuations, observed across a wide range of systems. In particular, we verify this on acoustic cases. The mechanism is based on amplitude modulation (AM) and demodulation (DM), where the 1/f spectral law arises not in the raw waveform but in its demodulated amplitude envelope. Two distinct yet complementary processes generate the required AM: (i) stochastic synchronization among oscillators, modeled via an extended Kuramoto framework that captures perpetual synchronization-desynchronization cycles, and (ii) frequency-selective resonance, modeled by spectral accumulation of eigenmodes in acoustic or structural environments. Numerical simulations demonstrate that both mechanisms, acting separately or in combination, robustly produce 1/f spectra over several decades when DM is applied, and that the classical Kuramoto critical point is not necessary for their emergence. We demonstrate the cross-domain relevance of this AM/DM framework through analyses of musical performances, seismic records, and astrophysical time series, revealing a common underlying structure. This work establishes demodulation as a general route to 1/f fluctuations, providing a simple and scalable explanation for its ubiquity in both natural and engineered systems. Keywords: 1/f fluctuation, amplitude modulation, synchronization, resonance, Kuramoto model, music, natural noise, demodulation
>
---
#### [new 008] Whisper Smarter, not Harder: Adversarial Attack on Partial Suppression
- **分类: cs.SD; cs.CR; cs.LG; eess.AS**

- **简介: 论文研究对抗攻击对部分抑制的鲁棒性，解决提升攻击不可感知性问题，通过调整优化目标提出低通滤波器防御。**

- **链接: [http://arxiv.org/pdf/2508.09994v1](http://arxiv.org/pdf/2508.09994v1)**

> **作者:** Zheng Jie Wong; Bingquan Shen
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Currently, Automatic Speech Recognition (ASR) models are deployed in an extensive range of applications. However, recent studies have demonstrated the possibility of adversarial attack on these models which could potentially suppress or disrupt model output. We investigate and verify the robustness of these attacks and explore if it is possible to increase their imperceptibility. We additionally find that by relaxing the optimisation objective from complete suppression to partial suppression, we can further decrease the imperceptibility of the attack. We also explore possible defences against these attacks and show a low-pass filter defence could potentially serve as an effective defence.
>
---
#### [new 009] A dataset and model for recognition of audiologically relevant environments for hearing aids: AHEAD-DS and YAMNet+
- **分类: cs.SD; eess.AS**

- **简介: 论文提出AHEAD-DS数据集及YAMNet+模型，解决现有数据集不足与边缘部署难题，实现听力相关环境的实时声学场景识别。**

- **链接: [http://arxiv.org/pdf/2508.10360v1](http://arxiv.org/pdf/2508.10360v1)**

> **作者:** Henry Zhong; Jörg M. Buchholz; Julian Maclaren; Simon Carlile; Richard Lyon
>
> **摘要:** Scene recognition of audiologically relevant environments is important for hearing aids; however, it is challenging, in part because of the limitations of existing datasets. Datasets often lack public accessibility, completeness, or audiologically relevant labels, hindering systematic comparison of machine learning models. Deploying these models on resource-constrained edge devices presents another challenge. Our solution is two-fold: we leverage several open source datasets to create AHEAD-DS, a dataset designed for scene recognition of audiologically relevant environments, and introduce YAMNet+, a sound recognition model. AHEAD-DS aims to provide a standardised, publicly available dataset with consistent labels relevant to hearing aids, facilitating model comparison. YAMNet+ is designed for deployment on edge devices like smartphones connected to hearing devices, such as hearing aids and wireless earphones with hearing aid functionality; serving as a baseline model for sound-based scene recognition. YAMNet+ achieved a mean average precision of 0.83 and accuracy of 0.93 on the testing set of AHEAD-DS across fourteen categories of audiologically relevant environments. We found that applying transfer learning from the pretrained YAMNet model was essential. We demonstrated real-time sound-based scene recognition capabilities on edge devices by deploying YAMNet+ to an Android smartphone. Even with a Google Pixel 3 (a phone with modest specifications, released in 2018), the model processes audio with approximately 50ms of latency to load the model, and an approximate linear increase of 30ms per 1 second of audio. Our website and code https://github.com/Australian-Future-Hearing-Initiative .
>
---
#### [new 010] Layer-Wise Analysis of Self-Supervised Representations for Age and Gender Classification in Children's Speech
- **分类: eess.AS; cs.AI; cs.HC; cs.LG; cs.SD**

- **简介: 本文提出通过层间分析揭示儿童语音自监督表示的结构，解决年龄/性别分类中因语音特征多变导致的挑战，采用Wav2Vec2变体与PCA优化分类效果，显示早期层更有效捕捉说话者特征，支持儿童语音界面的针对性设计。**

- **链接: [http://arxiv.org/pdf/2508.10332v1](http://arxiv.org/pdf/2508.10332v1)**

> **作者:** Abhijit Sinha; Harishankar Kumar; Mohit Joshi; Hemant Kumar Kathania; Shrikanth Narayanan; Sudarsana Reddy Kadiri
>
> **备注:** Accepted at Workshop on Child Computer Interaction (WOCCI 2025)
>
> **摘要:** Children's speech presents challenges for age and gender classification due to high variability in pitch, articulation, and developmental traits. While self-supervised learning (SSL) models perform well on adult speech tasks, their ability to encode speaker traits in children remains underexplored. This paper presents a detailed layer-wise analysis of four Wav2Vec2 variants using the PFSTAR and CMU Kids datasets. Results show that early layers (1-7) capture speaker-specific cues more effectively than deeper layers, which increasingly focus on linguistic information. Applying PCA further improves classification, reducing redundancy and highlighting the most informative components. The Wav2Vec2-large-lv60 model achieves 97.14% (age) and 98.20% (gender) on CMU Kids; base-100h and large-lv60 models reach 86.05% and 95.00% on PFSTAR. These results reveal how speaker traits are structured across SSL model depth and support more targeted, adaptive strategies for child-aware speech interfaces.
>
---
#### [new 011] MCP2OSC: Parametric Control by Natural Language
- **分类: cs.HC; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出MCP2OSC，通过自然语言控制OSC参数，解决传统方法精度不足问题，利用LLM生成消息并提升人机协作效率。**

- **链接: [http://arxiv.org/pdf/2508.10414v1](http://arxiv.org/pdf/2508.10414v1)**

> **作者:** Yuan-Yi Fan
>
> **摘要:** Text prompts enable intuitive content creation but may fall short in achieving high precision for intricate tasks; knob or slider controls offer precise adjustments at the cost of increased complexity. To address the gap between knobs and prompts, a new MCP (Model Context Protocol) server and a unique set of prompt design criteria are presented to enable exploring parametric OSC (OpenSoundControl) control by natural language prompts. Demonstrated by 14 practical QA examples with best practices and the generalized prompt templates, this study finds Claude integrated with the MCP2OSC server effective in generating OSC messages by natural language, interpreting, searching, and visualizing OSC messages, validating and debugging OSC messages, and managing OSC address patterns. MCP2OSC enhances human-machine collaboration by leveraging LLM (Large Language Model) to handle intricate OSC development tasks, and by empowering human creativity with an intuitive language interface featuring flexible precision controls: a prompt-based OSC tool. This study provides a novel perspective on the creative MCP application at the network protocol level by utilizing LLM's strength in directly processing and generating human-readable OSC messages. The results suggest its potential for a LLM-based universal control mechanism for multimedia devices.
>
---
#### [new 012] Beyond Hard Sharing: Efficient Multi-Task Speech-to-Text Modeling with Supervised Mixture of Experts
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出基于监督的混合专家模型（S-MoE）用于多任务语音转文字建模，解决硬参数共享导致的任务干扰问题，提升模型效率并实现ASR与ST联合处理，效果较传统方法提升6.35%。**

- **链接: [http://arxiv.org/pdf/2508.10009v1](http://arxiv.org/pdf/2508.10009v1)**

> **作者:** Hojun Jin; Eunsoo Hong; Ziwon Hyung; Sungjun Lim; Seungjin Lee; Keunseok Cho
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Hard-parameter sharing is a common strategy to train a single model jointly across diverse tasks. However, this often leads to task interference, impeding overall model performance. To address the issue, we propose a simple yet effective Supervised Mixture of Experts (S-MoE). Unlike traditional Mixture of Experts models, S-MoE eliminates the need for training gating functions by utilizing special guiding tokens to route each task to its designated expert. By assigning each task to a separate feedforward network, S-MoE overcomes the limitations of hard-parameter sharing. We further apply S-MoE to a speech-to-text model, enabling the model to process mixed-bandwidth input while jointly performing automatic speech recognition (ASR) and speech translation (ST). Experimental results demonstrate the effectiveness of the proposed S-MoE, achieving a 6.35% relative improvement in Word Error Rate (WER) when applied to both the encoder and decoder.
>
---
#### [new 013] Ensembling Synchronisation-based and Face-Voice Association Paradigms for Robust Active Speaker Detection in Egocentric Recordings
- **分类: cs.MM; cs.SD**

- **简介: 论文提出融合同步依赖与无关模型的简单方法，解决第一人称视频中因遮挡、运动模糊和音频干扰导致的时序同步难题，通过加权平均提升鲁棒性，并优化FVA组件预处理，实现70.2% mAP的性能。**

- **链接: [http://arxiv.org/pdf/2508.10580v1](http://arxiv.org/pdf/2508.10580v1)**

> **作者:** Jason Clarke; Yoshihiko Gotoh; Stefan Goetze
>
> **备注:** Accepted to SPECOM 2025, 13 pages, 4 figures. To appear in the Proceedings of the 27th International Conference on Speech and Computer (SPECOM) 2025, October 13-14, 2025, Szeged, Hungary
>
> **摘要:** Audiovisual active speaker detection (ASD) in egocentric recordings is challenged by frequent occlusions, motion blur, and audio interference, which undermine the discernability of temporal synchrony between lip movement and speech. Traditional synchronisation-based systems perform well under clean conditions but degrade sharply in first-person recordings. Conversely, face-voice association (FVA)-based methods forgo synchronisation modelling in favour of cross-modal biometric matching, exhibiting robustness to transient visual corruption but suffering when overlapping speech or front-end segmentation errors occur. In this paper, a simple yet effective ensemble approach is proposed to fuse synchronisation-dependent and synchronisation-agnostic model outputs via weighted averaging, thereby harnessing complementary cues without introducing complex fusion architectures. A refined preprocessing pipeline for the FVA-based component is also introduced to optimise ensemble integration. Experiments on the Ego4D-AVD validation set demonstrate that the ensemble attains 70.2% and 66.7% mean Average Precision (mAP) with TalkNet and Light-ASD backbones, respectively. A qualitative analysis stratified by face image quality and utterance masking prevalence further substantiates the complementary strengths of each component.
>
---
## 更新

#### [replaced 001] Speech Enhancement based on cascaded two flow
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.06842v2](http://arxiv.org/pdf/2508.06842v2)**

> **作者:** Seonggyu Lee; Sein Cheong; Sangwook Han; Kihyuk Kim; Jong Won Shin
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Speech enhancement (SE) based on diffusion probabilistic models has exhibited impressive performance, while requiring a relatively high number of function evaluations (NFE). Recently, SE based on flow matching has been proposed, which showed competitive performance with a small NFE. Early approaches adopted the noisy speech as the only conditioning variable. There have been other approaches which utilize speech enhanced with a predictive model as another conditioning variable and to sample an initial value, but they require a separate predictive model on top of the generative SE model. In this work, we propose to employ an identical model based on flow matching for both SE and generating enhanced speech used as an initial starting point and a conditioning variable. Experimental results showed that the proposed method required the same or fewer NFEs even with two cascaded generative methods while achieving equivalent or better performances to the previous baselines.
>
---
#### [replaced 002] Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17538v2](http://arxiv.org/pdf/2505.17538v2)**

> **作者:** Leonora Vesterbacka; Faton Rekathati; Robin Kurtz; Justyna Sikora; Agnes Toftgård
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** This work presents a suite of fine-tuned Whisper models for Swedish, trained on a dataset of unprecedented size and variability for this mid-resourced language. As languages of smaller sizes are often underrepresented in multilingual training datasets, substantial improvements in performance can be achieved by fine-tuning existing multilingual models, as shown in this work. This work reports an overall improvement across model sizes compared to OpenAI's Whisper evaluated on Swedish. Most notably, we report an average 47% reduction in WER comparing our best performing model to OpenAI's whisper-large-v3, in evaluations across FLEURS, Common Voice, and NST.
>
---
#### [replaced 003] Marco-Voice Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02038v4](http://arxiv.org/pdf/2508.02038v4)**

> **作者:** Fengping Tian; Chenyang Lyu; Xuanfan Ni; Haoqin Sun; Qingjuan Li; Zhiqiang Qian; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively
>
> **摘要:** This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively.
>
---
#### [replaced 004] A Training-Free Approach for Music Style Transfer with Latent Diffusion Models
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.15913v2](http://arxiv.org/pdf/2411.15913v2)**

> **作者:** Heehwan Wang; Joonwoo Kwon; Sooyoung Kim; Shinjae Yoo; Yuewei Lin; Jiook Cha
>
> **备注:** Codes will be released upon acceptance
>
> **摘要:** Music style transfer enables personalized music creation by combining the structure of one piece with the stylistic characteristics of another. While recent approaches have explored text-conditioned generation and diffusion-based synthesis, most require extensive training, paired datasets, or detailed textual annotations. In this work, we introduce Stylus, a novel training-free framework for music style transfer that directly manipulates the self-attention layers of a pre-trained Latent Diffusion Model (LDM). Operating in the mel-spectrogram domain, Stylus transfers musical style by replacing key and value representations from the content audio with those of the style reference, without any fine-tuning. To enhance stylization quality and controllability, we further incorporate query preservation, CFG-inspired guidance scaling, multi-style interpolation, and phase-preserving reconstruction. Our method significantly improves perceptual quality and structural preservation compared to prior work, while remaining lightweight and easy to deploy. This work highlights the potential of diffusion-based attention manipulation for efficient, high-fidelity, and interpretable music generation-without training. Codes will be released upon acceptance.
>
---
#### [replaced 005] Evaluation of Speech Foundation Models for ASR on Child-Adult Conversations in Autism Diagnostic Sessions
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.16135v2](http://arxiv.org/pdf/2409.16135v2)**

> **作者:** Aditya Ashvin; Rimita Lahiri; Aditya Kommineni; Somer Bishop; Catherine Lord; Sudarsana Reddy Kadiri; Shrikanth Narayanan
>
> **备注:** Accepted at Workshop on Child Computer Interaction (WOCCI 2025)
>
> **摘要:** Reliable transcription of child-adult conversations in clinical settings is crucial for diagnosing developmental disorders like Autism. Recent advances in deep learning and availability of large scale transcribed data has led to development of speech foundation models that have shown dramatic improvements in ASR performance. However, their performance on conversational child-adult interactions remains underexplored. In this work, we provide a comprehensive evaluation of ASR performance on a dataset containing child-adult interactions from autism diagnostic sessions, using Whisper, Wav2Vec2, HuBERT, and WavLM. We find that speech foundation models show a noticeable performance drop (15-20% absolute WER) for child speech compared to adult speech in the conversational setting. Then, we fine-tune the best-performing zero-shot model (Whisper-large) using LoRA in a low-resource setting, yielding 8% and 13% absolute WER improvements for child and adult speech, respectively.
>
---
