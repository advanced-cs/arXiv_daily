# 音频 cs.SD;  eess.SP

- **最新发布 11 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] EffiFusion-GAN: Efficient Fusion Generative Adversarial Network for Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 本文提出EffiFusion-GAN，用于语音增强，通过多尺度深度可分离卷积、增强注意力机制和动态剪枝，提升效率与性能，在VoiceBank+DEMAND数据集上达到PESQ 3.45。**

- **链接: [http://arxiv.org/pdf/2508.14525v1](http://arxiv.org/pdf/2508.14525v1)**

> **作者:** Bin Wen; Tien-Ping Tan
>
> **摘要:** We introduce EffiFusion-GAN (Efficient Fusion Generative Adversarial Network), a lightweight yet powerful model for speech enhancement. The model integrates depthwise separable convolutions within a multi-scale block to capture diverse acoustic features efficiently. An enhanced attention mechanism with dual normalization and residual refinement further improves training stability and convergence. Additionally, dynamic pruning is applied to reduce model size while maintaining performance, making the framework suitable for resource-constrained environments. Experimental evaluation on the public VoiceBank+DEMAND dataset shows that EffiFusion-GAN achieves a PESQ score of 3.45, outperforming existing models under the same parameter settings.
>
---
#### [new 002] Systematic FAIRness Assessment of Open Voice Biomarker Datasets for Mental Health and Neurodegenerative Diseases
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文系统评估27个开放语音生物标志物数据集的FAIR性，揭示其在访问性、互操作性及重用性方面的不足，提出采用结构化元数据和FAIR合规仓库以提升临床转化效能。**

- **链接: [http://arxiv.org/pdf/2508.14089v1](http://arxiv.org/pdf/2508.14089v1)**

> **作者:** Ishaan Mahapatra; Nihar R. Mahapatra
>
> **备注:** To appear in the Proceedings of the 28th International Conference on Text, Speech and Dialogue (TSD 2025), Erlangen, Germany, August 25-28, 2025
>
> **摘要:** Voice biomarkers--human-generated acoustic signals such as speech, coughing, and breathing--are promising tools for scalable, non-invasive detection and monitoring of mental health and neurodegenerative diseases. Yet, their clinical adoption remains constrained by inconsistent quality and limited usability of publicly available datasets. To address this gap, we present the first systematic FAIR (Findable, Accessible, Interoperable, Reusable) evaluation of 27 publicly available voice biomarker datasets focused on these disease areas. Using the FAIR Data Maturity Model and a structured, priority-weighted scoring method, we assessed FAIRness at subprinciple, principle, and composite levels. Our analysis revealed consistently high Findability but substantial variability and weaknesses in Accessibility, Interoperability, and Reusability. Mental health datasets exhibited greater variability in FAIR scores, while neurodegenerative datasets were slightly more consistent. Repository choice also significantly influenced FAIRness scores. To enhance dataset quality and clinical utility, we recommend adopting structured, domain-specific metadata standards, prioritizing FAIR-compliant repositories, and routinely applying structured FAIR evaluation frameworks. These findings provide actionable guidance to improve dataset interoperability and reuse, thereby accelerating the clinical translation of voice biomarker technologies.
>
---
#### [new 003] BioSonix: Can Physics-Based Sonification Perceptualize Tissue Deformations From Tool Interactions?
- **分类: cs.SD; cs.HC; eess.AS**

- **简介: 该论文旨在解决手术中工具与软组织交互的感知难题，提出基于物理的声音转换方法（BioSonix），通过生物力学模拟与优化实验验证其有效性，实现复杂交互的听觉感知。**

- **链接: [http://arxiv.org/pdf/2508.14688v1](http://arxiv.org/pdf/2508.14688v1)**

> **作者:** Veronica Ruozzi; Sasan Matinfar; Laura Schütz; Benedikt Wiestler; Alberto Redaelli; Emiliano Votta; Nassir Navab
>
> **备注:** V. Ruozzi and S. Matinfar contributed equally to this work
>
> **摘要:** Perceptualizing tool interactions with deformable structures in surgical procedures remains challenging, as unimodal visualization techniques often fail to capture the complexity of these interactions due to constraints such as occlusion and limited depth perception. This paper presents a novel approach to augment tool navigation in mixed reality environments by providing auditory representations of tool-tissue dynamics, particularly for interactions with soft tissue. BioSonix, a physics-informed design framework, utilizes tissue displacements in 3D space to compute excitation forces for a sound model encoding tissue properties such as stiffness and density. Biomechanical simulations were employed to model particle displacements resulting from tool-tissue interactions, establishing a robust foundation for the method. An optimization approach was used to define configurations for capturing diverse interaction scenarios with varying tool trajectories. Experiments were conducted to validate the accuracy of the sound-displacement mappings. Additionally, two user studies were performed: the first involved two clinical professionals (a neuroradiologist and a cardiologist), who confirmed the method's impact and achieved high task accuracy; the second included 22 biomedical experts, who demonstrated high discrimination accuracy in tissue differentiation and targeting tasks. The results revealed a strong correlation between tool-tissue dynamics and their corresponding auditory profiles, highlighting the potential of these sound representations to enhance the intuitive understanding of complex interactions.
>
---
#### [new 004] Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出基于Mamba2的语音源分离模型，解决间歇性语音分离问题，通过带分割与双路径架构提升长序列处理能力，取得最佳cSDR成绩（11.03dB）。**

- **链接: [http://arxiv.org/pdf/2508.14556v1](http://arxiv.org/pdf/2508.14556v1)**

> **作者:** Euiyeon Kim; Yong-Hoon Choi
>
> **摘要:** We introduce a new music source separation model tailored for accurate vocal isolation. Unlike Transformer-based approaches, which often fail to capture intermittently occurring vocals, our model leverages Mamba2, a recent state space model, to better capture long-range temporal dependencies. To handle long input sequences efficiently, we combine a band-splitting strategy with a dual-path architecture. Experiments show that our approach outperforms recent state-of-the-art models, achieving a cSDR of 11.03 dB-the best reported to date-and delivering substantial gains in uSDR. Moreover, the model exhibits stable and consistent performance across varying input lengths and vocal occurrence patterns. These results demonstrate the effectiveness of Mamba-based models for high-resolution audio processing and open up new directions for broader applications in audio research.
>
---
#### [new 005] ECHO: Frequency-aware Hierarchical Encoding for Variable-length Signal
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文针对机器信号建模任务，解决变长信号处理中固定输入长度和缺乏频率位置编码的问题，提出ECHO模型结合带宽分割与相对频率嵌入，支持任意长度输入并保留时频信息，在SIREN基准上实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.14689v1](http://arxiv.org/pdf/2508.14689v1)**

> **作者:** Yucong Zhang; Juan Liu; Ming Li
>
> **摘要:** Pre-trained foundation models have demonstrated remarkable success in vision and language, yet their potential for general machine signal modeling-covering acoustic, vibration, and other industrial sensor data-remains under-explored. Existing approach using sub-band-based encoders has achieved competitive results but are limited by fixed input lengths, and the absence of explicit frequency positional encoding. In this work, we propose a novel foundation model that integrates an advanced band-split architecture with relative frequency positional embeddings, enabling precise spectral localization across arbitrary sampling configurations. The model supports inputs of arbitrary length without padding or segmentation, producing a concise embedding that retains both temporal and spectral fidelity. We evaluate our method on SIREN (https://github.com/yucongzh/SIREN), a newly introduced large-scale benchmark for machine signal encoding that unifies multiple datasets, including all DCASE task 2 challenges (2020-2025) and widely-used industrial signal corpora. Experimental results demonstrate consistent state-of-the-art performance in anomaly detection and fault identification, confirming the effectiveness and generalization capability of the proposed model. We open-sourced ECHO on https://github.com/yucongzh/ECHO.
>
---
#### [new 006] Activity Coefficient-based Channel Selection for Electroencephalogram: A Task-Independent Approach
- **分类: q-bio.NC; cs.CV; cs.HC; cs.LG; eess.SP**

- **简介: 论文针对脑机接口中高密度EEG通道选择问题，提出任务无关的ACCS方法，通过Channel Activity Coefficient量化通道效用，选择前16通道提升多类分类准确率至34.97%，适应多样应用。**

- **链接: [http://arxiv.org/pdf/2508.14060v1](http://arxiv.org/pdf/2508.14060v1)**

> **作者:** Kartik Pandey; Arun Balasubramanian; Debasis Samanta
>
> **摘要:** Electroencephalogram (EEG) signals have gained widespread adoption in brain-computer interface (BCI) applications due to their non-invasive, low-cost, and relatively simple acquisition process. The demand for higher spatial resolution, particularly in clinical settings, has led to the development of high-density electrode arrays. However, increasing the number of channels introduces challenges such as cross-channel interference and computational overhead. To address these issues, modern BCI systems often employ channel selection algorithms. Existing methods, however, are typically task-specific and require re-optimization for each new application. This work proposes a task-agnostic channel selection method, Activity Coefficient-based Channel Selection (ACCS), which uses a novel metric called the Channel Activity Coefficient (CAC) to quantify channel utility based on activity levels. By selecting the top 16 channels ranked by CAC, ACCS achieves up to 34.97% improvement in multi-class classification accuracy. Unlike traditional approaches, ACCS identifies a reusable set of informative channels independent of the downstream task or model, making it highly adaptable for diverse EEG-based applications.
>
---
#### [new 007] A Study of the Scale Invariant Signal to Distortion Ratio in Speech Separation with Noisy References
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 论文研究语音分离任务，针对噪声参考下SI-SDR的局限性，提出增强参考与混合信号的方法以提升分离质量，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.14623v1](http://arxiv.org/pdf/2508.14623v1)**

> **作者:** Simon Dahl Jepsen; Mads Græsbøll Christensen; Jesper Rindom Jensen
>
> **备注:** Accepted for IEEE ASRU 2025, Workshop on Automatic Speech Recognition and Understanding. Copyright (c) 2025 IEEE. 8 pages, 6 figures, 2 tables
>
> **摘要:** This paper examines the implications of using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) as both evaluation and training objective in supervised speech separation, when the training references contain noise, as is the case with the de facto benchmark WSJ0-2Mix. A derivation of the SI-SDR with noisy references reveals that noise limits the achievable SI-SDR, or leads to undesired noise in the separated outputs. To address this, a method is proposed to enhance references and augment the mixtures with WHAM!, aiming to train models that avoid learning noisy references. Two models trained on these enhanced datasets are evaluated with the non-intrusive NISQA.v2 metric. Results show reduced noise in separated speech but suggest that processing references may introduce artefacts, limiting overall quality gains. Negative correlation is found between SI-SDR and perceived noisiness across models on the WSJ0-2Mix and Libri2Mix test sets, underlining the conclusion from the derivation.
>
---
#### [new 008] Towards Low-Latency Tracking of Multiple Speakers With Short-Context Speaker Embeddings
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 论文针对多说话人低延迟追踪任务，解决短上下文和重叠语音导致的身份识别难题，提出基于知识蒸馏的短上下文嵌入提取方法，结合波束成形与分块身份重分配，提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.14115v1](http://arxiv.org/pdf/2508.14115v1)**

> **作者:** Taous Iatariene; Alexandre Guérin; Romain Serizel
>
> **摘要:** Speaker embeddings are promising identity-related features that can enhance the identity assignment performance of a tracking system by leveraging its spatial predictions, i.e, by performing identity reassignment. Common speaker embedding extractors usually struggle with short temporal contexts and overlapping speech, which imposes long-term identity reassignment to exploit longer temporal contexts. However, this increases the probability of tracking system errors, which in turn impacts negatively on identity reassignment. To address this, we propose a Knowledge Distillation (KD) based training approach for short context speaker embedding extraction from two speaker mixtures. We leverage the spatial information of the speaker of interest using beamforming to reduce overlap. We study the feasibility of performing identity reassignment over blocks of fixed size, i.e., blockwise identity reassignment, to go towards a low-latency speaker embedding based tracking system. Results demonstrate that our distilled models are effective at short-context embedding extraction and more robust to overlap. Although, blockwise reassignment results indicate that further work is needed to handle simultaneous speech more effectively.
>
---
#### [new 009] Improving Resource-Efficient Speech Enhancement via Neural Differentiable DSP Vocoder Refinement
- **分类: eess.AS; cs.SD**

- **简介: 论文提出一种高效端到端语音增强框架，结合DDSP vocoder和STFT/对抗损失，提升语音质量与可懂度，适用于资源受限设备。**

- **链接: [http://arxiv.org/pdf/2508.14709v1](http://arxiv.org/pdf/2508.14709v1)**

> **作者:** Heitor R. Guimarães; Ke Tan; Juan Azcarreta; Jesus Alvarez; Prabhav Agrawal; Ashutosh Pandey; Buye Xu
>
> **备注:** Accepted to the 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)
>
> **摘要:** Deploying speech enhancement (SE) systems in wearable devices, such as smart glasses, is challenging due to the limited computational resources on the device. Although deep learning methods have achieved high-quality results, their computational cost limits their feasibility on embedded platforms. This work presents an efficient end-to-end SE framework that leverages a Differentiable Digital Signal Processing (DDSP) vocoder for high-quality speech synthesis. First, a compact neural network predicts enhanced acoustic features from noisy speech: spectral envelope, fundamental frequency (F0), and periodicity. These features are fed into the DDSP vocoder to synthesize the enhanced waveform. The system is trained end-to-end with STFT and adversarial losses, enabling direct optimization at the feature and waveform levels. Experimental results show that our method improves intelligibility and quality by 4% (STOI) and 19% (DNSMOS) over strong baselines without significantly increasing computation, making it well-suited for real-time applications.
>
---
#### [new 010] EmoTale: An Enacted Speech-emotion Dataset in Danish
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文提出EmoTale数据集，针对丹麦语情感语音数据不足问题，通过自监督模型和特征提取器评估其有效性，显示与现有数据集相当的语音情感识别性能。**

- **链接: [http://arxiv.org/pdf/2508.14548v1](http://arxiv.org/pdf/2508.14548v1)**

> **作者:** Maja J. Hjuler; Harald V. Skat-Rørdam; Line H. Clemmensen; Sneha Das
>
> **备注:** To appear in the proceedings of ASRU 2025
>
> **摘要:** While multiple emotional speech corpora exist for commonly spoken languages, there is a lack of functional datasets for smaller (spoken) languages, such as Danish. To our knowledge, Danish Emotional Speech (DES), published in 1997, is the only other database of Danish emotional speech. We present EmoTale; a corpus comprising Danish and English speech recordings with their associated enacted emotion annotations. We demonstrate the validity of the dataset by investigating and presenting its predictive power using speech emotion recognition (SER) models. We develop SER models for EmoTale and the reference datasets using self-supervised speech model (SSLM) embeddings and the openSMILE feature extractor. We find the embeddings superior to the hand-crafted features. The best model achieves an unweighted average recall (UAR) of 64.1% on the EmoTale corpus using leave-one-speaker-out cross-validation, comparable to the performance on DES.
>
---
#### [new 011] Long-Context Speech Synthesis with Context-Aware Memory
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对长文本语音合成中段落连贯性不足的问题，提出基于Context-Aware Memory的模型，通过整合长期记忆与局部上下文实现动态记忆更新，提升语音的自然度与连贯性。**

- **链接: [http://arxiv.org/pdf/2508.14713v1](http://arxiv.org/pdf/2508.14713v1)**

> **作者:** Zhipeng Li; Xiaofen Xing; Jingyuan Xing; Hangrui Hu; Heng Lu; Xiangmin Xu
>
> **备注:** Accepted by Interspeech25
>
> **摘要:** In long-text speech synthesis, current approaches typically convert text to speech at the sentence-level and concatenate the results to form pseudo-paragraph-level speech. These methods overlook the contextual coherence of paragraphs, leading to reduced naturalness and inconsistencies in style and timbre across the long-form speech. To address these issues, we propose a Context-Aware Memory (CAM)-based long-context Text-to-Speech (TTS) model. The CAM block integrates and retrieves both long-term memory and local context details, enabling dynamic memory updates and transfers within long paragraphs to guide sentence-level speech synthesis. Furthermore, the prefix mask enhances the in-context learning ability by enabling bidirectional attention on prefix tokens while maintaining unidirectional generation. Experimental results demonstrate that the proposed method outperforms baseline and state-of-the-art long-context methods in terms of prosody expressiveness, coherence and context inference cost across paragraph-level speech.
>
---
## 更新

#### [replaced 001] The Man Behind the Sound: Demystifying Audio Private Attribute Profiling via Multimodal Large Language Model Agents
- **分类: cs.CR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.10016v2](http://arxiv.org/pdf/2507.10016v2)**

> **作者:** Lixu Wang; Kaixiang Yao; Xinfeng Li; Dong Yang; Haoyang Li; Xiaofeng Wang; Wei Dong
>
> **备注:** 22 pages, 4 figures
>
> **摘要:** Our research uncovers a novel privacy risk associated with multimodal large language models (MLLMs): the ability to infer sensitive personal attributes from audio data -- a technique we term audio private attribute profiling. This capability poses a significant threat, as audio can be covertly captured without direct interaction or visibility. Moreover, compared to images and text, audio carries unique characteristics, such as tone and pitch, which can be exploited for more detailed profiling. However, two key challenges exist in understanding MLLM-employed private attribute profiling from audio: (1) the lack of audio benchmark datasets with sensitive attribute annotations and (2) the limited ability of current MLLMs to infer such attributes directly from audio. To address these challenges, we introduce AP^2, an audio benchmark dataset that consists of two subsets collected and composed from real-world data, and both are annotated with sensitive attribute labels. Additionally, we propose Gifts, a hybrid multi-agent framework that leverages the complementary strengths of audio-language models (ALMs) and large language models (LLMs) to enhance inference capabilities. Gifts employs an LLM to guide the ALM in inferring sensitive attributes, then forensically analyzes and consolidates the ALM's inferences, overcoming severe hallucinations of existing ALMs in generating long-context responses. Our evaluations demonstrate that Gifts significantly outperforms baseline approaches in inferring sensitive attributes. Finally, we investigate model-level and data-level defense strategies to mitigate the risks of audio private attribute profiling. Our work validates the feasibility of audio-based privacy attacks using MLLMs, highlighting the need for robust defenses, and provides a dataset and framework to facilitate future research.
>
---
#### [replaced 002] When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.03365v2](http://arxiv.org/pdf/2508.03365v2)**

> **作者:** Bodam Kim; Hiskias Dingeto; Taeyoun Kwon; Dasol Choi; DongGeon Lee; Haon Park; JaeHoon Lee; Jongho Shin
>
> **摘要:** As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.
>
---
#### [replaced 003] FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14351v3](http://arxiv.org/pdf/2505.14351v3)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Yuqing Cai; Yongbin Yu; Renzeng Duojie; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **备注:** 18 pages
>
> **摘要:** Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-\"U-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
>
---
#### [replaced 004] Is Transfer Learning Necessary for Violin Transcription?
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.13516v2](http://arxiv.org/pdf/2508.13516v2)**

> **作者:** Yueh-Po Peng; Ting-Kang Wang; Li Su; Vincent K. M. Cheung
>
> **备注:** Accepted at ISMIR 2025 as Late-Breaking Demo (LBD)
>
> **摘要:** Automatic music transcription (AMT) has achieved remarkable progress for instruments such as the piano, largely due to the availability of large-scale, high-quality datasets. In contrast, violin AMT remains underexplored due to limited annotated data. A common approach is to fine-tune pretrained models for other downstream tasks, but the effectiveness of such transfer remains unclear in the presence of timbral and articulatory differences. In this work, we investigate whether training from scratch on a medium-scale violin dataset can match the performance of fine-tuned piano-pretrained models. We adopt a piano transcription architecture without modification and train it on the MOSA dataset, which contains about 30 hours of aligned violin recordings. Our experiments on URMP and Bach10 show that models trained from scratch achieved competitive or even superior performance compared to fine-tuned counterparts. These findings suggest that strong violin AMT is possible without relying on pretrained piano representations, highlighting the importance of instrument-specific data collection and augmentation strategies.
>
---
#### [replaced 005] Customizing Speech Recognition Model with Large Language Model Feedback
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.11091v2](http://arxiv.org/pdf/2506.11091v2)**

> **作者:** Shaoshi Ling; Guoli Ye
>
> **摘要:** Automatic speech recognition (ASR) systems have achieved strong performance on general transcription tasks. However, they continue to struggle with recognizing rare named entities and adapting to domain mismatches. In contrast, large language models (LLMs), trained on massive internet-scale datasets, are often more effective across a wide range of domains. In this work, we propose a reinforcement learning based approach for unsupervised domain adaptation, leveraging unlabeled data to enhance transcription quality, particularly the named entities affected by domain mismatch, through feedback from a LLM. Given contextual information, our framework employs a LLM as the reward model to score the hypotheses from the ASR model. These scores serve as reward signals to fine-tune the ASR model via reinforcement learning. Our method achieves a 21\% improvement on entity word error rate over conventional self-training methods.
>
---
#### [replaced 006] ASAudio: A Survey of Advanced Spatial Audio Research
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.10924v2](http://arxiv.org/pdf/2508.10924v2)**

> **作者:** Zhiyuan Zhu; Yu Zhang; Wenxiang Guo; Changhao Pan; Zhou Zhao
>
> **摘要:** With the rapid development of spatial audio technologies today, applications in AR, VR, and other scenarios have garnered extensive attention. Unlike traditional mono sound, spatial audio offers a more realistic and immersive auditory experience. Despite notable progress in the field, there remains a lack of comprehensive surveys that systematically organize and analyze these methods and their underlying technologies. In this paper, we provide a comprehensive overview of spatial audio and systematically review recent literature in the area. To address this, we chronologically outlining existing work related to spatial audio and categorize these studies based on input-output representations, as well as generation and understanding tasks, thereby summarizing various research aspects of spatial audio. In addition, we review related datasets, evaluation metrics, and benchmarks, offering insights from both training and evaluation perspectives. Related materials are available at https://github.com/dieKarotte/ASAudio.
>
---
