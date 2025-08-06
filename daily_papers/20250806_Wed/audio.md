# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] TF-MLPNet: Tiny Real-Time Neural Speech Separation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 本研究提出一种基于时间-频率域的TF-MLPNet模型，旨在实现低功耗、实时的语音分离任务。该模型通过时间频域处理与混合精度量化训练相结合，克服了传统模型在低功率加速器上的性能瓶颈。**

- **链接: [http://arxiv.org/pdf/2508.03047v1](http://arxiv.org/pdf/2508.03047v1)**

> **作者:** Malek Itani; Tuochao Chen; Shyamnath Gollakota
>
> **备注:** The 6th Clarity Workshop on Improving Speech-in-Noise for Hearing Devices (Clarity 2025)
>
> **摘要:** Speech separation on hearable devices can enable transformative augmented and enhanced hearing capabilities. However, state-of-the-art speech separation networks cannot run in real-time on tiny, low-power neural accelerators designed for hearables, due to their limited compute capabilities. We present TF-MLPNet, the first speech separation network capable of running in real-time on such low-power accelerators while outperforming existing streaming models for blind speech separation and target speech extraction. Our network operates in the time-frequency domain, processing frequency sequences with stacks of fully connected layers that alternate along the channel and frequency dimensions, and independently processing the time sequence at each frequency bin using convolutional layers. Results show that our mixed-precision quantization-aware trained (QAT) model can process 6 ms audio chunks in real-time on the GAP9 processor, achieving a 3.5-4x runtime reduction compared to prior speech separation models.
>
---
#### [new 002] Adaptive Knowledge Distillation for Device-Directed Speech Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出一种针对设备导向语音检测任务的自适应知识蒸馏方法，旨在通过联合优化教师模型与学生模型的知识转移策略，提升DDSD的准确性和部署效率，解决了传统方法在跨模型适配和高效性方面的挑战。**

- **链接: [http://arxiv.org/pdf/2508.02801v1](http://arxiv.org/pdf/2508.02801v1)**

> **作者:** Hyung Gun Chi; Florian Pesce; Wonil Chang; Oggi Rudovic; Arturo Argueta; Stefan Braun; Vineet Garg; Ahmed Hussen Abdelaziz
>
> **备注:** 5 pages, 2 figures, Interspeech accepted
>
> **摘要:** Device-directed speech detection (DDSD) is a binary classification task that separates the user's queries to a voice assistant (VA) from background speech or side conversations. This is important for achieving naturalistic user experience. To this end, we propose knowledge distillation (KD) to enhance DDSD accuracy while ensuring efficient deployment. Specifically, we introduce a novel adaptive KD method that transfers knowledge from general representations of an ASR large pre-trained acoustic encoder (teacher). We apply task-specific adapters, on top of the (frozen) teacher encoder, trained jointly with the student model on DDSD. We demonstrate that the proposed adaptive KD outperforms the student model without distillation in the keyword and keyword-free (follow-up) invocations, with an improvement of +26% and +19% in terms of Equal Error Rate, respectively. We also show that this approach generalizes across the transformer and conformer-based model architectures.
>
---
#### [new 003] Neural Speech Extraction with Human Feedback
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出了一种结合人类反馈的神经语音提取方法，旨在通过人工标记优化模型输出并保留未标记区域，解决了传统方法难以收集大量标注数据的问题。通过自动生成合成数据集训练模型，实验表明噪声功率加权掩码（dBFS）与概率阈值法在人类标注下表现最佳，验证了人类-人工协作提升性能的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03041v1](http://arxiv.org/pdf/2508.03041v1)**

> **作者:** Malek Itani; Ashton Graves; Sefik Emre Eskimez; Shyamnath Gollakota
>
> **备注:** Interspeech 2025
>
> **摘要:** We present the first neural target speech extraction (TSE) system that uses human feedback for iterative refinement. Our approach allows users to mark specific segments of the TSE output, generating an edit mask. The refinement system then improves the marked sections while preserving unmarked regions. Since large-scale datasets of human-marked errors are difficult to collect, we generate synthetic datasets using various automated masking functions and train models on each. Evaluations show that models trained with noise power-based masking (in dBFS) and probabilistic thresholding perform best, aligning with human annotations. In a study with 22 participants, users showed a preference for refined outputs over baseline TSE. Our findings demonstrate that human-in-the-loop refinement is a promising approach for improving the performance of neural speech extraction.
>
---
#### [new 004] Fine-Tuning Text-to-Speech Diffusion Models Using Reinforcement Learning with Human Feedback
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出通过强化学习与人类反馈优化TTS扩散模型，解决传统模型因长denoising步骤和音调/节奏建模困难导致的效率低和质量差问题，通过整合训练损失并调整奖励机制提升实时生成能力。**

- **链接: [http://arxiv.org/pdf/2508.03123v1](http://arxiv.org/pdf/2508.03123v1)**

> **作者:** Jingyi Chen; Ju Seung Byun; Micha Elsner; Pichao Wang; Andrew Perrault
>
> **备注:** 4 pages, 1 figure, INTERSPEECH 2025. arXiv admin note: text overlap with arXiv:2405.14632
>
> **摘要:** Diffusion models produce high-fidelity speech but are inefficient for real-time use due to long denoising steps and challenges in modeling intonation and rhythm. To improve this, we propose Diffusion Loss-Guided Policy Optimization (DLPO), an RLHF framework for TTS diffusion models. DLPO integrates the original training loss into the reward function, preserving generative capabilities while reducing inefficiencies. Using naturalness scores as feedback, DLPO aligns reward optimization with the diffusion model's structure, improving speech quality. We evaluate DLPO on WaveGrad 2, a non-autoregressive diffusion-based TTS model. Results show significant improvements in objective metrics (UTMOS 3.65, NISQA 4.02) and subjective evaluations, with DLPO audio preferred 67\% of the time. These findings demonstrate DLPO's potential for efficient, high-quality diffusion TTS in real-time, resource-limited settings.
>
---
#### [new 005] When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **简介: 该论文提出了一种音频-语言模型的恶意攻击框架，旨在利用可感知但无害的输入攻击，通过两阶段对抗训练绕过安全机制并生成有害响应，验证其成功率超过86%，适用于多种大型模型。**

- **链接: [http://arxiv.org/pdf/2508.03365v1](http://arxiv.org/pdf/2508.03365v1)**

> **作者:** Bodam Kim; Hiskias Dingeto; Taeyoun Kwon; Dasol Choi; DongGeon Lee; Haon Park; JaeHoon Lee; Jongho Shin
>
> **摘要:** As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.
>
---
#### [new 006] Can Large Language Models Identify Materials from Radar Signals?
- **分类: eess.SP; cs.ET; cs.RO**

- **简介: 该论文研究了大语言模型（LLMs）能否直接从雷达信号中识别材料，解决了现有方法受限于类别和数据收集的问题，提出LMMaterial通过物理信号处理与检索增强生成技术，实现了对多种材料的开放集识别。**

- **链接: [http://arxiv.org/pdf/2508.03120v1](http://arxiv.org/pdf/2508.03120v1)**

> **作者:** Jiangyou Zhu; Hongyu Deng; He Chen
>
> **摘要:** Accurately identifying the material composition of objects is a critical capability for AI robots powered by large language models (LLMs) to perform context-aware manipulation. Radar technologies offer a promising sensing modality for material recognition task. When combined with deep learning, radar technologies have demonstrated strong potential in identifying the material of various objects. However, existing radar-based solutions are often constrained to closed-set object categories and typically require task-specific data collection to train deep learning models, largely limiting their practical applicability. This raises an important question: Can we leverage the powerful reasoning capabilities of pre-trained LLMs to directly infer material composition from raw radar signals? Answering this question is non-trivial due to the inherent redundancy of radar signals and the fact that pre-trained LLMs have no prior exposure to raw radar data during training. To address this, we introduce LLMaterial, the first study to investigate the feasibility of using LLM to identify materials directly from radar signals. First, we introduce a physics-informed signal processing pipeline that distills high-redundancy radar raw data into a set of compact intermediate parameters that encapsulate the material's intrinsic characteristics. Second, we adopt a retrieval-augmented generation (RAG) strategy to provide the LLM with domain-specific knowledge, enabling it to interpret and reason over the extracted intermediate parameters. Leveraging this integration, the LLM is empowered to perform step-by-step reasoning on the condensed radar features, achieving open-set material recognition directly from raw radar signals. Preliminary results show that LLMaterial can effectively distinguish among a variety of common materials, highlighting its strong potential for real-world material identification applications.
>
---
#### [new 007] EmoSteer-TTS: Fine-Grained and Training-Free Emotion-Controllable Text-to-Speech via Activation Steering
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出"EmoSteer-TTS"，通过激活引导实现细粒度情绪控制，解决传统TTS仅提供粗粒度控制的问题，开发训练自由算法并构建情感数据集，验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.03543v1](http://arxiv.org/pdf/2508.03543v1)**

> **作者:** Tianxin Xie; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **摘要:** Text-to-speech (TTS) has shown great progress in recent years. However, most existing TTS systems offer only coarse and rigid emotion control, typically via discrete emotion labels or a carefully crafted and detailed emotional text prompt, making fine-grained emotion manipulation either inaccessible or unstable. These models also require extensive, high-quality datasets for training. To address these limitations, we propose EmoSteer-TTS, a novel training-free approach, to achieve fine-grained speech emotion control (conversion, interpolation, erasure) by activation steering. We first empirically observe that modifying a subset of the internal activations within a flow matching-based TTS model can effectively alter the emotional tone of synthesized speech. Building on this insight, we then develop a training-free and efficient algorithm, including activation extraction, emotional token searching, and inference-time steering, which can be seamlessly integrated into a wide range of pretrained models (e.g., F5-TTS, CosyVoice2, and E2-TTS). In addition, to derive effective steering vectors, we construct a curated emotional speech dataset with diverse speakers. Extensive experiments demonstrate that EmoSteer-TTS enables fine-grained, interpretable, and continuous control over speech emotion, outperforming the state-of-the-art (SOTA). To the best of our knowledge, this is the first method that achieves training-free and continuous fine-grained emotion control in TTS.
>
---
#### [new 008] MiSTR: Multi-Modal iEEG-to-Speech Synthesis with Transformer-Based Prosody Prediction and Neural Phase Reconstruction
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出MiSTR，解决多模态iEEG语音合成中的特征表示、声调建模和相位恢复问题，融合波形分解、Transformer解码和神经相位编码器进行优化，实现高准确率的语音合成。**

- **链接: [http://arxiv.org/pdf/2508.03166v1](http://arxiv.org/pdf/2508.03166v1)**

> **作者:** Mohammed Salah Al-Radhi; Géza Németh; Branislav Gerazov
>
> **备注:** 5 pages, 2 figures, 1 table. Accepted for presentation at Interspeech 2025
>
> **摘要:** Speech synthesis from intracranial EEG (iEEG) signals offers a promising avenue for restoring communication in individuals with severe speech impairments. However, achieving intelligible and natural speech remains challenging due to limitations in feature representation, prosody modeling, and phase reconstruction. We introduce MiSTR, a deep-learning framework that integrates: 1) Wavelet-based feature extraction to capture fine-grained temporal, spectral, and neurophysiological representations of iEEG signals, 2) A Transformer-based decoder for prosody-aware spectrogram prediction, and 3) A neural phase vocoder enforcing harmonic consistency via adaptive spectral correction. Evaluated on a public iEEG dataset, MiSTR achieves state-of-the-art speech intelligibility, with a mean Pearson correlation of 0.91 between reconstructed and original Mel spectrograms, improving over existing neural speech synthesis baselines.
>
---
#### [new 009] Generating Light-based Fingerprints for Indoor Localization
- **分类: eess.SP; cs.RO; I.2.9; C.3**

- **简介: 该论文旨在解决室内定位中的信号干扰与覆盖问题，提出通过可见光通信（VLC）生成指纹并优化训练数据的方法，构建了两阶段框架提升定位精度20%。**

- **链接: [http://arxiv.org/pdf/2508.03011v1](http://arxiv.org/pdf/2508.03011v1)**

> **作者:** Hsun-Yu Lee; Jie Lin; Fang-Jing Wu
>
> **备注:** 5 pages, 12 figures; presented at the 2024 MC & WASN Conference (Best Paper Candidate)
>
> **摘要:** Accurate indoor localization underpins applications ranging from wayfinding and emergency response to asset tracking and smart-building services. Radio-frequency solutions (e.g. Wi-Fi, RFID, UWB) are widely adopted but remain vulnerable to multipath fading, interference, and uncontrollable coverage variation. We explore an orthogonal modality -- visible light communication (VLC) -- and demonstrate that the spectral signatures captured by a low-cost AS7341 sensor can serve as robust location fingerprints. We introduce a two-stage framework that (i) trains a multi-layer perceptron (MLP) on real spectral measurements and (ii) enlarges the training corpus with synthetic samples produced by TabGAN. The augmented dataset reduces the mean localization error from 62.9cm to 49.3cm -- a 20% improvement -- while requiring only 5% additional data-collection effort. Experimental results obtained on 42 reference points in a U-shaped laboratory confirm that GAN-based augmentation mitigates data-scarcity issues and enhances generalization.
>
---
#### [new 010] SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出了一种基于生成式模型的音乐恢复与混合法则，旨在解决非专业环境中音频质量差的问题。通过构建包含降噪、动态范围、回声等5个增强组的大型数据集并采用流匹配训练，实现了对多种音频问题的统一优化，最终通过实验验证了其显著提升音质的效果。**

- **链接: [http://arxiv.org/pdf/2508.03448v1](http://arxiv.org/pdf/2508.03448v1)**

> **作者:** Jan Melechovsky; Ambuj Mehrish; Dorien Herremans
>
> **摘要:** Music recordings often suffer from audio quality issues such as excessive reverberation, distortion, clipping, tonal imbalances, and a narrowed stereo image, especially when created in non-professional settings without specialized equipment or expertise. These problems are typically corrected using separate specialized tools and manual adjustments. In this paper, we introduce SonicMaster, the first unified generative model for music restoration and mastering that addresses a broad spectrum of audio artifacts with text-based control. SonicMaster is conditioned on natural language instructions to apply targeted enhancements, or can operate in an automatic mode for general restoration. To train this model, we construct the SonicMaster dataset, a large dataset of paired degraded and high-quality tracks by simulating common degradation types with nineteen degradation functions belonging to five enhancements groups: equalization, dynamics, reverb, amplitude, and stereo. Our approach leverages a flow-matching generative training paradigm to learn an audio transformation that maps degraded inputs to their cleaned, mastered versions guided by text prompts. Objective audio quality metrics demonstrate that SonicMaster significantly improves sound quality across all artifact categories. Furthermore, subjective listening tests confirm that listeners prefer SonicMaster's enhanced outputs over the original degraded audio, highlighting the effectiveness of our unified approach.
>
---
#### [new 011] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出了一种基于跨模态注意力的轻量级AI模型，解决传统TB筛查效率低、成本高的问题，通过融合CNN与梯度提升树，创新设计TRBL损失函数并验证了其在真实数据上的性能，适用于低资源环境中的实时诊断。**

- **链接: [http://arxiv.org/pdf/2508.02741v1](http://arxiv.org/pdf/2508.02741v1)**

> **作者:** Zhixiang Lu; Yulong Li; Feilong Tang; Zhengyong Jiang; Chong Li; Mian Zhou; Tenglong Li; Jionglong Su
>
> **摘要:** Large-scale tuberculosis (TB) screening is limited by the high cost and operational complexity of traditional diagnostics, creating a need for artificial-intelligence solutions. We propose DeepGB-TB, a non-invasive system that instantly assigns TB risk scores using only cough audio and basic demographic data. The model couples a lightweight one-dimensional convolutional neural network for audio processing with a gradient-boosted decision tree for tabular features. Its principal innovation is a Cross-Modal Bidirectional Cross-Attention module (CM-BCA) that iteratively exchanges salient cues between modalities, emulating the way clinicians integrate symptoms and risk factors. To meet the clinical priority of minimizing missed cases, we design a Tuberculosis Risk-Balanced Loss (TRBL) that places stronger penalties on false-negative predictions, thereby reducing high-risk misclassifications. DeepGB-TB is evaluated on a diverse dataset of 1,105 patients collected across seven countries, achieving an AUROC of 0.903 and an F1-score of 0.851, representing a new state of the art. Its computational efficiency enables real-time, offline inference directly on common mobile devices, making it ideal for low-resource settings. Importantly, the system produces clinically validated explanations that promote trust and adoption by frontline health workers. By coupling AI innovation with public-health requirements for speed, affordability, and reliability, DeepGB-TB offers a tool for advancing global TB control.
>
---
#### [new 012] SecoustiCodec: Cross-Modal Aligned Streaming Single-Codecbook Speech Codec
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出一种跨模态低带宽语音编码器SecoustiCodec，旨在解决语义与声学信息分离困难、提升语义完整性和重建质量的问题。通过引入VAE和FSQ实现语义编码，结合对比学习消除混音信息，并采用多阶段优化策略，有效缓解长尾分布问题，达到SOTA的PESQ性能（1.77/2.58）。**

- **链接: [http://arxiv.org/pdf/2508.02849v1](http://arxiv.org/pdf/2508.02849v1)**

> **作者:** Chunyu Qiang; Haoyu Wang; Cheng Gong; Tianrui Wang; Ruibo Fu; Tao Wang; Ruilong Chen; Jiangyan Yi; Zhengqi Wen; Chen Zhang; Longbiao Wang; Jianwu Dang; Jianhua Tao
>
> **摘要:** Speech codecs serve as a crucial bridge in unifying speech and text language models. Existing codec methods face several challenges in semantic encoding, such as residual paralinguistic information (e.g., timbre, emotion), insufficient semantic completeness, limited reconstruction capability, and lack of support for streaming. To address these challenges, we propose SecoustiCodec, a cross-modal aligned low-bitrate streaming speech codec that disentangles semantic and paralinguistic information in a single-codebook space. To ensure semantic completeness and reconstruction fidelity, paralinguistic encoding is introduced to bridge the information gap between semantic and acoustic encoding. A semantic-only efficient quantization method based on VAE (Variational Autoencoder) and FSQ (Finite Scalar Quantization) is proposed. This approach alleviates the long-tail distribution problem of tokens while maintaining high codebook utilization. A semantic disentanglement method based on contrastive learning is proposed, which aligns text and speech in a joint multimodal frame-level space, effectively removing paralinguistic information from semantic encoding. An acoustic-constrained multi-stage optimization strategy is proposed to ensure robust and stable convergence. Figure~\ref{fig:pesq_kbps_below_2kbps} shows SecoustiCodec achieves SOTA (state-of-the-art) reconstruction quality (PESQ) of 1.77/2.58 at 0.27/1 kbps. The code and model weights for SecoustiCodec will be open-sourced upon the completion of the peer-review process. We've open-sourced SecoustiCodec's demo, code, and model weights.
>
---
#### [new 013] Fast Algorithm for Moving Sound Source
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在解决现代语音增强模型在移动场景下的训练数据不足问题，提出Yang的运动时空采样重构理论，通过分解移动源的瞬态响应并构建物理合规的声场模型，结合层次化采样与高效合成架构，有效模拟连续时间变化的运动声源，实现了高精度数据恢复与多通道语音跟踪算法的鲁棒性提升。**

- **链接: [http://arxiv.org/pdf/2508.03065v1](http://arxiv.org/pdf/2508.03065v1)**

> **作者:** Dong Yang
>
> **摘要:** Modern neural network-based speech processing systems need reverberation resistance, relying on large amounts of reverberation data for training. Existing methods simulate dynamic scenarios by sampling static systems or supplement with measured data, but struggle to simulate motion data conforming to physical laws. To address insufficient training data for speech enhancement models in moving scenarios, this paper proposes Yang's motion spatio-temporal sampling reconstruction theory, enabling efficient simulation of motion-induced continuous time-varying reverberation. It breaks through the limitations of traditional static Image-Source Method (ISM) in time-varying systems by decomposing the moving image source's impulse response into linear time-invariant modulation and discrete time-varying fractional delay, establishing a physics-compliant moving sound field model. Based on the band-limited nature of motion displacement, a hierarchical sampling strategy is adopted: high sampling rates for low-order images to retain details, and low rates for high-order ones to reduce complexity, combined with a fast synthesis architecture for real-time simulation. Experiments show that compared to open-source model GSound, the theory more accurately restores amplitude and phase changes in moving scenarios, solving the industry challenge of motion sound source data simulation. It provides high-quality dynamic training data for speech enhancement models and improves the robustness of multi-channel end-to-end voice tracking algorithms.
>
---
#### [new 014] Real-World Receptivity to Adaptive Mental Health Interventions: Findings from an In-the-Wild Study
- **分类: cs.HC; cs.AI; cs.CY; eess.SP**

- **简介: 该论文旨在探讨智能设备在真实场景下的心理干预接收度，通过分析用户对即时反馈的接受与可行性的感知，构建基于强化学习的个性化干预系统。研究使用Android应用LogMe收集数据，验证其在动态情境中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.02817v1](http://arxiv.org/pdf/2508.02817v1)**

> **作者:** Nilesh Kumar Sahu; Aditya Sneh; Snehil Gupta; Haroon R Lone
>
> **摘要:** The rise of mobile health (mHealth) technologies has enabled real-time monitoring and intervention for mental health conditions using passively sensed smartphone data. Building on these capabilities, Just-in-Time Adaptive Interventions (JITAIs) seek to deliver personalized support at opportune moments, adapting to users' evolving contexts and needs. Although prior research has examined how context affects user responses to generic notifications and general mHealth messages, relatively little work has explored its influence on engagement with actual mental health interventions. Furthermore, while much of the existing research has focused on detecting when users might benefit from an intervention, less attention has been paid to understanding receptivity, i.e., users' willingness and ability to engage with and act upon the intervention. In this study, we investigate user receptivity through two components: acceptance(acknowledging or engaging with a prompt) and feasibility (ability to act given situational constraints). We conducted a two-week in-the-wild study with 70 students using a custom Android app, LogMe, which collected passive sensor data and active context reports to prompt mental health interventions. The adaptive intervention module was built using Thompson Sampling, a reinforcement learning algorithm. We address four research questions relating smartphone features and self-reported contexts to acceptance and feasibility, and examine whether an adaptive reinforcement learning approach can optimize intervention delivery by maximizing a combined receptivity reward. Our results show that several types of passively sensed data significantly influenced user receptivity to interventions. Our findings contribute insights into the design of context-aware, adaptive interventions that are not only timely but also actionable in real-world settings.
>
---
#### [new 015] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Indoor Scenes
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出一种材料控制的声学图谱生成任务，通过编码器-解码器架构从音频-视觉特征中提取场景信息并动态生成室内声场，解决了传统方法无法适应多材料配置的挑战，构建了Acoustic Wonderland数据集并验证了模型在高精度声场生成方面的优越性。**

- **链接: [http://arxiv.org/pdf/2508.02905v1](http://arxiv.org/pdf/2508.02905v1)**

> **作者:** Mahnoor Fatima Saad; Ziad Al-Halah
>
> **备注:** Accepted to ICCV 2025. Project Page: https://mahnoor-fatima-saad.github.io/m-capa.html
>
> **摘要:** How would the sound in a studio change with a carpeted floor and acoustic tiles on the walls? We introduce the task of material-controlled acoustic profile generation, where, given an indoor scene with specific audio-visual characteristics, the goal is to generate a target acoustic profile based on a user-defined material configuration at inference time. We address this task with a novel encoder-decoder approach that encodes the scene's key properties from an audio-visual observation and generates the target Room Impulse Response (RIR) conditioned on the material specifications provided by the user. Our model enables the generation of diverse RIRs based on various material configurations defined dynamically at inference time. To support this task, we create a new benchmark, the Acoustic Wonderland Dataset, designed for developing and evaluating material-aware RIR prediction methods under diverse and challenging settings. Our results demonstrate that the proposed model effectively encodes material information and generates high-fidelity RIRs, outperforming several baselines and state-of-the-art methods.
>
---
#### [new 016] READ: Real-time and Efficient Asynchronous Diffusion for Audio-driven Talking Head Generation
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **简介: 该论文旨在解决音频驱动头像生成中实时性与效率不足的问题，提出READ框架，通过时空压缩视频潜意识、语音-视觉对齐和异步调度优化，实现高效、高质量的实时头像生成。**

- **链接: [http://arxiv.org/pdf/2508.03457v1](http://arxiv.org/pdf/2508.03457v1)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Jianqing Gao; Qingfeng Liu
>
> **备注:** 9 pages
>
> **摘要:** The introduction of diffusion models has brought significant advances to the field of audio-driven talking head generation. However, the extremely slow inference speed severely limits the practical implementation of diffusion-based talking head generation models. In this study, we propose READ, the first real-time diffusion-transformer-based talking head generation framework. Our approach first learns a spatiotemporal highly compressed video latent space via a temporal VAE, significantly reducing the token count to accelerate generation. To achieve better audio-visual alignment within this compressed latent space, a pre-trained Speech Autoencoder (SpeechAE) is proposed to generate temporally compressed speech latent codes corresponding to the video latent space. These latent representations are then modeled by a carefully designed Audio-to-Video Diffusion Transformer (A2V-DiT) backbone for efficient talking head synthesis. Furthermore, to ensure temporal consistency and accelerated inference in extended generation, we propose a novel asynchronous noise scheduler (ANS) for both the training and inference process of our framework. The ANS leverages asynchronous add-noise and asynchronous motion-guided generation in the latent space, ensuring consistency in generated video clips. Experimental results demonstrate that READ outperforms state-of-the-art methods by generating competitive talking head videos with significantly reduced runtime, achieving an optimal balance between quality and speed while maintaining robust metric stability in long-time generation.
>
---
## 更新

#### [replaced 001] Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02175v2](http://arxiv.org/pdf/2508.02175v2)**

> **作者:** Liang Lin; Miao Yu; Kaiwen Luo; Yibo Zhang; Lilan Peng; Dexian Wang; Xuehai Tang; Yuanhe Zhang; Xikang Yang; Zhenhong Zhou; Kun Wang; Yang Liu
>
> **摘要:** As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.
>
---
#### [replaced 002] BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.14971v3](http://arxiv.org/pdf/2410.14971v3)**

> **作者:** Jilong Li; Zhenxi Song; Jiaqi Wang; Meishan Zhang; Honghai Liu; Min Zhang; Zhiguo Zhang
>
> **备注:** 8 pages (excluding references), accepted by Findings of ACL 2025
>
> **摘要:** Current EEG/MEG-to-text decoding systems suffer from three key limitations: (1) reliance on teacher-forcing methods, which compromises robustness during inference, (2) sensitivity to session-specific noise, hindering generalization across subjects, and (3) misalignment between brain signals and linguistic representations due to pre-trained language model over-dominance. To overcome these challenges, we propose BrainECHO (Brain signal decoding via vEctor-quantized speCtrogram reconstruction for WHisper-enhanced text generatiOn), a multi-stage framework that employs decoupled representation learning to achieve state-of-the-art performance on both EEG and MEG datasets. Specifically, BrainECHO consists of three stages: (1) Discrete autoencoding, which transforms continuous Mel spectrograms into a finite set of high-quality discrete representations for subsequent stages. (2) Frozen alignment, where brain signal embeddings are mapped to corresponding Mel spectrogram embeddings in a frozen latent space, effectively filtering session-specific noise through vector-quantized reconstruction, yielding a 3.65% improvement in BLEU-4 score. (3) Constrained decoding fine-tuning, which leverages the pre-trained Whisper model for audio-to-text translation, balancing signal adaptation with knowledge preservation, and achieving 74%-89% decoding BLEU scores without excessive reliance on teacher forcing. BrainECHO demonstrates robustness across sentence, session, and subject-independent conditions, passing Gaussian noise tests and showcasing its potential for enhancing language-based brain-computer interfaces.
>
---
#### [replaced 003] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04134v3](http://arxiv.org/pdf/2506.04134v3)**

> **作者:** Jinting Wang; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading via hand coding, offering visual phonemic cues that support precise speech perception for the hearing-impaired. The task of CS Video-to-Speech generation (CSV2S) aims to convert CS videos into intelligible speech signals. Most existing research focuses on CS Recognition (CSR), which transcribes video content into text. Consequently, a common solution for CSV2S is to integrate CSR with a text-to-speech (TTS) system. However, this pipeline relies on text as an intermediate medium, which may lead to error propagation and temporal misalignment between speech and CS video dynamics. In contrast, directly generating audio speech from CS video (direct CSV2S) often suffers from the inherent multimodal complexity and the limited availability of CS data. To address these challenges, we propose UniCUE, the first unified framework for CSV2S that directly generates speech from CS videos without relying on intermediate text. The core innovation of UniCUE lies in integrating an understanding task (CSR) that provides fine-grained CS visual-semantic cues to guide speech generation. Specifically, UniCUE incorporates a pose-aware visual processor, a semantic alignment pool that enables precise visual-semantic mapping, and a VisioPhonetic adapter to bridge the understanding and generation tasks within a unified architecture. To support this framework, we construct UniCUE-HI, a large-scale Mandarin CS dataset containing 11282 videos from 14 cuers, including both hearing-impaired and normal-hearing individuals. Extensive experiments on this dataset demonstrate that UniCUE achieves state-of-the-art performance across multiple evaluation metrics.
>
---
#### [replaced 004] AudioGenie: A Training-Free Multi-Agent Framework for Diverse Multimodality-to-Multiaudio Generation
- **分类: cs.SD; cs.MA; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.22053v2](http://arxiv.org/pdf/2505.22053v2)**

> **作者:** Yan Rong; Jinting Wang; Guangzhi Lei; Shan Yang; Li Liu
>
> **摘要:** Multimodality-to-Multiaudio (MM2MA) generation faces significant challenges in synthesizing diverse and contextually aligned audio types (e.g., sound effects, speech, music, and songs) from multimodal inputs (e.g., video, text, images), owing to the scarcity of high-quality paired datasets and the lack of robust multi-task learning frameworks. Recently, multi-agent system shows great potential in tackling the above issues. However, directly applying it to MM2MA task presents three critical challenges: (1) inadequate fine-grained understanding of multimodal inputs (especially for video), (2) the inability of single models to handle diverse audio events, and (3) the absence of self-correction mechanisms for reliable outputs. To this end, we propose AudioGenie, a novel training-free multi-agent system featuring a dual-layer architecture with a generation team and a supervisor team. For the generation team, a fine-grained task decomposition and an adaptive Mixture-of-Experts (MoE) collaborative entity are designed for detailed comprehensive multimodal understanding and dynamic model selection, and a trial-and-error iterative refinement module is designed for self-correction. The supervisor team ensures temporal-spatial consistency and verifies outputs through feedback loops. Moreover, we build MA-Bench, the first benchmark for MM2MA tasks, comprising 198 annotated videos with multi-type audios. Experiments demonstrate that our AudioGenie achieves state-of-the-art (SOTA) or comparable performance across 9 metrics in 8 tasks. User study further validates the effectiveness of our method in terms of quality, accuracy, alignment, and aesthetic. The project website with audio samples can be found at https://audiogenie.github.io/.
>
---
#### [replaced 005] SemiSegECG: A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation
- **分类: cs.CV; cs.AI; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.18323v2](http://arxiv.org/pdf/2507.18323v2)**

> **作者:** Minje Park; Jeonghwa Lim; Taehyung Yu; Sunghoon Joo
>
> **备注:** Accepted by CIKM 2025. The code is available at https://github.com/bakqui/semi-seg-ecg
>
> **摘要:** Electrocardiogram (ECG) delineation, the segmentation of meaningful waveform features, is critical for clinical diagnosis. Despite recent advances using deep learning, progress has been limited by the scarcity of publicly available annotated datasets. Semi-supervised learning presents a promising solution by leveraging abundant unlabeled ECG data. In this study, we present SemiSegECG, the first systematic benchmark for semi-supervised semantic segmentation (SemiSeg) in ECG delineation. We curated and unified multiple public datasets, including previously underused sources, to support robust and diverse evaluation. We adopted five representative SemiSeg algorithms from computer vision, implemented them on two different architectures: the convolutional network and the transformer, and evaluated them in two different settings: in-domain and cross-domain. Additionally, we propose ECG-specific training configurations and augmentation strategies and introduce a standardized evaluation framework. Our results show that the transformer outperforms the convolutional network in semi-supervised ECG delineation. We anticipate that SemiSegECG will serve as a foundation for advancing semi-supervised ECG delineation methods and will facilitate further research in this domain.
>
---
#### [replaced 006] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.00733v3](http://arxiv.org/pdf/2508.00733v3)**

> **作者:** Le Wang; Jun Wang; Feng Deng; Chen Zhang; Di Zhang; Kun Gai
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** We present AudioGen-Omni - a unified approach based on multimodal diffusion transformers (MMDit), capable of generating high-fidelity audio, speech, and songs coherently synchronized with the input video. AudioGen-Omni introduces a novel joint training paradigm that seamlessly integrates large-scale video-text-audio corpora, enabling a model capable of generating semantically rich, acoustically diverse audio conditioned on multimodal inputs and adaptable to a wide range of audio generation tasks. AudioGen-Omni employs a unified lyrics-transcription encoder that encodes graphemes and phonemes from both sung and spoken inputs into dense frame-level representations. Dense frame-level representations are fused using an AdaLN-based joint attention mechanism enhanced with phase-aligned anisotropic positional infusion (PAAPI), wherein RoPE is selectively applied to temporally structured modalities to ensure precise and robust cross-modal alignment. By unfreezing all modalities and masking missing inputs, AudioGen-Omni mitigates the semantic constraints of text-frozen paradigms, enabling effective cross-modal conditioning. This joint training approach enhances audio quality, semantic alignment, and lip-sync accuracy, while also achieving state-of-the-art results on Text-to-Audio/Speech/Song tasks. With an inference time of 1.91 seconds for 8 seconds of audio, it offers substantial improvements in both efficiency and generality.
>
---
