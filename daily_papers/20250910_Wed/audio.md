# 音频 cs.SD;  eess.SP

- **最新发布 12 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Progressive Facial Granularity Aggregation with Bilateral Attribute-based Enhancement for Face-to-Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文研究面向语音合成的面部到语音（FTV）任务，旨在通过面部图像生成对应语音。针对现有方法丢失细粒度面部信息、训练效率低的问题，提出多粒度面部属性建模与多视角训练策略，提升语音与面部的一致性与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.07376v1](http://arxiv.org/pdf/2509.07376v1)**

> **作者:** Yejin Jeon; Youngjae Kim; Jihyun Lee; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** EMNLP Findings
>
> **摘要:** For individuals who have experienced traumatic events such as strokes, speech may no longer be a viable means of communication. While text-to-speech (TTS) can be used as a communication aid since it generates synthetic speech, it fails to preserve the user's own voice. As such, face-to-voice (FTV) synthesis, which derives corresponding voices from facial images, provides a promising alternative. However, existing methods rely on pre-trained visual encoders, and finetune them to align with speech embeddings, which strips fine-grained information from facial inputs such as gender or ethnicity, despite their known correlation with vocal traits. Moreover, these pipelines are multi-stage, which requires separate training of multiple components, thus leading to training inefficiency. To address these limitations, we utilize fine-grained facial attribute modeling by decomposing facial images into non-overlapping segments and progressively integrating them into a multi-granular representation. This representation is further refined through multi-task learning of speaker attributes such as gender and ethnicity at both the visual and acoustic domains. Moreover, to improve alignment robustness, we adopt a multi-view training strategy by pairing various visual perspectives of a speaker in terms of different angles and lighting conditions, with identical speech recordings. Extensive subjective and objective evaluations confirm that our approach substantially enhances face-voice congruence and synthesis stability.
>
---
#### [new 002] Prototype: A Keyword Spotting-Based Intelligent Audio SoC for IoT
- **分类: cs.SD; cs.AR; cs.HC; eess.AS**

- **简介: 该论文提出一种集成关键词识别加速器的智能音频SoC，用于IoT设备。旨在解决低功耗、低延迟语音交互问题，通过算法与硬件协同设计提升能效，并展示基于FPGA的原型实现。**

- **链接: [http://arxiv.org/pdf/2509.06964v1](http://arxiv.org/pdf/2509.06964v1)**

> **作者:** Huihong Liang; Dongxuan Jia; Youquan Wang; Longtao Huang; Shida Zhong; Luping Xiang; Lei Huang; Tao Yuan
>
> **摘要:** In this demo, we present a compact intelligent audio system-on-chip (SoC) integrated with a keyword spotting accelerator, enabling ultra-low latency, low-power, and low-cost voice interaction in Internet of Things (IoT) devices. Through algorithm-hardware co-design, the system's energy efficiency is maximized. We demonstrate the system's capabilities through a live FPGA-based prototype, showcasing stable performance and real-time voice interaction for edge intelligence applications.
>
---
#### [new 003] End-to-End Efficiency in Keyword Spotting: A System-Level Approach for Embedded Microcontrollers
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究嵌入式微控制器上的关键词识别（KWS）任务，旨在解决内存和能耗限制下的高效部署问题。作者系统比较了多种轻量神经网络架构，并提出基于MobileNet的TKWS模型，在STM32平台实现高精度、低参数与低功耗的KWS系统。**

- **链接: [http://arxiv.org/pdf/2509.07051v1](http://arxiv.org/pdf/2509.07051v1)**

> **作者:** Pietro Bartoli; Tommaso Bondini; Christian Veronesi; Andrea Giudici; Niccolò Antonello; Franco Zappa
>
> **备注:** 4 pages, 2 figures, 1 table. Accepted for publication in IEEE Sensors 2025. \c{opyright} 2025 IEEE. Personal use permitted. Permission from IEEE required for all other uses
>
> **摘要:** Keyword spotting (KWS) is a key enabling technology for hands-free interaction in embedded and IoT devices, where stringent memory and energy constraints challenge the deployment of AI-enabeld devices. In this work, we systematically evaluate and compare several state-of-the-art lightweight neural network architectures, including DS-CNN, LiCoNet, and TENet, alongside our proposed Typman-KWS (TKWS) architecture built upon MobileNet, specifically designed for efficient KWS on microcontroller units (MCUs). Unlike prior studies focused solely on model inference, our analysis encompasses the entire processing pipeline, from Mel-Frequency Cepstral Coefficient (MFCC) feature extraction to neural inference, and is benchmarked across three STM32 platforms (N6, H7, and U5). Our results show that TKWS with three residual blocks achieves up to 92.4% F1-score with only 14.4k parameters, reducing memory footprint without compromising the accuracy. Moreover, the N6 MCU with integrated neural acceleration achieves the best energy-delay product (EDP), enabling efficient, low-latency operation even with high-resolution features. Our findings highlight the model accuracy alone does not determine real-world effectiveness; rather, optimal keyword spotting deployments require careful consideration of feature extraction parameters and hardware-specific optimization.
>
---
#### [new 004] Target matching based generative model for speech enhancement
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决生成模型中均值/方差调度设计及训练效率问题。提出基于目标信号估计的生成框架，消除随机成分，提升训练稳定性与效率，并设计新型扩散主干网络以优化音频处理性能。**

- **链接: [http://arxiv.org/pdf/2509.07521v1](http://arxiv.org/pdf/2509.07521v1)**

> **作者:** Taihui Wang; Rilin Chen; Tong Lei; Andong Li; Jinzheng Zhao; Meng Yu; Dong Yu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** The design of mean and variance schedules for the perturbed signal is a fundamental challenge in generative models. While score-based and Schr\"odinger bridge-based models require careful selection of the stochastic differential equation to derive the corresponding schedules, flow-based models address this issue via vector field matching. However, this strategy often leads to hallucination artifacts and inefficient training and inference processes due to the potential inclusion of stochastic components in the vector field. Additionally, the widely adopted diffusion backbone, NCSN++, suffers from high computational complexity. To overcome these limitations, we propose a novel target-based generative framework that enhances both the flexibility of mean/variance schedule design and the efficiency of training and inference processes. Specifically, we eliminate the stochastic components in the training loss by reformulating the generative speech enhancement task as a target signal estimation problem, which therefore leads to more stable and efficient training and inference processes. In addition, we employ a logistic mean schedule and a bridge variance schedule, which yield a more favorable signal-to-noise ratio trajectory compared to several widely used schedules and thus leads to a more efficient perturbation strategy. Furthermore, we propose a new diffusion backbone for audio, which significantly improves the efficiency over NCSN++ by explicitly modeling long-term frame correlations and cross-band dependencies.
>
---
#### [new 005] When Fine-Tuning is Not Enough: Lessons from HSAD on Hybrid and Adversarial Audio Spoof Detection
- **分类: cs.SD; cs.CR**

- **简介: 该论文属于音频防伪检测任务，旨在解决混合型语音攻击的检测难题。论文提出了HSAD数据集，并评估多种模型，揭示微调不足，强调需使用混合感知基准提升系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.07323v1](http://arxiv.org/pdf/2509.07323v1)**

> **作者:** Bin Hu; Kunyang Huang; Daehan Kwak; Meng Xu; Kuan Huang
>
> **备注:** 13 pages, 11 figures.This work has been submitted to the IEEE for possible publication
>
> **摘要:** The rapid advancement of AI has enabled highly realistic speech synthesis and voice cloning, posing serious risks to voice authentication, smart assistants, and telecom security. While most prior work frames spoof detection as a binary task, real-world attacks often involve hybrid utterances that mix genuine and synthetic speech, making detection substantially more challenging. To address this gap, we introduce the Hybrid Spoofed Audio Dataset (HSAD), a benchmark containing 1,248 clean and 41,044 degraded utterances across four classes: human, cloned, zero-shot AI-generated, and hybrid audio. Each sample is annotated with spoofing method, speaker identity, and degradation metadata to enable fine-grained analysis. We evaluate six transformer-based models, including spectrogram encoders (MIT-AST, MattyB95-AST) and self-supervised waveform models (Wav2Vec2, HuBERT). Results reveal critical lessons: pretrained models overgeneralize and collapse under hybrid conditions; spoof-specific fine-tuning improves separability but struggles with unseen compositions; and dataset-specific adaptation on HSAD yields large performance gains (AST greater than 97 percent and F1 score is approximately 99 percent), though residual errors persist for complex hybrids. These findings demonstrate that fine-tuning alone is not sufficient-robust hybrid-aware benchmarks like HSAD are essential to expose calibration failures, model biases, and factors affecting spoof detection in adversarial environments. HSAD thus provides both a dataset and an analytic framework for building resilient and trustworthy voice authentication systems.
>
---
#### [new 006] Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations
- **分类: cs.SD; cs.LG; eess.AS; 68T07; H.5.5; J.5; I.5.4**

- **简介: 论文提出用神经网络代理模拟合成器，解决自动合成器编程中非微分问题。通过预训练音频模型映射预设到音频嵌入空间，提升黑盒合成器的神经编程效果。**

- **链接: [http://arxiv.org/pdf/2509.07635v1](http://arxiv.org/pdf/2509.07635v1)**

> **作者:** Paolo Combes; Stefan Weinzierl; Klaus Obermayer
>
> **备注:** 17 pages, 4 figures, published in the Journal of the Audio Engineering Society
>
> **摘要:** Deep learning appears as an appealing solution for Automatic Synthesizer Programming (ASP), which aims to assist musicians and sound designers in programming sound synthesizers. However, integrating software synthesizers into training pipelines is challenging due to their potential non-differentiability. This work tackles this challenge by introducing a method to approximate arbitrary synthesizers. Specifically, we train a neural network to map synthesizer presets onto an audio embedding space derived from a pretrained model. This facilitates the definition of a neural proxy that produces compact yet effective representations, thereby enabling the integration of audio embedding loss into neural-based ASP systems for black-box synthesizers. We evaluate the representations derived by various pretrained audio models in the context of neural-based nASP and assess the effectiveness of several neural network architectures, including feedforward, recurrent, and transformer-based models, in defining neural proxies. We evaluate the proposed method using both synthetic and hand-crafted presets from three popular software synthesizers and assess its performance in a synthesizer sound matching downstream task. While the benefits of the learned representation are nuanced by resource requirements, encouraging results were obtained for all synthesizers, paving the way for future research into the application of synthesizer proxies for neural-based ASP systems.
>
---
#### [new 007] Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出SMIA攻击方法，针对语音认证与反欺骗系统，通过操控不可听频段生成对抗样本，成功绕过检测模型。研究揭示当前系统在面对自适应攻击时的脆弱性，强调需发展动态防御机制。**

- **链接: [http://arxiv.org/pdf/2509.07677v1](http://arxiv.org/pdf/2509.07677v1)**

> **作者:** Kamel Kamel; Hridoy Sankar Dutta; Keshav Sood; Sunil Aryal
>
> **摘要:** Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.
>
---
#### [new 008] Controllable Singing Voice Synthesis using Phoneme-Level Energy Sequence
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于可控歌声合成任务，旨在解决动态控制不足的问题。提出基于音素级能量序列的可控方法，通过真实频谱提取能量序列，提升用户对音量变化的控制能力，实验显示其显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.07038v1](http://arxiv.org/pdf/2509.07038v1)**

> **作者:** Yerin Ryu; Inseop Shin; Chanwoo Kim
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Controllable Singing Voice Synthesis (SVS) aims to generate expressive singing voices reflecting user intent. While recent SVS systems achieve high audio quality, most rely on probabilistic modeling, limiting precise control over attributes such as dynamics. We address this by focusing on dynamic control--temporal loudness variation essential for musical expressiveness--and explicitly condition the SVS model on energy sequences extracted from ground-truth spectrograms, reducing annotation costs and improving controllability. We also propose a phoneme-level energy sequence for user-friendly control. To the best of our knowledge, this is the first attempt enabling user-driven dynamics control in SVS. Experiments show our method achieves over 50% reduction in mean absolute error of energy sequences for phoneme-level inputs compared to baseline and energy-predictor models, without compromising synthesis quality.
>
---
#### [new 009] Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study
- **分类: cs.SD; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究音频深度伪造检测方法在对抗攻击下的性能，分析其优缺点，评估五种数据集上的表现，旨在提升检测器的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.07132v1](http://arxiv.org/pdf/2509.07132v1)**

> **作者:** Kutub Uddin; Muhammad Umar Farooq; Awais Khan; Khalid Mahmood Malik
>
> **摘要:** The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques.
>
---
#### [new 010] Competitive Audio-Language Models with Data-Efficient Single-Stage Training on Public Data
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Falcon3-Audio，一种高效训练的音频-语言模型，解决音频与语言整合不足的问题。利用少量公开音频数据（<30K小时），实现与大模型相当的性能，强调参数和数据效率，采用单阶段训练，无需复杂结构。**

- **链接: [http://arxiv.org/pdf/2509.07526v1](http://arxiv.org/pdf/2509.07526v1)**

> **作者:** Gokul Karthik Kumar; Rishabh Saraf; Ludovick Lepauloux; Abdul Muneer; Billel Mokeddem; Hakim Hacid
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** Large language models (LLMs) have transformed NLP, yet their integration with audio remains underexplored -- despite audio's centrality to human communication. We introduce Falcon3-Audio, a family of Audio-Language Models (ALMs) built on instruction-tuned LLMs and Whisper encoders. Using a remarkably small amount of public audio data -- less than 30K hours (5K unique) -- Falcon3-Audio-7B matches the best reported performance among open-weight models on the MMAU benchmark, with a score of 64.14, matching R1-AQA, while distinguishing itself through superior data and parameter efficiency, single-stage training, and transparency. Notably, our smallest 1B model remains competitive with larger open models ranging from 2B to 13B parameters. Through extensive ablations, we find that common complexities -- such as curriculum learning, multiple audio encoders, and intricate cross-attention connectors -- are not required for strong performance, even compared to models trained on over 500K hours of data.
>
---
#### [new 011] Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep Convolutional Neural Networks
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS**

- **简介: 论文研究使用深度卷积神经网络对音频进行分类任务，比较不同频谱和节奏特征在环境声音数据集上的表现，发现梅尔频谱图和MFCC效果最佳。**

- **链接: [http://arxiv.org/pdf/2509.07756v1](http://arxiv.org/pdf/2509.07756v1)**

> **作者:** Friedrich Wolf-Monheim
>
> **摘要:** Next to decision tree and k-nearest neighbours algorithms deep convolutional neural networks (CNNs) are widely used to classify audio data in many domains like music, speech or environmental sounds. To train a specific CNN various spectral and rhythm features like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCC), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams can be used as digital image input data for the neural network. The performance of these spectral and rhythm features for audio category level as well as audio class level classification is investigated in detail with a deep CNN and the ESC-50 dataset with 2,000 labeled environmental audio recordings using an end-to-end deep learning pipeline. The evaluated metrics accuracy, precision, recall and F1 score for multiclass classification clearly show that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCC) perform significantly better then the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs.
>
---
#### [new 012] Exploring System Adaptations For Minimum Latency Real-Time Piano Transcription
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究如何将现有在线钢琴转录模型适应于低延迟实时场景。通过消除非因果处理、优化计算共享与模型规模，探索预处理与标签编码策略，以实现低于30ms的延迟。实验表明，严格因果处理导致精度下降，存在延迟与准确率的权衡。**

- **链接: [http://arxiv.org/pdf/2509.07586v1](http://arxiv.org/pdf/2509.07586v1)**

> **作者:** Patricia Hu; Silvan David Peter; Jan Schlüter; Gerhard Widmer
>
> **备注:** to be published in Proceedings of the 26th International Society for Music Information Retrieval (ISMIR) Conference 2025, Daejeon, South Korea
>
> **摘要:** Advances in neural network design and the availability of large-scale labeled datasets have driven major improvements in piano transcription. Existing approaches target either offline applications, with no restrictions on computational demands, or online transcription, with delays of 128-320 ms. However, most real-time musical applications require latencies below 30 ms. In this work, we investigate whether and how the current state-of-the-art online transcription model can be adapted for real-time piano transcription. Specifically, we eliminate all non-causal processing, and reduce computational load through shared computations across core model components and variations in model size. Additionally, we explore different pre- and postprocessing strategies, and related label encoding schemes, and discuss their suitability for real-time transcription. Evaluating the adaptions on the MAESTRO dataset, we find a drop in transcription accuracy due to strictly causal processing as well as a tradeoff between the preprocessing latency and prediction accuracy. We release our system as a baseline to support researchers in designing models towards minimum latency real-time transcription.
>
---
## 更新

#### [replaced 001] HingeNet: A Harmonic-Aware Fine-Tuning Approach for Beat Tracking
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.09788v2](http://arxiv.org/pdf/2508.09788v2)**

> **作者:** Ganghui Ru; Jieying Wang; Jiahao Zhao; Yulun Wu; Yi Yu; Nannan Jiang; Wei Wang; Wei Li
>
> **备注:** Early draft for discussion only. Undergoing active revision, conclusions subject to change. Do not cite. Formal peer-reviewed version in preparation
>
> **摘要:** Fine-tuning pre-trained foundation models has made significant progress in music information retrieval. However, applying these models to beat tracking tasks remains unexplored as the limited annotated data renders conventional fine-tuning methods ineffective. To address this challenge, we propose HingeNet, a novel and general parameter-efficient fine-tuning method specifically designed for beat tracking tasks. HingeNet is a lightweight and separable network, visually resembling a hinge, designed to tightly interface with pre-trained foundation models by using their intermediate feature representations as input. This unique architecture grants HingeNet broad generalizability, enabling effective integration with various pre-trained foundation models. Furthermore, considering the significance of harmonics in beat tracking, we introduce harmonic-aware mechanism during the fine-tuning process to better capture and emphasize the harmonic structures in musical signals. Experiments on benchmark datasets demonstrate that HingeNet achieves state-of-the-art performance in beat and downbeat tracking
>
---
#### [replaced 002] Continuous Audio Language Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.06926v2](http://arxiv.org/pdf/2509.06926v2)**

> **作者:** Simon Rouard; Manu Orsini; Axel Roebel; Neil Zeghidour; Alexandre Défossez
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Audio Language Models (ALM) have emerged as the dominant paradigm for speech and music generation by representing audio as sequences of discrete tokens. Yet, unlike text tokens, which are invertible, audio tokens are extracted from lossy codecs with a limited bitrate. As a consequence, increasing audio quality requires generating more tokens, which imposes a trade-off between fidelity and computational cost. We address this issue by studying Continuous Audio Language Models (CALM). These models instantiate a large Transformer backbone that produces a contextual embedding at every timestep. This sequential information then conditions an MLP that generates the next continuous frame of an audio VAE through consistency modeling. By avoiding lossy compression, CALM achieves higher quality at lower computational cost than their discrete counterpart. Experiments on speech and music demonstrate improved efficiency and fidelity over state-of-the-art discrete audio language models, facilitating lightweight, high-quality audio generation. Samples are available at hf.co/spaces/kyutai/calm-samples
>
---
#### [replaced 003] Learning to Upsample and Upmix Audio in the Latent Domain
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00681v2](http://arxiv.org/pdf/2506.00681v2)**

> **作者:** Dimitrios Bralios; Paris Smaragdis; Jonah Casebeer
>
> **摘要:** Neural audio autoencoders create compact latent representations that preserve perceptually important information, serving as the foundation for both modern audio compression systems and generation approaches like next-token prediction and latent diffusion. Despite their prevalence, most audio processing operations, such as spatial and spectral up-sampling, still inefficiently operate on raw waveforms or spectral representations rather than directly on these compressed representations. We propose a framework that performs audio processing operations entirely within an autoencoder's latent space, eliminating the need to decode to raw audio formats. Our approach dramatically simplifies training by operating solely in the latent domain, with a latent L1 reconstruction term, augmented by a single latent adversarial discriminator. This contrasts sharply with raw-audio methods that typically require complex combinations of multi-scale losses and discriminators. Through experiments in bandwidth extension and mono-to-stereo up-mixing, we demonstrate computational efficiency gains of up to 100x while maintaining quality comparable to post-processing on raw audio. This work establishes a more efficient paradigm for audio processing pipelines that already incorporate autoencoders, enabling significantly faster and more resource-efficient workflows across various audio tasks.
>
---
#### [replaced 004] The Model Hears You: Audio Language Model Deployments Should Consider the Principle of Least Privilege
- **分类: cs.SD; cs.AI; cs.CL; cs.CY; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.16833v2](http://arxiv.org/pdf/2503.16833v2)**

> **作者:** Luxi He; Xiangyu Qi; Michel Liao; Inyoung Cheong; Prateek Mittal; Danqi Chen; Peter Henderson
>
> **备注:** Published at AIES 2025
>
> **摘要:** The latest Audio Language Models (Audio LMs) process speech directly instead of relying on a separate transcription step. This shift preserves detailed information, such as intonation or the presence of multiple speakers, that would otherwise be lost in transcription. However, it also introduces new safety risks, including the potential misuse of speaker identity cues and other sensitive vocal attributes, which could have legal implications. In this paper, we urge a closer examination of how these models are built and deployed. Our experiments show that end-to-end modeling, compared with cascaded pipelines, creates socio-technical safety risks such as identity inference, biased decision-making, and emotion detection. This raises concerns about whether Audio LMs store voiceprints and function in ways that create uncertainty under existing legal regimes. We then argue that the Principle of Least Privilege should be considered to guide the development and deployment of these models. Specifically, evaluations should assess (1) the privacy and safety risks associated with end-to-end modeling; and (2) the appropriate scope of information access. Finally, we highlight related gaps in current audio LM benchmarks and identify key open research questions, both technical and policy-related, that must be addressed to enable the responsible deployment of end-to-end Audio LMs.
>
---
#### [replaced 005] SaD: A Scenario-Aware Discriminator for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.00405v2](http://arxiv.org/pdf/2509.00405v2)**

> **作者:** Xihao Yuan; Siqi Liu; Yan Chen; Hang Zhou; Chang Liu; Hanting Chen; Jie Hu
>
> **备注:** 5 pages, 2 figures. Accepted by InterSpeech2025
>
> **摘要:** Generative adversarial network-based models have shown remarkable performance in the field of speech enhancement. However, the current optimization strategies for these models predominantly focus on refining the architecture of the generator or enhancing the quality evaluation metrics of the discriminator. This approach often overlooks the rich contextual information inherent in diverse scenarios. In this paper, we propose a scenario-aware discriminator that captures scene-specific features and performs frequency-domain division, thereby enabling a more accurate quality assessment of the enhanced speech generated by the generator. We conducted comprehensive experiments on three representative models using two publicly available datasets. The results demonstrate that our method can effectively adapt to various generator architectures without altering their structure, thereby unlocking further performance gains in speech enhancement across different scenarios.
>
---
#### [replaced 006] BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2508.09790v2](http://arxiv.org/pdf/2508.09790v2)**

> **作者:** Ganghui Ru; Jieying Wang; Jiahao Zhao; Yulun Wu; Yi Yu; Nannan Jiang; Wei Wang; Wei Li
>
> **备注:** Early draft for discussion only. Undergoing active revision, conclusions subject to change. Do not cite. Formal peer-reviewed version in preparation
>
> **摘要:** Beat tracking is a widely researched topic in music information retrieval. However, current beat tracking methods face challenges due to the scarcity of labeled data, which limits their ability to generalize across diverse musical styles and accurately capture complex rhythmic structures. To overcome these challenges, we propose a novel beat tracking paradigm BeatFM, which introduces a pre-trained music foundation model and leverages its rich semantic knowledge to improve beat tracking performance. Pre-training on diverse music datasets endows music foundation models with a robust understanding of music, thereby effectively addressing these challenges. To further adapt it for beat tracking, we design a plug-and-play multi-dimensional semantic aggregation module, which is composed of three parallel sub-modules, each focusing on semantic aggregation in the temporal, frequency, and channel domains, respectively. Extensive experiments demonstrate that our method achieves state-of-the-art performance in beat and downbeat tracking across multiple benchmark datasets.
>
---
#### [replaced 007] Re-Bottleneck: Latent Re-Structuring for Neural Audio Autoencoders
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07867v2](http://arxiv.org/pdf/2507.07867v2)**

> **作者:** Dimitrios Bralios; Jonah Casebeer; Paris Smaragdis
>
> **备注:** Accepted at IEEE MLSP 2025
>
> **摘要:** Neural audio codecs and autoencoders have emerged as versatile models for audio compression, transmission, feature-extraction, and latent-space generation. However, a key limitation is that most are trained to maximize reconstruction fidelity, often neglecting the specific latent structure necessary for optimal performance in diverse downstream applications. We propose a simple, post-hoc framework to address this by modifying the bottleneck of a pre-trained autoencoder. Our method introduces a "Re-Bottleneck", an inner bottleneck trained exclusively through latent space losses to instill user-defined structure. We demonstrate the framework's effectiveness in three experiments. First, we enforce an ordering on latent channels without sacrificing reconstruction quality. Second, we align latents with semantic embeddings, analyzing the impact on downstream diffusion modeling. Third, we introduce equivariance, ensuring that a filtering operation on the input waveform directly corresponds to a specific transformation in the latent space. Ultimately, our Re-Bottleneck framework offers a flexible and efficient way to tailor representations of neural audio models, enabling them to seamlessly meet the varied demands of different applications with minimal additional training.
>
---
#### [replaced 008] Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.04795v2](http://arxiv.org/pdf/2508.04795v2)**

> **作者:** Thomas Thebaud; Yen-Ju Lu; Matthew Wiesner; Peter Viechnicki; Najim Dehak
>
> **备注:** Accepted in the 2025 IEEE Automatic Speech Recognition and Understanding Workshop
>
> **摘要:** In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios.
>
---
#### [replaced 009] When Large Language Models Meet Speech: A Survey on Integration Approaches
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.19548v2](http://arxiv.org/pdf/2502.19548v2)**

> **作者:** Zhengdong Yang; Shuichiro Shimizu; Yahan Yu; Chenhui Chu
>
> **备注:** Accepted at Findings of ACL 2025 (Long Paper)
>
> **摘要:** Recent advancements in large language models (LLMs) have spurred interest in expanding their application beyond text-based tasks. A large number of studies have explored integrating other modalities with LLMs, notably speech modality, which is naturally related to text. This paper surveys the integration of speech with LLMs, categorizing the methodologies into three primary approaches: text-based, latent-representation-based, and audio-token-based integration. We also demonstrate how these methods are applied across various speech-related applications and highlight the challenges in this field to offer inspiration for
>
---
