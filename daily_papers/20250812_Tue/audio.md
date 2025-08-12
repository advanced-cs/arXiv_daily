# 音频 cs.SD;  eess.SP

- **最新发布 34 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] SEF-MK: Speaker-Embedding-Free Voice Anonymization through Multi-k-means Quantization
- **分类: cs.SD; cs.LG**

- **简介: 论文提出SEF-MK框架，通过多k-means模型随机选择不同子集进行语音匿名化，旨在保护说话人隐私同时保留语言和情感内容，用户视角下效果更好，但攻击者角度更易被破解。**

- **链接: [http://arxiv.org/pdf/2508.07086v1](http://arxiv.org/pdf/2508.07086v1)**

> **作者:** Beilong Tang; Xiaoxiao Miao; Xin Wang; Ming Li
>
> **备注:** 8 pages, 3 figures, accepted by 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)
>
> **摘要:** Voice anonymization protects speaker privacy by concealing identity while preserving linguistic and paralinguistic content. Self-supervised learning (SSL) representations encode linguistic features but preserve speaker traits. We propose a novel speaker-embedding-free framework called SEF-MK. Instead of using a single k-means model trained on the entire dataset, SEF-MK anonymizes SSL representations for each utterance by randomly selecting one of multiple k-means models, each trained on a different subset of speakers. We explore this approach from both attacker and user perspectives. Extensive experiments show that, compared to a single k-means model, SEF-MK with multiple k-means models better preserves linguistic and emotional content from the user's viewpoint. However, from the attacker's perspective, utilizing multiple k-means models boosts the effectiveness of privacy attacks. These insights can aid users in designing voice anonymization systems to mitigate attacker threats.
>
---
#### [new 002] Acoustic source depth estimation method based on a single hydrophone in Arctic underwater
- **分类: cs.SD; cs.NA; math.NA; physics.ao-ph; physics.app-ph**

- **简介: 本文基于单个水听器，利用正常模态和光线理论，提出多方法估算声源深度，验证其在北极深海中的适用性。**

- **链接: [http://arxiv.org/pdf/2508.07157v1](http://arxiv.org/pdf/2508.07157v1)**

> **作者:** Jinbao Weng; Yubo Qi; Yanming Yang; Hongtao Wen; Hongtao Zhou; Benqing Chen; Dewei Xu; Ruichao Xue; Caigao Zeng
>
> **摘要:** Based on the normal mode and ray theory, this article discusses the characteristics of surface sound source and reception at the surface layer, and explores depth estimation methods based on normal modes and rays, and proposes a depth estimation method based on the upper limit of modal frequency. Data verification is conducted to discuss the applicability and limitations of different methods. For the surface refracted normal mode waveguide, modes can be separated through warping transformation. Based on the characteristics of normal mode amplitude variation with frequency and number, the sound source depth can be estimated by matching amplitude information. Based on the spatial variation characteristics of eigenfunctions with frequency, a sound source depth estimation method matching the cutoff frequency of normal modes is proposed. For the deep Arctic sea, the sound ray arrival structure at the receiving end is obtained through the analysis of deep inversion sound ray trajectories, and the sound source depth can be estimated by matching the time difference of ray arrivals. Experimental data is used to verify the sound field patterns and the effectiveness of the sound source depth estimation method.
>
---
#### [new 003] AutoMashup: Automatic Music Mashups Creation
- **分类: cs.SD; eess.SP**

- **简介: 论文提出AutoMashup系统，通过源分离、音乐分析与兼容性评估自动创建音乐混音，发现兼容性存在不对称性且通用模型难以准确反映感知一致性。**

- **链接: [http://arxiv.org/pdf/2508.06516v1](http://arxiv.org/pdf/2508.06516v1)**

> **作者:** Marine Delabaere; Léa Miqueu; Michael Moreno; Gautier Bigois; Hoang Duong; Ella Fernandez; Flavie Manent; Maria Salgado-Herrera; Bastien Pasdeloup; Nicolas Farrugia; Axel Marmoret
>
> **摘要:** We introduce AutoMashup, a system for automatic mashup creation based on source separation, music analysis, and compatibility estimation. We propose using COCOLA to assess compatibility between separated stems and investigate whether general-purpose pretrained audio models (CLAP and MERT) can support zero-shot estimation of track pair compatibility. Our results show that mashup compatibility is asymmetric -- it depends on the role assigned to each track (vocals or accompaniment) -- and that current embeddings fail to reproduce the perceptual coherence measured by COCOLA. These findings underline the limitations of general-purpose audio representations for compatibility estimation in mashup creation.
>
---
#### [new 004] Whisfusion: Parallel ASR Decoding via a Diffusion Transformer
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 论文提出Whisfusion框架，通过扩散Transformer实现并行ASR解码，解决AR解码延迟问题，采用轻量跨注意力适配器和批处理策略提升精度与速度。**

- **链接: [http://arxiv.org/pdf/2508.07048v1](http://arxiv.org/pdf/2508.07048v1)**

> **作者:** Taeyoun Kwon; Junhyuk Ahn; Taegeun Yun; Heeju Jwa; Yoonchae Choi; Siwon Park; Nam-Joon Kim; Jangchan Kim; Hyun Gon Ryu; Hyuk-Jae Lee
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Fast Automatic Speech Recognition (ASR) is critical for latency-sensitive applications such as real-time captioning and meeting transcription. However, truly parallel ASR decoding remains challenging due to the sequential nature of autoregressive (AR) decoders and the context limitations of non-autoregressive (NAR) methods. While modern ASR encoders can process up to 30 seconds of audio at once, AR decoders still generate tokens sequentially, creating a latency bottleneck. We propose Whisfusion, the first framework to fuse a pre-trained Whisper encoder with a text diffusion decoder. This NAR architecture resolves the AR latency bottleneck by processing the entire acoustic context in parallel at every decoding step. A lightweight cross-attention adapter trained via parameter-efficient fine-tuning (PEFT) bridges the two modalities. We also introduce a batch-parallel, multi-step decoding strategy that improves accuracy by increasing the number of candidates with minimal impact on speed. Fine-tuned solely on LibriSpeech (960h), Whisfusion achieves a lower WER than Whisper-tiny (8.3% vs. 9.7%), and offers comparable latency on short audio. For longer utterances (>20s), it is up to 2.6x faster than the AR baseline, establishing a new, efficient operating point for long-form ASR. The implementation and training scripts are available at https://github.com/taeyoun811/Whisfusion.
>
---
#### [new 005] Noise-Robust Sound Event Detection and Counting via Language-Queried Sound Separation
- **分类: cs.SD**

- **简介: 论文提出一种基于语言查询的声学事件检测与计数方法，解决噪声环境下SED性能下降问题，通过事件出现检测（EAD）计数并结合多任务学习框架提升鲁棒性，任务约束增强预测一致性，实验显示在高噪声下优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.07176v1](http://arxiv.org/pdf/2508.07176v1)**

> **作者:** Yuanjian Chen; Yang Xiao; Han Yin; Yadong Guan; Xubo Liu
>
> **摘要:** Most sound event detection (SED) systems perform well on clean datasets but degrade significantly in noisy environments. Language-queried audio source separation (LASS) models show promise for robust SED by separating target events; existing methods require elaborate multi-stage training and lack explicit guidance for target events. To address these challenges, we introduce event appearance detection (EAD), a counting-based approach that counts event occurrences at both the clip and frame levels. Based on EAD, we propose a co-training-based multi-task learning framework for EAD and SED to enhance SED's performance in noisy environments. First, SED struggles to learn the same patterns as EAD. Then, a task-based constraint is designed to improve prediction consistency between SED and EAD. This framework provides more reliable clip-level predictions for LASS models and strengthens timestamp detection capability. Experiments on DESED and WildDESED datasets demonstrate better performance compared to existing methods, with advantages becoming more pronounced at higher noise levels.
>
---
#### [new 006] Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning
- **分类: cs.SD; cs.CL; cs.MM**

- **简介: 论文提出Audio-Thinker框架，通过强化学习指导音频语言模型动态调整思考策略，结合适应性奖励与外部评估提升推理一致性，解决现有模型在音频问答中表现不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.08039v1](http://arxiv.org/pdf/2508.08039v1)**

> **作者:** Shu Wu; Chenxing Li; Wenfu Wang; Hao Zhang; Hualei Wang; Meng Yu; Dong Yu
>
> **备注:** preprint
>
> **摘要:** Recent advancements in large language models, multimodal large language models, and large audio language models (LALMs) have significantly improved their reasoning capabilities through reinforcement learning with rule-based rewards. However, the explicit reasoning process has yet to show significant benefits for audio question answering, and effectively leveraging deep reasoning remains an open challenge, with LALMs still falling short of human-level auditory-language reasoning. To address these limitations, we propose Audio-Thinker, a reinforcement learning framework designed to enhance the reasoning capabilities of LALMs, with a focus on improving adaptability, consistency, and effectiveness. Our approach introduces an adaptive think accuracy reward, enabling the model to adjust its reasoning strategies based on task complexity dynamically. Furthermore, we incorporate an external reward model to evaluate the overall consistency and quality of the reasoning process, complemented by think-based rewards that help the model distinguish between valid and flawed reasoning paths during training. Experimental results demonstrate that our Audio-Thinker model outperforms existing reasoning-oriented LALMs across various benchmark tasks, exhibiting superior reasoning and generalization capabilities.
>
---
#### [new 007] Exploring Efficient Directional and Distance Cues for Regional Speech Separation
- **分类: cs.SD**

- **简介: 论文提出基于神经网络的麦克风阵列方法，通过改进延迟和求和技术及直达声/混响比增强方向性和距离感知，提升区域语音分离性能。**

- **链接: [http://arxiv.org/pdf/2508.07563v1](http://arxiv.org/pdf/2508.07563v1)**

> **作者:** Yiheng Jiang; Haoxu Wang; Yafeng Chen; Gang Qiao; Biao Tian
>
> **备注:** This paper has been accepted by Interspeech 2025
>
> **摘要:** In this paper, we introduce a neural network-based method for regional speech separation using a microphone array. This approach leverages novel spatial cues to extract the sound source not only from specified direction but also within defined distance. Specifically, our method employs an improved delay-and-sum technique to obtain directional cues, substantially enhancing the signal from the target direction. We further enhance separation by incorporating the direct-to-reverberant ratio into the input features, enabling the model to better discriminate sources within and beyond a specified distance. Experimental results demonstrate that our proposed method leads to substantial gains across multiple objective metrics. Furthermore, our method achieves state-of-the-art performance on the CHiME-8 MMCSG dataset, which was recorded in real-world conversational scenarios, underscoring its effectiveness for speech separation in practical applications.
>
---
#### [new 008] Maestro-EVC: Controllable Emotional Voice Conversion Guided by References and Explicit Prosody
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 论文提出一种可控的情绪语音转换（EVC）框架Maestro-EVC，通过分离内容、说话人身份和情感属性到独立参考，结合时间情感表示与显式音调建模，解决现有方法难以分离属性及捕捉细粒度情感动态的问题，提升跨条件下的情感迁移能力。**

- **链接: [http://arxiv.org/pdf/2508.06890v1](http://arxiv.org/pdf/2508.06890v1)**

> **作者:** Jinsung Yoon; Wooyeol Jeong; Jio Gim; Young-Joo Suh
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** Emotional voice conversion (EVC) aims to modify the emotional style of speech while preserving its linguistic content. In practical EVC, controllability, the ability to independently control speaker identity and emotional style using distinct references, is crucial. However, existing methods often struggle to fully disentangle these attributes and lack the ability to model fine-grained emotional expressions such as temporal dynamics. We propose Maestro-EVC, a controllable EVC framework that enables independent control of content, speaker identity, and emotion by effectively disentangling each attribute from separate references. We further introduce a temporal emotion representation and an explicit prosody modeling with prosody augmentation to robustly capture and transfer the temporal dynamics of the target emotion, even under prosody-mismatched conditions. Experimental results confirm that Maestro-EVC achieves high-quality, controllable, and emotionally expressive speech synthesis.
>
---
#### [new 009] Keyword Mamba: Spoken Keyword Spotting with State Space Models
- **分类: cs.SD; eess.AS**

- **简介: 论文提出Keyword Mamba，用于语音关键词检测（KWS），通过状态空间模型（SSM）解决长时依赖与效率矛盾，替代Transformer自注意力机制，实现高精度低计算成本，首次将SSM应用于KWS。**

- **链接: [http://arxiv.org/pdf/2508.07363v1](http://arxiv.org/pdf/2508.07363v1)**

> **作者:** Hanyu Ding; Wenlong Dong; Qirong Mao
>
> **备注:** Under peer review
>
> **摘要:** Keyword spotting (KWS) is an essential task in speech processing. It is widely used in voice assistants and smart devices. Deep learning models like CNNs, RNNs, and Transformers have performed well in KWS. However, they often struggle to handle long-term patterns and stay efficient at the same time. In this work, we present Keyword Mamba, a new architecture for KWS. It uses a neural state space model (SSM) called Mamba. We apply Mamba along the time axis and also explore how it can replace the self-attention part in Transformer models. We test our model on the Google Speech Commands datasets. The results show that Keyword Mamba reaches strong accuracy with fewer parameters and lower computational cost. To our knowledge, this is the first time a state space model has been used for KWS. These results suggest that Mamba has strong potential in speech-related tasks.
>
---
#### [new 010] Filling MIDI Velocity using U-Net Image Colorizer
- **分类: cs.SD; eess.AS**

- **简介: 论文提出使用U-Net网络填充MIDI速度参数，解决其缺失导致的音乐表现力不足问题，通过窗口注意力与自定义损失函数优化稀疏图像处理，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.07751v1](http://arxiv.org/pdf/2508.07751v1)**

> **作者:** Zhanhong He; David Cooper; Defeng Huang; Roberto Togneri
>
> **备注:** 12 pages, submitted to CMMR2025 conference
>
> **摘要:** Modern music producers commonly use MIDI (Musical Instrument Digital Interface) to store their musical compositions. However, MIDI files created with digital software may lack the expressive characteristics of human performances, essentially leaving the velocity parameter - a control for note loudness - undefined, which defaults to a flat value. The task of filling MIDI velocity is termed MIDI velocity prediction, which uses regression models to enhance music expressiveness by adjusting only this parameter. In this paper, we introduce the U-Net, a widely adopted architecture in image colorization, to this task. By conceptualizing MIDI data as images, we adopt window attention and develop a custom loss function to address the sparsity of MIDI-converted images. Current dataset availability restricts our experiments to piano data. Evaluated on the MAESTRO v3 and SMD datasets, our proposed method for filling MIDI velocity outperforms previous approaches in both quantitative metrics and qualitative listening tests.
>
---
#### [new 011] Bridging ASR and LLMs for Dysarthric Speech Recognition: Benchmarking Self-Supervised and Generative Approaches
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文针对失语性语音识别任务，评估自监督与生成模型的结合效果，解决传统ASR因音素扭曲和变异性导致的识别难题，通过对比CTC、seq2seq及LLM增强解码策略，验证LLM在音素修复与语法纠正中的优势。**

- **链接: [http://arxiv.org/pdf/2508.08027v1](http://arxiv.org/pdf/2508.08027v1)**

> **作者:** Ahmed Aboeitta; Ahmed Sharshar; Youssef Nafea; Shady Shehata
>
> **摘要:** Speech Recognition (ASR) due to phoneme distortions and high variability. While self-supervised ASR models like Wav2Vec, HuBERT, and Whisper have shown promise, their effectiveness in dysarthric speech remains unclear. This study systematically benchmarks these models with different decoding strategies, including CTC, seq2seq, and LLM-enhanced decoding (BART,GPT-2, Vicuna). Our contributions include (1) benchmarking ASR architectures for dysarthric speech, (2) introducing LLM-based decoding to improve intelligibility, (3) analyzing generalization across datasets, and (4) providing insights into recognition errors across severity levels. Findings highlight that LLM-enhanced decoding improves dysarthric ASR by leveraging linguistic constraints for phoneme restoration and grammatical correction.
>
---
#### [new 012] Inversion of Arctic dual-channel sound speed profile based on random airgun signal
- **分类: cs.SD; cs.NA; math.NA; physics.ao-ph; physics.app-ph**

- **简介: 论文提出基于双通道声速剖面的反演方法，利用折射正常模式特性提取频散结构，结合双参数表示与水平变异性解耦技术，实现北极低频长距声学传播的高效反演，具有参数少、速度快、低成本等优势。**

- **链接: [http://arxiv.org/pdf/2508.07152v1](http://arxiv.org/pdf/2508.07152v1)**

> **作者:** Jinbao Weng; Yubo Qi; Yanming Yang; Hongtao Wen; Hongtao Zhou; Benqing Chen; Dewei Xu; Ruichao Xue; Caigao Zeng
>
> **摘要:** For the unique dual-channel sound speed profiles of the Canadian Basin and the Chukchi Plateau in the Arctic, based on the propagation characteristics of refracted normal modes under dual-channel sound speed profiles, an inversion method using refracted normal modes for dual-channel sound speed profiles is proposed. This method proposes a dual-parameter representation method for dual-channel sound speed profiles, tailored to the characteristics of dual-channel sound speed profiles. A dispersion structure extraction method is proposed for the dispersion structure characteristics of refracted normal modes under dual-channel sound speed profiles. Combining the parameter representation method of sound speed profiles and the dispersion structure extraction method, an inversion method for dual-channel sound speed profiles is proposed. For the common horizontal variation of sound speed profiles in long-distance acoustic propagation, a method for inverting horizontally varying dual-channel sound speed profiles is proposed. Finally, this article verifies the effectiveness of the dual-channel sound speed profile inversion method using the Arctic low-frequency long-range acoustic propagation experiment. Compared with previous sound speed profile inversion methods, the method proposed in this article has the advantages of fewer inversion parameters and faster inversion speed. It can be implemented using only a single hydrophone passively receiving random air gun signals, and it also solves the inversion problem of horizontal variation of sound speed profiles. It has significant advantages such as low cost, easy deployment, and fast computation speed.
>
---
#### [new 013] SCDF: A Speaker Characteristics DeepFake Speech Dataset for Bias Analysis
- **分类: cs.SD; cs.AI; cs.CR**

- **简介: 本研究提出SCDF数据集，用于分析深伪语音中的偏见，涵盖多语言、性别、年龄及合成器的多样化样本，评估检测器性能，揭示性别、语言、年龄及合成器类型的差异，推动公平检测系统的开发。**

- **链接: [http://arxiv.org/pdf/2508.07944v1](http://arxiv.org/pdf/2508.07944v1)**

> **作者:** Vojtěch Staněk; Karel Srna; Anton Firc; Kamil Malinka
>
> **摘要:** Despite growing attention to deepfake speech detection, the aspects of bias and fairness remain underexplored in the speech domain. To address this gap, we introduce the Speaker Characteristics Deepfake (SCDF) dataset: a novel, richly annotated resource enabling systematic evaluation of demographic biases in deepfake speech detection. SCDF contains over 237,000 utterances in a balanced representation of both male and female speakers spanning five languages and a wide age range. We evaluate several state-of-the-art detectors and show that speaker characteristics significantly influence detection performance, revealing disparities across sex, language, age, and synthesizer type. These findings highlight the need for bias-aware development and provide a foundation for building non-discriminatory deepfake detection systems aligned with ethical and regulatory standards.
>
---
#### [new 014] A Small-footprint Acoustic Echo Cancellation Solution for Mobile Full-Duplex Speech Interactions
- **分类: cs.SD; cs.AI**

- **简介: 论文提出一种基于神经网络的AEC解决方案，针对移动全双工语音交互中的硬件差异、非线性畸变和延迟问题，通过数据增强、渐进学习及后处理优化VAD和ASR，实现小模型流式推理，提升语音质量。**

- **链接: [http://arxiv.org/pdf/2508.07561v1](http://arxiv.org/pdf/2508.07561v1)**

> **作者:** Yiheng Jiang; Tian Biao
>
> **备注:** This paper is accepted to ICASSP 2025
>
> **摘要:** In full-duplex speech interaction systems, effective Acoustic Echo Cancellation (AEC) is crucial for recovering echo-contaminated speech. This paper presents a neural network-based AEC solution to address challenges in mobile scenarios with varying hardware, nonlinear distortions and long latency. We first incorporate diverse data augmentation strategies to enhance the model's robustness across various environments. Moreover, progressive learning is employed to incrementally improve AEC effectiveness, resulting in a considerable improvement in speech quality. To further optimize AEC's downstream applications, we introduce a novel post-processing strategy employing tailored parameters designed specifically for tasks such as Voice Activity Detection (VAD) and Automatic Speech Recognition (ASR), thus enhancing their overall efficacy. Finally, our method employs a small-footprint model with streaming inference, enabling seamless deployment on mobile devices. Empirical results demonstrate effectiveness of the proposed method in Echo Return Loss Enhancement and Perceptual Evaluation of Speech Quality, alongside significant improvements in both VAD and ASR results.
>
---
#### [new 015] Exploring Procedural Data Generation for Automatic Acoustic Guitar Fingerpicking Transcription
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出通过生成式数据管道（知识编排、MIDI渲染、物理建模、音频增强）解决吉他指弹转录数据不足问题，训练CRNN模型并验证生成数据的有效性。**

- **链接: [http://arxiv.org/pdf/2508.07987v1](http://arxiv.org/pdf/2508.07987v1)**

> **作者:** Sebastian Murgul; Michael Heizmann
>
> **备注:** Accepted to the 6th Conference on AI Music Creativity (AIMC), 2025
>
> **摘要:** Automatic transcription of acoustic guitar fingerpicking performances remains a challenging task due to the scarcity of labeled training data and legal constraints connected with musical recordings. This work investigates a procedural data generation pipeline as an alternative to real audio recordings for training transcription models. Our approach synthesizes training data through four stages: knowledge-based fingerpicking tablature composition, MIDI performance rendering, physical modeling using an extended Karplus-Strong algorithm, and audio augmentation including reverb and distortion. We train and evaluate a CRNN-based note-tracking model on both real and synthetic datasets, demonstrating that procedural data can be used to achieve reasonable note-tracking results. Finetuning with a small amount of real data further enhances transcription accuracy, improving over models trained exclusively on real recordings. These results highlight the potential of procedurally generated audio for data-scarce music information retrieval tasks.
>
---
#### [new 016] Joint Transcription of Acoustic Guitar Strumming Directions and Chords
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出联合转录吉他弹奏方向与和弦的任务，解决现有数据不足导致的准确性瓶颈，通过结合传感器与合成数据训练CRNN模型，实现高精度转录，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2508.07973v1](http://arxiv.org/pdf/2508.07973v1)**

> **作者:** Sebastian Murgul; Johannes Schimper; Michael Heizmann
>
> **备注:** Accepted to the 26th International Society for Music Information Retrieval Conference (ISMIR), 2025
>
> **摘要:** Automatic transcription of guitar strumming is an underrepresented and challenging task in Music Information Retrieval (MIR), particularly for extracting both strumming directions and chord progressions from audio signals. While existing methods show promise, their effectiveness is often hindered by limited datasets. In this work, we extend a multimodal approach to guitar strumming transcription by introducing a novel dataset and a deep learning-based transcription model. We collect 90 min of real-world guitar recordings using an ESP32 smartwatch motion sensor and a structured recording protocol, complemented by a synthetic dataset of 4h of labeled strumming audio. A Convolutional Recurrent Neural Network (CRNN) model is trained to detect strumming events, classify their direction, and identify the corresponding chords using only microphone audio. Our evaluation demonstrates significant improvements over baseline onset detection algorithms, with a hybrid method combining synthetic and real-world data achieving the highest accuracy for both strumming action detection and chord classification. These results highlight the potential of deep learning for robust guitar strumming transcription and open new avenues for automatic rhythm guitar analysis.
>
---
#### [new 017] FlexCTC: GPU-powered CTC Beam Decoding with advanced Contextual Abilities
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 论文提出基于GPU的FlexCTC解码器，通过CUDA图优化和上下文化技术提升CTC模型的高效性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.07315v1](http://arxiv.org/pdf/2508.07315v1)**

> **作者:** Lilit Grigoryan; Vladimir Bataev; Nikolay Karpov; Andrei Andrusenko; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted to Automatic Speech Recognition and Understanding Workshop (ASRU) 2025
>
> **摘要:** While beam search improves speech recognition quality over greedy decoding, standard implementations are slow, often sequential, and CPU-bound. To fully leverage modern hardware capabilities, we present a novel open-source FlexCTC toolkit for fully GPU-based beam decoding, designed for Connectionist Temporal Classification (CTC) models. Developed entirely in Python and PyTorch, it offers a fast, user-friendly, and extensible alternative to traditional C++, CUDA, or WFST-based decoders. The toolkit features a high-performance, fully batched GPU implementation with eliminated CPU-GPU synchronization and minimized kernel launch overhead via CUDA Graphs. It also supports advanced contextualization techniques, including GPU-powered N-gram language model fusion and phrase-level boosting. These features enable accurate and efficient decoding, making them suitable for both research and production use.
>
---
#### [new 018] FlowSE: Flow Matching-based Speech Enhancement
- **分类: eess.AS; eess.SP**

- **简介: 论文提出基于流匹配的语音增强方法FlowSE，通过降低计算复杂度（NFE）实现高性能，与扩散模型在NFE=5时性能相当，无需额外微调。**

- **链接: [http://arxiv.org/pdf/2508.06840v1](http://arxiv.org/pdf/2508.06840v1)**

> **作者:** Seonggyu Lee; Sein Cheong; Sangwook Han; Jong Won Shin
>
> **备注:** Published in ICASSP 2025
>
> **摘要:** Diffusion probabilistic models have shown impressive performance for speech enhancement, but they typically require 25 to 60 function evaluations in the inference phase, resulting in heavy computational complexity. Recently, a fine-tuning method was proposed to correct the reverse process, which significantly lowered the number of function evaluations (NFE). Flow matching is a method to train continuous normalizing flows which model probability paths from known distributions to unknown distributions including those described by diffusion processes. In this paper, we propose a speech enhancement based on conditional flow matching. The proposed method achieved the performance comparable to those for the diffusion-based speech enhancement with the NFE of 60 when the NFE was 5, and showed similar performance with the diffusion model correcting the reverse process at the same NFE from 1 to 5 without additional fine tuning procedure. We also have shown that the corresponding diffusion model derived from the conditional probability path with a modified optimal transport conditional vector field demonstrated similar performances with the NFE of 5 without any fine-tuning procedure.
>
---
#### [new 019] Speech Enhancement based on cascaded two flow
- **分类: eess.AS; eess.SP**

- **简介: 论文提出基于流匹配的双流架构，通过统一模型实现语音增强与生成初始值，降低函数评估次数（NFE）并保持性能。**

- **链接: [http://arxiv.org/pdf/2508.06842v1](http://arxiv.org/pdf/2508.06842v1)**

> **作者:** Seonggyu Lee; Sein Cheong; Sangwook Han; Kihyuk Kim; Jong Won Shi
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Speech enhancement (SE) based on diffusion probabilistic models has exhibited impressive performance, while requiring a relatively high number of function evaluations (NFE). Recently, SE based on flow matching has been proposed, which showed competitive performance with a small NFE. Early approaches adopted the noisy speech as the only conditioning variable. There have been other approaches which utilize speech enhanced with a predictive model as another conditioning variable and to sample an initial value, but they require a separate predictive model on top of the generative SE model. In this work, we propose to employ an identical model based on flow matching for both SE and generating enhanced speech used as an initial starting point and a conditioning variable. Experimental results showed that the proposed method required the same or fewer NFEs even with two cascaded generative methods while achieving equivalent or better performances to the previous baselines.
>
---
#### [new 020] ParaNoise-SV: Integrated Approach for Noise-Robust Speaker Verification with Parallel Joint Learning of Speech Enhancement and Noise Extraction
- **分类: eess.AS; cs.SD; I.2.7; H.5.5; I.5.4**

- **简介: 论文提出ParaNoise-SV，通过双U-Net联合学习语音增强与噪声提取，利用并行连接实现噪声与说话人特征分离，提升噪声环境下说话人验证鲁棒性，较传统方法降低8.4% EER。**

- **链接: [http://arxiv.org/pdf/2508.07219v1](http://arxiv.org/pdf/2508.07219v1)**

> **作者:** Minu Kim; Kangwook Jang; Hoirin Kim
>
> **备注:** 5 pages, 3 figures, accepted to Interspeech 2025
>
> **摘要:** Noise-robust speaker verification leverages joint learning of speech enhancement (SE) and speaker verification (SV) to improve robustness. However, prevailing approaches rely on implicit noise suppression, which struggles to separate noise from speaker characteristics as they do not explicitly distinguish noise from speech during training. Although integrating SE and SV helps, it remains limited in handling noise effectively. Meanwhile, recent SE studies suggest that explicitly modeling noise, rather than merely suppressing it, enhances noise resilience. Reflecting this, we propose ParaNoise-SV, with dual U-Nets combining a noise extraction (NE) network and a speech enhancement (SE) network. The NE U-Net explicitly models noise, while the SE U-Net refines speech with guidance from NE through parallel connections, preserving speaker-relevant features. Experimental results show that ParaNoise-SV achieves a relatively 8.4% lower equal error rate (EER) than previous joint SE-SV models.
>
---
#### [new 021] Auditory Intelligence: Understanding the World Through Sound
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 论文聚焦听觉智能，解决表面识别不足问题，提出四类范式（ASPIRE等）实现层次化、情境化的理解，提升解释性与通用性。**

- **链接: [http://arxiv.org/pdf/2508.07829v1](http://arxiv.org/pdf/2508.07829v1)**

> **作者:** Hyeonuk Nam
>
> **备注:** Position paper without experimental/quantitative validation. Not submitted to any journal/conference
>
> **摘要:** Recent progress in auditory intelligence has yielded high-performing systems for sound event detection (SED), acoustic scene classification (ASC), automated audio captioning (AAC), and audio question answering (AQA). Yet these tasks remain largely constrained to surface-level recognition-capturing what happened but not why, what it implies, or how it unfolds in context. I propose a conceptual reframing of auditory intelligence as a layered, situated process that encompasses perception, reasoning, and interaction. To instantiate this view, I introduce four cognitively inspired task paradigms-ASPIRE, SODA, AUX, and AUGMENT-those structure auditory understanding across time-frequency pattern captioning, hierarchical event/scene description, causal explanation, and goal-driven interpretation, respectively. Together, these paradigms provide a roadmap toward more generalizable, explainable, and human-aligned auditory intelligence, and are intended to catalyze a broader discussion of what it means for machines to understand sound.
>
---
#### [new 022] Score-Informed BiLSTM Correction for Refining MIDI Velocity in Automatic Piano Transcription
- **分类: eess.AS; cs.SD**

- **简介: 论文提出基于乐谱信息的BiLSTM模型，用于自动钢琴转写中MIDI速度修正，解决AMT估计不足问题，虽未达最优但有效。**

- **链接: [http://arxiv.org/pdf/2508.07757v1](http://arxiv.org/pdf/2508.07757v1)**

> **作者:** Zhanhong He; Roberto Togneri; Defeng; Huang
>
> **备注:** 4 pages; rejected paper by WASPAA2025
>
> **摘要:** MIDI is a modern standard for storing music, recording how musical notes are played. Many piano performances have corresponding MIDI scores available online. Some of these are created by the original performer, recording on an electric piano alongside the audio, while others are through manual transcription. In recent years, automatic music transcription (AMT) has rapidly advanced, enabling machines to transcribe MIDI from audio. However, these transcriptions often require further correction. Assuming a perfect timing correction, we focus on the loudness correction in terms of MIDI velocity (a parameter in MIDI for loudness control). This task can be approached through score-informed MIDI velocity estimation, which has undergone several developments. While previous approaches introduced specifically built models to re-estimate MIDI velocity, thereby replacing AMT estimates, we propose a BiLSTM correction module to refine AMT-estimated velocity. Although we did not reach state-of-the-art performance, we validated our method on the well-known AMT system, the high-resolution piano transcription (HPT), and achieved significant improvements.
>
---
#### [new 023] VGGSounder: Audio-Visual Evaluations for Foundation Models
- **分类: cs.MM; cs.AI; cs.SD**

- **简介: 论文提出VGGSounder，针对现有音频-视觉评估数据集的不足，构建多标签测试集以精准评估基础模型的多模态理解能力，并引入模态混淆指标分析模型局限性。**

- **链接: [http://arxiv.org/pdf/2508.08237v1](http://arxiv.org/pdf/2508.08237v1)**

> **作者:** Daniil Zverev; Thaddäus Wiedemer; Ameya Prabhu; Matthias Bethge; Wieland Brendel; A. Sophia Koepke
>
> **备注:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** The emergence of audio-visual foundation models underscores the importance of reliably assessing their multi-modal understanding. The VGGSounder dataset is commonly used as a benchmark for evaluation audio-visual classification. However, our analysis identifies several limitations of VGGSounder, including incomplete labelling, partially overlapping classes, and misaligned modalities. These lead to distorted evaluations of auditory and visual capabilities. To address these limitations, we introduce VGGSounder, a comprehensively re-annotated, multi-label test set that extends VGGSound and is specifically designed to evaluate audio-visual foundation models. VGGSounder features detailed modality annotations, enabling precise analyses of modality-specific performance. Furthermore, we reveal model limitations by analysing performance degradation when adding another input modality with our new modality confusion metric.
>
---
#### [new 024] SLRTP2025 Sign Language Production Challenge: Methodology, Results, and Future Work
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 论文提出首个Sign Language Production挑战，解决标准化评估问题，通过T2P翻译任务评估口语到手语生成方法，使用RWTH数据集并发布标准化评估网络。**

- **链接: [http://arxiv.org/pdf/2508.06951v1](http://arxiv.org/pdf/2508.06951v1)**

> **作者:** Harry Walsh; Ed Fish; Ozge Mercanoglu Sincan; Mohamed Ilyes Lakhal; Richard Bowden; Neil Fox; Bencie Woll; Kepeng Wu; Zecheng Li; Weichao Zhao; Haodong Wang; Wengang Zhou; Houqiang Li; Shengeng Tang; Jiayi He; Xu Wang; Ruobei Zhang; Yaxiong Wang; Lechao Cheng; Meryem Tasyurek; Tugce Kiziltepe; Hacer Yalim Keles
>
> **备注:** 11 pages, 6 Figures, CVPR conference
>
> **摘要:** Sign Language Production (SLP) is the task of generating sign language video from spoken language inputs. The field has seen a range of innovations over the last few years, with the introduction of deep learning-based approaches providing significant improvements in the realism and naturalness of generated outputs. However, the lack of standardized evaluation metrics for SLP approaches hampers meaningful comparisons across different systems. To address this, we introduce the first Sign Language Production Challenge, held as part of the third SLRTP Workshop at CVPR 2025. The competition's aims are to evaluate architectures that translate from spoken language sentences to a sequence of skeleton poses, known as Text-to-Pose (T2P) translation, over a range of metrics. For our evaluation data, we use the RWTH-PHOENIX-Weather-2014T dataset, a German Sign Language - Deutsche Gebardensprache (DGS) weather broadcast dataset. In addition, we curate a custom hidden test set from a similar domain of discourse. This paper presents the challenge design and the winning methodologies. The challenge attracted 33 participants who submitted 231 solutions, with the top-performing team achieving BLEU-1 scores of 31.40 and DTW-MJE of 0.0574. The winning approach utilized a retrieval-based framework and a pre-trained language model. As part of the workshop, we release a standardized evaluation network, including high-quality skeleton extraction-based keypoints establishing a consistent baseline for the SLP field, which will enable future researchers to compare their work against a broader range of methods.
>
---
#### [new 025] MSU-Bench: Towards Understanding the Conversational Multi-talker Scenarios
- **分类: eess.AS; cs.SD**

- **简介: 论文提出MSU-Bench，针对多说话人对话理解任务，构建以说话人为中心的综合基准，包含四层级任务（静态/动态属性、背景/互动理解），评估模型性能并揭示开放源与闭源模型能力差距，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.08155v1](http://arxiv.org/pdf/2508.08155v1)**

> **作者:** Shuai Wang; Zhaokai Sun; Zhennan Lin; Chengyou Wang; Zhou Pan; Lei Xie
>
> **摘要:** Spoken Language Understanding (SLU) has progressed from traditional single-task methods to large audio language model (LALM) solutions. Yet, most existing speech benchmarks focus on single-speaker or isolated tasks, overlooking the challenges posed by multi-speaker conversations that are common in real-world scenarios. We introduce MSU-Bench, a comprehensive benchmark for evaluating multi-speaker conversational understanding with a speaker-centric design. Our hierarchical framework covers four progressive tiers: single-speaker static attribute understanding, single-speaker dynamic attribute understanding, multi-speaker background understanding, and multi-speaker interaction understanding. This structure ensures all tasks are grounded in speaker-centric contexts, from basic perception to complex reasoning across multiple speakers. By evaluating state-of-the-art models on MSU-Bench, we demonstrate that as task complexity increases across the benchmark's tiers, all models exhibit a significant performance decline. We also observe a persistent capability gap between open-source models and closed-source commercial ones, particularly in multi-speaker interaction reasoning. These findings validate the effectiveness of MSU-Bench for assessing and advancing conversational understanding in realistic multi-speaker environments. Demos can be found in the supplementary material.
>
---
#### [new 026] Real-time CARFAC Cochlea Model Acceleration on FPGA for Underwater Acoustic Sensing Systems
- **分类: eess.AS; cs.SD; 92C50 (Primary) 68Q25, 94A12 (Secondary)**

- **简介: 论文提出在FPGA上加速CARFAC cochlea模型，构建实时水下声学传感系统，解决传统方法资源占用高、速度慢的问题，实现低功耗高效处理。**

- **链接: [http://arxiv.org/pdf/2508.07523v1](http://arxiv.org/pdf/2508.07523v1)**

> **作者:** Bram Bremer; Matthew Bigelow; Stuart Anstee; Gregory Cohen; Andre van Schaik; Ying Xu
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** This paper presents a real-time, energy-efficient embedded system implementing an array of Cascade of Asymmetric Resonators with Fast-Acting Compression (CARFAC) cochlea models for underwater sound analysis. Built on the AMD Kria KV260 System-on-Module (SoM), the system integrates a Rust-based software framework on the processor for real-time interfacing and synchronization with multiple hydrophone inputs, and a hardware-accelerated implementation of the CARFAC models on a Field-Programmable Gate Array (FPGA) for real-time sound pre-processing. Compared to prior work, the CARFAC accelerator achieves improved scalability and processing speed while reducing resource usage through optimized time-multiplexing, pipelined design, and elimination of costly division circuits. Experimental results demonstrate 13.5% hardware utilization for a single 64-channel CARFAC instance and a whole board power consumption of 3.11 W when processing a 256 kHz input signal in real time.
>
---
#### [new 027] How Does a Deep Neural Network Look at Lexical Stress?
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 论文研究深度神经网络在词汇重音识别中的决策机制，通过构建英文双词数据集和CNN模型，揭示其依赖重音与轻音的频谱特性及单词内信息，提出特征相关性分析，表明最佳分类器受重音前两共振峰及基频影响，展现深度学习从自然语音数据中提取分布式重音线索的能力。**

- **链接: [http://arxiv.org/pdf/2508.07229v1](http://arxiv.org/pdf/2508.07229v1)**

> **作者:** Itai Allouche; Itay Asael; Rotem Rousso; Vered Dassa; Ann Bradlow; Seung-Eun Kim; Matthew Goldrick; Joseph Keshet
>
> **备注:** 10 pages, 4 figures, submitted to the Journal of the Acoustical Society of America (JASA)
>
> **摘要:** Despite their success in speech processing, neural networks often operate as black boxes, prompting the question: what informs their decisions, and how can we interpret them? This work examines this issue in the context of lexical stress. A dataset of English disyllabic words was automatically constructed from read and spontaneous speech. Several Convolutional Neural Network (CNN) architectures were trained to predict stress position from a spectrographic representation of disyllabic words lacking minimal stress pairs (e.g., initial stress WAllet, final stress exTEND), achieving up to 92% accuracy on held-out test data. Layerwise Relevance Propagation (LRP), a technique for CNN interpretability analysis, revealed that predictions for held-out minimal pairs (PROtest vs. proTEST ) were most strongly influenced by information in stressed versus unstressed syllables, particularly the spectral properties of stressed vowels. However, the classifiers also attended to information throughout the word. A feature-specific relevance analysis is proposed, and its results suggest that our best-performing classifier is strongly influenced by the stressed vowel's first and second formants, with some evidence that its pitch and third formant also contribute. These results reveal deep learning's ability to acquire distributed cues to stress from naturally occurring data, extending traditional phonetic work based around highly controlled stimuli.
>
---
#### [new 028] MMFformer: Multimodal Fusion Transformer Network for Depression Detection
- **分类: cs.CV; cs.AI; cs.LG; cs.SD**

- **简介: 论文提出MMFformer多模态Transformer网络，融合社交媒体时空特征，提升抑郁症检测准确率。**

- **链接: [http://arxiv.org/pdf/2508.06701v1](http://arxiv.org/pdf/2508.06701v1)**

> **作者:** Md Rezwanul Haque; Md. Milon Islam; S M Taslim Uddin Raju; Hamdi Altaheri; Lobna Nassar; Fakhri Karray
>
> **备注:** Accepted for the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Vienna, Austria
>
> **摘要:** Depression is a serious mental health illness that significantly affects an individual's well-being and quality of life, making early detection crucial for adequate care and treatment. Detecting depression is often difficult, as it is based primarily on subjective evaluations during clinical interviews. Hence, the early diagnosis of depression, thanks to the content of social networks, has become a prominent research area. The extensive and diverse nature of user-generated information poses a significant challenge, limiting the accurate extraction of relevant temporal information and the effective fusion of data across multiple modalities. This paper introduces MMFformer, a multimodal depression detection network designed to retrieve depressive spatio-temporal high-level patterns from multimodal social media information. The transformer network with residual connections captures spatial features from videos, and a transformer encoder is exploited to design important temporal dynamics in audio. Moreover, the fusion architecture fused the extracted features through late and intermediate fusion strategies to find out the most relevant intermodal correlations among them. Finally, the proposed network is assessed on two large-scale depression detection datasets, and the results clearly reveal that it surpasses existing state-of-the-art approaches, improving the F1-Score by 13.92% for D-Vlog dataset and 7.74% for LMVD dataset. The code is made available publicly at https://github.com/rezwanh001/Large-Scale-Multimodal-Depression-Detection.
>
---
#### [new 029] AD-AVSR: Asymmetric Dual-stream Enhancement for Robust Audio-Visual Speech Recognition
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 论文提出AD-AVSR框架，解决AVSR中异构关联不足问题，通过双向模态增强、双流音频编码及跨模态噪声抑制模块提升鲁棒性，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.07608v1](http://arxiv.org/pdf/2508.07608v1)**

> **作者:** Junxiao Xue; Xiaozhen Liu; Xuecheng Wu; Xinyi Yin; Danlei Huang; Fei Yu
>
> **备注:** Accepted by the ACM MM 2025 Workshop on SVC
>
> **摘要:** Audio-visual speech recognition (AVSR) combines audio-visual modalities to improve speech recognition, especially in noisy environments. However, most existing methods deploy the unidirectional enhancement or symmetric fusion manner, which limits their capability to capture heterogeneous and complementary correlations of audio-visual data-especially under asymmetric information conditions. To tackle these gaps, we introduce a new AVSR framework termed AD-AVSR based on bidirectional modality enhancement. Specifically, we first introduce the audio dual-stream encoding strategy to enrich audio representations from multiple perspectives and intentionally establish asymmetry to support subsequent cross-modal interactions. The enhancement process involves two key components, Audio-aware Visual Refinement Module for enhanced visual representations under audio guidance, and Cross-modal Noise Suppression Masking Module which refines audio representations using visual cues, collaboratively leading to the closed-loop and bidirectional information flow. To further enhance correlation robustness, we adopt a threshold-based selection mechanism to filter out irrelevant or weakly correlated audio-visual pairs. Extensive experimental results on the LRS2 and LRS3 datasets indicate that our AD-AVSR consistently surpasses SOTA methods in both performance and noise robustness, highlighting the effectiveness of our model design.
>
---
#### [new 030] TurboBias: Universal ASR Context-Biasing powered by GPU-accelerated Phrase-Boosting Tree
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 论文提出一种通用ASR上下文偏置框架，通过GPU加速的词级提升树解决训练依赖、解码速度慢等问题，支持CTC、Transducers和Attention模型，实现高效多任务解码。**

- **链接: [http://arxiv.org/pdf/2508.07014v1](http://arxiv.org/pdf/2508.07014v1)**

> **作者:** Andrei Andrusenko; Vladimir Bataev; Lilit Grigoryan; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Recognizing specific key phrases is an essential task for contextualized Automatic Speech Recognition (ASR). However, most existing context-biasing approaches have limitations associated with the necessity of additional model training, significantly slow down the decoding process, or constrain the choice of the ASR system type. This paper proposes a universal ASR context-biasing framework that supports all major types: CTC, Transducers, and Attention Encoder-Decoder models. The framework is based on a GPU-accelerated word boosting tree, which enables it to be used in shallow fusion mode for greedy and beam search decoding without noticeable speed degradation, even with a vast number of key phrases (up to 20K items). The obtained results showed high efficiency of the proposed method, surpassing the considered open-source context-biasing approaches in accuracy and decoding speed. Our context-biasing framework is open-sourced as a part of the NeMo toolkit.
>
---
#### [new 031] Think Before You Talk: Enhancing Meaningful Dialogue Generation in Full-Duplex Speech Language Models with Planning-Inspired Text Guidance
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文提出TurnGuide方法，通过规划启发式动态分割语音为对话回合并生成文本指导，解决FD-SLMs在长时间语音和有限数据下的对话能力下降问题，提升对话意义与流畅性。**

- **链接: [http://arxiv.org/pdf/2508.07375v1](http://arxiv.org/pdf/2508.07375v1)**

> **作者:** Wenqian Cui; Lei Zhu; Xiaohui Li; Zhihan Guo; Haoli Bai; Lu Hou; Irwin King
>
> **备注:** Work in progress
>
> **摘要:** Full-Duplex Speech Language Models (FD-SLMs) are specialized foundation models designed to enable natural, real-time spoken interactions by modeling complex conversational dynamics such as interruptions, backchannels, and overlapping speech, and End-to-end (e2e) FD-SLMs leverage real-world double-channel conversational data to capture nuanced two-speaker dialogue patterns for human-like interactions. However, they face a critical challenge -- their conversational abilities often degrade compared to pure-text conversation due to prolonged speech sequences and limited high-quality spoken dialogue data. While text-guided speech generation could mitigate these issues, it suffers from timing and length issues when integrating textual guidance into double-channel audio streams, disrupting the precise time alignment essential for natural interactions. To address these challenges, we propose TurnGuide, a novel planning-inspired approach that mimics human conversational planning by dynamically segmenting assistant speech into dialogue turns and generating turn-level text guidance before speech output, which effectively resolves both insertion timing and length challenges. Extensive experiments demonstrate our approach significantly improves e2e FD-SLMs' conversational abilities, enabling them to generate semantically meaningful and coherent speech while maintaining natural conversational flow. Demos are available at https://dreamtheater123.github.io/TurnGuide-Demo/. Code will be available at https://github.com/dreamtheater123/TurnGuide.
>
---
#### [new 032] Text to Speech System for Meitei Mayek Script
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 论文提出基于Tacotron 2和HiFi-GAN的文本转语音系统，针对Meitei Mayek书写系统，解决低资源环境下语音合成问题，通过音素映射与单人数据集实现自然可懂的合成语音，推动曼尼普尔语语言保存与技术包容。**

- **链接: [http://arxiv.org/pdf/2508.06870v1](http://arxiv.org/pdf/2508.06870v1)**

> **作者:** Gangular Singh Irengbam; Nirvash Singh Wahengbam; Lanthoiba Meitei Khumanthem; Paikhomba Oinam
>
> **摘要:** This paper presents the development of a Text-to-Speech (TTS) system for the Manipuri language using the Meitei Mayek script. Leveraging Tacotron 2 and HiFi-GAN, we introduce a neural TTS architecture adapted to support tonal phonology and under-resourced linguistic environments. We develop a phoneme mapping for Meitei Mayek to ARPAbet, curate a single-speaker dataset, and demonstrate intelligible and natural speech synthesis, validated through subjective and objective metrics. This system lays the groundwork for linguistic preservation and technological inclusion of Manipuri.
>
---
#### [new 033] Voice Pathology Detection Using Phonation
- **分类: cs.CV; cs.SD**

- **简介: 论文提出基于语音振荡数据的非侵入式语音病理学检测框架，利用MFCC、Mel谱等特征与LSTM-RNN模型实现病理分类，通过数据增强和尺度特征提升诊断准确性，支持AI驱动的早期诊疗。**

- **链接: [http://arxiv.org/pdf/2508.07587v1](http://arxiv.org/pdf/2508.07587v1)**

> **作者:** Sri Raksha Siva; Nived Suthahar; Prakash Boominathan; Uma Ranjan
>
> **备注:** 17 Pages, 11 Figures
>
> **摘要:** Voice disorders significantly affect communication and quality of life, requiring an early and accurate diagnosis. Traditional methods like laryngoscopy are invasive, subjective, and often inaccessible. This research proposes a noninvasive, machine learning-based framework for detecting voice pathologies using phonation data. Phonation data from the Saarbr\"ucken Voice Database are analyzed using acoustic features such as Mel Frequency Cepstral Coefficients (MFCCs), chroma features, and Mel spectrograms. Recurrent Neural Networks (RNNs), including LSTM and attention mechanisms, classify samples into normal and pathological categories. Data augmentation techniques, including pitch shifting and Gaussian noise addition, enhance model generalizability, while preprocessing ensures signal quality. Scale-based features, such as H\"older and Hurst exponents, further capture signal irregularities and long-term dependencies. The proposed framework offers a noninvasive, automated diagnostic tool for early detection of voice pathologies, supporting AI-driven healthcare, and improving patient outcomes.
>
---
#### [new 034] Pindrop it! Audio and Visual Deepfake Countermeasures for Robust Detection and Fine Grained-Localization
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 论文提出针对深度伪造视频的音频和视觉对抗措施，解决局部修改带来的检测难题，实现精准定位。**

- **链接: [http://arxiv.org/pdf/2508.08141v1](http://arxiv.org/pdf/2508.08141v1)**

> **作者:** Nicholas Klein; Hemlata Tak; James Fullwood; Krishna Regmi; Leonidas Spinoulas; Ganesh Sivaraman; Tianxiang Chen; Elie Khoury
>
> **摘要:** The field of visual and audio generation is burgeoning with new state-of-the-art methods. This rapid proliferation of new techniques underscores the need for robust solutions for detecting synthetic content in videos. In particular, when fine-grained alterations via localized manipulations are performed in visual, audio, or both domains, these subtle modifications add challenges to the detection algorithms. This paper presents solutions for the problems of deepfake video classification and localization. The methods were submitted to the ACM 1M Deepfakes Detection Challenge, achieving the best performance in the temporal localization task and a top four ranking in the classification task for the TestA split of the evaluation dataset.
>
---
## 更新

#### [replaced 001] DMF2Mel: A Dynamic Multiscale Fusion Network for EEG-Driven Mel Spectrogram Reconstruction
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07526v3](http://arxiv.org/pdf/2507.07526v3)**

> **作者:** Cunhang Fan; Sheng Zhang; Jingjing Zhang; Enrui Liu; Xinhui Li; Gangming Zhao; Zhao Lv
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Decoding speech from brain signals is a challenging research problem. Although existing technologies have made progress in reconstructing the mel spectrograms of auditory stimuli at the word or letter level, there remain core challenges in the precise reconstruction of minute-level continuous imagined speech: traditional models struggle to balance the efficiency of temporal dependency modeling and information retention in long-sequence decoding. To address this issue, this paper proposes the Dynamic Multiscale Fusion Network (DMF2Mel), which consists of four core components: the Dynamic Contrastive Feature Aggregation Module (DC-FAM), the Hierarchical Attention-Guided Multi-Scale Network (HAMS-Net), the SplineMap attention mechanism, and the bidirectional state space module (convMamba). Specifically, the DC-FAM separates speech-related "foreground features" from noisy "background features" through local convolution and global attention mechanisms, effectively suppressing interference and enhancing the representation of transient signals. HAMS-Net, based on the U-Net framework,achieves cross-scale fusion of high-level semantics and low-level details. The SplineMap attention mechanism integrates the Adaptive Gated Kolmogorov-Arnold Network (AGKAN) to combine global context modeling with spline-based local fitting. The convMamba captures long-range temporal dependencies with linear complexity and enhances nonlinear dynamic modeling capabilities. Results on the SparrKULee dataset show that DMF2Mel achieves a Pearson correlation coefficient of 0.074 in mel spectrogram reconstruction for known subjects (a 48% improvement over the baseline) and 0.048 for unknown subjects (a 35% improvement over the baseline).Code is available at: https://github.com/fchest/DMF2Mel.
>
---
#### [replaced 002] Enhancing Target Speaker Extraction with Explicit Speaker Consistency Modeling
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09510v3](http://arxiv.org/pdf/2507.09510v3)**

> **作者:** Shu Wu; Anbin Qi; Yanzhang Xie; Xiang Xie
>
> **备注:** preprint
>
> **摘要:** Target Speaker Extraction (TSE) uses a reference cue to extract the target speech from a mixture. In TSE systems relying on audio cues, the speaker embedding from the enrolled speech is crucial to performance. However, these embeddings may suffer from speaker identity confusion. Unlike previous studies that focus on improving speaker embedding extraction, we improve TSE performance from the perspective of speaker consistency. In this paper, we propose a speaker consistency-aware target speaker extraction method that incorporates a centroid-based speaker consistency loss. This approach enhances TSE performance by ensuring speaker consistency between the enrolled and extracted speech. In addition, we integrate conditional loss suppression into the training process. The experimental results validate the effectiveness of our proposed methods in advancing the TSE performance. A speech demo is available online:https://sc-tse.netlify.app/
>
---
#### [replaced 003] Learning Perceptually Relevant Temporal Envelope Morphing
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.01588v3](http://arxiv.org/pdf/2506.01588v3)**

> **作者:** Satvik Dixit; Sungjoon Park; Chris Donahue; Laurie M. Heller
>
> **备注:** Accepted at WASPAA 2025
>
> **摘要:** Temporal envelope morphing, the process of interpolating between the amplitude dynamics of two audio signals, is an emerging problem in generative audio systems that lacks sufficient perceptual grounding. Morphing of temporal envelopes in a perceptually intuitive manner should enable new methods for sound blending in creative media and for probing perceptual organization in psychoacoustics. However, existing audio morphing techniques often fail to produce intermediate temporal envelopes when input sounds have distinct temporal structures; many morphers effectively overlay both temporal structures, leading to perceptually unnatural results. In this paper, we introduce a novel workflow for learning envelope morphing with perceptual guidance: we first derive perceptually grounded morphing principles through human listening studies, then synthesize large-scale datasets encoding these principles, and finally train machine learning models to create perceptually intermediate morphs. Specifically, we present: (1) perceptual principles that guide envelope morphing, derived from our listening studies, (2) a supervised framework to learn these principles, (3) an autoencoder that learns to compress temporal envelope structures into latent representations, and (4) benchmarks for evaluating audio envelope morphs, using both synthetic and naturalistic data, and show that our approach outperforms existing methods in producing temporally intermediate morphs. All code, models, and checkpoints are available at https://github.com/TemporalMorphing/EnvelopeMorphing.
>
---
#### [replaced 004] Live Music Models
- **分类: cs.SD; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04651v2](http://arxiv.org/pdf/2508.04651v2)**

> **作者:** Lyria Team; Antoine Caillon; Brian McWilliams; Cassie Tarakajian; Ian Simon; Ilaria Manco; Jesse Engel; Noah Constant; Yunpeng Li; Timo I. Denk; Alberto Lalama; Andrea Agostinelli; Cheng-Zhi Anna Huang; Ethan Manilow; George Brower; Hakan Erdogan; Heidi Lei; Itai Rolnick; Ivan Grishchenko; Manu Orsini; Matej Kastelic; Mauricio Zuluaga; Mauro Verzetti; Michael Dooley; Ondrej Skopek; Rafael Ferrer; Zalán Borsos; Äaron van den Oord; Douglas Eck; Eli Collins; Jason Baldridge; Tom Hume; Chris Donahue; Kehang Han; Adam Roberts
>
> **摘要:** We introduce a new class of generative models for music called live music models that produce a continuous stream of music in real-time with synchronized user control. We release Magenta RealTime, an open-weights live music model that can be steered using text or audio prompts to control acoustic style. On automatic metrics of music quality, Magenta RealTime outperforms other open-weights music generation models, despite using fewer parameters and offering first-of-its-kind live generation capabilities. We also release Lyria RealTime, an API-based model with extended controls, offering access to our most powerful model with wide prompt coverage. These models demonstrate a new paradigm for AI-assisted music creation that emphasizes human-in-the-loop interaction for live music performance.
>
---
#### [replaced 005] Exploring Adapter Design Tradeoffs for Low Resource Music Generation
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21298v2](http://arxiv.org/pdf/2506.21298v2)**

> **作者:** Atharva Mehta; Shivam Chauhan; Monojit Choudhury
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Fine-tuning large-scale music generation models, such as MusicGen and Mustango, is a computationally expensive process, often requiring updates to billions of parameters and, therefore, significant hardware resources. Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly adapter-based methods, have emerged as a promising alternative, enabling adaptation with minimal trainable parameters while preserving model performance. However, the design choices for adapters, including their architecture, placement, and size, are numerous, and it is unclear which of these combinations would produce optimal adapters and why, for a given case of low-resource music genre. In this paper, we attempt to answer this question by studying various adapter configurations for two AI music models, MusicGen and Mustango, on two genres: Hindustani Classical and Turkish Makam music. Our findings reveal distinct trade-offs: convolution-based adapters excel in capturing fine-grained local musical details such as ornamentations and short melodic phrases, while transformer-based adapters better preserve long-range dependencies crucial for structured improvisation. Additionally, we analyze computational resource requirements across different adapter scales, demonstrating how mid-sized adapters (40M parameters) achieve an optimal balance between expressivity and quality. Furthermore, we find that Mustango, a diffusion-based model, generates more diverse outputs with better adherence to the description in the input prompt while lacking in providing stability in notes, rhythm alignment, and aesthetics. Also, it is computationally intensive and requires significantly more time to train. In contrast, autoregressive models like MusicGen offer faster training and are more efficient, and can produce better quality output in comparison, but have slightly higher redundancy in their generations.
>
---
#### [replaced 006] Direction Estimation of Sound Sources Using Microphone Arrays and Signal Strength
- **分类: cs.SD; cs.SY; eess.AS; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.03466v2](http://arxiv.org/pdf/2507.03466v2)**

> **作者:** Mahdi Ali Pour; Zahra Habibzadeh
>
> **备注:** 5 pages
>
> **摘要:** Sound-tracking refers to the process of determining the direction from which a sound originates, making it a fundamental component of sound source localization. This capability is essential in a variety of applications, including security systems, acoustic monitoring, and speaker tracking, where accurately identifying the direction of a sound source enables real-time responses, efficient resource allocation, and improved situational awareness. While sound-tracking is closely related to localization, it specifically focuses on identifying the direction of the sound source rather than estimating its exact position in space. Despite its utility, sound-tracking systems face several challenges, such as maintaining directional accuracy and precision, along with the need for sophisticated hardware configurations and complex signal processing algorithms. This paper presents a sound-tracking method using three electret microphones. We estimate the direction of a sound source using a lightweight method that analyzes signals from three strategically placed microphones. By comparing the average power of the received signals, the system infers the most probable direction of the sound. The results indicate that the power level from each microphone effectively determines the sound source direction. Our system employs a straightforward and cost-effective hardware design, ensuring simplicity and affordability in implementation. It achieves a localization error of less than 6 degrees and a precision of 98%. Additionally, its effortless integration with various systems makes it versatile and adaptable. Consequently, this technique presents a robust and reliable solution for sound-tracking and localization, with potential applications spanning diverse domains such as security systems, smart homes, and acoustic monitoring.
>
---
#### [replaced 007] CLAIR-A: Leveraging Large Language Models to Judge Audio Captions
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.12962v2](http://arxiv.org/pdf/2409.12962v2)**

> **作者:** Tsung-Han Wu; Joseph E. Gonzalez; Trevor Darrell; David M. Chan
>
> **备注:** Accepted to ASRU 2025; Code is publicly available at https://github.com/DavidMChan/clair-a
>
> **摘要:** The Automated Audio Captioning (AAC) task asks models to generate natural language descriptions of an audio input. Evaluating these machine-generated audio captions is a complex task that requires considering diverse factors, among them, auditory scene understanding, sound-object inference, temporal coherence, and the environmental context of the scene. While current methods focus on specific aspects, they often fail to provide an overall score that aligns well with human judgment. In this work, we propose CLAIR-A, a simple and flexible method that leverages the zero-shot capabilities of large language models (LLMs) to evaluate candidate audio captions by directly asking LLMs for a semantic distance score. In our evaluations, CLAIR-A better predicts human judgements of quality compared to traditional metrics, with a 5.8% relative accuracy improvement compared to the domain-specific FENSE metric and up to 11% over the best general-purpose measure on the Clotho-Eval dataset. Moreover, CLAIR-A offers more transparency by allowing the language model to explain the reasoning behind its scores, with these explanations rated up to 30% better by human evaluators than those provided by baseline methods. CLAIR-A is made publicly available at https://github.com/DavidMChan/clair-a.
>
---
#### [replaced 008] DanceChat: Large Language Model-Guided Music-to-Dance Generation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10574v2](http://arxiv.org/pdf/2506.10574v2)**

> **作者:** Qing Wang; Xiaohang Yang; Yilan Dong; Naveen Raj Govindaraj; Gregory Slabaugh; Shanxin Yuan
>
> **摘要:** Music-to-dance generation aims to synthesize human dance motion conditioned on musical input. Despite recent progress, significant challenges remain due to the semantic gap between music and dance motion, as music offers only abstract cues, such as melody, groove, and emotion, without explicitly specifying the physical movements. Moreover, a single piece of music can produce multiple plausible dance interpretations. This one-to-many mapping demands additional guidance, as music alone provides limited information for generating diverse dance movements. The challenge is further amplified by the scarcity of paired music and dance data, which restricts the model\^a\u{A}\'Zs ability to learn diverse dance patterns. In this paper, we introduce DanceChat, a Large Language Model (LLM)-guided music-to-dance generation approach. We use an LLM as a choreographer that provides textual motion instructions, offering explicit, high-level guidance for dance generation. This approach goes beyond implicit learning from music alone, enabling the model to generate dance that is both more diverse and better aligned with musical styles. Our approach consists of three components: (1) an LLM-based pseudo instruction generation module that produces textual dance guidance based on music style and structure, (2) a multi-modal feature extraction and fusion module that integrates music, rhythm, and textual guidance into a shared representation, and (3) a diffusion-based motion synthesis module together with a multi-modal alignment loss, which ensures that the generated dance is aligned with both musical and textual cues. Extensive experiments on AIST++ and human evaluations show that DanceChat outperforms state-of-the-art methods both qualitatively and quantitatively.
>
---
#### [replaced 009] Enhancing Lung Disease Diagnosis via Semi-Supervised Machine Learning
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.16845v2](http://arxiv.org/pdf/2507.16845v2)**

> **作者:** Xiaoran Xu; In-Ho Ra; Ravi Sankar
>
> **摘要:** Lung diseases, including lung cancer and COPD, are significant health concerns globally. Traditional diagnostic methods can be costly, time-consuming, and invasive. This study investigates the use of semi supervised learning methods for lung sound signal detection using a model combination of MFCC+CNN. By introducing semi supervised learning modules such as Mix Match, Co-Refinement, and Co Refurbishing, we aim to enhance the detection performance while reducing dependence on manual annotations. With the add-on semi-supervised modules, the accuracy rate of the MFCC+CNN model is 92.9%, an increase of 3.8% to the baseline model. The research contributes to the field of lung disease sound detection by addressing challenges such as individual differences, feature insufficient labeled data.
>
---
#### [replaced 010] Zero-Shot Voice Conversion via Content-Aware Timbre Ensemble and Conditional Flow Matching
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.02026v2](http://arxiv.org/pdf/2411.02026v2)**

> **作者:** Yu Pan; Yuguang Yang; Jixun Yao; Lei Ma; Jianjun Zhao
>
> **备注:** Work in progress; 5 pages;
>
> **摘要:** Despite recent advances in zero-shot voice conversion (VC), achieving speaker similarity and naturalness comparable to ground-truth recordings remains a significant challenge. In this letter, we propose CTEFM-VC, a zero-shot VC framework that integrates content-aware timbre ensemble modeling with conditional flow matching. Specifically, CTEFM-VC decouples utterances into content and timbre representations and leverages a conditional flow matching model to reconstruct the Mel-spectrogram of the source speech. To enhance its timbre modeling capability and naturalness of generated speech, we first introduce a context-aware timbre ensemble modeling approach that adaptively integrates diverse speaker verification embeddings and enables the effective utilization of source content and target timbre elements through a cross-attention module. Furthermore, a structural similarity-based timbre loss is presented to jointly train CTEFM-VC end-to-end. Experiments show that CTEFM-VC consistently achieves the best performance in all metrics assessing speaker similarity, speech naturalness, and intelligibility, significantly outperforming state-of-the-art zero-shot VC systems.
>
---
