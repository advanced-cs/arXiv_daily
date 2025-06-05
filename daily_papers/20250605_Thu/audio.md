# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Conformer-based Ultrasound-to-Speech Conversion
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于语音转换任务，旨在解决无声语音接口中从超声图像到语音的转换问题。作者采用两种基于Conformer的深度神经网络模型（Base和带bi-LSTM的版本），在Ultrasuite-Tal80数据集上进行实验，并使用HiFi-GAN合成语音。结果表明，Conformer模型在感知质量上优于传统CNN，且训练效率更高。**

- **链接: [http://arxiv.org/pdf/2506.03831v1](http://arxiv.org/pdf/2506.03831v1)**

> **作者:** Ibrahim Ibrahimov; Zainkó Csaba; Gábor Gosztolya
>
> **备注:** accepted to Interspeech 2025
>
> **摘要:** Deep neural networks have shown promising potential for ultrasound-to-speech conversion task towards Silent Speech Interfaces. In this work, we applied two Conformer-based DNN architectures (Base and one with bi-LSTM) for this task. Speaker-specific models were trained on the data of four speakers from the Ultrasuite-Tal80 dataset, while the generated mel spectrograms were synthesized to audio waveform using a HiFi-GAN vocoder. Compared to a standard 2D-CNN baseline, objective measurements (MSE and mel cepstral distortion) showed no statistically significant improvement for either model. However, a MUSHRA listening test revealed that Conformer with bi-LSTM provided better perceptual quality, while Conformer Base matched the performance of the baseline along with a 3x faster training time due to its simpler architecture. These findings suggest that Conformer-based models, especially the Conformer with bi-LSTM, offer a promising alternative to CNNs for ultrasound-to-speech conversion.
>
---
#### [new 002] Towards Better Disentanglement in Non-Autoregressive Zero-Shot Expressive Voice Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音转换任务，旨在实现无源语言的表达性语音转换。主要解决源音色泄露问题并提升风格迁移效果。工作包括：采用多语言离散语音单元、增强嵌入表示、引入混合风格归一化、结合局部F0信息与全局韵律特征等方法，以改善内容与风格的解耦和表达性迁移效果。**

- **链接: [http://arxiv.org/pdf/2506.04013v1](http://arxiv.org/pdf/2506.04013v1)**

> **作者:** Seymanur Akti; Tuan Nam Nguyen; Alexander Waibel
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Expressive voice conversion aims to transfer both speaker identity and expressive attributes from a target speech to a given source speech. In this work, we improve over a self-supervised, non-autoregressive framework with a conditional variational autoencoder, focusing on reducing source timbre leakage and improving linguistic-acoustic disentanglement for better style transfer. To minimize style leakage, we use multilingual discrete speech units for content representation and reinforce embeddings with augmentation-based similarity loss and mix-style layer normalization. To enhance expressivity transfer, we incorporate local F0 information via cross-attention and extract style embeddings enriched with global pitch and energy features. Experiments show our model outperforms baselines in emotion and speaker similarity, demonstrating superior style adaptation and reduced source style leakage.
>
---
#### [new 003] A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频生成任务，旨在解决纹理声音的合成与分析问题。提出了TexStat损失函数和验证指标，用于衡量纹理声音相似性；设计了轻量级可微分合成器TexEnv，并构建了TexDSP生成模型。通过实验验证其有效性，适用于生成任务及评估。**

- **链接: [http://arxiv.org/pdf/2506.04073v1](http://arxiv.org/pdf/2506.04073v1)**

> **作者:** Esteban Gutiérrez; Frederic Font; Xavier Serra; Lonce Wyse
>
> **备注:** Accepted to the 28th International Conference on Digital Audio Effects (DAFx 2025) to be held in Ancona, Italy. 8 pages, one diagram and 5 tables
>
> **摘要:** In this work, we introduce TexStat, a novel loss function specifically designed for the analysis and synthesis of texture sounds characterized by stochastic structure and perceptual stationarity. Drawing inspiration from the statistical and perceptual framework of McDermott and Simoncelli, TexStat identifies similarities between signals belonging to the same texture category without relying on temporal structure. We also propose using TexStat as a validation metric alongside Frechet Audio Distances (FAD) to evaluate texture sound synthesis models. In addition to TexStat, we present TexEnv, an efficient, lightweight and differentiable texture sound synthesizer that generates audio by imposing amplitude envelopes on filtered noise. We further integrate these components into TexDSP, a DDSP-inspired generative model tailored for texture sounds. Through extensive experiments across various texture sound types, we demonstrate that TexStat is perceptually meaningful, time-invariant, and robust to noise, features that make it effective both as a loss function for generative tasks and as a validation metric. All tools and code are provided as open-source contributions and our PyTorch implementations are efficient, differentiable, and highly configurable, enabling its use in both generative tasks and as a perceptually grounded evaluation metric.
>
---
#### [new 004] Comparative Analysis of Fast and High-Fidelity Neural Vocoders for Low-Latency Streaming Synthesis in Resource-Constrained Environments
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决低延迟流式神经声码器在资源受限环境中的效率问题。作者提出了多流Wavehax（MS-Wavehax），通过多流分解优化延迟-吞吐权衡，并分析流式合成瓶颈，提供优化见解。**

- **链接: [http://arxiv.org/pdf/2506.03554v1](http://arxiv.org/pdf/2506.03554v1)**

> **作者:** Reo Yoneyama; Masaya Kawamura; Ryo Terashima; Ryuichi Yamamoto; Tomoki Toda
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** In real-time speech synthesis, neural vocoders often require low-latency synthesis through causal processing and streaming. However, streaming introduces inefficiencies absent in batch synthesis, such as limited parallelism, inter-frame dependency management, and parameter loading overhead. This paper proposes multi-stream Wavehax (MS-Wavehax), an efficient neural vocoder for low-latency streaming, by extending the aliasing-free neural vocoder Wavehax with multi-stream decomposition. We analyze the latency-throughput trade-off in a CPU-only environment and identify key bottlenecks in streaming neural vocoders. Our findings provide practical insights for optimizing chunk sizes and designing vocoders tailored to specific application demands and hardware constraints. Furthermore, our subjective evaluations show that MS-Wavehax delivers high speech quality under causal and non-causal conditions while being remarkably compact and easily deployable in resource-constrained environments.
>
---
#### [new 005] Local Equivariance Error-Based Metrics for Evaluating Sampling-Frequency-Independent Property of Neural Network
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决神经网络在不同采样频率下性能下降的问题。作者提出基于局部等变误差（LEE）的评估指标，衡量网络对采样频率变化的鲁棒性（SFI属性），并通过实验验证其与性能退化的相关性。**

- **链接: [http://arxiv.org/pdf/2506.03550v1](http://arxiv.org/pdf/2506.03550v1)**

> **作者:** Kanami Imamura; Tomohiko Nakamura; Norihiro Takamune; Kohei Yatabe; Hiroshi Saruwatari
>
> **备注:** 5 pages, 4 figures, accepted for European Signal Processing Conference 2025 (EUSIPCO 2025)
>
> **摘要:** Audio signal processing methods based on deep neural networks (DNNs) are typically trained only at a single sampling frequency (SF) and therefore require signal resampling to handle untrained SFs. However, recent studies have shown that signal resampling can degrade performance with untrained SFs. This problem has been overlooked because most studies evaluate only the performance at trained SFs. In this paper, to assess the robustness of DNNs to SF changes, which we refer to as the SF-independent (SFI) property, we propose three metrics to quantify the SFI property on the basis of local equivariance error (LEE). LEE measures the robustness of DNNs to input transformations. By using signal resampling as input transformation, we extend LEE to measure the robustness of audio source separation methods to signal resampling. The proposed metrics are constructed to quantify the SFI property in specific network components responsible for predicting time-frequency masks. Experiments on music source separation demonstrated a strong correlation between the proposed metrics and performance degradation at untrained SFs.
>
---
#### [new 006] From Spikes to Speech: NeuroVoc -- A Biologically Plausible Vocoder Framework for Auditory Perception and Cochlear Implant Simulation
- **分类: cs.SD; q-bio.NC**

- **简介: 该论文提出了NeuroVoc，一种基于神经活动模式重建音频波形的通用声码器框架，用于模拟听觉感知和人工耳蜗植入效果。旨在解决不同听觉模型间语音编码兼容性差的问题，通过模块化设计实现正常听力与电刺激听力的对比评估，并验证其在噪声中语音识别的表现差异。**

- **链接: [http://arxiv.org/pdf/2506.03959v1](http://arxiv.org/pdf/2506.03959v1)**

> **作者:** Jacob de Nobel; Jeroen J. Briaire; Thomas H. W. Baeck; Anna V. Kononova; Johan H. M. Frijns
>
> **备注:** 43 Pages, 11 Figures, 2 Tables
>
> **摘要:** We present NeuroVoc, a flexible model-agnostic vocoder framework that reconstructs acoustic waveforms from simulated neural activity patterns using an inverse Fourier transform. The system applies straightforward signal processing to neurogram representations, time-frequency binned outputs from auditory nerve fiber models. Crucially, the model architecture is modular, allowing for easy substitution or modification of the underlying auditory models. This flexibility eliminates the need for speech-coding-strategy-specific vocoder implementations when simulating auditory perception in cochlear implant (CI) users. It also allows direct comparisons between normal hearing (NH) and electrical hearing (EH) models, as demonstrated in this study. The vocoder preserves distinctive features of each model; for example, the NH model retains harmonic structure more faithfully than the EH model. We evaluated perceptual intelligibility in noise using an online Digits-in-Noise (DIN) test, where participants completed three test conditions: one with standard speech, and two with vocoded speech using the NH and EH models. Both the standard DIN test and the EH-vocoded groups were statistically equivalent to clinically reported data for NH and CI listeners. On average, the NH and EH vocoded groups increased SRT compared to the standard test by 2.4 dB and 7.1 dB, respectively. These findings show that, although some degradation occurs, the vocoder can reconstruct intelligible speech under both hearing models and accurately reflects the reduced speech-in-noise performance experienced by CI users.
>
---
#### [new 007] A Novel Data Augmentation Approach for Automatic Speaking Assessment on Opinion Expressions
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动化口语评估任务，旨在解决意见表达口语评估中标记数据不足的问题。通过利用大语言模型生成多样化回答，结合文本到语音合成和动态损失加权，提升低资源场景下的评分效果，并融合多模态信息进行评分预测。**

- **链接: [http://arxiv.org/pdf/2506.04077v1](http://arxiv.org/pdf/2506.04077v1)**

> **作者:** Chung-Chun Wang; Jhen-Ke Lin; Hao-Chien Lu; Hong-Yun Lin; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Automated speaking assessment (ASA) on opinion expressions is often hampered by the scarcity of labeled recordings, which restricts prompt diversity and undermines scoring reliability. To address this challenge, we propose a novel training paradigm that leverages a large language models (LLM) to generate diverse responses of a given proficiency level, converts responses into synthesized speech via speaker-aware text-to-speech synthesis, and employs a dynamic importance loss to adaptively reweight training instances based on feature distribution differences between synthesized and real speech. Subsequently, a multimodal large language model integrates aligned textual features with speech signals to predict proficiency scores directly. Experiments conducted on the LTTC dataset show that our approach outperforms methods relying on real data or conventional augmentation, effectively mitigating low-resource constraints and enabling ASA on opinion expressions with cross-modal information.
>
---
#### [new 008] Acoustically Precise Hesitation Tagging Is Essential for End-to-End Verbatim Transcription Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决自动口语评估中对犹豫词（如填充词）精确标注的问题。通过微调Whisper模型并比较不同标注方案，发现基于Gemini 2.0 Flash的精确犹豫标注显著提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.04076v1](http://arxiv.org/pdf/2506.04076v1)**

> **作者:** Jhen-Ke Lin; Hao-Chien Lu; Chung-Chun Wang; Hong-Yun Lin; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Verbatim transcription for automatic speaking assessment demands accurate capture of disfluencies, crucial for downstream tasks like error analysis and feedback. However, many ASR systems discard or generalize hesitations, losing important acoustic details. We fine-tune Whisper models on the Speak & Improve 2025 corpus using low-rank adaptation (LoRA), without recourse to external audio training data. We compare three annotation schemes: removing hesitations (Pure), generic tags (Rich), and acoustically precise fillers inferred by Gemini 2.0 Flash from existing audio-transcript pairs (Extra). Our challenge system achieved 6.47% WER (Pure) and 5.81% WER (Extra). Post-challenge experiments reveal that fine-tuning Whisper Large V3 Turbo with the "Extra" scheme yielded a 5.5% WER, an 11.3% relative improvement over the "Pure" scheme (6.2% WER). This demonstrates that explicit, realistic filled-pause labeling significantly enhances ASR accuracy for verbatim L2 speech transcription.
>
---
#### [new 009] Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出“歌唱语音深度伪造溯源（SVDSA）”任务，旨在通过多模态基础模型（MMFM）识别伪造歌声的来源。研究验证了MMFM在捕捉音色、音高及合成特征上的优势，并提出了融合基础模型的新框架COFFE，采用Chernoff距离作为损失函数，有效提升了SVDSA性能。**

- **链接: [http://arxiv.org/pdf/2506.03364v1](http://arxiv.org/pdf/2506.03364v1)**

> **作者:** Orchid Chetia Phukan; Girish; Mohd Mujtaba Akhtar; Swarup Ranjan Behera; Priyabrata Mallick; Pailla Balakrishna Reddy; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this work, we introduce the task of singing voice deepfake source attribution (SVDSA). We hypothesize that multimodal foundation models (MMFMs) such as ImageBind, LanguageBind will be most effective for SVDSA as they are better equipped for capturing subtle source-specific characteristics-such as unique timbre, pitch manipulation, or synthesis artifacts of each singing voice deepfake source due to their cross-modality pre-training. Our experiments with MMFMs, speech foundation models and music foundation models verify the hypothesis that MMFMs are the most effective for SVDSA. Furthermore, inspired from related research, we also explore fusion of foundation models (FMs) for improved SVDSA. To this end, we propose a novel framework, COFFE which employs Chernoff Distance as novel loss function for effective fusion of FMs. Through COFFE with the symphony of MMFMs, we attain the topmost performance in comparison to all the individual FMs and baseline fusion methods.
>
---
#### [new 010] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 论文提出UniCUE，统一中文手势语音视频到语音生成框架，解决因数据不足和中间文本依赖导致的语音生成误差与同步问题。**

- **链接: [http://arxiv.org/pdf/2506.04134v1](http://arxiv.org/pdf/2506.04134v1)**

> **作者:** Jinting Wang; Shan Yang; Li Liu
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading through hand coding, providing precise speech perception support for the hearing-impaired. CS Video-to-Speech generation (CSV2S) task aims to convert the CS visual expressions (CS videos) of hearing-impaired individuals into comprehensible speech signals. Direct generation of speech from CS video (called single CSV2S) yields poor performance due to insufficient CS data. Current research mostly focuses on CS Recognition (CSR), which convert video content into linguistic text. Based on this, one straightforward way of CSV2S is to combine CSR with a Text-to-Speech system. This combined architecture relies on text as an intermediate medium for stepwise cross-modal alignment, which may lead to error propagation and temporal misalignment between speech and video dynamics. To address these challenges, we propose a novel approach that directly generates speech from CS videos without relying on intermediate text. Building upon this, we propose UniCUE, the first unified framework for CSV2S, whose core innovation lies in the integration of the CSR task that provides fine-grained visual-semantic information to facilitate speech generation from CS videos. More precisely, (1) a novel fine-grained semantic alignment pool to ensure precise mapping between visual features and speech contents; (2) a VisioPhonetic adapter to bridge cross-task representations, ensuring seamless compatibility between two distinct tasks (i.e., CSV2S and CSR); (3) a pose-aware visual processor is introduced to enhance fine-grained spatiotemporal correlations between lip and hand movements in CS video. Experiments on our new established Chinese CS dataset (14 cuers1: 8 hearing-impaired and 6 normal-hearing) show that our UniCUE significantly reduces Word Error Rate by 78.3% and improves lip-speech synchronization by 32% compared to the single CSV2S.
>
---
#### [new 011] Efficient Data Selection for Domain Adaptation of ASR Using Pseudo-Labels and Multi-Stage Filtering
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别（ASR）领域，旨在解决小组织在计算和标注资源有限情况下进行跨域模型适配的问题。通过使用伪标签和多阶段过滤方法，结合WER预测、NER和CER分析等策略，从大量数据中选取高质量训练样本，实现高效微调，在大幅减少训练数据量的同时保持识别性能。**

- **链接: [http://arxiv.org/pdf/2506.03681v1](http://arxiv.org/pdf/2506.03681v1)**

> **作者:** Pradeep Rangappa; Andres Carofilis; Jeena Prakash; Shashi Kumar; Sergio Burdisso; Srikanth Madikeri; Esau Villatoro-Tello; Bidisha Sharma; Petr Motlicek; Kadri Hacioglu; Shankar Venkatesan; Saurabh Vyas; Andreas Stolcke
>
> **备注:** Accepted at Interspeech 2025, Netherlands
>
> **摘要:** Fine-tuning pretrained ASR models for specific domains is challenging for small organizations with limited labeled data and computational resources. Here, we explore different data selection pipelines and propose a robust approach that improves ASR adaptation by filtering pseudo-labels generated using Whisper (encoder-decoder) and Zipformer (transducer) models. Our approach integrates multiple selection strategies -- including word error rate (WER) prediction, named entity recognition (NER), and character error rate (CER) analysis -- to extract high-quality training segments. We evaluate our method on Whisper and Zipformer using a 7500-hour baseline, comparing it to a CER-based approach relying on hypotheses from three ASR systems. Fine-tuning on 7500 hours of pseudo-labeled call center data achieves 12.3% WER, while our filtering reduces the dataset to 100 hours (1.4%) with similar performance; a similar trend is observed on Fisher English.
>
---
#### [new 012] Sounding that Object: Interactive Object-Aware Image to Audio Generation
- **分类: cs.CV; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于图像到音频生成任务，旨在解决复杂场景中多对象声音生成的问题。作者提出了一种交互式物体感知的音频生成模型，通过结合物体中心学习和条件扩散模型，使用户能基于图像中的特定物体生成对应声音，并验证了其注意力机制的有效性。**

- **链接: [http://arxiv.org/pdf/2506.04214v1](http://arxiv.org/pdf/2506.04214v1)**

> **作者:** Tingle Li; Baihe Huang; Xiaobin Zhuang; Dongya Jia; Jiawei Chen; Yuping Wang; Zhuo Chen; Gopala Anumanchipalli; Yuxuan Wang
>
> **备注:** ICML 2025
>
> **摘要:** Generating accurate sounds for complex audio-visual scenes is challenging, especially in the presence of multiple objects and sound sources. In this paper, we propose an {\em interactive object-aware audio generation} model that grounds sound generation in user-selected visual objects within images. Our method integrates object-centric learning into a conditional latent diffusion model, which learns to associate image regions with their corresponding sounds through multi-modal attention. At test time, our model employs image segmentation to allow users to interactively generate sounds at the {\em object} level. We theoretically validate that our attention mechanism functionally approximates test-time segmentation masks, ensuring the generated audio aligns with selected objects. Quantitative and qualitative evaluations show that our model outperforms baselines, achieving better alignment between objects and their associated sounds. Project page: https://tinglok.netlify.app/files/avobject/
>
---
#### [new 013] Brain-tuned Speech Models Better Reflect Speech Processing Stages in the Brain
- **分类: cs.CL; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文属于语音处理与认知科学交叉任务，旨在解决预训练语音模型语义表征与人脑处理不一致的问题。作者通过使用脑电数据微调模型，发现脑调后的模型不仅提升语义理解，还更准确反映大脑从声学到语义的层级处理机制。**

- **链接: [http://arxiv.org/pdf/2506.03832v1](http://arxiv.org/pdf/2506.03832v1)**

> **作者:** Omer Moussa; Mariya Toneva
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Pretrained self-supervised speech models excel in speech tasks but do not reflect the hierarchy of human speech processing, as they encode rich semantics in middle layers and poor semantics in late layers. Recent work showed that brain-tuning (fine-tuning models using human brain recordings) improves speech models' semantic understanding. Here, we examine how well brain-tuned models further reflect the brain's intermediate stages of speech processing. We find that late layers of brain-tuned models substantially improve over pretrained models in their alignment with semantic language regions. Further layer-wise probing reveals that early layers remain dedicated to low-level acoustic features, while late layers become the best at complex high-level tasks. These findings show that brain-tuned models not only perform better but also exhibit a well-defined hierarchical processing going from acoustic to semantic representations, making them better model organisms for human speech processing.
>
---
#### [new 014] MFLA: Monotonic Finite Look-ahead Attention for Streaming Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决预训练模型如Whisper在流式语音识别中的应用问题。通过设计连续积分放电机制和单调有限前瞻注意力机制，实现语音与文本的准单调对齐，并结合wait-k解码策略，达到延迟与质量的平衡。**

- **链接: [http://arxiv.org/pdf/2506.03722v1](http://arxiv.org/pdf/2506.03722v1)**

> **作者:** Yinfeng Xia; Huiyan Li; Chenyang Le; Manhong Wang; Yutao Sun; Xingyang Ma; Yanmin Qian
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Applying large pre-trained speech models like Whisper has shown promise in reducing training costs for various speech tasks. However, integrating these models into streaming systems remains a challenge. This paper presents a novel prefix-to-prefix training framework for streaming recognition by fine-tuning the Whisper. We introduce the Continuous Integrate-and-Fire mechanism to establish a quasi-monotonic alignment between continuous speech sequences and discrete text tokens. Additionally, we design Monotonic Finite Look-ahead Attention, allowing each token to attend to infinite left-context and finite right-context from the speech sequences. We also employ the wait-k decoding strategy to simplify the decoding process while ensuring consistency between training and testing. Our theoretical analysis and experiments demonstrate that this approach achieves a controllable trade-off between latency and quality, making it suitable for various streaming applications.
>
---
#### [new 015] BitTTS: Highly Compact Text-to-Speech Using 1.58-bit Quantization and Weight Indexing
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决模型体积大、难以部署到设备上的问题。作者通过引入1.58位量化和权重索引技术，大幅减小模型规模，同时保持合成质量，实现了高效的轻量级TTS模型BitTTS。**

- **链接: [http://arxiv.org/pdf/2506.03515v1](http://arxiv.org/pdf/2506.03515v1)**

> **作者:** Masaya Kawamura; Takuya Hasumi; Yuma Shirahata; Ryuichi Yamamoto
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** This paper proposes a highly compact, lightweight text-to-speech (TTS) model for on-device applications. To reduce the model size, the proposed model introduces two techniques. First, we introduce quantization-aware training (QAT), which quantizes model parameters during training to as low as 1.58-bit. In this case, most of 32-bit model parameters are quantized to ternary values {-1, 0, 1}. Second, we propose a method named weight indexing. In this method, we save a group of 1.58-bit weights as a single int8 index. This allows for efficient storage of model parameters, even on hardware that treats values in units of 8-bit. Experimental results demonstrate that the proposed method achieved 83 % reduction in model size, while outperforming the baseline of similar model size without quantization in synthesis quality.
>
---
#### [new 016] Tone recognition in low-resource languages of North-East India: peeling the layers of SSL-based speech models
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **简介: 该论文属于语音处理任务，旨在解决低资源语言的声调识别问题。研究者评估了基于自监督学习（SSL）的Wav2vec2.0模型在三种印度东北部低资源语言（Angami、Ao、Mizo）中的声调识别表现，分析不同模型层及预训练语言对识别效果的影响，并探讨声调类型、方言差异等因素的作用，以提升低资源场景下的声调识别能力。**

- **链接: [http://arxiv.org/pdf/2506.03606v1](http://arxiv.org/pdf/2506.03606v1)**

> **作者:** Parismita Gogoi; Sishir Kalita; Wendy Lalhminghlui; Viyazonuo Terhiija; Moakala Tzudir; Priyankoo Sarmah; S. R. M. Prasanna
>
> **备注:** Accepted in Interspeech2025
>
> **摘要:** This study explores the use of self-supervised learning (SSL) models for tone recognition in three low-resource languages from North Eastern India: Angami, Ao, and Mizo. We evaluate four Wav2vec2.0 base models that were pre-trained on both tonal and non-tonal languages. We analyze tone-wise performance across the layers for all three languages and compare the different models. Our results show that tone recognition works best for Mizo and worst for Angami. The middle layers of the SSL models are the most important for tone recognition, regardless of the pre-training language, i.e. tonal or non-tonal. We have also found that the tone inventory, tone types, and dialectal variations affect tone recognition. These findings provide useful insights into the strengths and weaknesses of SSL-based embeddings for tonal languages and highlight the potential for improving tone recognition in low-resource settings. The source code is available at GitHub 1 .
>
---
## 更新

#### [replaced 001] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2406.01205v3](http://arxiv.org/pdf/2406.01205v3)**

> **作者:** Shengpeng Ji; Qian Chen; Wen Wang; Jialong Zuo; Minghui Fang; Ziyue Jiang; Hai Huang; Zehan Wang; Xize Cheng; Siqi Zheng; Zhou Zhao
>
> **备注:** ACL 2025 Main
>
> **摘要:** In this paper, we present ControlSpeech, a text-to-speech (TTS) system capable of fully cloning the speaker's voice and enabling arbitrary control and adjustment of speaking style. Prior zero-shot TTS models only mimic the speaker's voice without further control and adjustment capabilities while prior controllable TTS models cannot perform speaker-specific voice generation. Therefore, ControlSpeech focuses on a more challenging task: a TTS system with controllable timbre, content, and style at the same time. ControlSpeech takes speech prompts, content prompts, and style prompts as inputs and utilizes bidirectional attention and mask-based parallel decoding to capture codec representations corresponding to timbre, content, and style in a discrete decoupling codec space. Moreover, we analyze the many-to-many issue in textual style control and propose the Style Mixture Semantic Density (SMSD) module, which is based on Gaussian mixture density networks, to resolve this problem. To facilitate empirical validations, we make available a new style controllable dataset called VccmDataset. Our experimental results demonstrate that ControlSpeech exhibits comparable or state-of-the-art (SOTA) performance in terms of controllability, timbre similarity, audio quality, robustness, and generalizability. The relevant code and demo are available at https://github.com/jishengpeng/ControlSpeech .
>
---
#### [replaced 002] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2402.12208v4](http://arxiv.org/pdf/2402.12208v4)**

> **作者:** Shengpeng Ji; Minghui Fang; Jialong Zuo; Ziyue Jiang; Dingdong Wang; Hanting Wang; Hai Huang; Zhou Zhao
>
> **备注:** ACL 2025 Main
>
> **摘要:** In recent years, large language models have achieved significant success in generative tasks related to speech, audio, music, and other signal domains. A crucial element of these models is the discrete acoustic codecs, which serve as an intermediate representation replacing the mel-spectrogram. However, there exist several gaps between discrete codecs and downstream speech language models. Specifically, 1) Due to the reconstruction paradigm of the Codec model and the structure of residual vector quantization, the initial channel of the codebooks contains excessive information, making it challenging to directly generate acoustic tokens from weakly supervised signals such as text in downstream tasks. 2) numerous codebooks increases the burden on downstream speech language models. Consequently, leveraging the characteristics of speech language models, we propose Language-Codec. In the Language-Codec, we introduce a Masked Channel Residual Vector Quantization (MCRVQ) mechanism along with improved fourier transform structures and attention blocks, refined discriminator design to address the aforementioned gaps. We compare our method with competing audio compression algorithms and observe significant outperformance across extensive evaluations. Furthermore, we also validate the efficiency of the Language-Codec on downstream speech language models. The source code and pre-trained models can be accessed at https://github.com/jishengpeng/languagecodec .
>
---
#### [replaced 003] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
- **分类: cs.MM; cs.CV; cs.GR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.16331v2](http://arxiv.org/pdf/2411.16331v2)**

> **作者:** Xiaozhong Ji; Xiaobin Hu; Zhihong Xu; Junwei Zhu; Chuming Lin; Qingdong He; Jiangning Zhang; Donghao Luo; Yi Chen; Qin Lin; Qinglin Lu; Chengjie Wang
>
> **备注:** refer to our main-page \url{https://jixiaozhong.github.io/Sonic/}
>
> **摘要:** The study of talking face generation mainly explores the intricacies of synchronizing facial movements and crafting visually appealing, temporally-coherent animations. However, due to the limited exploration of global audio perception, current approaches predominantly employ auxiliary visual and spatial knowledge to stabilize the movements, which often results in the deterioration of the naturalness and temporal inconsistencies.Considering the essence of audio-driven animation, the audio signal serves as the ideal and unique priors to adjust facial expressions and lip movements, without resorting to interference of any visual signals. Based on this motivation, we propose a novel paradigm, dubbed as Sonic, to {s}hift f{o}cus on the exploration of global audio per{c}ept{i}o{n}.To effectively leverage global audio knowledge, we disentangle it into intra- and inter-clip audio perception and collaborate with both aspects to enhance overall perception.For the intra-clip audio perception, 1). \textbf{Context-enhanced audio learning}, in which long-range intra-clip temporal audio knowledge is extracted to provide facial expression and lip motion priors implicitly expressed as the tone and speed of speech. 2). \textbf{Motion-decoupled controller}, in which the motion of the head and expression movement are disentangled and independently controlled by intra-audio clips. Most importantly, for inter-clip audio perception, as a bridge to connect the intra-clips to achieve the global perception, \textbf{Time-aware position shift fusion}, in which the global inter-clip audio information is considered and fused for long-audio inference via through consecutively time-aware shifted windows. Extensive experiments demonstrate that the novel audio-driven paradigm outperform existing SOTA methodologies in terms of video quality, temporally consistency, lip synchronization precision, and motion diversity.
>
---
#### [replaced 004] Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.12532v2](http://arxiv.org/pdf/2505.12532v2)**

> **作者:** Ahmet Bilican; M. Akın Yılmaz; A. Murat Tekalp; R. Gökberk Cinbiş
>
> **摘要:** Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA's minimum, ideal for extreme parameter-efficient scenarios. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity.
>
---
#### [replaced 005] Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2504.12423v2](http://arxiv.org/pdf/2504.12423v2)**

> **作者:** Haohan Shi; Xiyu Shi; Safak Dogan; Saif Alzubi; Tianjin Huang; Yunxiao Zhang
>
> **备注:** Accepted by EUSIPCO 2025
>
> **摘要:** Existing Audio Deepfake Detection (ADD) systems often struggle to generalise effectively due to the significantly degraded audio quality caused by audio codec compression and channel transmission effects in real-world communication scenarios. To address this challenge, we developed a rigorous benchmark to evaluate the performance of the ADD system under such scenarios. We introduced ADD-C, a new test dataset to evaluate the robustness of ADD systems under diverse communication conditions, including different combinations of audio codecs for compression and packet loss rates. Benchmarking three baseline ADD models on the ADD-C dataset demonstrated a significant decline in robustness under such conditions. A novel Data Augmentation (DA) strategy was proposed to improve the robustness of ADD systems. Experimental results demonstrated that the proposed approach significantly enhances the performance of ADD systems on the proposed ADD-C dataset. Our benchmark can assist future efforts towards building practical and robustly generalisable ADD systems.
>
---
#### [replaced 006] PAST: Phonetic-Acoustic Speech Tokenizer
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14470v2](http://arxiv.org/pdf/2505.14470v2)**

> **作者:** Nadav Har-Tuv; Or Tal; Yossi Adi
>
> **摘要:** We present PAST, a novel end-to-end framework that jointly models phonetic information alongside signal reconstruction, eliminating the need for external pretrained models. Unlike previous approaches that rely on pretrained self-supervised models, PAST employs supervised phonetic data, directly integrating domain knowledge into the tokenization process via auxiliary tasks. Additionally, we introduce a streamable, causal variant of PAST, enabling real-time speech applications. Results demonstrate that PAST surpasses existing evaluated baseline tokenizers across common evaluation metrics, including phonetic representation and speech reconstruction. Notably, PAST also achieves superior performance when serving as a speech representation for speech language models, further highlighting its effectiveness as a foundation for spoken language generation. To foster further research, we release the full implementation. For code, model checkpoints, and samples see: https://pages.cs.huji.ac.il/adiyoss-lab/PAST
>
---
#### [replaced 007] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.14627v3](http://arxiv.org/pdf/2502.14627v3)**

> **作者:** Yuguo Yin; Yuxin Xie; Wenyuan Yang; Dongchao Yang; Jinghan Ru; Xianwei Zhuang; Liming Liang; Yuexian Zou
>
> **摘要:** Multilingual audio-text retrieval (ML-ATR) is a challenging task that aims to retrieve audio clips or multilingual texts from databases. However, existing ML-ATR schemes suffer from inconsistencies for instance similarity matching across languages. We theoretically analyze the inconsistency in terms of both multilingual modal alignment direction error and weight error, and propose the theoretical weight error upper bound for quantifying the inconsistency. Based on the analysis of the weight error upper bound, we find that the inconsistency problem stems from the data distribution error caused by random sampling of languages. We propose a consistent ML-ATR scheme using 1-to-k contrastive learning and audio-English co-anchor contrastive learning, aiming to mitigate the negative impact of data distribution error on recall and consistency in ML-ATR. Experimental results on the translated AudioCaps and Clotho datasets show that our scheme achieves state-of-the-art performance on recall and consistency metrics for eight mainstream languages, including English. Our code will be available at https://github.com/ATRI-ACL/ATRI-ACL.
>
---
#### [replaced 008] Accelerating Flow-Matching-Based Text-to-Speech via Empirically Pruned Step Sampling
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.19931v2](http://arxiv.org/pdf/2505.19931v2)**

> **作者:** Qixi Zheng; Yushen Chen; Zhikang Niu; Ziyang Ma; Xiaofei Wang; Kai Yu; Xie Chen
>
> **摘要:** Flow-matching-based text-to-speech (TTS) models, such as Voicebox, E2 TTS, and F5-TTS, have attracted significant attention in recent years. These models require multiple sampling steps to reconstruct speech from noise, making inference speed a key challenge. Reducing the number of sampling steps can greatly improve inference efficiency. To this end, we introduce Fast F5-TTS, a training-free approach to accelerate the inference of flow-matching-based TTS models. By inspecting the sampling trajectory of F5-TTS, we identify redundant steps and propose Empirically Pruned Step Sampling (EPSS), a non-uniform time-step sampling strategy that effectively reduces the number of sampling steps. Our approach achieves a 7-step generation with an inference RTF of 0.030 on an NVIDIA RTX 3090 GPU, making it 4 times faster than the original F5-TTS while maintaining comparable performance. Furthermore, EPSS performs well on E2 TTS models, demonstrating its strong generalization ability.
>
---
#### [replaced 009] Analyzing the Impact of Accent on English Speech: Acoustic and Articulatory Perspectives
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.15965v2](http://arxiv.org/pdf/2505.15965v2)**

> **作者:** Gowtham Premananth; Vinith Kugathasan; Carol Espy-Wilson
>
> **备注:** Accepted to be presented at Interspeech 2025
>
> **摘要:** Advancements in AI-driven speech-based applications have transformed diverse industries ranging from healthcare to customer service. However, the increasing prevalence of non-native accented speech in global interactions poses significant challenges for speech-processing systems, which are often trained on datasets dominated by native speech. This study investigates accented English speech through articulatory and acoustic analysis, identifying simpler coordination patterns and higher average pitch than native speech. Using eigenspectra and Vocal Tract Variable-based coordination features, we establish an efficient method for quantifying accent strength without relying on resource-intensive phonetic transcriptions. Our findings provide a new avenue for research on the impacts of accents on speech intelligibility and offer insights for developing inclusive, robust speech processing systems that accommodate diverse linguistic communities.
>
---
#### [replaced 010] Neural Scoring: A Refreshed End-to-End Approach for Speaker Recognition in Complex Conditions
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.16428v2](http://arxiv.org/pdf/2410.16428v2)**

> **作者:** Wan Lin; Junhui Chen; Tianhao Wang; Zhenyu Zhou; Lantian Li; Dong Wang
>
> **摘要:** Modern speaker verification systems primarily rely on speaker embeddings and cosine similarity. While effective, these methods struggle with multi-talker speech due to the unidentifiability of embedding vectors. We propose Neural Scoring (NS), a novel end-to-end framework that directly estimates verification posterior probabilities without relying on test-side embeddings, making it more powerful and robust to complex conditions, e.g., with multiple talkers. To address the challenge of training such end-to-end models, we introduce a multi-enrollment training strategy, which pairs each test utterance with multiple enrolled speakers and proves essential to the model's success. Experiments on the VoxCeleb dataset demonstrate that NS consistently outperforms both the baseline and several competitive methods, achieving an overall 70.36% reduction in Equal Error Rate (EER) compared to the baseline.
>
---
#### [replaced 011] Transformers in Speech Processing: A Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2303.11607v2](http://arxiv.org/pdf/2303.11607v2)**

> **作者:** Siddique Latif; Aun Zaidi; Heriberto Cuayahuitl; Fahad Shamshad; Moazzam Shoukat; Muhammad Usama; Junaid Qadir
>
> **备注:** Accepted in Computer Science Review 2025
>
> **摘要:** The remarkable success of transformers in the field of natural language processing has sparked the interest of the speech-processing community, leading to an exploration of their potential for modeling long-range dependencies within speech sequences. Recently, transformers have gained prominence across various speech-related domains, including automatic speech recognition, speech synthesis, speech translation, speech para-linguistics, speech enhancement, spoken dialogue systems, and numerous multimodal applications. In this paper, we present a comprehensive survey that aims to bridge research studies from diverse subfields within speech technology. By consolidating findings from across the speech technology landscape, we provide a valuable resource for researchers interested in harnessing the power of transformers to advance the field. We identify the challenges encountered by transformers in speech processing while also offering insights into potential solutions to address these issues.
>
---
#### [replaced 012] InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training
- **分类: cs.SD; cs.CL; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.02769v2](http://arxiv.org/pdf/2503.02769v2)**

> **作者:** Dingdong Wang; Jin Xu; Ruihang Chu; Zhifang Guo; Xiong Wang; Jincenzi Wu; Dongchao Yang; Shengpeng Ji; Junyang Lin
>
> **备注:** Accepted to ACL 2025; Data is available at: https://huggingface.co/datasets/ddwang2000/SpeechInstructBench
>
> **摘要:** Recent advancements in speech large language models (SpeechLLMs) have attracted considerable attention. Nonetheless, current methods exhibit suboptimal performance in adhering to speech instructions. Notably, the intelligence of models significantly diminishes when processing speech-form input as compared to direct text-form input. Prior work has attempted to mitigate this semantic inconsistency between speech and text representations through techniques such as representation and behavior alignment, which involve the meticulous design of data pairs during the post-training phase. In this paper, we introduce a simple and scalable training method called InSerter, which stands for Interleaved Speech-Text Representation Pre-training. InSerter is designed to pre-train large-scale unsupervised speech-text sequences, where the speech is synthesized from randomly selected segments of an extensive text corpus using text-to-speech conversion. Consequently, the model acquires the ability to generate textual continuations corresponding to the provided speech segments, obviating the need for intensive data design endeavors. To systematically evaluate speech instruction-following capabilities, we introduce SpeechInstructBench, the first comprehensive benchmark specifically designed for speech-oriented instruction-following tasks. Our proposed InSerter achieves SOTA performance in SpeechInstructBench and demonstrates superior or competitive results across diverse speech processing tasks.
>
---
#### [replaced 013] Multimodal Biomarkers for Schizophrenia: Towards Individual Symptom Severity Estimation
- **分类: eess.AS; cs.LG; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.16044v2](http://arxiv.org/pdf/2505.16044v2)**

> **作者:** Gowtham Premananth; Philip Resnik; Sonia Bansal; Deanna L. Kelly; Carol Espy-Wilson
>
> **备注:** Accepted to be presented at Interspeech 2025
>
> **摘要:** Studies on schizophrenia assessments using deep learning typically treat it as a classification task to detect the presence or absence of the disorder, oversimplifying the condition and reducing its clinical applicability. This traditional approach overlooks the complexity of schizophrenia, limiting its practical value in healthcare settings. This study shifts the focus to individual symptom severity estimation using a multimodal approach that integrates speech, video, and text inputs. We develop unimodal models for each modality and a multimodal framework to improve accuracy and robustness. By capturing a more detailed symptom profile, this approach can help in enhancing diagnostic precision and support personalized treatment, offering a scalable and objective tool for mental health assessment.
>
---
