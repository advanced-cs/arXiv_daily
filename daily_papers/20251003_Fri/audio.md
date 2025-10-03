# 音频 cs.SD;  eess.SP

- **最新发布 22 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Emotional Text-To-Speech Based on Mutual-Information-Guided Emotion-Timbre Disentanglement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于情感文本转语音任务，旨在解决现有方法无法捕捉参考语音细微声学细节的问题。通过分离音色与情感特征，实现更自然、丰富的情感语音生成。**

- **链接: [http://arxiv.org/pdf/2510.01722v1](http://arxiv.org/pdf/2510.01722v1)**

> **作者:** Jianing Yang; Sheng Li; Takahiro Shinozaki; Yuki Saito; Hiroshi Saruwatari
>
> **备注:** In Proceedings of the 17th Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC 2025)
>
> **摘要:** Current emotional Text-To-Speech (TTS) and style transfer methods rely on reference encoders to control global style or emotion vectors, but do not capture nuanced acoustic details of the reference speech. To this end, we propose a novel emotional TTS method that enables fine-grained phoneme-level emotion embedding prediction while disentangling intrinsic attributes of the reference speech. The proposed method employs a style disentanglement method to guide two feature extractors, reducing mutual information between timbre and emotion features, and effectively separating distinct style components from the reference speech. Experimental results demonstrate that our method outperforms baseline TTS systems in generating natural and emotionally rich speech. This work highlights the potential of disentangled and fine-grained representations in advancing the quality and flexibility of emotional TTS systems.
>
---
#### [new 002] High-Fidelity Speech Enhancement via Discrete Audio Tokens
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决现有方法依赖复杂流程和低采样率编码的问题。提出DAC-SE1框架，使用高分辨率离散音频表示，提升语音质量与语义一致性。**

- **链接: [http://arxiv.org/pdf/2510.02187v1](http://arxiv.org/pdf/2510.02187v1)**

> **作者:** Luca A. Lanzendörfer; Frédéric Berdoz; Antonis Asonitis; Roger Wattenhofer
>
> **摘要:** Recent autoregressive transformer-based speech enhancement (SE) methods have shown promising results by leveraging advanced semantic understanding and contextual modeling of speech. However, these approaches often rely on complex multi-stage pipelines and low sampling rate codecs, limiting them to narrow and task-specific speech enhancement. In this work, we introduce DAC-SE1, a simplified language model-based SE framework leveraging discrete high-resolution audio representations; DAC-SE1 preserves fine-grained acoustic details while maintaining semantic coherence. Our experiments show that DAC-SE1 surpasses state-of-the-art autoregressive SE methods on both objective perceptual metrics and in a MUSHRA human evaluation. We release our codebase and model checkpoints to support further research in scalable, unified, and high-quality speech enhancement.
>
---
#### [new 003] JaneEye: A 12-nm 2K-FPS 18.9-$μ$J/Frame Event-based Eye Tracking Accelerator
- **分类: eess.SP; cs.AR; cs.CV; cs.HC; eess.IV**

- **简介: 该论文属于眼动追踪任务，解决传统系统在精度、延迟和能效上的不足。提出JaneEye硬件加速器，利用事件相机数据，实现高效低功耗的眼动追踪。**

- **链接: [http://arxiv.org/pdf/2510.01213v1](http://arxiv.org/pdf/2510.01213v1)**

> **作者:** Tao Han; Ang Li; Qinyu Chen; Chang Gao
>
> **备注:** Accepted to 2026 IEEE 31st Asia and South Pacific Design Automation Conference (ASP-DAC) 2026
>
> **摘要:** Eye tracking has become a key technology for gaze-based interactions in Extended Reality (XR). However, conventional frame-based eye-tracking systems often fall short of XR's stringent requirements for high accuracy, low latency, and energy efficiency. Event cameras present a compelling alternative, offering ultra-high temporal resolution and low power consumption. In this paper, we present JaneEye, an energy-efficient event-based eye-tracking hardware accelerator designed specifically for wearable devices, leveraging sparse, high-temporal-resolution event data. We introduce an ultra-lightweight neural network architecture featuring a novel ConvJANET layer, which simplifies the traditional ConvLSTM by retaining only the forget gate, thereby halving computational complexity without sacrificing temporal modeling capability. Our proposed model achieves high accuracy with a pixel error of 2.45 on the 3ET+ dataset, using only 17.6K parameters, with up to 1250 Hz event frame rate. To further enhance hardware efficiency, we employ custom linear approximations of activation functions (hardsigmoid and hardtanh) and fixed-point quantization. Through software-hardware co-design, our 12-nm ASIC implementation operates at 400 MHz, delivering an end-to-end latency of 0.5 ms (equivalent to 2000 Frames Per Second (FPS)) at an energy efficiency of 18.9 $\mu$J/frame. JaneEye sets a new benchmark in low-power, high-performance eye-tracking solutions suitable for integration into next-generation XR wearables.
>
---
#### [new 004] Multi-bit Audio Watermarking
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频水印任务，旨在实现鲁棒且隐形的多比特水印。通过优化音频VAE的潜在空间添加水印，无需训练嵌入检测模型，提升水印性能。**

- **链接: [http://arxiv.org/pdf/2510.01968v1](http://arxiv.org/pdf/2510.01968v1)**

> **作者:** Luca A. Lanzendörfer; Kyle Fearne; Florian Grötschla; Roger Wattenhofer
>
> **摘要:** We present Timbru, a post-hoc audio watermarking model that achieves state-of-the-art robustness and imperceptibility trade-offs without training an embedder-detector model. Given any 44.1 kHz stereo music snippet, our method performs per-audio gradient optimization to add imperceptible perturbations in the latent space of a pretrained audio VAE, guided by a combined message and perceptual loss. The watermark can then be extracted using a pretrained CLAP model. We evaluate 16-bit watermarking on MUSDB18-HQ against AudioSeal, WavMark, and SilentCipher across common filtering, noise, compression, resampling, cropping, and regeneration attacks. Our approach attains the best average bit error rates, while preserving perceptual quality, demonstrating an efficient, dataset-free path to imperceptible audio watermarking.
>
---
#### [new 005] Bias beyond Borders: Global Inequalities in AI-Generated Music
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于AI音乐生成任务，旨在解决跨国家、语言和文化背景下的模型偏见问题。通过构建GlobalDISCO数据集，分析不同地区音乐质量差异及模型表现不均。**

- **链接: [http://arxiv.org/pdf/2510.01963v1](http://arxiv.org/pdf/2510.01963v1)**

> **作者:** Ahmet Solak; Florian Grötschla; Luca A. Lanzendörfer; Roger Wattenhofer
>
> **摘要:** While recent years have seen remarkable progress in music generation models, research on their biases across countries, languages, cultures, and musical genres remains underexplored. This gap is compounded by the lack of datasets and benchmarks that capture the global diversity of music. To address these challenges, we introduce GlobalDISCO, a large-scale dataset consisting of 73k music tracks generated by state-of-the-art commercial generative music models, along with paired links to 93k reference tracks in LAION-DISCO-12M. The dataset spans 147 languages and includes musical style prompts extracted from MusicBrainz and Wikipedia. The dataset is globally balanced, representing musical styles from artists across 79 countries and five continents. Our evaluation reveals large disparities in music quality and alignment with reference music between high-resource and low-resource regions. Furthermore, we find marked differences in model performance between mainstream and geographically niche genres, including cases where models generate music for regional genres that more closely align with the distribution of mainstream styles.
>
---
#### [new 006] HRTFformer: A Spatially-Aware Transformer for Personalized HRTF Upsampling in Immersive Audio Rendering
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于HRTF上采样任务，旨在解决个性化HRTF测量困难的问题。提出基于Transformer的模型，提升空间一致性与精度。**

- **链接: [http://arxiv.org/pdf/2510.01891v1](http://arxiv.org/pdf/2510.01891v1)**

> **作者:** Xuyi Hu; Jian Li; Shaojie Zhang; Stefan Goetz; Lorenzo Picinali; Ozgur B. Akan; Aidan O. T. Hogg
>
> **备注:** 10 pages and 5 figures
>
> **摘要:** Personalized Head-Related Transfer Functions (HRTFs) are starting to be introduced in many commercial immersive audio applications and are crucial for realistic spatial audio rendering. However, one of the main hesitations regarding their introduction is that creating personalized HRTFs is impractical at scale due to the complexities of the HRTF measurement process. To mitigate this drawback, HRTF spatial upsampling has been proposed with the aim of reducing measurements required. While prior work has seen success with different machine learning (ML) approaches, these models often struggle with long-range spatial consistency and generalization at high upsampling factors. In this paper, we propose a novel transformer-based architecture for HRTF upsampling, leveraging the attention mechanism to better capture spatial correlations across the HRTF sphere. Working in the spherical harmonic (SH) domain, our model learns to reconstruct high-resolution HRTFs from sparse input measurements with significantly improved accuracy. To enhance spatial coherence, we introduce a neighbor dissimilarity loss that promotes magnitude smoothness, yielding more realistic upsampling. We evaluate our method using both perceptual localization models and objective spectral distortion metrics. Experiments show that our model surpasses leading methods by a substantial margin in generating realistic, high-fidelity HRTFs.
>
---
#### [new 007] Exploring Resolution-Wise Shared Attention in Hybrid Mamba-U-Nets for Improved Cross-Corpus Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升跨语料库的泛化性能。通过结合Mamba和注意力机制，提出RWSA-MambaUNet模型，在减少参数和计算量的同时取得更好效果。**

- **链接: [http://arxiv.org/pdf/2510.01958v1](http://arxiv.org/pdf/2510.01958v1)**

> **作者:** Nikolai Lund Kühne; Jesper Jensen; Jan Østergaard; Zheng-Hua Tan
>
> **备注:** Submitted to IEEE for possible publication
>
> **摘要:** Recent advances in speech enhancement have shown that models combining Mamba and attention mechanisms yield superior cross-corpus generalization performance. At the same time, integrating Mamba in a U-Net structure has yielded state-of-the-art enhancement performance, while reducing both model size and computational complexity. Inspired by these insights, we propose RWSA-MambaUNet, a novel and efficient hybrid model combining Mamba and multi-head attention in a U-Net structure for improved cross-corpus performance. Resolution-wise shared attention (RWSA) refers to layerwise attention-sharing across corresponding time- and frequency resolutions. Our best-performing RWSA-MambaUNet model achieves state-of-the-art generalization performance on two out-of-domain test sets. Notably, our smallest model surpasses all baselines on the out-of-domain DNS 2020 test set in terms of PESQ, SSNR, and ESTOI, and on the out-of-domain EARS-WHAM_v2 test set in terms of SSNR, ESTOI, and SI-SDR, while using less than half the model parameters and a fraction of the FLOPs.
>
---
#### [new 008] SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于歌唱质量评估任务，旨在解决主观评估成本高、客观指标有限的问题。工作包括构建SingMOS-Pro数据集，扩展多维度标注，并基准多种评估方法。**

- **链接: [http://arxiv.org/pdf/2510.01812v1](http://arxiv.org/pdf/2510.01812v1)**

> **作者:** Yuxun Tang; Lan Liu; Wenhao Feng; Yiwen Zhao; Jionghao Han; Yifeng Yu; Jiatong Shi; Qin Jin
>
> **备注:** 4 pages, 5 figures; submitted to ICASSP 2026
>
> **摘要:** Singing voice generation progresses rapidly, yet evaluating singing quality remains a critical challenge. Human subjective assessment, typically in the form of listening tests, is costly and time consuming, while existing objective metrics capture only limited perceptual aspects. In this work, we introduce SingMOS-Pro, a dataset for automatic singing quality assessment. Building on our preview version SingMOS, which provides only overall ratings, SingMOS-Pro expands annotations of the additional part to include lyrics, melody, and overall quality, offering broader coverage and greater diversity. The dataset contains 7,981 singing clips generated by 41 models across 12 datasets, spanning from early systems to recent advances. Each clip receives at least five ratings from professional annotators, ensuring reliability and consistency. Furthermore, we explore how to effectively utilize MOS data annotated under different standards and benchmark several widely used evaluation methods from related tasks on SingMOS-Pro, establishing strong baselines and practical references for future research. The dataset can be accessed at https://huggingface.co/datasets/TangRain/SingMOS-Pro.
>
---
#### [new 009] Go witheFlow: Real-time Emotion Driven Audio Effects Modulation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐与情感计算任务，旨在解决机器缺乏情感表达的问题。通过分析生物信号和音频特征，实现实时音频效果调节。**

- **链接: [http://arxiv.org/pdf/2510.02171v1](http://arxiv.org/pdf/2510.02171v1)**

> **作者:** Edmund Dervakos; Spyridon Kantarelis; Vassilis Lyberatos; Jason Liartis; Giorgos Stamou
>
> **备注:** Accepted at NeurIPS Creative AI Track 2025: Humanity
>
> **摘要:** Music performance is a distinctly human activity, intrinsically linked to the performer's ability to convey, evoke, or express emotion. Machines cannot perform music in the human sense; they can produce, reproduce, execute, or synthesize music, but they lack the capacity for affective or emotional experience. As such, music performance is an ideal candidate through which to explore aspects of collaboration between humans and machines. In this paper, we introduce the witheFlow system, designed to enhance real-time music performance by automatically modulating audio effects based on features extracted from both biosignals and the audio itself. The system, currently in a proof-of-concept phase, is designed to be lightweight, able to run locally on a laptop, and is open-source given the availability of a compatible Digital Audio Workstation and sensors.
>
---
#### [new 010] RealClass: A Framework for Classroom Speech Simulation with Public Datasets and Game Engines
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音处理任务，解决课堂语音数据不足的问题。通过游戏引擎合成课堂噪声和RIR，构建RealClass数据集，用于提升教育AI模型性能。**

- **链接: [http://arxiv.org/pdf/2510.01462v1](http://arxiv.org/pdf/2510.01462v1)**

> **作者:** Ahmed Adel Attia; Jing Liu; Carol Espy Wilson
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2506.09206
>
> **摘要:** The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Classroom datasets remain limited and not publicly available, and the absence of dedicated classroom noise or Room Impulse Response (RIR) corpora prevents the use of standard data augmentation techniques. In this paper, we introduce a scalable methodology for synthesizing classroom noise and RIRs using game engines, a versatile framework that can extend to other domains beyond the classroom. Building on this methodology, we present RealClass, a dataset that combines a synthesized classroom noise corpus with a classroom speech dataset compiled from publicly available corpora. The speech data pairs a children's speech corpus with instructional speech extracted from YouTube videos to approximate real classroom interactions in clean conditions. Experiments on clean and noisy speech show that RealClass closely approximates real classroom speech, making it a valuable asset in the absence of abundant real classroom speech.
>
---
#### [new 011] SoundReactor: Frame-level Online Video-to-Audio Generation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出SoundReactor，解决在线视频到音频生成任务，实现实时、低延迟的音视频同步生成。**

- **链接: [http://arxiv.org/pdf/2510.02110v1](http://arxiv.org/pdf/2510.02110v1)**

> **作者:** Koichi Saito; Julian Tanke; Christian Simon; Masato Ishii; Kazuki Shimada; Zachary Novack; Zhi Zhong; Akio Hayakawa; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** Prevailing Video-to-Audio (V2A) generation models operate offline, assuming an entire video sequence or chunks of frames are available beforehand. This critically limits their use in interactive applications such as live content creation and emerging generative world models. To address this gap, we introduce the novel task of frame-level online V2A generation, where a model autoregressively generates audio from video without access to future video frames. Furthermore, we propose SoundReactor, which, to the best of our knowledge, is the first simple yet effective framework explicitly tailored for this task. Our design enforces end-to-end causality and targets low per-frame latency with audio-visual synchronization. Our model's backbone is a decoder-only causal transformer over continuous audio latents. For vision conditioning, it leverages grid (patch) features extracted from the smallest variant of the DINOv2 vision encoder, which are aggregated into a single token per frame to maintain end-to-end causality and efficiency. The model is trained through a diffusion pre-training followed by consistency fine-tuning to accelerate the diffusion head decoding. On a benchmark of diverse gameplay videos from AAA titles, our model successfully generates semantically and temporally aligned, high-quality full-band stereo audio, validated by both objective and human evaluations. Furthermore, our model achieves low per-frame waveform-level latency (26.3ms with the head NFE=1, 31.5ms with NFE=4) on 30FPS, 480p videos using a single H100. Demo samples are available at https://koichi-saito-sony.github.io/soundreactor/.
>
---
#### [new 012] MelCap: A Unified Single-Codebook Neural Codec for High-Fidelity Audio Compression
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频压缩任务，解决单量化器与多量化器的不足，提出MelCap统一编码器，实现高质量音频压缩与实时解码。**

- **链接: [http://arxiv.org/pdf/2510.01903v1](http://arxiv.org/pdf/2510.01903v1)**

> **作者:** Jingyi Li; Zhiyuan Zhao; Yunfei Liu; Lijian Lin; Ye Zhu; Jiahao Wu; Qiuqiang Kong; Yu Li
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Neural audio codecs have recently emerged as powerful tools for high-quality and low-bitrate audio compression, leveraging deep generative models to learn latent representations of audio signals. However, existing approaches either rely on a single quantizer that only processes speech domain, or on multiple quantizers that are not well suited for downstream tasks. To address this issue, we propose MelCap, a unified "one-codebook-for-all" neural codec that effectively handles speech, music, and general sound. By decomposing audio reconstruction into two stages, our method preserves more acoustic details than previous single-codebook approaches, while achieving performance comparable to mainstream multi-codebook methods. In the first stage, audio is transformed into mel-spectrograms, which are compressed and quantized into compact single tokens using a 2D tokenizer. A perceptual loss is further applied to mitigate the over-smoothing artifacts observed in spectrogram reconstruction. In the second stage, a Vocoder recovers waveforms from the mel discrete tokens in a single forward pass, enabling real-time decoding. Both objective and subjective evaluations demonstrate that MelCap achieves quality on comparable to state-of-the-art multi-codebook codecs, while retaining the computational simplicity of a single-codebook design, thereby providing an effective representation for downstream tasks.
>
---
#### [new 013] EvolveCaptions: Empowering DHH Users Through Real-Time Collaborative Captioning
- **分类: cs.HC; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决DHH用户ASR识别不准的问题。通过实时协作方式，让听障者在少量录音下优化模型，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2510.02181v1](http://arxiv.org/pdf/2510.02181v1)**

> **作者:** Liang-Yuan Wu; Dhruv Jain
>
> **摘要:** Automatic Speech Recognition (ASR) systems often fail to accurately transcribe speech from Deaf and Hard of Hearing (DHH) individuals, especially during real-time conversations. Existing personalization approaches typically require extensive pre-recorded data and place the burden of adaptation on the DHH speaker. We present EvolveCaptions, a real-time, collaborative ASR adaptation system that supports in-situ personalization with minimal effort. Hearing participants correct ASR errors during live conversations. Based on these corrections, the system generates short, phonetically targeted prompts for the DHH speaker to record, which are then used to fine-tune the ASR model. In a study with 12 DHH and six hearing participants, EvolveCaptions reduced Word Error Rate (WER) across all DHH users within one hour of use, using only five minutes of recording time on average. Participants described the system as intuitive, low-effort, and well-integrated into communication. These findings demonstrate the promise of collaborative, real-time ASR adaptation for more equitable communication.
>
---
#### [new 014] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems
- **分类: cs.CV; eess.SP; math.OC**

- **简介: 该论文属于图像重建任务，解决欠采样测量下的逆问题。提出NPN方法，通过神经网络对零空间进行非线性投影，提升重建精度。**

- **链接: [http://arxiv.org/pdf/2510.01608v1](http://arxiv.org/pdf/2510.01608v1)**

> **作者:** Roman Jacome; Romario Gualdrón-Hurtado; Leon Suarez; Henry Arguello
>
> **备注:** 25 pages, 12 tables, 10 figures. Accepted to NeurIPS 2025
>
> **摘要:** Imaging inverse problems aims to recover high-dimensional signals from undersampled, noisy measurements, a fundamentally ill-posed task with infinite solutions in the null-space of the sensing operator. To resolve this ambiguity, prior information is typically incorporated through handcrafted regularizers or learned models that constrain the solution space. However, these priors typically ignore the task-specific structure of that null-space. In this work, we propose \textit{Non-Linear Projections of the Null-Space} (NPN), a novel class of regularization that, instead of enforcing structural constraints in the image domain, promotes solutions that lie in a low-dimensional projection of the sensing matrix's null-space with a neural network. Our approach has two key advantages: (1) Interpretability: by focusing on the structure of the null-space, we design sensing-matrix-specific priors that capture information orthogonal to the signal components that are fundamentally blind to the sensing process. (2) Flexibility: NPN is adaptable to various inverse problems, compatible with existing reconstruction frameworks, and complementary to conventional image-domain priors. We provide theoretical guarantees on convergence and reconstruction accuracy when used within plug-and-play methods. Empirical results across diverse sensing matrices demonstrate that NPN priors consistently enhance reconstruction fidelity in various imaging inverse problems, such as compressive sensing, deblurring, super-resolution, computed tomography, and magnetic resonance imaging, with plug-and-play methods, unrolling networks, deep image prior, and diffusion models.
>
---
#### [new 015] Chain-of-Thought Reasoning in Streaming Full-Duplex End-to-End Spoken Dialogue Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决传统系统依赖VAD导致的交互不流畅问题。提出SCoT框架，实现低延迟、连续响应的双工对话。**

- **链接: [http://arxiv.org/pdf/2510.02066v1](http://arxiv.org/pdf/2510.02066v1)**

> **作者:** Siddhant Arora; Jinchuan Tian; Hayato Futami; Jiatong Shi; Yosuke Kashiwagi; Emiru Tsunoo; Shinji Watanabe
>
> **摘要:** Most end-to-end (E2E) spoken dialogue systems (SDS) rely on voice activity detection (VAD) for turn-taking, but VAD fails to distinguish between pauses and turn completions. Duplex SDS models address this by predicting output continuously, including silence tokens, thus removing the need for explicit VAD. However, they often have complex dual-channel architecture and lag behind cascaded models in semantic reasoning. To overcome these challenges, we propose SCoT: a Streaming Chain-of-Thought (CoT) framework for Duplex SDS, alternating between processing fixed-duration user input and generating responses in a blockwise manner. Using frame-level alignments, we create intermediate targets-aligned user transcripts and system responses for each block. Experiments show that our approach produces more coherent and interpretable responses than existing duplex methods while supporting lower-latency and overlapping interactions compared to turn-by-turn systems.
>
---
#### [new 016] TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling
- **分类: cs.IR; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音乐推荐任务，旨在解决LLM在推荐中功能受限的问题。通过引入工具调用机制，整合多种检索方法，提升推荐效果与多样性。**

- **链接: [http://arxiv.org/pdf/2510.01698v1](http://arxiv.org/pdf/2510.01698v1)**

> **作者:** Seungheon Doh; Keunwoo Choi; Juhan Nam
>
> **备注:** Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
>
> **摘要:** While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
>
---
#### [new 017] Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于音频视频生成任务，旨在解决多模态同步与融合问题。通过双塔结构和跨模态融合，实现自然音画合成。**

- **链接: [http://arxiv.org/pdf/2510.01284v1](http://arxiv.org/pdf/2510.01284v1)**

> **作者:** Chetwin Low; Weimin Wang; Calder Katyal
>
> **摘要:** Audio-video generation has often relied on complex multi-stage architectures or sequential synthesis of sound and visuals. We introduce Ovi, a unified paradigm for audio-video generation that models the two modalities as a single generative process. By using blockwise cross-modal fusion of twin-DiT modules, Ovi achieves natural synchronization and removes the need for separate pipelines or post hoc alignment. To facilitate fine-grained multimodal fusion modeling, we initialize an audio tower with an architecture identical to that of a strong pretrained video model. Trained from scratch on hundreds of thousands of hours of raw audio, the audio tower learns to generate realistic sound effects, as well as speech that conveys rich speaker identity and emotion. Fusion is obtained by jointly training the identical video and audio towers via blockwise exchange of timing (via scaled-RoPE embeddings) and semantics (through bidirectional cross-attention) on a vast video corpus. Our model enables cinematic storytelling with natural speech and accurate, context-matched sound effects, producing movie-grade video clips. All the demos, code and model weights are published at https://aaxwaz.github.io/Ovi
>
---
#### [new 018] Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决端到端系统易幻觉的问题。通过引入流式工具调用和检索增强生成，提升准确性和响应速度。**

- **链接: [http://arxiv.org/pdf/2510.02044v1](http://arxiv.org/pdf/2510.02044v1)**

> **作者:** Siddhant Arora; Haidar Khan; Kai Sun; Xin Luna Dong; Sajal Choudhary; Seungwhan Moon; Xinyuan Zhang; Adithya Sagar; Surya Teja Appini; Kaushik Patnaik; Sanat Sharma; Shinji Watanabe; Anuj Kumar; Ahmed Aly; Yue Liu; Florian Metze; Zhaojiang Lin
>
> **摘要:** End-to-end speech-in speech-out dialogue systems are emerging as a powerful alternative to traditional ASR-LLM-TTS pipelines, generating more natural, expressive responses with significantly lower latency. However, these systems remain prone to hallucinations due to limited factual grounding. While text-based dialogue systems address this challenge by integrating tools such as web search and knowledge graph APIs, we introduce the first approach to extend tool use directly into speech-in speech-out systems. A key challenge is that tool integration substantially increases response latency, disrupting conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented Generation (Streaming RAG), a novel framework that reduces user-perceived latency by predicting tool queries in parallel with user speech, even before the user finishes speaking. Specifically, we develop a post-training pipeline that teaches the model when to issue tool calls during ongoing speech and how to generate spoken summaries that fuse audio queries with retrieved text results, thereby improving both accuracy and responsiveness. To evaluate our approach, we construct AudioCRAG, a benchmark created by converting queries from the publicly available CRAG dataset into speech form. Experimental results demonstrate that our streaming RAG approach increases QA accuracy by up to 200% relative (from 11.1% to 34.2% absolute) and further enhances user experience by reducing tool use latency by 20%. Importantly, our streaming RAG approach is modality-agnostic and can be applied equally to typed input, paving the way for more agentic, real-time AI assistants.
>
---
#### [new 019] SLAP: Learning Speaker and Health-Related Representations from Natural Language Supervision
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SLAP模型，用于从自然语言监督中学习说话人和健康相关表示。任务是语音的零样本和分布外泛化，解决传统模型不足的问题。通过对比学习实现语音与文本对齐。**

- **链接: [http://arxiv.org/pdf/2510.01860v1](http://arxiv.org/pdf/2510.01860v1)**

> **作者:** Angelika Ando; Auguste Crabeil; Adrien Lesage; Rachid Riad
>
> **摘要:** Speech encodes paralinguistic information such as demographics, voice quality, and health. Yet no audio foundation model supports zero-shot or out-of-distribution (OOD) generalization to these tasks. We introduce SLAP (Speaker contrastive Language-Audio Pretraining), the first model aligning speech with natural language descriptions of speaker and health metadata through contrastive learning. SLAP combines a Vision Transformer audio encoder with text encoders, trained on more than 3400 hours across 9 datasets with diverse speaker annotations. We evaluated on 38 binary classification tasks spanning demographics, voice characteristics, and clinical assessments across 14 datasets in 7 languages. SLAP achieves 62.9% average F1 in zero-shot evaluation, a 48% relative improvement over CLAP (42.4%), while demonstrating strong OOD generalization to unseen languages and clinical populations. When fine-tuned with linear probing, SLAP reaches 69.3% F1 overall and achieves best-in-class performance on health tasks (57.9% F1), surpassing larger foundation models.
>
---
#### [new 020] Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音大模型性别偏见评估任务，旨在检验MCQA基准是否具备跨任务泛化能力。通过微调模型并测试其在不同任务中的表现，发现现有基准难以预测其他任务结果。**

- **链接: [http://arxiv.org/pdf/2510.01254v1](http://arxiv.org/pdf/2510.01254v1)**

> **作者:** Shree Harsha Bokkahalli Satish; Gustav Eje Henter; Éva Székely
>
> **备注:** 5 pages, 2 Figures, Submitted to IEEE ICASSP 2026
>
> **摘要:** Recent work in benchmarking bias and fairness in speech large language models (SpeechLLMs) has relied heavily on multiple-choice question answering (MCQA) formats. The model is tasked to choose between stereotypical, anti-stereotypical, or neutral/irrelevant answers given an input speech prompt and an optional text prompt. Such MCQA benchmarks implicitly assume that model performance is consistent across other MCQA tasks, voices, and other task formats such as more realistic, long-form evaluations. In this paper, we probe that assumption. We fine-tune three SpeechLLMs using LoRA adapters to induce specific MCQA behaviours: preference for stereotypical, anti-stereotypical, or neutral/uncertain answers. We then evaluate whether these behaviours generalise to another, distinct MCQA benchmark, and more critically to long-form, creative generation tasks. Our results show that performance on MCQA bias benchmarks fails to reliably predict performances across other MCQA benchmarks, and more importantly across long-form tasks. We conclude that current MCQA bias benchmarks show limited evidence of cross-task generalisation in the speech domain, and also propose an evaluation suite for measuring behaviour transferability in future models and benchmarks.
>
---
#### [new 021] Mirage Fools the Ear, Mute Hides the Truth: Precise Targeted Adversarial Attacks on Polyphonic Sound Event Detection Systems
- **分类: cs.CR; cs.SD**

- **简介: 该论文属于声事件检测任务，解决对抗攻击中精度不足的问题。提出M2A框架，通过保留非目标区域输出实现精准攻击，并引入EP指标提升效果与精度。**

- **链接: [http://arxiv.org/pdf/2510.02158v1](http://arxiv.org/pdf/2510.02158v1)**

> **作者:** Junjie Su; Weifei Jin; Yuxin Cao; Derui Wang; Kai Ye; Jie Hao
>
> **摘要:** Sound Event Detection (SED) systems are increasingly deployed in safety-critical applications such as industrial monitoring and audio surveillance. However, their robustness against adversarial attacks has not been well explored. Existing audio adversarial attacks targeting SED systems, which incorporate both detection and localization capabilities, often lack effectiveness due to SED's strong contextual dependencies or lack precision by focusing solely on misclassifying the target region as the target event, inadvertently affecting non-target regions. To address these challenges, we propose the Mirage and Mute Attack (M2A) framework, which is designed for targeted adversarial attacks on polyphonic SED systems. In our optimization process, we impose specific constraints on the non-target output, which we refer to as preservation loss, ensuring that our attack does not alter the model outputs for non-target region, thus achieving precise attacks. Furthermore, we introduce a novel evaluation metric Editing Precison (EP) that balances effectiveness and precision, enabling our method to simultaneously enhance both. Comprehensive experiments show that M2A achieves 94.56% and 99.11% EP on two state-of-the-art SED models, demonstrating that the framework is sufficiently effective while significantly enhancing attack precision.
>
---
#### [new 022] Automated Defect Detection for Mass-Produced Electronic Components Based on YOLO Object Detection Models
- **分类: cs.CV; cs.AI; cs.LG; eess.SP; 68T07, 68U10; I.4.8; I.2.10**

- **简介: 该论文属于工业缺陷检测任务，解决传统检测效率低、依赖人工的问题。通过YOLO模型与ConSinGAN数据增强，实现DIP组件的自动化缺陷识别。**

- **链接: [http://arxiv.org/pdf/2510.01914v1](http://arxiv.org/pdf/2510.01914v1)**

> **作者:** Wei-Lung Mao; Chun-Chi Wang; Po-Heng Chou; Yen-Ting Liu
>
> **备注:** 12 pages, 16 figures, 7 tables, and published in IEEE Sensors Journal
>
> **摘要:** Since the defect detection of conventional industry components is time-consuming and labor-intensive, it leads to a significant burden on quality inspection personnel and makes it difficult to manage product quality. In this paper, we propose an automated defect detection system for the dual in-line package (DIP) that is widely used in industry, using digital camera optics and a deep learning (DL)-based model. The two most common defect categories of DIP are examined: (1) surface defects, and (2) pin-leg defects. However, the lack of defective component images leads to a challenge for detection tasks. To solve this problem, the ConSinGAN is used to generate a suitable-sized dataset for training and testing. Four varieties of the YOLO model are investigated (v3, v4, v7, and v9), both in isolation and with the ConSinGAN augmentation. The proposed YOLOv7 with ConSinGAN is superior to the other YOLO versions in accuracy of 95.50\%, detection time of 285 ms, and is far superior to threshold-based approaches. In addition, the supervisory control and data acquisition (SCADA) system is developed, and the associated sensor architecture is described. The proposed automated defect detection can be easily established with numerous types of defects or insufficient defect data.
>
---
## 更新

#### [replaced 001] XPPG-PCA: Reference-free automatic speech severity evaluation with principal components
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.00657v2](http://arxiv.org/pdf/2510.00657v2)**

> **作者:** Bence Mark Halpern; Thomas B. Tienkamp; Teja Rebernik; Rob J. J. H. van Son; Sebastiaan A. H. J. de Visscher; Max J. H. Witjes; Defne Abur; Tomoki Toda
>
> **备注:** 14 pages, 4 figures. Author Accepted Manuscript version of the IEEE Selected Topics in Signal Processing with the same title
>
> **摘要:** Reliably evaluating the severity of a speech pathology is crucial in healthcare. However, the current reliance on expert evaluations by speech-language pathologists presents several challenges: while their assessments are highly skilled, they are also subjective, time-consuming, and costly, which can limit the reproducibility of clinical studies and place a strain on healthcare resources. While automated methods exist, they have significant drawbacks. Reference-based approaches require transcriptions or healthy speech samples, restricting them to read speech and limiting their applicability. Existing reference-free methods are also flawed; supervised models often learn spurious shortcuts from data, while handcrafted features are often unreliable and restricted to specific speech tasks. This paper introduces XPPG-PCA (x-vector phonetic posteriorgram principal component analysis), a novel, unsupervised, reference-free method for speech severity evaluation. Using three Dutch oral cancer datasets, we demonstrate that XPPG-PCA performs comparably to, or exceeds established reference-based methods. Our experiments confirm its robustness against data shortcuts and noise, showing its potential for real-world clinical use. Taken together, our results show that XPPG-PCA provides a robust, generalizable solution for the objective assessment of speech pathology, with the potential to significantly improve the efficiency and reliability of clinical evaluations across a range of disorders. An open-source implementation is available.
>
---
#### [replaced 002] ARIONet: An Advanced Self-supervised Contrastive Representation Network for Birdsong Classification and Future Frame Prediction
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.00522v2](http://arxiv.org/pdf/2510.00522v2)**

> **作者:** Md. Abdur Rahman; Selvarajah Thuseethan; Kheng Cher Yeo; Reem E. Mohamed; Sami Azam
>
> **摘要:** Automated birdsong classification is essential for advancing ecological monitoring and biodiversity studies. Despite recent progress, existing methods often depend heavily on labeled data, use limited feature representations, and overlook temporal dynamics essential for accurate species identification. In this work, we propose a self-supervised contrastive network, ARIONet (Acoustic Representation for Interframe Objective Network), that jointly optimizes contrastive classification and future frame prediction using augmented audio representations. The model simultaneously integrates multiple complementary audio features within a transformer-based encoder model. Our framework is designed with two key objectives: (1) to learn discriminative species-specific representations for contrastive learning through maximizing similarity between augmented views of the same audio segment while pushing apart different samples, and (2) to model temporal dynamics by predicting future audio frames, both without requiring large-scale annotations. We validate our framework on four diverse birdsong datasets, including the British Birdsong Dataset, Bird Song Dataset, and two extended Xeno-Canto subsets (A-M and N-Z). Our method consistently outperforms existing baselines and achieves classification accuracies of 98.41%, 93.07%, 91.89%, and 91.58%, and F1-scores of 97.84%, 94.10%, 91.29%, and 90.94%, respectively. Furthermore, it demonstrates low mean absolute errors and high cosine similarity, up to 95%, in future frame prediction tasks. Extensive experiments further confirm the effectiveness of our self-supervised learning strategy in capturing complex acoustic patterns and temporal dependencies, as well as its potential for real-world applicability in ecological conservation and monitoring.
>
---
#### [replaced 003] MambAttention: Mamba with Multi-Head Attention for Generalizable Single-Channel Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.00966v2](http://arxiv.org/pdf/2507.00966v2)**

> **作者:** Nikolai Lund Kühne; Jesper Jensen; Jan Østergaard; Zheng-Hua Tan
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing for possible publication
>
> **摘要:** With the advent of new sequence models like Mamba and xLSTM, several studies have shown that these models match or outperform state-of-the-art models in single-channel speech enhancement, automatic speech recognition, and self-supervised audio representation learning. However, prior research has demonstrated that sequence models like LSTM and Mamba tend to overfit to the training set. To address this issue, previous works have shown that adding self-attention to LSTMs substantially improves generalization performance for single-channel speech enhancement. Nevertheless, neither the concept of hybrid Mamba and time-frequency attention models nor their generalization performance have been explored for speech enhancement. In this paper, we propose a novel hybrid architecture, MambAttention, which combines Mamba and shared time- and frequency-multi-head attention modules for generalizable single-channel speech enhancement. To train our model, we introduce VoiceBank+Demand Extended (VB-DemandEx), a dataset inspired by VoiceBank+Demand but with more challenging noise types and lower signal-to-noise ratios. Trained on VB-DemandEx, our proposed MambAttention model significantly outperforms existing state-of-the-art LSTM-, xLSTM-, Mamba-, and Conformer-based systems of similar complexity across all reported metrics on two out-of-domain datasets: DNS 2020 and EARS-WHAM_v2, while matching their performance on the in-domain dataset VB-DemandEx. Ablation studies highlight the role of weight sharing between the time- and frequency-multi-head attention modules for generalization performance. Finally, we explore integrating the shared time- and frequency-multi-head attention modules with LSTM and xLSTM, which yields a notable performance improvement on the out-of-domain datasets. However, our MambAttention model remains superior on both out-of-domain datasets across all reported evaluation metrics.
>
---
#### [replaced 004] Unmute the Patch Tokens: Rethinking Probing in Multi-Label Audio Classification
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.24901v2](http://arxiv.org/pdf/2509.24901v2)**

> **作者:** Lukas Rauch; René Heinrich; Houtan Ghaffari; Lukas Miklautz; Ilyass Moummad; Bernhard Sick; Christoph Scholz
>
> **备注:** Currently under review @ICLR2026
>
> **摘要:** Although probing frozen models has become a standard evaluation paradigm, self-supervised learning in audio defaults to fine-tuning. A key reason is that global pooling creates an information bottleneck causing linear probes to misrepresent the embedding quality: The $\texttt{cls}$-token discards crucial token information about dispersed, localized events in multi-label audio. This weakness is rooted in the mismatch between the pretraining objective (operating globally) and the downstream task (localized events). Across a comprehensive benchmark of 13 datasets and 6 spectrogram-based encoders, we first investigate the global pooling bottleneck. We then introduce binarized prototypical probes: a lightweight and simple pooling method that learns prototypes to perform class-wise information aggregation. Despite its simplicity, our method notably outperforms linear and attentive probing. Our work establishes probing as a competitive and efficient paradigm for evaluating audio SSL models, challenging the reliance on costly fine-tuning.
>
---
#### [replaced 005] Audio-Enhanced Vision-Language Modeling with Latent Space Broadening for High Quality Data Expansion
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.17551v2](http://arxiv.org/pdf/2503.17551v2)**

> **作者:** Yu Sun; Yin Li; Ruixiao Sun; Chunhui Liu; Fangming Zhou; Ze Jin; Linjie Wang; Xiang Shen; Zhuolin Hao; Hongyu Xiong
>
> **摘要:** Transformer-based multimodal models are widely used in industrial-scale recommendation, search, and advertising systems for content understanding and relevance ranking. Enhancing labeled training data quality and cross-modal fusion significantly improves model performance, influencing key metrics such as quality view rates and ad revenue. High-quality annotations are crucial for advancing content modeling, yet traditional statistical-based active learning (AL) methods face limitations: they struggle to detect overconfident misclassifications and are less effective in distinguishing semantically similar items in deep neural networks. Additionally, audio information plays an increasing role, especially in short-video platforms, yet most pre-trained multimodal architectures primarily focus on text and images. While training from scratch across all three modalities is possible, it sacrifices the benefits of leveraging existing pre-trained visual-language (VL) and audio models. To address these challenges, we propose kNN-based Latent Space Broadening (LSB) to enhance AL efficiency and Vision-Language Modeling with Audio Enhancement (VLMAE), a mid-fusion approach integrating audio into VL models. This system deployed in production systems, leading to significant business gains.
>
---
#### [replaced 006] PAGURI: a user experience study of creative interaction with text-to-music models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.04333v3](http://arxiv.org/pdf/2407.04333v3)**

> **作者:** Francesca Ronchini; Luca Comanducci; Gabriele Perego; Fabio Antonacci
>
> **摘要:** In recent years, text-to-music models have been the biggest breakthrough in automatic music generation. While they are unquestionably a showcase of technological progress, it is not clear yet how they can be realistically integrated into the artistic practice of musicians and music practitioners. This paper aims to address this question via Prompt Audio Generation User Research Investigation (PAGURI), a user experience study where we leverage recent text-to-music developments to study how musicians and practitioners interact with these systems, evaluating their satisfaction levels. We developed an online tool through which users can generate music samples and/or apply recently proposed personalization techniques based on fine-tuning to allow the text-to-music model to generate sounds closer to their needs and preferences. Using semi-structured interviews, we analyzed different aspects related to how participants interacted with the proposed tool to understand the current effectiveness and limitations of text-to-music models in enhancing users' creativity. Our research centers on user experiences to uncover insights that can guide the future development of TTM models and their role in AI-driven music creation. Additionally, they offered insightful perspectives on potential system improvements and their integration into their music practices. The results obtained through the study reveal the pros and cons of the use of TTMs for creative endeavors. Participants recognized the system's creative potential and appreciated the usefulness of its personalization features. However, they also identified several challenges that must be addressed before TTMs are ready for real-world music creation, particularly issues of prompt ambiguity, limited controllability, and integration into existing workflows.
>
---
#### [replaced 007] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.00981v2](http://arxiv.org/pdf/2510.00981v2)**

> **作者:** Jiaqi Li; Yao Qian; Yuxuan Hu; Leying Zhang; Xiaofei Wang; Heng Lu; Manthan Thakker; Jinyu Li; Sheng Zhao; Zhizheng Wu
>
> **摘要:** Neural audio codecs are foundational to speech language models. It is expected to have a low frame rate and decoupled semantic and acoustic information. A lower frame rate codec can reduce the computational cost of speech language models by shortening the sequence length. Recent studies have developed 12.5Hz low-frame-rate audio codecs, but even lower frame rate codecs remain underexplored. We find that a major challenge for very low frame rate tokens is missing semantic information. This paper introduces FlexiCodec to address this limitation. FlexiCodec improves semantic preservation with a dynamic frame rate approach and introduces a novel architecture featuring an ASR feature-assisted dual stream encoding and Transformer bottlenecks. With dynamic frame rates, it uses less frames at information-sparse regions through adaptively merging semantically similar frames. A dynamic frame rate also allows FlexiCodec to support inference-time controllable frame rates between 3Hz and 12.5Hz. Experiments on 6.25Hz, 8.3Hz and 12.5Hz average frame rates confirm that FlexiCodec excels over baseline systems in semantic information preservation and delivers a high audio reconstruction quality. We also validate the effectiveness of FlexiCodec in language model-based TTS. Demos are available at: https://flexicodec.github.io
>
---
#### [replaced 008] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.14270v2](http://arxiv.org/pdf/2509.14270v2)**

> **作者:** Karan Dua; Puneet Mittal; Ranjeet Gupta; Hitesh Laxmichand Patel
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** High-quality Text-to-Speech (TTS) model training requires extensive and diverse text and speech data. It is challenging to procure such data from real sources due to issues of domain specificity, licensing, and scalability. Large language models (LLMs) can certainly generate textual data, but they create repetitive text with insufficient variation in the prompt during the generation process. Another important aspect in TTS training data is text normalization. Tools for normalization might occasionally introduce anomalies or overlook valuable patterns, and thus impact data quality. Furthermore, it is also impractical to rely on voice artists for large scale speech recording in commercial TTS systems with standardized voices. To address these challenges, we propose SpeechWeave, a synthetic speech data generation pipeline that is capable of automating the generation of multilingual, domain-specific datasets for training TTS models. Our experiments reveal that our pipeline generates data that is 10-48% more diverse than the baseline across various linguistic and phonetic metrics, along with speaker-standardized speech audio while generating approximately 97% correctly normalized text. Our approach enables scalable, high-quality data generation for TTS training, improving diversity, normalization, and voice consistency in the generated datasets.
>
---
#### [replaced 009] NeRAF: 3D Scene Infused Neural Radiance and Acoustic Fields
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.18213v4](http://arxiv.org/pdf/2405.18213v4)**

> **作者:** Amandine Brunetto; Sascha Hornauer; Fabien Moutarde
>
> **备注:** ICLR 2025 (Poster). Camera ready version. Project Page: https://amandinebtto.github.io/NeRAF; 24 pages, 13 figures
>
> **摘要:** Sound plays a major role in human perception. Along with vision, it provides essential information for understanding our surroundings. Despite advances in neural implicit representations, learning acoustics that align with visual scenes remains a challenge. We propose NeRAF, a method that jointly learns acoustic and radiance fields. NeRAF synthesizes both novel views and spatialized room impulse responses (RIR) at new positions by conditioning the acoustic field on 3D scene geometric and appearance priors from the radiance field. The generated RIR can be applied to auralize any audio signal. Each modality can be rendered independently and at spatially distinct positions, offering greater versatility. We demonstrate that NeRAF generates high-quality audio on SoundSpaces and RAF datasets, achieving significant performance improvements over prior methods while being more data-efficient. Additionally, NeRAF enhances novel view synthesis of complex scenes trained with sparse data through cross-modal learning. NeRAF is designed as a Nerfstudio module, providing convenient access to realistic audio-visual generation.
>
---
