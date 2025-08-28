# 音频 cs.SD;  eess.SP

- **最新发布 8 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] MQAD: A Large-Scale Question Answering Dataset for Training Music Large Language Models
- **分类: cs.SD**

- **简介: 论文提出MQAD数据集，用于训练音乐大语言模型，解决音乐数据稀缺问题，基于MSD整合MIR与LLM生成300万+问答对，提升音乐音频描述效果。**

- **链接: [http://arxiv.org/pdf/2508.19514v1](http://arxiv.org/pdf/2508.19514v1)**

> **作者:** Zhihao Ouyang; Ju-Chiang Wang; Daiyu Zhang; Bin Chen; Shangjie Li; Quan Lin
>
> **摘要:** Question-answering (QA) is a natural approach for humans to understand a piece of music audio. However, for machines, accessing a large-scale dataset covering diverse aspects of music is crucial, yet challenging, due to the scarcity of publicly available music data of this type. This paper introduces MQAD, a music QA dataset built on the Million Song Dataset (MSD), encompassing a rich array of musical features, including beat, chord, key, structure, instrument, and genre -- across 270,000 tracks, featuring nearly 3 million diverse questions and captions. MQAD distinguishes itself by offering detailed time-varying musical information such as chords and sections, enabling exploration into the inherent structure of music within a song. To compile MQAD, our methodology leverages specialized Music Information Retrieval (MIR) models to extract higher-level musical features and Large Language Models (LLMs) to generate natural language QA pairs. Then, we leverage a multimodal LLM that integrates the LLaMA2 and Whisper architectures, along with novel subjective metrics to assess the performance of MQAD. In experiments, our model trained on MQAD demonstrates advancements over conventional music audio captioning approaches. The dataset and code are available at https://github.com/oyzh888/MQAD.
>
---
#### [new 002] CompLex: Music Theory Lexicon Constructed by Autonomous Agents for Automatic Music Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文旨在通过自主代理构建音乐理论词典CompLex，解决音乐生成中数据不足和手动工作量大的问题，提升文本到音乐生成模型的性能。**

- **链接: [http://arxiv.org/pdf/2508.19603v1](http://arxiv.org/pdf/2508.19603v1)**

> **作者:** Zhejing Hu; Yan Liu; Gong Chen; Bruce X. B. Yu
>
> **摘要:** Generative artificial intelligence in music has made significant strides, yet it still falls short of the substantial achievements seen in natural language processing, primarily due to the limited availability of music data. Knowledge-informed approaches have been shown to enhance the performance of music generation models, even when only a few pieces of musical knowledge are integrated. This paper seeks to leverage comprehensive music theory in AI-driven music generation tasks, such as algorithmic composition and style transfer, which traditionally require significant manual effort with existing techniques. We introduce a novel automatic music lexicon construction model that generates a lexicon, named CompLex, comprising 37,432 items derived from just 9 manually input category keywords and 5 sentence prompt templates. A new multi-agent algorithm is proposed to automatically detect and mitigate hallucinations. CompLex demonstrates impressive performance improvements across three state-of-the-art text-to-music generation models, encompassing both symbolic and audio-based methods. Furthermore, we evaluate CompLex in terms of completeness, accuracy, non-redundancy, and executability, confirming that it possesses the key characteristics of an effective lexicon.
>
---
#### [new 003] MuSpike: A Benchmark and Evaluation Framework for Symbolic Music Generation with Spiking Neural Networks
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对脉冲神经网络（SNN）在符号音乐生成中的评估不足问题，提出MuSpike框架，系统评估五类SNN模型在五组数据集上的表现，结合客观指标与大规模听测，揭示模型差异及主观感知的重要性。**

- **链接: [http://arxiv.org/pdf/2508.19251v1](http://arxiv.org/pdf/2508.19251v1)**

> **作者:** Qian Liang; Menghaoran Tang; Yi Zeng
>
> **摘要:** Symbolic music generation has seen rapid progress with artificial neural networks, yet remains underexplored in the biologically plausible domain of spiking neural networks (SNNs), where both standardized benchmarks and comprehensive evaluation methods are lacking. To address this gap, we introduce MuSpike, a unified benchmark and evaluation framework that systematically assesses five representative SNN architectures (SNN-CNN, SNN-RNN, SNN-LSTM, SNN-GAN and SNN-Transformer) across five typical datasets, covering tonal, structural, emotional, and stylistic variations. MuSpike emphasizes comprehensive evaluation, combining established objective metrics with a large-scale listening study. We propose new subjective metrics, targeting musical impression, autobiographical association, and personal preference, that capture perceptual dimensions often overlooked in prior work. Results reveal that (1) different SNN models exhibit distinct strengths across evaluation dimensions; (2) participants with different musical backgrounds exhibit diverse perceptual patterns, with experts showing greater tolerance toward AI-composed music; and (3) a noticeable misalignment exists between objective and subjective evaluations, highlighting the limitations of purely statistical metrics and underscoring the value of human perceptual judgment in assessing musical quality. MuSpike provides the first systematic benchmark and systemic evaluation framework for SNN models in symbolic music generation, establishing a solid foundation for future research into biologically plausible and cognitively grounded music generation.
>
---
#### [new 004] The IRMA Dataset: A Structured Audio-MIDI Corpus for Iranian Classical Music
- **分类: cs.SD; cs.DL**

- **简介: 论文构建了一个结构化音频- MIDI语料库，用于研究伊朗古典音乐的radif（结构化复调旋律）。通过多阶段标注与对齐，整合MIDI、音频、乐谱及理论数据，支持音乐分析、AI任务和文化保存。**

- **链接: [http://arxiv.org/pdf/2508.19876v1](http://arxiv.org/pdf/2508.19876v1)**

> **作者:** Sepideh Shafiei; Shapour Hakam
>
> **摘要:** We present the IRMA Dataset (Iranian Radif MIDI Audio), a multi-level, open-access corpus designed for the computational study of Iranian classical music, with a particular emphasis on the radif, a structured repertoire of modal-melodic units central to pedagogy and performance. The dataset combines symbolic MIDI representations, phrase-level audio-MIDI alignment, musicological transcriptions in PDF format, and comparative tables of theoretical information curated from a range of performers and scholars. We outline the multi-phase construction process, including segment annotation, alignment methods, and a structured system of identifier codes to reference individual musical units. The current release includes the complete radif of Karimi; MIDI files and metadata from Mirza Abdollah's radif; selected segments from the vocal radif of Davami, as transcribed by Payvar and Fereyduni; and a dedicated section featuring audio-MIDI examples of tahrir ornamentation performed by prominent 20th-century vocalists. While the symbolic and analytical components are released under an open-access license (CC BY-NC 4.0), some referenced audio recordings and third-party transcriptions are cited using discographic information to enable users to locate the original materials independently, pending copyright permission. Serving both as a scholarly archive and a resource for computational analysis, this dataset supports applications in ethnomusicology, pedagogy, symbolic audio research, cultural heritage preservation, and AI-driven tasks such as automatic transcription and music generation. We welcome collaboration and feedback to support its ongoing refinement and broader integration into musicological and machine learning workflows.
>
---
#### [new 005] Infant Cry Detection In Noisy Environment Using Blueprint Separable Convolutions and Time-Frequency Recurrent Neural Network
- **分类: cs.SD**

- **简介: 论文针对噪声环境下婴儿哭声检测任务，提出融合蓝图可分离卷积与时频RNN的轻量模型，结合注意力机制提升鲁棒性，通过多数据集训练实现高准确率与低复杂度。**

- **链接: [http://arxiv.org/pdf/2508.19308v1](http://arxiv.org/pdf/2508.19308v1)**

> **作者:** Haolin Yu; Yanxiong Li
>
> **摘要:** Infant cry detection is a crucial component of baby care system. In this paper, we propose a lightweight and robust method for infant cry detection. The method leverages blueprint separable convolutions to reduce computational complexity, and a time-frequency recurrent neural network for adaptive denoising. The overall framework of the method is structured as a multi-scale convolutional recurrent neural network, which is enhanced by efficient spatial attention mechanism and contrast-aware channel attention module, and acquire local and global information from the input feature of log Mel-spectrogram. Multiple public datasets are adopted to create a diverse and representative dataset, and environmental corruption techniques are used to generate the noisy samples encountered in real-world scenarios. Results show that our method exceeds many state-of-the-art methods in accuracy, F1-score, and complexity under various signal-to-noise ratio conditions. The code is at https://github.com/fhfjsd1/ICD_MMSP.
>
---
#### [new 006] Beat-Based Rhythm Quantization of MIDI Performances
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 论文提出基于Transformer的节奏量化模型，利用节拍与重拍信息将MIDI表演转为对齐的乐谱。通过预处理和优化模型结构，在MUSTER指标上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2508.19262v1](http://arxiv.org/pdf/2508.19262v1)**

> **作者:** Maximilian Wachter; Sebastian Murgul; Michael Heizmann
>
> **备注:** Accepted to the Late Breaking Demo Papers of the 1st AES International Conference on Artificial Intelligence and Machine Learning for Audio (AIMLA LBDP), 2025
>
> **摘要:** We propose a transformer-based rhythm quantization model that incorporates beat and downbeat information to quantize MIDI performances into metrically-aligned, human-readable scores. We propose a beat-based preprocessing method that transfers score and performance data into a unified token representation. We optimize our model architecture and data representation and train on piano and guitar performances. Our model exceeds state-of-the-art performance based on the MUSTER metric.
>
---
#### [new 007] FLASepformer: Efficient Speech Separation with Gated Focused Linear Attention Transformer
- **分类: eess.AS; cs.SD**

- **简介: 论文针对语音分离中长序列处理的计算与内存瓶颈，提出FLASepformer模型，通过线性注意力机制和门控模块提升效率，实验表明其在多个数据集上实现更优速度与更低内存消耗。**

- **链接: [http://arxiv.org/pdf/2508.19528v1](http://arxiv.org/pdf/2508.19528v1)**

> **作者:** Haoxu Wang; Yiheng Jiang; Gang Qiao; Pengteng Shi; Biao Tian
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Speech separation always faces the challenge of handling prolonged time sequences. Past methods try to reduce sequence lengths and use the Transformer to capture global information. However, due to the quadratic time complexity of the attention module, memory usage and inference time still increase significantly with longer segments. To tackle this, we introduce Focused Linear Attention and build FLASepformer with linear complexity for efficient speech separation. Inspired by SepReformer and TF-Locoformer, we have two variants: FLA-SepReformer and FLA-TFLocoformer. We also add a new Gated module to improve performance further. Experimental results on various datasets show that FLASepformer matches state-of-the-art performance with less memory consumption and faster inference. FLA-SepReformer-T/B/L increases speed by 2.29x, 1.91x, and 1.49x, with 15.8%, 20.9%, and 31.9% GPU memory usage, proving our model's effectiveness.
>
---
#### [new 008] AudioStory: Generating Long-Form Narrative Audio with Large Language Models
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 论文旨在解决长篇叙事音频生成中的时间一致性和逻辑连贯性问题，提出AudioStory框架，整合大语言模型与文本到音频系统，通过解耦协作机制和端到端训练生成结构化叙事音频，并建立基准数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2508.20088v1](http://arxiv.org/pdf/2508.20088v1)**

> **作者:** Yuxin Guo; Teng Wang; Yuying Ge; Shijie Ma; Yixiao Ge; Wei Zou; Ying Shan
>
> **摘要:** Recent advances in text-to-audio (TTA) generation excel at synthesizing short audio clips but struggle with long-form narrative audio, which requires temporal coherence and compositional reasoning. To address this gap, we propose AudioStory, a unified framework that integrates large language models (LLMs) with TTA systems to generate structured, long-form audio narratives. AudioStory possesses strong instruction-following reasoning generation capabilities. It employs LLMs to decompose complex narrative queries into temporally ordered sub-tasks with contextual cues, enabling coherent scene transitions and emotional tone consistency. AudioStory has two appealing features: (1) Decoupled bridging mechanism: AudioStory disentangles LLM-diffuser collaboration into two specialized components, i.e., a bridging query for intra-event semantic alignment and a residual query for cross-event coherence preservation. (2) End-to-end training: By unifying instruction comprehension and audio generation within a single end-to-end framework, AudioStory eliminates the need for modular training pipelines while enhancing synergy between components. Furthermore, we establish a benchmark AudioStory-10K, encompassing diverse domains such as animated soundscapes and natural sound narratives. Extensive experiments show the superiority of AudioStory on both single-audio generation and narrative audio generation, surpassing prior TTA baselines in both instruction-following ability and audio fidelity. Our code is available at https://github.com/TencentARC/AudioStory
>
---
## 更新

#### [replaced 001] Analysis and Synthesis Denoisers for Forward-Backward Plug-and-Play Algorithms
- **分类: math.OC; cs.CV; eess.IV; eess.SP; 90C59, 65K10, 68T07, 68U10, 94A08**

- **链接: [http://arxiv.org/pdf/2411.13276v3](http://arxiv.org/pdf/2411.13276v3)**

> **作者:** Matthieu Kowalski; Benoît Malézieux; Thomas Moreau; Audrey Repetti
>
> **摘要:** In this work we study the behavior of the forward-backward (FB) algorithm when the proximity operator is replaced by a sub-iterative procedure to approximate a Gaussian denoiser, in a Plug-and-Play (PnP) fashion. In particular, we consider both analysis and synthesis Gaussian denoisers within a dictionary framework, obtained by unrolling dual-FB iterations or FB iterations, respectively. We analyze the associated minimization problems as well as the asymptotic behavior of the resulting FB-PnP iterations. In particular, we show that the synthesis Gaussian denoising problem can be viewed as a proximity operator. For each case, analysis and synthesis, we show that the FB-PnP algorithms solve the same problem whether we use only one or an infinite number of sub-iteration to solve the denoising problem at each iteration. To this aim, we show that each "one sub-iteration" strategy within the FB-PnP can be interpreted as a primal-dual algorithm when a warm-restart strategy is used. We further present similar results when using a Moreau-Yosida smoothing of the global problem, for an arbitrary number of sub-iterations. Finally, we provide numerical simulations to illustrate our theoretical results. In particular we first consider a toy compressive sensing example, as well as an image restoration problem in a deep dictionary framework.
>
---
#### [replaced 002] Step-Audio 2 Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16632v3](http://arxiv.org/pdf/2507.16632v3)**

> **作者:** Boyong Wu; Chao Yan; Chen Hu; Cheng Yi; Chengli Feng; Fei Tian; Feiyu Shen; Gang Yu; Haoyang Zhang; Jingbei Li; Mingrui Chen; Peng Liu; Wang You; Xiangyu Tony Zhang; Xingyuan Li; Xuerui Yang; Yayue Deng; Yechang Huang; Yuxin Li; Yuxin Zhang; Zhao You; Brian Li; Changyi Wan; Hanpeng Hu; Jiangjie Zhen; Siyu Chen; Song Yuan; Xuelin Zhang; Yimin Jiang; Yu Zhou; Yuxiang Yang; Bingxin Li; Buyun Ma; Changhe Song; Dongqing Pang; Guoqiang Hu; Haiyang Sun; Kang An; Na Wang; Shuli Gao; Wei Ji; Wen Li; Wen Sun; Xuan Wen; Yong Ren; Yuankai Ma; Yufan Lu; Bin Wang; Bo Li; Changxin Miao; Che Liu; Chen Xu; Dapeng Shi; Dingyuan Hu; Donghang Wu; Enle Liu; Guanzhe Huang; Gulin Yan; Han Zhang; Hao Nie; Haonan Jia; Hongyu Zhou; Jianjian Sun; Jiaoren Wu; Jie Wu; Jie Yang; Jin Yang; Junzhe Lin; Kaixiang Li; Lei Yang; Liying Shi; Li Zhou; Longlong Gu; Ming Li; Mingliang Li; Mingxiao Li; Nan Wu; Qi Han; Qinyuan Tan; Shaoliang Pang; Shengjie Fan; Siqi Liu; Tiancheng Cao; Wanying Lu; Wenqing He; Wuxun Xie; Xu Zhao; Xueqi Li; Yanbo Yu; Yang Yang; Yi Liu; Yifan Lu; Yilei Wang; Yuanhao Ding; Yuanwei Liang; Yuanwei Lu; Yuchu Luo; Yuhe Yin; Yumeng Zhan; Yuxiang Zhang; Zidong Yang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **备注:** v3: Added introduction and evaluation results of Step-Audio 2 mini
>
> **摘要:** This paper presents Step-Audio 2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.
>
---
#### [replaced 003] Vocoder-Projected Feature Discriminator
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; stat.ML**

- **链接: [http://arxiv.org/pdf/2508.17874v2](http://arxiv.org/pdf/2508.17874v2)**

> **作者:** Takuhiro Kaneko; Hirokazu Kameoka; Kou Tanaka; Yuto Kondo
>
> **备注:** Accepted to Interspeech 2025. Project page: https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/vpfd/
>
> **摘要:** In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively.
>
---
#### [replaced 004] mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08400v3](http://arxiv.org/pdf/2506.08400v3)**

> **作者:** Luel Hagos Beyene; Vivek Verma; Min Ma; Jesujoba O. Alabi; Fabian David Schmidt; Joyce Nakatumba-Nabende; David Ifeoluwa Adelani
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
>
---
#### [replaced 005] Human-Inspired Computing for Robust and Efficient Audio-Visual Speech Recognition
- **分类: cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.16564v2](http://arxiv.org/pdf/2408.16564v2)**

> **作者:** Qianhui Liu; Jiadong Wang; Yang Wang; Xin Yang; Gang Pan; Haizhou Li
>
> **备注:** aceepted by IEEE TC
>
> **摘要:** Humans naturally perform audiovisual speech recognition (AVSR), enhancing the accuracy and robustness by integrating auditory and visual information. Spiking neural networks (SNNs), which mimic the brain's information-processing mechanisms, are well-suited for emulating the human capability of AVSR. Despite their potential, research on SNNs for AVSR is scarce, with most existing audio-visual multimodal methods focused on object or digit recognition. These models simply integrate features from both modalities, neglecting their unique characteristics and interactions. Additionally, they often rely on future information for current processing, which increases recognition latency and limits real-time applicability. Inspired by human speech perception, this paper proposes a novel human-inspired SNN named HI-AVSNN for AVSR, incorporating three key characteristics: cueing interaction, causal processing and spike activity. For cueing interaction, we propose a visual-cued auditory attention module (VCA2M) that leverages visual cues to guide attention to auditory features. We achieve causal processing by aligning the SNN's temporal dimension with that of visual and auditory features and applying temporal masking to utilize only past and current information. To implement spike activity, in addition to using SNNs, we leverage the event camera to capture lip movement as spikes, mimicking the human retina and providing efficient visual data. We evaluate HI-AVSNN on an audiovisual speech recognition dataset combining the DVS-Lip dataset with its corresponding audio samples. Experimental results demonstrate the superiority of our proposed fusion method, outperforming existing audio-visual SNN fusion methods and achieving a 2.27% improvement in accuracy over the only existing SNN-based AVSR method.
>
---
#### [replaced 006] Regularized autoregressive modeling and its application to audio signal reconstruction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.17790v2](http://arxiv.org/pdf/2410.17790v2)**

> **作者:** Ondřej Mokrý; Pavel Rajmic
>
> **摘要:** Autoregressive (AR) modeling is invaluable in signal processing, in particular in speech and audio fields. Attempts in the literature can be found that regularize or constrain either the time-domain signal values or the AR coefficients, which is done for various reasons, including the incorporation of prior information or numerical stabilization. Although these attempts are appealing, an encompassing and generic modeling framework is still missing. We propose such a framework and the related optimization problem and algorithm. We discuss the computational demands of the algorithm and explore the effects of various improvements on its convergence speed. In the experimental part, we demonstrate the usefulness of our approach on the audio declipping and the audio dequantization problems. We compare its performance against the state-of-the-art methods and demonstrate the competitiveness of the proposed method, especially for mildly clipped signals. The evaluation is extended by considering a heuristic algorithm of generalized linear prediction (GLP), a strong competitor which has only been presented as a patent and is new in the scientific community.
>
---
#### [replaced 007] Towards Understanding of Frequency Dependence on Sound Event Detection
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.07208v2](http://arxiv.org/pdf/2502.07208v2)**

> **作者:** Hyeonuk Nam; Seong-Hu Kim; Deokki Min; Byeong-Yun Ko; Yong-Hwa Park
>
> **备注:** Accepted to IEEE/ACM TASLP
>
> **摘要:** In this work, we conduct an in-depth analysis of two frequency-dependent methods for sound event detection (SED): FilterAugment and frequency dynamic convolution (FDY conv). The goal is to better understand their characteristics and behaviors in the context of SED. While SED has been rapidly advancing through the adoption of various deep learning techniques from other pattern recognition fields, such adopted techniques are often not suitable for SED. To address this issue, two frequency-dependent SED methods were previously proposed: FilterAugment, a data augmentation randomly weighting frequency bands, and FDY conv, an architecture applying frequency adaptive convolution kernels. These methods have demonstrated superior performance in SED, and we aim to further analyze their detailed effectiveness and characteristics in SED. We compare class-wise performance to find out specific pros and cons of FilterAugment and FDY conv. We apply Gradient-weighted Class Activation Mapping (Grad-CAM), which highlights time-frequency region that is more inferred by the model, on SED models with and without frequency masking and two types of FilterAugment to observe their detailed characteristics. We propose simpler frequency dependent convolution methods and compare them with FDY conv to further understand which components of FDY conv affects SED performance. Lastly, we apply PCA to show how FDY conv adapts dynamic kernel across frequency dimensions on different sound event classes. The results and discussions demonstrate that frequency dependency plays a significant role in sound event detection and further confirms the effectiveness of frequency dependent methods on SED.
>
---
#### [replaced 008] LABNet: A Lightweight Attentive Beamforming Network for Ad-hoc Multichannel Microphone Invariant Real-Time Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16190v3](http://arxiv.org/pdf/2507.16190v3)**

> **作者:** Haoyin Yan; Jie Zhang; Chengqian Jiang; Shuang Zhang
>
> **摘要:** Multichannel speech enhancement (SE) aims to restore clean speech from noisy measurements by leveraging spatiotemporal signal features. In ad-hoc array conditions, microphone invariance (MI) requires systems to handle different microphone numbers and array geometries. From a practical perspective, multichannel recordings inevitably increase the computational burden for edge-device applications, highlighting the necessity of lightweight and efficient deployments. In this work, we propose a lightweight attentive beamforming network (LABNet) to integrate MI in a low-complexity real-time SE system. We design a three-stage framework for efficient intra-channel modeling and inter-channel interaction. A cross-channel attention module is developed to aggregate features from each channel selectively. Experimental results demonstrate our LABNet achieves impressive performance with ultra-light resource overhead while maintaining the MI, indicating great potential for ad-hoc array processing. The code is available:https://github.com/Jokejiangv/LABNet.git
>
---
