# 音频 cs.SD;  eess.AS

- **最新发布 19 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Systematic evaluation of time-frequency features for binaural sound source localization
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究双耳声源定位任务，旨在通过系统评估时频特征设计提升模型性能。工作包括对比不同幅度与相位特征组合，在域内和域外数据上验证其效果，发现合理特征选择比增加模型复杂度更有效，为低复杂度高精度定位提供指导。**

- **链接: [https://arxiv.org/pdf/2511.13487v1](https://arxiv.org/pdf/2511.13487v1)**

> **作者:** Davoud Shariat Panah; Alessandro Ragano; Dan Barry; Jan Skoglund; Andrew Hines
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** This study presents a systematic evaluation of time-frequency feature design for binaural sound source localization (SSL), focusing on how feature selection influences model performance across diverse conditions. We investigate the performance of a convolutional neural network (CNN) model using various combinations of amplitude-based features (magnitude spectrogram, interaural level difference - ILD) and phase-based features (phase spectrogram, interaural phase difference - IPD). Evaluations on in-domain and out-of-domain data with mismatched head-related transfer functions (HRTFs) reveal that carefully chosen feature combinations often outperform increases in model complexity. While two-feature sets such as ILD + IPD are sufficient for in-domain SSL, generalization to diverse content requires richer inputs combining channel spectrograms with both ILD and IPD. Using the optimal feature sets, our low-complexity CNN model achieves competitive performance. Our findings underscore the importance of feature design in binaural SSL and provide practical guidance for both domain-specific and general-purpose localization.
>
---
#### [new 002] VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出VoiceCraft-X，统一处理多语言语音合成与编辑任务。针对跨语言文本处理和有限数据挑战，利用Qwen3模型与新型令牌重排序机制，实现零样本TTS与语音编辑一体化。**

- **链接: [https://arxiv.org/pdf/2511.12347v1](https://arxiv.org/pdf/2511.12347v1)**

> **作者:** Zhisheng Zheng; Puyuan Peng; Anuj Diwan; Cong Phuoc Huynh; Xiaohang Sun; Zhu Liu; Vimal Bhat; David Harwath
>
> **备注:** EMNLP 2025. Demo and code are available at https://zhishengzheng.com/voicecraft-x/
>
> **摘要:** We introduce VoiceCraft-X, an autoregressive neural codec language model which unifies multilingual speech editing and zero-shot Text-to-Speech (TTS) synthesis across 11 languages: English, Mandarin, Korean, Japanese, Spanish, French, German, Dutch, Italian, Portuguese, and Polish. VoiceCraft-X utilizes the Qwen3 large language model for phoneme-free cross-lingual text processing and a novel token reordering mechanism with time-aligned text and speech tokens to handle both tasks as a single sequence generation problem. The model generates high-quality, natural-sounding speech, seamlessly creating new audio or editing existing recordings within one framework. VoiceCraft-X shows robust performance in diverse linguistic settings, even with limited per-language data, underscoring the power of unified autoregressive approaches for advancing complex, real-world multilingual speech applications. Audio samples are available at https://zhishengzheng.com/voicecraft-x/.
>
---
#### [new 003] Lightweight Hopfield Neural Networks for Bioacoustic Detection and Call Monitoring of Captive Primates
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出轻量级霍普菲尔德神经网络用于灵长类动物叫声检测与监测，解决传统CNN模型资源消耗大、训练慢的问题。通过存储特定叫声信号实现快速准确识别，可在普通笔记本上高效处理音频数据。**

- **链接: [https://arxiv.org/pdf/2511.11615v1](https://arxiv.org/pdf/2511.11615v1)**

> **作者:** Wendy Lomas; Andrew Gascoyne; Colin Dubreuil; Stefano Vaglio; Liam Naughton
>
> **备注:** 16 pages, 3 figures, Proceedings of the Future Technologies Conference (FTC) 2025, Volume 1
>
> **摘要:** Passive acoustic monitoring is a sustainable method of monitoring wildlife and environments that leads to the generation of large datasets and, currently, a processing backlog. Academic research into automating this process is focused on the application of resource intensive convolutional neural networks which require large pre-labelled datasets for training and lack flexibility in application. We present a viable alternative relevant in both wild and captive settings; a transparent, lightweight and fast-to-train associative memory AI model with Hopfield neural network (HNN) architecture. Adapted from a model developed to detect bat echolocation calls, this model monitors captive endangered black-and-white ruffed lemur Varecia variegata vocalisations. Lemur social calls of interest when monitoring welfare are stored in the HNN in order to detect other call instances across the larger acoustic dataset. We make significant model improvements by storing an additional signal caused by movement and achieve an overall accuracy of 0.94. The model can perform $340$ classifications per second, processing over 5.5 hours of audio data per minute, on a standard laptop running other applications. It has broad applicability and trains in milliseconds. Our lightweight solution reduces data-to-insight turnaround times and can accelerate decision making in both captive and wild settings.
>
---
#### [new 004] Real-Time Speech Enhancement via a Hybrid ViT: A Dual-Input Acoustic-Image Feature Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出一种基于混合ViT的双输入声学图像特征融合框架，用于实时单通道语音增强。针对非平稳噪声下语音质量下降问题，该方法有效建模时频依赖关系，在嵌入式设备上实现高效降噪，显著提升语音清晰度和感知质量。**

- **链接: [https://arxiv.org/pdf/2511.11825v1](https://arxiv.org/pdf/2511.11825v1)**

> **作者:** Behnaz Bahmei; Siamak Arzanpour; Elina Birmingham
>
> **摘要:** Speech quality and intelligibility are significantly degraded in noisy environments. This paper presents a novel transformer-based learning framework to address the single-channel noise suppression problem for real-time applications. Although existing deep learning networks have shown remarkable improvements in handling stationary noise, their performance often diminishes in real-world environments characterized by non-stationary noise (e.g., dog barking, baby crying). The proposed dual-input acoustic-image feature fusion using a hybrid ViT framework effectively models both temporal and spectral dependencies in noisy signals. Designed for real-world audio environments, the proposed framework is computationally lightweight and suitable for implementation on embedded devices. To evaluate its effectiveness, four standard and commonly used quality measurements, namely PESQ, STOI, Seg SNR, and LLR, are utilized. Experimental results obtained using the Librispeech dataset as the clean speech source and the UrbanSound8K and Google Audioset datasets as the noise sources, demonstrate that the proposed method significantly improves noise reduction, speech intelligibility, and perceptual quality compared to the noisy input signal, achieving performance close to the clean reference.
>
---
#### [new 005] Eardrum sound pressure prediction from ear canal reflectance based on the inverse solution of Webster's horn equation
- **分类: eess.AS; physics.app-ph**

- **简介: 该论文属于听力系统个性化建模任务，旨在通过耳道反射特性反演声压分布。解决传统方法因高频缺失导致的面积函数精度不足问题，提出基于Webster方程逆解的优化空间分辨率方法，并验证了一维模型对三维数据的良好复现能力。**

- **链接: [https://arxiv.org/pdf/2511.12552v1](https://arxiv.org/pdf/2511.12552v1)**

> **作者:** Reinhild Roden; Tobias Sankowsky-Rothe; Nick Wulbusch; Alexey Chernov; Matthias Blau
>
> **备注:** Manuscript submitted to the Journal of the Acoustical Society of America (under minor revision)
>
> **摘要:** To derive ear canal transfer functions for individualized equalization algorithms of in-ear hearing systems, individual ear canal models are needed. In a one-dimensional approach, this requires the estimation of the individual area function of the ear canal. The area function can be effectively and reproducibly calculated as the inverse solution of Webster's horn equation by finite difference approximation of the time domain reflectance. Building upon previous research, the present study further investigates the termination of the approximation at an optimal spatial resolution, addressing the absence of higher frequencies in typical ear canal measurements and enhancing the accuracy of the inverse solution. Compared to the geometric reference, more precise area functions were achieved by extrapolating simulated input impedances of ear canal geometries up to a frequency of 3.5 MHz, corresponding to 0.1 mm spatial resolution. The low pass of the previous work was adopted but adjusted for its cut-off frequency depending on the highest frequency of the band-limited input impedance. Robust criteria for terminating the area function at the approximated ear canal length were found. Finally, three-dimensional simulated and measured ear canal transfer impedances were replicated well employing the previously introduced and herein validated one-dimensional electro-acoustic model fed by the area functions.
>
---
#### [new 006] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement
- **分类: cs.SD; cs.AI**

- **简介: 该论文聚焦于语音生成任务，旨在解决语音因素纠缠和控制粒度粗的问题。提出MF-Speech框架，通过因子解耦与动态融合实现内容、音色、情感的精细可控合成，显著提升语音质量和控制精度。**

- **链接: [https://arxiv.org/pdf/2511.12074v1](https://arxiv.org/pdf/2511.12074v1)**

> **作者:** Xinyue Yu; Youqing Fang; Pingyu Wu; Guoyang Ye; Wenbo Zhou; Weiming Zhang; Song Xiao
>
> **摘要:** Generating expressive and controllable human speech is one of the core goals of generative artificial intelligence, but its progress has long been constrained by two fundamental challenges: the deep entanglement of speech factors and the coarse granularity of existing control mechanisms. To overcome these challenges, we have proposed a novel framework called MF-Speech, which consists of two core components: MF-SpeechEncoder and MF-SpeechGenerator. MF-SpeechEncoder acts as a factor purifier, adopting a multi-objective optimization strategy to decompose the original speech signal into highly pure and independent representations of content, timbre, and emotion. Subsequently, MF-SpeechGenerator functions as a conductor, achieving precise, composable and fine-grained control over these factors through dynamic fusion and Hierarchical Style Adaptive Normalization (HSAN). Experiments demonstrate that in the highly challenging multi-factor compositional speech generation task, MF-Speech significantly outperforms current state-of-the-art methods, achieving a lower word error rate (WER=4.67%), superior style control (SECS=0.5685, Corr=0.68), and the highest subjective evaluation scores(nMOS=3.96, sMOS_emotion=3.86, sMOS_style=3.78). Furthermore, the learned discrete factors exhibit strong transferability, demonstrating their significant potential as a general-purpose speech representation.
>
---
#### [new 007] FoleyBench: A Benchmark For Video-to-Audio Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出FoleyBench，首个专为视频到音频生成（V2A）设计的基准数据集，解决现有评估数据与Foley应用场景不匹配的问题。包含5000个强音画对应视频-音频-文本三元组，支持细粒度模型评测。**

- **链接: [https://arxiv.org/pdf/2511.13219v1](https://arxiv.org/pdf/2511.13219v1)**

> **作者:** Satvik Dixit; Koichi Saito; Zhi Zhong; Yuki Mitsufuji; Chris Donahue
>
> **摘要:** Video-to-audio generation (V2A) is of increasing importance in domains such as film post-production, AR/VR, and sound design, particularly for the creation of Foley sound effects synchronized with on-screen actions. Foley requires generating audio that is both semantically aligned with visible events and temporally aligned with their timing. Yet, there is a mismatch between evaluation and downstream applications due to the absence of a benchmark tailored to Foley-style scenarios. We find that 74% of videos from past evaluation datasets have poor audio-visual correspondence. Moreover, they are dominated by speech and music, domains that lie outside the use case for Foley. To address this gap, we introduce FoleyBench, the first large-scale benchmark explicitly designed for Foley-style V2A evaluation. FoleyBench contains 5,000 (video, ground-truth audio, text caption) triplets, each featuring visible sound sources with audio causally tied to on-screen events. The dataset is built using an automated, scalable pipeline applied to in-the-wild internet videos from YouTube-based and Vimeo-based sources. Compared to past datasets, we show that videos from FoleyBench have stronger coverage of sound categories from a taxonomy specifically designed for Foley sound. Each clip is further labeled with metadata capturing source complexity, UCS/AudioSet category, and video length, enabling fine-grained analysis of model performance and failure modes. We benchmark several state-of-the-art V2A models, evaluating them on audio quality, audio-video alignment, temporal synchronization, and audio-text consistency. Samples are available at: https://gclef-cmu.org/foleybench
>
---
#### [new 008] How Far Do SSL Speech Models Listen for Tone? Temporal Focus of Tone Representation under Low-resource Transfer
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究SSL语音模型在低资源条件下对声调的时序感知范围，针对缅甸语、泰语、老挝语和越南语四种声调语言，发现模型听觉跨度因下游任务而异：语音识别任务聚焦语言特定声调线索，而韵律相关任务则偏向过长跨度。**

- **链接: [https://arxiv.org/pdf/2511.12285v1](https://arxiv.org/pdf/2511.12285v1)**

> **作者:** Minu Kim; Ji Sub Um; Hoirin Kim
>
> **备注:** 5 pages, 7 figures, submitted to ICASSP 2026
>
> **摘要:** Lexical tone is central to many languages but remains underexplored in self-supervised learning (SSL) speech models, especially beyond Mandarin. We study four languages with complex and diverse tone systems: Burmese, Thai, Lao, and Vietnamese, to examine how far such models listen for tone and how transfer operates in low-resource conditions. As a baseline reference, we estimate the temporal span of tone cues to be about 100 ms in Burmese and Thai, and about 180 ms in Lao and Vietnamese. Probes and gradient analyses on fine-tuned SSL models reveal that tone transfer varies by downstream task: automatic speech recognition fine-tuning aligns spans with language-specific tone cues, while prosody- and voice-related tasks bias the model toward overly long spans. These findings indicate that tone transfer is shaped by downstream task, highlighting task effects on temporal focus in tone modeling.
>
---
#### [new 009] Spatial Blind Spot: Auditory Motion Perception Deficits in Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音频大模型在听觉空间感知上的缺陷，提出AMPBench基准测试来评估模型对移动声源方向和轨迹的理解能力。结果表明当前模型表现不佳，平均准确率低于50%，揭示了其在空间推理上的根本不足。**

- **链接: [https://arxiv.org/pdf/2511.13273v1](https://arxiv.org/pdf/2511.13273v1)**

> **作者:** Zhe Sun; Yujun Cai; Jiayu Yao; Yiwei Wang
>
> **摘要:** Large Audio-Language Models (LALMs) have recently shown impressive progress in speech recognition, audio captioning, and auditory question answering. Yet, whether these models can perceive spatial dynamics, particularly the motion of sound sources, remains unclear. In this work, we uncover a systematic motion perception deficit in current ALLMs. To investigate this issue, we introduce AMPBench, the first benchmark explicitly designed to evaluate auditory motion understanding. AMPBench introduces a controlled question-answering benchmark designed to evaluate whether Audio-Language Models (LALMs) can infer the direction and trajectory of moving sound sources from binaural audio. Comprehensive quantitative and qualitative analyses reveal that current models struggle to reliably recognize motion cues or distinguish directional patterns. The average accuracy remains below 50%, underscoring a fundamental limitation in auditory spatial reasoning. Our study highlights a fundamental gap between human and model auditory spatial reasoning, providing both a diagnostic tool and new insight for enhancing spatial cognition in future Audio-Language Models.
>
---
#### [new 010] Towards Practical Real-Time Low-Latency Music Source Separation
- **分类: cs.SD; cs.MM**

- **简介: 论文针对实时低延迟音乐源分离任务，提出轻量级单路径模型RT-STT，通过通道扩展融合特征并量化优化，显著减少参数与推理时间，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.13146v1](https://arxiv.org/pdf/2511.13146v1)**

> **作者:** Junyu Wu; Jie Liu; Tianrui Pan; Jie Tang; Gangshan Wu
>
> **摘要:** In recent years, significant progress has been made in the field of deep learning for music demixing. However, there has been limited attention on real-time, low-latency music demixing, which holds potential for various applications, such as hearing aids, audio stream remixing, and live performances. Additionally, a notable tendency has emerged towards the development of larger models, limiting their applicability in certain scenarios. In this paper, we introduce a lightweight real-time low-latency model called Real-Time Single-Path TFC-TDF UNET (RT-STT), which is based on the Dual-Path TFC-TDF UNET (DTTNet). In RT-STT, we propose a feature fusion technique based on channel expansion. We also demonstrate the superiority of single-path modeling over dual-path modeling in real-time models. Moreover, we investigate the method of quantization to further reduce inference time. RT-STT exhibits superior performance with significantly fewer parameters and shorter inference times compared to state-of-the-art models.
>
---
#### [new 011] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决生成模型在强噪声下产生的语言和声学幻觉问题。作者提出PASE框架，利用WavLM的音系先验进行去噪，并通过双流表示训练声码器，有效降低幻觉并提升感知质量。**

- **链接: [https://arxiv.org/pdf/2511.13300v1](https://arxiv.org/pdf/2511.13300v1)**

> **作者:** Xiaobin Rong; Qinwen Hu; Mansur Yesilbursa; Kamil Wojcicki; Jing Lu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Generative models have shown remarkable performance in speech enhancement (SE), achieving superior perceptual quality over traditional discriminative approaches. However, existing generative SE approaches often overlook the risk of hallucination under severe noise, leading to incorrect spoken content or inconsistent speaker characteristics, which we term linguistic and acoustic hallucinations, respectively. We argue that linguistic hallucination stems from models' failure to constrain valid phonological structures and it is a more fundamental challenge. While language models (LMs) are well-suited for capturing the underlying speech structure through modeling the distribution of discrete tokens, existing approaches are limited in learning from noise-corrupted representations, which can lead to contaminated priors and hallucinations. To overcome these limitations, we propose the Phonologically Anchored Speech Enhancer (PASE), a generative SE framework that leverages the robust phonological prior embedded in the pre-trained WavLM model to mitigate hallucinations. First, we adapt WavLM into a denoising expert via representation distillation to clean its final-layer features. Guided by the model's intrinsic phonological prior, this process enables robust denoising while minimizing linguistic hallucinations. To further reduce acoustic hallucinations, we train the vocoder with a dual-stream representation: the high-level phonetic representation provides clean linguistic content, while a low-level acoustic representation retains speaker identity and prosody. Experimental results demonstrate that PASE not only surpasses state-of-the-art discriminative models in perceptual quality, but also significantly outperforms prior generative models with substantially lower linguistic and acoustic hallucinations.
>
---
#### [new 012] SynthGuard: An Open Platform for Detecting AI-Generated Multimedia with Multimodal LLMs
- **分类: cs.MM; cs.AI; cs.SD**

- **简介: 论文提出SynthGuard平台，用于检测AI生成的多媒体内容。针对现有工具闭源、单模态和缺乏透明性的问题，该平台结合传统检测器与多模态大语言模型，提供可解释的推理、统一的图像音频支持及交互界面，提升检测透明度与用户理解。**

- **链接: [https://arxiv.org/pdf/2511.12404v1](https://arxiv.org/pdf/2511.12404v1)**

> **作者:** Shail Desai; Aditya Pawar; Li Lin; Xin Wang; Shu Hu
>
> **摘要:** Artificial Intelligence (AI) has made it possible for anyone to create images, audio, and video with unprecedented ease, enriching education, communication, and creative expression. At the same time, the rapid rise of AI-generated media has introduced serious risks, including misinformation, identity misuse, and the erosion of public trust as synthetic content becomes increasingly indistinguishable from real media. Although deepfake detection has advanced, many existing tools remain closed-source, limited in modality, or lacking transparency and educational value, making it difficult for users to understand how detection decisions are made. To address these gaps, we introduce SynthGuard, an open, user-friendly platform for detecting and analyzing AI-generated multimedia using both traditional detectors and multimodal large language models (MLLMs). SynthGuard provides explainable inference, unified image and audio support, and an interactive interface designed to make forensic analysis accessible to researchers, educators, and the public. The SynthGuard platform is available at: https://in-engr-nova.it.purdue.edu/
>
---
#### [new 013] Lessons Learned from Developing a Privacy-Preserving Multimodal Wearable for Local Voice-and-Vision Inference
- **分类: cs.HC; eess.AS; eess.IV; eess.SY**

- **简介: 论文研究如何在可穿戴设备上实现本地多模态AI推理，解决隐私担忧问题。通过软硬件协同设计，集成摄像头、麦克风等组件于轻量设备中，支持离线运行量化模型，确保隐私与实时性。**

- **链接: [https://arxiv.org/pdf/2511.11811v1](https://arxiv.org/pdf/2511.11811v1)**

> **作者:** Yonatan Tussa; Andy Heredia; Nirupam Roy
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Many promising applications of multimodal wearables require continuous sensing and heavy computation, yet users reject such devices due to privacy concerns. This paper shares our experiences building an ear-mounted voice-and-vision wearable that performs local AI inference using a paired smartphone as a trusted personal edge. We describe the hardware--software co-design of this privacy-preserving system, including challenges in integrating a camera, microphone, and speaker within a 30-gram form factor, enabling wake word-triggered capture, and running quantized vision-language and large-language models entirely offline. Through iterative prototyping, we identify key design hurdles in power budgeting, connectivity, latency, and social acceptability. Our initial evaluation shows that fully local multimodal inference is feasible on commodity mobile hardware with interactive latency. We conclude with design lessons for researchers developing embedded AI systems that balance privacy, responsiveness, and usability in everyday settings.
>
---
#### [new 014] Beyond saliency: enhancing explanation of speech emotion recognition with expert-referenced acoustic cues
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于语音情感识别（SER）中的可解释AI任务，旨在解决现有基于显著性方法无法关联专家定义声学特征的问题。作者提出新框架，通过量化显著区域内的声学线索强度，提升解释的可信度与可理解性。**

- **链接: [https://arxiv.org/pdf/2511.11691v1](https://arxiv.org/pdf/2511.11691v1)**

> **作者:** Seham Nasr; Zhao Ren; David Johnson
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Explainable AI (XAI) for Speech Emotion Recognition (SER) is critical for building transparent, trustworthy models. Current saliency-based methods, adapted from vision, highlight spectrogram regions but fail to show whether these regions correspond to meaningful acoustic markers of emotion, limiting faithfulness and interpretability. We propose a framework that overcomes these limitations by quantifying the magnitudes of cues within salient regions. This clarifies "what" is highlighted and connects it to "why" it matters, linking saliency to expert-referenced acoustic cues of speech emotions. Experiments on benchmark SER datasets show that our approach improves explanation quality by explicitly linking salient regions to theory-driven speech emotions expert-referenced acoustics. Compared to standard saliency methods, it provides more understandable and plausible explanations of SER models, offering a foundational step towards trustworthy speech-based affective computing.
>
---
#### [new 015] Regularized Schrödinger: Alleviating Distortion and Exposure Bias in Solving Inverse Problems
- **分类: cs.LG; cs.SD**

- **简介: 论文针对扩散模型在求解逆问题时存在的失真与暴露偏差问题，提出正则化薛定谔桥（RSB）方法。通过扰动输入和目标，缓解暴露偏差，并利用后验均值插值降低失真，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2511.11686v1](https://arxiv.org/pdf/2511.11686v1)**

> **作者:** Qing Yao; Lijian Gao; Qirong Mao; Dong Ming
>
> **摘要:** Diffusion models serve as a powerful generative framework for solving inverse problems. However, they still face two key challenges: 1) the distortion-perception tradeoff, where improving perceptual quality often degrades reconstruction fidelity, and 2) the exposure bias problem, where the training-inference input mismatch leads to prediction error accumulation and reduced reconstruction quality. In this work, we propose the Regularized Schrödinger Bridge (RSB), an adaptation of Schrödinger Bridge tailored for inverse problems that addresses the above limitations. RSB employs a novel regularized training strategy that perturbs both the input states and targets, effectively mitigating exposure bias by exposing the model to simulated prediction errors and also alleviating distortion by well-designed interpolation via the posterior mean. Extensive experiments on two typical inverse problems for speech enhancement demonstrate that RSB outperforms state-of-the-art methods, significantly improving distortion metrics and effectively reducing exposure bias.
>
---
#### [new 016] Toward Conversational Hungarian Speech Recognition: Introducing the BEA-Large and BEA-Dialogue Datasets
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文聚焦匈牙利语对话语音识别任务，针对其缺乏自发对话语料的问题，构建了BEA-Large和BEA-Dialogue两个新数据集，并提供基线模型与性能指标，推动匈牙利语语音技术发展。**

- **链接: [https://arxiv.org/pdf/2511.13529v1](https://arxiv.org/pdf/2511.13529v1)**

> **作者:** Máté Gedeon; Piroska Zsófia Barta; Péter Mihajlik; Tekla Etelka Gráczi; Anna Kohári; Katalin Mády
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** The advancement of automatic speech recognition (ASR) has been largely enhanced by extensive datasets in high-resource languages, while languages such as Hungarian remain underrepresented due to limited spontaneous and conversational corpora. To address this gap, we introduce two new datasets -- BEA-Large and BEA-Dialogue -- constructed from the previously unprocessed portions of the Hungarian speech corpus named BEA. BEA-Large extends BEA-Base with 255 hours of spontaneous speech from 433 speakers, enriched with detailed segment-level metadata. BEA-Dialogue, comprising 85 hours of spontaneous conversations, is a Hungarian speech corpus featuring natural dialogues partitioned into speaker-independent subsets, supporting research in conversational ASR and speaker diarization. We establish reproducible baselines on these datasets using publicly available ASR models, with the fine-tuned Fast Conformer model achieving word error rates as low as 14.18\% on spontaneous and 4.8\% on repeated speech. Diarization experiments yield diarization error rates between 13.05\% and 18.26\%, providing reference points for future improvements. The results highlight the persistent difficulty of conversational ASR, particularly due to disfluencies, overlaps, and informal speech patterns. By releasing these datasets and baselines, we aim to advance Hungarian speech technology and offer a methodological framework for developing spontaneous and conversational benchmarks in other languages.
>
---
#### [new 017] ProAV-DiT: A Projected Latent Diffusion Transformer for Efficient Synchronized Audio-Video Generation
- **分类: cs.MM; cs.AI; cs.SD**

- **简介: 论文提出ProAV-DiT模型，用于高效同步的音频视频生成任务。针对音视频结构不一致和计算成本高问题，通过预处理和多尺度双流编码器对齐模态，并在统一3D潜空间中用扩散Transformer建模时序依赖，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2511.12072v1](https://arxiv.org/pdf/2511.12072v1)**

> **作者:** Jiahui Sun; Weining Wang; Mingzhen Sun; Yirong Yang; Xinxin Zhu; Jing Liu
>
> **摘要:** Sounding Video Generation (SVG) remains a challenging task due to the inherent structural misalignment between audio and video, as well as the high computational cost of multimodal data processing. In this paper, we introduce ProAV-DiT, a Projected Latent Diffusion Transformer designed for efficient and synchronized audio-video generation. To address structural inconsistencies, we preprocess raw audio into video-like representations, aligning both the temporal and spatial dimensions between audio and video. At its core, ProAV-DiT adopts a Multi-scale Dual-stream Spatio-Temporal Autoencoder (MDSA), which projects both modalities into a unified latent space using orthogonal decomposition, enabling fine-grained spatiotemporal modeling and semantic alignment. To further enhance temporal coherence and modality-specific fusion, we introduce a multi-scale attention mechanism, which consists of multi-scale temporal self-attention and group cross-modal attention. Furthermore, we stack the 2D latents from MDSA into a unified 3D latent space, which is processed by a spatio-temporal diffusion Transformer. This design efficiently models spatiotemporal dependencies, enabling the generation of high-fidelity synchronized audio-video content while reducing computational overhead. Extensive experiments conducted on standard benchmarks demonstrate that ProAV-DiT outperforms existing methods in both generation quality and computational efficiency.
>
---
#### [new 018] A Smart-Glasses for Emergency Medical Services via Multimodal Multitask Learning
- **分类: cs.LG; eess.AS; eess.IV**

- **简介: 论文提出EMSGlass系统，用于急救场景下的智能辅助决策。针对EMS中多模态数据处理慢、实时性差的问题，设计EMSNet多任务模型和EMSServe低延迟推理框架，实现文本、生命体征与图像的联合分析，提升急救效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.13078v1](https://arxiv.org/pdf/2511.13078v1)**

> **作者:** Liuyi Jin; Pasan Gunawardena; Amran Haroon; Runzhi Wang; Sangwoo Lee; Radu Stoleru; Michael Middleton; Zepeng Huo; Jeeeun Kim; Jason Moats
>
> **摘要:** Emergency Medical Technicians (EMTs) operate in high-pressure environments, making rapid, life-critical decisions under heavy cognitive and operational loads. We present EMSGlass, a smart-glasses system powered by EMSNet, the first multimodal multitask model for Emergency Medical Services (EMS), and EMSServe, a low-latency multimodal serving framework tailored to EMS scenarios. EMSNet integrates text, vital signs, and scene images to construct a unified real-time understanding of EMS incidents. Trained on real-world multimodal EMS datasets, EMSNet simultaneously supports up to five critical EMS tasks with superior accuracy compared to state-of-the-art unimodal baselines. Built on top of PyTorch, EMSServe introduces a modality-aware model splitter and a feature caching mechanism, achieving adaptive and efficient inference across heterogeneous hardware while addressing the challenge of asynchronous modality arrival in the field. By optimizing multimodal inference execution in EMS scenarios, EMSServe achieves 1.9x -- 11.7x speedup over direct PyTorch multimodal inference. A user study evaluation with six professional EMTs demonstrates that EMSGlass enhances real-time situational awareness, decision-making speed, and operational efficiency through intuitive on-glass interaction. In addition, qualitative insights from the user study provide actionable directions for extending EMSGlass toward next-generation AI-enabled EMS systems, bridging multimodal intelligence with real-world emergency response workflows.
>
---
#### [new 019] Enhancing XR Auditory Realism via Multimodal Scene-Aware Acoustic Rendering
- **分类: cs.HC; cs.CV; cs.LG; cs.SD**

- **简介: 论文提出SAMOSA系统，解决XR中声音与视觉不匹配问题。通过融合房间几何、材质和语义信息，动态生成高保真声学响应，提升虚拟环境听觉真实感。**

- **链接: [https://arxiv.org/pdf/2511.11930v1](https://arxiv.org/pdf/2511.11930v1)**

> **作者:** Tianyu Xu; Jihan Li; Penghe Zu; Pranav Sahay; Maruchi Kim; Jack Obeng-Marnu; Farley Miller; Xun Qian; Katrina Passarella; Mahitha Rachumalla; Rajeev Nongpiur; D. Shin
>
> **摘要:** In Extended Reality (XR), rendering sound that accurately simulates real-world acoustics is pivotal in creating lifelike and believable virtual experiences. However, existing XR spatial audio rendering methods often struggle with real-time adaptation to diverse physical scenes, causing a sensory mismatch between visual and auditory cues that disrupts user immersion. To address this, we introduce SAMOSA, a novel on-device system that renders spatially accurate sound by dynamically adapting to its physical environment. SAMOSA leverages a synergistic multimodal scene representation by fusing real-time estimations of room geometry, surface materials, and semantic-driven acoustic context. This rich representation then enables efficient acoustic calibration via scene priors, allowing the system to synthesize a highly realistic Room Impulse Response (RIR). We validate our system through technical evaluation using acoustic metrics for RIR synthesis across various room configurations and sound types, alongside an expert evaluation (N=12). Evaluation results demonstrate SAMOSA's feasibility and efficacy in enhancing XR auditory realism.
>
---
## 更新

#### [replaced 001] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.08496v3](https://arxiv.org/pdf/2511.08496v3)**

> **作者:** Bingsong Bai; Yizhong Geng; Fengping Wang; Cong Wang; Puyuan Guo; Yingming Gao; Ya Li
>
> **备注:** Accepted by AAAI 2026 main technical track
>
> **摘要:** Zero-shot singing voice conversion (SVC) transforms a source singer's timbre to an unseen target speaker's voice while preserving melodic content without fine-tuning. Existing methods model speaker timbre and vocal content separately, losing essential acoustic information that degrades output quality while requiring significant computational resources. To overcome these limitations, we propose HQ-SVC, an efficient framework for high-quality zero-shot SVC. HQ-SVC first extracts jointly content and speaker features using a decoupled codec. It then enhances fidelity through pitch and volume modeling, preserving critical acoustic information typically lost in separate modeling approaches, and progressively refines outputs via differentiable signal processing and diffusion techniques. Evaluations confirm HQ-SVC significantly outperforms state-of-the-art zero-shot SVC methods in conversion quality and efficiency. Beyond voice conversion, HQ-SVC achieves superior voice naturalness compared to specialized audio super-resolution methods while natively supporting voice super-resolution tasks.
>
---
#### [replaced 002] READ: Real-time and Efficient Asynchronous Diffusion for Audio-driven Talking Head Generation
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2508.03457v3](https://arxiv.org/pdf/2508.03457v3)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Jianqing Gao; Qingfeng Liu
>
> **备注:** Project page: https://readportrait.github.io/READ/
>
> **摘要:** The introduction of diffusion models has brought significant advances to the field of audio-driven talking head generation. However, the extremely slow inference speed severely limits the practical implementation of diffusion-based talking head generation models. In this study, we propose READ, a real-time diffusion-transformer-based talking head generation framework. Our approach first learns a spatiotemporal highly compressed video latent space via a temporal VAE, significantly reducing the token count to accelerate generation. To achieve better audio-visual alignment within this compressed latent space, a pre-trained Speech Autoencoder (SpeechAE) is proposed to generate temporally compressed speech latent codes corresponding to the video latent space. These latent representations are then modeled by a carefully designed Audio-to-Video Diffusion Transformer (A2V-DiT) backbone for efficient talking head synthesis. Furthermore, to ensure temporal consistency and accelerated inference in extended generation, we propose a novel asynchronous noise scheduler (ANS) for both the training and inference processes of our framework. The ANS leverages asynchronous add-noise and asynchronous motion-guided generation in the latent space, ensuring consistency in generated video clips. Experimental results demonstrate that READ outperforms state-of-the-art methods by generating competitive talking head videos with significantly reduced runtime, achieving an optimal balance between quality and speed while maintaining robust metric stability in long-time generation.
>
---
#### [replaced 003] Invisible Ears at Your Fingertips: Acoustic Eavesdropping via Mouse Sensors
- **分类: cs.CR; cs.SD**

- **链接: [https://arxiv.org/pdf/2509.13581v2](https://arxiv.org/pdf/2509.13581v2)**

> **作者:** Mohamad Fakih; Rahul Dharmaji; Youssef Mahmoud; Halima Bouzidi; Mohammad Abdullah Al Faruque
>
> **备注:** Appearing in the Annual Computer Security Applications Conference (ACSAC 2025)
>
> **摘要:** Modern optical mouse sensors, with their advanced precision and high responsiveness, possess an often overlooked vulnerability: they can be exploited for side-channel attacks. This paper introduces Mic-E-Mouse, the first-ever side-channel attack that targets high-performance optical mouse sensors to covertly eavesdrop on users. We demonstrate that audio signals can induce subtle surface vibrations detectable by a mouse's optical sensor. Remarkably, user-space software on popular operating systems can collect and broadcast this sensitive side channel, granting attackers access to raw mouse data without requiring direct system-level permissions. Initially, the vibration signals extracted from mouse data are of poor quality due to non-uniform sampling, a non-linear frequency response, and significant quantization. To overcome these limitations, Mic-E-Mouse employs a sophisticated end-to-end data filtering pipeline that combines Wiener filtering, resampling corrections, and an innovative encoder-only spectrogram neural filtering technique. We evaluate the attack's efficacy across diverse conditions, including speaking volume, mouse polling rate and DPI, surface materials, speaker languages, and environmental noise. In controlled environments, Mic-E-Mouse improves the signal-to-noise ratio (SNR) by up to +19 dB for speech reconstruction. Furthermore, our results demonstrate a speech recognition accuracy of roughly 42% to 61% on the AudioMNIST and VCTK datasets. All our code and datasets are publicly accessible on https://sites.google.com/view/mic-e-mouse.
>
---
#### [replaced 004] DRAGON: Distributional Rewards Optimize Diffusion Generative Models
- **分类: cs.SD; cs.AI; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2504.15217v2](https://arxiv.org/pdf/2504.15217v2)**

> **作者:** Yatong Bai; Jonah Casebeer; Somayeh Sojoudi; Nicholas J. Bryan
>
> **备注:** Accepted to TMLR
>
> **摘要:** We present Distributional RewArds for Generative OptimizatioN (DRAGON), a versatile framework for fine-tuning media generation models towards a desired outcome. Compared with traditional reinforcement learning with human feedback (RLHF) or pairwise preference approaches such as direct preference optimization (DPO), DRAGON is more flexible. It can optimize reward functions that evaluate either individual examples or distributions of them, making it compatible with a broad spectrum of instance-wise, instance-to-distribution, and distribution-to-distribution rewards. Leveraging this versatility, we construct novel reward functions by selecting an encoder and a set of reference examples to create an exemplar distribution. When cross-modal encoders such as CLAP are used, the reference may be of a different modality (text versus audio). Then, DRAGON gathers online and on-policy generations, scores them with the reward function to construct a positive demonstration set and a negative set, and leverages the contrast between the two finite sets to approximate distributional reward optimization. For evaluation, we fine-tune an audio-domain text-to-music diffusion model with 20 reward functions, including a custom music aesthetics model, CLAP score, Vendi diversity, and Frechet audio distance (FAD). We further compare instance-wise (per-song) and full-dataset FAD settings while ablating multiple FAD encoders and reference sets. Over all 20 target rewards, DRAGON achieves an 81.45% average win rate. Moreover, reward functions based on exemplar sets enhance generations and are comparable to model-based rewards. With an appropriate exemplar set, DRAGON achieves a 60.95% human-voted music quality win rate without training on human preference annotations. DRAGON is a new approach to designing and optimizing reward functions for improving human-perceived quality. Demos at https://ml-dragon.github.io/web
>
---
#### [replaced 005] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [https://arxiv.org/pdf/2509.01787v2](https://arxiv.org/pdf/2509.01787v2)**

> **作者:** Yiwei Guo; Bohan Li; Hankun Wang; Zhihan Li; Shuai Wang; Xie Chen; Kai Yu
>
> **备注:** 15 pages, 10 tables, 6 figures. This is the camera ready version for AAAI 2026, plus an appendix for supplementary experimental details and results
>
> **摘要:** Although current large audio language models (LALMs) extend text large language models (LLMs) with generic acoustic understanding abilities, they usually suffer from prompt sensitivity, where different instructions of the same intention can yield drastically different outcomes. In this work, we propose AHAMask, where we simply mask some of the attention heads in the decoder-only LLM backbone of LALMs, to trigger specific acoustic task functionalities without instructions. These masks are efficiently obtained by training on an LALM, with the number of trainable parameters equal to the attention head count in its LLM backbone. We show by experiments that applying such selective attention head masks achieves comparable or even better performance than using instructions, either on single or composite tasks. Besides achieving reliable acoustic task specification for LALMs, this also reveals that LALMs exhibit certain "functional pathways" in their attention heads.
>
---
#### [replaced 006] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2508.08961v3](https://arxiv.org/pdf/2508.08961v3)**

> **作者:** Yuanyuan Wang; Dongchao Yang; Yiwen Shao; Hangting Chen; Jiankun Zhao; Zhiyong Wu; Helen Meng; Xixin Wu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Extending pre-trained text Large Language Models (LLMs)'s speech understanding or generation abilities by introducing various effective speech tokens has attracted great attention in the speech community. However, building a unified speech understanding and generation model still faces the following challenges: (1) Due to the huge modality gap between speech and text tokens, extending text LLMs to unified speech LLMs relies on large-scale paired data for fine-tuning, and (2) Generation and understanding tasks prefer information at different levels, e.g., generation benefits from detailed acoustic features, while understanding favors high-level semantics. This divergence leads to difficult performance optimization in one unified model. To solve these challenges, in this paper, we present two key insights in speech tokenization and speech language modeling. Specifically, we first propose an Understanding-driven Speech Tokenizer (USTokenizer), which extracts high-level semantic information essential for accomplishing understanding tasks using text LLMs. In this way, USToken enjoys better modality commonality with text, which reduces the difficulty of modality alignment in adapting text LLMs to speech LLMs. Secondly, we present DualSpeechLM, a dual-token modeling framework that concurrently models USToken as input and acoustic token as output within a unified, end-to-end framework, seamlessly integrating speech understanding and generation capabilities. Furthermore, we propose a novel semantic supervision loss and a Chain-of-Condition (CoC) strategy to stabilize model training and enhance speech generation performance. Experimental results demonstrate that our proposed approach effectively fosters a complementary relationship between understanding and generation tasks, highlighting the promising strategy of mutually enhancing both tasks in one unified model.
>
---
#### [replaced 007] AcousTools: A 'Full-Stack', Python-Based, Acoustic Holography Library
- **分类: cs.SD; cs.ET**

- **链接: [https://arxiv.org/pdf/2511.07336v3](https://arxiv.org/pdf/2511.07336v3)**

> **作者:** Joshua Mukherjee; Giorgos Christopoulos; Zhouyang Shen; Sriram Subramanian; Ryuji Hirayama
>
> **备注:** 14 Pages, 7 Figures, 2 Tables
>
> **摘要:** Acoustic Holography is an emerging field where mid-air ultrasound is controlled and manipulated for novel and exciting applications. These range from mid-air haptics, volumetric displays, contactless fabrication, and even chemical and biomedical applications such as drug delivery. To develop these applications, a software framework to predict acoustic behaviour and simulating resulting effects, such as applied forces or scattering patterns is desirable. There have been various software libraries and platforms that attempt to fill this role, but there is yet to be a single piece of software that acts as a 'full-stack' solution. We define this full-stack as the process from abstraction to physicalisation starting with setup, modelling acoustic propagation, transducer phase retrieval, sound field analysis, and control of the acoustic holographic hardware itself. Existing methods fail to fulfil one or more of these categories. To address this, we present AcousTools, a Python-based acoustic holography library, designed to support the full suite of acoustic holographic applications and we show AcousTools's ability to meet each step of the full-stack's requirements. AcousTools has the potential to become the standard code library for acoustic holography, with the uniquely complete suite of features wrapped in a language that is known to be easy to use, AcousTools will increase the ability for researchers to develop novel applications as well as accurately review other's work. The full-stack, aside from software, will also be useful for researchers - providing a way to view and compare methodologies by understanding where they fit into the stack.
>
---
#### [replaced 008] Lina-Speech: Gated Linear Attention and Initial-State Tuning for Multi-Sample Prompting Text-To-Speech Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [https://arxiv.org/pdf/2410.23320v2](https://arxiv.org/pdf/2410.23320v2)**

> **作者:** Théodor Lemerle; Téo Guichoux; Axel Roebel; Nicolas Obin
>
> **备注:** Audio-AAAI Workshop, 2026
>
> **摘要:** Neural codec language models, built on transformer architecture, have revolutionized text-to-speech (TTS) synthesis, excelling in voice cloning by treating it as a prefix continuation task. However, their limited context length hinders their effectiveness to short speech samples. As a result, the voice cloning ability is restricted to a limited coverage and diversity of the speaker's prosody and style. Besides, adapting prosody, accent, or appropriate emotion from a short prefix remains a challenging task. Finally, the quadratic complexity of self-attention limits inference throughput. In this work, we introduce Lina-Speech, a TTS model with Gated Linear Attention (GLA) to replace standard self-attention as a principled backbone, improving inference throughput while matching state-of-the-art performance. Leveraging the stateful property of recurrent architecture, we introduce an Initial-State Tuning (IST) strategy that unlocks the possibility of multiple speech sample conditioning of arbitrary numbers and lengths and provides a comprehensive and efficient strategy for voice cloning and out-of-domain speaking style and emotion adaptation. We demonstrate the effectiveness of this approach for controlling fine-grained characteristics such as prosody and emotion. Code, checkpoints, and demo are freely available: https://github.com/theodorblackbird/lina-speech
>
---
#### [replaced 009] Multi-Metric Preference Alignment for Generative Speech Restoration
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [https://arxiv.org/pdf/2508.17229v2](https://arxiv.org/pdf/2508.17229v2)**

> **作者:** Junan Zhang; Xueyao Zhang; Jing Yang; Yuancheng Wang; Fan Fan; Zhizheng Wu
>
> **备注:** Accepted by AAAI 2026. Demopage: https://gensr-pref.github.io
>
> **摘要:** Recent generative models have significantly advanced speech restoration tasks, yet their training objectives often misalign with human perceptual preferences, resulting in suboptimal quality. While post-training alignment has proven effective in other generative domains like text and image generation, its application to generative speech restoration remains largely under-explored. This work investigates the challenges of applying preference-based post-training to this task, focusing on how to define a robust preference signal and curate high-quality data to avoid reward hacking. To address these challenges, we propose a multi-metric preference alignment strategy. We construct a new dataset, GenSR-Pref, comprising 80K preference pairs, where each chosen sample is unanimously favored by a complementary suite of metrics covering perceptual quality, signal fidelity, content consistency, and timbre preservation. This principled approach ensures a holistic preference signal. Applying Direct Preference Optimization (DPO) with our dataset, we observe consistent and significant performance gains across three diverse generative paradigms: autoregressive models (AR), masked generative models (MGM), and flow-matching models (FM) on various restoration benchmarks, in both objective and subjective evaluations. Ablation studies confirm the superiority of our multi-metric strategy over single-metric approaches in mitigating reward hacking. Furthermore, we demonstrate that our aligned models can serve as powerful ''data annotators'', generating high-quality pseudo-labels to serve as a supervision signal for traditional discriminative models in data-scarce scenarios like singing voice restoration. Demo Page:https://gensr-pref.github.io
>
---
#### [replaced 010] Audio Palette: A Diffusion Transformer with Multi-Signal Conditioning for Controllable Foley Synthesis
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2510.12175v2](https://arxiv.org/pdf/2510.12175v2)**

> **作者:** Junnuo Wang
>
> **备注:** Accepted for publication in the Artificial Intelligence Technology Research (AITR), Vol. 3, No. 2, December 2025
>
> **摘要:** Recent advances in diffusion-based generative models have enabled high-quality text-to-audio synthesis, but fine-grained acoustic control remains a significant challenge in open-source research. We present Audio Palette, a diffusion transformer (DiT) based model that extends the Stable Audio Open architecture to address this "control gap" in controllable audio generation. Unlike prior approaches that rely solely on semantic conditioning, Audio Palette introduces four time-varying control signals: loudness, pitch, spectral centroid, and timbre, for precise and interpretable manipulation of acoustic features. The model is efficiently adapted for the nuanced domain of Foley synthesis using Low-Rank Adaptation (LoRA) on a curated subset of AudioSet, requiring only 0.85 percent of the original parameters to be trained. Experiments demonstrate that Audio Palette achieves fine-grained, interpretable control of sound attributes. Crucially, it accomplishes this novel controllability while maintaining high audio quality and strong semantic alignment to text prompts, with performance on standard metrics such as Frechet Audio Distance (FAD) and LAION-CLAP scores remaining comparable to the original baseline model. We provide a scalable, modular pipeline for audio research, emphasizing sequence-based conditioning, memory efficiency, and a three-scale classifier-free guidance mechanism for nuanced inference-time control. This work establishes a robust foundation for controllable sound design and performative audio synthesis in open-source settings, enabling a more artist-centric workflow.
>
---
#### [replaced 011] Study on the Fairness of Speaker Verification Systems on Underrepresented Accents in English
- **分类: eess.AS; cs.SD**

- **链接: [https://arxiv.org/pdf/2204.12649v2](https://arxiv.org/pdf/2204.12649v2)**

> **作者:** Mariel Estevez; Luciana Ferrer
>
> **备注:** 5 pages, 2 figures, submitted to ICASSP
>
> **摘要:** Speaker verification (SV) systems are currently being used to make sensitive decisions like giving access to bank accounts or deciding whether the voice of a suspect coincides with that of the perpetrator of a crime. Ensuring that these systems are fair and do not disfavor any particular group is crucial. In this work, we analyze the performance of several state-of-the-art SV systems across groups defined by the accent of the speakers when speaking English. To this end, we curated a new dataset based on the VoxCeleb corpus where we carefully selected samples from speakers with accents from different countries. We use this dataset to evaluate system performance for several SV systems trained with VoxCeleb data. We show that, while discrimination performance is reasonably robust across accent groups, calibration performance degrades dramatically on some accents that are not well represented in the training data. Finally, we show that a simple data balancing approach mitigates this undesirable bias, being particularly effective when applied to our recently-proposed discriminative condition-aware backend.
>
---
#### [replaced 012] MusRec: Zero-Shot Text-to-Music Editing via Rectified Flow and Diffusion Transformers
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.04376v2](https://arxiv.org/pdf/2511.04376v2)**

> **作者:** Ali Boudaghi; Hadi Zare
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Music editing has emerged as an important and practical area of artificial intelligence, with applications ranging from video game and film music production to personalizing existing tracks according to user preferences. However, existing models face significant limitations, such as being restricted to editing synthesized music generated by their own models, requiring highly precise prompts, or necessitating task-specific retraining, thus lacking true zero-shot capability. leveraging recent advances in rectified flow and diffusion transformers, we introduce MusRec, a zero-shot text-to-music editing model capable of performing diverse editing tasks on real-world music efficiently and effectively. Experimental results demonstrate that our approach outperforms existing methods in preserving musical content, structural consistency, and editing fidelity, establishing a strong foundation for controllable music editing in real-world scenarios.
>
---
#### [replaced 013] DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition
- **分类: cs.SD; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.11000v2](https://arxiv.org/pdf/2511.11000v2)**

> **作者:** HongYu Liu; Junxin Li; Changxi Guo; Hao Chen; Yaqian Huang; Yifu Guo; Huan Yang; Lihua Cai
>
> **备注:** 8 pages, 2 figures. To appear in: Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025), Frontiers in Artificial Intelligence and Applications, Vol. 413. DOI: 10.3233/FAIA251182
>
> **摘要:** Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at https://github.com/david188888/DialogGraph-LLM.
>
---
