# 音频 cs.SD;  eess.SP

- **最新发布 27 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Text2Move: Text-to-moving sound generation via trajectory prediction and temporal alignment
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Text2Move，解决从文本生成可控移动声音的问题。构建合成数据集并训练文本到轨迹模型，结合预训练文本到音频模型生成时空对齐的立体声音。属于文本生成移动声音任务。**

- **链接: [http://arxiv.org/pdf/2509.21919v1](http://arxiv.org/pdf/2509.21919v1)**

> **作者:** Yunyi Liu; Shaofan Yang; Kai Li; Xu Li
>
> **摘要:** Human auditory perception is shaped by moving sound sources in 3D space, yet prior work in generative sound modelling has largely been restricted to mono signals or static spatial audio. In this work, we introduce a framework for generating moving sounds given text prompts in a controllable fashion. To enable training, we construct a synthetic dataset that records moving sounds in binaural format, their spatial trajectories, and text captions about the sound event and spatial motion. Using this dataset, we train a text-to-trajectory prediction model that outputs the three-dimensional trajectory of a moving sound source given text prompts. To generate spatial audio, we first fine-tune a pre-trained text-to-audio generative model to output temporally aligned mono sound with the trajectory. The spatial audio is then simulated using the predicted temporally-aligned trajectory. Experimental evaluation demonstrates reasonable spatial understanding of the text-to-trajectory model. This approach could be easily integrated into existing text-to-audio generative workflow and extended to moving sound generation in other spatial audio formats.
>
---
#### [new 002] Noise-to-Notes: Diffusion-based Generation and Refinement for Automatic Drum Transcription
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出Noise-to-Notes（N2N）框架，将自动鼓音转录（ADT）重新定义为生成任务，利用扩散模型从噪声生成鼓事件及力度。通过引入Annealed Pseudo-Huber损失和音乐基础模型特征，提升了鲁棒性和性能，达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2509.21739v1](http://arxiv.org/pdf/2509.21739v1)**

> **作者:** Michael Yeung; Keisuke Toyama; Toya Teramoto; Shusuke Takahashi; Tamaki Kojima
>
> **摘要:** Automatic drum transcription (ADT) is traditionally formulated as a discriminative task to predict drum events from audio spectrograms. In this work, we redefine ADT as a conditional generative task and introduce Noise-to-Notes (N2N), a framework leveraging diffusion modeling to transform audio-conditioned Gaussian noise into drum events with associated velocities. This generative diffusion approach offers distinct advantages, including a flexible speed-accuracy trade-off and strong inpainting capabilities. However, the generation of binary onset and continuous velocity values presents a challenge for diffusion models, and to overcome this, we introduce an Annealed Pseudo-Huber loss to facilitate effective joint optimization. Finally, to augment low-level spectrogram features, we propose incorporating features extracted from music foundation models (MFMs), which capture high-level semantic information and enhance robustness to out-of-domain drum audio. Experimental results demonstrate that including MFM features significantly improves robustness and N2N establishes a new state-of-the-art performance across multiple ADT benchmarks.
>
---
#### [new 003] Guiding Audio Editing with Audio Language Model
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出SmartDJ，一种结合音频语言模型与扩散模型的立体声音频编辑框架，旨在解决现有方法依赖模板指令、不支持声明式编辑的问题。通过将高层指令分解为原子操作并执行，实现更自然、高质量的音频编辑。**

- **链接: [http://arxiv.org/pdf/2509.21625v1](http://arxiv.org/pdf/2509.21625v1)**

> **作者:** Zitong Lan; Yiduo Hao; Mingmin Zhao
>
> **摘要:** Audio editing plays a central role in VR/AR immersion, virtual conferencing, sound design, and other interactive media. However, recent generative audio editing models depend on template-like instruction formats and are restricted to mono-channel audio. These models fail to deal with declarative audio editing, where the user declares what the desired outcome should be, while leaving the details of editing operations to the system. We introduce SmartDJ, a novel framework for stereo audio editing that combines the reasoning capability of audio language models with the generative power of latent diffusion. Given a high-level instruction, SmartDJ decomposes it into a sequence of atomic edit operations, such as adding, removing, or spatially relocating events. These operations are then executed by a diffusion model trained to manipulate stereo audio. To support this, we design a data synthesis pipeline that produces paired examples of high-level instructions, atomic edit operations, and audios before and after each edit operation. Experiments demonstrate that SmartDJ achieves superior perceptual quality, spatial realism, and semantic alignment compared to prior audio editing methods. Demos are available at https://zitonglan.github.io/project/smartdj/smartdj.html.
>
---
#### [new 004] Shortcut Flow Matching for Speech Enhancement: Step-Invariant flows via single stage training
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对语音增强任务，旨在解决基于扩散模型的高计算成本问题。提出Shortcut Flow Matching方法，通过单阶段训练构建步长不变模型，实现高效实时语音去噪，兼顾生成质量和低延迟需求。**

- **链接: [http://arxiv.org/pdf/2509.21522v1](http://arxiv.org/pdf/2509.21522v1)**

> **作者:** Naisong Zhou; Saisamarth Rajesh Phaye; Milos Cernak; Tijana Stojkovic; Andy Pearce; Andrea Cavallaro; Andy Harper
>
> **备注:** 5 pages, 2 figures, submitted to ICASSP2026
>
> **摘要:** Diffusion-based generative models have achieved state-of-the-art performance for perceptual quality in speech enhancement (SE). However, their iterative nature requires numerous Neural Function Evaluations (NFEs), posing a challenge for real-time applications. On the contrary, flow matching offers a more efficient alternative by learning a direct vector field, enabling high-quality synthesis in just a few steps using deterministic ordinary differential equation~(ODE) solvers. We thus introduce Shortcut Flow Matching for Speech Enhancement (SFMSE), a novel approach that trains a single, step-invariant model. By conditioning the velocity field on the target time step during a one-stage training process, SFMSE can perform single, few, or multi-step denoising without any architectural changes or fine-tuning. Our results demonstrate that a single-step SFMSE inference achieves a real-time factor (RTF) of 0.013 on a consumer GPU while delivering perceptual quality comparable to a strong diffusion baseline requiring 60 NFEs. This work also provides an empirical analysis of the role of stochasticity in training and inference, bridging the gap between high-quality generative SE and low-latency constraints.
>
---
#### [new 005] Real-time implementation of vibrato transfer as an audio effect
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出一种实时实现音震（vibrato）转移的音频算法，通过延迟线将目标信号的音震模式转移到输入声音上。针对原算法计算复杂的问题，改进了实时性，引入高效基频估计和IIR滤波器，并扩展了幅度调制转移功能，应用于音频设计与合成控制。**

- **链接: [http://arxiv.org/pdf/2509.21544v1](http://arxiv.org/pdf/2509.21544v1)**

> **作者:** Jeremy Hyrkas
>
> **备注:** 4 pages, 4 figures, ICMC 2025
>
> **摘要:** An algorithm for deriving delay functions based on real examples of vibrato was recently introduced and can be used to perform a vibrato transfer, in which the vibrato pattern of a target signal is imparted onto an incoming sound using a delay line. The algorithm contains methods that computationally restrict a real-time implementation. Here, a real-time approximation is presented that incorporates an efficient fundamental frequency estimation algorithm and time-domain polyphase IIR filters that approximate an analytic signal. The vibrato transfer algorithm is further supplemented with a proposed method to transfer the amplitude modulation of the target sound, moving this method beyond the capabilities of typical delay-based vibrato effects. Modifications to the original algorithm for real-time use are detailed here and available as source code for an implementation as a VST plugin. This algorithm has applications as an audio effect in sound design, sound morphing, and real-time vibrato control of synthesized sounds.
>
---
#### [new 006] Golden Tonnetz
- **分类: cs.SD; eess.AS; 00A65**

- **简介: 该论文探索音乐与黄金比例的关系，提出“黄金Tonnetz”模型，用黄金三角形和补形表示大小调音阶及三和弦，并通过其变换实现音乐理论中的转调操作。**

- **链接: [http://arxiv.org/pdf/2509.21428v1](http://arxiv.org/pdf/2509.21428v1)**

> **作者:** Yusuke Imai
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Musical concepts have been represented by geometry with tones. For example, in the chromatic circle, the twelve tones are represented by twelve points on a circle, and in Tonnetz, the relationships among harmonies are represented by a triangular lattice. Recently, we have shown that several arrangements of tones on the regular icosahedron can be associated with chromatic scales, whole-tone scales, major tones, and minor tones through the golden ratio. Here, we investigate another type of connection between music and the golden ratio. We show that there exists an arrangement of 7 tones on a golden triangle that can represent a given major/minor scale and its tonic, dominant, and subdominant chords by golden triangles. By applying this finding, we propose "golden Tonnetz" which represents all the major/minor scales and triads by the golden triangles or gnomons and also represents relative, parallel, and leading-tone exchange transformations in Neo-Riemannian theory by transformations among the golden triangles and gnomons.
>
---
#### [new 007] Decoding Deception: Understanding Automatic Speech Recognition Vulnerabilities in Evasion and Poisoning Attacks
- **分类: cs.SD; cs.AI; cs.CR**

- **简介: 该论文研究自动语音识别系统在对抗样本攻击下的脆弱性，探讨白盒和黑盒攻击方法，并分析投毒攻击对模型性能的影响，旨在揭示系统安全漏洞并强调对抗安全的重要性。**

- **链接: [http://arxiv.org/pdf/2509.22060v1](http://arxiv.org/pdf/2509.22060v1)**

> **作者:** Aravindhan G; Yuvaraj Govindarajulu; Parin Shah
>
> **摘要:** Recent studies have demonstrated the vulnerability of Automatic Speech Recognition systems to adversarial examples, which can deceive these systems into misinterpreting input speech commands. While previous research has primarily focused on white-box attacks with constrained optimizations, and transferability based black-box attacks against commercial Automatic Speech Recognition devices, this paper explores cost efficient white-box attack and non transferability black-box adversarial attacks on Automatic Speech Recognition systems, drawing insights from approaches such as Fast Gradient Sign Method and Zeroth-Order Optimization. Further, the novelty of the paper includes how poisoning attack can degrade the performances of state-of-the-art models leading to misinterpretation of audio signals. Through experimentation and analysis, we illustrate how hybrid models can generate subtle yet impactful adversarial examples with very little perturbation having Signal Noise Ratio of 35dB that can be generated within a minute. These vulnerabilities of state-of-the-art open source model have practical security implications, and emphasize the need for adversarial security.
>
---
#### [new 008] Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出一种基于VLM的高可解释性、低成本图像到音乐生成框架，利用ABC记谱法和多模态RAG技术实现无需外部训练的高质量音乐生成，并通过注意力图提供结果解释。**

- **链接: [http://arxiv.org/pdf/2509.22378v1](http://arxiv.org/pdf/2509.22378v1)**

> **作者:** Zijian Zhao; Dian Jin; Zijing Zhou
>
> **摘要:** Recently, Image-to-Music (I2M) generation has garnered significant attention, with potential applications in fields such as gaming, advertising, and multi-modal art creation. However, due to the ambiguous and subjective nature of I2M tasks, most end-to-end methods lack interpretability, leaving users puzzled about the generation results. Even methods based on emotion mapping face controversy, as emotion represents only a singular aspect of art. Additionally, most learning-based methods require substantial computational resources and large datasets for training, hindering accessibility for common users. To address these challenges, we propose the first Vision Language Model (VLM)-based I2M framework that offers high interpretability and low computational cost. Specifically, we utilize ABC notation to bridge the text and music modalities, enabling the VLM to generate music using natural language. We then apply multi-modal Retrieval-Augmented Generation (RAG) and self-refinement techniques to allow the VLM to produce high-quality music without external training. Furthermore, we leverage the generated motivations in text and the attention maps from the VLM to provide explanations for the generated results in both text and image modalities. To validate our method, we conduct both human studies and machine evaluations, where our method outperforms others in terms of music quality and music-image consistency, indicating promising results. Our code is available at https://github.com/RS2002/Image2Music .
>
---
#### [new 009] Preserving Russek's "Summermood" Using Reality Check and a DeltaLab DL-4 Approximation
- **分类: cs.SD**

- **简介: 该论文属于音乐技术领域的作品保存任务，旨在解决因DeltaLab DL-4设备停产导致Antonio Russek的《Summermood》无法演出的问题。作者使用Pure Data开发了替代方案，模拟DL-4功能，并通过测试确保跨环境可演奏性。**

- **链接: [http://arxiv.org/pdf/2509.21560v1](http://arxiv.org/pdf/2509.21560v1)**

> **作者:** Jeremy Hyrkas; Pablo Dodero Carrillo; Teresa Díaz de Cossio Sánchez
>
> **备注:** 6 pages, 10 figures, Pure Data Max Conference 2025
>
> **摘要:** As a contribution towards ongoing efforts to maintain electroacoustic compositions for live performance, we present a collection of Pure Data patches to preserve and perform Antonio Russek's piece "Summermood" for bass flute and live electronics. The piece, originally written for the DeltaLab DL-4 delay rack unit, contains score markings specific to the DL-4. Here, we approximate the sound and unique functionality of the DL-4 in Pure Data, then refine our implementation to better match the unit on which the piece was performed by comparing settings from the score to two official recordings of the piece. The DL-4 emulation is integrated into a patch for live performance based on the Null Piece, and regression tested using the Reality Check framework for Pure Data. Using this library of patches, Summermood can be brought back into live rotation without the use of the now discontinued DL-4. The patches will be continuously tested to ensure that the piece is playable across computer environments and as the Pure Data programming language is updated.
>
---
#### [new 010] Lightweight Front-end Enhancement for Robust ASR via Frame Resampling and Sub-Band Pruning
- **分类: cs.SD**

- **简介: 该论文针对语音识别中噪声环境下的鲁棒性问题，提出一种轻量级语音增强前端方法。通过逐层帧重采样和渐进子带剪枝降低计算开销，在保持识别性能的同时显著减少计算资源消耗。**

- **链接: [http://arxiv.org/pdf/2509.21833v1](http://arxiv.org/pdf/2509.21833v1)**

> **作者:** Siyi Zhao; Wei Wang; Yanmin Qian
>
> **备注:** Proceedings of Interspeech
>
> **摘要:** Recent advancements in automatic speech recognition (ASR) have achieved notable progress, whereas robustness in noisy environments remains challenging. While speech enhancement (SE) front-ends are widely used to mitigate noise as a preprocessing step for ASR, they often introduce computational non-negligible overhead. This paper proposes optimizations to reduce SE computational costs without compromising ASR performance. Our approach integrates layer-wise frame resampling and progressive sub-band pruning. Frame resampling downsamples inputs within layers, utilizing residual connections to mitigate information loss. Simultaneously, sub-band pruning progressively excludes less informative frequency bands, further reducing computational demands. Extensive experiments on synthetic and real-world noisy datasets demonstrate that our system reduces SE computational overhead over 66 compared to the standard BSRNN, while maintaining strong ASR performance.
>
---
#### [new 011] Comprehend and Talk: Text to Speech Synthesis via Dual Language Modeling
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对基于大语言模型的文本到语音（TTS）系统存在的信息损失和生成不稳定问题，提出CaT-TTS框架。通过引入S3Codec编码器、双Transformer结构和MAPI推理策略，实现更稳健、语义驱动的高质量语音合成。**

- **链接: [http://arxiv.org/pdf/2509.22062v1](http://arxiv.org/pdf/2509.22062v1)**

> **作者:** Junjie Cao; Yichen Han; Ruonan Zhang; Xiaoyang Hao; Hongxiang Li; Shuaijiang Zhao; Yue Liu; Xiao-Ping Zhng
>
> **备注:** conference paper about TTS
>
> **摘要:** Existing Large Language Model (LLM) based autoregressive (AR) text-to-speech (TTS) systems, while achieving state-of-the-art quality, still face critical challenges. The foundation of this LLM-based paradigm is the discretization of the continuous speech waveform into a sequence of discrete tokens by neural audio codec. However, single codebook modeling is well suited to text LLMs, but suffers from significant information loss; hierarchical acoustic tokens, typically generated via Residual Vector Quantization (RVQ), often lack explicit semantic structure, placing a heavy learning burden on the model. Furthermore, the autoregressive process is inherently susceptible to error accumulation, which can degrade generation stability. To address these limitations, we propose CaT-TTS, a novel framework for robust and semantically-grounded zero-shot synthesis. First, we introduce S3Codec, a split RVQ codec that injects explicit linguistic features into its primary codebook via semantic distillation from a state-of-the-art ASR model, providing a structured representation that simplifies the learning task. Second, we propose an ``Understand-then-Generate'' dual-Transformer architecture that decouples comprehension from rendering. An initial ``Understanding'' Transformer models the cross-modal relationship between text and the audio's semantic tokens to form a high-level utterance plan. A subsequent ``Generation'' Transformer then executes this plan, autoregressively synthesizing hierarchical acoustic tokens. Finally, to enhance generation stability, we introduce Masked Audio Parallel Inference (MAPI), a nearly parameter-free inference strategy that dynamically guides the decoding process to mitigate local errors.
>
---
#### [new 012] MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出了MDAR，一个多场景动态音频推理基准，旨在评估AI在复杂、多源音频环境中的推理能力。现有基准多为静态单一场景，无法全面反映真实情况。MDAR包含3000个问答对，覆盖五类复杂推理任务，测试26种模型表现，揭示其在多场景推理中的局限性。**

- **链接: [http://arxiv.org/pdf/2509.22461v1](http://arxiv.org/pdf/2509.22461v1)**

> **作者:** Hui Li; Changhao Jiang; Hongyu Wang; Ming Zhang; Jiajun Sun; Zhixiong Yang; Yifei Cao; Shihan Dou; Xiaoran Fan; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** The ability to reason from audio, including speech, paralinguistic cues, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce MDAR, a benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. MDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on MDAR and observe that they exhibit limitations in complex reasoning tasks. On single-choice questions, Qwen2.5-Omni (open-source) achieves 76.67% accuracy, whereas GPT-4o Audio (closed-source) reaches 68.47%; however, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice and open-ended tasks. Across all three question types, no model achieves 80% performance. These findings underscore the unique challenges posed by MDAR and its value as a benchmark for advancing audio reasoning research.Code and benchmark can be found at https://github.com/luckyerr/MDAR.
>
---
#### [new 013] MusicWeaver: Coherent Long-Range and Editable Music Generation from a Beat-Aligned Structural Plan
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出MusicWeaver，用于可编辑的长程音乐生成。针对现有模型结构不连贯、编辑能力弱的问题，设计了基于节拍对齐结构计划的生成框架，包含规划器和扩散生成器，并引入评估指标SCS和EFS。**

- **链接: [http://arxiv.org/pdf/2509.21714v1](http://arxiv.org/pdf/2509.21714v1)**

> **作者:** Xuanchen Wang; Heng Wang; Weidong Cai
>
> **备注:** 5 pages, 1 figure. demo page: https://musicweaver.github.io/
>
> **摘要:** Current music generators capture local textures but often fail to model long-range structure, leading to off-beat outputs, weak section transitions, and limited editing capability. We present MusicWeaver, a music generation model conditioned on a beat-aligned structural plan. This plan serves as an editable intermediate between the input prompt and the generated music, preserving global form and enabling professional, localized edits. MusicWeaver consists of a planner, which translates prompts into a structural plan encoding musical form and compositional cues, and a diffusion-based generator, which synthesizes music under the plan's guidance. To assess generation and editing quality, we introduce two metrics: the Structure Coherence Score (SCS) for evaluating long-range form and timing, and the Edit Fidelity Score (EFS) for measuring the accuracy of realizing plan edits. Experiments demonstrate that MusicWeaver achieves state-of-the-art fidelity and controllability, producing music closer to human-composed works. Music results can be found on our project page: https://musicweaver.github.io/.
>
---
#### [new 014] Frustratingly Easy Zero-Day Audio DeepFake Detection via Retrieval Augmentation and Profile Matching
- **分类: cs.SD**

- **简介: 该论文针对零日音频DeepFake检测任务，提出一种无需训练的框架，利用知识表示、检索增强和语音特征匹配，有效应对新型伪造攻击，在DeepFake-Eval-2024上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.21728v1](http://arxiv.org/pdf/2509.21728v1)**

> **作者:** Xuechen Liu; Xin Wang; Junichi Yamagishi
>
> **摘要:** Modern audio deepfake detectors using foundation models and large training datasets have achieved promising detection performance. However, they struggle with zero-day attacks, where the audio samples are generated by novel synthesis methods that models have not seen from reigning training data. Conventional approaches against such attacks require fine-tuning the detectors, which can be problematic when prompt response is required. This study introduces a training-free framework for zero-day audio deepfake detection based on knowledge representations, retrieval augmentation, and voice profile matching. Based on the framework, we propose simple yet effective knowledge retrieval and ensemble methods that achieve performance comparable to fine-tuned models on DeepFake-Eval-2024, without any additional model-wise training. We also conduct ablation studies on retrieval pool size and voice profile attributes, validating their relevance to the system efficacy.
>
---
#### [new 015] Cross-Dialect Bird Species Recognition with Dialect-Calibrated Augmentation
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究跨方言鸟类叫声识别任务，旨在解决方言差异影响自动识别的问题。基于TDNN框架，提出频率敏感归一化、对抗训练和DCA多级增强方法，提升了跨区域识别准确率并保持区域内性能。**

- **链接: [http://arxiv.org/pdf/2509.22317v1](http://arxiv.org/pdf/2509.22317v1)**

> **作者:** Jiani Ding; Qiyang Sun; Alican Akman; Björn W. Schuller
>
> **摘要:** Dialect variation hampers automatic recognition of bird calls collected by passive acoustic monitoring. We address the problem on DB3V, a three-region, ten-species corpus of 8-s clips, and propose a deployable framework built on Time-Delay Neural Networks (TDNNs). Frequency-sensitive normalisation (Instance Frequency Normalisation and a gated Relaxed-IFN) is paired with gradient-reversal adversarial training to learn region-invariant embeddings. A multi-level augmentation scheme combines waveform perturbations, Mixup for rare classes, and CycleGAN transfer that synthesises Region 2 (Interior Plains)-style audio, , with Dialect-Calibrated Augmentation (DCA) softly down-weighting synthetic samples to limit artifacts. The complete system lifts cross-dialect accuracy by up to twenty percentage points over baseline TDNNs while preserving in-region performance. Grad-CAM and LIME analyses show that robust models concentrate on stable harmonic bands, providing ecologically meaningful explanations. The study demonstrates that lightweight, transparent, and dialect-resilient bird-sound recognition is attainable.
>
---
#### [new 016] From Coarse to Fine: Recursive Audio-Visual Semantic Enhancement for Speech Separation
- **分类: cs.SD**

- **简介: 该论文针对语音分离任务，旨在解决视觉信息利用不足的问题。提出CSFNet，通过递归语义增强框架，结合粗分与细分阶段，并设计感知融合模块和多尺度分离网络，有效提升音视融合语音分离性能。**

- **链接: [http://arxiv.org/pdf/2509.22425v1](http://arxiv.org/pdf/2509.22425v1)**

> **作者:** Ke Xue; Rongfei Fan; Lixin; Dawei Zhao; Chao Zhu; Han Hu
>
> **摘要:** Audio-visual speech separation aims to isolate each speaker's clean voice from mixtures by leveraging visual cues such as lip movements and facial features. While visual information provides complementary semantic guidance, existing methods often underexploit its potential by relying on static visual representations. In this paper, we propose CSFNet, a Coarse-to-Separate-Fine Network that introduces a recursive semantic enhancement paradigm for more effective separation. CSFNet operates in two stages: (1) Coarse Separation, where a first-pass estimation reconstructs a coarse audio waveform from the mixture and visual input; and (2) Fine Separation, where the coarse audio is fed back into an audio-visual speech recognition (AVSR) model together with the visual stream. This recursive process produces more discriminative semantic representations, which are then used to extract refined audio. To further exploit these semantics, we design a speaker-aware perceptual fusion block to encode speaker identity across modalities, and a multi-range spectro-temporal separation network to capture both local and global time-frequency patterns. Extensive experiments on three benchmark datasets and two noisy datasets show that CSFNet achieves state-of-the-art (SOTA) performance, with substantial coarse-to-fine improvements, validating the necessity and effectiveness of our recursive semantic enhancement framework.
>
---
#### [new 017] A Parallel Ultra-Low Power Silent Speech Interface based on a Wearable, Fully-dry EMG Neckband
- **分类: eess.SY; cs.SD; cs.SY; eess.AS; eess.SP**

- **简介: 该论文提出一种基于可穿戴全干式EMG颈带的低功耗无声语音接口系统，用于无声语音识别。系统集成14个差分EMG通道，测试表明在不同条件下具有较高识别准确率，展示了其鲁棒性和能效优势。**

- **链接: [http://arxiv.org/pdf/2509.21964v1](http://arxiv.org/pdf/2509.21964v1)**

> **作者:** Fiona Meier; Giusy Spacone; Sebastian Frey; Luca Benini; Andrea Cossettini
>
> **备注:** 4 pages, 4 figures
>
> **摘要:** We present a wearable, fully-dry, and ultra-low power EMG system for silent speech recognition, integrated into a textile neckband to enable comfortable, non-intrusive use. The system features 14 fully-differential EMG channels and is based on the BioGAP-Ultra platform for ultra-low power (22 mW) biosignal acquisition and wireless transmission. We evaluate its performance on eight speech commands under both vocalized and silent articulation, achieving average classification accuracies of 87$\pm$3% and 68$\pm$3% respectively, with a 5-fold CV approach. To mimic everyday-life conditions, we introduce session-to-session variability by repositioning the neckband between sessions, achieving leave-one-session-out accuracies of 64$\pm$18% and 54$\pm$7% for the vocalized and silent experiments, respectively. These results highlight the robustness of the proposed approach and the promise of energy-efficient silent-speech decoding.
>
---
#### [new 018] VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.SD**

- **简介: 该论文提出了VoiceAssistant-Eval，一个用于评估语音助手在听、说、视三方面能力的综合基准，包含10,497个任务示例。通过评估21个开源模型和GPT-4o-Audio，揭示了当前模型在音频理解、多模态处理等方面的不足，为下一代AI助手的研发提供了指导框架。**

- **链接: [http://arxiv.org/pdf/2509.22651v1](http://arxiv.org/pdf/2509.22651v1)**

> **作者:** Ke Wang; Houxing Ren; Zimu Lu; Mingjie Zhan; Hongsheng Li
>
> **摘要:** The growing capabilities of large language models and multimodal systems have spurred interest in voice-first AI assistants, yet existing benchmarks are inadequate for evaluating the full range of these systems' capabilities. We introduce VoiceAssistant-Eval, a comprehensive benchmark designed to assess AI assistants across listening, speaking, and viewing. VoiceAssistant-Eval comprises 10,497 curated examples spanning 13 task categories. These tasks include natural sounds, music, and spoken dialogue for listening; multi-turn dialogue, role-play imitation, and various scenarios for speaking; and highly heterogeneous images for viewing. To demonstrate its utility, we evaluate 21 open-source models and GPT-4o-Audio, measuring the quality of the response content and speech, as well as their consistency. The results reveal three key findings: (1) proprietary models do not universally outperform open-source models; (2) most models excel at speaking tasks but lag in audio understanding; and (3) well-designed smaller models can rival much larger ones. Notably, the mid-sized Step-Audio-2-mini (7B) achieves more than double the listening accuracy of LLaMA-Omni2-32B-Bilingual. However, challenges remain: multimodal (audio plus visual) input and role-play voice imitation tasks are difficult for current models, and significant gaps persist in robustness and safety alignment. VoiceAssistant-Eval identifies these gaps and establishes a rigorous framework for evaluating and guiding the development of next-generation AI assistants. Code and data will be released at https://mathllm.github.io/VoiceAssistantEval/ .
>
---
#### [new 019] AUV: Teaching Audio Universal Vector Quantization with Single Nested Codebook
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出AUV，一种基于单码本的通用音频编码器，可在700 bps下高质量重建语音和音乐等混合域音频。通过单阶段训练与教师模型蒸馏，实现多领域音频的统一向量量化。**

- **链接: [http://arxiv.org/pdf/2509.21968v1](http://arxiv.org/pdf/2509.21968v1)**

> **作者:** Yushen Chen; Kai Hu; Long Zhou; Shulin Feng; Xusheng Yang; Hangting Chen; Xie Chen
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We propose AUV, a unified neural audio codec with a single codebook, which enables a favourable reconstruction of speech and further extends to general audio, including vocal, music, and sound. AUV is capable of tackling any 16 kHz mixed-domain audio segment at bit rates around 700 bps. To accomplish this, we guide the matryoshka codebook with nested domain-specific partitions, assigned with corresponding teacher models to perform distillation, all in a single-stage training. A conformer-style encoder-decoder architecture with STFT features as audio representation is employed, yielding better audio quality. Comprehensive evaluations demonstrate that AUV exhibits comparable audio reconstruction ability to state-of-the-art domain-specific single-layer quantizer codecs, showcasing the potential of audio universal vector quantization with a single codebook. The pre-trained model and demo samples are available at https://swivid.github.io/AUV/.
>
---
#### [new 020] Multi-Speaker DOA Estimation in Binaural Hearing Aids using Deep Learning and Speaker Count Fusion
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究双耳助听器中的多说话人方位估计（DOA）任务，旨在提升嘈杂环境下的语音提取效果。提出将声源数量信息作为辅助特征，通过不同融合策略整合到CRNN模型中，实验表明使用真实声源数的晚期融合可显著提升DOA性能。**

- **链接: [http://arxiv.org/pdf/2509.21382v1](http://arxiv.org/pdf/2509.21382v1)**

> **作者:** Farnaz Jazaeri; Homayoun Kamkar-Parsi; François Grondin; Martin Bouchard
>
> **备注:** 5 pages, 2 figures, submitted to IEEE ICASSP 2026
>
> **摘要:** For extracting a target speaker voice, direction-of-arrival (DOA) estimation is crucial for binaural hearing aids operating in noisy, multi-speaker environments. Among the solutions developed for this task, a deep learning convolutional recurrent neural network (CRNN) model leveraging spectral phase differences and magnitude ratios between microphone signals is a popular option. In this paper, we explore adding source-count information for multi-sources DOA estimation. The use of dual-task training with joint multi-sources DOA estimation and source counting is first considered. We then consider using the source count as an auxiliary feature in a standalone DOA estimation system, where the number of active sources (0, 1, or 2+) is integrated into the CRNN architecture through early, mid, and late fusion strategies. Experiments using real binaural recordings are performed. Results show that the dual-task training does not improve DOA estimation performance, although it benefits source-count prediction. However, a ground-truth (oracle) source count used as an auxiliary feature significantly enhances standalone DOA estimation performance, with late fusion yielding up to 14% higher average F1-scores over the baseline CRNN. This highlights the potential of using source-count estimation for robust DOA estimation in binaural hearing aids.
>
---
#### [new 021] Speaker Anonymisation for Speech-based Suicide Risk Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究语音匿名化在自杀风险检测中的应用，属于语音隐私保护任务。旨在解决语音数据泄露导致身份识别的问题，系统评估多种匿名化方法，在保护隐私的同时保持检测性能。**

- **链接: [http://arxiv.org/pdf/2509.22148v1](http://arxiv.org/pdf/2509.22148v1)**

> **作者:** Ziyun Cui; Sike Jia; Yang Lin; Yinan Duan; Diyang Qu; Runsen Chen; Chao Zhang; Chang Lei; Wen Wu
>
> **摘要:** Adolescent suicide is a critical global health issue, and speech provides a cost-effective modality for automatic suicide risk detection. Given the vulnerable population, protecting speaker identity is particularly important, as speech itself can reveal personally identifiable information if the data is leaked or maliciously exploited. This work presents the first systematic study of speaker anonymisation for speech-based suicide risk detection. A broad range of anonymisation methods are investigated, including techniques based on traditional signal processing, neural voice conversion, and speech synthesis. A comprehensive evaluation framework is built to assess the trade-off between protecting speaker identity and preserving information essential for suicide risk detection. Results show that combining anonymisation methods that retain complementary information yields detection performance comparable to that of original speech, while achieving protection of speaker identity for vulnerable populations.
>
---
#### [new 022] Thinking with Sound: Audio Chain-of-Thought Enables Multimodal Reasoning in Large Audio-Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文针对复杂声学场景下音频推理任务中大模型表现不佳的问题，提出TwS框架，通过结合语言推理与实时音频分析提升鲁棒性，并构建MELD-Hard1k基准进行评估。**

- **链接: [http://arxiv.org/pdf/2509.21749v1](http://arxiv.org/pdf/2509.21749v1)**

> **作者:** Zhen Xiong; Yujun Cai; Zhecheng Li; Junsong Yuan; Yiwei Wang
>
> **摘要:** Recent Large Audio-Language Models (LALMs) have shown strong performance on various audio understanding tasks such as speech translation and Audio Q\&A. However, they exhibit significant limitations on challenging audio reasoning tasks in complex acoustic scenarios. These situations would greatly benefit from the use of acoustic tools like noise suppression, source separation, and precise temporal alignment, but current LALMs lack access to such tools. To address this limitation, we introduce Thinking-with-Sound (TwS), a framework that equips LALMs with Audio CoT by combining linguistic reasoning with on-the-fly audio-domain analysis. Unlike existing approaches that treat audio as static input, TwS enables models to actively think with audio signals, performing numerical analysis and digital manipulation through multimodal reasoning. To evaluate this approach, we construct MELD-Hard1k, a new robustness benchmark created by introducing various acoustic perturbations. Experiments reveal that state-of-the-art LALMs suffer dramatic performance degradation on MELD-Hard1k, with accuracy dropping by more than $50\%$ compared to clean audio. TwS achieves substantial improvements in robustness, demonstrating both effectiveness and scalability: small models gain $24.73\%$ absolute accuracy, with improvements scaling consistently up to $36.61\%$ for larger models. Our findings demonstrate that Audio CoT can significantly enhance robustness without retraining, opening new directions for developing more robust audio understanding systems.
>
---
#### [new 023] High-Quality Sound Separation Across Diverse Categories via Visually-Guided Generative Modeling
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出DAVIS，一种基于扩散模型和流匹配的视听音源分离框架，旨在解决传统方法在分离多类别声音时质量受限的问题。通过生成式学习直接合成目标频谱，提升分离效果，并在标准数据集上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2509.22063v1](http://arxiv.org/pdf/2509.22063v1)**

> **作者:** Chao Huang; Susan Liang; Yapeng Tian; Anurag Kumar; Chenliang Xu
>
> **备注:** Accepted to IJCV
>
> **摘要:** We propose DAVIS, a Diffusion-based Audio-VIsual Separation framework that solves the audio-visual sound source separation task through generative learning. Existing methods typically frame sound separation as a mask-based regression problem, achieving significant progress. However, they face limitations in capturing the complex data distribution required for high-quality separation of sounds from diverse categories. In contrast, DAVIS circumvents these issues by leveraging potent generative modeling paradigms, specifically Denoising Diffusion Probabilistic Models (DDPM) and the more recent Flow Matching (FM), integrated within a specialized Separation U-Net architecture. Our framework operates by synthesizing the desired separated sound spectrograms directly from a noise distribution, conditioned concurrently on the mixed audio input and associated visual information. The inherent nature of its generative objective makes DAVIS particularly adept at producing high-quality sound separations for diverse sound categories. We present comparative evaluations of DAVIS, encompassing both its DDPM and Flow Matching variants, against leading methods on the standard AVE and MUSIC datasets. The results affirm that both variants surpass existing approaches in separation quality, highlighting the efficacy of our generative framework for tackling the audio-visual source separation task.
>
---
#### [new 024] Speak Your Mind: The Speech Continuation Task as a Probe of Voice-Based Model Bias
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究语音续说任务（SC）中语音基础模型的偏见问题，通过分析性别和发音类型对续说结果的影响，评估了三个模型在语音身份保持、语义连贯性及文本偏见方面的表现，揭示了系统性语音质量偏差。**

- **链接: [http://arxiv.org/pdf/2509.22061v1](http://arxiv.org/pdf/2509.22061v1)**

> **作者:** Shree Harsha Bokkahalli Satish; Harm Lameris; Olivier Perrotin; Gustav Eje Henter; Éva Székely
>
> **备注:** 6 pages, 1 figure, Submitted to IEEE ICASSP 2026
>
> **摘要:** Speech Continuation (SC) is the task of generating a coherent extension of a spoken prompt while preserving both semantic context and speaker identity. Because SC is constrained to a single audio stream, it offers a more direct setting for probing biases in speech foundation models than dialogue does. In this work we present the first systematic evaluation of bias in SC, investigating how gender and phonation type (breathy, creaky, end-creak) affect continuation behaviour. We evaluate three recent models: SpiritLM (base and expressive), VAE-GSLM, and SpeechGPT across speaker similarity, voice quality preservation, and text-based bias metrics. Results show that while both speaker similarity and coherence remain a challenge, textual evaluations reveal significant model and gender interactions: once coherence is sufficiently high (for VAE-GSLM), gender effects emerge on text-metrics such as agency and sentence polarity. In addition, continuations revert toward modal phonation more strongly for female prompts than for male ones, revealing a systematic voice-quality bias. These findings highlight SC as a controlled probe of socially relevant representational biases in speech foundation models, and suggest that it will become an increasingly informative diagnostic as continuation quality improves.
>
---
#### [new 025] HuLA: Prosody-Aware Anti-Spoofing with Multi-Task Learning for Expressive and Emotional Synthetic Speech
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出HuLA，一个用于合成语音检测的多任务学习框架。针对现有系统对情感和表达性合成语音防御不足的问题，通过引入韵律感知机制（如F0预测和清浊音分类），结合自监督学习，提升系统在复杂攻击下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.21676v1](http://arxiv.org/pdf/2509.21676v1)**

> **作者:** Aurosweta Mahapatra; Ismail Rasim Ulgen; Berrak Sisman
>
> **备注:** Submitted to IEEE Transactions on Affective Computing
>
> **摘要:** Current anti-spoofing systems remain vulnerable to expressive and emotional synthetic speech, since they rarely leverage prosody as a discriminative cue. Prosody is central to human expressiveness and emotion, and humans instinctively use prosodic cues such as F0 patterns and voiced/unvoiced structure to distinguish natural from synthetic speech. In this paper, we propose HuLA, a two-stage prosody-aware multi-task learning framework for spoof detection. In Stage 1, a self-supervised learning (SSL) backbone is trained on real speech with auxiliary tasks of F0 prediction and voiced/unvoiced classification, enhancing its ability to capture natural prosodic variation similar to human perceptual learning. In Stage 2, the model is jointly optimized for spoof detection and prosody tasks on both real and synthetic data, leveraging prosodic awareness to detect mismatches between natural and expressive synthetic speech. Experiments show that HuLA consistently outperforms strong baselines on challenging out-of-domain dataset, including expressive, emotional, and cross-lingual attacks. These results demonstrate that explicit prosodic supervision, combined with SSL embeddings, substantially improves robustness against advanced synthetic speech attacks.
>
---
#### [new 026] WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出WAVE，一种基于多模态大模型的统一音频-视觉嵌入方法。旨在解决动态模态（如音视频）表示不足的问题，通过层次特征融合和联合多任务训练，实现任意模态间检索与指令感知嵌入生成，提升跨模态应用效果。**

- **链接: [http://arxiv.org/pdf/2509.21990v1](http://arxiv.org/pdf/2509.21990v1)**

> **作者:** Changli Tang; Qinfan Xiao; Ke Mei; Tianyi Wang; Fengyun Rao; Chao Zhang
>
> **摘要:** While embeddings from multimodal large language models (LLMs) excel as general-purpose representations, their application to dynamic modalities like audio and video remains underexplored. We introduce WAVE (\textbf{u}nified \& \textbf{v}ersatile \textbf{a}udio-\textbf{v}isual \textbf{e}mbeddings), the first LLM-based embedding that creates a unified representation space for text, audio, and video modalities. WAVE employs a novel hierarchical feature fusion strategy and a joint multi-modal, multi-task training approach to enable two key capabilities: any-to-any cross-modal retrieval and the generation of prompt-aware embeddings tailored to user instructions. Experimentally, WAVE sets a new state-of-the-art on the MMEB-v2 video benchmark and achieves superior results in audio and video-to-audio retrieval. Its prompt-aware nature also yields remarkable performance in multimodal question answering, significantly outperforming existing embedding models. Ablation studies validate our joint training strategy, demonstrating improved performance across all modalities. With a newly introduced benchmark for versatile audio-visual learning, WAVE opens up broad possibilities for cross-modal, any-to-any applications. Our code, checkpoints, and data will be released.
>
---
#### [new 027] AUDDT: Audio Unified Deepfake Detection Benchmark Toolkit
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出了AUDDT，一个用于音频深度伪造检测的开源基准工具包。针对现有检测模型在数据集泛化能力不足的问题，系统梳理了28个数据集，并自动化评估预训练检测器的性能，揭示其在不同条件下的表现差异和局限性。**

- **链接: [http://arxiv.org/pdf/2509.21597v1](http://arxiv.org/pdf/2509.21597v1)**

> **作者:** Yi Zhu; Heitor R. Guimarães; Arthur Pimentel; Tiago Falk
>
> **摘要:** With the prevalence of artificial intelligence (AI)-generated content, such as audio deepfakes, a large body of recent work has focused on developing deepfake detection techniques. However, most models are evaluated on a narrow set of datasets, leaving their generalization to real-world conditions uncertain. In this paper, we systematically review 28 existing audio deepfake datasets and present an open-source benchmarking toolkit called AUDDT (https://github.com/MuSAELab/AUDDT). The goal of this toolkit is to automate the evaluation of pretrained detectors across these 28 datasets, giving users direct feedback on the advantages and shortcomings of their deepfake detectors. We start by showcasing the usage of the developed toolkit, the composition of our benchmark, and the breakdown of different deepfake subgroups. Next, using a widely adopted pretrained deepfake detector, we present in- and out-of-domain detection results, revealing notable differences across conditions and audio manipulation types. Lastly, we also analyze the limitations of these existing datasets and their gap relative to practical deployment scenarios.
>
---
## 更新

#### [replaced 001] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v5](http://arxiv.org/pdf/2505.14874v5)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Proceedings of Interspeech
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 002] VocalAgent: Large Language Models for Vocal Health Diagnostics with Safety-Aware Evaluation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13577v3](http://arxiv.org/pdf/2505.13577v3)**

> **作者:** Yubin Kim; Taehan Kim; Wonjune Kang; Eugene Park; Joonsik Yoon; Dongjae Lee; Xin Liu; Daniel McDuff; Hyeonhoon Lee; Cynthia Breazeal; Hae Won Park
>
> **备注:** Accepted by Proceedings of Interspeech 2025; Website: https://han811.github.io/VocalAgent2025/
>
> **摘要:** Vocal health plays a crucial role in peoples' lives, significantly impacting their communicative abilities and interactions. However, despite the global prevalence of voice disorders, many lack access to convenient diagnosis and treatment. This paper introduces VocalAgent, an audio large language model (LLM) to address these challenges through vocal health diagnosis. We leverage Qwen-Audio-Chat fine-tuned on three datasets collected in-situ from hospital patients, and present a multifaceted evaluation framework encompassing a safety assessment to mitigate diagnostic biases, cross-lingual performance analysis, and modality ablation studies. VocalAgent demonstrates superior accuracy on voice disorder classification compared to state-of-the-art baselines. Its LLM-based method offers a scalable solution for broader adoption of health diagnostics, while underscoring the importance of ethical and technical validation.
>
---
#### [replaced 003] CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.02863v2](http://arxiv.org/pdf/2506.02863v2)**

> **作者:** Helin Wang; Jiarui Hai; Dading Chong; Karan Thakkar; Tiantian Feng; Dongchao Yang; Junhyeok Lee; Thomas Thebaud; Laureano Moro Velazquez; Jesus Villalba; Zengyi Qin; Shrikanth Narayanan; Mounya Elhiali; Najim Dehak
>
> **摘要:** Recent advancements in generative artificial intelligence have significantly transformed the field of style-captioned text-to-speech synthesis (CapTTS). However, adapting CapTTS to real-world applications remains challenging due to the lack of standardized, comprehensive datasets and limited research on downstream tasks built upon CapTTS. To address these gaps, we introduce CapSpeech, a new benchmark designed for a series of CapTTS-related tasks, including style-captioned text-to-speech synthesis with sound events (CapTTS-SE), accent-captioned TTS (AccCapTTS), emotion-captioned TTS (EmoCapTTS), and text-to-speech synthesis for chat agent (AgentTTS). CapSpeech comprises over 10 million machine-annotated audio-caption pairs and nearly 0.36 million human-annotated audio-caption pairs. In addition, we introduce two new datasets collected and recorded by a professional voice actor and experienced audio engineers, specifically for the AgentTTS and CapTTS-SE tasks. Alongside the datasets, we conduct comprehensive experiments using both autoregressive and non-autoregressive models on CapSpeech. Our results demonstrate high-fidelity and highly intelligible speech synthesis across a diverse range of speaking styles. To the best of our knowledge, CapSpeech is the largest available dataset offering comprehensive annotations for CapTTS-related tasks. The experiments and findings further provide valuable insights into the challenges of developing CapTTS systems.
>
---
#### [replaced 004] FakeSound2: A Benchmark for Explainable and Generalizable Deepfake Sound Detection
- **分类: cs.SD; eess.AS; 68Txx; I.2**

- **链接: [http://arxiv.org/pdf/2509.17162v2](http://arxiv.org/pdf/2509.17162v2)**

> **作者:** Zeyu Xie; Yaoyun Zhang; Xuenan Xu; Yongkang Yin; Chenxing Li; Mengyue Wu; Yuexian Zou
>
> **摘要:** The rapid development of generative audio raises ethical and security concerns stemming from forged data, making deepfake sound detection an important safeguard against the malicious use of such technologies. Although prior studies have explored this task, existing methods largely focus on binary classification and fall short in explaining how manipulations occur, tracing where the sources originated, or generalizing to unseen sources-thereby limiting the explainability and reliability of detection. To address these limitations, we present FakeSound2, a benchmark designed to advance deepfake sound detection beyond binary accuracy. FakeSound2 evaluates models across three dimensions: localization, traceability, and generalization, covering 6 manipulation types and 12 diverse sources. Experimental results show that although current systems achieve high classification accuracy, they struggle to recognize forged pattern distributions and provide reliable explanations. By highlighting these gaps, FakeSound2 establishes a comprehensive benchmark that reveals key challenges and aims to foster robust, explainable, and generalizable approaches for trustworthy audio authentication.
>
---
#### [replaced 005] Hierarchical Graph Neural Network for Compressed Speech Steganalysis
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.21591v2](http://arxiv.org/pdf/2507.21591v2)**

> **作者:** Mustapha Hemis; Hamza Kheddar; Mohamed Chahine Ghanem; Bachir Boudraa
>
> **摘要:** Steganalysis methods based on deep learning (DL) often struggle with computational complexity and challenges in generalizing across different datasets. Incorporating a graph neural network (GNN) into steganalysis schemes enables the leveraging of relational data for improved detection accuracy and adaptability. This paper presents the first application of a Graph Neural Network (GNN), specifically the GraphSAGE architecture, for steganalysis of compressed voice over IP (VoIP) speech streams. The method involves straightforward graph construction from VoIP streams and employs GraphSAGE to capture hierarchical steganalysis information, including both fine grained details and high level patterns, thereby achieving high detection accuracy. Experimental results demonstrate that the developed approach performs well in uncovering quantization index modulation (QIM)-based steganographic patterns in VoIP signals. It achieves detection accuracy exceeding 98 percent even for short 0.5 second samples, and 95.17 percent accuracy under challenging conditions with low embedding rates, representing an improvement of 2.8 percent over the best performing state of the art methods. Furthermore, the model exhibits superior efficiency, with an average detection time as low as 0.016 seconds for 0.5-second samples an improvement of 0.003 seconds. This makes it efficient for online steganalysis tasks, providing a superior balance between detection accuracy and efficiency under the constraint of short samples with low embedding rates.
>
---
#### [replaced 006] SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.20802v2](http://arxiv.org/pdf/2509.20802v2)**

> **作者:** Tan Dat Nguyen; Jaehun Kim; Ji-Hoon Kim; Shukjae Choi; Youshin Lim; Joon Son Chung
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
>
---
#### [replaced 007] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.15176v4](http://arxiv.org/pdf/2408.15176v4)**

> **作者:** Longshen Ou; Jingwei Zhao; Ziyu Wang; Gus Xia; Qihao Liang; Torin Hopkins Ye Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present a unified framework for automatic multitrack music arrangement that enables a single pre-trained symbolic music model to handle diverse arrangement scenarios, including reinterpretation, simplification, and additive generation. At its core is a segment-level reconstruction objective operating on token-level disentangled content and style, allowing for flexible any-to-any instrumentation transformations at inference time. To support track-wise modeling, we introduce REMI-z, a structured tokenization scheme for multitrack symbolic music that enhances modeling efficiency and effectiveness for both arrangement tasks and unconditional generation. Our method outperforms task-specific state-of-the-art models on representative tasks in different arrangement scenarios -- band arrangement, piano reduction, and drum arrangement, in both objective metrics and perceptual evaluations. Taken together, our framework demonstrates strong generality and suggests broader applicability in symbolic music-to-music transformation.
>
---
#### [replaced 008] Xi+: Uncertainty Supervision for Robust Speaker Embedding
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.05993v2](http://arxiv.org/pdf/2509.05993v2)**

> **作者:** Junjie Li; Kong Aik Lee; Duc-Tuan Truong; Tianchi Liu; Man-Wai Mak
>
> **摘要:** There are various factors that can influence the performance of speaker recognition systems, such as emotion, language and other speaker-related or context-related variations. Since individual speech frames do not contribute equally to the utterance-level representation, it is essential to estimate the importance or reliability of each frame. The xi-vector model addresses this by assigning different weights to frames based on uncertainty estimation. However, its uncertainty estimation model is implicitly trained through classification loss alone and does not consider the temporal relationships between frames, which may lead to suboptimal supervision. In this paper, we propose an improved architecture, xi+. Compared to xi-vector, xi+ incorporates a temporal attention module to capture frame-level uncertainty in a context-aware manner. In addition, we introduce a novel loss function, Stochastic Variance Loss, which explicitly supervises the learning of uncertainty. Results demonstrate consistent performance improvements of about 10\% on the VoxCeleb1-O set and 11\% on the NIST SRE 2024 evaluation set.
>
---
#### [replaced 009] Description and Discussion on DCASE 2025 Challenge Task 2: First-shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10097v2](http://arxiv.org/pdf/2506.10097v2)**

> **作者:** Tomoya Nishida; Noboru Harada; Daisuke Niizumi; Davide Albertini; Roberto Sannino; Simone Pradolini; Filippo Augusti; Keisuke Imoto; Kota Dohi; Harsh Purohit; Takashi Endo; Yohei Kawaguchi
>
> **备注:** Accepted to DCASE Workshop 2025. this article draws heavily from arXiv:2406.07250v1
>
> **摘要:** This paper introduces the task description for the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge Task 2, titled "First-shot unsupervised anomalous sound detection (ASD) for machine condition monitoring". Building on the DCASE 2024 Challenge Task 2, this task is structured as a first-shot problem within a domain generalization framework. The primary objective of the first-shot approach is to facilitate the rapid deployment of ASD systems for new machine types without requiring machine-specific hyperparameter tunings. For DCASE 2025 Challenge Task 2, sounds from previously unseen machine types have been collected and provided as the evaluation dataset. We received 119 submissions from 35 teams, and an analysis of these submissions has been made in this paper. Analysis showed that various approaches can all be competitive, such as fine-tuning pre-trained models, using frozen pre-trained models, and training small models from scratch, when combined with appropriate cost functions, anomaly score normalization, and use of clean machine and noise sounds.
>
---
#### [replaced 010] On the Within-class Variation Issue in Alzheimer's Disease Detection
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2409.16322v3](http://arxiv.org/pdf/2409.16322v3)**

> **作者:** Jiawen Kang; Dongrui Han; Lingwei Meng; Jingyan Zhou; Jinchao Li; Xixin Wu; Helen Meng
>
> **备注:** Accepted for publication in Proc. of Interspeech 2025 conference. Note: this is an extended version of the conference paper, with an additional section included
>
> **摘要:** Alzheimer's Disease (AD) detection employs machine learning classification models to distinguish between individuals with AD and those without. Different from conventional classification tasks, we identify within-class variation as a critical challenge in AD detection: individuals with AD exhibit a spectrum of cognitive impairments. Therefore, simplistic binary AD classification may overlook two crucial aspects: within-class heterogeneity and instance-level imbalance. In this work, we found using a sample score estimator can generate sample-specific soft scores aligning with cognitive scores. We subsequently propose two simple yet effective methods: Soft Target Distillation (SoTD) and Instance-level Re-balancing (InRe), targeting two problems respectively. Based on the ADReSS and CU-MARVEL corpora, we demonstrated and analyzed the advantages of the proposed approaches in detection performance. These findings provide insights for developing robust and reliable AD detection models.
>
---
#### [replaced 011] Audio Super-Resolution with Latent Bridge Models
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.17609v2](http://arxiv.org/pdf/2509.17609v2)**

> **作者:** Chang Li; Zehua Chen; Liyuan Wang; Jun Zhu
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Audio super-resolution (SR), i.e., upsampling the low-resolution (LR) waveform to the high-resolution (HR) version, has recently been explored with diffusion and bridge models, while previous methods often suffer from sub-optimal upsampling quality due to their uninformative generation prior. Towards high-quality audio super-resolution, we present a new system with latent bridge models (LBMs), where we compress the audio waveform into a continuous latent space and design an LBM to enable a latent-to-latent generation process that naturally matches the LR-toHR upsampling process, thereby fully exploiting the instructive prior information contained in the LR waveform. To further enhance the training results despite the limited availability of HR samples, we introduce frequency-aware LBMs, where the prior and target frequency are taken as model input, enabling LBMs to explicitly learn an any-to-any upsampling process at the training stage. Furthermore, we design cascaded LBMs and present two prior augmentation strategies, where we make the first attempt to unlock the audio upsampling beyond 48 kHz and empower a seamless cascaded SR process, providing higher flexibility for audio post-production. Comprehensive experimental results evaluated on the VCTK, ESC-50, Song-Describer benchmark datasets and two internal testsets demonstrate that we achieve state-of-the-art objective and perceptual quality for any-to-48kHz SR across speech, audio, and music signals, as well as setting the first record for any-to-192kHz audio SR. Demo at https://AudioLBM.github.io/.
>
---
#### [replaced 012] Scaling to Multimodal and Multichannel Heart Sound Classification: Fine-Tuning Wav2Vec 2.0 with Synthetic and Augmented Biosignals
- **分类: cs.SD; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2509.11606v2](http://arxiv.org/pdf/2509.11606v2)**

> **作者:** Milan Marocchi; Matthew Fynn; Kayapanda Mandana; Yue Rong
>
> **备注:** 35 pages, 37 figures, 19 tables
>
> **摘要:** Cardiovascular diseases (CVDs) are the leading cause of death worldwide, accounting for approximately 17.9 million deaths each year. Early detection is critical, creating a demand for accurate and inexpensive pre-screening methods. Deep learning has recently been applied to classify abnormal heart sounds indicative of CVDs using synchronised phonocardiogram (PCG) and electrocardiogram (ECG) signals, as well as multichannel PCG (mPCG). However, state-of-the-art architectures remain underutilised due to the limited availability of synchronised and multichannel datasets. Augmented datasets and pre-trained models provide a pathway to overcome these limitations, enabling transformer-based architectures to be trained effectively. This work combines traditional signal processing with denoising diffusion models, WaveGrad and DiffWave, to create an augmented dataset to fine-tune a Wav2Vec 2.0-based classifier on multimodal and multichannel heart sound datasets. The approach achieves state-of-the-art performance. On the Computing in Cardiology (CinC) 2016 dataset of single channel PCG, accuracy, unweighted average recall (UAR), sensitivity, specificity and Matthew's correlation coefficient (MCC) reach 92.48%, 93.05%, 93.63%, 92.48%, 94.93% and 0.8283, respectively. Using the synchronised PCG and ECG signals of the training-a dataset from CinC, 93.14%, 92.21%, 94.35%, 90.10%, 95.12% and 0.8380 are achieved for accuracy, UAR, sensitivity, specificity and MCC, respectively. Using a wearable vest dataset consisting of mPCG data, the model achieves 77.13% accuracy, 74.25% UAR, 86.47% sensitivity, 62.04% specificity, and 0.5082 MCC. These results demonstrate the effectiveness of transformer-based models for CVD detection when supported by augmented datasets, highlighting their potential to advance multimodal and multichannel heart sound classification.
>
---
#### [replaced 013] Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.20410v2](http://arxiv.org/pdf/2509.20410v2)**

> **作者:** Weijie Wu; Wenhao Guan; Kaidi Wang; Peijie Chen; Zhuanling Zha; Junbo Li; Jun Fang; Lin Li; Qingyang Hong
>
> **摘要:** Spoken dialogue models have significantly advanced intelligent human-computer interaction, yet they lack a plug-and-play full-duplex prediction module for semantic endpoint detection, hindering seamless audio interactions. In this paper, we introduce Phoenix-VAD, an LLM-based model that enables streaming semantic endpoint detection. Specifically, Phoenix-VAD leverages the semantic comprehension capability of the LLM and a sliding window training strategy to achieve reliable semantic endpoint detection while supporting streaming inference. Experiments on both semantically complete and incomplete speech scenarios indicate that Phoenix-VAD achieves excellent and competitive performance. Furthermore, this design enables the full-duplex prediction module to be optimized independently of the dialogue model, providing more reliable and flexible support for next-generation human-computer interaction.
>
---
#### [replaced 014] video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15220v3](http://arxiv.org/pdf/2506.15220v3)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** We present video-SALMONN 2, a family of audio-visual large language models that set new state-of-the-art (SOTA) results in video description and question answering (QA). Our core contribution is multi-round direct preference optimisation (MrDPO), paired with a caption-quality objective that jointly rewards completeness and factual accuracy. Unlike standard DPO with a fixed reference policy, MrDPO periodically refreshes the reference by bootstrapping from a newly re-initialised lightweight adapter trained on the latest preferences, avoiding reference staleness and enabling continual improvement. This strategy produces captions that are consistently more detailed and accurate than those from proprietary systems such as GPT-4o and Gemini-1.5 Pro. We further distil these gains by using our model to generate a high-quality video-caption corpus for supervised fine-tuning of new models, transferring benefits beyond captioning to strong performance on complex video-QA tasks. Across widely used audio-visual and visual-only understanding benchmarks (including Video-MME, WorldSense, AVUT, Video-Holmes, DailyOmni, MLVU, and LVBench), our 3B and 7B models achieve SOTA results at comparable scales, while the 72B model surpasses all other open-source systems. Our source code, models, and data are released at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
