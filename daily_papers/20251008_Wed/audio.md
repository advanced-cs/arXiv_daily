# 音频 cs.SD;  eess.SP

- **最新发布 19 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] StereoSync: Spatially-Aware Stereo Audio Generation from Video
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决视频与音频在时间和空间上的对齐问题。论文提出StereoSync模型，利用深度图和边界框提取空间线索，通过扩散模型生成时空同步的立体音频，提升音频生成的沉浸感与真实感。**

- **链接: [http://arxiv.org/pdf/2510.05828v1](http://arxiv.org/pdf/2510.05828v1)**

> **作者:** Christian Marinoni; Riccardo Fosco Gramaccioni; Kazuki Shimada; Takashi Shibuya; Yuki Mitsufuji; Danilo Comminiello
>
> **备注:** Accepted at IJCNN 2025
>
> **摘要:** Although audio generation has been widely studied over recent years, video-aligned audio generation still remains a relatively unexplored frontier. To address this gap, we introduce StereoSync, a novel and efficient model designed to generate audio that is both temporally synchronized with a reference video and spatially aligned with its visual context. Moreover, StereoSync also achieves efficiency by leveraging pretrained foundation models, reducing the need for extensive training while maintaining high-quality synthesis. Unlike existing methods that primarily focus on temporal synchronization, StereoSync introduces a significant advancement by incorporating spatial awareness into video-aligned audio generation. Indeed, given an input video, our approach extracts spatial cues from depth maps and bounding boxes, using them as cross-attention conditioning in a diffusion-based audio generation model. Such an approach allows StereoSync to go beyond simple synchronization, producing stereo audio that dynamically adapts to the spatial structure and movement of a video scene. We evaluate StereoSync on Walking The Maps, a curated dataset comprising videos from video games that feature animated characters walking through diverse environments. Experimental results demonstrate the ability of StereoSync to achieve both temporal and spatial alignment, advancing the state of the art in video-to-audio generation and resulting in a significantly more immersive and realistic audio experience.
>
---
#### [new 002] EMORL-TTS: Reinforcement Learning for Fine-Grained Emotion Control in LLM-based TTS
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决当前基于大语言模型的TTS系统在精细情感控制上的不足。作者提出了EMORL-TTS框架，结合监督微调与强化学习，实现对情感类别、强度和重音的细粒度控制，提升了情感表达的准确性与清晰度，同时保持高质量语音合成。**

- **链接: [http://arxiv.org/pdf/2510.05758v1](http://arxiv.org/pdf/2510.05758v1)**

> **作者:** Haoxun Li; Yu Liu; Yuqing Sun; Hanlei Shi; Leyuan Qu; Taihao Li
>
> **备注:** Under review for ICASSP 2026
>
> **摘要:** Recent LLM-based TTS systems achieve strong quality and zero-shot ability, but lack fine-grained emotional control due to their reliance on discrete speech tokens. Existing approaches either limit emotions to categorical labels or cannot generalize to LLM-based architectures. We propose EMORL-TTS (Fine-grained Emotion-controllable TTS with Reinforcement Learning), a framework that unifies global intensity control in the VAD space with local emphasis regulation. Our method combines supervised fine-tuning with reinforcement learning guided by task-specific rewards for emotion category, intensity, and emphasis. Moreover, we further investigate how emphasis placement modulates fine-grained emotion intensity. Experiments show that EMORL-TTS improves emotion accuracy, intensity differentiation, and emphasis clarity, while preserving synthesis quality comparable to strong LLM-based baselines.
>
---
#### [new 003] Transcribing Rhythmic Patterns of the Guitar Track in Polyphonic Music
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决从复调音乐中准确提取吉他节奏模式的问题。作者提出三步框架：分离吉他音轨、检测拨弦、解码节奏模式，实现了高精度节奏模式识别，并提供可读性强的表示形式。**

- **链接: [http://arxiv.org/pdf/2510.05756v1](http://arxiv.org/pdf/2510.05756v1)**

> **作者:** Aleksandr Lukoianov; Anssi Klapuri
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** Whereas chord transcription has received considerable attention during the past couple of decades, far less work has been devoted to transcribing and encoding the rhythmic patterns that occur in a song. The topic is especially relevant for instruments such as the rhythm guitar, which is typically played by strumming rhythmic patterns that repeat and vary over time. However, in many cases one cannot objectively define a single "right" rhythmic pattern for a given song section. To create a dataset with well-defined ground-truth labels, we asked expert musicians to transcribe the rhythmic patterns in 410 popular songs and record cover versions where the guitar tracks followed those transcriptions. To transcribe the strums and their corresponding rhythmic patterns, we propose a three-step framework. Firstly, we perform approximate stem separation to extract the guitar part from the polyphonic mixture. Secondly, we detect individual strums within the separated guitar audio, using a pre-trained foundation model (MERT) as a backbone. Finally, we carry out a pattern-decoding process in which the transcribed sequence of guitar strums is represented by patterns drawn from an expert-curated vocabulary. We show that it is possible to transcribe the rhythmic patterns of the guitar track in polyphonic music with quite high accuracy, producing a representation that is human-readable and includes automatically detected bar lines and time signature markers. We perform ablation studies and error analysis and propose a set of evaluation metrics to assess the accuracy and readability of the predicted rhythmic pattern sequence.
>
---
#### [new 004] Segment-Factorized Full-Song Generation on Symbolic Piano Music
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决长序列钢琴音乐生成中的质量与效率问题。作者提出了一种分段生成模型SFS，通过将歌曲分段并利用相关段落进行注意力生成，支持用户自定义结构和交互式创作。**

- **链接: [http://arxiv.org/pdf/2510.05881v1](http://arxiv.org/pdf/2510.05881v1)**

> **作者:** Ping-Yi Chen; Chih-Pin Tan; Yi-Hsuan Yang
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI for Music
>
> **摘要:** We propose the Segmented Full-Song Model (SFS) for symbolic full-song generation. The model accepts a user-provided song structure and an optional short seed segment that anchors the main idea around which the song is developed. By factorizing a song into segments and generating each one through selective attention to related segments, the model achieves higher quality and efficiency compared to prior work. To demonstrate its suitability for human-AI interaction, we further wrap SFS into a web application that enables users to iteratively co-create music on a piano roll with customizable structures and flexible ordering.
>
---
#### [new 005] Sci-Phi: A Large Language Model Spatial Audio Descriptor
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出Sci-Phi，一种空间音频大语言模型，旨在解决单通道音频输入在空间理解上的限制。该模型通过双空间与频谱编码器，估计声音来源及环境参数，能描述最多四个定向声源、背景音与房间特性。模型基于合成数据训练，并在真实场景中表现良好，具备实际应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.05542v1](http://arxiv.org/pdf/2510.05542v1)**

> **作者:** Xilin Jiang; Hannes Gamper; Sebastian Braun
>
> **摘要:** Acoustic scene perception involves describing the type of sounds, their timing, their direction and distance, as well as their loudness and reverberation. While audio language models excel in sound recognition, single-channel input fundamentally limits spatial understanding. This work presents Sci-Phi, a spatial audio large language model with dual spatial and spectral encoders that estimates a complete parameter set for all sound sources and the surrounding environment. Learning from over 4,000 hours of synthetic first-order Ambisonics recordings including metadata, Sci-Phi enumerates and describes up to four directional sound sources in one pass, alongside non-directional background sounds and room characteristics. We evaluate the model with a permutation-invariant protocol and 15 metrics covering content, location, timing, loudness, and reverberation, and analyze its robustness across source counts, signal-to-noise ratios, reverberation levels, and challenging mixtures of acoustically, spatially, or temporally similar sources. Notably, Sci-Phi generalizes to real room impulse responses with only minor performance degradation. Overall, this work establishes the first audio LLM capable of full spatial-scene description, with strong potential for real-world deployment. Demo: https://sci-phi-audio.github.io/demo
>
---
#### [new 006] ECTSpeech: Enhancing Efficient Speech Synthesis via Easy Consistency Tuning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决扩散模型在语音合成中推理效率低的问题。通过提出ECTSpeech框架，首次将Easy Consistency Tuning策略应用于语音合成，实现高质量的一次性生成，同时降低训练复杂度。**

- **链接: [http://arxiv.org/pdf/2510.05984v1](http://arxiv.org/pdf/2510.05984v1)**

> **作者:** Tao Zhu; Yinfeng Yu; Liejun Wang; Fuchun Sun; Wendong Zheng
>
> **备注:** Accepted for publication by Proceedings of the 2025 ACM Multimedia Asia Conference(MMAsia '25)
>
> **摘要:** Diffusion models have demonstrated remarkable performance in speech synthesis, but typically require multi-step sampling, resulting in low inference efficiency. Recent studies address this issue by distilling diffusion models into consistency models, enabling efficient one-step generation. However, these approaches introduce additional training costs and rely heavily on the performance of pre-trained teacher models. In this paper, we propose ECTSpeech, a simple and effective one-step speech synthesis framework that, for the first time, incorporates the Easy Consistency Tuning (ECT) strategy into speech synthesis. By progressively tightening consistency constraints on a pre-trained diffusion model, ECTSpeech achieves high-quality one-step generation while significantly reducing training complexity. In addition, we design a multi-scale gate module (MSGate) to enhance the denoiser's ability to fuse features at different scales. Experimental results on the LJSpeech dataset demonstrate that ECTSpeech achieves audio quality comparable to state-of-the-art methods under single-step sampling, while substantially reducing the model's training cost and complexity.
>
---
#### [new 007] FoleyGRAM: Video-to-Audio Generation with GRAM-Aligned Multimodal Encoders
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决生成音频与视频内容语义对齐的问题。作者提出FoleyGRAM，利用GRAM对齐多模态编码器，结合扩散模型生成音频，提升了生成音频的语义准确性和与视频的同步性。**

- **链接: [http://arxiv.org/pdf/2510.05829v1](http://arxiv.org/pdf/2510.05829v1)**

> **作者:** Riccardo Fosco Gramaccioni; Christian Marinoni; Eleonora Grassucci; Giordano Cicchetti; Aurelio Uncini; Danilo Comminiello
>
> **备注:** Acepted at IJCNN 2025
>
> **摘要:** In this work, we present FoleyGRAM, a novel approach to video-to-audio generation that emphasizes semantic conditioning through the use of aligned multimodal encoders. Building on prior advancements in video-to-audio generation, FoleyGRAM leverages the Gramian Representation Alignment Measure (GRAM) to align embeddings across video, text, and audio modalities, enabling precise semantic control over the audio generation process. The core of FoleyGRAM is a diffusion-based audio synthesis model conditioned on GRAM-aligned embeddings and waveform envelopes, ensuring both semantic richness and temporal alignment with the corresponding input video. We evaluate FoleyGRAM on the Greatest Hits dataset, a standard benchmark for video-to-audio models. Our experiments demonstrate that aligning multimodal encoders using GRAM enhances the system's ability to semantically align generated audio with video content, advancing the state of the art in video-to-audio synthesis.
>
---
#### [new 008] AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文提出AUREXA-SE，用于音频-视觉语音增强（AVSE）任务，旨在通过融合音频和视觉信息提升嘈杂环境下的语音质量。方法包括使用1D卷积编码器、Swin Transformer V2、双向交叉注意力机制及Squeezeformer模块，实现语音去噪。实验结果显示其在多个指标上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.05295v1](http://arxiv.org/pdf/2510.05295v1)**

> **作者:** M. Sajid; Deepanshu Gupta; Yash Modi; Sanskriti Jain; Harshith Jai Surya Ganji; A. Rahaman; Harshvardhan Choudhary; Nasir Saleem; Amir Hussain; M. Tanveer
>
> **摘要:** In this paper, we propose AUREXA-SE (Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement), a progressive bimodal framework tailored for audio-visual speech enhancement (AVSE). AUREXA-SE jointly leverages raw audio waveforms and visual cues by employing a U-Net-based 1D convolutional encoder for audio and a Swin Transformer V2 for efficient and expressive visual feature extraction. Central to the architecture is a novel bidirectional cross-attention mechanism, which facilitates deep contextual fusion between modalities, enabling rich and complementary representation learning. To capture temporal dependencies within the fused embeddings, a stack of lightweight Squeezeformer blocks combining convolutional and attention modules is introduced. The enhanced embeddings are then decoded via a U-Net-style decoder for direct waveform reconstruction, ensuring perceptually consistent and intelligible speech output. Experimental evaluations demonstrate the effectiveness of AUREXA-SE, achieving significant performance improvements over noisy baselines, with STOI of 0.516, PESQ of 1.323, and SI-SDR of -4.322 dB. The source code of AUREXA-SE is available at https://github.com/mtanveer1/AVSEC-4-Challenge-2025.
>
---
#### [new 009] Provable Speech Attributes Conversion via Latent Independence
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音属性转换任务，旨在解决现有方法缺乏理论保证、控制不可靠的问题。作者提出一种基于潜在独立性的框架，通过非概率自编码器结构和独立性约束，实现对语音属性（如说话人身份、情感）的可控转换，同时保持内容不变。方法具备理论分析支持，并通过实验验证其有效性与通用性。**

- **链接: [http://arxiv.org/pdf/2510.05191v1](http://arxiv.org/pdf/2510.05191v1)**

> **作者:** Jonathan Svirsky; Ofir Lindenbaum; Uri Shaham
>
> **摘要:** While signal conversion and disentangled representation learning have shown promise for manipulating data attributes across domains such as audio, image, and multimodal generation, existing approaches, especially for speech style conversion, are largely empirical and lack rigorous theoretical foundations to guarantee reliable and interpretable control. In this work, we propose a general framework for speech attribute conversion, accompanied by theoretical analysis and guarantees under reasonable assumptions. Our framework builds on a non-probabilistic autoencoder architecture with an independence constraint between the predicted latent variable and the target controllable variable. This design ensures a consistent signal transformation, conditioned on an observed style variable, while preserving the original content and modifying the desired attribute. We further demonstrate the versatility of our method by evaluating it on speech styles, including speaker identity and emotion. Quantitative evaluations confirm the effectiveness and generality of the proposed approach.
>
---
#### [new 010] EmoHRNet: High-Resolution Neural Network Based Speech Emotion Recognition
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音情感识别（SER）任务，旨在提升人机交互体验。为解决语音情感识别中细节与整体特征难以兼顾的问题，作者提出EmoHRNet模型，基于HRNet架构，通过保持高分辨率表示，有效捕捉语音信号中的情感特征。实验表明其性能优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.06072v1](http://arxiv.org/pdf/2510.06072v1)**

> **作者:** Akshay Muppidi; Martin Radfar
>
> **摘要:** Speech emotion recognition (SER) is pivotal for enhancing human-machine interactions. This paper introduces "EmoHRNet", a novel adaptation of High-Resolution Networks (HRNet) tailored for SER. The HRNet structure is designed to maintain high-resolution representations from the initial to the final layers. By transforming audio samples into spectrograms, EmoHRNet leverages the HRNet architecture to extract high-level features. EmoHRNet's unique architecture maintains high-resolution representations throughout, capturing both granular and overarching emotional cues from speech signals. The model outperforms leading models, achieving accuracies of 92.45% on RAVDESS, 80.06% on IEMOCAP, and 92.77% on EMOVO. Thus, we show that EmoHRNet sets a new benchmark in the SER domain.
>
---
#### [new 011] MSF-SER: Enriching Acoustic Modeling with Multi-Granularity Semantics for Speech Emotion Recognition
- **分类: cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决现有方法忽略语义细节和高级语义线索的问题。作者提出MSF-SER模型，融合局部、全局和扩展语义信息，通过门控融合与跨模态Mixture-of-Experts提升情感识别效果。**

- **链接: [http://arxiv.org/pdf/2510.05749v1](http://arxiv.org/pdf/2510.05749v1)**

> **作者:** Haoxun Li; Yuqing Sun; Hanlei Shi; Yu Liu; Leyuan Qu; Taihao Li
>
> **备注:** Under review for ICASSP 2026
>
> **摘要:** Continuous dimensional speech emotion recognition captures affective variation along valence, arousal, and dominance, providing finer-grained representations than categorical approaches. Yet most multimodal methods rely solely on global transcripts, leading to two limitations: (1) all words are treated equally, overlooking that emphasis on different parts of a sentence can shift emotional meaning; (2) only surface lexical content is represented, lacking higher-level interpretive cues. To overcome these issues, we propose MSF-SER (Multi-granularity Semantic Fusion for Speech Emotion Recognition), which augments acoustic features with three complementary levels of textual semantics--Local Emphasized Semantics (LES), Global Semantics (GS), and Extended Semantics (ES). These are integrated via an intra-modal gated fusion and a cross-modal FiLM-modulated lightweight Mixture-of-Experts (FM-MOE). Experiments on MSP-Podcast and IEMOCAP show that MSF-SER consistently improves dimensional prediction, demonstrating the effectiveness of enriched semantic fusion for SER.
>
---
#### [new 012] LARA-Gen: Enabling Continuous Emotion Control for Music Generation Models via Latent Affective Representation Alignment
- **分类: cs.SD**

- **简介: 该论文属于音乐生成任务，旨在解决文本到音乐生成中的细粒度情感控制问题。作者提出了LARA-Gen框架，通过潜层情感表征对齐实现情感控制，并设计了基于连续情感空间的控制模块，有效分离情感与文本内容。**

- **链接: [http://arxiv.org/pdf/2510.05875v1](http://arxiv.org/pdf/2510.05875v1)**

> **作者:** Jiahao Mei; Xuenan Xu; Zeyu Xie; Zihao Zheng; Ye Tao; Yue Ding; Mengyue Wu
>
> **摘要:** Recent advances in text-to-music models have enabled coherent music generation from text prompts, yet fine-grained emotional control remains unresolved. We introduce LARA-Gen, a framework for continuous emotion control that aligns the internal hidden states with an external music understanding model through Latent Affective Representation Alignment (LARA), enabling effective training. In addition, we design an emotion control module based on a continuous valence-arousal space, disentangling emotional attributes from textual content and bypassing the bottlenecks of text-based prompting. Furthermore, we establish a benchmark with a curated test set and a robust Emotion Predictor, facilitating objective evaluation of emotional controllability in music generation. Extensive experiments demonstrate that LARA-Gen achieves continuous, fine-grained control of emotion and significantly outperforms baselines in both emotion adherence and music quality. Generated samples are available at https://nieeim.github.io/LARA-Gen/.
>
---
#### [new 013] Sparse deepfake detection promotes better disentanglement
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于语音深度伪造检测任务，旨在提升检测系统的可解释性。作者在AASIST模型的嵌入层使用TopK激活实现稀疏表示，提升了检测性能（EER为23.36%），并验证了稀疏表示具有更好的解耦特性，能直接编码攻击特征。**

- **链接: [http://arxiv.org/pdf/2510.05696v1](http://arxiv.org/pdf/2510.05696v1)**

> **作者:** Antoine Teissier; Marie Tahon; Nicolas Dugué; Aghilas Sini
>
> **摘要:** Due to the rapid progress of speech synthesis, deepfake detection has become a major concern in the speech processing community. Because it is a critical task, systems must not only be efficient and robust, but also provide interpretable explanations. Among the different approaches for explainability, we focus on the interpretation of latent representations. In such paper, we focus on the last layer of embeddings of AASIST, a deepfake detection architecture. We use a TopK activation inspired by SAEs on this layer to obtain sparse representations which are used in the decision process. We demonstrate that sparse deepfake detection can improve detection performance, with an EER of 23.36% on ASVSpoof5 test set, with 95% of sparsity. We then show that these representations provide better disentanglement, using completeness and modularity metrics based on mutual information. Notably, some attacks are directly encoded in the latent space.
>
---
#### [new 014] Modulation Discovery with Differentiable Digital Signal Processing
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决从音频中解析调制信号、提升可解释性与匹配精度的问题。作者结合调制提取、约束控制信号参数化与可微数字信号处理，提出一种新方法，并验证其有效性与适用性。**

- **链接: [http://arxiv.org/pdf/2510.06204v1](http://arxiv.org/pdf/2510.06204v1)**

> **作者:** Christopher Mitcheltree; Hao Hao Tan; Joshua D. Reiss
>
> **备注:** Accepted to WASPAA 2025 (best paper award candidate). Code, audio samples, and plugins can be found at https://christhetree.github.io/mod_discovery/
>
> **摘要:** Modulations are a critical part of sound design and music production, enabling the creation of complex and evolving audio. Modern synthesizers provide envelopes, low frequency oscillators (LFOs), and more parameter automation tools that allow users to modulate the output with ease. However, determining the modulation signals used to create a sound is difficult, and existing sound-matching / parameter estimation systems are often uninterpretable black boxes or predict high-dimensional framewise parameter values without considering the shape, structure, and routing of the underlying modulation curves. We propose a neural sound-matching approach that leverages modulation extraction, constrained control signal parameterizations, and differentiable digital signal processing (DDSP) to discover the modulations present in a sound. We demonstrate the effectiveness of our approach on highly modulated synthetic and real audio samples, its applicability to different DDSP synth architectures, and investigate the trade-off it incurs between interpretability and sound-matching accuracy. We make our code and audio samples available and provide the trained DDSP synths in a VST plugin.
>
---
#### [new 015] Leveraging Vision Transformers for Enhanced Classification of Emotions using ECG Signals
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于情感分类任务，旨在通过心电图（ECG）信号提升情绪识别效果。论文提出一种改进的视觉Transformer（ViT）模型，融合CNN和SE模块，并采用小波变换和谱分析预处理ECG信号。在YAAD和DREAMER数据集上验证，该方法在情绪分类表现上超越现有技术。**

- **链接: [http://arxiv.org/pdf/2510.05826v1](http://arxiv.org/pdf/2510.05826v1)**

> **作者:** Pubudu L. Indrasiri; Bipasha Kashyap; Pubudu N. Pathirana
>
> **备注:** 14pages, 2 figures
>
> **摘要:** Biomedical signals provide insights into various conditions affecting the human body. Beyond diagnostic capabilities, these signals offer a deeper understanding of how specific organs respond to an individual's emotions and feelings. For instance, ECG data can reveal changes in heart rate variability linked to emotional arousal, stress levels, and autonomic nervous system activity. This data offers a window into the physiological basis of our emotional states. Recent advancements in the field diverge from conventional approaches by leveraging the power of advanced transformer architectures, which surpass traditional machine learning and deep learning methods. We begin by assessing the effectiveness of the Vision Transformer (ViT), a forefront model in image classification, for identifying emotions in imaged ECGs. Following this, we present and evaluate an improved version of ViT, integrating both CNN and SE blocks, aiming to bolster performance on imaged ECGs associated with emotion detection. Our method unfolds in two critical phases: first, we apply advanced preprocessing techniques for signal purification and converting signals into interpretable images using continuous wavelet transform and power spectral density analysis; second, we unveil a performance-boosted vision transformer architecture, cleverly enhanced with convolutional neural network components, to adeptly tackle the challenges of emotion recognition. Our methodology's robustness and innovation were thoroughly tested using ECG data from the YAAD and DREAMER datasets, leading to remarkable outcomes. For the YAAD dataset, our approach outperformed existing state-of-the-art methods in classifying seven unique emotional states, as well as in valence and arousal classification. Similarly, in the DREAMER dataset, our method excelled in distinguishing between valence, arousal and dominance, surpassing current leading techniques.
>
---
#### [new 016] TokenChain: A Discrete Speech Chain via Semantic Token Modeling
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音处理任务，旨在提升语音识别（ASR）和语音合成（TTS）性能。论文提出TokenChain，通过语义token建模构建离散语音链，结合自回归文本到语义模型与掩码生成的语义到声学模型，实现端到端反馈学习，有效提升跨域迁移效果，减少错误率并缓解遗忘问题。**

- **链接: [http://arxiv.org/pdf/2510.06201v1](http://arxiv.org/pdf/2510.06201v1)**

> **作者:** Mingxuan Wang; Satoshi Nakamura
>
> **备注:** 5 pages, 3 figures. Submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Machine Speech Chain, simulating the human perception-production loop, proves effective in jointly improving ASR and TTS. We propose TokenChain, a fully discrete speech chain coupling semantic-token ASR with a two-stage TTS: an autoregressive text-to-semantic model co-trained with ASR and a masked-generative semantic-to-acoustic model for synthesis only. End-to-end feedback across the text interface is enabled with straight-through argmax/Gumbel-Softmax and balanced with supervised ASR via dynamic weight averaging. Ablations examine optimal temperature schedules for in- and cross-domain transfer. Evaluation reveals TokenChain surpasses baseline accuracy 2-6 epochs earlier and yields 5-13% lower equal-epoch error with stable T2S on LibriSpeech, and reduces relative ASR WER by 56% and T2S WER by 31% on TED-LIUM with minimal forgetting, showing that chain learning remains effective with token interfaces and models.
>
---
#### [new 017] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决传统TTS系统在发音对齐上的不足。现有方法依赖成对的优劣语句级样本，数据效率低且无法实现细粒度优化。论文提出TKTO，无需配对数据，实现更高效训练，并直接优化词元级发音对齐，显著提升日语TTS准确率并降低错误率。**

- **链接: [http://arxiv.org/pdf/2510.05799v1](http://arxiv.org/pdf/2510.05799v1)**

> **作者:** Rikuto Kotoge; Yuichi Sasaki
>
> **摘要:** Aligning text-to-speech (TTS) system outputs with human feedback through preference optimization has been shown to effectively improve the robustness and naturalness of language model-based TTS models. Current approaches primarily require paired desirable and undesirable samples at the utterance level. However, such pairs are often limited in TTS output data, and utterance-level formulation prevents fine-grained token-level optimization needed for accurate pronunciation alignment. In this study, we propose TKTO that eliminates the need for paired data, enabling a more data-efficient training paradigm, and directly targets token-level units, automatically providing fine-grained alignment signals without token-level annotations. TKTO improves the challenging Japanese TTS accuracy by 39% and reduces CER by 54%, automatically assigning 12.8 times stronger reward to targeted tokens.
>
---
#### [new 018] WaveSP-Net: Learnable Wavelet-Domain Sparse Prompt Tuning for Speech Deepfake Detection
- **分类: eess.AS; cs.CL; eess.SP**

- **简介: 该论文属于语音深伪检测任务，旨在解决现有方法依赖全量微调预训练模型、参数效率低且泛化能力弱的问题。作者提出WaveSP-Net，结合基于小波变换的参数高效前端与Mamba架构后端，有效捕捉多尺度特征，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.05305v1](http://arxiv.org/pdf/2510.05305v1)**

> **作者:** Xi Xuan; Xuechen Liu; Wenxin Zhang; Yi-Cheng Lin; Xiaojian Lin; Tomi Kinnunen
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Modern front-end design for speech deepfake detection relies on full fine-tuning of large pre-trained models like XLSR. However, this approach is not parameter-efficient and may lead to suboptimal generalization to realistic, in-the-wild data types. To address these limitations, we introduce a new family of parameter-efficient front-ends that fuse prompt-tuning with classical signal processing transforms. These include FourierPT-XLSR, which uses the Fourier Transform, and two variants based on the Wavelet Transform: WSPT-XLSR and Partial-WSPT-XLSR. We further propose WaveSP-Net, a novel architecture combining a Partial-WSPT-XLSR front-end and a bidirectional Mamba-based back-end. This design injects multi-resolution features into the prompt embeddings, which enhances the localization of subtle synthetic artifacts without altering the frozen XLSR parameters. Experimental results demonstrate that WaveSP-Net outperforms several state-of-the-art models on two new and challenging benchmarks, Deepfake-Eval-2024 and SpoofCeleb, with low trainable parameters and notable performance gains. The code and models are available at https://github.com/xxuan-acoustics/WaveSP-Net.
>
---
#### [new 019] Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices
- **分类: cs.DC; cs.AI; cs.CL; eess.SP**

- **简介: 该论文属于端侧多模态推理任务，旨在解决大模型在小型设备上运行时资源消耗高、效率低的问题。通过软硬件协同设计，将模型拆分为模块并分配至合适加速器，优化计算与内存使用，实现了高效的本地化推理。**

- **链接: [http://arxiv.org/pdf/2510.05109v1](http://arxiv.org/pdf/2510.05109v1)**

> **作者:** Yilong Li; Shuai Zhang; Yijing Zeng; Hao Zhang; Xinmiao Xiong; Jingyu Liu; Pan Hu; Suman Banerjee
>
> **摘要:** Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly half a day and LLaMA-3-8B for voice interactions up to almost 20.8 hours.
>
---
## 更新

#### [replaced 001] Large-Scale Training Data Attribution for Music Generative Models via Unlearning
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18312v2](http://arxiv.org/pdf/2506.18312v2)**

> **作者:** Woosung Choi; Junghyun Koo; Kin Wai Cheuk; Joan Serrà; Marco A. Martínez-Ramírez; Yukara Ikemiya; Naoki Murata; Yuhta Takida; Wei-Hsiang Liao; Yuki Mitsufuji
>
> **备注:** accepted at NeurIPS 2025 Creative AI Track
>
> **摘要:** This paper explores the use of unlearning methods for training data attribution (TDA) in music generative models trained on large-scale datasets. TDA aims to identify which specific training data points contributed the most to the generation of a particular output from a specific model. This is crucial in the context of AI-generated music, where proper recognition and credit for original artists are generally overlooked. By enabling white-box attribution, our work supports a fairer system for acknowledging artistic contributions and addresses pressing concerns related to AI ethics and copyright. We apply unlearning-based attribution to a text-to-music diffusion model trained on a large-scale dataset and investigate its feasibility and behavior in this setting. To validate the method, we perform a grid search over different hyperparameter configurations and quantitatively evaluate the consistency of the unlearning approach. We then compare attribution patterns from unlearning with non-counterfactual approaches. Our findings suggest that unlearning-based approaches can be effectively adapted to music generative models, introducing large-scale TDA to this domain and paving the way for more ethical and accountable AI systems for music creation.
>
---
#### [replaced 002] Combining Deterministic Enhanced Conditions with Dual-Streaming Encoding for Diffusion-Based Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13983v2](http://arxiv.org/pdf/2505.13983v2)**

> **作者:** Hao Shi; Xugang Lu; Kazuki Shimada; Tatsuya Kawahara
>
> **摘要:** Diffusion-based speech enhancement (SE) models need to incorporate correct prior knowledge as reliable conditions to generate accurate predictions. However, providing reliable conditions using noisy features is challenging. One solution is to use features enhanced by deterministic methods as conditions. However, the information distortion and loss caused by deterministic methods might affect the diffusion process. In this paper, we first investigate the effects of using different deterministic SE models as conditions for diffusion. We validate two conditions depending on whether the noisy feature was used as part of the condition: one using only the deterministic feature (deterministic-only), and the other using both deterministic and noisy features (deterministic-noisy). Preliminary investigation found that using deterministic enhanced conditions improves hearing experiences on real data, while the choice between using deterministic-only or deterministic-noisy conditions depends on the deterministic models. Based on these findings, we propose a dual-streaming encoding Repair-Diffusion Model for SE (DERDM-SE) to more effectively utilize both conditions. Moreover, we found that fine-grained deterministic models have greater potential in objective evaluation metrics, while UNet-based deterministic models provide more stable diffusion performance. Therefore, in the DERDM-SE, we propose a deterministic model that combines coarse- and fine-grained processing. Experimental results on CHiME4 show that the proposed models effectively leverage deterministic models to achieve better SE evaluation scores, along with more stable performance compared to other diffusion-based SE models.
>
---
#### [replaced 003] Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.20995v3](http://arxiv.org/pdf/2506.20995v3)**

> **作者:** Akio Hayakawa; Masato Ishii; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** We propose a step-by-step video-to-audio (V2A) generation method for finer controllability over the generation process and more realistic audio synthesis. Inspired by traditional Foley workflows, our approach aims to comprehensively capture all sound events induced by a video through the incremental generation of missing sound events. To avoid the need for costly multi-reference video-audio datasets, each generation step is formulated as a negatively guided V2A process that discourages duplication of existing sounds. The guidance model is trained by finetuning a pre-trained V2A model on audio pairs from adjacent segments of the same video, allowing training with standard single-reference audiovisual datasets that are easily accessible. Objective and subjective evaluations demonstrate that our method enhances the separability of generated sounds at each step and improves the overall quality of the final composite audio, outperforming existing baselines.
>
---
#### [replaced 004] An Investigation of Incorporating Mamba for Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.06573v2](http://arxiv.org/pdf/2405.06573v2)**

> **作者:** Rong Chao; Wen-Huang Cheng; Moreno La Quatra; Sabato Marco Siniscalchi; Chao-Han Huck Yang; Szu-Wei Fu; Yu Tsao
>
> **备注:** Accepted to IEEE SLT 2024
>
> **摘要:** This work aims to investigate the use of a recently proposed, attention-free, scalable state-space model (SSM), Mamba, for the speech enhancement (SE) task. In particular, we employ Mamba to deploy different regression-based SE models (SEMamba) with different configurations, namely basic, advanced, causal, and non-causal. Furthermore, loss functions either based on signal-level distances or metric-oriented are considered. Experimental evidence shows that SEMamba attains a competitive PESQ of 3.55 on the VoiceBank-DEMAND dataset with the advanced, non-causal configuration. A new state-of-the-art PESQ of 3.69 is also reported when SEMamba is combined with Perceptual Contrast Stretching (PCS). Compared against Transformed-based equivalent SE solutions, a noticeable FLOPs reduction up to ~12% is observed with the advanced non-causal configurations. Finally, SEMamba can be used as a pre-processing step before automatic speech recognition (ASR), showing competitive performance against recent SE solutions.
>
---
#### [replaced 005] CL-UZH submission to the NIST SRE 2024 Speaker Recognition Evaluation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.00952v2](http://arxiv.org/pdf/2510.00952v2)**

> **作者:** Aref Farhadipour; Shiran Liu; Masoumeh Chapariniya; Valeriia Vyshnevetska; Srikanth Madikeri; Teodora Vukovic; Volker Dellwo
>
> **备注:** CL-UZH submission for the NIST SRE 2024 Evaluation plan
>
> **摘要:** The CL-UZH team submitted one system each for the fixed and open conditions of the NIST SRE 2024 challenge. For the closed-set condition, results for the audio-only trials were achieved using the X-vector system developed with Kaldi. For the audio-visual results we used only models developed for the visual modality. Two sets of results were submitted for the open-set and closed-set conditions, one based on a pretrained model using the VoxBlink2 and VoxCeleb2 datasets. An Xvector-based model was trained from scratch using the CTS superset dataset for the closed set. In addition to the submission of the results of the SRE24 evaluation to the competition website, we talked about the performance of the proposed systems on the SRE24 evaluation in this report.
>
---
#### [replaced 006] Synthetic Audio Forensics Evaluation (SAFE) Challenge
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.03387v2](http://arxiv.org/pdf/2510.03387v2)**

> **作者:** Kirill Trapeznikov; Paul Cummer; Pranay Pherwani; Jai Aslam; Michael S. Davinroy; Peter Bautista; Laura Cassani; Matthew Stamm; Jill Crisman
>
> **摘要:** The increasing realism of synthetic speech generated by advanced text-to-speech (TTS) models, coupled with post-processing and laundering techniques, presents a significant challenge for audio forensic detection. In this paper, we introduce the SAFE (Synthetic Audio Forensics Evaluation) Challenge, a fully blind evaluation framework designed to benchmark detection models across progressively harder scenarios: raw synthetic speech, processed audio (e.g., compression, resampling), and laundered audio intended to evade forensic analysis. The SAFE challenge consisted of a total of 90 hours of audio and 21,000 audio samples split across 21 different real sources and 17 different TTS models and 3 tasks. We present the challenge, evaluation design and tasks, dataset details, and initial insights into the strengths and limitations of current approaches, offering a foundation for advancing synthetic audio detection research. More information is available at \href{https://stresearch.github.io/SAFE/}{https://stresearch.github.io/SAFE/}.
>
---
#### [replaced 007] Scattering Transformer: A Training-Free Transformer Architecture for Heart Murmur Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.18424v2](http://arxiv.org/pdf/2509.18424v2)**

> **作者:** Rami Zewail
>
> **备注:** This paper has been accepted for presentation at the 14th International Conference on Model and Data Engineering (MEDI 2025). The final authenticated Version of Record will be published by Springer in the Lecture Notes in Computer Science (LNCS) series
>
> **摘要:** In an attempt to address the need for skilled clinicians in heart sound interpretation, recent research efforts on automating cardiac auscultation have explored deep learning approaches. The majority of these approaches have been based on supervised learning that is always challenged in occasions where training data is limited. More recently, there has been a growing interest in potentials of pre-trained self-supervised audio foundation models for biomedical end tasks. Despite exhibiting promising results, these foundational models are typically computationally intensive. Within the context of automatic cardiac auscultation, this study explores a lightweight alternative to these general-purpose audio foundation models by introducing the Scattering Transformer, a novel, training-free transformer architecture for heart murmur detection. The proposed method leverages standard wavelet scattering networks by introducing contextual dependencies in a transformer-like architecture without any backpropagation. We evaluate our approach on the public CirCor DigiScope dataset, directly comparing it against leading general-purpose foundational models. The Scattering Transformer achieves a Weighted Accuracy(WAR) of 0.786 and an Unweighted Average Recall(UAR) of 0.697, demonstrating performance highly competitive with contemporary state of the art methods. This study establishes the Scattering Transformer as a viable and promising alternative in resource-constrained setups.
>
---
