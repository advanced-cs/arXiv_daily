# 音频 cs.SD;  eess.SP

- **最新发布 10 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Advancing Speech Quality Assessment Through Scientific Challenges and Open-source Activities
- **分类: cs.SD; eess.AS**

- **简介: 该论文旨在通过科学挑战和开源活动推动语音质量评估（SQA）的发展，解决提升自动SQA精度的问题，开发并维护相关工具与平台以促进SQA及生成式AI的持续发展。**

- **链接: [http://arxiv.org/pdf/2508.00317v1](http://arxiv.org/pdf/2508.00317v1)**

> **作者:** Wen-Chin Huang
>
> **备注:** APSIPA ASC 2025 perspective paper
>
> **摘要:** Speech quality assessment (SQA) refers to the evaluation of speech quality, and developing an accurate automatic SQA method that reflects human perception has become increasingly important, in order to keep up with the generative AI boom. In recent years, SQA has progressed to a point that researchers started to faithfully use automatic SQA in research papers as a rigorous measurement of goodness for speech generation systems. We believe that the scientific challenges and open-source activities of late have stimulated the growth in this field. In this paper, we review recent challenges as well as open-source implementations and toolkits for SQA, and highlight the importance of maintaining such activities to facilitate the development of not only SQA itself but also generative AI for speech.
>
---
#### [new 002] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文提出了一种多模态音频生成框架AudioGen-Omni，解决跨模态协同生成问题，通过联合训练方法融合图表示、利用AdaLN-PAAPI机制优化注意力，并冻结文本实现跨模态条件能力提升，有效提升了音频质量、语义和同步准确性，推理时间仅1.9秒。**

- **链接: [http://arxiv.org/pdf/2508.00733v1](http://arxiv.org/pdf/2508.00733v1)**

> **作者:** Le Wang; Jun Wang; Feng Deng; Chen Zhang; Kun Gai; Di Zhang
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** We present AudioGen-Omni - a unified approach based on multimodal diffusion transformers (MMDit), capable of generating high-fidelity audio, speech, and songs coherently synchronized with the input video. AudioGen-Omni introduces a novel joint training paradigm that seamlessly integrates large-scale video-text-audio corpora, enabling a model capable of generating semantically rich, acoustically diverse audio conditioned on multimodal inputs and adaptable to a wide range of audio generation tasks. AudioGen-Omni employs a unified lyrics-transcription encoder that encodes graphemes and phonemes from both sung and spoken inputs into dense frame-level representations. Dense frame-level representations are fused using an AdaLN-based joint attention mechanism enhanced with phase-aligned anisotropic positional infusion (PAAPI), wherein RoPE is selectively applied to temporally structured modalities to ensure precise and robust cross-modal alignment. By unfreezing all modalities and masking missing inputs, AudioGen-Omni mitigates the semantic constraints of text-frozen paradigms, enabling effective cross-modal conditioning. This joint training approach enhances audio quality, semantic alignment, and lip-sync accuracy, while also achieving state-of-the-art results on Text-to-Audio/Speech/Song tasks. With an inference time of 1.91 seconds for 8 seconds of audio, it offers substantial improvements in both efficiency and generality.
>
---
#### [new 003] Subband Architecture Aided Selective Fixed-Filter Active Noise Control
- **分类: eess.SP; cs.SY; eess.AS; eess.SY**

- **简介: 该论文提出一种基于延迟无子带结构的子带辅助选择性固定滤波器主动噪声控制方案，旨在解决传统算法收敛慢及非均匀噪声下的性能退化问题。通过离线预训练不同频率范围的子带滤波器并实时分解输入噪声，结合频率带匹配与加权叠加技术，实现快速抑制和高鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.00603v1](http://arxiv.org/pdf/2508.00603v1)**

> **作者:** Hong-Cheng Liang; Man-Wai Mak; Kong Aik Lee
>
> **摘要:** The feedforward selective fixed-filter method selects the most suitable pre-trained control filter based on the spectral features of the detected reference signal, effectively avoiding slow convergence in conventional adaptive algorithms. However, it can only handle limited types of noises, and the performance degrades when the input noise exhibits non-uniform power spectral density. To address these limitations, this paper devises a novel selective fixed-filter scheme based on a delayless subband structure. In the off-line training stage, subband control filters are pre-trained for different frequency ranges and stored in a dedicated sub-filter database. During the on-line control stage, the incoming noise is decomposed using a polyphase FFT filter bank, and a frequency-band-matching mechanism assigns each subband signal the most appropriate control filter. Subsequently, a weight stacking technique is employed to combine all subband weights into a fullband filter, enabling real-time noise suppression. Experimental results demonstrate that the proposed scheme provides fast convergence, effective noise reduction, and strong robustness in handling more complicated noisy environments.
>
---
#### [new 004] VR-PTOLEMAIC: A Virtual Environment for the Perceptual Testing of Spatial Audio Algorithms
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在开发虚拟环境用于评估空间音频算法，解决如何确保合成声场质量与沉浸体验的问题，通过MUSHRA方法实现对模拟声场重建算法的测试，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.00501v1](http://arxiv.org/pdf/2508.00501v1)**

> **作者:** Paolo Ostan; Francesca Del Gaudio; Federico Miotello; Mirco Pezzoli; Fabio Antonacci
>
> **备注:** to appear in EAA Forum Acusticum 2025
>
> **摘要:** The perceptual evaluation of spatial audio algorithms is an important step in the development of immersive audio applications, as it ensures that synthesized sound fields meet quality standards in terms of listening experience, spatial perception and auditory realism. To support these evaluations, virtual reality can offer a powerful platform by providing immersive and interactive testing environments. In this paper, we present VR-PTOLEMAIC, a virtual reality evaluation system designed for assessing spatial audio algorithms. The system implements the MUSHRA (MUlti-Stimulus test with Hidden Reference and Anchor) evaluation methodology into a virtual environment. In particular, users can position themselves in each of the 25 simulated listening positions of a virtually recreated seminar room and evaluate simulated acoustic responses with respect to the actually recorded second-order ambisonic room impulse responses, all convolved with various source signals. We evaluated the usability of the proposed framework through an extensive testing campaign in which assessors were asked to compare the reconstruction capabilities of various sound field reconstruction algorithms. Results show that the VR platform effectively supports the assessment of spatial audio algorithms, with generally positive feedback on user experience and immersivity.
>
---
#### [new 005] Ambisonics Super-Resolution Using A Waveform-Domain Neural Network
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在解决FOA空间音频格式在效率与精度之间的矛盾，通过构建基于波形域的神经网络（Conv-TasNet）实现更高阶Ambisonics（HOA）输出，相比传统渲染器显著提升位置误差0.6dB及感知质量80%。**

- **链接: [http://arxiv.org/pdf/2508.00240v1](http://arxiv.org/pdf/2508.00240v1)**

> **作者:** Ismael Nawfal; Symeon Delikaris Manias; Mehrez Souden; Juha Merimaa; Joshua Atkins; Elisabeth McMullin; Shadi Pirhosseinloo; Daniel Phillips
>
> **摘要:** Ambisonics is a spatial audio format describing a sound field. First-order Ambisonics (FOA) is a popular format comprising only four channels. This limited channel count comes at the expense of spatial accuracy. Ideally one would be able to take the efficiency of a FOA format without its limitations. We have devised a data-driven spatial audio solution that retains the efficiency of the FOA format but achieves quality that surpasses conventional renderers. Utilizing a fully convolutional time-domain audio neural network (Conv-TasNet), we created a solution that takes a FOA input and provides a higher order Ambisonics (HOA) output. This data driven approach is novel when compared to typical physics and psychoacoustic based renderers. Quantitative evaluations showed a 0.6dB average positional mean squared error difference between predicted and actual 3rd order HOA. The median qualitative rating showed an 80% improvement in perceived quality over the traditional rendering approach.
>
---
#### [new 006] SpA2V: Harnessing Spatial Auditory Cues for Audio-driven Spatially-aware Video Generation
- **分类: cs.GR; cs.AI; cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: SpA2V是一个基于空间音频提示的视频生成框架，解决了音频驱动视频生成中语义信息不足的问题，通过分解生成步骤实现高效、准确的视频生成。**

- **链接: [http://arxiv.org/pdf/2508.00782v1](http://arxiv.org/pdf/2508.00782v1)**

> **作者:** Kien T. Pham; Yingqing He; Yazhou Xing; Qifeng Chen; Long Chen
>
> **备注:** The 33rd ACM Multimedia Conference (MM '25)
>
> **摘要:** Audio-driven video generation aims to synthesize realistic videos that align with input audio recordings, akin to the human ability to visualize scenes from auditory input. However, existing approaches predominantly focus on exploring semantic information, such as the classes of sounding sources present in the audio, limiting their ability to generate videos with accurate content and spatial composition. In contrast, we humans can not only naturally identify the semantic categories of sounding sources but also determine their deeply encoded spatial attributes, including locations and movement directions. This useful information can be elucidated by considering specific spatial indicators derived from the inherent physical properties of sound, such as loudness or frequency. As prior methods largely ignore this factor, we present SpA2V, the first framework explicitly exploits these spatial auditory cues from audios to generate videos with high semantic and spatial correspondence. SpA2V decomposes the generation process into two stages: 1) Audio-guided Video Planning: We meticulously adapt a state-of-the-art MLLM for a novel task of harnessing spatial and semantic cues from input audio to construct Video Scene Layouts (VSLs). This serves as an intermediate representation to bridge the gap between the audio and video modalities. 2) Layout-grounded Video Generation: We develop an efficient and effective approach to seamlessly integrate VSLs as conditional guidance into pre-trained diffusion models, enabling VSL-grounded video generation in a training-free manner. Extensive experiments demonstrate that SpA2V excels in generating realistic videos with semantic and spatial alignment to the input audios.
>
---
#### [new 007] Beamformed 360° Sound Maps: U-Net-Driven Acoustic Source Segmentation and Localization
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文提出基于U-Net的360°声源分割与定位任务，解决传统SSL对密集空间音频的理解不足问题。利用DAS方法构建二维音频地图并结合频率域特征训练U-Net，通过Tversky损失缓解类别不平衡，实现数组独立建模，输出后处理为重心估计，适用于开放环境下的无人机音频数据。**

- **链接: [http://arxiv.org/pdf/2508.00307v1](http://arxiv.org/pdf/2508.00307v1)**

> **作者:** Belman Jahir Rodriguez; Sergio F. Chevtchenko; Marcelo Herrera Martinez; Yeshwant Bethy; Saeed Afshar
>
> **摘要:** We introduce a U-net model for 360{\deg} acoustic source localization formulated as a spherical semantic segmentation task. Rather than regressing discrete direction-of-arrival (DoA) angles, our model segments beamformed audio maps (azimuth and elevation) into regions of active sound presence. Using delay-and-sum (DAS) beamforming on a custom 24-microphone array, we generate signals aligned with drone GPS telemetry to create binary supervision masks. A modified U-Net, trained on frequency-domain representations of these maps, learns to identify spatially distributed source regions while addressing class imbalance via the Tversky loss. Because the network operates on beamformed energy maps, the approach is inherently array-independent and can adapt to different microphone configurations without retraining from scratch. The segmentation outputs are post-processed by computing centroids over activated regions, enabling robust DoA estimates. Our dataset includes real-world open-field recordings of a DJI Air 3 drone, synchronized with 360{\deg} video and flight logs across multiple dates and locations. Experimental results show that U-net generalizes across environments, providing improved angular precision, offering a new paradigm for dense spatial audio understanding beyond traditional Sound Source Localization (SSL).
>
---
#### [new 008] DeformTune: A Deformable XAI Music Prototype for Non-Musicians
- **分类: cs.HC; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出DeformTune作为非音乐家AI音乐原型，结合触觉变形界面与多模态学习模型，旨在提升AI音乐系统的可解释性与操作便捷性，解决传统工具缺乏非专业用户适应性的问题。**

- **链接: [http://arxiv.org/pdf/2508.00160v1](http://arxiv.org/pdf/2508.00160v1)**

> **作者:** Ziqing Xu; Nick Bryan-Kinns
>
> **备注:** In Proceedings of Explainable AI for the Arts Workshop 2025 (XAIxArts 2025) arXiv:2406.14485
>
> **摘要:** Many existing AI music generation tools rely on text prompts, complex interfaces, or instrument-like controls, which may require musical or technical knowledge that non-musicians do not possess. This paper introduces DeformTune, a prototype system that combines a tactile deformable interface with the MeasureVAE model to explore more intuitive, embodied, and explainable AI interaction. We conducted a preliminary study with 11 adult participants without formal musical training to investigate their experience with AI-assisted music creation. Thematic analysis of their feedback revealed recurring challenge--including unclear control mappings, limited expressive range, and the need for guidance throughout use. We discuss several design opportunities for enhancing explainability of AI, including multimodal feedback and progressive interaction support. These findings contribute early insights toward making AI music systems more explainable and empowering for novice users.
>
---
#### [new 009] Wavelet-Based Time-Frequency Fingerprinting for Feature Extraction of Traditional Irish Music
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文提出基于波浪分析的时间-频域指纹方法，用于传统爱尔兰音乐特征提取与音频识别，解决了时间序列特征提取与跨源比对问题，通过连续波形变换和波分相关分析实现高效音素匹配，应用于EEG信号分析和金融预测。**

- **链接: [http://arxiv.org/pdf/2508.00479v1](http://arxiv.org/pdf/2508.00479v1)**

> **作者:** Noah Shore
>
> **备注:** Master's thesis. The focus of the thesis is on the underlying techniques for signal fingerprinting
>
> **摘要:** This work presents a wavelet-based approach to time-frequency fingerprinting for time series feature extraction, with a focus on audio identification from live recordings of traditional Irish tunes. The challenges of identifying features in time-series data are addressed by employing a continuous wavelet transform to extract spectral features and wavelet coherence analysis is used to compare recorded audio spectrograms to synthetically generated tunes. The synthetic tunes are derived from ABC notation, which is a common symbolic representation for Irish music. Experimental results demonstrate that the wavelet-based method can accurately and efficiently identify recorded tunes. This research study also details the performance of the wavelet coherence model, highlighting its strengths over other methods of time-frequency decomposition. Additionally, we discuss and deploy the model on several applications beyond music, including in EEG signal analysis and financial time series forecasting.
>
---
#### [new 010] FMPlug: Plug-In Foundation Flow-Matching Priors for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文提出FMPlug框架，解决逆问题中的基流匹配（FM）先验优化任务，通过引入观察与目标对象的相似性及生成模型的高斯性，结合时间自适应策略和尖锐高斯约束，突破传统方法在无领域专有先验和非训练数据下的局限性，显著提升了图像超分辨率和高斯模糊修复性能。**

- **链接: [http://arxiv.org/pdf/2508.00721v1](http://arxiv.org/pdf/2508.00721v1)**

> **作者:** Yuxiang Wan; Ryan Devera; Wenjie Zhang; Ju Sun
>
> **摘要:** We present FMPlug, a novel plug-in framework that enhances foundation flow-matching (FM) priors for solving ill-posed inverse problems. Unlike traditional approaches that rely on domain-specific or untrained priors, FMPlug smartly leverages two simple but powerful insights: the similarity between observed and desired objects and the Gaussianity of generative flows. By introducing a time-adaptive warm-up strategy and sharp Gaussianity regularization, FMPlug unlocks the true potential of domain-agnostic foundation models. Our method beats state-of-the-art methods that use foundation FM priors by significant margins, on image super-resolution and Gaussian deblurring.
>
---
## 更新

#### [replaced 001] Data-driven tool wear prediction in milling, based on a process-integrated single-sensor approach
- **分类: cs.LG; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.19950v4](http://arxiv.org/pdf/2412.19950v4)**

> **作者:** Eric Hirsch; Christian Friedrich
>
> **备注:** This work has been submitted to the IEEE Transactions on Automation Science and Engineering for possible publication. ,14 pages, 12 figures
>
> **摘要:** Accurate tool wear prediction is essential for maintaining productivity and minimizing costs in machining. However, the complex nature of the tool wear process poses significant challenges to achieving reliable predictions. This study explores data-driven methods, in particular deep learning, for tool wear prediction. Traditional data-driven approaches often focus on a single process, relying on multi-sensor setups and extensive data generation, which limits generalization to new settings. Moreover, multi-sensor integration is often impractical in industrial environments. To address these limitations, this research investigates the transferability of predictive models using minimal training data, validated across two processes. Furthermore, it uses a simple setup with a single acceleration sensor to establish a low-cost data generation approach that facilitates the generalization of models to other processes via transfer learning. The study evaluates several machine learning models, including transformer-inspired convolutional neural networks (CNN), long short-term memory networks (LSTM), support vector machines (SVM), and decision trees, trained on different input formats such as feature vectors and short-time Fourier transform (STFT). The performance of the models is evaluated on two machines and on different amounts of training data, including scenarios with significantly reduced datasets, providing insight into their effectiveness under constrained data conditions. The results demonstrate the potential of specific models and configurations for effective tool wear prediction, contributing to the development of more adaptable and efficient predictive maintenance strategies in machining. Notably, the ConvNeXt model has an exceptional performance, achieving 99.1\% accuracy in identifying tool wear using data from only four milling tools operated until they are worn.
>
---
#### [replaced 002] AudioMiXR: Spatial Audio Object Manipulation with 6DoF for Sound Design in Augmented Reality
- **分类: cs.HC; cs.SD; eess.AS; H.5.2; H.5.5; H.5.1**

- **链接: [http://arxiv.org/pdf/2502.02929v3](http://arxiv.org/pdf/2502.02929v3)**

> **作者:** Brandon Woodard; Margarita Geleta; Joseph J. LaViola Jr.; Andrea Fanelli; Rhonda Wilson
>
> **备注:** Revision necessary for accuracy
>
> **摘要:** We present AudioMiXR, an augmented reality (AR) interface intended to assess how users manipulate virtual audio objects situated in their physical space using six degrees of freedom (6DoF) deployed on a head-mounted display (Apple Vision Pro) for 3D sound design. Existing tools for 3D sound design are typically constrained to desktop displays, which may limit spatial awareness of mixing within the execution environment. Utilizing an XR HMD to create soundscapes may provide a real-time test environment for 3D sound design, as modern HMDs can provide precise spatial localization assisted by cross-modal interactions. However, there is no research on design guidelines specific to sound design with six degrees of freedom (6DoF) in XR. To provide a first step toward identifying design-related research directions in this space, we conducted an exploratory study where we recruited 27 participants, consisting of expert and non-expert sound designers. The goal was to assess design lessons that can be used to inform future research venues in 3D sound design. We ran a within-subjects study where users designed both a music and cinematic soundscapes. After thematically analyzing participant data, we constructed two design lessons: 1. Proprioception for AR Sound Design, and 2. Balancing Audio-Visual Modalities in AR GUIs. Additionally, we provide application domains that can benefit most from 6DoF sound design based on our results.
>
---
#### [replaced 003] SwitchCodec: A High-Fidelity Nerual Audio Codec With Sparse Quantization
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24437v3](http://arxiv.org/pdf/2505.24437v3)**

> **作者:** Jin Wang; Wenbin Jiang; Xiangbo Wang; Yubo You; Sheng Fang
>
> **备注:** 12 pages,8 figures
>
> **摘要:** Neural audio compression has emerged as a promising technology for efficiently representing speech, music, and general audio. However, existing methods suffer from significant performance degradation at limited bitrates, where the available embedding space is sharply constrained. To address this, we propose a universal high-fidelity neural audio compression algorithm featuring Residual Experts Vector Quantization (REVQ), which substantially expands the embedding space with minimal impact on bandwidth. A gentle load-balancing strategy is introduced to ensure the full utilization of this expanded space. Furthermore, we develop a novel multi-tiered discriminator that periodically stratifies STFT spectra, guiding the generator to focus on critical spectral regions. To support multiple bitrates without quality loss at the lower end, we adopt an efficient post-training strategy. Our proposed model achieves impressive performance, with PESQ and ViSQOL scores of 2.87 and 4.27, respectively, at 2.67 kbps bandwidth. The approach effectively reduces spectral blur, decreasing the distance to the original mel-spectrogram by 13%. Notably, our post-training strategy achieves performance comparable to dedicated fixed-bitrate models while reducing the required training time by half. Extensive ablation studies confirm the superiority of our method over baselines.
>
---
#### [replaced 004] OpenACE: An Open Benchmark for Evaluating Audio Coding Performance
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.08374v2](http://arxiv.org/pdf/2409.08374v2)**

> **作者:** Jozef Coldenhoff; Niclas Granqvist; Milos Cernak
>
> **备注:** ICASSP 2025
>
> **摘要:** Audio and speech coding lack unified evaluation and open-source testing. Many candidate systems were evaluated on proprietary, non-reproducible, or small data, and machine learning-based codecs are often tested on datasets with similar distributions as trained on, which is unfairly compared to digital signal processing-based codecs that usually work well with unseen data. This paper presents a full-band audio and speech coding quality benchmark with more variable content types, including traditional open test vectors. An example use case of audio coding quality assessment is presented with open-source Opus, 3GPP's EVS, and recent ETSI's LC3 with LC3+ used in Bluetooth LE Audio profiles. Besides, quality variations of emotional speech encoding at 16 kbps are shown. The proposed open-source benchmark contributes to audio and speech coding democratization and is available at https://github.com/JozefColdenhoff/OpenACE.
>
---
#### [replaced 005] Next Tokens Denoising for Speech Synthesis
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.22746v2](http://arxiv.org/pdf/2507.22746v2)**

> **作者:** Yanqing Liu; Ruiqing Xue; Chong Zhang; Yufei Liu; Gang Wang; Bohan Li; Yao Qian; Lei He; Shujie Liu; Sheng Zhao
>
> **摘要:** While diffusion and autoregressive (AR) models have significantly advanced generative modeling, they each present distinct limitations. AR models, which rely on causal attention, cannot exploit future context and suffer from slow generation speeds. Conversely, diffusion models struggle with key-value (KV) caching. To overcome these challenges, we introduce Dragon-FM, a novel text-to-speech (TTS) design that unifies AR and flow-matching. This model processes 48 kHz audio codec tokens in chunks at a compact rate of 12.5 tokens per second. This design enables AR modeling across chunks, ensuring global coherence, while parallel flow-matching within chunks facilitates fast iterative denoising. Thus, the model leverages KV-cache across chunks and utilizes bidirectional context within each chunk. Furthermore, it bridges continuous and discrete feature modeling, demonstrating that continuous AR flow-matching can predict discrete tokens with finite scalar quantizers. This efficient codec and fast chunk-autoregressive architecture also make the model highly effective for generating long-form content, such as podcasts. Experiments on podcast datasets demonstrate its capability to efficiently generate high-quality zero-shot podcasts.
>
---
#### [replaced 006] Improving Code Switching with Supervised Fine Tuning and GELU Adapters
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00291v2](http://arxiv.org/pdf/2506.00291v2)**

> **作者:** Linh Pham
>
> **备注:** Incorrect results
>
> **摘要:** There are few code switching datasets, labeled or unlabled, that exist today. As a result, ASR requires new methods to utilize the vast monolingual data and models that exist. This paper uses OpenAI's open source ASR model, Whisper, which has been pre-trained on 680K hours of audio to perform monolingual ASR tasks. In Part 1, this paper examines how exploiting Whisper's monolingual ability to individually tokenize training text, called "Switching Tokenizers Method", improves transcription accuracy. In Part 2, we combine the Switching Tokenizers Method from part 1 and train a GELU based adapter on the encoder. These two methods reduced Total Mixed Error Rate (MER) to 9.4% for the ASCEND dataset, 6% for SEAME devman and 9.7% for SEAME devsge, outperforming current SoTA methods.
>
---
