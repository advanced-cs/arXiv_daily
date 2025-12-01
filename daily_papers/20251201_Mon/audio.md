# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] Probabilistic Fusion and Calibration of Neural Speaker Diarization Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对端到端神经说话人聚类（EEND）系统中概率输出未充分校准与融合效率低的问题，提出基于概率水平的统一校准与融合框架。通过对比多标签与幂集表示，验证联合校准与先融合后校准策略的有效性，在CallHome数据集上显著降低DER，提升置信度可靠性，推动软决策在说话人聚类中的应用。**

- **链接: [https://arxiv.org/pdf/2511.22696v1](https://arxiv.org/pdf/2511.22696v1)**

> **作者:** Juan Ignacio Alvarez-Trejos; Sergio A. Balanya; Daniel Ramos; Alicia Lozano-Diez
>
> **摘要:** End-to-End Neural Diarization (EEND) systems produce frame-level probabilistic speaker activity estimates, yet since evaluation focuses primarily on Diarization Error Rate (DER), the reliability and calibration of these confidence scores have been largely neglected. When fusing multiple diarization systems, DOVER-Lap remains the only established approach, operating at the segment level with hard decisions. We propose working with continuous probability outputs, which enables more sophisticated calibration and fusion techniques that can leverage model uncertainty and complementary strengths across different architectures. This paper presents the first comprehensive framework for calibrating and fusing EEND models at the probability level. We investigate two output formulations (multilabel and powerset representations) and their impact on calibration and fusion effectiveness. Through extensive experiments on the CallHome two-speaker benchmark, we demonstrate that proper calibration provides substantial improvements even for individual models (up to 19% relative DER reduction), in some cases mitigating the absence of domain adaptation. We reveal that joint calibration in powerset space consistently outperforms independent per-speaker calibration, and that the Fuse-then-Calibrate ordering generally outperforms calibrating individual models before fusion while requiring calibration of only a single combined model. Our best configuration outperforms DOVER-Lap in terms of DER while providing reliable confidence estimates essential for downstream applications. This work proposes best practices for probability-level fusion of EEND systems and demonstrates the advantages of leveraging soft outputs over hard decisions.
>
---
#### [new 002] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding
- **分类: cs.SD**

- **简介: 该论文提出HPSU基准，旨在评估语音大模型在真实语境下理解隐含意图与情感的人类级感知能力。针对现有模型在复杂语义理解上的不足，构建了超2万条中英文专家验证数据集，融合多模态信息实现高效高质量标注，并系统评测多种模型，揭示其与人类水平的显著差距。**

- **链接: [https://arxiv.org/pdf/2511.23178v1](https://arxiv.org/pdf/2511.23178v1)**

> **作者:** Chen Li; Peiji Yang; Yicheng Zhong; Jianxing Yu; Zhisheng Wang; Zihao Gou; Wenqing Chen; Jian Yin
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recent advances in Speech Large Language Models (Speech LLMs) have led to great progress in speech understanding tasks such as Automatic Speech Recognition (ASR) and Speech Emotion Recognition (SER). However, whether these models can achieve human-level auditory perception, particularly in terms of their ability to comprehend latent intentions and implicit emotions in real-world spoken language, remains underexplored. To this end, we introduce the Human-level Perception in Spoken Speech Understanding (HPSU), a new benchmark for fully evaluating the human-level perceptual and understanding capabilities of Speech LLMs. HPSU comprises over 20,000 expert-validated spoken language understanding samples in English and Chinese. It establishes a comprehensive evaluation framework by encompassing a spectrum of tasks, ranging from basic speaker attribute recognition to complex inference of latent intentions and implicit emotions. To address the issues of data scarcity and high cost of manual annotation in real-world scenarios, we developed a semi-automatic annotation process. This process fuses audio, textual, and visual information to enable precise speech understanding and labeling, thus enhancing both annotation efficiency and quality. We systematically evaluate various open-source and proprietary Speech LLMs. The results demonstrate that even top-performing models still fall considerably short of human capabilities in understanding genuine spoken interactions. Consequently, HPSU will be useful for guiding the development of Speech LLMs toward human-level perception and cognition.
>
---
#### [new 003] Advancing Marine Bioacoustics with Deep Generative Models: A Hybrid Augmentation Strategy for Southern Resident Killer Whale Detection
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文针对海洋哺乳动物叫声检测中数据标注少、环境噪声复杂的问题，提出融合深度生成模型与传统方法的混合数据增强策略。通过对比VAE、GAN、DDPM等生成模型在虎鲸叫声检测中的表现，发现扩散模型效果最优，混合策略实现最高F1-score（0.81），有效提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.21872v1](https://arxiv.org/pdf/2511.21872v1)**

> **作者:** Bruno Padovese; Fabio Frazao; Michael Dowd; Ruth Joy
>
> **备注:** 16 pages, 6 Figures, 2 Tables, submitted to Marine Mammal Science as part of a special issue on Machine Learning and Artificial Intelligence in Marine Mammal Research
>
> **摘要:** Automated detection and classification of marine mammals vocalizations is critical for conservation and management efforts but is hindered by limited annotated datasets and the acoustic complexity of real-world marine environments. Data augmentation has proven to be an effective strategy to address this limitation by increasing dataset diversity and improving model generalization without requiring additional field data. However, most augmentation techniques used to date rely on effective but relatively simple transformations, leaving open the question of whether deep generative models can provide additional benefits. In this study, we evaluate the potential of deep generative for data augmentation in marine mammal call detection including: Variational Autoencoders, Generative Adversarial Networks, and Denoising Diffusion Probabilistic Models. Using Southern Resident Killer Whale (Orcinus orca) vocalizations from two long-term hydrophone deployments in the Salish Sea, we compare these approaches against traditional augmentation methods such as time-shifting and vocalization masking. While all generative approaches improved classification performance relative to the baseline, diffusion-based augmentation yielded the highest recall (0.87) and overall F1-score (0.75). A hybrid strategy combining generative-based synthesis with traditional methods achieved the best overall performance with an F1-score of 0.81. We hope this study encourages further exploration of deep generative models as complementary augmentation strategies to advance acoustic monitoring of threatened marine mammal populations.
>
---
#### [new 004] PURE Codec: Progressive Unfolding of Residual Entropy for Speech Codec Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对神经语音编解码器中残差向量量化训练不稳、分解无效的问题，提出PURE Codec框架。通过预训练语音增强模型引导多阶段量化，先重构低熵去噪特征，再编码高熵残差，提升训练稳定性和重建质量，尤其在噪声环境下表现更优。**

- **链接: [https://arxiv.org/pdf/2511.22687v1](https://arxiv.org/pdf/2511.22687v1)**

> **作者:** Jiatong Shi; Haoran Wang; William Chen; Chenda Li; Wangyou Zhang; Jinchuan Tian; Shinji Watanabe
>
> **备注:** Accepted by ASRU2025
>
> **摘要:** Neural speech codecs have achieved strong performance in low-bitrate compression, but residual vector quantization (RVQ) often suffers from unstable training and ineffective decomposition, limiting reconstruction quality and efficiency. We propose PURE Codec (Progressive Unfolding of Residual Entropy), a novel framework that guides multi-stage quantization using a pre-trained speech enhancement model. The first quantization stage reconstructs low-entropy, denoised speech embeddings, while subsequent stages encode residual high-entropy components. This design improves training stability significantly. Experiments demonstrate that PURE consistently outperforms conventional RVQ-based codecs in reconstruction and downstream speech language model-based text-to-speech, particularly under noisy training conditions.
>
---
#### [new 005] GLA-Grad++: An Improved Griffin-Lim Guided Diffusion Model for Speech Synthesis
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于语音合成任务，针对扩散模型在非训练分布条件下的音质下降问题，提出GLA-Grad++。通过仅一次应用格里芬-利姆算法计算修正项，加速生成并提升音质，尤其在域外场景表现更优。**

- **链接: [https://arxiv.org/pdf/2511.22293v1](https://arxiv.org/pdf/2511.22293v1)**

> **作者:** Teysir Baoueb; Xiaoyu Bie; Mathieu Fontaine; Gaël Richard
>
> **摘要:** Recent advances in diffusion models have positioned them as powerful generative frameworks for speech synthesis, demonstrating substantial improvements in audio quality and stability. Nevertheless, their effectiveness in vocoders conditioned on mel spectrograms remains constrained, particularly when the conditioning diverges from the training distribution. The recently proposed GLA-Grad model introduced a phase-aware extension to the WaveGrad vocoder that integrated the Griffin-Lim algorithm (GLA) into the reverse process to reduce inconsistencies between generated signals and conditioning mel spectrogram. In this paper, we further improve GLA-Grad through an innovative choice in how to apply the correction. Particularly, we compute the correction term only once, with a single application of GLA, to accelerate the generation process. Experimental results demonstrate that our method consistently outperforms the baseline models, particularly in out-of-domain scenarios.
>
---
#### [new 006] Group-Aware Partial Model Merging for Children's Automatic Speech Recognition
- **分类: eess.AS**

- **简介: 该论文针对儿童语音识别中数据少、声学差异大的问题，提出GRAPAM方法。通过无监督聚类分组儿童语音，对每组部分微调预训练模型后进行参数级合并，提升模型性能。实验表明，该方法在相同数据下相对降低6%词错误率，优于全量微调。**

- **链接: [https://arxiv.org/pdf/2511.23098v1](https://arxiv.org/pdf/2511.23098v1)**

> **作者:** Thomas Rolland; Alberto Abad
>
> **备注:** IEEE ASRU 2025 Workshop AI4CSL
>
> **摘要:** Automatic Speech Recognition (ASR) for children remains challenging, primarily due to large acoustic variability and limited availability of training data. While supervised fine-tuning of adult pre-trained models has shown promise, it often fails to capture group-specific characteristics variations among children. To address this, we introduce GRoup-Aware PARtial model Merging (GRAPAM), a parameter-efficient approach that combines unsupervised clustering, partial fine-tuning, and model merging. Our approach adapts adult-pre-trained models to children by first grouping the children's data based on acoustic similarity. Each group is used to partially fine-tune an adult pre-trained model, and the resulting models are merged at the parameter level. Experiments conducted on the MyST children's speech corpus indicate that GRAPAM achieves a relative improvement of 6% of Word Error Rate (WER), using the same amount of data, outperforming full fine-tuning while training fewer parameters. These results highlight the promise of model merging as a scalable and effective strategy for children's ASR.
>
---
#### [new 007] On the Cross-lingual Transferability of Pre-trained wav2vec2-based Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究wav2vec2模型在跨语言场景下的知识迁移能力。针对预训练模型在不同语言上表现差异的问题，通过18种语言的语音识别任务实验，发现数据多样性比数量更重要，且相似语言间迁移效果更优。研究为模型选择与预训练提供指导。**

- **链接: [https://arxiv.org/pdf/2511.21704v1](https://arxiv.org/pdf/2511.21704v1)**

> **作者:** Jonatas Grosman; Cassio Almeida; Guilherme Schardong; Hélio Lopes
>
> **摘要:** Using representations provided by a large pre-trained model has become the primary strategy for achieving state-of-the-art results in a wide range of tasks. A recently proposed large pre-trained model, wav2vec 2.0, was seminal for several other works on pre-training large models on speech data. Many models are being pre-trained using the same architecture as wav2vec 2.0 and are getting state-of-the-art in various speech-related tasks. Previous work has demonstrated that the data used during the pre-training of these wav2vec2-based models can impact the model's performance in downstream tasks, and this should be taken into consideration before utilizing these models. However, few works have proposed investigating further how the transfer knowledge of these pre-trained models behaves in different languages, even when the target language differs from the one used during the model's pre-training. Our work aims to investigate the cross-lingual transferability of these wav2vec2-based models. We performed several fine-tuning experiments on the speech recognition task in 18 languages using 15 large pre-trained models. The results of our experiments showed us that the size of data used during the pre-training of these models is not as important to the final performance as the diversity. We noticed that the performance of Indo-European languages is superior to non-Indo-European languages in the evaluated models. We have observed a positive cross-lingual transfer of knowledge using monolingual models, which was evident in all the languages we used, but more pronounced when the language used during pre-training was more similar to the downstream task language. With these findings, we aim to assist the scientific community in utilizing existing wav2vec2-based pre-trained models, as well as facilitate the pre-training of new ones.
>
---
#### [new 008] 3MDiT: Unified Tri-Modal Diffusion Transformer for Text-Driven Synchronized Audio-Video Generation
- **分类: cs.MM; cs.SD**

- **简介: 该论文针对文本驱动的音视频同步生成任务，解决现有方法音频质量差、模态间同步弱的问题。提出3MDiT框架，通过统一的三模态扩散变压器，实现文本、音频、视频的联合建模与动态对齐，支持复用预训练视频模型，显著提升生成质量和多模态一致性。**

- **链接: [https://arxiv.org/pdf/2511.21780v1](https://arxiv.org/pdf/2511.21780v1)**

> **作者:** Yaoru Li; Heyu Si; Federico Landi; Pilar Oplustil Gallegos; Ioannis Koutsoumpas; O. Ricardo Cortez Vazquez; Ruiju Fu; Qi Guo; Xin Jin; Shunyu Liu; Mingli Song
>
> **摘要:** Text-to-video (T2V) diffusion models have recently achieved impressive visual quality, yet most systems still generate silent clips and treat audio as a secondary concern. Existing audio-video generation pipelines typically decompose the task into cascaded stages, which accumulate errors across modalities and are trained under separate objectives. Recent joint audio-video generators alleviate this issue but often rely on dual-tower architectures with ad-hoc cross-modal bridges and static, single-shot text conditioning, making it difficult to both reuse T2V backbones and to reason about how audio, video and language interact over time. To address these challenges, we propose 3MDiT, a unified tri-modal diffusion transformer for text-driven synchronized audio-video generation. Our framework models video, audio and text as jointly evolving streams: an isomorphic audio branch mirrors a T2V backbone, tri-modal omni-blocks perform feature-level fusion across the three modalities, and an optional dynamic text conditioning mechanism updates the text representation as audio and video evidence co-evolve. The design supports two regimes: training from scratch on audio-video data, and orthogonally adapting a pretrained T2V model without modifying its backbone. Experiments show that our approach generates high-quality videos and realistic audio while consistently improving audio-video synchronization and tri-modal alignment across a range of quantitative metrics.
>
---
#### [new 009] Adapting Neural Audio Codecs to EEG
- **分类: cs.LG; cs.SD**

- **简介: 该论文研究神经音频编解码器在脑电图（EEG）压缩中的适配问题。针对EEG与音频模态差异，通过预处理使EEG符合音频编解码器输入要求，直接复用预训练的音频编解码器，实现稳定重建。提出多通道扩展模型DAC-MC以捕捉电极间空间依赖，显著提升压缩性能与临床信息保留能力。**

- **链接: [https://arxiv.org/pdf/2511.23142v1](https://arxiv.org/pdf/2511.23142v1)**

> **作者:** Ard Kastrati; Luca Lanzendörfer; Riccardo Rigoni; John Staib Matilla; Roger Wattenhofer
>
> **备注:** Foundation Models for the Brain and Body (BrainBodyFM@NeurIPS)
>
> **摘要:** EEG and audio are inherently distinct modalities, differing in sampling rate, channel structure, and scale. Yet, we show that pretrained neural audio codecs can serve as effective starting points for EEG compression, provided that the data are preprocessed to be suitable to the codec's input constraints. Using DAC, a state-of-the-art neural audio codec as our base, we demonstrate that raw EEG can be mapped into the codec's stride-based framing, enabling direct reuse of the audio-pretrained encoder-decoder. Even without modification, this setup yields stable EEG reconstructions, and fine-tuning on EEG data further improves fidelity and generalization compared to training from scratch. We systematically explore compression-quality trade-offs by varying residual codebook depth, codebook (vocabulary) size, and input sampling rate. To capture spatial dependencies across electrodes, we propose DAC-MC, a multi-channel extension with attention-based cross-channel aggregation and channel-specific decoding, while retaining the audio-pretrained initialization. Evaluations on the TUH Abnormal and Epilepsy datasets show that the adapted codecs preserve clinically relevant information, as reflected in spectrogram-based reconstruction loss and downstream classification accuracy.
>
---
#### [new 010] Joint Speech and Text Training for LLM-Based End-to-End Spoken Dialogue State Tracking
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对端到端语音对话状态追踪（DST）中语音数据稀缺、跨领域泛化差的问题，提出联合训练语音与文本DST数据的方法。通过利用易获取的文本DST数据，提升模型在无语音训练数据的目标领域的泛化能力，有效实现跨域性能提升。**

- **链接: [https://arxiv.org/pdf/2511.22503v1](https://arxiv.org/pdf/2511.22503v1)**

> **作者:** Katia Vendrame; Bolaji Yusuf; Santosh Kesiraju; Šimon Sedláček; Oldřich Plchot; Jan Černocký
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** End-to-end spoken dialogue state tracking (DST) is made difficult by the tandem of having to handle speech input and data scarcity. Combining speech foundation encoders and large language models has been proposed in recent work as to alleviate some of this difficulty. Although this approach has been shown to result in strong spoken DST models, achieving state-of-the-art performance in realistic multi-turn DST, it struggles to generalize across domains and requires annotated spoken DST training data for each domain of interest. However, collecting such data for every target domain is both costly and difficult. Noting that textual DST data is more easily obtained for various domains, in this work, we propose jointly training on available spoken DST data and written textual data from other domains as a way to achieve cross-domain generalization. We conduct experiments which show the efficacy of our proposed method for getting good cross-domain DST performance without relying on spoken training data from the target domains.
>
---
## 更新

#### [replaced 001] Neural Audio Codecs for Prompt-Driven Universal Sound Separation
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出CodecSep，首个基于神经音频编码器的通用文本驱动音源分离模型。针对现有方法计算量大或仅支持固定类别分离的问题，CodecSep结合DAC压缩与CLAP引导的FiLM调制Transformer掩码器，在保持比特流兼容性的同时，显著降低计算开销（仅1.35 GMACs），实现高效、通用的边缘设备音源分离。**

- **链接: [https://arxiv.org/pdf/2509.11717v4](https://arxiv.org/pdf/2509.11717v4)**

> **作者:** Adhiraj Banerjee; Vipul Arora
>
> **备注:** main content- 11 pages, total - 29 pages, 4 figure, pre-print, under review
>
> **摘要:** Text-guided sound separation supports flexible audio editing across media and assistive applications, but existing models like AudioSep are too compute-heavy for edge deployment. Neural audio codec (NAC) models such as CodecFormer and SDCodec are compute-efficient but limited to fixed-class separation. We introduce CodecSep, the first NAC-based model for on-device universal, text-driven separation. CodecSep combines DAC compression with a Transformer masker modulated by CLAP-derived FiLM parameters. Across six open-domain benchmarks under matched training/prompt protocols, \textbf{CodecSep} surpasses \textbf{AudioSep} in separation fidelity (SI-SDR) while remaining competitive in perceptual quality (ViSQOL) and matching or exceeding fixed-stem baselines (TDANet, CodecFormer, SDCodec). In code-stream deployments, it needs just 1.35~GMACs end-to-end -- approximately $54\times$ less compute ($25\times$ architecture-only) than spectrogram-domain separators like AudioSep -- while remaining fully bitstream-compatible.
>
---
#### [replaced 002] Video Echoed in Music: Semantic, Temporal, and Rhythmic Alignment for Video-to-Music Generation
- **分类: cs.SD; cs.MM**

- **简介: 该论文针对视频到音乐生成任务，解决现有方法在语义、时间与节奏对齐上的不足。提出VeM模型，通过分层视频解析与跨模态注意力机制增强语义一致性，设计帧级过渡-节拍对齐模块实现精确节奏同步，并构建新数据集与评估指标，显著提升生成音乐的契合度。**

- **链接: [https://arxiv.org/pdf/2511.09585v3](https://arxiv.org/pdf/2511.09585v3)**

> **作者:** Xinyi Tong; Yiran Zhu; Jishang Chen; Chunru Zhan; Tianle Wang; Sirui Zhang; Nian Liu; Tiezheng Ge; Duo Xu; Xin Jin; Feng Yu; Song-Chun Zhu
>
> **摘要:** Video-to-Music generation seeks to generate musically appropriate background music that enhances audiovisual immersion for videos. However, current approaches suffer from two critical limitations: 1) incomplete representation of video details, leading to weak alignment, and 2) inadequate temporal and rhythmic correspondence, particularly in achieving precise beat synchronization. To address the challenges, we propose Video Echoed in Music (VeM), a latent music diffusion that generates high-quality soundtracks with semantic, temporal, and rhythmic alignment for input videos. To capture video details comprehensively, VeM employs a hierarchical video parsing that acts as a music conductor, orchestrating multi-level information across modalities. Modality-specific encoders, coupled with a storyboard-guided cross-attention mechanism (SG-CAtt), integrate semantic cues while maintaining temporal coherence through position and duration encoding. For rhythmic precision, the frame-level transition-beat aligner and adapter (TB-As) dynamically synchronize visual scene transitions with music beats. We further contribute a novel video-music paired dataset sourced from e-commerce advertisements and video-sharing platforms, which imposes stricter transition-beat synchronization requirements. Meanwhile, we introduce novel metrics tailored to the task. Experimental results demonstrate superiority, particularly in semantic relevance and rhythmic precision.
>
---
#### [replaced 003] Comparison Performance of Spectrogram and Scalogram as Input of Acoustic Recognition Task
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究声学识别任务中谱图与小波图作为CNN输入的性能对比。针对现有研究缺乏系统比较的问题，通过实验评估两种频谱特征的优劣，分析其适用场景与局限性，为后续研究提供参考。**

- **链接: [https://arxiv.org/pdf/2403.03611v4](https://arxiv.org/pdf/2403.03611v4)**

> **作者:** Dang Thoai Phan
>
> **摘要:** Acoustic recognition has emerged as a prominent task in deep learning research, frequently utilizing spectral feature extraction techniques such as the spectrogram from the Short-Time Fourier Transform and the scalogram from the Wavelet Transform. However, there is a notable deficiency in studies that comprehensively discuss the advantages, drawbacks, and performance comparisons of these methods. This paper aims to evaluate the characteristics of these two transforms as input data for acoustic recognition using Convolutional Neural Networks. The performance of the trained models employing both transforms is documented for comparison. Through this analysis, the paper elucidates the advantages and limitations of each method, provides insights into their respective application scenarios, and identifies potential directions for further research.
>
---
#### [replaced 004] LAPS-Diff: A Diffusion-Based Framework for Singing Voice Synthesis With Language Aware Prosody-Style Guided Learning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对低资源下印度宝莱坞风格的歌唱语音合成（SVS）难题，提出LAPS-Diff框架。通过语言感知嵌入与声乐风格引导学习，结合歌词、音高和音乐上下文特征，提升合成语音的自然度与表现力。实验表明，该方法在客观与主观评价上均优于现有先进模型。**

- **链接: [https://arxiv.org/pdf/2507.04966v2](https://arxiv.org/pdf/2507.04966v2)**

> **作者:** Sandipan Dhar; Mayank Gupta; Preeti Rao
>
> **备注:** 10 pages, 5 figures, 3 Tables
>
> **摘要:** The field of Singing Voice Synthesis (SVS) has seen significant advancements in recent years due to the rapid progress of diffusion-based approaches. However, capturing vocal style, genre-specific pitch inflections, and language-dependent characteristics remains challenging, particularly in low-resource scenarios. To address this, we propose LAPS-Diff, a diffusion model integrated with language-aware embeddings and a vocal-style guided learning mechanism, specifically designed for Bollywood Hindi singing style. We curate a Hindi SVS dataset and leverage pre-trained language models to extract word and phone-level embeddings for an enriched lyrics representation. Additionally, we incorporated a style encoder and a pitch extraction model to compute style and pitch losses, capturing features essential to the naturalness and expressiveness of the synthesized singing, particularly in terms of vocal style and pitch variations. Furthermore, we utilize MERT and IndicWav2Vec models to extract musical and contextual embeddings, serving as conditional priors to refine the acoustic feature generation process further. Based on objective and subjective evaluations, we demonstrate that LAPS-Diff significantly improves the quality of the generated samples compared to the considered state-of-the-art (SOTA) model for our constrained dataset that is typical of the low resource scenario.
>
---
#### [replaced 005] Gelina: Unified Speech and Gesture Synthesis via Interleaved Token Prediction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Gelina，一个统一的语音与手势生成框架，解决传统方法顺序合成导致的同步性与韵律不匹配问题。通过交错令牌预测实现语音与手势联合生成，支持多说话人、多风格克隆及仅由语音生成手势，显著提升手势生成质量。**

- **链接: [https://arxiv.org/pdf/2510.12834v2](https://arxiv.org/pdf/2510.12834v2)**

> **作者:** Téo Guichoux; Théodor Lemerle; Shivam Mehta; Jonas Beskow; Gustav Eje Henter; Laure Soulier; Catherine Pelachaud; Nicolas Obin
>
> **备注:** 5 pages
>
> **摘要:** Human communication is multimodal, with speech and gestures tightly coupled, yet most computational methods for generating speech and gestures synthesize them sequentially, weakening synchrony and prosody alignment. We introduce Gelina, a unified framework that jointly synthesizes speech and co-speech gestures from text using interleaved token sequences in a discrete autoregressive backbone, with modality-specific decoders. Gelina supports multi-speaker and multi-style cloning and enables gesture-only synthesis from speech inputs. Subjective and objective evaluations demonstrate competitive speech quality and improved gesture generation over unimodal baselines.
>
---
#### [replaced 006] Bridging Speech Emotion Recognition and Personality: Dataset and Temporal Interaction Condition Network
- **分类: cs.SD; eess.AS**

- **简介: 该论文聚焦于语音情感识别（SER）任务，旨在解决情感识别精度受限的问题。通过构建首个同时包含情感与人格标注的PA-IEMOCAP数据集，提出时序交互条件网络（TICN），融合人格特征提升情感识别效果，实验证明其显著改善了情感判别能力。**

- **链接: [https://arxiv.org/pdf/2505.13978v2](https://arxiv.org/pdf/2505.13978v2)**

> **作者:** Yuan Gao; Hao Shi; Yahui Fu; Chenhui Chu; Tatsuya Kawahara
>
> **摘要:** This study investigates the interaction between personality traits and emotion expression, exploring how personality information can improve speech emotion recognition (SER). We collect the personality annotation for the IEMOCAP dataset, making it the first speech dataset that contains both emotion and personality annotations (PA-IEMOCAP), and enabling direct integration of personality traits into SER. Statistical analysis on this dataset identified significant correlations between personality traits and emotional expressions. To extract finegrained personality features, we propose a temporal interaction condition network (TICN), in which personality features are integrated with HuBERT-based acoustic features for SER. Experiments show that incorporating ground-truth personality traits significantly enhances valence recognition, improving the concordance correlation coefficient (CCC) from 0.698 to 0.785 compared to the baseline without personality information. For practical applications in dialogue systems where personality information about the user is unavailable, we develop a front-end module of automatic personality recognition. Using these automatically predicted traits as inputs to our proposed TICN model, we achieve a CCC of 0.776 for valence recognition, representing an 11.17% relative improvement over the baseline. These findings confirm the effectiveness of personality-aware SER and provide a solid foundation for further exploration in personality-aware speech processing applications.
>
---
#### [replaced 007] STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出STAR-Bench，用于评估音频4D智能——即对时空中声音动态的细粒度感知与推理能力。针对现有音频基准依赖文本描述、忽视复杂听觉推理的问题，构建包含基础感知与整体时空推理的评测体系，通过合成数据与人工筛选确保质量，揭示当前模型在时空感知上的显著短板。**

- **链接: [https://arxiv.org/pdf/2510.24693v2](https://arxiv.org/pdf/2510.24693v2)**

> **作者:** Zihan Liu; Zhikang Niu; Qiuyang Xiao; Zhisheng Zheng; Ruoqi Yuan; Yuhang Zang; Yuhang Cao; Xiaoyi Dong; Jianze Liang; Xie Chen; Leilei Sun; Dahua Lin; Jiaqi Wang
>
> **备注:** Homepage: https://internlm.github.io/StarBench/
>
> **摘要:** Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning. We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.
>
---
#### [replaced 008] Unsupervised Variational Acoustic Clustering
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出一种无监督变分声学聚类模型，用于时频域音频数据聚类。针对传统方法难以捕捉复杂音频模式的问题，设计了卷积-循环变分自编码器，结合高斯混合先验，提升聚类精度。实验在语音数字数据集上验证了模型有效性。**

- **链接: [https://arxiv.org/pdf/2503.18579v2](https://arxiv.org/pdf/2503.18579v2)**

> **作者:** Luan Vinícius Fiorio; Bruno Defraene; Johan David; Frans Widdershoven; Wim van Houtum; Ronald M. Aarts
>
> **备注:** This work has been submitted to the IEEE for possible publication. Please refer to arXiv:2510.01940 for an extended version
>
> **摘要:** We propose an unsupervised variational acoustic clustering model for clustering audio data in the time-frequency domain. The model leverages variational inference, extended to an autoencoder framework, with a Gaussian mixture model as a prior for the latent space. Specifically designed for audio applications, we introduce a convolutional-recurrent variational autoencoder optimized for efficient time-frequency processing. Our experimental results considering a spoken digits dataset demonstrate a significant improvement in accuracy and clustering performance compared to traditional methods, showcasing the model's enhanced ability to capture complex audio patterns.
>
---
#### [replaced 009] Privacy Disclosure of Similarity Rank in Speech and Language Processing
- **分类: eess.AS**

- **简介: 该论文研究语音与语言处理中相似度排名泄露隐私的问题。针对生物特征识别中因相似度测量不准确导致的真身份信息泄露，提出通过估计相似度排名的概率分布（用熵衡量）来量化隐私披露。实验表明各类特征均含可识别信息，且披露随测试样本长度增加但受数据库长度限制。**

- **链接: [https://arxiv.org/pdf/2508.05250v3](https://arxiv.org/pdf/2508.05250v3)**

> **作者:** Tom Bäckström; Mohammad Hassan Vali; My Nguyen; Silas Rech
>
> **备注:** accepted to IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Speaker, author, and other biometric identification applications often compare a sample's similarity to a database of templates to determine the identity. Given that data may be noisy and similarity measures can be inaccurate, such a comparison may not reliably identify the true identity as the most similar. Still, even the similarity rank based on an inaccurate similarity measure can disclose private information about the true identity. We propose a methodology for quantifying the privacy disclosure of such a similarity rank by estimating its probability distribution. It is based on determining the histogram of the similarity rank of the true speaker, or when data is scarce, modeling the histogram with the beta-binomial distribution. We express the disclosure in terms of entropy (bits), such that the disclosure from independent features are additive. Our experiments demonstrate that all tested speaker and author characterizations contain personally identifying information (PII) that can aid in identification, with embeddings from speaker recognition algorithms containing the most information, followed by phone embeddings, linguistic embeddings, and fundamental frequency. Our initial experiments show that the disclosure of PII increases with the length of test samples, but it is bounded by the length of database templates. The provided metric, similarity rank disclosure, provides a way to compare the disclosure of PII between biometric features and merge them to aid identification. It can thus aid in the holistic evaluation of threats to privacy in speech and other biometric technologies.
>
---
#### [replaced 010] State-of-the-art Embeddings with Video-free Segmentation of the Source VoxCeleb Data
- **分类: eess.AS**

- **简介: 该论文研究弱监督下的说话人嵌入提取任务，旨在无需语音时间戳和多模态对齐的情况下，仅用音频与名人标签训练高质量嵌入。通过改进ResNet与WavLM模型，利用未标注的语音片段实现端到端训练，达到领先性能，为大规模弱标签语音数据应用提供高效、视觉无关的解决方案。**

- **链接: [https://arxiv.org/pdf/2410.02364v2](https://arxiv.org/pdf/2410.02364v2)**

> **作者:** Sara Barahona; Ladislav Mošner; Themos Stafylakis; Oldřich Plchot; Junyi Peng; Lukáš Burget; Jan Černocký
>
> **备注:** Accepted at the IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) 2025
>
> **摘要:** In this paper, we refine and validate our method for training speaker embedding extractors using weak annotations. More specifically, we use only the audio stream of the source VoxCeleb videos and the names of the celebrities without knowing the time intervals in which they appear in the recording. We experiment with hyperparameters and embedding extractors based on ResNet and WavLM. We show that the method achieves state-of-the-art results in speaker verification, comparable with training the extractors in a standard supervised way on the VoxCeleb dataset. We also extend it by considering segments belonging to unknown speakers appearing alongside the celebrities, which are typically discarded. Removing the need for speaker timestamps and multimodal alignment, our method unlocks the use of large-scale weakly labeled speech data, enabling direct training of state-of-the-art embedding extractors and offering a visual-free alternative to VoxCeleb-style dataset creation.
>
---
#### [replaced 011] Clustering of Acoustic Environments with Variational Autoencoders for Hearing Devices
- **分类: eess.AS**

- **简介: 该论文研究听觉设备中声学环境的无监督聚类任务。针对传统方法依赖标签或难以提取高维数据特征的问题，提出基于变分自编码器（VAE）与Gumbel-Softmax的分类潜空间模型，结合时间上下文窗设计，实现对复杂声学场景的有效聚类，尤其在城市声景中表现优异。**

- **链接: [https://arxiv.org/pdf/2510.01940v2](https://arxiv.org/pdf/2510.01940v2)**

> **作者:** Luan Vinícius Fiorio; Ivana Nikoloska; Wim van Houtum; Ronald M. Aarts
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Particularly in hearing devices, the environmental context is taken into account for audio processing, often through classification. Traditional acoustic environment classification relies on classical algorithms, which are unable to extract meaningful representations of high-dimensionality data, or on supervised learning, being limited by the availability of labels. Knowing that human-imposed labels do not always reflect the true structure of acoustic scenes, we explore the (unsupervised) clustering of acoustic environments using variational autoencoders (VAEs), presenting a structured latent space suitable for the task. We propose a VAE model for categorical latent clustering employing a Gumbel-Softmax reparameterization with a time-context windowing scheme, tailored for real-world hearing device scenarios. Additionally, general adaptations on VAE architectures for audio clustering are also proposed. The approaches are validated through the clustering of spoken digits, a simpler task where labels are meaningful, and urban soundscapes, which recordings present strong overlap in time and frequency. While all variational methods succeeded when clustering spoken digits, only the proposed model achieved effective clustering performance on urban acoustic scenes, given its categorical nature.
>
---
#### [replaced 012] Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文针对声学识别中连续小波变换（CWT）计算复杂度高的问题，提出通过优化小波核长度和尺度图步长来降低计算成本。工作聚焦于提升基于CNN的声学识别效率，显著减少计算开销的同时保持模型性能，实现高效特征提取。**

- **链接: [https://arxiv.org/pdf/2505.13017v5](https://arxiv.org/pdf/2505.13017v5)**

> **作者:** Dang Thoai Phan; Tuan Anh Huynh; Van Tuan Pham; Cao Minh Tran; Van Thuan Mai; Ngoc Quy Tran
>
> **摘要:** The Continuous Wavelet Transform (CWT) is an effective tool for feature extraction in acoustic recognition using Convolutional Neural Networks (CNNs), particularly when applied to non-stationary audio. However, its high computational cost poses a significant challenge, often leading researchers to prefer alternative methods such as the Short-Time Fourier Transform (STFT). To address this issue, this paper proposes a method to reduce the computational complexity of CWT by optimizing the length of the wavelet kernel and the hop size of the output scalogram. Experimental results demonstrate that the proposed approach significantly reduces computational cost while maintaining the robust performance of the trained model in acoustic recognition tasks.
>
---
#### [replaced 013] Reduce Computational Complexity for Continuous Wavelet Transform in Acoustic Recognition Using Hop Size
- **分类: eess.AS; eess.SP**

- **简介: 该论文针对声学识别中连续小波变换（CWT）计算复杂度高的问题，提出采用跳步（hop size）策略仅对部分音频样本应用CWT。通过减少计算量，显著降低耗时，同时保持模型性能，有效提升了效率。**

- **链接: [https://arxiv.org/pdf/2408.14302v2](https://arxiv.org/pdf/2408.14302v2)**

> **作者:** Dang Thoai Phan
>
> **摘要:** In recent years, the continuous wavelet transform (CWT) has been employed as a spectral feature extractor for acoustic recognition tasks in conjunction with machine learning and deep learning models. However, applying the CWT to each individual audio sample is computationally intensive. This paper proposes an approach that applies the CWT to a subset of samples, spaced according to a specified hop size. Experimental results demonstrate that this method significantly reduces computational costs while maintaining the robust performance of the trained models.
>
---
#### [replaced 014] Learning and composing of classical music using restricted Boltzmann machines
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究音乐生成任务，旨在探究机器学习模型如何习得作曲能力及内部表示方式。作者基于受限玻尔兹曼机（RBM）构建生成算法，将乐谱转为钢琴卷帘图像进行无监督训练，实现任意长度音乐生成。研究表明，模型虽能创作新曲，但其内部表征难以被人类直接解读，揭示了生成模型在创造性任务中的可解释性问题。**

- **链接: [https://arxiv.org/pdf/2509.04899v3](https://arxiv.org/pdf/2509.04899v3)**

> **作者:** Mutsumi Kobayashi; Hiroshi Watanabe
>
> **备注:** 19 pages, 12 figures, manuscript was revised
>
> **摘要:** We investigate how machine learning models acquire the ability to compose music and how musical information is internally represented within such models. We develop a composition algorithm based on a restricted Boltzmann machine (RBM), a simple generative model capable of producing musical pieces of arbitrary length. We convert musical scores into piano-roll image representations and train the RBM in an unsupervised manner. We confirm that the trained RBM can generate new musical pieces; however, by analyzing the model's responses and internal structure, we find that the learned information is not stored in a form directly interpretable by humans. This study contributes to a better understanding of how machine learning models capable of music composition may internally represent musical structure and highlights issues related to the interpretability of generative models in creative tasks.
>
---
#### [replaced 015] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文针对视频到音频生成任务，解决现有方法中多目标冲突与人类偏好不一致问题。提出PrismAudio框架，通过四类专用思维链模块与对应奖励函数实现多维强化学习优化，并引入Fast-GRPO与AudioCanvas提升效率与评估可靠性，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18833v3](https://arxiv.org/pdf/2511.18833v3)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** Preprint
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at https://PrismAudio-Project.github.io.
>
---
#### [replaced 016] Categorical Unsupervised Variational Acoustic Clustering
- **分类: eess.AS**

- **简介: 该论文提出一种基于类别分布的无监督变分声学聚类方法，用于时间-频率域音频数据聚类。针对城市声景中音源重叠严重的问题，利用Gumbel-Softmax近似实现可微训练，通过调节温度参数优化聚类性能，显著提升重叠音源的分离效果。**

- **链接: [https://arxiv.org/pdf/2504.07652v2](https://arxiv.org/pdf/2504.07652v2)**

> **作者:** Luan Vinícius Fiorio; Ivana Nikoloska; Ronald M. Aarts
>
> **备注:** This work has been submitted to the IEEE for possible publication. Please refer to arXiv:2510.01940 for an extended version
>
> **摘要:** We propose a categorical approach for unsupervised variational acoustic clustering of audio data in the time-frequency domain. The consideration of a categorical distribution enforces sharper clustering even when data points strongly overlap in time and frequency, which is the case for most datasets of urban acoustic scenes. To this end, we use a Gumbel-Softmax distribution as a soft approximation to the categorical distribution, allowing for training via backpropagation. In this settings, the softmax temperature serves as the main mechanism to tune clustering performance. The results show that the proposed model can obtain impressive clustering performance for all considered datasets, even when data points strongly overlap in time and frequency.
>
---
#### [replaced 017] Balancing Speech Understanding and Generation Using Continual Pre-training for Codec-based Speech LLM
- **分类: eess.AS**

- **简介: 该论文针对语音语言模型中理解与生成的平衡难题，提出持续预训练框架，将文本LLM适配至编码器离散化语音表示，实现端到端单次通过的语音到语音翻译，无需中间表示，有效解决模态错配问题，提升跨任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2502.16897v2](https://arxiv.org/pdf/2502.16897v2)**

> **作者:** Jiatong Shi; Chunlei Zhang; Jinchuan Tian; Junrui Ni; Hao Zhang; Shinji Watanabe; Dong Yu
>
> **备注:** Accepted by ASRU2025
>
> **摘要:** Recent advances in speech language models (LLMs) have extended textual LLMs to the speech domain, but balancing speech understanding and generation remains challenging, especially with codec-based representations. We propose a continual pre-training (CPT) framework that adapts a textual LLM to handle codec-discretized speech, mitigating modality mismatch and preserving linguistic reasoning. Our unified model supports both understanding and generation, achieving strong results across ASR, TTS, S2T-Trans, and S2S-Trans. Notably, we present the first end-to-end, single-pass S2S-Trans system using only neural codec tokens, without intermediate transcriptions, translations, or semantic tokens. CPT proves essential for cross-modal alignment and task generalization, making it a powerful tool for building robust, unified speech LLMs.
>
---
#### [replaced 018] LongCat-Flash-Omni Technical Report
- **分类: cs.MM; cs.AI; cs.CL; cs.DC; cs.LG; cs.SD**

- **简介: 该论文提出LongCat-Flash-Omni，一个560B参数的开源多模态模型，解决大规模多模态训练与实时交互效率问题。通过渐进式训练和模态解耦并行策略，实现高效跨模态理解与生成，在文本、图像、视频、音频等任务上达领先性能。**

- **链接: [https://arxiv.org/pdf/2511.00279v2](https://arxiv.org/pdf/2511.00279v2)**

> **作者:** Meituan LongCat Team; Bairui Wang; Bayan; Bin Xiao; Bo Zhang; Bolin Rong; Borun Chen; Chang Wan; Chao Zhang; Chen Huang; Chen Chen; Chen Chen; Chengxu Yang; Chengzuo Yang; Cong Han; Dandan Peng; Delian Ruan; Detai Xin; Disong Wang; Dongchao Yang; Fanfan Liu; Fengjiao Chen; Fengyu Yang; Gan Dong; Gang Huang; Gang Xu; Guanglu Wan; Guoqiang Tan; Guoqiao Yu; Haibo Qiu; Hao Lu; Hongbo Liu; Hongyu Xiang; Jiaheng Wu; Jian Yang; Jiaxing Liu; Jing Huang; Jingang Wang; Jinrui Ding; Juchao Jiang; Jun Kuang; Jun Wang; Junhui Mei; Ke Ding; Kefeng Zhang; Lei Chen; Liang Shi; Limeng Qiao; Liming Zheng; Lin Ma; Liuyang Guo; Liya Ma; Luying Sun; Man Gao; Mengshen Zhu; Miao Cao; Minliang Lin; Nuo Xu; Peng Shi; Qi Zhang; Qian Fang; Qian Wang; Qian Yang; Quanxiu Wang; Rongxiang Weng; Rongxin Guo; Ruoxuan Liang; Senbin Yang; Shanbo Xu; Shanglin Lei; Shengze Ye; Shimin Chen; Shuaiqi Chen; Shujie Hu; Shuo Li; Siqi Yang; Siyu Xu; Siyu Ren; Song Li; Songxiang Liu; Tianhao Bai; Tianye Dai; Wei Hong; Wei Wang; Weixiao Zhao; Wengang Cao; Wenlong Zhu; Wenlong He; Xi Su; Xi Nan; Xiaohan Zhao; Xiaohao Wang; Xiaoyu Zhao; Xiaoyu Wang; Xiaoyu Li; Xin Pan; Xin Chen; Xiusong Sun; Xu Xiang; Xudong Xing; Xuezhi Cao; Xunliang Cai; Yang Yang; Yanli Tan; Yao Yao; Yerui Sun; Yi Chen; Yifan Lu; Yin Gong; Yining Zhang; Yitian Chen; Yiyang Gan; Yuchen Tang; Yuchen Xie; Yueqian Wang; Yuewen Zheng; Yufei Zhang; Yufeng Zhong; Yulei Qian; Yuqi Peng; Yuqian Li; Yuwei Jiang; Zeyang Hu; Zheng Zhang; Zhengkun Tian; Zhiqing Hong; Zhixiong Zeng; Zhuqi Mi; Ziran Li; Ziwen Wang; Ziyi Zhao; Ziyuan Zhuang; Zizhe Zhao
>
> **摘要:** We introduce LongCat-Flash-Omni, a state-of-the-art open-source omni-modal model with 560 billion parameters, excelling at real-time audio-visual interaction. By adopting a curriculum-inspired progressive training strategy that transitions from simpler to increasingly complex modality sequence modeling tasks, LongCat-Flash-Omni attains comprehensive multimodal capabilities while maintaining strong unimodal capability. Building upon LongCat-Flash, which adopts a high-performance Shortcut-connected Mixture-of-Experts (MoE) architecture with zero-computation experts, LongCat-Flash-Omni integrates efficient multimodal perception and speech reconstruction modules. Despite its immense size of 560B parameters (with 27B activated), LongCat-Flash-Omni achieves low-latency real-time audio-visual interaction. For training infrastructure, we developed a modality-decoupled parallelism scheme specifically designed to manage the data and model heterogeneity inherent in large-scale multimodal training. This innovative approach demonstrates exceptional efficiency by sustaining over 90% of the throughput achieved by text-only training. Extensive evaluations show that LongCat-Flash-Omni achieves state-of-the-art performance on omni-modal benchmarks among open-source models. Furthermore, it delivers highly competitive results across a wide range of modality-specific tasks, including text, image, and video understanding, as well as audio understanding and generation. We provide a comprehensive overview of the model architecture design, training procedures, and data strategies, and open-source the model to foster future research and development in the community.
>
---
