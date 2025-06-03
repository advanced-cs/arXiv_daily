# 音频 cs.SD;  eess.SP

- **最新发布 68 篇**

- **更新 30 篇**

## 最新发布

#### [new 001] ReFlow-VC: Zero-shot Voice Conversion Based on Rectified Flow and Speaker Feature Optimization
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音转换任务，旨在解决扩散模型在语音转换中需要大量采样步骤的问题。作者提出了ReFlow-VC，基于修正流的语音转换方法，并优化说话人特征建模，提升小数据集和零样本场景下的语音转换效果。**

- **链接: [http://arxiv.org/pdf/2506.01032v1](http://arxiv.org/pdf/2506.01032v1)**

> **作者:** Pengyu Ren; Wenhao Guan; Kaidi Wang; Peijie Chen; Qingyang Hong; Lin Li
>
> **备注:** Comment: 5 pages, 2 figure, accepted by Interspeech 2025
>
> **摘要:** In recent years, diffusion-based generative models have demonstrated remarkable performance in speech conversion, including Denoising Diffusion Probabilistic Models (DDPM) and others. However, the advantages of these models come at the cost of requiring a large number of sampling steps. This limitation hinders their practical application in real-world scenarios. In this paper, we introduce ReFlow-VC, a novel high-fidelity speech conversion method based on rectified flow. Specifically, ReFlow-VC is an Ordinary Differential Equation (ODE) model that transforms a Gaussian distribution to the true Mel-spectrogram distribution along the most direct path. Furthermore, we propose a modeling approach that optimizes speaker features by utilizing both content and pitch information, allowing speaker features to reflect the properties of the current speech more accurately. Experimental results show that ReFlow-VC performs exceptionally well in small datasets and zero-shot scenarios.
>
---
#### [new 002] DS-TTS: Zero-Shot Speaker Style Adaptation from Voice Clips via Dynamic Dual-Style Feature Modulation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于零样本语音克隆任务，旨在仅用一段语音样本合成任意文本的目标语音。论文提出DS-TTS方法，通过双风格编码网络和动态生成网络，解决未见过的说话人风格适应问题，提升语音合成的自然度、表现力和说话人相似度。**

- **链接: [http://arxiv.org/pdf/2506.01020v1](http://arxiv.org/pdf/2506.01020v1)**

> **作者:** Ming Meng; Ziyi Yang; Jian Yang; Zhenjie Su; Yonggui Zhu; Zhaoxin Fan
>
> **摘要:** Recent advancements in text-to-speech (TTS) technology have increased demand for personalized audio synthesis. Zero-shot voice cloning, a specialized TTS task, aims to synthesize a target speaker's voice using only a single audio sample and arbitrary text, without prior exposure to the speaker during training. This process employs pattern recognition techniques to analyze and replicate the speaker's unique vocal features. Despite progress, challenges remain in adapting to the vocal style of unseen speakers, highlighting difficulties in generalizing TTS systems to handle diverse voices while maintaining naturalness, expressiveness, and speaker fidelity. To address the challenges of unseen speaker style adaptation, we propose DS-TTS, a novel approach aimed at enhancing the synthesis of diverse, previously unheard voices. Central to our method is a Dual-Style Encoding Network (DuSEN), where two distinct style encoders capture complementary aspects of a speaker's vocal identity. These speaker-specific style vectors are seamlessly integrated into the Dynamic Generator Network (DyGN) via a Style Gating-Film (SGF) mechanism, enabling more accurate and expressive reproduction of unseen speakers' unique vocal characteristics. In addition, we introduce a Dynamic Generator Network to tackle synthesis issues that arise with varying sentence lengths. By dynamically adapting to the length of the input, this component ensures robust performance across diverse text inputs and speaker styles, significantly improving the model's ability to generalize to unseen speakers in a more natural and expressive manner. Experimental evaluations on the VCTK dataset suggest that DS-TTS demonstrates superior overall performance in voice cloning tasks compared to existing state-of-the-art models, showing notable improvements in both word error rate and speaker similarity.
>
---
#### [new 003] Probing Audio-Generation Capabilities of Text-Based Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文探索文本大模型生成音频的能力，属于跨模态生成任务。研究如何通过代码中介提示模型生成音乐、环境音和语音，并使用FAD和CLAP评分评估效果。结果表明模型能生成基础音频，但复杂度越高表现越差。**

- **链接: [http://arxiv.org/pdf/2506.00003v1](http://arxiv.org/pdf/2506.00003v1)**

> **作者:** Arjun Prasaath Anbazhagan; Parteek Kumar; Ujjwal Kaur; Aslihan Akalin; Kevin Zhu; Sean O'Brien
>
> **备注:** Accepted at Conference of the North American Chapter of the Association for Computational Linguistics 2025, Student Research Workshop (NAACL SRW)
>
> **摘要:** How does textual representation of audio relate to the Large Language Model's (LLMs) learning about the audio world? This research investigates the extent to which LLMs can be prompted to generate audio, despite their primary training in textual data. We employ a three-tier approach, progressively increasing the complexity of audio generation: 1) Musical Notes, 2) Environmental Sounds, and 3) Human Speech. To bridge the gap between text and audio, we leverage code as an intermediary, prompting LLMs to generate code that, when executed, produces the desired audio output. To evaluate the quality and accuracy of the generated audio, we employ FAD and CLAP scores. Our findings reveal that while LLMs can generate basic audio features, their performance deteriorates as the complexity of the audio increases. This suggests that while LLMs possess a latent understanding of the auditory world, their ability to translate this understanding into tangible audio output remains rudimentary. Further research into techniques that can enhance the quality and diversity of LLM-generated audio can lead to an improvement in the performance of text-based LLMs in generating audio.
>
---
#### [new 004] Few-step Adversarial Schrödinger Bridge for Generative Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决生成模型在低信噪比下性能下降及采样步骤过多的问题。作者提出将薛定谔桥与GAN结合，在减少采样步数的同时提升去噪和去混响效果，实验证明其方法在少量步骤内优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.01460v1](http://arxiv.org/pdf/2506.01460v1)**

> **作者:** Seungu Han; Sungho Lee; Juheon Lee; Kyogu Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Deep generative models have recently been employed for speech enhancement to generate perceptually valid clean speech on large-scale datasets. Several diffusion models have been proposed, and more recently, a tractable Schr\"odinger Bridge has been introduced to transport between the clean and noisy speech distributions. However, these models often suffer from an iterative reverse process and require a large number of sampling steps -- more than 50. Our investigation reveals that the performance of baseline models significantly degrades when the number of sampling steps is reduced, particularly under low-SNR conditions. We propose integrating Schr\"odinger Bridge with GANs to effectively mitigate this issue, achieving high-quality outputs on full-band datasets while substantially reducing the required sampling steps. Experimental results demonstrate that our proposed model outperforms existing baselines, even with a single inference step, in both denoising and dereverberation tasks.
>
---
#### [new 005] DiffDSR: Dysarthric Speech Reconstruction Using Latent Diffusion Model
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音重建任务，旨在解决构音障碍语音理解度低和说话人身份保留差的问题。作者提出DiffDSR系统，结合潜扩散模型与预训练语音基础模型，提升语音可懂度与说话人相似性。**

- **链接: [http://arxiv.org/pdf/2506.00350v1](http://arxiv.org/pdf/2506.00350v1)**

> **作者:** Xueyuan Chen; Dongchao Yang; Wenxuan Wu; Minglin Wu; Jing Xu; Xixin Wu; Zhiyong Wu; Helen Meng
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Dysarthric speech reconstruction (DSR) aims to convert dysarthric speech into comprehensible speech while maintaining the speaker's identity. Despite significant advancements, existing methods often struggle with low speech intelligibility and poor speaker similarity. In this study, we introduce a novel diffusion-based DSR system that leverages a latent diffusion model to enhance the quality of speech reconstruction. Our model comprises: (i) a speech content encoder for phoneme embedding restoration via pre-trained self-supervised learning (SSL) speech foundation models; (ii) a speaker identity encoder for speaker-aware identity preservation by in-context learning mechanism; (iii) a diffusion-based speech generator to reconstruct the speech based on the restored phoneme embedding and preserved speaker identity. Through evaluations on the widely-used UASpeech corpus, our proposed model shows notable enhancements in speech intelligibility and speaker similarity.
>
---
#### [new 006] Fine-Tuning ASR for Stuttered Speech: Personalized vs. Generalized Approaches
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决口吃语音识别准确率低的问题。通过比较通用模型与个性化模型的微调效果，研究发现针对个体语言特征定制的ASR系统可显著降低错误率，提升对口吃者语音的识别效果。**

- **链接: [http://arxiv.org/pdf/2506.00853v1](http://arxiv.org/pdf/2506.00853v1)**

> **作者:** Dena Mujtaba; Nihar Mahapatra
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Stuttering -- characterized by involuntary disfluencies such as blocks, prolongations, and repetitions -- is often misinterpreted by automatic speech recognition (ASR) systems, resulting in elevated word error rates and making voice-driven technologies inaccessible to people who stutter. The variability of disfluencies across speakers and contexts further complicates ASR training, compounded by limited annotated stuttered speech data. In this paper, we investigate fine-tuning ASRs for stuttered speech, comparing generalized models (trained across multiple speakers) to personalized models tailored to individual speech characteristics. Using a diverse range of voice-AI scenarios, including virtual assistants and video interviews, we evaluate how personalization affects transcription accuracy. Our findings show that personalized ASRs significantly reduce word error rates, especially in spontaneous speech, highlighting the potential of tailored models for more inclusive voice technologies.
>
---
#### [new 007] ACE-Step: A Step Towards Music Generation Foundation Model
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决现有模型在生成速度、音乐连贯性和可控性之间的权衡问题。作者提出了ACE-Step，结合扩散模型、压缩自编码器和轻量级变换器，实现快速且高质量的音乐生成，支持多种控制功能，推动音乐AI基础模型的发展。**

- **链接: [http://arxiv.org/pdf/2506.00045v1](http://arxiv.org/pdf/2506.00045v1)**

> **作者:** Junmin Gong; Sean Zhao; Sen Wang; Shengyuan Xu; Joe Guo
>
> **备注:** 14 pages, 5 figures, ace-step's tech report
>
> **摘要:** We introduce ACE-Step, a novel open-source foundation model for music generation that overcomes key limitations of existing approaches and achieves state-of-the-art performance through a holistic architectural design. Current methods face inherent trade-offs between generation speed, musical coherence, and controllability. For example, LLM-based models (e.g. Yue, SongGen) excel at lyric alignment but suffer from slow inference and structural artifacts. Diffusion models (e.g. DiffRhythm), on the other hand, enable faster synthesis but often lack long-range structural coherence. ACE-Step bridges this gap by integrating diffusion-based generation with Sana's Deep Compression AutoEncoder (DCAE) and a lightweight linear transformer. It also leverages MERT and m-hubert to align semantic representations (REPA) during training, allowing rapid convergence. As a result, our model synthesizes up to 4 minutes of music in just 20 seconds on an A100 GPU-15x faster than LLM-based baselines-while achieving superior musical coherence and lyric alignment across melody, harmony, and rhythm metrics. Moreover, ACE-Step preserves fine-grained acoustic details, enabling advanced control mechanisms such as voice cloning, lyric editing, remixing, and track generation (e.g. lyric2vocal, singing2accompaniment). Rather than building yet another end-to-end text-to-music pipeline, our vision is to establish a foundation model for music AI: a fast, general-purpose, efficient yet flexible architecture that makes it easy to train subtasks on top of it. This paves the way for the development of powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. In short, our goal is to build a stable diffusion moment for music. The code, the model weights and the demo are available at: https://ace-step.github.io/.
>
---
#### [new 008] In-the-wild Audio Spatialization with Flexible Text-guided Localization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频空间化任务，旨在解决现有方法在多对象交互环境中缺乏灵活控制的问题。作者提出了一种文本引导的音频空间化框架（TAS），结合3D位置提示与翻转通道音频，提升双耳音频生成的质量与语义一致性，并构建了大规模数据集SpatialTAS用于训练。**

- **链接: [http://arxiv.org/pdf/2506.00927v1](http://arxiv.org/pdf/2506.00927v1)**

> **作者:** Tianrui Pan; Jie Liu; Zewen Huang; Jie Tang; Gangshan Wu
>
> **备注:** Accepted by ACL 2025 main
>
> **摘要:** To enhance immersive experiences, binaural audio offers spatial awareness of sounding objects in AR, VR, and embodied AI applications. While existing audio spatialization methods can generally map any available monaural audio to binaural audio signals, they often lack the flexible and interactive control needed in complex multi-object user-interactive environments. To address this, we propose a Text-guided Audio Spatialization (TAS) framework that utilizes flexible text prompts and evaluates our model from unified generation and comprehension perspectives. Due to the limited availability of premium and large-scale stereo data, we construct the SpatialTAS dataset, which encompasses 376,000 simulated binaural audio samples to facilitate the training of our model. Our model learns binaural differences guided by 3D spatial location and relative position prompts, augmented by flipped-channel audio. It outperforms existing methods on both simulated and real-recorded datasets, demonstrating superior generalization and accuracy. Besides, we develop an assessment model based on Llama-3.1-8B, which evaluates the spatial semantic coherence between our generated binaural audio and text prompts through a spatial reasoning task. Results demonstrate that text prompts provide flexible and interactive control to generate binaural audio with excellent quality and semantic consistency in spatial locations. Dataset is available at \href{https://github.com/Alice01010101/TASU}
>
---
#### [new 009] Learning Sparsity for Effective and Efficient Music Performance Question Answering
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于音乐表演音视频问答（Music AVQA）任务，旨在解决现有方法在处理密集、冗余数据时效率低下的问题。作者提出了Sparsify框架，通过三种稀疏学习策略，在保持准确率的同时提升训练效率，并设计了关键子集选择算法，仅用25%的数据达到70-80%的性能，显著提高数据效率。**

- **链接: [http://arxiv.org/pdf/2506.01319v1](http://arxiv.org/pdf/2506.01319v1)**

> **作者:** Xingjian Diao; Tianzhen Yang; Chunhui Zhang; Weiyi Wu; Ming Cheng; Jiang Gui
>
> **备注:** Accepted to the main conference of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Music performances, characterized by dense and continuous audio as well as seamless audio-visual integration, present unique challenges for multimodal scene understanding and reasoning. Recent Music Performance Audio-Visual Question Answering (Music AVQA) datasets have been proposed to reflect these challenges, highlighting the continued need for more effective integration of audio-visual representations in complex question answering. However, existing Music AVQA methods often rely on dense and unoptimized representations, leading to inefficiencies in the isolation of key information, the reduction of redundancy, and the prioritization of critical samples. To address these challenges, we introduce Sparsify, a sparse learning framework specifically designed for Music AVQA. It integrates three sparsification strategies into an end-to-end pipeline and achieves state-of-the-art performance on the Music AVQA datasets. In addition, it reduces training time by 28.32% compared to its fully trained dense counterpart while maintaining accuracy, demonstrating clear efficiency gains. To further improve data efficiency, we propose a key-subset selection algorithm that selects and uses approximately 25% of MUSIC-AVQA v2.0 training data and retains 70-80% of full-data performance across models.
>
---
#### [new 010] A Two-Stage Hierarchical Deep Filtering Framework for Real-Time Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升单通道语音在噪声环境中的质量。通过结合子带处理与深度滤波，利用目标时频单元及其周围信息，提出分阶段的层次化深度滤波框架，并引入TAConv模块强化特征提取，实现了更高效的语音增强。**

- **链接: [http://arxiv.org/pdf/2506.01023v1](http://arxiv.org/pdf/2506.01023v1)**

> **作者:** Shenghui Lu; Hukai Huang; Jinanglong Yao; Kaidi Wang; Qingyang Hong; Lin Li
>
> **备注:** 5 pages, 2 figure, accepted by Interspeech 2025
>
> **摘要:** This paper proposes a model that integrates sub-band processing and deep filtering to fully exploit information from the target time-frequency (TF) bin and its surrounding TF bins for single-channel speech enhancement. The sub-band module captures surrounding frequency bin information at the input, while the deep filtering module applies filtering at the output to both the target TF bin and its surrounding TF bins. To further improve the model performance, we decouple deep filtering into temporal and frequency components and introduce a two-stage framework, reducing the complexity of filter coefficient prediction at each stage. Additionally, we propose the TAConv module to strengthen convolutional feature extraction. Experimental results demonstrate that the proposed hierarchical deep filtering network (HDF-Net) effectively utilizes surrounding TF bin information and outperforms other advanced systems while using fewer resources.
>
---
#### [new 011] Attention Is Not Always the Answer: Optimizing Voice Activity Detection with Simple Feature Fusion
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音活动检测（VAD）任务，旨在提升语音与非语音片段的识别效果。现有方法依赖手工特征（如MFCC）或预训练模型（PTM）特征。作者提出FusionVAD框架，融合两类特征，比较拼接、加法和交叉注意力策略的效果。结果显示简单融合方法性能更优，优于当前最先进的Pyannote模型。**

- **链接: [http://arxiv.org/pdf/2506.01365v1](http://arxiv.org/pdf/2506.01365v1)**

> **作者:** Kumud Tripathi; Chowdam Venkata Kumar; Pankaj Wasnik
>
> **备注:** Accepted at INTERSPEECH 2025, 5 pages, 4 figures, 2 tables
>
> **摘要:** Voice Activity Detection (VAD) plays a key role in speech processing, often utilizing hand-crafted or neural features. This study examines the effectiveness of Mel-Frequency Cepstral Coefficients (MFCCs) and pre-trained model (PTM) features, including wav2vec 2.0, HuBERT, WavLM, UniSpeech, MMS, and Whisper. We propose FusionVAD, a unified framework that combines both feature types using three fusion strategies: concatenation, addition, and cross-attention (CA). Experimental results reveal that simple fusion techniques, particularly addition, outperform CA in both accuracy and efficiency. Fusion-based models consistently surpass single-feature models, highlighting the complementary nature of MFCCs and PTM features. Notably, our best-performing fusion model exceeds the state-of-the-art Pyannote across multiple datasets, achieving an absolute average improvement of 2.04%. These results confirm that simple feature fusion enhances VAD robustness while maintaining computational efficiency.
>
---
#### [new 012] CoVoMix2: Advancing Zero-Shot Dialogue Generation with Fully Non-Autoregressive Flow Matching
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于多说话人对话生成任务，旨在解决现有方法在语音一致性、重叠语音建模和生成效率上的不足。作者提出CoVoMix2，采用非自回归流匹配模型直接生成语音频谱，无需中间文本表示，并引入多种策略提升对话自然度与可控性，实现更优的零样本语音生成效果。**

- **链接: [http://arxiv.org/pdf/2506.00885v1](http://arxiv.org/pdf/2506.00885v1)**

> **作者:** Leying Zhang; Yao Qian; Xiaofei Wang; Manthan Thakker; Dongmei Wang; Jianwei Yu; Haibin Wu; Yuxuan Hu; Jinyu Li; Yanmin Qian; Sheng Zhao
>
> **摘要:** Generating natural-sounding, multi-speaker dialogue is crucial for applications such as podcast creation, virtual agents, and multimedia content generation. However, existing systems struggle to maintain speaker consistency, model overlapping speech, and synthesize coherent conversations efficiently. In this paper, we introduce CoVoMix2, a fully non-autoregressive framework for zero-shot multi-talker dialogue generation. CoVoMix2 directly predicts mel-spectrograms from multi-stream transcriptions using a flow-matching-based generative model, eliminating the reliance on intermediate token representations. To better capture realistic conversational dynamics, we propose transcription-level speaker disentanglement, sentence-level alignment, and prompt-level random masking strategies. Our approach achieves state-of-the-art performance, outperforming strong baselines like MoonCast and Sesame in speech quality, speaker consistency, and inference speed. Notably, CoVoMix2 operates without requiring transcriptions for the prompt and supports controllable dialogue generation, including overlapping speech and precise timing control, demonstrating strong generalizability to real-world speech generation scenarios.
>
---
#### [new 013] $\texttt{AVROBUSTBENCH}$: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于多模态鲁棒性评估任务，旨在解决现有基准无法充分评估音频-视觉模型在测试时面对分布偏移的鲁棒性问题。作者构建了AVROBUSTBENCH，包含四个数据集及75种双模态干扰，用于全面评估模型鲁棒性，并提出AV2C方法提升在线测试时适应性能。**

- **链接: [http://arxiv.org/pdf/2506.00358v1](http://arxiv.org/pdf/2506.00358v1)**

> **作者:** Sarthak Kumar Maharana; Saksham Singh Kushwaha; Baoming Zhang; Adrian Rodriguez; Songtao Wei; Yapeng Tian; Yunhui Guo
>
> **备注:** Under review. For uniformity, all TTA experiments are done with a batch size of 16
>
> **摘要:** While recent audio-visual models have demonstrated impressive performance, their robustness to distributional shifts at test-time remains not fully understood. Existing robustness benchmarks mainly focus on single modalities, making them insufficient for thoroughly assessing the robustness of audio-visual models. Motivated by real-world scenarios where shifts can occur $\textit{simultaneously}$ in both audio and visual modalities, we introduce $\texttt{AVROBUSTBENCH}$, a comprehensive benchmark designed to evaluate the test-time robustness of audio-visual recognition models. $\texttt{AVROBUSTBENCH}$ comprises four audio-visual benchmark datasets, $\texttt{AUDIOSET-2C}$, $\texttt{VGGSOUND-2C}$, $\texttt{KINETICS-2C}$, and $\texttt{EPICKITCHENS-2C}$, each incorporating 75 bimodal audio-visual corruptions that are $\textit{co-occurring}$ and $\textit{correlated}$. Through extensive evaluations, we observe that state-of-the-art supervised and self-supervised audio-visual models exhibit declining robustness as corruption severity increases. Furthermore, online test-time adaptation (TTA) methods, on $\texttt{VGGSOUND-2C}$ and $\texttt{KINETICS-2C}$, offer minimal improvements in performance under bimodal corruptions. We further propose $\texttt{AV2C}$, a simple TTA approach enabling on-the-fly cross-modal fusion by penalizing high-entropy samples, which achieves improvements on $\texttt{VGGSOUND-2C}$. We hope that $\texttt{AVROBUSTBENCH}$ will steer the development of more effective and robust audio-visual TTA approaches. Our code is available $\href{https://github.com/sarthaxxxxx/AV-C-Robustness-Benchmark}{here}$.
>
---
#### [new 014] Comparative Evaluation of Acoustic Feature Extraction Tools for Clinical Speech Analysis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音分析任务，旨在评估不同声学特征提取工具在临床语音数据中的表现差异。研究比较了三种工具（OpenSMILE、Praat、Librosa）对精神分裂症患者和健康对照组的语音特征提取效果，发现不同工具结果存在显著差异，强调需标准化方法与多工具交叉验证以提高可重复性。**

- **链接: [http://arxiv.org/pdf/2506.01129v1](http://arxiv.org/pdf/2506.01129v1)**

> **作者:** Anna Seo Gyeong Choi; Alexander Richardson; Ryan Partlan; Sunny Tang; Sunghye Cho
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This study compares three acoustic feature extraction toolkits (OpenSMILE, Praat, and Librosa) applied to clinical speech data from individuals with schizophrenia spectrum disorders (SSD) and healthy controls (HC). By standardizing extraction parameters across the toolkits, we analyzed speech samples from 77 SSD and 87 HC participants and found significant toolkit-dependent variations. While F0 percentiles showed high cross-toolkit correlation (r=0.962 to 0.999), measures like F0 standard deviation and formant values often had poor, even negative, agreement. Additionally, correlation patterns differed between SSD and HC groups. Classification analysis identified F0 mean, HNR, and MFCC1 (AUC greater than 0.70) as promising discriminators. These findings underscore reproducibility concerns and advocate for standardized protocols, multi-toolkit cross-validation, and transparent reporting.
>
---
#### [new 015] Universal Preference-Score-based Pairwise Speech Quality Assessment
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决如何有效比较两个语音生成系统的质量问题。作者提出了UPPSQA模型，先分别预测每段语音的绝对MOS分，再通过偏好函数融合为相对偏好分。为缓解偏好数据不足的问题，还构建了新的成对语音数据集。实验表明该方法在多种训练和测试场景下均优于基线模型，具有良好的通用性。**

- **链接: [http://arxiv.org/pdf/2506.01455v1](http://arxiv.org/pdf/2506.01455v1)**

> **作者:** Yu-Fei Shi; Yang Ai; Zhen-Hua Ling
>
> **摘要:** To compare the performance of two speech generation sys- tems, one of the most effective approaches is estimating the preference score between their generated speech. This pa- per proposes a novel universal preference-score-based pairwise speech quality assessment (UPPSQA) model, aimed at predict- ing the preference score between paired speech samples to de- termine which one has better quality. The model first predicts the absolute mean opinion score (MOS) for the two speech sam- ples separately, and then aggregates them into a relative prefer- ence score using a preference function. To address the scarcity of preference data, we also construct a new pairwise speech dataset based on a MOS dataset for experiments. Experimental results confirm that, whether in training scenarios with differ- ent data types and label conditions, or in both in-domain and out-of-domain test scenarios, the prediction accuracy of UPP- SQA outperforms that of the baseline models, demonstrating its universality.
>
---
#### [new 016] General-purpose audio representation learning for real-world sound scenes
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频表征学习任务，旨在解决现有音频模型在真实世界多源、空间化声音场景中表现不佳的问题。作者提出了一种自监督训练方法 GRAM，用于学习通用、自然场景下的音频表示，并结合 Transformer 和 Mamba 模型验证效果，显著提升了在复杂声学环境中的分类和定位性能。**

- **链接: [http://arxiv.org/pdf/2506.00934v1](http://arxiv.org/pdf/2506.00934v1)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **摘要:** While audio foundation models perform well on myriad of tasks from sound classification to speech analysis, these models are trained and tested on dry, non-spatial, single-source audio clips. This limits their success in real-world situations and results in spatially unaware audio embeddings. To address these limitations, we propose a novel self-supervised training approach for General-Purpose, Real-world Audio Models (GRAMs). The GRAM training approach enables robust spatial audio representation learning for naturalistic, noisy sound scenes and can be applied to any masking-based deep learning model. We demonstrate the success of our approach by training two state-of-the-art models, one with a transformer and one with a mamba backbone. We assess the quality of the extracted audio representations from GRAMs using the original version of the HEAR benchmark, a newly synthesized, naturalistic version of the HEAR benchmark, and novel sound localization tasks based on HEAR benchmark datasets. The results show that our approach minimizes the performance gap between dry, non-spatial, single-source sound scenes and naturalistic sound scenes for crucial tasks such as auditory scene analysis, outperforming existing state-of-the-art audio foundation models at a fraction of the training steps. Moreover, GRAMs show state-of-the-art performance on sound localization tasks, exceeding even supervised sound localization models. In sum, the proposed approach represents a significant advancement towards robust audio foundation models for real-world applications with state-of-the-art performance on naturalistic sound scenes as well as spatial audio representation learning.
>
---
#### [new 017] The iNaturalist Sounds Dataset
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文发布了iNaturalist Sounds Dataset（iNatSounds），包含23万条来自5500多个物种的声音，用于生物声音分析任务。旨在解决物种声音数据缺乏问题，支持生物多样性研究和生态监测。**

- **链接: [http://arxiv.org/pdf/2506.00343v1](http://arxiv.org/pdf/2506.00343v1)**

> **作者:** Mustafa Chasmai; Alexander Shepard; Subhransu Maji; Grant Van Horn
>
> **摘要:** We present the iNaturalist Sounds Dataset (iNatSounds), a collection of 230,000 audio files capturing sounds from over 5,500 species, contributed by more than 27,000 recordists worldwide. The dataset encompasses sounds from birds, mammals, insects, reptiles, and amphibians, with audio and species labels derived from observations submitted to iNaturalist, a global citizen science platform. Each recording in the dataset varies in length and includes a single species annotation. We benchmark multiple backbone architectures, comparing multiclass classification objectives with multilabel objectives. Despite weak labeling, we demonstrate that iNatSounds serves as a useful pretraining resource by benchmarking it on strongly labeled downstream evaluation datasets. The dataset is available as a single, freely accessible archive, promoting accessibility and research in this important domain. We envision models trained on this data powering next-generation public engagement applications, and assisting biologists, ecologists, and land use managers in processing large audio collections, thereby contributing to the understanding of species compositions in diverse soundscapes.
>
---
#### [new 018] Learning to Upsample and Upmix Audio in the Latent Domain
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决现有音频处理方法在原始波形或频谱上操作效率低的问题。作者提出了一种全在自编码器潜在空间内完成处理的框架，仅使用潜在L1重建和对抗判别器，简化训练并提升效率，实验证明其在带宽扩展和单声道转立体声任务中效率提高达100倍，同时保持质量。**

- **链接: [http://arxiv.org/pdf/2506.00681v1](http://arxiv.org/pdf/2506.00681v1)**

> **作者:** Dimitrios Bralios; Paris Smaragdis; Jonah Casebeer
>
> **摘要:** Neural audio autoencoders create compact latent representations that preserve perceptually important information, serving as the foundation for both modern audio compression systems and generation approaches like next-token prediction and latent diffusion. Despite their prevalence, most audio processing operations, such as spatial and spectral up-sampling, still inefficiently operate on raw waveforms or spectral representations rather than directly on these compressed representations. We propose a framework that performs audio processing operations entirely within an autoencoder's latent space, eliminating the need to decode to raw audio formats. Our approach dramatically simplifies training by operating solely in the latent domain, with a latent L1 reconstruction term, augmented by a single latent adversarial discriminator. This contrasts sharply with raw-audio methods that typically require complex combinations of multi-scale losses and discriminators. Through experiments in bandwidth extension and mono-to-stereo up-mixing, we demonstrate computational efficiency gains of up to 100x while maintaining quality comparable to post-processing on raw audio. This work establishes a more efficient paradigm for audio processing pipelines that already incorporate autoencoders, enabling significantly faster and more resource-efficient workflows across various audio tasks.
>
---
#### [new 019] FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频描述生成任务，旨在解决当前方法缺乏细节和语境准确性的问题。作者提出了FusionAudio-1.2M，一个包含120万精细描述的音频字幕数据集及600万问答对，并设计了基于多模态融合与大语言模型的两阶段生成方法，以提升复杂音频环境的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.01111v1](http://arxiv.org/pdf/2506.01111v1)**

> **作者:** Shunian Chen; Xinyuan Xie; Zheshu Chen; Liyan Zhao; Owen Lee; Zhan Su; Qilin Sun; Benyou Wang
>
> **摘要:** High-quality, large-scale audio captioning is crucial for advancing audio understanding, yet current automated methods often generate captions that lack fine-grained detail and contextual accuracy, primarily due to their reliance on limited unimodal or superficial multimodal information. Drawing inspiration from human auditory perception, which adeptly integrates cross-modal cues and performs sophisticated auditory scene analysis, we introduce a novel two-stage automated pipeline. This pipeline first employs specialized pretrained models to extract diverse contextual cues (e.g., speech, music, general sounds, and visual information from associated video). A large language model (LLM) then synthesizes these rich, multimodal inputs to generate detailed and context-aware audio captions. Key contributions of this work include: (1) the proposed scalable method for fine-grained audio caption generation; (2) FusionAudio, a new large-scale dataset comprising 1.2 million such detailed captions, combined with 6 million QA pairs; and (3) enhanced audio models developed using FusionAudio, specifically a CLAP-based audio encoder with superior audio-text alignment and instruction following. This paper paves the way for more nuanced and accurate automated understanding of complex audio environments. Code and data can be found in https://github.com/satsuki2486441738/FusionAudio.
>
---
#### [new 020] MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出MagiCodec，一种基于Transformer的音频编解码器，旨在提升音频重建质量和生成模型的语义表达能力。通过引入高斯噪声注入和潜变量正则化，优化编码token的语义表达与语言模型兼容性。任务是音频编解码，解决现有方法在模型适应性上的不足。**

- **链接: [http://arxiv.org/pdf/2506.00385v1](http://arxiv.org/pdf/2506.00385v1)**

> **作者:** Yakun Song; Jiawei Chen; Xiaobin Zhuang; Chenpeng Du; Ziyang Ma; Jian Wu; Jian Cong; Dongya Jia; Zhuo Chen; Yuping Wang; Yuxuan Wang; Xie Chen
>
> **备注:** 18 pages, 3 figures. The code and pre-trained models are available at https://github.com/Ereboas/MagiCodec
>
> **摘要:** Neural audio codecs have made significant strides in efficiently mapping raw audio waveforms into discrete token representations, which are foundational for contemporary audio generative models. However, most existing codecs are optimized primarily for reconstruction quality, often at the expense of the downstream modelability of the encoded tokens. Motivated by the need to overcome this bottleneck, we introduce $\textbf{MagiCodec}$, a novel single-layer, streaming Transformer-based audio codec. MagiCodec is designed with a multistage training pipeline that incorporates Gaussian noise injection and latent regularization, explicitly targeting the enhancement of semantic expressiveness in the generated codes while preserving high reconstruction fidelity. We analytically derive the effect of noise injection in the frequency domain, demonstrating its efficacy in attenuating high-frequency components and fostering robust tokenization. Extensive experimental evaluations show that MagiCodec surpasses state-of-the-art codecs in both reconstruction quality and downstream tasks. Notably, the tokens produced by MagiCodec exhibit Zipf-like distributions, as observed in natural languages, thereby improving compatibility with language-model-based generative architectures. The code and pre-trained models are available at https://github.com/Ereboas/MagiCodec.
>
---
#### [new 021] XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于音频深伪检测任务，旨在解决现有检测方法在跨域场景下性能急剧下降的问题。作者构建了大规模多语言跨域音频深伪基准XMAD-Bench，包含668.8小时真实与深伪语音，用于评估模型在不同语言、说话人和生成方法下的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.00462v1](http://arxiv.org/pdf/2506.00462v1)**

> **作者:** Ioan-Paul Ciobanu; Andrei-Iulian Hiji; Nicolae-Catalin Ristea; Paul Irofti; Cristian Rusu; Radu Tudor Ionescu
>
> **摘要:** Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to alleviate this issue, with many recent studies reporting accuracy rates close to 99%. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested ``in the wild''. Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as 100%, and the cross-domain performance of the same models, which is sometimes similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources. Our benchmark is publicly released at https://github.com/ristea/xmad-bench/.
>
---
#### [new 022] Learning Perceptually Relevant Temporal Envelope Morphing
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于音频生成任务，旨在解决时间包络形态插值缺乏感知基础的问题。通过人类听觉实验获取感知原则，构建大规模数据集，并训练机器学习模型，实现感知上自然的音频时间包络形态插值，提升了现有方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.01588v1](http://arxiv.org/pdf/2506.01588v1)**

> **作者:** Satvik Dixit; Sungjoon Park; Chris Donahue; Laurie M. Heller
>
> **摘要:** Temporal envelope morphing, the process of interpolating between the amplitude dynamics of two audio signals, is an emerging problem in generative audio systems that lacks sufficient perceptual grounding. Morphing of temporal envelopes in a perceptually intuitive manner should enable new methods for sound blending in creative media and for probing perceptual organization in psychoacoustics. However, existing audio morphing techniques often fail to produce intermediate temporal envelopes when input sounds have distinct temporal structures; many morphers effectively overlay both temporal structures, leading to perceptually unnatural results. In this paper, we introduce a novel workflow for learning envelope morphing with perceptual guidance: we first derive perceptually grounded morphing principles through human listening studies, then synthesize large-scale datasets encoding these principles, and finally train machine learning models to create perceptually intermediate morphs. Specifically, we present: (1) perceptual principles that guide envelope morphing, derived from our listening studies, (2) a supervised framework to learn these principles, (3) an autoencoder that learns to compress temporal envelope structures into latent representations, and (4) benchmarks for evaluating audio envelope morphs, using both synthetic and naturalistic data, and show that our approach outperforms existing methods in producing temporally intermediate morphs. All code, models, and datasets will be made publicly available upon publication.
>
---
#### [new 023] Counterfactual Activation Editing for Post-hoc Prosody and Mispronunciation Correction in TTS Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决预训练TTS模型在推理阶段对韵律控制和发音错误修正的局限性。作者提出了一种通用方法——反事实激活编辑，直接修改模型内部表示以实现后验调整，无需额外训练或特殊模块。实验表明该方法能有效改善韵律并纠正发音问题。**

- **链接: [http://arxiv.org/pdf/2506.00832v1](http://arxiv.org/pdf/2506.00832v1)**

> **作者:** Kyowoon Lee; Artyom Stitsyuk; Gunu Jho; Inchul Hwang; Jaesik Choi
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Recent advances in Text-to-Speech (TTS) have significantly improved speech naturalness, increasing the demand for precise prosody control and mispronunciation correction. Existing approaches for prosody manipulation often depend on specialized modules or additional training, limiting their capacity for post-hoc adjustments. Similarly, traditional mispronunciation correction relies on grapheme-to-phoneme dictionaries, making it less practical in low-resource settings. We introduce Counterfactual Activation Editing, a model-agnostic method that manipulates internal representations in a pre-trained TTS model to achieve post-hoc control of prosody and pronunciation. Experimental results show that our method effectively adjusts prosodic features and corrects mispronunciations while preserving synthesis quality. This opens the door to inference-time refinement of TTS outputs without retraining, bridging the gap between pre-trained TTS models and editable speech synthesis.
>
---
#### [new 024] RPRA-ADD: Forgery Trace Enhancement-Driven Audio Deepfake Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有方法在面对新型伪造技术时泛化能力不足的问题。作者提出RPRA-ADD框架，通过增强伪造痕迹感知、优化特征空间分布差异，并引入注意力机制提升对伪造音频的识别能力，从而实现更优的检测性能。**

- **链接: [http://arxiv.org/pdf/2506.00375v1](http://arxiv.org/pdf/2506.00375v1)**

> **作者:** Ruibo Fu; Xiaopeng Wang; Zhengqi Wen; Jianhua Tao; Yuankun Xie; Zhiyong Wang; Chunyu Qiang; Xuefei Liu; Cunhang Fan; Chenxing Li; Guanjun Li
>
> **摘要:** Existing methods for deepfake audio detection have demonstrated some effectiveness. However, they still face challenges in generalizing to new forgery techniques and evolving attack patterns. This limitation mainly arises because the models rely heavily on the distribution of the training data and fail to learn a decision boundary that captures the essential characteristics of forgeries. Additionally, relying solely on a classification loss makes it difficult to capture the intrinsic differences between real and fake audio. In this paper, we propose the RPRA-ADD, an integrated Reconstruction-Perception-Reinforcement-Attention networks based forgery trace enhancement-driven robust audio deepfake detection framework. First, we propose a Global-Local Forgery Perception (GLFP) module for enhancing the acoustic perception capacity of forgery traces. To significantly reinforce the feature space distribution differences between real and fake audio, the Multi-stage Dispersed Enhancement Loss (MDEL) is designed, which implements a dispersal strategy in multi-stage feature spaces. Furthermore, in order to enhance feature awareness towards forgery traces, the Fake Trace Focused Attention (FTFA) mechanism is introduced to adjust attention weights dynamically according to the reconstruction discrepancy matrix. Visualization experiments not only demonstrate that FTFA improves attention to voice segments, but also enhance the generalization capability. Experimental results demonstrate that the proposed method achieves state-of-the-art performance on 4 benchmark datasets, including ASVspoof2019, ASVspoof2021, CodecFake, and FakeSound, achieving over 20% performance improvement. In addition, it outperforms existing methods in rigorous 3*3 cross-domain evaluations across Speech, Sound, and Singing, demonstrating strong generalization capability across diverse audio domains.
>
---
#### [new 025] Improving Code Switching with Supervised Fine Tuning and GELU Adapters
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决代码切换（code switching）场景下的自动语音识别（ASR）准确性问题。由于现有代码切换数据集有限，作者利用Whisper模型的单语能力，提出“Switching Tokenizers Method”以提升转录准确率，并结合GELU适配器进行微调，有效降低了多个数据集的混合错误率（MER），优于当前最先进的方法。**

- **链接: [http://arxiv.org/pdf/2506.00291v1](http://arxiv.org/pdf/2506.00291v1)**

> **作者:** Linh Pham
>
> **摘要:** There are few code switching datasets, labeled or unlabled, that exist today. As a result, ASR requires new methods to utilize the vast monolingual data and models that exist. This paper uses OpenAI's open source ASR model, Whisper, which has been pre-trained on 680K hours of audio to perform monolingual ASR tasks. In Part 1, this paper examines how exploiting Whisper's monolingual ability to individually tokenize training text, called "Switching Tokenizers Method", improves transcription accuracy. In Part 2, we combine the Switching Tokenizers Method from part 1 and train a GELU based adapter on the encoder. These two methods reduced Total Mixed Error Rate (MER) to 9.4% for the ASCEND dataset, 6% for SEAME devman and 9.7% for SEAME devsge, outperforming current SoTA methods.
>
---
#### [new 026] FUSE: Universal Speech Enhancement using Multi-Stage Fusion of Sparse Compression and Token Generation Models for the URGENT 2025 Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升复杂噪声环境下语音质量。提出FUSE框架，分三阶段融合稀疏压缩、生成模型与原始信号，优化多语言、变采样率及多种失真类型的语音增强效果。**

- **链接: [http://arxiv.org/pdf/2506.00809v1](http://arxiv.org/pdf/2506.00809v1)**

> **作者:** Nabarun Goswami; Tatsuya Harada
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** We propose a multi-stage framework for universal speech enhancement, designed for the Interspeech 2025 URGENT Challenge. Our system first employs a Sparse Compression Network to robustly separate sources and extract an initial clean speech estimate from noisy inputs. This is followed by an efficient generative model that refines speech quality by leveraging self-supervised features and optimizing a masked language modeling objective on acoustic tokens derived from a neural audio codec. In the final stage, a fusion network integrates the outputs of the first two stages with the original noisy signal, achieving a balanced improvement in both signal fidelity and perceptual quality. Additionally, a shift trick that aggregates multiple time-shifted predictions, along with output blending, further boosts performance. Experimental results on challenging multilingual datasets with variable sampling rates and diverse distortion types validate the effectiveness of our approach.
>
---
#### [new 027] Causal Structure Discovery for Error Diagnostics of Children's ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决儿童语音识别错误分析中多因素相互依赖的问题。通过构建因果结构发现方法，揭示生理、认知、外部因素与ASR错误之间的复杂关系，并量化各因素影响。进一步分析微调模型对这些因素的缓解效果，验证方法在Whisper和Wav2Vec2.0系统上的通用性。**

- **链接: [http://arxiv.org/pdf/2506.00402v1](http://arxiv.org/pdf/2506.00402v1)**

> **作者:** Vishwanath Pratap Singh; Md. Sahidullah; Tomi Kinnunen
>
> **备注:** Interspeech 2025
>
> **摘要:** Children's automatic speech recognition (ASR) often underperforms compared to that of adults due to a confluence of interdependent factors: physiological (e.g., smaller vocal tracts), cognitive (e.g., underdeveloped pronunciation), and extrinsic (e.g., vocabulary limitations, background noise). Existing analysis methods examine the impact of these factors in isolation, neglecting interdependencies-such as age affecting ASR accuracy both directly and indirectly via pronunciation skills. In this paper, we introduce a causal structure discovery to unravel these interdependent relationships among physiology, cognition, extrinsic factors, and ASR errors. Then, we employ causal quantification to measure each factor's impact on children's ASR. We extend the analysis to fine-tuned models to identify which factors are mitigated by fine-tuning and which remain largely unaffected. Experiments on Whisper and Wav2Vec2.0 demonstrate the generalizability of our findings across different ASR systems.
>
---
#### [new 028] Chain-of-Thought Training for Open E2E Spoken Dialogue Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于对话系统任务，旨在解决端到端语音对话系统训练数据需求大、语义连贯性差的问题。作者提出了一种基于思维链（CoT）的训练策略，使模型在少量公开对话数据上也能高效训练，并提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.00722v1](http://arxiv.org/pdf/2506.00722v1)**

> **作者:** Siddhant Arora; Jinchuan Tian; Hayato Futami; Jee-weon Jung; Jiatong Shi; Yosuke Kashiwagi; Emiru Tsunoo; Shinji Watanabe
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Unlike traditional cascaded pipelines, end-to-end (E2E) spoken dialogue systems preserve full differentiability and capture non-phonemic information, making them well-suited for modeling spoken interactions. However, existing E2E approaches often require large-scale training data and generates responses lacking semantic coherence. We propose a simple yet effective strategy leveraging a chain-of-thought (CoT) formulation, ensuring that training on conversational data remains closely aligned with the multimodal language model (LM)'s pre-training on speech recognition~(ASR), text-to-speech synthesis (TTS), and text LM tasks. Our method achieves over 1.5 ROUGE-1 improvement over the baseline, successfully training spoken dialogue systems on publicly available human-human conversation datasets, while being compute-efficient enough to train on just 300 hours of public human-human conversation data, such as the Switchboard. We will publicly release our models and training code.
>
---
#### [new 029] Length Aware Speech Translation for Video Dubbing
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音翻译任务，旨在解决视频配音中译文语音与原文时长对齐的问题。作者提出了一种基于音素的端到端长度感知翻译模型（LSST）和长度感知束搜索算法（LABS），可在一次解码中生成不同长度的翻译，提升音频同步质量，实现实时设备端应用。**

- **链接: [http://arxiv.org/pdf/2506.00740v1](http://arxiv.org/pdf/2506.00740v1)**

> **作者:** Harveen Singh Chadha; Aswin Shanmugam Subramanian; Vikas Joshi; Shubham Bansal; Jian Xue; Rupeshkumar Mehta; Jinyu Li
>
> **备注:** This paper was accepted to Interspeech 2025
>
> **摘要:** In video dubbing, aligning translated audio with the source audio is a significant challenge. Our focus is on achieving this efficiently, tailored for real-time, on-device video dubbing scenarios. We developed a phoneme-based end-to-end length-sensitive speech translation (LSST) model, which generates translations of varying lengths short, normal, and long using predefined tags. Additionally, we introduced length-aware beam search (LABS), an efficient approach to generate translations of different lengths in a single decoding pass. This approach maintained comparable BLEU scores compared to a baseline without length awareness while significantly enhancing synchronization quality between source and target audio, achieving a mean opinion score (MOS) gain of 0.34 for Spanish and 0.65 for Korean, respectively.
>
---
#### [new 030] Quantifying and Reducing Speaker Heterogeneity within the Common Voice Corpus for Phonetic Analysis
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在减少Mozilla Common Voice语料库中的说话人异质性问题。通过使用ResNet语音嵌入计算相似度，进行说话人判别任务以确定最佳阈值，从而优化客户端ID，更准确地逼近真实说话人身份，提升语音分析和相关技术应用的准确性。**

- **链接: [http://arxiv.org/pdf/2506.00733v1](http://arxiv.org/pdf/2506.00733v1)**

> **作者:** Miao Zhang; Aref Farhadipour; Annie Baker; Jiachen Ma; Bogdan Pricop; Eleanor Chodroff
>
> **备注:** Accepted for Interspeech 2025
>
> **摘要:** With its crosslinguistic and cross-speaker diversity, the Mozilla Common Voice Corpus (CV) has been a valuable resource for multilingual speech technology and holds tremendous potential for research in crosslinguistic phonetics and speech sciences. Properly accounting for speaker variation is, however, key to the theoretical and statistical bases of speech research. While CV provides a client ID as an approximation to a speaker ID, multiple speakers can contribute under the same ID. This study aims to quantify and reduce heterogeneity in the client ID for a better approximation of a true, though still anonymous speaker ID. Using ResNet-based voice embeddings, we obtained a similarity score among recordings with the same client ID, then implemented a speaker discrimination task to identify an optimal threshold for reducing perceived speaker heterogeneity. These results have major downstream applications for phonetic analysis and the development of speaker-based speech technology.
>
---
#### [new 031] Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于医疗语音信号处理任务，旨在通过Rhythm Formant Analysis（RFA）提取节奏图谱特征，用于阿尔茨海默症的分类与评估。研究提出了手工特征与数据驱动融合方法，结合视觉变换器和BERT语言模型，提升了分类准确率，优于传统Mel频谱与eGeMAPs特征。**

- **链接: [http://arxiv.org/pdf/2506.00861v1](http://arxiv.org/pdf/2506.00861v1)**

> **作者:** Parismita Gogoi; Vishwanath Pratap Singh; Seema Khadirnaikar; Soma Siddhartha; Sishir Kalita; Jagabandhu Mishra; Md Sahidullah; Priyankoo Sarmah; S. R. M. Prasanna
>
> **备注:** Accepted in Interspeech, All codes are available in GitHub repo https://github.com/seemark11/DhiNirnayaAMFM
>
> **摘要:** This study explores the potential of Rhythm Formant Analysis (RFA) to capture long-term temporal modulations in dementia speech. Specifically, we introduce RFA-derived rhythm spectrograms as novel features for dementia classification and regression tasks. We propose two methodologies: (1) handcrafted features derived from rhythm spectrograms, and (2) a data-driven fusion approach, integrating proposed RFA-derived rhythm spectrograms with vision transformer (ViT) for acoustic representations along with BERT-based linguistic embeddings. We compare these with existing features. Notably, our handcrafted features outperform eGeMAPs with a relative improvement of $14.2\%$ in classification accuracy and comparable performance in the regression task. The fusion approach also shows improvement, with RFA spectrograms surpassing Mel spectrograms in classification by around a relative improvement of $13.1\%$ and a comparable regression score with the baselines.
>
---
#### [new 032] Silence is Golden: Leveraging Adversarial Examples to Nullify Audio Control in LDM-based Talking-Head Generation
- **分类: cs.GR; cs.CR; cs.CV; cs.SD**

- **简介: 该论文属于AI安全任务，旨在解决基于LDM的语音驱动 talking-head 视频生成技术可能引发的伦理问题。作者提出 Silencer 方法，通过 nullifying loss 和 anti-purification loss 抵抗音频控制与扩散净化，以保护肖像隐私。**

- **链接: [http://arxiv.org/pdf/2506.01591v1](http://arxiv.org/pdf/2506.01591v1)**

> **作者:** Yuan Gan; Jiaxu Miao; Yunze Wang; Yi Yang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Advances in talking-head animation based on Latent Diffusion Models (LDM) enable the creation of highly realistic, synchronized videos. These fabricated videos are indistinguishable from real ones, increasing the risk of potential misuse for scams, political manipulation, and misinformation. Hence, addressing these ethical concerns has become a pressing issue in AI security. Recent proactive defense studies focused on countering LDM-based models by adding perturbations to portraits. However, these methods are ineffective at protecting reference portraits from advanced image-to-video animation. The limitations are twofold: 1) they fail to prevent images from being manipulated by audio signals, and 2) diffusion-based purification techniques can effectively eliminate protective perturbations. To address these challenges, we propose Silencer, a two-stage method designed to proactively protect the privacy of portraits. First, a nullifying loss is proposed to ignore audio control in talking-head generation. Second, we apply anti-purification loss in LDM to optimize the inverted latent feature to generate robust perturbations. Extensive experiments demonstrate the effectiveness of Silencer in proactively protecting portrait privacy. We hope this work will raise awareness among the AI security community regarding critical ethical issues related to talking-head generation techniques. Code: https://github.com/yuangan/Silencer.
>
---
#### [new 033] Confidence intervals for forced alignment boundaries using model ensembles
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音对齐任务，旨在解决强制对齐边界缺乏不确定性估计的问题。作者使用神经网络集成方法，生成多个模型的对齐结果，并基于中位数和顺序统计量构建置信区间，提升了边界的估计效果，并将结果可视化输出，便于分析与诊断。**

- **链接: [http://arxiv.org/pdf/2506.01256v1](http://arxiv.org/pdf/2506.01256v1)**

> **作者:** Matthew C. Kelley
>
> **备注:** submitted for publication; 7 pages, 1 figure
>
> **摘要:** Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only a single estimate of a boundary. The present project introduces a method of deriving confidence intervals for these boundaries using a neural network ensemble technique. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each model. The alignment ensemble is then used to place the boundary at the median of the boundaries in the ensemble, and 97.85% confidence intervals are constructed using order statistics. On the Buckeye and TIMIT corpora, the ensemble boundaries show a slight improvement over using just a single model. The confidence intervals are incorporated into Praat TextGrids using a point tier, and they are also output as a table for researchers to analyze separately as diagnostics or to incorporate uncertainty into their analyses.
>
---
#### [new 034] CLAP-ART: Automated Audio Captioning with Semantic-rich Audio Representation Tokenizer
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 论文属于自动化音频描述（AAC）任务，旨在解决现有方法使用侧重波形重建而非语义表达的音频编码问题。作者提出CLAP-ART，通过量化预训练音频表示生成“语义丰富且离散”的token作为输入，提升了语言模型在AAC任务上的表现，验证了语义丰富token对音频描述的有效性。**

- **链接: [http://arxiv.org/pdf/2506.00800v1](http://arxiv.org/pdf/2506.00800v1)**

> **作者:** Daiki Takeuchi; Binh Thien Nguyen; Masahiro Yasuda; Yasunori Ohishi; Daisuke Niizumi; Noboru Harada
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** Automated Audio Captioning (AAC) aims to describe the semantic contexts of general sounds, including acoustic events and scenes, by leveraging effective acoustic features. To enhance performance, an AAC method, EnCLAP, employed discrete tokens from EnCodec as an effective input for fine-tuning a language model BART. However, EnCodec is designed to reconstruct waveforms rather than capture the semantic contexts of general sounds, which AAC should describe. To address this issue, we propose CLAP-ART, an AAC method that utilizes ``semantic-rich and discrete'' tokens as input. CLAP-ART computes semantic-rich discrete tokens from pre-trained audio representations through vector quantization. We experimentally confirmed that CLAP-ART outperforms baseline EnCLAP on two AAC benchmarks, indicating that semantic-rich discrete tokens derived from semantically rich AR are beneficial for AAC.
>
---
#### [new 035] PseudoVC: Improving One-shot Voice Conversion with Pseudo Paired Data
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音转换任务，旨在解决单次语音转换中因缺乏平行数据导致的输入不匹配问题。提出了PseudoVC方法，包括伪转换和说话人采样策略，提升模型训练效果。实验表明新方法优于现有模型。**

- **链接: [http://arxiv.org/pdf/2506.01039v1](http://arxiv.org/pdf/2506.01039v1)**

> **作者:** Songjun Cao; Qinghua Wu; Jie Chen; Jin Li; Long Ma
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** As parallel training data is scarce for one-shot voice conversion (VC) tasks, waveform reconstruction is typically performed by various VC systems. A typical one-shot VC system comprises a content encoder and a speaker encoder. However, two types of mismatches arise: one for the inputs to the content encoder during training and inference, and another for the inputs to the speaker encoder. To address these mismatches, we propose a novel VC training method called \textit{PseudoVC} in this paper. First, we introduce an innovative information perturbation approach named \textit{Pseudo Conversion} to tackle the first mismatch problem. This approach leverages pretrained VC models to convert the source utterance into a perturbed utterance, which is fed into the content encoder during training. Second, we propose an approach termed \textit{Speaker Sampling} to resolve the second mismatch problem, which will substitute the input to the speaker encoder by another utterance from the same speaker during training. Experimental results demonstrate that our proposed \textit{Pseudo Conversion} outperforms previous information perturbation methods, and the overall \textit{PseudoVC} method surpasses publicly available VC models. Audio examples are available.
>
---
#### [new 036] PARROT: Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在提升模型性能。它提出了一种新框架PARROT，通过结合Mamba和基于注意力的预训练模型，利用并行分支融合、最优传输和Hadamard积来发挥异构模型的优势。实验表明，该方法在语音情感识别上达到了当前最优效果。**

- **链接: [http://arxiv.org/pdf/2506.01138v1](http://arxiv.org/pdf/2506.01138v1)**

> **作者:** Orchid Chetia Phukan; Mohd Mujtaba Akhtar; Girish; Swarup Ranjan Behera; Jaya Sai Kiran Patibandla; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** The emergence of Mamba as an alternative to attention-based architectures has led to the development of Mamba-based self-supervised learning (SSL) pre-trained models (PTMs) for speech and audio processing. Recent studies suggest that these models achieve comparable or superior performance to state-of-the-art (SOTA) attention-based PTMs for speech emotion recognition (SER). Motivated by prior work demonstrating the benefits of PTM fusion across different speech processing tasks, we hypothesize that leveraging the complementary strengths of Mamba-based and attention-based PTMs will enhance SER performance beyond the fusion of homogenous attention-based PTMs. To this end, we introduce a novel framework, PARROT that integrates parallel branch fusion with Optimal Transport and Hadamard Product. Our approach achieves SOTA results against individual PTMs, homogeneous PTMs fusion, and baseline fusion techniques, thus, highlighting the potential of heterogeneous PTM fusion for SER.
>
---
#### [new 037] WCTC-Biasing: Retraining-free Contextual Biasing ASR with Wildcard CTC-based Keyword Spotting and Inter-layer Biasing
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决端到端模型对训练数据词汇偏差导致的未知词（如专有名词）识别不准问题。作者提出WCTC-Biasing方法，通过在推理过程中使用通配CTC进行关键词检测，并在后续层施加偏置，无需重新训练模型即可提升未知词识别效果。实验显示其在日本语数据上使未知词F1得分提升了29%。**

- **链接: [http://arxiv.org/pdf/2506.01263v1](http://arxiv.org/pdf/2506.01263v1)**

> **作者:** Yu Nakagome; Michael Hentschel
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Despite recent advances in end-to-end speech recognition methods, the output tends to be biased to the training data's vocabulary, resulting in inaccurate recognition of proper nouns and other unknown terms. To address this issue, we propose a method to improve recognition accuracy of such rare words in CTC-based models without additional training or text-to-speech systems. Specifically, keyword spotting is performed using acoustic features of intermediate layers during inference, and a bias is applied to the subsequent layers of the acoustic model for detected keywords. For keyword detection, we adopt a wildcard CTC that is both fast and tolerant of ambiguous matches, allowing flexible handling of words that are difficult to match strictly. Since this method does not require retraining of existing models, it can be easily applied to even large-scale models. In experiments on Japanese speech recognition, the proposed method achieved a 29% improvement in the F1 score for unknown words.
>
---
#### [new 038] SoundSculpt: Direction and Semantics Driven Ambisonic Target Sound Extraction
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出SoundSculpt，用于从全向声音录制中提取目标声场。它使用神经网络，结合空间信息和语义嵌入进行条件控制，提高了提取效果，尤其在目标与干扰声源位置接近时表现更优。**

- **链接: [http://arxiv.org/pdf/2506.00273v1](http://arxiv.org/pdf/2506.00273v1)**

> **作者:** Tuochao Chen; D Shin; Hakan Erdogan; Sinan Hersek
>
> **摘要:** This paper introduces SoundSculpt, a neural network designed to extract target sound fields from ambisonic recordings. SoundSculpt employs an ambisonic-in-ambisonic-out architecture and is conditioned on both spatial information (e.g., target direction obtained by pointing at an immersive video) and semantic embeddings (e.g., derived from image segmentation and captioning). Trained and evaluated on synthetic and real ambisonic mixtures, SoundSculpt demonstrates superior performance compared to various signal processing baselines. Our results further reveal that while spatial conditioning alone can be effective, the combination of spatial and semantic information is beneficial in scenarios where there are secondary sound sources spatially close to the target. Additionally, we compare two different semantic embeddings derived from a text description of the target sound using text encoders.
>
---
#### [new 039] Towards Fusion of Neural Audio Codec-based Representations with Spectral for Heart Murmur Classification via Bandit-based Cross-Attention Mechanism
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于心音杂音分类任务，旨在提升分类性能。通过融合神经音频编解码器表示与频谱特征，并提出基于多臂老虎机的跨注意力机制（BAOMI），有效结合两者互补优势，抑制噪声，实现更优分类效果。**

- **链接: [http://arxiv.org/pdf/2506.01148v1](http://arxiv.org/pdf/2506.01148v1)**

> **作者:** Orchid Chetia Phukan; Girish; Mohd Mujtaba Akhtar; Swarup Ranjan Behera; Priyabrata Mallick; Santanu Roy; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this study, we focus on heart murmur classification (HMC) and hypothesize that combining neural audio codec representations (NACRs) such as EnCodec with spectral features (SFs), such as MFCC, will yield superior performance. We believe such fusion will trigger their complementary behavior as NACRs excel at capturing fine-grained acoustic patterns such as rhythm changes, spectral features focus on frequency-domain properties such as harmonic structure, spectral energy distribution crucial for analyzing the complex of heart sounds. To this end, we propose, BAOMI, a novel framework banking on novel bandit-based cross-attention mechanism for effective fusion. Here, a agent provides more weightage to most important heads in multi-head cross-attention mechanism and helps in mitigating the noise. With BAOMI, we report the topmost performance in comparison to individual NACRs, SFs, and baseline fusion techniques and setting new state-of-the-art.
>
---
#### [new 040] M3ANet: Multi-scale and Multi-Modal Alignment Network for Brain-Assisted Target Speaker Extraction
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于脑辅助目标说话人提取（TSE）任务，旨在通过脑电信号（如EEG）从混合语音中提取关注语音。当前模型存在语音与EEG模态间时间不对齐及语音编码能力不足的问题。为此，论文提出M3ANet，包含模态对齐模块以解决时间不一致，并采用多尺度卷积与GroupMamba模块提升语音信息提取能力。实验表明该方法在多个数据集上优于现有模型。**

- **链接: [http://arxiv.org/pdf/2506.00466v1](http://arxiv.org/pdf/2506.00466v1)**

> **作者:** Cunhang Fan; Ying Chen; Jian Zhou; Zexu Pan; Jingjing Zhang; Youdian Gao; Xiaoke Yang; Zhengqi Wen; Zhao Lv
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** The brain-assisted target speaker extraction (TSE) aims to extract the attended speech from mixed speech by utilizing the brain neural activities, for example Electroencephalography (EEG). However, existing models overlook the issue of temporal misalignment between speech and EEG modalities, which hampers TSE performance. In addition, the speech encoder in current models typically uses basic temporal operations (e.g., one-dimensional convolution), which are unable to effectively extract target speaker information. To address these issues, this paper proposes a multi-scale and multi-modal alignment network (M3ANet) for brain-assisted TSE. Specifically, to eliminate the temporal inconsistency between EEG and speech modalities, the modal alignment module that uses a contrastive learning strategy is applied to align the temporal features of both modalities. Additionally, to fully extract speech information, multi-scale convolutions with GroupMamba modules are used as the speech encoder, which scans speech features at each scale from different directions, enabling the model to capture deep sequence information. Experimental results on three publicly available datasets show that the proposed model outperforms current state-of-the-art methods across various evaluation metrics, highlighting the effectiveness of our proposed method. The source code is available at: https://github.com/fchest/M3ANet.
>
---
#### [new 041] Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于零样本语音转换任务，旨在将源说话人音色转换为任意目标音色，同时保留语音内容。针对现有方法难以精细控制节奏的问题，提出R-VC模型，通过数据扰动、内容建模与扩散变换技术，实现高效、节奏可控的语音转换，在音色相似度、自然度和风格迁移方面表现优异。**

- **链接: [http://arxiv.org/pdf/2506.01014v1](http://arxiv.org/pdf/2506.01014v1)**

> **作者:** Jialong Zuo; Shengpeng Ji; Minghui Fang; Mingze Li; Ziyue Jiang; Xize Cheng; Xiaoda Yang; Chen Feiyang; Xinyu Duan; Zhou Zhao
>
> **备注:** Accepted by ACL 2025 (Main Conference)
>
> **摘要:** Zero-Shot Voice Conversion (VC) aims to transform the source speaker's timbre into an arbitrary unseen one while retaining speech content. Most prior work focuses on preserving the source's prosody, while fine-grained timbre information may leak through prosody, and transferring target prosody to synthesized speech is rarely studied. In light of this, we propose R-VC, a rhythm-controllable and efficient zero-shot voice conversion model. R-VC employs data perturbation techniques and discretize source speech into Hubert content tokens, eliminating much content-irrelevant information. By leveraging a Mask Generative Transformer for in-context duration modeling, our model adapts the linguistic content duration to the desired target speaking style, facilitating the transfer of the target speaker's rhythm. Furthermore, R-VC introduces a powerful Diffusion Transformer (DiT) with shortcut flow matching during training, conditioning the network not only on the current noise level but also on the desired step size, enabling high timbre similarity and quality speech generation in fewer sampling steps, even in just two, thus minimizing latency. Experimental results show that R-VC achieves comparable speaker similarity to state-of-the-art VC methods with a smaller dataset, and surpasses them in terms of speech naturalness, intelligibility and style transfer performance.
>
---
#### [new 042] Self-Supervised Speech Quality Assessment (S3QA): Leveraging Speech Foundation Models for a Scalable Speech Quality Metric
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音质量评估任务，旨在解决传统人工评分（如MOS）在一致性、泛化性和效率方面的不足。作者提出S3QA模型，利用自监督学习和预训练语音模型WavLM，通过计算干净与退化语音之间的余弦距离作为训练目标，训练一个Transformer模型预测语音质量。实验表明该方法在多个测试集上有效，并与人工评分及语音技术性能对齐。**

- **链接: [http://arxiv.org/pdf/2506.01655v1](http://arxiv.org/pdf/2506.01655v1)**

> **作者:** Mattson Ogg; Caitlyn Bishop; Han Yi; Sarah Robinson
>
> **备注:** Five tables, three figures, twelve pages
>
> **摘要:** Methods for automatically assessing speech quality are critical for many human language technologies. Behavioral ratings provided by human raters (e.g., mean opinion scores; MOS) are considered the gold standard, but they are susceptible to variability between individual raters, cannot easily be generalized across corpora, and are labor-intensive to collect, thus limiting the acoustic challenges they can quantify. Here, we present a new, scalable method for automatically assessing speech quality: the self-supervised speech quality assessment (S3QA) model. First, we processed high quality utterances from multiple speech corpora, using a wide range of acoustic manipulations intended to emulate common sources of quality degradation in the real-world: frequency filtering, reverberation, background noise, and digital compression. Second, we leveraged an existing, pre-trained speech foundation model, WavLM, to computationally derive a self-supervised training target for the level of signal degradation by calculating the cosine distances between the clean and degraded versions of each utterance in the embedding space. Next, we trained a transformer-based model to predict the cosine distance, or degradation index, given only the degraded versions of these utterances. Finally, the trained model was evaluated on unseen test corpora of synthetic mixtures, NISQA, and VOiCES. We show that the S3QA model trained on this task performs well and is aligned with both behavioral ratings (MOS), speech technology performance (automatic speech recognition) and other important features of the held-out data (e.g., microphone distances). This approach provides an automated, scalable method for assessing speech quality across a wide range of acoustic challenges, and could easily be adapted to other use cases where acoustic simulations are available.
>
---
#### [new 043] HASRD: Hierarchical Acoustic and Semantic Representation Disentanglement
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音表示学习任务，旨在解决现有方法难以同时兼顾语义相关性与声学保真度的问题。论文提出HASRD框架，通过分层离散化自监督学习表示，将语义与声学信息解耦，提升了语音识别性能和重建质量。**

- **链接: [http://arxiv.org/pdf/2506.00843v1](http://arxiv.org/pdf/2506.00843v1)**

> **作者:** Amir Hussein; Sameer Khurana; Gordon Wichern; Francois G. Germain; Jonathan Le Roux
>
> **摘要:** Effective speech representations for spoken language models must balance semantic relevance with acoustic fidelity for high-quality reconstruction. However, existing approaches struggle to achieve both simultaneously. To address this, we introduce Hierarchical Acoustic and Semantic Representation Disentanglement (HASRD, pronounced `hazard'), a framework that factorizes self-supervised learning representations into discrete semantic and acoustic tokens. HASRD assigns the semantic representation to the first codebook, while encoding acoustic residuals in subsequent codebooks. This preserves ASR performance while achieving high-quality reconstruction. Additionally, we enhance HASRD's encoder efficiency, improving ASR performance without compromising reconstruction quality. Compared to SpeechTokenizer, HASRD achieves a 44% relative WER improvement, superior reconstruction quality, and 2x lower bitrate, demonstrating its effectiveness in disentangling acoustic and semantic information.
>
---
#### [new 044] anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding
- **分类: cs.CL; cs.AI; eess.SP**

- **简介: 该论文属于医疗AI任务，旨在解决现有ECG分析模型输入单一、任务局限的问题。作者构建了anyECG-chat模型，支持灵活输入（多导联、长时程）及多任务理解（如报告生成、异常定位）。同时创建anyECG数据集以丰富训练与评估场景。**

- **链接: [http://arxiv.org/pdf/2506.00942v1](http://arxiv.org/pdf/2506.00942v1)**

> **作者:** Haitao Li; Ziyu Li; Yiheng Mao; Ziyi Liu; Zhoujian Sun; Zhengxing Huang
>
> **摘要:** The advent of multimodal large language models (MLLMs) has sparked interest in their application to electrocardiogram (ECG) analysis. However, existing ECG-focused MLLMs primarily focus on report generation tasks, often limited to single 12-lead, short-duration (10s) ECG inputs, thereby underutilizing the potential of MLLMs. To this end, we aim to develop a MLLM for ECG analysis that supports a broader range of tasks and more flexible ECG inputs. However, existing ECG-QA datasets are often monotonous. To address this gap, we first constructed the anyECG dataset, which encompasses a wide variety of tasks, including report generation, abnormal waveform localization, and open-ended question answering. In addition to standard hospital ECGs, we introduced long-duration reduced-lead ECGs for home environments and multiple ECG comparison scenarios commonly encountered in clinical practice. Furthermore, we propose the anyECG-chat model, which supports dynamic-length ECG inputs and multiple ECG inputs. We trained the model using a three-stage curriculum training recipe with the anyECG dataset. A comprehensive evaluation was conducted, demonstrating that anyECG-chat is capable of supporting various practical application scenarios, including not only common report generation tasks but also abnormal waveform localization for long-duration reduced-lead ECGs in home environments and comprehensive comparative analysis of multiple ECGs.
>
---
#### [new 045] Feature Fusion and Knowledge-Distilled Multi-Modal Multi-Target Detection
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于多目标检测任务，旨在解决异构数据源下资源受限设备的多目标检测难题。通过融合RGB与热成像特征，并引入知识蒸馏提升域适应性，实现精度优化。实验表明其方法在保持高精度的同时显著降低推理时间。**

- **链接: [http://arxiv.org/pdf/2506.00365v1](http://arxiv.org/pdf/2506.00365v1)**

> **作者:** Ngoc Tuyen Do; Tri Nhu Do
>
> **摘要:** In the surveillance and defense domain, multi-target detection and classification (MTD) is considered essential yet challenging due to heterogeneous inputs from diverse data sources and the computational complexity of algorithms designed for resource-constrained embedded devices, particularly for Al-based solutions. To address these challenges, we propose a feature fusion and knowledge-distilled framework for multi-modal MTD that leverages data fusion to enhance accuracy and employs knowledge distillation for improved domain adaptation. Specifically, our approach utilizes both RGB and thermal image inputs within a novel fusion-based multi-modal model, coupled with a distillation training pipeline. We formulate the problem as a posterior probability optimization task, which is solved through a multi-stage training pipeline supported by a composite loss function. This loss function effectively transfers knowledge from a teacher model to a student model. Experimental results demonstrate that our student model achieves approximately 95% of the teacher model's mean Average Precision while reducing inference time by approximately 50%, underscoring its suitability for practical MTD deployment scenarios.
>
---
#### [new 046] GigaAM: Efficient Self-Supervised Learner for Speech Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升自监督学习在语音识别中的效果。通过引入基于语音识别模型的掩码语言建模和动态分块注意力机制，实现高效预训练。论文提出了GigaAM模型，在俄语语音识别上超越了Whisper-large-v3，并开源了模型与代码。**

- **链接: [http://arxiv.org/pdf/2506.01192v1](http://arxiv.org/pdf/2506.01192v1)**

> **作者:** Aleksandr Kutsakov; Alexandr Maximenko; Georgii Gospodinov; Pavel Bogomolov; Fyodor Minkin
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Self-Supervised Learning (SSL) has demonstrated strong performance in speech processing, particularly in automatic speech recognition. In this paper, we explore an SSL pretraining framework that leverages masked language modeling with targets derived from a speech recognition model. We also present chunkwise attention with dynamic chunk size sampling during pretraining to enable both full-context and streaming fine-tuning. Our experiments examine scaling with respect to model size and the amount of data. Using our method, we train the GigaAM family of models, including a state-of-the-art model for Russian speech recognition that outperforms Whisper-large-v3 by 50%. We have released our foundation and ASR models, along with the inference code, under the MIT license as open-source resources to the research community. Available at https://github.com/salute-developers/gigaam.
>
---
#### [new 047] Zero-Shot Text-to-Speech for Vietnamese
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决越南语零样本语音合成问题。作者构建了高质量数据集 PhoAudiobook，并基于此评估了三种主流模型的性能，发现 VALL-E 和 VoiceCraft 在短句合成中表现优异，推动越南语 TTS 研究进展。**

- **链接: [http://arxiv.org/pdf/2506.01322v1](http://arxiv.org/pdf/2506.01322v1)**

> **作者:** Thi Vu; Linh The Nguyen; Dat Quoc Nguyen
>
> **备注:** To appear in Proceedings of ACL 2025 (Main conference paper)
>
> **摘要:** This paper introduces PhoAudiobook, a newly curated dataset comprising 941 hours of high-quality audio for Vietnamese text-to-speech. Using PhoAudiobook, we conduct experiments on three leading zero-shot TTS models: VALL-E, VoiceCraft, and XTTS-V2. Our findings demonstrate that PhoAudiobook consistently enhances model performance across various metrics. Moreover, VALL-E and VoiceCraft exhibit superior performance in synthesizing short sentences, highlighting their robustness in handling diverse linguistic contexts. We publicly release PhoAudiobook to facilitate further research and development in Vietnamese text-to-speech.
>
---
#### [new 048] Crowdsourcing MUSHRA Tests in the Age of Generative Speech Technologies: A Comparative Analysis of Subjective and Objective Testing Methods
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音质量评估任务，旨在解决传统MUSHRA测试依赖专家、成本高的问题。作者提出使用众包非专家听众进行MUSHRA测试，并对比特拉华平台与专家结果的一致性，验证其可靠性，同时评估六种客观指标在生成模型中的有效性，发现传统指标低估了生成模型的表现。**

- **链接: [http://arxiv.org/pdf/2506.00950v1](http://arxiv.org/pdf/2506.00950v1)**

> **作者:** Laura Lechler; Chamran Moradi; Ivana Balic
>
> **备注:** This is a preprint of a paper submitted to and accepted for INTERSPEECH 2025
>
> **摘要:** The MUSHRA framework is widely used for detecting subtle audio quality differences but traditionally relies on expert listeners in controlled environments, making it costly and impractical for model development. As a result, objective metrics are often used during development, with expert evaluations conducted later. While effective for traditional DSP codecs, these metrics often fail to reliably evaluate generative models. This paper proposes adaptations for conducting MUSHRA tests with non-expert, crowdsourced listeners, focusing on generative speech codecs. We validate our approach by comparing results from MTurk and Prolific crowdsourcing platforms with expert listener data, assessing test-retest reliability and alignment. Additionally, we evaluate six objective metrics, showing that traditional metrics undervalue generative models. Our findings reveal platform-specific biases and emphasize codec-aware metrics, offering guidance for scalable perceptual testing of speech codecs.
>
---
#### [new 049] Neuro2Semantic: A Transfer Learning Framework for Semantic Reconstruction of Continuous Language from Human Intracranial EEG
- **分类: cs.CL; eess.AS; eess.SP**

- **简介: 该论文属于神经解码与自然语言处理交叉任务，旨在从人类颅内脑电信号（iEEG）中重建连续语言的语义内容。论文提出Neuro2Semantic框架，分两阶段将神经信号对齐并生成自然文本，解决了低数据量下语言解码效果差的问题，提升了实用性与生成质量。**

- **链接: [http://arxiv.org/pdf/2506.00381v1](http://arxiv.org/pdf/2506.00381v1)**

> **作者:** Siavash Shams; Richard Antonello; Gavin Mischler; Stephan Bickel; Ashesh Mehta; Nima Mesgarani
>
> **备注:** Accepted at Interspeech 2025 Code at https://github.com/SiavashShams/neuro2semantic
>
> **摘要:** Decoding continuous language from neural signals remains a significant challenge in the intersection of neuroscience and artificial intelligence. We introduce Neuro2Semantic, a novel framework that reconstructs the semantic content of perceived speech from intracranial EEG (iEEG) recordings. Our approach consists of two phases: first, an LSTM-based adapter aligns neural signals with pre-trained text embeddings; second, a corrector module generates continuous, natural text directly from these aligned embeddings. This flexible method overcomes the limitations of previous decoding approaches and enables unconstrained text generation. Neuro2Semantic achieves strong performance with as little as 30 minutes of neural data, outperforming a recent state-of-the-art method in low-data settings. These results highlight the potential for practical applications in brain-computer interfaces and neural decoding technologies.
>
---
#### [new 050] Vedavani: A Benchmark Corpus for ASR on Vedic Sanskrit Poetry
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决梵语诗歌自动语音识别（ASR）的难题。由于梵语语音复杂且缺乏相关数据集，研究提出首个针对吠陀梵语诗歌的ASR数据集Vedavani，包含54小时标注音频，并在多种多语言模型上进行基准测试，结果显示IndicWhisper表现最佳。**

- **链接: [http://arxiv.org/pdf/2506.00145v1](http://arxiv.org/pdf/2506.00145v1)**

> **作者:** Sujeet Kumar; Pretam Ray; Abhinay Beerukuri; Shrey Kamoji; Manoj Balaji Jagadeeshan; Pawan Goyal
>
> **摘要:** Sanskrit, an ancient language with a rich linguistic heritage, presents unique challenges for automatic speech recognition (ASR) due to its phonemic complexity and the phonetic transformations that occur at word junctures, similar to the connected speech found in natural conversations. Due to these complexities, there has been limited exploration of ASR in Sanskrit, particularly in the context of its poetic verses, which are characterized by intricate prosodic and rhythmic patterns. This gap in research raises the question: How can we develop an effective ASR system for Sanskrit, particularly one that captures the nuanced features of its poetic form? In this study, we introduce Vedavani, the first comprehensive ASR study focused on Sanskrit Vedic poetry. We present a 54-hour Sanskrit ASR dataset, consisting of 30,779 labelled audio samples from the Rig Veda and Atharva Veda. This dataset captures the precise prosodic and rhythmic features that define the language. We also benchmark the dataset on various state-of-the-art multilingual speech models.$^{1}$ Experimentation revealed that IndicWhisper performed the best among the SOTA models.
>
---
#### [new 051] IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling
- **分类: eess.AS; cs.SD**

- **简介: 论文属于文本到音频生成任务，旨在解决扩散模型推理速度慢的问题。作者提出IMPACT框架，结合扩散建模与基于掩码的并行解码，在连续潜在空间中进行生成，兼顾高质量音频输出与快速推理，实验证明其在保持高保真度的同时显著降低延迟。**

- **链接: [http://arxiv.org/pdf/2506.00736v1](http://arxiv.org/pdf/2506.00736v1)**

> **作者:** Kuan-Po Huang; Shu-wen Yang; Huy Phan; Bo-Ru Lu; Byeonggeun Kim; Sashank Macha; Qingming Tang; Shalini Ghosh; Hung-yi Lee; Chieh-Chi Kao; Chao Wang
>
> **备注:** Accepted by ICML 2025. Project website: https://audio-impact.github.io/
>
> **摘要:** Text-to-audio generation synthesizes realistic sounds or music given a natural language prompt. Diffusion-based frameworks, including the Tango and the AudioLDM series, represent the state-of-the-art in text-to-audio generation. Despite achieving high audio fidelity, they incur significant inference latency due to the slow diffusion sampling process. MAGNET, a mask-based model operating on discrete tokens, addresses slow inference through iterative mask-based parallel decoding. However, its audio quality still lags behind that of diffusion-based models. In this work, we introduce IMPACT, a text-to-audio generation framework that achieves high performance in audio quality and fidelity while ensuring fast inference. IMPACT utilizes iterative mask-based parallel decoding in a continuous latent space powered by diffusion modeling. This approach eliminates the fidelity constraints of discrete tokens while maintaining competitive inference speed. Results on AudioCaps demonstrate that IMPACT achieves state-of-the-art performance on key metrics including Fr\'echet Distance (FD) and Fr\'echet Audio Distance (FAD) while significantly reducing latency compared to prior models. The project website is available at https://audio-impact.github.io/.
>
---
#### [new 052] Inter-Speaker Relative Cues for Text-Guided Target Speech Extraction
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，旨在通过文本引导提取目标说话人语音。利用说话人间的相对线索（如时间顺序、性别等）提升复杂场景下的语音分离效果，并验证了多种线索的有效性及模型优化方法。**

- **链接: [http://arxiv.org/pdf/2506.01483v1](http://arxiv.org/pdf/2506.01483v1)**

> **作者:** Wang Dai; Archontis Politis; Tuomas Virtanen
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We propose a novel approach that utilize inter-speaker relative cues for distinguishing target speakers and extracting their voices from mixtures. Continuous cues (e.g., temporal order, age, pitch level) are grouped by relative differences, while discrete cues (e.g., language, gender, emotion) retain their categories. Relative cues offers greater flexibility than fixed speech attribute classification, facilitating much easier expansion of text-guided target speech extraction datasets. Our experiments show that combining all relative cues yields better performance than random subsets, with gender and temporal order being the most robust across languages and reverberant conditions. Additional cues like pitch level, loudness, distance, speaking duration, language, and pitch range also demonstrate notable benefit in complex scenarios. Fine-tuning pre-trained WavLM Base+ CNN encoders improves overall performance over the baseline of using only a Conv1d encoder.
>
---
#### [new 053] Mispronunciation Detection Without L2 Pronunciation Dataset in Low-Resource Setting: A Case Study in Finland Swedish
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决低资源语言芬兰瑞典语中的发音错误检测问题。作者提出一种无需二语发音数据集的检测模型，通过多语言wav2vec 2.0结合熵正则化等方法实现，取得了较好的召回率和精确率平衡。**

- **链接: [http://arxiv.org/pdf/2506.01156v1](http://arxiv.org/pdf/2506.01156v1)**

> **作者:** Nhan Phan; Mikko Kuronen; Maria Kautonen; Riikka Ullakonoja; Anna von Zansen; Yaroslav Getman; Ekaterina Voskoboinik; Tamás Grósz; Mikko Kurimo
>
> **备注:** Accepted to Interspeech 2025 conference
>
> **摘要:** Mispronunciation detection (MD) models are the cornerstones of many language learning applications. Unfortunately, most systems are built for English and other major languages, while low-resourced language varieties, such as Finland Swedish (FS), lack such tools. In this paper, we introduce our MD model for FS, trained on 89 hours of first language (L1) speakers' spontaneous speech and tested on 33 minutes of L2 transcribed read-aloud speech. We trained a multilingual wav2vec 2.0 model with entropy regularization, followed by temperature scaling and top-k normalization after the inference to better adapt it for MD. The main novelty of our method lies in its simplicity, requiring minimal L2 data. The process is also language-independent, making it suitable for other low-resource languages. Our proposed algorithm allows us to balance Recall (43.2%) and Precision (29.8%), compared with the baseline model's Recall (77.5%) and Precision (17.6%).
>
---
#### [new 054] Online Audio-Visual Autoregressive Speaker Extraction
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音提取任务，旨在解决流媒体场景下音频-视觉信息利用不充分的问题。作者设计了轻量级视觉前端和自回归声学编码器，提升模型效率与注意力切换鲁棒性。实验验证了方法在LRS3数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2506.01270v1](http://arxiv.org/pdf/2506.01270v1)**

> **作者:** Zexu Pan; Wupeng Wang; Shengkui Zhao; Chong Zhang; Kun Zhou; Yukun Ma; Bin Ma
>
> **备注:** Interspeech2025
>
> **摘要:** This paper proposes a novel online audio-visual speaker extraction model. In the streaming regime, most studies optimize the audio network only, leaving the visual frontend less explored. We first propose a lightweight visual frontend based on depth-wise separable convolution. Then, we propose a lightweight autoregressive acoustic encoder to serve as the second cue, to actively explore the information in the separated speech signal from past steps. Scenario-wise, for the first time, we study how the algorithm performs when there is a change in focus of attention, i.e., the target speaker. Experimental results on LRS3 datasets show that our visual frontend performs comparably to the previous state-of-the-art on both SkiM and ConvTasNet audio backbones with only 0.1 million network parameters and 2.1 MACs per second of processing. The autoregressive acoustic encoder provides an additional 0.9 dB gain in terms of SI-SNRi, and its momentum is robust against the change in attention.
>
---
#### [new 055] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决现有模型未能充分利用双通道语音数据进行自然对话的问题。作者提出了一种新方法NTPP，通过下一对词预测实现双通道对话建模，并验证其在对话能力与推理效率上的提升。**

- **链接: [http://arxiv.org/pdf/2506.00975v1](http://arxiv.org/pdf/2506.00975v1)**

> **作者:** Qichao Wang; Ziqiao Meng; Wenqian Cui; Yifei Zhang; Pengcheng Wu; Bingzhe Wu; Irwin King; Liang Chen; Peilin Zhao
>
> **摘要:** Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
>
---
#### [new 056] Towards Temporally Explainable Dysarthric Speech Clarity Assessment
- **分类: eess.AS; cs.HC; cs.SD**

- **简介: 该论文属于语音处理与医疗辅助技术任务，旨在解决构音障碍患者的发音清晰度评估问题。作者构建了一个三阶段可解释框架，包括整体清晰度评分、发音错误定位和类型分类，并分析了预训练ASR模型在各阶段的效果，以提供可操作的反馈，支持患者自主练习和治疗师干预。**

- **链接: [http://arxiv.org/pdf/2506.00454v1](http://arxiv.org/pdf/2506.00454v1)**

> **作者:** Seohyun Park; Chitralekha Gupta; Michelle Kah Yian Kwan; Xinhui Fung; Alexander Wenjun Yip; Suranga Nanayakkara
>
> **备注:** Accepted in Interspeech 2025. First two authors were equal contributors
>
> **摘要:** Dysarthria, a motor speech disorder, affects intelligibility and requires targeted interventions for effective communication. In this work, we investigate automated mispronunciation feedback by collecting a dysarthric speech dataset from six speakers reading two passages, annotated by a speech therapist with temporal markers and mispronunciation descriptions. We design a three-stage framework for explainable mispronunciation evaluation: (1) overall clarity scoring, (2) mispronunciation localization, and (3) mispronunciation type classification. We systematically analyze pretrained Automatic Speech Recognition (ASR) models in each stage, assessing their effectiveness in dysarthric speech evaluation (Code available at: https://github.com/augmented-human-lab/interspeech25_speechtherapy, Supplementary webpage: https://apps.ahlab.org/interspeech25_speechtherapy/). Our findings offer clinically relevant insights for automating actionable feedback for pronunciation assessment, which could enable independent practice for patients and help therapists deliver more effective interventions.
>
---
#### [new 057] Pushing the Limits of Beam Search Decoding for Transducer-based ASR models
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 论文研究语音识别任务，旨在解决基于Transducer模型的束搜索解码速度慢的问题。通过批处理、树结构假设、空白评分优化和CUDA图执行，提出ALSD++与AES++算法，加速束搜索并提升识别准确率及低资源场景下的浅层融合效果。**

- **链接: [http://arxiv.org/pdf/2506.00185v1](http://arxiv.org/pdf/2506.00185v1)**

> **作者:** Lilit Grigoryan; Vladimir Bataev; Andrei Andrusenko; Hainan Xu; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Transducer models have emerged as a promising choice for end-to-end ASR systems, offering a balanced trade-off between recognition accuracy, streaming capabilities, and inference speed in greedy decoding. However, beam search significantly slows down Transducers due to repeated evaluations of key network components, limiting practical applications. This paper introduces a universal method to accelerate beam search for Transducers, enabling the implementation of two optimized algorithms: ALSD++ and AES++. The proposed method utilizes batch operations, a tree-based hypothesis structure, novel blank scoring for enhanced shallow fusion, and CUDA graph execution for efficient GPU inference. This narrows the speed gap between beam and greedy modes to only 10-20% for the whole system, achieves 14-30% relative improvement in WER compared to greedy decoding, and improves shallow fusion for low-resource up to 11% compared to existing implementations. All the algorithms are open sourced.
>
---
#### [new 058] What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文研究自监督语音模型对荷兰语语言特征的编码能力，属于语音表示学习任务。旨在探讨语言特异性预训练是否提升模型对特定语言的语音和词汇信息的捕捉。通过对比不同语言预训练模型在荷兰语特征解码及语音识别性能的表现，验证了语言特异性预训练的优势。**

- **链接: [http://arxiv.org/pdf/2506.00981v1](http://arxiv.org/pdf/2506.00981v1)**

> **作者:** Marianne de Heer Kloots; Hosein Mohebbi; Charlotte Pouw; Gaofei Shen; Willem Zuidema; Martijn Bentum
>
> **备注:** Accepted to Interspeech 2025. For model, code, and materials, see https://github.com/mdhk/SSL-NL-eval
>
> **摘要:** How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition.
>
---
#### [new 059] Lessons Learned from the URGENT 2024 Speech Enhancement Challenge
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于语音增强任务，旨在提升模型在多场景下的通用性与鲁棒性。论文重点分析了数据清洗和评估指标两个关键问题，指出了传统流程中的带宽不匹配、标签噪声及缺乏有效系统与难度衡量等问题，并提出结合多种评估指标以更好地反映人类判断。**

- **链接: [http://arxiv.org/pdf/2506.01611v1](http://arxiv.org/pdf/2506.01611v1)**

> **作者:** Wangyou Zhang; Kohei Saijo; Samuele Cornell; Robin Scheibler; Chenda Li; Zhaoheng Ni; Anurag Kumar; Marvin Sach; Wei Wang; Yihui Fu; Shinji Watanabe; Tim Fingscheidt; Yanmin Qian
>
> **备注:** 5 pages, 4 figures, 1 table. Accepted by Interspeech 2025. Code available at https://github.com/urgent-challenge/urgent2024_analysis
>
> **摘要:** The URGENT 2024 Challenge aims to foster speech enhancement (SE) techniques with great universality, robustness, and generalizability, featuring a broader task definition, large-scale multi-domain data, and comprehensive evaluation metrics. Nourished by the challenge outcomes, this paper presents an in-depth analysis of two key, yet understudied, issues in SE system development: data cleaning and evaluation metrics. We highlight several overlooked problems in traditional SE pipelines: (1) mismatches between declared and effective audio bandwidths, along with label noise even in various "high-quality" speech corpora; (2) lack of both effective SE systems to conquer the hardest conditions (e.g., speech overlap, strong noise / reverberation) and reliable measure of speech sample difficulty; (3) importance of combining multifaceted metrics for a comprehensive evaluation correlating well with human judgment. We hope that this endeavor can inspire improved SE pipeline designs in the future.
>
---
#### [new 060] On-device Streaming Discrete Speech Units
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 论文研究在设备端实时语音处理中使用离散语音单元（DSU），旨在解决传统方法依赖全语音输入和高计算成本的问题。通过缩短注意力窗口并减小模型规模，显著降低计算量，同时保持较低的字符错误率，推动DSU在资源受限环境中的应用。**

- **链接: [http://arxiv.org/pdf/2506.01845v1](http://arxiv.org/pdf/2506.01845v1)**

> **作者:** Kwanghee Choi; Masao Someki; Emma Strubell; Shinji Watanabe
>
> **备注:** Accepted to Interspeech 2025, source code at https://github.com/Masao-Someki/StreamingDSU
>
> **摘要:** Discrete speech units (DSUs) are derived from clustering the features of self-supervised speech models (S3Ms). DSUs offer significant advantages for on-device streaming speech applications due to their rich phonetic information, high transmission efficiency, and seamless integration with large language models. However, conventional DSU-based approaches are impractical as they require full-length speech input and computationally expensive S3Ms. In this work, we reduce both the attention window and the model size while preserving the effectiveness of DSUs. Our results demonstrate that we can reduce floating-point operations (FLOPs) by 50% with only a relative increase of 6.5% in character error rate (CER) on the ML-SUPERB 1h dataset. These findings highlight the potential of DSUs for real-time speech processing in resource-constrained environments.
>
---
#### [new 061] From Words to Waves: Analyzing Concept Formation in Speech and Text-Based Foundation Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文研究语言和语音基础模型中的概念形成，分析单模态与多模态训练下语义结构的差异。任务是理解不同模态对模型抽象能力的影响。方法为使用潜在概念分析技术，揭示神经网络中的语义表示。**

- **链接: [http://arxiv.org/pdf/2506.01133v1](http://arxiv.org/pdf/2506.01133v1)**

> **作者:** Asım Ersoy; Basel Mousi; Shammur Chowdhury; Firoj Alam; Fahim Dalvi; Nadir Durrani
>
> **备注:** Accepted Interspeech 2025
>
> **摘要:** The emergence of large language models (LLMs) has demonstrated that systems trained solely on text can acquire extensive world knowledge, develop reasoning capabilities, and internalize abstract semantic concepts--showcasing properties that can be associated with general intelligence. This raises an intriguing question: Do such concepts emerge in models trained on other modalities, such as speech? Furthermore, when models are trained jointly on multiple modalities: Do they develop a richer, more structured semantic understanding? To explore this, we analyze the conceptual structures learned by speech and textual models both individually and jointly. We employ Latent Concept Analysis, an unsupervised method for uncovering and interpreting latent representations in neural networks, to examine how semantic abstractions form across modalities. For reproducibility we made scripts and other resources available to the community.
>
---
#### [new 062] Speech Unlearning
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音任务中的机器遗忘研究，旨在解决如何高效移除特定数据对已训练语音模型的影响，而无需完全重新训练。论文提出了两种语音遗忘任务：样本遗忘（删除个别语音数据）和类别遗忘（删除整个说话人数据），并通过实验验证其挑战性，探讨未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.00848v1](http://arxiv.org/pdf/2506.00848v1)**

> **作者:** Jiali Cheng; Hadi Amiri
>
> **备注:** Interspeech 2025
>
> **摘要:** We introduce machine unlearning for speech tasks, a novel and underexplored research problem that aims to efficiently and effectively remove the influence of specific data from trained speech models without full retraining. This has important applications in privacy preservation, removal of outdated or noisy data, and bias mitigation. While machine unlearning has been studied in computer vision and natural language processing, its application to speech is largely unexplored due to the high-dimensional, sequential, and speaker-dependent nature of speech data. We define two fundamental speech unlearning tasks: sample unlearning, which removes individual data points (e.g., a voice recording), and class unlearning, which removes an entire category (e.g., all data from a speaker), while preserving performance on the remaining data. Experiments on keyword spotting and speaker identification demonstrate that unlearning speech data is significantly more challenging than unlearning image or text data. We conclude with key future directions in this area, including structured training, robust evaluation, feature-level unlearning, broader applications, scalable methods, and adversarial robustness.
>
---
#### [new 063] Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音源追踪任务，旨在识别合成语音的生成系统。作者提出使用基于韵律特征的预训练语音模型（SPTM）进行源追踪，并融合多种SPTM表示以提升性能。他们提出了TRIO框架，结合TRILLsson和x-vector模型，在STSGS任务上取得了最优效果。**

- **链接: [http://arxiv.org/pdf/2506.01157v1](http://arxiv.org/pdf/2506.01157v1)**

> **作者:** Girish; Mohd Mujtaba Akhtar; Orchid Chetia Phukan; Drishti Singh; Swarup Ranjan Behera; Pailla Balakrishna Reddy; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to EUSIPCO 2025
>
> **摘要:** In this work, we focus on source tracing of synthetic speech generation systems (STSGS). Each source embeds distinctive paralinguistic features--such as pitch, tone, rhythm, and intonation--into their synthesized speech, reflecting the underlying design of the generation model. While previous research has explored representations from speech pre-trained models (SPTMs), the use of representations from SPTM pre-trained for paralinguistic speech processing, which excel in paralinguistic tasks like synthetic speech detection, speech emotion recognition has not been investigated for STSGS. We hypothesize that representations from paralinguistic SPTM will be more effective due to its ability to capture source-specific paralinguistic cues attributing to its paralinguistic pre-training. Our comparative study of representations from various SOTA SPTMs, including paralinguistic, monolingual, multilingual, and speaker recognition, validates this hypothesis. Furthermore, we explore fusion of representations and propose TRIO, a novel framework that fuses SPTMs using a gated mechanism for adaptive weighting, followed by canonical correlation loss for inter-representation alignment and self-attention for feature refinement. By fusing TRILLsson (Paralinguistic SPTM) and x-vector (Speaker recognition SPTM), TRIO outperforms individual SPTMs, baseline fusion methods, and sets new SOTA for STSGS in comparison to previous works.
>
---
#### [new 064] Continual Speech Learning with Fused Speech Features
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决传统静态模型难以适应动态多样语音数据的问题。工作包括提出连续语音学习框架，采用编码器-解码器结构及可学习门控融合层，提升多任务准确率且无需完全重训练。**

- **链接: [http://arxiv.org/pdf/2506.01496v1](http://arxiv.org/pdf/2506.01496v1)**

> **作者:** Guitao Wang; Jinming Zhao; Hao Yang; Guilin Qi; Tongtong Wu; Gholamreza Haffari
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** Rapid growth in speech data demands adaptive models, as traditional static methods fail to keep pace with dynamic and diverse speech information. We introduce continuous speech learning, a new set-up targeting at bridging the adaptation gap in current speech models. We use the encoder-decoder Whisper model to standardize speech tasks into a generative format. We integrate a learnable gated-fusion layer on the top of the encoder to dynamically select task-specific features for downstream tasks. Our approach improves accuracy significantly over traditional methods in six speech processing tasks, demonstrating gains in adapting to new speech tasks without full retraining.
>
---
#### [new 065] Quality Assessment of Noisy and Enhanced Speech with Limited Data: UWB-NTIS System for VoiceMOS 2024 and Beyond
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音质量评估任务，旨在解决在数据有限的情况下对带噪和增强语音的ITU-T P.835质量指标（SIG、BAK、OVRL）进行自动评估的问题。作者提出了基于wav2vec 2.0的UWB-NTIS系统，采用两阶段微调方法，在VoiceMOS 2024挑战赛中取得了优异成绩，并在CHiME 7 - UDASE数据集上进行了验证。**

- **链接: [http://arxiv.org/pdf/2506.00506v1](http://arxiv.org/pdf/2506.00506v1)**

> **作者:** Marie Kunešová
>
> **备注:** This is a preliminary write-up of our initial work, posted as an early version preprint for cross-referencing purposes. We intend to further extend this research and submit it for publication at a conference, at which point this preprint will be updated with the full text
>
> **摘要:** In this preprint, we present the UWB-NTIS-TTS team's submission to Track 3 of the VoiceMOS 2024 Challenge, the goal of which was to automatically assess the speech quality of noisy and de-noised speech in terms of the ITU-T P.835 metrics of "SIG", "BAK", and "OVRL". Our proposed system, based on wav2vec 2.0, placed among the top systems in the challenge, achieving the best prediction of the BAK scores (background noise intrusiveness), the second-best prediction of the OVRL score (overall audio quality), and the third-best prediction of SIG (speech signal quality) out of the five participating systems. We describe our approach, such as the two-stage fine-tuning process we used to contend with the challenge's very limiting restrictions on allowable training data, and present the results achieved both on the VoiceMOS 2024 Challenge data and on the recently released CHiME 7 - UDASE dataset.
>
---
#### [new 066] AbsoluteNet: A Deep Learning Neural Network to Classify Cerebral Hemodynamic Responses of Auditory Processing
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于脑机接口任务，旨在解决通过fNIRS解码听觉处理相关的血流动力学响应问题。作者提出了AbsoluteNet深度学习模型，结合时空卷积和自定义激活函数，实现了对听觉刺激反应的高效分类，取得了优于现有模型的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.00039v1](http://arxiv.org/pdf/2506.00039v1)**

> **作者:** Behtom Adeli; John Mclinden; Pankaj Pandey; Ming Shao; Yalda Shahriari
>
> **摘要:** In recent years, deep learning (DL) approaches have demonstrated promising results in decoding hemodynamic responses captured by functional near-infrared spectroscopy (fNIRS), particularly in the context of brain-computer interface (BCI) applications. This work introduces AbsoluteNet, a novel deep learning architecture designed to classify auditory event-related responses recorded using fNIRS. The proposed network is built upon principles of spatio-temporal convolution and customized activation functions. Our model was compared against several models, namely fNIRSNET, MDNN, DeepConvNet, and ShallowConvNet. The results showed that AbsoluteNet outperforms existing models, reaching 87.0% accuracy, 84.8% sensitivity, and 89.2% specificity in binary classification, surpassing fNIRSNET, the second-best model, by 3.8% in accuracy. These findings underscore the effectiveness of our proposed deep learning model in decoding hemodynamic responses related to auditory processing and highlight the importance of spatio-temporal feature aggregation and customized activation functions to better fit fNIRS dynamics.
>
---
#### [new 067] OWSM v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音模型训练任务，旨在解决训练数据不足和质量低的问题。通过整合并清洗大规模网络数据集YODAS，构建了高质量多语言语音数据集，进而训练出性能优于以往模型的OWSM v4系列模型。**

- **链接: [http://arxiv.org/pdf/2506.00338v1](http://arxiv.org/pdf/2506.00338v1)**

> **作者:** Yifan Peng; Shakeel Muhammad; Yui Sudo; William Chen; Jinchuan Tian; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** The Open Whisper-style Speech Models (OWSM) project has developed a series of fully open speech foundation models using academic-scale resources, but their training data remains insufficient. This work enhances OWSM by integrating YODAS, a large-scale web-crawled dataset with a Creative Commons license. However, incorporating YODAS is nontrivial due to its wild nature, which introduces challenges such as incorrect language labels and audio-text misalignments. To address this, we develop a scalable data-cleaning pipeline using public toolkits, yielding a dataset with 166,000 hours of speech across 75 languages. Our new series of OWSM v4 models, trained on this curated dataset alongside existing OWSM data, significantly outperform previous versions on multilingual benchmarks. Our models even match or surpass frontier industrial models like Whisper and MMS in multiple scenarios. We will publicly release the cleaned YODAS data, pre-trained models, and all associated scripts via the ESPnet toolkit.
>
---
#### [new 068] Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决构音障碍语音识别效果差的问题。通过提出一种基于音节的节奏建模方法，改进RnV转换框架，实现从障碍语音到健康语音的无监督转换，并验证其对ASR性能的提升效果。**

- **链接: [http://arxiv.org/pdf/2506.01618v1](http://arxiv.org/pdf/2506.01618v1)**

> **作者:** Karl El Hajal; Enno Hermann; Sevada Hovsepyan; Mathew Magimai. -Doss
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) systems struggle with dysarthric speech due to high inter-speaker variability and slow speaking rates. To address this, we explore dysarthric-to-healthy speech conversion for improved ASR performance. Our approach extends the Rhythm and Voice (RnV) conversion framework by introducing a syllable-based rhythm modeling method suited for dysarthric speech. We assess its impact on ASR by training LF-MMI models and fine-tuning Whisper on converted speech. Experiments on the Torgo corpus reveal that LF-MMI achieves significant word error rate reductions, especially for more severe cases of dysarthria, while fine-tuning Whisper on converted data has minimal effect on its performance. These results highlight the potential of unsupervised rhythm and voice conversion for dysarthric ASR. Code available at: https://github.com/idiap/RnV
>
---
## 更新

#### [replaced 001] Exploring the Effect of Segmentation and Vocabulary Size on Speech Tokenization for Speech Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17446v2](http://arxiv.org/pdf/2505.17446v2)**

> **作者:** Shunsuke Kando; Yusuke Miyao; Shinnosuke Takamichi
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** The purpose of speech tokenization is to transform a speech signal into a sequence of discrete representations, serving as the foundation for speech language models (SLMs). While speech tokenization has many options, their effect on the performance of SLMs remains unclear. This paper investigates two key aspects of speech tokenization: the segmentation width and the cluster size of discrete units. First, we segment speech signals into fixed/variable widths and pooled representations. We then train K-means models in multiple cluster sizes. Through the evaluation on zero-shot spoken language understanding benchmarks, we find the positive effect of moderately coarse segmentation and bigger cluster size. Notably, among the best-performing models, the most efficient one achieves a 50% reduction in training data and a 70% decrease in training runtime. Our analysis highlights the importance of combining multiple tokens to enhance fine-grained spoken language understanding.
>
---
#### [replaced 002] MSDA: Combining Pseudo-labeling and Self-Supervision for Unsupervised Domain Adaptation in ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24656v2](http://arxiv.org/pdf/2505.24656v2)**

> **作者:** Dimitrios Damianos; Georgios Paraskevopoulos; Alexandros Potamianos
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** In this work, we investigate the Meta PL unsupervised domain adaptation framework for Automatic Speech Recognition (ASR). We introduce a Multi-Stage Domain Adaptation pipeline (MSDA), a sample-efficient, two-stage adaptation approach that integrates self-supervised learning with semi-supervised techniques. MSDA is designed to enhance the robustness and generalization of ASR models, making them more adaptable to diverse conditions. It is particularly effective for low-resource languages like Greek and in weakly supervised scenarios where labeled data is scarce or noisy. Through extensive experiments, we demonstrate that Meta PL can be applied effectively to ASR tasks, achieving state-of-the-art results, significantly outperforming state-of-the-art methods, and providing more robust solutions for unsupervised domain adaptation in ASR. Our ablations highlight the necessity of utilizing a cascading approach when combining self-supervision with self-training.
>
---
#### [replaced 003] Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.13772v3](http://arxiv.org/pdf/2501.13772v3)**

> **作者:** Hao Cheng; Erjia Xiao; Jing Shao; Yichi Wang; Le Yang; Chao Shen; Philip Torr; Jindong Gu; Renjing Xu
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant security risks, as models can be exploited to generate harmful or inappropriate content through jailbreak attack. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific Jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce \textbf{Jailbreak-AudioBench}, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also various editing techniques for injecting audio hidden semantics. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs and establish the most comprehensive Jailbreak benchmark to date for audio modality. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.
>
---
#### [replaced 004] Codec-Based Deepfake Source Tracing via Neural Audio Codec Taxonomy
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12994v2](http://arxiv.org/pdf/2505.12994v2)**

> **作者:** Xuanjun Chen; I-Ming Lin; Lin Zhang; Jiawei Du; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Recent advances in neural audio codec-based speech generation (CoSG) models have produced remarkably realistic audio deepfakes. We refer to deepfake speech generated by CoSG systems as codec-based deepfake, or CodecFake. Although existing anti-spoofing research on CodecFake predominantly focuses on verifying the authenticity of audio samples, almost no attention was given to tracing the CoSG used in generating these deepfakes. In CodecFake generation, processes such as speech-to-unit encoding, discrete unit modeling, and unit-to-speech decoding are fundamentally based on neural audio codecs. Motivated by this, we introduce source tracing for CodecFake via neural audio codec taxonomy, which dissects neural audio codecs to trace CoSG. Our experimental results on the CodecFake+ dataset provide promising initial evidence for the feasibility of CodecFake source tracing while also highlighting several challenges that warrant further investigation.
>
---
#### [replaced 005] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.16344v4](http://arxiv.org/pdf/2501.16344v4)**

> **作者:** Rajath Rao; Adithya Ganesan; Oscar Kjell; Jonah Luby; Akshay Raghavan; Scott Feltman; Whitney Ringwald; Ryan L. Boyd; Benjamin Luft; Camilo Ruggero; Neville Ryant; Roman Kotov; H. Andrew Schwartz
>
> **备注:** 16 pages, 8 figures, ACL 2025
>
> **摘要:** Current speech encoding pipelines often rely on an additional text-based LM to get robust representations of human communication, even though SotA speech-to-text models often have a LM within. This work proposes an approach to improve the LM within an audio model such that the subsequent text-LM is unnecessary. We introduce WhiSPA (Whisper with Semantic and Psychological Alignment), which leverages a novel audio training objective: contrastive loss with a language model embedding as a teacher. Using over 500k speech segments from mental health audio interviews, we evaluate the utility of aligning Whisper's latent space with semantic representations from a text autoencoder (SBERT) and lexically derived embeddings of basic psychological dimensions: emotion and personality. Over self-supervised affective tasks and downstream psychological tasks, WhiSPA surpasses current speech encoders, achieving an average error reduction of 73.4% and 83.8%, respectively. WhiSPA demonstrates that it is not always necessary to run a subsequent text LM on speech-to-text output in order to get a rich psychological representation of human communication.
>
---
#### [replaced 006] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17543v2](http://arxiv.org/pdf/2505.17543v2)**

> **作者:** Kaixing Yang; Xulong Tang; Ziqiao Peng; Yuxuan Hu; Jun He; Hongyan Liu
>
> **备注:** arXiv admin note: text overlap with arXiv:2505.14222
>
> **摘要:** Music-driven 3D dance generation has attracted increasing attention in recent years, with promising applications in choreography, virtual reality, and creative content creation. Previous research has generated promising realistic dance movement from audio signals. However, traditional methods underutilize genre conditioning, often treating it as auxiliary modifiers rather than core semantic drivers. This oversight compromises music-motion synchronization and disrupts dance genre continuity, particularly during complex rhythmic transitions, thereby leading to visually unsatisfactory effects. To address the challenge, we propose MEGADance, a novel architecture for music-driven 3D dance generation. By decoupling choreographic consistency into dance generality and genre specificity, MEGADance demonstrates significant dance quality and strong genre controllability. It consists of two stages: (1) High-Fidelity Dance Quantization Stage (HFDQ), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) and reconstructs them with kinematic-dynamic constraints, and (2) Genre-Aware Dance Generation Stage (GADG), which maps music into the latent representation by synergistic utilization of Mixture-of-Experts (MoE) mechanism with Mamba-Transformer hybrid backbone. Extensive experiments on the FineDance and AIST++ dataset demonstrate the state-of-the-art performance of MEGADance both qualitatively and quantitatively. Code will be released upon acceptance.
>
---
#### [replaced 007] A Comparative Study on Positional Encoding for Time-frequency Domain Dual-path Transformer-based Source Separation Models
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.19605v2](http://arxiv.org/pdf/2504.19605v2)**

> **作者:** Kohei Saijo; Tetsuji Ogawa
>
> **备注:** 5 pages, 3 tables, 2 figures. Accepted to EUSIPCO2025
>
> **摘要:** In this study, we investigate the impact of positional encoding (PE) on source separation performance and the generalization ability to long sequences (length extrapolation) in Transformer-based time-frequency (TF) domain dual-path models. The length extrapolation capability in TF-domain dual-path models is a crucial factor, as it affects not only their performance on long-duration inputs but also their generalizability to signals with unseen sampling rates. While PE is known to significantly impact length extrapolation, there has been limited research that explores the choice of PEs for TF-domain dual-path models from this perspective. To address this gap, we compare various PE methods using a recent state-of-the-art model, TF-Locoformer, as the base architecture. Our analysis yields the following key findings: (i) When handling sequences that are the same length as or shorter than those seen during training, models with PEs achieve better performance. (ii) However, models without PE exhibit superior length extrapolation. This trend is particularly pronounced when the model contains convolutional layers.
>
---
#### [replaced 008] Egocentric Speaker Classification in Child-Adult Dyadic Interactions: From Sensing to Computational Modeling
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.09340v2](http://arxiv.org/pdf/2409.09340v2)**

> **作者:** Tiantian Feng; Anfeng Xu; Xuan Shi; Somer Bishop; Shrikanth Narayanan
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** Autism spectrum disorder (ASD) is a neurodevelopmental condition characterized by challenges in social communication, repetitive behavior, and sensory processing. One important research area in ASD is evaluating children's behavioral changes over time during treatment. The standard protocol with this objective is BOSCC, which involves dyadic interactions between a child and clinicians performing a pre-defined set of activities. A fundamental aspect of understanding children's behavior in these interactions is automatic speech understanding, particularly identifying who speaks and when. Conventional approaches in this area heavily rely on speech samples recorded from a spectator perspective, and there is limited research on egocentric speech modeling. In this study, we design an experiment to perform speech sampling in BOSCC interviews from an egocentric perspective using wearable sensors and explore pre-training Ego4D speech samples to enhance child-adult speaker classification in dyadic interactions. Our findings highlight the potential of egocentric speech collection and pre-training to improve speaker classification accuracy.
>
---
#### [replaced 009] Self-supervised Reflective Learning through Self-distillation and Online Clustering for Speaker Representation Learning
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2401.01473v3](http://arxiv.org/pdf/2401.01473v3)**

> **作者:** Danwei Cai; Zexin Cai; Ze Li; Ming Li
>
> **摘要:** Speaker representation learning is crucial for voice recognition systems, with recent advances in self-supervised approaches reducing dependency on labeled data. Current two-stage iterative frameworks, while effective, suffer from significant computational overhead due to repeated rounds of clustering and training. They also struggle with noisy pseudo labels that can impair model learning. This paper introduces self-supervised reflective learning (SSRL), an improved framework that addresses these limitations by enabling continuous refinement of pseudo labels during training. Through a teacher-student architecture and online clustering mechanism, SSRL eliminates the need for iterative training rounds. To handle label noise, we incorporate noisy label modeling and pseudo label queues that maintain temporal consistency. Experiments on VoxCeleb show SSRL's superiority over current two-stage iterative approaches, surpassing the performance of a 5-round method in just a single training round. Ablation studies validate the contributions of key components like noisy label modeling and pseudo label queues. Moreover, consistent improvements in pseudo labeling and the convergence of cluster counts demonstrate SSRL's effectiveness in deciphering unlabeled data. This work marks an important advancement in efficient and accurate self-supervised speaker representation learning through the novel reflective learning paradigm.
>
---
#### [replaced 010] Towards Early Prediction of Self-Supervised Speech Model Performance
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.05966v2](http://arxiv.org/pdf/2501.05966v2)**

> **作者:** Ryan Whetten; Lucas Maison; Titouan Parcollet; Marco Dinarelli; Yannick Estève
>
> **摘要:** In Self-Supervised Learning (SSL), pre-training and evaluation are resource intensive. In the speech domain, current indicators of the quality of SSL models during pre-training, such as the loss, do not correlate well with downstream performance. Consequently, it is often difficult to gauge the final downstream performance in a cost efficient manner during pre-training. In this work, we propose unsupervised efficient methods that give insights into the quality of the pre-training of SSL speech models, namely, measuring the cluster quality and rank of the embeddings of the SSL model. Results show that measures of cluster quality and rank correlate better with downstream performance than the pre-training loss with only one hour of unlabeled audio, reducing the need for GPU hours and labeled data in SSL model evaluation.
>
---
#### [replaced 011] Replay Attacks Against Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14862v2](http://arxiv.org/pdf/2505.14862v2)**

> **作者:** Nicolas Müller; Piotr Kawa; Wei-Herng Choong; Adriana Stan; Aditya Tirumala Bukkapatnam; Karla Pizzi; Alexander Wagner; Philip Sperl
>
> **摘要:** We show how replay attacks undermine audio deepfake detection: By playing and re-recording deepfake audio through various speakers and microphones, we make spoofed samples appear authentic to the detection model. To study this phenomenon in more detail, we introduce ReplayDF, a dataset of recordings derived from M-AILABS and MLAAD, featuring 109 speaker-microphone combinations across six languages and four TTS models. It includes diverse acoustic conditions, some highly challenging for detection. Our analysis of six open-source detection models across five datasets reveals significant vulnerability, with the top-performing W2V2-AASIST model's Equal Error Rate (EER) surging from 4.7% to 18.2%. Even with adaptive Room Impulse Response (RIR) retraining, performance remains compromised with an 11.0% EER. We release ReplayDF for non-commercial research use.
>
---
#### [replaced 012] SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.17645v2](http://arxiv.org/pdf/2402.17645v2)**

> **作者:** Shuangrui Ding; Zihan Liu; Xiaoyi Dong; Pan Zhang; Rui Qian; Junhao Huang; Conghui He; Dahua Lin; Jiaqi Wang
>
> **备注:** ACL 2025 main. project page: https://pjlab-songcomposer.github.io/ code: https://github.com/pjlab-songcomposer/songcomposer
>
> **摘要:** Creating lyrics and melodies for the vocal track in a symbolic format, known as song composition, demands expert musical knowledge of melody, an advanced understanding of lyrics, and precise alignment between them. Despite achievements in sub-tasks such as lyric generation, lyric-to-melody, and melody-to-lyric, etc, a unified model for song composition has not yet been achieved. In this paper, we introduce SongComposer, a pioneering step towards a unified song composition model that can readily create symbolic lyrics and melodies following instructions. SongComposer is a music-specialized large language model (LLM) that, for the first time, integrates the capability of simultaneously composing lyrics and melodies into LLMs by leveraging three key innovations: 1) a flexible tuple format for word-level alignment of lyrics and melodies, 2) an extended tokenizer vocabulary for song notes, with scalar initialization based on musical knowledge to capture rhythm, and 3) a multi-stage pipeline that captures musical structure, starting with motif-level melody patterns and progressing to phrase-level structure for improved coherence. Extensive experiments demonstrate that SongComposer outperforms advanced LLMs, including GPT-4, in tasks such as lyric-to-melody generation, melody-to-lyric generation, song continuation, and text-to-song creation. Moreover, we will release SongCompose, a large-scale dataset for training, containing paired lyrics and melodies in Chinese and English.
>
---
#### [replaced 013] A General-Purpose Neuromorphic Sensor based on Spiketrum Algorithm: Hardware Details and Real-life Applications
- **分类: eess.SP; cs.SY; eess.AS; eess.SY**

- **链接: [http://arxiv.org/pdf/2501.18799v2](http://arxiv.org/pdf/2501.18799v2)**

> **作者:** MHD Anas Alsakkal; Runze Wang; Piotr Dudek; Jayawan Wijekoon
>
> **备注:** Currently under review with IEEE TCAS
>
> **摘要:** Spiking Neural Networks (SNNs) offer a biologically inspired computational paradigm, enabling energy-efficient data processing through spike-based information transmission. Despite notable advancements in hardware for SNNs, spike encoding has largely remained software-dependent, limiting efficiency. This paper addresses the need for adaptable and resource-efficient spike encoding hardware by presenting an area-optimized hardware implementation of the Spiketrum algorithm, which encodes time-varying analogue signals into spatiotemporal spike patterns. Unlike earlier performance-optimized designs, which prioritize speed, our approach focuses on reducing hardware footprint, achieving a 52% reduction in Block RAMs (BRAMs), 31% fewer Digital Signal Processing (DSP) slices, and a 6% decrease in Look-Up Tables (LUTs). The proposed implementation has been verified on an FPGA and successfully integrated into an IC using TSMC180 technology. Experimental results demonstrate the system's effectiveness in real-world applications, including sound and ECG classification. This work highlights the trade-offs between performance and resource efficiency, offering a flexible, scalable solution for neuromorphic systems in power-sensitive applications like cochlear implants and neural devices.
>
---
#### [replaced 014] Efficient Speech Translation through Model Compression and Knowledge Distillation
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.20237v2](http://arxiv.org/pdf/2505.20237v2)**

> **作者:** Yasmin Moslem
>
> **备注:** IWSLT 2025
>
> **摘要:** Efficient deployment of large audio-language models for speech translation remains challenging due to their significant computational requirements. In this paper, we address this challenge through our system submissions to the "Model Compression" track at the International Conference on Spoken Language Translation (IWSLT 2025). We experiment with a combination of approaches including iterative layer pruning based on layer importance evaluation, low-rank adaptation with 4-bit quantization (QLoRA), and knowledge distillation. In our experiments, we use Qwen2-Audio-7B-Instruct for speech translation into German and Chinese. Our pruned (student) models achieve up to a 50% reduction in both model parameters and storage footprint, while retaining 97-100% of the translation quality of the in-domain (teacher) models.
>
---
#### [replaced 015] SpeechT: Findings of the First Mentorship in Speech Translation
- **分类: cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.12050v3](http://arxiv.org/pdf/2502.12050v3)**

> **作者:** Yasmin Moslem; Juan Julián Cea Morán; Mariano Gonzalez-Gomez; Muhammad Hazim Al Farouq; Farah Abdou; Satarupa Deb
>
> **备注:** MT Summit 2025
>
> **摘要:** This work presents the details and findings of the first mentorship in speech translation (SpeechT), which took place in December 2024 and January 2025. To fulfil the mentorship requirements, the participants engaged in key activities, including data preparation, modelling, and advanced research. The participants explored data augmentation techniques and compared end-to-end and cascaded speech translation systems. The projects covered various languages other than English, including Arabic, Bengali, Galician, Indonesian, Japanese, and Spanish.
>
---
#### [replaced 016] SpeechVerifier: Robust Acoustic Fingerprint against Tampering Attacks via Watermarking
- **分类: cs.CR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.23821v2](http://arxiv.org/pdf/2505.23821v2)**

> **作者:** Lingfeng Yao; Chenpei Huang; Shengyao Wang; Junpei Xue; Hanqing Guo; Jiang Liu; Xun Chen; Miao Pan
>
> **摘要:** With the surge of social media, maliciously tampered public speeches, especially those from influential figures, have seriously affected social stability and public trust. Existing speech tampering detection methods remain insufficient: they either rely on external reference data or fail to be both sensitive to attacks and robust to benign operations, such as compression and resampling. To tackle these challenges, we introduce SpeechVerifer to proactively verify speech integrity using only the published speech itself, i.e., without requiring any external references. Inspired by audio fingerprinting and watermarking, SpeechVerifier can (i) effectively detect tampering attacks, (ii) be robust to benign operations and (iii) verify the integrity only based on published speeches. Briefly, SpeechVerifier utilizes multiscale feature extraction to capture speech features across different temporal resolutions. Then, it employs contrastive learning to generate fingerprints that can detect modifications at varying granularities. These fingerprints are designed to be robust to benign operations, but exhibit significant changes when malicious tampering occurs. To enable speech verification in a self-contained manner, the generated fingerprints are then embedded into the speech signal by segment-wise watermarking. Without external references, SpeechVerifier can retrieve the fingerprint from the published audio and check it with the embedded watermark to verify the integrity of the speech. Extensive experimental results demonstrate that the proposed SpeechVerifier is effective in detecting tampering attacks and robust to benign operations.
>
---
#### [replaced 017] SoloSpeech: Enhancing Intelligibility and Quality in Target Speech Extraction through a Cascaded Generative Pipeline
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.19314v2](http://arxiv.org/pdf/2505.19314v2)**

> **作者:** Helin Wang; Jiarui Hai; Dongchao Yang; Chen Chen; Kai Li; Junyi Peng; Thomas Thebaud; Laureano Moro Velazquez; Jesus Villalba; Najim Dehak
>
> **摘要:** Target Speech Extraction (TSE) aims to isolate a target speaker's voice from a mixture of multiple speakers by leveraging speaker-specific cues, typically provided as auxiliary audio (a.k.a. cue audio). Although recent advancements in TSE have primarily employed discriminative models that offer high perceptual quality, these models often introduce unwanted artifacts, reduce naturalness, and are sensitive to discrepancies between training and testing environments. On the other hand, generative models for TSE lag in perceptual quality and intelligibility. To address these challenges, we present SoloSpeech, a novel cascaded generative pipeline that integrates compression, extraction, reconstruction, and correction processes. SoloSpeech features a speaker-embedding-free target extractor that utilizes conditional information from the cue audio's latent space, aligning it with the mixture audio's latent space to prevent mismatches. Evaluated on the widely-used Libri2Mix dataset, SoloSpeech achieves the new state-of-the-art intelligibility and quality in target speech extraction and speech separation tasks while demonstrating exceptional generalization on out-of-domain data and real-world scenarios.
>
---
#### [replaced 018] MADUV: The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalization Challenge
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.04292v3](http://arxiv.org/pdf/2501.04292v3)**

> **作者:** Zijiang Yang; Meishu Song; Xin Jing; Haojie Zhang; Kun Qian; Bin Hu; Kota Tamada; Toru Takumi; Björn W. Schuller; Yoshiharu Yamamoto
>
> **备注:** 5 pages, 1 figure and 2 tables. Submitted to INTERSPEECH 2025. For MADUV Challenge 2025
>
> **摘要:** The Mice Autism Detection via Ultrasound Vocalization (MADUV) Challenge introduces the first INTERSPEECH challenge focused on detecting autism spectrum disorder (ASD) in mice through their vocalizations. Participants are tasked with developing models to automatically classify mice as either wild-type or ASD models based on recordings with a high sampling rate. Our baseline system employs a simple CNN-based classification using three different spectrogram features. Results demonstrate the feasibility of automated ASD detection, with the considered audible-range features achieving the best performance (UAR of 0.600 for segment-level and 0.625 for subject-level classification). This challenge bridges speech technology and biomedical research, offering opportunities to advance our understanding of ASD models through machine learning approaches. The findings suggest promising directions for vocalization analysis and highlight the potential value of audible and ultrasound vocalizations in ASD detection.
>
---
#### [replaced 019] Developing a Top-tier Framework in Naturalistic Conditions Challenge for Categorized Emotion Prediction: From Speech Foundation Models and Learning Objective to Data Augmentation and Engineering Choices
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.22133v2](http://arxiv.org/pdf/2505.22133v2)**

> **作者:** Tiantian Feng; Thanathai Lertpetchpun; Dani Byrd; Shrikanth Narayanan
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** Speech emotion recognition (SER), particularly for naturally expressed emotions, remains a challenging computational task. Key challenges include the inherent subjectivity in emotion annotation and the imbalanced distribution of emotion labels in datasets. This paper introduces the \texttt{SAILER} system developed for participation in the INTERSPEECH 2025 Emotion Recognition Challenge (Task 1). The challenge dataset, which contains natural emotional speech from podcasts, serves as a valuable resource for studying imbalanced and subjective emotion annotations. Our system is designed to be simple, reproducible, and effective, highlighting critical choices in modeling, learning objectives, data augmentation, and engineering choices. Results show that even a single system (without ensembling) can outperform more than 95\% of the submissions, with a Macro-F1 score exceeding 0.4. Moreover, an ensemble of three systems further improves performance, achieving a competitively ranked score (top-3 performing team). Our model is at: https://github.com/tiantiaf0627/vox-profile-release.
>
---
#### [replaced 020] FAMA: The First Large-Scale Open-Science Speech Foundation Model for English and Italian
- **分类: cs.CL; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.22759v2](http://arxiv.org/pdf/2505.22759v2)**

> **作者:** Sara Papi; Marco Gaido; Luisa Bentivogli; Alessio Brutti; Mauro Cettolo; Roberto Gretter; Marco Matassoni; Mohamed Nabih; Matteo Negri
>
> **摘要:** The development of speech foundation models (SFMs) like Whisper and SeamlessM4T has significantly advanced the field of speech processing. However, their closed nature--with inaccessible training data and code--poses major reproducibility and fair evaluation challenges. While other domains have made substantial progress toward open science by developing fully transparent models trained on open-source (OS) code and data, similar efforts in speech remain limited. To fill this gap, we introduce FAMA, the first family of open science SFMs for English and Italian, trained on 150k+ hours of OS speech data. Moreover, we present a new dataset containing 16k hours of cleaned and pseudo-labeled speech for both languages. Results show that FAMA achieves competitive performance compared to existing SFMs while being up to 8 times faster. All artifacts, including code, datasets, and models, are released under OS-compliant licenses, promoting openness in speech technology research.
>
---
#### [replaced 021] Improving Speech Emotion Recognition Through Cross Modal Attention Alignment and Balanced Stacking Model
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.20007v2](http://arxiv.org/pdf/2505.20007v2)**

> **作者:** Lucas Ueda; João Lima; Leonardo Marques; Paula Costa
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Emotion plays a fundamental role in human interaction, and therefore systems capable of identifying emotions in speech are crucial in the context of human-computer interaction. Speech emotion recognition (SER) is a challenging problem, particularly in natural speech and when the available data is imbalanced across emotions. This paper presents our proposed system in the context of the 2025 Speech Emotion Recognition in Naturalistic Conditions Challenge. Our proposed architecture leverages cross-modality, utilizing cross-modal attention to fuse representations from different modalities. To address class imbalance, we employed two training designs: (i) weighted crossentropy loss (WCE); and (ii) WCE with an additional neutralexpressive soft margin loss and balancing. We trained a total of 12 multimodal models, which were ensembled using a balanced stacking model. Our proposed system achieves a MacroF1 score of 0.4094 and an accuracy of 0.4128 on 8-class speech emotion recognition.
>
---
#### [replaced 022] Differential privacy enables fair and accurate AI-based analysis of speech disorders while protecting patient data
- **分类: cs.LG; cs.AI; cs.CR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.19078v3](http://arxiv.org/pdf/2409.19078v3)**

> **作者:** Soroosh Tayebi Arasteh; Mahshad Lotfinia; Paula Andrea Perez-Toro; Tomas Arias-Vergara; Mahtab Ranji; Juan Rafael Orozco-Arroyave; Maria Schuster; Andreas Maier; Seung Hee Yang
>
> **摘要:** Speech pathology has impacts on communication abilities and quality of life. While deep learning-based models have shown potential in diagnosing these disorders, the use of sensitive data raises critical privacy concerns. Although differential privacy (DP) has been explored in the medical imaging domain, its application in pathological speech analysis remains largely unexplored despite the equally critical privacy concerns. To the best of our knowledge, this study is the first to investigate DP's impact on pathological speech data, focusing on the trade-offs between privacy, diagnostic accuracy, and fairness. Using a large, real-world dataset of 200 hours of recordings from 2,839 German-speaking participants, we observed a maximum accuracy reduction of 3.85% when training with DP with high privacy levels. To highlight real-world privacy risks, we demonstrated the vulnerability of non-private models to gradient inversion attacks, reconstructing identifiable speech samples and showcasing DP's effectiveness in mitigating these risks. To explore the potential generalizability across languages and disorders, we validated our approach on a dataset of Spanish-speaking Parkinson's disease patients, leveraging pretrained models from healthy English-speaking datasets, and demonstrated that careful pretraining on large-scale task-specific datasets can maintain favorable accuracy under DP constraints. A comprehensive fairness analysis revealed minimal gender bias at reasonable privacy levels but underscored the need for addressing age-related disparities. Our results establish that DP can balance privacy and utility in speech disorder detection, while highlighting unique challenges in privacy-fairness trade-offs for speech data. This provides a foundation for refining DP methodologies and improving fairness across diverse patient groups in real-world deployments.
>
---
#### [replaced 023] Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.09439v2](http://arxiv.org/pdf/2505.09439v2)**

> **作者:** Andrew Rouditchenko; Saurabhchand Bhati; Edson Araujo; Samuel Thomas; Hilde Kuehne; Rogerio Feris; James Glass
>
> **摘要:** We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
>
---
#### [replaced 024] Forensic deepfake audio detection using segmental speech features
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13847v2](http://arxiv.org/pdf/2505.13847v2)**

> **作者:** Tianle Yang; Chengzhe Sun; Siwei Lyu; Phil Rose
>
> **摘要:** This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison (FVC) are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection using methods that are distinct from those employed in traditional FVC, and offer a new perspective on leveraging segmental features for this purpose.
>
---
#### [replaced 025] Bemba Speech Translation: Exploring a Low-Resource African Language
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.02518v3](http://arxiv.org/pdf/2505.02518v3)**

> **作者:** Muhammad Hazim Al Farouq; Aman Kassahun Wassie; Yasmin Moslem
>
> **备注:** IWSLT 2025
>
> **摘要:** This paper describes our system submission to the International Conference on Spoken Language Translation (IWSLT 2025), low-resource languages track, namely for Bemba-to-English speech translation. We built cascaded speech translation systems based on Whisper and NLLB-200, and employed data augmentation techniques, such as back-translation. We investigate the effect of using synthetic data and discuss our experimental setup.
>
---
#### [replaced 026] ClearSphere: Multi-Earphone Synergy for Enhanced Conversational Clarity
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2505.21004v2](http://arxiv.org/pdf/2505.21004v2)**

> **作者:** Lixing He
>
> **摘要:** In crowded places such as conferences, background noise, overlapping voices, and lively interactions make it difficult to have clear conversations. This situation often worsens the phenomenon known as "cocktail party deafness." We present ClearSphere, the collaborative system that enhances speech at the conversation level with multi-earphones. Real-time conversation enhancement requires a holistic modeling of all the members in the conversation, and an effective way to extract the speech from the mixture. ClearSphere bridges the acoustic sensor system and state-of-the-art deep learning for target speech extraction by making two key contributions: 1) a conversation-driven network protocol, and 2) a robust target conversation extraction model. Our networking protocol enables mobile, infrastructure-free coordination among earphone devices. Our conversation extraction model can leverage the relay audio in a bandwidth-efficient way. ClearSphere is evaluated in both real-world experiments and simulations. Results show that our conversation network obtains more than 90\% accuracy in group formation, improves the speech quality by up to 8.8 dB over state-of-the-art baselines, and demonstrates real-time performance on a mobile device. In a user study with 20 participants, ClearSphere has a much higher score than baseline with good usability.
>
---
#### [replaced 027] ReelWave: Multi-Agentic Movie Sound Generation through Multimodal LLM Conversation
- **分类: cs.SD; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07217v3](http://arxiv.org/pdf/2503.07217v3)**

> **作者:** Zixuan Wang; Chi-Keung Tang; Yu-Wing Tai
>
> **备注:** Project page: https://vincent2311.github.io/ReelWave_demo
>
> **摘要:** Current audio generation conditioned by text or video focuses on aligning audio with text/video modalities. Despite excellent alignment results, these multimodal frameworks still cannot be directly applied to compelling movie storytelling involving multiple scenes, where "on-screen" sounds require temporally-aligned audio generation, while "off-screen" sounds contribute to appropriate environment sounds accompanied by background music when applicable. Inspired by professional movie production, this paper proposes a multi-agentic framework for audio generation supervised by an autonomous Sound Director agent, engaging multi-turn conversations with other agents for on-screen and off-screen sound generation through multimodal LLM. To address on-screen sound generation, after detecting any talking humans in videos, we capture semantically and temporally synchronized sound by training a prediction model that forecasts interpretable, time-varying audio control signals: loudness, pitch, and timbre, which are used by a Foley Artist agent to condition a cross-attention module in the sound generation. The Foley Artist works cooperatively with the Composer and Voice Actor agents, and together they autonomously generate off-screen sound to complement the overall production. Each agent takes on specific roles similar to those of a movie production team. To temporally ground audio language models, in ReelWave, text/video conditions are decomposed into atomic, specific sound generation instructions synchronized with visuals when applicable. Consequently, our framework can generate rich and relevant audio content conditioned on video clips extracted from movies.
>
---
#### [replaced 028] Training Articulatory Inversion Models for Inter-Speaker Consistency
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.20529v2](http://arxiv.org/pdf/2505.20529v2)**

> **作者:** Charles McGhee; Mark J. F. Gales; Kate M. Knill
>
> **摘要:** Acoustic-to-Articulatory Inversion (AAI) attempts to model the inverse mapping from speech to articulation. Exact articulatory prediction from speech alone may be impossible, as speakers can choose different forms of articulation seemingly without reference to their vocal tract structure. However, once a speaker has selected an articulatory form, their productions vary minimally. Recent works in AAI have proposed adapting Self-Supervised Learning (SSL) models to single-speaker datasets, claiming that these single-speaker models provide a universal articulatory template. In this paper, we investigate whether SSL-adapted models trained on single and multi-speaker data produce articulatory targets which are consistent across speaker identities for English and Russian. We do this through the use of a novel evaluation method which extracts articulatory targets using minimal pair sets. We also present a training method which can improve interspeaker consistency using only speech data.
>
---
#### [replaced 029] VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.19462v2](http://arxiv.org/pdf/2505.19462v2)**

> **作者:** Puyuan Peng; Shang-Wen Li; Abdelrahman Mohamed; David Harwath
>
> **摘要:** We present VoiceStar, the first zero-shot TTS model that achieves both output duration control and extrapolation. VoiceStar is an autoregressive encoder-decoder neural codec language model, that leverages a novel Progress-Monitoring Rotary Position Embedding (PM-RoPE) and is trained with Continuation-Prompt Mixed (CPM) training. PM-RoPE enables the model to better align text and speech tokens, indicates the target duration for the generated speech, and also allows the model to generate speech waveforms much longer in duration than those seen during. CPM training also helps to mitigate the training/inference mismatch, and significantly improves the quality of the generated speech in terms of speaker similarity and intelligibility. VoiceStar outperforms or is on par with current state-of-the-art models on short-form benchmarks such as Librispeech and Seed-TTS, and significantly outperforms these models on long-form/extrapolation benchmarks (20-50s) in terms of intelligibility and naturalness. Code and models: https://github.com/jasonppy/VoiceStar. Audio samples: https://jasonppy.github.io/VoiceStar_web
>
---
#### [replaced 030] AfriHuBERT: A self-supervised speech representation model for African languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.20201v2](http://arxiv.org/pdf/2409.20201v2)**

> **作者:** Jesujoba O. Alabi; Xuechen Liu; Dietrich Klakow; Junichi Yamagishi
>
> **备注:** Interspeech 2025
>
> **摘要:** In this work, we present AfriHuBERT, an extension of mHuBERT-147, a compact self-supervised learning (SSL) model pretrained on 147 languages. While mHuBERT-147 covered 16 African languages, we expand this to 1,226 through continued pretraining on 10K+ hours of speech data from diverse sources, benefiting an African population of over 600M. We evaluate AfriHuBERT on two key speech tasks, Spoken Language Identification (SLID) and Automatic Speech Recognition (ASR), using the FLEURS benchmark. Our results show a +3.6% F1 score improvement for SLID and a -2.1% average Word Error Rate (WER) reduction for ASR over mHuBERT-147, and demonstrates competitiveness with larger SSL models such as MMS and XEUS. Further analysis shows that ASR models trained on AfriHuBERT exhibit improved cross-corpus generalization and are competitive in extremely low-resource ASR scenarios.
>
---
