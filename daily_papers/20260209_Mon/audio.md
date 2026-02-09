# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos
- **分类: cs.SD**

- **简介: 该论文属于空间音频生成任务，旨在解决复杂场景下360视频动态空间音频生成问题。提出DynFOA框架，结合视觉与音频信息，生成高保真一阶Ambisonics。**

- **链接: [https://arxiv.org/pdf/2602.06846v1](https://arxiv.org/pdf/2602.06846v1)**

> **作者:** Ziyu Luo; Lin Chen; Qiang Qu; Xiaoming Chen; Yiran Shen
>
> **摘要:** Spatial audio is crucial for creating compelling immersive 360-degree video experiences. However, generating realistic spatial audio, such as first-order ambisonics (FOA), from 360-degree videos in complex acoustic scenes remains challenging. Existing methods often overlook the dynamic nature and acoustic complexity of 360-degree scenes, fail to fully account for dynamic sound sources, and neglect complex environmental effects such as occlusion, reflections, and reverberation, which are influenced by scene geometries and materials. We propose DynFOA, a framework based on dynamic acoustic perception and conditional diffusion, for generating high-fidelity FOA from 360-degree videos. DynFOA first performs visual processing via a video encoder, which detects and localizes multiple dynamic sound sources, estimates their depth and semantics, and reconstructs the scene geometry and materials using a 3D Gaussian Splatting. This reconstruction technique accurately models occlusion, reflections, and reverberation based on the geometries and materials of the reconstructed 3D scene and the listener's viewpoint. The audio encoder then captures the spatial motion and temporal 4D sound source trajectories to fine-tune the diffusion-based FOA generator. The fine-tuned FOA generator adjusts spatial cues in real time, ensuring consistent directional fidelity during listener head rotation and complex environmental changes. Extensive evaluations demonstrate that DynFOA consistently outperforms existing methods across metrics such as spatial accuracy, acoustic fidelity, and distribution matching, while also improving the user experience. Therefore, DynFOA provides a robust and scalable approach to rendering realistic dynamic spatial audio for VR and immersive media applications.
>
---
#### [new 002] Hierarchical Activity Recognition and Captioning from Long-Form Audio
- **分类: cs.SD**

- **简介: 该论文属于音频活动识别与描述任务，旨在解决长时音频中层次化活动理解的问题。工作包括构建多级标注数据集MultiAct，并提出统一的分层模型进行多粒度分析。**

- **链接: [https://arxiv.org/pdf/2602.06765v1](https://arxiv.org/pdf/2602.06765v1)**

> **作者:** Peng Zhang; Qingyu Luo; Philip J. B. Jackson; Wenwu Wang
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Complex activities in real-world audio unfold over extended durations and exhibit hierarchical structure, yet most prior work focuses on short clips and isolated events. To bridge this gap, we introduce MultiAct, a new dataset and benchmark for multi-level structured understanding of human activities from long-form audio. MultiAct comprises long-duration kitchen recordings annotated at three semantic levels (activities, sub-activities and events) and paired with fine-grained captions and high-level summaries. We further propose a unified hierarchical model that jointly performs classification, detection, sequence prediction and multi-resolution captioning. Experiments on MultiAct establish strong baselines and reveal key challenges in modelling hierarchical and compositional structure of long-form audio. A promising direction for future work is the exploration of methods better suited to capturing the complex, long-range relationships in long-form audio.
>
---
#### [new 003] From Hallucination to Articulation: Language Model-Driven Losses for Ultra Low-Bitrate Neural Speech Coding
- **分类: eess.AS**

- **简介: 该论文属于语音编码任务，解决低比特率下语音模型生成的伪词问题。通过引入语言模型驱动的损失函数，提升语义一致性与输出质量。**

- **链接: [https://arxiv.org/pdf/2602.06213v1](https://arxiv.org/pdf/2602.06213v1)**

> **作者:** Jayeon Yi; Minje Kim
>
> **备注:** To appear in ICASSP 2026. Demo wavs, code, and checkpoints (currently) availble at https://github.com/stet-stet/lmloss-icassp2026
>
> **摘要:** ``Phoneme Hallucinations (PH)'' commonly occur in low-bitrate DNN-based codecs. It is the generative decoder's attempt to synthesize plausible outputs from excessively compressed tokens missing some semantic information. In this work, we propose language model-driven losses (LM loss) and show they may alleviate PHs better than a semantic distillation (SD) objective in very-low-bitrate settings. The proposed LM losses build upon language models pretrained to associate speech with text. When ground-truth transcripts are unavailable, we propose to modify a popular automatic speech recognition (ASR) model, Whisper, to compare the decoded utterance against the ASR-inferred transcriptions of the input speech. Else, we propose to use the timed-text regularizer (TTR) to compare WavLM representations of the decoded utterance against BERT representations of the ground-truth transcriptions. We test and compare LM losses against an SD objective, using a reference codec whose three-stage training regimen was designed after several popular codecs. Subjective and objective evaluations conclude that LM losses may provide stronger guidance to extract semantic information from self-supervised speech representations, boosting human-perceived semantic adherence while preserving overall output quality. Demo samples, code, and checkpoints are available online.
>
---
#### [new 004] AI-Generated Music Detection in Broadcast Monitoring
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于AI音乐检测任务，解决广播环境中AI生成音乐识别问题。针对短时、被语音掩盖的音乐片段，构建了AI-OpenBMAT数据集，并评估模型性能。**

- **链接: [https://arxiv.org/pdf/2602.06823v1](https://arxiv.org/pdf/2602.06823v1)**

> **作者:** David Lopez-Ayala; Asier Cabello; Pablo Zinemanas; Emilio Molina; Martin Rocamora
>
> **摘要:** AI music generators have advanced to the point where their outputs are often indistinguishable from human compositions. While detection methods have emerged, they are typically designed and validated in music streaming contexts with clean, full-length tracks. Broadcast audio, however, poses a different challenge: music appears as short excerpts, often masked by dominant speech, conditions under which existing detectors fail. In this work, we introduce AI-OpenBMAT, the first dataset tailored to broadcast-style AI-music detection. It contains 3,294 one-minute audio excerpts (54.9 hours) that follow the duration patterns and loudness relations of real television audio, combining human-made production music with stylistically matched continuations generated with Suno v3.5. We benchmark a CNN baseline and state-of-the-art SpectTTTra models to assess SNR and duration robustness, and evaluate on a full broadcast scenario. Across all settings, models that excel in streaming scenarios suffer substantial degradation, with F1-scores dropping below 60% when music is in the background or has a short duration. These results highlight speech masking and short music length as critical open challenges for AI music detection, and position AI-OpenBMAT as a benchmark for developing detectors capable of meeting industrial broadcast requirements.
>
---
#### [new 005] EMG-to-Speech with Fewer Channels
- **分类: cs.SD**

- **简介: 该论文属于EMG-to-Speech任务，解决传感器通道减少导致性能下降的问题。通过预训练和通道选择提升重建效果，支持轻量级系统开发。**

- **链接: [https://arxiv.org/pdf/2602.06460v1](https://arxiv.org/pdf/2602.06460v1)**

> **作者:** Injune Hwang; Jaejun Lee; Kyogu Lee
>
> **摘要:** Surface electromyography (EMG) is a promising modality for silent speech interfaces, but its effectiveness depends heavily on sensor placement and channel availability. In this work, we investigate the contribution of individual and combined EMG channels to speech reconstruction performance. Our findings reveal that while certain EMG channels are individually more informative, the highest performance arises from subsets that leverage complementary relationships among channels. We also analyzed phoneme classification accuracy under channel ablations and observed interpretable patterns reflecting the anatomical roles of the underlying muscles. To address performance degradation from channel reduction, we pretrained models on full 8-channel data using random channel dropout and fine-tuned them on reduced-channel subsets. Fine-tuning consistently outperformed training from scratch for 4 - 6 channel settings, with the best dropout strategy depending on the number of channels. These results suggest that performance degradation from sensor reduction can be mitigated through pretraining and channel-aware design, supporting the development of lightweight and practical EMG-based silent speech systems.
>
---
#### [new 006] Automatic Detection and Analysis of Singing Mistakes for Music Pedagogy
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于音乐教育中的自动演唱错误检测任务，旨在解决学习者演唱错误识别问题。通过构建数据集并开发深度学习模型，提出新的评估方法，提升错误检测效果。**

- **链接: [https://arxiv.org/pdf/2602.06917v1](https://arxiv.org/pdf/2602.06917v1)**

> **作者:** Sumit Kumar; Suraj Jaiswal; Parampreet Singh; Vipul Arora
>
> **备注:** Under Review at Transactions of Audio Speech and Language Processing
>
> **摘要:** The advancement of machine learning in audio analysis has opened new possibilities for technology-enhanced music education. This paper introduces a framework for automatic singing mistake detection in the context of music pedagogy, supported by a newly curated dataset. The dataset comprises synchronized teacher learner vocal recordings, with annotations marking different types of mistakes made by learners. Using this dataset, we develop different deep learning models for mistake detection and benchmark them. To compare the efficacy of mistake detection systems, a new evaluation methodology is proposed. Experiments indicate that the proposed learning-based methods are superior to rule-based methods. A systematic study of errors and a cross-teacher study reveal insights into music pedagogy that can be utilised for various music applications. This work sets out new directions of research in music pedagogy. The codes and dataset are publicly available.
>
---
#### [new 007] The Combination of Several Decorrelation Methods to Improve Acoustic Feedback Cancellation
- **分类: eess.AS**

- **简介: 该论文属于声反馈消除任务，旨在提升系统性能。通过结合多种去相关方法，如时延、预测和混响模型，优化系统效果。**

- **链接: [https://arxiv.org/pdf/2602.06921v1](https://arxiv.org/pdf/2602.06921v1)**

> **作者:** Klaus Linhard; Philipp Bulling
>
> **摘要:** This paper extends an acoustic feedback cancellation system by incorporating multiple decorrelation methods. The baseline system is based on a frequency-domain Kalman filter implemented in a multi-delay structure. The proposed extensions include a variable time delay line, prediction, distortion compensation, and a simplified reverberation model. Each extension is analyzed, and a practical parameter range is defined. While existing literature often focuses on a single extension, such as prediction, to describe an optimal system, this work demonstrates that each individual extension contributes to performance improvements. Furthermore, the combination of all proposed extensions results in a superior system. The evaluation is conducted using publicly available datasets, with performance assessed through system distance metrics and the objective speech quality measure PSEQ.
>
---
#### [new 008] Misophonia Trigger Sound Detection on Synthetic Soundscapes Using a Hybrid Model with a Frozen Pre-Trained CNN and a Time-Series Module
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于声音事件检测任务，旨在解决misophonia触发声音的检测问题。通过合成音景和混合模型实现高效准确的检测。**

- **链接: [https://arxiv.org/pdf/2602.06271v1](https://arxiv.org/pdf/2602.06271v1)**

> **作者:** Kurumi Sashida; Gouhei Tanaka
>
> **备注:** 13 pages, 3 figures. Submitted to IJCNN 2026
>
> **摘要:** Misophonia is a disorder characterized by a decreased tolerance to specific everyday sounds (trigger sounds) that can evoke intense negative emotional responses such as anger, panic, or anxiety. These reactions can substantially impair daily functioning and quality of life. Assistive technologies that selectively detect trigger sounds could help reduce distress and improve well-being. In this study, we investigate sound event detection (SED) to localize intervals of trigger sounds in continuous environmental audio as a foundational step toward such assistive support. Motivated by the scarcity of real-world misophonia data, we generate synthetic soundscapes tailored to misophonia trigger sound detection using audio synthesis techniques. Then, we perform trigger sound detection tasks using hybrid CNN-based models. The models combine feature extraction using a frozen pre-trained CNN backbone with a trainable time-series module such as gated recurrent units (GRUs), long short-term memories (LSTMs), echo state networks (ESNs), and their bidirectional variants. The detection performance is evaluated using common SED metrics, including Polyphonic Sound Detection Score 1 (PSDS1). On the multi-class trigger SED task, bidirectional temporal modeling consistently improves detection performance, with Bidirectional GRU (BiGRU) achieving the best overall accuracy. Notably, the Bidirectional ESN (BiESN) attains competitive performance while requiring orders of magnitude fewer trainable parameters by optimizing only the readout. We further simulate user personalization via a few-shot "eating sound" detection task with at most five support clips, in which BiGRU and BiESN are compared. In this strict adaptation setting, BiESN shows robust and stable performance, suggesting that lightweight temporal modules are promising for personalized misophonia trigger SED.
>
---
#### [new 009] Scaling Speech Tokenizers with Diffusion Autoencoders
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决语音分词器在语义与声学平衡及低比特率和低令牌率上的挑战。提出SiTok模型，通过扩散自编码器实现高效语音表示学习与高质量重建。**

- **链接: [https://arxiv.org/pdf/2602.06602v1](https://arxiv.org/pdf/2602.06602v1)**

> **作者:** Yuancheng Wang; Zhenyu Tang; Yun Wang; Arthur Hinsvark; Yingru Liu; Yinghao Li; Kainan Peng; Junyi Ao; Mingbo Ma; Mike Seltzer; Qing He; Xubo Liu
>
> **备注:** ICLR 2026
>
> **摘要:** Speech tokenizers are foundational to speech language models, yet existing approaches face two major challenges: (1) balancing trade-offs between encoding semantics for understanding and acoustics for reconstruction, and (2) achieving low bit rates and low token rates. We propose Speech Diffusion Tokenizer (SiTok), a diffusion autoencoder that jointly learns semantic-rich representations through supervised learning and enables high-fidelity audio reconstruction with diffusion. We scale SiTok to 1.6B parameters and train it on 2 million hours of speech. Experiments show that SiTok outperforms strong baselines on understanding, reconstruction and generation tasks, at an extremely low token rate of $12.5$ Hz and a bit-rate of 200 bits-per-second.
>
---
#### [new 010] STACodec: Semantic Token Assignment for Balancing Acoustic Fidelity and Semantic Information in Audio Codecs
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出STACodec，解决音频编码中平衡声学保真与语义信息的问题，通过语义令牌分配和预蒸馏模块提升性能。**

- **链接: [https://arxiv.org/pdf/2602.06180v1](https://arxiv.org/pdf/2602.06180v1)**

> **作者:** Kaiyuan Zhang; Mohan Shi; Eray Eren; Natarajan Balaji Shankar; Zilai Wang; Abeer Alwan
>
> **备注:** ICASSP 2026
>
> **摘要:** Neural audio codecs are widely used for audio compression and can be integrated into token-based language models. Traditional codecs preserve acoustic details well but lack semantic information. Recent hybrid codecs attempt to incorporate semantic information through distillation, but this often degrades reconstruction performance, making it difficult to achieve both. To address this limitation, we introduce STACodec, a unified codec that integrates semantic information from self-supervised learning (SSL) models into the first layer of residual vector quantization (RVQ-1) via semantic token assignment (STA). To further eliminate reliance on SSL-based semantic tokenizers and improve efficiency during inference, we propose a semantic pre-distillation (SPD) module, which predicts semantic tokens directly for assignment to the first RVQ layer during inference. Experimental results show that STACodec outperforms existing hybrid codecs in both audio reconstruction and downstream semantic tasks, demonstrating a better balance between acoustic fidelity and semantic capability.
>
---
#### [new 011] Reciprocal Latent Fields for Precomputed Sound Propagation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于声学模拟任务，解决大场景中预计算声音传播的存储问题。提出RLF框架，通过潜在编码高效存储和预测声学参数，显著降低内存占用并保持音质。**

- **链接: [https://arxiv.org/pdf/2602.06937v1](https://arxiv.org/pdf/2602.06937v1)**

> **作者:** Hugo Seuté; Pranai Vasudev; Etienne Richan; Louis-Xavier Buffoni
>
> **备注:** Temporary pre-print, will be updated. In review at a conference
>
> **摘要:** Realistic sound propagation is essential for immersion in a virtual scene, yet physically accurate wave-based simulations remain computationally prohibitive for real-time applications. Wave coding methods address this limitation by precomputing and compressing impulse responses of a given scene into a set of scalar acoustic parameters, which can reach unmanageable sizes in large environments with many source-receiver pairs. We introduce Reciprocal Latent Fields (RLF), a memory-efficient framework for encoding and predicting these acoustic parameters. The RLF framework employs a volumetric grid of trainable latent embeddings decoded with a symmetric function, ensuring acoustic reciprocity. We study a variety of decoders and show that leveraging Riemannian metric learning leads to a better reproduction of acoustic phenomena in complex scenes. Experimental validation demonstrates that RLF maintains replication quality while reducing the memory footprint by several orders of magnitude. Furthermore, a MUSHRA-like subjective listening test indicates that sound rendered via RLF is perceptually indistinguishable from ground-truth simulations.
>
---
#### [new 012] B-GRPO: Unsupervised Speech Emotion Recognition based on Batched-Group Relative Policy Optimization
- **分类: eess.AS**

- **简介: 该论文属于语音情感识别任务，解决数据稀疏和标注偏差问题。通过改进的GRPO方法，利用强化学习提升样本质量评估，提高模型性能。**

- **链接: [https://arxiv.org/pdf/2602.06290v1](https://arxiv.org/pdf/2602.06290v1)**

> **作者:** Yingying Gao; Shilei Zhang; Runyan Yang; Zihao Cui; Junlan Feng
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** Unsupervised speech emotion recognition (SER) focuses on addressing the problem of data sparsity and annotation bias of emotional speech. Reinforcement learning (RL) is a promising method which enhances the performance through rule-based or model-based verification functions rather than human annotations. We treat the sample selection during the learning process as a long-term procedure and whether to select a sample as the action to make policy, thus achieving the application of RL to measure sample quality in SER. We propose a modified Group Relative Policy Optimization (GRPO) to adapt it to classification problems, which takes the samples in a batch as a group and uses the average reward of these samples as the baseline to calculate the advantage. And rather than using a verifiable reward function as in GRPO, we put forward self-reward functions and teacher-reward functions to encourage the model to produce high-confidence outputs. Experiments indicate that the proposed method improves the performance of baseline without RL by 19.8%.
>
---
#### [new 013] Reading Between the Waves: Robust Topic Segmentation Using Inter-Sentence Audio Features
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于话题分割任务，旨在提升语音内容的自动分段效果。通过融合文本与音频特征，提出一种多模态方法，增强对ASR噪声的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06647v1](https://arxiv.org/pdf/2602.06647v1)**

> **作者:** Steffen Freisinger; Philipp Seeberger; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** Spoken content, such as online videos and podcasts, often spans multiple topics, which makes automatic topic segmentation essential for user navigation and downstream applications. However, current methods do not fully leverage acoustic features, leaving room for improvement. We propose a multi-modal approach that fine-tunes both a text encoder and a Siamese audio encoder, capturing acoustic cues around sentence boundaries. Experiments on a large-scale dataset of YouTube videos show substantial gains over text-only and multi-modal baselines. Our model also proves more resilient to ASR noise and outperforms a larger text-only baseline on three additional datasets in Portuguese, German, and English, underscoring the value of learned acoustic features for robust topic segmentation.
>
---
## 更新

#### [replaced 001] ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation
- **分类: cs.SD**

- **简介: 该论文介绍ACE-Step 1.5，一个高效开源音乐生成模型，解决在消费级硬件上实现高质量音乐生成的问题。通过混合架构和内在强化学习，提升生成效率与风格控制能力。**

- **链接: [https://arxiv.org/pdf/2602.00744v3](https://arxiv.org/pdf/2602.00744v3)**

> **作者:** Junmin Gong; Yulin Song; Wenxiao Zhao; Sen Wang; Shengyuan Xu; Jing Guo; Xuerui Yang
>
> **摘要:** We present ACE-Step v1.5, a highly efficient open-source music foundation model that brings commercial-grade generation to consumer hardware. On commonly used evaluation metrics, ACE-Step v1.5 achieves quality beyond most commercial music models while remaining extremely fast -- under 2 seconds per full song on an A100 and under 10 seconds on an RTX 3090. The model runs locally with less than 4GB of VRAM, and supports lightweight personalization: users can train a LoRA from just a few songs to capture their own style. At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints -- scaling from short loops to 10-minute compositions -- while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences. Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities -- such as cover generation, repainting, and vocal-to-BGM conversion -- while maintaining strict adherence to prompts across 50+ languages. This paves the way for powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. The code, the model weights and the demo are available at: https://ace-step.github.io/ace-step-v1.5.github.io/
>
---
#### [replaced 002] AudioSAE: Towards Understanding of Audio-Processing Models with Sparse AutoEncoders
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频模型解释任务，旨在提升音频处理模型的可解释性。通过训练稀疏自编码器，分析模型特征，提升对音频内容的理解与控制。**

- **链接: [https://arxiv.org/pdf/2602.05027v2](https://arxiv.org/pdf/2602.05027v2)**

> **作者:** Georgii Aparin; Tasnima Sadekova; Alexey Rukhovich; Assel Yermekova; Laida Kushnareva; Vadim Popov; Kristian Kuznetsov; Irina Piontkovskaya
>
> **备注:** Accepted to EACL 2026, main track
>
> **摘要:** Sparse Autoencoders (SAEs) are powerful tools for interpreting neural representations, yet their use in audio remains underexplored. We train SAEs across all encoder layers of Whisper and HuBERT, provide an extensive evaluation of their stability, interpretability, and show their practical utility. Over 50% of the features remain consistent across random seeds, and reconstruction quality is preserved. SAE features capture general acoustic and semantic information as well as specific events, including environmental noises and paralinguistic sounds (e.g. laughter, whispering) and disentangle them effectively, requiring removal of only 19-27% of features to erase a concept. Feature steering reduces Whisper's false speech detections by 70% with negligible WER increase, demonstrating real-world applicability. Finally, we find SAE features correlated with human EEG activity during speech perception, indicating alignment with human neural processing. The code and checkpoints are available at https://github.com/audiosae/audiosae_demo.
>
---
#### [replaced 003] Benchmarking Automatic Speech Recognition for Indian Languages in Agricultural Contexts
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决农业领域多语言ASR系统的性能评估问题。通过构建基准框架，分析不同语言和模型的表现，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2602.03868v2](https://arxiv.org/pdf/2602.03868v2)**

> **作者:** Chandrashekar M S; Vineet Singh; Lakshmi Pedapudi
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** The digitization of agricultural advisory services in India requires robust Automatic Speech Recognition (ASR) systems capable of accurately transcribing domain-specific terminology in multiple Indian languages. This paper presents a benchmarking framework for evaluating ASR performance in agricultural contexts across Hindi, Telugu, and Odia languages. We introduce evaluation metrics including Agriculture Weighted Word Error Rate (AWWER) and domain-specific utility scoring to complement traditional metrics. Our evaluation of 10,934 audio recordings, each transcribed by up to 10 ASR models, reveals performance variations across languages and models, with Hindi achieving the best overall performance (WER: 16.2%) while Odia presents the greatest challenges (best WER: 35.1%, achieved only with speaker diarization). We characterize audio quality challenges inherent to real-world agricultural field recordings and demonstrate that speaker diarization with best-speaker selection can substantially reduce WER for multi-speaker recordings (upto 66% depending on the proportion of multi-speaker audio). We identify recurring error patterns in agricultural terminology and provide practical recommendations for improving ASR systems in low-resource agricultural domains. The study establishes baseline benchmarks for future agricultural ASR development.
>
---
