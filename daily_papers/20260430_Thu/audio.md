# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] SongBench: A Fine-Grained Multi-Aspect Benchmark for Song Quality Assessment
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于歌曲质量评估任务，旨在解决现有基准缺乏专业细粒度评价的问题。提出SongBench框架，涵盖七个维度，构建专家标注数据集，用于评估和改进音乐生成模型。**

- **链接: [https://arxiv.org/pdf/2604.25937](https://arxiv.org/pdf/2604.25937)**

> **作者:** Dapeng Wu; Shun Lei; Wei Tan; Guangzheng Li; Yunzhe Wang; Huaicheng Zhang; Lishi Zuo; Zhiyong Wu
>
> **摘要:** Recent advancements in Text-to-Song generation have enabled realistic musical content production, yet existing evaluation benchmarks lack the professional granularity to capture multi-dimensional aesthetic nuances. In this paper, we propose SongBench, a specialized framework for fine-grained song assessment across seven key dimensions: Vocal, Instrument, Melody, Structure, Arrangement, Mixing, and Musicality. Utilizing this framework, we construct an expert-annotated database comprising 11,717 samples from state-of-the-art models, labeled by music professionals. Extensive experimental results demonstrate that SongBench achieves high correlation with expert ratings. By revealing fine-grained performance gaps in current state-of-the-art models, SongBench serves as a diagnostic benchmark to steer the development toward more professional and musically coherent song generation.
>
---
#### [new 002] Similarity Choice and Negative Scaling in Supervised Contrastive Learning for Deepfake Audio Detection
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于深度伪造音频检测任务，研究监督对比学习的相似性选择与负样本缩放，旨在提升检测性能。通过实验对比不同相似度计算方式和负样本策略，优化模型效果。**

- **链接: [https://arxiv.org/pdf/2604.26057](https://arxiv.org/pdf/2604.26057)**

> **作者:** Jaskirat Sudan; Hashim Ali; Surya Subramani; Hafiz Malik
>
> **摘要:** Supervised contrastive learning (SupCon) is widely used to shape representations, but has seen limited targeted study for audio deepfake detection. Existing work typically combines contrastive terms with broader pipelines; however, the focus on SupCon itself is missing. In this work, we run a controlled study on wav2vec2 XLS-R (300M) that varies (i) similarity in SupCon (cosine vs angular similarity derived from the hyperspherical angle) and (ii) negative scaling using a warm-started global cross-batch queue. Stage 1 fine-tunes the encoder and projection head with SupCon; Stage 2 freezes them and trains a linear classifier with BCE. Trained on ASVspoof 2019 LA and evaluated on ASV19 eval plus ITW and ASVspoof 2021 DF/LA, Cosine SupCon with a delayed queue achieves the best ITW EER (8.29%) and pooled EER (4.44), while angular similarity performs strongly without queued negatives (ITW 8.70), indicating reduced reliance on large negative sets.
>
---
#### [new 003] The False Resonance: A Critical Examination of Emotion Embedding Similarity for Speech Generation Evaluation
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音生成评估任务，旨在解决情感相似性度量的可靠性问题。研究发现现有方法受语言和说话者干扰，无法准确反映情感表达，提出其不适用于零样本评估。**

- **链接: [https://arxiv.org/pdf/2604.26347](https://arxiv.org/pdf/2604.26347)**

> **作者:** Yun-Shao Tsai; Yi-Cheng Lin; Huang-Cheng Chou; Tzu-Wen Hsu; Yun-Man Hsu; Chun Wei Chen; Shrikanth Narayanan; Hung-yi Lee
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Objective metrics for emotional expressiveness are vital for speech generation, particularly in expressive synthesis and voice conversion requiring emotional prosody transfer. To quantify this, the field widely relies on emotion similarity between reference and generated samples. This approach computes cosine similarity of embeddings from encoders like emotion2vec, assuming they capture affective cues despite linguistic and speaker variations. We challenge this assumption through controlled adversarial tasks and human alignment tests. Despite high classification accuracy, these latent spaces are unsuitable for zero-shot similarity evaluation. Representational limitations cause linguistic and speaker interference to overshadow emotional features, degrading discriminative ability. Consequently, the metric misaligns with human perception. This acoustic vulnerability reveals it rewards acoustic mimicry over genuine emotional synthesis.
>
---
#### [new 004] Diffusion Reconstruction towards Generalizable Audio Deepfake Detection
- **分类: cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在提升模型对未见攻击的泛化能力。通过硬样本分类、扩散重建和对比学习增强模型性能。**

- **链接: [https://arxiv.org/pdf/2604.26465](https://arxiv.org/pdf/2604.26465)**

> **作者:** Bo Cheng; Songjun Cao; Xiaoming Zhang; Jie Chen; Long Ma; Fei Chen
>
> **备注:** 5 pages, this paper was submitted to Interspeech2026 for review
>
> **摘要:** Achieving robust generalization against unseen attacks remains a challenge in Audio Deepfake Detection (ADD), driven by the rapid evolution of generative models. To address this, we propose a framework centered on hard sample classification. The core idea is that a model capable of distinguishing challenging hard samples is inherently equipped to handle simpler cases effectively. We investigate multiple reconstruction paradigms, identifying the diffusion-based method as optimal for generating hard samples. Furthermore, we leverage multi-layer feature aggregation and introduce a Regularization-Assisted Contrastive Learning (RACL) objective to enhance generalizability. Experiments demonstrate the superior generalization of our approach, with our best model achieving a significant reduction in the average Equal Error Rate (EER) compared to the baseline.
>
---
#### [new 005] A Toolkit for Detecting Spurious Correlations in Speech Datasets
- **分类: cs.SD; cs.AI; cs.DB**

- **简介: 该论文属于语音数据质量检测任务，旨在解决语音数据中虚假相关性问题。通过分析非语音区域，检测目标类别的信息泄露，从而识别虚假相关性。**

- **链接: [https://arxiv.org/pdf/2604.26676](https://arxiv.org/pdf/2604.26676)**

> **作者:** Lara Gauder; Pablo Riera; Andrea Slachevsky; Gonzalo Forno; Adolfo M. García; Luciana Ferrer
>
> **摘要:** We introduce a toolkit for uncovering spurious correlations between recording characteristics and target class in speech datasets. Spurious correlations may arise due to heterogeneous recording conditions, a common scenario for health-related datasets. When present both in the training and test data, these correlations result in an overestimation of the system performance -- a dangerous situation, specially in high-stakes application where systems are required to satisfy minimum performance requirements. Our toolkit implements a diagnostic method based on the detection of the target class using only the non-speech regions in the audio. Better than chance performance at this task indicates that information about the target class can be extracted from the non-speech regions, flagging the presence of spurious correlations. The toolkit is publicly available for research use.
>
---
#### [new 006] Full band denoising of room impulse response in the wavelet domain with dictionary learning
- **分类: cs.SD; math.OC**

- **简介: 该论文属于语音信号处理任务，旨在解决房间脉冲响应低频降噪问题。通过小波域稀疏字典学习和时变误差容忍度，提升低频降噪效果。**

- **链接: [https://arxiv.org/pdf/2604.26669](https://arxiv.org/pdf/2604.26669)**

> **作者:** Théophile Dupré; Romain Couderc; Miguel Moleron; Axel Coulon; Rémy Bruno; Arnaud Laborie
>
> **摘要:** Conventional wavelet-domain methods for room impulse response denoising rely on thresholding detail coefficients, which is unsuited for low frequencies. In this work, we introduce a wavelet-based post-processing algorithm that extends denoising to approximation coefficients by means of sparse dictionary learning with a time-varying error tolerance. The proposed method leverages an exponential decay envelope model to adapt reconstruction accuracy according to the local signal-to-noise ratio. This approach significantly improves low-frequency denoising of synthetic and measured room impulse responses compared to the baseline method, leading to more accurate estimation of acoustic parameters such as decay time.
>
---
#### [new 007] Dual-LoRA: Parameter-Efficient Adversarial Disentanglement for Cross-Lingual Speaker Verification
- **分类: eess.AS**

- **简介: 该论文属于跨语言语音验证任务，旨在解决语言与说话人混淆问题。通过引入Dual-LoRA框架，提升说话人区分能力，同时分离语言特征。**

- **链接: [https://arxiv.org/pdf/2604.26327](https://arxiv.org/pdf/2604.26327)**

> **作者:** Qituan Shangguan; Junhao Du; Kunyang Peng; Feng Xue; Hui Zhang; Xinsheng Wang; Kai Yu; Shuai Wang
>
> **备注:** Submitted to Interspeech 2026; 5 pages
>
> **摘要:** Cross-lingual speaker verification suffers from severe language-speaker entanglement. This causes systematic degradation in the hardest scenario: correctly accepting utterances from the same speaker across different languages while rejecting those from different speakers sharing the same language. Standard adversarial disentanglement degrades speaker discriminability; blind discriminators inadvertently penalize speaker-discriminative traits that merely correlate with language. To address this, we propose Dual-LoRA, injecting trainable task-factorized LoRA adapters into a frozen pre-trained backbone. Our core innovation is a Language-Anchored Adversary: by grounding the discriminator with an explicit language branch, adversarial gradients target true linguistic cues rather than arbitrary correlations, preserving essential speaker characteristics. Evaluated on the TidyVoice benchmark, our system achieves a 0.91% validation EER and achieves 3rd place in the official challenge.
>
---
#### [new 008] One Voice, Many Tongues: Cross-Lingual Voice Cloning for Scientific Speech
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于跨语言语音克隆任务，旨在保留说话人声音特征的同时生成不同语言的语音。通过优化模型和数据增强，提升科学文本的语音生成质量与可懂度。**

- **链接: [https://arxiv.org/pdf/2604.26136](https://arxiv.org/pdf/2604.26136)**

> **作者:** Amanuel Gizachew Abebe; Yasmin Moslem
>
> **备注:** IWSLT 2026
>
> **摘要:** Preserving a speaker's voice identity while generating speech in a different language remains a fundamental challenge in spoken language technology, particularly in specialized domains such as scientific communication. In this paper, we address this challenge through our system submission to the International Conference on Spoken Language Translation (IWSLT 2026), the Cross-Lingual Voice Cloning shared task. First, we evaluate several state-of-the-art voice cloning models for cross-lingual speech generation of scientific texts in Arabic, Chinese, and French. Then, we build voice cloning systems based on the OmniVoice foundation model. We employ data augmentation via multi-model ensemble distillation from the ACL 60/60 corpus. We investigate the effect of using this synthetic data for fine-tuning, demonstrating consistent improvements in intelligibility (WER and CER) across languages while preserving speaker similarity.
>
---
#### [new 009] Speech Emotion Recognition Using MFCC Features and LSTM-Based Deep Learning Model
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在通过MFCC和LSTM模型准确识别语音中的情感。工作包括特征提取与模型训练，实验表明LSTM效果优于SVM。**

- **链接: [https://arxiv.org/pdf/2604.25938](https://arxiv.org/pdf/2604.25938)**

> **作者:** Adelekun Oluwademilade; Ademola Adedamola; Abiola Abdulhakeem; Akinpelu Azeezat; Eraiyetan Israel; Omotosho Oluwadunsin; Ibenye Ikechukwu; Ayuba Muhammad; Olusanya Olamide; Kamorudeen Amuda
>
> **摘要:** Speech Emotion Recognition (SER) is the use of machines to detect the emotional state of humans based on the speech, which is gaining importance in natural human-computer interaction. Speech is a very valuable source of information, as emotions modify the patterns of speech; pitch, energy and even timing. Nonetheless, SER is not an easy task because speakers are not constant, and situations vary when recording and the sound similarity between specific feelings. In this work, the author introduces a speech emotion recognition system relying on the Mel-Frequency Cepstral Coefficient and Long Short-Term Memory (LSTM) neural network, as a feature extraction method. The Toronto Emotional Speech Set (TESS) speech signal was pre-processed, and transformed into MFCC features to understand the important aspects in terms of time. The resultant features were then introduced to LSTM model, which is able to learn long term features of sequential audio data. The trained model was measured over several emotion classes occurring in the dataset. As seen in the results of experiments, the proposed MFCC-LSTM approach succeeds in capturing the patterns of emotions in speech and provides highly realistic classifications in all the chosen emotion classifications. This study presents a speech emotion recognition system using Mel-Frequency Cepstral Coefficients (MFCCs) as features and a deep learning LSTM classifier. A Support Vector Machine (SVM) with an RBF kernel served as a classical baseline, achieving 98% accuracy, against which the proposed LSTM model, achieving 99% accuracy, was validated. Overall, it is possible to confirm that LSTM-based architectures can be used to address the task of speech emotion recognition. Actual applications of the proposed system may be virtual assistants and mental health surveillance.
>
---
#### [new 010] SPG-Codec: Exploring the Role and Boundaries of Semantic Priors in Ultra-Low-Bitrate Neural Speech Coding
- **分类: eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决超低比特率下语音可懂性下降的问题。通过引入语义先验，研究其作用与限制，并提出动态调节策略以平衡语义一致性和自然度。**

- **链接: [https://arxiv.org/pdf/2604.26296](https://arxiv.org/pdf/2604.26296)**

> **作者:** Mingyu Zhao; Zijian Lin; Kun Wei; Zhiyong Wu
>
> **备注:** 6 pages, 6 figures, accepted to ICME 2026
>
> **摘要:** Conventional neural speech codecs suffer from severe intelligibility degradation at ultra-low bitrates, where the bottleneck transitions from acoustic distortion to semantic loss. To address this issue, this paper conducts a systematic investigation into the role and fundamental limits of integrating frozen semantic priors -- specifically HuBERT and Whisper -- into neural speech coding. We introduce and quantitatively validate a novel Semantic Retirement phenomenon: while semantic constraints reduce the Word Error Rate (WER) by up to ~10% relatively at 1.5 kbps, their benefits rapidly diminish beyond 6 kbps, indicating a practical capacity boundary. We further uncover a clear trade-off between different prior types: acoustic-rich priors (HuBERT) better preserve prosodic and timbral details, whereas high-level linguistic priors (Whisper) effectively suppress phonetic hallucinations in noisy environments (reducing hallucination rates by 26 percent) and substantially narrow the generalization gap for unseen speakers. Building on these findings, we propose a bitrate-aware regulation strategy that dynamically adjusts prior strength to optimize the trade-off between semantic consistency and perceptual naturalness. Extensive experimental evaluations confirm that our approach achieves competitive intelligibility and noise robustness compared to existing baselines, offering a principled pathway toward ultra-low-bitrate generative speech coding.
>
---
#### [new 011] Recurrence-Based Nonlinear Vocal Dynamics as Digital Biomarkers for Depression Detection from Conversational Speech
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在通过对话语音中的非线性声学动态识别抑郁特征。工作包括构建基于复发结构的生物标志物，并验证其分类性能。**

- **链接: [https://arxiv.org/pdf/2604.26242](https://arxiv.org/pdf/2604.26242)**

> **作者:** Himadri S Samanta
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Digital biomarkers for depression have largely relied on static acoustic descriptors, pooled summary statistics, or conventional machine learning representations. Such approaches may miss nonlinear temporal organization embedded in conversational vocal dynamics. We hypothesized that depression is associated with altered recurrence structure in vocal state trajectories, reflecting changes in how the vocal system revisits acoustic states over time. Using the depression subset of the DAIC-WOZ corpus with 142 labeled participants, we modeled frame-level COVAREP trajectories as nonlinear dynamical systems and derived recurrence-based biomarkers from 74 vocal channels. Logistic regression with feature selection and stratified cross-validation evaluated classification performance. Recurrence-based biomarkers achieved a mean cross-validated AUC of 0.689, exceeding static acoustic baselines, entropy-dynamics features, Hurst exponent features, determinism features, and Lyapunov-like instability proxies. Permutation testing indicated statistical significance with $p=0.004$. Pooled cross-validated predictions yielded AUC 0.665 with a 95\% bootstrap confidence interval of [0.568, 0.758]. These findings suggest that depression may be characterized by altered recurrence structure in conversational vocal dynamics and support nonlinear state-space analysis as a promising direction for digital psychiatric biomarkers.
>
---
#### [new 012] DiffAnon: Diffusion-based Prosody Control for Voice Anonymization
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音匿名化任务，旨在解决隐私与语音特征保留的平衡问题。提出DiffAnon方法，通过扩散模型实现对语音韵律的连续控制，提升匿名化效果。**

- **链接: [https://arxiv.org/pdf/2604.26281](https://arxiv.org/pdf/2604.26281)**

> **作者:** Ismail Rasim Ulgen; Zexin Cai; Nicholas Andrews; Philipp Koehn; Berrak Sisman
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** To preserve or not to preserve prosody is a central question in voice anonymization. Prosody conveys meaning and affect, yet is tightly coupled with speaker identity. Existing methods either discard prosody for privacy or lack a principled mechanism to control the utility-privacy trade-off, operating at fixed design points. We propose DiffAnon, a diffusion-based anonymization method with classifier-free guidance (CFG) that provides explicit, continuous inference-time control over prosody preservation. DiffAnon refines acoustic detail over semantic embeddings of an RVQ codec, enabling smooth interpolation between anonymization strength and prosodic fidelity within a single model. To the best of our knowledge, it is the first voice anonymization framework to provide structured, interpolatable inference-time prosody control. Experiments demonstrate structured trade-off behavior, achieving strong utility while maintaining competitive privacy across controllable operating points.
>
---
#### [new 013] EmoTransCap: Dataset and Pipeline for Emotion Transition-Aware Speech Captioning in Discourses
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出EmoTransCap，解决话语层面情感过渡的语音描述问题。构建包含情感过渡的大型数据集，设计多任务模型与合成系统，提升情感理解与表达能力。**

- **链接: [https://arxiv.org/pdf/2604.26417](https://arxiv.org/pdf/2604.26417)**

> **作者:** Shuhao Xu; Yifan Hu; Jingjing Wu; Zhihao Du; Zheng Lian; Rui Liu
>
> **备注:** 15 pages, 5 figures, including appendix
>
> **摘要:** Emotion perception and adaptive expression are fundamental capabilities in human-agent interaction. While recent advances in speech emotion captioning (SEC) have improved fine-grained emotional modeling, existing systems remain limited to static, single-emotion characterization within isolated sentences, neglecting dynamic emotional transitions at the discourse level. To address this gap, we propose Emotion Transition-Aware Speech Captioning (EmoTransCap), a paradigm that integrates temporal emotion dynamics with discourse-level speech description. To construct a dataset rich in emotion transitions while enabling scalable expansion, we design an automated pipeline for dataset creation. This is the first large-scale dataset explicitly designed to capture discourse-level emotion transitions. To generate semantically rich descriptions, we incorporate acoustic attributes and temporal cues from discourse-level speech. Our Multi-Task Emotion Transition Recognition (MTETR) model performs joint emotion transition detection and diarization. Leveraging the semantic analysis capabilities of LLMs, we produce two annotation versions: descriptive and instruction-oriented. These data and annotations offer a valuable resource for advancing emotion perception and emotional expressiveness. The dataset enables speech captions that capture emotional transitions, facilitating temporal-dynamic and fine-grained emotion understanding. We also introduce a controllable, transition-aware emotional speech synthesis system at the discourse level, enhancing anthropomorphic emotional expressiveness and supporting emotionally intelligent conversational agents.
>
---
## 更新

#### [replaced 001] Woosh: A Sound Effects Foundation Model
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文介绍Woosh，一个用于音效生成的开源基础模型，解决音效生成与文本/视频对齐问题，包含编码器、解码器及多模态生成模块。**

- **链接: [https://arxiv.org/pdf/2604.01929](https://arxiv.org/pdf/2604.01929)**

> **作者:** Gaëtan Hadjeres; Marc Ferras; Khaled Koutini; Benno Weck; Alexandre Bittar; Thomas Hummel; Zineb Lahrichi; Hakim Missoum; Joan Serrà; Yuki Mitsufuji
>
> **摘要:** The audio research community depends on open generative models as foundational tools for building novel approaches and establishing baselines. In this report, we present Woosh, Sony AI's publicly released sound effect foundation model, detailing its architecture, training process, and an evaluation against other popular open models. Being optimized for sound effects, we provide (1) a high-quality audio encoder/decoder model and (2) a text-audio alignment model for conditioning, together with (3) text-to-audio and (4) video-to-audio generative models. Distilled text-to-audio and video-to-audio models are also included in the release, allowing for low-resource operation and fast inference. Our evaluation on both public and private data shows competitive or better performance for each module when compared to existing open alternatives like StableAudio-Open and TangoFlux. Inference code and model weights are available at this https URL. Demo samples can be found at this https URL.
>
---
#### [replaced 002] Graph Propagated Projection Unlearning: A Unified Framework for Vision and Audio Discriminative Models
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出GPPU，解决深度学习模型中类级信息擦除问题，通过图传播和投影实现高效、不可逆的模型遗忘，适用于视觉和音频任务。**

- **链接: [https://arxiv.org/pdf/2604.13127](https://arxiv.org/pdf/2604.13127)**

> **作者:** Shreyansh Pathak; Jyotishman Das
>
> **备注:** This submission has been withdrawn because it is posted accidentally without full author approval. A revised version may be submitted with full approval anytime soon
>
> **摘要:** The need to selectively and efficiently erase learned information from deep neural networks is becoming increasingly important for privacy, regulatory compliance, and adaptive system design. We introduce Graph-Propagated Projection Unlearning (GPPU), a unified and scalable algorithm for class-level unlearning that operates across both vision and audio models. GPPU employs graph-based propagation to identify class-specific directions in the feature space and projects representations onto the orthogonal subspace, followed by targeted fine-tuning, to ensure that target class information is effectively and irreversibly removed. Through comprehensive evaluations on six vision datasets and two large-scale audio benchmarks spanning a variety of architectures including CNNs, Vision Transformers, and Audio Transformers, we demonstrate that GPPU achieves highly efficient unlearning, realizing 10-20x speedups over prior methodologies while preserving model utility on retained classes. Our framework provides a principled and modality-agnostic approach to machine unlearning, evaluated at a scale that has received limited attention in prior work, contributing toward more efficient and responsible deep learning.
>
---
#### [replaced 003] Multi-Speaker DOA Estimation in Binaural Hearing Aids using Deep Learning and Speaker Count Fusion
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多说话人声源定位任务，旨在提升助听器在嘈杂环境中的目标语音提取效果。通过融合说话人数目信息，改进深度学习模型的定位性能。**

- **链接: [https://arxiv.org/pdf/2509.21382](https://arxiv.org/pdf/2509.21382)**

> **作者:** Farnaz Jazaeri; Homayoun Kamkar-Parsi; François Grondin; Martin Bouchard
>
> **备注:** 5 pages, 2 figures, to appear in IEEE ICASSP 2026
>
> **摘要:** For extracting a target speaker voice, direction-of-arrival (DOA) estimation is crucial for binaural hearing aids operating in noisy, multi-speaker environments. Among the solutions developed for this task, a deep learning convolutional recurrent neural network (CRNN) model leveraging spectral phase differences and magnitude ratios between microphone signals is a popular option. In this paper, we explore adding source-count information for multi-sources DOA estimation. The use of dual-task training with joint multi-sources DOA estimation and source counting is first considered. We then consider using the source count as an auxiliary feature in a standalone DOA estimation system, where the number of active sources (0, 1, or 2+) is integrated into the CRNN architecture through early, mid, and late fusion strategies. Experiments using real binaural recordings are performed. Results show that the dual-task training does not improve DOA estimation performance, although it benefits source-count prediction. However, a ground-truth (oracle) source count used as an auxiliary feature significantly enhances standalone DOA estimation performance, with late fusion yielding up to 14% higher average F1-scores over the baseline CRNN. This highlights the potential of using source-count estimation for robust DOA estimation in binaural hearing aids.
>
---
#### [replaced 004] Text-To-Speech with Chain-of-Details: modeling temporal dynamics in speech generation
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决语音生成中的时间动态建模问题。提出Chain-of-Details框架，通过级联结构逐步细化时间细节，提升合成自然度。**

- **链接: [https://arxiv.org/pdf/2604.19330](https://arxiv.org/pdf/2604.19330)**

> **作者:** Jianbo Ma; Richard Cartwright
>
> **摘要:** Recent advances in Text-To-Speech (TTS) synthesis have seen the popularity of multi-stage approaches that first predict semantic tokens and then generate acoustic tokens. In this paper, we extend the coarse-to-fine generation paradigm to the temporal domain and introduce Chain-of-Details (CoD), a novel framework that explicitly models temporal coarse-to-fine dynamics in speech generation using a cascaded architecture. Our method progressively refines temporal details across multiple stages, with each stage targeting a specific temporal granularity. All temporal detail predictions are performed using a shared decoder, enabling efficient parameter utilization across different temporal resolutions. Notably, we observe that the lowest detail level naturally performs phonetic planning without the need for an explicit phoneme duration predictor. We evaluate our method on several datasets and compare it against several baselines. Experimental results show that CoD achieves competitive performance with significantly fewer parameters than existing approaches. Our findings demonstrate that explicit modeling of temporal dynamics with the CoD framework leads to more natural speech synthesis.
>
---
#### [replaced 005] A Dataset for Automatic Vocal Mode Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音模式分类任务，旨在解决自动识别歌唱声音模式的问题。研究者构建了一个包含3,752个样本的新型数据集，并提供了基线分类结果。**

- **链接: [https://arxiv.org/pdf/2601.18339](https://arxiv.org/pdf/2601.18339)**

> **作者:** Reemt Hinrichs; Sonja Stephan; Alexander Lange; Jörn Ostermann
>
> **备注:** Extended manuscript of our Article in the proceedings of the EvoMUSART 2026: 15th International Conference on Artificial Intelligence in Music, Sound, Art and Design; Tiny corrigendum to v1, where the pitch distribution showed an incorrect F1. The truely lowest note of the dataset is a B1
>
> **摘要:** The Complete Vocal Technique (CVT) is a school of singing developed in the past decades by Cathrin Sadolin et al.. CVT groups the use of the voice into so called vocal modes, namely Neutral, Curbing, Overdrive and Edge. Knowledge of the desired vocal mode can be helpful for singing students. Automatic classification of vocal modes can thus be important for technology-assisted singing teaching. Previously, automatic classification of vocal modes has been attempted without major success, potentially due to a lack of data. Therefore, we recorded a novel vocal mode dataset consisting of sustained vowels recorded from four singers, three of which professional singers with more than five years of CVT-experience. The dataset covers the entire vocal range of the subjects, totaling 3,752 unique samples. By using four microphones, thereby offering a natural data augmentation, the dataset consists of more than 13,000 samples combined. An annotation was created using three CVT-experienced annotators, each providing an individual annotation. The merged annotation as well as the three individual annotations come with the published dataset. Additionally, we provide some baseline classification results. The best balanced accuracy across a 5-fold cross validation of 81.3\,\% was achieved with a ResNet18. The dataset can be downloaded under this https URL.
>
---
#### [replaced 006] Explainable Detection of Machine Generated Music and Early Systematic Evaluation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于机器生成音乐检测任务，旨在解决MGMD缺乏系统评估的问题。通过实验对比多种模型，并利用XAI工具分析模型行为，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2412.13421](https://arxiv.org/pdf/2412.13421)**

> **作者:** Yupei Li; Qiyang Sun; Hanqian Li; Lucia Specia; Björn W. Schuller
>
> **备注:** Accepted at Scientific report
>
> **摘要:** Machine-generated music (MGM) has become a groundbreaking innovation with wide-ranging applications, such as music therapy, personalised editing, and creative inspiration within the music industry. However, the unregulated proliferation of MGM presents considerable challenges to the entertainment, education, and arts sectors by potentially undermining the value of high-quality human compositions. Consequently, MGM detection (MGMD) is crucial for preserving the integrity of these fields. Despite its significance, MGMD domain lacks comprehensive systematic evaluation results necessary to drive meaningful progress. To address this gap, we conduct experiments on existing large-scale datasets using a range of foundational models for audio processing, establishing systematic evaluation results tailored to the MGMD task. Our selection includes traditional machine learning models, deep neural networks, Transformer-based architectures, and State space models (SSM). Recognising the inherently multimodal nature of music, which integrates both melody and lyrics, we also explore fundamental multimodal models in our experiments. Beyond providing basic binary classification outcomes, we delve deeper into model behaviour using multiple explainable Artificial Intelligence (XAI) tools, offering insights into their decision-making processes. Our analysis reveals that ResNet18 performs the best according to in-domain and out-of-domain tests. By providing a comprehensive comparison of systematic evaluation results and their interpretability, we propose several directions to inspire future research to develop more robust and effective detection methods for MGM. We provide our codes and some samples on Github repository.
>
---
#### [replaced 007] Omni2Sound: Towards Unified Video-Text-to-Audio Generation
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文属于视频-文本到音频生成任务，解决数据稀缺与多任务竞争问题。提出SoundAtlas数据集和Omni2Sound模型，实现统一的音视频生成。**

- **链接: [https://arxiv.org/pdf/2601.02731](https://arxiv.org/pdf/2601.02731)**

> **作者:** Yusheng Dai; Zehua Chen; Yuxuan Jiang; Baolong Gao; Qiuhong Ke; Jianfei Cai; Jun Zhu
>
> **摘要:** Training a unified model integrating video-to-audio (V2A), text-to-audio (T2A), and joint video-text-to-audio (VT2A) generation offers significant application flexibility, yet faces two unexplored foundational challenges: (1) the scarcity of high-quality audio captions with tight V-A-T alignment, leading to severe semantic conflict between multimodal conditions, and (2) cross-task and intra-task competition, manifesting as an adverse V2A-T2A performance trade-off and modality bias in the VT2A task. First, to address data scarcity, we introduce SoundAtlas, a large-scale dataset (470k pairs) that significantly outperforms existing benchmarks and even human experts in quality. Powered by a novel agentic pipeline, it integrates Vision-to-Language Compression to mitigate visual bias of MLLMs, a Junior-Senior Agent Handoff for a 5$\times$ cost reduction, and rigorous Post-hoc Filtering to ensure fidelity. Consequently, SoundAtlas delivers semantically rich and temporally detailed captions with tight V-A-T alignment. Second, we propose Omni2Sound, a unified VT2A diffusion model supporting flexible input modalities. To resolve the inherent cross-task and intra-task competition, we design a three-stage multi-task progressive training schedule that converts cross-task competition into joint optimization and mitigates modality bias in the VT2A task, maintaining both audio-visual alignment and off-screen audio generation faithfulness. Finally, we construct VGGSound-Omni, a comprehensive benchmark for unified evaluation, including challenging off-screen tracks. With a standard DiT backbone, Omni2Sound achieves unified SOTA performance across all three tasks within a single model, demonstrating strong generalization across benchmarks with heterogeneous input conditions.
>
---
