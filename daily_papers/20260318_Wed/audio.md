# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Diffusion Models for Joint Audio-Video Generation
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于多模态生成任务，旨在解决音频与视频联合生成的挑战。通过构建数据集、训练模型、探索联合扩散方法，并提出两步生成流程，提升生成质量与同步性。**

- **链接: [https://arxiv.org/pdf/2603.16093](https://arxiv.org/pdf/2603.16093)**

> **作者:** Alejandro Paredes La Torre
>
> **摘要:** Multimodal generative models have shown remarkable progress in single-modality video and audio synthesis, yet truly joint audio-video generation remains an open challenge. In this paper, I explore four key contributions to advance this field. First, I release two high-quality, paired audio-video datasets. The datasets consisting on 13 hours of video-game clips and 64 hours of concert performances, each segmented into consistent 34-second samples to facilitate reproducible research. Second, I train the MM-Diffusion architecture from scratch on our datasets, demonstrating its ability to produce semantically coherent audio-video pairs and quantitatively evaluating alignment on rapid actions and musical cues. Third, I investigate joint latent diffusion by leveraging pretrained video and audio encoder-decoders, uncovering challenges and inconsistencies in the multimodal decoding stage. Finally, I propose a sequential two-step text-to-audio-video generation pipeline: first generating video, then conditioning on both the video output and the original prompt to synthesize temporally synchronized audio. My experiments show that this modular approach yields high-fidelity generations of audio video generation.
>
---
#### [new 002] AILive Mixer: A Deep Learning based Zero Latency Automatic Music Mixer for Live Music Performances
- **分类: eess.AS**

- **简介: 该论文提出一种基于深度学习的实时音乐混音系统AILive Mixer，解决现场演出中音频干扰和零延迟问题。属于音乐混音任务，旨在实现低延迟、自动化的多轨音频混合。**

- **链接: [https://arxiv.org/pdf/2603.15995](https://arxiv.org/pdf/2603.15995)**

> **作者:** Devansh Zurale; Iris Lorente; Michael Lester; Alex Mitchell
>
> **备注:** 5 pages, 4 figures, accepted to ICASSP 2026
>
> **摘要:** In this work, we present a deep learning-based automatic multitrack music mixing system catered towards live performances. In a live performance, channels are often corrupted with acoustic bleeds of co-located instruments. Moreover, audio-visual synchronization is of critical importance thus putting a tight constraint on the audio latency. In this work we primarily tackle these two challenges of handling bleeds in the input channels to produce the music mix with zero latency. Although there have been several developments in the field of automatic music mixing in recent times, most or all previous works focus on offline production for isolated instrument signals and to the best of our knowledge, this is the first end-to-end deep learning system developed for live music performances. Our proposed system currently predicts mono gains for a multitrack input, but its design along with the precedent set in past works, allows for easy adaptation to future work of predicting other relevant music mixing parameters.
>
---
#### [new 003] Something from Nothing: Data Augmentation for Robust Severity Level Estimation of Dysarthric Speech
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文属于语音质量评估任务，解决 dysarthric 语音严重程度估计问题。通过数据增强和预训练方法，提升模型在缺乏标注数据情况下的性能。**

- **链接: [https://arxiv.org/pdf/2603.15988](https://arxiv.org/pdf/2603.15988)**

> **作者:** Jaesung Bae; Xiuwen Zheng; Minje Kim; Chang D. Yoo; Mark Hasegawa-Johnson
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Dysarthric speech quality assessment (DSQA) is critical for clinical diagnostics and inclusive speech technologies. However, subjective evaluation is costly and difficult to scale, and the scarcity of labeled data limits robust objective modeling. To address this, we propose a three-stage framework that leverages unlabeled dysarthric speech and large-scale typical speech datasets to scale training. A teacher model first generates pseudo-labels for unlabeled samples, followed by weakly supervised pretraining using a label-aware contrastive learning strategy that exposes the model to diverse speakers and acoustic conditions. The pretrained model is then fine-tuned for the downstream DSQA task. Experiments on five unseen datasets spanning multiple etiologies and languages demonstrate the robustness of our approach. Our Whisper-based baseline significantly outperforms SOTA DSQA predictors such as SpICE, and the full framework achieves an average SRCC of 0.761 across unseen test datasets.
>
---
#### [new 004] HRTF-guided Binaural Target Speaker Extraction with Real-World Validation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，旨在解决多声源中目标说话人提取的问题。通过引入HRTF作为空间先验，提升分离效果与空间感知。**

- **链接: [https://arxiv.org/pdf/2603.16668](https://arxiv.org/pdf/2603.16668)**

> **作者:** Yoav Ellinson; Sharon Gannot
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** This paper presents a Head-Related Transfer Function (HRTF)-guided framework for binaural Target Speaker Extraction (TSE) from mixtures of concurrent sources. Unlike conventional TSE methods based on Direction of Arrival (DOA) estimation or enrollment signals, which often distort perceived spatial location, the proposed approach leverages the listener's HRTF as an explicit spatial prior. The proposed framework is built upon a multi-channel deep blind source separation backbone, adapted to the binaural TSE setting. It is trained on measured HRTFs from a diverse population, enabling cross-listener generalization rather than subject-specific tuning. By conditioning the extraction on HRTF-derived spatial information, the method preserves binaural cues while enhancing speech quality and intelligibility. The performance of the proposed framework is validated through simulations and real recordings obtained from a head and torso simulator (HATS).
>
---
#### [new 005] Robust Generative Audio Quality Assessment: Disentangling Quality from Spurious Correlations
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于音频质量评估任务，旨在解决AI生成内容中因数据稀缺导致的虚假相关性问题。通过领域对抗训练，分离真实质量与噪声因素，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.16201](https://arxiv.org/pdf/2603.16201)**

> **作者:** Kuan-Tang Huang; Chien-Chun Wang; Cheng-Yeh Yang; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to IEEE ICME 2026
>
> **摘要:** The rapid proliferation of AI-Generated Content (AIGC) has necessitated robust metrics for perceptual quality assessment. However, automatic Mean Opinion Score (MOS) prediction models are often compromised by data scarcity, predisposing them to learn spurious correlations-- such as dataset-specific acoustic signatures-- rather than generalized quality features. To address this, we leverage domain adversarial training (DAT) to disentangle true quality perception from these nuisance factors. Unlike prior works that rely on static domain priors, we systematically investigate domain definition strategies ranging from explicit metadata-driven labels to implicit data-driven clusters. Our findings reveal that there is no "one-size-fits-all" domain definition; instead, the optimal strategy is highly dependent on the specific MOS aspect being evaluated. Experimental results demonstrate that our aspect-specific domain strategy effectively mitigates acoustic biases, significantly improving correlation with human ratings and achieving superior generalization on unseen generative scenarios.
>
---
#### [new 006] Making Separation-First Multi-Stream Audio Watermarking Feasible via Joint Training
- **分类: cs.SD**

- **简介: 该论文属于音频水印任务，解决多音轨独立水印与分离后恢复的问题。通过联合训练水印系统和分离器，提升分离后的水印恢复效果。**

- **链接: [https://arxiv.org/pdf/2603.16805](https://arxiv.org/pdf/2603.16805)**

> **作者:** Houmin Sun; Zi Hu; Linxi Li; Yechen Wang; Liwei Jin; Ming Li
>
> **摘要:** Modern audio is created by mixing stems from different sources, raising the question: can we independently watermark each stem and recover all watermarks after separation? We study a separation-first, multi-stream watermarking framework-embedding distinct information into stems using unique keys but a shared structure, mixing, separating, and decoding from each output. A naive pipeline (robust watermarking + off-the-shelf separation) yields poor bit recovery, showing robustness to generic distortions does not ensure robustness to separation artifacts. To enable this, we jointly train the watermark system and the separator in an end-to-end manner, encouraging the separator to preserve watermark cues while adapting embedding to separation-specific distortions. Experiments on speech+music and vocal+accompaniment mixtures show substantial gains in post-separation recovery while maintaining perceptual quality.
>
---
#### [new 007] A Semantic Timbre Dataset for the Electric Guitar
- **分类: cs.SD**

- **简介: 该论文提出一个包含19个语义描述符的电吉他音色数据集，解决音色与语义关联不足的问题，支持音色控制和生成任务。**

- **链接: [https://arxiv.org/pdf/2603.16682](https://arxiv.org/pdf/2603.16682)**

> **作者:** Joseph Cameron; Alan Blackwell
>
> **备注:** 5 pages, 7 figures, 2 tables
>
> **摘要:** Understanding and manipulating timbre is central to audio synthesis, yet this remains under-explored in machine learning due to a lack of annotated datasets linking perceptual timbre dimensions to semantic descriptors. We present the Semantic Timbre Dataset, a curated collection of monophonic electric guitar sounds, each labeled with one of 19 semantic timbre descriptors and corresponding magnitudes. These descriptors were derived from a qualitative analysis of physical and virtual guitar effect units and applied systematically to clean guitar tones. The dataset bridges perceptual timbre and machine learning representations, supporting learning for timbre control and semantic audio generation. We validate the dataset by training a variational autoencoder (VAE) on its latent space and evaluating it using human perceptual judgments and descriptor classifiers. Results show that the VAE captures timbral structure and enables smooth interpolation across descriptors. We release the dataset, code, and evaluation protocols to support timbre-aware generative AI research.
>
---
#### [new 008] CAST-TTS: A Simple Cross-Attention Framework for Unified Timbre Control in TTS
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在解决语音和文本音色控制统一的问题。提出CAST-TTS框架，通过跨注意力机制实现统一音色控制。**

- **链接: [https://arxiv.org/pdf/2603.16280](https://arxiv.org/pdf/2603.16280)**

> **作者:** Zihao Zheng; Wen Wu; Chao Zhang; Mengyue Wu; Xuenan Xu
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Current Text-to-Speech (TTS) systems typically use separate models for speech-prompted and text-prompted timbre control. While unifying both control signals into a single model is desirable, the challenge of cross-modal alignment often results in overly complex architectures and training objective. To address this challenge, we propose CAST-TTS, a simple yet effective framework for unified timbre control. Features are extracted from speech prompts and text prompts using pre-trained encoders. The multi-stage training strategy efficiently aligns the speech and projected text representations within a shared embedding space. A single cross-attention mechanism then allows the model to use either of these representations to control the timbre. Extensive experiments validate that the unified cross-attention mechanism is critical for achieving high-quality synthesis. CAST-TTS achieves performance comparable to specialized single-input models while operating within a unified architecture. The demo page can be accessed at this https URL.
>
---
#### [new 009] Speakers Localization Using Batch EM In Unfolding Neural Network
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音信号处理任务，旨在解决混响环境下的说话人定位问题。提出一种嵌入EM迭代的批处理网络，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.16278](https://arxiv.org/pdf/2603.16278)**

> **作者:** Rina Veler; Sharon Gannot
>
> **备注:** 3 pages, 1 figure, ICSEE 2026
>
> **摘要:** We propose an interpretable Batch-EM Unfolded Network for robust speaker localization. By embedding the iterative EM procedure within an encoder-EM-decoder architecture, the method mitigates initialization sensitivity and improves convergence. Experiments show superior accuracy and robustness over the classical Batch-EM in reverberant conditions.
>
---
#### [new 010] INSTRUMENTAL: Automatic Synthesizer Parameter Recovery from Audio via Evolutionary Optimization
- **分类: cs.SD**

- **简介: 该论文属于音频到合成器参数的逆问题，旨在恢复音频中的连续合成器参数。通过结合可微分合成器与进化优化方法，提升音色匹配效果。**

- **链接: [https://arxiv.org/pdf/2603.15905](https://arxiv.org/pdf/2603.15905)**

> **作者:** Philipp Bogdan
>
> **备注:** 5 pages
>
> **摘要:** Existing audio-to-MIDI tools extract notes but discard the timbral characteristics that define an instrument's identity. We present Instrumental, a system that recovers continuous synthesizer parameters from audio by coupling a differentiable 28-parameter subtractive synthesizer with CMA-ES, a derivative-free evolutionary optimizer. We optimize a composite perceptual loss combining mel-scaled STFT, spectral centroid, and MFCC divergence, achieving a matching loss of 2.09 on real recorded audio. We systematically evaluate eight hypotheses for improving convergence and find that only parametric EQ boosting yields meaningful improvement. Our results show that CMA-ES outperforms gradient descent on this non-convex landscape, that more parameters do not monotonically improve matching, and that spectral analysis initialization accelerates convergence over random starts.
>
---
#### [new 011] Evaluating Latent Space Structure in Timbre VAEs: A Comparative Study of Unsupervised, Descriptor-Conditioned, and Perceptual Feature-Conditioned Models
- **分类: cs.SD**

- **简介: 该论文属于音乐音色生成任务，旨在评估不同VAE模型的潜在空间结构。通过比较三种模型，分析其潜在空间的聚类与可解释性，提出更可控的音频生成方法。**

- **链接: [https://arxiv.org/pdf/2603.16713](https://arxiv.org/pdf/2603.16713)**

> **作者:** Joseph Cameron; Alan Blackwell
>
> **备注:** 5 pages, 1 figure, 1 table
>
> **摘要:** We present a comparative evaluation of latent space organization in three Variational Autoencoders (VAEs) for musical timbre generation: an unsupervised VAE, a descriptor-conditioned VAE, and a VAE conditioned on continuous perceptual features from the AudioCommons timbral models. Using a curated dataset of electric guitar sounds labeled with 19 semantic descriptors across four intensity levels, we assess each model's latent structure with a suite of clustering and interpretability metrics. These include silhouette scores, timbre descriptor compactness, pitch-conditional separation, trajectory linearity, and cross-pitch consistency. Our findings show that conditioning on perceptual features yields a more compact, discriminative, and pitch-invariant latent space, outperforming both the unsupervised and discrete descriptor-conditioned models. This work highlights the limitations of one-hot semantic conditioning and provides methodological tools for evaluating timbre latent spaces, contributing to the development of more controllable and interpretable generative audio models.
>
---
#### [new 012] PulmoVec: A Two-Stage Stacking Meta-Learning Architecture Built on the HeAR Foundation Model for Multi-Task Classification of Pediatric Respiratory Sounds
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出PulmoVec，用于儿童呼吸音的多任务分类，解决听诊主观性及数据量小问题，通过集成学习提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2603.15688](https://arxiv.org/pdf/2603.15688)**

> **作者:** Izzet Turkalp Akbasli; Oguzhan Serin
>
> **备注:** 14 pages, 2 figures, 4 tables; supplementary material included (4 tables, 3 multi-panel figures)
>
> **摘要:** Background: Respiratory diseases are a leading cause of childhood morbidity and mortality, yet lung auscultation remains subjective and limited by inter-listener variability, particularly in pediatric populations. Existing AI approaches are further constrained by small datasets and single-task designs. We developed PulmoVec, a multi-task framework built on the Health Acoustic Representations (HeAR) foundation model for classification of pediatric respiratory sounds. Methods: In this retrospective analysis of the SPRSound database, 24,808 event-level annotated segments from 1,652 pediatric patients were analyzed. Three task-specific classifiers were trained for screening, sound-pattern recognition, and disease-group prediction. Their out-of-fold probability outputs were combined with demographic metadata in a LightGBM stacking meta-model, and event-level predictions were aggregated to the patient level using ensemble voting. Results: At the event level, the screening model achieved an ROC-AUC of 0.96 (95% CI, 0.95-0.97), the sound-pattern recognition model a macro ROC-AUC of 0.96 (95% CI, 0.96-0.97), and the disease-group prediction model a macro ROC-AUC of 0.94 (95% CI, 0.93-0.94). At the patient level, disease-group classification yielded an accuracy of 0.74 (95% CI, 0.71-0.77), a weighted F1-score of 0.73, and a macro ROC-AUC of 0.91 (95% CI, 0.90-0.93). Stacking improved performance across all tasks compared with base models alone. Conclusions: PulmoVec links event-level acoustic phenotyping with patient-level clinical classification, supporting the potential of foundation-model-based digital auscultation in pediatric respiratory medicine. Multi-center external validation across devices and real-world conditions remains essential.
>
---
#### [new 013] Towards the Vision-Sound-Language-Action Paradigm: The HEAR Framework for Sound-Centric Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.SD**

- **简介: 该论文提出HEAR框架，解决实时声音驱动的交互任务，通过整合视觉、声音、语言和本体感知，实现连续音频感知与动作生成。**

- **链接: [https://arxiv.org/pdf/2603.16086](https://arxiv.org/pdf/2603.16086)**

> **作者:** Chang Nie; Tianchen Deng; Guangming Wang; Zhe Liu; Hesheng Wang
>
> **摘要:** While recent Vision-Language-Action (VLA) models have begun to incorporate audio, they typically treat sound as static pre-execution prompts or focus exclusively on human speech. This leaves a significant gap in real-time, sound-centric manipulation where fleeting environmental acoustics provide critical state verification during task execution. Consequently, key sounds are easily missed due to low-frequency updates or system latency. This problem is exacerbated by action chunking with open-loop execution, which creates a Blind Execution Interval where acoustic events are lost between discrete audio observation windows. Recognizing the necessity of continuous auditory awareness, we formalize Vision-Sound-Language-Action (VSLA) as a continuous control paradigm conditioned on vision, streaming audio, language, and proprioception under delayed decision loops. As an instantiation, we introduce HEAR, a VSLA framework integrating four components: (i) a streaming Historizer to maintain a compact, causal audio context across execution gaps; (ii) an Envisioner adapted from omni foundation models to reason over multi-sensory inputs; (iii) an Advancer, formulated as an audio world model, to learn temporal dynamics by predicting near-future audio codes; and (iv) a flow-matching Realizer policy to generate smooth action chunks. To address the scarcity of pretraining data and evaluations for VSLA, we construct OpenX-Sound for pretraining, alongside HEAR-Bench, the first sound-centric manipulation benchmark with strict causal timing rules. Our results suggest that robust sound-centric manipulation necessitates causal persistence and explicit temporal learning. This framework provides a practical step toward multi-sensory foundation models for embodied agents, enabling robots to perceive and interact with dynamic environments. Code and videos are available at this https URL.
>
---
#### [new 014] RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别中的实体纠错任务，解决罕见和专业术语识别错误问题。通过构建RECOVER框架，利用多种策略提升实体识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.16411](https://arxiv.org/pdf/2603.16411)**

> **作者:** Abhishek Kumar; Aashraya Sachdeva
>
> **备注:** Under review. Submitted to Interspeech 2026
>
> **摘要:** Entity recognition in Automatic Speech Recognition (ASR) is challenging for rare and domain-specific terms. In domains such as finance, medicine, and air traffic control, these errors are costly. If the entities are entirely absent from the ASR output, post-ASR correction becomes difficult. To address this, we introduce RECOVER, an agentic correction framework that serves as a tool-using agent. It leverages multiple hypotheses as evidence from ASR, retrieves relevant entities, and applies Large Language Model (LLM) correction under constraints. The hypotheses are used using different strategies, namely, 1-Best, Entity-Aware Select, Recognizer Output Voting Error Reduction (ROVER) Ensemble, and LLM-Select. Evaluated across five diverse datasets, it achieves 8-46% relative reductions in entity-phrase word error rate (E-WER) and increases recall by up to 22 percentage points. The LLM-Select achieves the best overall performance in entity correction while maintaining overall WER.
>
---
#### [new 015] DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression
- **分类: cs.MM; cs.AI; cs.CV; cs.SD**

- **简介: 该论文提出DASH框架，解决多模态序列压缩问题，通过动态语义分块提升压缩效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.15685](https://arxiv.org/pdf/2603.15685)**

> **作者:** Bingzhou Li; Tao Huang
>
> **摘要:** Omnimodal large language models (OmniLLMs) jointly process audio and visual streams, but the resulting long multimodal token sequences make inference prohibitively expensive. Existing compression methods typically rely on fixed window partitioning and attention-based pruning, which overlook the piecewise semantic structure of audio-visual signals and become fragile under aggressive token reduction. We propose Dynamic Audio-driven Semantic cHunking (DASH), a training-free framework that aligns token compression with semantic structure. DASH treats audio embeddings as a semantic anchor and detects boundary candidates via cosine-similarity discontinuities, inducing dynamic, variable-length segments that approximate the underlying piecewise-coherent organization of the sequence. These boundaries are projected onto video tokens to establish explicit cross-modal segmentation. Within each segment, token retention is determined by a tri-signal importance estimator that fuses structural boundary cues, representational distinctiveness, and attention-based salience, mitigating the sparsity bias of attention-only selection. This structure-aware allocation preserves transition-critical tokens while reducing redundant regions. Extensive experiments on AVUT, VideoMME, and WorldSense demonstrate that DASH maintains superior accuracy while achieving higher compression ratios compared to prior methods. Code is available at: this https URL.
>
---
## 更新

#### [replaced 001] Coherent Audio-Visual Editing via Conditional Audio Generation Following Video Edits
- **分类: cs.MM; cs.LG; cs.SD**

- **简介: 该论文属于音频视频协同编辑任务，旨在解决编辑后音视频不一致的问题。通过条件音频生成模型，使音频与视频修改保持同步，提升内容一致性。**

- **链接: [https://arxiv.org/pdf/2512.07209](https://arxiv.org/pdf/2512.07209)**

> **作者:** Masato Ishii; Akio Hayakawa; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Source code: this https URL
>
> **摘要:** We introduce a novel pipeline for joint audio-visual editing that enhances the coherence between edited video and its accompanying audio. Our approach first applies state-of-the-art video editing techniques to produce the target video, then performs audio editing to align with the visual changes. To achieve this, we present a new video-to-audio generation model that conditions on the source audio, target video, and a text prompt. We extend the model architecture to incorporate conditional audio input and propose a data augmentation strategy that improves training efficiency. Furthermore, our model dynamically adjusts the influence of the source audio based on the complexity of the edits, preserving the original audio structure where possible. Experimental results demonstrate that our method outperforms existing approaches in maintaining audio-visual alignment and content integrity.
>
---
#### [replaced 002] Mathematical Foundations of Polyphonic Music Generation via Structural Inductive Bias
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，解决多声部音乐生成中的“缺失中间”问题，通过结构归纳偏置和数学理论验证，提升模型稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03612](https://arxiv.org/pdf/2601.03612)**

> **作者:** Joonwon Seo
>
> **备注:** 81 pages. A comprehensive monograph detailing the Smart Embedding architecture for polyphonic music generation, including theoretical proofs (Information Theory, Rademacher Complexity, RPTP) and human evaluation results
>
> **摘要:** This monograph introduces a novel approach to polyphonic music generation by addressing the "Missing Middle" problem through structural inductive bias. Focusing on Beethoven's piano sonatas as a case study, we empirically verify the independence of pitch and hand attributes using normalized mutual information (NMI=0.167) and propose the Smart Embedding architecture, achieving a 48.30% reduction in parameters. We provide rigorous mathematical proofs using information theory (negligible loss bounded at 0.153 bits), Rademacher complexity (28.09% tighter generalization bound), and category theory to demonstrate improved stability and generalization. Empirical results show a 9.47% reduction in validation loss, confirmed by SVD analysis and an expert listening study (N=53). This dual theoretical and applied framework bridges gaps in AI music generation, offering verifiable insights for mathematically grounded deep learning.
>
---
#### [replaced 003] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于视频配音任务，解决语音与唇部动作同步及语音质量不足的问题。提出DiFlowDubber模型，通过跨模态对齐和离散流匹配实现高质量自动配音。**

- **链接: [https://arxiv.org/pdf/2603.14267](https://arxiv.org/pdf/2603.14267)**

> **作者:** Ngoc-Son Nguyen; Thanh V. T. Tran; Jeongsoo Choi; Hieu-Nghia Huynh-Nguyen; Truong-Son Hy; Van Nguyen
>
> **备注:** Accepted at CVPR 2026 Findings
>
> **摘要:** Video dubbing has broad applications in filmmaking, multimedia creation, and assistive speech technology. Existing approaches either train directly on limited dubbing datasets or adopt a two-stage pipeline that adapts pre-trained text-to-speech (TTS) models, which often struggle to produce expressive prosody, rich acoustic characteristics, and precise synchronization. To address these issues, we propose DiFlowDubber with a novel two-stage training framework that effectively transfers knowledge from a pre-trained TTS model to video-driven dubbing, with a discrete flow matching generative backbone. Specifically, we design a FaPro module that captures global prosody and stylistic cues from facial expressions and leverages this information to guide the modeling of subsequent speech attributes. To ensure precise speech-lip synchronization, we introduce a Synchronizer module that bridges the modality gap among text, video, and speech, thereby improving cross-modal alignment and generating speech that is temporally synchronized with lip movements. Experiments on two primary benchmark datasets demonstrate that DiFlowDubber outperforms previous methods across multiple metrics.
>
---
#### [replaced 004] Code-switching Speech Recognition Under the Lens: Model- and Data-Centric Perspectives
- **分类: eess.AS**

- **简介: 该论文属于代码转换语音识别任务，解决语言混淆和数据稀缺问题。通过模型和数据双角度分析，提出SECT生成高质量代码转换文本，提升ASR性能。**

- **链接: [https://arxiv.org/pdf/2509.24310](https://arxiv.org/pdf/2509.24310)**

> **作者:** Hexin Liu; Haoyang Zhang; Qiquan Zhang; Xiangyu Zhang; Dongyuan Shi; Eng Siong Chng; Haizhou Li
>
> **备注:** 14 pages, 4 figures, 10 tables, accepted to IEEE TASLP. Copyright has been transferred to IEEE
>
> **摘要:** Code-switching automatic speech recognition (CS-ASR) presents unique challenges due to language confusion introduced by spontaneous intra-sentence switching and accent bias that blurs the phonetic boundaries. Although the constituent languages may be individually high-resource, the scarcity of annotated code-switching data further compounds these challenges. In this paper, we systematically analyze CS-ASR from both model-centric and data-centric perspectives. By comparing state-of-the-art algorithmic methods, including language-specific processing and auxiliary language-aware multi-task learning, we discuss their varying effectiveness across datasets with different linguistic characteristics. On the data side, we first investigate TTS as a data augmentation method. By varying the textual characteristics and speaker accents, we analyze the impact of language confusion and accent bias on CS-ASR. To further mitigate data scarcity and enhance textual diversity, we propose a prompting strategy by simplifying the equivalence constraint theory (SECT) to guide large language models (LLMs) in generating linguistically valid code-switching text. The proposed SECT outperforms existing methods in ASR performance and linguistic quality assessments, generating code-switching text that more closely resembles real-world code-switching text. When used to generate speech-text pairs via TTS, SECT proves effective in improving CS-ASR performance. Our analysis of both model- and data-centric methods underscores that effective CS-ASR requires strategies to be carefully aligned with the specific linguistic characteristics of the code-switching data.
>
---
#### [replaced 005] Time-Layer Adaptive Alignment for Speaker Similarity in Flow-Matching Based Zero-Shot TTS
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，解决零样本TTS中说话人相似性不足的问题。提出时间-层自适应对齐方法，提升说话人一致性。**

- **链接: [https://arxiv.org/pdf/2511.09995](https://arxiv.org/pdf/2511.09995)**

> **作者:** Haoyu Li; Mingyang Han; Yu Xi; Dongxiao Wang; Hankun Wang; Haoxiang Shi; Boyu Li; Jun Song; Bo Zheng; Shuai Wang; Kai Yu
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Flow-Matching (FM)-based zero-shot text-to-speech (TTS) systems exhibit high-quality speech synthesis and robust generalization capabilities. However, the speaker representation ability of such systems remains underexplored, primarily due to the lack of explicit speaker-specific supervision in the FM framework. To this end, we conduct an empirical analysis of speaker information distribution and reveal its non-uniform allocation across time steps and network layers, underscoring the need for adaptive speaker alignment. Accordingly, we propose Time-Layer Adaptive Speaker Alignment (TLA-SA), a strategy that enhances speaker consistency by jointly leveraging temporal and hierarchical variations. Experimental results show that TLA-SA substantially improves speaker similarity over baseline systems on both research- and industrial-scale datasets and generalizes well across diverse model architectures, including decoder-only language model (LM)-based and free TTS systems. A demo is provided.
>
---
#### [replaced 006] LLM-Guided Reinforcement Learning for Audio-Visual Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频-视觉语音增强任务，旨在解决传统评估指标与感知质量关联弱的问题。通过引入基于大语言模型的强化学习框架，提升语音增强效果。**

- **链接: [https://arxiv.org/pdf/2603.13952](https://arxiv.org/pdf/2603.13952)**

> **作者:** Chih-Ning Chen; Jen-Cheng Hou; Hsin-Min Wang; Shao-Yi Chien; Yu Tsao; Fan-Gang Zeng
>
> **备注:** 6 pages, 4 figures, submitted to Interspeech 2026
>
> **摘要:** In existing Audio-Visual Speech Enhancement (AVSE) methods, objectives such as Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Mean Squared Error (MSE) are widely used; however, they often correlate poorly with perceptual quality and provide limited interpretability for optimization. This work proposes a reinforcement learning-based AVSE framework with a Large Language Model (LLM)-based interpretable reward model. An audio LLM generates natural language descriptions of enhanced speech, which are converted by a sentiment analysis model into a 1-5 rating score serving as the PPO reward for fine-tuning a pretrained AVSE model. Compared with scalar metrics, LLM-generated feedback is semantically rich and explicitly describes improvements in speech quality. Experiments on the 4th COG-MHEAR AVSE Challenge (AVSEC-4) dataset show that the proposed method outperforms a supervised baseline and a DNSMOS-based RL baseline in PESQ, STOI, neural quality metrics, and subjective listening tests.
>
---
#### [replaced 007] Building Enterprise Realtime Voice Agents from Scratch: A Technical Tutorial
- **分类: cs.SD**

- **简介: 本文介绍如何从零构建企业级实时语音代理系统，解决自托管实时语音代理的架构问题。通过集成STT、LLM和TTS组件，实现低延迟的语音交互。**

- **链接: [https://arxiv.org/pdf/2603.05413](https://arxiv.org/pdf/2603.05413)**

> **作者:** Jielin Qiu; Zixiang Chen; Liangwei Yang; Ming Zhu; Zhiwei Liu; Juntao Tan; Wenting Zhao; Rithesh Murthy; Roshan Ram; Akshara Prabhakar; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** We present a technical tutorial for building enterprise-grade realtime voice agents from first principles. While end-to-end speech-to-speech models may ultimately provide the best latency for voice agents, fully self-hosted end-to-end solutions are not yet available. We evaluate the closest candidate, Qwen3-Omni, across three configurations: its cloud-only DashScope Realtime API achieves $\sim$702ms audio-to-audio latency with streaming, but is not self-hostable; its local vLLM deployment supports only the Thinker (text generation from audio, 516ms), not the Talker (audio synthesis); and its local Transformers deployment runs the full pipeline but at $\sim$146s -- far too slow for realtime. The cascaded streaming pipeline (STT $\rightarrow$ LLM $\rightarrow$ TTS) therefore remains the practical architecture for self-hosted realtime voice agents, and the focus of this tutorial. We build a complete voice agent using Deepgram (streaming STT), vLLM-served LLMs with function calling (streaming text generation), and ElevenLabs (streaming TTS), achieving a measured time-to-first-audio of 755ms (best case 729ms) with full function calling support. We release the full codebase as a 9-chapter progressive tutorial with working, tested code for every component.
>
---
#### [replaced 008] Revisiting ASR Error Correction with Specialized Models
- **分类: cs.LG; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别错误修正任务，旨在解决传统方法对ASR错误模式不敏感及大模型引入延迟和幻觉的问题。通过构建合成数据并使用轻量seq2seq模型实现高效准确的纠错。**

- **链接: [https://arxiv.org/pdf/2405.15216](https://arxiv.org/pdf/2405.15216)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; He Bai; Erik McDermott; Ronan Collobert; Navdeep Jaitly
>
> **备注:** under review
>
> **摘要:** Language models play a central role in automatic speech recognition (ASR), yet most methods rely on text-only models unaware of ASR error patterns. Recently, large language models (LLMs) have been applied to ASR correction, but introduce latency and hallucination concerns. We revisit ASR error correction with compact seq2seq models, trained on ASR errors from real and synthetic audio. To scale training, we construct synthetic corpora via cascaded TTS and ASR, finding that matching the diversity of realistic error distributions is key. We propose correction-first decoding, where the correction model generates candidates rescored using ASR acoustic scores. With 15x fewer parameters than LLMs, our model achieves 1.5/3.3% WER on LibriSpeech test-clean/other, outperforms LLMs, generalizes across ASR architectures (CTC, Seq2seq, Transducer) and diverse domains, and provides precise corrections in the low-error regime where LLMs struggle.
>
---
#### [replaced 009] When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频干扰对文本推理的影响，属于多模态模型任务。解决音频噪声影响文本推理的问题，通过实验分析不同音频类型的影响，并测试缓解策略。**

- **链接: [https://arxiv.org/pdf/2510.00626](https://arxiv.org/pdf/2510.00626)**

> **作者:** Chen-An Li; Tzu-Han Lin; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) unify speech and text processing, but their robustness in noisy real-world settings remains underexplored. We investigate how irrelevant audio, such as silence, synthetic noise, and environmental sounds, affects text reasoning tasks where audio is unnecessary. Across three text-based benchmarks, we find that even non-informative audio reduces accuracy and increases prediction volatility; the severity of interference scales with longer durations, higher amplitudes, and elevated decoding temperatures. Silence, often assumed neutral, destabilizes outputs as strongly as synthetic noise. While larger models show greater resilience, vulnerabilities persist across all evaluated systems. We further test mitigation strategies and find that prompting shows limited effectiveness, whereas self-consistency improves stability at the cost of increased computation. Our results reveal cross-modal interference as a key robustness challenge and highlight the need for efficient fusion strategies that preserve reasoning performance in the presence of irrelevant inputs.
>
---
#### [replaced 010] VorTEX: Various overlap ratio for Target speech EXtraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音分离任务，解决真实场景下不同重叠比例的语音提取问题。提出VorTEX模型和PORTE数据集，提升分离效果并减少干扰。**

- **链接: [https://arxiv.org/pdf/2603.14803](https://arxiv.org/pdf/2603.14803)**

> **作者:** Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** arXiv Preprint
>
> **摘要:** Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts.
>
---
