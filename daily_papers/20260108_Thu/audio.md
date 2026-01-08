# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Investigation into respiratory sound classification for an imbalanced data set using hybrid LSTM-KAN architectures
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于呼吸音分类任务，旨在解决数据不平衡问题。通过融合LSTM和KAN模型，并结合数据增强等技术，提升少数类识别效果。**

- **链接: [https://arxiv.org/pdf/2601.03610v1](https://arxiv.org/pdf/2601.03610v1)**

> **作者:** Nithinkumar K.; Anand R
>
> **摘要:** Respiratory sounds captured via auscultation contain critical clues for diagnosing pulmonary conditions. Automated classification of these sounds faces challenges due to subtle acoustic differences and severe class imbalance in clinical datasets. This study investigates respiratory sound classification with a focus on mitigating pronounced class imbalance. We propose a hybrid deep learning model that combines a Long Short-Term Memory (LSTM) network for sequential feature encoding with a Kolmogorov-Arnold Network (KAN) for classification. The model is integrated with a comprehensive feature extraction pipeline and targeted imbalance mitigation strategies. Experiments were conducted on a public respiratory sound database comprising six classes with a highly skewed distribution. Techniques such as focal loss, class-specific data augmentation, and Synthetic Minority Over-sampling Technique (SMOTE) were employed to enhance minority class recognition. The proposed Hybrid LSTM-KAN model achieves an overall accuracy of 94.6 percent and a macro-averaged F1 score of 0.703, despite the dominant COPD class accounting for over 86 percent of the data. Improved detection performance is observed for minority classes compared to baseline approaches, demonstrating the effectiveness of the proposed architecture for imbalanced respiratory sound classification.
>
---
#### [new 002] Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于可控长歌词生成任务，旨在解决学术研究不可复现的问题。工作包括发布开源系统Muse及合成数据集，实现细粒度风格控制的歌曲生成。**

- **链接: [https://arxiv.org/pdf/2601.03973v1](https://arxiv.org/pdf/2601.03973v1)**

> **作者:** Changhao Jiang; Jiahao Chen; Zhenghao Xiang; Zhixiong Yang; Hanchen Wang; Jiabao Zhuang; Xinmeng Che; Jiajun Sun; Hui Li; Yifei Cao; Shihan Dou; Ming Zhang; Junjie Ye; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recent commercial systems such as Suno demonstrate strong capabilities in long-form song generation, while academic research remains largely non-reproducible due to the lack of publicly available training data, hindering fair comparison and progress. To this end, we release a fully open-source system for long-form song generation with fine-grained style conditioning, including a licensed synthetic dataset, training and evaluation pipelines, and Muse, an easy-to-deploy song generation model. The dataset consists of 116k fully licensed synthetic songs with automatically generated lyrics and style descriptions paired with audio synthesized by SunoV5. We train Muse via single-stage supervised finetuning of a Qwen-based language model extended with discrete audio tokens using MuCodec, without task-specific losses, auxiliary objectives, or additional architectural components. Our evaluations find that although Muse is trained with a modest data scale and model size, it achieves competitive performance on phoneme error rate, text--music style similarity, and audio aesthetic quality, while enabling controllable segment-level generation across different musical structures. All data, model weights, and training and evaluation pipelines will be publicly released, paving the way for continued progress in controllable long-form song generation research. The project repository is available at https://github.com/yuhui1038/Muse.
>
---
#### [new 003] Domain Adaptation of the Pyannote Diarization Pipeline for Conversational Indonesian Audio
- **分类: cs.SD**

- **简介: 该论文属于说话人分离任务，旨在解决将英语语音分离模型适应到印尼语口语音频的问题。通过生成合成数据进行域适应，提升模型在低资源语言上的性能。**

- **链接: [https://arxiv.org/pdf/2601.03684v1](https://arxiv.org/pdf/2601.03684v1)**

> **作者:** Muhammad Daffa'i Rafi Prasetyo; Ramadhan Andika Putra; Zaidan Naufal Ilmi; Kurniawati Azizah
>
> **备注:** Experiments conducted using synthetic Indonesian conversational speech for domain adaptation
>
> **摘要:** This study presents a domain adaptation approach for speaker diarization targeting conversational Indonesian audio. We address the challenge of adapting an English-centric diarization pipeline to a low-resource language by employing synthetic data generation using neural Text-to-Speech technology. Experiments were conducted with varying training configurations, a small dataset (171 samples) and a large dataset containing 25 hours of synthetic speech. Results demonstrate that the baseline \texttt{pyannote/segmentation-3.0} model, trained on the AMI Corpus, achieves a Diarization Error Rate (DER) of 53.47\% when applied zero-shot to Indonesian. Domain adaptation significantly improves performance, with the small dataset models reducing DER to 34.31\% (1 epoch) and 34.81\% (2 epochs). The model trained on the 25-hour dataset achieves the best performance with a DER of 29.24\%, representing a 13.68\% absolute improvement over the baseline while maintaining 99.06\% Recall and 87.14\% F1-Score.
>
---
#### [new 004] Learning from Limited Labels: Transductive Graph Label Propagation for Indian Music Analysis
- **分类: eess.AS; cs.LG**

- **简介: 该论文研究在标签有限情况下，利用图标签传播技术进行印度音乐分析，解决数据标注成本高、质量低的问题，通过构建相似性图实现半监督学习。**

- **链接: [https://arxiv.org/pdf/2601.03626v1](https://arxiv.org/pdf/2601.03626v1)**

> **作者:** Parampreet Singh; Akshay Raina; Sayeedul Islam Sheikh; Vipul Arora
>
> **备注:** Published at Journal of Acoustical Society of India, 2025
>
> **摘要:** Supervised machine learning frameworks rely on extensive labeled datasets for robust performance on real-world tasks. However, there is a lack of large annotated datasets in audio and music domains, as annotating such recordings is resource-intensive, laborious, and often require expert domain knowledge. In this work, we explore the use of label propagation (LP), a graph-based semi-supervised learning technique, for automatically labeling the unlabeled set in an unsupervised manner. By constructing a similarity graph over audio embeddings, we propagate limited label information from a small annotated subset to a larger unlabeled corpus in a transductive, semi-supervised setting. We apply this method to two tasks in Indian Art Music (IAM): Raga identification and Instrument classification. For both these tasks, we integrate multiple public datasets along with additional recordings we acquire from Prasar Bharati Archives to perform LP. Our experiments demonstrate that LP significantly reduces labeling overhead and produces higher-quality annotations compared to conventional baseline methods, including those based on pretrained inductive models. These results highlight the potential of graph-based semi-supervised learning to democratize data annotation and accelerate progress in music information retrieval.
>
---
#### [new 005] Lightweight and perceptually-guided voice conversion for electro-laryngeal speech
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音转换任务，旨在提升电声喉语音的自然度和可懂度。通过轻量级模型优化，结合感知损失，显著降低错误率并提高评分。**

- **链接: [https://arxiv.org/pdf/2601.03892v1](https://arxiv.org/pdf/2601.03892v1)**

> **作者:** Benedikt Mayrhofer; Franz Pernkopf; Philipp Aichinger; Martin Hagmüller
>
> **备注:** 5 pages, 5 figures. Audio samples available at https://spsc-tugraz.github.io/lw-elvc-icassp26/ Preprint submitted to ICASSP
>
> **摘要:** Electro-laryngeal (EL) speech is characterized by constant pitch, limited prosody, and mechanical noise, reducing naturalness and intelligibility. We propose a lightweight adaptation of the state-of-the-art StreamVC framework to this setting by removing pitch and energy modules and combining self-supervised pretraining with supervised fine-tuning on parallel EL and healthy (HE) speech data, guided by perceptual and intelligibility losses. Objective and subjective evaluations across different loss configurations confirm their influence: the best model variant, based on WavLM features and human-feedback predictions (+WavLM+HF), drastically reduces character error rate (CER) of EL inputs, raises naturalness mean opinion score (nMOS) from 1.1 to 3.3, and consistently narrows the gap to HE ground-truth speech in all evaluated metrics. These findings demonstrate the feasibility of adapting lightweight voice conversion architectures to EL voice rehabilitation while also identifying prosody generation and intelligibility improvements as the main remaining bottlenecks.
>
---
#### [new 006] IndexTTS 2.5 Technical Report
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到语音合成任务，解决零样本多语言情感语音生成问题，通过优化模型结构、提升速度与质量，实现更广泛的语言支持和更自然的语音合成。**

- **链接: [https://arxiv.org/pdf/2601.03888v1](https://arxiv.org/pdf/2601.03888v1)**

> **作者:** Yunpei Li; Xun Zhou; Jinchao Wang; Lu Wang; Yong Wu; Siyi Zhou; Yiquan Zhou; Jingchen Shu
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** In prior work, we introduced IndexTTS 2, a zero-shot neural text-to-speech foundation model comprising two core components: a transformer-based Text-to-Semantic (T2S) module and a non-autoregressive Semantic-to-Mel (S2M) module, which together enable faithful emotion replication and establish the first autoregressive duration-controllable generative paradigm. Building upon this, we present IndexTTS 2.5, which significantly enhances multilingual coverage, inference speed, and overall synthesis quality through four key improvements: 1) Semantic Codec Compression: we reduce the semantic codec frame rate from 50 Hz to 25 Hz, halving sequence length and substantially lowering both training and inference costs; 2) Architectural Upgrade: we replace the U-DiT-based backbone of the S2M module with a more efficient Zipformer-based modeling architecture, achieving notable parameter reduction and faster mel-spectrogram generation; 3) Multilingual Extension: We propose three explicit cross-lingual modeling strategies, boundary-aware alignment, token-level concatenation, and instruction-guided generation, establishing practical design principles for zero-shot multilingual emotional TTS that supports Chinese, English, Japanese, and Spanish, and enables robust emotion transfer even without target-language emotional training data; 4) Reinforcement Learning Optimization: we apply GRPO in post-training of the T2S module, improving pronunciation accuracy and natrualness. Experiments show that IndexTTS 2.5 not only supports broader language coverage but also replicates emotional prosody in unseen languages under the same zero-shot setting. IndexTTS 2.5 achieves a 2.28 times improvement in RTF while maintaining comparable WER and speaker similarity to IndexTTS 2.
>
---
#### [new 007] TellWhisper: Tell Whisper Who Speaks When
- **分类: eess.AS**

- **简介: 该论文属于多说话人语音识别任务，解决快速轮换和重叠语音下的身份与时间建模问题。提出TellWhisper框架，联合建模说话人身份与时间信息，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2601.03712v1](https://arxiv.org/pdf/2601.03712v1)**

> **作者:** Yifan Hu; Peiji Yang; Zhisheng Wang; Yicheng Zhong; Rui Liu
>
> **备注:** 14 pages, 6 figures, 8 tables, submitted to ACL 2026
>
> **摘要:** Multi-speaker automatic speech recognition (MASR) aims to predict ''who spoke when and what'' from multi-speaker speech, a key technology for multi-party dialogue understanding. However, most existing approaches decouple temporal modeling and speaker modeling when addressing ''when'' and ''who'': some inject speaker cues before encoding (e.g., speaker masking), which can cause irreversible information loss; others fuse identity by mixing speaker posteriors after encoding, which may entangle acoustic content with speaker identity. This separation is brittle under rapid turn-taking and overlapping speech, often leading to degraded performance. To address these limitations, we propose TellWhisper, a unified framework that jointly models speaker identity and temporal within the speech encoder. Specifically, we design TS-RoPE, a time-speaker rotary positional encoding: time coordinates are derived from frame indices, while speaker coordinates are derived from speaker activity and pause cues. By applying region-specific rotation angles, the model explicitly captures per-speaker continuity, speaker-turn transitions, and state dynamics, enabling the attention mechanism to simultaneously attend to ''when'' and ''who''. Moreover, to estimate frame-level speaker activity, we develop Hyper-SD, which casts speaker classification in hyperbolic space to enhance inter-class separation and refine speaker-activity estimates. Extensive experiments demonstrate the effectiveness of the proposed approach.
>
---
#### [new 008] Sound Event Detection with Boundary-Aware Optimization and Inference
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声事件检测任务，旨在提升时间事件检测的精度。通过建模事件起止边界，提出新的方法以减少后处理需求并提高性能。**

- **链接: [https://arxiv.org/pdf/2601.04178v1](https://arxiv.org/pdf/2601.04178v1)**

> **作者:** Florian Schmid; Chi Ian Tang; Sanjeel Parekh; Vamsi Krishna Ithapu; Juan Azcarreta Ortiz; Giacomo Ferroni; Yijun Qian; Arnoldas Jasonas; Cosmin Frateanu; Camilla Clark; Gerhard Widmer; Çağdaş Bilen
>
> **备注:** Submitted to IEEE Signal Processing Letters
>
> **摘要:** Temporal detection problems appear in many fields including time-series estimation, activity recognition and sound event detection (SED). In this work, we propose a new approach to temporal event modeling by explicitly modeling event onsets and offsets, and by introducing boundary-aware optimization and inference strategies that substantially enhance temporal event detection. The presented methodology incorporates new temporal modeling layers - Recurrent Event Detection (RED) and Event Proposal Network (EPN) - which, together with tailored loss functions, enable more effective and precise temporal event detection. We evaluate the proposed method in the SED domain using a subset of the temporally-strongly annotated portion of AudioSet. Experimental results show that our approach not only outperforms traditional frame-wise SED models with state-of-the-art post-processing, but also removes the need for post-processing hyperparameter tuning, and scales to achieve new state-of-the-art performance across all AudioSet Strong classes.
>
---
#### [new 009] Discriminating real and synthetic super-resolved audio samples using embedding-based classifiers
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于音频超分辨率任务，旨在解决合成与真实音频分布差异的问题。通过嵌入空间分类器分析音频样本的可分性，揭示了生成音频在感知质量高但分布不匹配的现象。**

- **链接: [https://arxiv.org/pdf/2601.03443v1](https://arxiv.org/pdf/2601.03443v1)**

> **作者:** Mikhail Silaev; Konstantinos Drossos; Tuomas Virtanen
>
> **备注:** Accepted for publication in Workshop Proceedingsof the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing
>
> **摘要:** Generative adversarial networks (GANs) and diffusion models have recently achieved state-of-the-art performance in audio super-resolution (ADSR), producing perceptually convincing wideband audio from narrowband inputs. However, existing evaluations primarily rely on signal-level or perceptual metrics, leaving open the question of how closely the distributions of synthetic super-resolved and real wideband audio match. Here we address this problem by analyzing the separability of real and super-resolved audio in various embedding spaces. We consider both middle-band ($4\to 16$~kHz) and full-band ($16\to 48$~kHz) upsampling tasks for speech and music, training linear classifiers to distinguish real from synthetic samples based on multiple types of audio embeddings. Comparisons with objective metrics and subjective listening tests reveal that embedding-based classifiers achieve near-perfect separation, even when the generated audio attains high perceptual quality and state-of-the-art metric scores. This behavior is consistent across datasets and models, including recent diffusion-based approaches, highlighting a persistent gap between perceptual quality and true distributional fidelity in ADSR models.
>
---
#### [new 010] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音合成任务，解决零样本TTS中风格控制不足的问题。提出ReStyle-TTS框架，实现连续、相对的风格控制，提升合成语音的多样性与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.03632v1](https://arxiv.org/pdf/2601.03632v1)**

> **作者:** Haitao Li; Chunxiang Jin; Chenglin Li; Wenhao Guan; Zhengxing Huang; Xie Chen
>
> **摘要:** Zero-shot text-to-speech models can clone a speaker's timbre from a short reference audio, but they also strongly inherit the speaking style present in the reference. As a result, synthesizing speech with a desired style often requires carefully selecting reference audio, which is impractical when only limited or mismatched references are available. While recent controllable TTS methods attempt to address this issue, they typically rely on absolute style targets and discrete textual prompts, and therefore do not support continuous and reference-relative style control. We propose ReStyle-TTS, a framework that enables continuous and reference-relative style control in zero-shot TTS. Our key insight is that effective style control requires first reducing the model's implicit dependence on reference style before introducing explicit control mechanisms. To this end, we introduce Decoupled Classifier-Free Guidance (DCFG), which independently controls text and reference guidance, reducing reliance on reference style while preserving text fidelity. On top of this, we apply style-specific LoRAs together with Orthogonal LoRA Fusion to enable continuous and disentangled multi-attribute control, and introduce a Timbre Consistency Optimization module to mitigate timbre drift caused by weakened reference guidance. Experiments show that ReStyle-TTS enables user-friendly, continuous, and relative control over pitch, energy, and multiple emotions while maintaining intelligibility and speaker timbre, and performs robustly in challenging mismatched reference-target style scenarios.
>
---
#### [new 011] ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于语音安全检测任务，旨在评估对抗攻击和深度伪造的检测方法。通过新收集的语音数据集，分析了不同方案在对抗攻击下的表现，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.03944v1](https://arxiv.org/pdf/2601.03944v1)**

> **作者:** Xin Wang; Héctor Delgado; Nicholas Evans; Xuechen Liu; Tomi Kinnunen; Hemlata Tak; Kong Aik Lee; Ivan Kukanov; Md Sahidullah; Massimiliano Todisco; Junichi Yamagishi
>
> **备注:** Submitted
>
> **摘要:** ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake detection solutions. A significant change from previous challenge editions is a new crowdsourced database collected from a substantially greater number of speakers under diverse recording conditions, and a mix of cutting-edge and legacy generative speech technology. With the new database described elsewhere, we provide in this paper an overview of the ASVspoof 5 challenge results for the submissions of 53 participating teams. While many solutions perform well, performance degrades under adversarial attacks and the application of neural encoding/compression schemes. Together with a review of post-challenge results, we also report a study of calibration in addition to other principal challenges and outline a road-map for the future of ASVspoof.
>
---
#### [new 012] Klear: Unified Multi-Task Audio-Video Joint Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文提出Klear，解决音频视频生成中的同步、对齐和模态退化问题，通过统一架构、多任务训练和高质量数据集，实现高效且精准的音视频联合生成。**

- **链接: [https://arxiv.org/pdf/2601.04151v1](https://arxiv.org/pdf/2601.04151v1)**

> **作者:** Jun Wang; Chunyu Qiang; Yuxin Guo; Yiran Wang; Xijuan Zeng; Chen Zhang; Pengfei Wan
>
> **摘要:** Audio-video joint generation has progressed rapidly, yet substantial challenges still remain. Non-commercial approaches still suffer audio-visual asynchrony, poor lip-speech alignment, and unimodal degradation, which can be stemmed from weak audio-visual correspondence modeling, limited generalization, and scarce high-quality dense-caption data. To address these issues, we introduce Klear and delve into three axes--model architecture, training strategy, and data curation. Architecturally, we adopt a single-tower design with unified DiT blocks and an Omni-Full Attention mechanism, achieving tight audio-visual alignment and strong scalability. Training-wise, we adopt a progressive multitask regime--random modality masking to joint optimization across tasks, and a multistage curriculum, yielding robust representations, strengthening A-V aligned world knowledge, and preventing unimodal collapse. For datasets, we present the first large-scale audio-video dataset with dense captions, and introduce a novel automated data-construction pipeline which annotates and filters millions of diverse, high-quality, strictly aligned audio-video-caption triplets. Building on this, Klear scales to large datasets, delivering high-fidelity, semantically and temporally aligned, instruction-following generation in both joint and unimodal settings while generalizing robustly to out-of-distribution scenarios. Across tasks, it substantially outperforms prior methods by a large margin and achieves performance comparable to Veo 3, offering a unified, scalable path toward next-generation audio-video synthesis.
>
---
#### [new 013] Objective comparison of auditory profiles using manifold learning and intrinsic measures
- **分类: physics.med-ph; eess.AS**

- **简介: 该论文属于听觉轮廓分析任务，旨在比较不同听觉轮廓框架的性能。研究通过聚类方法和轮廓数量的影响，评估框架的内部一致性和群集分离性，以找出最优方案。**

- **链接: [https://arxiv.org/pdf/2601.03827v1](https://arxiv.org/pdf/2601.03827v1)**

> **作者:** Chen Xu; Birger Kollmeier; Lena Schell-Majoor
>
> **摘要:** Assigning individuals with hearing impairment to auditory profiles can support a better understanding of the causes and consequences of hearing loss and facilitate profile-based hearing-aid fitting. However, the factors influencing auditory profile generation remain insufficiently understood, and existing profiling frameworks have rarely been compared systematically. This study therefore investigated the impact of two key factors - the clustering method and the number of profiles - on auditory profile generation. In addition, eight established auditory profiling frameworks were systematically reviewed and compared using intrinsic statistical measures and manifold learning techniques. Frameworks were evaluated with respect to internal consistency (i.e., grouping similar individuals) and cluster separation (i.e., clear differentiation between groups). To ensure comparability, all analyses were conducted on a common open-access dataset, the extended Oldenburg Hearing Health Record (OHHR), comprising 1,127 participants (mean age = 67.2 years, SD = 12.0). Results showed that both the clustering method and the chosen number of profiles substantially influenced the resulting auditory profiles. Among purely audiogram-based approaches, the Bisgaard auditory profiles demonstrated the strongest clustering performance, whereas audiometric phenotypes performed worst. Among frameworks incorporating supra-threshold information in addition to the audiogram, the Hearing4All auditory profiles were advantageous, combining a near-optimal number of profile classes (N = 13) with high clustering quality, as indicated by a low Davies-Bouldin index. In conclusion, manifold learning and intrinsic measures enable systematic comparison of auditory profiling frameworks and identify the Hearing4All auditory profile as a promising approach for future research.
>
---
#### [new 014] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，研究对抗攻击下推理变化问题。通过框架分析ALMs在声学感知、认知一致性和矛盾性方面的鲁棒性，揭示推理的双重影响。**

- **链接: [https://arxiv.org/pdf/2601.03615v1](https://arxiv.org/pdf/2601.03615v1)**

> **作者:** Binh Nguyen; Thai Le
>
> **备注:** Preprint for ACL 2026 submission
>
> **摘要:** Audio Language Models (ALMs) offer a promising shift towards explainable audio deepfake detections (ADDs), moving beyond \textit{black-box} classifiers by providing some level of transparency into their predictions via reasoning traces. This necessitates a new class of model robustness analysis: robustness of the predictive reasoning under adversarial attacks, which goes beyond existing paradigm that mainly focuses on the shifts of the final predictions (e.g., fake v.s. real). To analyze such reasoning shifts, we introduce a forensic auditing framework to evaluate the robustness of ALMs' reasoning under adversarial attacks in three inter-connected dimensions: acoustic perception, cognitive coherence, and cognitive dissonance. Our systematic analysis reveals that explicit reasoning does not universally enhance robustness. Instead, we observe a bifurcation: for models exhibiting robust acoustic perception, reasoning acts as a defensive \textit{``shield''}, protecting them from adversarial attacks. However, for others, it imposes a performance \textit{``tax''}, particularly under linguistic attacks which reduce cognitive coherence and increase attack success rate. Crucially, even when classification fails, high cognitive dissonance can serve as a \textit{silent alarm}, flagging potential manipulation. Overall, this work provides a critical evaluation of the role of reasoning in forensic audio deepfake analysis and its vulnerabilities.
>
---
#### [new 015] Mathematical Foundations of Polyphonic Music Generation via Structural Inductive Bias
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，解决polyphonic音乐生成中的“Missing Middle”问题，通过结构归纳偏置和数学理论验证，提升模型稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03612v1](https://arxiv.org/pdf/2601.03612v1)**

> **作者:** Joonwon Seo
>
> **备注:** Monograph. Code available at https://github.com/Chooseredone/Smart-Embedding-Music-Generation
>
> **摘要:** This monograph introduces a novel approach to polyphonic music generation by addressing the "Missing Middle" problem through structural inductive bias. Focusing on Beethoven's piano sonatas as a case study, we empirically verify the independence of pitch and hand attributes using normalized mutual information (NMI=0.167) and propose the Smart Embedding architecture, achieving a 48.30% reduction in parameters. We provide rigorous mathematical proofs using information theory (negligible loss bounded at 0.153 bits), Rademacher complexity (28.09% tighter generalization bound), and category theory to demonstrate improved stability and generalization. Empirical results show a 9.47% reduction in validation loss, confirmed by SVD analysis and an expert listening study (N=53). This dual theoretical and applied framework bridges gaps in AI music generation, offering verifiable insights for mathematically grounded deep learning.
>
---
#### [new 016] Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset
- **分类: cs.CR; cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于舞蹈生成任务，旨在解决现有方法在语义控制和长序列连贯性上的不足。提出LRCM框架，结合扩散模型与Mamba模块，实现多模态引导的自回归舞蹈生成。**

- **链接: [https://arxiv.org/pdf/2601.03323v1](https://arxiv.org/pdf/2601.03323v1)**

> **作者:** Oran Duan; Yinghua Shen; Yingzhu Lv; Luyang Jie; Yaxin Liu; Qiong Wu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Advances in generative models and sequence learning have greatly promoted research in dance motion generation, yet current methods still suffer from coarse semantic control and poor coherence in long sequences. In this work, we present Listen to Rhythm, Choose Movements (LRCM), a multimodal-guided diffusion framework supporting both diverse input modalities and autoregressive dance motion generation. We explore a feature decoupling paradigm for dance datasets and generalize it to the Motorica Dance dataset, separating motion capture data, audio rhythm, and professionally annotated global and local text descriptions. Our diffusion architecture integrates an audio-latent Conformer and a text-latent Cross-Conformer, and incorporates a Motion Temporal Mamba Module (MTMM) to enable smooth, long-duration autoregressive synthesis. Experimental results indicate that LRCM delivers strong performance in both functional capability and quantitative metrics, demonstrating notable potential in multimodal input scenarios and extended sequence generation. We will release the full codebase, dataset, and pretrained models publicly upon acceptance.
>
---
## 更新

#### [replaced 001] Zero-Day Audio DeepFake Detection via Retrieval Augmentation and Profile Matching
- **分类: cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决零日攻击检测问题。提出一种无需训练的检索增强框架，通过知识表示和语音档案匹配实现有效检测。**

- **链接: [https://arxiv.org/pdf/2509.21728v2](https://arxiv.org/pdf/2509.21728v2)**

> **作者:** Xuechen Liu; Xin Wang; Junichi Yamagishi
>
> **摘要:** Modern audio deepfake detectors built on foundation models and large training datasets achieve promising detection performance. However, they struggle with zero-day attacks, where the audio samples are generated by novel synthesis methods that models have not seen from reigning training data. Conventional approaches fine-tune the detector, which can be problematic when prompt response is needed. This paper proposes a training-free retrieval-augmented framework for zero-day audio deepfake detection that leverages knowledge representations and voice profile matching. Within this framework, we propose simple yet effective retrieval and ensemble methods that reach performance comparable to supervised baselines and their fine-tuned counterparts on the DeepFake-Eval-2024 benchmark, without any additional model training. We also conduct ablation on voice profile attributes, and demonstrate the cross-database generalizability of the framework with introducing simple and training-free fusion strategies.
>
---
#### [replaced 002] DiFlow-TTS: Compact and Low-Latency Zero-Shot Text-to-Speech with Factorized Discrete Flow Matching
- **分类: cs.SD; cs.CL; cs.CV**

- **简介: 该论文提出DiFlow-TTS，一种零样本文本转语音系统，通过离散流匹配实现高效生成。解决语音合成自然度与速度问题，采用因子化表示和确定性映射器提升性能。**

- **链接: [https://arxiv.org/pdf/2509.09631v3](https://arxiv.org/pdf/2509.09631v3)**

> **作者:** Ngoc-Son Nguyen; Thanh V. T. Tran; Hieu-Nghia Huynh-Nguyen; Truong-Son Hy; Van Nguyen
>
> **摘要:** This paper introduces DiFlow-TTS, a novel zero-shot text-to-speech (TTS) system that employs discrete flow matching for generative speech modeling. We position this work as an entry point that may facilitate further advances in this research direction. Through extensive empirical evaluation, we analyze both the strengths and limitations of this approach across key aspects, including naturalness, expressive attributes, speaker identity, and inference latency. To this end, we leverage factorized speech representations and design a deterministic Phoneme-Content Mapper for modeling linguistic content, together with a Factorized Discrete Flow Denoiser that jointly models multiple discrete token streams corresponding to prosody and acoustics to capture expressive speech attributes. Experimental results demonstrate that DiFlow-TTS achieves strong performance across multiple metrics while maintaining a compact model size, up to 11.7 times smaller, and enabling low-latency inference that is up to 34 times faster than recent state-of-the-art baselines. Audio samples are available on our demo page: https://diflow-tts.github.io.
>
---
#### [replaced 003] Improving Underwater Acoustic Classification Through Learnable Gabor Filter Convolution and Attention Mechanisms
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于 underwater acoustic classification 任务，旨在提升水下声学目标分类的准确性与稳定性。通过引入可学习的 Gabor 卷积和注意力机制，优化模型性能。**

- **链接: [https://arxiv.org/pdf/2512.14714v2](https://arxiv.org/pdf/2512.14714v2)**

> **作者:** Lucas Cesar Ferreira Domingos; Russell Brinkworth; Paulo Eduardo Santos; Karl Sammut
>
> **摘要:** Remotely detecting and classifying underwater acoustic targets is critical for environmental monitoring and defence. However, the complexity of ship-radiated and environmental noise poses significant challenges for accurate signal processing. While recent advancements in machine learning have improved classification accuracy, limited dataset availability and a lack of standardised experimentation hinder generalisation and robustness. This paper introduces GSE ResNeXt, a deep learning architecture integrating learnable Gabor convolutional layers with a ResNeXt backbone enhanced by squeeze-and-excitation attention. The Gabor filters serve as two-dimensional adaptive band-pass filters, extending the feature channel representation. Its combination with channel attention improves training stability and convergence while enhancing the model's ability to extract discriminative features. The model is evaluated using three training-test split strategies that reflect increasingly complex classification tasks, demonstrating how systematic evaluation design addresses issues such as data leakage, temporal separation, and taxonomy. Results show that GSE ResNeXt consistently outperforms baseline models like Xception, ResNet, and MobileNetV2, in terms of classification performance. Regarding stability and convergence, adding Gabor convolutions to the initial layers of the model reduced training time by up to 62%. During the evaluation of training-testing splits, temporal separation between subsets significantly affected performance, proving more influential than training data volume. These findings suggest that signal processing can enhance model reliability and generalisation under varying environmental conditions, particularly in data-limited underwater acoustic classification. Future developments should focus on mitigating environmental effects on input signals.
>
---
#### [replaced 004] HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决韩英混语识别问题。提出HiKE框架，包含真实混语数据和分层标注，用于评估和提升模型的混语识别能力。**

- **链接: [https://arxiv.org/pdf/2509.24613v3](https://arxiv.org/pdf/2509.24613v3)**

> **作者:** Gio Paik; Yongbeom Kim; Soungmin Lee; Sangmin Ahn; Chanwoo Kim
>
> **备注:** EACL Findings 2026
>
> **摘要:** Despite advances in multilingual automatic speech recognition (ASR), code-switching (CS), the mixing of languages within an utterance common in daily speech, remains a severely underexplored challenge. In this paper, we introduce HiKE: the Hierarchical Korean-English code-switching benchmark, the first globally accessible non-synthetic evaluation framework for Korean-English CS, aiming to provide a means for the precise evaluation of multilingual ASR models and to foster research in the field. The proposed framework not only consists of high-quality, natural CS data across various topics, but also provides meticulous loanword labels and a hierarchical CS-level labeling scheme (word, phrase, and sentence) that together enable a systematic evaluation of a model's ability to handle each distinct level of code-switching. Through evaluations of diverse multilingual ASR models and fine-tuning experiments, this paper demonstrates that although most multilingual ASR models initially exhibit inadequate CS-ASR performance, this capability can be enabled through fine-tuning with synthetic CS data. HiKE is available at https://github.com/ThetaOne-AI/HiKE.
>
---
#### [replaced 005] BENYO-S2ST-Corpus-1: A Bilingual English-to-Yoruba Direct Speech-to-Speech Translation Corpus
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音到语音翻译任务，旨在解决英语到约鲁巴语等低资源语言数据不足的问题。通过构建BENYO-S2ST-Corpus-1数据集，提升相关模型性能。**

- **链接: [https://arxiv.org/pdf/2507.09342v4](https://arxiv.org/pdf/2507.09342v4)**

> **作者:** Emmanuel Adetiba; Abdultaofeek Abayomi; Raymond J. Kala; Ayodele H. Ifijeh; Oluwatobi E. Dare; Olabode Idowu-Bismark; Gabriel O. Sobola; Joy N. Adetiba; Monsurat Adepeju Lateef
>
> **摘要:** There is a major shortage of Speech-to-Speech Translation (S2ST) datasets for high resource-to-low resource language pairs such as English-to-Yoruba. Thus, in this study, we curated the Bilingual English-to-Yoruba Speech-to-Speech Translation Corpus Version 1 (BENYO-S2ST-Corpus-1). The corpus is based on a hybrid architecture we developed for large-scale direct S2ST corpus creation at reduced cost. To achieve this, we leveraged non speech-to-speech Standard Yoruba (SY) real-time audios and transcripts in the YORULECT Corpus as well as the corresponding Standard English (SE) transcripts. YORULECT Corpus is small scale(1,504) samples, and it does not have paired English audios. Therefore, we generated the SE audios using pre-trained AI models (i.e. Facebook MMS). We also developed an audio augmentation algorithm named AcoustAug based on three latent acoustic features to generate augmented audios from the raw audios of the two languages. BENYO-S2ST-Corpus-1 has 12,032 audio samples per language, which gives a total of 24,064 sample size. The total audio duration for the two languages is 41.20 hours. This size is quite significant. Beyond building S2ST models, BENYO-S2ST-Corpus-1 can be used to build pretrained models or improve existing ones. The created corpus and Coqui framework were used to build a pretrained Yoruba TTS model (named YoruTTS-1.5) as a proof of concept. The YoruTTS-1.5 gave a F0 RMSE value of 63.54 after 1,000 epochs, which indicates moderate fundamental pitch similarity with the reference real-time audio. Ultimately, the corpus architecture in this study can be leveraged by researchers and developers to curate datasets for multilingual high-resource-to-low-resource African languages. This will bridge the huge digital divides in translations among high and low-resource language pairs. BENYO-S2ST-Corpus-1 and YoruTTS-1.5 are publicly available at (https://bit.ly/40bGMwi).
>
---
