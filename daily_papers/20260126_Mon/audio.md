# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Test-Time Adaptation for Speech Emotion Recognition
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音情感识别任务，解决领域漂移问题。通过评估11种测试时自适应方法，发现无需反向传播的方法最有效。**

- **链接: [https://arxiv.org/pdf/2601.16240v1](https://arxiv.org/pdf/2601.16240v1)**

> **作者:** Jiaheng Dong; Hong Jia; Ting Dang
>
> **备注:** Accepted by 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** The practical utility of Speech Emotion Recognition (SER) systems is undermined by their fragility to domain shifts, such as speaker variability, the distinction between acted and naturalistic emotions, and cross-corpus variations. While domain adaptation and fine-tuning are widely studied, they require either source data or labelled target data, which are often unavailable or raise privacy concerns in SER. Test-time adaptation (TTA) bridges this gap by adapting models at inference using only unlabeled target data. Yet, having been predominantly designed for image classification and speech recognition, the efficacy of TTA for mitigating the unique domain shifts in SER has not been investigated. In this paper, we present the first systematic evaluation and comparison covering 11 TTA methods across three representative SER tasks. The results indicate that backpropagation-free TTA methods are the most promising. Conversely, entropy minimization and pseudo-labeling generally fail, as their core assumption of a single, confident ground-truth label is incompatible with the inherent ambiguity of emotional expression. Further, no single method universally excels, and its effectiveness is highly dependent on the distributional shifts and tasks.
>
---
#### [new 002] A Novel Transfer Learning Approach for Mental Stability Classification from Voice Signal
- **分类: cs.SD; cs.NE; eess.AS**

- **简介: 该论文属于心理健康分类任务，旨在解决数据不足问题。通过结合数据增强与迁移学习，提升CNN在语音谱图上对心理稳定性的分类效果。**

- **链接: [https://arxiv.org/pdf/2601.16793v1](https://arxiv.org/pdf/2601.16793v1)**

> **作者:** Rafiul Islam; Md. Taimur Ahad
>
> **摘要:** This study presents a novel transfer learning approach and data augmentation technique for mental stability classification using human voice signals and addresses the challenges associated with limited data availability. Convolutional neural networks (CNNs) have been employed to analyse spectrogram images generated from voice recordings. Three CNN architectures, VGG16, InceptionV3, and DenseNet121, were evaluated across three experimental phases: training on non-augmented data, augmented data, and transfer learning. This proposed transfer learning approach involves pre-training models on the augmented dataset and fine-tuning them on the non-augmented dataset while ensuring strict data separation to prevent data leakage. The results demonstrate significant improvements in classification performance compared to the baseline approach. Among three CNN architectures, DenseNet121 achieved the highest accuracy of 94% and an AUC score of 99% using the proposed transfer learning approach. This finding highlights the effectiveness of combining data augmentation and transfer learning to enhance CNN-based classification of mental stability using voice spectrograms, offering a promising non-invasive tool for mental health diagnostics.
>
---
#### [new 003] E2E-AEC: Implementing an end-to-end neural network learning approach for acoustic echo cancellation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决传统声学回声消除（AEC）依赖复杂步骤的问题。提出一种端到端神经网络方法，通过渐进学习、知识迁移和注意力优化实现高效回声消除。**

- **链接: [https://arxiv.org/pdf/2601.16774v1](https://arxiv.org/pdf/2601.16774v1)**

> **作者:** Yiheng Jiang; Biao Tian; Haoxu Wang; Shengkui Zhao; Bin Ma; Daren Chen; Xiangang Li
>
> **备注:** This paper has been accepted by ICASSP2026
>
> **摘要:** We propose a novel neural network-based end-to-end acoustic echo cancellation (E2E-AEC) method capable of streaming inference, which operates effectively without reliance on traditional linear AEC (LAEC) techniques and time delay estimation. Our approach includes several key strategies: First, we introduce and refine progressive learning to gradually enhance echo suppression. Second, our model employs knowledge transfer by initializing with a pre-trained LAECbased model, harnessing the insights gained from LAEC training. Third, we optimize the attention mechanism with a loss function applied on attention weights to achieve precise time alignment between the reference and microphone signals. Lastly, we incorporate voice activity detection to enhance speech quality and improve echo removal by masking the network output when near-end speech is absent. The effectiveness of our approach is validated through experiments conducted on public datasets.
>
---
#### [new 004] TidyVoice: A Curated Multilingual Dataset for Speaker Verification Derived from Common Voice
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出TidyVoice数据集，解决多语言说话人验证数据不足的问题。通过清理Common Voice数据，构建包含单语和双语说话人的数据集，并验证了模型在该数据集上的性能提升。**

- **链接: [https://arxiv.org/pdf/2601.16358v1](https://arxiv.org/pdf/2601.16358v1)**

> **作者:** Aref Farhadipour; Jan Marquenie; Srikanth Madikeri; Eleanor Chodroff
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** The development of robust, multilingual speaker recognition systems is hindered by a lack of large-scale, publicly available and multilingual datasets, particularly for the read-speech style crucial for applications like anti-spoofing. To address this gap, we introduce the TidyVoice dataset derived from the Mozilla Common Voice corpus after mitigating its inherent speaker heterogeneity within the provided client IDs. TidyVoice currently contains training and test data from over 212,000 monolingual speakers (Tidy-M) and around 4,500 multilingual speakers (Tidy-X) from which we derive two distinct conditions. The Tidy-M condition contains target and non-target trials from monolingual speakers across 81 languages. The Tidy-X condition contains target and non-target trials from multilingual speakers in both same- and cross-language trials. We employ two architectures of ResNet models, achieving a 0.35% EER by fine-tuning on our comprehensive Tidy-M partition. Moreover, we show that this fine-tuning enhances the model's generalization, improving performance on unseen conversational interview data from the CANDOR corpus. The complete dataset, evaluation trials, and our models are publicly released to provide a new resource for the community.
>
---
#### [new 005] I Guess That's Why They Call it the Blues: Causal Analysis for Audio Classifiers
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频分类任务，旨在解决音频分类器依赖非音乐相关特征的问题。通过因果分析，发现分类所需的关键频率特征，并实现对模型输出的精准操控。**

- **链接: [https://arxiv.org/pdf/2601.16675v1](https://arxiv.org/pdf/2601.16675v1)**

> **作者:** David A. Kelly; Hana Chockler
>
> **摘要:** It is well-known that audio classifiers often rely on non-musically relevant features and spurious correlations to classify audio. Hence audio classifiers are easy to manipulate or confuse, resulting in wrong classifications. While inducing a misclassification is not hard, until now the set of features that the classifiers rely on was not well understood. In this paper we introduce a new method that uses causal reasoning to discover features of the frequency space that are sufficient and necessary for a given classification. We describe an implementation of this algorithm in the tool FreqReX and provide experimental results on a number of standard benchmark datasets. Our experiments show that causally sufficient and necessary subsets allow us to manipulate the outputs of the models in a variety of ways by changing the input very slightly. Namely, a change to one out of 240,000 frequencies results in a change in classification 58% of the time, and the change can be so small that it is practically inaudible. These results show that causal analysis is useful for understanding the reasoning process of audio classifiers and can be used to successfully manipulate their outputs.
>
---
#### [new 006] SoundBreak: A Systematic Study of Audio-Only Adversarial Attacks on Trimodal Models
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究音频对抗攻击对多模态模型的影响，旨在揭示单模态攻击漏洞。通过分析不同攻击目标，验证音频扰动可导致多模态系统失效。**

- **链接: [https://arxiv.org/pdf/2601.16231v1](https://arxiv.org/pdf/2601.16231v1)**

> **作者:** Aafiya Hussain; Gaurav Srivastava; Alvi Ishmam; Zaber Hakim; Chris Thomas
>
> **摘要:** Multimodal foundation models that integrate audio, vision, and language achieve strong performance on reasoning and generation tasks, yet their robustness to adversarial manipulation remains poorly understood. We study a realistic and underexplored threat model: untargeted, audio-only adversarial attacks on trimodal audio-video-language models. We analyze six complementary attack objectives that target different stages of multimodal processing, including audio encoder representations, cross-modal attention, hidden states, and output likelihoods. Across three state-of-the-art models and multiple benchmarks, we show that audio-only perturbations can induce severe multimodal failures, achieving up to 96% attack success rate. We further show that attacks can be successful at low perceptual distortions (LPIPS <= 0.08, SI-SNR >= 0) and benefit more from extended optimization than increased data scale. Transferability across models and encoders remains limited, while speech recognition systems such as Whisper primarily respond to perturbation magnitude, achieving >97% attack success under severe distortion. These results expose a previously overlooked single-modality attack surface in multimodal systems and motivate defenses that enforce cross-modal consistency.
>
---
#### [new 007] Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech: Challenges and Opportunities
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语言评估任务，旨在解决L2英语发音自动评分难题。研究评估了Qwen2-Audio-7B-Instruct在5000个语音样本上的零样本性能，探讨其在准确性、流利度等方面的评估能力。**

- **链接: [https://arxiv.org/pdf/2601.16230v1](https://arxiv.org/pdf/2601.16230v1)**

> **作者:** Aditya Kamlesh Parikh; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants which is financed by the Dutch Research Council (NWO)
>
> **摘要:** An accurate assessment of L2 English pronunciation is crucial for language learning, as it provides personalized feedback and ensures a fair evaluation of individual progress. However, automated scoring remains challenging due to the complexity of sentence-level fluency, prosody, and completeness. This paper evaluates the zero-shot performance of Qwen2-Audio-7B-Instruct, an instruction-tuned speech-LLM, on 5,000 Speechocean762 utterances. The model generates rubric-aligned scores for accuracy, fluency, prosody, and completeness, showing strong agreement with human ratings within +-2 tolerance, especially for high-quality speech. However, it tends to overpredict low-quality speech scores and lacks precision in error detection. These findings demonstrate the strong potential of speech LLMs in scalable pronunciation assessment and suggest future improvements through enhanced prompting, calibration, and phonetic integration to advance Computer-Assisted Pronunciation Training.
>
---
#### [new 008] Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频大模型与脑电研究任务，旨在探究音频LLMs的内部表征是否与人类神经活动对齐。通过分析12个模型与EEG信号的相似性，揭示了模型表征的结构特征和神经机制。**

- **链接: [https://arxiv.org/pdf/2601.16540v1](https://arxiv.org/pdf/2601.16540v1)**

> **作者:** Haoyun Yang; Xin Xiao; Jiang Zhong; Yu Tian; Dong Xiaohua; Yu Mao; Hao Wu; Kaiwen Wei
>
> **摘要:** Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
>
---
#### [new 009] FlowSE-GRPO: Training Flow Matching Speech Enhancement via Online Reinforcement Learning
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决生成模型与人类偏好对齐问题。通过引入在线GRPO算法，实现高效后训练对齐，提升音频质量与任务指标。**

- **链接: [https://arxiv.org/pdf/2601.16483v1](https://arxiv.org/pdf/2601.16483v1)**

> **作者:** Haoxu Wang; Biao Tian; Yiheng Jiang; Zexu Pan; Shengkui Zhao; Bin Ma; Daren Chen; Xiangang Li
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Generative speech enhancement offers a promising alternative to traditional discriminative methods by modeling the distribution of clean speech conditioned on noisy inputs. Post-training alignment via reinforcement learning (RL) effectively aligns generative models with human preferences and downstream metrics in domains such as natural language processing, but its use in speech enhancement remains limited, especially for online RL. Prior work explores offline methods like Direct Preference Optimization (DPO); online methods such as Group Relative Policy Optimization (GRPO) remain largely uninvestigated. In this paper, we present the first successful integration of online GRPO into a flow-matching speech enhancement framework, enabling efficient post-training alignment to perceptual and task-oriented metrics with few update steps. Unlike prior GRPO work on Large Language Models, we adapt the algorithm to the continuous, time-series nature of speech and to the dynamics of flow-matching generative models. We show that optimizing a single reward yields rapid metric gains but often induces reward hacking that degrades audio fidelity despite higher scores. To mitigate this, we propose a multi-metric reward optimization strategy that balances competing objectives, substantially reducing overfitting and improving overall performance. Our experiments validate online GRPO for speech enhancement and provide practical guidance for RL-based post-training of generative audio models.
>
---
#### [new 010] The CMU-AIST submission for the ICME 2025 Audio Encoder Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频编码任务，旨在提升音频编码性能。通过扩展BEATs模型并融合多数据源，构建高效音频编码系统。**

- **链接: [https://arxiv.org/pdf/2601.16273v1](https://arxiv.org/pdf/2601.16273v1)**

> **作者:** Shikhar Bharadwaj; Samuele Cornell; Kwanghee Choi; Hye-jin Shim; Soham Deshmukh; Satoru Fukayama; Shinji Watanabe
>
> **摘要:** This technical report describes our submission to the ICME 2025 audio encoder challenge. Our submitted system is built on BEATs, a masked speech token prediction based audio encoder. We extend the BEATs model using 74,000 hours of data derived from various speech, music, and sound corpora and scale its architecture upto 300 million parameters. We experiment with speech-heavy and balanced pre-training mixtures to study the impact of different domains on final performance. Our submitted system consists of an ensemble of the Dasheng 1.2 billion model with two custom scaled-up BEATs models trained on the aforementioned pre-training data mixtures. We also propose a simple ensembling technique that retains the best capabilities of constituent models and surpasses both the baseline and Dasheng 1.2B. For open science, we publicly release our trained checkpoints via huggingface at https://huggingface.co/shikhar7ssu/OpenBEATs-ICME-SOUND and https://huggingface.co/shikhar7ssu/OpenBEATs-ICME.
>
---
#### [new 011] ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于情感对话生成任务，旨在解决多轮对话中情感信息与语境连贯性弱的问题。提出ES4R框架，通过显式建模情感上下文并融合文本语义生成共情回复。**

- **链接: [https://arxiv.org/pdf/2601.16225v1](https://arxiv.org/pdf/2601.16225v1)**

> **作者:** Zhuoyue Gao; Xiaohui Wang; Xiaocui Yang; Wen Zhang; Daling Wang; Shi Feng; Yifei Zhang
>
> **摘要:** Empathetic speech dialogue requires not only understanding linguistic content but also perceiving rich paralinguistic information such as prosody, tone, and emotional intensity for affective understandings. Existing speech-to-speech large language models either rely on ASR transcription or use encoders to extract latent representations, often weakening affective information and contextual coherence in multi-turn dialogues. To address this, we propose \textbf{ES4R}, a framework for speech-based empathetic response generation. Our core innovation lies in explicitly modeling structured affective context before speech encoding, rather than relying on implicit learning by the encoder or explicit emotion supervision. Specifically, we introduce a dual-level attention mechanism to capture turn-level affective states and dialogue-level affective dynamics. The resulting affective representations are then integrated with textual semantics through speech-guided cross-modal attention to generate empathetic responses. For speech output, we employ energy-based strategy selection and style fusion to achieve empathetic speech synthesis. ES4R consistently outperforms strong baselines in both automatic and human evaluations and remains robust across different LLM backbones.
>
---
#### [new 012] Omni-directional attention mechanism based on Mamba for speech separation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音分离任务，旨在解决Mamba模型在二维谱图中捕捉全局依赖能力不足的问题。提出一种双向注意力机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16603v1](https://arxiv.org/pdf/2601.16603v1)**

> **作者:** Ke Xue; Chang Sun; Rongfei Fan; Jing Wang; Han Hu
>
> **摘要:** Mamba, a selective state-space model (SSM), has emerged as an efficient alternative to Transformers for speech modeling, enabling long-sequence processing with linear complexity. While effective in speech separation, existing approaches, whether in the time or time-frequency domain, typically decompose the input along a single dimension into short one-dimensional sequences before processing them with Mamba, which restricts it to local 1D modeling and limits its ability to capture global dependencies across the 2D spectrogram. In this work, we propose an efficient omni-directional attention (OA) mechanism built upon unidirectional Mamba, which models global dependencies from ten different directions on the spectrogram. We expand the proposed mechanism into two baseline separation models and evaluate on three public datasets. Experimental results show that our approach consistently achieves significant performance gains over the baselines while preserving linear complexity, outperforming existing state-of-the-art (SOTA) systems.
>
---
#### [new 013] CORD: Bridging the Audio-Text Reasoning Gap via Weighted On-policy Cross-modal Distillation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频-文本推理任务，旨在解决LALMs在知识和推理能力上的退化问题。提出CORD框架，通过多粒度对齐提升音频条件推理性能。**

- **链接: [https://arxiv.org/pdf/2601.16547v1](https://arxiv.org/pdf/2601.16547v1)**

> **作者:** Jing Hu; Danxiang Zhu; Xianlong Luo; Dan Zhang; Shuwei He; Yishu Lei; Haitao Zheng; Shikun Feng; Jingzhou He; Yu Sun; Hua Wu; Haifeng Wang
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Large Audio Language Models (LALMs) have garnered significant research interest. Despite being built upon text-based large language models (LLMs), LALMs frequently exhibit a degradation in knowledge and reasoning capabilities. We hypothesize that this limitation stems from the failure of current training paradigms to effectively bridge the acoustic-semantic gap within the feature representation space. To address this challenge, we propose CORD, a unified alignment framework that performs online cross-modal self-distillation. Specifically, it aligns audio-conditioned reasoning with its text-conditioned counterpart within a unified model. Leveraging the text modality as an internal teacher, CORD performs multi-granularity alignment throughout the audio rollout process. At the token level, it employs on-policy reverse KL divergence with importance-aware weighting to prioritize early and semantically critical tokens. At the sequence level, CORD introduces a judge-based global reward to optimize complete reasoning trajectories via Group Relative Policy Optimization (GRPO). Empirical results across multiple benchmarks demonstrate that CORD consistently enhances audio-conditioned reasoning and substantially bridges the audio-text performance gap with only 80k synthetic training samples, validating the efficacy and data efficiency of our on-policy, multi-level cross-modal alignment approach.
>
---
#### [new 014] EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出EdgeSpot模型，用于边缘设备的关键词检测任务，解决少样本下准确率与效率的问题。通过优化网络结构和知识蒸馏提升性能。**

- **链接: [https://arxiv.org/pdf/2601.16316v1](https://arxiv.org/pdf/2601.16316v1)**

> **作者:** Oguzhan Buyuksolak; Alican Gok; Osman Erman Okman
>
> **备注:** Accepted to be presented in IEEE ICASSP 2026
>
> **摘要:** We introduce an efficient few-shot keyword spotting model for edge devices, EdgeSpot, that pairs an optimized version of a BC-ResNet-based acoustic backbone with a trainable Per-Channel Energy Normalization frontend and lightweight temporal self-attention. Knowledge distillation is utilized during training by employing a self-supervised teacher model, optimized with Sub-center ArcFace loss. This study demonstrates that the EdgeSpot model consistently provides better accuracy at a fixed false-alarm rate (FAR) than strong BC-ResNet baselines. The largest variant, EdgeSpot-4, improves the 10-shot accuracy at 1% FAR from 73.7% to 82.0%, which requires only 29.4M MACs with 128k parameters.
>
---
#### [new 015] Contrastive Knowledge Distillation for Embedding Refinement in Personalized Speech Enhancement
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于个性化语音增强任务，旨在解决目标语音在推理时的变异问题。通过引入对比知识蒸馏方法，训练轻量级说话人编码器以实时优化语音嵌入，提升增强效果。**

- **链接: [https://arxiv.org/pdf/2601.16235v1](https://arxiv.org/pdf/2601.16235v1)**

> **作者:** Thomas Serre; Mathieu Fontaine; Éric Benhaim; Slim Essid
>
> **摘要:** Personalized speech enhancement (PSE) has shown convincing results when it comes to extracting a known target voice among interfering ones. The corresponding systems usually incorporate a representation of the target voice within the enhancement system, which is extracted from an enrollment clip of the target voice with upstream models. Those models are generally heavy as the speaker embedding's quality directly affects PSE performances. Yet, embeddings generated beforehand cannot account for the variations of the target voice during inference time. In this paper, we propose to perform on-thefly refinement of the speaker embedding using a tiny speaker encoder. We first introduce a novel contrastive knowledge distillation methodology in order to train a 150k-parameter encoder from complex embeddings. We then use this encoder within the enhancement system during inference and show that the proposed method greatly improves PSE performances while maintaining a low computational load.
>
---
#### [new 016] Auditory Attention Decoding without Spatial Information: A Diotic EEG Study
- **分类: eess.SP; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于 auditory attention decoding 任务，旨在解决无空间信息下的注意力解码问题。通过构建共享潜在空间模型，提升在混音环境中的解码准确率。**

- **链接: [https://arxiv.org/pdf/2601.16442v1](https://arxiv.org/pdf/2601.16442v1)**

> **作者:** Masahiro Yoshino; Haruki Yokota; Junya Hara; Yuichi Tanaka; Hiroshi Higashi
>
> **摘要:** Auditory attention decoding (AAD) identifies the attended speech stream in multi-speaker environments by decoding brain signals such as electroencephalography (EEG). This technology is essential for realizing smart hearing aids that address the cocktail party problem and for facilitating objective audiometry systems. Existing AAD research mainly utilizes dichotic environments where different speech signals are presented to the left and right ears, enabling models to classify directional attention rather than speech content. However, this spatial reliance limits applicability to real-world scenarios, such as the "cocktail party" situation, where speakers overlap or move dynamically. To address this challenge, we propose an AAD framework for diotic environments where identical speech mixtures are presented to both ears, eliminating spatial cues. Our approach maps EEG and speech signals into a shared latent space using independent encoders. We extract speech features using wav2vec 2.0 and encode them with a 2-layer 1D convolutional neural network (CNN), while employing the BrainNetwork architecture for EEG encoding. The model identifies the attended speech by calculating the cosine similarity between EEG and speech representations. We evaluate our method on a diotic EEG dataset and achieve 72.70% accuracy, which is 22.58% higher than the state-of-the-art direction-based AAD method.
>
---
## 更新

#### [replaced 001] Cross-Lingual F5-TTS: Towards Language-Agnostic Voice Cloning and Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决跨语言语音克隆中依赖参考文本的问题。通过强制对齐获取词界，无需文本直接合成，实现跨语言语音克隆。**

- **链接: [https://arxiv.org/pdf/2509.14579v3](https://arxiv.org/pdf/2509.14579v3)**

> **作者:** Qingyu Liu; Yushen Chen; Zhikang Niu; Chunhui Wang; Yunting Yang; Bowen Zhang; Jian Zhao; Pengcheng Zhu; Kai Yu; Xie Chen
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Flow-matching-based text-to-speech (TTS) models have shown high-quality speech synthesis. However, most current flow-matching-based TTS models still rely on reference transcripts corresponding to the audio prompt for synthesis. This dependency prevents cross-lingual voice cloning when audio prompt transcripts are unavailable, particularly for unseen languages. The key challenges for flow-matching-based TTS models to remove audio prompt transcripts are identifying word boundaries during training and determining appropriate duration during inference. In this paper, we introduce Cross-Lingual F5-TTS, a framework that enables cross-lingual voice cloning without audio prompt transcripts. Our method preprocesses audio prompts by forced alignment to obtain word boundaries, enabling direct synthesis from audio prompts while excluding transcripts during training. To address the duration modeling challenge, we train speaking rate predictors at different linguistic granularities to derive duration from speaker pace. Experiments show that our approach matches the performance of F5-TTS while enabling cross-lingual voice cloning.
>
---
#### [replaced 002] Lightweight Implicit Neural Network for Binaural Audio Synthesis
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于空间音频合成任务，旨在解决高保真双耳音频合成在边缘设备上的计算资源消耗过高的问题。提出Lite-INN框架，通过两阶段方法提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2509.14069v2](https://arxiv.org/pdf/2509.14069v2)**

> **作者:** Xikun Lu; Fang Liu; Weizhi Shi; Jinqiu Sang
>
> **备注:** Accepted at IEEE ICASSP 2026
>
> **摘要:** High-fidelity binaural audio synthesis is crucial for immersive listening, but existing methods require extensive computational resources, limiting their edge-device application. To address this, we propose the Lightweight Implicit Neural Network (Lite-INN), a novel two-stage framework. Lite-INN first generates initial estimates using a time-domain warping, which is then refined by an Implicit Binaural Corrector (IBC) module. IBC is an implicit neural network that predicts amplitude and phase corrections directly, resulting in a highly compact model architecture. Experimental results show that Lite-INN achieves statistically comparable perceptual quality to the best-performing baseline model while significantly improving computational efficiency. Compared to the previous state-of-the-art method (NFS), Lite-INN achieves a 72.7% reduction in parameters and requires significantly fewer compute operations (MACs). This demonstrates that our approach effectively addresses the trade-off between synthesis quality and computational efficiency, providing a new solution for high-fidelity edge-device spatial audio applications.
>
---
#### [replaced 003] Audio dequantization using instantaneous frequency
- **分类: eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决音频去量化问题。通过引入相位感知正则化方法，提升音频信号的时频连续性，减少能量损失伪影。**

- **链接: [https://arxiv.org/pdf/2510.16813v2](https://arxiv.org/pdf/2510.16813v2)**

> **作者:** Vojtěch Kovanda; Pavel Rajmic
>
> **摘要:** We present a dequantization method that employs a phase-aware regularizer, originally successfully applied in an audio inpainting problem. The method promotes a temporal continuity of sinusoidal components in time-frequency representation of the audio signal, and avoids energy loss artifacts commonly encountered with l1-based regularization approaches. The proposed method is called the Phase-Aware Audio Dequantizer (PHADQ). The method are evaluated against the state-of-the-art using the SDR and PEMO-Q ODG objective metrics, and a~subjective MUSHRA-like test.
>
---
#### [replaced 004] Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，探讨CoT与直接提示方法的效果差异。通过增加数据量比较两种策略，发现直接提示在数据增多时表现更稳定，可能更有效。**

- **链接: [https://arxiv.org/pdf/2510.03093v2](https://arxiv.org/pdf/2510.03093v2)**

> **作者:** Oriol Pareras; Gerard I. Gállego; Federico Costa; Cristina España-Bonet; Javier Hernando
>
> **备注:** To appear in Proc. ICASSP 2026, May 04-08, 2026, Barcelona, Spain
>
> **摘要:** Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
>
---
#### [replaced 005] MusiCRS: Benchmarking Audio-Centric Conversational Recommendation
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出MusiCRS，首个面向音频对话推荐的基准数据集，解决音乐推荐中音频内容理解与对话结合的问题，支持多模态评估并揭示当前系统在跨模态整合上的不足。**

- **链接: [https://arxiv.org/pdf/2509.19469v2](https://arxiv.org/pdf/2509.19469v2)**

> **作者:** Rohan Surana; Amit Namburi; Gagan Mundada; Abhay Lal; Zachary Novack; Julian McAuley; Junda Wu
>
> **备注:** 5 pages
>
> **摘要:** Conversational recommendation has advanced rapidly with large language models (LLMs), yet music remains a uniquely challenging domain in which effective recommendations require reasoning over audio content beyond what text or metadata can capture. We present MusiCRS, the first benchmark for audio-centric conversational recommendation that links authentic user conversations from Reddit with corresponding tracks. MusiCRS includes 477 high-quality conversations spanning diverse genres (classical, hip-hop, electronic, metal, pop, indie, jazz), with 3,589 unique musical entities and audio grounding via YouTube links. MusiCRS supports evaluation under three input modality configurations: audio-only, query-only, and audio+query, allowing systematic comparison of audio-LLMs, retrieval models, and traditional approaches. Our experiments reveal that current systems struggle with cross-modal integration, with optimal performance frequently occurring in single-modality settings rather than multimodal configurations. This highlights fundamental limitations in cross-modal knowledge integration, as models excel at dialogue semantics but struggle when grounding abstract musical concepts in audio. To facilitate progress, we release the MusiCRS dataset (https://huggingface.co/datasets/rohan2810/MusiCRS), evaluation code (https://github.com/rohan2810/musiCRS), and comprehensive baselines.
>
---
#### [replaced 006] Speaker Anonymisation for Speech-based Suicide Risk Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音隐私保护任务，旨在解决青少年自杀风险检测中说话人身份泄露问题。通过研究多种匿名化方法，在保护隐私的同时保持检测性能。**

- **链接: [https://arxiv.org/pdf/2509.22148v2](https://arxiv.org/pdf/2509.22148v2)**

> **作者:** Ziyun Cui; Sike Jia; Yang Lin; Yinan Duan; Diyang Qu; Runsen Chen; Chao Zhang; Chang Lei; Wen Wu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Adolescent suicide is a critical global health issue, and speech provides a cost-effective modality for automatic suicide risk detection. Given the vulnerable population, protecting speaker identity is particularly important, as speech itself can reveal personally identifiable information if the data is leaked or maliciously exploited. This work presents the first systematic study of speaker anonymisation for speech-based suicide risk detection. A broad range of anonymisation methods are investigated, including techniques based on traditional signal processing, neural voice conversion, and speech synthesis. A comprehensive evaluation framework is built to assess the trade-off between protecting speaker identity and preserving information essential for suicide risk detection. Results show that combining anonymisation methods that retain complementary information yields detection performance comparable to that of original speech, while achieving protection of speaker identity for vulnerable populations.
>
---
#### [replaced 007] WildScore: Benchmarking MLLMs in-the-Wild Symbolic Music Reasoning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出WildScore，一个用于评估多模态大语言模型在真实音乐符号推理能力的基准。任务是解决MLLMs在音乐分析中的符号推理问题，通过构建真实音乐数据集和多选题形式进行评估。**

- **链接: [https://arxiv.org/pdf/2509.04744v2](https://arxiv.org/pdf/2509.04744v2)**

> **作者:** Gagan Mundada; Yash Vishe; Amit Namburi; Xin Xu; Zachary Novack; Julian McAuley; Junda Wu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, their reasoning abilities in the multimodal symbolic music domain remain largely unexplored. We introduce WildScore, the first in-the-wild multimodal symbolic music reasoning and analysis benchmark, designed to evaluate MLLMs' capacity to interpret real-world music scores and answer complex musicological queries. Each instance in WildScore is sourced from genuine musical compositions and accompanied by authentic user-generated questions and discussions, capturing the intricacies of practical music analysis. To facilitate systematic evaluation, we propose a systematic taxonomy, comprising both high-level and fine-grained musicological ontologies. Furthermore, we frame complex music reasoning as multiple-choice question answering, enabling controlled and scalable assessment of MLLMs' symbolic music understanding. Empirical benchmarking of state-of-the-art MLLMs on WildScore reveals intriguing patterns in their visual-symbolic reasoning, uncovering both promising directions and persistent challenges for MLLMs in symbolic music reasoning and analysis. We release the dataset and code.
>
---
#### [replaced 008] A Lightweight Fourier-based Network for Binaural Speech Enhancement with Spatial Cue Preservation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决轻量级模型与性能之间的平衡问题。提出GAF-Net，通过傅里叶变换和空间线索保留实现高效高保真双耳语音增强。**

- **链接: [https://arxiv.org/pdf/2509.14076v2](https://arxiv.org/pdf/2509.14076v2)**

> **作者:** Xikun Lu; Yujian Ma; Xianquan Jiang; Xuelong Wang; Jinqiu Sang
>
> **备注:** Accepted at IEEE ICASSP 2026
>
> **摘要:** Binaural speech enhancement faces a severe trade-off challenge, where state-of-the-art performance is achieved by computationally intensive architectures, while lightweight solutions often come at the cost of significant performance degradation. To bridge this gap, we propose the Global Adaptive Fourier Network (GAF-Net), a lightweight deep complex network that aims to establish a balance between performance and computational efficiency. The GAF-Net architecture consists of three components. First, a dual-feature encoder combining short-time Fourier transform and gammatone features enhances the robustness of acoustic representation. Second, a channel-independent globally adaptive Fourier modulator efficiently captures long-term temporal dependencies while preserving the spatial cues. Finally, a dynamic gating mechanism is implemented to reduce processing artifacts. Experimental results show that GAF-Net achieves competitive performance, particularly in terms of binaural cues (ILD and IPD error) and objective intelligibility (MBSTOI), with fewer parameters and computational cost. These results confirm that GAF-Net provides a feasible way to achieve high-fidelity binaural processing on resource-constrained devices.
>
---
#### [replaced 009] Etude: Piano Cover Generation with a Three-Stage Approach -- Extract, strucTUralize, and DEcode
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于钢琴伴奏生成任务，旨在解决现有模型结构不一致的问题。提出三阶段模型Etude，通过提取节奏信息和简化tokenization，提升生成音乐的结构和质量。**

- **链接: [https://arxiv.org/pdf/2509.16522v2](https://arxiv.org/pdf/2509.16522v2)**

> **作者:** Tse-Yang Chen; Yuh-Jzer Joung
>
> **摘要:** Piano cover generation aims to automatically transform a pop song into a piano arrangement. While numerous deep learning approaches have been proposed, existing models often fail to maintain structural consistency with the original song, likely due to the absence of beat-aware mechanisms or the difficulty of modeling complex rhythmic patterns. Rhythmic information is crucial, as it defines structural similarity (e.g., tempo, BPM) and directly impacts the overall quality of the generated music. In this paper, we introduce Etude, a three-stage architecture consisting of Extract, strucTUralize, and DEcode stages. By pre-extracting rhythmic information and applying a novel, simplified REMI-based tokenization, our model produces covers that preserve proper song structure, enhance fluency and musical dynamics, and support highly controllable generation through style injection. Subjective evaluations with human listeners show that Etude substantially outperforms prior models, achieving a quality level comparable to that of human composers.
>
---
#### [replaced 010] Enhanced Generative Machine Listener
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文属于音频质量评估任务，旨在提升主观质量预测的准确性。提出GMLv2模型，引入Beta分布损失和更多数据集，以提高与主观评分的相关性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.21463v2](https://arxiv.org/pdf/2509.21463v2)**

> **作者:** Vishnu Raj; Gouthaman KV; Shiv Gehlot; Lars Villemoes; Arijit Biswas
>
> **备注:** Accepted to the 51st IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 4-8 May 2026
>
> **摘要:** We present GMLv2, a reference-based model designed for the prediction of subjective audio quality as measured by MUSHRA scores. GMLv2 introduces a Beta distribution-based loss to model the listener ratings and incorporates additional neural audio coding (NAC) subjective datasets to extend its generalization and applicability. Extensive evaluations on diverse testset demonstrate that proposed GMLv2 consistently outperforms widely used metrics, such as PEAQ and ViSQOL, both in terms of correlation with subjective scores and in reliably predicting these scores across diverse content types and codec configurations. Consequently, GMLv2 offers a scalable and automated framework for perceptual audio quality evaluation, poised to accelerate research and development in modern audio coding technologies.
>
---
#### [replaced 011] Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音生成任务，解决多代码本结构下的依赖问题。通过帧堆叠和局部Transformer提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2509.19592v2](https://arxiv.org/pdf/2509.19592v2)**

> **作者:** Roy Fejgin; Paarth Neekhara; Xuesong Yang; Edresson Casanova; Ryan Langman; Jaehyeon Kim; Subhankar Ghosh; Shehzeen Hussain; Jason Li
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity.
>
---
#### [replaced 012] Adaptive Multimodal Person Recognition: A Robust Framework for Handling Missing Modalities
- **分类: cs.CV; cs.SD; eess.AS; eess.IV**

- **简介: 该论文属于人物识别任务，解决多模态数据缺失问题。提出一种融合手势的鲁棒框架，通过特征与分数级融合及动态适应策略，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.14961v2](https://arxiv.org/pdf/2512.14961v2)**

> **作者:** Aref Farhadipour; Teodora Vukovic; Volker Dellwo; Petr Motlicek; Srikanth Madikeri
>
> **备注:** 9 pages and 8 tables
>
> **摘要:** Person identification systems often rely on audio, visual, or behavioral cues, but real-world conditions frequently result in missing or degraded modalities. To address this challenge, we propose a multimodal person identification framework that utilizes gesture as a situational enhancer to supplement traditional modalities like voice and face. Our model employs a unified hybrid fusion strategy, integrating both feature-level and score-level information to maximize representational richness and decision accuracy. Specifically, it leverages multi-task learning to process modalities independently, followed by cross-attention and gated fusion mechanisms. Finally, a confidence-weighted strategy dynamically adapts to missing data, ensuring that our single classification head achieves optimal performance even in unimodal and bimodal scenarios. We evaluate our method on CANDOR, a newly introduced interview-based multimodal dataset, which we benchmark in this work for the first time. Our results demonstrate that the proposed trimodal system achieves 99.51% Top-1 accuracy on person identification tasks. In addition, we evaluate our model on the VoxCeleb1 dataset as a benchmark and reach 99.92% accuracy in bimodal mode, outperforming conventional approaches. Moreover, we show that our system maintains high accuracy even when one or two modalities are unavailable, making it a robust solution for real-world person recognition applications. The code and data for this work are publicly available.
>
---
#### [replaced 013] SONAR: Self-Distilled Continual Pre-training for Domain Adaptive Audio Representation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频表示学习任务，解决领域自适应中的持续预训练问题。提出SONAR框架，通过自蒸馏和动态扩展解决遗忘与适应难题。**

- **链接: [https://arxiv.org/pdf/2509.15703v2](https://arxiv.org/pdf/2509.15703v2)**

> **作者:** Yizhou Zhang; Yuan Gao; Wangjin Zhou; Zicheng Yuan; Keisuke Imoto; Tatsuya Kawahara
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Self-supervised learning (SSL) on large-scale datasets like AudioSet has become the dominant paradigm for audio representation learning. While the continuous influx of new, unlabeled audio presents an opportunity to enrich these static representations, a naive approach is to retrain the model from scratch using all available data. However, this method is computationally prohibitive and discards the valuable knowledge embedded in the previously trained model weights. To address this inefficiency, we propose SONAR (Self-distilled cONtinual pre-training for domain adaptive Audio Representation), a continual pre-training framework built upon BEATs. SONAR effectively adapts to new domains while mitigating catastrophic forgetting by tackling three key challenges: implementing a joint sampling strategy for new and prior data, applying regularization to balance specificity and generality, and dynamically expanding the tokenizer codebook for novel acoustic patterns. Experiments across four distinct domains demonstrate that our method achieves both high adaptability and robust resistance to forgetting.
>
---
