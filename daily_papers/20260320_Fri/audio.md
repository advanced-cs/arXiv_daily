# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Modeling Overlapped Speech with Shuffles
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音处理任务，解决重叠语音的对齐与说话人识别问题。通过shuffle产品和部分有序有限状态自动机实现单次遍历的多说话人语音对齐与转录。**

- **链接: [https://arxiv.org/pdf/2603.17769](https://arxiv.org/pdf/2603.17769)**

> **作者:** Matthew Wiesner; Samuele Cornell; Alexander Polok; Lucas Ondel Yang; Lukáš Burget; Sanjeev Khudanpur
>
> **摘要:** We propose to model parallel streams of data, such as overlapped speech, using shuffles. Specifically, this paper shows how the shuffle product and partial order finite-state automata (FSAs) can be used for alignment and speaker-attributed transcription of overlapped speech. We train using the total score on these FSAs as a loss function, marginalizing over all possible serializations of overlapping sequences at subword, word, and phrase levels. To reduce graph size, we impose temporal constraints by constructing partial order FSAs. We address speaker attribution by modeling (token, speaker) tuples directly. Viterbi alignment through the shuffle product FSA directly enables one-pass alignment. We evaluate performance on synthetic LibriSpeech overlaps. To our knowledge, this is the first algorithm that enables single-pass alignment of multi-talker recordings. All algorithms are implemented using k2 / Icefall.
>
---
#### [new 002] Few-shot Acoustic Synthesis with Multimodal Flow Matching
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于声学合成任务，解决场景音频生成问题。提出FLAC方法，通过少量样本生成合理房间冲激响应，提升生成效果与数据效率。**

- **链接: [https://arxiv.org/pdf/2603.19176](https://arxiv.org/pdf/2603.19176)**

> **作者:** Amandine Brunetto
>
> **备注:** To appear at CVPR 2026. 23 pages, 16 figures. Project Page: this https URL
>
> **摘要:** Generating audio that is acoustically consistent with a scene is essential for immersive virtual environments. Recent neural acoustic field methods enable spatially continuous sound rendering but remain scene-specific, requiring dense audio measurements and costly training for each environment. Few-shot approaches improve scalability across rooms but still rely on multiple recordings and, being deterministic, fail to capture the inherent uncertainty of scene acoustics under sparse context. We introduce flow-matching acoustic generation (FLAC), a probabilistic method for few-shot acoustic synthesis that models the distribution of plausible room impulse responses (RIRs) given minimal scene context. FLAC leverages a diffusion transformer trained with a flow-matching objective to generate RIRs at arbitrary positions in novel scenes, conditioned on spatial, geometric, and acoustic cues. FLAC outperforms state-of-the-art eight-shot baselines with one-shot on both the AcousticRooms and Hearing Anything Anywhere datasets. To complement standard perceptual metrics, we further introduce AGREE, a joint acoustic-geometry embedding, enabling geometry-consistent evaluation of generated RIRs through retrieval and distributional metrics. This work is the first to apply generative flow matching to explicit RIR synthesis, establishing a new direction for robust and data-efficient acoustic synthesis.
>
---
#### [new 003] ARTT: Augmented Reverberant-Target Training for Unsupervised Monaural Speech Dereverberation
- **分类: eess.AS**

- **简介: 该论文属于语音去混响任务，解决无参考信号下的单通道语音去混响问题。提出ARTT方法，通过两阶段训练提升去混响效果。**

- **链接: [https://arxiv.org/pdf/2603.18485](https://arxiv.org/pdf/2603.18485)**

> **作者:** Siqi Song; Fulin Wu; Zhong-Qiu Wang
>
> **备注:** in submission
>
> **摘要:** Due to the absence of clean reference signals and spatial cues, monaural unsupervised speech dereverberation is a challenging ill-posed inverse problem. To realize it, we propose augmented reverberant-target training (ARTT), which consists of two stages. In the first stage, reverberant-target training (RTT) is proposed to first further reverberate the observed reverberant mixture signal, and then train a deep neural network (DNN) to recover the observed reverberant mixture via discriminative training. Although the target signal to fit is reverberant, we find that the resulting DNN can effectively reduce reverberation. In the second stage, an online self-distillation mechanism based on the mean-teacher algorithm is proposed to further improve dereverberation. Evaluation results demonstrate that ARTT achieves strong unsupervised dereverberation performance, significantly outperforming previous baselines.
>
---
#### [new 004] PCOV-KWS: Multi-task Learning for Personalized Customizable Open Vocabulary Keyword Spotting
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音处理任务，解决个性化关键词检测问题。提出多任务学习框架PCOV-KWS，同时实现关键词识别和说话人验证，提升效率与隐私保护。**

- **链接: [https://arxiv.org/pdf/2603.18023](https://arxiv.org/pdf/2603.18023)**

> **作者:** Jianan Pan; Kejie Huang
>
> **摘要:** As advancements in technologies like Internet of Things (IoT), Automatic Speech Recognition (ASR), Speaker Verification (SV), and Text-to-Speech (TTS) lead to increased usage of intelligent voice assistants, the demand for privacy and personalization has escalated. In this paper, we introduce a multi-task learning framework for personalized, customizable open-vocabulary Keyword Spotting (PCOV-KWS). This framework employs a lightweight network to simultaneously perform Keyword Spotting (KWS) and SV to address personalized KWS requirements. We have integrated a training criterion distinct from softmax-based loss, transforming multi-class classification into multiple binary classifications, which eliminates inter-category competition, while an optimization strategy for multi-task loss weighting is employed during training. We evaluated our PCOV-KWS system in multiple datasets, demonstrating that it outperforms the baselines in evaluation results, while also requiring fewer parameters and lower computational resources.
>
---
#### [new 005] Towards Interpretable Framework for Neural Audio Codecs via Sparse Autoencoders: A Case Study on Accent Information
- **分类: cs.SD**

- **简介: 该论文属于语音编码领域，旨在提升神经音频编解码器（NAC）的可解释性。通过稀疏自编码器分析NAC如何编码口音信息，揭示不同NAC模型的表征特点。**

- **链接: [https://arxiv.org/pdf/2603.18359](https://arxiv.org/pdf/2603.18359)**

> **作者:** Shih-Heng Wang; Tiantian Feng; Aditya Kommineni; Thanathai Lertpetchpun; Bowen Yi; Xuan Shi; Shrikanth Narayanan
>
> **摘要:** Neural Audio Codecs (NACs) are widely adopted in modern speech systems, yet how they encode linguistic and paralinguistic information remains unclear. Improving the interpretability of NAC representations is critical for understanding and deploying them in sensitive applications. Hence, we employ Sparse Autoencoders (SAEs) to decompose dense NAC representations into sparse, interpretable activations. In this work, we focus on a challenging paralinguistic attribute-accent-and propose a framework to quantify NAC interpretability. We evaluate four NAC models under 16 SAE configurations using a relative performance index. Our results show that DAC and SpeechTokenizer achieve the highest interpretability. We further reveal that acoustic-oriented NACs encode accent information primarily in activation magnitudes of sparse representations, whereas phonetic-oriented NACs rely more on activation positions, and that low-bitrate EnCodec variants show higher interpretability.
>
---
#### [new 006] Words at Play: Benchmarking Audio Pun Understanding in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频幽默理解任务，旨在解决音频双关语识别与理解问题。提出首个基准APUN-Bench，评估大音频语言模型，揭示其在定位和解释上的挑战。**

- **链接: [https://arxiv.org/pdf/2603.18678](https://arxiv.org/pdf/2603.18678)**

> **作者:** Yuchen Su; Shaoxin Zhong; Yonghua Zhu; Ruofan Wang; Zijian Huang; Qiqi Wang; Na Zhao; Diana Benavides-Prado; Michael Witbrock
>
> **备注:** The paper is currently under review
>
> **摘要:** Puns represent a typical linguistic phenomenon that exploits polysemy and phonetic ambiguity to generate humour, posing unique challenges for natural language understanding. Within pun research, audio plays a central role in human communication except text and images, while datasets and systematic resources for spoken puns remain scarce, leaving this crucial modality largely underexplored. In this paper, we present APUN-Bench, the first benchmark dedicated to evaluating large audio language models (LALMs) on audio pun understanding. Our benchmark contains 4,434 audio samples annotated across three stages: pun recognition, pun word location and pun meaning inference. We conduct a deep analysis of APUN-Bench by systematically evaluating 10 state-of-the-art LALMs, uncovering substantial performance gaps in recognizing, localizing, and interpreting audio puns. This analysis reveals key challenges, such as positional biases in audio pun location and error cases in meaning inference, offering actionable insights for advancing humour-aware audio intelligence.
>
---
#### [new 007] MOSS-TTS Technical Report
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文介绍MOSS-TTS，一种基于离散音频标记和自回归建模的语音生成模型，解决多语言、开放域的语音合成问题，支持零样本语音克隆和精细控制。**

- **链接: [https://arxiv.org/pdf/2603.18090](https://arxiv.org/pdf/2603.18090)**

> **作者:** Yitian Gong; Botian Jiang; Yiwei Zhao; Yucheng Yuan; Kuangwei Chen; Yaozhou Jiang; Cheng Chang; Dong Hong; Mingshu Chen; Ruixiao Li; Yiyang Zhang; Yang Gao; Hanfu Chen; Ke Chen; Songlin Wang; Xiaogui Yang; Yuqian Zhang; Kexin Huang; ZhengYuan Lin; Kang Yu; Ziqi Chen; Jin Wang; Zhaoye Fei; Qinyuan Cheng; Shimin Li; Xipeng Qiu
>
> **备注:** Project page: this https URL
>
> **摘要:** This technical report presents MOSS-TTS, a speech generation foundation model built on a scalable recipe: discrete audio tokens, autoregressive modeling, and large-scale pretraining. Built on MOSS-Audio-Tokenizer, a causal Transformer tokenizer that compresses 24 kHz audio to 12.5 fps with variable-bitrate RVQ and unified semantic-acoustic representations, we release two complementary generators: MOSS-TTS, which emphasizes structural simplicity, scalability, and long-context/control-oriented deployment, and MOSS-TTS-Local-Transformer, which introduces a frame-local autoregressive module for higher modeling efficiency, stronger speaker preservation, and a shorter time to first audio. Across multilingual and open-domain settings, MOSS-TTS supports zero-shot voice cloning, token-level duration control, phoneme-/pinyin-level pronunciation control, smooth code-switching, and stable long-form generation. This report summarizes the design, training recipe, and empirical characteristics of the released models.
>
---
#### [new 008] How Auditory Knowledge in LLM Backbones Shapes Audio Language Models: A Holistic Evaluation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究LLM在音频任务中的表现，探讨其文本预训练如何影响音频理解。属于音频与语言模型交叉任务，旨在评估LLM的听觉知识及其对下游任务的影响。**

- **链接: [https://arxiv.org/pdf/2603.19195](https://arxiv.org/pdf/2603.19195)**

> **作者:** Ke-Han Lu; Szu-Wei Fu; Chao-Han Huck Yang; Zhehuai Chen; Sung-Feng Huang; Chih-Kai Yang; Yi-Cheng Lin; Chi-Yuan Hsiao; Wenze Ren; En-Pei Hu; Yu-Han Huang; An-Yu Cheng; Cheng-Han Chiang; Yu Tsao; Yu-Chiang Frank Wang; Hung-yi Lee
>
> **备注:** Project website: this https URL
>
> **摘要:** Large language models (LLMs) have been widely used as knowledge backbones of Large Audio Language Models (LALMs), yet how much auditory knowledge they encode through text-only pre-training and how this affects downstream performance remains unclear. We study this gap by comparing different LLMs under two text-only and one audio-grounded setting: (1) direct probing on AKB-2000, a curated benchmark testing the breadth and depth of auditory knowledge; (2) cascade evaluation, where LLMs reason over text descriptions from an audio captioner; and (3) audio-grounded evaluation, where each LLM is fine-tuned into a Large Audio Language Model (LALM) with an audio encoder. Our findings reveal that auditory knowledge varies substantially across families, and text-only results are strongly correlated with audio performance. Our work provides empirical grounding for a comprehensive understanding of LLMs in audio research.
>
---
#### [new 009] ProKWS: Personalized Keyword Spotting via Collaborative Learning of Phonemes and Prosody
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出ProKWS，解决个性化关键词检测问题，融合音素和韵律学习，提升系统适应性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.18024](https://arxiv.org/pdf/2603.18024)**

> **作者:** Jianan Pan; Yuanming Zhang; Kejie Huang
>
> **摘要:** Current keyword spotting systems primarily use phoneme-level matching to distinguish confusable words but ignore user-specific pronunciation traits like prosody (intonation, stress, rhythm). This paper presents ProKWS, a novel framework integrating fine-grained phoneme learning with personalized prosody modeling. We design a dual-stream encoder where one stream derives robust phonemic representations through contrastive learning, while the other extracts speaker-specific prosodic patterns. A collaborative fusion module dynamically combines phonemic and prosodic information, enhancing adaptability across acoustic environments. Experiments show ProKWS delivers highly competitive performance, comparable to state-of-the-art models on standard benchmarks and demonstrates strong robustness for personalized keywords with tone and intent variations.
>
---
#### [new 010] Dual-Model Prediction of Affective Engagement and Vocal Attractiveness from Speaker Expressiveness in Video Learning
- **分类: cs.HC; cs.CV; cs.SD**

- **简介: 该论文属于情感计算任务，旨在通过说话者表达预测观众情感参与度和语音吸引力，解决传统需观众输入的隐私与可扩展性问题。工作包括构建双模型，利用多模态特征进行预测。**

- **链接: [https://arxiv.org/pdf/2603.18758](https://arxiv.org/pdf/2603.18758)**

> **作者:** Hung-Yue Suen; Kuo-En Hung; Fan-Hsun Tseng
>
> **备注:** Preprint. Accepted for publication in IEEE Transactions on Computational Social Systems
>
> **摘要:** This paper outlines a machine learning-enabled speaker-centric Emotion AI approach capable of predicting audience-affective engagement and vocal attractiveness in asynchronous video-based learning, relying solely on speaker-side affective expressions. Inspired by the demand for scalable, privacy-preserving affective computing applications, this speaker-centric Emotion AI approach incorporates two distinct regression models that leverage a massive corpus developed within Massive Open Online Courses (MOOCs) to enable affectively engaging experiences. The regression model predicting affective engagement is developed by assimilating emotional expressions emanating from facial dynamics, oculomotor features, prosody, and cognitive semantics, while incorporating a second regression model to predict vocal attractiveness based exclusively on speaker-side acoustic features. Notably, on speaker-independent test sets, both regression models yielded impressive predictive performance (R2 = 0.85 for affective engagement and R2 = 0.88 for vocal attractiveness), confirming that speaker-side affect can functionally represent aggregated audience feedback. This paper provides a speaker-centric Emotion AI approach substantiated by an empirical study discovering that speaker-side multimodal features, including acoustics, can prospectively forecast audience feedback without necessarily employing audience-side input information.
>
---
#### [new 011] DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型评估任务，旨在解决模型是否真正理解声音信号的问题。通过构建基准DEAF和设计评估框架，分析模型对文本与声音的依赖程度。**

- **链接: [https://arxiv.org/pdf/2603.18048](https://arxiv.org/pdf/2603.18048)**

> **作者:** Jiaqi Xiong; Yunjia Qi; Qi Cao; Yu Zheng; Weisheng Xu; Ziteng Wang; Ruofan Liao; Yutong Zhang; Sichen Liu
>
> **备注:** 14 pages,6 figures
>
> **摘要:** Recent Audio Multimodal Large Language Models (Audio MLLMs) demonstrate impressive performance on speech benchmarks, yet it remains unclear whether these models genuinely process acoustic signals or rely on text-based semantic inference. To systematically study this question, we introduce DEAF (Diagnostic Evaluation of Acoustic Faithfulness), a benchmark of over 2,700 conflict stimuli spanning three acoustic dimensions: emotional prosody, background sounds, and speaker identity. Then, we design a controlled multi-level evaluation framework that progressively increases textual influence, ranging from semantic conflicts in the content to misleading prompts and their combination, allowing us to disentangle content-driven bias from prompt-induced sycophancy. We further introduce diagnostic metrics to quantify model reliance on textual cues over acoustic signals. Our evaluation of seven Audio MLLMs reveals a consistent pattern of text dominance: models are sensitive to acoustic variations, yet predictions are predominantly driven by textual inputs, revealing a gap between high performance on standard speech benchmarks and genuine acoustic understanding.
>
---
#### [new 012] STEP: Detecting Audio Backdoor Attacks via Stability-based Trigger Exposure Profiling
- **分类: cs.CR; cs.LG; cs.SD**

- **简介: 该论文属于音频安全任务，解决后门攻击检测问题。提出STEP方法，通过分析标签稳定性检测触发器，有效识别音频模型中的后门攻击。**

- **链接: [https://arxiv.org/pdf/2603.18103](https://arxiv.org/pdf/2603.18103)**

> **作者:** Kun Wang; Meng Chen; Junhao Wang; Yuli Wu; Li Lu; Chong Zhang; Peng Cheng; Jiaheng Zhang; Kui Ren
>
> **摘要:** With the widespread deployment of deep-learning-based speech models in security-critical applications, backdoor attacks have emerged as a serious threat: an adversary who poisons a small fraction of training data can implant a hidden trigger that controls the model's output while preserving normal behavior on clean inputs. Existing inference-time defenses are not well suited to the audio domain, as they either rely on trigger over-robustness assumptions that fail on transformation-based and semantic triggers, or depend on properties specific to image or text modalities. In this paper, we propose STEP (Stability-based Trigger Exposure Profiling), a black-box, retraining-free backdoor detector that operates under hard-label-only access. Its core idea is to exploit a characteristic dual anomaly of backdoor triggers: anomalous label stability under semantic-breaking perturbations, and anomalous label fragility under semantic-preserving perturbations. STEP profiles each test sample with two complementary perturbation branches that target these two properties respectively, scores the resulting stability features with one-class anomaly detectors trained on benign references, and fuses the two scores via unsupervised weighting. Extensive experiments across seven backdoor attacks show that STEP achieves an average AUROC of 97.92% and EER of 4.54%, substantially outperforming state-of-the-art baselines, and generalizes across model architectures, speech tasks, an open-set verification scenario, and over-the-air physical-world settings.
>
---
#### [new 013] ALIGN: Adversarial Learning for Generalizable Speech Neuroprosthesis
- **分类: cs.LG; cs.NE; cs.SD**

- **简介: 该论文属于脑机接口任务，解决跨会话性能下降问题。通过对抗学习框架ALIGN，提升模型在无标签数据下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.18299](https://arxiv.org/pdf/2603.18299)**

> **作者:** Zhanqi Zhang; Shun Li; Bernardo L. Sabatini; Mikio Aoi; Gal Mishne
>
> **摘要:** Intracortical brain-computer interfaces (BCIs) can decode speech from neural activity with high accuracy when trained on data pooled across recording sessions. In realistic deployment, however, models must generalize to new sessions without labeled data, and performance often degrades due to cross-session nonstationarities (e.g., electrode shifts, neural turnover, and changes in user strategy). In this paper, we propose ALIGN, a session-invariant learning framework based on multi-domain adversarial neural networks for semi-supervised cross-session adaptation. ALIGN trains a feature encoder jointly with a phoneme classifier and a domain classifier operating on the latent representation. Through adversarial optimization, the encoder is encouraged to preserve task-relevant information while suppressing session-specific cues. We evaluate ALIGN on intracortical speech decoding and find that it generalizes consistently better to previously unseen sessions, improving both phoneme error rate and word error rate relative to baselines. These results indicate that adversarial domain alignment is an effective approach for mitigating session-level distribution shift and enabling robust longitudinal BCI decoding.
>
---
#### [new 014] EgoAdapt: Enhancing Robustness in Egocentric Interactive Speaker Detection Under Missing Modalities
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文属于"Talking to Me"（TTM）任务，旨在解决egocentric场景下说话人检测问题。针对视觉数据缺失、头部方向忽略和背景噪声等问题，提出EgoAdapt框架，融合多模态信息提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.18082](https://arxiv.org/pdf/2603.18082)**

> **作者:** Xinyuan Qian; Xinjia Zhu; Alessio Brutti; Dong Liang
>
> **摘要:** TTM (Talking to Me) task is a pivotal component in understanding human social interactions, aiming to determine who is engaged in conversation with the camera-wearer. Traditional models often face challenges in real-world scenarios due to missing visual data, neglecting the role of head orientation, and background noise. This study addresses these limitations by introducing EgoAdapt, an adaptive framework designed for robust egocentric "Talking to Me" speaker detection under missing modalities. Specifically, EgoAdapt incorporates three key modules: (1) a Visual Speaker Target Recognition (VSTR) module that captures head orientation as a non-verbal cue and lip movement as a verbal cue, allowing a comprehensive interpretation of both verbal and non-verbal signals to address TTM, setting it apart from tasks focused solely on detecting speaking status; (2) a Parallel Shared-weight Audio (PSA) encoder for enhanced audio feature extraction in noisy environments; and (3) a Visual Modality Missing Awareness (VMMA) module that estimates the presence or absence of each modality at each frame to adjust the system response this http URL evaluations on the TTM benchmark of the Ego4D dataset demonstrate that EgoAdapt achieves a mean Average Precision (mAP) of 67.39% and an Accuracy (Acc) of 62.01%, significantly outperforming the state-of-the-art method by 4.96% in Accuracy and 1.56% in mAP.
>
---
#### [new 015] DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出DiscoPhon，一个用于评估从离散语音单元中无监督发现音素的多语言基准。任务是音素发现，解决如何在无监督情况下准确识别语音中的音素结构。工作包括构建数据集、提供基线模型并评估性能。**

- **链接: [https://arxiv.org/pdf/2603.18612](https://arxiv.org/pdf/2603.18612)**

> **作者:** Maxime Poli; Manel Khentout; Angelo Ortiz Tandazo; Ewan Dunbar; Emmanuel Chemla; Emmanuel Dupoux
>
> **备注:** 6 pages, 2 figures. Submitted to Interspeech 2026
>
> **摘要:** We introduce DiscoPhon, a multilingual benchmark for evaluating unsupervised phoneme discovery from discrete speech units. DiscoPhon covers 6 dev and 6 test languages, chosen to span a wide range of phonemic contrasts. Given only 10 hours of speech in a previously unseen language, systems must produce discrete units that are mapped to a predefined phoneme inventory, through either a many-to-one or a one-to-one assignment. The resulting sequences are evaluated for unit quality, recognition and segmentation. We provide four pretrained multilingual HuBERT and SpidR baselines, and show that phonemic information is available enough in current models for derived units to correlate well with phonemes, though with variations across languages.
>
---
## 更新

#### [replaced 001] Evaluating Hallucinations in Audio-Visual Multimodal LLMs with Spoken Queries under Diverse Acoustic Conditions
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究语音查询下多模态大模型的幻觉问题，提出RePOPE-Spk数据集，验证语音输入会加剧幻觉现象，为可靠语音系统提供新方向。**

- **链接: [https://arxiv.org/pdf/2510.08581](https://arxiv.org/pdf/2510.08581)**

> **作者:** Hansol Park; Hoseong Ahn; Junwon Moon; Yejin Lee; Kyuhong Shim
>
> **备注:** Submitted to Interspeech2026
>
> **摘要:** Hallucinations in multimodal models have been extensively studied using benchmarks that probe reliability in image-text query settings. However, the effect of spoken queries on multimodal hallucinations remains largely unexplored, despite the growing role of voice interfaces. In this paper, we introduce a systematic pipeline that converts existing multimodal hallucination benchmarks into spoken-query versions while preserving the original tasks and labels. We instantiate this pipeline on RePOPE and release RePOPE-Spk, where all queries are provided as spoken audio under diverse input conditions. Experimental results show that hallucinations escalate when queries are spoken rather than written: error rates increase by 3-6% with clean speech and by up to 30% under environmental noise. Furthermore, many-shot prompting and chain-of-thought reasoning provide only partial mitigation. Our findings motivate new directions for building reliable voice interface systems and evaluations.
>
---
#### [replaced 002] Investigating Faithfulness in Large Audio Language Models
- **分类: cs.LG; eess.AS**

- **简介: 该论文属于多模态模型研究任务，旨在解决LALMs中CoT解释的可信度问题。通过定义评估标准和构建基准，分析模型推理与音频输入的关联性。**

- **链接: [https://arxiv.org/pdf/2509.22363](https://arxiv.org/pdf/2509.22363)**

> **作者:** Pooneh Mousavi; Lovenya Jain; Mirco Ravanelli; Cem Subakan
>
> **摘要:** Large Audio Language Models (LALMs) integrate audio encoders with pretrained Large Language Models to perform complex multimodal reasoning tasks. While these models can generate Chain-of-Thought (CoT) explanations, the faithfulness of these reasoning chains remains unclear. In this work, we propose a systematic framework to evaluate CoT faithfulness in LALMs with respect to both the input audio and the final model prediction. We define three criteria for audio faithfulness: hallucination-free, holistic, and attentive listening. We also introduce a benchmark based on both audio and CoT interventions to assess faithfulness. Experiments on Audio Flamingo 3 and Qwen2.5-Omni suggest a potential multimodal disconnect: reasoning often aligns with the final prediction but is not always strongly grounded in the audio and can be vulnerable to hallucinations or adversarial perturbations.
>
---
#### [replaced 003] MPDR Beamforming for Almost-Cyclostationary Processes
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声学噪声抑制任务，针对周期性噪声源提出cMPDR波束成形方法，解决传统方法忽略频谱相关性的问题。通过利用频谱和空间相关性，提升降噪效果。**

- **链接: [https://arxiv.org/pdf/2510.18391](https://arxiv.org/pdf/2510.18391)**

> **作者:** Giovanni Bologni; Martin Bo Møller; Richard Heusdens; Richard C. Hendriks
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Conventional acoustic beamformers typically assume short-time stationarity and process frequency bins independently, ignoring inter-frequency correlations. This is suboptimal for almost-periodic noise sources such as engines, fans, and musical instruments: these signals are better modeled as (almost) cyclostationary (ACS) processes with statistically correlated spectral components. This paper introduces the cyclic minimum power distortionless response (cMPDR) beamformer, which extends the conventional MPDR to jointly exploit spatial and spectral correlations. Building on frequency-shifted (FRESH) filtering, it suppresses noise components that are coherent across harmonically related frequencies, reducing residual noise beyond what spatial filtering alone achieves. To address inharmonicity, where partials deviate from exact integer multiples of a fundamental frequency, we estimate resonant frequencies from a periodogram and derive frequency shifts from their pairwise spacing. Theoretical analysis yields closed-form expressions for residual noise and proves that output power decreases monotonically with the number of cyclic components. Experiments on synthetic harmonic noise and real UAV motor recordings confirm these findings: in low-SNR scenarios, the cMPDR achieves up to 5dB improvement in SI-SDR over the MPDR, yields consistent STOI gains, and remains effective with a single microphone. When spectral correlation is absent, the method reduces to conventional MPDR and does not degrade performance. These results suggest that cyclic processing is a viable direction for acoustic noise reduction that deserves further investigation. Code is available at this https URL.
>
---
#### [replaced 004] Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于多语言语音识别任务，旨在解决Speech-LLM在数据不平衡下的稳定性与可塑性矛盾。提出Zipper-LoRA框架，动态融合共享与语言特定参数，提升低资源场景性能。**

- **链接: [https://arxiv.org/pdf/2603.17558](https://arxiv.org/pdf/2603.17558)**

> **作者:** Yuxiang Mei; Delai Qiu; Shengping Liu; Jiaen Liang; Yanhua Long
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Speech Large Language Models (Speech-LLMs) have emerged as a powerful approach for automatic speech recognition (ASR) by aligning speech encoders with large language models. However, adapting these systems to multilingual settings with imbalanced data distributions remains challenging. In such scenarios, a stability-plasticity dilemma often arises: fully shared Parameter-Efficient Fine-Tuning (PEFT) can cause negative inter-lingual interference for under-represented languages, while fully language-specific tuning limits the cross-lingual beneficial knowledge transfer needed for low-resource tasks. To address this, we propose Zipper-LoRA, a novel rank-level decoupling framework with three variants (Static, Hard, and Soft) that dynamically synthesizes LoRA updates from shared and language-specific subspaces. By using a lightweight language-conditioned router, Zipper-LoRA dynamically controls the contribution of each subspace at the LoRA rank level, enabling fine-grained sharing where languages are compatible and strict decoupling when conflicts occur. To further stabilize optimization under imbalanced data, we propose a two-stage training strategy with an Initial-B warm start that significantly accelerates convergence. Experiments on a 12-language mixed-resource setting show that Zipper-LoRA consistently outperforms both fully shared and independent baselines, particularly in extremely low-resource scenarios. Moreover, we demonstrate that these gains are robust across both chunked and non-chunked encoder configurations, confirming the framework's reliability for practical, large-scale multilingual ASR. Our code and data will be available at this https URL for reproducibility.
>
---
#### [replaced 005] Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音生物识别任务，解决性别相关的性能差异问题。提出Fair-Gate框架，通过风险外推和互补门机制提升公平性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.11360](https://arxiv.org/pdf/2603.11360)**

> **作者:** Yangyang Qu; Todisco Massimiliano; Galdi Chiara; Evans Nicholas
>
> **摘要:** Voice biometric systems can exhibit sex-related performance gaps even when overall verification accuracy is strong. We attribute these gaps to two practical mechanisms: (i) demographic shortcut learning, where speaker classification training exploits spurious correlations between sex and speaker identity, and (ii) feature entanglement, where sex-linked acoustic variation overlaps with identity cues and cannot be removed without degrading speaker discrimination. We propose Fair-Gate, a fairness-aware and interpretable risk-gating framework that addresses both mechanisms in a single pipeline. Fair-Gate applies risk extrapolation to reduce variation in speaker-classification risk across proxy sex groups, and introduces a local complementary gate that routes intermediate features into an identity branch and a sex branch. The gate provides interpretability by producing an explicit routing mask that can be inspected to understand which features are allocated to identity versus sex-related pathways. Experiments on VoxCeleb1 show that Fair-Gate improves the utility--fairness trade-off, yielding more sex-fair ASV performance under challenging evaluation conditions.
>
---
#### [replaced 006] GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR
- **分类: cs.SD**

- **简介: 该论文属于多说话人自动语音识别任务，旨在解决重叠语音识别困难的问题。提出GLAD架构，通过全局-局部融合的动态专家混合机制提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2509.13093](https://arxiv.org/pdf/2509.13093)**

> **作者:** Yujie Guo; Jiaming Zhou; Yuhang Jia; Shiwan Zhao; Yong Qin
>
> **备注:** This paper has been submitted to Interspeech 2026 for review
>
> **摘要:** End-to-end multi-talker automatic speech recognition (MTASR) faces significant challenges in accurately transcribing overlapping speech. A critical bottleneck is that speaker-specific acoustic characteristics, which are essential for distinguishing overlapping speech, are often diluted in deep network layers. To address this, we propose the Global-Local Aware Dynamic Mixture-of-Experts (GLAD) architecture. GLAD introduces a novel routing mechanism that dynamically fuses speaker-aware global context with fine-grained local acoustic details to adaptively guide expert selection. Experiments on the LibriSpeechMix and CH109 datasets demonstrate that GLAD significantly outperforms existing Serialized Output Training (SOT)-based MTASR approaches, exhibiting exceptional robustness in challenging, high-overlap scenarios. To the best of our knowledge, this is the first work to apply a global-local fusion MoE strategy to MTASR.
>
---
#### [replaced 007] Group-Aware Partial Model Merging for Children's Automatic Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于儿童自动语音识别任务，旨在解决成人预训练模型在儿童语音上表现不佳的问题。通过聚类和部分微调实现模型合并，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2511.23098](https://arxiv.org/pdf/2511.23098)**

> **作者:** Thomas Rolland; Alberto Abad
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** While supervised fine-tuning of adult pre-trained models for children's ASR has shown promise, it often fails to capture group-specific characteristics and variations among children. To address this, we introduce GRoup-Aware PARtial model Merging, a parameter-efficient approach that combines unsupervised clustering, partial fine-tuning, and model merging. Our approach adapts adult-pre-trained models to children by first grouping the children's data based on acoustic similarity. Each group is used to partially fine-tune an adult pre-trained model, and the resulting models are merged at the parameter level. Experiments conducted on the MyST children's speech corpus indicate that GRAPAM achieves a relative WER improvement of 6%, using the same amount of data, outperforming full fine-tuning while training fewer parameters.
>
---
#### [replaced 008] DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出DeSTA2.5-Audio，解决LALM在音频感知与语言能力间的平衡问题，通过自生成跨模态对齐策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2507.02768](https://arxiv.org/pdf/2507.02768)**

> **作者:** Ke-Han Lu; Zhehuai Chen; Szu-Wei Fu; Chao-Han Huck Yang; Sung-Feng Huang; Chih-Kai Yang; Chee-En Yu; Chun-Wei Chen; Wei-Chih Chen; Chien-yu Huang; Yi-Cheng Lin; Yu-Xiang Lin; Chi-An Fu; Chun-Yi Kuan; Wenze Ren; Xuanjun Chen; Wei-Ping Huang; En-Pei Hu; Tzu-Quan Lin; Yuan-Kuei Wu; Kuan-Po Huang; Hsiao-Ying Huang; Huang-Cheng Chou; Kai-Wei Chang; Cheng-Han Chiang; Boris Ginsburg; Yu-Chiang Frank Wang; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech and Language Processing (TASLP). Model and code available at: this https URL
>
> **摘要:** We introduce DeSTA2.5-Audio, a general-purpose Large Audio Language Model (LALM) designed for robust auditory perception and instruction-following. Recent LALMs augment Large Language Models (LLMs) with auditory capabilities by training on large-scale audio-instruction datasets. However, existing LALMs have often suffered from the catastrophic forgetting of the LLM's original abilities. Therefore, balancing knowledge retention and audio perception has become a critical challenge. To address this, we revisit the data construction pipeline and propose a self-generated cross-modal alignment strategy in which the backbone LLM generates its own training targets, named DeSTA. This approach aims at preserving the LLM's native language proficiency thereby enabling zero-shot generalization without task-specific tuning. We construct DeSTA-AQA5M, a large-scale, task-agnostic dataset containing 5 million training samples derived from 7,000 hours of audio spanning 50 diverse datasets, including speech, environmental sounds, and music. DeSTA2.5-Audio achieves state-of-the-art or competitive performance across a wide range of audio-language benchmarks, including Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, and VoiceBench. Comprehensive comparative studies demonstrate that our self-generated strategy outperforms existing training strategies. Our findings underscore the importance of carefully designed data construction in LALM development and offer practical insights for building robust, general-purpose LALMs.
>
---
#### [replaced 009] Affect Decoding in Phonated and Silent Speech Production from Surface EMG
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于情感解码任务，旨在通过表面肌电信号（sEMG）识别情绪，解决语音与无声言语中情感表达的关联问题。研究构建数据集并评估不同特征和模型的解码效果。**

- **链接: [https://arxiv.org/pdf/2603.11715](https://arxiv.org/pdf/2603.11715)**

> **作者:** Simon Pistrosch; Kleanthis Avramidis; Zhao Ren; Tiantian Feng; Jihwan Lee; Monica Gonzalez-Machorro; Anton Batliner; Tanja Schultz; Shrikanth Narayanan; Björn W. Schuller
>
> **摘要:** The expression of affect is integral to spoken communication, yet, its link to underlying articulatory execution remains unclear. Measures of articulatory muscle activity such as EMG could reveal how speech production is modulated by emotion alongside acoustic speech analyses. We investigate affect decoding from facial and neck surface electromyography (sEMG) during phonated and silent speech production. For this purpose, we introduce a dataset comprising 2,780 utterances from 12 participants across 3 tasks, on which we evaluate both intra- and inter-subject decoding using a range of features and model embeddings. Our results reveal that EMG representations reliably discriminate frustration with up to 0.845 AUC, and generalize well across articulation modes. Our ablation study further demonstrates that affective signatures are embedded in facial motor activity and persist in the absence of phonation, highlighting the potential of EMG sensing for affect-aware silent speech interfaces.
>
---
