# 音频 cs.SD;  eess.AS

- **最新发布 32 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] UniVoice: A Unified Model for Speech and Singing Voice Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出UniVoice，解决语音和歌唱合成统一建模问题。通过分解条件为内容、旋律和音色，实现两者生成的协同优化。**

- **链接: [https://arxiv.org/pdf/2606.05852](https://arxiv.org/pdf/2606.05852)**

> **作者:** Junjie Zheng; Huixin Xue; Shihong Ren; Chaofan Ding; Hao Liu; Zihao Chen
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Text-to-speech (TTS) and singing voice synthesis (SVS) both aim to generate human vocal audio from symbolic inputs, but they impose different requirements on the generation process. Speech generation relies on flexible, language-driven prosody, whereas singing generation requires explicit melody control and accurate rhythmic alignment. This mismatch makes it challenging to train a single model that can generate both natural speech and controllable singing, since melody-related conditions should strongly constrain singing but should not restrict speech prosody. We present UniVoice, a unified speech and singing voice generation framework based on conditional flow matching. Instead of using a single undifferentiated conditioning representation, UniVoice factorizes the condition into content, melody, and timbre, which are encoded by modality-appropriate encoders and consumed by a shared Diffusion Transformer (DiT) backbone. For singing, the melody condition is represented by MIDI note sequences; for speech, it is replaced with a learned null melody token, allowing the model to infer prosody from linguistic and acoustic context. This design preserves explicit melody control for singing while avoiding the need to impose melody constraints on speech. We further analyze the null melody token as an approximation to melody marginalization in the conditional flow. Trained on 30k hours of speech and 35k hours of singing data, UniVoice achieves a speech PER of 5.26\%, comparable to dedicated TTS systems such as F5-TTS (5.21\%) and CosyVoice3 (5.30\%). On singing generation, UniVoice achieves a PER of 16.22\%, outperforming the unified baseline Vevo1.5 (24.72\%).
>
---
#### [new 002] Enhancing Audio Captioning with Auxiliary AudioSet Semantics
- **分类: eess.AS**

- **简介: 该论文属于音频描述生成任务，旨在解决模型依赖大模型和词选择不确定性问题。通过引入AudioSet语义增强音频表示，设计轻量级解码器提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2606.05717](https://arxiv.org/pdf/2606.05717)**

> **作者:** Shubham Gupta; Adarsh Arigala; Sri Rama Murty Kodukula
>
> **摘要:** Automatic Audio Captioning (AAC) seeks to generate natural language descriptions of complex acoustic scenes, bridging auditory perception and language understanding. However, word-selection indeterminacy and increasing reliance on large-scale sequence-to-sequence or LLM-based models limit practical deployment. We propose a resource-efficient AAC framework that explicitly grounds caption generation in auxiliary AudioSet semantics. Frame-level acoustic representations extracted using a ConvNeXt encoder are augmented with top-$K$ predicted AudioSet keywords, providing structured contextual cues for decoding. A compact six-layer BART-style decoder conditions on this joint acoustic-semantic representation, enabling caption generation without LLM-scale decoding. The proposed design balances semantic grounding and computational efficiency within a compact architecture. Evaluations on Clotho V2 and AudioCaps confirm competitive caption quality under practical deployment constraints.
>
---
#### [new 003] Age-Aware Adapter Tuning for Children's Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于儿童语音识别任务，旨在解决儿童语音差异大、年龄依赖性强的问题。通过设计年龄感知的适配器，提升模型对不同年龄段儿童语音的适应能力。**

- **链接: [https://arxiv.org/pdf/2606.05440](https://arxiv.org/pdf/2606.05440)**

> **作者:** Jialu Li
>
> **备注:** Our code is available at this https URL
>
> **摘要:** Children's automatic speech recognition (ASR) remains challenging because child speech differs from adult speech and varies substantially across developmental stages. While adapter tuning provides a promising way to adapt large pretrained ASR models to children's speech, a single shared child adapter may not fully capture age-dependent variation. In this work, we present one of the first systematic studies of age-aware adapter tuning for child ASR, focusing on speech from children aged 3--12 and older years. We propose age-specialized adapters trained separately for different age groups and compare them with a unified age-conditioned FiLM adapter. With ground-truth age routing, age-specialized adapters improve over the standard shared child adapter baseline from 12.6% to 12.3% overall word error rate (WER) and from 18.4% to 17.6% macro WER, while consistently improving WER for all age groups. We further show that predicted-age routing remains close to ground-truth routing, achieving 12.3% overall WER and 17.8% macro WER without ground-truth age labels at inference. In contrast, unified FiLM conditioning provides smaller gains, indicating that a single unified adapter may be insufficient to capture developmental variation in child speech.
>
---
#### [new 004] Revisiting Lexicon Evaluation in Unsupervised Word Discovery
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于无监督词发现任务，旨在解决词典评估的公平性问题。研究指出现有评估指标存在偏差，提出两种改进指标以更准确衡量词典质量。**

- **链接: [https://arxiv.org/pdf/2606.06183](https://arxiv.org/pdf/2606.06183)**

> **作者:** Simon Malan; Danel Slabbert; Herman Kamper
>
> **备注:** 6 figures
>
> **摘要:** Building a lexicon from discovered word-like units is a central goal in zero-resource speech processing. But do our evaluations provide a trustworthy indication of lexicon quality? A common metric, normalized edit distance, averages the phoneme edit distances between discovered units in each cluster. We show that this metric has an inherent bias toward the quality of large clusters, inhibiting fair evaluation. Moreover, it ignores how well true classes are distributed across clusters. Based on established theory in clustering literature, we propose two metrics that address these shortcomings: a modified metric that weighs cluster size when assessing within-cluster consistency, and an inverse metric that assesses how true words are spread across clusters. Through experiments on synthetic and real-world lexicons, we demonstrate that combined, these metrics are: (1) more closely correlated with how similar a lexicon is to the ground-truth distribution, and (2) more robust to biases that skew lexicon evaluations.
>
---
#### [new 005] M2S-AVSR: Modality-aware Multi-view Self-supervised Representation for Robust Audio-Visual Speech Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频-视觉语音识别任务，旨在提升复杂场景下的识别鲁棒性。提出M2S-AVSR框架，通过多视角自监督学习和模态感知融合，解决视角变化、音视频不同步等问题。**

- **链接: [https://arxiv.org/pdf/2606.05763](https://arxiv.org/pdf/2606.05763)**

> **作者:** Fei Su; Cancan Li; Juan Liu; Ming Li
>
> **备注:** submitted to IEEE Transactions on Audio, Speech, and Language Processing
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) enhances speech recognition robustness by leveraging visual cues, while real-world scenarios remain challenging due to viewpoint variation, audio distortion, and visual occlusion, which degrade modality quality and increase audio-visual asynchrony. In this paper, we propose a novel Modality-aware Multi-view Self-supervised representation framework for robust Audio-Visual Speech Recognition (M2S-AVSR). First, we introduce a multi-view representation learning encoder to learn view-invariant visual speech representations. Next, we employ a modality-aware module that explicitly models modality quality and cross-modal synchrony to perform fine-grained modality-aware fusion, enabling fine-grained visual information injection during decoding. In addition, we present AISHELL8-RealScene, a public multi-scenario, multi-view conversational audio-visual dataset recorded in real-world environments, and establish a speech recognition benchmark on it. Experiments on English and Mandarin benchmarks demonstrate the effectiveness of the proposed method under challenging conditions. On LRS3, M2S-AVSR achieves up to 29.4% relative improvement under viewpoint perturbation and visual degradation settings. Our method also achieves new state-of-the-art performance on the MISP2021-AVSR test set. On AISHELL8-RealScene, it achieves the best result in outdoor scenes. The proposed method and dataset provide useful support for future research on robust speech and multimodal tasks under realistic conditions.
>
---
#### [new 006] VoCodec: A Low-bitrate Streamable Neural Speech Codec with Voicing-driven Quantization
- **分类: eess.AS**

- **简介: 该论文提出VoCodec，一种低比特率流式神经语音编解码器，解决传统方法比特率浪费问题。通过声调驱动量化，提升编码效率。**

- **链接: [https://arxiv.org/pdf/2606.05892](https://arxiv.org/pdf/2606.05892)**

> **作者:** Xiao-Hang Jiang; Yang Ai; Rui-Chen Zheng; Li-Rong Dai; Zhen-Hua Ling; Ji Wu
>
> **备注:** Accepted to INTERSPEECH 2026
>
> **摘要:** Neural speech codecs are key to speech transmission and storage, but most use uniform quantization across frames, allocating the same bitrate regardless of content and wasting bits. We propose VoCodec, a low-bitrate streamable neural speech codec with voicing-driven quantization that assigns higher bitrate to voiced frames and lower bitrate to unvoiced frames according to perceptual sensitivity. VoCodec embeds a voicing detector in a fully causal encoder-quantizer-decoder neural coding framework, using residual scalar-vector quantization for voiced frames and simple scalar quantization for unvoiced ones. Experiments show that on the LibriTTS dataset at a 16 kHz sampling rate, VoCodec outperforms baseline neural speech codecs even at a bitrate as low as 1.1 kbps. Our further experiments also confirm that introducing voicing-driven quantization can effectively reduce the bitrate by approximately 27% compared with uniform quantization strategy.
>
---
#### [new 007] SB-RF: Schrödinger Bridge Rectified Flow for One-Step Robust Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，解决生成模型多步推理效率低的问题。提出SB-RF框架，结合熵正则化最优传输与修正流，实现单步高质量语音增强。**

- **链接: [https://arxiv.org/pdf/2606.05575](https://arxiv.org/pdf/2606.05575)**

> **作者:** Caixia Lu; Xueyang Lv; Penglong Hu; Jiaming Xu
>
> **摘要:** Generative models have shown impressive results in speech enhancement but often suffer from multi-step inference. We propose SB-RF, a one-step generative framework integrating Rectified Flow (RF) with Schrödinger Bridge (SB) theory. SB-RF constructs a conditional bridge between clean and noisy speech distributions via entropy-regularized optimal transport. By aligning SB trajectories with the optimal transport geodesic through the velocity-matching objective of RF, SB-RF enables high-quality enhancement with one-step generation. Experiments demonstrate that SB-RF achieves leading performance among generative methods on the VoiceBank-DEMAND benchmark. Furthermore, to fully assess performance in challenging real-world scenarios, we evaluate SB-RF on a simulated low signal-to-noise ratio test set using an expanded training dataset. Under these conditions, SB-RF exhibits strong and competitive robustness with high efficiency, validating its potential for real-world applications.
>
---
#### [new 008] GLASS: GRPO-Trained LoRA for Acoustic Style Steering in Zero-Shot Text-to-Speech
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出GLASS，用于零样本文本转语音中的声学风格控制。解决风格与说话人身份纠缠问题，通过奖励学习实现风格调节。**

- **链接: [https://arxiv.org/pdf/2606.05889](https://arxiv.org/pdf/2606.05889)**

> **作者:** Jaehoon Kang; Yejin Lee; Kyuhong Shim
>
> **摘要:** We propose GLASS, a framework for composable acoustic style control in zero-shot autoregressive text-to-speech (TTS) that learns controls from post-generation rewards rather than style labels. In zero-shot TTS, a speaker prompt often entangles speaker identity with prosodic attributes such as speaking rate and pitch, making it difficult to change style without changing the prompt itself. GLASS instead treats each acoustic attribute as a reward-defined control direction. For each control axis, GLASS freezes the TTS backbone and trains one lightweight LoRA adapter with Group Relative Policy Optimization (GRPO), using speech-token length and mean F0 as style rewards and WER as an intelligibility anchor. Because each control is represented as a LoRA weight update, independently trained adapters can be swapped, interpolated, and composed through linear LoRA arithmetic without retraining the backbone. Experiments on speaking rate and pitch control show targeted style shifts while preserving naturalness, speaker similarity, and intelligibility, and demonstrate smooth interpolation and multi-axis composition across independently trained adapters.
>
---
#### [new 009] Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决情感表达控制问题。通过分析发现说话人向量是情感韵律的主要载体，提出一种无需训练的语义向量算术方法，实现跨语言情感控制。**

- **链接: [https://arxiv.org/pdf/2606.05367](https://arxiv.org/pdf/2606.05367)**

> **作者:** Daniel Oliveira de Brito; Arnaldo Candido Junior
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We investigate whether task-vector arithmetic, successful for cross-speaker emotional intensity control in modular text-to-speech (TTS), transfers to large-scale TTS systems built on language-model backbones with in-context learning (LM-TTS). Through a systematic elimination study over four progressively narrower operands on Qwen3-TTS-12Hz-1.7B - model weights via LoRA fine-tuning, continuous codec embeddings, discrete codec tokens, and the speaker embedding (x-vector) produced by an ECAPA-TDNN encoder jointly trained with the synthesis backbone - we localize the dominant carrier of emotional prosody to the x-vector. Building on this finding, we propose a training-free method based on centroid arithmetic in x-vector space: an emotion direction $\tau = \mathbb{E}_i[x(s_i,\text{emo})] -\mathbb{E}_i[x(s_i,\text{neutral})]$ applied to an unseen target speaker as $x_{\text{new}} = x(\text{target},\text{neutral}) + \alpha\cdot\tau$. Using ESD (English) as the $\tau$ source and emoUERJ (Brazilian Portuguese) as a cross-lingual ground-truth target, we observe average gains of $+0.29$ in emotion2vec cosine over the ICL baseline on English held-out speakers and $+0.09$ on Brazilian Portuguese held-out speakers, while largely preserving identity (WavLM SECS $\gtrsim 0.88$ for the multi-speaker $\tau$ variant) and intelligibility (WER $\approx 0$ in PT-BR). These results offer initial evidence that the reported incompatibility of centroid-arithmetic style control with token-based TTS architectures may be circumvented when the arithmetic operates on the speaker embedding.
>
---
#### [new 010] Do speech foundation models perceive speaker similarity as humans do?
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，研究语音基础模型的说话人嵌入是否与人类感知的说话人相似性一致。通过对比模型距离与人类评分，探索影响模型感知性能的因素。**

- **链接: [https://arxiv.org/pdf/2606.05739](https://arxiv.org/pdf/2606.05739)**

> **作者:** Minoru Kishi; Hayato Yagi; Shinnosuke Takamichi; Yuki Saito
>
> **备注:** Accepted by INTERSPEECH 2026
>
> **摘要:** This study presents a comparative analysis between the speaker embeddings of speech foundation models and human subjective perception of speaker similarity. Human listeners have the ability to judge speaker similarity on a continuous scale discerning how similar two voices are. In contrast, speech foundation models embed speaker characteristics into numerical representation. However, a question remains: does the numerical distance between speaker embeddings in these models truly align with the similarity perceived by humans? To address this, we conduct a comprehensive investigation using more than 40 models to compare model-derived distances with human-perceived similarity scores. Furthermore, we identify which factors in model configuration contribute most to a speaker embedding that mirrors human perception. Our findings provide insights for the development of more perceptually grounded speech foundation models.
>
---
#### [new 011] Sound Effects Dataset Unification With the Universal Category System
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于声音分类任务，旨在解决SFX数据集标签不统一的问题。提出一种基于UCS的框架，实现数据集的统一标注与合并。**

- **链接: [https://arxiv.org/pdf/2606.05571](https://arxiv.org/pdf/2606.05571)**

> **作者:** Jun Woo Beck; Alexander Lerch
>
> **备注:** DAFx 2026 camera-ready version
>
> **摘要:** Sound effects (SFX) datasets and libraries often employ distinct tagging schemes, taxonomies, and metadata structures. This creates challenges for research on SFX classification and generation because incompatible taxonomies lead to siloed datasets that might require individualized approaches, result in non-comparable outcomes, and prevent data merging strategies. We propose a modular dataset relabeling framework that adopts the Universal Category System (UCS), an industry-standard hierarchical taxonomy for sound effects, as a shared structural foundation. This open-source framework enables us (i) to convert tags of existing datasets to UCS with a rule-based multi-stage pipeline and conflict resolution to achieve high automatic conversion rates, (ii) to suggest a stratified dataset split for the new labels, and (iii) to combine multiple datasets. To showcase the practical utility, we introduce the EnvSound-UCS dataset, a publicly available unified UCS-compliant dataset of environmental sounds with 58,057 sound clips from three sources: AudioSet, FSD50K, and ESC-50.
>
---
#### [new 012] CoSTA: Cognitive-State-Conditioned TTS Data Augmentation Using ASR Transcripts for Alzheimer's Disease Detection
- **分类: eess.AS**

- **简介: 该论文属于阿尔茨海默病检测任务，针对病理语音数据不足的问题，提出CoSTA框架，利用TTS生成增强数据，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2606.06170](https://arxiv.org/pdf/2606.06170)**

> **作者:** Yin-Long Liu; Yuanchao Li; Yiming Wang; Yue Li; Rui Feng; Jiaxin Chen; Shaobo Liu; Liu He; Yuang Chen; Jiahong Yuan; Zhen-Hua Ling
>
> **备注:** Accepted by Interspeech 2026
>
> **摘要:** Speech-based Alzheimer's Disease (AD) detection is constrained by scarce pathological speech data. To address this, we propose CoSTA, a Text-to-Speech (TTS)-based data augmentation framework. Specifically, we first develop two Cognitive-State-Conditioned (CS-Cond) TTS models by adapting CosyVoice2 and F5-TTS to synthesize speech with distinct AD and Healthy Control characteristics. Furthermore, by constructing a transcript pool comprising Manual Transcripts (MT) and 36 Automatic Speech Recognition (ASR) transcripts, we investigate the impact of text sources on TTS-based augmentation. We also perform augmentation-factor analysis and test-time augmentation. Experiments on the ADReSS dataset show that CS-Cond TTS significantly improves synthetic speech utility, and ASR-driven augmentation frequently outperforms MT-driven augmentation. Finally, CoSTA yields a 4.16% gain over the baseline, achieving an audio-only accuracy of 85.83% on the ADReSS test set and outperforming prior methods.
>
---
#### [new 013] USAD 2.0: Scaling Representation Distillation for Universal Audio Understanding
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出USAD 2.0，解决音频统一表示学习问题。通过融合自监督与监督模型，提升多领域音频理解性能。**

- **链接: [https://arxiv.org/pdf/2606.06444](https://arxiv.org/pdf/2606.06444)**

> **作者:** Heng-Jui Chang; Alexander H. Liu; Saurabhchand Bhati; Mrudula Athi; Anton Ratnarajah; Amit Chhetri; James Glass
>
> **备注:** Accepted to Interspeech 2026
>
> **摘要:** Audio encoders are critical to modern audio applications as large language models (LLMs) increasingly rely on a single encoder for diverse inputs. While self-supervised learning (SSL) has yielded strong domain-specific encoders like speech or music experts, multi-domain approaches like USAD and SPEAR remain limited in coverage and evaluation. Recent studies also suggest supervised encoders align better with audio LLMs. We present USAD 2.0, a universal encoder integrating knowledge from both SSL and supervised foundation models. USAD 2.0 introduces domain-aware distillation to address teacher mismatch, extends coverage to the music domain, and adds second-stage supervised distillation for downstream use. We further scale the model to one billion parameters via depth scaling. Experiments show USAD 2.0 achieves strong or state-of-the-art performance across probing and LLM-based evaluations.
>
---
#### [new 014] SpeechJBB: Probing Safety Alignment and Comprehension in Large Audio Language Models under Code-Switched Speech
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于安全对齐任务，旨在解决大音频语言模型在多语言及代码转换语音中的安全漏洞问题。工作包括构建SpeechJBB数据集，并测试语音混淆攻击的有效性。**

- **链接: [https://arxiv.org/pdf/2606.06037](https://arxiv.org/pdf/2606.06037)**

> **作者:** Virginia Ceccatelli; Yejin Jeon; David Ifeoluwa Adelani
>
> **摘要:** Large audio language models (LALMs) are increasingly deployed in real-world applications, yet their safety alignment is still primarily evaluated on monolingual, text-based harmful prompts. This leaves their generalizability under multilingual and spoken settings, particularly code-switched speech, largely underexplored. To address this gap, we introduce SpeechJBB, an audio jailbreak dataset for benchmarking across multiple state-of-the-art LALMs. The extent of safety weaknesses is further probed by introducing an augmented setting where phonologically plausible pseudo-words are inserted around safety-critical terms to simulate localized obfuscation. Across models, code-switched harmful audio yields substantially high jailbreak success rates (JSR), with non-English monolingual and non-English code-switched pairs exhibiting the highest attack success. Pseudo-word insertion further reduces refusal rates, which demonstrates that natural-sounding obfuscation can effectively bypass safety policies.
>
---
#### [new 015] An Ultra-Low-Bitrate Neural Speech Codec with Plain-to-Pseudo Synergistic Vector Quantization
- **分类: eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决传统神经语音编解码器在低比特率下的效率问题。提出P2PSynCodec，通过结合纯量化与预测伪量化，显著降低比特率并保持语音质量。**

- **链接: [https://arxiv.org/pdf/2606.05876](https://arxiv.org/pdf/2606.05876)**

> **作者:** Xiao-Hang Jiang; Yang Ai; Fei Liu; Rui-Chen Zheng; Jian-Qing Gao; Zhen-Hua Ling; Ji Wu
>
> **备注:** Accepted to INTERSPEECH 2026
>
> **摘要:** Most neural speech codecs use residual vector quantization (RVQ), in which later VQs contribute less but consume the same bitrate, leading to inefficiency. We propose P2PSynCodec, an ultra-low-bitrate neural speech codec with a plain-to-pseudo synergistic vector quantizer (P2PSVQ). P2PSVQ consists of one plain VQ and multiple pseudo VQs. The plain VQ produces basic tokens by quantization, while the pseudo VQs generate auxiliary tokens by neural prediction and incur zero transmitted bitrate. Thus, speech is decoded from the plain-VQ tokens together with predicted pseudo-VQ tokens, greatly reducing bitrate. Experiments show that P2PSynCodec achieves speech reconstruction quality comparable to competing codecs at 2.0 kbps while operating at only 0.5 kbps, demonstrating high efficiency for ultra-low-bitrate speech coding.
>
---
#### [new 016] Exploring LLMs for South Asian Music Understanding and Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐理解与生成任务，旨在探讨LLMs在南亚古典音乐中的表现。研究解决了现有模型对非西方音乐结构处理不足的问题，通过构建基准测试和生成框架进行评估。**

- **链接: [https://arxiv.org/pdf/2606.05522](https://arxiv.org/pdf/2606.05522)**

> **作者:** Faria Binte Kader; Mohtasim Hadi Rafi; Shah Wasif Sajjad; Santu Karmaker
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have shown promising results in music understanding and generation tasks. However, existing works remain confined to Western tonal traditions, offering little insight into whether current LLMs can handle structurally distinct low-resource musical traditions. We present the first systematic evaluation of LLM competence in South Asian classical music, a tradition governed by raga, tala-based melodic constraints that impose fundamentally different structural principles from Western harmony-driven music. We ground our evaluation in Hindustani classical theory and Bengali classical forms, including Rabindra and Nazrul Sangeet -- representative low-resource traditions within South Asian classical music. For music understanding evaluation, we introduce a 504-question-answer benchmark spanning raga grammar, cultural knowledge, and symbolic notation reasoning, evaluating 33 LLMs where frontier models such as Gemini 2.5 Pro achieve 85-90% accuracy, while most open-source models remain in the 23-40% range. For music generation, we design a five-level controlled prompting framework and find that even the strongest model produces stylistically faithful outputs only 40% of the time. These results reveal that structural validity and stylistic faithfulness in music generation are distinct objectives and highlight an open challenge for culturally grounded music modeling.
>
---
#### [new 017] Probing Spatial Structure in Pretrained Audio Representations
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频表示学习任务，旨在评估预训练音频模型的空间编码能力。通过构建SARL基准，分析源级和房间级因素的解码难度，揭示现有模型的系统性偏差。**

- **链接: [https://arxiv.org/pdf/2606.05544](https://arxiv.org/pdf/2606.05544)**

> **作者:** Chuyang Chen; Sivan Ding; Adrian S. Roman; Juan Pablo Bello
>
> **备注:** Accepted to Interspeech 2026
>
> **摘要:** Pretrained spatial audio encoders are increasingly used as general-purpose representations for perceptual tasks, yet their spatial encoding capabilities remain poorly understood. We introduce the Spatial Audio Representation Learning (SARL) benchmark, a controlled framework for evaluating spatial information in pretrained audio models. SARL probes source-level factors (azimuth, elevation, distance, class) and room-level factors (RT60, volume, shape). Experiments across diverse encoders reveal three patterns: input configuration and training paradigm shape spatial encoding; source factors are consistently easier to decode than room factors; and sensitivity analysis under controlled perturbations shows heterogeneous responses to source and room variation. These results reveal systematic biases in current pretrained audio representations. SARL is released as an open-source benchmark for reproducible evaluation of spatial audio representations.
>
---
#### [new 018] DBHN-Net: Dual-Branch Hybrid Neural Network For Low-Complexity Monaural Speech Enhancement
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在降低计算复杂度并保持性能。提出DBHN-Net，结合ANN与SNN，减少能耗并缓解信息丢失。**

- **链接: [https://arxiv.org/pdf/2606.05911](https://arxiv.org/pdf/2606.05911)**

> **作者:** Cunhang Fan; Enrui Liu; Jing Zhou; Jian Kang; Jie Li; Andong Li; Jian Zhou; Zhao Lv; Xuelong Li
>
> **备注:** This article has been accepted for publication in IEEE Transactions on Pattern Analysis and Machine Intelligence(TPAMI)
>
> **摘要:** Although artificial neural network (ANN) based speech enhancement (SE) methods demonstrate excellent performance, the high computational complexity and high energy consumption hinder their deployment in practical front-end processing tasks.} Currently, the spiking neural networks (SNNs) have shown potential in reducing power consumption. However, the discrete binary activation and complex spatio-temporal dynamics of SNNs often result in information loss. The current challenge therefore focuses on how to maintain performance and reduce computational complexity. To address this issue, this work propose a Dual-Branch Hybrid Neural (DBHN) Network. 1) In terms of network architecture: A dual-branch network integrating ANN and SNN was designed, where the SNN branch reduces power consumption while the ANN branch addresses information loss; The BandSplit and Time-Frequency (TF) -Mamba modules were developed to simultaneously compress energy consumption and enhance model performance; Spiking Feature Extraction Group (SFEG) and Information Transformation Block (ITB) components were implemented with residual connections to mitigate information loss while further refining feature representations. 2) To facilitate inter-branch information fusion: An Interaction module was designed to promote information exchange at various stages of the dual-branch network; A TF-Cross Attention-Fusion module was designed to perform time-frequency domain fusion of dual-branch information while data-adaptively guiding the SNN branch to retain more critical information. Results show that the proposed model maintains superior performance across three public datasets while achieving an average 7.5 fold reduction in computational complexity compared to baseline models.
>
---
#### [new 019] Beyond Waveform Robustness: Robust Feature-Vocoder Adversarial Attacks on Automatic Speech Recognition
- **分类: cs.SD; cs.AI; cs.CR**

- **简介: 该论文属于语音识别安全领域，解决ASR系统对对抗攻击的脆弱性问题。提出一种基于SSL表示的对抗攻击方法，提升攻击效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.05678](https://arxiv.org/pdf/2606.05678)**

> **作者:** Yifan Liao; Zongmin Zhang; Zhen Sun; Yuhui Sun; Xinhu Zheng; Xinlei He
>
> **备注:** 11 pages
>
> **摘要:** Automatic speech recognition (ASR) systems have become widely used for multilingual speech-to-text transcription. Their robustness to adversarial attacks has become an important topic for the community. Existing adversarial attacks directly add adversarial noise to the speech audio. However, prior work has shown that existing adversarial attacks face two limitations: they often transfer poorly to black-box ASR systems and are increasingly mitigated by defenses tailored to input-space perturbations. In this work, we propose a Clean-Referenced Feature-Vocoder Attack, a surrogate-based black-box attack that moves the adversarial search space from raw waveforms to self-supervised learning (SSL) representations. To address the transferability limitation, we perturb more generalizable acoustic-phonetic representations rather than low-level waveform samples, reducing dependence on surrogate-specific waveform gradients and encouraging adversarial perturbations that generalize across ASR systems. To bypass different defenses, we shift the adversarial signal from explicit additive waveform noise to SSL feature-space perturbations and reconstruct them through a vocoder into speech-like waveform adversarial signals, making the resulting samples less aligned with waveform-bounded defenses. Extensive experiments show that, when optimized only on raw Whisper-small as a public surrogate model, our attack transfers effectively to black-box ASR models with a +26.6 WER improvement over the SOTA baseline, while also remaining effective against multiple training defenses with a +36.2 WER improvement. These results reveal a blind spot in current ASR robustness evaluation.
>
---
#### [new 020] F3-Tokenizer: Taming Audio Autoencoder Latents for Understanding and Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频理解与生成任务，旨在解决音频自动编码器在结构性和可解码性上的不足。通过改进编码器和引入表示编码器，实现高效理解和生成。**

- **链接: [https://arxiv.org/pdf/2606.06357](https://arxiv.org/pdf/2606.06357)**

> **作者:** Dinghao Zhou; Xingchen Song; Di Wu; Pengyu Cheng; Shengfan Shen; Sixiang Lv
>
> **备注:** Technical report; early work; 9 pages, 2 figures, 5 tables
>
> **摘要:** Continuous audio autoencoders reconstruct waveforms well but often produce latents with weak structure for understanding, while self-supervised audio encoders capture semantics but are not directly decodable. This mismatch complicates a single audio tokenizer that must support both understanding and generation. We adapt continuous autoencoder latents to this setting with two components: a noise-regularized autoencoder bottleneck and a latent-side representation encoder. The bottleneck uses channel normalization and stochastic perturbation instead of KL-based variational training, yielding scale-controlled continuous latents for reconstruction and autoregressive generation. The representation encoder is trained on frozen autoencoder latents with RQ-MTP and frozen-LLM supervision. The resulting tokenizer provides high-dimensional representations for understanding while preserving normalized continuous latents as generation targets
>
---
#### [new 021] Learning Emotion-discriminative Representations for Zero-Shot Cross-lingual Speech Emotion Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于跨语言语音情感识别任务，解决目标语言无标注数据时模型性能下降的问题。通过引入情绪判别表示学习，提升模型在零样本跨语言场景下的表现。**

- **链接: [https://arxiv.org/pdf/2606.06200](https://arxiv.org/pdf/2606.06200)**

> **作者:** Jinyi Mi; Ding Ma; Tomoki Toda
>
> **备注:** Accepted to Interspeech 2026
>
> **摘要:** Zero-shot cross-lingual speech emotion recognition (SER) remains challenging due to distribution mismatches across languages and the lack of emotion annotations in target language. Under such conditions, models trained solely on source-language data frequently suffer from degraded generalization when evaluated on unseen target languages. To address this limitation, we propose an emotion-discriminative representation learning method that integrates supervised contrastive learning and speaker adversarial learning. The contrastive learning promotes cross-lingual emotion alignment, while speaker adversarial learning suppresses speaker-related cues to encourage speaker-invariant representations. Experimental results under a zero-shot cross-lingual SER setting demonstrate that the proposed method significantly improves SER performance over conventional training strategies.
>
---
#### [new 022] Beyond WER: A Paired Acoustic Stress Test for Ambient Clinical Scribes
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于医疗语音识别任务，旨在解决噪声对临床安全的影响问题。通过声学压力测试，发现噪声虽小幅提升错误率，却显著影响临床判断，提出轻量级缓解策略。**

- **链接: [https://arxiv.org/pdf/2606.05909](https://arxiv.org/pdf/2606.05909)**

> **作者:** Xiao-Hang Jiang; Han-Jie Guo; Ying-Si Liang; Yang Ai; Zhen-Hua Ling; Lei Jiang; Zhi-Yang He
>
> **备注:** Accepted to INTERSPEECH 2026
>
> **摘要:** Ambient clinical scribes increasingly combine Automatic Speech Recognition with Large Language Models to automate documentation. However, traditional metrics like Word Error Rate mask systemic safety degradation. We present a paired acoustic stress test to isolate the causal impact of noise on clinical reasoning. For the same dialogues, we inject diverse noise types while keeping the downstream model configuration frozen. Crucially, we uncover a dangerous disconnect between signal fidelity and clinical safety. Stationary ambient noise increased the Word Error Rate by a negligible 0.71 percentage points yet nearly doubled the rate of unsafe outputs. Our analysis reveals that minor acoustic perturbations can invert clinical meaning without substantially inflating error rates. Furthermore, we demonstrate a lightweight mitigation strategy that mitigates safety degradation under noisy conditions without requiring model fine tuning.
>
---
#### [new 023] nnAudio 2: Overcoming Dynamic Compilation Barriers and Transform Inconsistencies
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频特征提取任务，解决nnAudio在PyTorch中的兼容性问题，优化STFT/iSTFT实现，提升稳定性与可靠性。**

- **链接: [https://arxiv.org/pdf/2606.05394](https://arxiv.org/pdf/2606.05394)**

> **作者:** Abhinaba Roy; Junyi Liang; Dorien Herremans
>
> **摘要:** nnAudio is an open-source audio feature extraction toolbox for deep learning, but its use in current environments is hindered by TorchScript incompatibilities, inverse-transform edge cases, and dependency drift. We present a targeted modernization for modern PyTorch and scientific Python. We resolve TorchScript compilation failures in STFT and iSTFT by removing dynamic state mutation and module construction from scripted code paths and tightening argument handling in inverse-related helpers. We clarify inverse-STFT behavior by restricting reliable inversion to the uniform-bin setting (freq_scale=`no') and raising explicit runtime errors for unsupported frequency scales, preventing silently degraded reconstructions. We restore CFP compatibility with modern SciPy and ensure VQT reduces to CQT when gamma = 0. Regression tests cover the new STFT/iSTFT behaviors, and the updated codebase passes the full repository test suite in a modern Python environment. These improvements provide a more robust foundation for differentiable audio analysis in research and deployment.
>
---
#### [new 024] SagnacAssisted Enhanced OTDR for Distributed Acoustic Sensing: A Standardized Benchmark and Engineering Evaluation Framework
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于分布式声学传感任务，旨在解决$\phi$-OTDR系统中的性能退化问题。通过引入Sagnac干涉仪和双分支融合模型，提升事件识别准确率与稳定性。**

- **链接: [https://arxiv.org/pdf/2606.05754](https://arxiv.org/pdf/2606.05754)**

> **作者:** Weiguang Wang; Fugen Wu; Hailing Wang; Xuechen Liang; Xiaobin Li; Ru Han; Tianchang Xie
>
> **摘要:** Phase-sensitive optical time-domain reflectometry ($\phi$-OTDR) is widely used in large-scale distributed acoustic sensing (DAS) because it provides distributed spatiotemporal monitoring over long sensing distances. Its field performance can still deteriorate because of polarization-induced fading (PIF), local signal degradation, and strong environmental interference. This study develops a Sagnac-assisted enhanced $\phi$-OTDR sensing architecture and a standardized benchmark framework for engineering-oriented DAS event recognition. The Sagnac interferometer provides a continuous phase response that supplements fading-prone observations in the $\phi$-OTDR channel, and heterogeneous signal alignment is achieved using a cross-correlation procedure implemented on an FPGA platform. The benchmark protocol compares conventional feature-engineering methods, probabilistic shallow classifiers, single-branch deep models, and dual-branch fusion models under consistent data partitioning, preprocessing, and metric definitions. Experiments on a 10-km sensing fiber with six representative acoustic event classes show that the dual-branch fusion model provides the most favorable trade-off among the evaluated methods, reaching 89.79\% accuracy, 89.83\% macro-F1, and a nuisance alarm rate of 5.00\% on the balanced test set. The results also show that channel grouping strongly affects dual-branch evaluation, indicating that deployment-oriented conclusions should be based on accuracy, macro-F1, nuisance alarm rate, false negative rate, and latency rather than accuracy alone. This work provides a physically motivated enhancement strategy for $\phi$-OTDR-based DAS and a reproducible benchmark protocol for future fusion-oriented sensing research. The implementation and scripts for reproducing the DAS event-recognition experiments are publicly available at this https URL.
>
---
#### [new 025] Multi-task Learning is Not Enough: Representational Entanglement in Dual-output Second Language Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于双输出第二语言语音识别任务，旨在解决多任务学习中表示纠缠导致的表面转录质量下降问题。通过分析编码器与解码器的交互，提出缓解编码器纠缠的方法。**

- **链接: [https://arxiv.org/pdf/2606.06065](https://arxiv.org/pdf/2606.06065)**

> **作者:** Seung Hwan Cho; Young-Min Kim
>
> **备注:** 5 pages, 2 figures, Accepted to the 43rd International Conference on Machine Learning Workshop on Machine Learning for Audio
>
> **摘要:** Second-language (L2) speech recognition often requires transcriptions of pronunciations and intended meanings. Multi-task learning (MTL) is a natural approach because it assumes that shared representations benefit both outputs. However, this paper shows that this assumption does not hold across Korean and English. MTL improves meaning but degrades surface transcription, especially in English, where the degradation scales with surface-meaning divergence measured by Levenshtein edit this http URL analysis links these patterns to encoder-level entanglement, with Korean preserving distinct task representations while English produces nearly identical ones. Cross-task decoder analysis shows that the meaning dual-output decoder adapts with a unique representation, while the surface dual-output decoder remains constrained by the encoder. These findings motivate the design of MTL frameworks that mitigate encoder-level entanglement to reduce surface degradation in dual-output L2 automatic speech recognition.
>
---
#### [new 026] FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决病理语音识别难题。通过FiLM技术对语音LLM进行说话人条件化，提升模型适应不同病理说话人的能力，同时保持对正常语音的识别性能。**

- **链接: [https://arxiv.org/pdf/2606.06211](https://arxiv.org/pdf/2606.06211)**

> **作者:** Fernando López; Santosh Kesiraju; Jordi Luque
>
> **备注:** Accepted in Odyssey 2026: The Speaker and Language Recognition Workshop
>
> **摘要:** Automatic speech recognition (ASR) has advanced remarkably for standard speech; however, pathological speech from neurological conditions remains a significant challenge. We investigate speaker conditioning via Feature-wise Linear Modulation (FiLM), injecting x-vector-derived information into each transformer layer of a frozen ASR encoder to adapt internal representations to individual pathological speakers without modifying base model weights. We benchmark this for the ASR task against standard and parameter-efficient fine-tuning baselines, complemented by post-processing, on Spanish and English pathological speech. Additionally, we evaluate if the adapted model preserves the ability to answer speech-related questions. Results show that speaker-conditioned ASR is competitive with established adaptation strategies while retaining performance on non-conditioned speech.
>
---
#### [new 027] MCBench: A Multicontext Safety Assessment Benchmark for Omni Large Language Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于多模态安全评估任务，旨在解决Omni LLMs在跨模态安全判断中的不足。工作包括构建MCBench基准，分析模型在多模态安全场景下的表现与问题。**

- **链接: [https://arxiv.org/pdf/2606.05177](https://arxiv.org/pdf/2606.05177)**

> **作者:** Manh Luong; Tamas Abraham; Junae Kim; Amar Kaur; Rollin Omari; Gholamreza Haffari; Trang Vu; Lizhen Qu; Dinh Phung
>
> **摘要:** Existing multimodal safety benchmarks focus solely on visual inputs and cannot assess Omni Large Language Models (LLMs) that process vision, audio, and text. We introduce MCBench, a benchmark with 1196 scenarios spanning four safety categories that require integrating multiple modalities for accurate safety assessment. Each unsafe scenario is paired with a minimally different safe counterpart to assess model sensitivity. Our evaluations of state-of-the-art models reveal significant challenges. Omni LLMs struggle with subtle or non-physical risks but perform better when salient visual or acoustic cues are present. Analysis of reasoning traces shows that, although models can extract modality-specific information, they often fail to integrate these cues effectively for safety judgments. Our findings reveal that current Omni LLMs lack robust cross-modal reasoning in safety-critical settings, underscoring the need for improved architectures and training strategies for multimodal safety.
>
---
#### [new 028] FORTE: FOL-guided Optimal Refinement for Text-audio rEtrieval
- **分类: cs.MM; eess.AS**

- **简介: 该论文属于文本-音频检索任务，旨在解决跨模态语义对齐问题。提出FORTE框架，结合逻辑推理与轻量对齐模块，提升检索精度。**

- **链接: [https://arxiv.org/pdf/2606.05812](https://arxiv.org/pdf/2606.05812)**

> **作者:** Arghya Pal; Sailaja Rajanala
>
> **备注:** Under Review
>
> **摘要:** Text-to-audio retrieval has made significant progress with shared embedding models such as CLAP and Pengi, yet they often struggle with fine-grained semantic alignment due to the inherent modality gap between text and audio. In this work, we propose FORTE, a unified framework that integrates structured logical reasoning with parameter-efficient cross-modal alignment to improve retrieval precision. Our approach first transforms queries into first-order logic and refines them via a constrained search that preserves semantic invariance while introducing discriminative attributes. The refined representation is then aligned with audio embeddings using a lightweight projection module, followed by a predicate-aware re-ranking step that enforces logical consistency at inference. Extensive experiments on AudioCaps and Clotho demonstrate consistent improvements over strong baselines, particularly in challenging fine-grained scenarios. Our results highlight the effectiveness of combining symbolic reasoning with representation learning for cross-modal retrieval.
>
---
#### [new 029] To Be Multimodal or Not to Be: Query-Adaptive Audio-Visual Person Retrieval via Active Modality Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于跨模态检索任务，解决视频中根据语音和人脸定位目标的问题。通过检测有效模态，提升检索精度，优于单一模态和固定融合方法。**

- **链接: [https://arxiv.org/pdf/2606.05931](https://arxiv.org/pdf/2606.05931)**

> **作者:** Erfan Loweimi; Mengjie Qian; Kate Knill; Guanfeng Wu; Chi-Ho Chan; Abbas Haider; Muhammad Awan; Josef Kittler; Hui Wang; Mark Gales
>
> **备注:** INTERSPEECH 2026
>
> **摘要:** When retrieving a person from a video archive by voice and face, should the system be multimodal or not? In real-world broadcast archives, unlike curated benchmarks, a target may be heard but unseen, seen but unheard, or both. Fusing scores from an absent modality injects noise, degrading precision below the best unimodal system. We propose a query-adaptive framework that detects active modalities via cross-modal score consistency: when both modalities are active, files retrieved by one also score highly on the other; this agreement breaks down when a modality is absent. Classifiers driven by these cross-modal features achieve 89% detection accuracy. On the BBC Rewind corpus (with over 12,000 broadcast videos) the adaptive system attains 94.2% P@1, outperforming speaker-only (82.9%), face-only (93.4%), and fixed fusion (90.0%), recovering 64% of the gap to an oracle with ground-truth modality labels (96.6%).
>
---
#### [new 030] Domain-Aware Mispronunciation Detection and Diagnosis Using Language-Specific Statistical Graphs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别中的发音错误检测任务，旨在解决二语学习者发音错误的识别与诊断问题。通过构建语言特定的统计图模型，捕捉音素混淆模式和母语差异，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2606.05569](https://arxiv.org/pdf/2606.05569)**

> **作者:** Huu Tuong Tu; Hanh Nguyen; Thien Van Luong; Nguyen Tien Cuong; Vu Huan; Nguyen Thi Thu Trang
>
> **备注:** Accepted at Interspeech 2026
>
> **摘要:** Mispronunciation Detection and Diagnosis (MDD) has gained increasing importance in computer-assisted language learning and speech technology in recent years. In this paper, we propose a method for constructing statistical graphs that enable models to learn phoneme confusion patterns represented as directed graphs. Furthermore, we introduce a language-specific strategy to capture systematic pronunciation differences across various native language (L1) backgrounds. The effectiveness of our approach is demonstrated through extensive experiments on the L2-ARCTIC benchmark, where it achieves an F1-score of 59.52%, outperforming several competitive baselines.
>
---
#### [new 031] Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态情感分析任务，解决生成式读出方法的不足，提出一种判别式隐藏状态读出机制，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2606.05713](https://arxiv.org/pdf/2606.05713)**

> **作者:** Bin Wen; Tien-Ping Tan
>
> **备注:** 18 pages, 4 figures, 6 tables
>
> **摘要:** Multimodal sentiment analysis (MSA) infers human affect from language, acoustic, and visual signals. Recent methods increasingly adapt large multimodal models (LMMs) via generative readout: prompting the model to emit a sentiment score as a text string. While convenient, this ties continuous regression to discrete autoregressive decoding, incurring unmeasured costs. We revisit this readout mechanism and propose a discriminative formulation built on the Thinker module of a native omni-modal LLM (Qwen2.5-Omni-7B). Instead of text decoding, we map the final-layer hidden state of the last non-padding token to a continuous score via a lightweight regression head in a single forward pass. Using 4-bit quantization and low-rank adaptation (QLoRA), the entire 7B pipeline -- including video and audio processing -- trains on a single consumer GPU (RTX 5090, 32 GB) with 10-21 GB peak memory and 1.14% trainable parameters. Through a controlled comparison fixing the backbone, data, and LoRA configuration, we isolate the impact of the readout. On CMU-MOSI and CMU-MOSEI, our discriminative readout reaches state-of-the-art accuracy without task-specific feature engineering (MOSI: MAE 0.551, Corr 0.888; MOSEI: MAE 0.506, Corr 0.790) and exhibits strong multi-seed stability. In contrast, the generative readout -- even after equivalent supervised training -- more than doubles the mean absolute error, yields unparsable or out-of-range outputs (2.8% zero-shot), and suffers from higher latency. Modality ablations reveal a text-dominant regime on CMU-MOSI. Our findings indicate that how an LMM is read out is as consequential as how it is trained, demonstrating that a discriminative readout offers a more accurate, efficient, and reliable alternative for continuous MSA.
>
---
#### [new 032] Towards Truly Multilingual ASR: Generalizing Code-Switching ASR to Unseen Language Pairs
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决代码转换ASR在未见语言对上的泛化问题。通过模型合并和领域泛化方法，探索已见语言对的CS能力是否能迁移到未见语言对。**

- **链接: [https://arxiv.org/pdf/2606.05846](https://arxiv.org/pdf/2606.05846)**

> **作者:** Gio Paik; Hyunseo Shin; Soungmin Lee
>
> **备注:** ICML 2026 Workshop on Machine Learning for Audio
>
> **摘要:** Automatic Speech Recognition (ASR) has become a key technology for human--AI interaction. However, code-switching ASR (CS-ASR) remains particularly challenging due to the severe scarcity of multilingual CS speech resources across diverse language pairs. Existing approaches primarily improve CS-ASR performance through synthetic CS speech generation or pair-specific fine-tuning on limited bilingual datasets. Nevertheless, these approaches face an inherent scalability limitation, as support for CS must be developed separately for language pairs whose number grows combinatorially with the number of supported languages. In this work, we investigate whether CS capabilities learned from a limited set of seen language pairs can generalize to unseen language pairs through model merging and domain generalization methods. Our experiments show that merged bilingual CS-ASR models modestly generalize to unseen language pairs, suggesting limited transfer of bilingual CS capabilities across language pairs.
>
---
## 更新

#### [replaced 001] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出FLAIR方法，用于全双工对话系统中的隐式推理，解决语音交互中同时进行思考与响应的问题，通过连续潜向量推理提升对话质量。**

- **链接: [https://arxiv.org/pdf/2603.17837](https://arxiv.org/pdf/2603.17837)**

> **作者:** Donghang Wu; Tianyu Zhang; Yuxin Li; Hexin Liu; Chen Chen; Eng Siong Chng; Yoshua Bengio
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** During conversational interactions, humans subconsciously engage in concurrent thinking while listening to a speaker. Although this internal cognitive processing may not always manifest as explicit linguistic structures, it is instrumental in formulating high-quality responses. Inspired by this cognitive phenomenon, we propose a novel Full-duplex LAtent and Internal Reasoning method named FLAIR that conducts latent thinking simultaneously with speech perception. Unlike conventional "thinking" mechanisms in NLP, which require post-hoc generation, our approach aligns seamlessly with spoken dialogue systems: during the user's speaking phase, it recursively feeds the latent embedding output from the previous step into the next step, enabling continuous reasoning that strictly adheres to causality without introducing additional latency. To enable this latent reasoning, we design an Evidence Lower Bound-based objective that supports efficient supervised finetuning via teacher forcing, circumventing the need for explicit reasoning annotations. Experiments demonstrate the effectiveness of this think-while-listening design, which achieves competitive results on a range of speech benchmarks. Furthermore, FLAIR robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics.
>
---
#### [replaced 002] RAS: a Reliability Oriented Metric for Automatic Speech Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于自动语音识别任务，解决模型在噪声下误判的问题。提出RAS度量标准，提升转录可靠性同时保持准确率。**

- **链接: [https://arxiv.org/pdf/2604.24278](https://arxiv.org/pdf/2604.24278)**

> **作者:** Wenbin Huang; Yuhang Qiu; Bohan Li; Yiwei Guo; Jing Peng; Hankun Wang; Xie Chen; Kai Yu
>
> **备注:** 6 pages, 4 figures; Accepted at InterSpeech 2026
>
> **摘要:** Automatic speech recognition systems often produce confident yet incorrect transcriptions under noisy or ambiguous conditions, which can be misleading for both users and downstream applications. Standard evaluation based on Word Error Rate focuses solely on accuracy and fails to capture transcription reliability. We introduce an abstention-aware transcription framework that enables ASR models to explicitly abstain from uncertain segments. To evaluate reliability under abstention, we propose RAS, a reliability-oriented metric that balances transcription informativeness and error aversion, with its trade-off parameter calibrated by human preference. We then train an abstention-aware ASR model through supervised bootstrapping followed by reinforcement learning. Our experiments demonstrate substantial improvements in transcription reliability while maintaining competitive accuracy.
>
---
#### [replaced 003] LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于对话管理系统任务，旨在解决全双工语音对话中的实时话轮切换问题。通过引入语义VAD模块，实现高效、准确的对话管理。**

- **链接: [https://arxiv.org/pdf/2502.14145](https://arxiv.org/pdf/2502.14145)**

> **作者:** Hao Zhang; Weiwei Li; Rilin Chen; Vinay Kothapally; Meng Yu; Dong Yu
>
> **摘要:** Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS.
>
---
#### [replaced 004] Absorbing Discrete Diffusion for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在通过吸收离散扩散模型提升噪声语音质量。提出ADDSE方法，结合神经音频编码器和扩散模型，有效处理低信噪比情况。**

- **链接: [https://arxiv.org/pdf/2602.22417](https://arxiv.org/pdf/2602.22417)**

> **作者:** Philippe Gonzalez
>
> **备注:** Accepted at Interspeech 2026
>
> **摘要:** Inspired by recent developments in neural speech coding and diffusion-based language modeling, we tackle speech enhancement by modeling the conditional distribution of clean speech codes given noisy speech codes using absorbing discrete diffusion. The proposed approach, which we call ADDSE, leverages both the expressive latent space of neural audio codecs and the non-autoregressive sampling procedure of diffusion models. To efficiently model the hierarchical structure of residual vector quantization codes, we propose RQDiT, which combines techniques from RQ-Transformer and diffusion Transformers for non-autoregressive modeling. Results show competitive performance in terms of non-intrusive objective metrics on two datasets, especially at low signal-to-noise ratios and with few sampling steps. Code and audio examples are available online.
>
---
#### [replaced 005] DuoGesture: Neuro-Inspired and Biomechanically Informed Dual-Stream Co-Speech Gesture Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于语音驱动手势生成任务，旨在解决语义表达与生物力学合理性之间的矛盾。提出DuoGesture模型，通过双流结构分离语义与节奏手势，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26236](https://arxiv.org/pdf/2605.26236)**

> **作者:** Ferdinand Paar; Lanmiao Liu; Aslı Özyürek; Serge Thill; Esam Ghaleb
>
> **摘要:** Co-speech gesture generation requires both semantic expressivity and biomechanically plausible rhythmic motion. Existing holistic gesture models mix lexically grounded semantic gestures with frequent prosody-aligned beat gestures. This limits semantic grounding, speech-motion alignment, and kinematic smoothness. We propose \emph{DuoGesture}, a neuro-inspired and biomechanically informed dual-stream approach that decomposes co-speech gesture synthesis into coupled semantic and beat streams. The two streams are coordinated by a \emph{Semantic Variational Information Bottleneck}, a stochastic frame-level gate that learns when semantic gestures should override rhythmic beat motion. The semantic stream is controlled by \emph{Motion-Grounded Semantic Conditioning}, which replaces purely linguistic word embeddings with motion-language representations to provide motion-aligned semantic priors for long-tailed lexical triggers of gestures. The beat stream is further regularised by an \emph{Inertial Beat Prior}, an anthropometry-weighted arm-chain module that reduces jitter and improves rhythmic consistency without constraining semantic frames. Objective evaluations and subjective experiments show that DuoGesture outperforms strong holistic baselines, while component ablations confirm the complementary roles of semantic grounding, stochastic stream selection, and biomechanical regularisation.
>
---
