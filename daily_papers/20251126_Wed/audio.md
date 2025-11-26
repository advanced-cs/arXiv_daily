# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] Evaluating Objective Speech Quality Metrics for Neural Audio Codecs
- **分类: cs.SD**

- **简介: 该论文研究神经音频编解码器的客观质量评估问题。针对人类听觉测试成本高的局限，通过MUSHRA测试分析主观评分与多种客观指标的相关性，评估现有指标在语音质量评价中的可靠性，为神经音频编解码器的性能评估提供实用指导。**

- **链接: [https://arxiv.org/pdf/2511.19734v1](https://arxiv.org/pdf/2511.19734v1)**

> **作者:** Luca A. Lanzendörfer; Florian Grötschla
>
> **摘要:** Neural audio codecs have gained recent popularity for their use in generative modeling as they offer high-fidelity audio reconstruction at low bitrates. While human listening studies remain the gold standard for assessing perceptual quality, they are time-consuming and impractical. In this work, we examine the reliability of existing objective quality metrics in assessing the performance of recent neural audio codecs. To this end, we conduct a MUSHRA listening test on high-fidelity speech signals and analyze the correlation between subjective scores and widely used objective metrics. Our results show that, while some metrics align well with human perception, others struggle to capture relevant distortions. Our findings provide practical guidance for selecting appropriate evaluation metrics when using neural audio codecs for speech.
>
---
#### [new 002] DUO-TOK: Dual-Track Semantic Music Tokenizer for Vocal-Accompaniment Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对歌词到歌曲生成中重建质量与语言模型可学习性之间的矛盾，提出双轨语义音乐分词器DUO-TOK。通过四阶段自监督学习流程，构建声乐与伴奏分离的双码本，实现高保真重建与低困惑度语言建模的平衡，显著提升生成性能。**

- **链接: [https://arxiv.org/pdf/2511.20224v1](https://arxiv.org/pdf/2511.20224v1)**

> **作者:** Rui Lin; Zhiyue Wu; Jiahe Le; Kangdi Wang; Weixiong Chen; Junyu Dai; Tao Jiang
>
> **备注:** 17 pages, 5 figures, 8 tables. Project page: https://eps-acoustic-revolution-lab.github.io/DUO_TOK/
>
> **摘要:** Duo-Tok is a source-aware dual-codebook tokenizer for vocal-accompaniment music that targets the growing tension between reconstruction quality and language-model (LM) learnability in modern lyrics-to-song systems. Existing codecs either prioritize high-fidelity reconstruction with difficult-to-model acoustic tokens or compress aggressively into semantic tokens that are LM-friendly but lossy, and they rarely make the tokenizer itself aware of dual-track structure. Duo-Tok follows a four-stage, SSL-centered pipeline: we first pretrain a BEST-RQ-style encoder on large-scale audio, then stabilize and factorize the representation with Gaussian replacement noise and multi-task supervision, before freezing the encoder to learn SimVQ-based dual codebooks with hard routing for vocals and accompaniment, and finally training latent diffusion decoders on top of the discrete tokens. Duo-Tok at 0.75 kbps shifts the empirical reconstruction-generation Pareto frontier, achieving the best music-tagging AP and the lowest vocabulary-normalized LM perplexity among compared codecs while maintaining reconstruction quality comparable to state-of-the-art music tokenizers.
>
---
#### [new 003] Continual Audio Deepfake Detection via Universal Adversarial Perturbation
- **分类: cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决模型在持续演化攻击下性能退化及频繁微调带来的高成本问题。提出结合通用对抗扰动（UAP）的框架，使模型在无需历史数据的情况下持续学习，有效提升对新型攻击的检测能力。**

- **链接: [https://arxiv.org/pdf/2511.19974v1](https://arxiv.org/pdf/2511.19974v1)**

> **作者:** Wangjie Li; Lin Li; Qingyang Hong
>
> **摘要:** The rapid advancement of speech synthesis and voice conversion technologies has raised significant security concerns in multimedia forensics. Although current detection models demonstrate impressive performance, they struggle to maintain effectiveness against constantly evolving deepfake attacks. Additionally, continually fine-tuning these models using historical training data incurs substantial computational and storage costs. To address these limitations, we propose a novel framework that incorporates Universal Adversarial Perturbation (UAP) into audio deepfake detection, enabling models to retain knowledge of historical spoofing distribution without direct access to past data. Our method integrates UAP seamlessly with pre-trained self-supervised audio models during fine-tuning. Extensive experiments validate the effectiveness of our approach, showcasing its potential as an efficient solution for continual learning in audio deepfake detection.
>
---
#### [new 004] Efficient and Fast Generative-Based Singing Voice Separation using a Latent Diffusion Model
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究生成式歌唱声分离任务，针对现有方法在分离性能与推理效率上的不足，提出基于潜在扩散模型的方案。仅用孤立人声与混音对训练，利用潜空间生成实现高效优化与快速推理，显著提升分离质量与噪声鲁棒性，并开源工具包促进后续研究。**

- **链接: [https://arxiv.org/pdf/2511.20470v1](https://arxiv.org/pdf/2511.20470v1)**

> **作者:** Genís Plaja-Roglans; Yun-Ning Hung; Xavier Serra; Igor Pereira
>
> **备注:** Accepted for oral presentation at IJCNN 2025
>
> **摘要:** Extracting individual elements from music mixtures is a valuable tool for music production and practice. While neural networks optimized to mask or transform mixture spectrograms into the individual source(s) have been the leading approach, the source overlap and correlation in music signals poses an inherent challenge. Also, accessing all sources in the mixture is crucial to train these systems, while complicated. Attempts to address these challenges in a generative fashion exist, however, the separation performance and inference efficiency remain limited. In this work, we study the potential of diffusion models to advance toward bridging this gap, focusing on generative singing voice separation relying only on corresponding pairs of isolated vocals and mixtures for training. To align with creative workflows, we leverage latent diffusion: the system generates samples encoded in a compact latent space, and subsequently decodes these into audio. This enables efficient optimization and faster inference. Our system is trained using only open data. We outperform existing generative separation systems, and level the compared non-generative systems on a list of signal quality measures and on interference removal. We provide a noise robustness study on the latent encoder, providing insights on its potential for the task. We release a modular toolkit for further research on the topic.
>
---
#### [new 005] Differentiable Attenuation Filters for Feedback Delay Networks
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对数字混响系统中反馈延迟网络（FDN）的频率依赖衰减问题，提出一种可微分的二阶段无限冲激响应滤波器设计方法。通过共享参数的参数均衡器结构，实现高效、灵活且可优化的滤波器配置，显著降低计算成本并支持梯度学习。**

- **链接: [https://arxiv.org/pdf/2511.20380v1](https://arxiv.org/pdf/2511.20380v1)**

> **作者:** Ilias Ibnyahya; Joshua D. Reiss
>
> **摘要:** We introduce a novel method for designing attenuation filters in digital audio reverberation systems based on Feedback Delay Networks (FDNs). Our approach uses Second Order Sections (SOS) of Infinite Impulse Response (IIR) filters arranged as parametric equalizers (PEQ), enabling fine control over frequency-dependent reverberation decay. Unlike traditional graphic equalizer designs, which require numerous filters per delay line, we propose a scalable solution where the number of filters can be adjusted. The frequency, gain, and quality factor (Q) parameters are shared parameters across delay lines and only the gain is adjusted based on delay length. This design not only reduces the number of optimization parameters, but also remains fully differentiable and compatible with gradient-based learning frameworks. Leveraging principles of analog filter design, our method allows for efficient and accurate filter fitting using supervised learning. Our method delivers a flexible and differentiable design, achieving state-of-the-art performance while significantly reducing computational cost.
>
---
#### [new 006] BERT-APC: A Reference-free Framework for Automatic Pitch Correction via Musical Context Inference
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出BERT-APC，一种无需参考音高的自动声乐调音框架。针对现有方法依赖参考音高或丢失表现力的问题，利用音乐语言模型推断乐句上下文，结合可学习数据增强，实现精准且自然的音高修正，显著提升音准准确率与听感质量。**

- **链接: [https://arxiv.org/pdf/2511.20006v1](https://arxiv.org/pdf/2511.20006v1)**

> **作者:** Sungjae Kim; Kihyun Na; Jinyoung Choi; Injung Kim
>
> **备注:** 12 pages, 6 figures, 5 tables
>
> **摘要:** Automatic Pitch Correction (APC) enhances vocal recordings by aligning pitch deviations with the intended musical notes. However, existing APC systems either rely on reference pitches, which limits their practical applicability, or employ simple pitch estimation algorithms that often fail to preserve expressiveness and naturalness. We propose BERT-APC, a novel reference-free APC framework that corrects pitch errors while maintaining the natural expressiveness of vocal performances. In BERT-APC, a novel stationary pitch predictor first estimates the perceived pitch of each note from the detuned singing voice. A context-aware note pitch predictor estimates the intended pitch sequence by leveraging a music language model repurposed to incorporate musical context. Finally, a note-level correction algorithm fixes pitch errors while preserving intentional pitch deviations for emotional expression. In addition, we introduce a learnable data augmentation strategy that improves the robustness of the music language model by simulating realistic detuning patterns. Compared to two recent singing voice transcription models, BERT-APC demonstrated superior performance in note pitch prediction, outperforming the second-best model, ROSVOT, by 10.49%p on highly detuned samples in terms of the raw pitch accuracy. In the MOS test, BERT-APC achieved the highest score of $4.32 \pm 0.15$, which is significantly higher than those of the widely-used commercial APC tools, AutoTune ($3.22 \pm 0.18$) and Melodyne ($3.08 \pm 0.18$), while maintaining a comparable ability to preserve expressive nuances. To the best of our knowledge, this is the first APC model that leverages a music language model to achieve reference-free pitch correction with symbolic musical context. The corrected audio samples of BERT-APC are available online.
>
---
#### [new 007] It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models
- **分类: cs.MM; cs.CV; cs.LG; eess.AS**

- **简介: 该论文针对抑郁症检测任务，解决传统语言模型无法处理视听非语言线索的问题。提出一种融合音频与视觉理解的多模态大模型，通过时间戳级对齐实现跨模态动态建模，提升检测精度并降低资源需求，实验验证其优于单模态及现有多模态方法。**

- **链接: [https://arxiv.org/pdf/2511.19877v1](https://arxiv.org/pdf/2511.19877v1)**

> **作者:** Xiangyu Zhao; Yaling Shen; Yiwen Jiang; Zimu Wang; Jiahe Liu; Maxmartwell H Cheng; Guilherme C Oliveira; Robert Desimone; Dominic Dwyer; Zongyuan Ge
>
> **摘要:** Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
>
---
#### [new 008] Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究语音识别中的误发音检测与诊断任务，旨在无需模型训练即可准确识别和分析发音错误。提出一种基于检索的无训练框架，利用预训练ASR模型实现高效、精准的误发音检测，避免了传统方法对音素建模或额外训练的依赖，在L2-ARCTIC数据集上取得69.60%的F1分数。**

- **链接: [https://arxiv.org/pdf/2511.20107v1](https://arxiv.org/pdf/2511.20107v1)**

> **作者:** Huu Tuong Tu; Ha Viet Khanh; Tran Tien Dat; Vu Huan; Thien Van Luong; Nguyen Tien Cuong; Nguyen Thi Thu Trang
>
> **摘要:** Mispronunciation Detection and Diagnosis (MDD) is crucial for language learning and speech therapy. Unlike conventional methods that require scoring models or training phoneme-level models, we propose a novel training-free framework that leverages retrieval techniques with a pretrained Automatic Speech Recognition model. Our method avoids phoneme-specific modeling or additional task-specific training, while still achieving accurate detection and diagnosis of pronunciation errors. Experiments on the L2-ARCTIC dataset show that our method achieves a superior F1 score of 69.60% while avoiding the complexity of model training.
>
---
## 更新

#### [replaced 001] Lessons Learned from Developing a Privacy-Preserving Multimodal Wearable for Local Voice-and-Vision Inference
- **分类: cs.HC; eess.AS; eess.IV; eess.SY**

- **简介: 该论文研究隐私保护的多模态可穿戴设备，解决用户对持续传感与数据泄露的担忧。通过软硬件协同设计，实现耳戴设备在本地运行语音视觉联合推理，利用手机作为可信边缘，完成离线量化模型推理，验证了在消费级硬件上实现低延迟、高隐私保护的可行性。**

- **链接: [https://arxiv.org/pdf/2511.11811v2](https://arxiv.org/pdf/2511.11811v2)**

> **作者:** Yonatan Tussa; Andy Heredia; Nirupam Roy
>
> **摘要:** Many promising applications of multimodal wearables require continuous sensing and heavy computation, yet users reject such devices due to privacy concerns. This paper shares our experiences building an ear-mounted voice-and-vision wearable that performs local AI inference using a paired smartphone as a trusted personal edge. We describe the hardware-software co-design of this privacy-preserving system, including challenges in integrating a camera, microphone, and speaker within a 30-gram form factor, enabling wake word-triggered capture, and running quantized vision-language and large-language models entirely offline. Through iterative prototyping, we identify key design hurdles in power budgeting, connectivity, latency, and social acceptability. Our initial evaluation shows that fully local multimodal inference is feasible on commodity mobile hardware with interactive latency. We conclude with design lessons for researchers developing embedded AI systems that balance privacy, responsiveness, and usability in everyday settings.
>
---
#### [replaced 002] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文针对视频到音频生成任务，解决现有方法因目标耦合导致的感知维度失衡与人类偏好不符问题。提出PrismAudio框架，通过四类分解式思维链与多维奖励机制实现可解释的强化学习优化，并引入Fast-GRPO加速训练及AudioCanvas基准测试，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18833v2](https://arxiv.org/pdf/2511.18833v2)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** Preprint
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at https://PrismAudio-Project.github.io.
>
---
