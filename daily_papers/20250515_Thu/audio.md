# 音频 cs.SD;  eess.SP

- **最新发布 9 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] Adaptive Noise Resilient Keyword Spotting Using One-Shot Learning
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究抗噪声关键词检测（KWS），属于语音识别任务。针对嵌入式设备在真实噪声环境下性能下降的问题，提出一种低计算量单次学习方法，仅需1样本和1轮训练即可动态适配预训练模型，提升噪声鲁棒性。实验表明，该方法在低信噪比（≤18dB）时准确率提升4.9%-46%，且满足资源受限设备的轻量化部署需求。**

- **链接: [http://arxiv.org/pdf/2505.09304v1](http://arxiv.org/pdf/2505.09304v1)**

> **作者:** Luciano Sebastian Martinez-Rau; Quynh Nguyen Phuong Vu; Yuxuan Zhang; Bengt Oelmann; Sebastian Bader
>
> **备注:** Preprint submitted to the IEEE 11th World Forum on Internet of Things
>
> **摘要:** Keyword spotting (KWS) is a key component of smart devices, enabling efficient and intuitive audio interaction. However, standard KWS systems deployed on embedded devices often suffer performance degradation under real-world operating conditions. Resilient KWS systems address this issue by enabling dynamic adaptation, with applications such as adding or replacing keywords, adjusting to specific users, and improving noise robustness. However, deploying resilient, standalone KWS systems with low latency on resource-constrained devices remains challenging due to limited memory and computational resources. This study proposes a low computational approach for continuous noise adaptation of pretrained neural networks used for KWS classification, requiring only 1-shot learning and one epoch. The proposed method was assessed using two pretrained models and three real-world noise sources at signal-to-noise ratios (SNRs) ranging from 24 to -3 dB. The adapted models consistently outperformed the pretrained models across all scenarios, especially at SNR $\leq$ 18 dB, achieving accuracy improvements of 4.9% to 46.0%. These results highlight the efficacy of the proposed methodology while being lightweight enough for deployment on resource-constrained devices.
>
---
#### [new 002] DPN-GAN: Inducing Periodic Activations in Generative Adversarial Networks for High-Fidelity Audio Synthesis
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS**

- **简介: 该论文提出DPN-GAN，用于高保真音频合成任务，解决传统GAN依赖梅尔谱导致分辨率低、模式崩溃的问题。通过周期性ReLU激活函数引入音频周期特性，设计可变形卷积模块实现多分辨率生成，并改进判别器结构。实验证明其音频质量优于现有模型，且具备鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.09091v1](http://arxiv.org/pdf/2505.09091v1)**

> **作者:** Zeeshan Ahmad; Shudi Bao; Meng Chen
>
> **摘要:** In recent years, generative adversarial networks (GANs) have made significant progress in generating audio sequences. However, these models typically rely on bandwidth-limited mel-spectrograms, which constrain the resolution of generated audio sequences, and lead to mode collapse during conditional generation. To address this issue, we propose Deformable Periodic Network based GAN (DPN-GAN), a novel GAN architecture that incorporates a kernel-based periodic ReLU activation function to induce periodic bias in audio generation. This innovative approach enhances the model's ability to capture and reproduce intricate audio patterns. In particular, our proposed model features a DPN module for multi-resolution generation utilizing deformable convolution operations, allowing for adaptive receptive fields that improve the quality and fidelity of the synthetic audio. Additionally, we enhance the discriminator network using deformable convolution to better distinguish between real and generated samples, further refining the audio quality. We trained two versions of the model: DPN-GAN small (38.67M parameters) and DPN-GAN large (124M parameters). For evaluation, we use five different datasets, covering both speech synthesis and music generation tasks, to demonstrate the efficiency of the DPN-GAN. The experimental results demonstrate that DPN-GAN delivers superior performance on both out-of-distribution and noisy data, showcasing its robustness and adaptability. Trained across various datasets, DPN-GAN outperforms state-of-the-art GAN architectures on standard evaluation metrics, and exhibits increased robustness in synthesized audio.
>
---
#### [new 003] SingNet: Towards a Large-Scale, Diverse, and In-the-Wild Singing Voice Dataset
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决歌声合成与转换中缺乏大规模公开数据集的问题。作者构建了包含3000小时多语言、多风格歌声的SingNet数据集，并提出数据处理流程，同时预训练了多个SOTA模型，进行了多项基准实验。**

- **链接: [http://arxiv.org/pdf/2505.09325v1](http://arxiv.org/pdf/2505.09325v1)**

> **作者:** Yicheng Gu; Chaoren Wang; Junan Zhang; Xueyao Zhang; Zihao Fang; Haorui He; Zhizheng Wu
>
> **摘要:** The lack of a publicly-available large-scale and diverse dataset has long been a significant bottleneck for singing voice applications like Singing Voice Synthesis (SVS) and Singing Voice Conversion (SVC). To tackle this problem, we present SingNet, an extensive, diverse, and in-the-wild singing voice dataset. Specifically, we propose a data processing pipeline to extract ready-to-use training data from sample packs and songs on the internet, forming 3000 hours of singing voices in various languages and styles. Furthermore, to facilitate the use and demonstrate the effectiveness of SingNet, we pre-train and open-source various state-of-the-art (SOTA) models on Wav2vec2, BigVGAN, and NSF-HiFiGAN based on our collected singing voice data. We also conduct benchmark experiments on Automatic Lyric Transcription (ALT), Neural Vocoder, and Singing Voice Conversion (SVC). Audio demos are available at: https://singnet-dataset.github.io/.
>
---
#### [new 004] The Voice Timbre Attribute Detection 2025 Challenge Evaluation Plan
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音属性检测任务，旨在通过比较方法解释人类对音色感知的感官属性（如明亮、磁性）。解决主观听觉特征的量化问题，组织2025年挑战赛并制定评估方案，推动基于描述符的语音分析技术发展。**

- **链接: [http://arxiv.org/pdf/2505.09382v1](http://arxiv.org/pdf/2505.09382v1)**

> **作者:** Zhengyan Sheng; Jinghao He; Liping Chen; Kong Aik Lee; Zhen-Hua Ling
>
> **摘要:** Voice timbre refers to the unique quality or character of a person's voice that distinguishes it from others as perceived by human hearing. The Voice Timbre Attribute Detection (VtaD) 2025 challenge focuses on explaining the voice timbre attribute in a comparative manner. In this challenge, the human impression of voice timbre is verbalized with a set of sensory descriptors, including bright, coarse, soft, magnetic, and so on. The timbre is explained from the comparison between two voices in their intensity within a specific descriptor dimension. The VtaD 2025 challenge starts in May and culminates in a special proposal at the NCMMSC2025 conference in October 2025 in Zhenjiang, China.
>
---
#### [new 005] SaFARi: State-Space Models for Frame-Agnostic Representation
- **分类: cs.LG; eess.AS; eess.IV; eess.SP**

- **简介: 该论文提出SaFARi框架，改进状态空间模型（SSMs）的架构设计任务。针对现有SSM仅依赖有限多项式基函数的问题，通过构建支持任意数学框架/基的通用方法，突破传统SSM的基函数限制，扩展模型表达能力，兼容并超越HiPPO等经典方法，实现更灵活的长期依赖数据处理能力。**

- **链接: [http://arxiv.org/pdf/2505.08977v1](http://arxiv.org/pdf/2505.08977v1)**

> **作者:** Hossein Babaei; Mel White; Sina Alemohammad; Richard G. Baraniuk
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** State-Space Models (SSMs) have re-emerged as a powerful tool for online function approximation, and as the backbone of machine learning models for long-range dependent data. However, to date, only a few polynomial bases have been explored for this purpose, and the state-of-the-art implementations were built upon the best of a few limited options. In this paper, we present a generalized method for building an SSM with any frame or basis, rather than being restricted to polynomials. This framework encompasses the approach known as HiPPO, but also permits an infinite diversity of other possible "species" within the SSM architecture. We dub this approach SaFARi: SSMs for Frame-Agnostic Representation.
>
---
#### [new 006] Inference Attacks for X-Vector Speaker Anonymization
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文研究语音匿名化中的隐私保护问题，针对现有x-vector说话人匿名化方法依赖复杂机器学习攻击评估隐私的局限，提出了一种无需机器学习的新型推理攻击方法，实验证明其去匿名化效果优于传统方案。**

- **链接: [http://arxiv.org/pdf/2505.08978v1](http://arxiv.org/pdf/2505.08978v1)**

> **作者:** Luke Bauer; Wenxuan Bao; Malvika Jadhav; Vincent Bindschaedler
>
> **摘要:** We revisit the privacy-utility tradeoff of x-vector speaker anonymization. Existing approaches quantify privacy through training complex speaker verification or identification models that are later used as attacks. Instead, we propose a novel inference attack for de-anonymization. Our attack is simple and ML-free yet we show experimentally that it outperforms existing approaches.
>
---
#### [new 007] Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对多模态音频问答任务，研究是否必须使用音频数据优化音频大语言模型。作者通过GRPO强化学习方法微调Qwen2.5-Omni模型，在MMAU基准取得最优性能，发现性能提升主要源于文本推理优化，并揭示仅用文本数据微调也能有效增强音频任务表现。**

- **链接: [http://arxiv.org/pdf/2505.09439v1](http://arxiv.org/pdf/2505.09439v1)**

> **作者:** Andrew Rouditchenko; Saurabhchand Bhati; Edson Araujo; Samuel Thomas; Hilde Kuehne; Rogerio Feris; James Glass
>
> **摘要:** We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU benchmark. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
>
---
#### [new 008] UWAV: Uncertainty-weighted Weakly-supervised Audio-Visual Video Parsing
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文研究弱监督音频-视觉视频解析任务，解决现有方法生成伪标签时忽略片段关联及预测偏差的问题。提出UWAV模型，通过不确定性加权评估伪标签可靠性，并引入特征混合正则化优化训练。实验表明其在多数据集上超越现有方法，验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.09615v1](http://arxiv.org/pdf/2505.09615v1)**

> **作者:** Yung-Hsuan Lai; Janek Ebbers; Yu-Chiang Frank Wang; François Germain; Michael Jeffrey Jones; Moitreya Chatterjee
>
> **备注:** CVPR 2025
>
> **摘要:** Audio-Visual Video Parsing (AVVP) entails the challenging task of localizing both uni-modal events (i.e., those occurring exclusively in either the visual or acoustic modality of a video) and multi-modal events (i.e., those occurring in both modalities concurrently). Moreover, the prohibitive cost of annotating training data with the class labels of all these events, along with their start and end times, imposes constraints on the scalability of AVVP techniques unless they can be trained in a weakly-supervised setting, where only modality-agnostic, video-level labels are available in the training data. To this end, recently proposed approaches seek to generate segment-level pseudo-labels to better guide model training. However, the absence of inter-segment dependencies when generating these pseudo-labels and the general bias towards predicting labels that are absent in a segment limit their performance. This work proposes a novel approach towards overcoming these weaknesses called Uncertainty-weighted Weakly-supervised Audio-visual Video Parsing (UWAV). Additionally, our innovative approach factors in the uncertainty associated with these estimated pseudo-labels and incorporates a feature mixup based training regularization for improved training. Empirical results show that UWAV outperforms state-of-the-art methods for the AVVP task on multiple metrics, across two different datasets, attesting to its effectiveness and generalizability.
>
---
#### [new 009] WavReward: Spoken Dialogue Models With Generalist Reward Evaluators
- **分类: eess.AS; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于语音对话模型评估任务，旨在解决现有文本模型无法有效评估语音对话系统非文本信息（如IQ/EQ）的问题。提出WavReward音频奖励模型，结合深度推理和非线性奖励机制，并构建ChatReward-30K数据集进行训练，在客观准确率和主观测试中显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09558v1](http://arxiv.org/pdf/2505.09558v1)**

> **作者:** Shengpeng Ji; Tianle Liang; Yangzhuo Li; Jialong Zuo; Minghui Fang; Jinzheng He; Yifu Chen; Zhengqing Liu; Ziyue Jiang; Xize Cheng; Siqi Zheng; Jin Xu; Junyang Lin; Zhou Zhao
>
> **摘要:** End-to-end spoken dialogue models such as GPT-4o-audio have recently garnered significant attention in the speech domain. However, the evaluation of spoken dialogue models' conversational performance has largely been overlooked. This is primarily due to the intelligent chatbots convey a wealth of non-textual information which cannot be easily measured using text-based language models like ChatGPT. To address this gap, we propose WavReward, a reward feedback model based on audio language models that can evaluate both the IQ and EQ of spoken dialogue systems with speech input. Specifically, 1) based on audio language models, WavReward incorporates the deep reasoning process and the nonlinear reward mechanism for post-training. By utilizing multi-sample feedback via the reinforcement learning algorithm, we construct a specialized evaluator tailored to spoken dialogue models. 2) We introduce ChatReward-30K, a preference dataset used to train WavReward. ChatReward-30K includes both comprehension and generation aspects of spoken dialogue models. These scenarios span various tasks, such as text-based chats, nine acoustic attributes of instruction chats, and implicit chats. WavReward outperforms previous state-of-the-art evaluation models across multiple spoken dialogue scenarios, achieving a substantial improvement about Qwen2.5-Omni in objective accuracy from 55.1$\%$ to 91.5$\%$. In subjective A/B testing, WavReward also leads by a margin of 83$\%$. Comprehensive ablation studies confirm the necessity of each component of WavReward. All data and code will be publicly at https://github.com/jishengpeng/WavReward after the paper is accepted.
>
---
## 更新

#### [replaced 001] Deconstructing Jazz Piano Style Using Machine Learning
- **分类: cs.SD; cs.IR; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.05009v2](http://arxiv.org/pdf/2504.05009v2)**

> **作者:** Huw Cheston; Reuben Bance; Peter M. C. Harrison
>
> **备注:** Paper: 40 pages, 11 figures, 1 table; added information on training time + computation cost, corrections to Table 1. Supplementary material: 33 pages, 48 figures, 6 tables; corrections to Table S.5
>
> **摘要:** Artistic style has been studied for centuries, and recent advances in machine learning create new possibilities for understanding it computationally. However, ensuring that machine-learning models produce insights aligned with the interests of practitioners and critics remains a significant challenge. Here, we focus on musical style, which benefits from a rich theoretical and mathematical analysis tradition. We train a variety of supervised-learning models to identify 20 iconic jazz musicians across a carefully curated dataset of 84 hours of recordings, and interpret their decision-making processes. Our models include a novel multi-input architecture that enables four musical domains (melody, harmony, rhythm, and dynamics) to be analysed separately. These models enable us to address fundamental questions in music theory and also advance the state-of-the-art in music performer identification (94% accuracy across 20 classes). We release open-source implementations of our models and an accompanying web application for exploring musical styles.
>
---
#### [replaced 002] Fast Text-to-Audio Generation with Adversarial Post-Training
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.08175v2](http://arxiv.org/pdf/2505.08175v2)**

> **作者:** Zachary Novack; Zach Evans; Zack Zukowski; Josiah Taylor; CJ Carr; Julian Parker; Adnan Al-Sinan; Gian Marco Iodice; Julian McAuley; Taylor Berg-Kirkpatrick; Jordi Pons
>
> **摘要:** Text-to-audio systems, while increasingly performant, are slow at inference time, thus making their latency unpractical for many creative applications. We present Adversarial Relativistic-Contrastive (ARC) post-training, the first adversarial acceleration algorithm for diffusion/flow models not based on distillation. While past adversarial post-training methods have struggled to compare against their expensive distillation counterparts, ARC post-training is a simple procedure that (1) extends a recent relativistic adversarial formulation to diffusion/flow post-training and (2) combines it with a novel contrastive discriminator objective to encourage better prompt adherence. We pair ARC post-training with a number optimizations to Stable Audio Open and build a model capable of generating $\approx$12s of 44.1kHz stereo audio in $\approx$75ms on an H100, and $\approx$7s on a mobile edge-device, the fastest text-to-audio model to our knowledge.
>
---
#### [replaced 003] Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.11197v4](http://arxiv.org/pdf/2503.11197v4)**

> **作者:** Gang Li; Jizhong Liu; Heinrich Dinkel; Yadong Niu; Junbo Zhang; Jian Luan
>
> **摘要:** Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at https://github.com/xiaomi-research/r1-aqa and https://huggingface.co/mispeech/r1-aqa.
>
---
