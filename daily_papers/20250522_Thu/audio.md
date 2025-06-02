# 音频 cs.SD;  eess.SP

- **最新发布 24 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Replay Attacks Against Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于对抗攻击评估任务，解决音频深伪检测对重放攻击的脆弱性问题。通过构建ReplayDF数据集（含109种设备组合及多语言/声学条件），测试六种开源检测模型，发现其性能显著下降（最佳模型EER从4.7%升至18.2%），即使采用环境自适应重训仍存漏洞，最终公开数据集促进研究。**

- **链接: [http://arxiv.org/pdf/2505.14862v1](http://arxiv.org/pdf/2505.14862v1)**

> **作者:** Nicolas Müller; Piotr Kawa; Wei-Herng Choong; Adriana Stan; Aditya Tirumala Bukkapatnam; Karla Pizzi; Alexander Wagner; Philip Sperl
>
> **摘要:** We show how replay attacks undermine audio deepfake detection: By playing and re-recording deepfake audio through various speakers and microphones, we make spoofed samples appear authentic to the detection model. To study this phenomenon in more detail, we introduce ReplayDF, a dataset of recordings derived from M-AILABS and MLAAD, featuring 109 speaker-microphone combinations across six languages and four TTS models. It includes diverse acoustic conditions, some highly challenging for detection. Our analysis of six open-source detection models across five datasets reveals significant vulnerability, with the top-performing W2V2-AASIST model's Equal Error Rate (EER) surging from 4.7% to 18.2%. Even with adaptive Room Impulse Response (RIR) retraining, performance remains compromised with an 11.0% EER. We release ReplayDF for non-commercial research use.
>
---
#### [new 002] Hybrid Audio Detection Using Fine-Tuned Audio Spectrogram Transformers: A Dataset-Driven Evaluation of Mixed AI-Human Speech
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于AI语音检测任务，针对现实攻击中混合真实与克隆语音的检测难题，构建混合音频数据集并提出基于微调AST模型的检测方法，实现97%分类准确率，提升语音认证系统安全性。**

- **链接: [http://arxiv.org/pdf/2505.15136v1](http://arxiv.org/pdf/2505.15136v1)**

> **作者:** Kunyang Huang; Bin Hu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** The rapid advancement of artificial intelligence (AI) has enabled sophisticated audio generation and voice cloning technologies, posing significant security risks for applications reliant on voice authentication. While existing datasets and models primarily focus on distinguishing between human and fully synthetic speech, real-world attacks often involve audio that combines both genuine and cloned segments. To address this gap, we construct a novel hybrid audio dataset incorporating human, AI-generated, cloned, and mixed audio samples. We further propose fine-tuned Audio Spectrogram Transformer (AST)-based models tailored for detecting these complex acoustic patterns. Extensive experiments demonstrate that our approach significantly outperforms existing baselines in mixed-audio detection, achieving 97\% classification accuracy. Our findings highlight the importance of hybrid datasets and tailored models in advancing the robustness of speech-based authentication systems.
>
---
#### [new 003] Prosody-Adaptable Audio Codecs for Zero-Shot Voice Conversion via In-Context Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出基于VALLE-X框架的零样本语音转换模型，通过引入Prosody-Aware Codec Encoder（PACE）模块分离并优化语音韵律，解决现有方法在韵律控制与音色保持上的不足，提升语音自然度与灵活性，实验显示优于基线系统。**

- **链接: [http://arxiv.org/pdf/2505.15402v1](http://arxiv.org/pdf/2505.15402v1)**

> **作者:** Junchuan Zhao; Xintong Wang; Ye Wang
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Recent advances in discrete audio codecs have significantly improved speech representation modeling, while codec language models have enabled in-context learning for zero-shot speech synthesis. Inspired by this, we propose a voice conversion (VC) model within the VALLE-X framework, leveraging its strong in-context learning capabilities for speaker adaptation. To enhance prosody control, we introduce a prosody-aware audio codec encoder (PACE) module, which isolates and refines prosody from other sources, improving expressiveness and control. By integrating PACE into our VC model, we achieve greater flexibility in prosody manipulation while preserving speaker timbre. Experimental evaluation results demonstrate that our approach outperforms baseline VC systems in prosody preservation, timbre consistency, and overall naturalness, surpassing baseline VC systems.
>
---
#### [new 004] Discrete Audio Representations for Automated Audio Captioning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于自动音频描述任务，旨在解决无监督生成的音频标记在该任务中表现不佳的问题。通过提出基于监督训练的音频 tokenizer（以音频标记为目标），有效捕捉音频事件信息，在Clotho数据集上验证了其优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.14989v1](http://arxiv.org/pdf/2505.14989v1)**

> **作者:** Jingguang Tian; Haoqin Sun; Xinhui Hu; Xinkang Xu
>
> **备注:** Interspeech 2025
>
> **摘要:** Discrete audio representations, termed audio tokens, are broadly categorized into semantic and acoustic tokens, typically generated through unsupervised tokenization of continuous audio representations. However, their applicability to automated audio captioning (AAC) remains underexplored. This paper systematically investigates the viability of audio token-driven models for AAC through comparative analyses of various tokenization methods. Our findings reveal that audio tokenization leads to performance degradation in AAC models compared to those that directly utilize continuous audio representations. To address this issue, we introduce a supervised audio tokenizer trained with an audio tagging objective. Unlike unsupervised tokenizers, which lack explicit semantic understanding, the proposed tokenizer effectively captures audio event information. Experiments conducted on the Clotho dataset demonstrate that the proposed audio tokens outperform conventional audio tokens in the AAC task.
>
---
#### [new 005] GraphemeAug: A Systematic Approach to Synthesized Hard Negative Keyword Spotting Examples
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音关键词检测（KWS）任务，针对训练数据中边界样本不足导致模型性能受限的问题，提出GraphemeAug方法，通过字符级插入/删除/替换生成接近决策边界的对抗样本。实验显示该方法使合成数据AUC提升61%，同时保持正样本和环境噪声的识别质量。**

- **链接: [http://arxiv.org/pdf/2505.14814v1](http://arxiv.org/pdf/2505.14814v1)**

> **作者:** Harry Zhang; Kurt Partridge; Pai Zhu; Neng Chen; Hyun Jin Park; Dhruuv Agarwal; Quan Wang
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Spoken Keyword Spotting (KWS) is the task of distinguishing between the presence and absence of a keyword in audio. The accuracy of a KWS model hinges on its ability to correctly classify examples close to the keyword and non-keyword boundary. These boundary examples are often scarce in training data, limiting model performance. In this paper, we propose a method to systematically generate adversarial examples close to the decision boundary by making insertion/deletion/substitution edits on the keyword's graphemes. We evaluate this technique on held-out data for a popular keyword and show that the technique improves AUC on a dataset of synthetic hard negatives by 61% while maintaining quality on positives and ambient negative audio data.
>
---
#### [new 006] Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决自回归模型推理速度慢的问题。提出Speech Speculative Decoding（SSD）框架：通过轻量级草稿模型生成候选音素序列，再由目标模型并行验证，实现1.4倍加速且保持合成质量。**

- **链接: [http://arxiv.org/pdf/2505.15380v1](http://arxiv.org/pdf/2505.15380v1)**

> **作者:** Zijian Lin; Yang Zhang; Yougen Yuan; Yuming Yan; Jinjiang Liu; Zhiyong Wu; Pengfei Hu; Qun Yu
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Modern autoregressive speech synthesis models leveraging language models have demonstrated remarkable performance. However, the sequential nature of next token prediction in these models leads to significant latency, hindering their deployment in scenarios where inference speed is critical. In this work, we propose Speech Speculative Decoding (SSD), a novel framework for autoregressive speech synthesis acceleration. Specifically, our method employs a lightweight draft model to generate candidate token sequences, which are subsequently verified in parallel by the target model using the proposed SSD framework. Experimental results demonstrate that SSD achieves a significant speedup of 1.4x compared with conventional autoregressive decoding, while maintaining high fidelity and naturalness. Subjective evaluations further validate the effectiveness of SSD in preserving the perceptual quality of the target model while accelerating inference.
>
---
#### [new 007] Neurodyne: Neural Pitch Manipulation with Representation Learning and Cycle-Consistency GAN
- **分类: cs.SD; eess.AS**

- **简介: Neurodyne针对神经音调操控任务，解决特征解缠不准确和缺乏配对数据问题。通过对抗表征学习获得音调无关的潜在表示，并利用循环一致性GAN生成配对数据，提升合成质量同时保留原唱特征。**

- **链接: [http://arxiv.org/pdf/2505.15368v1](http://arxiv.org/pdf/2505.15368v1)**

> **作者:** Yicheng Gu; Chaoren Wang; Zhizheng Wu; Lauri Juvela
>
> **摘要:** Pitch manipulation is the process of producers adjusting the pitch of an audio segment to a specific key and intonation, which is essential in music production. Neural-network-based pitch-manipulation systems have been popular in recent years due to their superior synthesis quality compared to classical DSP methods. However, their performance is still limited due to their inaccurate feature disentanglement using source-filter models and the lack of paired in- and out-of-tune training data. This work proposes Neurodyne to address these issues. Specifically, Neurodyne uses adversarial representation learning to learn a pitch-independent latent representation to avoid inaccurate disentanglement and cycle-consistency training to create paired training data implicitly. Experimental results on global-key and template-based pitch manipulation demonstrate the effectiveness of the proposed system, marking improved synthesis quality while maintaining the original singer identity.
>
---
#### [new 008] AsynFusion: Towards Asynchronous Latent Consistency Models for Decoupled Whole-Body Audio-Driven Avatars
- **分类: cs.SD; cs.AI; cs.CV; eess.AS; 68T10**

- **简介: 该论文提出AsynFusion框架，解决现有语音驱动虚拟形象面部与手势生成不协调的问题。任务为生成全身自然动画，通过双分支DiT架构并行生成表情与手势，引入协同同步模块促进模态交互，采用异步采样降低计算量，实现实时高质量同步动画。**

- **链接: [http://arxiv.org/pdf/2505.15058v1](http://arxiv.org/pdf/2505.15058v1)**

> **作者:** Tianbao Zhang; Jian Zhao; Yuer Li; Zheng Zhu; Ping Hu; Zhaoxin Fan; Wenjun Wu; Xuelong Li
>
> **备注:** 11pages, conference
>
> **摘要:** Whole-body audio-driven avatar pose and expression generation is a critical task for creating lifelike digital humans and enhancing the capabilities of interactive virtual agents, with wide-ranging applications in virtual reality, digital entertainment, and remote communication. Existing approaches often generate audio-driven facial expressions and gestures independently, which introduces a significant limitation: the lack of seamless coordination between facial and gestural elements, resulting in less natural and cohesive animations. To address this limitation, we propose AsynFusion, a novel framework that leverages diffusion transformers to achieve harmonious expression and gesture synthesis. The proposed method is built upon a dual-branch DiT architecture, which enables the parallel generation of facial expressions and gestures. Within the model, we introduce a Cooperative Synchronization Module to facilitate bidirectional feature interaction between the two modalities, and an Asynchronous LCM Sampling strategy to reduce computational overhead while maintaining high-quality outputs. Extensive experiments demonstrate that AsynFusion achieves state-of-the-art performance in generating real-time, synchronized whole-body animations, consistently outperforming existing methods in both quantitative and qualitative evaluations.
>
---
#### [new 009] Moonbeam: A MIDI Foundation Model Using Both Absolute and Relative Music Attributes
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Moonbeam，一个基于Transformer的MIDI音乐基础模型，通过结合绝对与相对音乐属性（新型标记化方法+多维相对注意力MRA），解决音乐理解与生成任务中的信息捕捉问题。模型预训练81.6K小时MIDI数据，微调后在分类任务和条件生成（含音乐补全）中优于现有模型，代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.15559v1](http://arxiv.org/pdf/2505.15559v1)**

> **作者:** Zixun Guo; Simon Dixon
>
> **摘要:** Moonbeam is a transformer-based foundation model for symbolic music, pretrained on a large and diverse collection of MIDI data totaling 81.6K hours of music and 18 billion tokens. Moonbeam incorporates music-domain inductive biases by capturing both absolute and relative musical attributes through the introduction of a novel domain-knowledge-inspired tokenization method and Multidimensional Relative Attention (MRA), which captures relative music information without additional trainable parameters. Leveraging the pretrained Moonbeam, we propose 2 finetuning architectures with full anticipatory capabilities, targeting 2 categories of downstream tasks: symbolic music understanding and conditional music generation (including music infilling). Our model outperforms other large-scale pretrained music models in most cases in terms of accuracy and F1 score across 3 downstream music classification tasks on 4 datasets. Moreover, our finetuned conditional music generation model outperforms a strong transformer baseline with a REMI-like tokenizer. We open-source the code, pretrained model, and generated samples on Github.
>
---
#### [new 010] Voice-ENHANCE: Speech Restoration using a Diffusion-based Voice Conversion Framework
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Voice-ENHANCE系统，结合生成式语音修复（GSR）与扩散模型语音转换（VC），解决噪声环境下高质量语音修复问题。首先通过GSR模型降噪并恢复语音损伤，再利用干净说话人嵌入进一步优化，实现接近SOTA的语音质量。任务属语音增强，采用两阶段框架提升抗噪鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.15254v1](http://arxiv.org/pdf/2505.15254v1)**

> **作者:** Kyungguen Byun; Jason Filos; Erik Visser; Sunkuk Moon
>
> **备注:** 5 pages, 3 figures, Accepted to INTERSPEECH 2025
>
> **摘要:** We propose a speech enhancement system that combines speaker-agnostic speech restoration with voice conversion (VC) to obtain a studio-level quality speech signal. While voice conversion models are typically used to change speaker characteristics, they can also serve as a means of speech restoration when the target speaker is the same as the source speaker. However, since VC models are vulnerable to noisy conditions, we have included a generative speech restoration (GSR) model at the front end of our proposed system. The GSR model performs noise suppression and restores speech damage incurred during that process without knowledge about the target speaker. The VC stage then uses guidance from clean speaker embeddings to further restore the output speech. By employing this two-stage approach, we have achieved speech quality objective metric scores comparable to state-of-the-art (SOTA) methods across multiple datasets.
>
---
#### [new 011] SHEET: A Multi-purpose Open-source Speech Human Evaluation Estimation Toolkit
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在加速主观评估研究并提升模型性能。通过开发SHEET工具包，提供多模型训练脚本、预训练模型及跨数据集支持，并重新评估SSL-MOS模型，验证了其选择的SSL模型优于原版且接近当前最优方法。**

- **链接: [http://arxiv.org/pdf/2505.15061v1](http://arxiv.org/pdf/2505.15061v1)**

> **作者:** Wen-Chin Huang; Erica Cooper; Tomoki Toda
>
> **备注:** INTERSPEECH 2025. Codebase: https://github.com/unilight/sheet
>
> **摘要:** We introduce SHEET, a multi-purpose open-source toolkit designed to accelerate subjective speech quality assessment (SSQA) research. SHEET stands for the Speech Human Evaluation Estimation Toolkit, which focuses on data-driven deep neural network-based models trained to predict human-labeled quality scores of speech samples. SHEET provides comprehensive training and evaluation scripts, multi-dataset and multi-model support, as well as pre-trained models accessible via Torch Hub and HuggingFace Spaces. To demonstrate its capabilities, we re-evaluated SSL-MOS, a speech self-supervised learning (SSL)-based SSQA model widely used in recent scientific papers, on an extensive list of speech SSL models. Experiments were conducted on two representative SSQA datasets named BVCC and NISQA, and we identified the optimal speech SSL model, whose performance surpassed the original SSL-MOS implementation and was comparable to state-of-the-art methods.
>
---
#### [new 012] MIKU-PAL: An Automated and Standardized Multi-Modal Method for Speech Paralinguistic and Affect Labeling
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出MIKU-PAL，一种自动化多模态方法，解决大规模情感语音数据采集的一致性与成本问题。通过面部检测、追踪及多模态语言模型分析，实现高精度（68.5%）与高一致性（0.93 Fleiss kappa），标注26种情感类别，并发布131.2小时的MIKU-EmoBench数据集，用于情感TTS和视觉克隆基准。**

- **链接: [http://arxiv.org/pdf/2505.15772v1](http://arxiv.org/pdf/2505.15772v1)**

> **作者:** Cheng Yifan; Zhang Ruoyi; Shi Jiatong
>
> **备注:** Accepted by Interspeech
>
> **摘要:** Acquiring large-scale emotional speech data with strong consistency remains a challenge for speech synthesis. This paper presents MIKU-PAL, a fully automated multimodal pipeline for extracting high-consistency emotional speech from unlabeled video data. Leveraging face detection and tracking algorithms, we developed an automatic emotion analysis system using a multimodal large language model (MLLM). Our results demonstrate that MIKU-PAL can achieve human-level accuracy (68.5% on MELD) and superior consistency (0.93 Fleiss kappa score) while being much cheaper and faster than human annotation. With the high-quality, flexible, and consistent annotation from MIKU-PAL, we can annotate fine-grained speech emotion categories of up to 26 types, validated by human annotators with 83% rationality ratings. Based on our proposed system, we further released a fine-grained emotional speech dataset MIKU-EmoBench(131.2 hours) as a new benchmark for emotional text-to-speech and visual voice cloning.
>
---
#### [new 013] Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于大型音频语言模型（LAMs）安全评估任务，针对对抗性语音攻击（jailbreak）漏洞问题，构建了首个基准AJailBench，含1495个跨10类违规的对抗音频，并提出Audio Perturbation Toolkit生成优化扰动，揭示模型脆弱性，强调需改进防御机制。**

- **链接: [http://arxiv.org/pdf/2505.15406v1](http://arxiv.org/pdf/2505.15406v1)**

> **作者:** Zirui Song; Qian Jiang; Mingxuan Cui; Mingzhe Li; Lang Gao; Zeyu Zhang; Zixiang Xu; Yanbo Wang; Chenxi Wang; Guangxian Ouyang; Zhenhao Chen; Xiuying Chen
>
> **备注:** We release AJailBench, including both static and optimized adversarial data, to facilitate future research: https://github.com/mbzuai-nlp/AudioJailbreak
>
> **摘要:** The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms.
>
---
#### [new 014] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文针对低资源语言中构音障碍语音识别数据稀缺问题，提出通过语音转换模型将健康非英语语音转化为类似构音障碍语音，结合多语言ASR模型微调提升识别性能，实验验证该方法优于传统数据增强技术。**

- **链接: [http://arxiv.org/pdf/2505.14874v1](http://arxiv.org/pdf/2505.14874v1)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [new 015] Word Level Timestamp Generation for Automatic Speech Recognition and Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出基于Canary模型的单词级时间戳生成方法，用于自动语音识别与翻译任务。针对传统方法依赖外部对齐模块的问题，通过NeMo强制对齐器生成教师数据，引入<|timestamp|>标记直接预测词时间戳，在四语种中实现80-90%的精度（误差20-120ms），并扩展至翻译任务（误差约200ms），WER仅小幅下降。**

- **链接: [http://arxiv.org/pdf/2505.15646v1](http://arxiv.org/pdf/2505.15646v1)**

> **作者:** Ke Hu; Krishna Puvvada; Elena Rastorgueva; Zhehuai Chen; He Huang; Shuoyang Ding; Kunal Dhawan; Hainan Xu; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We introduce a data-driven approach for enabling word-level timestamp prediction in the Canary model. Accurate timestamp information is crucial for a variety of downstream tasks such as speech content retrieval and timed subtitles. While traditional hybrid systems and end-to-end (E2E) models may employ external modules for timestamp prediction, our approach eliminates the need for separate alignment mechanisms. By leveraging the NeMo Forced Aligner (NFA) as a teacher model, we generate word-level timestamps and train the Canary model to predict timestamps directly. We introduce a new <|timestamp|> token, enabling the Canary model to predict start and end timestamps for each word. Our method demonstrates precision and recall rates between 80% and 90%, with timestamp prediction errors ranging from 20 to 120 ms across four languages, with minimal WER degradation. Additionally, we extend our system to automatic speech translation (AST) tasks, achieving timestamp prediction errors around 200 milliseconds.
>
---
#### [new 016] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出TCSinger 2，一种多任务多语言零样本歌声合成模型。针对现有模型依赖音素/音符边界标注、过渡不自然及风格控制不足的问题，其通过模糊边界编码器优化过渡、定制音频编码器提取多模态风格特征、流式变换器提升音质与风格可控性，实验证明效果更优。**

- **链接: [http://arxiv.org/pdf/2505.14910v1](http://arxiv.org/pdf/2505.14910v1)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Dongyu Yao; Zhiyuan Zhu; Ziyue Jiang; Yuhan Wang; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Customizable multilingual zero-shot singing voice synthesis (SVS) has various potential applications in music composition and short video dubbing. However, existing SVS models overly depend on phoneme and note boundary annotations, limiting their robustness in zero-shot scenarios and producing poor transitions between phonemes and notes. Moreover, they also lack effective multi-level style control via diverse prompts. To overcome these challenges, we introduce TCSinger 2, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts. TCSinger 2 mainly includes three key modules: 1) Blurred Boundary Content (BBC) Encoder, predicts duration, extends content embedding, and applies masking to the boundaries to enable smooth transitions. 2) Custom Audio Encoder, uses contrastive learning to extract aligned representations from singing, speech, and textual prompts. 3) Flow-based Custom Transformer, leverages Cus-MOE, with F0 supervision, enhancing both the synthesis quality and style modeling of the generated singing voice. Experimental results show that TCSinger 2 outperforms baseline models in both subjective and objective metrics across multiple related tasks.
>
---
#### [new 017] MHANet: Multi-scale Hybrid Attention Network for Auditory Attention Detection
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文提出MHANet，用于听觉注意检测（AAD），解决现有方法忽略EEG多尺度时空依赖的问题。通过多尺度混合注意力（MHA）与时空卷积（STC）模块，结合通道、多尺度时域及全局注意力，提取时空特征。实验显示其参数更少但性能最优。**

- **链接: [http://arxiv.org/pdf/2505.15364v1](http://arxiv.org/pdf/2505.15364v1)**

> **作者:** Lu Li; Cunhang Fan; Hongyu Zhang; Jingjing Zhang; Xiaoke Yang; Jian Zhou; Zhao Lv
>
> **摘要:** Auditory attention detection (AAD) aims to detect the target speaker in a multi-talker environment from brain signals, such as electroencephalography (EEG), which has made great progress. However, most AAD methods solely utilize attention mechanisms sequentially and overlook valuable multi-scale contextual information within EEG signals, limiting their ability to capture long-short range spatiotemporal dependencies simultaneously. To address these issues, this paper proposes a multi-scale hybrid attention network (MHANet) for AAD, which consists of the multi-scale hybrid attention (MHA) module and the spatiotemporal convolution (STC) module. Specifically, MHA combines channel attention and multi-scale temporal and global attention mechanisms. This effectively extracts multi-scale temporal patterns within EEG signals and captures long-short range spatiotemporal dependencies simultaneously. To further improve the performance of AAD, STC utilizes temporal and spatial convolutions to aggregate expressive spatiotemporal representations. Experimental results show that the proposed MHANet achieves state-of-the-art performance with fewer trainable parameters across three datasets, 3 times lower than that of the most advanced model. Code is available at: https://github.com/fchest/MHANet.
>
---
#### [new 018] Towards Pre-training an Effective Respiratory Audio Foundation Model
- **分类: eess.AS; cs.SD; 68T07; J.3**

- **简介: 该论文聚焦呼吸音基础模型预训练任务，旨在解决小规模非多样化数据集下模型效果不足的问题。通过对比多种预训练方案，发现通用音频数据集（AudioSet）预训练优于专用呼吸音数据集，结合两者并保留频率特征可提升性能，最终在OPERA基准创当前最优。**

- **链接: [http://arxiv.org/pdf/2505.15307v1](http://arxiv.org/pdf/2505.15307v1)**

> **作者:** Daisuke Niizumi; Daiki Takeuchi; Masahiro Yasuda; Binh Thien Nguyen; Yasunori Ohishi; Noboru Harada
>
> **备注:** 5 pages, 2 figures, 4 tables, Accepted by Interspeech 2025
>
> **摘要:** Recent advancements in foundation models have sparked interest in respiratory audio foundation models. However, the effectiveness of applying conventional pre-training schemes to datasets that are small-sized and lack diversity has not been sufficiently verified. This study aims to explore better pre-training practices for respiratory sounds by comparing numerous pre-trained audio models. Our investigation reveals that models pre-trained on AudioSet, a general audio dataset, are more effective than the models specifically pre-trained on respiratory sounds. Moreover, combining AudioSet and respiratory sound datasets for further pre-training enhances performance, and preserving the frequency-wise information when aggregating features is vital. Along with more insights found in the experiments, we establish a new state-of-the-art for the OPERA benchmark, contributing to advancing respiratory audio foundation models. Our code is available online at https://github.com/nttcslab/eval-audio-repr/tree/main/plugin/OPERA.
>
---
#### [new 019] Decoding Phone Pairs from MEG Signals Across Speech Modalities
- **分类: cs.CL; cs.LG; cs.NE; cs.SD; eess.AS; I.2.6; I.5.1**

- **简介: 该研究通过MEG信号解码语音中的音素对，对比了主动说话与被动听觉任务的神经机制差异。旨在探索更优的脑电信号解码方法以改进脑机接口。研究使用17人数据，比较机器学习模型效果，发现主动说话解码准确率更高（76.6%），弹性网络模型最优，低频脑电波贡献显著，但需解决伪影干扰问题。**

- **链接: [http://arxiv.org/pdf/2505.15355v1](http://arxiv.org/pdf/2505.15355v1)**

> **作者:** Xabier de Zuazo; Eva Navas; Ibon Saratxaga; Mathieu Bourguignon; Nicola Molinaro
>
> **备注:** 21 pages, 4 figures, 1 graphical abstract, submitted to Computer Speech and Language (special issue on Iberian Languages)
>
> **摘要:** Understanding the neural mechanisms underlying speech production is essential for both advancing cognitive neuroscience theory and developing practical communication technologies. In this study, we investigated magnetoencephalography signals to decode phones from brain activity during speech production and perception (passive listening and voice playback) tasks. Using a dataset comprising 17 participants, we performed pairwise phone classification, extending our analysis to 15 phonetic pairs. Multiple machine learning approaches, including regularized linear models and neural network architectures, were compared to determine their effectiveness in decoding phonetic information. Our results demonstrate significantly higher decoding accuracy during speech production (76.6%) compared to passive listening and playback modalities (~51%), emphasizing the richer neural information available during overt speech. Among the models, the Elastic Net classifier consistently outperformed more complex neural networks, highlighting the effectiveness of traditional regularization techniques when applied to limited and high-dimensional MEG datasets. Besides, analysis of specific brain frequency bands revealed that low-frequency oscillations, particularly Delta (0.2-3 Hz) and Theta (4-7 Hz), contributed the most substantially to decoding accuracy, suggesting that these bands encode critical speech production-related neural processes. Despite using advanced denoising methods, it remains unclear whether decoding solely reflects neural activity or if residual muscular or movement artifacts also contributed, indicating the need for further methodological refinement. Overall, our findings underline the critical importance of examining overt speech production paradigms, which, despite their complexity, offer opportunities to improve brain-computer interfaces to help individuals with severe speech impairments.
>
---
#### [new 020] EASY: Emotion-aware Speaker Anonymization via Factorized Distillation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出EASY框架，解决现有说话人匿名化系统忽略情感保留的问题。通过顺序解纠缠分离说话人身份、语言内容和情感表征，采用因子蒸馏独立约束各子空间，实现隐私保护同时保留原始情感和语言信息，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.15004v1](http://arxiv.org/pdf/2505.15004v1)**

> **作者:** Jixun Yao; Hexin Liu; Eng Siong Chng; Lei Xie
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Emotion plays a significant role in speech interaction, conveyed through tone, pitch, and rhythm, enabling the expression of feelings and intentions beyond words to create a more personalized experience. However, most existing speaker anonymization systems employ parallel disentanglement methods, which only separate speech into linguistic content and speaker identity, often neglecting the preservation of the original emotional state. In this study, we introduce EASY, an emotion-aware speaker anonymization framework. EASY employs a novel sequential disentanglement process to disentangle speaker identity, linguistic content, and emotional representation, modeling each speech attribute in distinct subspaces through a factorized distillation approach. By independently constraining speaker identity and emotional representation, EASY minimizes information leakage, enhancing privacy protection while preserving original linguistic content and emotional state. Experimental results on the VoicePrivacy Challenge official datasets demonstrate that our proposed approach outperforms all baseline systems, effectively protecting speaker privacy while maintaining linguistic content and emotional state.
>
---
#### [new 021] Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出新型双工语音到语音模型，解决现有语音模型仅支持轮流交互、缺乏实时适应（如用户打断）的问题。通过连续输入输出、流式编码器及分离架构，提升推理、轮流与打断能力，降低比特率并简化训练流程，为首个开源方案。**

- **链接: [http://arxiv.org/pdf/2505.15670v1](http://arxiv.org/pdf/2505.15670v1)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
#### [new 022] QUADS: QUAntized Distillation Framework for Efficient Speech Language Understanding
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音语言理解（SLU）模型压缩任务，针对现有方法分别进行知识蒸馏和量化导致精度与效率失衡的问题，提出QUADS框架，通过多阶段联合优化蒸馏与量化，提升低比特环境下的模型效率与精度，在SLURP/FSC数据集上实现高准确率同时压缩模型规模83-700倍，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2505.14723v1](http://arxiv.org/pdf/2505.14723v1)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Spoken Language Understanding (SLU) systems must balance performance and efficiency, particularly in resource-constrained environments. Existing methods apply distillation and quantization separately, leading to suboptimal compression as distillation ignores quantization constraints. We propose QUADS, a unified framework that optimizes both through multi-stage training with a pre-tuned model, enhancing adaptability to low-bit regimes while maintaining accuracy. QUADS achieves 71.13\% accuracy on SLURP and 99.20\% on FSC, with only minor degradations of up to 5.56\% compared to state-of-the-art models. Additionally, it reduces computational complexity by 60--73$\times$ (GMACs) and model size by 83--700$\times$, demonstrating strong robustness under extreme quantization. These results establish QUADS as a highly efficient solution for real-world, resource-constrained SLU applications.
>
---
#### [new 023] Segmentation-Variant Codebooks for Preservation of Paralinguistic and Prosodic Information
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文针对语音自监督模型量化导致韵律及副语言信息丢失的问题，提出分段可变码本（SVC）方法，通过在帧、音素、词、utterance等不同语言单元进行多流量化，有效保留情感、重音等信息。实验表明其优于传统方法，在重合成任务中提升风格表现与质量同时保持可懂度。**

- **链接: [http://arxiv.org/pdf/2505.15667v1](http://arxiv.org/pdf/2505.15667v1)**

> **作者:** Nicholas Sanders; Yuanchao Li; Korin Richmond; Simon King
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Quantization in SSL speech models (e.g., HuBERT) improves compression and performance in tasks like language modeling, resynthesis, and text-to-speech but often discards prosodic and paralinguistic information (e.g., emotion, prominence). While increasing codebook size mitigates some loss, it inefficiently raises bitrates. We propose Segmentation-Variant Codebooks (SVCs), which quantize speech at distinct linguistic units (frame, phone, word, utterance), factorizing it into multiple streams of segment-specific discrete features. Our results show that SVCs are significantly more effective at preserving prosodic and paralinguistic information across probing tasks. Additionally, we find that pooling before rather than after discretization better retains segment-level information. Resynthesis experiments further confirm improved style realization and slightly improved quality while preserving intelligibility.
>
---
#### [new 024] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文针对无文本语音到语音翻译任务，解决跨模态语言特征提取与跨语言长序列对齐问题。提出基于n-gram的单元语言表示，并通过多任务学习结合任务提示模型缓解源-目标语言冲突，实验显示显著性能提升。**

- **链接: [http://arxiv.org/pdf/2505.15333v1](http://arxiv.org/pdf/2505.15333v1)**

> **作者:** Yuhao Zhang; Xiangnan Ma; Kaiqi Kou; Peizhuo Liu; Weiqiao Shan; Benyou Wang; Tong Xiao; Yuxin Huang; Zhengtao Yu; Jingbo Zhu
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** The success of building textless speech-to-speech translation (S2ST) models has attracted much attention. However, S2ST still faces two main challenges: 1) extracting linguistic features for various speech signals, called cross-modal (CM), and 2) learning alignment of difference languages in long sequences, called cross-lingual (CL). We propose the unit language to overcome the two modeling challenges. The unit language can be considered a text-like representation format, constructed using $n$-gram language modeling. We implement multi-task learning to utilize the unit language in guiding the speech modeling process. Our initial results reveal a conflict when applying source and target unit languages simultaneously. We propose task prompt modeling to mitigate this conflict. We conduct experiments on four languages of the Voxpupil dataset. Our method demonstrates significant improvements over a strong baseline and achieves performance comparable to models trained with text.
>
---
## 更新

#### [replaced 001] Streaming Sequence Transduction through Dynamic Compression
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.01172v3](http://arxiv.org/pdf/2402.01172v3)**

> **作者:** Weiting Tan; Yunmo Chen; Tongfei Chen; Guanghui Qin; Haoran Xu; Heidi C. Zhang; Benjamin Van Durme; Philipp Koehn
>
> **备注:** IWSLT 2025
>
> **摘要:** We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
>
---
#### [replaced 002] dMel: Speech Tokenization made Simple
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.15835v3](http://arxiv.org/pdf/2407.15835v3)**

> **作者:** Richard He Bai; Tatiana Likhomanenko; Ruixiang Zhang; Zijin Gu; Zakaria Aldeneh; Navdeep Jaitly
>
> **备注:** preprint
>
> **摘要:** Large language models have revolutionized natural language processing by leveraging self-supervised pretraining on vast textual data. Inspired by this success, researchers have investigated various compression-based speech tokenization methods to discretize continuous speech signals, enabling the application of language modeling techniques to discrete tokens. However, audio compressor introduces additional complexity and computational cost, and often fail on out-of-domain audio signals. In this work, we introduce a novel speech representation (dmel) that discretizes mel-filterbank channels into intensity bins, creating a simpler yet more effective representation compared to existing speech tokenization methods. Our approach demonstrates superior performance in preserving audio content, robustness to out-of-domain data, and offers a training-free, natural, and streamable representation. To address the high-dimensional nature of log-mel spectrograms, we propose an efficient parallel encoding and decoding method for high-dimensional tokens using an LM-style transformer architecture. This innovation enables us to develop RichTTS and RichASR, two models sharing the same architecture while achieving comparable or better results than specialized existing methods. Our results demonstrate the effectiveness of dmel in achieving high performance on both speech synthesis and recognition tasks within a unified framework, paving the way for efficient and effective joint modeling of speech and text.
>
---
#### [replaced 003] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.16344v3](http://arxiv.org/pdf/2501.16344v3)**

> **作者:** Rajath Rao; Adithya Ganesan; Oscar Kjell; Jonah Luby; Akshay Raghavan; Scott Feltman; Whitney Ringwald; Ryan L. Boyd; Benjamin Luft; Camilo Ruggero; Neville Ryant; Roman Kotov; H. Andrew Schwartz
>
> **备注:** 16 pages, 8 figures, ACL 2025
>
> **摘要:** Current speech encoding pipelines often rely on an additional text-based LM to get robust representations of human communication, even though SotA speech-to-text models often have a LM within. This work proposes an approach to improve the LM within an audio model such that the subsequent text-LM is unnecessary. We introduce WhiSPA (Whisper with Semantic and Psychological Alignment), which leverages a novel audio training objective: contrastive loss with a language model embedding as a teacher. Using over 500k speech segments from mental health audio interviews, we evaluate the utility of aligning Whisper's latent space with semantic representations from a text autoencoder (SBERT) and lexically derived embeddings of basic psychological dimensions: emotion and personality. Over self-supervised affective tasks and downstream psychological tasks, WhiSPA surpasses current speech encoders, achieving an average error reduction of 73.4% and 83.8%, respectively. WhiSPA demonstrates that it is not always necessary to run a subsequent text LM on speech-to-text output in order to get a rich psychological representation of human communication.
>
---
#### [replaced 004] CAV-MAE Sync: Improving Contrastive Audio-Visual Mask Autoencoders via Fine-Grained Alignment
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.01237v2](http://arxiv.org/pdf/2505.01237v2)**

> **作者:** Edson Araujo; Andrew Rouditchenko; Yuan Gong; Saurabhchand Bhati; Samuel Thomas; Brian Kingsbury; Leonid Karlinsky; Rogerio Feris; James R. Glass; Hilde Kuehne
>
> **备注:** To be published at CVPR 2025, code available at https://github.com/edsonroteia/cav-mae-sync
>
> **摘要:** Recent advances in audio-visual learning have shown promising results in learning representations across modalities. However, most approaches rely on global audio representations that fail to capture fine-grained temporal correspondences with visual frames. Additionally, existing methods often struggle with conflicting optimization objectives when trying to jointly learn reconstruction and cross-modal alignment. In this work, we propose CAV-MAE Sync as a simple yet effective extension of the original CAV-MAE framework for self-supervised audio-visual learning. We address three key challenges: First, we tackle the granularity mismatch between modalities by treating audio as a temporal sequence aligned with video frames, rather than using global representations. Second, we resolve conflicting optimization goals by separating contrastive and reconstruction objectives through dedicated global tokens. Third, we improve spatial localization by introducing learnable register tokens that reduce semantic load on patch tokens. We evaluate the proposed approach on AudioSet, VGG Sound, and the ADE20K Sound dataset on zero-shot retrieval, classification and localization tasks demonstrating state-of-the-art performance and outperforming more complex architectures.
>
---
#### [replaced 005] Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14449v2](http://arxiv.org/pdf/2505.14449v2)**

> **作者:** Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** Accepted by InterSpeech 2025. 7 pages including 2 pages of appendix
>
> **摘要:** While subgroup disparities and performance bias are increasingly studied in computational research, fairness in categorical Speech Emotion Recognition (SER) remains underexplored. Existing methods often rely on explicit demographic labels, which are difficult to obtain due to privacy concerns. To address this limitation, we introduce an Implicit Demography Inference (IDI) module that leverages pseudo-labeling from a pre-trained model and unsupervised learning using k-means clustering to mitigate bias in SER. Our experiments show that pseudo-labeling IDI reduces subgroup disparities, improving fairness metrics by over 33% with less than a 3% decrease in SER accuracy. Also, the unsupervised IDI yields more than a 26% improvement in fairness metrics with a drop of less than 4% in SER performance. Further analyses reveal that the unsupervised IDI consistently mitigates race and age disparities, demonstrating its potential in scenarios where explicit demographic information is unavailable.
>
---
#### [replaced 006] Enhancing Intelligibility for Generative Target Speech Extraction via Joint Optimization with Target Speaker ASR
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.14477v2](http://arxiv.org/pdf/2501.14477v2)**

> **作者:** Hao Ma; Rujin Chen; Xiao-Lei Zhang; Ju Liu; Xuelong Li
>
> **备注:** Submitted to IEEE Signal Processing Letters
>
> **摘要:** Target speech extraction (TSE) isolates the speech of a specific speaker from a multi-talker overlapped speech mixture. Most existing TSE models rely on discriminative methods, typically predicting a time-frequency spectrogram mask for the target speech. However, imperfections in these masks often result in over-/under-suppression of target/non-target speech, degrading perceptual quality. Generative methods, by contrast, re-synthesize target speech based on the mixture and target speaker cues, achieving superior perceptual quality. Nevertheless, these methods often overlook speech intelligibility, leading to alterations or loss of semantic content in the re-synthesized speech. Inspired by the Whisper model's success in target speaker ASR, we propose a generative TSE framework based on the pre-trained Whisper model to address the above issues. This framework integrates semantic modeling with flow-based acoustic modeling to achieve both high intelligibility and perceptual quality. Results from multiple benchmarks demonstrate that the proposed method outperforms existing generative and discriminative baselines. We present speech samples on https://aisaka0v0.github.io/GenerativeTSE_demo/.
>
---
#### [replaced 007] Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach
- **分类: eess.AS; cs.CV; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14336v2](http://arxiv.org/pdf/2505.14336v2)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Stavros Petridis; Daniele Falavigna; Alessio Brutti
>
> **备注:** Interspeech 2025
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
>
---
#### [replaced 008] Solid State Bus-Comp: A Large-Scale and Diverse Dataset for Dynamic Range Compressor Virtual Analog Modeling
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2504.04589v2](http://arxiv.org/pdf/2504.04589v2)**

> **作者:** Yicheng Gu; Runsong Zhang; Lauri Juvela; Zhizheng Wu
>
> **摘要:** Virtual Analog (VA) modeling aims to simulate the behavior of hardware circuits via algorithms to replicate their tone digitally. Dynamic Range Compressor (DRC) is an audio processing module that controls the dynamics of a track by reducing and amplifying the volumes of loud and quiet sounds, which is essential in music production. In recent years, neural-network-based VA modeling has shown great potential in producing high-fidelity models. However, due to the lack of data quantity and diversity, their generalization ability in different parameter settings and input sounds is still limited. To tackle this problem, we present Solid State Bus-Comp, the first large-scale and diverse dataset for modeling the classical VCA compressor -- SSL 500 G-Bus. Specifically, we manually collected 175 unmastered songs from the Cambridge Multitrack Library. We recorded the compressed audio in 220 parameter combinations, resulting in an extensive 2528-hour dataset with diverse genres, instruments, tempos, and keys. Moreover, to facilitate the use of our proposed dataset, we conducted benchmark experiments in various open-sourced black-box and grey-box models, as well as white-box plugins. We also conducted ablation studies in different data subsets to illustrate the effectiveness of the improved data diversity and quantity. The dataset and demos are on our project page: https://www.yichenggu.com/SolidStateBusComp/.
>
---
#### [replaced 009] Recreating Neural Activity During Speech Production with Language and Speech Model Embeddings
- **分类: cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14074v2](http://arxiv.org/pdf/2505.14074v2)**

> **作者:** Owais Mujtaba Khanday; Pablo Rodroguez San Esteban; Zubair Ahmad Lone; Marc Ouellet; Jose Andres Gonzalez Lopez
>
> **备注:** Accepted for presentation at Interspeech2025
>
> **摘要:** Understanding how neural activity encodes speech and language production is a fundamental challenge in neuroscience and artificial intelligence. This study investigates whether embeddings from large-scale, self-supervised language and speech models can effectively reconstruct high-gamma neural activity characteristics, key indicators of cortical processing, recorded during speech production. We leverage pre-trained embeddings from deep learning models trained on linguistic and acoustic data to represent high-level speech features and map them onto these high-gamma signals. We analyze the extent to which these embeddings preserve the spatio-temporal dynamics of brain activity. Reconstructed neural signals are evaluated against high-gamma ground-truth activity using correlation metrics and signal reconstruction quality assessments. The results indicate that high-gamma activity can be effectively reconstructed using large language and speech model embeddings in all study participants, generating Pearson's correlation coefficients ranging from 0.79 to 0.99.
>
---
#### [replaced 010] MatchDance: Collaborative Mamba-Transformer Architecture Matching for High-Quality 3D Dance Synthesis
- **分类: cs.SD; cs.GR; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14222v2](http://arxiv.org/pdf/2505.14222v2)**

> **作者:** Kaixing Yang; Xulong Tang; Yuxuan Hu; Jiahao Yang; Hongyan Liu; Qinnan Zhang; Jun He; Zhaoxin Fan
>
> **摘要:** Music-to-dance generation represents a challenging yet pivotal task at the intersection of choreography, virtual reality, and creative content generation. Despite its significance, existing methods face substantial limitation in achieving choreographic consistency. To address the challenge, we propose MatchDance, a novel framework for music-to-dance generation that constructs a latent representation to enhance choreographic consistency. MatchDance employs a two-stage design: (1) a Kinematic-Dynamic-based Quantization Stage (KDQS), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) with kinematic-dynamic constraints and reconstructs them with high fidelity, and (2) a Hybrid Music-to-Dance Generation Stage(HMDGS), which uses a Mamba-Transformer hybrid architecture to map music into the latent representation, followed by the KDQS decoder to generate 3D dance motions. Additionally, a music-dance retrieval framework and comprehensive metrics are introduced for evaluation. Extensive experiments on the FineDance dataset demonstrate state-of-the-art performance. Code will be released upon acceptance.
>
---
#### [replaced 011] AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models
- **分类: cs.CR; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14103v2](http://arxiv.org/pdf/2505.14103v2)**

> **作者:** Guangke Chen; Fu Song; Zhe Zhao; Xiaojun Jia; Yang Liu; Yanchen Qiao; Weizhe Zhang
>
> **摘要:** Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website https://audiojailbreak.github.io/AudioJailbreak.
>
---
#### [replaced 012] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12332v2](http://arxiv.org/pdf/2505.12332v2)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning.
>
---
#### [replaced 013] Hearing from Silence: Reasoning Audio Descriptions from Silent Videos via Vision-Language Model
- **分类: cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13062v2](http://arxiv.org/pdf/2505.13062v2)**

> **作者:** Yong Ren; Chenxing Li; Le Xu; Hao Gu; Duzhen Zhang; Yujie Chen; Manjie Xu; Ruibo Fu; Shan Yang; Dong Yu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Humans can intuitively infer sounds from silent videos, but whether multimodal large language models can perform modal-mismatch reasoning without accessing target modalities remains relatively unexplored. Current text-assisted-video-to-audio (VT2A) methods excel in video foley tasks but struggle to acquire audio descriptions during inference. We introduce the task of Reasoning Audio Descriptions from Silent Videos (SVAD) to address this challenge and investigate vision-language models' (VLMs) capabilities on this task. To further enhance the VLMs' reasoning capacity for the SVAD task, we construct a CoT-AudioCaps dataset and propose a Chain-of-Thought-based supervised fine-tuning strategy. Experiments on SVAD and subsequent VT2A tasks demonstrate our method's effectiveness in two key aspects: significantly improving VLMs' modal-mismatch reasoning for SVAD and effectively addressing the challenge of acquiring audio descriptions during VT2A inference.
>
---
#### [replaced 014] LSCodec: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.15764v3](http://arxiv.org/pdf/2410.15764v3)**

> **作者:** Yiwei Guo; Zhihan Li; Chenpeng Du; Hankun Wang; Xie Chen; Kai Yu
>
> **备注:** 5 pages, 2 figures, 3 tables. Demo page: https://cantabile-kwok.github.io/LSCodec/. Accepted to Interspeech 2025
>
> **摘要:** Although discrete speech tokens have exhibited strong potential for language model-based speech generation, their high bitrates and redundant timbre information restrict the development of such models. In this work, we propose LSCodec, a discrete speech codec that has both low bitrate and speaker decoupling ability. LSCodec adopts a multi-stage unsupervised training framework with a speaker perturbation technique. A continuous information bottleneck is first established, followed by vector quantization that produces a discrete speaker-decoupled space. A discrete token vocoder finally refines acoustic details from LSCodec. By reconstruction evaluations, LSCodec demonstrates superior intelligibility and audio quality with only a single codebook and smaller vocabulary size than baselines. Voice conversion and speaker probing experiments prove the excellent speaker disentanglement of LSCodec, and ablation study verifies the effectiveness of the proposed training framework.
>
---
#### [replaced 015] Unified Microphone Conversion: Many-to-Many Device Mapping via Feature-wise Linear Modulation
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.18322v3](http://arxiv.org/pdf/2410.18322v3)**

> **作者:** Myeonghoon Ryu; Hongseok Oh; Suji Lee; Han Park
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We present Unified Microphone Conversion, a unified generative framework designed to bolster sound event classification (SEC) systems against device variability. While our prior CycleGAN-based methods effectively simulate device characteristics, they require separate models for each device pair, limiting scalability. Our approach overcomes this constraint by conditioning the generator on frequency response data, enabling many-to-many device mappings through unpaired training. We integrate frequency-response information via Feature-wise Linear Modulation, further enhancing scalability. Additionally, incorporating synthetic frequency response differences improves the applicability of our framework for real-world application. Experimental results show that our method outperforms the state-of-the-art by 2.6% and reduces variability by 0.8% in macro-average F1 score.
>
---
