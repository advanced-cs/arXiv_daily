# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Speaker Disentanglement of Speech Pre-trained Model Based on Interpretability
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音预训练模型中的说话人解耦任务，旨在解决内容与音色信息难以分离的问题。通过引入可解释性方法，提出InterpTRQE-SptME基准和InterpTF-SptME解耦方法，实现音色信息去除并保留内容，提升语音处理任务性能。**

- **链接: [http://arxiv.org/pdf/2507.17851v1](http://arxiv.org/pdf/2507.17851v1)**

> **作者:** Xiaoxu Zhu; Junhua Li
>
> **备注:** 20 pages, 9 figures, 2 tables
>
> **摘要:** Speech pretrained models contain task-specific information across different layers, but decoupling content and timbre information remains challenging as removing speaker-specific information often causes content loss. Current research lacks direct metrics to quantify timbre residual in model encodings, relying on indirect evaluation through downstream tasks. This paper addresses these challenges through interpretability-based speaker disentanglement in speech pretraining models. We quantitatively evaluate timbre residual in model embeddings and improve speaker disentanglement using interpretive representations. Our contributions include: (1) InterpTRQE-SptME Benchmark - a timbre residual recognition framework using interpretability. The benchmark concatenates content embeddings with timbre embeddings for speaker classification, then applies Gradient SHAP Explainer to quantify timbre residual. We evaluate seven speech pretraining model variations. (2) InterpTF-SptME method - an interpretability-based timbre filtering approach using SHAP Noise and SHAP Cropping techniques. This model-agnostic method transforms intermediate encodings to remove timbre while preserving content. Experiments on VCTK dataset with HuBERT LARGE demonstrate successful content preservation and significant speaker disentanglement optimization. Results show the SHAP Noise method can reduce timbre residual from 18.05% to near 0% while maintaining content integrity, contributing to enhanced performance in content-related speech processing tasks and preventing timbre privacy leakage.
>
---
#### [new 002] Resnet-conformer network with shared weights and attention mechanism for sound event localization, detection, and distance estimation
- **分类: cs.SD; eess.AS**

- **简介: 该论文参与DCASE 2024 Task 3A，解决音频声事件定位、检测与距离估计问题。作者提出基于ResNet-Conformer的网络结构，结合共享权重和注意力机制，使用log-mel谱图和强度向量作为输入，并采用多种数据增强方法。在测试集上取得了40.2%的F-score、17.7度的DOA误差和0.32的RDE结果。**

- **链接: [http://arxiv.org/pdf/2507.17941v1](http://arxiv.org/pdf/2507.17941v1)**

> **作者:** Quoc Thinh Vo; David Han
>
> **备注:** This paper has been submitted as a technical report outlining our approach to Task 3A of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2024 and can be found in DCASE2024 technical reports
>
> **摘要:** This technical report outlines our approach to Task 3A of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2024, focusing on Sound Event Localization and Detection (SELD). SELD provides valuable insights by estimating sound event localization and detection, aiding in various machine cognition tasks such as environmental inference, navigation, and other sound localization-related applications. This year's challenge evaluates models using either audio-only (Track A) or audiovisual (Track B) inputs on annotated recordings of real sound scenes. A notable change this year is the introduction of distance estimation, with evaluation metrics adjusted accordingly for a comprehensive assessment. Our submission is for Task A of the Challenge, which focuses on the audio-only track. Our approach utilizes log-mel spectrograms, intensity vectors, and employs multiple data augmentations. We proposed an EINV2-based [1] network architecture, achieving improved results: an F-score of 40.2%, Angular Error (DOA) of 17.7 degrees, and Relative Distance Error (RDE) of 0.32 on the test set of the Development Dataset [2 ,3].
>
---
#### [new 003] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 论文研究了歌词到歌曲生成模型的数据记忆漏洞，提出了一种对抗攻击方法APT，通过同音替换改变歌词语义但保留语音结构。结果显示，模型如SUNO和YuE仍能生成与训练数据高度相似的音频，且这种语音修改还能触发文本到视频模型的视觉记忆，引发版权与安全问题。**

- **链接: [http://arxiv.org/pdf/2507.17937v1](http://arxiv.org/pdf/2507.17937v1)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (jrohsc.github.io/music_attack/).
>
---
#### [new 004] DIFFA: Large Language Diffusion Models Can Listen and Understand
- **分类: cs.SD**

- **简介: 该论文提出DIFFA，首个基于扩散模型的大型音文模型，用于解决语音理解任务。通过结合冻结扩散语言模型与双适配器架构，并采用两阶段训练方法，在少量数据上实现高效训练，超越自回归模型表现，探索扩散模型在语音理解中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.18452v1](http://arxiv.org/pdf/2507.18452v1)**

> **作者:** Jiaming Zhou; Hongjie Chen; Shiwan Zhao; Jian Kang; Jie Li; Enzhi Wang; Yujie Guo; Haoqin Sun; Hui Wang; Aobo Kong; Yong Qin; Xuelong Li
>
> **摘要:** Recent advances in Large language models (LLMs) have shown remarkable capabilities across textual and multimodal domains. In parallel, diffusion-based language models have emerged as a promising alternative to the autoregressive paradigm, offering improved controllability, bidirectional context modeling, and robust generation. However, their application to the audio modality remains underexplored. In this work, we introduce \textbf{DIFFA}, the first diffusion-based Large Audio-Language Model designed to perform spoken language understanding. DIFFA integrates a frozen diffusion language model with a lightweight dual-adapter architecture that bridges speech understanding and natural language reasoning. We employ a two-stage training pipeline: first, aligning semantic representations via an ASR objective; then, learning instruction-following abilities through synthetic audio-caption pairs automatically generated by prompting LLMs. Despite being trained on only 960 hours of ASR and 127 hours of synthetic instruction data, DIFFA demonstrates competitive performance on major benchmarks, including MMSU, MMAU, and VoiceBench, outperforming several autoregressive open-source baselines. Our results reveal the potential of diffusion-based language models for efficient and scalable audio understanding, opening a new direction for speech-driven AI. Our code will be available at https://github.com/NKU-HLT/DIFFA.git.
>
---
#### [new 005] The TEA-ASLP System for Multilingual Conversational Speech Recognition and Speech Diarization in MLC-SLM 2025 Challenge
- **分类: cs.SD; eess.AS**

- **简介: 论文针对MLC-SLM 2025挑战中的多语言语音识别（Task I）和语音说话人分割识别（Task II）任务，提出TEA-ASLP系统。通过改进Ideal-LLM模型，引入语言识别、多语言LoRA结构和CTC提示，提升了识别效果；在Task II中改用更适合的英文模型。最终在两项任务中分别取得第一和第二名。**

- **链接: [http://arxiv.org/pdf/2507.18051v1](http://arxiv.org/pdf/2507.18051v1)**

> **作者:** Hongfei Xue; Kaixun Huang; Zhikai Zhou; Shen Huang; Shidong Shang
>
> **备注:** Interspeech 2025 workshop
>
> **摘要:** This paper presents the TEA-ASLP's system submitted to the MLC-SLM 2025 Challenge, addressing multilingual conversational automatic speech recognition (ASR) in Task I and speech diarization ASR in Task II. For Task I, we enhance Ideal-LLM model by integrating known language identification and a multilingual MOE LoRA structure, along with using CTC-predicted tokens as prompts to improve autoregressive generation. The model is trained on approximately 180k hours of multilingual ASR data. In Task II, we replace the baseline English-Chinese speaker diarization model with a more suitable English-only version. Our approach achieves a 30.8% reduction in word error rate (WER) compared to the baseline speech language model, resulting in a final WER of 9.60% in Task I and a time-constrained minimum-permutation WER of 17.49% in Task II, earning first and second place in the respective challenge tasks.
>
---
#### [new 006] A Concept-based approach to Voice Disorder Detection
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于医疗AI任务，旨在解决语音障碍自动诊断中深度学习模型可解释性不足的问题。作者采用基于概念的模型（如CBM和CEM），在保持诊断性能的同时提升模型透明度，以增强临床可信度。**

- **链接: [http://arxiv.org/pdf/2507.17799v1](http://arxiv.org/pdf/2507.17799v1)**

> **作者:** Davide Ghia; Gabriele Ciravegna; Alkis Koudounas; Marco Fantini; Erika Crosetti; Giovanni Succo; Tania Cerquitelli
>
> **摘要:** Voice disorders affect a significant portion of the population, and the ability to diagnose them using automated, non-invasive techniques would represent a substantial advancement in healthcare, improving the quality of life of patients. Recent studies have demonstrated that artificial intelligence models, particularly Deep Neural Networks (DNNs), can effectively address this task. However, due to their complexity, the decision-making process of such models often remain opaque, limiting their trustworthiness in clinical contexts. This paper investigates an alternative approach based on Explainable AI (XAI), a field that aims to improve the interpretability of DNNs by providing different forms of explanations. Specifically, this works focuses on concept-based models such as Concept Bottleneck Model (CBM) and Concept Embedding Model (CEM) and how they can achieve performance comparable to traditional deep learning methods, while offering a more transparent and interpretable decision framework.
>
---
#### [new 007] Improving Bird Classification with Primary Color Additives
- **分类: cs.CV; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于鸟类分类任务，旨在解决因环境噪声、重叠鸣叫和标签缺失导致的分类困难。作者通过将频率信息转化为颜色增强频谱图，提升深度学习模型对鸟类物种的区分能力，显著提高了分类准确率。**

- **链接: [http://arxiv.org/pdf/2507.18334v1](http://arxiv.org/pdf/2507.18334v1)**

> **作者:** Ezhini Rasendiran R; Chandresh Kumar Maurya
>
> **备注:** 5 pages (Accepted to Interspeech 2025)
>
> **摘要:** We address the problem of classifying bird species using their song recordings, a challenging task due to environmental noise, overlapping vocalizations, and missing labels. Existing models struggle with low-SNR or multi-species recordings. We hypothesize that birds can be classified by visualizing their pitch pattern, speed, and repetition, collectively called motifs. Deep learning models applied to spectrogram images help, but similar motifs across species cause confusion. To mitigate this, we embed frequency information into spectrograms using primary color additives. This enhances species distinction and improves classification accuracy. Our experiments show that the proposed approach achieves statistically significant gains over models without colorization and surpasses the BirdCLEF 2024 winner, improving F1 by 7.3%, ROC-AUC by 6.2%, and CMAP by 6.6%. These results demonstrate the effectiveness of incorporating frequency information via colorization.
>
---
#### [new 008] Speech Enhancement with Dual-path Multi-Channel Linear Prediction Filter and Multi-norm Beamforming
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决高混响环境下语音信号的干扰问题。通过设计双路径多通道线性预测滤波器和多范数波束成形方法，提升语音质量。同时提出预测阶数选择策略，增强了方法的适用性。**

- **链接: [http://arxiv.org/pdf/2507.18350v1](http://arxiv.org/pdf/2507.18350v1)**

> **作者:** Chengyuan Qin; Wenmeng Xiong; Jing Zhou; Maoshen Jia; Changchun Bao
>
> **备注:** Paper accepted by Interspeech 2025
>
> **摘要:** In this paper, we propose a speech enhancement method us ing dual-path Multi-Channel Linear Prediction (MCLP) filters and multi-norm beamforming. Specifically, the MCLP part in the proposed method is designed with dual-path filters in both time and frequency dimensions. For the beamforming part, we minimize the power of the microphone array output as well as the l1 norm of the denoised signals while preserving source sig nals from the target directions. An efficient method to select the prediction orders in the dual-path filters is also proposed, which is robust for signals with different reverberation time (T60) val ues and can be applied to other MCLP-based methods. Eval uations demonstrate that our proposed method outperforms the baseline methods for speech enhancement, particularly in high reverberation scenarios.
>
---
#### [new 009] Modular Robot and Landmark Localisation Using Relative Bearing Measurements
- **分类: cs.RO; cs.SY; eess.SP; eess.SY**

- **简介: 论文研究模块化非线性最小二乘滤波方法，用于由多个独立子系统组成的系统状态估计，特别应用于机器人与路标定位问题。通过结合协方差交集（CI）算法避免信息重复计算，提升估计精度。论文通过仿真对比验证了该方法在通信受限下的性能表现。**

- **链接: [http://arxiv.org/pdf/2507.18070v1](http://arxiv.org/pdf/2507.18070v1)**

> **作者:** Behzad Zamani; Jochen Trumpf; Chris Manzie
>
> **备注:** Submitted to RA-L
>
> **摘要:** In this paper we propose a modular nonlinear least squares filtering approach for systems composed of independent subsystems. The state and error covariance estimate of each subsystem is updated independently, even when a relative measurement simultaneously depends on the states of multiple subsystems. We integrate the Covariance Intersection (CI) algorithm as part of our solution in order to prevent double counting of information when subsystems share estimates with each other. An alternative derivation of the CI algorithm based on least squares estimation makes this integration possible. We particularise the proposed approach to the robot-landmark localization problem. In this problem, noisy measurements of the bearing angle to a stationary landmark position measured relative to the SE(2) pose of a moving robot couple the estimation problems for the robot pose and the landmark position. In a randomized simulation study, we benchmark the proposed modular method against a monolithic joint state filter to elucidate their respective trade-offs. In this study we also include variants of the proposed method that achieve a graceful degradation of performance with reduced communication and bandwidth requirements.
>
---
#### [new 010] SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决大语言模型（LLM）在自动语音识别（ASR）中解码延迟高的问题。论文提出了SpecASR框架，通过自适应草稿生成、草稿复用和稀疏令牌树生成，提升解码效率，实现显著加速，且不损失识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.18181v1](http://arxiv.org/pdf/2507.18181v1)**

> **作者:** Linye Wei; Shuzhang Zhong; Songqiang Xu; Runsheng Wang; Ru Huang; Meng Li
>
> **摘要:** Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
>
---
#### [new 011] TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在解决现有中文语音语言模型评估基准不贴合真实对话场景的问题。作者提出了TELEVAL动态评估基准，从语义、副语言与系统能力三方面评估模型在真实交互中的表现，强调对用户隐含意图的理解与响应能力。**

- **链接: [http://arxiv.org/pdf/2507.18061v1](http://arxiv.org/pdf/2507.18061v1)**

> **作者:** Zehan Li; Hongjie Chen; Yuxin Zhang; Jing Zhou; Xuening Wang; Hang Lv; Mengjie Du; Yaodong Song; Jie Lian; Jian Kang; Jie Li; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **摘要:** Spoken language models (SLMs) have seen rapid progress in recent years, along with the development of numerous benchmarks for evaluating their performance. However, most existing benchmarks primarily focus on evaluating whether SLMs can perform complex tasks comparable to those tackled by large language models (LLMs), often failing to align with how users naturally interact in real-world conversational scenarios. In this paper, we propose TELEVAL, a dynamic benchmark specifically designed to evaluate SLMs' effectiveness as conversational agents in realistic Chinese interactive settings. TELEVAL defines three evaluation dimensions: Explicit Semantics, Paralinguistic and Implicit Semantics, and System Abilities. It adopts a dialogue format consistent with real-world usage and evaluates text and audio outputs separately. TELEVAL particularly focuses on the model's ability to extract implicit cues from user speech and respond appropriately without additional instructions. Our experiments demonstrate that despite recent progress, existing SLMs still have considerable room for improvement in natural conversational tasks. We hope that TELEVAL can serve as a user-centered evaluation framework that directly reflects the user experience and contributes to the development of more capable dialogue-oriented SLMs.
>
---
#### [new 012] Tiny is not small enough: High-quality, low-resource facial animation models through hybrid knowledge distillation
- **分类: cs.GR; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文研究语音驱动的3D面部动画模型，旨在解决高质量模型因体积大而难以在设备上实时运行的问题。通过混合知识蒸馏与伪标签技术，训练小型学生模型，在减少内存占用和延迟的同时保持动画质量，推动了设备端数字角色的实现。**

- **链接: [http://arxiv.org/pdf/2507.18352v1](http://arxiv.org/pdf/2507.18352v1)**

> **作者:** Zhen Han; Mattias Teye; Derek Yadgaroff; Judith Bütepage
>
> **备注:** Accepted to ACM Transactions on Graphics 2025 (SIGGRAPH journal track)
>
> **摘要:** The training of high-quality, robust machine learning models for speech-driven 3D facial animation requires a large, diverse dataset of high-quality audio-animation pairs. To overcome the lack of such a dataset, recent work has introduced large pre-trained speech encoders that are robust to variations in the input audio and, therefore, enable the facial animation model to generalize across speakers, audio quality, and languages. However, the resulting facial animation models are prohibitively large and lend themselves only to offline inference on a dedicated machine. In this work, we explore on-device, real-time facial animation models in the context of game development. We overcome the lack of large datasets by using hybrid knowledge distillation with pseudo-labeling. Given a large audio dataset, we employ a high-performing teacher model to train very small student models. In contrast to the pre-trained speech encoders, our student models only consist of convolutional and fully-connected layers, removing the need for attention context or recurrent updates. In our experiments, we demonstrate that we can reduce the memory footprint to up to 3.4 MB and required future audio context to up to 81 ms while maintaining high-quality animations. This paves the way for on-device inference, an important step towards realistic, model-driven digital characters.
>
---
#### [new 013] Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决实时多说话人追踪问题。为实现说话人输出顺序与到达时间一致，论文提出Streaming Sortformer方法，引入按到达时间排序的说话人缓存机制（AOSC），动态存储并更新帧级声学嵌入，提升低延迟场景下的追踪精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.18446v1](http://arxiv.org/pdf/2507.18446v1)**

> **作者:** Ivan Medennikov; Taejin Park; Weiqing Wang; He Huang; Kunal Dhawan; Jinhan Wang; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This paper presents a streaming extension for the Sortformer speaker diarization framework, whose key property is the arrival-time ordering of output speakers. The proposed approach employs an Arrival-Order Speaker Cache (AOSC) to store frame-level acoustic embeddings of previously observed speakers. Unlike conventional speaker-tracing buffers, AOSC orders embeddings by speaker index corresponding to their arrival time order, and is dynamically updated by selecting frames with the highest scores based on the model's past predictions. Notably, the number of stored embeddings per speaker is determined dynamically by the update mechanism, ensuring efficient cache utilization and precise speaker tracking. Experiments on benchmark datasets confirm the effectiveness and flexibility of our approach, even in low-latency setups. These results establish Streaming Sortformer as a robust solution for real-time multi-speaker tracking and a foundation for streaming multi-talker speech processing.
>
---
#### [new 014] GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音语言建模任务，旨在解决现有模型忽略语音中韵律、情感、年龄等非语言信息的问题。作者提出GOAT-SLM，采用双模态结构分离语言与声学信息，通过分阶段训练提升模型对情感、方言和年龄等特征的建模能力，从而实现更自然、更具社会意识的语音交互。**

- **链接: [http://arxiv.org/pdf/2507.18119v1](http://arxiv.org/pdf/2507.18119v1)**

> **作者:** Hongjie Chen; Zehan Li; Yaodong Song; Wenming Deng; Yitong Yao; Yuxin Zhang; Hang Lv; Xuechao Zhu; Jian Kang; Jie Lian; Jie Li; Chao Wang; Shuangyong Song; Yongxiang Li; Zhongjiang He
>
> **摘要:** Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems.
>
---
#### [new 015] Recent Trends in Distant Conversational Speech Recognition: A Review of CHiME-7 and 8 DASR Challenges
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升远场对话语音识别的准确性和鲁棒性。论文总结了CHiME-7和8挑战赛的设计、数据集及评估指标，分析了参赛系统的趋势与问题，如端到端模型的普及、神经语音增强技术的局限性、说话人日志优化的重要性及会议摘要对识别质量的弱相关性。**

- **链接: [http://arxiv.org/pdf/2507.18161v1](http://arxiv.org/pdf/2507.18161v1)**

> **作者:** Samuele Cornell; Christoph Boeddeker; Taejin Park; He Huang; Desh Raj; Matthew Wiesner; Yoshiki Masuyama; Xuankai Chang; Zhong-Qiu Wang; Stefano Squartini; Paola Garcia; Shinji Watanabe
>
> **摘要:** The CHiME-7 and 8 distant speech recognition (DASR) challenges focus on multi-channel, generalizable, joint automatic speech recognition (ASR) and diarization of conversational speech. With participation from 9 teams submitting 32 diverse systems, these challenges have contributed to state-of-the-art research in the field. This paper outlines the challenges' design, evaluation metrics, datasets, and baseline systems while analyzing key trends from participant submissions. From this analysis it emerges that: 1) Most participants use end-to-end (e2e) ASR systems, whereas hybrid systems were prevalent in previous CHiME challenges. This transition is mainly due to the availability of robust large-scale pre-trained models, which lowers the data burden for e2e-ASR. 2) Despite recent advances in neural speech separation and enhancement (SSE), all teams still heavily rely on guided source separation, suggesting that current neural SSE techniques are still unable to reliably deal with complex scenarios and different recording setups. 3) All best systems employ diarization refinement via target-speaker diarization techniques. Accurate speaker counting in the first diarization pass is thus crucial to avoid compounding errors and CHiME-8 DASR participants especially focused on this part. 4) Downstream evaluation via meeting summarization can correlate weakly with transcription quality due to the remarkable effectiveness of large-language models in handling errors. On the NOTSOFAR-1 scenario, even systems with over 50\% time-constrained minimum permutation WER can perform roughly on par with the most effective ones (around 11\%). 5) Despite recent progress, accurately transcribing spontaneous speech in challenging acoustic environments remains difficult, even when using computationally intensive system ensembles.
>
---
#### [new 016] A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation
- **分类: cs.CV; cs.AI; cs.LG; eess.SP**

- **简介: 该论文属于医学信号处理与深度学习交叉任务，旨在解决心电图（ECG）波形自动分割中标注数据不足的问题。作者构建了首个用于半监督语义分割的ECG基准数据集，整合多个公开数据源，采用五种半监督算法，在卷积网络与Transformer架构上进行实验，并提出ECG专用训练策略与评估框架。结果表明Transformer表现更优，为后续研究提供了标准参考。**

- **链接: [http://arxiv.org/pdf/2507.18323v1](http://arxiv.org/pdf/2507.18323v1)**

> **作者:** Minje Park; Jeonghwa Lim; Taehyung Yu; Sunghoon Joo
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Electrocardiogram (ECG) delineation, the segmentation of meaningful waveform features, is critical for clinical diagnosis. Despite recent advances using deep learning, progress has been limited by the scarcity of publicly available annotated datasets. Semi-supervised learning presents a promising solution by leveraging abundant unlabeled ECG data. In this study, we present the first systematic benchmark for semi-supervised semantic segmentation (SemiSeg) in ECG delineation. We curated and unified multiple public datasets, including previously underused sources, to support robust and diverse evaluation. We adopted five representative SemiSeg algorithms from computer vision, implemented them on two different architectures: the convolutional network and the transformer, and evaluated them in two different settings: in-domain and cross-domain. Additionally, we propose ECG-specific training configurations and augmentation strategies and introduce a standardized evaluation framework. Our results show that the transformer outperforms the convolutional network in semi-supervised ECG delineation. We anticipate that our benchmark will serve as a foundation for advancing semi-supervised ECG delineation methods and will facilitate further research in this domain.
>
---
## 更新

#### [replaced 001] Information and motor constraints shape melodic diversity across cultures
- **分类: cs.SD; cs.IT; eess.AS; math.IT; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2408.12635v4](http://arxiv.org/pdf/2408.12635v4)**

> **作者:** John M McBride; Nahie Kim; Yuri Nishikawa; Mekhmed Saadakeev; Marcus T Pearce; Tsvi Tlusty
>
> **摘要:** The number of possible melodies is unfathomably large, yet despite this virtually unlimited potential for melodic variation, melodies from different societies can be surprisingly similar. The motor constraint hypothesis accounts for certain similarities, such as scalar motion and contour shape, but not for other major common features, such as repetition, song length, and scale size. Here we investigate the role of information constraints in shaping these hallmarks of melodies. We measure determinants of information rate in 62 corpora of Folk melodies spanning several continents, finding multiple trade-offs that all act to constrain the information rate across societies. By contrast, 39 corpora of Art music from Europe (including Turkey) show longer, more complex melodies, and increased complexity over time, suggesting different cultural-evolutionary selection pressures in Art and Folk music, possibly due to the use of written versus oral transmission. Our parameter-free model predicts the empirical scale degree distribution using information constraints on scalar motion, melody length, and, most importantly, information rate. These results provide strong evidence that information constraints during cultural transmission of music limit the number of notes in a scale, and suggests that a tendency for intermediate melodic complexity reflects a fundamental constraint on the cultural evolution of melody.
>
---
#### [replaced 002] Step-Audio 2 Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16632v2](http://arxiv.org/pdf/2507.16632v2)**

> **作者:** Boyong Wu; Chao Yan; Chen Hu; Cheng Yi; Chengli Feng; Fei Tian; Feiyu Shen; Gang Yu; Haoyang Zhang; Jingbei Li; Mingrui Chen; Peng Liu; Wang You; Xiangyu Tony Zhang; Xingyuan Li; Xuerui Yang; Yayue Deng; Yechang Huang; Yuxin Li; Yuxin Zhang; Zhao You; Brian Li; Changyi Wan; Hanpeng Hu; Jiangjie Zhen; Siyu Chen; Song Yuan; Xuelin Zhang; Yimin Jiang; Yu Zhou; Yuxiang Yang; Bingxin Li; Buyun Ma; Changhe Song; Dongqing Pang; Guoqiang Hu; Haiyang Sun; Kang An; Na Wang; Shuli Gao; Wei Ji; Wen Li; Wen Sun; Xuan Wen; Yong Ren; Yuankai Ma; Yufan Lu; Bin Wang; Bo Li; Changxin Miao; Che Liu; Chen Xu; Dapeng Shi; Dingyuan Hu; Donghang Wu; Enle Liu; Guanzhe Huang; Gulin Yan; Han Zhang; Hao Nie; Haonan Jia; Hongyu Zhou; Jianjian Sun; Jiaoren Wu; Jie Wu; Jie Yang; Jin Yang; Junzhe Lin; Kaixiang Li; Lei Yang; Liying Shi; Li Zhou; Longlong Gu; Ming Li; Mingliang Li; Mingxiao Li; Nan Wu; Qi Han; Qinyuan Tan; Shaoliang Pang; Shengjie Fan; Siqi Liu; Tiancheng Cao; Wanying Lu; Wenqing He; Wuxun Xie; Xu Zhao; Xueqi Li; Yanbo Yu; Yang Yang; Yi Liu; Yifan Lu; Yilei Wang; Yuanhao Ding; Yuanwei Liang; Yuanwei Lu; Yuchu Luo; Yuhe Yin; Yumeng Zhan; Yuxiang Zhang; Zidong Yang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** This paper presents Step-Audio 2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.
>
---
#### [replaced 003] Benchmarking Cross-Domain Audio-Visual Deception Detection
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.06995v3](http://arxiv.org/pdf/2405.06995v3)**

> **作者:** Xiaobao Guo; Zitong Yu; Nithish Muthuchamy Selvaraj; Bingquan Shen; Adams Wai-Kin Kong; Alex C. Kot
>
> **备注:** 15 pages
>
> **摘要:** Automated deception detection is crucial for assisting humans in accurately assessing truthfulness and identifying deceptive behavior. Conventional contact-based techniques, like polygraph devices, rely on physiological signals to determine the authenticity of an individual's statements. Nevertheless, recent developments in automated deception detection have demonstrated that multimodal features derived from both audio and video modalities may outperform human observers on publicly available datasets. Despite these positive findings, the generalizability of existing audio-visual deception detection approaches across different scenarios remains largely unexplored. To close this gap, we present the first cross-domain audio-visual deception detection benchmark, that enables us to assess how well these methods generalize for use in real-world scenarios. We used widely adopted audio and visual features and different architectures for benchmarking, comparing single-to-single and multi-to-single domain generalization performance. To further exploit the impacts using data from multiple source domains for training, we investigate three types of domain sampling strategies, including domain-simultaneous, domain-alternating, and domain-by-domain for multi-to-single domain generalization evaluation. We also propose an algorithm to enhance the generalization performance by maximizing the gradient inner products between modality encoders, named ``MM-IDGM". Furthermore, we proposed the Attention-Mixer fusion method to improve performance, and we believe that this new cross-domain benchmark will facilitate future research in audio-visual deception detection.
>
---
#### [replaced 004] DiffRhythm+: Controllable and Flexible Full-Length Song Generation with Preference Optimization
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.12890v2](http://arxiv.org/pdf/2507.12890v2)**

> **作者:** Huakang Chen; Yuepeng Jiang; Guobin Ma; Chunbo Hao; Shuai Wang; Jixun Yao; Ziqian Ning; Meng Meng; Jian Luan; Lei Xie
>
> **摘要:** Songs, as a central form of musical art, exemplify the richness of human intelligence and creativity. While recent advances in generative modeling have enabled notable progress in long-form song generation, current systems for full-length song synthesis still face major challenges, including data imbalance, insufficient controllability, and inconsistent musical quality. DiffRhythm, a pioneering diffusion-based model, advanced the field by generating full-length songs with expressive vocals and accompaniment. However, its performance was constrained by an unbalanced model training dataset and limited controllability over musical style, resulting in noticeable quality disparities and restricted creative flexibility. To address these limitations, we propose DiffRhythm+, an enhanced diffusion-based framework for controllable and flexible full-length song generation. DiffRhythm+ leverages a substantially expanded and balanced training dataset to mitigate issues such as repetition and omission of lyrics, while also fostering the emergence of richer musical skills and expressiveness. The framework introduces a multi-modal style conditioning strategy, enabling users to precisely specify musical styles through both descriptive text and reference audio, thereby significantly enhancing creative control and diversity. We further introduce direct performance optimization aligned with user preferences, guiding the model toward consistently preferred outputs across evaluation metrics. Extensive experiments demonstrate that DiffRhythm+ achieves significant improvements in naturalness, arrangement complexity, and listener satisfaction over previous systems.
>
---
#### [replaced 005] LENS-DF: Deepfake Detection and Temporal Localization for Long-Form Noisy Speech
- **分类: cs.SD; cs.CR; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16220v2](http://arxiv.org/pdf/2507.16220v2)**

> **作者:** Xuechen Liu; Wanying Ge; Xin Wang; Junichi Yamagishi
>
> **备注:** Accepted by IEEE International Joint Conference on Biometrics (IJCB) 2025, Osaka, Japan
>
> **摘要:** This study introduces LENS-DF, a novel and comprehensive recipe for training and evaluating audio deepfake detection and temporal localization under complicated and realistic audio conditions. The generation part of the recipe outputs audios from the input dataset with several critical characteristics, such as longer duration, noisy conditions, and containing multiple speakers, in a controllable fashion. The corresponding detection and localization protocol uses models. We conduct experiments based on self-supervised learning front-end and simple back-end. The results indicate that models trained using data generated with LENS-DF consistently outperform those trained via conventional recipes, demonstrating the effectiveness and usefulness of LENS-DF for robust audio deepfake detection and localization. We also conduct ablation studies on the variations introduced, investigating their impact on and relevance to realistic challenges in the field.
>
---
#### [replaced 006] Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17527v2](http://arxiv.org/pdf/2507.17527v2)**

> **作者:** Shanbo Cheng; Yu Bao; Zhichao Huang; Yu Lu; Ningxin Peng; Lu Xu; Runsheng Yu; Rong Cao; Ting Han; Zeyang Li; Sitong Liu; Shengtao Ma; Shiguang Pan; Jiongchen Xiao; Nuo Xu; Meng Yang; Rong Ye; Yiming Yu; Ruofei Zhang; Wanyi Zhang; Wenhao Zhu; Liehao Zou; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** Seed-LiveInterpret 2.0 Technical Report
>
> **摘要:** Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
>
---
#### [replaced 007] RadioDUN: A Physics-Inspired Deep Unfolding Network for Radio Map Estimation
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.08418v2](http://arxiv.org/pdf/2506.08418v2)**

> **作者:** Taiqin Chen; Zikun Zhou; Zheng Fang; Wenzhen Zou; Kangjun Liu; Ke Chen; Yongbing Zhang; Yaowei Wang
>
> **摘要:** The radio map represents the spatial distribution of spectrum resources within a region, supporting efficient resource allocation and interference mitigation. However, it is difficult to construct a dense radio map as a limited number of samples can be measured in practical scenarios. While existing works have used deep learning to estimate dense radio maps from sparse samples, they are hard to integrate with the physical characteristics of the radio map. To address this challenge, we cast radio map estimation as the sparse signal recovery problem. A physical propagation model is further incorporated to decompose the problem into multiple factor optimization sub-problems, thereby reducing recovery complexity. Inspired by the existing compressive sensing methods, we propose the Radio Deep Unfolding Network (RadioDUN) to unfold the optimization process, achieving adaptive parameter adjusting and prior fitting in a learnable manner. To account for the radio propagation characteristics, we develop a dynamic reweighting module (DRM) to adaptively model the importance of each factor for the radio map. Inspired by the shadowing factor in the physical propagation model, we integrate obstacle-related factors to express the obstacle-induced signal stochastic decay. The shadowing loss is further designed to constrain the factor prediction and act as a supplementary supervised objective, which enhances the performance of RadioDUN. Extensive experiments have been conducted to demonstrate that the proposed method outperforms the state-of-the-art methods. Our code will be made publicly available upon publication.
>
---
