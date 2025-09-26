# 音频 cs.SD;  eess.SP

- **最新发布 19 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Thinking While Listening: Simple Test Time Scaling For Audio Classification
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文针对音频分类任务，旨在提升模型在类别空间中的推理能力。提出了一种“边听边思考”的框架，并探索了如何通过测试时扩展和轻量级微调（如仅调整嵌入矩阵）来提高分类性能。**

- **链接: [http://arxiv.org/pdf/2509.19676v1](http://arxiv.org/pdf/2509.19676v1)**

> **作者:** Prateek Verma; Mert Pilanci
>
> **备注:** 6 pages, 3 figures, 2 Tables, ICASSP 2026
>
> **摘要:** We propose a framework that enables neural models to "think while listening" to everyday sounds, thereby enhancing audio classification performance. Motivated by recent advances in the reasoning capabilities of large language models, we address two central questions: (i) how can thinking be incorporated into existing audio classification pipelines to enable reasoning in the category space and improve performance, and (ii) can a new architecture be designed from the ground up to support both thinking and test-time scaling? We demonstrate that in both settings, our models exhibit improved classification accuracy. Leveraging test-time scaling, we observe consistent gains as the number of sampled traces increases. Furthermore, we evaluate two open-source reasoning models, GPT-OSS-20B and Qwen3-14B, showing that while such models are capable of zero-shot reasoning, a lightweight approach--retraining only the embedding matrix of a frozen, smaller model like GPT-2--can surpass the performance of billion-parameter text-based reasoning models.
>
---
#### [new 002] Can Audio Large Language Models Verify Speaker Identity?
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究将音频大语言模型（ALLMs）用于说话人验证（SV）任务。针对ALLMs在零样本条件下性能有限的问题，提出监督微调和硬样本采样策略，并扩展到文本依赖的SV，展示了ALLMs在统一音频理解中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.19755v1](http://arxiv.org/pdf/2509.19755v1)**

> **作者:** Yiming Ren; Xuenan Xu; Baoxiang Li; Shuai Wang; Chao Zhang
>
> **摘要:** This paper investigates adapting Audio Large Language Models (ALLMs) for speaker verification (SV). We reformulate SV as an audio question-answering task and conduct comprehensive zero-shot evaluations on public benchmarks, showing that current ALLMs have limited zero-shot SV capability and often struggle in diverse acoustic conditions. To address this challenge, we perform supervised fine-tuning on speaker verification data. A rule-based hard pair sampling strategy is proposed to construct more challenging training pairs. Lightweight fine-tuning substantially improves the performance, though there is still a gap between ALLMs and conventional models. Then, we extend to text-dependent SV by jointly querying ALLMs to verify speaker identity and spoken content, yielding results competitive with cascaded ASR-SV systems. Our findings demonstrate that with proper adaptation, ALLMs hold substantial potential as a unified model for robust speaker verification systems, while maintaining the general audio understanding capabilities.
>
---
#### [new 003] Enabling Multi-Species Bird Classification on Low-Power Bioacoustic Loggers
- **分类: cs.SD; cs.CE**

- **简介: 该论文提出WrenNet，一种高效的神经网络，用于在低功耗设备上实现实时多物种鸟类音频分类。针对生物声学监测任务，设计了半可学习的频谱特征提取器，提升了分类性能与能效，部署于AudioMoth设备，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20103v1](http://arxiv.org/pdf/2509.20103v1)**

> **作者:** Stefano Ciapponi; Leonardo Mannini; Jarek Scanferla; Matteo Anderle; Elisabetta Farella
>
> **摘要:** This paper introduces WrenNet, an efficient neural network enabling real-time multi-species bird audio classification on low-power microcontrollers for scalable biodiversity monitoring. We propose a semi-learnable spectral feature extractor that adapts to avian vocalizations, outperforming standard mel-scale and fully-learnable alternatives. On an expert-curated 70-species dataset, WrenNet achieves up to 90.8\% accuracy on acoustically distinctive species and 70.1\% on the full task. When deployed on an AudioMoth device ($\leq$1MB RAM), it consumes only 77mJ per inference. Moreover, the proposed model is over 16x more energy-efficient compared to Birdnet when running on a Raspberry Pi 3B+. This work demonstrates the first practical framework for continuous, multi-species acoustic monitoring on low-power edge devices.
>
---
#### [new 004] MusiCRS: Benchmarking Audio-Centric Conversational Recommendation
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出MusiCRS，首个面向音频的对话推荐基准，解决音乐推荐中音频内容理解不足的问题。通过链接Reddit对话与YouTube音频，支持多模态评估，揭示现有模型在音频推理上的局限性，并提供数据集与基线促进研究。**

- **链接: [http://arxiv.org/pdf/2509.19469v1](http://arxiv.org/pdf/2509.19469v1)**

> **作者:** Rohan Surana; Amit Namburi; Gagan Mundada; Abhay Lal; Zachary Novack; Julian McAuley; Junda Wu
>
> **备注:** 6 pages
>
> **摘要:** Conversational recommendation has advanced rapidly with large language models (LLMs), yet music remains a uniquely challenging domain where effective recommendations require reasoning over audio content beyond what text or metadata can capture. We present MusiCRS, the first benchmark for audio-centric conversational recommendation that links authentic user conversations from Reddit with corresponding audio tracks. MusiCRS contains 477 high-quality conversations spanning diverse genres (classical, hip-hop, electronic, metal, pop, indie, jazz) with 3,589 unique musical entities and audio grounding via YouTube links. MusiCRS enables evaluation across three input modality configurations: audio-only, query-only, and audio+query (multimodal), allowing systematic comparison of audio-LLMs, retrieval models, and traditional approaches. Our experiments reveal that current systems rely heavily on textual signals and struggle with nuanced audio reasoning. This exposes fundamental limitations in cross-modal knowledge integration where models excel at dialogue semantics but cannot effectively ground abstract musical concepts in actual audio content. To facilitate progress, we release the MusiCRS dataset (https://huggingface.co/datasets/rohan2810/MusiCRS), evaluation code (https://github.com/rohan2810/musiCRS), and comprehensive baselines.
>
---
#### [new 005] CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出CoMelSinger，一种基于离散编码器的零样本歌唱合成框架，旨在解决旋律控制与音色分离问题。通过结构化歌词和音高标记建模，并采用对比学习策略抑制韵律泄露，提升了音高准确性、音色一致性及泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.19883v1](http://arxiv.org/pdf/2509.19883v1)**

> **作者:** Junchuan Zhao; Wei Zeng; Tianle Lyu; Ye Wang
>
> **备注:** 13 pages, 5 figures, 5 tables
>
> **摘要:** Singing Voice Synthesis (SVS) aims to generate expressive vocal performances from structured musical inputs such as lyrics and pitch sequences. While recent progress in discrete codec-based speech synthesis has enabled zero-shot generation via in-context learning, directly extending these techniques to SVS remains non-trivial due to the requirement for precise melody control. In particular, prompt-based generation often introduces prosody leakage, where pitch information is inadvertently entangled within the timbre prompt, compromising controllability. We present CoMelSinger, a zero-shot SVS framework that enables structured and disentangled melody control within a discrete codec modeling paradigm. Built on the non-autoregressive MaskGCT architecture, CoMelSinger replaces conventional text inputs with lyric and pitch tokens, preserving in-context generalization while enhancing melody conditioning. To suppress prosody leakage, we propose a coarse-to-fine contrastive learning strategy that explicitly regularizes pitch redundancy between the acoustic prompt and melody input. Furthermore, we incorporate a lightweight encoder-only Singing Voice Transcription (SVT) module to align acoustic tokens with pitch and duration, offering fine-grained frame-level supervision. Experimental results demonstrate that CoMelSinger achieves notable improvements in pitch accuracy, timbre consistency, and zero-shot transferability over competitive baselines.
>
---
#### [new 006] Eliminating stability hallucinations in llm-based tts models via attention guidance
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对LLM-based TTS模型中的稳定性幻觉问题（如重复或遗漏语音），通过改进注意力机制，提出了一种基于Viterbi算法的对齐质量评估指标OAS，并结合预训练注意力值进行链式推理训练，有效减少了合成语音中的异常。**

- **链接: [http://arxiv.org/pdf/2509.19852v1](http://arxiv.org/pdf/2509.19852v1)**

> **作者:** ShiMing Wang; ZhiHao Du; Yang Xiang; TianYu Zhao; Han Zhao; Qian Chen; XianGang Li; HanJie Guo; ZhenHua Ling
>
> **备注:** 5 pages, submitted to ICASSP2026
>
> **摘要:** This paper focuses on resolving stability hallucinations (e.g., repetitive or omitted speech) in LLM-based Text-to-Speech (TTS) models by improving and leveraging the attention mechanism. First, we analyzed the alignment mechanism between text tokens and speech tokens in LLMs. We then proposed a metric termed the Optimal Alignment Score (OAS), which employs the Viterbi algorithm to evaluate text-speech alignment quality. Subsequently, OAS was integrated into the training of CosyVoice2 to assist LLMs in learning continuous, stable alignment. Additionally, the pre-trained attention value is employed to guide the training of the student CosyVoice2 via chain-of-thought (CoT), which further reduces stability hallucinations in synthesized speech. Experiments on the Seed-TTS-Eval and CV3-Eval test sets demonstrate that the proposed methods can effectively reduce the stability hallucinations of CosyVoice2 without introducing additional negative effects. The appendix is available at https://wsmzzz.github.io/llm_attn.
>
---
#### [new 007] Scensory: Automated Real-Time Fungal Identification and Spatial Mapping
- **分类: eess.SP; cs.RO**

- **简介: 该论文提出Scensory，一种基于机器人和VOC传感器的实时真菌识别与定位系统。针对传统方法耗时、昂贵且缺乏空间分辨率的问题，利用深度学习分析VOC动态，实现快速、低成本的环境监测与源追踪。**

- **链接: [http://arxiv.org/pdf/2509.19318v1](http://arxiv.org/pdf/2509.19318v1)**

> **作者:** Yanbaihui Liu; Erica Babusci; Claudia K. Gunsch; Boyuan Chen
>
> **备注:** Our project website is at: http://generalroboticslab.com/Scensory
>
> **摘要:** Indoor fungal contamination poses significant risks to public health, yet existing detection methods are slow, costly, and lack spatial resolution. Conventional approaches rely on laboratory analysis or high-concentration sampling, making them unsuitable for real-time monitoring and scalable deployment. We introduce \textbf{\textit{Scensory}}, a robot-enabled olfactory system that simultaneously identifies fungal species and localizes their spatial origin using affordable volatile organic compound (VOC) sensor arrays and deep learning. Our key idea is that temporal VOC dynamics encode both chemical and spatial signatures, which we decode through neural architectures trained on robot-automated data collection. We demonstrate two operational modes: a passive multi-array configuration for environmental monitoring, and a mobile single-array configuration for active source tracking. Across five fungal species, our system achieves up to 89.85\% accuracy in species detection and 87.31\% accuracy in localization under ambient conditions, where each prediction only takes 3--7\,s sensor inputs. Additionally, by computationally analyzing model behavior, we can uncover key biochemical signatures without additional laboratory experiments. Our approach enables real-time, spatially aware fungal monitoring and establishes a scalable and affordable framework for autonomous environmental sensing.
>
---
#### [new 008] Non-locally averaged pruned reassigned spectrograms: a tool for glottal pulse visualization and analysis
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 该论文提出了一种改进的重分配频谱图方法NAPReS，用于更清晰、可量化地可视化和分析声门脉冲特征。针对传统方法在数据可视化与抗噪性上的不足，结合GMM方法，提升了高噪声环境下共振峰估计的可重复性。**

- **链接: [http://arxiv.org/pdf/2509.19686v1](http://arxiv.org/pdf/2509.19686v1)**

> **作者:** Gabriel J. Griswold; Mark A. Griswold
>
> **备注:** Submitted to Speech Communications. 16 pages, 7 figs, 1 table
>
> **摘要:** Reassigned spectrograms have shown advantages in precise formant measuring and inter-speaker differentiation. However, reassigned spectrograms suffer from their inability to visualize larger amounts of data in an easily comprehensible and reproducible manner. Utilizing the techniques and tools developed by Fulop and Fitz, a variation of the reassigned spectrogram is proposed. Non-locally Averaged Pruned Reassigned Spectrograms (NAPReS) provide a simplified view into the characteristics of a speaker's glottal pulsation patterns throughout the centroid of a vowel through the stacking, summing, and pruning of large numbers of glottal pulses. In this exploratory study, NAPReS has been shown to display a large amount of data in an easily comprehensible and quantifiable manner, while also making the observation of low-amplitude cyclical structures more accessible. NAPReS also allows for alternative formant fitting methods such as Gaussian mixture modeling. In this study, NAPReS with GMM was compared against conventional LPC fitting of formant values and was shown to be more reproducible than conventional LPC fitting in high-noise situations.
>
---
#### [new 009] Efficient Speech Watermarking for Speech Synthesis via Progressive Knowledge Distillation
- **分类: cs.SD**

- **简介: 该论文提出PKDMark，一种基于渐进式知识蒸馏的轻量级语音水印方法，旨在解决语音合成中未经授权的语音克隆问题。通过两阶段训练，提升计算效率和鲁棒性，在保证不可感知性的同时，降低93.6%的计算成本。**

- **链接: [http://arxiv.org/pdf/2509.19812v1](http://arxiv.org/pdf/2509.19812v1)**

> **作者:** Yang Cui; Peter Pan; Lei He; Sheng Zhao
>
> **备注:** 6 pages of main text, 1 page of references, 2 figures, 2 tables, accepted at ASRU 2025
>
> **摘要:** With the rapid advancement of speech generative models, unauthorized voice cloning poses significant privacy and security risks. Speech watermarking offers a viable solution for tracing sources and preventing misuse. Current watermarking technologies fall mainly into two categories: DSP-based methods and deep learning-based methods. DSP-based methods are efficient but vulnerable to attacks, whereas deep learning-based methods offer robust protection at the expense of significantly higher computational cost. To improve the computational efficiency and enhance the robustness, we propose PKDMark, a lightweight deep learning-based speech watermarking method that leverages progressive knowledge distillation (PKD). Our approach proceeds in two stages: (1) training a high-performance teacher model using an invertible neural network-based architecture, and (2) transferring the teacher's capabilities to a compact student model through progressive knowledge distillation. This process reduces computational costs by 93.6% while maintaining high level of robust performance and imperceptibility. Experimental results demonstrate that our distilled model achieves an average detection F1 score of 99.6% with a PESQ of 4.30 in advanced distortions, enabling efficient speech watermarking for real-time speech synthesis applications.
>
---
#### [new 010] ArtiFree: Detecting and Reducing Generative Artifacts in Diffusion-based Speech Enhancement
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究基于扩散模型的语音增强任务，旨在解决生成伪影和高推理延迟问题。提出了利用语义一致性进行集成推理的方法，并分析了扩散步数的影响，有效提升了语音准确性和自然度。**

- **链接: [http://arxiv.org/pdf/2509.19495v1](http://arxiv.org/pdf/2509.19495v1)**

> **作者:** Bhawana Chhaglani; Yang Gao; Julius Richter; Xilin Li; Syavosh Zadissa; Tarun Pruthi; Andrew Lovitt
>
> **摘要:** Diffusion-based speech enhancement (SE) achieves natural-sounding speech and strong generalization, yet suffers from key limitations like generative artifacts and high inference latency. In this work, we systematically study artifact prediction and reduction in diffusion-based SE. We show that variance in speech embeddings can be used to predict phonetic errors during inference. Building on these findings, we propose an ensemble inference method guided by semantic consistency across multiple diffusion runs. This technique reduces WER by 15% in low-SNR conditions, effectively improving phonetic accuracy and semantic plausibility. Finally, we analyze the effect of the number of diffusion steps, showing that adaptive diffusion steps balance artifact suppression and latency. Our findings highlight semantic priors as a powerful tool to guide generative SE toward artifact-free outputs.
>
---
#### [new 011] SEA-Spoof: Bridging The Gap in Multilingual Audio Deepfake Detection for South-East Asian
- **分类: cs.SD**

- **简介: 该论文聚焦音频深度伪造检测任务，针对东南亚语言数据稀缺问题，构建了首个大规模SEA-Spoof数据集，涵盖6种语言共300+小时真实与伪造语音对，推动跨语言鲁棒检测模型研究。**

- **链接: [http://arxiv.org/pdf/2509.19865v1](http://arxiv.org/pdf/2509.19865v1)**

> **作者:** Jinyang Wu; Nana Hou; Zihan Pan; Qiquan Zhang; Sailor Hardik Bhupendra; Soumik Mondal
>
> **备注:** 5 pages, 1 figure, 3 tables
>
> **摘要:** The rapid growth of the digital economy in South-East Asia (SEA) has amplified the risks of audio deepfakes, yet current datasets cover SEA languages only sparsely, leaving models poorly equipped to handle this critical region. This omission is critical: detection models trained on high-resource languages collapse when applied to SEA, due to mismatches in synthesis quality, language-specific characteristics, and data scarcity. To close this gap, we present SEA-Spoof, the first large-scale Audio Deepfake Detection (ADD) dataset especially for SEA languages. SEA-Spoof spans 300+ hours of paired real and spoof speech across Tamil, Hindi, Thai, Indonesian, Malay, and Vietnamese. Spoof samples are generated from a diverse mix of state-of-the-art open-source and commercial systems, capturing wide variability in style and fidelity. Benchmarking state-of-the-art detection models reveals severe cross-lingual degradation, but fine-tuning on SEA-Spoof dramatically restores performance across languages and synthesis sources. These results highlight the urgent need for SEA-focused research and establish SEA-Spoof as a foundation for developing robust, cross-lingual, and fraud-resilient detection systems.
>
---
#### [new 012] On the Invariance of Cross-Correlation Peak Positions Under Monotonic Signal Transformations, with Application to Fast Time Difference Estimation
- **分类: eess.SP; eess.AS**

- **简介: 该论文研究了时间差估计问题，提出了一种基于信号交叉相关峰位置不变性的新方法。通过单调变换和低比特量化，使用整数运算实现快速估计，相比FFT方法计算效率更高。**

- **链接: [http://arxiv.org/pdf/2509.19974v1](http://arxiv.org/pdf/2509.19974v1)**

> **作者:** Natsuki Ueno; Ryotaro Sato; Nobutaka Ono
>
> **摘要:** We present a theorem concerning the invariance of cross-correlation peak positions, which provides a foundation for a new method for time difference estimation that is potentially faster than the conventional fast Fourier transform (FFT) approach for real/complex sequences. This theoretical result shows that the peak position of the cross-correlation function between two shifted discrete-time signals remains unchanged under arbitrary monotonic transformations of the input signals. By exploiting this property, we design an efficient estimation algorithm based on the cross-correlation function between signals quantized into low-bit integers. The proposed method requires only integer arithmetic instead of real-valued operations, and further computational efficiency can be achieved through number-theoretic algorithms. Numerical experiments demonstrate that the proposed method achieves a shorter processing time than conventional FFT-based approaches.
>
---
#### [new 013] InconVAD: A Two-Stage Dual-Tower Framework for Multimodal Emotion Inconsistency Detection
- **分类: cs.MM; cs.SD**

- **简介: 该论文提出InconVAD，用于多模态情感不一致性检测。针对语音与文本常存在冲突线索的问题，设计两阶段双塔框架：第一阶段获取鲁棒的单模态预测，第二阶段分类并融合一致信号。实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20140v1](http://arxiv.org/pdf/2509.20140v1)**

> **作者:** Zongyi Li; Junchuan Zhao; Francis Bu Sung Lee; Andrew Zi Han Yee
>
> **备注:** 5 pages, 1 figure, 3 tables
>
> **摘要:** Detecting emotional inconsistency across modalities is a key challenge in affective computing, as speech and text often convey conflicting cues. Existing approaches generally rely on incomplete emotion representations and employ unconditional fusion, which weakens performance when modalities are inconsistent. Moreover, little prior work explicitly addresses inconsistency detection itself. We propose InconVAD, a two-stage framework grounded in the Valence/Arousal/Dominance (VAD) space. In the first stage, independent uncertainty-aware models yield robust unimodal predictions. In the second stage, a classifier identifies cross-modal inconsistency and selectively integrates consistent signals. Extensive experiments show that InconVAD surpasses existing methods in both multimodal emotion inconsistency detection and modeling, offering a more reliable and interpretable solution for emotion analysis.
>
---
#### [new 014] MAGE: A Coarse-to-Fine Speech Enhancer with Masked Generative Model
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MAGE，一种用于语音增强的生成模型。针对效率与感知质量的权衡问题，设计了稀疏感知的粗到细掩码策略和轻量校正模块，显著提升了语音增强效果，在多个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2509.19881v1](http://arxiv.org/pdf/2509.19881v1)**

> **作者:** The Hieu Pham; Tan Dat Nguyen; Phuong Thanh Tran; Joon Son Chun; Duc Dung Nguyen
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech enhancement remains challenging due to the trade-off between efficiency and perceptual quality. In this paper, we introduce MAGE, a Masked Audio Generative Enhancer that advances generative speech enhancement through a compact and robust design. Unlike prior masked generative models with random masking, MAGE employs a scarcity-aware coarse-to-fine masking strategy that prioritizes frequent tokens in early steps and rare tokens in later refinements, improving efficiency and generalization. We also propose a lightweight corrector module that further stabilizes inference by detecting low-confidence predictions and re-masking them for refinement. Built on BigCodec and finetuned from Qwen2.5-0.5B, MAGE is reduced to 200M parameters through selective layer retention. Experiments on DNS Challenge and noisy LibriSpeech show that MAGE achieves state-of-the-art perceptual quality and significantly reduces word error rate for downstream recognition, outperforming larger baselines. Audio examples are available at https://hieugiaosu.github.io/MAGE/.
>
---
#### [new 015] Selective Classifier-free Guidance for Zero-shot Text-to-speech
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文研究零样本文本到语音任务，旨在平衡语音保真度与文本一致性。作者探索了分类器自由引导（CFG）策略在语音合成中的适用性，提出分阶段选择性CFG方法，并发现其效果依赖于文本表示方式。**

- **链接: [http://arxiv.org/pdf/2509.19668v1](http://arxiv.org/pdf/2509.19668v1)**

> **作者:** John Zheng; Farhad Maleki
>
> **备注:** 5 pages, 7 figures, 1 table. Submitted to ICASSP 2026
>
> **摘要:** In zero-shot text-to-speech, achieving a balance between fidelity to the target speaker and adherence to text content remains a challenge. While classifier-free guidance (CFG) strategies have shown promising results in image generation, their application to speech synthesis are underexplored. Separating the conditions used for CFG enables trade-offs between different desired characteristics in speech synthesis. In this paper, we evaluate the adaptability of CFG strategies originally developed for image generation to speech synthesis and extend separated-condition CFG approaches for this domain. Our results show that CFG strategies effective in image generation generally fail to improve speech synthesis. We also find that we can improve speaker similarity while limiting degradation of text adherence by applying standard CFG during early timesteps and switching to selective CFG only in later timesteps. Surprisingly, we observe that the effectiveness of a selective CFG strategy is highly text-representation dependent, as differences between the two languages of English and Mandarin can lead to different results even with the same model.
>
---
#### [new 016] PART: Progressive Alignment Representation Training for Multilingual Speech-To-Text with LLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出PART方法，用于多语言语音到文本任务。针对语音与文本对齐在多语言场景下的挑战，设计了分阶段、多任务框架，动态激活LLM参数，实现语言内与跨语言对齐分离，提升了多语言语音模型的性能和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.19745v1](http://arxiv.org/pdf/2509.19745v1)**

> **作者:** Pei Zhang; Andong Chen; Xi Chen; Baosong Yang; Derek F. Wong; Fei Huang
>
> **摘要:** Large language models (LLMs) have expanded from text to speech, giving rise to Speech Large Models (SLMs) that support recognition, translation, and synthesis. A key challenge is aligning speech and text representations, which becomes harder in multilingual settings. Existing methods often freeze LLM parameters and train encoders on multilingual data, but this forces cross-language convergence and limits performance. We introduce Progressive Alignment Representation Training (PART), a multi-stage and multi-task framework that separates within-language from cross-language alignment. During cross-language training, LLM parameters are dynamically activated, and text-based tasks are later introduced to enhance multilingual understanding. Experiments on CommonVoice 15, Fleurs, Wenetspeech, and CoVoST2 show that PART surpasses conventional approaches, with analysis confirming its ability to balance language-specific distinctions and cross-language generalization. These results demonstrate PART's effectiveness and generality for multilingual speech modality alignment.
>
---
#### [new 017] Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文研究多码本语音生成任务，针对并行预测导致的音质下降问题，提出帧堆叠局部Transformer架构，比较自回归与MaskGIT方法，探索速度与质量的平衡策略。**

- **链接: [http://arxiv.org/pdf/2509.19592v1](http://arxiv.org/pdf/2509.19592v1)**

> **作者:** Roy Fejgin; Paarth Neekhara; Xuesong Yang; Edresson Casanova; Ryan Langman Jaehyeon Kim; Subhankar Ghosh; Shehzeen Hussain; Jason Li
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity.
>
---
#### [new 018] Vision-Based Perception for Autonomous Vehicles in Off-Road Environment Using Deep Learning
- **分类: cs.CV; cs.AR; cs.LG; eess.IV; eess.SP**

- **简介: 该论文研究了自动驾驶车辆在非铺装道路环境下的视觉感知任务，旨在解决复杂地形中实时识别可行驶区域和障碍物的问题。提出了CMSNet模块化分割网络，并构建了Kamino数据集，通过优化实现实时推理，验证了系统有效性。**

- **链接: [http://arxiv.org/pdf/2509.19378v1](http://arxiv.org/pdf/2509.19378v1)**

> **作者:** Nelson Alves Ferreira Neto
>
> **备注:** 2022. 117p. Electrical Engineering PhD Thesis - Graduate Program in Electrical and Computer Engineering, Federal University of Bahia, 40210-630, Salvador, Brazil
>
> **摘要:** Low-latency intelligent systems are required for autonomous driving on non-uniform terrain in open-pit mines and developing countries. This work proposes a perception system for autonomous vehicles on unpaved roads and off-road environments, capable of navigating rough terrain without a predefined trail. The Configurable Modular Segmentation Network (CMSNet) framework is proposed, facilitating different architectural arrangements. CMSNet configurations were trained to segment obstacles and trafficable ground on new images from unpaved/off-road scenarios with adverse conditions (night, rain, dust). We investigated applying deep learning to detect drivable regions without explicit track boundaries, studied algorithm behavior under visibility impairment, and evaluated field tests with real-time semantic segmentation. A new dataset, Kamino, is presented with almost 12,000 images from an operating vehicle with eight synchronized cameras. The Kamino dataset has a high number of labeled pixels compared to similar public collections and includes images from an off-road proving ground emulating a mine under adverse visibility. To achieve real-time inference, CMSNet CNN layers were methodically removed and fused using TensorRT, C++, and CUDA. Empirical experiments on two datasets validated the proposed system's effectiveness.
>
---
#### [new 019] MultiSoundGen: Video-to-Audio Generation for Multi-Event Scenarios via SlowFast Contrastive Audio-Visual Pretraining and Direct Preference Optimization
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出MultiSoundGen，针对复杂多事件场景下的视频生成音频任务（V2A），解决语义-时间对齐和音频质量优化问题。工作包括：设计SlowFast对比视听预训练模型（SF-CAVP）和引入AVP-RPO方法，实现更高质量的多事件音频生成。**

- **链接: [http://arxiv.org/pdf/2509.19999v1](http://arxiv.org/pdf/2509.19999v1)**

> **作者:** Jianxuan Yang; Xiaoran Yang; Lipan Zhang; Xinyue Guo; Zhao Wang; Gongping Huang
>
> **摘要:** Current video-to-audio (V2A) methods struggle in complex multi-event scenarios (video scenarios involving multiple sound sources, sound events, or transitions) due to two critical limitations. First, existing methods face challenges in precisely aligning intricate semantic information together with rapid dynamic features. Second, foundational training lacks quantitative preference optimization for semantic-temporal alignment and audio quality. As a result, it fails to enhance integrated generation quality in cluttered multi-event scenes. To address these core limitations, this study proposes a novel V2A framework: MultiSoundGen. It introduces direct preference optimization (DPO) into the V2A domain, leveraging audio-visual pretraining (AVP) to enhance performance in complex multi-event scenarios. Our contributions include two key innovations: the first is SlowFast Contrastive AVP (SF-CAVP), a pioneering AVP model with a unified dual-stream architecture. SF-CAVP explicitly aligns core semantic representations and rapid dynamic features of audio-visual data to handle multi-event complexity; second, we integrate the DPO method into V2A task and propose AVP-Ranked Preference Optimization (AVP-RPO). It uses SF-CAVP as a reward model to quantify and prioritize critical semantic-temporal matches while enhancing audio quality. Experiments demonstrate that MultiSoundGen achieves state-of-the-art (SOTA) performance in multi-event scenarios, delivering comprehensive gains across distribution matching, audio quality, semantic alignment, and temporal synchronization. The complete code and dataset will be released soon.
>
---
## 更新

#### [replaced 001] To Fold or Not to Fold: Graph Regularized Tensor Train for Visual Data Completion
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.11123v2](http://arxiv.org/pdf/2306.11123v2)**

> **作者:** Le Xu; Lei Cheng; Ngai Wong; Yik-Chung Wu
>
> **摘要:** Tensor train (TT) representation has achieved tremendous success in visual data completion tasks, especially when it is combined with tensor folding. However, folding an image or video tensor breaks the original data structure, leading to local information loss as nearby pixels may be assigned into different dimensions and become far away from each other. In this paper, to fully preserve the local information of the original visual data, we explore not folding the data tensor, and at the same time adopt graph information to regularize local similarity between nearby entries. To overcome the high computational complexity introduced by the graph-based regularization in the TT completion problem, we propose to break the original problem into multiple sub-problems with respect to each TT core fiber, instead of each TT core as in traditional methods. Furthermore, to avoid heavy parameter tuning, a sparsity promoting probabilistic model is built based on the generalized inverse Gaussian (GIG) prior, and an inference algorithm is derived under the mean-field approximation. Experiments on both synthetic data and real-world visual data show the superiority of the proposed methods.
>
---
#### [replaced 002] A GEN AI Framework for Medical Note Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.IR; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.01841v2](http://arxiv.org/pdf/2410.01841v2)**

> **作者:** Hui Yi Leong; Yi Fan Gao; Shuai Ji; Bora Kalaycioglu; Uktu Pamuksuz
>
> **备注:** 8 Figures, 7 page, IEEE standard research paper
>
> **摘要:** The increasing administrative burden of medical documentation, particularly through Electronic Health Records (EHR), significantly reduces the time available for direct patient care and contributes to physician burnout. To address this issue, we propose MediNotes, an advanced generative AI framework designed to automate the creation of SOAP (Subjective, Objective, Assessment, Plan) notes from medical conversations. MediNotes integrates Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Automatic Speech Recognition (ASR) to capture and process both text and voice inputs in real time or from recorded audio, generating structured and contextually accurate medical notes. The framework also incorporates advanced techniques like Quantized Low-Rank Adaptation (QLoRA) and Parameter-Efficient Fine-Tuning (PEFT) for efficient model fine-tuning in resource-constrained environments. Additionally, MediNotes offers a query-based retrieval system, allowing healthcare providers and patients to access relevant medical information quickly and accurately. Evaluations using the ACI-BENCH dataset demonstrate that MediNotes significantly improves the accuracy, efficiency, and usability of automated medical documentation, offering a robust solution to reduce the administrative burden on healthcare professionals while improving the quality of clinical workflows.
>
---
#### [replaced 003] Robust Audio-Visual Target Speaker Extraction with Emotion-Aware Multiple Enrollment Fusion
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.12583v2](http://arxiv.org/pdf/2509.12583v2)**

> **作者:** Zhan Jin; Bang Zeng; Peijun Yang; Jiarong Du; Juan Liu; Ming Li
>
> **摘要:** Target Speaker Extraction (TSE) is a critical challenge in cocktail party scenarios. While leveraging multiple modalities, such as voice, lip, face, and expression embeddings, can enhance performance, real-world applications often suffer from intermittent modality dropout. This paper presents a comprehensive study on the interactions and robustness of various multimodal fusion strategies under varying degrees of modality dropout. We build upon a state-of-the-art audio-visual speech enhancement system and integrate four distinct speaker identity cues: lip embeddings for synchronized contextual information, a voice speaker embedding extracted via cross-attention for acoustic consistency, a static face embedding for speaker identity, and a novel dynamic expression embedding for frame-wise emotional features. We systematically evaluate different combinations of these modalities under two key training regimes: zero dropout and 80% modality dropout. Extensive experiments demonstrate that while a full multimodal ensemble achieves optimal performance under ideal (zero dropout) conditions, its effectiveness diminishes significantly when test-time dropout occurs without prior exposure during training. Crucially, we show that training with a high (80%) modality dropout rate dramatically enhances model robustness, enabling the system to maintain superior performance even under severe test-time missing modalities. Our findings highlight that voice embeddings exhibit consistent robustness, while the proposed expression embedding provides valuable complementary information. This work underscores the importance of training strategies that account for real-world imperfection, moving beyond pure performance maximization to achieve practical reliability in multimodal speech enhancement systems.
>
---
#### [replaced 004] Stylus: Repurposing Stable Diffusion for Training-Free Music Style Transfer on Mel-Spectrograms
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.15913v3](http://arxiv.org/pdf/2411.15913v3)**

> **作者:** Heehwan Wang; Joonwoo Kwon; Sooyoung Kim; Jungwoo Seo; Shinjae Yoo; Yuewei Lin; Jiook Cha
>
> **备注:** Codes will be released upon acceptance
>
> **摘要:** Music style transfer enables personalized music creation by blending the structure of a source with the stylistic attributes of a reference. Existing text-conditioned and diffusion-based approaches show promise but often require paired datasets, extensive training, or detailed annotations. We present Stylus, a training-free framework that repurposes a pre-trained Stable Diffusion model for music style transfer in the mel-spectrogram domain. Stylus manipulates self-attention by injecting style key-value features while preserving source queries to maintain musical structure. To improve fidelity, we introduce a phase-preserving reconstruction strategy that avoids artifacts from Griffin-Lim reconstruction, and we adopt classifier-free-guidance-inspired control for adjustable stylization and multi-style blending. In extensive evaluations, Stylus outperforms state-of-the-art baselines, achieving 34.1% higher content preservation and 25.7% better perceptual quality without any additional training.
>
---
#### [replaced 005] ctPuLSE: Close-Talk, and Pseudo-Label Based Far-Field, Speech Enhancement
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2407.19485v2](http://arxiv.org/pdf/2407.19485v2)**

> **作者:** Zhong-Qiu Wang
>
> **备注:** in Journal of The Acoustical Society of America (JASA)
>
> **摘要:** The current dominant approach for neural speech enhancement is via purely-supervised deep learning on simulated pairs of far-field noisy-reverberant speech (i.e., mixtures) and clean speech. The trained models, however, often exhibit limited generalizability to real-recorded mixtures. To deal with this, this paper investigates training enhancement models directly on real mixtures. However, a major difficulty challenging this approach is that, since the clean speech of real mixtures is unavailable, there lacks a good supervision for real mixtures. In this context, assuming that a training set consisting of real-recorded pairs of close-talk and far-field mixtures is available, we propose to address this difficulty via close-talk speech enhancement, where an enhancement model is first trained on simulated mixtures to enhance real-recorded close-talk mixtures and the estimated close-talk speech can then be utilized as a supervision (i.e., pseudo-label) for training far-field speech enhancement models directly on the paired real-recorded far-field mixtures. We name the proposed system ctPuLSE. Evaluation results on the popular CHiME-4 dataset show that ctPuLSE can derive high-quality pseudo-labels and yield far-field speech enhancement models with strong generalizability to real data.
>
---
#### [replaced 006] EAI-Avatar: Emotion-Aware Interactive Talking Head Generation
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.18337v2](http://arxiv.org/pdf/2508.18337v2)**

> **作者:** Haijie Yang; Zhenyu Zhang; Hao Tang; Jianjun Qian; Jian Yang
>
> **备注:** The submission is withdrawn at the request of the authors due to internal reasons within the research team
>
> **摘要:** Generative models have advanced rapidly, enabling impressive talking head generation that brings AI to life. However, most existing methods focus solely on one-way portrait animation. Even the few that support bidirectional conversational interactions lack precise emotion-adaptive capabilities, significantly limiting their practical applicability. In this paper, we propose EAI-Avatar, a novel emotion-aware talking head generation framework for dyadic interactions. Leveraging the dialogue generation capability of large language models (LLMs, e.g., GPT-4), our method produces temporally consistent virtual avatars with rich emotional variations that seamlessly transition between speaking and listening states. Specifically, we design a Transformer-based head mask generator that learns temporally consistent motion features in a latent mask space, capable of generating arbitrary-length, temporally consistent mask sequences to constrain head motions. Furthermore, we introduce an interactive talking tree structure to represent dialogue state transitions, where each tree node contains information such as child/parent/sibling nodes and the current character's emotional state. By performing reverse-level traversal, we extract rich historical emotional cues from the current node to guide expression synthesis. Extensive experiments demonstrate the superior performance and effectiveness of our method.
>
---
#### [replaced 007] Embedding Alignment in Code Generation for Audio
- **分类: cs.MM; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.05473v2](http://arxiv.org/pdf/2508.05473v2)**

> **作者:** Sam Kouteili; Hiren Madhu; George Typaldos; Mark Santolucito
>
> **备注:** Accepted to NeurIPS 2025 AI4Music Workshop
>
> **摘要:** LLM-powered code generation has the potential to revolutionize creative coding endeavors, such as live-coding, by enabling users to focus on structural motifs over syntactic details. In such domains, when prompting an LLM, users may benefit from considering multiple varied code candidates to better realize their musical intentions. Code generation models, however, struggle to present unique and diverse code candidates, with no direct insight into the code's audio output. To better establish a relationship between code candidates and produced audio, we investigate the topology of the mapping between code and audio embedding spaces. We find that code and audio embeddings do not exhibit a simple linear relationship, but supplement this with a constructed predictive model that shows an embedding alignment map could be learned. Supplementing the aim for musically diverse output, we present a model that given code predicts output audio embedding, constructing a code-audio embedding alignment map.
>
---
#### [replaced 008] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.15176v3](http://arxiv.org/pdf/2408.15176v3)**

> **作者:** Longshen Ou; Jingwei Zhao; Ziyu Wang; Gus Xia; Qihao Liang; Torin Hopkins Ye Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present a unified framework for automatic multitrack music arrangement that enables a single pre-trained symbolic music model to handle diverse arrangement scenarios, including reinterpretation, simplification, and additive generation. At its core is a segment-level reconstruction objective operating on token-level disentangled content and style, allowing for flexible any-to-any instrumentation transformations at inference time. To support track-wise modeling, we introduce REMI-z, a structured tokenization scheme for multitrack symbolic music that enhances modeling efficiency and effectiveness for both arrangement tasks and unconditional generation. Our method outperforms task-specific state-of-the-art models on representative tasks in different arrangement scenarios -- band arrangement, piano reduction, and drum arrangement, in both objective metrics and perceptual evaluations. Taken together, our framework demonstrates strong generality and suggests broader applicability in symbolic music-to-music transformation.
>
---
