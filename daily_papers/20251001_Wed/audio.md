# 音频 cs.SD;  eess.SP

- **最新发布 22 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Benchmarking Diarization Models
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于说话人日志任务，旨在解决多说话人音频分割问题。通过评估五种模型在多个数据集上的表现，分析错误原因并提出改进方向。**

- **链接: [http://arxiv.org/pdf/2509.26177v1](http://arxiv.org/pdf/2509.26177v1)**

> **作者:** Luca A. Lanzendörfer; Florian Grötschla; Cesare Blaser; Roger Wattenhofer
>
> **摘要:** Speaker diarization is the task of partitioning audio into segments according to speaker identity, answering the question of "who spoke when" in multi-speaker conversation recordings. While diarization is an essential task for many downstream applications, it remains an unsolved problem. Errors in diarization propagate to downstream systems and cause wide-ranging failures. To this end, we examine exact failure modes by evaluating five state-of-the-art diarization models, across four diarization datasets spanning multiple languages and acoustic conditions. The evaluation datasets consist of 196.6 hours of multilingual audio, including English, Mandarin, German, Japanese, and Spanish. Overall, we find that PyannoteAI achieves the best performance at 11.2% DER, while DiariZen provides a competitive open-source alternative at 13.3% DER. When analyzing failure cases, we find that the primary cause of diarization errors stem from missed speech segments followed by speaker confusion, especially in high-speaker count settings.
>
---
#### [new 002] Source Separation for A Cappella Music
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于多歌手音源分离任务，解决a cappella音乐中不同数量歌手混合的分离问题。通过数据增强和模型改进，提升分离效果。**

- **链接: [http://arxiv.org/pdf/2509.26580v1](http://arxiv.org/pdf/2509.26580v1)**

> **作者:** Luca A. Lanzendörfer; Constantin Pinkl; Florian Grötschla
>
> **摘要:** In this work, we study the task of multi-singer separation in a cappella music, where the number of active singers varies across mixtures. To address this, we use a power set-based data augmentation strategy that expands limited multi-singer datasets into exponentially more training samples. To separate singers, we introduce SepACap, an adaptation of SepReformer, a state-of-the-art speaker separation model architecture. We adapt the model with periodic activations and a composite loss function that remains effective when stems are silent, enabling robust detection and separation. Experiments on the JaCappella dataset demonstrate that our approach achieves state-of-the-art performance in both full-ensemble and subset singer separation scenarios, outperforming spectrogram-based baselines while generalizing to realistic mixtures with varying numbers of singers.
>
---
#### [new 003] VoiceBridge: Designing Latent Bridge Models for General Speech Restoration at Scale
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音修复任务，旨在解决大规模通用语音恢复问题。提出VoiceBridge系统，利用潜在桥模型实现高质量语音重建。**

- **链接: [http://arxiv.org/pdf/2509.25275v1](http://arxiv.org/pdf/2509.25275v1)**

> **作者:** Chi Zhang; Zehua Chen; Kaiwen Zheng; Jun Zhu
>
> **摘要:** Bridge models have recently been explored for speech enhancement tasks such as denoising, dereverberation, and super-resolution, while these efforts are typically confined to a single task or small-scale datasets, with constrained general speech restoration (GSR) capability at scale. In this work, we introduce VoiceBridge, a GSR system rooted in latent bridge models (LBMs), capable of reconstructing high-fidelity speech at full-band (\textit{i.e.,} 48~kHz) from various distortions. By compressing speech waveform into continuous latent representations, VoiceBridge models the~\textit{diverse LQ-to-HQ tasks} (namely, low-quality to high-quality) in GSR with~\textit{a single latent-to-latent generative process} backed by a scalable transformer architecture. To better inherit the advantages of bridge models from the data domain to the latent space, we present an energy-preserving variational autoencoder, enhancing the alignment between the waveform and latent space over varying energy levels. Furthermore, to address the difficulty of HQ reconstruction from distinctively different LQ priors, we propose a joint neural prior, uniformly alleviating the reconstruction burden of LBM. At last, considering the key requirement of GSR systems, human perceptual quality, a perceptually aware fine-tuning stage is designed to mitigate the cascading mismatch in generation while improving perceptual alignment. Extensive validation across in-domain and out-of-domain tasks and datasets (\textit{e.g.}, refining recent zero-shot speech and podcast generation results) demonstrates the superior performance of VoiceBridge. Demo samples can be visited at: https://VoiceBridge-demo.github.io/.
>
---
#### [new 004] LTA-L2S: Lexical Tone-Aware Lip-to-Speech Synthesis for Mandarin with Cross-Lingual Transfer Learning
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于唇语到语音合成任务，旨在解决中文中声调对语音可懂性的影响问题。通过跨语言迁移学习和流匹配模型提升音高轮廓生成效果。**

- **链接: [http://arxiv.org/pdf/2509.25670v1](http://arxiv.org/pdf/2509.25670v1)**

> **作者:** Kang Yang; Yifan Liang; Fangkun Liu; Zhenping Xie; Chengshi Zheng
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Lip-to-speech (L2S) synthesis for Mandarin is a significant challenge, hindered by complex viseme-to-phoneme mappings and the critical role of lexical tones in intelligibility. To address this issue, we propose Lexical Tone-Aware Lip-to-Speech (LTA-L2S). To tackle viseme-to-phoneme complexity, our model adapts an English pre-trained audio-visual self-supervised learning (SSL) model via a cross-lingual transfer learning strategy. This strategy not only transfers universal knowledge learned from extensive English data to the Mandarin domain but also circumvents the prohibitive cost of training such a model from scratch. To specifically model lexical tones and enhance intelligibility, we further employ a flow-matching model to generate the F0 contour. This generation process is guided by ASR-fine-tuned SSL speech units, which contain crucial suprasegmental information. The overall speech quality is then elevated through a two-stage training paradigm, where a flow-matching postnet refines the coarse spectrogram from the first stage. Extensive experiments demonstrate that LTA-L2S significantly outperforms existing methods in both speech intelligibility and tonal accuracy.
>
---
#### [new 005] OWL: Geometry-Aware Spatial Reasoning for Audio Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频大模型任务，旨在提升空间推理能力。解决现有模型依赖粗略标签和缺乏几何监督的问题，提出SAGE和OWL框架，结合3D空间信息进行更精确的方向与距离估计。**

- **链接: [http://arxiv.org/pdf/2509.26140v1](http://arxiv.org/pdf/2509.26140v1)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Spatial reasoning is fundamental to auditory perception, yet current audio large language models (ALLMs) largely rely on unstructured binaural cues and single step inference. This limits both perceptual accuracy in direction and distance estimation and the capacity for interpretable reasoning. Recent work such as BAT demonstrates spatial QA with binaural audio, but its reliance on coarse categorical labels (left, right, up, down) and the absence of explicit geometric supervision constrain resolution and robustness. We introduce the $\textbf{Spatial-Acoustic Geometry Encoder (SAGE}$), a geometry-aware audio encoder that aligns binaural acoustic features with 3D spatial structure using panoramic depth images and room-impulse responses at training time, while requiring only audio at inference. Building on this representation, we present $\textbf{OWL}$, an ALLM that integrates $\textbf{SAGE}$ with a spatially grounded chain-of-thought to rationalize over direction-of-arrivals (DoA) and distance estimates. Through curriculum learning from perceptual QA to multi-step reasoning, $\textbf{OWL}$ supports o'clock-level azimuth and DoA estimation. To enable large-scale training and evaluation, we construct and release $\textbf{BiDepth}$, a dataset of over one million QA pairs combining binaural audio with panoramic depth images and room impulse responses across both in-room and out-of-room scenarios. Across two benchmark datasets, our new $\textbf{BiDepth}$ and the public SpatialSoundQA, $\textbf{OWL}$ reduces mean DoA error by $\textbf{11$^{\circ}$}$ through $\textbf{SAGE}$ and improves spatial reasoning QA accuracy by up to $\textbf{25}$\% over BAT.
>
---
#### [new 006] Learning Relationships Between Separate Audio Tracks for Creative Applications
- **分类: cs.SD; cs.AI; cs.HC; cs.LG; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在通过学习音频轨道间关系，实现实时音乐输出与输入的协调。工作包括构建数据集、设计基于Transformer的决策模块及音频合成方法。**

- **链接: [http://arxiv.org/pdf/2509.25296v1](http://arxiv.org/pdf/2509.25296v1)**

> **作者:** Balthazar Bujard; Jérôme Nika; Fédéric Bevilacqua; Nicolas Obin
>
> **摘要:** This paper presents the first step in a research project situated within the field of musical agents. The objective is to achieve, through training, the tuning of the desired musical relationship between a live musical input and a real-time generated musical output, through the curation of a database of separated tracks. We propose an architecture integrating a symbolic decision module capable of learning and exploiting musical relationships from such musical corpus. We detail an offline implementation of this architecture employing Transformers as the decision module, associated with a perception module based on Wav2Vec 2.0, and concatenative synthesis as audio renderer. We present a quantitative evaluation of the decision module's ability to reproduce learned relationships extracted during training. We demonstrate that our decision module can predict a coherent track B when conditioned by its corresponding ''guide'' track A, based on a corpus of paired tracks (A, B).
>
---
#### [new 007] HNote: Extending YNote with Hexadecimal Encoding for Fine-Tuning LLMs in Music Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐建模任务，旨在解决传统音乐格式不适配LLMs的问题。提出HNote编码系统，实现音乐数据的高效表示与模型微调。**

- **链接: [http://arxiv.org/pdf/2509.25694v1](http://arxiv.org/pdf/2509.25694v1)**

> **作者:** Hung-Ying Chu; Shao-Yu Wei; Guan-Wei Chen; Tzu-Wei Hung; ChengYang Tsai; Yu-Cheng Lin
>
> **摘要:** Recent advances in large language models (LLMs) have created new opportunities for symbolic music generation. However, existing formats such as MIDI, ABC, and MusicXML are either overly complex or structurally inconsistent, limiting their suitability for token-based learning architectures. To address these challenges, we propose HNote, a novel hexadecimal-based notation system extended from YNote, which encodes both pitch and duration within a fixed 32-unit measure framework. This design ensures alignment, reduces ambiguity, and is directly compatible with LLM architectures. We converted 12,300 Jiangnan-style songs generated from traditional folk pieces from YNote into HNote, and fine-tuned LLaMA-3.1(8B) using parameter-efficient LoRA. Experimental results show that HNote achieves a syntactic correctness rate of 82.5%, and BLEU and ROUGE evaluations demonstrate strong symbolic and structural similarity, producing stylistically coherent compositions. This study establishes HNote as an effective framework for integrating LLMs with cultural music modeling.
>
---
#### [new 008] MUSE-Explainer: Counterfactual Explanations for Symbolic Music Graph Classification Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐分析任务，旨在提升图神经网络模型的可解释性。通过生成符合音乐逻辑的反事实解释，帮助理解模型决策过程。**

- **链接: [http://arxiv.org/pdf/2509.26521v1](http://arxiv.org/pdf/2509.26521v1)**

> **作者:** Baptiste Hilaire; Emmanouil Karystinaios; Gerhard Widmer
>
> **备注:** Accepted at the 17th International Symposium on Computer Music Multidisciplinary Research (CMMR) 2025
>
> **摘要:** Interpretability is essential for deploying deep learning models in symbolic music analysis, yet most research emphasizes model performance over explanation. To address this, we introduce MUSE-Explainer, a new method that helps reveal how music Graph Neural Network models make decisions by providing clear, human-friendly explanations. Our approach generates counterfactual explanations by making small, meaningful changes to musical score graphs that alter a model's prediction while ensuring the results remain musically coherent. Unlike existing methods, MUSE-Explainer tailors its explanations to the structure of musical data and avoids unrealistic or confusing outputs. We evaluate our method on a music analysis task and show it offers intuitive insights that can be visualized with standard music tools such as Verovio.
>
---
#### [new 009] The silence of the weights: an investigation of structural pruning strategies for attention-based audio signal architectures
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决Transformer模型参数过多的问题。通过结构化剪枝技术，对注意力机制进行有效裁剪，提升模型效率。**

- **链接: [http://arxiv.org/pdf/2509.26207v1](http://arxiv.org/pdf/2509.26207v1)**

> **作者:** Andrea Diecidue; Carlo Alberto Barbano; Piero Fraternali; Mathieu Fontaine; Enzo Tartaglione
>
> **摘要:** Transformer-based models have become the state of the art across multiple domains, from natural language processing to machine listening, thanks to attention mechanisms. However, the attention layers require a large number of parameters and high-end hardware for both training and inference. We propose a novel pruning technique targeted explicitly at the attention mechanism, where we decouple the pruning of the four layers in the attention block, namely: query, keys, values and outputs' projection matrices. We also investigate pruning strategies to prune along the head and channel dimensions, and compare the performance of the Audio Spectrogram Transformer (AST) model under different pruning scenarios. Our results show that even by pruning 50\% of the attention parameters we incur in performance degradation of less than 1\%
>
---
#### [new 010] MARS: Audio Generation via Multi-Channel Autoregression on Spectrograms
- **分类: cs.SD**

- **简介: 该论文属于音频生成任务，旨在提升音频合成的精度与效率。提出MARS框架，通过多通道自回归和谱图处理，实现高质量音频生成。**

- **链接: [http://arxiv.org/pdf/2509.26007v1](http://arxiv.org/pdf/2509.26007v1)**

> **作者:** Eleonora Ristori; Luca Bindini; Paolo Frasconi
>
> **摘要:** Research on audio generation has progressively shifted from waveform-based approaches to spectrogram-based methods, which more naturally capture harmonic and temporal structures. At the same time, advances in image synthesis have shown that autoregression across scales, rather than tokens, improves coherence and detail. Building on these ideas, we introduce MARS (Multi-channel AutoRegression on Spectrograms), a framework that treats spectrograms as multi-channel images and employs channel multiplexing (CMX), a reshaping technique that lowers height and width without discarding information. A shared tokenizer provides consistent discrete representations across scales, enabling a transformer-based autoregressor to refine spectrograms from coarse to fine resolutions efficiently. Experiments on a large-scale dataset demonstrate that MARS performs comparably or better than state-of-the-art baselines across multiple evaluation metrics, establishing an efficient and scalable paradigm for high-fidelity audio generation.
>
---
#### [new 011] EMO-TTA: Improving Test-Time Adaptation of Audio-Language Models for Speech Emotion Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音情感识别任务，解决音频语言模型在测试时分布变化导致的性能下降问题。提出Emo-TTA框架，通过统计适应提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.25495v1](http://arxiv.org/pdf/2509.25495v1)**

> **作者:** Jiacheng Shi; Hongfei Du; Y. Alicia Hong; Ye Gao
>
> **摘要:** Speech emotion recognition (SER) with audio-language models (ALMs) remains vulnerable to distribution shifts at test time, leading to performance degradation in out-of-domain scenarios. Test-time adaptation (TTA) provides a promising solution but often relies on gradient-based updates or prompt tuning, limiting flexibility and practicality. We propose Emo-TTA, a lightweight, training-free adaptation framework that incrementally updates class-conditional statistics via an Expectation-Maximization procedure for explicit test-time distribution estimation, using ALM predictions as priors. Emo-TTA operates on individual test samples without modifying model weights. Experiments on six out-of-domain SER benchmarks show consistent accuracy improvements over prior TTA baselines, demonstrating the effectiveness of statistical adaptation in aligning model predictions with evolving test distributions.
>
---
#### [new 012] Representation-Based Data Quality Audits for Audio
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于音频数据质量审计任务，解决数据中的错误样本、重复和标签问题。通过自监督音频表示提升数据审核效果，实现高效的人工审查。**

- **链接: [http://arxiv.org/pdf/2509.26291v1](http://arxiv.org/pdf/2509.26291v1)**

> **作者:** Alvaro Gonzalez-Jimenez; Fabian Gröger; Linda Wermelinger; Andrin Bürli; Iason Kastanis; Simone Lionetti; Marc Pouly
>
> **摘要:** Data quality issues such as off-topic samples, near duplicates, and label errors often limit the performance of audio-based systems. This paper addresses these issues by adapting SelfClean, a representation-to-rank data auditing framework, from the image to the audio domain. This approach leverages self-supervised audio representations to identify common data quality issues, creating ranked review lists that surface distinct issues within a single, unified process. The method is benchmarked on the ESC-50, GTZAN, and a proprietary industrial dataset, using both synthetic and naturally occurring corruptions. The results demonstrate that this framework achieves state-of-the-art ranking performance, often outperforming issue-specific baselines and enabling significant annotation savings by efficiently guiding human review.
>
---
#### [new 013] Ethics Statements in AI Music Papers: The Effective and the Ineffective
- **分类: cs.CY; cs.SD**

- **简介: 该论文属于AI音乐研究中的伦理分析任务，旨在解决伦理声明在AI音乐论文中使用效果不佳的问题，通过文献回顾提出改进建议。**

- **链接: [http://arxiv.org/pdf/2509.25496v1](http://arxiv.org/pdf/2509.25496v1)**

> **作者:** Julia Barnett; Patrick O'Reilly; Jason Brent Smith; Annie Chu; Bryan Pardo
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI for Music
>
> **摘要:** While research in AI methods for music generation and analysis has grown in scope and impact, AI researchers' engagement with the ethical consequences of this work has not kept pace. To encourage such engagement, many publication venues have introduced optional or required ethics statements for AI research papers. Though some authors use these ethics statements to critically engage with the broader implications of their research, we find that the majority of ethics statements in the AI music literature do not appear to be effectively utilized for this purpose. In this work, we conduct a review of ethics statements across ISMIR, NIME, and selected prominent works in AI music from the past five years. We then offer suggestions for both audio conferences and researchers for engaging with ethics statements in ways that foster meaningful reflection rather than formulaic compliance.
>
---
#### [new 014] Optimizing Speech Language Models for Acoustic Consistency
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音语言模型优化任务，解决语音生成一致性问题。通过语义初始化和损失函数设计，提升模型在不同条件下的稳定性与语义对齐能力。**

- **链接: [http://arxiv.org/pdf/2509.26276v1](http://arxiv.org/pdf/2509.26276v1)**

> **作者:** Morteza Rohanian; Michael Krauthammer
>
> **摘要:** We study speech language models that incorporate semantic initialization and planning losses to achieve robust and consistent generation. Our approach initializes speech tokens with self-supervised features, applies a light alignment loss, and trains with thinning and auxiliary objectives that target robustness and content planning. We train three models: a 0.7B speech-only model, a 1.0B speech-only model, and a 1.0B interleaved model with both text and speech. Acoustic studies show that the speech-only models achieve the highest consistency across speaker, gender, sentiment, room, and background factors, surpassing larger systems. Interleaving improves lexical and syntactic probes and semantic--acoustic alignment but reduces consistency. Linear probes show that our initialization biases the model toward content structure while trading off prosody detail. These results show that LM-side design and training mix control the balance between acoustic stability and semantic grounding without changes to the tokenizer or runtime architecture. A demo and model weights are available for exploration.
>
---
#### [new 015] YOLO-Based Defect Detection for Metal Sheets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; eess.SP; 68T45, 68T07; I.2.10; I.4.7; I.5.4**

- **简介: 该论文属于工业缺陷检测任务，旨在解决金属板表面缺陷识别效率低的问题。通过结合YOLOv9与ConSinGAN提升检测精度与速度，构建自动化光学检测系统。**

- **链接: [http://arxiv.org/pdf/2509.25659v1](http://arxiv.org/pdf/2509.25659v1)**

> **作者:** Po-Heng Chou; Chun-Chi Wang; Wei-Lung Mao
>
> **备注:** 5 pages, 8 figures, 2 tables, and published in IEEE IST 2024
>
> **摘要:** In this paper, we propose a YOLO-based deep learning (DL) model for automatic defect detection to solve the time-consuming and labor-intensive tasks in industrial manufacturing. In our experiments, the images of metal sheets are used as the dataset for training the YOLO model to detect the defects on the surfaces and in the holes of metal sheets. However, the lack of metal sheet images significantly degrades the performance of detection accuracy. To address this issue, the ConSinGAN is used to generate a considerable amount of data. Four versions of the YOLO model (i.e., YOLOv3, v4, v7, and v9) are combined with the ConSinGAN for data augmentation. The proposed YOLOv9 model with ConSinGAN outperforms the other YOLO models with an accuracy of 91.3%, and a detection time of 146 ms. The proposed YOLOv9 model is integrated into manufacturing hardware and a supervisory control and data acquisition (SCADA) system to establish a practical automated optical inspection (AOI) system. Additionally, the proposed automated defect detection is easily applied to other components in industrial manufacturing.
>
---
#### [new 016] Iterative Residual Cross-Attention Mechanism: An Integrated Approach for Audio-Visual Navigation Tasks
- **分类: cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于音频-视觉导航任务，解决传统模块化方法中的信息冗余与传输不一致问题，提出IRCAM-AVN框架实现端到端多模态融合与序列建模。**

- **链接: [http://arxiv.org/pdf/2509.25652v1](http://arxiv.org/pdf/2509.25652v1)**

> **作者:** Hailong Zhang; Yinfeng Yu; Liejun Wang; Fuchun Sun; Wendong Zheng
>
> **备注:** Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2025
>
> **摘要:** Audio-visual navigation represents a significant area of research in which intelligent agents utilize egocentric visual and auditory perceptions to identify audio targets. Conventional navigation methodologies typically adopt a staged modular design, which involves first executing feature fusion, then utilizing Gated Recurrent Unit (GRU) modules for sequence modeling, and finally making decisions through reinforcement learning. While this modular approach has demonstrated effectiveness, it may also lead to redundant information processing and inconsistencies in information transmission between the various modules during the feature fusion and GRU sequence modeling phases. This paper presents IRCAM-AVN (Iterative Residual Cross-Attention Mechanism for Audiovisual Navigation), an end-to-end framework that integrates multimodal information fusion and sequence modeling within a unified IRCAM module, thereby replacing the traditional separate components for fusion and GRU. This innovative mechanism employs a multi-level residual design that concatenates initial multimodal sequences with processed information sequences. This methodological shift progressively optimizes the feature extraction process while reducing model bias and enhancing the model's stability and generalization capabilities. Empirical results indicate that intelligent agents employing the iterative residual cross-attention mechanism exhibit superior navigation performance.
>
---
#### [new 017] Contrastive Diffusion Guidance for Spatial Inverse Problems
- **分类: cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于空间逆问题任务，旨在从用户移动轨迹重建房间布局。针对路径规划不可逆的问题，提出基于对比损失的嵌入空间优化方法，提升重建一致性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.26489v1](http://arxiv.org/pdf/2509.26489v1)**

> **作者:** Sattwik Basu; Chaitanya Amballa; Zhongweiyang Xu; Jorge Vančo Sampedro; Srihari Nelakuditi; Romit Roy Choudhury
>
> **摘要:** We consider the inverse problem of reconstructing the spatial layout of a place, a home floorplan for example, from a user`s movements inside that layout. Direct inversion is ill-posed since many floorplans can explain the same movement trajectories. We adopt a diffusion-based posterior sampler to generate layouts consistent with the measurements. While active research is in progress on generative inverse solvers, we find that the forward operator in our problem poses new challenges. The path-planning process inside a floorplan is a non-invertible, non-differentiable function, and causes instability while optimizing using the likelihood score. We break-away from existing approaches and reformulate the likelihood score in a smoother embedding space. The embedding space is trained with a contrastive loss which brings compatible floorplans and trajectories close to each other, while pushing mismatched pairs far apart. We show that a surrogate form of the likelihood score in this embedding space is a valid approximation of the true likelihood score, making it possible to steer the denoising process towards the posterior. Across extensive experiments, our model CoGuide produces more consistent floorplans from trajectories, and is more robust than differentiable-planner baselines and guided-diffusion methods.
>
---
#### [new 018] Emotion-Aligned Generation in Diffusion Text to Speech Models via Preference-Guided Optimization
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音生成任务，解决情感控制不足的问题。提出EASPO框架，通过细粒度情感偏好优化，提升情感表达自然度。**

- **链接: [http://arxiv.org/pdf/2509.25416v1](http://arxiv.org/pdf/2509.25416v1)**

> **作者:** Jiacheng Shi; Hongfei Du; Yangfan He; Y. Alicia Hong; Ye Gao
>
> **摘要:** Emotional text-to-speech seeks to convey affect while preserving intelligibility and prosody, yet existing methods rely on coarse labels or proxy classifiers and receive only utterance-level feedback. We introduce Emotion-Aware Stepwise Preference Optimization (EASPO), a post-training framework that aligns diffusion TTS with fine-grained emotional preferences at intermediate denoising steps. Central to our approach is EASPM, a time-conditioned model that scores noisy intermediate speech states and enables automatic preference pair construction. EASPO optimizes generation to match these stepwise preferences, enabling controllable emotional shaping. Experiments show superior performance over existing methods in both expressiveness and naturalness.
>
---
#### [new 019] Voice Evaluation of Reasoning Ability: Diagnosing the Modality-Induced Performance Gap
- **分类: eess.AS; cs.MM; cs.SD**

- **简介: 该论文提出VERA，评估语音交互系统在实时对话中的推理能力，解决语音与文本模态性能差异问题，通过对比实验分析不同架构的影响。**

- **链接: [http://arxiv.org/pdf/2509.26542v1](http://arxiv.org/pdf/2509.26542v1)**

> **作者:** Yueqian Lin; Zhengmian Hu; Qinsi Wang; Yudong Liu; Hengfan Zhang; Jayakumar Subramanian; Nikos Vlassis; Hai Helen Li; Yiran Chen
>
> **备注:** Code and data available at https://github.com/linyueqian/VERA
>
> **摘要:** We present Voice Evaluation of Reasoning Ability (VERA), a benchmark for evaluating reasoning ability in voice-interactive systems under real-time conversational constraints. VERA comprises 2,931 voice-native episodes derived from established text benchmarks and organized into five tracks (Math, Web, Science, Long-Context, Factual). Each item is adapted for speech interaction while preserving reasoning difficulty. VERA enables direct text-voice comparison within model families and supports analysis of how architectural choices affect reliability. We assess 12 contemporary voice systems alongside strong text baselines and observe large, consistent modality gaps: on competition mathematics a leading text model attains 74.8% accuracy while its voice counterpart reaches 6.1%; macro-averaged across tracks the best text models achieve 54.0% versus 11.3% for voice. Latency-accuracy analyses reveal a low-latency plateau, where fast voice systems cluster around ~10% accuracy, while approaching text performance requires sacrificing real-time interaction. Diagnostic experiments indicate that common mitigations are insufficient. Increasing "thinking time" yields negligible gains; a decoupled cascade that separates reasoning from narration improves accuracy but still falls well short of text and introduces characteristic grounding/consistency errors. Failure analyses further show distinct error signatures across native streaming, end-to-end, and cascade designs. VERA provides a reproducible testbed and targeted diagnostics for architectures that decouple thinking from speaking, offering a principled way to measure progress toward real-time voice assistants that are both fluent and reliably reasoned.
>
---
#### [new 020] TAU: A Benchmark for Cultural Sound Understanding Beyond Semantics
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于多模态理解任务，旨在解决现有模型对文化特有非语义音频理解不足的问题。通过构建TAU基准，评估模型在本地化音频上的表现。**

- **链接: [http://arxiv.org/pdf/2509.26329v1](http://arxiv.org/pdf/2509.26329v1)**

> **作者:** Yi-Cheng Lin; Yu-Hua Chen; Jia-Kai Dong; Yueh-Hsuan Huang; Szu-Chi Chen; Yu-Chen Chen; Chih-Yao Chen; Yu-Jung Lin; Yu-Ling Chen; Zih-Yu Chen; I-Ning Tsai; Hsiu-Hsuan Wang; Ho-Lam Chung; Ke-Han Lu; Hung-yi Lee
>
> **备注:** 5 pages; submitted to ICASSP 2026
>
> **摘要:** Large audio-language models are advancing rapidly, yet most evaluations emphasize speech or globally sourced sounds, overlooking culturally distinctive cues. This gap raises a critical question: can current models generalize to localized, non-semantic audio that communities instantly recognize but outsiders do not? To address this, we present TAU (Taiwan Audio Understanding), a benchmark of everyday Taiwanese "soundmarks." TAU is built through a pipeline combining curated sources, human editing, and LLM-assisted question generation, producing 702 clips and 1,794 multiple-choice items that cannot be solved by transcripts alone. Experiments show that state-of-the-art LALMs, including Gemini 2.5 and Qwen2-Audio, perform far below local humans. TAU demonstrates the need for localized benchmarks to reveal cultural blind spots, guide more equitable multimodal evaluation, and ensure models serve communities beyond the global mainstream.
>
---
#### [new 021] Can VLM Pseudo-Labels Train a Time-Series QA Model That Outperforms the VLM?
- **分类: cs.LG; cs.CL; eess.SP**

- **简介: 该论文属于时间序列问答任务，解决标签数据不足的问题。通过使用VLM生成的伪标签训练模型，利用大量未标注数据提升性能，效果优于VLM本身。**

- **链接: [http://arxiv.org/pdf/2509.25696v1](http://arxiv.org/pdf/2509.25696v1)**

> **作者:** Takuya Fujimura; Kota Dohi; Natsuo Yamashita; Yohei Kawaguchi
>
> **摘要:** Time-series question answering (TSQA) tasks face significant challenges due to the lack of labeled data. Alternatively, with recent advancements in large-scale models, vision-language models (VLMs) have demonstrated the potential to analyze time-series signals in a zero-shot manner. In this paper, we propose a training approach that uses pseudo labels generated by a VLM. Although VLMs can produce incorrect labels, TSQA models can still be effectively trained based on the property that deep neural networks are inherently robust to such noisy labels. Our experimental results demonstrate that TSQA models are not only successfully trained with pseudo labels, but also surpass the performance of the VLM itself by leveraging a large amount of unlabeled data.
>
---
#### [new 022] Echoes of Humanity: Exploring the Perceived Humanness of AI Music
- **分类: cs.AI; cs.HC; cs.SD**

- **简介: 该论文属于AI音乐感知任务，研究人类如何区分AI与人工创作的音乐。通过实验和分析，探讨影响判断的因素及线索。**

- **链接: [http://arxiv.org/pdf/2509.25601v1](http://arxiv.org/pdf/2509.25601v1)**

> **作者:** Flavio Figueiredo; Giovanni Martinelli; Henrique Sousa; Pedro Rodrigues; Frederico Pedrosa; Lucas N. Ferreira
>
> **备注:** Accepted at NeuRIPs 2025 Creative AI Track
>
> **摘要:** Recent advances in AI music (AIM) generation services are currently transforming the music industry. Given these advances, understanding how humans perceive AIM is crucial both to educate users on identifying AIM songs, and, conversely, to improve current models. We present results from a listener-focused experiment aimed at understanding how humans perceive AIM. In a blind, Turing-like test, participants were asked to distinguish, from a pair, the AIM and human-made song. We contrast with other studies by utilizing a randomized controlled crossover trial that controls for pairwise similarity and allows for a causal interpretation. We are also the first study to employ a novel, author-uncontrolled dataset of AIM songs from real-world usage of commercial models (i.e., Suno). We establish that listeners' reliability in distinguishing AIM causally increases when pairs are similar. Lastly, we conduct a mixed-methods content analysis of listeners' free-form feedback, revealing a focus on vocal and technical cues in their judgments.
>
---
## 更新

#### [replaced 001] Regularizing Learnable Feature Extraction for Automatic Speech Recognition
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.09804v2](http://arxiv.org/pdf/2506.09804v2)**

> **作者:** Peter Vieting; Maximilian Kannen; Benedikt Hilmes; Ralf Schlüter; Hermann Ney
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Neural front-ends are an appealing alternative to traditional, fixed feature extraction pipelines for automatic speech recognition (ASR) systems since they can be directly trained to fit the acoustic model. However, their performance often falls short compared to classical methods, which we show is largely due to their increased susceptibility to overfitting. This work therefore investigates regularization methods for training ASR models with learnable feature extraction front-ends. First, we examine audio perturbation methods and show that larger relative improvements can be obtained for learnable features. Additionally, we identify two limitations in the standard use of SpecAugment for these front-ends and propose masking in the short time Fourier transform (STFT)-domain as a simple but effective modification to address these challenges. Finally, integrating both regularization approaches effectively closes the performance gap between traditional and learnable features.
>
---
#### [replaced 002] The Inverse Drum Machine: Source Separation Through Joint Transcription and Analysis-by-Synthesis
- **分类: cs.SD; eess.AS; eess.SP; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.03337v2](http://arxiv.org/pdf/2505.03337v2)**

> **作者:** Bernardo Torres; Geoffroy Peeters; Gael Richard
>
> **摘要:** We present the Inverse Drum Machine, a novel approach to Drum Source Separation that leverages an analysis-by-synthesis framework combined with deep learning. Unlike recent supervised methods that require isolated stem recordings for training, our approach is trained on drum mixtures with only transcription annotations. IDM integrates Automatic Drum Transcription and One-shot Drum Sample Synthesis, jointly optimizing these tasks in an end-to-end manner. By convolving synthesized one-shot samples with estimated onsets, akin to a drum machine, we reconstruct the individual drum stems and train a Deep Neural Network on the reconstruction of the mixture. Experiments on the StemGMD dataset demonstrate that IDM achieves separation quality comparable to state-of-the-art supervised methods that require isolated stems data.
>
---
#### [replaced 003] VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.24773v2](http://arxiv.org/pdf/2509.24773v2)**

> **作者:** Xin Cheng; Yuyue Wang; Xihua Wang; Yihan Wu; Kaisi Guan; Yijing Chen; Peng Zhang; Xiaojiang Liu; Meng Cao; Ruihua Song
>
> **备注:** Paper Under Review
>
> **摘要:** Video-conditioned sound and speech generation, encompassing video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks, are conventionally addressed as separate tasks, with limited exploration to unify them within a signle framework. Recent attempts to unify V2S and VisualTTS face challenges in handling distinct condition types (e.g., heterogeneous video and transcript conditions) and require complex training stages. Unifying these two tasks remains an open problem. To bridge this gap, we present VSSFlow, which seamlessly integrates both V2S and VisualTTS tasks into a unified flow-matching framework. VSSFlow uses a novel condition aggregation mechanism to handle distinct input signals. We find that cross-attention and self-attention layer exhibit different inductive biases in the process of introducing condition. Therefore, VSSFlow leverages these inductive biases to effectively handle different representations: cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. Furthermore, contrary to the prevailing belief that joint training on the two tasks requires complex training strategies and may degrade performance, we find that VSSFlow benefits from the end-to-end joint learning process for sound and speech generation without extra designs on training stages. Detailed analysis attributes it to the learned general audio prior shared between tasks, which accelerates convergence, enhances conditional generation, and stabilizes the classifier-free guidance process. Extensive experiments demonstrate that VSSFlow surpasses the state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks, underscoring the critical potential of unified generative models.
>
---
#### [replaced 004] Universal Speech Enhancement with Regression and Generative Mamba
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.21198v2](http://arxiv.org/pdf/2505.21198v2)**

> **作者:** Rong Chao; Rauf Nasretdinov; Yu-Chiang Frank Wang; Ante Jukić; Szu-Wei Fu; Yu Tsao
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The Interspeech 2025 URGENT Challenge aimed to advance universal, robust, and generalizable speech enhancement by unifying speech enhancement tasks across a wide variety of conditions, including seven different distortion types and five languages. We present Universal Speech Enhancement Mamba (USEMamba), a state-space speech enhancement model designed to handle long-range sequence modeling, time-frequency structured processing, and sampling frequency-independent feature extraction. Our approach primarily relies on regression-based modeling, which performs well across most distortions. However, for packet loss and bandwidth extension, where missing content must be inferred, a generative variant of the proposed USEMamba proves more effective. Despite being trained on only a subset of the full training data, USEMamba achieved 2nd place in Track 1 during the blind test phase, demonstrating strong generalization across diverse conditions.
>
---
#### [replaced 005] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14142v2](http://arxiv.org/pdf/2505.14142v2)**

> **作者:** Gijs Wijngaard; Elia Formisano; Michele Esposito; Michel Dumontier
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Audio-language models have shown promising results in various sound understanding tasks, yet they remain limited in their ability to reason over the fine-grained semantics of sound. In this paper, we present AudSemThinker, a model whose reasoning is structured around a framework of auditory semantics inspired by human cognition. To support this, we introduce AudSem, a novel dataset specifically curated for semantic descriptor reasoning in audio-language models. AudSem addresses the persistent challenge of data contamination in zero-shot evaluations by providing a carefully filtered collection of audio samples paired with captions generated through a robust multi-stage pipeline. Our experiments demonstrate that AudSemThinker outperforms state-of-the-art models across multiple training settings, highlighting its strength in semantic audio reasoning. Both AudSemThinker and the AudSem dataset are released publicly.
>
---
#### [replaced 006] Filling MIDI Velocity using U-Net Image Colorizer
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.07751v2](http://arxiv.org/pdf/2508.07751v2)**

> **作者:** Zhanhong He; David Cooper; Defeng Huang; Roberto Togneri
>
> **备注:** accepted to CMMR2025 conference
>
> **摘要:** Modern music producers commonly use MIDI (Musical Instrument Digital Interface) to store their musical compositions. However, MIDI files created with digital software may lack the expressive characteristics of human performances, essentially leaving the velocity parameter - a control for note loudness - undefined, which defaults to a flat value. The task of filling MIDI velocity is termed MIDI velocity prediction, which uses regression models to enhance music expressiveness by adjusting only this parameter. In this paper, we introduce the U-Net, a widely adopted architecture in image colorization, to this task. By conceptualizing MIDI data as images, we adopt window attention and develop a custom loss function to address the sparsity of MIDI-converted images. Current dataset availability restricts our experiments to piano data. Evaluated on the MAESTRO v3 and SMD datasets, our proposed method for filling MIDI velocity outperforms previous approaches in both quantitative metrics and qualitative listening tests.
>
---
#### [replaced 007] Discovering and Steering Interpretable Concepts in Large Generative Music Models
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18186v2](http://arxiv.org/pdf/2505.18186v2)**

> **作者:** Nikhil Singh; Manuel Cherep; Pattie Maes
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** The fidelity with which neural networks can now generate content such as music presents a scientific opportunity: these systems appear to have learned implicit theories of such content's structure through statistical learning alone. This offers a potentially new lens on theories of human-generated media. When internal representations align with traditional constructs (e.g. chord progressions in music), they show how such categories can emerge from statistical regularities; when they diverge, they expose limits of existing frameworks and patterns we may have overlooked but that nonetheless carry explanatory power. In this paper, focusing on music generators, we introduce a method for discovering interpretable concepts using sparse autoencoders (SAEs), extracting interpretable features from the residual stream of a transformer model. We make this approach scalable and evaluable using automated labeling and validation pipelines. Our results reveal both familiar musical concepts and coherent but uncodified patterns lacking clear counterparts in theory or language. As an extension, we show such concepts can be used to steer model generations. Beyond improving model transparency, our work provides an empirical tool for uncovering organizing principles that have eluded traditional methods of analysis and synthesis.
>
---
#### [replaced 008] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.16211v3](http://arxiv.org/pdf/2505.16211v3)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Zhe Wang; Shun Zhang; Xingjian Du; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Gelei Deng; Haoyang Li; Yiming Li; Xiaobin Zhuang; Tianlong Chen; Qingsong Wen; Tianwei Zhang; Yang Liu; Haibo Hu; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; Wenyuan Xu; XiaoFeng Wang; Wei Dong; Xinfeng Li
>
> **备注:** Technical Report
>
> **摘要:** Audio Large Language Models (ALLMs) have gained widespread adoption, yet their trustworthiness remains underexplored. Existing evaluation frameworks, designed primarily for text, fail to address unique vulnerabilities introduced by audio's acoustic properties. We identify significant trustworthiness risks in ALLMs arising from non-semantic acoustic cues, including timbre, accent, and background noise, which can manipulate model behavior. We propose AudioTrust, a comprehensive framework for systematic evaluation of ALLM trustworthiness across audio-specific risks. AudioTrust encompasses six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. The framework implements 26 distinct sub-tasks using a curated dataset of over 4,420 audio samples from real-world scenarios, including daily conversations, emergency calls, and voice assistant interactions. We conduct comprehensive evaluations across 18 experimental configurations using human-validated automated pipelines. Our evaluation of 14 state-of-the-art open-source and closed-source ALLMs reveals significant limitations when confronted with diverse high-risk audio scenarios, providing insights for secure deployment of audio models. Code and data are available at https://github.com/JusperLee/AudioTrust.
>
---
#### [replaced 009] Leveraging Mamba with Full-Face Vision for Audio-Visual Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.13624v2](http://arxiv.org/pdf/2508.13624v2)**

> **作者:** Rong Chao; Wenze Ren; You-Jin Li; Kuo-Hsuan Hung; Sung-Feng Huang; Szu-Wei Fu; Wen-Huang Cheng; Yu Tsao
>
> **备注:** Accepted to Interspeech 2025 Workshop
>
> **摘要:** Recent Mamba-based models have shown promise in speech enhancement by efficiently modeling long-range temporal dependencies. However, models like Speech Enhancement Mamba (SEMamba) remain limited to single-speaker scenarios and struggle in complex multi-speaker environments such as the cocktail party problem. To overcome this, we introduce AVSEMamba, an audio-visual speech enhancement model that integrates full-face visual cues with a Mamba-based temporal backbone. By leveraging spatiotemporal visual information, AVSEMamba enables more accurate extraction of target speech in challenging conditions. Evaluated on the AVSEC-4 Challenge development and blind test sets, AVSEMamba outperforms other monaural baselines in speech intelligibility (STOI), perceptual quality (PESQ), and non-intrusive quality (UTMOS), and achieves \textbf{1st place} on the monaural leaderboard.
>
---
#### [replaced 010] VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.23759v2](http://arxiv.org/pdf/2509.23759v2)**

> **作者:** Ting-Kang Wang; Yueh-Po Peng; Li Su; Vincent K. M. Cheung
>
> **摘要:** While automatic music transcription is well-established in music information retrieval, most models are limited to transcribing pitch and timing information from audio, and thus omit crucial expressive and instrument-specific nuances. One example is playing technique on the violin, which affords its distinct palette of timbres for maximal emotional impact. Here, we propose VioPTT (Violin Playing Technique-aware Transcription), a lightweight, end-to-end model that directly transcribes violin playing technique in addition to pitch onset and offset. Furthermore, we release MOSA-VPT, a novel, high-quality synthetic violin playing technique dataset to circumvent the need for manually labeled annotations. Leveraging this dataset, our model demonstrated strong generalization to real-world note-level violin technique recordings in addition to achieving state-of-the-art transcription performance. To our knowledge, VioPTT is the first to jointly combine violin transcription and playing technique prediction within a unified framework.
>
---
#### [replaced 011] Exploring the Impact of Data Quantity on ASR in Extremely Low-resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.08872v2](http://arxiv.org/pdf/2409.08872v2)**

> **作者:** Yao-Fei Cheng; Li-Wei Chen; Hung-Shin Lee; Hsin-Min Wang
>
> **备注:** Accepted to O-COCOSDA 2025
>
> **摘要:** This study investigates the efficacy of data augmentation techniques for low-resource automatic speech recognition (ASR), focusing on two endangered Austronesian languages, Amis and Seediq. Recognizing the potential of self-supervised learning (SSL) in low-resource settings, we explore the impact of data volume on the continued pre-training of SSL models. We propose a novel data-selection scheme leveraging a multilingual corpus to augment the limited target language data. This scheme utilizes a language classifier to extract utterance embeddings and employs one-class classifiers to identify utterances phonetically and phonologically proximate to the target languages. Utterances are ranked and selected based on their decision scores, ensuring the inclusion of highly relevant data in the SSL-ASR pipeline. Our experimental results demonstrate the effectiveness of this approach, yielding substantial improvements in ASR performance for both Amis and Seediq. These findings underscore the feasibility and promise of data augmentation through cross-lingual transfer learning for low-resource language ASR.
>
---
#### [replaced 012] A dataset and model for recognition of audiologically relevant environments for hearing aids: AHEAD-DS and YAMNet+
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.10360v2](http://arxiv.org/pdf/2508.10360v2)**

> **作者:** Henry Zhong; Jörg M. Buchholz; Julian Maclaren; Simon Carlile; Richard Lyon
>
> **摘要:** Scene recognition of audiologically relevant environments is important for hearing aids; however, it is challenging, in part because of the limitations of existing datasets. Datasets often lack public accessibility, completeness, or audiologically relevant labels, hindering systematic comparison of machine learning models. Deploying these models on resource-constrained edge devices presents another challenge. Our solution is two-fold: we leverage several open source datasets to create AHEAD-DS, a dataset designed for scene recognition of audiologically relevant environments, and introduce YAMNet+, a sound recognition model. AHEAD-DS aims to provide a standardised, publicly available dataset with consistent labels relevant to hearing aids, facilitating model comparison. YAMNet+ is designed for deployment on edge devices like smartphones connected to hearing devices, such as hearing aids and wireless earphones with hearing aid functionality; serving as a baseline model for sound-based scene recognition. YAMNet+ achieved a mean average precision of 0.83 and accuracy of 0.93 on the testing set of AHEAD-DS across fourteen categories of audiologically relevant environments. We found that applying transfer learning from the pretrained YAMNet model was essential. We demonstrated real-time sound-based scene recognition capabilities on edge devices by deploying YAMNet+ to an Android smartphone. Even with a Google Pixel 3 (a phone with modest specifications, released in 2018), the model processes audio with approximately 50ms of latency to load the model, and an approximate linear increase of 30ms per 1 second of audio. Our website and code https://github.com/Australian-Future-Hearing-Initiative .
>
---
#### [replaced 013] MeanFlowSE: One-Step Generative Speech Enhancement via MeanFlow
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.23299v2](http://arxiv.org/pdf/2509.23299v2)**

> **作者:** Yike Zhu; Boyi Kang; Ziqian Wang; Xingchen Li; Zihan Zhang; Wenjie Li; Longshuai Xiao; Wei Xue; Lei Xie
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech enhancement (SE) recovers clean speech from noisy signals and is vital for applications such as telecommunications and automatic speech recognition (ASR). While generative approaches achieve strong perceptual quality, they often rely on multi-step sampling (diffusion/flow-matching) or large language models, limiting real-time deployment. To mitigate these constraints, we present MeanFlowSE, a one-step generative SE framework. It adopts MeanFlow to predict an average-velocity field for one-step latent refinement and conditions the model on self-supervised learning (SSL) representations rather than VAE latents. This design accelerates inference and provides robust acoustic-semantic guidance during training. In the Interspeech 2020 DNS Challenge blind test set and simulated test set, MeanFlowSE attains state-of-the-art (SOTA) level perceptual quality and competitive intelligibility while significantly lowering both real-time factor (RTF) and model size compared with recent generative competitors, making it suitable for practical use. The code will be released upon publication at https://github.com/Hello3orld/MeanFlowSE.
>
---
#### [replaced 014] From Voice to Safety: Language AI Powered Pilot-ATC Communication Understanding for Airport Surface Movement Collision Risk Assessment
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.04974v2](http://arxiv.org/pdf/2503.04974v2)**

> **作者:** Yutian Pang; Andrew Paul Kendall; Alex Porcayo; Mariah Barsotti; Anahita Jain; John-Paul Clarke
>
> **摘要:** This work provides a feasible solution to the existing airport surface safety monitoring capabilities (i.e., Airport Surface Surveillance Capability (ASSC)), namely language AI-based voice communication understanding for collision risk assessment. The proposed framework consists of two major parts, (a) rule-enhanced Named Entity Recognition (NER); (b) surface collision risk modeling. NER module generates information tables by processing voice communication transcripts, which serve as references for producing potential taxi plans and calculating the surface movement collision risk. We first collect and annotate our dataset based on open-sourced video recordings and safety investigation reports. Additionally, we refer to FAA Order JO 7110.65W and FAA Order JO 7340.2N to get the list of heuristic rules and phase contractions of communication between the pilot and the Air Traffic Controller (ATCo). Then, we propose the novel ATC Rule-Enhanced NER method, which integrates the heuristic rules into the model training and inference stages, resulting in a hybrid rule-based NER model. We show the effectiveness of this hybrid approach by comparing different setups with different token-level embedding models. For the risk modeling, we adopt the node-link airport layout graph from NASA FACET and model the aircraft taxi speed at each link as a log-normal distribution and derive the total taxi time distribution. Then, we propose a spatiotemporal formulation of the risk probability of two aircraft moving across potential collision nodes during ground movement. Furthermore, we propose the real-time implementation of such a method to obtain the lead time, with a comparison with a Petri-Net based method.
>
---
#### [replaced 015] StereoFoley: Object-Aware Stereo Audio Generation from Video
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.18272v2](http://arxiv.org/pdf/2509.18272v2)**

> **作者:** Tornike Karchkhadze; Kuan-Lin Chen; Mojtaba; Heydari; Robert Henzel; Alessandro Toso; Mehrez Souden; Joshua Atkins
>
> **摘要:** We present StereoFoley, a video-to-audio generation framework that produces semantically aligned, temporally synchronized, and spatially accurate stereo sound at 48 kHz. While recent generative video-to-audio models achieve strong semantic and temporal fidelity, they largely remain limited to mono or fail to deliver object-aware stereo imaging, constrained by the lack of professionally mixed, spatially accurate video-to-audio datasets. First, we develop and train a base model that generates stereo audio from video, achieving state-of-the-art in both semantic accuracy and synchronization. Next, to overcome dataset limitations, we introduce a synthetic data generation pipeline that combines video analysis, object tracking, and audio synthesis with dynamic panning and distance-based loudness controls, enabling spatially accurate object-aware sound. Finally, we fine-tune the base model on this synthetic dataset, yielding clear object-audio correspondence. Since no established metrics exist, we introduce stereo object-awareness measures and validate it through a human listening study, showing strong correlation with perception. This work establishes the first end-to-end framework for stereo object-aware video-to-audio generation, addressing a critical gap and setting a new benchmark in the field.
>
---
