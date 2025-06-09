# 音频 cs.SD;  eess.SP

- **最新发布 8 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Label-Context-Dependent Internal Language Model Estimation for CTC
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决CTC模型隐含学习的上下文相关内部语言模型（ILM）估计问题。通过知识蒸馏和正则化方法，提出新的上下文相关ILM估计方法，并在跨领域场景下验证其优于传统上下文无关先验。**

- **链接: [http://arxiv.org/pdf/2506.06096v1](http://arxiv.org/pdf/2506.06096v1)**

> **作者:** Zijian Yang; Minh-Nghia Phan; Ralf Schlüter; Hermann Ney
>
> **备注:** accepted to Interspeech 2025
>
> **摘要:** Although connectionist temporal classification (CTC) has the label context independence assumption, it can still implicitly learn a context-dependent internal language model (ILM) due to modern powerful encoders. In this work, we investigate the implicit context dependency modeled in the ILM of CTC. To this end, we propose novel context-dependent ILM estimation methods for CTC based on knowledge distillation (KD) with theoretical justifications. Furthermore, we introduce two regularization methods for KD. We conduct experiments on Librispeech and TED-LIUM Release 2 datasets for in-domain and cross-domain evaluation, respectively. Experimental results show that context-dependent ILMs outperform the context-independent priors in cross-domain evaluation, indicating that CTC learns a context-dependent ILM. The proposed label-level KD with smoothing method surpasses other ILM estimation approaches, with more than 13% relative improvement in word error rate compared to shallow fusion.
>
---
#### [new 002] Voice Impression Control in Zero-Shot TTS
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于零样本语音合成任务，旨在解决如何通过调节副语言信息控制生成语音的听感印象问题。作者提出了一种低维向量表示方法，用于调控多种声音印象对，并利用大语言模型生成目标印象描述对应的向量，实现了无需手动优化的印象控制。**

- **链接: [http://arxiv.org/pdf/2506.05688v1](http://arxiv.org/pdf/2506.05688v1)**

> **作者:** Keinichi Fujita; Shota Horiguchi; Yusuke Ijima
>
> **备注:** 5 pages,5 figures, Accepted to INTERSPEECH 2025
>
> **摘要:** Para-/non-linguistic information in speech is pivotal in shaping the listeners' impression. Although zero-shot text-to-speech (TTS) has achieved high speaker fidelity, modulating subtle para-/non-linguistic information to control perceived voice characteristics, i.e., impressions, remains challenging. We have therefore developed a voice impression control method in zero-shot TTS that utilizes a low-dimensional vector to represent the intensities of various voice impression pairs (e.g., dark-bright). The results of both objective and subjective evaluations have demonstrated our method's effectiveness in impression control. Furthermore, generating this vector via a large language model enables target-impression generation from a natural language description of the desired impression, thus eliminating the need for manual optimization.
>
---
#### [new 003] NAT: Neural Acoustic Transfer for Interactive Scenes in Real Time
- **分类: cs.SD; cs.GR; eess.AS**

- **简介: 该论文属于实时交互场景下的声学建模任务，旨在解决动态复杂环境中声音传播与反馈的实时计算难题。现有方法依赖大量预计算且难以应对物体位置、材质等动态变化。论文提出Neural Acoustic Transfer（NAT），利用隐式神经表示编码声学传递特性，并通过快速边界元方法生成训练数据，实现毫秒级高效准确的声音场预测，适用于虚拟现实等交互应用。**

- **链接: [http://arxiv.org/pdf/2506.06190v1](http://arxiv.org/pdf/2506.06190v1)**

> **作者:** Xutong Jin; Bo Pang; Chenxi Xu; Xinyun Hou; Guoping Wang; Sheng Li
>
> **摘要:** Previous acoustic transfer methods rely on extensive precomputation and storage of data to enable real-time interaction and auditory feedback. However, these methods struggle with complex scenes, especially when dynamic changes in object position, material, and size significantly alter sound effects. These continuous variations lead to fluctuating acoustic transfer distributions, making it challenging to represent with basic data structures and render efficiently in real time. To address this challenge, we present Neural Acoustic Transfer, a novel approach that utilizes an implicit neural representation to encode precomputed acoustic transfer and its variations, allowing for real-time prediction of sound fields under varying conditions. To efficiently generate the training data required for the neural acoustic field, we developed a fast Monte-Carlo-based boundary element method (BEM) approximation for general scenarios with smooth Neumann conditions. Additionally, we implemented a GPU-accelerated version of standard BEM for scenarios requiring higher precision. These methods provide the necessary training data, enabling our neural network to accurately model the sound radiation space. We demonstrate our method's numerical accuracy and runtime efficiency (within several milliseconds for 30s audio) through comprehensive validation and comparisons in diverse acoustic transfer scenarios. Our approach allows for efficient and accurate modeling of sound behavior in dynamically changing environments, which can benefit a wide range of interactive applications such as virtual reality, augmented reality, and advanced audio production.
>
---
#### [new 004] WAKE: Watermarking Audio with Key Enrichment
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频水印任务，旨在解决音频生成中的安全与版权保护问题。现有方法存在无法防止未授权访问、多次嵌入后难以解码初始水印及不支持变长水印等问题。论文提出WAKE框架，通过密钥控制实现安全嵌入与解码，支持多次嵌入与变长水印，提升了音频质量和检测准确率。**

- **链接: [http://arxiv.org/pdf/2506.05891v1](http://arxiv.org/pdf/2506.05891v1)**

> **作者:** Yaoxun Xu; Jianwei Yu; Hangting Chen; Zhiyong Wu; Xixin Wu; Dong Yu; Rongzhi Gu; Yi Luo
>
> **备注:** Accepted by InterSpeech2025
>
> **摘要:** As deep learning advances in audio generation, challenges in audio security and copyright protection highlight the need for robust audio watermarking. Recent neural network-based methods have made progress but still face three main issues: preventing unauthorized access, decoding initial watermarks after multiple embeddings, and embedding varying lengths of watermarks. To address these issues, we propose WAKE, the first key-controllable audio watermark framework. WAKE embeds watermarks using specific keys and recovers them with corresponding keys, enhancing security by making incorrect key decoding impossible. It also resolves the overwriting issue by allowing watermark decoding after multiple embeddings and supports variable-length watermark insertion. WAKE outperforms existing models in both watermarked audio quality and watermark detection accuracy. Code, more results, and demo page: https://thuhcsi.github.io/WAKE.
>
---
#### [new 005] Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音说话人日志（speaker diarization）任务，旨在解决多人对话中识别和分割不同说话人片段的问题。论文通过引入说话人属性吸引子和使用Conformer建模局部依赖，改进了端到端的神经网络方法，在CALLHOME数据集上取得了更好的性能。**

- **链接: [http://arxiv.org/pdf/2506.05593v1](http://arxiv.org/pdf/2506.05593v1)**

> **作者:** David Palzer; Matthew Maciejewski; Eric Fosler-Lussier
>
> **备注:** ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 11911-11915
>
> **摘要:** In recent years, end-to-end approaches have made notable progress in addressing the challenge of speaker diarization, which involves segmenting and identifying speakers in multi-talker recordings. One such approach, Encoder-Decoder Attractors (EDA), has been proposed to handle variable speaker counts as well as better guide the network during training. In this study, we extend the attractor paradigm by moving beyond direct speaker modeling and instead focus on representing more detailed `speaker attributes' through a multi-stage process of intermediate representations. Additionally, we enhance the architecture by replacing transformers with conformers, a convolution-augmented transformer, to model local dependencies. Experiments demonstrate improved diarization performance on the CALLHOME dataset.
>
---
#### [new 006] WhisQ: Cross-Modal Representation Learning for Text-to-Music MOS Prediction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到音乐评估任务，旨在预测生成音乐的主观质量（MOS），包括整体音乐质量和文本对齐。论文提出WhisQ模型，结合音频和文本编码器，引入序列级共注意力机制和最优传输正则化，提升跨模态对齐与评价性能。实验表明其在相关数据集上显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2506.05899v1](http://arxiv.org/pdf/2506.05899v1)**

> **作者:** Jakaria Islam Emon; Kazi Tamanna Alam; Md. Abu Salek
>
> **备注:** 3 pages
>
> **摘要:** Mean Opinion Score (MOS) prediction for text to music systems requires evaluating both overall musical quality and text prompt alignment. This paper introduces WhisQ, a multimodal architecture that addresses this dual-assessment challenge through sequence level co-attention and optimal transport regularization. WhisQ employs the Whisper Base pretrained model for temporal audio encoding and Qwen 3, a 0.6B Small Language Model (SLM), for text encoding, with both maintaining sequence structure for fine grained cross-modal modeling. The architecture features specialized prediction pathways: OMQ is predicted from pooled audio embeddings, while TA leverages bidirectional sequence co-attention between audio and text. Sinkhorn optimal transport loss further enforce semantic alignment in the shared embedding space. On the MusicEval Track-1 dataset, WhisQ achieves substantial improvements over the baseline: 7% improvement in Spearman correlation for OMQ and 14% for TA. Ablation studies reveal that optimal transport regularization provides the largest performance gain (10% SRCC improvement), demonstrating the importance of explicit cross-modal alignment for text-to-music evaluation.
>
---
#### [new 007] DeepFake Doctor: Diagnosing and Treating Audio-Video Fake Detection
- **分类: cs.MM; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音视频深度伪造检测任务，旨在解决当前数据集不可靠、评估方法不足的问题。作者分析了关键问题，提出了新的评估协议，并设计了SIMBA模型进行实验，探索了音频捷径问题及其缓解策略，提升了现有数据集的评估效果。**

- **链接: [http://arxiv.org/pdf/2506.05851v1](http://arxiv.org/pdf/2506.05851v1)**

> **作者:** Marcel Klemt; Carlotta Segna; Anna Rohrbach
>
> **摘要:** Generative AI advances rapidly, allowing the creation of very realistic manipulated video and audio. This progress presents a significant security and ethical threat, as malicious users can exploit DeepFake techniques to spread misinformation. Recent DeepFake detection approaches explore the multimodal (audio-video) threat scenario. In particular, there is a lack of reproducibility and critical issues with existing datasets - such as the recently uncovered silence shortcut in the widely used FakeAVCeleb dataset. Considering the importance of this topic, we aim to gain a deeper understanding of the key issues affecting benchmarking in audio-video DeepFake detection. We examine these challenges through the lens of the three core benchmarking pillars: datasets, detection methods, and evaluation protocols. To address these issues, we spotlight the recent DeepSpeak v1 dataset and are the first to propose an evaluation protocol and benchmark it using SOTA models. We introduce SImple Multimodal BAseline (SIMBA), a competitive yet minimalistic approach that enables the exploration of diverse design choices. We also deepen insights into the issue of audio shortcuts and present a promising mitigation strategy. Finally, we analyze and enhance the evaluation scheme on the widely used FakeAVCeleb dataset. Our findings offer a way forward in the complex area of audio-video DeepFake detection.
>
---
#### [new 008] SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态3D空间推理任务，旨在解决现有AV-LLMs在动态音频-视觉环境中缺乏3D空间推理能力的问题。作者提出了SAVVY-Bench基准和SAVVY方法，通过空间轨迹估计与全局地图构建，实现动态3D空间理解，并显著提升现有模型的表现。**

- **链接: [http://arxiv.org/pdf/2506.05414v1](http://arxiv.org/pdf/2506.05414v1)**

> **作者:** Mingfei Chen; Zijun Cui; Xiulong Liu; Jinlin Xiang; Caleb Zheng; Jingyuan Li; Eli Shlizerman
>
> **备注:** Project website with demo videos: https://zijuncui02.github.io/SAVVY/
>
> **摘要:** 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
>
---
## 更新

#### [replaced 001] Distributed Expectation Propagation for Multi-Object Tracking over Sensor Networks
- **分类: eess.SP; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18795v2](http://arxiv.org/pdf/2505.18795v2)**

> **作者:** Qing Li; Runze Gan; James R. Hopgood; Michael E. Davies; Simon J. Godsill
>
> **摘要:** In this paper, we present a novel distributed expectation propagation algorithm for multiple sensors, multiple objects tracking in cluttered environments. The proposed framework enables each sensor to operate locally while collaboratively exchanging moment estimates with other sensors, thus eliminating the need to transmit all data to a central processing node. Specifically, we introduce a fast and parallelisable Rao-Blackwellised Gibbs sampling scheme to approximate the tilted distributions, which enhances the accuracy and efficiency of expectation propagation updates. Results demonstrate that the proposed algorithm improves both communication and inference efficiency for multi-object tracking tasks with dynamic sensor connectivity and varying clutter levels.
>
---
#### [replaced 002] DualSpec: Text-to-spatial-audio Generation via Dual-Spectrogram Guided Diffusion Model
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.18952v2](http://arxiv.org/pdf/2502.18952v2)**

> **作者:** Lei Zhao; Sizhou Chen; Linfeng Feng; Jichao Zhang; Xiao-Lei Zhang; Chi Zhang; Xuelong Li
>
> **摘要:** Text-to-audio (TTA), which generates audio signals from textual descriptions, has received huge attention in recent years. However, recent works focused on text to monaural audio only. As we know, spatial audio provides more immersive auditory experience than monaural audio, e.g. in virtual reality. To address this issue, we propose a text-to-spatial-audio (TTSA) generation framework named DualSpec. Specifically, it first trains variational autoencoders (VAEs) for extracting the latent acoustic representations from sound event audio. Then, given text that describes sound events and event directions, the proposed method uses the encoder of a pretrained large language model to transform the text into text features. Finally, it trains a diffusion model from the latent acoustic representations and text features for the spatial audio generation. In the inference stage, only the text description is needed to generate spatial audio. Particularly, to improve the synthesis quality and azimuth accuracy of the spatial sound events simultaneously, we propose to use two kinds of acoustic features. One is the Mel spectrograms which is good for improving the synthesis quality, and the other is the short-time Fourier transform spectrograms which is good at improving the azimuth accuracy. We provide a pipeline of constructing spatial audio dataset with text prompts, for the training of the VAEs and diffusion model. We also introduce new spatial-aware evaluation metrics to quantify the azimuth errors of the generated spatial audio recordings. Experimental results demonstrate that the proposed method can generate spatial audio with high directional and event consistency.
>
---
#### [replaced 003] Quality Assessment of Noisy and Enhanced Speech with Limited Data: UWB-NTIS System for VoiceMOS 2024 and Beyond
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.00506v2](http://arxiv.org/pdf/2506.00506v2)**

> **作者:** Marie Kunešová
>
> **备注:** This is a preliminary write-up of our initial work, posted as an early version preprint for cross-referencing purposes. We intend to further extend this research and submit it for publication at a conference, at which point this preprint will be updated with the full text. v2 changes: Fixed CHiME 7 - UDASE dataset overlapping with VMC 2024 training data
>
> **摘要:** In this preprint, we present the UWB-NTIS-TTS team's submission to Track 3 of the VoiceMOS 2024 Challenge, the goal of which was to automatically assess the speech quality of noisy and de-noised speech in terms of the ITU-T P.835 metrics of "SIG", "BAK", and "OVRL". Our proposed system, based on wav2vec 2.0, placed among the top systems in the challenge, achieving the best prediction of the BAK scores (background noise intrusiveness), the second-best prediction of the OVRL score (overall audio quality), and the third-best prediction of SIG (speech signal quality) out of the five participating systems. We describe our approach, such as the two-stage fine-tuning process we used to contend with the challenge's very limiting restrictions on allowable training data, and present the results achieved both on the VoiceMOS 2024 Challenge data and on the recently released CHiME 7 - UDASE dataset.
>
---
#### [replaced 004] Efficient Fine-Grained Guidance for Diffusion Model Based Symbolic Music Generation
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.08435v3](http://arxiv.org/pdf/2410.08435v3)**

> **作者:** Tingyu Zhu; Haoyu Liu; Ziyu Wang; Zhimin Jiang; Zeyu Zheng
>
> **摘要:** Developing generative models to create or conditionally create symbolic music presents unique challenges due to the combination of limited data availability and the need for high precision in note pitch. To address these challenges, we introduce an efficient Fine-Grained Guidance (FGG) approach within diffusion models. FGG guides the diffusion models to generate music that aligns more closely with the control and intent of expert composers, which is critical to improve the accuracy, listenability, and quality of generated music. This approach empowers diffusion models to excel in advanced applications such as improvisation, and interactive music creation. We derive theoretical characterizations for both the challenges in symbolic music generation and the effects of the FGG approach. We provide numerical experiments and subjective evaluation to demonstrate the effectiveness of our approach. We have published a demo page to showcase performances, which enables real-time interactive generation.
>
---
#### [replaced 005] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04113v2](http://arxiv.org/pdf/2505.04113v2)**

> **作者:** Xueyao Zhang; Yuancheng Wang; Chaoren Wang; Ziniu Li; Zhuo Chen; Zhizheng Wu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Modern zero-shot text-to-speech (TTS) systems, despite using extensive pre-training, often struggle in challenging scenarios such as tongue twisters, repeated words, code-switching, and cross-lingual synthesis, leading to intelligibility issues. To address these limitations, this paper leverages preference alignment techniques, which enable targeted construction of out-of-pretraining-distribution data to enhance performance. We introduce a new dataset, named the Intelligibility Preference Speech Dataset (INTP), and extend the Direct Preference Optimization (DPO) framework to accommodate diverse TTS architectures. After INTP alignment, in addition to intelligibility, we observe overall improvements including naturalness, similarity, and audio quality for multiple TTS models across diverse domains. Based on that, we also verify the weak-to-strong generalization ability of INTP for more intelligible models such as CosyVoice 2 and Ints. Moreover, we showcase the potential for further improvements through iterative alignment based on Ints. Audio samples are available at https://intalign.github.io/.
>
---
#### [replaced 006] Audiobox TTA-RAG: Improving Zero-Shot and Few-Shot Text-To-Audio with Retrieval-Augmented Generation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2411.05141v2](http://arxiv.org/pdf/2411.05141v2)**

> **作者:** Mu Yang; Bowen Shi; Matthew Le; Wei-Ning Hsu; Andros Tjandra
>
> **备注:** Interspeech 2025
>
> **摘要:** This work focuses on improving Text-To-Audio (TTA) generation on zero-shot and few-shot settings (i.e. generating unseen or uncommon audio events). Inspired by the success of Retrieval-Augmented Generation (RAG) in Large Language Models, we propose Audiobox TTA-RAG, a novel retrieval-augmented TTA approach based on Audiobox, a flow-matching audio generation model. Unlike the vanilla Audiobox TTA solution that generates audio conditioned on text only, we extend the TTA process by augmenting the conditioning input with both text and retrieved audio samples. Our retrieval method does not require the external database to have labeled audio, offering more practical use cases. We show that the proposed model can effectively leverage the retrieved audio samples and significantly improve zero-shot and few-shot TTA performance, with large margins on multiple evaluation metrics, while maintaining the ability to generate semantically aligned audio for the in-domain setting.
>
---
#### [replaced 007] Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15670v2](http://arxiv.org/pdf/2505.15670v2)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
#### [replaced 008] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.14842v2](http://arxiv.org/pdf/2411.14842v2)**

> **作者:** Wanqi Yang; Yanda Li; Meng Fang; Yunchao Wei; Ling Chen
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Adversarial audio attacks pose a significant threat to the growing use of large audio-language models (LALMs) in voice-based human-machine interactions. While existing research focused on model-specific adversarial methods, real-world applications demand a more generalizable and universal approach to audio adversarial attacks. In this paper, we introduce the Chat-Audio Attacks (CAA) benchmark including four distinct types of audio attacks, which aims to explore the vulnerabilities of LALMs to these audio attacks in conversational scenarios. To evaluate the robustness of LALMs, we propose three evaluation strategies: Standard Evaluation, utilizing traditional metrics to quantify model performance under attacks; GPT-4o-Based Evaluation, which simulates real-world conversational complexities; and Human Evaluation, offering insights into user perception and trust. We evaluate six state-of-the-art LALMs with voice interaction capabilities, including Gemini-1.5-Pro, GPT-4o, and others, using three distinct evaluation methods on the CAA benchmark. Our comprehensive analysis reveals the impact of four types of audio attacks on the performance of these models, demonstrating that GPT-4o exhibits the highest level of resilience. Our data can be accessed via the following link: \href{https://github.com/crystraldo/CAA}{CAA}.
>
---
#### [replaced 009] Can Masked Autoencoders Also Listen to Birds?
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.12880v3](http://arxiv.org/pdf/2504.12880v3)**

> **作者:** Lukas Rauch; René Heinrich; Ilyass Moummad; Alexis Joly; Bernhard Sick; Christoph Scholz
>
> **备注:** under review @TMLR
>
> **摘要:** Masked Autoencoders (MAEs) have shown competitive results in audio classification by learning rich semantic representations through an efficient self-supervised reconstruction task. However, general-purpose models fail to generalize well when applied directly to fine-grained audio domains. Specifically, bird-sound classification requires distinguishing subtle inter-species differences and managing high intra-species acoustic variability, thereby revealing the performance limitations of general-domain Audio-MAE models. This work demonstrates that bridging this domain gap requires more than domain-specific pretraining data; adapting the entire training pipeline is crucial. We systematically revisit and adapt the pretraining recipe, fine-tuning methods, and frozen feature utilization to bird sounds using BirdSet, a large-scale bioacoustic dataset comparable to AudioSet. Our resulting Bird-MAE achieves new state-of-the-art results in BirdSet's multi-label classification benchmark. Additionally, we introduce the parameter-efficient prototypical probing, enhancing the utility of frozen MAE representations and closely approaching fine-tuning performance in low-resource settings. Bird-MAE's prototypical probes outperform linear probing by up to 37%$_\text{p}$ in MAP and narrow the gap to fine-tuning to approximately 3.3%$_\text{p}$ on average across BirdSet downstream tasks. Bird-MAE also demonstrates robust few-shot capabilities with prototypical probing in our newly established few-shot benchmark on BirdSet, highlighting the potential of tailored self-supervised learning pipelines for fine-grained audio domains.
>
---
#### [replaced 010] MARS: Radio Map Super-resolution and Reconstruction Method under Sparse Channel Measurements
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.04682v2](http://arxiv.org/pdf/2506.04682v2)**

> **作者:** Chuyun Deng; Na Liu; Wei Xie; Lianming Xu; Li Wang
>
> **摘要:** Radio maps reflect the spatial distribution of signal strength and are essential for applications like smart cities, IoT, and wireless network planning. However, reconstructing accurate radio maps from sparse measurements remains challenging. Traditional interpolation and inpainting methods lack environmental awareness, while many deep learning approaches depend on detailed scene data, limiting generalization. To address this, we propose MARS, a Multi-scale Aware Radiomap Super-resolution method that combines CNNs and Transformers with multi-scale feature fusion and residual connections. MARS focuses on both global and local feature extraction, enhancing feature representation across different receptive fields and improving reconstruction accuracy. Experiments across different scenes and antenna locations show that MARS outperforms baseline models in both MSE and SSIM, while maintaining low computational cost, demonstrating strong practical potential.
>
---
#### [replaced 011] Multi-Channel Acoustic Echo Cancellation Based on Direction-of-Arrival Estimation
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.19493v2](http://arxiv.org/pdf/2505.19493v2)**

> **作者:** Fei Zhao; Xueliang Zhang; Zhong-Qiu Wang
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Acoustic echo cancellation (AEC) is an important speech signal processing technology that can remove echoes from microphone signals to enable natural-sounding full-duplex speech communication. While single-channel AEC is widely adopted, multi-channel AEC can leverage spatial cues afforded by multiple microphones to achieve better performance. Existing multi-channel AEC approaches typically combine beamforming with deep neural networks (DNN). This work proposes a two-stage algorithm that enhances multi-channel AEC by incorporating sound source directional cues. Specifically, a lightweight DNN is first trained to predict the sound source directions, and then the predicted directional information, multi-channel microphone signals, and single-channel far-end signal are jointly fed into an AEC network to estimate the near-end signal. Evaluation results show that the proposed algorithm outperforms baseline approaches and exhibits robust generalization across diverse acoustic environments.
>
---
#### [replaced 012] WER We Stand: Benchmarking Urdu ASR Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.11252v3](http://arxiv.org/pdf/2409.11252v3)**

> **作者:** Samee Arif; Sualeha Farid; Aamina Jamal Khan; Mustafa Abbas; Agha Ali Raza; Awais Athar
>
> **摘要:** This paper presents a comprehensive evaluation of Urdu Automatic Speech Recognition (ASR) models. We analyze the performance of three ASR model families: Whisper, MMS, and Seamless-M4T using Word Error Rate (WER), along with a detailed examination of the most frequent wrong words and error types including insertions, deletions, and substitutions. Our analysis is conducted using two types of datasets, read speech and conversational speech. Notably, we present the first conversational speech dataset designed for benchmarking Urdu ASR models. We find that seamless-large outperforms other ASR models on the read speech dataset, while whisper-large performs best on the conversational speech dataset. Furthermore, this evaluation highlights the complexities of assessing ASR models for low-resource languages like Urdu using quantitative metrics alone and emphasizes the need for a robust Urdu text normalization system. Our findings contribute valuable insights for developing robust ASR systems for low-resource languages like Urdu.
>
---
