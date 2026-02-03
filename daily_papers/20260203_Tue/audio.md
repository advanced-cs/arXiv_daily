# 音频 cs.SD;  eess.AS

- **最新发布 36 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation
- **分类: cs.SD**

- **简介: 该论文提出ACE-Step 1.5，一个高效开源音乐生成模型，解决在消费级硬件上实现高质量音乐生成的问题。通过混合架构与内在强化学习，提升生成效率与风格控制能力。**

- **链接: [https://arxiv.org/pdf/2602.00744v1](https://arxiv.org/pdf/2602.00744v1)**

> **作者:** Junmin Gong; Yulin Song; Wenxiao Zhao; Sen Wang; Shengyuan Xu; Jing Guo
>
> **摘要:** We present ACE-Step v1.5, a highly efficient open-source music foundation model that brings commercial-grade generation to consumer hardware. On commonly used evaluation metrics, ACE-Step v1.5 achieves quality beyond most commercial music models while remaining extremely fast -- under 2 seconds per full song on an A100 and under 10 seconds on an RTX 3090. The model runs locally with less than 4GB of VRAM, and supports lightweight personalization: users can train a LoRA from just a few songs to capture their own style. At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints -- scaling from short loops to 10-minute compositions -- while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences. Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities -- such as cover generation, repainting, and vocal-to-BGM conversion -- while maintaining strict adherence to prompts across 50+ languages. This paves the way for powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. The code, the model weights and the demo are available at: https://ace-step.github.io/ace-step-v1.5.github.io/
>
---
#### [new 002] Audio-to-Image Bird Species Retrieval without Audio-Image Pairs via Text Distillation
- **分类: cs.SD; cs.IR; cs.LG**

- **简介: 该论文属于音频到图像的物种检索任务，解决缺乏配对数据的问题。通过文本中介将视觉语义迁移到音频模型，实现音频与图像的对齐。**

- **链接: [https://arxiv.org/pdf/2602.00681v1](https://arxiv.org/pdf/2602.00681v1)**

> **作者:** Ilyass Moummad; Marius Miron; Lukas Rauch; David Robinson; Alexis Joly; Olivier Pietquin; Emmanuel Chemla; Matthieu Geist
>
> **摘要:** Audio-to-image retrieval offers an interpretable alternative to audio-only classification for bioacoustic species recognition, but learning aligned audio-image representations is challenging due to the scarcity of paired audio-image data. We propose a simple and data-efficient approach that enables audio-to-image retrieval without any audio-image supervision. Our proposed method uses text as a semantic intermediary: we distill the text embedding space of a pretrained image-text model (BioCLIP-2), which encodes rich visual and taxonomic structure, into a pretrained audio-text model (BioLingual) by fine-tuning its audio encoder with a contrastive objective. This distillation transfers visually grounded semantics into the audio representation, inducing emergent alignment between audio and image embeddings without using images during training. We evaluate the resulting model on multiple bioacoustic benchmarks. The distilled audio encoder preserves audio discriminative power while substantially improving audio-text alignment on focal recordings and soundscape datasets. Most importantly, on the SSW60 benchmark, the proposed approach achieves strong audio-to-image retrieval performance exceeding baselines based on zero-shot model combinations or learned mappings between text embeddings, despite not training on paired audio-image data. These results demonstrate that indirect semantic transfer through text is sufficient to induce meaningful audio-image alignment, providing a practical solution for visually grounded species recognition in data-scarce bioacoustic settings.
>
---
#### [new 003] Multi-Speaker Conversational Audio Deepfake: Taxonomy, Dataset and Pilot Study
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决多说话人对话场景下的深度伪造检测问题。提出一种分类体系并构建数据集，进行初步检测模型评估。**

- **链接: [https://arxiv.org/pdf/2602.00295v1](https://arxiv.org/pdf/2602.00295v1)**

> **作者:** Alabi Ahmed; Vandana Janeja; Sanjay Purushotham
>
> **备注:** This work was presented at the 2025 IEEE International Conference on Data Mining, ICDM 2025, November 12-15,2025, Washington DC, USA
>
> **摘要:** The rapid advances in text-to-speech (TTS) technologies have made audio deepfakes increasingly realistic and accessible, raising significant security and trust concerns. While existing research has largely focused on detecting single-speaker audio deepfakes, real-world malicious applications with multi-speaker conversational settings is also emerging as a major underexplored threat. To address this gap, we propose a conceptual taxonomy of multi-speaker conversational audio deepfakes, distinguishing between partial manipulations (one or multiple speakers altered) and full manipulations (entire conversations synthesized). As a first step, we introduce a new Multi-speaker Conversational Audio Deepfakes Dataset (MsCADD) of 2,830 audio clips containing real and fully synthetic two-speaker conversations, generated using VITS and SoundStorm-based NotebookLM models to simulate natural dialogue with variations in speaker gender, and conversational spontaneity. MsCADD is limited to text-to-speech (TTS) types of deepfake. We benchmark three neural baseline models; LFCC-LCNN, RawNet2, and Wav2Vec 2.0 on this dataset and report performance in terms of F1 score, accuracy, true positive rate (TPR), and true negative rate (TNR). Results show that these baseline models provided a useful benchmark, however, the results also highlight that there is a significant gap in multi-speaker deepfake research in reliably detecting synthetic voices under varied conversational dynamics. Our dataset and benchmarks provide a foundation for future research on deepfake detection in conversational scenarios, which is a highly underexplored area of research but also a major area of threat to trustworthy information in audio settings. The MsCADD dataset is publicly available to support reproducibility and benchmarking by the research community.
>
---
#### [new 004] RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于语音克隆任务，旨在评估模型在真实场景下的鲁棒性。通过构建RVCBench基准，分析输入变化、后处理等对性能的影响，揭示现有模型的不足。**

- **链接: [https://arxiv.org/pdf/2602.00443v1](https://arxiv.org/pdf/2602.00443v1)**

> **作者:** Xinting Liao; Ruinan Jin; Hanlin Yu; Deval Pandya; Xiaoxiao Li
>
> **备注:** 40 pages, 12figures
>
> **摘要:** Modern voice cloning (VC) can synthesize speech that closely matches a target speaker from only seconds of reference audio, enabling applications such as personalized speech interfaces and dubbing. In practical deployments, modern audio generation models inevitably encounter noisy reference audios, imperfect text prompts, and diverse downstream processing, which can significantly hurt robustness. Despite rapid progress in VC driven by autoregressive codec-token language models and diffusion-based models, robustness under realistic deployment shifts remains underexplored. This paper introduces RVCBench, a comprehensive benchmark that evaluates Robustness in VC across the full generation pipeline, including input variation, generation challenges, output post-processing, and adversarial perturbations, covering 10 robustness tasks, 225 speakers, 14,370 utterances, and 11 representative modern VC models. Our evaluation uncovers substantial robustness gaps in VC: performance can deteriorate sharply under common input shifts and post-processing; long-context and cross-lingual scenarios further expose stability limitations; and both passive noise and proactive perturbation influence generation robustness. Collectively, these findings provide a unified picture of how current VC models fail in practice and introduce a standardized, open-source testbed to support the development of more robust and deployable VC models. We open-source our project at https://github.com/Nanboy-Ronan/RVCBench.
>
---
#### [new 005] High-Fidelity Generative Audio Compression at 0.275kbps
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频压缩任务，旨在解决超低比特率下音频质量下降的问题。提出GAC方法，通过语义理解和生成合成实现高效压缩，显著提升压缩比和音频质量。**

- **链接: [https://arxiv.org/pdf/2602.00648v1](https://arxiv.org/pdf/2602.00648v1)**

> **作者:** Hao Ma; Ruihao Jing; Shansong Liu; Cheng Gong; Chi Zhang; Xiao-Lei Zhang; Xuelong Li
>
> **备注:** Technical Report
>
> **摘要:** High-fidelity general audio compression at ultra-low bitrates is crucial for applications ranging from low-bandwidth communication to generative audio-language modeling. Traditional audio compression methods and contemporary neural codecs are fundamentally designed for waveform reconstruction. As a result, when operating at ultra-low bitrates, these methods degrade rapidly and often fail to preserve essential information, leading to severe acoustic artifacts and pronounced semantic distortion. To overcome these limitations, we introduce Generative Audio Compression (GAC), a novel paradigm shift from signal fidelity to task-oriented effectiveness. Implemented within the AI Flow framework, GAC is theoretically grounded in the Law of Information Capacity. These foundations posit that abundant computational power can be leveraged at the receiver to offset extreme communication bottlenecks--exemplifying the More Computation, Less Bandwidth philosophy. By integrating semantic understanding at the transmitter with scalable generative synthesis at the receiver, GAC offloads the information burden to powerful model priors. Our 1.8B-parameter model achieves high-fidelity reconstruction of 32kHz general audio at an unprecedented bitrate of 0.275kbps. Even at 0.175kbps, it still preserves a strong intelligible audio transmission capability, which represents an about 3000x compression ratio, significantly outperforming current state-of-the-art neural codecs in maintaining both perceptual quality and semantic consistency.
>
---
#### [new 006] Masked Autoencoders as Universal Speech Enhancer
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音增强任务，旨在解决缺乏干净语音数据的问题。提出一种基于掩码自编码器的通用语音增强方法，实现自监督预训练并适应多种噪声和混响场景。**

- **链接: [https://arxiv.org/pdf/2602.02413v1](https://arxiv.org/pdf/2602.02413v1)**

> **作者:** Rajalaxmi Rajagopalan; Ritwik Giri; Zhiqiang Tang; Kyu Han
>
> **摘要:** Supervised speech enhancement methods have been very successful. However, in practical scenarios, there is a lack of clean speech, and self-supervised learning-based (SSL) speech enhancement methods that offer comparable enhancement performance and can be applied to other speech-related downstream applications are desired. In this work, we develop a masked autoencoder based universal speech enhancer that is agnostic to the type of distortion affecting speech, can handle multiple distortions simultaneously, and is trained in a self-supervised manner. An augmentation stack adds further distortions to the noisy input data. The masked autoencoder model learns to remove the added distortions along with reconstructing the masked regions of the spectrogram during pre-training. The pre-trained embeddings are then used by fine-tuning models trained on a small amount of paired data for specific downstream tasks. We evaluate the pre-trained features for denoising and dereverberation downstream tasks. We explore different augmentations (like single or multi-speaker) in the pre-training augmentation stack and the effect of different noisy input feature representations (like $log1p$ compression) on pre-trained embeddings and downstream fine-tuning enhancement performance. We show that the proposed method not only outperforms the baseline but also achieves state-of-the-art performance for both in-domain and out-of-domain evaluation datasets.
>
---
#### [new 007] LipSody: Lip-to-Speech Synthesis with Enhanced Prosody Consistency
- **分类: cs.SD**

- **简介: 该论文属于唇语到语音合成任务，解决语音情感一致性不足的问题。提出LipSody框架，融合说话人身份、语言内容和情感信息，提升语音质量。**

- **链接: [https://arxiv.org/pdf/2602.01908v1](https://arxiv.org/pdf/2602.01908v1)**

> **作者:** Jaejun Lee; Yoori Oh; Kyogu Lee
>
> **备注:** This paper has been accepted to ICASSP 2026
>
> **摘要:** Lip-to-speech synthesis aims to generate speech audio directly from silent facial video by reconstructing linguistic content from lip movements, providing valuable applications in situations where audio signals are unavailable or degraded. While recent diffusion-based models such as LipVoicer have demonstrated impressive performance in reconstructing linguistic content, they often lack prosodic consistency. In this work, we propose LipSody, a lip-to-speech framework enhanced for prosody consistency. LipSody introduces a prosody-guiding strategy that leverages three complementary cues: speaker identity extracted from facial images, linguistic content derived from lip movements, and emotional context inferred from face video. Experimental results demonstrate that LipSody substantially improves prosody-related metrics, including global and local pitch deviations, energy consistency, and speaker similarity, compared to prior approaches.
>
---
#### [new 008] HierCon: Hierarchical Contrastive Attention for Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在识别合成语音。针对现有方法忽略层次依赖的问题，提出HierCon框架，通过层次注意力和对比学习提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.01032v1](https://arxiv.org/pdf/2602.01032v1)**

> **作者:** Zhili Nicholas Liang; Soyeon Caren Han; Qizhou Wang; Christopher Leckie
>
> **备注:** Proceedings of The Web Conference 2026 (WWW'26), short track
>
> **摘要:** Audio deepfakes generated by modern TTS and voice conversion systems are increasingly difficult to distinguish from real speech, raising serious risks for security and online trust. While state-of-the-art self-supervised models provide rich multi-layer representations, existing detectors treat layers independently and overlook temporal and hierarchical dependencies critical for identifying synthetic artefacts. We propose HierCon, a hierarchical layer attention framework combined with margin-based contrastive learning that models dependencies across temporal frames, neighbouring layers, and layer groups, while encouraging domain-invariant embeddings. Evaluated on ASVspoof 2021 DF and In-the-Wild datasets, our method achieves state-of-the-art performance (1.93% and 6.87% EER), improving over independent layer weighting by 36.6% and 22.5% respectively. The results and attention visualisations confirm that hierarchical modelling enhances generalisation to cross-domain generation techniques and recording conditions.
>
---
#### [new 009] ParaGSE: Parallel Generative Speech Enhancement with Group-Vector-Quantization-based Neural Speech Codec
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决现有方法复杂度高、效率低和语音质量不佳的问题。提出ParaGSE框架，利用GVQ编码器实现并行生成，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2602.01793v1](https://arxiv.org/pdf/2602.01793v1)**

> **作者:** Fei Liu; Yang Ai
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Recently, generative speech enhancement has garnered considerable interest; however, existing approaches are hindered by excessive complexity, limited efficiency, and suboptimal speech quality. To overcome these challenges, this paper proposes a novel parallel generative speech enhancement (ParaGSE) framework that leverages a group vector quantization (GVQ)-based neural speech codec. The GVQ-based codec adopts separate VQs to produce mutually independent tokens, enabling efficient parallel token prediction in ParaGSE. Specifically, ParaGSE leverages the GVQ-based codec to encode degraded speech into distinct tokens, predicts the corresponding clean tokens through parallel branches conditioned on degraded spectral features, and ultimately reconstructs clean speech via the codec decoder. Experimental results demonstrate that ParaGSE consistently produces superior enhanced speech compared to both discriminative and generative baselines, under a wide range of distortions including noise, reverberation, band-limiting, and their mixtures. Furthermore, empowered by parallel computation in token prediction, ParaGSE attains about a 1.5-fold improvement in generation efficiency on CPU compared with serial generative speech enhancement approaches.
>
---
#### [new 010] Membership Inference Attack Against Music Diffusion Models via Generative Manifold Perturbation
- **分类: cs.SD**

- **简介: 该论文属于隐私安全任务，旨在解决音乐扩散模型的成员推理攻击问题。通过分析生成过程的几何特性，提出LSA-Probe方法提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2602.01645v1](https://arxiv.org/pdf/2602.01645v1)**

> **作者:** Yuxuan Liu; Peihong Zhang; Rui Sang; Zhixin Li; Yizhou Tan; Yiqiang Cai; Shengchen Li
>
> **摘要:** Membership inference attacks (MIAs) test whether a specific audio clip was used to train a model, making them a key tool for auditing generative music models for copyright compliance. However, loss-based signals (e.g., reconstruction error) are weakly aligned with human perception in practice, yielding poor separability at the low false-positive rates (FPRs) required for forensics. We propose the Latent Stability Adversarial Probe (LSA-Probe), a white-box method that measures a geometric property of the reverse diffusion: the minimal time-normalized perturbation budget needed to cross a fixed perceptual degradation threshold at an intermediate diffusion state. We show that training members, residing in more stable regions, exhibit a significantly higher degradation cost.
>
---
#### [new 011] TLDiffGAN: A Latent Diffusion-GAN Framework with Temporal Information Fusion for Anomalous Sound Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于异常声音检测任务，旨在解决现有生成模型无法充分捕捉正常声音特征的问题。提出TLDiffGAN框架，结合扩散模型与音频编码器，提升异常检测性能。**

- **链接: [https://arxiv.org/pdf/2602.01060v1](https://arxiv.org/pdf/2602.01060v1)**

> **作者:** Chengyuan Ma; Peng Jia; Hongyue Guo; Wenming Yang
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Existing generative models for unsupervised anomalous sound detection are limited by their inability to fully capture the complex feature distribution of normal sounds, while the potential of powerful diffusion models in this domain remains largely unexplored. To address this challenge, we propose a novel framework, TLDiffGAN, which consists of two complementary branches. One branch incorporates a latent diffusion model into the GAN generator for adversarial training, thereby making the discriminator's task more challenging and improving the quality of generated samples. The other branch leverages pretrained audio model encoders to extract features directly from raw audio waveforms for auxiliary discrimination. This framework effectively captures feature representations of normal sounds from both raw audio and Mel spectrograms. Moreover, we introduce a TMixup spectrogram augmentation technique to enhance sensitivity to subtle and localized temporal patterns that are often overlooked. Extensive experiments on the DCASE 2020 Challenge Task 2 dataset demonstrate the superior detection performance of TLDiffGAN, as well as its strong capability in anomalous time-frequency localization.
>
---
#### [new 012] Causally Disentangled Contrastive Learning for Multilingual Speaker Embeddings
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究多语言说话人嵌入中的性别、年龄和口音信息泄露问题，提出两种去偏策略，评估其对验证性能的影响，揭示去偏与性能间的权衡。**

- **链接: [https://arxiv.org/pdf/2602.01363v1](https://arxiv.org/pdf/2602.01363v1)**

> **作者:** Mariëtte Olijslager; Seyed Sahand Mohammadi Ziabari; Ali Mohammed Mansoor Alsahag
>
> **摘要:** Self-supervised speaker embeddings are widely used in speaker verification systems, but prior work has shown that they often encode sensitive demographic attributes, raising fairness and privacy concerns. This paper investigates the extent to which demographic information, specifically gender, age, and accent, is present in SimCLR-trained speaker embeddings and whether such leakage can be mitigated without severely degrading speaker verification performance. We study two debiasing strategies: adversarial training through gradient reversal and a causal bottleneck architecture that explicitly separates demographic and residual information. Demographic leakage is quantified using both linear and nonlinear probing classifiers, while speaker verification performance is evaluated using ROC-AUC and EER. Our results show that gender information is strongly and linearly encoded in baseline embeddings, whereas age and accent are weaker and primarily nonlinearly represented. Adversarial debiasing reduces gender leakage but has limited effect on age and accent and introduces a clear trade-off with verification accuracy. The causal bottleneck further suppresses demographic information, particularly in the residual representation, but incurs substantial performance degradation. These findings highlight fundamental limitations in mitigating demographic leakage in self-supervised speaker embeddings and clarify the trade-offs inherent in current debiasing approaches.
>
---
#### [new 013] HuPER: A Human-Inspired Framework for Phonetic Perception
- **分类: eess.AS; cs.AI**

- **简介: 该论文提出HuPER框架，用于语音感知任务，解决语音识别中的错误率和跨语言适应问题。通过结合声学和语言知识，实现高效学习与多语言迁移。**

- **链接: [https://arxiv.org/pdf/2602.01634v1](https://arxiv.org/pdf/2602.01634v1)**

> **作者:** Chenxu Guo; Jiachen Lian; Yisi Liu; Baihe Huang; Shriyaa Narayanan; Cheol Jun Cho; Gopala Anumanchipalli
>
> **摘要:** We propose HuPER, a human-inspired framework that models phonetic perception as adaptive inference over acoustic-phonetics evidence and linguistic knowledge. With only 100 hours of training data, HuPER achieves state-of-the-art phonetic error rates on five English benchmarks and strong zero-shot transfer to 95 unseen languages. HuPER is also the first framework to enable adaptive, multi-path phonetic perception under diverse acoustic conditions. All training data, models, and code are open-sourced. Code and demo avaliable at https://github.com/HuPER29/HuPER.
>
---
#### [new 014] Attention-weighted Centered Kernel Alignment for Knowledge Distillation in Large Audio-Language Models Applied to Speech Emotion Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别任务，解决大模型压缩问题。提出PL-Distill框架，通过投影层和logits层知识蒸馏，提升小模型性能。**

- **链接: [https://arxiv.org/pdf/2602.01547v1](https://arxiv.org/pdf/2602.01547v1)**

> **作者:** Qingran Yang; Botao Zhao; Zuheng Kang; Xue Li; Yayun He; Chuhang Liu; Xulong Zhang; Xiaoyang Qu; Junqing Peng; Jianzong Wang
>
> **备注:** Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** The emergence of Large Audio-Language Models (LALMs) has advanced Speech Emotion Recognition (SER), but their size limits deployment in resource-constrained environments. While Knowledge Distillation is effective for LALM compression, existing methods remain underexplored in distilling the cross-modal projection module (Projector), and often struggle with alignment due to differences in feature dimensions. We propose PL-Distill, a KD framework that combines Projector-Level Distillation (PDist) to align audio embeddings and Logits-Level Distillation (LDist) to align output logits. PDist introduces Attention-weighted Centered Kernel Alignment, a novel approach we propose to highlight important time steps and address dimension mismatches. Meanwhile, LDist minimizes the Kullback-Leibler divergence between teacher and student logits from audio and text modalities. On IEMOCAP, RAVDESS, and SAVEE, PL-Distill compresses an 8.4B-parameter teacher to a compact 1.1B-parameter student, consistently outperforming the teacher, state-of-the-art pretrained models, and other KD baselines across all metrics.
>
---
#### [new 015] DFKI-Speech System for WildSpoof Challenge: A robust framework for SASV In-the-Wild
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于语音 spoofing 检测与说话人验证任务，旨在提升真实场景下的说话人验证鲁棒性。提出融合检测器与验证网络的框架，结合多尺度特征和自适应损失函数，增强系统区分能力。**

- **链接: [https://arxiv.org/pdf/2602.02286v1](https://arxiv.org/pdf/2602.02286v1)**

> **作者:** Arnab Das; Yassine El Kheir; Enes Erdem Erdogan; Feidi Kallel; Tim Polzehl; Sebastian Moeller
>
> **摘要:** This paper presents the DFKI-Speech system developed for the WildSpoof Challenge under the Spoofing aware Automatic Speaker Verification (SASV) track. We propose a robust SASV framework in which a spoofing detector and a speaker verification (SV) network operate in tandem. The spoofing detector employs a self-supervised speech embedding extractor as the frontend, combined with a state-of-the-art graph neural network backend. In addition, a top-3 layer based mixture-of-experts (MoE) is used to fuse high-level and low-level features for effective spoofed utterance detection. For speaker verification, we adapt a low-complexity convolutional neural network that fuses 2D and 1D features at multiple scales, trained with the SphereFace loss. Additionally, contrastive circle loss is applied to adaptively weight positive and negative pairs within each training batch, enabling the network to better distinguish between hard and easy sample pairs. Finally, fixed imposter cohort based AS Norm score normalization and model ensembling are used to further enhance the discriminative capability of the speaker verification system.
>
---
#### [new 016] Dual-View Predictive Diffusion: Lightweight Speech Enhancement via Spectrogram-Image Synergy
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决现有模型对语音谱图处理效率低的问题。提出DVPD模型，通过双视角协同提升效果，同时显著降低参数和计算量。**

- **链接: [https://arxiv.org/pdf/2602.00568v1](https://arxiv.org/pdf/2602.00568v1)**

> **作者:** Ke Xue; Rongfei Fan; Kai Li; Shanping Yu; Puning Zhao; Jianping An
>
> **摘要:** Diffusion models have recently set new benchmarks in Speech Enhancement (SE). However, most existing score-based models treat speech spectrograms merely as generic 2D images, applying uniform processing that ignores the intrinsic structural sparsity of audio, which results in inefficient spectral representation and prohibitive computational complexity. To bridge this gap, we propose DVPD, an extremely lightweight Dual-View Predictive Diffusion model, which uniquely exploits the dual nature of spectrograms as both visual textures and physical frequency-domain representations across both training and inference stages. Specifically, during training, we optimize spectral utilization via the Frequency-Adaptive Non-uniform Compression (FANC) encoder, which preserves critical low-frequency harmonics while pruning high-frequency redundancies. Simultaneously, we introduce a Lightweight Image-based Spectro-Awareness (LISA) module to capture features from a visual perspective with minimal overhead. During inference, we propose a Training-free Lossless Boost (TLB) strategy that leverages the same dual-view priors to refine generation quality without any additional fine-tuning. Extensive experiments across various benchmarks demonstrate that DVPD achieves state-of-the-art performance while requiring only 35% of the parameters and 40% of the inference MACs compared to SOTA lightweight model, PGUSE. These results highlight DVPD's superior ability to balance high-fidelity speech quality with extreme architectural efficiency. Code and audio samples are available at the anonymous website: {https://anonymous.4open.science/r/dvpd_demo-E630}
>
---
#### [new 017] Adapting Where It Matters: Depth-Aware Adaptation for Efficient Multilingual Speech Recognition in Low-Resource Languages
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多语言语音识别任务，解决低资源语言适应问题。提出DAMA框架，按层分配适配能力，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.01008v1](https://arxiv.org/pdf/2602.01008v1)**

> **作者:** Yang Xiao; Eun-Jung Holden; Ting Dang
>
> **备注:** 13 pages
>
> **摘要:** Recent speech foundation models excel at multilingual automatic speech recognition (ASR) for high-resource languages, but adapting them to low-resource languages remains challenging due to data scarcity and efficiency constraints. Full-model fine-tuning is computationally expensive and prone to overfitting, while parameter-efficient methods like LoRA apply adaptation uniformly across layers, overlooking internal representations thus compromising effectiveness and efficiency. We analyze multilingual ASR models and reveal a U-shaped adaptability pattern: early and late layers are language-specific and require more adaptation, while intermediate layers retain shared semantics and need less. Building on this observation, we propose DAMA, a Depth-Aware Model Adaptation framework that allocates adaptation capacity according to each layer's role. DAMA also introduces Singular Value Decomposition (SVD)-based initialization to constrain adaptation and preserve the U-shaped pattern, as well as a frozen middle-layer basis for further efficiency. Evaluated on 18 low-resource languages across two benchmark datasets, DAMA matches or surpasses state-of-the-art accuracy with 80% fewer trainable parameters, achieves a 29% error reduction under extreme data scarcity, and significantly improves memory, training time, and computational efficiency over baselines. These results highlight the benefits of structure-aware adaptation for efficient, scalable multilingual ASR.
>
---
#### [new 018] Solving Room Impulse Response Inverse Problems Using Flow Matching with Analytic Wiener Denoiser
- **分类: eess.AS**

- **简介: 该论文属于声学逆问题任务，旨在解决房间脉冲响应（RIR）估计问题。提出RIRFlow框架，结合流匹配与解析维纳去噪器，无需训练即可有效处理逆问题。**

- **链接: [https://arxiv.org/pdf/2602.00652v1](https://arxiv.org/pdf/2602.00652v1)**

> **作者:** Kyung Yun Lee; Nils Meyer-Kahlen; Vesa Välimäki; Sebastian J. Schlecht
>
> **备注:** Submitted to the Journal of the Acoustical Society of America (JASA)
>
> **摘要:** Room impulse response (RIR) estimation naturally arises as a class of inverse problems, including denoising and deconvolution. While recent approaches often rely on supervised learning or learned generative priors, such methods require large amounts of training data and may generalize poorly outside the training distribution. In this work, we present RIRFlow, a training-free Bayesian framework for RIR inverse problems using flow matching. We derive a flow-consistent analytic prior from the statistical structure of RIRs, eliminating the need for data-driven priors. Specifically, we model RIR as a Gaussian process with exponentially decaying variance, which yields a closed-form minimum mean squared error (MMSE) Wiener denoiser. This analytic denoiser is integrated as a prior in an existing flow-based inverse solver, where inverse problems are solved via guided posterior sampling. Furthermore, we extend the solver to nonlinear and non-Gaussian inverse problems via a local Gaussian approximation of the guided posterior, and empirically demonstrate that this approximation remains effective in practice. Experiments on real RIRs across different inverse problems demonstrate robust performance, highlighting the effectiveness of combining a classic RIR model with the recent flow-based generative inference.
>
---
#### [new 019] LPIPS-AttnWav2Lip: Generic Audio-Driven lip synchronization for Talking Head Generation in the Wild
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于音频驱动的说话头生成任务，旨在解决唇音同步问题。通过改进的U-Net结构和LPIPS损失函数，提升生成图像的质量与同步精度。**

- **链接: [https://arxiv.org/pdf/2602.00189v1](https://arxiv.org/pdf/2602.00189v1)**

> **作者:** Zhipeng Chen; Xinheng Wang; Lun Xie; Haijie Yuan; Hang Pan
>
> **备注:** This paper has been accepted by Elsevier's \textit{Speech Communication} journal. Official publication link: https://doi.org/10.1016/j.specom.2023.103028 The code for the paper is available at the following link: https://github.com/FelixChan9527/LPIPS-AttnWav2Lip
>
> **摘要:** Researchers have shown a growing interest in Audio-driven Talking Head Generation. The primary challenge in talking head generation is achieving audio-visual coherence between the lips and the audio, known as lip synchronization. This paper proposes a generic method, LPIPS-AttnWav2Lip, for reconstructing face images of any speaker based on audio. We used the U-Net architecture based on residual CBAM to better encode and fuse audio and visual modal information. Additionally, the semantic alignment module extends the receptive field of the generator network to obtain the spatial and channel information of the visual features efficiently; and match statistical information of visual features with audio latent vector to achieve the adjustment and injection of the audio content information to the visual information. To achieve exact lip synchronization and to generate realistic high-quality images, our approach adopts LPIPS Loss, which simulates human judgment of image quality and reduces instability possibility during the training process. The proposed method achieves outstanding performance in terms of lip synchronization accuracy and visual quality as demonstrated by subjective and objective evaluation results. The code for the paper is available at the following link: https://github.com/FelixChan9527/LPIPS-AttnWav2Lip
>
---
#### [new 020] Speaking Without Sound: Multi-speaker Silent Speech Voicing with Facial Inputs Only
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在通过面部输入生成多说话人无声语音。解决无声音输入下语音生成与身份匹配的问题，提出基于EMG和面部图像的框架及音高解耦嵌入方法。**

- **链接: [https://arxiv.org/pdf/2602.01879v1](https://arxiv.org/pdf/2602.01879v1)**

> **作者:** Jaejun Lee; Yoori Oh; Kyogu Lee
>
> **备注:** This paper was presented at ICASSP 2025
>
> **摘要:** In this paper, we introduce a novel framework for generating multi-speaker speech without relying on any audible inputs. Our approach leverages silent electromyography (EMG) signals to capture linguistic content, while facial images are used to match with the vocal identity of the target speaker. Notably, we present a pitch-disentangled content embedding that enhances the extraction of linguistic content from EMG signals. Extensive analysis demonstrates that our method can generate multi-speaker speech without any audible inputs and confirms the effectiveness of the proposed pitch-disentanglement approach.
>
---
#### [new 021] Voting-based Pitch Estimation with Temporal and Frequential Alignment and Correlation Aware Selection
- **分类: cs.SD**

- **简介: 该论文属于语音处理中的基频估计任务，旨在提高估计的准确性和鲁棒性。通过投票方法结合时频对齐和相关性选择，优化了多估计器的集成效果。**

- **链接: [https://arxiv.org/pdf/2602.01727v1](https://arxiv.org/pdf/2602.01727v1)**

> **作者:** Junya Koguchi; Tomoki Koriyama
>
> **备注:** Accepted for ICASSP 2026
>
> **摘要:** The voting method, an ensemble approach for fundamental frequency estimation, is empirically known for its robustness but lacks thorough investigation. This paper provides a principled analysis and improvement of this technique. First, we offer a theoretical basis for its effectiveness, explaining the error variance reduction for fundamental frequency estimation and invoking Condorcet's jury theorem for voiced/unvoiced detection accuracy. To address its practical limitations, we propose two key improvements: 1) a pre-voting alignment procedure to correct temporal and frequential biases among estimators, and 2) a greedy algorithm to select a compact yet effective subset of estimators based on error correlation. Experiments on a diverse dataset of speech, singing, and music show that our proposed method with alignment outperforms individual state-of-the-art estimators in clean conditions and maintains robust voiced/unvoiced detection in noisy environments.
>
---
#### [new 022] Short-wave admittance correction for a time-domain cochlear transmission line model
- **分类: eess.AS; physics.bio-ph**

- **简介: 该论文属于听觉模型研究任务，旨在解决时间域传输线模型中二维效应导致的模拟偏差问题。通过引入自回归和回归方法修正基底膜导纳，提升模型精度与动态范围。**

- **链接: [https://arxiv.org/pdf/2602.01758v1](https://arxiv.org/pdf/2602.01758v1)**

> **作者:** François Deloche; Morgan Thienpont; Sarah Verhulst
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** Transmission line (TL) models implemented in the time domain can efficiently simulate basilar-membrane (BM) displacement in response to transient or non-stationary sounds. By design, a TL model is well-suited for an one-dimensional (1-D) characterization of the traveling wave, but the real configuration of the cochlea also introduces higher-dimensional effects. Such effects include the focusing of the pressure around the BM and transverse viscous damping, both of which are magnified in the short-wave region. The two effects depend on the wavelength and are more readily expressed in the frequency domain. In this paper, we introduce a numerical correction for the BM admittance to account for 2-D effects in the time domain using autoregressive filtering and regression techniques. The correction was required for the implementation of a TL model tailored to the gerbil cochlear physiology. The model, which includes instantaneous nonlinearities in the form of variable damping, initially presented insufficient compression with increasing sound levels. This limitation was explained by the strong coupling between gain and frequency selectivity assumed in the 1-D nonlinear TL model, whereas cochlear frequency selectivity shows only a moderate dependence on sound level in small mammals. The correction factor was implemented in the gerbil model and made level-dependent using a feedback loop. The updated model achieved some decoupling between frequency selectivity and gain, providing 5 dB of additional gain and extending the range of sound levels of the compressive regime by 10 dB. We discuss the relevance of this work through two key features: the integration of both analytical and regression methods for characterizing BM admittance, and the combination of instantaneous and non-instantaneous nonlinearities.
>
---
#### [new 023] RIR-Former: Coordinate-Guided Transformer for Continuous Reconstruction of Room Impulse Responses
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于声学信号处理任务，旨在解决密集测量房间脉冲响应（RIR）不实际的问题。提出RIR-Former模型，通过Transformer实现无网格的RIR重建与插值。**

- **链接: [https://arxiv.org/pdf/2602.01861v1](https://arxiv.org/pdf/2602.01861v1)**

> **作者:** Shaoheng Xu; Chunyi Sun; Jihui; Zhang; Prasanga N. Samarasinghe; Thushara D. Abhayapala
>
> **备注:** Accepted to International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026. Equal contribution: Shaoheng Xu and Chunyi Sun
>
> **摘要:** Room impulse responses (RIRs) are essential for many acoustic signal processing tasks, yet measuring them densely across space is often impractical. In this work, we propose RIR-Former, a grid-free, one-step feed-forward model for RIR reconstruction. By introducing a sinusoidal encoding module into a transformer backbone, our method effectively incorporates microphone position information, enabling interpolation at arbitrary array locations. Furthermore, a segmented multi-branch decoder is designed to separately handle early reflections and late reverberation, improving reconstruction across the entire RIR. Experiments on diverse simulated acoustic environments demonstrate that RIR-Former consistently outperforms state-of-the-art baselines in terms of normalized mean square error (NMSE) and cosine distance (CD), under varying missing rates and array configurations. These results highlight the potential of our approach for practical deployment and motivate future work on scaling from randomly spaced linear arrays to complex array geometries, dynamic acoustic scenes, and real-world environments.
>
---
#### [new 024] The TMU System for the XACLE Challenge: Training Large Audio Language Models with CLAP Pseudo-Labels
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对XACLE任务，解决音频与文本语义对齐问题。提出基于大音频语言模型的系统，通过三阶段训练提升性能，取得优异结果。**

- **链接: [https://arxiv.org/pdf/2602.00604v1](https://arxiv.org/pdf/2602.00604v1)**

> **作者:** Ayuto Tsutsumi; Kohei Tanaka; Sayaka Shiota
>
> **备注:** 3 pages; 2 figures; 2 tables; Accepted at ICASSP 2026 Workshop (SP Grand Challenges, GC-12: XACLE)
>
> **摘要:** In this paper, we propose a submission to the x-to-audio alignment (XACLE) challenge. The goal is to predict semantic alignment of a given general audio and text pair. The proposed system is based on a large audio language model (LALM) architecture. We employ a three-stage training pipeline: automated audio captioning pretraining, pretraining with CLAP pseudo-labels, and fine-tuning on the XACLE dataset. Our experiments show that pretraining with CLAP pseudo-labels is the primary performance driver. On the XACLE test set, our system reaches an SRCC of 0.632, significantly outperforming the baseline system (0.334) and securing third place in the challenge team ranking. Code and models can be found at https://github.com/shiotalab-tmu/tmu-xacle2026
>
---
#### [new 025] Edit Content, Preserve Acoustics: Imperceptible Text-Based Speech Editing via Self-Consistency Rewards
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音编辑任务，旨在实现文本驱动的语音修改，解决内容与声学特征纠缠导致的不稳定和边界伪影问题。工作包括构建结构基础和感知对齐机制，提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2602.00560v1](https://arxiv.org/pdf/2602.00560v1)**

> **作者:** Yong Ren; Jiangyan Yi; Jianhua Tao; Zhengqi Wen; Tao Wang
>
> **摘要:** Imperceptible text-based speech editing allows users to modify spoken content by altering the transcript. It demands that modified segments fuse seamlessly with the surrounding context. Prevalent methods operating in the acoustic space suffer from inherent content-style entanglement, leading to generation instability and boundary artifacts. In this paper, we propose a novel framework grounded in the principle of "Edit Content, Preserve Acoustics". Our approach relies on two core components: (1) Structural Foundations, which decouples editing into a stable semantic space while delegating acoustic reconstruction to a Flow Matching decoder; and (2) Perceptual Alignment, which employs a novel Self-Consistency Rewards Group Relative Policy Optimization. By leveraging a pre-trained Text-to-Speech model as an implicit critic -- complemented by strict intelligibility and duration constraints -- we effectively align the edited semantic token sequence with the original context. Empirical evaluations demonstrate that our method significantly outperforms state-of-the-art autoregressive and non-autoregressive baselines, achieving superior intelligibility, robustness, and perceptual quality.
>
---
#### [new 026] SSNAPS: Audio-Visual Separation of Speech and Background Noise with Diffusion Inverse Sampling
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音分离任务，解决单麦克风下语音与背景噪声的分离问题。通过生成逆采样方法，联合建模语音和噪声，实现无监督分离，并适用于多说话人场景。**

- **链接: [https://arxiv.org/pdf/2602.01394v1](https://arxiv.org/pdf/2602.01394v1)**

> **作者:** Yochai Yemini; Yoav Ellinson; Rami Ben-Ari; Sharon Gannot; Ethan Fetaya
>
> **摘要:** This paper addresses the challenge of audio-visual single-microphone speech separation and enhancement in the presence of real-world environmental noise. Our approach is based on generative inverse sampling, where we model clean speech and ambient noise with dedicated diffusion priors and jointly leverage them to recover all underlying sources. To achieve this, we reformulate a recent inverse sampler to match our setting. We evaluate on mixtures of 1, 2, and 3 speakers with noise and show that, despite being entirely unsupervised, our method consistently outperforms leading supervised baselines in \ac{WER} across all conditions. We further extend our framework to handle off-screen speaker separation. Moreover, the high fidelity of the separated noise component makes it suitable for downstream acoustic scene detection. Demo page: https://ssnapsicml.github.io/ssnapsicml2026/
>
---
#### [new 027] Joint Optimization of ASV and CM tasks: BTUEF Team's Submission for WildSpoof Challenge
- **分类: eess.AS**

- **简介: 该论文属于语音安全领域，解决对抗攻击下的说话人验证问题。提出联合优化框架，提升ASV与CM系统的协同效果。**

- **链接: [https://arxiv.org/pdf/2602.01722v1](https://arxiv.org/pdf/2602.01722v1)**

> **作者:** Oguzhan Kurnaz; Jagabandhu Mishra; Tomi Kinnunen; Cemal Hanilci
>
> **摘要:** Spoofing-aware speaker verification (SASV) jointly addresses automatic speaker verification and spoofing countermeasures to improve robustness against adversarial attacks. In this paper, we investigate our recently proposed modular SASV framework that enables effective reuse of publicly available ASV and CM systems through non-linear fusion, explicitly modeling their interaction, and optimization with an operating-condition-dependent trainable a-DCF loss. The framework is evaluated using ECAPA-TDNN and ReDimNet as ASV embedding extractors and SSL-AASIST as the CM model, with experiments conducted both with and without fine-tuning on the WildSpoof SASV training data. Results show that the best performance is achieved by combining ReDimNet-based ASV embeddings with fine-tuned SSL-AASIST representations, yielding an a-DCF of 0.0515 on the progress evaluation set and 0.2163 on the final evaluation set.
>
---
#### [new 028] VoxServe: Streaming-Centric Serving System for Speech Language Models
- **分类: cs.LG; cs.AI; cs.DC; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型服务任务，旨在解决流式设置下的低延迟、高吞吐问题。提出VoxServe系统，通过抽象模型执行和优化调度提升性能。**

- **链接: [https://arxiv.org/pdf/2602.00269v1](https://arxiv.org/pdf/2602.00269v1)**

> **作者:** Keisuke Kamahori; Wei-Tzu Lee; Atindra Jha; Rohan Kadekodi; Stephanie Wang; Arvind Krishnamurthy; Baris Kasikci
>
> **备注:** The code is available at https://github.com/vox-serve/vox-serve
>
> **摘要:** Deploying modern Speech Language Models (SpeechLMs) in streaming settings requires systems that provide low latency, high throughput, and strong guarantees of streamability. Existing systems fall short of supporting diverse models flexibly and efficiently. We present VoxServe, a unified serving system for SpeechLMs that optimizes streaming performance. VoxServe introduces a model-execution abstraction that decouples model architecture from system-level optimizations, thereby enabling support for diverse SpeechLM architectures within a single framework. Building on this abstraction, VoxServe implements streaming-aware scheduling and an asynchronous inference pipeline to improve end-to-end efficiency. Evaluations across multiple modern SpeechLMs show that VoxServe achieves 10-20x higher throughput than existing implementations at comparable latency while maintaining high streaming viability. The code of VoxServe is available at https://github.com/vox-serve/vox-serve.
>
---
#### [new 029] MTAVG-Bench: A Comprehensive Benchmark for Evaluating Multi-Talker Dialogue-Centric Audio-Video Generation
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于多说话人对话生成任务，旨在解决现有评估基准无法有效检测生成视频中身份漂移、转场不自然等问题。工作包括构建MTAVG-Bench基准，评估多模态生成质量。**

- **链接: [https://arxiv.org/pdf/2602.00607v1](https://arxiv.org/pdf/2602.00607v1)**

> **作者:** Yang-Hao Zhou; Haitian Li; Rexar Lin; Heyan Huang; Jinxing Zhou; Changsen Yuan; Tian Lan; Ziqin Zhou; Yudong Li; Jiajun Xu; Jingyun Liao; Yi-Ming Cheng; Xuefeng Chen; Xian-Ling Mao; Yousheng Feng
>
> **摘要:** Recent advances in text-to-audio-video (T2AV) generation have enabled models to synthesize audio-visual videos with multi-participant dialogues. However, existing evaluation benchmarks remain largely designed for human-recorded videos or single-speaker settings. As a result, potential errors that occur in generated multi-talker dialogue videos, such as identity drift, unnatural turn transitions, and audio-visual misalignment, cannot be effectively captured and analyzed. To address this issue, we introduce MTAVG-Bench, a benchmark for evaluating audio-visual multi-speaker dialogue generation. MTAVG-Bench is built via a semi-automatic pipeline, where 1.8k videos are generated using multiple popular models with carefully designed prompts, yielding 2.4k manually annotated QA pairs. The benchmark evaluates multi-speaker dialogue generation at four levels: audio-visual signal fidelity, temporal attribute consistency, social interaction, and cinematic expression. We benchmark 12 proprietary and open-source omni-models on MTAVG-Bench, with Gemini 3 Pro achieving the strongest overall performance, while leading open-source models remain competitive in signal fidelity and consistency. Overall, MTAVG-Bench enables fine-grained failure analysis for rigorous model comparison and targeted video generation refinement.
>
---
#### [new 030] Evaluating Acoustic Data Transmission Schemes for Ad-Hoc Communication Between Nearby Smart Devices
- **分类: cs.NI; cs.SD**

- **简介: 该论文属于无线通信任务，旨在评估声学数据传输方案的可靠性。解决真实环境下方案性能不佳的问题，通过实验测试和数据集构建，提出更稳健的设计方法。**

- **链接: [https://arxiv.org/pdf/2602.02249v1](https://arxiv.org/pdf/2602.02249v1)**

> **作者:** Florentin Putz; Philipp Fortmann; Jan Frank; Christoph Haugwitz; Mario Kupnik; Matthias Hollick
>
> **备注:** 31 pages, 9 figures, the dataset is available at https://doi.org/10.5281/zenodo.17661991
>
> **摘要:** Acoustic data transmission offers a compelling alternative to Bluetooth and NFC by leveraging the ubiquitous speakers and microphones in smartphones and IoT devices. However, most research in this field relies on simulations or limited on-device testing, which makes the real-world reliability of proposed schemes difficult to assess. We systematically reviewed 31 acoustic communication studies for commodity devices and found that none provided accessible source code. After contacting authors and re-implementing three promising schemes, we assembled a testbed of eight representative acoustic communication systems. Using over 11000 smartphone transmissions in both realistic indoor environments and an anechoic chamber, we provide a systematic and repeatable methodology for evaluating the reliability and generalizability of these schemes under real-world conditions. Our results show that many existing schemes face challenges in practical usage, largely due to severe multipath propagation indoors and varying audio characteristics across device models. To support future research and foster more robust evaluations, we release our re-implementations alongside the first comprehensive dataset of real-world acoustic transmissions. Overall, our findings highlight the importance of rigorous on-device testing and underscore the need for robust design strategies to bridge the gap between simulation results and reliable IoT deployments.
>
---
#### [new 031] QuietPrint: Protecting 3D Printers Against Acoustic Side-Channel Attacks
- **分类: cs.CR; eess.AS**

- **简介: 该论文属于信息安全任务，旨在解决3D打印机遭受声学侧信道攻击导致的IP泄露问题，通过修改G-code实现无硬件保护。**

- **链接: [https://arxiv.org/pdf/2602.02198v1](https://arxiv.org/pdf/2602.02198v1)**

> **作者:** Seyed Ali Ghazi Asgar; Narasimha Reddy
>
> **摘要:** The 3D printing market has experienced significant growth in recent years, with an estimated revenue of 15 billion USD for 2025. Cyber-attacks targeting the 3D printing process whether through the machine itself, the supply chain, or the fabricated components are becoming increasingly common. One major concern is intellectual property (IP) theft, where a malicious attacker gains access to the design file. One method for carrying out such theft is through side-channel attacks. In this work, we investigate the possibility of IP theft via acoustic side channels and propose a novel method to protect 3D printers against such attacks. The primary advantage of our approach is that it requires no additional hardware, such as large speakers or noise-canceling devices. Instead, it secures printed parts by minimal modifications to the G-code.
>
---
#### [new 032] Bias in the Ear of the Listener: Assessing Sensitivity in Audio Language Models Across Linguistic, Demographic, and Positional Variations
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究多语言大模型中的语音偏见问题，构建了BiasInEar数据集，评估模型在语言、性别和选项顺序等变化下的表现，旨在提升语音集成模型的公平性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.01030v1](https://arxiv.org/pdf/2602.01030v1)**

> **作者:** Sheng-Lun Wei; Yu-Ling Liao; Yen-Hua Chang; Hen-Hsen Huang; Hsin-Hsi Chen
>
> **备注:** Accepted as a long findings paper at EACL 2026
>
> **摘要:** This work presents the first systematic investigation of speech bias in multilingual MLLMs. We construct and release the BiasInEar dataset, a speech-augmented benchmark based on Global MMLU Lite, spanning English, Chinese, and Korean, balanced by gender and accent, and totaling 70.8 hours ($\approx$4,249 minutes) of speech with 11,200 questions. Using four complementary metrics (accuracy, entropy, APES, and Fleiss' $κ$), we evaluate nine representative models under linguistic (language and accent), demographic (gender), and structural (option order) perturbations. Our findings reveal that MLLMs are relatively robust to demographic factors but highly sensitive to language and option order, suggesting that speech can amplify existing structural biases. Moreover, architectural design and reasoning strategy substantially affect robustness across languages. Overall, this study establishes a unified framework for assessing fairness and robustness in speech-integrated LLMs, bridging the gap between text- and speech-based evaluation. The resources can be found at https://github.com/ntunlplab/BiasInEar.
>
---
#### [new 033] Cross-Modal Binary Attention: An Energy-Efficient Fusion Framework for Audio-Visual Learning
- **分类: cs.MM; cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于多模态融合任务，旨在解决传统方法在计算复杂度与信息提取间的矛盾。提出CMQKA机制，实现高效跨模态融合，构建SNNergy框架，提升音频-视觉学习的能效与效果。**

- **链接: [https://arxiv.org/pdf/2602.00701v1](https://arxiv.org/pdf/2602.00701v1)**

> **作者:** Mohamed Saleh; Zahra Ahmadi
>
> **摘要:** Effective multimodal fusion requires mechanisms that can capture complex cross-modal dependencies while remaining computationally scalable for real-world deployment. Existing audio-visual fusion approaches face a fundamental trade-off: attention-based methods effectively model cross-modal relationships but incur quadratic computational complexity that prevents hierarchical, multi-scale architectures, while efficient fusion strategies rely on simplistic concatenation that fails to extract complementary cross-modal information. We introduce CMQKA, a novel cross-modal fusion mechanism that achieves linear O(N) complexity through efficient binary operations, enabling scalable hierarchical fusion previously infeasible with conventional attention. CMQKA employs bidirectional cross-modal Query-Key attention to extract complementary spatiotemporal features and uses learnable residual fusion to preserve modality-specific characteristics while enriching representations with cross-modal information. Building upon CMQKA, we present SNNergy, an energy-efficient multimodal fusion framework with a hierarchical architecture that processes inputs through progressively decreasing spatial resolutions and increasing semantic abstraction. This multi-scale fusion capability allows the framework to capture both local patterns and global context across modalities. Implemented with event-driven binary spike operations, SNNergy achieves remarkable energy efficiency while maintaining fusion effectiveness and establishing new state-of-the-art results on challenging audio-visual benchmarks, including CREMA-D, AVE, and UrbanSound8K-AV, significantly outperforming existing multimodal fusion baselines. Our framework advances multimodal fusion by introducing a scalable fusion mechanism that enables hierarchical cross-modal integration with practical energy efficiency for real-world audio-visual intelligence systems.
>
---
#### [new 034] Generative AI in Signal Processing Education: An Audio Foundation Model Based Approach
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于教育技术任务，旨在将生成式AI应用于信号处理教学。通过提出SPEduAFM，解决传统教学与AI创新结合的问题，实现更高效、互动的学习体验。**

- **链接: [https://arxiv.org/pdf/2602.01249v1](https://arxiv.org/pdf/2602.01249v1)**

> **作者:** Muhammad Salman Khan; Ahmad Ullah; Siddique Latif; Junaid Qadir
>
> **备注:** accepted at IEEE EDUCON 2026
>
> **摘要:** Audio Foundation Models (AFMs), a specialized category of Generative AI (GenAI), have the potential to transform signal processing (SP) education by integrating core applications such as speech and audio enhancement, denoising, source separation, feature extraction, automatic classification, and real-time signal analysis into learning and research. This paper introduces SPEduAFM, a conceptual AFM tailored for SP education, bridging traditional SP principles with GenAI-driven innovations. Through an envisioned case study, we outline how AFMs can enable a range of applications, including automated lecture transcription, interactive demonstrations, and inclusive learning tools, showcasing their potential to transform abstract concepts into engaging, practical experiences. This paper also addresses challenges such as ethics, explainability, and customization by highlighting dynamic, real-time auditory interactions that foster experiential and authentic learning. By presenting SPEduAFM as a forward-looking vision, we aim to inspire broader adoption of GenAI in engineering education, enhancing accessibility, engagement, and innovation in the classroom and beyond.
>
---
#### [new 035] A Baseline Multimodal Approach to Emotion Recognition in Conversations
- **分类: cs.CL; cs.AI; cs.CY; cs.SD; eess.AS**

- **简介: 该论文属于对话情感识别任务，旨在通过融合文本和语音模型提升情感识别效果。工作包括构建轻量级多模态基线方法并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.00914v1](https://arxiv.org/pdf/2602.00914v1)**

> **作者:** Víctor Yeste; Rodrigo Rivas-Arévalo
>
> **备注:** 10 pages
>
> **摘要:** We present a lightweight multimodal baseline for emotion recognition in conversations using the SemEval-2024 Task 3 dataset built from the sitcom Friends. The goal of this report is not to propose a novel state-of-the-art method, but to document an accessible reference implementation that combines (i) a transformer-based text classifier and (ii) a self-supervised speech representation model, with a simple late-fusion ensemble. We report the baseline setup and empirical results obtained under a limited training protocol, highlighting when multimodal fusion improves over unimodal models. This preprint is provided for transparency and to support future, more rigorous comparisons.
>
---
#### [new 036] Kanade: A Simple Disentangled Tokenizer for Spoken Language Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音建模任务，旨在解决语音分词中语言与非语言信息混杂的问题。提出Kanade模型，分离声学常量，提升语音合成质量。**

- **链接: [https://arxiv.org/pdf/2602.00594v1](https://arxiv.org/pdf/2602.00594v1)**

> **作者:** Zhijie Huang; Stephen McIntosh; Daisuke Saito; Nobuaki Minematsu
>
> **摘要:** A good language model starts with a good tokenizer. Tokenization is especially important for speech modeling, which must handle continuous signals that mix linguistic and non-linguistic information. A speech tokenizer should extract phonetics and prosody, suppress linguistically irrelevant information like speaker identity, and enable high-quality synthesis. We present Kanade, a single-layer disentangled speech tokenizer that realizes this ideal. Kanade separates out acoustic constants to create a single stream of tokens that captures rich phonetics and prosody. It does so without the need for auxiliary methods that existing disentangled codecs often rely on. Experiments show that Kanade achieves state-of-the-art speaker disentanglement and lexical availability, while maintaining excellent reconstruction quality.
>
---
## 更新

#### [replaced 001] FastSLM: Hierarchical Frame Q-Former for Effective Speech Modality Adaptation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音与语言模型融合任务，旨在解决长时语音输入导致的计算瓶颈问题。通过提出HFQ-Former架构，实现高效语音表示压缩，提升模型处理长文本的能力。**

- **链接: [https://arxiv.org/pdf/2601.06199v2](https://arxiv.org/pdf/2601.06199v2)**

> **作者:** Junseok Lee; Sangyong Lee; Chang-Jae Chun
>
> **摘要:** Although Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in vision, language, and video understanding tasks, scaling them to long-form speech remains a critical bottleneck due to the explosive growth of input tokens. Existing speech-language models typically project high-frame-rate acoustic features directly into the LLM input space, rendering long-context processing computationally prohibitive as audio duration increases. In this paper, we present FastSLM, a token-efficient architecture designed to overcome this scalability limit through extreme temporal compression. At its core is the Hierarchical Frame Querying Transformer (HFQ-Former), which progressively distills local acoustic details into compact, semantically rich representations across multiple temporal scales. This hierarchical abstraction reduces the speech representation rate to just 1.67 tokens per second, achieving a 93 percent reduction in tokens compared to standard frame-level adapters, while preserving the critical context required for complex reasoning. Experimental results demonstrate that FastSLM achieves competitive performance with state-of-the-art models on long-form benchmarks, despite operating with significantly lower FLOPs and parameter counts. Our findings establish that extreme token compression is a viable pathway to making real-time, long-context speech understanding feasible for LLMs, even under strict computational constraints. The source code and model checkpoints are available at https://anonymous.4open.science/r/FastSLM-8BD3
>
---
#### [replaced 002] Investigating Modality Contribution in Audio LLMs for Music
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于音频大语言模型研究任务，旨在解决模型是否真正依赖音频输入的问题。通过MM-SHAP框架量化模态贡献，发现模型主要依赖文本，但音频仍被部分利用。**

- **链接: [https://arxiv.org/pdf/2509.20641v2](https://arxiv.org/pdf/2509.20641v2)**

> **作者:** Giovana Morais; Magdalena Fuentes
>
> **备注:** 5 pages, 2 figures, accepted at ICASSP 2026
>
> **摘要:** Audio Large Language Models (Audio LLMs) enable human-like conversation about music, yet it is unclear if they are truly listening to the audio or just using textual reasoning, as recent benchmarks suggest. This paper investigates this issue by quantifying the contribution of each modality to a model's output. We adapt the MM-SHAP framework, a performance-agnostic score based on Shapley values that quantifies the relative contribution of each modality to a model's prediction. We evaluate two models on the MuChoMusic benchmark and find that the model with higher accuracy relies more on text to answer questions, but further inspection shows that even if the overall audio contribution is low, models can successfully localize key sound events, suggesting that audio is not entirely ignored. Our study is the first application of MM-SHAP to Audio LLMs and we hope it will serve as a foundational step for future research in explainable AI and audio.
>
---
#### [replaced 003] SupCLAP: Controlling Optimization Trajectory Drift in Audio-Text Contrastive Learning with Support Vector Regularization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于多模态预训练任务，旨在解决对比学习中优化轨迹漂移问题。通过引入支持向量正则化（SVR），控制负样本的推力分量，提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2509.21033v2](https://arxiv.org/pdf/2509.21033v2)**

> **作者:** Jiehui Luo; Yuguo Yin; Yuxin Xie; Jinghan Ru; Xianwei Zhuang; Minghua He; Aofan Liu; Zihan Xiong; Dongchao Yang
>
> **摘要:** Contrastive language-audio pretraining, which aims to unify multimodal representations in a shared embedding space, serves as a cornerstone for building a wide range of applications, from cross-modal retrieval to cutting-edge multimodal large language models. However, we find that the perpendicular component of the pushing force from negative samples in contrastive learning is a double-edged sword: it contains rich supplementary information from negative samples, yet its unconstrained nature causes optimization trajectory drift and training instability. To address this, we propose Support Vector Regularization (SVR), a method that introduces an auxiliary support vector to control this perpendicular component, aiming to harness its rich information while mitigating the associated trajectory drift. The efficacy of SVR is critically governed by its semantic radius, for which we explore two unsupervised modeling strategies: direct parameterization and an adaptive radius predictor module enhanced with constraints to improve its predicting accuracy. Extensive experimental results demonstrate that our method surpasses widely used baselines like InfoNCE and SigLIP loss across classification, monolingual retrieval, and multilingual retrieval on standard audio-text datasets. Both the theoretical analysis and the experimental results on optimizing trajectory drift validate the correctness and effectiveness of our SVR method. Notably, our method is highly efficient, it operates without the need for extra training data or inference computation, and adds only a negligible overhead to the training.
>
---
#### [replaced 004] UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在开发轻量级模型以实现实时处理。通过网络架构搜索优化U-Net结构，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2503.00340v2](https://arxiv.org/pdf/2503.00340v2)**

> **作者:** Xiaobin Rong; Leyan Yang; Dahan Wang; Yuxiang Hu; Changbao Zhu; Kai Chen; Jing Lu
>
> **备注:** Accepted by IEEE TASLP
>
> **摘要:** Lightweight models are essential for real-time speech enhancement applications. In recent years, there has been a growing trend toward developing increasingly compact models for speech enhancement. In this paper, we propose an Ultra-Lightweight U-net optimized by Network Architecture Search (UL-UNAS), which is suitable for implementation in low-footprint devices. Firstly, we explore the application of various efficient convolutional blocks within the U-Net framework to identify the most promising candidates. Secondly, we introduce two boosting components to enhance the capacity of these convolutional blocks: a novel activation function named affine PReLU and a causal time-frequency attention module. Furthermore, we leverage neural architecture search to discover an optimal architecture within our carefully designed search space. By integrating the above strategies, UL-UNAS not only significantly outperforms the latest ultra-lightweight models with the same or lower computational complexity, but also delivers competitive performance compared to recent baseline models that require substantially higher computational resources. Source code and audio demos are available at https://github.com/Xiaobin-Rong/ul-unas.
>
---
#### [replaced 005] Estimating Respiratory Effort from Nocturnal Breathing Sounds for Obstructive Sleep Apnoea Screening
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于睡眠呼吸暂停筛查任务，旨在通过夜间呼吸声估计呼吸努力，解决传统方法依赖传感器的问题。工作包括提出一种融合框架，提升OSA检测效果。**

- **链接: [https://arxiv.org/pdf/2509.14944v2](https://arxiv.org/pdf/2509.14944v2)**

> **作者:** Xiaolei Xu; Chaoyue Niu; Guy J. Brown; Hector Romero; Ning Ma
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Obstructive sleep apnoea (OSA) is a prevalent condition with significant health consequences, yet many patients remain undiagnosed due to the complexity and cost of over-night polysomnography. Acoustic-based screening provides a scalable alternative, yet performance is limited by environmental noise and the lack of physiological context. Respiratory effort is a key signal used in clinical scoring of OSA events, but current approaches require additional contact sensors that reduce scalability and patient comfort. This paper presents the first study to estimate respiratory effort directly from nocturnal audio, enabling physiological context to be recovered from sound alone. We propose a latent-space fusion framework that integrates the estimated effort embeddings with acoustic features for OSA detection. Using a dataset of 157 nights from 103 participants recorded in home environments, our respiratory effort estimator achieves a concordance correlation coefficient of 0.48, capturing meaningful respiratory dynamics. Fusing effort and audio improves sensitivity and AUC over audio-only baselines, especially at low apnoea-hypopnoea index thresholds. The proposed approach requires only smartphone audio at test time, which enables sensor-free, scalable, and longitudinal OSA monitoring.
>
---
#### [replaced 006] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出DeepGB-TB，用于快速、可解释的结核病筛查。解决传统诊断成本高、复杂的问题，结合音频和人口数据，采用交叉注意力机制和风险平衡损失提升准确率与可靠性。**

- **链接: [https://arxiv.org/pdf/2508.02741v2](https://arxiv.org/pdf/2508.02741v2)**

> **作者:** Zhixiang Lu; Yulong Li; Feilong Tang; Zhengyong Jiang; Chong Li; Mian Zhou; Tenglong Li; Jionglong Su
>
> **备注:** Accepted by AAAI 2026 (oral)
>
> **摘要:** Large-scale tuberculosis (TB) screening is limited by the high cost and operational complexity of traditional diagnostics, creating a need for artificial-intelligence solutions. We propose DeepGB-TB, a non-invasive system that instantly assigns TB risk scores using only cough audio and basic demographic data. The model couples a lightweight one-dimensional convolutional neural network for audio processing with a gradient-boosted decision tree for tabular features. Its principal innovation is a Cross-Modal Bidirectional Cross-Attention module (CM-BCA) that iteratively exchanges salient cues between modalities, emulating the way clinicians integrate symptoms and risk factors. To meet the clinical priority of minimizing missed cases, we design a Tuberculosis Risk-Balanced Loss (TRBL) that places stronger penalties on false-negative predictions, thereby reducing high-risk misclassifications. DeepGB-TB is evaluated on a diverse dataset of 1,105 patients collected across seven countries, achieving an AUROC of 0.903 and an F1-score of 0.851, representing a new state of the art. Its computational efficiency enables real-time, offline inference directly on common mobile devices, making it ideal for low-resource settings. Importantly, the system produces clinically validated explanations that promote trust and adoption by frontline health workers. By coupling AI innovation with public-health requirements for speed, affordability, and reliability, DeepGB-TB offers a tool for advancing global TB control.
>
---
#### [replaced 007] Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文聚焦多语言对话语音识别任务，旨在解决LLM与端到端模型性能差距问题。通过融合细调的Whisper和mHuBERT编码器，提升语音表示，取得良好效果。**

- **链接: [https://arxiv.org/pdf/2601.01461v3](https://arxiv.org/pdf/2601.01461v3)**

> **作者:** Yuxiang Mei; Dongxing Xu; Jiaen Liang; Yanhua Long
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
>
---
#### [replaced 008] Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决多语言ASR中的资源不足与泛化问题。通过基于语言家族的连接器共享策略，提升模型效率与跨域适应性。**

- **链接: [https://arxiv.org/pdf/2601.18899v2](https://arxiv.org/pdf/2601.18899v2)**

> **作者:** Yuchen Zhang; Ravi Shekhar; Haralambos Mouratidis
>
> **备注:** Accepted by EACL'26 main
>
> **摘要:** Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment.
>
---
#### [replaced 009] I-DCCRN-VAE: An Improved Deep Representation Learning Framework for Complex VAE-based Single-channel Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升单通道语音质量。通过改进VAE框架，增强模型泛化能力，解决噪声环境下语音清晰度不足的问题。**

- **链接: [https://arxiv.org/pdf/2510.12485v2](https://arxiv.org/pdf/2510.12485v2)**

> **作者:** Jiatong Li; Simon Doclo
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** Recently, a complex variational autoencoder (VAE)-based single-channel speech enhancement system based on the DCCRN architecture has been proposed. In this system, a noise suppression VAE (NSVAE) learns to extract clean speech representations from noisy speech using pretrained clean speech and noise VAEs with skip connections. In this paper, we improve DCCRN-VAE by incorporating three key modifications: 1) removing the skip connections in the pretrained VAEs to encourage more informative speech and noise latent representations; 2) using $β$-VAE in pretraining to better balance reconstruction and latent space regularization; and 3) a NSVAE generating both speech and noise latent representations. Experiments show that the proposed system achieves comparable performance as the DCCRN and DCCRN-VAE baselines on the matched DNS3 dataset but outperforms the baselines on mismatched datasets (WSJ0-QUT, Voicebank-DEMEND), demonstrating improved generalization ability. In addition, an ablation study shows that a similar performance can be achieved with classical fine-tuning instead of adversarial training, resulting in a simpler training pipeline.
>
---
#### [replaced 010] PAL: Probing Audio Encoders via LLMs -- Audio Information Transfer into LLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于多模态融合任务，旨在解决音频信息高效注入大语言模型的问题。提出LAL和PAL方法，提升效率并减少计算开销。**

- **链接: [https://arxiv.org/pdf/2506.10423v3](https://arxiv.org/pdf/2506.10423v3)**

> **作者:** Tony Alex; Wish Suharitdamrong; Sara Atito; Armin Mustafa; Philip J. B. Jackson; Imran Razzak; Muhammad Awais
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** Integration of audio perception into large language models (LLMs) is an emerging research area for enabling machine listening applications, yet efficient transfer of rich audio semantics from audio encoders to LLMs remains underexplored. The most widely used integration paradigm projects audio-encoder output tokens into the LLM input space (e.g., via an MLP or a Q-Former) and then prepends or inserts them into the text token sequence. We refer to this generic scheme as Prepend to the LLM's input token space (PLITS) integration. We propose an efficient alternative, Lightweight Audio LLM Integration (LAL). LAL injects audio representations solely through the attention mechanism at selected LLM layers, bypassing the feed-forward module. It encodes rich audio semantics at an appropriate level of abstraction for integration into different transformer blocks, substantially reducing computational overhead compared to existing approaches. We further introduce PAL, a hybrid integration approach for efficiently Probing Audio encoders via LLM. PAL applies PLITS only to a compact set of summary tokens while integrating the full audio token sequence via LAL. Under an identical training curriculum, LAL consistently matches or outperforms existing integration approaches across multiple base LLMs and tasks, with improvements of up to 30% over a strong PLITS baseline, while reducing memory usage by about 60% and increasing throughput by about 190%. Moreover, PAL matches or exceeds PLITS performance while offering substantially better computational and memory efficiency.
>
---
#### [replaced 011] Trade-offs between structural richness and communication efficiency in music network representations
- **分类: physics.soc-ph; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文属于音乐结构分析任务，探讨特征选择对音乐网络表示中结构丰富性与通信效率的权衡。通过比较不同编码方式，揭示特征影响网络拓扑和感知误差的关系。**

- **链接: [https://arxiv.org/pdf/2509.14053v2](https://arxiv.org/pdf/2509.14053v2)**

> **作者:** Lluc Bono Rosselló; Robert Jankowski; Hugues Bersini; Marián Boguñá; M. Ángeles Serrano
>
> **摘要:** Music is a structured and perceptually rich sequence of sounds in time with well-defined symbolic features, whose perception is shaped by the interplay of expectation and uncertainty. Network science offers a powerful framework for studying its structural organization and communication efficiency. However, it remains unclear how feature selection affects the properties of reconstructed networks and perceptual alignment. Here, we systematically compare eight encodings of musical sequences, ranging from single-feature descriptions to richer multi-feature combinations. We show that representational choices fundamentally shape network topology, the distribution of uncertainty, and the estimated communication efficiency under perceptual constraints. Single-feature representations compress sequences into dense transition structures that support efficient communication, yielding high entropy rates with low modeled perceptual error, but they discard structural richness. By contrast, multi-feature representations preserve descriptive detail and structural specificity, expanding the state space and producing sharper transition profiles and lower entropy rates, which leads to higher modeled perceptual error. Across representations, we found that uncertainty increasingly concentrates in nodes with higher diffusion-based centrality while their perceptual error remains low, unveiling an interplay between predictable structure and localized surprise. Together, these results show that feature choice directly shapes music network representation, describing trade-offs between descriptive richness and communication efficiency and suggesting structural conditions that may support efficient learning and prediction.
>
---
#### [replaced 012] Music Plagiarism Detection: Problem Formulation and a Segment-based Solution
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音乐抄袭检测任务，旨在明确任务定义并解决检测问题。研究提出基于段落的解决方案，并发布数据集与代码。**

- **链接: [https://arxiv.org/pdf/2601.21260v2](https://arxiv.org/pdf/2601.21260v2)**

> **作者:** Seonghyeon Go; Yumin Kim
>
> **摘要:** Recently, the problem of music plagiarism has emerged as an even more pressing social issue. As music information retrieval research advances, there is a growing effort to address issues related to music plagiarism. However, many studies, including our previous work, have conducted research without clearly defining what the music plagiarism detection task actually involves. This lack of a clear definition has slowed research progress and made it hard to apply results to real-world scenarios. To fix this situation, we defined how Music Plagiarism Detection is different from other MIR tasks and explained what problems need to be solved. We introduce the Similar Music Pair dataset to support this newly defined task. In addition, we propose a method based on segment transcription as one way to solve the task. Our demo and dataset are available at https://github.com/Mippia/ICASSP2026-MPD.
>
---
#### [replaced 013] Evaluating Spatialized Auditory Cues for Rapid Attention Capture in XR
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文研究空间音频在XR中快速引导注意力的可行性，解决如何在短时间内通过听觉线索传递方向信息的问题。通过实验评估空间音频的准确性及短期训练的影响。**

- **链接: [https://arxiv.org/pdf/2601.21264v2](https://arxiv.org/pdf/2601.21264v2)**

> **作者:** Yoonsang Kim; Swapnil Dey; Arie Kaufman
>
> **备注:** 8 pages, 4 figures. This is the author's version of the article that will appear at the IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (IEEE VRW) 2026
>
> **摘要:** In time-critical eXtended reality (XR) scenarios where users must rapidly reorient their attention to hazards, alerts, or instructions while engaged in a primary task, spatial audio can provide an immediate directional cue without occupying visual bandwidth. However, such scenarios can afford only a brief auditory exposure, requiring users to interpret sound direction quickly and without extended listening or head-driven refinement. This paper reports a controlled exploratory study of rapid spatial-audio localization in XR. Using HRTF-rendered broadband stimuli presented from a semi-dense set of directions around the listener, we quantify how accurately users can infer coarse direction from brief audio alone. We further examine the effects of short-term visuo-auditory feedback training as a lightweight calibration mechanism. Our findings show that brief spatial cues can convey coarse directional information, and that even short calibration can improve users' perception of aural signals. While these results highlight the potential of spatial audio for rapid attention guidance, they also show that auditory cues alone may not provide sufficient precision for complex or high-stakes tasks, and that spatial audio may be most effective when complemented by other sensory modalities or visual cues, without relying on head-driven refinement. We leverage this study on spatial audio as a preliminary investigation into a first-stage attention-guidance channel for wearable XR (e.g., VR head-mounted displays and AR smart glasses), and provide design insights on stimulus selection and calibration for time-critical use.
>
---
#### [replaced 014] Towards Automatic Evaluation and High-Quality Pseudo-Parallel Dataset Construction for Audio Editing: A Human-in-the-Loop Method
- **分类: cs.SD**

- **简介: 该论文属于音频编辑任务，旨在解决缺乏高质量数据集和评估指标的问题。提出AuditScore和AuditEval，结合专家知识进行自动评估与高质量伪并行数据构建。**

- **链接: [https://arxiv.org/pdf/2508.11966v2](https://arxiv.org/pdf/2508.11966v2)**

> **作者:** Yuhang Jia; Hui Wang; Xin Nie; Yujie Guo; Lianru Gao; Yong Qin
>
> **摘要:** Audio editing aims to manipulate audio content based on textual descriptions, supporting tasks such as adding, removing, or replacing audio events. Despite recent progress, the lack of high-quality benchmark datasets and comprehensive evaluation metrics remains a major challenge for both assessing audio editing quality and improving the task itself. In this work, we propose a novel approach for audio editing task by incorporating expert knowledge into both the evaluation and dataset construction processes: 1) First, we establish AuditScore, the first comprehensive dataset for subjective evaluation of audio editing, consisting of over 6,300 edited samples generated from 7 representative audio editing frameworks and 23 system configurations. Each sample is annotated by professional raters on three key aspects of audio editing quality: overall Quality, Relevance to editing intent, and Faithfulness to original features. 2) Based on this dataset, we systematically propose AuditEval, a family of automatic MOS-style evaluators tailored for audio editing, covering both SSL-based and LLM-based approaches. It addresses the lack of effective objective metrics and the prohibitive cost of subjective evaluation in this field. 3) We further leverage AuditEval to evaluate and filter a large amount of synthetically mixed editing pairs, mining a high-quality pseudo-parallel subset by selecting the most plausible samples. Comprehensive experiments validate that our expert-informed filtering strategy effectively yields higher-quality data, while also exposing the limitations of traditional objective metrics and the advantages of AuditEval. The dataset, codes and tools can be found at: https://github.com/NKU-HLT/AuditEval.
>
---
#### [replaced 015] Investigation of Speech and Noise Latent Representations in Single-channel VAE-based Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决单通道语音增强中语音与噪声区分不清的问题。通过研究VAE生成的语音和噪声潜在表示，发现清晰分离的潜在空间能提升性能。**

- **链接: [https://arxiv.org/pdf/2508.05293v2](https://arxiv.org/pdf/2508.05293v2)**

> **作者:** Jiatong Li; Simon Doclo
>
> **备注:** Accepted by ITG2025
>
> **摘要:** Recently, a variational autoencoder (VAE)-based single-channel speech enhancement system using Bayesian permutation training has been proposed, which uses two pretrained VAEs to obtain latent representations for speech and noise. Based on these pretrained VAEs, a noisy VAE learns to generate speech and noise latent representations from noisy speech for speech enhancement. Modifying the pretrained VAE loss terms affects the pretrained speech and noise latent representations. In this paper, we investigate how these different representations affect speech enhancement performance. Experiments on the DNS3, WSJ0-QUT, and VoiceBank-DEMAND datasets show that a latent space where speech and noise representations are clearly separated significantly improves performance over standard VAEs, which produce overlapping speech and noise representations.
>
---
#### [replaced 016] Game-Time: Evaluating Temporal Dynamics in Spoken Language Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决口语语言模型在时间动态上的不足。提出Game-Time基准，评估模型在时间管理、节奏和同步响应方面的能力。**

- **链接: [https://arxiv.org/pdf/2509.26388v2](https://arxiv.org/pdf/2509.26388v2)**

> **作者:** Kai-Wei Chang; En-Pei Hu; Chun-Yi Kuan; Wenze Ren; Wei-Chih Chen; Guan-Ting Lin; Yu Tsao; Shao-Hua Sun; Hung-yi Lee; James Glass
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Conversational Spoken Language Models (SLMs) are emerging as a promising paradigm for real-time speech interaction. However, their capacity of temporal dynamics, including the ability to manage timing, tempo and simultaneous speaking, remains a critical and unevaluated challenge for conversational fluency. To address this gap, we introduce the Game-Time Benchmark, a framework to systematically assess these temporal capabilities. Inspired by how humans learn a language through language activities, Game-Time consists of basic instruction-following tasks and advanced tasks with temporal constraints, such as tempo adherence and synchronized responses. Our evaluation of diverse SLM architectures reveals a clear performance disparity: while state-of-the-art models handle basic tasks well, many contemporary systems still struggle with fundamental instruction-following. More critically, nearly all models degrade substantially under temporal constraints, exposing persistent weaknesses in time awareness and full-duplex interaction. The Game-Time Benchmark provides a foundation for guiding future research toward more temporally-aware conversational AI. Demos and datasets are available on our project website https://ga642381.github.io/Game-Time.
>
---
#### [replaced 017] Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音大模型偏见评估任务，旨在检验MCQA基准的泛化能力。通过微调模型并测试其在不同任务中的表现，发现MCQA结果无法可靠预测其他任务表现。**

- **链接: [https://arxiv.org/pdf/2510.01254v2](https://arxiv.org/pdf/2510.01254v2)**

> **作者:** Shree Harsha Bokkahalli Satish; Gustav Eje Henter; Éva Székely
>
> **备注:** 5 pages, 2 Figures, Accepted to IEEE ICASSP 2026
>
> **摘要:** Recent work in benchmarking bias and fairness in speech large language models (SpeechLLMs) has relied heavily on multiple-choice question answering (MCQA) formats. The model is tasked to choose between stereotypical, anti-stereotypical, or neutral/irrelevant answers given an input speech prompt and an optional text prompt. Such MCQA benchmarks implicitly assume that model performance is consistent across other MCQA tasks, voices, and other task formats such as more realistic, long-form evaluations. In this paper, we probe that assumption. We fine-tune three SpeechLLMs using LoRA adapters to induce specific MCQA behaviours: preference for stereotypical, anti-stereotypical, or neutral/uncertain answers. We then evaluate whether these behaviours generalise to another, distinct MCQA benchmark, and more critically to long-form, creative generation tasks. Our results show that performance on MCQA bias benchmarks fails to reliably predict performances across other MCQA benchmarks, and more importantly across long-form tasks. We conclude that current MCQA bias benchmarks show limited evidence of cross-task generalisation in the speech domain, and also propose an evaluation suite for measuring behaviour transferability in future models and benchmarks.
>
---
#### [replaced 018] Attention Isn't All You Need for Emotion Recognition:Domain Features Outperform Transformers on the EAV Dataset
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于情感识别任务，研究复杂注意力机制在小数据集上的效果。实验表明，简单领域特征优于复杂模型，强调领域知识的重要性。**

- **链接: [https://arxiv.org/pdf/2601.22161v2](https://arxiv.org/pdf/2601.22161v2)**

> **作者:** Anmol Guragain
>
> **备注:** 2 figures, 10 Pages
>
> **摘要:** We present a systematic study of multimodal emotion recognition using the EAV dataset, investigating whether complex attention mechanisms improve performance on small datasets. We implement three model categories: baseline transformers (M1), novel factorized attention mechanisms (M2), and improved CNN baselines (M3). Our experiments show that sophisticated attention mechanisms consistently underperform on small datasets. M2 models achieved 5 to 13 percentage points below baselines due to overfitting and destruction of pretrained features. In contrast, simple domain-appropriate modifications proved effective: adding delta MFCCs to the audio CNN improved accuracy from 61.9% to 65.56% (+3.66pp), while frequency-domain features for EEG achieved 67.62% (+7.62pp over the paper baseline). Our vision transformer baseline (M1) reached 75.30%, exceeding the paper's ViViT result (74.5%) through domain-specific pretraining, and vision delta features achieved 72.68% (+1.28pp over the paper CNN). These findings demonstrate that for small-scale emotion recognition, domain knowledge and proper implementation outperform architectural complexity.
>
---
#### [replaced 019] CAARMA: Class Augmentation with Adversarial Mixup Regularization
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出CAARMA框架，解决说话人验证中的类不平衡问题，通过生成合成类别提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2503.16718v4](https://arxiv.org/pdf/2503.16718v4)**

> **作者:** Massa Baali; Xiang Li; Hao Chen; Syed Abdul Hannan; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. The models performing inference must hence naturally generate embeddings that cluster same-class instances compactly, while maintaining separation across classes. In order to learn to do so, they are typically trained on a large number of classes (speakers), often using specialized losses. However real-world speaker datasets often lack the class diversity needed to effectively learn this in a generalizable manner. We introduce CAARMA, a class augmentation framework that addresses this problem by generating synthetic classes through data mixing in the embedding space, expanding the number of training classes. To ensure the authenticity of the synthetic classes we adopt a novel adversarial refinement mechanism that minimizes categorical distinctions between synthetic and real classes. We evaluate CAARMA on multiple speaker verification tasks, as well as other representative zero-shot comparison-based speech analysis tasks and obtain consistent improvements: our framework demonstrates a significant improvement of 8\% over all baseline models. The code is available at: https://github.com/massabaali7/CAARMA/
>
---
#### [replaced 020] ConceptCaps: a Distilled Concept Dataset for Interpretability in Music Models
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出ConceptCaps数据集，解决音乐模型可解释性问题。通过分离语义建模与文本生成，提升音频-文本对齐和概念控制能力。**

- **链接: [https://arxiv.org/pdf/2601.14157v2](https://arxiv.org/pdf/2601.14157v2)**

> **作者:** Bruno Sienkiewicz; Łukasz Neumann; Mateusz Modrzejewski
>
> **摘要:** Concept-based interpretability methods like TCAV require clean, well-separated positive and negative examples for each concept. Existing music datasets lack this structure: tags are sparse, noisy, or ill-defined. We introduce ConceptCaps, a dataset of 23k music-caption-audio triplets with explicit labels from a 200-attribute taxonomy. Our pipeline separates semantic modeling from text generation: a VAE learns plausible attribute co-occurrence patterns, a fine-tuned LLM converts attribute lists into professional descriptions, and MusicGen synthesizes corresponding audio. This separation improves coherence and controllability over end-to-end approaches. We validate the dataset through audio-text alignment (CLAP), linguistic quality metrics (BERTScore, MAUVE), and TCAV analysis confirming that concept probes recover musically meaningful patterns. Dataset and code are available online.
>
---
#### [replaced 021] The T12 System for AudioMOS Challenge 2025: Audio Aesthetics Score Prediction System Using KAN- and VERSA-based Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出T12系统，用于音频美学评分预测，解决音频质量评估问题。采用KAN和VERSA模型，提升评分准确性。**

- **链接: [https://arxiv.org/pdf/2512.05592v2](https://arxiv.org/pdf/2512.05592v2)**

> **作者:** Katsuhiko Yamamoto; Koichi Miyazaki; Shogo Seki
>
> **备注:** Accepted to IEEE ASRU 2025. We also released the inference model of the proposed KAN-based predictor. https://github.com/CyberAgentAILab/aesca
>
> **摘要:** We propose an audio aesthetics score (AES) prediction system by CyberAgent (AESCA) for AudioMOS Challenge 2025 (AMC25) Track 2. The AESCA comprises a Kolmogorov--Arnold Network (KAN)-based audiobox aesthetics and a predictor from the metric scores using the VERSA toolkit. In the KAN-based predictor, we replaced each multi-layer perceptron layer in the baseline model with a group-rational KAN and trained the model with labeled and pseudo-labeled audio samples. The VERSA-based predictor was designed as a regression model using extreme gradient boosting, incorporating outputs from existing metrics. Both the KAN- and VERSA-based models predicted the AES, including the four evaluation axes. The final AES values were calculated using an ensemble model that combined four KAN-based models and a VERSA-based model. Our proposed T12 system yielded the best correlations among the submitted systems, in three axes at the utterance level, two axes at the system level, and the overall average. We also released the inference model of the proposed KAN-based predictor (KAN #1-#4).
>
---
#### [replaced 022] Neural acoustic multipole splatting for room impulse response synthesis
- **分类: eess.AS**

- **简介: 该论文属于声学场景建模任务，旨在解决任意接收位置的房间脉冲响应合成问题。通过神经声学多极子点云和网络学习，实现高效准确的RIR生成。**

- **链接: [https://arxiv.org/pdf/2509.17410v2](https://arxiv.org/pdf/2509.17410v2)**

> **作者:** Geonwoo Baek; Jung-Woo Choi
>
> **备注:** 5 pages, 5 figures, accepted to ICASSP 2026
>
> **摘要:** Room Impulse Response (RIR) prediction at arbitrary receiver positions is essential for practical applications such as spatial audio rendering. We propose Neural Acoustic Multipole Splatting (NAMS), which synthesizes RIRs at unseen receiver positions by learning the positions of neural acoustic multipoles and predicting their emitted signals and directivities using a neural network. Representing sound fields through a combination of multipoles offers sufficient flexibility to express complex acoustic scenes while adhering to physical constraints such as the Helmholtz equation. We also introduce a pruning strategy that starts from a dense splatting of neural acoustic multipoles and progressively eliminates redundant ones during training. Experiments conducted on both real and synthetic datasets indicate that the proposed method surpasses previous approaches on most metrics while maintaining rapid inference. Ablation studies reveal that multipole splatting with pruning achieves better performance than the monopole model with just 20% of the poles.
>
---
