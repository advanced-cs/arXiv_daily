# 音频 cs.SD;  eess.AS

- **最新发布 9 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Academic Text-to-Music Grand Challenge: Datasets, Baselines, and Evaluation Methods
- **分类: cs.SD**

- **简介: 该论文介绍学术文本到音乐生成挑战，解决工业模型垄断学术研究的问题，提供公开数据集和评估方法，促进学术研究。**

- **链接: [https://arxiv.org/pdf/2605.21538](https://arxiv.org/pdf/2605.21538)**

> **作者:** Fang-Chih Hsieh; Wei-Jaw Lee; Chun-Ping Wang; Hung-yi Lee; Hao-Wen Dong; Yi-Hsuan Yang
>
> **备注:** Accepted to IEEE ICME 2026 Grand Challenge Paper
>
> **摘要:** This paper presents an overview and the technical framework of the ICME 2026 Grand Challenge on Academic Text-to-Music Generation (ATTM). Despite the rapid progress in text-to-music generation (TTM) systems, the field is currently dominated by models trained on massive proprietary datasets with industrial-scale computational resources, creating a significant barrier for academic research. To address this, the ATTM Challenge establishes a fair-play benchmark that requires participants to train generative models strictly from scratch using a standardized, CC-licensed subset of the MTG-Jamendo dataset containing only instrumental music. The challenge is divided into two tracks: the Efficiency Track (limited to 500M parameters) and the Performance Track (no parameter limit). Submissions are evaluated through a multi-stage process involving objective metrics, including Frechet Audio Distance, CLAP score, and a novel Concept Coverage Score (CCS), followed by a subjective listening test. By providing open-source baselines, preprocessing pipelines, reference captions, and public evaluation code for computing FAD and CLAP, this challenge aims to facilitate and promote TTM research in academic contexts.
>
---
#### [new 002] Real-time, EDM-inspired sonfication of the activity of a supercomputer
- **分类: cs.SD**

- **简介: 论文探讨实时超计算机活动的音乐化表达，属于数据声学任务。旨在通过EDM风格实现持续、可理解的系统监控，解决长期数据监听中的音乐一致性问题。**

- **链接: [https://arxiv.org/pdf/2605.21874](https://arxiv.org/pdf/2605.21874)**

> **作者:** Marco Alunno; Paolo Bientinesi
>
> **备注:** 7 pages, 2 figures, accepted conference paper
>
> **摘要:** The project described in this paper explores the informative sonification of data received in real time from a supercomputer. These data capture the current activities in all the nodes of the computer, therefore, their sonification functions as a form of continuous monitoring of the nodes' behavior and, by extension, of the system as a whole. Because such monitoring is theoretically unending, the resulting sonification must be musically capable of conveying information through sound in a way that remains both intelligible and engaging over long durations. Rather than imposing a predefined musical style onto the data, we sought to identify one which the data themselves could plausibly support. From a small set of candidates, we selected EDM because it is a family of genres whose structural and temporal characteristics align well with continuous, data-driven processes and long-term listening. Through this style-based approach, this research builds on the long tradition of computer data sonification while uniquely combining three elements rarely addressed together: monitoring (rather than debugging) as the primary goal, real-time (rather than post-mortem) data interpretation, and generation of virtually infinite and stylistically coherent (rather than incongruous) music.
>
---
#### [new 003] Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于用户自定义关键词检测任务，解决混淆词区分、跨说话人性能不稳定和数据成本高的问题。提出DMA-KWS框架，结合双阶段匹配、多模态注册和持续适应机制，提升检测效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.22120](https://arxiv.org/pdf/2605.22120)**

> **作者:** Zhiqi Ai; Han Cheng; Shiyi Mu; Xinnuo Li; Yongjin Zhou; Shugong Xu
>
> **备注:** 14 pages, 13 figures, 12 tables. Accepted by TASLP
>
> **摘要:** User-defined keyword spotting (KWS) is crucial for personalized voice interaction, yet existing methods face several challenges: (1) insufficient discriminability among confusable words, (2) performance inconsistency across speakers with varying pronunciations, and (3) high data cost to ensure reliable wake-word performance. In this paper, we introduce DMA-KWS, an efficient and robust framework for user-defined keyword spotting. First, it adopts a dual-stage matching pipeline: CTC decoding with streaming phoneme search to locate candidate segments, followed by QbyT with a phoneme matcher for fine-grained verification, enabling it to better distinguish confusable words. Next, multi-modal enrollment fuses user-specific speech with text embeddings to further improve accuracy for registered users. Finally, a parameter-efficient continual adaptation mechanism performs lightweight updates using synthetic and real data. Extensive experiments demonstrate the superior performance of DMA-KWS. On the LibriPhrase Hard subset, it achieves 97.85% AUC and 6.13% EER, reaching state-of-the-art performance. In speaker-dependent settings, DMA-KWS consistently outperforms text-only enrollment, demonstrating significant performance gains. Moreover, the proposed parameter-efficient fine-tuning mechanism adapts DMA-KWS with only 187k updated parameters, further enhancing KWS performance while ensuring suitability for on-device deployment.
>
---
#### [new 004] RobustSpeechFlow: Learning Robust Text-to-Speech Trajectories via Augmentation-based Contrastive Flow Matching
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决对齐不准确导致的错漏问题。通过增强对比流匹配，提升模型鲁棒性，减少错误率。**

- **链接: [https://arxiv.org/pdf/2605.22083](https://arxiv.org/pdf/2605.22083)**

> **作者:** Jinhyeok Yang; Hyeongju Kim; Yechan Yu; Joon Byun; Frederik Bous; Juheon Lee
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** While flow-matching text-to-speech (TTS) achieves strong zero-shot speaker similarity and naturalness, it remains susceptible to content fidelity issues, particularly skip and repeat errors from imperfect alignment. We propose RobustSpeechFlow, a training strategy that improves alignment robustness by extending contrastive flow matching with length-preserving repeat and skip latent augmentations. Requiring no external aligners or preference data, our method directly penalizes realistic failure modes and readily integrates into existing pipelines. On Seed-TTS-eval, it reduces the word error rate (WER) from 1.44 to 1.38 using only 0.06B parameters. On our ZERO500 benchmark, it delivers consistent intelligibility improvements across diverse speaker and prosody conditions; at NFE=24, it reduces English character error rate (CER) from 0.48\% to 0.35\% and Korean CER from 0.81\% to 0.57\%. Audio samples: this https URL
>
---
#### [new 005] Neighbor-Consistent Neural Filters for Robust Personal Sound Zones Under Localization Uncertainty
- **分类: eess.AS**

- **简介: 该论文属于声学信号处理任务，旨在解决定位不确定性下的个人音区稳定性问题。通过引入邻域一致性正则化，减少滤波器变化，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.21891](https://arxiv.org/pdf/2605.21891)**

> **作者:** Hao Jiang; Edgar Choueiri
>
> **摘要:** Coordinate-conditioned neural networks can generate head-tracked personal sound zone (PSZ) loudspeaker filters in real time, but they are sensitive to localization uncertainty. Small fluctuations in estimated listener coordinates, caused by optical distortion, temporary occlusions, or tracking jitter, may produce large filter changes even when listeners are physically stationary. This paper proposes neighbor-consistent neural filters that regularize the coordinate-to-filter mapping by penalizing filter differences at randomly perturbed neighboring coordinates during training. To evaluate robustness against tracking noise, we introduce a decoupled protocol that fixes the acoustic transfer functions at a physical anchor while perturbing only the coordinate inputs used for filter generation. Isolation quality and local stability are evaluated using neighborhood median and lower-tail statistics of inter-zone and inter-program isolation, together with spatial variation rates that quantify metric sensitivity within a coordinate neighborhood. In simulation with a split-band woofer-tweeter system and 25 randomly sampled anchor positions, neighbor consistency reduces the root-mean-square (RMS) variation rate by up to 55.9% in the woofer band and 30.3% in the tweeter band while largely preserving isolation quality and improving lower-tail robustness. In in-situ measurements using a 24-driver array and two stationary head-and-torso simulators, the proposed regularization improves worst-case neighborhood isolation by up to 16.9% and reduces spatial variation rates by up to 61.8%. These results demonstrate that neighbor-consistency regularization effectively stabilizes PSZ rendering under localization uncertainty.
>
---
#### [new 006] Automatic Contextual Audio Denoising
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频去噪任务，解决固定噪声定义导致的误删问题。通过自动识别音频场景上下文，区分有用与无关成分，提升去噪效果。**

- **链接: [https://arxiv.org/pdf/2605.22262](https://arxiv.org/pdf/2605.22262)**

> **作者:** Diep Luong; Konstantinos Drossos; Mikko Heikkinen; Tuomas Virtanen
>
> **摘要:** Audio context determines which sound components and sources are relevant and which can be perceived as irrelevant (noise) by listeners. For example, traffic noise is informative in urban surveillance but noise for a phone call at the same location. Most current audio denoising systems apply fixed target-noise definitions, often removing useful components in one context while failing to suppress irrelevant components. To address this, we introduce the concept automatic contextual audio denoising (ACAD) which defines target and noise based on the inferred context. In this work, we restrict context to be associated with an acoustic scene class. We label sound events outside the event distribution of a scene class (noise) as out-of-context (OC) and events typical for that scene as in-context (IC). We implement a deep learning method that automatically infers the context of the audio signal and removes OC components, and benchmark it against variants: without context inference, with oracle context, and with separately provided uninformative context. On paired clean/noisy data across diverse contexts, where OC components in one context may be IC in another, our proposed method outperforms other approaches across standard objective metrics, indicating that the model can infer context and context-dependent processing can enhance denoising.
>
---
#### [new 007] Live Music Diffusion Models: Efficient Fine-Tuning and Post-Training of Interactive Diffusion Music Generators
- **分类: cs.SD; cs.AI; cs.LG; cs.MM**

- **简介: 该论文研究如何将音频扩散模型转化为高效的实时音乐生成系统，解决传统模型计算效率低的问题，提出LMDMs方法，提升推理速度并支持实时互动创作。**

- **链接: [https://arxiv.org/pdf/2605.22717](https://arxiv.org/pdf/2605.22717)**

> **作者:** Zachary Novack; Stephen Brade; Haven Kim; Hugo Flores García; Nithya Shikarpur; Chinmay Talegaonkar; Suwan Kim; Valerie K. Chen; Julian McAuley; Taylor Berg-Kirkpatrick; Cheng-Zhi Anna Huang
>
> **摘要:** Interactive streaming music generation promises the use of generative models for live performance and co-creation that is impossible with offline models. However, SOTA models exist in the discrete-AR regime, requiring industrial levels of compute for both training and inference. In this work, we investigate whether audio diffusion models, with their wide support in the open-source community but non-streaming bidirectional nature, can be repurposed efficiently into interactive models accessible on consumer hardware. By taking a critical look at the modern pipeline for block-wise outpainting diffusion, we identify critical inefficiencies during inference that result in strictly worse computational efficiency than their discrete-AR counterparts. We propose Live Music Diffusion Models (LMDMs), a simple modification of the generative diffusion process that recovers, and then outperforms, the inference complexity of the discrete Live Music Models (LMMs) through block-wise KV Caching. Unlike LMMs, LMDMs further enable stable post-training alignment through our novel ARC-Forcing paradigm, reducing error accumulation without any explicit RL or reward models. We demonstrate the application of LMDMs in a number of creative domains, including text-conditioned generation, sketch-based music synthesis, and jamming. We finally show how LMDMs can be used as a generative instrument in a real artist-AI collaboration, utilizing LMDMs as a "generative delay" to transform musicians' improvisation live for variable timbral effects while running locally on a consumer gaming laptop.
>
---
#### [new 008] Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models
- **分类: cs.AI; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于政治演讲中的情感分析任务，旨在比较声学模型与大语言模型在路径学分析中的效果，解决情感识别准确性问题。**

- **链接: [https://arxiv.org/pdf/2605.22732](https://arxiv.org/pdf/2605.22732)**

> **作者:** Juergen Dietrich
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** We investigate whether acoustic emotion recognition models can serve as proxies for the Pathos dimension in political speech analysis, as operationalised by the TRUST multi-agent large language model (LLM) pipeline. Using a Bundestag plenary speech by Felix Banaszak (51 segments, 245 s) as a case study, we compare three analysis modalities: (1) emotion2vec_plus_large, an acoustic speech emotion recognition (SER) model whose continuous Arousal and Valence values are derived via post-hoc Russell Circumplex projection; (2) Gemini 2.5 Flash, an LLM analysing the full speech audio together with its transcript in an open-ended, context-aware fashion; and (3) TRUST-Pathos scores from a three-advocate LLM supervisor ensemble. Spearman rank correlations reveal that Gemini Valence correlates strongly with TRUST-Pathos (rho = +0.664, p < 0.001), whereas emotion2vec Valence does not (rho = +0.097, p = 0.499). We further demonstrate, via a systematic quality evaluation of the Berlin Database of Emotional Speech (EMO-DB) using Gemini in an open-ended annotation paradigm, that standard SER benchmark corpora suffer from acted speech, cultural bias, and category incompatibility. Our results suggest that LLM-based multimodal analysis captures semantically defined political emotion substantially better than acoustic models alone, while acoustic features remain informative for low-level Arousal estimation. Future work will extend this approach to video-based analysis incorporating facial expression and gaze.
>
---
#### [new 009] Plug-in Losses for Evidential Deep Learning: A Simplified Framework for Uncertainty Estimation that Includes the Softmax Classifier
- **分类: cs.LG; eess.AS; stat.ML**

- **简介: 该论文属于不确定性估计任务，解决EDL计算复杂的问题。通过引入插件损失简化框架，提升计算效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2605.22746](https://arxiv.org/pdf/2605.22746)**

> **作者:** Berk Hayta; Hannah Laus; Simon Mittermaier; Felix Krahmer
>
> **摘要:** Real-world sensor-based learning systems require uncertainty estimation that is both reliable and computationally efficient. Evidential Deep Learning (EDL) provides single-pass uncertainty estimation by modeling the class probabilities via Dirichlet distributions, where the Dirichlet parameters are predicted by a learned neural network mapping. However, this approach can lead to computational challenges, as Dirichlet expected objectives are more complex than standard supervised learning losses, complicating their analysis and implementation. We address this issue by approximating the objective of the first-order empirical risk minimization problem induced by EDL with a plug-in loss evaluated at the Dirichlet mean and show that, under mild assumptions, the approximation error decays with growing evidence for a broad class of loss functions, including mean-squared error and cross-entropy loss. As a special case, our analysis provides justification for the use of softmax in the context of uncertainty estimation, since under a particular evidence-to-Dirichlet mapping, our framework includes the standard softmax classifier. We validate the proposed simplified objectives on the Google Speech Commands dataset and show that they achieve predictive accuracy and selective prediction performance comparable to classical EDL, while being simpler to implement using standard deep learning losses and training pipelines. To the best of our knowledge, this empirical analysis is the first to obtain coverage-accuracy trade-offs for speech recognition tasks through EDL.
>
---
## 更新

#### [replaced 001] DASM: Domain-Aware Sharpness Minimization for Multi-Domain Voice Stream Steganalysis
- **分类: cs.CR; cs.SD**

- **简介: 该论文属于语音流隐写分析任务，解决多领域数据分布不一致导致的检测性能下降问题。提出DASM优化器，提升模型泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19955](https://arxiv.org/pdf/2605.19955)**

> **作者:** Pengcheng Zhou; Pianran Guo; Shuhua Chen; Mengqin Zhao; Zhongliang Yang; Linna Zhou
>
> **摘要:** The growing use of information hiding in network streaming media for covert communication poses a significant security threat, necessitating the development of robust detection technologies. However, existing steganalysis methods for network voice streams mostly rely on data distributions in specific scenarios, making it difficult to adapt to the practical detection needs of non-homologous data distributions. Through Hessian analysis, we find that the loss landscapes of mainstream models are dominated by numerous saddle points and sharp local minima, rendering them highly sensitive to data distribution shifts and fundamentally limiting generalization. Therefore, we propose a new optimizer, Domain-Aware Sharpness Minimization (DASM). The core mechanisms of DASM consist of two aspects: first, it integrates domain-supervised contrastive learning with sharpness-aware optimization, explicitly preserving inter-domain feature separation while seeking flat minima; second, we design an adaptive domain gap modulation strategy that dynamically calibrates the optimization loss weights by sensing the real-time feature separability of different domains. Extensive experimental results demonstrate that our method outperforms the state-of-the-art methods by a large margin and achieves excellent generalization and robustness.
>
---
#### [replaced 002] A strongly annotated passive acoustic dataset for tropical bird monitoring
- **分类: cs.SD; cs.CV**

- **简介: 该论文提出PteroSet数据集，用于热带鸟类监测。解决监督学习所需标注数据稀缺的问题，包含大量音频标注，支持机器学习任务。**

- **链接: [https://arxiv.org/pdf/2605.20578](https://arxiv.org/pdf/2605.20578)**

> **作者:** Daniela Ruiz; Juan Sebastián Ulloa; Zhongqi Miao; Nicolás Betancourt; Maria Paula Toro-Gómez; Andrés Hernández; Bruno Demuro; Eliana Barona-Cortés; Angela Mendoza-Henao; Andrés Sierra-Ricaurte; Sebastián Pérez-Peña; Rahul Dodhia; Pablo Arbeláez; Juan M. Lavista Ferres
>
> **摘要:** Passive acoustic monitoring enables continuous, non-invasive biodiversity assessment across diverse ecosystems. The scale of these datasets has driven the adoption of machine learning, with supervised approaches showing strong performance. However, supervised methods require time-resolved annotated datasets, which remain scarce, especially in complex tropical soundscapes. We present PteroSet, a curated dataset of strongly annotated Neotropical bird vocalizations recorded in Puerto Asis (Putumayo) and Pivijay (Magdalena), Colombia, between 2023 and 2025. The dataset comprises 563 recordings (73.62 h) and 15,372 time-frequency annotations, including 6,702 events identified to the species level across 168 species. We release the annotations in a COCO-inspired JSON schema that unifies audio files, taxonomic categories, and labels for machine learning workflows. Beyond providing annotated data, PteroSet serves as a realistic benchmark that highlights key characteristics of tropical soundscapes, including acoustic co-occurrence and domain shift across recording sites. We provide a deep learning baseline for binary bird detection, demonstrating PteroSet's usability and the challenges it presents.
>
---
#### [replaced 003] Quantizing Whisper-small: How design choices affect ASR performance
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究语音识别模型Whisper-small的量化方法，旨在解决其在边缘设备部署中的计算需求问题。通过对比不同量化方案，找到提升性能与压缩模型的最佳方法。**

- **链接: [https://arxiv.org/pdf/2511.08093](https://arxiv.org/pdf/2511.08093)**

> **作者:** Arthur Söhler; Julian Irigoyen; Andreas Søeborg Kirkedal
>
> **备注:** Accepted to SPEAKABLE workshop at LREC 2026
>
> **摘要:** Large speech recognition models like Whisper-small achieve high accuracy but are difficult to deploy on edge devices due to their high computational demand. To this end, we present a unified, cross-library evaluation of post-training quantization (PTQ) on Whisper-small that disentangles the impact of quantization scheme, method, granularity, and bit-width. Our study is based on four libraries: PyTorch, Optimum-Quanto, HQQ, and bitsandbytes. Experiments on LibriSpeech test-clean and test-other show that dynamic int8 quantization with Quanto offers the best trade-off, reducing model size by 57% while improving on the baseline's word error rate. Static quantization performed worse, likely due to Whisper's Transformer architecture, while more aggressive formats (e.g., nf4, int3) achieved up to 71% compression at the cost of accuracy in noisy conditions. Overall, our results demonstrate that carefully chosen PTQ methods can substantially reduce model size and inference cost without retraining, enabling efficient deployment of Whisper-small on constrained hardware.
>
---
#### [replaced 004] Go witheFlow: Real-time Emotion Driven Audio Effects Modulation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐与情感计算任务，旨在解决机器缺乏情感表达的问题。通过提取生物信号和音频特征，实现实时音频效果调制。**

- **链接: [https://arxiv.org/pdf/2510.02171](https://arxiv.org/pdf/2510.02171)**

> **作者:** Edmund Dervakos; Spyridon Kantarelis; Vassilis Lyberatos; Jason Liartis; Giorgos Stamou
>
> **备注:** Accepted at NeurIPS Creative AI Track 2025: Humanity
>
> **摘要:** Music performance is a distinctly human activity, intrinsically linked to the performer's ability to convey, evoke, or express emotion. Machines cannot perform music in the human sense; they can produce, reproduce, execute, or synthesize music, but they lack the capacity for affective or emotional experience. As such, music performance is an ideal candidate through which to explore aspects of collaboration between humans and machines. In this paper, we introduce the witheFlow system, designed to enhance real-time music performance by automatically modulating audio effects based on features extracted from both biosignals and the audio itself. The system, currently in a proof-of-concept phase, is designed to be lightweight, able to run locally on a laptop, and is open-source given the availability of a compatible Digital Audio Workstation and sensors.
>
---
#### [replaced 005] Towards Open World Sound Event Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于声事件检测任务，解决传统系统在开放环境中的局限性。提出OW-SED框架，实现已知事件检测、未知事件识别与增量学习。**

- **链接: [https://arxiv.org/pdf/2605.03934](https://arxiv.org/pdf/2605.03934)**

> **作者:** P.H.Hai; L.T.Minh; L.H.Son
>
> **备注:** 32 pages, 3 figures. Accepted to Signal Processing (Elsevier)
>
> **摘要:** Sound Event Detection (SED) plays a vital role in audio understanding, with applications in surveillance, smart cities, healthcare, and multimedia indexing. However, conventional SED systems operate under a closed-world assumption, limiting their effectiveness in real-world environments where novel acoustic events frequently emerge. Inspired by the success of open-world learning in computer vision, we introduce the Open-World Sound Event Detection (OW-SED) paradigm, where models must detect known events, identify unseen ones, and incrementally learn from them. To tackle the unique challenges of OW-SED, such as overlapping and ambiguous events, we propose a 1D Deformable architecture that leverages deformable attention to adaptively focus on salient temporal regions. Furthermore, we design a novel Open-World Deformable Sound Event Detection Transformer (WOOT) framework incorporating feature disentanglement to separate class-specific and class-agnostic representations, together with a one-to-many matching strategy and a diversity loss to enhance representation diversity. Experimental results demonstrate that our method achieves marginally superior performance compared to existing leading techniques in closed-world settings and significantly improves over existing baselines in open-world scenarios.
>
---
#### [replaced 006] Modulation Feature Enhancement with a Multi-Stage Attention Network for Underwater Acoustic Target Recognition
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于水下声目标识别任务，旨在解决船舶辐射噪声复杂性带来的识别难题。通过引入多阶段注意力机制和改进损失函数，提升特征表示与分类性能。**

- **链接: [https://arxiv.org/pdf/2605.16304](https://arxiv.org/pdf/2605.16304)**

> **作者:** Jiaping Yu; Shefeng Yan; Linlin Mao; Zeping Sui; Chunjin Jiang
>
> **备注:** 31 pages, 14 figures, Accepted by Signal Processing
>
> **摘要:** Underwater acoustic target recognition is critical for maritime applications, yet it faces challenges arising from the complex and diverse nature of ship-radiated noise. To address these issues, we propose a robust deep learning-based framework. First, we introduce a feature extraction and fusion method based on variational mode decomposition (VMD) and the 3/2-D spectrum to generate high-fidelity 2-D DEMON spectral features, which effectively capture modulation envelope information. To further enhance feature representation, we design a one-dimensional convolutional neural network (1-D CNN) integrated with a novel Multi-Stage Multi-Type Attention Mechanism (MMATT) that adaptively refines features at different network depths. Within this mechanism, we propose a Residual Channel-Independent Spectral Attention Mechanism (R-CISAM) and a Multi-Scale Separate-and-Fuse Spectral Attention Mechanism (MS-SFSAM). Moreover, to mitigate performance degradation caused by severe class imbalance inherent in real-world ship-radiated noise data, we devise an Adjustable Class-Balanced Focal Loss (ACBFL), which provides flexibility across tasks with varying degrees of imbalance. Experimental results on a real-world ship-radiated noise dataset demonstrate that the proposed solutions effectively enhance underwater acoustic target recognition performance.
>
---
#### [replaced 007] CoarseSoundNet: Building a reliable model for ecological soundscape analysis
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于 soundscape 分类任务，旨在解决生态声音成分（生物声、自然声、人为声）难以准确量化的问题。工作包括构建 CoarseSoundNet 模型，并验证其在真实 PAM 数据中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.21143](https://arxiv.org/pdf/2605.21143)**

> **作者:** Alexander Gebhard; Andreas Triantafyllopoulos; Dominik Arend; Sandra Müller; Svenja Schmidt; Michael Scherer-Lorenzen; Björn W. Schuller
>
> **备注:** Currently under review
>
> **摘要:** A soundscape is composed of three types of sound: biophony (sounds made by animals), geophony (natural abiotic sounds) and anthropophony (sounds made by humans). A key research question in the field of soundscape ecology is how these components interact with each other, specifically how biophony responds to geophony and anthropophony. Nevertheless, as of today, there are not many analytical instruments that enable the distinct quantification of these elements. Recent machine learning (ML) approaches aim to support automated analysis but often rely on task-specific or clean data, limiting generalisation to noisy passive acoustic monitoring (PAM) recordings. This study presents a clear and reproducible structure to build ML models for coarse soundscape classification and introduces CoarseSoundNet, a deep learning model trained to distinguish biophony, geophony, and anthropophony under realistic PAM conditions. We systematically investigate model architectures, the influence of an additional training class, data composition, and evaluation strategies. Our findings suggest that model performance improves with additional PAM data, especially when similar to the target domain, and by introducing an explicit silence class during training. Class-specific decision thresholds and duration-based constraints further enhance performance, particularly for anthropophony and geophony. Error analyses exhibit challenges for anthropophony due to masking effects and confusions for silence and insect sounds for geophony and biophony. Finally, we conduct an ecological case study which shows that pre-filtering recordings with CoarseSoundNet yields acoustic index trends comparable to ground-truth filtering, supporting its use as an effective preprocessing tool for ecoacoustic analyses.
>
---
#### [replaced 008] Exploring How Audio Effects Alter Emotion with Foundation Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音频效果对情感的影响，属于情感计算任务。旨在解决音频效果如何影响情绪感知的问题，通过基础模型分析音频效果与情绪的关系。**

- **链接: [https://arxiv.org/pdf/2509.15151](https://arxiv.org/pdf/2509.15151)**

> **作者:** Stelios Katsis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **备注:** this https URL
>
> **摘要:** Audio effects (FX) such as reverberation, distortion, modulation, and dynamic range processing play a pivotal role in shaping emotional responses during music listening. While prior studies have examined links between low-level audio features and affective perception, the systematic impact of audio FX on emotion remains underexplored. This work investigates how foundation models - large-scale neural architectures pretrained on multimodal data - can be leveraged to analyze these effects. Such models encode rich associations between musical structure, timbre, and affective meaning, offering a powerful framework for probing the emotional consequences of sound design techniques. By applying various probing methods to embeddings from deep learning models, we examine the complex, nonlinear relationships between audio FX and estimated emotion, uncovering patterns tied to specific effects and evaluating the robustness of foundation audio models. Our findings aim to advance understanding of the perceptual impact of audio production practices, with implications for music cognition, performance, and affective computing.
>
---
#### [replaced 009] OneVoice: One Model, Triple Scenarios-Towards Unified Zero-shot Voice Conversion
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音转换任务，旨在解决多场景下模型碎片化问题。提出OneVoice框架，统一处理语言保留、情感表达和歌唱三种场景，采用MoE结构实现高效建模与灵活控制。**

- **链接: [https://arxiv.org/pdf/2601.18094](https://arxiv.org/pdf/2601.18094)**

> **作者:** Zhichao Wang; Tao Li; Wenshuo Ge; Zihao Cui; Shilei Zhang; Junlan Feng
>
> **摘要:** Recent progress of voice conversion~(VC) has achieved a new milestone in speaker cloning and linguistic preservation. But the field remains fragmented, relying on specialized models for linguistic-preserving, expressive, and singing scenarios. We propose OneVoice, a unified zero-shot framework capable of handling all three scenarios within a single model. OneVoice is built upon a continuous language model trained with VAE-free next-patch diffusion, ensuring high fidelity and efficient sequence modeling. Its core design for unification lies in a Mixture-of-Experts (MoE) designed to explicitly model shared conversion knowledge and scenario-specific expressivity. Expert selection is coordinated by a dual-path routing mechanism, including shared expert isolation and scenario-aware domain expert assignment with global-local cues. For precise conditioning, scenario-specific prosodic features are fused into each layer via a gated mechanism, allowing adaptive usage of prosody information. Furthermore, to enable the core idea and alleviate the imbalanced issue (abundant speech vs. scarce singing), we adopt a two-stage progressive training that includes foundational pre-training and scenario enhancement with LoRA-based domain experts. Experiments show that OneVoice matches or surpasses specialized models across all three scenarios, while verifying flexible control over scenarios and offering a fast decoding version as few as 2 steps. Audio samples are available on demo page.
>
---
