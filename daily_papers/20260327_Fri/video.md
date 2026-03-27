# 计算机视觉 cs.CV

- **最新发布 172 篇**

- **更新 104 篇**

## 最新发布

#### [new 001] AG-EgoPose: Leveraging Action-Guided Motion and Kinematic Joint Encoding for Egocentric 3D Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于第一视角3D人体姿态估计任务，解决因视角限制导致的姿势估计难题。提出AG-EgoPose框架，融合时空信息提升估计精度。**

- **链接: [https://arxiv.org/pdf/2603.25175](https://arxiv.org/pdf/2603.25175)**

> **作者:** Md Mushfiqur Azam; John Quarles; Kevin Desai
>
> **摘要:** Egocentric 3D human pose estimation remains challenging due to severe perspective distortion, limited body visibility, and complex camera motion inherent in first-person viewpoints. Existing methods typically rely on single-frame analysis or limited temporal fusion, which fails to effectively leverage the rich motion context available in egocentric videos. We introduce AG-EgoPose, a novel dual-stream framework that integrates short- and long-range motion context with fine-grained spatial cues for robust pose estimation from fisheye camera input. Our framework features two parallel streams: A spatial stream uses a weight-sharing ResNet-18 encoder-decoder to generate 2D joint heatmaps and corresponding joint-specific spatial feature tokens. Simultaneously, a temporal stream uses a ResNet-50 backbone to extract visual features, which are then processed by an action recognition backbone to capture the motion dynamics. These complementary representations are fused and refined in a transformer decoder with learnable joint tokens, which allows for the joint-level integration of spatial and temporal evidence while maintaining anatomical constraints. Experiments on real-world datasets demonstrate that AG-EgoPose achieves state-of-the-art performance in both quantitative and qualitative metrics. Code is available at: this https URL.
>
---
#### [new 002] Seeing to Ground: Visual Attention for Hallucination-Resilient MDLLMs
- **分类: cs.CV**

- **简介: 该论文针对多模态扩散大语言模型中的幻觉问题，提出VISAGE框架，通过校准解码目标提升视觉 grounding 能力。**

- **链接: [https://arxiv.org/pdf/2603.25711](https://arxiv.org/pdf/2603.25711)**

> **作者:** Vishal Narnaware; Animesh Gupta; Kevin Zhai; Zhenyi Wang; Mubarak Shah
>
> **摘要:** Multimodal Diffusion Large Language Models (MDLLMs) achieve high-concurrency generation through parallel masked decoding, yet the architectures remain prone to multimodal hallucinations. This structural vulnerability stems from an algorithmic flaw: the decoder ranks candidate tokens based on textual likelihood without verifying localized visual support. We establish that this language-only ranking induces an objective mismatch, where language probability mass acts as a misspecified proxy for the intended multimodal task. Consequently, we reinterpret hallucination as a localized optimization error, a phenomenon where the decoder exploits language shortcuts to maximize a proxy score at the expense of visual grounding. To address this objective mismatch, we introduce VISAGE, a training-free decoding framework that calibrates the objective at inference time. VISAGE estimates the proxy discrepancy by quantifying the spatial entropy of cross-attention distributions. By enforcing a localization consensus across attention heads, the method penalizes spatially uniform distributions and re-ranks token commitments to favor visually grounded outcomes. We provide an analytical stability guarantee establishing that VISAGE maintains a bounded objective loss under estimation error. Evaluations across hallucination-sensitive and general-purpose benchmarks demonstrate the robustness of the framework, yielding relative gains of 8.59% on MMMU-val and 7.75% on HallusionBench.
>
---
#### [new 003] MedOpenClaw: Auditable Medical Imaging Agents Reasoning over Uncurated Full Studies
- **分类: cs.CV**

- **简介: 该论文提出MEDOPENCLAW和MEDFLOWBENCH，解决医疗影像中代理模型在真实临床流程中的性能问题，推动可审计的全研究影像分析。**

- **链接: [https://arxiv.org/pdf/2603.24649](https://arxiv.org/pdf/2603.24649)**

> **作者:** Weixiang Shen; Yanzhu Hu; Che Liu; Junde Wu; Jiayuan Zhu; Chengzhi Shen; Min Xu; Yueming Jin; Benedikt Wiestler; Daniel Rueckert; Jiazhen Pan
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Currently, evaluating vision-language models (VLMs) in medical imaging tasks oversimplifies clinical reality by relying on pre-selected 2D images that demand significant manual labor to curate. This setup misses the core challenge of realworld diagnostics: a true clinical agent must actively navigate full 3D volumes across multiple sequences or modalities to gather evidence and ultimately support a final decision. To address this, we propose MEDOPENCLAW, an auditable runtime designed to let VLMs operate dynamically within standard medical tools or viewers (e.g., 3D Slicer). On top of this runtime, we introduce MEDFLOWBENCH, a full-study medical imaging benchmark covering multi-sequence brain MRI and lung CT/PET. It systematically evaluates medical agentic capabilities across viewer-only, tool-use, and open-method tracks. Initial results reveal a critical insight: while state-of-the-art LLMs/VLMs (e.g., Gemini 3.1 Pro and GPT-5.4) can successfully navigate the viewer to solve basic study-level tasks, their performance paradoxically degrades when given access to professional support tools due to a lack of precise spatial grounding. By bridging the gap between static-image perception and interactive clinical workflows, MEDOPENCLAW and MEDFLOWBENCH establish a reproducible foundation for developing auditable, full-study medical imaging agents.
>
---
#### [new 004] TIGFlow-GRPO: Trajectory Forecasting via Interaction-Aware Flow Matching and Reward-Driven Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人类轨迹预测任务，解决现有方法忽视社会规范与场景约束的问题。提出TIGFlow-GRPO框架，结合交互感知与奖励驱动优化，提升轨迹的合理性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.24936](https://arxiv.org/pdf/2603.24936)**

> **作者:** Xuepeng Jing; Wenhuan Lu; Hao Meng; Zhizhi Yu; Jianguo Wei
>
> **摘要:** Human trajectory forecasting is important for intelligent multimedia systems operating in visually complex environments, such as autonomous driving and crowd surveillance. Although Conditional Flow Matching (CFM) has shown strong ability in modeling trajectory distributions from spatio-temporal observations, existing approaches still focus primarily on supervised fitting, which may leave social norms and scene constraints insufficiently reflected in generated trajectories. To address this issue, we propose TIGFlow-GRPO, a two-stage generative framework that aligns flow-based trajectory generation with behavioral rules. In the first stage, we build a CFM-based predictor with a Trajectory-Interaction-Graph (TIG) module to model fine-grained visual-spatial interactions and strengthen context encoding. This stage captures both agent-agent and agent-scene relations more effectively, providing more informative conditional features for subsequent alignment. In the second stage, we perform Flow-GRPO post-training,where deterministic flow rollout is reformulated as stochastic ODE-to-SDE sampling to enable trajectory exploration, and a composite reward combines view-aware social compliance with map-aware physical feasibility. By evaluating trajectories explored through SDE rollout, GRPO progressively steers multimodal predictions toward behaviorally plausible futures. Experiments on the ETH/UCY and SDD datasets show that TIGFlow-GRPO improves forecasting accuracy and long-horizon stability while generating trajectories that are more socially compliant and physically feasible. These results suggest that the proposed framework provides an effective way to connect flow-based trajectory modeling with behavior-aware alignment in dynamic multimedia environments.
>
---
#### [new 005] A Unified Spatial Alignment Framework for Highly Transferable Transformation-Based Attacks on Spatially Structured Tasks
- **分类: cs.CV**

- **简介: 该论文针对结构化任务中的对抗攻击问题，提出统一的空间对齐框架（SAF），解决标签与输入空间不匹配导致的攻击效果差的问题。**

- **链接: [https://arxiv.org/pdf/2603.25230](https://arxiv.org/pdf/2603.25230)**

> **作者:** Jiaming Liang; Chi-Man Pun
>
> **摘要:** Transformation-based adversarial attacks (TAAs) demonstrate strong transferability when deceiving classification models. However, existing TAAs often perform unsatisfactorily or even fail when applied to structured tasks such as semantic segmentation and object detection. Encouragingly, recent studies that categorize transformations into non-spatial and spatial transformations inspire us to address this challenge. We find that for non-structured tasks, labels are spatially non-structured, and thus TAAs are not required to adjust labels when applying spatial transformations. In contrast, for structured tasks, labels are spatially structured, and failing to transform labels synchronously with inputs can cause spatial misalignment and yield erroneous gradients. To address these issues, we propose a novel unified Spatial Alignment Framework (SAF) for highly transferable TAAs on spatially structured tasks, where the TAAs spatially transform labels synchronously with the input using the proposed Spatial Alignment (SA) algorithm. Extensive experiments demonstrate the crucial role of our SAF for TAAs on structured tasks. Specifically, in non-targeted attacks, our SAF degrades the average mIoU on Cityscapes from 24.50 to 11.34, and on Kvasir-SEG from 49.91 to 31.80, while reducing the average mAP of COCO from 17.89 to 5.25.
>
---
#### [new 006] FD$^2$: A Dedicated Framework for Fine-Grained Dataset Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数据蒸馏任务，旨在解决细粒度数据集中样本相似度过高、区分度不足的问题。提出FD²框架，通过定位判别区域和约束优化，提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2603.25144](https://arxiv.org/pdf/2603.25144)**

> **作者:** Hongxu Ma; Guang Li; Shijie Wang; Dongzhan Zhou; Baoli Sun; Takahiro Ogawa; Miki Haseyama; Zhihui Wang
>
> **摘要:** Dataset distillation (DD) compresses a large training set into a small synthetic set, reducing storage and training cost, and has shown strong results on general benchmarks. Decoupled DD further improves efficiency by splitting the pipeline into pretraining, sample distillation, and soft-label generation. However, existing decoupled methods largely rely on coarse class-label supervision and optimize samples within each class in a nearly identical manner. On fine-grained datasets, this often yields distilled samples that (i) retain large intra-class variation with subtle inter-class differences and (ii) become overly similar within the same class, limiting localized discriminative cues and hurting recognition. To solve the above-mentioned problems, we propose FD$^{2}$, a dedicated framework for Fine-grained Dataset Distillation. FD$^{2}$ localizes discriminative regions and constructs fine-grained representations for distillation. During pretraining, counterfactual attention learning aggregates discriminative representations to update class prototypes. During distillation, a fine-grained characteristic constraint aligns each sample with its class prototype while repelling others, and a similarity constraint diversifies attention across same-class samples. Experiments on multiple fine-grained and general datasets show that FD$^{2}$ integrates seamlessly with decoupled DD and improves performance in most settings, indicating strong transferability.
>
---
#### [new 007] Lookalike3D: Seeing Double in 3D
- **分类: cs.CV**

- **简介: 该论文提出Lookalike3D任务，解决室内场景中重复物体的检测问题。通过多视角图像和语义先验，区分物体对的相似性，并构建3DTwins数据集提升3D感知性能。**

- **链接: [https://arxiv.org/pdf/2603.24713](https://arxiv.org/pdf/2603.24713)**

> **作者:** Chandan Yeshwanth; Angela Dai
>
> **备注:** Project page: this https URL, Video: this https URL
>
> **摘要:** 3D object understanding and generation methods produce impressive results, yet they often overlook a pervasive source of information in real-world scenes: repeated objects. We introduce the task of lookalike object detection in indoor scenes, which leverages repeated and complementary cues from identical and near-identical object pairs. Given an input scene, the task is to classify pairs of objects as identical, similar or different using multiview images as input. To address this, we present Lookalike3D, a multiview image transformer that effectively distinguishes such object pairs by harnessing strong semantic priors from large image foundation models. To support this task, we collected the 3DTwins dataset, containing 76k manually annotated identical, similar and different pairs of objects based on ScanNet++, and show an improvement of 104% IoU over baselines. We demonstrate how our method improves downstream tasks such as enabling joint 3D object reconstruction and part co-segmentation, turning repeated and lookalike objects into a powerful cue for consistent, high-quality 3D perception. Our code, dataset and models will be made publicly available.
>
---
#### [new 008] GeoNDC: A Queryable Neural Data Cube for Planetary-Scale Earth Observation
- **分类: cs.CV; physics.geo-ph**

- **简介: 该论文提出GeoNDC，解决地球观测数据存储与查询效率问题，通过神经数据立方体实现高效压缩与时空查询。**

- **链接: [https://arxiv.org/pdf/2603.25037](https://arxiv.org/pdf/2603.25037)**

> **作者:** Jianbo Qi; Mengyao Li; Baogui Jiang; Yidan Chen; Qiao Wang
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** Satellite Earth observation has accumulated massive spatiotemporal archives essential for monitoring environmental change, yet these remain organized as discrete raster files, making them costly to store, transmit, and query. We present GeoNDC, a queryable neural data cube that encodes planetary-scale Earth observation data as a continuous spatiotemporal implicit neural field, enabling on-demand queries and continuous-time reconstruction without full decompression. Experiments on a 20-year global MODIS MCD43A4 reflectance record (7 bands, 5\,km, 8-day sampling) show that the learned representation supports direct spatiotemporal queries on consumer hardware. On Sentinel-2 imagery (10\,m), continuous temporal parameterization recovers cloud-free dynamics with high fidelity ($R^2 > 0.85$) under simulated 2-km cloud occlusion. On HiGLASS biophysical products (LAI and FPAR), GeoNDC attains near-perfect accuracy ($R^2 > 0.98$). The representation compresses the 20-year MODIS archive to 0.44\,GB -- approximately 95:1 relative to an optimized Int16 baseline -- with high spectral fidelity (mean $R^2 > 0.98$, mean RMSE $= 0.021$). These results suggest GeoNDC offers a unified AI-native representation for planetary-scale Earth observation, complementing raw archives with a compact, analysis-ready data layer integrating query, reconstruction, and compression in a single framework.
>
---
#### [new 009] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决MoE专家选择单一化问题，提出MoE-GRPO框架，通过强化学习提升专家选择多样性与模型性能。**

- **链接: [https://arxiv.org/pdf/2603.24984](https://arxiv.org/pdf/2603.24984)**

> **作者:** Dohwan Ko; Jinyoung Park; Seoung Choi; Sanghyeok Lee; Seohyun Lee; Hyunwoo J. Kim
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Mixture-of-Experts (MoE) has emerged as an effective approach to reduce the computational overhead of Transformer architectures by sparsely activating a subset of parameters for each token while preserving high model capacity. This paradigm has recently been extended to Vision-Language Models (VLMs), enabling scalable multi-modal understanding with reduced computational cost. However, the widely adopted deterministic top-K routing mechanism may overlook more optimal expert combinations and lead to expert overfitting. To address this limitation and improve the diversity of expert selection, we propose MoE-GRPO, a reinforcement learning (RL)-based framework for optimizing expert routing in MoE-based VLMs. Specifically, we formulate expert selection as a sequential decision-making problem and optimize it using Group Relative Policy Optimization (GRPO), allowing the model to learn adaptive expert routing policies through exploration and reward-based feedback. Furthermore, we introduce a modality-aware router guidance that enhances training stability and efficiency by discouraging the router from exploring experts that are infrequently activated for a given modality. Extensive experiments on multi-modal image and video benchmarks show that MoE-GRPO consistently outperforms standard top-K routing and its variants by promoting more diverse expert selection, thereby mitigating expert overfitting and enabling a task-level expert specialization.
>
---
#### [new 010] Challenges in Hyperspectral Imaging for Autonomous Driving: The HSI-Drive Case
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于自动驾驶视觉任务，探讨HSI在AD中的应用挑战，如光照变化、实时性要求等，并分析相关技术与算法。**

- **链接: [https://arxiv.org/pdf/2603.25510](https://arxiv.org/pdf/2603.25510)**

> **作者:** Koldo Basterretxea; Jon Gutiérrez-Zaballa; Javier Echanobe
>
> **摘要:** The use of hyperspectral imaging (HSI) in autonomous driving (AD), while promising, faces many challenges related to the specifics and requirements of this application domain. On the one hand, non-controlled and variable lighting conditions, the wide depth-of-field ranges, and dynamic scenes with fast-moving objects. On the other hand, the requirements for real-time operation and the limited computational resources of embedded platforms. The combination of these factors determines both the criteria for selecting appropriate HSI technologies and the development of custom vision algorithms that leverage the spectral and spatial information obtained from the sensors. In this article, we analyse several techniques explored in the research of HSI-based vision systems with application to AD, using as an example results obtained from experiments using data from the most recent version of the HSI-Drive dataset.
>
---
#### [new 011] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation
- **分类: cs.CV**

- **简介: 该论文提出TacSIm，用于足球战术风格模仿任务，解决现有研究忽视真实战术行为复制的问题，构建数据集并提供评估方法。**

- **链接: [https://arxiv.org/pdf/2603.25199](https://arxiv.org/pdf/2603.25199)**

> **作者:** Peng Wen; Yuting Wang; Qiurui Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Current football imitation research primarily aims to opti mize reward-based objectives, such as goals scored or win rate proxies, paying less attention to accurately replicat ing real-world team tactical behaviors. We introduce Tac SIm, a large-scale dataset and benchmark for Tactical Style Imitation in football. TacSIm imitates the acitons of all 11 players in one team in the given broadcast footage of Pre mier League matches under a single broadcast view. Under a offensive or defensive broadcast footage, TacSIm projects the beginning positions and actions of all 22 players from both sides onto a standard pitch coordinate system. Tac SIm offers an explicit style imitation task and evaluation protocols. Tactics style imitation is measured by using spatial occupancy similarity and movement vector similarity in defined time, supporting the evaluation of spatial and tem poral similarities for one team. We run multiple baseline methods in a unified virtual environment to generate full team behaviors, enabling both quantitative and visual as sessment of tactical coordination. By using unified data and metrics from broadcast to simulation, TacSIm estab lishes a rigorous benchmark for measuring and modeling style-aligned tactical imitation task in football.
>
---
#### [new 012] DeepFAN, a transformer-based deep learning model for human-artificial intelligence collaborative assessment of incidental pulmonary nodules in CT scans: a multi-reader, multi-case trial
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于肺结节分类任务，旨在解决深度学习模型难以全面融合全局与局部特征的问题。研究开发了DeepFAN模型，并通过临床试验验证其辅助放射科医生提高诊断准确性的效果。**

- **链接: [https://arxiv.org/pdf/2603.25607](https://arxiv.org/pdf/2603.25607)**

> **作者:** Zhenchen Zhu; Ge Hu; Weixiong Tan; Kai Gao; Chao Sun; Zhen Zhou; Kepei Xu; Wei Han; Meixia Shang; Xiaoming Qiu; Yiqing Tan; Jinhua Wang; Zhoumeng Ying; Li Peng; Wei Song; Lan Song; Zhengyu Jin; Nan Hong; Yizhou Yu
>
> **备注:** 28 pages for main text and 37 pages for supplementary information, 7 figures in main text and 9 figures in supplementary information
>
> **摘要:** The widespread adoption of CT has notably increased the number of detected lung nodules. However, current deep learning methods for classifying benign and malignant nodules often fail to comprehensively integrate global and local features, and most of them have not been validated through clinical trials. To address this, we developed DeepFAN, a transformer-based model trained on over 10K pathology-confirmed nodules and further conducted a multi-reader, multi-case clinical trial to evaluate its efficacy in assisting junior radiologists. DeepFAN achieved diagnostic area under the curve (AUC) of 0.939 (95% CI 0.930-0.948) on an internal test set and 0.954 (95% CI 0.934-0.973) on the clinical trial dataset involving 400 cases across three independent medical institutions. Explainability analysis indicated higher contributions from global than local features. Twelve readers' average performance significantly improved by 10.9% (95% CI 8.3%-13.5%) in AUC, 10.0% (95% CI 8.9%-11.1%) in accuracy, 7.6% (95% CI 6.1%-9.2%) in sensitivity, and 12.6% (95% CI 10.9%-14.3%) in specificity (P<0.001 for all). Nodule-level inter-reader diagnostic consistency improved from fair to moderate (overall k: 0.313 vs. 0.421; P=0.019). In conclusion, DeepFAN effectively assisted junior radiologists and may help homogenize diagnostic quality and reduce unnecessary follow-up of indeterminate pulmonary nodules. Chinese Clinical Trial Registry: ChiCTR2400084624.
>
---
#### [new 013] A Framework for Generating Semantically Ambiguous Images to Probe Human and Machine Perception
- **分类: cs.CV**

- **简介: 该论文属于视觉感知研究任务，旨在探究人类与机器在语义模糊图像上的感知差异。通过生成模糊图像，分析两者对概念边界的判断差异。**

- **链接: [https://arxiv.org/pdf/2603.24730](https://arxiv.org/pdf/2603.24730)**

> **作者:** Yuqi Hu; Vasha DuTell; Ahna R. Girshick; Jennifer E. Corbett
>
> **摘要:** The classic duck-rabbit illusion reveals that when visual evidence is ambiguous, the human brain must decide what it sees. But where exactly do human observers draw the line between ''duck'' and ''rabbit'', and do machine classifiers draw it in the same place? We use semantically ambiguous images as interpretability probes to expose how vision models represent the boundaries between concepts. We present a psychophysically-informed framework that interpolates between concepts in the CLIP embedding space to generate continuous spectra of ambiguous images, allowing us to precisely measure where and how humans and machine classifiers place their semantic boundaries. Using this framework, we show that machine classifiers are more biased towards seeing ''rabbit'', whereas humans are more aligned with the CLIP embedding used for synthesis, and the guidance scale seems to affect human sensitivity more strongly than machine classifiers. Our framework demonstrates how controlled ambiguity can serve as a diagnostic tool to bridge the gap between human psychophysical analysis, image classification, and generative image models, offering insight into human-model alignment, robustness, model interpretability, and image synthesis methods.
>
---
#### [new 014] SAVe: Self-Supervised Audio-visual Deepfake Detection Exploiting Visual Artifacts and Audio-visual Misalignment
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于多模态深度伪造检测任务，旨在解决合成数据依赖导致的泛化能力不足问题。通过自监督学习，利用视觉伪篡改和音画不同步特征提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.25140](https://arxiv.org/pdf/2603.25140)**

> **作者:** Sahibzada Adil Shahzad; Ammarah Hashmi; Junichi Yamagishi; Yusuke Yasuda; Yu Tsao; Chia-Wen Lin; Yan-Tsung Peng; Hsin-Min Wang
>
> **摘要:** Multimodal deepfakes can exhibit subtle visual artifacts and cross-modal inconsistencies, which remain challenging to detect, especially when detectors are trained primarily on curated synthetic forgeries. Such synthetic dependence can introduce dataset and generator bias, limiting scalability and robustness to unseen manipulations. We propose SAVe, a self-supervised audio-visual deepfake detection framework that learns entirely on authentic videos. SAVe generates on-the-fly, identity-preserving, region-aware self-blended pseudo-manipulations to emulate tampering artifacts, enabling the model to learn complementary visual cues across multiple facial granularities. To capture cross-modal evidence, SAVe also models lip-speech synchronization via an audio-visual alignment component that detects temporal misalignment patterns characteristic of audio-visual forgeries. Experiments on FakeAVCeleb and AV-LipSync-TIMIT demonstrate competitive in-domain performance and strong cross-dataset generalization, highlighting self-supervised learning as a scalable paradigm for multimodal deepfake detection.
>
---
#### [new 015] PASDiff: Physics-Aware Semantic Guidance for Joint Real-world Low-Light Face Enhancement and Restoration
- **分类: cs.CV**

- **简介: 该论文属于低光人脸增强与修复任务，解决真实场景下低光人脸图像的多维度退化问题。提出PASDiff模型，结合物理约束和语义引导，提升光照、色彩和身份一致性。**

- **链接: [https://arxiv.org/pdf/2603.24969](https://arxiv.org/pdf/2603.24969)**

> **作者:** Yilin Ni; Wenjie Li; Zhengxue Wang; Juncheng Li; Guangwei Gao; Jian Yang
>
> **摘要:** Face images captured in real-world low light suffer multiple degradations-low illumination, blur, noise, and low visibility, etc. Existing cascaded solutions often suffer from severe error accumulation, while generic joint models lack explicit facial priors and struggle to resolve clear face structures. In this paper, we propose PASDiff, a Physics-Aware Semantic Diffusion with a training-free manner. To achieve a plausible illumination and color distribution, we leverage inverse intensity weighting and Retinex theory to introduce photometric constraints, thereby reliably recovering visibility and natural chromaticity. To faithfully reconstruct facial details, our Style-Agnostic Structural Injection (SASI) extracts structures from an off-the-shelf facial prior while filtering out its intrinsic photometric biases, seamlessly harmonizing identity features with physical constraints. Furthermore, we construct WildDark-Face, a real-world benchmark of 700 low-light facial images with complex degradations. Extensive experiments demonstrate that PASDiff significantly outperforms existing methods, achieving a superior balance among natural illumination, color recovery, and identity consistency.
>
---
#### [new 016] NeuroVLM-Bench: Evaluation of Vision-Enabled Large Language Models for Clinical Reasoning in Neurological Disorders
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在评估视觉增强的大语言模型在神经系统疾病临床推理中的表现，解决模型在诊断、亚型预测等任务上的可靠性与效率问题。研究对比了多个模型，分析其性能与 trade-off。**

- **链接: [https://arxiv.org/pdf/2603.24846](https://arxiv.org/pdf/2603.24846)**

> **作者:** Katarina Trojachanec Dineva; Stefan Andonov; Ilinka Ivanoska; Ivan Kitanovski; Sasho Gramatikov; Tamara Kostova; Monika Simjanoska Misheva; Kostadin Mishev
>
> **备注:** 53 pages, 12 figures. Manuscript submitted to the BMC Medical Informatics and Decision Making journal
>
> **摘要:** Recent advances in multimodal large language models enable new possibilities for image-based decision support. However, their reliability and operational trade-offs in neuroimaging remain insufficiently understood. We present a comprehensive benchmarking study of vision-enabled large language models for 2D neuroimaging using curated MRI and CT datasets covering multiple sclerosis, stroke, brain tumors, other abnormalities, and normal controls. Models are required to generate multiple outputs simultaneously, including diagnosis, diagnosis subtype, imaging modality, specialized sequence, and anatomical plane. Performance is evaluated across four directions: discriminative classification with abstention, calibration, structured-output validity, and computational efficiency. A multi-phase framework ensures fair comparison while controlling for selection bias. Across twenty frontier multimodal models, the results show that technical imaging attributes such as modality and plane are nearly solved, whereas diagnostic reasoning, especially subtype prediction, remains challenging. Tumor classification emerges as the most reliable task, stroke is moderately solvable, while multiple sclerosis and rare abnormalities remain difficult. Few-shot prompting improves performance for several models but increases token usage, latency, and cost. Gemini-2.5-Pro and GPT-5-Chat achieve the strongest overall diagnostic performance, while Gemini-2.5-Flash offers the best efficiency-performance trade-off. Among open-weight architectures, MedGemma-1.5-4B demonstrates the most promising results, as under few-shot prompting, it approaches the zero-shot performance of several proprietary models, while maintaining perfect structured output. These findings provide practical insights into performance, reliability, and efficiency trade-offs, supporting standardized evaluation of multimodal LLMs in neuroimaging.
>
---
#### [new 017] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人群计数任务，研究对抗样本在不同模型间的迁移性。提出一种新框架，同时攻击密度图和点回归模型，提升攻击效果并保持视觉隐蔽性。**

- **链接: [https://arxiv.org/pdf/2603.24821](https://arxiv.org/pdf/2603.24821)**

> **作者:** Alabi Mehzabin Anisha; Guangjing Wang; Sriram Chellappan
>
> **备注:** Accepted at CVPR 2026 Main Conference
>
> **摘要:** State-of-the-art crowd counting and localization are primarily modeled using two paradigms: density maps and point regression. Given the field's security ramifications, there is active interest in model robustness against adversarial attacks. Recent studies have demonstrated transferability across density-map-based approaches via adversarial patches, but cross-paradigm attacks (i.e., across both density map-based models and point regression-based models) remain unexplored. We introduce a novel adversarial framework that compromises both density map and point regression architectural paradigms through a comprehensive multi-task loss optimization. For point-regression models, we employ scene-density-specific high-confidence logit suppression; for density-map approaches, we use peak-targeted density map suppression. Both are combined with model-agnostic perceptual constraints to ensure that perturbations are effective and imperceptible to the human eye. Extensive experiments demonstrate the effectiveness of our attack, achieving on average a 7X increase in Mean Absolute Error compared to clean images while maintaining competitive visual quality, and successfully transferring across seven state-of-the-art crowd models with transfer ratios ranging from 0.55 to 1.69. Our approach strikes a balance between attack effectiveness and imperceptibility compared to state-of-the-art transferable attack strategies. The source code is available at this https URL
>
---
#### [new 018] VolDiT: Controllable Volumetric Medical Image Synthesis with Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像生成任务，旨在解决传统方法局部性限制和可控性不足的问题。提出VolDiT，采用纯Transformer架构实现更优的全局建模与精确控制。**

- **链接: [https://arxiv.org/pdf/2603.25181](https://arxiv.org/pdf/2603.25181)**

> **作者:** Marvin Seyfarth; Salman Ul Hassan Dar; Yannik Frisch; Philipp Wild; Norbert Frey; Florian André; Sandy Engelhardt
>
> **摘要:** Diffusion models have become a leading approach for high-fidelity medical image synthesis. However, most existing methods for 3D medical image generation rely on convolutional U-Net backbones within latent diffusion frameworks. While effective, these architectures impose strong locality biases and limited receptive fields, which may constrain scalability, global context integration, and flexible conditioning. In this work, we introduce VolDiT, the first purely transformer-based 3D Diffusion Transformer for volumetric medical image synthesis. Our approach extends diffusion transformers to native 3D data through volumetric patch embeddings and global self-attention operating directly over 3D tokens. To enable structured control, we propose a timestep-gated control adapter that maps segmentation masks into learnable control tokens that modulate transformer layers during denoising. This token-level conditioning mechanism allows precise spatial guidance while preserving the modeling advantages of transformer architectures. We evaluate our model on high-resolution 3D medical image synthesis tasks and compare it to state-of-the-art 3D latent diffusion models based on U-Nets. Results demonstrate improved global coherence, superior generative fidelity, and enhanced controllability. Our findings suggest that fully transformerbased diffusion models provide a flexible foundation for volumetric medical image synthesis. The code and models trained on public data are available at this https URL.
>
---
#### [new 019] UNIC: Neural Garment Deformation Field for Real-time Clothed Character Animation
- **分类: cs.CV**

- **简介: 该论文属于虚拟角色服装动画任务，解决物理模拟耗时、计算成本高的问题。提出UNIC方法，通过神经变形场实现实时服装动画，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.25580](https://arxiv.org/pdf/2603.25580)**

> **作者:** Chengfeng Zhao; Junbo Qi; Yulou Liu; Zhiyang Dou; Minchen Li; Taku Komura; Ziwei Liu; Wenping Wang; Yuan Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Simulating physically realistic garment deformations is an essential task for virtual immersive experience, which is often achieved by physics simulation methods. However, these methods are typically time-consuming, computationally demanding, and require costly hardware, which is not suitable for real-time applications. Recent learning-based methods tried to resolve this problem by training graph neural networks to learn the garment deformation on vertices, which, however, fail to capture the intricate deformation of complex garment meshes with complex topologies. In this paper, we introduce a novel neural deformation field-based method, named UNIC, to animate the garments of an avatar in real time, given the motion sequences. Our key idea is to learn the instance-specific neural deformation field to animate the garment meshes. Such an instance-specific learning scheme does not require UNIC to generalize to new garments but only to new motion sequences, which greatly reduces the difficulty in training and improves the deformation quality. Moreover, neural deformation fields map the 3D points to their deformation offsets, which not only avoids handling topologies of the complex garments but also injects a natural smoothness constraint in the deformation learning. Extensive experiments have been conducted on various kinds of garment meshes to demonstrate the effectiveness and efficiency of UNIC over baseline methods, making it potentially practical and useful in real-world interactive applications like video games.
>
---
#### [new 020] Confidence-Based Mesh Extraction from 3D Gaussians
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决复杂场景下网格提取效率与精度问题。通过引入自监督置信度框架，动态平衡视觉与几何监督，提升提取效果。**

- **链接: [https://arxiv.org/pdf/2603.24725](https://arxiv.org/pdf/2603.24725)**

> **作者:** Lukas Radl; Felix Windisch; Andreas Kurz; Thomas Köhler; Michael Steiner; Markus Steinberger
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS) greatly accelerated mesh extraction from posed images due to its explicit representation and fast software rasterization. While the addition of geometric losses and other priors has improved the accuracy of extracted surfaces, mesh extraction remains difficult in scenes with abundant view-dependent effects. To resolve the resulting ambiguities, prior works rely on multi-view techniques, iterative mesh extraction, or large pre-trained models, sacrificing the inherent efficiency of 3DGS. In this work, we present a simple and efficient alternative by introducing a self-supervised confidence framework to 3DGS: within this framework, learnable confidence values dynamically balance photometric and geometric supervision. Extending our confidence-driven formulation, we introduce losses which penalize per-primitive color and normal variance and demonstrate their benefits to surface extraction. Finally, we complement the above with an improved appearance model, by decoupling the individual terms of the D-SSIM loss. Our final approach delivers state-of-the-art results for unbounded meshes while remaining highly efficient.
>
---
#### [new 021] LEMMA: Laplacian pyramids for Efficient Marine SeMAntic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于海洋语义分割任务，旨在解决传统方法计算成本高、不适用于资源受限场景的问题。提出LEMMA模型，利用拉普拉斯金字塔提升边缘识别，降低模型复杂度和推理时间。**

- **链接: [https://arxiv.org/pdf/2603.25689](https://arxiv.org/pdf/2603.25689)**

> **作者:** Ishaan Gakhar; Laven Srivastava; Sankarshanaa Sagaram; Aditya Kasliwal; Ujjwal Verma
>
> **备注:** Accepted at the MaCVi Workshop, CVPR 2026
>
> **摘要:** Semantic segmentation in marine environments is crucial for the autonomous navigation of unmanned surface vessels (USVs) and coastal Earth Observation events such as oil spills. However, existing methods, often relying on deep CNNs and transformer-based architectures, face challenges in deployment due to their high computational costs and resource-intensive nature. These limitations hinder the practicality of real-time, low-cost applications in real-world marine settings. To address this, we propose LEMMA, a lightweight semantic segmentation model designed specifically for accurate remote sensing segmentation under resource constraints. The proposed architecture leverages Laplacian Pyramids to enhance edge recognition, a critical component for effective feature extraction in complex marine environments for disaster response, environmental surveillance, and coastal monitoring. By integrating edge information early in the feature extraction process, LEMMA eliminates the need for computationally expensive feature map computations in deeper network layers, drastically reducing model size, complexity and inference time. LEMMA demonstrates state-of-the-art performance across datasets captured from diverse platforms while reducing trainable parameters and computational requirements by up to 71x, GFLOPs by up to 88.5\%, and inference time by up to 84.65\%, as compared to existing models. Experimental results highlight its effectiveness and real-world applicability, including 93.42\% IoU on the Oil Spill dataset and 98.97\% mIoU on Mastr1325.
>
---
#### [new 022] FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于空间转录组学任务，旨在提升基因表达预测效果。针对现有方法依赖稀疏图结构的问题，提出FEAST框架，通过全连接图和负样本感知注意力建模生物交互，提升预测精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.25247](https://arxiv.org/pdf/2603.25247)**

> **作者:** Taejin Jeong; Joohyeok Kim; Jinyeong Kim; Chanyoung Kim; Seong Jae Hwang
>
> **摘要:** Spatial Transcriptomics (ST) provides spatially-resolved gene expression, offering crucial insights into tissue architecture and complex diseases. However, its prohibitive cost limits widespread adoption, leading to significant attention on inferring spatial gene expression from readily available whole slide images. While graph neural networks have been proposed to model interactions between tissue regions, their reliance on pre-defined sparse graphs prevents them from considering potentially interacting spot pairs, resulting in a structural limitation in capturing complex biological relationships. To address this, we propose FEAST (Fully connected Expressive Attention for Spatial Transcriptomics), an attention-based framework that models the tissue as a fully connected graph, enabling the consideration of all pairwise interactions. To better reflect biological interactions, we introduce negative-aware attention, which models both excitatory and inhibitory interactions, capturing essential negative relationships that standard attention often overlooks. Furthermore, to mitigate the information loss from truncated or ignored context in standard spot image extraction, we introduce an off-grid sampling strategy that gathers additional images from intermediate regions, allowing the model to capture a richer morphological context. Experiments on public ST datasets show that FEAST surpasses state-of-the-art methods in gene expression prediction while providing biologically plausible attention maps that clarify positive and negative interactions. Our code is available at this https URL FEAST.
>
---
#### [new 023] Few-Shot Left Atrial Wall Segmentation in 3D LGE MRI via Meta-Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决3D LGE MRI中左心房壁薄、对比度低导致的分割难题。通过MAML框架实现少量标注数据下的精准分割。**

- **链接: [https://arxiv.org/pdf/2603.24985](https://arxiv.org/pdf/2603.24985)**

> **作者:** Yusri Al-Sanaani; Rebecca Thornhill; Pablo Nery; Elena Pena; Robert deKemp; Calum Redpath; David Birnie; Sreeraman Rajan
>
> **备注:** Submitted to IEEE EMBC 2026
>
> **摘要:** Segmenting the left atrial wall from late gadolinium enhancement magnetic resonance images (MRI) is challenging due to the wall's thin geometry, low contrast, and the scarcity of expert annotations. We propose a Model-Agnostic Meta-Learning (MAML) framework for K-shot (K = 5, 10, 20) 3D left atrial wall segmentation that is meta-trained on the wall task together with auxiliary left atrial and right atrial cavity tasks and uses a boundary-aware composite loss to emphasize thin-structure accuracy. We evaluated MAML segmentation performance on a hold-out test set and assessed robustness under an unseen synthetic shift and on a distinct local cohort. On the hold-out test set, MAML appeared to improve segmentation performance compared to the supervised fine-tuning model, achieving a Dice score (DSC) of 0.64 vs. 0.52 and HD95 of 5.70 vs. 7.60 mm at 5-shot, and approached the fully supervised reference at 20-shot (0.69 vs. 0.71 DSC). Under unseen shift, performance degraded but remained robust: at 5-shot, MAML attained 0.59 DSC and 5.99 mm HD95 on the unseen domain shift and 0.57 DSC and 6.01 mm HD95 on the local cohort, with consistent gains as K increased. These results suggest that more accurate and reliable thin-wall boundaries are achievable in low-shot adaptation, potentially enabling clinical translation with minimal additional labeling for the assessment of atrial remodeling.
>
---
#### [new 024] Image Rotation Angle Estimation: Comparing Circular-Aware Methods
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文研究图像旋转角度估计任务，解决角度的循环拓扑带来的边界不连续问题。通过比较五种圆周感知方法，评估其在不同架构上的性能，提升旋转估计精度。**

- **链接: [https://arxiv.org/pdf/2603.25351](https://arxiv.org/pdf/2603.25351)**

> **作者:** Maximilian Woehrer
>
> **备注:** 7 pages, 3 figures, 2 tables. Under review at Pattern Recognition Letters
>
> **摘要:** Automatic image rotation estimation is a key preprocessing step in many vision pipelines. This task is challenging because angles have circular topology, creating boundary discontinuities that hinder standard regression methods. We present a comprehensive study of five circular-aware methods for global orientation estimation: direct angle regression with circular loss, classification via angular binning, unit-vector regression, phase-shifting coder, and circular Gaussian distribution. Using transfer learning from ImageNet-pretrained models, we systematically evaluate these methods across sixteen modern architectures by adapting their output heads for rotation-specific predictions. Our results show that probabilistic methods, particularly the circular Gaussian distribution, are the most robust across architectures, while classification achieves the best accuracy on well-matched backbones but suffers training instabilities on others. The best configuration (classification with EfficientViT-B3) achieves a mean absolute error (MAE) of 1.23° (mean across five independent runs) on the DRC-D dataset, while the circular Gaussian distribution with MambaOut Base achieves a virtually identical 1.24° with greater robustness across backbones. Training and evaluating our top-performing method-architecture combinations on COCO 2014, the best configuration reaches 3.71° MAE, improving substantially over prior work, with further improvement to 2.84° on the larger COCO 2017 dataset.
>
---
#### [new 025] SportSkills: Physical Skill Learning from Sports Instructional Videos
- **分类: cs.CV**

- **简介: 该论文提出SportSkills数据集，解决物理技能学习中的细粒度动作理解问题。通过大量体育教学视频，提升模型对动作差异的识别能力，并实现错误条件下的视频检索任务。**

- **链接: [https://arxiv.org/pdf/2603.25163](https://arxiv.org/pdf/2603.25163)**

> **作者:** Kumar Ashutosh; Chi Hsuan Wu; Kristen Grauman
>
> **备注:** Technical report
>
> **摘要:** Current large-scale video datasets focus on general human activity, but lack depth of coverage on fine-grained activities needed to address physical skill learning. We introduce SportSkills, the first large-scale sports dataset geared towards physical skill learning with in-the-wild video. SportSkills has more than 360k instructional videos containing more than 630k visual demonstrations paired with instructional narrations explaining the know-how behind the actions from 55 varied sports. Through a suite of experiments, we show that SportSkills unlocks the ability to understand fine-grained differences between physical actions. Our representation achieves gains of up to 4x with the same model trained on traditional activity-centric datasets. Crucially, building on SportSkills, we introduce the first large-scale task formulation of mistake-conditioned instructional video retrieval, bridging representation learning and actionable feedback generation (e.g., "here's my execution of a skill; which video clip should I watch to improve it?"). Formal evaluations by professional coaches show our retrieval approach significantly advances the ability of video models to personalize visual instructions for a user query.
>
---
#### [new 026] PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow
- **分类: cs.CV**

- **简介: 该论文提出PSDesigner，解决自动化图形设计问题，模拟人类设计师的创作流程，通过专用组件和标注数据集，提升设计灵活性与质量。**

- **链接: [https://arxiv.org/pdf/2603.25738](https://arxiv.org/pdf/2603.25738)**

> **作者:** Xincheng Shuai; Song Tang; Yutong Huang; Henghui Ding; Dacheng Tao
>
> **备注:** CVPR 2026, Project Page: this https URL
>
> **摘要:** Graphic design is a creative and innovative process that plays a crucial role in applications such as e-commerce and advertising. However, developing an automated design system that can faithfully translate user intentions into editable design files remains an open challenge. Although recent studies have leveraged powerful text-to-image models and MLLMs to assist graphic design, they typically simplify professional workflows, resulting in limited flexibility and intuitiveness. To address these limitations, we propose PSDesigner, an automated graphic design system that emulates the creative workflow of human designers. Building upon multiple specialized components, PSDesigner collects theme-related assets based on user instructions, and autonomously infers and executes tool calls to manipulate design files, such as integrating new assets or refining inferior elements. To endow the system with strong tool-use capabilities, we construct a design dataset, CreativePSD, which contains a large amount of high-quality PSD design files annotated with operation traces across a wide range of design scenarios and artistic styles, enabling models to learn expert design procedures. Extensive experiments demonstrate that PSDesigner outperforms existing methods across diverse graphic design tasks, empowering non-specialists to conveniently create production-quality designs.
>
---
#### [new 027] MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决视觉基础模型推理时仅限单一尺度的问题。提出MuRF方法，在推理时融合多尺度特征，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2603.25744](https://arxiv.org/pdf/2603.25744)**

> **作者:** Bocheng Zou; Mu Cai; Mark Stanley; Dingfu Lu; Yong Jae Lee
>
> **摘要:** Vision Foundation Models (VFMs) have become the cornerstone of modern computer vision, offering robust representations across a wide array of tasks. While recent advances allow these models to handle varying input sizes during training, inference typically remains restricted to a single, fixed scale. This prevalent single-scale paradigm overlooks a fundamental property of visual perception: varying resolutions offer complementary inductive biases, where low-resolution views excel at global semantic recognition and high-resolution views are essential for fine-grained refinement. In this work, we propose Multi-Resolution Fusion (MuRF), a simple yet universally effective strategy to harness this synergy at inference time. Instead of relying on a single view, MuRF constructs a unified representation by processing an image at multiple resolutions through a frozen VFM and fusing the resulting features. The universality of MuRF is its most compelling attribute. It is not tied to a specific architecture, serving instead as a fundamental, training-free enhancement to visual representation. We empirically validate this by applying MuRF to a broad spectrum of critical computer vision tasks across multiple distinct VFM families - primarily DINOv2, but also demonstrating successful generalization to contrastive models like SigLIP2.
>
---
#### [new 028] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，解决长视频生成中因分布偏差导致的视觉质量下降问题。提出FreeLOC框架，通过位置重编码和分层注意力机制提升长视频质量。**

- **链接: [https://arxiv.org/pdf/2603.25209](https://arxiv.org/pdf/2603.25209)**

> **作者:** Jiahao Tian; Chenxi Song; Wei Cheng; Chi Zhang
>
> **备注:** Accepted to CVPR 2026. Code: this https URL
>
> **摘要:** Generating long videos using pre-trained video diffusion models, which are typically trained on short clips, presents a significant challenge. Directly applying these models for long-video inference often leads to a notable degradation in visual quality. This paper identifies that this issue primarily stems from two out-of-distribution (O.O.D) problems: frame-level relative position O.O.D and context-length O.O.D. To address these challenges, we propose FreeLOC, a novel training-free, layer-adaptive framework that introduces two core techniques: Video-based Relative Position Re-encoding (VRPR) for frame-level relative position O.O.D, a multi-granularity strategy that hierarchically re-encodes temporal relative positions to align with the model's pre-trained distribution, and Tiered Sparse Attention (TSA) for context-length O.O.D, which preserves both local detail and long-range dependencies by structuring attention density across different temporal scales. Crucially, we introduce a layer-adaptive probing mechanism that identifies the sensitivity of each transformer layer to these O.O.D issues, allowing for the selective and efficient application of our methods. Extensive experiments demonstrate that our approach significantly outperforms existing training-free methods, achieving state-of-the-art results in both temporal consistency and visual quality. Code is available at this https URL.
>
---
#### [new 029] AVControl: Efficient Framework for Training Audio-Visual Controls
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出AVControl框架，解决多模态音频-视频生成控制问题。通过LoRA方法实现高效、可扩展的模型训练，支持多种控制方式。**

- **链接: [https://arxiv.org/pdf/2603.24793](https://arxiv.org/pdf/2603.24793)**

> **作者:** Matan Ben-Yosef; Tavi Halperin; Naomi Ken Korem; Mohammad Salama; Harel Cain; Asaf Joseph; Anthony Chen; Urska Jelercic; Ofir Bibi
>
> **备注:** Project page: this https URL
>
> **摘要:** Controlling video and audio generation requires diverse modalities, from depth and pose to camera trajectories and audio transformations, yet existing approaches either train a single monolithic model for a fixed set of controls or introduce costly architectural changes for each new modality. We introduce AVControl, a lightweight, extendable framework built on LTX-2, a joint audio-visual foundation model, where each control modality is trained as a separate LoRA on a parallel canvas that provides the reference signal as additional tokens in the attention layers, requiring no architectural changes beyond the LoRA adapters themselves. We show that simply extending image-based in-context methods to video fails for structural control, and that our parallel canvas approach resolves this. On the VACE Benchmark, we outperform all evaluated baselines on depth- and pose-guided generation, inpainting, and outpainting, and show competitive results on camera control and audio-visual benchmarks. Our framework supports a diverse set of independently trained modalities: spatially-aligned controls such as depth, pose, and edges, camera trajectory with intrinsics, sparse motion control, video editing, and, to our knowledge, the first modular audio-visual controls for a joint generation model. Our method is both compute- and data-efficient: each modality requires only a small dataset and converges within a few hundred to a few thousand training steps, a fraction of the budget of monolithic alternatives. We publicly release our code and trained LoRA checkpoints.
>
---
#### [new 030] InstanceAnimator: Multi-Instance Sketch Video Colorization
- **分类: cs.CV**

- **简介: 该论文属于多实例素描视频着色任务，解决用户控制不足、实例对齐差和细节失真问题，提出InstanceAnimator框架实现更精确的多角色着色。**

- **链接: [https://arxiv.org/pdf/2603.25357](https://arxiv.org/pdf/2603.25357)**

> **作者:** Yinhan Zhang; Yue Ma; Bingyuan Wang; Kunyu Feng; Yeying Jin; Qifeng Chen; Anyi Rao; Zeyu Wang
>
> **摘要:** We propose InstanceAnimator, a novel Diffusion Transformer framework for multi-instance sketch video colorization. Existing methods suffer from three core limitations: inflexible user control due to heavy reliance on single reference frames, poor instance controllability leading to misalignment in multi-character scenarios, and degraded detail fidelity in fine-grained regions. To address these challenges, we introduce three corresponding innovations. First, a Canvas Guidance Condition eliminates workflow fragmentation by allowing free placement of reference elements and background, enabling unprecedented user flexibility. Second, an Instance Matching Mechanism resolves misalignment by integrating instance features with the sketches, ensuring precise control over multiple characters. Third, an Adaptive Decoupled Control Module enhances detail fidelity by injecting semantic features from characters, backgrounds, and text conditions into the diffusion process. Extensive experiments demonstrate that InstanceAnimator achieves superior multi-instance colorization with enhanced user control, high visual quality, and strong instance consistency.
>
---
#### [new 031] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training
- **分类: cs.CV**

- **简介: 该论文提出Wan-Weaver框架，解决多模态内容交织生成问题。通过文本规划与视觉一致性建模，实现长距离文本连贯与视觉一致的生成。**

- **链接: [https://arxiv.org/pdf/2603.25706](https://arxiv.org/pdf/2603.25706)**

> **作者:** Jinbo Xing; Zeyinzi Jiang; Yuxiang Tuo; Chaojie Mao; Xiaotang Gai; Xi Chen; Jingfeng Zhang; Yulin Pan; Zhen Han; Jie Xiao; Keyu Yan; Chenwei Xie; Chongyang Zhong; Kai Zhu; Tong Shen; Lianghua Huang; Yu Liu; Yujiu Yang
>
> **备注:** CVPR 2026 Camera-ready, Webpage: this https URL
>
> **摘要:** Recent unified models have made unprecedented progress in both understanding and generation. However, while most of them accept multi-modal inputs, they typically produce only single-modality outputs. This challenge of producing interleaved content is mainly due to training data scarcity and the difficulty of modeling long-range cross-modal context. To address this issue, we decompose interleaved generation into textual planning and visual consistency modeling, and introduce a framework consisting of a planner and a visualizer. The planner produces dense textual descriptions for visual content, while the visualizer synthesizes images accordingly. Under this guidance, we construct large-scale textual-proxy interleaved data (where visual content is represented in text) to train the planner, and curate reference-guided image data to train the visualizer. These designs give rise to Wan-Weaver, which exhibits emergent interleaved generation ability with long-range textual coherence and visual consistency. Meanwhile, the integration of diverse understanding and generation data into planner training enables Wan-Weaver to achieve robust task reasoning and generation proficiency. To assess the model's capability in interleaved generation, we further construct a benchmark that spans a wide range of use cases across multiple dimensions. Extensive experiments demonstrate that, even without access to any real interleaved data, Wan-Weaver achieves superior performance over existing methods.
>
---
#### [new 032] Accurate Point Measurement in 3DGS -- A New Alternative to Traditional Stereoscopic-View Based Measurements
- **分类: cs.CV**

- **简介: 该论文属于三维几何测量任务，旨在解决传统立体测量依赖昂贵设备和低精度问题。通过3DGS技术实现更准确的点测量，无需专业设备，提升测量精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.24716](https://arxiv.org/pdf/2603.24716)**

> **作者:** Deyan Deng; Rongjun Qin
>
> **备注:** Accepted to the 2026 ISPRS Congress
>
> **摘要:** 3D Gaussian Splatting (3DGS) has revolutionized real-time rendering with its state-of-the-art novel view synthesis, but its utility for accurate geometric measurement remains underutilized. Compared to multi-view stereo (MVS) point clouds or meshes, 3DGS rendered views present superior visual quality and completeness. However, current point measurement methods still rely on demanding stereoscopic workstations or direct picking on often-incomplete and inaccurate 3D meshes. As a novel view synthesizer, 3DGS renders exact source views and smoothly interpolates in-between views. This allows users to intuitively pick congruent points across different views while operating 3DGS models. By triangulating these congruent points, one can precisely generate 3D point measurements. This approach mimics traditional stereoscopic measurement but is significantly less demanding: it requires neither a stereo workstation nor specialized operator stereoscopic capability. Furthermore, it enables multi-view intersection (more than two views) for higher measurement accuracy. We implemented a web-based application to demonstrate this proof-of-concept (PoC). Using several UAV aerial datasets, we show this PoC allows users to successfully perform highly accurate point measurements, achieving accuracy matching or exceeding traditional stereoscopic methods on standard hardware. Specifically, our approach significantly outperforms direct mesh-based measurements. Quantitatively, our method achieves RMSEs in the 1-2 cm range on well-defined points. More critically, on challenging thin structures where mesh-based RMSE was 0.062 m, our method achieved 0.037 m. On sharp corners poorly reconstructed in the mesh, our method successfully measured all points with a 0.013 m RMSE, whereas the mesh method failed entirely. Code is available at: this https URL.
>
---
#### [new 033] Towards Foundation Models for 3D Scene Understanding: Instance-Aware Self-Supervised Learning for Point Clouds
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决点云表示缺乏实例感知的问题。提出PointINS框架，通过几何感知学习提升实例定位能力。**

- **链接: [https://arxiv.org/pdf/2603.25165](https://arxiv.org/pdf/2603.25165)**

> **作者:** Bin Yang; Mohamed Abdelsamad; Miao Zhang; Alexandru Paul Condurache
>
> **备注:** The paper was accepted by CVPR2026
>
> **摘要:** Recent advances in self-supervised learning (SSL) for point clouds have substantially improved 3D scene understanding without human annotations. Existing approaches emphasize semantic awareness by enforcing feature consistency across augmented views or by masked scene modeling. However, the resulting representations transfer poorly to instance localization, and often require full finetuning for strong performance. Instance awareness is a fundamental component of 3D perception, thus bridging this gap is crucial for progressing toward true 3D foundation models that support all downstream tasks on 3D data. In this work, we introduce PointINS, an instance-oriented self-supervised framework that enriches point cloud representations through geometry-aware learning. PointINS employs an orthogonal offset branch to jointly learn high-level semantic understanding and geometric reasoning, yielding instance awareness. We identify two consistent properties essential for robust instance localization and formulate them as complementary regularization strategies, Offset Distribution Regularization (ODR), which aligns predicted offsets with empirically observed geometric priors, and Spatial Clustering Regularization (SCR), which enforces local coherence by regularizing offsets with pseudo-instance masks. Through extensive experiments across five datasets, PointINS achieves on average +3.5% mAP improvement for indoor instance segmentation and +4.1% PQ gain for outdoor panoptic segmentation, paving the way for scalable 3D foundation models.
>
---
#### [new 034] THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics
- **分类: cs.CV**

- **简介: 该论文提出THEMIS基准，用于评估多模态大语言模型在学术论文欺诈检测中的表现，解决真实场景下视觉欺诈推理问题。**

- **链接: [https://arxiv.org/pdf/2603.25089](https://arxiv.org/pdf/2603.25089)**

> **作者:** Tzu-Yen Ma; Bo Zhang; Zichen Tang; Junpeng Ding; Haolin Tian; Yuanze Li; Zhuodi Hao; Zixin Ding; Zirui Wang; Xinyu Yu; Shiyao Peng; Yizhuo Zhao; Ruomeng Jiang; Yiling Huang; Peizhi Zhao; Jiayuan Chen; Weisheng Tan; Haocheng Gao; Yang Liu; Jiacheng Liu; Zhongjun Yang; Jiayu Huang; Haihong E
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** We present THEMIS, a novel multi-task benchmark designed to comprehensively evaluate multimodal large language models (MLLMs) on visual fraud reasoning within real-world academic scenarios. Compared to existing benchmarks, THEMIS introduces three major advances. (1) Real-World Scenarios and Complexity: Our benchmark comprises over 4,000 questions spanning seven scenarios, derived from authentic retracted-paper cases and carefully curated multimodal synthetic data. With 60.47% complex-texture images, THEMIS bridges the critical gap between existing benchmarks and the complexity of real-world academic fraud. (2) Fraud-Type Diversity and Granularity: THEMIS systematically covers five challenging fraud types and introduces 16 fine-grained manipulation operations. On average, each sample undergoes multiple stacked manipulation operations, with the diversity and difficulty of these manipulations demanding a high level of visual fraud reasoning from the models. (3) Multi-Dimensional Capability Evaluation: We establish a mapping from fraud types to five core visual fraud reasoning capabilities, thereby enabling an evaluation that reveals the distinct strengths and specific weaknesses of different models across these core capabilities. Experiments on 16 leading MLLMs show that even the best-performing model, GPT-5, achieves an overall performance of only 56.15%, demonstrating that our benchmark presents a stringent test. We expect THEMIS to advance the development of MLLMs for complex, real-world fraud reasoning tasks.
>
---
#### [new 035] Vega: Learning to Drive with Natural Language Instructions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Vega模型，解决自动驾驶中根据自然语言指令进行决策的问题。构建了包含10万场景的InstructScene数据集，融合视觉、语言和动作模态，提升个性化驾驶能力。**

- **链接: [https://arxiv.org/pdf/2603.25741](https://arxiv.org/pdf/2603.25741)**

> **作者:** Sicheng Zuo; Yuxuan Li; Wenzhao Zheng; Zheng Zhu; Jie Zhou; Jiwen Lu
>
> **备注:** Code is available at this https URL
>
> **摘要:** Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified Vision-Language-World-Action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.
>
---
#### [new 036] ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling
- **分类: cs.CV**

- **简介: 该论文提出ShotStream，解决多镜头视频生成中的交互性与延迟问题，通过因果架构实现高效实时生成。**

- **链接: [https://arxiv.org/pdf/2603.25746](https://arxiv.org/pdf/2603.25746)**

> **作者:** Yawen Luo; Xiaoyu Shi; Junhao Zhuang; Yutian Chen; Quande Liu; Xintao Wang; Pengfei Wan; Tianfan Xue
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Multi-shot video generation is crucial for long narrative storytelling, yet current bidirectional architectures suffer from limited interactivity and high latency. We propose ShotStream, a novel causal multi-shot architecture that enables interactive storytelling and efficient on-the-fly frame generation. By reformulating the task as next-shot generation conditioned on historical context, ShotStream allows users to dynamically instruct ongoing narratives via streaming prompts. We achieve this by first fine-tuning a text-to-video model into a bidirectional next-shot generator, which is then distilled into a causal student via Distribution Matching Distillation. To overcome the challenges of inter-shot consistency and error accumulation inherent in autoregressive generation, we introduce two key innovations. First, a dual-cache memory mechanism preserves visual coherence: a global context cache retains conditional frames for inter-shot consistency, while a local context cache holds generated frames within the current shot for intra-shot consistency. And a RoPE discontinuity indicator is employed to explicitly distinguish the two caches to eliminate ambiguity. Second, to mitigate error accumulation, we propose a two-stage distillation strategy. This begins with intra-shot self-forcing conditioned on ground-truth historical shots and progressively extends to inter-shot self-forcing using self-generated histories, effectively bridging the train-test gap. Extensive experiments demonstrate that ShotStream generates coherent multi-shot videos with sub-second latency, achieving 16 FPS on a single GPU. It matches or exceeds the quality of slower bidirectional models, paving the way for real-time interactive storytelling. Training and inference code, as well as the models, are available on our
>
---
#### [new 037] Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于3D空间推理任务，旨在解决大语言模型在3D场景中缺乏有效空间关系编码的问题。提出QuatRoPE方法，实现高效且准确的空间关系建模。**

- **链接: [https://arxiv.org/pdf/2603.24721](https://arxiv.org/pdf/2603.24721)**

> **作者:** Shengli Zhou; Minghang Zheng; Feng Zheng; Yang Liu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Spatial reasoning focuses on locating target objects based on spatial relations in 3D scenes, which plays a crucial role in developing intelligent embodied agents. Due to the limited availability of 3D scene-language paired data, it is challenging to train models with strong reasoning ability from scratch. Previous approaches have attempted to inject 3D scene representations into the input space of Large Language Models (LLMs) and leverage the pretrained comprehension and reasoning abilities for spatial reasoning. However, models encoding absolute positions struggle to extract spatial relations from prematurely fused features, while methods explicitly encoding all spatial relations (which is quadratic in the number of objects) as input tokens suffer from poor scalability. To address these limitations, we propose QuatRoPE, a novel positional embedding method with an input length that is linear to the number of objects, and explicitly calculates pairwise spatial relations through the dot product in attention layers. QuatRoPE's holistic vector encoding of 3D coordinates guarantees a high degree of spatial consistency, maintaining fidelity to the scene's geometric integrity. Additionally, we introduce the Isolated Gated RoPE Extension (IGRE), which effectively limits QuatRoPE's influence to object-related tokens, thereby minimizing interference with the LLM's existing positional embeddings and maintaining the LLM's original capabilities. Extensive experiments demonstrate the effectiveness of our approaches. The code and data are available at this https URL.
>
---
#### [new 038] CardioDiT: Latent Diffusion Transformers for 4D Cardiac MRI Synthesis
- **分类: cs.CV**

- **简介: 该论文属于医学图像合成任务，旨在解决4D心脏MRI生成中的时空一致性问题。提出CardioDiT模型，直接建模4D数据，提升图像质量与生理合理性。**

- **链接: [https://arxiv.org/pdf/2603.25194](https://arxiv.org/pdf/2603.25194)**

> **作者:** Marvin Seyfarth; Sarah Kaye Müller; Arman Ghanaat; Isabelle Ayx; Fabian Fastenrath; Philipp Wild; Alexander Hertel; Theano Papavassiliu; Salman Ul Hassan Dar; Sandy Engelhardt
>
> **摘要:** Latent diffusion models (LDMs) have recently achieved strong performance in 3D medical image synthesis. However, modalities like cine cardiac MRI (CMR), representing a temporally synchronized 3D volume across the cardiac cycle, add an additional dimension that most generative approaches do not model directly. Instead, they factorize space and time or enforce temporal consistency through auxiliary mechanisms such as anatomical masks. Such strategies introduce structural biases that may limit global context integration and lead to subtle spatiotemporal discontinuities or physiologically inconsistent cardiac dynamics. We investigate whether a unified 4D generative model can learn continuous cardiac dynamics without architectural factorization. We propose CardioDiT, a fully 4D latent diffusion framework for short-axis cine CMR synthesis based on diffusion transformers. A spatiotemporal VQ-VAE encodes 2D+t slices into compact latents, which a diffusion transformer then models jointly as complete 3D+t volumes, coupling space and time throughout the generative process. We evaluate CardioDiT on public CMR datasets and a larger private cohort, comparing it to baselines with progressively stronger spatiotemporal coupling. Results show improved inter-slice consistency, temporally coherent motion, and realistic cardiac function distributions, suggesting that explicit 4D modeling with a diffusion transformer provides a principled foundation for spatiotemporal cardiac image synthesis. Code and models trained on public data are available at this https URL.
>
---
#### [new 039] AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments
- **分类: cs.CV**

- **简介: 该论文针对室内单目语义场景补全任务，解决复杂布局和遮挡带来的挑战。提出AdaSFormer模型，通过自适应序列化Transformer、相对位置编码和卷积调制归一化提升性能。**

- **链接: [https://arxiv.org/pdf/2603.25494](https://arxiv.org/pdf/2603.25494)**

> **作者:** Xuzhi Wang; Xinran Wu; Song Wang; Lingdong Kong; Ziping Zhao
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Indoor monocular semantic scene completion (MSSC) is notably more challenging than its outdoor counterpart due to complex spatial layouts and severe occlusions. While transformers are well suited for modeling global dependencies, their high memory cost and difficulty in reconstructing fine-grained details have limited their use in indoor MSSC. To address these limitations, we introduce AdaSFormer, a serialized transformer framework tailored for indoor MSSC. Our model features three key designs: (1) an Adaptive Serialized Transformer with learnable shifts that dynamically adjust receptive fields; (2) a Center-Relative Positional Encoding that captures spatial information richness; and (3) a Convolution-Modulated Layer Normalization that bridges heterogeneous representations between convolutional and transformer features. Extensive experiments on NYUv2 and Occ-ScanNet demonstrate that AdaSFormer achieves state-of-the-art performance. The code is publicly available at: this https URL.
>
---
#### [new 040] Is Geometry Enough? An Evaluation of Landmark-Based Gaze Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于 gaze estimation 任务，旨在评估基于几何特征的注视估计方法。通过构建标准化管道提取面部关键点，训练轻量模型，验证其在跨域任务中的性能，探索几何特征的有效性。**

- **链接: [https://arxiv.org/pdf/2603.24724](https://arxiv.org/pdf/2603.24724)**

> **作者:** Daniele Agostinelli; Thomas Agostinelli; Andrea Generosi; Maura Mengoni
>
> **摘要:** Appearance-based gaze estimation frequently relies on deep Convolutional Neural Networks (CNNs). These models are accurate, but computationally expensive and act as "black boxes", offering little interpretability. Geometric methods based on facial landmarks are a lightweight alternative, but their performance limits and generalization capabilities remain underexplored in modern benchmarks. In this study, we conduct a comprehensive evaluation of landmark-based gaze estimation. We introduce a standardized pipeline to extract and normalize landmarks from three large-scale datasets (Gaze360, ETH-XGaze, and GazeGene) and train lightweight regression models, specifically Extreme Gradient Boosted trees and two neural architectures: a holistic Multi-Layer Perceptron (MLP) and a siamese MLP designed to capture binocular geometry. We find that landmark-based models exhibit lower performance in within-domain evaluation, likely due to noise introduced into the datasets by the landmark detector. Nevertheless, in cross-domain evaluation, the proposed MLP architectures show generalization capabilities comparable to those of ResNet18 baselines. These findings suggest that sparse geometric features encode sufficient information for robust gaze estimation, paving the way for efficient, interpretable, and privacy-friendly edge applications. The source code and generated landmark-based datasets are available at this https URL.
>
---
#### [new 041] CIAR: Interval-based Collaborative Decoding for Image Generation Acceleration
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决AR模型计算密集、部署困难的问题。提出CIAR框架，通过区间解码和设备端验证加速生成过程，提升效率并减少云端请求。**

- **链接: [https://arxiv.org/pdf/2603.25463](https://arxiv.org/pdf/2603.25463)**

> **作者:** Keming Ye; Zhou Zhao; Fan Wu; Shengyu Zhang
>
> **备注:** 23 pages, 10 tables, 7 figures
>
> **摘要:** Auto-regressive (AR) models have recently made notable progress in image generation, achieving performance comparable to diffusion-based approaches. However, their computational intensity and sequential nature impede on-device deployment, causing disruptive latency. We address this via a cloud-device collaboration framework \textbf{CIAR}, which utilizes on-device self-verification to handle two key properties of visual synthesis: \textit{the vast token vocabulary} required for high-fidelity images and \textit{inherent spatial redundancy} which leads to extreme predictability in homogeneous regions, while object boundaries exhibit high uncertainty. Uniform verification wastes resources on such redundant tokens. Our solution centers on an on-device token uncertainty quantifier, which adopts continuous probability intervals to accelerate processing and make it feasible for large visual vocabularies instead of conventional discrete solution sets. Additionally, we incorporate a Interval-enhanced decoding module to further speed up decoding while maintaining visual fidelity and semantic consistency via a distribution alignment training strategy. Extensive experiments demonstrate that CIAR achieves a 2.18x speed-up and reduces cloud requests by 70\%, while preserving image quality compared to existing methods.
>
---
#### [new 042] Multimodal Dataset Distillation via Phased Teacher Models
- **分类: cs.CV**

- **简介: 该论文属于多模态数据蒸馏任务，旨在解决教师模型后期知识难以捕捉的问题。提出PTM-ST框架，提升蒸馏稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2603.25388](https://arxiv.org/pdf/2603.25388)**

> **作者:** Shengbin Guo; Hang Zhao; Senqiao Yang; Chenyang Jiang; Yuhang Cheng; Xiangru Peng; Rui Shao; Zhuotao Tian
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Multimodal dataset distillation aims to construct compact synthetic datasets that enable efficient compression and knowledge transfer from large-scale image-text data. However, existing approaches often fail to capture the complex, dynamically evolving knowledge embedded in the later training stages of teacher models. This limitation leads to degraded student performance and compromises the quality of the distilled data. To address critical challenges such as pronounced cross-stage performance gaps and unstable teacher trajectories, we propose Phased Teacher Model with Shortcut Trajectory (PTM-ST) -- a novel phased distillation framework. PTM-ST leverages stage-aware teacher modeling and a shortcut-based trajectory construction strategy to accurately fit the teacher's learning dynamics across distinct training phases. This enhances both the stability and expressiveness of the distillation process. Through theoretical analysis and comprehensive experiments, we show that PTM-ST significantly mitigates optimization oscillations and inter-phase knowledge gaps, while also reducing storage overhead. Our method consistently surpasses state-of-the-art baselines on Flickr30k and COCO, achieving up to 13.5% absolute improvement and an average gain of 9.53% on Flickr30k. Code: this https URL.
>
---
#### [new 043] Z-Erase: Enabling Concept Erasure in Single-Stream Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成中的概念擦除任务，旨在解决单流扩散Transformer中概念擦除导致的生成崩溃问题。提出Z-Erase方法实现稳定的概念擦除。**

- **链接: [https://arxiv.org/pdf/2603.25074](https://arxiv.org/pdf/2603.25074)**

> **作者:** Nanxiang Jiang; Zhaoxin Fan; Baisen Wang; Daiheng Gao; Junhang Cheng; Jifeng Guo; Yalan Qin; Yeying Jin; Hongwei Zheng; Faguo Wu; Wenjun Wu
>
> **摘要:** Concept erasure serves as a vital safety mechanism for removing unwanted concepts from text-to-image (T2I) models. While extensively studied in U-Net and dual-stream architectures (e.g., Flux), this task remains under-explored in the recent emerging paradigm of single-stream diffusion transformers (e.g., Z-Image). In this new paradigm, text and image tokens are processed as a single unified sequence via shared parameters. Consequently, directly applying prior erasure methods typically leads to generation collapse. To bridge this gap, we introduce Z-Erase, the first concept erasure method tailored for single-stream T2I models. To guarantee stable image generation, Z-Erase first proposes a Stream Disentangled Concept Erasure Framework that decouples updates and enables existing methods on single-stream models. Subsequently, within this framework, we introduce Lagrangian-Guided Adaptive Erasure Modulation, a constrained algorithm that further balances the sensitive erasure-preservation trade-off. Moreover, we provide a rigorous convergence analysis proving that Z-Erase can converge to a Pareto stationary point. Experiments demonstrate that Z-Erase successfully overcomes the generation collapse issue, achieving state-of-the-art performance across a wide range of tasks.
>
---
#### [new 044] GIFT: Global Irreplaceability Frame Targeting for Efficient Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决视频大语言模型计算成本高的问题。提出GIFT框架，通过评估帧的不可替代性高效选择关键帧。**

- **链接: [https://arxiv.org/pdf/2603.25072](https://arxiv.org/pdf/2603.25072)**

> **作者:** Junpeng Ma; Sashuai Zhou; Guanghao Li; Xin Gao; Yue Cao; Hengyu Zeng; Yuxiang Yan; Zhibin Wang; Jun Song; Bo Zheng; Shanghang Zhang; Jian Pu
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Video Large Language Models (VLMs) have achieved remarkable success in video understanding, but the significant computational cost from processing dense frames severely limits their practical application. Existing methods alleviate this by selecting keyframes, but their greedy decision-making, combined with a decoupled evaluation of relevance and diversity, often falls into local optima and results in erroneously selecting irrelevant noise frames. To address these challenges, we propose GIFT: Global Irreplaceability Frame Targeting, a novel training-free framework that selects frames by assessing their intrinsic irreplaceability. Specifically, we first introduce Directed Diversity to quantify a frame's uniqueness conditioned on relevance, which allows us to formulate a unified irreplaceability score. Subsequently, our Budget-Aware Refinement strategy employs a adaptive iterative process that first secures a core set of frames with the highest irreplaceability, and then shifts its priority to building crucial temporal context around these selections as the budget expands. Extensive experiments demonstrate that GIFT achieves a maximum average improvement of 12.5% across long-form video benchmarks on LLaVA-Video-7B compared to uniform sampling.
>
---
#### [new 045] CLIP-RD: Relational Distillation for Efficient CLIP Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在提升轻量模型对CLIP结构关系的保留能力。通过引入VRD和XRD方法，增强师生嵌入间的多向关系建模。**

- **链接: [https://arxiv.org/pdf/2603.25383](https://arxiv.org/pdf/2603.25383)**

> **作者:** Jeannie Chung; Hanna Jang; Ingyeong Yang; Uiwon Hwang; Jaehyung Sim
>
> **摘要:** CLIP aligns image and text embeddings via contrastive learning and demonstrates strong zero-shot generalization. Its large-scale architecture requires substantial computational and memory resources, motivating the distillation of its capabilities into lightweight student models. However, existing CLIP distillation methods do not explicitly model multi-directional relational dependencies between teacher and student embeddings, limiting the student's ability to preserve the structural relationships encoded by the teacher. To address this, we propose a relational knowledge distillation framework that introduces two novel methods, Vertical Relational Distillation (VRD) and Cross Relational Distillation (XRD). VRD enforces consistency of teacher-student distillation strength across modalities at the distribution level, while XRD imposes bidirectional symmetry on cross-modal teacher-student similarity distributions. By jointly modeling multi-directional relational structures, CLIP-RD promotes faithful alignment of the student embedding geometry with that of the teacher, outperforming existing methods by 0.8%p.
>
---
#### [new 046] Relaxed Rigidity with Ray-based Grouping for Dynamic Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于动态3D场景重建任务，旨在解决Gaussian Splatting中运动与物理动态不一致的问题。通过引入基于视图空间射线的分组策略，保持局部几何结构稳定，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.24994](https://arxiv.org/pdf/2603.24994)**

> **作者:** Junoh Leea; Junmyeong Lee; Yeon-Ji Song; Inhwan Bae; Jisu Shin; Hae-Gon Jeon; Jin-Hwa Kim
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** The reconstruction of dynamic 3D scenes using 3D Gaussian Splatting has shown significant promise. A key challenge, however, remains in modeling realistic motion, as most methods fail to align the motion of Gaussians with real-world physical dynamics. This misalignment is particularly problematic for monocular video datasets, where failing to maintain coherent motion undermines local geometric structure, ultimately leading to degraded reconstruction quality. Consequently, many state-of-the-art approaches rely heavily on external priors, such as optical flow or 2D tracks, to enforce temporal coherence. In this work, we propose a novel method to explicitly preserve the local geometric structure of Gaussians across time in 4D scenes. Our core idea is to introduce a view-space ray grouping strategy that clusters Gaussians intersected by the same ray, considering only those whose $\alpha$-blending weights exceed a threshold. We then apply constraints to these groups to maintain a consistent spatial distribution, effectively preserving their local geometry. This approach enforces a more physically plausible motion model by ensuring that local geometry remains stable over time, eliminating the reliance on external guidance. We demonstrate the efficacy of our method by integrating it into two distinct baseline models. Extensive experiments on challenging monocular datasets show that our approach significantly outperforms existing methods, achieving superior temporal consistency and reconstruction quality.
>
---
#### [new 047] Interpretable Zero-shot Referring Expression Comprehension with Query-driven Scene Graphs
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于零样本指代表达理解任务，旨在解决传统方法在细粒度视觉细节和复杂关系理解上的不足。提出SGREC方法，利用查询驱动的场景图增强可解释性。**

- **链接: [https://arxiv.org/pdf/2603.25004](https://arxiv.org/pdf/2603.25004)**

> **作者:** Yike Wu; Necva Bolucu; Stephen Wan; Dadong Wang; Jiahao Xia; Jian Zhang
>
> **备注:** Accepted by T-MM
>
> **摘要:** Zero-shot referring expression comprehension (REC) aims to locate target objects in images given natural language queries without relying on task-specific training data, demanding strong visual understanding capabilities. Existing Vision-Language Models~(VLMs), such as CLIP, commonly address zero-shot REC by directly measuring feature similarities between textual queries and image regions. However, these methods struggle to capture fine-grained visual details and understand complex object relationships. Meanwhile, Large Language Models~(LLMs) excel at high-level semantic reasoning, their inability to directly abstract visual features into textual semantics limits their application in REC tasks. To overcome these limitations, we propose \textbf{SGREC}, an interpretable zero-shot REC method leveraging query-driven scene graphs as structured intermediaries. Specifically, we first employ a VLM to construct a query-driven scene graph that explicitly encodes spatial relationships, descriptive captions, and object interactions relevant to the given query. By leveraging this scene graph, we bridge the gap between low-level image regions and higher-level semantic understanding required by LLMs. Finally, an LLM infers the target object from the structured textual representation provided by the scene graph, responding with detailed explanations for its decisions that ensure interpretability in the inference process. Extensive experiments show that SGREC achieves top-1 accuracy on most zero-shot REC benchmarks, including RefCOCO val (66.78\%), RefCOCO+ testB (53.43\%), and RefCOCOg val (73.28\%), highlighting its strong visual scene understanding.
>
---
#### [new 048] Distributed Real-Time Vehicle Control for Emergency Vehicle Transit: A Scalable Cooperative Method
- **分类: cs.CV**

- **简介: 该论文属于应急车辆实时协同控制任务，旨在解决传统方法计算成本高、可扩展性差的问题。提出一种基于局部信息的分布式控制方法，实现快速决策与安全冲突规避。**

- **链接: [https://arxiv.org/pdf/2603.25000](https://arxiv.org/pdf/2603.25000)**

> **作者:** WenXi Wang; JunQi Zhang
>
> **备注:** Submitted to IEEE Transactions on Cybernetics
>
> **摘要:** Rapid transit of emergency vehicles is critical for saving lives and reducing property loss but often relies on surrounding ordinary vehicles to cooperatively adjust their driving behaviors. It is important to ensure rapid transit of emergency vehicles while minimizing the impact on ordinary vehicles. Centralized mathematical solver and reinforcement learning are the state-of-the-art methods. The former obtains optimal solutions but is only practical for small-scale scenarios. The latter implicitly learns through extensive centralized training but the trained model exhibits limited scalability to different traffic conditions. Hence, existing methods suffer from two fundamental limitations: high computational cost and lack of scalability. To overcome above limitations, this work proposes a scalable distributed vehicle control method, where vehicles adjust their driving behaviors in a distributed manner online using only local instead of global information. We proved that the proposed distributed method using only local information is approximately equivalent to the one using global information, which enables vehicles to evaluate their candidate states and make approximately optimal decisions in real time without pre-training and with natural adaptability to varying traffic conditions. Then, a distributed conflict resolution mechanism is further proposed to guarantee vehicles' safety by avoiding their decision conflicts, which eliminates the single-point-of-failure risk of centralized methods and provides deterministic safety guarantees that learned methods cannot offer. Compared with existing methods, simulation experiments based on real-world traffic datasets demonstrate that the proposed method achieves faster decision-making, less impact on ordinary vehicles, and maintains much stronger scalability across different traffic densities and road configurations.
>
---
#### [new 049] Designing Any Imaging System from Natural Language: Agent-Constrained Composition over a Finite Primitive Basis
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决影像系统设计效率低的问题。通过自然语言描述自动生成有效成像模型，提升设计自动化水平。**

- **链接: [https://arxiv.org/pdf/2603.25636](https://arxiv.org/pdf/2603.25636)**

> **作者:** Chengshuai Yang
>
> **备注:** 28 pages, 7 figures, 8 tables, includes Supplementary Information (sections S1-S6)
>
> **摘要:** Designing a computational imaging system -- selecting operators, setting parameters, validating consistency -- requires weeks of specialist effort per modality, creating an expertise bottleneck that excludes the broader scientific community from prototyping imaging instruments. We introduce this http URL, a structured specification format, and three autonomous agents -- Plan, Judge, and Execute -- that translate a one-sentence natural-language description into a validated forward model with bounded reconstruction error. A design-to-real error theorem decomposes total reconstruction error into five independently bounded terms, each linked to a corrective action. On 6 real-data modalities spanning all 5 carrier families, the automated pipeline matches expert-library quality (98.1 +/- 4.2%). Ten novel designs -- composing primitives into chains from 3D to 5D -- demonstrate compositional reach beyond any single-modality tool.
>
---
#### [new 050] LaMP: Learning Vision-Language-Action Policies with 3D Scene Flow as Latent Motion Prior
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LaMP框架，解决机器人操作中复杂3D交互的感知与控制问题。通过结合视觉-语言-动作和3D场景流，提升任务执行的鲁棒性与成功率。**

- **链接: [https://arxiv.org/pdf/2603.25399](https://arxiv.org/pdf/2603.25399)**

> **作者:** Xinkai Wang; Chenyi Wang; Yifu Xu; Mingzhe Ye; Fu-Cheng Zhang; Jialin Tian; Xinyu Zhan; Lifeng Zhu; Cewu Lu; Lixin Yang
>
> **摘要:** We introduce \textbf{LaMP}, a dual-expert Vision-Language-Action framework that embeds dense 3D scene flow as a latent motion prior for robotic manipulation. Existing VLA models regress actions directly from 2D semantic visual features, forcing them to learn complex 3D physical interactions implicitly. This implicit learning strategy degrades under unfamiliar spatial dynamics. LaMP addresses this limitation by aligning a flow-matching \emph{Motion Expert} with a policy-predicting \emph{Action Expert} through gated cross-attention. Specifically, the Motion Expert generates a one-step partially denoised 3D scene flow, and its hidden states condition the Action Expert without full multi-step reconstruction. We evaluate LaMP on the LIBERO, LIBERO-Plus, and SimplerEnv-WidowX simulation benchmarks as well as real-world experiments. LaMP consistently outperforms evaluated VLA baselines across LIBERO, LIBERO-Plus, and SimplerEnv-WidowX benchmarks, achieving the highest reported average success rates under the same training budgets. On LIBERO-Plus OOD perturbations, LaMP shows improved robustness with an average 9.7% gain over the strongest prior baseline. Our project page is available at this https URL.
>
---
#### [new 051] Towards automatic smoke detector inspection: Recognition of the smoke detectors in industrial facilities and preparation for future drone integration
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在自动识别工业设施中的烟雾探测器，以支持无人机巡检。研究比较了多种检测模型，并探索了数据增强策略，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.24850](https://arxiv.org/pdf/2603.24850)**

> **作者:** Lukas Kratochvila; Jakub Stefansky; Simon Bilik; Robert Rous; Tomas Zemcik; Michal Wolny; Frantisek Rusnak; Ondrej Cech; Karel Horak
>
> **摘要:** Fire safety consists of a complex pipeline, and it is a very important topic of concern. One of its frontal parts are the smoke detectors, which are supposed to provide an alarm prior to a massive fire appears. As they are often difficult to reach due to high ceilings or problematic locations, an automatic inspection system would be very beneficial as it could allow faster revisions, prevent workers from dangerous work in heights, and make the whole process cheaper. In this study, we present the smoke detector recognition part of the automatic inspection system, which could easily be integrated to the drone system. As part of our research, we compare two popular convolutional-based object detectors YOLOv11 and SSD widely used on embedded devices together with the state-of-the-art transformer-based RT-DETRv2 with the backbones of different sizes. Due to a complicated way of collecting a sufficient amount of data for training in the real-world environment, we also compare several training strategies using the real and semi-synthetic data together with various augmentation methods. To achieve a robust testing, all models were evaluated on two test datasets with an expected and difficult appearance of the smoke detectors including motion blur, small resolution, or not complete objects. The best performing detector is the YOLOv11n, which reaches the average mAP@0.5 score of 0.884. Our code, pretrained models and dataset are publicly available.
>
---
#### [new 052] How good was my shot? Quantifying Player Skill Level in Table Tennis
- **分类: cs.CV**

- **简介: 该论文属于技能评估任务，旨在量化乒乓球运动员的技能水平。通过构建生成模型和嵌入空间，捕捉球员战术特征，实现技能的相对与绝对预测。**

- **链接: [https://arxiv.org/pdf/2603.25736](https://arxiv.org/pdf/2603.25736)**

> **作者:** Akihiro Kubota; Tomoya Hasegawa; Ryo Kawahara; Ko Nishino
>
> **摘要:** Gauging an individual's skill level is crucial, as it inherently shapes their behavior. Quantifying skill, however, is challenging because it is latent to the observed actions. To explore skill understanding in human behavior, we focus on dyadic sports -- specifically table tennis -- where skill manifests not just in complex movements, but in the subtle nuances of execution conditioned on game context. Our key idea is to learn a generative model of each player's tactical racket strokes and jointly embed them in a common latent space that encodes individual characteristics, including those pertaining to skill levels. By training these player models on a large-scale dataset of 3D-reconstructed professional matches and conditioning them on comprehensive game context -- including player positioning and opponent behaviors -- the models capture individual tactical identities within their latent space. We probe this learned player space and find that it reflects distinct play styles and attributes that collectively represent skill. By training a simple relative ranking network on these embeddings, we demonstrate that both relative and absolute skill predictions can be achieved. These results demonstrate that the learned player space effectively quantifies skill levels, providing a foundation for automated skill assessment in complex, interactive behaviors.
>
---
#### [new 053] Visual Attention Drifts,but Anchors Hold:Mitigating Hallucination in Multimodal Large Language Models via Cross-Layer Visual Anchors
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型任务，解决对象幻觉问题。通过引入跨层视觉锚点（CLVA），抑制深层注意力回归噪声，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25088](https://arxiv.org/pdf/2603.25088)**

> **作者:** Chengxu Yang; Jingling Yuan; Chuang Hu; Jiawei Jiang
>
> **摘要:** Multimodal Large Language Models often suffer from object hallucination. While existing research utilizes attention enhancement and visual retracing, we find these works lack sufficient interpretability regarding attention drift in final model stages. In this paper, we investigate the layer wise evolution of visual features and discover that hallucination stems from deep layer attention regressing toward initial visual noise from early layers. We observe that output reliability depends on acquiring visual anchors at intermediate layers rather than final layers. Based on these insights, we propose CLVA, which stands for Cross-Layer Visual Anchors, a training free method that reinforces critical mid layer features while suppressing regressive noise. This approach effectively pulls deep layer attention back to correct visual regions by utilizing essential anchors captured from attention dynamics. We evaluate our method across diverse architectures and benchmarks, demonstrating outstanding performance without significant increase in computational time and GPU memory.
>
---
#### [new 054] Robust Principal Component Completion
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出RPCC方法，用于从数据中分离低秩背景和稀疏前景，解决RPCA在遮挡场景下的不足，通过概率模型实现更准确的异常检测与前景提取。**

- **链接: [https://arxiv.org/pdf/2603.25132](https://arxiv.org/pdf/2603.25132)**

> **作者:** Yinjian Wang; Wei Li; Yuanyuan Gui; James E. Fowler; Gemine Vivone
>
> **摘要:** Robust principal component analysis (RPCA) seeks a low-rank component and a sparse component from their summation. Yet, in many applications of interest, the sparse foreground actually replaces, or occludes, elements from the low-rank background. To address this mismatch, a new framework is proposed in which the sparse component is identified indirectly through determining its support. This approach, called robust principal component completion (RPCC), is solved via variational Bayesian inference applied to a fully probabilistic Bayesian sparse tensor factorization. Convergence to a hard classifier for the support is shown, thereby eliminating the post-hoc thresholding required of most prior RPCA-driven approaches. Experimental results reveal that the proposed approach delivers near-optimal estimates on synthetic data as well as robust foreground-extraction and anomaly-detection performance on real color video and hyperspectral datasets, respectively. Source implementation and Appendices are available at this https URL.
>
---
#### [new 055] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于OOD检测任务，旨在解决传统方法中负类标签激活不足的问题。通过动态评估测试阶段的标签激活，提出TANL方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.25250](https://arxiv.org/pdf/2603.25250)**

> **作者:** Yabin Zhang; Maya Varma; Yunhe Gao; Jean-Benoit Delbrouck; Jiaming Liu; Chong Wang; Curtis Langlotz
>
> **备注:** CVPR 2026 main track, Codes are available at this https URL
>
> **摘要:** Out-of-distribution (OOD) detection aims to identify samples that deviate from in-distribution (ID). One popular pipeline addresses this by introducing negative labels distant from ID classes and detecting OOD based on their distance to these labels. However, such labels may present poor activation on OOD samples, failing to capture the OOD characteristics. To address this, we propose \underline{T}est-time \underline{A}ctivated \underline{N}egative \underline{L}abels (TANL) by dynamically evaluating activation levels across the corpus dataset and mining candidate labels with high activation responses during the testing process. Specifically, TANL identifies high-confidence test images online and accumulates their assignment probabilities over the corpus to construct a label activation metric. Such a metric leverages historical test samples to adaptively align with the test distribution, enabling the selection of distribution-adaptive activated negative labels. By further exploring the activation information within the current testing batch, we introduce a more fine-grained, batch-adaptive variant. To fully utilize label activation knowledge, we propose an activation-aware score function that emphasizes negative labels with stronger activations, boosting performance and enhancing its robustness to the label number. Our TANL is training-free, test-efficient, and grounded in theoretical justification. Experiments on diverse backbones and wide task settings validate its effectiveness. Notably, on the large-scale ImageNet benchmark, TANL significantly reduces the FPR95 from 17.5\% to 9.8\%. Codes are available at \href{this https URL}{YBZh/OpenOOD-VLM}.
>
---
#### [new 056] Label What Matters: Modality-Balanced and Difficulty-Aware Multimodal Active Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决标注数据稀缺下的主动学习问题。提出RL-MBA框架，动态平衡模态贡献并感知样本难度，提升模型性能与公平性。**

- **链接: [https://arxiv.org/pdf/2603.25107](https://arxiv.org/pdf/2603.25107)**

> **作者:** Yuqiao Zeng; Xu Wang; Tengfei Liang; Yiqing Hao; Yi Jin; Hui Yu
>
> **摘要:** Multimodal learning integrates complementary information from different modalities such as image, text, and audio to improve model performance, but its success relies on large-scale labeled data, which is costly to obtain. Active learning (AL) mitigates this challenge by selectively annotating informative samples. In multimodal settings, many approaches implicitly assume that modality importance is stable across rounds and keep selection rules fixed at the fusion stage, which leaves them insensitive to the dynamic nature of multimodal learning, where the relative value of modalities and the difficulty of instances shift as training proceeds. To address this issue, we propose RL-MBA, a reinforcement-learning framework for modality-balanced, difficulty-aware multimodal active learning. RL-MBA models sample selection as a Markov Decision Process, where the policy adapts to modality contributions, uncertainty, and diversity, and the reward encourages accuracy gains and balance. Two key components drive this adaptability: (1) Adaptive Modality Contribution Balancing (AMCB), which dynamically adjusts modality weights via reinforcement feedback, and (2) Evidential Fusion for DifficultyAware Policy Adjustment (EFDA), which estimates sample difficulty via uncertainty-based evidential fusion to prioritize informative samples. Experiments on Food101, KineticsSound, and VGGSound demonstrate that RL-MBA consistently outperforms strong baselines, improving both classification accuracy and modality fairness under limited labeling budgets.
>
---
#### [new 057] Photon: Speedup Volume Understanding with Efficient Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉问答任务，解决3D医学影像处理中计算成本高和信息丢失的问题。提出Photon框架，通过可变长度token序列和自适应压缩技术提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.25155](https://arxiv.org/pdf/2603.25155)**

> **作者:** Chengyu Fang; Heng Guo; Zheng Jiang; Chunming He; Xiu Li; Minfeng Xu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Multimodal large language models are promising for clinical visual question answering tasks, but scaling to 3D imaging is hindered by high computational costs. Prior methods often rely on 2D slices or fixed-length token compression, disrupting volumetric continuity and obscuring subtle findings. We present Photon, a framework that represents 3D medical volumes with token sequences of variable length. Photon introduces instruction-conditioned token scheduling and surrogate gradient propagation to adaptively reduce tokens during both training and inference, which lowers computational cost while mitigating the attention dilution caused by redundant tokens. It incorporates a custom backpropagation rule with gradient restoration to enable differentiable optimization despite discrete token drop. To stabilize token compression and ensure reliable use of visual evidence, Photon further applies regularization objectives that mitigate language-only bias and improve reliability. Experiments on diverse medical visual question answering tasks show that Photon achieves state-of-the-art accuracy while reducing resource usage and accelerating both training and inference.
>
---
#### [new 058] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文属于动态场景重建任务，旨在解决单目视频中高精度动态高斯点云的生成问题。通过显式建模连续运动，提升新视角合成效果。**

- **链接: [https://arxiv.org/pdf/2603.25058](https://arxiv.org/pdf/2603.25058)**

> **作者:** Xuankai Zhang; Junjin Xiao; Shangwei Huang; Wei-shi Zheng; Qing Zhang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present an approach for high-quality dynamic Gaussian Splatting from monocular videos. To this end, we in this work go one step further beyond previous methods to explicitly model continuous position and orientation deformation of dynamic Gaussians, using an SE(3) B-spline motion bases with a compact set of control points. To improve computational efficiency while enhancing the ability to model complex motions, an adaptive control mechanism is devised to dynamically adjust the number of motion bases and control points. Besides, we develop a soft segment reconstruction strategy to mitigate long-interval motion interference, and employ a multi-view diffusion model to provide multi-view cues for avoiding overfitting to training views. Extensive experiments demonstrate that our method outperforms state-of-the-art methods in novel view synthesis. Our code is available at this https URL.
>
---
#### [new 059] MegaFlow: Zero-Shot Large Displacement Optical Flow
- **分类: cs.CV**

- **简介: 该论文属于光学流估计任务，解决大位移和零样本泛化问题。通过引入MegaFlow模型，利用预训练视觉先验进行全局匹配，提升准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.25739](https://arxiv.org/pdf/2603.25739)**

> **作者:** Dingxi Zhang; Fangjinhua Wang; Marc Pollefeys; Haofei Xu
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Accurate estimation of large displacement optical flow remains a critical challenge. Existing methods typically rely on iterative local search or/and domain-specific fine-tuning, which severely limits their performance in large displacement and zero-shot generalization scenarios. To overcome this, we introduce MegaFlow, a simple yet powerful model for zero-shot large displacement optical flow. Rather than relying on highly complex, task-specific architectural designs, MegaFlow adapts powerful pre-trained vision priors to produce temporally consistent motion fields. In particular, we formulate flow estimation as a global matching problem by leveraging pre-trained global Vision Transformer features, which naturally capture large displacements. This is followed by a few lightweight iterative refinements to further improve the sub-pixel accuracy. Extensive experiments demonstrate that MegaFlow achieves state-of-the-art zero-shot performance across multiple optical flow benchmarks. Moreover, our model also delivers highly competitive zero-shot performance on long-range point tracking benchmarks, demonstrating its robust transferability and suggesting a unified paradigm for generalizable motion estimation. Our project page is at: this https URL.
>
---
#### [new 060] PixelSmile: Toward Fine-Grained Facial Expression Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于面部表情编辑任务，旨在解决语义重叠导致的细粒度表达编辑难题。通过构建数据集和提出PixelSmile框架，实现更精确、稳定的表情控制与身份保留。**

- **链接: [https://arxiv.org/pdf/2603.25728](https://arxiv.org/pdf/2603.25728)**

> **作者:** Jiabin Hua; Hengyuan Xu; Aojie Li; Wei Cheng; Gang Yu; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 21 Pages; Project Page: this https URL Code: this https URL
>
> **摘要:** Fine-grained facial expression editing has long been limited by intrinsic semantic overlap. To address this, we construct the Flex Facial Expression (FFE) dataset with continuous affective annotations and establish FFE-Bench to evaluate structural confusion, editing accuracy, linear controllability, and the trade-off between expression editing and identity preservation. We propose PixelSmile, a diffusion framework that disentangles expression semantics via fully symmetric joint training. PixelSmile combines intensity supervision with contrastive learning to produce stronger and more distinguishable expressions, achieving precise and stable linear expression control through textual latent interpolation. Extensive experiments demonstrate that PixelSmile achieves superior disentanglement and robust identity preservation, confirming its effectiveness for continuous, controllable, and fine-grained expression editing, while naturally supporting smooth expression blending.
>
---
#### [new 061] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于3D异常检测任务，解决统一模型中类别语义混淆问题。通过引入语义解耦机制，提升异常检测的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25159](https://arxiv.org/pdf/2603.25159)**

> **作者:** SuYeon Kim; Wongyu Lee; MyeongAh Cho
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** 3D anomaly detection targets the detection and localization of defects in 3D point clouds trained solely on normal data. While a unified model improves scalability by learning across multiple categories, it often suffers from Inter-Category Entanglement (ICE)-where latent features from different categories overlap, causing the model to adopt incorrect semantic priors during reconstruction and ultimately yielding unreliable anomaly scores. To address this issue, we propose the Semantically Disentangled Unified Model for 3D Anomaly Detection, which reconstructs features conditioned on disentangled semantic representations. Our framework consists of three key components: (i) Coarse-to-Fine Global Tokenization for forming instance-level semantic identity, (ii) Category-Conditioned Contrastive Learning for disentangling category semantics, and (iii) a Geometry-Guided Decoder for semantically consistent reconstruction. Extensive experiments on Real3D-AD and Anomaly-ShapeNet demonstrate that our method achieves state-of-the-art for both unified and category-specific models, improving object-level AUROC by 2.8% and 9.1%, respectively, while enhancing the reliability of unified 3D anomaly detection.
>
---
#### [new 062] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态虚假信息检测任务，旨在解决传统检测方法脆弱、不透明的问题。提出PCGR框架，通过构建概念图进行可解释的推理，提升检测准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.25203](https://arxiv.org/pdf/2603.25203)**

> **作者:** Ruichao Yang; Wei Gao; Xiaobin Zhu; Jing Ma; Hongzhan Lin; Ziyang Luo; Bo-Wen Zhang; Xu-Cheng Yin
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multimodal misinformation poses an escalating challenge that often evades traditional detectors, which are opaque black boxes and fragile against new manipulation tactics. We present Probabilistic Concept Graph Reasoning (PCGR), an interpretable and evolvable framework that reframes multimodal misinformation detection (MMD) as structured and concept-based reasoning. PCGR follows a build-then-infer paradigm, which first constructs a graph of human-understandable concept nodes, including novel high-level concepts automatically discovered and validated by multimodal large language models (MLLMs), and then applies hierarchical attention over this concept graph to infer claim veracity. This design produces interpretable reasoning chains linking evidence to conclusions. Experiments demonstrate that PCGR achieves state-of-the-art MMD accuracy and robustness to emerging manipulation types, outperforming prior methods in both coarse detection and fine-grained manipulation recognition.
>
---
#### [new 063] LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决行星科学中缺乏多模态数据的问题。研究构建了LUCID数据集，并提出LLaVA-LE模型，提升月球地形分析能力。**

- **链接: [https://arxiv.org/pdf/2603.24696](https://arxiv.org/pdf/2603.24696)**

> **作者:** Gokce Inal; Pouyan Navard; Alper Yilmaz
>
> **备注:** Accepted in AI4Space Workshop CVPR2026. Website: this https URL, Dataset: this https URL
>
> **摘要:** Recent advances in multimodal vision-language models (VLMs) have enabled joint reasoning over visual and textual information, yet their application to planetary science remains largely unexplored. A key hindrance is the absence of large-scale datasets that pair real planetary imagery with detailed scientific descriptions. In this work, we introduce LLaVA-LE (Large Language-and-Vision Assistant for Lunar Exploration), a vision-language model specialized for lunar surface and subsurface characterization. To enable this capability, we curate a new large-scale multimodal lunar dataset, LUCID (LUnar Caption Image Dataset) consisting of 96k high-resolution panchromatic images paired with detailed captions describing lunar terrain characteristics, and 81k question-answer (QA) pairs derived from approximately 20k images in the LUCID dataset. Leveraging this dataset, we fine-tune LLaVA using a two-stage training curriculum: (1) concept alignment for domain-specific terrain description, and (2) instruction-tuned visual question answering. We further design evaluation benchmarks spanning multiple levels of reasoning complexity relevant to lunar terrain analysis. Evaluated against GPT and Gemini judges, LLaVA-LE achieves a 3.3x overall performance gain over Base LLaVA and 2.1x over our Stage 1 model, with a reasoning score of 1.070, exceeding the judge's own reference score, highlighting the effectiveness of domain-specific multimodal data and instruction tuning to advance VLMs in planetary exploration. Code is available at this https URL.
>
---
#### [new 064] PackForcing: Short Video Training Suffices for Long Video Sampling and Long Context Inference
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PackForcing，解决长视频生成中的KV缓存膨胀和时间重复问题。通过三段式缓存策略实现高效上下文管理，提升长视频生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.25730](https://arxiv.org/pdf/2603.25730)**

> **作者:** Xiaofeng Mao; Shaohao Rui; Kaining Ying; Bo Zheng; Chuanhao Li; Mingmin Chi; Kaipeng Zhang
>
> **摘要:** Autoregressive video diffusion models have demonstrated remarkable progress, yet they remain bottlenecked by intractable linear KV-cache growth, temporal repetition, and compounding errors during long-video generation. To address these challenges, we present PackForcing, a unified framework that efficiently manages the generation history through a novel three-partition KV-cache strategy. Specifically, we categorize the historical context into three distinct types: (1) Sink tokens, which preserve early anchor frames at full resolution to maintain global semantics; (2) Mid tokens, which achieve a massive spatiotemporal compression (32x token reduction) via a dual-branch network fusing progressive 3D convolutions with low-resolution VAE re-encoding; and (3) Recent tokens, kept at full resolution to ensure local temporal coherence. To strictly bound the memory footprint without sacrificing quality, we introduce a dynamic top-$k$ context selection mechanism for the mid tokens, coupled with a continuous Temporal RoPE Adjustment that seamlessly re-aligns position gaps caused by dropped tokens with negligible overhead. Empowered by this principled hierarchical context compression, PackForcing can generate coherent 2-minute, 832x480 videos at 16 FPS on a single H200 GPU. It achieves a bounded KV cache of just 4 GB and enables a remarkable 24x temporal extrapolation (5s to 120s), operating effectively either zero-shot or trained on merely 5-second clips. Extensive results on VBench demonstrate state-of-the-art temporal consistency (26.07) and dynamic degree (56.25), proving that short-video supervision is sufficient for high-quality, long-video synthesis. this https URL
>
---
#### [new 065] Efficient Preemptive Robustification with Image Sharpening
- **分类: cs.CV**

- **简介: 该论文属于对抗鲁棒性任务，旨在提升深度神经网络对对抗扰动的抵抗力。通过图像锐化实现无需代理、优化和生成器的高效鲁棒化方法。**

- **链接: [https://arxiv.org/pdf/2603.25244](https://arxiv.org/pdf/2603.25244)**

> **作者:** Jiaming Liang; Chi-Man Pun
>
> **摘要:** Despite their great success, deep neural networks rely on high-dimensional, non-robust representations, making them vulnerable to imperceptible perturbations, even in transfer scenarios. To address this, both training-time defenses (e.g., adversarial training and robust architecture design) and post-attack defenses (e.g., input purification and adversarial detection) have been extensively studied. Recently, a limited body of work has preliminarily explored a pre-attack defense paradigm, termed preemptive robustification, which introduces subtle modifications to benign samples prior to attack to proactively resist adversarial perturbations. Unfortunately, their practical applicability remains questionable due to several limitations, including (1) reliance on well-trained classifiers as surrogates to provide robustness priors, (2) substantial computational overhead arising from iterative optimization or trained generators for robustification, and (3) limited interpretability of the optimization- or generation-based robustification processes. Inspired by recent studies revealing a positive correlation between texture intensity and the robustness of benign samples, we show that image sharpening alone can efficiently robustify images. To the best of our knowledge, this is the first surrogate-free, optimization-free, generator-free, and human-interpretable robustification approach. Extensive experiments demonstrate that sharpening yields remarkable robustness gains with low computational cost, especially in transfer scenarios.
>
---
#### [new 066] Beyond Attention Magnitude: Leveraging Inter-layer Rank Consistency for Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决推理延迟过高的问题。通过引入TIES框架，利用层间排名一致性动态选择关键token，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.24941](https://arxiv.org/pdf/2603.24941)**

> **作者:** Peiju Liu; Jinming Liu; Xipeng Qiu; Xuanjing Huang
>
> **备注:** 10 pages, 7 figures, preprint
>
> **摘要:** Vision-Language-Action (VLA) models excel in robotic manipulation but suffer from significant inference latency due to processing dense visual tokens. Existing token reduction methods predominantly rely on attention magnitude as a static selection. In this work, we challenge this assumption, revealing that high-attention tokens are task-dependent and can even degrade policy performance. To address this, we introduce \textbf{TIES} (\textbf{T}au-guided \textbf{I}nter-layer \textbf{E}fficient \textbf{S}election), a dynamic framework guided by inter-layer token ranking consistency. By adaptively balancing attention magnitude with ranking consistency, TIES ensures robust token selection without requiring additional training. On the CogACT + SIMPLER benchmark, TIES improves average success rates by 6\% while reducing token usage by 78\%, and demonstrate strong generalization across diverse decoders and benchmarks.
>
---
#### [new 067] V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception
- **分类: cs.CV**

- **简介: 该论文提出V2U4Real数据集，解决车辆与无人机协同感知问题，通过多模态数据提升复杂环境下的目标检测与跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.25275](https://arxiv.org/pdf/2603.25275)**

> **作者:** Weijia Li; Haoen Xiang; Tianxu Wang; Shuaibing Wu; Qiming Xia; Cheng Wang; Chenglu Wen
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Modern autonomous vehicle perception systems are often constrained by occlusions, blind spots, and limited sensing range. While existing cooperative perception paradigms, such as Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I), have demonstrated their effectiveness in mitigating these challenges, they remain limited to ground-level collaboration and cannot fully address large-scale occlusions or long-range perception in complex environments. To advance research in cross-view cooperative perception, we present V2U4Real, the first large-scale real-world multi-modal dataset for Vehicle-to-UAV (V2U) cooperative object perception. V2U4Real is collected by a ground vehicle and a UAV equipped with multi-view LiDARs and RGB cameras. The dataset covers urban streets, university campuses, and rural roads under diverse traffic scenarios, comprising over 56K LiDAR frames, 56K multi-view camera images, and 700K annotated 3D bounding boxes across four classes. To support a wide range of research tasks, we establish benchmarks for single-agent 3D object detection, cooperative 3D object detection, and object tracking. Comprehensive evaluations of several state-of-the-art models demonstrate the effectiveness of V2U cooperation in enhancing perception robustness and long-range awareness. The V2U4Real dataset and codebase is available at this https URL.
>
---
#### [new 068] AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization
- **分类: cs.CV**

- **简介: 该论文提出AnyDoc，解决文档生成任务中的数据不足与内容溢出问题，通过合成HTML/CSS数据并引入强化学习优化生成效果。**

- **链接: [https://arxiv.org/pdf/2603.25118](https://arxiv.org/pdf/2603.25118)**

> **作者:** Jiawei Lin; Wanrong Zhu; Vlad I Morariu; Christopher Tensmeyer
>
> **备注:** CVPR 2026 Main Conference
>
> **摘要:** Document generation has gained growing attention in the field of AI-driven content creation. In this work, we push its boundaries by introducing AnyDoc, a framework capable of handling multiple generation tasks across a wide spectrum of document categories, all represented in a unified HTML/CSS format. To overcome the limited coverage and scale of existing human-crafted document datasets, AnyDoc first establishes a scalable data synthesis pipeline to automatically generate documents in HTML/CSS form. This pipeline yields DocHTML, a large-scale dataset containing 265,206 document samples, while spanning 111 categories and 32 distinct styles. Additionally, all documents are equipped with comprehensive metadata, including design intentions, HTML/CSS source code, visual assets, and rendered screenshots. Building on the curated dataset, AnyDoc fine-tunes multi-modal large language models (MLLMs) to achieve three practical document generation tasks: intention-to-document, document derendering, and element-to-document. To address the content overflow issue observed during fine-tuning, AnyDoc further incorporates a height-aware reinforcement learning (HARL) post-training procedure. By defining a reward function based on the difference between predicted and target document heights, overflow is penalized and gradually mitigated during HARL, thereby enhancing overall performance. Qualitative and quantitative experiments demonstrate that AnyDoc outperforms both general-purpose MLLMs and task-specific baselines across all three tasks.
>
---
#### [new 069] MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data
- **分类: cs.CV**

- **简介: 该论文属于多参考图像生成任务，解决输入参考增多导致性能下降的问题。构建了大规模数据集MacroData和基准MacroBench，提升多参考生成效果。**

- **链接: [https://arxiv.org/pdf/2603.25319](https://arxiv.org/pdf/2603.25319)**

> **作者:** Zhekai Chen; Yuqing Wang; Manyuan Zhang; Xihui Liu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Generating images conditioned on multiple visual references is critical for real-world applications such as multi-subject composition, narrative illustration, and novel view synthesis, yet current models suffer from severe performance degradation as the number of input references grows. We identify the root cause as a fundamental data bottleneck: existing datasets are dominated by single- or few-reference pairs and lack the structured, long-context supervision needed to learn dense inter-reference dependencies. To address this, we introduce MacroData, a large-scale dataset of 400K samples, each containing up to 10 reference images, systematically organized across four complementary dimensions -- Customization, Illustration, Spatial reasoning, and Temporal dynamics -- to provide comprehensive coverage of the multi-reference generation space. Recognizing the concurrent absence of standardized evaluation protocols, we further propose MacroBench, a benchmark of 4,000 samples that assesses generative coherence across graded task dimensions and input scales. Extensive experiments show that fine-tuning on MacroData yields substantial improvements in multi-reference generation, and ablation studies further reveal synergistic benefits of cross-task co-training and effective strategies for handling long-context complexity. The dataset and benchmark will be publicly released.
>
---
#### [new 070] BiFM: Bidirectional Flow Matching for Few-Step Image Editing and Generation
- **分类: cs.CV**

- **简介: 该论文提出BiFM，解决少步图像编辑与生成中的质量下降问题。通过双向流匹配，统一生成与逆向过程，提升编辑效果与模型通用性。**

- **链接: [https://arxiv.org/pdf/2603.24942](https://arxiv.org/pdf/2603.24942)**

> **作者:** Yasong Dai; Zeeshan Hayder; David Ahmedt-Aristizabal; Hongdong Li
>
> **备注:** Accepted in CVPR2026
>
> **摘要:** Recent diffusion and flow matching models have demonstrated strong capabilities in image generation and editing by progressively removing noise through iterative sampling. While this enables flexible inversion for semantic-preserving edits, few-step sampling regimes suffer from poor forward process approximation, leading to degraded editing quality. Existing few-step inversion methods often rely on pretrained generators and auxiliary modules, limiting scalability and generalization across different architectures. To address these limitations, we propose BiFM (Bidirectional Flow Matching), a unified framework that jointly learns generation and inversion within a single model. BiFM directly estimates average velocity fields in both ``image $\to$ noise" and ``noise $\to$ image" directions, constrained by a shared instantaneous velocity field derived from either predefined schedules or pretrained multi-step diffusion models. Additionally, BiFM introduces a novel training strategy using continuous time-interval supervision, stabilized by a bidirectional consistency objective and a lightweight time-interval embedding. This bidirectional formulation also enables one-step inversion and can integrate seamlessly into popular diffusion and flow matching backbones. Across diverse image editing and generation tasks, BiFM consistently outperforms existing few-step approaches, achieving superior performance and editability.
>
---
#### [new 071] HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT
- **分类: cs.CV**

- **简介: 该论文针对VGGT模型中全局注意力层计算成本高的问题，提出HeSS方法优化稀疏化策略，提升模型在高稀疏度下的性能。**

- **链接: [https://arxiv.org/pdf/2603.25336](https://arxiv.org/pdf/2603.25336)**

> **作者:** Yongsung Kim; Wooseok Song; Jaihyun Lew; Hun Hwangbo; Jaehoon Lee; Sungroh Yoon
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Visual Geometry Grounded Transformer (VGGT) has advanced 3D vision, yet its global attention layers suffer from quadratic computational costs that hinder scalability. Several sparsification-based acceleration techniques have been proposed to alleviate this issue, but they often suffer from substantial accuracy degradation. We hypothesize that the accuracy degradation stems from the heterogeneity in head-wise sparsification sensitivity, as the existing methods apply a uniform sparsity pattern across all heads. Motivated by this hypothesis, we present a two-stage sparsification pipeline that effectively quantifies and exploits headwise sparsification sensitivity. In the first stage, we measure head-wise sparsification sensitivity using a novel metric, the Head Sensitivity Score (HeSS), which approximates the Hessian with respect to two distinct error terms on a small calibration set. In the inference stage, we perform HeSS-Guided Sparsification, leveraging the pre-computed HeSS to reallocate the total attention budget-assigning denser attention to sensitive heads and sparser attention to more robust ones. We demonstrate that HeSS effectively captures head-wise sparsification sensitivity and empirically confirm that attention heads in the global attention layers exhibit heterogeneous sensitivity characteristics. Extensive experiments further show that our method effectively mitigates performance degradation under high sparsity, demonstrating strong robustness across varying sparsification levels. Code is available at this https URL.
>
---
#### [new 072] RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决真实场景下图像恢复模型泛化能力不足的问题。通过构建大规模数据集和训练开源模型，提升恢复效果与一致性。**

- **链接: [https://arxiv.org/pdf/2603.25502](https://arxiv.org/pdf/2603.25502)**

> **作者:** Yufeng Yang; Xianfang Zeng; Zhangqi Jiang; Fukun Yin; Jianzhuang Liu; Wei Cheng; jinghong lan; Shiyu Liu; Yuqi Peng; Gang YU; Shifeng Chen
>
> **备注:** 27 pages, 15 figures, Project homepage: this https URL
>
> **摘要:** Image restoration under real-world degradations is critical for downstream tasks such as autonomous driving and object detection. However, existing restoration models are often limited by the scale and distribution of their training data, resulting in poor generalization to real-world scenarios. Recently, large-scale image editing models have shown strong generalization ability in restoration tasks, especially for closed-source models like Nano Banana Pro, which can restore images while preserving consistency. Nevertheless, achieving such performance with those large universal models requires substantial data and computational costs. To address this issue, we construct a large-scale dataset covering nine common real-world degradation types and train a state-of-the-art open-source model to narrow the gap with closed-source alternatives. Furthermore, we introduce RealIR-Bench, which contains 464 real-world degraded images and tailored evaluation metrics focusing on degradation removal and consistency preservation. Extensive experiments demonstrate our model ranks first among open-source methods, achieving state-of-the-art performance.
>
---
#### [new 073] SurgPhase: Time efficient pituitary tumor surgery phase recognition via an interactive web platform
- **分类: cs.CV**

- **简介: 该论文属于手术阶段识别任务，旨在提升垂体瘤手术视频的阶段分析效率。通过自监督学习和协同平台，实现高精度、可扩展的手术阶段识别。**

- **链接: [https://arxiv.org/pdf/2603.24897](https://arxiv.org/pdf/2603.24897)**

> **作者:** Yan Meng; Jack Cook; X.Y. Han; Kaan Duman; Shauna Otto; Dhiraj Pangal; Jonathan Chainey; Ruth Lau; Margaux Masson-Forsythe; Daniel A. Donoho; Danielle Levy; Gabriel Zada; Sébastien Froelich; Juan Fernandez-Miranda; Mike Chang
>
> **摘要:** Accurate surgical phase recognition is essential for analyzing procedural workflows, supporting intraoperative decision-making, and enabling data-driven improvements in surgical education and performance evaluation. In this work, we present a comprehensive framework for phase recognition in pituitary tumor surgery (PTS) videos, combining self-supervised representation learning, robust temporal modeling, and scalable data annotation strategies. Our method achieves 90\% accuracy on a held-out test set, outperforming current state-of-the-art approaches and demonstrating strong generalization across variable surgical cases. A central contribution of this work is the integration of a collaborative online platform designed for surgeons to upload surgical videos, receive automated phase analysis, and contribute to a growing dataset. This platform not only facilitates large-scale data collection but also fosters knowledge sharing and continuous model improvement. To address the challenge of limited labeled data, we pretrain a ResNet-50 model using the self-supervised framework on 251 unlabeled PTS videos, enabling the extraction of high-quality feature representations. Fine-tuning is performed on a labeled dataset of 81 procedures using a modified training regime that incorporates focal loss, gradual layer unfreezing, and dynamic sampling to address class imbalance and procedural variability.
>
---
#### [new 074] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval
- **分类: cs.CV**

- **简介: 该论文提出EagleNet，解决文本-视频检索中文本无法捕捉视频帧上下文的问题。通过构建文本-帧图并学习细粒度关系，生成更具上下文感知的文本嵌入。**

- **链接: [https://arxiv.org/pdf/2603.25267](https://arxiv.org/pdf/2603.25267)**

> **作者:** Yuhan Chen; Pengwen Dai; Chuan Wang; Dayan Wu; Xiaochun Cao
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Text-video retrieval tasks have seen significant improvements due to the recent development of large-scale vision-language pre-trained models. Traditional methods primarily focus on video representations or cross-modal alignment, while recent works shift toward enriching text expressiveness to better match the rich semantics in videos. However, these methods use only interactions between text and frames/video, and ignore rich interactions among the internal frames within a video, so the final expanded text cannot capture frame contextual information, leading to disparities between text and video. In response, we introduce Energy-Aware Fine-Grained Relationship Learning Network (EagleNet) to generate accurate and context-aware enriched text embeddings. Specifically, the proposed Fine-Grained Relationship Learning mechanism (FRL) first constructs a text-frame graph by the generated text candidates and frames, then learns relationships among texts and frames, which are finally used to aggregate text candidates into an enriched text embedding that incorporates frame contextual information. To further improve fine-grained relationship learning in FRL, we design Energy-Aware Matching (EAM) to model the energy of text-frame interactions and thus accurately capture the distribution of real text-video pairs. Moreover, for more effective cross-modal alignment and stable training, we replace the conventional softmax-based contrastive loss with the sigmoid loss. Extensive experiments have demonstrated the superiority of EagleNet across MSRVTT, DiDeMo, MSVD, and VATEX. Codes are available at this https URL.
>
---
#### [new 075] Semantic-Aware Prefix Learning for Token-Efficient Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有视觉分词器语义对齐不足的问题。提出SMAP框架，通过注入语义条件提升潜在表示的语义基础，增强生成效果。**

- **链接: [https://arxiv.org/pdf/2603.25249](https://arxiv.org/pdf/2603.25249)**

> **作者:** Qingfeng Li; Haoxian Zhang; Xu He; Songlin Tang; Zhixue Fang; Xiaoqiang Liu; Pengfei Wan Guoqi Li
>
> **摘要:** Visual tokenizers play a central role in latent image generation by bridging high-dimensional images and tractable generative modeling. However, most existing tokenizers are still trained with reconstruction-dominated objectives, which often yield latent representations that are only weakly grounded in high-level semantics. Recent approaches improve semantic alignment, but typically treat semantic signals as auxiliary regularization rather than making them functionally necessary for representation learning. We propose SMAP, a SeMantic-Aware Prefix tokenizer that injects class-level semantic conditions into a query-based 1D tokenization framework. To make semantics indispensable during training, SMAP introduces a tail token dropping strategy, which forces semantic conditions and early latent prefixes to bear increasing responsibility under progressively reduced token budgets. To verify that the resulting latent space is useful for generation rather than reconstruction alone, we further introduce CARD, a hybrid Causal AutoRegressive--Diffusion generator. Extensive experiments on ImageNet show that SMAP consistently improves reconstruction quality across discrete and continuous tokenization settings, and that its semantically grounded latent space yields strong downstream generation performance under compact token budgets.
>
---
#### [new 076] GeoHeight-Bench: Towards Height-Aware Multimodal Reasoning in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感多模态推理任务，旨在解决模型对垂直维度感知不足的问题。通过构建基准和提出新模型，提升对地形高度的综合理解能力。**

- **链接: [https://arxiv.org/pdf/2603.25565](https://arxiv.org/pdf/2603.25565)**

> **作者:** Xuran Hu; Zhitong Xiong; Zhongcheng Hong; Yifang Ban; Xiaoxiang Zhu; Wufan Zhao
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Current Large Multimodal Models (LMMs) in Earth Observation typically neglect the critical "vertical" dimension, limiting their reasoning capabilities in complex remote sensing geometries and disaster scenarios where physical spatial structures often outweigh planar visual textures. To bridge this gap, we introduce a comprehensive evaluation framework dedicated to height-aware remote sensing understanding. First, to overcome the severe scarcity of annotated data, we develop a scalable, VLM-driven data generation pipeline utilizing systematic prompt engineering and metadata extraction. This pipeline constructs two complementary benchmarks: GeoHeight-Bench for relative height analysis, and a more challenging GeoHeight-Bench+ for holistic, terrain-aware reasoning. Furthermore, to validate the necessity of height perception, we propose GeoHeightChat, the first height-aware remote sensing LMM baseline. Serving as a strong proof of concept, our baseline demonstrates that synergizing visual semantics with implicitly injected height geometric features effectively mitigates the "vertical blind spot", successfully unlocking a new paradigm of interactive height reasoning in existing optical models.
>
---
#### [new 077] RefAlign: Representation Alignment for Reference-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文属于参考图像到视频生成任务，解决多主体混淆和复制粘贴伪影问题。提出RefAlign框架，通过显式对齐参考特征与视觉基础模型语义空间，提升身份一致性和语义区分度。**

- **链接: [https://arxiv.org/pdf/2603.25743](https://arxiv.org/pdf/2603.25743)**

> **作者:** Lei Wang; YuXin Song; Ge Wu; Haocheng Feng; Hang Zhou; Jingdong Wang; Yaxing Wang; jian Yang
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Reference-to-video (R2V) generation is a controllable video synthesis paradigm that constrains the generation process using both text prompts and reference images, enabling applications such as personalized advertising and virtual try-on. In practice, existing R2V methods typically introduce additional high-level semantic or cross-modal features alongside the VAE latent representation of the reference image and jointly feed them into the diffusion Transformer (DiT). These auxiliary representations provide semantic guidance and act as implicit alignment signals, which can partially alleviate pixel-level information leakage in the VAE latent space. However, they may still struggle to address copy--paste artifacts and multi-subject confusion caused by modality mismatch across heterogeneous encoder features. In this paper, we propose RefAlign, a representation alignment framework that explicitly aligns DiT reference-branch features to the semantic space of a visual foundation model (VFM). The core of RefAlign is a reference alignment loss that pulls the reference features and VFM features of the same subject closer to improve identity consistency, while pushing apart the corresponding features of different subjects to enhance semantic discriminability. This simple yet effective strategy is applied only during training, incurring no inference-time overhead, and achieves a better balance between text controllability and reference fidelity. Extensive experiments on the OpenS2V-Eval benchmark demonstrate that RefAlign outperforms current state-of-the-art methods in TotalScore, validating the effectiveness of explicit reference alignment for R2V tasks.
>
---
#### [new 078] AnyID: Ultra-Fidelity Universal Identity-Preserving Video Generation from Any Visual References
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决多源身份参考下的高保真身份保持问题。提出AnyID框架，实现多样本统一表示和属性级控制。**

- **链接: [https://arxiv.org/pdf/2603.25188](https://arxiv.org/pdf/2603.25188)**

> **作者:** Jiahao Wang; Hualian Sheng; Sijia Cai; Yuxiao Yang; Weizhan Zhang; Caixia Yan; Bing Deng; Jieping Ye
>
> **摘要:** Identity-preserving video generation offers powerful tools for creative expression, allowing users to customize videos featuring their beloved characters. However, prevailing methods are typically designed and optimized for a single identity reference. This underlying assumption restricts creative flexibility by inadequately accommodating diverse real-world input formats. Relying on a single source also constitutes an ill-posed scenario, causing an inherently ambiguous setting that makes it difficult for the model to faithfully reproduce an identity across novel contexts. To address these issues, we present AnyID, an ultra-fidelity identity-preservation video generation framework that features two core contributions. First, we introduce a scalable omni-referenced architecture that effectively unifies heterogeneous identity inputs (e.g., faces, portraits, and videos) into a cohesive representation. Second, we propose a primary-referenced generation paradigm, which designates one reference as a canonical anchor and uses a novel differential prompt to enable precise, attribute-level controllability. We conduct training on a large-scale, meticulously curated dataset to ensure robustness and high fidelity, and then perform a final fine-tuning stage using reinforcement learning. This process leverages a preference dataset constructed from human evaluations, where annotators performed pairwise comparisons of videos based on two key criteria: identity fidelity and prompt controllability. Extensive evaluations validate that AnyID achieves ultra-high identity fidelity as well as superior attribute-level controllability across different task settings.
>
---
#### [new 079] ReDiPrune: Relevance-Diversity Pre-Projection Token Pruning for Efficient Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文提出ReDiPrune，用于高效多模态大模型的视觉令牌剪枝。解决计算成本高的问题，通过预投影阶段选择相关且多样化的视觉令牌，提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2603.24680](https://arxiv.org/pdf/2603.24680)**

> **作者:** An Yu; Ting Yu Tsai; Zhenfei Zhang; Weiheng Lu; Felix X.-F. Ye; Ming-Ching Chang
>
> **摘要:** Recent multimodal large language models are computationally expensive because Transformers must process a large number of visual tokens. We present \textbf{ReDiPrune}, a training-free token pruning method applied before the vision-language projector, where visual features remain rich and discriminative. Unlike post-projection pruning methods that operate on compressed representations, ReDiPrune selects informative tokens directly from vision encoder outputs, preserving fine-grained spatial and semantic cues. Each token is scored by a lightweight rule that jointly consider text-conditioned relevance and max-min diversity, ensuring the selected tokens are both query-relevant and non-redundant. ReDiPrune is fully plug-and-play, requiring no retraining or architectural modifications, and can be seamlessly inserted between the encoder and projector. Across four video and five image benchmarks, it consistently improves the accuracy-efficiency trade-off. For example, on EgoSchema with LLaVA-NeXT-Video-7B, retaining only 15\% of visual tokens yields a +2.0\% absolute accuracy gain while reducing computation by more than $6\times$ in TFLOPs. Code is available at this https URL.
>
---
#### [new 080] AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于3D手部姿态估计任务，旨在解决真实数据集覆盖不足和合成数据缺乏遮挡等问题。工作包括构建大规模合成数据集AnyHand，并验证其提升模型性能的效果。**

- **链接: [https://arxiv.org/pdf/2603.25726](https://arxiv.org/pdf/2603.25726)**

> **作者:** Chen Si; Yulin Liu; Bo Ai; Jianwen Xie; Rolandos Alexandros Potamias; Chuanxia Zheng; Hao Su
>
> **摘要:** We present AnyHand, a large-scale synthetic dataset designed to advance the state of the art in 3D hand pose estimation from both RGB-only and RGB-D inputs. While recent works with foundation approaches have shown that an increase in the quantity and diversity of training data can markedly improve performance and robustness in hand pose estimation, existing real-world-collected datasets on this task are limited in coverage, and prior synthetic datasets rarely provide occlusions, arm details, and aligned depth together at scale. To address this bottleneck, our AnyHand contains 2.5M single-hand and 4.1M hand-object interaction RGB-D images, with rich geometric annotations. In the RGB-only setting, we show that extending the original training sets of existing baselines with AnyHand yields significant gains on multiple benchmarks (FreiHAND and HO-3D), even when keeping the architecture and training scheme fixed. More impressively, the model trained with AnyHand shows stronger generalization to the out-of-domain HO-Cap dataset, without any fine-tuning. We also contribute a lightweight depth fusion module that can be easily integrated into existing RGB-based models. Trained with AnyHand, the resulting RGB-D model achieves superior performance on the HO-3D benchmark, showing the benefits of depth integration and the effectiveness of our synthetic data.
>
---
#### [new 081] Synergistic Event-SVE Imaging for Quantitative Propellant Combustion Diagnostics
- **分类: cs.CV**

- **简介: 该论文属于燃烧诊断任务，解决高动态范围、烟雾干扰下的实时微秒级粒子运动测量问题。通过融合事件相机与SVE相机，实现3D粒子状态精确估计。**

- **链接: [https://arxiv.org/pdf/2603.25054](https://arxiv.org/pdf/2603.25054)**

> **作者:** Jing Tao; Taihang Lei; Banglei Guan; Ying Qu; Xudong Na; Likun Ma; Yang Shang; Qifeng Yu
>
> **摘要:** Real-time monitoring of high-energy propellant combustion is difficult. Extreme high dynamic range (HDR), microsecond-scale particle motion, and heavy smoke often occur together. These conditions drive saturation, motion blur, and unstable particle extraction in conventional imaging. We present a closed-loop Event--SVE measurement system that couples a spatially variant exposure (SVE) camera with a stereo pair of neuromorphic event cameras. The SVE branch produces HDR maps with an explicit smoke-aware fusion strategy. A multi-cue smoke-likelihood map is used to separate particle emission from smoke scattering, yielding calibrated intensity maps for downstream analysis. The resulting HDR maps also provide the absolute-intensity reference missing in event cameras. This reference is used to suppress smoke-driven event artifacts and to improve particle-state discrimination. Based on the cleaned event observations, a stereo event-based 3D pipeline estimates separation height and equivalent particle size through feature extraction and triangulation (maximum calibration error 0.56%). Experiments on boron-based propellants show multimodal equivalent-radius statistics. The system also captures fast separation transients that are difficult to observe with conventional sensors. Overall, the proposed framework provides a practical, calibration-consistent route to microsecond-resolved 3D combustion measurement under smoke-obscured HDR conditions.
>
---
#### [new 082] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多模态大语言模型在人脸验证中的性别和种族公平性问题，评估不同模型的准确性与偏差，旨在提升模型的公平性。**

- **链接: [https://arxiv.org/pdf/2603.25613](https://arxiv.org/pdf/2603.25613)**

> **作者:** Ünsal Öztürk; Hatef Otroshi Shahreza; Sébastien Marcel
>
> **备注:** Accepted in CVPR 2026 workshops
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently been explored as face verification systems that determine whether two face images are of the same person. Unlike dedicated face recognition systems, MLLMs approach this task through visual prompting and rely on general visual and reasoning abilities. However, the demographic fairness of these models remains largely unexplored. In this paper, we present a benchmarking study that evaluates nine open-source MLLMs from six model families, ranging from 2B to 8B parameters, on the IJB-C and RFW face verification protocols across four ethnicity groups and two gender groups. We measure verification accuracy with the Equal Error Rate and True Match Rate at multiple operating points per demographic group, and we quantify demographic disparity with four FMR-based fairness metrics. Our results show that FaceLLM-8B, the only face-specialised model in our study, substantially outperforms general-purpose MLLMs on both benchmarks. The bias patterns we observe differ from those commonly reported for traditional face recognition, with different groups being most affected depending on the benchmark and the model. We also note that the most accurate models are not necessarily the fairest and that models with poor overall accuracy can appear fair simply because they produce uniformly high error rates across all demographic groups.
>
---
#### [new 083] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在提升模型的组合性表达能力。针对现有方法依赖硬负样本导致性能下降的问题，提出基于概念中心的短文本对齐和跨模态注意力池化，提升组合性同时保持零样本能力。**

- **链接: [https://arxiv.org/pdf/2603.25722](https://arxiv.org/pdf/2603.25722)**

> **作者:** Hai X. Pham; David T. Hoffmann; Ricardo Guerrero; Brais Martinez
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Contrastive vision-language (V&L) models remain a popular choice for various applications. However, several limitations have emerged, most notably the limited ability of V&L models to learn compositional representations. Prior methods often addressed this limitation by generating custom training data to obtain hard negative samples. Hard negatives have been shown to improve performance on compositionality tasks, but are often specific to a single benchmark, do not generalize, and can cause substantial degradation of basic V&L capabilities such as zero-shot or retrieval performance, rendering them impractical. In this work we follow a different approach. We identify two root causes that limit compositionality performance of V&Ls: 1) Long training captions do not require a compositional representation; and 2) The final global pooling in the text and image encoders lead to a complete loss of the necessary information to learn binding in the first place. As a remedy, we propose two simple solutions: 1) We obtain short concept centric caption parts using standard NLP software and align those with the image; and 2) We introduce a parameter-free cross-modal attention-pooling to obtain concept centric visual embeddings from the image encoder. With these two changes and simple auxiliary contrastive losses, we obtain SOTA performance on standard compositionality benchmarks, while maintaining or improving strong zero-shot and retrieval capabilities. This is achieved without increasing inference cost. We release the code for this work at this https URL.
>
---
#### [new 084] Pixelis: Reasoning in Pixels, from Seeing to Acting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Pixelis，一种在像素空间中操作的智能体，解决视觉-语言系统缺乏行动能力的问题。通过执行图像操作并从结果中学习，提升视觉智能的物理 grounded 性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.25091](https://arxiv.org/pdf/2603.25091)**

> **作者:** Yunpeng Zhou
>
> **备注:** 28pages, 16figures, 18tables
>
> **摘要:** Most vision-language systems are static observers: they describe pixels, do not act, and cannot safely improve under shift. This passivity limits generalizable, physically grounded visual intelligence. Learning through action, not static description, is essential beyond curated data. We present Pixelis, a pixel-space agent that operates directly on images and videos via a compact set of executable operations (zoom/crop, segment, track, OCR, temporal localization) and learns from its consequences. Pixelis trains in three phases: (1) Supervised Fine-Tuning learns a pixel-tool grammar from Chain-of-Thought-Action traces with a masked imitation loss that upweights operation/argument tokens and auxiliary heads to stabilize pixel-grounded arguments; (2) Curiosity-Coherence Reward Fine-Tuning optimizes a dual-drive objective marrying prediction-error curiosity with adjacent-step coherence and a mild efficiency prior under a KL anchor, yielding short, valid, structured toolchains; (3) Pixel Test-Time RL performs label-free adaptation by retrieving neighbors, voting over complete trajectories rather than answers, and updating toward short, high-fidelity exemplars while constraining drift with a KL-to-EMA safety control. Across six public image and video benchmarks, Pixelis yields consistent improvements: the average relative gain is +4.08% over the same 8B baseline (peaking at +6.03% on VSI-Bench), computed as (ours-baseline)/baseline, while producing shorter, auditable toolchains and maintaining in-corridor KL during test-time learning. Acting within pixels, rather than abstract tokens, grounds multimodal perception in the physical world, linking visual reasoning with actionable outcomes, and enables embodied adaptation without external feedback.
>
---
#### [new 085] GoldiCLIP: The Goldilocks Approach for Balancing Explicit Supervision for Language-Image Pretraining
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉语言预训练任务，旨在解决数据量大、监督信号不足的问题。提出GoldiCLIP框架，通过平衡监督信号提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.24804](https://arxiv.org/pdf/2603.24804)**

> **作者:** Deen Dayal Mohan; Hossein Souri; Vitali Petsiuk; Juhong Min; Gopal Sharma; Luowei Zhou; Suren Kumar
>
> **摘要:** Until recently, the success of large-scale vision-language models (VLMs) has primarily relied on billion-sample datasets, posing a significant barrier to progress. Latest works have begun to close this gap by improving supervision quality, but each addresses only a subset of the weaknesses in contrastive pretraining. We present GoldiCLIP, a framework built on a Goldilocks principle of finding the right balance of supervision signals. Our multifaceted training framework synergistically combines three key innovations: (1) a text-conditioned self-distillation method to align both text-agnostic and text-conditioned features; (2) an encoder integrated decoder with Visual Question Answering (VQA) objective that enables the encoder to generalize beyond the caption-like queries; and (3) an uncertainty-based weighting mechanism that automatically balances all heterogeneous losses. Trained on just 30 million images, 300x less data than leading methods, GoldiCLIP achieves state-of-the-art among data-efficient approaches, improving over the best comparable baseline by 2.2 points on MSCOCO retrieval, 2.0 on fine-grained retrieval, and 5.9 on question-based retrieval, while remaining competitive with billion-scale models. Project page: this https URL.
>
---
#### [new 086] VideoTIR: Accurate Understanding for Long Videos with Efficient Tool-Integrated Reasoning
- **分类: cs.CV**

- **简介: 该论文针对长视频理解任务，解决MLLM在处理长视频时的幻觉问题。通过引入强化学习和工具调用优化，提升视频理解的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.25021](https://arxiv.org/pdf/2603.25021)**

> **作者:** Zhe Gao; Shiyu Shen; Taifeng Chai; Weinong Wang; Haotian Xu; Xing W; Wenbin Li; Qi Fan; Yang Gao; Dacheng Tao
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) often suffer from hallucinations in long video understanding (LVU), primarily due to the imbalance between textual and visual tokens. Observing that MLLMs handle short visual inputs well, recent LVU works alleviate hallucinations by automatically parsing the vast visual data into manageable segments that can be effectively processed by MLLMs. SFT-based tool-calling methods can serve this purpose, but they typically require vast amounts of fine-grained, high-quality data and suffer from constrained tool-calling trajectories. We propose a novel VideoTIR that leverages Reinforcement Learning (RL) to encourage proper usage of comprehensive multi-level toolkits for efficient long video understanding. VideoTIR explores both Zero-RL and SFT cold-starting to enable MLLMs to retrieve and focus on meaningful video segments/images/regions, enhancing long video understanding both accurately and efficiently. To reduce redundant tool-calling, we propose Toolkit Action Grouped Policy Optimization (TAGPO), which enhances the efficiency of the calling process through stepwise reward assignment and reuse of failed rollouts. Additionally, we develop a sandbox-based trajectory synthesis framework to generate high-quality trajectories data. Extensive experiments on three long-video QA benchmarks demonstrate the effectiveness and efficiency of our method.
>
---
#### [new 087] CHIRP dataset: towards long-term, individual-level, behavioral monitoring of bird populations in the wild
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于行为监测任务，旨在解决野生动物个体行为分析难题。构建了CHIRP数据集并提出CORVID方法，实现鸟类个体识别与行为分析。**

- **链接: [https://arxiv.org/pdf/2603.25524](https://arxiv.org/pdf/2603.25524)**

> **作者:** Alex Hoi Hang Chan; Neha Singhal; Onur Kocahan; Andrea Meltzer; Saverio Lubrano; Miyako H. Warrington; Michel Griesser; Fumihiro Kano; Hemal Naik
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Long-term behavioral monitoring of individual animals is crucial for studying behavioral changes that occur over different time scales, especially for conservation and evolutionary biology. Computer vision methods have proven to benefit biodiversity monitoring, but automated behavior monitoring in wild populations remains challenging. This stems from the lack of datasets that cover a range of computer vision tasks necessary to extract biologically meaningful measurements of individual animals. Here, we introduce such a dataset (CHIRP) with a new method (CORVID) for individual re-identification of wild birds. The CHIRP (Combining beHaviour, Individual Re-identification and Postures) dataset is curated from a long-term population of wild Siberian jays studied in Swedish Lapland, supporting re-identification (re-id), action recognition, 2D keypoint estimation, object detection, and instance segmentation. In addition to traditional task-specific benchmarking, we introduce application-specific benchmarking with biologically relevant metrics (feeding rates, co-occurrence rates) to evaluate the performance of models in real-world use cases. Finally, we present CORVID (COlouR-based Video re-ID), a novel pipeline for individual identification of birds based on the segmentation and classification of colored leg rings, a widespread approach for visual identification of individual birds. CORVID offers a probability-based id tracking method by matching the detected combination of color rings with a database. We use application-specific benchmarking to show that CORVID outperforms state-of-the-art re-id methods. We hope this work offers the community a blueprint for curating real-world datasets from ethically approved biological studies to bridge the gap between computer vision research and biological applications.
>
---
#### [new 088] ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects
- **分类: cs.CV**

- **简介: 该论文提出ICTPolarReal数据集，用于真实材料反射建模。解决逆渲染中真实反射数据不足的问题，通过多视角、多光照等采集方式，提升材质分离与光照真实性。**

- **链接: [https://arxiv.org/pdf/2603.24912](https://arxiv.org/pdf/2603.24912)**

> **作者:** Jing Yang; Krithika Dharanikota; Emily Jia; Haiwei Chen; Yajie Zhao
>
> **备注:** CVPR 2026
>
> **摘要:** Accurately modeling how real-world materials reflect light remains a core challenge in inverse rendering, largely due to the scarcity of real measured reflectance data. Existing approaches rely heavily on synthetic datasets with simplified illumination and limited material realism, preventing models from generalizing to real-world images. We introduce a large-scale polarized reflection and material dataset of real-world objects, captured with an 8-camera, 346-light Light Stage equipped with cross/parallel polarization. Our dataset spans 218 everyday objects across five acquisition dimensions-multiview, multi-illumination, polarization, reflectance separation, and material attributes-yielding over 1.2M high-resolution images with diffuse-specular separation and analytically derived diffuse albedo, specular albedo, and surface normals. Using this dataset, we train and evaluate state-of-the-art inverse and forward rendering models on intrinsic decomposition, relighting, and sparse-view 3D reconstruction, demonstrating significant improvements in material separation, illumination fidelity, and geometric consistency. We hope that our work can establish a new foundation for physically grounded material understanding and enable real-world generalization beyond synthetic training regimes. Project page: this https URL
>
---
#### [new 089] OpenCap Monocular: 3D Human Kinematics and Musculoskeletal Dynamics from a Single Smartphone Video
- **分类: cs.CV; eess.IV; q-bio.QM**

- **简介: 该论文提出OpenCap Monocular，用于从单个手机视频中估计3D人体运动学和力学，解决传统方法成本高、效率低的问题。**

- **链接: [https://arxiv.org/pdf/2603.24733](https://arxiv.org/pdf/2603.24733)**

> **作者:** Selim Gilon; Emily Y. Miller; Scott D. Uhlrich
>
> **摘要:** Quantifying human movement (kinematics) and musculoskeletal forces (kinetics) at scale, such as estimating quadriceps force during a sit-to-stand movement, could transform prediction, treatment, and monitoring of mobility-related conditions. However, quantifying kinematics and kinetics traditionally requires costly, time-intensive analysis in specialized laboratories, limiting clinical translation. Scalable, accurate tools for biomechanical assessment are needed. We introduce OpenCap Monocular, an algorithm that estimates 3D skeletal kinematics and kinetics from a single smartphone video. The method refines 3D human pose estimates from a monocular pose estimation model (WHAM) via optimization, computes kinematics of a biomechanically constrained skeletal model, and estimates kinetics via physics-based simulation and machine learning. We validated OpenCap Monocular against marker-based motion capture and force plate data for walking, squatting, and sit-to-stand tasks. OpenCap Monocular achieved low kinematic error (4.8° mean absolute error for rotational degrees of freedom; 3.4 cm for pelvis translations), outperforming a regression-only computer vision baseline by 48% in rotational accuracy (p = 0.036) and 69% in translational accuracy (p < 0.001). OpenCap Monocular also estimated ground reaction forces during walking with accuracy comparable to, or better than, our prior two-camera OpenCap system. We demonstrate that the algorithm estimates important kinetic outcomes with clinically meaningful accuracy in applications related to frailty and knee osteoarthritis, including estimating knee extension moment during sit-to-stand transitions and knee adduction moment during walking. OpenCap Monocular is deployed via a smartphone app, web app, and secure cloud computing (this https URL), enabling free, accessible single-smartphone biomechanical assessments.
>
---
#### [new 090] Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，解决动态物体隐藏后重现时的连贯性问题。提出Hybrid Memory和HyDRA架构，提升模型对动态主体的跟踪与记忆能力。**

- **链接: [https://arxiv.org/pdf/2603.25716](https://arxiv.org/pdf/2603.25716)**

> **作者:** Kaijin Chen; Dingkang Liang; Xin Zhou; Yikang Ding; Xiaoqiang Liu; Pengfei Wan; Xiang Bai
>
> **摘要:** Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily treat environments as static canvases. When dynamic subjects hide out of sight and later re-emerge, current methods often struggle, leading to frozen, distorted, or vanishing subjects. To address this, we introduce Hybrid Memory, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals. To facilitate research in this direction, we construct HM-World, the first large-scale video dataset dedicated to hybrid memory. It features 59K high-fidelity clips with decoupled camera and subject trajectories, encompassing 17 diverse scenes, 49 distinct subjects, and meticulously designed exit-entry events to rigorously evaluate hybrid coherence. Furthermore, we propose HyDRA, a specialized memory architecture that compresses memory into tokens and utilizes a spatiotemporal relevance-driven retrieval mechanism. By selectively attending to relevant motion cues, HyDRA effectively preserves the identity and motion of hidden subjects. Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality.
>
---
#### [new 091] BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation
- **分类: cs.CV**

- **简介: 该论文提出BizGenEval，用于评估商业视觉内容生成模型。针对现有基准不足，解决真实商业设计任务中的多约束评价问题，涵盖五类文档和四个能力维度，包含大量测试任务与人工验证问题。**

- **链接: [https://arxiv.org/pdf/2603.25732](https://arxiv.org/pdf/2603.25732)**

> **作者:** Yan Li; Zezi Zeng; Ziwei Zhou; Xin Gao; Muzhao Tian; Yifan Yang; Mingxi Cheng; Qi Dai; Yuqing Yang; Lili Qiu; Zhendong Wang; Zhengyuan Yang; Xue Yang; Lijuan Wang; Ji Li; Chong Luo
>
> **摘要:** Recent advances in image generation models have expanded their applications beyond aesthetic imagery toward practical visual content creation. However, existing benchmarks mainly focus on natural image synthesis and fail to systematically evaluate models under the structured and multi-constraint requirements of real-world commercial design tasks. In this work, we introduce BizGenEval, a systematic benchmark for commercial visual content generation. The benchmark spans five representative document types: slides, charts, webpages, posters, and scientific figures, and evaluates four key capability dimensions: text rendering, layout control, attribute binding, and knowledge-based reasoning, forming 20 diverse evaluation tasks. BizGenEval contains 400 carefully curated prompts and 8000 human-verified checklist questions to rigorously assess whether generated images satisfy complex visual and semantic constraints. We conduct large-scale benchmarking on 26 popular image generation systems, including state-of-the-art commercial APIs and leading open-source models. The results reveal substantial capability gaps between current generative models and the requirements of professional visual content creation. We hope BizGenEval serves as a standardized benchmark for real-world commercial visual content generation.
>
---
#### [new 092] Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决视频数据中视觉质量与运动质量的矛盾问题。通过时间步选择训练方法，提升模型性能，无需依赖完美数据。**

- **链接: [https://arxiv.org/pdf/2603.25527](https://arxiv.org/pdf/2603.25527)**

> **作者:** Xiangyang Luo; Qingyu Li; Yuming Li; Guanbo Huang; Yongjie Zhu; Wenyu Qin; Meng Wang; Pengfei Wan; Shao-Lun Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent advances in video generation models have achieved impressive results. However, these models heavily rely on the use of high-quality data that combines both high visual quality and high motion quality. In this paper, we identify a key challenge in video data curation: the Motion-Vision Quality Dilemma. We discovered that visual quality and motion intensity inherently exhibit a negative correlation, making it hard to obtain golden data that excels in both aspects. To address this challenge, we first examine the hierarchical learning dynamics of video diffusion models and conduct gradient-based analysis on quality-degraded samples. We discover that quality-imbalanced data can produce gradients similar to golden data at appropriate timesteps. Based on this, we introduce the novel concept of Timestep selection in Training Process. We propose Timestep-aware Quality Decoupling (TQD), which modifies the data sampling distribution to better match the model's learning process. For certain types of data, the sampling distribution is skewed toward higher timesteps for motion-rich data, while high visual quality data is more likely to be sampled during lower timesteps. Through extensive experiments, we demonstrate that TQD enables training exclusively on separated imbalanced data to achieve performance surpassing conventional training with better data, challenging the necessity of perfect data in video generation. Moreover, our method also boosts model performance when trained on high-quality data, showcasing its effectiveness across different data scenarios.
>
---
#### [new 093] Just Zoom In: Cross-View Geo-Localization via Autoregressive Zooming
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究跨视角地理定位任务，解决街景与卫星图匹配中的定位问题。提出通过自回归缩放实现粗到细的空间推理，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.25686](https://arxiv.org/pdf/2603.25686)**

> **作者:** Yunus Talha Erzurumlu; Jiyong Kwag; Alper Yilmaz
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Cross-view geo-localization (CVGL) estimates a camera's location by matching a street-view image to geo-referenced overhead imagery, enabling GPS-denied localization and navigation. Existing methods almost universally formulate CVGL as an image-retrieval problem in a contrastively trained embedding space. This ties performance to large batches and hard negative mining, and it ignores both the geometric structure of maps and the coverage mismatch between street-view and overhead imagery. In particular, salient landmarks visible from the street view can fall outside a fixed satellite crop, making retrieval targets ambiguous and limiting explicit spatial inference over the map. We propose Just Zoom In, an alternative formulation that performs CVGL via autoregressive zooming over a city-scale overhead map. Starting from a coarse satellite view, the model takes a short sequence of zoom-in decisions to select a terminal satellite cell at a target resolution, without contrastive losses or hard negative mining. We further introduce a realistic benchmark with crowd-sourced street views and high-resolution satellite imagery that reflects real capture conditions. On this benchmark, Just Zoom In achieves state-of-the-art performance, improving Recall@1 within 50 m by 5.5% and Recall@1 within 100 m by 9.6% over the strongest contrastive-retrieval baseline. These results demonstrate the effectiveness of sequential coarse-to-fine spatial reasoning for cross-view geo-localization.
>
---
#### [new 094] TRACE: Object Motion Editing in Videos with First-Frame Trajectory Guidance
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在实现对象运动路径的可控编辑。通过用户在首帧设计轨迹，生成连贯的编辑视频，解决传统方法依赖点跟踪或难以控制的问题。**

- **链接: [https://arxiv.org/pdf/2603.25707](https://arxiv.org/pdf/2603.25707)**

> **作者:** Quynh Phung; Long Mai; Cusuh Ham; Feng Liu; Jia-Bin Huang; Aniruddha Mahapatra
>
> **备注:** webpage: this https URL
>
> **摘要:** We study object motion path editing in videos, where the goal is to alter a target object's trajectory while preserving the original scene content. Unlike prior video editing methods that primarily manipulate appearance or rely on point-track-based trajectory control, which is often challenging for users to provide during inference, especially in videos with camera motion, we offer a practical, easy-to-use approach to controllable object-centric motion editing. We present Trace, a framework that enables users to design the desired trajectory in a single anchor frame and then synthesizes a temporally consistent edited video. Our approach addresses this task with a two-stage pipeline: a cross-view motion transformation module that maps first-frame path design to frame-aligned box trajectories under camera motion, and a motion-conditioned video re-synthesis module that follows these trajectories to regenerate the object while preserving the remaining content of the input video. Experiments on diverse real-world videos show that our method produces more coherent, realistic, and controllable motion edits than recent image-to-video and video-to-video methods.
>
---
#### [new 095] Learning domain-invariant features through channel-level sparsification for Out-Of Distribution Generalization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分析中的OOD泛化任务，旨在解决模型依赖非因果特征导致性能不一致的问题。提出HCD方法，通过通道级因果掩码实现特征稀疏性，提升模型的域不变性。**

- **链接: [https://arxiv.org/pdf/2603.25083](https://arxiv.org/pdf/2603.25083)**

> **作者:** Haoran Pei; Yuguang Yang; Kexin Liu; Juan Zhang; Baochang Zhang
>
> **摘要:** Out-of-Distribution (OOD) generalization has become a primary metric for evaluating image analysis systems. Since deep learning models tend to capture domain-specific context, they often develop shortcut dependencies on these non-causal features, leading to inconsistent performance across different data sources. Current techniques, such as invariance learning, attempt to mitigate this. However, they struggle to isolate highly mixed features within deep latent spaces. This limitation prevents them from fully resolving the shortcut learning this http URL this paper, we propose Hierarchical Causal Dropout (HCD), a method that uses channel-level causal masks to enforce feature sparsity. This approach allows the model to separate causal features from spurious ones, effectively performing a causal intervention at the representation level. The training is guided by a Matrix-based Mutual Information (MMI) objective to minimize the mutual information between latent features and domain labels, while simultaneously maximizing the information shared with class this http URL ensure stability, we incorporate a StyleMix-driven VICReg module, which prevents the masks from accidentally filtering out essential causal data. Experimental results on OOD benchmarks show that HCD performs better than existing top-tier methods.
>
---
#### [new 096] Towards Controllable Low-Light Image Enhancement: A Continuous Multi-illumination Dataset and Efficient State Space Framework
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决传统方法在多模态解空间中的亮度不一致问题。提出CLE-RWKV框架，结合新数据集和噪声解耦策略，实现更可控的增强效果。**

- **链接: [https://arxiv.org/pdf/2603.25296](https://arxiv.org/pdf/2603.25296)**

> **作者:** Hongru Han; Tingrui Guo; Liming Zhang; Yan Su; Qiwen Xu; Zhuohua Ye
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Low-light image enhancement (LLIE) has traditionally been formulated as a deterministic mapping. However, this paradigm often struggles to account for the ill-posed nature of the task, where unknown ambient conditions and sensor parameters create a multimodal solution space. Consequently, state-of-the-art methods frequently encounter luminance discrepancies between predictions and labels, often necessitating "gt-mean" post-processing to align output luminance for evaluation. To address this fundamental limitation, we propose a transition toward Controllable Low-light Enhancement (CLE), explicitly reformulating the task as a well-posed conditional problem. To this end, we introduce CLE-RWKV, a holistic framework supported by Light100, a new benchmark featuring continuous real-world illumination transitions. To resolve the conflict between luminance control and chromatic fidelity, a noise-decoupled supervision strategy in the HVI color space is employed, effectively separating illumination modulation from texture restoration. Architecturally, to adapt efficient State Space Models (SSMs) for dense prediction, we leverage a Space-to-Depth (S2D) strategy. By folding spatial neighborhoods into channel dimensions, this design allows the model to recover local inductive biases and effectively bridge the "scanning gap" inherent in flattened visual sequences without sacrificing linear complexity. Experiments across seven benchmarks demonstrate that our approach achieves competitive performance and robust controllability, providing a real-world multi-illumination alternative that significantly reduces the reliance on gt-mean post-processing.
>
---
#### [new 097] Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation
- **分类: cs.CV**

- **简介: 该论文属于神经网络解释任务，旨在解决神经元概念解释不准确的问题。通过提出Select-Hypothesize-Verify框架，提升神经元功能解释的准确性。**

- **链接: [https://arxiv.org/pdf/2603.24953](https://arxiv.org/pdf/2603.24953)**

> **作者:** ZeBin Ji; Yang Hu; Xiuli Bi; Bo Liu; Bin Xiao
>
> **备注:** Accepted in CVPR 2026
>
> **摘要:** It is essential for understanding neural network decisions to interpret the functionality (also known as concepts) of neurons. Existing approaches describe neuron concepts by generating natural language descriptions, thereby advancing the understanding of the neural network's decision-making mechanism. However, these approaches assume that each neuron has well-defined functions and provides discriminative features for neural network decision-making. In fact, some neurons may be redundant or may offer misleading concepts. Thus, the descriptions for such neurons may cause misinterpretations of the factors driving the neural network's decisions. To address the issue, we introduce a verification of neuron functions, which checks whether the generated concept highly activates the corresponding neuron. Furthermore, we propose a Select-Hypothesize-Verify framework for interpreting neuron functionality. This framework consists of: 1) selecting activation samples that best capture a neuron's well-defined functional behavior through activation-distribution analysis; 2) forming hypotheses about concepts for the selected neurons; and 3) verifying whether the generated concepts accurately reflect the functionality of the neuron. Extensive experiments show that our method produces more accurate neuron concepts. Our generated concepts activate the corresponding neurons with a probability approximately 1.5 times that of the current state-of-the-art method.
>
---
#### [new 098] TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval
- **分类: cs.CV**

- **简介: 该论文提出TIGeR框架，解决图像的时空检索任务，通过统一建模图像、位置和时间，提升地理时间相关检索效果。**

- **链接: [https://arxiv.org/pdf/2603.24749](https://arxiv.org/pdf/2603.24749)**

> **作者:** David G. Shatwell; Sirnam Swetha; Mubarak Shah
>
> **备注:** Accepted in CVPR 2026
>
> **摘要:** Many real-world applications in digital forensics, urban monitoring, and environmental analysis require jointly reasoning about visual appearance, geolocation, and time. Beyond standard geo-localization and time-of-capture prediction, these applications increasingly demand more complex capabilities, such as retrieving an image captured at the same location as a query image but at a specified target time. We formalize this problem as Geo-Time Aware Image Retrieval and curate a diverse benchmark of 4.5M paired image-location-time triplets for training and 86k high-quality triplets for evaluation. We then propose TIGeR, a multi-modal-transformer-based model that maps image, geolocation, and time into a unified geo-temporal embedding space. TIGeR supports flexible input configurations (single-modality and multi-modality queries) and uses the same representation to perform (i) geo-localization, (ii) time-of-capture prediction, and (iii) geo-time-aware retrieval. By better preserving underlying location identity under large appearance changes, TIGeR enables retrieval based on where and when a scene is, rather than purely on visual similarity. Extensive experiments show that TIGeR consistently outperforms strong baselines and state-of-the-art methods by up to 16% on time-of-year, 8% time-of-day prediction, and 14% in geo-time aware retrieval recall, highlighting the benefits of unified geo-temporal modeling.
>
---
#### [new 099] BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning
- **分类: cs.CV**

- **简介: 该论文提出BFMD数据集，解决羽毛球全场比赛密集标注与战术分析问题，包含19场完整比赛的多模态标注数据，并构建了基于VideoMAE的多模态标题生成框架。**

- **链接: [https://arxiv.org/pdf/2603.25533](https://arxiv.org/pdf/2603.25533)**

> **作者:** Ning Ding; Keisuke Fujii; Toru Tamaki
>
> **备注:** CVSports2026 accepted
>
> **摘要:** Understanding tactical dynamics in badminton requires analyzing entire matches rather than isolated clips. However, existing badminton datasets mainly focus on short clips or task-specific annotations and rarely provide full-match data with dense multimodal annotations. This limitation makes it difficult to generate accurate shot captions and perform match-level analysis. To address this limitation, we introduce the first Badminton Full Match Dense (BFMD) dataset, with 19 broadcast matches (including both singles and doubles) covering over 20 hours of play, comprising 1,687 rallies and 16,751 hit events, each annotated with a shot caption. The dataset provides hierarchical annotations including match segments, rally events, and dense rally-level multimodal annotations such as shot types, shuttle trajectories, player pose keypoints, and shot captions. We develop a VideoMAE-based multimodal captioning framework with a Semantic Feedback mechanism that leverages shot semantics to guide caption generation and improve semantic consistency. Experimental results demonstrate that multimodal modeling and semantic feedback improve shot caption quality over RGB-only baselines. We further showcase the potential of BFMD by analyzing the temporal evolution of tactical patterns across full matches.
>
---
#### [new 100] An Image Dataset of Common Skin Diseases of Bangladesh and Benchmarking Performance with Machine Learning Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于皮肤疾病分类任务，旨在解决缺乏专业诊断资源的问题。构建了包含五种常见皮肤病的公开数据集，并应用机器学习模型进行分类评估。**

- **链接: [https://arxiv.org/pdf/2603.25229](https://arxiv.org/pdf/2603.25229)**

> **作者:** Sazzad Hossain; Saiful Islam; Muhammad Ibrahim; Md. Rasel Ahmed; Md Shuayb; Ahmedul Kabir
>
> **备注:** 14 pages
>
> **摘要:** Skin diseases are a major public health concern worldwide, and their detection is often challenging without access to dermatological expertise. In countries like Bangladesh, which is highly populated, the number of qualified skin specialists and diagnostic instruments is insufficient to meet the demand. Due to the lack of proper detection and treatment of skin diseases, that may lead to severe health consequences including death. Common properties of skin diseases are, changing the color, texture, and pattern of skin and in this era of artificial intelligence and machine learning, we are able to detect skin diseases by using image processing and computer vision techniques. In response to this challenge, we develop a publicly available dataset focused on common skin disease detection using machine learning techniques. We focus on five prevalent skin diseases in Bangladesh: Contact Dermatitis, Vitiligo, Eczema, Scabies, and Tinea Ringworm. The dataset consists of 1612 images (of which, 250 are distinct while others are augmented), collected directly from patients at the outpatient department of Faridpur Medical College, Faridpur, Bangladesh. The data comprises of 302, 381, 301, 316, and 312 images of Dermatitis, Eczema, Scabies, Tinea Ringworm, and Vitiligo, respectively. Although the data are collected regionally, the selected diseases are common across many countries especially in South Asia, making the dataset potentially valuable for global applications in machine learning-based dermatology. We also apply several machine learning and deep learning models on the dataset and report classification performance. We expect that this research would garner attention from machine learning and deep learning researchers and practitioners working in the field of automated disease diagnosis.
>
---
#### [new 101] MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes
- **分类: cs.CV**

- **简介: 该论文属于动态3D场景在线重建任务，解决现有方法无法准确学习每个高斯的运动问题。通过引入运动线索和置信度，提升4D重建质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.25042](https://arxiv.org/pdf/2603.25042)**

> **作者:** Wonjoon Lee; Sungmin Woo; Donghyeong Kim; Jungho Lee; Sangheon Park; Sangyoun Lee
>
> **摘要:** Online reconstruction of dynamic scenes aims to learn from streaming multi-view inputs under low-latency constraints. The fast training and real-time rendering capabilities of 3D Gaussian Splatting have made on-the-fly reconstruction practically feasible, enabling online 4D reconstruction. However, existing online approaches, despite their efficiency and visual quality, fail to learn per-Gaussian motion that reflects true scene dynamics. Without explicit motion cues, appearance and motion are optimized solely under photometric loss, causing per-Gaussian motion to chase pixel residuals rather than true 3D motion. To address this, we propose MoRGS, an efficient online per-Gaussian motion reasoning framework that explicitly models per-Gaussian motion to improve 4D reconstruction quality. Specifically, we leverage optical flow on a sparse set of key views as lightweight motion cues that regularize per-Gaussian motion beyond photometric supervision. To compensate for the sparsity of flow supervision, we learn a per-Gaussian motion offset field that reconciles discrepancies between projected 3D motion and observed flow across views and time. In addition, we introduce a per-Gaussian motion confidence that separates dynamic from static Gaussians and weights Gaussian attribute residual updates, thereby suppressing redundant motion in static regions for better temporal consistency and accelerating the modeling of large motions. Extensive experiments demonstrate that MoRGS achieves state-of-the-art reconstruction quality and motion fidelity among online methods, while maintaining streamable performance.
>
---
#### [new 102] Hyperspectral Trajectory Image for Multi-Month Trajectory Anomaly Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于轨迹异常检测任务，解决多月密集GPS轨迹分析难题。提出TITAnD模型，将轨迹转化为高光谱图像，结合视觉方法实现高效异常检测。**

- **链接: [https://arxiv.org/pdf/2603.25255](https://arxiv.org/pdf/2603.25255)**

> **作者:** Md Awsafur Rahman; Chandrakanth Gudavalli; Hardik Prajapati; B. S. Manjunath
>
> **摘要:** Trajectory anomaly detection underpins applications from fraud detection to urban mobility analysis. Dense GPS methods preserve fine-grained evidence such as abnormal speeds and short-duration events, but their quadratic cost makes multi-month analysis intractable; consequently, no existing approach detects anomalies over multi-month dense GPS trajectories. The field instead relies on scalable sparse stay-point methods that discard this evidence, forcing separate architectures for each regime and preventing knowledge transfer. We argue this bottleneck is unnecessary: human trajectories, dense or sparse, share a natural two-dimensional cyclic structure along within-day and across-day axes. We therefore propose TITAnD (Trajectory Image Transformer for Anomaly Detection), which reformulates trajectory anomaly detection as a vision problem by representing trajectories as a Hyperspectral Trajectory Image (HTI): a day x time-of-day grid whose channels encode spatial, semantic, temporal, and kinematic information from either modality, unifying both under a single representation. Under this formulation, agent-level detection reduces to image classification and temporal localization to semantic segmentation. To model this representation, we introduce the Cyclic Factorized Transformer (CFT), which factorizes attention along the two temporal axes, encoding the cyclic inductive bias of human routines, while reducing attention cost by orders of magnitude and enabling dense multi-month anomaly detection for the first time. Empirically, TITAnD achieves the best AUC-PR across sparse and dense benchmarks, surpassing vision models like UNet while being 11-75x faster than the Transformer with comparable memory, demonstrating that vision reformulation and structure-aware modeling are jointly essential. Code will be made public soon.
>
---
#### [new 103] Self-Corrected Image Generation with Explainable Latent Rewards
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂提示对齐问题。通过引入xLARD框架，利用可解释的潜在奖励进行自我修正，提升语义一致性和图像质量。**

- **链接: [https://arxiv.org/pdf/2603.24965](https://arxiv.org/pdf/2603.24965)**

> **作者:** Yinyi Luo; Hrishikesh Gokhale; Marios Savvides; Jindong Wang; Shengfeng He
>
> **备注:** CVPR 2026
>
> **摘要:** Despite significant progress in text-to-image generation, aligning outputs with complex prompts remains challenging, particularly for fine-grained semantics and spatial relations. This difficulty stems from the feed-forward nature of generation, which requires anticipating alignment without fully understanding the output. In contrast, evaluating generated images is more tractable. Motivated by this asymmetry, we propose xLARD, a self-correcting framework that uses multimodal large language models to guide generation through Explainable LAtent RewarDs. xLARD introduces a lightweight corrector that refines latent representations based on structured feedback from model-generated references. A key component is a differentiable mapping from latent edits to interpretable reward signals, enabling continuous latent-level guidance from non-differentiable image-level evaluations. This mechanism allows the model to understand, assess, and correct itself during generation. Experiments across diverse generation and editing tasks show that xLARD improves semantic alignment and visual fidelity while maintaining generative priors. Code is available at this https URL.
>
---
#### [new 104] Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D重建任务，旨在提升稀疏视角下的重建质量与速度。通过结合TensorRF与FreeNeRF，提出Few TensoRF方法，优化了渲染效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.25008](https://arxiv.org/pdf/2603.25008)**

> **作者:** Thanh-Hai Le; Hoang-Hau Tran; Trong-Nghia Vu
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** This paper presents Few TensoRF, a 3D reconstruction framework that combines TensorRF's efficient tensor based representation with FreeNeRF's frequency driven few shot regularization. Using TensorRF to significantly accelerate rendering speed and introducing frequency and occlusion masks, the method improves stability and reconstruction quality under sparse input views. Experiments on the Synthesis NeRF benchmark show that Few TensoRF method improves the average PSNR from 21.45 dB (TensorRF) to 23.70 dB, with the fine tuned version reaching 24.52 dB, while maintaining TensorRF's fast \(\approx10-15\) minute training time. Experiments on the THuman 2.0 dataset further demonstrate competitive performance in human body reconstruction, achieving 27.37 - 34.00 dB with only eight input images. These results highlight Few TensoRF as an efficient and data effective solution for real-time 3D reconstruction across diverse scenes.
>
---
#### [new 105] ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis
- **分类: cs.CV**

- **简介: 该论文属于新视角合成任务，解决单次前馈网络在多视角一致性上的不足。通过引入视图自适应的动态高斯点云渲染机制，提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.25265](https://arxiv.org/pdf/2603.25265)**

> **作者:** Moonyeon Jeong; Seunggi Min; Suhyeon Lee; Hongje Seong
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** We present ViewSplat, a view-adaptive 3D Gaussian splatting network for novel view synthesis from unposed images. While recent feed-forward 3D Gaussian splatting has significantly accelerated 3D scene reconstruction by bypassing per-scene optimization, a fundamental fidelity gap remains. We attribute this bottleneck to the limited capacity of single-step feed-forward networks to regress static Gaussian primitives that satisfy all viewpoints. To address this limitation, we shift the paradigm from static primitive regression to view-adaptive dynamic splatting. Instead of a rigid Gaussian representation, our pipeline learns a view-adaptable latent representation. Specifically, ViewSplat initially predicts base Gaussian primitives alongside the weights of dynamic MLPs. During rendering, these MLPs take target view coordinates as input and predict view-dependent residual updates for each Gaussian attribute (i.e., 3D position, scale, rotation, opacity, and color). This mechanism, which we term view-adaptive dynamic splatting, allows each primitive to rectify initial estimation errors, effectively capturing high-fidelity appearances. Extensive experiments demonstrate that ViewSplat achieves state-of-the-art fidelity while maintaining fast inference (17 FPS) and real-time rendering (154 FPS).
>
---
#### [new 106] Knowledge-Guided Failure Prediction: Detecting When Object Detectors Miss Safety-Critical Objects
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，解决安全关键物体被遗漏的问题。提出KGFP方法，通过语义对齐检测功能故障，提升检测可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25499](https://arxiv.org/pdf/2603.25499)**

> **作者:** Jakob Paul Zimmermann; Gerrit Holzbach; David Lerch
>
> **摘要:** Object detectors deployed in safety-critical environments can fail silently, e.g. missing pedestrians, workers, or other safety-critical objects without emitting any warning. Traditional Out Of Distribution (OOD) detection methods focus on identifying unfamiliar inputs, but do not directly predict functional failures of the detector itself. We introduce Knowledge Guided Failure Prediction (KGFP), a representation-based monitoring framework that treats missed safety-critical detections as anomalies to be detected at runtime. KGFP measures semantic misalignment between internal object detector features and visual foundation model embeddings using a dual-encoder architecture with an angular distance metric. A key property is that when either the detector is operating outside its competence or the visual foundation model itself encounters novel inputs, the two embeddings diverge, producing a high-angle signal that reliably flags unsafe images. We compare our novel KGFS method to baseline OOD detection methods. On COCO person detection, applying KGFP as a selective-prediction gate raises person recall among accepted images from 64.3% to 84.5% at 5% False Positive Rate (FPR), and maintains strong performance across six COCO-O visual domains, outperforming OOD baselines by large margins. Our code, models, and features are published at this https URL.
>
---
#### [new 107] DCARL: A Divide-and-Conquer Framework for Autoregressive Long-Trajectory Video Generation
- **分类: cs.CV**

- **简介: 该论文提出DCARL框架，解决长轨迹视频生成中的视觉漂移和可控性差问题，结合分治策略与扩散模型，提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.24835](https://arxiv.org/pdf/2603.24835)**

> **作者:** Junyi Ouyang; Wenbin Teng; Gonglin Chen; Yajie Zhao; Haiwei Chen
>
> **备注:** 29 pages, 11 figures. Project page: this https URL
>
> **摘要:** Long-trajectory video generation is a crucial yet challenging task for world modeling primarily due to the limited scalability of existing video diffusion models (VDMs). Autoregressive models, while offering infinite rollout, suffer from visual drift and poor controllability. To address these issues, we propose DCARL, a novel divide-and-conquer, autoregressive framework that effectively combines the structural stability of the divide-and-conquer scheme with the high-fidelity generation of VDMs. Our approach first employs a dedicated Keyframe Generator trained without temporal compression to establish long-range, globally consistent structural anchors. Subsequently, an Interpolation Generator synthesizes the dense frames in an autoregressive manner with overlapping segments, utilizing the keyframes for global context and a single clean preceding frame for local coherence. Trained on a large-scale internet long trajectory video dataset, our method achieves superior performance in both visual quality (lower FID and FVD) and camera adherence (lower ATE and ARE) compared to state-of-the-art autoregressive and divide-and-conquer baselines, demonstrating stable and high-fidelity generation for long trajectory videos up to 32 seconds in length.
>
---
#### [new 108] Denoise and Align: Towards Source-Free UDA for Robust Panoramic Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于源域自适应任务，解决无监督域适应中源数据不可用的问题。提出DAPASS框架，通过去噪和对齐提升全景语义分割性能。**

- **链接: [https://arxiv.org/pdf/2603.25131](https://arxiv.org/pdf/2603.25131)**

> **作者:** Yaowen Chang; Zhen Cao; Xu Zheng; Xiaoxin Mi; Zhen Dong
>
> **备注:** Accepted to CVPR26
>
> **摘要:** Panoramic semantic segmentation is pivotal for comprehensive 360° scene understanding in critical applications like autonomous driving and virtual reality. However, progress in this domain is constrained by two key challenges: the severe geometric distortions inherent in panoramic projections and the prohibitive cost of dense annotation. While Unsupervised Domain Adaptation (UDA) from label-rich pinhole-camera datasets offers a viable alternative, many real-world tasks impose a stricter source-free (SFUDA) constraint where source data is inaccessible for privacy or proprietary reasons. This constraint significantly amplifies the core problems of domain shift, leading to unreliable pseudo-labels and dramatic performance degradation, particularly for minority classes. To overcome these limitations, we propose the DAPASS framework. DAPASS introduces two synergistic modules to robustly transfer knowledge without source data. First, our Panoramic Confidence-Guided Denoising (PCGD) module generates high-fidelity, class-balanced pseudo-labels by enforcing perturbation consistency and incorporating neighborhood-level confidence to filter noise. Second, a Contextual Resolution Adversarial Module (CRAM) explicitly addresses scale variance and distortion by adversarially aligning fine-grained details from high-resolution crops with global semantics from low-resolution contexts. DAPASS achieves state-of-the-art performances on outdoor (Cityscapes-to-DensePASS) and indoor (Stanford2D3D) benchmarks, yielding 55.04% (+2.05%) and 70.38% (+1.54%) mIoU, respectively.
>
---
#### [new 109] CORA: A Pathology Synthesis Driven Foundation Model for Coronary CT Angiography Analysis and MACE Risk Assessment
- **分类: cs.CV**

- **简介: 该论文提出CORAL模型，用于冠状动脉CT血管造影分析和MACE风险评估。针对标注数据稀缺和病理特征捕捉不足的问题，采用病理驱动的自监督学习方法，提升诊断与风险预测性能。**

- **链接: [https://arxiv.org/pdf/2603.24847](https://arxiv.org/pdf/2603.24847)**

> **作者:** Jinkui Hao; Gorkem Durak; Halil Ertugrul Aktas; Ulas Bagci; Bradley D. Allen; Nilay S. Shah; Bo Zhou
>
> **摘要:** Coronary artery disease, the leading cause of cardiovascular mortality worldwide, can be assessed non-invasively by coronary computed tomography angiography (CCTA). Despite progress in automated CCTA analysis using deep learning, clinical translation is constrained by the scarcity of expert-annotated datasets. Furthermore, widely adopted label-free pretraining strategies, such as masked image modeling, are intrinsically biased toward global anatomical statistics, frequently failing to capture the spatially localized pathological features of coronary plaques. Here, we introduce CORA, a 3D vision foundation model for comprehensive cardiovascular risk assessment. CORA learns directly from volumetric CCTA via a pathology-centric, synthesis-driven self-supervised framework. By utilizing an anatomy-guided lesion synthesis engine, the model is explicitly trained to detect simulated vascular abnormalities, biasing representation learning toward clinically relevant disease features rather than dominant background anatomy. We trained CORA on a large-scale cohort of 12,801 unlabeled CCTA volumes and comprehensively evaluated the model across multi-center datasets from nine independent hospitals. Across diagnostic and anatomical tasks, including plaque characterization, stenosis detection, and coronary artery segmentation, CORA consistently outperformed the state-of-the-art 3D vision foundation models, achieving up to a 29\% performance gain. Crucially, by coupling the imaging encoder with a large language model, we extended CORA into a multimodal framework that significantly improved 30-day major adverse cardiac event (MACE) risk stratification. Our results establish CORA as a scalable and extensible foundation for unified anatomical assessment and cardiovascular risk prediction.
>
---
#### [new 110] AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于三维视觉任务，旨在解决无姿态新颖视图合成（NVS）的挑战。提出AirSplat框架，通过自洽对齐和评分匹配提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.25129](https://arxiv.org/pdf/2603.25129)**

> **作者:** Minh-Quan Viet Bui; Jaeho Moon; Munchurl Kim
>
> **备注:** Project page: this https URL
>
> **摘要:** While 3D Vision Foundation Models (3DVFMs) have demonstrated remarkable zero-shot capabilities in visual geometry estimation, their direct application to generalizable novel view synthesis (NVS) remains challenging. In this paper, we propose AirSplat, a novel training framework that effectively adapts the robust geometric priors of 3DVFMs into high-fidelity, pose-free NVS. Our approach introduces two key technical contributions: (1) Self-Consistent Pose Alignment (SCPA), a training-time feedback loop that ensures pixel-aligned supervision to resolve pose-geometry discrepancy; and (2) Rating-based Opacity Matching (ROM), which leverages the local 3D geometry consistency knowledge from a sparse-view NVS teacher model to filter out degraded primitives. Experimental results on large-scale benchmarks demonstrate that our method significantly outperforms state-of-the-art pose-free NVS approaches in reconstruction quality. Our AirSplat highlights the potential of adapting 3DVFMs to enable simultaneous visual geometry estimation and high-quality view synthesis.
>
---
#### [new 111] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition
- **分类: cs.CV**

- **简介: 该论文属于模型可解释性任务，旨在解决视觉-语言模型内部机制难以理解的问题。通过SITH框架，直接分析CLIP的权重，实现数据无关的语义解释与模型编辑。**

- **链接: [https://arxiv.org/pdf/2603.24653](https://arxiv.org/pdf/2603.24653)**

> **作者:** Francesco Gentile; Nicola Dall'Asen; Francesco Tonini; Massimiliano Mancini; Lorenzo Vaquero; Elisa Ricci
>
> **备注:** Accepted @ CVPR 2026. Project page: this https URL
>
> **摘要:** As vision-language models are deployed at scale, understanding their internal mechanisms becomes increasingly critical. Existing interpretability methods predominantly rely on activations, making them dataset-dependent, vulnerable to data bias, and often restricted to coarse head-level explanations. We introduce SITH (Semantic Inspection of Transformer Heads), a fully data-free, training-free framework that directly analyzes CLIP's vision transformer in weight space. For each attention head, we decompose its value-output matrix into singular vectors and interpret each one via COMP (Coherent Orthogonal Matching Pursuit), a new algorithm that explains them as sparse, semantically coherent combinations of human-interpretable concepts. We show that SITH yields coherent, faithful intra-head explanations, validated through reconstruction fidelity and interpretability experiments. This allows us to use SITH for precise, interpretable weight-space model edits that amplify or suppress specific concepts, improving downstream performance without retraining. Furthermore, we use SITH to study model adaptation, showing how fine-tuning primarily reweights a stable semantic basis rather than learning entirely new features.
>
---
#### [new 112] Self-Supervised Learning for Knee Osteoarthritis: Diagnostic Limitations and Prognostic Value of Uncurated Hospital Data
- **分类: cs.CV**

- **简介: 该论文研究自监督学习在膝骨关节炎诊断与预后的应用。任务为医学影像分析，解决数据偏差影响模型效果的问题。通过对比不同预训练方法，发现未校准数据适合预后预测但不适合诊断。**

- **链接: [https://arxiv.org/pdf/2603.24903](https://arxiv.org/pdf/2603.24903)**

> **作者:** Haresh Rengaraj Rajamohan; Yuxuan Chen; Kyunghyun Cho; Cem M. Deniz
>
> **摘要:** This study assesses whether self-supervised learning (SSL) improves knee osteoarthritis (OA) modeling for diagnosis and prognosis relative to ImageNet-pretrained initialization. We compared (i) image-only SSL pretrained on knee radiographs from the OAI, MOST, and NYU cohorts, and (ii) multimodal image-text SSL pretrained on uncurated hospital knee radiographs paired with radiologist impressions. For diagnostic Kellgren-Lawrence (KL) grade prediction, SSL offered mixed results. While image-only SSL improved accuracy during linear probing (frozen encoder), it did not outperform ImageNet pretraining during full fine-tuning. Similarly, multimodal SSL failed to improve grading performance. We attribute this to severe bias in the uncurated hospital pretraining corpus (93% estimated KL grade 3), which limited alignment with the balanced diagnostic task. In contrast, this same multimodal initialization significantly improved prognostic modeling. It outperformed ImageNet baselines in predicting 4-year structural incidence and progression, including on external validation (MOST AUROC: 0.701 vs. 0.599 at 10% labeled data). Overall, while uncurated hospital image-text data may be ineffective for learning diagnosis due to severity bias, it provides a strong signal for prognostic modeling when the downstream task aligns with pretraining data distribution
>
---
#### [new 113] GridVAD: Open-Set Video Anomaly Detection via Spatial Reasoning over Stratified Frame Grids
- **分类: cs.CV**

- **简介: 该论文提出GridVAD，用于视频异常检测。解决开放集异常检测中VLM误检和漏检问题，通过空间推理生成异常掩码，无需训练。**

- **链接: [https://arxiv.org/pdf/2603.25467](https://arxiv.org/pdf/2603.25467)**

> **作者:** Mohamed Eltahir; Ahmed O. Ibrahim; Obada Siralkhatim; Tabarak Abdallah; Sondos Mohamed
>
> **摘要:** Vision-Language Models (VLMs) are powerful open-set reasoners, yet their direct use as anomaly detectors in video surveillance is fragile: without calibrated anomaly priors, they alternate between missed detections and hallucinated false alarms. We argue the problem is not the VLM itself but how it is used. VLMs should function as anomaly proposers, generating open-set candidate descriptions that are then grounded and tracked by purpose-built spatial and temporal modules. We instantiate this propose-ground-propagate principle in GridVAD, a training-free pipeline that produces pixel-level anomaly masks without any domain-specific training. A VLM reasons over stratified grid representations of video clips to generate natural-language anomaly proposals. Self-Consistency Consolidation (SCC) filters hallucinations by retaining only proposals that recur across multiple independent samplings. Grounding DINO anchors each surviving proposal to a bounding box, and SAM2 propagates it as a dense mask through the anomaly interval. The per-clip VLM budget is fixed at M+1 calls regardless of video length, where M can be set according to the proposals needed. On UCSD Ped2, GridVAD achieves the highest Pixel-AUROC (77.59) among all compared methods, surpassing even the partially fine-tuned TAO (75.11) and outperforms other zero-shot approaches on object-level RBDC by over 5x. Ablations reveal that SCC provides a controllable precision-recall tradeoff: filtering improves all pixel level metrics at a modest cost in object-level recall. Efficiency experiments show GridVAD is 2.7x more call-efficient than uniform per-frame VLM querying while additionally producing dense segmentation this http URL and qualitative video results are available at this https URL.
>
---
#### [new 114] WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在提高匹配效率与精度。提出WAFT-Stereo方法，无需成本体积，通过变形实现高效准确的立体匹配。**

- **链接: [https://arxiv.org/pdf/2603.24836](https://arxiv.org/pdf/2603.24836)**

> **作者:** Yihan Wang; Jia Deng
>
> **摘要:** We introduce WAFT-Stereo, a simple and effective warping-based method for stereo matching. WAFT-Stereo demonstrates that cost volumes, a common design used in many leading methods, are not necessary for strong performance and can be replaced by warping with improved efficiency. WAFT-Stereo ranks first on ETH3D, KITTI and Middlebury public benchmarks, reducing the zero-shot error by 81% on ETH3D benchmark, while being 1.8-6.7x faster than competitive methods. Code and model weights are available at this https URL.
>
---
#### [new 115] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态奖励建模任务，旨在解决RLVR训练依赖大量标注数据的问题。提出MSRL方法，通过多阶段强化学习和跨模态知识蒸馏，提升模型性能并减少数据依赖。**

- **链接: [https://arxiv.org/pdf/2603.25108](https://arxiv.org/pdf/2603.25108)**

> **作者:** Chenglong Wang; Yifu Huo; Yang Gan; Qiaozhi He; Qi Meng; Bei Li; Yan Wang; Junfu Liu; Tianhua Zhou; Jingbo Zhu; Tong Xiao
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Recent advances in multimodal reward modeling have been largely driven by a paradigm shift from discriminative to generative approaches. Building on this progress, recent studies have further employed reinforcement learning from verifiable rewards (RLVR) to enhance multimodal reward models (MRMs). Despite their success, RLVR-based training typically relies on labeled multimodal preference data, which are costly and labor-intensive to obtain, making it difficult to scale MRM training. To overcome this limitation, we propose a Multi-Stage Reinforcement Learning (MSRL) approach, which can achieve scalable RL for MRMs with limited multimodal data. MSRL replaces the conventional RLVR-based training paradigm by first learning a generalizable reward reasoning capability from large-scale textual preference data, and then progressively transferring this capability to multimodal tasks through caption-based and fully multimodal reinforcement-learning stages. Furthermore, we introduce a cross-modal knowledge distillation approach to improve preference generalization within MSRL. Extensive experiments demonstrate that MSRL effectively scales the RLVR-based training of generative MRMs and substantially improves their performance across both visual understanding and visual generation tasks (e.g., from 66.6% to 75.9% on VL-RewardBench and from 70.2% to 75.7% on GenAI-Bench), without requiring additional multimodal preference annotations. Our code is available at: this https URL.
>
---
#### [new 116] Towards Practical Lossless Neural Compression for LiDAR Point Clouds
- **分类: cs.CV**

- **简介: 该论文属于点云压缩任务，旨在解决LiDAR点云稀疏性导致的压缩效率低问题。提出两个轻量模块，提升压缩速度与性能。**

- **链接: [https://arxiv.org/pdf/2603.25260](https://arxiv.org/pdf/2603.25260)**

> **作者:** Pengpeng Yu; Haoran Li; Runqing Jiang; Dingquan Li; Jing Wang; Liang Lin; Yulan Guo
>
> **摘要:** LiDAR point clouds are fundamental to various applications, yet the extreme sparsity of high-precision geometric details hinders efficient context modeling, thereby limiting the compression speed and performance of existing methods. To address this challenge, we propose a compact representation for efficient predictive lossless coding. Our framework comprises two lightweight modules. First, the Geometry Re-Densification Module iteratively densifies encoded sparse geometry, extracts features at a dense scale, and then sparsifies the features for predictive coding. This module avoids costly computation on highly sparse details while maintaining a lightweight prediction head. Second, the Cross-scale Feature Propagation Module leverages occupancy cues from multiple resolution levels to guide hierarchical feature propagation, enabling information sharing across scales and reducing redundant feature extraction. Additionally, we introduce an integer-only inference pipeline to enable bit-exact cross-platform consistency, which avoids the entropy-coding collapse observed in existing neural compression methods and further accelerates coding. Experiments demonstrate competitive compression performance at real-time speed. Code will be released upon acceptance. Code is available at this https URL.
>
---
#### [new 117] VideoWeaver: Multimodal Multi-View Video-to-Video Transfer for Embodied Agents
- **分类: cs.CV**

- **简介: 该论文提出VideoWeaver，解决多视角视频到视频的迁移问题，通过共享4D潜空间实现视角一致性，支持多相机同步生成。**

- **链接: [https://arxiv.org/pdf/2603.25420](https://arxiv.org/pdf/2603.25420)**

> **作者:** George Eskandar; Fengyi Shen; Mohammad Altillawi; Dong Chen; Yang Bai; Liudi Yang; Ziyuan Liu
>
> **摘要:** Recent progress in video-to-video (V2V) translation has enabled realistic resimulation of embodied AI demonstrations, a capability that allows pretrained robot policies to be transferable to new environments without additional data collection. However, prior works can only operate on a single view at a time, while embodied AI tasks are commonly captured from multiple synchronized cameras to support policy learning. Naively applying single-view models independently to each camera leads to inconsistent appearance across views, and standard transformer architectures do not scale to multi-view settings due to the quadratic cost of cross-view attention. We present VideoWeaver, the first multimodal multi-view V2V translation framework. VideoWeaver is initially trained as a single-view flow-based V2V model. To achieve an extension to the multi-view regime, we propose to ground all views in a shared 4D latent space derived from a feed-forward spatial foundation model, namely, Pi3. This encourages view-consistent appearance even under wide baselines and dynamic camera motion. To scale beyond a fixed number of cameras, we train views at distinct diffusion timesteps, enabling the model to learn both joint and conditional view distributions. This in turn allows autoregressive synthesis of new viewpoints conditioned on existing ones. Experiments show superior or similar performance to the state-of-the-art on the single-view translation benchmarks and, for the first time, physically and stylistically consistent multi-view translations, including challenging egocentric and heterogeneous-camera setups central to world randomization for robot learning.
>
---
#### [new 118] PAWS: Perception of Articulation in the Wild at Scale from Egocentric Videos
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决人工标注数据不足导致的物体关节运动感知难题。提出PAWS方法，从大规模第一视角视频中自动提取物体关节信息，提升模型泛化能力与应用效果。**

- **链接: [https://arxiv.org/pdf/2603.25539](https://arxiv.org/pdf/2603.25539)**

> **作者:** Yihao Wang; Yang Miao; Wenshuai Zhao; Wenyan Yang; Zihan Wang; Joni Pajarinen; Luc Van Gool; Danda Pani Paudel; Juho Kannala; Xi Wang; Arno Solin
>
> **备注:** 32 pages, 13 figures. Project page: this https URL
>
> **摘要:** Articulation perception aims to recover the motion and structure of articulated objects (e.g., drawers and cupboards), and is fundamental to 3D scene understanding in robotics, simulation, and animation. Existing learning-based methods rely heavily on supervised training with high-quality 3D data and manual annotations, limiting scalability and diversity. To address this limitation, we propose PAWS, a method that directly extracts object articulations from hand-object interactions in large-scale in-the-wild egocentric videos. We evaluate our method on the public data sets, including HD-EPIC and Arti4D data sets, achieving significant improvements over baselines. We further demonstrate that the extracted articulations benefit downstream tasks, including fine-tuning 3D articulation prediction models and enabling robot manipulation. See the project website at this https URL.
>
---
#### [new 119] Towards Comprehensive Real-Time Scene Understanding in Ophthalmic Surgery through Multimodal Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在提升眼科手术中实时场景理解。通过融合显微镜与光学相干断层扫描图像，实现精准器械定位与距离估计，解决单模态信息不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.25555](https://arxiv.org/pdf/2603.25555)**

> **作者:** Nikolo Rohrmoser; Ghazal Ghazaei; Michael Sommersperger; Nassir Navab
>
> **摘要:** Purpose: The integration of multimodal imaging into operating rooms paves the way for comprehensive surgical scene understanding. In ophthalmic surgery, by now, two complementary imaging modalities are available: operating microscope (OPMI) imaging and real-time intraoperative optical coherence tomography (iOCT). This first work toward temporal OPMI and iOCT feature fusion demonstrates the potential of multimodal image processing for multi-head prediction through the example of precise instrument tracking in vitreoretinal surgery. Methods: We propose a multimodal, temporal, real-time capable network architecture to perform joint instrument detection, keypoint localization, and tool-tissue distance estimation. Our network design integrates a cross-attention fusion module to merge OPMI and iOCT image features, which are efficiently extracted via a YoloNAS and a CNN encoder, respectively. Furthermore, a region-based recurrent module leverages temporal coherence. Results: Our experiments demonstrate reliable instrument localization and keypoint detection (95.79% mAP50) and show that the incorporation of iOCT significantly improves tool-tissue distance estimation, while achieving real-time processing rates of 22.5 ms per frame. Especially for close distances to the retina (below 1 mm), the distance estimation accuracy improved from 284 $\mu m$ (OPMI only) to 33 $\mu m$ (multimodal). Conclusion: Feature fusion of multimodal imaging can enhance multi-task prediction accuracy compared to single-modality processing and real-time processing performance can be achieved through tailored network design. While our results demonstrate the potential of multi-modal processing for image-guided vitreoretinal surgery, they also underline key challenges that motivate future research toward more reliable, consistent, and comprehensive surgical scene understanding.
>
---
#### [new 120] Improving Fine-Grained Rice Leaf Disease Detection via Angular-Compactness Dual Loss Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于细粒度分类任务，旨在解决水稻叶片疾病识别中的类内差异大、类间相似的问题。通过引入角紧致双损失框架，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2603.25006](https://arxiv.org/pdf/2603.25006)**

> **作者:** Md. Rokon Mia; Rakib Hossain Sajib; Abdullah Al Noman; Abir Ahmed; B M Taslimul Haque
>
> **摘要:** Early detection of rice leaf diseases is critical, as rice is a staple crop supporting a substantial share of the world's population. Timely identification of these diseases enables more effective intervention and significantly reduces the risk of large-scale crop losses. However, traditional deep learning models primarily rely on cross entropy loss, which often struggles with high intra-class variance and inter-class similarity, common challenges in plant pathology datasets. To tackle this, we propose a dual-loss framework that combines Center Loss and ArcFace Loss to enhance fine-grained classification of rice leaf diseases. The method is applied into three state-of-the-art backbone architectures: InceptionNetV3, DenseNet201, and EfficientNetB0 trained on the public Rice Leaf Dataset. Our approach achieves significant performance gains, with accuracies of 99.6%, 99.2% and 99.2% respectively. The results demonstrate that angular margin-based and center-based constraints substantially boost the discriminative strength of feature embeddings. In particular, the framework does not require major architectural modifications, making it efficient and practical for real-world deployment in farming environments.
>
---
#### [new 121] Calibri: Enhancing Diffusion Transformers via Parameter-Efficient Calibration
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在提升扩散Transformer的性能。通过引入少量参数优化模型，提出Calibri方法，有效提高生成质量并减少推理步骤。**

- **链接: [https://arxiv.org/pdf/2603.24800](https://arxiv.org/pdf/2603.24800)**

> **作者:** Danil Tokhchukov; Aysel Mirzoeva; Andrey Kuznetsov; Konstantin Sobolev
>
> **备注:** Accepted to CVRP 2026, Project page: this https URL
>
> **摘要:** In this paper, we uncover the hidden potential of Diffusion Transformers (DiTs) to significantly enhance generative tasks. Through an in-depth analysis of the denoising process, we demonstrate that introducing a single learned scaling parameter can significantly improve the performance of DiT blocks. Building on this insight, we propose Calibri, a parameter-efficient approach that optimally calibrates DiT components to elevate generative quality. Calibri frames DiT calibration as a black-box reward optimization problem, which is efficiently solved using an evolutionary algorithm and modifies just ~100 parameters. Experimental results reveal that despite its lightweight design, Calibri consistently improves performance across various text-to-image models. Notably, Calibri also reduces the inference steps required for image generation, all while maintaining high-quality outputs.
>
---
#### [new 122] CARE: Training-Free Controllable Restoration for Medical Images via Dual-Latent Steering
- **分类: cs.CV**

- **简介: 该论文属于医学图像修复任务，旨在解决现有方法依赖重新训练且控制性差的问题。提出CARE框架，通过双潜在空间策略实现结构保真与增强的平衡。**

- **链接: [https://arxiv.org/pdf/2603.25026](https://arxiv.org/pdf/2603.25026)**

> **作者:** Xu Liu
>
> **摘要:** Medical image restoration is essential for improving the usability of noisy, incomplete, and artifact-corrupted clinical scans, yet existing methods often rely on task-specific retraining and offer limited control over the trade-off between faithful reconstruction and prior-driven enhancement. This lack of controllability is especially problematic in clinical settings, where overly aggressive restoration may introduce hallucinated details or alter diagnostically important structures. In this work, we propose CARE, a training-free controllable restoration framework for real-world medical images that explicitly balances structure preservation and prior-guided refinement during inference. CARE uses a dual-latent restoration strategy, in which one branch enforces data fidelity and anatomical consistency while the other leverages a generative prior to recover missing or degraded information. A risk-aware adaptive controller dynamically adjusts the contribution of each branch based on restoration uncertainty and local structural reliability, enabling conservative or enhancement-focused restoration modes without additional model training. We evaluate CARE on noisy and incomplete medical imaging scenarios and show that it achieves strong restoration quality while better preserving clinically relevant structures and reducing the risk of implausible reconstructions and show that it achieves strong restoration quality while better preserving clinically relevant structures and reducing the risk of implausible reconstructions. The proposed approach offers a practical step toward safer, more controllable, and more deployment-ready medical image restoration.
>
---
#### [new 123] Training-free Detection and 6D Pose Estimation of Unseen Surgical Instruments
- **分类: cs.CV**

- **简介: 该论文属于手术器械检测与6D位姿估计任务，解决未见过的器械在无监督情况下的精准定位问题。通过多视图几何和轮廓优化，实现无需训练的高精度位姿估计。**

- **链接: [https://arxiv.org/pdf/2603.25228](https://arxiv.org/pdf/2603.25228)**

> **作者:** Jonas Hein; Lilian Calvet; Matthias Seibold; Siyu Tang; Marc Pollefeys; Philipp Fürnstahl
>
> **备注:** Accepted at IJCARS: IPCAI 2026
>
> **摘要:** Purpose: Accurate detection and 6D pose estimation of surgical instruments are crucial for many computer-assisted interventions. However, supervised methods lack flexibility for new or unseen tools and require extensive annotated data. This work introduces a training-free pipeline for accurate multi-view 6D pose estimation of unseen surgical instruments, which only requires a textured CAD model as prior knowledge. Methods: Our pipeline consists of two main stages. First, for detection, we generate object mask proposals in each view and score their similarity to rendered templates using a pre-trained feature extractor. Detections are matched across views, triangulated into 3D instance candidates, and filtered using multi-view geometric consistency. Second, for pose estimation, a set of pose hypotheses is iteratively refined and scored using feature-metric scores with cross-view attention. The best hypothesis undergoes a final refinement using a novel multi-view, occlusion-aware contour registration, which minimizes reprojection errors of unoccluded contour points. Results: The proposed method was rigorously evaluated on real-world surgical data from the MVPSP dataset. The method achieves millimeter-accurate pose estimates that are on par with supervised methods under controlled conditions, while maintaining full generalization to unseen instruments. These results demonstrate the feasibility of training-free, marker-less detection and tracking in surgical scenes, and highlight the unique challenges in surgical environments. Conclusion: We present a novel and flexible pipeline that effectively combines state-of-the-art foundational models, multi-view geometry, and contour-based refinement for high-accuracy 6D pose estimation of surgical instruments without task-specific training. This approach enables robust instrument tracking and scene understanding in dynamic clinical environments.
>
---
#### [new 124] Bilingual Text-to-Motion Generation: A New Benchmark and Baselines
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到动作生成任务，旨在解决跨语言语义理解不足和缺乏双语数据的问题。提出BiHumanML3D数据集和BiMD模型，通过跨语言对齐提升双语输入的运动生成效果。**

- **链接: [https://arxiv.org/pdf/2603.25178](https://arxiv.org/pdf/2603.25178)**

> **作者:** Wanjiang Weng; Xiaofeng Tan; Xiangbo Shu; Guo-Sen Xie; Pan Zhou; Hongsong Wang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Text-to-motion generation holds significant potential for cross-linguistic applications, yet it is hindered by the lack of bilingual datasets and the poor cross-lingual semantic understanding of existing language models. To address these gaps, we introduce BiHumanML3D, the first bilingual text-to-motion benchmark, constructed via LLM-assisted annotation and rigorous manual correction. Furthermore, we propose a simple yet effective baseline, Bilingual Motion Diffusion (BiMD), featuring Cross-Lingual Alignment (CLA). CLA explicitly aligns semantic representations across languages, creating a robust conditional space that enables high-quality motion generation from bilingual inputs, including zero-shot code-switching scenarios. Extensive experiments demonstrate that BiMD with CLA achieves an FID of 0.045 vs. 0.169 and R@3 of 82.8\% vs. 80.8\%, significantly outperforms monolingual diffusion models and translation baselines on BiHumanML3D, underscoring the critical necessity and reliability of our dataset and the effectiveness of our alignment strategy for cross-lingual motion synthesis. The dataset and code are released at \href{this https URL}{this https URL}
>
---
#### [new 125] Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决高分辨率合成中传统方法因几何复杂度导致的扩展性问题。通过减少高斯粒子数量并引入纹理，实现4K级高质量视图合成。**

- **链接: [https://arxiv.org/pdf/2603.25745](https://arxiv.org/pdf/2603.25745)**

> **作者:** Yixing Lao; Xuyang Bai; Xiaoyang Wu; Nuoyuan Yan; Zixin Luo; Tian Fang; Jean-Daniel Nahmias; Yanghai Tsin; Shiwei Li; Hengshuang Zhao
>
> **摘要:** Existing feed-forward 3D Gaussian Splatting methods predict pixel-aligned primitives, leading to a quadratic growth in primitive count as resolution increases. This fundamentally limits their scalability, making high-resolution synthesis such as 4K intractable. We introduce LGTM (Less Gaussians, Texture More), a feed-forward framework that overcomes this resolution scaling barrier. By predicting compact Gaussian primitives coupled with per-primitive textures, LGTM decouples geometric complexity from rendering resolution. This approach enables high-fidelity 4K novel view synthesis without per-scene optimization, a capability previously out of reach for feed-forward methods, all while using significantly fewer Gaussian primitives. Project page: this https URL
>
---
#### [new 126] LanteRn: Latent Visual Structured Reasoning
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LanteRn，解决视觉推理效率问题，通过引入潜在视觉表示实现语言与视觉的高效融合。**

- **链接: [https://arxiv.org/pdf/2603.25629](https://arxiv.org/pdf/2603.25629)**

> **作者:** André G. Viveiros; Nuno Gonçalves; Matthias Lindemann; André Martins
>
> **摘要:** While language reasoning models excel in many tasks, visual reasoning remains challenging for current large multimodal models (LMMs). As a result, most LMMs default to verbalizing perceptual content into text, a strong limitation for tasks requiring fine-grained spatial and visual understanding. While recent approaches take steps toward thinking with images by invoking tools or generating intermediate images, they either rely on external modules, or incur unnecessary computation by reasoning directly in pixel space. In this paper, we introduce LanteRn, a framework that enables LMMs to interleave language with compact latent visual representations, allowing visual reasoning to occur directly in latent space. LanteRn augments a vision-language transformer with the ability to generate and attend to continuous visual thought embeddings during inference. We train the model in two stages: supervised fine-tuning to ground visual features in latent states, followed by reinforcement learning to align latent reasoning with task-level utility. We evaluate LanteRn on three perception-centric benchmarks (VisCoT, V*, and Blink), observing consistent improvements in visual grounding and fine-grained reasoning. These results suggest that internal latent representations provide a promising direction for more efficient multimodal reasoning.
>
---
#### [new 127] PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders
- **分类: cs.CV**

- **简介: 该论文提出PMT模型，用于图像和视频分割任务。针对现有方法需微调编码器的问题，设计了不改变编码器的快速分割解码器，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.25398](https://arxiv.org/pdf/2603.25398)**

> **作者:** Niccolò Cavagnero; Narges Norouzi; Gijs Dubbelman; Daan de Geus
>
> **备注:** 8 pages, ECV 2026, CVPR Workshop
>
> **摘要:** Vision Foundation Models (VFMs) pre-trained at scale enable a single frozen encoder to serve multiple downstream tasks simultaneously. Recent VFM-based encoder-only models for image and video segmentation, such as EoMT and VidEoMT, achieve competitive accuracy with remarkably low latency, yet they require finetuning the encoder, sacrificing the multi-task encoder sharing that makes VFMs practically attractive for large-scale deployment. To reconcile encoder-only simplicity and speed with frozen VFM features, we propose the Plain Mask Decoder (PMD), a fast Transformer-based segmentation decoder that operates on top of frozen VFM features. The resulting model, the Plain Mask Transformer (PMT), preserves the architectural simplicity and low latency of encoder-only designs while keeping the encoder representation unchanged and shareable. The design seamlessly applies to both image and video segmentation, inheriting the generality of the encoder-only framework. On standard image segmentation benchmarks, PMT matches the frozen-encoder state of the art while running up to ~3x faster. For video segmentation, it even performs on par with fully finetuned methods, while being up to 8x faster than state-of-the-art frozen-encoder models. Code: this https URL.
>
---
#### [new 128] Towards Video Anomaly Detection from Event Streams: A Baseline and Benchmark Datasets
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，解决事件流数据缺乏和建模策略不足的问题。构建了基准数据集，提出EWAD框架，提升异常检测效果。**

- **链接: [https://arxiv.org/pdf/2603.24991](https://arxiv.org/pdf/2603.24991)**

> **作者:** Peng Wu; Yuting Yan; Guansong Pang; Yujia Sun; Qingsen Yan; Peng Wang; Yanning Zhang
>
> **摘要:** Event-based vision, characterized by low redundancy, focus on dynamic motion, and inherent privacy-preserving properties, naturally fits the demands of video anomaly detection (VAD). However, the absence of dedicated event-stream anomaly detection datasets and effective modeling strategies has significantly hindered progress in this field. In this work, we take the first major step toward establishing event-based VAD as a unified research direction. We first construct multiple event-stream based benchmarks for video anomaly detection, featuring synchronized event and RGB recordings. Leveraging the unique properties of events, we then propose an EVent-centric spatiotemporal Video Anomaly Detection framework, namely EWAD, with three key innovations: an event density aware dynamic sampling strategy to select temporally informative segments; a density-modulated temporal modeling approach that captures contextual relations from sparse event streams; and an RGB-to-event knowledge distillation mechanism to enhance event-based representations under weak supervision. Extensive experiments on three benchmarks demonstrate that our EWAD achieves significant improvements over existing approaches, highlighting the potential and effectiveness of event-driven modeling for video anomaly detection. The benchmark datasets will be made publicly available.
>
---
#### [new 129] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding
- **分类: cs.CV**

- **简介: 该论文针对视频时间定位任务，解决模型泛化能力不足的问题。提出SlotVTG框架，通过对象中心的轻量适配器提升模型对实际视觉内容的准确理解。**

- **链接: [https://arxiv.org/pdf/2603.25733](https://arxiv.org/pdf/2603.25733)**

> **作者:** Jiwook Han; Geo Ahn; Youngrae Kim; Jinwoo Choi
>
> **备注:** Accepted to GRAIL-V workshop at CVPR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown strong performance on Video Temporal Grounding (VTG). However, their coarse recognition capabilities are insufficient for fine-grained temporal understanding, making task-specific fine-tuning indispensable. This fine-tuning causes models to memorize dataset-specific shortcuts rather than faithfully grounding in the actual visual content, leading to poor Out-of-Domain (OOD) generalization. Object-centric learning offers a promising remedy by decomposing scenes into entity-level representations, but existing approaches require re-running the entire multi-stage training pipeline from scratch. We propose SlotVTG, a framework that steers MLLMs toward object-centric, input-grounded visual reasoning at minimal cost. SlotVTG introduces a lightweight slot adapter that decomposes visual tokens into abstract slots via slot attention and reconstructs the original sequence, where objectness priors from a self-supervised vision model encourage semantically coherent slot formation. Cross-domain evaluation on standard VTG benchmarks demonstrates that our approach significantly improves OOD robustness while maintaining competitive In-Domain (ID) performance with minimal overhead.
>
---
#### [new 130] Knowledge-Guided Adversarial Training for Infrared Object Detection via Thermal Radiation Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于红外目标检测任务，旨在提升模型在复杂环境下的鲁棒性。针对红外图像易受干扰的问题，提出KGAT方法，结合物理知识与对抗训练，增强检测稳定性。**

- **链接: [https://arxiv.org/pdf/2603.25170](https://arxiv.org/pdf/2603.25170)**

> **作者:** Shiji Zhao; Shukun Xiong; Maoxun Yuan; Yao Huang; Ranjie Duan; Qing Guo; Jiansheng Chen; Haibin Duan; Xingxing Wei
>
> **备注:** Accepted for publication in the International Journal of Computer Vision (IJCV)
>
> **摘要:** In complex environments, infrared object detection exhibits broad applicability and stability across diverse scenarios. However, infrared object detection is vulnerable to both common corruptions and adversarial examples, leading to potential security risks. To improve the robustness of infrared object detection, current methods mostly adopt a data-driven ideology, which only superficially drives the network to fit the training data without specifically considering the unique characteristics of infrared images, resulting in limited robustness. In this paper, we revisit infrared physical knowledge and find that relative thermal radiation relations between different classes can be regarded as a reliable knowledge source under the complex scenarios of adversarial examples and common corruptions. Thus, we theoretically model thermal radiation relations based on the rank order of gray values for different classes, and further quantify the stability of various inter-class thermal radiation relations. Based on the above theoretical framework, we propose Knowledge-Guided Adversarial Training (KGAT) for infrared object detection, in which infrared physical knowledge is embedded into the adversarial training process, and the predicted results are optimized to be consistent with the actual physical laws. Extensive experiments on three infrared datasets and six mainstream infrared object detection models demonstrate that KGAT effectively enhances both clean accuracy and robustness against adversarial attacks and common corruptions.
>
---
#### [new 131] Hierarchy-Guided Multimodal Representation Learning for Taxonomic Inference
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生物分类任务，解决多模态数据下的分类问题。通过引入层次结构和融合机制，提升分类准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.25573](https://arxiv.org/pdf/2603.25573)**

> **作者:** Sk Miraj Ahmed; Xi Yu; Yunqi Li; Yuewei Lin; Wei Xu
>
> **备注:** Accepted at the ICLR 2026 Workshop on Foundation Models for Science (FM4Science)
>
> **摘要:** Accurate biodiversity identification from large-scale field data is a foundational problem with direct impact on ecology, conservation, and environmental monitoring. In practice, the core task is taxonomic prediction - inferring order, family, genus, or species from imperfect inputs such as specimen images, DNA barcodes, or both. Existing multimodal methods often treat taxonomy as a flat label space and therefore fail to encode the hierarchical structure of biological classification, which is critical for robustness under noise and missing modalities. We present two end-to-end variants for hierarchy-aware multimodal learning: CLiBD-HiR, which introduces Hierarchical Information Regularization (HiR) to shape embedding geometry across taxonomic levels, yielding structured and noise-robust representations; and CLiBD-HiR-Fuse, which additionally trains a lightweight fusion predictor that supports image-only, DNA-only, or joint inference and is resilient to modality corruption. Across large-scale biodiversity benchmarks, our approach improves taxonomic classification accuracy by over 14 percent compared to strong multimodal baselines, with particularly large gains under partial and corrupted DNA conditions. These results highlight that explicitly encoding biological hierarchy, together with flexible fusion, is key for practical biodiversity foundation models.
>
---
#### [new 132] GDPO-Listener: Expressive Interactive Head Generation via Auto-Regressive Flow Matching and Group reward-Decoupled Policy Optimization
- **分类: cs.CV**

- **简介: 该论文属于虚拟人类合成任务，解决对话中听众动作生成不自然、缺乏表达的问题。提出GDPO-Listener框架，结合流匹配和分组奖励优化，提升动作的多样性与可控性。**

- **链接: [https://arxiv.org/pdf/2603.25020](https://arxiv.org/pdf/2603.25020)**

> **作者:** Zhangyu Jin; Maksim Siniukov; Deuksin Kwon; Ashutosh Chaubey; Mohammad Soleymani
>
> **摘要:** Generating realistic 3D head motion for dyadic interactions is a significant challenge in virtual human synthesis. While recent methods achieve impressive results with speaking heads, they frequently suffer from the `Regression-to-the-Mean' problem in listener motions, collapsing into static faces, and lack the parameter space for complex nonverbal motions. In this paper, we propose GDPO-Listener, a novel framework that achieves highly expressive speaking and listening motion generation. First, we introduce an Auto-Regressive Flow Matching architecture enabling stable supervised learning. Second, to overcome kinematic stillness, we apply the Group reward-Decoupled Policy Optimization (GDPO). By isolating reward normalization across distinct FLAME parameter groups, GDPO explicitly incentivizes high variance expressive generations. Finally, we enable explicit semantic text control for customizable responses. Extensive evaluations across the Seamless Interaction and DualTalk datasets demonstrate superior performance compared to existing baselines on long-term kinematic variance, visual expressivity and semantic controllability.
>
---
#### [new 133] C2W-Tune: Cavity-to -Wall Transfer Learning for Thin Atrial Wall Segmentation in 3D Late Gadolinium-enhanced Magnetic Resonance
- **分类: cs.CV**

- **简介: 该论文属于左心房壁分割任务，解决3D LGE-MRI中薄壁分割困难的问题。通过C2W-Tune框架，利用腔体模型提升分割精度。**

- **链接: [https://arxiv.org/pdf/2603.24992](https://arxiv.org/pdf/2603.24992)**

> **作者:** Yusri Al-Sanaani; Rebecca Thornhill; Sreeraman Rajan
>
> **备注:** Submitted this to the International Conference on Artificial Intelligence in Medicine (AIME 2026)
>
> **摘要:** Accurate segmentation of the left atrial (LA) wall in 3D late gadolinium-enhanced MRI (LGE-MRI) is essential for wall thickness mapping and fibrosis quantification, yet it remains challenging due to the wall's thinness, complex anatomy, and low contrast. We propose C2W-Tune, a two-stage cavity-to-wall transfer framework that leverages a high-accuracy LA cavity model as an anatomical prior to improve thin-wall delineation. Using a 3D U-Net with a ResNeXt encoder and instance normalization, Stage 1 pre-trains the network to segment the LA cavity, learning robust atrial representations. Stage 2 transfers these weights and adapts the network to LA wall segmentation using a progressive layer-unfreezing schedule to preserve endocardial features while enabling wall-specific refinement. Experiments on the 2018 LA Segmentation Challenge dataset demonstrate substantial gains over an architecture-matched baseline trained from scratch: wall Dice improves from 0.623 to 0.814, and Surface Dice at 1 mm improves from 0.553 to 0.731. Boundary errors were substantially reduced, with the 95th-percentile Hausdorff distance (HD95) decreasing from 2.95 mm to 2.55 mm and the average symmetric surface distance (ASSD) from 0.71 mm to 0.63 mm. Furthermore, even with reduced supervision (70 training volumes sampled from the same training pool), C2W-Tune achieved a Dice score of 0.78 and an HD95 of 3.15 mm, maintaining competitive performance and exceeding multi-class benchmarks that typically report Dice values around 0.6-0.7. These results show that anatomically grounded task transfer with controlled fine-tuning improves boundary accuracy for thin LA wall segmentation in 3D LGE-MRI.
>
---
#### [new 134] CIV-DG: Conditional Instrumental Variables for Domain Generalization in Medical Imaging
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于医学影像领域的域泛化任务，旨在解决因选择偏差导致的模型跨医院泛化能力差的问题。提出CIV-DG框架，利用条件工具变量方法分离病理特征与设备伪影，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.25202](https://arxiv.org/pdf/2603.25202)**

> **作者:** Shaojin Bai; Yuting Su; Weizhi Nie
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Cross-site generalizability in medical AI is fundamentally compromised by selection bias, a structural mechanism where patient demographics (e.g., age, severity) non-randomly dictate hospital assignment. Conventional Domain Generalization (DG) paradigms, which predominantly target image-level distribution shifts, fail to address the resulting spurious correlations between site-specific variations and diagnostic labels. To surmount this identifiability barrier, we propose CIV-DG, a causal framework that leverages Conditional Instrumental Variables to disentangle pathological semantics from scanner-induced artifacts. By relaxing the strict random assignment assumption of standard IV methods, CIV-DG accommodates complex clinical scenarios where hospital selection is endogenously driven by patient demographics. We instantiate this theory via a Deep Generalized Method of Moments (DeepGMM) architecture, employing a conditional critic to minimize moment violations and enforce instrument-error orthogonality within demographic strata. Extensive experiments on the Camelyon17 benchmark and large-scale Chest X-Ray datasets demonstrate that CIV-DG significantly outperforms leading baselines, validating the efficacy of conditional causal mechanisms in resolving structural confounding for robust medical AI.
>
---
#### [new 135] Adaptive Learned Image Compression with Graph Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决传统方法在建模局部与全局冗余上的不足。通过引入图神经网络，构建自适应图结构以提升压缩效率。**

- **链接: [https://arxiv.org/pdf/2603.25316](https://arxiv.org/pdf/2603.25316)**

> **作者:** Yunuo Chen; Bing He; Zezheng Lyu; Hongwei Hu; Qunshan Gu; Yuan Tian; Guo Lu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Efficient image compression relies on modeling both local and global redundancy. Most state-of-the-art (SOTA) learned image compression (LIC) methods are based on CNNs or Transformers, which are inherently rigid. Standard CNN kernels and window-based attention mechanisms impose fixed receptive fields and static connectivity patterns, which potentially couple non-redundant pixels simply due to their proximity in Euclidean space. This rigidity limits the model's ability to adaptively capture spatially varying redundancy across the image, particularly at the global level. To overcome these limitations, we propose a content-adaptive image compression framework based on Graph Neural Networks (GNNs). Specifically, our approach constructs dual-scale graphs that enable flexible, data-driven receptive fields. Furthermore, we introduce adaptive connectivity by dynamically adjusting the number of neighbors for each node based on local content complexity. These innovations empower our Graph-based Learned Image Compression (GLIC) model to effectively model diverse redundancy patterns across images, leading to more efficient and adaptive compression. Experiments demonstrate that GLIC achieves state-of-the-art performance, achieving BD-rate reductions of 19.29%, 21.69%, and 18.71% relative to VTM-9.1 on Kodak, Tecnick, and CLIC, respectively. Code will be released at this https URL.
>
---
#### [new 136] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的3D空间理解任务，旨在提升模型对三维结构、物体属性及关系的推理能力。通过构建层次化框架和数据集，增强模型的空间智能。**

- **链接: [https://arxiv.org/pdf/2603.25411](https://arxiv.org/pdf/2603.25411)**

> **作者:** Huizhi Liang; Yichao Shen; Yu Deng; Sicheng Xu; Zhiyuan Feng; Tong Zhang; Yaobo Liang; Jiaolong Yang
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Achieving human-like spatial intelligence for vision-language models (VLMs) requires inferring 3D structures from 2D observations, recognizing object properties and relations in 3D space, and performing high-level spatial reasoning. In this paper, we propose a principled hierarchical framework that decomposes the learning of 3D spatial understanding in VLMs into four progressively complex levels, from geometric perception to abstract spatial reasoning. Guided by this framework, we construct an automated pipeline that processes approximately 5M images with over 45M objects to generate 3D spatial VQA pairs across diverse tasks and scenes for VLM supervised fine-tuning. We also develop an RGB-D VLM incorporating metric-scale point maps as auxiliary inputs to further enhance spatial understanding. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on multiple spatial understanding and reasoning benchmarks, surpassing specialized spatial models and large proprietary systems such as Gemini-2.5-pro and GPT-5. Moreover, our analysis reveals clear dependencies among hierarchical task levels, offering new insights into how multi-level task design facilitates the emergence of 3D spatial intelligence.
>
---
#### [new 137] SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment
- **分类: cs.CV**

- **简介: 该论文属于小目标检测任务，旨在解决地面对空反无人机监测中的小目标检测难题。提出SDD-YOLO框架，提升检测精度与部署效率。**

- **链接: [https://arxiv.org/pdf/2603.25218](https://arxiv.org/pdf/2603.25218)**

> **作者:** Pengyu Chen; Haotian Sa; Yiwei Hu; Yuhan Cheng; Junbo Wang
>
> **摘要:** Detecting small unmanned aerial vehicles (UAVs) from a ground-to-air (G2A) perspective presents significant challenges, including extremely low pixel occupancy, cluttered aerial backgrounds, and strict real-time constraints. Existing YOLO-based detectors are primarily optimized for general object detection and often lack adequate feature resolution for sub-pixel targets, while introducing complexities during deployment. In this paper, we propose SDD-YOLO, a small-target detection framework tailored for G2A anti-UAV surveillance. To capture fine-grained spatial details critical for micro-targets, SDD-YOLO introduces a P2 high-resolution detection head operating at 4 times downsampling. Furthermore, we integrate the recent architectural advancements from YOLO26, including a DFL-free, NMS-free architecture for streamlined inference, and the MuSGD hybrid training strategy with ProgLoss and STAL, which substantially mitigates gradient oscillation on sparse small-target signals. To support our evaluation, we construct DroneSOD-30K, a large-scale G2A dataset comprising approximately 30,000 annotated images covering diverse meteorological conditions. Experiments demonstrate that SDD-YOLO-n achieves a mAP@0.5 of 86.0% on DroneSOD-30K, surpassing the YOLOv5n baseline by 7.8 percentage points. Extensive inference analysis shows our model attains 226 FPS on an NVIDIA RTX 5090 and 35 FPS on an Intel Xeon CPU, demonstrating exceptional efficiency for future edge deployment.
>
---
#### [new 138] FSGNet: A Frequency-Aware and Semantic Guidance Network for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决U-Net在特征传递中的语义退化问题。提出FSGNet，结合频域感知与语义引导机制，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.25389](https://arxiv.org/pdf/2603.25389)**

> **作者:** Yingmei Zhang; Wangtao Bao; Yong Yang; Weiguo Wan; Qin Xiao; Xueting Zou
>
> **摘要:** Infrared small target detection (IRSTD) aims to identify and distinguish small targets from complex backgrounds. Leveraging the powerful multi-scale feature fusion capability of the U-Net architecture, IRSTD has achieved significant progress. However, U-Net suffers from semantic degradation when transferring high-level features from deep to shallow layers, limiting the precise localization of small targets. To address this issue, this paper proposes FSGNet, a lightweight and effective detection framework incorporating frequency-aware and semantic guidance mechanisms. Specifically, a multi-directional interactive attention module is proposed throughout the encoder to capture fine-grained and directional features, enhancing the network's sensitivity to small, low-contrast targets. To suppress background interference propagated through skip connections, a multi-scale frequency-aware module leverages Fast Fourier transform to filter out target-similar clutter while preserving salient target structures. At the deepest layer, a global pooling module captures high-level semantic information, which is subsequently upsampled and propagated to each decoder stage through the global semantic guidance flows, ensuring semantic consistency and precise localization across scales. Extensive experiments on four public IRSTD datasets demonstrate that FSGNet achieves superior detection performance and maintains high efficiency, highlighting its practical applicability and robustness. The codes will be released on this https URL.
>
---
#### [new 139] ET-SAM: Efficient Point Prompt Prediction in SAM for Unified Scene Text Detection and Layout Analysis
- **分类: cs.CV**

- **简介: 该论文提出ET-SAM，解决SAM在场景文本检测与版面分析中的推理慢和数据利用率低问题，通过轻量点解码器和联合训练策略提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.25168](https://arxiv.org/pdf/2603.25168)**

> **作者:** Xike Zhang; Maoyuan Ye; Juhua Liu; Bo Du
>
> **备注:** 20 pages, 8 figures, 8 tables. Submitted to ECCV 2026
>
> **摘要:** Previous works based on Segment Anything Model (SAM) have achieved promising performance in unified scene text detection and layout analysis. However, the typical reliance on pixel-level text segmentation for sampling thousands of foreground points as prompts leads to unsatisfied inference latency and limited data utilization. To address above issues, we propose ET-SAM, an Efficient framework with two decoders for unified scene Text detection and layout analysis based on SAM. Technically, we customize a lightweight point decoder that produces word heatmaps for achieving a few foreground points, thereby eliminating excessive point prompts and accelerating inference. Without the dependence on pixel-level segmentation, we further design a joint training strategy to leverage existing data with heterogeneous text-level annotations. Specifically, the datasets with multi-level, word-level only, and line-level only annotations are combined in parallel as a unified training set. For these datasets, we introduce three corresponding sets of learnable task prompts in both the point decoder and hierarchical mask decoder to mitigate discrepancies across this http URL experiments demonstrate that, compared to the previous SAM-based architecture, ET-SAM achieves about 3$\times$ inference acceleration while obtaining competitive performance on HierText, and improves an average of 11.0% F-score on Total-Text, CTW1500, and ICDAR15.
>
---
#### [new 140] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator
- **分类: cs.CV**

- **简介: 该论文提出GaussFusion，用于提升3DGS在真实场景中的重建质量。解决相机姿态误差、覆盖不足等问题，通过几何感知视频生成优化重建结果。**

- **链接: [https://arxiv.org/pdf/2603.25053](https://arxiv.org/pdf/2603.25053)**

> **作者:** Liyuan Zhu; Manjunath Narayana; Michal Stary; Will Hutchcroft; Gordon Wetzstein; Iro Armeni
>
> **备注:** CVPR 2026 main paper camera-ready. Project page: this http URL
>
> **摘要:** We present GaussFusion, a novel approach for improving 3D Gaussian splatting (3DGS) reconstructions in the wild through geometry-informed video generation. GaussFusion mitigates common 3DGS artifacts, including floaters, flickering, and blur caused by camera pose errors, incomplete coverage, and noisy geometry initialization. Unlike prior RGB-based approaches limited to a single reconstruction pipeline, our method introduces a geometry-informed video-to-video generator that refines 3DGS renderings across both optimization-based and feed-forward methods. Given an existing reconstruction, we render a Gaussian primitive video buffer encoding depth, normals, opacity, and covariance, which the generator refines to produce temporally coherent, artifact-free frames. We further introduce an artifact synthesis pipeline that simulates diverse degradation patterns, ensuring robustness and generalization. GaussFusion achieves state-of-the-art performance on novel-view synthesis benchmarks, and an efficient variant runs in real time at 21 FPS while maintaining similar performance, enabling interactive 3D applications.
>
---
#### [new 141] Learning to Rank Caption Chains for Video-Text Alignment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频-文本对齐任务，旨在解决传统DPO方法在视觉语言模型中的不足。通过排名优化提升生成内容的视觉一致性，提出生成降级字幕链的方法。**

- **链接: [https://arxiv.org/pdf/2603.25145](https://arxiv.org/pdf/2603.25145)**

> **作者:** Ansel Blume; Burak Uzkent; Shalini Chaudhuri; Garin Kessler
>
> **摘要:** Direct preference optimization (DPO) is an effective technique to train language models to generate preferred over dispreferred responses. However, this binary "winner-takes-all" approach is suboptimal for vision-language models whose response quality is highly dependent on visual content. In particular, a response may still be faithful to the visual inputs even if it is less preferable than an alternative. The standard Bradley-Terry DPO formulation lacks this nuance, upweighting winning responses without sufficient regard for whether the "losing" response still maintains high visual fidelity. In this work, we investigate ranking optimization as an alternative that more precisely situates responses' faithfulness to visual inputs. We focus on video-text alignment using detailed video captions, proposing a method to generate challenging, totally ordered caption chains at scale through repeated caption degradation. Our results show ranking optimization outperforms binary DPO for long-form content generation and assessment, and importantly, we find that these approaches require finetuning of the vision encoder to be effective, challenging the view of DPO as purely a language-reweighting process.
>
---
#### [new 142] DC-Reg: Globally Optimal Point Cloud Registration via Tight Bounding with Difference of Convex Programming
- **分类: cs.CV**

- **简介: 该论文属于点云配准任务，解决部分重叠和大偏移下的全局最优配准问题。提出DC-Reg框架，通过DC编程紧致下界，提升配准效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.25442](https://arxiv.org/pdf/2603.25442)**

> **作者:** Wei Lian; Fei Ma; Hang Pan; Zhesen Cui; Wangmeng Zuo
>
> **摘要:** Achieving globally optimal point cloud registration under partial overlaps and large misalignments remains a fundamental challenge. While simultaneous transformation ($\boldsymbol{\theta}$) and correspondence ($\mathbf{P}$) estimation has the advantage of being robust to nonrigid deformation, its non-convex coupled objective often leads to local minima for heuristic methods and prohibitive convergence times for existing global solvers due to loose lower bounds. To address this, we propose DC-Reg, a robust globally optimal framework that significantly tightens the Branch-and-Bound (BnB) search. Our core innovation is the derivation of a holistic concave underestimator for the coupled transformation-assignment objective, grounded in the Difference of Convex (DC) programming paradigm. Unlike prior works that rely on term-wise relaxations (e.g., McCormick envelopes) which neglect variable interplay, our holistic DC decomposition captures the joint structural interaction between $\boldsymbol{\theta}$ and $\mathbf{P}$. This formulation enables the computation of remarkably tight lower bounds via efficient Linear Assignment Problems (LAP) evaluated at the vertices of the search boxes. We validate our framework on 2D similarity and 3D rigid registration, utilizing rotation-invariant features for the latter to achieve high efficiency without sacrificing optimality. Experimental results on synthetic data and the 3DMatch benchmark demonstrate that DC-Reg achieves significantly faster convergence and superior robustness to extreme noise and outliers compared to state-of-the-art global techniques.
>
---
#### [new 143] Infinite Gaze Generation for Videos with Autoregressive Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频中人类注视预测任务，解决长期时间依赖和精细动态捕捉问题。通过自回归扩散模型生成连续空间坐标和高分辨率时间戳的注视轨迹。**

- **链接: [https://arxiv.org/pdf/2603.24938](https://arxiv.org/pdf/2603.24938)**

> **作者:** Jenna Kang; Colin Groth; Tong Wu; Finley Torrens; Patsorn Sangkloy; Gordon Wetzstein; Qi Sun
>
> **摘要:** Predicting human gaze in video is fundamental to advancing scene understanding and multimodal interaction. While traditional saliency maps provide spatial probability distributions and scanpaths offer ordered fixations, both abstractions often collapse the fine-grained temporal dynamics of raw gaze. Furthermore, existing models are typically constrained to short-term windows ($\approx$ 3-5s), failing to capture the long-range behavioral dependencies inherent in real-world content. We present a generative framework for infinite-horizon raw gaze prediction in videos of arbitrary length. By leveraging an autoregressive diffusion model, we synthesize gaze trajectories characterized by continuous spatial coordinates and high-resolution timestamps. Our model is conditioned on a saliency-aware visual latent space. Quantitative and qualitative evaluations demonstrate that our approach significantly outperforms existing approaches in long-range spatio-temporal accuracy and trajectory realism.
>
---
#### [new 144] Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型的强化学习任务，解决感知与推理 tokens 耦合优化问题，提出 ToR 策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.25077](https://arxiv.org/pdf/2603.25077)**

> **作者:** Jinda Lu; Junkang Wu; Jinghan Li; Kexin Huang; Shuo Yang; Guoyin Wang; Jiancan Wu; Xiang Wang; Xiangnan He
>
> **摘要:** Extending Reinforcement Learning with Verifiable Rewards (RLVR) to multimodal large language models (MLLMs) faces a fundamental challenge: their responses inherently interleave perception-related tokens, which ground visual content, with reasoning-related tokens, which construct reasoning chains. These token types instantiate distinct yet interdependent capacities -- visual grounding and symbolic reasoning -- making isolated optimization insufficient. Through token-level empirical analysis, we demonstrate that optimizing either perception- or reasoning-only tokens consistently underperforms full optimization, underscoring their inherent coupling. To address this, we propose a plug-and-play Token-Reweighting (ToR) strategy that explicitly models this interdependence by identifying critical tokens of both types and dynamically reweighting them during RLVR training. Applied on top of existing methods (e.g., GRPO and DAPO), ToR delivers consistent performance gains across multiple multi-modal reasoning benchmarks, achieving state-of-the-art performance with both accurate visual grounding and coherent reasoning.
>
---
#### [new 145] MoireMix: A Formula-Based Data Augmentation for Improving Image Classification Robustness
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，旨在提升模型的鲁棒性。针对现有数据增强方法计算成本高或依赖外部数据的问题，提出基于莫尔干涉图案的轻量级增强方法，通过数学公式生成结构化扰动，有效提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.25109](https://arxiv.org/pdf/2603.25109)**

> **作者:** Yuto Matsuo; Yoshihiro Fukuhara; Yuki M. Asano; Rintaro Yanagi; Hirokatsu Kataoka; Akio Nakamura
>
> **摘要:** Data augmentation is a key technique for improving the robustness of image classification models. However, many recent approaches rely on diffusion-based synthesis or complex feature mixing strategies, which introduce substantial computational overhead or require external datasets. In this work, we explore a different direction: procedural augmentation based on analytic interference patterns. Unlike conventional augmentation methods that rely on stochastic noise, feature mixing, or generative models, our approach exploits Moire interference to generate structured perturbations spanning a wide range of spatial frequencies. We propose a lightweight augmentation method that procedurally generates Moire textures on-the-fly using a closed-form mathematical formulation. The patterns are synthesized directly in memory with negligible computational cost (0.0026 seconds per image), mixed with training images during training, and immediately discarded, enabling a storage-free augmentation pipeline without external data. Extensive experiments with Vision Transformers demonstrate that the proposed method consistently improves robustness across multiple benchmarks, including ImageNet-C, ImageNet-R, and adversarial benchmarks, outperforming standard augmentation baselines and existing external-data-free augmentation approaches. These results suggest that analytic interference patterns provide a practical and efficient alternative to data-driven generative augmentation methods.
>
---
#### [new 146] Insights on back marking for the automated identification of animals
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于动物个体识别任务，旨在解决统一外观物种（如猪）的个性化标记设计问题。通过分析神经网络模型，提出有效后背标记设计的指导原则。**

- **链接: [https://arxiv.org/pdf/2603.25535](https://arxiv.org/pdf/2603.25535)**

> **作者:** David Brunner; Marie Bordes; Elisabeth Mayrhuber; Stephan M. Winkler; Viktoria Dorfer; Maciej Oczak
>
> **摘要:** To date, there is little research on how to design back marks to best support individual-level monitoring of uniform looking species like pigs. With the recent surge of machine learning-based monitoring solutions, there is a particular need for guidelines on the design of marks that can be effectively recognised by such algorithms. This study provides valuable insights on effective back mark design, based on the analysis of a machine learning model, trained to distinguish pigs via their back marks. Specifically, a neural network of type ResNet-50 was trained to classify ten pigs with unique back marks. The analysis of the model's predictions highlights the significance of certain design choices, even in controlled settings. Most importantly, the set of back marks must be designed such that each mark remains unambiguous under conditions of motion blur, diverse view angles and occlusions, caused by animal behaviour. Further, the back mark design must consider data augmentation strategies commonly employed during model training, like colour, flip and crop augmentations. The generated insights can support individual-level monitoring in future studies and real-world applications by optimizing back mark design.
>
---
#### [new 147] Dissecting Model Failures in Abdominal Aortic Aneurysm Segmentation through Explainability-Driven Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分割任务，解决复杂腹主动脉瘤分割模型失效问题。通过XAI引导的编码器优化，提升模型关注关键结构的能力。**

- **链接: [https://arxiv.org/pdf/2603.24801](https://arxiv.org/pdf/2603.24801)**

> **作者:** Abu Noman Md Sakib; Merjulah Roby; Zijie Zhang; Satish Muluk; Mark K. Eskandari; Ender A. Finol
>
> **摘要:** Computed tomography image segmentation of complex abdominal aortic aneurysms (AAA) often fails because the models assign internal focus to irrelevant structures or do not focus on thin, low-contrast targets. Where the model looks is the primary training signal, and thus we propose an Explainable AI (XAI) guided encoder shaping framework. Our method computes a dense, attribution-based encoder focus map ("XAI field") from the final encoder block and uses it in two complementary ways: (i) we align the predicted probability mass to the XAI field to promote agreement between focus and output; and (ii) we route the field into a lightweight refinement pathway and a confidence prior that modulates logits at inference, suppressing distractors while preserving subtle structures. The objective terms serve only as control signals; the contribution is the integration of attribution guidance into representation and decoding. We evaluate clinically validated challenging cases curated for failure-prone scenarios. Compared to a base SAM setup, our implementation yields substantial improvements. The observed gains suggest that explicitly optimizing encoder focus via XAI guidance is a practical and effective principle for reliable segmentation in complex scenarios.
>
---
#### [new 148] OptiSAR-Net++: A Large-Scale Benchmark and Transformer-Free Framework for Cross-Domain Remote Sensing Visual Grounding
- **分类: cs.CV**

- **简介: 该论文属于跨域遥感视觉定位任务，解决单传感器域方法的局限性。提出OptiSAR-Net++框架，实现高效跨域特征解耦与语义对齐，提升定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.24876](https://arxiv.org/pdf/2603.24876)**

> **作者:** Xiaoyu Tang; Jun Dong; Jintao Cheng; Rui Fan
>
> **摘要:** Remote sensing visual grounding (RSVG) aims to localize specific targets in remote sensing images using natural language expressions. However, existing methods are restricted to single-sensor domains, i.e., either optical or synthetic aperture radar (SAR), limiting their real-world applicability. In this paper, we introduce the Cross-Domain RSVG (CD-RSVG) task and construct OptSAR-RSVG, the first large-scale benchmark dataset for this setting. To tackle the challenges of cross-domain feature modeling, computational inefficiency, and fine-grained semantic discrimination, we propose OptiSAR-Net++. Our framework features a patch-level Low-Rank Adaptation Mixture of Experts (PL-MoE) for efficient cross-domain feature decoupling. To mitigate the substantial computational overhead of Transformer decoding frameworks, we adopt a CLIP-based contrastive paradigm and further incorporate dynamic adversarial negative sampling, thereby transforming generative regression into an efficient cross-modal matching process. Additionally, a text-guided dual-gate fusion module (TGDF-SSA) and a region-aware auxiliary head are introduced to enhance semantic-visual alignment and spatial modeling. Extensive experiments demonstrate that OptiSAR-Net++ achieves SOTA performance on both OptSAR-RSVG and DIOR-RSVG benchmarks, offering significant advantages in localization accuracy and efficiency. Our code and dataset will be made publicly available.
>
---
#### [new 149] Unleashing Guidance Without Classifiers for Human-Object Interaction Animation
- **分类: cs.CV**

- **简介: 该论文属于人-物交互动画生成任务，旨在解决动态人体动作与多样物体几何建模的难题。提出LIGHT方法，通过数据驱动方式实现无需分类器的引导，提升接触质量与生成 realism。**

- **链接: [https://arxiv.org/pdf/2603.25734](https://arxiv.org/pdf/2603.25734)**

> **作者:** Ziyin Wang; Sirui Xu; Chuan Guo; Bing Zhou; Jiangshan Gong; Jian Wang; Yu-Xiong Wang; Liang-Yan Gui
>
> **备注:** Project Page: this http URL
>
> **摘要:** Generating realistic human-object interaction (HOI) animations remains challenging because it requires jointly modeling dynamic human actions and diverse object geometries. Prior diffusion-based approaches often rely on hand-crafted contact priors or human-imposed kinematic constraints to improve contact quality. We propose LIGHT, a data-driven alternative in which guidance emerges from the denoising pace itself, reducing dependence on manually designed priors. Building on diffusion forcing, we factor the representation into modality-specific components and assign individualized noise levels with asynchronous denoising schedules. In this paradigm, cleaner components guide noisier ones through cross-attention, yielding guidance without auxiliary classifiers. We find that this data-driven guidance is inherently contact-aware, and can be enhanced when training is augmented with a broad spectrum of synthetic object geometries, encouraging invariance of contact semantics to geometric diversity. Extensive experiments show that pace-induced guidance more effectively mirrors the benefits of contact priors than conventional classifier-free guidance, while achieving higher contact fidelity, more realistic HOI generation, and stronger generalization to unseen objects and tasks.
>
---
#### [new 150] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions
- **分类: cs.CV**

- **简介: 该论文属于6D物体姿态估计任务，旨在解决真实场景下极端条件下的姿态估计问题。通过构建EgoXtreme数据集，评估现有方法在极端条件下的表现，并探索有效解决方案。**

- **链接: [https://arxiv.org/pdf/2603.25135](https://arxiv.org/pdf/2603.25135)**

> **作者:** Taegyoon Yoon; Yegyu Han; Seojin Ji; Jaewoo Park; Sojeong Kim; Taein Kwon; Hyung-Sin Kim
>
> **备注:** Camera ready version for CVPR 2026, appendix included
>
> **摘要:** Smart glass is emerging as an useful device since it provides plenty of insights under hands-busy, eyes-on-task situations. To understand the context of the wearer, 6D object pose estimation in egocentric view is becoming essential. However, existing 6D object pose estimation benchmarks fail to capture the challenges of real-world egocentric applications, which are often dominated by severe motion blur, dynamic illumination, and visual obstructions. This discrepancy creates a significant gap between controlled lab data and chaotic real-world application. To bridge this gap, we introduce EgoXtreme, a new large-scale 6D pose estimation dataset captured entirely from an egocentric perspective. EgoXtreme features three challenging scenarios - industrial maintenance, sports, and emergency rescue - designed to introduce severe perceptual ambiguities through extreme lighting, heavy motion blur, and smoke. Evaluations of state-of-the-art generalizable pose estimators on EgoXtreme indicate that their generalization fails to hold in extreme conditions, especially under low light. We further demonstrate that simply applying image restoration (e.g., deblurring) offers no positive improvement for extreme conditions. While performance gain has appeared in tracking-based approach, implying using temporal information in fast-motion scenarios is meaningful. We conclude that EgoXtreme is an essential resource for developing and evaluating the next generation of pose estimation models robust enough for real-world egocentric vision. The dataset and code are available at this https URL
>
---
#### [new 151] KitchenTwin: Semantically and Geometrically Grounded 3D Kitchen Digital Twins
- **分类: cs.CV**

- **简介: 该论文属于3D数字孪生任务，解决点云与物体网格融合中的尺度和坐标不一致问题，提出一种尺度感知的融合框架，提升几何一致性。**

- **链接: [https://arxiv.org/pdf/2603.24684](https://arxiv.org/pdf/2603.24684)**

> **作者:** Quanyun Wu; Kyle Gao; Daniel Long; David A. Clausi; Jonathan Li; Yuhao Chen
>
> **摘要:** Embodied AI training and evaluation require object-centric digital twin environments with accurate metric geometry and semantic grounding. Recent transformer-based feedforward reconstruction methods can efficiently predict global point clouds from sparse monocular videos, yet these geometries suffer from inherent scale ambiguity and inconsistent coordinate conventions. This mismatch prevents the reliable fusion of these dimensionless point cloud predictions with locally reconstructed object meshes. We propose a novel scale-aware 3D fusion framework that registers visually grounded object meshes with transformer-predicted global point clouds to construct metrically consistent digital twins. Our method introduces a Vision-Language Model (VLM)-guided geometric anchor mechanism that resolves this fundamental coordinate mismatch by recovering an accurate real-world metric scale. To fuse these networks, we propose a geometry-aware registration pipeline that explicitly enforces physical plausibility through gravity-aligned vertical estimation, Manhattan-world structural constraints, and collision-free local refinement. Experiments on real indoor kitchen environments demonstrate improved cross-network object alignment and geometric consistency for downstream tasks, including multi-primitive fitting and metric measurement. We additionally introduce an open-source indoor digital twin dataset with metrically scaled scenes and semantically grounded and registered object-centric mesh annotations.
>
---
#### [new 152] UniICL: Systematizing Unified Multimodal In-context Learning through a Capability-Oriented Taxonomy
- **分类: cs.CV**

- **简介: 该论文属于多模态少样本学习任务，旨在解决In-context Learning在统一模型中的适应性问题。通过构建评估基准和提出新模块，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.24690](https://arxiv.org/pdf/2603.24690)**

> **作者:** Yicheng Xu; Jiangning Zhang; Zhucun Xue; Teng Hu; Ran Yi; Xiaobin Hu; Yong Liu; Dacheng Tao
>
> **备注:** ECCV2026 under review
>
> **摘要:** In-context Learning enables training-free adaptation via demonstrations but remains highly sensitive to example selection and formatting. In unified multimodal models spanning understanding and generation, this sensitivity is exacerbated by cross-modal interference and varying cognitive demands. Consequently, In-context Learning efficacy is often non-monotonic and highly task-dependent. To diagnose these behaviors, we introduce a six-level capability-oriented taxonomy that categorizes the functional role of demonstrations from basic perception to high-order discernment. Guided by this cognitive framework, we construct UniICL-760K, a large-scale corpus featuring curated 8-shot In-context Learning episodes across 15 subtasks, alongside UniICL-Bench for rigorous, controlled evaluation. As an architectural intervention to stabilize few-shot adaptation, we propose the Context-Adaptive Prototype Modulator, a lightweight, plug-and-play module. Evaluations on UniICL-Bench show that our approach yields highly competitive unified results, outperforming larger-parameter multimodal large language model baselines on most understanding In-context Learning tasks. Data and code will be available soon at this https URL.
>
---
#### [new 153] DRoPS: Dynamic 3D Reconstruction of Pre-Scanned Objects
- **分类: cs.CV**

- **简介: 该论文属于动态3D重建任务，旨在解决从视频中重建动态物体的问题。通过利用预扫描的静态信息，提出DRoPS方法提升重建质量和跟踪精度。**

- **链接: [https://arxiv.org/pdf/2603.24770](https://arxiv.org/pdf/2603.24770)**

> **作者:** Narek Tumanyan; Samuel Rota Bulò; Denis Rozumny; Lorenzo Porzi; Adam Harley; Tali Dekel; Peter Kontschieder; Jonathon Luiten
>
> **备注:** Project page: this https URL
>
> **摘要:** Dynamic scene reconstruction from casual videos has seen recent remarkable progress. Numerous approaches have attempted to overcome the ill-posedness of the task by distilling priors from 2D foundational models and by imposing hand-crafted regularization on the optimized motion. However, these methods struggle to reconstruct scenes from extreme novel viewpoints, especially when highly articulated motions are present. In this paper, we present DRoPS, a novel approach that leverages a static pre-scan of the dynamic object as an explicit geometric and appearance prior. While existing state-of-the-art methods fail to fully exploit the pre-scan, DRoPS leverages our novel setup to effectively constrain the solution space and ensure geometrical consistency throughout the sequence. The core of our novelty is twofold: first, we establish a grid-structured and surface-aligned model by organizing Gaussian primitives into pixel grids anchored to the object surface. Second, by leveraging the grid structure of our primitives, we parameterize motion using a CNN conditioned on those grids, injecting strong implicit regularization and correlating the motion of nearby points. Extensive experiments demonstrate that our method significantly outperforms the current state of the art in rendering quality and 3D tracking accuracy.
>
---
#### [new 154] Attention-based Pin Site Image Classification in Orthopaedic Patients with External Fixators
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决外固定器针孔感染的自动识别问题。通过深度学习方法对针孔图像进行分类，提升感染检测效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.24815](https://arxiv.org/pdf/2603.24815)**

> **作者:** Yubo Wang; Marie Fridberg; Anirejuoritse Bafor; Ole Rahbek; Christopher Iobst; Søren Vedding Kold; Ming Shen
>
> **摘要:** Pin sites represent the interface where a metal pin or wire from the external environment passes through the skin into the internal environment of the limb. These pins or wires connect an external fixator to the bone to stabilize the bone segments in a patient with trauma or deformity. Because these pin sites represent an opportunity for external skin flora to enter the internal environment of the limb, infections of the pin site are common. These pin site infections are painful, annoying, and cause increased morbidity to the patients. Improving the identification and management of pin site infections would greatly enhance the patient experience when external fixators are used. For this, this paper collects and produces a dataset on pin sites wound infections and proposes a deep learning (DL) method to classify pin sites images based on their appearance: Group A displayed signs of inflammation or infection, while Group B showed no evident complications. Unlike studies that primarily focus on open wounds, our research includes potential interventions at the metal pin/skin interface. Our attention-based deep learning model addresses this complexity by emphasizing relevant regions and minimizing distractions from the pins. Moreover, we introduce an Efficient Redundant Reconstruction Convolution (ERRC) method to enhance the richness of feature maps while reducing the number of parameters. Our model outperforms baseline methods with an AUC of 0.975 and an F1-score of 0.927, requiring only 5.77 M parameters. These results highlight the potential of DL in differentiating pin sites only based on visual signs of infection, aligning with healthcare professional assessments, while further validation with more data remains essential.
>
---
#### [new 155] Synthetic Cardiac MRI Image Generation using Deep Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像生成任务，旨在解决标注数据不足和隐私问题。通过深度生成模型生成合成心脏MRI图像，提升数据质量和安全性。**

- **链接: [https://arxiv.org/pdf/2603.24764](https://arxiv.org/pdf/2603.24764)**

> **作者:** Ishan Kumarasinghe; Dasuni Kawya; Madhura Edirisooriya; Isuri Devindi; Isuru Nawinne; Vajira Thambawita
>
> **备注:** 12 pages, 2 figures, Preprint
>
> **摘要:** Synthetic cardiac MRI (CMRI) generation has emerged as a promising strategy to overcome the scarcity of annotated medical imaging data. Recent advances in GANs, VAEs, diffusion probabilistic models, and flow-matching techniques aim to generate anatomically accurate images while addressing challenges such as limited labeled datasets, vendor variability, and risks of privacy leakage through model memorization. Maskconditioned generation improves structural fidelity by guiding synthesis with segmentation maps, while diffusion and flowmatching models offer strong boundary preservation and efficient deterministic transformations. Cross-domain generalization is further supported through vendor-style conditioning and preprocessing steps like intensity normalization. To ensure privacy, studies increasingly incorporate membership inference attacks, nearest-neighbor analyses, and differential privacy mechanisms. Utility evaluations commonly measure downstream segmentation performance, with evidence showing that anatomically constrained synthetic data can enhance accuracy and robustness across multi-vendor settings. This review aims to compare existing CMRI generation approaches through the lenses of fidelity, utility, and privacy, highlighting current limitations and the need for integrated, evaluation-driven frameworks for reliable clinical workflows.
>
---
#### [new 156] BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决混合领域半监督学习中的域偏移和标注不足问题。提出BCMDA框架，通过双向相关图域适应提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.24691](https://arxiv.org/pdf/2603.24691)**

> **作者:** Bentao Song; Jun Huang; Qingfeng Wang
>
> **备注:** Accepted at Neural Networks
>
> **摘要:** In mixed domain semi-supervised medical image segmentation (MiDSS), achieving superior performance under domain shift and limited annotations is challenging. This scenario presents two primary issues: (1) distributional differences between labeled and unlabeled data hinder effective knowledge transfer, and (2) inefficient learning from unlabeled data causes severe confirmation bias. In this paper, we propose the bidirectional correlation maps domain adaptation (BCMDA) framework to overcome these issues. On the one hand, we employ knowledge transfer via virtual domain bridging (KTVDB) to facilitate cross-domain learning. First, to construct a distribution-aligned virtual domain, we leverage bidirectional correlation maps between labeled and unlabeled data to synthesize both labeled and unlabeled images, which are then mixed with the original images to generate virtual images using two strategies, a fixed ratio and a progressive dynamic MixUp. Next, dual bidirectional CutMix is used to enable initial knowledge transfer within the fixed virtual domain and gradual knowledge transfer from the dynamically transitioning labeled domain to the real unlabeled domains. On the other hand, to alleviate confirmation bias, we adopt prototypical alignment and pseudo label correction (PAPLC), which utilizes learnable prototype cosine similarity classifiers for bidirectional prototype alignment between the virtual and real domains, yielding smoother and more compact feature representations. Finally, we use prototypical pseudo label correction to generate more reliable pseudo labels. Empirical evaluations on three public multi-domain datasets demonstrate the superiority of our method, particularly showing excellent performance even with very limited labeled samples. Code available at this https URL.
>
---
#### [new 157] Integrating Deep RL and Bayesian Inference for ObjectNav in Mobile Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于移动机器人目标导航任务，旨在解决部分可观测环境下的物体搜索问题。融合贝叶斯推理与深度强化学习，提升搜索效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25366](https://arxiv.org/pdf/2603.25366)**

> **作者:** João Castelo-Branco; José Santos-Victor; Alexandre Bernardino
>
> **备注:** Accepted and to be published in the ICARSC 2026 26th IEEE International Conference on Autonomous Robot Systems and Competitions
>
> **摘要:** Autonomous object search is challenging for mobile robots operating in indoor environments due to partial observability, perceptual uncertainty, and the need to trade off exploration and navigation efficiency. Classical probabilistic approaches explicitly represent uncertainty but typically rely on handcrafted action-selection heuristics, while deep reinforcement learning enables adaptive policies but often suffers from slow convergence and limited interpretability. This paper proposes a hybrid object-search framework that integrates Bayesian inference with deep reinforcement learning. The method maintains a spatial belief map over target locations, updated online through Bayesian inference from calibrated object detections, and trains a reinforcement learning policy to select navigation actions directly from this probabilistic representation. The approach is evaluated in realistic indoor simulation using Habitat 3.0 and compared against developed baseline strategies. Across two indoor environments, the proposed method improves success rate while reducing search effort. Overall, the results support the value of combining Bayesian belief estimation with learned action selection to achieve more efficient and reliable objectsearch behavior under partial observability.
>
---
#### [new 158] Vision Hopfield Memory Networks
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文提出一种脑启发的视觉基础模型V-HMN，解决现有模型数据效率低、可解释性差的问题，通过整合记忆机制提升性能与生物合理性。**

- **链接: [https://arxiv.org/pdf/2603.25157](https://arxiv.org/pdf/2603.25157)**

> **作者:** Jianfeng Wang; Amine M'Charrak; Luk Koska; Xiangtao Wang; Daniel Petriceanu; Mykyta Smyrnov; Ruizhi Wang; Michael Bumbar; Luca Pinchetti; Thomas Lukasiewicz
>
> **摘要:** Recent vision and multimodal foundation backbones, such as Transformer families and state-space models like Mamba, have achieved remarkable progress, enabling unified modeling across images, text, and beyond. Despite their empirical success, these architectures remain far from the computational principles of the human brain, often demanding enormous amounts of training data while offering limited interpretability. In this work, we propose the Vision Hopfield Memory Network (V-HMN), a brain-inspired foundation backbone that integrates hierarchical memory mechanisms with iterative refinement updates. Specifically, V-HMN incorporates local Hopfield modules that provide associative memory dynamics at the image patch level, global Hopfield modules that function as episodic memory for contextual modulation, and a predictive-coding-inspired refinement rule for iterative error correction. By organizing these memory-based modules hierarchically, V-HMN captures both local and global dynamics in a unified framework. Memory retrieval exposes the relationship between inputs and stored patterns, making decisions more interpretable, while the reuse of stored patterns improves data efficiency. This brain-inspired design therefore enhances interpretability and data efficiency beyond existing self-attention- or state-space-based approaches. We conducted extensive experiments on public computer vision benchmarks, and V-HMN achieved competitive results against widely adopted backbone architectures, while offering better interpretability, higher data efficiency, and stronger biological plausibility. These findings highlight the potential of V-HMN to serve as a next-generation vision foundation model, while also providing a generalizable blueprint for multimodal backbones in domains such as text and audio, thereby bridging brain-inspired computation with large-scale machine learning.
>
---
#### [new 159] Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于教育AI任务，旨在解决手写数学草稿中的错误分析问题。研究构建了ScratchMath数据集，用于错误原因解释与分类，并评估MLLMs的表现。**

- **链接: [https://arxiv.org/pdf/2603.24961](https://arxiv.org/pdf/2603.24961)**

> **作者:** Dingjie Song; Tianlong Xu; Yi-Fan Zhang; Hang Li; Zhiling Yan; Xing Fan; Haoyang Li; Lichao Sun; Qingsong Wen
>
> **备注:** Accepted by the 27th International Conference on Artificial Intelligence in Education (AIED'26)
>
> **摘要:** Assessing student handwritten scratchwork is crucial for personalized educational feedback but presents unique challenges due to diverse handwriting, complex layouts, and varied problem-solving approaches. Existing educational NLP primarily focuses on textual responses and neglects the complexity and multimodality inherent in authentic handwritten scratchwork. Current multimodal large language models (MLLMs) excel at visual reasoning but typically adopt an "examinee perspective", prioritizing generating correct answers rather than diagnosing student errors. To bridge these gaps, we introduce ScratchMath, a novel benchmark specifically designed for explaining and classifying errors in authentic handwritten mathematics scratchwork. Our dataset comprises 1,720 mathematics samples from Chinese primary and middle school students, supporting two key tasks: Error Cause Explanation (ECE) and Error Cause Classification (ECC), with seven defined error types. The dataset is meticulously annotated through rigorous human-machine collaborative approaches involving multiple stages of expert labeling, review, and verification. We systematically evaluate 16 leading MLLMs on ScratchMath, revealing significant performance gaps relative to human experts, especially in visual recognition and logical reasoning. Proprietary models notably outperform open-source models, with large reasoning models showing strong potential for error explanation. All evaluation data and frameworks are publicly available to facilitate further research.
>
---
#### [new 160] Light Cones For Vision: Simple Causal Priors For Visual Hierarchy
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉任务，解决对象层次结构建模问题。提出Worldline Slot Attention，利用洛伦兹几何捕捉因果关系，提升层次结构学习效果。**

- **链接: [https://arxiv.org/pdf/2603.24753](https://arxiv.org/pdf/2603.24753)**

> **作者:** Manglam Kartik; Neel Tushar Shah
>
> **备注:** ICLR GRaM Workshop 2026
>
> **摘要:** Standard vision models treat objects as independent points in Euclidean space, unable to capture hierarchical structure like parts within wholes. We introduce Worldline Slot Attention, which models objects as persistent trajectories through spacetime worldlines, where each object has multiple slots at different hierarchy levels sharing the same spatial position but differing in temporal coordinates. This architecture consistently fails without geometric structure: Euclidean worldlines achieve 0.078 level accuracy, below random chance (0.33), while Lorentzian worldlines achieve 0.479-0.661 across three datasets: a 6x improvement replicated over 20+ independent runs. Lorentzian geometry also outperforms hyperbolic embeddings showing visual hierarchies require causal structure (temporal dependency) rather than tree structure (radial branching). Our results demonstrate that hierarchical object discovery requires geometric structure encoding asymmetric causality, an inductive bias absent from Euclidean space but natural to Lorentzian light cones, achieved with only 11K parameters. The code is available at: this https URL.
>
---
#### [new 161] Can Users Specify Driving Speed? Bench2Drive-Speed: Benchmark and Baselines for Desired-Speed Conditioned Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶任务，解决用户自定义车速和变道指令的问题。通过构建数据集和基准模型，研究如何在不依赖额外数据的情况下实现速度控制与变道执行。**

- **链接: [https://arxiv.org/pdf/2603.25672](https://arxiv.org/pdf/2603.25672)**

> **作者:** Yuqian Shao; Xiaosong Jia; Langechuan Liu; Junchi Yan
>
> **备注:** Project page: this https URL
>
> **摘要:** End-to-end autonomous driving (E2E-AD) has achieved remarkable progress. However, one practical and useful function has been long overlooked: users may wish to customize the desired speed of the policy or specify whether to allow the autonomous vehicle to overtake. To bridge this gap, we present Bench2Drive-Speed, a benchmark with metrics, dataset, and baselines for desired-speed conditioned autonomous driving. We introduce explicit inputs of users' desired target-speed and overtake/follow instructions to driving policy models. We design quantitative metrics, including Speed-Adherence Score and Overtake Score, to measure how faithfully policies follow user specifications, while remaining compatible with standard autonomous driving metrics. To enable training of speed-conditioned policies, one approach is to collect expert demonstrations that strictly follow speed requirements, an expensive and unscalable process in the real world. An alternative is to adapt existing regular driving data by treating the speed observed in future frames as the target speed for training. To investigate this, we construct CustomizedSpeedDataset, composed of 2,100 clips annotated with experts demonstrations, enabling systematic investigation of supervision strategies. Our experiments show that, under proper re-annotation, models trained on regular driving data perform comparably to on expert demonstrations, suggesting that speed supervision can be introduced without additional complex real-world data collection. Furthermore, we find that while target-speed following can be achieved without degrading regular driving performance, executing overtaking commands remains challenging due to the inherent difficulty of interactive behaviors. All code, datasets and baselines are available at this https URL
>
---
#### [new 162] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于个性化自动驾驶任务，旨在解决现有系统无法适应用户驾驶习惯和语言指令的问题。提出DMW框架，通过用户嵌入和自然语言指导实现个性化行为生成。**

- **链接: [https://arxiv.org/pdf/2603.25740](https://arxiv.org/pdf/2603.25740)**

> **作者:** Zehao Wang; Huaide Jiang; Shuaiwu Dong; Yuping Wang; Hang Qiu; Jiachen Li
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026); Project website: this https URL
>
> **摘要:** Human driving behavior is inherently personal, which is shaped by long-term habits and influenced by short-term intentions. Individuals differ in how they accelerate, brake, merge, yield, and overtake across diverse situations. However, existing end-to-end autonomous driving systems either optimize for generic objectives or rely on fixed driving modes, lacking the ability to adapt to individual preferences or interpret natural language intent. To address this gap, we propose Drive My Way (DMW), a personalized Vision-Language-Action (VLA) driving framework that aligns with users' long-term driving habits and adapts to real-time user instructions. DMW learns a user embedding from our personalized driving dataset collected across multiple real drivers and conditions the policy on this embedding during planning, while natural language instructions provide additional short-term guidance. Closed-loop evaluation on the Bench2Drive benchmark demonstrates that DMW improves style instruction adaptation, and user studies show that its generated behaviors are recognizable as each driver's own style, highlighting personalization as a key capability for human-centered autonomous driving. Our data and code are available at this https URL.
>
---
#### [new 163] CVA: Context-aware Video-text Alignment for Video Temporal Grounding
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于视频文本对齐任务，旨在解决视频时间定位中的背景干扰问题。通过引入数据增强、边界损失和增强编码器，提升模型对无关背景的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.24934](https://arxiv.org/pdf/2603.24934)**

> **作者:** Sungho Moon; Seunghun Lee; Jiwan Seo; Sunghoon Im
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We propose Context-aware Video-text Alignment (CVA), a novel framework to address a significant challenge in video temporal grounding: achieving temporally sensitive video-text alignment that remains robust to irrelevant background context. Our framework is built on three key components. First, we propose Query-aware Context Diversification (QCD), a new data augmentation strategy that ensures only semantically unrelated content is mixed in. It builds a video-text similarity-based pool of replacement clips to simulate diverse contexts while preventing the ``false negative" caused by query-agnostic mixing. Second, we introduce the Context-invariant Boundary Discrimination (CBD) loss, a contrastive loss that enforces semantic consistency at challenging temporal boundaries, making their representations robust to contextual shifts and hard negatives. Third, we introduce the Context-enhanced Transformer Encoder (CTE), a hierarchical architecture that combines windowed self-attention and bidirectional cross-attention with learnable queries to capture multi-scale temporal context. Through the synergy of these data-centric and architectural enhancements, CVA achieves state-of-the-art performance on major VTG benchmarks, including QVHighlights and Charades-STA. Notably, our method achieves a significant improvement of approximately 5 points in Recall@1 (R1) scores over state-of-the-art methods, highlighting its effectiveness in mitigating false negatives.
>
---
#### [new 164] How Far Are Vision-Language Models from Constructing the Real World? A Benchmark for Physical Generative Reasoning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态智能任务，旨在解决VLMs在物理构建推理上的不足。提出DreamHouse基准，评估模型在满足结构、施工等物理约束下的生成能力。**

- **链接: [https://arxiv.org/pdf/2603.24866](https://arxiv.org/pdf/2603.24866)**

> **作者:** Luyu Yang; Yutong Dai; An Yan; Viraj Prabhu; Ran Xu; Zeyuan Chen
>
> **摘要:** The physical world is not merely visual; it is governed by rigorous structural and procedural constraints. Yet, the evaluation of vision-language models (VLMs) remains heavily skewed toward perceptual realism, prioritizing the generation of visually plausible 3D layouts, shapes, and appearances. Current benchmarks rarely test whether models grasp the step-by-step processes and physical dependencies required to actually build these artifacts, a capability essential for automating design-to-construction pipelines. To address this, we introduce DreamHouse, a novel benchmark for physical generative reasoning: the capacity to synthesize artifacts that concurrently satisfy geometric, structural, constructability, and code-compliance constraints. We ground this benchmark in residential timber-frame construction, a domain with fully codified engineering standards and objectively verifiable correctness. We curate over 26,000 structures spanning 13 architectural styles, ach verified to construction-document standards (LOD 350) and develop a deterministic 10-test structural validation framework. Unlike static benchmarks that assess only final outputs, DreamHouse supports iterative agentic interaction. Models observe intermediate build states, generate construction actions, and receive structured environmental feedback, enabling a fine-grained evaluation of planning, structural reasoning, and self-correction. Extensive experiments with state-of-the-art VLMs reveal substantial capability gaps that are largely invisible on existing leaderboards. These findings establish physical validity as a critical evaluation axis orthogonal to visual realism, highlighting physical generative reasoning as a distinct and underdeveloped frontier in multimodal intelligence. Available at this https URL
>
---
#### [new 165] Gaze patterns predict preference and confidence in pairwise AI image evaluation
- **分类: cs.HC; cs.AI; cs.CV; cs.CY**

- **简介: 该论文属于偏好学习任务，研究眼动模式如何预测人类在对比AI生成图像时的偏好与信心。通过眼动实验，发现 gaze 特征能有效预测选择和决策信心。**

- **链接: [https://arxiv.org/pdf/2603.24849](https://arxiv.org/pdf/2603.24849)**

> **作者:** Nikolas Papadopoulos; Shreenithi Navaneethan; Sheng Bai; Ankur Samanta; Paul Sajda
>
> **备注:** This paper has been accepted to ACM ETRA 2026
>
> **摘要:** Preference learning methods, such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on pairwise human judgments, yet little is known about the cognitive processes underlying these judgments. We investigate whether eye-tracking can reveal preference formation during pairwise AI-generated image evaluation. Thirty participants completed 1,800 trials while their gaze was recorded. We replicated the gaze cascade effect, with gaze shifting toward chosen images approximately one second before the decision. Cascade dynamics were consistent across confidence levels. Gaze features predicted binary choice (68% accuracy), with chosen images receiving more dwell time, fixations, and revisits. Gaze transitions distinguished high-confidence from uncertain decisions (66% accuracy), with low-confidence trials showing more image switches per second. These results show that gaze patterns predict both choice and confidence in pairwise image evaluations, suggesting that eye-tracking provides implicit signals relevant to the quality of preference annotations.
>
---
#### [new 166] R-C2: Cycle-Consistent Reinforcement Learning Improves Multimodal Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决模型在不同感官模态间不一致的问题。通过引入RC2框架，利用循环一致性提升模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2603.25720](https://arxiv.org/pdf/2603.25720)**

> **作者:** Zirui Zhang; Haoyu Dong; Kexin Pei; Chengzhi Mao
>
> **摘要:** Robust perception and reasoning require consistency across sensory modalities. Yet current multimodal models often violate this principle, yielding contradictory predictions for visual and textual representations of the same concept. Rather than masking these failures with standard voting mechanisms, which can amplify systematic biases, we show that cross-modal inconsistency provides a rich and natural signal for learning. We introduce RC2, a reinforcement learning framework that resolves internal conflicts by enforcing cross-modal cycle consistency. By requiring a model to perform backward inference, switch modalities, and reliably reconstruct the answer through forward inference, we obtain a dense, label-free reward. This cyclic constraint encourages the model to align its internal representations autonomously. Optimizing for this structure mitigates modality-specific errors and improves reasoning accuracy by up to 7.6 points. Our results suggest that advanced reasoning emerges not only from scaling data, but also from enforcing a structurally consistent understanding of the world.
>
---
#### [new 167] Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决预训练VLA模型在微调中性能提升有限且成本高的问题。通过解耦辅助任务目标，提出一种高效微调方法，提升性能并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2603.25661](https://arxiv.org/pdf/2603.25661)**

> **作者:** Wenxuan Song; Jiayi Chen; Shuai Chen; Jingbo Wang; Pengxiang Ding; Han Zhao; Yikai Qin; Xinhu Zheng; Donglin Wang; Yan Wang; Haoang Li
>
> **摘要:** This paper proposes a novel approach to address the challenge that pretrained VLA models often fail to effectively improve performance and reduce adaptation costs during standard supervised finetuning (SFT). Some advanced finetuning methods with auxiliary training objectives can improve performance and reduce the number of convergence steps. However, they typically incur significant computational overhead due to the additional losses from auxiliary tasks. To simultaneously achieve the enhanced capabilities of auxiliary training with the simplicity of standard SFT, we decouple the two objectives of auxiliary task training within the parameter space, namely, enhancing general capabilities and fitting task-specific action distributions. To deliver this goal, we only need to train the model to converge on a small-scale task set using two distinct training strategies. The difference between the resulting model parameters can then be interpreted as capability vectors provided by auxiliary tasks. These vectors are then merged with pretrained parameters to form a capability-enhanced meta model. Moreover, when standard SFT is augmented with a lightweight orthogonal regularization loss, the merged model attains performance comparable to auxiliary finetuned baselines with reduced computational overhead. Experimental results demonstrate that this approach is highly effective across diverse robot tasks. Project page: this https URL
>
---
#### [new 168] Amplified Patch-Level Differential Privacy for Free via Random Cropping
- **分类: cs.LG; cs.CR; cs.CV**

- **简介: 该论文属于隐私保护机器学习任务，解决如何在不改变模型结构的情况下增强差分隐私的问题。通过随机裁剪引入额外随机性，提升隐私保障。**

- **链接: [https://arxiv.org/pdf/2603.24695](https://arxiv.org/pdf/2603.24695)**

> **作者:** Kaan Durmaz; Jan Schuchardt; Sebastian Schmidt; Stephan Günnemann
>
> **备注:** Published at TMLR
>
> **摘要:** Random cropping is one of the most common data augmentation techniques in computer vision, yet the role of its inherent randomness in training differentially private machine learning models has thus far gone unexplored. We observe that when sensitive content in an image is spatially localized, such as a face or license plate, random cropping can probabilistically exclude that content from the model's input. This introduces a third source of stochasticity in differentially private training with stochastic gradient descent, in addition to gradient noise and minibatch sampling. This additional randomness amplifies differential privacy without requiring changes to model architecture or training procedure. We formalize this effect by introducing a patch-level neighboring relation for vision data and deriving tight privacy bounds for differentially private stochastic gradient descent (DP-SGD) when combined with random cropping. Our analysis quantifies the patch inclusion probability and shows how it composes with minibatch sampling to yield a lower effective sampling rate. Empirically, we validate that patch-level amplification improves the privacy-utility trade-off across multiple segmentation architectures and datasets. Our results demonstrate that aligning privacy accounting with domain structure and additional existing sources of randomness can yield stronger guarantees at no additional cost.
>
---
#### [new 169] Colon-Bench: An Agentic Workflow for Scalable Dense Lesion Annotation in Full-Procedure Colonoscopy Videos
- **分类: eess.IV; cs.CV; cs.HC**

- **简介: 该论文提出Colon-Bench，解决结肠镜视频中密集病灶标注难题，通过多阶段智能流程生成大规模标注数据，用于评估多模态大语言模型在医学影像任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.25645](https://arxiv.org/pdf/2603.25645)**

> **作者:** Abdullah Hamdi; Changchun Yang; Xin Gao
>
> **备注:** preprint
>
> **摘要:** Early screening via colonoscopy is critical for colon cancer prevention, yet developing robust AI systems for this domain is hindered by the lack of densely annotated, long-sequence video datasets. Existing datasets predominantly focus on single-class polyp detection and lack the rich spatial, temporal, and linguistic annotations required to evaluate modern Multimodal Large Language Models (MLLMs). To address this critical gap, we introduce Colon-Bench, generated via a novel multi-stage agentic workflow. Our pipeline seamlessly integrates temporal proposals, bounding-box tracking, AI-driven visual confirmation, and human-in-the-loop review to scalably annotate full-procedure videos. The resulting verified benchmark is unprecedented in scope, encompassing 528 videos, 14 distinct lesion categories (including polyps, ulcers, and bleeding), over 300,000 bounding boxes, 213,000 segmentation masks, and 133,000 words of clinical descriptions. We utilize Colon-Bench to rigorously evaluate state-of-the-art MLLMs across lesion classification, Open-Vocabulary Video Object Segmentation (OV-VOS), and video Visual Question Answering (VQA). The MLLM results demonstrate surprisingly high localization performance in medical domains compared to SAM-3. Finally, we analyze common VQA errors from MLLMs to introduce a novel "colon-skill" prompting strategy, improving zero-shot MLLM performance by up to 9.7% across most MLLMs. The dataset and the code are available at this https URL .
>
---
#### [new 170] Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉预测任务，解决多步预测中误差累积导致的视觉质量下降问题。通过强化学习优化模型，提升长期预测的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.25685](https://arxiv.org/pdf/2603.25685)**

> **作者:** Jai Bardhan; Patrik Drozdik; Josef Sivic; Vladimir Petrik
>
> **备注:** 34 pages, 11 figures, 12 tables
>
> **摘要:** Action-conditioned robot world models generate future video frames of the manipulated scene given a robot action sequence, offering a promising alternative for simulating tasks that are difficult to model with traditional physics engines. However, these models are optimized for short-term prediction and break down when deployed autoregressively: each predicted clip feeds back as context for the next, causing errors to compound and visual quality to rapidly degrade. We address this through the following contributions. First, we introduce a reinforcement learning (RL) post-training scheme that trains the world model on its own autoregressive rollouts rather than on ground-truth histories. We achieve this by adapting a recent contrastive RL objective for diffusion models to our setting and show that its convergence guarantees carry over exactly. Second, we design a training protocol that generates and compares multiple candidate variable-length futures from the same rollout state, reinforcing higher-fidelity predictions over lower-fidelity ones. Third, we develop efficient, multi-view visual fidelity rewards that combine complementary perceptual metrics across camera views and are aggregated at the clip level for dense, low-variance training signal. Fourth, we show that our approach establishes a new state-of-the-art for rollout fidelity on the DROID dataset, outperforming the strongest baseline on all metrics (e.g., LPIPS reduced by 14% on external cameras, SSIM improved by 9.1% on the wrist camera), winning 98% of paired comparisons, and achieving an 80% preference rate in a blind human study.
>
---
#### [new 171] AI Security in the Foundation Model Era: A Comprehensive Survey from a Unified Perspective
- **分类: cs.CR; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI安全领域，旨在解决基础模型中数据与模型间的安全威胁问题。提出统一的闭环威胁分类框架，分析四种方向的安全攻击类型。**

- **链接: [https://arxiv.org/pdf/2603.24857](https://arxiv.org/pdf/2603.24857)**

> **作者:** Zhenyi Wang; Siyu Luan
>
> **备注:** Published at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** As machine learning (ML) systems expand in both scale and functionality, the security landscape has become increasingly complex, with a proliferation of attacks and defenses. However, existing studies largely treat these threats in isolation, lacking a coherent framework to expose their shared principles and interdependencies. This fragmented view hinders systematic understanding and limits the design of comprehensive defenses. Crucially, the two foundational assets of ML -- \textbf{data} and \textbf{models} -- are no longer independent; vulnerabilities in one directly compromise the other. The absence of a holistic framework leaves open questions about how these bidirectional risks propagate across the ML pipeline. To address this critical gap, we propose a \emph{unified closed-loop threat taxonomy} that explicitly frames model-data interactions along four directional axes. Our framework offers a principled lens for analyzing and defending foundation models. The resulting four classes of security threats represent distinct but interrelated categories of attacks: (1) Data$\rightarrow$Data (D$\rightarrow$D): including \emph{data decryption attacks and watermark removal attacks}; (2) Data$\rightarrow$Model (D$\rightarrow$M): including \emph{poisoning, harmful fine-tuning attacks, and jailbreak attacks}; (3) Model$\rightarrow$Data (M$\rightarrow$D): including \emph{model inversion, membership inference attacks, and training data extraction attacks}; (4) Model$\rightarrow$Model (M$\rightarrow$M): including \emph{model extraction attacks}. Our unified framework elucidates the underlying connections among these security threats and establishes a foundation for developing scalable, transferable, and cross-modal security strategies, particularly within the landscape of foundation models.
>
---
#### [new 172] Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文介绍Intern-S1-Pro，一个万亿参数的科学多模态基础模型，解决通用与科学领域任务。通过大规模训练和高效框架，提升推理与专业任务能力。**

- **链接: [https://arxiv.org/pdf/2603.25040](https://arxiv.org/pdf/2603.25040)**

> **作者:** Yicheng Zou; Dongsheng Zhu; Lin Zhu; Tong Zhu; Yunhua Zhou; Peiheng Zhou; Xinyu Zhou; Dongzhan Zhou; Zhiwang Zhou; Yuhao Zhou; Bowen Zhou; Zhanping Zhong; Zhijie Zhong; Haiteng Zhao; Penghao Zhao; Xiaomeng Zhao; Zhiyuan Zhao; Yechen Zhang; Jin Zhang; Wenwei Zhang; Hongjie Zhang; Zhuo Zhang; Wenlong Zhang; Bo Zhang; Chao Zhang; Chen Zhang; Yuhang Zang; Fei Yuan; Jiakang Yuan; Jiashuo Yu; Jinhui Yin; Haochen Ye; Qian Yao; Bowen Yang; Danni Yang; Kaichen Yang; Ziang Yan; Jun Xu; Yicheng Xu; Wanghan Xu; Xuenan Xu; Chao Xu; Ruiliang Xu; Shuhao Xing; Long Xing; Xinchen Xie; Ling-I Wu; Zijian Wu; Zhenyu Wu; Lijun Wu; Yue Wu; Jianyu Wu; Wen Wu; Fan Wu; Xilin Wei; Qi Wei; Bingli Wang; Rui Wang; Ziyi Wang; Zun Wang; Yi Wang; Haomin Wang; Yizhou Wang; Lintao Wang; Yiheng Wang; Longjiang Wang; Bin Wang; Jian Tong; Zhongbo Tian; Huanze Tang; Chen Tang; Shixiang Tang; Yu Sun; Qiushi Sun; Xuerui Su; Qisheng Su; Chenlin Su; Demin Song; Jin Shi; Fukai Shang; Yuchen Ren; Pengli Ren; Xiaoye Qu; Yuan Qu; Jiantao Qiu; Yu Qiao; Runyu Peng; Tianshuo Peng; Jiahui Peng; Qizhi Pei; Zhuoshi Pan; Linke Ouyang; Wenchang Ning; Yichuan Ma; Zerun Ma; Ningsheng Ma; Runyuan Ma; Chengqi Lyu; Haijun Lv; Han Lv
>
> **摘要:** We introduce Intern-S1-Pro, the first one-trillion-parameter scientific multimodal foundation model. Scaling to this unprecedented size, the model delivers a comprehensive enhancement across both general and scientific domains. Beyond stronger reasoning and image-text understanding capabilities, its intelligence is augmented with advanced agent capabilities. Simultaneously, its scientific expertise has been vastly expanded to master over 100 specialized tasks across critical science fields, including chemistry, materials, life sciences, and earth sciences. Achieving this massive scale is made possible by the robust infrastructure support of XTuner and LMDeploy, which facilitates highly efficient Reinforcement Learning (RL) training at the 1-trillion parameter level while ensuring strict precision consistency between training and inference. By seamlessly integrating these advancements, Intern-S1-Pro further fortifies the fusion of general and specialized intelligence, working as a Specializable Generalist, demonstrating its position in the top tier of open-source models for general capabilities, while outperforming proprietary models in the depth of specialized scientific tasks.
>
---
## 更新

#### [replaced 001] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15186](https://arxiv.org/pdf/2511.15186)**

> **作者:** Geon Choi; Hangyul Yoon; Hyunju Shin; Hyunki Park; Sang Hoon Seo; Eunho Yang; Edward Choi
>
> **备注:** Camera-ready version for CVPR 2026
>
> **摘要:** The applicability of current lesion segmentation models for chest X-rays (CXRs) has been limited both by a small number of target labels and the reliance on complex, expert-level text inputs, creating a barrier to practical use. To address these limitations, we introduce instruction-guided lesion segmentation (ILS), a medical-domain adaptation of referring image segmentation (RIS) designed to segment diverse lesion types based on simple, user-friendly instructions. Under this task, we construct MIMIC-ILS, the first large-scale instruction-answer dataset for CXR lesion segmentation, using our fully automated multimodal pipeline that generates annotations from CXR images and their corresponding reports. MIMIC-ILS contains 1.1M instruction-answer pairs derived from 192K images and 91K unique segmentation masks, covering seven major lesion types. To empirically demonstrate its utility, we present ROSALIA, a LISA model fine-tuned on the MIMIC-ILS dataset. ROSALIA can segment diverse lesions and provide textual explanations in response to user instructions. The model achieves high accuracy in our newly proposed task, highlighting the effectiveness of our pipeline and the value of MIMIC-ILS as a foundational resource for pixel-level CXR lesion grounding. The dataset and model are available at this https URL.
>
---
#### [replaced 002] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19682](https://arxiv.org/pdf/2603.19682)**

> **作者:** Takeshi Noda; Yu-Shen Liu; Zhizhong Han
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Rendering 3D surfaces has been revolutionized within the modeling of radiance fields through either 3DGS or NeRF. Although 3DGS has shown advantages over NeRF in terms of rendering quality or speed, there is still room for improvement in recovering high fidelity surfaces through 3DGS. To resolve this issue, we propose a self-constrained prior to constrain the learning of 3D Gaussians, aiming for more accurate depth rendering. Our self-constrained prior is derived from a TSDF grid that is obtained by fusing the depth maps rendered with current 3D Gaussians. The prior measures a distance field around the estimated surface, offering a band centered at the surface for imposing more specific constraints on 3D Gaussians, such as removing Gaussians outside the band, moving Gaussians closer to the surface, and encouraging larger or smaller opacity in a geometry-aware manner. More importantly, our prior can be regularly updated by the most recent depth images which are usually more accurate and complete. In addition, the prior can also progressively narrow the band to tighten the imposed constraints. We justify our idea and report our superiority over the state-of-the-art methods in evaluations on widely used benchmarks.
>
---
#### [replaced 003] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20525](https://arxiv.org/pdf/2511.20525)**

> **作者:** Yayuan Li; Aadit Jain; Filippos Bellos; Jason J. Corso
>
> **备注:** 12 pages, 5 figures, 7 tables. Accepted to CVPR 2026
>
> **摘要:** We introduce Mistake Attribution (MATT), a new task for fine-grained understanding of human mistakes in egocentric videos. While prior work detects whether a mistake occurs, MATT attributes the mistake to what part of the instruction is violated (semantic role), when in the video the deviation becomes irreversible (the Point-of-No-Return, PNR), and where the mistake appears in the PNR frame. We develop MisEngine, a data engine that automatically constructs mistake samples from existing datasets with attribution-rich annotations. Applied to large egocentric corpora, MisEngine yields EPIC-KITCHENS-M and Ego4D-M -- two datasets up to two orders of magnitude larger than prior mistake datasets. We then present MisFormer, a unified attention-based model for mistake attribution across semantic, temporal, and spatial dimensions, trained with MisEngine supervision. A human study demonstrates the ecological validity of our MisEngine-constructed mistake samples, confirming that EPIC-KITCHENS-M and Ego4D-M can serve as reliable benchmarks for mistake understanding. Experiments on both our datasets and prior benchmarks show that MisFormer, as a single unified model, outperforms task-specific SOTA methods by at least 6.66%, 21.81%, 18.7%, and 3.00% in video-language understanding, temporal localization, hand-object interaction, and mistake detection, respectively. Project page: this https URL
>
---
#### [replaced 004] ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13303](https://arxiv.org/pdf/2512.13303)**

> **作者:** Zhihang Liu; Xiaoyi Bao; Pandeng Li; Junjie Zhou; Zhaohe Liao; Yefei He; Kaixun Jiang; Chen-Wei Xie; Yun Zheng; Hongtao Xie
>
> **备注:** Accepted to CVPR 2026, project page: this https URL
>
> **摘要:** While existing generation and unified models excel at general image generation, they struggle with tasks requiring deep reasoning, planning, and precise data-to-visual mapping abilities beyond general scenarios. To push beyond the existing limitations, we introduce a new and challenging task: creative table visualization, requiring the model to generate an infographic that faithfully and aesthetically visualizes the data from a given table. To address this challenge, we propose ShowTable, a pipeline that synergizes MLLMs with diffusion models via a progressive self-correcting process. The MLLM acts as the central orchestrator for reasoning the visual plan and judging visual errors to provide refined instructions, the diffusion execute the commands from MLLM, achieving high-fidelity results. To support this task and our pipeline, we introduce three automated data construction pipelines for training different modules. Furthermore, we introduce TableVisBench, a new benchmark with 800 challenging instances across 5 evaluation dimensions, to assess performance on this task. Experiments demonstrate that our pipeline, instantiated with different models, significantly outperforms baselines, highlighting its effective multi-modal reasoning, generation, and error correction capabilities.
>
---
#### [replaced 005] MOGeo: Beyond One-to-One Cross-View Object Geo-localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13843](https://arxiv.org/pdf/2603.13843)**

> **作者:** Bo Lv; Qingwang Zhang; Le Wu; Yuanyuan Li; Yingying Zhu
>
> **摘要:** Cross-View Object Geo-Localization (CVOGL) aims to locate an object of interest in a query image within a corresponding satellite image. Existing methods typically assume that the query image contains only a single object, which does not align with the complex, multi-object geo-localization requirements in real-world applications, making them unsuitable for practical scenarios. To bridge the gap between the realistic setting and existing task, we propose a new task, called Cross-View Multi-Object Geo-Localization (CVMOGL). To advance the CVMOGL task, we first construct a benchmark, CMLocation, which includes two datasets: CMLocation-V1 and CMLocation-V2. Furthermore, we propose a novel cross-view multi-object geo-localization method, MOGeo, and benchmark it against existing state-of-the-art methods. Extensive experiments are conducted under various application scenarios to validate the effectiveness of our method. The results demonstrate that cross-view object geo-localization in the more realistic setting remains a challenging problem, encouraging further research in this area.
>
---
#### [replaced 006] GenMask: Adapting DiT for Segmentation via Direct Mask Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23906](https://arxiv.org/pdf/2603.23906)**

> **作者:** Yuhuan Yang; Xianwei Zhuang; Yuxuan Cai; Chaofan Ma; Shuai Bai; Jiangchao Yao; Ya Zhang; Junyang Lin; Yanfeng Wang
>
> **备注:** Accepted by cvpr 2026
>
> **摘要:** Recent approaches for segmentation have leveraged pretrained generative models as feature extractors, treating segmentation as a downstream adaptation task via indirect feature retrieval. This implicit use suffers from a fundamental misalignment in representation. It also depends heavily on indirect feature extraction pipelines, which complicate the workflow and limit adaptation. In this paper, we argue that instead of indirect adaptation, segmentation tasks should be trained directly in a generative manner. We identify a key obstacle to this unified formulation: VAE latents of binary masks are sharply distributed, noise robust, and linearly separable, distinct from natural image latents. To bridge this gap, we introduce timesteps sampling strategy for binary masks that emphasizes extreme noise levels for segmentation and moderate noise for image generation, enabling harmonious joint training. We present GenMask, a DiT trains to generate black-and-white segmentation masks as well as colorful images in RGB space under the original generative objective. GenMask preserves the original DiT architecture while removing the need of feature extraction pipelines tailored for segmentation tasks. Empirically, GenMask attains state-of-the-art performance on referring and reasoning segmentation benchmarks and ablations quantify the contribution of each component.
>
---
#### [replaced 007] MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09270](https://arxiv.org/pdf/2512.09270)**

> **作者:** Sangwoon Kwak; Weeyoung Kwon; Jun Young Jeong; Geonho Kim; Won-Sik Cheong; Jihyong Oh
>
> **备注:** CVPR 2026 (camera ready ver.). The first two authors contributed equally to this work (equal contribution). Please visit our project page at this https URL
>
> **摘要:** Recent advances in 4D Gaussian Splatting (4DGS) have extended the high-speed rendering capability of 3D Gaussian Splatting (3DGS) into the temporal domain, enabling real-time rendering of dynamic scenes. However, one of the major remaining challenges lies in modeling long-range motion-contained dynamic videos, where a naive extension of existing methods leads to severe memory explosion, temporal flickering, and failure to handle appearing or disappearing occlusions over time. To address these challenges, we propose a novel 4DGS framework characterized by an Anchor Relay-based Bidirectional Blending (ARBB) mechanism, named MoRel, which enables temporally consistent and memory-efficient modeling of long-range dynamic scenes. Our method progressively constructs locally canonical anchor spaces at key-frame time index and models inter-frame deformations at the anchor level, enhancing temporal coherence. By learning bidirectional deformations between KfA and adaptively blending them through learnable opacity control, our approach mitigates temporal discontinuities and flickering artifacts. We further introduce a Feature-variance-guided Hierarchical Densification (FHD) scheme that effectively densifies KfA's while keeping rendering quality, based on an assigned level of feature-variance. To effectively evaluate our model's capability to handle real-world long-range 4D motion, we newly compose long-range 4D motion-contained dataset, called SelfCap$_{\text{LR}}$. It has larger average dynamic motion magnitude, captured at spatially wider spaces, compared to previous dynamic video datasets. Overall, our MoRel achieves temporally coherent and flicker-free long-range 4D reconstruction while maintaining bounded memory usage, demonstrating both scalability and efficiency in dynamic Gaussian-based representations.
>
---
#### [replaced 008] Revealing Human Attention Patterns from Gameplay Analysis for Reinforcement Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.11118](https://arxiv.org/pdf/2504.11118)**

> **作者:** Henrik Krauss; Takehisa Yairi
>
> **摘要:** This study introduces a novel method for revealing human internal attention patterns (decision-relevant attention) from gameplay data alone, leveraging offline attention techniques from reinforcement learning (RL). We propose contextualized, task-relevant (CTR) attention networks, which generate attention maps from both human and RL agent gameplay in Atari environments. To evaluate whether the human CTR maps reveal internal attention patterns, we validate our model by quantitative and qualitative comparison to the agent maps as well as to a temporally integrated overt attention (TIOA) model based on human eye-tracking data. Our results show that human CTR maps are more sparse than the agent ones and align better with the TIOA maps. Following a qualitative visual comparison we conclude that they likely capture patterns of internal attention. As a further application, we use these maps to guide RL agents, finding that human attention-guided agents achieve slightly improved and more stable learning compared to baselines, and significantly outperform TIOA-based agents. This work advances the understanding of human-agent attention differences and provides a new approach for extracting and validating internal attention patterns from behavioral data.
>
---
#### [replaced 009] DiP: Taming Diffusion Models in Pixel Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18822](https://arxiv.org/pdf/2511.18822)**

> **作者:** Zhennan Chen; Junwei Zhu; Xu Chen; Jiangning Zhang; Xiaobin Hu; Hanzhen Zhao; Chengjie Wang; Jian Yang; Ying Tai
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Diffusion models face a fundamental trade-off between generation quality and computational efficiency. Latent Diffusion Models (LDMs) offer an efficient solution but suffer from potential information loss and non-end-to-end training. In contrast, existing pixel space models bypass VAEs but are computationally prohibitive for high-resolution synthesis. To resolve this dilemma, we propose DiP, an efficient pixel space diffusion framework. DiP decouples generation into a global and a local stage: a Diffusion Transformer (DiT) backbone operates on large patches for efficient global structure construction, while a co-trained lightweight Patch Detailer Head leverages contextual features to restore fine-grained local details. This synergistic design achieves computational efficiency comparable to LDMs without relying on a VAE. DiP is accomplished with up to 10$\times$ faster inference speeds than previous method while increasing the total number of parameters by only 0.3%, and achieves an 1.79 FID score on ImageNet 256$\times$256.
>
---
#### [replaced 010] Seeking Physics in Diffusion Noise
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究视频扩散模型是否包含物理合理性信号，提出方法在推理时通过物理验证器优化轨迹选择，提升物理一致性并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.14294](https://arxiv.org/pdf/2603.14294)**

> **作者:** Chujun Tang; Lei Zhong; Fangqiang Ding
>
> **备注:** 32 pages, 8 figures, 10 tables
>
> **摘要:** Do video diffusion models encode signals predictive of physical plausibility? We probe intermediate denoising representations of a pretrained Diffusion Transformer (DiT) and find that physically plausible and implausible videos are partially separable in mid-layer feature space across noise levels. This separability cannot be fully attributed to visual quality or generator identity, suggesting recoverable physics-related cues in frozen DiT features. Leveraging this observation, we introduce progressive trajectory selection, an inference-time strategy that scores parallel denoising trajectories at a few intermediate checkpoints using a lightweight physics verifier trained on frozen features, and prunes low-scoring candidates early. Extensive experiments on PhyGenBench demonstrate that our method improves physical consistency while reducing inference cost, achieving comparable results to Best-of-K sampling with substantially fewer denoising steps.
>
---
#### [replaced 011] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23022](https://arxiv.org/pdf/2602.23022)**

> **作者:** Xinglong Luo; Ao Luo; Zhengning Wang; Yueqi Yang; Chaoyu Feng; Lei Lei; Bing Zeng; Shuaicheng Liu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at this https URL.
>
---
#### [replaced 012] Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23324](https://arxiv.org/pdf/2603.23324)**

> **作者:** Chuanqing Zhuang; Xin Lu; Zehui Deng; Zhengda Lu; Yiqun Wang; Junqi Diao; Jun Xiao
>
> **摘要:** Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at this https URL.
>
---
#### [replaced 013] VOLMO: Versatile and Open Large Models for Ophthalmology
- **分类: cs.CV; cs.ET**

- **链接: [https://arxiv.org/pdf/2603.23953](https://arxiv.org/pdf/2603.23953)**

> **作者:** Zhenyue Qin; Younjoon Chung; Elijah Lee; Wanyue Feng; Xuguang Ai; Serina Applebaum; Minjie Zou; Yang Liu; Pan Xiao; Mac Singer; Amisha Dave; Aidan Gilson; Tiarnan D. L. Keenan; Emily Y. Chew; Zhiyong Lu; Yih-Chung Tham; Ron Adelman; Luciano V. Del Priore; Qingyu Chen
>
> **摘要:** Vision impairment affects millions globally, and early detection is critical to preventing irreversible vision loss. Ophthalmology workflows require clinicians to integrate medical images, structured clinical data, and free-text notes to determine disease severity and management, which is time-consuming and burdensome. Recent multimodal large language models (MLLMs) show promise, but existing general and medical MLLMs perform poorly in ophthalmology, and few ophthalmology-specific MLLMs are openly available. We present VOLMO (Versatile and Open Large Models for Ophthalmology), a model-agnostic, data-open framework for developing ophthalmology-specific MLLMs. VOLMO includes three stages: ophthalmology knowledge pretraining on 86,965 image-text pairs from 26,569 articles across 82 journals; domain task fine-tuning on 26,929 annotated instances spanning 12 eye conditions for disease screening and severity classification; and multi-step clinical reasoning on 913 patient case reports for assessment, planning, and follow-up care. Using this framework, we trained a compact 2B-parameter MLLM and compared it with strong baselines, including InternVL-2B, LLaVA-Med-7B, MedGemma-4B, MedGemma-27B, and RETFound. We evaluated these models on image description generation, disease screening and staging classification, and assessment-and-management generation, with additional manual review by two healthcare professionals and external validation on three independent cohorts for age-related macular degeneration and diabetic retinopathy. Across settings, VOLMO-2B consistently outperformed baselines, achieving stronger image description performance, an average F1 of 87.4% across 12 eye conditions, and higher scores in external validation.
>
---
#### [replaced 014] Corruption-Aware Training of Latent Video Diffusion Models for Robust Text-to-Video Generation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.21545](https://arxiv.org/pdf/2505.21545)**

> **作者:** Chika Maduabuchi; Hao Chen; Yujin Han; Jindong Wang
>
> **备注:** ICLR 2026 ReALM-GEN
>
> **摘要:** Latent Video Diffusion Models (LVDMs) have achieved state-of-the-art generative quality for image and video generation; however, they remain brittle under noisy conditioning, where small perturbations in text or multimodal embeddings can cascade over timesteps and cause semantic drift. Existing corruption strategies from image diffusion (Gaussian, Uniform) fail in video settings because static noise disrupts temporal fidelity. In this paper, we propose CAT-LVDM, a corruption-aware training framework with structured, data-aligned noise injection tailored for video diffusion. Our two operators, Batch-Centered Noise Injection (BCNI) and Spectrum-Aware Contextual Noise (SACN), align perturbations with batch semantics or spectral dynamics to preserve coherence. CAT-LVDM yields substantial gains: BCNI reduces FVD by 31.9 percent on WebVid-2M, MSR-VTT, and MSVD, while SACN improves UCF-101 by 12.3 percent, outperforming Gaussian, Uniform, and even large diffusion baselines like DEMO (2.3B) and Lavie (3B) despite training on 5x less data. Ablations confirm the unique value of low-rank, data-aligned noise, and theory establishes why these operators tighten robustness and generalization bounds. CAT-LVDM thus sets a new framework for robust video diffusion, and our experiments show that it can also be extended to autoregressive generation and multimodal video understanding LLMs. Code, models, and samples are available at this https URL
>
---
#### [replaced 015] Embedding Compression via Spherical Coordinates
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.00079](https://arxiv.org/pdf/2602.00079)**

> **作者:** Han Xiao
>
> **备注:** Accepted at ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM). 13 pages, 2 figures. Code: this https URL
>
> **摘要:** We present an $\epsilon$-bounded compression method for unit-norm embeddings that achieves 1.5$\times$ compression, 25% better than the best prior lossless method. The method exploits that spherical coordinates of high-dimensional unit vectors concentrate around $\pi/2$, causing IEEE 754 exponents to collapse to a single value and high-order mantissa bits to become predictable, enabling entropy coding of both. Reconstruction error is bounded by float32 machine epsilon ($1.19 \times 10^{-7}$), making reconstructed values indistinguishable from originals at float32 precision. Evaluation across 26 configurations spanning text, image, and multi-vector embeddings confirms consistent compression improvement with zero measurable retrieval degradation on BEIR benchmarks.
>
---
#### [replaced 016] Group Editing: Edit Multiple Images in One Go
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22883](https://arxiv.org/pdf/2603.22883)**

> **作者:** Yue Ma; Xinyu Wang; Qianli Ma; Qinghe Wang; Mingzhe Zheng; Xiangpeng Yang; Hao Li; Chongbo Zhao; Jixuan Ying; Harry Yang; Hongyu Liu; Qifeng Chen
>
> **备注:** Accepted by CVPR 2026, Project page: this https URL, Github: this https URL
>
> **摘要:** In this paper, we tackle the problem of performing consistent and unified modifications across a set of related images. This task is particularly challenging because these images may vary significantly in pose, viewpoint, and spatial layout. Achieving coherent edits requires establishing reliable correspondences across the images, so that modifications can be applied accurately to semantically aligned regions. To address this, we propose GroupEditing, a novel framework that builds both explicit and implicit relationships among images within a group. On the explicit side, we extract geometric correspondences using VGGT, which provides spatial alignment based on visual features. On the implicit side, we reformulate the image group as a pseudo-video and leverage the temporal coherence priors learned by pre-trained video models to capture latent relationships. To effectively fuse these two types of correspondences, we inject the explicit geometric cues from VGGT into the video model through a novel fusion mechanism. To support large-scale training, we construct GroupEditData, a new dataset containing high-quality masks and detailed captions for numerous image groups. Furthermore, to ensure identity preservation during editing, we introduce an alignment-enhanced RoPE module, which improves the model's ability to maintain consistent appearance across multiple images. Finally, we present GroupEditBench, a dedicated benchmark designed to evaluate the effectiveness of group-level image editing. Extensive experiments demonstrate that GroupEditing significantly outperforms existing methods in terms of visual quality, cross-view consistency, and semantic alignment.
>
---
#### [replaced 017] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2507.02803](https://arxiv.org/pdf/2507.02803)**

> **作者:** Gent Serifi; Marcel C. Buehler
>
> **备注:** CVPR 2026, Project page: this https URL, Code: this https URL
>
> **摘要:** We introduce HyperGaussians, a novel extension of 3D Gaussian Splatting for high-quality animatable face avatars. Creating such detailed face avatars from videos is a challenging problem and has numerous applications in augmented and virtual reality. While tremendous successes have been achieved for static faces, animatable avatars from monocular videos still fall in the uncanny valley. The de facto standard, 3D Gaussian Splatting (3DGS), represents a face through a collection of 3D Gaussian primitives. 3DGS excels at rendering static faces, but the state-of-the-art still struggles with nonlinear deformations, complex lighting effects, and fine details. While most related works focus on predicting better Gaussian parameters from expression codes, we rethink the 3D Gaussian representation itself and how to make it more expressive. Our insights lead to a novel extension of 3D Gaussians to high-dimensional multivariate Gaussians, dubbed 'HyperGaussians'. The higher dimensionality increases expressivity through conditioning on a learnable local embedding. However, splatting HyperGaussians is computationally expensive because it requires inverting a high-dimensional covariance matrix. We solve this by reparameterizing the covariance matrix, dubbed the 'inverse covariance trick'. This trick boosts the efficiency so that HyperGaussians can be seamlessly integrated into existing models. To demonstrate this, we plug in HyperGaussians into the state-of-the-art in fast monocular face avatars: FlashAvatar. Our evaluation on 19 subjects from 4 face datasets shows that HyperGaussians outperform 3DGS numerically and visually, particularly for high-frequency details like eyeglass frames, teeth, complex facial movements, and specular reflections.
>
---
#### [replaced 018] Easy3D-Labels: Supervising Semantic Occupancy Estimation with 3D Pseudo-Labels for Automotive Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.26087](https://arxiv.org/pdf/2509.26087)**

> **作者:** Seamie Hayes; Ganesh Sistu; Tim Brophy; Ciaran Eising
>
> **摘要:** In perception for automated vehicles, safety is critical not only for the driver but also for other agents in the scene, particularly vulnerable road users such as pedestrians and cyclists. Previous representation methods, such as Bird's Eye View, collapse vertical information, leading to ambiguity in 3D object localisation and limiting accurate understanding of the environment for downstream tasks such as motion planning and scene forecasting. In contrast, semantic occupancy provides a full 3D representation of the surroundings, addressing these limitations. Furthermore, self-supervised semantic occupancy has seen increased attention in the automated vehicle domain. Unlike supervised methods that rely on manually annotated data, these approaches use 2D pseudo-labels, improving scalability by reducing the need for labour-intensive annotation. Consequently, such models employ techniques such as novel view synthesis, cross-view rendering, and depth estimation to allow for model supervision against the 2D labels. However, such approaches often incur high computational and memory costs during training, especially for novel view synthesis. To address these issues, we propose Easy3D-Labels, which are 3D pseudo-ground-truth labels generated using Grounded-SAM and Metric3Dv2, with temporal aggregation for densification, permitting supervision directly in 3D space. Easy3D-Labels can be readily integrated into existing models to provide model supervision, yielding substantial performance gains, with mIoU increasing by 45% and RayIoU by 49% when applied to OccNeRF on the Occ3D-nuScenes dataset. Additionally, we introduce EasyOcc, a streamlined model trained solely on these 3D pseudo-labels, avoiding the need for complex rendering strategies, and achieving 15.7 mIoU on Occ3D-nuScenes. Easy3D-Labels improve scene understanding by reducing object duplication and enhancing depth estimation accuracy.
>
---
#### [replaced 019] Monocular Normal Estimation via Shading Sequence Estimation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.09929](https://arxiv.org/pdf/2602.09929)**

> **作者:** Zongrui Li; Xinhua Ma; Minghui Hu; Yunqing Zhao; Yingchen Yu; Qian Zheng; Chang Liu; Xudong Jiang; Song Bai
>
> **备注:** ICLR 2026 (Oral), Project page: this https URL
>
> **摘要:** Monocular normal estimation aims to estimate the normal map from a single RGB image of an object under arbitrary lights. Existing methods rely on deep models to directly predict normal maps. However, they often suffer from 3D misalignment: while the estimated normal maps may appear to have a correct appearance, the reconstructed surfaces often fail to align with the geometric details. We argue that this misalignment stems from the current paradigm: the model struggles to distinguish and reconstruct varying geometry represented in normal maps, as the differences in underlying geometry are reflected only through relatively subtle color variations. To address this issue, we propose a new paradigm that reformulates normal estimation as shading sequence estimation, where shading sequences are more sensitive to various geometric information. Building on this paradigm, we present RoSE, a method that leverages image-to-video generative models to predict shading sequences. The predicted shading sequences are then converted into normal maps by solving a simple ordinary least-squares problem. To enhance robustness and better handle complex objects, RoSE is trained on a synthetic dataset, MultiShade, with diverse shapes, materials, and light conditions. Experiments demonstrate that RoSE achieves state-of-the-art performance on real-world benchmark datasets for object-based monocular normal estimation.
>
---
#### [replaced 020] Foundry: Distilling 3D Foundation Models for the Edge
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **链接: [https://arxiv.org/pdf/2511.20721](https://arxiv.org/pdf/2511.20721)**

> **作者:** Guillaume Letellier; Siddharth Srivastava; Frédéric Jurie; Gaurav Sharma
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Foundation models pre-trained with self-supervised learning (SSL) on large-scale datasets have become powerful general-purpose feature extractors. However, their immense size and computational cost make them prohibitive for deployment on edge devices such as robots and AR/VR headsets. Existing compression techniques like standard knowledge distillation create efficient 'specialist' models but sacrifice the crucial, downstream-agnostic generality that makes foundation models so valuable. In this paper, we introduce Foundation Model Distillation (FMD), a new paradigm for compressing large SSL models into compact, efficient, and faithful proxies that retain their general-purpose representational power. We present Foundry, the first implementation of FMD for 3D point clouds. Our approach, Foundry, trains a student to learn a compressed set of SuperTokens that reconstruct the teacher's token-level representations, capturing a compact basis of its latent space. A single distilled model maintains strong transferability across diverse downstream tasks-classification, part segmentation, and few-shot scenarios-approaching full foundation-model performance while using significantly fewer tokens and FLOPs, making such models more practical for deployment on resourceconstrained hardware.
>
---
#### [replaced 021] From Scale to Speed: Adaptive Test-Time Scaling for Image Editing
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2603.00141](https://arxiv.org/pdf/2603.00141)**

> **作者:** Xiangyan Qu; Zhenlong Yuan; Jing Tang; Rui Chen; Datao Tang; Meng Yu; Lei Sun; Yancheng Bai; Xiangxiang Chu; Gaopeng Gou; Gang Xiong; Yujun Cai
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Image Chain-of-Thought (Image-CoT) is a test-time scaling paradigm that improves image generation by extending inference time. Most Image-CoT methods focus on text-to-image (T2I) generation. Unlike T2I generation, image editing is goal-directed: the solution space is constrained by the source image and instruction. This mismatch causes three challenges when applying Image-CoT to editing: inefficient resource allocation with fixed sampling budgets, unreliable early-stage verification using general MLLM scores, and redundant edited results from large-scale sampling. To address this, we propose ADaptive Edit-CoT (ADE-CoT), an on-demand test-time scaling framework to enhance editing efficiency and performance. It incorporates three key strategies: (1) a difficulty-aware resource allocation that assigns dynamic budgets based on estimated edit difficulty; (2) edit-specific verification in early pruning that uses region localization and caption consistency to select promising candidates; and (3) depth-first opportunistic stopping, guided by an instance-specific verifier, that terminates when intent-aligned results are found. Extensive experiments on three SOTA editing models (Step1X-Edit, BAGEL, FLUX.1 Kontext) across three benchmarks show that ADE-CoT achieves superior performance-efficiency trade-offs. With comparable sampling budgets, ADE-CoT obtains better performance with more than 2x speedup over Best-of-N.
>
---
#### [replaced 022] Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2401.11605](https://arxiv.org/pdf/2401.11605)**

> **作者:** Katherine Crowson; Stefan Andreas Baumann; Alex Birch; Tanishq Mathew Abraham; Daniel Z. Kaplan; Enrico Shippole
>
> **备注:** 20 pages, 13 figures, project page and code available at this https URL
>
> **摘要:** We present the Hourglass Diffusion Transformer (HDiT), an image generative model that exhibits linear scaling with pixel count, supporting training at high-resolution (e.g. $1024 \times 1024$) directly in pixel-space. Building on the Transformer architecture, which is known to scale to billions of parameters, it bridges the gap between the efficiency of convolutional U-Nets and the scalability of Transformers. HDiT trains successfully without typical high-resolution training techniques such as multiscale architectures, latent autoencoders or self-conditioning. We demonstrate that HDiT performs competitively with existing models on ImageNet $256^2$, and sets a new state-of-the-art for diffusion models on FFHQ-$1024^2$.
>
---
#### [replaced 023] CompBench: Benchmarking Complex Instruction-guided Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.12200](https://arxiv.org/pdf/2505.12200)**

> **作者:** Bohan Jia; Wenxuan Huang; Yuntian Tang; Junbo Qiao; Jincheng Liao; Shaosheng Cao; Fei Zhao; Zhaopeng Feng; Zhouhong Gu; Zhenfei Yin; Lei Bai; Wanli Ouyang; Lin Chen; Fei Zhao; Yao Hu; Zihan Wang; Yuan Xie; Shaohui Lin
>
> **摘要:** While real-world applications increasingly demand intricate scene manipulation, existing instruction-guided image editing benchmarks often oversimplify task complexity and lack comprehensive, fine-grained instructions. To bridge this gap, we introduce CompBench, a large-scale benchmark specifically designed for complex instruction-guided image editing. CompBench features challenging editing scenarios that incorporate fine-grained instruction following, spatial and contextual reasoning, thereby enabling comprehensive evaluation of image editing models' precise manipulation capabilities. To construct CompBench, we propose an MLLM-human collaborative framework with tailored task pipelines. Furthermore, we propose an instruction decoupling strategy that disentangles editing intents into four key dimensions: location, appearance, dynamics, and objects, ensuring closer alignment between instructions and complex editing requirements. Extensive evaluations reveal that CompBench exposes fundamental limitations of current image editing models and provides critical insights for the development of next-generation instruction-guided image editing systems. Our project page is available at this https URL.
>
---
#### [replaced 024] ScrollScape: Unlocking 32K Image Generation With Video Diffusion Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24270](https://arxiv.org/pdf/2603.24270)**

> **作者:** Haodong Yu; Yabo Zhang; Donglin Di; Ruyi Zhang; Wangmeng Zuo
>
> **摘要:** While diffusion models excel at generating images with conventional dimensions, pushing them to synthesize ultra-high-resolution imagery at extreme aspect ratios (EAR) often triggers catastrophic structural failures, such as object repetition and spatial fragmentation. This limitation fundamentally stems from a lack of robust spatial priors, as static text-to-image models are primarily trained on image distributions with conventional dimensions. To overcome this bottleneck, we present ScrollScape, a novel framework that reformulates EAR image synthesis into a continuous video generation process through two core innovations. By mapping the spatial expansion of a massive canvas to the temporal evolution of video frames, ScrollScape leverages the inherent temporal consistency of video models as a powerful global constraint to ensure long-range structural integrity. Specifically, Scanning Positional Encoding (ScanPE) distributes global coordinates across frames to act as a flexible moving camera, while Scrolling Super-Resolution (ScrollSR) leverages video super-resolution priors to circumvent memory bottlenecks, efficiently scaling outputs to an unprecedented 32K resolution. Fine-tuned on a curated 3K multi-ratio image dataset, ScrollScape effectively aligns pre-trained video priors with the EAR generation task. Extensive evaluations demonstrate that it significantly outperforms existing image-diffusion baselines by eliminating severe localized artifacts. Consequently, our method overcomes inherent structural bottlenecks to ensure exceptional global coherence and visual fidelity across diverse domains at extreme scales.
>
---
#### [replaced 025] Test-Time Modification: Inverse Domain Transformation for Robust Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13454](https://arxiv.org/pdf/2512.13454)**

> **作者:** Arpit Jadon; Joshua Niemeijer; Yuki M. Asano
>
> **备注:** Preprint
>
> **摘要:** Generative foundation models contain broad visual knowledge and can produce diverse image variations, making them particularly promising for advancing domain generalization tasks. They can be used for training data augmentation, but synthesizing comprehensive target-domain variations remains slow, expensive, and incomplete. We propose an alternative: using diffusion models at test time to map target images back to the source distribution where the downstream model was trained. This approach requires only a source domain description, preserves the task model, and eliminates large-scale synthetic data generation. We demonstrate consistent improvements across segmentation, detection, and classification tasks under challenging environmental shifts in real-to-real domain generalization scenarios with unknown target distributions. Our analysis spans multiple generative and downstream models, including an ensemble variant for enhanced robustness. The method improves BDD100K-Night-Det mAP@50 from 10.2 to 31.8, ImageNet-R top-1 from 36.1 to 60.8, and DarkZurich mIoU from 28.6 to 46.3.
>
---
#### [replaced 026] Mario: Multimodal Graph Reasoning with Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05181](https://arxiv.org/pdf/2603.05181)**

> **作者:** Yuanfu Sun; Kang Li; Pengkang Guo; Jiajin Liu; Qiaoyu Tan
>
> **备注:** CVPR 2026
>
> **摘要:** Recent advances in large language models (LLMs) have opened new avenues for multimodal reasoning. Yet, most existing methods still rely on pretrained vision-language models (VLMs) to encode image-text pairs in isolation, ignoring the relational structure that real-world multimodal data naturally form. This motivates reasoning on multimodal graphs (MMGs), where each node has textual and visual attributes and edges provide structural cues. Enabling LLM-based reasoning on such heterogeneous multimodal signals while preserving graph topology introduces two key challenges: resolving weak cross-modal consistency and handling heterogeneous modality preference. To address this, we propose Mario, a unified framework that simultaneously resolves the two above challenges and enables effective LLM-based reasoning over MMGs. Mario consists of two innovative stages. Firstly, a graph-conditioned VLM design that jointly refines textual and visual features through fine-grained cross-modal contrastive learning guided by graph topology. Secondly, a modality-adaptive graph instruction tuning mechanism that organizes aligned multimodal features into graph-aware instruction views and employs a learnable router to surface, for each node and its neighborhood, the most informative modality configuration to the LLM. Extensive experiments across diverse MMG benchmarks demonstrate that Mario consistently outperforms state-of-the-art graph models in both supervised and zero-shot scenarios for node classification and link prediction. The code will be made available at this https URL.
>
---
#### [replaced 027] Closing the Navigation Compliance Gap in End-to-end Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10660](https://arxiv.org/pdf/2512.10660)**

> **作者:** Hanfeng Wu; Marlon Steiner; Michael Schmidt; Alvaro Marcos-Ramiro; Christoph Stiller
>
> **摘要:** Trajectory-scoring planners achieve high navigation compliance when following the expert's original command, yet they struggle at intersections when presented with alternative commands; over 30 percent of such commands are ignored. We attribute this navigation compliance gap to two root causes: (1) existing metrics like Ego Progress do not explicitly measure navigation adherence, diluting the gap between on-route and off-route trajectories; and (2) current datasets pair each scenario with a single command, preventing models from learning command-dependent behavior. We address the metric gap by introducing the binary Navigation Compliance metric (NAVI) and the derived Controllability Measure (CM), and the data gap with the NavControl dataset, 14,918 intersection scenarios augmented with all feasible alternative commands and routing annotations, yielding over 34,000 direction samples. Building on these, we propose NaviHydra, a trajectory-scoring planner incorporating NAVI distillation and Bird's Eye View (BEV)-based trajectory gathering for context-position-aware trajectory feature extraction. NaviHydra achieves 92.7 PDM score on NAVSIM navtest split and 77.5 CM on NavControl test split. Training with NavControl improves controllability across diverse architectures, confirming it as a broadly effective augmentation for navigation compliance.
>
---
#### [replaced 028] ByteLoom: Weaving Geometry-Consistent Human-Object Interactions through Progressive Curriculum Learning
- **分类: cs.CV; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.22854](https://arxiv.org/pdf/2512.22854)**

> **作者:** Bangya Liu; Xinyu Gong; Zelin Zhao; Ziyang Song; Yulei Lu; Suhui Wu; Jun Zhang; Suman Banerjee; Hao Zhang
>
> **摘要:** Human-object interaction (HOI) video generation has garnered increasing attention due to its promising applications in digital humans, e-commerce, advertising, and robotics imitation learning. However, existing methods face two critical limitations: (1) a lack of effective mechanisms to inject multi-view information of the object into the model, leading to poor cross-view consistency, and (2) heavy reliance on fine-grained hand mesh annotations for modeling interaction occlusions. To address these challenges, we introduce ByteLoom, a Diffusion Transformer (DiT)-based framework that generates realistic HOI videos with geometrically consistent object illustration, using simplified human conditioning and 3D object inputs. We first propose an RCM-cache mechanism that leverages Relative Coordinate Maps (RCM) as a universal representation to maintain object's geometry consistency and precisely control 6-DoF object transformations in the meantime. To compensate HOI dataset scarcity and leverage existing datasets, we further design a training curriculum that enhances model capabilities in a progressive style and relaxes the demand of hand mesh. Extensive experiments demonstrate that our method faithfully preserves human identity and the object's multi-view geometry, while maintaining smooth motion and object manipulation.
>
---
#### [replaced 029] Gastric-X: A Multimodal Multi-Phase Benchmark Dataset for Advancing Vision-Language Models in Gastric Cancer Analysis
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.19516](https://arxiv.org/pdf/2603.19516)**

> **作者:** Sheng Lu; Hao Chen; Rui Yin; Juyan Ba; Yu Zhang; Yuanzhe Li
>
> **备注:** Computer Vision and Pattern Recognition 2026
>
> **摘要:** Recent vision-language models (VLMs) have shown strong generalization and multimodal reasoning abilities in natural domains. However, their application to medical diagnosis remains limited by the lack of comprehensive and structured datasets that capture real clinical workflows. To advance the development of VLMs for clinical applications, particularly in gastric cancer, we introduce Gastric-X, a large-scale multimodal benchmark for gastric cancer analysis providing 1.7K cases. Each case in Gastric-X includes paired resting and dynamic CT scans, endoscopic image, a set of structured biochemical indicators, expert-authored diagnostic notes, and bounding box annotations of tumor regions, reflecting realistic clinical conditions. We systematically examine the capability of recent VLMs on five core tasks: Visual Question Answering (VQA), report generation, cross-modal retrieval, disease classification, and lesion localization. These tasks simulate critical stages of clinical workflow, from visual understanding and reasoning to multimodal decision support. Through this evaluation, we aim not only to assess model performance but also to probe the nature of VLM understanding: Can current VLMs meaningfully correlate biochemical signals with spatial tumor features and textual reports? We envision Gastric-X as a step toward aligning machine intelligence with the cognitive and evidential reasoning processes of physicians, and as a resource to inspire the development of next-generation medical VLMs.
>
---
#### [replaced 030] 3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，解决深度运动下操纵性能不足的问题。通过引入3D动态建模与策略学习，提升操纵政策的3D预见能力。**

- **链接: [https://arxiv.org/pdf/2502.10028](https://arxiv.org/pdf/2502.10028)**

> **作者:** Yuxin He; Ruihao Zhang; Xianzu Wu; Zhiyuan Zhang; Cheng Ding; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** The incorporation of world modeling into manipulation policy learning has pushed the boundary of manipulation performance. However, existing efforts simply model the 2D visual dynamics, which is insufficient for robust manipulation when target tasks involve prominent depth-wise movement. To address this, we present a 3D dynamics-aware manipulation framework that seamlessly integrates 3D world modeling and policy learning. Three self-supervised learning tasks (current depth estimation, future RGB-D prediction, 3D flow prediction) are introduced within our framework, which complement each other and endow the policy model with 3D foresight. Extensive experiments on simulation and the real world show that 3D foresight can greatly boost the performance of manipulation policies without sacrificing inference speed. Code is available at this https URL.
>
---
#### [replaced 031] Patch2Loc: Learning to Localize Patches for Unsupervised Brain Lesion Detection
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.22504](https://arxiv.org/pdf/2506.22504)**

> **作者:** Hassan Baker; Austin J. Brockmeier
>
> **备注:** Accepted at AISTATS 2026 (Proceedings of Machine Learning Research)
>
> **摘要:** Detecting brain lesions as abnormalities observed in magnetic resonance imaging (MRI) is essential for diagnosis and treatment. In the search of abnormalities, such as tumors and malformations, radiologists may benefit from computer-aided diagnostics that use computer vision systems trained with machine learning to segment normal tissue from abnormal brain tissue. While supervised learning methods require annotated lesions, we propose a new unsupervised approach (Patch2Loc) that learns from normal patches taken from structural MRI. We train a neural network model to map a patch back to its spatial location within a slice of the brain volume. During inference, abnormal patches are detected by the relatively higher error and/or variance of the location prediction. This generates a heatmap that can be integrated into pixel-wise methods to achieve finer-grained segmentation. We demonstrate the ability of our model to segment abnormal brain tissues by applying our approach to the detection of tumor tissues in MRI on T2-weighted images from BraTS2021 and MSLUB datasets and T1-weighted images from ATLAS and WMH datasets. We show that it outperforms the state-of-the art in unsupervised segmentation. The implementation for this work can be found on our \href{this https URL}{GitHub page}. This paper has been accepted at AISTATS 2026.
>
---
#### [replaced 032] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23637](https://arxiv.org/pdf/2603.23637)**

> **作者:** Peiyu Xu; Xin Sun; Krishna Mullia; Raymond Fei; Iliyan Georgiev; Shuang Zhao
>
> **备注:** Project Page: this https URL
>
> **摘要:** Ray-tracing-based 3D Gaussian splatting (3DGS) methods overcome the limitations of rasterization -- rigid pinhole camera assumptions, inaccurate shadows, and lack of native reflection or refraction -- but remain slower due to the cost of sorting all intersecting Gaussians along every ray. Moreover, existing ray-tracing methods still rely on rasterization-style approximations such as shadow mapping for relightable scenes, undermining the generality that ray tracing promises. We present a differentiable, sorting-free stochastic formulation for ray-traced 3DGS -- the first framework that uses stochastic ray tracing to both reconstruct and render standard and relightable 3DGS scenes. At its core is an unbiased Monte Carlo estimator for pixel-color gradients that evaluates only a small sampled subset of Gaussians per ray, bypassing the need for sorting. For standard 3DGS, our method matches the reconstruction quality and speed of rasterization-based 3DGS while substantially outperforming sorting-based ray tracing. For relightable 3DGS, the same stochastic estimator drives per-Gaussian shading with fully ray-traced shadow rays, delivering notably higher reconstruction fidelity than prior work.
>
---
#### [replaced 033] PokeFusion Attention: Enhancing Reference-Free Style-Conditioned Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03220](https://arxiv.org/pdf/2602.03220)**

> **作者:** Jingbang Tang
>
> **备注:** The authors withdraw this submission to make substantial revisions and improvements. A revised version will be submitted in the future
>
> **摘要:** This paper studies reference-free style-conditioned character generation in text-to-image diffusion models, where high-quality synthesis requires both stable character structure and consistent, fine-grained style expression across diverse prompts. Existing approaches primarily rely on text-only prompting, which is often under-specified for visual style and tends to produce noticeable style drift and geometric inconsistency, or introduce reference-based adapters that depend on external images at inference time, increasing architectural complexity and limiting deployment this http URL propose PokeFusion Attention, a lightweight decoder-level cross-attention mechanism that fuses textual semantics with learned style embeddings directly inside the diffusion decoder. By decoupling text and style conditioning at the attention level, our method enables effective reference-free stylized generation while keeping the pretrained diffusion backbone fully this http URL Attention trains only decoder cross-attention layers together with a compact style projection module, resulting in a parameter-efficient and plug-and-play control component that can be easily integrated into existing diffusion pipelines and transferred across different this http URL on a stylized character generation benchmark (Pokemon-style) demonstrate that our method consistently improves style fidelity, semantic alignment, and character shape consistency compared with representative adapter-based baselines, while maintaining low parameter overhead and inference-time simplicity.
>
---
#### [replaced 034] See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20951](https://arxiv.org/pdf/2602.20951)**

> **作者:** Jaehyun Park; Minyoung Ahn; Minkyu Kim; Jonghyun Lee; Jae-Gil Lee; Dongmin Park
>
> **摘要:** Despite recent advances in diffusion models, AI generated images still often contain visual artifacts that compromise realism. Although more thorough pre-training and bigger models might reduce artifacts, there is no assurance that they can be completely eliminated, which makes artifact mitigation a highly crucial area of study. Previous artifact-aware methodologies depend on human-labeled artifact datasets, which are costly and difficult to scale, underscoring the need for an automated approach to reliably acquire artifact-annotated datasets. In this paper, we propose ArtiAgent, which efficiently creates pairs of real and artifact-injected images. It comprises three agents: a perception agent that recognizes and grounds entities and subentities from real images, a synthesis agent that introduces artifacts via artifact injection tools through novel patch-wise embedding manipulation within a diffusion transformer, and a curation agent that filters the synthesized artifacts and generates both local and global explanations for each instance. Using ArtiAgent, we synthesize 100K images with rich artifact annotations and demonstrate both efficacy and versatility across diverse applications. Code is available at link.
>
---
#### [replaced 035] JANUS: A Lightweight Framework for Jailbreaking Text-to-Image Models via Distribution Optimization
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.21208](https://arxiv.org/pdf/2603.21208)**

> **作者:** Haolun Zheng; Yu He; Tailun Chen; Shuo Shao; Zhixuan Chu; Hongbin Zhou; Lan Tao; Zhan Qin; Kui Ren
>
> **备注:** This paper is accepted by the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026. 18 pages, 8 figures
>
> **摘要:** Text-to-image (T2I) models such as Stable Diffusion and DALLE remain susceptible to generating harmful or Not-Safe-For-Work (NSFW) content under jailbreak attacks despite deployed safety filters. Existing jailbreak attacks either rely on proxy-loss optimization instead of the true end-to-end objective, or depend on large-scale and costly RL-trained generators. Motivated by these limitations, we propose JANUS , a lightweight framework that formulates jailbreak as optimizing a structured prompt distribution under a black-box, end-to-end reward from the T2I system and its safety filters. JANUS replaces a high-capacity generator with a low-dimensional mixing policy over two semantically anchored prompt distributions, enabling efficient exploration while preserving the target semantics. On modern T2I models, we outperform state-of-the-art jailbreak methods, improving ASR-8 from 25.30% to 43.15% on Stable Diffusion 3.5 Large Turbo with consistently higher CLIP and NSFW scores. JANUS succeeds across both open-source and commercial models. These findings expose structural weaknesses in current T2I safety pipelines and motivate stronger, distribution-aware defenses. Warning: This paper contains model outputs that may be offensive.
>
---
#### [replaced 036] Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.24821](https://arxiv.org/pdf/2510.24821)**

> **作者:** Inclusion AI; Bowen Ma; Cheng Zou; ChengKun Du; Canxiang Yan; Chunxiang Jin; Chunjie Shen; Chenyu Lian; Chengxiang Fan; Dandan Zheng; Fudong Wang; Furong Xu; Guangming Yao; Haohao Liu; Han Peng; Jun Zhou; Junluan Xia; Jingdong Chen; Jianing Li; Jianxin Sun; Jianjiang Zhu; Jianping Jiang; Jinpeng Ou; Jun Peng; Jin Peng; Kaixiang Ji; Li Tang; Libin Wang; Lixiang Ru; Longhua Tan; Lu Ma; Lan Wang; Mochen Bai; Minghong Cai; Mingxue Yang; Ning Gao; Qingpei Guo; Qinglong Zhang; Qiang Xu; Qin Zhao; Rui Liu; Ruijie Xiong; Ruobing Zheng; Sirui Gao; Shaoxiong Lin; Tao Zhang; Tianqi Li; Tinghao Liu; Tongli Wang; Taoye Huang; Weilong Chai; Xiaomei Wang; Xiaolong Wang; Xiaojian Liu; Xiao Lu; Xiaoyu Li; Xingning Dong; Xuzheng Yu; Xuezhi Wang; Yi Yuan; Yuting Gao; Yuting Xiao; Yunxiao Sun; Yipeng Chen; Yifan Mao; Yifei Wu; Yongjie Lyu; Yingying Zhang; YuQian Li; Ziping Ma; Zhiqiang Fang; Zhihao Qiu; Ziyuan Huang; Zizheng Yang; Zhengyu He
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** We propose Ming-Flash-Omni, an upgraded version of Ming-Omni, built upon a sparser Mixture-of-Experts (MoE) variant of Ling-Flash-2.0 with 100 billion total parameters, of which only 6.1 billion are active per token. This architecture enables highly efficient scaling (dramatically improving computational efficiency while significantly expanding model capacity) and empowers stronger unified multimodal intelligence across vision, speech, and language, representing a key step toward Artificial General Intelligence (AGI). Compared to its predecessor, the upgraded version exhibits substantial improvements across multimodal understanding and generation. Notably, it achieves strong performance on vision-language understanding benchmarks, with overall scores on par with Gemini 2.5 Pro, and enables seamless switching among multimodal tasks in multi-turn interactions. In speech, it achieves strong performance in contextual and dialect-aware ASR while enabling joint, continuous-generation of speech, sound, and music. In vision, it introduces generative semantic segmentation that achieves competitive standalone performance and enhances spatial control and editing consistency, alongside marked improvements in identity preservation, and high-fidelity in-image text rendering. Together, these capabilities demonstrate that a single unified model can serve as a practical foundation for general-purpose multimodal intelligence.
>
---
#### [replaced 037] MindSet: Vision. A toolbox for testing DNNs on key psychological experiments
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2404.05290](https://arxiv.org/pdf/2404.05290)**

> **作者:** Valerio Biscione; Milton L. Montero; Marin Dujmovic; Gaurav Malhotra; Dong Yin; Guillermo Puebla; Federico Adolfi; Rachel F. Heaton; John E. Hummel; Benjamin D. Evans; Karim Habashy; Jeffrey S. Bowers
>
> **备注:** 34 pages, 12 figures. Updated version with additional model evaluations
>
> **摘要:** Multiple benchmarks have been developed to assess the alignment between deep neural networks (DNNs) and human vision. In almost all cases these benchmarks are observational in the sense they are composed of behavioural and brain responses to naturalistic images that have not been manipulated to test hypotheses regarding how DNNs or humans perceive and identify objects. Here we introduce the toolbox \textit{MindSet: Vision}, consisting of a collection of image datasets and related scripts designed to test DNNs on 30 psychological findings. In all experimental conditions, the stimuli are systematically manipulated to test specific hypotheses regarding human visual perception and object recognition. In addition to providing pre-generated datasets of images, we provide code to regenerate these datasets, offering many configurable parameters which greatly extend the dataset versatility for different research contexts, and code to facilitate the testing of DNNs on these image datasets using three different methods (similarity judgments, out-of-distribution classification, and decoder method), accessible via this https URL. To illustrate the challenges these datasets pose for developing better DNN models of human vision, we test several models on range of datasets included in the toolbox.
>
---
#### [replaced 038] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21936](https://arxiv.org/pdf/2603.21936)**

> **作者:** Roy Amoyal; Oren Freifeld; Chaim Baskin
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present Gaussian Splatting Alignment (GSA), a novel method for aligning two independent 3D Gaussian Splatting (3DGS) models via a similarity transformation (rotation, translation, and scale), even when they are of different objects in the same category (e.g., different cars). In contrast, existing methods can only align 3DGS models of the same object (e.g., the same car) and often must be given true scale as input, while we estimate it successfully. GSA leverages viewpoint-guided spherical map features to obtain robust correspondences and introduces a two-step optimization framework that aligns 3DGS models while keeping them fixed. First, we apply an iterative feature-guided absolute orientation solver as our coarse registration, which is robust to poor initialization (e.g., 180 degrees misalignment or a 10x scale gap). Next, we use a fine registration step that enforces multi-view feature consistency, inspired by inverse radiance-field formulations. The first step already achieves state-of-the-art performance, and the second further improves results. In the same-object case, GSA outperforms prior works, often by a large margin, even when the other methods are given the true scale. In the harder case of different objects in the same category, GSA vastly surpasses them, providing the first effective solution for category-level 3DGS registration and unlocking new applications. Project webpage: this https URL
>
---
#### [replaced 039] Debugging Concept Bottleneck Models through Removal and Retraining
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.21385](https://arxiv.org/pdf/2509.21385)**

> **作者:** Eric Enouen; Sainyam Galhotra
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Concept Bottleneck Models (CBMs) use a set of human-interpretable concepts to predict the final task label, enabling domain experts to not only validate the CBM's predictions, but also intervene on incorrect concepts at test time. However, these interventions fail to address systemic misalignment between the CBM and the expert's reasoning, such as when the model learns shortcuts from biased data. To address this, we present a general interpretable debugging framework for CBMs that follows a two-step process of Removal and Retraining. In the Removal step, experts use concept explanations to identify and remove any undesired concepts. In the Retraining step, we introduce CBDebug, a novel method that leverages the interpretability of CBMs as a bridge for converting concept-level user feedback into sample-level auxiliary labels. These labels are then used to apply supervised bias mitigation and targeted augmentation, reducing the model's reliance on undesired concepts. We evaluate our framework with both real and automated expert feedback, and find that CBDebug significantly outperforms prior retraining methods across multiple CBM architectures (PIP-Net, Post-hoc CBM) and benchmarks with known spurious correlations.
>
---
#### [replaced 040] PE3R: Perception-Efficient 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07507](https://arxiv.org/pdf/2503.07507)**

> **作者:** Jie Hu; Shizun Wang; Xinchao Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent advances in 2D-to-3D perception have enabled the recovery of 3D scene semantics from unposed images. However, prevailing methods often suffer from limited generalization, reliance on per-scene optimization, and semantic inconsistencies across viewpoints. To address these limitations, we introduce PE3R, a tuning-free framework for efficient and generalizable 3D semantic reconstruction. By integrating multi-view geometry with 2D semantic priors in a feed-forward pipeline, PE3R achieves zero-shot generalization across diverse scenes and object categories without any scene-specific fine-tuning. Extensive evaluations on open-vocabulary segmentation and multi-view depth estimation show that PE3R not only achieves up to 9$\times$ faster inference but also sets new state-of-the-art accuracy in both semantic and geometric metrics. Our approach paves the way for scalable, language-driven 3D scene understanding. Code is available at this http URL.
>
---
#### [replaced 041] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多摄像头3D目标检测任务，旨在解决模型在不同摄像头配置间泛化能力差的问题。提出CoIn3D框架，通过空间感知特征调制和相机感知数据增强提升跨配置性能。**

- **链接: [https://arxiv.org/pdf/2603.05042](https://arxiv.org/pdf/2603.05042)**

> **作者:** Zhaonian Kuang; Rui Ding; Haotian Wang; Xinhu Zheng; Meng Yang; Gang Hua
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** Multi-camera 3D object detection (MC3D) has attracted increasing attention with the growing deployment of multi-sensor physical agents, such as robots and autonomous vehicles. However, MC3D models still struggle to generalize to unseen platforms with new multi-camera configurations. Current solutions simply employ a meta-camera for unified representation but lack comprehensive consideration. In this paper, we revisit this issue and identify that the devil lies in spatial prior discrepancies across source and target configurations, including different intrinsics, extrinsics, and array layouts. To address this, we propose CoIn3D, a generalizable MC3D framework that enables strong transferability from source configurations to unseen target ones. CoIn3D explicitly incorporates all identified spatial priors into both feature embedding and image observation through spatial-aware feature modulation (SFM) and camera-aware data augmentation (CDA), respectively. SFM enriches feature space by integrating four spatial representations, such as focal length, ground depth, ground gradient, and Plücker coordinate. CDA improves observation diversity under various configurations via a training-free dynamic novel-view image synthesis scheme. Extensive experiments demonstrate that CoIn3D achieves strong cross-configuration performance on landmark datasets such as NuScenes, Waymo, and Lyft, under three dominant MC3D paradigms represented by BEVDepth, BEVFormer, and PETR.
>
---
#### [replaced 042] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于视频时间定位任务，旨在提升多模态大语言模型的视频时间定位能力。通过优化数据质量和算法设计，提出TimeLens框架，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.14698](https://arxiv.org/pdf/2512.14698)**

> **作者:** Jun Zhang; Teng Wang; Yuying Ge; Yixiao Ge; Xinhao Li; Ying Shan; Limin Wang
>
> **备注:** CVPR 2026. Website: this https URL
>
> **摘要:** This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. We first expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability of prior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipes for RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released to facilitate future research.
>
---
#### [replaced 043] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决失败诊断与学习问题。提出ViFailback框架，结合视觉符号进行故障分析与纠正，并发布相关数据集与基准测试。**

- **链接: [https://arxiv.org/pdf/2512.02787](https://arxiv.org/pdf/2512.02787)**

> **作者:** Xianchao Zeng; Xinyu Zhou; Youcheng Li; Jiayou Shi; Tianle Li; Liangming Chen; Lei Ren; Yong-Lu Li
>
> **备注:** Accepted by CVPR 2026. Project Website: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic manipulation, yet they remain limited in failure diagnosis and learning from failures. Additionally, existing failure datasets are mostly generated programmatically in simulation, which limits their generalization to the real world. In light of these, we introduce ViFailback, a framework designed to diagnose robotic manipulation failures and provide both textual and visual correction guidance. Our framework utilizes explicit visual symbols to enhance annotation efficiency. We further release the ViFailback dataset, a large-scale collection of 58,126 Visual Question Answering (VQA) pairs along with their corresponding 5,202 real-world manipulation trajectories. Based on the dataset, we establish ViFailback-Bench, a benchmark of 11 fine-grained VQA tasks designed to assess the failure diagnosis and correction abilities of Vision-Language Models (VLMs), featuring ViFailback-Bench Lite for closed-ended and ViFailback-Bench Hard for open-ended evaluation. To demonstrate the effectiveness of our framework, we built the ViFailback-8B VLM, which not only achieves significant overall performance improvement on ViFailback-Bench but also generates visual symbols for corrective action guidance. Finally, by integrating ViFailback-8B with a VLA model, we conduct real-world robotic experiments demonstrating its ability to assist the VLA model in recovering from failures. Project Website: this https URL
>
---
#### [replaced 044] ConcreTizer: Model Inversion Attack via Occupancy Classification and Dispersion Control for 3D Point Cloud Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06986](https://arxiv.org/pdf/2503.06986)**

> **作者:** Youngseok Kim; Sunwook Hwang; Hyung-Sin Kim; Saewoong Bahk
>
> **备注:** Added acceptance note (ICLR 2025) to the heading
>
> **摘要:** The growing use of 3D point cloud data in autonomous vehicles (AVs) has raised serious privacy concerns, particularly due to the sensitive information that can be extracted from 3D data. While model inversion attacks have been widely studied in the context of 2D data, their application to 3D point clouds remains largely unexplored. To fill this gap, we present the first in-depth study of model inversion attacks aimed at restoring 3D point cloud scenes. Our analysis reveals the unique challenges, the inherent sparsity of 3D point clouds and the ambiguity between empty and non-empty voxels after voxelization, which are further exacerbated by the dispersion of non-empty voxels across feature extractor layers. To address these challenges, we introduce ConcreTizer, a simple yet effective model inversion attack designed specifically for voxel-based 3D point cloud data. ConcreTizer incorporates Voxel Occupancy Classification to distinguish between empty and non-empty voxels and Dispersion-Controlled Supervision to mitigate non-empty voxel dispersion. Extensive experiments on widely used 3D feature extractors and benchmark datasets, such as KITTI and Waymo, demonstrate that ConcreTizer concretely restores the original 3D point cloud scene from disrupted 3D feature data. Our findings highlight both the vulnerability of 3D data to inversion attacks and the urgent need for robust defense strategies.
>
---
#### [replaced 045] ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22666](https://arxiv.org/pdf/2602.22666)**

> **作者:** Xuelu Li; Zhaonan Wang; Xiaogang Wang; Lei Wu; Manyi Li; Changhe Tu
>
> **摘要:** Reconstructing articulated objects into high-fidelity digital twins is crucial for applications such as robotic manipulation and interactive simulation. Recent self-supervised methods using differentiable rendering frameworks like 3D Gaussian Splatting remain highly sensitive to the initial part segmentation. Their reliance on heuristic clustering or pre-trained models often causes optimization to converge to local minima, especially for complex multi-part objects. To address these limitations, we propose ArtPro, a novel self-supervised framework that introduces adaptive integration of mobility proposals. Our approach begins with an over-segmentation initialization guided by geometry features and motion priors, generating part proposals with plausible motion hypotheses. During optimization, we dynamically merge these proposals by analyzing motion consistency among spatial neighbors, while a collision-aware motion pruning mechanism prevents erroneous kinematic estimation. Extensive experiments on both synthetic and real-world objects demonstrate that ArtPro achieves robust reconstruction of complex multi-part objects, significantly outperforming existing methods in accuracy and stability.
>
---
#### [replaced 046] Weight Space Representation Learning on Diverse NeRF Architectures
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.09623](https://arxiv.org/pdf/2502.09623)**

> **作者:** Francesco Ballerini; Pierluigi Zama Ramirez; Luigi Di Stefano; Samuele Salti
>
> **备注:** v5: fixed typo in tabs. 11-13. Accepted at ICLR 2026
>
> **摘要:** Neural Radiance Fields (NeRFs) have emerged as a groundbreaking paradigm for representing 3D objects and scenes by encoding shape and appearance information into the weights of a neural network. Recent studies have demonstrated that these weights can be used as input for frameworks designed to address deep learning tasks; however, such frameworks require NeRFs to adhere to a specific, predefined architecture. In this paper, we introduce the first framework capable of processing NeRFs with diverse architectures and performing inference on architectures unseen at training time. We achieve this by training a Graph Meta-Network within an unsupervised representation learning framework, and show that a contrastive objective is conducive to obtaining an architecture-agnostic latent space. In experiments conducted across 13 NeRF architectures belonging to three families (MLPs, tri-planes, and, for the first time, hash tables), our approach demonstrates robust performance in classification, retrieval, and language tasks involving multiple architectures, even unseen at training time, while also matching or exceeding the results of existing frameworks limited to single architectures. Our code and data are available at this https URL.
>
---
#### [replaced 047] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06581](https://arxiv.org/pdf/2512.06581)**

> **作者:** Yuhao Su; Anwesa Choudhuri; Zhongpai Gao; Benjamin Planche; Van Nguyen Nguyen; Meng Zheng; Yuhan Shen; Arun Innanje; Terrence Chen; Ehsan Elhamifar; Ziyan Wu
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Large vision-language models struggle with medical video understanding, where spatial precision, temporal reasoning, and clinical semantics are critical. To address this, we first introduce \textbf{MedVidBench}, a large-scale benchmark of 531,850 video-instruction pairs across 8 medical sources spanning video, segment, and frame-level tasks, curated through a rigorous quality assurance pipeline with expert-guided prompting and dual-model validation. While supervised fine-tuning on MedVidBench yields noticeable gains, standard Reinforcement Learning (RL) fails due to imbalanced reward scales across datasets, which destabilizes optimization and leads to training collapse. To overcome this, we introduce \textbf{MedGRPO}, a novel RL framework for balanced multi-dataset training with two key innovations: (1) \emph{cross-dataset reward normalization} that maps each dataset's median performance to a common reward value, ensuring fair optimization regardless of difficulty, and (2) a \emph{medical LLM judge} that evaluates caption quality on five clinical dimensions through comparative similarity scoring. Supervised fine-tuning Qwen2.5-VL-7B on MedVidBench substantially outperforms GPT-4.1 and Gemini-2.5-Flash across all tasks, demonstrating MedVidBench's efficacy, while our MedGRPO framework further improves upon the SFT baseline across grounding and captioning tasks. Our work establishes a foundational benchmark and robust training methodology for advancing vision-language models in medical domains. Our project website is available at this https URL.
>
---
#### [replaced 048] Cov2Pose: Leveraging Spatial Covariance for Direct Manifold-aware 6-DoF Object Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19961](https://arxiv.org/pdf/2603.19961)**

> **作者:** Nassim Ali Ousalah; Peyman Rostami; Vincent Gaudillière; Emmanuel Koumandakis; Anis Kacem; Enjie Ghorbel; Djamila Aouada
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** In this paper, we address the problem of 6-DoF object pose estimation from a single RGB image. Indirect methods that typically predict intermediate 2D keypoints, followed by a Perspective-n-Point solver, have shown great performance. Direct approaches, which regress the pose in an end-to-end manner, are usually computationally more efficient but less accurate. However, direct pose regression heads rely on globally pooled features, ignoring spatial second-order statistics despite their informativeness in pose prediction. They also predict, in most cases, discontinuous pose representations that lack robustness. Herein, we therefore propose a covariance-pooled representation that encodes convolutional feature distributions as a symmetric positive definite (SPD) matrix. Moreover, we propose a novel pose encoding in the form of an SPD matrix via its Cholesky decomposition. Pose is then regressed in an end-to-end manner with a manifold-aware network head, taking into account the Riemannian geometry of SPD matrices. Experiments and ablations consistently demonstrate the relevance of second-order pooling and continuous representations for direct pose regression, including under partial occlusion.
>
---
#### [replaced 049] High-speed Imaging through Turbulence with Event-based Light Fields
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14023](https://arxiv.org/pdf/2603.14023)**

> **作者:** Yu-Hsiang Huang; Levi Burner; Sachin Shah; Ziyuan Qu; Adithya Pediredla; Christopher A. Metzler
>
> **摘要:** This work introduces and demonstrates the first system capable of imaging fast-moving extended non-rigid objects through strong atmospheric turbulence at high frame rate. Event cameras are a novel sensing architecture capable of estimating high-speed imagery at thousands of frames per second. However, on their own event cameras are unable to disambiguate scene motion from turbulence. In this work, we overcome this limitation using event-based light field cameras: By simultaneously capturing multiple views of a scene, event-based light field cameras and machine learning-based reconstruction algorithms are able to disambiguate motion-induced dynamics, which produce events that are strongly correlated across views, from turbulence-induced dynamics, which produce events that are weakly correlated across view. Tabletop experiments demonstrate event-based light field can overcome strong turbulence while imaging high-speed objects traveling at up to 16,000 pixels per second.
>
---
#### [replaced 050] Elastic Weight Consolidation Done Right for Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.18596](https://arxiv.org/pdf/2603.18596)**

> **作者:** Xuan Liu; Xiaobin Chang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Weight regularization methods in continual learning (CL) alleviate catastrophic forgetting by assessing and penalizing changes to important model weights. Elastic Weight Consolidation (EWC) is a foundational and widely used approach within this framework that estimates weight importance based on gradients. However, it has consistently shown suboptimal performance. In this paper, we conduct a systematic analysis of importance estimation in EWC from a gradient-based perspective. For the first time, we find that EWC's reliance on the Fisher Information Matrix (FIM) results in gradient vanishing and inaccurate importance estimation in certain scenarios. Our analysis also reveals that Memory Aware Synapses (MAS), a variant of EWC, imposes unnecessary constraints on parameters irrelevant to prior tasks, termed the redundant protection. Consequently, both EWC and its variants exhibit fundamental misalignments in estimating weight importance, leading to inferior performance. To tackle these issues, we propose the Logits Reversal (LR) operation, a simple yet effective modification that rectifies EWC's importance estimation. Specifically, reversing the logit values during the calculation of FIM can effectively prevent both gradient vanishing and redundant protection. Extensive experiments across various CL tasks and datasets show that the proposed method significantly outperforms existing EWC and its variants. Therefore, we refer to it as EWC Done Right (EWC-DR). Code is available at this https URL.
>
---
#### [replaced 051] Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.15869](https://arxiv.org/pdf/2411.15869)**

> **作者:** Sule Bai; Yong Liu; Yifei Han; Haoji Zhang; Yansong Tang; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted by IEEE TIP
>
> **摘要:** Recent advancements in pre-trained vision-language models like CLIP have enabled the task of open-vocabulary segmentation. CLIP demonstrates impressive zero-shot capabilities in various downstream tasks that require holistic image understanding. However, due to the image-level contrastive learning and fully global feature interaction, ViT-based CLIP struggles to capture local details, resulting in poor performance in segmentation tasks. Our analysis of ViT-based CLIP reveals that anomaly tokens emerge during the forward process, attracting disproportionate attention from normal patch tokens and thereby diminishing spatial awareness. To address this issue, we propose Self-Calibrated CLIP (SC-CLIP), a training-free method that calibrates CLIP to generate finer representations while preserving its original generalization ability-without introducing new parameters or relying on additional backbones. Specifically, we mitigate the negative impact of anomaly tokens from two complementary perspectives. First, we explicitly identify the anomaly tokens and replace them based on local context. Second, we reduce their influence on normal tokens by enhancing feature discriminability and attention correlation, leveraging the inherent semantic consistency within CLIP's mid-level features. In addition, we introduce a two-pass strategy that effectively integrates multi-level features to enrich local details under the training-free setting. Together, these strategies enhance CLIP's feature representations with improved granularity and semantic coherence. Experimental results demonstrate the effectiveness of SC-CLIP, achieving state-of-the-art results across all datasets and surpassing previous methods by 9.5%. Notably, SC-CLIP boosts the performance of vanilla CLIP ViT-L/14 by 6.8 times. Our source code is available at this https URL.
>
---
#### [replaced 052] XtraLight-MedMamba for Classification of Neoplastic Tubular Adenomas
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.04819](https://arxiv.org/pdf/2602.04819)**

> **作者:** Aqsa Sultana; Rayan Afsar; Ahmed Rahu; Surendra P. Singh; Brian Shula; Brandon Combs; Derrick Forchetti; Vijayan K. Asari
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Accurate risk stratification of precancerous polyps during routine colonoscopy screening is a key strategy to reduce the incidence of colorectal cancer (CRC). However, assessment of low-grade dysplasia remains limited by subjective histopathologic interpretation. Advances in computational pathology and deep learning offer new opportunities to identify subtle, fine morphologic patterns associated with malignant progression that may be imperceptible to the human eye. In this work, we propose XtraLight-MedMamba, an ultra-lightweight state-space-based deep learning framework to classify neoplastic tubular adenomas from whole-slide images (WSIs). The architecture is a blend of a ConvNeXt-based shallow feature extractor with parallel vision mamba blocks to efficiently model local texture cues within global contextual structure. An integration of the Spatial and Channel Attention Bridge (SCAB) module enhances multiscale feature extraction, while the Fixed Non-Negative Orthogonal Classifier (FNOClassifier) enables substantial parameter reduction and improved generalization. The model was evaluated on a curated dataset acquired from patients with low-grade tubular adenomas, stratified into case and control cohorts based on subsequent CRC development. XtraLight-MedMamba achieved an accuracy of 97.18\% and an F1-score of 0.9767 using approximately 32,000 parameters, outperforming transformer-based and conventional Mamba architectures, which have significantly higher model complexity and computational burden, making it suitable for resource-constrained areas.
>
---
#### [replaced 053] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶轨迹生成任务，解决传统方法依赖离散锚点导致的效率与鲁棒性矛盾。提出MeanFuser，通过连续表示、均值流和自适应重建提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2602.20060](https://arxiv.org/pdf/2602.20060)**

> **作者:** Junli Wang; Yinan Zheng; Xueyi Liu; Zebin Xing; Pengfei Li; Guang Li; Kun Ma; Guang Chen; Hangjun Ye; Zhongpu Xia; Long Chen; Qichao Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Generative models have shown great potential in trajectory planning. Recent studies demonstrate that anchor-guided generative models are effective in modeling the uncertainty of driving behaviors and improving overall performance. However, these methods rely on discrete anchor vocabularies that must sufficiently cover the trajectory distribution during testing to ensure robustness, inducing an inherent trade-off between vocabulary size and model performance. To overcome this limitation, we propose MeanFuser, an end-to-end autonomous driving method that enhances both efficiency and robustness through three key designs. (1) We introduce Gaussian Mixture Noise (GMN) to guide generative sampling, enabling a continuous representation of the trajectory space and eliminating the dependency on discrete anchor vocabularies. (2) We adapt ``MeanFlow Identity" to end-to-end planning, which models the mean velocity field between GMN and trajectory distribution instead of the instantaneous velocity field used in vanilla flow matching methods, effectively eliminating numerical errors from ODE solvers and significantly accelerating inference. (3) We design a lightweight Adaptive Reconstruction Module (ARM) that enables the model to implicitly select from all sampled proposals or reconstruct a new trajectory when none is satisfactory via attention this http URL on the NAVSIM closed-loop benchmark demonstrate that MeanFuser achieves outstanding performance without the supervision of the PDM Score and exceptional inference efficiency, offering a robust and efficient solution for end-to-end autonomous driving. Our code and model are available at this https URL.
>
---
#### [replaced 054] IDESplat: Iterative Depth Probability Estimation for Generalizable 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.03824](https://arxiv.org/pdf/2601.03824)**

> **作者:** Wei Long; Haifeng Wu; Shiyin Jiang; Jinhua Zhang; Xinchun Ji; Shuhang Gu
>
> **摘要:** Generalizable 3D Gaussian Splatting aims to directly predict Gaussian parameters using a feed-forward network for scene reconstruction. Among these parameters, Gaussian means are particularly difficult to predict, so depth is usually estimated first and then unprojected to obtain the Gaussian sphere centers. Existing methods typically rely solely on a single warp to estimate depth probability, which hinders their ability to fully leverage cross-view geometric cues, resulting in unstable and coarse depth maps. To address this limitation, we propose IDESplat, which iteratively applies warp operations to boost depth probability estimation for accurate Gaussian mean prediction. First, to eliminate the inherent instability of a single warp, we introduce a Depth Probability Boosting Unit (DPBU) that integrates epipolar attention maps produced by cascading warp operations in a multiplicative manner. Next, we construct an iterative depth estimation process by stacking multiple DPBUs, progressively identifying potential depth candidates with high likelihood. As IDESplat iteratively boosts depth probability estimates and updates the depth candidates, the depth map is gradually refined, resulting in accurate Gaussian means. We conduct experiments on RealEstate10K, ACID, and DL3DV. IDESplat achieves outstanding reconstruction quality and state-of-the-art performance with real-time efficiency. On RE10K, it outperforms DepthSplat by 0.33 dB in PSNR, using only 10.7% of the parameters and 70% of the memory. Additionally, our IDESplat improves PSNR by 2.95 dB over DepthSplat on the DTU dataset in cross-dataset experiments, demonstrating its strong generalization ability.
>
---
#### [replaced 055] Graph Memory: A Structured and Interpretable Framework for Modality-Agnostic Embedding-Based Inference
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14961](https://arxiv.org/pdf/2511.14961)**

> **作者:** Artur A. Oliveira; Mateus Espadoto; Roberto M. Cesar Jr.; Roberto Hirata Jr
>
> **备注:** This version expands the published conference paper (VISAPP 2026) with additional methodological details, experiments, and analysis that were omitted due to page limits. The final published version is available via DOI: https://doi.org/10.5220/0014578800004084
>
> **摘要:** We introduce Graph Memory (GM), a structured non-parametric framework that represents an embedding space through a compact graph of reliability-annotated prototype regions. GM encodes local geometry and regional ambiguity through prototype relations and performs inference by diffusing query evidence across this structure, unifying instance retrieval, prototype-based reasoning, and graph diffusion within a single inductive and interpretable model. The framework is inherently modality-agnostic: in multimodal settings, independent prototype graphs are constructed for each modality and their calibrated predictions are combined through reliability-aware late fusion, enabling transparent integration of heterogeneous sources such as whole-slide images and gene-expression profiles. Experiments on synthetic benchmarks, breast histopathology (IDC), and the multimodal AURORA dataset show that GM matches or exceeds the accuracy of kNN and Label Spreading while providing substantially better calibration, smoother decision boundaries, and an order-of-magnitude smaller memory footprint. By explicitly modeling regional reliability and relational structure, GM offers a principled and interpretable approach to non-parametric inference across single- and multi-modal domains.
>
---
#### [replaced 056] HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23997](https://arxiv.org/pdf/2603.23997)**

> **作者:** Yumeng Liu; Xiao-Xiao Long; Marc Habermann; Xuanze Yang; Cheng Lin; Yuan Liu; Yuexin Ma; Wenping Wang; Ligang Liu
>
> **备注:** project page: this https URL
>
> **摘要:** Recovering high-fidelity 3D hand geometry from images is a critical task in computer vision, holding significant value for domains such as robotics, animation and VR/AR. Crucially, scalable applications demand both accuracy and deployment flexibility, requiring the ability to leverage massive amounts of unstructured image data from the internet or enable deployment on consumer-grade RGB cameras without complex calibration. However, current methods face a dilemma. While single-view approaches are easy to deploy, they suffer from depth ambiguity and occlusion. Conversely, multi-view systems resolve these uncertainties but typically demand fixed, calibrated setups, limiting their real-world utility. To bridge this gap, we draw inspiration from 3D foundation models that learn explicit geometry directly from visual data. By reformulating hand reconstruction from arbitrary views as a visual-geometry grounded task, we propose a feed-forward architecture that, for the first time in literature, jointly infers 3D hand meshes and camera poses from uncalibrated views. Extensive evaluations show that our approach outperforms state-of-the-art benchmarks and demonstrates strong generalization to uncalibrated, in-the-wild scenarios. Here is the link of our project page: this https URL.
>
---
#### [replaced 057] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24278](https://arxiv.org/pdf/2603.24278)**

> **作者:** Guan Luo; Xiu Li; Rui Chen; Xuanyu Yi; Jing Lin; Chia-Hao Chen; Jiahang Liu; Song-Hai Zhang; Jianfeng Zhang
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** The dominant paradigm for high-fidelity 3D generation relies on a VAE-Diffusion pipeline, where the VAE's reconstruction capability sets a firm upper bound on generation quality. A fundamental challenge limiting existing VAEs is the representation mismatch between ground-truth meshes and network predictions: GT meshes have arbitrary, variable topology, while VAEs typically predict fixed-structure implicit fields (\eg, SDF on regular grids). This inherent misalignment prevents establishing explicit mesh-level correspondences, forcing prior work to rely on indirect supervision signals such as SDF or rendering losses. Consequently, fine geometric details, particularly sharp features, are poorly preserved during reconstruction. To address this, we introduce TopoMesh, a sparse voxel-based VAE that unifies both GT and predicted meshes under a shared Dual Marching Cubes (DMC) topological framework. Specifically, we convert arbitrary input meshes into DMC-compliant representations via a remeshing algorithm that preserves sharp edges using an L$\infty$ distance metric. Our decoder outputs meshes in the same DMC format, ensuring that both predicted and target meshes share identical topological structures. This establishes explicit correspondences at the vertex and face level, allowing us to derive explicit mesh-level supervision signals for topology, vertex positions, and face orientations with clear gradients. Our sparse VAE architecture employs this unified framework and is trained with Teacher Forcing and progressive resolution training for stable and efficient convergence. Extensive experiments demonstrate that TopoMesh significantly outperforms existing VAEs in reconstruction fidelity, achieving superior preservation of sharp features and geometric details.
>
---
#### [replaced 058] PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18801](https://arxiv.org/pdf/2511.18801)**

> **作者:** Yichen Yang; Hong Li; Haodong Zhu; Linin Yang; Guojun Lei; Sheng Xu; Baochang Zhang
>
> **摘要:** Existing autoregressive (AR) methods for generating artist-designed meshes struggle to balance global structural consistency with high-fidelity local details, and are susceptible to error accumulation. To address this, we propose PartDiffuser, a novel semi-autoregressive diffusion framework for point-cloud-to-mesh generation. The method first performs semantic segmentation on the mesh and then operates in a "part-wise" manner: it employs autoregression between parts to ensure global topology, while utilizing a parallel discrete diffusion process within each semantic part to precisely reconstruct high-frequency geometric features. PartDiffuser is based on the DiT architecture and introduces a part-aware cross-attention mechanism, using point clouds as hierarchical geometric conditioning to dynamically control the generation process, thereby effectively decoupling the global and local generation tasks. Experiments demonstrate that this method significantly outperforms state-of-the-art (SOTA) models in generating 3D meshes with rich detail, exhibiting exceptional detail representation suitable for real-world applications.
>
---
#### [replaced 059] MoLingo: Motion-Language Alignment for Text-to-Motion Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13840](https://arxiv.org/pdf/2512.13840)**

> **作者:** Yannan He; Garvita Tiwari; Xiaohan Zhang; Pankaj Bora; Tolga Birdal; Jan Eric Lenssen; Gerard Pons-Moll
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** We introduce MoLingo, a text-to-motion (T2M) model that generates realistic, lifelike human motion by denoising in a continuous latent space. Recent works perform latent space diffusion, either on the whole latent at once or auto-regressively over multiple latents. In this paper, we study how to make diffusion on continuous motion latents work best. We focus on two questions: (1) how to build a semantically aligned latent space so diffusion becomes more effective, and (2) how to best inject text conditioning so the motion follows the description closely. We propose a semantic-aligned motion encoder trained with frame-level text labels so that latents with similar text meaning stay close, which makes the latent space more diffusion-friendly. We also compare single-token conditioning with a multi-token cross-attention scheme and find that cross-attention gives better motion realism and text-motion alignment. With semantically aligned latents, auto-regressive generation, and cross-attention text conditioning, our model sets a new state of the art in human motion generation on standard metrics and in a user study. We will release our code and models for further research and downstream usage.
>
---
#### [replaced 060] WiT: Waypoint Diffusion Transformers via Trajectory Conflict Navigation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15132](https://arxiv.org/pdf/2603.15132)**

> **作者:** Hainuo Wang; Mingjia Li; Xiaojie Guo
>
> **摘要:** While recent Flow Matching models avoid the reconstruction bottlenecks of latent autoencoders by operating directly in pixel space, the lack of semantic continuity in the pixel manifold severely intertwines optimal transport paths. This induces severe trajectory conflicts near intersections, yielding sub-optimal solutions. Rather than bypassing this issue via information-lossy latent representations, we directly untangle the pixel-space trajectories by proposing Waypoint Diffusion Transformers (WiT). WiT factorizes the continuous vector field via intermediate semantic waypoints projected from pre-trained vision models. It effectively disentangles the generation trajectories by breaking the optimal transport into prior-to-waypoint and waypoint-to-pixel segments. Specifically, during the iterative denoising process, a lightweight generator dynamically infers these intermediate waypoints from the current noisy state. They then continuously condition the primary diffusion transformer via the Just-Pixel AdaLN mechanism, steering the evolution towards the next state, ultimately yielding the final RGB pixels. Evaluated on ImageNet 256x256, WiT beats strong pixel-space baselines, accelerating JiT training convergence by 2.2x. Code will be publicly released at this https URL.
>
---
#### [replaced 061] Diffusion Probe: Generated Image Result Prediction Using CNN Probes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23783](https://arxiv.org/pdf/2602.23783)**

> **作者:** Benlei Cui; Bukun Huang; Zhizeng Ye; Xuemei Dong; Tuo Chen; Hui Xue; Dingkang Yang; Longtao Huang; Jingqun Tang; Haiwen Hong
>
> **备注:** CVPR 2026
>
> **摘要:** Text-to-image (T2I) diffusion models lack an efficient mechanism for early quality assessment, leading to costly trial-and-error in multi-generation scenarios such as prompt iteration, agent-based generation, and flow-grpo. We reveal a strong correlation between early diffusion cross-attention distributions and final image quality. Based on this finding, we introduce Diffusion Probe, a framework that leverages internal cross-attention maps as predictive signals. We design a lightweight predictor that maps statistical properties of early-stage cross-attention extracted from initial denoising steps to the final image's overall quality. This enables accurate forecasting of image quality across diverse evaluation metrics long before full synthesis is complete. We validate Diffusion Probe across a wide range of settings. On multiple T2I models, across early denoising windows, resolutions, and quality metrics, it achieves strong correlation (PCC > 0.7) and high classification performance (AUC-ROC > 0.9). Its reliability translates into practical gains. By enabling early quality-aware decisions in workflows such as prompt optimization, seed selection, and accelerated RL training, the probe supports more targeted sampling and avoids computation on low-potential generations. This reduces computational overhead while improving final output this http URL Probe is model-agnostic, efficient, and broadly applicable, offering a practical solution for improving T2I generation efficiency through early quality prediction.
>
---
#### [replaced 062] Towards Mitigating Modality Bias in Vision-Language Models for Temporal Action Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.21078](https://arxiv.org/pdf/2601.21078)**

> **作者:** Jiaqi Li; Guangming Wang; Shuntian Zheng; Minzhe Ni; Xiaoman Lu; Guanghui Ye; Yu Guan
>
> **摘要:** Temporal Action Localization (TAL) requires identifying both the boundaries and categories of actions in untrimmed videos. While vision-language models (VLMs) offer rich semantics to complement visual evidence, existing approaches tend to overemphasize linguistic priors at the expense of visual performance, leading to a pronounced modality bias. We propose ActionVLM, a vision-language aggregation framework that systematically mitigates modality bias in TAL. Our key insight is to preserve vision as the dominant signal while adaptively exploiting language only when beneficial. To this end, we introduce (i) a debiasing reweighting module that estimates the language advantage-the incremental benefit of language over vision-only predictions-and dynamically reweights language modality accordingly, and (ii) a residual aggregation strategy that treats language as a complementary refinement rather than the primary driver. This combination alleviates modality bias, reduces overconfidence from linguistic priors, and strengthens temporal reasoning. Experiments on THUMOS14 show that our model outperforms state-of-the-art by up to 3.2% mAP.
>
---
#### [replaced 063] Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的时序理解任务，旨在评估模型对视频时间方向的判断能力。研究提出一个心理物理基准，发现现有模型在时间因果推理上表现不佳。**

- **链接: [https://arxiv.org/pdf/2510.26241](https://arxiv.org/pdf/2510.26241)**

> **作者:** Shiho Matta; Lis Kanashiro Pereira; Peitao Han; Fei Cheng; Shigeru Kitazawa
>
> **备注:** 12 pages
>
> **摘要:** Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and has not been adequately evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best model lags far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.
>
---
#### [replaced 064] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01010](https://arxiv.org/pdf/2603.01010)**

> **作者:** Xuqin Wang; Tao Wu; Yanfeng Zhang; Lu Liu; Mingwei Sun; Yongliang Wang; Niclas Zeller; Daniel Cremers
>
> **备注:** Accepted by CVPR 2026; Project Page see this https URL
>
> **摘要:** Recent advances in generative modeling have substantially enhanced novel view synthesis, yet maintaining consistency across viewpoints remains challenging. Diffusion-based models rely on stochastic noise-to-data transitions, which obscure deterministic structures and yield inconsistent view predictions. We advocate a Data-to-Data Flow Matching framework that learns deterministic transformations between paired views, enhancing view-consistent synthesis through explicit data coupling. Building on this, we propose Probability Density Geodesic Flow Matching (PDG-FM), which aligns interpolation trajectories with density-based geodesics of a data manifold. To enable tractable geodesic estimation, we employ a teacher-student framework that distills density-based geodesic interpolants into an efficient ambient-space predictor. Empirically, our method surpasses diffusion-based baselines on Objaverse and GSO30 datasets, demonstrating improved structural coherence and smoother transitions across views. These results highlight the advantages of incorporating data-dependent geometric regularization into deterministic flow matching for consistent novel view generation.
>
---
#### [replaced 065] Generative deep learning for foundational video translation in ultrasound
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.03255](https://arxiv.org/pdf/2511.03255)**

> **作者:** Nikolina Tomic; Roshni Bhatnagar; Sarthak Jain; Connor Lau; Tien-Yu Liu; Laura Gambini; Rima Arnaout
>
> **摘要:** Deep learning (DL) has the potential to revolutionize image acquisition and interpretation across medicine, however, attention to data imbalance and missingness is required. Ultrasound data presents a particular challenge because in addition to different views and structures, it includes several sub-modalities-such as greyscale and color flow doppler (CFD)-that are often imbalanced in clinical studies. Image translation can help balance datasets but is challenging for ultrasound sub-modalities to date. Here, we present a generative method for ultrasound CFD-greyscale video translation, trained on 54,975 videos and tested on 8,368. The method developed leveraged pixel-wise, adversarial, and perceptual loses and utilized two networks: one for reconstructing anatomic structures and one for denoising to achieve realistic ultrasound imaging. Average pairwise SSIM between synthetic videos and ground truth was 0.91+/-0.04. Synthetic videos performed indistinguishably from real ones in DL classification and segmentation tasks and when evaluated by blinded clinical experts: F1 score was 0.9 for real and 0.89 for synthetic videos; Dice score between real and synthetic segmentation was 0.97. Overall clinician accuracy in distinguishing real vs synthetic videos was 54+/-6% (42-61%), indicating realistic synthetic videos. Although trained only on heart videos, the model worked well on ultrasound spanning several clinical domains (average SSIM 0.91+/-0.05), demonstrating foundational abilities. Together, these data expand the utility of retrospectively collected imaging and augment the dataset design toolbox for medical imaging.
>
---
#### [replaced 066] SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22893](https://arxiv.org/pdf/2603.22893)**

> **作者:** Zhicheng Qiu; Jiarui Meng; Tong-an Luo; Yican Huang; Xuan Feng; Xuanfu Li; ZHan Xu
>
> **摘要:** We propose SLARM, a feed-forward model that unifies dynamic scene reconstruction, semantic understanding, and real-time streaming inference. SLARM captures complex, non-uniform motion through higher-order motion modeling, trained solely on differentiable renderings without any flow supervision. Besides, SLARM distills semantic features from LSeg to obtain language-aligned representations. This design enables semantic querying via natural language, and the tight coupling between semantics and geometry further enhances the accuracy and robustness of dynamic reconstruction. Moreover, SLARM processes image sequences using window-based causal attention, achieving stable, low-latency streaming inference without accumulating memory cost. Within this unified framework, SLARM achieves state-of-the-art results in dynamic estimation, rendering quality, and scene parsing, improving motion accuracy by 21%, reconstruction PSNR by 1.6 dB, and segmentation mIoU by 20% over existing methods.
>
---
#### [replaced 067] A User-Friendly Framework for Generating Model-Preferred Prompts in Text-to-Image Synthesis
- **分类: cs.MM; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2402.12760](https://arxiv.org/pdf/2402.12760)**

> **作者:** Nailei Hei; Qianyu Guo; Zihao Wang; Yan Wang; Haofen Wang; Wenqiang Zhang
>
> **备注:** Accepted by The 38th Annual AAAI Conference on Artificial Intelligence (AAAI 2024)
>
> **摘要:** Well-designed prompts have demonstrated the potential to guide text-to-image models in generating amazing images. Although existing prompt engineering methods can provide high-level guidance, it is challenging for novice users to achieve the desired results by manually entering prompts due to a discrepancy between novice-user-input prompts and the model-preferred prompts. To bridge the distribution gap between user input behavior and model training datasets, we first construct a novel Coarse-Fine Granularity Prompts dataset (CFP) and propose a novel User-Friendly Fine-Grained Text Generation framework (UF-FGTG) for automated prompt optimization. For CFP, we construct a novel dataset for text-to-image tasks that combines coarse and fine-grained prompts to facilitate the development of automated prompt generation methods. For UF-FGTG, we propose a novel framework that automatically translates user-input prompts into model-preferred prompts. Specifically, we propose a prompt refiner that continually rewrites prompts to empower users to select results that align with their unique needs. Meanwhile, we integrate image-related loss functions from the text-to-image model into the training process of text generation to generate model-preferred prompts. Additionally, we propose an adaptive feature extraction module to ensure diversity in the generated results. Experiments demonstrate that our approach is capable of generating more visually appealing and diverse images than previous state-of-the-art methods, achieving an average improvement of 5% across six quality and aesthetic metrics.
>
---
#### [replaced 068] Diffusion Forcing for Multi-Agent Interaction Sequence Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多智能体交互序列建模任务，旨在解决复杂多人互动生成问题。提出MAGNet框架，实现灵活的多智能体运动生成与协调。**

- **链接: [https://arxiv.org/pdf/2512.17900](https://arxiv.org/pdf/2512.17900)**

> **作者:** Vongani H. Maluleke; Kie Horiuchi; Lea Wilken; Evonne Ng; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project page: this https URL ; Code: this https URL
>
> **摘要:** Understanding and generating multi-person interactions is a fundamental challenge with broad implications for robotics and social computing. While humans naturally coordinate in groups, modeling such interactions remains difficult due to long temporal horizons, strong inter-agent dependencies, and variable group sizes. Existing motion generation methods are largely task-specific and do not generalize to flexible multi-agent generation. We introduce MAGNet (Multi-Agent Generative Network), a unified autoregressive diffusion framework for multi-agent motion generation that supports a wide range of interaction tasks through flexible conditioning and sampling. MAGNet performs dyadic and polyadic prediction, partner inpainting, partner prediction, and agentic generation all within a single model, and can autoregressively generate ultra-long sequences spanning hundreds of motion steps. We explicitly model inter-agent coupling during autoregressive denoising, enabling coherent coordination across agents. As a result, MAGNet captures both tightly synchronized activities (e.g., dancing, boxing) and loosely structured social interactions. Our approach performs on par with specialized methods on dyadic benchmarks while naturally extending to polyadic scenarios involving three or more interacting people. Please watch the supplemental video, where the temporal dynamics and spatial coordination of generated interactions are best appreciated. Project page: this https URL
>
---
#### [replaced 069] Hyper-Connections for Adaptive Multi-Modal MRI Brain Tumor Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19844](https://arxiv.org/pdf/2603.19844)**

> **作者:** Lokendra Kumar; Shubham Aggarwal
>
> **备注:** 29 pages,6 tables,17 figures
>
> **摘要:** We present the first study of Hyper-Connections (HC) for volumetric multi-modal brain tumor segmentation, integrating them as a drop-in replacement for fixed residual connections across five architectures: nnU-Net, SwinUNETR, VT-UNet, U-Net, and U-Netpp. Dynamic HC consistently improves all 3D models on the BraTS 2021 dataset, yielding up to +1.03 percent mean Dice gain with negligible parameter overhead. Gains are most pronounced in the Enhancing Tumor sub-region, reflecting improved fine-grained boundary delineation. Modality ablation further reveals that HC-equipped models develop sharper sensitivity toward clinically dominant sequences, specifically T1ce for Tumor Core and Enhancing Tumor, and FLAIR for Whole Tumor, a behavior absent in fixed-connection baselines and consistent across all architectures. In 2D settings, improvements are smaller and configuration-sensitive, suggesting that volumetric spatial context amplifies the benefit of adaptive aggregation. These results establish HC as a simple, efficient, and broadly applicable mechanism for multi-modal feature fusion in medical image segmentation.
>
---
#### [replaced 070] EchoTorrent: Towards Swift, Sustained, and Streaming Multi-Modal Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13669](https://arxiv.org/pdf/2602.13669)**

> **作者:** Rang Meng; Yingjie Yin; Yuming Li; Chenguang Ma
>
> **摘要:** Recent multi-modal video generation models have achieved high visual quality, but their prohibitive latency and limited temporal stability hinder real-time deployment. Streaming inference exacerbates these issues, leading to pronounced multimodal degradation, such as spatial blurring, temporal drift, and lip desynchronization, which creates an unresolved efficiency-performance trade-off. To this end, we propose EchoTorrent, a novel schema with a fourfold design: (1) Multi-Teacher Training fine-tunes a pre-trained model on distinct preference domains to obtain specialized domain experts, which sequentially transfer domain-specific knowledge to a student model; (2) Adaptive CFG Calibration (ACC-DMD), which calibrates the audio CFG augmentation errors in DMD via a phased spatiotemporal schedule, eliminating redundant CFG computations and enabling single-pass inference per step; (3) Hybrid Long Tail Forcing, which enforces alignment exclusively on tail frames during long-horizon self-rollout training via a causal-bidirectional hybrid architecture, effectively mitigates spatiotemporal degradation in streaming mode while enhancing fidelity to reference frames; and (4) VAE Decoder Refiner through pixel-domain optimization of the VAE decoder to recover high-frequency details while circumventing latent-space ambiguities. Extensive experiments and analysis demonstrate that EchoTorrent achieves few-pass autoregressive generation with substantially extended temporal consistency, identity preservation, and audio-lip synchronization.
>
---
#### [replaced 071] AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.01061](https://arxiv.org/pdf/2506.01061)**

> **作者:** Dahyeon Kye; Changhyun Roh; Sukhun Ko; Chanho Eom; Jihyong Oh
>
> **备注:** Accepted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). Please visit our project page at this https URL
>
> **摘要:** Video Frame Interpolation (VFI) is a core low-level vision task that synthesizes intermediate frames between existing ones while ensuring spatial and temporal coherence. Over the past decades, VFI methodologies have evolved from classical motion compensation-based approach to a wide spectrum of deep learning-based approaches, including kernel-, flow-, hybrid-, phase-, GAN-, Transformer-, Mamba-, and most recently, diffusion-based models. We introduce AceVFI, a comprehensive and up-to-date review of the VFI field, covering over 250 representative papers. We systematically categorize VFI methods based on their core design principles and architectural characteristics. Further, we classify them into two major learning paradigms: Center-Time Frame Interpolation (CTFI) and Arbitrary-Time Frame Interpolation (ATFI). We analyze key challenges in VFI, including large motion, occlusion, lighting variation, and non-linear motion. In addition, we review standard datasets, loss functions, evaluation metrics. We also explore VFI applications in other domains and highlight future research directions. This survey aims to serve as a valuable reference for researchers and practitioners seeking a thorough understanding of the modern VFI landscape.
>
---
#### [replaced 072] ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.10800](https://arxiv.org/pdf/2507.10800)**

> **作者:** Ali Hojjat; Janek Haberer; Soren Pirk; Olaf Landsiedel
>
> **备注:** Accepted at CVPR'26, please cite the conference version
>
> **摘要:** ViTs deliver SOTA performance, yet their fixed computational budget prevents scalable deployment across heterogeneous hardware. Recent Matryoshka-style Transformer architectures mitigate this by embedding nested subnetworks within a single model to enable scalable inference. However, these models allocate the same amount of compute to all inputs, regardless of their complexity, which leads to inefficiencies. To address this, we introduce ThinkingViT, a nested ViT architecture that employs progressive thinking stages to dynamically adjust inference computation based on input difficulty. ThinkingViT first activates a small subset of the most important attention heads to produce an initial prediction. If the prediction confidence exceeds a predefined threshold, inference terminates early. Otherwise, within the same backbone, it activates a larger subset of attention heads and conducts a new forward pass. This process continues iteratively until the model reaches the predefined confidence level or exhausts its maximum capacity. To boost the performance of subsequent rounds, we introduce a Token Recycling approach that fuses the input embeddings with the embeddings from the previous stage. Experiments show that ThinkingViT surpasses nested baselines by up to 2.0 percentage points (p.p.) in accuracy at the same throughput and by up to 2.9 p.p. at equal GMACs on ImageNet-1K. We show that the backbone-preserving design of ThinkingViT allows it to serve as a plug-in upgrade for ViTs in downstream tasks such as semantic segmentation. We also demonstrate that ThinkingViT transfers effectively to other architectures such as Swin Transformers. The source code is available at this https URL.
>
---
#### [replaced 073] OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.05631](https://arxiv.org/pdf/2507.05631)**

> **作者:** Zhiwei Chen; Yupeng Hu; Zixu Li; Zhiheng Fu; Xuemeng Song; Liqiang Nie
>
> **摘要:** Composed Image Retrieval (CIR) represents a novel retrieval paradigm that is capable of expressing users' intricate retrieval requirements flexibly. It enables the user to give a multimodal query, comprising a reference image and a modification text, and subsequently retrieve the target image. Notwithstanding the considerable advances made by prevailing methodologies, CIR remains in its nascent stages due to two limitations: 1) inhomogeneity between dominant and noisy portions in visual data is ignored, leading to query feature degradation, and 2) the priority of textual data in the image modification process is overlooked, which leads to a visual focus bias. To address these two limitations, this work presents a focus mapping-based feature extractor, which consists of two modules: dominant portion segmentation and dual focus mapping. It is designed to identify significant dominant portions in images and guide the extraction of visual and textual data features, thereby reducing the impact of noise interference. Subsequently, we propose a textually guided focus revision module, which can utilize the modification requirements implied in the text to perform adaptive focus revision on the reference image, thereby enhancing the perception of the modification focus on the composed features. The aforementioned modules collectively constitute the segmentatiOn-based Focus shiFt reviSion nETwork (\mbox{OFFSET}), and comprehensive experiments on four benchmark datasets substantiate the superiority of our proposed method. The codes and data are available on this https URL
>
---
#### [replaced 074] Structure Causal Models and LLMs Integration in Medical Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.02703](https://arxiv.org/pdf/2505.02703)**

> **作者:** Zibo Xu; Qiang Li; Weizhi Nie; Weijie Wang; Anan Liu
>
> **备注:** Accepted by IEEE TMI 2025
>
> **摘要:** Medical Visual Question Answering (MedVQA) aims to answer medical questions according to medical images. However, the complexity of medical data leads to confounders that are difficult to observe, so bias between images and questions is inevitable. Such cross-modal bias makes it challenging to infer medically meaningful answers. In this work, we propose a causal inference framework for the MedVQA task, which effectively eliminates the relative confounding effect between the image and the question to ensure the precision of the question-answering (QA) session. We are the first to introduce a novel causal graph structure that represents the interaction between visual and textual elements, explicitly capturing how different questions influence visual features. During optimization, we apply the mutual information to discover spurious correlations and propose a multi-variable resampling front-door adjustment method to eliminate the relative confounding effect, which aims to align features based on their true causal relevance to the question-answering task. In addition, we also introduce a prompt strategy that combines multiple prompt forms to improve the model's ability to understand complex medical data and answer accurately. Extensive experiments on three MedVQA datasets demonstrate that 1) our method significantly improves the accuracy of MedVQA, and 2) our method achieves true causal correlations in the face of complex medical data.
>
---
#### [replaced 075] Inferring Compositional 4D Scenes without Ever Seeing One
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05272](https://arxiv.org/pdf/2512.05272)**

> **作者:** Ahmet Berke Gokmen; Ajad Chhatkuli; Luc Van Gool; Danda Pani Paudel
>
> **备注:** Project page: this https URL
>
> **摘要:** Scenes in the real world are often composed of several static and dynamic objects. Capturing their 4-dimensional structures, composition and spatio-temporal configuration in-the-wild, though extremely interesting, is equally hard. Therefore, existing works often focus on one object at a time, while relying on some category-specific parametric shape model for dynamic objects. This can lead to inconsistent scene configurations, in addition to being limited to the modeled object categories. We propose COM4D (Compositional 4D), a method that consistently and jointly predicts the structure and spatio-temporal configuration of 4D/3D objects using only static multi-object or dynamic single object supervision. We achieve this by a carefully designed training of spatial and temporal attentions on 2D video input. The training is disentangled into learning from object compositions on the one hand, and single object dynamics throughout the video on the other, thus completely avoiding reliance on 4D compositional training data. At inference time, our proposed attention mixing mechanism combines these independently learned attentions, without requiring any 4D composition examples. By alternating between spatial and temporal reasoning, COM4D reconstructs complete and persistent 4D scenes with multiple interacting objects directly from monocular videos. Furthermore, COM4D provides state-of-the-art results in existing separate problems of 4D object and composed 3D reconstruction despite being purely data-driven.
>
---
#### [replaced 076] 360° Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.16179](https://arxiv.org/pdf/2603.16179)**

> **作者:** Huyen T. T. Tran; Van-Quang Nguyen; Farros Alferro; Kang-Jun Liu; Takayuki Okatani
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown impressive abilities in understanding and reasoning over conventional images. However, their perception of 360° images remains largely underexplored. Unlike conventional images, 360° images capture the entire surrounding environment, enabling holistic spatial reasoning but introducing challenges such as geometric distortion and complex spatial relations. To comprehensively assess MLLMs' capabilities to perceive 360° images, we introduce 360Bench, a Visual Question Answering (VQA) benchmark featuring 7K-resolution 360° images, seven representative (sub)tasks with annotations carefully curated by human annotators. Using 360Bench, we systematically evaluate seven MLLMs and six enhancement methods, revealing their shortcomings in 360° image perception. To address these challenges, we propose Free360, a training-free scene-graph-based framework for high-resolution 360° VQA. Free360 decomposes the reasoning process into modular steps, applies adaptive spherical image transformations to 360° images tailored to each step, and seamlessly integrates the resulting information into a unified graph representation for answer generation. Experiments show that Free360 consistently improves its base MLLM and provides a strong training-free solution for 360° VQA tasks. The source code and dataset will be publicly released upon acceptance.
>
---
#### [replaced 077] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.04733](https://arxiv.org/pdf/2603.04733)**

> **作者:** Xingyu Wang; Tao Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Test-Time Adaptation (TTA) is essential for enabling deep learning models to handle real-world data distribution shifts. However, current approaches face significant limitations: backpropagation-based methods are not suitable for low-end deployment devices, due to their high computation and memory requirements, as well as their tendency to modify model weights during adaptation; while traditional backpropagation-free techniques exhibit constrained adaptation capabilities. In this work, we propose Forward-Only Zeroth-Order Optimization (FOZO), a novel and practical backpropagation-free paradigm for TTA. FOZO leverages a memory-efficient zeroth-order prompt optimization, which is led by objectives optimizing both intermediate feature statistics and prediction entropy. To ensure efficient and stable adaptation over the out-of-distribution data stream, we introduce a dynamically decaying perturbation scale during zeroth-order gradient estimation and theoretically prove its convergence under the TTA data stream assumption. Extensive continual adaptation experiments on ImageNet-C, ImageNet-R, and ImageNet-Sketch demonstrate FOZO's superior performance, achieving 59.52% Top-1 accuracy on ImageNet-C (5K, level 5) and outperforming main gradient-based methods and SOTA forward-only FOA (58.13%). Furthermore, FOZO exhibits strong generalization on quantized (INT8) models. These findings demonstrate that FOZO is a highly competitive solution for TTA deployment in resource-limited scenarios.
>
---
#### [replaced 078] The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉识别任务，旨在解决开放源代码大语言模型在层次化视觉识别中的瓶颈问题。研究发现模型缺乏层次化知识，通过构建VQA任务验证并提出解决方案。**

- **链接: [https://arxiv.org/pdf/2505.24840](https://arxiv.org/pdf/2505.24840)**

> **作者:** Yuwen Tan; Yuan Qing; Boqing Gong
>
> **备注:** Accepted to CVPR 2026. Project page and code: this https URL
>
> **摘要:** This paper reveals that many open-source large language models (LLMs) lack hierarchical knowledge about our visual world, unaware of even well-established biology taxonomies. This shortcoming makes LLMs a bottleneck for vision LLMs' hierarchical visual recognition (e.g., recognizing Anemone Fish but not Vertebrate). We arrive at these findings using about one million four-choice visual question answering (VQA) tasks constructed from six taxonomies and four image datasets. Interestingly, finetuning a vision LLM using our VQA tasks reaffirms LLMs' bottleneck effect because the VQA tasks improve the LLMs' hierarchical consistency more than the vision LLMs'. We conjecture that one cannot make open-source vision LLMs understand visual concepts hierarchically until LLMs possess corresponding taxonomy knowledge.
>
---
#### [replaced 079] ShowMak3r: Compositional TV Show Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.19584](https://arxiv.org/pdf/2504.19584)**

> **作者:** Sangmin Kim; Seunguk Do; Jaesik Park
>
> **备注:** Project page : this https URL
>
> **摘要:** Reconstructing dynamic radiance fields from video clips is challenging, especially when entertainment videos like TV shows are given. Many challenges make the reconstruction difficult due to (1) actors occluding with each other and having diverse facial expressions, (2) cluttered stages, and (3) small baseline views or sudden shot changes. To address these issues, we present ShowMak3r, a comprehensive reconstruction pipeline that allows the editing of scenes like how video clips are made in a production control room. In ShowMak3r, a 3DLocator module locates recovered actors on the stage using depth prior and estimates unseen human poses via interpolation. The proposed ShotMatcher module then tracks the actors under shot changes. Furthermore, ShowMak3r introduces a face-fitting network that dynamically recovers the actors' expressions. Experiments on Sitcoms3D dataset show that our pipeline can reassemble TV show scenes with new cameras at different timestamps. We also demonstrate that ShowMak3r enables interesting applications such as synthetic shot-making, actor relocation, insertion, deletion, and pose manipulation. Project page : this https URL
>
---
#### [replaced 080] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.06663](https://arxiv.org/pdf/2603.06663)**

> **作者:** Giacomo Frisoni; Lorenzo Molfetta; Mattia Buzzoni; Gianluca Moro
>
> **备注:** Please cite the definitive, copyrighted, and peer-reviewed version of this article published in AAAI 2026, edited by Sven Koenig et al., AAAI Press, Vol. 40, No. 36, Technical Track, pp. 30726-30734, 2026. DOI: this https URL
>
> **摘要:** Recent advances in training-free visual prompting, such as Set-of-Mark, have emerged as a promising direction for enhancing the grounding capabilities of multimodal language models (MLMs). These techniques operate by partitioning the input image into object regions and annotating them with marks, predominantly boxes with numeric identifiers, before feeding the augmented image to the MLM. However, these approaches treat marked objects as isolated entities, failing to capture the relationships between them. On these premises, we propose Graph-of-Mark (GoM), the first pixel-level visual prompting technique that overlays scene graphs onto the input image for spatial reasoning tasks. We evaluate GoM across 3 open-source MLMs and 4 different datasets, conducting extensive ablations on drawn components and investigating the impact of auxiliary graph descriptions in the text prompt. Our results demonstrate that GoM consistently improves the zero-shot capability of MLMs in interpreting object positions and relative directions, improving base accuracy in visual question answering and localization up to 11 percentage points.
>
---
#### [replaced 081] Thinking with Frames: Generative Video Distortion Evaluation via Frame Reward Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.04033](https://arxiv.org/pdf/2601.04033)**

> **作者:** Yuan Wang; Borui Liao; Huijuan Huang; Jinda Lu; Ouxiang Li; Kuien Liu; Meng Wang; Xiang Wang
>
> **摘要:** Recent advances in video reward models and post-training strategies have improved text-to-video (T2V) generation. While these models typically assess visual quality, motion quality, and text alignment, they often overlook key structural distortions, such as abnormal object appearances and interactions, which can degrade the overall quality of the generative video. To address this gap, we introduce REACT, a frame-level reward model designed specifically for structural distortions evaluation in generative videos. REACT assigns point-wise scores and attribution labels by reasoning over video frames, focusing on recognizing distortions. To support this, we construct a large-scale human preference dataset, annotated based on our proposed taxonomy of structural distortions, and generate additional data using a efficient Chain-of-Thought (CoT) synthesis pipeline. REACT is trained with a two-stage framework: (1) supervised fine-tuning with masked loss for domain knowledge injection, followed by (2) reinforcement learning with Group Relative Policy Optimization (GRPO) and pairwise rewards to enhance reasoning capability and align output scores with human preferences. During inference, a dynamic sampling mechanism is introduced to focus on frames most likely to exhibit distortion. We also present REACT-Bench, a benchmark for generative video distortion evaluation. Experimental results demonstrate that REACT complements existing reward models in assessing structutal distortion, achieving both accurate quantitative evaluations and interpretable attribution analysis.
>
---
#### [replaced 082] 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23042](https://arxiv.org/pdf/2512.23042)**

> **作者:** Ryousuke Yamada; Kohsuke Ide; Yoshihiro Fukuhara; Hirokatsu Kataoka; Gilles Puy; Andrei Bursuc; Yuki M. Asano
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Despite recent progress in 3D self-supervised learning, collecting large-scale 3D scene scans remains expensive and labor-intensive. In this work, we investigate whether 3D representations can be learned from unlabeled videos recorded without any real 3D sensors. We present Laplacian-Aware Multi-level 3D Clustering with Sinkhorn-Knopp (LAM3C), a self-supervised framework that learns from video-generated point clouds reconstructed from unlabeled videos. We first introduce RoomTours, a video-generated point cloud dataset constructed by collecting room-walkthrough videos from the web (e.g., real-estate tours) and generating 49,219 scenes using an off-the-shelf feed-forward reconstruction model. We also propose a noise-regularized loss that stabilizes representation learning by enforcing local geometric smoothness and ensuring feature stability under noisy point clouds. Remarkably, without using any real 3D scans, LAM3C achieves better performance than previous self-supervised methods on indoor semantic and instance segmentation. These results suggest that unlabeled videos represent an abundant source of data for 3D self-supervised learning. Our source code is available at this https URL.
>
---
#### [replaced 083] Unified Camera Positional Encoding for Controlled Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07237](https://arxiv.org/pdf/2512.07237)**

> **作者:** Cheng Zhang; Boying Li; Meng Wei; Yan-Pei Cao; Camilo Cruz Gambardella; Dinh Phung; Jianfei Cai
>
> **备注:** Camera Ready of CVPR2026. Project Page: this https URL Code: this https URL
>
> **摘要:** Transformers have emerged as a universal backbone across 3D perception, video generation, and world models for autonomous driving and embodied AI, where understanding camera geometry is essential for grounding visual observations in three-dimensional space. However, existing camera encoding methods often rely on simplified pinhole assumptions, restricting generalization across the diverse intrinsics and lens distortions in real-world cameras. We introduce Relative Ray Encoding, a geometry-consistent representation that unifies complete camera information, including 6-DoF poses, intrinsics, and lens distortions. To evaluate its capability under diverse controllability demands, we adopt camera-controlled text-to-video generation as a testbed task. Within this setting, we further identify pitch and roll as two components effective for Absolute Orientation Encoding, enabling full control over the initial camera orientation. Together, these designs form UCPE (Unified Camera Positional Encoding), which integrates into a pretrained video Diffusion Transformer through a lightweight spatial attention adapter, adding less than 1% trainable parameters while achieving state-of-the-art camera controllability and visual fidelity. To facilitate systematic training and evaluation, we construct a large video dataset covering a wide range of camera motions and lens types. Extensive experiments validate the effectiveness of UCPE in camera-controllable video generation and highlight its potential as a general camera representation for Transformers across future multi-view, video, and 3D tasks. Code will be available at this https URL.
>
---
#### [replaced 084] SPR-128K: A New Benchmark for Spatial Plausibility Reasoning with Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23265](https://arxiv.org/pdf/2505.23265)**

> **作者:** Zhiyuan Hu; Zheng Sun; Yi Wei; Long Yu
>
> **摘要:** The performance of image generation has been significantly improved in recent years. However, the study of image screening is rare, and its performance with Multimodal Large Language Models (MLLMs) is unsatisfactory due to the lack of data and the weak spatial plausibility reasoning ability in MLLMs. In this work, we propose a complete solution to address these problems in terms of data and methodology. For data, we collect a comprehensive spatial plausibility reasoning (SPR) dataset with over 128k samples, called SPR-128K. The dataset evaluates spatial plausibility reasoning ability under four aspects. Regarding data annotation, we investigate multiple approaches to acquire high-quality Chain-of-Thought (CoT) data in the most cost-effective manner. Methodologically, we introduce a Dynamic Proportional Accuracy (DPA) reward into the Group Relative Policy Optimization (GRPO) framework, called DPA-GRPO. This enhanced method demonstrates superior performance compared to the original GRPO. Our experiments reveal that even leading MLLMs exhibit unsatisfactory performance in spatial plausibility reasoning. In contrast, our much smaller model, leveraging DPA-GRPO, substantially surpasses both large open-source and leading closed-source models.
>
---
#### [replaced 085] Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03904](https://arxiv.org/pdf/2603.03904)**

> **作者:** Augustin Borne; Pierre Notin; Christophe Hennequin; Sebastien Changey; Stephane Bazeille; Christophe Cudel; Franz Quint
>
> **摘要:** Object tracking from Unmanned Aerial Vehicles (UAVs) is challenged by platform dynamics, camera motion, and limited onboard resources. Existing visual trackers either lack robustness in complex scenarios or are too computationally demanding for real-time embedded use. We propose an Modular Asynchronous Tracking Architecture (MATA) that combines a transformer-based tracker with an Extended Kalman Filter, integrating ego-motion compensation from sparse optical flow and an object trajectory model. We further introduce a hardware-independent, embedded oriented evaluation protocol and a new metric called Normalized time to Failure (NT2F) to quantify how long a tracker can sustain a tracking sequence without external help. Experiments on UAV benchmarks, including an augmented UAV123 dataset with synthetic occlusions, show consistent improvements in Success and NT2F metrics across multiple tracking processing frequency. A ROS 2 implementation on a Nvidia Jetson AGX Orin confirms that the evaluation protocol more closely matches real-time performance on embedded systems.
>
---
#### [replaced 086] MedShift: Implicit Conditional Transport for X-Ray Domain Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.21435](https://arxiv.org/pdf/2508.21435)**

> **作者:** Francisco Caetano; Christiaan Viviers; Peter H.N. de With; Fons van der Sommen
>
> **备注:** Accepted at the ICCV 2025 AIM Workshop
>
> **摘要:** Synthetic medical data offers a scalable solution for training robust models, but significant domain gaps limit its generalizability to real-world clinical settings. This paper addresses the challenge of cross-domain translation between synthetic and real X-ray images of the head, focusing on bridging discrepancies in attenuation behavior, noise characteristics, and soft tissue representation. We propose MedShift, a unified class-conditional generative model based on Flow Matching and Schrodinger Bridges, which enables high-fidelity, unpaired image translation across multiple domains. Unlike prior approaches that require domain-specific training or rely on paired data, MedShift learns a shared domain-agnostic latent space and supports seamless translation between any pair of domains seen during training. We introduce X-DigiSkull, a new dataset comprising aligned synthetic and real skull X-rays under varying radiation doses, to benchmark domain translation models. Experimental results demonstrate that, despite its smaller model size compared to diffusion-based approaches, MedShift offers strong performance and remains flexible at inference time, as it can be tuned to prioritize either perceptual fidelity or structural consistency, making it a scalable and generalizable solution for domain adaptation in medical imaging. The code and dataset are available at this https URL
>
---
#### [replaced 087] Enhancing Cross-View UAV Geolocalization via LVLM-Driven Relational Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08063](https://arxiv.org/pdf/2603.08063)**

> **作者:** Bowen Liu; Pengyue Jia; Wanyu Wang; Derong Xu; Jiawei Cheng; Jiancheng Dong; Xiao Han; Zimo Zhao; Chao Zhang; Bowen Yu; Fangyu Hong; Xiangyu Zhao
>
> **摘要:** The primary objective of cross-view UAV geolocalization is to identify the exact spatial coordinates of drone-captured imagery by aligning it with extensive, geo-referenced satellite databases. Current approaches typically extract features independently from each perspective and rely on basic heuristics to compute similarity, thereby failing to explicitly capture the essential interactions between different views. To address this limitation, we introduce a novel, plug-and-play ranking architecture designed to explicitly perform joint relational modeling for improved UAV-to-satellite image matching. By harnessing the capabilities of a Large Vision-Language Model (LVLM), our framework effectively learns the deep visual-semantic correlations linking UAV and satellite imagery. Furthermore, we present a novel relational-aware loss function to optimize the training phase. By employing soft labels, this loss provides fine-grained supervision that avoids overly penalizing near-positive matches, ultimately boosting both the model's discriminative power and training stability. Comprehensive evaluations across various baseline architectures and standard benchmarks reveal that the proposed method substantially boosts the retrieval accuracy of existing models, yielding superior performance even under highly demanding conditions.
>
---
#### [replaced 088] See the Text: From Tokenization to Visual Reading
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SeeTok方法，将文本转为图像输入，解决传统子词分词在低资源语言中的不足，提升模型效率与鲁棒性。任务为自然语言处理中的文本表示与理解。**

- **链接: [https://arxiv.org/pdf/2510.18840](https://arxiv.org/pdf/2510.18840)**

> **作者:** Ling Xing; Rui Yan; Alex Jinpeng Wang; Zechao Li; Jinhui Tang
>
> **摘要:** People see text. Humans read by recognizing words as visual objects, including their shapes, layouts, and patterns, before connecting them to meaning, which enables us to handle typos, distorted fonts, and various scripts effectively. Modern large language models (LLMs), however, rely on subword tokenization, fragmenting text into pieces from a fixed vocabulary. While effective for high-resource languages, this approach over-segments low-resource languages, yielding long, linguistically meaningless sequences and inflating computation. In this work, we challenge this entrenched paradigm and move toward a vision-centric alternative. Our method, SeeTok, renders text as images (visual-text) and leverages pretrained multimodal LLMs to interpret them, reusing strong OCR and text-vision alignment abilities learned from large-scale multimodal training. Across three different language tasks, SeeTok matches or surpasses subword tokenizers while requiring 4.43 times fewer tokens and reducing FLOPs by 70.5%, with additional gains in cross-lingual generalization, robustness to typographic noise, and linguistic hierarchy. SeeTok signals a shift from symbolic tokenization to human-like visual reading, and takes a step toward more natural and cognitively inspired language models.
>
---
#### [replaced 089] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22989](https://arxiv.org/pdf/2511.22989)**

> **作者:** Yuta Oshima; Daiki Miyake; Kohsei Matsutani; Yusuke Iwasawa; Masahiro Suzuki; Yutaka Matsuo; Hiroki Furuta
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Recent text-to-image generation models have acquired the ability of multi-reference generation and editing; that is, to inherit the appearance of subjects from multiple reference images and re-render them in new contexts. However, existing benchmark datasets often focus on generation using a single or a few reference images, which prevents us from measuring progress in model performance or identifying weaknesses when following instructions with a larger number of references. In addition, their task definitions are still vague, limited to axes such as ``what to edit'' or ``how many references are given'', and therefore fail to capture the challenges inherent in combining heterogeneous references. To address this gap, we introduce MultiBanana, which is designed to assess the edge of model capabilities by widely covering problems specific to multi-reference settings: (1) varying the number of references (up to 8), (2) domain mismatch among references (e.g., photo vs. anime), (3) scale mismatch between reference and target scenes, (4) references containing rare concepts (e.g., a red banana), and (5) multilingual textual references for rendering. Our analysis among a variety of text-to-image models reveals their respective performances, typical failure modes, and areas for improvement. MultiBanana is released as an open benchmark to push the boundaries and establish a standardized basis for fair comparison in multi-reference image generation. Our data and code are available at this https URL .
>
---
#### [replaced 090] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19918](https://arxiv.org/pdf/2512.19918)**

> **作者:** Houston H. Zhang; Tao Zhang; Baoze Lin; Yuanqi Xue; Yincheng Zhu; Huan Liu; Li Gu; Linfeng Ye; Ziqiang Wang; Xinxin Zuo; Yang Wang; Yuanhao Yu; Zhixiang Chi
>
> **备注:** CVPR 2026, Code: this https URL
>
> **摘要:** User interface to code (UI2Code) aims to generate executable code that can faithfully reconstruct a given input UI. Prior work focuses largely on web pages and mobile screens, leaving app widgets underexplored. Unlike web or mobile UIs with rich hierarchical context, widgets are compact, context-free micro-interfaces that summarize key information through dense layouts and iconography under strict spatial constraints. Moreover, while (image, code) pairs are widely available for web or mobile UIs, widget designs are proprietary and lack accessible markup. We formalize this setting as the Widget-to-Code (Widget2Code) and introduce an image-only widget benchmark with fine-grained, multi-dimensional evaluation metrics. Benchmarking shows that although generalized multimodal large language models (MLLMs) outperform specialized UI2Code methods, they still produce unreliable and visually inconsistent code. To address these limitations, we develop a baseline that jointly advances perceptual understanding and structured code generation. At the perceptual level, we follow widget design principles to assemble atomic components into complete layouts, equipped with icon retrieval and reusable visualization modules. At the system level, we design an end-to-end infrastructure, WidgetFactory, which includes a framework-agnostic widget-tailored domain-specific language (WidgetDSL) and a compiler that translates it into multiple front-end implementations (e.g., React, HTML/CSS). An adaptive rendering module further refines spatial dimensions to satisfy compactness constraints. Together, these contributions substantially enhance visual fidelity, establishing a strong baseline and unified infrastructure for future Widget2Code research.
>
---
#### [replaced 091] TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.08881](https://arxiv.org/pdf/2601.08881)**

> **作者:** Yu Xu; Hongbin Yan; Juan Cao; Yiji Cheng; Tiankai Hang; Runze He; Zijin Yin; Shiyi Zhang; Yuxin Zhang; Jintao Li; Chunyu Wang; Qinglin Lu; Tong-Yee Lee; Fan Tang
>
> **备注:** Accept by CVPR 2026. Project page: this https URL
>
> **摘要:** Unified image generation and editing models suffer from severe task interference in dense diffusion transformers architectures, where a shared parameter space must compromise between conflicting objectives (e.g., local editing v.s. subject-driven generation). While the sparse Mixture-of-Experts (MoE) paradigm is a promising solution, its gating networks remain task-agnostic, operating based on local features, unaware of global task intent. This task-agnostic nature prevents meaningful specialization and fails to resolve the underlying task interference. In this paper, we propose a novel framework to inject semantic intent into MoE routing. We introduce a Hierarchical Task Semantic Annotation scheme to create structured task descriptors (e.g., scope, type, preservation). We then design Predictive Alignment Regularization to align internal routing decisions with the task's high-level semantics. This regularization evolves the gating network from a task-agnostic executor to a dispatch center. Our model effectively mitigates task interference, outperforming dense baselines in fidelity and quality, and our analysis shows that experts naturally develop clear and semantically correlated specializations.
>
---
#### [replaced 092] HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12760](https://arxiv.org/pdf/2603.12760)**

> **作者:** Xiaoyu Li; Yuhang Liu; Zheng Luo; Xuanshuo Kang; Fangqi Lou; Xiaohua Wu; Zihan Xiong
>
> **备注:** Accepted to CVPR 2026. Code available at this https URL
>
> **摘要:** In-Context Learning (ICL) is a significant paradigm for Large Multimodal Models (LMMs), using a few in-context demonstrations (ICDs) for new task adaptation. However, its performance is sensitive to demonstration configurations and computationally expensive. Mathematically, the influence of these demonstrations can be decomposed into a dynamic mixture of the standard attention output and the context values. Current approximation methods simplify this process by learning a "shift vector". Inspired by the exact decomposition, we introduce High-Fidelity In-Context Learning (HIFICL) to more faithfully model the ICL mechanism. HIFICL consists of three key components: 1) a set of "virtual key-value pairs" to act as a learnable context, 2) a low-rank factorization for stable and regularized training, and 3) a simple end-to-end training objective. From another perspective, this mechanism constitutes a form of context-aware Parameter-Efficient Fine-Tuning (PEFT). Extensive experiments show that HiFICL consistently outperforms existing approximation methods on several multimodal benchmarks. The code is available at this https URL.
>
---
#### [replaced 093] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00393](https://arxiv.org/pdf/2601.00393)**

> **作者:** Yuxue Yang; Lue Fan; Ziqi Shi; Junran Peng; Feng Wang; Zhaoxiang Zhang
>
> **备注:** CVPR 2026; Project Page: this https URL
>
> **摘要:** In this paper, we propose NeoVerse, a versatile 4D world model that is capable of 4D reconstruction, novel-trajectory video generation, and rich downstream applications. We first identify a common limitation of scalability in current 4D world modeling methods, caused either by expensive and specialized multi-view 4D data or by cumbersome training pre-processing. In contrast, our NeoVerse is built upon a core philosophy that makes the full pipeline scalable to diverse in-the-wild monocular videos. Specifically, NeoVerse features pose-free feed-forward 4D reconstruction, online monocular degradation pattern simulation, and other well-aligned techniques. These designs empower NeoVerse with versatility and generalization to various domains. Meanwhile, NeoVerse achieves state-of-the-art performance in standard reconstruction and generation benchmarks. Our project page is available at this https URL.
>
---
#### [replaced 094] CODER: Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2208.09843](https://arxiv.org/pdf/2208.09843)**

> **作者:** Haoran Wang; Dongliang He; Wenhao Wu; Boyang Xia; Min Yang; Fu Li; Yunlong Yu; Zhong Ji; Errui Ding; Jingdong Wang
>
> **备注:** Accepted by ECCV 2022
>
> **摘要:** Image-Text Retrieval (ITR) is challenging in bridging visual and lingual modalities. Contrastive learning has been adopted by most prior arts. Except for limited amount of negative image-text pairs, the capability of constrastive learning is restricted by manually weighting negative pairs as well as unawareness of external knowledge. In this paper, we propose our novel Coupled Diversity-Sensitive Momentum Constrastive Learning (CODER) for improving cross-modal representation. Firstly, a novel diversity-sensitive contrastive learning (DCL) architecture is invented. We introduce dynamic dictionaries for both modalities to enlarge the scale of image-text pairs, and diversity-sensitiveness is achieved by adaptive negative pair weighting. Furthermore, two branches are designed in CODER. One learns instance-level embeddings from image/text, and it also generates pseudo online clustering labels for its input image/text based on their embeddings. Meanwhile, the other branch learns to query from commonsense knowledge graph to form concept-level descriptors for both modalities. Afterwards, both branches leverage DCL to align the cross-modal embedding spaces while an extra pseudo clustering label prediction loss is utilized to promote concept-level representation learning for the second branch. Extensive experiments conducted on two popular benchmarks, i.e. MSCOCO and Flicker30K, validate CODER remarkably outperforms the state-of-the-art approaches. Our code is available at: this https URL.
>
---
#### [replaced 095] Complex-Valued Holographic Radiance Fields
- **分类: cs.GR; cs.CV; cs.ET**

- **链接: [https://arxiv.org/pdf/2506.08350](https://arxiv.org/pdf/2506.08350)**

> **作者:** Yicheng Zhan; Dong-Ha Shin; Seung-Hwan Baek; Kaan Akşit
>
> **备注:** 36 pages, 25 figures
>
> **摘要:** Modeling wave properties of light is an important milestone for advancing physically-based rendering. In this paper, we propose complex-valued holographic radiance fields, a method that optimizes scenes without relying on intensity-based intermediaries. By leveraging multi-view images, our method directly optimizes a scene representation using complex-valued Gaussian primitives representing amplitude and phase values aligned with the scene geometry. Our approach eliminates the need for computationally expensive holographic rendering that typically utilizes a single view of a given scene. This accelerates holographic rendering speed by 30x-10,000x while achieving on-par image quality with state-of-the-art holography methods, representing a promising step towards bridging the representation gap between modeling wave properties of light and 3D geometry of scenes.
>
---
#### [replaced 096] SSI-DM: Singularity Skipping Inversion of Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02193](https://arxiv.org/pdf/2602.02193)**

> **作者:** Chen Min; Enze Jiang; Jishen Peng; Zheng Ma
>
> **备注:** A complete revision is needed
>
> **摘要:** Inverting real images into the noise space is essential for editing tasks using diffusion models, yet existing methods produce non-Gaussian noise with poor editability due to the inaccuracy in early noising steps. We identify the root cause: a mathematical singularity that renders inversion fundamentally ill-posed. We propose Singularity Skipping Inversion of Diffusion Models (SSI-DM), which bypasses this singular region by adding small noise before standard inversion. This simple approach produces inverted noise with natural Gaussian properties while maintaining reconstruction fidelity. As a plug-and-play technique compatible with general diffusion models, our method achieves superior performance on public image datasets for reconstruction and interpolation tasks, providing a principled and efficient solution to diffusion model inversion.
>
---
#### [replaced 097] Context Matters: Peer-Aware Student Behavioral Engagement Measurement via VLM Action Parsing and LLM Sequence Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.06394](https://arxiv.org/pdf/2601.06394)**

> **作者:** Ahmed Abdelkawy; Ahmed Elsayed; Asem Ali; Aly Farag; Thomas Tretter; Michael McIntyre
>
> **摘要:** Understanding student behavior in the classroom is essential to improve both pedagogical quality and student engagement. Existing methods for predicting student engagement typically require substantial annotated data to model the diversity of student behaviors, yet privacy concerns often restrict researchers to their own proprietary datasets. Moreover, the classroom context, represented in peers' actions, is ignored. To address the aforementioned limitation, we propose a novel three-stage framework for video-based student engagement measurement. First, we explore the few-shot adaptation of the vision-language model for student action recognition, which is fine-tuned to distinguish among action categories with a few training samples. Second, to handle continuous and unpredictable student actions, we utilize the sliding temporal window technique to divide each student's 2-minute-long video into non-overlapping segments. Each segment is assigned an action category via the fine-tuned VLM model, generating a sequence of action predictions. Finally, we leverage the large language model to classify this entire sequence of actions, together with the classroom context, as belonging to an engaged or disengaged student. The experimental results demonstrate the effectiveness of the proposed approach in identifying student engagement. The source code and dataset will be available upon request
>
---
#### [replaced 098] Unified Primitive Proxies for Structured Shape Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00759](https://arxiv.org/pdf/2601.00759)**

> **作者:** Zhaiyu Chen; Yuqing Wang; Xiao Xiang Zhu
>
> **备注:** CVPR 2026
>
> **摘要:** Structured shape completion recovers missing geometry as primitives rather than as unstructured points, which enables primitive-based surface reconstruction. Instead of following the prevailing cascade, we rethink how primitives and points should interact, and find it more effective to decode primitives in a dedicated pathway that attends to shared shape features. Following this principle, we present UniCo, which in a single feed-forward pass predicts a set of primitives with complete geometry, semantics, and inlier membership. To drive this unified representation, we introduce primitive proxies, learnable queries that are contextualized to produce assembly-ready outputs. To ensure consistent optimization, our training strategy couples primitives and points with online target updates. Across synthetic and real-world benchmarks with four independent assembly solvers, UniCo consistently outperforms recent baselines, lowering Chamfer distance by up to 50% and improving normal consistency by up to 7%. These results establish an attractive recipe for structured 3D understanding from incomplete data. Project page: this https URL.
>
---
#### [replaced 099] Verifier Threshold: An Efficient Test-Time Scaling Approach for Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08985](https://arxiv.org/pdf/2512.08985)**

> **作者:** Vignesh Sundaresha; Akash Haridas; Vikram Appia; Lav R. Varshney
>
> **备注:** ICLR 2026 ReALM-Gen and DeLTa
>
> **摘要:** Image generation has emerged as a mainstream application of large generative models. Just as test-time compute and reasoning have improved language model capabilities, similar benefits have been observed for image generation models. In particular, searching over noise samples for diffusion and flow models has been shown to scale well with test-time compute. While recent works explore allocating non-uniform inference-compute budgets across denoising steps, existing approaches rely on greedy heuristics and often allocate the compute budget ineffectively. In this work, we study this problem and propose a simple fix. We propose Verifier-Threshold, which automatically reallocates test-time compute and delivers substantial efficiency improvements. For the same performance on the GenEval benchmark, we achieve a 2-4x reduction in computational time over the state-of-the-art method.
>
---
#### [replaced 100] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24295](https://arxiv.org/pdf/2603.24295)**

> **作者:** Kai Zhu; Zhenyu Cui; Zehua Zang; Jiahuan Zhou
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Recently, state space models have demonstrated efficient video segmentation through linear-complexity state space compression. However, Video Semantic Segmentation (VSS) requires pixel-level spatiotemporal modeling capabilities to maintain temporal consistency in segmentation of semantic objects. While state space models can preserve common semantic information during state space compression, the fixed-size state space inevitably forgets specific information, which limits the models' capability for pixel-level segmentation. To tackle the above issue, we proposed a Refining Specifics State Space Model approach (RS-SSM) for video semantic segmentation, which performs complementary refining of forgotten spatiotemporal specifics. Specifically, a Channel-wise Amplitude Perceptron (CwAP) is designed to extract and align the distribution characteristics of specific information in the state space. Besides, a Forgetting Gate Information Refiner (FGIR) is proposed to adaptively invert and refine the forgetting gate matrix in the state space model based on the specific information distribution. Consequently, our RS-SSM leverages the inverted forgetting gate to complementarily refine the specific information forgotten during state space compression, thereby enhancing the model's capability for spatiotemporal pixel-level segmentation. Extensive experiments on four VSS benchmarks demonstrate that our RS-SSM achieves state-of-the-art performance while maintaining high computational efficiency. The code is available at this https URL.
>
---
#### [replaced 101] Multimodal classification of Radiation-Induced Contrast Enhancements and tumor recurrence using deep learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11827](https://arxiv.org/pdf/2603.11827)**

> **作者:** Robin Peretzke; Marlin Hanstein; Maximilian Fischer; Lars Badhi Wessel; Obada Alhalabi; Sebastian Regnery; Andreas Kudak; Maximilian Deng; Tanja Eichkorn; Philipp Hoegen Saßmannshausen; Fabian Allmendinger; Jan-Hendrik Bolten; Philipp Schröter; Christine Jungk; Jürgen Peter Debus; Peter Neher; Laila König; Klaus Maier-Hein
>
> **摘要:** The differentiation between tumor recurrence and radiation-induced contrast enhancements in post-treatment glioblastoma patients remains a major clinical challenge. Existing approaches rely on clinically sparsely available diffusion MRI or do not consider radiation maps, which are gaining increasing interest in the tumor board for this differentiation. We introduce RICE-NET, a multimodal 3D deep learning model that integrates longitudinal MRI data with radiotherapy dose distributions for automated lesion classification using conventional T1-weighted MRI data. Using a cohort of 92 patients, the model achieved an F1 score of 0.92 on an independent test set. During extensive ablation experiments, we quantified the contribution of each timepoint and modality and showed that reliable classification largely depends on the radiation map. Occlusion-based interpretability analyses further confirmed the model's focus on clinically relevant regions. These findings highlight the potential of multimodal deep learning to enhance diagnostic accuracy and support clinical decision-making in neuro-oncology.
>
---
#### [replaced 102] StreamingClaw Technical Report
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22120](https://arxiv.org/pdf/2603.22120)**

> **作者:** Jiawei Chen; Zhe Chen; Chaoqun Du; Maokui He; Wei He; Hengtao Li; Qizhen Li; Zide Liu; Hao Ma; Xuhao Pan; Chang Ren; Xudong Rao; Xintian Shen; Chenfeng Wang; Tao Wei; Chengjun Yu; Pengfei Yu; Shengyu Yao; Chunpeng Zhou; Kun Zhan; Lihao Zheng; Pan Zhou; Xuhan Zhu; Yufei Zheng
>
> **备注:** Under Progress
>
> **摘要:** Emerging applications such as embodied intelligence, AI hardware, autonomous driving, and intelligent cockpits rely on a real-time perception-decision-action closed loop, posing stringent challenges for streaming video understanding. However, current agents mostly suffer from fragmented capabilities, such as supporting only offline video understanding, lacking long-term multimodal memory mechanisms, or struggling to achieve real-time reasoning and proactive interaction under streaming input. These shortcomings have become a key bottleneck for preventing agents from sustaining perception, making real-time decisions, and executing closed-loop actions in complex real-world environments, constraining their deployment and potential in dynamic, open physical worlds. To alleviate these issues, we propose StreamingClaw, a unified agent framework for streaming video understanding and embodied intelligence. Beyond maintaining full compatibility with the OpenClaw framework, it natively supports real-time, multimodal streaming interactions. StreamingClaw integrates five core capabilities: (1) It supports real-time streaming reasoning. (2) It supports reasoning about future events and proactive interaction under the online evolution of interaction objectives. (3) It supports multimodal long-term memory storage, hierarchical memory evolution, efficient memory retrieval, and memory sharing across multiple agents. (4) It supports a closed loop of perception-decision-action. In addition to conventional tools and skills, it also provides streaming tools and action-centric skills tailored for real-world physical environments. (5) It is compatible with the OpenClaw framework, allowing it to leverage the resources and support of the open-source community.
>
---
#### [replaced 103] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22013](https://arxiv.org/pdf/2602.22013)**

> **作者:** I-Hsiang Chen; Yu-Wei Liu; Tse-Yu Wu; Yu-Chien Chiang; Jen-Chien Yang; Wei-Ting Chen
>
> **备注:** Accepted by CVPR2026; Project Page: this https URL
>
> **摘要:** Vision-based Retrieval-Augmented Generation (VisRAG) leverages vision-language models (VLMs) to jointly retrieve relevant visual documents and generate grounded answers based on multimodal evidence. However, existing VisRAG models degrade in performance when visual inputs suffer from distortions such as blur, noise, low light, or shadow, where semantic and degradation factors become entangled within pretrained visual encoders, leading to errors in both retrieval and generation stages. To address this limitation, we introduce RobustVisRAG, a causality-guided dual-path framework that improves VisRAG robustness while preserving efficiency and zero-shot generalization. RobustVisRAG uses a non-causal path to capture degradation signals through unidirectional attention and a causal path to learn purified semantics guided by these signals. Together with the proposed Non-Causal Distortion Modeling and Causal Semantic Alignment objectives, the framework enforces a clear separation between semantics and degradations, enabling stable retrieval and generation under challenging visual conditions. To evaluate robustness under realistic conditions, we introduce the Distortion-VisRAG dataset, a large-scale benchmark containing both synthetic and real-world degraded documents across seven domains, with 12 synthetic and 5 real distortion types that comprehensively reflect practical visual degradations. Experimental results show that RobustVisRAG improves retrieval, generation, and end-to-end performance by 7.35%, 6.35%, and 12.40%, respectively, on real-world degradations, while maintaining comparable accuracy on clean inputs.
>
---
#### [replaced 104] One Dimensional CNN ECG Mamba for Multilabel Abnormality Classification in 12 Lead ECG
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13046](https://arxiv.org/pdf/2510.13046)**

> **作者:** Huawei Jiang; Husna Mutahira; Gan Huang; Mannan Saeed Muhammad
>
> **备注:** 6 Pages, 2 figures
>
> **摘要:** Accurate detection of cardiac abnormalities from electrocardiogram recordings is regarded as essential for clinical diagnostics and decision support. Traditional deep learning models such as residual networks and transformer architectures have been applied successfully to this task, but their performance has been limited when long sequential signals are processed. Recently, state space models have been introduced as an efficient alternative. In this study, a hybrid framework named One Dimensional Convolutional Neural Network Electrocardiogram Mamba is introduced, in which convolutional feature extraction is combined with Mamba, a selective state space model designed for effective sequence modeling. The model is built upon Vision Mamba, a bidirectional variant through which the representation of temporal dependencies in electrocardiogram data is enhanced. Comprehensive experiments on the PhysioNet Computing in Cardiology Challenges of 2020 and 2021 were conducted, and superior performance compared with existing methods was achieved. Specifically, the proposed model achieved substantially higher AUPRC and AUROC scores than those reported by the best previously published algorithms on twelve lead electrocardiograms. These results demonstrate the potential of Mamba-based architectures to advance reliable ECG classification. This capability supports early diagnosis and personalized treatment, while enhancing accessibility in telemedicine and resource-constrained healthcare systems.
>
---
