# 计算机视觉 cs.CV

- **最新发布 320 篇**

- **更新 197 篇**

## 最新发布

#### [new 001] Retrieval-Augmented Anatomical Guidance for Text-to-CT Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到CT生成任务，旨在解决生成图像 anatomically inconsistent 的问题。通过结合语义与解剖信息，提升生成质量与空间可控性。**

- **链接: [https://arxiv.org/pdf/2603.08305](https://arxiv.org/pdf/2603.08305)**

> **作者:** Daniele Molino; Camillo Maria Caruso; Paolo Soda; Valerio Guarrasi
>
> **摘要:** Text-conditioned generative models for volumetric medical imaging provide semantic control but lack explicit anatomical guidance, often resulting in outputs that are spatially ambiguous or anatomically inconsistent. In contrast, structure-driven methods ensure strong anatomical consistency but typically assume access to ground-truth annotations, which are unavailable when the target image is to be synthesized. We propose a retrieval-augmented approach for Text-to-CT generation that integrates semantic and anatomical information under a realistic inference setting. Given a radiology report, our method retrieves a semantically related clinical case using a 3D vision-language encoder and leverages its associated anatomical annotation as a structural proxy. This proxy is injected into a text-conditioned latent diffusion model via a ControlNet branch, providing coarse anatomical guidance while maintaining semantic flexibility. Experiments on the CT-RATE dataset show that retrieval-augmented generation improves image fidelity and clinical consistency compared to text-only baselines, while additionally enabling explicit spatial controllability, a capability inherently absent in such approaches. Further analysis highlights the importance of retrieval quality, with semantically aligned proxies yielding consistent gains across all evaluation axes. This work introduces a principled and scalable mechanism to bridge semantic conditioning and anatomical plausibility in volumetric medical image synthesis. Code will be released.
>
---
#### [new 002] RayD3D: Distilling Depth Knowledge Along the Ray for Robust Multi-View 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多视角3D目标检测任务，旨在提升模型在真实场景下的鲁棒性。针对深度预测不准确问题，提出RayD3D方法，通过沿光线传递深度知识，有效减少无关信息干扰。**

- **链接: [https://arxiv.org/pdf/2603.07493](https://arxiv.org/pdf/2603.07493)**

> **作者:** Rui Ding; Zhaonian Kuang; Zongwei Zhou; Meng Yang; Xinhu Zheng; Gang Hua
>
> **摘要:** Multi-view 3D detection with bird's eye view (BEV) is crucial for autonomous driving and robotics, but its robustness in real-world is limited as it struggles to predict accurate depth values. A mainstream solution, cross-modal distillation, transfers depth information from LiDAR to camera models but also unintentionally transfers depth-irrelevant information (e.g. LiDAR density). To mitigate this issue, we propose RayD3D, which transfers crucial depth knowledge along the ray: a line projecting from the camera to true location of an object. It is based on the fundamental imaging principle that predicted location of this object can only vary along this ray, which is finally determined by predicted depth value. Therefore, distilling along the ray enables more effective depth information transfer. More specifically, we design two ray-based distillation modules. Ray-based Contrastive Distillation (RCD) incorporates contrastive learning into distillation by sampling along the ray to learn how LiDAR accurately locates objects. Ray-based Weighted Distillation (RWD) adaptively adjusts distillation weight based on the ray to minimize the interference of depth-irrelevant information in LiDAR. For validation, we widely apply RayD3D into three representative types of BEV-based models, including BEVDet, BEVDepth4D, and BEVFormer. Our method is trained on clean NuScenes, and tested on both clean NuScenes and RoboBEV with a variety types of data corruptions. Our method significantly improves the robustness of all the three base models in all scenarios without increasing inference costs, and achieves the best when compared to recently released multi-view and distillation models.
>
---
#### [new 003] SAMoE-VLA: A Scene Adaptive Mixture-of-Experts Vision-Language-Action Model for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出SAMoE-VLA模型，解决自动驾驶中VLA模型性能不稳定问题，通过场景自适应的专家选择机制提升安全性与效果。**

- **链接: [https://arxiv.org/pdf/2603.08113](https://arxiv.org/pdf/2603.08113)**

> **作者:** Zihan You; Hongwei Liu; Chenxu Dang; Zhe Wang; Sining Ang; Aoqi Wang; Yan Wang
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models have shown promising capabilities in autonomous driving by leveraging the understanding and reasoning strengths of Large Language Models(LLMs).However, our empirical analysis reveals that directly applying existing token-level MoE mechanisms--which are inherited from LLM architectures--to VLA models results in unstable performance and safety degradation in autonomous driving, highlighting a misalignment between token-based expert specialization and scene-level this http URL address this, we propose SAMoE-VLA, a scene-adaptive Vision-Language-Action framework that conditions expert selection on structured scene representations instead of token embeddings. Our key idea is to derive the MoE routing signal from bird's-eye-view (BEV) features that encapsulates traffic scene context, enabling scenario-dependent expert weighting and merging tailored to distinct driving conditions. Furthermore, to support temporally consistent reasoning across world-knowledge, perception, language, and action, we introduce a Conditional Cross-Modal Causal Attention mechanism that integrates world state, linguistic intent, and action history into a unified causal reasoning process. Extensive experiments on the nuScenes open loop planning dataset and LangAuto closed-loop benchmark demonstrate that SAMoE-VLA achieves state-of-the-art performance, outperforming prior VLA-based and world-model-based approaches with fewer this http URL code will be released soon.
>
---
#### [new 004] Reading $\neq$ Seeing: Diagnosing and Closing the Typography Gap in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，解决模型对字体样式识别不足的问题。通过系统评估不同字体属性，发现模型在字体风格上表现差，提出微调方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.08497](https://arxiv.org/pdf/2603.08497)**

> **作者:** Heng Zhou; Ao Yu; Li Kang; Yuchen Fan; Yutao Fan; Xiufeng Song; Hejia Geng; Yiran Qin
>
> **摘要:** Vision-Language Models achieve near-perfect accuracy at reading text in images, yet prove largely typography-blind: capable of recognizing what text says, but not how it looks. We systematically investigate this gap by evaluating font family, size, style, and color recognition across 26 fonts, four scripts, and three difficulty levels. Our evaluation of 15 state-of-the-art VLMs reveals a striking perception hierarchy: color recognition is near-perfect, yet font style detection remains universally poor. We further find that model scale fails to predict performance and that accuracy is uniform across difficulty levels, together pointing to a training-data omission rather than a capacity ceiling. LoRA fine-tuning on a small set of synthetic samples substantially improves an open-source model, narrowing the gap to the best closed-source system and surpassing it on font size recognition. Font style alone remains resistant to fine-tuning, suggesting that relational visual reasoning may require architectural innovation beyond current patch-based encoders. We release our evaluation framework, data, and fine-tuning recipe to support progress in closing the typographic gap in vision-language understanding.
>
---
#### [new 005] Adaptive MLP Pruning for Large Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于模型压缩任务，旨在减少大视觉Transformer的参数量。通过自适应剪枝MLP模块，在不显著影响性能的前提下，实现约40%的参数和FLOPs减少。**

- **链接: [https://arxiv.org/pdf/2603.08100](https://arxiv.org/pdf/2603.08100)**

> **作者:** Chengchao Shen
>
> **摘要:** Large vision transformers present impressive scalability, as their performance can be well improved with increased model capacity. Nevertheless, their cumbersome parameters results in exorbitant computational and memory demands. By analyzing prevalent transformer structures, we find that multilayer perceptron (MLP) modules constitute the largest share of the model's parameters. In this paper, we propose an Adaptive MLP Pruning (AMP) method to substantially reduce the parameters of large vision transformers without obvious performance degradation. First, we adopt Taylor based method to evaluate neuron importance of MLP. However, the importance computation using one-hot cross entropy loss ignores the potential predictions on other categories, thus degrading the quality of the evaluated importance scores. To address this issue, we introduce label-free information entropy criterion to fully model the predictions of the original model for more accurate importance evaluation. Second, we rank the hidden neurons of MLP by the above importance scores and apply binary search algorithm to adaptively prune the ranked neurons according to the redundancy of different MLP modules, thereby avoiding the predefined compression ratio. Experimental results on several state-of-the-art large vision transformers, including CLIP and DINOv2, demonstrate that our method achieves roughly 40\% parameter and FLOPs reduction in a near lossless manner. Moreover, when the models are not finetuned after pruning, our method outperforms other pruning methods by significantly large margin. The source code and trained weights are available at this https URL.
>
---
#### [new 006] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型中的空间推理任务，旨在解决模型在空间理解上的不足。通过引入3ViewSense框架，利用正交视图提升空间推理能力。**

- **链接: [https://arxiv.org/pdf/2603.07751](https://arxiv.org/pdf/2603.07751)**

> **作者:** Shaoxiong Zhan; Yanlin Lai; Zheng Liu; Hai Lin; Shen Li; Xiaodong Cai; Zijian Lin; Wen Huang; Hai-Tao Zheng
>
> **摘要:** Current Large Language Models have achieved Olympiad-level logic, yet Vision-Language Models paradoxically falter on elementary spatial tasks like block counting. This capability mismatch reveals a critical ``spatial intelligence gap,'' where models fail to construct coherent 3D mental representations from 2D observations. We uncover this gap via diagnostic analyses showing the bottleneck is a missing view-consistent spatial interface rather than insufficient visual features or weak reasoning. To bridge this, we introduce \textbf{3ViewSense}, a framework that grounds spatial reasoning in Orthographic Views. Drawing on engineering cognition, we propose a ``Simulate-and-Reason'' mechanism that decomposes complex scenes into canonical orthographic projections to resolve geometric ambiguities. By aligning egocentric perceptions with these allocentric references, our method facilitates explicit mental rotation and reconstruction. Empirical results on spatial reasoning benchmarks demonstrate that our method significantly outperforms existing baselines, with consistent gains on occlusion-heavy counting and view-consistent spatial reasoning. The framework also improves the stability and consistency of spatial descriptions, offering a scalable path toward stronger spatial intelligence in multimodal systems.
>
---
#### [new 007] TrajPred: Trajectory-Conditioned Joint Embedding Prediction for Surgical Instrument-Tissue Interaction Recognition in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于手术器械-组织交互识别任务，旨在解决视觉语言模型在该任务上的性能不足问题。通过引入轨迹编码和预测模块，提升动作细节的对齐与识别效果。**

- **链接: [https://arxiv.org/pdf/2603.06999](https://arxiv.org/pdf/2603.06999)**

> **作者:** Jiajun Cheng; Xiaofan Yu; Subarna; Sainan Liu; Shan Lin
>
> **摘要:** Recognizing instruments' interactions with tissues is essential for building context-aware AI assistants in robotic surgery. Vision-language models (VLMs) have opened a new avenue for surgical perception and achieved better generalization on a wide range of tasks compared to conventional task-specific deep learning approaches. However, their performance on instrument--tissue interaction recognition remains limited, largely due to two challenges: (1) many models do not effectively leverage temporal information, and (2) alignment between vision and text often misses fine-grained action details. To address these issues, we propose TrajPred, a framework that encodes instrument trajectories to incorporate temporal motion cues and, conditioned on these trajectories, introduces a predictor module to generate visual semantic embeddings that better capture fine-grained action details. We further incorporate prompt tuning and a verb-rephrasing technique to enable smooth adaptation to the instrument--tissue interaction recognition task. Extensive experiments on the public laparoscopic benchmark, CholecT50, show that our method improves both Average Precision and Top-K accuracy. We also investigate whether visual embeddings of instrument--tissue interaction regions align better with the corresponding text by visualizing the cosine similarity between visual and textual embeddings. The visualization results indicate that the proposed method improves alignment between relevant visual and textual representations.
>
---
#### [new 008] Information Maximization for Long-Tailed Semi-Supervised Domain Generalization
- **分类: cs.CV**

- **简介: 该论文属于半监督域泛化任务，解决长尾类别分布下的性能下降问题。提出IMaX方法，通过最大化特征与标签的互信息提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.08434](https://arxiv.org/pdf/2603.08434)**

> **作者:** Leo Fillioux; Omprakash Chakraborty; Quentin Gopée; Pierre Marza; Paul-Henry Cournède; Stergios Christodoulidis; Maria Vakalopoulou; Ismail Ben Ayed; Jose Dolz
>
> **摘要:** Semi-supervised domain generalization (SSDG) has recently emerged as an appealing alternative to tackle domain generalization when labeled data is scarce but unlabeled samples across domains are abundant. In this work, we identify an important limitation that hampers the deployment of state-of-the-art methods on more challenging but practical scenarios. In particular, state-of-the-art SSDG severely suffers in the presence of long-tailed class distributions, an arguably common situation in real-world settings. To alleviate this limitation, we propose IMaX, a simple yet effective objective based on the well-known InfoMax principle adapted to the SSDG scenario, where the Mutual Information (MI) between the learned features and latent labels is maximized, constrained by the supervision from the labeled samples. Our formulation integrates an {\alpha}-entropic objective, which mitigates the class-balance bias encoded in the standard marginal entropy term of the MI, thereby better handling arbitrary class distributions. IMaX can be seamlessly plugged into recent state-of-the-art SSDG, consistently enhancing their performance, as demonstrated empirically across two different image modalities.
>
---
#### [new 009] Evaluating Synthetic Data for Baggage Trolley Detection in Airport Logistics
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文针对机场行李推车检测任务，解决真实数据不足与标注成本高的问题。通过生成合成数据并结合少量真实数据训练模型，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.07645](https://arxiv.org/pdf/2603.07645)**

> **作者:** Abdeldjalil Taibi; Mohmoud Badlis; Amina Bensalem; Belkacem Zouilekh; Mohammed Brahimi
>
> **摘要:** Efficient luggage trolley management is critical for reducing congestion and ensuring asset availability in modern airports. Automated detection systems face two main challenges. First, strict security and privacy regulations limit large-scale data collection. Second, existing public datasets lack the diversity, scale, and annotation quality needed to handle dense, overlapping trolley arrangements typical of real-world operations. To address these limitations, we introduce a synthetic data generation pipeline based on a high-fidelity Digital Twin of Algiers International Airport using NVIDIA Omniverse. The pipeline produces richly annotated data with oriented bounding boxes, capturing complex trolley formations, including tightly nested chains. We evaluate YOLO-OBB using five training strategies: real-only, synthetic-only, linear probing, full fine-tuning, and mixed training. This allows us to assess how synthetic data can complement limited real-world annotations. Our results show that mixed training with synthetic data and only 40 percent of real annotations matches or exceeds the full real-data baseline, achieving 0.94 mAP@50 and 0.77 mAP@50-95, while reducing annotation effort by 25 to 35 percent. Multi-seed experiments confirm strong reproducibility with a standard deviation below 0.01 on mAP@50, demonstrating the practical effectiveness of synthetic data for automated trolley detection.
>
---
#### [new 010] ObjChangeVR: Object State Change Reasoning from Continuous Egocentric Views in VR Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于VR环境中物体状态变化推理任务，解决无直接交互下物体状态变化检测难题。提出ObjChangeVR框架与ObjChangeVR-Dataset，提升多视角下状态变化识别效果。**

- **链接: [https://arxiv.org/pdf/2603.06648](https://arxiv.org/pdf/2603.06648)**

> **作者:** Shiyi Ding; Shaoen Wu; Ying Chen
>
> **备注:** European Chapter of the Association for Computational Linguistics (EACL) 2026 Main
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) offer a promising approach for natural language-based scene change queries in virtual reality (VR). Prior work on applying MLLMs for object state understanding has focused on egocentric videos that capture the camera wearer's interactions with objects. However, object state changes may occur in the background without direct user interaction, lacking explicit motion cues and making them difficult to detect. Moreover, no benchmark exists for evaluating this challenging scenario. To address these challenges, we introduce ObjChangeVR-Dataset, specifically for benchmarking the question-answering task of object state change. We also propose ObjChangeVR, a framework that combines viewpoint-aware and temporal-based retrieval to identify relevant frames, along with cross-view reasoning that reconciles inconsistent evidence from multiple viewpoints. Extensive experiments demonstrate that ObjChangeVR significantly outperforms baseline approaches across multiple MLLMs.
>
---
#### [new 011] Unmixing microinfrared spectroscopic images of cross-sections of historical oil paintings
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于光谱图像解混任务，旨在解决历史油画截面ATR-μFTIR光谱图像中混合成分的自动识别问题。通过引入基于CNN的无监督方法，提高解混效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.06673](https://arxiv.org/pdf/2603.06673)**

> **作者:** Shivam Pande; Nicolas Nadisic; Francisco Mederos-Henry; Aleksandra Pizurica
>
> **备注:** 5 pages
>
> **摘要:** Spectroscopic imaging (SI) has become central to heritage science because it enables non-invasive, spatially resolved characterisation of materials in artefacts. In particular, attenuated total reflection Fourier transform infrared microscopy (ATR-$\mu$FTIR) is widely used to analyse painting cross-sections, where a spectrum is recorded at each pixel to form a hyperspectral image (HSI). Interpreting these data is difficult: spectra are often mixtures of several species in heterogeneous, multi-layered and degraded samples, and current practice still relies heavily on manual comparison with reference libraries. This workflow is slow, subjective and hard to scale. We propose an unsupervised CNN autoencoder for blind unmixing of ATR-$\mu$FTIR HSIs, estimating endmember spectra and their abundance maps while exploiting local spatial structure through patch-based modelling. To reduce sensitivity to atmospheric and acquisition artefacts across $>1500$ bands, we introduce a weighted spectral angle distance (WSAD) loss with automatic band-reliability weights derived from robust measures of spatial flatness, neighbour agreement and spectral roughness. Compared with standard SAD training, WSAD improves interpretability in contamination-prone spectral regions. We demonstrate the method on an ATR-$\mu$FTIR cross-section from the Ghent Altarpiece attributed to the Van Eyck brothers.
>
---
#### [new 012] On the Generalization Capacities of MLLMs for Spatial Intelligence
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究多模态大语言模型在空间智能任务中的泛化能力，解决其在不同摄像头下的泛化问题。提出Camera-Aware框架，通过注入相机参数、数据增强和几何先验提升模型的跨摄像头泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.06704](https://arxiv.org/pdf/2603.06704)**

> **作者:** Gongjie Zhang; Wenhao Li; Quanhao Qian; Jiuniu Wang; Deli Zhao; Shijian Lu; Ran Xu
>
> **备注:** ICLR 2026 (Oral)
>
> **摘要:** Multimodal Large Language Models (MLLMs) that directly process RGB inputs for tasks like 3D localization and navigation have shown remarkable potential. However, we argue that these RGB-only approaches are fundamentally flawed in their ability to generalize across cameras. By ignoring camera parameters, they entangle an object's physical properties with the camera's perspective, creating an irresolvable ambiguity. We show this leads MLLMs to overfit to the training camera distribution, rather than learning true and generalizable 3D geometric principles. To address this, we propose Camera-Aware MLLM framework for spatial MLLMs. It learns generalizable spatial reasoning by: (i) injecting camera intrinsics via a dense embedding that conditions each visual token; (ii) introducing a camera-aware data augmentation strategy that synthetically varies camera parameters, forcing the model to disentangle camera properties from scene content; and (iii) distilling geometric priors from a 3D vision foundation model. Extensive experiments demonstrate that camera-aware MLLMs substantially outperform their naive counterparts, particularly in cross-camera generalization tests on spatially-grounded tasks, indicating that camera-awareness is not only beneficial but also a prerequisite for robust and generalizable spatial intelligence in MLLMs.
>
---
#### [new 013] DreamSAC: Learning Hamiltonian World Models via Symmetry Exploration
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决世界模型在物理属性外推上的不足。通过引入对称性探索和哈密顿世界模型，提升模型对物理规律的掌握。**

- **链接: [https://arxiv.org/pdf/2603.07545](https://arxiv.org/pdf/2603.07545)**

> **作者:** Jinzhou Tang; Fan Feng; Minghao Fu; Wenjun Lin; Biwei Huang; Keze Wang
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Learned world models excel at interpolative generalization but fail at extrapolative generalization to novel physical properties. This limitation arises because they learn statistical correlations rather than the environment's underlying generative rules, such as physical invariances and conservation laws. We argue that learning these invariances is key to robust extrapolation. To achieve this, we first introduce \textbf{Symmetry Exploration}, an unsupervised exploration strategy where an agent is intrinsically motivated by a Hamiltonian-based curiosity bonus to actively probe and challenge its understanding of conservation laws, thereby collecting physically informative data. Second, we design a Hamiltonian-based world model that learns from the collected data, using a novel self-supervised contrastive objective to identify the invariant physical state from raw, view-dependent pixel observations. Our framework, \textbf{DreamSAC}, trained on this actively curated data, significantly outperforms state-of-the-art baselines in 3D physics simulations on tasks requiring extrapolation.
>
---
#### [new 014] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于多模态理解任务，旨在解决视频问答中模型在不确定情况下的可信度评估问题。提出VirtueBench基准，评估模型在缺乏关键帧时的拒绝回答能力。**

- **链接: [https://arxiv.org/pdf/2603.07071](https://arxiv.org/pdf/2603.07071)**

> **作者:** Xueqing Yu; Bohan Li; Yan Li; Zhenheng Yang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent Vision-Language Models (VLMs) have made remarkable progress in multimodal understanding tasks, yet their evaluation on long video understanding remains unreliable. Due to limited frame inputs, key frames necessary for answering the question may be missing from the model's input. However, models that truthfully refuse to answer under such uncertainty are marked as incorrect, while those that guess may coincidentally produce the correct answer and thus obtain deceptively higher accuracy, leading to misleading evaluation results and encouraging models to guess rather than respond honestly. To address this issue, we introduce VirtueBench, a benchmark explicitly designed to assess model trustworthiness under uncertainty. VirtueBench constructs multiple frame-sampling levels for each video and provides ground truths that distinguish between answerable and unanswerable cases. Evaluations on 25 open-source and commercial VLMs reveal distinct refusal behaviors across different model families, with refusal accuracy ranging from over 70% in the best models to nearly 0% in the worst. Moreover, most models exhibit a substantial drop in refusal when the prompt does not explicitly require them to do so. These findings highlight the need for developing trustworthy VLMs for multimodal understanding, guided by benchmarks and leaderboards that emphasize reliability and trustworthiness.
>
---
#### [new 015] Better Eyes, Better Thoughts: Why Vision Chain-of-Thought Fails in Medicine
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究医学视觉问答任务，探讨CoT在医疗领域效果不佳的原因，并提出两种无需训练的视觉定位方法以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.06665](https://arxiv.org/pdf/2603.06665)**

> **作者:** Yuan Wu; Zongxian Yang; Jiayu Qian; Songpan Gao; Guanxing Chen; Qiankun Li; Yu-An Huang; Zhi-An Huang
>
> **摘要:** Large vision-language models (VLMs) often benefit from chain-of-thought (CoT) prompting in general domains, yet its efficacy in medical vision-language tasks remains underexplored. We report a counter-intuitive trend: on medical visual question answering, CoT frequently underperforms direct answering (DirA) across general-purpose and medical-specific models. We attribute this to a \emph{medical perception bottleneck}: subtle, domain-specific cues can weaken visual grounding, and CoT may compound early perceptual uncertainty rather than correct it. To probe this hypothesis, we introduce two training-free, inference-time grounding interventions: (i) \emph{perception anchoring} via region-of-interest cues and (ii) \emph{description grounding} via high-quality textual guidance. Across multiple benchmarks and model families, these interventions improve accuracy, mitigate CoT degradation, and in several settings reverse the CoT--DirA inversion. Our findings suggest that reliable clinical VLMs require robust visual grounding and cross-modal alignment, beyond extending text-driven reasoning chains. Code is available \href{this https URL}{here}.
>
---
#### [new 016] Global Cross-Modal Geo-Localization: A Million-Scale Dataset and a Physical Consistency Learning Framework
- **分类: cs.CV**

- **简介: 该论文属于跨模态地理定位任务，旨在解决现有研究地理覆盖有限、场景多样性不足的问题。作者构建了百万级数据集CORE，并提出PLANET模型，提升全球范围内的定位性能。**

- **链接: [https://arxiv.org/pdf/2603.08491](https://arxiv.org/pdf/2603.08491)**

> **作者:** Yutong Hu; Jinhui Chen; Chaoqiang Xu; Yuan Kou; Sili Zhou; Shaocheng Yan; Pengcheng Shi; Qingwu Hu; Jiayuan Li
>
> **摘要:** Cross-modal Geo-localization (CMGL) matches ground-level text descriptions with geo-tagged aerial imagery, which is crucial for pedestrian navigation and emergency response. However, existing researches are constrained by narrow geographic coverage and simplistic scene diversity, failing to reflect the immense spatial heterogeneity of global architectural styles and topographic features. To bridge this gap and facilitate universal positioning, we introduce CORE, the first million-scale dataset dedicated to global CMGL. CORE comprises 1,034,786 cross-view images sampled from 225 distinct geographic regions across all continents, offering an unprecedented variety of perspectives in varying environmental conditions and urban layouts. We leverage the zero-shot reasoning of Large Vision-Language Models (LVLMs) to synthesize high-quality scene descriptions rich in discriminative cues. Furthermore, we propose a physical-law-aware network (PLANET) for cross-modal geo-localization. PLANET introduces a novel contrastive learning paradigm to guide textual representations in capturing the intrinsic physical signatures of satellite imagery. Extensive experiments across varied geographic regions demonstrate that PLANet significantly outperforms state-of-the-art methods, establishing a new benchmark for robust, global-scale geo-localization. The dataset and source code will be released at this https URL.
>
---
#### [new 017] GameVerse: Can Vision-Language Models Learn from Video-based Reflection?
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GameVerse，一个用于评估视觉语言模型通过视频反思学习的基准。任务是研究VLM能否从视频中学习，解决如何有效评估其视觉经验内化问题。工作包括构建基准、设计评估范式和分析模型表现。**

- **链接: [https://arxiv.org/pdf/2603.06656](https://arxiv.org/pdf/2603.06656)**

> **作者:** Kuan Zhang; Dongchen Liu; Qiyue Zhao; Jinkun Hou; Xinran Zhang; Qinlei Xie; Miao Liu; Yiming Li
>
> **摘要:** Human gameplay is a visually grounded interaction loop in which players act, reflect on failures, and watch tutorials to refine strategies. Can Vision-Language Models (VLMs) also learn from video-based reflection? We present GameVerse, a comprehensive video game benchmark that enables a reflective visual interaction loop. Moving beyond traditional fire-and-forget evaluations, it uses a novel reflect-and-retry paradigm to assess how VLMs internalize visual experience and improve policies. To facilitate systematic and scalable evaluation, we also introduce a cognitive hierarchical taxonomy spanning 15 globally popular games, dual action space for both semantic and GUI control, and milestone evaluation using advanced VLMs to quantify progress. Our experiments show that VLMs benefit from video-based reflection in varied settings, and perform best by combining failure trajectories and expert tutorials-a training-free analogue to reinforcement learning (RL) plus supervised fine-tuning (SFT).
>
---
#### [new 018] DogWeave: High-Fidelity 3D Canine Reconstruction from a Single Image via Normal Fusion and Conditional Inpainting
- **分类: cs.CV**

- **简介: 该论文属于单图像3D动物重建任务，旨在解决几何失真和纹理不一致问题。提出DogWeave框架，通过法线融合和条件修复生成高保真犬类3D模型。**

- **链接: [https://arxiv.org/pdf/2603.07441](https://arxiv.org/pdf/2603.07441)**

> **作者:** Shufan Sun; Chenchen Wang; Zongfu Yu
>
> **摘要:** Monocular 3D animal reconstruction is challenging due to complex articulation, self-occlusion, and fine-scale details such as fur. Existing methods often produce distorted geometry and inconsistent textures due to the lack of articulated 3D supervision and limited availability of back-view images in 2D datasets, which makes reconstructing unobserved regions particularly difficult. To address these limitations, we propose DogWeave, a model-based framework for reconstructing high-fidelity 3D canine models from a single RGB image. DogWeave improves geometry by refining a coarsely-initiated parametric mesh into a detailed SDF representation through multi-view normal field optimization using diffusion-enhanced normals. It then generates view-consistent textures through conditional partial inpainting guided by structure and style cues, enabling realistic reconstruction of unobserved regions. Using only about 7,000 dog images processed via our 2D pipeline for training, DogWeave produces complete, realistic 3D models and outperforms state-of-the-art single image to 3d reconstruction methods in both shape accuracy and texture realism for canines.
>
---
#### [new 019] SurgCUT3R: Surgical Scene-Aware Continuous Understanding of Temporal 3D Representation
- **分类: cs.CV**

- **简介: 该论文属于手术场景的3D重建任务，旨在解决单目内镜视频重建中的数据不足和长期性能下降问题。通过生成伪真值数据、混合监督策略和分层推理框架提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.06971](https://arxiv.org/pdf/2603.06971)**

> **作者:** Kaiyuan Xu; Fangzhou Hong; Daniel Elson; Baoru Huang
>
> **摘要:** Reconstructing surgical scenes from monocular endoscopic video is critical for advancing robotic-assisted surgery. However, the application of state-of-the-art general-purpose reconstruction models is constrained by two key challenges: the lack of supervised training data and performance degradation over long video sequences. To overcome these limitations, we propose SurgCUT3R, a systematic framework that adapts unified 3D reconstruction models to the surgical domain. Our contributions are threefold. First, we develop a data generation pipeline that exploits public stereo surgical datasets to produce large-scale, metric-scale pseudo-ground-truth depth maps, effectively bridging the data gap. Second, we propose a hybrid supervision strategy that couples our pseudo-ground-truth with geometric self-correction to enhance robustness against inherent data imperfections. Third, we introduce a hierarchical inference framework that employs two specialized models to effectively mitigate accumulated pose drift over long surgical videos: one for global stability and one for local accuracy. Experiments on the SCARED and StereoMIS datasets demonstrate that our method achieves a competitive balance between accuracy and efficiency, delivering near state-of-the-art but substantially faster pose estimation and offering a practical and effective solution for robust reconstruction in surgical environments. Project page: this https URL.
>
---
#### [new 020] Med-Evo: Test-time Self-evolution for Medical Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于医疗多模态大模型任务，解决标注数据不足问题，提出Med-Evo框架，通过无标签强化学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07443](https://arxiv.org/pdf/2603.07443)**

> **作者:** Dunyuan Xu; Xikai Yang; Juzheng Miao; Yaoqian Li; Jinpeng Li; Pheng-Ann Heng
>
> **摘要:** Medical Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse healthcare tasks. However, current post-training strategies, such as supervised fine-tuning and reinforcement learning, heavily depend on substantial annotated data while overlooking the potential of unlabeled test data for model enhancement. This limitation becomes particularly pronounced in medical domains, where acquiring extensive labeled medical data is difficult due to the strict data sensitivity and annotation complexity. Moreover, leveraging test data poses challenges in generating reliable supervision signals from unlabeled samples and maintaining stable self-evolution. To address these limitations, we propose Med-Evo, the first self-evolution framework for medical MLLMs that utilizes label-free reinforcement learning to promote model performance without requiring additional labeled data. Our framework introduces two key innovations: $1)$ Feature-driven Pseudo Labeling (FPL) that identifies semantic centroids from all heterogeneous candidate responses to select pseudo labels in each rollout, and $2)$ Hard-Soft Reward (HSR) that combines exact match with token-level assessment and semantic similarity to provide hierarchical reward. Experiments on three medical VQA benchmarks and two base MLLMs show clear advantages of our approach over SOTA methods, with significant improvements of 10.43\% accuracy and 4.68\% recall on the SLAKE dataset using Qwen2.5-VL, showing the effectiveness of our method.
>
---
#### [new 021] Duala: Dual-Level Alignment of Subjects and Stimuli for Cross-Subject fMRI Decoding
- **分类: cs.CV**

- **简介: 该论文属于跨被试fMRI解码任务，旨在解决新被试数据有限时性能下降的问题。提出Duala框架，通过双层对齐提升刺激一致性与被试间脑响应对齐。**

- **链接: [https://arxiv.org/pdf/2603.07625](https://arxiv.org/pdf/2603.07625)**

> **作者:** Shumeng Li; Jintao Guo; Jian Zhang; Yulin Zhou; Luyang Cao; Yinghuan Shi
>
> **摘要:** Cross-subject visual decoding aims to reconstruct visual experiences from brain activity across individuals, enabling more scalable and practical brain-computer interfaces. However, existing methods often suffer from degraded performance when adapting to new subjects with limited data, as they struggle to preserve both the semantic consistency of stimuli and the alignment of brain responses. To address these challenges, we propose Duala, a dual-level alignment framework designed to achieve stimulus-level consistency and subject-level alignment in fMRI-based cross-subject visual decoding. (1) At the stimulus level, Duala introduces a semantic alignment and relational consistency strategy that preserves intra-class similarity and inter-class separability, maintaining clear semantic boundaries during adaptation. (2) At the subject level, a distribution-based feature perturbation mechanism is developed to capture both global and subject-specific variations, enabling adaptation to individual neural representations without overfitting. Experiments on the Natural Scenes Dataset (NSD) demonstrate that Duala effectively improves alignment across subjects. Remarkably, even when fine-tuned with only about one hour of fMRI data, Duala achieves over 81.1% image-to-brain retrieval accuracy and consistently outperforms existing fine-tuning strategies in both retrieval and reconstruction. Our code is available at this https URL.
>
---
#### [new 022] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing
- **分类: cs.CV**

- **简介: 该论文提出CARE-Edit，解决图像编辑中多条件冲突问题，通过动态路由专家模型提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2603.08589](https://arxiv.org/pdf/2603.08589)**

> **作者:** Yucheng Wang; Zedong Wang; Yuetong Wu; Yue Ma; Dan Xu
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Unified diffusion editors often rely on a fixed, shared backbone for diverse tasks, suffering from task interference and poor adaptation to heterogeneous demands (e.g., local vs global, semantic vs photometric). In particular, prevalent ControlNet and OmniControl variants combine multiple conditioning signals (e.g., text, mask, reference) via static concatenation or additive adapters which cannot dynamically prioritize or suppress conflicting modalities, thus resulting in artifacts like color bleeding across mask boundaries, identity or style drift, and unpredictable behavior under multi-condition inputs. To address this, we propose Condition-Aware Routing of Experts (CARE-Edit) that aligns model computation with specific editing competencies. At its core, a lightweight latent-attention router assigns encoded diffusion tokens to four specialized experts--Text, Mask, Reference, and Base--based on multi-modal conditions and diffusion timesteps: (i) a Mask Repaint module first refines coarse user-defined masks for precise spatial guidance; (ii) the router applies sparse top-K selection to dynamically allocate computation to the most relevant experts; (iii) a Latent Mixture module subsequently fuses expert outputs, coherently integrating semantic, spatial, and stylistic information to the base images. Experiments validate CARE-Edit's strong performance on contextual editing tasks, including erasure, replacement, text-driven edits, and style transfer. Empirical analysis further reveals task-specific behavior of specialized experts, showcasing the importance of dynamic, condition-aware processing to mitigate multi-condition conflicts.
>
---
#### [new 023] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决DETR模型中查询利用不平衡问题。提出PaQ-DETR，通过动态生成查询和质量感知分配策略提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.06917](https://arxiv.org/pdf/2603.06917)**

> **作者:** Zhengjian Kang; Jun Zhuang; Kangtong Mo; Qi Chen; Rui Liu; Ye Zhang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Detection Transformer (DETR) has redefined object detection by casting it as a set prediction task within an end-to-end framework. Despite its elegance, DETR and its variants still rely on fixed learnable queries and suffer from severe query utilization imbalance, which limits adaptability and leaves the model capacity underused. We propose PaQ-DETR (Pattern and Quality-Aware DETR), a unified framework that enhances both query adaptivity and supervision balance. It learns a compact set of shared latent patterns capturing global semantics and dynamically generates image-specific queries through content-conditioned weighting. In parallel, a quality-aware one-to-many assignment strategy adaptively selects positive samples based on localizatio-classification consistency, enriching supervision and promoting balanced query optimization. Experiments on COCO, CityScapes, and other benchmarks show consistent gains of 1.5%-4.2% mAP across DETR backbones, including ResNet and Swin-Transformer. Beyond accuracy improvement, our method provides interpretable insights into how dynamic patterns cluster semantically across object categories.
>
---
#### [new 024] OPTED: Open Preprocessed Trachoma Eye Dataset Using Zero-Shot SAM 3 Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决缺乏高质量预处理的沙眼数据集问题。通过SAM 3模型提取感兴趣区域，构建公开数据集OPTED。**

- **链接: [https://arxiv.org/pdf/2603.06885](https://arxiv.org/pdf/2603.06885)**

> **作者:** Kibrom Gebremedhin; Hadush Hailu; Bruk Gebregziabher
>
> **备注:** 9 figure, 3 tables
>
> **摘要:** Trachoma remains the leading infectious cause of blindness worldwide, with Sub-Saharan Africa bearing over 85% of the global burden and Ethiopia alone accounting for more than half of all cases. Yet publicly available preprocessed datasets for automated trachoma classification are scarce, and none originate from the most affected region. Raw clinical photographs of eyelids contain significant background noise that hinders direct use in machine learning pipelines. We present OPTED, an open-source preprocessed trachoma eye dataset constructed using the Segment Anything Model 3 (SAM 3) for automated region-of-interest extraction. We describe a reproducible four-step pipeline: (1) text-prompt-based zero-shot segmentation of the tarsal conjunctiva using SAM 3, (2) background removal and bounding-box cropping with alignment, (3) quality filtering based on confidence scores, and (4) Lanczos resizing to 224x224 pixels. A separate prompt-selection stage identifies the optimal text prompt, and manual quality assurance verifies outputs. Through comparison of five candidate prompts on all 2,832 known-label images, we identify "inner surface of eyelid with red tissue" as optimal, achieving a mean confidence of 0.872 (std 0.070) and 99.5% detection rate (the remaining 13 images are recovered via fallback prompts). The pipeline produces outputs in two formats: cropped and aligned images preserving the original aspect ratio, and standardized 224x224 images ready for pre-trained architectures. The OPTED dataset, preprocessing code, and all experimental artifacts are released as open source to facilitate reproducible trachoma classification research.
>
---
#### [new 025] Models as Lego Builders: Assembling Malice from Benign Blocks via Semantic Blueprints
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究LVLMs的安全漏洞，通过语义槽填充实现恶意输出。属于安全漏洞分析任务，解决模型在黑盒环境下被攻击的问题，提出StructAttack框架进行有效攻击。**

- **链接: [https://arxiv.org/pdf/2603.07590](https://arxiv.org/pdf/2603.07590)**

> **作者:** Chenxi Li; Xianggan Liu; Dake Shen; Yaosong Du; Zhibo Yao; Hao Jiang; Linyi Jiang; Chengwei Cao; Jingzhe Zhang; RanYi Peng; Peiling Bai; Xiande Huang
>
> **摘要:** Despite the rapid progress of Large Vision-Language Models (LVLMs), the integration of visual modalities introduces new safety vulnerabilities that adversaries can exploit to elicit biased or malicious outputs. In this paper, we demonstrate an underexplored vulnerability via semantic slot filling, where LVLMs complete missing slot values with unsafe content even when the slot types are deliberately crafted to appear benign. Building on this finding, we propose StructAttack, a simple yet effective single-query jailbreak framework under black-box settings. StructAttack decomposes a harmful query into a central topic and a set of benign-looking slot types, then embeds them as structured visual prompts (e.g., mind maps, tables, or sunburst diagrams) with small random perturbations. Paired with a completion-guided instruction, LVLMs automatically recompose the concealed semantics and generate unsafe outputs without triggering safety mechanisms. Although each slot appears benign in isolation (local benignness), StructAttack exploits LVLMs' reasoning to assemble these slots into coherent harmful semantics. Extensive experiments on multiple models and benchmarks show the efficacy of our proposed StructAttack.
>
---
#### [new 026] Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition
- **分类: cs.CV**

- **简介: 该论文属于零样本图像分类任务，旨在解决VLMs中提示工程不足和适应性差的问题。通过引入概念引导的贝叶斯框架，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07911](https://arxiv.org/pdf/2603.07911)**

> **作者:** Hui Liu; Kecheng Chen; Jialiang Wang; Xianming Liu; Wenya Wang; Haoliang Li
>
> **备注:** 19 pages, Accepted by CVPR 2026
>
> **摘要:** Vision-Language Models (VLMs), such as CLIP, have significantly advanced zero-shot image recognition. However, their performance remains limited by suboptimal prompt engineering and poor adaptability to target classes. While recent methods attempt to improve prompts through diverse class descriptions, they often rely on heuristic designs, lack versatility, and are vulnerable to outlier prompts. This paper enhances prompt by incorporating class-specific concepts. By treating concepts as latent variables, we rethink zero-shot image classification from a Bayesian perspective, casting prediction as marginalization over the concept space, where each concept is weighted by a prior and a test-image conditioned likelihood. This formulation underscores the importance of both a well-structured concept proposal distribution and the refinement of concept priors. To construct an expressive and efficient proposal distribution, we introduce a multi-stage concept synthesis pipeline driven by LLMs to generate discriminative and compositional concepts, followed by a Determinantal Point Process to enforce diversity. To mitigate the influence of outlier concepts, we propose a training-free, adaptive soft-trim likelihood, which attenuates their impact in a single forward pass. We further provide robustness guarantees and derive multi-class excess risk bounds for our framework. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art approaches, validating its effectiveness in zero-shot image classification. Our code is available at this https URL.
>
---
#### [new 027] PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition
- **分类: cs.CV**

- **简介: 该论文提出PRISM模型，解决文本到动作生成中的潜在空间设计和条件注入问题，通过结构化潜空间和无噪声条件注入，提升生成质量与多任务适应性。**

- **链接: [https://arxiv.org/pdf/2603.08590](https://arxiv.org/pdf/2603.08590)**

> **作者:** Zeyu Ling; Qing Shuai; Teng Zhang; Shiyang Li; Bo Han; Changqing Zou
>
> **摘要:** Text-to-motion generation has advanced rapidly, yet two challenges persist. First, existing motion autoencoders compress each frame into a single monolithic latent vector, entangling trajectory and per-joint rotations in an unstructured representation that downstream generators struggle to model faithfully. Second, text-to-motion, pose-conditioned generation, and long-horizon sequential synthesis typically require separate models or task-specific mechanisms, with autoregressive approaches suffering from severe error accumulation over extended rollouts. We present PRISM, addressing each challenge with a dedicated contribution. (1) A joint-factorized motion latent space: each body joint occupies its own token, forming a structured 2D grid (time joints) compressed by a causal VAE with forward-kinematics supervision. This simple change to the latent space -- without modifying the generator -- substantially improves generation quality, revealing that latent space design has been an underestimated bottleneck. (2) Noise-free condition injection: each latent token carries its own timestep embedding, allowing conditioning frames to be injected as clean tokens (timestep0) while the remaining tokens are denoised. This unifies text-to-motion and pose-conditioned generation in a single model, and directly enables autoregressive segment chaining for streaming synthesis. Self-forcing training further suppresses drift in long rollouts. With these two components, we train a single motion generation foundation model that seamlessly handles text-to-motion, pose-conditioned generation, autoregressive sequential generation, and narrative motion composition, achieving state-of-the-art on HumanML3D, MotionHub, BABEL, and a 50-scenario user study.
>
---
#### [new 028] Fusion-Poly: A Polyhedral Framework Based on Spatial-Temporal Fusion for 3D Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D多目标跟踪任务，解决异步传感器数据融合问题。提出Fusion-Poly框架，实现时空联合跟踪，提升轨迹一致性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08199](https://arxiv.org/pdf/2603.08199)**

> **作者:** Xian Wu; Yitao Wu; Xiaoyu Li; Zijia Li; Lijun Zhao; Lining Sun
>
> **摘要:** LiDAR-camera 3D multi-object tracking (MOT) combines rich visual semantics with accurate depth cues to improve trajectory consistency and tracking reliability. In practice, however, LiDAR and cameras operate at different sampling rates. To maintain temporal alignment, existing data pipelines usually synchronize heterogeneous sensor streams and annotate them at a reduced shared frequency, forcing most prior methods to perform spatial fusion only at synchronized timestamps through projection-based or learnable cross-sensor association. As a result, abundant asynchronous observations remain underexploited, despite their potential to support more frequent association and more robust trajectory estimation over short temporal intervals. To address this limitation, we propose Fusion-Poly, a spatial-temporal fusion framework for 3D MOT that integrates asynchronous LiDAR and camera data. Fusion-Poly associates trajectories with multi-modal observations at synchronized timestamps and with single-modal observations at asynchronous timestamps, enabling higher-frequency updates of motion and existence states. The framework contains three key components: a frequency-aware cascade matching module that adapts to synchronized and asynchronous frames according to available detection modalities; a frequency-aware trajectory estimation module that maintains trajectories through high-frequency motion prediction, differential updates, and confidence-calibrated lifecycle management; and a full-state observation alignment module that improves cross-modal consistency at synchronized timestamps by optimizing image-projection errors. On the nuScenes test set, Fusion-Poly achieves 76.5% AMOTA, establishing a new state of the art among tracking-by-detection 3D MOT methods. Extensive ablation studies further validate the effectiveness of each component. Code will be released.
>
---
#### [new 029] Brain-WM: Brain Glioblastoma World Model
- **分类: cs.CV**

- **简介: 该论文属于医疗影像分析任务，旨在解决GBM治疗与肿瘤演变动态关系建模问题。提出Brain-WM模型，联合预测治疗方案和生成MRI，提升治疗规划准确性。**

- **链接: [https://arxiv.org/pdf/2603.07562](https://arxiv.org/pdf/2603.07562)**

> **作者:** Chenhui Wang; Boyun Zheng; Liuxin Bao; Zhihao Peng; Peter Y.M. Woo; Hongming Shan; Yixuan Yuan
>
> **摘要:** Precise prognostic modeling of glioblastoma (GBM) under varying treatment interventions is essential for optimizing clinical outcomes. While generative AI has shown promise in simulating GBM evolution, existing methods typically treat interventions as static conditional inputs rather than dynamic decision variables. Consequently, they fail to capture the complex, reciprocal interplay between tumor evolution and treatment response. To bridge this gap, we present Brain-WM, a pioneering brain GBM world model that unifies next-step treatment prediction and future MRI generation, thereby capturing the co-evolutionary dynamics between tumor and treatment. Specifically, Brain-WM encodes spatiotemporal dynamics into a shared latent space for joint autoregressive treatment prediction and flow-based future MRI generation. Then, instead of a conventional monolithic framework, Brain-WM adopts a novel Y-shaped Mixture-of-Transformers (MoT) architecture. This design structurally disentangles heterogeneous objectives, successfully leveraging cross-task synergies while preventing feature collapse. Finally, a synergistic multi-timepoint mask alignment objective explicitly anchors latent representations to anatomically grounded tumor structures and progression-aware semantics. Extensive validation on internal and external multi-institutional cohorts demonstrates the superiority of Brain-WM, achieving 91.5% accuracy in treatment planning and SSIMs of 0.8524, 0.8581, and 0.8404 for FLAIR, T1CE, and T2W sequences, respectively. Ultimately, Brain-WM offers a robust clinical sandbox for optimizing patient healthcare. The source code is made available at this https URL.
>
---
#### [new 030] SiMO: Single-Modality-Operable Multimodal Collaborative Perception
- **分类: cs.CV**

- **简介: 该论文属于多模态协同感知任务，旨在解决传感器失效时的语义不一致问题。提出SiMO框架，通过LAMMA和“Pretrain-Align-Fuse-RD”策略实现多模态融合与独立性保障。**

- **链接: [https://arxiv.org/pdf/2603.08240](https://arxiv.org/pdf/2603.08240)**

> **作者:** Jiageng Wen; Shengjie Zhao; Bing Li; Jiafeng Huang; Kenan Ye; Hao Deng
>
> **备注:** Accepted to ICLR 2026. This arXiv version includes an additional appendix (Appendix 15) containing further philosophical discussion not included in the official ICLR peer-reviewed version
>
> **摘要:** Collaborative perception integrates multi-agent perspectives to enhance the sensing range and overcome occlusion issues. While existing multimodal approaches leverage complementary sensors to improve performance, they are highly prone to failure--especially when a key sensor like LiDAR is unavailable. The root cause is that feature fusion leads to semantic mismatches between single-modality features and the downstream modules. This paper addresses this challenge for the first time in the field of collaborative perception, introducing Single-Modality-Operable Multimodal Collaborative Perception (SiMO). By adopting the proposed Length-Adaptive Multi-Modal Fusion (LAMMA), SiMO can adaptively handle remaining modal features during modal failures while maintaining consistency of the semantic space. Additionally, leveraging the innovative "Pretrain-Align-Fuse-RD" training strategy, SiMO addresses the issue of modality competition--generally overlooked by existing methods--ensuring the independence of each individual modality branch. Experiments demonstrate that SiMO effectively aligns multimodal features while simultaneously preserving modality-specific features, enabling it to maintain optimal performance across all individual modalities. The implementation details can be found in this https URL.
>
---
#### [new 031] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决LVLMs的语言偏差和敏感性问题。提出SCI框架和DRBench评估体系，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07659](https://arxiv.org/pdf/2603.07659)**

> **作者:** Kaihua Tang; Jiaxin Qi; Jinli Ou; Yuhua Zheng; Jianqiang Huang
>
> **备注:** Accepted to CVPR 2026. Code: this https URL
>
> **摘要:** The emergence of Large Language Models (LLMs) has driven rapid progress in multi-modal learning, particularly in the development of Large Vision-Language Models (LVLMs). However, existing LVLM training paradigms place excessive reliance on the LLM component, giving rise to two critical robustness challenges: language bias and language sensitivity. To address both issues simultaneously, we propose a novel Self-Critical Inference (SCI) framework that extends Visual Contrastive Decoding by conducting multi-round counterfactual reasoning through both textual and visual perturbations. This process further introduces a new strategy for improving robustness by scaling the number of counterfactual rounds. Moreover, we also observe that failure cases of LVLMs differ significantly across models, indicating that fixed robustness benchmarks may not be able to capture the true reliability of LVLMs. To this end, we propose the Dynamic Robustness Benchmark (DRBench), a model-specific evaluation framework targeting both language bias and sensitivity issues. Extensive experiments show that SCI consistently outperforms baseline methods on DRBench, and that increasing the number of inference rounds further boosts robustness beyond existing single-step counterfactual reasoning methods.
>
---
#### [new 032] T2SGrid: Temporal-to-Spatial Gridification for Video Temporal Grounding
- **分类: cs.CV**

- **简介: 该论文属于视频时间定位任务，解决现有方法在处理时间动态信息时的局限性。提出T2SGrid框架，将视频时间理解转化为空间问题，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.06973](https://arxiv.org/pdf/2603.06973)**

> **作者:** Chaohong Guo; Yihan He; Yongwei Nie; Fei Ma; Xuemiao Xu; Chengjiang Long
>
> **摘要:** Video Temporal Grounding (VTG) aims to localize the video segment that corresponds to a natural language query, which requires a comprehensive understanding of complex temporal dynamics. Existing Vision-LMMs typically perceive temporal dynamics via positional encoding, text-based timestamps, or visual frame numbering. However, these approaches exhibit notable limitations: assigning each frame a text-based timestamp token introduces additional computational overhead and leads to sparsity in visual attention, positional encoding struggles to capture absolute temporal information, and visual frame numbering often compromises spatial detail. To address these issues, we propose Temporal to Spatial Gridification (T2SGrid), a novel framework that reformulates video temporal understanding as a spatial understanding task. The core idea of T2SGrid is to process video content in clips rather than individual frames. we employ a overlapping sliding windows mechanism to segment the video into temporal clips. Within each window, frames are arranged chronologically in a row-major order into a composite grid image, effectively transforming temporal sequences into structured 2D layouts. The gridification not only encodes temporal information but also enhances local attention within each grid. Furthermore, T2SGrid enables the use of composite text timestamps to establish global temporal awareness. Experiments on standard VTG benchmarks demonstrate that T2SGrid achieves superior performance.
>
---
#### [new 033] XAI and Few-shot-based Hybrid Classification Model for Plant Leaf Disease Prognosis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于植物病害分类任务，旨在解决小样本下疾病阶段识别问题。融合XAI与FSL，构建可解释的分类模型，提升诊断准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2603.06676](https://arxiv.org/pdf/2603.06676)**

> **作者:** Diana Susan Joseph; Pranav M Pawar; Raja Muthalagu; Mithun Mukharjee
>
> **备注:** 27 pages, 8 figures
>
> **摘要:** Performing a timely and accurate identification of crop diseases is vital to maintain agricultural productivity and food security. The current work presents a hybrid few-shot learning model that integrates Explainable Artificial Intelligence (XAI) and Few-Shot Learning (FSL) to address the challenge of identifying and classifying the stages of disease of the diseases of maize, rice, and wheat leaves under limited annotated data conditions. The proposed model integrates Siamese and Prototypical Networks within an episodic training paradigm to effectively learn discriminative disease features from a few examples. To ensure model transparency and trustworthiness, Gradient-weighted Class Activation Mapping (Grad-CAM) is employed for visualizing key decision regions in the leaf images, offering interpretable insights into the classification process. Experimental evaluations on custom few-shot datasets developed in the study prove that the model consistently achieves high accuracy, precision, recall, and F1-scores, frequently exceeding 92% across various disease stages. Comparative analyses against baseline FSL models further confirm the superior performance and explainability of the proposed approach. The framework offers a promising solution for real-world, data-constrained agricultural disease monitoring applications.
>
---
#### [new 034] Tracking Phenological Status and Ecological Interactions in a Hawaiian Cloud Forest Understory using Low-Cost Camera Traps and Visual Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于生态监测任务，旨在解决热带植物物候和动植物相互作用的观测难题。通过低成像相机和视觉模型，分析植物周期性变化及动物访问数据，揭示生态驱动因素。**

- **链接: [https://arxiv.org/pdf/2603.07817](https://arxiv.org/pdf/2603.07817)**

> **作者:** Luke Meyers; Anirudh Potlapally; Yuyan Chen; Mike Long; Tanya Berger-Wolf; Hari Subramoni; Remi Megret; Daniel Rubenstein
>
> **摘要:** Plant phenology, the study of cyclical events such as leafing out, flowering, or fruiting, has wide ecological impacts but is broadly understudied, especially in the tropics. Image analysis has greatly enhanced remote phenological monitoring, yet capturing phenology at the individual level remains challenging. In this project, we deployed low-cost, animal-triggered camera traps at the Pu'u Maka'ala Natural Area Reserve in Hawaii to simultaneously document shifts in plant phenology and flora-faunal interactions. Using a combination of foundation vision models and traditional computer vision methods, we measure phenological trends from images comparable to on-the-ground observations without relying on supervised learning techniques. These temporally fine-grained phenology measurements from camera-trap images uncover trends that coarser traditional sampling fails to detect. When combined with detailed visitation data detected from images, these trends can begin to elucidate drivers of both plant phenology and animal ecology.
>
---
#### [new 035] $Δ$VLA: Prior-Guided Vision-Language-Action Models via World Knowledge Variation
- **分类: cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决动作生成中对世界变化过程推理不足的问题。提出$\Delta$VLA框架，通过建模世界知识变化而非预测绝对状态，提升动作生成效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.08361](https://arxiv.org/pdf/2603.08361)**

> **作者:** Yijie Zhu; Jie He; Rui Shao; Kaishen Yuan; Tao Tan; Xiaochen Yuan; Zitong Yu
>
> **摘要:** Recent vision-language-action (VLA) models have significantly advanced robotic manipulation by unifying perception, reasoning, and control. To achieve such integration, recent studies adopt a predictive paradigm that models future visual states or world knowledge to guide action generation. However, these models emphasize forecasting outcomes rather than reasoning about the underlying process of change, which is essential for determining how to act. To address this, we propose $\Delta$VLA, a prior-guided framework that models world-knowledge variations relative to an explicit current-world knowledge prior for action generation, rather than regressing absolute future world states. Specifically, 1) to construct the current world knowledge prior, we propose the Prior-Guided WorldKnowledge Extractor (PWKE). It extracts manipulable regions, spatial relations, and semantic cues from the visual input, guided by auxiliary heads and prior pseudo labels, thus reducing redundancy. 2) Building upon this, to represent how world knowledge evolves under actions, we introduce the Latent World Variation Quantization (LWVQ). It learns a discrete latent space via a VQ-VAE objective to encode world knowledge variations, shifting prediction from full modalities to compact latent. 3)Moreover, to mitigate interference during variation modeling, we design the Conditional Variation Attention (CV-Atten), whichpromotes disentangled learning and preserves the independence of knowledge representations. Extensive experiments on both simulated benchmarks and real-world robotic tasks demonstrate $\Delta$VLA achieves state-of-the-art performance while improving efficiency. Code and real-world execution videos are available at this https URL.
>
---
#### [new 036] PARSE: Part-Aware Relational Spatial Modeling
- **分类: cs.CV**

- **简介: 该论文提出PARSE框架，解决3D场景中对象间关系建模问题，通过部分级关系图增强空间推理与物理一致性。**

- **链接: [https://arxiv.org/pdf/2603.07704](https://arxiv.org/pdf/2603.07704)**

> **作者:** Yinuo Bai; Peijun Xu; Kuixiang Shao; Yuyang Jiao; Jingxuan Zhang; Kaixin Yao; Jiayuan Gu; Jingyi Yu
>
> **摘要:** Inter-object relations underpin spatial intelligence, yet existing representations -- linguistic prepositions or object-level scene graphs -- are too coarse to specify which regions actually support, contain, or contact one another, leading to ambiguous and physically inconsistent layouts. To address these ambiguities, a part-level formulation is needed; therefore, we introduce PARSE, a framework that explicitly models how object parts interact to determine feasible and spatially grounded scene configurations. PARSE centers on the Part-centric Assembly Graph (PAG), which encodes geometric relations between specific object parts, and a Part-Aware Spatial Configuration Solver that converts these relations into geometric constraints to assemble collision-free, physically valid scenes. Using PARSE, we build PARSE-10K, a dataset of 10,000 3D indoor scenes constructed from real-image layout priors and a curated part-annotated shape database, each with dense contact structures and a part-level contact graph. With this structured, spatially grounded supervision, fine-tuning Qwen3-VL on PARSE-10K yields stronger object-level layout reasoning and more accurate part-level relation understanding; furthermore, leveraging PAGs as structural priors in 3D generation models leads to scenes with substantially improved physical realism and structural complexity. Together, these results show that PARSE significantly advances geometry-grounded spatial reasoning and supports the generation of physically consistent 3D scenes.
>
---
#### [new 037] RPG-SAM: Reliability-Weighted Prototypes and Geometric Adaptive Threshold Selection for Training-Free One-Shot Polyp Segmentation
- **分类: cs.CV**

- **简介: 该论文提出RPG-SAM，解决训练-free 一次性息肉分割中的区域与响应异质性问题，通过可靠性加权原型和几何自适应阈值提升分割精度。**

- **链接: [https://arxiv.org/pdf/2603.07436](https://arxiv.org/pdf/2603.07436)**

> **作者:** Weikun Lin; Yunhao Bai; Yan Wang
>
> **备注:** Under review at MICCAI 2026. 8 pages, 3 figures
>
> **摘要:** Training-free one-shot segmentation offers a scalable alternative to expert annotations where knowledge is often transferred from support images and foundation models. But existing methods often treat all pixels in support images and query response intensities models in a homogeneous way. They ignore the regional heterogeity in support images and response heterogeity in this http URL resolve this, we propose RPG-SAM, a framework that systematically tackles these heterogeneity gaps. Specifically, to address regional heterogeneity, we introduce Reliability-Weighted Prototype Mining (RWPM) to prioritize high-fidelity support features while utilizing background anchors as contrastive references for noise suppression. To address response heterogeneity, we develop Geometric Adaptive Selection (GAS) to dynamically recalibrate binarization thresholds by evaluating the morphological consensus of candidates. Finally, an iterative refinement loop method is designed to polishes anatomical boundaries. By accounting for multi-layered information heterogeneity, RPG-SAM achieves a 5.56\% mIoU improvement on the Kvasir dataset. Code will be released.
>
---
#### [new 038] Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence
- **分类: cs.CV**

- **简介: 该论文提出Holi-Spatial，一个自动构建的大规模3D空间智能数据集，解决传统方法依赖少量标注数据导致的扩展性差和性能受限问题。**

- **链接: [https://arxiv.org/pdf/2603.07660](https://arxiv.org/pdf/2603.07660)**

> **作者:** Yuanyuan Gao; Hao Li; Yifei Liu; Xinhao Ji; Yuning Gong; Yuanjun Liao; Fangfu Liu; Manyuan Zhang; Yuchen Yang; Dan Xu; Xue Yang; Huaxi Huang; Hongjie Zhang; Ziwei Liu; Xiao Sun; Dingwen Zhang; Zhihang Zhong
>
> **备注:** project page: this https URL
>
> **摘要:** The pursuit of spatial intelligence fundamentally relies on access to large-scale, fine-grained 3D data. However, existing approaches predominantly construct spatial understanding benchmarks by generating question-answer (QA) pairs from a limited number of manually annotated datasets, rather than systematically annotating new large-scale 3D scenes from raw web data. As a result, their scalability is severely constrained, and model performance is further hindered by domain gaps inherent in these narrowly curated datasets. In this work, we propose Holi-Spatial, the first fully automated, large-scale, spatially-aware multimodal dataset, constructed from raw video inputs without human intervention, using the proposed data curation pipeline. Holi-Spatial supports multi-level spatial supervision, ranging from geometrically accurate 3D Gaussian Splatting (3DGS) reconstructions with rendered depth maps to object-level and relational semantic annotations, together with corresponding spatial Question-Answer (QA) pairs. Following a principled and systematic pipeline, we further construct Holi-Spatial-4M, the first large-scale, high-quality 3D semantic dataset, containing 12K optimized 3DGS scenes, 1.3M 2D masks, 320K 3D bounding boxes, 320K instance captions, 1.2M 3D grounding instances, and 1.2M spatial QA pairs spanning diverse geometric, relational, and semantic reasoning tasks. Holi-Spatial demonstrates exceptional performance in data curation quality, significantly outperforming existing feed-forward and per-scene optimized methods on datasets such as ScanNet, ScanNet++, and DL3DV. Furthermore, fine-tuning Vision-Language Models (VLMs) on spatial reasoning tasks using this dataset has also led to substantial improvements in model performance.
>
---
#### [new 039] FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，旨在解决长尾分布数据下的检测难题。通过融合视觉基础模型的语义和深度先验，提出FOMO-3D方法提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.08611](https://arxiv.org/pdf/2603.08611)**

> **作者:** Anqi Joyce Yang; James Tu; Nikita Dvornik; Enxu Li; Raquel Urtasun
>
> **备注:** Published at 9th Annual Conference on Robot Learning (CoRL 2025)
>
> **摘要:** In order to navigate complex traffic environments, self-driving vehicles must recognize many semantic classes pertaining to vulnerable road users or traffic control devices. However, many safety-critical objects (e.g., construction worker) appear infrequently in nominal traffic conditions, leading to a severe shortage of training examples from driving data alone. Recent vision foundation models, which are trained on a large corpus of data, can serve as a good source of external prior knowledge to improve generalization. We propose FOMO-3D, the first multi-modal 3D detector to leverage vision foundation models for long-tailed 3D detection. Specifically, FOMO-3D exploits rich semantic and depth priors from OWLv2 and Metric3Dv2 within a two-stage detection paradigm that first generates proposals with a LiDAR-based branch and a novel camera-based branch, and refines them with attention especially to image features from OWL. Evaluations on real-world driving data show that using rich priors from vision foundation models with careful multi-modal fusion designs leads to large gains for long-tailed 3D detection. Project website is at this https URL.
>
---
#### [new 040] HyperTokens: Controlling Token Dynamics for Continual Video-Language Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于持续视频问答任务，解决多模态大模型中的任务干扰与记忆成本问题。提出HyperTokens，通过动态生成token和正则化方法，提升模型持续学习能力。**

- **链接: [https://arxiv.org/pdf/2603.06662](https://arxiv.org/pdf/2603.06662)**

> **作者:** Toan Nguyen; Yang Liu; Celso De Melo; Flora D. Salim
>
> **摘要:** Continual VideoQA with multimodal LLMs is hindered by interference between tasks and the prohibitive cost of storing task-specific prompts. We introduce HyperTokens, a transformer-based token generator that produces fine-tuning tokens on demand, giving explicit control over prompt updates while keeping memory fixed. To suppress forgetting, we propose meta-inspired regularisers that look ahead to avoid task-specific sharp directions and anchor the evolving generator to prior tasks. We further connect our objective to sharpness-aware optimisation, providing insight into why it encourages flatter cross-task minima and improves retention. Beyond regularisation, HyperTokens exploits lightweight auxiliary multimodal supervision through shared generation weights; guided by a causal perspective, we design feasible objectives and surrogate mutual-information losses to regularise anti-causal cross-modal directions. Across two standard continual VideoQA benchmarks, HyperTokens achieves higher average accuracy with substantially lower forgetting. Finally, we introduce a challenging cross-modal ImageQA->VideoQA protocol and show that HyperTokens enables robust continual transfer in this setting.
>
---
#### [new 041] Boosting MLLM Spatial Reasoning with Geometrically Referenced 3D Scene Representations
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型任务，旨在提升MLLM在3D空间推理上的能力。通过引入GR3D表示，将3D几何信息转化为文本引用，增强模型的空间理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.08592](https://arxiv.org/pdf/2603.08592)**

> **作者:** Jiangye Yuan; Gowri Kumar; Baoyuan Wang
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have achieved remarkable success in 2D visual understanding, their ability to reason about 3D space remains limited. To address this gap, we introduce geometrically referenced 3D scene representations (GR3D). Given a set of input images, GR3D annotates objects in the images with unique IDs and encodes their 3D geometric attributes as textual references indexed by these IDs. This representation enables MLLMs to interpret 3D cues using their advanced language-based skills in mathematical reasoning, while concurrently analyzing 2D visual features in a tightly coupled way. We present a simple yet effective approach based on GR3D, which requires no additional training and is readily applicable to different MLLMs. Implemented in a zero-shot setting, our approach boosts GPT-5's performance on VSI-Bench by 8% overall and more than 11% on tasks that rely heavily on spatial layout understanding. Qualitative studies further demonstrate that GR3D empowers MLLMs to perform complex spatial reasoning with highly sparse input views.
>
---
#### [new 042] MotionBits: Video Segmentation through Motion-Level Analysis of Rigid Bodies
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视频分割任务，旨在解决运动刚体的准确分割问题。提出MotionBit概念及无需学习的图方法，在基准上提升性能37.3%。**

- **链接: [https://arxiv.org/pdf/2603.06846](https://arxiv.org/pdf/2603.06846)**

> **作者:** Howard H. Qian; Kejia Ren; Yu Xiang; Vicente Ordonez; Kaiyu Hang
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Rigid bodies constitute the smallest manipulable elements in the real world, and understanding how they physically interact is fundamental to embodied reasoning and robotic manipulation. Thus, accurate detection, segmentation, and tracking of moving rigid bodies is essential for enabling reasoning modules to interpret and act in diverse environments. However, current segmentation models trained on semantic grouping are limited in their ability to provide meaningful interaction-level cues for completing embodied tasks. To address this gap, we introduce MotionBit, a novel concept that, unlike prior formulations, defines the smallest unit in motion-based segmentation through kinematic spatial twist equivalence, independent of semantics. In this paper, we contribute (1) the MotionBit concept and definition, (2) a hand-labeled benchmark, called MoRiBo, for evaluating moving rigid-body segmentation across robotic manipulation and human-in-the-wild videos, and (3) a learning-free graph-based MotionBits segmentation method that outperforms state-of-the-art embodied perception methods by 37.3\% in macro-averaged mIoU on the MoRiBo benchmark. Finally, we demonstrate the effectiveness of MotionBits segmentation for downstream embodied reasoning and manipulation tasks, highlighting its importance as a fundamental primitive for understanding physical interactions.
>
---
#### [new 043] PCFEx: Point Cloud Feature Extraction for Graph Neural Networks
- **分类: cs.CV; cs.IR**

- **简介: 该论文将图神经网络应用于点云数据处理，解决人体姿态估计和活动识别问题，提出PCFEx特征提取方法，提升点云处理精度。**

- **链接: [https://arxiv.org/pdf/2603.08540](https://arxiv.org/pdf/2603.08540)**

> **作者:** Abdullah Al Masud; Shi Xintong; Mondher Bouazizi; Ohtsuki Tomoaki
>
> **备注:** ©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Graph neural networks (GNNs) have gained significant attention for their effectiveness across various domains. This study focuses on applying GNN to process 3D point cloud data for human pose estimation (HPE) and human activity recognition (HAR). We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph. Moreover, we introduce a GNN architecture designed to efficiently process these features. Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR. The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models. This work demonstrates the great potential of feature extraction incorporated with GNN modeling approach to enhance the precision of point cloud processing.
>
---
#### [new 044] From Reactive to Map-Based AI: Tuned Local LLMs for Semantic Zone Inference in Object-Goal Navigation
- **分类: cs.CV**

- **简介: 该论文属于对象目标导航任务，解决传统AI在未知环境中探索效率低的问题。通过结合语义推理与地图系统，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.08086](https://arxiv.org/pdf/2603.08086)**

> **作者:** Yudai Noda; Kanji Tanaka
>
> **备注:** 6 pages, 5 figures, technical report
>
> **摘要:** Object-Goal Navigation (ObjectNav) requires an agent to find and navigate to a target object category in unknown environments. While recent Large Language Model (LLM)-based agents exhibit zero-shot reasoning, they often rely on a "reactive" paradigm that lacks explicit spatial memory, leading to redundant exploration and myopic behaviors. To address these limitations, we propose a transition from reactive AI to "Map-Based AI" by integrating LLM-based semantic inference with a hybrid topological-grid mapping system. Our framework employs a fine-tuned Llama-2 model via Low-Rank Adaptation (LoRA) to infer semantic zone categories and target existence probabilities from verbalized object observations. In this study, a "zone" is defined as a functional area described by the set of observed objects, providing crucial semantic co-occurrence cues for finding the target. This semantic information is integrated into a topological graph, enabling the agent to prioritize high-probability areas and perform systematic exploration via Traveling Salesman Problem (TSP) optimization. Evaluations in the AI2-THOR simulator demonstrate that our approach significantly outperforms traditional frontier exploration and reactive LLM baselines, achieving a superior Success Rate (SR) and Success weighted by Path Length (SPL).
>
---
#### [new 045] Class Visualizations and Activation Atlases for Enhancing Interpretability in Deep Learning-Based Computational Pathology
- **分类: cs.CV**

- **简介: 该论文属于计算病理学任务，旨在提升深度学习模型的可解释性。通过类可视化和激活图谱方法评估模型，分析其在组织和多器官癌症分类中的表现与专家一致性。**

- **链接: [https://arxiv.org/pdf/2603.07170](https://arxiv.org/pdf/2603.07170)**

> **作者:** Marco Gustav; Fabian Wolf; Christina Glasner; Nic G. Reitsam; Stefan Schulz; Kira Aschenbroich; Bruno Märkl; Sebastian Foersch; Jakob Nikolas Kather
>
> **摘要:** The rapid adoption of transformer-based models in computational pathology has enabled prediction of molecular and clinical biomarkers from H&E whole-slide images, yet interpretability has not kept pace with model complexity. While attribution- and generative-based methods are common, feature visualization approaches such as class visualizations (CVs) and activation atlases (AAs) have not been systematically evaluated for these models. We developed a visualization framework and assessed CVs and AAs for a transformer-based foundation model across tissue and multi-organ cancer classification tasks with increasing label granularity. Four pathologists annotated real and generated images to quantify inter-observer agreement, complemented by attribution and similarity metrics. CVs preserved recognizability for morphologically distinct tissues but showed reduced separability for overlapping cancer subclasses. In tissue classification, agreement decreased from Fleiss k = 0.75 (scans) to k = 0.31 (CVs), with similar trends in cancer subclass tasks. AAs revealed layer-dependent organization: coarse tissue-level concepts formed coherent regions, whereas finer subclasses exhibited dispersion and overlap. Agreement was moderate for tissue classification (k = 0.58), high for coarse cancer groupings (k = 0.82), and low at subclass level (k = 0.11). Atlas separability closely tracked expert agreement on real images, indicating that representational ambiguity reflects intrinsic pathological complexity. Attribution-based metrics approximated expert variability in low-complexity settings, whereas perceptual and distributional metrics showed limited alignment. Overall, concept-level feature visualization reveals structured morphological manifolds in transformer-based pathology models and provides a framework for expert-centered interrogation of learned representations across label granularities.
>
---
#### [new 046] A Hybrid Machine Learning Model for Cerebral Palsy Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在早期检测脑瘫。通过融合多个CNN模型和Bi-LSTM构建混合模型，提升诊断准确率。**

- **链接: [https://arxiv.org/pdf/2603.06803](https://arxiv.org/pdf/2603.06803)**

> **作者:** Karan Kumar Singh; Nikita Gajbhiye; Gouri Sankar Mishra
>
> **备注:** 28 pages, 19 figures, 8 tables. This manuscript is based on the article published in the International Journal of Intelligent Systems and Applications in Engineering (IJISAE), 2024. The arXiv version is provided for open accessibility and wider dissemination
>
> **摘要:** The development of effective treatments for Cerebral Palsy (CP) can begin with the early identification of affected children while they are still in the early stages of the disorder. Pathological issues in the brain can be better diagnosed with the use of one of many medical imaging techniques. Magnetic Resonance Imaging (MRI) has revolutionized medical imaging with its unparalleled image resolution. A unique Machine Learning (ML) model that was built to identify CP disorder is presented in this paper. The model is intended to assist in the early diagnosis of CP in newborns. In this study, the brain MRI images dataset was first collected, and then the preprocessing techniques were applied to this dataset to make it ready for use in the proposed model. Following this, the proposed model was constructed by combining three CNN models, specifically VGG 19, Efficient-Net, and the ResNet50 model, to extract features from the image. Following this, a Bi-LSTM was utilized as a classifier to determine whether or not CP was present, and finally, the proposed model was employed for training and testing. The results show that the proposed model achieved an accuracy of 98.83%, which is higher than VGG-19 (96.79%), Efficient-Net (97.29%), and VGG-16 (97.50%).. When the suggested model is compared to other models that have been pre-trained in the past, the accuracy scores seem to be much higher.
>
---
#### [new 047] Deep Expert Injection for Anchoring Retinal VLMs with Domain-Specific Knowledge
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉问答任务，解决LVLM在眼科诊断中因缺乏领域知识导致的感知与推理问题。通过引入专家知识注入框架，提升模型对病理特征的识别与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.07131](https://arxiv.org/pdf/2603.07131)**

> **作者:** Shuai Lu; Meng Wang; Jia Guo; Jiawei Du; Bo Liu; Shengzhu Yang; Weihang Zhang; Huazhu Fu; Huiqi Li
>
> **摘要:** Large Vision Language Models (LVLMs) show immense potential for automated ophthalmic diagnosis. However, their clinical deployment is severely hindered by lacking domain-specific knowledge. In this work, we identify two structural deficiencies hindering reliable medical reasoning: 1) the Perception Gap, where general-purpose visual encoders fail to resolve fine-grained pathological cues (e.g., microaneurysms); and 2) the Reasoning Gap, where sparse visual evidence is progressively overridden by massive language priors in deeper transformer layers, leading to ungrounded hallucinations. To bridge these gaps, we propose EyExIn, a data-efficient framework designed to anchor retinal VLMs with expert knowledge via a Deep Expert Injection mechanism. Our architecture employs an Expert-Aware Dual-Stream encoding strategy that decouples visual representation into a general stream for anatomical context and a specialized expert stream for pathological semantics. To ensure high-fidelity integration, we design a Semantic-Adaptive Gated Fusion module, which dynamically amplifies subtle lesion signals while filtering irrelevant background noise. Furthermore, we introduce Adaptive Deep Expert Injection to embed persistent "Vision Anchors" by integrating fused visual features as residual biases directly into intermediate LLM layers. This mechanism creates a visual shortcut that forces the reasoning stack to remain strictly grounded in visual evidence. Extensive experiments across four benchmarks demonstrate that our model consistently outperforms massive proprietary systems. EyExIn significantly enhances domain-specific knowledge embedding and achieves state-of-the-art precision in ophthalmic visual question answering, advancing the development of trustworthy ophthalmic AI.
>
---
#### [new 048] It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决VLMs在真实场景中读取模拟钟表的困难。通过构建新数据集和优化方法提升模型的时空推理能力。**

- **链接: [https://arxiv.org/pdf/2603.08011](https://arxiv.org/pdf/2603.08011)**

> **作者:** Jaeha Choi; Jin Won Lee; Siwoo You; Jangho Lee
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Advances in vision-language models (VLMs) have achieved remarkable success on complex multimodal reasoning tasks, leading to the assumption that they should also excel at reading analog clocks. However, contrary to this expectation, our study reveals that reading analog clocks in real-world environments remains a significant challenge for state-of-the-art VLMs. Existing analog clock datasets are largely synthetic or planar with limited stylistic diversity and minimal background context, failing to capture the visual variability of real-world scenes. As a result, VLMs trained on such data exhibit weak spatial-temporal reasoning, frequently confusing the hour and minute hands and struggling under common visual conditions such as occlusion, lighting variation, and cluttered backgrounds. To address this issue, we introduce TickTockVQA, a human-annotated dataset containing analog clocks in diverse real-world scenarios. TickTockVQA provides explicit hour and minute annotations, and includes an AM/PM tag when it is inferable from the visual context. Furthermore, we propose Swap-DPO, a direct preference optimization based fine-tuning framework to align model reasoning toward accurate time interpretation. Experimental results demonstrate that our approach substantially enhances clock reading accuracy and robustness under real-world conditions, establishing a foundation for future research on spatial-temporal reasoning and visual understanding in VLMs.
>
---
#### [new 049] StructSAM: Structure- and Spectrum-Preserving Token Merging for Segment Anything Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出StructSAM，解决SAM模型中token合并导致边界模糊和信息泄露的问题，通过能量评分与网格筛选实现高效且边界保留的合并。**

- **链接: [https://arxiv.org/pdf/2603.07307](https://arxiv.org/pdf/2603.07307)**

> **作者:** Duy M. H. Nguyen; Tuan A. Tran; Duong Nguyen; Siwei Xie; Trung Q. Nguyen; Mai T. N. Truong; Daniel Palenicek; An T. Le; Michael Barz; TrungTin Nguyen; Tuan Dam; Ngan Le; Minh Vu; Khoa Doan; Vien Ngo; Pengtao Xie; James Zou; Daniel Sonntag; Jan Peters; Mathias Niepert
>
> **备注:** Firsrt version
>
> **摘要:** Recent token merging techniques for Vision Transformers (ViTs) provide substantial speedups by reducing the number of tokens processed by self-attention, often without retraining. However, their direct application to the Segment Anything Model (SAM) family is nontrivial: SAM's image encoder mixes windowed and global attention, and its mask decoder relies on dense, prompt-conditioned features for precise boundary prediction. We systematically evaluate representative token-merging methods on SAM and Medical SAM in a strict off-the-shelf setting, and find that existing destination-selection heuristics can erode boundaries and leak prompt information as merge rates increase. We propose \textbf{StructSAM}, a resolution-preserving merge-unmerge framework tailored to SAM. StructSAM computes a lightweight token-energy score from first-order feature gradients, uses grid-based flatness screening to protect boundary and prompt regions, and merges tokens within flat areas toward low-energy destinations with explicit token recovery. We further provide a spectral graph coarsening view showing that score-guided merging yields bounded Laplacian spectral distortion compared to random or window-restricted baselines. Across eight natural and medical benchmarks, StructSAM reduces encoder FLOPs by 25-30\% (up to 40\%+ with prompt-aware merging) with minor drops in mIoU/Dice, consistently outperforming ToMe, PiToMe, ToMeSD, VidToMe, and ALGM at the same compute.
>
---
#### [new 050] OSCAR: Occupancy-based Shape Completion via Acoustic Neural Implicit Representations
- **分类: cs.CV**

- **简介: 该论文属于3D形状补全任务，旨在解决超声图像中因声影导致的椎体结构不完整问题。通过结合声学信息与隐式表示，实现无需标签的准确重建。**

- **链接: [https://arxiv.org/pdf/2603.08279](https://arxiv.org/pdf/2603.08279)**

> **作者:** Magdalena Wysocki; Kadir Burak Buldu; Miruna-Alexandra Gafencu; Mohammad Farid Azampour; Nassir Navab
>
> **摘要:** Accurate 3D reconstruction of vertebral anatomy from ultrasound is important for guiding minimally invasive spine interventions, but it remains challenging due to acoustic shadowing and view-dependent signal variations. We propose an occupancy-based shape completion method that reconstructs complete 3D anatomical geometry from partial ultrasound observations. Crucially for intra-operative applications, our approach extracts the anatomical surface directly from the image, avoiding the need for anatomical labels during inference. This label-free completion relies on a coupled latent space representing both the image appearance and the underlying anatomical shape. By leveraging a Neural Implicit Representation (NIR) that jointly models both spatial occupancy and acoustic interactions, the method uses acoustic parameters to become implicitly aware of the unseen regions without explicit shadowing labels through tracking acoustic signal transmission. We show that this method outperforms state-of-the-art shape completion for B-mode ultrasound by 80% in HD95 score. We validate our approach both in-silico and on phantom US images with registered mesh models from CT labels, demonstrating accurate reconstruction of occluded anatomy and robust generalization across diverse imaging conditions. Code and data will be released on publication.
>
---
#### [new 051] OrdinalBench: A Benchmark Dataset for Diagnosing Generalization Limits in Ordinal Number Understanding of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出OrdinalBench，用于评估视觉语言模型在序数理解上的泛化能力。针对模型在处理大序数和复杂路径时表现下降的问题，构建了包含多种难度的任务数据集，并提供评估工具。**

- **链接: [https://arxiv.org/pdf/2603.07786](https://arxiv.org/pdf/2603.07786)**

> **作者:** Yusuke Tozaki; Hisashi Miyamori
>
> **备注:** Accepted as a Short Paper at VISAPP 2026
>
> **摘要:** Vision-Language Models (VLMs) have advanced across multimodal benchmarks but still show clear gaps in ordinal number understanding, i.e., the ability to track relative positions and generalize to large indices. We present OrdinalBench, a diagnostic benchmark that standardizes ordinal number understanding as an evaluation task for VLMs. The core task is N-th object identification, defined by a starting reference and traversal rule. Task difficulty is controlled along three axes: (i) ordinal magnitude, from small numbers to extreme cases up to 300; (ii) arrangement complexity, from single loops to maze-like paths; and (iii) object count. The benchmark provides 39,000 question-answer pairs, each annotated with a ground-truth reasoning trajectory and balanced across difficulty levels for controlled large-scale testing. Beyond answer-only evaluation, our framework requires models to generate structured stepwise traces of the counting process and provides an open evaluation toolkit that measures both final accuracy and step-level path consistency. Zero-shot evaluations of GPT-5, Gemini 2.5 Flash Lite, Qwen2.5-VL, InternVL3.5, and Molmo reveal sharp degradation under large-ordinal and complex-path conditions, highlighting weak generalization despite strong scores on standard multimodal tasks. By framing ordinal number understanding as a core target, OrdinalBench provides a reproducible benchmark and diagnostic framework for developing VLMs with stronger sequential reasoning. All data and code are available at this https URL
>
---
#### [new 052] ViSA-Enhanced Aerial VLN: A Visual-Spatial Reasoning Enhanced Framework for Aerial Vision-Language Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于航空视觉语言导航任务，旨在解决空间推理不足和语言歧义问题。提出ViSA框架，通过视觉提示增强VLM直接图像推理能力，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2603.08007](https://arxiv.org/pdf/2603.08007)**

> **作者:** Haoyu Tong; Xiangyu Dong; Xiaoguang Ma; Haoran Zhao; Yaoming Zhou; Chenghao Lin
>
> **备注:** 8 pages
>
> **摘要:** Existing aerial Vision-Language Navigation (VLN) methods predominantly adopt a detection-and-planning pipeline, which converts open-vocabulary detections into discrete textual scene graphs. These approaches are plagued by inadequate spatial reasoning capabilities and inherent linguistic ambiguities. To address these bottlenecks, we propose a Visual-Spatial Reasoning (ViSA) enhanced framework for aerial VLN. Specifically, a triple-phase collaborative architecture is designed to leverage structured visual prompting, enabling Vision-Language Models (VLMs) to perform direct reasoning on image planes without the need for additional training or complex intermediate representations. Comprehensive evaluations on the CityNav benchmark demonstrate that the ViSA-enhanced VLN achieves a 70.3\% improvement in success rate compared to the fully trained state-of-the-art (SOTA) method, elucidating its great potential as a backbone for aerial VLN systems.
>
---
#### [new 053] EarthBridge: A Solution for 4th Multi-modal Aerial View Image Challenge Translation Track
- **分类: cs.CV**

- **简介: 该论文属于多模态遥感图像翻译任务，解决EO、IR和SAR图像之间的跨模态转换问题。提出EarthBridge框架，结合DBIM和CUT方法，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2603.06753](https://arxiv.org/pdf/2603.06753)**

> **作者:** Zhenyuan Chen; Guanyuan Shen; Feng Zhang
>
> **备注:** tech report
>
> **摘要:** Cross-modal image-to-image translation among Electro-Optical (EO), Infrared (IR), and Synthetic Aperture Radar (SAR) sensors is essential for comprehensive multi-modal aerial-view analysis. However, translating between these modalities is notoriously difficult due to their distinct electromagnetic signatures and geometric characteristics. This paper presents \textbf{EarthBridge}, a high-fidelity translation framework developed for the 4th Multi-modal Aerial View Image Challenge -- Translation (MAVIC-T). We explore two distinct methodologies: \textbf{Diffusion Bridge Implicit Models (DBIM)}, which we generalize using non-Markovian bridge processes for high-quality deterministic sampling, and \textbf{Contrastive Unpaired Translation (CUT)}, which utilizes contrastive learning for structural consistency. Our EarthBridge framework employs a channel-concatenated UNet denoiser trained with Karras-weighted bridge scalings and a specialized "booting noise" initialization to handle the inherent ambiguity in cross-modal mappings. We evaluate these methods across all four challenge tasks (SAR$\rightarrow$EO, SAR$\rightarrow$RGB, SAR$\rightarrow$IR, RGB$\rightarrow$IR), achieving superior spatial detail and spectral accuracy. Our solution achieved a composite score of 0.38, securing the second position on the MAVIC-T leaderboard. Code is available at this https URL.
>
---
#### [new 054] Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型中的幻觉检测任务，旨在解决模型生成虚假对象的问题。通过分析模型推理过程，提出“过度思考分数”以提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.07619](https://arxiv.org/pdf/2603.07619)**

> **作者:** Abin Shoby; Ta Duc Huy; Tuan Dung Nguyen; Minh Khoi Ho; Qi Chen; Anton van den Hengel; Phi Le Nguyen; Johan W. Verjans; Vu Minh Hieu Phan
>
> **备注:** CVPR2026 Findings
>
> **摘要:** Vision Language models (VLMs) often hallucinate non-existent objects. Detecting hallucination is analogous to detecting deception: a single final statement is insufficient, one must examine the underlying reasoning process. Yet existing detectors rely mostly on final-layer signals. Attention-based methods assume hallucinated tokens exhibit low attention, while entropy-based ones use final-step uncertainty. Our analysis reveals the opposite: hallucinated objects can exhibit peaked attention due to contextual priors; and models often express high confidence because intermediate layers have already converged to an incorrect hypothesis. We show that the key to hallucination detection lies within the model's thought process, not its final output. By probing decoder layers, we uncover a previously overlooked behavior, overthinking: models repeatedly revise object hypotheses across layers before committing to an incorrect answer. Once the model latches onto a confounded hypothesis, it can propagate through subsequent layers, ultimately causing hallucination. To capture this behavior, we introduce the Overthinking Score, a metric to measure how many competing hypotheses the model entertains and how unstable these hypotheses are across layers. This score significantly improves hallucination detection: 78.9% F1 on MSCOCO and 71.58% on AMBER.
>
---
#### [new 055] Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning
- **分类: cs.CV**

- **简介: 该论文属于超分辨率任务，解决未配准高光谱图像的分辨率提升问题。通过解混与特征融合方法，提升模型的学能力和图像质量。**

- **链接: [https://arxiv.org/pdf/2603.07918](https://arxiv.org/pdf/2603.07918)**

> **作者:** Yingkai Zhang; Tao Zhang; Jing Nie; Ying Fu
>
> **摘要:** Unregistered hyperspectral image (HSI) super-resolution (SR) typically aims to enhance a low-resolution HSI using an unregistered high-resolution reference image. In this paper, we propose an unmixing-based fusion framework that decouples spatial-spectral information to simultaneously mitigate the impact of unregistered fusion and enhance the learnability of SR models. Specifically, we first utilize singular value decomposition for initial spectral unmixing, preserving the original endmembers while dedicating the subsequent network to enhancing the initial abundance map. To leverage the spatial texture of the unregistered reference, we introduce a coarse-to-fine deformable aggregation module, which first estimates a pixel-level flow and a similarity map using a coarse pyramid predictor. It further performs fine sub-pixel refinement to achieve deformable aggregation of the reference features. The aggregative features are then refined via a series of spatial-channel abundance cross-attention blocks. Furthermore, a spatial-channel modulated fusion module is presented to merge encoder-decoder features using dynamic gating weights, yielding a high-quality, high-resolution HSI. Experimental results on simulated and real datasets confirm that our proposed method achieves state-of-the-art super-resolution performance. The code will be available at this https URL.
>
---
#### [new 056] Looking Into the Water by Unsupervised Learning of the Surface Shape
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决从空中拍摄水面时因折射导致的图像失真问题。通过构建两个神经场网络，预测水面高度和图像颜色，实现无监督训练与图像重建。**

- **链接: [https://arxiv.org/pdf/2603.07614](https://arxiv.org/pdf/2603.07614)**

> **作者:** Ori Lifschitz; Tali Treibitz; Dan Rosenbaum
>
> **摘要:** We address the problem of looking into the water from the air, where we seek to remove image distortions caused by refractions at the water surface. Our approach is based on modeling the different water surface structures at various points in time, assuming the underlying image is constant. To this end, we propose a model that consists of two neural-field networks. The first network predicts the height of the water surface at each spatial position and time, and the second network predicts the image color at each position. Using both networks, we reconstruct the observed sequence of images and can therefore use unsupervised training. We show that using implicit neural representations with periodic activation functions (SIREN) leads to effective modeling of the surface height spatio-temporal signal and its derivative, as required for image reconstruction. Using both simulated and real data we show that our method outperforms the latest unsupervised image restoration approach. In addition, it provides an estimate of the water surface.
>
---
#### [new 057] FreeFly-Thinking : Aligning Chain-of-Thought Reasoning with Continuous UAV Navigation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决UAV在复杂户外环境中的导航问题。通过构建数据集并采用两阶段训练策略，提升导航的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2603.07181](https://arxiv.org/pdf/2603.07181)**

> **作者:** Jiaxu Zhou; Shaobo Wang; Zhiyuan Yang; Zhenjun Yu; Tao Li
>
> **备注:** 10 pages, 5 figures,
>
> **摘要:** Vision-Language Navigation aims to enable agents to understand natural language instructions and carry out appropriate navigation actions in real-world environments. Most work focuses on indoor settings, with little research in complex outdoor scenes. Current UAV Vision-and-Language Navigation models typically act as black boxes without explicit reasoning. We introduce FreeFly-thinking, an end-to-end VLN framework that converts the UAV agent's egocentric images and language instructions into a series of actions, inspired by environment of urban architecture proposed by OpenFly. We first construct a UAV dataset for navigation task, and then performing natural language chain of thought. We adopt a two-stage training strategy: Supervised fine-tuning and Reinforcement fine-tuning. Experiments on unseen test demonstrate a strong performance, presenting robustness and efficiency in UAV navigation issue.
>
---
#### [new 058] CAST: Modeling Visual State Transitions for Consistent Video Retrieval
- **分类: cs.CV**

- **简介: 该论文属于视频检索任务，解决现有方法忽略状态一致性的问题。提出CAST模型，通过视觉状态转换提升视频连贯性检索效果。**

- **链接: [https://arxiv.org/pdf/2603.08648](https://arxiv.org/pdf/2603.08648)**

> **作者:** Yanqing Liu; Yingcheng Liu; Fanghong Dong; Budianto Budianto; Cihang Xie; Yan Jiao
>
> **摘要:** As video content creation shifts toward long-form narratives, composing short clips into coherent storylines becomes increasingly important. However, prevailing retrieval formulations remain context-agnostic at inference time, prioritizing local semantic alignment while neglecting state and identity consistency. To address this structural limitation, we formalize the task of Consistent Video Retrieval (CVR) and introduce a diagnostic benchmark spanning YouCook2, COIN, and CrossTask. We propose CAST (Context-Aware State Transition), a lightweight, plug-and-play adapter compatible with diverse frozen vision-language embedding spaces. By predicting a state-conditioned residual update ($\Delta$) from visual history, CAST introduces an explicit inductive bias for latent state evolution. Extensive experiments show that CAST improves performance on YouCook2 and CrossTask, remains competitive on COIN, and consistently outperforms zero-shot baselines across diverse foundation backbones. Furthermore, CAST provides a useful reranking signal for black-box video generation candidates (e.g., from Veo), promoting more temporally coherent continuations.
>
---
#### [new 059] Text to Automata Diagrams: Comparing TikZ Code Generation with Direct Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于将文本转换为自动机图的任务，旨在解决学生手绘图的准确表示问题。通过对比TikZ代码生成与直接图像合成的效果，验证了人类校正对提升描述质量的重要性。**

- **链接: [https://arxiv.org/pdf/2603.07936](https://arxiv.org/pdf/2603.07936)**

> **作者:** Ethan Young; Zichun Wang; Aiden Taylor; Chance Jewell; Julian Myers; Satya Sri Rajiteswari Nimmagadda; Anthony White; Aniruddha Maiti; Ananya Jana
>
> **备注:** Accepted to ASEE North Central Section 2026
>
> **摘要:** Diagrams are widely used in teaching computer science courses. They are useful in subjects such as automata and formal languages, data structures, etc. These diagrams, often drawn by students during exams or assignments, vary in structure, layout, and correctness. This study examines whether current vision-language and large language models can process such diagrams and produce accurate textual and digital representations. In this study, scanned student-drawn diagrams are used as input. Then, textual descriptions are generated from these images using a vision-language model. The descriptions are checked and revised by human reviewers to make them accurate. Both the generated and the revised descriptions are then fed to a large language model to generate TikZ code. The resulting diagrams are compiled and then evaluated against the original scanned diagrams. We found descriptions generated directly from images using vision-language models are often incorrect and human correction can substantially improve the quality of vision language model generated descriptions. This research can help computer science education by paving the way for automated grading and feedback and creating more accessible instructional materials.
>
---
#### [new 060] Local-Global Prompt Learning via Sparse Optimal Transport
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的少样本学习任务，旨在解决局部与全局对齐中的冗余和重叠问题。提出SOT-GLP方法，通过稀疏最优传输实现局部特征的合理分配。**

- **链接: [https://arxiv.org/pdf/2603.08347](https://arxiv.org/pdf/2603.08347)**

> **作者:** Deniz Kizaroğlu; Ülku Tuncer Küçüktas; Emre Çakmakyurdu; Alptekin Temizel
>
> **备注:** 9 pages, 3 figures, 4 tables. Code available at GitHub
>
> **摘要:** Few-shot adaptation of vision-language models (VLMs) like CLIP typically relies on learning textual prompts matched to global image embeddings. Recent works extend this paradigm by incorporating local image-text alignment to capture fine-grained visual cues, yet these approaches often select local regions independently for each prompt, leading to redundant local feature usage and prompt overlap. We propose SOT-GLP, which introduces a shared sparse patch support and balanced optimal transport allocation to explicitly partition salient visual regions among class-specific local prompts while preserving global alignment. Our method learns shared global prompts and class-specific local prompts. The global branch maintains standard image-text matching for robust category-level alignment. The local branch constructs a class-conditioned sparse patch set using V-V attention and aligns it to multiple class-specific prompts via balanced entropic optimal transport, yielding a soft partition of patches that prevents prompt overlap and collapse. We evaluate our method on two complementary objectives: (i) few-shot classification accuracy on 11 standard benchmarks and (ii) out-of-distribution (OOD) detection. On the standard 11-dataset benchmark with 16-shot ViT-B/16, SOT-GLP achieves 85.1% average accuracy, outperforming prior prompt-learning methods. We identify a distinct accuracy-robustness trade-off in prompt learning: while learnable projections optimize in-distribution fit, they alter the foundational feature space. We demonstrate that a projection-free local alignment preserves the native geometry of the CLIP manifold, yielding state-of-the-art OOD detection performance (94.2% AUC) that surpasses fully adapted models. Implementation available at: this https URL
>
---
#### [new 061] DECADE: A Temporally-Consistent Unsupervised Diffusion Model for Enhanced Rb-82 Dynamic Cardiac PET Image Denoising
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像处理任务，旨在解决Rb-82动态心脏PET图像的噪声问题。提出DECADE模型，实现无监督降噪，保持时间一致性与定量准确性。**

- **链接: [https://arxiv.org/pdf/2603.07759](https://arxiv.org/pdf/2603.07759)**

> **作者:** Yinchi Zhou; Liang Guo; Huidong Xie; Yuexi Du; Ashley Wang; Menghua Xia; Tian Yu; Ramesh Fazzone-Chettiar; Christopher Weyman; Bruce Spottiswoode; Vladimir Panin; Kuangyu Shi; Edward J. Miller; Attila Feher; Albert J. Sinusas; Nicha C. Dvornek; Chi Liu
>
> **摘要:** Rb-82 dynamic cardiac PET imaging is widely used for the clinical diagnosis of coronary artery disease (CAD), but its short half-life results in high noise levels that degrade dynamic frame quality and parametric imaging. The lack of paired clean-noisy training data, rapid tracer kinetics, and frame-dependent noise variations further limit the effectiveness of existing deep learning denoising methods. We propose DECADE (A Temporally-Consistent Unsupervised Diffusion model for Enhanced Rb-82 CArdiac PET DEnoising), an unsupervised diffusion framework that generalizes across early- to late-phase dynamic frames. DECADE incorporates temporal consistency during both training and iterative sampling, using noisy frames as guidance to preserve quantitative accuracy. The method was trained and evaluated on datasets acquired from Siemens Vision 450 and Siemens Biograph Vision Quadra scanners. On the Vision 450 dataset, DECADE consistently produced high-quality dynamic and parametric images with reduced noise while preserving myocardial blood flow (MBF) and myocardial flow reserve (MFR). On the Quadra dataset, using 15%-count images as input and full-count images as reference, DECADE outperformed UNet-based and other diffusion models in image quality and K1/MBF quantification. The proposed framework enables effective unsupervised denoising of Rb-82 dynamic cardiac PET without paired training data, supporting clearer visualization while maintaining quantitative integrity.
>
---
#### [new 062] Disentangled Textual Priors for Diffusion-based Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决扩散模型中语义先验整合不足的问题。提出DTPSR框架，通过分离空间层次和频率语义先验，提升生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2603.07430](https://arxiv.org/pdf/2603.07430)**

> **作者:** Lei Jiang; Xin Liu; Xinze Tong; Zhiliang Li; Jie Liu; Jie Tang; Gangshan Wu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Image Super-Resolution (SR) aims to reconstruct high-resolution images from degraded low-resolution inputs. While diffusion-based SR methods offer powerful generative capabilities, their performance heavily depends on how semantic priors are structured and integrated into the generation process. Existing approaches often rely on entangled or coarse-grained priors that mix global layout with local details, or conflate structural and textural cues, thereby limiting semantic controllability and interpretability. In this work, we propose DTPSR, a novel diffusion-based SR framework that introduces disentangled textual priors along two complementary dimensions: spatial hierarchy (global vs. local) and frequency semantics (low- vs. high-frequency). By explicitly separating these priors, DTPSR enables the model to simultaneously capture scene-level structure and object-specific details with frequency-aware semantic guidance. The corresponding embeddings are injected via specialized cross-attention modules, forming a progressive generation pipeline that reflects the semantic granularity of visual content, from global layout to fine-grained textures. To support this paradigm, we construct DisText-SR, a large-scale dataset containing approximately 95,000 image-text pairs with carefully disentangled global, low-frequency, and high-frequency descriptions. To further enhance controllability and consistency, we adopt a multi-branch classifier-free guidance strategy with frequency-aware negative prompts to suppress hallucinations and semantic drift. Extensive experiments on synthetic and real-world benchmarks show that DTPSR achieves high perceptual quality, competitive fidelity, and strong generalization across diverse degradation scenarios.
>
---
#### [new 063] ImageEdit-R1: Boosting Multi-Agent Image Editing via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像编辑任务，旨在解决复杂用户指令下编辑效果不佳的问题。提出ImageEdit-R1框架，利用强化学习协调多代理实现高效、精准的图像编辑。**

- **链接: [https://arxiv.org/pdf/2603.08059](https://arxiv.org/pdf/2603.08059)**

> **作者:** Yiran Zhao; Yaoqi Ye; Xiang Liu; Michael Qizhe Shieh; Trung Bui
>
> **摘要:** With the rapid advancement of commercial multi-modal models, image editing has garnered significant attention due to its widespread applicability in daily life. Despite impressive progress, existing image editing systems, particularly closed-source or proprietary models, often struggle with complex, indirect, or multi-step user instructions. These limitations hinder their ability to perform nuanced, context-aware edits that align with human intent. In this work, we propose ImageEdit-R1, a multi-agent framework for intelligent image editing that leverages reinforcement learning to coordinate high-level decision-making across a set of specialized, pretrained vision-language and generative agents. Each agent is responsible for distinct capabilities--such as understanding user intent, identifying regions of interest, selecting appropriate editing actions, and synthesizing visual content--while reinforcement learning governs their collaboration to ensure coherent and goal-directed behavior. Unlike existing approaches that rely on monolithic models or hand-crafted pipelines, our method treats image editing as a sequential decision-making problem, enabling dynamic and context-aware editing strategies. Experimental results demonstrate that ImageEdit-R1 consistently outperforms both individual closed-source diffusion models and alternative multi-agent framework baselines across multiple image editing datasets.
>
---
#### [new 064] Parameterized Brushstroke Style Transfer
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于风格迁移任务，旨在解决像素方法无法自然表现艺术笔触的问题，提出在笔触域而非RGB域进行风格迁移，提升视觉效果。**

- **链接: [https://arxiv.org/pdf/2603.07776](https://arxiv.org/pdf/2603.07776)**

> **作者:** Uma Meleti; Siyu Huang
>
> **摘要:** Computer Vision-based Style Transfer techniques have been used for many years to represent artistic style. However, most contemporary methods have been restricted to the pixel domain; in other words, the style transfer approach has been modifying the image pixels to incorporate artistic style. However, real artistic work is made of brush strokes with different colors on a canvas. Pixel-based approaches are unnatural for representing these images. Hence, this paper discusses a style transfer method that represents the image in the brush stroke domain instead of the RGB domain, which has better visual improvement over pixel-based methods.
>
---
#### [new 065] MWM: Mobile World Models for Action-Conditioned Consistent Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决世界模型在多步预测中的动作一致性问题。提出MWM框架，结合结构预训练和动作一致性训练，提升预测一致性与推理效率。**

- **链接: [https://arxiv.org/pdf/2603.07799](https://arxiv.org/pdf/2603.07799)**

> **作者:** Han Yan; Zishang Xiang; Zeyu Zhang; Hao Tang
>
> **摘要:** World models enable planning in imagined future predicted space, offering a promising framework for embodied navigation. However, existing navigation world models often lack action-conditioned consistency, so visually plausible predictions can still drift under multi-step rollout and degrade planning. Moreover, efficient deployment requires few-step diffusion inference, but existing distillation methods do not explicitly preserve rollout consistency, creating a training-inference mismatch. To address these challenges, we propose MWM, a mobile world model for planning-based image-goal navigation. Specifically, we introduce a two-stage training framework that combines structure pretraining with Action-Conditioned Consistency (ACC) post-training to improve action-conditioned rollout consistency. We further introduce Inference-Consistent State Distillation (ICSD) for few-step diffusion distillation with improved rollout consistency. Our experiments on benchmark and real-world tasks demonstrate consistent gains in visual fidelity, trajectory accuracy, planning success, and inference efficiency. Code: this https URL. Website: this https URL.
>
---
#### [new 066] HERO: Hierarchical Embedding-Refinement for Open-Vocabulary Temporal Sentence Grounding in Videos
- **分类: cs.CV**

- **简介: 该论文属于视频时序句子定位任务，解决开放词汇下模型泛化能力不足的问题。提出HERO框架，通过层次化语义嵌入和跨模态优化，提升视频与语言的对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.06732](https://arxiv.org/pdf/2603.06732)**

> **作者:** Tingting Han; Xinsong Tao; Yufei Yin; Min Tan; Sicheng Zhao; Zhou Yu
>
> **摘要:** Temporal Sentence Grounding in Videos (TSGV) aims to temporally localize segments of a video that correspond to a given natural language query. Despite recent progress, most existing TSGV approaches operate under closed-vocabulary settings, limiting their ability to generalize to real-world queries involving novel or diverse linguistic expressions. To bridge this critical gap, we introduce the Open-Vocabulary TSGV (OV-TSGV) task and construct the first dedicated benchmarks--Charades-OV and ActivityNet-OV--that simulate realistic vocabulary shifts and paraphrastic variations. These benchmarks facilitate systematic evaluation of model generalization beyond seen training concepts. To tackle OV-TSGV, we propose HERO(Hierarchical Embedding-Refinement for Open-Vocabulary grounding), a unified framework that leverages hierarchical linguistic embeddings and performs parallel cross-modal refinement. HERO jointly models multi-level semantics and enhances video-language alignment via semantic-guided visual filtering and contrastive masked text refinement. Extensive experiments on both standard and open vocabulary benchmarks demonstrate that HERO consistently surpasses state-of-the-art methods, particularly under open-vocabulary scenarios, validating its strong generalization capability and underscoring the significance of OV-TSGV as a new research direction.
>
---
#### [new 067] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval
- **分类: cs.CV**

- **简介: 该论文属于视频-文本检索任务，旨在解决视频中语音内容表示不足和视觉音频融合效果差的问题。提出SAVE方法，引入语音分支和早期对齐机制，提升检索性能。**

- **链接: [https://arxiv.org/pdf/2603.08224](https://arxiv.org/pdf/2603.08224)**

> **作者:** Ruixiang Zhao; Zhihao Xu; Bangxiang Lan; Zijie Xin; Jingyu Liu; Xirong Li
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** For video-text retrieval, the use of CLIP has been a de facto choice. Since CLIP provides only image and text encoders, this consensus has led to a biased paradigm that entirely ignores the sound track of videos. While several attempts have been made to reintroduce audio -- typically by incorporating an audio encoder and fusing its output with visual features -- these methods face two challenges: ineffective representation of speech content and suboptimal vision-audio fusion. To address these issues jointly, we propose SAVE, a Speech Aware Video rEpresentation learning method. SAVE improves upon AVIGATE, a SOTA audiovisual method, with a dedicated speech branch for more effective speech embedding. Furthermore, we introduce soft-ALBEF for early vision-audio alignment that facilitates fusion. Extensive experiments on five benchmarks show that SAVE compares favorably against the SOTA, outperforming AVIGATE by +4.1% on MSRVTT-9k, +1.9% on MSRVTT-7k, +2.5% on VATEX, +9.8% on Charades, and +2.1% on LSMDC, in light of the SumR metric.
>
---
#### [new 068] Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于模型安全任务，研究 pruning-based unlearning 的概念恢复风险。工作包括发现权重剪枝位置泄露信息，并提出数据无关的恢复方法，探讨防御策略。**

- **链接: [https://arxiv.org/pdf/2603.06640](https://arxiv.org/pdf/2603.06640)**

> **作者:** Ci Zhang; Zhaojun Ding; Chence Yang; Jun Liu; Xiaoming Zhai; Shaoyi Huang; Beiwen Li; Xiaolong Ma; Jin Lu; Geng Yuan
>
> **备注:** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Pruning-based unlearning has recently emerged as a fast, training-free, and data-independent approach to remove undesired concepts from diffusion models. It promises high efficiency and robustness, offering an attractive alternative to traditional fine-tuning or editing-based unlearning. However, in this paper we uncover a hidden danger behind this promising paradigm. We find that the locations of pruned weights, typically set to zero during unlearning, can act as side-channel signals that leak critical information about the erased concepts. To verify this vulnerability, we design a novel attack framework capable of reviving erased concepts from pruned diffusion models in a fully data-free and training-free manner. Our experiments confirm that pruning-based unlearning is not inherently secure, as erased concepts can be effectively revived without any additional data or retraining. Extensive experiments on diffusion-based unlearning based on concept related weights lead to the conclusion: once the critical concept-related weights in diffusion models are identified, our method can effectively recover the original concept regardless of how the weights are manipulated. Finally, we explore potential defense strategies and advocate safer pruning mechanisms that conceal pruning locations while preserving unlearning effectiveness, providing practical insights for designing more secure pruning-based unlearning frameworks.
>
---
#### [new 069] Image Generation Models: A Technical History
- **分类: cs.CV; cs.AI; cs.CL; cs.GR**

- **简介: 该论文属于图像生成任务，旨在系统梳理各类生成模型，解决模型碎片化问题，总结其技术原理、优化方法及应用挑战。**

- **链接: [https://arxiv.org/pdf/2603.07455](https://arxiv.org/pdf/2603.07455)**

> **作者:** Rouzbeh Shirvani
>
> **摘要:** Image generation has advanced rapidly over the past decade, yet the literature seems fragmented across different models and application domains. This paper aims to offer a comprehensive survey of breakthrough image generation models, including variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, autoregressive and transformer-based generators, and diffusion-based methods. We provide a detailed technical walkthrough of each model type, including their underlying objectives, architectural building blocks, and algorithmic training steps. For each model type, we present the optimization techniques as well as common failure modes and limitations. We also go over recent developments in video generation and present the research works that made it possible to go from still frames to high quality videos. Lastly, we cover the growing importance of robustness and responsible deployment of these models, including deepfake risks, detection, artifacts, and watermarking.
>
---
#### [new 070] PromptGate Client Adaptive Vision Language Gating for Open Set Federated Active Learning
- **分类: cs.CV**

- **简介: 该论文属于医疗AI领域，解决联邦学习中开放集样本干扰问题。提出PromptGate框架，通过动态视觉语言门控提升标注效率与数据纯度。**

- **链接: [https://arxiv.org/pdf/2603.07163](https://arxiv.org/pdf/2603.07163)**

> **作者:** Adea Nesturi; David Dueñas Gaviria; Jiajun Zeng; Shadi Albarqouni
>
> **备注:** 3 Figures, 2 Tables, 10 pages
>
> **摘要:** Deploying medical AI across resource-constrained institutions demands data-efficient learning pipelines that respect patient privacy. Federated Learning (FL) enables collaborative medical AI without centralising data, yet real-world clinical pools are inherently open-set, containing out-of-distribution (OOD) noise such as imaging artifacts and wrong modalities. Standard Active Learning (AL) query strategies mistake this noise for informative samples, wasting scarce annotation budgets. We propose PromptGate, a dynamic VLM-gated framework for Open-Set Federated AL that purifies unlabeled pools before querying. PromptGate introduces a federated Class-Specific Context Optimization: lightweight, learnable prompt vectors that adapt a frozen BiomedCLIP backbone to local clinical domains and aggregate globally via FedAvg -- without sharing patient data. As new annotations arrive, prompts progressively sharpen the ID/OOD boundary, turning the VLM into a dynamic gatekeeper that is strategy-agnostic: a plug-and-play pre-selection module enhancing any downstream AL strategy. Experiments on distributed dermatology and breast imaging benchmarks show that while static VLM prompting degrades to 50% ID purity, PromptGate maintains $>$95% purity with 98% OOD recall.
>
---
#### [new 071] PresentBench: A Fine-Grained Rubric-Based Benchmark for Slide Generation
- **分类: cs.CV**

- **简介: 该论文属于幻灯片生成任务，旨在解决现有评估方法粗略、缺乏细粒度标准的问题。提出PresentBench基准，包含238个实例和54.1个检查项，实现更精确的评估。**

- **链接: [https://arxiv.org/pdf/2603.07244](https://arxiv.org/pdf/2603.07244)**

> **作者:** Xin-Sheng Chen; Jiayu Zhu; Pei-lin Li; Hanzheng Wang; Shuojin Yang; Meng-Hao Guo
>
> **备注:** 27 pages, 9 figures
>
> **摘要:** Slides serve as a critical medium for conveying information in presentation-oriented scenarios such as academia, education, and business. Despite their importance, creating high-quality slide decks remains time-consuming and cognitively demanding. Recent advances in generative models, such as Nano Banana Pro, have made automated slide generation increasingly feasible. However, existing evaluations of slide generation are often coarse-grained and rely on holistic judgments, making it difficult to accurately assess model capabilities or track meaningful advances in the field. In practice, the lack of fine-grained, verifiable evaluation criteria poses a critical bottleneck for both research and real-world deployment. In this paper, we propose PresentBench, a fine-grained, rubric-based benchmark for evaluating automated real-world slide generation. It contains 238 evaluation instances, each supplemented with background materials required for slide creation. Moreover, we manually design an average of 54.1 checklist items per instance, each formulated as a binary question, to enable fine-grained, instance-specific evaluation of the generated slide decks. Extensive experiments show that PresentBench provides more reliable evaluation results than existing methods, and exhibits significantly stronger alignment with human preferences. Furthermore, our benchmark reveals that NotebookLM significantly outperforms other slide generation methods, highlighting substantial recent progress in this domain.
>
---
#### [new 072] VINO: Video-driven Invariance for Non-contextual Objects via Structural Prior Guided De-contextualization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VINO框架，解决视频中对象与背景混淆的问题，通过结构先验引导的去上下文化，提升图像编码器的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07222](https://arxiv.org/pdf/2603.07222)**

> **作者:** Seul-Ki Yeom; Marcel Simon; Eunbin Lee; Tae-Ho Kim
>
> **备注:** 18 pages, 2 Tables, 3 Figures
>
> **摘要:** Self-supervised learning (SSL) has made rapid progress, yet learned features often over-rely on contextual shortcuts-background textures and co-occurrence statistics. While video provides rich temporal variation, dense in-the-wild streams with strong ego-motion create a co-occurrence trap: foreground objects and background context move coherently, encouraging representations to collapse into scene encoders. To address this, we propose VINO (Video-driven Invariance for Non-Contextual Objects), a teacher-student framework that learns robust image encoders from dense video by imposing a structural information bottleneck. Using a class-agnostic structural prior solely to generate views-not as semantic pseudo-labels-VINO forms an asymmetric distillation problem. The teacher predicts from a foreground-union view with the background suppressed, while the student observes object-conditioned scene views that retain surrounding context but remove competing instances. Matching these targets via masked distillation makes background cues unreliable, pushing the representation toward object-centric invariances. We further enforce temporal object permanence via teacher-anchored cross-time distillation over track-matched objects, and stabilize part-to-whole consistency with mask-guided local views. Through attention visualization and unsupervised object discovery on PASCAL VOC, we demonstrate that VINO effectively disentangles foreground from background. Pretrained on the dense Walking Tours Venice video, VINO achieves 34.8 CorLoc, yielding highly focused, shape-biased representations that substantially outperform prior dense-video and motion-guided SSL baselines.
>
---
#### [new 073] Extracting and analyzing 3D histomorphometric features related to perineural and lymphovascular invasion in prostate cancer
- **分类: cs.CV**

- **简介: 该论文属于癌症诊断任务，旨在解决2D病理分析的局限性。通过3D影像分析，提取与神经和血管浸润相关的特征，提升前列腺癌预后评估。**

- **链接: [https://arxiv.org/pdf/2603.06936](https://arxiv.org/pdf/2603.06936)**

> **作者:** Sarah S.L. Chow; Rui Wang; Robert B. Serafin; Yujie Zhao; Elena Baraznenok; Xavier Farré; Jennifer Salguero-Lopez; Gan Gao; Huai-Ching Hsieh; Lawrence D. True; Priti Lal; Anant Madabhushi; Jonathan T.C. Liu
>
> **摘要:** Diagnostic grading of prostate cancer (PCa) relies on the examination of 2D histology sections. However, the limited sampling of specimens afforded by 2D histopathology, and ambiguities when viewing 2D cross-sections, can lead to suboptimal treatment decisions. Recent studies have shown that 3D histomorphometric analysis of glands and nuclei can improve PCa risk assessment compared to analogous 2D features. Here, we expand on these efforts by developing an analytical pipeline to extract 3D features related to perineural invasion (PNI) and lymphovascular invasion (LVI), which correlate with poor prognosis for a variety of cancers. A 3D segmentation model (nnU-Net) was trained to segment nerves and vessels in 3D datasets of archived prostatectomy specimens that were optically cleared, labeled with a fluorescent analog of H&E, and imaged with open-top light-sheet (OTLS) microscopy. PNI- and LVI-related features, including metrics describing cancer-nerve and cancer-vessel proximity, were then extracted based on the 3D nerve/vessel segmentation masks in conjunction with 3D masks of cancer-enriched regions. As a preliminary exploration of the prognostic value of these features, we trained a supervised machine learning classifier to predict 5-year biochemical recurrence (BCR) outcomes, finding that 3D PNI-related features are moderately prognostic and outperform 2D PNI-related features (AUC = 0.71 vs. 0.52). Source code is available at this https URL.
>
---
#### [new 074] AMR-CCR: Anchored Modular Retrieval for Continual Chinese Character Recognition
- **分类: cs.CV**

- **简介: 该论文属于汉字识别任务，解决持续增量类别的识别问题。提出AMR-CCR框架，通过嵌入字典匹配实现新类别的灵活扩展。**

- **链接: [https://arxiv.org/pdf/2603.07497](https://arxiv.org/pdf/2603.07497)**

> **作者:** Yuchuan Wu; Yinglian Zhu; Haiyang Yu; Ke Niu; Bin Li; Xiangyang Xue
>
> **摘要:** Ancient Chinese character recognition is a core capability for cultural heritage digitization, yet real-world workflows are inherently non-stationary: newly excavated materials are continuously onboarded, bringing new classes in different scripts, and expanding the class space over time. We formalize this process as Continual Chinese Character Recognition (Continual CCR), a script-staged, class-incremental setting that couples two challenges: (i) scalable learning under continual class growth with subtle inter-class differences and scarce incremental data, and (ii) pronounced intra-class diversity caused by writing-style variations across writers and carrier conditions. To overcome the limitations of conventional closed-set classification, we propose AMR-CCR, an anchored modular retrieval framework that performs recognition via embedding-based dictionary matching in a shared multimodal space, allowing new classes to be added by simply extending the dictionary. AMR-CCR further introduces a lightweight script-conditioned injection module (SIA+SAR) to calibrate newly onboarded scripts while preserving cross-stage embedding compatibility, and an image-derived multi-prototype dictionary that clusters within-class embeddings to better cover diverse style modes. To support systematic evaluation, we build EvoCON, a six-stage benchmark for continual script onboarding, covering six scripts (OBC, BI, SS, SAC, WSC, CS), augmented with meaning/shape descriptions and an explicit zero-shot split for unseen characters without image exemplars.
>
---
#### [new 075] GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于工业视觉异常检测任务，旨在解决传统方法依赖预处理且泛化能力差的问题。提出GRD-Net架构，结合生成与判别模块，提升缺陷定位精度。**

- **链接: [https://arxiv.org/pdf/2603.07566](https://arxiv.org/pdf/2603.07566)**

> **作者:** Niccolò Ferrari; Michele Fraccaroli; Evelina Lamma
>
> **备注:** Peer-reviewed journal version published. 18 pages, 12 figures, 7 tables
>
> **摘要:** Anomaly detection is nowadays increasingly used in industrial applications and processes. One of the main fields of the appliance is the visual inspection for surface anomaly detection, which aims to spot regions that deviate from regularity and consequently identify abnormal products. Defect localization is a key task, that usually is achieved using a basic comparison between generated image and the original one, implementing some blob-analysis or image-editing algorithms, in the post-processing step, which is very biased towards the source dataset, and they are unable to generalize. Furthermore, in industrial applications, the totality of the image is not always interesting but could be one or some regions of interest (ROIs), where only in those areas there are relevant anomalies to be spotted. For these reasons, we propose a new architecture composed by two blocks. The first block is a Generative Adversarial Network (GAN), based on a residual autoencoder (ResAE), to perform reconstruction and denoising processes, while the second block produces image segmentation, spotting defects. This method learns from a dataset composed of good products and generated synthetic defects. The discriminative network is trained using a ROI for each image contained in the training dataset. The network will learn in which area anomalies are relevant. This approach guarantees the reduction of using pre-processing algorithms, formerly developed with blob-analysis and image-editing procedures. To test our model we used challenging MVTec anomaly detection datasets and an industrial large dataset of pharmaceutical BFS strips of vials. This set constitutes a more realistic use case of the aforementioned network.
>
---
#### [new 076] On the Feasibility and Opportunity of Autoregressive 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决传统方法依赖人工设计组件的问题。提出AutoReg3D，通过序列生成实现自回归检测，提升灵活性与性能。**

- **链接: [https://arxiv.org/pdf/2603.07985](https://arxiv.org/pdf/2603.07985)**

> **作者:** Zanming Huang; Jinsu Yoo; Sooyoung Jeon; Zhenzhen Liu; Mark Campbell; Kilian Q Weinberger; Bharath Hariharan; Wei-Lun Chao; Katie Z Luo
>
> **备注:** CVPR 2026 Findings Project Page: this https URL
>
> **摘要:** LiDAR-based 3D object detectors typically rely on proposal heads with hand-crafted components like anchor assignment and non-maximum suppression (NMS), complicating training and limiting extensibility. We present AutoReg3D, an autoregressive 3D detector that casts detection as sequence generation. Given point-cloud features, AutoReg3D emits objects in a range-causal (near-to-far) order and encodes each object as a short, discrete-token sequence consisting of its center, size, orientation, velocity, and class. This near-to-far ordering mirrors LiDAR geometry--near objects occlude far ones but not vice versa--enabling straightforward teacher forcing during training and autoregressive decoding at test time. AutoReg3D is compatible across diverse point-cloud or backbones and attains competitive nuScenes performance without anchors or NMS. Beyond parity, the sequential formulation unlocks language-model advances for 3D perception, including GRPO-style reinforcement learning for task-aligned objectives. These results position autoregressive decoding as a viable, flexible alternative for LiDAR-based detection and open a path to importing modern sequence-modeling tools into 3D perception.
>
---
#### [new 077] Retinex Meets Language: A Physics-Semantics-Guided Underwater Image Enhancement Network
- **分类: cs.CV**

- **简介: 该论文属于水下图像增强任务，旨在解决颜色失真、对比度低等问题。提出PSG-UIENet网络，结合物理模型与语言指导，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2603.07076](https://arxiv.org/pdf/2603.07076)**

> **作者:** Shixuan Xu; Yabo Liu; Junyu Dong; Xinghui Dong
>
> **摘要:** Underwater images often suffer from severe degradation caused by light absorption and scattering, leading to color distortion, low contrast and reduced visibility. Existing Underwater Image Enhancement (UIE) methods can be divided into two categories, i.e., prior-based and learning-based methods. The former rely on rigid physical assumptions that limit the adaptability, while the latter often face data scarcity and weak generalization. To address these issues, we propose a Physics-Semantics-Guided Underwater Image Enhancement Network (PSG-UIENet), which couples the Retinex-grounded illumination correction with the language-informed guidance. This network comprises a Prior-Free Illumination Estimator, a Cross-Modal Text Aligner and a Semantics-Guided Image Restorer. In particular, the restorer leverages the textual descriptions generated by the Contrastive Language-Image Pre-training (CLIP) model to inject high-level semantics for perceptually meaningful guidance. Since multimodal UIE data sets are not publicly available, we also construct a large-scale image-text UIE data set, namely, LUIQD-TD, which contains 6,418 image-reference-text triplets. To explicitly measure and optimize semantic consistency between textual descriptions and images, we further design an Image-Text Semantic Similarity (ITSS) loss function. To our knowledge, this study makes the first effort to introduce both textual guidance and the multimodal data set into UIE tasks. Extensive experiments on our data set and four publicly available data sets demonstrate that the proposed PSG-UIENet achieves superior or comparable performance against fifteen state-of-the-art methods.
>
---
#### [new 078] VSDiffusion: Taming Ill-Posed Shadow Generation via Visibility-Constrained Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，旨在解决复杂场景中生成几何一致阴影的问题。提出VSDiffusion框架，通过可见性约束和扩散机制生成准确阴影。**

- **链接: [https://arxiv.org/pdf/2603.08020](https://arxiv.org/pdf/2603.08020)**

> **作者:** Jing Li; Jing Zhang
>
> **备注:** 12 pages,8 figures
>
> **摘要:** Generating realistic cast shadows for inserted foreground objects is a crucial yet challenging problem in image composition, where maintaining geometric consistency of shadow and object in complex scenes remains difficult due to the ill-posed nature of shadow formation. To address this issue, we propose VSDiffusion, a visibility-constrained two-stage framework designed to narrow the solution space by incorporating visibility priors. In Stage I, we predict a coarse shadow mask to localize plausible shadow generated regions. And in Stage II, conditional diffusion is performed guided by lighting and depth cues estimated from the composite to generate accurate shadows. In VSDiffusion, we inject visibility priors through two complementary pathways. First, a visibility control branch with shadow-gated cross attention that provides multi-scale structural guidance. Then, a learned soft prior map that reweights training loss in error-prone regions to enhance geometric correction. Additionally, we also introduce high-frequency guided enhancement module to sharpen boundaries and improve texture interaction with the background. Experiments on widely used public DESOBAv2 dataset demonstrated that our proposed VSDiffusion can generate accurate shadow, and establishes new SOTA results across most evaluation metrics.
>
---
#### [new 079] Talking Together: Synthesizing Co-Located 3D Conversations from Audio
- **分类: cs.CV**

- **简介: 该论文属于3D人脸动画生成任务，解决从音频合成真实互动对话的问题。通过建模空间关系和眼神交流，生成可控的双人动画。**

- **链接: [https://arxiv.org/pdf/2603.08674](https://arxiv.org/pdf/2603.08674)**

> **作者:** Mengyi Shan; Shouchieh Chang; Ziqian Bai; Shichen Liu; Yinda Zhang; Luchuan Song; Rohit Pandey; Sean Fanello; Zeng Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We tackle the challenging task of generating complete 3D facial animations for two interacting, co-located participants from a mixed audio stream. While existing methods often produce disembodied "talking heads" akin to a video conference call, our work is the first to explicitly model the dynamic 3D spatial relationship -- including relative position, orientation, and mutual gaze -- that is crucial for realistic in-person dialogues. Our system synthesizes the full performance of both individuals, including precise lip-sync, and uniquely allows their relative head poses to be controlled via textual descriptions. To achieve this, we propose a dual-stream architecture where each stream is responsible for one participant's output. We employ speaker's role embeddings and inter-speaker cross-attention mechanisms designed to disentangle the mixed audio and model the interaction. Furthermore, we introduce a novel eye gaze loss to promote natural, mutual eye contact. To power our data-hungry approach, we introduce a novel pipeline to curate a large-scale conversational dataset consisting of over 2 million dyadic pairs from in-the-wild videos. Our method generates fluid, controllable, and spatially aware dyadic animations suitable for immersive applications in VR and telepresence, significantly outperforming existing baselines in perceived realism and interaction coherence.
>
---
#### [new 080] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型推理速度慢的问题。通过分析权重方向变化，提出WaDi框架，实现高效的一步生成。**

- **链接: [https://arxiv.org/pdf/2603.08258](https://arxiv.org/pdf/2603.08258)**

> **作者:** Lei Wang; Yang Cheng; Senmao Li; Ge Wu; Yaxing Wang; Jian Yang
>
> **备注:** Accepted to CVPR 2026;Code:this https URL
>
> **摘要:** Despite the impressive performance of diffusion models such as Stable Diffusion (SD) in image generation, their slow inference limits practical deployment. Recent works accelerate inference by distilling multi-step diffusion into one-step generators. To better understand the distillation mechanism, we analyze U-Net/DiT weight changes between one-step students and their multi-step teacher counterparts. Our analysis reveals that changes in weight direction significantly exceed those in weight norm, highlighting it as the key factor during distillation. Motivated by this insight, we propose the Low-rank Rotation of weight Direction (LoRaD), a parameter-efficient adapter tailored to one-step diffusion distillation. LoRaD is designed to model these structured directional changes using learnable low-rank rotation matrices. We further integrate LoRaD into Variational Score Distillation (VSD), resulting in Weight Direction-aware Distillation (WaDi)-a novel one-step distillation framework. WaDi achieves state-of-the-art FID scores on COCO 2014 and COCO 2017 while using only approximately 10% of the trainable parameters of the U-Net/DiT. Furthermore, the distilled one-step model demonstrates strong versatility and scalability, generalizing well to various downstream tasks such as controllable generation, relation inversion, and high-resolution synthesis.
>
---
#### [new 081] Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频生成任务，解决复杂人体动作生成难题。通过文本到骨骼的级联框架，生成高质量动作视频，提升运动控制精度与视频质量。**

- **链接: [https://arxiv.org/pdf/2603.08028](https://arxiv.org/pdf/2603.08028)**

> **作者:** Ashkan Taghipour; Morteza Ghahremani; Zinuo Li; Hamid Laga; Farid Boussaid; Mohammed Bennamoun
>
> **摘要:** Generating videos of complex human motions such as flips, cartwheels, and martial arts remains challenging for current video diffusion models. Text-only conditioning is temporally ambiguous for fine-grained motion control, while explicit pose-based controls, though effective, require users to provide complete skeleton sequences that are costly to produce for long and dynamic actions. We propose a two-stage cascaded framework that addresses both limitations. First, an autoregressive text-to-skeleton model generates 2D pose sequences from natural language descriptions by predicting each joint conditioned on previously generated poses. This design captures long-range temporal dependencies and inter-joint coordination required for complex motions. Second, a pose-conditioned video diffusion model synthesizes videos from a reference image and the generated skeleton sequence. It employs DINO-ALF (Adaptive Layer Fusion), a multi-level reference encoder that preserves appearance and clothing details under large pose changes and self-occlusions. To address the lack of publicly available datasets for complex human motion video generation, we introduce a Blender-based synthetic dataset containing 2,000 videos with diverse characters performing acrobatic and stunt-like motions. The dataset provides full control over appearance, motion, and environment. It fills an important gap because existing benchmarks significantly under-represent acrobatic motions while web-collected datasets raise copyright and privacy concerns. Experiments on our synthetic dataset and the Motion-X Fitness benchmark show that our text-to-skeleton model outperforms prior methods on FID, R-precision, and motion diversity. Our pose-to-video model also achieves the best results among all compared methods on VBench metrics for temporal consistency, motion smoothness, and subject preservation.
>
---
#### [new 082] HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决长视频生成中时间连续性与质量退化问题。提出HiAR框架，通过分层去噪和并行推理提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.08703](https://arxiv.org/pdf/2603.08703)**

> **作者:** Kai Zou; Dian Zheng; Hongbo Liu; Tiankai Hang; Bin Liu; Nenghai Yu
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Autoregressive (AR) diffusion offers a promising framework for generating videos of theoretically infinite length. However, a major challenge is maintaining temporal continuity while preventing the progressive quality degradation caused by error accumulation. To ensure continuity, existing methods typically condition on highly denoised contexts; yet, this practice propagates prediction errors with high certainty, thereby exacerbating degradation. In this paper, we argue that a highly clean context is unnecessary. Drawing inspiration from bidirectional diffusion models, which denoise frames at a shared noise level while maintaining coherence, we propose that conditioning on context at the same noise level as the current block provides sufficient signal for temporal consistency while effectively mitigating error propagation. Building on this insight, we propose HiAR, a hierarchical denoising framework that reverses the conventional generation order: instead of completing each block sequentially, it performs causal generation across all blocks at every denoising step, so that each block is always conditioned on context at the same noise level. This hierarchy naturally admits pipelined parallel inference, yielding a 1.8 wall-clock speedup in our 4-step setting. We further observe that self-rollout distillation under this paradigm amplifies a low-motion shortcut inherent to the mode-seeking reverse-KL objective. To counteract this, we introduce a forward-KL regulariser in bidirectional-attention mode, which preserves motion diversity for causal inference without interfering with the distillation loss. On VBench (20s generation), HiAR achieves the best overall score and the lowest temporal drift among all compared methods.
>
---
#### [new 083] X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于深度伪造检测任务，旨在解决真实与合成视频识别问题。提出X-AVDT模型，利用音频-视觉交叉注意力机制提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08483](https://arxiv.org/pdf/2603.08483)**

> **作者:** Youngseo Kim; Kwan Yun; Seokhyeon Hong; Sihun Cha; Colette Suhjung Koo; Junyong Noh
>
> **摘要:** The surge of highly realistic synthetic videos produced by contemporary generative systems has significantly increased the risk of malicious use, challenging both humans and existing detectors. Against this backdrop, we take a generator-side view and observe that internal cross-attention mechanisms in these models encode fine-grained speech-motion alignment, offering useful correspondence cues for forgery detection. Building on this insight, we propose X-AVDT, a robust and generalizable deepfake detector that probes generator-internal audio-visual signals accessed via DDIM inversion to expose these cues. X-AVDT extracts two complementary signals: (i) a video composite capturing inversion-induced discrepancies, and (ii) an audio-visual cross-attention feature reflecting modality alignment enforced during generation. To enable faithful cross-generator evaluation, we further introduce MMDF, a new multimodal deepfake dataset spanning diverse manipulation types and rapidly evolving synthesis paradigms, including GANs, diffusion, and flow-matching. Extensive experiments demonstrate that X-AVDT achieves leading performance on MMDF and generalizes strongly to external benchmarks and unseen generators, outperforming existing methods with accuracy improved by 13.1%. Our findings highlight the importance of leveraging internal audio-visual consistency cues for robustness to future generators in deepfake detection.
>
---
#### [new 084] Multi-Modal Decouple and Recouple Network for Robust 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决数据损坏下多模态融合性能下降的问题。通过解耦和重新融合特征，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07486](https://arxiv.org/pdf/2603.07486)**

> **作者:** Rui Ding; Zhaonian Kuang; Yuzhe Ji; Meng Yang; Xinhu Zheng; Gang Hua
>
> **摘要:** Multi-modal 3D object detection with bird's eye view (BEV) has achieved desired advances on benchmarks. Nonetheless, the accuracy may drop significantly in the real world due to data corruption such as sensor configurations for LiDAR and scene conditions for camera. One design bottleneck of previous models resides in the tightly coupling of multi-modal BEV features during fusion, which may degrade the overall system performance if one modality or both is corrupted. To mitigate, we propose a Multi-Modal Decouple and Recouple Network for robust 3D object detection under data corruption. Different modalities commonly share some high-level invariant features. We observe that these invariant features across modalities do not always fail simultaneously, because different types of data corruption affect each modality in distinct this http URL invariant features can be recovered across modalities for robust fusion under data this http URL this end, we explicitly decouple Camera/LiDAR BEV features into modality-invariant and modality-specific parts. It allows invariant features to compensate each other while mitigates the negative impact of a corrupted modality on the this http URL then recouple these features into three experts to handle different types of data corruption, respectively, i.e., LiDAR, camera, and this http URL each expert, we use modality-invariant features as robust information, while modality-specific features serve as a this http URL, we adaptively fuse the three experts to exact robust features for 3D object detection. For validation, we collect a benchmark with a large quantity of data corruption for LiDAR, camera, and both based on nuScenes. Our model is trained on clean nuScenes and tested on all types of data corruption. Our model consistently achieves the best accuracy on both corrupted and clean data compared to recent models.
>
---
#### [new 085] Alfa: Attentive Low-Rank Filter Adaptation for Structure-Aware Cross-Domain Personalized Gaze Estimation
- **分类: cs.CV**

- **简介: 该论文属于跨域个性化眼动估计任务，解决用户特定差异导致模型性能下降的问题。通过重加权预训练特征，提出Alfa方法实现高效微调。**

- **链接: [https://arxiv.org/pdf/2603.08445](https://arxiv.org/pdf/2603.08445)**

> **作者:** He-Yen Hsieh; Wei-Te Mark Ting; H.T. Kung
>
> **备注:** 21 pages, 16 figures, AAAI2026
>
> **摘要:** Pre-trained gaze models learn to identify useful patterns commonly found across users, but subtle user-specific variations (i.e., eyelid shape or facial structure) can degrade model performance. Test-time personalization (TTP) adapts pre-trained models to these user-specific domain shifts using only a few unlabeled samples. Efficient fine-tuning is critical in performing this domain adaptation: data and computation resources can be limited-especially for on-device customization. While popular parameter-efficient fine-tuning (PEFT) methods address adaptation costs by updating only a small set of weights, they may not be taking full advantage of structures encoded in pre-trained filters. To more effectively leverage existing structures learned during pre-training, we reframe personalization as a process to reweight existing features rather than learning entirely new ones. We present Attentive Low-Rank Filter Adaptation (Alfa) to adapt gaze models by reweighting semantic patterns in pre-trained filters. With Alfa, singular value decomposition (SVD) extracts dominant spatial components that capture eye and facial characteristics across users. Via an attention mechanism, we need only a few unlabeled samples to adjust and reweight pre-trained structures, selectively amplifying those relevant to a target user. Alfa achieves the lowest average gaze errors across four cross-dataset gaze benchmarks, outperforming existing TTP methods and low-rank adaptation (LoRA)-based variants. We also show that Alfa's attentive low-rank methods can be applied to applications beyond vision, such as diffusion-based language models.
>
---
#### [new 086] StreamReady: Learning What to Answer and When in Long Streaming Videos
- **分类: cs.CV**

- **简介: 该论文提出StreamReady框架，解决长视频流中何时回答的问题，通过ARS衡量回答时机与准确性，提升实时视频理解效果。**

- **链接: [https://arxiv.org/pdf/2603.08620](https://arxiv.org/pdf/2603.08620)**

> **作者:** Shehreen Azad; Vibhav Vineet; Yogesh Singh Rawat
>
> **备注:** Accepted in CVPR 2026
>
> **摘要:** Streaming video understanding often involves time-sensitive scenarios where models need to answer exactly when the supporting visual evidence appears: answering before the evidence reflects speculation, answering after it has passed reduces real-time utility. To capture this behavior, we introduce a readiness-aware formulation of streaming video understanding with the Answer Readiness Score (ARS), a timing-aware objective with asymmetric early and late penalties. When combined with correctness, ARS defines an effective accuracy that measures not just whether a model is right, but whether it answers at the appropriate moment. Building on this formulation, we introduce StreamReady, a framework to unify temporal reasoning with on-time answering through a lightweight readiness mechanism that decides if sufficient evidence has been observed before responding. To evaluate this capability, we further introduce ProReady-QA, a benchmark with annotated answer evidence windows and proactive multi-turn questions across local and global contexts. StreamReady achieves superior performance on ProReady-QA, and consistently outperforms prior methods across eight additional streaming and offline long-video benchmarks, demonstrating robust and broadly generalizable video understanding capability.
>
---
#### [new 087] Optimizing Multi-Modal Models for Image-Based Shape Retrieval: The Role of Pre-Alignment and Hard Contrastive Learning
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于图像到3D形状检索任务，解决2D图像与3D形状匹配问题。通过预对齐编码器和硬对比损失提升检索性能，无需视图合成即可实现零样本和跨域检索。**

- **链接: [https://arxiv.org/pdf/2603.06982](https://arxiv.org/pdf/2603.06982)**

> **作者:** Paul Julius Kühn; Cedric Spengler; Michael Weinmann; Arjan Kuijper; Saptarshi Neil Sinha
>
> **摘要:** Image-based shape retrieval (IBSR) aims to retrieve 3D models from a database given a query image, hence addressing a classical task in computer vision, computer graphics, and robotics. Recent approaches typically rely on bridging the domain gap between 2D images and 3D shapes based on the use of multi-view renderings as well as task-specific metric learning to embed shapes and images into a common latent space. In contrast, we address IBSR through large-scale multi-modal pretraining and show that explicit view-based supervision is not required. Inspired by pre-aligned image--point-cloud encoders from ULIP and OpenShape that have been used for tasks such as 3D shape classification, we propose the use of pre-aligned image and shape encoders for zero-shot and standard IBSR by embedding images and point clouds into a shared representation space and performing retrieval via similarity search over compact single-embedding shape descriptors. This formulation allows skipping view synthesis and naturally enables zero-shot and cross-domain retrieval without retraining on the target database. We evaluate pre-aligned encoders in both zero-shot and supervised IBSR settings and additionally introduce a multi-modal hard contrastive loss (HCL) to further increase retrieval performance. Our evaluation demonstrates state-of-the-art performance, outperforming related methods on $Acc_{Top1}$ and $Acc_{Top10}$ for shape retrieval across multiple datasets, with best results observed for OpenShape combined with Point-BERT. Furthermore, training on our proposed multi-modal HCL yields dataset-dependent gains in standard instance retrieval tasks on shape-centric data, underscoring the value of pretraining and hard contrastive learning for 3D shape retrieval. The code will be made available via the project website.
>
---
#### [new 088] MedQ-Deg: A Multidimensional Benchmark for Evaluating MLLMs Across Medical Image Quality Degradations
- **分类: cs.CV**

- **简介: 该论文属于医学多模态大模型评估任务，旨在解决真实临床环境中图像质量退化对模型性能的影响问题。提出MedQ-Deg基准，涵盖多种退化类型和评估维度，分析模型可靠性与信心偏差。**

- **链接: [https://arxiv.org/pdf/2603.07769](https://arxiv.org/pdf/2603.07769)**

> **作者:** Jiyao Liu; Junzhi Ning; Chenglong Ma; Wanying Qu; Jianghan Shen; Siqi Luo; Jinjie Wei; Jin Ye; Pengze Li; Tianbin Li; Jiashi Lin; Hongming Shan; Xinzhe Luo; Xiaohong Liu; Lihao Liu; Junjun He; Ningsheng Xu
>
> **备注:** 29 pages, 11 figures
>
> **摘要:** Despite impressive performance on standard benchmarks, multimodal large language models (MLLMs) face critical challenges in real-world clinical environments where medical images inevitably suffer various quality degradations. Existing benchmarks exhibit two key limitations: (1) absence of large-scale, multidimensional assessment across medical image quality gradients and (2) no systematic confidence calibration analysis. To address these gaps, we present MedQ-Deg, a comprehensive benchmark for evaluating medical MLLMs under image quality degradations. MedQ-Deg provides multi-dimensional evaluation spanning 18 distinct degradation types, 30 fine-grained capability dimensions, and 7 imaging modalities, with 24,894 question-answer pairs. Each degradation is implemented at 3 severity degrees, calibrated by expert radiologists. We further introduce Calibration Shift metric, which quantifies the gap between a model's perceived confidence and actual performance to assess metacognitive reliability under degradation. Our comprehensive evaluation of 40 mainstream MLLMs reveals several critical findings: (1) overall model performance degrades systematically as degradation severity increases, (2) models universally exhibit the AI Dunning-Kruger Effect, maintaining inappropriately high confidence despite severe accuracy collapse, and (3) models display markedly differentiated behavioral patterns across capability dimensions, imaging modalities, and degradation types. We hope MedQ-Deg drives progress toward medical MLLMs that are robust and trustworthy in real clinical practice.
>
---
#### [new 089] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer
- **分类: cs.CV**

- **简介: 该论文提出VisualAD，用于零样本异常检测任务，解决无需目标类异常样本的检测问题。通过纯视觉框架和可学习标记，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.07952](https://arxiv.org/pdf/2603.07952)**

> **作者:** Yanning Hou; Peiyuan Li; Zirui Liu; Yitong Wang; Yanran Ruan; Jianfeng Qiu; Ke Xu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Zero-shot anomaly detection (ZSAD) requires detecting and localizing anomalies without access to target-class anomaly samples. Mainstream methods rely on vision-language models (VLMs) such as CLIP: they build hand-crafted or learned prompt sets for normal and abnormal semantics, then compute image-text similarities for open-set discrimination. While effective, this paradigm depends on a text encoder and cross-modal alignment, which can lead to training instability and parameter redundancy. This work revisits the necessity of the text branch in ZSAD and presents VisualAD, a purely visual framework built on Vision Transformers. We introduce two learnable tokens within a frozen backbone to directly encode normality and abnormality. Through multi-layer self-attention, these tokens interact with patch tokens, gradually acquiring high-level notions of normality and anomaly while guiding patches to highlight anomaly-related cues. Additionally, we incorporate a Spatial-Aware Cross-Attention (SCA) module and a lightweight Self-Alignment Function (SAF): SCA injects fine-grained spatial information into the tokens, and SAF recalibrates patch features before anomaly scoring. VisualAD achieves state-of-the-art performance on 13 zero-shot anomaly detection benchmarks spanning industrial and medical domains, and adapts seamlessly to pretrained vision backbones such as the CLIP image encoder and DINOv2. Code: this https URL
>
---
#### [new 090] ButterflyViT: 354$\times$ Expert Compression for Edge Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ButterflyViT，解决边缘设备部署稀疏MoE Vision Transformer的内存瓶颈问题。通过几何参数化实现专家压缩，显著降低内存消耗并保持精度。任务为视觉Transformer的模型压缩。**

- **链接: [https://arxiv.org/pdf/2603.06746](https://arxiv.org/pdf/2603.06746)**

> **作者:** Aryan Karmore
>
> **摘要:** Deploying sparse Mixture of Experts(MoE) Vision Transformers remains a challenge due to linear expert memory scaling. Linear memory scaling stores $N$ independent expert weight matrices requiring $\mathcal{O}(N_E \cdot d^2)$ memory, which exceeds edge devices memory budget. Current compression methods like quantization, pruning and low-rank factorization reduce constant factors but leave the scaling bottleneck unresolved. We introduce ButterflyViT, a method that treats experts not as independent weight matrices but as geometric reorientations of a unified shared quantized substrate. Diversity among experts arises from viewing different angles of shared capacity, not from redundant storage. By applying learned rotations to a shared ternary prototype, each expert yields $\mathcal{O}(d_{\text{model}} \cdot d_{\text{ff}} + N_E \cdot n_\ell \cdot d)$ memory which is sub-linear in the number of experts. To address the unique challenges of vision, a spatial smoothness regulariser is introduced that penalises routing irregularities between adjacent patch tokens, turning patch correlation into a training signal. Across image classification tasks on CIFAR-100, ButterflyViT achieves 354$\times$ memory reduction at 64 experts with negligible accuracy loss. ButterflyViT allows multiple experts to fit on edge-constrained devices showing that geometric parameterization breaks linear scaling.
>
---
#### [new 091] SRNeRV: A Scale-wise Recursive Framework for Neural Video Representation
- **分类: cs.CV**

- **简介: 该论文属于视频表示与压缩任务，旨在解决多尺度INR生成中的参数冗余问题。提出SRNeRV框架，通过共享架构减少参数量，同时保持尺度特异性。**

- **链接: [https://arxiv.org/pdf/2603.08227](https://arxiv.org/pdf/2603.08227)**

> **作者:** Jia Wang; Jun Zhu; Xinfeng Zhang
>
> **备注:** Accepted by IEEE ISCAS 2026
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as a promising paradigm for video representation and compression. However, existing multi-scale INR generators often suffer from significant parameter redundancy by stacking independent processing blocks for each scale. Inspired by the principle of scale self-similarity in the generation process, we propose SRNeRV, a novel scale-wise recursive framework that replaces this stacked design with a parameter-efficient shared architecture. The core of our approach is a hybrid sharing scheme derived from decoupling the processing block into a scale-specific spatial mixing module and a scale-invariant channel mixing module. We recursively apply the same shared channel mixing module, which contains the majority of the parameters, across all scales, significantly reducing the model size while preserving the crucial capacity to learn scale-specific spatial patterns. Extensive experiments demonstrate that SRNeRV achieves a significant rate-distortion performance boost, especially in INR-friendly scenarios, validating that our sharing scheme successfully amplifies the core strengths of the INR paradigm.
>
---
#### [new 092] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决遥感图像中小目标检测难题。提出ESM-YOLO+模型，通过MEAF模块和SR增强提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.06925](https://arxiv.org/pdf/2603.06925)**

> **作者:** Qianqian Zhang; Xiaolong Jia; Ahmed M. Abdelmoniem; Li Zhou; Junshe An
>
> **备注:** The manuscript has been submitted to the journal and is currently under review
>
> **摘要:** Targets in remote sensing images are usually small, weakly textured, and easily disturbed by complex backgrounds, challenging high-precision detection with general algorithms. Building on our earlier ESM-YOLO, this work presents ESM-YOLO+ as a lightweight visible infrared fusion network. To enhance detection, ESM-YOLO+ includes two key innovations. (1) A Mask-Enhanced Attention Fusion (MEAF) module fuses features at the pixel level via learnable spatial masks and spatial attention, effectively aligning RGB and infrared features, enhancing small-target representation, and alleviating cross-modal misalignment and scale heterogeneity. (2) Training-time Structural Representation (SR) enhancement provides auxiliary supervision to preserve fine-grained spatial structures during training, boosting feature discriminability without extra inference cost. Extensive experiments on the VEDAI and DroneVehicle datasets validate ESM-YOLO+'s superiority. The model achieves 84.71\% mAP on VEDAI and 74.0\% mAP on DroneVehicle, while greatly reducing model complexity, with 93.6\% fewer parameters and 68.0\% lower GFLOPs than the baseline. These results confirm that ESM-YOLO+ integrates strong performance with practicality for real-time deployment, providing an effective solution for high-performance small-target detection in complex remote sensing scenes.
>
---
#### [new 093] Compressed-Domain-Aware Online Video Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于在线视频超分辨率任务，旨在解决高分辨率视频实时处理效率低的问题。通过利用压缩域信息，设计高效模块提升处理速度与质量。**

- **链接: [https://arxiv.org/pdf/2603.07694](https://arxiv.org/pdf/2603.07694)**

> **作者:** Yuhang Wang; Hai Li; Shujuan Hou; Zhetao Dong; Xiaoyao Yang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** In bandwidth-limited online video streaming, videos are usually downsampled and compressed. Although recent online video super-resolution (online VSR) approaches achieve promising results, they are still compute-intensive and fall short of real-time processing at higher resolutions, due to complex motion estimation for alignment and redundant processing of consecutive frames. To address these issues, we propose a compressed-domain-aware network (CDA-VSR) for online VSR, which utilizes compressed-domain information, including motion vectors, residual maps, and frame types to balance quality and efficiency. Specifically, we propose a motion-vector-guided deformable alignment module that uses motion vectors for coarse warping and learns only local residual offsets for fine-tuned adjustments, thereby maintaining accuracy while reducing computation. Then, we utilize a residual map gated fusion module to derive spatial weights from residual maps, suppressing mismatched regions and emphasizing reliable details. Further, we design a frame-type-aware reconstruction module for adaptive compute allocation across frame types, balancing accuracy and efficiency. On the REDS4 dataset, our CDA-VSR surpasses the state-of-the-art method TMP, with a maximum PSNR improvement of 0.13 dB while delivering more than double the inference speed. The code will be released at this https URL.
>
---
#### [new 094] A prior information informed learning architecture for flying trajectory prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于轨迹预测任务，旨在解决传统方法在物理建模和计算效率上的不足。通过融合环境先验信息与双Transformer级联架构，提升飞行轨迹预测精度。**

- **链接: [https://arxiv.org/pdf/2603.06863](https://arxiv.org/pdf/2603.06863)**

> **作者:** Xianda Huang; Zidong Han; Ruibo Jin; Zhenyu Wang; Wenyu Li; Xiaoyang Li; Yi Gong
>
> **摘要:** Trajectory prediction for flying objects is critical in domains ranging from sports analytics to aerospace. However, traditional methods struggle with complex physical modeling, computational inefficiencies, and high hardware demands, often neglecting critical trajectory events like landing points. This paper introduces a novel, hardware-efficient trajectory prediction framework that integrates environmental priors with a Dual-Transformer-Cascaded (DTC) architecture. We demonstrate this approach by predicting the landing points of tennis balls in real-world outdoor courts. Using a single industrial camera and YOLO-based detection, we extract high-speed flight coordinates. These coordinates, fused with structural environmental priors (e.g., court boundaries), form a comprehensive dataset fed into our proposed DTC model. A first-level Transformer classifies the trajectory, while a second-level Transformer synthesizes these features to precisely predict the landing point. Extensive ablation and comparative experiments demonstrate that integrating environmental priors within the DTC architecture significantly outperforms existing trajectory prediction frameworks
>
---
#### [new 095] Three-dimensional reconstruction and segmentation of an aggregate stockpile for size and shape analyses
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于3D重建与分割任务，旨在解决现场 aggregate 尺寸和形状分析的问题。通过SfM技术从视频重建点云，并分割提取单个颗粒，用于质量控制。**

- **链接: [https://arxiv.org/pdf/2603.06684](https://arxiv.org/pdf/2603.06684)**

> **作者:** Erol Tutumluer; Haohang Huang; Jiayi Luo; Issam Qamhia; John M. Hart
>
> **备注:** 7 pages, 4 figures, Proceedings of the 20th International Conference on Soil Mechanics and Geotechnical Engineering
>
> **摘要:** Aggregate size and shape are key properties for determining quality of aggregate materials used in road construction and transportation geotechnics applications. The composition and packing, layer stiffness, and load response are all influenced by these morphological characteristics of aggregates. Many aggregate imaging systems developed to date only focus on analyses of individual or manually separated aggregate particles. There is a need to develop a convenient and affordable system for acquiring 3D aggregate information from stockpiles in the field. This paper presents an innovative 3D imaging approach for potential field evaluation of large-sized aggregates, whereby engineers can perform inspection by taking videos/images with mobile devices such as smartphone cameras. The approach leverages Structure-from-Motion (SfM) techniques to reconstruct the stockpile surface as 3D spatial data, i.e. point cloud, and uses a 3D segmentation algorithm to separate and extract individual aggregates from the reconstructed stockpile. The preliminary results presented in this paper demonstrate the future potential of using 3D aggregate size and shape information for onsite Quality Assurance/Quality Control (QA/QC) tasks.
>
---
#### [new 096] SJD-PV: Speculative Jacobi Decoding with Phrase Verification for Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文属于自回归图像生成任务，旨在解决推理速度慢的问题。通过引入短语级推测验证，提升解码效率并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2603.06666](https://arxiv.org/pdf/2603.06666)**

> **作者:** Zhehao Yu; Baoquan Zhang; Bingqi Shan; Xinhao Liu; Dongliang Zhou; Guotao Liang; Guangming Ye; Yunming Ye
>
> **摘要:** Autoregressive (AR) image models have recently demonstrated remarkable generative capability, but their sequential nature results in significant inference latency. Existing training-free acceleration methods typically verify tokens independently, overlooking the strong co-occurrence patterns between adjacent visual tokens. This independence assumption often leads to contextual inconsistency and limits decoding efficiency. In this work, we introduce a novel training-free acceleration framework that performs phrase-level speculative verification, enabling the model to jointly validate multiple correlated tokens within each decoding window. To construct such phrase units, we analyze token co-occurrence statistics from the training corpus and group frequently co-occurring tokens into semantically coherent visual phrases. During inference, the proposed phrase-level verification evaluates aggregated likelihood ratios over each phrase, allowing simultaneous acceptance of multiple tokens while preserving generation quality. Extensive experiments on autoregressive text-to-image generation show that our method significantly reduces the number of function evaluations (NFE) and achieves up to 30% faster decoding without compromising visual fidelity. Our findings reveal that modeling short-range token co-occurrence provides an effective and general principle for accelerating autoregressive inference.
>
---
#### [new 097] Margin-Consistent Deep Subtyping of Invasive Lung Adenocarcinoma via Perturbation Fidelity in Whole-Slide Image Analysis
- **分类: cs.CV**

- **简介: 该论文属于肺腺癌亚型分类任务，解决图像扰动导致模型不可靠的问题。通过注意力聚合与边缘一致性训练，结合扰动保真度评分，提升分类准确率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.06650](https://arxiv.org/pdf/2603.06650)**

> **作者:** Meghdad Sabouri Rad; Junze; Huang; Mohammad Mehdi Hosseini; Rakesh Choudhary; Saverio J. Carello; Ola El-Zammar; Michel R. Nasr; Bardia Rodd
>
> **备注:** This document is the author's accepted manuscript (author version). The final published version is available online in the Journal of Imaging Informatics in Medicine at DOI: https://doi.org/10.1007/s10278-026-01875-6
>
> **摘要:** Whole-slide image classification for invasive lung adenocarcinoma subtyping remains vulnerable to real-world imaging perturbations that undermine model reliability at the decision boundary. We propose a margin consistency framework evaluated on 203,226 patches from 143 whole-slide images spanning five adenocarcinoma subtypes in the BMIRDS-LUAD dataset. By combining attention-weighted patch aggregation with margin-aware training, our approach achieves robust feature-logit space alignment measured by Kendall correlations of 0.88 during training and 0.64 during validation. Contrastive regularization, while effective at improving class separation, tends to over-cluster features and suppress fine-grained morphological variation; to counteract this, we introduce Perturbation Fidelity (PF) scoring, which imposes structured perturbations through Bayesian-optimized parameters. Vision Transformer-Large achieves 95.20 +/- 4.65% accuracy, representing a 40% error reduction from the 92.00 +/- 5.36% baseline, while ResNet101 with an attention mechanism reaches 95.89 +/- 5.37% from 91.73 +/- 9.23%, a 50% error reduction. All five subtypes exceed an area under the receiver operating characteristic curve (AUC) of 0.99. On the WSSS4LUAD external benchmark, ResNet50 with an attention mechanism attains 80.1% accuracy, demonstrating cross-institutional generalizability despite approximately 15-20% domain-shift-related degradation and identifying opportunities for future adaptation research.
>
---
#### [new 098] Physics-Guided VLM Priors for All-Cloud Removal
- **分类: cs.CV**

- **简介: 该论文属于遥感图像处理任务，旨在解决云层遮挡导致的影像恢复问题。通过融合物理模型与视觉语言模型，实现统一高效的全云去除。**

- **链接: [https://arxiv.org/pdf/2603.07074](https://arxiv.org/pdf/2603.07074)**

> **作者:** Liying Xu; Huifang Li; Huanfeng Shen
>
> **摘要:** Cloud removal is a fundamental challenge in optical remote sensing due to the heterogeneous degradation. Thin clouds distort radiometry via partial transmission, while thick clouds occlude the surface. Existing pipelines separate thin-cloud correction from thick-cloud reconstruction, requiring explicit cloud-type decisions and often leading to error accumulation and discontinuities in mixed-cloud scenes. Therefore, a novel approach named Physical-VLM All-Cloud Removal (PhyVLM-CR) that integrates the semantic capability of Vision-Language Model (VLM) into a physical restoration model, achieving high-fidelity unified cloud removal. Specifically, the cognitive prior from a VLM (e.g., Qwen) is transformed into physical scattering parameters and a hallucination confidence map. Leveraging this confidence map as a continuous soft gate, our method achieves a unified restoration via adaptive weighting: it prioritizes physical inversion in high-transmission regions to preserve radiometric fidelity, while seamlessly transitioning to temporal reference reconstruction in low-confidence occluded areas. This mechanism eliminates the need for explicit boundary delineation, ensuring a coherent removal across heterogeneous cloud covers. Experiments on real-world Sentinel-2 surface reflectance imagery confirm that our approach achieves a remarkable balance between cloud removal and content preservation, delivering hallucination-free results with substantially improved quantitative accuracy compared to existing methods.
>
---
#### [new 099] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语言模型任务，旨在提升模型的空间推理能力。针对现有方法忽略对象间关系的问题，提出Graph-of-Mark技术，通过图结构增强视觉提示，显著提升模型在定位和方向理解上的零样本性能。**

- **链接: [https://arxiv.org/pdf/2603.06663](https://arxiv.org/pdf/2603.06663)**

> **作者:** Giacomo Frisoni; Lorenzo Molfetta; Mattia Buzzoni; Gianluca Moro
>
> **备注:** AAAI-26 (Main Track)
>
> **摘要:** Recent advances in training-free visual prompting, such as Set-of-Mark, have emerged as a promising direction for enhancing the grounding capabilities of multimodal language models (MLMs). These techniques operate by partitioning the input image into object regions and annotating them with marks, predominantly boxes with numeric identifiers, before feeding the augmented image to the MLM. However, these approaches treat marked objects as isolated entities, failing to capture the relationships between them. On these premises, we propose Graph-of-Mark (GoM), the first pixel-level visual prompting technique that overlays scene graphs onto the input image for spatial reasoning tasks. We evaluate GoM across 3 open-source MLMs and 4 different datasets, conducting extensive ablations on drawn components and investigating the impact of auxiliary graph descriptions in the text prompt. Our results demonstrate that GoM consistently improves the zero-shot capability of MLMs in interpreting object positions and relative directions, improving base accuracy in visual question answering and localization up to 11 percentage points.
>
---
#### [new 100] 4DRC-OCC: Robust Semantic Occupancy Prediction Through Fusion of 4D Radar and Camera
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的3D语义占用预测任务，旨在解决恶劣环境下感知不准确的问题。通过融合4D雷达和相机数据，提升场景理解的鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2603.07794](https://arxiv.org/pdf/2603.07794)**

> **作者:** David Ninfa; Andras Palffy; Holger Caesar
>
> **摘要:** Autonomous driving requires robust perception across diverse environmental conditions, yet 3D semantic occupancy prediction remains challenging under adverse weather and lighting. In this work, we present the first study combining 4D radar and camera data for 3D semantic occupancy prediction. Our fusion leverages the complementary strengths of both modalities: 4D radar provides reliable range, velocity, and angle measurements in challenging conditions, while cameras contribute rich semantic and texture information. We further show that integrating depth cues from camera pixels enables lifting 2D images to 3D, improving scene reconstruction accuracy. Additionally, we introduce a fully automatically labeled dataset for training semantic occupancy models, substantially reducing reliance on costly manual annotation. Experiments demonstrate the robustness of 4D radar across diverse scenarios, highlighting its potential to advance autonomous vehicle perception.
>
---
#### [new 101] TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery
- **分类: cs.CV**

- **简介: 该论文属于在线类别发现任务，旨在识别已知类别并发现新类别。针对现有方法的不足，提出一种测试时自适应学习框架，通过动态更新原型和编码器，提升分类性能并减少类别爆炸。**

- **链接: [https://arxiv.org/pdf/2603.08075](https://arxiv.org/pdf/2603.08075)**

> **作者:** Yanan Wu; Yuhan Yan; Tailai Chen; Zhixiang Chi; ZiZhang Wu; Yi Jin; Yang Wang; Zhenbo Li
>
> **备注:** 14 pages, 6 figures, accepted by CVPR 2026
>
> **摘要:** On-the-fly category discovery (OCD) aims to recognize known categories while simultaneously discovering novel ones from an unlabeled online stream, using a model trained only on labeled data. Existing approaches freeze the feature extractor trained offline and employ a hash-based framework that quantizes features into binary codes as class prototypes. However, discovering novel categories with a fixed knowledge base is counterintuitive, as the learning potential of incoming data is entirely neglected. In addition, feature quantization introduces information loss, diminishes representational expressiveness, and amplifies intra-class variance. It often results in category explosion, where a single class is fragmented into multiple pseudo-classes. To overcome these limitations, we propose a test-time adaptation framework that enables learning through discovery. It incorporates two complementary strategies: a semantic-aware prototype update and a stable test-time encoder update. The former dynamically refines class prototypes to enhance classification, whereas the latter integrates new information directly into the parameter space. Together, these components allow the model to continuously expand its knowledge base with newly encountered samples. Furthermore, we introduce a margin-aware logit calibration in the offline stage to enlarge inter-class margins and improve intra-class compactness, thereby reserving embedding space for future class discovery. Experiments on standard OCD benchmarks demonstrate that our method substantially outperforms existing hash-based state-of-the-art approaches, yielding notable improvements in novel-class accuracy and effectively mitigating category explosion. The code is publicly available at \textcolor{blue}{this https URL}.
>
---
#### [new 102] Structure and Progress Aware Diffusion for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决目标边界模糊与噪声问题。提出SPAD方法，通过结构与进度感知的扩散机制，分离学习粗结构与细边界。**

- **链接: [https://arxiv.org/pdf/2603.07889](https://arxiv.org/pdf/2603.07889)**

> **作者:** Siyuan Song; Guyue Hu; Chenglong Li; Dengdi Sun; Zhe Jin; Jin Tang
>
> **摘要:** Medical image segmentation is crucial for computer-aided diagnosis, which necessitates understanding both coarse morphological and semantic structures, as well as carving fine boundaries. The morphological and semantic structures in medical images are beneficial and stable clues for target understanding. While the fine boundaries of medical targets (like tumors and lesions) are usually ambiguous and noisy since lesion overlap, annotation uncertainty, and so on, making it not reliable to serve as early supervision. However, existing methods simultaneously learn coarse structures and fine boundaries throughout the training process. In this paper, we propose a structure and progress-aware diffusion (SPAD) for medical image segmentation, which consists of a semantic-concentrated diffusion (ScD) and a boundary-centralized diffusion (BcD) modulated by a progress-aware scheduler (PaS). Specifically, the semantic-concentrated diffusion introduces anchor-preserved target perturbation, which perturbs pixels within a medical target but preserves unaltered areas as semantic anchors, encouraging the model to infer noisy target areas from the surrounding semantic context. The boundary-centralized diffusion introduces progress-aware boundary noise, which blurs unreliable and ambiguous boundaries, thus compelling the model to focus on coarse but stable anatomical morphology and global semantics. Furthermore, the progress-aware scheduler gradually modulates noise intensity of the ScD and BcD forming a coarse-to-fine diffusion paradigm, which encourage focusing on coarse morphological and semantic structures during early target understanding stages and gradually shifting to fine target boundaries during later contour adjusting stages.
>
---
#### [new 103] HY-WU (Part I): An Extensible Functional Neural Memory Framework and An Instantiation in Text-Guided Image Editing
- **分类: cs.CV**

- **简介: 该论文提出HY-WU框架，解决持续学习与个性化问题，通过功能记忆模块生成实例特定的权重更新，避免参数覆盖导致的性能下降。**

- **链接: [https://arxiv.org/pdf/2603.07236](https://arxiv.org/pdf/2603.07236)**

> **作者:** Tencent HY Team
>
> **摘要:** Foundation models are transitioning from offline predictors to deployed systems expected to operate over long time horizons. In real deployments, objectives are not fixed: domains drift, user preferences evolve, and new tasks appear after the model has shipped. This elevates continual learning and instant personalization from optional features to core architectural requirements. Yet most adaptation pipelines still follow a static weight paradigm: after training (or after any adaptation step), inference executes a single parameter vector regardless of user intent, domain, or instance-specific constraints. This treats the trained or adapted model as a single point in parameter space. In heterogeneous and continually evolving regimes, distinct objectives can induce separated feasible regions over parameters, forcing any single shared update into compromise, interference, or overspecialization. As a result, continual learning and personalization are often implemented as repeated overwriting of shared weights, risking degradation of previously learned behaviors. We propose HY-WU (Weight Unleashing), a memory-first adaptation framework that shifts adaptation pressure away from overwriting a single shared parameter point. HY-WU implements functional (operator-level) memory as a neural module: a generator that synthesizes weight updates on-the-fly from the instance condition, yielding instance-specific operators without test-time optimization.
>
---
#### [new 104] Multi-label Instance-level Generalised Visual Grounding in Agriculture
- **分类: cs.CV**

- **简介: 该论文属于农业领域的多标签实例级视觉定位任务，旨在解决作物与杂草识别中的定位问题。提出gRef-CW数据集和Weed-VG框架，提升农业图像中目标的精准定位能力。**

- **链接: [https://arxiv.org/pdf/2603.06699](https://arxiv.org/pdf/2603.06699)**

> **作者:** Mohammadreza Haghighat; Alzayat Saleh; Mostafa Rahimi Azghadi
>
> **摘要:** Understanding field imagery such as detecting plants and distinguishing individual crop and weed instances is a central challenge in precision agriculture. Despite progress in vision-language tasks like captioning and visual question answering, Visual Grounding (VG), localising language-referred objects, remains unexplored in agriculture. A key reason is the lack of suitable benchmark datasets for evaluating grounding models in field conditions, where many plants look highly similar, appear at multiple scales, and the referred target may be absent from the image. To address these limitations, we introduce gRef-CW, the first dataset designed for generalised visual grounding in agriculture, including negative expressions. Benchmarking current state-of-the-art grounding models on gRef-CW reveals a substantial domain gap, highlighting their inability to ground instances of crops and weeds. Motivated by these findings, we introduce Weed-VG, a modular framework that incorporates multi-label hierarchical relevance scoring and interpolation-driven regression. Weed-VG advances instance-level visual grounding and provides a clear baseline for developing VG methods in precision agriculture. Code will be released upon acceptance.
>
---
#### [new 105] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，旨在解决扩散Transformer推理效率低的问题。通过动态感知敏感性，结合缓存与剪枝，提升生成质量与加速效果。**

- **链接: [https://arxiv.org/pdf/2603.07057](https://arxiv.org/pdf/2603.07057)**

> **作者:** Tong Shao; Yusen Fu; Guoying Sun; Jingde Kong; Zhuotao Tian; Jingyong Su
>
> **备注:** 23 pages, CVPR 2026 accepted
>
> **摘要:** Diffusion Transformers have become a dominant paradigm in visual generation, yet their low inference efficiency remains a key bottleneck hindering further advancement. Among common training-free techniques, caching offers high acceleration efficiency but often compromises fidelity, whereas pruning shows the opposite trade-off. Integrating caching with pruning achieves a balance between acceleration and generation quality. However, existing methods typically employ fixed and heuristic schemes to configure caching and pruning strategies. While they roughly follow the overall sensitivity trend of generation models to acceleration, they fail to capture fine-grained and complex variations, inevitably skipping highly sensitive computations and leading to quality degradation. Furthermore, such manually designed strategies exhibit poor generalization. To address these issues, we propose SODA, a Sensitivity-Oriented Dynamic Acceleration method that adaptively performs caching and pruning based on fine-grained sensitivity. SODA builds an offline sensitivity error modeling framework across timesteps, layers, and modules to capture the sensitivity to different acceleration operations. The cache intervals are optimized via dynamic programming with sensitivity error as the cost function, minimizing the impact of caching on model sensitivity. During pruning and cache reuse, SODA adaptively determines the pruning timing and rate to preserve computations of highly sensitive tokens, significantly enhancing generation fidelity. Extensive experiments on DiT-XL/2, PixArt-$\alpha$, and OpenSora demonstrate that SODA achieves state-of-the-art generation fidelity under controllable acceleration ratios. Our code is released publicly at: this https URL.
>
---
#### [new 106] Real-Time Drone Detection in Event Cameras via Per-Pixel Frequency Analysis
- **分类: cs.CV**

- **简介: 该论文属于实时无人机检测任务，解决事件相机数据中快速移动目标检测问题。提出DDHF方法，利用非均匀傅里叶变换分析像素频率特征，实现高效准确的无人机定位。**

- **链接: [https://arxiv.org/pdf/2603.08386](https://arxiv.org/pdf/2603.08386)**

> **作者:** Michael Bezick; Majid Sahin
>
> **摘要:** Detecting fast-moving objects, such as unmanned aerial vehicle (UAV), from event camera data is challenging due to the sparse, asynchronous nature of the input. Traditional Discrete Fourier Transforms (DFT) are effective at identifying periodic signals, such as spinning rotors, but they assume uniformly sampled data, which event cameras do not provide. We propose a novel per-pixel temporal analysis framework using the Non-uniform Discrete Fourier Transform (NDFT), which we call Drone Detection via Harmonic Fingerprinting (DDHF). Our method uses purely analytical techniques that identify the frequency signature of drone rotors, as characterized by frequency combs in their power spectra, enabling a tunable and generalizable algorithm that achieves accurate real-time localization of UAV. We compare against a YOLO detector under equivalent conditions, demonstrating improvement in accuracy and latency across a difficult array of drone speeds, distances, and scenarios. DDHF achieves an average localization F1 score of 90.89% and average latency of 2.39ms per frame, while YOLO achieves an F1 score of 66.74% and requires 12.40ms per frame. Through utilization of purely analytic techniques, DDHF is quickly tuned on small data, easily interpretable, and achieves competitive accuracies and latencies to deep learning alternatives.
>
---
#### [new 107] Generalization in Online Reinforcement Learning for Mobile Agents
- **分类: cs.CV; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于在线强化学习任务，旨在提升移动代理的泛化能力。针对缺乏基准和系统的问题，提出AndroidWorld-Generalization基准和GRPO训练系统，实验显示RL在未见任务上表现优于监督微调。**

- **链接: [https://arxiv.org/pdf/2603.07432](https://arxiv.org/pdf/2603.07432)**

> **作者:** Li Gu; Zihuan Jiang; Zhixiang Chi; Huan Liu; Ziqiang Wang; Yuanhao Yu; Glen Berseth; Yang Wang
>
> **摘要:** Graphical user interface (GUI)-based mobile agents automate digital tasks on mobile devices by interpreting natural-language instructions and interacting with the screen. While recent methods apply reinforcement learning (RL) to train vision-language-model(VLM) agents in interactive environments with a primary focus on performance, generalization remains underexplored due to the lack of standardized benchmarks and open-source RL systems. In this work, we formalize the problem as a Contextual Markov Decision Process (CMDP) and introduce \textbf{AndroidWorld-Generalization}, a benchmark with three increasingly challenging regimes for evaluating zero-shot generalization to unseen task instances, templates, and applications. We further propose an RL training system that integrates Group Relative Policy Optimization (GRPO) with a scalable rollout collection system, consisting of containerized infrastructure and asynchronous execution % , and error recovery to support reliable and efficient training. Experiments on AndroidWorld-Generalization show that RL enables a 7B-parameter VLM agent to surpass supervised fine-tuning baselines, yielding a 26.1\% improvement on unseen instances but only limited gains on unseen templates (15.7\%) and apps (8.3\%), underscoring the challenges of generalization. As a preliminary step, we demonstrate that few-shot adaptation at test-time improves performance on unseen apps, motivating future research in this direction. To support reproducibility and fair comparison, we open-source the full RL training system, including the environment, task suite, models, prompt configurations, and the underlying infrastructure \footnote{this https URL}.
>
---
#### [new 108] $L^3$:Scene-agnostic Visual Localization in the Wild
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，解决无需离线预处理的野外场景定位问题。通过在线3D重建和姿态优化，提出无地图的定位框架$L^3$，实现高精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07937](https://arxiv.org/pdf/2603.07937)**

> **作者:** Yu Zhang; Muhua Zhu; Yifei Xue; Tie Ji; Yizhen Lao
>
> **摘要:** Standard visual localization methods typically require offline pre-processing of scenes to obtain 3D structural information for better performance. This inevitably introduces additional computational and time costs, as well as the overhead of storing scene representations. Can we visually localize in a wild scene without any off-line preprocessing step? In this paper, we leverage the online inference capabilities of feed-forward 3D reconstruction networks to propose a novel map-free visual localization framework $L^3$. Specifically, by performing direct online 3D reconstruction on RGB images, followed by two-stage metric scale recovery and pose refinement based on 2D-3D correspondences, $L^3$ achieves high accuracy without the need to pre-build or store any offline scene representations. Extensive experiments demonstrate $L^3$ not only that the performance is comparable to state-of-the-art solutions on various benchmarks, but also that it exhibits significantly superior robustness in sparse scenes (fewer reference images per scene).
>
---
#### [new 109] Fine-Grained 3D Facial Reconstruction for Micro-Expressions
- **分类: cs.CV**

- **简介: 该论文属于3D人脸微表情重建任务，旨在解决微表情特征细微、短暂且强度低导致的重建难题。通过融合全局动态特征与局部丰富信息，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.07043](https://arxiv.org/pdf/2603.07043)**

> **作者:** Che Sun; Xinjie Zhang; Rui Gao; Xu Chen; Yuwei Wu; Yunde Jia
>
> **摘要:** Recent advances in 3D facial expression reconstruction have demonstrated remarkable performance in capturing macro-expressions, yet the reconstruction of micro-expressions remains unexplored. This novel task is particularly challenging due to the subtle, transient, and low-intensity nature of micro-expressions, which complicate the extraction of stable and discriminative features essential for accurate reconstruction. In this paper, we propose a fine-grained micro-expression reconstruction method that integrates a global dynamic feature capturing stable facial motion patterns with a locally-enriched feature incorporating multiple informative cues from 2D motions, facial priors and 3D facial geometry. Specifically, we devise a plug-and-play dynamic-encoded module to extract micro-expression feature for global facial action, allowing it to leverage prior knowledge from abundant macro-expression data to mitigate the scarcity of micro-expression data. Subsequently, a dynamic-guided mesh deformation module is designed for extracting aggregated local features from dense optical flow, sparse landmark cues and facial mesh geometry, which adaptively refines fine-grained facial micro-expression without compromising global 3D geometry. Extensive experiments on micro-expression datasets demonstrate that our method consistently outperforms state-of-the-art methods in both geometric accuracy and perceptual detail.
>
---
#### [new 110] OV-DEIM: Real-time DETR-Style Open-Vocabulary Object Detection with GridSynthetic Augmentation
- **分类: cs.CV**

- **简介: 该论文属于实时开放词汇目标检测任务，解决DETR模型在实时性与性能上的不足。提出OV-DEIM框架，结合视觉-语言建模和GridSynthetic数据增强，提升检测效率与罕见类别识别能力。**

- **链接: [https://arxiv.org/pdf/2603.07022](https://arxiv.org/pdf/2603.07022)**

> **作者:** Leilei Wang; Longfei Liu; Xi Shen; Xuanlong Yu; Ying Tiffany He; Fei Richard Yu; Yingyi Chen
>
> **摘要:** Real-time open-vocabulary object detection (OVOD) is essential for practical deployment in dynamic environments, where models must recognize a large and evolving set of categories under strict latency constraints. Current real-time OVOD methods are predominantly built upon YOLO-style models. In contrast, real-time DETR-based methods still lag behind in terms of inference latency, model lightweightness, and overall performance. In this work, we present OV-DEIM, an end-to-end DETR-style open-vocabulary detector built upon the recent DEIMv2 framework with integrated vision-language modeling for efficient open-vocabulary inference. We further introduce a simple query supplement strategy that improves Fixed AP without compromising inference speed. Beyond architectural improvements, we introduce GridSynthetic, a simple yet effective data augmentation strategy that composes multiple training samples into structured image grids. By exposing the model to richer object co-occurrence patterns and spatial layouts within a single forward pass, GridSynthetic mitigates the negative impact of noisy localization signals on the classification loss and improves semantic discrimination, particularly for rare categories. Extensive experiments demonstrate that OV-DEIM achieves state-of-the-art performance on open-vocabulary detection benchmarks, delivering superior efficiency and notable improvements on challenging rare categories. Code and pretrained models are available at this https URL.
>
---
#### [new 111] EmbedTalk: Triplane-Free Talking Head Synthesis using Embedding-Driven Gaussian Deformation
- **分类: cs.CV**

- **简介: 该论文属于 Talking Head Synthesis 任务，解决实时语音驱动面部动画问题。提出 EmbedTalk，用学习的嵌入代替三平面编码，提升渲染质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.07604](https://arxiv.org/pdf/2603.07604)**

> **作者:** Arpita Saggar; Jonathan C. Darling; Duygu Sarikaya; David C. Hogg
>
> **备注:** Preprint
>
> **摘要:** Real-time talking head synthesis increasingly relies on deformable 3D Gaussian Splatting (3DGS) due to its low latency. Tri-planes are the standard choice for encoding Gaussians prior to deformation, since they provide a continuous domain with explicit spatial relationships. However, tri-plane representations are limited by grid resolution and approximation errors introduced by projecting 3D volumetric fields onto 2D subspaces. Recent work has shown the superiority of learnt embeddings for driving temporal deformations in 4D scene reconstruction. We introduce $\textbf{EmbedTalk}$, which shows how such embeddings can be leveraged for modelling speech deformations in talking head synthesis. Through comprehensive experiments, we show that EmbedTalk outperforms existing 3DGS-based methods in rendering quality, lip synchronisation, and motion consistency, while remaining competitive with state-of-the-art generative models. Moreover, replacing the tri-plane encoding with learnt embeddings enables significantly more compact models that achieve over 60 FPS on a mobile GPU (RTX 2060 6 GB). Our code will be placed in the public domain on acceptance.
>
---
#### [new 112] MedSteer: Counterfactual Endoscopic Synthesis via Training-Free Activation Steering
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedSteer，用于医学内镜图像的因果生成。解决传统方法生成数据不准确的问题，通过激活调节生成反事实图像，保持结构不变。**

- **链接: [https://arxiv.org/pdf/2603.07066](https://arxiv.org/pdf/2603.07066)**

> **作者:** Trong-Thang Pham; Loc Nguyen; Anh Nguyen; Hien Nguyen; Ngan Le
>
> **摘要:** Generative diffusion models are increasingly used for medical imaging data augmentation, but text prompting cannot produce causal training data. Re-prompting rerolls the entire generation trajectory, altering anatomy, texture, and background. Inversion-based editing methods introduce reconstruction error that causes structural drift. We propose MedSteer, a training-free activation-steering framework for endoscopic synthesis. MedSteer identifies a pathology vector for each contrastive prompt pair in the cross-attention layers of a diffusion transformer. At inference time, it steers image activations along this vector, generating counterfactual pairs from scratch where the only difference is the steered concept. All other structure is preserved by construction. We evaluate MedSteer across three experiments on Kvasir v3 and HyperKvasir. On counterfactual generation across three clinical concept pairs, MedSteer achieves flip rates of 0.800, 0.925, and 0.950, outperforming the best inversion-based baseline in both concept flip rate and structural preservation. On dye disentanglement, MedSteer achieves 75% dye removal against 20% (PnP) and 10% (h-Edit). On downstream polyp detection, augmenting with MedSteer counterfactual pairs achieves ViT AUC of 0.9755 versus 0.9083 for quantity-matched re-prompting, confirming that counterfactual structure drives the gain. Code is at link this https URL
>
---
#### [new 113] CONSTANT: Towards High-Quality One-Shot Handwriting Generation with Patch Contrastive Enhancement and Style-Aware Quantization
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于手写生成任务，旨在解决单张参考图像生成高质量手写图像的问题。提出CONSTANT方法，通过风格感知量化和对比增强提升生成质量与风格适应性。**

- **链接: [https://arxiv.org/pdf/2603.07543](https://arxiv.org/pdf/2603.07543)**

> **作者:** Anh-Duy Le; Van-Linh Pham; Thanh-Nam Vo; Xuan Toan Mai; Tuan-Anh Tran
>
> **备注:** Accepted as oral presentation at WACV 2026
>
> **摘要:** One-shot styled handwriting image generation, despite achieving impressive results in recent years, remains challenging due to the difficulty in capturing the intricate and diverse characteristics of human handwriting by using solely a single reference image. Existing methods still struggle to generate visually appealing and realistic handwritten images and adapt to complex, unseen writer styles, struggling to isolate invariant style features (e.g., slant, stroke width, curvature) while ignoring irrelevant noise. To tackle this problem, we introduce Patch Contrastive Enhancement and Style-Aware Quantization via Denoising Diffusion (CONSTANT), a novel one-shot handwriting generation via diffusion model. CONSTANT leverages three key innovations: 1) a Style-Aware Quantization (SAQ) module that models style as discrete visual tokens capturing distinct concepts; 2) a contrastive objective to ensure these tokens are well-separated and meaningful in the embedding style space; 3) a latent patch-based contrastive (LLatentPCE) objective help improving quality and local structures by aligning multiscale spatial patches of generated and real features in latent space. Extensive experiments and analysis on benchmark datasets from multiple languages, including English, Chinese, and our proposed ViHTGen dataset for Vietnamese, demonstrate the superiority of adapting to new reference styles and producing highly detailed images of our method over state-of-the-art approaches. Code is available at GitHub
>
---
#### [new 114] AutoFigure-Edit: Generating Editable Scientific Illustration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于科学插图生成任务，旨在解决现有系统在可编辑性、风格控制和效率上的不足。提出AutoFigure-Edit系统，实现从文本生成可编辑的科学插图，并支持风格迁移。**

- **链接: [https://arxiv.org/pdf/2603.06674](https://arxiv.org/pdf/2603.06674)**

> **作者:** Zhen Lin; Qiujie Xie; Minjun Zhu; Shichen Li; Qiyao Sun; Enhao Gu; Yiran Ding; Ke Sun; Fang Guo; Panzhong Lu; Zhiyuan Ning; Yixuan Weng; Yue Zhang
>
> **摘要:** High-quality scientific illustrations are essential for communicating complex scientific and technical concepts, yet existing automated systems remain limited in editability, stylistic controllability, and efficiency. We present AutoFigure-Edit, an end-to-end system that generates fully editable scientific illustrations from long-form scientific text while enabling flexible style adaptation through user-provided reference images. By combining long-context understanding, reference-guided styling, and native SVG editing, it enables efficient creation and refinement of high-quality scientific illustrations. To facilitate further progress in this field, we release the video at this https URL, full codebase at this https URL and provide a website for easy access and interactive use at this https URL.
>
---
#### [new 115] ACD-U: Asymmetric co-teaching with machine unlearning for robust learning with noisy labels
- **分类: cs.CV**

- **简介: 该论文属于噪声标签下的鲁棒学习任务，旨在解决模型因记忆错误标签而泛化能力下降的问题。通过引入不对称协同教学和机器遗忘机制，提升模型在噪声数据中的性能。**

- **链接: [https://arxiv.org/pdf/2603.07166](https://arxiv.org/pdf/2603.07166)**

> **作者:** Reo Fukunaga; Soh Yoshida; Mitsuji Muneyasu
>
> **摘要:** Deep neural networks are prone to memorizing incorrect labels during training, which degrades their generalizability. Although recent methods have combined sample selection with semi-supervised learning (SSL) to exploit the memorization effect -- where networks learn from clean data before noisy data -- they cannot correct selection errors once a sample is misclassified. To overcome this, we propose asymmetric co-teaching with different architectures (ACD)-U, an asymmetric co-teaching framework that uses different model architectures and incorporates machine unlearning. ACD-U addresses this limitation through two core mechanisms. First, its asymmetric co-teaching pairs a contrastive language-image pretraining (CLIP)-pretrained vision Transformer with a convolutional neural network (CNN), leveraging their complementary learning behaviors: the pretrained model provides stable predictions, whereas the CNN adapts throughout training. This asymmetry, where the vision Transformer is trained only on clean samples and the CNN is trained through SSL, effectively mitigates confirmation bias. Second, selective unlearning enables post-hoc error correction by identifying incorrectly memorized samples through loss trajectory analysis and CLIP consistency checks, and then removing their influence via Kullback--Leibler divergence-based forgetting. This approach shifts the learning paradigm from passive error avoidance to active error correction. Experiments on synthetic and real-world noisy datasets, including CIFAR-10/100, CIFAR-N, WebVision, Clothing1M, and Red Mini-ImageNet, demonstrate state-of-the-art performance, particularly in high-noise regimes and under instance-dependent noise. The code is publicly available at this https URL.
>
---
#### [new 116] TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization
- **分类: cs.CV**

- **简介: 该论文提出TrianguLang，解决无标定的3D定位问题，通过几何感知语义注意力提升文本引导的分割与定位效率。**

- **链接: [https://arxiv.org/pdf/2603.08096](https://arxiv.org/pdf/2603.08096)**

> **作者:** Bryce Grant; Aryeh Rothenberg; Atri Banerjee; Peng Wang
>
> **摘要:** Localizing objects and parts from natural language in 3D space is essential for robotics, AR, and embodied AI, yet existing methods face a trade-off between the accuracy and geometric consistency of per-scene optimization and the efficiency of feed-forward inference. We present TrianguLang, a feed-forward framework for 3D localization that requires no camera calibration at inference. Unlike prior methods that treat views independently, we introduce Geometry-Aware Semantic Attention (GASA), which utilizes predicted geometry to gate cross-view feature correspondence, suppressing semantically plausible but geometrically inconsistent matches without requiring ground-truth poses. Validated on five benchmarks including ScanNet++ and uCO3D, TrianguLang achieves state-of-the-art feed-forward text-guided segmentation and localization, reducing user effort from $O(N)$ clicks to a single text query. The model processes each frame at 1008x1008 resolution in $\sim$57ms ($\sim$18 FPS) without optimization, enabling practical deployment for interactive robotics and AR applications. Code and checkpoints are available at this https URL.
>
---
#### [new 117] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于测试时适应（TTA）任务，旨在解决模型在分布偏移数据上的性能下降问题。通过调整Vision Transformer的奇异值并引入多样性损失，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.07926](https://arxiv.org/pdf/2603.07926)**

> **作者:** Sunghyun Baek; Jaemyung Yu; Seunghee Koh; Minsu Kim; Hyeonseong Jeon; Junmo Kim
>
> **备注:** ICLR 2026
>
> **摘要:** Test-time adaptation (TTA) has been widely explored to prevent performance degradation when test data differ from the training distribution. However, fully leveraging the rich representations of large pretrained models with minimal parameter updates remains underexplored. In this paper, we propose Intrinsic Mixture of Spectral Experts (IMSE) that leverages the spectral experts inherently embedded in Vision Transformers. We decompose each linear layer via singular value decomposition (SVD) and adapt only the singular values, while keeping the singular vectors fixed. We further identify a key limitation of entropy minimization in TTA: it often induces feature collapse, causing the model to rely on domain-specific features rather than class-discriminative features. To address this, we propose a diversity maximization loss based on expert-input alignment, which encourages diverse utilization of spectral experts during adaptation. In the continual test-time adaptation (CTTA) scenario, beyond preserving pretrained knowledge, it is crucial to retain and reuse knowledge from previously observed domains. We introduce Domain-Aware Spectral Code Retrieval, which estimates input distributions to detect domain shifts, and retrieves adapted singular values for rapid adaptation. Consequently, our method achieves state-of-the-art performance on various distribution-shift benchmarks under the TTA setting. In CTTA and Gradual CTTA, it further improves accuracy by 3.4 percentage points (pp) and 2.4 pp, respectively, while requiring 385 times fewer trainable parameters. Our code is available at this https URL.
>
---
#### [new 118] Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning
- **分类: cs.CV**

- **简介: 该论文属于微动作识别任务，解决低样本、噪声和跨被试条件下的识别难题。提出基于主动推理的框架，结合EFE引导采样和自适应学习，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07559](https://arxiv.org/pdf/2603.07559)**

> **作者:** Weijia Feng; Jingyu Yang; Ruojia Zhang; Fengtao Sun; Qian Gao; Chenyang Wang; Tongtong Su; Jia Guo; Xiaobai Li; Minglai Shao
>
> **备注:** 10 pages, accepted by CVPR 2026
>
> **摘要:** Micro-gestures are subtle and transient movements triggered by unconscious neural and emotional activities, holding great potential for human-computer interaction and clinical monitoring. However, their low amplitude, short duration, and strong inter-subject variability make existing deep models prone to degradation under low-sample, noisy, and cross-subject conditions. This paper presents an active inference-based framework for micro-gesture recognition, featuring Expected Free Energy (EFE)-guided temporal sampling and uncertainty-aware adaptive learning. The model actively selects the most discriminative temporal segments under EFE guidance, enabling dynamic observation and information gain maximization. Meanwhile, sample weighting driven by predictive uncertainty mitigates the effects of label noise and distribution shift. Experiments on the SMG dataset demonstrate the effectiveness of the proposed method, achieving consistent improvements across multiple mainstream backbones. Ablation studies confirm that both the EFE-guided observation and the adaptive learning mechanism are crucial to the performance gains. This work offers an interpretable and scalable paradigm for temporal behavior modeling under low-resource and noisy conditions, with broad applicability to wearable sensing, HCI, and clinical emotion monitoring.
>
---
#### [new 119] SketchGraphNet: A Memory-Efficient Hybrid Graph Transformer for Large-Scale Sketch Corpora Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SketchGraphNet，用于大规模草图识别任务，通过图结构建模解决传统方法依赖图像或笔画序列的问题。**

- **链接: [https://arxiv.org/pdf/2603.07521](https://arxiv.org/pdf/2603.07521)**

> **作者:** Shilong Chen; Mingyuan Li; Zhaoyang Wang; Zhonglin Ye; Haixing Zhao
>
> **摘要:** This work investigates large-scale sketch recognition from a graph-native perspective, where free-hand sketches are directly modeled as structured graphs rather than raster images or stroke sequences. We propose SketchGraphNet, a hybrid graph neural architecture that integrates local message passing with a memory-efficient global attention mechanism, without relying on auxiliary positional or structural encodings. To support systematic evaluation, we construct SketchGraph, a large-scale benchmark comprising 3.44 million graph-structured sketches across 344 categories, with two variants (A and R) to reflect different noise conditions. Each sketch is represented as a spatiotemporal graph with normalized stroke-order attributes. On SketchGraph-A and SketchGraph-R, SketchGraphNet achieves Top-1 accuracies of 83.62% and 87.61%, respectively, under a unified training configuration. MemEffAttn further reduces peak GPU memory by over 40% and training time by more than 30% compared with Performer-based global attention, while maintaining comparable accuracy.
>
---
#### [new 120] A Systematic Comparison of Training Objectives for Out-of-Distribution Detection in Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类中的分布外检测任务，旨在比较不同训练目标对OOD检测的影响。研究对比了四种损失函数在标准化协议下的表现。**

- **链接: [https://arxiv.org/pdf/2603.07571](https://arxiv.org/pdf/2603.07571)**

> **作者:** Furkan Genç; Onat Özdemir; Emre Akbaş
>
> **摘要:** Out-of-distribution (OOD) detection is critical in safety-sensitive applications. While this challenge has been addressed from various perspectives, the influence of training objectives on OOD behavior remains comparatively underexplored. In this paper, we present a systematic comparison of four widely used training objectives: Cross-Entropy Loss, Prototype Loss, Triplet Loss, and Average Precision (AP) Loss, spanning probabilistic, prototype-based, metric-learning, and ranking-based supervision, for OOD detection in image classification under standardized OpenOOD protocols. Across CIFAR-10/100 and ImageNet-200, we find that Cross-Entropy Loss, Prototype Loss, and AP Loss achieve comparable in-distribution accuracy, while Cross-Entropy Loss provides the most consistent near- and far-OOD performance overall; the other objectives can be competitive in specific settings.
>
---
#### [new 121] Evaluating Generative Models via One-Dimensional Code Distributions
- **分类: cs.CV**

- **简介: 该论文属于生成模型评估任务，旨在解决传统指标忽略感知质量的问题。通过离散视觉标记空间，提出CHD和CMMS两种新指标，提升与人类判断的相关性。**

- **链接: [https://arxiv.org/pdf/2603.08064](https://arxiv.org/pdf/2603.08064)**

> **作者:** Zexi Jia; Pengcheng Luo; Yijia Zhong; Jinchao Zhang; Jie Zhou
>
> **摘要:** Most evaluations of generative models rely on feature-distribution metrics such as FID, which operate on continuous recognition features that are explicitly trained to be invariant to appearance variations, and thus discard cues critical for perceptual quality. We instead evaluate models in the space of \emph{discrete} visual tokens, where modern 1D image tokenizers compactly encode both semantic and perceptual information and quality manifests as predictable token statistics. We introduce \emph{Codebook Histogram Distance} (CHD), a training-free distribution metric in token space, and \emph{Code Mixture Model Score} (CMMS), a no-reference quality metric learned from synthetic degradations of token sequences. To stress-test metrics under broad distribution shifts, we further propose \emph{VisForm}, a benchmark of 210K images spanning 62 visual forms and 12 generative models with expert annotations. Across AGIQA, HPDv2/3, and VisForm, our token-based metrics achieve state-of-the-art correlation with human judgments, and we will release all code and datasets to facilitate future research.
>
---
#### [new 122] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于数据蒸馏任务，旨在解决生成式数据蒸馏中视觉特征被文本主导的问题。通过早期视觉-语言融合，提升生成数据的语义忠实性和视觉一致性。**

- **链接: [https://arxiv.org/pdf/2603.07476](https://arxiv.org/pdf/2603.07476)**

> **作者:** Wenqi Cai; Yawen Zou; Guang Li; Chunzhi Gu; Chao Zhang
>
> **备注:** CVPR2026 (main conference)
>
> **摘要:** Dataset distillation (DD) aims to synthesize compact training sets that enable models to achieve high accuracy with significantly fewer samples. Recent diffusion-based DD methods commonly introduce semantic guidance through late-stage cross-attention, where textual prompts tend to dominate the generative process. Although this strategy enforces label relevance, it diminishes the contribution of visual latents, resulting in over-corrected samples that mirror prompt patterns rather than reflecting intrinsic visual features. To solve this problem, we introduce an Early Vision-Language Fusion (EVLF) method that aligns textual and visual embeddings at the transition between the encoder and the generative backbone. By incorporating a lightweight cross-attention module at this transition, the early representations simultaneously encode local textures and global semantic directions across the denoising process. Importantly, EVLF is plug-and-play and can be easily integrated into any diffusion-based dataset distillation pipeline with an encoder. It works across different denoiser architectures and sampling schedules without any task-specific modifications. Extensive experiments demonstrate that EVLF generates semantically faithful and visually coherent synthetic data, yielding consistent improvements in downstream classification accuracy across varied settings. Source code is available at this https URL.
>
---
#### [new 123] Single Image Super-Resolution via Bivariate `A Trous Wavelet Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决扩散模型生成高频细节不一致的问题。提出BATDiff模型，利用双变量小波变换实现跨尺度结构引导，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.07234](https://arxiv.org/pdf/2603.07234)**

> **作者:** Heidari Maryam; Anantrasirichai Nantheera; Achim Alin
>
> **备注:** 17 pages
>
> **摘要:** The effectiveness of super resolution (SR) models hinges on their ability to recover high frequency structure without introducing artifacts. Diffusion based approaches have recently advanced the state of the art in SR. However, most diffusion based SR pipelines operate purely in the spatial domain, which may yield high frequency details that are not well supported by the underlying low resolution evidence. On the other hand, unlike supervised SR models that may inject dataset specific textures, single image SR relies primarily on internal image statistics and can therefore be less prone to dataset-driven hallucinations; nevertheless, ambiguity in the LR observation can still lead to inconsistent high frequency details. To tackle this problem, we introduce BATDiff, an unsupervised Bivariate A trous Wavelet Diffusion model designed to provide structured cross scale guidance during the generative process. BATDiff employs an a Trous wavelet transform that constructs an undecimated multiscale representation in which high frequency components are progressively revealed while the full spatial resolution is preserved. As the core inference mechanism, BATDiff includes a bivariate cross scale module that models parent child dependencies between adjacent scales. It improves high frequency coherence and reduces mismatch artifacts in diffusion based SR. Experiments on standard benchmarks demonstrate that BATDiff produces sharper and more structurally consistent reconstructions than existing diffusion and non diffusion baselines, achieving improvements in fidelity and perceptual quality.
>
---
#### [new 124] LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于卫星遥感任务，解决几何匹配问题。针对预计算嵌入与用户区域不匹配的问题，提出LEPA模型，通过几何增强预测变换后的嵌入，提升匹配精度。**

- **链接: [https://arxiv.org/pdf/2603.07246](https://arxiv.org/pdf/2603.07246)**

> **作者:** Erik Scheurer; Rocco Sedona; Stefan Kesselheim; Gabriele Cavallaro
>
> **摘要:** Geospatial foundation models provide precomputed embeddings that serve as compact feature vectors for large-scale satellite remote sensing data. While these embeddings can reduce data-transfer bottlenecks and computational costs, Earth observation (EO) applications can still face geometric mismatches between user-defined areas of interest and the fixed precomputed embedding grid. Standard latent-space interpolation is unreliable in this setting because the embedding manifold is highly non-convex, yielding representations that do not correspond to realistic inputs. We verify this using Prithvi-EO-2.0 to understand the shortcomings of interpolation applied to patch embeddings. As a substitute, we propose a Learned Equivariance-Predicting Architecture (LEPA). Instead of averaging vectors, LEPA conditions a predictor on geometric augmentations to directly predict the transformed embedding. We evaluate LEPA on NASA/USGS Harmonized Landsat-Sentinel (HLS) imagery and ImageNet-1k. Experiments show that standard interpolation achieves a mean reciprocal rank (MRR) below 0.2, whereas LEPA increases MRR to over 0.8, enabling accurate geometric adjustment without re-encoding.
>
---
#### [new 125] SIQA: Toward Reliable Scientific Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于科学图像质量评估任务，旨在解决传统IQA无法准确评价科学图像科学性的问题。提出SIQA框架，从知识和感知两方面评估图像质量，并设计相关评测协议。**

- **链接: [https://arxiv.org/pdf/2603.06700](https://arxiv.org/pdf/2603.06700)**

> **作者:** Wenzhe Li; Liang Chen; Junying Wang; Yijing Guo; Ye Shen; Farong Wen; Chunyi Li; Zicheng Zhang; Guangtao Zhai
>
> **摘要:** Scientific images fundamentally differ from natural and AI-generated images in that they encode structured domain knowledge rather than merely depict visual scenes. Assessing their quality therefore requires evaluating not only perceptual fidelity but also scientific correctness and logical completeness. However, existing image quality assessment (IQA) paradigms primarily focus on perceptual distortions or image-text alignment, implicitly assuming that depicted content is factually valid. This assumption breaks down in scientific contexts, where visually plausible figures may still contain conceptual errors or incomplete reasoning. To address this gap, we introduce Scientific Image Quality Assessment (SIQA), a framework that models scientific image quality along two complementary dimensions: Knowledge (Scientific Validity and Scientific Completeness) and Perception (Cognitive Clarity and Disciplinary Conformity). To operationalize this formulation, we design two evaluation protocols: SIQA-U (Understanding), which measures semantic comprehension of scientific content through multiple-choice tasks, and SIQA-S (Scoring), which evaluates alignment with expert quality judgments. We further construct the SIQA Challenge, consisting of an expert-annotated benchmark and a large-scale training set. Experiments across representative multimodal large language models (MLLMs) reveal a consistent discrepancy between scoring alignment and scientific understanding. While models can achieve strong agreement with expert ratings under SIQA-S, their performance on SIQA-U remains substantially lower. Fine-tuning improves both metrics, yet gains in scoring consistently outpace improvements in understanding. These results suggest that rating consistency alone may not reliably reflect scientific comprehension, underscoring the necessity of multidimensional evaluation for scientific image quality assessment.
>
---
#### [new 126] Scale Space Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Scale Space Diffusion，将尺度空间理论与扩散模型结合，解决图像生成中高分辨率处理效率低的问题，通过下采样实现更高效的去噪。**

- **链接: [https://arxiv.org/pdf/2603.08709](https://arxiv.org/pdf/2603.08709)**

> **作者:** Soumik Mukhopadhyay; Prateksha Udhayanan; Abhinav Shrivastava
>
> **备注:** Project website: this https URL . The first two authors contributed equally
>
> **摘要:** Diffusion models degrade images through noise, and reversing this process reveals an information hierarchy across timesteps. Scale-space theory exhibits a similar hierarchy via low-pass filtering. We formalize this connection and show that highly noisy diffusion states contain no more information than small, downsampled images - raising the question of why they must be processed at full resolution. To address this, we fuse scale spaces into the diffusion process by formulating a family of diffusion models with generalized linear degradations and practical implementations. Using downsampling as the degradation yields our proposed Scale Space Diffusion. To support Scale Space Diffusion, we introduce Flexi-UNet, a UNet variant that performs resolution-preserving and resolution-increasing denoising using only the necessary parts of the network. We evaluate our framework on CelebA and ImageNet and analyze its scaling behavior across resolutions and network depths. Our project website ( this https URL ) is available publicly.
>
---
#### [new 127] XMACNet: An Explainable Lightweight Attention based CNN with Multi Modal Fusion for Chili Disease Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于植物病害分类任务，旨在通过图像识别检测辣椒病害。提出XMACNet模型，结合注意力机制和多模态融合，提升分类性能并实现可解释性。**

- **链接: [https://arxiv.org/pdf/2603.06750](https://arxiv.org/pdf/2603.06750)**

> **作者:** Tapon Kumer Ray; Rajkumar Y; Shalini R; Srigayathri K; Jayashree S; Lokeswari P
>
> **备注:** 14 pages, 8 figures, Conference Paper
>
> **摘要:** Plant disease classification via imaging is a critical task in precision agriculture. We propose XMACNet, a novel light-weight Convolutional Neural Network (CNN) that integrates self-attention and multi-modal fusion of visible imagery and vegetation indices for chili disease detection. XMACNet uses an EfficientNetV2S backbone enhanced by a self-attention module and a fusion branch that processes both RGB images and computed vegetation index maps (NDVI, NPCI, MCARI). We curated a new dataset of 12,000 chili leaf images across six classes (five disease types plus healthy), augmented synthetically via StyleGAN to mitigate data scarcity. Trained on this dataset, XMACNet achieves high accuracy, F1-score, and AUC, outperforming baseline models such as ResNet-50, MobileNetV2, and a Swin Transformer variant. Crucially, XMACNet is explainable: we use Grad-CAM++ and SHAP to visualize and quantify the models focus on disease features. The models compact size and fast inference make it suitable for edge deployment in real-world farming scenarios.
>
---
#### [new 128] FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT
- **分类: cs.CV**

- **简介: 该论文提出FrameVGGT，解决流式3D感知中KV缓存无限增长的问题。通过帧级记忆管理，在有限内存下保持稳定几何推理。**

- **链接: [https://arxiv.org/pdf/2603.07690](https://arxiv.org/pdf/2603.07690)**

> **作者:** Zhisong Xu; Takeshi Oishi
>
> **备注:** 24pages including appendix
>
> **摘要:** Streaming Visual Geometry Transformers such as StreamVGGT enable strong online 3D perception but suffer from unbounded KV-cache growth, which limits deployment over long streams. We revisit bounded-memory streaming from the perspective of geometric support. In geometry-driven reasoning, memory quality depends not only on how many tokens are retained, but also on whether the retained memory still preserves sufficiently coherent local support. This suggests that token-level retention may become less suitable under fixed budgets, as it can thin the evidence available within each contributing frame and make subsequent fusion more sensitive to weakly aligned history. Motivated by this observation, we propose FrameVGGT, a frame-driven rolling explicit-memory framework that treats each frame's incremental KV contribution as a coherent evidence block. FrameVGGT summarizes each block into a compact prototype and maintains a fixed-capacity mid-term bank of complementary frame blocks under strict budgets, with an optional lightweight anchor tier for rare prolonged degradation. Across long-sequence 3D reconstruction, video depth estimation, and camera pose benchmarks, FrameVGGT achieves favorable accuracy--memory trade-offs under bounded memory, while maintaining more stable geometry over long streams.
>
---
#### [new 129] A Hybrid Vision Transformer Approach for Mathematical Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于数学表达式识别任务，旨在解决二维结构和符号大小差异带来的识别难题。提出混合视觉Transformer模型，结合2D位置编码和覆盖注意力机制，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.07929](https://arxiv.org/pdf/2603.07929)**

> **作者:** Anh Duy Le; Van Linh Pham; Vinh Loi Ly; Nam Quan Nguyen; Huu Thang Nguyen; Tuan Anh Tran
>
> **备注:** Accepted as oral presentation at DICTA 2022
>
> **摘要:** One of the crucial challenges taken in document analysis is mathematical expression recognition. Unlike text recognition which only focuses on one-dimensional structure images, mathematical expression recognition is a much more complicated problem because of its two-dimensional structure and different symbol size. In this paper, we propose using a Hybrid Vision Transformer (HVT) with 2D positional encoding as the encoder to extract the complex relationship between symbols from the image. A coverage attention decoder is used to better track attention's history to handle the under-parsing and over-parsing problems. We also showed the benefit of using the [CLS] token of ViT as the initial embedding of the decoder. Experiments performed on the IM2LATEX-100K dataset have shown the effectiveness of our method by achieving a BLEU score of 89.94 and outperforming current state-of-the-art methods.
>
---
#### [new 130] PaLMR: Towards Faithful Visual Reasoning via Multimodal Process Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态推理任务，旨在解决模型在推理过程中忽视视觉证据的问题。提出PaLMR框架，通过对齐推理过程与视觉信息，提升模型的视觉忠实性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.06652](https://arxiv.org/pdf/2603.06652)**

> **作者:** Yantao Li; Qiang Hui; Chenyang Yan; Kanzhi Cheng; Fang Zhao; Chao Tan; Huanling Gao; Jianbing Zhang; Kai Wang; Xinyu Dai; Shiguo Lian
>
> **摘要:** Reinforcement learning has recently improved the reasoning ability of Large Language Models and Multimodal LLMs, yet prevailing reward designs emphasise final-answer correctness and consequently tolerate process hallucinations--cases where models reach the right answer while misperceiving visual evidence. We address this process-level misalignment with PaLMR, a framework that aligns not only outcomes but also the reasoning process itself. PaLMR comprises two complementary components: a perception-aligned data layer that constructs process-aware reasoning data with structured pseudo-ground-truths and verifiable visual facts, and a process-aligned optimisation layer that constructs a hierarchical reward fusion scheme with a process-aware scoring function to encourage visually faithful chains-of-thought and improve training stability. Experiments on Qwen2.5-VL-7B show that our approach substantially reduces reasoning hallucinations and improves visual reasoning fidelity, achieving state-of-the-art results on HallusionBench while maintaining strong performance on MMMU, MathVista, and MathVerse. These findings indicate that PaLMR offers a principled and practical route to process-aligned multimodal reasoning, advancing the reliability and interpretability of MLLMs.
>
---
#### [new 131] The Model Knows Which Tokens Matter: Automatic Token Selection via Noise Gating
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决视觉令牌冗余问题。通过自动选择关键令牌，提升推理效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2603.07135](https://arxiv.org/pdf/2603.07135)**

> **作者:** Landi He; Xiaoyu Yang; Lijian Xu
>
> **摘要:** Visual tokens dominate inference cost in vision-language models (VLMs), yet many carry redundant information. Existing pruning methods alleviate this but typically rely on attention magnitude or similarity scores. We reformulate visual token pruning as capacity constrained communication: given a fixed budget K, the model must allocate limited bandwidth to maximally preserve visual information. We propose AutoSelect, which attaches a lightweight Scorer and Denoiser to a frozen VLM and trains with only the standard next token prediction loss, without auxiliary objectives or extra annotations. During training, a variance preserving noise gate modulates each token's information flow according to its predicted importance so that gradients propagate through all tokens; a diagonal attention Denoiser then recovers the perturbed representations. At inference, only the Scorer and a hard top-K selection remain, adding negligible latency. On ten VLM benchmarks, AutoSelect retains 96.5% of full model accuracy while accelerating LLM prefill by 2.85x with only 0.69 ms overhead, and transfers to different VLM backbones without architecture-specific tuning. Code is available at this https URL.
>
---
#### [new 132] 3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决3DGS中因动态遮挡导致的性能下降问题。提出3DGS-HPC框架，结合局部一致性与混合分类策略，提升去噪效果。**

- **链接: [https://arxiv.org/pdf/2603.07587](https://arxiv.org/pdf/2603.07587)**

> **作者:** Jiahao Chen; Yipeng Qin; Ganlong Zhao; Xin Li; Wenping Wang; Guanbin Li
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and 3D scene reconstruction, yet its quality often degrades in real-world environments due to transient distractors, such as moving objects and varying shadows. Existing methods commonly rely on semantic cues extracted from pre-trained vision models to identify and suppress these distractors, but such semantics are misaligned with the binary distinction between static and transient regions and remain fragile under the appearance perturbations introduced during 3DGS optimization. We propose 3DGS-HPC, a framework that circumvents these limitations by combining two complementary principles: a patch-wise classification strategy that leverages local spatial consistency for robust region-level decisions, and a hybrid classification metric that adaptively integrates photometric and perceptual cues for more reliable separation. Extensive experiments demonstrate the superiority and robustness of our method in mitigating distractors to improve 3DGS-based novel view synthesis.
>
---
#### [new 133] PureCC: Pure Learning for Text-to-Image Concept Customization
- **分类: cs.CV**

- **简介: 该论文属于文本到图像的概念定制任务，旨在解决现有方法忽视模型原有行为的问题。提出PureCC，通过解耦学习目标和双分支训练，实现高质量定制同时保持模型能力。**

- **链接: [https://arxiv.org/pdf/2603.07561](https://arxiv.org/pdf/2603.07561)**

> **作者:** Zhichao Liao; Xiaole Xian; Qingyu Li; Wenyu Qin; Meng Wang; Weicheng Xie; Siyang Song; Pingfa Feng; Long Zeng; Liang Pan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Existing concept customization methods have achieved remarkable outcomes in high-fidelity and multi-concept customization. However, they often neglect the influence on the original model's behavior and capabilities when learning new personalized concepts. To address this issue, we propose PureCC. PureCC introduces a novel decoupled learning objective for concept customization, which combines the implicit guidance of the target concept with the original conditional prediction. This separated form enables PureCC to substantially focus on the original model during training. Moreover, based on this objective, PureCC designs a dual-branch training pipeline that includes a frozen extractor providing purified target concept representations as implicit guidance and a trainable flow model producing the original conditional prediction, jointly achieving pure learning for personalized concepts. Furthermore, PureCC introduces a novel adaptive guidance scale $\lambda^\star$ to dynamically adjust the guidance strength of the target concept, balancing customization fidelity and model preservation. Extensive experiments show that PureCC achieves state-of-the-art performance in preserving the original behavior and capabilities while enabling high-fidelity concept customization. The code is available at this https URL.
>
---
#### [new 134] Solution to the 10th ABAW Expression Recognition Challenge: A Robust Multimodal Framework with Safe Cross-Attention and Modality Dropout
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感识别任务，解决真实环境中部分遮挡、模态缺失和类别不平衡问题。提出多模态框架，结合安全交叉注意力和模态丢弃策略，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08034](https://arxiv.org/pdf/2603.08034)**

> **作者:** Jun Yu; Naixiang Zheng; Guoyuan Wang; Yunxiang Zhang; Lingsi Zhu; Jiaen Liang; Wei Huang; Shengping Liu
>
> **摘要:** Emotion recognition in real-world environments is hindered by partial occlusions, missing modalities, and severe class imbalance. To address these issues, particularly for the Affective Behavior Analysis in-the-wild (ABAW) Expression challenge, we propose a multimodal framework that dynamically fuses visual and audio representations. Our approach uses a dual-branch Transformer architecture featuring a safe cross-attention mechanism and a modality dropout strategy. This design allows the network to rely on audio-based predictions when visual cues are absent. To mitigate the long-tail distribution of the Aff-Wild2 dataset, we apply focal loss optimization, combined with a sliding-window soft voting strategy to capture dynamic emotional transitions and reduce frame-level classification jitter. Experiments demonstrate that our framework effectively handles missing modalities and complex spatiotemporal dependencies, achieving an accuracy of 60.79% and an F1-score of 0.5029 on the Aff-Wild2 validation set.
>
---
#### [new 135] Not Like Transformers: Drop the Beat Representation for Dance Generation with Mamba-Based Diffusion Model
- **分类: cs.CV; cs.AI; cs.GR; cs.SD**

- **简介: 该论文属于舞蹈生成任务，旨在解决现有方法难以捕捉舞蹈的序列性、节奏性和音乐同步性问题。提出MambaDance模型，利用Mamba和高斯节拍表示提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.08023](https://arxiv.org/pdf/2603.08023)**

> **作者:** Sangjune Park; Inhyeok Choi; Donghyeon Soon; Youngwoo Jeon; Kyungdon Joo
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Dance is a form of human motion characterized by emotional expression and communication, playing a role in various fields such as music, virtual reality, and content creation. Existing methods for dance generation often fail to adequately capture the inherently sequential, rhythmical, and music-synchronized characteristics of dance. In this paper, we propose \emph{MambaDance}, a new dance generation approach that leverages a Mamba-based diffusion model. Mamba, well-suited to handling long and autoregressive sequences, is integrated into our two-stage diffusion architecture, substituting off-the-shelf Transformer. Additionally, considering the critical role of musical beats in dance choreography, we propose a Gaussian-based beat representation to explicitly guide the decoding of dance sequences. Experiments on AIST++ and FineDance datasets for each sequence length show that our proposed method effectively generates plausible dance movements while reflecting essential characteristics, consistently from short to long dances, compared to the previous methods. Additional qualitative results and demo videos are available at \small{this https URL}.
>
---
#### [new 136] MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals
- **分类: cs.CV**

- **简介: 该论文属于电磁信号与文本的多模态任务，旨在解决数据少、评估基准缺失及低信噪比环境下的模型鲁棒性问题。工作包括构建数据集、设计评估基准和提出增强鲁棒性的训练框架MERLIN。**

- **链接: [https://arxiv.org/pdf/2603.08174](https://arxiv.org/pdf/2603.08174)**

> **作者:** Junyu Shen; Zhendong She; Chenghanyu Zhang; Yuchuang Sun; Luqing Luo; Dingwei Tan; Zonghao Guo; Bo Guo; Zehua Han; Wupeng Xie; Yaxin Mu; Peng Zhang; Peipei Li; Fengxiang Wang; Yangang Sun; Maosong Sun
>
> **摘要:** The paradigm of Multimodal Large Language Models (MLLMs) offers a promising blueprint for advancing the electromagnetic (EM) domain. However, prevailing approaches often deviate from the native MLLM paradigm, instead using task-specific or pipelined architectures that lead to fundamental limitations in model performance and generalization. Fully realizing the MLLM potential in EM domain requires overcoming three main challenges: (1) Data. The scarcity of high-quality datasets with paired EM signals and descriptive text annotations used for MLLMs pre-training; (2) Benchmark. The absence of comprehensive benchmarks to systematically evaluate and compare the performance of models on EM signal-to-text tasks; (3) Model. A critical fragility in low Signal-to-Noise Ratio (SNR) environments, where critical signal features can be obscured, leading to significant performance degradation. To address these challenges, we introduce a tripartite contribution to establish a foundation for MLLMs in the EM domain. First, to overcome data scarcity, we construct and release EM-100k, a large-scale dataset comprising over 100,000 EM signal-text pairs. Second, to enable rigorous and standardized evaluation, we propose EM-Bench, the most comprehensive benchmark featuring diverse downstream tasks spanning from perception to reasoning. Finally, to tackle the core modeling challenge, we present MERLIN, a novel training framework designed not only to align low-level signal representations with high-level semantic text, but also to explicitly enhance model robustness and performance in challenging low-SNR environments. Comprehensive experiments validate our method, showing that MERLIN is state-of-the-art in the EM-Bench and exhibits remarkable robustness in low-SNR settings.
>
---
#### [new 137] Fast Low-light Enhancement and Deblurring for 3D Dark Scenes
- **分类: cs.CV**

- **简介: 该论文属于低光场景的3D重建任务，解决低光、噪声和运动模糊问题。提出FLED-GS框架，通过交替增强与重建实现快速去模糊和增强。**

- **链接: [https://arxiv.org/pdf/2603.08133](https://arxiv.org/pdf/2603.08133)**

> **作者:** Feng Zhang; Jinglong Wang; Ze Li; Yanghong Zhou; Yang Chen; Lei Chen; Xiatian Zhu
>
> **备注:** 5 pages, 2 figures, Accepted at ICASSP 2026
>
> **摘要:** Novel view synthesis from low-light, noisy, and motion-blurred imagery remains a valuable and challenging task. Current volumetric rendering methods struggle with compound degradation, and sequential 2D preprocessing introduces artifacts due to interdependencies. In this work, we introduce FLED-GS, a fast low-light enhancement and deblurring framework that reformulates 3D scene restoration as an alternating cycle of enhancement and reconstruction. Specifically, FLED-GS inserts several intermediate brightness anchors to enable progressive recovery, preventing noise blow-up from harming deblurring or geometry. Each iteration sharpens inputs with an off-the-shelf 2D deblurrer and then performs noise-aware 3DGS reconstruction that estimates and suppresses noise while producing clean priors for the next level. Experiments show FLED-GS outperforms state-of-the-art LuSh-NeRF, achieving 21$\times$ faster training and 11$\times$ faster rendering.
>
---
#### [new 138] A Parameter-efficient Convolutional Approach for Weed Detection in Multispectral Aerial Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于农业图像分割任务，旨在提高杂草检测的效率与精度。提出FCBNet模型，采用冻结骨干网络和轻量解码器，提升性能并减少参数量。**

- **链接: [https://arxiv.org/pdf/2603.06655](https://arxiv.org/pdf/2603.06655)**

> **作者:** Leo Thomas Ramos; Angel D. Sappa
>
> **备注:** 10 pages, 6 figures, 9 tables
>
> **摘要:** We introduce FCBNet, an efficient model designed for weed segmentation. The architecture is based on a fully frozen ConvNeXt backbone, the proposed Feature Correction Block (FCB), which leverages efficient convolutions for feature refinement, and a lightweight decoder. FCBNet is evaluated on the WeedBananaCOD and WeedMap datasets under both RGB and multispectral modalities, showing that FCBNet outperforms models such as U-Net, DeepLabV3+, SK-U-Net, SegFormer, and WeedSense in terms of mIoU, exceeding 85%, while also achieving superior computational efficiency, requiring only 0.06 to 0.2 hours for training. Furthermore, the frozen backbone strategy reduces the number of trainable parameters by more than 90%, significantly lowering memory requirements.
>
---
#### [new 139] Alignment-Aware and Reliability-Gated Multimodal Fusion for Unmanned Aerial Vehicle Detection Across Heterogeneous Thermal-Visual Sensors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态目标检测任务，旨在解决异构热成像与可见光传感器融合中的对齐和可靠性问题。提出两种融合策略，提升无人机检测性能。**

- **链接: [https://arxiv.org/pdf/2603.08208](https://arxiv.org/pdf/2603.08208)**

> **作者:** Ishrat Jahan; Molla E Majid; M Murugappan; Muhammad E. H. Chowdhury; N.B.Prakash; Saad Bin Abul Kashem; Balamurugan Balusamy; Amith Khandakar
>
> **摘要:** Reliable unmanned aerial vehicle (UAV) detection is critical for autonomous airspace monitoring but remains challenging when integrating sensor streams that differ substantially in resolution, perspective, and field of view. Conventional fusion methods-such as wavelet-, Laplacian-, and decision-level approaches-often fail to preserve spatial correspondence across modalities and suffer from annotation of inconsistencies, limiting their robustness in real-world settings. This study introduces two fusion strategies, Registration-aware Guided Image Fusion (RGIF) and Reliability-Gated Modality-Attention Fusion (RGMAF), designed to overcome these limitations. RGIF employs Enhanced Correlation Coefficient (ECC)-based affine registration combined with guided filtering to maintain thermal saliency while enhancing structural detail. RGMAF integrates affine and optical-flow registration with a reliability-weighted attention mechanism that adaptively balances thermal contrast and visual sharpness. Experiments were conducted on the Multi-Sensor and Multi-View Fixed-Wing (MMFW)-UAV dataset comprising 147,417 annotated air-to-air frames collected from infrared, wide-angle, and zoom sensors. Among single-modality detectors, YOLOv10x demonstrated the most stable cross-domain performance and was selected as the detection backbone for evaluating fused imagery. RGIF improved the visual baseline by 2.13% mAP@50 (achieving 97.65%), while RGMAF attained the highest recall of 98.64%. These findings show that registration-aware and reliability-adaptive fusion provides a robust framework for integrating heterogeneous modalities, substantially enhancing UAV detection performance in multimodal environments.
>
---
#### [new 140] Transferable Optimization Network for Cross-Domain Image Reconstruction
- **分类: cs.CV; cs.LG; math.OC**

- **简介: 该论文属于图像重建任务，解决数据不足的问题。提出一种迁移学习框架，通过通用特征提取器和任务适配器，实现跨域高质量图像重建。**

- **链接: [https://arxiv.org/pdf/2603.07831](https://arxiv.org/pdf/2603.07831)**

> **作者:** Yunmei Chen; Chi Ding; Xiaojing Ye
>
> **备注:** 30 pages, 7 figures
>
> **摘要:** We develop a novel transfer learning framework to tackle the challenge of limited training data in image reconstruction problems. The proposed framework consists of two training steps, both of which are formed as bi-level optimizations. In the first step, we train a powerful universal feature-extractor that is capable of learning important knowledge from large, heterogeneous data sets in various domains. In the second step, we train a task-specific domain-adapter for a new target domain or task with only a limited amount of data available for training. Then the composition of the adapter and the universal feature-extractor effectively explores feature which serve as an important component of image regularization for the new domains, and this leads to high-quality reconstruction despite the data limitation issue. We apply this framework to reconstruct under-sampled MR images with limited data by using a collection of diverse data samples from different domains, such as images of other anatomies, measurements of various sampling ratios, and even different image modalities, including natural images. Experimental results demonstrate a promising transfer learning capability of the proposed method.
>
---
#### [new 141] AULLM++: Structural Reasoning with Large Language Models for Micro-Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于微表情识别任务，解决传统方法在特征表达、细粒度建模和AU相关性建模上的不足。提出AULLM++框架，结合视觉与语言模型进行结构化推理，提升识别性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08387](https://arxiv.org/pdf/2603.08387)**

> **作者:** Zhishu Liu; Kaishen Yuan; Bo Zhao; Hui Ma; Zitong Yu
>
> **摘要:** Micro-expression Action Unit (AU) detection identifies localized AUs from subtle facial muscle activations, providing a foundation for decoding affective cues. Previous methods face three key limitations: (1) heavy reliance on low-density visual information, rendering discriminative evidence vulnerable to background noise; (2) coarse-grained feature processing that misaligns with the demand for fine-grained representations; and (3) neglect of inter-AU correlations, restricting the parsing of complex expression patterns. We propose AULLM++, a reasoning-oriented framework leveraging Large Language Models (LLMs), which injects visual features into textual prompts as actionable semantic premises to guide inference. It formulates AU prediction into three stages: evidence construction, structure modeling, and deduction-based prediction. Specifically, a Multi-Granularity Evidence-Enhanced Fusion Projector (MGE-EFP) fuses mid-level texture cues with high-level semantics, distilling them into a compact Content Token (CT). Furthermore, inspired by micro- and macro-expression AU correspondence, we encode AU relationships as a sparse structural prior and learn interaction strengths via a Relation-Aware AU Graph Neural Network (R-AUGNN), producing an Instruction Token (IT). We then fuse CT and IT into a structured textual prompt and introduce Counterfactual Consistency Regularization (CCR) to construct counterfactual samples, enhancing the model's generalization. Extensive experiments demonstrate AULLM++ achieves state-of-the-art performance on standard benchmarks and exhibits superior cross-domain generalization.
>
---
#### [new 142] ASMIL: Attention-Stabilized Multiple Instance Learning for Whole Slide Imaging
- **分类: cs.CV**

- **简介: 该论文提出ASMIL，解决WSI诊断中注意力不稳定问题，通过稳定注意力机制提升MIL性能。**

- **链接: [https://arxiv.org/pdf/2603.06658](https://arxiv.org/pdf/2603.06658)**

> **作者:** Linfeng Ye; Shayan Mohajer Hamidi; Zhixiang Chi; Guang Li; Mert Pilanci; Takahiro Ogawa; Miki Haseyama; Konstantinos N. Plataniotis
>
> **备注:** 39 pages, 26 figures
>
> **摘要:** Attention-based multiple instance learning (MIL) has emerged as a powerful framework for whole slide image (WSI) diagnosis, leveraging attention to aggregate instance-level features into bag-level predictions. Despite this success, we find that such methods exhibit a new failure mode: unstable attention dynamics. Across four representative attention-based MIL methods and two public WSI datasets, we observe that attention distributions oscillate across epochs rather than converging to a consistent pattern, degrading performance. This instability adds to two previously reported challenges: overfitting and over-concentrated attention distribution. To simultaneously overcome these three limitations, we introduce attention-stabilized multiple instance learning (ASMIL), a novel unified framework. ASMIL uses an anchor model to stabilize attention, replaces softmax with a normalized sigmoid function in the anchor to prevent over-concentration, and applies token random dropping to mitigate overfitting. Extensive experiments demonstrate that ASMIL achieves up to a 6.49\% F1 score improvement over state-of-the-art methods. Moreover, integrating the anchor model and normalized sigmoid into existing attention-based MIL methods consistently boosts their performance, with F1 score gains up to 10.73\%. All code and data are publicly available at this https URL.
>
---
#### [new 143] calibfusion: Transformer-Based Differentiable Calibration for Radar-Camera Fusion Detection in Water-Surface Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多传感器融合检测任务，旨在解决水面上雷达与相机标定不准确导致的检测性能下降问题。提出CalibFusion方法，通过Transformer实现端到端标定优化。**

- **链接: [https://arxiv.org/pdf/2603.06670](https://arxiv.org/pdf/2603.06670)**

> **作者:** Yuting Wan; Liguo Sun; Jiuwu Hao; Pin LV
>
> **摘要:** Millimeter-wave (mmWave) Radar--Camera fusion improves perception under adverse illumination and weather, but its performance is sensitive to Radar--Camera extrinsic calibration: residual misalignment biases Radar-to-image projection and degrades cross-modal aggregation for downstream 2D detection. Existing calibration and auto-calibration methods are mainly developed for road and urban scenes with abundant structures and object constraints, whereas water-surface environments feature large textureless regions, sparse and intermittent targets, and wave-/specular-induced Radar clutter, which weakens explicit object-centric matching. We propose CalibFusion, a calibration-conditioned Radar--Camera fusion detector that learns implicit extrinsic refinement end-to-end with the detection objective. CalibFusion builds a multi-frame persistence-aware Radar density representation with intensity weighting and Doppler-guided suppression of fast-varying clutter. A cross-modal transformer interaction module predicts a confidence-gated refinement of the initial extrinsics, which is integrated through a differentiable projection-and-splatting operator to generate calibration-conditioned image-plane Radar features. Experiments on WaterScenes and FLOW show improved fusion-based 2D detection and robustness under synthetic miscalibration, supported by sensitivity analyses and qualitative Radar-to-image overlays. Results on nuScenes indicate that the refinement mechanism transfers beyond water-surface scenarios.
>
---
#### [new 144] PDD: Manifold-Prior Diverse Distillation for Medical Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像异常检测任务，针对医学数据中细微且多样的异常难以检测的问题，提出PDD框架，通过双教师知识蒸馏提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.07142](https://arxiv.org/pdf/2603.07142)**

> **作者:** Xijun Lu; Hongying Liu; Fanhua Shang; Yanming Hui; Liang Wan
>
> **备注:** Accepted by CVPR'2026
>
> **摘要:** Medical image anomaly detection faces unique challenges due to subtle, heterogeneous anomalies embedded in complex anatomical structures. Through systematic Grad-CAM analysis, we reveal that discriminative activation maps fail on medical data, unlike their success on industrial datasets, motivating the need for manifold-level modeling. We propose PDD (Manifold-Prior Diverse Distillation), a framework that unifies dual-teacher priors into a shared high-dimensional manifold and distills this knowledge into dual students with complementary behaviors. Specifically, frozen VMamba-Tiny and wide-ResNet50 encoders provide global contextual and local structural priors, respectively. Their features are unified through a Manifold Matching and Unification (MMU) module, while an Inter-Level Feature Adaption (InA) module enriches intermediate representations. The unified manifold is distilled into two students: one performs layer-wise distillation via InA for local consistency, while the other receives skip-projected representations through a Manifold Prior Affine (MPA) module to capture cross-layer dependencies. A diversity loss prevents representation collapse while maintaining detection sensitivity. Extensive experiments on multiple medical datasets demonstrate that PDD significantly outperforms existing state-of-the-art methods, achieving improvements of up to 11.8%, 5.1%, and 8.5% in AUROC on HeadCT, BrainMRI, and ZhangLab datasets, respectively, and 3.4% in F1 max on the Uni-Medical dataset, establishing new state-of-the-art performance in medical image anomaly detection. The implementation will be released at this https URL
>
---
#### [new 145] GLASS: Graph and Vision-Language Assisted Semantic Shape Correspondence
- **分类: cs.CV**

- **简介: 该论文属于3D形状对应任务，解决非等距变形和跨类形状的密集对应问题。通过结合视觉语言模型和图结构，提出GLASS框架提升对应精度。**

- **链接: [https://arxiv.org/pdf/2603.07652](https://arxiv.org/pdf/2603.07652)**

> **作者:** Qinfeng Xiao; Guofeng Mei; Qilong Liu; Chenyuan Yi; Fabio Poiesi; Jian Zhang; Bo Yang; Yick Kit-lun
>
> **摘要:** Establishing dense correspondence across 3D shapes is crucial for fundamental downstream tasks, including texture transfer, shape interpolation, and robotic manipulation. However, learning these mappings without manual supervision remains a formidable challenge, particularly under severe non-isometric deformations and in inter-class settings where geometric cues are ambiguous. Conventional functional map methods, while elegant, typically struggle in these regimes due to their reliance on isometry. To address this, we present GLASS, a framework that bridges the gap by integrating geometric spectral analysis with rich semantic priors from vision-language foundation models. GLASS introduces three key innovations: (i) a view-consistent strategy that enables robust multi-view visual feature extraction from powerful vision foundation models; (ii) the injection of language embeddings into vertex descriptors via zero-shot 3D segmentation, capturing high-level part semantics; and (iii) a graph-assisted contrastive loss that enforces structural consistency between regions (e.g., source's head'' $\leftrightarrow$ target's head'') by leveraging geodesic and topological relationships between regions. This design allows GLASS to learn globally coherent and semantically consistent maps without ground-truth supervision. Extensive experiments demonstrate that GLASS achieves state-of-the-art performance across all regimes, maintaining high accuracy on standard near-isometric tasks while significantly advancing performance in challenging settings. Specifically, it achieves average geodesic errors of 0.21, 4.5, and 5.6 on the inter-class benchmark SNIS and non-isometric benchmarks SMAL and TOPKIDS, reducing errors from URSSM baselines of 0.49, 6.0, and 8.9 by 57%, 25%, and 37%, respectively.
>
---
#### [new 146] BuildMamba: A Visual State-Space Based Model for Multi-Task Building Segmentation and Height Estimation from Satellite Images
- **分类: cs.CV**

- **简介: 该论文提出BuildMamba，解决卫星图像中建筑分割与高度估计任务，通过视觉状态空间模型提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.08523](https://arxiv.org/pdf/2603.08523)**

> **作者:** Sinan U. Ulu; A. Enes Doruk; I. Can Yagmur; Bahadir K. Gunturk; Oguz Hanoglu; Hasan F. Ates
>
> **摘要:** Accurate building segmentation and height estimation from single-view RGB satellite imagery are fundamental for urban analytics, yet remain ill-posed due to structural variability and the high computational cost of global context modeling. While current approaches typically adapt monocular depth architectures, they often suffer from boundary bleeding and systematic underestimation of high-rise structures. To address these limitations, we propose BuildMamba, a unified multi-task framework designed to exploit the linear-time global modeling of visual state-space models. Motivated by the need for stronger structural coupling and computational efficiency, we introduce three modules: a Mamba Attention Module for dynamic spatial recalibration, a Spatial-Aware Mamba-FPN for multi-scale feature aggregation via gated state-space scans, and a Mask-Aware Height Refinement module using semantic priors to suppress height artifacts. Extensive experiments demonstrate that BuildMamba establishes a new performance upper bound across three benchmarks. Specifically, it achieves an IoU of 0.93 and RMSE of 1.77~m on DFC23 benchmark, surpassing state-of-the-art by 0.82~m in height estimation. Simulation results confirm the model's superior robustness and scalability for large-scale 3D urban reconstruction.
>
---
#### [new 147] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings
- **分类: cs.CV; cs.CL; cs.ET; cs.MM; cs.RO**

- **简介: 该论文属于视觉-语言模型的地理时间理解任务，旨在解决模型在真实场景中对时空属性推理能力不足的问题。作者构建了TimeSpot基准数据集，用于评估和提升模型的geo-temporal推理能力。**

- **链接: [https://arxiv.org/pdf/2603.06687](https://arxiv.org/pdf/2603.06687)**

> **作者:** Azmine Toushik Wasi; Shahriyar Zaman Ridoy; Koushik Ahamed Tonmoy; Kinga Tshering; S. M. Muhtasimul Hasan; Wahid Faisal; Tasnim Mohiuddin; Md Rizwan Parvez
>
> **备注:** 66 Pages. In Review
>
> **摘要:** Geo-temporal understanding, the ability to infer location, time, and contextual properties from visual input alone, underpins applications such as disaster management, traffic planning, embodied navigation, world modeling, and geography education. Although recent vision-language models (VLMs) have advanced image geo-localization using cues like landmarks and road signs, their ability to reason about temporal signals and physically grounded spatial cues remains limited. To address this gap, we introduce TimeSpot, a benchmark for evaluating real-world geo-temporal reasoning in VLMs. TimeSpot comprises 1,455 ground-level images from 80 countries and requires structured prediction of temporal attributes (season, month, time of day, daylight phase) and geographic attributes (continent, country, climate zone, environment type, latitude-longitude) directly from visual evidence. It also includes spatial-temporal reasoning tasks that test physical plausibility under real-world uncertainty. Evaluations of state-of-the-art open- and closed-source VLMs show low performance, particularly for temporal inference. While supervised fine-tuning yields improvements, results remain insufficient, highlighting the need for new methods to achieve robust, physically grounded geo-temporal understanding. TimeSpot is available at: this https URL.
>
---
#### [new 148] DSH-Bench: A Difficulty- and Scenario-Aware Benchmark with Hierarchical Subject Taxonomy for Subject-Driven Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有评估基准的不足。提出DSH-Bench，通过多维度评估模型性能，提升模型优化方向的准确性。**

- **链接: [https://arxiv.org/pdf/2603.08090](https://arxiv.org/pdf/2603.08090)**

> **作者:** Zhenyu Hu; Qing Wang; Te Cao; Luo Liao; Longfei Lu; Liqun Liu; Shuang Li; Hang Chen; Mengge Xue; Yuan Chen; Chao Deng; Peng Shu; Huan Yu; Jie Jiang
>
> **摘要:** Significant progress has been achieved in subject-driven text-to-image (T2I) generation, which aims to synthesize new images depicting target subjects according to user instructions. However, evaluating these models remains a significant challenge. Existing benchmarks exhibit critical limitations: 1) insufficient diversity and comprehensiveness in subject images, 2) inadequate granularity in assessing model performance across different subject difficulty levels and prompt scenarios, and 3) a profound lack of actionable insights and diagnostic guidance for subsequent model refinement. To address these limitations, we propose DSH-Bench, a comprehensive benchmark that enables systematic multi-perspective analysis of subject-driven T2I models through four principal innovations: 1) a hierarchical taxonomy sampling mechanism ensuring comprehensive subject representation across 58 fine-grained categories, 2) an innovative classification scheme categorizing both subject difficulty level and prompt scenario for granular capability assessment, 3) a novel Subject Identity Consistency Score (SICS) metric demonstrating a 9.4\% higher correlation with human evaluation compared to existing measures in quantifying subject preservation, and 4) a comprehensive set of diagnostic insights derived from the benchmark, offering critical guidance for optimizing future model training paradigms and data construction strategies. Through an extensive empirical evaluation of 19 leading models, DSH-Bench uncovers previously obscured limitations in current approaches, establishing concrete directions for future research and development.
>
---
#### [new 149] SecAgent: Efficient Mobile GUI Agent with Semantic Context
- **分类: cs.CV**

- **简介: 该论文提出SecAgent，解决移动GUI自动化中多语言数据不足和历史表示效率低的问题，通过构建中文数据集并引入语义上下文机制提升性能。**

- **链接: [https://arxiv.org/pdf/2603.08533](https://arxiv.org/pdf/2603.08533)**

> **作者:** Yiping Xie; Song Chen; Jingxuan Xing; Wei Jiang; Zekun Zhu; Yingyao Wang; Pi Bu; Jun Song; Yuning Jiang; Bo Zheng
>
> **摘要:** Mobile Graphical User Interface (GUI) agents powered by multimodal large language models have demonstrated promising capabilities in automating complex smartphone tasks. However, existing approaches face two critical limitations: the scarcity of high-quality multilingual datasets, particularly for non-English ecosystems, and inefficient history representation methods. To address these challenges, we present SecAgent, an efficient mobile GUI agent at 3B scale. We first construct a human-verified Chinese mobile GUI dataset with 18k grounding samples and 121k navigation steps across 44 applications, along with a Chinese navigation benchmark featuring multi-choice action annotations. Building upon this dataset, we propose a semantic context mechanism that distills history screenshots and actions into concise, natural language summaries, significantly reducing computational costs while preserving task-relevant information. Through supervised and reinforcement fine-tuning, SecAgent outperforms similar-scale baselines and achieves performance comparable to 7B-8B models on our and public navigation benchmarks. We will open-source the training dataset, benchmark, model, and code to advance research in multilingual mobile GUI automation.
>
---
#### [new 150] Fast Attention-Based Simplification of LiDAR Point Clouds for Object Detection and Classification
- **分类: cs.CV**

- **简介: 该论文属于点云简化任务，旨在解决LiDAR数据计算成本高、实时性差的问题。提出一种基于注意力的简化方法，在保持精度的同时提升速度。**

- **链接: [https://arxiv.org/pdf/2603.07593](https://arxiv.org/pdf/2603.07593)**

> **作者:** Z. Rozsa; Á. Madaras; Q. Wei; X. Lu; M. Golarits; H. Yuan; T. Sziranyi; R. Hamzaoui
>
> **摘要:** LiDAR point clouds are widely used in autonomous driving and consist of large numbers of 3D points captured at high frequency to represent surrounding objects such as vehicles, pedestrians, and traffic signs. While this dense data enables accurate perception, it also increases computational cost and power consumption, which can limit real-time deployment. Existing point cloud sampling methods typically face a trade-off: very fast approaches tend to reduce accuracy, while more accurate methods are computationally expensive. To address this limitation, we propose an efficient learned point cloud simplification method for LiDAR data. The method combines a feature embedding module with an attention-based sampling module to prioritize task-relevant regions and is trained end-to-end. We evaluate the method against farthest point sampling (FPS) and random sampling (RS) on 3D object detection on the KITTI dataset and on object classification across four datasets. The method was consistently faster than FPS and achieved similar, and in some settings better, accuracy, with the largest gains under aggressive downsampling. It was slower than RS, but it typically preserved accuracy more reliably at high sampling ratios.
>
---
#### [new 151] LiveWorld: Simulating Out-of-Sight Dynamics in Generative Video World Models
- **分类: cs.CV**

- **简介: 该论文属于视频世界建模任务，解决"视野外动态"问题，提出LiveWorld框架，实现持续的世界演化与场景一致性。**

- **链接: [https://arxiv.org/pdf/2603.07145](https://arxiv.org/pdf/2603.07145)**

> **作者:** Zicheng Duan; Jiatong Xia; Zeyu Zhang; Wenbo Zhang; Gengze Zhou; Chenhui Gou; Yefei He; Feng Chen; Xinyu Zhang; Lingqiao Liu
>
> **摘要:** Recent generative video world models aim to simulate visual environment evolution, allowing an observer to interactively explore the scene via camera control. However, they implicitly assume that the world only evolves within the observer's field of view. Once an object leaves the observer's view, its state is "frozen" in memory, and revisiting the same region later often fails to reflect events that should have occurred in the meantime. In this work, we identify and formalize this overlooked limitation as the "out-of-sight dynamics" problem, which impedes video world models from representing a continuously evolving world. To address this issue, we propose LiveWorld, a novel framework that extends video world models to support persistent world evolution. Instead of treating the world as static observational memory, LiveWorld models a persistent global state composed of a static 3D background and dynamic entities that continue evolving even when unobserved. To maintain these unseen dynamics, LiveWorld introduces a monitor-based mechanism that autonomously simulates the temporal progression of active entities and synchronizes their evolved states upon revisiting, ensuring spatially coherent rendering. For evaluation, we further introduce LiveBench, a dedicated benchmark for the task of maintaining out-of-sight dynamics. Extensive experiments show that LiveWorld enables persistent event evolution and long-term scene consistency, bridging the gap between existing 2D observation-based memory and true 4D dynamic world simulation. The baseline and benchmark will be publicly available at this https URL.
>
---
#### [new 152] Training-free Temporal Object Tracking in Surgical Videos
- **分类: cs.CV**

- **简介: 该论文属于手术视频中的目标跟踪任务，解决标注成本高和标签不一致的问题。利用扩散模型提取特征，结合跨帧交互实现准确跟踪。**

- **链接: [https://arxiv.org/pdf/2603.07839](https://arxiv.org/pdf/2603.07839)**

> **作者:** Subhadeep Koley; Abdolrahim Kadkhodamohammadi; Santiago Barbarisi; Danail Stoyanov; Imanol Luengo
>
> **备注:** Accepted in IPCAI 2025
>
> **摘要:** Purpose: In this paper, we present a novel approach for online object tracking in laparoscopic cholecystectomy (LC) surgical videos, targeting localisation and tracking of critical anatomical structures and instruments. Our method addresses the challenges of costly pixel-level annotations and label inconsistencies inherent in existing datasets. Methods: Leveraging the inherent object localisation capabilities of pre-trained text-to-image diffusion models, we extract representative features from surgical frames without any training or fine-tuning. Our tracking framework uses these features, along with cross-frame interactions via an affinity matrix inspired by query-key-value attention, to ensure temporal continuity in the tracking process. Results: Through a pilot study, we first demonstrate that diffusion features exhibit superior object localisation and consistent semantics across different decoder levels and temporal frames. Later, we perform extensive experiments to validate the effectiveness of our approach, showcasing its superiority over competitors for the task of temporal object tracking. Specifically, we achieve a per-pixel classification accuracy of 79.19%, mean Jaccard Score of 56.20%, and mean F-Score of 79.48% on the publicly available CholeSeg8K dataset. Conclusion: Our work not only introduces a novel application of text-to-image diffusion models but also contributes to advancing the field of surgical video analysis, offering a promising avenue for accurate and cost-effective temporal object tracking in minimally invasive surgery videos.
>
---
#### [new 153] RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的雷达-激光雷达场景识别任务，解决恶劣天气下定位失效问题，提出RLPR框架实现跨模态对齐与精准识别。**

- **链接: [https://arxiv.org/pdf/2603.07920](https://arxiv.org/pdf/2603.07920)**

> **作者:** Zhangshuo Qi; Jingyi Xu; Luqi Cheng; Shichen Wen; Guangming Xiong
>
> **摘要:** All-weather autonomy is critical for autonomous driving, which necessitates reliable localization across diverse scenarios. While LiDAR place recognition is widely deployed for this task, its performance degrades in adverse weather. Conversely, radar-based methods, though weather-resilient, are hindered by the general unavailability of radar maps. To bridge this gap, radar-to-LiDAR place recognition, which localizes radar scans within existing LiDAR maps, has garnered increasing interest. However, extracting discriminative and generalizable features shared between modalities remains challenging, compounded by the scarcity of large-scale paired training data and the signal heterogeneity across radar types. In this work, we propose RLPR, a robust radar-to-LiDAR place recognition framework compatible with single-chip, scanning, and 4D radars. We first design a dual-stream network to extract structural features that abstract away from sensor-specific signal properties (e.g., Doppler or RCS). Subsequently, motivated by our task-specific asymmetry observation between radar and LiDAR, we introduce a two-stage asymmetric cross-modal alignment (TACMA) strategy, which leverages the pre-trained radar branch as a discriminative anchor to guide the alignment process. Experiments on four datasets demonstrate that RLPR achieves state-of-the-art recognition accuracy with strong zero-shot generalization capabilities.
>
---
#### [new 154] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出"Narrative Weaver"，解决生成式AI中长序列视觉内容的一致性问题。融合多模态控制与叙事规划，提升视频生成的连贯性与可控性。**

- **链接: [https://arxiv.org/pdf/2603.06688](https://arxiv.org/pdf/2603.06688)**

> **作者:** Zhengjian Yao; Yongzhi Li; Xinyuan Gao; Quan Chen; Peng Jiang; Yanye Lu
>
> **摘要:** We present "Narrative Weaver", a novel framework that addresses a fundamental challenge in generative AI: achieving multi-modal controllable, long-range, and consistent visual content generation. While existing models excel at generating high-fidelity short-form visual content, they struggle to maintain narrative coherence and visual consistency across extended sequences - a critical limitation for real-world applications such as filmmaking and e-commerce advertising. Narrative Weaver introduces the first holistic solution that seamlessly integrates three essential capabilities: fine-grained control, automatic narrative planning, and long-range coherence. Our architecture combines a Multimodal Large Language Model (MLLM) for high-level narrative planning with a novel fine-grained control module featuring a dynamic Memory Bank that prevents visual drift. To enable practical deployment, we develop a progressive, multi-stage training strategy that efficiently leverages existing pre-trained models, achieving state-of-the-art performance even with limited training data. Recognizing the absence of suitable evaluation benchmarks, we construct and release the E-commerce Advertising Video Storyboard Dataset (EAVSD) - the first comprehensive dataset for this task, containing over 330K high-quality images with rich narrative annotations. Through extensive experiments across three distinct scenarios (controllable multi-scene generation, autonomous storytelling, and e-commerce advertising), we demonstrate our method's superiority while opening new possibilities for AI-driven content creation.
>
---
#### [new 155] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields
- **分类: cs.CV**

- **简介: 该论文属于动态HDR成像任务，解决传统方法在动态场景中出现的伪影和时间不一致问题。提出HDR-NSFF框架，通过4D建模实现更精确的辐射场重建与运动估计。**

- **链接: [https://arxiv.org/pdf/2603.08313](https://arxiv.org/pdf/2603.08313)**

> **作者:** Shin Dong-Yeon; Kim Jun-Seong; Kwon Byung-Ki; Tae-Hyun Oh
>
> **备注:** ICLR 2026. Project page: this https URL
>
> **摘要:** Radiance of real-world scenes typically spans a much wider dynamic range than what standard cameras can capture. While conventional HDR methods merge alternating-exposure frames, these approaches are inherently constrained to 2D pixel-level alignment, often leading to ghosting artifacts and temporal inconsistency in dynamic scenes. To address these limitations, we present HDR-NSFF, a paradigm shift from 2D-based merging to 4D spatio-temporal modeling. Our framework reconstructs dynamic HDR radiance fields from alternating-exposure monocular videos by representing the scene as a continuous function of space and time, and is compatible with both neural radiance field and 4D Gaussian Splatting (4DGS) based dynamic representations. This unified end-to-end pipeline explicitly models HDR radiance, 3D scene flow, geometry, and tone-mapping, ensuring physical plausibility and global coherence. We further enhance robustness by (i) extending semantic-based optical flow with DINO features to achieve exposure-invariant motion estimation, and (ii) incorporating a generative prior as a regularizer to compensate for limited observation in monocular captures and saturation-induced information loss. To evaluate HDR space-time view synthesis, we present the first real-world HDR-GoPro dataset specifically designed for dynamic HDR scenes. Experiments demonstrate that HDR-NSFF recovers fine radiance details and coherent dynamics even under challenging exposure variations, thereby achieving state-of-the-art performance in novel space-time view synthesis. Project page: this https URL
>
---
#### [new 156] High-Fidelity Medical Shape Generation via Skeletal Latent Diffusion
- **分类: cs.CV**

- **简介: 该论文属于医学形状生成任务，解决解剖结构复杂性和拓扑变化带来的生成难题。提出骨骼潜在扩散框架，结合结构先验，实现高效高保真医学形状生成。**

- **链接: [https://arxiv.org/pdf/2603.07504](https://arxiv.org/pdf/2603.07504)**

> **作者:** Guoqing Zhang; Jingyun Yang; Siqi Chen; Anping Zhang; Yang Li
>
> **备注:** 10 pages, 5 figures, journal
>
> **摘要:** Anatomy shape modeling is a fundamental problem in medical data analysis. However, the geometric complexity and topological variability of anatomical structures pose significant challenges to accurate anatomical shape generation. In this work, we propose a skeletal latent diffusion framework that explicitly incorporates structural priors for efficient and high-fidelity medical shape generation. We introduce a shape auto-encoder in which the encoder captures global geometric information through a differentiable skeletonization module and aggregates local surface features into shape latents, while the decoder predicts the corresponding implicit fields over sparsely sampled coordinates. New shapes are generated via a latent-space diffusion model, followed by neural implicit decoding and mesh extraction. To address the limited availability of medical shape data, we construct a large-scale dataset, \textit{MedSDF}, comprising surface point clouds and corresponding signed distance fields across multiple anatomical categories. Extensive experiments on MedSDF and vessel datasets demonstrate that the proposed method achieves superior reconstruction and generation quality while maintaining a higher computational efficiency compared with existing approaches. Code is available at: this https URL.
>
---
#### [new 157] EnsAug: Augmentation-Driven Ensembles for Human Motion Sequence Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体动作序列分析任务，旨在解决数据增强方法生成不真实运动模式的问题。提出EnsAug方法，通过训练多个专家模型提升性能。**

- **链接: [https://arxiv.org/pdf/2603.06661](https://arxiv.org/pdf/2603.06661)**

> **作者:** Bikram De; Habib Irani; Vangelis Metsis
>
> **摘要:** Data augmentation is a crucial technique for training robust deep learning models for human motion, where annotated datasets are often scarce. However, generic augmentation methods often ignore the underlying geometric and kinematic constraints of the human body, risking the generation of unrealistic motion patterns that can degrade model performance. Furthermore, the conventional approach of training a single generalist model on a dataset expanded with a mixture of all available transformations does not fully exploit the unique learning signals provided by each distinct augmentation type. We challenge this convention by introducing a novel training paradigm, EnsAug, that strategically uses augmentation to foster model diversity within an ensemble. Our method involves training an ensemble of specialists, where each model learns from the original dataset augmented by only a single, distinct geometric transformation. Experiments on sign language and human activity recognition benchmarks demonstrate that our diversified ensemble methodology significantly outperforms the standard practice of training one model on a combined augmented dataset and achieves state-of-the-art accuracy on two sign language and one human activity recognition dataset while offering greater modularity and efficiency. Our primary contribution is the empirical validation of this training strategy, establishing an effective baseline for leveraging data augmentation in skeletal motion analysis.
>
---
#### [new 158] EvolveReason: Self-Evolving Reasoning Paradigm for Explainable Deepfake Facial Image Identification
- **分类: cs.CV**

- **简介: 该论文属于深度伪造图像识别任务，旨在解决传统方法缺乏解释性、VLM方法存在幻觉的问题。提出EvolveReason框架，通过模拟人类推理过程提升识别准确性和解释可靠性。**

- **链接: [https://arxiv.org/pdf/2603.07515](https://arxiv.org/pdf/2603.07515)**

> **作者:** Binjia Zhou; Dawei Luo; Shuai Chen; Feng Xu; Seow; Haoyuan Li; Jiachi Wang; Jiawen Wang; Zunlei Feng; Yijun Bei
>
> **摘要:** With the rapid advancement of AIGC technology, developing identification methods to address the security challenges posed by deepfakes has become urgent. Face forgery identification techniques can be categorized into two types: traditional classification methods and explainable VLM approaches. The former provides classification results but lacks explanatory ability, while the latter, although capable of providing coarse-grained explanations, often suffers from hallucinations and insufficient detail. To overcome these limitations, we propose EvolveReason, which mimics the reasoning and observational processes of human auditors when identifying face forgeries. By constructing a chain-of-thought dataset, CoT-Face, tailored for advanced VLMs, our approach guides the model to think in a human-like way, prompting it to output reasoning processes and judgment results. This provides practitioners with reliable analysis and helps alleviate hallucination. Additionally, our framework incorporates a forgery latent-space distribution capture module, enabling EvolveReason to identify high-frequency forgery cues difficult to extract from the original images. To further enhance the reliability of textual explanations, we introduce a self-evolution exploration strategy, leveraging reinforcement learning to allow the model to iteratively explore and optimize its textual descriptions in a two-stage process. Experimental results show that EvolveReason not only outperforms the current state-of-the-art methods in identification performance but also accurately identifies forgery details and demonstrates generalization capabilities.
>
---
#### [new 159] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
- **分类: cs.CV**

- **简介: 该论文属于红外可见图像融合任务，解决缺失红外模态下的融合问题。提出基于字典的系数域框架，实现可解释的跨模态融合。**

- **链接: [https://arxiv.org/pdf/2603.08018](https://arxiv.org/pdf/2603.08018)**

> **作者:** Yafei Zhang; Meng Ma; Huafeng Li; Yu Liu
>
> **备注:** This paper has been accepted by CVPR 2026
>
> **摘要:** Infrared-visible (IR-VIS) image fusion is vital for perception and security, yet most methods rely on the availability of both modalities during training and inference. When the infrared modality is absent, pixel-space generative substitutes become hard to control and inherently lack interpretability. We address missing-IR fusion by proposing a dictionary-guided, coefficient-domain framework built upon a shared convolutional dictionary. The pipeline comprises three key components: (1) Joint Shared-dictionary Representation Learning (JSRL) learns a unified and interpretable atom space shared by both IR and VIS modalities; (2) VIS-Guided IR Inference (VGII) transfers VIS coefficients to pseudo-IR coefficients in the coefficient domain and performs a one-step closed-loop refinement guided by a frozen large language model as a weak semantic prior; and (3) Adaptive Fusion via Representation Inference (AFRI) merges VIS structures and inferred IR cues at the atom level through window attention and convolutional mixing, followed by reconstruction with the shared dictionary. This encode-transfer-fuse-reconstruct pipeline avoids uncontrolled pixel-space generation while ensuring prior preservation within interpretable dictionary-coefficient representation. Experiments under missing-IR settings demonstrate consistent improvements in perceptual quality and downstream detection performance. To our knowledge, this represents the first framework that jointly learns a shared dictionary and performs coefficient-domain inference-fusion to tackle missing-IR fusion. The source code is publicly available at this https URL.
>
---
#### [new 160] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉语言模型的比较推理任务，旨在评估模型在细微差异识别上的表现。针对现有基准不足，提出VLM-SubtleBench，涵盖多种差异类型，揭示模型与人类性能差距。**

- **链接: [https://arxiv.org/pdf/2603.07888](https://arxiv.org/pdf/2603.07888)**

> **作者:** Minkyu Kim; Sangheon Lee; Dongmin Park
>
> **备注:** ICLR 2026
>
> **摘要:** The ability to distinguish subtle differences between visually similar images is essential for diverse domains such as industrial anomaly detection, medical imaging, and aerial surveillance. While comparative reasoning benchmarks for vision-language models (VLMs) have recently emerged, they primarily focus on images with large, salient differences and fail to capture the nuanced reasoning required for real-world applications. In this work, we introduce VLM-SubtleBench, a benchmark designed to evaluate VLMs on subtle comparative reasoning. Our benchmark covers ten difference types - Attribute, State, Emotion, Temporal, Spatial, Existence, Quantity, Quality, Viewpoint, and Action - and curate paired question-image sets reflecting these fine-grained variations. Unlike prior benchmarks restricted to natural image datasets, our benchmark spans diverse domains, including industrial, aerial, and medical imagery. Through extensive evaluation of both proprietary and open-source VLMs, we reveal systematic gaps between model and human performance across difference types and domains, and provide controlled analyses highlighting where VLMs' reasoning sharply deteriorates. Together, our benchmark and findings establish a foundation for advancing VLMs toward human-level comparative reasoning.
>
---
#### [new 161] Fusion Complexity Inversion: Why Simpler Cross View Modules Outperform SSMs and Cross View Attention Transformers for Pasture Biomass Regression
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业图像回归任务，旨在提升牧草生物量估计精度。针对数据稀缺问题，研究比较了不同融合模块，发现简单结构更优，并提出优先选择骨干网络和局部模块的指导原则。**

- **链接: [https://arxiv.org/pdf/2603.07819](https://arxiv.org/pdf/2603.07819)**

> **作者:** Mridankan Mandal
>
> **摘要:** Accurate estimation of pasture biomass from agricultural imagery is critical for sustainable livestock management, yet existing methods are limited by the small, imbalanced, and sparsely annotated datasets typical of real world monitoring. In this study, adaptation of vision foundation models to agricultural regression is systematically evaluated on the CSIRO Pasture Biomass benchmark, a 357 image dual view dataset with laboratory validated, component wise ground truth for five biomass targets, through 17 configurations spanning four backbones (EfficientNet-B3 to DINOv3-ViT-L), five cross view fusion mechanisms, and a 4x2 metadata factorial. A counterintuitive principle, termed "fusion complexity inversion", is uncovered: on scarce agricultural data, a two layer gated depthwise convolution (R^2 = 0.903) outperforms cross view attention transformers (0.833), bidirectional SSMs (0.819), and full Mamba (0.793, below the no fusion baseline). Backbone pretraining scale is found to monotonically dominate all architectural choices, with the DINOv2 -> DINOv3 upgrade alone yielding +5.0 R^2 points. Training only metadata (species, state, and NDVI) is shown to create a universal ceiling at R^2 ~ 0.829, collapsing an 8.4 point fusion spread to 0.1 points. Actionable guidelines for sparse agricultural benchmarks are established: backbone quality should be prioritized over fusion complexity, local modules preferred over global alternatives, and features unavailable at inference excluded.
>
---
#### [new 162] Active View Selection with Perturbed Gaussian Ensemble for Tomographic Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于CT重建任务，解决稀疏视角下重建精度不足的问题。提出Perturbed Gaussian Ensemble框架，通过不确定性建模和序列决策提升视图选择效果。**

- **链接: [https://arxiv.org/pdf/2603.06852](https://arxiv.org/pdf/2603.06852)**

> **作者:** Yulun Wu; Ruyi Zha; Wei Cao; Yingying Li; Yuanhao Cai; Yaoyao Liu
>
> **摘要:** Sparse-view computed tomography (CT) is critical for reducing radiation exposure to patients. Recent advances in radiative 3D Gaussian Splatting (3DGS) have enabled fast and accurate sparse-view CT reconstruction. Despite these algorithmic advancements, practical reconstruction fidelity remains fundamentally bounded by the quality of the captured data, raising the crucial yet underexplored problem of X-ray active view selection. Existing active view selection methods are primarily designed for natural-light scenes and fail to capture the unique geometric ambiguities and physical attenuation properties inherent in X-ray imaging. In this paper, we present Perturbed Gaussian Ensemble, an active view selection framework that integrates uncertainty modeling with sequential decision-making, tailored for X-ray Gaussian Splatting. Specifically, we identify low-density Gaussian primitives that are likely to be uncertain and apply stochastic density scaling to construct an ensemble of plausible Gaussian density fields. For each candidate projection, we measure the structural variance of the ensemble predictions and select the one with the highest variance as the next best view. Extensive experimental results on arbitrary-trajectory CT benchmarks demonstrate that our density-guided perturbation strategy effectively eliminates geometric artifacts and consistently outperforms existing baselines in progressive tomographic reconstruction under unified view selection protocols.
>
---
#### [new 163] MipSLAM: Alias-Free Gaussian Splatting SLAM
- **分类: cs.CV**

- **简介: 该论文提出MipSLAM，解决SLAM中的模糊和轨迹漂移问题，通过频率感知的优化方法提升渲染质量和定位精度。**

- **链接: [https://arxiv.org/pdf/2603.06989](https://arxiv.org/pdf/2603.06989)**

> **作者:** Yingzhao Li; Yan Li; Shixiong Tian; Yanjie Liu; Lijun Zhao; Gim Hee Lee
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** This paper introduces MipSLAM, a frequency-aware 3D Gaussian Splatting (3DGS) SLAM framework capable of high-fidelity anti-aliased novel view synthesis and robust pose estimation under varying camera configurations. Existing 3DGS-based SLAM systems often suffer from aliasing artifacts and trajectory drift due to inadequate filtering and purely spatial optimization. To overcome these limitations, we propose an Elliptical Adaptive Anti-aliasing (EAA) algorithm that approximates Gaussian contributions via geometry-aware numerical integration, avoiding costly analytic computation. Furthermore, we present a Spectral-Aware Pose Graph Optimization (SA-PGO) module that reformulates trajectory estimation in the frequency domain, effectively suppressing high-frequency noise and drift through graph Laplacian analysis. A novel local frequency-domain perceptual loss is also introduced to enhance fine-grained geometric detail recovery. Extensive evaluations on Replica and TUM datasets demonstrate that MipSLAM achieves state-of-the-art rendering quality and localization accuracy across multiple resolutions while maintaining real-time capability. Code is available at this https URL.
>
---
#### [new 164] RADAR: A Multimodal Benchmark for 3D Image-Based Radiology Report Review
- **分类: cs.CV**

- **简介: 该论文提出RADAR，一个用于放射学报告差异分析的多模态基准。任务是评估报告修改的临床合理性，解决报告差异分析缺乏标准基准的问题。工作包括构建包含3D影像和报告修改的数据集，并定义结构化评估任务。**

- **链接: [https://arxiv.org/pdf/2603.06681](https://arxiv.org/pdf/2603.06681)**

> **作者:** Zhaoyi Sun; Minal Jagtiani; Wen-wai Yim; Fei Xia; Martin Gunn; Meliha Yetisgen; Asma Ben Abacha
>
> **摘要:** Radiology reports for the same patient examination may contain clinically meaningful discrepancies arising from interpretation differences, reporting variability, or evolving assessments. Systematic analysis of such discrepancies is important for quality assurance, clinical decision support, and multimodal model development, yet remains limited by the lack of standardized benchmarks. We present RADAR, a multimodal benchmark for radiology report discrepancy analysis that pairs 3D medical images with a preliminary report and corresponding candidate edits for the same study. The dataset reflects a standard clinical workflow in which trainee radiologists author preliminary reports that are subsequently reviewed and revised by attending radiologists. RADAR defines a structured discrepancy assessment task requiring models to evaluate proposed edits by determining image-level agreement, assessing clinical severity, and classifying edit type (correction, addition, or clarification). In contrast to prior work emphasizing binary error detection or comparison against fully independent reference reports, RADAR targets fine-grained clinical reasoning and image-text alignment at the report review stage. The benchmark consists of expert-annotated abdominal CT examinations and is accompanied by standardized evaluation protocols to support systematic comparison of multimodal models. RADAR provides a clinically grounded testbed for evaluating multimodal systems as reviewers of radiology report edits.
>
---
#### [new 165] Geometric Knowledge-Assisted Federated Dual Knowledge Distillation Approach Towards Remote Sensing Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分析任务，解决多卫星数据异构性带来的模型训练难题。提出GK-FedDKD框架，通过知识蒸馏和几何知识聚合提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07774](https://arxiv.org/pdf/2603.07774)**

> **作者:** Luyao Zou; Fei Pan; Jueying Li; Yan Kyaw Tun; Apurba Adhikary; Zhu Han; Hayoung Oh
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Federated learning (FL) has recently become a promising solution for analyzing remote sensing satellite imagery (RSSI). However, the large scale and inherent data heterogeneity of images collected from multiple satellites, where the local data distribution of each satellite differs from the global one, present significant challenges to effective model training. To address this issue, we propose a Geometric Knowledge-Guided Federated Dual Knowledge Distillation (GK-FedDKD) framework for RSSI analysis. In our approach, each local client first distills a teacher encoder (TE) from multiple student encoders (SEs) trained with unlabeled augmented data. The TE is then connected with a shared classifier to form a teacher network (TN) that supervises the training of a new student network (SN). The intermediate representations of the TN are used to compute local covariance matrices, which are aggregated at the server to generate global geometric knowledge (GGK). This GGK is subsequently employed for local embedding augmentation to further guide SN training. We also design a novel loss function and a multi-prototype generation pipeline to stabilize the training process. Evaluation over multiple datasets showcases that the proposed GK-FedDKD approach is superior to the considered state-of-the-art baselines, e.g., the proposed approach with the Swin-T backbone surpasses previous SOTA approaches by an average 68.89% on the EuroSAT dataset.
>
---
#### [new 166] Novel Semantic Prompting for Zero-Shot Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于零样本动作识别任务，旨在通过语义提示提升模型对未见动作的识别能力。工作包括引入SP-CLIP框架，利用多层级语义提示增强文本语义，无需修改视觉编码器即可提高识别效果。**

- **链接: [https://arxiv.org/pdf/2603.08289](https://arxiv.org/pdf/2603.08289)**

> **作者:** Salman Iqbal; Waheed Rehman
>
> **摘要:** Zero-shot action recognition relies on transferring knowledge from vision-language models to unseen actions using semantic descriptions. While recent methods focus on temporal modeling or architectural adaptations to handle video data, we argue that semantic prompting alone provides a strong and underexplored signal for zero-shot action understanding. We introduce SP-CLIP, a lightweight framework that augments frozen vision-language models with structured semantic prompts describing actions at multiple levels of abstraction, such as intent, motion, and object interaction. Without modifying the visual encoder or learning additional parameters, SP-CLIP aligns video representations with enriched textual semantics through prompt aggregation and consistency scoring. Experiments across standard benchmarks show that semantic prompting substantially improves zero-shot action recognition, particularly for fine-grained and compositional actions, while preserving the efficiency and generalization of pretrained models.
>
---
#### [new 167] PICS: Pairwise Image Compositing with Spatial Interactions
- **分类: cs.CV**

- **简介: 该论文提出PICS，解决扩散模型在图像合成中保持空间关系的问题。通过自监督分解方式，建模物体间交互，提升合成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.06873](https://arxiv.org/pdf/2603.06873)**

> **作者:** Hang Zhou; Xinxin Zuo; Sen Wang; Li Cheng
>
> **备注:** ICLR 2026. Project page: this https URL , code: this https URL
>
> **摘要:** Despite strong single-turn performance, diffusion-based image compositing often struggles to preserve coherent spatial relations in pairwise or sequential edits, where subsequent insertions may overwrite previously generated content and disrupt physical consistency. We introduce PICS, a self-supervised composition-by-decomposition paradigm that composes objects in parallel while explicitly modeling the compositional interactions among (fully-/partially-)visible objects and background. At its core, an Interaction Transformer employs mask-guided Mixture-of-Experts to route background, exclusive, and overlap regions to dedicated experts, with an adaptive {\alpha}-blending strategy that infers a compatibility-aware fusion of overlapping objects while preserving boundary fidelity. To further enhance robustness to geometric variations, we incorporate geometry-aware augmentations covering both out-of-plane and in-plane pose changes of objects. Our method delivers superior pairwise compositing quality and substantially improved stability, with extensive evaluations across virtual try-on, indoor, and street scene settings showing consistent gains over state-of-the-art baselines. Code and data are available at this https URL
>
---
#### [new 168] QualiTeacher: Quality-Conditioned Pseudo-Labeling for Real-World Image Restoration
- **分类: cs.CV**

- **简介: 该论文针对真实世界图像修复任务，解决伪标签质量不佳导致模型学习缺陷的问题，提出QualiTeacher框架，通过质量条件监督提升修复效果。**

- **链接: [https://arxiv.org/pdf/2603.08030](https://arxiv.org/pdf/2603.08030)**

> **作者:** Fengyang Xiao; Jingjia Feng; Peng Hu; Dingming Zhang; Lei Xu; Guanyi Qin; Lu Li; Chunming He; Sina Farsiu
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Real-world image restoration (RWIR) is a highly challenging task due to the absence of clean ground-truth images. Many recent methods resort to pseudo-label (PL) supervision, often within a Mean-Teacher (MT) framework. However, these methods face a critical paradox: unconditionally trusting the often imperfect, low-quality PLs forces the student model to learn undesirable artifacts, while discarding them severely limits data diversity and impairs model generalization. In this paper, we propose QualiTeacher, a novel framework that transforms pseudo-label quality from a noisy liability into a conditional supervisory signal. Instead of filtering, QualiTeacher explicitly conditions the student model on the quality of the PLs, estimated by an ensemble of complementary non-reference image quality assessment (NR-IQA) models spanning low-level distortion and semantic-level assessment. This strategy teaches the student network to learn a quality-graded restoration manifold, enabling it to understand what constitutes different quality levels. Consequently, it can not only avoid mimicking artifacts from low-quality labels but also extrapolate to generate results of higher quality than the teacher itself. To ensure the robustness and accuracy of this quality-driven learning, we further enhance the process with a multi-augmentation scheme to diversify the PL quality spectrum, a score-based preference optimization strategy inspired by Direct Preference Optimization (DPO) to enforce a monotonically ordered quality separation, and a cropped consistency loss to prevent adversarial over-optimization (reward hacking) of the IQA models. Experiments on standard RWIR benchmarks demonstrate that QualiTeacher can serve as a plug-and-play strategy to improve the quality of the existing pseudo-labeling framework, establishing a new paradigm for learning from imperfect supervision. Code will be released.
>
---
#### [new 169] Edged USLAM: Edge-Aware Event-Based SLAM with Learning-Based Depth Priors
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位与建图任务，解决传统SLAM在极端条件下的失效问题。通过融合事件相机与IMU，提出Edged USLAM系统，提升稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2603.08150](https://arxiv.org/pdf/2603.08150)**

> **作者:** Şebnem Sarıözkan; Hürkan Şahin; Olaya Álvarez-Tuñón; Erdal Kayacan
>
> **备注:** 8 pages, 7 figures, 3 tables. Accepted to ICRA 2026. Project code and datasets available at this https URL
>
> **摘要:** Conventional visual simultaneous localization and mapping (SLAM) algorithms often fail under rapid motion, low illumination, or abrupt lighting transitions due to motion blur and limited dynamic range. Event cameras mitigate these issues with high temporal resolution and high dynamic range (HDR), but their sparse, asynchronous outputs complicate feature extraction and integration with other sensors; e.g. inertial measurement units (IMUs) and standard cameras. We present Edged USLAM, a hybrid visual-inertial system that extends Ultimate SLAM (USLAM) with an edge-aware front-end and a lightweight depth module. The frontend enhances event frames for robust feature tracking and nonlinear motion compensation, while the depth module provides coarse, region-of-interest (ROI)-based scene depth to improve motion compensation and scale consistency. Evaluations across public benchmarks and real-world unmanned air vehicle (UAV) flights demonstrate that performance varies significantly by scenario. For instance, event-only methods like point-line event-based visual-inertial odometry (PL-EVIO) or learning-based pipelines such as deep event-based visual odometry (DEVO) excel in highly aggressive or extreme HDR conditions. In contrast, Edged USLAM provides superior stability and minimal drift in slow or structured trajectories, ensuring consistently accurate localization on real flights under challenging illumination. These findings highlight the complementary strengths of event-only, learning-based, and hybrid approaches, while positioning Edged USLAM as a robust solution for diverse aerial navigation tasks.
>
---
#### [new 170] NuNext: Reframing Nucleus Detection as Next-Point Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决核检测问题。通过将核检测重新定义为下一步点预测，提出一种新方法提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.07098](https://arxiv.org/pdf/2603.07098)**

> **作者:** Zhongyi Shui; Honglin Li; Xiaozhong Ji; Ye Zhang; Zijiang Yang; Chenglu Zhu; Yuxuan Sun; Kai Yao; Conghui He; Cheng Tan
>
> **摘要:** Nucleus detection in histopathology is pivotal for a wide range of clinical applications. Existing approaches either regress nuclear proxy maps that require complex post-processing, or employ dense anchors or queries that introduce severe foreground-background imbalance. In this work, we reformulate nucleus detection as next-point prediction, wherein a multimodal large language model is developed to directly output foreground nucleus centroids from the input image. The model is trained in two stages. In the supervised learning stage, we propose spatial-aware soft supervision to relax strict centroid matching and a chain-of-visual-thought strategy to incorporate visual priors that facilitate coordinate prediction. In the reinforcement fine-tuning stage, we design distribution matching reward, low-variance group filtering, and fine-grained advantage shaping to further improve the model's detection quality. Extensive experiments on nine widely used benchmarks demonstrate the superiority of our method. Code will be released soon.
>
---
#### [new 171] AgrI Challenge: A Data-Centric AI Competition for Cross-Team Validation in Agricultural Vision
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于农业视觉任务，旨在解决模型泛化能力不足的问题。通过数据驱动的竞赛框架，探索数据收集对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.07356](https://arxiv.org/pdf/2603.07356)**

> **作者:** Mohammed Brahimi; Karim Laabassi; Mohamed Seghir Hadj Ameur; Aicha Boutorh; Badia Siab-Farsi; Amin Khouani; Omar Farouk Zouak; Seif Eddine Bouziane; Kheira Lakhdari; Abdelkader Nabil Benghanem
>
> **备注:** 17 pages, 8 figures, 6 tables. Introduces the AgrI Challenge dataset containing 50,673 field images of six tree species collected by twelve independent teams
>
> **摘要:** Machine learning models in agricultural vision often achieve high accuracy on curated datasets but fail to generalize under real field conditions due to distribution shifts between training and deployment environments. Moreover, most machine learning competitions focus primarily on model design while treating datasets as fixed resources, leaving the role of data collection practices in model generalization largely unexplored. We introduce the AgrI Challenge, a data-centric competition framework in which multiple teams independently collect field datasets, producing a heterogeneous multi-source benchmark that reflects realistic variability in acquisition conditions. To systematically evaluate cross-domain generalization across independently collected datasets, we propose Cross-Team Validation (CTV), an evaluation paradigm that treats each team's dataset as a distinct domain. CTV includes two complementary protocols: Train-on-One-Team-Only (TOTO), which measures single-source generalization, and Leave-One-Team-Out (LOTO), which evaluates collaborative multi-source training. Experiments reveal substantial generalization gaps under single-source training: models achieve near-perfect validation accuracy yet exhibit validation-test gaps of up to 16.20% (DenseNet121) and 11.37% (Swin Transformer) when evaluated on datasets collected by other teams. In contrast, collaborative multi-source training dramatically improves robustness, reducing the gap to 2.82% and 1.78%, respectively. The challenge also produced a publicly available dataset of 50,673 field images of six tree species collected by twelve independent teams, providing a diverse benchmark for studying domain shift and data-centric learning in agricultural vision.
>
---
#### [new 172] SIGMAE: A Spectral-Index-Guided Foundation Model for Multispectral Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分析任务，旨在解决多光谱图像预训练中的特征学习难题。提出SIGMAE模型，通过光谱指数引导动态掩码，提升空间-光谱特征表示能力。**

- **链接: [https://arxiv.org/pdf/2603.07463](https://arxiv.org/pdf/2603.07463)**

> **作者:** Xiaokang Zhang; Bo Li; Chufeng Zhou; Weikang Yu; Lefei Zhang
>
> **备注:** 17pages,10figures
>
> **摘要:** Pretraining and fine-tuning have emerged as a new paradigm in remote sensing image interpretation. Among them, Masked Autoencoder (MAE)-based pretraining stands out for its strong capability to learn general feature representations via reconstructing masked image regions. However, applying MAE to multispectral remote sensing images remains challenging due to complex backgrounds, indistinct targets, and the lack of semantic guidance during masking, which hinders the learning of underlying structures and meaningful spatial-spectral features. To address this, we propose a simple yet effective approach, Spectral Index-Guided MAE (SIGMAE), for multispectral image pretraining. The core idea is to incorporate domain-specific spectral indices as prior knowledge to guide dynamic token masking toward informative regions. SIGMAE introduces Semantic Saliency-Guided Dynamic Token Masking (SSDTM), a curriculum-style strategy that quantifies each patch's semantic richness and internal heterogeneity to adaptively select the most informative tokens during training. By prioritizing semantically salient regions and progressively increasing sample difficulty, SSDTM enhances spectrally rich and structurally aware representation learning, mitigates overfitting, and reduces redundant computation compared with random masking. Extensive experiments on five widely used datasets covering various downstream tasks, including scene classification, semantic segmentation, object extraction and change detection, demonstrate that SIGMAE outperforms other pretrained geospatial foundation models. Moreover, it exhibits strong spatial-spectral reconstruction capability, even with a 90% mask ratio, and improves complex target recognition under limited labeled data. The source codes and model weights will be released at this https URL.
>
---
#### [new 173] Geometric Transformation-Embedded Mamba for Learned Video Compression
- **分类: cs.CV**

- **简介: 该论文属于视频压缩任务，旨在解决传统方法复杂的问题。提出一种基于非线性变换的框架，结合几何变换和局部优化模块，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2603.07912](https://arxiv.org/pdf/2603.07912)**

> **作者:** Hao Wei; Yanhui Zhou; Chenyang Ge
>
> **摘要:** Although learned video compression methods have exhibited outstanding performance, most of them typically follow a hybrid coding paradigm that requires explicit motion estimation and compensation, resulting in a complex solution for video compression. In contrast, we introduce a streamlined yet effective video compression framework founded on a direct transform strategy, i.e., nonlinear transform, quantization, and entropy coding. We first develop a cascaded Mamba module (CMM) with different embedded geometric transformations to effectively explore both long-range spatial and temporal dependencies. To improve local spatial representation, we introduce a locality refinement feed-forward network (LRFFN) that incorporates a hybrid convolution block based on difference convolutions. We integrate the proposed CMM and LRFFN into the encoder and decoder of our compression framework. Moreover, we present a conditional channel-wise entropy model that effectively utilizes conditional temporal priors to accurately estimate the probability distributions of current latent features. Extensive experiments demonstrate that our method outperforms state-of-the-art video compression approaches in terms of perceptual quality and temporal consistency under low-bitrate constraints. Our source codes and models will be available at this https URL.
>
---
#### [new 174] Foley-Flow: Coordinated Video-to-Audio Generation with Masked Audio-Visual Alignment and Dynamic Conditional Flows
- **分类: cs.CV; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决音频与视频在语义和节奏上的同步问题。通过掩码音频视觉对齐和动态条件流方法，提升生成音频的协调性。**

- **链接: [https://arxiv.org/pdf/2603.08126](https://arxiv.org/pdf/2603.08126)**

> **作者:** Shentong Mo; Yibing Song
>
> **摘要:** Coordinated audio generation based on video inputs typically requires a strict audio-visual (AV) alignment, where both semantics and rhythmics of the generated audio segments shall correspond to those in the video frames. Previous studies leverage a two-stage design where the AV encoders are firstly aligned via contrastive learning, then the encoded video representations guide the audio generation process. We observe that both contrastive learning and global video guidance are effective in aligning overall AV semantics while limiting temporally rhythmic synchronization. In this work, we propose FoleyFlow to first align unimodal AV encoders via masked modeling training, where the masked audio segments are recovered under the guidance of the corresponding video segments. After training, the AV encoders which are separately pretrained using only unimodal data are aligned with semantic and rhythmic consistency. Then, we develop a dynamic conditional flow for the final audio generation. Built upon the efficient velocity flow generation framework, our dynamic conditional flow utilizes temporally varying video features as the dynamic condition to guide corresponding audio segment generations. To this end, we extract coherent semantic and rhythmic representations during masked AV alignment, and use this representation of video segments to guide audio generation temporally. Our audio results are evaluated on the standard benchmarks and largely surpass existing results under several metrics. The superior performance indicates that FoleyFlow is effective in generating coordinated audios that are both semantically and rhythmically coherent to various video sequences.
>
---
#### [new 175] GarmentPainter: Efficient 3D Garment Texture Synthesis with Character-Guided Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出GarmentPainter，解决3D服装纹理生成中的一致性与效率问题。通过UV位置图和类型选择模块，实现高效、高质量的3D纹理合成。**

- **链接: [https://arxiv.org/pdf/2603.08228](https://arxiv.org/pdf/2603.08228)**

> **作者:** Jinbo Wu; Xiaobo Gao; Xing Liu; Chen Zhao; Jialun Liu
>
> **摘要:** Generating high-fidelity, 3D-consistent garment textures remains a challenging problem due to the inherent complexities of garment structures and the stringent requirement for detailed, globally consistent texture synthesis. Existing approaches either rely on 2D-based diffusion models, which inherently struggle with 3D consistency, require expensive multi-step optimization or depend on strict spatial alignment between 2D reference images and 3D meshes, which limits their flexibility and scalability. In this work, we introduce GarmentPainter, a simple yet efficient framework for synthesizing high-quality, 3D-aware garment textures in UV space. Our method leverages a UV position map as the 3D structural guidance, ensuring texture consistency across the garment surface during texture generation. To enhance control and adaptability, we introduce a type selection module, enabling fine-grained texture generation for specific garment components based on a character reference image, without requiring alignment between the reference image and the 3D mesh. GarmentPainter efficiently integrates all guidance signals into the input of a diffusion model in a spatially aligned manner, without modifying the underlying UNet architecture. Extensive experiments demonstrate that GarmentPainter achieves state-of-the-art performance in terms of visual fidelity, 3D consistency, and computational efficiency, outperforming existing methods in both qualitative and quantitative evaluations.
>
---
#### [new 176] TDM-R1: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成模型的强化学习任务，旨在解决少步扩散模型难以使用非可微奖励的问题。提出TDM-R1方法，通过分离奖励学习与生成学习，有效整合非可微奖励，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07700](https://arxiv.org/pdf/2603.07700)**

> **作者:** Yihong Luo; Tianyang Hu; Weijian Luo; Jing Tang
>
> **备注:** this https URL
>
> **摘要:** While few-step generative models have enabled powerful image and video generation at significantly lower cost, generic reinforcement learning (RL) paradigms for few-step models remain an unsolved problem. Existing RL approaches for few-step diffusion models strongly rely on back-propagating through differentiable reward models, thereby excluding the majority of important real-world reward signals, e.g., non-differentiable rewards such as humans' binary likeness, object counts, etc. To properly incorporate non-differentiable rewards to improve few-step generative models, we introduce TDM-R1, a novel reinforcement learning paradigm built upon a leading few-step model, Trajectory Distribution Matching (TDM). TDM-R1 decouples the learning process into surrogate reward learning and generator learning. Furthermore, we developed practical methods to obtain per-step reward signals along the deterministic generation trajectory of TDM, resulting in a unified RL post-training method that significantly improves few-step models' ability with generic rewards. We conduct extensive experiments ranging from text-rendering, visual quality, and preference alignment. All results demonstrate that TDM-R1 is a powerful reinforcement learning paradigm for few-step text-to-image models, achieving state-of-the-art reinforcement learning performances on both in-domain and out-of-domain metrics. Furthermore, TDM-R1 also scales effectively to the recent strong Z-Image model, consistently outperforming both its 100-NFE and few-step variants with only 4 NFEs. Project page: this https URL
>
---
#### [new 177] OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出OccTrack360基准和FoSOcc框架，解决环绕鱼眼相机的4D全景占用跟踪问题，提升动态3D环境理解的准确性与连续性。**

- **链接: [https://arxiv.org/pdf/2603.08521](https://arxiv.org/pdf/2603.08521)**

> **作者:** Yongzhi Lin; Kai Luo; Yuanfan Zheng; Hao Shi; Mengfei Duan; Yang Liu; Kailun Yang
>
> **备注:** The benchmark and source code will be made publicly available at this https URL
>
> **摘要:** Understanding dynamic 3D environments in a spatially continuous and temporally consistent manner is fundamental for robotics and autonomous driving. While recent advances in occupancy prediction provide a unified representation of scene geometry and semantics, progress in 4D panoptic occupancy tracking remains limited by the lack of benchmarks that support surround-view fisheye sensing, long temporal sequences, and instance-level voxel tracking. To address this gap, we present OccTrack360, a new benchmark for 4D panoptic occupancy tracking from surround-view fisheye cameras. OccTrack360 provides substantially longer and more diverse sequences (174~2234 frames) than prior benchmarks, together with principled voxel visibility annotations, including an all-direction occlusion mask and an MEI-based fisheye field-of-view mask. To establish a strong fisheye-oriented baseline, we further propose Focus on Sphere Occ (FoSOcc), a framework that addresses two core challenges in fisheye occupancy tracking: distorted spherical projection and inaccurate voxel-space localization. FoSOcc includes a Center Focusing Module (CFM) to enhance instance-aware spatial localization through supervised focus guidance, and a Spherical Lift Module (SLM) that extends perspective lifting to fisheye imaging under the Unified Projection Model. Extensive experiments on Occ3D-Waymo and OccTrack360 show that our method improves occupancy tracking quality with notable gains on geometrically regular categories, and establishes a strong baseline for future research on surround-view fisheye 4D occupancy tracking. The benchmark and source code will be made publicly available at this https URL.
>
---
#### [new 178] Variational Flow Maps: Make Some Noise for One-Step Conditional Generation
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文提出Variational Flow Maps（VFM），解决单步条件生成问题。通过学习初始噪声分布，实现高效、准确的图像生成，提升采样速度与条件约束满足度。**

- **链接: [https://arxiv.org/pdf/2603.07276](https://arxiv.org/pdf/2603.07276)**

> **作者:** Abbas Mammadov; So Takao; Bohan Chen; Ricardo Baptista; Morteza Mardani; Yee Whye Teh; Julius Berner
>
> **摘要:** Flow maps enable high-quality image generation in a single forward pass. However, unlike iterative diffusion models, their lack of an explicit sampling trajectory impedes incorporating external constraints for conditional generation and solving inverse problems. We put forth Variational Flow Maps, a framework for conditional sampling that shifts the perspective of conditioning from "guiding a sampling path", to that of "learning the proper initial noise". Specifically, given an observation, we seek to learn a noise adapter model that outputs a noise distribution, so that after mapping to the data space via flow map, the samples respect the observation and data prior. To this end, we develop a principled variational objective that jointly trains the noise adapter and the flow map, improving noise-data alignment, such that sampling from complex data posterior is achieved with a simple adapter. Experiments on various inverse problems show that VFMs produce well-calibrated conditional samples in a single (or few) steps. For ImageNet, VFM attains competitive fidelity while accelerating the sampling by orders of magnitude compared to alternative iterative diffusion/flow models. Code is available at this https URL
>
---
#### [new 179] QdaVPR: A novel query-based domain-agnostic model for visual place recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，解决域变化带来的挑战。提出QdaVPR模型，通过对抗学习和三元组监督提升模型的域不变性和识别能力。**

- **链接: [https://arxiv.org/pdf/2603.07414](https://arxiv.org/pdf/2603.07414)**

> **作者:** Shanshan Wan; Lai Kang; Yingmei Wei; Tianrui Shen; Haixuan Wang; Chao Zuo
>
> **摘要:** Visual place recognition (VPR) aiming at predicting the location of an image based solely on its visual features is a fundamental task in robotics and autonomous systems. Domain variation remains one of the main challenges in VPR and is relatively unexplored. Existing VPR models attempt to achieve domain agnosticism either by training on large-scale datasets that inherently contain some domain variations, or by being specifically adapted to particular target domains. In practice, the former lacks explicit domain supervision, while the latter generalizes poorly to unseen domain shifts. This paper proposes a novel query-based domain-agnostic VPR model called QdaVPR. First, a dual-level adversarial learning framework is designed to encourage domain invariance for both the query features forming the global descriptor and the image features from which these query features are derived. Then, a triplet supervision based on query combinations is designed to enhance the discriminative power of the global descriptors. To support the learning process, we augment a large-scale VPR dataset using style transfer methods, generating various synthetic domains with corresponding domain labels as auxiliary supervision. Extensive experiments show that QdaVPR achieves state-of-the-art performance on multiple VPR benchmarks with significant domain variations. Specifically, it attains the best Recall@1 and Recall@10 on nearly all test scenarios: 93.5%/98.6% on Nordland (seasonal changes), 97.5%/99.0% on Tokyo24/7 (day-night transitions), and the highest Recall@1 across almost all weather conditions on the SVOX dataset. Our code will be released at this https URL.
>
---
#### [new 180] FedEU: Evidential Uncertainty-Driven Federated Fine-Tuning of Vision Foundation Models for Remote Sensing Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分割任务，解决联邦学习中模型适应异构数据时的不确定性问题。提出FedEU框架，通过证据不确定性建模和自适应聚合策略提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.07468](https://arxiv.org/pdf/2603.07468)**

> **作者:** Xiaokang Zhang; Xuran Xiong; Jianzhong Huang; Lefei Zhang
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Remote sensing image segmentation (RSIS) in federated environments has gained increasing attention because it enables collaborative model training across distributed datasets without sharing raw imagery or annotations. Federated RSIS combined with parameter-efficient fine-tuning (PEFT) can unleash the generalization power of pretrained foundation models for real-world applications, with minimal parameter aggregation and communication overhead. However, the dynamic adaptation of pretrained models to heterogeneous client data inevitably increases update uncertainty and compromises the reliability of collaborative optimization due to the lack of uncertainty estimation for each local model. To bridge this gap, we present FedEU, a federated optimization framework for fine-tuning RSIS models driven by evidential uncertainty. Specifically, personalized evidential uncertainty modeling is introduced to quantify epistemic variations of local models and identify high-risk areas under local data distributions. Furthermore, the client-specific feature embedding (CFE) is exploited to enhance channel-aware feature representation while preserving client-specific properties through personalized attention and an element-aware parameter update approach. These uncertainty estimates are uploaded to the server to enable adaptive global aggregation via a Top-k uncertainty-guided weighting (TUW) strategy, which mitigates the impact of distribution shifts and unreliable updates. Extensive experiments on three large-scale heterogeneous datasets demonstrate the superior performance of FedEU. More importantly, FedEU enables balanced model adaptation across diverse clients by explicitly reducing prediction uncertainty, resulting in more robust and reliable federated outcomes. The source codes will be available at this https URL.
>
---
#### [new 181] Perception-Aware Multimodal Spatial Reasoning from Monocular Images
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型在单目图像中的空间推理任务，旨在解决几何感知不足的问题。通过引入对象中心的视觉参考标记和多模态思维链数据集，提升模型的空间理解能力。**

- **链接: [https://arxiv.org/pdf/2603.06985](https://arxiv.org/pdf/2603.06985)**

> **作者:** Yanchun Cheng; Rundong Wang; Xulei Yang; Alok Prakash; Daniela Rus; Marcelo H Ang Jr; ShiJie Li
>
> **摘要:** Spatial reasoning from monocular images is essential for autonomous driving, yet current Vision-Language Models (VLMs) still struggle with fine-grained geometric perception, particularly under large scale variation and ambiguous object appearance. We propose a simple yet effective perception-aware multimodal reasoning framework that equips VLMs with explicit object-centric grounding ability. Instead of relying on textual bounding-box outputs, each referred object is represented using all Visual Reference Tokens (VRTs) within its spatial extent, enabling visual evidence and textual reasoning to be processed jointly in a unified token space. To further strengthen cross-modal interaction, we construct a Multimodal Chain-of-Thought (MM-CoT) dataset that injects aligned visual and textual reasoning signals. A deterministic ordering strategy is introduced to make supervision over inherently unordered VRT sets fully compatible with the VLM's autoregressive next-token prediction. With only standard supervised fine-tuning, our method achieves substantial improvements on the SURDS benchmark, outperforming previous approaches - including those using RL-based post-training - by a large margin across both single-object and multi-object tasks. These results demonstrate that accurate perception and multimodal reasoning are mutually reinforcing, and together form the key to robust spatial understanding in challenging monocular driving scenarios.
>
---
#### [new 182] Virtual Try-On for Cultural Clothing: A Benchmarking Study
- **分类: cs.CV**

- **简介: 该论文属于虚拟试衣任务，旨在解决现有系统对文化服饰泛化能力不足的问题。工作中构建了BD-VITON数据集，并测试了多个模型，验证了性能提升。**

- **链接: [https://arxiv.org/pdf/2603.07291](https://arxiv.org/pdf/2603.07291)**

> **作者:** Muhammad Tausif Ul Islam; Shahir Awlad; Sameen Yeaser Adib; Md. Atiqur Rahman; Sabbir Ahmed; Md. Hasanul Kabir
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Although existing virtual try-on systems have made significant progress with the advent of diffusion models, the current benchmarks of these models are based on datasets that are dominant in western-style clothing and female models, limiting their ability to generalize culturally diverse clothing styles. In this work, we introduce BD-VITON, a virtual try-on dataset focused on Bangladeshi garments, including saree, panjabi and salwar kameez, covering both male and female categories as well. These garments present unique structural challenges such as complex draping, asymmetric layering, and high deformation complexities which are underrepresented in the original VITON dataset. To establish strong baselines, we retrain and evaluate try-on models, namely StableViton, HR-VITON, and VITON-HD on our dataset. Our experiments demonstrate consistent improvements in terms of both quantitative and qualitative analysis, compared to zero shot inference.
>
---
#### [new 183] Listening with the Eyes: Benchmarking Egocentric Co-Speech Grounding across Space and Time
- **分类: cs.CV**

- **简介: 该论文提出EcoG-Bench，用于评估多模态大模型在时空联合定位任务中的表现，解决语言与视觉对齐不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.07966](https://arxiv.org/pdf/2603.07966)**

> **作者:** Weijie Zhou; Xuantang Xiong; Zhenlin Hu; Xiaomeng Zhu; Chaoyang Zhao; Honghui Dong; Zhengyou Zhang; Ming Tang; Jinqiao Wang
>
> **摘要:** In situated collaboration, speakers often use intentionally underspecified deictic commands (e.g., ``pass me \textit{that}''), whose referent becomes identifiable only by aligning speech with a brief co-speech pointing \emph{stroke}. However, many embodied benchmarks admit language-only shortcuts, allowing MLLMs to perform well without learning the \emph{audio--visual alignment} required by deictic interaction. To bridge this gap, we introduce \textbf{Egocentric Co-Speech Grounding (EcoG)}, where grounding is executable only if an agent jointly predicts \textit{What}, \textit{Where}, and \textit{When}. To operationalize this, we present \textbf{EcoG-Bench}, an evaluation-only bilingual (EN/ZH) diagnostic benchmark of \textbf{811} egocentric clips with dense spatial annotations and millisecond-level stroke supervision. It is organized under a \textbf{Progressive Cognitive Evaluation} protocol. Benchmarking state-of-the-art MLLMs reveals a severe executability gap: while human subjects achieve near-ceiling performance on EcoG-Bench (\textbf{96.9\%} strict Eco-Accuracy), the best native video-audio setting remains low (Gemini-3-Pro: \textbf{17.0\%}). Moreover, in a diagnostic ablation, replacing the native video--audio interface with timestamped frame samples and externally verified ASR (with word-level timing) substantially improves the same model (\textbf{17.0\%}$\to$\textbf{42.9\%}). Overall, EcoG-Bench provides a strict, executable testbed for event-level speech--gesture binding, and suggests that multimodal interfaces may bottleneck the observability of temporal alignment cues, independently of model reasoning.
>
---
#### [new 184] Toward Unified Multimodal Representation Learning for Autonomous Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态表示学习任务，旨在解决自动驾驶中多模态对齐不一致的问题。提出CTP框架，通过张量损失实现多模态统一对齐。**

- **链接: [https://arxiv.org/pdf/2603.07874](https://arxiv.org/pdf/2603.07874)**

> **作者:** Ximeng Tao; Dimitar Filev; Gaurav Pandey
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) has shown impressive performance in aligning visual and textual representations. Recent studies have extended this paradigm to 3D vision to improve scene understanding for autonomous driving. A common strategy is to employ pairwise cosine similarity between modalities to guide the training of a 3D encoder. However, considering the similarity between individual modality pairs rather than all modalities jointly fails to ensure consistent and unified alignment across the entire multimodal space. In this paper, we propose a Contrastive Tensor Pre-training (CTP) framework that simultaneously aligns multiple modalities in a unified embedding space to enhance end-to-end autonomous driving. Compared with pairwise cosine similarity alignment, our method extends the 2D similarity matrix into a multimodal similarity tensor. Furthermore, we introduce a tensor loss to enable joint contrastive learning across all modalities. For experimental validation of our framework, we construct a text-image-point cloud triplet dataset derived from existing autonomous driving datasets. The results show that our proposed unified multimodal alignment framework achieves favorable performance for both scenarios: (i) aligning a 3D encoder with pretrained CLIP encoders, and (ii) pretraining all encoders from scratch.
>
---
#### [new 185] Inter-Image Pixel Shuffling for Multi-focus Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于多聚焦图像融合任务，旨在解决训练数据不足的问题。提出IPS方法，通过像素混洗生成训练数据，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2603.07120](https://arxiv.org/pdf/2603.07120)**

> **作者:** Huangxing Lin; Rongrong Ma; Cheng Wang
>
> **摘要:** Multi-focus image fusion aims to combine multiple partially focused images into a single all-in-focus image. Although deep learning has shown promise in this task, its effectiveness is often limited by the scarcity of suitable training data. This paper introduces Inter-image Pixel Shuffling (IPS), a novel method that allows neural networks to learn multi-focus image fusion without requiring actual multi-focus images. IPS reformulates the task as a pixel-wise classification problem, where the goal is to identify the focused pixel from a pixel group at each spatial position. In this method, pixels from a clear optical image are treated as focused, while pixels from a low-pass filtered version of the same image are considered defocused. By randomly shuffling the focused and defocused pixels at identical spatial positions in the original and filtered images, IPS generates training data that preserves spatial structure while mixing focus-defocus information. The model is trained to select the focused pixel from each spatially aligned pixel group, thus learning to reconstruct an all-in-focus image by aggregating sharp content from the input. To further enhance fusion quality, IPS adopts a cross-image fusion network that integrates the localized representation power of convolutional neural networks with the long-range modeling capabilities of state space models. This design effectively leverages both spatial detail and contextual information to produce high-quality fused results. Experimental results indicate that IPS significantly outperforms existing multi-focus image fusion methods, even without training on multi-focus images.
>
---
#### [new 186] Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction
- **分类: cs.CV; cs.GR; cs.RO; eess.IV**

- **简介: 该论文属于3D场景重建任务，旨在解决全景图像中3D高斯点云渲染的几何不一致问题。提出Spherical-GOF框架，实现更精确的全景渲染与深度估计。**

- **链接: [https://arxiv.org/pdf/2603.08503](https://arxiv.org/pdf/2603.08503)**

> **作者:** Zhe Yang; Guoqiang Zhao; Sheng Wu; Kai Luo; Kailun Yang
>
> **备注:** The source code and dataset will be released at this https URL
>
> **摘要:** Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at this https URL.
>
---
#### [new 187] Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究文本到视频生成任务，探讨语义噪声初始化是否适用于视频扩散模型。通过对比实验发现其在时间维度有微弱提升，但整体效果与基线相当。**

- **链接: [https://arxiv.org/pdf/2603.06672](https://arxiv.org/pdf/2603.06672)**

> **作者:** Yixiao Jing; Chaoyu Zhang; Zixuan Zhong; Peizhou Huang
>
> **备注:** 8 pages, 1 figure. Accepted to the ICLR 2026 Workshop on Multimodal Intelligence: Next Token Prediction & Beyond
>
> **摘要:** Semantic noise initialization has been reported to improve robustness and controllability in image diffusion models. Whether these gains transfer to text-to-video (T2V) generation remains unclear, since temporal coupling can introduce extra degrees of freedom and instability. We benchmark semantic noise initialization against standard Gaussian noise using a frozen VideoCrafter-style T2V diffusion backbone and VBench on 100 prompts. Using prompt-level paired tests with bootstrap confidence intervals and a sign-flip permutation test, we observe a small positive trend on temporal-related dimensions; however, the 95 percent confidence interval includes zero (p ~ 0.17) and the overall score remains on par with the baseline. To understand this outcome, we analyze the induced perturbations in noise space and find patterns consistent with weak or unstable signal. We recommend prompt-level paired evaluation and noise-space diagnostics as standard practice when studying initialization schemes for T2V diffusion.
>
---
#### [new 188] Can Vision-Language Models Solve the Shell Game?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决视频中物体跟踪问题。针对现有模型依赖静态特征、无法持续跟踪的问题，提出SGCoT方法，通过生成物体轨迹提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.08436](https://arxiv.org/pdf/2603.08436)**

> **作者:** Tiedong Liu; Wee Sun Lee
>
> **摘要:** Visual entity tracking is an innate cognitive ability in humans, yet it remains a critical bottleneck for Vision-Language Models (VLMs). This deficit is often obscured in existing video benchmarks by visual shortcuts. We introduce VET-Bench, a synthetic diagnostic testbed featuring visually identical objects that necessitate tracking exclusively through spatiotemporal continuity. Our experiments reveal that current state-of-the-art VLMs perform at or near chance level on VET-Bench, exposing a fundamental limitation: an over-reliance on static frame-level features and a failure to maintain entity representations over time. We provide a theoretical analysis drawing connections to the state-tracking problem, proving that fixed-depth transformer-based VLMs are fundamentally limited in tracking indistinguishable objects without intermediate supervision due to expressivity constraints. To address this, we propose Spatiotemporal Grounded Chain-of-Thought (SGCoT): generating object trajectories as explicit intermediate states. Leveraging Molmo2's object tracking ability, we elicit SGCoT reasoning by fine-tuning on synthesized text-only data for alignment. Our method achieves state-of-the-art accuracy exceeding 90% on VET-Bench, demonstrating that VLMs can reliably solve the video shell-game task end-to-end without external tools. Our code and data are available at this https URL .
>
---
#### [new 189] Asymmetric Distillation and Information Retention in Capacity-Constrained Cross-Modal Transfer
- **分类: cs.CV**

- **简介: 该论文研究跨模态迁移中的容量约束问题，探讨教师模型到学生模型的知识蒸馏过程。任务为知识蒸馏，解决几何约束与信息丢失问题，通过分析有效秩和信息理论揭示容量与鲁棒性的关系。**

- **链接: [https://arxiv.org/pdf/2603.06698](https://arxiv.org/pdf/2603.06698)**

> **作者:** Kabir Thayani
>
> **备注:** 3 pages, 3 figures, 1 table
>
> **摘要:** Knowledge distillation between asymmetric architectures often induces severe geometric constraints on the learned representation space. In this work, we investigate the Dimensional Collapse phenomenon when distilling a 500M parameter global Vision Transformer (CLIP ViT-B/32) into strictly capacity-constrained, local-receptive-field CNNs (0.5M to 8.0M parameters) on the CIFAR-10 dataset. By employing strictly centered Singular Value Decomposition (SVD) and Variance-based Shannon Entropy Effective Rank, we isolate true structural variance from mean-vector artifacts. Our empirical results demonstrate a capacity-agnostic phase transition: while the Teacher exhibits an Effective Rank of 88.68, all Student models experience severe dimensional collapse to an intrinsic Effective Rank of ~16. By probing robustness, we uncover that this 81% reduction in effective dimensionality strips away the Teacher's inherent noise immunity (which retains 89.35% accuracy under \sigma=0.1 Gaussian noise). Furthermore, information-theoretic analysis using InfoNCE reveals a critical trade-off within this bottleneck: excess Student capacity densely packs the collapsed subspace for clean data, but induces severe brittleness (43.76% at \sigma=0.1). Conversely, extreme capacity constraints (0.5M parameters) act as a robust low-pass filter, preserving higher noise immunity (54.84%). Explicit input augmentation fails to restore the larger model's robustness, proving this fragility is a fundamental geometric limitation of asymmetric cosine distillation.
>
---
#### [new 190] mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于人体姿态估计任务，旨在解决传统方法在隐私和低光环境下的不足，通过mmWave雷达点云与图注意力网络结合，提升姿态估计精度。**

- **链接: [https://arxiv.org/pdf/2603.08551](https://arxiv.org/pdf/2603.08551)**

> **作者:** Abdullah Al Masud; Shi Xintong; Mondher Bouazizi; Ohtsuki Tomoaki
>
> **备注:** copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.
>
---
#### [new 191] HybridStitch: Pixel and Timestep Level Model Stitching for Diffusion Acceleration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型计算开销大的问题。提出HybridStitch方法，通过混合大小模型提升生成速度。**

- **链接: [https://arxiv.org/pdf/2603.07815](https://arxiv.org/pdf/2603.07815)**

> **作者:** Desen Sun; Jason Hon; Jintao Zhang; Sihang Liu
>
> **摘要:** Diffusion models have demonstrated a remarkable ability in Text-to-Image (T2I) generation applications. Despite the advanced generation output, they suffer from heavy computation overhead, especially for large models that contain tens of billions of parameters. Prior work has illustrated that replacing part of the denoising steps with a smaller model still maintains the generation quality. However, these methods only focus on saving computation for some timesteps, ignoring the difference in compute demand within one timestep. In this work, we propose HybridStitch, a new T2I generation paradigm that treats generation like editing. Specifically, we introduce a hybrid stage that jointly incorporates both the large model and the small model. HybridStitch separates the entire image into two regions: one that is relatively easy to render, enabling an early transition to the smaller model, and another that is more complex and therefore requires refinement by the large model. HybridStitch employs the small model to construct a coarse sketch while exploiting the large model to edit and refine the complex regions. According to our evaluation, HybridStitch achieves 1.83$\times$ speedup on Stable Diffusion 3, which is faster than all existing mixture of model methods.
>
---
#### [new 192] ColonSplat: Reconstruction of Peristaltic Motion in Colonoscopy with Dynamic Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决结肠镜检查中复杂蠕动运动的重建问题。提出ColonSplat框架，实现更精确的几何一致性与运动模拟。**

- **链接: [https://arxiv.org/pdf/2603.06860](https://arxiv.org/pdf/2603.06860)**

> **作者:** Weronika Smolak-Dyżewska; Joanna Kaleta; Diego Dall'Alba; Przemysław Spurek
>
> **摘要:** Accurate 3D reconstruction of colonoscopy data, accounting for complex peristaltic movements, is crucial for advanced surgical navigation and retrospective diagnostics. While recent novel view synthesis and 3D reconstruction methods have demonstrated remarkable success in general endoscopic scenarios, they struggle in the highly constrained environment of the colon. Due to the limited field of view of a camera moving through an actively deforming tubular structure, existing endoscopic methods reconstruct the colon appearance only for initial camera trajectory. However, the underlying anatomy remains largely static; instead of updating Gaussians' spatial coordinates (xyz), these methods encode deformation through either rotation, scale or opacity adjustments. In this paper, we first present a benchmark analysis of state-of-the-art dynamic endoscopic methods for realistic colonoscopic scenes, showing that they fail to model true anatomical motion. To enable rigorous evaluation of global reconstruction quality, we introduce DynamicColon, a synthetic dataset with ground-truth point clouds at every timestep. Building on these insights, we propose ColonSplat, a dynamic Gaussian Splatting framework that captures peristaltic-like motion while preserving global geometric consistency, achieving superior geometric fidelity on C3VDv2 and DynamicColon datasets. Project page: this https URL
>
---
#### [new 193] Aligning What EEG Can See: Structural Representations for Brain-Vision Matching
- **分类: cs.CV**

- **简介: 该论文属于脑-视觉匹配任务，旨在解决EEG与视觉信息间的跨模态不匹配问题。通过引入神经可见性概念和分层融合框架，提升视觉解码准确率。**

- **链接: [https://arxiv.org/pdf/2603.07077](https://arxiv.org/pdf/2603.07077)**

> **作者:** Jingyi Tang; Shuai Jiang; Fei Su; Zhicheng Zhao
>
> **摘要:** Visual decoding from electroencephalography (EEG) has emerged as a highly promising avenue for non-invasive brain-computer interfaces (BCIs). Existing EEG-based decoding methods predominantly align brain signals with the final-layer semantic embeddings of deep visual models. However, relying on these highly abstracted embeddings inevitably leads to severe cross-modal information mismatch. In this work, we introduce the concept of Neural Visibility and accordingly propose the EEG-Visible Layer Selection Strategy, aligning EEG signals with intermediate visual layers to minimize this mismatch. Furthermore, to accommodate the multi-stage nature of human visual processing, we propose a novel Hierarchically Complementary Fusion (HCF) framework that jointly integrates visual representations from different hierarchical levels. Extensive experiments demonstrate that our method achieves state-of-the-art performance, reaching an 84.6% accuracy (+21.4%) on zero-shot visual decoding on the THINGS-EEG dataset. Moreover, our method achieves up to a 129.8% performance gain across diverse EEG baselines, demonstrating its robust generalizability.
>
---
#### [new 194] CanoVerse: 3D Object Scalable Canonicalization and Dataset for Generation and Pose
- **分类: cs.CV**

- **简介: 该论文提出CanoVerse，解决3D物体方向对齐问题，构建大规模标准化3D数据集，提升生成稳定性与跨模态检索性能。**

- **链接: [https://arxiv.org/pdf/2603.07144](https://arxiv.org/pdf/2603.07144)**

> **作者:** Li Jin; Yuchen Yang; Weikai Chen; Yujie Wang; Dehao Hao; Tanghui Jia; Yingda Yin; Zeyu Hu; Runze Zhang; Keyang Luo; Li Yuan; Long Quan; Xin Wang; Xueying Qin
>
> **摘要:** 3D learning systems implicitly assume that objects occupy a coherent reference frame. Nonetheless, in practice, every asset arrives with an arbitrary global rotation, and models are left to resolve directional ambiguity on their own. This persistent misalignment suppresses pose-consistent generation, and blocks the emergence of stable directional semantics. To address this issue, we construct \methodName{}, a massive canonical 3D dataset of 320K objects over 1,156 categories -- an order-of-magnitude increase over prior work. At this scale, directional semantics become statistically learnable: Canoverse improves 3D generation stability, enables precise cross-modal 3D shape retrieval, and unlocks zero-shot point-cloud orientation estimation even for out-of-distribution data. This is achieved by a new canonicalization framework that reduces alignment from minutes to seconds per object via compact hypothesis generation and lightweight human discrimination, transforming canonicalization from manual curation into a high-throughput data generation pipeline. The Canoverse dataset will be publicly released upon acceptance. Project page: this https URL
>
---
#### [new 195] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size
- **分类: cs.CV; cs.GR; cs.MA; cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决多人类-物体交互的协同控制问题。通过统一策略实现不同规模团队的协作，提升运动真实性和任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.07988](https://arxiv.org/pdf/2603.07988)**

> **作者:** Stefan Lionar; Gim Hee Lee
>
> **备注:** CVPR 2026. Project page: this https URL Code: this https URL
>
> **摘要:** Physics-based humanoid control has achieved remarkable progress in enabling realistic and high-performing single-agent behaviors, yet extending these capabilities to cooperative human-object interaction (HOI) remains challenging. We present TeamHOI, a framework that enables a single decentralized policy to handle cooperative HOIs across any number of cooperating agents. Each agent operates using local observations while attending to other teammates through a Transformer-based policy network with teammate tokens, allowing scalable coordination across variable team sizes. To enforce motion realism while addressing the scarcity of cooperative HOI data, we further introduce a masked Adversarial Motion Prior (AMP) strategy that uses single-human reference motions while masking object-interacting body parts during training. The masked regions are then guided through task rewards to produce diverse and physically plausible cooperative behaviors. We evaluate TeamHOI on a challenging cooperative carrying task involving two to eight humanoid agents and varied object geometries. Finally, to promote stable carrying, we design a team-size- and shape-agnostic formation reward. TeamHOI achieves high success rates and demonstrates coherent cooperation across diverse configurations with a single policy.
>
---
#### [new 196] Scale-Aware UAV-to-Satellite Cross-View Geo-Localization: A Semantic Geometric Approach
- **分类: cs.CV**

- **简介: 该论文属于跨视角地理定位任务，解决UAV与卫星图像间尺度不一致导致的定位误差问题。通过语义几何方法估计绝对尺度，提升定位鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07535](https://arxiv.org/pdf/2603.07535)**

> **作者:** Yibin Ye; Shuo Chen; Kun Wang; Xiaokai Song; Jisheng Dang; Qifeng Yu; Xichao Teng; Zhang Li
>
> **备注:** 14 pages
>
> **摘要:** Cross-View Geo-Localization (CVGL) between UAV imagery and satellite images plays a crucial role in target localization and UAV self-positioning. However, most existing methods rely on the idealized assumption of scale consistency between UAV queries and satellite galleries, overlooking the severe scale ambiguity commonly encountered in real-world scenarios. This discrepancy leads to field-of-view misalignment and feature mismatch, significantly degrading CVGL robustness. To address this issue, we propose a geometric framework that recovers the absolute metric scale from monocular UAV images using semantic anchors. Specifically, small vehicles (SVs), characterized by relatively stable prior size distributions and high detectability, are exploited as metric references. A Decoupled Stereoscopic Projection Model is introduced to estimate the absolute image scale from these semantic targets. By decomposing vehicle dimensions into radial and tangential components, the model compensates for perspective distortions in 2D detections of 3D vehicles, enabling more accurate scale estimation. To further reduce intra-class size variation and detection noise, a dual-dimension fusion strategy with Interquartile Range (IQR)-based robust aggregation is employed. The estimated global scale is then used as a physical constraint for scale-adaptive satellite image cropping, improving UAV-to-satellite feature alignment. Experiments on augmented DenseUAV and UAV-VisLoc datasets demonstrate that the proposed method significantly improves CVGL robustness under unknown UAV image scales. Additionally, the framework shows strong potential for downstream applications such as passive UAV altitude estimation and 3D model scale recovery.
>
---
#### [new 197] VB: Visibility Benchmark for Visibility and Perspective Reasoning in Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VB基准，用于评估视觉语言模型在图像中判断可见性及视角推理的能力，解决模型对不可见内容的识别与回避问题。**

- **链接: [https://arxiv.org/pdf/2603.06680](https://arxiv.org/pdf/2603.06680)**

> **作者:** Neil Tripathi
>
> **备注:** 18 pages, 1 figure, 3 tables. Code and data: this https URL
>
> **摘要:** We present VB, a benchmark that tests whether vision-language models can determine what is and is not visible in a photograph, and abstain when a human viewer cannot reliably answer. Each item pairs a single photo with a short yes/no visibility claim; the model must output VISIBLY_TRUE, VISIBLY_FALSE, or ABSTAIN, together with a confidence score. Items are organized into 100 families using a 2x2 design that crosses a minimal image edit with a minimal text edit, yielding 300 headline evaluation cells. Unlike prior unanswerable-VQA benchmarks, VB tests not only whether a question is unanswerable but why (via reason codes tied to specific visibility factors), and uses controlled minimal edits to verify that model judgments change when and only when the underlying evidence changes. We score models on confidence-aware accuracy with abstention (CAA), minimal-edit flip rate (MEFR), confidence-ranked selective prediction (SelRank), and second-order perspective reasoning (ToMAcc); all headline numbers are computed on the strict XOR subset (three cells per family, 300 scored items per model). We evaluate nine models spanning flagship and prior-generation closed-source systems, and open-source models from 8B to 12B parameters. GPT-4o and Gemini 3.1 Pro effectively tie for the best composite score (0.728 and 0.727), followed by Gemini 2.5 Pro (0.678). The best open-source model, Gemma 3 12B (0.505), surpasses one prior-generation closed-source system. Text-flip robustness exceeds image-flip robustness for six of nine models, and confidence calibration varies substantially: GPT-4o and Gemini 2.5 Pro achieve similar accuracy yet differ sharply in selective prediction quality.
>
---
#### [new 198] How Long Can Unified Multimodal Models Generate Images Reliably? Taming Long-Horizon Interleaved Image Generation via Context Curation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态生成任务，解决长序列生成质量下降问题。通过动态筛选视觉信息，提升生成稳定性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.07540](https://arxiv.org/pdf/2603.07540)**

> **作者:** Haoyu Chen; Qing Liu; Yuqian Zhou; He Zhang; Zhaowen Wang; Mengwei Ren; Jingjing Ren; Xiang Wang; Zhe Lin; Lei Zhu
>
> **摘要:** Unified multimodal models hold the promise of generating extensive, interleaved narratives, weaving text and imagery into coherent long-form stories. However, current systems suffer from a critical reliability gap: as sequences grow, generation quality rapidly collapses. In this work, we investigate the mechanism behind this failure and argue that it is distinct from standard long-context challenges. We reveal that in generation, accumulated visual history acts as a source of active pollution, a decay governed specifically by the number of image events rather than raw token count. We identify a structural vulnerability where dense visual tokens overwhelm the attention mechanism, creating noise that distorts future synthesis. Guided by these mechanistic insights, we propose UniLongGen, a training-free inference strategy that prioritizes safe conditioning over total recall. Instead of retaining all history, UniLongGen dynamically curates the model's memory, identifying and discarding interfering visual signals based on the model's own internal relevance rankings. Extensive experiments demonstrate that this active forgetting approach is essential for stability: UniLongGen significantly outperforms baselines in long-horizon fidelity and consistency, while simultaneously reducing memory footprint and inference time.
>
---
#### [new 199] MINT: Molecularly Informed Training with Spatial Transcriptomics Supervision for Pathology Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出MINT框架，将空间转录组数据融入病理学基础模型训练，解决传统方法忽略分子信息的问题，提升病理分析性能。**

- **链接: [https://arxiv.org/pdf/2603.07895](https://arxiv.org/pdf/2603.07895)**

> **作者:** Minsoo Lee; Jonghyun Kim; Juseung Yun; Sunwoo Yu; Jongseong Jang
>
> **摘要:** Pathology foundation models learn morphological representations through self-supervised pretraining on large-scale whole-slide images, yet they do not explicitly capture the underlying molecular state of the tissue. Spatial transcriptomics technologies bridge this gap by measuring gene expression in situ, offering a natural cross-modal supervisory signal. We propose MINT (Molecularly Informed Training), a fine-tuning framework that incorporates spatial transcriptomics supervision into pretrained pathology Vision Transformers. MINT appends a learnable ST token to the ViT input to encode transcriptomic information separately from the morphological CLS token, preventing catastrophic forgetting through DINO self-distillation and explicit feature anchoring to the frozen pretrained encoder. Gene expression regression at both spot-level (Visium) and patch-level (Xenium) resolutions provides complementary supervision across spatial scales. Trained on 577 publicly available HEST samples, MINT achieves the best overall performance on both HEST-Bench for gene expression prediction (mean Pearson r = 0.440) and EVA for general pathology tasks (0.803), demonstrating that spatial transcriptomics supervision complements morphology-centric self-supervised pretraining.
>
---
#### [new 200] Speed3R: Sparse Feed-forward 3D Reconstruction Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D重建任务，旨在解决传统模型计算复杂度高的问题。通过引入稀疏注意力机制，提升推理速度并保持重建质量。**

- **链接: [https://arxiv.org/pdf/2603.08055](https://arxiv.org/pdf/2603.08055)**

> **作者:** Weining Ren; Xiao Tan; Kai Han
>
> **备注:** CVPR 2026 Findings, project page: this https URL
>
> **摘要:** While recent feed-forward 3D reconstruction models accelerate 3D reconstruction by jointly inferring dense geometry and camera poses in a single pass, their reliance on dense attention imposes a quadratic complexity, creating a prohibitive computational bottleneck that severely limits inference speed. To resolve this, we introduce Speed3R, an end-to-end trainable model inspired by the core principle of Structure-from-Motion: that a sparse set of keypoints is sufficient for robust pose estimation. Speed3R features a dual-branch attention mechanism where a compression branch creates a coarse contextual prior to guide a selection branch, which performs fine-grained attention only on the most informative image tokens. This strategy mimics the efficiency of traditional keypoint matching, achieving a remarkable 12.4x inference speedup on 1000-view sequences, while introducing a minimal, controlled trade-off in geometric accuracy. Validated on standard benchmarks with both VGGT and $\pi^3$ backbones, our method delivers high-quality reconstructions at a fraction of computational cost, paving the way for efficient large-scale scene modeling.
>
---
#### [new 201] Learning Context-Adaptive Motion Priors for Masked Motion Diffusion Models with Efficient Kinematic Attention Aggregation
- **分类: cs.CV**

- **简介: 该论文提出MMDM模型，解决运动捕捉中的遮挡和数据不稳定问题。通过引入KAA机制，学习上下文自适应的运动先验，提升运动数据的重建效果。**

- **链接: [https://arxiv.org/pdf/2603.07697](https://arxiv.org/pdf/2603.07697)**

> **作者:** Junkun Jiang; Jie Chen; Ho Yin Au; Jingyu Xiang
>
> **备注:** Accepted by IEEE Transactions on Multimedia. Supplementary material is included
>
> **摘要:** Vision-based motion capture solutions often struggle with occlusions, which result in the loss of critical joint information and hinder accurate 3D motion reconstruction. Other wearable alternatives also suffer from noisy or unstable data, often requiring extensive manual cleaning and correction to achieve reliable results. To address these challenges, we introduce the Masked Motion Diffusion Model (MMDM), a diffusion-based generative reconstruction framework that enhances incomplete or low-confidence motion data using partially available high-quality reconstructions within a Masked Autoencoder architecture. Central to our design is the Kinematic Attention Aggregation (KAA) mechanism, which enables efficient, deep, and iterative encoding of both joint-level and pose-level features, capturing structural and temporal motion patterns essential for task-specific reconstruction. We focus on learning context-adaptive motion priors, specialized structural and temporal features extracted by the same reusable architecture, where each learned prior emphasizes different aspects of motion dynamics and is specifically efficient for its corresponding task. This enables the architecture to adaptively specialize without altering its structure. Such versatility allows MMDM to efficiently learn motion priors tailored to scenarios such as motion refinement, completion, and in-betweening. Extensive evaluations on public benchmarks demonstrate that MMDM achieves strong performance across diverse masking strategies and task settings. The source code is available at this https URL.
>
---
#### [new 202] Looking Back and Forth: Cross-Image Attention Calibration and Attentive Preference Learning for Multi-Image Hallucination Mitigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多图像任务，旨在解决大视觉语言模型的幻觉问题。提出CAPL框架，通过跨图像注意力校准和偏好学习增强图像间关联建模，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2603.07048](https://arxiv.org/pdf/2603.07048)**

> **作者:** Xiaochen Yang; Hao Fang; Jiawei Kong; Yaoxin Mao; Bin Chen; Shu-Tao Xia
>
> **摘要:** Although large vision-language models (LVLMs) have demonstrated remarkable capabilities, they are prone to hallucinations in multi-image tasks. We attribute this issue to limitations in existing attention mechanisms and insufficient cross-image modeling. Inspired by this, we propose a structured hallucination mitigation framework involving Cross-Image Attention calibration and Preference Learning (CAPL). CAPL explicitly enhances inter-image interactions at the architectural level while reinforcing reliance on genuine cross-image evidence during training, thereby improving the model's perception and modeling of cross-image associations. Specifically, we (i) introduce a selectable image token interaction attention mechanism to establish fine-grained cross-image entity alignment and information flow; (ii) design a cross-image modeling-based preference optimization strategy that contrasts reasoning outcomes under full inter-image interaction and those obtained when images are mutually invisible, encouraging the model to ground its predictions in authentic visual evidence and mitigating erroneous inferences driven by textual priors. Experimental results demonstrate that CAPL consistently improves performance across multiple model architectures, achieving stable gains on both multi-image hallucination and general benchmarks. Notably, performance on single-image visual tasks remains stable or slightly improves, indicating strong generalization capability.
>
---
#### [new 203] SLNet: A Super-Lightweight Geometry-Adaptive Network for 3D Point Cloud Recognition
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出SLNet，用于3D点云识别任务，旨在以轻量模型实现高精度。通过NAPE和GMU结构，在保持性能的同时显著降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.07454](https://arxiv.org/pdf/2603.07454)**

> **作者:** Mohammad Saeid; Amir Salarpour; Pedram MohajerAnsari; Mert D. Pesé
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** We present SLNet, a lightweight backbone for 3D point cloud recognition designed to achieve strong performance without the computational cost of many recent attention, graph, and deep MLP based models. The model is built on two simple ideas: NAPE (Nonparametric Adaptive Point Embedding), which captures spatial structure using a combination of Gaussian RBF and cosine bases with input adaptive bandwidth and blending, and GMU (Geometric Modulation Unit), a per channel affine modulator that adds only 2D learnable parameters. These components are used within a four stage hierarchical encoder with FPS+kNN grouping, nonparametric normalization, and shared residual MLPs. In experiments, SLNet shows that a very small model can still remain highly competitive across several 3D recognition tasks. On ModelNet40, SLNet-S with 0.14M parameters and 0.31 GFLOPs achieves 93.64% overall accuracy, outperforming PointMLP-elite with 5x fewer parameters, while SLNet-M with 0.55M parameters and 1.22 GFLOPs reaches 93.92%, exceeding PointMLP with 24x fewer parameters. On ScanObjectNN, SLNet-M achieves 84.25% overall accuracy within 1.2 percentage points of PointMLP while using 28x fewer parameters. For large scale scene segmentation, SLNet-T extends the backbone with local Point Transformer attention and reaches 58.2% mIoU on S3DIS Area 5 with only 2.5M parameters, more than 17x fewer than Point Transformer V3. We also introduce NetScore+, which extends NetScore by incorporating latency and peak memory so that efficiency can be evaluated in a more deployment oriented way. Across multiple benchmarks and hardware settings, SLNet delivers a strong overall balance between accuracy and efficiency. Code is available at: this https URL.
>
---
#### [new 204] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决 colorectal 癌病理中腺体结构分割的问题。通过弱监督框架，利用少量标注和伪掩码优化，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2603.08605](https://arxiv.org/pdf/2603.08605)**

> **作者:** Hikmat Khan; Wei Chen; Muhammad Khalid Khan Niazi
>
> **摘要:** Background and objectives: Colorectal cancer histopathological grading depends on accurate segmentation of glandular structures. Current deep learning approaches rely on large scale pixel level annotations that are labor intensive and difficult to obtain in routine clinical practice. Weakly supervised semantic segmentation offers a promising alternative. However, class activation map based methods often produce incomplete pseudo masks that emphasize highly discriminative regions and fail to supervise unannotated glandular structures. We propose a weakly supervised teacher student framework that leverages sparse pathologist annotations and an Exponential Moving Average stabilized teacher network to generate refined pseudo masks. Methods: The framework integrates confidence based filtering, adaptive fusion of teacher predictions with limited ground truth, and curriculum guided refinement to progressively segment unannotated glandular regions. The method was evaluated on an institutional colorectal cancer cohort from The Ohio State University Wexner Medical Center consisting of 60 hematoxylin and eosin stained whole slide images and on public datasets including the Gland Segmentation dataset, TCGA COAD, TCGA READ, and SPIDER. Results: On the Gland Segmentation dataset the framework achieved a mean Intersection over Union of 80.10 and a mean Dice coefficient of 89.10. Cross cohort evaluation demonstrated robust generalization on TCGA COAD and TCGA READ without additional annotations, while reduced performance on SPIDER reflected domain shift. Conclusions: The proposed framework provides an annotation efficient and generalizable approach for gland segmentation in colorectal histopathology.
>
---
#### [new 205] High-Resolution Image Reconstruction with Unsupervised Learning and Noisy Data Applied to Ion-Beam Dynamics for Particle Accelerators
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像重建任务，解决高能物理加速器中噪声数据下的图像清晰化问题。通过无监督学习方法提升束流晕结构的分辨率。**

- **链接: [https://arxiv.org/pdf/2603.06689](https://arxiv.org/pdf/2603.06689)**

> **作者:** Francis Osswald; Mohammed Chahbaoui; Xinyi Liang
>
> **摘要:** Image reconstruction in the presence of severe degradation remains a challenging inverse problem, particularly in beam diagnostics for high-energy physics accelerators. As modern facilities demand precise detection of beam halo structures to control losses, traditional analysis tools have reached their performance limits. This work reviews existing image-processing techniques for data cleaning, contour extraction, and emittance reconstruction, and introduces a novel approach based on convolutional filtering and neural networks with optimized early-stopping strategies in order to control overfitting. Despite the absence of training datasets, the proposed unsupervised framework achieves robust denoising and high-fidelity reconstruction of beam emittance images under low signal-to-noise conditions. The method extends measurable amplitudes beyond seven standard deviations, enabling unprecedented halo resolution.
>
---
#### [new 206] HARP: HARmonizing in-vivo diffusion MRI using Phantom-only training
- **分类: cs.CV**

- **简介: 该论文属于扩散MRI数据标准化任务，旨在解决多扫描仪数据差异问题。通过使用扩散幻像数据训练模型，减少扫描仪间差异，提升数据一致性。**

- **链接: [https://arxiv.org/pdf/2603.06696](https://arxiv.org/pdf/2603.06696)**

> **作者:** Hwihun Jeong; Qiang Liu; Kathryn E. Keenan; Elisabeth A. Wilde; Walter Schneider; Sudhir Pathak; Anthony Zuccolotto; Lauren J. O'Donnell; Lipeng Ning; Yogesh Rathi
>
> **摘要:** Purpose: Combining multi-site diffusion MRI (dMRI) data is hindered by inter-scanner variability, which confounds subsequent analysis. Previous harmonization methods require large, matched or traveling human subjects from multiple sites, which are impractical to acquire in many situations. This study aims to develop a deep learning-based dMRI harmonization framework that eliminates the reliance on multi-site in-vivo traveling human data for training. Methods: HARP employs a voxel-wise 1D neural network trained on an easily transportable diffusion phantom. The model learns relationships between spherical harmonics coefficients of different sites without memorizing spatial structures. Results: HARP reduced inter-scanner variability levels significantly in various measures. Quantitatively, it decreased inter-scanner variability as measured by standard error in FA (12%), MD (10%), and GFA (30%) with scan-rescan standard error as the baseline, while preserving fiber orientations and tractography after harmonization. Conclusion: We believe that HARP represents an important first step toward dMRI harmonization using only phantom data, thereby obviating the need for complex, matched in vivo multi-site cohorts. This phantom-only strategy substantially enhances the feasibility and scalability of quantitative dMRI for large-scale clinical studies.
>
---
#### [new 207] All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference
- **分类: cs.CV**

- **简介: 该论文属于协同感知任务，解决完全不可信车辆环境下的对抗攻击问题。提出PRBI框架，通过时间差异和贝叶斯推理高效检测恶意车辆。**

- **链接: [https://arxiv.org/pdf/2603.08498](https://arxiv.org/pdf/2603.08498)**

> **作者:** Yi Yu; Libing Wu; Zhuangzhuang Zhang; Jing Qiu; Lijuan Huo; Jiaqi Feng
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Collaborative perception (CP) enables multiple vehicles to augment their individual perception capacities through the exchange of feature-level sensory data. However, this fusion mechanism is inherently vulnerable to adversarial attacks, especially in fully untrusted-vehicle environments. Existing defense approaches often assume a trusted ego vehicle as a reference or incorporate additional binary classifiers. These assumptions limit their practicality in real-world deployments due to the questionable trustworthiness of ego vehicles, the requirement for real-time detection, and the need for generalizability across diverse scenarios. To address these challenges, we propose a novel Pseudo-Random Bayesian Inference (PRBI) framework, a first efficient defense method tailored for fully untrusted-vehicle CP. PRBI detects adversarial behavior by leveraging temporal perceptual discrepancies, using the reliable perception from the preceding frame as a dynamic reference. Additionally, it employs a pseudo-random grouping strategy that requires only two verifications per frame, while applying Bayesian inference to estimate both the number and identities of malicious vehicles. Theoretical analysis has proven the convergence and stability of the proposed PRBI framework. Extensive experiments show that PRBI requires only 2.5 verifications per frame on average, outperforming existing methods significantly, and restores detection precision to between 79.4% and 86.9% of pre-attack levels.
>
---
#### [new 208] ALOOD: Exploiting Language Representations for LiDAR-based Out-of-Distribution Object Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D目标检测任务，旨在解决LiDAR数据中分布外（OOD）物体检测问题。通过引入语言表示，将OOD检测转化为零样本分类任务。**

- **链接: [https://arxiv.org/pdf/2603.08180](https://arxiv.org/pdf/2603.08180)**

> **作者:** Michael Kösel; Marcel Schreiber; Michael Ulrich; Claudius Gläser; Klaus Dietmayer
>
> **备注:** Accepted for publication at the 2025 IEEE Intelligent Transportation Systems Conference (ITSC)
>
> **摘要:** LiDAR-based 3D object detection plays a critical role for reliable and safe autonomous driving systems. However, existing detectors often produce overly confident predictions for objects not belonging to known categories, posing significant safety risks. This is caused by so-called out-of-distribution (OOD) objects, which were not part of the training data, resulting in incorrect predictions. To address this challenge, we propose ALOOD (Aligned LiDAR representations for Out-Of-Distribution Detection), a novel approach that incorporates language representations from a vision-language model (VLM). By aligning the object features from the object detector to the feature space of the VLM, we can treat the detection of OOD objects as a zero-shot classification task. We demonstrate competitive performance on the nuScenes OOD benchmark, establishing a novel approach to OOD object detection in LiDAR using language representations. The source code is available at this https URL.
>
---
#### [new 209] Accelerating Video Generation Inference with Sequential-Parallel 3D Positional Encoding Using a Global Time Index
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决长视频合成和实时推理中的效率问题。通过优化注意力机制和位置编码，提升推理速度并降低延迟。**

- **链接: [https://arxiv.org/pdf/2603.06664](https://arxiv.org/pdf/2603.06664)**

> **作者:** Chao Yuan; Pan Li
>
> **摘要:** Diffusion Transformer (DiT)-based video generation models inherently suffer from bottlenecks in long video synthesis and real-time inference, which can be attributed to the use of full spatiotemporal attention. Specifically, this mechanism leads to explosive O(N^2) memory consumption and high first-frame latency. To address these issues, we implement system-level inference optimizations for a causal autoregressive video generation pipeline. We adapt the Self-Forcing causal autoregressive framework to sequence parallel inference and implement a sequence-parallel variant of the causal rotary position embedding which we refer to as Causal-RoPE SP. This adaptation enables localized computation and reduces cross-rank communication in sequence parallel execution. In addition, computation and communication pipelines are optimized through operator fusion and RoPE precomputation. Experiments conducted on an eight GPU A800 cluster show that the optimized system achieves comparable generation quality, sub-second first-frame latency, and near real-time inference speed. For generating five second 480P videos, a 1.58x speedup is achieved, thereby providing effective support for real-time interactive applications.
>
---
#### [new 210] A Lightweight Digital-Twin-Based Framework for Edge-Assisted Vehicle Tracking and Collision Prediction
- **分类: cs.CV; cs.NI; cs.RO; eess.SP**

- **简介: 该论文属于智能交通系统中的车辆跟踪与碰撞预测任务，旨在解决边缘设备计算资源有限的问题。通过轻量级数字孪生框架，利用目标检测实现高效跟踪与碰撞预测。**

- **链接: [https://arxiv.org/pdf/2603.07338](https://arxiv.org/pdf/2603.07338)**

> **作者:** Murat Arda Onsu; Poonam Lohan; Burak Kantarci; Aisha Syed; Matthew Andrews; Sean Kennedy
>
> **备注:** 6 pages, 2 figures, IEEE ICC 2026 Workshops (under submission)
>
> **摘要:** Vehicle tracking, motion estimation, and collision prediction are fundamental components of traffic safety and management in Intelligent Transportation Systems (ITS). Many recent approaches rely on computationally intensive prediction models, which limits their practical deployment on resource-constrained edge devices. This paper presents a lightweight digital-twin-based framework for vehicle tracking and spatiotemporal collision prediction that relies solely on object detection, without requiring complex trajectory prediction networks. The framework is implemented and evaluated in Quanser Interactive Labs (QLabs), a high-fidelity digital twin of an urban traffic environment that enables controlled and repeatable scenario generation. A YOLO-based detector is deployed on simulated edge cameras to localize vehicles and extract frame-level centroid trajectories. Offline path maps are constructed from multiple traversals and indexed using K-D trees to support efficient online association between detected vehicles and road segments. During runtime, consistent vehicle identifiers are maintained, vehicle speed and direction are estimated from the temporal evolution of path indices, and future positions are predicted accordingly. Potential collisions are identified by analyzing both spatial proximity and temporal overlap of predicted future trajectories. Our experimental results across diverse simulated urban scenarios show that the proposed framework predicts approximately 88% of collision events prior to occurrence while maintaining low computational overhead suitable for edge deployment. Rather than introducing a computationally intensive prediction model, this work introduces a lightweight digital-twin-based solution for vehicle tracking and collision prediction, tailored for real-time edge deployment in ITS.
>
---
#### [new 211] Step-Level Visual Grounding Faithfulness Predicts Out-of-Distribution Generalization in Long-Horizon Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究长时视觉语言模型的泛化能力，解决模型推理与视觉输入一致性问题。通过定义步骤接地率（SGR）衡量模型中间推理的可信度，发现其能有效预测模型在分布外数据上的表现。**

- **链接: [https://arxiv.org/pdf/2603.06828](https://arxiv.org/pdf/2603.06828)**

> **作者:** Md Ashikur Rahman; Md Arifur Rahman; Niamul Hassan Samin; Abdullah Ibne Hanif Arean; Juena Ahmed Noshin
>
> **摘要:** We uncover a behavioral law of long-horizon vision-language models: models that maintain temporally grounded beliefs generalize better. Standard benchmarks measure only final-answer accuracy, which obscures how models use visual information; a model can guess correctly while its step-by-step reasoning is entirely unanchored to the visual input. We formalize this as behavioral faithfulness over long horizons, an empirically measurable property that quantifies whether a model's intermediate reasoning remains consistent with the evolving visual state. Across eight models on three long-horizon benchmarks, we demonstrate that temporal grounding quality is a leading indicator of robustness: the Step Grounding Rate (SGR) predicts out-of-distribution retention with $r = 0.83$ (permutation test $p = 0.003$), a relationship that holds within capacity-matched models and cannot be explained by scale or in-distribution accuracy. Critically, grounding quality varies by up to 10.8 percentage points within parameter-matched 7B models despite similar accuracy, revealing it as an independent axis of model capability. Multiple robustness checks confirm the signal reflects genuine visual reliance: counterfactual traces drop SGR by 26--41 percentage points, cross-architecture verifiers agree at $\rho = 0.96$, random reasoning scores near chance ($\sim 18\%$), and the predictor remains strong even without explicit reasoning disclosure ($r = 0.78$).
>
---
#### [new 212] UnSCAR: Universal, Scalable, Controllable, and Adaptable Image Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像修复任务，解决多退化场景下模型扩展性差的问题。提出UnSCAR框架，通过多分支专家系统实现可扩展、可控的通用图像恢复。**

- **链接: [https://arxiv.org/pdf/2603.07406](https://arxiv.org/pdf/2603.07406)**

> **作者:** Debabrata Mandal; Soumitri Chattopadhyay; Yujie Wang; Marc Niethammer; Praneeth Chakravarthula
>
> **摘要:** Universal image restoration aims to recover clean images from arbitrary real-world degradations using a single inference model. Despite significant progress, existing all-in-one restoration networks do not scale to multiple degradations. As the number of degradations increases, training becomes unstable, models grow excessively large, and performance drops across both seen and unseen domains. In this work, we show that scaling universal restoration is fundamentally limited by interference across degradations during joint learning, leading to catastrophic task forgetting. To address this challenge, we introduce a unified inference pipeline with a multi-branch mixture-of-experts architecture that decomposes restoration knowledge across specialized task-adaptable experts. Our approach enables scalable learning (over sixteen degradations), adapts and generalizes robustly to unseen domains, and supports user-controllable restoration across degradations. Beyond achieving superior performance across benchmarks, this work establishes a new design paradigm for scalable and controllable universal image restoration.
>
---
#### [new 213] SGG-R$^{\rm 3}$: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation
- **分类: cs.CV**

- **简介: 该论文属于场景图生成任务，解决现有方法召回率低和预测偏差的问题。通过引入结构化推理框架SGG-R$^{\rm 3}$，结合监督微调与强化学习，提升场景图生成的准确性和覆盖率。**

- **链接: [https://arxiv.org/pdf/2603.07961](https://arxiv.org/pdf/2603.07961)**

> **作者:** Jiaye Feng; Qixiang Yin; Yuankun Liu; Tong Mo; Weiping Li
>
> **摘要:** Scene Graph Generation (SGG) structures visual scenes as graphs of objects and their relations. While Multimodal Large Language Models (MLLMs) have advanced end-to-end SGG, current methods are hindered by both a lack of task-specific structured reasoning and the challenges of sparse, long-tailed relation distributions, resulting in incomplete scene graphs characterized by low recall and biased predictions. To address these issues, we introduce SGG-R$^{\rm 3}$, a structured reasoning framework that integrates task-specific chain-of-thought (CoT)-guided supervised fine-tuning (SFT) and reinforcement learning (RL) with group sequence policy optimization (GSPO), designed to engage in three sequential stages to achieve end-to-end unbiased scene graph generation. During the SFT phase, we propose a relation augmentation strategy by leveraging an MLLM and refined via embedding similarity filtering to alleviate relation sparsity. Subsequently, a stage-aligned reward scheme optimizes the procedural reasoning during RL. Specifically, we propose a novel dual-granularity reward which integrates fine-grained and coarse-grained relation rewards, simultaneously mitigating the long-tail issue via frequency-based adaptive weighting of predicates and improving relation coverage through semantic clustering. Experiments on two benchmarks show that SGG-R$^{\rm 3}$ achieves superior performance compared to existing methods, demonstrating the effectiveness and generalization of the framework.
>
---
#### [new 214] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于工业视觉检测任务，旨在解决高精度、低延迟的异常检测问题。通过集成生成式异常检测算法，实现高效在线检测。**

- **链接: [https://arxiv.org/pdf/2603.07577](https://arxiv.org/pdf/2603.07577)**

> **作者:** Niccolò Ferrari; Nicola Zanarini; Michele Fraccaroli; Alice Bizzarri; Evelina Lamma
>
> **备注:** Preprint under review at a Springer Nature journal. 36 pages, 3 tables, 29 figures. Updated and expanded version of the SSRN preprint (abstract_id=4858664), with substantial revisions and Springer Nature formatting
>
> **摘要:** Industrial visual inspection in pharmaceutical production requires high accuracy under strict constraints on cycle time, hardware footprint, and operational cost. Manual inline inspection is still common, but it is affected by operator variability and limited throughput. Classical rule-based computer vision pipelines are often rigid and difficult to scale to highly variable production scenarios. To address these limitations, we present a semi-supervised anomaly detection framework based on a generative adversarial architecture with a residual autoencoder and a dense bottleneck, specifically designed for online deployment on a high-speed Blow-Fill-Seal (BFS) line. The model is trained only on nominal samples and detects anomalies through reconstruction residuals, providing both classification and spatial localization via heatmaps. The training set contains 2,815,200 grayscale patches. Experiments on a real industrial test kit show high detection performance while satisfying timing constraints compatible with a 500 ms acquisition slot.
>
---
#### [new 215] MV-Fashion: Towards Enabling Virtual Try-On and Size Estimation with Multi-View Paired Data
- **分类: cs.CV**

- **简介: 该论文提出MV-Fashion数据集，解决时尚领域虚拟试穿和尺寸估计问题，包含多视角真实服装动态数据及详细标注。**

- **链接: [https://arxiv.org/pdf/2603.08147](https://arxiv.org/pdf/2603.08147)**

> **作者:** Hunor Laczkó; Libang Jia; Loc-Phat Truong; Diego Hernández; Sergio Escalera; Jordi Gonzalez; Meysam Madadi
>
> **摘要:** Existing 4D human datasets fall short for fashion-specific research, lacking either realistic garment dynamics or task-specific annotations. Synthetic datasets suffer from a realism gap, whereas real-world captures lack the detailed annotations and paired data required for virtual try-on (VTON) and size estimation tasks. To bridge this gap, we introduce MV-Fashion, a large-scale, multi-view video dataset engineered for domain-specific fashion analysis. MV-Fashion features 3,273 sequences (72.5 million frames) from 80 diverse subjects wearing 3-10 outfits each. It is designed to capture complex, real-world garment dynamics, including multiple layers and varied styling (e.g. rolled sleeves, tucked shirt). A core contribution is a rich data representation that includes pixel-level semantic annotations, ground-truth material properties like elasticity, and 3D point clouds. Crucially for VTON applications, MV-Fashion provides paired data: multi-view synchronized captures of worn garments alongside their corresponding flat, catalogue images. We leverage this dataset to establish baselines for fashion-centric tasks, including virtual try-on, clothing size estimation, and novel view synthesis. The dataset is available at this https URL .
>
---
#### [new 216] TIQA: Human-Aligned Text Quality Assessment in Generated Images
- **分类: cs.CV**

- **简介: 该论文提出TIQA任务，解决生成图像中文本质量评估问题。通过构建数据集和提出ANTIQA方法，提升文本质量评估与人类判断的一致性。**

- **链接: [https://arxiv.org/pdf/2603.07119](https://arxiv.org/pdf/2603.07119)**

> **作者:** Kirill Koltsov; Aleksandr Gushchin; Dmitriy Vatolin; Anastasia Antsiferova
>
> **摘要:** Text rendering remains a persistent failure mode of modern text-to-image models (T2I), yet existing evaluations rely on OCR correctness or VLM-based judging procedures that are poorly aligned with perceptual text artifacts. We introduce Text-in-Image Quality Assessment (TIQA), a task that predicts a scalar quality score that matches human judgments of rendered-text fidelity within cropped text regions. We release two MOS-labeled datasets: TIQA-Crops (10k text crops) and TIQA-Images (1,500 images), spanning 20+ T2I models, including proprietary ones. We also propose ANTIQA, a lightweight method with text-specific biases, and show that it improves correlation with human scores over OCR confidence, VLM judges, and generic NR-IQA metrics by at least $\sim0.05$ on TIQA-Crops and $\sim0.08$ on TIQA-Images, as measured by PLCC. Finally, we show that TIQA models are valuable in downstream tasks: for example, selecting the best-of-5 generations with ANTIQA improves human-rated text quality by $+14\%$ on average, demonstrating practical value for filtering and reranking in generation pipelines.
>
---
#### [new 217] FusionRegister: Every Infrared and Visible Image Fusion Deserves Registration
- **分类: cs.CV**

- **简介: 该论文属于红外与可见光图像融合任务，解决跨模态配准难题。提出FusionRegister方法，通过学习错位表示实现高效、鲁棒的融合，提升细节对齐与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.07667](https://arxiv.org/pdf/2603.07667)**

> **作者:** Congcong Bian; Haolong Ma; Hui Li; Zhongwei Shen; Xiaoqing Luo; Xiaoning Song; Xiao-Jun Wu
>
> **摘要:** Spatial registration across different visual modalities is a critical but formidable step in multi-modality image fusion for real-world perception. Although several methods are proposed to address this issue, the existing registration-based fusion methods typically require extensive pre-registration operations, limiting their efficiency. To overcome these limitations, a general cross-modality registration method guided by visual priors is proposed for infrared and visible image fusion task, termed FusionRegister. Firstly, FusionRegister achieves robustness by learning cross-modality misregistration representations rather than forcing alignment of all differences, ensuring stable outputs even under challenging input conditions. Moreover, FusionRegister demonstrates strong generality by operating directly on fused results, where misregistration is explicitly represented and effectively handled, enabling seamless integration with diverse fusion methods while preserving their intrinsic properties. In addition, its efficiency is further enhanced by serving the backbone fusion method as a natural visual prior provider, which guides the registration process to focus only on mismatch regions, thereby avoiding redundant operations. Extensive experiments on three datasets demonstrate that FusionRegister not only inherits the fusion quality of state-of-the-art methods, but also delivers superior detail alignment and robustness, making it highly suitable for infrared and visible image fusion method. The code will be available at this https URL.
>
---
#### [new 218] UWPD: A General Paradigm for Invisible Watermark Detection Agnostic to Embedding Algorithms
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UWPD任务，解决未知水印检测问题。构建UniFreq-100K数据集，设计FSNet模型，通过频域分析实现高效水印检测。**

- **链接: [https://arxiv.org/pdf/2603.06723](https://arxiv.org/pdf/2603.06723)**

> **作者:** Xiang Ao; Yiling Du; Zidan Wang; Mengru Chen
>
> **备注:** 26 pages, 7 figures
>
> **摘要:** Invisible watermarks, as an essential technology for image copyright protection, have been widely deployed with the rapid development of social media and AIGC. However, existing invisible watermark detection heavily relies on prior knowledge of specific algorithms, leading to limited detection capabilities for "unknown watermarks" in open environments. To this end, we propose a novel task named Universal Watermark Presence Detection (UWPD), which aims to identify whether an image carries a copyright mark without requiring decoding information. We construct the UniFreq-100K dataset, comprising large-scale samples across various invisible watermark embedding algorithms. Furthermore, we propose the Frequency Shield Network (FSNet). This model deploys an Adaptive Spectral Perception Module (ASPM) in the shallow layers, utilizing learnable frequency gating to dynamically amplify high-frequency watermark signals while suppressing low-frequency semantics. In the deep layers, the network introduces Dynamic Multi-Spectral Attention (DMSA) combined with tri-stream extremum pooling to deeply mine watermark energy anomalies, forcing the model to precisely focus on sensitive frequency bands. Extensive experiments demonstrate that FSNet exhibits superior zero-shot detection capabilities on the UWPD task, outperforming existing baseline models. Code and datasets will be released upon acceptance.
>
---
#### [new 219] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation
- **分类: cs.CV**

- **简介: 该论文属于图像表示任务，旨在解决高分辨率图像高效存储与渲染问题。提出SGI框架，通过结构化2D高斯分解和多尺度优化，实现更高效的压缩与更快的优化。**

- **链接: [https://arxiv.org/pdf/2603.07789](https://arxiv.org/pdf/2603.07789)**

> **作者:** Zixuan Pan; Kaiyuan Tang; Jun Xia; Yifan Qin; Lin Gu; Chaoli Wang; Jianxu Chen; Yiyu Shi
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** 2D Gaussian Splatting has emerged as a novel image representation technique that can support efficient rendering on low-end devices. However, scaling to high-resolution images requires optimizing and storing millions of unstructured Gaussian primitives independently, leading to slow convergence and redundant parameters. To address this, we propose Structured Gaussian Image (SGI), a compact and efficient framework for representing high-resolution images. SGI decomposes a complex image into multi-scale local spaces defined by a set of seeds. Each seed corresponds to a spatially coherent region and, together with lightweight multi-layer perceptrons (MLPs), generates structured implicit 2D neural Gaussians. This seed-based formulation imposes structural regularity on otherwise unstructured Gaussian primitives, which facilitates entropy-based compression at the seed level to reduce the total storage. However, optimizing seed parameters directly on high-resolution images is a challenging and non-trivial task. Therefore, we designed a multi-scale fitting strategy that refines the seed representation in a coarse-to-fine manner, substantially accelerating convergence. Quantitative and qualitative evaluations demonstrate that SGI achieves up to 7.5x compression over prior non-quantized 2D Gaussian methods and 1.6x over quantized ones, while also delivering 1.6x and 6.5x faster optimization, respectively, without degrading, and often improving, image fidelity. Code is available at this https URL.
>
---
#### [new 220] An Extended Topological Model For High-Contrast Optical Flow
- **分类: cs.CV; math.AT**

- **简介: 该论文属于视觉数据推理任务，研究高对比度光流的拓扑结构。通过构建3-流形模型，解释了原有环面模型无法验证的原因，并揭示了高对比度光流集中于二值阶跃边缘附近。**

- **链接: [https://arxiv.org/pdf/2603.06853](https://arxiv.org/pdf/2603.06853)**

> **作者:** Brad Turow; Jose A. Perea
>
> **备注:** 28 pages, 31 figures
>
> **摘要:** In this paper, we identify low-dimensional models for dense core subsets in the space of $3\times 3$ high-contrast optical flow patches sampled from the Sintel dataset. In particular, we leverage the theory of approximate and discrete circle bundles to identify a 3-manifold whose boundary is a previously proposed optical flow torus, together with disjoint circles corresponding to pairs of binary step-edge range image patches. The 3-manifold model we introduce provides an explanation for why the previously-proposed torus model could not be verified with direct methods (e.g., a straightforward persistent homology computation). We also demonstrate that nearly all optical flow patches in the top 1 percent by contrast norm are found near the family of binary step-edge circles described above, rather than the optical flow torus, and that these frequently occurring patches are concentrated near motion boundaries (which are of particular importance for computer vision tasks such as object segmentation and tracking). Our findings offer insights on the subtle interplay between topology and geometry in inference for visual data.
>
---
#### [new 221] AutoTraces: Autoregressive Trajectory Forecasting via Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出AutoTraces，用于机器人轨迹预测任务，解决复杂人类行为建模问题，通过多模态LLM和轨迹分词方法实现精准长时预测。**

- **链接: [https://arxiv.org/pdf/2603.07989](https://arxiv.org/pdf/2603.07989)**

> **作者:** Teng Wang; Yanting Lu; Ruize Wang
>
> **摘要:** We present AutoTraces, an autoregressive vision-language-trajectory model for robot trajectory forecasting in humam-populated environments, which harnesses the inherent reasoning capabilities of large language models (LLMs) to model complex human behaviors. In contrast to prior works that rely solely on textual representations, our key innovation lies in a novel trajectory tokenization scheme, which represents waypoints with point tokens as categorical and positional markers while encoding waypoint numerical values as corresponding point embeddings, seamlessly integrated into the LLM's space through a lightweight encoder-decoder architecture. This design preserves the LLM's native autoregressive generation mechanism while extending it to physical coordinate spaces, facilitates modeling of long-term interactions in trajectory data. We further introduce an automated chain-of-thought (CoT) generation mechanism that leverages a multimodal LLM to infer spatio-temporal relationships from visual observations and trajectory data, eliminating reliance on manual annotation. Through a two-stage training strategy, our AutoTraces achieves SOTA forecasting accuracy, particularly in long-horizon prediction, while exhibiting strong cross-scene generalization and supporting flexible-length forecasting.
>
---
#### [new 222] UNBOX: Unveiling Black-box visual models with Natural-language
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UNBOX框架，解决黑盒视觉模型的可解释性问题，通过自然语言技术揭示模型内部概念，提升系统可信度。**

- **链接: [https://arxiv.org/pdf/2603.08639](https://arxiv.org/pdf/2603.08639)**

> **作者:** Simone Carnemolla; Chiara Russo; Simone Palazzo; Quentin Bouniot; Daniela Giordano; Zeynep Akata; Matteo Pennisi; Concetto Spampinato
>
> **备注:** Under review at IJCV
>
> **摘要:** Ensuring trustworthiness in open-world visual recognition requires models that are interpretable, fair, and robust to distribution shifts. Yet modern vision systems are increasingly deployed as proprietary black-box APIs, exposing only output probabilities and hiding architecture, parameters, gradients, and training data. This opacity prevents meaningful auditing, bias detection, and failure analysis. Existing explanation methods assume white- or gray-box access or knowledge of the training distribution, making them unusable in these real-world settings. We introduce UNBOX, a framework for class-wise model dissection under fully data-free, gradient-free, and backpropagation-free constraints. UNBOX leverages Large Language Models and text-to-image diffusion models to recast activation maximization as a purely semantic search driven by output probabilities. The method produces human-interpretable text descriptors that maximally activate each class, revealing the concepts a model has implicitly learned, the training distribution it reflects, and potential sources of bias. We evaluate UNBOX on ImageNet-1K, Waterbirds, and CelebA through semantic fidelity tests, visual-feature correlation analyses and slice-discovery auditing. Despite operating under the strictest black-box constraints, UNBOX performs competitively with state-of-the-art white-box interpretability methods. This demonstrates that meaningful insight into a model's internal reasoning can be recovered without any internal access, enabling more trustworthy and accountable visual recognition systems.
>
---
#### [new 223] Soft Equivariance Regularization for Invariant Self-Supervised Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自监督学习任务，解决强不变性抑制有用结构的问题。通过引入软等变正则化（SER），在不改变最终表示的情况下，对中间特征图施加等变约束，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.06693](https://arxiv.org/pdf/2603.06693)**

> **作者:** Joohyung Lee; Changhun Kim; Hyunsu Kim; Kwanhyung Lee; Juho Lee
>
> **备注:** 14th International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Self-supervised learning (SSL) typically learns representations invariant to semantic-preserving augmentations. While effective for recognition, enforcing strong invariance can suppress transformation-dependent structure that is useful for robustness to geometric perturbations and spatially sensitive transfer. A growing body of work, therefore, augments invariance-based SSL with equivariance objectives, but these objectives are often imposed on the same final representation. We empirically observe a trade-off in this coupled setting: pushing equivariance regularization toward deeper layers improves equivariance scores but degrades ImageNet-1k linear evaluation, motivating a layer-decoupled design. Motivated by this trade-off, we propose Soft Equivariance Regularization (SER), a plug-in regularizer that decouples where invariance and equivariance are enforced: we keep the base SSL objective unchanged on the final embedding, while softly encouraging equivariance on an intermediate spatial token map via analytically specified group actions $\rho_g$ applied directly in feature space. SER learns/predicts no per-sample transformation codes/labels, requires no auxiliary transformation-prediction head, and adds only 1.008x training FLOPs. On ImageNet-1k ViT-S/16 pretraining, SER improves MoCo-v3 by +0.84 Top-1 in linear evaluation under a strictly matched 2-view setting and consistently improves DINO and Barlow Twins; under matched view counts, SER achieves the best ImageNet-1k linear-eval Top-1 among the compared invariance+equivariance add-ons. SER further improves ImageNet-C/P by +1.11/+1.22 Top-1 and frozen-backbone COCO detection by +1.7 mAP. Finally, applying the same layer-decoupling recipe to existing invariance+equivariance baselinesimproves their accuracy, suggesting layer decoupling as a general design principle for combining invariance and equivariance.
>
---
#### [new 224] Online Sparse Synthetic Aperture Radar Imaging
- **分类: cs.CV**

- **简介: 该论文属于SAR图像重建任务，旨在解决无人机载SAR数据处理中的计算与存储效率问题。提出Online FISTA算法，实现在线稀疏重构，降低内存需求并支持实时目标识别。**

- **链接: [https://arxiv.org/pdf/2603.08582](https://arxiv.org/pdf/2603.08582)**

> **作者:** Conor Flynn; Radoslav Ivanov; Birsen Yazici
>
> **备注:** IEEE Radar Conference 2026
>
> **摘要:** With modern defense applications increasingly relying on inexpensive, autonomous drones, lies the major challenge of designing computationally and memory-efficient onboard algorithms to fulfill mission objectives. This challenge is particularly significant in Synthetic Aperture Radar (SAR), where large volumes of data must be collected and processed for downstream tasks. We propose an online reconstruction method, the Online Fast Iterative Shrinkage-Thresholding Algorithm (Online FISTA), which incrementally reconstructs a scene with limited data through sparse coding. Rather than requiring storage of all received signal data, the algorithm recursively updates storage matrices for each iteration, greatly reducing memory demands. Online SAR image reconstruction facilitates more complex downstream tasks, such as Automatic Target Recognition (ATR), in an online manner, resulting in a more versatile and integrated framework compared to existing post-collection reconstruction and ATR approaches.
>
---
#### [new 225] Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出E$^2$OAL，解决开放集主动学习中未知类样本标注问题，通过统一框架提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.07898](https://arxiv.org/pdf/2603.07898)**

> **作者:** Chen-Chen Zong; Yu-Qi Chi; Xie-Yang Wang; Yan Cui; Sheng-Jun Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Open-set active learning (OSAL) aims to identify informative samples for annotation when unlabeled data may contain previously unseen classes-a common challenge in safety-critical and open-world scenarios. Existing approaches typically rely on separately trained open-set detectors, introducing substantial training overhead and overlooking the supervisory value of labeled unknowns for improving known-class learning. In this paper, we propose E$^2$OAL (Effective and Efficient Open-set Active Learning), a unified and detector-free framework that fully exploits labeled unknowns for both stronger supervision and more reliable querying. E$^2$OAL first uncovers the latent class structure of unknowns through label-guided clustering in a frozen contrastively pre-trained feature space, optimized by a structure-aware F1-product objective. To leverage labeled unknowns, it employs a Dirichlet-calibrated auxiliary head that jointly models known and unknown categories, improving both confidence calibration and known-class discrimination. Building on this, a logit-margin purity score estimates the likelihood of known classes to construct a high-purity candidate pool, while an OSAL-specific informativeness metric prioritizes partially ambiguous yet reliable samples. These components together form a flexible two-stage query strategy with adaptive precision control and minimal hyperparameter sensitivity. Extensive experiments across multiple OSAL benchmarks demonstrate that E$^2$OAL consistently surpasses state-of-the-art methods in accuracy, efficiency, and query precision, highlighting its effectiveness and practicality for real-world applications. The code is available at this http URL.
>
---
#### [new 226] Efficient Chest X-ray Representation Learning via Semantic-Partitioned Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决CXR预训练中的效率与效果问题。提出S-PCL方法，通过语义分块对比学习，提升模型性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.07113](https://arxiv.org/pdf/2603.07113)**

> **作者:** Wangyu Feng; Shawn Young; Lijian Xu
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful paradigm for Chest X-ray (CXR) analysis under limited annotations. Yet, existing SSL strategies remain suboptimal for medical imaging. Masked image modeling allocates substantial computation to reconstructing high-frequency background details with limited diagnostic value. Contrastive learning, on the other hand, often depends on aggressive augmentations that risk altering clinically meaningful structures. We introduce Semantic-Partitioned Contrastive Learning (S-PCL), an efficient pre-training framework tailored for CXR representation learning. Instead of reconstructing pixels or relying on heavy augmentations, S-PCL randomly partitions patch tokens from a single CXR into two non-overlapping semantic subsets. Each subset provides a complementary but incomplete view. The encoder must maximize agreement between these partitions, implicitly inferring global anatomical layout and local pathological cues from partial evidence. This semantic partitioning forms an internal bottleneck that enforces long-range dependency modeling and structural coherence. S-PCL eliminates the need for hand-crafted augmentations, auxiliary decoders, and momentum encoders. The resulting architecture is streamlined, computationally efficient, and easy to scale. Extensive experiments on large-scale CXR benchmarks, including ChestX-ray14, CheXpert, RSNA Pneumonia and SIIM-ACR Pneumothorax, show that S-PCL achieves competitive performance while attaining the lowest GFLOPs and superior accuracy among existing SSL approaches.
>
---
#### [new 227] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution
- **分类: cs.CV**

- **简介: 该论文提出SWIFT任务，解决生成视频的溯源问题。通过滑动窗口重构，无需训练即可实现高精度视频来源识别。**

- **链接: [https://arxiv.org/pdf/2603.08536](https://arxiv.org/pdf/2603.08536)**

> **作者:** Chao Wang; Zijin Yang; Yaofei Wang; Yuang Qi; Weiming Zhang; Nenghai Yu; Kejiang Chen
>
> **摘要:** Recent advancements in video generation technologies have been significant, resulting in their widespread application across multiple domains. However, concerns have been mounting over the potential misuse of generated content. Tracing the origin of generated videos has become crucial to mitigate potential misuse and identify responsible parties. Existing video attribution methods require additional operations or the training of source attribution models, which may degrade video quality or necessitate large amounts of training samples. To address these challenges, we define for the first time the "few-shot training-free generated video attribution" task and propose SWIFT, which is tightly integrated with the temporal characteristics of the video. By leveraging the "Pixel Frames(many) to Latent Frame(one)" temporal mapping within each video chunk, SWIFT applies a fixed-length sliding window to perform two distinct reconstructions: normal and corrupted. The variation in the losses between two reconstructions is then used as an attribution signal. We conducted an extensive evaluation of five state-of-the-art (SOTA) video generation models. Experimental results show that SWIFT achieves over 90% average attribution accuracy with merely 20 video samples across all models and even enables zero-shot attribution for HunyuanVideo, EasyAnimate, and Wan2.2. Our source code is available at this https URL.
>
---
#### [new 228] Video2LoRA: Unified Semantic-Controlled Video Generation via Per-Reference-Video LoRA
- **分类: cs.CV**

- **简介: 该论文提出Video2LoRA，解决视频生成中语义对齐难题。通过参考视频生成统一的语义控制视频，无需额外训练，实现高效、灵活的视频生成。**

- **链接: [https://arxiv.org/pdf/2603.08210](https://arxiv.org/pdf/2603.08210)**

> **作者:** Zexi Wu; Qinghe Wang; Jing Dai; Baolu Li; Yiming Zhang; Yue Ma; Xu Jia; Hongming Xu
>
> **备注:** 10 pages
>
> **摘要:** Achieving semantic alignment across diverse video generation conditions remains a significant challenge. Methods that rely on explicit structural guidance often enforce rigid spatial constraints that limit semantic flexibility, whereas models tailored for individual control types lack interoperability and adaptability. These design bottlenecks hinder progress toward flexible and efficient semantic video generation. To address this, we propose Video2LoRA, a scalable and generalizable framework for semantic-controlled video generation that conditions on a reference video. Video2LoRA employs a lightweight hypernetwork to predict personalized LoRA weights for each semantic input, which are combined with auxiliary matrices to form adaptive LoRA modules integrated into a frozen diffusion backbone. This design enables the model to generate videos consistent with the reference semantics while preserving key style and content variations, eliminating the need for any per-condition training. Notably, the final model weights less than 150MB, making it highly efficient for storage and deployment. Video2LoRA achieves coherent, semantically aligned generation across diverse conditions and exhibits strong zero-shot generalization to unseen semantics.
>
---
#### [new 229] FastSTAR: Spatiotemporal Token Pruning for Efficient Autoregressive Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决STAR模型中因分辨率和帧数增加导致的“token爆炸”问题。通过时空token剪枝和部分更新机制，提升生成效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2603.07192](https://arxiv.org/pdf/2603.07192)**

> **作者:** Sungwoong Yune; Suheon Jeong; Joo-Young Kim
>
> **摘要:** Visual Autoregressive modeling (VAR) has emerged as a highly efficient alternative to diffusion-based frameworks, achieving comparable synthesis quality. However, as this paradigm extends to Spacetime Autoregressive modeling (STAR) for video generation, scaling resolution and frame counts leads to a "token explosion" that creates a massive computational bottleneck in the final refinement stages. To address this, we propose FastSTAR, a training-free acceleration framework designed for high-quality video generation. Our core method, Spatiotemporal Token Pruning, identifies essential tokens by integrating two specialized terms: (1) Spatial similarity, which evaluates structural convergence across hierarchical scales to skip computations in regions where further refinement becomes redundant, and (2) Temporal similarity, which identifies active motion trajectories by assessing feature-level variations relative to the preceding clip. Combined with a Partial Update mechanism, FastSTAR ensures that only non-converged regions are refined, maintaining fluid motion while bypassing redundant computations. Experimental results on InfinityStar demonstrate that FastSTAR achieves up to a 2.01x speedup with a PSNR of 28.29 and less than 1% performance degradation, proving a superior efficiency-quality trade-off for STAR-based video synthesis.
>
---
#### [new 230] Synthetic Defect Image Generation for Power Line Insulator Inspection Using Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于电力线路绝缘子缺陷分类任务，解决数据稀缺问题。通过多模态大语言模型生成缺陷图像，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2603.08069](https://arxiv.org/pdf/2603.08069)**

> **作者:** Xuesong Wang; Caisheng Wang
>
> **备注:** Submitted to Engineering Applications of Artificial Intelligence, Feb. 16, 2026
>
> **摘要:** Utility companies increasingly rely on drone imagery for post-event and routine inspection, but training accurate defect-type classifiers remains difficult because defect examples are rare and inspection datasets are often limited or proprietary. We address this data-scarcity setting by using an off-the-shelf multimodal large language model (MLLM) as a training-free image generator to synthesize defect images from visual references and text prompts. Our pipeline increases diversity via dual-reference conditioning, improves label fidelity with lightweight human verification and prompt refinement, and filters the resulting synthetic pool using an embedding-based selection rule based on distances to class centroids computed from the real training split. We evaluate on ceramic insulator defect-type classification (shell vs. glaze) using a public dataset with a realistic low training-data regime (104 real training images; 152 validation; 308 test). Augmenting the 10% real training set with embedding-selected synthetic images improves test F1 score (harmonic mean of precision and recall) from 0.615 to 0.739 (20% relative), corresponding to an estimated 4--5x data-efficiency gain, and the gains persist with stronger backbone models and frozen-feature linear-probe baselines. These results suggest a practical, low-barrier path for improving defect recognition when collecting additional real defects is slow or infeasible.
>
---
#### [new 231] Training for Trustworthy Saliency Maps: Adversarial Training Meets Feature-Map Smoothing
- **分类: cs.CV**

- **简介: 该论文属于可解释性AI任务，旨在提升梯度类显著图的稳定性与可信度。通过对抗训练和特征图平滑，解决显著图噪声大、不稳定的问题。**

- **链接: [https://arxiv.org/pdf/2603.07302](https://arxiv.org/pdf/2603.07302)**

> **作者:** Dipkamal Bhusal; Md Tanvirul Alam; Nidhi Rastogi
>
> **摘要:** Gradient-based saliency methods such as Vanilla Gradient (VG) and Integrated Gradients (IG) are widely used to explain image classifiers, yet the resulting maps are often noisy and unstable, limiting their usefulness in high-stakes settings. Most prior work improves explanations by modifying the attribution algorithm, leaving open how the training procedure shapes explanation quality. We take a training-centered view and first provide a curvature-based analysis linking attribution stability to how smoothly the input-gradient field varies locally. Guided by this connection, we study adversarial training and identify a consistent trade-off: it yields sparser and more input-stable saliency maps, but can degrade output-side stability, causing explanations to change even when predictions remain unchanged and logits vary only slightly. To mitigate this, we propose augmenting adversarial training with a lightweight feature-map smoothing block that applies a differentiable Gaussian filter in an intermediate layer. Across FMNIST, CIFAR-10, and ImageNette, our method preserves the sparsity benefits of adversarial training while improving both input-side stability and output-side stability. A human study with 65 participants further shows that smoothed adversarial saliency maps are perceived as more sufficient and trustworthy. Overall, our results demonstrate that explanation quality is critically shaped by training, and that simple smoothing with robust training provides a practical path toward saliency maps that are both sparse and stable.
>
---
#### [new 232] Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究深度学习与超广角成像在糖尿病视网膜病变和黄斑水肿检测中的应用，解决传统方法的局限性，通过模型对比与特征融合提升诊断性能。**

- **链接: [https://arxiv.org/pdf/2603.08235](https://arxiv.org/pdf/2603.08235)**

> **作者:** Pablo Jimenez-Lizcano; Sergio Romero-Tapiador; Ruben Tolosana; Aythami Morales; Guillermo González de Rivera; Ruben Vera-Rodriguez; Julian Fierrez
>
> **备注:** 6 pages, 4 figures, 2 tables
>
> **摘要:** Diabetic retinopathy (DR) and diabetic macular edema (DME) are leading causes of preventable blindness among working-age adults. Traditional approaches in the literature focus on standard color fundus photography (CFP) for the detection of these conditions. Nevertheless, recent ultra-widefield imaging (UWF) offers a significantly wider field of view in comparison to CFP. Motivated by this, the present study explores state-of-the-art deep learning (DL) methods and UWF imaging on three clinically relevant tasks: i) image quality assessment for UWF, ii) identification of referable diabetic retinopathy (RDR), and iii) identification of DME. Using the publicly available UWF4DR Challenge dataset, released as part of the MICCAI 2024 conference, we benchmark DL models in the spatial (RGB) and frequency domains, including popular convolutional neural networks (CNNs) as well as recent vision transformers (ViTs) and foundation models. In addition, we explore a final feature-level fusion to increase robustness. Finally, we also analyze the decisions of the DL models using Grad-CAM, increasing the explainability. Our proposal achieves consistently strong performance across all architectures, underscoring the competitiveness of emerging ViTs and foundation models and the promise of feature-level fusion and frequency-domain representations for UWF analysis.
>
---
#### [new 233] VIVECaption: A Split Approach to Caption Quality Improvement
- **分类: cs.CV**

- **简介: 该论文属于图像/视频生成任务，旨在解决caption质量低导致的图像-文本对齐问题。通过构建高质量数据集和模型微调提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.07401](https://arxiv.org/pdf/2603.07401)**

> **作者:** Varun Ananth; Baqiao Liu; Haoran Cai
>
> **摘要:** Caption quality has emerged as a critical bottleneck in training high-quality text-to-image (T2I) and text-to-video (T2V) generative models. While visual language models (VLMs) are commonly deployed to generate captions from visual data, they suffer from hallucinations, poor compositional reasoning, and limited fine-grained understanding, resulting in misaligned image-caption pairs that degrade downstream model performance. This technical report introduces VIVECaption, a systematic two-sided approach to caption quality improvement. We first establish a comprehensive taxonomy of caption evaluation metrics, distinguishing between "universal" and "instance-grounded" metrics, with the ultimate goal of showcasing the use-cases and tradeoffs between different caption quality metrics. We then use this language to describe our two-sided approach to caption quality improvement: (1) a gold-standard dataset creation methodology using stratified sampling and (2) a model alignment strategy encompassing context alignment and parameter-level finetuning using SFT. We demonstrate our methodology on open-source models, focusing on structured caption formats that enable better parsing and downstream utilization. We ultimately show that using a finetuned character detection model in an image captioning pipeline significantly improves holistic image-caption alignment quality. Our work addresses the growing need for high-quality "vegan" training data in enterprise AI development, providing practical solutions for teams seeking to improve caption-image alignment without relying on potentially copyright-protected web-scraped content.
>
---
#### [new 234] Chart Deep Research in LVLMs via Parallel Relative Policy Optimization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图表数据分析任务，旨在解决现有方法在深度研究能力上的不足。提出PRPO优化训练和MCDR-Bench评估体系，提升模型的复杂推理与分析能力。**

- **链接: [https://arxiv.org/pdf/2603.06677](https://arxiv.org/pdf/2603.06677)**

> **作者:** Jiajin Tang; Gaoyang; Wenjie Wang; Sibei Yang; Xing Chen
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** With the rapid advancement of data science, charts have evolved from simple numerical presentation tools to essential instruments for insight discovery and decision-making support. However, current chart data intelligence exhibits significant limitations in deep research capabilities, with existing methods predominantly addressing shallow tasks such as visual recognition or factual question-answering, rather than the complex reasoning and high-level data analysis that deep research requires. This limitation stems from two primary technical bottlenecks: at the training level, existing post-training techniques exhibit deficiencies in handling multi-dimensional reward signal interference and heterogeneous data gradient conflicts, preventing models from achieving balanced development across multiple capability dimensions; at the evaluation level, current methods remain limited to factual retrieval and basic computation, failing to assess end-to-end analytic reasoning and other deep research capabilities. To address the training challenge, we propose PRPO, which performs parallel optimization across reward dimensions and capability partitioning across data types, effectively disentangling conflicts between heterogeneous data and multi-dimensional reward signals while ensuring optimization stability. For the evaluation challenge, we construct MCDR-Bench based on the ``error uniqueness principle," transforming subjective generation assessment into objective error identification through controllable error injection, enabling quantifiable evaluation of deep research capabilities. Experimental validation confirms that the proposed PRPO and MCDR-Bench jointly establish a unified framework that systematically advances chart deep research through enhanced collaborative training and objective evaluation.
>
---
#### [new 235] AR2-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos
- **分类: cs.CV**

- **简介: 该论文属于视频目标定位任务，解决固定视角视频中长期语言引导定位问题。通过引入锚点机制和重识别策略，提升目标丢失后的重新定位效果。**

- **链接: [https://arxiv.org/pdf/2603.07758](https://arxiv.org/pdf/2603.07758)**

> **作者:** Teng Yan; Yihan Liu; Jiongxu Chen; Teng Wang; Jiaqi Li; Bingzhuo Zhong
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Long-term language-guided referring in fixed-view videos is challenging: the referent may be occluded or leave the scene for long intervals and later re-enter, while framewise referring pipelines drift as re-identification (ReID) becomes unreliable. AR2-4FV leverages background stability for long-term referring. An offline Anchor Bank is distilled from static background structures; at inference, the text query is aligned with this bank to produce an Anchor Map that serves as persistent semantic memory when the referent is absent. An anchor-based re-entry prior accelerates re-capture upon return, and a lightweight ReID-Gating mechanism maintains identity continuity using displacement cues in the anchor frame. The system predicts per-frame bounding boxes without assuming the target is visible in the first frame or explicitly modeling appearance variations. AR2-4FV achieves +10.3% Re-Capture Rate (RCR) improvement and -24.2% Re-Capture Latency (RCL) reduction over the best baseline, and ablation studies further confirm the benefits of the Anchor Map, re-entry prior, and ReID-Gating.
>
---
#### [new 236] ReconDrive: Fast Feed-Forward 4D Gaussian Splatting for Autonomous Driving Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶场景重建任务，旨在解决4D高斯溅射的效率与质量问题。提出ReconDrive框架，通过改进模型结构实现快速高质量重建。**

- **链接: [https://arxiv.org/pdf/2603.07552](https://arxiv.org/pdf/2603.07552)**

> **作者:** Haibao Yu; Kuntao Xiao; Jiahang Wang; Ruiyang Hao; Yuxin Huang; Guoran Hu; Haifang Qin; Bowen Jing; Yuntian Bo; Ping Luo
>
> **摘要:** High-fidelity visual reconstruction and novel-view synthesis are essential for realistic closed-loop evaluation in autonomous driving. While 4D Gaussian Splatting (4DGS) offers a promising balance of accuracy and efficiency, existing per-scene optimization methods require costly iterative refinement, rendering them unscalable for extensive urban environments. Conversely, current feed-forward approaches often suffer from degraded photometric quality. To address these limitations, we propose ReconDrive, a feed-forward framework that leverages and extends the 3D foundation model VGGT for rapid, high-fidelity 4DGS generation. Our architecture introduces two core adaptations to tailor the foundation model to dynamic driving scenes: (1) Hybrid Gaussian Prediction Heads, which decouple the regression of spatial coordinates and appearance attributes to overcome the photometric deficiencies inherent in generalized foundation features; and (2) a Static-Dynamic 4D Composition strategy that explicitly captures temporal motion via velocity modeling to represent complex dynamic environments. Benchmarked on nuScenes, ReconDrive significantly outperforms existing feed-forward baselines in reconstruction, novel-view synthesis, and 3D perception. It achieves performance competitive with per-scene optimization while being orders of magnitude faster, providing a scalable and practical solution for realistic driving simulation.
>
---
#### [new 237] GazeShift: Unsupervised Gaze Estimation and Dataset for VR
- **分类: cs.CV**

- **简介: 该论文属于VR眼动追踪任务，解决数据稀缺与标注困难问题。提出VRGaze数据集和GazeShift框架，实现无监督眼动估计与实时推理。**

- **链接: [https://arxiv.org/pdf/2603.07832](https://arxiv.org/pdf/2603.07832)**

> **作者:** Gil Shapira; Ishay Goldin; Evgeny Artyomov; Donghoon Kim; Yosi Keller; Niv Zehngut
>
> **备注:** Accepted to CVPR26
>
> **摘要:** Gaze estimation is instrumental in modern virtual reality (VR) systems. Despite significant progress in remote-camera gaze estimation, VR gaze research remains constrained by data scarcity - particularly the lack of large-scale, accurately labeled datasets captured with the off-axis camera configurations typical of modern headsets. Gaze annotation is difficult since fixation on intended targets cannot be guaranteed. To address these challenges, we introduce VRGaze - the first large-scale off-axis gaze estimation dataset for VR - comprising 2.1 million near-eye infrared images collected from 68 participants. We further propose GazeShift, an attention-guided unsupervised framework for learning gaze representations without labeled data. Unlike prior redirection-based methods that rely on multi-view or 3D geometry, GazeShift is tailored to near-eye infrared imagery, achieving effective gaze-appearance disentanglement in a compact, real-time model. GazeShift embeddings can be optionally adapted to individual users via lightweight few-shot calibration, achieving a 1.84-degree mean error on VRGaze. On the remote-camera MPIIGaze dataset, the model achieves a 7.15-degree person-agnostic error, doing so with 10x fewer parameters and 35x fewer FLOPs than baseline methods. Deployed natively on a VR headset GPU, inference takes only 5 ms. Combined with demonstrated robustness to illumination changes, these results highlight GazeShift as a label-efficient, real-time solution for VR gaze tracking. Project code and the VRGaze dataset are released at this https URL.
>
---
#### [new 238] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind
- **分类: cs.CV**

- **简介: 该论文研究多模态基础模型TerraMind在高光谱图像下游任务中的适应性，解决其缺乏高光谱数据原生支持的问题。通过对比两种通道适配策略，验证了模型的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.06690](https://arxiv.org/pdf/2603.06690)**

> **作者:** Julia Anna Leonardi; Johannes Jakubik; Paolo Fraccaro; Maria Antonia Brovelli
>
> **备注:** Accepted to ICLR 2026 Machine Learning for Remote Sensing (ML4RS) Workshop
>
> **摘要:** Geospatial Foundation Models (GFMs) typically lack native support for Hyperspectral Imaging (HSI) due to the complexity and sheer size of high-dimensional spectral data. This study investigates the adaptability of TerraMind, a multimodal GFM, to address HSI downstream tasks \emph{without} HSI-specific pretraining. Therefore, we implement and compare two channel adaptation strategies: Naive Band Selection and physics-aware Spectral Response Function (SRF) grouping. Overall, our results indicate a general superiority of deep learning models with native support of HSI data. Our experiments also demonstrate the ability of TerraMind to adapt to HSI downstream tasks through band selection with moderate performance decline. Therefore, the findings of this research establish a critical baseline for HSI integration, motivating the need for native spectral tokenization in future multimodal model architectures.
>
---
#### [new 239] Real-Time Glottis Detection Framework via Spatial-decoupled Feature Learning for Nasal Transnasal Intubation
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决鼻插管中声门快速准确检测的问题。提出轻量级模型Mobile GlottisNet，提升实时性与适用性。**

- **链接: [https://arxiv.org/pdf/2603.07630](https://arxiv.org/pdf/2603.07630)**

> **作者:** Jinyu Liu; Gaoyang Zhang; Yang Zhou; Ruoyi Hao; Yang Zhang; Hongliang Ren
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Nasotracheal intubation (NTI) is a vital procedure in emergency airway management, where rapid and accurate glottis detection is essential to ensure patient safety. However, existing machine assisted visual detection systems often rely on high performance computational resources and suffer from significant inference delays, which limits their applicability in time critical and resource constrained scenarios. To overcome these limitations, we propose Mobile GlottisNet, a lightweight and efficient glottis detection framework designed for real time inference on embedded and edge devices. The model incorporates structural awareness and spatial alignment mechanisms, enabling robust glottis localization under complex anatomical and visual conditions. We implement a hierarchical dynamic thresholding strategy to enhance sample assignment, and introduce an adaptive feature decoupling module based on deformable convolution to support dynamic spatial reconstruction. A cross layer dynamic weighting scheme further facilitates the fusion of semantic and detail features across multiple scales. Experimental results demonstrate that the model, with a size of only 5MB on both our PID dataset and Clinical datasets, achieves inference speeds of over 62 FPS on devices and 33 FPS on edge platforms, showing great potential in the application of emergency NTI.
>
---
#### [new 240] MM-TS: Multi-Modal Temperature and Margin Schedules for Contrastive Learning with Long-Tail Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态对比学习任务，针对长尾数据分布问题，提出MM-TS方法动态调整温度与边界，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.08202](https://arxiv.org/pdf/2603.08202)**

> **作者:** Siarhei Sheludzko; Dhimitrios Duka; Bernt Schiele; Hilde Kuehne; Anna Kukleva
>
> **备注:** 18 pages, 11 figures. Accepted at WACV 2026
>
> **摘要:** Contrastive learning has become a fundamental approach in both uni-modal and multi-modal frameworks. This learning paradigm pulls positive pairs of samples closer while pushing negatives apart. In the uni-modal setting (e.g., image-based learning), previous research has shown that the strength of these forces can be controlled through the temperature parameter. In this work, we propose Multi-Modal Temperature and Margin Schedules (MM-TS), extending the concept of uni-modal temperature scheduling to multi-modal contrastive learning. Our method dynamically adjusts the temperature in the contrastive loss during training, modulating the attraction and repulsion forces in the multi-modal setting. Additionally, recognizing that standard multi-modal datasets often follow imbalanced, long-tail distributions, we adapt the temperature based on the local distribution of each training sample. Specifically, samples from dense clusters are assigned a higher temperature to better preserve their semantic structure. Furthermore, we demonstrate that temperature scheduling can be effectively integrated within a max-margin framework, thereby unifying the two predominant approaches in multi-modal contrastive learning: InfoNCE loss and max-margin objective. We evaluate our approach on four widely used image- and video-language datasets, Flickr30K, MSCOCO, EPIC-KITCHENS-100, and YouCook2, and show that our dynamic temperature and margin schedules improve performance and lead to new state-of-the-art results in the field.
>
---
#### [new 241] Ref-DGS: Reflective Dual Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于表面重建与新视角合成任务，解决强近场镜面反射的建模问题。提出Ref-DGS框架，通过双高斯表示和混合着色器高效处理反射，提升重建精度与速度。**

- **链接: [https://arxiv.org/pdf/2603.07664](https://arxiv.org/pdf/2603.07664)**

> **作者:** Ningjing Fan; Yiqun Wang; Dongming Yan; Peter Wonka
>
> **备注:** Project page: this https URL
>
> **摘要:** Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods.
>
---
#### [new 242] Visual Self-Fulfilling Alignment: Shaping Safety-Oriented Personas via Threat-Related Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型安全对齐任务，旨在解决模型因视觉输入产生有害输出的问题。通过在威胁图像上微调模型，使其内化警惕性，提升安全性。**

- **链接: [https://arxiv.org/pdf/2603.08486](https://arxiv.org/pdf/2603.08486)**

> **作者:** Qishun Yang; Shu Yang; Lijie Hu; Di Wang
>
> **摘要:** Multimodal large language models (MLLMs) face safety misalignment, where visual inputs enable harmful outputs. To address this, existing methods require explicit safety labels or contrastive data; yet, threat-related concepts are concrete and visually depictable, while safety concepts, like helpfulness, are abstract and lack visual referents. Inspired by the Self-Fulfilling mechanism underlying emergent misalignment, we propose Visual Self-Fulfilling Alignment (VSFA). VSFA fine-tunes vision-language models (VLMs) on neutral VQA tasks constructed around threat-related images, without any safety labels. Through repeated exposure to threat-related visual content, models internalize the implicit semantics of vigilance and caution, shaping safety-oriented personas. Experiments across multiple VLMs and safety benchmarks demonstrate that VSFA reduces the attack success rate, improves response quality, and mitigates over-refusal while preserving general capabilities. Our work extends the self-fulfilling mechanism from text to visual modalities, offering a label-free approach to VLMs alignment.
>
---
#### [new 243] SiamGM: Siamese Geometry-Aware and Motion-Guided Network for Real-Time Satellite Video Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于卫星视频目标跟踪任务，解决小目标、背景模糊等问题。提出SiamGM网络，结合几何与运动信息，提升跟踪精度与实时性。**

- **链接: [https://arxiv.org/pdf/2603.07564](https://arxiv.org/pdf/2603.07564)**

> **作者:** Zixiao Wen; Zhen Yang; Jiawei Li; Xiantai Xiang; Guangyao Zhou; Yuxin Hu; Yuhan Liu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Single object tracking in satellite videos is inherently challenged by small target, blurred background, large aspect ratio changes, and frequent visual occlusions. These constraints often cause appearance-based trackers to accumulate errors and lose targets irreversibly. To systematically mitigate both spatial ambiguities and temporal information loss, we propose SiamGM, a novel geometry-aware and motion-guided Siamese network. From a spatial perspective, we introduce an Inter-Frame Graph Attention (IFGA) module, closely integrated with an Aspect Ratio-Constrained Label Assignment (LA) method, establishing fine-grained topological correspondences and explicitly preventing surrounding background noise. From a temporal perspective, we introduce the Motion Vector-Guided Online Tracking Optimization method. By adopting the Normalized Peak-to-Sidelobe Ratio (nPSR) as a dynamic confidence indicator, we propose an Online Motion Model Refinement (OMMR) strategy to utilize historical trajectory information. Evaluations on two challenging SatSOT and SV248S benchmarks confirm that SiamGM outperforms most state-of-the-art trackers in both precision and success metrics. Notably, the proposed components of SiamGM introduce virtually no computational overhead, enabling real-time tracking at 130 frames per second (FPS). Codes and tracking results are available at this https URL.
>
---
#### [new 244] Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉模型鲁棒性提升任务，旨在解决ViTs依赖虚假相关性的问题。通过概念引导的微调框架，优化模型关注语义概念而非背景，提升模型在分布偏移下的性能。**

- **链接: [https://arxiv.org/pdf/2603.08309](https://arxiv.org/pdf/2603.08309)**

> **作者:** Yehonatan Elisha; Oren Barkan; Noam Koenigstein
>
> **备注:** CVPR 2026 ; Project page: this https URL
>
> **摘要:** Vision Transformers (ViTs) often degrade under distribution shifts because they rely on spurious correlations, such as background cues, rather than semantically meaningful features. Existing regularization methods, typically relying on simple foreground-background masks, which fail to capture the fine-grained semantic concepts that define an object (e.g., ``long beak'' and ``wings'' for a ``bird''). As a result, these methods provide limited robustness to distribution shifts. To address this limitation, we introduce a novel finetuning framework that steers model reasoning toward concept-level semantics. Our approach optimizes the model's internal relevance maps to align with spatially grounded concept masks. These masks are generated automatically, without manual annotation: class-relevant concepts are first proposed using an LLM-based, label-free method, and then segmented using a VLM. The finetuning objective aligns relevance with these concept regions while simultaneously suppressing focus on spurious background areas. Notably, this process requires only a minimal set of images and uses half of the dataset classes. Extensive experiments on five out-of-distribution benchmarks demonstrate that our method improves robustness across multiple ViT-based models. Furthermore, we show that the resulting relevance maps exhibit stronger alignment with semantic object parts, offering a scalable path toward more robust and interpretable vision models. Finally, we confirm that concept-guided masks provide more effective supervision for model robustness than conventional segmentation maps, supporting our central hypothesis.
>
---
#### [new 245] Prototype-Guided Concept Erasure in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中难以消除宽泛概念（如“性”或“暴力”）的问题。通过分析嵌入空间，提取概念原型作为负条件信号，实现更可靠的去除。**

- **链接: [https://arxiv.org/pdf/2603.08271](https://arxiv.org/pdf/2603.08271)**

> **作者:** Yuze Cai; Jiahao Lu; Hongxiang Shi; Yichao Zhou; Hong Lu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Concept erasure is extensively utilized in image generation to prevent text-to-image models from generating undesired content. Existing methods can effectively erase narrow concepts that are specific and concrete, such as distinct intellectual properties (e.g. Pikachu) or recognizable characters (e.g. Elon Musk). However, their performance degrades on broad concepts such as ``sexual'' or ``violent'', whose wide scope and multi-faceted nature make them difficult to erase reliably. To overcome this limitation, we exploit the model's intrinsic embedding geometry to identify latent embeddings that encode a given concept. By clustering these embeddings, we derive a set of concept prototypes that summarize the model's internal representations of the concept, and employ them as negative conditioning signals during inference to achieve precise and reliable erasure. Extensive experiments across multiple benchmarks show that our approach achieves substantially more reliable removal of broad concepts while preserving overall image quality, marking a step towards safer and more controllable image generation.
>
---
#### [new 246] DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于4D场景重建任务，旨在解决自动驾驶中动态场景的准确重建问题。提出DynamicVGGT框架，通过动态点图和运动感知机制实现高效4D重建。**

- **链接: [https://arxiv.org/pdf/2603.08254](https://arxiv.org/pdf/2603.08254)**

> **作者:** Zhuolin He; Jing Li; Guanghao Li; Xiaolei Chen; Jiacheng Tang; Siyang Zhang; Zhounan Jin; Feipeng Cai; Bin Li; Jian Pu; Jia Cai; Xiangyang Xue
>
> **摘要:** Dynamic scene reconstruction in autonomous driving remains a fundamental challenge due to significant temporal variations, moving objects, and complex scene dynamics. Existing feed-forward 3D models have demonstrated strong performance in static reconstruction but still struggle to capture dynamic motion. To address these limitations, we propose DynamicVGGT, a unified feed-forward framework that extends VGGT from static 3D perception to dynamic 4D reconstruction. Our goal is to model point motion within feed-forward 3D models in a dynamic and temporally coherent manner. To this end, we jointly predict the current and future point maps within a shared reference coordinate system, allowing the model to implicitly learn dynamic point representations through temporal correspondence. To efficiently capture temporal dependencies, we introduce a Motion-aware Temporal Attention (MTA) module that learns motion continuity. Furthermore, we design a Dynamic 3D Gaussian Splatting Head that explicitly models point motion by predicting Gaussian velocities using learnable motion tokens under scene flow supervision. It refines dynamic geometry through continuous 3D Gaussian optimization. Extensive experiments on autonomous driving datasets demonstrate that DynamicVGGT significantly outperforms existing methods in reconstruction accuracy, achieving robust feed-forward 4D dynamic scene reconstruction under complex driving scenarios.
>
---
#### [new 247] This Looks Distinctly Like That: Grounding Interpretable Recognition in Stiefel Geometry against Neural Collapse
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，解决原型网络中的原型坍缩问题，提出AMP框架通过流形优化提升可解释性与分类性能。**

- **链接: [https://arxiv.org/pdf/2603.08374](https://arxiv.org/pdf/2603.08374)**

> **作者:** Junhao Jia; Jiaqi Wang; Yunyou Liu; Haodong Jing; Yueyi Wu; Xian Wu; Yefeng Zheng
>
> **摘要:** Prototype networks provide an intrinsic case based explanation mechanism, but their interpretability is often undermined by prototype collapse, where multiple prototypes degenerate to highly redundant evidence. We attribute this failure mode to the terminal dynamics of Neural Collapse, where cross entropy optimization suppresses intra class variance and drives class conditional features toward a low dimensional limit. To mitigate this, we propose Adaptive Manifold Prototypes (AMP), a framework that leverages Riemannian optimization on the Stiefel manifold to represent class prototypes as orthonormal bases and make rank one prototype collapse infeasible by construction. AMP further learns class specific effective rank via a proximal gradient update on a nonnegative capacity vector, and introduces spatial regularizers that reduce rotational ambiguity and encourage localized, non overlapping part evidence. Extensive experiments on fine-grained benchmarks demonstrate that AMP achieves state-of-the-art classification accuracy while significantly improving causal faithfulness over prior interpretable models.
>
---
#### [new 248] BioGait-VLM: A Tri-Modal Vision-Language-Biomechanics Framework for Interpretable Clinical Gait Assessment
- **分类: cs.CV**

- **简介: 该论文属于临床步态分析任务，旨在解决视频分析模型过拟合环境偏差的问题。提出BioGait-VLM框架，融合视觉、语言和生物力学信息，提升分析的可解释性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.08564](https://arxiv.org/pdf/2603.08564)**

> **作者:** Erdong Chen; Yuyang Ji; Jacob K. Greenberg; Benjamin Steel; Faraz Arkam; Abigail Lewis; Pranay Singh; Feng Liu
>
> **摘要:** Video-based Clinical Gait Analysis often suffers from poor generalization as models overfit environmental biases instead of capturing pathological motion. To address this, we propose BioGait-VLM, a tri-modal Vision-Language-Biomechanics framework for interpretable clinical gait assessment. Unlike standard video encoders, our architecture incorporates a Temporal Evidence Distillation branch to capture rhythmic dynamics and a Biomechanical Tokenization branch that projects 3D skeleton sequences into language-aligned semantic tokens. This enables the model to explicitly reason about joint mechanics independent of visual shortcuts. To ensure rigorous benchmarking, we augment the public GAVD dataset with a high-fidelity Degenerative Cervical Myelopathy (DCM) cohort to form a unified 8-class taxonomy, establishing a strict subject-disjoint protocol to prevent data leakage. Under this setting, BioGait-VLM achieves state-of-the-art recognition accuracy. Furthermore, a blinded expert study confirms that biomechanical tokens significantly improve clinical plausibility and evidence grounding, offering a path toward transparent, privacy-enhanced gait assessment.
>
---
#### [new 249] Vessel-Aware Deep Learning for OCTA-Based Detection of AMD
- **分类: cs.CV**

- **简介: 该论文属于AMD检测任务，旨在解决传统模型忽略血管生物标志物的问题。通过引入血管特异性特征图，提升OCTA图像的分类性能。**

- **链接: [https://arxiv.org/pdf/2603.06735](https://arxiv.org/pdf/2603.06735)**

> **作者:** Margalit G. Mitzner; Moinak Bhattacharya; Zhilin Zou; Chao Chen; Prateek Prasanna
>
> **摘要:** Age-related macular degeneration (AMD) is characterized by early micro-vascular alterations that can be captured non-invasively using optical coherence tomography angiography (OCTA), yet most deep learning (DL) models rely on global features and fail to exploit clinically meaningful vascular biomarkers. We introduce an external multiplicative attention framework that incorporates vessel-specific tortuosity maps and vasculature dropout maps derived from arteries, veins, and capillaries. These biomarker maps are generated from vessel segmentations and smoothed across multiple spatial scales to highlight coherent patterns of vascular remodeling and capillary rarefaction. Tortuosity reflects abnormalities in vessel geometry linked to impaired auto-regulation, while dropout maps capture localized perfusion deficits that precede structural retinal damage. The maps are fused with the OCTA projection to guide a deep classifier toward physiologically relevant regions. Arterial tortuosity provided the most consistent discriminative value, while capillary dropout maps performed best among density-based variants, especially at larger smoothing scales. Our proposed method offers interpretable insights aligned with known AMD pathophysiology.
>
---
#### [new 250] Prompt-Based Caption Generation for Single-Tooth Dental Images Using Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型在牙科图像中的应用任务，旨在解决缺乏单颗牙图像带描述数据集的问题，通过生成 captions 来提升模型对牙科图像的理解能力。**

- **链接: [https://arxiv.org/pdf/2603.07403](https://arxiv.org/pdf/2603.07403)**

> **作者:** Anastasiia Sukhanova; Aiden Taylor; Julian Myers; Zichun Wang; Kartha Veerya Jammuladinne; Satya Sri Rajiteswari Nimmagadda; Aniruddha Maiti; Ananya Jana
>
> **备注:** Accepted to IEEE International Conference on Semantic Computing (IEEE ICSC 2026)
>
> **摘要:** Digital dentistry has made significant advances with the advent of deep learning. However, the majority of these deep learning-based dental image analysis models focus on very specific tasks such as tooth segmentation, tooth detection, cavity detection, and gingivitis classification. There is a lack of a specialized model that has holistic knowledge of teeth and can perform dental image analysis tasks based on that knowledge. Datasets of dental images with captions can help build such a model. To the best of our knowledge, existing dental image datasets with captions are few in number and limited in scope. In many of these datasets, the captions describe the entire mouth, while the images are limited to the anterior view. As a result, posterior teeth such as molars are not clearly visible, limiting the usefulness of the captions for training vision-language models. Additionally, the captions focus only on a specific disease (gingivitis) and do not provide a holistic assessment of each tooth. Moreover, tooth disease scores are typically assigned to individual teeth, and each tooth is treated as a separate entity in orthodontic procedures. Therefore, it is important to have captions for single-tooth images. As far as we know, no such dataset of single-tooth images with dental captions exists. In this work, we aim to bridge that gap by assessing the possibility of generating captions for dental images using Vision-Language Models (VLMs) and evaluating the extent and quality of those captions. Our findings suggest that guided prompts help VLMs generate meaningful captions. We show that the prompts generated by our framework are better anchored in describing the visual aspects of dental images. We selected RGB images as they have greater potential in consumer scenarios.
>
---
#### [new 251] Selective Transfer Learning of Cross-Modality Distillation for Monocular 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，旨在解决因模态差异导致的知识迁移效果差问题。提出MonoSTL方法，通过深度感知的特征与关系蒸馏，提升模型精度。**

- **链接: [https://arxiv.org/pdf/2603.07464](https://arxiv.org/pdf/2603.07464)**

> **作者:** Rui Ding; Meng Yang; Nanning Zheng
>
> **摘要:** Monocular 3D object detection is a promising yet ill-posed task for autonomous vehicles due to the lack of accurate depth information. Cross-modality knowledge distillation could effectively transfer depth information from LiDAR to image-based network. However, modality gap between image and LiDAR seriously limits its accuracy. In this paper, we systematically investigate the negative transfer problem induced by modality gap in cross-modality distillation for the first time, including not only the architecture inconsistency issue but more importantly the feature overfitting issue. We propose a selective learning approach named MonoSTL to overcome these issues, which encourages positive transfer of depth information from LiDAR while alleviates the negative transfer on image-based network. On the one hand, we utilize similar architectures to ensure spatial alignment of features between image-based and LiDAR-based networks. On the other hand, we develop two novel distillation modules, namely Depth-Aware Selective Feature Distillation (DASFD) and Depth-Aware Selective Relation Distillation (DASRD), which selectively learn positive features and relationships of objects by integrating depth uncertainty into feature and relation distillations, respectively. Our approach can be seamlessly integrated into various CNN-based and DETR-based models, where we take three recent models on KITTI and a recent model on NuScenes for validation. Extensive experiments show that our approach considerably improves the accuracy of the base models and thereby achieves the best accuracy compared with all recently released SOTA models.
>
---
#### [new 252] Thinking with Gaze: Sequential Eye-Tracking as Visual Reasoning Supervision for Medical VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉语言模型任务，旨在提升模型的视觉推理能力。通过引入眼动追踪作为监督信号，指导模型更贴近人类的视觉搜索过程，增强其在医学影像中的推理效果。**

- **链接: [https://arxiv.org/pdf/2603.06697](https://arxiv.org/pdf/2603.06697)**

> **作者:** Yiwei Li; Zihao Wu; Yanjun Lv; Hanqi Jiang; Weihang You; Zhengliang Liu; Dajiang Zhu; Xiang Li; Quanzheng Li; Tianming Liu; Lin Zhao
>
> **摘要:** Vision--language models (VLMs) process images as visual tokens, yet their intermediate reasoning is often carried out in text, which can be suboptimal for visually grounded radiology tasks. Radiologists instead diagnose via sequential visual search; eye-tracking captures this process as time-ordered gaze trajectories that reveal how evidence is acquired over time. We use eye-gaze as supervision to guide VLM reasoning by introducing a small set of dedicated gaze tokens. These tokens are trained to predict gaze-selected image patch indices in temporal order, encouraging the model to follow human-like evidence acquisition and integration. Experiments on MIMIC-EYE and multiple external zero-shot benchmarks show consistent gains over baselines, achieving state-of-the-art in-domain performance and improved out-of-domain robustness. These results highlight temporally ordered gaze as an effective supervision signal for learning visually grounded medical reasoning.
>
---
#### [new 253] ECHO: Event-Centric Hypergraph Operations via Multi-Agent Collaboration for Multimedia Event Extraction
- **分类: cs.CV**

- **简介: 该论文属于多媒体事件抽取任务，解决跨模态对齐错误导致的下游错误传播问题，提出ECHO框架通过多智能体协作迭代优化事件超图。**

- **链接: [https://arxiv.org/pdf/2603.06683](https://arxiv.org/pdf/2603.06683)**

> **作者:** Hailong Chu; Shuo Zhang; Yunlong Chu; Shutai Huang; Xingyue Zhang; Tinghe Yan; Jinsong Zhang; Lei Li
>
> **摘要:** Multimedia Event Extraction (M2E2) involves extracting structured event records from both textual and visual content. Existing approaches, ranging from specialized architectures to direct Large Language Model (LLM) prompting, typically rely on a linear, end-to-end generation and thus suffer from cascading errors: early cross-modal misalignments often corrupt downstream role assignment under strict grounding constraints. We propose ECHO (Event-Centric Hypergraph Operations), a multi-agent framework that iteratively refines a shared Multimedia Event Hypergraph (MEHG), which serves as an explicit intermediate structure for multimodal event hypotheses. Unlike dialogue-centric frameworks, ECHO coordinates specialized agents by applying atomic hypergraph operations to the MEHG. Furthermore, we introduce a Link-then-Bind strategy that enforces deferred commitment: agents first identify relevant arguments and only then determine their precise roles, mitigating incorrect grounding and limiting error propagation. Extensive experiments on the M2E2 benchmark show that ECHO significantly outperforms the state-of-the-art (SOTA) : with Qwen3-32B, it achieves a 7.3% and 15.5% improvement in average event mention and argument role F1, respectively.
>
---
#### [new 254] Human-AI Divergence in Ego-centric Action Recognition under Spatial and Spatiotemporal Manipulations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动作识别任务，旨在探究人类与AI在视角动作识别中的差异。通过对比分析人类与模型在不同空间和时空条件下的表现，揭示性能差距的原因。**

- **链接: [https://arxiv.org/pdf/2603.08317](https://arxiv.org/pdf/2603.08317)**

> **作者:** Sadegh Rahmaniboldaji; Filip Rybansky; Quoc C. Vuong; Anya C. Hurlbert; Frank Guerin; Andrew Gilbert
>
> **摘要:** Humans consistently outperform state-of-the-art AI models in action recognition, particularly in challenging real-world conditions involving low resolution, occlusion, and visual clutter. Understanding the sources of this performance gap is essential for developing more robust and human-aligned models. In this paper, we present a large-scale human-AI comparative study of egocentric action recognition using Minimal Identifiable Recognition Crops (MIRCs), defined as the smallest spatial or spatiotemporal regions sufficient for reliable human recognition. We used our previously introduced, Epic ReduAct, a systematically spatially reduced and temporally scrambled dataset derived from 36 EPIC KITCHENS videos, spanning multiple spatial reduction levels and temporal conditions. Recognition performance is evaluated using over 3,000 human participants and the Side4Video model. Our analysis combines quantitative metrics, Average Reduction Rate and Recognition Gap, with qualitative analyses of spatial (high-, mid-, and low-level visual features) and spatiotemporal factors, including a categorisation of actions into Low Temporal Actions (LTA) and High Temporal Actions (HTA). Results show that human performance exhibits sharp declines when transitioning from MIRCs to subMIRCs, reflecting a strong reliance on sparse, semantically critical cues such as hand-object interactions. In contrast, the model degrades more gradually and often relies on contextual and mid- to low-level features, sometimes even exhibiting increased confidence under spatial reduction. Temporally, humans remain robust to scrambling when key spatial cues are preserved, whereas the model often shows insensitivity to temporal disruption, revealing class-dependent temporal sensitivities.
>
---
#### [new 255] FabricGen: Microstructure-Aware Woven Fabric Generation
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于纺织材料生成任务，解决传统方法复杂且细节不足的问题。通过分解宏观纹理与微观编织结构，结合微调模型与语言模型生成高质量织物。**

- **链接: [https://arxiv.org/pdf/2603.07240](https://arxiv.org/pdf/2603.07240)**

> **作者:** Yingjie Tang; Di Luo; Zixiong Wang; Xiaoli Ling; jian Yang; Beibei Wang
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** Woven fabric materials are widely used in rendering applications, yet designing realistic examples typically involves multiple stages, requiring expertise in weaving principles and texture authoring. Recent advances have explored diffusion models to streamline this process; however, pre-trained diffusion models often struggle to generate intricate yarn-level details that conform to weaving rules. To address this, we present FabricGen, an end-to-end framework for generating high-quality woven fabric materials from textual descriptions. A key insight of our method is the decomposition of macro-scale textures and micro-scale weaving patterns. To generate macro-scale textures free from microstructures, we fine-tune pre-trained diffusion models on a collected dataset of microstructure-free fabrics. As for micro-scale weaving patterns, we develop an enhanced procedural geometric model capable of synthesizing natural yarn-level geometry with yarn sliding and flyaway fibers. The procedural model is driven by a specialized large language model, WeavingLLM, which is fine-tuned on an annotated dataset of formatted weaving drafts, and prompt-tuned with domain-specific fabric expertise. Through fine-tuning and prompt tuning, WeavingLLM learns to design weaving drafts and fabric parameters from textual prompts, enabling the procedural model to produce diverse weaving patterns that stick to weaving principles. The generated macro-scale texture, along with the micro-scale geometry, can be used for fabric rendering. Consequently, our framework produces materials with significantly richer detail and realism compared to prior generative models.
>
---
#### [new 256] Interpretable Aneurysm Classification via 3D Concept Bottleneck Models: Integrating Morphological and Hemodynamic Clinical Features
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于医学图像分类任务，旨在解决颅内动脉瘤的可靠分类问题，通过3D概念瓶颈模型融合形态和血流特征，提升模型的可解释性与诊断准确性。**

- **链接: [https://arxiv.org/pdf/2603.07399](https://arxiv.org/pdf/2603.07399)**

> **作者:** Toqa Khaled; Ahmad Al-Kabbany
>
> **摘要:** We are concerned with the challenge of reliably classifying and assessing intracranial aneurysms using deep learning without compromising clinical transparency. While traditional black-box models achieve high predictive accuracy, their lack of inherent interpretability remains a significant barrier to clinical adoption and regulatory approval. Explainability is paramount in medical modeling to ensure that AI-driven diagnoses align with established neurosurgical principles. Unlike traditional eXplainable AI (XAI) methods -- such as saliency maps, which often provide post-hoc, non-causal visual correlations -- Concept Bottleneck Models (CBMs) offer a robust alternative by constraining the model's internal logic to human-understandable clinical indices. In this article, we propose an end-to-end 3D Concept Bottleneck framework that maps high-dimensional neuroimaging features to a discrete set of morphological and hemodynamic concepts for aneurysm identification. We implemented this pipeline using a pre-trained 3D ResNet-34 backbone and a 3D DenseNet-121 to extract features from CTA volumes, which were subsequently processed through a soft bottleneck layer representing human-interpretable clinical concepts. The model was optimized using a joint-loss function to balance diagnostic focal loss and concept mean squared error (MSE), validated via stratified five-fold cross-validation. Our results demonstrate a peak task classification accuracy of 93.33% +/- 4.5% for the ResNet-34 architecture and 91.43% +/- 5.8% for the DenseNet-121 model. Furthermore, the implementation of 8-pass Test-Time Augmentation (TTA) yielded a robust mean accuracy of 88.31%, ensuring diagnostic stability during inference. By maintaining an accuracy-generalization gap of less than 0.04, this framework proves that high predictive performance can be achieved without sacrificing interpretability.
>
---
#### [new 257] Beyond Attention Heatmaps: How to Get Better Explanations for Multiple Instance Learning Models in Histopathology
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算病理学中的多实例学习任务，旨在解决MIL模型热图解释性不足的问题。通过实验评估不同解释方法，提升模型可解释性并提供生物学验证。**

- **链接: [https://arxiv.org/pdf/2603.08328](https://arxiv.org/pdf/2603.08328)**

> **作者:** Mina Jamshidi Idaji; Julius Hense; Tom Neuhäuser; Augustin Krause; Yanqing Luo; Oliver Eberle; Thomas Schnake; Laure Ciernik; Farnoush Rezaei Jafari; Reza Vahidimajd; Jonas Dippel; Christoph Walz; Frederick Klauschen; Andreas Mock; Klaus-Robert Müller
>
> **摘要:** Multiple instance learning (MIL) has enabled substantial progress in computational histopathology, where a large amount of patches from gigapixel whole slide images are aggregated into slide-level predictions. Heatmaps are widely used to validate MIL models and to discover tissue biomarkers. Yet, the validity of these heatmaps has barely been investigated. In this work, we introduce a general framework for evaluating the quality of MIL heatmaps without requiring additional labels. We conduct a large-scale benchmark experiment to assess six explanation methods across histopathology task types (classification, regression, survival), MIL model architectures (Attention-, Transformer-, Mamba-based), and patch encoder backbones (UNI2, Virchow2). Our results show that explanation quality mostly depends on MIL model architecture and task type, with perturbation ("Single"), layer-wise relevance propagation (LRP), and integrated gradients (IG) consistently outperforming attention-based and gradient-based saliency heatmaps, which often fail to reflect model decision mechanisms. We further demonstrate the advanced capabilities of the best-performing explanation methods: (i) We provide a proof-of-concept that MIL heatmaps of a bulk gene expression prediction model can be correlated with spatial transcriptomics for biological validation, and (ii) showcase the discovery of distinct model strategies for predicting human papillomavirus (HPV) infection from head and neck cancer slides. Our work highlights the importance of validating MIL heatmaps and establishes that improved explainability can enable more reliable model validation and yield biological insights, making a case for a broader adoption of explainable AI in digital pathology. Our code is provided in a public GitHub repository: this https URL
>
---
#### [new 258] Faster-HEAL: An Efficient and Privacy-Preserving Collaborative Perception Framework for Heterogeneous Autonomous Vehicles
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的协同感知任务，旨在解决异构车辆间特征域差异导致的检测性能下降问题。提出Faster-HEAL框架，通过低秩视觉提示和金字塔融合实现高效隐私保护的特征对齐。**

- **链接: [https://arxiv.org/pdf/2603.07314](https://arxiv.org/pdf/2603.07314)**

> **作者:** Armin Maleki; Hayder Radha
>
> **备注:** Accepted to appear in the 2026 IEEE Intelligent Vehicles Symposium (IV 2026), Detroit, MI, USA, June 22-25, 2026. 6 pages, 1 figure, 4 tables
>
> **摘要:** Collaborative perception (CP) is a promising paradigm for improving situational awareness in autonomous vehicles by overcoming the limitations of single-agent perception. However, most existing approaches assume homogeneous agents, which restricts their applicability in real-world scenarios where vehicles use diverse sensors and perception models. This heterogeneity introduces a feature domain gap that degrades detection performance. Prior works address this issue by retraining entire models/major components, or using feature interpreters for each new agent type, which is computationally expensive, compromises privacy, and may reduce single-agent accuracy. We propose Faster-HEAL, a lightweight and privacy-preserving CP framework that fine-tunes a low-rank visual prompt to align heterogeneous features with a unified feature space while leveraging pyramid fusion for robust feature aggregation. This approach reduces the trainable parameters by 94%, enabling efficient adaptation to new agents without retraining large models. Experiments on the OPV2V-H dataset show that Faster-HEAL improves detection performance by 2% over state-of-the-art methods with significantly lower computational overhead, offering a practical solution for scalable heterogeneous CP.
>
---
#### [new 259] Virtual Intraoperative CT (viCT): Sequential Anatomic Updates for Modeling Tissue Resection Throughout Endoscopic Sinus Surgery
- **分类: cs.CV**

- **简介: 该论文提出viCT方法，用于内镜鼻窦手术中实时更新解剖结构，解决传统CT无法反映术中变化的问题。通过3D重建与配准实现动态解剖建模。**

- **链接: [https://arxiv.org/pdf/2603.06956](https://arxiv.org/pdf/2603.06956)**

> **作者:** Nicole M. Gunderson; Graham J. Harris; Jeremy S. Ruthberg; Pengcheng Chen; Di Mao; Randall A. Bly; Waleed M. Abuzeid; Eric J. Seibel
>
> **摘要:** Purpose: Incomplete dissection is a common cause of persistent disease and revision endoscopic sinus surgery (ESS) in chronic rhinosinusitis. Current image-guided surgery systems typically reference static preoperative CT (pCT), and do not model evolving resection boundaries. We present Virtual Intraoperative CT (viCT), a method for sequentially updating pCT throughout ESS using intraoperative 3D reconstructions from monocular endoscopic video to enable visualization of evolving anatomy in CT format. Methods: Monocular endoscopic video is processed using a depth-supervised NeRF framework with virtual stereo synthesis to generate metrically scaled 3D reconstructions at multiple surgical intervals. Reconstructions undergo rigid, landmark-based registration in 3D Slicer guided by anatomical correspondences, and are then voxelized into the pCT grid. viCT volumes were generated using a ray-based occupancy comparison between pCT and reconstruction to delete outdated voxels and remap preserved anatomy and updated boundaries. Performance is evaluated in a cadaveric feasibility study of four specimens across four ESS stages using volumetric overlap (DSC, Jaccard) and surface metrics (HD95, Chamfer, MSD, RMSD), and qualitative comparisons to ground-truth CT. Results: viCT updates show agreement with ground-truth anatomy across surgical stages, with submillimeter mean surface errors. Dice Similarity Coefficient (DSC) = 0.88 +/- 0.05 and Jaccard Index = 0.79 +/- 0.07, and Hausdorff Distance 95% (HD95) = 0.69 +/- 0.28 mm, Chamfer Distance = 0.09 +/- 0.05 mm, Mean Surface Distance (MSD) = 0.11 +/- 0.05 mm, and Root Mean Square Distance (RMSD) = 0.32 +/- 0.10 mm. Conclusion: viCT enables CT-format anatomic updating in an ESS setting without ancillary hardware. Future work will focus on fully automating registration, validation in live cases, and optimizing runtime for real-time deployment.
>
---
#### [new 260] Enhancing Cross-View UAV Geolocalization via LVLM-Driven Relational Modeling
- **分类: cs.CV**

- **简介: 该论文属于跨视角无人机定位任务，旨在提升无人机图像与卫星图像的匹配精度。通过LVLM驱动的关系建模和新损失函数，增强视图间交互学习，提高定位准确性。**

- **链接: [https://arxiv.org/pdf/2603.08063](https://arxiv.org/pdf/2603.08063)**

> **作者:** Bowen Liu; Pengyue Jia; Wanyu Wang; Derong Xu; Jiawei Cheng; Jiancheng Dong; Xiao Han; Zimo Zhao; Chao Zhang; Bowen Yu; Fangyu Hong; Xiangyu Zhao
>
> **摘要:** The primary objective of cross-view UAV geolocalization is to identify the exact spatial coordinates of drone-captured imagery by aligning it with extensive, geo-referenced satellite databases. Current approaches typically extract features independently from each perspective and rely on basic heuristics to compute similarity, thereby failing to explicitly capture the essential interactions between different views. To address this limitation, we introduce a novel, plug-and-play ranking architecture designed to explicitly perform joint relational modeling for improved UAV-to-satellite image matching. By harnessing the capabilities of a Large Vision-Language Model (LVLM), our framework effectively learns the deep visual-semantic correlations linking UAV and satellite imagery. Furthermore, we present a novel relational-aware loss function to optimize the training phase. By employing soft labels, this loss provides fine-grained supervision that avoids overly penalizing near-positive matches, ultimately boosting both the model's discriminative power and training stability. Comprehensive evaluations across various baseline architectures and standard benchmarks reveal that the proposed method substantially boosts the retrieval accuracy of existing models, yielding superior performance even under highly demanding conditions.
>
---
#### [new 261] ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在提升3D高斯溅射的效率与质量。通过C++/CUDA优化实现ImprovedGS+，减少训练时间并提高重建效果。**

- **链接: [https://arxiv.org/pdf/2603.08661](https://arxiv.org/pdf/2603.08661)**

> **作者:** Jordi Muñoz Vicente
>
> **备注:** 6 pages, 1 figure. Technical Report. This work introduces ImprovedGS+, a library-free C++/CUDA implementation for 3D Gaussian Splatting within the LichtFeld-Studio framework. Source code available at this https URL
>
> **摘要:** Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.
>
---
#### [new 262] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance
- **分类: cs.CV**

- **简介: 该论文属于RGB-D场景理解任务，旨在解决传统方法在遮挡、边界模糊及适应性不足的问题。提出多任务自适应学习和跨维度特征引导模型，提升分割精度与速度。**

- **链接: [https://arxiv.org/pdf/2603.07570](https://arxiv.org/pdf/2603.07570)**

> **作者:** Guodong Sun; Junjie Liu; Gaoyang Zhang; Bo Wu; Yang Zhang
>
> **备注:** 23 pages, 13 figures
>
> **摘要:** Scene understanding plays a critical role in enabling intelligence and autonomy in robotic systems. Traditional approaches often face challenges, including occlusions, ambiguous boundaries, and the inability to adapt attention based on task-specific requirements and sample variations. To address these limitations, this paper presents an efficient RGB-D scene understanding model that performs a range of tasks, including semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification. The proposed model incorporates an enhanced fusion encoder, which effectively leverages redundant information from both RGB and depth inputs. For semantic segmentation, we introduce normalized focus channel layers and a context feature interaction layer, designed to mitigate issues such as shallow feature misguidance and insufficient local-global feature representation. The instance segmentation task benefits from a non-bottleneck 1D structure, which achieves superior contour representation with fewer parameters. Additionally, we propose a multi-task adaptive loss function that dynamically adjusts the learning strategy for different tasks based on scene variations. Extensive experiments on the NYUv2, SUN RGB-D, and Cityscapes datasets demonstrate that our approach outperforms existing methods in both segmentation accuracy and processing speed.
>
---
#### [new 263] Event-based Motion & Appearance Fusion for 6D Object Pose Tracking
- **分类: cs.CV**

- **简介: 该论文属于6D物体位姿跟踪任务，旨在解决高速动态环境下传统传感器性能受限的问题。通过结合事件相机的高时间分辨率，提出一种无需学习的方法，提升跟踪精度与速度。**

- **链接: [https://arxiv.org/pdf/2603.08264](https://arxiv.org/pdf/2603.08264)**

> **作者:** Zhichao Li; Chiara Bartolozzi; Lorenzo Natale; Arren Glover
>
> **摘要:** Object pose tracking is a fundamental and essential task for robotics to perform tasks in the home and industrial settings. The most commonly used sensors to do so are RGB-D cameras, which can hit limitations in highly dynamic environments due to motion blur and frame-rate constraints. Event cameras have remarkable features such as high temporal resolution and low latency, which make them a potentially ideal vision sensors for object pose tracking at high speed. Even so, there are still only few works on 6D pose tracking with event cameras. In this work, we take advantage of the high temporal resolution and propose a method that uses both a propagation step fused with a pose correction strategy. Specifically, we use 6D object velocity obtained from event-based optical flow for pose propagation, after which, a template-based local pose correction module is utilized for pose correction. Our learning-free method has comparable performance to the state-of-the-art algorithms, and in some cases out performs them for fast-moving objects. The results indicate the potential for using event cameras in highly-dynamic scenarios where the use of deep network approaches are limited by low update rates.
>
---
#### [new 264] FVG-PT: Adaptive Foreground View-Guided Prompt Tuning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的适应任务，解决提示调优中前景注意力偏移问题。提出FVG-PT模块，通过增强前景质量和补偿注意力来提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.08708](https://arxiv.org/pdf/2603.08708)**

> **作者:** Haoyang Li; Liang Wang; Siyu Zhou; Jiacheng Sun; Jing Jiang; Chao Wang; Guodong Long; Yan Peng
>
> **备注:** 27 Pages, 9 Figures, 15 Tables
>
> **摘要:** CLIP-based prompt tuning enables pretrained Vision-Language Models (VLMs) to efficiently adapt to downstream tasks. Although existing studies have made significant progress, they pay limited attention to changes in the internal attention representations of VLMs during the tuning process. In this paper, we attribute the failure modes of prompt tuning predictions to shifts in foreground attention of the visual encoder, and propose Foreground View-Guided Prompt Tuning (FVG-PT), an adaptive plug-and-play foreground attention guidance module, to alleviate the shifts. Concretely, FVG-PT introduces a learnable Foreground Reliability Gate to automatically enhance the foreground view quality, applies a Foreground Distillation Compensation module to guide visual attention toward the foreground, and further introduces a Prior Calibration module to mitigate generalization degradation caused by excessive focus on the foreground. Experiments on multiple backbone models and datasets show the effectiveness and compatibility of FVG-PT. Codes are available at: this https URL
>
---
#### [new 265] MAviS: A Multimodal Conversational Assistant For Avian Species
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态问答任务，旨在解决鸟类物种细粒度理解与跨模态问答难题。构建了MAviS-Dataset和MAviS-Chat模型，并提出MAviS-Bench进行评估。**

- **链接: [https://arxiv.org/pdf/2603.07294](https://arxiv.org/pdf/2603.07294)**

> **作者:** Yevheniia Kryklyvets; Mohammed Irfan Kurpath; Sahal Shaji Mullappilly; Jinxing Zhou; Fahad Shabzan Khan; Rao Anwer; Salman Khan; Hisham Cholakkal
>
> **备注:** EMNLP 2025
>
> **摘要:** Fine-grained understanding and species-specific multimodal question answering are vital for advancing biodiversity conservation and ecological monitoring. However, existing multimodal large language models face challenges when it comes to specialized topics like avian species, making it harder to provide accurate and contextually relevant information in these areas. To address this limitation, we introduce the MAviS-Dataset, a large-scale multimodal avian species dataset that integrates image, audio, and text modalities for over 1,000 bird species, comprising both pretraining and instruction-tuning subsets enriched with structured question-answer pairs. Building on the MAviS-Dataset, we introduce MAviS-Chat, a multimodal LLM that supports audio, vision, and text and is designed for fine-grained species understanding, multimodal question answering, and scene-specific description generation. Finally, for quantitative evaluation, we present MAviS-Bench, a benchmark of over 25,000 QA pairs designed to assess avian species-specific perceptual and reasoning abilities across modalities. Experimental results show that MAviS-Chat outperforms the baseline MiniCPM-o-2.6 by a large margin, achieving state-of-the-art open-source results and demonstrating the effectiveness of our instruction-tuned MAviS-Dataset. Our findings highlight the necessity of domain-adaptive multimodal LLMs for ecological applications.
>
---
#### [new 266] HIERAMP: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在提升小规模数据集的语义表达。针对传统方法仅关注全局语义的问题，提出HIERAMP方法，通过层次化语义放大增强不同层级的结构信息。**

- **链接: [https://arxiv.org/pdf/2603.06932](https://arxiv.org/pdf/2603.06932)**

> **作者:** Lin Zhao; Xinru Jiang; Xi Xiao; Qihui Fan; Lei Lu; Yanzhi Wang; Xue Lin; Octavia Camps; Pu Zhao; Jianyang Gu
>
> **备注:** The paper is accepted by CVPR 2026
>
> **摘要:** Dataset distillation often prioritizes global semantic proximity when creating small surrogate datasets for original large-scale ones. However, object semantics are inherently hierarchical. For example, the position and appearance of a bird's eyes are constrained by the outline of its head. Global proximity alone fails to capture how object-relevant structures at different levels support recognition. In this work, we investigate the contributions of hierarchical semantics to effective distilled data. We leverage the vision autoregressive (VAR) model whose coarse-to-fine generation mirrors this hierarchy and propose HIERAMP to amplify semantics at different levels. At each VAR scale, we inject class tokens that dynamically identify salient regions and use their induced maps to guide amplification at that scale. This adds only marginal inference cost while steering synthesis toward discriminative parts and structures. Empirically, we find that semantic amplification leads to more diverse token choices in constructing coarse-scale object layouts. Conversely, at fine scales, the amplification concentrates token usage, increasing focus on object-related details. Across popular dataset distillation benchmarks, HIERAMP consistently improves validation performance without explicitly optimizing global proximity, demonstrating the importance of semantic amplification for effective dataset distillation.
>
---
#### [new 267] Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决数据稀缺下的性能提升问题。通过构建统一框架UniDiffDA，分析并评估不同扩散数据增强方法的有效性。**

- **链接: [https://arxiv.org/pdf/2603.08364](https://arxiv.org/pdf/2603.08364)**

> **作者:** Zekun Li; Yinghuan Shi; Yang Gao; Dong Xu
>
> **摘要:** Diffusion-based data augmentation (DiffDA) has emerged as a promising approach to improving classification performance under data scarcity. However, existing works vary significantly in task configurations, model choices, and experimental pipelines, making it difficult to fairly compare methods or assess their effectiveness across different scenarios. Moreover, there remains a lack of systematic understanding of the full DiffDA workflow. In this work, we introduce UniDiffDA, a unified analytical framework that decomposes DiffDA methods into three core components: model fine-tuning, sample generation, and sample utilization. This perspective enables us to identify key differences among existing methods and clarify the overall design space. Building on this framework, we develop a comprehensive and fair evaluation protocol, benchmarking representative DiffDA methods across diverse low-data classification tasks. Extensive experiments reveal the relative strengths and limitations of different DiffDA strategies and offer practical insights into method design and deployment. All methods are re-implemented within a unified codebase, with full release of code and configurations to ensure reproducibility and to facilitate future research.
>
---
#### [new 268] Retrieval-Augmented Gaussian Avatars: Improving Expression Generalization
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于人脸动画生成任务，旨在解决模板无关头像表达泛化能力不足的问题。通过引入检索增强方法，提升表达多样性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08645](https://arxiv.org/pdf/2603.08645)**

> **作者:** Matan Levy; Gavriel Habib; Issar Tzachor; Dvir Samuel; Rami Ben-Ari; Nir Darshan; Or Litany; Dani Lischinski
>
> **摘要:** Template-free animatable head avatars can achieve high visual fidelity by learning expression-dependent facial deformation directly from a subject's capture, avoiding parametric face templates and hand-designed blendshape spaces. However, since learned deformation is supervised only by the expressions observed for a single identity, these models suffer from limited expression coverage and often struggle when driven by motions that deviate from the training distribution. We introduce RAF (Retrieval-Augmented Faces), a simple training-time augmentation designed for template-free head avatars that learn deformation from data. RAF constructs a large unlabeled expression bank and, during training, replaces a subset of the subject's expression features with nearest-neighbor expressions retrieved from this bank while still reconstructing the subject's original frames. This exposes the deformation field to a broader range of expression conditions, encouraging stronger identity-expression decoupling and improving robustness to expression distribution shift without requiring paired cross-identity data, additional annotations, or architectural changes. We further analyze how retrieval augmentation increases expression diversity and validate retrieval quality with a user study showing that retrieved neighbors are perceptually closer in expression and pose. Experiments on the NeRSemble benchmark demonstrate that RAF consistently improves expression fidelity over the baseline, in both self-driving and cross-driving scenarios.
>
---
#### [new 269] DocCogito: Aligning Layout Cognition and Step-Level Grounded Reasoning for Document Understanding
- **分类: cs.CV**

- **简介: 该论文属于文档理解任务，旨在解决多模态大模型在高风险场景中缺乏完整、证据驱动的推理问题。提出DocCogito框架，整合布局感知与结构化推理，提升推理准确性与对齐度。**

- **链接: [https://arxiv.org/pdf/2603.07494](https://arxiv.org/pdf/2603.07494)**

> **作者:** Yuchuan Wu; Minghan Zhuo; Teng Fu; Mengyang Zhao; Bin Li; Xiangyang Xue
>
> **摘要:** Document understanding with multimodal large language models (MLLMs) requires not only accurate answers but also explicit, evidence-grounded reasoning, especially in high-stakes scenarios. However, current document MLLMs still fall short of forming a complete, human-like reasoning process, because even when they improve both layout encoding and CoT-style prompting, the interaction between the two is typically learned implicitly and remains loosely coupled rather than being enforced as a systematic mechanism. So we propose DocCogito, a unified framework that integrates global layout perception with structured, region-grounded reasoning. DocCogito introduces a lightweight layout tower that distills page structure into learnable global layout prior tokens, and a deterministic Visual-Semantic Chain (VSC)-a concise structured representation less ambiguous than free-form natural-language CoT-to supervise fine-grained intermediate reasoning aligned with evidence regions. Training follows a progressive recipe, including layout perception pretraining, VSC-guided cold start, rejection sampling, and GRPO. To further strengthen the internal coupling between layout priors and VSC execution, we augment standard rewards with a fine-grained region-confidence signal that encourages reasoning traces to stay aligned with corresponding evidence regions. Extensive experiments on six benchmarks (DocVQA, WTQ, ChartQA, TextVQA, OCRBench, and InfoVQA) demonstrate strong generalization, achieving state-of-the-art results on four benchmarks.
>
---
#### [new 270] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉问答任务，旨在解决模糊视觉问题的策略性回答问题。提出AQuA数据集，分类模糊程度并指导响应策略，提升模型应对不确定性的能力。**

- **链接: [https://arxiv.org/pdf/2603.07394](https://arxiv.org/pdf/2603.07394)**

> **作者:** Jihyoung Jang; Hyounghun Kim
>
> **备注:** ICLR 2026 (28 pages); Project website: this https URL
>
> **摘要:** Visual Question Answering (VQA) is a core task for evaluating the capabilities of Vision-Language Models (VLMs). Existing VQA benchmarks primarily feature clear and unambiguous image-question pairs, whereas real-world scenarios often involve varying degrees of ambiguity that require nuanced reasoning and context-appropriate response strategies. Although recent studies have begun to address ambiguity in VQA, they lack (1) a systematic categorization of ambiguity levels and (2) datasets and models that support strategy-aware responses. In this paper, we introduce Ambiguous Visual Question Answering (AQuA), a fine-grained dataset that classifies ambiguous VQA instances into four levels according to the nature and degree of ambiguity, along with the optimal response strategy for each case. Our evaluation of diverse open-source and proprietary VLMs shows that most models fail to adapt their strategy to the ambiguity type, frequently producing overconfident answers rather than seeking clarification or acknowledging uncertainty. To address this challenge, we fine-tune VLMs on AQuA, enabling them to adaptively choose among multiple response strategies, such as directly answering, inferring intent from contextual cues, listing plausible alternatives, or requesting clarification. VLMs trained on AQuA achieve strategic response generation for ambiguous VQA, demonstrating the ability to recognize ambiguity, manage uncertainty, and respond with context-appropriate strategies, while outperforming both open-source and closed-source baselines.
>
---
#### [new 271] Facial Expression Generation Aligned with Human Preference for Natural Dyadic Interaction
- **分类: cs.CV**

- **简介: 该论文属于人脸表情生成任务，旨在解决自然对话中表情与人类偏好对齐的问题。通过引入人类反馈，构建闭环系统，提升表情生成的自然性和社会适应性。**

- **链接: [https://arxiv.org/pdf/2603.07093](https://arxiv.org/pdf/2603.07093)**

> **作者:** Xu Chen; Rui Gao; Xinjie Zhang; Haoyu Zhang; Che Sun; Zhi Gao; Yuwei Wu; Yunde Jia
>
> **摘要:** Achieving natural dyadic interaction requires generating facial expressions that are emotionally appropriate and socially aligned with human preference. Human feedback offers a compelling mechanism to guide such alignment, yet how to effectively incorporate this feedback into facial expression generation remains underexplored. In this paper, we propose a facial expression generation method aligned with human preference by leveraging human feedback to produce contextually and emotionally appropriate expressions for natural dyadic interaction. A key to our method is framing the generation of identity-independent facial expressions as an action learning process, allowing human feedback to assess their validity free from visual or identity bias. We establish a closed feedback loop in which listener expressions dynamically respond to evolving conversational cues of the speaker. Concretely, we train a vision-language-action model via supervised fine-tuning to map the speaker's multimodal signals into controllable low-dimensional expression representations of a 3D morphable model. We further introduce a human-feedback reinforcement learning strategy that integrates the imitation of high-quality expression response with critic-guided optimization. Experiments on two benchmarks demonstrate that our method effectively aligns facial expressions with human preference and achieves superior performance.
>
---
#### [new 272] ER-Pose: Rethinking Keypoint-Driven Representation Learning for Real-Time Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于实时多人姿态估计任务，解决传统方法因框驱动带来的任务偏差问题。提出关键点驱动的学习范式，优化样本分配和损失函数，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.08681](https://arxiv.org/pdf/2603.08681)**

> **作者:** Nanjun Li; Pinqi Cheng; Zean Liu; Minghe Tian; Xuanyin Wang
>
> **摘要:** Single-stage multi-person pose estimation aims to jointly perform human localization and keypoint prediction within a unified framework, offering advantages in inference efficiency and architectural simplicity. Consequently, multi-scale real-time detection architectures, such as YOLO-like models, are widely adopted for real-time pose estimation. However, these approaches typically inherit a box-driven modeling paradigm from object detection, in which pose estimation is implicitly constrained by bounding-box supervision during training. This formulation introduces biases in sample assignment and feature representation, resulting in task misalignment and ultimately limiting pose estimation accuracy. In this work, we revisit box-driven single-stage pose estimation from a keypoint-driven perspective and identify semantic conflicts among parallel objectives as a key source of performance degradation. To address this issue, we propose a keypoint-driven learning paradigm that elevates pose estimation to a primary prediction objective. Specifically, we remove bounding-box prediction and redesign the prediction head to better accommodate the high-dimensional structured representations for pose estimation. We further introduce a keypoint-driven dynamic sample assignment strategy to align training objectives with pose evaluation metrics, enabling dense supervision during training and efficient NMS-free inference. In addition, we propose a smooth OKS-based loss function to stabilize optimization in regression-based pose estimation. Based on these designs, we develop a single-stage multi-person pose estimation framework, termed ER-Pose. On MS COCO and CrowdPose, ER-Pose-n achieves AP improvements of 3.2/6.7 without pre-training and 7.4/4.9 with pre-training respectively compared with the baseline YOLO-Pose. These improvements are achieved with fewer parameters and higher inference efficiency.
>
---
#### [new 273] Beyond Hungarian: Match-Free Supervision for End-to-End Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决DETR框架中依赖匈牙利算法带来的计算开销和训练复杂问题。提出一种无需显式匹配的训练方案，通过交叉注意力机制实现端到端检测。**

- **链接: [https://arxiv.org/pdf/2603.08514](https://arxiv.org/pdf/2603.08514)**

> **作者:** Shoumeng Qiu; Xinrun Li; Yang Long
>
> **摘要:** Recent DEtection TRansformer (DETR) based frameworks have achieved remarkable success in end-to-end object detection. However, the reliance on the Hungarian algorithm for bipartite matching between queries and ground truths introduces computational overhead and complicates the training dynamics. In this paper, we propose a novel matching-free training scheme for DETR-based detectors that eliminates the need for explicit heuristic matching. At the core of our approach is a dedicated Cross-Attention-based Query Selection (CAQS) module. Instead of discrete assignment, we utilize encoded ground-truth information to probe the decoder queries through a cross-attention mechanism. By minimizing the weighted error between the queried results and the ground truths, the model autonomously learns the implicit correspondences between object queries and specific targets. This learned relationship further provides supervision signals for the learning of queries. Experimental results demonstrate that our proposed method bypasses the traditional matching process, significantly enhancing training efficiency, reducing the matching latency by over 50\%, effectively eliminating the discrete matching bottleneck through differentiable correspondence learning, and also achieving superior performance compared to existing state-of-the-art methods.
>
---
#### [new 274] SPIRAL: A Closed-Loop Framework for Self-Improving Action World Models via Reflective Planning Agents
- **分类: cs.CV**

- **简介: 该论文提出SPIRAL框架，解决长时视频生成中的语义对齐与时间一致性问题，通过闭环计划与反思机制提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.08403](https://arxiv.org/pdf/2603.08403)**

> **作者:** Yu Yang; Yue Liao; Jianbiao Mei; Baisen Wang; Xuemeng Yang; Licheng Wen; Jiangning Zhang; Xiangtai Li; Hanlin Chen; Botian Shi; Yong Liu; Shuicheng Yan; Gim Hee Lee
>
> **备注:** 22 Pages, 11 Figures
>
> **摘要:** We introduce SPIRAL, a self-improving planning and iterative reflective action world modeling closed-loop framework that enables controllable long-horizon video generation conditioned on high-level semantic actions. Existing one-shot video generation models operate in open-loop, often resulting in incomplete action execution, weak semantic grounding, and temporal drift. SPIRAL formulates ActWM as a closed-loop think-act-reflect process, where generation proceeds step by step under explicit planning and feedback. A PlanAgent decomposes abstract actions into object-centric sub-actions, while a CriticAgent evaluates intermediate results and guides iterative refinement with long-horizon memory. This closed-loop design naturally supports RL evolving optimization, improving semantic alignment and temporal consistency over extended horizons. We further introduce the ActWM-Dataset and ActWM-Bench for training and evaluation. Experiments across multiple TI2V backbones demonstrate consistent gains on ActWM-Bench and mainstream video generation benchmarks, validating SPIRAL's effectiveness.
>
---
#### [new 275] VesselFusion: Diffusion Models for Vessel Centerline Extraction from 3D CT Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，解决3D CT图像中血管中心线提取问题。提出VesselFusion方法，利用扩散模型实现更准确、自然的提取。**

- **链接: [https://arxiv.org/pdf/2603.08135](https://arxiv.org/pdf/2603.08135)**

> **作者:** Soichi Mita; Shumpei Takezaki; Ryoma Bise
>
> **摘要:** Vessel centerline extraction from 3D CT images is an important task because it reduces annotation effort to build a model that estimates a vessel structure. It is challenging to estimate natural vessel structures since conventional approaches are deterministic models, which cannot capture a complex human structure. In this study, we propose VesselFusion, which is a diffusion model to extract the vessel centerline from 3D CT image. The proposed method uses a coarse-to-fine representation of the centerline and a voting-based aggregation for a natural and stable extraction. VesselFusion was evaluated on a publicly available CT image dataset and achieved higher extraction accuracy and a more natural result than conventional approaches.
>
---
#### [new 276] Classifying Novel 3D-Printed Objects without Retraining: Towards Post-Production Automation in Additive Manufacturing
- **分类: cs.CV**

- **简介: 论文研究3D打印物体分类任务，解决工业制造中因对象变化频繁导致的模型需频繁重训练问题。通过引入新数据集ThingiPrint和对比微调方法，实现无需重训练的高效分类。**

- **链接: [https://arxiv.org/pdf/2603.07465](https://arxiv.org/pdf/2603.07465)**

> **作者:** Fanis Mathioulakis; Gorjan Radevski; Silke GC Cleuren; Michel Janssens; Brecht Das; Koen Schauwaert; Tinne Tuytelaars
>
> **摘要:** Reliable classification of 3D-printed objects is essential for automating post-production workflows in industrial additive manufacturing. Despite extensive automation in other stages of the printing pipeline, this task still relies heavily on manual inspection, as the set of objects to be classified can change daily, making frequent model retraining impractical. Automating the identification step is therefore critical for improving operational efficiency. A vision model that could classify any set of objects by utilizing their corresponding CAD models and avoiding retraining would be highly beneficial in this setting. To enable systematic evaluation of vision models on this task, we introduce ThingiPrint, a new publicly available dataset that pairs CAD models with real photographs of their 3D-printed counterparts. Using ThingiPrint, we benchmark a range of existing vision models on the task of 3D-printed object classification. We additionally show that contrastive fine-tuning with a rotation-invariant objective allows effective prototype-based classification of previously unseen 3D-printed objects. By relying solely on the available CAD models, this avoids the need for retraining when new objects are introduced. Experiments show that this approach outperforms standard pretrained baselines, suggesting improved generalization and practical relevance for real-world use.
>
---
#### [new 277] Improving Continual Learning for Gaussian Splatting based Environments Reconstruction on Commercial Off-the-Shelf Edge Devices
- **分类: cs.CV**

- **简介: 该论文针对边缘设备上的3D场景重建任务，解决VBGS算法在资源受限设备上训练困难的问题。通过优化框架降低内存和时间开销，提升训练效率与可行性。**

- **链接: [https://arxiv.org/pdf/2603.08499](https://arxiv.org/pdf/2603.08499)**

> **作者:** Ivan Zaino; Matteo Risso; Daniele Jahier Pagliari; Miguel de Prado; Toon Van de Maele; Alessio Burrello
>
> **摘要:** Novel view synthesis (NVS) is increasingly relevant for edge robotics, where compact and incrementally updatable 3D scene models are needed for SLAM, navigation, and inspection under tight memory and latency budgets. Variational Bayesian Gaussian Splatting (VBGS) enables replay-free continual updates for the 3DGS algorithm by maintaining a probabilistic scene model, but its high-precision computations and large intermediate tensors make on-device training impractical. We present a precision-adaptive optimization framework that enables VBGS training on resource-constrained hardware without altering its variational formulation. We (i) profile VBGS to identify memory/latency hotspots, (ii) fuse memory-dominant kernels to reduce materialized intermediate tensors, and (iii) automatically assign operation-level precisions via a mixed-precision search with bounded relative error. Across the Blender, Habitat, and Replica datasets, our optimised pipeline reduces peak memory from 9.44 GB to 1.11 GB and training time from ~234 min to ~61 min on an A5000 GPU, while preserving (and in some cases improving) reconstruction quality of the state-of-the-art VBGS baseline. We also enable for the first time NVS training on a commercial embedded platform, the Jetson Orin Nano, reducing per-frame latency by 19x compared to 3DGS.
>
---
#### [new 278] One-Shot Badminton Shuttle Detection for Mobile Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，解决移动机器人视角下羽毛球的实时检测问题。构建了首个相关数据集，提出优化的YOLOv8模型，在不同环境中实现高效检测。**

- **链接: [https://arxiv.org/pdf/2603.06691](https://arxiv.org/pdf/2603.06691)**

> **作者:** Florentin Dipner; William Talbot; Turcan Tuna; Andrei Cramariuc; Marco Hutter
>
> **备注:** Under review for IEEE R-AP
>
> **摘要:** This paper presents a robust one-shot badminton shuttlecock detection framework for non-stationary robots. To address the lack of egocentric shuttlecock detection datasets, we introduce a dataset of 20,510 semi-automatically annotated frames captured across 11 distinct backgrounds in diverse indoor and outdoor environments, and categorize each frame into one of three difficulty levels. For labeling, we present a novel semi-automatic annotation pipeline, that enables efficient labeling from stationary camera footage. We propose a metric suited to our downstream use case and fine-tune a YOLOv8 network optimized for real-time shuttlecock detection, achieving an F1-score of 0.86 under our metric in test environments similar to training, and 0.70 in entirely unseen environments. Our analysis reveals that detection performance is critically dependent on shuttlecock size and background texture complexity. Qualitative experiments confirm their applicability to robots with moving cameras. Unlike prior work with stationary camera setups, our detector is specifically designed for the egocentric, dynamic viewpoints of mobile robots, providing a foundational building block for downstream tasks, including tracking, trajectory estimation, and system (re)-initialization.
>
---
#### [new 279] RobustSCI: Beyond Reconstruction to Restoration for Snapshot Compressive Imaging under Real-World Degradations
- **分类: cs.CV**

- **简介: 该论文属于视频快照压缩成像任务，旨在解决真实场景中运动模糊和低光导致的图像退化问题。通过构建基准数据集并提出RobustSCI网络，实现从退化测量中恢复原始场景。**

- **链接: [https://arxiv.org/pdf/2603.07489](https://arxiv.org/pdf/2603.07489)**

> **作者:** Hao Wang; Yuanfan Li; Qi Zhou; Zhankuo Xu; Jiong Ni; Xin Yuan
>
> **摘要:** Deep learning algorithms for video Snapshot Compressive Imaging (SCI) have achieved great success, yet they predominantly focus on reconstructing from clean measurements. This overlooks a critical real-world challenge: the captured signal itself is often severely degraded by motion blur and low light. Consequently, existing models falter in practical applications. To break this limitation, we pioneer the first study on robust video SCI restoration, shifting the goal from "reconstruction" to "restoration"--recovering the underlying pristine scene from a degraded measurement. To facilitate this new task, we first construct a large-scale benchmark by simulating realistic, continuous degradations on the DAVIS 2017 dataset. Second, we propose RobustSCI, a network that enhances a strong encoder-decoder backbone with a novel RobustCFormer block. This block introduces two parallel branches--a multi-scale deblur branch and a frequency enhancement branch--to explicitly disentangle and remove degradations during the recovery process. Furthermore, we introduce RobustSCI-C (RobustSCI-Cascade), which integrates a pre-trained Lightweight Post-processing Deblurring Network to significantly boost restoration performance with minimal overhead. Extensive experiments demonstrate that our methods outperform all SOTA models on the new degraded testbeds, with additional validation on real-world degraded SCI data confirming their practical effectiveness, elevating SCI from merely reconstructing what is captured to restoring what truly happened.
>
---
#### [new 280] AdaGen: Learning Adaptive Policy for Image Synthesis
- **分类: cs.CV**

- **简介: 该论文提出AdaGen，用于图像生成中的自适应策略学习，解决传统方法依赖手动规则和静态调度的问题。通过强化学习和对抗奖励设计，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.06993](https://arxiv.org/pdf/2603.06993)**

> **作者:** Zanlin Ni; Yulin Wang; Yeguo Hua; Renping Zhou; Jiayi Guo; Jun Song; Bo Zheng; Gao Huang
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Journal version of arXiv:2409.00342 (ECCV 2024). Code is available at: this https URL
>
> **摘要:** Recent advances in image synthesis have been propelled by powerful generative models, such as Masked Generative Transformers (MaskGIT), autoregressive models, diffusion models, and rectified flow models. A common principle behind their success is the decomposition of synthesis into multiple steps. However, this introduces a proliferation of step-specific parameters (e.g., noise level or temperature at each step). Existing approaches typically rely on manually-designed rules to manage this complexity, demanding expert knowledge and trial-and-error. Furthermore, these static schedules lack the flexibility to adapt to the unique characteristics of each sample, yielding sub-optimal performance. To address this issue, we present AdaGen, a general, learnable, and sample-adaptive framework for scheduling the iterative generation process. Specifically, we formulate the scheduling problem as a Markov Decision Process, where a lightweight policy network determines suitable parameters given the current generation state, and can be trained through reinforcement learning. Importantly, we demonstrate that simple reward designs, such as FID or pre-trained reward models, can be easily hacked and may not reliably guarantee the desired quality or diversity of generated samples. Therefore, we propose an adversarial reward design to guide the training of the policy networks. Finally, we introduce an inference-time refinement strategy and a controllable fidelity-diversity trade-off mechanism to further enhance the performance and flexibility of AdaGen. Comprehensive experiments on four generative paradigms validate the superiority of AdaGen. For example, AdaGen achieves better performance on DiT-XL with 3 times lower inference cost and improves the FID of VAR from 1.92 to 1.59 with negligible computational overhead.
>
---
#### [new 281] DLRMamba: Distilling Low-Rank Mamba for Edge Multispectral Fusion Object Detection
- **分类: cs.CV**

- **简介: 论文聚焦于边缘多光谱目标检测任务，针对Mamba模型参数冗余和压缩后信息丢失问题，提出低秩SS2D结构与结构感知蒸馏方法，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.06920](https://arxiv.org/pdf/2603.06920)**

> **作者:** Qianqian Zhang; Leon Tabaro; Ahmed M. Abdelmoniem; Junshe An
>
> **备注:** Has been submitted to the IEEE TGRS journal
>
> **摘要:** Multispectral fusion object detection is a critical task for edge-based maritime surveillance and remote sensing, demanding both high inference efficiency and robust feature representation for high-resolution inputs. However, current State Space Models (SSMs) like Mamba suffer from significant parameter redundancy in their standard 2D Selective Scan (SS2D) blocks, which hinders deployment on resource-constrained hardware and leads to the loss of fine-grained structural information during conventional compression. To address these challenges, we propose the Low-Rank Two-Dimensional Selective Structured State Space Model (Low-Rank SS2D), which reformulates state transitions via matrix factorization to exploit intrinsic feature sparsity. Furthermore, we introduce a Structure-Aware Distillation strategy that aligns the internal latent state dynamics of the student with a full-rank teacher model to compensate for potential representation degradation. This approach substantially reduces computational complexity and memory footprint while preserving the high-fidelity spatial modeling required for object recognition. Extensive experiments on five benchmark datasets and real-world edge platforms, such as Raspberry Pi 5, demonstrate that our method achieves a superior efficiency-accuracy trade-off, significantly outperforming existing lightweight architectures in practical deployment scenarios.
>
---
#### [new 282] Shaping Parameter Contribution Patterns for Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决深度模型过早自信的问题。通过引入SPCP方法，使分类器学习更密集的参数贡献模式，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07195](https://arxiv.org/pdf/2603.07195)**

> **作者:** Haonan Xu; Yang Yang
>
> **摘要:** Out-of-distribution (OOD) detection is a well-known challenge due to deep models often producing overconfident. In this paper, we reveal a key insight that trained classifiers tend to rely on sparse parameter contribution patterns, meaning that only a few dominant parameters drive predictions. This brittleness can be exploited by OOD inputs that anomalously trigger these parameters, resulting in overconfident predictions. To address this issue, we propose a simple yet effective method called Shaping Parameter Contribution Patterns (SPCP), which enhances OOD detection robustness by encouraging the classifier to learn boundary-oriented dense contribution patterns. Specifically, SPCP operates during training by rectifying excessively high parameter contributions based on a dynamically estimated threshold. This mechanism promotes the classifier to rely on a broader set of parameters for decision-making, thereby reducing the risk of overconfident predictions caused by anomalously triggered parameters, while preserving in-distribution (ID) performance. Extensive experiments under various OOD detection setups verify the effectiveness of SPCP.
>
---
#### [new 283] N-Tree Diffusion for Long-Horizon Wildfire Risk Forecasting
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于长期野火风险预测任务，解决多时间步预测中的计算冗余问题。提出NT-Diffusion模型，通过共享早期去噪阶段提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2603.07361](https://arxiv.org/pdf/2603.07361)**

> **作者:** Yucheng Xing; Xin Wang
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Long-horizon wildfire risk forecasting requires generating probabilistic spatial fields under sparse event supervision while maintaining computational efficiency across multiple prediction horizons. Extending diffusion models to multi-step forecasting typically repeats the denoising process independently for each horizon, leading to redundant computation. We introduce N-Tree Diffusion (NT-Diffusion), a hierarchical diffusion model designed for long-horizon wildfire risk forecasting. Fire occurrences are represented as continuous Fire Risk Maps (FRMs), which provide a smoothed spatial risk field suitable for probabilistic modeling. Instead of running separate diffusion trajectories for each predicted timestamp, NT-Diffusion shares early denoising stages and branches at later levels, allowing horizon-specific refinement while reducing redundant sampling. We evaluate the proposed framework on a newly collected real-world wildfire dataset constructed for long-horizon probabilistic prediction. Results indicate that NT-Diffusion achieves consistent accuracy improvements and reduced inference cost compared to baseline forecasting approaches.
>
---
#### [new 284] Interactive World Simulator for Robot Policy Training and Evaluation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出Interactive World Simulator，用于机器人策略训练与评估。解决传统世界模型速度慢、物理一致性差的问题，通过一致性模型实现快速稳定模拟，支持长时序交互。**

- **链接: [https://arxiv.org/pdf/2603.08546](https://arxiv.org/pdf/2603.08546)**

> **作者:** Yixuan Wang; Rhythm Syed; Fangyu Wu; Mengchao Zhang; Aykut Onol; Jose Barreiros; Hooshang Nayyeri; Tony Dear; Huan Zhang; Yunzhu Li
>
> **备注:** Project Page: this https URL
>
> **摘要:** Action-conditioned video prediction models (often referred to as world models) have shown strong potential for robotics applications, but existing approaches are often slow and struggle to capture physically consistent interactions over long horizons, limiting their usefulness for scalable robot policy training and evaluation. We present Interactive World Simulator, a framework for building interactive world models from a moderate-sized robot interaction dataset. Our approach leverages consistency models for both image decoding and latent-space dynamics prediction, enabling fast and stable simulation of physical interactions. In our experiments, the learned world models produce interaction-consistent pixel-level predictions and support stable long-horizon interactions for more than 10 minutes at 15 FPS on a single RTX 4090 GPU. Our framework enables scalable demonstration collection solely within the world models to train state-of-the-art imitation policies. Through extensive real-world evaluation across diverse tasks involving rigid objects, deformable objects, object piles, and their interactions, we find that policies trained on world-model-generated data perform comparably to those trained on the same amount of real-world data. Additionally, we evaluate policies both within the world models and in the real world across diverse tasks, and observe a strong correlation between simulated and real-world performance. Together, these results establish the Interactive World Simulator as a stable and physically consistent surrogate for scalable robotic data generation and faithful, reproducible policy evaluation.
>
---
#### [new 285] Heterogeneous Decentralized Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究生成模型训练任务，解决资源消耗高和训练目标单一的问题。提出一种异构去中心化框架，支持不同训练目标并降低计算需求。**

- **链接: [https://arxiv.org/pdf/2603.06741](https://arxiv.org/pdf/2603.06741)**

> **作者:** Zhiying Jiang; Raihan Seraj; Marcos Villagra; Bidhan Roy
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Training frontier-scale diffusion models often requires substantial computational resources concentrated in tightly coupled clusters, limiting participation to well-resourced institutions. While Decentralized Diffusion Models (DDM) enable training multiple experts in isolation, existing approaches require 1176 GPU-days and homogeneous training objectives across all experts. We present an efficient framework that reduces resource requirements while supporting heterogeneous training objectives. Our approach combines three contributions: (1) a heterogeneous decentralized training paradigm that allows experts to use different objectives (DDPM and Flow Matching), unified at inference time via a deterministic schedule-aware conversion into a common velocity space without retraining; (2) pretrained checkpoint conversion from ImageNet-DDPM to Flow Matching objectives, accelerating convergence and enabling initialization without objective-specific pretraining; and (3) PixArt-alpha's efficient AdaLN-Single architecture, reducing parameters while maintaining quality. Experiments on LAION-Aesthetics show that, relative to the training scale reported for prior DDM work, our approach reduces compute from 1176 to 72 GPU-days (16x) and data from 158M to 11M (14x). Under aligned inference settings, our heterogeneous 2DDPM:6FM configuration achieves better FID (11.88 vs. 12.45) and higher intra-prompt diversity (LPIPS 0.631 vs. 0.617) than the homogeneous 8FM baseline. By eliminating synchronization requirements and enabling mixed DDPM/FM objectives, our framework lowers infrastructure requirements for decentralized generative model training.
>
---
#### [new 286] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决现有方法感知效率低、推理不聚焦的问题。提出ProFocus框架，结合大语言模型和视觉语言模型，实现主动感知和聚焦推理。**

- **链接: [https://arxiv.org/pdf/2603.05530](https://arxiv.org/pdf/2603.05530)**

> **作者:** Wei Xue; Mingcheng Li; Xuecheng Wu; Jingqun Tang; Dingkang Yang; Lihua Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to accurately perceive complex visual environments and reason over navigation instructions and histories. However, existing methods passively process redundant visual inputs and treat all historical contexts indiscriminately, resulting in inefficient perception and unfocused reasoning. To address these challenges, we propose \textbf{ProFocus}, a training-free progressive framework that unifies \underline{Pro}active Perception and \underline{Focus}ed Reasoning through collaboration between large language models (LLMs) and vision-language models (VLMs). For proactive perception, ProFocus transforms panoramic observations into structured ego-centric semantic maps, enabling the orchestration agent to identify missing visual information needed for reliable decision-making, and to generate targeted visual queries with corresponding focus regions that guide the perception agent to acquire the required observations. For focused reasoning, we propose Branch-Diverse Monte Carlo Tree Search (BD-MCTS) to identify top-$k$ high-value waypoints from extensive historical candidates. The decision agent focuses reasoning on the historical contexts associated with these waypoints, rather than considering all historical waypoints equally. Extensive experiments validate the effectiveness of ProFocus, achieving state-of-the-art performance among zero-shot methods on R2R and REVERIE benchmarks.
>
---
#### [new 287] HiDE: Hierarchical Dictionary-Based Entropy Modeling for Learned Image Compression
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于图像压缩任务，旨在解决熵模型中外部先验利用不足的问题。提出HiDE框架，通过分层字典和上下文感知参数估计提升压缩性能。**

- **链接: [https://arxiv.org/pdf/2603.06766](https://arxiv.org/pdf/2603.06766)**

> **作者:** Haoxuan Xiong; Yuanyuan Xu; Kun Zhu; Yiming Wang; Baoliu Ye
>
> **摘要:** Learned image compression (LIC) has achieved remarkable coding efficiency, where entropy modeling plays a pivotal role in minimizing bitrate through informative priors. Existing methods predominantly exploit internal contexts within the input image, yet the rich external priors embedded in large-scale training data remain largely underutilized. Recent advances in dictionary-based entropy models have demonstrated that incorporating external priors can substantially enhance compression performance. However, current approaches organize heterogeneous external priors within a single-level dictionary, resulting in imbalanced utilization and limited representational capacity. Moreover, effective entropy modeling requires not only expressive priors but also a parameter estimation network capable of interpreting them. To address these challenges, we propose HiDE, a Hierarchical Dictionary-based Entropy modeling framework for learned image compression. HiDE decomposes external priors into global structural and local detail dictionaries with cascaded retrieval, enabling structured and efficient utilization of external information. Moreover, a context-aware parameter estimator with parallel multi-receptive-field design is introduced to adaptively exploit heterogeneous contexts for accurate conditional probability estimation. Experimental results show that HiDE achieves 18.5%, 21.99%, and 24.01% BD-rate savings over VTM-12.1 on the Kodak, CLIC, and Tecnick datasets, respectively.
>
---
#### [new 288] UniGround: Universal 3D Visual Grounding via Training-Free Scene Parsing
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UniGround，解决3D视觉定位任务中的泛化与鲁棒性问题，通过无需训练的视觉和几何推理实现开放世界3D定位。**

- **链接: [https://arxiv.org/pdf/2603.08131](https://arxiv.org/pdf/2603.08131)**

> **作者:** Jiaxi Zhang; Yunheng Wang; Wei Lu; Taowen Wang; Weisheng Xu; Shuning Zhang; Yixiao Feng; Yuetong Fang; Renjing Xu
>
> **备注:** 14 pages,6 figures,3 tables
>
> **摘要:** Understanding and localizing objects in complex 3D environments from natural language descriptions, known as 3D Visual Grounding (3DVG), is a foundational challenge in embodied AI, with broad implications for robotics, augmented reality, and human-machine interaction. Large-scale pre-trained foundation models have driven significant progress on this front, enabling open-vocabulary 3DVG that allows systems to locate arbitrary objects in a given scene. However, their reliance on pre-trained models constrains 3D perception and reasoning within the inherited knowledge boundaries, resulting in limited generalization to unseen spatial relationships and poor robustness to out-of-distribution scenes. In this paper, we replace this constrained perception with training-free visual and geometric reasoning, thereby unlocking open-world 3DVG that enables the localization of any object in any scene beyond the training data. Specifically, the proposed UniGround operates in two stages: a Global Candidate Filtering stage that constructs scene candidates through training-free 3D topology and multi-view semantic encoding, and a Local Precision Grounding stage that leverages multi-scale visual prompting and structured reasoning to precisely identify the target object. Experiments on ScanRefer and EmbodiedScan show that UniGround achieves 46.1\%/34.1\% Acc@0.25/0.5 on ScanRefer and 28.7\% Acc@0.25 on EmbodiedScan, establishing a new state-of-the-art among zero-shot methods on EmbodiedScan without any 3D supervision. We further evaluate UniGround in real-world environments under uncontrolled reconstruction conditions and substantial domain shift, showing training-free reasoning generalizes robustly beyond curated benchmarks.
>
---
#### [new 289] mAVE: A Watermark for Joint Audio-Visual Generation Models
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于多媒体水印任务，解决联合音视频生成模型的版权保护问题。针对现有技术的模态解耦缺陷，提出mAVE框架，实现音视频潜伏的加密绑定，提升内容溯源安全性。**

- **链接: [https://arxiv.org/pdf/2603.07090](https://arxiv.org/pdf/2603.07090)**

> **作者:** Luyang Si; Leyi Pan; Lijie Wen
>
> **摘要:** As Joint Audio-Visual Generation Models see widespread commercial deployment, embedding watermarks has become essential for protecting vendor copyright and ensuring content provenance. However, existing techniques suffer from an architectural mismatch by treating modalities as decoupled entities, exposing a critical Binding Vulnerability. Adversaries exploit this via Swap Attacks by replacing authentic audio with malicious deepfakes while retaining the watermarked video. Because current detectors rely on independent verification ($Video_{wm}\vee Audio_{wm}$), they incorrectly authenticate the manipulated content, falsely attributing harmful media to the original vendor and severely damaging their reputation. To address this, we propose mAVE (Manifold Audio-Visual Entanglement), the first watermarking framework natively designed for joint architectures. mAVE cryptographically binds audio and video latents at initialization without fine-tuning, defining a Legitimate Entanglement Manifold via Inverse Transform Sampling. Experiments on state-of-the-art models (LTX-2, MOVA) demonstrate that mAVE guarantees performance-losslessness and provides an exponential security bound against Swap Attacks. Achieving near-perfect binding integrity ($>99\%$), mAVE offers a robust cryptographic defense for vendor copyright.
>
---
#### [new 290] Conditional Unbalanced Optimal Transport Maps: An Outlier-Robust Framework for Conditional Generative Modeling
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于条件生成建模任务，旨在解决传统条件最优传输对异常值敏感的问题。通过引入CUOT框架，放松分布匹配约束，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.06972](https://arxiv.org/pdf/2603.06972)**

> **作者:** Jiwoo Yoon; Kyumin Choi; Jaewoong Choi
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Conditional Optimal Transport (COT) problem aims to find a transport map between conditional source and target distributions while minimizing the transport cost. Recently, these transport maps have been utilized in conditional generative modeling tasks to establish efficient mappings between the distributions. However, classical COT inherits a fundamental limitation of optimal transport, i.e., sensitivity to outliers, which arises from the hard distribution matching constraints. This limitation becomes more pronounced in a conditional setting, where each conditional distribution is estimated from a limited subset of data. To address this, we introduce the Conditional Unbalanced Optimal Transport (CUOT) framework, which relaxes conditional distribution-matching constraints through Csiszár divergence penalties while strictly preserving the conditioning marginals. We establish a rigorous formulation of the CUOT problem and derive its dual and semi-dual formulations. Based on the semi-dual form, we propose Conditional Unbalanced Optimal Transport Maps (CUOTM), an outlier-robust conditional generative model built upon a triangular $c$-transform parameterization. We theoretically justify the validity of this parameterization by proving that the optimal triangular map satisfies the $c$-transform relationships. Our experiments on 2D synthetic and image-scale datasets demonstrate that CUOTM achieves superior outlier robustness and competitive distribution-matching performance compared to existing COT-based baselines, while maintaining high sampling efficiency.
>
---
#### [new 291] SlowBA: An efficiency backdoor attack towards VLM-based GUI agents
- **分类: cs.CR; cs.CL; cs.CV**

- **简介: 该论文属于GUI安全任务，旨在解决VLM代理响应效率被攻击的问题。提出SlowBA攻击方法，通过诱导长推理链增加延迟，同时保持任务准确性。**

- **链接: [https://arxiv.org/pdf/2603.08316](https://arxiv.org/pdf/2603.08316)**

> **作者:** Junxian Li; Tu Lan; Haozhen Tan; Yan Meng; Haojin Zhu
>
> **备注:** 25 pages
>
> **摘要:** Modern vision-language-model (VLM) based graphical user interface (GUI) agents are expected not only to execute actions accurately but also to respond to user instructions with low latency. While existing research on GUI-agent security mainly focuses on manipulating action correctness, the security risks related to response efficiency remain largely unexplored. In this paper, we introduce SlowBA, a novel backdoor attack that targets the responsiveness of VLM-based GUI agents. The key idea is to manipulate response latency by inducing excessively long reasoning chains under specific trigger patterns. To achieve this, we propose a two-stage reward-level backdoor injection (RBI) strategy that first aligns the long-response format and then learns trigger-aware activation through reinforcement learning. In addition, we design realistic pop-up windows as triggers that naturally appear in GUI environments, improving the stealthiness of the attack. Extensive experiments across multiple datasets and baselines demonstrate that SlowBA can significantly increase response length and latency while largely preserving task accuracy. The attack remains effective even with a small poisoning ratio and under several defense settings. These findings reveal a previously overlooked security vulnerability in GUI agents and highlight the need for defenses that consider both action correctness and response efficiency. Code can be found in this https URL.
>
---
#### [new 292] See and Switch: Vision-Based Branching for Interactive Robot-Skill Programming
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人技能编程任务，解决现实环境下的分支选择与异常检测问题。提出See & Switch框架，利用视觉实现交互式条件分支，提升机器人执行灵活性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08057](https://arxiv.org/pdf/2603.08057)**

> **作者:** Petr Vanc; Jan Kristof Behrens; Václav Hlaváč; Karla Stepanova
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** Programming robots by demonstration (PbD) is an intuitive concept, but scaling it to real-world variability remains a challenge for most current teaching frameworks. Conditional task graphs are very expressive and can be defined incrementally, which fits very well with the PbD idea. However, acting using conditional task graphs requires reliable perception-grounded online branch selection. In this paper, we present See & Switch, an interactive teaching-and-execution framework that represents tasks as user-extendable graphs of skill parts connected via decision states (DS), enabling conditional branching during replay. Unlike prior approaches that rely on manual branching or low-dimensional signals (e.g., proprioception), our vision-based Switcher uses eye-in-hand images (high-dimensional) to select among competing successor skill parts and to detect out-of-distribution contexts that require new demonstrations. We integrate kinesthetic teaching, joystick control, and hand gestures via an input-modality-abstraction layer and demonstrate that our proposed method is teaching modality-independent, enabling efficient in-situ recovery demonstrations. The system is validated in experiments on three challenging dexterous manipulation tasks. We evaluate our method under diverse conditions and furthermore conduct user studies with 8 participants. We show that the proposed method reliably performs branch selection and anomaly detection for novice users, achieving 90.7 % and 87.9 % accuracy, respectively, across 576 real-robot rollouts. We provide all code and data required to reproduce our experiments at this http URL.
>
---
#### [new 293] Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出一种基于扩散模型的隐式视觉表示框架，用于高效视频压缩。任务是视觉压缩与生成的统一，解决传统表示无法充分利用模型知识的问题。通过低秩适配和函数编码实现紧凑存储与灵活控制。**

- **链接: [https://arxiv.org/pdf/2603.07615](https://arxiv.org/pdf/2603.07615)**

> **作者:** Jiajun He; Zongyu Guo; Zhaoyang Jia; Xiaoyi Zhang; Jiahao Li; Xiao Li; Bin Li; José Miguel Hernández-Lobato; Yan Lu
>
> **摘要:** Modern visual generative models acquire rich visual knowledge through large-scale training, yet existing visual representations (such as pixels, latents, or tokens) remain external to the model and cannot directly exploit this knowledge for compact storage or reuse. In this work, we introduce a new visual representation framework that encodes a signal as a function, which is parametrized by low-rank adaptations attached to a frozen visual generative model. Such implicit representations of visual signals, \textit{e.g.}, an 81-frame video, can further be hashed into a single compact vector, achieving strong perceptual video compression at extremely low bitrates. Beyond basic compression, the functional nature of this representation enables inference-time scaling and control, allowing additional refinement on the compression performance. More broadly, as the implicit representations directly act as a function of the generation process, this suggests a unified framework bridging visual compression and generation.
>
---
#### [new 294] Topologically Stable Hough Transform
- **分类: cs.CG; cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决点云中直线检测的问题。通过引入连续得分函数替代传统离散投票机制，结合持久同调方法提取稳定特征，提出高效算法生成候选直线。**

- **链接: [https://arxiv.org/pdf/2603.08245](https://arxiv.org/pdf/2603.08245)**

> **作者:** Stefan Huber; Kristóf Huszár; Michael Kerber; Martin Uray
>
> **备注:** Extended abstract will be presented at EuroCG'26; 11 pages, 7 figures
>
> **摘要:** We propose an alternative formulation of the well-known Hough transform to detect lines in point clouds. Replacing the discretized voting scheme of the classical Hough transform by a continuous score function, its persistent features in the sense of persistent homology give a set of candidate lines. We also devise and implement an algorithm to efficiently compute these candidate lines.
>
---
#### [new 295] Two Frames Matter: A Temporal Attack for Text-to-Video Model Jailbreaking
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于文本到视频模型安全研究，解决模型在接收到碎片化提示时生成有害内容的问题。通过提出TFM框架，提升越狱攻击效果。**

- **链接: [https://arxiv.org/pdf/2603.07028](https://arxiv.org/pdf/2603.07028)**

> **作者:** Moyang Chen; Zonghao Ying; Wenzhuo Xu; Quancheng Zou; Deyue Zhang; Dongdong Yang; Xiangzheng Zhang
>
> **摘要:** Recent text-to-video (T2V) models can synthesize complex videos from lightweight natural language prompts, raising urgent concerns about safety alignment in the event of misuse in the real world. Prior jailbreak attacks typically rewrite unsafe prompts into paraphrases that evade content filters while preserving meaning. Yet, these approaches often still retain explicit sensitive cues in the input text and therefore overlook a more profound, video-specific weakness. In this paper, we identify a temporal trajectory infilling vulnerability of T2V systems under fragmented prompts: when the prompt specifies only sparse boundary conditions (e.g., start and end frames) and leaves the intermediate evolution underspecified, the model may autonomously reconstruct a plausible trajectory that includes harmful intermediate frames, despite the prompt appearing benign to input or output side filtering. Building on this observation, we propose TFM. This fragmented prompting framework converts an originally unsafe request into a temporally sparse two-frame extraction and further reduces overtly sensitive cues via implicit substitution. Extensive evaluations across multiple open-source and commercial T2V models demonstrate that TFM consistently enhances jailbreak effectiveness, achieving up to a 12% increase in attack success rate on commercial systems. Our findings highlight the need for temporally aware safety mechanisms that account for model-driven completion beyond prompt surface form.
>
---
#### [new 296] Grow, Assess, Compress: Adaptive Backbone Scaling for Memory-Efficient Class Incremental Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于类增量学习任务，旨在解决模型在学习新类别时遗忘旧知识的问题。提出GRACE框架，通过动态扩展与压缩模型结构，有效减少内存占用并提升性能。**

- **链接: [https://arxiv.org/pdf/2603.08426](https://arxiv.org/pdf/2603.08426)**

> **作者:** Adrian Garcia-Castañeda; Jon Irureta; Jon Imaz; Aizea Lojo
>
> **摘要:** Class Incremental Learning (CIL) poses a fundamental challenge: maintaining a balance between the plasticity required to learn new tasks and the stability needed to prevent catastrophic forgetting. While expansion-based methods effectively mitigate forgetting by adding task-specific parameters, they suffer from uncontrolled architectural growth and memory overhead. In this paper, we propose a novel dynamic scaling framework that adaptively manages model capacity through a cyclic "GRow, Assess, ComprEss" (GRACE) strategy. Crucially, we supplement backbone expansion with a novel saturation assessment phase that evaluates the utilization of the model's capacity. This assessment allows the framework to make informed decisions to either expand the architecture or compress the backbones into a streamlined representation, preventing parameter explosion. Experimental results demonstrate that our approach achieves state-of-the-art performance across multiple CIL benchmarks, while reducing memory footprint by up to a 73% compared to purely expansionist models.
>
---
#### [new 297] RECAP: Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics
- **分类: cs.NE; cs.AI; cs.CV; cs.LG; q-bio.NC**

- **简介: 该论文提出RECAP方法，用于图像分类任务，解决传统系统依赖反向传播的问题。通过结合非训练的动态池和自组织Hebbian原型读出，实现鲁棒分类。**

- **链接: [https://arxiv.org/pdf/2603.06639](https://arxiv.org/pdf/2603.06639)**

> **作者:** Heng Zhang
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Robust perception in brains is often attributed to high-dimensional population activity together with local plasticity mechanisms that reinforce recurring structure. In contrast, most modern image recognition systems are trained by error backpropagation and end-to-end gradient optimization, which are not naturally aligned with local computation and local plasticity. We introduce RECAP (Reservoir Computing with Hebbian Co-Activation Prototypes), a bio-inspired learning strategy for robust image classification that couples untrained reservoir dynamics with a self-organizing Hebbian prototype readout. RECAP discretizes time-averaged reservoir responses into activation levels, constructs a co-activation mask over reservoir unit pairs, and incrementally updates class-wise prototype matrices via a Hebbian-like potentiation-decay rule. Inference is performed by overlap-based prototype matching. The method avoids error backpropagation and is naturally compatible with online prototype updates. We illustrate the resulting robustness behavior on MNIST-C, where RECAP remains robust under diverse corruptions without exposure to corrupted training samples.
>
---
#### [new 298] ADAS-TO: A Large-Scale Multimodal Naturalistic Dataset and Empirical Characterization of Human Takeovers during ADAS Engagement
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于自动驾驶安全研究任务，旨在解决ADAS接管过程中的安全问题，通过构建大规模数据集并分析接管行为，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2603.06986](https://arxiv.org/pdf/2603.06986)**

> **作者:** Yuhang Wang; Yiyao Xu; Jingran Sun; Hao Zhou
>
> **摘要:** Takeovers remain a key safety vulnerability in production ADAS, yet existing public resources rarely provide takeover-centered, real-world data. We present ADAS-TO, the first large-scale naturalistic dataset dedicated to ADAS-to-manual transitions, containing 15,659 takeover-centered 20s clips from 327 drivers across 22 vehicle brands. Each clip synchronizes front-view video with CAN logs. Takeovers are defined as ADAS ON $\rightarrow$ OFF transitions, with the primary trigger labeled as brake, steer, gas, mixed, or system disengagement. We further separate planned driver-initiated terminations (Ego) from forced takeovers (Non-ego) using a rule-based partition. While most events occur within conservative kinematic margins, we identify a long tail of 285 safety-critical cases. For these events, we combine kinematic screening with vision--language (VLM) annotation to attribute hazards and relate them to intervention dynamics. The resulting cross-modal analysis shows distinct kinematic signatures across traffic dynamics, infrastructure degradation, and adverse environments, and finds that in 59.3% of critical cases, actionable visual cues emerge at least 3s before takeover, supporting the potential for semantics-aware early warning beyond late-stage kinematic triggers. The dataset is publicly released at this http URL.
>
---
#### [new 299] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手势生成任务，旨在解决3D物体与文本指令间模态差异大、抓取不物理有效的问题。提出AffordGrasp框架，结合语义与空间约束，生成精确且合理的抓取姿态。**

- **链接: [https://arxiv.org/pdf/2603.08021](https://arxiv.org/pdf/2603.08021)**

> **作者:** Xiaofei Wu; Yi Zhang; Yumeng Liu; Yuexin Ma; Yujiao Shi; Xuming He
>
> **摘要:** Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
>
---
#### [new 300] IGLU: The Integrated Gaussian Linear Unit Activation Function
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出一种新的激活函数IGLU，用于深度神经网络，解决传统激活函数在梯度消失和性能上的问题。通过引入重尾分布提升模型表现。**

- **链接: [https://arxiv.org/pdf/2603.06861](https://arxiv.org/pdf/2603.06861)**

> **作者:** Mingi Kang; Zai Yang; Jeova Farias Sales Rocha Neto
>
> **摘要:** Activation functions are fundamental to deep neural networks, governing gradient flow, optimization stability, and representational capacity. Within historic deep architectures, while ReLU has been the dominant choice for the activation function, modern transformer-based models increasingly are adopting smoother alternatives such as GELU and other self-gated alternatives. Despite their empirical success, the mathematical relationships among these functions and the principles underlying their effectiveness remains only partially understood. We introduce IGLU, a parametric activation function derived as a scale mixture of GELU gates under a half-normal mixing distribution. This derivation yields a closed-form expression whose gating component is exactly the Cauchy CDF, providing a principled one-parameter family that continuously interpolates between identity-like and ReLU-like behavior via a single sharpness parameter $\sigma$. Unlike GELU's Gaussian gate, IGLU's heavy-tailed Cauchy gate decays polynomially in the negative tail, guaranteeing non-zero gradients for all finite inputs and offering greater robustness to vanishing gradients. We further introduce IGLU-Approx, a computationally efficient rational approximation of IGLU expressed entirely in terms of ReLU operations that eliminates transcendental function evaluation. Through evaluations on CIFAR-10, CIFAR-100, and WikiText-103 across ResNet-20, ViT-Tiny, and GPT-2 Small, IGLU achieves competitive or superior performance on both vision and language datasets against ReLU and GELU baselines, with IGLU-Approx recovering this performance at substantially reduced computational cost. In particular, we show that employing a heavy-tailed gate leads to considerable performance gains in heavily imbalanced classification datasets.
>
---
#### [new 301] SoundWeaver: Semantic Warm-Starting for Text-to-Audio Diffusion Serving
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决扩散模型推理慢的问题。通过语义缓存预热，提升生成速度并保持质量。**

- **链接: [https://arxiv.org/pdf/2603.07865](https://arxiv.org/pdf/2603.07865)**

> **作者:** Ayush Barik; Sofia Stoica; Nikhil Sarda; Arnav Kethana; Abhinav Khanduja; Muchen Xu; Fan Lai
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Text-to-audio diffusion models produce high-fidelity audio but require tens of function evaluations (NFEs), incurring multi-second latency and limited throughput. We present SoundWeaver, the first training-free, model-agnostic serving system that accelerates text-to-audio diffusion by warm-starting from semantically similar cached audio. SoundWeaver introduces three components: a Reference Selector that retrieves and temporally aligns cached candidates via semantic and duration-aware gating; a Skip Gater that dynamically determines the percentage of NFEs to skip; and a lightweight Cache Manager that maintains cache utility through quality-aware eviction and refinement. On real-world audio traces, SoundWeaver achieves 1.8--3.0$ \times $ latency reduction with a cache of only ${\sim}$1K entries while preserving or improving perceptual quality.
>
---
#### [new 302] RoboPCA: Pose-centered Affordance Learning from Human Demonstrations for Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboPCA，解决机器人操作中空间可操作性预测问题，通过联合预测接触区域和姿态，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.07691](https://arxiv.org/pdf/2603.07691)**

> **作者:** Zhanqi Xiao; Ruiping Wang; Xilin Chen
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Understanding spatial affordances -- comprising the contact regions of object interaction and the corresponding contact poses -- is essential for robots to effectively manipulate objects and accomplish diverse tasks. However, existing spatial affordance prediction methods mainly focus on locating the contact regions while delegating the pose to independent pose estimation approaches, which can lead to task failures due to inconsistencies between predicted contact regions and candidate poses. In this work, we propose RoboPCA, a pose-centered affordance prediction framework that jointly predicts task-appropriate contact regions and poses conditioned on instructions. To enable scalable data collection for pose-centered affordance learning, we devise Human2Afford, a data curation pipeline that automatically recovers scene-level 3D information and infers pose-centered affordance annotations from human demonstrations. With Human2Afford, scene depth and the interaction object's mask are extracted to provide 3D context and object localization, while pose-centered affordance annotations are obtained by tracking object points within the contact region and analyzing hand-object interaction patterns to establish a mapping from the 3D hand mesh to the robot end-effector orientation. By integrating geometry-appearance cues through an RGB-D encoder and incorporating mask-enhanced features to emphasize task-relevant object regions into the diffusion-based framework, RoboPCA outperforms baseline methods on image datasets, simulation, and real robots, and exhibits strong generalization across tasks and categories.
>
---
#### [new 303] UniUncer: Unified Dynamic Static Uncertainty for End to End Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决E2E驾驶中的不确定性问题。提出UniUncer框架，统一建模静态与动态场景的不确定性，提升规划可靠性。**

- **链接: [https://arxiv.org/pdf/2603.07686](https://arxiv.org/pdf/2603.07686)**

> **作者:** Yu Gao; Jijun Wang; Zongzheng Zhang; Anqing Jiang; Yiru Wang; Yuwen Heng; Shuo Wang; Hao Sun; Zhangfeng Hu; Hao Zhao
>
> **备注:** ICRA 2026
>
> **摘要:** End-to-end (E2E) driving has become a cornerstone of both industry deployment and academic research, offering a single learnable pipeline that maps multi-sensor inputs to actions while avoiding hand-engineered modules. However, the reliability of such pipelines strongly depends on how well they handle uncertainty: sensors are noisy, semantics can be ambiguous, and interaction with other road users is inherently stochastic. Uncertainty also appears in multiple forms: classification vs. localization, and, crucially, in both static map elements and dynamic agents. Existing E2E approaches model only static-map uncertainty, leaving planning vulnerable to overconfident and unreliable inputs. We present UniUncer, the first lightweight, unified uncertainty framework that jointly estimates and uses uncertainty for both static and dynamic scene elements inside an E2E planner. Concretely: (1) we convert deterministic heads to probabilistic Laplace regressors that output per-vertex location and scale for vectorized static and dynamic entities; (2) we introduce an uncertainty-fusion module that encodes these parameters and injects them into object/map queries to form uncertainty-aware queries; and (3) we design an uncertainty-aware gate that adaptively modulates reliance on historical inputs (ego status or temporal perception queries) based on current uncertainty levels. The design adds minimal overhead and drops throughput by only $\sim$0.5 FPS while remaining plug-and-play for common E2E backbones. On nuScenes (open-loop), UniUncer reduces average L2 trajectory error by 7\%. On NavsimV2 (pseudo closed-loop), it improves overall EPDMS by 10.8\% and notable stage two gains in challenging, interaction-heavy scenes. Ablations confirm that dynamic-agent uncertainty and the uncertainty-aware gate are both necessary.
>
---
#### [new 304] MultiGen: Level-Design for Editable Multiplayer Worlds in Diffusion Game Engines
- **分类: cs.AI; cs.CV; cs.GR**

- **简介: 该论文属于游戏生成任务，旨在解决用户控制与多人共享世界的问题。通过引入外部记忆模块，实现可编辑的环境和实时多人互动。**

- **链接: [https://arxiv.org/pdf/2603.06679](https://arxiv.org/pdf/2603.06679)**

> **作者:** Ryan Po; David Junhao Zhang; Amir Hertz; Gordon Wetzstein; Neal Wadhwa; Nataniel Ruiz
>
> **备注:** Project page here: this https URL
>
> **摘要:** Video world models have shown immense promise for interactive simulation and entertainment, but current systems still struggle with two important aspects of interactivity: user control over the environment for reproducible, editable experiences, and shared inference where players hold influence over a common world. To address these limitations, we introduce an explicit external memory into the system, a persistent state operating independent of the model's context window, that is continually updated by user actions and queried throughout the generation roll-out. Unlike conventional diffusion game engines that operate as next-frame predictors, our approach decomposes generation into Memory, Observation, and Dynamics modules. This design gives users direct, editable control over environment structure via an editable memory representation, and it naturally extends to real-time multiplayer rollouts with coherent viewpoints and consistent cross-player interactions.
>
---
#### [new 305] StructBiHOI: Structured Articulation Modeling for Long--Horizon Bimanual Hand--Object Interaction Generation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多手物体交互生成任务，解决长时序、精细关节协调和跨手协作问题。提出StructBiHOI框架，通过结构化关节规划与帧级优化实现稳定、真实的双臂操作生成。**

- **链接: [https://arxiv.org/pdf/2603.08390](https://arxiv.org/pdf/2603.08390)**

> **作者:** Zhi Wang; Liu Liu; Ruonan Liu; Dan Guo; Meng Wang
>
> **摘要:** Recent progress in 3D hand--object interaction (HOI) generation has primarily focused on single--hand grasp synthesis, while bimanual manipulation remains significantly more challenging. Long--horizon planning instability, fine--grained joint articulation, and complex cross--hand coordination make coherent bimanual generation difficult, especially under multimodal conditions. Existing approaches often struggle to simultaneously ensure temporal consistency, physical plausibility, and semantic alignment over extended sequences. We propose StructBiHOI, a Structured articulation modeling framework for long-horizon Bimanual HOI generation. Our key insight is to structurally disentangle temporal joint planning from frame--level manipulation refinement. Specifically, a jointVAE models long-term joint evolution conditioned on object geometry and task semantics, while a maniVAE refines fine-grained hand poses at the single--frame level. To enable stable and efficient long--sequence generation, we incorporate a state--space--inspired diffusion denoiser based on Mamba, which models long--range dependencies with linear complexity. This hierarchical design facilitates coherent dual-hand coordination and articulated object interaction. Extensive experiments on bimanual manipulation and single-hand grasping benchmarks demonstrate that our method achieves superior long--horizon stability, motion realism, and computational efficiency compared to strong baselines.
>
---
#### [new 306] Visualizing Coalition Formation: From Hedonic Games to Image Segmentation
- **分类: cs.AI; cs.CV**

- **简介: 论文将联盟形成问题映射到图像分割任务，研究参数对均衡结构的影响，旨在连接多智能体系统与图像分割。**

- **链接: [https://arxiv.org/pdf/2603.07890](https://arxiv.org/pdf/2603.07890)**

> **作者:** Pedro Henrique de Paula França; Lucas Lopes Felipe; Daniel Sadoc Menasché
>
> **备注:** The First Workshop on AI for Mechanism Design and Strategic Decision Making -- Workshop AIMS at ICLR 2026
>
> **摘要:** We propose image segmentation as a visual diagnostic testbed for coalition formation in hedonic games. Modeling pixels as agents on a graph, we study how a granularization parameter shapes equilibrium fragmentation and boundary structure. On the Weizmann single-object benchmark, we relate multi-coalition equilibria to binary protocols by measuring whether the converged coalitions overlap with a foreground ground-truth. We observe transitions from cohesive to fragmented yet recoverable equilibria, and finally to intrinsic failure under excessive fragmentation. Our core contribution links multi-agent systems with image segmentation by quantifying the impact of mechanism design parameters on equilibrium structures.
>
---
#### [new 307] Uncertainty-Aware Solar Flare Regression
- **分类: astro-ph.SR; cs.CV; cs.LG**

- **简介: 该论文属于空间天气预测任务，旨在提升太阳耀斑预测的可靠性。通过引入置信区间，解决现有预测缺乏准确可信度评估的问题，采用共形预测等方法优化预测效果。**

- **链接: [https://arxiv.org/pdf/2603.06712](https://arxiv.org/pdf/2603.06712)**

> **作者:** Jinsu Hong; Chetraj Pandey; Berkay Aydin
>
> **摘要:** Current solar flare predictions often lack precise quantification of their reliability, resulting in frequent false alarms, particularly when dealing with datasets skewed towards extreme events. To improve the trustworthiness of space weather forecasting, it is crucial to establish confidence intervals for model predictions. Conformal prediction, a machine learning framework, presents a promising avenue for this purpose by constructing prediction intervals that ensure valid coverage in finite samples without making assumptions about the underlying data distribution. In this study, we explore the application of conformal prediction to regression tasks in space weather forecasting. Specifically, we implement full-disk solar flare prediction using images created from magnetic field maps and adapt four pre-trained deep learning models to incorporate three distinct methods for constructing confidence intervals: conformal prediction, quantile regression, and conformalized quantile regression. Our experiments demonstrate that conformalized quantile regression achieves higher coverage rates and more favorable average interval lengths compared to alternative methods, underscoring its effectiveness in enhancing the reliability of solar weather forecasting models.
>
---
#### [new 308] DualFlexKAN: Dual-stage Kolmogorov-Arnold Networks with Independent Function Control
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出DualFlexKAN，解决传统网络表达能力与计算成本的平衡问题。通过双阶段机制和多样化基函数，提升模型精度与效率，适用于科学计算任务。**

- **链接: [https://arxiv.org/pdf/2603.08583](https://arxiv.org/pdf/2603.08583)**

> **作者:** Andrés Ortiz; Nicolás J. Gallego-Molina; Carmen Jiménez-Mesa; Juan M. Górriz; Javier Ramírez
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Multi-Layer Perceptrons (MLPs) rely on pre-defined, fixed activation functions, imposing a static inductive bias that forces the network to approximate complex topologies solely through increased depth and width. Kolmogorov-Arnold Networks (KANs) address this limitation through edge-centric learnable functions, yet their formulation suffers from quadratic parameter scaling and architectural rigidity that hinders the effective integration of standard regularization techniques. This paper introduces the DualFlexKAN (DFKAN), a flexible architecture featuring a dual-stage mechanism that independently controls pre-linear input transformations and post-linear output activations. This decoupling enables hybrid networks that optimize the trade-off between expressiveness and computational cost. Unlike standard formulations, DFKAN supports diverse basis function families, including orthogonal polynomials, B-splines, and radial basis functions, integrated with configurable regularization strategies that stabilize training dynamics. Comprehensive evaluations across regression benchmarks, physics-informed tasks, and function approximation demonstrate that DFKAN outperforms both MLPs and conventional KANs in accuracy, convergence speed, and gradient fidelity. The proposed hybrid configurations achieve superior performance with one to two orders of magnitude fewer parameters than standard KANs, effectively mitigating the parameter explosion problem while preserving KAN-style expressiveness. DFKAN provides a principled, scalable framework for incorporating adaptive non-linearities, proving particularly advantageous for data-efficient learning and interpretable function discovery in scientific applications.
>
---
#### [new 309] Learning From Design Procedure To Generate CAD Programs for Data Augmentation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于CAD程序生成任务，旨在解决LLM生成的CAD模型几何复杂度不足的问题。通过数据增强方法，提升生成模型的多样性与工业设计相似度。**

- **链接: [https://arxiv.org/pdf/2603.06894](https://arxiv.org/pdf/2603.06894)**

> **作者:** Yan-Ying Chen; Dule Shu; Matthew Hong; Andrew Taber; Jonathan Li; Matthew Klenk
>
> **备注:** Accepted by NeurIPS 2025 Workshop: Deep Learning for Code in the Agentic Era
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in a wide range of code generation tasks. However, generating code for certain domains remains challenging. One such domain is Computer-Aided Design (CAD) program, where the goal is to produce scripted parametric models that define object geometry for precise design and manufacturing applications. A key challenge in LLM-based CAD program generation is the limited geometric complexity of generated shapes compared to those found in real-world industrial designs. This shortfall is in part due to the lack of diversity in the available CAD program training data. To address this, we propose a novel data augmentation paradigm that prompts an LLM to generate CAD programs conditioned on a reference surface program and a modeling procedure - an idea inspired by practices in industrial design. By varying the reference surface using a collection of organic shapes, our method enriches the geometric distribution of generated CAD models. In particular, it introduces edges and faces defined by spline-based curvature, which are typically missing or underrepresented in existing open-source CAD program datasets. Experiments show that our method produces CAD samples with significantly greater geometric diversity and a higher resemblance to industry-grade CAD designs in terms of the proportion of organic shape primitives. This enhancement makes our CAD data augmentation approach a useful tool for training LLMs and other deep learning models in CAD generation.
>
---
#### [new 310] A Novel Approach for Testing Water Safety Using Deep Learning Inference of Microscopic Images of Unincubated Water Samples
- **分类: cs.OH; cs.CV; cs.CY; cs.LG**

- **简介: 该论文属于水质检测任务，旨在解决传统水安全测试耗时长、成本高的问题。通过深度学习分析显微图像，实现快速准确的水样检测。**

- **链接: [https://arxiv.org/pdf/2603.06611](https://arxiv.org/pdf/2603.06611)**

> **作者:** Sanjay Srinivasan
>
> **摘要:** Fecal-contaminated water causes diseases and even death. Current microbial water safety tests require pathogen incubation, taking 24-72 hours and costing \$20-\$50 per test. This paper presents a solution (DeepScope) exceeding UNICEF's ideal Target Product Profile requirements for presence/absence testing, with an estimated per-test cost of \$0.44. By eliminating the need for pathogen incubation, DeepScope reduces testing time by over 98\%. In DeepScope, a dataset of microscope images of bacteria and water samples was assembled. An innovative augmentation technique, generating up to 21 trillion images from a single microscope image, was developed. Four convolutional neural network models were developed using transfer learning and regularization techniques, then evaluated on a field-test dataset comprising 100,000 microscope images of unseen, real-world water samples collected from fourteen different water sources across Sammamish, WA. Precision-recall analysis showed the DeepScope model achieves 93\% accuracy, with precision of 90\% and recall exceeding 94\%. The DeepScope model was deployed on a web server, and mobile applications for Android and iOS were developed, enabling Internet-based or smartphone-based water safety testing, with results obtained in seconds.
>
---
#### [new 311] OptiRoulette Optimizer: A New Stochastic Meta-Optimizer for up to 5.3x Faster Convergence
- **分类: cs.LG; cs.AI; cs.CV; cs.NE**

- **简介: 该论文提出OptiRoulette，一种用于图像分类的新型随机元优化器，解决传统优化器收敛慢且不稳定的问题。通过动态选择更新规则，提升训练效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.06613](https://arxiv.org/pdf/2603.06613)**

> **作者:** Stamatis Mastromichalakis
>
> **备注:** 23 pages, 10 figures, 7 tables
>
> **摘要:** This paper presents OptiRoulette, a stochastic meta-optimizer that selects update rules during training instead of fixing a single optimizer. The method combines warmup optimizer locking, random sampling from an active optimizer pool, compatibility-aware learning-rate scaling during optimizer transitions, and failure-aware pool replacement. OptiRoulette is implemented as a drop-in, this http URL-compatible component and packaged for pip installation. We report completed 10-seed results on five image-classification suites: CIFAR-100, CIFAR-100-C, SVHN, Tiny ImageNet, and Caltech-256. Against a single-optimizer AdamW baseline, OptiRoulette improves mean test accuracy from 0.6734 to 0.7656 on CIFAR-100 (+9.22 percentage points), 0.2904 to 0.3355 on CIFAR-100-C (+4.52), 0.9667 to 0.9756 on SVHN (+0.89), 0.5669 to 0.6642 on Tiny ImageNet (+9.73), and 0.5946 to 0.6920 on Caltech-256 (+9.74). Its main advantage is convergence reliability at higher targets: it reaches CIFAR-100/CIFAR-100-C 0.75, SVHN 0.96, Tiny ImageNet 0.65, and Caltech-256 0.62 validation accuracy in 10/10 runs, while the AdamW baseline reaches none of these targets within budget. On shared targets, OptiRoulette also reduces time-to-target (e.g., Caltech-256 at 0.59: 25.7 vs 77.0 epochs). Paired-seed deltas are positive on all datasets; CIFAR-100-C test ROC-AUC is the only metric not statistically significant in the current 10-seed study.
>
---
#### [new 312] ACCURATE: Arbitrary-shaped Continuum Reconstruction Under Robust Adaptive Two-view Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决柔性连续体（如导管）的高精度重建问题。通过结合神经网络与几何约束算法，提升重建准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07533](https://arxiv.org/pdf/2603.07533)**

> **作者:** Yaozhi Zhang; Shun Yu; Yugang Zhang; Yang Liu
>
> **摘要:** Accurate reconstruction of arbitrary-shaped long slender continuum bodies, such as guidewires, catheters and other soft continuum manipulators, is essential for accurate mechanical simulation. However, existing image-based reconstruction approaches often suffer from limited accuracy because they often underutilize camera geometry, or lack generality as they rely on rigid geometric assumptions that may fail for continuum robots with complex and highly deformable shapes. To address these limitations, we propose ACCURATE, a 3D reconstruction framework integrating an image segmentation neural network with a geometry-constrained topology traversal and dynamic programming algorithm that enforces global biplanar geometric consistency, minimizes the cumulative point-to-epipolar-line distance, and remains robust to occlusions and epipolar ambiguities cases caused by noise and discretization. Our method achieves high reconstruction accuracy on both simulated and real phantom datasets acquired using a clinical X-ray C-arm system, with mean absolute errors below 1.0 mm.
>
---
#### [new 313] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出AtomicVLA，解决机器人长序列任务和持续学习问题。通过原子技能抽象与动态专家组合，提升任务规划与执行效果。**

- **链接: [https://arxiv.org/pdf/2603.07648](https://arxiv.org/pdf/2603.07648)**

> **作者:** Likui Zhang; Tao Tang; Zhihao Zhan; Xiuwei Chen; Zisheng Chen; Jianhua Han; Jiangtong Zhu; Pei Xu; Hang Xu; Hefeng Wu; Liang Lin; Xiaodan Liang
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Recent advances in Visual-Language-Action (VLA) models have shown promising potential for robotic manipulation tasks. However, real-world robotic tasks often involve long-horizon, multi-step problem-solving and require generalization for continual skill acquisition, extending beyond single actions or skills. These challenges present significant barriers for existing VLA models, which use monolithic action decoders trained on aggregated data, resulting in poor scalability. To address these challenges, we propose AtomicVLA, a unified planning-and-execution framework that jointly generates task-level plans, atomic skill abstractions, and fine-grained actions. AtomicVLA constructs a scalable atomic skill library through a Skill-Guided Mixture-of-Experts (SG-MoE), where each expert specializes in mastering generic yet precise atomic skills. Furthermore, we introduce a flexible routing encoder that automatically assigns dedicated atomic experts to new skills, enabling continual learning. We validate our approach through extensive experiments. In simulation, AtomicVLA outperforms $\pi_{0}$ by 2.4\% on LIBERO, 10\% on LIBERO-LONG, and outperforms $\pi_{0}$ and $\pi_{0.5}$ by 0.22 and 0.25 in average task length on CALVIN. Additionally, our AtomicVLA consistently surpasses baselines by 18.3\% and 21\% in real-world long-horizon tasks and continual learning. These results highlight the effectiveness of atomic skill abstraction and dynamic expert composition for long-horizon and lifelong robotic tasks. The project page is \href{this https URL}{here}.
>
---
#### [new 314] Correlation Analysis of Generative Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型研究任务，旨在解决现有扩散模型与流匹配中噪声数据与目标相关性弱的问题。通过提出统一表示和理论分析，提升模型预测效果。**

- **链接: [https://arxiv.org/pdf/2603.06614](https://arxiv.org/pdf/2603.06614)**

> **作者:** Zhengguo Li; Chaobing Zheng; Wei Wang
>
> **摘要:** Based on literature review about existing diffusion models and flow matching with a neural network to predict a predefined target from noisy data, a unified representation is first proposed for these models using two simple linear equations in this paper. Theoretical analysis of the proposed model is then presented. Our theoretical analysis shows that the correlation between the noisy data and the predicted target is sometimes weak in the existing diffusion models and flow matching. This might affect the prediction (or learning) process which plays a crucial role in all models.
>
---
#### [new 315] Extend Your Horizon: A Device-Agnostic Surgical Tool Tracking Framework with Multi-View Optimization for Augmented Reality
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于手术导航任务，解决动态环境中手术工具跟踪问题。通过多模态传感器融合与动态场景图，提升AR可视化在遮挡下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07981](https://arxiv.org/pdf/2603.07981)**

> **作者:** Jiaming Zhang; Mingxu Liu; Hongchao Shu; Ruixing Liang; Yihao Liu; Ojas Taskar; Amir Kheradmand; Mehran Armand; Alejandro Martin-Gomez
>
> **备注:** accepted by IEEE VR 2026
>
> **摘要:** Surgical navigation provides real-time guidance by estimating the pose of patient anatomy and surgical instruments to visualize relevant intraoperative information. In conventional systems, instruments are typically tracked using fiducial markers and stationary optical tracking systems (OTS). Augmented reality (AR) has further enabled intuitive visualization and motivated tracking using sensors embedded in head-mounted displays (HMDs). However, most existing approaches rely on a clear line of sight, which is difficult to maintain in dynamic operating room environments due to frequent occlusions caused by equipment, surgical tools, and personnel. This work introduces a framework for tracking surgical instruments under occlusion by fusing multiple sensing modalities within a dynamic scene graph representation. The proposed approach integrates tracking systems with different accuracy levels and motion characteristics while estimating tracking reliability in real time. Experimental results demonstrate improved robustness and enhanced consistency of AR visualization in the presence of occlusions.
>
---
#### [new 316] Rectified flow-based prediction of post-treatment brain MRI from pre-radiotherapy priors for patients with glioma
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像生成任务，旨在通过AI生成胶质瘤患者放疗后的脑部MRI，以支持治疗优化和个性化预测。**

- **链接: [https://arxiv.org/pdf/2603.08385](https://arxiv.org/pdf/2603.08385)**

> **作者:** Selena Huisman; Nordin Belkacemi; Vera Keil; Joost Verhoeff; Szabolcs David
>
> **备注:** 10 pages, 6 figures, 1 supplementary table
>
> **摘要:** Purpose/Objective: Brain tumors result in 20 years of lost life on average. Standard therapies induce complex structural changes in the brain that are monitored through MRI. Recent developments in artificial intelligence (AI) enable conditional multimodal image generation from clinical data. In this study, we investigate AI-driven generation of follow-up MRI in patients with in- tracranial tumors through conditional image generation. This approach enables realistic modeling of post-radiotherapy changes, allowing for treatment optimization. Material/Methods: The public SAILOR dataset of 25 patients was used to create a 2D rectified flow model conditioned on axial slices of pre-treatment MRI and RT dose maps. Cross-attention conditioning was used to incorporate temporal and chemotherapy data. The resulting images were validated with structural similarity index measure (SSIM), peak signal-to-noise ratio (PSNR), Dice scores and Jacobian determinants. Results: The resulting model generates realistic follow-up MRI for any time point, while integrating treatment information. Comparing real versus predicted images, SSIM is 0.88, and PSNR is 22.82. Tissue segmentations from real versus predicted MRI result in a mean Dice-Sørensen coefficient (DSC) of 0.91. The rectified flow (RF) model enables up to 250x faster inference than Denoising Diffusion Probabilistic Models (DDPM). Conclusion: The proposed model generates realistic follow-up MRI in real-time, preserving both semantic and visual fidelity as confirmed by image quality metrics and tissue segmentations. Conditional generation allows counterfactual simulations by varying treatment parameters, producing predicted morphological changes. This capability has potential to support adaptive treatment dose planning and personalized outcome prediction for patients with intracranial tumors.
>
---
#### [new 317] Task learning increases information redundancy of neural responses in macaque visual cortex
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文研究视觉任务学习对猕猴V4区神经响应冗余的影响。旨在解决学习如何优化感官信息的问题，通过实验发现任务学习增加了冗余并提升单个神经元的信息量。**

- **链接: [https://arxiv.org/pdf/2603.07369](https://arxiv.org/pdf/2603.07369)**

> **作者:** Shizhao Liu; Anton Pletenev; Ralf M. Haefner; Adam C. Snyder
>
> **备注:** published in Science, accepted manuscript prior to editing, main text: 33 pages, 5 figures, 39 supplementary pages, 22 supplementary figures, 7 supplementary tables
>
> **摘要:** How does the brain optimize sensory information for decision-making in new tasks? One hypothesis suggests learning reduces redundancy in neural representations to improve efficiency, while another, based on Bayesian inference, predicts learning increases redundancy by distributing information across neurons. We tested these hypotheses by tracking population responses in macaque cortical area V4 as monkeys learned visual discrimination tasks. We found strong support for the Bayesian predictions: task learning increased redundancy in neural responses over weeks of training and within single trials. This redundancy did not reduce information but instead increased the information carried by individual neurons. These insights suggest sensory processing in the brain reflects a generative rather than discriminative inference process.
>
---
#### [new 318] Data Agent: Learning to Select Data via End-to-End Dynamic Optimization
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出Data Agent，用于动态数据选择，解决传统方法依赖人工指标、难以适应变化的问题。通过强化学习，实现高效训练并提升性能。**

- **链接: [https://arxiv.org/pdf/2603.07433](https://arxiv.org/pdf/2603.07433)**

> **作者:** Suorong Yang; Fangjian Su; Hai Gan; Ziqi Ye; Jie Li; Baile Xu; Furao Shen; Soujanya Poria
>
> **摘要:** Dynamic Data selection aims to accelerate training by prioritizing informative samples during online training. However, existing methods typically rely on task-specific handcrafted metrics or static/snapshot-based criteria to estimate sample importance, limiting scalability across learning paradigms and making it difficult to capture the evolving utility of data throughout training. To address this challenge, we propose Data Agent, an end-to-end dynamic data selection framework that formulates data selection as a training-aware sequential decision-making problem. The agent learns a sample-wise selection policy that co-evolves with model optimization, guided by a composite reward that integrates loss-based difficulty and confidence-based uncertainty signals. The reward signals capture complementary objectives of optimization impact and information gain, together with a tuning-free adaptive weighting mechanism that balances these signals over training. Extensive experiments across a wide range of datasets and architectures demonstrate that Data Agent consistently accelerates training while preserving or improving performance, e.g., reducing costs by over 50\% on ImageNet-1k and MMLU with lossless performance. Moreover, its dataset-agnostic formulation and modular reward make it plug-and-play across tasks and scenarios, e.g., robustness to noisy datasets, highlighting its potential in real-world scenarios.
>
---
#### [new 319] A Unified View of Drifting and Score-Based Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文探讨了漂移模型与基于分数的模型之间的关系，旨在揭示其在分布匹配中的联系。通过理论分析，证明了漂移模型可视为一种分数匹配方法，解决了模型与数据分布对齐的问题。**

- **链接: [https://arxiv.org/pdf/2603.07514](https://arxiv.org/pdf/2603.07514)**

> **作者:** Chieh-Hsin Lai; Bac Nguyen; Naoki Murata; Yuhta Takida; Toshimitsu Uesaka; Yuki Mitsufuji; Stefano Ermon; Molei Tao
>
> **摘要:** Drifting models train one-step generators by optimizing a mean-shift discrepancy induced by a kernel between the data and model distributions, with Laplace kernels used by default in practice. At each point, this discrepancy compares the kernel-weighted displacement toward nearby data samples with the corresponding displacement toward nearby model samples, yielding a transport direction for generated samples. In this paper, we make its relationship to the score-matching principle behind diffusion models precise by showing that drifting admits a score-based formulation on kernel-smoothed distributions. For Gaussian kernels, the population mean-shift field coincides with the score difference between the Gaussian-smoothed data and model distributions. This identity follows from Tweedie's formula, which links the score of a Gaussian-smoothed density to the corresponding conditional mean, and implies that Gaussian-kernel drifting is exactly a score-matching-style objective on smoothed distributions. It also clarifies the connection to Distribution Matching Distillation (DMD): both methods use score-mismatch transport directions, but drifting realizes the score signal nonparametrically from kernel neighborhoods, whereas DMD uses a pretrained diffusion teacher. Beyond Gaussians, we derive an exact decomposition for general radial kernels, and for the Laplace kernel we prove rigorous error bounds showing that drifting remains an accurate proxy for score matching in low-temperature and high-dimensional regimes.
>
---
#### [new 320] LightMedSeg: Lightweight 3D Medical Image Segmentation with Learned Spatial Anchors
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于3D医学图像分割任务，旨在解决模型参数多、计算量大及泛化能力弱的问题。提出LightMedSeg架构，结合解剖先验与自适应上下文建模，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.07228](https://arxiv.org/pdf/2603.07228)**

> **作者:** Kavyansh Tyagi; Vishwas Rathi; Puneet Goyal
>
> **备注:** 8 pages, X figures. Submitted to CVPRW ECV 2026
>
> **摘要:** Accurate and efficient 3D medical image segmentation is essential for clinical AI, where models must remain reliable under stringent memory, latency, and data availability constraints. Transformer-based methods achieve strong accuracy but suffer from excessive parameters, high FLOPs, and limited generalization. We propose LightMedSeg, a modular UNet-style segmentation architecture that integrates anatomical priors with adaptive context modeling. Anchor-conditioned FiLM modulation enables anatomy-aware feature calibration, while a local structural prior module and texture-aware routing dynamically allocate representational capacity to boundary-rich regions. Computational redundancy is minimized through ghost and depthwise convolutions, and multi-scale features are adaptively fused via a learned skip router with anchor-relative spatial position bias. Despite requiring only 0.48M parameters and 14.64~GFLOPs, LightMedSeg achieves segmentation accuracy within a few Dice points of heavy transformer baselines. Therefore, LightMedSeg is a deployable and data-efficient solution for 3D medical image segmentation. Code will be released publicly upon acceptance.
>
---
## 更新

#### [replaced 001] ViTaPEs: Visuotactile Position Encodings for Cross-Modal Alignment in Multimodal Transformers
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ViTaPEs，解决多模态对齐问题，通过双阶段位置编码提升视觉与触觉信息融合，增强模型泛化与迁移能力。**

- **链接: [https://arxiv.org/pdf/2505.20032](https://arxiv.org/pdf/2505.20032)**

> **作者:** Fotios Lygerakis; Ozan Özdenizci; Elmar Rückert
>
> **摘要:** Tactile sensing provides local essential information that is complementary to visual perception, such as texture, compliance, and force. Despite recent advances in visuotactile representation learning, challenges remain in fusing these modalities and generalizing across tasks and environments without heavy reliance on pre-trained vision-language models. Moreover, existing methods do not study positional encodings, thereby overlooking the multi-stage spatial reasoning needed to capture fine-grained visuotactile correlations. We introduce ViTaPEs, a transformer-based architecture for learning task-agnostic visuotactile representations from paired vision and tactile inputs. Our key idea is a two-stage positional injection: local (modality-specific) positional encodings are added within each stream, and a global positional encoding is added on the joint token sequence immediately before attention, providing a shared positional vocabulary at the stage where cross-modal interaction occurs. We make the positional injection points explicit and conduct controlled ablations that isolate their effect before a token-wise nonlinearity versus immediately before self-attention. Experiments on multiple large-scale real-world datasets show that ViTaPEs not only surpasses state-of-the-art baselines across various recognition tasks but also demonstrates zero-shot generalization to unseen, out-of-domain scenarios. We further demonstrate the transfer-learning strength of ViTaPEs in a robotic grasping task, where it outperforms state-of-the-art baselines in predicting grasp success. Project page: this https URL
>
---
#### [replaced 002] TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.04989](https://arxiv.org/pdf/2603.04989)**

> **作者:** Jiaxiong Liu; Zhen Tan; Jinpu Zhang; Yi Zhou; Hui Shen; Xieyuanli Chen; Dewen Hu
>
> **摘要:** Tracking any point (TAP) is a fundamental yet challenging task in computer vision, requiring high precision and long-term motion reasoning. Recent attempts to combine RGB frames and event streams have shown promise, yet they typically rely on synchronous or non-adaptive fusion, leading to temporal misalignment and severe degradation when one modality fails. We introduce TAPFormer, a transformer-based framework that performs asynchronous temporal-consistent fusion of frames and events for robust and high-frequency arbitrary point tracking. Our key innovation is a Transient Asynchronous Fusion (TAF) mechanism, which explicitly models the temporal evolution between discrete frames through continuous event updates, bridging the gap between low-rate frames and high-rate events. In addition, a Cross-modal Locally Weighted Fusion (CLWF) module adaptively adjusts spatial attention according to modality reliability, yielding stable and discriminative features even under blur or low light. To evaluate our approach under realistic conditions, we construct a novel real-world frame-event TAP dataset under diverse illumination and motion conditions. Our method outperforms existing point trackers, achieving a 28.2% improvement in average pixel error within threshold. Moreover, on standard point tracking benchmarks, our tracker consistently achieves the best performance. Project website: this http URL
>
---
#### [replaced 003] OVerSeeC: Open-Vocabulary Costmap Generation from Satellite Images and Natural Language
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OVerSeeC，解决从卫星图像和自然语言生成全局成本地图的任务。针对未知实体和动态任务需求，通过分解为解析、定位、合成模块，实现灵活、可扩展的路径规划。**

- **链接: [https://arxiv.org/pdf/2602.18606](https://arxiv.org/pdf/2602.18606)**

> **作者:** Rwik Rana; Jesse Quattrociocchi; Dongmyeong Lee; Christian Ellis; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **备注:** Website : this https URL
>
> **摘要:** Aerial imagery provides essential global context for autonomous navigation, enabling route planning at scales inaccessible to onboard sensing. We address the problem of generating global costmaps for long-range planning directly from satellite imagery when entities and mission-specific traversal rules are expressed in natural language at test time. This setting is challenging since mission requirements vary, terrain entities may be unknown at deployment, and user prompts often encode compositional traversal logic. Existing approaches relying on fixed ontologies and static cost mappings cannot accommodate such flexibility. While foundation models excel at language interpretation and open-vocabulary perception, no single model can simultaneously parse nuanced mission directives, locate arbitrary entities in large-scale imagery, and synthesize them into an executable cost function for planners. We therefore propose OVerSeeC, a zero-shot modular framework that decomposes the problem into Interpret-Locate-Synthesize: (i) an LLM extracts entities and ranked preferences, (ii) an open-vocabulary segmentation pipeline identifies these entities from high-resolution imagery, and (iii) the LLM uses the user's natural language preferences and masks to synthesize executable costmap code. Empirically, OVerSeeC handles novel entities, respects ranked and compositional preferences, and produces routes consistent with human-drawn trajectories across diverse regions, demonstrating robustness to distribution shifts. This shows that modular composition of foundation models enables open-vocabulary, preference-aligned costmap generation for scalable, mission-adaptive global planning.
>
---
#### [replaced 004] From Pixels to Predicates: Learning Symbolic World Models via Pretrained Vision-Language Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人决策任务，旨在解决复杂环境中的长期规划问题。通过预训练视觉-语言模型学习符号化世界模型，实现零样本泛化与新目标的达成。**

- **链接: [https://arxiv.org/pdf/2501.00296](https://arxiv.org/pdf/2501.00296)**

> **作者:** Ashay Athalye; Nishanth Kumar; Tom Silver; Yichao Liang; Jiuguang Wang; Tomás Lozano-Pérez; Leslie Pack Kaelbling
>
> **备注:** A version of this paper appears in the official proceedings of RA-L, Volume 11, Issue 4
>
> **摘要:** Our aim is to learn to solve long-horizon decision-making problems in complex robotics domains given low-level skills and a handful of short-horizon demonstrations containing sequences of images. To this end, we focus on learning abstract symbolic world models that facilitate zero-shot generalization to novel goals via planning. A critical component of such models is the set of symbolic predicates that define properties of and relationships between objects. In this work, we leverage pretrained vision-language models (VLMs) to propose a large set of visual predicates potentially relevant for decision-making, and to evaluate those predicates directly from camera images. At training time, we pass the proposed predicates and demonstrations into an optimization-based model-learning algorithm to obtain an abstract symbolic world model that is defined in terms of a compact subset of the proposed predicates. At test time, given a novel goal in a novel setting, we use the VLM to construct a symbolic description of the current world state, and then use a search-based planning algorithm to find a sequence of low-level skills that achieves the goal. We demonstrate empirically across experiments in both simulation and the real world that our method can generalize aggressively, applying its learned world model to solve problems with a wide variety of object types, arrangements, numbers of objects, and visual backgrounds, as well as novel goals and much longer horizons than those seen at training time.
>
---
#### [replaced 005] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.00790](https://arxiv.org/pdf/2507.00790)**

> **作者:** Huaqiu Li; Yong Wang; Tongwen Huang; Hailang Huang; Haoqian Wang; Xiangxiang Chu
>
> **摘要:** Unified image restoration is a significantly challenging task in low-level vision. Existing methods either make tailored designs for specific tasks, limiting their generalizability across various types of degradation, or rely on training with paired datasets, thereby suffering from closed-set constraints. To address these issues, we propose a novel, dataset-free, and unified approach through recurrent posterior sampling utilizing a pretrained latent diffusion model. Our method incorporates the multimodal understanding model to provide sematic priors for the generative model under a task-blind condition. Furthermore, it utilizes a lightweight module to align the degraded input with the generated preference of the diffusion model, and employs recurrent refinement for posterior sampling. Extensive experiments demonstrate that our method outperforms state-of-the-art methods, validating its effectiveness and robustness. Our code and data are available at this https URL.
>
---
#### [replaced 006] NS-Net: Decoupling CLIP Semantic Information through NULL-Space for Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01248](https://arxiv.org/pdf/2508.01248)**

> **作者:** Jiazhen Yan; Fan Wang; Weiwei Jiang; Ziqiang Li; Zhangjie Fu
>
> **摘要:** The rapid progress of generative models, such as GANs and diffusion models, has facilitated the creation of highly realistic images, raising growing concerns over their misuse in security-sensitive domains. While existing detectors perform well under known generative settings, they often fail to generalize to unknown generative models, especially when semantic content between real and fake images is closely aligned. In this paper, we revisit the use of CLIP features for AI-generated image detection and uncover a critical limitation: the high-level semantic information embedded in CLIP's visual features hinders effective discrimination. To address this, we propose NS-Net, a novel detection framework that leverages NULL-Space projection to decouple semantic information from CLIP's visual features, followed by contrastive learning to capture intrinsic distributional differences between real and generated images. Furthermore, we design a Patch Selection strategy to preserve fine-grained artifacts by mitigating semantic bias caused by global image structures. Extensive experiments on an open-world benchmark comprising images generated by 40 diverse generative models show that NS-Net outperforms existing state-of-the-art methods, achieving a 7.4\% improvement in detection accuracy, thereby demonstrating strong generalization across both GAN- and diffusion-based image generation techniques.
>
---
#### [replaced 007] iLLaVA: An Image is Worth Fewer Than 1/3 Input Tokens in Large Multimodal Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.06263](https://arxiv.org/pdf/2412.06263)**

> **作者:** Lianyu Hu; Liqing Gao; Fanhua Shang; Liang Wan; Wei Feng
>
> **备注:** Accepted by ICLR2026,code is released at this https URL
>
> **摘要:** Recent methods have made notable progress in accelerating Large Vision-Language Models (LVLMs) by exploiting the inherent redundancy in visual inputs. Most existing approaches, however, focus narrowly on reducing image tokens before or within the Large Language Model (LLM) stage to lower computational cost. This overlooks other major bottlenecks, particularly the image encoder, which itself requires substantial computation. As a result, these methods fall short of achieving true end-to-end acceleration. Importantly, the image encoder is the primary contributor of input tokens to the LLM. Thus, reducing visual redundancy at the encoder stage not only speeds up the encoder itself but also significantly lightens the workload for the subsequent LLM. Motivated by this, we investigate how to jointly optimize the image encoder and the LLM along with other LVLM components for comprehensive acceleration. To mitigate the risk of performance degradation from token reduction, we propose a novel token merging strategy that recycles useful information from otherwise discarded tokens. Our approach, iLLaVA, delivers consistent improvements across both image and video understanding tasks, achieving up to a 2 times throughput boost and a 4 times reduction in prefilling time. Notably, iLLaVA enables a larger model (e.g., InternVL-2.5 26B) to surpass a smaller counterpart (e.g., InternVL-2.5 8B) in both accuracy and efficiency. Extensive comparisons with state-of-the-art token pruning and merging techniques demonstrate the clear superiority of our method. Finally, we provide detailed visualizations for the merging steps of iLLaVA , offering deeper insights into how different LVLM components contribute to efficient computation.
>
---
#### [replaced 008] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.05437](https://arxiv.org/pdf/2603.05437)**

> **作者:** Ye-Chan Kim; SeungJu Cha; Si-Woo Kim; Minju Jeon; Hyungee Kim; Dong-Jin Kim
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Weakly-Supervised Dense Video Captioning aims to localize and describe events in videos trained only on caption annotations, without temporal boundaries. Prior work introduced an implicit supervision paradigm based on Gaussian masking and complementary captioning. However, existing method focuses merely on generating non-overlapping masks without considering their semantic relationship to corresponding events, resulting in simplistic, uniformly distributed masks that fail to capture semantically meaningful regions. Moreover, relying solely on ground-truth captions leads to sub-optimal performance due to the inherent sparsity of existing datasets. In this work, we propose SAIL, which constructs semantically-aware masks through cross-modal alignment. Our similarity aware training objective guides masks to emphasize video regions with high similarity to their corresponding event captions. Furthermore, to guide more accurate mask generation under sparse annotation settings, we introduce an LLM-based augmentation strategy that generates synthetic captions to provide additional alignment signals. These synthetic captions are incorporated through an inter-mask mechanism, providing auxiliary guidance for precise temporal localization without degrading the main objective. Experiments on ActivityNet Captions and YouCook2 demonstrate state-of-the-art performance on both captioning and localization metrics.
>
---
#### [replaced 009] Event-Based Visual Teach-and-Repeat via Fast Fourier-Domain Cross-Correlation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉教-重复导航任务，解决机器人实时路径重现已知轨迹的问题。通过事件相机与傅里叶域互相关方法，实现高效低延迟的视觉匹配。**

- **链接: [https://arxiv.org/pdf/2509.17287](https://arxiv.org/pdf/2509.17287)**

> **作者:** Gokul B. Nair; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **备注:** 8 Pages, 5 Figures, Under Review
>
> **摘要:** Visual teach-and-repeat (VT&R) navigation enables robots to autonomously traverse previously demonstrated paths using visual feedback. We present a novel event-camera-based VT\&R system. Our system formulates event-stream matching as frequency-domain cross-correlation, transforming spatial convolutions into efficient Fourier-space multiplications. By exploiting the binary structure of event frames and applying image compression techniques, we achieve a processing latency of just 2.88 ms, about 3.5 times faster than conventional camera-based baselines that are optimised for runtime efficiency. Experiments using a Prophesee EVK4 HD event camera mounted on an AgileX Scout Mini robot demonstrate successful autonomous navigation across 3000+ meters of indoor and outdoor trajectories in daytime and nighttime conditions. Our system maintains Cross-Track Errors (XTE) below 15 cm, demonstrating the practical viability of event-based perception for real-time VT\&R navigation.
>
---
#### [replaced 010] LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03711](https://arxiv.org/pdf/2603.03711)**

> **作者:** Yuanming Cao; Chengqi Li; Wenbo He
>
> **摘要:** Local Differential Privacy (LDP) is the gold standard trust model for privacy-preserving machine learning by guaranteeing privacy at the data source. However, its application to image data has long been considered impractical due to the high dimensionality of pixel space. Canonical LDP mechanisms are designed for low-dimensional data, resulting in severe utility degradation when applied to high-dimensional pixel spaces. This paper demonstrates that this utility loss is not inherent to LDP, but from its application to an inappropriate data representation. We introduce LDP-Slicing, a lightweight, training-free framework that resolves this domain mismatch. Our key insight is to decompose pixel values into a sequence of binary bit-planes. This transformation allows us to apply the LDP mechanism directly to the bit-level representation. To further strengthen privacy and preserve utility, we integrate a perceptual obfuscation module that mitigates human-perceivable leakage and an optimization-based privacy budget allocation strategy. This pipeline satisfies rigorous pixel-level $\varepsilon$-LDP while producing images that retain high utility for downstream tasks. Extensive experiments on face recognition and image classification demonstrate that LDP-Slicing outperforms existing DP/LDP baselines under comparable privacy budgets, with negligible computational overhead.
>
---
#### [replaced 011] Quantized Visual Geometry Grounded Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21302](https://arxiv.org/pdf/2509.21302)**

> **作者:** Weilun Feng; Haotong Qin; Mingqiang Wu; Chuanguang Yang; Yuqi Li; Xiangqi Li; Zhulin An; Libo Huang; Yulun Zhang; Michele Magno; Yongjun Xu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Learning-based 3D reconstruction models, represented by Visual Geometry Grounded Transformers (VGGTs), have made remarkable progress with the use of large-scale transformers. Their prohibitive computational and memory costs severely hinder real-world deployment. Post-Training Quantization (PTQ) has become a common practice for compressing and accelerating models. However, we empirically observe that PTQ faces unique obstacles when compressing billion-scale VGGTs: the data-independent special tokens induce heavy-tailed activation distributions, while the multi-view nature of 3D data makes calibration sample selection highly unstable. This paper proposes the first Quantization framework for VGGTs, namely QuantVGGT. This mainly relies on two technical contributions: First, we introduce Dual-Smoothed Fine-Grained Quantization, which integrates pre-global Hadamard rotation and post-local channel smoothing to mitigate heavy-tailed distributions and inter-channel variance robustly. Second, we design Noise-Filtered Diverse Sampling, which filters outliers via deep-layer statistics and constructs frame-aware diverse calibration clusters to ensure stable quantization ranges. Comprehensive experiments demonstrate that QuantVGGT achieves the state-of-the-art results across different benchmarks and bit-width, surpassing the previous state-of-the-art generic quantization method with a great margin. We highlight that our 4-bit QuantVGGT can deliver a 3.7$\times$ memory reduction and 2.5$\times$ acceleration in real-hardware inference, while maintaining reconstruction accuracy above 98\% of its full-precision counterpart. This demonstrates the vast advantages and practicality of QuantVGGT in resource-constrained scenarios. Our code is released in this https URL.
>
---
#### [replaced 012] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.06572](https://arxiv.org/pdf/2603.06572)**

> **作者:** Vishal Thengane; Zhaochong An; Tianjin Huang; Son Lam Phung; Abdesselam Bouzerdoum; Lu Yin; Na Zhao; Xiatian Zhu
>
> **备注:** Accepted at CVPR 2026 (Findings)
>
> **摘要:** Incremental Few-Shot (IFS) segmentation aims to learn new categories over time from only a few annotations. Although widely studied in 2D, it remains underexplored for 3D point clouds. Existing methods suffer from catastrophic forgetting or fail to learn discriminative prototypes under sparse supervision, and often overlook a key cue: novel categories frequently appear as unlabelled background in base-training scenes. We introduce SCOPE (Scene-COntextualised Prototype Enrichment), a plug-and-play background-guided prototype enrichment framework that integrates with any prototype-based 3D segmentation method. After base training, a class-agnostic segmentation model extracts high-confidence pseudo-instances from background regions to build a prototype pool. When novel classes arrive with few labelled samples, relevant background prototypes are retrieved and fused with few-shot prototypes to form enriched representations without retraining the backbone or adding parameters. Experiments on ScanNet and S3DIS show that SCOPE achieves SOTA performance, improving novel-class IoU by up to 6.98% and 3.61%, and mean IoU by 2.25% and 1.70%, respectively, while maintaining low forgetting. Code is available this https URL.
>
---
#### [replaced 013] UniUGG: Unified 3D Understanding and Generation via Geometric-Semantic Encoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11952](https://arxiv.org/pdf/2508.11952)**

> **作者:** Yueming Xu; Jiahui Zhang; Ze Huang; Yurui Chen; Yanpeng Zhou; Zhenyu Chen; Yu-Jie Yuan; Pengxiang Xia; Guowei Huang; Xinyue Cai; Zhongang Qi; Xingyue Quan; Jianye Hao; Hang Xu; Li Zhang
>
> **摘要:** Despite the impressive progress on understanding and generating images shown by the recent unified architectures, the integration of 3D tasks remains challenging and largely unexplored. In this paper, we introduce UniUGG, the first unified understanding and generation framework for 3D modalities. Our unified framework employs an LLM to comprehend and decode sentences and 3D representations. At its core, we propose a spatial decoder leveraging a latent diffusion model to generate high-quality 3D representations. This allows for the generation and imagination of 3D scenes based on a reference image and an arbitrary view transformation, while remaining supports for spatial visual question answering (VQA) tasks. Additionally, we propose a geometric-semantic learning strategy to pretrain the vision encoder. This design jointly captures the input's semantic and geometric cues, enhancing both spatial understanding and generation. Extensive experimental results demonstrate the superiority of our method in visual representation, spatial understanding, and 3D generation.
>
---
#### [replaced 014] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DrivingGen，用于评估自动驾驶中的生成式世界模型。解决缺乏全面基准的问题，整合多样数据与新指标，评估视觉真实、轨迹合理性等。**

- **链接: [https://arxiv.org/pdf/2601.01528](https://arxiv.org/pdf/2601.01528)**

> **作者:** Yang Zhou; Hao Shao; Letian Wang; Zhuofan Zong; Hongsheng Li; Steven L. Waslander
>
> **备注:** ICLR 2026 Poster; Project Website: this https URL
>
> **摘要:** Video generation models, as one form of world models, have emerged as one of the most exciting frontiers in AI, promising agents the ability to imagine the future by modeling the temporal evolution of complex scenes. In autonomous driving, this vision gives rise to driving world models: generative simulators that imagine ego and agent futures, enabling scalable simulation, safe testing of corner cases, and rich synthetic data generation. Yet, despite fast-growing research activity, the field lacks a rigorous benchmark to measure progress and guide priorities. Existing evaluations remain limited: generic video metrics overlook safety-critical imaging factors; trajectory plausibility is rarely quantified; temporal and agent-level consistency is neglected; and controllability with respect to ego conditioning is ignored. Moreover, current datasets fail to cover the diversity of conditions required for real-world deployment. To address these gaps, we present DrivingGen, the first comprehensive benchmark for generative driving world models. DrivingGen combines a diverse evaluation dataset curated from both driving datasets and internet-scale video sources, spanning varied weather, time of day, geographic regions, and complex maneuvers, with a suite of new metrics that jointly assess visual realism, trajectory plausibility, temporal coherence, and controllability. Benchmarking 14 state-of-the-art models reveals clear trade-offs: general models look better but break physics, while driving-specific ones capture motion realistically but lag in visual quality. DrivingGen offers a unified evaluation framework to foster reliable, controllable, and deployable driving world models, enabling scalable simulation, planning, and data-driven decision-making.
>
---
#### [replaced 015] PointSlice: Accurate and Efficient Slice-Based Representation for 3D Object Detection from Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.01487](https://arxiv.org/pdf/2509.01487)**

> **作者:** Liu Qifeng; Zhao Dawei; Dong Yabo; Xiao Liang; Wang Juan; Min Chen; Li Fuyang; Jiang Weizhong; Lu Dongming; Nie Yiming
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** 3D object detection from point clouds plays a critical role in autonomous driving. Currently, the primary methods for point cloud processing are voxel-based and pillar-based approaches. Voxel-based methods offer high accuracy through fine-grained spatial segmentation but suffer from slower inference speeds. Pillar-based methods enhance inference speed but typically lag behind voxel-based methods in detection accuracy. To address this trade-off, we propose a novel point cloud processing method, PointSlice, which slices point clouds along the horizontal plane and incorporates a dedicated detection network. The main contributions of PointSlice are: (1) A novel slice-based representation that converts 3D point clouds into multiple sets of 2D (x-y) data slices. The model explicitly learns 2D data distributions by treating the 3D point cloud as separate batches of 2D data, which significantly reduces the parameter count and enhances inference speed; (2) The introduction of a Slice Interaction Network (SIN). To preserve vertical geometric relationships across slices, we incorporate SIN into the 2D backbone network, thereby improving the model's 3D perception capability. Extensive experiments demonstrate that PointSlice achieves a superior balance between detection accuracy and efficiency. On the Waymo Open Dataset, PointSlice achieves a 1.13$\times$ speedup and uses 0.79$\times$ the parameters of the state-of-the-art voxel-based method (SAFDNet), with a marginal 1.2 mAPH accuracy reduction. On the nuScenes dataset, we achieve a state-of-the-art 66.7 mAP. On the Argoverse 2 dataset, PointSlice is 1.10$\times$ faster with 0.66$\times$ the parameters, while showing a negligible accuracy drop of 1.0 mAP. The source code is available at this https URL.
>
---
#### [replaced 016] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07580](https://arxiv.org/pdf/2512.07580)**

> **作者:** Yahong Wang; Juncheng Wu; Zhangkai Ni; Longzhen Yang; Yihang Liu; Chengmei Yang; Ying Wen; Lianghua He; Xianfeng Tang; Hui Liu; Yuyin Zhou
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision Large Language Models (VLLMs) incur high computational costs due to their reliance on hundreds of visual tokens to represent images. While token pruning offers a promising solution for accelerating inference, this paper, however, identifies a key observation: in deeper layers (e.g., beyond the 20th), existing training-free pruning methods perform no better than random pruning. We hypothesize that this degradation is caused by \textbf{``vanishing token information''}, where visual tokens progressively lose their salience with increasing network depth. To validate this hypothesis, we quantify a token's information content by measuring the change in the model output probabilities upon its removal. Using this proposed metric, our analysis of the information of visual tokens across layers reveals three key findings: (1) As layers deepen, the information of visual tokens gradually becomes uniform and eventually vanishes at an intermediate layer, which we term as ``information horizon", beyond which the visual tokens become redundant; (2) The position of this horizon is not static; it extends deeper for visually intensive tasks, such as Optical Character Recognition (OCR), compared to more general tasks like Visual Question Answering (VQA); (3) This horizon is also strongly correlated with model capacity, as stronger VLLMs (e.g., Qwen2.5-VL) employ deeper visual tokens than weaker models (e.g., LLaVA-1.5). Based on our findings, we show that simple random pruning in deep layers efficiently balances performance and efficiency. Moreover, integrating random pruning consistently enhances existing methods. Using DivPrune with random pruning achieves state-of-the-art results, maintaining 96.9\% of Qwen-2.5-VL-7B performance while pruning 50\% of visual tokens. The code is available at this https URL.
>
---
#### [replaced 017] Are vision-language models ready to zero-shot replace supervised classification models in agriculture?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15977](https://arxiv.org/pdf/2512.15977)**

> **作者:** Earl Ranario; Mason J. Earles
>
> **摘要:** Vision-language models (VLMs) are increasingly proposed as general-purpose solutions for visual recognition tasks, yet their reliability for agricultural decision support remains poorly understood. We benchmark a diverse set of open-source and closed-source VLMs on 27 agricultural image classification datasets from the AgML collection (this https URL), spanning 162 classes and 248,000 images across plant disease, pest and damage, and plant and weed species identification. Across all tasks, zero-shot VLMs substantially underperform a supervised task-specific baseline (YOLO11), which consistently achieves markedly higher accuracy than any foundation model. Under multiple-choice prompting, the best-performing VLM (Gemini-3 Pro) reaches approximately 62% average accuracy, while open-ended prompting yields much lower performance, with raw accuracies typically below 25%. Applying LLM-based semantic judging increases open-ended accuracy (e.g., from ~21% to ~30% for top models) and alters model rankings, demonstrating that evaluation methodology meaningfully affects reported conclusions. Among open-source models, Qwen-VL-72B performs best, approaching closed-source performance under constrained prompting but still trailing top proprietary systems. Task-level analysis shows that plant and weed species classification is consistently easier than pest and damage identification, which remains the most challenging category across models. Overall, these results indicate that current off-the-shelf VLMs are not yet suitable as standalone agricultural diagnostic systems, but can function as assistive components when paired with constrained interfaces, explicit label ontologies, and domain-aware evaluation strategies.
>
---
#### [replaced 018] Post-Disaster Affected Area Segmentation with a Vision Transformer (ViT)-based EVAP Model using Sentinel-2 and Formosat-5 Imagery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.16849](https://arxiv.org/pdf/2507.16849)**

> **作者:** Yi-Shan Chu; Hsuan-Cheng Wei
>
> **摘要:** We propose a vision transformer (ViT)-based deep learning framework to refine disaster-affected area segmentation from remote sensing imagery, aiming to support and enhance the Emergent Value Added Product (EVAP) developed by the Taiwan Space Agency (TASA). The process starts with a small set of manually annotated regions. We then apply principal component analysis (PCA)-based feature space analysis and construct a confidence index (CI) to expand these labels, producing a weakly supervised training set. These expanded labels are then used to train ViT-based encoder-decoder models with multi-band inputs from Sentinel-2 and Formosat-5 imagery. Our architecture supports multiple decoder variants and multi-stage loss strategies to improve performance under limited supervision. During the evaluation, model predictions are compared with higher-resolution EVAP output to assess spatial coherence and segmentation consistency. Case studies on the 2022 Poyang Lake drought and the 2023 Rhodes wildfire demonstrate that our framework improves the smoothness and reliability of segmentation results, offering a scalable approach for disaster mapping when accurate ground truth is unavailable.
>
---
#### [replaced 019] Real-Time Motion-Controllable Autoregressive Video Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.08131](https://arxiv.org/pdf/2510.08131)**

> **作者:** Kesen Zhao; Jiaxin Shi; Beier Zhu; Junbao Zhou; Xiaolong Shen; Yuan Zhou; Qianru Sun; Hanwang Zhang
>
> **摘要:** Real-time motion-controllable video generation remains challenging due to the inherent latency of bidirectional diffusion models and the lack of effective autoregressive (AR) approaches. Existing AR video diffusion models are limited to simple control signals or text-to-video generation, and often suffer from quality degradation and motion artifacts in few-step generation. To address these challenges, we propose AR-Drag, the first RL-enhanced few-step AR video diffusion model for real-time image-to-video generation with diverse motion control. We first fine-tune a base I2V model to support basic motion control, then further improve it via reinforcement learning with a trajectory-based reward model. Our design preserves the Markov property through a Self-Rollout mechanism and accelerates training by selectively introducing stochasticity in denoising steps. Extensive experiments demonstrate that AR-Drag achieves high visual fidelity and precise motion alignment, significantly reducing latency compared with state-of-the-art motion-controllable VDMs, while using only 1.3B parameters. Additional visualizations can be found on our project page: this https URL.
>
---
#### [replaced 020] Efficient Domain-Adaptive Multi-Task Dense Prediction with Vision Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23626](https://arxiv.org/pdf/2509.23626)**

> **作者:** Beomseok Kang; Niluthpol Chowdhury Mithun; Mikhail Sizintsev; Han-Pang Chiu; Supun Samarasekera
>
> **摘要:** Multi-task dense prediction, which aims to jointly solve tasks like semantic segmentation and depth estimation, is crucial for robotics applications but suffers from domain shift when deploying models in new environments. While unsupervised domain adaptation (UDA) addresses this challenge for single tasks, existing multi-task UDA methods primarily rely on adversarial learning approaches that are less effective than recent self-training techniques. In this paper, we introduce FAMDA, a simple yet effective UDA framework that addresses this limitation by leveraging Vision Foundation Models (VFMs) as powerful teachers within a self-training paradigm. Our approach integrates Segmentation and Depth foundation models into a self-training paradigm to generate high-quality pseudo-labels for the target domain, effectively distilling their robust generalization capabilities into a single, efficient student network. Extensive experiments show that FAMDA achieves state-of-the-art (SOTA) performance on standard synthetic-to-real UDA multi-task learning (MTL) benchmarks and a challenging new day-to-night adaptation task. Our framework enables the training of highly efficient models; a lightweight variant achieves SOTA accuracy while being more than 10X smaller than foundation models, highlighting FAMDA's suitability for creating domain-adaptive and efficient models for resource-constrained robotics applications.
>
---
#### [replaced 021] LaVCa: LLM-assisted Visual Cortex Captioning
- **分类: q-bio.NC; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出LaVCa方法，用于生成脑区选择性图像的自然语言描述，解决神经元群体属性解释难题，通过LLM提升对视觉皮层功能的理解。**

- **链接: [https://arxiv.org/pdf/2502.13606](https://arxiv.org/pdf/2502.13606)**

> **作者:** Takuya Matsuyama; Shinji Nishimoto; Yu Takagi
>
> **备注:** Accepted to ICLR 2026. Website: this https URL
>
> **摘要:** Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations.
>
---
#### [replaced 022] Cycle-Consistent Tuning for Layered Image Decomposition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20989](https://arxiv.org/pdf/2602.20989)**

> **作者:** Zheng Gu; Min Lu; Zhida Sun; Dani Lischinski; Daniel Cohen-Or; Hui Huang
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Disentangling visual layers in real-world images is a persistent challenge in vision and graphics, as such layers often involve non-linear and globally coupled interactions, including shading, reflection, and perspective distortion. In this work, we present an in-context image decomposition framework that leverages large diffusion foundation models for layered separation. We focus on the challenging case of logo-object decomposition, where the goal is to disentangle a logo from the surface on which it appears while faithfully preserving both layers. Our method fine-tunes a pretrained diffusion model via lightweight LoRA adaptation and introduces a cycle-consistent tuning strategy that jointly trains decomposition and composition models, enforcing reconstruction consistency between decomposed and recomposed images. This bidirectional supervision substantially enhances robustness in cases where the layers exhibit complex interactions. Furthermore, we introduce a progressive self-improving process, which iteratively augments the training set with high-quality model-generated examples to refine performance. Extensive experiments demonstrate that our approach achieves accurate and coherent decompositions and also generalizes effectively across other decomposition types, suggesting its potential as a unified framework for layered image decomposition.
>
---
#### [replaced 023] ReMeDI: Refined Memory for Disambiguation of Identities with SAM3 in Surgical Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.16880](https://arxiv.org/pdf/2512.16880)**

> **作者:** Valay Bundele; Mehran Hosseinzadeh; Hendrik P.A. Lensch
>
> **备注:** Under Review
>
> **摘要:** Accurate surgical instrument segmentation in endoscopy is crucial for computer-assisted interventions, yet remains challenging due to frequent occlusions, rapid motion, and long-term instrument re-entry. While SAM3 provides a powerful spatio-temporal framework for video object segmentation, its performance in surgical scenes is limited by indiscriminate memory updates, fixed memory capacity, and weak identity recovery after occlusions. We propose ReMeDI-SAM3, a training-free extension of SAM3, that addresses these limitations through three components: (i) relevance-aware memory filtering with a dedicated occlusion-aware memory for storing pre-occlusion frames, (ii) a piecewise interpolation scheme that expands effective memory capacity, and (iii) a feature-based re-identification module with temporal voting for reliable post-occlusion identity disambiguation. Together, these components mitigate error accumulation and enable reliable recovery after occlusions. Evaluations on EndoVis17, EndoVis18 and CholecSeg8k under a zero-shot setting show mcIoU improvements of around 5.8\%, 8\%, and 2\% respectively, over vanilla SAM3, outperforming even prior training-based approaches.
>
---
#### [replaced 024] Multimodal Large Language Models as Image Classifiers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06578](https://arxiv.org/pdf/2603.06578)**

> **作者:** Nikita Kisel; Illia Volkov; Klara Janouskova; Jiri Matas
>
> **摘要:** Multimodal Large Language Models (MLLM) classification performance depends critically on evaluation protocol and ground truth quality. Studies comparing MLLMs with supervised and vision-language models report conflicting conclusions, and we show these conflicts stem from protocols that either inflate or underestimate performance. Across the most common evaluation protocols, we identify and fix key issues: model outputs that fall outside the provided class list and are discarded, inflated results from weak multiple-choice distractors, and an open-world setting that underperforms only due to poor output mapping. We additionally quantify the impact of commonly overlooked design choices - batch size, image ordering, and text encoder selection - showing they substantially affect accuracy. Evaluating on ReGT, our multilabel reannotation of 625 ImageNet-1k classes, reveals that MLLMs benefit most from corrected labels (up to +10.8%), substantially narrowing the perceived gap with supervised models. Much of the reported MLLMs underperformance on classification is thus an artifact of noisy ground truth and flawed evaluation protocol rather than genuine model deficiency. Models less reliant on supervised training signals prove most sensitive to annotation quality. Finally, we show that MLLMs can assist human annotators: in a controlled case study, annotators confirmed or integrated MLLMs predictions in approximately 50% of difficult cases, demonstrating their potential for large-scale dataset curation. This work is part of the Aiming for Perfect ImageNet-1k project, see this https URL.
>
---
#### [replaced 025] Enhancing Alzheimer's Diagnosis: Leveraging Anatomical Landmarks in Graph Convolutional Neural Networks on Tetrahedral Meshes
- **分类: eess.IV; cs.AI; cs.CV; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2503.05031](https://arxiv.org/pdf/2503.05031)**

> **作者:** Yanxi Chen; Mohammad Farazi; Zhangsihao Yang; Yonghui Fan; Nicholas Ashton; Eric M Reiman; Yi Su; Yalin Wang
>
> **摘要:** Alzheimer's disease (AD) is a major neurodegenerative condition that affects millions around the world. As one of the main biomarkers in the AD diagnosis procedure, brain amyloid positivity is typically identified by positron emission tomography (PET), which is costly and invasive. Brain structural magnetic resonance imaging (sMRI) may provide a safer and more convenient solution for the AD diagnosis. Recent advances in geometric deep learning have facilitated sMRI analysis and early diagnosis of AD. However, determining AD pathology, such as brain amyloid deposition, in preclinical stage remains challenging, as less significant morphological changes can be observed. As a result, few AD classification models are generalizable to the brain amyloid positivity classification task. Blood-based biomarkers (BBBMs), on the other hand, have recently achieved remarkable success in predicting brain amyloid positivity and identifying individuals with high risk of being brain amyloid positive. However, individuals in medium risk group still require gold standard tests such as Amyloid PET for further evaluation. Inspired by the recent success of transformer architectures, we propose a geometric deep learning model based on transformer that is both scalable and robust to variations in input volumetric mesh size. Our work introduced a novel tokenization scheme for tetrahedral meshes, incorporating anatomical landmarks generated by a pre-trained Gaussian process model. Our model achieved superior classification performance in AD classification task. In addition, we showed that the model was also generalizable to the brain amyloid positivity prediction with individuals in the medium risk class, where BM alone cannot achieve a clear classification. Our work may enrich geometric deep learning research and improve AD diagnosis accuracy without using expensive and invasive PET scans.
>
---
#### [replaced 026] Multi-Scale Distillation for RGB-D Anomaly Detection on the PD-REAL Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2311.04095](https://arxiv.org/pdf/2311.04095)**

> **作者:** Jianjian Qin; Chao Zhang; Chunzhi Gu; Zi Wang; Jun Yu; Yijin Wei; Hui Xiao; Xin Yua
>
> **摘要:** We present PD-REAL, a novel large-scale dataset for unsupervised anomaly detection (AD) in the 3D domain. It is motivated by the fact that 2D-only representations in the AD task may fail to capture the geometric structures of anomalies due to uncertainty in lighting conditions or shooting angles. PD-REAL consists entirely of Play-Doh models for 15 object categories and focuses on the analysis of potential benefits from 3D information in a controlled environment. Specifically, objects are first created with six types of anomalies, such as \textit{dent}, \textit{crack}, or \textit{perforation}, and then photographed under different lighting conditions to mimic real-world inspection scenarios. To demonstrate the usefulness of 3D information, we use a commercially available RealSense camera to capture RGB and depth images. Compared to the existing 3D dataset for AD tasks, the data acquisition of PD-REAL is significantly cheaper, easily scalable, and easier to control variables. \qin{Furthermore, we introduce a multi-scale teacher--student framework with hierarchical distillation for multimodal anomaly detection. This architecture overcomes the inherent limitation of single-scale distillation approaches, which often struggle to reconcile global context with local features. Leveraging multi-level guidance from the teacher network, the student network can effectively capture richer features for anomaly detection. Extensive evaluations with our method and state-of-the-art AD algorithms on our dataset qualitatively and quantitatively demonstrate the higher detection accuracy of our method. }Our dataset can be downloaded from this https URL
>
---
#### [replaced 027] Radiative-Structured Neural Operator for Continuous and Extrapolative Spectral Super-Resolution
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17895](https://arxiv.org/pdf/2511.17895)**

> **作者:** Ziye Zhang; Bin Pan; Zhenwei Shi
>
> **摘要:** Spectral super-resolution (SSR) aims to reconstruct hyperspectral images (HSIs) from multispectral observations, with broad applications in computer vision and remote sensing. Deep learning-based methods have been widely used, but they often treat spectra as discrete vectors learned from data, rather than continuous curves constrained by physics principles, leading to unrealistic predictions and limited applicability. To address this challenge, we propose the Radiative-Structured Neural Operator (RSNO), which learns a continuous mapping for spectral super-resolution while enforcing physical consistency under the radiative prior. The proposed RSNO consists of three stages: upsampling, reconstruction, and refinement. In the upsampling stage, we leverage prior information to expand the input multispectral image, producing a physically plausible hyperspectral estimate. Subsequently, we adopt a neural operator backbone in the reconstruction stage to learn a continuous mapping across the spectral domain. Finally, the refinement stage imposes a hard constraint on the output HSI to eliminate color distortion. The upsampling and refinement stages are implemented via the proposed angular-consistent projection (ACP), which is derived from a non-convex optimization problem. Moreover, we theoretically demonstrated the optimality of ACP by null-space decomposition. Various experiments validate the effectiveness of the proposed approach across conventional spectral super-resolution, continuous spectral reconstruction, and infrared extrapolation.
>
---
#### [replaced 028] VOIC: Visible-Occluded Integrated Guidance for 3D Semantic Scene Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18954](https://arxiv.org/pdf/2512.18954)**

> **作者:** Zaidao Han; Risa Higashita; Jiang Liu
>
> **摘要:** Camera-based 3D Semantic Scene Completion (SSC) is a critical task for autonomous driving and robotic scene understanding. It aims to infer a complete 3D volumetric representation of both semantics and geometry from a single image. Existing methods typically focus on end-to-end 2D-to-3D feature lifting and voxel completion. However, they often overlook the interference between high-confidence visible-region perception and low-confidence occluded-region reasoning caused by single-image input, which can lead to feature dilution and error propagation. To address these challenges, we introduce an offline Visible Region Label Extraction (VRLE) strategy that explicitly separates and extracts voxel-level supervision for visible regions from dense 3D ground truth. This strategy purifies the supervisory space for two complementary sub-tasks: visible-region perception and occluded-region reasoning. Building on this idea, we propose the Visible-Occluded Interactive Completion Network (VOIC), a novel dual-decoder framework that explicitly decouples SSC into visible-region semantic perception and occluded-region scene completion. VOIC first constructs a base 3D voxel representation by fusing image features with depth-derived occupancy. The visible decoder focuses on generating high-fidelity geometric and semantic priors, while the occlusion decoder leverages these priors together with cross-modal interaction to perform coherent global scene reasoning. Extensive experiments on the SemanticKITTI and SSCBench-KITTI360 benchmarks demonstrate that VOIC outperforms existing monocular SSC methods in both geometric completion and semantic segmentation accuracy, achieving state-of-the-art performance.
>
---
#### [replaced 029] Towards High-resolution and Disentangled Reference-based Sketch Colorization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05971](https://arxiv.org/pdf/2603.05971)**

> **作者:** Dingkun Yan; Xinrui Wang; Ru Wang; Zhuoru Li; Jinze Yu; Yusuke Iwasawa; Yutaka Matsuo; Jiaxian Guo
>
> **摘要:** Sketch colorization is a critical task for automating and assisting in the creation of animations and digital illustrations. Previous research identified the primary difficulty as the distribution shift between semantically aligned training data and highly diverse test data, and focused on mitigating the artifacts caused by the distribution shift instead of fundamentally resolving the problem. In this paper, we present a framework that directly minimizes the distribution shift, thereby achieving superior quality, resolution, and controllability of colorization. We propose a dual-branch framework to explicitly model the data distributions of the training process and inference process with a semantic-aligned branch and a semantic-misaligned branch, respectively. A Gram Regularization Loss is applied across the feature maps of both branches, effectively enforcing cross-domain distribution coherence and stability. Furthermore, we adopt an anime-specific Tagger Network to extract fine-grained attributions from reference images and modulate SDXL's conditional encoders to ensure precise control, and a plugin module to enhance texture transfer. Quantitative and qualitative comparisons, alongside user studies, confirm that our method effectively overcomes the distribution shift challenge, establishing State-of-the-Art performance across both quality and controllability metrics. Ablation study reveals the influence of each component.
>
---
#### [replaced 030] Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.02919](https://arxiv.org/pdf/2603.02919)**

> **作者:** Youngjun Jun; Seil Kang; Woojung Han; Seong Jae Hwang
>
> **备注:** CVPR 2026
>
> **摘要:** Video Diffusion Transformers (DiTs) have been synthesizing high-quality video with high fidelity from given text descriptions involving motion. However, understanding how Video DiTs convert motion words into video remains insufficient. Furthermore, while prior studies on interpretable saliency maps primarily target objects, motion-related behavior in Video DiTs remains largely unexplored. In this paper, we investigate concrete motion features that specify when and which object moves for a given motion concept. First, to spatially localize, we introduce GramCol, which adaptively produces per-frame saliency maps for any text concept, including both motion and non-motion. Second, we propose a motion-feature selection algorithm to obtain an Interpretable Motion-Attentive Map (IMAP) that localizes motion spatially and temporally. Our method discovers concept saliency maps without the need for any gradient calculation or parameter update. Experimentally, our method shows outstanding localization capability on the motion localization task and zero-shot video semantic segmentation, providing interpretable and clearer saliency maps for both motion and non-motion concepts.
>
---
#### [replaced 031] Taming Modality Entanglement in Continual Audio-Visual Segmentation
- **分类: cs.MM; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.17234](https://arxiv.org/pdf/2510.17234)**

> **作者:** Yuyang Hong; Qi Yang; Tao Zhang; Zili Wang; Zhaojin Fu; Kun Ding; Bin Fan; Shiming Xiang
>
> **摘要:** Recently, significant progress has been made in multi-modal continual learning, aiming to learn new tasks sequentially in multi-modal settings while preserving performance on previously learned ones. However, existing methods mainly focus on coarse-grained tasks, with limitations in addressing modality entanglement in fine-grained continual learning settings. To bridge this gap, we introduce a novel Continual Audio-Visual Segmentation (CAVS) task, aiming to continuously segment new classes guided by audio. Through comprehensive analysis, two critical challenges are identified: 1) multi-modal semantic drift, where a sounding objects is labeled as background in sequential tasks; 2) co-occurrence confusion, where frequent co-occurring classes tend to be confused. In this work, a Collision-based Multi-modal Rehearsal (CMR) framework is designed to address these challenges. Specifically, for multi-modal semantic drift, a Multi-modal Sample Selection (MSS) strategy is proposed to select samples with high modal consistency for rehearsal. Meanwhile, for co-occurence confusion, a Collision-based Sample Rehearsal (CSR) mechanism is designed, allowing for the increase of rehearsal sample frequency of those confusable classes during training process. Moreover, we construct three audio-visual incremental scenarios to verify effectiveness of our method. Comprehensive experiments demonstrate that our method significantly outperforms single-modal continual learning methods.
>
---
#### [replaced 032] It is not always greener on the other side: Greenery perception across demographics and personalities in multiple cities
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17186](https://arxiv.org/pdf/2512.17186)**

> **作者:** Matias Quintana; Fangqi Liu; Jussi Torkko; Youlong Gu; Xiucheng Liang; Yujun Hou; Koichi Ito; Yihan Zhu; Mahmoud Abdelrahman; Tuuli Toivonen; Yi Lu; Filip Biljecki
>
> **摘要:** Quantifying and assessing urban greenery is consequential for planning and development, reflecting the everlasting importance of green spaces for multiple climate and well-being dimensions of cities. Evaluation can be broadly grouped into objective (e.g., measuring the amount of greenery) and subjective (e.g., polling the perception of people) approaches, which may differ -- what people see and feel about how green a place is might not match the measurements of the actual amount of vegetation. In this work, we advance the state of the art by measuring such differences and explaining them through human, geographic, and spatial dimensions. The experiments rely on contextual information extracted from street view imagery and a comprehensive urban visual perception survey collected from 1,000 people across five countries with their extensive demographic and personality information. We analyze the discrepancies between objective measures (e.g., Green View Index (GVI)) and subjective scores (e.g., pairwise ratings), examining whether they can be explained by a variety of human and visual factors such as age group and spatial variation of greenery in the scene. The findings reveal that such discrepancies are comparable around the world and that demographics and personality do not play a significant role in perception. Further, while perceived and measured greenery correlate consistently across geographies (both where people and where imagery are from), where people live plays a significant role in explaining perceptual differences, with these two, as the top among seven, features that influences perceived greenery the most. This location influence suggests that cultural, environmental, and experiential factors substantially shape how individuals observe greenery in cities.
>
---
#### [replaced 033] Universal 3D Shape Matching via Coarse-to-Fine Language Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19112](https://arxiv.org/pdf/2602.19112)**

> **作者:** Qinfeng Xiao; Guofeng Mei; Bo Yang; Liying Zhang; Jian Zhang; Kit-lun Yick
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Establishing dense correspondences between shapes is a crucial task in computer vision and graphics, while prior approaches depend on near-isometric assumptions and homogeneous subject types (i.e., only operate for human shapes). However, building semantic correspondences for cross-category objects remains challenging and has received relatively little attention. To achieve this, we propose UniMatch, a semantic-aware, coarse-to-fine framework for constructing dense semantic correspondences between strongly non-isometric shapes without restricting object categories. The key insight is to lift "coarse" semantic cues into "fine" correspondence, which is achieved through two stages. In the "coarse" stage, we perform class-agnostic 3D segmentation to obtain non-overlapping semantic parts and prompt multimodal large language models (MLLMs) to identify part names. Then, we employ pretrained vision language models (VLMs) to extract text embeddings, enabling the construction of matched semantic parts. In the "fine" stage, we leverage these coarse correspondences to guide the learning of dense correspondences through a dedicated rank-based contrastive scheme. Thanks to class-agnostic segmentation, language guiding, and rank-based contrastive learning, our method is versatile for universal object categories and requires no predefined part proposals, enabling universal matching for inter-class and non-isometric shapes. Extensive experiments demonstrate UniMatch consistently outperforms competing methods in various challenging scenarios.
>
---
#### [replaced 034] S$^2$Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04016](https://arxiv.org/pdf/2508.04016)**

> **作者:** Weilun Feng; Haotong Qin; Chuanguang Yang; Xiangqi Li; Han Yang; Yuqi Li; Zhulin An; Libo Huang; Michele Magno; Yongjun Xu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Diffusion transformers have emerged as the mainstream paradigm for video generation models. However, the use of up to billions of parameters incurs significant computational costs. Quantization offers a promising solution by reducing memory usage and accelerating inference. Nonetheless, we observe that the joint modeling of spatial and temporal information in video diffusion models (V-DMs) leads to extremely long token sequences, which introduces high calibration variance and learning challenges. To address these issues, we propose S$^2$Q-VDiT, a post-training quantization framework for V-DMs that leverages Salient data and Sparse token distillation. During the calibration phase, we identify that quantization performance is highly sensitive to the choice of calibration data. To mitigate this, we introduce \textit{Hessian-aware Salient Data Selection}, which constructs high-quality calibration datasets by considering both diffusion and quantization characteristics unique to V-DMs. To tackle the learning challenges, we further analyze the sparse attention patterns inherent in V-DMs. Based on this observation, we propose \textit{Attention-guided Sparse Token Distillation}, which exploits token-wise attention distributions to emphasize tokens that are more influential to the model's output. Under W4A6 quantization, S$^2$Q-VDiT achieves lossless performance while delivering $3.9\times$ model compression and $1.3\times$ inference acceleration. Code will be available at this https URL.
>
---
#### [replaced 035] Unified and Semantically Grounded Domain Adaptation for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.08660](https://arxiv.org/pdf/2508.08660)**

> **作者:** Xin Wang; Yin Guo; Jiamin Xia; Kaiyu Zhang; Niranjan Balu; Mahmud Mossa-Basha; Linda Shapiro; Chun Yuan
>
> **备注:** Accepted by IEEE Transactions on Medical Imaging
>
> **摘要:** Most prior unsupervised domain adaptation approaches for medical image segmentation are narrowly tailored to either the source-accessible setting, where adaptation is guided by source-target alignment, or the source-free setting, which typically resorts to implicit adaptation mechanisms such as pseudo-labeling and network distillation. This substantial divergence in methodological designs between the two settings reveals an inherent flaw: the lack of an explicit, structured construction of anatomical knowledge that naturally generalizes across domains and settings. To bridge this longstanding divide, we introduce a unified, semantically grounded framework that supports both source-accessible and source-free adaptation. Fundamentally distinct from all prior works, our framework's adaptability emerges naturally as a direct consequence of the model architecture, without relying on explicit cross-domain alignment strategies. Specifically, our model learns a domain-agnostic probabilistic manifold as a global space of anatomical regularities, mirroring how humans establish visual understanding. Thus, the structural content in each image can be interpreted as a canonical anatomy retrieved from the manifold and a spatial transformation capturing individual-specific geometry. This disentangled, interpretable formulation enables semantically meaningful prediction with intrinsic adaptability. Extensive experiments on challenging cardiac and abdominal datasets show that our framework achieves state-of-the-art results in both settings, with source-free performance closely approaching its source-accessible counterpart, a level of consistency rarely observed in prior works. The results provide a principled foundation for anatomically informed, interpretable, and unified solutions for domain adaptation in medical imaging. The code is available at this https URL
>
---
#### [replaced 036] Task-Oriented Semantic Compression for Localization at the Network Edge
- **分类: cs.CV; cs.NI**

- **链接: [https://arxiv.org/pdf/2504.18317](https://arxiv.org/pdf/2504.18317)**

> **作者:** Zhengru Fang; Senkang Hu; Yu Guo; Yiqin Deng; Yuguang Fang
>
> **摘要:** Achieving precise visual localization in GPS-limited urban environments poses significant challenges for resource-constrained mobile platforms, particularly under strict bandwidth, memory, and processing limitations. Inspired by mammalian spatial cognition, we propose a task-oriented communication framework in which bandwidth-limited endpoints equipped with multi-camera systems extract compact multi-view features and offload localization tasks to collaborative edge servers. We introduce the Orthogonally-constrained Variational Information Bottleneck encoder (O-VIB), which incorporates automatic relevance determination (ARD) to prune non-informative features while enforcing orthogonality to minimize redundancy. This enables efficient and accurate localization with minimal transmission overhead. Extensive evaluation on a real-world urban localization dataset demonstrates that O-VIB achieves high-precision localization under stringent bandwidth budgets, outperforming existing methods across diverse communication constraints.
>
---
#### [replaced 037] Traffic-MLLM: Curiosity-Regularized Supervised Learning for Traffic Scenario Case-Based Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11165](https://arxiv.org/pdf/2509.11165)**

> **作者:** Waikit Xiu; Qiang Lu; Bingchen Liu; Chen Sun; Xiying Li
>
> **摘要:** For safe and robust autonomous driving, decision-making systems must effectively leverage past experiences to handle the inherent long-tail of traffic scenarios. Case-Based Reasoning (CBR) provides a natural paradigm for this by adapting solutions from prior cases. However, in complex and dynamic traffic environments, traditional CBR methods struggle to effectively abstract and adapt knowledge under uncertainty. Meanwhile, although multimodal large language models (MLLMs) exhibit strong perceptual and linguistic capabilities, their reasoning behavior often relies on empirical pattern fitting, limiting robustness under distribution shift and long-tail scenarios. We propose Traffic-MLLM, a retrieval-free neural case modeling framework for multimodal traffic reasoning. Instead of performing explicit case retrieval at inference time, Traffic-MLLM learns a structured and generalizable case space directly during training. To support this learning process, we construct a multi-source case base by integrating dynamic traffic videos and large-scale static visual question-answering data, serving as a unified training substrate for learning structured case representations. To further improve representation quality near knowledge boundaries, we introduce a curiosity-driven refinement mechanism based on Random Network Distillation (RND), encouraging the model to internalize cross-case structural regularities rather than surface correlations. Experiments on the SUTD-TrafficQA and DriveQA benchmarks demonstrate consistent improvements in dynamic reasoning, regulatory understanding, and cross-domain transfer. Traffic-MLLM achieves 50.8% accuracy on SUTD-TrafficQA, 74.8% on the CARLA-based DriveQA split, and 83.1% on the real-world Mapillary split, indicating that representation-level case-space refinement provides an effective alternative to explicit retrieval for scalable multimodal case adaptation.
>
---
#### [replaced 038] Video2Layout: Recall and Reconstruct Metric-Grounded Cognitive Map for Spatial Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16160](https://arxiv.org/pdf/2511.16160)**

> **作者:** Yibin Huang; Wang Xu; Wanyue Zhang; Helu Zhi; Jingjing Huang; Yangbin Xu; Yangang Sun; Conghui Zhu; Tiejun Zhao
>
> **摘要:** Spatial intelligence is a critical frontier for Multimodal Large Language Models (MLLMs), empowering them to comprehend the physical world. Drawing inspiration from human perception mechanisms, prior studies attempt to construct a spatial understanding via grid-based cognitive maps. However, current grid-based map methods rely on discretized representations, which limit the model's ability in fine-grained spatial reasoning. To overcome this limitation, we propose Video2Layout, a framework for reconstructing metric-grounded spatial layouts from video. The framework uses continuous object boundary coordinates to enable quantitative spatial computation, which effectively reduces ambiguity in natural language descriptions of spatial relationships. Specifically, our method comprises two stages. First, in supervised fine-tuning stage, we construct a high-quality dataset from the AI2THOR simulator, which enables the model to learn the mapping from visual inputs to precise boundary coordinates. Subsequently, a reinforcement fine-tuning stage enhances the model's real-world generalization capabilities. Based on the above framework, we investigate factors that affect cognitive map accuracy and quantify its relationship with task performance. Evaluated on mainstream spatial reasoning benchmarks, our model, V2LO-7B, achieves an average improvement of 3.24\% over the model trained on grid maps, validating the superiority of our method.
>
---
#### [replaced 039] Snapmoji: Instant Generation of Animatable Dual-Stylized Avatars
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11978](https://arxiv.org/pdf/2503.11978)**

> **作者:** Eric M. Chen; Di Liu; Sizhuo Ma; Michael Vasilkovsky; Bing Zhou; Qiang Gao; Wenzhou Wang; Jiahao Luo; Dimitris N. Metaxas; Vincent Sitzmann; Jian Wang
>
> **备注:** N/A
>
> **摘要:** Despite the increasing popularity of avatar systems such as Snapchat Bitmojis, existing production avatar platforms face several limitations, such as a limited number of predefined assets, tedious customization processes, and inefficient rendering requirements. Addressing these shortcomings, we introduce Snapmoji, an avatar generation system that instantly creates 3D avatars, and enables customization in a process we call dual-stylization. Snapmoji first maps a selfie of a user to a primary avatar (e.g., Bitmoji style) using a new technique we name Gaussian Domain Adaptation (GDA), then applies a secondary style (e.g., skeleton, yarn, toy) to the primary avatar, all while preserving the user's identity. The generated 3D avatars can then be rendered an animated on mobile devices at 30-40 FPS.
>
---
#### [replaced 040] ExGS: Extreme 3D Gaussian Compression with Diffusion Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24758](https://arxiv.org/pdf/2509.24758)**

> **作者:** Jiaqi Chen; Xinhao Ji; Yuanyuan Gao; Hao Li; Yuning Gong; Yifei Liu; Dan Xu; Zhihang Zhong; Dingwen Zhang; Xiao Sun
>
> **摘要:** Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality. We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings. To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering. Our code repository will be released at: this https URL
>
---
#### [replaced 041] BotaCLIP: Contrastive Learning for Botany-Aware Representation of Earth Observation Data
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.21194](https://arxiv.org/pdf/2511.21194)**

> **作者:** Selene Cerna; Sara Si-Moussi; Wilfried Thuiller; Hadrien Hendrikx; Vincent Miele
>
> **摘要:** Foundation models have demonstrated a remarkable ability to learn rich, transferable representations across diverse modalities such as images, text, and audio. In modern machine learning pipelines, these representations often replace raw data as the primary input for downstream tasks. In this paper, we address the challenge of adapting a pre-trained foundation model to inject domain-specific knowledge, without retraining from scratch or incurring significant computational costs. To this end, we introduce BotaCLIP, a lightweight multimodal contrastive framework that adapts a pre-trained Earth Observation foundation model (DOFA) by aligning high-resolution aerial imagery with botanical relevés. Unlike generic embeddings, BotaCLIP internalizes ecological structure through contrastive learning with a regularization strategy that mitigates catastrophic forgetting. Once trained, the resulting embeddings serve as transferable representations for downstream predictors. Motivated by real-world applications in biodiversity modeling, we evaluated BotaCLIP representations in three ecological tasks: plant presence prediction, butterfly occurrence modeling, and soil trophic group abundance estimation. The results showed consistent improvements over those derived from DOFA and supervised baselines. More broadly, this work illustrates how domain-aware adaptation of foundation models can inject expert knowledge into data-scarce settings, enabling frugal representation learning.
>
---
#### [replaced 042] Gated Differential Linear Attention: A Linear-Time Decoder for High-Fidelity Medical Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02727](https://arxiv.org/pdf/2603.02727)**

> **作者:** Hongbo Zheng; Afshin Bozorgpour; Dorit Merhof; Minjia Zhang
>
> **摘要:** Medical image segmentation requires models that preserve fine anatomical boundaries while remaining efficient for clinical deployment. While transformers capture long-range dependencies, they suffer from quadratic attention cost and large data requirements, whereas CNNs are compute-friendly yet struggle with global reasoning. Linear attention offers $\mathcal{O}(N)$ scaling, but often exhibits training instability and attention dilution, yielding diffuse maps. We introduce PVT-GDLA, a decoder-centric Transformer that restores sharp, long-range dependencies at linear time. Its core, Gated Differential Linear Attention (GDLA), computes two kernelized attention paths on complementary query/key subspaces and subtracts them with a learnable, channel-wise scale to cancel common-mode noise and amplify relevant context. A lightweight, head-specific gate injects nonlinearity and input-adaptive sparsity, mitigating attention sink, and a parallel local token-mixing branch with depthwise convolution strengthens neighboring-token interactions, improving boundary fidelity, all while retaining $\mathcal{O}(N)$ complexity and low parameter overhead. Coupled with a pretrained Pyramid Vision Transformer (PVT) encoder, PVT-GDLA achieves state-of-the-art accuracy across CT, MRI, ultrasound, and dermoscopy benchmarks under equal training budgets, with comparable parameters but lower FLOPs than CNN-, Transformer-, hybrid-, and linear-attention baselines. PVT-GDLA provides a practical path to fast, scalable, high-fidelity medical segmentation in clinical environments and other resource-constrained settings.
>
---
#### [replaced 043] GS-2M: Material-aware Gaussian Splatting for High-fidelity Mesh Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22276](https://arxiv.org/pdf/2509.22276)**

> **作者:** Dinh Minh Nguyen; Malte Avenhaus; Thomas Lindemeier
>
> **备注:** This is the author's version of a paper accepted to Eurographics 2026, to appear in Computer Graphics Form. The final version will be available via Wiley
>
> **摘要:** We propose a material-aware optimization framework for high-fidelity mesh reconstruction from multi-view images based on 3D Gaussian Splatting, referred to as GS-2M. Previous works handle these tasks separately and struggle to reconstruct highly reflective surfaces, often relying on priors from external models to enhance the decomposition results. Conversely, our method addresses these two problems by jointly optimizing attributes relevant to the quality of rendered depth and normals, maintaining geometric details while being resilient to reflective surfaces. Although contemporary works effectively solve these tasks together, they often employ sophisticated neural components to learn scene properties, which hinders their performance at scale. To further eliminate these neural components, we propose a novel roughness supervision strategy based on multi-view photometric variation. When combined with a carefully designed loss and optimization process, our unified framework produces reconstruction results comparable to state-of-the-art methods, delivering accurate triangle meshes even for reflective surfaces. We validate the effectiveness of our approach with widely used datasets from previous works and qualitative comparisons with state-of-the-art surface reconstruction methods. Project page: this https URL.
>
---
#### [replaced 044] Scalable Aerial GNSS Localization for Marine Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于水下机器人定位任务，旨在解决传统GNSS在水面反射和高成本问题。通过使用搭载GNSS的无人机辅助定位，实现高效、可扩展的海洋机器人定位。**

- **链接: [https://arxiv.org/pdf/2505.04095](https://arxiv.org/pdf/2505.04095)**

> **作者:** Shuo Wen; Edwin Meriaux; Mariana Sosa Guzmán; Charlotte Morissette; Chloe Si; Bobak Baghi; Gregory Dudek
>
> **备注:** International Conference on Robotics and Automation 2025 Workshop Robots in the Wild
>
> **摘要:** Accurate localization is crucial for water robotics, yet traditional onboard Global Navigation Satellite System (GNSS) approaches are difficult or ineffective due to signal reflection on the water's surface and its high cost of aquatic GNSS receivers. Existing approaches, such as inertial navigation, Doppler Velocity Loggers (DVL), SLAM, and acoustic-based methods, face challenges like error accumulation and high computational complexity. Therefore, a more efficient and scalable solution remains necessary. This paper proposes an alternative approach that leverages an aerial drone equipped with GNSS localization to track and localize a marine robot once it is near the surface of the water. Our results show that this novel adaptation enables accurate single and multi-robot marine robot localization.
>
---
#### [replaced 045] PhysDrape: Learning Explicit Forces and Collision Constraints for Physically Realistic Garment Draping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.08020](https://arxiv.org/pdf/2602.08020)**

> **作者:** Minghai Chen; Mingyuan Liu; Ning Ma; Jianqing Li; Yuxiang Huan
>
> **摘要:** Deep learning-based garment draping has emerged as a promising alternative to traditional Physics-Based Simulation (PBS), yet robust collision handling remains a critical bottleneck. Most existing methods enforce physical validity through soft penalties, creating an intrinsic trade-off between geometric feasibility and physical plausibility: penalizing collisions often distorts mesh structure, while preserving shape leads to interpenetration. To resolve this conflict, we present PhysDrape, a hybrid neural-physical solver for physically realistic garment draping driven by explicit forces and constraints. Unlike soft-constrained frameworks, PhysDrape integrates neural inference with explicit geometric solvers in a fully differentiable pipeline. Specifically, we propose a Physics-Informed Graph Neural Network conditioned on a physics-enriched graph -- encoding material parameters and body proximity -- to predict residual displacements. Crucially, we integrate a differentiable two-stage solver: first, a learnable Force Solver iteratively resolves unbalanced forces derived from the Saint Venant-Kirchhoff (StVK) model to ensure quasi-static equilibrium; second, a Differentiable Projection strictly enforces collision constraints against the body surface. This differentiable design guarantees physical validity through explicit constraints, while enabling end-to-end learning to optimize the network for physically consistent predictions. Extensive experiments demonstrate that PhysDrape achieves state-of-the-art performance, ensuring negligible interpenetration with significantly lower strain energy compared to existing baselines, achieving superior physical fidelity and robustness in real-time.
>
---
#### [replaced 046] RoboLayout: Differentiable 3D Scene Generation for Embodied Agents
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出RoboLayout，解决3D场景生成中满足实体代理交互需求的问题。通过引入可达性约束和局部优化，提升场景可操作性与生成效率。**

- **链接: [https://arxiv.org/pdf/2603.05522](https://arxiv.org/pdf/2603.05522)**

> **作者:** Ali Shamsaddinlou
>
> **摘要:** Recent advances in vision language models (VLMs) have shown strong potential for spatial reasoning and 3D scene layout generation from open-ended language instructions. However, generating layouts that are not only semantically coherent but also feasible for interaction by embodied agents remains challenging, particularly in physically constrained indoor environments. In this paper, RoboLayout is introduced as an extension of LayoutVLM that augments the original framework with agent-aware reasoning and improved optimization stability. RoboLayout integrates explicit reachability constraints into a differentiable layout optimization process, enabling the generation of layouts that are navigable and actionable by embodied agents. Importantly, the agent abstraction is not limited to a specific robot platform and can represent diverse entities with distinct physical capabilities, such as service robots, warehouse robots, humans of different age groups, or animals, allowing environment design to be tailored to the intended agent. In addition, a local refinement stage is proposed that selectively reoptimizes problematic object placements while keeping the remainder of the scene fixed, improving convergence efficiency without increasing global optimization iterations. Overall, RoboLayout preserves the strong semantic alignment and physical plausibility of LayoutVLM while enhancing applicability to agent-centric indoor scene generation, as demonstrated by experimental results across diverse scene configurations.
>
---
#### [replaced 047] RDM: Recurrent Diffusion Model for Human Motion Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.07169](https://arxiv.org/pdf/2406.07169)**

> **作者:** Mirgahney Mohamed; Harry Jake Cunningham; Marc P. Deisenroth; Lourdes Agapito
>
> **备注:** v2: Major revision with extensive text polishing and structural updates. Added new experiments on the rollout effect, specifically analyzing the trade-offs between compute time and sequence length. Includes several new visualizations (Figures 6, 9, 10) and an expanded discussion in Section 4
>
> **摘要:** Human motion generation is a challenging task due to its high dimensionality and the difficulty of generating fine-grained motions. Diffusion methods have been proposed due to their high sample quality and expressiveness. Early approaches treat the entire sequence as a whole, which is computationally expensive and restricts sequence length. In contrast, autoregressive diffusion models generate longer sequences. However, their reliance on fully denoising previous frames complicates training and inference. Consequently, we propose \textit{RDM}, a new recurrent diffusion formulation similar to Recurrent Neural Networks (RNNs).RDMs explicitly condition diffusion processes on preceding noisy frames, avoiding the cost of full denoising. Nonetheless, maintaining its probabilistic nature is non-trivial. Therefore, we employ Normalizing Flows to model recurrent connections. Our evaluations demonstrate RDM's effectiveness: it achieves comparable performance to autoregressive baselines and generates long sequences that remain aligned with the text. RDM also skips diffusion steps during inference, significantly reducing computational cost.
>
---
#### [replaced 048] Goldilocks Test Sets for Face Verification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.15965](https://arxiv.org/pdf/2405.15965)**

> **作者:** Haiyu Wu; Sicong Tian; Aman Bhatta; Jacob Gutierrez; Grace Bezold; Genesis Argueta; Karl Ricanek Jr.; Michael C. King; Kevin W. Bowyer
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Reported face verification accuracy has reached a plateau on current well-known test sets. As a result, some difficult test sets have been assembled by reducing the image quality or adding artifacts to the image. However, we argue that test sets can be challenging without artificially reducing the image quality because the face recognition (FR) models suffer from correctly recognizing 1) the pairs from the same identity (i.e., genuine pairs) with a large face attribute difference, 2) the pairs from different identities (i.e., impostor pairs) with a small face attribute difference, and 3) the pairs of similar-looking identities (e.g., twins and relatives). We propose three challenging test sets to reveal important but ignored weaknesses of the existing FR algorithms. To challenge models on variation of facial attributes, we propose Hadrian and Eclipse to address facial hair differences and face exposure differences. The images in both test sets are high-quality and collected in a controlled environment. To challenge FR models on similar-looking persons, we propose ND-Twins, which contains images from a dedicated twins dataset. The LFW test protocol is used to structure the proposed test sets. Moreover, we introduce additional rules to assemble ``Goldilocks\footnote{this https URL}" level test sets, including 1) restricted number of occurrence of hard samples, 2) equal chance evaluation across demographic groups, and 3) constrained identity overlap across validation folds. Quantitatively, without further processing the images, the proposed test sets have on-par or higher difficulties than the existing test sets that add artifacts to the images. The datasets are available at: this https URL.
>
---
#### [replaced 049] VL-Nav: A Neuro-Symbolic Approach for Reasoning-based Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决机器人在复杂指令下自主导航的问题。提出VL-Nav系统，结合神经符号方法提升任务分解与探索效率。**

- **链接: [https://arxiv.org/pdf/2502.00931](https://arxiv.org/pdf/2502.00931)**

> **作者:** Yi Du; Taimeng Fu; Zhipeng Zhao; Shaoshu Su; Zitong Zhan; Zhuoqun Chen; Bowen Li; Chen Wang
>
> **摘要:** Navigating unseen, large-scale environments based on complex and abstract human instructions remains a formidable challenge for autonomous mobile robots. Addressing this requires robots to infer implicit semantics and efficiently explore large-scale task spaces. However, existing methods, ranging from end-to-end learning to foundation model-based modular architectures, often lack the capability to decompose complex tasks or employ efficient exploration strategies, leading to robot aimless wandering or target recognition failures. To address these limitations, we propose VL-Nav, a neuro-symbolic (NeSy) vision-language navigation system. The proposed system intertwines neural reasoning with symbolic guidance through two core components: (1) a NeSy task planner that leverages a symbolic 3D scene graph and image memory system to enhance the vision language models' (VLMs) neural reasoning capabilities for task decomposition and replanning; and (2) a NeSy exploration system that couples neural semantic cues with the symbolic heuristic function to efficiently gather the task-related information while minimizing unnecessary repeat travel during exploration. Validated on the DARPA TIAMAT Challenge navigation tasks, our system achieved an 83.4% success rate (SR) in indoor environments and 75% in outdoor scenarios. VL-Nav achieved an 86.3% SR in real-world experiments, including a challenging 483-meter run. Finally, we validate the system with complex instructions in a 3D multi-floor scenario.
>
---
#### [replaced 050] Generative Prior-Guided Neural Interface Reconstruction for 3D Electrical Impedance Tomography
- **分类: math.NA; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.16487](https://arxiv.org/pdf/2505.16487)**

> **作者:** Haibo Liu; Junqing Chen; Guang Lin
>
> **摘要:** Reconstructing complex 3D interfaces from indirect measurements remains a grand challenge in scientific computing, particularly for ill-posed inverse problems like Electrical Impedance Tomography (EIT). Traditional shape optimization struggles with topological changes and regularization tuning, while emerging deep learning approaches often compromise physical fidelity or require prohibitive amounts of paired training data. We present a transformative ``solver-in-the-loop'' framework that bridges this divide by coupling a pre-trained 3D generative prior with a rigorous boundary integral equation (BIE) solver. Unlike Physics-Informed Neural Networks (PINNs) that treat physics as soft constraints, our architecture enforces the governing elliptic PDE as a hard constraint at every optimization step, ensuring strict physical consistency. Simultaneously, we navigate a compact latent manifold of plausible geometries learned by a differentiable neural shape representation, effectively regularizing the ill-posed problem through data-driven priors rather than heuristic smoothing. By propagating adjoint shape derivatives directly through the neural decoder, we achieve fast, stable convergence with dramatically reduced degrees of freedom. Extensive experiments on 3D high-contrast EIT demonstrate that this principled hybrid approach yields superior geometric accuracy and data efficiency which is difficult to achieve using traditional methods, establishing a robust new paradigm for physics-constrained geometric discovery.
>
---
#### [replaced 051] Strengthening Generative Robot Policies through Predictive World Modeling
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出生成式预测控制（GPC），用于强化机器人策略。解决机器人控制中泛化能力不足的问题，通过生成式策略克隆、预测世界模型训练和在线规划优化，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2502.00622](https://arxiv.org/pdf/2502.00622)**

> **作者:** Han Qi; Haocheng Yin; Aris Zhu; Yilun Du; Heng Yang
>
> **备注:** Acceptance to RAL. Website: this https URL
>
> **摘要:** We present generative predictive control (GPC), a learning control framework that (i) clones a generative diffusion-based policy from expert demonstrations, (ii) trains a predictive action-conditioned world model from both expert demonstrations and random explorations, and (iii) synthesizes an online planner that ranks and optimizes the action proposals from (i) by looking ahead into the future using the world model from (ii). Across a variety of robotic manipulation tasks, we demonstrate that GPC consistently outperforms behavior cloning in both state-based and vision-based settings, in simulation and in the real world.
>
---
#### [replaced 052] Elytra: A Flexible Framework for Securing Large Vision Systems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00661](https://arxiv.org/pdf/2506.00661)**

> **作者:** Richard E. Neddo; Emmanuel Atindama; Zander W. Blasingame; Chen Liu
>
> **备注:** Updated pre-print. Under review
>
> **摘要:** Adversarial attacks have emerged as a critical threat to autonomous driving systems. These attacks exploit the underlying neural network, allowing small, almost invisible, perturbations to alter the behavior of such systems in potentially malicious ways, e.g., causing a traffic sign classification network to misclassify a stop sign as a speed limit sign. Prior work in hardening such systems against adversarial attacks has looked at fine-tuning of the system or adding additional pre-processing steps to the input pipeline. Such solutions either have a hard time generalizing, require knowledge of adversarial attacks during training, or are computationally undesirable. Instead, we propose a framework called ELYTRA to take insights for parameter-efficient fine-tuning and use low-rank adaptation (LoRA) to train a lightweight security patch (or patches), enabling us to dynamically patch large pre-existing vision systems as new vulnerabilities are discovered. We demonstrate that the ELYTRA framework can patch pre-trained large vision models to improve classification accuracy by up to 24.09% in the presence of adversarial examples.
>
---
#### [replaced 053] M3CAD: Towards Generic Cooperative Autonomous Driving Benchmark
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出M³CAD基准，用于通用协同自动驾驶研究，解决多车辆协同感知与通信效率问题，设计多模态数据集并提出多级融合方法。**

- **链接: [https://arxiv.org/pdf/2505.06746](https://arxiv.org/pdf/2505.06746)**

> **作者:** Morui Zhu; Yongqi Zhu; Yihao Zhu; Qi Chen; Deyuan Qu; Song Fu; Qing Yang
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** We introduce M$^3$CAD, a comprehensive benchmark designed to advance research in generic cooperative autonomous driving. M$^3$CAD comprises 204 sequences with 30,000 frames. Each sequence includes data from multiple vehicles and different types of sensors, e.g., LiDAR point clouds, RGB images, and GPS/IMU, supporting a variety of autonomous driving tasks, including object detection and tracking, mapping, motion forecasting, occupancy prediction, and path planning. This rich multimodal setup enables M$^3$CAD to support both single-vehicle and multi-vehicle cooperative autonomous driving research. To the best of our knowledge, M$^3$CAD is the most complete benchmark specifically designed for cooperative, multi-task autonomous driving research. To test its effectiveness, we use M$^3$CAD to evaluate both state-of-the-art single-vehicle and cooperative driving solutions, setting baseline performance results. Since most existing cooperative perception methods focus on merging features but often ignore network bandwidth requirements, we propose a new multi-level fusion approach which adaptively balances communication efficiency and perception accuracy based on the current network conditions. We release M$^3$CAD, along with the baseline models and evaluation results, to support the development of robust cooperative autonomous driving systems. All resources will be made publicly available on this https URL
>
---
#### [replaced 054] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19821](https://arxiv.org/pdf/2601.19821)**

> **作者:** Kun Li; Michael Ying Yang; Sami Sebastian Brandt
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Audio--Visual Question Answering (AVQA) is a challenging multimodal task that requires jointly reasoning over audio, visual, and textual information in a given video to answer natural language questions. Inspired by recent advances in Video QA, many existing AVQA approaches primarily focus on visual information processing, leveraging pre-trained models to extract object-level and motion-level representations. However, in those methods, the audio input is primarily treated as complementary to video analysis, and the textual question information contributes minimally to audio--visual understanding, as it is typically integrated only in the final stages of reasoning. To address these limitations, we propose a novel Query-guided Spatial--Temporal--Frequency (QSTar) interaction method, which effectively incorporates question-guided clues and exploits the distinctive frequency-domain characteristics of audio signals, alongside spatial and temporal perception, to enhance audio--visual understanding. Furthermore, we introduce a Query Context Reasoning (QCR) block inspired by prompting, which guides the model to focus more precisely on semantically relevant audio and visual features. Extensive experiments conducted on several AVQA benchmarks demonstrate the effectiveness of our proposed method, achieving significant performance improvements over existing Audio QA, Visual QA, Video QA, and AVQA approaches. The code and pretrained models will be released after publication.
>
---
#### [replaced 055] SAGE: Structure-Aware Generative Video Transitions between Diverse Clips
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.24667](https://arxiv.org/pdf/2510.24667)**

> **作者:** Mia Kan; Yilin Liu; Niloy Mitra
>
> **备注:** Project Website: this https URL
>
> **摘要:** Video transitions aim to synthesize intermediate frames between two clips, but naive approaches such as linear blending introduce artifacts that limit professional use or break temporal coherence. Traditional techniques (cross-fades, morphing, frame interpolation) and recent generative inbetweening methods can produce high-quality plausible intermediates, but they struggle with bridging diverse clips involving large temporal gaps or significant semantic differences, leaving a gap for content-aware and visually coherent transitions. We address this challenge by drawing on artistic workflows, distilling strategies such as aligning silhouettes and interpolating salient features to preserve structure and perceptual continuity. Building on these strategies, we propose SAGE (Structure-Aware Generative vidEo transitions) as a simple yet effective zeroshot approach that combines structural guidance, provided via line maps and motion flow, with generative synthesis, enabling smooth, motion-consistent transitions without fine-tuning. Extensive experiments and comparison with current alternatives, namely [FILM, TVG, DiffMorpher, VACE, GI], demonstrate that SAGE outperforms both classical and the latest generative baselines on quantitative metrics and user studies for producing transitions between diverse clips. The simple method effectively bypasses the need to acquire suitable training data, which is particularly difficult in our creative setting involving diverse clips. Code is available via the project page at this https URL.
>
---
#### [replaced 056] Differentiable Microscopy Designs an All Optical Phase Retrieval Microscope
- **分类: physics.optics; cs.CV; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2203.14944](https://arxiv.org/pdf/2203.14944)**

> **作者:** Kithmini Herath; Hasindu Kariyawasam; Ramith Hettiarachchi; Udith Haputhanthri; Dineth Jayakody; Raja N. Ahmad; Azeem Ahmad; Balpreet S. Ahluwalia; Chamira U. S. Edussooriya; Dushan N. Wadduwage
>
> **摘要:** Designing new optical systems from the ground up for microscopy imaging tasks such as phase retrieval, requires substantial scientific expertise and creativity. To augment the traditional design process, we propose differentiable microscopy ($\partial\mu$), which introduces a top-down design approach. Using all optical phase retrieval as an illustrative example, we demonstrate the effectiveness of data-driven microscopy design through $\partial\mu$. Furthermore, we conduct comprehensive comparisons with existing all-optical phase retrieval methods, showcasing the consistent superiority of our learned designs across multiple datasets, including biological samples. To substantiate our ideas, we experimentally validate the functionality of one of the learned designs, providing a proof of concept. The proposed differentiable microscopy framework supplements the creative process of designing new phase microscopy systems and may be extended to other similar applications in optical design.
>
---
#### [replaced 057] Prototype Perturbation for Relaxing Alignment Constraints in Backward-Compatible Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.14824](https://arxiv.org/pdf/2503.14824)**

> **作者:** Zikun Zhou; Yushuai Sun; Wenjie Pei; Xin Li; Yaowei Wang
>
> **备注:** Accept to IEEE TMM
>
> **摘要:** The traditional paradigm to update retrieval models requires re-computing the embeddings of the gallery data, a time-consuming and computationally intensive process known as backfilling. To circumvent backfilling, Backward-Compatible Learning (BCL) has been widely explored, which aims to train a new model compatible with the old one. Many previous works focus on effectively aligning the embeddings of the new model with those of the old one to enhance the backward-compatibility. Nevertheless, such strong alignment constraints would compromise the discriminative ability of the new model, particularly when different classes are closely clustered and hard to distinguish in the old feature space. To address this issue, we propose to relax the constraints by introducing perturbations to the old feature prototypes. This allows us to align the new feature space with a pseudo-old feature space defined by these perturbed prototypes, thereby preserving the discriminative ability of the new model in backward-compatible learning. We have developed two approaches for calculating the perturbations: Neighbor-Driven Prototype Perturbation (NDPP) and Optimization-Driven Prototype Perturbation (ODPP). Particularly, they take into account the feature distributions of not only the old but also the new models to obtain proper perturbations along with new model updating. Extensive experiments on the landmark and commodity datasets demonstrate that our approaches perform favorably against state-of-the-art BCL algorithms.
>
---
#### [replaced 058] OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05959](https://arxiv.org/pdf/2603.05959)**

> **作者:** Si-Yu Lu; Po-Ting Chen; Hui-Che Hsu; Sin-Ye Jhong; Wen-Huang Cheng; Yung-Yao Chen
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Reconstructing 3D geometry from streaming video requires continuous inference under bounded resources. Recent geometric foundation models achieve impressive reconstruction quality through all-to-all attention, yet their quadratic cost confines them to short, offline sequences. Causal-attention variants such as StreamVGGT enable single-pass streaming but accumulate an ever-growing KV cache, exhausting GPU memory within hundreds of frames and precluding the long-horizon deployment that motivates streaming inference in the first place. We present OVGGT, a training-free framework that bounds both memory and compute to a fixed budget regardless of sequence length. Our approach combines Self-Selective Caching, which leverages FFN residual magnitudes to compress the KV cache while remaining fully compatible with FlashAttention, with Dynamic Anchor Protection, which shields coordinate-critical tokens from eviction to suppress geometric drift over extended trajectories. Extensive experiments on indoor, outdoor, and ultra-long sequence benchmarks demonstrate that OVGGT processes arbitrarily long videos within a constant VRAM envelope while achieving state-of-the-art 3D geometric accuracy.
>
---
#### [replaced 059] Single Image, Any Face: Generalisable 3D Face Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.16990](https://arxiv.org/pdf/2409.16990)**

> **作者:** Wenqing Wang; Haosen Yang; Josef Kittler; Xiatian Zhu
>
> **备注:** Accepted by Pattern Recognition, March 2026
>
> **摘要:** The creation of 3D human face avatars from a single unconstrained image is a fundamental task that underlies numerous real-world vision and graphics applications. Despite the significant progress made in generative models, existing methods are either less suited in design for human faces or fail to generalise from the restrictive training domain to unconstrained facial images. To address these limitations, we propose a novel model, Gen3D-Face, which generates 3D human faces with unconstrained single image input within a multi-view consistent diffusion framework. Given a specific input image, our model first produces multi-view images, followed by neural surface construction. To incorporate face geometry information while preserving generalisation to in-the-wild inputs, we estimate a subject-specific mesh directly from the input image, enabling training and evaluation without ground-truth 3D supervision. Importantly, we introduce a multi-view joint generation scheme to enhance the appearance consistency among different views. To the best of our knowledge, this is the first attempt and benchmark for creating photorealistic 3D human face avatars from single images for generic human subject across domains. Extensive experiments demonstrate the efficacy and superiority of our method over previous alternatives for out-of-domain single image 3D face generation and the top ranking competition for the in-domain setting.
>
---
#### [replaced 060] FoldNet: Learning Generalizable Closed-Loop Policy for Garment Folding via Keypoint-Driven Asset and Demonstration Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对服装折叠任务，解决数据生成困难问题，提出合成数据集和关键点驱动的策略，提升机器人折叠成功率。**

- **链接: [https://arxiv.org/pdf/2505.09109](https://arxiv.org/pdf/2505.09109)**

> **作者:** Yuxing Chen; Bowen Xiao; He Wang
>
> **备注:** Project: this https URL
>
> **摘要:** Due to the deformability of garments, generating a large amount of high-quality data for robotic garment manipulation tasks is highly challenging. In this paper, we present a synthetic garment dataset that can be used for robotic garment folding. We begin by constructing geometric garment templates based on keypoints and applying generative models to generate realistic texture patterns. Leveraging these keypoint annotations, we generate folding demonstrations in simulation and train folding policies via closed-loop imitation learning. To improve robustness, we propose KG-DAgger, which uses a keypoint-based strategy to generate demonstration data for recovering from failures. KG-DAgger significantly improves the model performance, boosting the real-world success rate by 25\%. After training with 15K trajectories (about 2M image-action pairs), the model achieves a 75\% success rate in the real world. Experiments in both simulation and real-world settings validate the effectiveness of our proposed framework.
>
---
#### [replaced 061] Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.19195](https://arxiv.org/pdf/2510.19195)**

> **作者:** Kai Zeng; Zhanqian Wu; Kaixin Xiong; Xiaobao Wei; Xiangyu Guo; Zhenxin Zhu; Kalok Ho; Lijun Zhou; Bohan Zeng; Ming Lu; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Wentao Zhang
>
> **摘要:** Recent advancements in driving world models enable controllable generation of high-quality RGB videos or multimodal videos. Existing methods primarily focus on metrics related to generation quality and controllability. However, they often overlook the evaluation of downstream perception tasks, which are $\mathbf{really\ crucial}$ for the performance of autonomous driving. Existing methods usually leverage a training strategy that first pretrains on synthetic data and finetunes on real data, resulting in twice the epochs compared to the baseline (real data only). When we double the epochs in the baseline, the benefit of synthetic data becomes negligible. To thoroughly demonstrate the benefit of synthetic data, we introduce Dream4Drive, a novel synthetic data generation framework designed for enhancing the downstream perception tasks. Dream4Drive first decomposes the input video into several 3D-aware guidance maps and subsequently renders the 3D assets onto these guidance maps. Finally, the driving world model is fine-tuned to produce the edited, multi-view photorealistic videos, which can be used to train the downstream perception models. Dream4Drive enables unprecedented flexibility in generating multi-view corner cases at scale, significantly boosting corner case perception in autonomous driving. To facilitate future research, we also contribute a large-scale 3D asset dataset named DriveObj3D, covering the typical categories in driving scenarios and enabling diverse 3D-aware video editing. We conduct comprehensive experiments to show that Dream4Drive can effectively boost the performance of downstream perception models under various training epochs. Page: this https URL GitHub Link: this https URL
>
---
#### [replaced 062] Move What Matters: Parameter-Efficient Domain Adaptation via Optimal Transport Flow for Collaborative Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.11565](https://arxiv.org/pdf/2602.11565)**

> **作者:** Zesheng Jia; Jin Wang; Siao Liu; Lingzhi Li; Ziyao Huang; Yunjiang Xu; Jianping Wang
>
> **摘要:** Fast domain adaptation remains a fundamental challenge for deploying multi-agent systems across diverse environments in Vehicle-to-Everything (V2X) collaborative perception. Despite the success of Parameter-Efficient Fine-Tuning (PEFT) in natural language processing and conventional vision tasks, directly applying PEFT to multi-agent settings leads to significant performance degradation and training instability. In this work, we conduct a detailed analysis and identify two key factors: (i) inter-frame redundancy in heterogeneous sensory streams, and (ii) erosion of fine-grained semantics in deep-layer representations under PEFT adaptation. To address these issues, we propose FlowAdapt, a parameter-efficient framework grounded in optimal transport theory, which minimizes information transport costs across both data distributions and network hierarchies. Specifically, we introduce a Wasserstein Greedy Sampling strategy to selectively filter redundant samples via a bounded covering radius. Furthermore, Progressive Knowledge Transfer module is designed to progressively inject compressed early-stage representations into later stages through learnable pathways, alleviating semantic degradation in late-stage adaptation. Extensive experiments on three benchmarks demonstrate that FlowAdapt achieves state-of-the-art performance with only 1% of trainable parameters, effectively bridging domain gaps with superior sample efficiency and generalization.
>
---
#### [replaced 063] A Two-Stage Multitask Vision-Language Framework for Explainable Crop Disease Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于作物病害视觉问答任务，旨在提升作物和病害识别的准确性与可解释性。通过两阶段多任务框架，结合视觉与语言模型，实现高效且可解释的病害分析。**

- **链接: [https://arxiv.org/pdf/2601.05143](https://arxiv.org/pdf/2601.05143)**

> **作者:** Md. Zahid Hossain; Most. Sharmin Sultana Samu; Md. Rakibul Islam; Md. Siam Ansary
>
> **备注:** Preprint, manuscript is under review
>
> **摘要:** Visual question answering (VQA) for crop disease analysis requires accurate visual understanding and reliable language generation. In this work, we present a lightweight and explainable vision-language framework for crop and disease identification from leaf images. The proposed approach integrates a Swin Transformer vision encoder with sequence-to-sequence language decoders. The vision encoder is first trained in a multitask setup for both plant and disease classification, and then frozen while the text decoders are trained, forming a two-stage training strategy that enhances visual representation learning and cross-modal alignment. We evaluate the model on the large-scale Crop Disease Domain Multimodal (CDDM) dataset using both classification and natural language generation metrics. Experimental results demonstrate near-perfect recognition performance, achieving 99.94% plant classification accuracy and 99.06% disease classification accuracy, along with strong BLEU, ROUGE and BERTScore results. Without fine-tuning, the model further generalizes well to the external PlantVillageVQA benchmark, achieving 83.18% micro accuracy in the VQA task. Our lightweight design outperforms larger vision-language baselines while using significantly fewer parameters. Explainability is assessed through Grad-CAM and token-level attribution, providing interpretable visual and textual evidence for predictions. Qualitative results demonstrate robust performance under diverse user-driven queries, highlighting the effectiveness of task-specific visual pretraining and the two-stage training methodology for crop disease visual question answering. An interactive demo of the proposed Swin-T5 model is publicly available as a Gradio-based application at this https URL for community use.
>
---
#### [replaced 064] Efficient Vision Mamba for MRI Super-Resolution via Hybrid Selective Scanning
- **分类: cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2512.19676](https://arxiv.org/pdf/2512.19676)**

> **作者:** Mojtaba Safari; Shansong Wang; Vanessa L Wildman; Mingzhe Hu; Zach Eidex; Chih-Wei Chang; Erik H Middlebrooks; Richard L.J Qiu; Pretesh Patel; Ashesh B. Jani; Hui Mao; Zhen Tian; Xiaofeng Yang
>
> **摘要:** Background: High-resolution MRI is critical for diagnosis, but long acquisition times limit clinical use. Super-resolution (SR) can enhance resolution post-scan, yet existing deep learning methods face fidelity-efficiency trade-offs. Purpose: To develop a computationally efficient and accurate deep learning framework for MRI SR that preserves anatomical detail for clinical integration. Materials and Methods: We propose a novel SR framework combining multi-head selective state-space models (MHSSM) with a lightweight channel MLP. The model uses 2D patch extraction with hybrid scanning to capture long-range dependencies. Each MambaFormer block integrates MHSSM, depthwise convolutions, and gated channel mixing. Evaluation used 7T brain T1 MP2RAGE maps (n=142) and 1.5T prostate T2w MRI (n=334). Comparisons included Bicubic interpolation, GANs (CycleGAN, Pix2pix, SPSR), transformers (SwinIR), Mamba (MambaIR), and diffusion models (I2SB, Res-SRDiff). Results: Our model achieved superior performance with exceptional efficiency. For 7T brain data: SSIM=0.951+-0.021, PSNR=26.90+-1.41 dB, LPIPS=0.076+-0.022, GMSD=0.083+-0.017, significantly outperforming all baselines (p<0.001). For prostate data: SSIM=0.770+-0.049, PSNR=27.15+-2.19 dB, LPIPS=0.190+-0.095, GMSD=0.087+-0.013. The framework used only 0.9M parameters and 57 GFLOPs, reducing parameters by 99.8% and computation by 97.5% versus Res-SRDiff, while outperforming SwinIR and MambaIR in accuracy and efficiency. Conclusion: The proposed framework provides an efficient, accurate MRI SR solution, delivering enhanced anatomical detail across datasets. Its low computational demand and state-of-the-art performance show strong potential for clinical translation.
>
---
#### [replaced 065] Tokenizing Semantic Segmentation with RLE
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21627](https://arxiv.org/pdf/2602.21627)**

> **作者:** Abhineet Singh; Justin Rozeboom; Nilanjan Ray
>
> **备注:** Code and models available at: this https URL
>
> **摘要:** This paper presents a new unified approach to semantic segmentation in both images and videos by using language modeling to output the masks as sequences of discrete tokens. We use run length encoding (RLE) to discretize the segmentation masks and then train a modified version of Pix2Seq to output these RLE tokens through autoregression. We propose novel tokenization strategies to compress the length of the token sequence to make it practicable to extend this approach to videos. We also show how instance information can be incorporated into the tokenization process to perform panoptic segmentation. We evaluate our proposed models on two datasets to show that they are competitive with the state of the art in some scenarios in spite of being bottlenecked by our limited computational resources. We make our code and models publicly available to facilitate further work in this domain.
>
---
#### [replaced 066] Learning to Think Fast and Slow for Visual Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16670](https://arxiv.org/pdf/2511.16670)**

> **作者:** Chenyu Lin; Cheng Chi; Jinlin Wu; Sharon Li; Kaiyang Zhou
>
> **摘要:** When faced with complex problems, we tend to engage in slower, more deliberate thinking. In contrast, for simple questions we give quick, intuitive responses. This dual-system thinking approach allows us to allocate cognitive resources efficiently, reserving deeper analytical effort for tasks that truly require it. However, existing reasoning-oriented visual language models (VLMs) are mostly trained to generate uniformly long reasoning, leading to substantial token waste when concise answers would suffice. In this paper, we observe that pre-trained, general-purpose VLMs manifest variations in response length for different question types, e.g., longer reasoning for math questions while shorter on perception problems. Different from existing work that overrides this prior by stimulating long reasoning without considering the problem complexity, we propose to leverage this prior to develop an explicit dual-mode thinking mechanism. Specifically, we anchor each training instance to either a fast or slow thinking prefix consistent with the model's natural response length tendency. Then, GRPO is adapted to learning dual-system thinking, enabling both autonomous and manual thinking mode selection. Extensive experiments across a wide variety of visual reasoning benchmarks demonstrate that our model, named DualMindVLM, significantly outperforms the base model and achieves state-of-the-art reasoning performance while maintaining high token efficiency.
>
---
#### [replaced 067] MTVCraft: Tokenizing 4D Motion for Arbitrary Character Animation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10238](https://arxiv.org/pdf/2505.10238)**

> **作者:** Yanbo Ding; Xirui Hu; Zhizhi Guo; Yan Zhang; Xinrui Wang; Zhixiang He; Chi Zhang; Yali Wang; Xuelong Li
>
> **摘要:** Character image animation has rapidly advanced with the rise of digital humans. However, existing methods rely largely on 2D-rendered pose images for motion guidance, which limits generalization and discards essential 4D information for open-world animation. To address this, we propose MTVCraft (Motion Tokenization Video Crafter), the first framework that directly models raw 3D motion sequences (i.e., 4D motion) for character image animation. Specifically, we introduce 4DMoT (4D motion tokenizer) to quantize 3D motion sequences into 4D motion tokens. Compared to 2D-rendered pose images, 4D motion tokens offer more robust spatial-temporal cues and avoid strict pixel-level alignment between pose images and the character, enabling more flexible and disentangled control. Next, we introduce MV-DiT (Motion-aware Video DiT). By designing unique motion attention with 4D positional encodings, MV-DiT can effectively leverage motion tokens as 4D compact yet expressive context for character image animation in the complex 4D world. We implement MTVCraft on both CogVideoX-5B (small scale) and Wan-2.1-14B (large scale), demonstrating that our framework is easily scalable and can be applied to models of varying sizes. Experiments on the TikTok and Fashion benchmarks demonstrate our state-of-the-art performance. Moreover, powered by robust motion tokens, MTVCraft showcases unparalleled zero-shot generalization. It can animate arbitrary characters in full-body and half-body forms, and even non-human objects across diverse styles and scenarios. Hence, it marks a significant step forward in this field and opens a new direction for pose-guided video generation. Our project page is available at this https URL. A scaled version has been commercially deployed and is available at this https URL.
>
---
#### [replaced 068] Self-Attention And Beyond the Infinite: Towards Linear Transformers with Infinite Self-Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00175](https://arxiv.org/pdf/2603.00175)**

> **作者:** Giorgio Roffo; Luke Palmer
>
> **备注:** This work builds in part on conceptual directions previously explored in the MVL/Toyota Motor Europe collaboration
>
> **摘要:** The quadratic cost of softmax attention limits Transformer scalability in high-resolution vision. We introduce Infinite Self-Attention (InfSA), a spectral reformulation that treats each attention layer as a diffusion step on a content-adaptive token graph, accumulating multi-hop interactions through a discounted Neumann series over attention matrices. This links self-attention to classical graph centrality (Katz, PageRank, eigenvector centrality) for interpretable token weighting. We also show the Neumann kernel equals the fundamental matrix of an absorbing Markov chain, so a token's centrality is its expected number of random-walk visits before absorption. We then propose Linear-InfSA, a linear-time variant that approximates the principal eigenvector of the implicit attention operator without forming the full attention matrix. It keeps an auxiliary state of fixed size proportional to per-head dimension dh (independent of sequence length N), is drop-in compatible with Vision Transformers, and supports stable training at 4096 by 4096 and inference at 9216 by 9216 (about 332k tokens). In a 4-layer ViT (53.5M parameters, 59 GFLOPs at 224 by 224), Linear-InfSA reaches 84.7% top-1 on ImageNet-1K, a +3.2 point architectural gain over an equal-depth softmax ViT trained with the same recipe. On ImageNet-V2, InfViT variants outperform all compared baselines (up to 79.8% vs 76.8%), indicating robustness under distribution shift. On an A100 40GB GPU, Linear-InfViT runs at 231 images/s and 0.87 J/image (13x better throughput and energy than equal-depth ViT) and is the only tested model to complete 9216 by 9216 inference without out-of-memory. The linear approximation closely matches the dominant eigenvector of the quadratic operator (cosine 0.985).
>
---
#### [replaced 069] MAViD: A Multimodal Framework for Audio-Visual Dialogue Understanding and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03034](https://arxiv.org/pdf/2512.03034)**

> **作者:** Youxin Pang; Jiajun Liu; Lingfeng Tan; Yong Zhang; Feng Gao; Xiang Deng; Zhuoliang Kang; Xiaoming Wei; Yebin Liu
>
> **摘要:** We propose MAViD, a novel Multimodal framework for Audio-Visual Dialogue understanding and generation. Existing approaches primarily focus on non-interactive systems and are limited to producing constrained and unnatural human speech. The primary challenge of this task lies in effectively integrating understanding and generation capabilities, as well as achieving seamless multimodal audio-video fusion. To solve these problems, we propose a Conductor-Creator architecture that divides the dialogue system into two primary components. The Conductor is tasked with understanding, reasoning, and generating instructions by breaking them down into motion and speech components, thereby enabling fine-grained control over interactions. The Creator then delivers interactive responses based on these instructions. Furthermore, to address the difficulty of generating long videos with consistent identity, timbre, and tone using dual DiT structures, the Creator adopts a structure that combines autoregressive (AR) and diffusion models. The AR model is responsible for audio generation, while the diffusion model ensures high-quality video generation. Additionally, we propose a novel fusion module to enhance connections between contextually consecutive clips and modalities, enabling synchronized long-duration audio-visual content generation. Extensive experiments demonstrate that our framework can generate vivid and contextually coherent long-duration dialogue interactions and accurately interpret users' multimodal queries.
>
---
#### [replaced 070] Video-EM: Event-Centric Episodic Memory for Long-Form Video Understanding
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2508.09486](https://arxiv.org/pdf/2508.09486)**

> **作者:** Yun Wang; Long Zhang; Jingren Liu; Jiaqi Yan; Zhanjie Zhang; Jiahao Zheng; Ao Ma; Run Ling; Xun Yang; Dapeng Wu; Xiangyu Chen; Xuelong Li
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Video Large Language Models (Video-LLMs) have shown strong video understanding, yet their application to long-form videos remains constrained by limited context windows. A common workaround is to compress long videos into a handful of representative frames via retrieval or summarization. However, most existing pipelines score frames in isolation, implicitly assuming that frame-level saliency is sufficient for downstream reasoning. This often yields redundant selections, fragmented temporal evidence, and weakened narrative grounding for long-form video question answering. We present \textbf{Video-EM}, a training-free, event-centric episodic memory framework that reframes long-form VideoQA as \emph{episodic event construction} followed by \emph{memory refinement}. Instead of treating retrieved keyframes as independent visuals, Video-EM employs an LLM as an active memory agent to orchestrate off-the-shelf tools: it first localizes query-relevant moments via multi-grained semantic matching, then groups and segments them into temporally coherent events, and finally encodes each event as a grounded episodic memory with explicit temporal indices and spatio-temporal cues (capturing \emph{when}, \emph{where}, \emph{what}, and involved entities). To further suppress verbosity and noise from imperfect upstream signals, Video-EM integrates a reasoning-driven self-reflection loop that iteratively verifies evidence sufficiency and cross-event consistency, removes redundancy, and adaptively adjusts event granularity. The outcome is a compact yet reliable \emph{event timeline} -- a minimal but sufficient episodic memory set that can be directly consumed by existing Video-LLMs without additional training or architectural changes.
>
---
#### [replaced 071] Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI科学探索任务，旨在开发自主AI科学家系统Jr. AI Scientist，解决AI在科研中的可靠性与风险问题。工作包括构建模拟研究流程的系统，并评估其生成论文的质量与风险。**

- **链接: [https://arxiv.org/pdf/2511.04583](https://arxiv.org/pdf/2511.04583)**

> **作者:** Atsuyuki Miyai; Mashiro Toyooka; Takashi Otonari; Zaiying Zhao; Kiyoharu Aizawa
>
> **备注:** TMLR2026. Issues, comments, and questions are all welcome in this https URL
>
> **摘要:** Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, validates them through rigorous experimentation, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel algorithms. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores by DeepReviewer than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.
>
---
#### [replaced 072] BEV-Patch-PF: Particle Filtering with BEV-Aerial Feature Matching for Off-Road Geo-Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于定位任务，解决无GPS的越野地理定位问题。通过融合BEV特征与航拍图匹配，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.15111](https://arxiv.org/pdf/2512.15111)**

> **作者:** Dongmyeong Lee; Jesse Quattrociocchi; Christian Ellis; Rwik Rana; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **摘要:** We propose BEV-Patch-PF, a GPS-free sequential geo-localization system that integrates a particle filter with learned bird's-eye-view (BEV) and aerial feature maps. From onboard RGB and depth images, we construct a BEV feature map. For each 3-DoF particle pose hypothesis, we crop the corresponding patch from an aerial feature map computed from a local aerial image queried around the approximate location. BEV-Patch-PF computes a per-particle log-likelihood by matching the BEV feature to the aerial patch feature. On two real-world off-road datasets, our method achieves 9.7x lower absolute trajectory error (ATE) on seen routes and 6.6x lower ATE on unseen routes than a retrieval-based baseline, while maintaining accuracy under dense canopy and shadow. The system runs in real time at 10 Hz on an NVIDIA Tesla T4, enabling practical robot deployment.
>
---
#### [replaced 073] Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.24232](https://arxiv.org/pdf/2510.24232)**

> **作者:** Qing Zhao; Weijian Deng; Pengxu Wei; ZiYi Dong; Hannan Lu; Xiangyang Ji; Liang Lin
>
> **备注:** NeurIPS 2025
>
> **摘要:** To improve detection robustness in adverse conditions (e.g., haze and low light), image restoration is commonly applied as a pre-processing step to enhance image quality for the detector. However, the functional mismatch between restoration and detection networks can introduce instability and hinder effective integration -- an issue that remains underexplored. We revisit this limitation through the lens of Lipschitz continuity, analyzing the functional differences between restoration and detection networks in both the input space and the parameter space. Our analysis shows that restoration networks perform smooth, continuous transformations, while object detectors operate with discontinuous decision boundaries, making them highly sensitive to minor perturbations. This mismatch introduces instability in traditional cascade frameworks, where even imperceptible noise from restoration is amplified during detection, disrupting gradient flow and hindering optimization. To address this, we propose Lipschitz-regularized object detection (LROD), a simple yet effective framework that integrates image restoration directly into the detector's feature learning, harmonizing the Lipschitz continuity of both tasks during training. We implement this framework as Lipschitz-regularized YOLO (LR-YOLO), extending seamlessly to existing YOLO detectors. Extensive experiments on haze and low-light benchmarks demonstrate that LR-YOLO consistently improves detection stability, optimization smoothness, and overall accuracy.
>
---
#### [replaced 074] Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.10416](https://arxiv.org/pdf/2512.10416)**

> **作者:** Wenfei Guan; Jilin Mei; Tong Shen; Xumin Wu; Shuo Wang; Chen Min; Yu Hu
>
> **备注:** This revision improves clarity and consistency throughout the paper. We refine terminology to more precisely describe the vertex extraction optimization, add motivational context to the edge feature encoding section, and clarify the overall inference pipeline. We also add an Acknowledgments section
>
> **摘要:** Deep learning has advanced vectorized road extraction in urban settings, yet off-road environments remain underexplored and challenging. A significant domain gap causes advanced models to fail in wild terrains due to two key issues: lack of large-scale vectorized datasets and structural weakness in prevailing methods. Models such as SAM-Road employ a node-centric paradigm that reasons at sparse endpoints, making them fragile to occlusions and ambiguous junctions in off-road scenes, leading to topological errors. This work addresses these limitations in two complementary ways. First, we release WildRoad, a global off-road road network dataset constructed efficiently with a dedicated interactive annotation tool tailored for road-network labeling. Second, we introduce MaGRoad (Mask-aware Geodesic Road network extractor), a path-centric framework that aggregates multi-scale visual evidence along candidate paths to infer connectivity robustly. Extensive experiments show that MaGRoad achieves state-of-the-art performance on our challenging WildRoad benchmark while generalizing well to urban datasets. An efficient vertex extraction strategy also yields roughly 2.5X faster inference, improving practical applicability. Together, the dataset and path-centric paradigm provide a stronger foundation for mapping roads in the wild. We release both the dataset and code at this repository. We release both the dataset and code at this https URL.
>
---
#### [replaced 075] ClearDepth: Enhanced Stereo Perception of Transparent Objects for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人感知任务，旨在解决透明物体深度感知难题。通过视觉Transformer和特征后融合模块提升深度恢复精度，并利用Sim2Real仿真生成数据，提高实际应用效果。**

- **链接: [https://arxiv.org/pdf/2409.08926](https://arxiv.org/pdf/2409.08926)**

> **作者:** Kaixin Bai; Huajian Zeng; Lei Zhang; Yiwen Liu; Hongli Xu; Zhaopeng Chen; Jianwei Zhang
>
> **备注:** 9 pages
>
> **摘要:** Transparent object depth perception poses a challenge in everyday life and logistics, primarily due to the inability of standard 3D sensors to accurately capture depth on transparent or reflective surfaces. This limitation significantly affects depth map and point cloud-reliant applications, especially in robotic manipulation. We developed a vision transformer-based algorithm for stereo depth recovery of transparent objects. This approach is complemented by an innovative feature post-fusion module, which enhances the accuracy of depth recovery by structural features in images. To address the high costs associated with dataset collection for stereo camera-based perception of transparent objects, our method incorporates a parameter-aligned, domain-adaptive, and physically realistic Sim2Real simulation for efficient data generation, accelerated by AI algorithm. Our experimental results demonstrate the model's exceptional Sim2Real generalizability in real-world scenarios, enabling precise depth mapping of transparent objects to assist in robotic manipulation. Project details are available at this https URL .
>
---
#### [replaced 076] ODI-Bench: Can MLLMs Understand Immersive Omnidirectional Environments?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11549](https://arxiv.org/pdf/2510.11549)**

> **作者:** Liu Yang; Huiyu Duan; Ran Tao; Juntao Cheng; Sijing Wu; Yunhao Li; Jing Liu; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** Omnidirectional images (ODIs) provide full 360x180 view which are widely adopted in VR, AR and embodied intelligence applications. While multi-modal large language models (MLLMs) have demonstrated remarkable performance on conventional 2D image and video understanding benchmarks, their ability to comprehend the immersive environments captured by ODIs remains largely unexplored. To address this gap, we first present ODI-Bench, a novel comprehensive benchmark specifically designed for omnidirectional image understanding. ODI-Bench contains 2,000 high-quality omnidirectional images and over 4,000 manually annotated question-answering (QA) pairs across 10 fine-grained tasks, covering both general-level and spatial-level ODI understanding. Extensive experiments are conducted to benchmark 20 representative MLLMs, including proprietary and open-source models, under both close-ended and open-ended settings. Experimental results reveal that current MLLMs still struggle to capture the immersive context provided by ODIs. To this end, we further introduce Omni-CoT, a training-free method which significantly enhances MLLMs' comprehension ability in the omnidirectional environment through chain-of-thought reasoning across both textual information and visual cues. Both the benchmark and the code will be released at this https URL.
>
---
#### [replaced 077] TransUNet-GradCAM: A Hybrid Transformer-U-Net with Self-Attention and Explainable Visualizations for Foot Ulcer Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03758](https://arxiv.org/pdf/2508.03758)**

> **作者:** Akwasi Asare; Mary Sagoe; Justice Williams Asare; Stephen Edward Moore
>
> **摘要:** Automated segmentation of diabetic foot ulcers (DFUs) plays a critical role in clinical diagnosis, therapeutic planning, and longitudinal wound monitoring. However, this task remains challenging due to the heterogeneous appearance, irregular morphology, and complex backgrounds associated with ulcer regions in clinical photographs. Traditional convolutional neural networks (CNNs), such as U-Net, provide strong localization capabilities but struggle to model long-range spatial dependencies due to their inherently limited receptive fields. To address this, we employ the TransUNet architecture, a hybrid framework that integrates the global attention mechanism of Vision Transformers (ViTs) into the U-Net structure. This combination allows the model to extract global contextual features while maintaining fine-grained spatial resolution. We trained the model on the public Foot Ulcer Segmentation Challenge (FUSeg) dataset using a robust augmentation pipeline and a hybrid loss function to mitigate class imbalance. On the internal validation set, the model achieved a Dice Similarity Coefficient (F1-score) of 0.8886 using an optimized threshold of 0.4843. Crucially, to assess generalizability, we performed external validation on two independent datasets: the AZH Wound Care Center dataset (n=278) and the Medetec dataset (n=152). Without any retraining, the model achieved Dice scores of 0.6209 and 0.7850, respectively, demonstrating robust zero-shot transferability to unseen clinical domains. Furthermore, clinical utility analysis revealed a strong correlation (Pearson r = 0.9749) between predicted and ground-truth wound areas. These outcomes demonstrate that our approach effectively integrates global and local feature extraction, offering a reliable, effective, and explainable solution for automated foot ulcer assessment.
>
---
#### [replaced 078] CR-QAT: Curriculum Relational Quantization-Aware Training for Open-Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05964](https://arxiv.org/pdf/2603.05964)**

> **作者:** Jinyeong Park; Donghwa Kang; Brent ByungHoon Kang; Hyeongboo Baek; Jibum Kim
>
> **摘要:** Open-vocabulary object detection (OVOD) enables novel category detection via vision-language alignment, but massive model sizes hinder deployment on resource-constrained devices. While quantization offers practical compression, we reveal that naive extreme low-bit (e.g., 4-bit) quantization severely degrades fine-grained vision-language alignment and distorts inter-region relational structures. To address this, we propose curriculum relational quantization-aware training (CR-QAT), an integrated framework combining stage-by-stage optimization with relational knowledge distillation. Within CR-QAT, curriculum QAT (CQAT) mitigates error accumulation by partitioning the model for progressive quantization, ensuring stable optimization via error isolation. Concurrently, text-centric relational KD (TRKD) is applied to task-relevant modules. By constructing text-anchored pairwise similarity matrices, TRKD comprehensively transfers the teacher's multi-dimensional relational knowledge. Experiments on LVIS and COCO zero-shot benchmarks demonstrate that CR-QAT consistently outperforms existing QAT baselines under aggressive low-bit settings, achieving relative AP improvements of up to 38.9% and 40.9%, respectively.
>
---
#### [replaced 079] Deepfake Generation and Detection: A Benchmark and Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.17881](https://arxiv.org/pdf/2403.17881)**

> **作者:** Gan Pei; Jiangning Zhang; Menghan Hu; Zhenyu Zhang; Chengjie Wang; Yunsheng Wu; Guangtao Zhai; Jian Yang; Dacheng Tao
>
> **备注:** This paper has been accepted by ACM Computing Surveys. We closely follow the latest developments in this \href{this https URL}{project}
>
> **摘要:** Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.
>
---
#### [replaced 080] Multi-modal, Multi-task, Multi-criteria Automatic Evaluation with Vision Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决VLM生成文本的评估问题。提出HarmonicEval评估方法，适应多任务场景，并构建MMHE基准测试。**

- **链接: [https://arxiv.org/pdf/2412.14613](https://arxiv.org/pdf/2412.14613)**

> **作者:** Masanari Ohi; Masahiro Kaneko; Naoaki Okazaki; Nakamasa Inoue
>
> **摘要:** Vision-language models (VLMs) have shown impressive abilities across a range of multi-modal tasks. However, existing metrics for evaluating the quality of text generated by VLMs typically focus on an overall evaluation for a specific task, such as image captioning. While the overall evaluation is essential for any task, the criteria prioritized can differ depending on the task, making it challenging for current metrics to adapt to multi-task scenarios. To address this limitation, we propose HarmonicEval, a reference-free comprehensive evaluation metric that aggregates criterion-wise scores to produce the overall score in a bottom-up manner. Furthermore, to assess the generalizability of automatic evaluation metrics in multi-task scenarios, we construct the Multi-task Multi-criteria Human Evaluation (MMHE) benchmark, which comprises 18,000 expert human judgments across four multi-modal tasks. Our experiments demonstrate that HarmonicEval achieves higher correlations with human judgments than conventional metrics while providing numerical scores for each criterion. Project page: this https URL
>
---
#### [replaced 081] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06034](https://arxiv.org/pdf/2603.06034)**

> **作者:** Chunjiang Li; Jianbo Ma; Li Shen; Yanru Chen; Liangyin Chen
>
> **备注:** Accepted to CVPR 2026. [The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 (CVPR2026)]
>
> **摘要:** Multi-object tracking (MOT) involves analyzing object trajectories and counting the number of objects in video sequences. However, 2D MOT faces challenges due to positional cost confusion arising from partial occlusion. To address this issue, we present the novel Occlusion-Aware SORT (OA-SORT) framework, a plug-and-play and training-free framework that includes the Occlusion-Aware Module (OAM), the Occlusion-Aware Offset (OAO), and the Bias-Aware Momentum (BAM). Specifically, OAM analyzes the occlusion status of objects, where a Gaussian Map (GM) is introduced to reduce background influence. In contrast, OAO and BAM leverage the OAM-described occlusion status to mitigate cost confusion and suppress estimation instability. Comprehensive evaluations on the DanceTrack, SportsMOT, and MOT17 datasets demonstrate the importance of occlusion handling in MOT. On the DanceTrack test set, OA-SORT achieves 63.1% and 64.2% in HOTA and IDF1, respectively. Furthermore, integrating the Occlusion-Aware framework into the four additional trackers improves HOTA and IDF1 by an average of 2.08% and 3.05%, demonstrating the reusability of the occlusion awareness.
>
---
#### [replaced 082] MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06830](https://arxiv.org/pdf/2511.06830)**

> **作者:** Tianang Chen; Jian Jin; Shilv Cai; Zhuangzi Li; Weisi Lin
>
> **备注:** ICASSP 2026
>
> **摘要:** Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and code are available at this https URL.
>
---
#### [replaced 083] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22013](https://arxiv.org/pdf/2602.22013)**

> **作者:** I-Hsiang Chen; Yu-Wei Liu; Tse-Yu Wu; Yu-Chien Chiang; Jen-Chien Yang; Wei-Ting Chen
>
> **备注:** Accepted by CVPR2026; Project Page: this https URL
>
> **摘要:** Vision-based Retrieval-Augmented Generation (VisRAG) leverages vision-language models (VLMs) to jointly retrieve relevant visual documents and generate grounded answers based on multimodal evidence. However, existing VisRAG models degrade in performance when visual inputs suffer from distortions such as blur, noise, low light, or shadow, where semantic and degradation factors become entangled within pretrained visual encoders, leading to errors in both retrieval and generation stages. To address this limitation, we introduce RobustVisRAG, a causality-guided dual-path framework that improves VisRAG robustness while preserving efficiency and zero-shot generalization. RobustVisRAG uses a non-causal path to capture degradation signals through unidirectional attention and a causal path to learn purified semantics guided by these signals. Together with the proposed Non-Causal Distortion Modeling and Causal Semantic Alignment objectives, the framework enforces a clear separation between semantics and degradations, enabling stable retrieval and generation under challenging visual conditions. To evaluate robustness under realistic conditions, we introduce the Distortion-VisRAG dataset, a large-scale benchmark containing both synthetic and real-world degraded documents across seven domains, with 12 synthetic and 5 real distortion types that comprehensively reflect practical visual degradations. Experimental results show that RobustVisRAG improves retrieval, generation, and end-to-end performance by 7.35%, 6.35%, and 12.40%, respectively, on real-world degradations, while maintaining comparable accuracy on clean inputs.
>
---
#### [replaced 084] ReViP: Mitigating False Completion in Vision-Language-Action Models with Vision-Proprioception Rebalance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决VLA模型中的错误完成问题。通过引入ReViP框架和False-Completion基准，提升模型在扰动下的视觉感知与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.16667](https://arxiv.org/pdf/2601.16667)**

> **作者:** Zhuohao Li; Yinghao Li; Jian-Jian Jiang; Lang Zhou; Tianyu Zhang; Jiadong Yin; Mu Lin; Yi-Kin Wei; Wei-Shi Zheng
>
> **摘要:** Vision-Language-Action (VLA) models have advanced robotic manipulation by combining vision, language, and proprioception to predict actions. However, previous methods fuse proprioceptive signals directly with vision-language features, resulting in state-dominant bias and \textbf{false completions} despite visible execution failures. We systematically analyze this failure mode, attributing it to modality imbalance, where policies overly rely on internal state progression and underuse visual evidence. To address this, we introduce the first \textbf{False-Completion Benchmark Suite}, featuring eight tasks with three controlled perturbations (\emph{Object Drop}, \emph{Distractor Swap}, \emph{Relayout}) to comprehensively evaluate false completion. Moreover, we propose \textbf{ReViP}, a novel VLA framework with \textbf{Vi}sion-\textbf{P}roprioception \textbf{Re}balance to enhance visual grounding and robustness under perturbations. The key insight is to introduce auxiliary \emph{progress-aware visual cues} to adaptively modulate the coupling between semantic perception and proprioceptive dynamics. Specifically, progress-aware visual cues are extracted by an external Task-Stage Observer, which performs task-relevant reasoning on real-time observations to drive task-stage feature-wise linear modulation, enhancing environmental awareness and mitigating state-driven errors. Extensive experiments show that ReViP effectively mitigates false completion and improves success rates over strong VLA baselines, achieving a \textbf{26\%} gain over $\pi_0$ model on our suite, with gains extending to LIBERO, RoboTwin 2.0, and real-world evaluations.
>
---
#### [replaced 085] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15938](https://arxiv.org/pdf/2512.15938)**

> **作者:** Vegard Flovik
>
> **备注:** Accepted to ICLR 2026, Trustworthy AI Workshop
>
> **摘要:** Deep neural networks achieve impressive performance but remain difficult to interpret and control. We present SALVE (Sparse Autoencoder-Latent Vector Editing), a unified "discover, validate, and control" framework that bridges mechanistic interpretability and model editing. Using an $\ell_1$-regularized autoencoder, we learn a sparse, model-native feature basis without supervision. We validate these features with Grad-FAM, a feature-level saliency mapping method that visually grounds latent features in input data. Leveraging the autoencoder's structure, we perform precise and permanent weight-space interventions, enabling continuous modulation of both class-defining and cross-class features. We further derive a critical suppression threshold, $\alpha_{crit}$, quantifying each class's reliance on its dominant feature, supporting fine-grained robustness diagnostics. Our approach is validated on both convolutional (ResNet-18) and transformer-based (ViT-B/16) models, demonstrating consistent, interpretable control over their behavior. This work contributes a principled methodology for turning feature discovery into actionable model edits, advancing the development of transparent and controllable AI systems.
>
---
#### [replaced 086] Test-Time Modification: Inverse Domain Transformation for Robust Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13454](https://arxiv.org/pdf/2512.13454)**

> **作者:** Arpit Jadon; Joshua Niemeijer; Yuki M. Asano
>
> **备注:** Preprint
>
> **摘要:** Generative foundation models contain broad visual knowledge and can produce diverse image variations, making them particularly promising for advancing domain generalization tasks. They can be used for training data augmentation, but synthesizing comprehensive target-domain variations remains slow, expensive, and incomplete. We propose an alternative: using diffusion models at test time to map target images back to the source distribution where the downstream model was trained. This approach requires only a source domain description, preserves the task model, and eliminates large-scale synthetic data generation. We demonstrate consistent improvements across segmentation, detection, and classification tasks under challenging environmental shifts in real-to-real domain generalization scenarios with unknown target distributions. Our analysis spans multiple generative and downstream models, including an ensemble variant for enhanced robustness. The method improves BDD100K-Night-Det mAP@50 from 10.2 to 31.8, ImageNet-R top-1 from 36.1 to 60.8, and DarkZurich mIoU from 28.6 to 46.3.
>
---
#### [replaced 087] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24099](https://arxiv.org/pdf/2509.24099)**

> **作者:** Prerit Gupta; Shourya Verma; Ananth Grama; Aniket Bera
>
> **备注:** Under review at ICLR 2026
>
> **摘要:** Generating realistic, context-aware two-person motion conditioned on diverse modalities remains a fundamental challenge for graphics, animation and embodied AI systems. Real-world applications such as VR/AR companions, social robotics and game agents require models capable of producing coordinated interpersonal behaviour while flexibly switching between interactive and reactive generation. We introduce DualFlow, the first unified and efficient framework for multi-modal two-person motion generation. DualFlow conditions 3D motion generation on diverse inputs, including text, music, and prior motion sequences. Leveraging rectified flow, it achieves deterministic straight-line sampling paths between noise and data, reducing inference time and mitigating error accumulation common in diffusion-based models. To enhance semantic grounding, DualFlow employs a novel Retrieval-Augmented Generation (RAG) module for two-person motion that retrieves motion exemplars using music features and LLM-based text decompositions of spatial relations, body movements, and rhythmic patterns. We use a contrastive rectified flow objective to further sharpen alignment with conditioning signals and add synchronisation loss to improve inter-person temporal coordination. Extensive evaluations across interactive, reactive, and multi-modal benchmarks demonstrate that DualFlow consistently improves motion quality, responsiveness, and semantic fidelity. DualFlow achieves state-of-the-art performance in two-person multi-modal motion generation, producing coherent, expressive, and rhythmically synchronized motion.
>
---
#### [replaced 088] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21497](https://arxiv.org/pdf/2602.21497)**

> **作者:** Yongchang Zhang; Oliver Ma; Tianyi Liu; Guangquan Zhou; Yang Chen
>
> **备注:** CVPR2026 Accepted
>
> **摘要:** Recent large vision-language models (LVLMs) have demonstrated impressive reasoning ability by generating long chain-of-thought (CoT) responses. However, CoT reasoning in multimodal contexts is highly vulnerable to visual hallucination propagation: once an intermediate reasoning step becomes inconsistent with the visual evidence, subsequent steps-even if logically valid-can still lead to incorrect final answers. Existing solutions attempt to mitigate this issue by training models to "think with images" via reinforcement learning (RL). While effective, these methods are costly, model-specific, and difficult to generalize across architectures. Differently, we present a lightweight method that bypasses RL training and provides an iterative, training-free, plug-and-play framework for visually-grounded multimodal reasoning. Our key idea is to supervise each reasoning step at test time with visual evidence, ensuring that every decoded token is justified by corresponding visual cues. Concretely, we construct a textual visual-evidence pool that guides the model's reasoning generation. When existing evidence is insufficient, a visual decider module dynamically extracts additional relevant evidence from the image based on the ongoing reasoning context, expanding the pool until the model achieves sufficient visual certainty to terminate reasoning and produce the final answer. Extensive experiments on multiple LVLM backbones and benchmarks demonstrate the effectiveness of our approach. Our method achieves 16.5%-29.5% improvements on TreeBench and 13.7% RH-AUC gains on RH-Bench, substantially reducing hallucination rates while improving reasoning accuracy without additional training.
>
---
#### [replaced 089] Route, Retrieve, Reflect, Repair: Self-Improving Agentic Framework for Visual Detection and Linguistic Reasoning in Medical Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08192](https://arxiv.org/pdf/2601.08192)**

> **作者:** Md. Faiyaz Abdullah Sayeedi; Rashedur Rahman; Siam Tahsin Bhuiyan; Sefatul Wasi; Ashraful Islam; Saadia Binte Alam; AKM Mahbubur Rahman
>
> **摘要:** Medical image analysis increasingly relies on large vision-language models (VLMs), yet most systems remain single-pass black boxes that offer limited control over reasoning, safety, and spatial grounding. We propose R^4, an agentic framework that decomposes medical imaging workflows into four coordinated agents: a Router that configures task- and specialization-aware prompts from the image, patient history, and metadata; a Retriever that uses exemplar memory and pass@k sampling to jointly generate free-text reports and bounding boxes; a Reflector that critiques each draft-box pair for key clinical error modes (negation, laterality, unsupported claims, contradictions, missing findings, and localization errors); and a Repairer that iteratively revises both narrative and spatial outputs under targeted constraints while curating high-quality exemplars for future cases. Instantiated on chest X-ray analysis with multiple modern VLM backbones and evaluated on report generation and weakly supervised detection, R^4 consistently boosts LLM-as-a-Judge scores by roughly +1.7-+2.5 points and mAP50 by +2.5-+3.5 absolute points over strong single-VLM baselines, without any gradient-based fine-tuning. These results show that agentic routing, reflection, and repair can turn strong but brittle VLMs into more reliable and better grounded tools for clinical image interpretation. Our code can be found at: this https URL
>
---
#### [replaced 090] LAHNet: Local Attentive Hashing Network for Point Cloud Registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00927](https://arxiv.org/pdf/2512.00927)**

> **作者:** Wentao Qu; Xiaoshui Huang; Liang Xiao
>
> **摘要:** Most existing learning-based point cloud descriptors for point cloud registration focus on perceiving local information of point clouds to generate distinctive features. However, a reasonable and broader receptive field is essential for enhancing feature distinctiveness. In this paper, we propose a Local Attentive Hashing Network for point cloud registration, called LAHNet, which introduces a local attention mechanism with the inductive bias of locality of convolution-like operators into point cloud descriptors. Specifically, a Group Transformer is designed to capture reasonable long-range context between points. This employs a linear neighborhood search strategy, Locality-Sensitive Hashing, enabling uniformly partitioning point clouds into non-overlapping windows. Meanwhile, an efficient cross-window strategy is adopted to further expand the reasonable feature receptive field. Furthermore, building on this effective windowing strategy, we propose an Interaction Transformer to enhance the feature interactions of the overlap regions within point cloud pairs. This computes an overlap matrix to match overlap regions between point cloud pairs by representing each window as a global signal. Extensive results demonstrate that LAHNet can learn robust and distinctive features, achieving significant registration results on real-world indoor and outdoor benchmarks.
>
---
#### [replaced 091] Mix-modal Federated Learning for MRI Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.02541](https://arxiv.org/pdf/2509.02541)**

> **作者:** Guyue Hu; Siyuan Song; Jingpeng Sun; Zhe Jin; Chenglong Li; Jin Tang
>
> **摘要:** Magnetic resonance imaging (MRI) image segmentation is crucial in diagnosing and treating many diseases, such as brain tumors. Existing MRI image segmentation methods mainly fall into a centralized multimodal paradigm, which is inapplicable in engineering non-centralized mix-modal medical scenarios. In this situation, each distributed client (hospital) processes multiple mixed MRI modalities, and the modality set and image data for each client are diverse, suffering from extensive client-wise modality heterogeneity and data heterogeneity. In this paper, we first formulate non-centralized mix-modal MRI image segmentation as a new paradigm for federated learning (FL) that involves multiple modalities, called mix-modal federated learning (MixMFL). It distinguishes from existing multimodal federating learning (MulMFL) and cross-modal federating learning (CroMFL) paradigms. Then, we proposed a novel modality decoupling and memorizing mix-modal federated learning framework (MDM-MixMFL) for MRI image segmentation, which is characterized by a modality decoupling strategy and a modality memorizing mechanism. Specifically, the modality decoupling strategy disentangles each modality into modality-tailored and modality-shared information. During mix-modal federated updating, corresponding modality encoders undergo tailored and shared updating, respectively. It facilitates stable and adaptive federating aggregation of heterogeneous data and modalities from distributed clients. Besides, the modality memorizing mechanism stores client-shared modality prototypes dynamically refreshed from every modality-tailored encoder to compensate for incomplete modalities in each local client.
>
---
#### [replaced 092] UnfoldLDM: Deep Unfolding-based Blind Image Restoration with Latent Diffusion Priors
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18152](https://arxiv.org/pdf/2511.18152)**

> **作者:** Chunming He; Rihan Zhang; Zheng Chen; Bowen Yang; Chengyu Fang; Yunlong Lin; Yulun Zhang; Fengyang Xiao; Sina Farsiu
>
> **备注:** 6 figures, 11 tables
>
> **摘要:** Deep unfolding networks (DUNs) combine the interpretability of model-based methods with the learning ability of deep networks, yet remain limited for blind image restoration (BIR). Existing DUNs suffer from: (1) \textbf{Degradation-specific dependency}, as their optimization frameworks are tied to a known degradation model, making them unsuitable for BIR tasks; and (2) \textbf{Over-smoothing bias}, resulting from the direct feeding of gradient descent outputs, dominated by low-frequency content, into the proximal term, suppressing fine textures. To overcome these issues, we propose UnfoldLDM to integrate DUNs with latent diffusion model (LDM) for BIR. In each stage, UnfoldLDM employs a multi-granularity degradation-aware (MGDA) module as the gradient descent step. MGDA models BIR as an unknown degradation estimation problem and estimates both the holistic degradation matrix and its decomposed forms, enabling robust degradation removal. For the proximal step, we design a degradation-resistant LDM (DR-LDM) to extract compact degradation-invariant priors from the MGDA output. Guided by this prior, an over-smoothing correction transformer (OCFormer) explicitly recovers high-frequency components and enhances texture details. This unique combination ensures the final result is degradation-free and visually rich. Experiments show that our UnfoldLDM achieves a leading place on various BIR tasks and benefits downstream tasks. Moreover, our design is compatible with existing DUN-based methods, serving as a plug-and-play framework. Code will be released.
>
---
#### [replaced 093] Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于隐式表面建模任务，解决从单张图像高效构建隐式距离表示的问题。提出FINS框架，利用单图快速生成高保真表面和SDF场。**

- **链接: [https://arxiv.org/pdf/2509.20681](https://arxiv.org/pdf/2509.20681)**

> **作者:** Wei-Teng Chu; Tianyi Zhang; Matthew Johnson-Roberson; Weiming Zhi
>
> **摘要:** Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as NeuS and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets.
>
---
#### [replaced 094] $π$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出$\pi$-StepNFT，解决流式视觉-语言-动作模型在在线强化学习中的多步采样问题，通过细粒度步骤引导提升性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.02083](https://arxiv.org/pdf/2603.02083)**

> **作者:** Siting Wang; Xiaofeng Wang; Zheng Zhu; Minnan Pei; Xinyu Cui; Cheng Deng; Jian Zhao; Guan Huang; Haifeng Zhang; Jun Wang
>
> **摘要:** Flow-based vision-language-action (VLA) models excel in embodied control but suffer from intractable likelihoods during multi-step sampling, hindering online reinforcement learning. We propose \textbf{\textit{$\boldsymbol{\pi}$-StepNFT}} (Step-wise Negative-aware Fine-Tuning), a critic-and-likelihood-free framework that requires only a single forward pass per optimization step and eliminates auxiliary value networks. We identify that wider exploration spaces necessitate finer-grained, step-wise guidance for alignment. Empirically, $\pi$-StepNFT unlocks latent potential on LIBERO with competitive few-shot robustness. Moreover, it achieves superior generalization on ManiSkill, outperforming value-based baselines in OOD scenarios by preventing overfitting to multimodal features. This property offers a scalable solution promising for complex real-world applications.
>
---
#### [replaced 095] UltraUPConvNet: A UPerNet- and ConvNeXt-Based Multi-Task Network for Ultrasound Tissue Segmentation and Disease Prediction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11108](https://arxiv.org/pdf/2509.11108)**

> **作者:** Zhi Chen; Le Zhang
>
> **备注:** 8 pages
>
> **摘要:** Ultrasound imaging is widely used in clinical practice due to its cost-effectiveness, mobility, and safety. However, current AI research often treats disease prediction and tissue segmentation as two separate tasks and their model requires substantial computational overhead. In such a situation, we introduce UltraUPConvNet, a computationally efficient universal framework designed for both ultrasound image classification and segmentation. Trained on a large-scale dataset containing more than 9,700 annotations across seven different anatomical regions, our model achieves state-of-the-art performance on certain datasets with lower computational overhead. Our model weights and codes are available at this https URL
>
---
#### [replaced 096] SAGA: Selective Adaptive Gating for Efficient and Expressive Linear Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12817](https://arxiv.org/pdf/2509.12817)**

> **作者:** Yuan Cao; Dong Wang
>
> **摘要:** While Transformer architecture excel at modeling long-range dependencies contributing to its widespread adoption in vision tasks the quadratic complexity of softmax-based attention mechanisms imposes a major bottleneck, particularly when processing high-resolution images. Linear attention presents a promising alternative by reformulating the attention computation from $(QK)V$ to $Q(KV)$, thereby reducing the complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ while preserving the global receptive field. However, most existing methods compress historical key-value (KV) information uniformly, which can lead to feature redundancy and the loss of directional alignment with the query (Q). This uniform compression results in low-rank $KV$ feature maps, contributing to a performance gap compared to softmax attention. To mitigate this limitation, we propose \textbf{S}elective \textbf{A}daptive \textbf{GA}ting for Efficient and Expressive Linear Attention (SAGA) , which introduces input-adaptive learnable gates to selectively modulate information aggregation into the $KV$ feature map. These gates enhance semantic diversity and alleviate the low-rank constraint inherent in conventional linear attention. Additionally, we propose an efficient Hadamard-product decomposition method for gate computation, which introduces no additional memory overhead. Experiments demonstrate that SAGA achieves a 1.76$\times$ improvement in throughput and a 2.69$\times$ reduction in peak GPU memory compared to PVT-T at a resolution of $1280 \times 1280$. Moreover, it improves top-1 accuracy by up to 4.4\% on the ImageNet dataset, demonstrating both computational efficiency and model effectiveness.
>
---
#### [replaced 097] ITO: Images and Texts as One via Synergizing Multiple Alignment and Training-Time Fusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.02767](https://arxiv.org/pdf/2603.02767)**

> **作者:** Hanpeng Liu; Yaqian Li; Zidan Wang; Shuoxi Zhang; Zonglin Zhao; Zihao Bo; Rinyoichi Takezoe; Kaiwen Long; Kun He
>
> **摘要:** Image-text contrastive pretraining has become a dominant paradigm for visual representation learning, yet existing methods often yield representations that remain partially organized by modality. We propose ITO, a framework addressing this limitation through two synergistic mechanisms. Multimodal multiple alignment enriches supervision by mining diverse image-text correspondences, while a lightweight training-time multimodal fusion module enforces structured cross-modal interaction. Crucially, the fusion module is discarded at inference, preserving the efficiency of standard dual-encoder architectures. Extensive experiments show that ITO consistently outperforms strong baselines across classification, retrieval, and multimodal benchmarks. Our analysis reveals that while multiple alignment drives discriminative power, training-time fusion acts as a critical structural regularizer -- eliminating the modality gap and stabilizing training dynamics to prevent the early saturation often observed in aggressive contrastive learning.
>
---
#### [replaced 098] Reversible Inversion for Training-Free Exemplar-guided Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01382](https://arxiv.org/pdf/2512.01382)**

> **作者:** Yuke Li; Lianli Gao; Ji Zhang; Pengpeng Zeng; Lichuan Xiang; Hongkai Wen; Heng Tao Shen; Jingkuan Song
>
> **摘要:** Exemplar-guided Image Editing (EIE) aims to modify a source image according to a visual reference. Existing approaches often require large-scale pre-training to learn relationships between the source and reference images, incurring high computational costs. As a training-free alternative, inversion techniques can be used to map the source image into a latent space for manipulation. However, our empirical study reveals that standard inversion is sub-optimal for EIE, leading to poor quality and inefficiency. To tackle this challenge, we introduce \textbf{Reversible Inversion ({ReInversion})} for effective and efficient EIE. Specifically, ReInversion operates as a two-stage denoising process, which is first conditioned on the source image and subsequently on the reference. Besides, we introduce a Mask-Guided Selective Denoising (MSD) strategy to constrain edits to target regions, preserving the structural consistency of the background. Both qualitative and quantitative comparisons demonstrate that our ReInversion method achieves state-of-the-art EIE performance with the lowest computational overhead.
>
---
#### [replaced 099] WildActor: Unconstrained Identity-Preserving Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00586](https://arxiv.org/pdf/2603.00586)**

> **作者:** Qin Guo; Tianyu Yang; Xuanhua He; Fei Shen; Yong Zhang; Zhuoliang Kang; Xiaoming Wei; Dan Xu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Production-ready human video generation requires digital actors to maintain strictly consistent full-body identities across dynamic shots, viewpoints and motions, a setting that remains challenging for existing methods. Prior methods often suffer from face-centric behavior that neglects body-level consistency, or produce copy-paste artifacts where subjects appear rigid due to pose locking. We present Actor-18M, a large-scale human video dataset designed to capture identity consistency under unconstrained viewpoints and environments. Actor-18M comprises 1.6M videos with 18M corresponding human images, covering both arbitrary views and canonical three-view representations. Leveraging Actor-18M, we propose WildActor, a framework for any-view conditioned human video generation. We introduce an Asymmetric Identity-Preserving Attention mechanism coupled with a Viewpoint-Adaptive Monte Carlo Sampling strategy that iteratively re-weights reference conditions by marginal utility for balanced manifold coverage. Evaluated on the proposed Actor-Bench, WildActor consistently preserves body identity under diverse shot compositions, large viewpoint transitions, and substantial motions, surpassing existing methods in these challenging settings.
>
---
#### [replaced 100] Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决人体模型控制数据不足与难度限制问题。通过闭环生成高质量运动数据并迭代优化，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2602.21599](https://arxiv.org/pdf/2602.21599)**

> **作者:** Weisheng Xu; Qiwei Wu; Jiaxi Zhang; Tan Jing; Yangfan Li; Yuetong Fang; Jiaqi Xiong; Kai Wu; Rong Ou; Renjing Xu
>
> **摘要:** Physics-based humanoid control relies on training with motion datasets that have diverse data distributions. However, the fixed difficulty distribution of datasets limits the performance ceiling of the trained control policies. Additionally, the method of acquiring high-quality data through professional motion capture systems is constrained by costs, making it difficult to achieve large-scale scalability. To address these issues, we propose a closed-loop automated motion data generation and iterative framework. It can generate high-quality motion data with rich action semantics, including martial arts, dance, combat, sports, gymnastics, and more. Furthermore, our framework enables difficulty iteration of policies and data through physical metrics and objective evaluations, allowing the trained tracker to break through its original difficulty limits. On the PHC single-primitive tracker, using only approximately 1/10 of the AMASS dataset size, the average failure rate on the test set (2201 clips) is reduced by 45% compared to the baseline. Finally, we conduct comprehensive ablation and comparative experiments to highlight the rationality and advantages of our framework.
>
---
#### [replaced 101] A Detection-Gated Pipeline for Robust Glottal Area Waveform Extraction and Clinical Pathology Assessment
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.02087](https://arxiv.org/pdf/2603.02087)**

> **作者:** Harikrishnan Unnikrishnan
>
> **备注:** for associated code see: this https URL
>
> **摘要:** Background: Accurate glottal segmentation in high-speed videoendoscopy (HSV) is essential for extracting kinematic biomarkers of laryngeal function. However, existing deep learning models often produce spurious artifacts in non-glottal frames and fail to generalize across different clinical settings. Methods: We propose a detection-gated pipeline that integrates a localizer with a segmenter. A temporal consistency wrapper ensures robustness by suppressing false positives during glottal closure and occlusion. The segmenter was trained on a limited subset of the GIRAFE dataset (600 frames), while the localizer was trained on the BAGLS training set. The in-distribution localizer provides a tight region of interest (ROI), removing geometric anatomical variations and enabling cross-dataset generalization without fine-tuning. Results: The pipeline achieved state-of-the-art performance on the GIRAFE (DSC=0.81) and BAGLS (DSC=0.85) benchmarks and demonstrated superior generalizability. Notably, the framework maintained robust cross-dataset generalization (DSC=0.77). Downstream validation on a 65-subject clinical cohort confirmed that automated kinematic features - specifically the Open Quotient and Glottal Area Waveform (GAW) - remained consistent with clinical benchmarks. The coefficient of variation (CV) of the glottal area was a significant marker for distinguishing healthy from pathological vocal function (p=0.006). Conclusions: This architecture provides a computationally efficient solution (~35 frames/s) suitable for real-time clinical use. By overcoming cross-dataset variability, this framework facilitates the standardized, large-scale extraction of clinical biomarkers across diverse endoscopy platforms. Code, trained weights, and evaluation scripts are released at this https URL.
>
---
#### [replaced 102] $π^3$: Permutation-Equivariant Visual Geometry Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.13347](https://arxiv.org/pdf/2507.13347)**

> **作者:** Yifan Wang; Jianjun Zhou; Haoyi Zhu; Wenzheng Chang; Yang Zhou; Zizun Li; Junyi Chen; Jiangmiao Pang; Chunhua Shen; Tong He
>
> **备注:** Project page: this https URL
>
> **摘要:** We introduce $\pi^3$, a feed-forward neural network that offers a novel approach to visual geometry reconstruction, breaking the reliance on a conventional fixed reference view. Previous methods often anchor their reconstructions to a designated viewpoint, an inductive bias that can lead to instability and failures if the reference is suboptimal. In contrast, $\pi^3$ employs a fully permutation-equivariant architecture to predict affine-invariant camera poses and scale-invariant local point maps without any reference frames. This design not only makes our model inherently robust to input ordering, but also leads to higher accuracy and performance. These advantages enable our simple and bias-free approach to achieve state-of-the-art performance on a wide range of tasks, including camera pose estimation, monocular/video depth estimation, and dense point map reconstruction. Code and models are available at this https URL.
>
---
#### [replaced 103] Advances in 4D Representation: Geometry, Motion, and Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.19255](https://arxiv.org/pdf/2510.19255)**

> **作者:** Mingrui Zhao; Sauradip Nag; Kai Wang; Aditya Vora; Guangda Ji; Peter Chun; Ali Mahdavi-Amiri; Hao Zhang
>
> **备注:** 21 pages. Project Page: this https URL
>
> **摘要:** We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well as 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:this https URL
>
---
#### [replaced 104] CountFormer: A Transformer Framework for Learning Visual Repetition and Structure in Class-Agnostic Object Counting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.23785](https://arxiv.org/pdf/2510.23785)**

> **作者:** Md Tanvir Hossain; Akif Islam; Mohd Ruhul Ameen
>
> **备注:** Accepted at the 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence and Networking (QPAIN 2026)
>
> **摘要:** Humans can often count unfamiliar objects by observing visual repetition and composition, rather than relying only on object categories. However, many exemplar-free counting models struggle in such situations and may overcount when objects contain symmetric components, repeated substructures, or partial occlusion. We introduce CountFormer, a controlled adaptation of a density-regression framework inspired by CounTR, where the image encoder is replaced with the self-supervised vision foundation model DINOv2. The resulting transformer features are combined with explicit two-dimensional positional embeddings and decoded by a lightweight convolutional network to produce a density map whose integral gives the final count. Our goal is not to propose a new counting architecture, but to study whether foundation-based representations improve structural consistency under a strictly exemplar-free setting. On FSC-147, CountFormer achieves competitive performance under the official benchmark (MAE 19.06, RMSE 118.45). Qualitative analysis suggests fewer part-level overcounting errors for some structurally complex objects, while overall error remains broadly consistent with prior approaches. Sensitivity analysis shows that evaluation metrics are strongly affected by a small number of extreme high-density scenes. Overall, the results highlight the role of representation quality in exemplar-free object counting.
>
---
#### [replaced 105] MSP-ReID: Hairstyle-Robust Cloth-Changing Person Re-Identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01640](https://arxiv.org/pdf/2603.01640)**

> **作者:** Xiangyang He; Lin Wan
>
> **备注:** Accepted to the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026). The GitHub code for this paper is available at: this https URL
>
> **摘要:** Cloth-Changing Person Re-Identification (CC-ReID) aims to match the same individual across cameras under varying clothing conditions. Existing approaches often remove apparel and focus on the head region to reduce clothing bias. However, treating the head holistically without distinguishing between face and hair leads to over-reliance on volatile hairstyle cues, causing performance degradation under hairstyle changes. To address this issue, we propose the Mitigating Hairstyle Distraction and Structural Preservation (MSP) framework. Specifically, MSP introduces Hairstyle-Oriented Augmentation (HSOA), which generates intra-identity hairstyle diversity to reduce hairstyle dependence and enhance attention to stable facial and body cues. To prevent the loss of structural information, we design Cloth-Preserved Random Erasing (CPRE), which performs ratio-controlled erasing within clothing regions to suppress texture bias while retaining body shape and context. Furthermore, we employ Region-based Parsing Attention (RPA) to incorporate parsing-guided priors that highlight face and limb regions while suppressing hair features. Extensive experiments on multiple CC-ReID benchmarks demonstrate that MSP achieves state-of-the-art performance, providing a robust and practical solution for long-term person re-identification.
>
---
#### [replaced 106] Efficient Diffusion-Based 3D Human Pose Estimation with Hierarchical Temporal Pruning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.21363](https://arxiv.org/pdf/2508.21363)**

> **作者:** Yuquan Bi; Hongsong Wang; Xinli Shi; Zhipeng Gui; Jie Gui; Yuan Yan Tang
>
> **备注:** Accepted by IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHOLOGY
>
> **摘要:** Diffusion models have demonstrated strong capabilities in generating high-fidelity 3D human poses, yet their iterative nature and multi-hypothesis requirements incur substantial computational cost. In this paper, we propose an Efficient Diffusion-Based 3D Human Pose Estimation framework with a Hierarchical Temporal Pruning (HTP) strategy, which dynamically prunes redundant pose tokens across both frame and semantic levels while preserving critical motion dynamics. HTP operates in a staged, top-down manner: (1) Temporal Correlation-Enhanced Pruning (TCEP) identifies essential frames by analyzing inter-frame motion correlations through adaptive temporal graph construction; (2) Sparse-Focused Temporal MHSA (SFT MHSA) leverages the resulting frame-level sparsity to reduce attention computation, focusing on motion-relevant tokens; and (3) Mask-Guided Pose Token Pruner (MGPTP) performs fine-grained semantic pruning via clustering, retaining only the most informative pose tokens. Experiments on Human3.6M and MPI-INF-3DHP show that HTP reduces training MACs by 38.5\%, inference MACs by 56.8\%, and improves inference speed by an average of 81.1\% compared to prior diffusion-based methods, while achieving state-of-the-art performance.
>
---
#### [replaced 107] Towards Generalizable Forgery Detection and Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.21210](https://arxiv.org/pdf/2503.21210)**

> **作者:** Yueying Gao; Dongliang Chang; Bingyao Yu; Haotian Qin; Muxi Diao; Lei Chen; Kongming Liang; Zhanyu Ma
>
> **备注:** Accepted to IEEE TIP
>
> **摘要:** Accurate and interpretable detection of AI-generated images is essential for mitigating risks associated with AI misuse. However, the substantial domain gap among generative models makes it challenging to develop a generalizable forgery detection model. Moreover, since every pixel in an AI-generated image is synthesized, traditional saliency-based forgery explanation methods are not well suited for this task. To address these challenges, we formulate detection and explanation as a unified Forgery Detection and Reasoning task (FDR-Task), leveraging Multi-Modal Large Language Models (MLLMs) to provide accurate detection through reliable reasoning over forgery attributes. To facilitate this task, we introduce the Multi-Modal Forgery Reasoning dataset (MMFR-Dataset), a large-scale dataset containing 120K images across 10 generative models, with 378K reasoning annotations on forgery attributes, enabling comprehensive evaluation of the FDR-Task. Furthermore, we propose FakeReasoning, a forgery detection and reasoning framework with three key components: 1) a dual-branch visual encoder that integrates CLIP and DINO to capture both high-level semantics and low-level artifacts; 2) a Forgery-Aware Feature Fusion Module that leverages DINO's attention maps and cross-attention mechanisms to guide MLLMs toward forgery-related clues; 3) a Classification Probability Mapper that couples language modeling and forgery detection, enhancing overall performance. Experiments across multiple generative models demonstrate that FakeReasoning not only achieves robust generalization but also outperforms state-of-the-art methods on both detection and reasoning tasks.
>
---
#### [replaced 108] Annotation-Free Visual Reasoning for High-Resolution Large Multimodal Models via Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23615](https://arxiv.org/pdf/2602.23615)**

> **作者:** Jiacheng Yang; Anqi Chen; Yunkai Dang; Qi Fan; Cong Wang; Wenbin Li; Feng Miao; Yang Gao
>
> **摘要:** Current Large Multimodal Models (LMMs) struggle with high-resolution visual inputs during the reasoning process, as the number of image tokens increases quadratically with resolution, introducing substantial redundancy and irrelevant information. A common practice is to identify key image regions and refer to their high-resolution counterparts during reasoning, typically trained with external visual supervision. However, such visual supervision cues require costly grounding labels from human annotators. Meanwhile, it remains an open question how to enhance a model's grounding abilities to support reasoning without relying on additional annotations. In this paper, we propose High-resolution Annotation-free Reasoning Technique (HART), a closed-loop framework that enables LMMs to focus on and self-verify key regions of high-resolution visual inputs. HART incorporates a post-training paradigm in which we design Advantage Preference Group Relative Policy Optimization (AP-GRPO) to encourage accurate localization of key regions without external visual annotations. Notably, HART provides explainable reasoning pathways and enables efficient optimization of localization. Extensive experiments on MME-RealWorld-Lite, TreeBench, V* Bench, HR-Bench-4K/8K, and MMStar demonstrate that HART improves performance across a wide range of high-resolution visual tasks, consistently outperforming strong baselines.
>
---
#### [replaced 109] Multi-Order Matching Network for Alignment-Free Depth Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16361](https://arxiv.org/pdf/2511.16361)**

> **作者:** Zhengxue Wang; Zhiqiang Yan; Yuan Wu; Guangwei Gao; Xiang Li; Jian Yang
>
> **摘要:** Recent guided depth super-resolution methods are premised on the assumption of strictly spatial alignment between depth and RGB, achieving high-quality depth reconstruction. However, in real-world scenarios, the acquisition of strictly aligned RGB-D is hindered by inherent hardware limitations (e.g., physically separate RGB-D sensors) and unavoidable calibration drift induced by mechanical vibrations or temperature variations. Consequently, existing approaches often suffer inevitable performance degradation when applied to misaligned real-world scenes. In this paper, we propose the Multi-Order Matching Network (MOMNet), a novel alignment-free framework that adaptively retrieves and selects the most relevant information from misaligned RGB. Specifically, our method begins with a multi-order matching mechanism, which jointly performs zero-order, first-order, and second-order matching to comprehensively identify RGB information consistent with depth across multi-order feature spaces. To effectively integrate the retrieved RGB and depth, we further introduce a multi-order aggregation composed of multiple structure detectors. This strategy uses multi-order priors as prompts to facilitate the selective feature transfer from RGB to depth. Extensive experiments demonstrate that MOMNet achieves state-of-the-art performance and exhibits outstanding robustness.
>
---
#### [replaced 110] PackUV: Packed Gaussian UV Maps for 4D Volumetric Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23040](https://arxiv.org/pdf/2602.23040)**

> **作者:** Aashish Rai; Angela Xing; Anushka Agarwal; Xiaoyan Cong; Zekun Li; Tao Lu; Aayush Prakash; Srinath Sridhar
>
> **备注:** this https URL
>
> **摘要:** Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.
>
---
#### [replaced 111] SPEX: A Vision-Language Model for Land Cover Extraction on Spectral Remote Sensing Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05202](https://arxiv.org/pdf/2508.05202)**

> **作者:** Dongchen Si; Di Wang; Erzhong Gao; Xiaolei Qin; Liu Zhao; Jing Zhang; Minqiang Xu; Jianbo Zhan; Jianshe Wang; Lin Liu; Bo Du; Liangpei Zhang
>
> **备注:** Accepted to IEEE TGRS
>
> **摘要:** Spectral information has long been recognized as a critical cue in remote sensing observations. Although numerous vision-language models have been developed for pixel-level interpretation, spectral information remains underutilized, resulting in suboptimal performance, particularly in multispectral scenarios. To address this limitation, we construct a vision-language instruction-following dataset named SPIE, which encodes spectral priors of land-cover objects into textual attributes recognizable by large language models (LLMs), based on classical spectral index computations. Leveraging this dataset, we propose SPEX, a multimodal LLM designed for instruction-driven land cover extraction. To this end, we introduce several carefully designed components and training strategies, including multiscale feature aggregation, token context condensation, and multispectral visual pre-training, to achieve precise and flexible pixel-level interpretation. To the best of our knowledge, SPEX is the first multimodal vision-language model dedicated to land cover extraction in spectral remote sensing imagery. Extensive experiments on five public multispectral datasets demonstrate that SPEX consistently outperforms existing state-of-the-art methods in extracting typical land cover categories such as vegetation, buildings, and water bodies. Moreover, SPEX is capable of generating textual explanations for its predictions, thereby enhancing interpretability and user-friendliness. Code will be released at: this https URL.
>
---
#### [replaced 112] ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.04385](https://arxiv.org/pdf/2603.04385)**

> **作者:** Haian Jin; Rundi Wu; Tianyuan Zhang; Ruiqi Gao; Jonathan T. Barron; Noah Snavely; Aleksander Holynski
>
> **备注:** Project page: this https URL
>
> **摘要:** Feed-forward transformer models have driven rapid progress in 3D vision, but state-of-the-art methods such as VGGT and $\pi^3$ have a computational cost that scales quadratically with the number of input images, making them inefficient when applied to large image collections. Sequential-reconstruction approaches reduce this cost but sacrifice reconstruction quality. We introduce ZipMap, a stateful feed-forward model that achieves linear-time, bidirectional 3D reconstruction while matching or surpassing the accuracy of quadratic-time methods. ZipMap employs test-time training layers to zip an entire image collection into a compact hidden scene state in a single forward pass, enabling reconstruction of over 700 frames in under 10 seconds on a single H100 GPU, more than $20\times$ faster than state-of-the-art methods such as VGGT. Moreover, we demonstrate the benefits of having a stateful representation in real-time scene-state querying and its extension to sequential streaming reconstruction.
>
---
#### [replaced 113] Shortcut Invariance: Targeted Jacobian Regularization in Disentangled Latent Space
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.19525](https://arxiv.org/pdf/2511.19525)**

> **作者:** Shivam Pal; Sakshi Varshney; Piyush Rai
>
> **摘要:** Deep neural networks are prone to learning shortcuts, spurious correlations present in the training data that undermine out-of-distribution (OOD) generalization. Most prior work mitigates shortcut learning through input-space reweighting, either relying on explicit shortcut labels or inferring shortcut structure from heuristics such as per-sample loss. Moreover, these approaches typically assume the presence of some shortcut-conflicting examples in the training set, an assumption that is often violated in practice, particularly in medical imaging where data is aggregated across institutions with different acquisition protocols. We propose a latent-space method that views shortcut learning as over-reliance on shortcut-aligned axes. In a disentangled latent space, we identify candidate shortcut-aligned axes via their strong correlation with labels and reduce classifier reliance on them by injecting targeted anisotropic noise during training. Unlike prior latent-space based approaches that remove, project out, or adversarially suppress shortcut features, our method preserves the full representation and instead impose functional invariance by regularizing the classifier's sensitivity along those axes. We show that injecting anisotropic noise induces targeted Jacobian and curvature regularization, effectively flattening the decision boundary along shortcut axes while leaving core feature dimensions largely unaffected. Our method achieves state-of-the-art OOD performance across standard shortcut-learning benchmarks without requiring shortcut labels or shortcut-conflicting samples.
>
---
#### [replaced 114] Vid2World: Crafting Video Diffusion Models to Interactive World Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.14357](https://arxiv.org/pdf/2505.14357)**

> **作者:** Siqiao Huang; Jialong Wu; Qixing Zhou; Shangchen Miao; Mingsheng Long
>
> **备注:** Project page: this http URL
>
> **摘要:** World models, which predict future transitions from past observation and action sequences, have shown great promise for improving data efficiency in sequential decision-making. However, existing world models often require extensive domain-specific training and still produce low-fidelity, coarse predictions, limiting their usefulness in complex environments. In contrast, video diffusion models trained on large-scale internet data have demonstrated impressive capabilities in generating high-quality videos that capture diverse real-world dynamics. In this work, we present Vid2World, a general approach for leveraging and transferring pre-trained video diffusion models into interactive world models. To bridge the gap, Vid2World systematically explores video diffusion causalization, reshaping both the architecture and training objective of pre-trained models to enable autoregressive generation. Additionally, it incorporates a causal action guidance mechanism to enhance action controllability in the resulting interactive world models. Extensive experiments across multiple domains, including robot manipulation, 3D game simulation, and open-world navigation, demonstrate that our method offers a scalable and effective pathway for repurposing highly capable video diffusion models into interactive world models.
>
---
#### [replaced 115] Modular Neural Image Signal Processing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08564](https://arxiv.org/pdf/2512.08564)**

> **作者:** Mahmoud Afifi; Zhongling Wang; Ran Zhang; Michael S. Brown
>
> **摘要:** This paper presents a modular neural image signal processing (ISP) framework that processes raw inputs and renders high-quality display-referred images. Unlike prior neural ISP designs, our method introduces a high degree of modularity, providing full control over multiple intermediate stages of the rendering process.~This modular design not only achieves high rendering accuracy but also improves scalability, debuggability, generalization to unseen cameras, and flexibility to match different user-preference styles. To demonstrate the advantages of this design, we built a user-interactive photo-editing tool that leverages our neural ISP to support diverse editing operations and picture styles. The tool is carefully engineered to take advantage of the high-quality rendering of our neural ISP and to enable unlimited post-editable re-rendering. Our method is a fully learning-based framework with variants of different capacities, all of moderate size (ranging from ~0.5 M to ~3.9 M parameters for the entire pipeline), and consistently delivers competitive qualitative and quantitative results across multiple test sets. Watch the supplemental video at: this https URL
>
---
#### [replaced 116] S2DiT: Sandwich Diffusion Transformer for Mobile Streaming Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12719](https://arxiv.org/pdf/2601.12719)**

> **作者:** Lin Zhao; Yushu Wu; Aleksei Lebedev; Dishani Lahiri; Meng Dong; Arpit Sahni; Michael Vasilkovsky; Hao Chen; Ju Hu; Aliaksandr Siarohin; Sergey Tulyakov; Yanzhi Wang; Anil Kag; Yanyu Li
>
> **备注:** this https URL
>
> **摘要:** Diffusion Transformers (DiTs) have recently improved video generation quality. However, their heavy computational cost makes real-time or on-device generation infeasible. In this work, we introduce S2DiT, a Streaming Sandwich Diffusion Transformer designed for efficient, high-fidelity, and streaming video generation on mobile hardware. S2DiT generates more tokens but maintains efficiency with novel efficient attentions: a mixture of LinConv Hybrid Attention (LCHA) and Stride Self-Attention (SSA). Based on this, we uncover the sandwich design via a budget-aware dynamic programming search, achieving superior quality and efficiency. We further propose a 2-in-1 distillation framework that transfers the capacity of large teacher models (e.g., Wan 2.2-14B) to the compact few-step sandwich model. Together, S2DiT achieves quality on par with state-of-the-art server video models, while streaming at over 10 FPS on an iPhone.
>
---
#### [replaced 117] SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决多模态大语言模型计算成本高的问题。提出SToRM框架，在减少视觉token的同时保持性能，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.11656](https://arxiv.org/pdf/2602.11656)**

> **作者:** Seo Hyun Kim; Jin Bok Park; Do Yeon Koo; Hogun Park; Il Yong Chun
>
> **摘要:** In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens. To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x.
>
---
#### [replaced 118] Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2405.19931](https://arxiv.org/pdf/2405.19931)**

> **作者:** Xiaoyu Wu; Jiaru Zhang; Yang Hua; Bohan Lyu; Hao Wang; Tao Song; Haibing Guan
>
> **备注:** Accepted by KDD' 26
>
> **摘要:** Few-shot fine-tuning of Diffusion Models (DMs) is a key advancement, significantly reducing training costs and enabling personalized AI applications. However, we explore the training dynamics of DMs and observe an unanticipated phenomenon: during the training process, image fidelity initially improves, then unexpectedly deteriorates with the emergence of noisy patterns, only to recover later with severe overfitting. We term the stage with generated noisy patterns as corruption stage. To understand this corruption stage, we begin by theoretically modeling the one-shot fine-tuning scenario, and then extend this modeling to more general cases. Through this modeling, we identify the primary cause of this corruption stage: a narrowed learning distribution inherent in the nature of few-shot fine-tuning. To tackle this, we apply Bayesian Neural Networks (BNNs) on DMs with variational inference to implicitly broaden the learned distribution, and present that the learning target of the BNNs can be naturally regarded as an expectation of the diffusion loss and a further regularization with the pretrained DMs. This approach is highly compatible with current few-shot fine-tuning methods in DMs and does not introduce any extra inference costs. Experimental results demonstrate that our method significantly mitigates corruption, and improves the fidelity, quality and diversity of the generated images in both object-driven and subject-driven generation tasks. Code is available at this https URL.
>
---
#### [replaced 119] DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01111](https://arxiv.org/pdf/2603.01111)**

> **作者:** Yiming Ma; Hongkun Yang; Lionel Z. Wang; Bin Chen; Weizhi Xian; Jianzhi Teng
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Prompt learning is a dominant paradigm for adapting pre-trained Vision-Language Models (VLMs) to downstream tasks. However, existing methods often rely on a simplistic, layer-centric view, assuming shallow layers capture general features while deep layers handle task-specific knowledge. This assumption results in uncontrolled interactions between learnable tokens and original tokens. Task-specific knowledge could degrades the model's core generalization and creates a trade-off between task adaptation and the preservation of zero-shot generalization. To address this, we challenge the layer-centric view and propose \textbf{DeAR}, a framework that achieves fine-grained VLM adaptation by \textbf{De}composing \textbf{A}ttention head \textbf{R}oles. We posit that the functional specialization within VLMs occurs not between layers, but at the finer-grained level of individual attention heads in the deeper layers. Based on this insight, we introduce a novel metric, Concept Entropy, to systematically classify attention heads into distinct functional roles: \textit{Attribute}, \textit{Generalization}, and \textit{Mixed}. Guided by these roles, we introduce specialized attribute tokens and a Role-Based Attention Mask mechanism to precisely control information flow, ensuring generalization heads remain isolated from task-specific knowledge. We further incorporate a Task-Adaptive Fusion Strategy for inference. Extensive experiments on fifteen datasets show that DeAR achieves a strong balance between task adaptation and generalization, outperforming previous methods across various tasks.
>
---
#### [replaced 120] DeepSparse: A Foundation Model for Sparse-View CBCT Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.02628](https://arxiv.org/pdf/2505.02628)**

> **作者:** Yiqun Lin; Jixiang Chen; Hualiang Wang; Jiewen Yang; Jiarong Guo; Yi Zhang; Xiaomeng Li
>
> **摘要:** Cone-beam computed tomography (CBCT) is a critical 3D imaging technology in the medical field, while the high radiation exposure required for high-quality imaging raises significant concerns, particularly for vulnerable populations. Sparse-view reconstruction reduces radiation by using fewer X-ray projections while maintaining image quality, yet existing methods face challenges such as high computational demands and poor generalizability to different datasets. To overcome these limitations, we propose DeepSparse, the first foundation model for sparse-view CBCT reconstruction, featuring DiCE (Dual-Dimensional Cross-Scale Embedding), a novel network that integrates multi-view 2D features and multi-scale 3D features. Additionally, we introduce the HyViP (Hybrid View Sampling Pretraining) framework, which pretrains the model on large datasets with both sparse-view and dense-view projections, and a two-step finetuning strategy to adapt and refine the model for new datasets. Extensive experiments and ablation studies demonstrate that our proposed DeepSparse achieves superior reconstruction quality compared to state-of-the-art methods, paving the way for safer and more efficient CBCT imaging.
>
---
#### [replaced 121] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13911](https://arxiv.org/pdf/2508.13911)**

> **作者:** Chunji Lv; Zequn Chen; Donglin Di; Weinan Zhang; Hao Li; Wei Chen; Yinjie Lei; Changsheng Li
>
> **备注:** CVPR 2026
>
> **摘要:** Despite advances in physics-based 3D motion synthesis, current methods face key limitations: reliance on pre-reconstructed 3D Gaussian Splatting (3DGS) built from dense multi-view images with time-consuming per-scene optimization; physics integration via either inflexible, hand-specified attributes or unstable, optimization-heavy guidance from video models using Score Distillation Sampling (SDS); and naive concatenation of prebuilt 3DGS with physics modules, which ignores physical information embedded in appearance and yields suboptimal performance. To address these issues, we propose PhysGM, a feed-forward framework that jointly predicts 3D Gaussian representation and physical properties from a single image, enabling immediate simulation and high-fidelity 4D rendering. Unlike slow appearance-agnostic optimization methods, we first pre-train a physics-aware reconstruction model that directly infers both Gaussian and physical parameters. We further refine the model with Direct Preference Optimization (DPO), aligning simulations with the physically plausible reference videos and avoiding the high-cost SDS optimization. To address the absence of a supporting dataset for this task, we propose PhysAssets, a dataset of 50K+ 3D assets annotated with physical properties and corresponding reference videos. Experiments show that PhysGM produces high-fidelity 4D simulations from a single image in one minute, achieving a significant speedup over prior work while delivering realistic this http URL project page is at:this https URL
>
---
#### [replaced 122] Open-Vocabulary Camouflaged Object Segmentation with Cascaded Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.19300](https://arxiv.org/pdf/2506.19300)**

> **作者:** Kai Zhao; Wubang Yuan; Zheng Wang; Guanyi Li; Xiaoqiang Zhu; Deng-ping Fan; Dan Zeng
>
> **备注:** Accepted to Computational Visual Media (CVMJ) 2026
>
> **摘要:** Open-Vocabulary Camouflaged Object Segmentation (OVCOS) seeks to segment and classify camouflaged objects from arbitrary categories, presenting unique challenges due to visual ambiguity and unseen this http URL approaches typically adopt a two-stage paradigm: first segmenting objects, then classifying the segmented regions using Vision Language Models (VLMs).However, these methods (1) suffer from a domain gap caused by the mismatch between VLMs' full-image training and cropped-region inference, and (2) depend on generic segmentation models optimized for well-delineated objects, making them less effective for camouflaged this http URL explicit guidance, generic segmentation models often overlook subtle boundaries, leading to imprecise this http URL this paper,we introduce a novel VLM-guided cascaded framework to address these issues in this http URL segmentation, we leverage the Segment Anything Model (SAM), guided by the this http URL framework uses VLM-derived features as explicit prompts to SAM, effectively directing attention to camouflaged regions and significantly improving localization this http URL classification, we avoid the domain gap introduced by hard this http URL, we treat the segmentation output as a soft spatial prior via the alpha channel, which retains the full image context while providing precise spatial guidance, leading to more accurate and context-aware classification of camouflaged this http URL same VLM is shared across both segmentation and classification to ensure efficiency and semantic this http URL experiments on both OVCOS and conventional camouflaged object segmentation benchmarks demonstrate the clear superiority of our method, highlighting the effectiveness of leveraging rich VLM semantics for both segmentation and classification of camouflaged objects.
>
---
#### [replaced 123] Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable Remedy
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频理解任务，旨在解决现有基准未充分评估音频作用的问题。通过引入语音编码器，验证音频在跨模态任务中的重要性，并提出有效压缩方案。**

- **链接: [https://arxiv.org/pdf/2509.17901](https://arxiv.org/pdf/2509.17901)**

> **作者:** Geewook Kim; Minjoon Seo
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Speech and audio encoders developed over years of community effort are routinely excluded from video understanding pipelines -- not because they fail, but because benchmarks never required listening. We audit 10 video benchmarks and find items largely solvable from visual cues alone: a single-frame probe answers ~77% of AVQA without audio, suggesting poor measurement of audio-visual reasoning. Building on LLaVA-OneVision, we attach a speech/audio encoder and compare five compressor architectures under 25x token reduction (25 Hz to 1 Hz). Across 10 benchmarks -- with and without filtering -- audio yields clear gains on tasks requiring speech comprehension or cross-modal grounding, while vision-centric suites remain largely unaffected. Our results show that speech encoders play a larger role in video understanding than current benchmarks suggest. We will fully open-source our work at this https URL.
>
---
#### [replaced 124] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出EgoDex数据集，解决仿生操作数据稀缺问题。通过收集大量第一视角视频及手部姿态数据，支持模仿学习，推动机器人、视觉和模型发展。**

- **链接: [https://arxiv.org/pdf/2505.11709](https://arxiv.org/pdf/2505.11709)**

> **作者:** Ryan Hoque; Peide Huang; David J. Yoon; Mouli Sivapurapu; Jian Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** Imitation learning for manipulation has a well-known data scarcity problem. Unlike natural language and 2D computer vision, there is no Internet-scale corpus of data for dexterous manipulation. One appealing option is egocentric human video, a passively scalable data source. However, existing large-scale datasets such as Ego4D do not have native hand pose annotations and do not focus on object manipulation. To this end, we use Apple Vision Pro to collect EgoDex: the largest and most diverse dataset of dexterous human manipulation to date. EgoDex has 829 hours of egocentric video with paired 3D hand and finger tracking data collected at the time of recording, where multiple calibrated cameras and on-device SLAM can be used to precisely track the pose of every joint of each hand. The dataset covers a wide range of diverse manipulation behaviors with everyday household objects in 194 different tabletop tasks ranging from tying shoelaces to folding laundry. Furthermore, we train and systematically evaluate imitation learning policies for hand trajectory prediction on the dataset, introducing metrics and benchmarks for measuring progress in this increasingly important area. By releasing this large-scale dataset, we hope to push the frontier of robotics, computer vision, and foundation models. EgoDex is publicly available for download at this https URL.
>
---
#### [replaced 125] MAP-based Problem-Agnostic diffusion model for Inverse Problems
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.15128](https://arxiv.org/pdf/2501.15128)**

> **作者:** Pingping Tao; Haixia Liu; Jing Su
>
> **备注:** 26 pages, 13 figures
>
> **摘要:** Diffusion models have indeed shown great promise in solving inverse problems in image processing. In this paper, we propose a novel, problem-agnostic diffusion model called the maximum a posteriori (MAP)-based guided term estimation method for inverse problems. To leverage unconditionally pretrained diffusion models to address conditional generation tasks, we divide the conditional score function into two terms according to Bayes' rule: an unconditional score function (approximated by a pretrained score network) and a guided term, which is estimated using a novel MAP-based method that incorporates a Gaussian-type prior of natural images. This innovation allows us to better capture the intrinsic properties of the data, leading to improved performance. Numerical results demonstrate that our method preserves contents more effectively compared to state-of-the-art methods--for example, maintaining the structure of glasses in super-resolution tasks and producing more coherent results in the neighborhood of masked regions during inpainting.
>
---
#### [replaced 126] LMOD+: A Comprehensive Multimodal Dataset and Benchmark for Developing and Evaluating Multimodal Large Language Models in Ophthalmology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25620](https://arxiv.org/pdf/2509.25620)**

> **作者:** Zhenyue Qin; Yang Liu; Yu Yin; Jinyu Ding; Haoran Zhang; Anran Li; Dylan Campbell; Xuansheng Wu; Ke Zou; Tiarnan D. L. Keenan; Emily Y. Chew; Zhiyong Lu; Yih Chung Tham; Ninghao Liu; Xiuzhen Zhang; Qingyu Chen
>
> **备注:** ACM Transactions on Computing for Healthcare
>
> **摘要:** Vision-threatening eye diseases pose a major global health burden, with timely diagnosis limited by workforce shortages and restricted access to specialized care. While multimodal large language models (MLLMs) show promise for medical image interpretation, advancing MLLMs for ophthalmology is hindered by the lack of comprehensive benchmark datasets suitable for evaluating generative models. We present a large-scale multimodal ophthalmology benchmark comprising 32,633 instances with multi-granular annotations across 12 common ophthalmic conditions and 5 imaging modalities. The dataset integrates imaging, anatomical structures, demographics, and free-text annotations, supporting anatomical structure recognition, disease screening, disease staging, and demographic prediction for bias evaluation. This work extends our preliminary LMOD benchmark with three major enhancements: (1) nearly 50% dataset expansion with substantial enlargement of color fundus photography; (2) broadened task coverage including binary disease diagnosis, multi-class diagnosis, severity classification with international grading standards, and demographic prediction; and (3) systematic evaluation of 24 state-of-the-art MLLMs. Our evaluations reveal both promise and limitations. Top-performing models achieved ~58% accuracy in disease screening under zero-shot settings, and performance remained suboptimal for challenging tasks like disease staging. We will publicly release the dataset, curation pipeline, and leaderboard to potentially advance ophthalmic AI applications and reduce the global burden of vision-threatening diseases.
>
---
#### [replaced 127] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18853](https://arxiv.org/pdf/2602.18853)**

> **作者:** Dong Zhao; Qi Zang; Nan Pu; Wenjing Li; Nicu Sebe; Zhun Zhong
>
> **摘要:** Domain Generalization in Semantic Segmentation (DG-SS) aims to enable segmentation models to perform robustly in unseen environments. However, conventional DG-SS methods are restricted to a fixed set of known categories, limiting their applicability in open-world scenarios. Recent progress in Vision-Language Models (VLMs) has advanced Open-Vocabulary Semantic Segmentation (OV-SS) by enabling models to recognize a broader range of concepts. Yet, these models remain sensitive to domain shifts and struggle to maintain robustness when deployed in unseen environments, a challenge that is particularly severe in urban-driving scenarios. To bridge this gap, we introduce Open-Vocabulary Domain Generalization in Semantic Segmentation (OVDG-SS), a new setting that jointly addresses unseen domains and unseen categories. We introduce the first benchmark for OVDG-SS in autonomous driving, addressing a previously unexplored problem and covering both synthetic-to-real and real-to-real generalization across diverse unseen domains and unseen categories. In OVDG-SS, we observe that domain shifts often distort text-image correlations in pre-trained VLMs, which hinders the performance of OV-SS models. To tackle this challenge, we propose S2-Corr, a state-space-driven text-image correlation refinement mechanism that mitigates domain-induced distortions and produces more consistent text-image correlations under distribution changes. Extensive experiments on our constructed benchmark demonstrate that the proposed method achieves superior cross-domain performance and efficiency compared to existing OV-SS approaches.
>
---
#### [replaced 128] The Algorithmic Gaze of Image Quality Assessment: An Audit and Trace Ethnography of the LAION-Aesthetics Predictor
- **分类: cs.HC; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.09896](https://arxiv.org/pdf/2601.09896)**

> **作者:** Jordan Taylor; William Agnew; Maarten Sap; Sarah E. Fox; Haiyi Zhu
>
> **备注:** To Appear at FAccT 2026
>
> **摘要:** Visual generative AI models are trained using a one-size-fits-all measure of aesthetic appeal. However, what is deemed "aesthetic" is inextricably linked to personal taste and cultural values, raising the question of whose taste is represented in visual generative AI models. In this work, we study an aesthetic evaluation model--LAION-Aesthetics Predictor (LAP)--that is widely used to curate datasets to train visual generative image models, like Stable Diffusion, and evaluate the quality of AI-generated images. To understand what LAP measures, we audited the model across three datasets. First, we examined the impact of aesthetic filtering on the LAION-Aesthetics Dataset (approximately 1.2B images), which was curated from LAION-5B using LAP. We find that the LAP disproportionally filters in images with captions mentioning women, while filtering out images with captions mentioning men or LGBTQ+ people. Then, we used LAP to score approximately 330k images across two art datasets, finding the model rates realistic images of landscapes, cityscapes, and portraits from western and Japanese artists most highly. In doing so, the algorithmic gaze of this aesthetic evaluation model reinforces the imperial and male gazes found within western art history. In order to understand where these biases may have originated, we performed a digital ethnography of public materials related to the creation of LAP. We find that the development of LAP reflects the biases we found in our audits, such as the aesthetic scores used to train LAP primarily coming from English-speaking photographers and western AI-enthusiasts. In response, we discuss how aesthetic evaluation can perpetuate representational harms and call on AI developers to shift away from prescriptive measures of "aesthetics" toward more pluralistic evaluation.
>
---
#### [replaced 129] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18734](https://arxiv.org/pdf/2511.18734)**

> **作者:** Keyang Lu; Sifan Zhou; Hongbin Xu; Gang Xu; Zhifei Yang; Yikai Wang; Zhen Xiao; Jieyi Long; Ming Li
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Realistic 3D city generation is fundamental to a wide range of applications, including virtual reality and digital twins. However, most existing methods rely on training a single diffusion model, which limits their ability to generate personalized and boundless city-scale scenes. In this paper, we present Yo'City, a novel agentic framework that enables user-customized and infinitely expandable 3D city generation by leveraging the reasoning and compositional capabilities of off-the-shelf large models. Specifically, Yo'City first conceptualizes the city through a top-down planning strategy that defines a hierarchical "City-District-Grid" structure. The Global Planner determines the overall layout and potential functional districts, while the Local Designer further refines each district with detailed grid-level descriptions. Subsequently, the grid-level 3D generation is achieved through a "produce-refine-evaluate" isometric image synthesis loop, followed by image-to-3D generation. To simulate continuous city evolution, Yo'City further introduces a user-interactive, relationship-guided expansion mechanism, which performs scene graph-based distance- and semantics-aware layout optimization, ensuring spatially coherent city growth. To comprehensively evaluate our method, we construct a diverse benchmark dataset and design six multi-dimensional metrics that assess generation quality from the perspectives of semantics, geometry, texture, and layout. Extensive experiments demonstrate that Yo'City consistently outperforms existing state-of-the-art methods across all evaluation aspects.
>
---
#### [replaced 130] Transforming H&E images into IHC: A Variance-Penalized GAN for Precision Oncology
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.18371](https://arxiv.org/pdf/2506.18371)**

> **作者:** Sara Rehmat; Hafeez Ur Rehman; Byeong-Gwon Kang; Sarra Ayouni; Yunyoung Nam
>
> **摘要:** The overexpression of the human epidermal growth factor receptor 2 (HER2) in breast cells is a key driver of HER2-positive breast cancer, a highly aggressive subtype requiring precise diagnosis and targeted therapy. Immunohistochemistry (IHC) is the standard technique for HER2 assessment but is costly, labor-intensive, and highly dependent on antibody selection. In contrast, hematoxylin and eosin (H&E) staining, a routine histopathological procedure, offers broader accessibility but lacks HER2 specificity. This study proposes an advanced deep learning-based image translation framework to generate high-fidelity IHC images from H&E-stained tissue samples, enabling cost-effective and scalable HER2 assessment. By modifying the loss function of pyramid pix2pix, we mitigate mode collapse, a fundamental limitation in generative adversarial networks (GANs), and introduce a novel variance-based penalty that enforces structural diversity in generated images. Our model particularly excels in translating HER2-positive (IHC 3+) images, which have remained challenging for existing methods. Quantitative evaluations on the overall BCI dataset reveal that our approach outperforms baseline models, achieving a peak signal-to-noise ratio (PSNR) of 22.16, a structural similarity index (SSIM) of 0.47, and a Fréchet Inception Distance (FID) of 346.37. In comparison, the pyramid pix2pix baseline attained PSNR 21.15, SSIM 0.43, and FID 516.75, while the standard pix2pix model yielded PSNR 20.74, SSIM 0.44, and FID 472.6. These results affirm the superior fidelity and realism of our generated IHC images. Beyond medical imaging, our model exhibits superior performance in general image-to-image translation tasks, showcasing its potential across multiple domains. This work marks a significant step toward AI-driven precision oncology, offering a reliable and efficient alternative to traditional HER2 diagnostics.
>
---
#### [replaced 131] M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于移动操作任务，解决复杂环境中操作效率与鲁棒性问题。提出M4Diffuser框架，结合多视角扩散策略与新型QP控制器，提升任务成功率和安全性。**

- **链接: [https://arxiv.org/pdf/2509.14980](https://arxiv.org/pdf/2509.14980)**

> **作者:** Ju Dong; Lei Zhang; Liding Zhang; Yao Ling; Yu Fu; Kaixin Bai; Zoltán-Csaba Márton; Zhenshan Bing; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** Project page: this https URL, 10 pages, 9 figures
>
> **摘要:** Mobile manipulation requires the coordinated control of a mobile base and a robotic arm while simultaneously perceiving both global scene context and fine-grained object details. Existing single-view approaches often fail in unstructured environments due to limited fields of view, exploration, and generalization abilities. Moreover, classical controllers, although stable, struggle with efficiency and manipulability near singularities. To address these challenges, we propose M4Diffuser, a hybrid framework that integrates a Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP (ReM-QP) controller for mobile manipulation. The diffusion policy leverages proprioceptive states and complementary camera perspectives with both close-range object details and global scene context to generate task-relevant end-effector goals in the world frame. These high-level goals are then executed by the ReM-QP controller, which eliminates slack variables for computational efficiency and incorporates manipulability-aware preferences for robustness near singularities. Comprehensive experiments in simulation and real-world environments show that M4Diffuser achieves 7 to 56 percent higher success rates and reduces collisions by 3 to 31 percent over baselines. Our approach demonstrates robust performance for smooth whole-body coordination, and strong generalization to unseen tasks, paving the way for reliable mobile manipulation in unstructured environments. Details of the demo and supplemental material are available on our project website this https URL.
>
---
#### [replaced 132] Detecting AI-Generated Images via Diffusion Snap-Back Reconstruction: A Forensic Approach
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.00352](https://arxiv.org/pdf/2511.00352)**

> **作者:** Mohd Ruhul Ameen; Akif Islam
>
> **备注:** Accepted at the 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence and Networking (QPAIN 2026)
>
> **摘要:** The rapid advancement of generative image models has transformed digital media to the point where AI generated images can no longer be reliably distinguished from authentic photographs by human observers or many conventional detection methods. Modern text to image systems such as Stable Diffusion and DALL E can now generate images so realistic that they often appear completely natural, leaving little to no visible artifacts for traditional deepfake detectors to rely on. This challenge has practical consequences for misinformation control, institutional identity verification, and digital trust in political and legal contexts. Instead of searching for hidden pixel level traces, we take a different approach: we observe how an image responds when it is gently disturbed and reconstructed by a diffusion model. We call this behavior diffusion snap back. By tracking how perceptual similarity measures (LPIPS, SSIM, and PSNR) change across different reconstruction strengths, we capture compact and interpretable signals that reveal how closely an image aligns with the diffusion model's learned denoising behavior. Evaluated on a balanced dataset of 4,000 human and AI generated images, the proposed method achieves an AUROC of 0.993 under stratified five fold cross validation and 0.990 on a holdout split using only logistic regression. Initial robustness tests show that the method remains stable under common real world distortions such as image compression and added noise. Although our experiments were conducted using a single diffusion backbone, the results indicate that reconstruction behavior can serve as a reliable and scalable foundation for synthetic media detection as generative models continue to grow more realistic.
>
---
#### [replaced 133] InfScene-SR: Arbitrary-Size Image Super-Resolution via Iterative Joint-Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19736](https://arxiv.org/pdf/2602.19736)**

> **作者:** Shoukun Sun; Zhe Wang; Xiang Que; Jiyin Zhang; Xiaogang Ma
>
> **摘要:** While diffusion models have achieved state-of-the-art performance in Image Super-Resolution (SR), their prohibitive computational and memory demands restrict their training and inference to fixed-size inputs. The standard workaround to super-resolve larger images relies on partitioning the image, super-resolving patches independently, and stitching them together -- a process that inevitably introduces severe boundary artifacts and spatial inconsistencies in large-scale scenes. To achieve spatially continuous, arbitrary-size image super-resolution, we propose InfScene-SR, a diffusion-based SR approach. Building upon SR3, our approach leverages Variance-Corrected Fusion (VCF) to perform joint-denoising across overlapping patches. VCF guarantees continuous transitions while preserving the stochastic variance crucial for high-fidelity texture reconstruction. To overcome the prohibitive synchronization overhead of scaling joint-denoising to gigapixel imagery, we introduce Spatially-Decoupled Variance Correction (SDVC). SDVC reformulates the global fusion process into independent, atomic patch operations, drastically reducing memory complexity to $\mathcal{O}(1)$ and naturally enabling fully distributed, parallelized inference. Extensive experiments on large-scale remote sensing datasets demonstrate that InfScene-SR strictly eliminates boundary seams, achieves superior perceptual quality, and significantly boosts performance in downstream semantic segmentation task.
>
---
#### [replaced 134] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.17788](https://arxiv.org/pdf/2503.17788)**

> **作者:** Gaoge Han; Yongkang Cheng; Zhe Chen; Shaoli Huang; Tongliang Liu
>
> **备注:** Accepted by CVPR 2026 Main, Project: this https URL
>
> **摘要:** Two-hand reconstruction from monocular images is hampered by complex poses and severe occlusions, which often cause interaction misalignment and two-hand penetration. We address this by decoupling the problem into 2D structural alignment and 3D spatial interaction alignment, each handled by a tailored component. For 2D alignment, we pioneer the attempt to unify heterogeneous structural priors (keypoints, segmentation, and depth) from vision foundation models as complementary structured guidance for two-hand recovery. Instead of extracting priors prediction as explicit inputs, we propose a fusion-alignment encoder that absorbs their structural knowledge implicitly, achieving foundation-level guidance without foundation-level cost. For 3D spatial alignment, we propose a two-hand penetration-free diffusion model that learns a generative mapping from interpenetrated poses to realistic, collision-free configurations. Guided by collision gradients during denoising, the model converges toward the manifold of valid two-hand interactions, preserving geometric and kinematic coherence. This generative formulation approach enables physically credible reconstructions even under occlusion or ambiguous visual input. Extensive experiments on InterHand2.6M and HIC show state-of-the-art or leading performance in interaction alignment and penetration suppression. Project: this https URL
>
---
#### [replaced 135] QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23681](https://arxiv.org/pdf/2509.23681)**

> **作者:** Weilun Feng; Chuanguang Yang; Haotong Qin; Mingqiang Wu; Yuqi Li; Xiangqi Li; Zhulin An; Libo Huang; Yulun Zhang; Michele Magno; Yongjun Xu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Diffusion transformers exhibit remarkable video generation capability, yet their prohibitive computational and memory costs hinder practical deployment. Model quantization and attention sparsification are two promising directions for compression, but each alone suffers severe performance degradation under aggressive compression. Combining them promises compounded efficiency gains, but naive integration is ineffective. The sparsity-induced information loss exacerbates quantization noise, leading to amplified attention shifts. To address this, we propose \textbf{QuantSparse}, a unified framework that integrates model quantization with attention sparsification. Specifically, we introduce \textit{Multi-Scale Salient Attention Distillation}, which leverages both global structural guidance and local salient supervision to mitigate quantization-induced bias. In addition, we develop \textit{Second-Order Sparse Attention Reparameterization}, which exploits the temporal stability of second-order residuals to efficiently recover information lost under sparsity. Experiments on HunyuanVideo-13B demonstrate that QuantSparse achieves 20.88 PSNR, substantially outperforming the state-of-the-art quantization baseline Q-VDiT (16.85 PSNR), while simultaneously delivering a \textbf{3.68$\times$} reduction in storage and \textbf{1.88$\times$} acceleration in end-to-end inference. Our code will be released in this https URL.
>
---
#### [replaced 136] FVO: Fast Visual Odometry with Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03348](https://arxiv.org/pdf/2510.03348)**

> **作者:** Vlardimir Yugay; Duy-Kien Nguyen; Theo Gevers; Cees G. M. Snoek; Martin R. Oswald
>
> **摘要:** Hybrid pipelines that combine deep learning with classical optimization have established themselves as the dominant approach to visual odometry (VO). By integrating neural network predictions with bundle adjustment, these models estimate camera trajectories with high accuracy. Still, hybrid VO methods fall short of the speed and capabilities of pure end-to-end approaches. Current hybrid frameworks rely on massive, pre-trained 3D networks to predict geometry. Because these backends are trained to be scale-ambiguous and frozen rather than retrained, the pipelines essentially inherit this limitation and, by design, fails to estimate absolute scale. Furthermore, their slow optimization and post-processing steps bottleneck the pipeline's inference speed. We propose to replace post-processing entirely by formulating monocular visual odometry as a direct relative pose regression problem. This formulation enables us to train a fast, high-capacity transformer to predict relative camera poses and corresponding confidences using only camera poses as supervision. More importantly, it allows us to employ a confidence-aware inference scheme that aggregates overlapping pose predictions for robust trajectory estimation. We demonstrate on multiple visual odometry benchmarks that our method, Fast Visual Odometry (FVO), successfully leverages diverse data to achieve competitive or superior performance while being nearly 2 times faster than the fastest baselines.
>
---
#### [replaced 137] LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.17635](https://arxiv.org/pdf/2412.17635)**

> **作者:** Hao Li; Minghan Qin; Zhengyu Zou; Diqi He; Xinhao Ji; Bohan Li; Bingquan Dai; Dingewn Zhang; Junwei Han
>
> **备注:** \url{this https URL}
>
> **摘要:** Applying Gaussian Splatting to perception tasks for 3D scene understanding is becoming increasingly popular. Most existing works primarily focus on rendering 2D feature maps from novel viewpoints, which leads to an imprecise 3D language field with outlier languages, ultimately failing to align objects in 3D space. By utilizing masked images for feature extraction, these approaches also lack essential contextual information, leading to inaccurate feature representation. To this end, we propose a Language-Embedded Surface Field (LangSurf), which accurately aligns the 3D language fields with the surface of objects, facilitating precise 2D and 3D segmentation with text query, widely expanding the downstream tasks such as removal and editing. The core of LangSurf is a joint training strategy that flattens the language Gaussian on the object surfaces using geometry supervision and contrastive losses to assign accurate language features to the Gaussians of objects. In addition, we also introduce the Hierarchical-Context Awareness Module to extract features at the image level for contextual information then perform hierarchical mask pooling using masks segmented by SAM to obtain fine-grained language features in different hierarchies. Extensive experiments on open-vocabulary 2D and 3D semantic segmentation demonstrate that LangSurf outperforms the previous state-of-the-art method LangSplat by a large margin. As shown in Fig. 1, our method is capable of segmenting objects in 3D space, thus boosting the effectiveness of our approach in instance recognition, removal, and editing, which is also supported by comprehensive experiments. this https URL.
>
---
#### [replaced 138] Autoassociative Learning of Structural Representations for Modeling and Classification in Medical Imaging
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.12070](https://arxiv.org/pdf/2411.12070)**

> **作者:** Zuzanna Buchnajzer; Kacper Dobek; Stanisław Hapke; Daniel Jankowski; Krzysztof Krawiec
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Deep learning architectures based on convolutional neural networks tend to rely on continuous, smooth features. While this characteristics provides significant robustness and proves useful in many real-world tasks, it is strikingly incompatible with the physical characteristic of the world, which, at the scale in which humans operate, comprises crisp objects, typically representing well-defined categories. This study proposes a class of neurosymbolic systems that learn by reconstructing images in terms of visual primitives and are thus forced to form high-level, structural explanations of them. When applied to the task of diagnosing abnormalities in histological imaging, the method proved superior to a conventional deep learning architecture in terms of classification accuracy, while being more transparent.
>
---
#### [replaced 139] ExpGest: Expressive Speaker Generation Using Diffusion Model and Hybrid Audio-Text Guidance
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于手势生成任务，旨在解决现有方法生成手势不自然、缺乏情感的问题。提出ExpGest框架，结合音频和文本信息生成更具表现力的全身手势。**

- **链接: [https://arxiv.org/pdf/2410.09396](https://arxiv.org/pdf/2410.09396)**

> **作者:** Yongkang Cheng; Mingjiang Liang; Shaoli Huang; Gaoge Han; Jifeng Ning; Wei Liu
>
> **备注:** Accepted by ICME 2024
>
> **摘要:** Existing gesture generation methods primarily focus on upper body gestures based on audio features, neglecting speech content, emotion, and locomotion. These limitations result in stiff, mechanical gestures that fail to convey the true meaning of audio content. We introduce ExpGest, a novel framework leveraging synchronized text and audio information to generate expressive full-body gestures. Unlike AdaIN or one-hot encoding methods, we design a noise emotion classifier for optimizing adversarial direction noise, avoiding melody distortion and guiding results towards specified emotions. Moreover, aligning semantic and gestures in the latent space provides better generalization capabilities. ExpGest, a diffusion model-based gesture generation framework, is the first attempt to offer mixed generation modes, including audio-driven gestures and text-shaped motion. Experiments show that our framework effectively learns from combined text-driven motion and audio-induced gesture datasets, and preliminary results demonstrate that ExpGest achieves more expressive, natural, and controllable global motion in speakers compared to state-of-the-art models.
>
---
#### [replaced 140] CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2311.10605](https://arxiv.org/pdf/2311.10605)**

> **作者:** Yiyu Chen; Zheyi Fan; Zhaoru Chen; Yixuan Zhu
>
> **备注:** This paper is accepted by CVPR 2024
>
> **摘要:** Person re-identification (re-ID) is a challenging task that aims to learn discriminative features for person retrieval. In person re-ID, Jaccard distance is a widely used distance metric, especially in re-ranking and clustering scenarios. However, we discover that camera variation has a significant negative impact on the reliability of Jaccard distance. In particular, Jaccard distance calculates the distance based on the overlap of relevant neighbors. Due to camera variation, intra-camera samples dominate the relevant neighbors, which reduces the reliability of the neighbors by introducing intra-camera negative samples and excluding inter-camera positive samples. To overcome this problem, we propose a novel camera-aware Jaccard (CA-Jaccard) distance that leverages camera information to enhance the reliability of Jaccard distance. Specifically, we design camera-aware k-reciprocal nearest neighbors (CKRNNs) to find k-reciprocal nearest neighbors on the intra-camera and inter-camera ranking lists, which improves the reliability of relevant neighbors and guarantees the contribution of inter-camera samples in the overlap. Moreover, we propose a camera-aware local query expansion (CLQE) to mine reliable samples in relevant neighbors by exploiting camera variation as a strong constraint and assign these samples higher weights in overlap, further improving the reliability. Our CA-Jaccard distance is simple yet effective and can serve as a general distance metric for person re-ID methods with high reliability and low computational cost. Extensive experiments demonstrate the effectiveness of our method.
>
---
#### [replaced 141] Cumulative Consensus Score: Label-Free and Model-Agnostic Evaluation of Object Detectors in Deployment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12871](https://arxiv.org/pdf/2509.12871)**

> **作者:** Avinaash Manoharan; Xiangyu Yin; Domenik Helm; Chih-Hong Cheng
>
> **摘要:** Evaluating object detection models in deployment is challenging because ground-truth annotations are rarely available. We introduce the Cumulative Consensus Score (CCS), a label-free monitoring signal for continuous evaluation and comparison of detectors in real-world settings. CCS applies test-time data augmentation to each image and measures the spatial consistency of predicted bounding boxes across augmented views using Intersection over Union. The resulting consensus score serves as a proxy for reliability without requiring bounding box annotations. In controlled experiments on Open Images and KITTI, CCS achieved over 90% congruence with F1-score, Probabilistic Detection Quality, and Optimal Correction Cost, with qualitative consistency further confirmed on COCO and BDD100K across model pairs. The method is model-agnostic, working across single-stage and two-stage detectors, and operates at the case level to highlight under-performing scenarios. We also provide a simplified theoretical link between expected CCS and detection correctness. Altogether, CCS provides a robust foundation for DevOps-style monitoring of object detectors.
>
---
#### [replaced 142] iGVLM: Dynamic Instruction-Guided Vision Encoding for Question-Aware Multimodal Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.02748](https://arxiv.org/pdf/2603.02748)**

> **作者:** Hanpeng Liu; Yaqian Li; Zidan Wang; Shuoxi Zhang; Zihao Bo; Rinyoichi Takezoe; Kaiwen Long; Kun He
>
> **摘要:** Despite the success of Large Vision--Language Models (LVLMs), most existing architectures suffer from a representation bottleneck: they rely on static, instruction-agnostic vision encoders whose visual representations are utilized in an invariant manner across different textual tasks. This rigidity hinders fine-grained reasoning where task-specific visual cues are critical. To address this issue, we propose iGVLM, a general framework for instruction-guided visual modulation. iGVLM introduces a decoupled dual-branch architecture: a frozen representation branch that preserves task-agnostic visual representations learned during pre-training, and a dynamic conditioning branch that performs affine feature modulation via Adaptive Layer Normalization (AdaLN). This design enables a smooth transition from general-purpose perception to instruction-aware reasoning while maintaining the structural integrity and stability of pre-trained visual priors. Beyond standard benchmarks, we introduce MM4, a controlled diagnostic probe for quantifying logical consistency under multi-query, multi-instruction settings. Extensive results show that iGVLM consistently enhances instruction sensitivity across diverse language backbones, offering a plug-and-play paradigm for bridging passive perception and active reasoning.
>
---
#### [replaced 143] HiconAgent: History Context-aware Policy Optimization for GUI Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01763](https://arxiv.org/pdf/2512.01763)**

> **作者:** Xurui Zhou; Gongwei Chen; Yuquan Xie; Zaijing Li; Kaiwen Zhou; Shuai Wang; Shuo Yang; Zhuotao Tian; Rui Shao
>
> **摘要:** Graphical User Interface (GUI) agents require effective use of historical context to perform sequential navigation tasks. While incorporating past actions and observations can improve decision making, naive use of full history leads to excessive computational overhead and distraction from irrelevant information. To address this, we introduce HiconAgent, a GUI agent trained with History Context-aware Policy Optimization (HCPO) for efficient and effective utilization of historical information. HCPO optimizes history usage in both sampling and policy updates through two complementary components: (1) Dynamic Context Sampling (DCS) presents the agent with variable length histories during sampling, enabling adaptive use of the most relevant context; (2) Anchor-guided History Compression (AHC) refines the policy update phase with a dual branch strategy where the compressed branch removes history observations while keeping history actions as information flow anchors. The compressed and uncompressed branches are coupled through a history-enhanced alignment loss to enforce consistent history usage while maintaining efficiency. Experiments on mainstream GUI navigation benchmarks demonstrate strong performance. Despite being smaller, HiconAgent-3B outperforms GUI-R1-7B by +8.46 percent grounding accuracy and +11.32 percent step success rate on GUI-Odyssey, while achieving comparable results on AndroidControl and AITW with up to 2.47x computational speedup and 60 percent FLOPs reduction.
>
---
#### [replaced 144] Input-Adaptive Generative Dynamics in Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.15199](https://arxiv.org/pdf/2411.15199)**

> **作者:** Yucheng Xing; Xiaodong Liu; Xin Wang
>
> **备注:** 14 pages, 6 figures. Updated version with revised title
>
> **摘要:** Diffusion models typically generate data through a fixed denoising trajectory that is shared across all samples. However, generation targets can differ in complexity, suggesting that a single pre-defined diffusion process may not be optimal for every input. In this work, we investigate input-adaptive generative dynamics for diffusion models, where the generation process itself adapts to the conditions of each sample. Instead of relying on a fixed diffusion trajectory, the proposed framework allows the generative dynamics to adjust across inputs according to their generation requirements. To enable this behavior, we train the diffusion backbone under varying horizons and noise schedules, so that it can operate consistently under different input-adaptive trajectories. Experiments on conditional image generation show that diffusion trajectories can vary across inputs while maintaining generation quality and reducing the average number of sampling steps. These results provide a proof of the concept that diffusion processes can benefit from input-adaptive generative dynamics rather than relying on a single fixed trajectory.
>
---
#### [replaced 145] Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.13795](https://arxiv.org/pdf/2510.13795)**

> **作者:** Yi Zhang; Bolin Ni; Xin-Sheng Chen; Heng-Rui Zhang; Yongming Rao; Houwen Peng; Qinglin Lu; Han Hu; Meng-Hao Guo; Shi-Min Hu
>
> **备注:** homepage: this https URL
>
> **摘要:** Fully open multimodal large language models (MLLMs) currently lag behind proprietary counterparts, primarily due to a significant gap in data quality for supervised fine-tuning (SFT). Existing open-source datasets are often plagued by widespread noise and a critical deficit in complex reasoning data, such as Chain-of-Thought (CoT), which hinders the development of advanced model capabilities. Addressing these challenges, our work makes three primary contributions. First, we introduce Honey-Data-15M, a new SFT dataset comprising approximately 15 million QA pairs, processed through multiple cleaning techniques and enhanced with a novel dual-level (short and long) CoT enrichment strategy. Second, we introduce HoneyPipe, the data curation pipeline, and its underlying framework DataStudio, providing the community with a transparent and adaptable methodology for data curation that moves beyond static dataset releases. Finally, to validate our dataset and pipeline, we train Bee-8B, an 8B model on Honey-Data-15M. Experiments show that Bee-8B establishes a new state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is competitive with, and in some cases surpasses, recent semi-open models such as InternVL3.5-8B. Our work delivers to the community a suite of foundational resources, including: the Honey-Data-15M corpus; the full-stack suite comprising HoneyPipe and DataStudio; training recipes; an evaluation harness; and the model weights. This effort demonstrates that a principled focus on data quality is a key pathway to developing fully open MLLMs that are highly competitive with their semi-open counterparts.
>
---
#### [replaced 146] MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MCGS-SLAM，用于高保真建图的多相机SLAM系统，解决单目系统鲁棒性差、覆盖范围小的问题，通过RGB输入和高斯点云优化实现精准定位与重建。**

- **链接: [https://arxiv.org/pdf/2509.14191](https://arxiv.org/pdf/2509.14191)**

> **作者:** Zhihao Cao; Hanyu Wu; Li Wa Tang; Zizhou Luo; Wei Zhang; Marc Pollefeys; Zihan Zhu; Martin R. Oswald
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
>
---
#### [replaced 147] Streaming Drag-Oriented Interactive Video Manipulation: Drag Anything, Anytime!
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03550](https://arxiv.org/pdf/2510.03550)**

> **作者:** Junbao Zhou; Yuan Zhou; Kesen Zhao; Qingshan Xu; Beier Zhu; Richang Hong; Hanwang Zhang
>
> **摘要:** Achieving streaming, fine-grained control over the outputs of autoregressive video diffusion models remains challenging, making it difficult to ensure that they consistently align with user expectations. To bridge this gap, we propose \textbf{stReaming drag-oriEnted interactiVe vidEo manipuLation (REVEL)}, a new task that enables users to modify generated videos \emph{anytime} on \emph{anything} via fine-grained, interactive drag. Beyond DragVideo and SG-I2V, REVEL unifies drag-style video manipulation as editing and animating video frames with both supporting user-specified translation, deformation, and rotation effects, making drag operations versatile. In resolving REVEL, we observe: \emph{i}) drag-induced perturbations accumulate in latent space, causing severe latent distribution drift that halts the drag process; \emph{ii}) streaming drag is easily disturbed by context frames, thereby yielding visually unnatural outcomes. We thus propose a training-free approach, \textbf{DragStream}, comprising: \emph{i}) an adaptive distribution self-rectification strategy that leverages neighboring frames' statistics to effectively constrain the drift of latent embeddings; \emph{ii}) a spatial-frequency selective optimization mechanism, allowing the model to fully exploit contextual information while mitigating its interference via selectively propagating visual cues along generation. Our method can be seamlessly integrated into existing autoregressive video diffusion models, and extensive experiments firmly demonstrate the effectiveness of our DragStream.
>
---
#### [replaced 148] Deep Unrolled Meta-Learning for Multi-Coil and Multi-Modality MRI with Adaptive Optimization
- **分类: math.OC; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.11518](https://arxiv.org/pdf/2505.11518)**

> **作者:** Merham Fouladvand; Peuroly Batra
>
> **摘要:** We propose a unified deep meta-learning framework for accelerated magnetic resonance imaging (MRI) that jointly addresses multi-coil reconstruction and cross-modality synthesis. Motivated by the limitations of conventional methods in handling undersampled data and missing modalities, our approach unrolls a provably convergent optimization algorithm into a structured neural network architecture. Each phase of the network mimics a step of an adaptive forward-backward scheme with extrapolation, enabling the model to incorporate both data fidelity and nonconvex regularization in a principled manner. To enhance generalization across different acquisition settings, we integrate meta-learning, which enables the model to rapidly adapt to unseen sampling patterns and modality combinations using task-specific meta-knowledge. The proposed method is evaluated on the open source datasets, showing significant improvements in PSNR and SSIM over conventional supervised learning, especially under aggressive undersampling and domain shifts. Our results demonstrate the synergy of unrolled optimization, task-aware meta-learning, and modality fusion, offering a scalable and generalizable solution for real-world clinical MRI reconstruction.
>
---
#### [replaced 149] Automated Pest Counting in Water Traps through Active Robotic Stirring for Occlusion Handling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动化 pest 计数任务，解决 occlusion 导致的计数不准问题。通过主动机械搅拌和自适应控制，提升计数准确性和效率。**

- **链接: [https://arxiv.org/pdf/2510.21732](https://arxiv.org/pdf/2510.21732)**

> **作者:** Xumin Gao; Mark Stevens; Grzegorz Cielniak
>
> **摘要:** Existing image-based pest counting methods rely on single static images and often produce inaccurate results under occlusion. To address this issue, this paper proposes an automated pest counting method in water traps through active robotic stirring. First, an automated robotic arm-based stirring system is developed to redistribute pests and reveal occluded individuals for counting. Then, the effects of different stirring patterns on pest counting performance are investigated. Six stirring patterns are designed and evaluated across different pest density scenarios to identify the optimal one. Finally, a heuristic counting confidence-driven closed-loop control system is proposed for adaptive-speed robotic stirring, adjusting the stirring speed based on the average change rate of counting confidence between consecutive frames. Experimental results show that the four circles is the optimal stirring pattern, achieving the lowest overall mean absolute counting error of 4.384 and the highest overall mean counting confidence of 0.721. Compared with constant-speed stirring, adaptive-speed stirring reduces task execution time by up to 44.7% and achieves more stable performance across different pest density scenarios. Moreover, the proposed pest counting method reduces the mean absolute counting error by up to 3.428 compared to the single static image counting method under high-density scenarios where occlusion is severe.
>
---
#### [replaced 150] Latent Equivariant Operators for Robust Object Recognition: Promise and Challenges
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.18406](https://arxiv.org/pdf/2602.18406)**

> **作者:** Minh Dinh; Stéphane Deny
>
> **备注:** Version accepted at GrAM Workshop of ICLR 2026, Tiny Paper Track
>
> **摘要:** Despite the successes of deep learning in computer vision, difficulties persist in recognizing objects that have undergone group-symmetric transformations rarely seen during training$\unicode{x2013}$for example objects seen in unusual poses, scales, positions, or combinations thereof. Equivariant neural networks are a solution to the problem of generalizing across symmetric transformations, but require knowledge of transformations a priori. An alternative family of architectures proposes to learn equivariant operators in a latent space, from examples of symmetric transformations. Here, using simple datasets of rotated and translated noisy MNIST, we illustrate how such architectures can successfully be harnessed for out-of-distribution classification, thus overcoming the limitations of both traditional and equivariant networks. While conceptually enticing, we discuss challenges ahead on the path of scaling these architectures to more complex datasets. Our code is available at this https URL.
>
---
#### [replaced 151] Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.02732](https://arxiv.org/pdf/2412.02732)**

> **作者:** Daniela Szwarcman; Sujit Roy; Paolo Fraccaro; Þorsteinn Elí Gíslason; Benedikt Blumenstiel; Rinki Ghosal; Pedro Henrique de Oliveira; Joao Lucas de Sousa Almeida; Rocco Sedona; Yanghui Kang; Srija Chakraborty; Sizhe Wang; Carlos Gomes; Ankur Kumar; Myscon Truong; Denys Godwin; Hyunho Lee; Chia-Yu Hsu; Rohit Lal; Ata Akbari Asanjan; Besart Mujeci; Disha Shidham; Trevor Keenan; Paulo Arevalo; Wenwen Li; Hamed Alemohammad; Pontus Olofsson; Christopher Hain; Robert Kennedy; Bianca Zadrozny; David Bell; Gabriele Cavallaro; Campbell Watson; Manil Maskey; Rahul Ramachandran; Juan Bernabe Moreno
>
> **摘要:** This paper presents Prithvi-EO-2.0, a new geospatial foundation model that offers significant improvements over its predecessor, Prithvi-EO-1.0. Trained on 4.2 million global time series samples from NASA's Harmonized Landsat and Sentinel-2 data archive at 30-m resolution, the new model incorporates temporal and location embeddings for enhanced performance across various geospatial tasks. Through extensive benchmarking with GEO-Bench, the model outperforms the previous Prithvi-EO model by 8% across a range of tasks. It also outperforms six other geospatial foundation models when benchmarked on remote sensing tasks from different domains and resolutions (i.e. from 0.1 m to 15 m). The results demonstrate the versatility of the model in both classical Earth observation and high-resolution applications. Early involvement of end-users and subject matter experts (SMEs) allowed constant feedback on model and dataset design, enabling customization across diverse SME-led applications in disaster response, land cover and crop mapping, and ecosystem dynamics monitoring. Prithvi-EO-2.0 is available as an open-source model on Hugging Face and IBM TerraTorch, with additional resources on GitHub. The project exemplifies the Trusted Open Science approach embraced by all involved organizations.
>
---
#### [replaced 152] Empowering Microscopic Traffic Simulators with Realistic Perception using Surrogate Sensor Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02858](https://arxiv.org/pdf/2508.02858)**

> **作者:** Tianheng Zhu; Yiheng Feng
>
> **备注:** 27 pages, 8 figures
>
> **摘要:** Simulation is central to the evaluation of intelligent transportation system (ITS) applications. As ITS increasingly incorporates autonomous vehicle (AV) technologies as fleet vehicles and/or mobile sensors, accurate modeling of their perception capabilities becomes essential in high-fidelity simulations. While game-engine-based simulators reproduce realistic perception environments through 3D scene rendering and raw sensor data generation, they face scalability challenges in simulating traffic networks with a large number of AVs due to high computational cost. In contrast, microscopic traffic simulators (MTS) can scale efficiently but lack perception modeling capabilities. To bridge this gap, we propose MIDAR, a surrogate LiDAR detection model that mimics realistic LiDAR detections using only high-level features readily available from MTS. Specifically, MIDAR predicts true-positive and false-negative LiDAR detections based on the relative positions and dimensions of surrounding objects. To capture LiDAR visibility and occlusion effects, MIDAR introduces a ray-hit feature and a Refined Multi-hop Line-of-Sight (RM-LoS) graph processed by a geometry-aware Graph Transformer. MIDAR achieves an AUC of 0.94 in approximating LiDAR detection results using CARLA-generated point cloud data, and an AUC of 0.86 with real-world data from the nuScenes dataset. Two ITS applications, cooperative-perception-based adaptive signal control and vehicle trajectory reconstruction, are integrated with MIDAR to further validate its realism and necessity. Results show that MIDAR generates more realistic detection outputs as well as application-level performance metrics than simplified perception models while introducing minimal computational overhead, enabling seamless integration into large-scale, real-time traffic simulations. The code and data are publicly available at this https URL.
>
---
#### [replaced 153] Motion-Aware Transformer for Multi-Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21715](https://arxiv.org/pdf/2509.21715)**

> **作者:** Xu Yang; Gady Agam
>
> **摘要:** Multi-object tracking (MOT) in videos remains challenging due to complex object motions and crowded scenes. Recent DETR-based frameworks offer end-to-end solutions but typically process detection and tracking queries jointly within a single Transformer Decoder layer, leading to conflicts and degraded association accuracy. We introduce the Motion-Aware Transformer (MATR), which explicitly predicts object movements across frames to update track queries in advance. By reducing query collisions, MATR enables more consistent training and improves both detection and association. Extensive experiments on DanceTrack, SportsMOT, and BDD100k show that MATR delivers significant gains across standard metrics. On DanceTrack, MATR improves HOTA by more than 9 points over MOTR without additional data and reaches a new state-of-the-art score of 71.3 with supplementary data. MATR also achieves state-of-the-art results on SportsMOT (72.2 HOTA) and BDD100k (54.7 mTETA, 41.6 mHOTA) without relying on external datasets. These results demonstrate that explicitly modeling motion within end-to-end Transformers offers a simple yet highly effective approach to advancing multi-object tracking.
>
---
#### [replaced 154] From Semantic To Instance: A Semi-Self-Supervised Learning Approach
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.16563](https://arxiv.org/pdf/2506.16563)**

> **作者:** Keyhan Najafian; Farhad Maleki; Lingling Jin; Ian Stavness
>
> **摘要:** Instance segmentation is essential for applications such as automated monitoring of plant health, growth, and yield. However, extensive effort is required to create large-scale datasets with pixel-level annotations of each object instance for developing instance segmentation models that restrict the use of deep learning in these areas. This challenge is more significant in images with densely packed, self-occluded objects, which are common in agriculture. To address this challenge, we propose a semi-self-supervised learning approach that requires minimal manual annotation to develop a high-performing instance segmentation model. We design GLMask, an image-mask representation for the model to focus on shape, texture, and pattern while minimizing its dependence on color features. We develop a pipeline to generate semantic segmentation and then transform it into instance-level segmentation. The proposed approach substantially outperforms the conventional instance segmentation models, establishing a state-of-the-art wheat head instance segmentation model with mAP@50 of 98.5%. Additionally, we assessed the proposed methodology on the general-purpose Microsoft COCO dataset, achieving a significant performance improvement of over 12.6% mAP@50. This highlights that the utility of our proposed approach extends beyond precision agriculture and applies to other domains, specifically those with similar data characteristics.
>
---
#### [replaced 155] Unified Medical Image Segmentation with State Space Modeling Snake
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.12760](https://arxiv.org/pdf/2507.12760)**

> **作者:** Ruicheng Zhang; Haowei Guo; Kanghui Tian; Jun Zhou; Mingliang Yan; Zeyu Zhang; Shen Zhao
>
> **备注:** This paper has been accepted by ACM MM 2025
>
> **摘要:** Unified Medical Image Segmentation (UMIS) is critical for comprehensive anatomical assessment but faces challenges due to multi-scale structural heterogeneity. Conventional pixel-based approaches, lacking object-level anatomical insight and inter-organ relational modeling, struggle with morphological complexity and feature conflicts, limiting their efficacy in UMIS. We propose Mamba Snake, a novel deep snake framework enhanced by state space modeling for UMIS. Mamba Snake frames multi-contour evolution as a hierarchical state space atlas, effectively modeling macroscopic inter-organ topological relationships and microscopic contour refinements. We introduce a snake-specific vision state space module, the Mamba Evolution Block (MEB), which leverages effective spatiotemporal information aggregation for adaptive refinement of complex morphologies. Energy map shape priors further ensure robust long-range contour evolution in heterogeneous data. Additionally, a dual-classification synergy mechanism is incorporated to concurrently optimize detection and segmentation, mitigating under-segmentation of microstructures in UMIS. Extensive evaluations across five clinical datasets reveal Mamba Snake's superior performance, with an average Dice improvement of 3\% over state-of-the-art methods.
>
---
#### [replaced 156] Efficient Semi-Supervised Adversarial Training via Latent Clustering-Based Data Reduction
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.10466](https://arxiv.org/pdf/2501.10466)**

> **作者:** Somrita Ghosh; Yuelin Xu; Xiao Zhang
>
> **备注:** Shorter version of this work accepted by NextGenAISafety Workshop at ICML 2024
>
> **摘要:** Learning robust models under adversarial settings is widely recognized as requiring a considerably large number of training samples. Recent work proposes semi-supervised adversarial training (SSAT), which utilizes external unlabeled or synthetically generated data and is currently the state of the art. However, SSAT requires substantial extra data to attain high robustness, resulting in prolonged training time and increased memory usage. In this paper, we propose data reduction strategies to improve the efficiency of SSAT by optimizing the amount of additional data incorporated. Specifically, we design novel latent clustering-based techniques to select or generate a small, critical subset of data samples near the model's decision boundary. While focusing on boundary-adjacent points, our methods maintain a balanced ratio between boundary and non-boundary data points, thereby avoiding overfitting. Comprehensive experiments across image benchmarks demonstrate that our methods can effectively reduce SSAT's data requirements and computational costs while preserving its strong robustness advantages. In particular, our latent-space selection scheme based on k-means clustering and our guided diffusion-based approach with LCG-KM are the most effective, achieving nearly identical robust accuracies with 5 times to 10 times less unlabeled data. When compared to full SSAT trained to convergence, our methods reduce total runtime by approximately 3 times to 4 times due to strategic prioritization of unlabeled data.
>
---
#### [replaced 157] DINOv3 Visual Representations for Blueberry Perception Toward Robotic Harvesting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02419](https://arxiv.org/pdf/2603.02419)**

> **作者:** Rui-Feng Wang; Daniel Petti; Yue Chen; Changying Li
>
> **备注:** 16 pages, 9 figures, 5 tables
>
> **摘要:** Vision Foundation Models trained via large-scale self-supervised learning have demonstrated strong generalization in visual perception; however, their practical role and performance limits in agricultural settings remain insufficiently understood. This work evaluates DINOv3 as a frozen backbone for blueberry robotic harvesting-related visual tasks, including fruit and bruise segmentation, as well as fruit and cluster detection. Under a unified protocol with lightweight decoders, segmentation benefits consistently from stable patch-level representations and scales with backbone size. In contrast, detection is constrained by target scale variation, patch discretization, and localization compatibility. The failure of cluster detection highlights limitations in modeling relational targets defined by spatial aggregation. Overall, DINOv3 is best viewed not as an end-to-end task model, but as a semantic backbone whose effectiveness depends on downstream spatial modeling aligned with fruit-scale and aggregation structures, providing guidance for blueberry robotic harvesting. Code and dataset will be available upon acceptance.
>
---
#### [replaced 158] Stable Multi-Drone GNSS Tracking System for Marine Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于海洋机器人跟踪任务，解决水下GNSS信号失效问题。通过多无人机协同，结合视觉与EKF算法，实现稳定实时定位与多机一致性跟踪。**

- **链接: [https://arxiv.org/pdf/2511.18694](https://arxiv.org/pdf/2511.18694)**

> **作者:** Shuo Wen; Edwin Meriaux; Mariana Sosa Guzmán; Zhizun Wang; Junming Shi; Gregory Dudek
>
> **摘要:** Stable and accurate tracking is essential for marine robotics, yet Global Navigation Satellite System (GNSS) signals vanish immediately below the sea surface. Traditional alternatives suffer from error accumulation, high computational demands, or infrastructure dependence. In this work, we present a multi-drone GNSS-based tracking system for surface and near-surface marine robots. Our approach combines efficient visual detection, lightweight multi-object tracking, GNSS-based triangulation, and a confidence-weighted Extended Kalman Filter (EKF) to provide stable GNSS estimation in real time. We further introduce a cross-drone tracking ID alignment algorithm that enforces global consistency across views, enabling robust multi-robot tracking with cooperative aerial coverage. We validate our system in diversified complex settings to show the accuracy and robustness of the proposed algorithm.
>
---
#### [replaced 159] Improving Visual Object Tracking through Visual Prompting
- **分类: cs.CV; cs.AI; cs.MM; eess.IV**

- **链接: [https://arxiv.org/pdf/2409.18901](https://arxiv.org/pdf/2409.18901)**

> **作者:** Shih-Fang Chen; Jun-Cheng Chen; I-Hong Jhuo; Yen-Yu Lin
>
> **备注:** This article was accepted by IEEE Transactions on Multimedia (TMM) in 2024 and published in 2025
>
> **摘要:** Learning a discriminative model that distinguishes the specified target from surrounding distractors across frames is essential for generic object tracking (GOT). Dynamic adaptation of target representation against distractors remains challenging because prevailing trackers exhibit limited discriminative capability. To address this issue, we present a new visual prompting mechanism for generic object tracking, termed PiVOT. PiVOT introduces mechanisms that leverage the pretrained foundation model (CLIP) to automatically generate and refine visual prompts online, thereby enabling the tracker to suppress distractors through contrastive guidance. To transfer contrastive knowledge from the foundation model to the tracker, PiVOT automatically propagates this knowledge online and dynamically generates and updates visual prompts. Specifically, it proposes a prompt initialization mechanism that produces an initial visual prompt highlighting potential target locations. The foundation model is then used to refine the prompt based on appearance similarities between candidate objects and reference templates across potential targets. After refinement, the visual prompt better highlights potential target locations and reduces irrelevant prompt information. With the proposed prompting mechanism, the tracker can generate instance-aware feature maps guided by the visual prompts, which are incrementally and automatically updated during tracking, thereby effectively suppressing distractors. Extensive experiments across multiple benchmarks indicate that PiVOT, with the proposed prompting mechanism, can suppress distracting objects and improve tracking performance.
>
---
#### [replaced 160] LEL: Lipschitz Continuity Constrained Ensemble Learning for Efficient EEG-Based Intra-subject Emotion Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.09156](https://arxiv.org/pdf/2504.09156)**

> **作者:** Shengyu Gong; Yueyang Li; Zijian Kang; Bo Chai; Weiming Zeng; Hongjie Yan; Zhiguo Zhang; Wai Ting Siok; Nizhuan Wang
>
> **摘要:** Accurate and efficient recognition of emotional states is critical for human social functioning, and impairments in this ability are associated with significant psychosocial difficulties. While electroencephalography (EEG) offers a powerful tool for objective emotion detection, existing EEG-based Emotion Recognition (EER) methods suffer from three key limitations: (1) insufficient model stability, (2) limited accuracy in processing high-dimensional nonlinear EEG signals, and (3) poor robustness against intra-subject variability and signal noise. To address these challenges, we introduce Lipschitz continuity-constrained Ensemble Learning (LEL), a novel framework that enhances EEG-based emotion recognition by enforcing Lipschitz continuity constraints on Transformer-based attention mechanisms, spectral extraction, and normalization modules. This constraint ensures model stability, reduces sensitivity to signal variability and noise, and improves generalization capability. Additionally, LEL employs a learnable ensemble fusion strategy that optimally combines decisions from multiple heterogeneous classifiers to mitigate single-model bias and variance. Extensive experiments on three public benchmark datasets (EAV, FACED, and SEED) demonstrate superior performance, achieving average recognition accuracies of 74.25%, 81.19%, and 86.79%, respectively. The official implementation codes are available at this https URL.
>
---
#### [replaced 161] PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24850](https://arxiv.org/pdf/2509.24850)**

> **作者:** Bo Zhao; Dan Guo; Junzhe Cao; Yong Xu; Bochao Zou; Tao Tan; Yue Sun; Zitong Yu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Remote photoplethysmography (rPPG) measurement enables non-contact physiological monitoring but suffers from accuracy degradation under head motion and illumination changes. Existing deep learning methods are mostly heuristic and lack theoretical grounding, limiting robustness and interpretability. In this work, we propose a physics-informed rPPG paradigm derived from the Navier-Stokes equations of hemodynamics, showing that the pulse signal follows a second-order dynamical system whose discrete solution naturally leads to a causal convolution, justifying the use of a Temporal Convolutional Network (TCN). Based on this principle, we design the PHASE-Net, a lightweight model with three key components: 1) Zero-FLOPs Axial Swapper module to swap or transpose a few spatial channels to mix distant facial regions, boosting cross-region feature interaction without changing temporal order; 2) Adaptive Spatial Filter to learn a soft spatial mask per frame to highlight signal-rich areas and suppress noise for cleaner feature maps; and 3) Gated TCN, a causal dilated TCN with gating that models long-range temporal dynamics for accurate pulse recovery. Extensive experiments demonstrate that PHASE-Net achieves state-of-the-art performance and strong efficiency, offering a theoretically grounded and deployment-ready rPPG solution. The source code is available at this https URL.
>
---
#### [replaced 162] CrystaL: Spontaneous Emergence of Visual Latents in MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20980](https://arxiv.org/pdf/2602.20980)**

> **作者:** Yang Zhang; Danyang Li; Yuxuan Li; Xin Zhang; Tianyu Xie; Mingming Cheng; Xiang Li
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable performance by integrating powerful language backbones with large-scale visual encoders. Among these, latent Chain-of-Thought (CoT) methods enable implicit reasoning in continuous hidden states, facilitating seamless vision-language integration and faster inference. However, existing heuristically predefined supervision signals in latent CoT provide limited guidance for preserving critical visual information in intermediate latent states. To address this limitation, we propose CrystaL (Crystallized Latent Reasoning), a single-stage framework with two paths to process intact and corrupted images, respectively. By explicitly aligning the attention patterns and prediction distributions across the two paths, CrystaL crystallizes latent representations into task-relevant visual semantics, without relying on auxiliary annotations or external modules. Extensive experiments on perception-intensive benchmarks demonstrate that CrystaL consistently outperforms state-of-the-art baselines, achieving substantial gains in fine-grained visual understanding while maintaining robust reasoning capabilities.
>
---
#### [replaced 163] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于视觉定位任务，针对VLM系统的安全性问题，提出IAG方法实现多目标后门攻击，通过动态生成文本引导的触发器，在不影响正常性能的情况下完成攻击。**

- **链接: [https://arxiv.org/pdf/2508.09456](https://arxiv.org/pdf/2508.09456)**

> **作者:** Junxian Li; Beining Xu; Simin Chen; Jiatong Li; Jingdi Lei; Haodong Zhao; Di Zhang
>
> **备注:** 20 pages, 13 Figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack on VLM-based visual grounding. Unlike prior attacks that rely on static triggers or fixed targets, we propose IAG, a method that dynamically generates input-aware, text-guided triggers conditioned on any specified target object description to execute the attack. This is achieved through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth. Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+, RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate that IAG achieves the best ASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses, and exhibiting transferability across datasets and models. These findings underscore critical security risks in grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
>
---
#### [replaced 164] 3D Gaussian Splatting with Fisheye Images: Field of View Analysis and Depth-Based Initialization
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2508.06968](https://arxiv.org/pdf/2508.06968)**

> **作者:** Ulas Gunes; Matias Turkulainen; Mikhail Silaev; Juho Kannala; Esa Rahtu
>
> **备注:** VISAPP 2026 Accepted Camera Ready Version
>
> **摘要:** We present the first evaluation of 3D Gaussian Splatting methods on real fisheye imagery with fields of view above 180\textdegree{}. Our study evaluates Fisheye-GS \cite{liao2024fisheyegslightweightextensiblegaussian} and 3DGUT \cite{wu20253dgut} on indoor and outdoor scenes captured with 200\textdegree{} fisheye cameras, with the aim of assessing the practicality of wide-angle reconstruction under severe distortion. By comparing reconstructions at 200\textdegree{}, 160\textdegree{}, and 120\textdegree{} field-of-view, we show that both methods achieve their best results at 160\textdegree{}, which balances scene coverage with image quality, while distortion at 200\textdegree{} degrades performance. To address the common failure of Structure-from-Motion (SfM) initialization at such wide angles, we introduce a depth-based alternative using UniK3D (Universal Camera Monocular 3D Estimation) \cite{piccinelli2025unik3d}. This represents the first application of UniK3D to fisheye imagery beyond 200\textdegree{}, despite the model not being trained on such data. With the number of predicted points controlled to match SfM for fairness, UniK3D produces geometrically accurate reconstructions that rival or surpass SfM, even in challenging scenes with fog, glare, or open sky. These results demonstrate the feasibility of fisheye-based 3D Gaussian Splatting and provides a benchmark for future research on wide-angle reconstruction from sparse and distorted inputs.
>
---
#### [replaced 165] ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.15695](https://arxiv.org/pdf/2509.15695)**

> **作者:** Zhaoyang Li; Zhan Ling; Yuchen Zhou; Litian Gong; Erdem Bıyık; Hao Su
>
> **摘要:** Large Vision-Language Models (LVLMs) excel at captioning, visual question answering, and robotics by combining vision and language, yet they often miss obvious objects or hallucinate nonexistent ones in atypical scenes. We examine these failures through the lens of uncertainty, focusing on contextual incongruity, where objects appear unexpectedly or fail to appear in expected contexts, and show that such cases increase recognition difficulty for state-of-the- art LVLMs. To study this regime, we introduce the Object Recognition in Incongruous Context (ORIC) framework, which constructs incongruous object-context pairs through two complementary strategies: (1) LLM-guided sampling to identify hard-to-recognize objects present in the image and (2) CLIP-guided sampling to mine plausible but absent ones. Applied to MSCOCO, ORIC creates ORIC-Bench and ORIC-style training data. Evaluating 18 LVLMs and 2 open-vocabulary detectors reveals significant degradation and bias under incongruous contexts. Visual Reinforcement Fine-Tuning of Qwen3-VL-8B-Instruct on 600 ORIC samples improves performance on ORIC-Bench, AMBER, and HallusionBench. Overall, we show that contextual incongruity is a key source of uncertainty and provide tools for more reliable LVLMs. The dataset and code are publicly available at this https URL.
>
---
#### [replaced 166] Counting Through Occlusion: Framework for Open World Amodal Counting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12702](https://arxiv.org/pdf/2511.12702)**

> **作者:** Safaeid Hossain Arib; Rabeya Akter; Abdul Monaf Chowdhury; Md Jubair Ahmed Sourov; Md Mehedi Hasan
>
> **摘要:** Object counting has achieved remarkable success on visible instances, yet state-of-the-art (SOTA) methods fail under occlusion. This failure stems from a fundamental architectural limitation where backbone networks encode occluding surfaces rather than target objects, thereby corrupting the feature representations required for accurate enumeration. To address this, we present CountOCC, an amodal counting framework that explicitly reconstructs occluded object features through hierarchical multimodal guidance. Rather than accepting degraded encodings, we synthesize complete representations by integrating spatial context from visible fragments with semantic priors from text and visual embeddings, generating features at occluded locations across multiple pyramid levels. We further introduce a visual equivalence objective that enforces consistency in attention space, ensuring that both occluded and unoccluded views of the same scene produce spatially aligned gradient-based attention maps. Together, these complementary mechanisms preserve discriminative properties essential for accurate counting under occlusion. For rigorous evaluation, we establish occlusion-augmented versions of FSC-147 and CARPK (FSC-147-OCC and CARPK-OCC). CountOCC achieves SOTA performance on FSC-147-OCC with 26.72% and 20.80% MAE reduction over prior baselines under occlusion in validation and test, respectively. CountOCC also demonstrates exceptional generalization by setting new SOTA results on CARPK-OCC with 49.89% MAE reduction and on CAPTURe-Real with 28.79% MAE reduction, validating robust amodal counting.
>
---
#### [replaced 167] MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19961](https://arxiv.org/pdf/2601.19961)**

> **作者:** Huanlin Gao; Ping Chen; Fuyuan Shi; Ruijia Wu; Li YanTao; Qiang Hui; Yuren You; Ting Lu; Chao Tan; Shaoan Zhao; Zhaoxiang Liu; Fang Zhao; Kai Wang; Shiguo Lian
>
> **摘要:** We present MeanCache, a training-free caching framework for efficient Flow Matching inference. Existing caching methods reduce redundant computation but typically rely on instantaneous velocity information (e.g., feature caching), which often leads to severe trajectory deviations and error accumulation under high acceleration ratios. MeanCache introduces an average-velocity perspective: by leveraging cached Jacobian--vector products (JVP) to construct interval average velocities from instantaneous velocities, it effectively mitigates local error accumulation. To further improve cache timing and JVP reuse stability, we develop a trajectory-stability scheduling strategy as a practical tool, employing a Peak-Suppressed Shortest Path under budget constraints to determine the schedule. Experiments on FLUX.1, Qwen-Image, and HunyuanVideo demonstrate that MeanCache achieves 4.12X and 4.56X and 3.59X acceleration, respectively, while consistently outperforming state-of-the-art caching baselines in generation quality. We believe this simple yet effective approach provides a new perspective for Flow Matching inference and will inspire further exploration of stability-driven acceleration in commercial-scale generative models.
>
---
#### [replaced 168] Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12985](https://arxiv.org/pdf/2511.12985)**

> **作者:** Minsoo Jo; Dongyoon Yang; Taesup Kim
>
> **备注:** Accepted by AAAI 2026. Code available at: this https URL
>
> **摘要:** Adversarial examples in neural networks have been extensively studied in Euclidean geometry, but recent advances in \textit{hyperbolic networks} call for a reevaluation of attack strategies in non-Euclidean geometries. Existing methods such as FGSM and PGD apply perturbations without regard to the underlying hyperbolic structure, potentially leading to inefficient or geometrically inconsistent attacks. In this work, we propose a novel adversarial attack that explicitly leverages the geometric properties of hyperbolic space. Specifically, we compute the gradient of the loss function in the tangent space of hyperbolic space, decompose it into a radial (depth) component and an angular (semantic) component, and apply perturbation derived solely from the angular direction. Our method generates adversarial examples by focusing perturbations in semantically sensitive directions encoded in angular movement within the hyperbolic geometry. Empirical results on image classification, cross-modal retrieval tasks and network architectures demonstrate that our attack achieves higher fooling rates than conventional adversarial attacks, while producing high-impact perturbations with deeper insights into vulnerabilities of hyperbolic embeddings. This work highlights the importance of geometry-aware adversarial strategies in curved representation spaces and provides a principled framework for attacking hierarchical embeddings.
>
---
#### [replaced 169] Two-Step Data Augmentation for Masked Face Detection and Recognition: Turning Fake Masks to Real
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.15774](https://arxiv.org/pdf/2512.15774)**

> **作者:** Yan Yang; George Bebis; Mircea Nicolescu
>
> **备注:** 9 pages, 9 figures. Conference version
>
> **摘要:** The absence of large-scale masked face datasets poses challenges for masked face detection and recognition. We propose a two-step generative data augmentation framework combining rule-based mask warping with unpaired image-to-image translation using GANs, producing masked face samples that go beyond rule-based geometric overlays. Trained on 3390 images, about 0.7% of the training data used by IAMGAN, the proposed approach yields consistent improvements over rule-based warping alone and achieves complementary results at a small fraction of IAMGAN data scale, showing that both steps contribute. Evaluation is conducted directly on the generated samples and is qualitative; quantitative metrics like FID and KID were not applied as any real reference distribution would unfairly favor the model with closer training data. We introduce a non-mask preservation loss to reduce non-mask distortions and stabilize training, and stochastic noise injection to enhance sample diversity. Note: This paper originated as a coursework submission completed under resource constraints. Following an inexplicable termination of scholarship, the author took on part-time employment to maintain research continuity, which introduced a mid-semester domain pivot from medical imaging to masked face tasks due to restrictions on company data. The work was completed alongside concurrent coursework with delayed compute access and without AI assistance of any kind. It was submitted to a small venue at the semester end under an obligatory publication requirement and accepted without revision requests. Subsequent invitations to submit to first-tier venues were not pursued due to continued funding absence. Downstream evaluation on recognition or detection performance was not completed by the submission deadline. These notes are added in response to subsequent comparisons and criticisms that did not account for these conditions.
>
---
#### [replaced 170] Pose Prior Learner: Unsupervised Categorical Prior Learning for Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.03858](https://arxiv.org/pdf/2410.03858)**

> **作者:** Ziyu Wang; Shuangpeng Han; Mengmi Zhang
>
> **摘要:** A prior represents a set of beliefs or assumptions about a system, aiding inference and decision-making. In this paper, we introduce the challenge of unsupervised categorical prior learning in pose estimation, where AI models learn a general pose prior for an object category from images in a self-supervised manner. Although priors are effective in estimating pose, acquiring them can be difficult. We propose a novel method, named Pose Prior Learner (PPL), to learn a general pose prior for any object category. PPL uses a hierarchical memory to store compositional parts of prototypical poses, from which we distill a general pose prior. This prior improves pose estimation accuracy through template transformation and image reconstruction. PPL learns meaningful pose priors without any additional human annotations or interventions, outperforming competitive baselines on both human and animal pose estimation datasets. Notably, our experimental results reveal the effectiveness of PPL using learned prototypical poses for pose estimation on occluded images. Through iterative inference, PPL leverages the pose prior to refine estimated poses, regressing them to any prototypical poses stored in memory. Our code, model, and data are publicly available at: this https URL.
>
---
#### [replaced 171] ReDepth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.17908](https://arxiv.org/pdf/2512.17908)**

> **作者:** Ananta R. Bhattarai; Helge Rhodin
>
> **备注:** Accepted at CVPR 2026 (Findings). Project Page: this https URL Code: this https URL
>
> **摘要:** Monocular depth estimation remains challenging, as foundation models such as Depth Anything V2 (DA-V2) struggle with real-world images that are far from the training distribution. We introduce Re-Depth Anything, a test-time self-supervision framework that bridges this domain gap by fusing foundation models with the powerful priors of large-scale 2D diffusion models. Our method performs label-free refinement directly on the input image by re-lighting the predicted depth map and augmenting the input. This re-synthesis method replaces classical photometric reconstruction by leveraging shape from shading (SfS) cues in a new, generative context with Score Distillation Sampling (SDS). To prevent optimization collapse, our framework updates only intermediate embeddings and the decoder's weights, rather than optimizing the depth tensor directly or fine-tuning the full model. Across diverse benchmarks, Re-Depth Anything yields substantial gains in depth accuracy and realism over DA-V2, and applied on top of Depth Anything 3 (DA3) achieves state-of-the-art results, showcasing new avenues for self-supervision by geometric reasoning.
>
---
#### [replaced 172] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23029](https://arxiv.org/pdf/2602.23029)**

> **作者:** Tianyue Wang; Leigang Qu; Tianyu Yang; Xiangzhao Hao; Yifan Xu; Haiyun Guo; Jinqiao Wang
>
> **备注:** Accept to CVPR 2026
>
> **摘要:** Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images given a multimodal query (comprising a reference image and a modification text), without training on annotated triplets. Existing methods typically convert the multimodal query into a single modality-either as an edited caption for Text-to-Image retrieval (T2I) or as an edited image for Image-to-Image retrieval (I2I). However, each paradigm has inherent limitations: T2I often loses fine-grained visual details, while I2I struggles with complex semantic modifications. To effectively leverage their complementary strengths under diverse query intents, we propose WISER, a training-free framework that unifies T2I and I2I via a "retrieve-verify-refine" pipeline, explicitly modeling intent awareness and uncertainty awareness. Specifically, WISER first performs Wider Search by generating both edited captions and images for parallel retrieval to broaden the candidate pool. Then, it conducts Adaptive Fusion with a verifier to assess retrieval confidence, triggering refinement for uncertain retrievals, and dynamically fusing the dual-path for reliable ones. For uncertain retrievals, WISER generates refinement suggestions through structured self-reflection to guide the next retrieval round toward Deeper Thinking. Extensive experiments demonstrate that WISER significantly outperforms previous methods across multiple benchmarks, achieving relative improvements of 45% on CIRCO (mAP@5) and 57% on CIRR (Recall@1) over existing training-free methods. Notably, it even surpasses many training-dependent methods, highlighting its superiority and generalization under diverse scenarios. Code will be released at this https URL.
>
---
#### [replaced 173] Position: Evaluation of Visual Processing Should Be Human-Centered, Not Metric-Centered
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00643](https://arxiv.org/pdf/2603.00643)**

> **作者:** Jinfan Hu; Fanghua Yu; Zhiyuan You; Xiang Yin; Hongyu An; Xinqi Lin; Chao Dong; Jinjin Gu
>
> **摘要:** This position paper argues that the evaluation of modern visual processing systems should no longer be driven primarily by single-metric image quality assessment benchmarks, particularly in the era of generative and perception-oriented methods. Image restoration exemplifies this divergence: while objective IQA metrics enable reproducible, scalable evaluation, they have increasingly drifted apart from human perception and user preferences. We contend that this mismatch risks constraining innovation and misguiding research progress across visual processing tasks. Rather than rejecting metrics altogether, this paper calls for a rebalancing of evaluation paradigms, advocating a more human-centered, context-aware, and fine-grained approach to assessing the visual models' outcomes.
>
---
#### [replaced 174] 3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18064](https://arxiv.org/pdf/2602.18064)**

> **作者:** Ziyue Wang; Linghan Cai; Chang Han Low; Haofeng Liu; Junde Wu; Jingyu Wang; Rui Wang; Lei Song; Jiang Bian; Jingjing Fu; Yueming Jin
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** 3D CT analysis spans a continuum from low-level perception to high-level clinical understanding. Existing 3D-oriented analysis methods adopt either isolated task-specific modeling or task-agnostic end-to-end paradigms to produce one-hop outputs, impeding the systematic accumulation of perceptual evidence for downstream reasoning. In parallel, recent multimodal large language models (MLLMs) exhibit improved visual perception and can integrate visual and textual information effectively, yet their predominantly 2D-oriented designs fundamentally limit their ability to perceive and analyze volumetric medical data. To bridge this gap, we propose 3DMedAgent, a unified agent that enables 2D MLLMs to perform general 3D CT analysis without 3D-specific fine-tuning. 3DMedAgent coordinates heterogeneous visual and textual tools through a flexible MLLM agent, progressively decomposing complex 3D analysis into tractable subtasks that transition from global to regional views, from 3D volumes to informative 2D slices, and from visual evidence to structured textual representations. Central to this design, 3DMedAgent maintains a long-term structured memory that aggregates intermediate tool outputs and supports query-adaptive, evidence-driven multi-step reasoning. We further introduce the DeepChestVQA benchmark for evaluating unified perception-to-understanding capabilities in 3D thoracic imaging. Experiments across over 40 tasks demonstrate that 3DMedAgent consistently outperforms general, medical, and 3D-specific MLLMs, highlighting a scalable path toward general-purpose 3D clinical this http URL and data are available at \href{this https URL}{this https URL}.
>
---
#### [replaced 175] S2AM3D: Scale-controllable Part Segmentation of 3D Point Cloud
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00995](https://arxiv.org/pdf/2512.00995)**

> **作者:** Han Su; Tianyu Huang; Zichen Wan; Xiaohe Wu; Wangmeng Zuo
>
> **备注:** Accepted by CVPR 2026. Project page:this https URL
>
> **摘要:** Part-level point cloud segmentation has recently attracted significant attention in 3D computer vision. Nevertheless, existing research is constrained by two major challenges: native 3D models lack generalization due to data scarcity, while introducing 2D pre-trained knowledge often leads to inconsistent segmentation results across different views. To address these challenges, we propose S2AM3D, which incorporates 2D segmentation priors with 3D consistent supervision. We design a point-consistent part encoder that aggregates multi-view 2D features through native 3D contrastive learning, producing globally consistent point features. A scale-aware prompt decoder is then proposed to enable real-time adjustment of segmentation granularity via continuous scale signals. Simultaneously, we introduce a large-scale, high-quality part-level point cloud dataset with more than 100k samples, providing ample supervision signals for model training. Extensive experiments demonstrate that S2AM3D achieves leading performance across multiple evaluation settings, exhibiting exceptional robustness and controllability when handling complex structures and parts with significant size variations.
>
---
#### [replaced 176] Class Overwhelms: Mutual Conditional Blended-Target Domain Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2302.01516](https://arxiv.org/pdf/2302.01516)**

> **作者:** Pengcheng Xu; Boyu Wang; Charles Ling
>
> **摘要:** Current methods of blended targets domain adaptation (BTDA) usually infer or consider domain label information but underemphasize hybrid categorical feature structures of targets, which yields limited performance, especially under the label distribution shift. We demonstrate that domain labels are not directly necessary for BTDA if categorical distributions of various domains are sufficiently aligned even facing the imbalance of domains and the label distribution shift of classes. However, we observe that the cluster assumption in BTDA does not comprehensively hold. The hybrid categorical feature space hinders the modeling of categorical distributions and the generation of reliable pseudo labels for categorical alignment. To address these, we propose a categorical domain discriminator guided by uncertainty to explicitly model and directly align categorical distributions $P(Z|Y)$. Simultaneously, we utilize the low-level features to augment the single source features with diverse target styles to rectify the biased classifier $P(Y|Z)$ among diverse targets. Such a mutual conditional alignment of $P(Z|Y)$ and $P(Y|Z)$ forms a mutual reinforced mechanism. Our approach outperforms the state-of-the-art in BTDA even compared with methods utilizing domain labels, especially under the label distribution shift, and in single target DA on DomainNet. Source codes are available at this https URL.
>
---
#### [replaced 177] CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02951](https://arxiv.org/pdf/2603.02951)**

> **作者:** Zhenquan Yao; Zitong Huang; Yihan Zeng; Jianhua Han; Hang Xu; Chun-Mei Feng; Jianwei Ma; Wangmeng Zuo
>
> **摘要:** Graphical User Interface (GUI) Agents, benefiting from recent advances in multimodal large language models (MLLM), have achieved significant development. However, due to the frequent updates of GUI applications, adapting to new tasks without forgetting old tasks in GUI continual learning remains an open problem. In this work, we reveal that while Supervised Fine-Tuning (SFT) facilitates fast adaptation, it often triggers knowledge overwriting, whereas Reinforcement Learning (RL) demonstrates an inherent resilience that shields prior interaction logic from erasure. Based on this insight, we propose a \textbf{C}ontinual \textbf{G}UI \textbf{L}earning (CGL) framework that dynamically balances adaptation efficiency and skill retention by enhancing the synergy between SFT and RL. Specifically, we introduce an SFT proportion adjustment mechanism guided by policy entropy to dynamically control the weight allocation between the SFT and RL training phases. To resolve explicit gradient interference, we further develop a specialized gradient surgery strategy. By projecting exploratory SFT gradients onto GRPO-based anchor gradients, our method explicitly clips the components of SFT gradients that conflict with GRPO. On top of that, we establish an AndroidControl-CL benchmark, which divides GUI applications into distinct task groups to effectively simulate and evaluate the performance of continual GUI learning. Experimental results demonstrate the effectiveness of our proposed CGL framework across continual learning scenarios. The benchmark, code, and model will be made publicly available.
>
---
#### [replaced 178] Detecting AI-Generated Images via Contextual Anomaly Estimation in Masked AutoEncoders
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [https://arxiv.org/pdf/2511.06325](https://arxiv.org/pdf/2511.06325)**

> **作者:** Minsuk Jang; Hyunseo Jeong; Minseok Son; Changick Kim
>
> **摘要:** Context-based detection methods such as DetectGPT achieve strong generalization in identifying AI-generated text by evaluating content compatibility with a model's learned distribution. In contrast, existing image detectors rely on discriminative features from pretrained backbones such as CLIP, which implicitly capture generator-specific artifacts. However, as modern generative models rapidly advance in visual fidelity, the artifacts these detectors depend on are becoming increasingly subtle or absent, undermining their reliability. Masked AutoEncoders (MAE) are inherently trained to reconstruct masked patches from visible context, naturally modeling patch-level contextual plausibility akin to conditional probability estimation, while also serving as a powerful semantic feature extractor through its encoder. We propose CINEMAE, a novel architecture that exploits both capabilities of MAE for AI-generated image detection: we derive per-patch anomaly signals from the reconstruction mechanism and extract global semantic features from the encoder, fusing both context-based and feature-based cues for robust detection. CINEMAE achieves highly competitive mean accuracies of 96.63\% on GenImage and 93.96\% on AIGCDetectBenchmark, maintaining over 93\% accuracy even under JPEG compression at QF=50.
>
---
#### [replaced 179] Leveraging Hallucinations to Reduce Manual Prompt Dependency in Promptable Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.15205](https://arxiv.org/pdf/2408.15205)**

> **作者:** Jian Hu; Jiayi Lin; Junchi Yan; Shaogang Gong
>
> **备注:** NeurIPS 2024
>
> **摘要:** Promptable segmentation typically requires instance-specific manual prompts to guide the segmentation of each desired object. To minimize such a need, task-generic promptable segmentation has been introduced, which employs a single task-generic prompt to segment various images of different objects in the same task. Current methods use Multimodal Large Language Models (MLLMs) to reason detailed instance-specific prompts from a task-generic prompt for improving segmentation accuracy. The effectiveness of this segmentation heavily depends on the precision of these derived prompts. However, MLLMs often suffer hallucinations during reasoning, resulting in inaccurate prompting. While existing methods focus on eliminating hallucinations to improve a model, we argue that MLLM hallucinations can reveal valuable contextual insights when leveraged correctly, as they represent pre-trained large-scale knowledge beyond individual images. In this paper, we utilize hallucinations to mine task-related information from images and verify its accuracy for enhancing precision of the generated prompts. Specifically, we introduce an iterative Prompt-Mask Cycle generation framework (ProMaC) with a prompt generator and a mask this http URL prompt generator uses a multi-scale chain of thought prompting, initially exploring hallucinations for extracting extended contextual knowledge on a test this http URL hallucinations are then reduced to formulate precise instance-specific prompts, directing the mask generator to produce masks that are consistent with task semantics by mask semantic alignment. The generated masks iteratively induce the prompt generator to focus more on task-relevant image areas and reduce irrelevant hallucinations, resulting jointly in better prompts and masks. Experiments on 5 benchmarks demonstrate the effectiveness of ProMaC. Code given in this https URL.
>
---
#### [replaced 180] Attribute Distribution Modeling and Semantic-Visual Alignment for Generative Zero-shot Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06281](https://arxiv.org/pdf/2603.06281)**

> **作者:** Haojie Pu; Zhuoming Li; Yongbiao Gao; Yuheng Jia
>
> **备注:** 17 pages, 13 figures(Under review)
>
> **摘要:** Generative zero-shot learning (ZSL) synthesizes features for unseen classes, leveraging semantic conditions to transfer knowledge from seen classes. However, it also introduces two intrinsic challenges: (1) class-level attributes fails to capture instance-specific visual appearances due to substantial intra-class variability, thus causing the class-instance gap; (2) the substantial mismatch between semantic and visual feature distributions, manifested in inter-class correlations, gives rise to the semantic-visual domain gap. To address these challenges, we propose an Attribute Distribution Modeling and Semantic-Visual Alignment (ADiVA) approach, jointly modeling attribute distributions and performing explicit semantic-visual alignment. Specifically, our ADiVA consists of two modules: an Attribute Distribution Modeling (ADM) module that learns a transferable attribute distribution for each class and samples instance-level attributes for unseen classes, and a Visual-Guided Alignment (VGA) module that refines semantic representations to better reflect visual structures. Experiments on three widely used benchmark datasets demonstrate that ADiVA significantly outperforms state-of-the-art methods (e.g., achieving gains of 4.7% and 6.1% on AWA2 and SUN, respectively). Moreover, our approach can serve as a plugin to enhance existing generative ZSL methods.
>
---
#### [replaced 181] FLARE: Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.05611](https://arxiv.org/pdf/2601.05611)**

> **作者:** Chengen Xie; Chonghao Sima; Tianyu Li; Bin Sun; Junjie Wu; Zhihui Hao; Hongyang Li
>
> **摘要:** While Vision-Language Models (VLMs) offer rich world knowledge for end-to-end autonomous driving, current approaches heavily rely on labor-intensive language annotations (e.g., VQA) to bridge perception and control. This paradigm suffers from a fundamental mismatch between discrete linguistic tokens and continuous driving trajectories, often leading to suboptimal control policies and inefficient utilization of pre-trained knowledge. To address these challenges, we propose FLARE (Future-aware LAtent REpresentation), a novel framework that activates the visual-semantic capabilities of pre-trained VLMs without requiring language supervision. Instead of aligning with text, we introduce a self-supervised future feature prediction objective. This mechanism compels the model to anticipate scene dynamics and ego-motion directly in the latent space, enabling the learning of robust driving representations from large-scale unlabeled trajectory data. Furthermore, we integrate Group Relative Policy Optimization (GRPO) into the planning process to refine decision-making quality. Extensive experiments on the NAVSIM benchmark demonstrate that FLARE achieves state-of-the-art performance, validating the effectiveness of leveraging VLM knowledge via predictive self-supervision rather than explicit language generation.
>
---
#### [replaced 182] Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.05089](https://arxiv.org/pdf/2504.05089)**

> **作者:** Johannes Dollinger; Damien Robert; Elena Plekhanova; Lukas Drees; Jan Dirk Wegner
>
> **备注:** Published as a workshop paper at "Tackling Climate Change with Machine Learning", ICLR 2025
>
> **摘要:** Deep learning on climatic data holds potential for macroecological applications. However, its adoption remains limited among scientists outside the deep learning community due to storage, compute, and technical expertise barriers. To address this, we introduce Climplicit, a spatio-temporal geolocation encoder pretrained to generate implicit climatic representations anywhere on Earth. By bypassing the need to download raw climatic rasters and train feature extractors, our model uses x3500 less disk space and significantly reduces computational needs for downstream tasks. We evaluate our Climplicit embeddings on biomes classification, species distribution modeling, and plant trait regression. We find that single-layer probing our Climplicit embeddings consistently performs better or on par with training a model from scratch on downstream tasks and overall better than alternative geolocation encoding models.
>
---
#### [replaced 183] Adopting a human developmental visual diet yields robust, shape-based AI vision
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.03168](https://arxiv.org/pdf/2507.03168)**

> **作者:** Zejin Lu; Sushrut Thorat; Radoslaw M Cichy; Tim C Kietzmann
>
> **摘要:** Despite years of research and the dramatic scaling of artificial intelligence (AI) systems, a striking misalignment between artificial and human vision persists. Contrary to humans, AI relies heavily on texture-features rather than shape information, lacks robustness to image distortions, remains highly vulnerable to adversarial attacks, and struggles to recognise simple abstract shapes within complex backgrounds. To close this gap, here we take inspiration from how human vision develops from early infancy into adulthood. We quantified visual maturation by synthesising decades of research into a novel developmental visual diet (DVD) for AI vision. Guiding AI systems through this human-inspired curriculum, which considers the development of visual acuity, contrast sensitivity, and colour, produces models that better align with human behaviour on every hallmark of robust vision tested, yielding the strongest reported reliance on shape information to date, abstract shape recognition beyond the state of the art, and higher resilience to image corruptions and adversarial attacks. Our results thus demonstrate that robust AI vision can be achieved by guiding how a model learns, not merely how much it learns, offering a resource-efficient route toward safer and more human-like artificial visual systems.
>
---
#### [replaced 184] Prompt-SID: Learning Structural Representation Prompt via Latent Diffusion for Single-Image Denoising
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.06432](https://arxiv.org/pdf/2502.06432)**

> **作者:** Huaqiu Li; Wang Zhang; Xiaowan Hu; Tao Jiang; Zikang Chen; Haoqian Wang
>
> **摘要:** Many studies have concentrated on constructing supervised models utilizing paired datasets for image denoising, which proves to be expensive and time-consuming. Current self-supervised and unsupervised approaches typically rely on blind-spot networks or sub-image pairs sampling, resulting in pixel information loss and destruction of detailed structural information, thereby significantly constraining the efficacy of such methods. In this paper, we introduce Prompt-SID, a prompt-learning-based single image denoising framework that emphasizes preserving of structural details. This approach is trained in a self-supervised manner using downsampled image pairs. It captures original-scale image information through structural encoding and integrates this prompt into the denoiser. To achieve this, we propose a structural representation generation model based on the latent diffusion process and design a structural attention module within the transformer-based denoiser architecture to decode the prompt. Additionally, we introduce a scale replay training mechanism, which effectively mitigates the scale gap from images of different resolutions. We conduct comprehensive experiments on synthetic, real-world, and fluorescence imaging datasets, showcasing the remarkable effectiveness of Prompt-SID. Our code will be released at this https URL.
>
---
#### [replaced 185] Efficient Test-Time Optimization for Depth Completion via Low-Rank Decoder Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01765](https://arxiv.org/pdf/2603.01765)**

> **作者:** Minseok Seo; Wonjun Lee; Jaehyuk Jang; Changick Kim
>
> **备注:** 17 pages, 7 figures [We achieved a new Pareto frontier in test-time depth completion.]
>
> **摘要:** Zero-shot depth completion has gained attention for its ability to generalize across environments without sensor-specific datasets or retraining. However, most existing approaches rely on diffusion-based test-time optimization, which is computationally expensive due to iterative denoising. Recent visual-prompt-based methods reduce training cost but still require repeated forward--backward passes through the full frozen network to optimize input-level prompts, resulting in slow inference. In this work, we show that adapting only the decoder is sufficient for effective test-time optimization, as depth foundation models concentrate depth-relevant information within a low-dimensional decoder subspace. Based on this insight, we propose a lightweight test-time adaptation method that updates only this low-dimensional subspace using sparse depth supervision. Our approach achieves state-of-the-art performance, establishing a new Pareto frontier between accuracy and efficiency for test-time adaptation. Extensive experiments on five indoor and outdoor datasets demonstrate consistent improvements over prior methods, highlighting the practicality of fast zero-shot depth completion.
>
---
#### [replaced 186] Bridging Domains through Subspace-Aware Model Merging
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05768](https://arxiv.org/pdf/2603.05768)**

> **作者:** Levy Chaves; Chao Zhou; Rebekka Burkholz; Eduardo Valle; Sandra Avila
>
> **备注:** Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 (CVPR)
>
> **摘要:** Model merging integrates multiple task-specific models into a single consolidated one. Recent research has made progress in improving merging performance for in-distribution or multi-task scenarios, but domain generalization in model merging remains underexplored. We investigate how merging models fine-tuned on distinct domains affects generalization to unseen domains. Through an analysis of parameter competition in the task matrix using singular value decomposition, we show that merging models trained under different distribution shifts induces stronger conflicts between their subspaces compared to traditional multi-task settings. To mitigate this issue, we propose SCORE (Subspace COnflict-Resolving mErging), a method designed to alleviate such singular subspace conflicts. SCORE finds a shared orthogonal basis by computing the principal components of the concatenated leading singular vectors of all models. It then projects each task matrix into the shared basis, pruning off-diagonal components to remove conflicting singular directions. SCORE consistently outperforms, on average, existing model merging approaches in domain generalization settings across a variety of architectures and model scales, demonstrating its effectiveness and scalability.
>
---
#### [replaced 187] TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05867](https://arxiv.org/pdf/2603.05867)**

> **作者:** Sijing Li; Zhongwei Qiu; Jiang Liu; Wenqiao Zhang; Tianwei Lin; Yihan Xie; Jianxiang An; Boxiang Yun; Chenglin Yang; Jun Xiao; Guangyu Guo; Jiawen Yao; Wei Liu; Yuan Gao; Ke Yan; Weiwei Cao; Zhilin Zheng; Tony C. W. Mok; Kai Cao; Yu Shi; Jiuyu Zhang; Jian Zhou; Beng Chin Ooi; Yingda Xia; Ling Zhang
>
> **备注:** Accepted at ICLR 2026. 10 pages + appendix
>
> **摘要:** Accurate tumor analysis is central to clinical radiology and precision oncology, where early detection, reliable lesion characterization, and pathology-level risk assessment guide diagnosis and treatment planning. Chain-of-Thought (CoT) reasoning is particularly important in this setting because it enables step-by-step interpretation from imaging findings to clinical impressions and pathology conclusions, improving traceability and reducing diagnostic errors. Here, we target the clinical tumor analysis task and build a large-scale benchmark that operationalizes a multimodal reasoning pipeline, spanning findings, impressions, and pathology predictions. We curate TumorCoT, a large-scale dataset of 1.5M CoT-labeled VQA instructions paired with 3D CT scans, with step-aligned rationales and cross-modal alignments along the trajectory from findings to impression to pathology, enabling evaluation of both answer accuracy and reasoning consistency. We further propose TumorChain, a multimodal interleaved reasoning framework that tightly couples 3D imaging encoders, clinical text understanding, and organ-level vision-language alignment. Through cross-modal alignment and iterative interleaved causal reasoning, TumorChain grounds visual evidence, aggregates conclusions, and issues pathology predictions after multiple rounds of self-refinement, improving traceability and reducing hallucination risk. Experiments show consistent improvements over strong baselines in lesion detection, impression generation, and pathology classification, and demonstrate strong generalization on the DeepTumorVQA benchmark. These results highlight the potential of multimodal reasoning for reliable and interpretable tumor analysis in clinical practice. Detailed information about our project can be found on our project homepage at this https URL.
>
---
#### [replaced 188] SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.14756](https://arxiv.org/pdf/2503.14756)**

> **作者:** Hou In Ivan Tam; Hou In Derek Pun; Austin T. Wang; Angel X. Chang; Manolis Savva
>
> **备注:** Accepted at WACV 2026 (Oral). Project page: this https URL . Minor revisions for camera-ready version
>
> **摘要:** Despite recent advances in text-conditioned 3D indoor scene generation, there remain gaps in the evaluation of these methods. Existing metrics often measure realism by comparing generated scenes to a set of ground-truth scenes, but they overlook how well scenes follow the input text and capture implicit expectations of plausibility. We present SceneEval, an evaluation framework designed to address these limitations. SceneEval introduces fine-grained metrics for explicit user requirements-including object counts, attributes, and spatial relationships-and complementary metrics for implicit expectations such as support, collisions, and navigability. Together, these provide interpretable and comprehensive assessments of scene quality. To ground evaluation, we curate SceneEval-500, a benchmark of 500 text descriptions with detailed annotations of expected scene properties. This dataset establishes a common reference for reproducible and systematic comparison across scene generation methods. We evaluate six recent scene generation approaches using SceneEval and demonstrate its ability to provide detailed assessments of the generated scenes, highlighting strengths and areas for improvement across multiple dimensions. Our results identify significant gaps in current methods, underscoring the need for further research toward practical and controllable scene synthesis.
>
---
#### [replaced 189] MICA: Multi-Agent Industrial Coordination Assistant
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.15237](https://arxiv.org/pdf/2509.15237)**

> **作者:** Di Wen; Kunyu Peng; Junwei Zheng; Yufan Chen; Yitian Shi; Jiale Wei; Ruiping Liu; Kailun Yang; Rainer Stiefelhagen
>
> **备注:** Accepted to ICRA 2026. The source code will be made publicly available at this https URL
>
> **摘要:** Industrial workflows demand adaptive and trustworthy assistance that can operate under limited computing, connectivity, and strict privacy constraints. In this work, we present MICA (Multi-Agent Industrial Coordination Assistant), a perception-grounded and speech-interactive system that delivers real-time guidance for assembly, troubleshooting, part queries, and maintenance. MICA coordinates five role-specialized language agents, audited by a safety checker, to ensure accurate and compliant support. To achieve robust step understanding, we introduce Adaptive Step Fusion (ASF), which dynamically blends expert reasoning with online adaptation from natural speech feedback. Furthermore, we establish a new multi-agent coordination benchmark across representative task categories and propose evaluation metrics tailored to industrial assistance, enabling systematic comparison of different coordination topologies. Our experiments demonstrate that MICA consistently improves task success, reliability, and responsiveness over baseline structures, while remaining deployable on practical offline hardware. Together, these contributions highlight MICA as a step toward deployable, privacy-preserving multi-agent assistants for dynamic factory environments. The source code will be made publicly available at this https URL.
>
---
#### [replaced 190] Unsupervised Deep Generative Models for Anomaly Detection in Neuroimaging: A Systematic Scoping Review
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14462](https://arxiv.org/pdf/2510.14462)**

> **作者:** Youwan Mahé; Elise Bannier; Stéphanie Leplaideur; Elisa Fromont; Francesca Galassi
>
> **摘要:** Unsupervised anomaly detection (UAD) based on deep generative modelling has been increasingly explored for identifying pathological brain abnormalities without requiring voxel-level annotations. By learning the distribution of healthy anatomy and generating pseudo-healthy reconstructions, these methods aim to localise deviations in a pathology-agnostic manner. Despite rapid methodological development - from autoencoders and variational autoencoders to generative adversarial networks and diffusion-based models - a structured synthesis of their application in structural neuroimaging is lacking. We conducted a PRISMA-ScR-guided scoping review of studies published between January 2018-December 2025 that applied unsupervised deep generative models to anomaly detection in brain MRI (and, less frequently, CT). Thirty-three studies met inclusion criteria. Methods were categorised by architectural family, and reported performance was synthesised across major pathology groups, with segmentation (Dice) and detection metrics (AUROC, AUPRC) disaggregated by evaluation level (voxel, slice, subject). For transparency, we also summarised dataset characteristics, dimensionality (2D vs. 3D), and thresholding strategies. Overall, unsupervised generative approaches demonstrate potential for pathology-agnostic anomaly localisation, particularly in settings where annotated data are scarce. However, methodological heterogeneity, limited external validation, and sensitivity to dataset characteristics remain important challenges. Emerging paradigms - including anatomy-aware modelling, diffusion-based frameworks, and alternative normative evaluation metrics - seek to address these limitations and improve robustness and clinical relevance.
>
---
#### [replaced 191] ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19582](https://arxiv.org/pdf/2601.19582)**

> **作者:** Yujin Wang; Yutong Zheng; Wenxian Fan; Tianyi Wang; Hongqing Chu; Li Zhang; Bingzhao Gao; Daxin Tian; Jianqiang Wang; Hong Chen
>
> **摘要:** In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.
>
---
#### [replaced 192] ForamDeepSlice: A High-Accuracy Deep Learning Framework for Foraminifera Species Classification from 2D Micro-CT Slices
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.00912](https://arxiv.org/pdf/2512.00912)**

> **作者:** Abdelghafour Halimi; Ali Alibrahim; Didier Barradas-Bautista; Ronell Sicat; Abdulkader M. Afifi
>
> **摘要:** This study presents a comprehensive deep learning pipeline for the automated classification of foraminifera species using 2D micro-CT slices derived from 3D scans. We curated a scientifically rigorous dataset of 97 micro-CT scanned specimens spanning 27 species, from which we selected 12 representative species with sufficient specimen counts (at least four 3D models each) for robust classification. To ensure methodological integrity and prevent data leakage, we employed specimen-level data splitting, resulting in 109,617 high-quality 2D slices (44,103 for training, 14,046 for validation, and 51,468 for testing). We evaluated seven state-of-the-art 2D convolutional neural network (CNN) architectures using transfer learning. Our final ensemble model, ForamDeepSlice (FDS), combining ConvNeXt-Large and EfficientNetV2-Small, achieved a test accuracy of 95.64%, with a top-3 accuracy of 99.6% and an area under the ROC curve (AUC) of 0.998 across all species. To facilitate practical deployment, we developed an interactive advanced dashboard that supports real-time slice classification and 3D slice matching using advanced similarity metrics, including SSIM, NCC, and the Dice coefficient. This work establishes new benchmarks for AI-assisted micropaleontological identification and provides a fully reproducible framework for foraminifera classification research, bridging the gap between deep learning and applied geosciences.
>
---
#### [replaced 193] LIVE-GS: Online LiDAR-Inertial-Visual State Estimation and Globally Consistent Mapping with 3D Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，解决3DGS与LiDAR融合中的精度和效率问题，提出LIVE-GS框架实现高精度全局一致映射。**

- **链接: [https://arxiv.org/pdf/2507.23273](https://arxiv.org/pdf/2507.23273)**

> **作者:** Jaeseok Park; Chanoh Park; Minsu Kim; Minkyoung Kim; Soohwan Kim
>
> **摘要:** While 3D Gaussian Splatting (3DGS) enabled photorealistic mapping, its integration into SLAM has largely followed traditional camera-centric pipelines. As a result, they inherit well-known weaknesses such as high computational load, failure in texture-poor or illumination-varying environments, and limited operational range, particularly for RGB-D setups. On the other hand, LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for tighter global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose LIVE-GS, an online LiDAR-Inertial Visual SLAM framework that tightly couples 3D Gaussian Splatting with LiDAR-based surfels to ensure high-precision map consistency through global geometric optimization. Particularly, to handle sparse data, our system employs a depth-invariant Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate competitive performance in rendering quality and map-building efficiency compared with representative 3DGS SLAM baselines.
>
---
#### [replaced 194] Point-based Instance Completion with Scene Constraints
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.05698](https://arxiv.org/pdf/2504.05698)**

> **作者:** Wesley Khademi; Li Fuxin
>
> **备注:** Published in ICLR 2025. Project Page: this https URL
>
> **摘要:** Recent point-based object completion methods have demonstrated the ability to accurately recover the missing geometry of partially observed objects. However, these approaches are not well-suited for completing objects within a scene, as they do not consider known scene constraints (e.g., other observed surfaces) in their completions and further expect the partial input to be in a canonical coordinate system, which does not hold for objects within scenes. While instance scene completion methods have been proposed for completing objects within a scene, they lag behind point-based object completion methods in terms of object completion quality and still do not consider known scene constraints during completion. To overcome these limitations, we propose a point cloud-based instance completion model that can robustly complete objects at arbitrary scales and pose in the scene. To enable reasoning at the scene level, we introduce a sparse set of scene constraints represented as point clouds and integrate them into our completion model via a cross-attention mechanism. To evaluate the instance scene completion task on indoor scenes, we further build a new dataset called ScanWCF, which contains labeled partial scans as well as aligned ground truth scene completions that are watertight and collision-free. Through several experiments, we demonstrate that our method achieves improved fidelity to partial scans, higher completion quality, and greater plausibility over existing state-of-the-art methods.
>
---
#### [replaced 195] MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.19210](https://arxiv.org/pdf/2510.19210)**

> **作者:** In-Hwan Jin; Hyeongju Mun; Joonsoo Kim; Kugjin Yun; Kyeongbo Kong
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Recent advances in dynamic scene reconstruction have significantly benefited from 3D Gaussian Splatting, yet existing methods show inconsistent performance across diverse scenes, indicating no single approach effectively handles all dynamic challenges. To overcome these limitations, we propose Mixture of Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework integrating multiple specialized experts via a novel Volume-aware Pixel Router. Unlike sparsity-oriented MoE architectures in large language models, MoE-GS is designed to improve dynamic novel view synthesis quality by combining heterogeneous deformation priors, rather than to reduce training or inference-time FLOPs. Our router adaptively blends expert outputs by projecting volumetric Gaussian-level weights into pixel space through differentiable weight splatting, ensuring spatially and temporally coherent results. Although MoE-GS improves rendering quality, the increased model capacity and reduced FPS are inherent to the MoE architecture. To mitigate this, we explore two complementary directions: (1) single-pass multi-expert rendering and gate-aware Gaussian pruning, which improve efficiency within the MoE framework, and (2) a distillation strategy that transfers MoE performance to individual experts, enabling lightweight deployment without architectural changes. To the best of our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts techniques into dynamic Gaussian splatting. Extensive experiments on the N3V and Technicolor datasets demonstrate that MoE-GS consistently outperforms state-of-the-art methods with improved efficiency. Video demonstrations are available at this http URL.
>
---
#### [replaced 196] AnyPcc: Compressing Any Point Cloud with a Single Universal Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20331](https://arxiv.org/pdf/2510.20331)**

> **作者:** Kangli Wang; Qianxi Yi; Yuqi Ye; Shihao Li; Wei Gao
>
> **备注:** CVPR 2026
>
> **摘要:** Generalization remains a critical challenge in deep learning-based point cloud geometry compression. While existing methods perform well on standard benchmarks, their performance collapses in real-world scenarios due to two fundamental limitations: the lack of context models that are robust across diverse data densities, and the inability to efficiently adapt to out-of-distribution (OOD) data. To overcome both challenges, we introduce AnyPcc, a universal point cloud compression framework. AnyPcc first employs a Universal Context Model that leverages coarse-grained spatial priors with fine-grained channel priors to ensure robust context modeling across the entire density spectrum. Second, our novel Instance-Adaptive Fine-Tuning (IAFT) strategy tackles OOD data by synergizing explicit and implicit compression paradigms. For each instance, it fine-tunes a small subset of network weights and transmits them within the bitstream. The minimal bitrate overhead from these weights is significantly outweighed by the resulting gains in geometry compression. Extensive experiments on a benchmark of 15 diverse datasets confirm that AnyPcc sets a new state-of-the-art in point cloud compression while maintaining low complexity. Our code and datasets have been released to encourage reproducible research.
>
---
#### [replaced 197] MetricNet: Recovering Metric Scale in Generative Navigation Policies
- **分类: cs.RO; cs.CV**

- **简介: 论文提出MetricNet，解决生成式导航中路径无度量尺度和短视问题，通过预测路标间距离提升导航安全性与效果。**

- **链接: [https://arxiv.org/pdf/2509.13965](https://arxiv.org/pdf/2509.13965)**

> **作者:** Abhijeet Nayak; Débora Oliveira Makowski; Samiran Gode; Cordelia Schmid; Wolfram Burgard
>
> **备注:** Accepted to ICRA'26
>
> **摘要:** Generative navigation policies have made rapid progress in improving end-to-end learned navigation. Despite their promising results, this paradigm has two structural problems. First, the sampled trajectories exist in an abstract, unscaled space without metric grounding. Second, the control strategy discards the full path, instead moving directly towards a single waypoint. This leads to short-sighted and unsafe actions, moving the robot towards obstacles that a complete and correctly scaled path would circumvent. To address these issues, we propose MetricNet, an effective add-on for generative navigation that predicts the metric distance between waypoints, grounding policy outputs in metric coordinates. We evaluate our method in simulation with a new benchmarking framework and show that executing MetricNet-scaled waypoints significantly improves both navigation and exploration performance. Beyond simulation, we further validate our approach in real-world experiments. Finally, we propose MetricNav, which integrates MetricNet into a navigation policy to guide the robot away from obstacles while still moving towards the goal.
>
---
