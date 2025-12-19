# 计算机视觉 cs.CV

- **最新发布 130 篇**

- **更新 69 篇**

## 最新发布

#### [new 001] VenusBench-GD: A Comprehensive Multi-Platform GUI Benchmark for Diverse Grounding Tasks
- **分类: cs.CV**

- **简介: 该论文面向GUI接地任务，旨在解决现有基准数据量少、跨平台弱、任务单一等问题。作者构建了多平台、双语、分层的VenusBench-GD基准，提出高质量标注流程与六类接地子任务，并通过实验揭示通用多模态模型在基础任务上已媲美专用模型，但高级任务仍需改进。**

- **链接: [https://arxiv.org/pdf/2512.16501v1](https://arxiv.org/pdf/2512.16501v1)**

> **作者:** Beitong Zhou; Zhexiao Huang; Yuan Guo; Zhangxuan Gu; Tianyu Xia; Zichen Luo; Fei Tang; Dehan Kong; Yanyi Shang; Suling Ou; Zhenlin Guo; Changhua Meng; Shuheng Shen
>
> **摘要:** GUI grounding is a critical component in building capable GUI agents. However, existing grounding benchmarks suffer from significant limitations: they either provide insufficient data volume and narrow domain coverage, or focus excessively on a single platform and require highly specialized domain knowledge. In this work, we present VenusBench-GD, a comprehensive, bilingual benchmark for GUI grounding that spans multiple platforms, enabling hierarchical evaluation for real-word applications. VenusBench-GD contributes as follows: (i) we introduce a large-scale, cross-platform benchmark with extensive coverage of applications, diverse UI elements, and rich annotated data, (ii) we establish a high-quality data construction pipeline for grounding tasks, achieving higher annotation accuracy than existing benchmarks, and (iii) we extend the scope of element grounding by proposing a hierarchical task taxonomy that divides grounding into basic and advanced categories, encompassing six distinct subtasks designed to evaluate models from complementary perspectives. Our experimental findings reveal critical insights: general-purpose multimodal models now match or even surpass specialized GUI models on basic grounding tasks. In contrast, advanced tasks, still favor GUI-specialized models, though they exhibit significant overfitting and poor robustness. These results underscore the necessity of comprehensive, multi-tiered evaluation frameworks.
>
---
#### [new 002] Geometric Disentanglement of Text Embeddings for Subject-Consistent Text-to-Image Generation using A Single Prompt
- **分类: cs.CV**

- **简介: 该论文面向文本到图像生成任务，旨在解决多图生成中主体一致性差与文本对齐不准的问题。提出一种无需训练的几何解耦方法，通过几何空间重构文本嵌入以抑制语义纠缠，显著提升主体一致性和文本对齐效果。**

- **链接: [https://arxiv.org/pdf/2512.16443v1](https://arxiv.org/pdf/2512.16443v1)**

> **作者:** Shangxun Li; Youngjung Uh
>
> **摘要:** Text-to-image diffusion models excel at generating high-quality images from natural language descriptions but often fail to preserve subject consistency across multiple outputs, limiting their use in visual storytelling. Existing approaches rely on model fine-tuning or image conditioning, which are computationally expensive and require per-subject optimization. 1Prompt1Story, a training-free approach, concatenates all scene descriptions into a single prompt and rescales token embeddings, but it suffers from semantic leakage, where embeddings across frames become entangled, causing text misalignment. In this paper, we propose a simple yet effective training-free approach that addresses semantic entanglement from a geometric perspective by refining text embeddings to suppress unwanted semantics. Extensive experiments prove that our approach significantly improves both subject consistency and text alignment over existing baselines.
>
---
#### [new 003] Flowing from Reasoning to Motion: Learning 3D Hand Trajectory Prediction from Egocentric Human Interaction Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文聚焦于**3D手部轨迹预测**任务，旨在解决现有方法中运动与语义脱节、推理与动作弱关联的问题。工作包括：构建大规模注视视角（egocentric）数据集EgoMAN（含219K轨迹与3M结构化QA），并提出“推理到运动”框架EgoMAN模型，通过轨迹-词元接口联合视觉语言推理与运动生成。**

- **链接: [https://arxiv.org/pdf/2512.16907v1](https://arxiv.org/pdf/2512.16907v1)**

> **作者:** Mingfei Chen; Yifan Wang; Zhengqin Li; Homanga Bharadhwaj; Yujin Chen; Chuan Qin; Ziyi Kou; Yuan Tian; Eric Whitmire; Rajinder Sodhi; Hrvoje Benko; Eli Shlizerman; Yue Liu
>
> **备注:** Project website: https://egoman-project.github.io
>
> **摘要:** Prior works on 3D hand trajectory prediction are constrained by datasets that decouple motion from semantic supervision and by models that weakly link reasoning and action. To address these, we first present the EgoMAN dataset, a large-scale egocentric dataset for interaction stage-aware 3D hand trajectory prediction with 219K 6DoF trajectories and 3M structured QA pairs for semantic, spatial, and motion reasoning. We then introduce the EgoMAN model, a reasoning-to-motion framework that links vision-language reasoning and motion generation via a trajectory-token interface. Trained progressively to align reasoning with motion dynamics, our approach yields accurate and stage-aware trajectories with generalization across real-world scenes.
>
---
#### [new 004] Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属三维人脸重建任务，旨在从少量未标定图像中高保真恢复人脸几何与纹理。提出基于高斯点绘（Gaussian Splatting）的方法，结合语义分割对齐、三角网格软约束及光照解耦，生成可直接接入标准图形管线的几何模型与去光照纹理。**

- **链接: [https://arxiv.org/pdf/2512.16397v1](https://arxiv.org/pdf/2512.16397v1)**

> **作者:** Haodi He; Jihun Yu; Ronald Fedkiw
>
> **备注:** Submitted to CVPR 2026. 21 pages, 22 figures
>
> **摘要:** We leverage increasingly popular three-dimensional neural representations in order to construct a unified and consistent explanation of a collection of uncalibrated images of the human face. Our approach utilizes Gaussian Splatting, since it is more explicit and thus more amenable to constraints than NeRFs. We leverage segmentation annotations to align the semantic regions of the face, facilitating the reconstruction of a neutral pose from only 11 images (as opposed to requiring a long video). We soft constrain the Gaussians to an underlying triangulated surface in order to provide a more structured Gaussian Splat reconstruction, which in turn informs subsequent perturbations to increase the accuracy of the underlying triangulated surface. The resulting triangulated surface can then be used in a standard graphics pipeline. In addition, and perhaps most impactful, we show how accurate geometry enables the Gaussian Splats to be transformed into texture space where they can be treated as a view-dependent neural texture. This allows one to use high visual fidelity Gaussian Splatting on any asset in a scene without the need to modify any other asset or any other aspect (geometry, lighting, renderer, etc.) of the graphics pipeline. We utilize a relightable Gaussian model to disentangle texture from lighting in order to obtain a delit high-resolution albedo texture that is also readily usable in a standard graphics pipeline. The flexibility of our system allows for training with disparate images, even with incompatible lighting, facilitating robust regularization. Finally, we demonstrate the efficacy of our approach by illustrating its use in a text-driven asset creation pipeline.
>
---
#### [new 005] BrepLLM: Native Boundary Representation Understanding with Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出BrepLLM，解决LLM难以直接理解3D边界表示（Brep）几何与拓扑信息的问题。通过两阶段训练（跨模态对齐预训练+多阶段LLM微调）和自建Brep2Text数据集，实现Brep到文本的端到端理解，在3D分类与描述任务达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.16413v1](https://arxiv.org/pdf/2512.16413v1)**

> **作者:** Liyuan Deng; Hao Guo; Yunpeng Bai; Yongkang Dai; Huaxi Huang; Yilei Shi
>
> **摘要:** Current token-sequence-based Large Language Models (LLMs) are not well-suited for directly processing 3D Boundary Representation (Brep) models that contain complex geometric and topological information. We propose BrepLLM, the first framework that enables LLMs to parse and reason over raw Brep data, bridging the modality gap between structured 3D geometry and natural language. BrepLLM employs a two-stage training pipeline: Cross-modal Alignment Pre-training and Multi-stage LLM Fine-tuning. In the first stage, an adaptive UV sampling strategy converts Breps into graphs representation with geometric and topological information. We then design a hierarchical BrepEncoder to extract features from geometry (i.e., faces and edges) and topology, producing both a single global token and a sequence of node tokens. Then we align the global token with text embeddings from a frozen CLIP text encoder (ViT-L/14) via contrastive learning. In the second stage, we integrate the pretrained BrepEncoder into an LLM. We then align its sequence of node tokens using a three-stage progressive training strategy: (1) training an MLP-based semantic mapping from Brep representation to 2D with 2D-LLM priors. (2) performing fine-tuning of the LLM. (3) designing a Mixture-of-Query Experts (MQE) to enhance geometric diversity modeling. We also construct Brep2Text, a dataset comprising 269,444 Brep-text question-answer pairs. Experiments show that BrepLLM achieves state-of-the-art (SOTA) results on 3D object classification and captioning tasks.
>
---
#### [new 006] Differences That Matter: Auditing Models for Capability Gap Discovery and Rectification
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AuditDM框架，属模型诊断与改进任务，旨在解决多模态大模型能力差距难发现、难解释的问题。通过强化学习训练审计员模型生成挑战性问题和反事实图像，主动暴露模型分歧与弱点，并利用发现的失败样例无标注微调，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.16921v1](https://arxiv.org/pdf/2512.16921v1)**

> **作者:** Qihao Liu; Chengzhi Mao; Yaojie Liu; Alan Yuille; Wen-Sheng Chu
>
> **备注:** project page: https://auditdm.github.io/
>
> **摘要:** Conventional evaluation methods for multimodal LLMs (MLLMs) lack interpretability and are often insufficient to fully disclose significant capability gaps across models. To address this, we introduce AuditDM, an automated framework that actively discovers and rectifies MLLM failure modes by auditing their divergence. AuditDM fine-tunes an MLLM as an auditor via reinforcement learning to generate challenging questions and counterfactual images that maximize disagreement among target models. Once trained, the auditor uncovers diverse, interpretable exemplars that reveal model weaknesses and serve as annotation-free data for rectification. When applied to SoTA models like Gemma-3 and PaliGemma-2, AuditDM discovers more than 20 distinct failure types. Fine-tuning on these discoveries consistently improves all models across 16 benchmarks, and enables a 3B model to surpass its 28B counterpart. Our results suggest that as data scaling hits diminishing returns, targeted model auditing offers an effective path to model diagnosis and improvement.
>
---
#### [new 007] R4: Retrieval-Augmented Reasoning for Vision-Language Models in 4D Spatio-Temporal Space
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出R4框架，面向具身智能中的4D时空推理任务，解决VLM缺乏持续时空记忆与联合推理能力的问题。它无需训练，通过构建锚定在度量空间与时间的4D知识库，支持语义、空间、时间三键检索，实现跨场景、协作式推理。**

- **链接: [https://arxiv.org/pdf/2512.15940v1](https://arxiv.org/pdf/2512.15940v1)**

> **作者:** Tin Stribor Sohn; Maximilian Dillitzer; Jason J. Corso; Eric Sax
>
> **摘要:** Humans perceive and reason about their surroundings in four dimensions by building persistent, structured internal representations that encode semantic meaning, spatial layout, and temporal dynamics. These multimodal memories enable them to recall past events, infer unobserved states, and integrate new information into context-dependent reasoning. Inspired by this capability, we introduce R4, a training-free framework for retrieval-augmented reasoning in 4D spatio-temporal space that equips vision-language models (VLMs) with structured, lifelong memory. R4 continuously constructs a 4D knowledge database by anchoring object-level semantic descriptions in metric space and time, yielding a persistent world model that can be shared across agents. At inference, natural language queries are decomposed into semantic, spatial, and temporal keys to retrieve relevant observations, which are integrated into the VLM's reasoning. Unlike classical retrieval-augmented generation methods, retrieval in R4 operates directly in 4D space, enabling episodic and collaborative reasoning without training. Experiments on embodied question answering and navigation benchmarks demonstrate that R4 substantially improves retrieval and reasoning over spatio-temporal information compared to baselines, advancing a new paradigm for embodied 4D reasoning in dynamic environments.
>
---
#### [new 008] Collaborative Edge-to-Server Inference for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向视觉-语言模型（VLM）的边缘-云协同推理任务，旨在解决边缘端上传压缩图像导致细节丢失、精度下降与通信开销大的问题。提出两阶段框架：服务器先用全局图推理并基于输出熵判断是否需重传局部高保真图，再融合双图精调结果，实现低通信、高精度推理。**

- **链接: [https://arxiv.org/pdf/2512.16349v1](https://arxiv.org/pdf/2512.16349v1)**

> **作者:** Soochang Song; Yongjune Kim
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** We propose a collaborative edge-to-server inference framework for vision-language models (VLMs) that reduces the communication cost while maintaining inference accuracy. In typical deployments, visual data captured at edge devices (clients) is transmitted to the server for VLM inference. However, resizing the original image (global image) to match the vision encoder's input resolution often discards fine-grained details, leading to accuracy degradation. To overcome this limitation, we design a two-stage framework. In the first stage, the server performs inference on the global image and identifies a region of interest (RoI) using the VLM's internal attention. The min-entropy of the output tokens is then computed as a confidence measure to determine whether retransmission is required. If the min-entropy exceeds a predefined threshold, the server requests the edge device to send a detail-preserved local image of the RoI. The server then refines its inference by jointly leveraging the global and local images. This selective retransmission strategy ensures that only essential visual content is transmitted. Experiments across multiple VLM architectures show that the proposed framework significantly reduces communication cost while maintaining inference accuracy.
>
---
#### [new 009] ResDynUNet++: A nested U-Net with residual dynamic convolution blocks for dual-spectral CT
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文面向双能CT（DSCT）图像重建任务，旨在解决基物质分解中通道不平衡与界面伪影问题。提出ResDynUNet++网络，结合知识驱动的斜投影修正法（OPMT）生成初解，并用含残差动态卷积的UNet++精修，提升重建精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.16140v1](https://arxiv.org/pdf/2512.16140v1)**

> **作者:** Ze Yuan; Wenbin Li; Shusen Zhao
>
> **摘要:** We propose a hybrid reconstruction framework for dual-spectral CT (DSCT) that integrates iterative methods with deep learning models. The reconstruction process consists of two complementary components: a knowledge-driven module and a data-driven module. In the knowledge-driven phase, we employ the oblique projection modification technique (OPMT) to reconstruct an intermediate solution of the basis material images from the projection data. We select OPMT for this role because of its fast convergence, which allows it to rapidly generate an intermediate solution that successfully achieves basis material decomposition. Subsequently, in the data-driven phase, we introduce a novel neural network, ResDynUNet++, to refine this intermediate solution. The ResDynUNet++ is built upon a UNet++ backbone by replacing standard convolutions with residual dynamic convolution blocks, which combine the adaptive, input-specific feature extraction of dynamic convolution with the stable training of residual connections. This architecture is designed to address challenges like channel imbalance and near-interface large artifacts in DSCT, producing clean and accurate final solutions. Extensive experiments on both synthetic phantoms and real clinical datasets validate the efficacy and superior performance of the proposed method.
>
---
#### [new 010] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向AR/VR中基于头显稀疏信号的全身运动跟踪任务，旨在解决精度、时序连贯性与效率难以兼顾的问题。提出KineST模型：引入运动学引导的双向扫描与混合时空表征学习，并设计几何角速度损失，实现轻量、高精度、高稳定性的姿态重建。**

- **链接: [https://arxiv.org/pdf/2512.16791v1](https://arxiv.org/pdf/2512.16791v1)**

> **作者:** Shuting Zhao; Zeyu Xiao; Xinrong Chen
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Full-body motion tracking plays an essential role in AR/VR applications, bridging physical and virtual interactions. However, it is challenging to reconstruct realistic and diverse full-body poses based on sparse signals obtained by head-mounted displays, which are the main devices in AR/VR scenarios. Existing methods for pose reconstruction often incur high computational costs or rely on separately modeling spatial and temporal dependencies, making it difficult to balance accuracy, temporal coherence, and efficiency. To address this problem, we propose KineST, a novel kinematics-guided state space model, which effectively extracts spatiotemporal dependencies while integrating local and global pose perception. The innovation comes from two core ideas. Firstly, in order to better capture intricate joint relationships, the scanning strategy within the State Space Duality framework is reformulated into kinematics-guided bidirectional scanning, which embeds kinematic priors. Secondly, a mixed spatiotemporal representation learning approach is employed to tightly couple spatial and temporal contexts, balancing accuracy and smoothness. Additionally, a geometric angular velocity loss is introduced to impose physically meaningful constraints on rotational variations for further improving motion stability. Extensive experiments demonstrate that KineST has superior performance in both accuracy and temporal consistency within a lightweight framework. Project page: https://kaka-1314.github.io/KineST/
>
---
#### [new 011] City Navigation in the Wild: Exploring Emergent Navigation from Web-Scale Knowledge in MLLMs
- **分类: cs.CV**

- **简介: 该论文提出“稀疏标注视觉导航”任务，旨在评估MLLM在真实城市环境中的自主导航能力。针对现有基准偏重语言或仿真、缺乏知识密集型实景评测的问题，构建CityNav多城基准，并提出VoP方法提升定位与路径规划效果。**

- **链接: [https://arxiv.org/pdf/2512.15933v1](https://arxiv.org/pdf/2512.15933v1)**

> **作者:** Dwip Dalal; Utkarsh Mishra; Narendra Ahuja; Nebojsa Jojic
>
> **摘要:** Leveraging multimodal large language models (MLLMs) to develop embodied agents offers significant promise for addressing complex real-world tasks. However, current evaluation benchmarks remain predominantly language-centric or heavily reliant on simulated environments, rarely probing the nuanced, knowledge-intensive reasoning essential for practical, real-world scenarios. To bridge this critical gap, we introduce the task of Sparsely Grounded Visual Navigation, explicitly designed to evaluate the sequential decision-making abilities of MLLMs in challenging, knowledge-intensive real-world environments. We operationalize this task with CityNav, a comprehensive benchmark encompassing four diverse global cities, specifically constructed to assess raw MLLM-driven agents in city navigation. Agents are required to rely solely on visual inputs and internal multimodal reasoning to sequentially navigate 50+ decision points without additional environmental annotations or specialized architectural modifications. Crucially, agents must autonomously achieve localization through interpreting city-specific cues and recognizing landmarks, perform spatial reasoning, and strategically plan and execute routes to their destinations. Through extensive evaluations, we demonstrate that current state-of-the-art MLLMs and standard reasoning techniques (e.g., Chain-of-Thought, Reflection) significantly underperform in this challenging setting. To address this, we propose Verbalization of Path (VoP), which explicitly grounds the agent's internal reasoning by probing an explicit cognitive map (key landmarks and directions toward the destination) from the MLLMs, substantially enhancing navigation success. Project Webpage: https://dwipddalal.github.io/AgentNav/
>
---
#### [new 012] TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属视频生成任务，旨在解决扩散模型推理速度慢的问题。提出TurboDiffusion框架，通过注意力加速（SageAttention与稀疏线性注意力）、步数蒸馏（rCM）和W8A8量化等技术，实现100–200倍加速，同时保持视频质量。**

- **链接: [https://arxiv.org/pdf/2512.16093v1](https://arxiv.org/pdf/2512.16093v1)**

> **作者:** Jintao Zhang; Kaiwen Zheng; Kai Jiang; Haoxu Wang; Ion Stoica; Joseph E. Gonzalez; Jianfei Chen; Jun Zhu
>
> **摘要:** We introduce TurboDiffusion, a video generation acceleration framework that can speed up end-to-end diffusion generation by 100-200x while maintaining video quality. TurboDiffusion mainly relies on several components for acceleration: (1) Attention acceleration: TurboDiffusion uses low-bit SageAttention and trainable Sparse-Linear Attention (SLA) to speed up attention computation. (2) Step distillation: TurboDiffusion adopts rCM for efficient step distillation. (3) W8A8 quantization: TurboDiffusion quantizes model parameters and activations to 8 bits to accelerate linear layers and compress the model. In addition, TurboDiffusion incorporates several other engineering optimizations. We conduct experiments on the Wan2.2-I2V-14B-720P, Wan2.1-T2V-1.3B-480P, Wan2.1-T2V-14B-720P, and Wan2.1-T2V-14B-480P models. Experimental results show that TurboDiffusion achieves 100-200x speedup for video generation even on a single RTX 5090 GPU, while maintaining comparable video quality. The GitHub repository, which includes model checkpoints and easy-to-use code, is available at https://github.com/thu-ml/TurboDiffusion.
>
---
#### [new 013] GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GeoPredict，一种面向机器人操纵的几何感知VLA框架，旨在解决现有VLA模型在3D空间推理与精确操作上的不足。通过引入预测性运动学与3D高斯几何模块，在训练时提供几何监督，推理时仅增轻量查询令牌，显著提升几何密集型任务性能。**

- **链接: [https://arxiv.org/pdf/2512.16811v1](https://arxiv.org/pdf/2512.16811v1)**

> **作者:** Jingjing Qian; Boyao Han; Chen Shi; Lei Xiao; Long Yang; Shaoshuai Shi; Li Jiang
>
> **摘要:** Vision-Language-Action (VLA) models achieve strong generalization in robotic manipulation but remain largely reactive and 2D-centric, making them unreliable in tasks that require precise 3D reasoning. We propose GeoPredict, a geometry-aware VLA framework that augments a continuous-action policy with predictive kinematic and geometric priors. GeoPredict introduces a trajectory-level module that encodes motion history and predicts multi-step 3D keypoint trajectories of robot arms, and a predictive 3D Gaussian geometry module that forecasts workspace geometry with track-guided refinement along future keypoint trajectories. These predictive modules serve exclusively as training-time supervision through depth-based rendering, while inference requires only lightweight additional query tokens without invoking any 3D decoding. Experiments on RoboCasa Human-50, LIBERO, and real-world manipulation tasks show that GeoPredict consistently outperforms strong VLA baselines, especially in geometry-intensive and spatially demanding scenarios.
>
---
#### [new 014] Adaptive Frequency Domain Alignment Network for Medical image segmentation
- **分类: cs.CV**

- **简介: 该论文属医学图像分割任务，旨在缓解标注数据稀缺导致的跨域性能下降问题。提出自适应频域对齐网络（AFDAN），通过对抗域学习、源-目标频域融合及空-频特征集成三大模块，实现鲁棒跨域知识迁移，在白癜风和视网膜血管分割上达SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.16393v1](https://arxiv.org/pdf/2512.16393v1)**

> **作者:** Zhanwei Li; Liang Li; Jiawan Zhang
>
> **摘要:** High-quality annotated data plays a crucial role in achieving accurate segmentation. However, such data for medical image segmentation are often scarce due to the time-consuming and labor-intensive nature of manual annotation. To address this challenge, we propose the Adaptive Frequency Domain Alignment Network (AFDAN)--a novel domain adaptation framework designed to align features in the frequency domain and alleviate data scarcity. AFDAN integrates three core components to enable robust cross-domain knowledge transfer: an Adversarial Domain Learning Module that transfers features from the source to the target domain; a Source-Target Frequency Fusion Module that blends frequency representations across domains; and a Spatial-Frequency Integration Module that combines both frequency and spatial features to further enhance segmentation accuracy across domains. Extensive experiments demonstrate the effectiveness of AFDAN: it achieves an Intersection over Union (IoU) of 90.9% for vitiligo segmentation in the newly constructed VITILIGO2025 dataset and a competitive IoU of 82.6% on the retinal vessel segmentation benchmark DRIVE, surpassing existing state-of-the-art approaches.
>
---
#### [new 015] Driving in Corner Case: A Real-World Adversarial Closed-Loop Evaluation Platform for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向端到端自动驾驶的**安全评估任务**，解决真实世界中难采集的**安全关键边缘案例（corner cases）评估难**问题。提出首个**真实场景闭环对抗评估平台**：用基于流匹配的图像生成器合成真实感图像，结合高效对抗交通策略，主动激发模型失效，验证其鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.16055v1](https://arxiv.org/pdf/2512.16055v1)**

> **作者:** Jiaheng Geng; Jiatong Du; Xinyu Zhang; Ye Li; Panqu Wang; Yanjun Huang
>
> **摘要:** Safety-critical corner cases, difficult to collect in the real world, are crucial for evaluating end-to-end autonomous driving. Adversarial interaction is an effective method to generate such safety-critical corner cases. While existing adversarial evaluation methods are built for models operating in simplified simulation environments, adversarial evaluation for real-world end-to-end autonomous driving has been little explored. To address this challenge, we propose a closed-loop evaluation platform for end-to-end autonomous driving, which can generate adversarial interactions in real-world scenes. In our platform, the real-world image generator cooperates with an adversarial traffic policy to evaluate various end-to-end models trained on real-world data. The generator, based on flow matching, efficiently and stably generates real-world images according to the traffic environment information. The efficient adversarial surrounding vehicle policy is designed to model challenging interactions and create corner cases that current autonomous driving systems struggle to handle. Experimental results demonstrate that the platform can generate realistic driving images efficiently. Through evaluating the end-to-end models such as UniAD and VAD, we demonstrate that based on the adversarial policy, our platform evaluates the performance degradation of the tested model in corner cases. This result indicates that this platform can effectively detect the model's potential issues, which will facilitate the safety and robustness of end-to-end autonomous driving.
>
---
#### [new 016] Open Ad-hoc Categorization with Contextualized Feature Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究开放式即兴视觉分类任务，旨在基于少量标注样本和大量无标签数据，动态发现语义上下文并扩展即兴类别。提出OAK模型：在冻结CLIP基础上引入可学习上下文令牌，联合优化图文对齐与视觉聚类目标，在准确率和概念发现上达SOTA，并生成可解释的显著图。**

- **链接: [https://arxiv.org/pdf/2512.16202v1](https://arxiv.org/pdf/2512.16202v1)**

> **作者:** Zilin Wang; Sangwoo Mo; Stella X. Yu; Sima Behpour; Liu Ren
>
> **备注:** 26 pages, 17 figures
>
> **摘要:** Adaptive categorization of visual scenes is essential for AI agents to handle changing tasks. Unlike fixed common categories for plants or animals, ad-hoc categories are created dynamically to serve specific goals. We study open ad-hoc categorization: Given a few labeled exemplars and abundant unlabeled data, the goal is to discover the underlying context and to expand ad-hoc categories through semantic extension and visual clustering around it. Building on the insight that ad-hoc and common categories rely on similar perceptual mechanisms, we propose OAK, a simple model that introduces a small set of learnable context tokens at the input of a frozen CLIP and optimizes with both CLIP's image-text alignment objective and GCD's visual clustering objective. On Stanford and Clevr-4 datasets, OAK achieves state-of-the-art in accuracy and concept discovery across multiple categorizations, including 87.4% novel accuracy on Stanford Mood, surpassing CLIP and GCD by over 50%. Moreover, OAK produces interpretable saliency maps, focusing on hands for Action, faces for Mood, and backgrounds for Location, promoting transparency and trust while enabling adaptive and generalizable categorization.
>
---
#### [new 017] Skeleton-Snippet Contrastive Learning with Multiscale Feature Fusion for Action Localization
- **分类: cs.CV**

- **简介: 该论文面向骨架动作定位任务，解决自监督预训练中时序敏感特征不足的问题。提出片段判别式对比学习 pretext 任务，并设计U型多尺度特征融合模块，提升帧级定位能力，在BABEL和PKUMMD上取得SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.16504v1](https://arxiv.org/pdf/2512.16504v1)**

> **作者:** Qiushuo Cheng; Jingjing Liu; Catherine Morgan; Alan Whone; Majid Mirmehdi
>
> **摘要:** The self-supervised pretraining paradigm has achieved great success in learning 3D action representations for skeleton-based action recognition using contrastive learning. However, learning effective representations for skeleton-based temporal action localization remains challenging and underexplored. Unlike video-level {action} recognition, detecting action boundaries requires temporally sensitive features that capture subtle differences between adjacent frames where labels change. To this end, we formulate a snippet discrimination pretext task for self-supervised pretraining, which densely projects skeleton sequences into non-overlapping segments and promotes features that distinguish them across videos via contrastive learning. Additionally, we build on strong backbones of skeleton-based action recognition models by fusing intermediate features with a U-shaped module to enhance feature resolution for frame-level localization. Our approach consistently improves existing skeleton-based contrastive learning methods for action localization on BABEL across diverse subsets and evaluation protocols. We also achieve state-of-the-art transfer learning performance on PKUMMD with pretraining on NTU RGB+D and BABEL.
>
---
#### [new 018] Plug to Place: Indoor Multimedia Geolocation from Electrical Sockets for Digital Investigation
- **分类: cs.CV**

- **简介: 该论文属数字取证中的室内多媒体地理定位任务，旨在解决室内场景因GPS失效、环境多变导致的定位难问题。提出基于电源插座类型（国家/地区标准化）的三阶段深度学习 pipeline：检测→分类→国家映射，并构建专用数据集，在真实酒店图像上验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.16620v1](https://arxiv.org/pdf/2512.16620v1)**

> **作者:** Kanwal Aftab; Graham Adams; Mark Scanlon
>
> **摘要:** Computer vision is a rapidly evolving field, giving rise to powerful new tools and techniques in digital forensic investigation, and shows great promise for novel digital forensic applications. One such application, indoor multimedia geolocation, has the potential to become a crucial aid for law enforcement in the fight against human trafficking, child exploitation, and other serious crimes. While outdoor multimedia geolocation has been widely explored, its indoor counterpart remains underdeveloped due to challenges such as similar room layouts, frequent renovations, visual ambiguity, indoor lighting variability, unreliable GPS signals, and limited datasets in sensitive domains. This paper introduces a pipeline that uses electric sockets as consistent indoor markers for geolocation, since plug socket types are standardised by country or region. The three-stage deep learning pipeline detects plug sockets (YOLOv11, mAP@0.5 = 0.843), classifies them into one of 12 plug socket types (Xception, accuracy = 0.912), and maps the detected socket types to countries (accuracy = 0.96 at >90% threshold confidence). To address data scarcity, two dedicated datasets were created: socket detection dataset of 2,328 annotated images expanded to 4,072 through augmentation, and a classification dataset of 3,187 images across 12 plug socket classes. The pipeline was evaluated on the Hotels-50K dataset, focusing on the TraffickCam subset of crowd-sourced hotel images, which capture real-world conditions such as poor lighting and amateur angles. This dataset provides a more realistic evaluation than using professional, well-lit, often wide-angle images from travel websites. This framework demonstrates a practical step toward real-world digital forensic applications. The code, trained models, and the data for this paper are available open source.
>
---
#### [new 019] Radiology Report Generation with Layer-Wise Anatomical Attention
- **分类: cs.CV**

- **简介: 该论文属医学图像到文本生成任务，旨在解决现有胸片报告生成模型资源消耗大、依赖多模态输入的问题。作者提出轻量级单图生成模型，用冻结DINOv3编码器与GPT-2解码器结合层式解剖注意力机制，仅凭单张正位胸片生成报告“Findings”部分，显著提升病理识别与结构连贯性。**

- **链接: [https://arxiv.org/pdf/2512.16841v1](https://arxiv.org/pdf/2512.16841v1)**

> **作者:** Emmanuel D. Muñiz-De-León; Jorge A. Rosales-de-Golferichs; Ana S. Muñoz-Rodríguez; Alejandro I. Trejo-Castro; Eduardo de Avila-Armenta; Antonio Martínez-Torteya
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Automatic radiology report generation is a promising application of multimodal deep learning, aiming to reduce reporting workload and improve consistency. However, current state-of-the-art (SOTA) systems - such as Multimodal AI for Radiology Applications (MAIRA-2) and Medical Pathways Language Model-Multimodal (MedPaLM-M) - depend on large-scale multimodal training, clinical metadata, and multiple imaging views, making them resource-intensive and inaccessible for most settings. We introduce a compact image-to-text architecture that generates the Findings section of chest X-ray reports from a single frontal image. The model combines a frozen Self-Distillation with No Labels v3 (DINOv3) Vision Transformer (ViT) encoder with a Generative Pre-trained Transformer 2 (GPT-2) decoder enhanced by layer-wise anatomical attention. This mechanism integrates lung and heart segmentation masks through hierarchical Gaussian smoothing, biasing attention toward clinically relevant regions without adding trainable parameters. Evaluated on the official Medical Information Mart for Intensive Care-Chest X-ray (MIMIC-CXR) dataset using Chest Radiograph Expert (CheXpert) and Radiology Graph (RadGraph) metrics, our approach achieved substantial gains: CheXpert Macro-F1 for five key pathologies increased by 168% (0.083 -> 0.238) and Micro-F1 by 146% (0.137 -> 0.337), while broader performance across 14 observations improved by 86% (0.170 -> 0.316). Structural coherence also improved, with RadGraph F1 rising by 9.7%. Despite its small size and purely image-conditioned design, the model demonstrates that decoder-level anatomical guidance improves spatial grounding and enhances coherence in clinically relevant regions. The source code is publicly available at: https://github.com/devMuniz02/UDEM-CXR-Reporting-Thesis-2025.
>
---
#### [new 020] Enhanced 3D Shape Analysis via Information Geometry
- **分类: cs.CV; math.DG**

- **简介: 该论文属3D点云形状分析任务，旨在解决点云比较中传统度量对全局统计结构捕捉不足、易受噪声干扰及KL散度不稳定的难题。提出基于信息几何的GMM表示框架，定义有界稳定的MSKL散度，并在人体姿态与动物形状数据集上验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.16213v1](https://arxiv.org/pdf/2512.16213v1)**

> **作者:** Amit Vishwakarma; K. S. Subrahamanian Moosath
>
> **摘要:** Three-dimensional point clouds provide highly accurate digital representations of objects, essential for applications in computer graphics, photogrammetry, computer vision, and robotics. However, comparing point clouds faces significant challenges due to their unstructured nature and the complex geometry of the surfaces they represent. Traditional geometric metrics such as Hausdorff and Chamfer distances often fail to capture global statistical structure and exhibit sensitivity to outliers, while existing Kullback-Leibler (KL) divergence approximations for Gaussian Mixture Models can produce unbounded or numerically unstable values. This paper introduces an information geometric framework for 3D point cloud shape analysis by representing point clouds as Gaussian Mixture Models (GMMs) on a statistical manifold. We prove that the space of GMMs forms a statistical manifold and propose the Modified Symmetric Kullback-Leibler (MSKL) divergence with theoretically guaranteed upper and lower bounds, ensuring numerical stability for all GMM comparisons. Through comprehensive experiments on human pose discrimination (MPI-FAUST dataset) and animal shape comparison (G-PCD dataset), we demonstrate that MSKL provides stable and monotonically varying values that directly reflect geometric variation, outperforming traditional distances and existing KL approximations.
>
---
#### [new 021] 4D Primitive-Mâché: Glueing Primitives for Persistent 4D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于4D场景重建任务，旨在解决单目视频中动态场景的完整、持久化重建问题。提出4D Primitive-Mâché方法：将场景分解为刚性3D基元，通过稠密2D对应估计其联合刚体运动，并外推遮挡物体运动，实现可回放的时空一致4D重建。**

- **链接: [https://arxiv.org/pdf/2512.16564v1](https://arxiv.org/pdf/2512.16564v1)**

> **作者:** Kirill Mazur; Marwan Taher; Andrew J. Davison
>
> **备注:** For project page, see https://makezur.github.io/4DPM/
>
> **摘要:** We present a dynamic reconstruction system that receives a casual monocular RGB video as input, and outputs a complete and persistent reconstruction of the scene. In other words, we reconstruct not only the the currently visible parts of the scene, but also all previously viewed parts, which enables replaying the complete reconstruction across all timesteps. Our method decomposes the scene into a set of rigid 3D primitives, which are assumed to be moving throughout the scene. Using estimated dense 2D correspondences, we jointly infer the rigid motion of these primitives through an optimisation pipeline, yielding a 4D reconstruction of the scene, i.e. providing 3D geometry dynamically moving through time. To achieve this, we also introduce a mechanism to extrapolate motion for objects that become invisible, employing motion-grouping techniques to maintain continuity. The resulting system enables 4D spatio-temporal awareness, offering capabilities such as replayable 3D reconstructions of articulated objects through time, multi-object scanning, and object permanence. On object scanning and multi-object datasets, our system significantly outperforms existing methods both quantitatively and qualitatively.
>
---
#### [new 022] Ridge Estimation-Based Vision and Laser Ranging Fusion Localization Method for UAVs
- **分类: cs.CV**

- **简介: 该论文属无人机目标定位任务，旨在解决长距、小交角、大倾角下视觉与激光测距融合定位中因设计矩阵病态导致的精度低、鲁棒性差问题。提出基于岭估计的融合定位方法，抑制多重共线性，提升定位精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.16314v1](https://arxiv.org/pdf/2512.16314v1)**

> **作者:** Huayu Huang; Chen Chen; Banglei Guan; Ze Tan; Yang Shang; Zhang Li; Qifeng Yu
>
> **摘要:** Tracking and measuring targets using a variety of sensors mounted on UAVs is an effective means to quickly and accurately locate the target. This paper proposes a fusion localization method based on ridge estimation, combining the advantages of rich scene information from sequential imagery with the high precision of laser ranging to enhance localization accuracy. Under limited conditions such as long distances, small intersection angles, and large inclination angles, the column vectors of the design matrix have serious multicollinearity when using the least squares estimation algorithm. The multicollinearity will lead to ill-conditioned problems, resulting in significant instability and low robustness. Ridge estimation is introduced to mitigate the serious multicollinearity under the condition of limited observation. Experimental results demonstrate that our method achieves higher localization accuracy compared to ground localization algorithms based on single information. Moreover, the introduction of ridge estimation effectively enhances the robustness, particularly under limited observation conditions.
>
---
#### [new 023] PoseMoE: Mixture-of-Experts Network for Monocular 3D Human Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向单目3D人体姿态估计任务，旨在解决 lifting-based 方法中2D姿态与深度特征耦合导致深度不确定性干扰2D精度的问题。提出PoseMoE：采用专家网络解耦2D与深度特征编码，并设计跨专家知识聚合模块增强双向上下文建模。**

- **链接: [https://arxiv.org/pdf/2512.16494v1](https://arxiv.org/pdf/2512.16494v1)**

> **作者:** Mengyuan Liu; Jiajie Liu; Jinyan Zhang; Wenhao Li; Junsong Yuan
>
> **备注:** IEEE Transactions on Image Processing (T-IP)
>
> **摘要:** The lifting-based methods have dominated monocular 3D human pose estimation by leveraging detected 2D poses as intermediate representations. The 2D component of the final 3D human pose benefits from the detected 2D poses, whereas its depth counterpart must be estimated from scratch. The lifting-based methods encode the detected 2D pose and unknown depth in an entangled feature space, explicitly introducing depth uncertainty to the detected 2D pose, thereby limiting overall estimation accuracy. This work reveals that the depth representation is pivotal for the estimation process. Specifically, when depth is in an initial, completely unknown state, jointly encoding depth features with 2D pose features is detrimental to the estimation process. In contrast, when depth is initially refined to a more dependable state via network-based estimation, encoding it together with 2D pose information is beneficial. To address this limitation, we present a Mixture-of-Experts network for monocular 3D pose estimation named PoseMoE. Our approach introduces: (1) A mixture-of-experts network where specialized expert modules refine the well-detected 2D pose features and learn the depth features. This mixture-of-experts design disentangles the feature encoding process for 2D pose and depth, therefore reducing the explicit influence of uncertain depth features on 2D pose features. (2) A cross-expert knowledge aggregation module is proposed to aggregate cross-expert spatio-temporal contextual information. This step enhances features through bidirectional mapping between 2D pose and depth. Extensive experiments show that our proposed PoseMoE outperforms the conventional lifting-based methods on three widely used datasets: Human3.6M, MPI-INF-3DHP, and 3DPW.
>
---
#### [new 024] Instant Expressive Gaussian Head Avatar via 3D-Aware Expression Distillation
- **分类: cs.CV**

- **简介: 该论文属3D人脸动画任务，旨在解决2D扩散模型缺乏3D一致性、3D方法表达力弱的问题。提出基于表达蒸馏的高斯头像方法：将2D扩散模型的知识蒸馏至轻量编码器，实现单图瞬时生成3D一致、高表达力、107 FPS实时可驱动头像。**

- **链接: [https://arxiv.org/pdf/2512.16893v1](https://arxiv.org/pdf/2512.16893v1)**

> **作者:** Kaiwen Jiang; Xueting Li; Seonwook Park; Ravi Ramamoorthi; Shalini De Mello; Koki Nagano
>
> **备注:** Project website is https://research.nvidia.com/labs/amri/projects/instant4d
>
> **摘要:** Portrait animation has witnessed tremendous quality improvements thanks to recent advances in video diffusion models. However, these 2D methods often compromise 3D consistency and speed, limiting their applicability in real-world scenarios, such as digital twins or telepresence. In contrast, 3D-aware facial animation feedforward methods -- built upon explicit 3D representations, such as neural radiance fields or Gaussian splatting -- ensure 3D consistency and achieve faster inference speed, but come with inferior expression details. In this paper, we aim to combine their strengths by distilling knowledge from a 2D diffusion-based method into a feed-forward encoder, which instantly converts an in-the-wild single image into a 3D-consistent, fast yet expressive animatable representation. Our animation representation is decoupled from the face's 3D representation and learns motion implicitly from data, eliminating the dependency on pre-defined parametric models that often constrain animation capabilities. Unlike previous computationally intensive global fusion mechanisms (e.g., multiple attention layers) for fusing 3D structural and animation information, our design employs an efficient lightweight local fusion strategy to achieve high animation expressivity. As a result, our method runs at 107.31 FPS for animation and pose control while achieving comparable animation quality to the state-of-the-art, surpassing alternative designs that trade speed for quality or vice versa. Project website is https://research.nvidia.com/labs/amri/projects/instant4d
>
---
#### [new 025] GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属文本到图像（T2I）模型评估任务，旨在解决基准漂移问题——即旧基准（如GenEval）因模型进步而偏离人类判断。作者提出新基准GenEval 2，增强视觉原语覆盖与组合性，并引入更鲁棒的Soft-TIFA评估法。**

- **链接: [https://arxiv.org/pdf/2512.16853v1](https://arxiv.org/pdf/2512.16853v1)**

> **作者:** Amita Kamath; Kai-Wei Chang; Ranjay Krishna; Luke Zettlemoyer; Yushi Hu; Marjan Ghazvininejad
>
> **摘要:** Automating Text-to-Image (T2I) model evaluation is challenging; a judge model must be used to score correctness, and test prompts must be selected to be challenging for current T2I models but not the judge. We argue that satisfying these constraints can lead to benchmark drift over time, where the static benchmark judges fail to keep up with newer model capabilities. We show that benchmark drift is a significant problem for GenEval, one of the most popular T2I benchmarks. Although GenEval was well-aligned with human judgment at the time of its release, it has drifted far from human judgment over time -- resulting in an absolute error of as much as 17.7% for current models. This level of drift strongly suggests that GenEval has been saturated for some time, as we verify via a large-scale human study. To help fill this benchmarking gap, we introduce a new benchmark, GenEval 2, with improved coverage of primitive visual concepts and higher degrees of compositionality, which we show is more challenging for current models. We also introduce Soft-TIFA, an evaluation method for GenEval 2 that combines judgments for visual primitives, which we show is more well-aligned with human judgment and argue is less likely to drift from human-alignment over time (as compared to more holistic judges such as VQAScore). Although we hope GenEval 2 will provide a strong benchmark for many years, avoiding benchmark drift is far from guaranteed and our work, more generally, highlights the importance of continual audits and improvement for T2I and related automated model evaluation benchmarks.
>
---
#### [new 026] Multi-scale Attention-Guided Intrinsic Decomposition and Rendering Pass Prediction for Facial Images
- **分类: cs.CV; cs.GR**

- **简介: 该论文面向人脸图像本征分解任务，解决无约束光照下准确分离内在属性的难题。提出MAGINet网络，先预测光照归一化漫反射反照率，再经RefinementNet细化，并用Pix2PixHD生成其余五种渲染通道，共六通道完整本征分解，支持高质量重打光与材质编辑。**

- **链接: [https://arxiv.org/pdf/2512.16511v1](https://arxiv.org/pdf/2512.16511v1)**

> **作者:** Hossein Javidnia
>
> **摘要:** Accurate intrinsic decomposition of face images under unconstrained lighting is a prerequisite for photorealistic relighting, high-fidelity digital doubles, and augmented-reality effects. This paper introduces MAGINet, a Multi-scale Attention-Guided Intrinsics Network that predicts a $512\times512$ light-normalized diffuse albedo map from a single RGB portrait. MAGINet employs hierarchical residual encoding, spatial-and-channel attention in a bottleneck, and adaptive multi-scale feature fusion in the decoder, yielding sharper albedo boundaries and stronger lighting invariance than prior U-Net variants. The initial albedo prediction is upsampled to $1024\times1024$ and refined by a lightweight three-layer CNN (RefinementNet). Conditioned on this refined albedo, a Pix2PixHD-based translator then predicts a comprehensive set of five additional physically based rendering passes: ambient occlusion, surface normal, specular reflectance, translucency, and raw diffuse colour (with residual lighting). Together with the refined albedo, these six passes form the complete intrinsic decomposition. Trained with a combination of masked-MSE, VGG, edge, and patch-LPIPS losses on the FFHQ-UV-Intrinsics dataset, the full pipeline achieves state-of-the-art performance for diffuse albedo estimation and demonstrates significantly improved fidelity for the complete rendering stack compared to prior methods. The resulting passes enable high-quality relighting and material editing of real faces.
>
---
#### [new 027] C-DGPA: Class-Centric Dual-Alignment Generative Prompt Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向无监督域自适应（UDA）任务，旨在解决视觉语言模型（VLMs）在提示调优中忽略条件分布对齐导致的类原型错位与语义判别力下降问题。提出C-DGPA方法，通过双分支架构协同优化边缘与条件分布对齐，引入类映射机制增强语义一致性与领域不变性。**

- **链接: [https://arxiv.org/pdf/2512.16164v1](https://arxiv.org/pdf/2512.16164v1)**

> **作者:** Chao Li; Dasha Hu; Chengyang Li; Yuming Jiang; Yuncheng Shen
>
> **摘要:** Unsupervised Domain Adaptation transfers knowledge from a labeled source domain to an unlabeled target domain. Directly deploying Vision-Language Models (VLMs) with prompt tuning in downstream UDA tasks faces the signifi cant challenge of mitigating domain discrepancies. Existing prompt-tuning strategies primarily align marginal distribu tion, but neglect conditional distribution discrepancies, lead ing to critical issues such as class prototype misalignment and degraded semantic discriminability. To address these lim itations, the work proposes C-DGPA: Class-Centric Dual Alignment Generative Prompt Adaptation. C-DGPA syner gistically optimizes marginal distribution alignment and con ditional distribution alignment through a novel dual-branch architecture. The marginal distribution alignment branch em ploys a dynamic adversarial training framework to bridge marginal distribution discrepancies. Simultaneously, the con ditional distribution alignment branch introduces a Class Mapping Mechanism (CMM) to align conditional distribu tion discrepancies by standardizing semantic prompt under standing and preventing source domain over-reliance. This dual alignment strategy effectively integrates domain knowl edge into prompt learning via synergistic optimization, ensur ing domain-invariant and semantically discriminative repre sentations. Extensive experiments on OfficeHome, Office31, and VisDA-2017 validate the superiority of C-DGPA. It achieves new state-of-the-art results on all benchmarks.
>
---
#### [new 028] Yuan-TecSwin: A text conditioned Diffusion model with Swin-transformer blocks
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Yuan-TecSwin，一种文本条件扩散模型，旨在解决CNN局部性限制导致的长程语义建模不足问题。工作包括：用Swin Transformer替换U-Net中CNN模块、优化文本编码与条件融合、改进时间步调度。任务为文本到图像生成，在ImageNet上达FID 1.37，性能领先。**

- **链接: [https://arxiv.org/pdf/2512.16586v1](https://arxiv.org/pdf/2512.16586v1)**

> **作者:** Shaohua Wu; Tong Yu; Shenling Wang; Xudong Zhao
>
> **摘要:** Diffusion models have shown remarkable capacity in image synthesis based on their U-shaped architecture and convolutional neural networks (CNN) as basic blocks. The locality of the convolution operation in CNN may limit the model's ability to understand long-range semantic information. To address this issue, we propose Yuan-TecSwin, a text-conditioned diffusion model with Swin-transformer in this work. The Swin-transformer blocks take the place of CNN blocks in the encoder and decoder, to improve the non-local modeling ability in feature extraction and image restoration. The text-image alignment is improved with a well-chosen text encoder, effective utilization of text embedding, and careful design in the incorporation of text condition. Using an adapted time step to search in different diffusion stages, inference performance is further improved by 10%. Yuan-TecSwin achieves the state-of-the-art FID score of 1.37 on ImageNet generation benchmark, without any additional models at different denoising stages. In a side-by-side comparison, we find it difficult for human interviewees to tell the model-generated images from the human-painted ones.
>
---
#### [new 029] Pixel Super-Resolved Fluorescence Lifetime Imaging Using Deep Learning
- **分类: cs.CV; cs.LG; physics.med-ph; physics.optics**

- **简介: 该论文提出FLIM_PSR_k，一种基于cGAN的深度学习方法，解决FLIM因长采集时间与低SNR导致的分辨率-速度矛盾。通过多通道像素超分辨，从5倍增大像素尺寸的数据中重建高分辨FLIM图像，提升空间带宽积25倍，增强细部结构可视性，推动FLIM临床转化。**

- **链接: [https://arxiv.org/pdf/2512.16266v1](https://arxiv.org/pdf/2512.16266v1)**

> **作者:** Paloma Casteleiro Costa; Parnian Ghapandar Kashani; Xuhui Liu; Alexander Chen; Ary Portes; Julien Bec; Laura Marcu; Aydogan Ozcan
>
> **备注:** 30 Pages, 9 Figures
>
> **摘要:** Fluorescence lifetime imaging microscopy (FLIM) is a powerful quantitative technique that provides metabolic and molecular contrast, offering strong translational potential for label-free, real-time diagnostics. However, its clinical adoption remains limited by long pixel dwell times and low signal-to-noise ratio (SNR), which impose a stricter resolution-speed trade-off than conventional optical imaging approaches. Here, we introduce FLIM_PSR_k, a deep learning-based multi-channel pixel super-resolution (PSR) framework that reconstructs high-resolution FLIM images from data acquired with up to a 5-fold increased pixel size. The model is trained using the conditional generative adversarial network (cGAN) framework, which, compared to diffusion model-based alternatives, delivers a more robust PSR reconstruction with substantially shorter inference times, a crucial advantage for practical deployment. FLIM_PSR_k not only enables faster image acquisition but can also alleviate SNR limitations in autofluorescence-based FLIM. Blind testing on held-out patient-derived tumor tissue samples demonstrates that FLIM_PSR_k reliably achieves a super-resolution factor of k = 5, resulting in a 25-fold increase in the space-bandwidth product of the output images and revealing fine architectural features lost in lower-resolution inputs, with statistically significant improvements across various image quality metrics. By increasing FLIM's effective spatial resolution, FLIM_PSR_k advances lifetime imaging toward faster, higher-resolution, and hardware-flexible implementations compatible with low-numerical-aperture and miniaturized platforms, better positioning FLIM for translational applications.
>
---
#### [new 030] MACL: Multi-Label Adaptive Contrastive Learning Loss for Remote Sensing Image Retrieval
- **分类: cs.CV**

- **简介: 该论文面向遥感图像多标签检索任务，解决语义重叠、标签分布不均衡及类间共现复杂等挑战。提出MACL方法，融合标签感知采样、频次敏感加权与动态温度缩放，实现均衡表征学习。在多个基准数据集上显著优于对比学习基线。**

- **链接: [https://arxiv.org/pdf/2512.16294v1](https://arxiv.org/pdf/2512.16294v1)**

> **作者:** Amna Amir; Erchan Aptoula
>
> **摘要:** Semantic overlap among land-cover categories, highly imbalanced label distributions, and complex inter-class co-occurrence patterns constitute significant challenges for multi-label remote-sensing image retrieval. In this article, Multi-Label Adaptive Contrastive Learning (MACL) is introduced as an extension of contrastive learning to address them. It integrates label-aware sampling, frequency-sensitive weighting, and dynamic-temperature scaling to achieve balanced representation learning across both common and rare categories. Extensive experiments on three benchmark datasets (DLRSD, ML-AID, and WHDLD), show that MACL consistently outperforms contrastive-loss based baselines, effectively mitigating semantic imbalance and delivering more reliable retrieval performance in large-scale remote-sensing archives. Code, pretrained models, and evaluation scripts will be released at https://github.com/amna/MACL upon acceptance.
>
---
#### [new 031] Collimator-assisted high-precision calibration method for event cameras
- **分类: cs.CV**

- **简介: 该论文属于事件相机几何标定任务，旨在解决长距离场景下标定精度低、可靠性差的问题。提出一种基于准直仪与闪烁星图的标定方法：先用球面运动模型线性求解参数，再通过非线性优化提升精度。实验表明其精度与鲁棒性优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.16092v1](https://arxiv.org/pdf/2512.16092v1)**

> **作者:** Zibin Liu; Shunkun Liang; Banglei Guan; Dongcai Tan; Yang Shang; Qifeng Yu
>
> **备注:** 4 pages, 3 figures
>
> **摘要:** Event cameras are a new type of brain-inspired visual sensor with advantages such as high dynamic range and high temporal resolution. The geometric calibration of event cameras, which involves determining their intrinsic and extrinsic parameters, particularly in long-range measurement scenarios, remains a significant challenge. To address the dual requirements of long-distance and high-precision measurement, we propose an event camera calibration method utilizing a collimator with flickering star-based patterns. The proposed method first linearly solves camera parameters using the sphere motion model of the collimator, followed by nonlinear optimization to refine these parameters with high precision. Through comprehensive real-world experiments across varying conditions, we demonstrate that the proposed method consistently outperforms existing event camera calibration methods in terms of accuracy and reliability.
>
---
#### [new 032] SegGraph: Leveraging Graphs of SAM Segments for Few-Shot 3D Part Segmentation
- **分类: cs.CV**

- **简介: 该论文面向少样本3D部件分割任务，旨在解决2D基础模型知识向3D几何结构有效迁移的难题。提出SegGraph框架：构建SAM分割段图，建模段间重叠/邻接关系，通过GNN传播几何特征，并用视角加权融合提升段内语义一致性。**

- **链接: [https://arxiv.org/pdf/2512.16143v1](https://arxiv.org/pdf/2512.16143v1)**

> **作者:** Yueyang Hu; Haiyong Jiang; Haoxuan Song; Jun Xiao; Hao Pan
>
> **摘要:** This work presents a novel framework for few-shot 3D part segmentation. Recent advances have demonstrated the significant potential of 2D foundation models for low-shot 3D part segmentation. However, it is still an open problem that how to effectively aggregate 2D knowledge from foundation models to 3D. Existing methods either ignore geometric structures for 3D feature learning or neglects the high-quality grouping clues from SAM, leading to under-segmentation and inconsistent part labels. We devise a novel SAM segment graph-based propagation method, named SegGraph, to explicitly learn geometric features encoded within SAM's segmentation masks. Our method encodes geometric features by modeling mutual overlap and adjacency between segments while preserving intra-segment semantic consistency. We construct a segment graph, conceptually similar to an atlas, where nodes represent segments and edges capture their spatial relationships (overlap/adjacency). Each node adaptively modulates 2D foundation model features, which are then propagated via a graph neural network to learn global geometric structures. To enforce intra-segment semantic consistency, we map segment features to 3D points with a novel view-direction-weighted fusion attenuating contributions from low-quality segments. Extensive experiments on PartNet-E demonstrate that our method outperforms all competing baselines by at least 6.9 percent mIoU. Further analysis reveals that SegGraph achieves particularly strong performance on small components and part boundaries, demonstrating its superior geometric understanding. The code is available at: https://github.com/YueyangHu2000/SegGraph.
>
---
#### [new 033] Smile on the Face, Sadness in the Eyes: Bridging the Emotion Gap with a Multimodal Dataset of Eye and Facial Behaviors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属多模态情感识别任务，旨在解决面部表情易伪装、难以反映真实情绪的问题。作者构建首个眼行为辅助的EMER数据集，提出EMERT模型，融合眼动与面部特征，显著提升真实情感识别性能。**

- **链接: [https://arxiv.org/pdf/2512.16485v1](https://arxiv.org/pdf/2512.16485v1)**

> **作者:** Kejun Liu; Yuanyuan Liu; Lin Wei; Chang Tang; Yibing Zhan; Zijing Chen; Zhe Chen
>
> **备注:** Accepted by TMM
>
> **摘要:** Emotion Recognition (ER) is the process of analyzing and identifying human emotions from sensing data. Currently, the field heavily relies on facial expression recognition (FER) because visual channel conveys rich emotional cues. However, facial expressions are often used as social tools rather than manifestations of genuine inner emotions. To understand and bridge this gap between FER and ER, we introduce eye behaviors as an important emotional cue and construct an Eye-behavior-aided Multimodal Emotion Recognition (EMER) dataset. To collect data with genuine emotions, spontaneous emotion induction paradigm is exploited with stimulus material, during which non-invasive eye behavior data, like eye movement sequences and eye fixation maps, is captured together with facial expression videos. To better illustrate the gap between ER and FER, multi-view emotion labels for mutimodal ER and FER are separately annotated. Furthermore, based on the new dataset, we design a simple yet effective Eye-behavior-aided MER Transformer (EMERT) that enhances ER by bridging the emotion gap. EMERT leverages modality-adversarial feature decoupling and a multitask Transformer to model eye behaviors as a strong complement to facial expressions. In the experiment, we introduce seven multimodal benchmark protocols for a variety of comprehensive evaluations of the EMER dataset. The results show that the EMERT outperforms other state-of-the-art multimodal methods by a great margin, revealing the importance of modeling eye behaviors for robust ER. To sum up, we provide a comprehensive analysis of the importance of eye behaviors in ER, advancing the study on addressing the gap between FER and ER for more robust ER performance. Our EMER dataset and the trained EMERT models will be publicly available at https://github.com/kejun1/EMER.
>
---
#### [new 034] YOLO11-4K: An Efficient Architecture for Real-Time Small Object Detection in 4K Panoramic Images
- **分类: cs.CV**

- **简介: 该论文属目标检测任务，旨在解决4K全景图像中因畸变、高分辨率导致的小目标检测难、实时性差问题。提出YOLO11-4K架构：引入P2多尺度检测头提升小目标敏感性，采用GhostConv骨干网降计算量，并构建标注的CVIP360基准数据集。**

- **链接: [https://arxiv.org/pdf/2512.16493v1](https://arxiv.org/pdf/2512.16493v1)**

> **作者:** Huma Hafeez; Matthew Garratt; Jo Plested; Sankaran Iyer; Arcot Sowmya
>
> **备注:** Conference paper just submitted
>
> **摘要:** The processing of omnidirectional 360-degree images poses significant challenges for object detection due to inherent spatial distortions, wide fields of view, and ultra-high-resolution inputs. Conventional detectors such as YOLO are optimised for standard image sizes (for example, 640x640 pixels) and often struggle with the computational demands of 4K or higher-resolution imagery typical of 360-degree vision. To address these limitations, we introduce YOLO11-4K, an efficient real-time detection framework tailored for 4K panoramic images. The architecture incorporates a novel multi-scale detection head with a P2 layer to improve sensitivity to small objects often missed at coarser scales, and a GhostConv-based backbone to reduce computational complexity without sacrificing representational power. To enable evaluation, we manually annotated the CVIP360 dataset, generating 6,876 frame-level bounding boxes and producing a publicly available, detection-ready benchmark for 4K panoramic scenes. YOLO11-4K achieves 0.95 mAP at 0.50 IoU with 28.3 milliseconds inference per frame, representing a 75 percent latency reduction compared to YOLO11 (112.3 milliseconds), while also improving accuracy (mAP at 0.50 of 0.95 versus 0.908). This balance of efficiency and precision enables robust object detection in expansive 360-degree environments, making the framework suitable for real-world high-resolution panoramic applications. While this work focuses on 4K omnidirectional images, the approach is broadly applicable to high-resolution detection tasks in autonomous navigation, surveillance, and augmented reality.
>
---
#### [new 035] Visual Alignment of Medical Vision-Language Models for Grounded Radiology Report Generation
- **分类: cs.CV**

- **简介: 该论文属放射科报告生成任务，旨在解决现有医学视觉语言模型因跨模态对齐差导致的幻觉问题。提出VALOR方法：分两阶段强化学习对齐——先用文本奖励提升术语准确性，再对齐视觉投影模块与疾病区域，增强视觉接地性。**

- **链接: [https://arxiv.org/pdf/2512.16201v1](https://arxiv.org/pdf/2512.16201v1)**

> **作者:** Sarosij Bose; Ravi K. Rajendran; Biplob Debnath; Konstantinos Karydis; Amit K. Roy-Chowdhury; Srimat Chakradhar
>
> **摘要:** Radiology Report Generation (RRG) is a critical step toward automating healthcare workflows, facilitating accurate patient assessments, and reducing the workload of medical professionals. Despite recent progress in Large Medical Vision-Language Models (Med-VLMs), generating radiology reports that are both visually grounded and clinically accurate remains a significant challenge. Existing approaches often rely on large labeled corpora for pre-training, costly task-specific preference data, or retrieval-based methods. However, these strategies do not adequately mitigate hallucinations arising from poor cross-modal alignment between visual and linguistic representations. To address these limitations, we propose VALOR:Visual Alignment of Medical Vision-Language Models for GrOunded Radiology Report Generation. Our method introduces a reinforcement learning-based post-alignment framework utilizing Group-Relative Proximal Optimization (GRPO). The training proceeds in two stages: (1) improving the Med-VLM with textual rewards to encourage clinically precise terminology, and (2) aligning the vision projection module of the textually grounded model with disease findings, thereby guiding attention toward image re gions most relevant to the diagnostic task. Extensive experiments on multiple benchmarks demonstrate that VALOR substantially improves factual accuracy and visual grounding, achieving significant performance gains over state-of-the-art report generation methods.
>
---
#### [new 036] SDFoam: Signed-Distance Foam for explicit surface reconstruction
- **分类: cs.CV; cs.GR**

- **简介: 该论文属三维场景重建任务，旨在解决NeRF及3DGS等方法难以精确提取网格表面的问题。提出SDFoam：联合学习显式Voronoi图与隐式SDF，通过射线追踪优化并以Eikonal约束正则化，实现高精度、拓扑合理、视图一致的显式表面重建。**

- **链接: [https://arxiv.org/pdf/2512.16706v1](https://arxiv.org/pdf/2512.16706v1)**

> **作者:** Antonella Rech; Nicola Conci; Nicola Garau
>
> **摘要:** Neural radiance fields (NeRF) have driven impressive progress in view synthesis by using ray-traced volumetric rendering. Splatting-based methods such as 3D Gaussian Splatting (3DGS) provide faster rendering by rasterizing 3D primitives. RadiantFoam (RF) brought ray tracing back, achieving throughput comparable to Gaussian Splatting by organizing radiance with an explicit Voronoi Diagram (VD). Yet, all the mentioned methods still struggle with precise mesh reconstruction. We address this gap by jointly learning an explicit VD with an implicit Signed Distance Field (SDF). The scene is optimized via ray tracing and regularized by an Eikonal objective. The SDF introduces metric-consistent isosurfaces, which, in turn, bias near-surface Voronoi cell faces to align with the zero level set. The resulting model produces crisper, view-consistent surfaces with fewer floaters and improved topology, while preserving photometric quality and maintaining training speed on par with RadiantFoam. Across diverse scenes, our hybrid implicit-explicit formulation, which we name SDFoam, substantially improves mesh reconstruction accuracy (Chamfer distance) with comparable appearance (PSNR, SSIM), without sacrificing efficiency.
>
---
#### [new 037] FOD-Diff: 3D Multi-Channel Patch Diffusion Model for Fiber Orientation Distribution
- **分类: cs.CV; cs.LG**

- **简介: 该论文属医学图像生成任务，旨在解决单壳低角分辨率dMRI（LAR-FOD）难以准确估计高角分辨率FOD（HAR-FOD）的问题。提出FOD-Diff模型：采用3D多通道补丁扩散架构，引入FOD-patch适配器、体素级条件协调模块和球谐注意力模块，实现高效、精准的HAR-FOD预测。**

- **链接: [https://arxiv.org/pdf/2512.16075v1](https://arxiv.org/pdf/2512.16075v1)**

> **作者:** Hao Tang; Hanyu Liu; Alessandro Perelli; Xi Chen; Chao Li
>
> **摘要:** Diffusion MRI (dMRI) is a critical non-invasive technique to estimate fiber orientation distribution (FOD) for characterizing white matter integrity. Estimating FOD from single-shell low angular resolution dMRI (LAR-FOD) is limited by accuracy, whereas estimating FOD from multi-shell high angular resolution dMRI (HAR-FOD) requires a long scanning time, which limits its applicability. Diffusion models have shown promise in estimating HAR-FOD based on LAR-FOD. However, using diffusion models to efficiently generate HAR-FOD is challenging due to the large number of spherical harmonic (SH) coefficients in FOD. Here, we propose a 3D multi-channel patch diffusion model to predict HAR-FOD from LAR-FOD. We design the FOD-patch adapter by introducing the prior brain anatomy for more efficient patch-based learning. Furthermore, we introduce a voxel-level conditional coordinating module to enhance the global understanding of the model. We design the SH attention module to effectively learn the complex correlations of the SH coefficients. Our experimental results show that our method achieves the best performance in HAR-FOD prediction and outperforms other state-of-the-art methods.
>
---
#### [new 038] From Words to Wavelengths: VLMs for Few-Shot Multispectral Object Detection
- **分类: cs.CV**

- **简介: 该论文研究少样本多光谱目标检测任务，旨在解决多光谱标注数据稀缺导致模型性能受限的问题。作者适配Grounding DINO和YOLO-World等视觉语言模型，提出文本-视觉-热成像多模态融合机制，在FLIR和M3FD上验证其在少样本及全监督场景下的优越性。**

- **链接: [https://arxiv.org/pdf/2512.15971v1](https://arxiv.org/pdf/2512.15971v1)**

> **作者:** Manuel Nkegoum; Minh-Tan Pham; Élisa Fromont; Bruno Avignon; Sébastien Lefèvre
>
> **摘要:** Multispectral object detection is critical for safety-sensitive applications such as autonomous driving and surveillance, where robust perception under diverse illumination conditions is essential. However, the limited availability of annotated multispectral data severely restricts the training of deep detectors. In such data-scarce scenarios, textual class information can serve as a valuable source of semantic supervision. Motivated by the recent success of Vision-Language Models (VLMs) in computer vision, we explore their potential for few-shot multispectral object detection. Specifically, we adapt two representative VLM-based detectors, Grounding DINO and YOLO-World, to handle multispectral inputs and propose an effective mechanism to integrate text, visual and thermal modalities. Through extensive experiments on two popular multispectral image benchmarks, FLIR and M3FD, we demonstrate that VLM-based detectors not only excel in few-shot regimes, significantly outperforming specialized multispectral models trained with comparable data, but also achieve competitive or superior results under fully supervised settings. Our findings reveal that the semantic priors learned by large-scale VLMs effectively transfer to unseen spectral modalities, ofFering a powerful pathway toward data-efficient multispectral perception.
>
---
#### [new 039] OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition
- **分类: cs.CV; cs.HC**

- **简介: 该论文面向骨架驱动的在线微手势识别任务，解决微手势数据稀缺、标注难及实时检测分类难的问题；构建首个大规模公开基准OMG-Bench，并提出分层记忆增强Transformer（HMATr）模型，统一检测与分类，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.16727v1](https://arxiv.org/pdf/2512.16727v1)**

> **作者:** Haochen Chang; Pengfei Ren; Buyuan Zhang; Da Li; Tianhao Han; Haoyang Zhang; Liang Xie; Hongbo Chen; Erwei Yin
>
> **备注:** Project page: https://omg-bench.github.io/
>
> **摘要:** Online micro gesture recognition from hand skeletons is critical for VR/AR interaction but faces challenges due to limited public datasets and task-specific algorithms. Micro gestures involve subtle motion patterns, which make constructing datasets with precise skeletons and frame-level annotations difficult. To this end, we develop a multi-view self-supervised pipeline to automatically generate skeleton data, complemented by heuristic rules and expert refinement for semi-automatic annotation. Based on this pipeline, we introduce OMG-Bench, the first large-scale public benchmark for skeleton-based online micro gesture recognition. It features 40 fine-grained gesture classes with 13,948 instances across 1,272 sequences, characterized by subtle motions, rapid dynamics, and continuous execution. To tackle these challenges, we propose Hierarchical Memory-Augmented Transformer (HMATr), an end-to-end framework that unifies gesture detection and classification by leveraging hierarchical memory banks which store frame-level details and window-level semantics to preserve historical context. In addition, it employs learnable position-aware queries initialized from the memory to implicitly encode gesture positions and semantics. Experiments show that HMATr outperforms state-of-the-art methods by 7.6\% in detection rate, establishing a strong baseline for online micro gesture recognition. Project page: https://omg-bench.github.io/
>
---
#### [new 040] EverybodyDance: Bipartite Graph-Based Identity Correspondence for Multi-Character Animation
- **分类: cs.CV**

- **简介: 该论文属于多角色动画生成任务，旨在解决多角色间身份对应（IC）错误问题，尤其在位置交换时。提出EverybodyDance框架，构建身份匹配图（IMG），用Mask-Query Attention计算角色亲和度，并通过图优化、身份引导等策略提升IC正确性，配套新评估基准验证效果。**

- **链接: [https://arxiv.org/pdf/2512.16360v1](https://arxiv.org/pdf/2512.16360v1)**

> **作者:** Haotian Ling; Zequn Chen; Qiuying Chen; Donglin Di; Yongjia Ma; Hao Li; Chen Wei; Zhulin Tao; Xun Yang
>
> **摘要:** Consistent pose-driven character animation has achieved remarkable progress in single-character scenarios. However, extending these advances to multi-character settings is non-trivial, especially when position swap is involved. Beyond mere scaling, the core challenge lies in enforcing correct Identity Correspondence (IC) between characters in reference and generated frames. To address this, we introduce EverybodyDance, a systematic solution targeting IC correctness in multi-character animation. EverybodyDance is built around the Identity Matching Graph (IMG), which models characters in the generated and reference frames as two node sets in a weighted complete bipartite graph. Edge weights, computed via our proposed Mask-Query Attention (MQA), quantify the affinity between each pair of characters. Our key insight is to formalize IC correctness as a graph structural metric and to optimize it during training. We also propose a series of targeted strategies tailored for multi-character animation, including identity-embedded guidance, a multi-scale matching strategy, and pre-classified sampling, which work synergistically. Finally, to evaluate IC performance, we curate the Identity Correspondence Evaluation benchmark, dedicated to multi-character IC correctness. Extensive experiments demonstrate that EverybodyDance substantially outperforms state-of-the-art baselines in both IC and visual fidelity.
>
---
#### [new 041] Factorized Video Generation: Decoupling Scene Construction and Temporal Synthesis in Text-to-Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属文本到视频（T2V）生成任务，旨在解决现有扩散模型因初始帧语义/逻辑错误导致的场景构建失败与时间逻辑不符问题。提出Factorized Video Generation（FVG）框架，将T2V解耦为LLM提示重写、T2I锚帧生成、视频时序合成三阶段，提升质量、效率与可控性。**

- **链接: [https://arxiv.org/pdf/2512.16371v1](https://arxiv.org/pdf/2512.16371v1)**

> **作者:** Mariam Hassan; Bastien Van Delft; Wuyang Li; Alexandre Alahi
>
> **摘要:** State-of-the-art Text-to-Video (T2V) diffusion models can generate visually impressive results, yet they still frequently fail to compose complex scenes or follow logical temporal instructions. In this paper, we argue that many errors, including apparent motion failures, originate from the model's inability to construct a semantically correct or logically consistent initial frame. We introduce Factorized Video Generation (FVG), a pipeline that decouples these tasks by decomposing the Text-to-Video generation into three specialized stages: (1) Reasoning, where a Large Language Model (LLM) rewrites the video prompt to describe only the initial scene, resolving temporal ambiguities; (2) Composition, where a Text-to-Image (T2I) model synthesizes a high-quality, compositionally-correct anchor frame from this new prompt; and (3) Temporal Synthesis, where a video model, finetuned to understand this anchor, focuses its entire capacity on animating the scene and following the prompt. Our decomposed approach sets a new state-of-the-art on the T2V CompBench benchmark and significantly improves all tested models on VBench2. Furthermore, we show that visual anchoring allows us to cut the number of sampling steps by 70% without any loss in performance, leading to a substantial speed-up in sampling. Factorized Video Generation offers a simple yet practical path toward more efficient, robust, and controllable video synthesis
>
---
#### [new 042] StereoPilot: Learning Unified and Efficient Stereo Conversion via Generative Priors
- **分类: cs.CV**

- **简介: 该论文面向单目转立体视频任务，解决传统DWI流程误差传播、深度模糊及格式不一致问题。提出UniStereo统一数据集，并设计高效前馈模型StereoPilot，无需显式深度图或迭代采样，支持多格式自适应转换，提升质量与效率。**

- **链接: [https://arxiv.org/pdf/2512.16915v1](https://arxiv.org/pdf/2512.16915v1)**

> **作者:** Guibao Shen; Yihua Du; Wenhang Ge; Jing He; Chirui Chang; Donghao Zhou; Zhen Yang; Luozhou Wang; Xin Tao; Ying-Cong Chen
>
> **摘要:** The rapid growth of stereoscopic displays, including VR headsets and 3D cinemas, has led to increasing demand for high-quality stereo video content. However, producing 3D videos remains costly and complex, while automatic Monocular-to-Stereo conversion is hindered by the limitations of the multi-stage ``Depth-Warp-Inpaint'' (DWI) pipeline. This paradigm suffers from error propagation, depth ambiguity, and format inconsistency between parallel and converged stereo configurations. To address these challenges, we introduce UniStereo, the first large-scale unified dataset for stereo video conversion, covering both stereo formats to enable fair benchmarking and robust model training. Building upon this dataset, we propose StereoPilot, an efficient feed-forward model that directly synthesizes the target view without relying on explicit depth maps or iterative diffusion sampling. Equipped with a learnable domain switcher and a cycle consistency loss, StereoPilot adapts seamlessly to different stereo formats and achieves improved consistency. Extensive experiments demonstrate that StereoPilot significantly outperforms state-of-the-art methods in both visual fidelity and computational efficiency. Project page: https://hit-perfect.github.io/StereoPilot/.
>
---
#### [new 043] Learning High-Quality Initial Noise for Single-View Synthesis with Diffusion Models
- **分类: cs.CV; eess.IV**

- **简介: 该论文面向单视图新视角合成（NVS）任务，旨在解决扩散模型中初始噪声质量影响生成效果的问题。提出离散化欧拉逆方法构建高质量噪声数据集，并设计编码器-解码器网络（EDN）学习从随机噪声到高质量噪声的映射，可即插即用地提升SV3D等NVS模型性能。**

- **链接: [https://arxiv.org/pdf/2512.16219v1](https://arxiv.org/pdf/2512.16219v1)**

> **作者:** Zhihao Zhang; Xuejun Yang; Weihua Liu; Mouquan Shen
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Single-view novel view synthesis (NVS) models based on diffusion models have recently attracted increasing attention, as they can generate a series of novel view images from a single image prompt and camera pose information as conditions. It has been observed that in diffusion models, certain high-quality initial noise patterns lead to better generation results than others. However, there remains a lack of dedicated learning frameworks that enable NVS models to learn such high-quality noise. To obtain high-quality initial noise from random Gaussian noise, we make the following contributions. First, we design a discretized Euler inversion method to inject image semantic information into random noise, thereby constructing paired datasets of random and high-quality noise. Second, we propose a learning framework based on an encoder-decoder network (EDN) that directly transforms random noise into high-quality noise. Experiments demonstrate that the proposed EDN can be seamlessly plugged into various NVS models, such as SV3D and MV-Adapter, achieving significant performance improvements across multiple datasets. Code is available at: https://github.com/zhihao0512/EDN.
>
---
#### [new 044] Kling-Omni Technical Report
- **分类: cs.CV**

- **简介: 该论文提出Kling-Omni，一种端到端多模态视频生成框架，旨在统一解决视频生成、编辑与推理任务。它融合文本、图像、视频输入，构建统一表征，支持高保真、智能化视频创作，并通过大规模预训练与数据系统实现突破。**

- **链接: [https://arxiv.org/pdf/2512.16776v1](https://arxiv.org/pdf/2512.16776v1)**

> **作者:** Kling Team; Jialu Chen; Yuanzheng Ci; Xiangyu Du; Zipeng Feng; Kun Gai; Sainan Guo; Feng Han; Jingbin He; Kang He; Xiao Hu; Xiaohua Hu; Boyuan Jiang; Fangyuan Kong; Hang Li; Jie Li; Qingyu Li; Shen Li; Xiaohan Li; Yan Li; Jiajun Liang; Borui Liao; Yiqiao Liao; Weihong Lin; Quande Liu; Xiaokun Liu; Yilun Liu; Yuliang Liu; Shun Lu; Hangyu Mao; Yunyao Mao; Haodong Ouyang; Wenyu Qin; Wanqi Shi; Xiaoyu Shi; Lianghao Su; Haozhi Sun; Peiqin Sun; Pengfei Wan; Chao Wang; Chenyu Wang; Meng Wang; Qiulin Wang; Runqi Wang; Xintao Wang; Xuebo Wang; Zekun Wang; Min Wei; Tiancheng Wen; Guohao Wu; Xiaoshi Wu; Zhenhua Wu; Da Xie; Yingtong Xiong; Yulong Xu; Sile Yang; Zikang Yang; Weicai Ye; Ziyang Yuan; Shenglong Zhang; Shuaiyu Zhang; Yuanxing Zhang; Yufan Zhang; Wenzheng Zhao; Ruiliang Zhou; Yan Zhou; Guosheng Zhu; Yongjie Zhu
>
> **备注:** Kling-Omni Technical Report
>
> **摘要:** We present Kling-Omni, a generalist generative framework designed to synthesize high-fidelity videos directly from multimodal visual language inputs. Adopting an end-to-end perspective, Kling-Omni bridges the functional separation among diverse video generation, editing, and intelligent reasoning tasks, integrating them into a holistic system. Unlike disjointed pipeline approaches, Kling-Omni supports a diverse range of user inputs, including text instructions, reference images, and video contexts, processing them into a unified multimodal representation to deliver cinematic-quality and highly-intelligent video content creation. To support these capabilities, we constructed a comprehensive data system that serves as the foundation for multimodal video creation. The framework is further empowered by efficient large-scale pre-training strategies and infrastructure optimizations for inference. Comprehensive evaluations reveal that Kling-Omni demonstrates exceptional capabilities in in-context generation, reasoning-based editing, and multimodal instruction following. Moving beyond a content creation tool, we believe Kling-Omni is a pivotal advancement toward multimodal world simulators capable of perceiving, reasoning, generating and interacting with the dynamic and complex worlds.
>
---
#### [new 045] ARMFlow: AutoRegressive MeanFlow for Online 3D Human Reaction Generation
- **分类: cs.CV**

- **简介: 该论文面向在线3D人体反应生成任务，解决高保真、实时推理与自回归适应性难以兼顾的问题。提出ARMFlow框架：基于MeanFlow的自回归模型，含因果上下文编码器和MLP速度预测器；引入Bootstrap Contextual Encoding缓解误差累积；并设计离线变体ReMFlow。**

- **链接: [https://arxiv.org/pdf/2512.16234v1](https://arxiv.org/pdf/2512.16234v1)**

> **作者:** Zichen Geng; Zeeshan Hayder; Wei Liu; Hesheng Wang; Ajmal Mian
>
> **摘要:** 3D human reaction generation faces three main challenges:(1) high motion fidelity, (2) real-time inference, and (3) autoregressive adaptability for online scenarios. Existing methods fail to meet all three simultaneously. We propose ARMFlow, a MeanFlow-based autoregressive framework that models temporal dependencies between actor and reactor motions. It consists of a causal context encoder and an MLP-based velocity predictor. We introduce Bootstrap Contextual Encoding (BSCE) in training, encoding generated history instead of the ground-truth ones, to alleviate error accumulation in autoregressive generation. We further introduce the offline variant ReMFlow, achieving state-of-the-art performance with the fastest inference among offline methods. Our ARMFlow addresses key limitations of online settings by: (1) enhancing semantic alignment via a global contextual encoder; (2) achieving high accuracy and low latency in a single-step inference; and (3) reducing accumulated errors through BSCE. Our single-step online generation surpasses existing online methods on InterHuman and InterX by over 40% in FID, while matching offline state-of-the-art performance despite using only partial sequence conditions.
>
---
#### [new 046] DVGT: Driving Visual Geometry Transformer
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DVGT模型，解决自动驾驶中多视角图像到全局稠密3D几何重建的任务。针对现有方法依赖精确相机参数、泛化性差的问题，DVGT基于DINO特征，融合局部、跨视图与跨帧注意力，直接从无位姿多图序列预测度量尺度的3D点云及自车姿态，无需几何先验或外部传感器对齐。**

- **链接: [https://arxiv.org/pdf/2512.16919v1](https://arxiv.org/pdf/2512.16919v1)**

> **作者:** Sicheng Zuo; Zixun Xie; Wenzhao Zheng; Shaoqing Xu; Fang Li; Shengyin Jiang; Long Chen; Zhi-Xin Yang; Jiwen Lu
>
> **备注:** Code is available at https://github.com/wzzheng/DVGT
>
> **摘要:** Perceiving and reconstructing 3D scene geometry from visual inputs is crucial for autonomous driving. However, there still lacks a driving-targeted dense geometry perception model that can adapt to different scenarios and camera configurations. To bridge this gap, we propose a Driving Visual Geometry Transformer (DVGT), which reconstructs a global dense 3D point map from a sequence of unposed multi-view visual inputs. We first extract visual features for each image using a DINO backbone, and employ alternating intra-view local attention, cross-view spatial attention, and cross-frame temporal attention to infer geometric relations across images. We then use multiple heads to decode a global point map in the ego coordinate of the first frame and the ego poses for each frame. Unlike conventional methods that rely on precise camera parameters, DVGT is free of explicit 3D geometric priors, enabling flexible processing of arbitrary camera configurations. DVGT directly predicts metric-scaled geometry from image sequences, eliminating the need for post-alignment with external sensors. Trained on a large mixture of driving datasets including nuScenes, OpenScene, Waymo, KITTI, and DDAD, DVGT significantly outperforms existing models on various scenarios. Code is available at https://github.com/wzzheng/DVGT.
>
---
#### [new 047] MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Model for Embodied Task Planning
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向具身智能任务规划，解决现有场景图缺乏状态感知、功能-空间割裂及任务导向性不足的问题。提出MomaGraph统一场景表示、配套数据集MomaGraph-Scenes与评测基准MomaGraph-Bench，并训练7B视觉语言模型MomaGraph-R1，实现零样本任务规划，性能达开源SOTA。**

- **链接: [https://arxiv.org/pdf/2512.16909v1](https://arxiv.org/pdf/2512.16909v1)**

> **作者:** Yuanchen Ju; Yongyuan Liang; Yen-Jen Wang; Nandiraju Gireesh; Yuanliang Ju; Seungjae Lee; Qiao Gu; Elvis Hsieh; Furong Huang; Koushil Sreenath
>
> **备注:** 25 pages, 10 figures. Project page:https://hybridrobotics.github.io/MomaGraph/
>
> **摘要:** Mobile manipulators in households must both navigate and manipulate. This requires a compact, semantically rich scene representation that captures where objects are, how they function, and which parts are actionable. Scene graphs are a natural choice, yet prior work often separates spatial and functional relations, treats scenes as static snapshots without object states or temporal updates, and overlooks information most relevant for accomplishing the current task. To address these limitations, we introduce MomaGraph, a unified scene representation for embodied agents that integrates spatial-functional relationships and part-level interactive elements. However, advancing such a representation requires both suitable data and rigorous evaluation, which have been largely missing. We thus contribute MomaGraph-Scenes, the first large-scale dataset of richly annotated, task-driven scene graphs in household environments, along with MomaGraph-Bench, a systematic evaluation suite spanning six reasoning capabilities from high-level planning to fine-grained scene understanding. Built upon this foundation, we further develop MomaGraph-R1, a 7B vision-language model trained with reinforcement learning on MomaGraph-Scenes. MomaGraph-R1 predicts task-oriented scene graphs and serves as a zero-shot task planner under a Graph-then-Plan framework. Extensive experiments demonstrate that our model achieves state-of-the-art results among open-source models, reaching 71.6% accuracy on the benchmark (+11.4% over the best baseline), while generalizing across public benchmarks and transferring effectively to real-robot experiments.
>
---
#### [new 048] PixelArena: A benchmark for Pixel-Precision Visual Intelligence
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PixelArena基准，聚焦像素级视觉智能评估。针对多模态大模型图像生成能力缺乏细粒度评测的问题，采用语义分割任务进行客观、精准的零样本生成能力测试，发现Gemini 3 Pro Image展现显著突破，并开展定量/定性分析与失败案例研究。**

- **链接: [https://arxiv.org/pdf/2512.16303v1](https://arxiv.org/pdf/2512.16303v1)**

> **作者:** Feng Liang; Sizhe Cheng; Chenqi Yi
>
> **备注:** 7 pages, 11 figures, project page: https://pixelarena.reify.ing/project
>
> **摘要:** Multi-modal large language models that have image output are emerging. Many image generation benchmarks focus on aesthetics instead of fine-grained generation capabilities. In PixelArena, we propose using semantic segmentation tasks to objectively examine their fine-grained generative intelligence with pixel precision. We find the latest Gemini 3 Pro Image has emergent image generation capabilities that generate semantic masks with high fidelity under zero-shot settings, showcasing visual intelligence unseen before and true generalization in new image generation tasks. We further investigate its results, compare them qualitatively and quantitatively with those of other models, and present failure cases. The findings not only signal exciting progress in the field but also provide insights into future research related to multimodality, reasoning, interpretability and benchmarking.
>
---
#### [new 049] M-PhyGs: Multi-Material Object Dynamics from Video
- **分类: cs.CV**

- **简介: 该论文提出M-PhyGs方法，解决从自然视频中估计多材料复杂物体（如花朵）物理参数的难题。任务是多材料物体力学建模，突破单材质、预设动力学等局限。工作包括：构建多材料高斯表征、设计级联3D/2D损失与时序小批量优化，并发布Phlowers数据集验证。**

- **链接: [https://arxiv.org/pdf/2512.16885v1](https://arxiv.org/pdf/2512.16885v1)**

> **作者:** Norika Wada; Kohei Yamashita; Ryo Kawahara; Ko Nishino
>
> **摘要:** Knowledge of the physical material properties governing the dynamics of a real-world object becomes necessary to accurately anticipate its response to unseen interactions. Existing methods for estimating such physical material parameters from visual data assume homogeneous single-material objects, pre-learned dynamics, or simplistic topologies. Real-world objects, however, are often complex in material composition and geometry lying outside the realm of these assumptions. In this paper, we particularly focus on flowers as a representative common object. We introduce Multi-material Physical Gaussians (M-PhyGs) to estimate the material composition and parameters of such multi-material complex natural objects from video. From a short video captured in a natural setting, M-PhyGs jointly segments the object into similar materials and recovers their continuum mechanical parameters while accounting for gravity. M-PhyGs achieves this efficiently with newly introduced cascaded 3D and 2D losses, and by leveraging temporal mini-batching. We introduce a dataset, Phlowers, of people interacting with flowers as a novel platform to evaluate the accuracy of this challenging task of multi-material physical parameter estimation. Experimental results on Phlowers dataset demonstrate the accuracy and effectiveness of M-PhyGs and its components.
>
---
#### [new 050] CoVAR: Co-generation of Video and Action for Robotic Manipulation via Multi-Modal Diffusion
- **分类: cs.CV**

- **简介: 该论文属具身智能中的视频-动作协同生成任务，旨在解决机器人操作中视频扩散模型缺乏动作标注、跨模态耦合弱的问题。提出CoVAR框架：扩展预训练视频扩散模型，引入并行动作扩散分支、Bridge Attention机制和动作精炼模块，实现高质量视频与精准动作的联合生成。**

- **链接: [https://arxiv.org/pdf/2512.16023v1](https://arxiv.org/pdf/2512.16023v1)**

> **作者:** Liudi Yang; Yang Bai; George Eskandar; Fengyi Shen; Mohammad Altillawi; Dong Chen; Ziyuan Liu; Abhinav Valada
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** We present a method to generate video-action pairs that follow text instructions, starting from an initial image observation and the robot's joint states. Our approach automatically provides action labels for video diffusion models, overcoming the common lack of action annotations and enabling their full use for robotic policy learning. Existing methods either adopt two-stage pipelines, which limit tightly coupled cross-modal information sharing, or rely on adapting a single-modal diffusion model for a joint distribution that cannot fully leverage pretrained video knowledge. To overcome these limitations, we (1) extend a pretrained video diffusion model with a parallel, dedicated action diffusion model that preserves pretrained knowledge, (2) introduce a Bridge Attention mechanism to enable effective cross-modal interaction, and (3) design an action refinement module to convert coarse actions into precise controls for low-resolution datasets. Extensive evaluations on multiple public benchmarks and real-world datasets demonstrate that our method generates higher-quality videos, more accurate actions, and significantly outperforms existing baselines, offering a scalable framework for leveraging large-scale video data for robotic learning.
>
---
#### [new 051] Pixel Seal: Adversarial-only training for invisible image and video watermarking
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属图像/视频隐写任务，旨在解决现有水印方法在 imperceptibility、robustness 和高分辨率扩展上的三大缺陷。提出 Pixel Seal：采用纯对抗训练、三阶段优化调度和基于 JND 的高分辨率适配，显著提升水印的不可见性与鲁棒性，并支持视频时序聚合。**

- **链接: [https://arxiv.org/pdf/2512.16874v1](https://arxiv.org/pdf/2512.16874v1)**

> **作者:** Tomáš Souček; Pierre Fernandez; Hady Elsahar; Sylvestre-Alvise Rebuffi; Valeriu Lacatusu; Tuan Tran; Tom Sander; Alexandre Mourachko
>
> **备注:** Code and model available at https://github.com/facebookresearch/videoseal
>
> **摘要:** Invisible watermarking is essential for tracing the provenance of digital content. However, training state-of-the-art models remains notoriously difficult, with current approaches often struggling to balance robustness against true imperceptibility. This work introduces Pixel Seal, which sets a new state-of-the-art for image and video watermarking. We first identify three fundamental issues of existing methods: (i) the reliance on proxy perceptual losses such as MSE and LPIPS that fail to mimic human perception and result in visible watermark artifacts; (ii) the optimization instability caused by conflicting objectives, which necessitates exhaustive hyperparameter tuning; and (iii) reduced robustness and imperceptibility of watermarks when scaling models to high-resolution images and videos. To overcome these issues, we first propose an adversarial-only training paradigm that eliminates unreliable pixel-wise imperceptibility losses. Second, we introduce a three-stage training schedule that stabilizes convergence by decoupling robustness and imperceptibility. Third, we address the resolution gap via high-resolution adaptation, employing JND-based attenuation and training-time inference simulation to eliminate upscaling artifacts. We thoroughly evaluate the robustness and imperceptibility of Pixel Seal on different image types and across a wide range of transformations, and show clear improvements over the state-of-the-art. We finally demonstrate that the model efficiently adapts to video via temporal watermark pooling, positioning Pixel Seal as a practical and scalable solution for reliable provenance in real-world image and video settings.
>
---
#### [new 052] QUIDS: Quality-informed Incentive-driven Multi-agent Dispatching System for Mobile Crowdsensing
- **分类: cs.CV**

- **简介: 该论文面向非专用车载移动众包感知（NVMCS）任务，解决动态车辆参与下感知覆盖率与可靠性难以兼顾的问题。提出QUIDS系统：定义聚合感知质量（ASQ）指标，设计信念感知的多车协同调度与激励分配算法，在预算约束下提升感知质量。**

- **链接: [https://arxiv.org/pdf/2512.16325v1](https://arxiv.org/pdf/2512.16325v1)**

> **作者:** Nan Zhou; Zuxin Li; Fanhang Man; Xuecheng Chen; Susu Xu; Fan Dang; Chaopeng Hong; Yunhao Liu; Xiao-Ping Zhang; Xinlei Chen
>
> **摘要:** This paper addresses the challenge of achieving optimal Quality of Information (QoI) in non-dedicated vehicular mobile crowdsensing (NVMCS) systems. The key obstacles are the interrelated issues of sensing coverage, sensing reliability, and the dynamic participation of vehicles. To tackle these, we propose QUIDS, a QUality-informed Incentive-driven multi-agent Dispatching System, which ensures high sensing coverage and reliability under budget constraints. QUIDS introduces a novel metric, Aggregated Sensing Quality (ASQ), to quantitatively capture QoI by integrating both coverage and reliability. We also develop a Mutually Assisted Belief-aware Vehicle Dispatching algorithm that estimates sensing reliability and allocates incentives under uncertainty, further improving ASQ. Evaluation using real-world data from a metropolitan NVMCS deployment shows QUIDS improves ASQ by 38% over non-dispatching scenarios and by 10% over state-of-the-art methods. It also reduces reconstruction map errors by 39-74% across algorithms. By jointly optimizing coverage and reliability via a quality-informed incentive mechanism, QUIDS enables low-cost, high-quality urban monitoring without dedicated infrastructure, applicable to smart-city scenarios like traffic and environmental sensing.
>
---
#### [new 053] Flexible Camera Calibration using a Collimator System
- **分类: cs.CV**

- **简介: 该论文属于相机标定任务，旨在解决传统标定需复杂运动或精密靶标的问题。提出基于准直器系统的柔性标定方法，利用角度不变性约束将相对运动简化为球面运动，设计了多图/双图/单图三种闭式求解器，实现无需相机运动的快速标定。**

- **链接: [https://arxiv.org/pdf/2512.16113v1](https://arxiv.org/pdf/2512.16113v1)**

> **作者:** Shunkun Liang; Banglei Guan; Zhenbao Yu; Dongcai Tan; Pengju Sun; Zibin Liu; Qifeng Yu; Yang Shang
>
> **摘要:** Camera calibration is a crucial step in photogrammetry and 3D vision applications. This paper introduces a novel camera calibration method using a designed collimator system. Our collimator system provides a reliable and controllable calibration environment for the camera. Exploiting the unique optical geometry property of our collimator system, we introduce an angle invariance constraint and further prove that the relative motion between the calibration target and camera conforms to a spherical motion model. This constraint reduces the original 6DOF relative motion between target and camera to a 3DOF pure rotation motion. Using spherical motion constraint, a closed-form linear solver for multiple images and a minimal solver for two images are proposed for camera calibration. Furthermore, we propose a single collimator image calibration algorithm based on the angle invariance constraint. This algorithm eliminates the requirement for camera motion, providing a novel solution for flexible and fast calibration. The performance of our method is evaluated in both synthetic and real-world experiments, which verify the feasibility of calibration using the collimator system and demonstrate that our method is superior to existing baseline methods. Demo code is available at https://github.com/LiangSK98/CollimatorCalibration
>
---
#### [new 054] Two-Step Data Augmentation for Masked Face Detection and Recognition: Turning Fake Masks to Real
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向戴口罩人脸检测与识别任务，解决真实遮挡数据稀缺与分布偏移问题。提出两步生成式数据增强框架：先规则化口罩形变，再通过GAN进行无配对图像翻译，辅以非掩码保留损失和随机噪声提升多样性与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.15774v1](https://arxiv.org/pdf/2512.15774v1)**

> **作者:** Yan Yang; George Bebis; Mircea Nicolescu
>
> **备注:** 9 pages, 9 figures. Conference version
>
> **摘要:** Data scarcity and distribution shift pose major challenges for masked face detection and recognition. We propose a two-step generative data augmentation framework that combines rule-based mask warping with unpaired image-to-image translation using GANs, enabling the generation of realistic masked-face samples beyond purely synthetic transformations. Compared to rule-based warping alone, the proposed approach yields consistent qualitative improvements and complements existing GAN-based masked face generation methods such as IAMGAN. We introduce a non-mask preservation loss and stochastic noise injection to stabilize training and enhance sample diversity. Experimental observations highlight the effectiveness of the proposed components and suggest directions for future improvements in data-centric augmentation for face recognition tasks.
>
---
#### [new 055] SNOW: Spatio-Temporal Scene Understanding with World Knowledge for Open-World Embodied Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SNOW框架，解决开放世界具身推理中语义与几何、时序脱节的问题。它融合VLM语义、点云几何和时间一致性，构建可查询的4D场景图（4DSG），实现无需训练、骨干无关的统一4D场景理解。**

- **链接: [https://arxiv.org/pdf/2512.16461v1](https://arxiv.org/pdf/2512.16461v1)**

> **作者:** Tin Stribor Sohn; Maximilian Dillitzer; Jason J. Corso; Eric Sax
>
> **摘要:** Autonomous robotic systems require spatio-temporal understanding of dynamic environments to ensure reliable navigation and interaction. While Vision-Language Models (VLMs) provide open-world semantic priors, they lack grounding in 3D geometry and temporal dynamics. Conversely, geometric perception captures structure and motion but remains semantically sparse. We propose SNOW (Scene Understanding with Open-World Knowledge), a training-free and backbone-agnostic framework for unified 4D scene understanding that integrates VLM-derived semantics with point cloud geometry and temporal consistency. SNOW processes synchronized RGB images and 3D point clouds, using HDBSCAN clustering to generate object-level proposals that guide SAM2-based segmentation. Each segmented region is encoded through our proposed Spatio-Temporal Tokenized Patch Encoding (STEP), producing multimodal tokens that capture localized semantic, geometric, and temporal attributes. These tokens are incrementally integrated into a 4D Scene Graph (4DSG), which serves as 4D prior for downstream reasoning. A lightweight SLAM backend anchors all STEP tokens spatially in the environment, providing the global reference alignment, and ensuring unambiguous spatial grounding across time. The resulting 4DSG forms a queryable, unified world model through which VLMs can directly interpret spatial scene structure and temporal dynamics. Experiments on a diverse set of benchmarks demonstrate that SNOW enables precise 4D scene understanding and spatially grounded inference, thereby setting new state-of-the-art performance in several settings, highlighting the importance of structured 4D priors for embodied reasoning and autonomous robotics.
>
---
#### [new 056] A multi-centre, multi-device benchmark dataset for landmark-based comprehensive fetal biometry
- **分类: cs.CV**

- **简介: 该论文构建了首个公开的多中心、多设备胎儿超声 landmark 标注基准数据集，用于解决人工测量耗时、主观、泛化差的问题。工作包括采集4513张图像、提供标准划分与评估代码，并验证跨中心性能下降，推动鲁棒AI辅助胎儿生物测量。**

- **链接: [https://arxiv.org/pdf/2512.16710v1](https://arxiv.org/pdf/2512.16710v1)**

> **作者:** Chiara Di Vece; Zhehua Mao; Netanell Avisdris; Brian Dromey; Raffaele Napolitano; Dafna Ben Bashat; Francisco Vasconcelos; Danail Stoyanov; Leo Joskowicz; Sophia Bano
>
> **备注:** 11 pages, 5 figures, 3 tables
>
> **摘要:** Accurate fetal growth assessment from ultrasound (US) relies on precise biometry measured by manually identifying anatomical landmarks in standard planes. Manual landmarking is time-consuming, operator-dependent, and sensitive to variability across scanners and sites, limiting the reproducibility of automated approaches. There is a need for multi-source annotated datasets to develop artificial intelligence-assisted fetal growth assessment methods. To address this bottleneck, we present an open, multi-centre, multi-device benchmark dataset of fetal US images with expert anatomical landmark annotations for clinically used fetal biometric measurements. These measurements include head bi-parietal and occipito-frontal diameters, abdominal transverse and antero-posterior diameters, and femoral length. The dataset contains 4,513 de-identified US images from 1,904 subjects acquired at three clinical sites using seven different US devices. We provide standardised, subject-disjoint train/test splits, evaluation code, and baseline results to enable fair and reproducible comparison of methods. Using an automatic biometry model, we quantify domain shift and demonstrate that training and evaluation confined to a single centre substantially overestimate performance relative to multi-centre testing. To the best of our knowledge, this is the first publicly available multi-centre, multi-device, landmark-annotated dataset that covers all primary fetal biometry measures, providing a robust benchmark for domain adaptation and multi-centre generalisation in fetal biometry and enabling more reliable AI-assisted fetal growth assessment across centres. All data, annotations, training code, and evaluation pipelines are made publicly available.
>
---
#### [new 057] AI-Powered Dermatological Diagnosis: From Interpretable Models to Clinical Implementation A Comprehensive Framework for Accessible and Trustworthy Skin Disease Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属医疗AI任务，旨在解决皮肤科诊断中专家短缺与家族史利用不足问题。提出多模态可解释AI框架，融合皮肤影像与结构化家族史数据，结合CNN与临床决策树，提升 melanoma 等遗传性皮肤病诊断准确率，并支持临床落地与验证。**

- **链接: [https://arxiv.org/pdf/2512.16235v1](https://arxiv.org/pdf/2512.16235v1)**

> **作者:** Satya Narayana Panda; Vaishnavi Kukkala; Spandana Iyer
>
> **备注:** 9 pages, 5 figures, 1 table. Code available at https://github.com/colabre2020/Enhancing-Skin-Disease-Diagnosis
>
> **摘要:** Dermatological conditions affect 1.9 billion people globally, yet accurate diagnosis remains challenging due to limited specialist availability and complex clinical presentations. Family history significantly influences skin disease susceptibility and treatment responses, but is often underutilized in diagnostic processes. This research addresses the critical question: How can AI-powered systems integrate family history data with clinical imaging to enhance dermatological diagnosis while supporting clinical trial validation and real-world implementation? We developed a comprehensive multi-modal AI framework that combines deep learning-based image analysis with structured clinical data, including detailed family history patterns. Our approach employs interpretable convolutional neural networks integrated with clinical decision trees that incorporate hereditary risk factors. The methodology includes prospective clinical trials across diverse healthcare settings to validate AI-assisted diagnosis against traditional clinical assessment. In this work, validation was conducted with healthcare professionals to assess AI-assisted outputs against clinical expectations; prospective clinical trials across diverse healthcare settings are proposed as future work. The integrated AI system demonstrates enhanced diagnostic accuracy when family history data is incorporated, particularly for hereditary skin conditions such as melanoma, psoriasis, and atopic dermatitis. Expert feedback indicates potential for improved early detection and more personalized recommendations; formal clinical trials are planned. The framework is designed for integration into clinical workflows while maintaining interpretability through explainable AI mechanisms.
>
---
#### [new 058] LinkedOut: Linking World Knowledge Representation Out of Video LLM for Next-Generation Video Recommendation
- **分类: cs.CV; cs.AI; cs.IR; cs.LG; cs.MM**

- **简介: 该论文面向视频推荐任务，解决VLLM难以直接用于低延迟、多视频、细粒度视觉推荐的问题。提出LinkedOut表示法，从原始视频帧中提取知识感知的语义token，结合跨层MoE融合，实现轻量、快速、可解释的推荐。**

- **链接: [https://arxiv.org/pdf/2512.16891v1](https://arxiv.org/pdf/2512.16891v1)**

> **作者:** Haichao Zhang; Yao Lu; Lichen Wang; Yunzhe Li; Daiwei Chen; Yunpeng Xu; Yun Fu
>
> **摘要:** Video Large Language Models (VLLMs) unlock world-knowledge-aware video understanding through pretraining on internet-scale data and have already shown promise on tasks such as movie analysis and video question answering. However, deploying VLLMs for downstream tasks such as video recommendation remains challenging, since real systems require multi-video inputs, lightweight backbones, low-latency sequential inference, and rapid response. In practice, (1) decode-only generation yields high latency for sequential inference, (2) typical interfaces do not support multi-video inputs, and (3) constraining outputs to language discards fine-grained visual details that matter for downstream vision tasks. We argue that these limitations stem from the absence of a representation that preserves pixel-level detail while leveraging world knowledge. We present LinkedOut, a representation that extracts VLLM world knowledge directly from video to enable fast inference, supports multi-video histories, and removes the language bottleneck. LinkedOut extracts semantically grounded, knowledge-aware tokens from raw frames using VLLMs, guided by promptable queries and optional auxiliary modalities. We introduce a cross-layer knowledge fusion MoE that selects the appropriate level of abstraction from the rich VLLM features, enabling personalized, interpretable, and low-latency recommendation. To our knowledge, LinkedOut is the first VLLM-based video recommendation method that operates on raw frames without handcrafted labels, achieving state-of-the-art results on standard benchmarks. Interpretability studies and ablations confirm the benefits of layer diversity and layer-wise fusion, pointing to a practical path that fully leverages VLLM world-knowledge priors and visual reasoning for downstream vision tasks such as recommendation.
>
---
#### [new 059] Few-Shot Fingerprinting Subject Re-Identification in 3D-MRI and 2D-X-Ray
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究跨模态（3D-MRI/2D-X-ray）少样本主体重识别任务，旨在解决因数据集混用导致的主体泄露与性能虚高问题。提出基于ResNet-50和三元组损失的主体指纹学习方法，通过隐空间聚类实现主体匹配，在多个医学数据集上验证了高召回率。**

- **链接: [https://arxiv.org/pdf/2512.16685v1](https://arxiv.org/pdf/2512.16685v1)**

> **作者:** Gonçalo Gaspar Alves; Shekoufeh Gorgi Zadeh; Andreas Husch; Ben Bausch
>
> **摘要:** Combining open-source datasets can introduce data leakage if the same subject appears in multiple sets, leading to inflated model performance. To address this, we explore subject fingerprinting, mapping all images of a subject to a distinct region in latent space, to enable subject re-identification via similarity matching. Using a ResNet-50 trained with triplet margin loss, we evaluate few-shot fingerprinting on 3D MRI and 2D X-ray data in both standard (20-way 1-shot) and challenging (1000-way 1-shot) scenarios. The model achieves high Mean- Recall-@-K scores: 99.10% (20-way 1-shot) and 90.06% (500-way 5-shot) on ChestXray-14; 99.20% (20-way 1-shot) and 98.86% (100-way 3-shot) on BraTS- 2021.
>
---
#### [new 060] R3ST: A Synthetic 3D Dataset With Realistic Trajectories
- **分类: cs.CV**

- **简介: 该论文面向轨迹预测任务，旨在解决合成数据缺乏真实车辆运动模式的问题。作者提出R3ST数据集，通过将真实世界轨迹（源自SinD无人机数据）融入合成3D环境，实现高精度、多模态真值标注与真实驾驶轨迹的结合。**

- **链接: [https://arxiv.org/pdf/2512.16784v1](https://arxiv.org/pdf/2512.16784v1)**

> **作者:** Simone Teglia; Claudia Melis Tonti; Francesco Pro; Leonardo Russo; Andrea Alfarano; Leonardo Pentassuglia; Irene Amerini
>
> **摘要:** Datasets are essential to train and evaluate computer vision models used for traffic analysis and to enhance road safety. Existing real datasets fit real-world scenarios, capturing authentic road object behaviors, however, they typically lack precise ground-truth annotations. In contrast, synthetic datasets play a crucial role, allowing for the annotation of a large number of frames without additional costs or extra time. However, a general drawback of synthetic datasets is the lack of realistic vehicle motion, since trajectories are generated using AI models or rule-based systems. In this work, we introduce R3ST (Realistic 3D Synthetic Trajectories), a synthetic dataset that overcomes this limitation by generating a synthetic 3D environment and integrating real-world trajectories derived from SinD, a bird's-eye-view dataset recorded from drone footage. The proposed dataset closes the gap between synthetic data and realistic trajectories, advancing the research in trajectory forecasting of road vehicles, offering both accurate multimodal ground-truth annotations and authentic human-driven vehicle trajectories.
>
---
#### [new 061] LAPX: Lightweight Hourglass Network with Global Context
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向人体姿态估计任务，旨在解决轻量模型精度低、边缘设备部署难的问题。提出轻量级沙漏网络LAPX，引入自注意力机制捕获全局上下文，优化阶段设计与注意力模块，在仅2.3M参数下实现高精度与实时性。**

- **链接: [https://arxiv.org/pdf/2512.16089v1](https://arxiv.org/pdf/2512.16089v1)**

> **作者:** Haopeng Zhao; Marsha Mariya Kappan; Mahdi Bamdad; Francisco Cruz
>
> **备注:** 10 pages
>
> **摘要:** Human pose estimation is a crucial task in computer vision. Methods that have SOTA (State-of-the-Art) accuracy, often involve a large number of parameters and incur substantial computational cost. Many lightweight variants have been proposed to reduce the model size and computational cost of them. However, several of these methods still contain components that are not well suited for efficient deployment on edge devices. Moreover, models that primarily emphasize inference speed on edge devices often suffer from limited accuracy due to their overly simplified designs. To address these limitations, we propose LAPX, an Hourglass network with self-attention that captures global contextual information, based on previous work, LAP. In addition to adopting the self-attention module, LAPX advances the stage design and refine the lightweight attention modules. It achieves competitive results on two benchmark datasets, MPII and COCO, with only 2.3M parameters, and demonstrates real-time performance, confirming its edge-device suitability.
>
---
#### [new 062] Next-Generation License Plate Detection and Recognition System using YOLOv8
- **分类: cs.CV; cs.AI**

- **简介: 该论文属智能交通中的车牌检测与识别（LPR）任务，旨在解决复杂环境下实时高精度LPR难题。提出基于YOLOv8 Nano（车牌检测）和YOLOv8 Small（字符识别）的轻量级协同 pipeline，并设计x轴排序的字符序列方法，兼顾精度与边缘部署效率。**

- **链接: [https://arxiv.org/pdf/2512.16826v1](https://arxiv.org/pdf/2512.16826v1)**

> **作者:** Arslan Amin; Rafia Mumtaz; Muhammad Jawad Bashir; Syed Mohammad Hassan Zaidi
>
> **备注:** 6 pages, 5 figures. Accepted and published in the 2023 IEEE 20th International Conference on Smart Communities: Improving Quality of Life using AI, Robotics and IoT (HONET)
>
> **摘要:** In the evolving landscape of traffic management and vehicle surveillance, efficient license plate detection and recognition are indispensable. Historically, many methodologies have tackled this challenge, but consistent real-time accuracy, especially in diverse environments, remains elusive. This study examines the performance of YOLOv8 variants on License Plate Recognition (LPR) and Character Recognition tasks, crucial for advancing Intelligent Transportation Systems. Two distinct datasets were employed for training and evaluation, yielding notable findings. The YOLOv8 Nano variant demonstrated a precision of 0.964 and mAP50 of 0.918 on the LPR task, while the YOLOv8 Small variant exhibited a precision of 0.92 and mAP50 of 0.91 on the Character Recognition task. A custom method for character sequencing was introduced, effectively sequencing the detected characters based on their x-axis positions. An optimized pipeline, utilizing YOLOv8 Nano for LPR and YOLOv8 Small for Character Recognition, is proposed. This configuration not only maintains computational efficiency but also ensures high accuracy, establishing a robust foundation for future real-world deployments on edge devices within Intelligent Transportation Systems. This effort marks a significant stride towards the development of smarter and more efficient urban infrastructures.
>
---
#### [new 063] CRONOS: Continuous Time Reconstruction for 4D Medical Longitudinal Series
- **分类: cs.CV**

- **简介: 该论文提出CRONOS框架，解决3D医学影像在不规则时间采样下的连续时间体素级预测任务。它通过学习时空速度场，实现多时序扫描到任意时刻目标扫描的端到端重建，支持离散与连续时间建模，是首个面向3D医学数据的连续序列到图像预测方法。**

- **链接: [https://arxiv.org/pdf/2512.16577v1](https://arxiv.org/pdf/2512.16577v1)**

> **作者:** Nico Albert Disch; Saikat Roy; Constantin Ulrich; Yannick Kirchhoff; Maximilian Rokuss; Robin Peretzke; David Zimmerer; Klaus Maier-Hein
>
> **备注:** https://github.com/MIC-DKFZ/Longitudinal4DMed
>
> **摘要:** Forecasting how 3D medical scans evolve over time is important for disease progression, treatment planning, and developmental assessment. Yet existing models either rely on a single prior scan, fixed grid times, or target global labels, which limits voxel-level forecasting under irregular sampling. We present CRONOS, a unified framework for many-to-one prediction from multiple past scans that supports both discrete (grid-based) and continuous (real-valued) timestamps in one model, to the best of our knowledge the first to achieve continuous sequence-to-image forecasting for 3D medical data. CRONOS learns a spatio-temporal velocity field that transports context volumes toward a target volume at an arbitrary time, while operating directly in 3D voxel space. Across three public datasets spanning Cine-MRI, perfusion CT, and longitudinal MRI, CRONOS outperforms other baselines, while remaining computationally competitive. We will release code and evaluation protocols to enable reproducible, multi-dataset benchmarking of multi-context, continuous-time forecasting.
>
---
#### [new 064] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文面向域泛化语义分割任务，旨在解决视觉基础模型（VFMs）因预训练伪影导致的跨域泛化性能下降问题。提出Causal-Tune方法，通过DCT频谱分析分离因果与非因果特征成分，用可学习令牌增强因果部分、抑制非因果部分，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.16567v1](https://arxiv.org/pdf/2512.16567v1)**

> **作者:** Yin Zhang; Yongqiang Zhang; Yaoyue Zheng; Bogdan Raducanu; Dan Liu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Fine-tuning Vision Foundation Models (VFMs) with a small number of parameters has shown remarkable performance in Domain Generalized Semantic Segmentation (DGSS). Most existing works either train lightweight adapters or refine intermediate features to achieve better generalization on unseen domains. However, they both overlook the fact that long-term pre-trained VFMs often exhibit artifacts, which hinder the utilization of valuable representations and ultimately degrade DGSS performance. Inspired by causal mechanisms, we observe that these artifacts are associated with non-causal factors, which usually reside in the low- and high-frequency components of the VFM spectrum. In this paper, we explicitly examine the causal and non-causal factors of features within VFMs for DGSS, and propose a simple yet effective method to identify and disentangle them, enabling more robust domain generalization. Specifically, we propose Causal-Tune, a novel fine-tuning strategy designed to extract causal factors and suppress non-causal ones from the features of VFMs. First, we extract the frequency spectrum of features from each layer using the Discrete Cosine Transform (DCT). A Gaussian band-pass filter is then applied to separate the spectrum into causal and non-causal components. To further refine the causal components, we introduce a set of causal-aware learnable tokens that operate in the frequency domain, while the non-causal components are discarded. Finally, refined features are transformed back into the spatial domain via inverse DCT and passed to the next layer. Extensive experiments conducted on various cross-domain tasks demonstrate the effectiveness of Causal-Tune. In particular, our method achieves superior performance under adverse weather conditions, improving +4.8% mIoU over the baseline in snow conditions.
>
---
#### [new 065] Are vision-language models ready to zero-shot replace supervised classification models in agriculture?
- **分类: cs.CV**

- **简介: 该论文属农业视觉识别任务，旨在评估VLMs能否零样本替代监督模型。作者在27个农业数据集上系统评测多类VLMs，发现其性能显著低于YOLO11基线；揭示提示方式、评估方法对结果影响大，指出当前VLMs尚不能独立用于农业诊断，但可作为辅助组件。**

- **链接: [https://arxiv.org/pdf/2512.15977v1](https://arxiv.org/pdf/2512.15977v1)**

> **作者:** Earl Ranario; Mason J. Earles
>
> **备注:** Draft version
>
> **摘要:** Vision-language models (VLMs) are increasingly proposed as general-purpose solutions for visual recognition tasks, yet their reliability for agricultural decision support remains poorly understood. We benchmark a diverse set of open-source and closed-source VLMs on 27 agricultural classification datasets from the AgML collection, spanning 162 classes across plant disease, pest and damage, and plant and weed species identification. Across all tasks, zero-shot VLMs substantially underperform a supervised task-specific baseline (YOLO11), which consistently achieves markedly higher accuracy than any foundation model. Under multiple-choice prompting, the best-performing VLM (Gemini-3 Pro) reaches approximately 62% average accuracy, while open-ended prompting yields much lower performance, with raw accuracies typically below 25%. Applying LLM-based semantic judging increases open-ended accuracy (for example, from 21% to 30% for top models) and alters model rankings, demonstrating that evaluation methodology meaningfully affects reported conclusions. Among open-source models, Qwen-VL-72B performs best, approaching closed-source performance under constrained prompting but still trailing top proprietary systems. Task-level analysis shows that plant and weed species classification is consistently easier than pest and damage identification, which remains the most challenging category across models. Overall, these results indicate that current off-the-shelf VLMs are not yet suitable as standalone agricultural diagnostic systems, but can function as assistive components when paired with constrained interfaces, explicit label ontologies, and domain-aware evaluation strategies.
>
---
#### [new 066] Hazedefy: A Lightweight Real-Time Image and Video Dehazing Pipeline for Practical Deployment
- **分类: cs.CV**

- **简介: 该论文提出Hazedefy，一种轻量级实时图像/视频去雾方法，面向消费级硬件部署。针对传统去雾算法计算复杂、难实用的问题，它基于暗通道先验与大气散射模型，改进了传输图估计、大气光估算和重建策略，支持无GPU实时运行。**

- **链接: [https://arxiv.org/pdf/2512.16609v1](https://arxiv.org/pdf/2512.16609v1)**

> **作者:** Ayush Bhavsar
>
> **备注:** 4 pages, 2 figures. Code and demo available at https://doi.org/10.5281/zenodo.17915355
>
> **摘要:** This paper introduces Hazedefy, a lightweight and application-focused dehazing pipeline intended for real-time video and live camera feed enhancement. Hazedefy prioritizes computational simplicity and practical deployability on consumer-grade hardware, building upon the Dark Channel Prior (DCP) concept and the atmospheric scattering model. Key elements include gamma-adaptive reconstruction, a fast transmission approximation with lower bounds for numerical stability, a stabilized atmospheric light estimator based on fractional top-pixel averaging, and an optional color balance stage. The pipeline is suitable for mobile and embedded applications, as experimental demonstrations on real-world images and videos show improved visibility and contrast without requiring GPU acceleration.
>
---
#### [new 067] N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出N3D-VLM框架，解决现有视觉语言模型缺乏原生3D感知、难以准确理解空间关系的问题。工作包括：构建原生3D物体感知与3D空间推理统一模型；设计基于深度估计的大规模3D标注生成流水线；创建支持链式推理的3D空间问答数据集。**

- **链接: [https://arxiv.org/pdf/2512.16561v1](https://arxiv.org/pdf/2512.16561v1)**

> **作者:** Yuxin Wang; Lei Ke; Boqiang Zhang; Tianyuan Qu; Hanxun Yu; Zhenpeng Huang; Meng Yu; Dan Xu; Dong Yu
>
> **备注:** Project Page: https://n3d-vlm.github.io
>
> **摘要:** While current multimodal models can answer questions based on 2D images, they lack intrinsic 3D object perception, limiting their ability to comprehend spatial relationships and depth cues in 3D scenes. In this work, we propose N3D-VLM, a novel unified framework that seamlessly integrates native 3D object perception with 3D-aware visual reasoning, enabling both precise 3D grounding and interpretable spatial understanding. Unlike conventional end-to-end models that directly predict answers from RGB/RGB-D inputs, our approach equips the model with native 3D object perception capabilities, enabling it to directly localize objects in 3D space based on textual descriptions. Building upon accurate 3D object localization, the model further performs explicit reasoning in 3D, achieving more interpretable and structured spatial understanding. To support robust training for these capabilities, we develop a scalable data construction pipeline that leverages depth estimation to lift large-scale 2D annotations into 3D space, significantly increasing the diversity and coverage for 3D object grounding data, yielding over six times larger than the largest existing single-image 3D detection dataset. Moreover, the pipeline generates spatial question-answering datasets that target chain-of-thought (CoT) reasoning in 3D, facilitating joint training for both 3D object localization and 3D spatial reasoning. Experimental results demonstrate that our unified framework not only achieves state-of-the-art performance on 3D grounding tasks, but also consistently surpasses existing methods in 3D spatial reasoning in vision-language model.
>
---
#### [new 068] SFTok: Bridging the Performance Gap in Discrete Tokenizers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属图像tokenization任务，旨在解决离散tokenizer重建质量差、落后于连续tokenizer的问题。提出SFTok，通过自强制引导重建与去偏-拟合训练，提升高倍压缩下的图像重建与生成性能。**

- **链接: [https://arxiv.org/pdf/2512.16910v1](https://arxiv.org/pdf/2512.16910v1)**

> **作者:** Qihang Rao; Borui Zhang; Wenzhao Zheng; Jie Zhou; Jiwen Lu
>
> **备注:** Under review. Code is available at https://github.com/Neur-IO/SFTok
>
> **摘要:** Recent advances in multimodal models highlight the pivotal role of image tokenization in high-resolution image generation. By compressing images into compact latent representations, tokenizers enable generative models to operate in lower-dimensional spaces, thereby improving computational efficiency and reducing complexity. Discrete tokenizers naturally align with the autoregressive paradigm but still lag behind continuous ones, limiting their adoption in multimodal systems. To address this, we propose \textbf{SFTok}, a discrete tokenizer that incorporates a multi-step iterative mechanism for precise reconstruction. By integrating \textbf{self-forcing guided visual reconstruction} and \textbf{debias-and-fitting training strategy}, SFTok resolves the training-inference inconsistency in multi-step process, significantly enhancing image reconstruction quality. At a high compression rate of only 64 tokens per image, SFTok achieves state-of-the-art reconstruction quality on ImageNet (rFID = 1.21) and demonstrates exceptional performance in class-to-image generation tasks (gFID = 2.29).
>
---
#### [new 069] VIVA: VLM-Guided Instruction-Based Video Editing with Reward Optimization
- **分类: cs.CV**

- **简介: 该论文面向指令式视频编辑任务，解决现有扩散模型泛化能力弱、难处理复杂真实指令的问题。提出VIVA框架：用VLM编码多模态输入以增强语义理解，并通过Edit-GRPO奖励优化提升指令遵循性与编辑质量，辅以合成数据构建 pipeline。**

- **链接: [https://arxiv.org/pdf/2512.16906v1](https://arxiv.org/pdf/2512.16906v1)**

> **作者:** Xiaoyan Cong; Haotian Yang; Angtian Wang; Yizhi Wang; Yiding Yang; Canyu Zhang; Chongyang Ma
>
> **摘要:** Instruction-based video editing aims to modify an input video according to a natural-language instruction while preserving content fidelity and temporal coherence. However, existing diffusion-based approaches are often trained on paired data of simple editing operations, which fundamentally limits their ability to generalize to diverse and complex, real-world instructions. To address this generalization gap, we propose VIVA, a scalable framework for instruction-based video editing that leverages VLM-guided encoding and reward optimization. First, we introduce a VLM-based instructor that encodes the textual instruction, the first frame of the source video, and an optional reference image into visually-grounded instruction representations, providing fine-grained spatial and semantic context for the diffusion transformer backbone. Second, we propose a post-training stage, Edit-GRPO, which adapts Group Relative Policy Optimization to the domain of video editing, directly optimizing the model for instruction-faithful, content-preserving, and aesthetically pleasing edits using relative rewards. Furthermore, we propose a data construction pipeline designed to synthetically generate diverse, high-fidelity paired video-instruction data of basic editing operations. Extensive experiments show that VIVA achieves superior instruction following, generalization, and editing quality over state-of-the-art methods. Website: https://viva-paper.github.io
>
---
#### [new 070] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文面向遥感语义分割任务，解决合成数据控制难、采样质量不稳定问题。提出任务导向的数据合成框架TODSynth，包含多模态扩散Transformer（MM-DiT）与控制-修正流匹配（CRFM）采样策略，提升少样本与复杂场景下合成数据的实用性与任务适配性。**

- **链接: [https://arxiv.org/pdf/2512.16740v1](https://arxiv.org/pdf/2512.16740v1)**

> **作者:** Yunkai Yang; Yudong Zhang; Kunquan Zhang; Jinxiao Zhang; Xinying Chen; Haohuan Fu; Runmin Dong
>
> **摘要:** With the rapid progress of controllable generation, training data synthesis has become a promising way to expand labeled datasets and alleviate manual annotation in remote sensing (RS). However, the complexity of semantic mask control and the uncertainty of sampling quality often limit the utility of synthetic data in downstream semantic segmentation tasks. To address these challenges, we propose a task-oriented data synthesis framework (TODSynth), including a Multimodal Diffusion Transformer (MM-DiT) with unified triple attention and a plug-and-play sampling strategy guided by task feedback. Built upon the powerful DiT-based generative foundation model, we systematically evaluate different control schemes, showing that a text-image-mask joint attention scheme combined with full fine-tuning of the image and mask branches significantly enhances the effectiveness of RS semantic segmentation data synthesis, particularly in few-shot and complex-scene scenarios. Furthermore, we propose a control-rectify flow matching (CRFM) method, which dynamically adjusts sampling directions guided by semantic loss during the early high-plasticity stage, mitigating the instability of generated images and bridging the gap between synthetic data and downstream segmentation tasks. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art controllable generation methods, producing more stable and task-oriented synthetic data for RS semantic segmentation.
>
---
#### [new 071] Generative Refocusing: Flexible Defocus Control from a Single Image
- **分类: cs.CV**

- **简介: 该论文提出“生成式重聚焦”方法，解决单图景深控制难题。它分两步：先用DeblurNet恢复全焦图像，再用BokehNet合成可控散景。创新在于半监督训练，融合合成配对数据与真实非配对散景图像，并利用EXIF元数据建模真实光学特性，支持文本引导和自定义光圈形状。**

- **链接: [https://arxiv.org/pdf/2512.16923v1](https://arxiv.org/pdf/2512.16923v1)**

> **作者:** Chun-Wei Tuan Mu; Jia-Bin Huang; Yu-Lun Liu
>
> **备注:** Project website: https://generative-refocusing.github.io/
>
> **摘要:** Depth-of-field control is essential in photography, but getting the perfect focus often takes several tries or special equipment. Single-image refocusing is still difficult. It involves recovering sharp content and creating realistic bokeh. Current methods have significant drawbacks. They need all-in-focus inputs, depend on synthetic data from simulators, and have limited control over aperture. We introduce Generative Refocusing, a two-step process that uses DeblurNet to recover all-in-focus images from various inputs and BokehNet for creating controllable bokeh. Our main innovation is semi-supervised training. This method combines synthetic paired data with unpaired real bokeh images, using EXIF metadata to capture real optical characteristics beyond what simulators can provide. Our experiments show we achieve top performance in defocus deblurring, bokeh synthesis, and refocusing benchmarks. Additionally, our Generative Refocusing allows text-guided adjustments and custom aperture shapes.
>
---
#### [new 072] Detecting Localized Deepfakes: How Well Do Synthetic Image Detectors Handle Inpainting?
- **分类: cs.CV**

- **简介: 该论文属图像伪造检测任务，旨在评估现有深度伪造检测器对局部修复（inpainting）伪造的泛化能力。研究系统测试了多种先进检测器在不同生成器、掩模尺寸和修复方法下的表现，发现其对中大型区域或再生式修复具有一定检测能力。**

- **链接: [https://arxiv.org/pdf/2512.16688v1](https://arxiv.org/pdf/2512.16688v1)**

> **作者:** Serafino Pandolfini; Lorenzo Pellegrini; Matteo Ferrara; Davide Maltoni
>
> **备注:** 17 pages, 5 figures, 9 tables
>
> **摘要:** The rapid progress of generative AI has enabled highly realistic image manipulations, including inpainting and region-level editing. These approaches preserve most of the original visual context and are increasingly exploited in cybersecurity-relevant threat scenarios. While numerous detectors have been proposed for identifying fully synthetic images, their ability to generalize to localized manipulations remains insufficiently characterized. This work presents a systematic evaluation of state-of-the-art detectors, originally trained for the deepfake detection on fully synthetic images, when applied to a distinct challenge: localized inpainting detection. The study leverages multiple datasets spanning diverse generators, mask sizes, and inpainting techniques. Our experiments show that models trained on a large set of generators exhibit partial transferability to inpainting-based edits and can reliably detect medium- and large-area manipulations or regeneration-style inpainting, outperforming many existing ad hoc detection approaches.
>
---
#### [new 073] GFLAN: Generative Functional Layouts
- **分类: cs.CV; cs.AI**

- **简介: 该论文属建筑生成任务，旨在解决自动平面图生成中拓扑关系与几何约束难以协同建模的问题。提出GFLAN框架，分两阶段：先用双编码CNN分配房间中心点，再用Transformer增强GNN联合回归房间边界。**

- **链接: [https://arxiv.org/pdf/2512.16275v1](https://arxiv.org/pdf/2512.16275v1)**

> **作者:** Mohamed Abouagour; Eleftherios Garyfallidis
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Automated floor plan generation lies at the intersection of combinatorial search, geometric constraint satisfaction, and functional design requirements -- a confluence that has historically resisted a unified computational treatment. While recent deep learning approaches have improved the state of the art, they often struggle to capture architectural reasoning: the precedence of topological relationships over geometric instantiation, the propagation of functional constraints through adjacency networks, and the emergence of circulation patterns from local connectivity decisions. To address these fundamental challenges, this paper introduces GFLAN, a generative framework that restructures floor plan synthesis through explicit factorization into topological planning and geometric realization. Given a single exterior boundary and a front-door location, our approach departs from direct pixel-to-pixel or wall-tracing generation in favor of a principled two-stage decomposition. Stage A employs a specialized convolutional architecture with dual encoders -- separating invariant spatial context from evolving layout state -- to sequentially allocate room centroids within the building envelope via discrete probability maps over feasible placements. Stage B constructs a heterogeneous graph linking room nodes to boundary vertices, then applies a Transformer-augmented graph neural network (GNN) that jointly regresses room boundaries.
>
---
#### [new 074] The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text
- **分类: cs.CV**

- **简介: 该论文提出WorldCanvas框架，属多模态视频生成任务，旨在解决现有方法在事件可控性、多主体交互与视觉一致性上的不足。它融合文本语义、运动轨迹和参考图像，实现用户可提示的、具身份一致性和场景连贯性的世界事件模拟。**

- **链接: [https://arxiv.org/pdf/2512.16924v1](https://arxiv.org/pdf/2512.16924v1)**

> **作者:** Hanlin Wang; Hao Ouyang; Qiuyu Wang; Yue Yu; Yihao Meng; Wen Wang; Ka Leong Cheng; Shuailei Ma; Qingyan Bai; Yixuan Li; Cheng Chen; Yanhong Zeng; Xing Zhu; Yujun Shen; Qifeng Chen
>
> **备注:** Project page and code: https://worldcanvas.github.io/
>
> **摘要:** We present WorldCanvas, a framework for promptable world events that enables rich, user-directed simulation by combining text, trajectories, and reference images. Unlike text-only approaches and existing trajectory-controlled image-to-video methods, our multimodal approach combines trajectories -- encoding motion, timing, and visibility -- with natural language for semantic intent and reference images for visual grounding of object identity, enabling the generation of coherent, controllable events that include multi-agent interactions, object entry/exit, reference-guided appearance and counterintuitive events. The resulting videos demonstrate not only temporal coherence but also emergent consistency, preserving object identity and scene despite temporary disappearance. By supporting expressive world events generation, WorldCanvas advances world models from passive predictors to interactive, user-shaped simulators. Our project page is available at: https://worldcanvas.github.io/.
>
---
#### [new 075] TreeNet: A Light Weight Model for Low Bitrate Image Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文属图像压缩任务，旨在解决学习型压缩模型计算复杂度高、难部署的问题。提出轻量级TreeNet模型，采用二叉树编解码结构与注意力特征融合，在低码率下较JPEG AI提升4.83% BD-rate，降低87.82%复杂度。**

- **链接: [https://arxiv.org/pdf/2512.16743v1](https://arxiv.org/pdf/2512.16743v1)**

> **作者:** Mahadev Prasad Panda; Purnachandra Rao Makkena; Srivatsa Prativadibhayankaram; Siegfried Fößel; André Kaup
>
> **摘要:** Reducing computational complexity remains a critical challenge for the widespread adoption of learning-based image compression techniques. In this work, we propose TreeNet, a novel low-complexity image compression model that leverages a binary tree-structured encoder-decoder architecture to achieve efficient representation and reconstruction. We employ attentional feature fusion mechanism to effectively integrate features from multiple branches. We evaluate TreeNet on three widely used benchmark datasets and compare its performance against competing methods including JPEG AI, a recent standard in learning-based image compression. At low bitrates, TreeNet achieves an average improvement of 4.83% in BD-rate over JPEG AI, while reducing model complexity by 87.82%. Furthermore, we conduct extensive ablation studies to investigate the influence of various latent representations within TreeNet, offering deeper insights into the factors contributing to reconstruction.
>
---
#### [new 076] Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation
- **分类: cs.CV**

- **简介: 该论文面向全景图像的**单目深度估计任务**，旨在解决全景图中远近场景深度预测不准确、跨域泛化差的问题。工作包括：构建多源全景数据集；提出三阶段伪标签清洗流程；设计带范围掩码头、锐度与几何优化的DINOv3基础模型，实现强零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.16913v1](https://arxiv.org/pdf/2512.16913v1)**

> **作者:** Xin Lin; Meixi Song; Dizhe Zhang; Wenxuan Lu; Haodong Li; Bo Du; Ming-Hsuan Yang; Truong Nguyen; Lu Qi
>
> **备注:** Project Page: https://insta360-research-team.github.io/DAP_website/
>
> **摘要:** In this work, we present a panoramic metric depth foundation model that generalizes across diverse scene distances. We explore a data-in-the-loop paradigm from the view of both data construction and framework design. We collect a large-scale dataset by combining public datasets, high-quality synthetic data from our UE5 simulator and text-to-image models, and real panoramic images from the web. To reduce domain gaps between indoor/outdoor and synthetic/real data, we introduce a three-stage pseudo-label curation pipeline to generate reliable ground truth for unlabeled images. For the model, we adopt DINOv3-Large as the backbone for its strong pre-trained generalization, and introduce a plug-and-play range mask head, sharpness-centric optimization, and geometry-centric optimization to improve robustness to varying distances and enforce geometric consistency across views. Experiments on multiple benchmarks (e.g., Stanford2D3D, Matterport3D, and Deep360) demonstrate strong performance and zero-shot generalization, with particularly robust and stable metric predictions in diverse real-world scenes. The project page can be found at: \href{https://insta360-research-team.github.io/DAP_website/} {https://insta360-research-team.github.io/DAP\_website/}
>
---
#### [new 077] Memory-Enhanced SAM3 for Occlusion-Robust Surgical Instrument Segmentation
- **分类: cs.CV**

- **简介: 该论文面向手术视频中器械分割任务，解决遮挡、快速运动等导致的分割不鲁棒问题。提出无需训练的ReMeDI-SAM3方法，通过遮挡感知记忆过滤、分段插值扩容和特征重识别模块，显著提升遮挡后恢复能力，在EndoVis数据集上零样本下mcIoU提升7%–16%。**

- **链接: [https://arxiv.org/pdf/2512.16880v1](https://arxiv.org/pdf/2512.16880v1)**

> **作者:** Valay Bundele; Mehran Hosseinzadeh; Hendrik P. A. Lensch
>
> **备注:** Under Review
>
> **摘要:** Accurate surgical instrument segmentation in endoscopic videos is crucial for computer-assisted interventions, yet remains challenging due to frequent occlusions, rapid motion, specular artefacts, and long-term instrument re-entry. While SAM3 provides a powerful spatio-temporal framework for video object segmentation, its performance in surgical scenes is limited by indiscriminate memory updates, fixed memory capacity, and weak identity recovery after occlusions. We propose ReMeDI-SAM3, a training-free memory-enhanced extension of SAM3, that addresses these limitations through three components: (i) relevance-aware memory filtering with a dedicated occlusion-aware memory for storing pre-occlusion frames, (ii) a piecewise interpolation scheme that expands the effective memory capacity, and (iii) a feature-based re-identification module with temporal voting for reliable post-occlusion identity disambiguation. Together, these components mitigate error accumulation and enable reliable recovery after occlusions. Evaluations on EndoVis17 and EndoVis18 under a zero-shot setting show absolute mcIoU improvements of around 7% and 16%, respectively, over vanilla SAM3, outperforming even prior training-based approaches. Project page: https://valaybundele.github.io/remedi-sam3/.
>
---
#### [new 078] SceneDiff: A Benchmark and Method for Multiview Object Change Detection
- **分类: cs.CV**

- **简介: 该论文聚焦多视角场景下的物体变化检测任务，旨在准确识别同一场景不同时刻图像/视频中新增、消失或移动的物体。作者构建了首个带实例标注的多视角变化检测基准SceneDiff（含350视频对），并提出无需训练的SceneDiff方法，通过3D对齐、区域提取与时空语义特征比对实现鲁棒检测，在多个基准上显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.16908v1](https://arxiv.org/pdf/2512.16908v1)**

> **作者:** Yuqun Wu; Chih-hao Lin; Henry Che; Aditi Tiwari; Chuhang Zou; Shenlong Wang; Derek Hoiem
>
> **摘要:** We investigate the problem of identifying objects that have been added, removed, or moved between a pair of captures (images or videos) of the same scene at different times. Detecting such changes is important for many applications, such as robotic tidying or construction progress and safety monitoring. A major challenge is that varying viewpoints can cause objects to falsely appear changed. We introduce SceneDiff Benchmark, the first multiview change detection benchmark with object instance annotations, comprising 350 diverse video pairs with thousands of changed objects. We also introduce the SceneDiff method, a new training-free approach for multiview object change detection that leverages pretrained 3D, segmentation, and image encoding models to robustly predict across multiple benchmarks. Our method aligns the captures in 3D, extracts object regions, and compares spatial and semantic region features to detect changes. Experiments on multi-view and two-view benchmarks demonstrate that our method outperforms existing approaches by large margins (94% and 37.4% relative AP improvements). The benchmark and code will be publicly released.
>
---
#### [new 079] LaverNet: Lightweight All-in-one Video Restoration via Selective Propagation
- **分类: cs.CV**

- **简介: 该论文面向视频复原任务，解决时间变化退化下模型易受伪影干扰、现有方法参数量大两大问题。提出轻量级LaverNet（仅362K参数），通过选择性传播退化无关特征，实现高效多退化统一恢复。**

- **链接: [https://arxiv.org/pdf/2512.16313v1](https://arxiv.org/pdf/2512.16313v1)**

> **作者:** Haiyu Zhao; Yiwen Shan; Yuanbiao Gou; Xi Peng
>
> **摘要:** Recent studies have explored all-in-one video restoration, which handles multiple degradations with a unified model. However, these approaches still face two challenges when dealing with time-varying degradations. First, the degradation can dominate temporal modeling, confusing the model to focus on artifacts rather than the video content. Second, current methods typically rely on large models to handle all-in-one restoration, concealing those underlying difficulties. To address these challenges, we propose a lightweight all-in-one video restoration network, LaverNet, with only 362K parameters. To mitigate the impact of degradations on temporal modeling, we introduce a novel propagation mechanism that selectively transmits only degradation-agnostic features across frames. Through LaverNet, we demonstrate that strong all-in-one restoration can be achieved with a compact network. Despite its small size, less than 1\% of the parameters of existing models, LaverNet achieves comparable, even superior performance across benchmarks.
>
---
#### [new 080] Interaction-via-Actions: Cattle Interaction Detection with Joint Learning of Action-Interaction Latent Space
- **分类: cs.CV**

- **简介: 该论文属计算机视觉中的行为理解任务，旨在解决牛群稀疏交互检测难题。提出CattleAct方法：先构建牛个体动作潜在空间，再通过对比学习微调以联合建模动作与交互，实现单图交互检测，并集成视频/GPS开发实用系统。**

- **链接: [https://arxiv.org/pdf/2512.16133v1](https://arxiv.org/pdf/2512.16133v1)**

> **作者:** Ren Nakagawa; Yang Yang; Risa Shinoda; Hiroaki Santo; Kenji Oyama; Fumio Okura; Takenao Ohkawa
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** This paper introduces a method and application for automatically detecting behavioral interactions between grazing cattle from a single image, which is essential for smart livestock management in the cattle industry, such as for detecting estrus. Although interaction detection for humans has been actively studied, a non-trivial challenge lies in cattle interaction detection, specifically the lack of a comprehensive behavioral dataset that includes interactions, as the interactions of grazing cattle are rare events. We, therefore, propose CattleAct, a data-efficient method for interaction detection by decomposing interactions into the combinations of actions by individual cattle. Specifically, we first learn an action latent space from a large-scale cattle action dataset. Then, we embed rare interactions via the fine-tuning of the pre-trained latent space using contrastive learning, thereby constructing a unified latent space of actions and interactions. On top of the proposed method, we develop a practical working system integrating video and GPS inputs. Experiments on a commercial-scale pasture demonstrate the accurate interaction detection achieved by our method compared to the baselines. Our implementation is available at https://github.com/rakawanegan/CattleAct.
>
---
#### [new 081] Towards Closing the Domain Gap with Event Cameras
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属自动驾驶感知任务，旨在解决传统相机因昼夜光照变化导致的域间性能下降（域间隙）问题。作者提出采用事件相机替代传统相机，实验证明其在跨光照域场景下性能更稳定，域偏移惩罚更小，基线表现更优。**

- **链接: [https://arxiv.org/pdf/2512.16178v1](https://arxiv.org/pdf/2512.16178v1)**

> **作者:** M. Oltan Sevinc; Liao Wu; Francisco Cruz
>
> **备注:** Accepted to Australasian Conference on Robotics and Automation (ACRA), 2025
>
> **摘要:** Although traditional cameras are the primary sensor for end-to-end driving, their performance suffers greatly when the conditions of the data they were trained on does not match the deployment environment, a problem known as the domain gap. In this work, we consider the day-night lighting difference domain gap. Instead of traditional cameras we propose event cameras as a potential alternative which can maintain performance across lighting condition domain gaps without requiring additional adjustments. Our results show that event cameras maintain more consistent performance across lighting conditions, exhibiting domain-shift penalties that are generally comparable to or smaller than grayscale frames and provide superior baseline performance in cross-domain scenarios.
>
---
#### [new 082] Alchemist: Unlocking Efficiency in Text-to-Image Model Training via Meta-Gradient Data Selection
- **分类: cs.CV**

- **简介: 该论文面向文本到图像（T2I）模型训练的数据效率问题，提出Alchemist框架：基于元梯度自动评估样本影响力，通过轻量级评分器与Shift-Gsampling策略实现高效数据选择。首次将元梯度思想引入T2I数据筛选，显著提升训练效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2512.16905v1](https://arxiv.org/pdf/2512.16905v1)**

> **作者:** Kaixin Ding; Yang Zhou; Xi Chen; Miao Yang; Jiarong Ou; Rui Chen; Xin Tao; Hengshuang Zhao
>
> **备注:** project page: https://kxding.github.io/project/Alchemist/
>
> **摘要:** Recent advances in Text-to-Image (T2I) generative models, such as Imagen, Stable Diffusion, and FLUX, have led to remarkable improvements in visual quality. However, their performance is fundamentally limited by the quality of training data. Web-crawled and synthetic image datasets often contain low-quality or redundant samples, which lead to degraded visual fidelity, unstable training, and inefficient computation. Hence, effective data selection is crucial for improving data efficiency. Existing approaches rely on costly manual curation or heuristic scoring based on single-dimensional features in Text-to-Image data filtering. Although meta-learning based method has been explored in LLM, there is no adaptation for image modalities. To this end, we propose **Alchemist**, a meta-gradient-based framework to select a suitable subset from large-scale text-image data pairs. Our approach automatically learns to assess the influence of each sample by iteratively optimizing the model from a data-centric perspective. Alchemist consists of two key stages: data rating and data pruning. We train a lightweight rater to estimate each sample's influence based on gradient information, enhanced with multi-granularity perception. We then use the Shift-Gsampling strategy to select informative subsets for efficient model training. Alchemist is the first automatic, scalable, meta-gradient-based data selection framework for Text-to-Image model training. Experiments on both synthetic and web-crawled datasets demonstrate that Alchemist consistently improves visual quality and downstream performance. Training on an Alchemist-selected 50% of the data can outperform training on the full dataset.
>
---
#### [new 083] Guiding Perception-Reasoning Closer to Human in Blind Image Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属盲图像质量评估（BIQA）任务，旨在使模型兼具人类感知与自洽推理能力。作者采集人类感知-推理链数据，用强化学习以人类标注为奖励，并设计自描述推理奖励，提升模型在评分精度与解释对齐（ROUGE-1达0.512）上的表现。**

- **链接: [https://arxiv.org/pdf/2512.16484v1](https://arxiv.org/pdf/2512.16484v1)**

> **作者:** Yuan Li; Yahan Yu; Youyuan Lin; Yong-Hao Yang; Chenhui Chu; Shin'ya Nishida
>
> **备注:** Under review
>
> **摘要:** Humans assess image quality through a perception-reasoning cascade, integrating sensory cues with implicit reasoning to form self-consistent judgments. In this work, we investigate how a model can acquire both human-like and self-consistent reasoning capability for blind image quality assessment (BIQA). We first collect human evaluation data that capture several aspects of human perception-reasoning pipeline. Then, we adopt reinforcement learning, using human annotations as reward signals to guide the model toward human-like perception and reasoning. To enable the model to internalize self-consistent reasoning capability, we design a reward that drives the model to infer the image quality purely from self-generated descriptions. Empirically, our approach achieves score prediction performance comparable to state-of-the-art BIQA systems under general metrics, including Pearson and Spearman correlation coefficients. In addition to the rating score, we assess human-model alignment using ROUGE-1 to measure the similarity between model-generated and human perception-reasoning chains. On over 1,000 human-annotated samples, our model reaches a ROUGE-1 score of 0.512 (cf. 0.443 for baseline), indicating substantial coverage of human explanations and marking a step toward human-like interpretable reasoning in BIQA.
>
---
#### [new 084] Seeing is Believing (and Predicting): Context-Aware Multi-Human Behavior Prediction with Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向多人体行为预测任务，解决第三方视角下机器人对多人与场景交互行为的准确预测问题。提出CAMP-VLM框架，融合视觉特征与场景图空间信息，利用合成数据微调，并结合SFT与DPO优化，在预测精度上显著超越基线。**

- **链接: [https://arxiv.org/pdf/2512.15957v1](https://arxiv.org/pdf/2512.15957v1)**

> **作者:** Utsav Panchal; Yuchen Liu; Luigi Palmieri; Ilche Georgievski; Marco Aiello
>
> **备注:** Accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Accurately predicting human behaviors is crucial for mobile robots operating in human-populated environments. While prior research primarily focuses on predicting actions in single-human scenarios from an egocentric view, several robotic applications require understanding multiple human behaviors from a third-person perspective. To this end, we present CAMP-VLM (Context-Aware Multi-human behavior Prediction): a Vision Language Model (VLM)-based framework that incorporates contextual features from visual input and spatial awareness from scene graphs to enhance prediction of humans-scene interactions. Due to the lack of suitable datasets for multi-human behavior prediction from an observer view, we perform fine-tuning of CAMP-VLM with synthetic human behavior data generated by a photorealistic simulator, and evaluate the resulting models on both synthetic and real-world sequences to assess their generalization capabilities. Leveraging Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), CAMP-VLM outperforms the best-performing baseline by up to 66.9% in prediction accuracy.
>
---
#### [new 085] FlowDet: Unifying Object Detection and Generative Transport Flows
- **分类: cs.CV**

- **简介: 该论文提出FlowDet，将目标检测建模为条件流匹配下的生成式传输问题，替代DiffusionDet的扩散建模。旨在解决检测中推理步数多、路径复杂、性能增长慢的问题。通过学习更直简的传输路径，提升检测精度与效率，在COCO和LVIS上显著超越DiffusionDet及非生成式基线。**

- **链接: [https://arxiv.org/pdf/2512.16771v1](https://arxiv.org/pdf/2512.16771v1)**

> **作者:** Enis Baty; C. P. Bridges; Simon Hadfield
>
> **摘要:** We present FlowDet, the first formulation of object detection using modern Conditional Flow Matching techniques. This work follows from DiffusionDet, which originally framed detection as a generative denoising problem in the bounding box space via diffusion. We revisit and generalise this formulation to a broader class of generative transport problems, while maintaining the ability to vary the number of boxes and inference steps without re-training. In contrast to the curved stochastic transport paths induced by diffusion, FlowDet learns simpler and straighter paths resulting in faster scaling of detection performance as the number of inference steps grows. We find that this reformulation enables us to outperform diffusion based detection systems (as well as non-generative baselines) across a wide range of experiments, including various precision/recall operating points using multiple feature backbones and datasets. In particular, when evaluating under recall-constrained settings, we can highlight the effects of the generative transport without over-compensating with large numbers of proposals. This provides gains of up to +3.6% AP and +4.2% AP$_{rare}$ over DiffusionDet on the COCO and LVIS datasets, respectively.
>
---
#### [new 086] Prime and Reach: Synthesising Body Motion for Gaze-Primed Object Reach
- **分类: cs.CV**

- **简介: 该论文属人体运动生成任务，旨在解决 gaze-primed 物体抓取动作的合成问题。作者首次构建23.7K条注视引导的伸手运动数据集，基于扩散模型预训练并以目标位姿/位置微调，提出“Prime Success”新指标，验证生成动作在注视引导与实际到达两阶段的真实性。**

- **链接: [https://arxiv.org/pdf/2512.16456v1](https://arxiv.org/pdf/2512.16456v1)**

> **作者:** Masashi Hatano; Saptarshi Sinha; Jacob Chalk; Wei-Hong Li; Hideo Saito; Dima Damen
>
> **备注:** Project Page: https://masashi-hatano.github.io/prime-and-reach/
>
> **摘要:** Human motion generation is a challenging task that aims to create realistic motion imitating natural human behaviour. We focus on the well-studied behaviour of priming an object/location for pick up or put down -- that is, the spotting of an object/location from a distance, known as gaze priming, followed by the motion of approaching and reaching the target location. To that end, we curate, for the first time, 23.7K gaze-primed human motion sequences for reaching target object locations from five publicly available datasets, i.e., HD-EPIC, MoGaze, HOT3D, ADT, and GIMO. We pre-train a text-conditioned diffusion-based motion generation model, then fine-tune it conditioned on goal pose or location, on our curated sequences. Importantly, we evaluate the ability of the generated motion to imitate natural human movement through several metrics, including the 'Reach Success' and a newly introduced 'Prime Success' metric. On the largest dataset, HD-EPIC, our model achieves 60% prime success and 89% reach success when conditioned on the goal object location.
>
---
#### [new 087] Seeing Beyond Words: Self-Supervised Visual Learning for Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属多模态大模型视觉增强任务，旨在解决MLLM因依赖文本监督而视觉推理能力弱的问题。提出JARVIS框架，将I-JEPA自监督学习融入视觉-语言对齐流程，用冻结视觉模型作编码器，LLM早期层作预测器，仅从图像学习结构与语义规律，提升视觉理解能力。**

- **链接: [https://arxiv.org/pdf/2512.15885v1](https://arxiv.org/pdf/2512.15885v1)**

> **作者:** Davide Caffagni; Sara Sarto; Marcella Cornia; Lorenzo Baraldi; Pier Luigi Dovesi; Shaghayegh Roohi; Mark Granroth-Wilding; Rita Cucchiara
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently demonstrated impressive capabilities in connecting vision and language, yet their proficiency in fundamental visual reasoning tasks remains limited. This limitation can be attributed to the fact that MLLMs learn visual understanding primarily from textual descriptions, which constitute a subjective and inherently incomplete supervisory signal. Furthermore, the modest scale of multimodal instruction tuning compared to massive text-only pre-training leads MLLMs to overfit language priors while overlooking visual details. To address these issues, we introduce JARVIS, a JEPA-inspired framework for self-supervised visual enhancement in MLLMs. Specifically, we integrate the I-JEPA learning paradigm into the standard vision-language alignment pipeline of MLLMs training. Our approach leverages frozen vision foundation models as context and target encoders, while training the predictor, implemented as the early layers of an LLM, to learn structural and semantic regularities from images without relying exclusively on language supervision. Extensive experiments on standard MLLM benchmarks show that JARVIS consistently improves performance on vision-centric benchmarks across different LLM families, without degrading multimodal reasoning abilities. Our source code is publicly available at: https://github.com/aimagelab/JARVIS.
>
---
#### [new 088] Eyes on the Grass: Biodiversity-Increasing Robotic Mowing Using Deep Visual Embeddings
- **分类: cs.CV**

- **简介: 该论文属智能农业/生态机器人任务，旨在解决传统草坪单一化导致的生物多样性下降问题。提出基于ResNet50视觉嵌入的机器人割草系统，通过无监督计算图像特征空间离散度评估植被多样性，并动态停启刀片以保留高多样性区域，实现在割草中主动增益生态价值。**

- **链接: [https://arxiv.org/pdf/2512.15993v1](https://arxiv.org/pdf/2512.15993v1)**

> **作者:** Lars Beckers; Arno Waes; Aaron Van Campenhout; Toon Goedemé
>
> **摘要:** This paper presents a robotic mowing framework that actively enhances garden biodiversity through visual perception and adaptive decision-making. Unlike passive rewilding approaches, the proposed system uses deep feature-space analysis to identify and preserve visually diverse vegetation patches in camera images by selectively deactivating the mower blades. A ResNet50 network pretrained on PlantNet300K provides ecologically meaningful embeddings, from which a global deviation metric estimates biodiversity without species-level supervision. These estimates drive a selective mowing algorithm that dynamically alternates between mowing and conservation behavior. The system was implemented on a modified commercial robotic mower and validated both in a controlled mock-up lawn and on real garden datasets. Results demonstrate a strong correlation between embedding-space dispersion and expert biodiversity assessment, confirming the feasibility of deep visual diversity as a proxy for ecological richness and the effectiveness of the proposed mowing decision approach. Widespread adoption of such systems will turn ecologically worthless, monocultural lawns into vibrant, valuable biotopes that boost urban biodiversity.
>
---
#### [new 089] CountZES: Counting via Zero-Shot Exemplar Selection
- **分类: cs.CV**

- **简介: 该论文面向零-shot目标计数（ZOC）任务，解决 unseen 类别仅凭类别名计数难的问题。提出无训练框架CountZES，通过检测锚定、密度引导和特征共识三阶段协同筛选高质量单实例 exemplar，提升计数准确性和跨域泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.16415v1](https://arxiv.org/pdf/2512.16415v1)**

> **作者:** Muhammad Ibraheem Siddiqui; Muhammad Haris Khan
>
> **摘要:** Object counting in complex scenes remains challenging, particularly in the zero-shot setting, where the goal is to count instances of unseen categories specified only by a class name. Existing zero-shot object counting (ZOC) methods that infer exemplars from text either rely on open-vocabulary detectors, which often yield multi-instance candidates, or on random patch sampling, which fails to accurately delineate object instances. To address this, we propose CountZES, a training-free framework for object counting via zero-shot exemplar selection. CountZES progressively discovers diverse exemplars through three synergistic stages: Detection-Anchored Exemplar (DAE), Density-Guided Exemplar (DGE), and Feature-Consensus Exemplar (FCE). DAE refines open-vocabulary detections to isolate precise single-instance exemplars. DGE introduces a density-driven, self-supervised paradigm to identify statistically consistent and semantically compact exemplars, while FCE reinforces visual coherence through feature-space clustering. Together, these stages yield a diverse, complementary exemplar set that balances textual grounding, count consistency, and feature representativeness. Experiments on diverse datasets demonstrate CountZES superior performance among ZOC methods while generalizing effectively across natural, aerial and medical domains.
>
---
#### [new 090] Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文面向扩散Transformer（DiT）的长序列高效训练任务，解决其自注意力计算复杂度高（O(n²)）导致的扩展瓶颈。提出可训练的对数线性稀疏注意力（LLSA），通过分层Top-K选择与层级KV增强，将复杂度降至O(n log n)，并实现高效GPU实现，在高分辨率图像生成中显著加速训练与推理。**

- **链接: [https://arxiv.org/pdf/2512.16615v1](https://arxiv.org/pdf/2512.16615v1)**

> **作者:** Yifan Zhou; Zeqi Xiao; Tianyi Wei; Shuai Yang; Xingang Pan
>
> **备注:** Code is available at: https://github.com/SingleZombie/LLSA
>
> **摘要:** Diffusion Transformers (DiTs) set the state of the art in visual generation, yet their quadratic self-attention cost fundamentally limits scaling to long token sequences. Recent Top-K sparse attention approaches reduce the computation of DiTs by compressing tokens into block-wise representation and selecting a small set of relevant key blocks, but still suffer from (i) quadratic selection cost on compressed tokens and (ii) increasing K required to maintain model quality as sequences grow. We identify that their inefficiency is due to the single-level design, as a single coarse level is insufficient to represent the global structure. In this paper, we introduce Log-linear Sparse Attention (LLSA), a trainable sparse attention mechanism for extremely long token sequences that reduces both selection and attention costs from quadratic to log-linear complexity by utilizing a hierarchical structure. LLSA performs hierarchical Top-K selection, progressively adopting sparse Top-K selection with the indices found at the previous level, and introduces a Hierarchical KV Enrichment mechanism that preserves global context while using fewer tokens of different granularity during attention computation. To support efficient training, we develop a high-performance GPU implementation that uses only sparse indices for both the forward and backward passes, eliminating the need for dense attention masks. We evaluate LLSA on high-resolution pixel-space image generation without using patchification and VAE encoding. LLSA accelerates attention inference by 28.27x and DiT training by 6.09x on 256x256 pixel token sequences, while maintaining generation quality. The results demonstrate that LLSA offers a promising direction for training long-sequence DiTs efficiently. Code is available at: https://github.com/SingleZombie/LLSA
>
---
#### [new 091] AdaTooler-V: Adaptive Tool-Use for Images and Videos
- **分类: cs.CV**

- **简介: 该论文提出AdaTooler-V，解决多模态大模型盲目调用视觉工具导致推理开销大、性能下降的问题。通过AT-GRPO强化学习算法和自建数据集，实现视觉任务中工具使用的自适应决策，在图像/视频推理任务上显著提升准确率与效率。**

- **链接: [https://arxiv.org/pdf/2512.16918v1](https://arxiv.org/pdf/2512.16918v1)**

> **作者:** Chaoyang Wang; Kaituo Feng; Dongyang Chen; Zhongyu Wang; Zhixun Li; Sicheng Gao; Meng Meng; Xu Zhou; Manyuan Zhang; Yuzhang Shang; Xiangyu Yue
>
> **备注:** Project page: https://github.com/CYWang735/AdaTooler-V
>
> **摘要:** Recent advances have shown that multimodal large language models (MLLMs) benefit from multimodal interleaved chain-of-thought (CoT) with vision tool interactions. However, existing open-source models often exhibit blind tool-use reasoning patterns, invoking vision tools even when they are unnecessary, which significantly increases inference overhead and degrades model performance. To this end, we propose AdaTooler-V, an MLLM that performs adaptive tool-use by determining whether a visual problem truly requires tools. First, we introduce AT-GRPO, a reinforcement learning algorithm that adaptively adjusts reward scales based on the Tool Benefit Score of each sample, encouraging the model to invoke tools only when they provide genuine improvements. Moreover, we construct two datasets to support training: AdaTooler-V-CoT-100k for SFT cold start and AdaTooler-V-300k for RL with verifiable rewards across single-image, multi-image, and video data. Experiments across twelve benchmarks demonstrate the strong reasoning capability of AdaTooler-V, outperforming existing methods in diverse visual reasoning tasks. Notably, AdaTooler-V-7B achieves an accuracy of 89.8\% on the high-resolution benchmark V*, surpassing the commercial proprietary model GPT-4o and Gemini 1.5 Pro. All code, models, and data are released.
>
---
#### [new 092] Auto-Vocabulary 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文研究自动词汇3D目标检测（AV3DOD）任务，旨在无需人工指定类别、自动为检测到的3D物体生成语义类名。提出语义得分（SS）评估类名质量，构建基于2D视觉语言模型的框架，实现定位精度与语义质量双SOTA。**

- **链接: [https://arxiv.org/pdf/2512.16077v1](https://arxiv.org/pdf/2512.16077v1)**

> **作者:** Haomeng Zhang; Kuan-Chuan Peng; Suhas Lohit; Raymond A. Yeh
>
> **备注:** technical report
>
> **摘要:** Open-vocabulary 3D object detection methods are able to localize 3D boxes of classes unseen during training. Despite the name, existing methods rely on user-specified classes both at training and inference. We propose to study Auto-Vocabulary 3D Object Detection (AV3DOD), where the classes are automatically generated for the detected objects without any user input. To this end, we introduce Semantic Score (SS) to evaluate the quality of the generated class names. We then develop a novel framework, AV3DOD, which leverages 2D vision-language models (VLMs) to generate rich semantic candidates through image captioning, pseudo 3D box generation, and feature-space semantics expansion. AV3DOD achieves the state-of-the-art (SOTA) performance on both localization (mAP) and semantic quality (SS) on the ScanNetV2 and SUNRGB-D datasets. Notably, it surpasses the SOTA, CoDA, by 3.48 overall mAP and attains a 24.5% relative improvement in SS on ScanNetV2.
>
---
#### [new 093] StageVAR: Stage-Aware Acceleration for Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文面向视觉自回归（VAR）图像生成任务，解决其大规模步数下计算开销大、现有加速方法依赖人工调参且忽视阶段差异的问题。提出StageVAR框架，基于阶段重要性分析，对早期保真、后期轻量优化，实现免训练、插件式加速，达3.4×提速。**

- **链接: [https://arxiv.org/pdf/2512.16483v1](https://arxiv.org/pdf/2512.16483v1)**

> **作者:** Senmao Li; Kai Wang; Salman Khan; Fahad Shahbaz Khan; Jian Yang; Yaxing Wang
>
> **摘要:** Visual Autoregressive (VAR) modeling departs from the next-token prediction paradigm of traditional Autoregressive (AR) models through next-scale prediction, enabling high-quality image generation. However, the VAR paradigm suffers from sharply increased computational complexity and running time at large-scale steps. Although existing acceleration methods reduce runtime for large-scale steps, but rely on manual step selection and overlook the varying importance of different stages in the generation process. To address this challenge, we present StageVAR, a systematic study and stage-aware acceleration framework for VAR models. Our analysis shows that early steps are critical for preserving semantic and structural consistency and should remain intact, while later steps mainly refine details and can be pruned or approximated for acceleration. Building on these insights, StageVAR introduces a plug-and-play acceleration strategy that exploits semantic irrelevance and low-rank properties in late-stage computations, without requiring additional training. Our proposed StageVAR achieves up to 3.4x speedup with only a 0.01 drop on GenEval and a 0.26 decrease on DPG, consistently outperforming existing acceleration baselines. These results highlight stage-aware design as a powerful principle for efficient visual autoregressive image generation.
>
---
#### [new 094] Next-Embedding Prediction Makes Strong Vision Learners
- **分类: cs.CV**

- **简介: 该论文提出Next-Embedding Predictive Autoregression（NEPA），属视觉自监督学习任务，旨在解决无需像素重建或对比损失的高效预训练问题。工作是让Transformer直接预测未来图像块嵌入，仅用因果掩码和停梯度，在ImageNet上预训练后取得优异迁移性能。**

- **链接: [https://arxiv.org/pdf/2512.16922v1](https://arxiv.org/pdf/2512.16922v1)**

> **作者:** Sihan Xu; Ziqiao Ma; Wenhao Chai; Xuweiyi Chen; Weiyang Jin; Joyce Chai; Saining Xie; Stella X. Yu
>
> **备注:** Project Page: https://sihanxu.me/nepa
>
> **摘要:** Inspired by the success of generative pretraining in natural language, we ask whether the same principles can yield strong self-supervised visual learners. Instead of training models to output features for downstream use, we train them to generate embeddings to perform predictive tasks directly. This work explores such a shift from learning representations to learning models. Specifically, models learn to predict future patch embeddings conditioned on past ones, using causal masking and stop gradient, which we refer to as Next-Embedding Predictive Autoregression (NEPA). We demonstrate that a simple Transformer pretrained on ImageNet-1k with next embedding prediction as its sole learning objective is effective - no pixel reconstruction, discrete tokens, contrastive loss, or task-specific heads. This formulation retains architectural simplicity and scalability, without requiring additional design complexity. NEPA achieves strong results across tasks, attaining 83.8% and 85.3% top-1 accuracy on ImageNet-1K with ViT-B and ViT-L backbones after fine-tuning, and transferring effectively to semantic segmentation on ADE20K. We believe generative pretraining from embeddings provides a simple, scalable, and potentially modality-agnostic alternative to visual self-supervised learning.
>
---
#### [new 095] EasyV2V: A High-quality Instruction-based Video Editing Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EasyV2V，一种基于指令的高质量视频编辑框架，旨在解决视频编辑中一致性差、控制难、泛化弱的问题。通过创新数据构建（伪对、运动建模、过渡监督）、简化架构（文本-视频模型+LoRA微调）和统一掩码控制，实现灵活输入下的SOTA效果。**

- **链接: [https://arxiv.org/pdf/2512.16920v1](https://arxiv.org/pdf/2512.16920v1)**

> **作者:** Jinjie Mai; Chaoyang Wang; Guocheng Gordon Qian; Willi Menapace; Sergey Tulyakov; Bernard Ghanem; Peter Wonka; Ashkan Mirzaei
>
> **备注:** Project page: https://snap-research.github.io/easyv2v/
>
> **摘要:** While image editing has advanced rapidly, video editing remains less explored, facing challenges in consistency, control, and generalization. We study the design space of data, architecture, and control, and introduce \emph{EasyV2V}, a simple and effective framework for instruction-based video editing. On the data side, we compose existing experts with fast inverses to build diverse video pairs, lift image edit pairs into videos via single-frame supervision and pseudo pairs with shared affine motion, mine dense-captioned clips for video pairs, and add transition supervision to teach how edits unfold. On the model side, we observe that pretrained text-to-video models possess editing capability, motivating a simplified design. Simple sequence concatenation for conditioning with light LoRA fine-tuning suffices to train a strong model. For control, we unify spatiotemporal control via a single mask mechanism and support optional reference images. Overall, EasyV2V works with flexible inputs, e.g., video+text, video+mask+text, video+mask+reference+text, and achieves state-of-the-art video editing results, surpassing concurrent and commercial systems. Project page: https://snap-research.github.io/easyv2v/
>
---
#### [new 096] GMODiff: One-Step Gain Map Refinement with Diffusion Priors for HDR Reconstruction
- **分类: cs.CV**

- **简介: 该论文属多曝光HDR重建任务，旨在解决LDM直接用于HDR时的动态范围受限、推理慢及内容幻觉问题。提出GMODiff：以增益图（GM）估计替代全HDR生成，用回归先验初始化单步扩散，兼顾保真度与感知质量，速度提升100倍。**

- **链接: [https://arxiv.org/pdf/2512.16357v1](https://arxiv.org/pdf/2512.16357v1)**

> **作者:** Tao Hu; Weiyu Zhou; Yanjie Tu; Peng Wu; Wei Dong; Qingsen Yan; Yanning Zhang
>
> **摘要:** Pre-trained Latent Diffusion Models (LDMs) have recently shown strong perceptual priors for low-level vision tasks, making them a promising direction for multi-exposure High Dynamic Range (HDR) reconstruction. However, directly applying LDMs to HDR remains challenging due to: (1) limited dynamic-range representation caused by 8-bit latent compression, (2) high inference cost from multi-step denoising, and (3) content hallucination inherent to generative nature. To address these challenges, we introduce GMODiff, a gain map-driven one-step diffusion framework for multi-exposure HDR reconstruction. Instead of reconstructing full HDR content, we reformulate HDR reconstruction as a conditionally guided Gain Map (GM) estimation task, where the GM encodes the extended dynamic range while retaining the same bit depth as LDR images. We initialize the denoising process from an informative regression-based estimate rather than pure noise, enabling the model to generate high-quality GMs in a single denoising step. Furthermore, recognizing that regression-based models excel in content fidelity while LDMs favor perceptual quality, we leverage regression priors to guide both the denoising process and latent decoding of the LDM, suppressing hallucinations while preserving structural accuracy. Extensive experiments demonstrate that our GMODiff performs favorably against several state-of-the-art methods and is 100 faster than previous LDM-based methods.
>
---
#### [new 097] OPENTOUCH: Bringing Full-Hand Touch to Real-World Interaction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属多模态感知任务，旨在解决真实场景中第一人称视频与全手触觉信号缺乏对齐的问题。作者构建首个野外全手触觉数据集OpenTouch（5.1小时同步视频-触觉-姿态数据+2900标注片段），并提出检索与分类基准，验证触觉对抓取理解、跨模态对齐和视频驱动触觉检索的有效性。**

- **链接: [https://arxiv.org/pdf/2512.16842v1](https://arxiv.org/pdf/2512.16842v1)**

> **作者:** Yuxin Ray Song; Jinzhou Li; Rao Fu; Devin Murphy; Kaichen Zhou; Rishi Shiv; Yaqi Li; Haoyu Xiong; Crystal Elaine Owens; Yilun Du; Yiyue Luo; Xianyi Cheng; Antonio Torralba; Wojciech Matusik; Paul Pu Liang
>
> **备注:** https://opentouch-tactile.github.io/
>
> **摘要:** The human hand is our primary interface to the physical world, yet egocentric perception rarely knows when, where, or how forcefully it makes contact. Robust wearable tactile sensors are scarce, and no existing in-the-wild datasets align first-person video with full-hand touch. To bridge the gap between visual perception and physical interaction, we present OpenTouch, the first in-the-wild egocentric full-hand tactile dataset, containing 5.1 hours of synchronized video-touch-pose data and 2,900 curated clips with detailed text annotations. Using OpenTouch, we introduce retrieval and classification benchmarks that probe how touch grounds perception and action. We show that tactile signals provide a compact yet powerful cue for grasp understanding, strengthen cross-modal alignment, and can be reliably retrieved from in-the-wild video queries. By releasing this annotated vision-touch-pose dataset and benchmark, we aim to advance multimodal egocentric perception, embodied learning, and contact-rich robotic manipulation.
>
---
#### [new 098] Image Compression Using Singular Value Decomposition
- **分类: cs.CV**

- **简介: 该论文研究基于奇异值分解（SVD）的图像压缩任务，旨在解决高效图像压缩问题。工作包括：应用SVD低秩近似压缩灰度与多通道图像，以相对Frobenius误差和压缩比评估性能，并与JPEG、JPEG2000、WEBP等标准格式对比。结果表明SVD压缩效率更低，不具实用竞争力。**

- **链接: [https://arxiv.org/pdf/2512.16226v1](https://arxiv.org/pdf/2512.16226v1)**

> **作者:** Justin Jiang
>
> **摘要:** Images are a substantial portion of the internet, making efficient compression important for reducing storage and bandwidth demands. This study investigates the use of Singular Value Decomposition and low-rank matrix approximations for image compression, evaluating performance using relative Frobenius error and compression ratio. The approach is applied to both grayscale and multichannel images to assess its generality. Results show that the low-rank approximations often produce images that appear visually similar to the originals, but the compression efficiency remains consistently worse than established formats such as JPEG, JPEG2000, and WEBP at comparable error levels. At low tolerated error levels, the compressed representation produced by Singular Value Decomposition can even exceed the size of the original image, indicating that this method is not competitive with industry-standard codecs for practical image compression.
>
---
#### [new 099] FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出FrameDiffuser，面向神经前向帧渲染任务，解决G-buffer到图像合成中实时性、时序一致性与计算效率的矛盾。通过G-buffer与上一帧双条件自回归生成，结合ControlNet与ControlLoRA，并采用环境特化三阶段训练，实现高质量、稳定、低延迟的交互式渲染。**

- **链接: [https://arxiv.org/pdf/2512.16670v1](https://arxiv.org/pdf/2512.16670v1)**

> **作者:** Ole Beisswenger; Jan-Niklas Dihlmann; Hendrik P. A. Lensch
>
> **备注:** Project Page: https://framediffuser.jdihlmann.com/
>
> **摘要:** Neural rendering for interactive applications requires translating geometric and material properties (G-buffer) to photorealistic images with realistic lighting on a frame-by-frame basis. While recent diffusion-based approaches show promise for G-buffer-conditioned image synthesis, they face critical limitations: single-image models like RGBX generate frames independently without temporal consistency, while video models like DiffusionRenderer are too computationally expensive for most consumer gaming sets ups and require complete sequences upfront, making them unsuitable for interactive applications where future frames depend on user input. We introduce FrameDiffuser, an autoregressive neural rendering framework that generates temporally consistent, photorealistic frames by conditioning on G-buffer data and the models own previous output. After an initial frame, FrameDiffuser operates purely on incoming G-buffer data, comprising geometry, materials, and surface properties, while using its previously generated frame for temporal guidance, maintaining stable, temporal consistent generation over hundreds to thousands of frames. Our dual-conditioning architecture combines ControlNet for structural guidance with ControlLoRA for temporal coherence. A three-stage training strategy enables stable autoregressive generation. We specialize our model to individual environments, prioritizing consistency and inference speed over broad generalization, demonstrating that environment-specific training achieves superior photorealistic quality with accurate lighting, shadows, and reflections compared to generalized approaches.
>
---
#### [new 100] Avatar4D: Synthesizing Domain-Specific 4D Humans for Real-World Pose Estimation
- **分类: cs.CV**

- **简介: 论文提出Avatar4D，属合成数据生成任务，旨在解决域特定（如体育）真实场景下人体姿态估计缺乏高质量标注数据的问题。工作包括构建可定制的4D人体合成管线、发布Syn2Sport体育合成数据集，并验证其在监督训练、零样本迁移与跨体育泛化中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.16199v1](https://arxiv.org/pdf/2512.16199v1)**

> **作者:** Jerrin Bright; Zhibo Wang; Dmytro Klepachevskyi; Yuhao Chen; Sirisha Rambhatla; David Clausi; John Zelek
>
> **摘要:** We present Avatar4D, a real-world transferable pipeline for generating customizable synthetic human motion datasets tailored to domain-specific applications. Unlike prior works, which focus on general, everyday motions and offer limited flexibility, our approach provides fine-grained control over body pose, appearance, camera viewpoint, and environmental context, without requiring any manual annotations. To validate the impact of Avatar4D, we focus on sports, where domain-specific human actions and movement patterns pose unique challenges for motion understanding. In this setting, we introduce Syn2Sport, a large-scale synthetic dataset spanning sports, including baseball and ice hockey. Avatar4D features high-fidelity 4D (3D geometry over time) human motion sequences with varying player appearances rendered in diverse environments. We benchmark several state-of-the-art pose estimation models on Syn2Sport and demonstrate their effectiveness for supervised learning, zero-shot transfer to real-world data, and generalization across sports. Furthermore, we evaluate how closely the generated synthetic data aligns with real-world datasets in feature space. Our results highlight the potential of such systems to generate scalable, controllable, and transferable human datasets for diverse domain-specific tasks without relying on domain-specific real data.
>
---
#### [new 101] DeContext as Defense: Safe Image Editing in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属图像安全防御任务，旨在防止扩散Transformer模型对个人图像的未授权编辑。作者提出DeContext方法，通过在关键多模态注意力层注入微小扰动，阻断上下文信息传播，从而有效抵御恶意编辑，同时保持图像质量。**

- **链接: [https://arxiv.org/pdf/2512.16625v1](https://arxiv.org/pdf/2512.16625v1)**

> **作者:** Linghui Shen; Mingyue Cui; Xingyi Yang
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** In-context diffusion models allow users to modify images with remarkable ease and realism. However, the same power raises serious privacy concerns: personal images can be easily manipulated for identity impersonation, misinformation, or other malicious uses, all without the owner's consent. While prior work has explored input perturbations to protect against misuse in personalized text-to-image generation, the robustness of modern, large-scale in-context DiT-based models remains largely unexamined. In this paper, we propose DeContext, a new method to safeguard input images from unauthorized in-context editing. Our key insight is that contextual information from the source image propagates to the output primarily through multimodal attention layers. By injecting small, targeted perturbations that weaken these cross-attention pathways, DeContext breaks this flow, effectively decouples the link between input and output. This simple defense is both efficient and robust. We further show that early denoising steps and specific transformer blocks dominate context propagation, which allows us to concentrate perturbations where they matter most. Experiments on Flux Kontext and Step1X-Edit show that DeContext consistently blocks unwanted image edits while preserving visual quality. These results highlight the effectiveness of attention-based perturbations as a powerful defense against image manipulation.
>
---
#### [new 102] Semi-Supervised Multi-View Crowd Counting by Ranking Multi-View Fusion Models
- **分类: cs.CV**

- **简介: 该论文属半监督多视角人群计数任务，旨在缓解多视角标注数据稀缺问题。提出两种基于模型排序的半监督方法：一是约束少视角预测值不大于多视角预测值；二是约束多视角模型不确定性不高于少视角。通过引入排序约束提升有限标注下的计数性能。**

- **链接: [https://arxiv.org/pdf/2512.16243v1](https://arxiv.org/pdf/2512.16243v1)**

> **作者:** Qi Zhang; Yunfei Gong; Zhidan Xie; Zhizi Wang; Antoni B. Chan; Hui Huang
>
> **备注:** 13 pages, 7 figures, under review
>
> **摘要:** Multi-view crowd counting has been proposed to deal with the severe occlusion issue of crowd counting in large and wide scenes. However, due to the difficulty of collecting and annotating multi-view images, the datasets for multi-view counting have a limited number of multi-view frames and scenes. To solve the problem of limited data, one approach is to collect synthetic data to bypass the annotating step, while another is to propose semi- or weakly-supervised or unsupervised methods that demand less multi-view data. In this paper, we propose two semi-supervised multi-view crowd counting frameworks by ranking the multi-view fusion models of different numbers of input views, in terms of the model predictions or the model uncertainties. Specifically, for the first method (vanilla model), we rank the multi-view fusion models' prediction results of different numbers of camera-view inputs, namely, the model's predictions with fewer camera views shall not be larger than the predictions with more camera views. For the second method, we rank the estimated model uncertainties of the multi-view fusion models with a variable number of view inputs, guided by the multi-view fusion models' prediction errors, namely, the model uncertainties with more camera views shall not be larger than those with fewer camera views. These constraints are introduced into the model training in a semi-supervised fashion for multi-view counting with limited labeled data. The experiments demonstrate the advantages of the proposed multi-view model ranking methods compared with other semi-supervised counting methods.
>
---
#### [new 103] Make-It-Poseable: Feed-forward Latent Posing Model for 3D Humanoid Character Animation
- **分类: cs.CV**

- **简介: 该论文属3D人体角色动画任务，旨在解决传统 posing 方法中蒙皮权重不准、拓扑缺陷和姿态失配等问题。提出“Make-It-Poseable”前馈潜空间建模框架，用姿态驱动的潜空间变换替代顶点变形，引入潜空间监督与自适应补全模块，提升姿态保真度与泛化性。**

- **链接: [https://arxiv.org/pdf/2512.16767v1](https://arxiv.org/pdf/2512.16767v1)**

> **作者:** Zhiyang Guo; Ori Zhang; Jax Xiang; Alan Zhao; Wengang Zhou; Houqiang Li
>
> **备注:** Project page: https://jasongzy.github.io/Make-It-Poseable/
>
> **摘要:** Posing 3D characters is a fundamental task in computer graphics and vision. However, existing methods like auto-rigging and pose-conditioned generation often struggle with challenges such as inaccurate skinning weight prediction, topological imperfections, and poor pose conformance, limiting their robustness and generalizability. To overcome these limitations, we introduce Make-It-Poseable, a novel feed-forward framework that reformulates character posing as a latent-space transformation problem. Instead of deforming mesh vertices as in traditional pipelines, our method reconstructs the character in new poses by directly manipulating its latent representation. At the core of our method is a latent posing transformer that manipulates shape tokens based on skeletal motion. This process is facilitated by a dense pose representation for precise control. To ensure high-fidelity geometry and accommodate topological changes, we also introduce a latent-space supervision strategy and an adaptive completion module. Our method demonstrates superior performance in posing quality. It also naturally extends to 3D editing applications like part replacement and refinement.
>
---
#### [new 104] RePlan: Reasoning-guided Region Planning for Complex Instruction-based Image Editing
- **分类: cs.CV**

- **简介: 该论文面向指令驱动的图像编辑任务，解决复杂指令与杂乱/模糊图像导致的编辑失败问题。提出RePlan框架：先用视觉语言规划器分步推理并定位目标区域，再通过无训练注意力注入机制实现精准多区域并行编辑，并引入强化学习优化规划，显著提升区域精度与编辑保真度。**

- **链接: [https://arxiv.org/pdf/2512.16864v1](https://arxiv.org/pdf/2512.16864v1)**

> **作者:** Tianyuan Qu; Lei Ke; Xiaohang Zhan; Longxiang Tang; Yuqi Liu; Bohao Peng; Bei Yu; Dong Yu; Jiaya Jia
>
> **备注:** Precise region control and planning for instruction-based image editing. Our project page: https://replan-iv-edit.github.io
>
> **摘要:** Instruction-based image editing enables natural-language control over visual modifications, yet existing models falter under Instruction-Visual Complexity (IV-Complexity), where intricate instructions meet cluttered or ambiguous scenes. We introduce RePlan (Region-aligned Planning), a plan-then-execute framework that couples a vision-language planner with a diffusion editor. The planner decomposes instructions via step-by-step reasoning and explicitly grounds them to target regions; the editor then applies changes using a training-free attention-region injection mechanism, enabling precise, parallel multi-region edits without iterative inpainting. To strengthen planning, we apply GRPO-based reinforcement learning using 1K instruction-only examples, yielding substantial gains in reasoning fidelity and format reliability. We further present IV-Edit, a benchmark focused on fine-grained grounding and knowledge-intensive edits. Across IV-Complex settings, RePlan consistently outperforms strong baselines trained on far larger datasets, improving regional precision and overall fidelity. Our project page: https://replan-iv-edit.github.io
>
---
#### [new 105] TTP: Test-Time Padding for Adversarial Detection and Robust Adaptation on Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（如CLIP）易受对抗攻击、现有防御方法难以兼顾鲁棒性与干净准确率的问题，提出测试时填充（TTP）框架：通过填充前后特征余弦相似度变化检测对抗样本，并对恶意样本自适应调整填充与集成预测，实现轻量、通用、无损的测试时防御。**

- **链接: [https://arxiv.org/pdf/2512.16523v1](https://arxiv.org/pdf/2512.16523v1)**

> **作者:** Zhiwei Li; Yitian Pang; Weining Wang; Zhenan Sun; Qi Li
>
> **摘要:** Vision-Language Models (VLMs), such as CLIP, have achieved impressive zero-shot recognition performance but remain highly susceptible to adversarial perturbations, posing significant risks in safety-critical scenarios. Previous training-time defenses rely on adversarial fine-tuning, which requires labeled data and costly retraining, while existing test-time strategies fail to reliably distinguish between clean and adversarial inputs, thereby preventing both adversarial robustness and clean accuracy from reaching their optimum. To address these limitations, we propose Test-Time Padding (TTP), a lightweight defense framework that performs adversarial detection followed by targeted adaptation at inference. TTP identifies adversarial inputs via the cosine similarity shift between CLIP feature embeddings computed before and after spatial padding, yielding a universal threshold for reliable detection across architectures and datasets. For detected adversarial cases, TTP employs trainable padding to restore disrupted attention patterns, coupled with a similarity-aware ensemble strategy for a more robust final prediction. For clean inputs, TTP leaves them unchanged by default or optionally integrates existing test-time adaptation techniques for further accuracy gains. Comprehensive experiments on diverse CLIP backbones and fine-grained benchmarks show that TTP consistently surpasses state-of-the-art test-time defenses, delivering substantial improvements in adversarial robustness without compromising clean accuracy. The code for this paper will be released soon.
>
---
#### [new 106] DenseBEV: Transforming BEV Grid Cells into 3D Objects
- **分类: cs.CV**

- **简介: 该论文面向多相机3D目标检测任务，旨在解决BEV Transformer中锚点设计低效、查询冗余及小物体检测弱的问题。提出DenseBEV方法：以BEV特征网格单元为密集锚点，设计两阶段锚生成与BEV-NMS优化，并融合时序检测先验，显著提升nuScenes和Waymo上小物体（如行人）检测精度。**

- **链接: [https://arxiv.org/pdf/2512.16818v1](https://arxiv.org/pdf/2512.16818v1)**

> **作者:** Marius Dähling; Sebastian Krebs; J. Marius Zöllner
>
> **备注:** 15 pages, 8 figures, accepted by WACV 2026
>
> **摘要:** In current research, Bird's-Eye-View (BEV)-based transformers are increasingly utilized for multi-camera 3D object detection. Traditional models often employ random queries as anchors, optimizing them successively. Recent advancements complement or replace these random queries with detections from auxiliary networks. We propose a more intuitive and efficient approach by using BEV feature cells directly as anchors. This end-to-end approach leverages the dense grid of BEV queries, considering each cell as a potential object for the final detection task. As a result, we introduce a novel two-stage anchor generation method specifically designed for multi-camera 3D object detection. To address the scaling issues of attention with a large number of queries, we apply BEV-based Non-Maximum Suppression, allowing gradients to flow only through non-suppressed objects. This ensures efficient training without the need for post-processing. By using BEV features from encoders such as BEVFormer directly as object queries, temporal BEV information is inherently embedded. Building on the temporal BEV information already embedded in our object queries, we introduce a hybrid temporal modeling approach by integrating prior detections to further enhance detection performance. Evaluating our method on the nuScenes dataset shows consistent and significant improvements in NDS and mAP over the baseline, even with sparser BEV grids and therefore fewer initial anchors. It is particularly effective for small objects, enhancing pedestrian detection with a 3.8% mAP increase on nuScenes and an 8% increase in LET-mAP on Waymo. Applying our method, named DenseBEV, to the challenging Waymo Open dataset yields state-of-the-art performance, achieving a LET-mAP of 60.7%, surpassing the previous best by 5.4%. Code is available at https://github.com/mdaehl/DenseBEV.
>
---
#### [new 107] FlashPortrait: 6x Faster Infinite Portrait Animation with Adaptive Latent Prediction
- **分类: cs.CV**

- **简介: 该论文属视频生成任务，旨在解决扩散模型生成长肖像动画时ID不一致与推理慢的问题。提出FlashPortrait：用表情特征提取、归一化表达块增强ID一致性；设计动态滑窗与高阶隐式导数预测，跳步降噪，实现6倍加速。**

- **链接: [https://arxiv.org/pdf/2512.16900v1](https://arxiv.org/pdf/2512.16900v1)**

> **作者:** Shuyuan Tu; Yueming Pan; Yinming Huang; Xintong Han; Zhen Xing; Qi Dai; Kai Qiu; Chong Luo; Zuxuan Wu
>
> **摘要:** Current diffusion-based acceleration methods for long-portrait animation struggle to ensure identity (ID) consistency. This paper presents FlashPortrait, an end-to-end video diffusion transformer capable of synthesizing ID-preserving, infinite-length videos while achieving up to 6x acceleration in inference speed. In particular, FlashPortrait begins by computing the identity-agnostic facial expression features with an off-the-shelf extractor. It then introduces a Normalized Facial Expression Block to align facial features with diffusion latents by normalizing them with their respective means and variances, thereby improving identity stability in facial modeling. During inference, FlashPortrait adopts a dynamic sliding-window scheme with weighted blending in overlapping areas, ensuring smooth transitions and ID consistency in long animations. In each context window, based on the latent variation rate at particular timesteps and the derivative magnitude ratio among diffusion layers, FlashPortrait utilizes higher-order latent derivatives at the current timestep to directly predict latents at future timesteps, thereby skipping several denoising steps and achieving 6x speed acceleration. Experiments on benchmarks show the effectiveness of FlashPortrait both qualitatively and quantitatively.
>
---
#### [new 108] TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦图像中文本编辑任务，旨在解决现有模型在语义一致性、物理合理性和上下文连贯性方面的不足。作者提出TextEditBench基准，强调推理感知的文本编辑，并引入新指标“语义期望（SE）”评估跨模态推理能力。**

- **链接: [https://arxiv.org/pdf/2512.16270v1](https://arxiv.org/pdf/2512.16270v1)**

> **作者:** Rui Gui; Yang Wan; Haochen Han; Dongxing Mao; Fangming Liu; Min Li; Alex Jinpeng Wang
>
> **摘要:** Text rendering has recently emerged as one of the most challenging frontiers in visual generation, drawing significant attention from large-scale diffusion and multimodal models. However, text editing within images remains largely unexplored, as it requires generating legible characters while preserving semantic, geometric, and contextual coherence. To fill this gap, we introduce TextEditBench, a comprehensive evaluation benchmark that explicitly focuses on text-centric regions in images. Beyond basic pixel manipulations, our benchmark emphasizes reasoning-intensive editing scenarios that require models to understand physical plausibility, linguistic meaning, and cross-modal dependencies. We further propose a novel evaluation dimension, Semantic Expectation (SE), which measures reasoning ability of model to maintain semantic consistency, contextual coherence, and cross-modal alignment during text editing. Extensive experiments on state-of-the-art editing systems reveal that while current models can follow simple textual instructions, they still struggle with context-dependent reasoning, physical consistency, and layout-aware integration. By focusing evaluation on this long-overlooked yet fundamental capability, TextEditBench establishes a new testing ground for advancing text-guided image editing and reasoning in multimodal generation.
>
---
#### [new 109] SARMAE: Masked Autoencoder for SAR Representation Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属自监督表示学习任务，旨在解决SAR图像因数据稀缺和固有斑点噪声导致的语义表征能力弱问题。提出SARMAE框架：构建百万级SAR-1M数据集；设计斑点感知增强模块（SARE）；引入光学先验引导的语义锚定约束（SARC）。**

- **链接: [https://arxiv.org/pdf/2512.16635v1](https://arxiv.org/pdf/2512.16635v1)**

> **作者:** Danxu Liu; Di Wang; Hebaixu Wang; Haoyang Chen; Wentao Jiang; Yilin Cheng; Haonan Guo; Wei Cui; Jing Zhang
>
> **备注:** Code and models will be available at https://github.com/MiliLab/SARMAE
>
> **摘要:** Synthetic Aperture Radar (SAR) imagery plays a critical role in all-weather, day-and-night remote sensing applications. However, existing SAR-oriented deep learning is constrained by data scarcity, while the physically grounded speckle noise in SAR imagery further hampers fine-grained semantic representation learning. To address these challenges, we propose SARMAE, a Noise-Aware Masked Autoencoder for self-supervised SAR representation learning. Specifically, we construct SAR-1M, the first million-scale SAR dataset, with additional paired optical images, to enable large-scale pre-training. Building upon this, we design Speckle-Aware Representation Enhancement (SARE), which injects SAR-specific speckle noise into masked autoencoders to facilitate noise-aware and robust representation learning. Furthermore, we introduce Semantic Anchor Representation Constraint (SARC), which leverages paired optical priors to align SAR features and ensure semantic consistency. Extensive experiments across multiple SAR datasets demonstrate that SARMAE achieves state-of-the-art performance on classification, detection, and segmentation tasks. Code and models will be available at https://github.com/MiliLab/SARMAE.
>
---
#### [new 110] Sketch-in-Latents: Eliciting Unified Reasoning in MLLMs
- **分类: cs.CV**

- **简介: 该论文提出Sketch-in-Latents（SkiLa），旨在解决MLLMs缺乏视觉想象力的问题。它让模型在统一隐空间中交替生成文本token和连续视觉隐式草图token，实现原生多模态推理，无需外部工具或图像生成。**

- **链接: [https://arxiv.org/pdf/2512.16584v1](https://arxiv.org/pdf/2512.16584v1)**

> **作者:** Jintao Tong; Jiaqi Gu; Yujing Lou; Lubin Fan; Yixiong Zou; Yue Wu; Jieping Ye; Ruixuan Li
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at visual understanding tasks through text reasoning, they often fall short in scenarios requiring visual imagination. Unlike current works that take predefined external toolkits or generate images during thinking, however, humans can form flexible visual-text imagination and interactions during thinking without predefined toolkits, where one important reason is that humans construct the visual-text thinking process in a unified space inside the brain. Inspired by this capability, given that current MLLMs already encode visual and text information in the same feature space, we hold that visual tokens can be seamlessly inserted into the reasoning process carried by text tokens, where ideally, all visual imagination processes can be encoded by the latent features. To achieve this goal, we propose Sketch-in-Latents (SkiLa), a novel paradigm for unified multi-modal reasoning that expands the auto-regressive capabilities of MLLMs to natively generate continuous visual embeddings, termed latent sketch tokens, as visual thoughts. During multi-step reasoning, the model dynamically alternates between textual thinking mode for generating textual think tokens and visual sketching mode for generating latent sketch tokens. A latent visual semantics reconstruction mechanism is proposed to ensure these latent sketch tokens are semantically grounded. Extensive experiments demonstrate that SkiLa achieves superior performance on vision-centric tasks while exhibiting strong generalization to diverse general multi-modal benchmarks. Codes will be released at https://github.com/TungChintao/SkiLa.
>
---
#### [new 111] REGLUE Your Latents with Global and Local Semantics for Entangled Diffusion
- **分类: cs.CV**

- **简介: 该论文属图像生成任务，旨在解决扩散模型语义监督弱、训练慢、质量受限问题。提出REGLUE框架，统一建模VAE隐变量、VFM局部语义（压缩多层空间特征）与全局[CLS]标记，并引入外部对齐损失，提升生成质量与收敛速度。**

- **链接: [https://arxiv.org/pdf/2512.16636v1](https://arxiv.org/pdf/2512.16636v1)**

> **作者:** Giorgos Petsangourakis; Christos Sgouropoulos; Bill Psomas; Theodoros Giannakopoulos; Giorgos Sfikas; Ioannis Kakogeorgiou
>
> **摘要:** Latent diffusion models (LDMs) achieve state-of-the-art image synthesis, yet their reconstruction-style denoising objective provides only indirect semantic supervision: high-level semantics emerge slowly, requiring longer training and limiting sample quality. Recent works inject semantics from Vision Foundation Models (VFMs) either externally via representation alignment or internally by jointly modeling only a narrow slice of VFM features inside the diffusion process, under-utilizing the rich, nonlinear, multi-layer spatial semantics available. We introduce REGLUE (Representation Entanglement with Global-Local Unified Encoding), a unified latent diffusion framework that jointly models (i) VAE image latents, (ii) compact local (patch-level) VFM semantics, and (iii) a global (image-level) [CLS] token within a single SiT backbone. A lightweight convolutional semantic compressor nonlinearly aggregates multi-layer VFM features into a low-dimensional, spatially structured representation, which is entangled with the VAE latents in the diffusion process. An external alignment loss further regularizes internal representations toward frozen VFM targets. On ImageNet 256x256, REGLUE consistently improves FID and accelerates convergence over SiT-B/2 and SiT-XL/2 baselines, as well as over REPA, ReDi, and REG. Extensive experiments show that (a) spatial VFM semantics are crucial, (b) non-linear compression is key to unlocking their full benefit, and (c) global tokens and external alignment act as complementary, lightweight enhancements within our global-local-latent joint modeling framework. The code is available at https://github.com/giorgospets/reglue .
>
---
#### [new 112] The Perceptual Observatory Characterizing Robustness and Grounding in MLLMs
- **分类: cs.CV**

- **简介: 该论文属多模态模型评估任务，旨在解决MLLM视觉感知能力缺乏系统刻画的问题。作者提出“感知观测站”框架，通过可控扰动下的细粒度视觉任务（如人脸匹配、属性定位等），评估模型的鲁棒性与真实视觉接地能力，超越传统端到端准确率评测。**

- **链接: [https://arxiv.org/pdf/2512.15949v1](https://arxiv.org/pdf/2512.15949v1)**

> **作者:** Tejas Anvekar; Fenil Bardoliya; Pavan K. Turaga; Chitta Baral; Vivek Gupta
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have yielded increasingly powerful models, yet their perceptual capacities remain poorly characterized. In practice, most model families scale language component while reusing nearly identical vision encoders (e.g., Qwen2.5-VL 3B/7B/72B), which raises pivotal concerns about whether progress reflects genuine visual grounding or reliance on internet-scale textual world knowledge. Existing evaluation methods emphasize end-task accuracy, overlooking robustness, attribution fidelity, and reasoning under controlled perturbations. We present The Perceptual Observatory, a framework that characterizes MLLMs across verticals like: (i) simple vision tasks, such as face matching and text-in-vision comprehension capabilities; (ii) local-to-global understanding, encompassing image matching, grid pointing game, and attribute localization, which tests general visual grounding. Each vertical is instantiated with ground-truth datasets of faces and words, systematically perturbed through pixel-based augmentations and diffusion-based stylized illusions. The Perceptual Observatory moves beyond leaderboard accuracy to yield insights into how MLLMs preserve perceptual grounding and relational structure under perturbations, providing a principled foundation for analyzing strengths and weaknesses of current and future models.
>
---
#### [new 113] Machine Learning Enabled Graph Analysis of Particulate Composites: Application to Solid-state Battery Cathodes
- **分类: cond-mat.mtrl-sci; cs.CV**

- **简介: 该论文属材料信息学任务，旨在解决高通量X射线图像中多相颗粒复合材料微结构解析难、物性关联弱的问题。作者构建了机器学习驱动的图分析框架，将实验图像自动转化为拓扑感知图，揭示固态电池正极中三相界面与双导电通道对电化学活性的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.16085v1](https://arxiv.org/pdf/2512.16085v1)**

> **作者:** Zebin Li; Shimao Deng; Yijin Liu; Jia-Mian Hu
>
> **摘要:** Particulate composites underpin many solid-state chemical and electrochemical systems, where microstructural features such as multiphase boundaries and inter-particle connections strongly influence system performance. Advances in X-ray microscopy enable capturing large-scale, multimodal images of these complex microstructures with an unprecedentedly high throughput. However, harnessing these datasets to discover new physical insights and guide microstructure optimization remains a major challenge. Here, we develop a machine learning (ML) enabled framework that enables automated transformation of experimental multimodal X-ray images of multiphase particulate composites into scalable, topology-aware graphs for extracting physical insights and establishing local microstructure-property relationships at both the particle and network level. Using the multiphase particulate cathode of solid-state lithium batteries as an example, our ML-enabled graph analysis corroborates the critical role of triple phase junctions and concurrent ion/electron conduction channels in realizing desirable local electrochemical activity. Our work establishes graph-based microstructure representation as a powerful paradigm for bridging multimodal experimental imaging and functional understanding, and facilitating microstructure-aware data-driven materials design in a broad range of particulate composites.
>
---
#### [new 114] Large Video Planner Enables Generalizable Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出“大视频规划器”，旨在解决通用机器人跨任务、跨环境泛化控制问题。它摒弃传统视觉-语言-动作范式，首次基于互联网规模人类活动视频数据，训练开源视频基础模型，实现零样本视频规划，并转化为可执行动作，在真实机器人上验证了泛化性与可行性。**

- **链接: [https://arxiv.org/pdf/2512.15840v1](https://arxiv.org/pdf/2512.15840v1)**

> **作者:** Boyuan Chen; Tianyuan Zhang; Haoran Geng; Kiwhan Song; Caiyi Zhang; Peihao Li; William T. Freeman; Jitendra Malik; Pieter Abbeel; Russ Tedrake; Vincent Sitzmann; Yilun Du
>
> **备注:** 29 pages, 16 figures
>
> **摘要:** General-purpose robots require decision-making models that generalize across diverse tasks and environments. Recent works build robot foundation models by extending multimodal large language models (MLLMs) with action outputs, creating vision-language-action (VLA) systems. These efforts are motivated by the intuition that MLLMs' large-scale language and image pretraining can be effectively transferred to the action output modality. In this work, we explore an alternative paradigm of using large-scale video pretraining as a primary modality for building robot foundation models. Unlike static images and language, videos capture spatio-temporal sequences of states and actions in the physical world that are naturally aligned with robotic behavior. We curate an internet-scale video dataset of human activities and task demonstrations, and train, for the first time at a foundation-model scale, an open video model for generative robotics planning. The model produces zero-shot video plans for novel scenes and tasks, which we post-process to extract executable robot actions. We evaluate task-level generalization through third-party selected tasks in the wild and real-robot experiments, demonstrating successful physical execution. Together, these results show robust instruction following, strong generalization, and real-world feasibility. We release both the model and dataset to support open, reproducible video-based robot learning. Our website is available at https://www.boyuan.space/large-video-planner/.
>
---
#### [new 115] Training Together, Diagnosing Better: Federated Learning for Collagen VI-Related Dystrophies
- **分类: cs.LG; cs.AI; cs.CV; cs.DC**

- **简介: 该论文属医学图像分类任务，旨在解决罕见病COL6-RD因数据孤岛导致的诊断模型性能差问题。作者采用联邦学习，在两家国际机构间协同训练模型，基于胶原VI免疫荧光图像识别三类致病机制，F1达0.82，优于单中心模型。**

- **链接: [https://arxiv.org/pdf/2512.16876v1](https://arxiv.org/pdf/2512.16876v1)**

> **作者:** Astrid Brull; Sara Aguti; Véronique Bolduc; Ying Hu; Daniel M. Jimenez-Gutierrez; Enrique Zuazua; Joaquin Del-Rio; Oleksii Sliusarenko; Haiyan Zhou; Francesco Muntoni; Carsten G. Bönnemann; Xabi Uribe-Etxebarria
>
> **摘要:** The application of Machine Learning (ML) to the diagnosis of rare diseases, such as collagen VI-related dystrophies (COL6-RD), is fundamentally limited by the scarcity and fragmentation of available data. Attempts to expand sampling across hospitals, institutions, or countries with differing regulations face severe privacy, regulatory, and logistical obstacles that are often difficult to overcome. The Federated Learning (FL) provides a promising solution by enabling collaborative model training across decentralized datasets while keeping patient data local and private. Here, we report a novel global FL initiative using the Sherpa.ai FL platform, which leverages FL across distributed datasets in two international organizations for the diagnosis of COL6-RD, using collagen VI immunofluorescence microscopy images from patient-derived fibroblast cultures. Our solution resulted in an ML model capable of classifying collagen VI patient images into the three primary pathogenic mechanism groups associated with COL6-RD: exon skipping, glycine substitution, and pseudoexon insertion. This new approach achieved an F1-score of 0.82, outperforming single-organization models (0.57-0.75). These results demonstrate that FL substantially improves diagnostic utility and generalizability compared to isolated institutional models. Beyond enabling more accurate diagnosis, we anticipate that this approach will support the interpretation of variants of uncertain significance and guide the prioritization of sequencing strategies to identify novel pathogenic variants.
>
---
#### [new 116] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出SALVE框架，属模型可解释性与编辑任务，旨在解决深度网络“黑箱”难控问题。通过稀疏自编码器无监督发现模型原生特征，用Grad-FAM验证并可视化，实现精准、永久的权重干预，并定义临界抑制阈值支持鲁棒性诊断。**

- **链接: [https://arxiv.org/pdf/2512.15938v1](https://arxiv.org/pdf/2512.15938v1)**

> **作者:** Vegard Flovik
>
> **备注:** Under review
>
> **摘要:** Deep neural networks achieve impressive performance but remain difficult to interpret and control. We present SALVE (Sparse Autoencoder-Latent Vector Editing), a unified "discover, validate, and control" framework that bridges mechanistic interpretability and model editing. Using an $\ell_1$-regularized autoencoder, we learn a sparse, model-native feature basis without supervision. We validate these features with Grad-FAM, a feature-level saliency mapping method that visually grounds latent features in input data. Leveraging the autoencoder's structure, we perform precise and permanent weight-space interventions, enabling continuous modulation of both class-defining and cross-class features. We further derive a critical suppression threshold, $α_{crit}$, quantifying each class's reliance on its dominant feature, supporting fine-grained robustness diagnostics. Our approach is validated on both convolutional (ResNet-18) and transformer-based (ViT-B/16) models, demonstrating consistent, interpretable control over their behavior. This work contributes a principled methodology for turning feature discovery into actionable model edits, advancing the development of transparent and controllable AI systems.
>
---
#### [new 117] Don't Guess, Escalate: Towards Explainable Uncertainty-Calibrated AI Forensic Agents
- **分类: cs.MA; cs.AI; cs.CV; cs.MM**

- **简介: 该论文面向多媒体取证任务，解决现有AI forensic方法不确定性建模不足、可解释性差的问题。提出“不确定性校准的可解释AI取证代理”框架，实现检测器协同选择、溯源分析与不确定性感知评估。**

- **链接: [https://arxiv.org/pdf/2512.16614v1](https://arxiv.org/pdf/2512.16614v1)**

> **作者:** Giulia Boato; Andrea Montibeller; Edward Delp; Luisa Verdoliva; Daniele Miorandi
>
> **摘要:** AI is reshaping the landscape of multimedia forensics. We propose AI forensic agents: reliable orchestrators that select and combine forensic detectors, identify provenance and context, and provide uncertainty-aware assessments. We highlight pitfalls in current solutions and introduce a unified framework to improve the authenticity verification process.
>
---
#### [new 118] Foundation Models in Biomedical Imaging: Turning Hype into Reality
- **分类: q-bio.QM; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属综述与批判性分析任务，旨在厘清基础模型（FMs）在生物医学影像中的实际能力与临床落地障碍。工作包括：剖析FMs的推理能力与局限，构建推理能力分类体系，强调因果推断与可信部署（公平性、安全、验证），主张发展可验证、因果增强的辅助式AI系统。**

- **链接: [https://arxiv.org/pdf/2512.15808v1](https://arxiv.org/pdf/2512.15808v1)**

> **作者:** Amgad Muneer; Kai Zhang; Ibraheem Hamdi; Rizwan Qureshi; Muhammad Waqas; Shereen Fouad; Hazrat Ali; Syed Muhammad Anwar; Jia Wu
>
> **备注:** 5 figures and 3 tables
>
> **摘要:** Foundation models (FMs) are driving a prominent shift in artificial intelligence across different domains, including biomedical imaging. These models are designed to move beyond narrow pattern recognition towards emulating sophisticated clinical reasoning, understanding complex spatial relationships, and integrating multimodal data with unprecedented flexibility. However, a critical gap exists between this potential and the current reality, where the clinical evaluation and deployment of FMs are hampered by significant challenges. Herein, we critically assess the current state-of-the-art, analyzing hype by examining the core capabilities and limitations of FMs in the biomedical domain. We also provide a taxonomy of reasoning, ranging from emulated sequential logic and spatial understanding to the integration of explicit symbolic knowledge, to evaluate whether these models exhibit genuine cognition or merely mimic surface-level patterns. We argue that a critical frontier lies beyond statistical correlation, in the pursuit of causal inference, which is essential for building robust models that understand cause and effect. Furthermore, we discuss the paramount issues in deployment stemming from trustworthiness, bias, and safety, dissecting the challenges of algorithmic bias, data bias and privacy, and model hallucinations. We also draw attention to the need for more inclusive, rigorous, and clinically relevant validation frameworks to ensure their safe and ethical application. We conclude that while the vision of autonomous AI-doctors remains distant, the immediate reality is the emergence of powerful technology and assistive tools that would benefit clinical practice. The future of FMs in biomedical imaging hinges not on scale alone, but on developing hybrid, causally aware, and verifiably safe systems that augment, rather than replace, human expertise.
>
---
#### [new 119] Surely Large Multimodal Models (Don't) Excel in Visual Species Recognition?
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究视觉物种识别（VSR）任务，旨在解决小样本下模型性能不足的问题。发现大模型（LMMs）直接用于VSR效果差，但可事后修正小样本专家模型的错误预测；据此提出无需训练的“事后修正”（POC）方法，显著提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.15748v1](https://arxiv.org/pdf/2512.15748v1)**

> **作者:** Tian Liu; Anwesha Basu; James Caverlee; Shu Kong
>
> **备注:** website and code: https://tian1327.github.io/POC
>
> **摘要:** Visual Species Recognition (VSR) is pivotal to biodiversity assessment and conservation, evolution research, and ecology and ecosystem management. Training a machine-learned model for VSR typically requires vast amounts of annotated images. Yet, species-level annotation demands domain expertise, making it realistic for domain experts to annotate only a few examples. These limited labeled data motivate training an ''expert'' model via few-shot learning (FSL). Meanwhile, advanced Large Multimodal Models (LMMs) have demonstrated prominent performance on general recognition tasks. It is straightforward to ask whether LMMs excel in the highly specialized VSR task and whether they outshine FSL expert models. Somewhat surprisingly, we find that LMMs struggle in this task, despite using various established prompting techniques. LMMs even significantly underperform FSL expert models, which are as simple as finetuning a pretrained visual encoder on the few-shot images. However, our in-depth analysis reveals that LMMs can effectively post-hoc correct the expert models' incorrect predictions. Briefly, given a test image, when prompted with the top predictions from an FSL expert model, LMMs can recover the ground-truth label. Building on this insight, we derive a simple method called Post-hoc Correction (POC), which prompts an LMM to re-rank the expert model's top predictions using enriched prompts that include softmax confidence scores and few-shot visual examples. Across five challenging VSR benchmarks, POC outperforms prior art of FSL by +6.4% in accuracy without extra training, validation, or manual intervention. Importantly, POC generalizes to different pretrained backbones and LMMs, serving as a plug-and-play module to significantly enhance existing FSL methods.
>
---
#### [new 120] Autoencoder-based Denoising Defense against Adversarial Attacks on Object Detection
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属对抗防御任务，旨在缓解对抗攻击对目标检测模型的干扰。提出基于单层卷积自编码器的去噪防御方法，在COCO车辆图像上用Perlin噪声攻击YOLOv5，再通过自编码器净化输入，实现部分性能恢复，无需重训练模型。**

- **链接: [https://arxiv.org/pdf/2512.16123v1](https://arxiv.org/pdf/2512.16123v1)**

> **作者:** Min Geun Song; Gang Min Kim; Woonmin Kim; Yongsik Kim; Jeonghyun Sim; Sangbeom Park; Huy Kang Kim
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Deep learning-based object detection models play a critical role in real-world applications such as autonomous driving and security surveillance systems, yet they remain vulnerable to adversarial examples. In this work, we propose an autoencoder-based denoising defense to recover object detection performance degraded by adversarial perturbations. We conduct adversarial attacks using Perlin noise on vehicle-related images from the COCO dataset, apply a single-layer convolutional autoencoder to remove the perturbations, and evaluate detection performance using YOLOv5. Our experiments demonstrate that adversarial attacks reduce bbox mAP from 0.2890 to 0.1640, representing a 43.3% performance degradation. After applying the proposed autoencoder defense, bbox mAP improves to 0.1700 (3.7% recovery) and bbox mAP@50 increases from 0.2780 to 0.3080 (10.8% improvement). These results indicate that autoencoder-based denoising can provide partial defense against adversarial attacks without requiring model retraining.
>
---
#### [new 121] Sceniris: A Fast Procedural Scene Generation Framework
- **分类: cs.RO; cs.CV; cs.GR**

- **简介: 该论文提出Sceniris，一种高效程序化3D场景生成框架，旨在解决现有方法吞吐量低、难以规模化构建物理AI与生成模型所需合成数据集的问题。通过批处理采样和加速碰撞检测，实现234×加速，并支持机器人可达性验证与多样化空间关系建模。**

- **链接: [https://arxiv.org/pdf/2512.16896v1](https://arxiv.org/pdf/2512.16896v1)**

> **作者:** Jinghuan Shang; Harsh Patel; Ran Gong; Karl Schmeckpeper
>
> **备注:** Code is available at https://github.com/rai-inst/sceniris
>
> **摘要:** Synthetic 3D scenes are essential for developing Physical AI and generative models. Existing procedural generation methods often have low output throughput, creating a significant bottleneck in scaling up dataset creation. In this work, we introduce Sceniris, a highly efficient procedural scene generation framework for rapidly generating large-scale, collision-free scene variations. Sceniris also provides an optional robot reachability check, providing manipulation-feasible scenes for robot tasks. Sceniris is designed for maximum efficiency by addressing the primary performance limitations of the prior method, Scene Synthesizer. Leveraging batch sampling and faster collision checking in cuRobo, Sceniris achieves at least 234x speed-up over Scene Synthesizer. Sceniris also expands the object-wise spatial relationships available in prior work to support diverse scene requirements. Our code is available at https://github.com/rai-inst/sceniris
>
---
#### [new 122] MCR-VQGAN: A Scalable and Cost-Effective Tau PET Synthesis Approach for Alzheimer's Disease Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属医学图像合成任务，旨在解决tau PET成像成本高、辐射大、普及难的问题。作者提出MCR-VQGAN模型，利用T1 MRI合成高质量tau PET图像；在ADNI数据上验证其性能优越且保留诊断特征，可作为可靠替代方案。**

- **链接: [https://arxiv.org/pdf/2512.15947v1](https://arxiv.org/pdf/2512.15947v1)**

> **作者:** Jin Young Kim; Jeremy Hudson; Jeongchul Kim; Qing Lyu; Christopher T. Whitlow
>
> **备注:** 12 pages, 5 figures. A preliminary version of this work was presented at RSNA 2025
>
> **摘要:** Tau positron emission tomography (PET) is a critical diagnostic modality for Alzheimer's disease (AD) because it visualizes and quantifies neurofibrillary tangles, a hallmark of AD pathology. However, its widespread clinical adoption is hindered by significant challenges, such as radiation exposure, limited availability, high clinical workload, and substantial financial costs. To overcome these limitations, we propose Multi-scale CBAM Residual Vector Quantized Generative Adversarial Network (MCR-VQGAN) to synthesize high-fidelity tau PET images from structural T1-weighted MRI scans. MCR-VQGAN improves standard VQGAN by integrating three key architectural enhancements: multi-scale convolutions, ResNet blocks, and Convolutional Block Attention Modules (CBAM). Using 222 paired structural T1-weighted MRI and tau PET scans from Alzheimer's Disease Neuroimaging Initiative (ADNI), we trained and compared MCR-VQGAN with cGAN, WGAN-GP, CycleGAN, and VQGAN. Our proposed model achieved superior image synthesis performance across all metrics: MSE of 0.0056 +/- 0.0061, PSNR of 24.39 +/- 4.49 dB, and SSIM of 0.9000 +/- 0.0453. To assess the clinical utility of the synthetic images, we trained and evaluated a CNN-based AD classifier. The classifier achieved comparable accuracy when tested on real (63.64%) and synthetic (65.91%) images. This result indicates that our synthesis process successfully preserves diagnostically relevant features without significant information loss. Our results demonstrate that MCR-VQGAN can offer a reliable and scalable surrogate for conventional tau PET imaging, potentially improving the accessibility and scalability of tau imaging biomarkers for AD research and clinical workflows.
>
---
#### [new 123] Human-like Working Memory from Artificial Intrinsic Plasticity Neurons
- **分类: cs.ET; cs.AI; cs.CV; cs.NE**

- **简介: 该论文提出IPNet——一种基于磁隧道结（MTJ）器件的类脑工作记忆神经形态架构，旨在解决传统人工网络能耗高、噪声敏感、缺乏生物真实性的问题。通过硬件-软件协同设计，实现人类相似的工作记忆行为，并在动态视觉识别与自动驾驶任务中显著提升精度与能效。**

- **链接: [https://arxiv.org/pdf/2512.15829v1](https://arxiv.org/pdf/2512.15829v1)**

> **作者:** Jingli Liu; Huannan Zheng; Bohao Zou; Kezhou Yang
>
> **摘要:** Working memory enables the brain to integrate transient information for rapid decision-making. Artificial networks typically replicate this via recurrent or parallel architectures, yet incur high energy costs and noise sensitivity. Here we report IPNet, a hardware-software co-designed neuromorphic architecture realizing human-like working memory via neuronal intrinsic plasticity. Exploiting Joule-heating dynamics of Magnetic Tunnel Junctions (MTJs), IPNet physically emulates biological memory volatility. The memory behavior of the proposed architecture shows similar trends in n-back, free recall and memory interference tasks to that of reported human subjects. Implemented exclusively with MTJ neurons, the architecture with human-like working memory achieves 99.65% accuracy on 11-class DVS gesture datasets and maintains 99.48% on a novel 22-class time-reversed benchmark, outperforming RNN, LSTM, and 2+1D CNN baselines sharing identical backbones. For autonomous driving (DDD-20), IPNet reduces steering prediction error by 14.4% compared to ResNet-LSTM. Architecturally, we identify a 'Memory-at-the-Frontier' effect where performance is maximized at the sensing interface, validating a bio-plausible near-sensor processing paradigm. Crucially, all results rely on raw parameters from fabricated devices without optimization. Hardware-in-the-loop validation confirms the system's physical realizability. Separately, energy analysis reveals a reduction in memory power of 2,874x compared to LSTMs and 90,920x versus parallel 3D-CNNs. This capacitor-free design enables a compact ~1.5um2 footprint (28 nm CMOS): a >20-fold reduction over standard LIF neurons. Ultimately, we demonstrate that instantiating human-like working memory via intrinsic neuronal plasticity endows neural networks with the dual biological advantages of superior dynamic vision processing and minimal metabolic cost.
>
---
#### [new 124] D3G: Diverse Demographic Data Generation Increases Zero-Shot Image Classification Accuracy within Multimodal Models
- **分类: cs.LG; cs.CL; cs.CV; cs.CY**

- **简介: 该论文属零-shot图像分类任务，旨在缓解多模态模型（如CLIP）因训练数据人口统计失衡导致的偏差与性能下降。提出无需训练的D3G方法：在推理时用Stable Diffusion XL生成多样化人口统计数据，提升准确率并减少 demographic bias。**

- **链接: [https://arxiv.org/pdf/2512.15747v1](https://arxiv.org/pdf/2512.15747v1)**

> **作者:** Javon Hickmon
>
> **摘要:** Image classification is a task essential for machine perception to achieve human-level image understanding. Multimodal models such as CLIP have been able to perform well on this task by learning semantic similarities across vision and language; however, despite these advances, image classification is still a challenging task. Models with low capacity often suffer from underfitting and thus underperform on fine-grained image classification. Along with this, it is important to ensure high-quality data with rich cross-modal representations of each class, which is often difficult to generate. When datasets do not enforce balanced demographics, the predictions will be biased toward the more represented class, while others will be neglected. We focus on how these issues can lead to harmful bias for zero-shot image classification, and explore how to combat these issues in demographic bias. We propose Diverse Demographic Data Generation (D3G), a training-free, zero-shot method of boosting classification accuracy while reducing demographic bias in pre-trained multimodal models. With this method, we utilize CLIP as our base multimodal model and Stable Diffusion XL as our generative model. We demonstrate that providing diverse demographic data at inference time improves performance for these models, and explore the impact of individual demographics on the resulting accuracy metric.
>
---
#### [new 125] Dual-View Inference Attack: Machine Unlearning Amplifies Privacy Exposure
- **分类: cs.LG; cs.CV**

- **简介: 该论文属隐私安全任务，揭示机器遗忘后“双视角”（原模型+遗忘模型）下对**保留数据**的新型隐私风险。提出DVIA攻击方法，利用黑盒查询两模型，通过轻量似然比推理推断成员身份，无需训练攻击模型，实证验证遗忘反而加剧隐私泄露。**

- **链接: [https://arxiv.org/pdf/2512.16126v1](https://arxiv.org/pdf/2512.16126v1)**

> **作者:** Lulu Xue; Shengshan Hu; Linqiang Qian; Peijin Guo; Yechao Zhang; Minghui Li; Yanjun Zhang; Dayong Ye; Leo Yu Zhang
>
> **备注:** Accepeted by AAAI2026
>
> **摘要:** Machine unlearning is a newly popularized technique for removing specific training data from a trained model, enabling it to comply with data deletion requests. While it protects the rights of users requesting unlearning, it also introduces new privacy risks. Prior works have primarily focused on the privacy of data that has been unlearned, while the risks to retained data remain largely unexplored. To address this gap, we focus on the privacy risks of retained data and, for the first time, reveal the vulnerabilities introduced by machine unlearning under the dual-view setting, where an adversary can query both the original and the unlearned models. From an information-theoretic perspective, we introduce the concept of {privacy knowledge gain} and demonstrate that the dual-view setting allows adversaries to obtain more information than querying either model alone, thereby amplifying privacy leakage. To effectively demonstrate this threat, we propose DVIA, a Dual-View Inference Attack, which extracts membership information on retained data using black-box queries to both models. DVIA eliminates the need to train an attack model and employs a lightweight likelihood ratio inference module for efficient inference. Experiments across different datasets and model architectures validate the effectiveness of DVIA and highlight the privacy risks inherent in the dual-view setting.
>
---
#### [new 126] In search of truth: Evaluating concordance of AI-based anatomy segmentation models
- **分类: eess.IV; cs.CV**

- **简介: 该论文属医学图像分割模型评估任务，旨在解决无金标准标注时多AI模型性能比较难题。作者提出标准化分割结果表示框架，扩展3D Slicer与OHIF工具，实现跨模型结构级自动化比对与可视化，并在NLST数据上验证六种开源模型对31解剖结构的分割一致性。**

- **链接: [https://arxiv.org/pdf/2512.15921v1](https://arxiv.org/pdf/2512.15921v1)**

> **作者:** Lena Giebeler; Deepa Krishnaswamy; David Clunie; Jakob Wasserthal; Lalith Kumar Shiyam Sundar; Andres Diaz-Pinto; Klaus H. Maier-Hein; Murong Xu; Bjoern Menze; Steve Pieper; Ron Kikinis; Andrey Fedorov
>
> **摘要:** Purpose AI-based methods for anatomy segmentation can help automate characterization of large imaging datasets. The growing number of similar in functionality models raises the challenge of evaluating them on datasets that do not contain ground truth annotations. We introduce a practical framework to assist in this task. Approach We harmonize the segmentation results into a standard, interoperable representation, which enables consistent, terminology-based labeling of the structures. We extend 3D Slicer to streamline loading and comparison of these harmonized segmentations, and demonstrate how standard representation simplifies review of the results using interactive summary plots and browser-based visualization using OHIF Viewer. To demonstrate the utility of the approach we apply it to evaluating segmentation of 31 anatomical structures (lungs, vertebrae, ribs, and heart) by six open-source models - TotalSegmentator 1.5 and 2.6, Auto3DSeg, MOOSE, MultiTalent, and CADS - for a sample of Computed Tomography (CT) scans from the publicly available National Lung Screening Trial (NLST) dataset. Results We demonstrate the utility of the framework in enabling automating loading, structure-wise inspection and comparison across models. Preliminary results ascertain practical utility of the approach in allowing quick detection and review of problematic results. The comparison shows excellent agreement segmenting some (e.g., lung) but not all structures (e.g., some models produce invalid vertebrae or rib segmentations). Conclusions The resources developed are linked from https://imagingdatacommons.github.io/segmentation-comparison/ including segmentation harmonization scripts, summary plots, and visualization tools. This work assists in model evaluation in absence of ground truth, ultimately enabling informed model selection.
>
---
#### [new 127] A Tri-Dynamic Preprocessing Framework for UGC Video Compression
- **分类: cs.MM; cs.CV**

- **简介: 该论文属视频压缩任务，旨在解决UGC视频因高度可变性导致的编码优化困难问题。提出三动态预处理框架：自适应调节预处理强度、量化等级和率失真损失权衡，提升机器学习编码器在UGC场景下的泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.16101v1](https://arxiv.org/pdf/2512.16101v1)**

> **作者:** Fei Zhao; Mengxi Guo; Shijie Zhao; Junlin Li; Li Zhang; Xiaodong Xie
>
> **备注:** Accepted as a POSTER and for publication in the ICASSP 2024 proceedings
>
> **摘要:** In recent years, user generated content (UGC) has become the dominant force in internet traffic. However, UGC videos exhibit a higher degree of variability and diverse characteristics compared to traditional encoding test videos. This variance challenges the effectiveness of data-driven machine learning algorithms for optimizing encoding in the broader context of UGC scenarios. To address this issue, we propose a Tri-Dynamic Preprocessing framework for UGC. Firstly, we employ an adaptive factor to regulate preprocessing intensity. Secondly, an adaptive quantization level is employed to fine-tune the codec simulator. Thirdly, we utilize an adaptive lambda tradeoff to adjust the rate-distortion loss function. Experimental results on large-scale test sets demonstrate that our method attains exceptional performance.
>
---
#### [new 128] VERM: Leveraging Foundation Models to Create a Virtual Eye for Efficient 3D Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向3D机器人操作任务，解决多相机感知冗余、遮挡及计算开销大问题。提出VERM方法：利用基础模型从3D点云生成任务自适应虚拟视角，并设计深度感知模块与动态粗到细规划策略，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2512.16724v1](https://arxiv.org/pdf/2512.16724v1)**

> **作者:** Yixiang Chen; Yan Huang; Keji He; Peiyan Li; Liang Wang
>
> **备注:** Accepted at RA-L 2025
>
> **摘要:** When performing 3D manipulation tasks, robots have to execute action planning based on perceptions from multiple fixed cameras. The multi-camera setup introduces substantial redundancy and irrelevant information, which increases computational costs and forces the model to spend extra training time extracting crucial task-relevant details. To filter out redundant information and accurately extract task-relevant features, we propose the VERM (Virtual Eye for Robotic Manipulation) method, leveraging the knowledge in foundation models to imagine a virtual task-adaptive view from the constructed 3D point cloud, which efficiently captures necessary information and mitigates occlusion. To facilitate 3D action planning and fine-grained manipulation, we further design a depth-aware module and a dynamic coarse-to-fine procedure. Extensive experimental results on both simulation benchmark RLBench and real-world evaluations demonstrate the effectiveness of our method, surpassing previous state-of-the-art methods while achieving 1.89x speedup in training time and 1.54x speedup in inference speed. More results can be found on our project website at https://verm-ral.github.io .
>
---
#### [new 129] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Multimodal RewardBench 2（MMRB2），首个面向图文交错模态的奖励模型基准，涵盖文本生成图像、图像编辑、交错生成和多模态推理四任务。旨在评估与提升多模态奖励模型性能，解决其在 omni 模型中缺乏系统评测的问题。**

- **链接: [https://arxiv.org/pdf/2512.16899v1](https://arxiv.org/pdf/2512.16899v1)**

> **作者:** Yushi Hu; Reyhane Askari-Hemmat; Melissa Hall; Emily Dinan; Luke Zettlemoyer; Marjan Ghazvininejad
>
> **备注:** Code and data available at https://github.com/facebookresearch/MMRB2
>
> **摘要:** Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning ("thinking-with-images"), providing 1,000 expert-annotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66-75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.
>
---
#### [new 130] BioimageAIpub: a toolbox for AI-ready bioimaging data publishing
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出BioimageAIpub工具箱，解决生物影像数据因格式不兼容、标注/元数据不足而难以被AI模型直接使用的难题；通过自动化转换与标准化，实现生物影像数据一键发布至HuggingFace平台，提升AI-ready数据共享效率。**

- **链接: [https://arxiv.org/pdf/2512.15820v1](https://arxiv.org/pdf/2512.15820v1)**

> **作者:** Stefan Dvoretskii; Anwai Archit; Constantin Pape; Josh Moore; Marco Nolden
>
> **摘要:** Modern bioimage analysis approaches are data hungry, making it necessary for researchers to scavenge data beyond those collected within their (bio)imaging facilities. In addition to scale, bioimaging datasets must be accompanied with suitable, high-quality annotations and metadata. Although established data repositories such as the Image Data Resource (IDR) and BioImage Archive offer rich metadata, their contents typically cannot be directly consumed by image analysis tools without substantial data wrangling. Such a tedious assembly and conversion of (meta)data can account for a dedicated amount of time investment for researchers, hindering the development of more powerful analysis tools. Here, we introduce BioimageAIpub, a workflow that streamlines bioimaging data conversion, enabling a seamless upload to HuggingFace, a widely used platform for sharing machine learning datasets and models.
>
---
## 更新

#### [replaced 001] Deep generative priors for 3D brain analysis
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.15119v2](https://arxiv.org/pdf/2510.15119v2)**

> **作者:** Ana Lawry Aguila; Dina Zemlyanker; You Cheng; Sudeshna Das; Daniel C. Alexander; Oula Puonti; Annabel Sorby-Adams; W. Taylor Kimberly; Juan Eugenio Iglesias
>
> **摘要:** Diffusion models have recently emerged as powerful generative models in medical imaging. However, it remains a major challenge to combine these data-driven models with domain knowledge to guide brain imaging problems. In neuroimaging, Bayesian inverse problems have long provided a successful framework for inference tasks, where incorporating domain knowledge of the imaging process enables robust performance without requiring extensive training data. However, the anatomical modeling component of these approaches typically relies on classical mathematical priors that often fail to capture the complex structure of brain anatomy. In this work, we present the first general-purpose application of diffusion models as priors for solving a wide range of medical imaging inverse problems. Our approach leverages a score-based diffusion prior trained extensively on diverse brain MRI data, paired with flexible forward models that capture common image processing tasks such as super-resolution, bias field correction, inpainting, and combinations thereof. We further demonstrate how our framework can refine outputs from existing deep learning methods to improve anatomical fidelity. Experiments on heterogeneous clinical and research MRI data show that our method achieves state-of-the-art performance producing consistent, high-quality solutions without requiring paired training datasets. These results highlight the potential of diffusion priors as versatile tools for brain MRI analysis.
>
---
#### [replaced 002] From Logits to Hierarchies: Hierarchical Clustering made Simple
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2410.07858v2](https://arxiv.org/pdf/2410.07858v2)**

> **作者:** Emanuele Palumbo; Moritz Vandenhirtz; Alain Ryser; Imant Daunhawer; Julia E. Vogt
>
> **备注:** ICML 2025 camera-ready version
>
> **摘要:** The hierarchical structure inherent in many real-world datasets makes the modeling of such hierarchies a crucial objective in both unsupervised and supervised machine learning. While recent advancements have introduced deep architectures specifically designed for hierarchical clustering, we adopt a critical perspective on this line of research. Our findings reveal that these methods face significant limitations in scalability and performance when applied to realistic datasets. Given these findings, we present an alternative approach and introduce a lightweight method that builds on pre-trained non-hierarchical clustering models. Remarkably, our approach outperforms specialized deep models for hierarchical clustering, and it is broadly applicable to any pre-trained clustering model that outputs logits, without requiring any fine-tuning. To highlight the generality of our approach, we extend its application to a supervised setting, demonstrating its ability to recover meaningful hierarchies from a pre-trained ImageNet classifier. Our results establish a practical and effective alternative to existing deep hierarchical clustering methods, with significant advantages in efficiency, scalability and performance.
>
---
#### [replaced 003] Scene-aware SAR ship detection guided by unsupervised sea-land segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.12775v2](https://arxiv.org/pdf/2506.12775v2)**

> **作者:** Han Ke; Xiao Ke; Ye Yan; Rui Liu; Jinpeng Yang; Tianwen Zhang; Xu Zhan; Xiaowo Xu
>
> **摘要:** DL based Synthetic Aperture Radar (SAR) ship detection has tremendous advantages in numerous areas. However, it still faces some problems, such as the lack of prior knowledge, which seriously affects detection accuracy. In order to solve this problem, we propose a scene-aware SAR ship detection method based on unsupervised sea-land segmentation. This method follows a classical two-stage framework and is enhanced by two models: the unsupervised land and sea segmentation module (ULSM) and the land attention suppression module (LASM). ULSM and LASM can adaptively guide the network to reduce attention on land according to the type of scenes (inshore scene and offshore scene) and add prior knowledge (sea land segmentation information) to the network, thereby reducing the network's attention to land directly and enhancing offshore detection performance relatively. This increases the accuracy of ship detection and enhances the interpretability of the model. Specifically, in consideration of the lack of land sea segmentation labels in existing deep learning-based SAR ship detection datasets, ULSM uses an unsupervised approach to classify the input data scene into inshore and offshore types and performs sea-land segmentation for inshore scenes. LASM uses the sea-land segmentation information as prior knowledge to reduce the network's attention to land. We conducted our experiments using the publicly available SSDD dataset, which demonstrated the effectiveness of our network.
>
---
#### [replaced 004] Learning Multimodal Embeddings for Traffic Accident Prediction and Causal Estimation
- **分类: cs.LG; cs.CV; cs.SI**

- **链接: [https://arxiv.org/pdf/2512.02920v2](https://arxiv.org/pdf/2512.02920v2)**

> **作者:** Ziniu Zhang; Minxuan Duan; Haris N. Koutsopoulos; Hongyang R. Zhang
>
> **备注:** 17 pages. To appear in KDD'26 Datasets
>
> **摘要:** We consider analyzing traffic accident patterns using both road network data and satellite images aligned to road graph nodes. Previous work for predicting accident occurrences relies primarily on road network structural features while overlooking physical and environmental information from the road surface and its surroundings. In this work, we construct a large multimodal dataset across six U.S. states, containing nine million traffic accident records from official sources, and one million high-resolution satellite images for each node of the road network. Additionally, every node is annotated with features such as the region's weather statistics and road type (e.g., residential vs. motorway), and each edge is annotated with traffic volume information (i.e., Average Annual Daily Traffic). Utilizing this dataset, we conduct a comprehensive evaluation of multimodal learning methods that integrate both visual and network embeddings. Our findings show that integrating both data modalities improves prediction accuracy, achieving an average AUROC of $90.1\%$, which is a $3.7\%$ gain over graph neural network models that only utilize graph structures. With the improved embeddings, we conduct a causal analysis based on a matching estimator to estimate the key contributing factors influencing traffic accidents. We find that accident rates rise by $24\%$ under higher precipitation, by $22\%$ on higher-speed roads such as motorways, and by $29\%$ due to seasonal patterns, after adjusting for other confounding factors. Ablation studies confirm that satellite imagery features are essential for achieving accurate prediction.
>
---
#### [replaced 005] MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.21503v3](https://arxiv.org/pdf/2507.21503v3)**

> **作者:** Yanxu Zhu; Shitong Duan; Xiangxu Zhang; Jitao Sang; Peng Zhang; Tun Lu; Xiao Zhou; Jing Yao; Xiaoyuan Yi; Xing Xie
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Recently Multimodal Large Language Models (MLLMs) have achieved considerable advancements in vision-language tasks, yet produce potentially harmful or untrustworthy content. Despite substantial work investigating the trustworthiness of language models, MMLMs' capability to act honestly, especially when faced with visually unanswerable questions, remains largely underexplored. This work presents the first systematic assessment of honesty behaviors across various MLLMs. We ground honesty in models' response behaviors to unanswerable visual questions, define four representative types of such questions, and construct MoHoBench, a large-scale MMLM honest benchmark, consisting of 12k+ visual question samples, whose quality is guaranteed by multi-stage filtering and human verification. Using MoHoBench, we benchmarked the honesty of 28 popular MMLMs and conducted a comprehensive analysis. Our findings show that: (1) most models fail to appropriately refuse to answer when necessary, and (2) MMLMs' honesty is not solely a language modeling issue, but is deeply influenced by visual information, necessitating the development of dedicated methods for multimodal honesty alignment. Therefore, we implemented initial alignment methods using supervised and preference learning to improve honesty behavior, providing a foundation for future work on trustworthy MLLMs. Our data and code can be found at https://github.com/yanxuzhu/MoHoBench.
>
---
#### [replaced 006] StructDiff: Structure-aware Diffusion Model for 3D Fine-grained Medical Image Synthesis
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09560v2](https://arxiv.org/pdf/2503.09560v2)**

> **作者:** Jiahao Xia; Yutao Hu; Yaolei Qi; Zhenliang Li; Wenqi Shao; Junjun He; Ying Fu; Longjiang Zhang; Guanyu Yang
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Solving medical imaging data scarcity through semantic image generation has attracted growing attention in recent years. However, existing generative models mainly focus on synthesizing whole-organ or large-tissue structures, showing limited capability in reproducing fine-grained anatomical details. Due to the stringent requirement of topological consistency and the complex 3D morphological heterogeneity of medical data, accurately reconstructing fine-grained anatomical details remains a significant challenge. To address these limitations, we propose StructDiff, a Structure-aware Diffusion Model for fine-grained 3D medical image synthesis, which enables precise generation of topologically complex anatomies. In addition to the conventional mask-based guidance, StructDiff further introduces a paired image-mask template to guide the generation process, providing structural constrains and offering explicit knowledge of mask-to-image correspondence. Moreover, a Mask Generation Module (MGM) is designed to enrich mask diversity and alleviate the scarcity of high-quality reference masks. Furthermore, we propose a Confidence-aware Adaptive Learning (CAL) strategy based on Skip-Sampling Variance (SSV), which mitigates uncertainty introduced by imperfect synthetic data when transferring to downstream tasks. Extensive experiments demonstrate that StructDiff achieves state-of-the-art performance in terms of topological consistency and visual realism, and significantly boosts downstream segmentation performance. Code will be released upon acceptance.
>
---
#### [replaced 007] Low-Resolution Action Recognition for Tiny Actions Challenge
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2209.14711v2](https://arxiv.org/pdf/2209.14711v2)**

> **作者:** Boyu Chen; Yu Qiao; Yali Wang
>
> **备注:** This article is the report of the CVPR 2022 ActivityNet workshop Tiny Actions Challenge(https://tinyactions-cvpr22.github.io/). The time of the first submission to the organizers is June 6th
>
> **摘要:** Tiny Actions Challenge focuses on understanding human activities in real-world surveillance. Basically, there are two main difficulties for activity recognition in this scenario. First, human activities are often recorded at a distance, and appear in a small resolution without much discriminative clue. Second, these activities are naturally distributed in a long-tailed way. It is hard to alleviate data bias for such heavy category imbalance. To tackle these problems, we propose a comprehensive recognition solution in this paper. First, we train video backbones with data balance, in order to alleviate overfitting in the challenge benchmark. Second, we design a dual-resolution distillation framework, which can effectively guide low-resolution action recognition by super-resolution knowledge. Finally, we apply model en-semble with post-processing, which can further boost per-formance on the long-tailed categories. Our solution ranks Top-1 on the leaderboard.
>
---
#### [replaced 008] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属视觉-语言预训练任务，旨在解决现有方法因过度依赖目标帧导致的错误跨模态对齐问题。作者提出Action Temporal Coherence Learning（AcTOL），通过帧间语义对比和局部布朗桥约束，学习有序、连续的视觉-语言表征，提升具身智能体在下游操作任务中的泛化性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2502.01218v3](https://arxiv.org/pdf/2502.01218v3)**

> **作者:** Zhizhen Zhang; Lei Zhu; Zhen Fang; Zi Huang; Yadan Luo
>
> **备注:** NeurIPS 2025 Poster
>
> **摘要:** Pre-training vision-language representations on human action videos has emerged as a promising approach to reduce reliance on large-scale expert demonstrations for training embodied agents. However, prior methods often employ time contrastive learning based on goal-reaching heuristics, progressively aligning language instructions from the initial to the final frame. This overemphasis on future frames can result in erroneous vision-language associations, as actions may terminate early or include irrelevant moments in the end. To address this issue, we propose Action Temporal Coherence Learning (AcTOL) to learn ordered and continuous vision-language representations without rigid goal-based constraint. AcTOL treats a video as a continuous trajectory where it (1) contrasts semantic differences between frames to reflect their natural ordering, and (2) imposes a local Brownian bridge constraint to ensure smooth transitions across intermediate frames. Extensive imitation learning experiments on both simulated and real robots show that the pretrained features significantly enhance downstream manipulation tasks with high robustness to different linguistic styles of instructions, offering a viable pathway toward generalized embodied agents.
>
---
#### [replaced 009] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12998v3](https://arxiv.org/pdf/2511.12998v3)**

> **作者:** Zewei Chang; Zheng-Peng Duan; Jianxing Zhang; Chun-Le Guo; Siyu Liu; Hyungju Chun; Hyunhee Park; Zikun Liu; Chongyi Li
>
> **备注:** To appear at AAAI 2026
>
> **摘要:** Image retouching aims to enhance visual quality while aligning with users' personalized aesthetic preferences. To address the challenge of balancing controllability and subjectivity, we propose a unified diffusion-based image retouching framework called PerTouch. Our method supports semantic-level image retouching while maintaining global aesthetics. Using parameter maps containing attribute values in specific semantic regions as input, PerTouch constructs an explicit parameter-to-image mapping for fine-grained image retouching. To improve semantic boundary perception, we introduce semantic replacement and parameter perturbation mechanisms during training. To connect natural language instructions with visual control, we develop a VLM-driven agent to handle both strong and weak user instructions. Equipped with mechanisms of feedback-driven rethinking and scene-aware memory, PerTouch better aligns with user intent and captures long-term preferences. Extensive experiments demonstrate each component's effectiveness and the superior performance of PerTouch in personalized image retouching. Code Pages: https://github.com/Auroral703/PerTouch.
>
---
#### [replaced 010] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03508v2](https://arxiv.org/pdf/2512.03508v2)**

> **作者:** Seogkyu Jeon; Kibeom Hong; Hyeran Byun
>
> **备注:** ICCV 2025 (poster)
>
> **摘要:** Recent domain generalized semantic segmentation (DGSS) studies have achieved notable improvements by distilling semantic knowledge from Vision-Language Models (VLMs). However, they overlook the semantic misalignment between visual and textual contexts, which arises due to the rigidity of a fixed context prompt learned on a single source domain. To this end, we present a novel domain generalization framework for semantic segmentation, namely Domain-aware Prompt-driven Masked Transformer (DPMFormer). Firstly, we introduce domain-aware prompt learning to facilitate semantic alignment between visual and textual cues. To capture various domain-specific properties with a single source dataset, we propose domain-aware contrastive learning along with the texture perturbation that diversifies the observable domains. Lastly, to establish a framework resilient against diverse environmental changes, we have proposed the domain-robust consistency learning which guides the model to minimize discrepancies of prediction from original and the augmented images. Through experiments and analyses, we demonstrate the superiority of the proposed framework, which establishes a new state-of-the-art on various DGSS benchmarks. The code is available at https://github.com/jone1222/DPMFormer.
>
---
#### [replaced 011] Towards Practical Alzheimer's Disease Diagnosis: A Lightweight and Interpretable Spiking Neural Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.09695v3](https://arxiv.org/pdf/2506.09695v3)**

> **作者:** Changwei Wu; Yifei Chen; Yuxin Du; Jinying Zong; Jie Dong; Mingxuan Liu; Feiwei Qin; Yong Peng; Jin Fan; Changmiao Wang
>
> **备注:** 35 pages, 8 figures
>
> **摘要:** Early diagnosis of Alzheimer's Disease (AD), particularly at the mild cognitive impairment stage, is essential for timely intervention. However, this process faces significant barriers, including reliance on subjective assessments and the high cost of advanced imaging techniques. While deep learning offers automated solutions to improve diagnostic accuracy, its widespread adoption remains constrained due to high energy requirements and computational demands, particularly in resource-limited settings. Spiking neural networks (SNNs) provide a promising alternative, as their brain-inspired design is well-suited to model the sparse and event-driven patterns characteristic of neural degeneration in AD. These networks offer the potential for developing interpretable, energy-efficient diagnostic tools. Despite their advantages, existing SNNs often suffer from limited expressiveness and challenges in stable training, which reduce their effectiveness in handling complex medical tasks. To address these shortcomings, we introduce FasterSNN, a hybrid neural architecture that combines biologically inspired Leaky Integrate-and-Fire (LIF) neurons with region-adaptive convolution and multi-scale spiking attention mechanisms. This approach facilitates efficient, sparse processing of 3D MRI data while maintaining high diagnostic accuracy. Experimental results on benchmark datasets reveal that FasterSNN delivers competitive performance with significantly enhanced efficiency and training stability, highlighting its potential for practical application in AD screening. Our source code is available at https://github.com/wuchangw/FasterSNN.
>
---
#### [replaced 012] Team Westwood Solution for MIDOG 2025 Challenge: An Ensemble-CNN-Based Approach For Mitosis Detection And Classification
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.02600v3](https://arxiv.org/pdf/2509.02600v3)**

> **作者:** Tengyou Xu; Haochen Yang; Xiang 'Anthony' Chen; Hongyan Gu; Mohammad Haeri
>
> **备注:** To appear Lecture Notes in Computer Science
>
> **摘要:** This abstract presents our solution (Team Westwood) for mitosis detection and atypical mitosis classification in the MItosis DOmain Generalization (MIDOG) 2025 challenge. For mitosis detection, we trained an nnUNetV2 for initial mitosis candidate screening with high sensitivity, followed by a random forest classifier ensembling predictions of three convolutional neural networks (CNNs): EfficientNet-b3, EfficientNet-b5, and EfficientNetV2-s. For the atypical mitosis classification, we trained another random forest classifier ensembling the predictions of three CNNs: EfficientNet-b3, EfficientNet-b5, and InceptionV3. On the preliminary test set, our solution achieved an F1 score of 0.7450 for track 1 mitosis detection, and a balanced accuracy of 0.8722 for track 2 atypical mitosis classification. On the final test set, our solution achieved an F1 score of 0.6972 for track 1 mitosis detection, and a balanced accuracy of 0.8242 for track 2 atypical mitosis classification.
>
---
#### [replaced 013] Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04677v3](https://arxiv.org/pdf/2512.04677v3)**

> **作者:** Yubo Huang; Hailong Guo; Fangtai Wu; Shifeng Zhang; Shijie Huang; Qijun Gan; Lin Liu; Sirui Zhao; Enhong Chen; Jiaming Liu; Steven Hoi
>
> **摘要:** Existing diffusion-based video generation methods are fundamentally constrained by sequential computation and long-horizon inconsistency, limiting their practical adoption in real-time, streaming audio-driven avatar synthesis. We present Live Avatar, an algorithm-system co-designed framework that enables efficient, high-fidelity, and infinite-length avatar generation using a 14-billion-parameter diffusion model. Our approach introduces Timestep-forcing Pipeline Parallelism (TPP), a distributed inference paradigm that pipelines denoising steps across multiple GPUs, effectively breaking the autoregressive bottleneck and ensuring stable, low-latency real-time streaming. To further enhance temporal consistency and mitigate identity drift and color artifacts, we propose the Rolling Sink Frame Mechanism (RSFM), which maintains sequence fidelity by dynamically recalibrating appearance using a cached reference image. Additionally, we leverage Self-Forcing Distribution Matching Distillation to facilitate causal, streamable adaptation of large-scale models without sacrificing visual quality. Live Avatar demonstrates state-of-the-art performance, reaching 20 FPS end-to-end generation on 5 H800 GPUs, and, to the best of our knowledge, is the first to achieve practical, real-time, high-fidelity avatar generation at this scale. Our work establishes a new paradigm for deploying advanced diffusion models in industrial long-form video synthesis applications.
>
---
#### [replaced 014] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.24862v4](https://arxiv.org/pdf/2505.24862v4)**

> **作者:** Cailin Zhuang; Ailin Huang; Yaoqi Hu; Jingwei Wu; Wei Cheng; Jiaqi Liao; Hongyuan Wang; Xinyao Liao; Weiwei Cai; Hengyuan Xu; Xuanyang Zhang; Xianfang Zeng; Zhewei Huang; Gang Yu; Chi Zhang
>
> **备注:** 33 Pages, Project Page: https://vistorybench.github.io/, Code: https://github.com/vistorybench/vistorybench
>
> **摘要:** Story visualization aims to generate coherent image sequences that faithfully depict a narrative and align with character references. Despite progress in generative models, existing benchmarks are narrow in scope, often limited to short prompts, lacking character references, or single-image cases, and fail to capture real-world storytelling complexity. This hinders a nuanced understanding of model capabilities and limitations. We present \textbf{ViStoryBench}, a comprehensive benchmark designed to evaluate story visualization models across diverse narrative structures, visual styles, and character settings. The benchmark features richly annotated multi-shot scripts derived from curated stories spanning literature, film, and folklore. Large language models assist in story summarization and script generation, with all outputs human-verified to ensure coherence and fidelity. Character references are carefully curated to maintain intra-story consistency across varying artistic styles. To enable thorough evaluation, ViStoryBench introduces a set of automated metrics that assess character consistency, style similarity, prompt alignment, aesthetic quality, and generation artifacts such as copy-paste behavior. These metrics are validated through human studies, and used to benchmark a broad range of open-source and commercial models. ViStoryBench offers a multi-dimensional evaluation suite that facilitates systematic analysis and fosters future progress in visual storytelling.
>
---
#### [replaced 015] DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09117v2](https://arxiv.org/pdf/2511.09117v2)**

> **作者:** Rui-Yang Ju; Kohei Yamashita; Hirotaka Kameko; Shinsuke Mori
>
> **摘要:** Kuzushiji, a pre-modern Japanese cursive script, can currently be read and understood by only a few thousand trained experts in Japan. With the rapid development of deep learning, researchers have begun applying Optical Character Recognition (OCR) techniques to transcribe Kuzushiji into modern Japanese. Although existing OCR methods perform well on clean pre-modern Japanese documents written in Kuzushiji, they often fail to consider various types of noise, such as document degradation and seals, which significantly affect recognition accuracy. To the best of our knowledge, no existing dataset specifically addresses these challenges. To address this gap, we introduce the Degraded Kuzushiji Documents with Seals (DKDS) dataset as a new benchmark for related tasks. We describe the dataset construction process, which required the assistance of a trained Kuzushiji expert, and define two benchmark tracks: (1) text and seal detection and (2) document binarization. For the text and seal detection track, we provide baseline results using several recent versions of the You Only Look Once (YOLO) models for detecting Kuzushiji characters and seals. For the document binarization track, we present baseline results from traditional binarization algorithms, traditional algorithms combined with K-means clustering, two state-of-the-art (SOTA) Generative Adversarial Network (GAN) methods, as well as our Conditional GAN (cGAN) baseline. The DKDS dataset and the implementation code for baseline methods are available at https://ruiyangju.github.io/DKDS.
>
---
#### [replaced 016] WildFit: Autonomous In-situ Model Adaptation for Resource-Constrained IoT Systems
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2409.07796v4](https://arxiv.org/pdf/2409.07796v4)**

> **作者:** Mohammad Mehdi Rastikerdar; Jin Huang; Hui Guan; Deepak Ganesan
>
> **备注:** Accepted by ACM SenSys 2026
>
> **摘要:** Resource-constrained IoT devices increasingly rely on deep learning models, however, these models experience significant accuracy drops due to domain shifts when encountering variations in lighting, weather, and seasonal conditions. While cloud-based retraining can address this issue, many IoT deployments operate with limited connectivity and energy constraints, making traditional fine-tuning approaches impractical. We explore this challenge through the lens of wildlife ecology, where camera traps must maintain accurate species classification across changing seasons, weather, and habitats without reliable connectivity. We introduce WildFit, an autonomous in-situ adaptation framework that leverages the key insight that background scenes change more frequently than the visual characteristics of monitored species. WildFit combines background-aware synthesis to generate training samples on-device with drift-aware fine-tuning that triggers model updates only when necessary to conserve resources. Our background-aware synthesis surpasses efficient baselines by 7.3% and diffusion models by 3.0% while being orders of magnitude faster, our drift-aware fine-tuning achieves Pareto optimality with 50% fewer updates and 1.5% higher accuracy, and the end-to-end system outperforms domain adaptation approaches by 20-35% while consuming only 11.2 Wh over 37 days-enabling battery-powered deployment.
>
---
#### [replaced 017] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12142v2](https://arxiv.org/pdf/2511.12142v2)**

> **作者:** Seokwon Song; Minsu Park; Gunhee Kim
>
> **备注:** AAAI 2026; code is available at https://github.com/seokwon99/MAVIS
>
> **摘要:** Source attribution aims to enhance the reliability of AI-generated answers by including references for each statement, helping users validate the provided answers. However, existing work has primarily focused on text-only scenario and largely overlooked the role of multimodality. We introduce MAVIS, the first benchmark designed to evaluate multimodal source attribution systems that understand user intent behind visual questions, retrieve multimodal evidence, and generate long-form answers with citations. Our dataset comprises 157K visual QA instances, where each answer is annotated with fact-level citations referring to multimodal documents. We develop fine-grained automatic metrics along three dimensions of informativeness, groundedness, and fluency, and demonstrate their strong correlation with human judgments. Our key findings are threefold: (1) LVLMs with multimodal RAG generate more informative and fluent answers than unimodal RAG, but they exhibit weaker groundedness for image documents than for text documents, a gap amplified in multimodal settings. (2) Given the same multimodal documents, there is a trade-off between informativeness and groundedness across different prompting methods. (3) Our proposed method highlights mitigating contextual bias in interpreting image documents as a crucial direction for future research.
>
---
#### [replaced 018] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11688v2](https://arxiv.org/pdf/2511.11688v2)**

> **作者:** Aihua Zhu; Rui Su; Qinglin Zhao; Li Feng; Meng Shen; Shibo He
>
> **备注:** Preprint, accepted to AAAI 2026
>
> **摘要:** Diffusion probabilistic models have set a new standard for generative fidelity but are hindered by a slow iterative sampling process. A powerful training-free strategy to accelerate this process is Schedule Optimization, which aims to find an optimal distribution of timesteps for a fixed and small Number of Function Evaluations (NFE) to maximize sample quality. To this end, a successful schedule optimization method must adhere to four core principles: effectiveness, adaptivity, practical robustness, and computational efficiency. However, existing paradigms struggle to satisfy these principles simultaneously, motivating the need for a more advanced solution. To overcome these limitations, we propose the Hierarchical-Schedule-Optimizer (HSO), a novel and efficient bi-level optimization framework. HSO reframes the search for a globally optimal schedule into a more tractable problem by iteratively alternating between two synergistic levels: an upper-level global search for an optimal initialization strategy and a lower-level local optimization for schedule refinement. This process is guided by two key innovations: the Midpoint Error Proxy (MEP), a solver-agnostic and numerically stable objective for effective local optimization, and the Spacing-Penalized Fitness (SPF) function, which ensures practical robustness by penalizing pathologically close timesteps. Extensive experiments show that HSO sets a new state-of-the-art for training-free sampling in the extremely low-NFE regime. For instance, with an NFE of just 5, HSO achieves a remarkable FID of 11.94 on LAION-Aesthetics with Stable Diffusion v2.1. Crucially, this level of performance is attained not through costly retraining, but with a one-time optimization cost of less than 8 seconds, presenting a highly practical and efficient paradigm for diffusion model acceleration.
>
---
#### [replaced 019] Unified Semantic Transformer for 3D Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14364v2](https://arxiv.org/pdf/2512.14364v2)**

> **作者:** Sebastian Koch; Johanna Wald; Hidenobu Matsuki; Pedro Hermosilla; Timo Ropinski; Federico Tombari
>
> **备注:** Project page: https://unite-page.github.io/
>
> **摘要:** Holistic 3D scene understanding involves capturing and parsing unstructured 3D environments. Due to the inherent complexity of the real world, existing models have predominantly been developed and limited to be task-specific. We introduce UNITE, a Unified Semantic Transformer for 3D scene understanding, a novel feed-forward neural network that unifies a diverse set of 3D semantic tasks within a single model. Our model operates on unseen scenes in a fully end-to-end manner and only takes a few seconds to infer the full 3D semantic geometry. Our approach is capable of directly predicting multiple semantic attributes, including 3D scene segmentation, instance embeddings, open-vocabulary features, as well as affordance and articulations, solely from RGB images. The method is trained using a combination of 2D distillation, heavily relying on self-supervision and leverages novel multi-view losses designed to ensure 3D view consistency. We demonstrate that UNITE achieves state-of-the-art performance on several different semantic tasks and even outperforms task-specific models, in many cases, surpassing methods that operate on ground truth 3D geometry. See the project website at unite-page.github.io
>
---
#### [replaced 020] SpatialVID: A Large-Scale Video Dataset with Spatial Annotations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09676v2](https://arxiv.org/pdf/2509.09676v2)**

> **作者:** Jiahao Wang; Yufeng Yuan; Rujie Zheng; Youtian Lin; Jian Gao; Lin-Zhuo Chen; Yajie Bao; Yi Zhang; Chang Zeng; Yanxi Zhou; Xiao-Xiao Long; Hao Zhu; Zhaoxiang Zhang; Xun Cao; Yao Yao
>
> **备注:** Project page: https://nju-3dv.github.io/projects/SpatialVID/
>
> **摘要:** Significant progress has been made in spatial intelligence, spanning both spatial reconstruction and world exploration. However, the scalability and real-world fidelity of current models remain severely constrained by the scarcity of large-scale, high-quality training data. While several datasets provide camera pose information, they are typically limited in scale, diversity, and annotation richness, particularly for real-world dynamic scenes with ground-truth camera motion. To this end, we collect SpatialVID, a dataset consists of a large corpus of in-the-wild videos with diverse scenes, camera movements and dense 3D annotations such as per-frame camera poses, depth, and motion instructions. Specifically, we collect more than 21,000 hours of raw videos, and process them into 2.7 million clips through a hierarchical filtering pipeline, totaling 7,089 hours of dynamic content. A subsequent annotation pipeline enriches these clips with detailed spatial and semantic information, including camera poses, depth maps, dynamic masks, structured captions, and serialized motion instructions. Analysis of SpatialVID's data statistics reveals a richness and diversity that directly fosters improved model generalization and performance, establishing it as a key asset for the video and 3D vision research community.
>
---
#### [replaced 021] Meta-learners for few-shot weakly-supervised optic disc and cup segmentation on fundus images
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15061v2](https://arxiv.org/pdf/2512.15061v2)**

> **作者:** Pandega Abyan Zumarsyah; Igi Ardiyanto; Hanung Adi Nugroho
>
> **备注:** Published in Computers in Biology and Medicine
>
> **摘要:** This study develops meta-learners for few-shot weakly-supervised segmentation (FWS) to address the challenge of optic disc (OD) and optic cup (OC) segmentation for glaucoma diagnosis with limited labeled fundus images. We significantly improve existing meta-learners by introducing Omni meta-training which balances data usage and diversifies the number of shots. We also develop their efficient versions that reduce computational costs. In addition, we develop sparsification techniques that generate more customizable and representative scribbles and other sparse labels. After evaluating multiple datasets, we find that Omni and efficient versions outperform the original versions, with the best meta-learner being Efficient Omni ProtoSeg (EO-ProtoSeg). It achieves intersection over union (IoU) scores of 88.15% for OD and 71.17% for OC on the REFUGE dataset using just one sparsely labeled image, outperforming few-shot and semi-supervised methods which require more labeled images. Its best performance reaches 86.80% for OD and 71.78%for OC on DRISHTIGS, 88.21% for OD and 73.70% for OC on REFUGE, 80.39% for OD and 52.65% for OC on REFUGE. EO-ProtoSeg is comparable to unsupervised domain adaptation methods yet much lighter with less than two million parameters and does not require any retraining.
>
---
#### [replaced 022] Memory Backdoor Attacks on Neural Networks
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.14516v2](https://arxiv.org/pdf/2411.14516v2)**

> **作者:** Eden Luzon; Guy Amit; Roy Weiss; Torsten Kraub; Alexandra Dmitrienko; Yisroel Mirsky
>
> **摘要:** Neural networks are often trained on proprietary datasets, making them attractive attack targets. We present a novel dataset extraction method leveraging an innovative training time backdoor attack, allowing a malicious federated learning server to systematically and deterministically extract complete client training samples through a simple indexing process. Unlike prior techniques, our approach guarantees exact data recovery rather than probabilistic reconstructions or hallucinations, provides precise control over which samples are memorized and how many, and shows high capacity and robustness. Infected models output data samples when they receive a patternbased index trigger, enabling systematic extraction of meaningful patches from each clients local data without disrupting global model utility. To address small model output sizes, we extract patches and then recombined them. The attack requires only a minor modification to the training code that can easily evade detection during client-side verification. Hence, this vulnerability represents a realistic FL supply-chain threat, where a malicious server can distribute modified training code to clients and later recover private data from their updates. Evaluations across classifiers, segmentation models, and large language models demonstrate that thousands of sensitive training samples can be recovered from client models with minimal impact on task performance, and a clients entire dataset can be stolen after multiple FL rounds. For instance, a medical segmentation dataset can be extracted with only a 3 percent utility drop. These findings expose a critical privacy vulnerability in FL systems, emphasizing the need for stronger integrity and transparency in distributed training pipelines.
>
---
#### [replaced 023] Expert Switching for Robust AAV Landing: A Dual-Detector Framework in Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向AAV视觉着陆任务，解决GPS拒止/恶劣视觉下因尺度剧烈变化导致单模型检测鲁棒性差的问题。提出双专家YOLOv8框架：分别专精远距小目标与近距高精度检测，通过几何门控动态路由选择最优专家，显著提升着陆稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2512.14054v2](https://arxiv.org/pdf/2512.14054v2)**

> **作者:** Humaira Tasnim; Ashik E Rasul; Bruce Jo; Hyung-Jin Yoon
>
> **备注:** To be Published in AIAA SciTech 2026
>
> **摘要:** Reliable helipad detection is essential for Autonomous Aerial Vehicle (AAV) landing, especially under GPS-denied or visually degraded conditions. While modern detectors such as YOLOv8 offer strong baseline performance, single-model pipelines struggle to remain robust across the extreme scale transitions that occur during descent, where helipads appear small at high altitude and large near touchdown. To address this limitation, we propose a scale-adaptive dual-expert perception framework that decomposes the detection task into far-range and close-range regimes. Two YOLOv8 experts are trained on scale-specialized versions of the HelipadCat dataset, enabling one model to excel at detecting small, low-resolution helipads and the other to provide high-precision localization when the target dominates the field of view. During inference, both experts operate in parallel, and a geometric gating mechanism selects the expert whose prediction is most consistent with the AAV's viewpoint. This adaptive routing prevents the degradation commonly observed in single-detector systems when operating across wide altitude ranges. The dual-expert perception module is evaluated in a closed-loop landing environment that integrates CARLA's photorealistic rendering with NASA's GUAM flight-dynamics engine. Results show substantial improvements in alignment stability, landing accuracy, and overall robustness compared to single-detector baselines. By introducing a scale-aware expert routing strategy tailored to the landing problem, this work advances resilient vision-based perception for autonomous descent and provides a foundation for future multi-expert AAV frameworks.
>
---
#### [replaced 024] Scaling Laws for Black box Adversarial Attacks
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16782v4](https://arxiv.org/pdf/2411.16782v4)**

> **作者:** Chuan Liu; Huanran Chen; Yichi Zhang; Jun Zhu; Yinpeng Dong
>
> **摘要:** Adversarial examples exhibit cross-model transferability, enabling threatening black-box attacks on commercial models. Model ensembling, which attacks multiple surrogate models, is a known strategy to improve this transferability. However, prior studies typically use small, fixed ensembles, which leaves open an intriguing question of whether scaling the number of surrogate models can further improve black-box attacks. In this work, we conduct the first large-scale empirical study of this question. We show that by resolving gradient conflict with advanced optimizers, we discover a robust and universal log-linear scaling law through both theoretical analysis and empirical evaluations: the Attack Success Rate (ASR) scales linearly with the logarithm of the ensemble size $T$. We rigorously verify this law across standard classifiers, SOTA defenses, and MLLMs, and find that scaling distills robust, semantic features of the target class. Consequently, we apply this fundamental insight to benchmark SOTA MLLMs. This reveals both the attack's devastating power and a clear robustness hierarchy: we achieve 80\%+ transfer attack success rate on proprietary models like GPT-4o, while also highlighting the exceptional resilience of Claude-3.5-Sonnet. Our findings urge a shift in focus for robustness evaluation: from designing intricate algorithms on small ensembles to understanding the principled and powerful threat of scaling.
>
---
#### [replaced 025] CompEvent: Complex-valued Event-RGB Fusion for Low-light Video Enhancement and Deblurring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14469v2](https://arxiv.org/pdf/2511.14469v2)**

> **作者:** Mingchen Zhong; Xin Lu; Dong Li; Senyan Xu; Ruixuan Jiang; Xueyang Fu; Baocai Yin
>
> **摘要:** Low-light video deblurring poses significant challenges in applications like nighttime surveillance and autonomous driving due to dim lighting and long exposures. While event cameras offer potential solutions with superior low-light sensitivity and high temporal resolution, existing fusion methods typically employ staged strategies, limiting their effectiveness against combined low-light and motion blur degradations. To overcome this, we propose CompEvent, a complex neural network framework enabling holistic full-process fusion of event data and RGB frames for enhanced joint restoration. CompEvent features two core components: 1) Complex Temporal Alignment GRU, which utilizes complex-valued convolutions and processes video and event streams iteratively via GRU to achieve temporal alignment and continuous fusion; and 2) Complex Space-Frequency Learning module, which performs unified complex-valued signal processing in both spatial and frequency domains, facilitating deep fusion through spatial structures and system-level characteristics. By leveraging the holistic representation capability of complex-valued neural networks, CompEvent achieves full-process spatiotemporal fusion, maximizes complementary learning between modalities, and significantly strengthens low-light video deblurring capability. Extensive experiments demonstrate that CompEvent outperforms SOTA methods in addressing this challenging task.
>
---
#### [replaced 026] Diffusion-Based Restoration for Multi-Modal 3D Object Detection in Adverse Weather
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.13107v2](https://arxiv.org/pdf/2512.13107v2)**

> **作者:** Zhijian He; Feifei Liu; Yuwei Li; Zhanpeng Luo; Jintao Cheng; Xieyuanli Chen; Xiaoyu Tang
>
> **摘要:** Multi-modal 3D object detection is important for reliable perception in robotics and autonomous driving. However, its effectiveness remains limited under adverse weather conditions due to weather-induced distortions and misalignment between different data modalities. In this work, we propose DiffFusion, a novel framework designed to enhance robustness in challenging weather through diffusion-based restoration and adaptive cross-modal fusion. Our key insight is that diffusion models possess strong capabilities for denoising and generating data that can adapt to various weather conditions. Building on this, DiffFusion introduces Diffusion-IR restoring images degraded by weather effects and Point Cloud Restoration (PCR) compensating for corrupted LiDAR data using image object cues. To tackle misalignments between two modalities, we develop Bidirectional Adaptive Fusion and Alignment Module (BAFAM). It enables dynamic multi-modal fusion and bidirectional bird's-eye view (BEV) alignment to maintain consistent spatial correspondence. Extensive experiments on three public datasets show that DiffFusion achieves state-of-the-art robustness under adverse weather while preserving strong clean-data performance. Zero-shot results on the real-world DENSE dataset further validate its generalization. The implementation of our DiffFusion will be released as open-source.
>
---
#### [replaced 027] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23334v2](https://arxiv.org/pdf/2511.23334v2)**

> **作者:** Yu Zhang; Jingyi Liu; Yiwei Shi; Qi Zhang; Duoqian Miao; Changwei Wang; Longbing Cao
>
> **摘要:** Visual AutoRegressive modeling (VAR) based on next-scale prediction has revitalized autoregressive visual generation. Although its full-context dependency, i.e., modeling all previous scales for next-scale prediction, facilitates more stable and comprehensive representation learning by leveraging complete information flow, the resulting computational inefficiency and substantial overhead severely hinder VAR's practicality and scalability. This motivates us to develop a new VAR model with better performance and efficiency without full-context dependency. To address this, we reformulate VAR as a non-full-context Markov process, proposing Markov-VAR. It is achieved via Markovian Scale Prediction: we treat each scale as a Markov state and introduce a sliding window that compresses certain previous scales into a compact history vector to compensate for historical information loss owing to non-full-context dependency. Integrating the history vector with the Markov state yields a representative dynamic state that evolves under a Markov process. Extensive experiments demonstrate that Markov-VAR is extremely simple yet highly effective: Compared to VAR on ImageNet, Markov-VAR reduces FID by 10.5% (256 $\times$ 256) and decreases peak memory consumption by 83.8% (1024 $\times$ 1024). We believe that Markov-VAR can serve as a foundation for future research on visual autoregressive generation and other downstream tasks.
>
---
#### [replaced 028] MMRel: Benchmarking Relation Understanding in Multi-Modal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.09121v3](https://arxiv.org/pdf/2406.09121v3)**

> **作者:** Jiahao Nie; Gongjie Zhang; Wenbin An; Yun Xing; Yap-Peng Tan; Alex C. Kot; Shijian Lu
>
> **摘要:** Though Multi-modal Large Language Models (MLLMs) have recently achieved significant progress, they often struggle to understand diverse and complicated inter-object relations. Specifically, the lack of large-scale and high-quality relation data has greatly hindered the progress of MLLMs in various vision-language perception tasks. We attempt to address this challenge by contributing the Multi-Modal Relation Understanding benchmark (MMRel), which features large-scale, high-quality, and diverse data on inter-object relations. MMRel has three distinctive attributes: (i) it contains 22,500 question-answer pairs spanning three distinct domains and around 400 relations, ensuring both scale and diversity; (ii) it provides manually verified, high-quality labels to ensure exceptional annotation accuracy; and (iii) it includes adversarial cases with highly unusual relations, offering a challenging setting for evaluating relation hallucination. These features make MMRel ideal for evaluating MLLMs on relation understanding, as well as for fine-tuning MLLMs to enhance relation comprehension capability. Extensive experiments on 28 MLLMs demonstrate the effectiveness of MMRel in both evaluating and enhancing MLLMs' relation understanding, and the accompanying analyses provide insights for future research. The benchmark has been made publicly available at: https://niejiahao1998.github.io/MMRel
>
---
#### [replaced 029] Null-LoRA: Low-Rank Adaptation on Null Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15233v2](https://arxiv.org/pdf/2512.15233v2)**

> **作者:** Yi Zhang; Yulei Kang; Haoxuan Chen; Jinxuan Li; Jian-Fang Hu
>
> **摘要:** Parameter-efficient fine-tuning methods have gained considerable popularity for adapting large-scale models to downstream tasks, particularly LoRA and its variants. Existing methods perform low-rank adaptation over the full parameter space. However, fine-tuning within a subspace can achieve comparable effectiveness. Inspired by the observation that pre-trained models possess non-trivial null spaces, we propose Null-space based Low-Rank Adaptation (Null-LoRA). Null-LoRA effectively reduces redundancy and enhances effective rank by freezing portions of the low-rank matrices. To further improve parameter efficiency, Null-LoRA constrains the entire incremental update within the null space, maximizing the utilization of incremental updates to adapt to new task paradigms. Null-LoRA surpasses the state of the art with fewer parameters in extensive experiments across image-text retrieval and visual question answering tasks.
>
---
#### [replaced 030] DAFM: Dynamic Adaptive Fusion for Multi-Model Collaboration in Composed Image Retrieval
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05020v2](https://arxiv.org/pdf/2511.05020v2)**

> **作者:** Yawei Cai; Jiapeng Mi; Nan Ji; Haotian Rong; Yawei Zhang; Zhangti Li; Wenbin Guo; Rensong Xie
>
> **备注:** We discovered an error that affects the main conclusions, so we decided to withdraw the paper
>
> **摘要:** Composed Image Retrieval (CIR) is a cross-modal task that aims to retrieve target images from large-scale databases using a reference image and a modification text. Most existing methods rely on a single model to perform feature fusion and similarity matching. However, this paradigm faces two major challenges. First, one model alone can't see the whole picture and the tiny details at the same time; it has to handle different tasks with the same weights, so it often misses the small but important links between image and text. Second, the absence of dynamic weight allocation prevents adaptive leveraging of complementary model strengths, so the resulting embedding drifts away from the target and misleads the nearest-neighbor search in CIR. To address these limitations, we propose Dynamic Adaptive Fusion (DAFM) for multi-model collaboration in CIR. Rather than optimizing a single method in isolation, DAFM exploits the complementary strengths of heterogeneous models and adaptively rebalances their contributions. This not only maximizes retrieval accuracy but also ensures that the performance gains are independent of the fusion order, highlighting the robustness of our approach. Experiments on the CIRR and FashionIQ benchmarks demonstrate consistent improvements. Our method achieves a Recall@10 of 93.21 and an Rmean of 84.43 on CIRR, and an average Rmean of 67.48 on FashionIQ, surpassing recent strong baselines by up to 4.5%. These results confirm that dynamic multi-model collaboration provides an effective and general solution for CIR.
>
---
#### [replaced 031] From Frames to Clips: Training-free Adaptive Key Clip Selection for Long-Form Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.02262v2](https://arxiv.org/pdf/2510.02262v2)**

> **作者:** Guangyu Sun; Archit Singhal; Burak Uzkent; Mubarak Shah; Chen Chen; Garin Kessler
>
> **摘要:** Video Large Language Models (VLMs) have achieved strong performance on various vision-language tasks, yet their practical use is limited by the massive number of visual tokens produced from raw video frames, which quickly exhausts the model's context window. Existing solutions mitigate this issue by selecting a sparse set of frames, but such frame-wise selection discards essential temporal dynamics in long-form videos, leading to suboptimal reasoning about motion and event continuity. In this work, we systematically examine the role of temporal information and show that extending selection from isolated key frames to temporally coherent key clips improves video understanding. To maintain a fixed computational budget while accommodating the larger token footprint of clips, we introduce frame resolution as a controllable factor in frame selection, enabling a trade-off between spatial resolution and clip length. Building on this idea, we propose an adaptive clip length module that dynamically balances these factors to ensure a constant token count per video. Experiments on three long-form video benchmarks demonstrate that our training-free approach, F2C, outperforms uniform sampling by up to 8.1%, 5.6%, and 10.3% on Video-MME, LongVideoBench, and MLVU, respectively. These results highlight the importance of preserving temporal coherence in frame selection and provide a practical pathway for scaling VLMs to real-world video understanding applications. Project webpage is available at https://guangyusun.com/f2c .
>
---
#### [replaced 032] ConsistTalk: Intensity Controllable Temporally Consistent Talking Head Generation with Diffusion Noise Search
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06833v2](https://arxiv.org/pdf/2511.06833v2)**

> **作者:** Zhenjie Liu; Jianzhang Lu; Renjie Lu; Cong Liang; Shangfei Wang
>
> **备注:** AAAI26 poster
>
> **摘要:** Recent advancements in video diffusion models have significantly enhanced audio-driven portrait animation. However, current methods still suffer from flickering, identity drift, and poor audio-visual synchronization. These issues primarily stem from entangled appearance-motion representations and unstable inference strategies. In this paper, we introduce \textbf{ConsistTalk}, a novel intensity-controllable and temporally consistent talking head generation framework with diffusion noise search inference. First, we propose \textbf{an optical flow-guided temporal module (OFT)} that decouples motion features from static appearance by leveraging facial optical flow, thereby reducing visual flicker and improving temporal consistency. Second, we present an \textbf{Audio-to-Intensity (A2I) model} obtained through multimodal teacher-student knowledge distillation. By transforming audio and facial velocity features into a frame-wise intensity sequence, the A2I model enables joint modeling of audio and visual motion, resulting in more natural dynamics. This further enables fine-grained, frame-wise control of motion dynamics while maintaining tight audio-visual synchronization. Third, we introduce a \textbf{diffusion noise initialization strategy (IC-Init)}. By enforcing explicit constraints on background coherence and motion continuity during inference-time noise search, we achieve better identity preservation and refine motion dynamics compared to the current autoregressive strategy. Extensive experiments demonstrate that ConsistTalk significantly outperforms prior methods in reducing flicker, preserving identity, and delivering temporally stable, high-fidelity talking head videos.
>
---
#### [replaced 033] Self-localization on a 3D map by fusing global and local features from a monocular camera
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人自定位任务，旨在解决单目相机在含动态障碍物场景下3D地图定位精度低的问题。提出融合CNN（提取局部特征）与Vision Transformer（提取全局特征）的新方法，显著提升定位精度，尤其在动态场景下效果更优。**

- **链接: [https://arxiv.org/pdf/2510.26170v2](https://arxiv.org/pdf/2510.26170v2)**

> **作者:** Satoshi Kikuchi; Masaya Kato; Tsuyoshi Tasaki
>
> **摘要:** Self-localization on a 3D map by using an inexpensive monocular camera is required to realize autonomous driving. Self-localization based on a camera often uses a convolutional neural network (CNN) that can extract local features that are calculated by nearby pixels. However, when dynamic obstacles, such as people, are present, CNN does not work well. This study proposes a new method combining CNN with Vision Transformer, which excels at extracting global features that show the relationship of patches on whole image. Experimental results showed that, compared to the state-of-the-art method (SOTA), the accuracy improvement rate in a CG dataset with dynamic obstacles is 1.5 times higher than that without dynamic obstacles. Moreover, the self-localization error of our method is 20.1% smaller than that of SOTA on public datasets. Additionally, our robot using our method can localize itself with 7.51cm error on average, which is more accurate than SOTA.
>
---
#### [replaced 034] Core-Set Selection for Data-efficient Land Cover Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.01225v3](https://arxiv.org/pdf/2505.01225v3)**

> **作者:** Keiller Nogueira; Akram Zaytar; Wanli Ma; Ribana Roscher; Ronny Hansch; Caleb Robinson; Anthony Ortiz; Simone Nsutezo; Rahul Dodhia; Juan M. Lavista Ferres; Oktay Karakus; Paul L. Rosin
>
> **摘要:** The increasing accessibility of remotely sensed data and their potential to support large-scale decision-making have driven the development of deep learning models for many Earth Observation tasks. Traditionally, such models rely on large datasets. However, the common assumption that larger training datasets lead to better performance tends to overlook issues related to data redundancy, noise, and the computational cost of processing massive datasets. Effective solutions must therefore consider not only the quantity but also the quality of data. Towards this, in this paper, we introduce six basic core-set selection approaches -- that rely on imagery only, labels only, or a combination of both -- and investigate whether they can identify high-quality subsets of data capable of maintaining -- or even surpassing -- the performance achieved when using full datasets for remote sensing semantic segmentation. We benchmark such approaches against two traditional baselines on three widely used land-cover classification datasets (DFC2022, Vaihingen, and Potsdam) using two different architectures (SegFormer and U-Net), thus establishing a general baseline for future works. Our experiments show that all proposed methods consistently outperform the baselines across multiple subset sizes, with some approaches even selecting core sets that surpass training on all available data. Notably, on DFC2022, a selected subset comprising only 25% of the training data yields slightly higher SegFormer performance than training with the entire dataset. This result shows the importance and potential of data-centric learning for the remote sensing domain. The code is available at https://github.com/keillernogueira/data-centric-rs-classification/.
>
---
#### [replaced 035] VAEER: Visual Attention-Inspired Emotion Elicitation Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉情感诱发（VEE）任务，即预测图像引发的多标签情绪。为解决可解释性不足问题，提出VAEER框架：融合视觉注意力机制提取关键线索，结合情感知识图谱进行逐情绪推理，生成透明、情绪特异性理由，在多个基准上达到SOTA性能。**

- **链接: [https://arxiv.org/pdf/2505.24342v2](https://arxiv.org/pdf/2505.24342v2)**

> **作者:** Fanhang Man; Xiaoyue Chen; Huandong Wang; Baining Zhao; Han Li; Xinlei Chen
>
> **备注:** Currently under review as conference paper
>
> **摘要:** Images shared online strongly influence emotions and public well-being. Understanding the emotions an image elicits is therefore vital for fostering healthier and more sustainable digital communities, especially during public crises. We study Visual Emotion Elicitation (VEE), predicting the set of emotions that an image evokes in viewers. We introduce VAEER, an interpretable multi-label VEE framework that combines attention-inspired cue extraction with knowledge-grounded reasoning. VAEER isolates salient visual foci and contextual signals, aligns them with structured affective knowledge, and performs per-emotion inference to yield transparent, emotion-specific rationales. Across three heterogeneous benchmarks, including social imagery and disaster-related photos, VAEER achieves state-of-the-art results with up to 19% per-emotion improvements and a 12.3% average gain over strong CNN and VLM baselines. Our findings highlight interpretable multi-label emotion elicitation as a scalable foundation for responsible visual media analysis and emotionally sustainable online ecosystems.
>
---
#### [replaced 036] DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.12796v2](https://arxiv.org/pdf/2510.12796v2)**

> **作者:** Yingyan Li; Shuyao Shang; Weisong Liu; Bing Zhan; Haochen Wang; Yuqi Wang; Yuntao Chen; Xiaoman Wang; Yasong An; Chufeng Tang; Lu Hou; Lue Fan; Zhaoxiang Zhang
>
> **摘要:** Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases.
>
---
#### [replaced 037] From Engineering Diagrams to Graphs: Digitizing P&IDs with Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.13929v2](https://arxiv.org/pdf/2411.13929v2)**

> **作者:** Jan Marius Stürmer; Marius Graumann; Tobias Koch
>
> **备注:** \c{opyright}2025 IEEE. Published in the conference proceedings of the 2025 IEEE 12th International Conference on Data Science and Advanced Analytics (DSAA)
>
> **摘要:** Digitizing engineering diagrams like Piping and Instrumentation Diagrams (P&IDs) plays a vital role in maintainability and operational efficiency of process and hydraulic systems. Previous methods typically decompose the task into separate steps such as symbol detection and line detection, which can limit their ability to capture the structure in these diagrams. In this work, a transformer-based approach leveraging the Relationformer that addresses this limitation by jointly extracting symbols and their interconnections from P&IDs is introduced. To evaluate our approach and compare it to a modular digitization approach, we present the first publicly accessible benchmark dataset for P&ID digitization, annotated with graph-level ground truth. Experimental results on real-world diagrams show that our method significantly outperforms the modular baseline, achieving over 25% improvement in edge detection accuracy. This research contributes a reproducible evaluation framework and demonstrates the effectiveness of transformer models for structural understanding of complex engineering diagrams. The dataset is available under https://zenodo.org/records/14803338.
>
---
#### [replaced 038] Sparse-Tuning: Adapting Vision Transformers with Efficient Fine-tuning and Inference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.14700v3](https://arxiv.org/pdf/2405.14700v3)**

> **作者:** Ting Liu; Xuyang Liu; Liangtao Shi; Zunnan Xu; Yue Hu; Siteng Huang; Yi Xin; Bineng Zhong; Donglin Wang
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) has emerged as a popular solution for adapting pre-trained Vision Transformer (ViT) models to downstream applications by updating only a small subset of parameters. While current PEFT methods have achieved fine-tuning efficiency, they overlook the efficiency of computation and GPU memory during inference, falling short of practical requirements. To address this limitation, we propose Sparse-Tuning, an efficient and effective framework that leverages popular token sparsification (TS) techniques to reduce information redundancy in images and videos, thereby significantly improving computational and memory efficiency. However, TS often compromises performance due to inevitable information loss. To address this limitation, we further introduce Dense Adapters (DA) to compensate for the information losses incurred by token sparsification. DA integrates comprehensive token information from shallow layers into the retained tokens of deeper layers, ensuring minimal performance degradation. Through the integration of TS techniques and DA, Sparse-Tuning achieves a significant reduction in computation and memory overhead while maintaining performance. Empirical results on VTAB-1K, three image datasets, and two video datasets show that Sparse-Tuning reduces GFLOPs to 66\% of the original ViT-B while achieving state-of-the-art performance compared to full fine-tuning and other PEFT baselines.
>
---
#### [replaced 039] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出D2E框架，解决机器人具身智能预训练数据稀缺问题，利用桌面（尤其是游戏）交互数据进行可扩展预训练。工作包括：OWA工具包统一数据格式、Generalist-IDM实现跨游戏零样本预测与伪标签生成、VAPT模型迁移至物理操作与导航任务，显著提升LIBERO和CANVAS基准性能。**

- **链接: [https://arxiv.org/pdf/2510.05684v2](https://arxiv.org/pdf/2510.05684v2)**

> **作者:** Suhwan Choi; Jaeyoon Jung; Haebin Seong; Minchan Kim; Minyeong Kim; Yongjun Cho; Yoonshik Kim; Yubeen Park; Youngjae Yu; Yunsung Lee
>
> **摘要:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at https://worv-ai.github.io/d2e/
>
---
#### [replaced 040] FARM: Fine-Tuning Geospatial Foundation Models for Intra-Field Crop Yield Regression
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2510.26609v2](https://arxiv.org/pdf/2510.26609v2)**

> **作者:** Shayan Nejadshamsi; Yuanyuan Zhang; Shadi Zaki; Brock Porth; Lysa Porth; Vahab Khoshdel
>
> **摘要:** Accurate and timely crop yield prediction is crucial for global food security and modern agricultural management. Traditional methods often lack the scalability and granularity required for precision farming. This paper introduces FARM: Fine-tuning Agricultural Regression Models, a deep learning framework designed for high-resolution, intra-field canola yield prediction. FARM leverages a pre-trained, large-scale geospatial foundation model (Prithvi-EO-2.0-600M) and adapts it for a continuous regression task, transforming multi-temporal satellite imagery into dense, pixel-level (30 m) yield maps. Evaluated on a comprehensive dataset from the Canadian Prairies, FARM achieves a Root Mean Squared Error (RMSE) of 0.44 and an R^2 of 0.81. Using an independent high-resolution yield monitor dataset, we further show that fine-tuning FARM on limited ground-truth labels outperforms training the same architecture from scratch, confirming the benefit of pre-training on large, upsampled county-level data for data-scarce precision agriculture. These results represent improvement over baseline architectures like 3D-CNN and DeepYield, which highlight the effectiveness of fine-tuning foundation models for specialized agricultural applications. By providing a continuous, high-resolution output, FARM offers a more actionable tool for precision agriculture than conventional classification or county-level aggregation methods. This work validates a novel approach that bridges the gap between large-scale Earth observation and on-farm decision-making, offering a scalable solution for detailed agricultural monitoring.
>
---
#### [replaced 041] Generation is Required for Data-Efficient Perception
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.08854v2](https://arxiv.org/pdf/2512.08854v2)**

> **作者:** Jack Brady; Bernhard Schölkopf; Thomas Kipf; Simon Buchholz; Wieland Brendel
>
> **备注:** Preprint
>
> **摘要:** It has been hypothesized that human-level visual perception requires a generative approach in which internal representations result from inverting a decoder. Yet today's most successful vision models are non-generative, relying on an encoder that maps images to representations without decoder inversion. This raises the question of whether generation is, in fact, necessary for machines to achieve human-level visual perception. To address this, we study whether generative and non-generative methods can achieve compositional generalization, a hallmark of human perception. Under a compositional data generating process, we formalize the inductive biases required to guarantee compositional generalization in decoder-based (generative) and encoder-based (non-generative) methods. We then show theoretically that enforcing these inductive biases on encoders is generally infeasible using regularization or architectural constraints. In contrast, for generative methods, the inductive biases can be enforced straightforwardly, thereby enabling compositional generalization by constraining a decoder and inverting it. We highlight how this inversion can be performed efficiently, either online through gradient-based search or offline through generative replay. We examine the empirical implications of our theory by training a range of generative and non-generative methods on photorealistic image datasets. We find that, without the necessary inductive biases, non-generative methods often fail to generalize compositionally and require large-scale pretraining or added supervision to improve generalization. By comparison, generative methods yield significant improvements in compositional generalization, without requiring additional data, by leveraging suitable inductive biases on a decoder along with search and replay.
>
---
#### [replaced 042] BoostDream: Efficient Refining for High-Quality Text-to-3D Generation from Multi-View Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2401.16764v4](https://arxiv.org/pdf/2401.16764v4)**

> **作者:** Yonghao Yu; Shunan Zhu; Huai Qin; Haorui Li
>
> **摘要:** Witnessing the evolution of text-to-image diffusion models, significant strides have been made in text-to-3D generation. Currently, two primary paradigms dominate the field of text-to-3D: the feed-forward generation solutions, capable of swiftly producing 3D assets but often yielding coarse results, and the Score Distillation Sampling (SDS) based solutions, known for generating high-fidelity 3D assets albeit at a slower pace. The synergistic integration of these methods holds substantial promise for advancing 3D generation techniques. In this paper, we present BoostDream, a highly efficient plug-and-play 3D refining method designed to transform coarse 3D assets into high-quality. The BoostDream framework comprises three distinct processes: (1) We introduce 3D model distillation that fits differentiable representations from the 3D assets obtained through feed-forward generation. (2) A novel multi-view SDS loss is designed, which utilizes a multi-view aware 2D diffusion model to refine the 3D assets. (3) We propose to use prompt and multi-view consistent normal maps as guidance in refinement.Our extensive experiment is conducted on different differentiable 3D representations, revealing that BoostDream excels in generating high-quality 3D assets rapidly, overcoming the Janus problem compared to conventional SDS-based methods. This breakthrough signifies a substantial advancement in both the efficiency and quality of 3D generation processes.
>
---
#### [replaced 043] UniVCD: A New Method for Unsupervised Change Detection in the Open-Vocabulary Era
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.13089v2](https://arxiv.org/pdf/2512.13089v2)**

> **作者:** Ziqiang Zhu; Bowei Yang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Change detection (CD) identifies scene changes from multi-temporal observations and is widely used in urban development and environmental monitoring. Most existing CD methods rely on supervised learning, making performance strongly dataset-dependent and incurring high annotation costs; they typically focus on a few predefined categories and generalize poorly to diverse scenes. With the rise of vision foundation models such as SAM2 and CLIP, new opportunities have emerged to relax these constraints. We propose Unified Open-Vocabulary Change Detection (UniVCD), an unsupervised, open-vocabulary change detection method built on frozen SAM2 and CLIP. UniVCD detects category-agnostic changes across diverse scenes and imaging geometries without any labeled data or paired change images. A lightweight feature alignment module is introduced to bridge the spatially detailed representations from SAM2 and the semantic priors from CLIP, enabling high-resolution, semantically aware change estimation while keeping the number of trainable parameters small. On top of this, a streamlined post-processing pipeline is further introduced to suppress noise and pseudo-changes, improving the detection accuracy for objects with well-defined boundaries. Experiments on several public BCD (Binary Change Detection) and SCD (Semantic Change Detection) benchmarks show that UniVCD achieves consistently strong performance and matches or surpasses existing open-vocabulary CD methods in key metrics such as F1 and IoU. The results demonstrate that unsupervised change detection with frozen vision foundation models and lightweight multi-modal alignment is a practical and effective paradigm for open-vocabulary CD. Code and pretrained models will be released at https://github.com/Die-Xie/UniVCD.
>
---
#### [replaced 044] Automated Building Heritage Assessment Using Street-Level Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11486v3](https://arxiv.org/pdf/2508.11486v3)**

> **作者:** Kristina Dabrock; Tim Johansson; Anna Donarelli; Mikael Mangold; Noah Pflugradt; Jann Michael Weinand; Jochen Linßen
>
> **摘要:** Registration of heritage values in buildings is important to safeguard heritage values that can be lost in renovation and energy efficiency projects. However, registering heritage values is a cumbersome process. Novel artificial intelligence tools may improve efficiency in identifying heritage values in buildings compared to costly and time-consuming traditional inventories. In this study, OpenAI's large language model GPT was used to detect various aspects of cultural heritage value in facade images. Using GPT derived data and building register data, machine learning models were trained to classify multi-family and non-residential buildings in Stockholm, Sweden. Validation against a heritage expert-created inventory shows a macro F1-score of 0.71 using a combination of register data and features retrieved from GPT, and a score of 0.60 using only GPT-derived data. The methods presented can contribute to higher-quality datasets and support decision making.
>
---
#### [replaced 045] D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2507.05859v3](https://arxiv.org/pdf/2507.05859v3)**

> **作者:** Wenkang Zhang; Yan Zhao; Qiang Wang; Zhixin Xu; Li Song; Zhengxue Cheng
>
> **备注:** changes of some major results
>
> **摘要:** Free-Viewpoint Video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representation remains a major challenge. Existing dynamic 3D Gaussian Splatting methods couple reconstruction with optimization-dependent compression and customized motion formats, limiting generalization and standardization. To address this, we propose D-FCGS, a novel Feedforward Compression framework for Dynamic Gaussian Splatting. Key innovations include: (1) a standardized Group-of-Frames (GoF) structure with I-P coding, leveraging sparse control points to extract inter-frame motion tensors; (2) a dual prior-aware entropy model that fuses hyperprior and spatial-temporal priors for accurate rate estimation; (3) a control-point-guided motion compensation mechanism and refinement network to enhance view-consistent fidelity. Trained on Gaussian frames derived from multi-view videos, D-FCGS generalizes across diverse scenes in a zero-shot fashion. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 40 times compression compared to the baseline while preserving visual quality across viewpoints. This work advances feedforward compression of dynamic 3DGS, facilitating scalable FVV transmission and storage for immersive applications.
>
---
#### [replaced 046] Improved Segmentation of Polyps and Visual Explainability Analysis
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.18159v5](https://arxiv.org/pdf/2509.18159v5)**

> **作者:** Akwasi Asare; Thanh-Huy Nguyen; Ulas Bagci
>
> **摘要:** Colorectal cancer (CRC) remains one of the leading causes of cancer-related morbidity and mortality worldwide, with gastrointestinal (GI) polyps serving as critical precursors according to the World Health Organization (WHO). Early and accurate segmentation of polyps during colonoscopy is essential for reducing CRC progression, yet manual delineation is labor-intensive and prone to observer variability. Deep learning methods have demonstrated strong potential for automated polyp analysis, but their limited interpretability remains a barrier to clinical adoption. In this study, we present PolypSeg-GradCAM, an explainable deep learning framework that integrates a U-Net architecture with a pre-trained ResNet-34 backbone and Gradient-weighted Class Activation Mapping (Grad-CAM) for transparent polyp segmentation. To ensure rigorous benchmarking, the model was trained and evaluated using 5-Fold Cross-Validation on the Kvasir-SEG dataset of 1,000 annotated endoscopic images. Experimental results show a mean Dice coefficient of 0.8902 +/- 0.0125, a mean Intersection-over-Union (IoU) of 0.8023, and an Area Under the Receiver Operating Characteristic Curve (AUC-ROC) of 0.9722. Advanced quantitative analysis using an optimal threshold yielded a Sensitivity of 0.9058 and Precision of 0.9083. Additionally, Grad-CAM visualizations confirmed that predictions were guided by clinically relevant regions, offering insight into the model's decision-making process. This study demonstrates that integrating segmentation accuracy with interpretability can support the development of trustworthy AI-assisted colonoscopy tools.
>
---
#### [replaced 047] MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17509v2](https://arxiv.org/pdf/2505.17509v2)**

> **作者:** Shiji Zhao; Qihui Zhu; Shukun Xiong; Shouwei Ruan; Maoxun Yuan; Jialing Tao; Jiexi Liu; Ranjie Duan; Jie Zhang; Jie Zhang; Xingxing Wei
>
> **摘要:** Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.
>
---
#### [replaced 048] Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space
- **分类: cs.CV; cs.CL**

- **简介: 该论文属多模态推理任务，旨在解决现有MLLMs依赖显式分步推理、感知-推理交互不稳定及计算开销大的问题。提出DMLR框架，通过置信度引导的隐空间策略优化与动态视觉特征注入，实现隐式、动态的图文交织推理。**

- **链接: [https://arxiv.org/pdf/2512.12623v2](https://arxiv.org/pdf/2512.12623v2)**

> **作者:** Chengzhi Liu; Yuzhe Yang; Yue Fan; Qingyue Wei; Sheng Liu; Xin Eric Wang
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced cross-modal understanding and reasoning by incorporating Chain-of-Thought (CoT) reasoning in the semantic space. Building upon this, recent studies extend the CoT mechanism to the visual modality, enabling models to integrate visual information during reasoning through external tools or explicit image generation. However, these methods remain dependent on explicit step-by-step reasoning, unstable perception-reasoning interaction and notable computational overhead. Inspired by human cognition, we posit that thinking unfolds not linearly but through the dynamic interleaving of reasoning and perception within the mind. Motivated by this perspective, we propose DMLR, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning. Furthermore, a Dynamic Visual Injection Strategy is introduced, which retrieves the most relevant visual features at each latent think token and updates the set of best visual patches. The updated patches are then injected into latent think token to achieve dynamic visual-textual interleaving. Experiments across seven multimodal reasoning benchmarks and various model architectures demonstrate that DMLR significantly improves reasoning and perception performance while maintaining high inference efficiency.
>
---
#### [replaced 049] Stylized Synthetic Augmentation further improves Corruption Robustness
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.15675v2](https://arxiv.org/pdf/2512.15675v2)**

> **作者:** Georg Siedel; Rojan Regmi; Abhirami Anand; Weijia Shao; Silvia Vock; Andrey Morozov
>
> **备注:** Accepted at VISAPP 2026 conference
>
> **摘要:** This paper proposes a training data augmentation pipeline that combines synthetic image data with neural style transfer in order to address the vulnerability of deep vision models to common corruptions. We show that although applying style transfer on synthetic images degrades their quality with respect to the common FID metric, these images are surprisingly beneficial for model training. We conduct a systematic empirical analysis of the effects of both augmentations and their key hyperparameters on the performance of image classifiers. Our results demonstrate that stylization and synthetic data complement each other well and can be combined with popular rule-based data augmentation techniques such as TrivialAugment, while not working with others. Our method achieves state-of-the-art corruption robustness on several small-scale image classification benchmarks, reaching 93.54%, 74.9% and 50.86% robust accuracy on CIFAR-10-C, CIFAR-100-C and TinyImageNet-C, respectively
>
---
#### [replaced 050] VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12977v2](https://arxiv.org/pdf/2512.12977v2)**

> **作者:** Shengling Qin; Hao Yu; Chenxin Wu; Zheng Li; Yizhong Cao; Zhengyang Zhuge; Yuxin Zhou; Wentao Yao; Yi Zhang; Zhengheng Wang; Shuai Bai; Jianwei Zhang; Junyang Lin
>
> **摘要:** This paper presents VLCache, a cache reuse framework that exploits both Key-Value (KV) cache and encoder cache from prior multimodal inputs to eliminate costly recomputation when the same multimodal inputs recur. Unlike previous heuristic approaches, we formally identify the cumulative reuse error effect and demonstrate how to minimize the non-prefix cache reuse error effectively. We further analyze the varying importance of model layers and propose a dynamic, layer-aware recomputation strategy to balance accuracy and efficiency. Experimental results show that VLCache achieves an accuracy on par with full recomputation, while requiring only 2-5% of the tokens to compute, yielding 1.2x-16x TTFT speedups. We develop an experimental implementation of the proposed VLCache pipeline based on SGLang, enabling significantly faster inference in practical deployments.
>
---
#### [replaced 051] Video Reality Test: Can AI-Generated ASMR Videos fool VLMs and Humans?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13281v3](https://arxiv.org/pdf/2512.13281v3)**

> **作者:** Jiaqi Wang; Weijia Wu; Yi Zhan; Rui Zhao; Ming Hu; James Cheng; Wei Liu; Philip Torr; Kevin Qinghong Lin
>
> **备注:** Code is at https://github.com/video-reality-test/video-reality-test, page is at https://video-reality-test.github.io/
>
> **摘要:** Recent advances in video generation have produced vivid content that are often indistinguishable from real videos, making AI-generated video detection an emerging societal challenge. Prior AIGC detection benchmarks mostly evaluate video without audio, target broad narrative domains, and focus on classification solely. Yet it remains unclear whether state-of-the-art video generation models can produce immersive, audio-paired videos that reliably deceive humans and VLMs. To this end, we introduce Video Reality Test, an ASMR-sourced video benchmark suite for testing perceptual realism under tight audio-visual coupling, featuring the following dimensions: (i) Immersive ASMR video-audio sources. Built on carefully curated real ASMR videos, the benchmark targets fine-grained action-object interactions with diversity across objects, actions, and backgrounds. (ii) Peer-Review evaluation. An adversarial creator-reviewer protocol where video generation models act as creators aiming to fool reviewers, while VLMs serve as reviewers seeking to identify fakeness. Our experimental findings show: The best creator Veo3.1-Fast even fools most VLMs: the strongest reviewer (Gemini 2.5-Pro) achieves only 56% accuracy (random 50%), far below that of human experts (81.25%). Adding audio improves real-fake discrimination, yet superficial cues such as watermarks can still significantly mislead models. These findings delineate the current boundary of video generation realism and expose limitations of VLMs in perceptual fidelity and audio-visual consistency. Our code is available at https://github.com/video-reality-test/video-reality-test.
>
---
#### [replaced 052] Matérn Kernels for Tunable Implicit Surface Reconstruction
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2409.15466v3](https://arxiv.org/pdf/2409.15466v3)**

> **作者:** Maximilian Weiherer; Bernhard Egger
>
> **备注:** ICLR'25
>
> **摘要:** We propose to use the family of Matérn kernels for implicit surface reconstruction, building upon the recent success of kernel methods for 3D reconstruction of oriented point clouds. As we show from a theoretical and practical perspective, Matérn kernels have some appealing properties which make them particularly well suited for surface reconstruction -- outperforming state-of-the-art methods based on the arc-cosine kernel while being significantly easier to implement, faster to compute, and scalable. Being stationary, we demonstrate that Matérn kernels allow for tunable surface reconstruction in the same way as Fourier feature mappings help coordinate-based MLPs overcome spectral bias. Moreover, we theoretically analyze Matérn kernels' connection to SIREN networks as well as their relation to previously employed arc-cosine kernels. Finally, based on recently introduced Neural Kernel Fields, we present data-dependent Matérn kernels and conclude that especially the Laplace kernel (being part of the Matérn family) is extremely competitive, performing almost on par with state-of-the-art methods in the noise-free case while having a more than five times shorter training time.
>
---
#### [replaced 053] UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.20110v2](https://arxiv.org/pdf/2502.20110v2)**

> **作者:** Luigi Piccinelli; Christos Sakaridis; Yung-Hsu Yang; Mattia Segu; Siyuan Li; Wim Abbeloos; Luc Van Gool
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2403.18913
>
> **摘要:** Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepthV2, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE paradigm, UniDepthV2 directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepthV2 implements a self-promptable camera module predicting a dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles the camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. UniDepthV2 improves its predecessor UniDepth model via a new edge-guided loss which enhances the localization and sharpness of edges in the metric depth outputs, a revisited, simplified and more efficient architectural design, and an additional uncertainty-level output which enables downstream tasks requiring confidence. Thorough evaluations on ten depth datasets in a zero-shot regime consistently demonstrate the superior performance and generalization of UniDepthV2. Code and models are available at https://github.com/lpiccinelli-eth/UniDepth
>
---
#### [replaced 054] Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13654v2](https://arxiv.org/pdf/2511.13654v2)**

> **作者:** Pascal Zimmer; Ghassan Karame
>
> **备注:** To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026
>
> **摘要:** In this paper, we present the first detailed analysis of how training hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the training hyperparameter space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.
>
---
#### [replaced 055] GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15705v2](https://arxiv.org/pdf/2511.15705v2)**

> **作者:** Yikun Wang; Zuyan Liu; Ziyi Wang; Han Hu; Pengfei Liu; Yongming Rao
>
> **摘要:** Current research on agentic visual reasoning enables deep multimodal understanding but primarily focuses on image manipulation tools, leaving a gap toward more general-purpose agentic models. In this work, we revisit the geolocalization task, which requires not only nuanced visual grounding but also web search to confirm or refine hypotheses during reasoning. Since existing geolocalization benchmarks fail to meet the need for high-resolution imagery and the localization challenge for deep agentic reasoning, we curate GeoBench, a benchmark that includes photos and panoramas from around the world, along with a subset of satellite images of different cities to rigorously evaluate the geolocalization ability of agentic models. We also propose GeoVista, an agentic model that seamlessly integrates tool invocation within the reasoning loop, including an image-zoom-in tool to magnify regions of interest and a web-search tool to retrieve related web information. We develop a complete training pipeline for it, including a cold-start supervised fine-tuning (SFT) stage to learn reasoning patterns and tool-use priors, followed by a reinforcement learning (RL) stage to further enhance reasoning ability. We adopt a hierarchical reward to leverage multi-level geographical information and improve overall geolocalization performance. Experimental results show that GeoVista surpasses other open-source agentic models on the geolocalization task greatly and achieves performance comparable to closed-source models such as Gemini-2.5-flash and GPT-5 on most metrics.
>
---
#### [replaced 056] HyperET: Efficient Training in Hyperbolic Space for Multi-modal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20322v3](https://arxiv.org/pdf/2510.20322v3)**

> **作者:** Zelin Peng; Zhengqin Xu; Qingyang Liu; Xiaokang Yang; Wei Shen
>
> **备注:** Accepted by NeurIPS2025 (Oral)
>
> **摘要:** Multi-modal large language models (MLLMs) have emerged as a transformative approach for aligning visual and textual understanding. They typically require extremely high computational resources (e.g., thousands of GPUs) for training to achieve cross-modal alignment at multi-granularity levels. We argue that a key source of this inefficiency lies in the vision encoders they widely equip with, e.g., CLIP and SAM, which lack the alignment with language at multi-granularity levels. To address this issue, in this paper, we leverage hyperbolic space, which inherently models hierarchical levels and thus provides a principled framework for bridging the granularity gap between visual and textual modalities at an arbitrary granularity level. Concretely, we propose an efficient training paradigm for MLLMs, dubbed as HyperET, which can optimize visual representations to align with their textual counterparts at an arbitrary granularity level through dynamic hyperbolic radius adjustment in hyperbolic space. HyperET employs learnable matrices with Möbius multiplication operations, implemented via three effective configurations: diagonal scaling matrices, block-diagonal matrices, and banded matrices, providing a flexible yet efficient parametrization strategy. Comprehensive experiments across multiple MLLM benchmarks demonstrate that HyperET consistently improves both existing pre-training and fine-tuning MLLMs clearly with less than 1\% additional parameters. Code is available at https://github.com/godlin-sjtu/HyperET
>
---
#### [replaced 057] $\mathrm{D}^\mathrm{3}$-Predictor: Noise-Free Deterministic Diffusion for Dense Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.07062v3](https://arxiv.org/pdf/2512.07062v3)**

> **作者:** Changliang Xia; Chengyou Jia; Minnan Luo; Zhuohang Dang; Xin Shen; Bowen Ping
>
> **摘要:** Although diffusion models with strong visual priors have emerged as powerful dense prediction backboens, they overlook a core limitation: the stochastic noise at the core of diffusion sampling is inherently misaligned with dense prediction that requires a deterministic mapping from image to geometry. In this paper, we show that this stochastic noise corrupts fine-grained spatial cues and pushes the model toward timestep-specific noise objectives, consequently destroying meaningful geometric structure mappings. To address this, we introduce $\mathrm{D}^\mathrm{3}$-Predictor, a noise-free deterministic framework built by reformulating a pretrained diffusion model without stochasticity noise. Instead of relying on noisy inputs to leverage diffusion priors, $\mathrm{D}^\mathrm{3}$-Predictor views the pretrained diffusion network as an ensemble of timestep-dependent visual experts and self-supervisedly aggregates their heterogeneous priors into a single, clean, and complete geometric prior. Meanwhile, we utilize task-specific supervision to seamlessly adapt this noise-free prior to dense prediction tasks. Extensive experiments on various dense prediction tasks demonstrate that $\mathrm{D}^\mathrm{3}$-Predictor achieves competitive or state-of-the-art performance in diverse scenarios. In addition, it requires less than half the training data previously used and efficiently performs inference in a single step. Our code, data, and checkpoints are publicly available at https://x-gengroup.github.io/HomePage_D3-Predictor/.
>
---
#### [replaced 058] TransUNet-GradCAM: A Hybrid Transformer-U-Net with Self-Attention and Explainable Visualizations for Foot Ulcer Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03758v4](https://arxiv.org/pdf/2508.03758v4)**

> **作者:** Akwasi Asare; Mary Sagoe; Justice Williams Asare; Stephen Edward Moore
>
> **摘要:** Automated segmentation of diabetic foot ulcers (DFUs) plays a critical role in clinical diagnosis, therapeutic planning, and longitudinal wound monitoring. However, this task remains challenging due to the heterogeneous appearance, irregular morphology, and complex backgrounds associated with ulcer regions in clinical photographs. Traditional convolutional neural networks (CNNs), such as U-Net, provide strong localization capabilities but struggle to model long-range spatial dependencies due to their inherently limited receptive fields. To address this, we employ the TransUNet architecture, a hybrid framework that integrates the global attention mechanism of Vision Transformers (ViTs) into the U-Net structure. This combination allows the model to extract global contextual features while maintaining fine-grained spatial resolution. We trained the model on the public Foot Ulcer Segmentation Challenge (FUSeg) dataset using a robust augmentation pipeline and a hybrid loss function to mitigate class imbalance. On the internal validation set, the model achieved a Dice Similarity Coefficient (F1-score) of 0.8886 using an optimized threshold of 0.4843. Crucially, to assess generalizability, we performed external validation on two independent datasets: the AZH Wound Care Center dataset (n=278) and the Medetec dataset (n=152). Without any retraining, the model achieved Dice scores of 0.6209 and 0.7850, respectively, demonstrating robust zero-shot transferability to unseen clinical domains. Furthermore, clinical utility analysis revealed a strong correlation (Pearson r = 0.9749) between predicted and ground-truth wound areas. These outcomes demonstrate that our approach effectively integrates global and local feature extraction, offering a reliable, effective, and explainable solution for automated foot ulcer assessment.
>
---
#### [replaced 059] NeAR: Coupled Neural Asset-Renderer Stack
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18600v2](https://arxiv.org/pdf/2511.18600v2)**

> **作者:** Hong Li; Chongjie Ye; Houyuan Chen; Weiqing Xiao; Ziyang Yan; Lixing Xiao; Zhaoxi Chen; Jianfeng Xiang; Shaocong Xu; Xuhui Liu; Yikai Wang; Baochang Zhang; Xiaoguang Han; Jiaolong Yang; Hao Zhao
>
> **备注:** 20 pages, 19 figures. The project page: https://near-project.github.io/
>
> **摘要:** Neural asset authoring and neural rendering have traditionally evolved as disjoint paradigms: one generates digital assets for fixed graphics pipelines, while the other maps conventional assets to images. However, treating them as independent entities limits the potential for end-to-end optimization in fidelity and consistency. In this paper, we bridge this gap with NeAR, a Coupled Neural Asset--Renderer Stack. We argue that co-designing the asset representation and the renderer creates a robust "contract" for superior generation. On the asset side, we introduce the Lighting-Homogenized SLAT (LH-SLAT). Leveraging a rectified-flow model, NeAR lifts casually lit single images into a canonical, illumination-invariant latent space, effectively suppressing baked-in shadows and highlights. On the renderer side, we design a lighting-aware neural decoder tailored to interpret these homogenized latents. Conditioned on HDR environment maps and camera views, it synthesizes relightable 3D Gaussian splats in real-time without per-object optimization. We validate NeAR on four tasks: (1) G-buffer-based forward rendering, (2) random-lit reconstruction, (3) unknown-lit relighting, and (4) novel-view relighting. Extensive experiments demonstrate that our coupled stack outperforms state-of-the-art baselines in both quantitative metrics and perceptual quality. We hope this coupled asset-renderer perspective inspires future graphics stacks that view neural assets and renderers as co-designed components instead of independent entities.
>
---
#### [replaced 060] Radar-Guided Polynomial Fitting for Metric Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.17182v3](https://arxiv.org/pdf/2503.17182v3)**

> **作者:** Patrick Rim; Hyoungseob Park; Vadim Ezhov; Jeffrey Moon; Alex Wong
>
> **摘要:** We propose POLAR, a novel radar-guided depth estimation method that introduces polynomial fitting to efficiently transform scaleless depth predictions from pretrained monocular depth estimation (MDE) models into metric depth maps. Unlike existing approaches that rely on complex architectures or expensive sensors, our method is grounded in a fundamental insight: although MDE models often infer reasonable local depth structure within each object or local region, they may misalign these regions relative to one another, making a linear scale and shift (affine) transformation insufficient given three or more of these regions. To address this limitation, we use polynomial coefficients predicted from cheap, ubiquitous radar data to adaptively adjust predictions non-uniformly across depth ranges. In this way, POLAR generalizes beyond affine transformations and is able to correct such misalignments by introducing inflection points. Importantly, our polynomial fitting framework preserves structural consistency through a novel training objective that enforces local monotonicity via first-derivative regularization. POLAR achieves state-of-the-art performance across three datasets, outperforming existing methods by an average of 24.9% in MAE and 33.2% in RMSE, while also achieving state-of-the-art efficiency in terms of latency and computational cost.
>
---
#### [replaced 061] STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.15216v2](https://arxiv.org/pdf/2508.15216v2)**

> **作者:** Vipooshan Vipulananthan; Kumudu Mohottala; Kavindu Chinthana; Nimsara Paramulla; Charith D Chitraranjan
>
> **备注:** Published in IEEE Access (Early Access)
>
> **摘要:** Accident prediction and timely warnings play a key role in improving road safety by reducing the risk of injury to road users and minimizing property damage. Advanced Driver Assistance Systems (ADAS) are designed to support human drivers and are especially useful when they can anticipate potential accidents before they happen. While many existing systems depend on a range of sensors such as LiDAR, radar, and GPS, relying solely on dash-cam video input presents a more challenging but a more cost-effective and easily deployable solution. In this work, we incorporate better spatio-temporal features and aggregate them through a recurrent network to improve upon state-of-the-art graph neural networks for predicting accidents from dash-cam videos. Experiments using three publicly available datasets show that our proposed STAGNet model achieves higher average precision and mean time-to-collision values than previous methods, both when cross-validated on a given dataset and when trained and tested on different datasets.
>
---
#### [replaced 062] An Efficient Deep Learning Framework for Brain Stroke Diagnosis Using Computed Tomography Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.03558v3](https://arxiv.org/pdf/2507.03558v3)**

> **作者:** Md. Sabbir Hossen; Eshat Ahmed Shuvo; Shibbir Ahmed Arif; Pabon Shaha; Anichur Rahman; Md. Saiduzzaman; Fahmid Al Farid; Hezerul Abdul Karim; Abu Saleh Musa Miah
>
> **备注:** Preprint version. Submitted for peer review
>
> **摘要:** Brain stroke is a leading cause of mortality and long-term disability worldwide, underscoring the need for precise and rapid prediction techniques. Computed Tomography (CT) scan is considered one of the most effective methods for diagnosing brain strokes. Most stroke classification techniques use a single slice-level prediction mechanism, requiring radiologists to manually select the most critical CT slice from the original CT volume. Although clinical evaluations are often used in traditional diagnostic procedures, machine learning (ML) has opened up new avenues for improving stroke diagnosis. To supplement traditional diagnostic techniques, this study investigates machine learning models for early brain stroke prediction using CT scan images. This research proposes a novel machine learning approach to brain stroke detection, focusing on optimizing classification performance with pre-trained deep learning models and advanced optimization strategies. Pre-trained models, including DenseNet201, InceptionV3, MobileNetV2, ResNet50, and Xception, are used for feature extraction. Feature engineering techniques, including BFO, PCA, and LDA, further enhance model performance. These features are then classified using machine learning algorithms, including SVC, RF, XGB, DT, LR, KNN, and GNB. Our experiments demonstrate that the combination of MobileNetV2, LDA, and SVC achieved the highest classification accuracy of 97.93%, significantly outperforming other model-optimizer-classifier combinations. The results underline the effectiveness of integrating lightweight pre-trained models with robust optimization and classification techniques for brain stroke diagnosis.
>
---
#### [replaced 063] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22862v2](https://arxiv.org/pdf/2511.22862v2)**

> **作者:** Jiacheng Li; Songhe Feng
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Test-time adaptation (TTA) enables online model adaptation using only unlabeled test data, aiming to bridge the gap between source and target distributions. However, in multimodal scenarios, varying degrees of distribution shift across different modalities give rise to a complex coupling effect of unimodal shallow feature shift and cross-modal high-level semantic misalignment, posing a major obstacle to extending existing TTA methods to the multimodal field. To address this challenge, we propose a novel multimodal test-time adaptation (MMTTA) framework, termed as Bridging Modalities via Progressive Re-alignment (BriMPR). BriMPR, consisting of two progressively enhanced modules, tackles the coupling effect with a divide-and-conquer strategy. Specifically, we first decompose MMTTA into multiple unimodal feature alignment sub-problems. By leveraging the strong function approximation ability of prompt tuning, we calibrate the unimodal global feature distributions to their respective source distributions, so as to achieve the initial semantic re-alignment across modalities. Subsequently, we assign the credible pseudo-labels to combinations of masked and complete modalities, and introduce inter-modal instance-wise contrastive learning to further enhance the information interaction among modalities and refine the alignment. Extensive experiments on MMTTA tasks, including both corruption-based and real-world domain shift benchmarks, demonstrate the superiority of our method. Our source code is available at https://github.com/Luchicken/BriMPR.
>
---
#### [replaced 064] SlumpGuard: An AI-Powered Real-Time System for Automated Concrete Slump Prediction via Video Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.10171v3](https://arxiv.org/pdf/2507.10171v3)**

> **作者:** Youngmin Kim; Giyeong Oh; Kwangsoo Youm; Youngjae Yu
>
> **备注:** Our project page: https://winston1214.github.io/SlumpGuard/
>
> **摘要:** Concrete workability is essential for construction quality, with the slump test being the most widely used on-site method for its assessment. However, traditional slump testing is manual, time-consuming, and highly operator-dependent, making it unsuitable for continuous or real-time monitoring during placement. To address these limitations, we present SlumpGuard, an AI-powered vision system that analyzes the natural discharge flow from a mixer-truck chute using a single fixed camera. The system performs automatic chute detection, pouring-event identification, and video-based slump classification, enabling quality monitoring without sensors, hardware installation, or manual intervention. We introduce the system design, construct a site-replicated dataset of over 6,000 video clips, and report extensive evaluations demonstrating reliable chute localization, accurate pouring detection, and robust slump prediction under diverse field conditions. An expert study further reveals significant disagreement in human visual estimates, highlighting the need for automated assessment.
>
---
#### [replaced 065] CompareBench: A Benchmark for Visual Comparison Reasoning in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.22737v2](https://arxiv.org/pdf/2509.22737v2)**

> **作者:** Jie Cai; Kangning Yang; Lan Fu; Jiaming Ding; Jinlong Li; Huiming Sun; Daitao Xing; Jinglin Shen; Zibo Meng
>
> **摘要:** We introduce CompareBench, a benchmark for evaluating visual comparison reasoning in vision-language models (VLMs), a fundamental yet understudied skill. CompareBench consists of 1000 QA pairs across four tasks: quantity (600), temporal (100), geometric (200), and spatial (100). It is derived from two auxiliary datasets that we constructed: TallyBench (2000 counting images with QA) and HistCaps (515 historical images with bilingual captions). We evaluate both closed-source APIs (OpenAI, Gemini, Claude) and open-source models (Qwen2.5-VL and Qwen3-VL series). Results show clear scaling trends but also reveal critical limitations: even the strongest models consistently fail at temporal ordering and spatial relations, and they often make mistakes in basic counting and geometric comparisons that are trivial for humans. These findings demonstrate that visual comparison remains a systematic blind spot for current VLMs. By providing controlled, diverse, and diagnostic evaluation, CompareBench establishes a foundation for advancing more reliable multimodal reasoning.
>
---
#### [replaced 066] Percept, Chat, and then Adapt: Multimodal Knowledge Transfer of Foundation Models for Open-World Video Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2402.18951v2](https://arxiv.org/pdf/2402.18951v2)**

> **作者:** Boyu Chen; Siran Chen; Kunchang Li; Qinglin Xu; Yu Qiao; Yali Wang
>
> **备注:** 35 pages, 6 figures, 8 tables
>
> **摘要:** Open-world video recognition is challenging since traditional networks are not generalized well on complex environment variations. Alternatively, foundation models with rich knowledge have recently shown their generalization power. However, how to apply such knowledge has not been fully explored for open-world video recognition. To this end, we propose a generic knowledge transfer pipeline, which progressively exploits and integrates external multimodal knowledge from foundation models to boost open-world video recognition. We name it PCA, based on three stages of Percept, Chat, and Adapt. First, we perform Percept process to reduce the video domain gap and obtain external visual knowledge. Second, we generate rich linguistic semantics as external textual knowledge in Chat stage. Finally, we blend external multimodal knowledge in Adapt stage, by inserting multimodal knowledge adaptation modules into networks. We conduct extensive experiments on three challenging open-world video benchmarks, i.e., TinyVIRAT, ARID, and QV-Pipe. Our approach achieves state-of-the-art performance on all three datasets.
>
---
#### [replaced 067] V-Thinker: Interactive Thinking with Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.04460v2](https://arxiv.org/pdf/2511.04460v2)**

> **作者:** Runqi Qiao; Qiuna Tan; Minghan Yang; Guanting Dong; Peiqing Yang; Shiqiang Lang; Enhui Wan; Xiaowan Wang; Yida Xu; Lan Yang; Chong Sun; Chen Li; Jing Lyu; Honggang Zhang
>
> **备注:** Working in progress
>
> **摘要:** Empowering Large Multimodal Models (LMMs) to deeply integrate image interaction with long-horizon reasoning capabilities remains a long-standing challenge in this field. Recent advances in vision-centric reasoning explore a promising "Thinking with Images" paradigm for LMMs, marking a shift from image-assisted reasoning to image-interactive thinking. While this milestone enables models to focus on fine-grained image regions, progress remains constrained by limited visual tool spaces and task-specific workflow designs. To bridge this gap, we present V-Thinker, a general-purpose multimodal reasoning assistant that enables interactive, vision-centric thinking through end-to-end reinforcement learning. V-Thinker comprises two key components: (1) a Data Evolution Flywheel that automatically synthesizes, evolves, and verifies interactive reasoning datasets across three dimensions-diversity, quality, and difficulty; and (2) a Visual Progressive Training Curriculum that first aligns perception via point-level supervision, then integrates interactive reasoning through a two-stage reinforcement learning framework. Furthermore, we introduce VTBench, an expert-verified benchmark targeting vision-centric interactive reasoning tasks. Extensive experiments demonstrate that V-Thinker consistently outperforms strong LMM-based baselines in both general and interactive reasoning scenarios, providing valuable insights for advancing image-interactive reasoning applications.
>
---
#### [replaced 068] SemanticBridge - A Dataset for 3D Semantic Segmentation of Bridges and Domain Gap Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15369v2](https://arxiv.org/pdf/2512.15369v2)**

> **作者:** Maximilian Kellner; Mariana Ferrandon Cervantes; Yuandong Pan; Ruodan Lu; Ioannis Brilakis; Alexander Reiterer
>
> **摘要:** We propose a novel dataset that has been specifically designed for 3D semantic segmentation of bridges and the domain gap analysis caused by varying sensors. This addresses a critical need in the field of infrastructure inspection and maintenance, which is essential for modern society. The dataset comprises high-resolution 3D scans of a diverse range of bridge structures from various countries, with detailed semantic labels provided for each. Our initial objective is to facilitate accurate and automated segmentation of bridge components, thereby advancing the structural health monitoring practice. To evaluate the effectiveness of existing 3D deep learning models on this novel dataset, we conduct a comprehensive analysis of three distinct state-of-the-art architectures. Furthermore, we present data acquired through diverse sensors to quantify the domain gap resulting from sensor variations. Our findings indicate that all architectures demonstrate robust performance on the specified task. However, the domain gap can potentially lead to a decline in the performance of up to 11.4% mIoU.
>
---
#### [replaced 069] Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2407.20836v5](https://arxiv.org/pdf/2407.20836v5)**

> **作者:** Yunfeng Diao; Naixin Zhai; Changtao Miao; Zitong Yu; Xingxing Wei; Xun Yang; Meng Wang
>
> **摘要:** Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. Our code is available at https://github.com/onotoa/fpba.
>
---
