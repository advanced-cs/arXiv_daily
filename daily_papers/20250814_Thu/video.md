# 计算机视觉 cs.CV

- **最新发布 149 篇**

- **更新 84 篇**

## 最新发布

#### [new 001] PERSONA: Personalized Whole-Body 3D Avatar with Pose-Driven Deformations from a Single Image
- **分类: cs.CV**

- **简介: 论文提出PERSONA框架，从单张图像生成个性化人体3D化身并实现姿态驱动形变，解决传统方法需大量姿态数据及身份保留难题，通过扩散生成姿态视频与几何加权优化提升真实感。**

- **链接: [http://arxiv.org/pdf/2508.09973v1](http://arxiv.org/pdf/2508.09973v1)**

> **作者:** Geonhee Sim; Gyeongsik Moon
>
> **备注:** Accepted to ICCV 2025. https://mks0601.github.io/PERSONA/
>
> **摘要:** Two major approaches exist for creating animatable human avatars. The first, a 3D-based approach, optimizes a NeRF- or 3DGS-based avatar from videos of a single person, achieving personalization through a disentangled identity representation. However, modeling pose-driven deformations, such as non-rigid cloth deformations, requires numerous pose-rich videos, which are costly and impractical to capture in daily life. The second, a diffusion-based approach, learns pose-driven deformations from large-scale in-the-wild videos but struggles with identity preservation and pose-dependent identity entanglement. We present PERSONA, a framework that combines the strengths of both approaches to obtain a personalized 3D human avatar with pose-driven deformations from a single image. PERSONA leverages a diffusion-based approach to generate pose-rich videos from the input image and optimizes a 3D avatar based on them. To ensure high authenticity and sharp renderings across diverse poses, we introduce balanced sampling and geometry-weighted optimization. Balanced sampling oversamples the input image to mitigate identity shifts in diffusion-generated training videos. Geometry-weighted optimization prioritizes geometry constraints over image loss, preserving rendering quality in diverse poses.
>
---
#### [new 002] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing
- **分类: cs.CV**

- **简介: 论文提出SVG-Head，通过混合表面-体积高斯模型实现高保真头像重建与实时编辑，解决几何与外观隐式表示及耦合问题，利用解耦纹理捕捉全局外观，并采用FLAME网格UV映射提升渲染效率。**

- **链接: [http://arxiv.org/pdf/2508.09597v1](http://arxiv.org/pdf/2508.09597v1)**

> **作者:** Heyi Sun; Cong Wang; Tian-Xing Xu; Jingwei Huang; Di Kang; Chunchao Guo; Song-Hai Zhang
>
> **摘要:** Creating high-fidelity and editable head avatars is a pivotal challenge in computer vision and graphics, boosting many AR/VR applications. While recent advancements have achieved photorealistic renderings and plausible animation, head editing, especially real-time appearance editing, remains challenging due to the implicit representation and entangled modeling of the geometry and global appearance. To address this, we propose Surface-Volumetric Gaussian Head Avatar (SVG-Head), a novel hybrid representation that explicitly models the geometry with 3D Gaussians bound on a FLAME mesh and leverages disentangled texture images to capture the global appearance. Technically, it contains two types of Gaussians, in which surface Gaussians explicitly model the appearance of head avatars using learnable texture images, facilitating real-time texture editing, while volumetric Gaussians enhance the reconstruction quality of non-Lambertian regions (e.g., lips and hair). To model the correspondence between 3D world and texture space, we provide a mesh-aware Gaussian UV mapping method, which leverages UV coordinates given by the FLAME mesh to obtain sharp texture images and real-time rendering speed. A hierarchical optimization strategy is further designed to pursue the optimal performance in both reconstruction quality and editing flexibility. Experiments on the NeRSemble dataset show that SVG-Head not only generates high-fidelity rendering results, but also is the first method to obtain explicit texture images for Gaussian head avatars and support real-time appearance editing.
>
---
#### [new 003] NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation
- **分类: cs.CV; cs.LG**

- **简介: 论文提出基于注意力引导的剪枝方法，解决多模态医疗影像在资源受限场景下的存储与传输问题，通过语义压缩生成统一图表示，提升诊断性能。**

- **链接: [http://arxiv.org/pdf/2508.09715v1](http://arxiv.org/pdf/2508.09715v1)**

> **作者:** Devvrat Joshi; Islem Rekik
>
> **摘要:** The rapid growth of multimodal medical imaging data presents significant storage and transmission challenges, particularly in resource-constrained clinical settings. We propose NEURAL, a novel framework that addresses this by using semantics-guided data compression. Our approach repurposes cross-attention scores between the image and its radiological report from a fine-tuned generative vision-language model to structurally prune chest X-rays, preserving only diagnostically critical regions. This process transforms the image into a highly compressed, graph representation. This unified graph-based representation fuses the pruned visual graph with a knowledge graph derived from the clinical report, creating a universal data structure that simplifies downstream modeling. Validated on the MIMIC-CXR and CheXpert Plus dataset for pneumonia detection, NEURAL achieves a 93.4-97.7\% reduction in image data size while maintaining a high diagnostic performance of 0.88-0.95 AUC, outperforming other baseline models that use uncompressed data. By creating a persistent, task-agnostic data asset, NEURAL resolves the trade-off between data size and clinical utility, enabling efficient workflows and teleradiology without sacrificing performance. Our NEURAL code is available at https://github.com/basiralab/NEURAL.
>
---
#### [new 004] Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 论文针对物体目标导航任务，提出GOAL框架，通过蒸馏LLM先验知识至生成流模型，解决传统方法受限于室内布局不确定性的泛化问题，实现更强的环境适应性。**

- **链接: [http://arxiv.org/pdf/2508.09423v1](http://arxiv.org/pdf/2508.09423v1)**

> **作者:** Badi Li; Ren-jie Lu; Yu Zhou; Jingke Meng; Wei-shi Zheng
>
> **摘要:** The Object Goal Navigation (ObjectNav) task challenges agents to locate a specified object in an unseen environment by imagining unobserved regions of the scene. Prior approaches rely on deterministic and discriminative models to complete semantic maps, overlooking the inherent uncertainty in indoor layouts and limiting their ability to generalize to unseen environments. In this work, we propose GOAL, a generative flow-based framework that models the semantic distribution of indoor environments by bridging observed regions with LLM-enriched full-scene semantic maps. During training, spatial priors inferred from large language models (LLMs) are encoded as two-dimensional Gaussian fields and injected into target maps, distilling rich contextual knowledge into the flow model and enabling more generalizable completions. Extensive experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D and Gibson, and shows strong generalization in transfer settings to HM3D. Codes and pretrained models are available at https://github.com/Badi-Li/GOAL.
>
---
#### [new 005] Leveraging Failed Samples: A Few-Shot and Training-Free Framework for Generalized Deepfake Detection
- **分类: cs.CV**

- **简介: 论文提出FTNet，无需训练，利用少量样本检测深度伪造，解决模型在未知样本上的泛化问题，实现8.7%性能提升。**

- **链接: [http://arxiv.org/pdf/2508.09475v1](http://arxiv.org/pdf/2508.09475v1)**

> **作者:** Shibo Yao; Renshuai Tao; Xiaolong Zheng; Chao Liang; Chunjie Zhang
>
> **摘要:** Recent deepfake detection studies often treat unseen sample detection as a ``zero-shot" task, training on images generated by known models but generalizing to unknown ones. A key real-world challenge arises when a model performs poorly on unknown samples, yet these samples remain available for analysis. This highlights that it should be approached as a ``few-shot" task, where effectively utilizing a small number of samples can lead to significant improvement. Unlike typical few-shot tasks focused on semantic understanding, deepfake detection prioritizes image realism, which closely mirrors real-world distributions. In this work, we propose the Few-shot Training-free Network (FTNet) for real-world few-shot deepfake detection. Simple yet effective, FTNet differs from traditional methods that rely on large-scale known data for training. Instead, FTNet uses only one fake samplefrom an evaluation set, mimicking the scenario where new samples emerge in the real world and can be gathered for use, without any training or parameter updates. During evaluation, each test sample is compared to the known fake and real samples, and it is classified based on the category of the nearest sample. We conduct a comprehensive analysis of AI-generated images from 29 different generative models and achieve a new SoTA performance, with an average improvement of 8.7\% compared to existing methods. This work introduces a fresh perspective on real-world deepfake detection: when the model struggles to generalize on a few-shot sample, leveraging the failed samples leads to better performance.
>
---
#### [new 006] Enhancing Monocular 3D Hand Reconstruction with Learned Texture Priors
- **分类: cs.CV**

- **简介: 论文提出基于学习纹理先验的轻量级模块，通过密集对齐损失提升单目3D手重建精度，利用可微渲染管道和改进HaMeR模型。**

- **链接: [http://arxiv.org/pdf/2508.09629v1](http://arxiv.org/pdf/2508.09629v1)**

> **作者:** Giorgos Karvounas; Nikolaos Kyriazis; Iason Oikonomidis; Georgios Pavlakos; Antonis A. Argyros
>
> **摘要:** We revisit the role of texture in monocular 3D hand reconstruction, not as an afterthought for photorealism, but as a dense, spatially grounded cue that can actively support pose and shape estimation. Our observation is simple: even in high-performing models, the overlay between predicted hand geometry and image appearance is often imperfect, suggesting that texture alignment may be an underused supervisory signal. We propose a lightweight texture module that embeds per-pixel observations into UV texture space and enables a novel dense alignment loss between predicted and observed hand appearances. Our approach assumes access to a differentiable rendering pipeline and a model that maps images to 3D hand meshes with known topology, allowing us to back-project a textured hand onto the image and perform pixel-based alignment. The module is self-contained and easily pluggable into existing reconstruction pipelines. To isolate and highlight the value of texture-guided supervision, we augment HaMeR, a high-performing yet unadorned transformer architecture for 3D hand pose estimation. The resulting system improves both accuracy and realism, demonstrating the value of appearance-guided alignment in hand reconstruction.
>
---
#### [new 007] Learning Spatial Decay for Vision Transformers
- **分类: cs.CV**

- **简介: 论文提出Spatial Decay Transformer（SDT），通过内容感知的动态空间衰减机制解决视觉Transformer在空间结构任务中的性能瓶颈，结合上下文感知门控（CAG）融合空间与内容信息，提升图像分类与生成任务性能。**

- **链接: [http://arxiv.org/pdf/2508.09525v1](http://arxiv.org/pdf/2508.09525v1)**

> **作者:** Yuxin Mao; Zhen Qin; Jinxing Zhou; Bin Fan; Jing Zhang; Yiran Zhong; Yuchao Dai
>
> **摘要:** Vision Transformers (ViTs) have revolutionized computer vision, yet their self-attention mechanism lacks explicit spatial inductive biases, leading to suboptimal performance on spatially-structured tasks. Existing approaches introduce data-independent spatial decay based on fixed distance metrics, applying uniform attention weighting regardless of image content and limiting adaptability to diverse visual scenarios. Inspired by recent advances in large language models where content-aware gating mechanisms (e.g., GLA, HGRN2, FOX) significantly outperform static alternatives, we present the first successful adaptation of data-dependent spatial decay to 2D vision transformers. We introduce \textbf{Spatial Decay Transformer (SDT)}, featuring a novel Context-Aware Gating (CAG) mechanism that generates dynamic, data-dependent decay for patch interactions. Our approach learns to modulate spatial attention based on both content relevance and spatial proximity. We address the fundamental challenge of 1D-to-2D adaptation through a unified spatial-content fusion framework that integrates manhattan distance-based spatial priors with learned content representations. Extensive experiments on ImageNet-1K classification and generation tasks demonstrate consistent improvements over strong baselines. Our work establishes data-dependent spatial decay as a new paradigm for enhancing spatial attention in vision transformers.
>
---
#### [new 008] Story2Board: A Training-Free Approach for Expressive Storyboard Generation
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 论文提出训练-free的Story2Board框架，解决视觉讲故事关键要素缺失问题，通过潜层锚定与互惠注意力混合增强一致性，生成动态连贯分镜。**

- **链接: [http://arxiv.org/pdf/2508.09983v1](http://arxiv.org/pdf/2508.09983v1)**

> **作者:** David Dinkevich; Matan Levy; Omri Avrahami; Dvir Samuel; Dani Lischinski
>
> **备注:** Project page is available at https://daviddinkevich.github.io/Story2Board/
>
> **摘要:** We present Story2Board, a training-free framework for expressive storyboard generation from natural language. Existing methods narrowly focus on subject identity, overlooking key aspects of visual storytelling such as spatial composition, background evolution, and narrative pacing. To address this, we introduce a lightweight consistency framework composed of two components: Latent Panel Anchoring, which preserves a shared character reference across panels, and Reciprocal Attention Value Mixing, which softly blends visual features between token pairs with strong reciprocal attention. Together, these mechanisms enhance coherence without architectural changes or fine-tuning, enabling state-of-the-art diffusion models to generate visually diverse yet consistent storyboards. To structure generation, we use an off-the-shelf language model to convert free-form stories into grounded panel-level prompts. To evaluate, we propose the Rich Storyboard Benchmark, a suite of open-domain narratives designed to assess layout diversity and background-grounded storytelling, in addition to consistency. We also introduce a new Scene Diversity metric that quantifies spatial and pose variation across storyboards. Our qualitative and quantitative results, as well as a user study, show that Story2Board produces more dynamic, coherent, and narratively engaging storyboards than existing baselines.
>
---
#### [new 009] What-Meets-Where: Unified Learning of Action and Contact Localization in a New Dataset
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种联合学习动作与接触定位的视觉任务，解决现有方法无法同时建模动作语义和空间关系的问题。提出PaIR-Net框架，包含接触先验模块、像素级分割模块和交互推理模块，并构建PaIR数据集，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.09428v1](http://arxiv.org/pdf/2508.09428v1)**

> **作者:** Yuxiao Wang; Yu Lei; Wolin Liang; Weiying Xue; Zhenao Wei; Nan Zhuang; Qi Liu
>
> **摘要:** People control their bodies to establish contact with the environment. To comprehensively understand actions across diverse visual contexts, it is essential to simultaneously consider \textbf{what} action is occurring and \textbf{where} it is happening. Current methodologies, however, often inadequately capture this duality, typically failing to jointly model both action semantics and their spatial contextualization within scenes. To bridge this gap, we introduce a novel vision task that simultaneously predicts high-level action semantics and fine-grained body-part contact regions. Our proposed framework, PaIR-Net, comprises three key components: the Contact Prior Aware Module (CPAM) for identifying contact-relevant body parts, the Prior-Guided Concat Segmenter (PGCS) for pixel-wise contact segmentation, and the Interaction Inference Module (IIM) responsible for integrating global interaction relationships. To facilitate this task, we present PaIR (Part-aware Interaction Representation), a comprehensive dataset containing 13,979 images that encompass 654 actions, 80 object categories, and 17 body parts. Experimental evaluation demonstrates that PaIR-Net significantly outperforms baseline approaches, while ablation studies confirm the efficacy of each architectural component. The code and dataset will be released upon publication.
>
---
#### [new 010] A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation
- **分类: cs.CV**

- **简介: 论文综述3D Gaussian Splatting在场景分割、编辑与生成等任务的应用，总结方法、监督策略及学习范式，提出资源库支持研究。**

- **链接: [http://arxiv.org/pdf/2508.09977v1](http://arxiv.org/pdf/2508.09977v1)**

> **作者:** Shuting He; Peilin Ji; Yitong Yang; Changshuo Wang; Jiayi Ji; Yinglin Wang; Henghui Ding
>
> **备注:** GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a powerful alternative to Neural Radiance Fields (NeRF) for 3D scene representation, offering high-fidelity photorealistic rendering with real-time performance. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first introduces 2D foundation models that support semantic understanding and control in 3DGS applications, followed by a review of NeRF-based methods that inform their 3DGS counterparts. We then categorize 3DGS applications into segmentation, editing, generation, and other functional tasks. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
>
---
#### [new 011] Skyshield: Event-Driven Submillimetre Thin Obstacle Detection for Drone Flight Safety
- **分类: cs.CV**

- **简介: 论文提出事件驱动的SkyShield框架，用于无人机亚毫米级薄障碍检测，通过轻量U-Net和Dice-Contour损失实现高精度低延迟。**

- **链接: [http://arxiv.org/pdf/2508.09397v1](http://arxiv.org/pdf/2508.09397v1)**

> **作者:** Zhengli Zhang; Xinyu Luo; Yuchen Sun; Wenhua Ding; Dongyu Huang; Xinlei Chen
>
> **摘要:** Drones operating in complex environments face a significant threat from thin obstacles, such as steel wires and kite strings at the submillimeter level, which are notoriously difficult for conventional sensors like RGB cameras, LiDAR, and depth cameras to detect. This paper introduces SkyShield, an event-driven, end-to-end framework designed for the perception of submillimeter scale obstacles. Drawing upon the unique features that thin obstacles present in the event stream, our method employs a lightweight U-Net architecture and an innovative Dice-Contour Regularization Loss to ensure precise detection. Experimental results demonstrate that our event-based approach achieves mean F1 Score of 0.7088 with a low latency of 21.2 ms, making it ideal for deployment on edge and mobile platforms.
>
---
#### [new 012] Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory
- **分类: cs.CV**

- **简介: 论文提出多模态代理M3-Agent，具备长期记忆，能处理视觉听觉输入并进行多轮推理，通过M3-Bench评估其性能，实验显示其在多项指标上优于基线模型，推动多模态代理向更人类似的表现发展。**

- **链接: [http://arxiv.org/pdf/2508.09736v1](http://arxiv.org/pdf/2508.09736v1)**

> **作者:** Lin Long; Yichen He; Wentao Ye; Yiyuan Pan; Yuan Lin; Hang Li; Junbo Zhao; Wei Li
>
> **摘要:** We introduce M3-Agent, a novel multimodal agent framework equipped with long-term memory. Like humans, M3-Agent can process real-time visual and auditory inputs to build and update its long-term memory. Beyond episodic memory, it also develops semantic memory, enabling it to accumulate world knowledge over time. Its memory is organized in an entity-centric, multimodal format, allowing deeper and more consistent understanding of the environment. Given an instruction, M3-Agent autonomously performs multi-turn, iterative reasoning and retrieves relevant information from memory to accomplish the task. To evaluate memory effectiveness and memory-based reasoning in multimodal agents, we develop M3-Bench, a new long-video question answering benchmark. M3-Bench comprises 100 newly recorded real-world videos captured from a robot's perspective (M3-Bench-robot) and 929 web-sourced videos across diverse scenarios (M3-Bench-web). We annotate question-answer pairs designed to test key capabilities essential for agent applications, such as human understanding, general knowledge extraction, and cross-modal reasoning. Experimental results show that M3-Agent, trained via reinforcement learning, outperforms the strongest baseline, a prompting agent using Gemini-1.5-pro and GPT-4o, achieving 6.7%, 7.7%, and 5.3% higher accuracy on M3-Bench-robot, M3-Bench-web and VideoMME-long, respectively. Our work advances the multimodal agents toward more human-like long-term memory and provides insights into their practical design. Model, code and data are available at https://github.com/bytedance-seed/m3-agent
>
---
#### [new 013] Personalized Feature Translation for Expression Recognition: An Efficient Source-Free Domain Adaptation Method
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于潜在空间的个性化特征翻译（PFT）方法，解决源域数据缺失时目标域仅含中性表情导致的适应效率低下问题，通过优化表达一致性与风格特征，实现轻量级域适应，提升表情识别性能。**

- **链接: [http://arxiv.org/pdf/2508.09202v1](http://arxiv.org/pdf/2508.09202v1)**

> **作者:** Masoumeh Sharafi; Soufiane Belharbi; Houssem Ben Salem; Ali Etemad; Alessandro Lameiras Koerich; Marco Pedersoli; Simon Bacon; Eric Granger
>
> **摘要:** Facial expression recognition (FER) models are employed in many video-based affective computing applications, such as human-computer interaction and healthcare monitoring. However, deep FER models often struggle with subtle expressions and high inter-subject variability, limiting their performance in real-world applications. To improve their performance, source-free domain adaptation (SFDA) methods have been proposed to personalize a pretrained source model using only unlabeled target domain data, thereby avoiding data privacy, storage, and transmission constraints. This paper addresses a challenging scenario where source data is unavailable for adaptation, and only unlabeled target data consisting solely of neutral expressions is available. SFDA methods are not typically designed to adapt using target data from only a single class. Further, using models to generate facial images with non-neutral expressions can be unstable and computationally intensive. In this paper, personalized feature translation (PFT) is proposed for SFDA. Unlike current image translation methods for SFDA, our lightweight method operates in the latent space. We first pre-train the translator on the source domain data to transform the subject-specific style features from one source subject into another. Expression information is preserved by optimizing a combination of expression consistency and style-aware objectives. Then, the translator is adapted on neutral target data, without using source data or image synthesis. By translating in the latent space, PFT avoids the complexity and noise of face expression generation, producing discriminative embeddings optimized for classification. Using PFT eliminates the need for image synthesis, reduces computational overhead (using a lightweight translator), and only adapts part of the model, making the method efficient compared to image-based translation.
>
---
#### [new 014] Surg-InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出基于可逆NeRF的Surg-InvNeRF方法，用于手术视觉中的长期3D点跟踪与重建，解决传统方法受限于2D运动和缺乏一致性的问题，通过逆向渲染和多尺度结构提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.09681v1](http://arxiv.org/pdf/2508.09681v1)**

> **作者:** Gerardo Loza; Junlei Hu; Dominic Jones; Sharib Ali; Pietro Valdastri
>
> **备注:** 10 pages
>
> **摘要:** We proposed a novel test-time optimisation (TTO) approach framed by a NeRF-based architecture for long-term 3D point tracking. Most current methods in point tracking struggle to obtain consistent motion or are limited to 2D motion. TTO approaches frame the solution for long-term tracking as optimising a function that aggregates correspondences from other specialised state-of-the-art methods. Unlike the state-of-the-art on TTO, we propose parametrising such a function with our new invertible Neural Radiance Field (InvNeRF) architecture to perform both 2D and 3D tracking in surgical scenarios. Our approach allows us to exploit the advantages of a rendering-based approach by supervising the reprojection of pixel correspondences. It adapts strategies from recent rendering-based methods to obtain a bidirectional deformable-canonical mapping, to efficiently handle a defined workspace, and to guide the rays' density. It also presents our multi-scale HexPlanes for fast inference and a new algorithm for efficient pixel sampling and convergence criteria. We present results in the STIR and SCARE datasets, for evaluating point tracking and testing the integration of kinematic data in our pipeline, respectively. In 2D point tracking, our approach surpasses the precision and accuracy of the TTO state-of-the-art methods by nearly 50% on average precision, while competing with other approaches. In 3D point tracking, this is the first TTO approach, surpassing feed-forward methods while incorporating the benefits of a deformable NeRF-based reconstruction.
>
---
#### [new 015] SHALE: A Scalable Benchmark for Fine-grained Hallucination Evaluation in LVLMs
- **分类: cs.CV**

- **简介: 论文提出SHALE基准，解决LVLM幻觉评估中粗粒度不足和数据泄露问题，通过自动化数据管道与层次化框架构建细粒度评估体系，覆盖12视觉维度和6知识领域，评估信度与事实性幻觉。**

- **链接: [http://arxiv.org/pdf/2508.09584v1](http://arxiv.org/pdf/2508.09584v1)**

> **作者:** Bei Yan; Zhiyuan Chen; Yuecong Min; Jie Zhang; Jiahao Wang; Xiaozhen Wang; Shiguang Shan
>
> **摘要:** Despite rapid advances, Large Vision-Language Models (LVLMs) still suffer from hallucinations, i.e., generating content inconsistent with input or established world knowledge, which correspond to faithfulness and factuality hallucinations, respectively. Prior studies primarily evaluate faithfulness hallucination at a coarse level (e.g., object-level) and lack fine-grained analysis. Additionally, existing benchmarks rely on costly manual curation or reused public datasets, raising concerns about scalability and data leakage. To address these limitations, we propose an automated data construction pipeline that produces scalable, controllable, and diverse evaluation data. We also design a hierarchical hallucination induction framework with input perturbations to simulate realistic noisy scenarios. Integrating these designs, we construct SHALE, a Scalable HALlucination Evaluation benchmark designed to assess both faithfulness and factuality hallucinations via a fine-grained hallucination categorization scheme. SHALE comprises over 30K image-instruction pairs spanning 12 representative visual perception aspects for faithfulness and 6 knowledge domains for factuality, considering both clean and noisy scenarios. Extensive experiments on over 20 mainstream LVLMs reveal significant factuality hallucinations and high sensitivity to semantic perturbations.
>
---
#### [new 016] Multi-Sequence Parotid Gland Lesion Segmentation via Expert Text-Guided Segment Anything Model
- **分类: cs.CV**

- **简介: 论文提出基于专家文本引导的多序列腮腺病变分割模型PG-SAM，解决传统方法依赖人工标注的难题，通过跨模态注意力与领域知识融合提升分割精度。**

- **链接: [http://arxiv.org/pdf/2508.09645v1](http://arxiv.org/pdf/2508.09645v1)**

> **作者:** Zhongyuan Wu; Chuan-Xian Ren; Yu Wang; Xiaohua Ban; Jianning Xiao; Xiaohui Duan
>
> **摘要:** Parotid gland lesion segmentation is essential for the treatment of parotid gland diseases. However, due to the variable size and complex lesion boundaries, accurate parotid gland lesion segmentation remains challenging. Recently, the Segment Anything Model (SAM) fine-tuning has shown remarkable performance in the field of medical image segmentation. Nevertheless, SAM's interaction segmentation model relies heavily on precise lesion prompts (points, boxes, masks, etc.), which are very difficult to obtain in real-world applications. Besides, current medical image segmentation methods are automatically generated, ignoring the domain knowledge of medical experts when performing segmentation. To address these limitations, we propose the parotid gland segment anything model (PG-SAM), an expert diagnosis text-guided SAM incorporating expert domain knowledge for cross-sequence parotid gland lesion segmentation. Specifically, we first propose an expert diagnosis report guided prompt generation module that can automatically generate prompt information containing the prior domain knowledge to guide the subsequent lesion segmentation process. Then, we introduce a cross-sequence attention module, which integrates the complementary information of different modalities to enhance the segmentation effect. Finally, the multi-sequence image features and generated prompts are feed into the decoder to get segmentation result. Experimental results demonstrate that PG-SAM achieves state-of-the-art performance in parotid gland lesion segmentation across three independent clinical centers, validating its clinical applicability and the effectiveness of diagnostic text for enhancing image segmentation in real-world clinical settings.
>
---
#### [new 017] Region-to-Region: Enhancing Generative Image Harmonization with Adaptive Regional Injection
- **分类: cs.CV; cs.AI**

- **简介: 本文提出Region-to-Region变换，通过适配区域注入提升图像调和能力，解决细节保留与合成数据局限问题，设计Clear-VAE与MACA增强前景调控，构建RPHarmony数据集，验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2508.09746v1](http://arxiv.org/pdf/2508.09746v1)**

> **作者:** Zhiqiu Zhang; Dongqi Fan; Mingjie Wang; Qiang Tang; Jian Yang; Zili Yi
>
> **摘要:** The goal of image harmonization is to adjust the foreground in a composite image to achieve visual consistency with the background. Recently, latent diffusion model (LDM) are applied for harmonization, achieving remarkable results. However, LDM-based harmonization faces challenges in detail preservation and limited harmonization ability. Additionally, current synthetic datasets rely on color transfer, which lacks local variations and fails to capture complex real-world lighting conditions. To enhance harmonization capabilities, we propose the Region-to-Region transformation. By injecting information from appropriate regions into the foreground, this approach preserves original details while achieving image harmonization or, conversely, generating new composite data. From this perspective, We propose a novel model R2R. Specifically, we design Clear-VAE to preserve high-frequency details in the foreground using Adaptive Filter while eliminating disharmonious elements. To further enhance harmonization, we introduce the Harmony Controller with Mask-aware Adaptive Channel Attention (MACA), which dynamically adjusts the foreground based on the channel importance of both foreground and background regions. To address the limitation of existing datasets, we propose Random Poisson Blending, which transfers color and lighting information from a suitable region to the foreground, thereby generating more diverse and challenging synthetic images. Using this method, we construct a new synthetic dataset, RPHarmony. Experiments demonstrate the superiority of our method over other methods in both quantitative metrics and visual harmony. Moreover, our dataset helps the model generate more realistic images in real examples. Our code, dataset, and model weights have all been released for open access.
>
---
#### [new 018] A Chain of Diagnosis Framework for Accurate and Explainable Radiology Report Generation
- **分类: cs.CV**

- **简介: 该论文提出一种链式诊断框架（CoD），解决放射学报告生成中临床效果差和可解释性不足的问题，通过QA生成、大模型生成及图像/病变锚定模块提升准确性与可解释性，并采用多标注训练策略实现高效学习。**

- **链接: [http://arxiv.org/pdf/2508.09566v1](http://arxiv.org/pdf/2508.09566v1)**

> **作者:** Haibo Jin; Haoxuan Che; Sunan He; Hao Chen
>
> **备注:** Accepted to IEEE TMI
>
> **摘要:** Despite the progress of radiology report generation (RRG), existing works face two challenges: 1) The performances in clinical efficacy are unsatisfactory, especially for lesion attributes description; 2) the generated text lacks explainability, making it difficult for radiologists to trust the results. To address the challenges, we focus on a trustworthy RRG model, which not only generates accurate descriptions of abnormalities, but also provides basis of its predictions. To this end, we propose a framework named chain of diagnosis (CoD), which maintains a chain of diagnostic process for clinically accurate and explainable RRG. It first generates question-answer (QA) pairs via diagnostic conversation to extract key findings, then prompts a large language model with QA diagnoses for accurate generation. To enhance explainability, a diagnosis grounding module is designed to match QA diagnoses and generated sentences, where the diagnoses act as a reference. Moreover, a lesion grounding module is designed to locate abnormalities in the image, further improving the working efficiency of radiologists. To facilitate label-efficient training, we propose an omni-supervised learning strategy with clinical consistency to leverage various types of annotations from different datasets. Our efforts lead to 1) an omni-labeled RRG dataset with QA pairs and lesion boxes; 2) a evaluation tool for assessing the accuracy of reports in describing lesion location and severity; 3) extensive experiments to demonstrate the effectiveness of CoD, where it outperforms both specialist and generalist models consistently on two RRG benchmarks and shows promising explainability by accurately grounding generated sentences to QA diagnoses and images.
>
---
#### [new 019] Offline Auto Labeling: BAAS
- **分类: cs.CV; cs.SY; eess.SY**

- **简介: 论文提出BAAS框架，用于雷达检测的扩展对象跟踪与融合标注，解决标注精度与效率问题，采用贝叶斯方法及模块化设计，提升轨迹和形状估计精度，在复杂场景中验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.09585v1](http://arxiv.org/pdf/2508.09585v1)**

> **作者:** Stefan Haag; Bharanidhar Duraisamy; Felix Govaers; Wolfgang Koch; Martin Fritzsche; Juergen Dickmann
>
> **摘要:** This paper introduces BAAS, a new Extended Object Tracking (EOT) and fusion-based label annotation framework for radar detections in autonomous driving. Our framework utilizes Bayesian-based tracking, smoothing and eventually fusion methods to provide veritable and precise object trajectories along with shape estimation to provide annotation labels on the detection level under various supervision levels. Simultaneously, the framework provides evaluation of tracking performance and label annotation. If manually labeled data is available, each processing module can be analyzed independently or combined with other modules to enable closed-loop continuous improvements. The framework performance is evaluated in a challenging urban real-world scenario in terms of tracking performance and the label annotation errors. We demonstrate the functionality of the proposed approach for varying dynamic objects and class types
>
---
#### [new 020] Animate-X++: Universal Character Image Animation with Dynamic Backgrounds
- **分类: cs.CV**

- **简介: 论文提出一种基于DiT的通用字符动画框架Animate-X++，解决人类动画无法推广至拟人化角色及静态背景限制问题，通过Pose Indicator增强运动表示并引入多任务训练实现动态背景生成，构建A2Bench基准评估。**

- **链接: [http://arxiv.org/pdf/2508.09454v1](http://arxiv.org/pdf/2508.09454v1)**

> **作者:** Shuai Tan; Biao Gong; Zhuoxin Liu; Yan Wang; Xi Chen; Yifan Feng; Hengshuang Zhao
>
> **备注:** Project page: https://lucaria-academy.github.io/Animate-X++/
>
> **摘要:** Character image animation, which generates high-quality videos from a reference image and target pose sequence, has seen significant progress in recent years. However, most existing methods only apply to human figures, which usually do not generalize well on anthropomorphic characters commonly used in industries like gaming and entertainment. Furthermore, previous methods could only generate videos with static backgrounds, which limits the realism of the videos. For the first challenge, our in-depth analysis suggests to attribute this limitation to their insufficient modeling of motion, which is unable to comprehend the movement pattern of the driving video, thus imposing a pose sequence rigidly onto the target character. To this end, this paper proposes Animate-X++, a universal animation framework based on DiT for various character types, including anthropomorphic characters. To enhance motion representation, we introduce the Pose Indicator, which captures comprehensive motion pattern from the driving video through both implicit and explicit manner. The former leverages CLIP visual features of a driving video to extract its gist of motion, like the overall movement pattern and temporal relations among motions, while the latter strengthens the generalization of DiT by simulating possible inputs in advance that may arise during inference. For the second challenge, we introduce a multi-task training strategy that jointly trains the animation and TI2V tasks. Combined with the proposed partial parameter training, this approach achieves not only character animation but also text-driven background dynamics, making the videos more realistic. Moreover, we introduce a new Animated Anthropomorphic Benchmark (A2Bench) to evaluate the performance of Animate-X++ on universal and widely applicable animation images. Extensive experiments demonstrate the superiority and effectiveness of Animate-X++.
>
---
#### [new 021] MME-Emotion: A Holistic Evaluation Benchmark for Emotional Intelligence in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 论文提出MME-Emotion作为多模态大语言模型情感智能的全面评估基准，解决现有基准不足问题，通过大规模视频数据和多任务设计，评估模型的泛化与推理能力。**

- **链接: [http://arxiv.org/pdf/2508.09210v1](http://arxiv.org/pdf/2508.09210v1)**

> **作者:** Fan Zhang; Zebang Cheng; Chong Deng; Haoxuan Li; Zheng Lian; Qian Chen; Huadai Liu; Wen Wang; Yi-Fan Zhang; Renrui Zhang; Ziyu Guo; Zhihong Zhu; Hao Wu; Haixin Wang; Yefeng Zheng; Xiaojiang Peng; Xian Wu; Kun Wang; Xiangang Li; Jieping Ye; Pheng-Ann Heng
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have catalyzed transformative progress in affective computing, enabling models to exhibit emergent emotional intelligence. Despite substantial methodological progress, current emotional benchmarks remain limited, as it is still unknown: (a) the generalization abilities of MLLMs across distinct scenarios, and (b) their reasoning capabilities to identify the triggering factors behind emotional states. To bridge these gaps, we present \textbf{MME-Emotion}, a systematic benchmark that assesses both emotional understanding and reasoning capabilities of MLLMs, enjoying \textit{scalable capacity}, \textit{diverse settings}, and \textit{unified protocols}. As the largest emotional intelligence benchmark for MLLMs, MME-Emotion contains over 6,000 curated video clips with task-specific questioning-answering (QA) pairs, spanning broad scenarios to formulate eight emotional tasks. It further incorporates a holistic evaluation suite with hybrid metrics for emotion recognition and reasoning, analyzed through a multi-agent system framework. Through a rigorous evaluation of 20 advanced MLLMs, we uncover both their strengths and limitations, yielding several key insights: \ding{182} Current MLLMs exhibit unsatisfactory emotional intelligence, with the best-performing model achieving only $39.3\%$ recognition score and $56.0\%$ Chain-of-Thought (CoT) score on our benchmark. \ding{183} Generalist models (\emph{e.g.}, Gemini-2.5-Pro) derive emotional intelligence from generalized multimodal understanding capabilities, while specialist models (\emph{e.g.}, R1-Omni) can achieve comparable performance through domain-specific post-training adaptation. By introducing MME-Emotion, we hope that it can serve as a foundation for advancing MLLMs' emotional intelligence in the future.
>
---
#### [new 022] Gradient-Direction-Aware Density Control for 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GDAGS框架，通过梯度相干比和动态权重机制解决3D高斯溅射中的过重建与过密集化问题，提升渲染质量并减少内存消耗。**

- **链接: [http://arxiv.org/pdf/2508.09239v1](http://arxiv.org/pdf/2508.09239v1)**

> **作者:** Zheng Zhou; Yu-Jie Xiong; Chun-Ming Xia; Jia-Chen Zhang; Hong-Jian Zhan
>
> **摘要:** The emergence of 3D Gaussian Splatting (3DGS) has significantly advanced novel view synthesis through explicit scene representation, enabling real-time photorealistic rendering. However, existing approaches manifest two critical limitations in complex scenarios: (1) Over-reconstruction occurs when persistent large Gaussians cannot meet adaptive splitting thresholds during density control. This is exacerbated by conflicting gradient directions that prevent effective splitting of these Gaussians; (2) Over-densification of Gaussians occurs in regions with aligned gradient aggregation, leading to redundant component proliferation. This redundancy significantly increases memory overhead due to unnecessary data retention. We present Gradient-Direction-Aware Gaussian Splatting (GDAGS), a gradient-direction-aware adaptive density control framework to address these challenges. Our key innovations: the gradient coherence ratio (GCR), computed through normalized gradient vector norms, which explicitly discriminates Gaussians with concordant versus conflicting gradient directions; and a nonlinear dynamic weighting mechanism leverages the GCR to enable gradient-direction-aware density control. Specifically, GDAGS prioritizes conflicting-gradient Gaussians during splitting operations to enhance geometric details while suppressing redundant concordant-direction Gaussians. Conversely, in cloning processes, GDAGS promotes concordant-direction Gaussian densification for structural completion while preventing conflicting-direction Gaussian overpopulation. Comprehensive evaluations across diverse real-world benchmarks demonstrate that GDAGS achieves superior rendering quality while effectively mitigating over-reconstruction, suppressing over-densification, and constructing compact scene representations with 50\% reduced memory consumption through optimized Gaussians utilization.
>
---
#### [new 023] SARE: Semantic-Aware Reconstruction Error for Generalizable Diffusion-Generated Image Detection
- **分类: cs.CV**

- **简介: 论文提出SARE（语义感知重建误差）方法，针对扩散生成图像检测中对未知分布模型泛化能力不足的问题，通过衡量图像与caption引导重建的语义差异，构建判别特征提升检测鲁棒性，实验表明其在GenImage等数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.09487v1](http://arxiv.org/pdf/2508.09487v1)**

> **作者:** Ju Yeon Kang; Jaehong Park; Semin Kim; Ji Won Yoon; Nam Soo Kim
>
> **备注:** Work in progress
>
> **摘要:** Recently, diffusion-generated image detection has gained increasing attention, as the rapid advancement of diffusion models has raised serious concerns about their potential misuse. While existing detection methods have achieved promising results, their performance often degrades significantly when facing fake images from unseen, out-of-distribution (OOD) generative models, since they primarily rely on model-specific artifacts. To address this limitation, we explore a fundamental property commonly observed in fake images. Motivated by the observation that fake images tend to exhibit higher similarity to their captions than real images, we propose a novel representation, namely Semantic-Aware Reconstruction Error (SARE), that measures the semantic difference between an image and its caption-guided reconstruction. The hypothesis behind SARE is that real images, whose captions often fail to fully capture their complex visual content, may undergo noticeable semantic shifts during the caption-guided reconstruction process. In contrast, fake images, which closely align with their captions, show minimal semantic changes. By quantifying these semantic shifts, SARE can be utilized as a discriminative feature for robust detection across diverse generative models. We empirically demonstrate that the proposed method exhibits strong generalization, outperforming existing baselines on benchmarks including GenImage and CommunityForensics.
>
---
#### [new 024] Beyond Blanket Masking: Examining Granularity for Privacy Protection in Images Captured by Blind and Low Vision Users
- **分类: cs.CV**

- **简介: 论文提出FiGPriv框架，通过细粒度分割与风险评分机制，针对盲人/低视力用户图像的隐私保护问题，实现高风险信息精准遮蔽，提升VLM性能11%及识别率45%，兼顾隐私与可用性。**

- **链接: [http://arxiv.org/pdf/2508.09245v1](http://arxiv.org/pdf/2508.09245v1)**

> **作者:** Jeffri Murrugarra-LLerena; Haoran Niu; K. Suzanne Barber; Hal Daumé III; Yang Trista Cao; Paola Cascante-Bonilla
>
> **摘要:** As visual assistant systems powered by visual language models (VLMs) become more prevalent, concerns over user privacy have grown, particularly for blind and low vision users who may unknowingly capture personal private information in their images. Existing privacy protection methods rely on coarse-grained segmentation, which uniformly masks entire private objects, often at the cost of usability. In this work, we propose FiGPriv, a fine-grained privacy protection framework that selectively masks only high-risk private information while preserving low-risk information. Our approach integrates fine-grained segmentation with a data-driven risk scoring mechanism. We evaluate our framework using the BIV-Priv-Seg dataset and show that FiG-Priv preserves +26% of image content, enhancing the ability of VLMs to provide useful responses by 11% and identify the image content by 45%, while ensuring privacy protection. Project Page: https://artcs1.github.io/VLMPrivacy/
>
---
#### [new 025] MoIIE: Mixture of Intra- and Inter-Modality Experts for Large Vision Language Models
- **分类: cs.CV**

- **简介: 论文提出MoIIE架构，针对大视觉语言模型的计算成本与跨模态关联难题，通过结合内模态与跨模态专家路由，实现高效参数激活与多模态学习，两阶段训练策略提升效果，参数量匹敌复杂模型。**

- **链接: [http://arxiv.org/pdf/2508.09779v1](http://arxiv.org/pdf/2508.09779v1)**

> **作者:** Dianyi Wang; Siyuan Wang; Zejun Li; Yikun Wang; Yitong Li; Duyu Tang; Xiaoyu Shen; Xuanjing Huang; Zhongyu Wei
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across multi-modal tasks by scaling model size and training data. However, these dense LVLMs incur significant computational costs and motivate the exploration of sparse Mixture of Experts (MoE) architectures. While MoE improve parameter efficiency, effectively applying MoE to simultaneously model modality-specific features and cross-modal associations in LVLMs remains challenging. In this work, we propose to incorporate Mixture of Intra- and Inter-Modality Experts (MoIIE) to LVLMs. For each token, expert routing is guided by its modality, directing tokens to their respective intra-modality experts as well as a shared pool of inter-modality experts, enabling the model to jointly learn rich intra-modal features and cross-modal interactions. We further introduce an effective and straightforward two-stage training strategy, which facilitates the direct activation of both MoE and multi-modal capabilities. Extensive experiments across different data scales and LLM backbone demonstrate the effectiveness, efficiency and generality of our approach. Notably, our MoIIE models with 5.5B and 11.3B activated parameters match or even surpass the performance of existing advanced open-source MoE-LLMs based multi-modal models that involve more activated parameters. The code is available at https://github.com/AlenjandroWang/MoIIE.
>
---
#### [new 026] RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization
- **分类: cs.CV; cs.AI**

- **简介: 论文提出RelayFormer框架，针对视觉操控定位（VML）任务，解决跨模态泛化不足及高分辨率/长时序输入处理效率低问题，通过全局-局部relay机制与轻量解码器实现可扩展的跨模态处理。**

- **链接: [http://arxiv.org/pdf/2508.09459v1](http://arxiv.org/pdf/2508.09459v1)**

> **作者:** Wen Huang; Jiarui Yang; Tao Dai; Jiawei Li; Shaoxiong Zhan; Bin Wang; Shu-Tao Xia
>
> **摘要:** Visual manipulation localization (VML) -- across both images and videos -- is a crucial task in digital forensics that involves identifying tampered regions in visual content. However, existing methods often lack cross-modal generalization and struggle to handle high-resolution or long-duration inputs efficiently. We propose RelayFormer, a unified and modular architecture for visual manipulation localization across images and videos. By leveraging flexible local units and a Global-Local Relay Attention (GLoRA) mechanism, it enables scalable, resolution-agnostic processing with strong generalization. Our framework integrates seamlessly with existing Transformer-based backbones, such as ViT and SegFormer, via lightweight adaptation modules that require only minimal architectural changes, ensuring compatibility without disrupting pretrained representations. Furthermore, we design a lightweight, query-based mask decoder that supports one-shot inference across video sequences with linear complexity. Extensive experiments across multiple benchmarks demonstrate that our approach achieves state-of-the-art localization performance, setting a new baseline for scalable and modality-agnostic VML. Code is available at: https://github.com/WenOOI/RelayFormer.
>
---
#### [new 027] Do Vision Transformers See Like Humans? Evaluating their Perceptual Alignment
- **分类: cs.CV**

- **简介: 论文评估视觉Transformer（ViT）的感知对齐，分析模型规模、数据增强及正则化对其影响，发现增大模型和数据增强会降低对齐，强调复杂度与人类感知的权衡。**

- **链接: [http://arxiv.org/pdf/2508.09850v1](http://arxiv.org/pdf/2508.09850v1)**

> **作者:** Pablo Hernández-Cámara; Jose Manuel Jaén-Lorites; Jorge Vila-Tomás; Valero Laparra; Jesus Malo
>
> **摘要:** Vision Transformers (ViTs) achieve remarkable performance in image recognition tasks, yet their alignment with human perception remains largely unexplored. This study systematically analyzes how model size, dataset size, data augmentation and regularization impact ViT perceptual alignment with human judgments on the TID2013 dataset. Our findings confirm that larger models exhibit lower perceptual alignment, consistent with previous works. Increasing dataset diversity has a minimal impact, but exposing models to the same images more times reduces alignment. Stronger data augmentation and regularization further decrease alignment, especially in models exposed to repeated training cycles. These results highlight a trade-off between model complexity, training strategies, and alignment with human perception, raising important considerations for applications requiring human-like visual understanding.
>
---
#### [new 028] Quo Vadis Handwritten Text Generation for Handwritten Text Recognition?
- **分类: cs.CV; cs.DL**

- **简介: 本文探讨手写文本生成在HTR中的应用，解决低资源环境下HTR性能不足问题，比较三种HTG模型对微调的影响，并分析合成数据的视觉与语言特征对效果的影响，提出选择模型的指南。**

- **链接: [http://arxiv.org/pdf/2508.09936v1](http://arxiv.org/pdf/2508.09936v1)**

> **作者:** Vittorio Pippi; Konstantina Nikolaidou; Silvia Cascianelli; George Retsinas; Giorgos Sfikas; Rita Cucchiara; Marcus Liwicki
>
> **备注:** Accepted at ICCV Workshop VisionDocs
>
> **摘要:** The digitization of historical manuscripts presents significant challenges for Handwritten Text Recognition (HTR) systems, particularly when dealing with small, author-specific collections that diverge from the training data distributions. Handwritten Text Generation (HTG) techniques, which generate synthetic data tailored to specific handwriting styles, offer a promising solution to address these challenges. However, the effectiveness of various HTG models in enhancing HTR performance, especially in low-resource transcription settings, has not been thoroughly evaluated. In this work, we systematically compare three state-of-the-art styled HTG models (representing the generative adversarial, diffusion, and autoregressive paradigms for HTG) to assess their impact on HTR fine-tuning. We analyze how visual and linguistic characteristics of synthetic data influence fine-tuning outcomes and provide quantitative guidelines for selecting the most effective HTG model. The results of our analysis provide insights into the current capabilities of HTG methods and highlight key areas for further improvement in their application to low-resource HTR.
>
---
#### [new 029] Blink-to-code: real-time Morse code communication via eye blink detection and classification
- **分类: cs.CV; 68T45, 92C55; H.5.2; I.2.10; J.3**

- **简介: 论文提出实时眨眼至摩斯电码系统，利用摄像头和计算机视觉检测分类眨眼，实现低成本辅助沟通。**

- **链接: [http://arxiv.org/pdf/2508.09344v1](http://arxiv.org/pdf/2508.09344v1)**

> **作者:** Anushka Bhatt
>
> **备注:** 4 pages, 4 figures. Preprint on blink-based Morse code communication via webcam for assistive technology. Relevant to computer vision and human-computer interaction
>
> **摘要:** This study proposes a real-time system that translates voluntary eye blinks into Morse code, enabling communication for individuals with severe motor impairments. Using a standard webcam and computer vision, the system detects and classifies blinks as short (dot) or long (dash), then decodes them into alphanumeric characters. Experiments with five participants show 62% decoding accuracy and 18-20 seconds response times, demonstrating a viable, low-cost assistive communication method.
>
---
#### [new 030] A Neurosymbolic Framework for Interpretable Cognitive Attack Detection in Augmented Reality
- **分类: cs.CV; cs.AI**

- **简介: 论文提出神经符号框架CADAR用于AR认知攻击检测，解决现有方法依赖视觉或黑盒模型的局限，通过融合多模态输入与粒子滤波实现可解释的语义推理，提升检测准确性10.7%。**

- **链接: [http://arxiv.org/pdf/2508.09185v1](http://arxiv.org/pdf/2508.09185v1)**

> **作者:** Rongqian Chen; Allison Andreyev; Yanming Xiu; Mahdi Imani; Bin Li; Maria Gorlatova; Gang Tan; Tian Lan
>
> **摘要:** Augmented Reality (AR) enriches perception by overlaying virtual elements on the physical world. Due to its growing popularity, cognitive attacks that alter AR content to manipulate users' semantic perception have received increasing attention. Existing detection methods often focus on visual changes, which are restricted to pixel- or image-level processing and lack semantic reasoning capabilities, or they rely on pre-trained vision-language models (VLMs), which function as black-box approaches with limited interpretability. In this paper, we present CADAR, a novel neurosymbolic approach for cognitive attack detection in AR. It fuses multimodal vision-language inputs using neural VLMs to obtain a symbolic perception-graph representation, incorporating prior knowledge, salience weighting, and temporal correlations. The model then enables particle-filter based statistical reasoning -- a sequential Monte Carlo method -- to detect cognitive attacks. Thus, CADAR inherits the adaptability of pre-trained VLM and the interpretability and reasoning rigor of particle filtering. Experiments on an extended AR cognitive attack dataset show accuracy improvements of up to 10.7% over strong baselines on challenging AR attack scenarios, underscoring the promise of neurosymbolic methods for effective and interpretable cognitive attack detection.
>
---
#### [new 031] Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于四维评估框架（输入相关性、OOD强度、危害性、拒绝率）的MLLM jailbreaking方法，通过递归重构策略BSD提升攻击成功率与危害性，揭示现有安全机制的缺陷。**

- **链接: [http://arxiv.org/pdf/2508.09218v1](http://arxiv.org/pdf/2508.09218v1)**

> **作者:** Zuoou Li; Weitong Zhang; Jingyuan Wang; Shuyuan Zhang; Wenjia Bai; Bernhard Kainz; Mengyun Qiao
>
> **摘要:** Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems.
>
---
#### [new 032] Towards Scalable Training for Handwritten Mathematical Expression Recognition
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于可扩展数据引擎生成的Tex80M数据集及TexTeller模型，解决手写数学表达式识别中数据稀缺与标注成本高的问题，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.09220v1](http://arxiv.org/pdf/2508.09220v1)**

> **作者:** Haoyang Li; Jiaqing Li; Jialun Cao; Zongyuan Yang; Yongping Xiong
>
> **摘要:** Large foundation models have achieved significant performance gains through scalable training on massive datasets. However, the field of \textbf{H}andwritten \textbf{M}athematical \textbf{E}xpression \textbf{R}ecognition (HMER) has been impeded by the scarcity of data, primarily due to the arduous and costly process of manual annotation. To bridge this gap, we propose a novel method integrating limited handwritten formulas with large-scale LaTeX-rendered formulas by developing a scalable data engine to generate complex and consistent LaTeX sequences. With this engine, we built the largest formula dataset to date, termed \texttt{Tex80M}, comprising over 80 million high-quality training instances. Then we propose \texttt{TexTeller}, the first HMER model trained at scale, by mix-training \texttt{Tex80M} with a relatively small HME dataset. The expansive training dataset and our refined pipeline have equipped \texttt{TexTeller} with state-of-the-art (SOTA) performance across nearly all benchmarks. To advance the field, we will openly release our complete model, entire dataset, and full codebase, enabling further research building upon our contributions.
>
---
#### [new 033] MPT: Motion Prompt Tuning for Micro-Expression Recognition
- **分类: cs.CV; I.2.8**

- **简介: 论文提出Motion Prompt Tuning（MPT）方法，针对微表情识别中数据稀缺和模型无法捕捉细微运动的问题，通过运动提示生成与组适配器提升模型性能，实现更精准的微表情识别。**

- **链接: [http://arxiv.org/pdf/2508.09446v1](http://arxiv.org/pdf/2508.09446v1)**

> **作者:** Jiateng Liu; Hengcan Shi; Feng Chen; Zhiwen Shao; Yaonan Wang; Jianfei Cai; Wenming Zheng
>
> **摘要:** Micro-expression recognition (MER) is crucial in the affective computing field due to its wide application in medical diagnosis, lie detection, and criminal investigation. Despite its significance, obtaining micro-expression (ME) annotations is challenging due to the expertise required from psychological professionals. Consequently, ME datasets often suffer from a scarcity of training samples, severely constraining the learning of MER models. While current large pre-training models (LMs) offer general and discriminative representations, their direct application to MER is hindered by an inability to capture transitory and subtle facial movements-essential elements for effective MER. This paper introduces Motion Prompt Tuning (MPT) as a novel approach to adapting LMs for MER, representing a pioneering method for subtle motion prompt tuning. Particularly, we introduce motion prompt generation, including motion magnification and Gaussian tokenization, to extract subtle motions as prompts for LMs. Additionally, a group adapter is carefully designed and inserted into the LM to enhance it in the target MER domain, facilitating a more nuanced distinction of ME representation. Furthermore, extensive experiments conducted on three widely used MER datasets demonstrate that our proposed MPT consistently surpasses state-of-the-art approaches and verifies its effectiveness.
>
---
#### [new 034] HyperKD: Distilling Cross-Spectral Knowledge in Masked Autoencoders via Inverse Domain Shift with Spatial-Aware Masking and Specialized Loss
- **分类: cs.CV; cs.LG**

- **简介: 论文提出HyperKD框架，通过逆向领域适应策略与特征对齐，解决高光谱遥感数据光谱差异及数据稀缺问题，提升模型表示学习与下游任务性能。**

- **链接: [http://arxiv.org/pdf/2508.09453v1](http://arxiv.org/pdf/2508.09453v1)**

> **作者:** Abdul Matin; Tanjim Bin Faruk; Shrideep Pallickara; Sangmi Lee Pallickara
>
> **摘要:** The proliferation of foundation models, pretrained on large-scale unlabeled datasets, has emerged as an effective approach in creating adaptable and reusable architectures that can be leveraged for various downstream tasks using satellite observations. However, their direct application to hyperspectral remote sensing remains challenging due to inherent spectral disparities and the scarcity of available observations. In this work, we present HyperKD, a novel knowledge distillation framework that enables transferring learned representations from a teacher model into a student model for effective development of a foundation model on hyperspectral images. Unlike typical knowledge distillation frameworks, which use a complex teacher to guide a simpler student, HyperKD enables an inverse form of knowledge transfer across different types of spectral data, guided by a simpler teacher model. Building upon a Masked Autoencoder, HyperKD distills knowledge from the Prithvi foundational model into a student tailored for EnMAP hyperspectral imagery. HyperKD addresses the inverse domain adaptation problem with spectral gaps by introducing a feature-based strategy that includes spectral range-based channel alignment, spatial feature-guided masking, and an enhanced loss function tailored for hyperspectral images. HyperKD bridges the substantial spectral domain gap, enabling the effective use of pretrained foundation models for geospatial applications. Extensive experiments show that HyperKD significantly improves representation learning in MAEs, leading to enhanced reconstruction fidelity and more robust performance on downstream tasks such as land cover classification, crop type identification, and soil organic carbon prediction, underpinning the potential of knowledge distillation frameworks in remote sensing analytics with hyperspectral imagery.
>
---
#### [new 035] Exploring the Equivalence of Closed-Set Generative and Real Data Augmentation in Image Classification
- **分类: cs.CV**

- **简介: 论文研究封闭集生成与真实数据增强在图像分类中的等效性，解决如何利用生成模型提升分类性能的问题，通过实验确定合成图像规模并量化等效关系，揭示效果受训练集大小和合成数据量的影响。**

- **链接: [http://arxiv.org/pdf/2508.09550v1](http://arxiv.org/pdf/2508.09550v1)**

> **作者:** Haowen Wang; Guowei Zhang; Xiang Zhang; Zeyuan Chen; Haiyang Xu; Dou Hoon Kwark; Zhuowen Tu
>
> **摘要:** In this paper, we address a key scientific problem in machine learning: Given a training set for an image classification task, can we train a generative model on this dataset to enhance the classification performance? (i.e., closed-set generative data augmentation). We start by exploring the distinctions and similarities between real images and closed-set synthetic images generated by advanced generative models. Through extensive experiments, we offer systematic insights into the effective use of closed-set synthetic data for augmentation. Notably, we empirically determine the equivalent scale of synthetic images needed for augmentation. In addition, we also show quantitative equivalence between the real data augmentation and open-set generative augmentation (generative models trained using data beyond the given training set). While it aligns with the common intuition that real images are generally preferred, our empirical formulation also offers a guideline to quantify the increased scale of synthetic data augmentation required to achieve comparable image classification performance. Our results on natural and medical image datasets further illustrate how this effect varies with the baseline training set size and the amount of synthetic data incorporated.
>
---
#### [new 036] Synthetic Data Generation for Emotional Depth Faces: Optimizing Conditional DCGANs via Genetic Algorithms in the Latent Space and Stabilizing Training with Knowledge Distillation
- **分类: cs.CV**

- **简介: 论文提出通过遗传算法优化DCGANs生成情感深度面部数据，结合知识蒸馏稳定训练，提升生成质量与多样性，实现94%以上分类准确率。**

- **链接: [http://arxiv.org/pdf/2508.09188v1](http://arxiv.org/pdf/2508.09188v1)**

> **作者:** Seyed Muhammad Hossein Mousavi; S. Younes Mirinezhad
>
> **摘要:** Affective computing faces a major challenge: the lack of high-quality, diverse depth facial datasets for recognizing subtle emotional expressions. We propose a framework for synthetic depth face generation using an optimized GAN with Knowledge Distillation (EMA teacher models) to stabilize training, improve quality, and prevent mode collapse. We also apply Genetic Algorithms to evolve GAN latent vectors based on image statistics, boosting diversity and visual quality for target emotions. The approach outperforms GAN, VAE, GMM, and KDE in both diversity and quality. For classification, we extract and concatenate LBP, HOG, Sobel edge, and intensity histogram features, achieving 94% and 96% accuracy with XGBoost. Evaluation using FID, IS, SSIM, and PSNR shows consistent improvement over state-of-the-art methods.
>
---
#### [new 037] Event-driven Robust Fitting on Neuromorphic Hardware
- **分类: cs.CV; cs.NE**

- **简介: 该论文提出在神经形态硬件上实现鲁棒拟合任务，解决传统方法高能耗问题，通过事件驱动架构与硬件优化，使能耗降低15%。**

- **链接: [http://arxiv.org/pdf/2508.09466v1](http://arxiv.org/pdf/2508.09466v1)**

> **作者:** Tam Ngoc-Bang Nguyen; Anh-Dzung Doan; Zhipeng Cai; Tat-Jun Chin
>
> **备注:** 11 pages, accepted in ICCV 2025 Workshop on Neuromorphic Vision (NeVI)
>
> **摘要:** Robust fitting of geometric models is a fundamental task in many computer vision pipelines. Numerous innovations have been produced on the topic, from improving the efficiency and accuracy of random sampling heuristics to generating novel theoretical insights that underpin new approaches with mathematical guarantees. However, one aspect of robust fitting that has received little attention is energy efficiency. This performance metric has become critical as high energy consumption is a growing concern for AI adoption. In this paper, we explore energy-efficient robust fitting via the neuromorphic computing paradigm. Specifically, we designed a novel spiking neural network for robust fitting on real neuromorphic hardware, the Intel Loihi 2. Enabling this are novel event-driven formulations of model estimation that allow robust fitting to be implemented in the unique architecture of Loihi 2, and algorithmic strategies to alleviate the current limited precision and instruction set of the hardware. Results show that our neuromorphic robust fitting consumes only a fraction (15%) of the energy required to run the established robust fitting algorithm on a standard CPU to equivalent accuracy.
>
---
#### [new 038] Combinative Matching for Geometric Shape Assembly
- **分类: cs.CV; cs.AI**

- **简介: 论文提出组合匹配方法，解决几何形状组装中嵌套部件局部模糊问题，通过建模表面形状一致性和体积互斥性，结合等变神经网络实现旋转对齐，提升装配精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.09780v1](http://arxiv.org/pdf/2508.09780v1)**

> **作者:** Nahyuk Lee; Juhong Min; Junhong Lee; Chunghyun Park; Minsu Cho
>
> **备注:** Accepted to ICCV 2025 (Highlight)
>
> **摘要:** This paper introduces a new shape-matching methodology, combinative matching, to combine interlocking parts for geometric shape assembly. Previous methods for geometric assembly typically rely on aligning parts by finding identical surfaces between the parts as in conventional shape matching and registration. In contrast, we explicitly model two distinct properties of interlocking shapes: 'identical surface shape' and 'opposite volume occupancy.' Our method thus learns to establish correspondences across regions where their surface shapes appear identical but their volumes occupy the inverted space to each other. To facilitate this process, we also learn to align regions in rotation by estimating their shape orientations via equivariant neural networks. The proposed approach significantly reduces local ambiguities in matching and allows a robust combination of parts in assembly. Experimental results on geometric assembly benchmarks demonstrate the efficacy of our method, consistently outperforming the state of the art. Project page: https://nahyuklee.github.io/cmnet.
>
---
#### [new 039] MInDI-3D: Iterative Deep Learning in 3D for Sparse-view Cone Beam Computed Tomography
- **分类: cs.CV; cs.AI**

- **简介: 论文提出3D条件扩散模型MInDI-3D，针对稀疏视锥束CT伪影去除，通过迭代去噪与大规模伪CT数据集训练，实现低辐射下高保真重建，临床评估显示其性能媲美3D U-Net，适用于多种扫描设备。**

- **链接: [http://arxiv.org/pdf/2508.09616v1](http://arxiv.org/pdf/2508.09616v1)**

> **作者:** Daniel Barco; Marc Stadelmann; Martin Oswald; Ivo Herzig; Lukas Lichtensteiger; Pascal Paysan; Igor Peterlik; Michal Walczak; Bjoern Menze; Frank-Peter Schilling
>
> **摘要:** We present MInDI-3D (Medical Inversion by Direct Iteration in 3D), the first 3D conditional diffusion-based model for real-world sparse-view Cone Beam Computed Tomography (CBCT) artefact removal, aiming to reduce imaging radiation exposure. A key contribution is extending the "InDI" concept from 2D to a full 3D volumetric approach for medical images, implementing an iterative denoising process that refines the CBCT volume directly from sparse-view input. A further contribution is the generation of a large pseudo-CBCT dataset (16,182) from chest CT volumes of the CT-RATE public dataset to robustly train MInDI-3D. We performed a comprehensive evaluation, including quantitative metrics, scalability analysis, generalisation tests, and a clinical assessment by 11 clinicians. Our results show MInDI-3D's effectiveness, achieving a 12.96 (6.10) dB PSNR gain over uncorrected scans with only 50 projections on the CT-RATE pseudo-CBCT (independent real-world) test set and enabling an 8x reduction in imaging radiation exposure. We demonstrate its scalability by showing that performance improves with more training data. Importantly, MInDI-3D matches the performance of a 3D U-Net on real-world scans from 16 cancer patients across distortion and task-based metrics. It also generalises to new CBCT scanner geometries. Clinicians rated our model as sufficient for patient positioning across all anatomical sites and found it preserved lung tumour boundaries well.
>
---
#### [new 040] The Brain Resection Multimodal Image Registration (ReMIND2Reg) 2025 Challenge
- **分类: cs.CV**

- **简介: 论文提出脑切除多模态图像注册挑战，解决术中MRI因脑移位精度下降问题，通过ReMIND2Reg 2025数据集构建标准化评估框架，推动鲁棒通用的多模态注册算法发展。**

- **链接: [http://arxiv.org/pdf/2508.09649v1](http://arxiv.org/pdf/2508.09649v1)**

> **作者:** Reuben Dorent; Laura Rigolo; Colin P. Galvin; Junyu Chen; Mattias P. Heinrich; Aaron Carass; Olivier Colliot; Demian Wassermann; Alexandra Golby; Tina Kapur; William Wells
>
> **摘要:** Accurate intraoperative image guidance is critical for achieving maximal safe resection in brain tumor surgery, yet neuronavigation systems based on preoperative MRI lose accuracy during the procedure due to brain shift. Aligning post-resection intraoperative ultrasound (iUS) with preoperative MRI can restore spatial accuracy by estimating brain shift deformations, but it remains a challenging problem given the large anatomical and topological changes and substantial modality intensity gap. The ReMIND2Reg 2025 Challenge provides the largest public benchmark for this task, built upon the ReMIND dataset. It offers 99 training cases, 5 validation cases, and 10 private test cases comprising paired 3D ceT1 MRI, T2 MRI, and post-resection 3D iUS volumes. Data are provided without annotations for training, while validation and test performance are evaluated on manually annotated anatomical landmarks. Metrics include target registration error (TRE), robustness to worst-case landmark misalignment (TRE30), and runtime. By establishing a standardized evaluation framework for this clinically critical and technically complex problem, ReMIND2Reg aims to accelerate the development of robust, generalizable, and clinically deployable multimodal registration algorithms for image-guided neurosurgery.
>
---
#### [new 041] Noise-adapted Neural Operator for Robust Non-Line-of-Sight Imaging
- **分类: cs.CV**

- **简介: 论文提出噪声适应神经操作符框架，解决非视距成像中噪声干扰问题，通过参数化逆问题与时空特征融合提升重建精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.09655v1](http://arxiv.org/pdf/2508.09655v1)**

> **作者:** Lianfang Wang; Kuilin Qin; Xueying Liu; Huibin Chang; Yong Wang; Yuping Duan
>
> **摘要:** Computational imaging, especially non-line-of-sight (NLOS) imaging, the extraction of information from obscured or hidden scenes is achieved through the utilization of indirect light signals resulting from multiple reflections or scattering. The inherently weak nature of these signals, coupled with their susceptibility to noise, necessitates the integration of physical processes to ensure accurate reconstruction. This paper presents a parameterized inverse problem framework tailored for large-scale linear problems in 3D imaging reconstruction. Initially, a noise estimation module is employed to adaptively assess the noise levels present in transient data. Subsequently, a parameterized neural operator is developed to approximate the inverse mapping, facilitating end-to-end rapid image reconstruction. Our 3D image reconstruction framework, grounded in operator learning, is constructed through deep algorithm unfolding, which not only provides commendable model interpretability but also enables dynamic adaptation to varying noise levels in the acquired data, thereby ensuring consistently robust and accurate reconstruction outcomes. Furthermore, we introduce a novel method for the fusion of global and local spatiotemporal data features. By integrating structural and detailed information, this method significantly enhances both accuracy and robustness. Comprehensive numerical experiments conducted on both simulated and real datasets substantiate the efficacy of the proposed method. It demonstrates remarkable performance with fast scanning data and sparse illumination point data, offering a viable solution for NLOS imaging in complex scenarios.
>
---
#### [new 042] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit
- **分类: cs.CV**

- **简介: 论文提出LLMC+作为视觉-语言模型压缩基准工具包，解决技术分解、单一任务评估及孤立使用问题，系统研究不同压缩策略效果，揭示空间/时间冗余差异及综合压缩优势。**

- **链接: [http://arxiv.org/pdf/2508.09981v1](http://arxiv.org/pdf/2508.09981v1)**

> **作者:** Chengtao Lv; Bilang Zhang; Yang Yong; Ruihao Gong; Yushi Huang; Shiqiao Gu; Jiajun Wu; Yumeng Shi; Jinyang Guo; Wenya Wang
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Large Vision-Language Models (VLMs) exhibit impressive multi-modal capabilities but suffer from prohibitive computational and memory demands, due to their long visual token sequences and massive parameter sizes. To address these issues, recent works have proposed training-free compression methods. However, existing efforts often suffer from three major limitations: (1) Current approaches do not decompose techniques into comparable modules, hindering fair evaluation across spatial and temporal redundancy. (2) Evaluation confined to simple single-turn tasks, failing to reflect performance in realistic scenarios. (3) Isolated use of individual compression techniques, without exploring their joint potential. To overcome these gaps, we introduce LLMC+, a comprehensive VLM compression benchmark with a versatile, plug-and-play toolkit. LLMC+ supports over 20 algorithms across five representative VLM families and enables systematic study of token-level and model-level compression. Our benchmark reveals that: (1) Spatial and temporal redundancies demand distinct technical strategies. (2) Token reduction methods degrade significantly in multi-turn dialogue and detail-sensitive tasks. (3) Combining token and model compression achieves extreme compression with minimal performance loss. We believe LLMC+ will facilitate fair evaluation and inspire future research in efficient VLM. Our code is available at https://github.com/ModelTC/LightCompress.
>
---
#### [new 043] FineState-Bench: A Comprehensive Benchmark for Fine-Grained State Control in GUI Agents
- **分类: cs.CV**

- **简介: 论文提出FineState-Bench，针对GUI代理的细粒度状态控制问题，构建多平台基准框架，量化视觉定位能力，揭示当前GUI代理在基本定位上的瓶颈。**

- **链接: [http://arxiv.org/pdf/2508.09241v1](http://arxiv.org/pdf/2508.09241v1)**

> **作者:** Fengxian Ji; Jingpu Yang; Zirui Song; Yuanxi Wang; Zhexuan Cui; Yuke Li; Qian Jiang; Miao Fang; Xiuying Chen
>
> **备注:** submit/6682470 (Fengxian Ji)
>
> **摘要:** With the rapid advancement of generative artificial intelligence technology, Graphical User Interface (GUI) agents have demonstrated tremendous potential for autonomously managing daily tasks through natural language instructions. However, current evaluation frameworks for GUI agents suffer from fundamental flaws: existing benchmarks overly focus on coarse-grained task completion while neglecting fine-grained control capabilities crucial for real-world applications. To address this, we introduce FineState-Bench, the first evaluation and diagnostic standard for fine-grained GUI proxy operations, designed to quantify fine-grained control. This multi-platform (desktop, Web, mobile) framework includes 2257 task benchmarks in four components and uses a four-phase indicator for comprehensive perception-to-control assessment. To analyze perception and positioning for refined operations, we developed the plug-and-play Visual Diagnostic Assistant (VDA), enabling the first quantitative decoupling analysis of these capabilities. Experimental results on our benchmark show that the most advanced models achieve only 32.8% fine-grained interaction accuracy. Using our VDA in controlled experiments, quantifying the impact of visual capabilities, we showed that ideal visual localization boosts Gemini-2.5-Flash's success rate by 14.9\%. Our diagnostic framework confirms for the first time that the primary bottleneck for current GUI proxies is basic visual positioning capability.All resources are fully open-source. github: https://github.com/AnonymousThewarehouse/FineState-Bench huggingface: https://huggingface.co/datasets/Willtime2006/Static-FineBench
>
---
#### [new 044] January Food Benchmark (JFB): A Public Benchmark Dataset and Evaluation Suite for Multimodal Food Analysis
- **分类: cs.CV; cs.AI**

- **简介: 论文提出January Food Benchmark（JFB）数据集及评估框架，解决多模态食品分析中缺乏标准化数据与评价体系的问题，通过公开1000张食物图像及专用模型性能验证，实现对VLMs的全面评估。**

- **链接: [http://arxiv.org/pdf/2508.09966v1](http://arxiv.org/pdf/2508.09966v1)**

> **作者:** Amir Hosseinian; Ashkan Dehghani Zahedani; Umer Mansoor; Noosheen Hashemi; Mark Woodward
>
> **摘要:** Progress in AI for automated nutritional analysis is critically hampered by the lack of standardized evaluation methodologies and high-quality, real-world benchmark datasets. To address this, we introduce three primary contributions. First, we present the January Food Benchmark (JFB), a publicly available collection of 1,000 food images with human-validated annotations. Second, we detail a comprehensive benchmarking framework, including robust metrics and a novel, application-oriented overall score designed to assess model performance holistically. Third, we provide baseline results from both general-purpose Vision-Language Models (VLMs) and our own specialized model, january/food-vision-v1. Our evaluation demonstrates that the specialized model achieves an Overall Score of 86.2, a 12.1-point improvement over the best-performing general-purpose configuration. This work offers the research community a valuable new evaluation dataset and a rigorous framework to guide and benchmark future developments in automated nutritional analysis.
>
---
#### [new 045] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 论文提出RayletDF方法，通过射线距离场直接预测表面点，解决传统坐标方法计算成本高、泛化性差的问题，实现3D表面重建。**

- **链接: [http://arxiv.org/pdf/2508.09830v1](http://arxiv.org/pdf/2508.09830v1)**

> **作者:** Shenxing Wei; Jinxi Li; Yafei Yang; Siyuan Zhou; Bo Yang
>
> **备注:** ICCV 2025 Highlight. Shenxing and Jinxi are co-first authors. Code and data are available at: https://github.com/vLAR-group/RayletDF
>
> **摘要:** In this paper, we present a generalizable method for 3D surface reconstruction from raw point clouds or pre-estimated 3D Gaussians by 3DGS from RGB images. Unlike existing coordinate-based methods which are often computationally intensive when rendering explicit surfaces, our proposed method, named RayletDF, introduces a new technique called raylet distance field, which aims to directly predict surface points from query rays. Our pipeline consists of three key modules: a raylet feature extractor, a raylet distance field predictor, and a multi-raylet blender. These components work together to extract fine-grained local geometric features, predict raylet distances, and aggregate multiple predictions to reconstruct precise surface points. We extensively evaluate our method on multiple public real-world datasets, demonstrating superior performance in surface reconstruction from point clouds or 3D Gaussians. Most notably, our method achieves exceptional generalization ability, successfully recovering 3D surfaces in a single-forward pass across unseen datasets in testing.
>
---
#### [new 046] Automated Segmentation of Coronal Brain Tissue Slabs for 3D Neuropathology
- **分类: cs.CV; cs.AI**

- **简介: 本文提出基于U-Net的深度学习模型，自动化分割脑组织切片以实现3D神经病理学分析，解决人工分割成本高的问题，通过合成数据增强泛化能力，性能达高精度标准。**

- **链接: [http://arxiv.org/pdf/2508.09805v1](http://arxiv.org/pdf/2508.09805v1)**

> **作者:** Jonathan Williams Ramirez; Dina Zemlyanker; Lucas Deden-Binder; Rogeny Herisse; Erendira Garcia Pallares; Karthik Gopinath; Harshvardhan Gazula; Christopher Mount; Liana N. Kozanno; Michael S. Marshall; Theresa R. Connors; Matthew P. Frosch; Mark Montine; Derek H. Oakley; Christine L. Mac Donald; C. Dirk Keene; Bradley T. Hyman; Juan Eugenio Iglesias
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Advances in image registration and machine learning have recently enabled volumetric analysis of \emph{postmortem} brain tissue from conventional photographs of coronal slabs, which are routinely collected in brain banks and neuropathology laboratories worldwide. One caveat of this methodology is the requirement of segmentation of the tissue from photographs, which currently requires costly manual intervention. In this article, we present a deep learning model to automate this process. The automatic segmentation tool relies on a U-Net architecture that was trained with a combination of \textit{(i)}1,414 manually segmented images of both fixed and fresh tissue, from specimens with varying diagnoses, photographed at two different sites; and \textit{(ii)}~2,000 synthetic images with randomized contrast and corresponding masks generated from MRI scans for improved generalizability to unseen photographic setups. Automated model predictions on a subset of photographs not seen in training were analyzed to estimate performance compared to manual labels -- including both inter- and intra-rater variability. Our model achieved a median Dice score over 0.98, mean surface distance under 0.4~mm, and 95\% Hausdorff distance under 1.60~mm, which approaches inter-/intra-rater levels. Our tool is publicly available at surfer.nmr.mgh.harvard.edu/fswiki/PhotoTools.
>
---
#### [new 047] UltraLight Med-Vision Mamba for Classification of Neoplastic Progression in Tubular Adenomas
- **分类: cs.CV**

- **简介: 论文提出基于状态空间模型的Ultralight Med-Vision Mamba模型，用于分类管状腺瘤进展，提升早期癌前息肉识别与风险评估精度，适用于临床实时应用。**

- **链接: [http://arxiv.org/pdf/2508.09339v1](http://arxiv.org/pdf/2508.09339v1)**

> **作者:** Aqsa Sultana; Nordin Abouzahra; Ahmed Rahu; Brian Shula; Brandon Combs; Derrick Forchetti; Theus Aspiras; Vijayan K. Asari
>
> **摘要:** Identification of precancerous polyps during routine colonoscopy screenings is vital for their excision, lowering the risk of developing colorectal cancer. Advanced deep learning algorithms enable precise adenoma classification and stratification, improving risk assessment accuracy and enabling personalized surveillance protocols that optimize patient outcomes. Ultralight Med-Vision Mamba, a state-space based model (SSM), has excelled in modeling long- and short-range dependencies and image generalization, critical factors for analyzing whole slide images. Furthermore, Ultralight Med-Vision Mamba's efficient architecture offers advantages in both computational speed and scalability, making it a promising tool for real-time clinical deployment.
>
---
#### [new 048] NegFaceDiff: The Power of Negative Context in Identity-Conditioned Diffusion for Synthetic Face Generation
- **分类: cs.CV**

- **简介: 论文提出NegFaceDiff，通过负条件增强身份条件扩散模型的身份分离，提升合成人脸生成的准确性，实验显示FDR显著提高。**

- **链接: [http://arxiv.org/pdf/2508.09661v1](http://arxiv.org/pdf/2508.09661v1)**

> **作者:** Eduarda Caldeira; Naser Damer; Fadi Boutros
>
> **备注:** Accepted at ICCV Workshops
>
> **摘要:** The use of synthetic data as an alternative to authentic datasets in face recognition (FR) development has gained significant attention, addressing privacy, ethical, and practical concerns associated with collecting and using authentic data. Recent state-of-the-art approaches have proposed identity-conditioned diffusion models to generate identity-consistent face images, facilitating their use in training FR models. However, these methods often lack explicit sampling mechanisms to enforce inter-class separability, leading to identity overlap in the generated data and, consequently, suboptimal FR performance. In this work, we introduce NegFaceDiff, a novel sampling method that incorporates negative conditions into the identity-conditioned diffusion process. NegFaceDiff enhances identity separation by leveraging negative conditions that explicitly guide the model away from unwanted features while preserving intra-class consistency. Extensive experiments demonstrate that NegFaceDiff significantly improves the identity consistency and separability of data generated by identity-conditioned diffusion models. Specifically, identity separability, measured by the Fisher Discriminant Ratio (FDR), increases from 2.427 to 5.687. These improvements are reflected in FR systems trained on the NegFaceDiff dataset, which outperform models trained on data generated without negative conditions across multiple benchmarks.
>
---
#### [new 049] A Context-aware Attention and Graph Neural Network-based Multimodal Framework for Misogyny Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种多模态框架用于检测性别歧视内容，解决现有方法对女性针对性攻击识别不足的问题。框架结合注意力机制与图神经网络，通过多模态上下文建模与特征重建，提升毒性和描述特征学习能力，实现对 misogynistic 内容的有效检测。**

- **链接: [http://arxiv.org/pdf/2508.09175v1](http://arxiv.org/pdf/2508.09175v1)**

> **作者:** Mohammad Zia Ur Rehman; Sufyaan Zahoor; Areeb Manzoor; Musharaf Maqbool; Nagendra Kumar
>
> **备注:** Published in Information Processing & Management
>
> **摘要:** A substantial portion of offensive content on social media is directed towards women. Since the approaches for general offensive content detection face a challenge in detecting misogynistic content, it requires solutions tailored to address offensive content against women. To this end, we propose a novel multimodal framework for the detection of misogynistic and sexist content. The framework comprises three modules: the Multimodal Attention module (MANM), the Graph-based Feature Reconstruction Module (GFRM), and the Content-specific Features Learning Module (CFLM). The MANM employs adaptive gating-based multimodal context-aware attention, enabling the model to focus on relevant visual and textual information and generating contextually relevant features. The GFRM module utilizes graphs to refine features within individual modalities, while the CFLM focuses on learning text and image-specific features such as toxicity features and caption features. Additionally, we curate a set of misogynous lexicons to compute the misogyny-specific lexicon score from the text. We apply test-time augmentation in feature space to better generalize the predictions on diverse inputs. The performance of the proposed approach has been evaluated on two multimodal datasets, MAMI and MMHS150K, with 11,000 and 13,494 samples, respectively. The proposed method demonstrates an average improvement of 10.17% and 8.88% in macro-F1 over existing methods on the MAMI and MMHS150K datasets, respectively.
>
---
#### [new 050] Hierarchical Graph Attention Network for No-Reference Omnidirectional Image Quality Assessment
- **分类: cs.CV**

- **简介: 论文提出一种基于图注意力网络的无参考全景图像质量评估方法，解决局部非均匀失真问题，通过图结构建模视口关系，融合多阶段特征提取与长程质量交互，显著提升评估效果。**

- **链接: [http://arxiv.org/pdf/2508.09843v1](http://arxiv.org/pdf/2508.09843v1)**

> **作者:** Hao Yang; Xu Zhang; Jiaqi Ma; Linwei Zhu; Yun Zhang; Huan Zhang
>
> **摘要:** Current Omnidirectional Image Quality Assessment (OIQA) methods struggle to evaluate locally non-uniform distortions due to inadequate modeling of spatial variations in quality and ineffective feature representation capturing both local details and global context. To address this, we propose a graph neural network-based OIQA framework that explicitly models structural relationships between viewports to enhance perception of spatial distortion non-uniformity. Our approach employs Fibonacci sphere sampling to generate viewports with well-structured topology, representing each as a graph node. Multi-stage feature extraction networks then derive high-dimensional node representation. To holistically capture spatial dependencies, we integrate a Graph Attention Network (GAT) modeling fine-grained local distortion variations among adjacent viewports, and a graph transformer capturing long-range quality interactions across distant regions. Extensive experiments on two large-scale OIQA databases with complex spatial distortions demonstrate that our method significantly outperforms existing approaches, confirming its effectiveness and strong generalization capability.
>
---
#### [new 051] Echo-4o: Harnessing the Power of GPT-4o Synthetic Images for Improved Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出利用GPT-4o合成图像提升图像生成性能，解决开源模型不足问题，通过构建Echo-4o-Image数据集及两项评估基准，验证合成数据在补全罕见场景与可控监督方面的优势。**

- **链接: [http://arxiv.org/pdf/2508.09987v1](http://arxiv.org/pdf/2508.09987v1)**

> **作者:** Junyan Ye; Dongzhi Jiang; Zihao Wang; Leqi Zhu; Zhenghao Hu; Zilong Huang; Jun He; Zhiyuan Yan; Jinghua Yu; Hongsheng Li; Conghui He; Weijia Li
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Recently, GPT-4o has garnered significant attention for its strong performance in image generation, yet open-source models still lag behind. Several studies have explored distilling image data from GPT-4o to enhance open-source models, achieving notable progress. However, a key question remains: given that real-world image datasets already constitute a natural source of high-quality data, why should we use GPT-4o-generated synthetic data? In this work, we identify two key advantages of synthetic images. First, they can complement rare scenarios in real-world datasets, such as surreal fantasy or multi-reference image generation, which frequently occur in user queries. Second, they provide clean and controllable supervision. Real-world data often contains complex background noise and inherent misalignment between text descriptions and image content, whereas synthetic images offer pure backgrounds and long-tailed supervision signals, facilitating more accurate text-to-image alignment. Building on these insights, we introduce Echo-4o-Image, a 180K-scale synthetic dataset generated by GPT-4o, harnessing the power of synthetic image data to address blind spots in real-world coverage. Using this dataset, we fine-tune the unified multimodal generation baseline Bagel to obtain Echo-4o. In addition, we propose two new evaluation benchmarks for a more accurate and challenging assessment of image generation capabilities: GenEval++, which increases instruction complexity to mitigate score saturation, and Imagine-Bench, which focuses on evaluating both the understanding and generation of imaginative content. Echo-4o demonstrates strong performance across standard benchmarks. Moreover, applying Echo-4o-Image to other foundation models (e.g., OmniGen2, BLIP3-o) yields consistent performance gains across multiple metrics, highlighting the datasets strong transferability.
>
---
#### [new 052] CWFBind: Geometry-Awareness for Fast and Accurate Protein-Ligand Docking
- **分类: cs.CV; cs.CG; cs.LG**

- **简介: 该论文提出CWFBind，通过引入局部曲率特征和度感知权重机制，解决蛋白质-配体对接中几何信息缺失导致的准确性问题，提升预测效率与精度。**

- **链接: [http://arxiv.org/pdf/2508.09499v1](http://arxiv.org/pdf/2508.09499v1)**

> **作者:** Liyan Jia; Chuan-Xian Ren; Hong Yan
>
> **摘要:** Accurately predicting the binding conformation of small-molecule ligands to protein targets is a critical step in rational drug design. Although recent deep learning-based docking surpasses traditional methods in speed and accuracy, many approaches rely on graph representations and language model-inspired encoders while neglecting critical geometric information, resulting in inaccurate pocket localization and unrealistic binding conformations. In this study, we introduce CWFBind, a weighted, fast, and accurate docking method based on local curvature features. Specifically, we integrate local curvature descriptors during the feature extraction phase to enrich the geometric representation of both proteins and ligands, complementing existing chemical, sequence, and structural features. Furthermore, we embed degree-aware weighting mechanisms into the message passing process, enhancing the model's ability to capture spatial structural distinctions and interaction strengths. To address the class imbalance challenge in pocket prediction, CWFBind employs a ligand-aware dynamic radius strategy alongside an enhanced loss function, facilitating more precise identification of binding regions and key residues. Comprehensive experimental evaluations demonstrate that CWFBind achieves competitive performance across multiple docking benchmarks, offering a balanced trade-off between accuracy and efficiency.
>
---
#### [new 053] Predictive Uncertainty for Runtime Assurance of a Real-Time Computer Vision-Based Landing System
- **分类: cs.CV; cs.RO**

- **简介: 论文提出一种实时计算机视觉降落系统中的预测不确定性方法，通过高效神经网络、校准损失函数和Residual-based RAIM实现安全关键应用的运行时保障，提升姿态估计精度与故障检测能力。**

- **链接: [http://arxiv.org/pdf/2508.09732v1](http://arxiv.org/pdf/2508.09732v1)**

> **作者:** Romeo Valentin; Sydney M. Katz; Artur B. Carneiro; Don Walker; Mykel J. Kochenderfer
>
> **备注:** 8 pages, 5 figures, accepted at DASC 2025
>
> **摘要:** Recent advances in data-driven computer vision have enabled robust autonomous navigation capabilities for civil aviation, including automated landing and runway detection. However, ensuring that these systems meet the robustness and safety requirements for aviation applications remains a major challenge. In this work, we present a practical vision-based pipeline for aircraft pose estimation from runway images that represents a step toward the ability to certify these systems for use in safety-critical aviation applications. Our approach features three key innovations: (i) an efficient, flexible neural architecture based on a spatial Soft Argmax operator for probabilistic keypoint regression, supporting diverse vision backbones with real-time inference; (ii) a principled loss function producing calibrated predictive uncertainties, which are evaluated via sharpness and calibration metrics; and (iii) an adaptation of Residual-based Receiver Autonomous Integrity Monitoring (RAIM), enabling runtime detection and rejection of faulty model outputs. We implement and evaluate our pose estimation pipeline on a dataset of runway images. We show that our model outperforms baseline architectures in terms of accuracy while also producing well-calibrated uncertainty estimates with sub-pixel precision that can be used downstream for fault detection.
>
---
#### [new 054] CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection
- **分类: cs.CV; cs.CR**

- **简介: 论文提出一种基于异常检测的通用AI生成图像检测方法，利用CLIP编码器与无监督流模型，通过代理图像训练提升对新型生成模型的检测能力，解决传统方法对新模型泛化差的问题。**

- **链接: [http://arxiv.org/pdf/2508.09477v1](http://arxiv.org/pdf/2508.09477v1)**

> **作者:** Zhipeng Yuan; Kai Wang; Weize Quan; Dong-Ming Yan; Tieru Wu
>
> **摘要:** With the rapid advancement of AI generative models, the visual quality of AI-generated images (AIIs) has become increasingly close to natural images, which inevitably raises security concerns. Most AII detectors often employ the conventional image classification pipeline with natural images and AIIs (generated by a generative model), which can result in limited detection performance for AIIs from unseen generative models. To solve this, we proposed a universal AI-generated image detector from the perspective of anomaly detection. Our discriminator does not need to access any AIIs and learn a generalizable representation with unsupervised learning. Specifically, we use the pre-trained CLIP encoder as the feature extractor and design a normalizing flow-like unsupervised model. Instead of AIIs, proxy images, e.g., obtained by applying a spectral modification operation on natural images, are used for training. Our models are trained by minimizing the likelihood of proxy images, optionally combined with maximizing the likelihood of natural images. Extensive experiments demonstrate the effectiveness of our method on AIIs produced by various image generators.
>
---
#### [new 055] Physical Autoregressive Model for Robotic Manipulation without Action Pretraining
- **分类: cs.CV**

- **简介: 论文提出一种无需动作预训练的物理自回归模型（PAR），通过结合视频帧与动作生成机器人操作任务，利用视频预训练的物理知识实现精准预测与一致动作轨迹，提升机器人操控效率。**

- **链接: [http://arxiv.org/pdf/2508.09822v1](http://arxiv.org/pdf/2508.09822v1)**

> **作者:** Zijian Song; Sihan Qin; Tianshui Chen; Liang Lin; Guangrun Wang
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** The scarcity of manipulation data has motivated the use of pretrained large models from other modalities in robotics. In this work, we build upon autoregressive video generation models to propose a Physical Autoregressive Model (PAR), where physical tokens combine frames and actions to represent the joint evolution of the robot and its environment. PAR leverages the world knowledge embedded in video pretraining to understand physical dynamics without requiring action pretraining, enabling accurate video prediction and consistent action trajectories. It also adopts a DiT-based de-tokenizer to model frames and actions as continuous tokens, mitigating quantization errors and facilitating mutual enhancement. Furthermore, we incorporate a causal mask with inverse kinematics, parallel training, and the KV-cache mechanism to further improve performance and efficiency. Experiments on the ManiSkill benchmark show that PAR achieves a 100\% success rate on the PushCube task, matches the performance of action-pretrained baselines on other tasks, and accurately predicts future videos with tightly aligned action trajectories. These findings underscore a promising direction for robotic manipulation by transferring world knowledge from autoregressive video pretraining.
>
---
#### [new 056] Stable Diffusion Models are Secretly Good at Visual In-Context Learning
- **分类: cs.CV; cs.LG**

- **简介: 论文提出利用Stable Diffusion模型进行视觉上下文学习（V-ICL），无需微调即可适应多种任务，通过自注意力机制融合查询与示例提示的上下文，提升性能。**

- **链接: [http://arxiv.org/pdf/2508.09949v1](http://arxiv.org/pdf/2508.09949v1)**

> **作者:** Trevine Oorloff; Vishwanath Sindagi; Wele Gedara Chaminda Bandara; Ali Shafahi; Amin Ghiasi; Charan Prakash; Reza Ardekani
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Large language models (LLM) in natural language processing (NLP) have demonstrated great potential for in-context learning (ICL) -- the ability to leverage a few sets of example prompts to adapt to various tasks without having to explicitly update the model weights. ICL has recently been explored for computer vision tasks with promising early outcomes. These approaches involve specialized training and/or additional data that complicate the process and limit its generalizability. In this work, we show that off-the-shelf Stable Diffusion models can be repurposed for visual in-context learning (V-ICL). Specifically, we formulate an in-place attention re-computation within the self-attention layers of the Stable Diffusion architecture that explicitly incorporates context between the query and example prompts. Without any additional fine-tuning, we show that this repurposed Stable Diffusion model is able to adapt to six different tasks: foreground segmentation, single object detection, semantic segmentation, keypoint detection, edge detection, and colorization. For example, the proposed approach improves the mean intersection over union (mIoU) for the foreground segmentation task on Pascal-5i dataset by 8.9% and 3.2% over recent methods such as Visual Prompting and IMProv, respectively. Additionally, we show that the proposed method is able to effectively leverage multiple prompts through ensembling to infer the task better and further improve the performance.
>
---
#### [new 057] RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata
- **分类: cs.CV; cs.AI; I.2**

- **简介: 论文提出两阶段RampNet方法，利用政府数据生成大规模标注图像，提升检测性能至94%精度和0.9236 AP，是首个此类数据集。**

- **链接: [http://arxiv.org/pdf/2508.09415v1](http://arxiv.org/pdf/2508.09415v1)**

> **作者:** John S. O'Meara; Jared Hwang; Zeyu Wang; Michael Saugstad; Jon E. Froehlich
>
> **备注:** Accepted to the ICCV'25 Workshop on Vision Foundation Models and Generative AI for Accessibility: Challenges and Opportunities
>
> **摘要:** Curb ramps are critical for urban accessibility, but robustly detecting them in images remains an open problem due to the lack of large-scale, high-quality datasets. While prior work has attempted to improve data availability with crowdsourced or manually labeled data, these efforts often fall short in either quality or scale. In this paper, we introduce and evaluate a two-stage pipeline called RampNet to scale curb ramp detection datasets and improve model performance. In Stage 1, we generate a dataset of more than 210,000 annotated Google Street View (GSV) panoramas by auto-translating government-provided curb ramp location data to pixel coordinates in panoramic images. In Stage 2, we train a curb ramp detection model (modified ConvNeXt V2) from the generated dataset, achieving state-of-the-art performance. To evaluate both stages of our pipeline, we compare to manually labeled panoramas. Our generated dataset achieves 94.0% precision and 92.5% recall, and our detection model reaches 0.9236 AP -- far exceeding prior work. Our work contributes the first large-scale, high-quality curb ramp detection dataset, benchmark, and model.
>
---
#### [new 058] E-4DGS: High-Fidelity Dynamic Reconstruction from the Multi-view Event Cameras
- **分类: cs.CV**

- **简介: 论文提出基于多视角事件相机的动态高保真重建方法E-4DGS，解决传统RGB相机受限于光照、运动模糊和动态范围的问题，通过事件驱动的初始化、时间感知切片平滑及强度剪枝等技术提升重建精度，构建合成多视角数据集并验证其性能优势。**

- **链接: [http://arxiv.org/pdf/2508.09912v1](http://arxiv.org/pdf/2508.09912v1)**

> **作者:** Chaoran Feng; Zhenyu Tang; Wangbo Yu; Yatian Pang; Yian Zhao; Jianbin Zhao; Li Yuan; Yonghong Tian
>
> **备注:** 16 pages, 10 figures, 5 Tables, accepted by ACMMM 2025
>
> **摘要:** Novel view synthesis and 4D reconstruction techniques predominantly rely on RGB cameras, thereby inheriting inherent limitations such as the dependence on adequate lighting, susceptibility to motion blur, and a limited dynamic range. Event cameras, offering advantages of low power, high temporal resolution and high dynamic range, have brought a new perspective to addressing the scene reconstruction challenges in high-speed motion and low-light scenes. To this end, we propose E-4DGS, the first event-driven dynamic Gaussian Splatting approach, for novel view synthesis from multi-view event streams with fast-moving cameras. Specifically, we introduce an event-based initialization scheme to ensure stable training and propose event-adaptive slicing splatting for time-aware reconstruction. Additionally, we employ intensity importance pruning to eliminate floating artifacts and enhance 3D consistency, while incorporating an adaptive contrast threshold for more precise optimization. We design a synthetic multi-view camera setup with six moving event cameras surrounding the object in a 360-degree configuration and provide a benchmark multi-view event stream dataset that captures challenging motion scenarios. Our approach outperforms both event-only and event-RGB fusion baselines and paves the way for the exploration of multi-view event-based reconstruction as a novel approach for rapid scene capture.
>
---
#### [new 059] PaCo-FR: Patch-Pixel Aligned End-to-End Codebook Learning for Facial Representation Pre-training
- **分类: cs.CV**

- **简介: 论文提出PaCo-FR框架，通过结合掩码建模与补丁对齐，解决面部特征捕捉、结构保留及数据效率问题，实现高效预训练。**

- **链接: [http://arxiv.org/pdf/2508.09691v1](http://arxiv.org/pdf/2508.09691v1)**

> **作者:** Yin Xie; Zhichao Chen; Xiaoze Yu; Yongle Zhao; Xiang An; Kaicheng Yang; Zimin Ran; Jia Guo; Ziyong Feng; Jiankang Deng
>
> **摘要:** Facial representation pre-training is crucial for tasks like facial recognition, expression analysis, and virtual reality. However, existing methods face three key challenges: (1) failing to capture distinct facial features and fine-grained semantics, (2) ignoring the spatial structure inherent to facial anatomy, and (3) inefficiently utilizing limited labeled data. To overcome these, we introduce PaCo-FR, an unsupervised framework that combines masked image modeling with patch-pixel alignment. Our approach integrates three innovative components: (1) a structured masking strategy that preserves spatial coherence by aligning with semantically meaningful facial regions, (2) a novel patch-based codebook that enhances feature discrimination with multiple candidate tokens, and (3) spatial consistency constraints that preserve geometric relationships between facial components. PaCo-FR achieves state-of-the-art performance across several facial analysis tasks with just 2 million unlabeled images for pre-training. Our method demonstrates significant improvements, particularly in scenarios with varying poses, occlusions, and lighting conditions. We believe this work advances facial representation learning and offers a scalable, efficient solution that reduces reliance on expensive annotated datasets, driving more effective facial analysis systems.
>
---
#### [new 060] Topological Invariant-Based Iris Identification via Digital Homology and Machine Learning
- **分类: cs.CV; 55N31, 55U10, 68U10, 68T07; I.4.6; I.5.4; G.2.3**

- **简介: 论文提出基于数字同源拓扑不变量的虹膜识别方法，通过计算Betti数及比值提取特征，结合逻辑回归、SVM等模型与CNN对比，实现高精度识别，首次将拓扑不变量应用于虹膜识别，适用于其他生物特征领域。**

- **链接: [http://arxiv.org/pdf/2508.09555v1](http://arxiv.org/pdf/2508.09555v1)**

> **作者:** Ahmet Öztel; İsmet Karaca
>
> **备注:** 10 pages, 5 figures, includes visual abstract, focuses on topological invariants for iris recognition
>
> **摘要:** Objective - This study presents a biometric identification method based on topological invariants from 2D iris images, representing iris texture via formally defined digital homology and evaluating classification performance. Methods - Each normalized iris image (48x482 pixels) is divided into grids (e.g., 6x54 or 3x27). For each subregion, we compute Betti0, Betti1, and their ratio using a recent algorithm for homology groups in 2D digital images. The resulting invariants form a feature matrix used with logistic regression, KNN, and SVM (with PCA and 100 randomized repetitions). A convolutional neural network (CNN) is trained on raw images for comparison. Results - Logistic regression achieved 97.78 +/- 0.82% accuracy, outperforming CNN (96.44 +/- 1.32%) and other feature-based models. The topological features showed high accuracy with low variance. Conclusion - This is the first use of topological invariants from formal digital homology for iris recognition. The method offers a compact, interpretable, and accurate alternative to deep learning, useful when explainability or limited data is important. Beyond iris recognition, it can apply to other biometrics, medical imaging, materials science, remote sensing, and interpretable AI. It runs efficiently on CPU-only systems and produces robust, explainable features valuable for security-critical domains.
>
---
#### [new 061] Multimodal Sheaf-based Network for Glioblastoma Molecular Subtype Prediction
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种基于sheaf的多模态网络，用于胶质瘤分子亚型预测，解决现有方法无法有效融合MRI与组织病理学数据、丢失结构信息及处理缺失数据的问题，通过结构感知融合提升分类性能。**

- **链接: [http://arxiv.org/pdf/2508.09717v1](http://arxiv.org/pdf/2508.09717v1)**

> **作者:** Shekhnaz Idrissova; Islem Rekik
>
> **摘要:** Glioblastoma is a highly invasive brain tumor with rapid progression rates. Recent studies have shown that glioblastoma molecular subtype classification serves as a significant biomarker for effective targeted therapy selection. However, this classification currently requires invasive tissue extraction for comprehensive histopathological analysis. Existing multimodal approaches combining MRI and histopathology images are limited and lack robust mechanisms for preserving shared structural information across modalities. In particular, graph-based models often fail to retain discriminative features within heterogeneous graphs, and structural reconstruction mechanisms for handling missing or incomplete modality data are largely underexplored. To address these limitations, we propose a novel sheaf-based framework for structure-aware and consistent fusion of MRI and histopathology data. Our model outperforms baseline methods and demonstrates robustness in incomplete or missing data scenarios, contributing to the development of virtual biopsy tools for rapid diagnostics. Our source code is available at https://github.com/basiralab/MMSN/.
>
---
#### [new 062] Episodic Memory Representation for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 论文提出Video-EM框架，通过将关键帧建模为有序事件并利用链式思维，解决长视频理解中的时空关系和冗余问题，提升问答准确率。**

- **链接: [http://arxiv.org/pdf/2508.09486v1](http://arxiv.org/pdf/2508.09486v1)**

> **作者:** Yun Wang; Long Zhang; Jingren Liu; Jiaqi Yan; Zhanjie Zhang; Jiahao Zheng; Xun Yang; Dapeng Wu; Xiangyu Chen; Xuelong Li
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Video Large Language Models (Video-LLMs) excel at general video understanding but struggle with long-form videos due to context window limits. Consequently, recent approaches focus on keyframe retrieval, condensing lengthy videos into a small set of informative frames. Despite their practicality, these methods simplify the problem to static text image matching, overlooking spatio temporal relationships crucial for capturing scene transitions and contextual continuity, and may yield redundant keyframes with limited information, diluting salient cues essential for accurate video question answering. To address these limitations, we introduce Video-EM, a training free framework inspired by the principles of human episodic memory, designed to facilitate robust and contextually grounded reasoning. Rather than treating keyframes as isolated visual entities, Video-EM explicitly models them as temporally ordered episodic events, capturing both spatial relationships and temporal dynamics necessary for accurately reconstructing the underlying narrative. Furthermore, the framework leverages chain of thought (CoT) thinking with LLMs to iteratively identify a minimal yet highly informative subset of episodic memories, enabling efficient and accurate question answering by Video-LLMs. Extensive evaluations on the Video-MME, EgoSchema, HourVideo, and LVBench benchmarks confirm the superiority of Video-EM, which achieves highly competitive results with performance gains of 4-9 percent over respective baselines while utilizing fewer frames.
>
---
#### [new 063] TOTNet: Occlusion-Aware Temporal Tracking for Robust Ball Detection in Sports Videos
- **分类: cs.CV**

- **简介: 论文提出TOTNet，通过3D卷积、可见性加权损失和遮挡增强解决体育视频中球检测的遮挡问题，构建TTA数据集并实现RMSE降低至7.19，提升全遮挡帧识别精度至0.80，适用于快速场景下的离线分析。**

- **链接: [http://arxiv.org/pdf/2508.09650v1](http://arxiv.org/pdf/2508.09650v1)**

> **作者:** Hao Xu; Arbind Agrahari Baniya; Sam Wells; Mohamed Reda Bouadjenek; Richard Dazely; Sunil Aryal
>
> **备注:** 8 pages, 6 figures,
>
> **摘要:** Robust ball tracking under occlusion remains a key challenge in sports video analysis, affecting tasks like event detection and officiating. We present TOTNet, a Temporal Occlusion Tracking Network that leverages 3D convolutions, visibility-weighted loss, and occlusion augmentation to improve performance under partial and full occlusions. Developed in collaboration with Paralympics Australia, TOTNet is designed for real-world sports analytics. We introduce TTA, a new occlusion-rich table tennis dataset collected from professional-level Paralympic matches, comprising 9,159 samples with 1,996 occlusion cases. Evaluated on four datasets across tennis, badminton, and table tennis, TOTNet significantly outperforms prior state-of-the-art methods, reducing RMSE from 37.30 to 7.19 and improving accuracy on fully occluded frames from 0.63 to 0.80. These results demonstrate TOTNets effectiveness for offline sports analytics in fast-paced scenarios. Code and data access:\href{https://github.com/AugustRushG/TOTNet}{AugustRushG/TOTNet}.
>
---
#### [new 064] What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文研究标注者差异对皮肤病变分割的影响，构建IMAA++数据集，分析标注者、恶性等因子，提出IAA预测方法并提升多任务学习模型性能。**

- **链接: [http://arxiv.org/pdf/2508.09381v1](http://arxiv.org/pdf/2508.09381v1)**

> **作者:** Kumar Abhishek; Jeremy Kawahara; Ghassan Hamarneh
>
> **备注:** Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (MICCAI ISIC) 2025; 12 pages, 4 tables, 3 figures
>
> **摘要:** Medical image segmentation exhibits intra- and inter-annotator variability due to ambiguous object boundaries, annotator preferences, expertise, and tools, among other factors. Lesions with ambiguous boundaries, e.g., spiculated or infiltrative nodules, or irregular borders per the ABCD rule, are particularly prone to disagreement and are often associated with malignancy. In this work, we curate IMA++, the largest multi-annotator skin lesion segmentation dataset, on which we conduct an in-depth study of variability due to annotator, malignancy, tool, and skill factors. We find a statistically significant (p<0.001) association between inter-annotator agreement (IAA), measured using Dice, and the malignancy of skin lesions. We further show that IAA can be accurately predicted directly from dermoscopic images, achieving a mean absolute error of 0.108. Finally, we leverage this association by utilizing IAA as a "soft" clinical feature within a multi-task learning objective, yielding a 4.2% improvement in balanced accuracy averaged across multiple model architectures and across IMA++ and four public dermoscopic datasets. The code is available at https://github.com/sfu-mial/skin-IAV.
>
---
#### [new 065] SOI is the Root of All Evil: Quantifying and Breaking Similar Object Interference in Single Object Tracking
- **分类: cs.CV**

- **简介: 论文针对单目标跟踪中的相似对象干扰（SOI）问题，提出系统量化分析方法，构建SOIBench基准并开发基于视觉-语言模型的解决方案，显著提升跟踪性能。**

- **链接: [http://arxiv.org/pdf/2508.09524v1](http://arxiv.org/pdf/2508.09524v1)**

> **作者:** Yipei Wang; Shiyu Hu; Shukun Jia; Panxi Xu; Hongfei Ma; Yiping Ma; Jing Zhang; Xiaobo Lu; Xin Zhao
>
> **摘要:** In this paper, we present the first systematic investigation and quantification of Similar Object Interference (SOI), a long-overlooked yet critical bottleneck in Single Object Tracking (SOT). Through controlled Online Interference Masking (OIM) experiments, we quantitatively demonstrate that eliminating interference sources leads to substantial performance improvements (AUC gains up to 4.35) across all SOTA trackers, directly validating SOI as a primary constraint for robust tracking and highlighting the feasibility of external cognitive guidance. Building upon these insights, we adopt natural language as a practical form of external guidance, and construct SOIBench-the first semantic cognitive guidance benchmark specifically targeting SOI challenges. It automatically mines SOI frames through multi-tracker collective judgment and introduces a multi-level annotation protocol to generate precise semantic guidance texts. Systematic evaluation on SOIBench reveals a striking finding: existing vision-language tracking (VLT) methods fail to effectively exploit semantic cognitive guidance, achieving only marginal improvements or even performance degradation (AUC changes of -0.26 to +0.71). In contrast, we propose a novel paradigm employing large-scale vision-language models (VLM) as external cognitive engines that can be seamlessly integrated into arbitrary RGB trackers. This approach demonstrates substantial improvements under semantic cognitive guidance (AUC gains up to 0.93), representing a significant advancement over existing VLT methods. We hope SOIBench will serve as a standardized evaluation platform to advance semantic cognitive tracking research and contribute new insights to the tracking research community.
>
---
#### [new 066] Towards Comprehensive Cellular Characterisation of H&E slides
- **分类: cs.CV; q-bio.QM; I.2.10; I.4.8**

- **简介: 该论文提出HistoPLUS模型，针对H&E切片中的细胞检测与分类任务，解决罕见细胞类型识别及跨领域泛化问题，通过108,722核的13种细胞数据集训练，提升检测精度23.7%并扩展至7种罕见细胞类型，支持2个新癌症应用。**

- **链接: [http://arxiv.org/pdf/2508.09926v1](http://arxiv.org/pdf/2508.09926v1)**

> **作者:** Benjamin Adjadj; Pierre-Antoine Bannier; Guillaume Horent; Sebastien Mandela; Aurore Lyon; Kathryn Schutte; Ulysse Marteau; Valentin Gaury; Laura Dumont; Thomas Mathieu; Reda Belbahri; Benoît Schmauch; Eric Durand; Katharina Von Loga; Lucie Gillet
>
> **备注:** 33 pages, 4 figures
>
> **摘要:** Cell detection, segmentation and classification are essential for analyzing tumor microenvironments (TME) on hematoxylin and eosin (H&E) slides. Existing methods suffer from poor performance on understudied cell types (rare or not present in public datasets) and limited cross-domain generalization. To address these shortcomings, we introduce HistoPLUS, a state-of-the-art model for cell analysis, trained on a novel curated pan-cancer dataset of 108,722 nuclei covering 13 cell types. In external validation across 4 independent cohorts, HistoPLUS outperforms current state-of-the-art models in detection quality by 5.2% and overall F1 classification score by 23.7%, while using 5x fewer parameters. Notably, HistoPLUS unlocks the study of 7 understudied cell types and brings significant improvements on 8 of 13 cell types. Moreover, we show that HistoPLUS robustly transfers to two oncology indications unseen during training. To support broader TME biomarker research, we release the model weights and inference code at https://github.com/owkin/histoplus/.
>
---
#### [new 067] DSS-Prompt: Dynamic-Static Synergistic Prompting for Few-Shot Class-Incremental Learning
- **分类: cs.CV**

- **简介: 论文提出DSS-Prompt方法，针对少样本类增量学习中灾难性遗忘问题，通过动态与静态提示结合，利用多模态模型生成互补输入特征，实现增量学习时有效保留旧知识并提升性能。**

- **链接: [http://arxiv.org/pdf/2508.09785v1](http://arxiv.org/pdf/2508.09785v1)**

> **作者:** Linpu He; Yanan Li; Bingze Li; Elvis Han Cui; Donghui Wang
>
> **备注:** Accepted to ACMMM 2025
>
> **摘要:** Learning from large-scale pre-trained models with strong generalization ability has shown remarkable success in a wide range of downstream tasks recently, but it is still underexplored in the challenging few-shot class-incremental learning (FSCIL) task. It aims to continually learn new concepts from limited training samples without forgetting the old ones at the same time. In this paper, we introduce DSS-Prompt, a simple yet effective approach that transforms the pre-trained Vision Transformer with minimal modifications in the way of prompts into a strong FSCIL classifier. Concretely, we synergistically utilize two complementary types of prompts in each Transformer block: static prompts to bridge the domain gap between the pre-training and downstream datasets, thus enabling better adaption; and dynamic prompts to capture instance-aware semantics, thus enabling easy transfer from base to novel classes. Specially, to generate dynamic prompts, we leverage a pre-trained multi-modal model to extract input-related diverse semantics, thereby generating complementary input-aware prompts, and then adaptively adjust their importance across different layers. In this way, on top of the prompted visual embeddings, a simple prototype classifier can beat state-of-the-arts without further training on the incremental tasks. We conduct extensive experiments on four benchmarks to validate the effectiveness of our DSS-Prompt and show that it consistently achieves better performance than existing approaches on all datasets and can alleviate the catastrophic forgetting issue as well.
>
---
#### [new 068] Hierarchical Brain Structure Modeling for Predicting Genotype of Glioma
- **分类: cs.CV; cs.AI**

- **简介: 本研究提出Hi-SMGNN，通过整合区域与模块级结构/形态连接图，结合多模态交互与特征融合，提升IDH突变预测的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.09593v1](http://arxiv.org/pdf/2508.09593v1)**

> **作者:** Haotian Tang; Jianwei Chen; Xinrui Tang; Yunjia Wu; Zhengyang Miao; Chao Li
>
> **摘要:** Isocitrate DeHydrogenase (IDH) mutation status is a crucial biomarker for glioma prognosis. However, current prediction methods are limited by the low availability and noise of functional MRI. Structural and morphological connectomes offer a non-invasive alternative, yet existing approaches often ignore the brain's hierarchical organisation and multiscale interactions. To address this, we propose Hi-SMGNN, a hierarchical framework that integrates structural and morphological connectomes from regional to modular levels. It features a multimodal interaction module with a Siamese network and cross-modal attention, a multiscale feature fusion mechanism for reducing redundancy, and a personalised modular partitioning strategy to enhance individual specificity and interpretability. Experiments on the UCSF-PDGM dataset demonstrate that Hi-SMGNN outperforms baseline and state-of-the-art models, showing improved robustness and effectiveness in IDH mutation prediction.
>
---
#### [new 069] Generation of Indian Sign Language Letters, Numbers, and Words
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出结合ProGAN与SAGAN的注意力模型，生成高质量印度手语图像，解决分辨率与细节平衡问题，提升Inception Score和FID。**

- **链接: [http://arxiv.org/pdf/2508.09522v1](http://arxiv.org/pdf/2508.09522v1)**

> **作者:** Ajeet Kumar Yadav; Nishant Kumar; Rathna G N
>
> **备注:** 6 pages, 5 figures, 2024 International Conference on Intelligent Algorithms for Computational Intelligence Systems (IACIS)
>
> **摘要:** Sign language, which contains hand movements, facial expressions and bodily gestures, is a significant medium for communicating with hard-of-hearing people. A well-trained sign language community communicates easily, but those who don't know sign language face significant challenges. Recognition and generation are basic communication methods between hearing and hard-of-hearing individuals. Despite progress in recognition, sign language generation still needs to be explored. The Progressive Growing of Generative Adversarial Network (ProGAN) excels at producing high-quality images, while the Self-Attention Generative Adversarial Network (SAGAN) generates feature-rich images at medium resolutions. Balancing resolution and detail is crucial for sign language image generation. We are developing a Generative Adversarial Network (GAN) variant that combines both models to generate feature-rich, high-resolution, and class-conditional sign language images. Our modified Attention-based model generates high-quality images of Indian Sign Language letters, numbers, and words, outperforming the traditional ProGAN in Inception Score (IS) and Fr\'echet Inception Distance (FID), with improvements of 3.2 and 30.12, respectively. Additionally, we are publishing a large dataset incorporating high-quality images of Indian Sign Language alphabets, numbers, and 129 words.
>
---
#### [new 070] COXNet: Cross-Layer Fusion with Adaptive Alignment and Scale Integration for RGBT Tiny Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出COXNet框架，针对多模态RGBT小目标检测，解决空间错位、低光等挑战，通过跨层融合、动态对齐和优化标签分配提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.09533v1](http://arxiv.org/pdf/2508.09533v1)**

> **作者:** Peiran Peng; Tingfa Xu; Liqiang Song; Mengqi Zhu; Yuqiang Fang; Jianan Li
>
> **摘要:** Detecting tiny objects in multimodal Red-Green-Blue-Thermal (RGBT) imagery is a critical challenge in computer vision, particularly in surveillance, search and rescue, and autonomous navigation. Drone-based scenarios exacerbate these challenges due to spatial misalignment, low-light conditions, occlusion, and cluttered backgrounds. Current methods struggle to leverage the complementary information between visible and thermal modalities effectively. We propose COXNet, a novel framework for RGBT tiny object detection, addressing these issues through three core innovations: i) the Cross-Layer Fusion Module, fusing high-level visible and low-level thermal features for enhanced semantic and spatial accuracy; ii) the Dynamic Alignment and Scale Refinement module, correcting cross-modal spatial misalignments and preserving multi-scale features; and iii) an optimized label assignment strategy using the GeoShape Similarity Measure for better localization. COXNet achieves a 3.32\% mAP$_{50}$ improvement on the RGBTDronePerson dataset over state-of-the-art methods, demonstrating its effectiveness for robust detection in complex environments.
>
---
#### [new 071] MangaDiT: Reference-Guided Line Art Colorization with Hierarchical Attention in Diffusion Transformers
- **分类: cs.CV**

- **简介: 论文提出MangaDiT，用于参考引导的线条艺术着色，通过分层注意力机制与动态权重策略解决区域一致性问题，提升性能。**

- **链接: [http://arxiv.org/pdf/2508.09709v1](http://arxiv.org/pdf/2508.09709v1)**

> **作者:** Qianru Qiu; Jiafeng Mao; Kento Masui; Xueting Wang
>
> **备注:** Codes and benchmarks will be released soon
>
> **摘要:** Recent advances in diffusion models have significantly improved the performance of reference-guided line art colorization. However, existing methods still struggle with region-level color consistency, especially when the reference and target images differ in character pose or motion. Instead of relying on external matching annotations between the reference and target, we propose to discover semantic correspondences implicitly through internal attention mechanisms. In this paper, we present MangaDiT, a powerful model for reference-guided line art colorization based on Diffusion Transformers (DiT). Our model takes both line art and reference images as conditional inputs and introduces a hierarchical attention mechanism with a dynamic attention weighting strategy. This mechanism augments the vanilla attention with an additional context-aware path that leverages pooled spatial features, effectively expanding the model's receptive field and enhancing region-level color alignment. Experiments on two benchmark datasets demonstrate that our method significantly outperforms state-of-the-art approaches, achieving superior performance in both qualitative and quantitative evaluations.
>
---
#### [new 072] GoViG: Goal-Conditioned Visual Navigation Instruction Generation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GoViG任务，通过视觉预测与指令生成解决自主导航指令生成问题，利用原始视觉数据提升适应性，采用自回归多模态模型及两种推理策略，实现跨域泛化。**

- **链接: [http://arxiv.org/pdf/2508.09547v1](http://arxiv.org/pdf/2508.09547v1)**

> **作者:** Fengyi Wu; Yifei Dong; Zhi-Qi Cheng; Yilong Dai; Guangyu Chen; Hang Wang; Qi Dai; Alexander G. Hauptmann
>
> **备注:** Under review. Code: https://github.com/F1y1113/GoViG
>
> **摘要:** We introduce Goal-Conditioned Visual Navigation Instruction Generation (GoViG), a new task that aims to autonomously generate precise and contextually coherent navigation instructions solely from egocentric visual observations of initial and goal states. Unlike conventional approaches that rely on structured inputs such as semantic annotations or environmental maps, GoViG exclusively leverages raw egocentric visual data, substantially improving its adaptability to unseen and unstructured environments. Our method addresses this task by decomposing it into two interconnected subtasks: (1) visual forecasting, which predicts intermediate visual states bridging the initial and goal views; and (2) instruction generation, which synthesizes linguistically coherent instructions grounded in both observed and anticipated visuals. These subtasks are integrated within an autoregressive multimodal large language model trained with tailored objectives to ensure spatial accuracy and linguistic clarity. Furthermore, we introduce two complementary multimodal reasoning strategies, one-pass and interleaved reasoning, to mimic incremental human cognitive processes during navigation. To evaluate our method, we propose the R2R-Goal dataset, combining diverse synthetic and real-world trajectories. Empirical results demonstrate significant improvements over state-of-the-art methods, achieving superior BLEU-4 and CIDEr scores along with robust cross-domain generalization.
>
---
#### [new 073] Physics-guided Deep Unfolding Network for Enhanced Kronecker Compressive sensing
- **分类: cs.CV**

- **简介: 论文提出基于物理引导的深度展开网络，解决压缩感知中测量不相干性及隐式表示问题，通过AKCS模型与MACA机制提升重建精度与速度。**

- **链接: [http://arxiv.org/pdf/2508.09528v1](http://arxiv.org/pdf/2508.09528v1)**

> **作者:** Gang Qu; Ping Wang; Siming Zheng; Xin Yuan
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Deep networks have achieved remarkable success in image compressed sensing (CS) task, namely reconstructing a high-fidelity image from its compressed measurement. However, existing works are deficient inincoherent compressed measurement at sensing phase and implicit measurement representations at reconstruction phase, limiting the overall performance. In this work, we answer two questions: 1) how to improve the measurement incoherence for decreasing the ill-posedness; 2) how to learn informative representations from measurements. To this end, we propose a novel asymmetric Kronecker CS (AKCS) model and theoretically present its better incoherence than previous Kronecker CS with minimal complexity increase. Moreover, we reveal that the unfolding networks' superiority over non-unfolding ones result from sufficient gradient descents, called explicit measurement representations. We propose a measurement-aware cross attention (MACA) mechanism to learn implicit measurement representations. We integrate AKCS and MACA into widely-used unfolding architecture to get a measurement-enhanced unfolding network (MEUNet). Extensive experiences demonstrate that our MEUNet achieves state-of-the-art performance in reconstruction accuracy and inference speed.
>
---
#### [new 074] Autonomous AI Bird Feeder for Backyard Biodiversity Monitoring
- **分类: cs.CV**

- **简介: 论文提出一种低成本自主鸟类监测系统，通过AI模型实现本地化鸟类识别与分类，解决传统方法依赖云服务、隐私风险及误报问题，验证其在城市花园的实际应用可行性。**

- **链接: [http://arxiv.org/pdf/2508.09398v1](http://arxiv.org/pdf/2508.09398v1)**

> **作者:** El Mustapha Mansouri
>
> **备注:** Preprint; 8 pages, 5 figures, 1 table; IEEEtran conference format. Code: https://github.com/E-zClap/bird-classifier
>
> **摘要:** This paper presents a low cost, on premise system for autonomous backyard bird monitoring in Belgian urban gardens. A motion triggered IP camera uploads short clips via FTP to a local server, where frames are sampled and birds are localized with Detectron2; cropped regions are then classified by an EfficientNet-B3 model fine tuned on a 40-species Belgian subset derived from a larger Kaggle corpus. All processing runs on commodity hardware without a discrete GPU, preserving privacy and avoiding cloud fees. The physical feeder uses small entry ports (30 mm) to exclude pigeons and reduce nuisance triggers. Detector-guided cropping improves classification accuracy over raw-frame classification. The classifier attains high validation performance on the curated subset (about 99.5 percent) and delivers practical field accuracy (top-1 about 88 percent) on held-out species, demonstrating feasibility for citizen-science-grade biodiversity logging at home.
>
---
#### [new 075] GANime: Generating Anime and Manga Character Drawings from Sketches with Deep Learning
- **分类: cs.CV; cs.LG**

- **简介: 论文提出基于C-GAN的GANime模型，解决从草图生成高质量动漫角色图像的任务，通过实验验证C-GAN在生成质量与分辨率上的优势。**

- **链接: [http://arxiv.org/pdf/2508.09207v1](http://arxiv.org/pdf/2508.09207v1)**

> **作者:** Tai Vu; Robert Yang
>
> **摘要:** The process of generating fully colorized drawings from sketches is a large, usually costly bottleneck in the manga and anime industry. In this study, we examine multiple models for image-to-image translation between anime characters and their sketches, including Neural Style Transfer, C-GAN, and CycleGAN. By assessing them qualitatively and quantitatively, we find that C-GAN is the most effective model that is able to produce high-quality and high-resolution images close to those created by humans.
>
---
#### [new 076] GazeLT: Visual attention-guided long-tailed disease classification in chest radiographs
- **分类: cs.CV**

- **简介: 论文提出GazeLT，通过视觉注意力的整合与分解机制解决长尾病分类问题，利用放射科医生的眼动规律提升模型性能，实验证明其在NIH-CXR-LT和MIMIC-CXR-LT数据集上准确率提升4.1%和21.7%。**

- **链接: [http://arxiv.org/pdf/2508.09478v1](http://arxiv.org/pdf/2508.09478v1)**

> **作者:** Moinak Bhattacharya; Gagandeep Singh; Shubham Jain; Prateek Prasanna
>
> **摘要:** In this work, we present GazeLT, a human visual attention integration-disintegration approach for long-tailed disease classification. A radiologist's eye gaze has distinct patterns that capture both fine-grained and coarser level disease related information. While interpreting an image, a radiologist's attention varies throughout the duration; it is critical to incorporate this into a deep learning framework to improve automated image interpretation. Another important aspect of visual attention is that apart from looking at major/obvious disease patterns, experts also look at minor/incidental findings (few of these constituting long-tailed classes) during the course of image interpretation. GazeLT harnesses the temporal aspect of the visual search process, via an integration and disintegration mechanism, to improve long-tailed disease classification. We show the efficacy of GazeLT on two publicly available datasets for long-tailed disease classification, namely the NIH-CXR-LT (n=89237) and the MIMIC-CXR-LT (n=111898) datasets. GazeLT outperforms the best long-tailed loss by 4.1% and the visual attention-based baseline by 21.7% in average accuracy metrics for these datasets. Our code is available at https://github.com/lordmoinak1/gazelt.
>
---
#### [new 077] WEC-DG: Multi-Exposure Wavelet Correction Method Guided by Degradation Description
- **分类: cs.CV**

- **简介: 论文提出一种基于小波的多曝光校正方法WEC-DG，解决光照条件差异导致的曝光不一致与细节丢失问题，通过退化描述符保障曝光一致性并分步处理低频与高频信息，提升图像质量。**

- **链接: [http://arxiv.org/pdf/2508.09565v1](http://arxiv.org/pdf/2508.09565v1)**

> **作者:** Ming Zhao; Pingping Liu; Tongshun Zhang; Zhe Zhang
>
> **摘要:** Multi-exposure correction technology is essential for restoring images affected by insufficient or excessive lighting, enhancing the visual experience by improving brightness, contrast, and detail richness. However, current multi-exposure correction methods often encounter challenges in addressing intra-class variability caused by diverse lighting conditions, shooting environments, and weather factors, particularly when processing images captured at a single exposure level. To enhance the adaptability of these models under complex imaging conditions, this paper proposes a Wavelet-based Exposure Correction method with Degradation Guidance (WEC-DG). Specifically, we introduce a degradation descriptor within the Exposure Consistency Alignment Module (ECAM) at both ends of the processing pipeline to ensure exposure consistency and achieve final alignment. This mechanism effectively addresses miscorrected exposure anomalies caused by existing methods' failure to recognize 'blurred' exposure degradation. Additionally, we investigate the light-detail decoupling properties of the wavelet transform to design the Exposure Restoration and Detail Reconstruction Module (EDRM), which processes low-frequency information related to exposure enhancement before utilizing high-frequency information as a prior guide for reconstructing spatial domain details. This serial processing strategy guarantees precise light correction and enhances detail recovery. Extensive experiments conducted on multiple public datasets demonstrate that the proposed method outperforms existing algorithms, achieving significant performance improvements and validating its effectiveness and practical applicability.
>
---
#### [new 078] WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization
- **分类: cs.CV; cs.RO; I.4.10**

- **简介: 论文提出WeatherPrompt框架，通过多模态融合解决无人机视觉地理定位在恶劣天气下的泛化问题，采用动态门控机制与对比学习优化场景与天气特征分离，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2508.09560v1](http://arxiv.org/pdf/2508.09560v1)**

> **作者:** Jiahao Wen; Hang Yu; Zhedong Zheng
>
> **备注:** 13 pages, 4figures
>
> **摘要:** Visual geo-localization for drones faces critical degradation under weather perturbations, \eg, rain and fog, where existing methods struggle with two inherent limitations: 1) Heavy reliance on limited weather categories that constrain generalization, and 2) Suboptimal disentanglement of entangled scene-weather features through pseudo weather categories. We present WeatherPrompt, a multi-modality learning paradigm that establishes weather-invariant representations through fusing the image embedding with the text context. Our framework introduces two key contributions: First, a Training-free Weather Reasoning mechanism that employs off-the-shelf large multi-modality models to synthesize multi-weather textual descriptions through human-like reasoning. It improves the scalability to unseen or complex weather, and could reflect different weather strength. Second, to better disentangle the scene and weather feature, we propose a multi-modality framework with the dynamic gating mechanism driven by the text embedding to adaptively reweight and fuse visual features across modalities. The framework is further optimized by the cross-modal objectives, including image-text contrastive learning and image-text matching, which maps the same scene with different weather conditions closer in the respresentation space. Extensive experiments validate that, under diverse weather conditions, our method achieves competitive recall rates compared to state-of-the-art drone geo-localization methods. Notably, it improves Recall@1 by +13.37\% under night conditions and by 18.69\% under fog and snow conditions.
>
---
#### [new 079] ARI3D: A Software for Interactive Quantification of Regions in X-Ray CT 3D Images
- **分类: cs.CV; cs.SE**

- **简介: 论文提出ARI3D软件，用于交互式量化三维X射线CT图像，解决伪影问题，提升微结构分析精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.09849v1](http://arxiv.org/pdf/2508.09849v1)**

> **作者:** Jan Phillipp Albrecht; Jose R. A. Godinho; Christina Hübers; Deborah Schmidt
>
> **备注:** 2 figures and 6 pages main article, 17 pages total, 8 figures total, to be published in SoftwareX
>
> **摘要:** X-ray computed tomography (CT) is the main 3D technique for imaging the internal microstructures of materials. Quantitative analysis of the microstructures is usually achieved by applying a sequence of steps that are implemented to the entire 3D image. This is challenged by various imaging artifacts inherent from the technique, e.g., beam hardening and partial volume. Consequently, the analysis requires users to make a number of decisions to segment and classify the microstructures based on the voxel gray-values. In this context, a software tool, here called ARI3D, is proposed to interactively analyze regions in three-dimensional X-ray CT images, assisting users through the various steps of a protocol designed to classify and quantify objects within regions of a three-dimensional image. ARI3D aims to 1) Improve phase identification; 2) Account for partial volume effect; 3) Increase the detection limit and accuracy of object quantification; and 4) Harmonize quantitative 3D analysis that can be implemented in different fields of science.
>
---
#### [new 080] Slot Attention-based Feature Filtering for Few-Shot Learning
- **分类: cs.CV**

- **简介: 该论文针对少样本学习中无关特征干扰问题，提出SAFF方法，通过槽注意力与patch嵌入结合，过滤弱特征以提升分类性能，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.09699v1](http://arxiv.org/pdf/2508.09699v1)**

> **作者:** Javier Rodenas; Eduardo Aguilar; Petia Radeva
>
> **备注:** CVPR Workshop LatinX 2025
>
> **摘要:** Irrelevant features can significantly degrade few-shot learn ing performance. This problem is used to match queries and support images based on meaningful similarities despite the limited data. However, in this process, non-relevant fea tures such as background elements can easily lead to confu sion and misclassification. To address this issue, we pro pose Slot Attention-based Feature Filtering for Few-Shot Learning (SAFF) that leverages slot attention mechanisms to discriminate and filter weak features, thereby improving few-shot classification performance. The key innovation of SAFF lies in its integration of slot attention with patch em beddings, unifying class-aware slots into a single attention mechanism to filter irrelevant features effectively. We intro duce a similarity matrix that computes across support and query images to quantify the relevance of filtered embed dings for classification. Through experiments, we demon strate that Slot Attention performs better than other atten tion mechanisms, capturing discriminative features while reducing irrelevant information. We validate our approach through extensive experiments on few-shot learning bench marks: CIFAR-FS, FC100, miniImageNet and tieredIma geNet, outperforming several state-of-the-art methods.
>
---
#### [new 081] FusionEnsemble-Net: An Attention-Based Ensemble of Spatiotemporal Networks for Multimodal Sign Language Recognition
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出一种基于注意力机制的多模态时空网络融合框架FusionEnsemble-Net，用于解决复杂手语识别任务，通过动态融合视觉与运动数据提升准确率，实验显示在MultiMeDaLIS数据集上达到99.44%测试精度。**

- **链接: [http://arxiv.org/pdf/2508.09362v1](http://arxiv.org/pdf/2508.09362v1)**

> **作者:** Md. Milon Islam; Md Rezwanul Haque; S M Taslim Uddin Raju; Fakhri Karray
>
> **备注:** Accepted for the IEEE/CVF International Conference on Computer Vision (ICCV), Honolulu, Hawaii, USA. 1st MSLR Workshop 2025
>
> **摘要:** Accurate recognition of sign language in healthcare communication poses a significant challenge, requiring frameworks that can accurately interpret complex multimodal gestures. To deal with this, we propose FusionEnsemble-Net, a novel attention-based ensemble of spatiotemporal networks that dynamically fuses visual and motion data to enhance recognition accuracy. The proposed approach processes RGB video and range Doppler map radar modalities synchronously through four different spatiotemporal networks. For each network, features from both modalities are continuously fused using an attention-based fusion module before being fed into an ensemble of classifiers. Finally, the outputs of these four different fused channels are combined in an ensemble classification head, thereby enhancing the model's robustness. Experiments demonstrate that FusionEnsemble-Net outperforms state-of-the-art approaches with a test accuracy of 99.44% on the large-scale MultiMeDaLIS dataset for Italian Sign Language. Our findings indicate that an ensemble of diverse spatiotemporal networks, unified by attention-based fusion, yields a robust and accurate framework for complex, multimodal isolated gesture recognition tasks. The source code is available at: https://github.com/rezwanh001/Multimodal-Isolated-Italian-Sign-Language-Recognition.
>
---
#### [new 082] COME: Dual Structure-Semantic Learning with Collaborative MoE for Universal Lesion Detection Across Heterogeneous Ultrasound Datasets
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出基于协作混合专家（COME）的通用病变检测方法，解决异质超声数据集泛化问题，通过双结构-语义共享与跨数据集经验提升性能。**

- **链接: [http://arxiv.org/pdf/2508.09886v1](http://arxiv.org/pdf/2508.09886v1)**

> **作者:** Lingyu Chen; Yawen Zeng; Yue Wang; Peng Wan; Guo-chen Ning; Hongen Liao; Daoqiang Zhang; Fang Chen
>
> **备注:** ICCV 2025
>
> **摘要:** Conventional single-dataset training often fails with new data distributions, especially in ultrasound (US) image analysis due to limited data, acoustic shadows, and speckle noise. Therefore, constructing a universal framework for multi-heterogeneous US datasets is imperative. However, a key challenge arises: how to effectively mitigate inter-dataset interference while preserving dataset-specific discriminative features for robust downstream task? Previous approaches utilize either a single source-specific decoder or a domain adaptation strategy, but these methods experienced a decline in performance when applied to other domains. Considering this, we propose a Universal Collaborative Mixture of Heterogeneous Source-Specific Experts (COME). Specifically, COME establishes dual structure-semantic shared experts that create a universal representation space and then collaborate with source-specific experts to extract discriminative features through providing complementary features. This design enables robust generalization by leveraging cross-datasets experience distributions and providing universal US priors for small-batch or unseen data scenarios. Extensive experiments under three evaluation modes (single-dataset, intra-organ, and inter-organ integration datasets) demonstrate COME's superiority, achieving significant mean AP improvements over state-of-the-art methods. Our project is available at: https://universalcome.github.io/UniversalCOME/.
>
---
#### [new 083] ViMoNet: A Multimodal Vision-Language Framework for Human Behavior Understanding from Motion and Video
- **分类: cs.CV**

- **简介: 本论文提出ViMoNet多模态框架，融合运动与视频数据研究人类行为理解，通过联合训练策略结合详细运动文本与通用视频文本，构建高效模型并创建VIMOS数据集及ViMoNet-Bench基准，验证其在动作描述、理解与解释上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.09818v1](http://arxiv.org/pdf/2508.09818v1)**

> **作者:** Rajan Das Gupta; Md Yeasin Rahat; Nafiz Fahad; Abir Ahmed; Liew Tze Hui
>
> **备注:** Accepted in ICCVDM '25
>
> **摘要:** This study investigates how large language models (LLMs) can be used to understand human behavior using motion and video data. We think that mixing both types is essential to completely capture the nuanced movements and meanings of human actions, in contrast to recent models that simply concentrate on motion data or films. To address this, we provide ViMoNet, a straightforward yet effective framework for comprehending, characterizing, and deducing human action. ViMoNet employs a joint training strategy that leverages the advantages of two data types: detailed motion-text data, which is more exact, and generic video-text data, which is more comprehensive but less detailed. This aids in the model's acquisition of rich data regarding time and space in human behavior. Additionally, we provide a brand new dataset named VIMOS that contains a variety of films, motion sequences, instructions, and subtitles. We developed ViMoNet-Bench, a standardized benchmark with carefully labeled samples, to evaluate how well models understand human behavior. Our tests show that ViMoNet outperforms existing methods in caption generation, motion understanding, and behavior interpretation.
>
---
#### [new 084] MUJICA: Reforming SISR Models for PBR Material Super-Resolution via Cross-Map Attention
- **分类: cs.CV**

- **简介: 论文提出MUJICA，通过跨模态注意力改进SISR模型，解决PBR材料跨模态不一致及特征建模问题，提升超分辨率性能。**

- **链接: [http://arxiv.org/pdf/2508.09802v1](http://arxiv.org/pdf/2508.09802v1)**

> **作者:** Xin Du; Maoyuan Xu; Zhi Ying
>
> **摘要:** Physically Based Rendering (PBR) materials are typically characterized by multiple 2D texture maps such as basecolor, normal, metallic, and roughness which encode spatially-varying bi-directional reflectance distribution function (SVBRDF) parameters to model surface reflectance properties and microfacet interactions. Upscaling SVBRDF material is valuable for modern 3D graphics applications. However, existing Single Image Super-Resolution (SISR) methods struggle with cross-map inconsistency, inadequate modeling of modality-specific features, and limited generalization due to data distribution shifts. In this work, we propose Multi-modal Upscaling Joint Inference via Cross-map Attention (MUJICA), a flexible adapter that reforms pre-trained Swin-transformer-based SISR models for PBR material super-resolution. MUJICA is seamlessly attached after the pre-trained and frozen SISR backbone. It leverages cross-map attention to fuse features while preserving remarkable reconstruction ability of the pre-trained SISR model. Applied to SISR models such as SwinIR, DRCT, and HMANet, MUJICA improves PSNR, SSIM, and LPIPS scores while preserving cross-map consistency. Experiments demonstrate that MUJICA enables efficient training even with limited resources and delivers state-of-the-art performance on PBR material datasets.
>
---
#### [new 085] SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images
- **分类: cs.CV**

- **简介: 论文提出SkySplat框架，针对多时相稀疏卫星图像的3D重建难题，通过整合RPC模型与自监督机制，消除真高程依赖，提升几何约束与泛化能力，实现86倍速度提升及1.8m精度，跨数据集表现优异。**

- **链接: [http://arxiv.org/pdf/2508.09479v1](http://arxiv.org/pdf/2508.09479v1)**

> **作者:** Xuejun Huang; Xinyi Liu; Yi Wan; Zhi Zheng; Bin Zhang; Mingtao Xiong; Yingying Pei; Yongjun Zhang
>
> **摘要:** Three-dimensional scene reconstruction from sparse-view satellite images is a long-standing and challenging task. While 3D Gaussian Splatting (3DGS) and its variants have recently attracted attention for its high efficiency, existing methods remain unsuitable for satellite images due to incompatibility with rational polynomial coefficient (RPC) models and limited generalization capability. Recent advances in generalizable 3DGS approaches show potential, but they perform poorly on multi-temporal sparse satellite images due to limited geometric constraints, transient objects, and radiometric inconsistencies. To address these limitations, we propose SkySplat, a novel self-supervised framework that integrates the RPC model into the generalizable 3DGS pipeline, enabling more effective use of sparse geometric cues for improved reconstruction. SkySplat relies only on RGB images and radiometric-robust relative height supervision, thereby eliminating the need for ground-truth height maps. Key components include a Cross-Self Consistency Module (CSCM), which mitigates transient object interference via consistency-based masking, and a multi-view consistency aggregation strategy that refines reconstruction results. Compared to per-scene optimization methods, SkySplat achieves an 86 times speedup over EOGS with higher accuracy. It also outperforms generalizable 3DGS baselines, reducing MAE from 13.18 m to 1.80 m on the DFC19 dataset significantly, and demonstrates strong cross-dataset generalization on the MVS3D benchmark.
>
---
#### [new 086] Lung-DDPM+: Efficient Thoracic CT Image Synthesis using Diffusion Probabilistic Model
- **分类: cs.CV**

- **简介: 论文提出Lung-DDPM+，通过结合结节语义布局与肺部DPM求解器，提升胸CT图像生成效率与质量，解决传统模型效率低、解剖精度不足的问题，实验表明其采样速度提升14倍，保持高保真度，适用于肿瘤合成等医疗影像任务。**

- **链接: [http://arxiv.org/pdf/2508.09327v1](http://arxiv.org/pdf/2508.09327v1)**

> **作者:** Yifan Jiang; Ahmad Shariftabrizi; Venkata SK. Manem
>
> **摘要:** Generative artificial intelligence (AI) has been playing an important role in various domains. Leveraging its high capability to generate high-fidelity and diverse synthetic data, generative AI is widely applied in diagnostic tasks, such as lung cancer diagnosis using computed tomography (CT). However, existing generative models for lung cancer diagnosis suffer from low efficiency and anatomical imprecision, which limit their clinical applicability. To address these drawbacks, we propose Lung-DDPM+, an improved version of our previous model, Lung-DDPM. This novel approach is a denoising diffusion probabilistic model (DDPM) guided by nodule semantic layouts and accelerated by a pulmonary DPM-solver, enabling the method to focus on lesion areas while achieving a better trade-off between sampling efficiency and quality. Evaluation results on the public LIDC-IDRI dataset suggest that the proposed method achieves 8$\times$ fewer FLOPs (floating point operations per second), 6.8$\times$ lower GPU memory consumption, and 14$\times$ faster sampling compared to Lung-DDPM. Moreover, it maintains comparable sample quality to both Lung-DDPM and other state-of-the-art (SOTA) generative models in two downstream segmentation tasks. We also conducted a Visual Turing Test by an experienced radiologist, showing the advanced quality and fidelity of synthetic samples generated by the proposed method. These experimental results demonstrate that Lung-DDPM+ can effectively generate high-quality thoracic CT images with lung nodules, highlighting its potential for broader applications, such as general tumor synthesis and lesion generation in medical imaging. The code and pretrained models are available at https://github.com/Manem-Lab/Lung-DDPM-PLUS.
>
---
#### [new 087] $Δ$-AttnMask: Attention-Guided Masked Hidden States for Efficient Data Selection and Augmentation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出Δ-AttnMask框架，通过注意力引导的掩码量化样本质量，实现多模态数据高效选择与增强，无需领域标签或辅助模型，提升VIF性能。**

- **链接: [http://arxiv.org/pdf/2508.09199v1](http://arxiv.org/pdf/2508.09199v1)**

> **作者:** Jucheng Hu; Suorong Yang; Dongzhan Zhou
>
> **摘要:** Visual Instruction Finetuning (VIF) is pivotal for post-training Vision-Language Models (VLMs). Unlike unimodal instruction finetuning in plain-text large language models, which mainly requires instruction datasets to enable model instruction-following ability, VIF also requires multimodal data to enable joint visual and textual understanding; therefore, it typically requires more data. Consequently, VIF imposes stricter data selection challenges: the method must scale efficiently to handle larger data demands while ensuring the quality of both visual and textual content, as well as their alignment. Despite its critical impact on performance, data selection for VIF remains an understudied area. In this paper, we propose $\Delta$-AttnMask. This data-efficient framework quantifies sample quality through attention-guided masking of the model's hidden states, jointly evaluating image-text pairs without requiring domain labels, auxiliary models, or extra training. By computing loss differences ($\Delta$) between the original states and states masked using high-attention regions, $\Delta$-AttnMask intrinsically assesses sample quality. Experiments across multiple VLMs and datasets show that $\Delta$-AttnMask achieves state-of-the-art performance with just 20% of data, accelerating training by 5x while surpassing full-dataset baselines by +10.1% in overall accuracy. Its model-agnostic and data-agnostic design ensures broad applicability across modalities and architectures.
>
---
#### [new 088] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion
- **分类: cs.CV**

- **简介: 论文提出训练免费的Dual Recursive Feedback（DRF）系统，通过外观与生成反馈递归优化潜层，解决T2I模型在结构与姿态控制中的不足，提升图像生成质量。**

- **链接: [http://arxiv.org/pdf/2508.09575v1](http://arxiv.org/pdf/2508.09575v1)**

> **作者:** Jiwon Kim; Pureum Kim; SeonHwa Kim; Soobin Park; Eunju Cha; Kyong Hwan Jin
>
> **摘要:** Recent advancements in controllable text-to-image (T2I) diffusion models, such as Ctrl-X and FreeControl, have demonstrated robust spatial and appearance control without requiring auxiliary module training. However, these models often struggle to accurately preserve spatial structures and fail to capture fine-grained conditions related to object poses and scene layouts. To address these challenges, we propose a training-free Dual Recursive Feedback (DRF) system that properly reflects control conditions in controllable T2I models. The proposed DRF consists of appearance feedback and generation feedback that recursively refines the intermediate latents to better reflect the given appearance information and the user's intent. This dual-update mechanism guides latent representations toward reliable manifolds, effectively integrating structural and appearance attributes. Our approach enables fine-grained generation even between class-invariant structure-appearance fusion, such as transferring human motion onto a tiger's form. Extensive experiments demonstrate the efficacy of our method in producing high-quality, semantically coherent, and structurally consistent image generations. Our source code is available at https://github.com/jwonkm/DRF.
>
---
#### [new 089] HumanGenesis: Agent-Based Geometric and Generative Modeling for Synthetic Human Dynamics
- **分类: cs.CV**

- **简介: 该论文提出HumanGenesis框架，通过四代理协同解决合成人类动态中的几何不一致、重建粗糙及运动泛化问题，实现高保真视频生成，显著提升表现力与场景整合。**

- **链接: [http://arxiv.org/pdf/2508.09858v1](http://arxiv.org/pdf/2508.09858v1)**

> **作者:** Weiqi Li; Zehao Zhang; Liang Lin; Guangrun Wang
>
> **摘要:** \textbf{Synthetic human dynamics} aims to generate photorealistic videos of human subjects performing expressive, intention-driven motions. However, current approaches face two core challenges: (1) \emph{geometric inconsistency} and \emph{coarse reconstruction}, due to limited 3D modeling and detail preservation; and (2) \emph{motion generalization limitations} and \emph{scene inharmonization}, stemming from weak generative capabilities. To address these, we present \textbf{HumanGenesis}, a framework that integrates geometric and generative modeling through four collaborative agents: (1) \textbf{Reconstructor} builds 3D-consistent human-scene representations from monocular video using 3D Gaussian Splatting and deformation decomposition. (2) \textbf{Critique Agent} enhances reconstruction fidelity by identifying and refining poor regions via multi-round MLLM-based reflection. (3) \textbf{Pose Guider} enables motion generalization by generating expressive pose sequences using time-aware parametric encoders. (4) \textbf{Video Harmonizer} synthesizes photorealistic, coherent video via a hybrid rendering pipeline with diffusion, refining the Reconstructor through a Back-to-4D feedback loop. HumanGenesis achieves state-of-the-art performance on tasks including text-guided synthesis, video reenactment, and novel-pose generalization, significantly improving expressiveness, geometric fidelity, and scene integration.
>
---
#### [new 090] Semantic-aware DropSplat: Adaptive Pruning of Redundant Gaussians for 3D Aerial-View Segmentation
- **分类: cs.CV**

- **简介: 论文提出SAD-Splat方法，解决3D空中视图语义分割中尺度变化与结构遮挡导致的语义模糊问题，通过可学习稀疏机制与伪标签生成，消除冗余点，提升分割精度与紧凑性。**

- **链接: [http://arxiv.org/pdf/2508.09626v1](http://arxiv.org/pdf/2508.09626v1)**

> **作者:** Xu Tang; Junan Jia; Yijing Wang; Jingjing Ma; Xiangrong Zhang
>
> **备注:** 9 pages, 4 figures, AAAI 2026
>
> **摘要:** In the task of 3D Aerial-view Scene Semantic Segmentation (3D-AVS-SS), traditional methods struggle to address semantic ambiguity caused by scale variations and structural occlusions in aerial images. This limits their segmentation accuracy and consistency. To tackle these challenges, we propose a novel 3D-AVS-SS approach named SAD-Splat. Our method introduces a Gaussian point drop module, which integrates semantic confidence estimation with a learnable sparsity mechanism based on the Hard Concrete distribution. This module effectively eliminates redundant and semantically ambiguous Gaussian points, enhancing both segmentation performance and representation compactness. Furthermore, SAD-Splat incorporates a high-confidence pseudo-label generation pipeline. It leverages 2D foundation models to enhance supervision when ground-truth labels are limited, thereby further improving segmentation accuracy. To advance research in this domain, we introduce a challenging benchmark dataset: 3D Aerial Semantic (3D-AS), which encompasses diverse real-world aerial scenes with sparse annotations. Experimental results demonstrate that SAD-Splat achieves an excellent balance between segmentation accuracy and representation compactness. It offers an efficient and scalable solution for 3D aerial scene understanding.
>
---
#### [new 091] Reverse Convolution and Its Applications to Image Restoration
- **分类: cs.CV**

- **简介: 本文提出深度wise反向卷积操作，用于图像恢复，解决转置卷积无法逆向的问题，构建Transformer结构应用于去噪、超分辨率和去模糊。**

- **链接: [http://arxiv.org/pdf/2508.09824v1](http://arxiv.org/pdf/2508.09824v1)**

> **作者:** Xuhong Huang; Shiqi Liu; Kai Zhang; Ying Tai; Jian Yang; Hui Zeng; Lei Zhang
>
> **备注:** ICCV 2025; https://github.com/cszn/ConverseNet
>
> **摘要:** Convolution and transposed convolution are fundamental operators widely used in neural networks. However, transposed convolution (a.k.a. deconvolution) does not serve as a true inverse of convolution due to inherent differences in their mathematical formulations. To date, no reverse convolution operator has been established as a standard component in neural architectures. In this paper, we propose a novel depthwise reverse convolution operator as an initial attempt to effectively reverse depthwise convolution by formulating and solving a regularized least-squares optimization problem. We thoroughly investigate its kernel initialization, padding strategies, and other critical aspects to ensure its effective implementation. Building upon this operator, we further construct a reverse convolution block by combining it with layer normalization, 1$\times$1 convolution, and GELU activation, forming a Transformer-like structure. The proposed operator and block can directly replace conventional convolution and transposed convolution layers in existing architectures, leading to the development of ConverseNet. Corresponding to typical image restoration models such as DnCNN, SRResNet and USRNet, we train three variants of ConverseNet for Gaussian denoising, super-resolution and deblurring, respectively. Extensive experiments demonstrate the effectiveness of the proposed reverse convolution operator as a basic building module. We hope this work could pave the way for developing new operators in deep model design and applications.
>
---
#### [new 092] CitySeg: A 3D Open Vocabulary Semantic Segmentation Foundation Model in City-scale Scenarios
- **分类: cs.CV**

- **简介: 论文提出CitySeg，针对城市点云语义分割中数据规模与领域差距导致的泛化瓶颈，引入文本模态实现开放词汇分割与零样本推理，通过局部全局交叉注意力网络、层次分类策略及两阶段训练提升子类特征分离性，取得SOTA性能并首次实现无视觉信息的零样本泛化。**

- **链接: [http://arxiv.org/pdf/2508.09470v1](http://arxiv.org/pdf/2508.09470v1)**

> **作者:** Jialei Xu; Zizhuang Wei; Weikang You; Linyun Li; Weijian Sun
>
> **摘要:** Semantic segmentation of city-scale point clouds is a critical technology for Unmanned Aerial Vehicle (UAV) perception systems, enabling the classification of 3D points without relying on any visual information to achieve comprehensive 3D understanding. However, existing models are frequently constrained by the limited scale of 3D data and the domain gap between datasets, which lead to reduced generalization capability. To address these challenges, we propose CitySeg, a foundation model for city-scale point cloud semantic segmentation that incorporates text modality to achieve open vocabulary segmentation and zero-shot inference. Specifically, in order to mitigate the issue of non-uniform data distribution across multiple domains, we customize the data preprocessing rules, and propose a local-global cross-attention network to enhance the perception capabilities of point networks in UAV scenarios. To resolve semantic label discrepancies across datasets, we introduce a hierarchical classification strategy. A hierarchical graph established according to the data annotation rules consolidates the data labels, and the graph encoder is used to model the hierarchical relationships between categories. In addition, we propose a two-stage training strategy and employ hinge loss to increase the feature separability of subcategories. Experimental results demonstrate that the proposed CitySeg achieves state-of-the-art (SOTA) performance on nine closed-set benchmarks, significantly outperforming existing approaches. Moreover, for the first time, CitySeg enables zero-shot generalization in city-scale point cloud scenarios without relying on visual information.
>
---
#### [new 093] Gen-AFFECT: Generation of Avatar Fine-grained Facial Expressions with Consistent identiTy
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GEN-AFFECT框架，用于生成具有身份一致性与细粒度面部表情的个性化3D角色。核心问题在于传统方法难以兼顾身份保持与表情多样性，该方法通过多模态扩散模型与一致注意力机制实现身份保真与表情多样性的平衡。**

- **链接: [http://arxiv.org/pdf/2508.09461v1](http://arxiv.org/pdf/2508.09461v1)**

> **作者:** Hao Yu; Rupayan Mallick; Margrit Betke; Sarah Adel Bargal
>
> **摘要:** Different forms of customized 2D avatars are widely used in gaming applications, virtual communication, education, and content creation. However, existing approaches often fail to capture fine-grained facial expressions and struggle to preserve identity across different expressions. We propose GEN-AFFECT, a novel framework for personalized avatar generation that generates expressive and identity-consistent avatars with a diverse set of facial expressions. Our framework proposes conditioning a multimodal diffusion transformer on an extracted identity-expression representation. This enables identity preservation and representation of a wide range of facial expressions. GEN-AFFECT additionally employs consistent attention at inference for information sharing across the set of generated expressions, enabling the generation process to maintain identity consistency over the array of generated fine-grained expressions. GEN-AFFECT demonstrates superior performance compared to previous state-of-the-art methods on the basis of the accuracy of the generated expressions, the preservation of the identity and the consistency of the target identity across an array of fine-grained facial expressions.
>
---
#### [new 094] Iterative Volume Fusion for Asymmetric Stereo Matching
- **分类: cs.CV**

- **简介: 本文提出用于不对称立体匹配的迭代体积融合网络IVF-AStereo，解决传统方法因对称假设失效的问题，通过两阶段融合优化成本体积，提升细粒度特征提取能力，适用于异构多摄像头系统。**

- **链接: [http://arxiv.org/pdf/2508.09543v1](http://arxiv.org/pdf/2508.09543v1)**

> **作者:** Yuanting Gao; Linghao Shen
>
> **摘要:** Stereo matching is vital in 3D computer vision, with most algorithms assuming symmetric visual properties between binocular visions. However, the rise of asymmetric multi-camera systems (e.g., tele-wide cameras) challenges this assumption and complicates stereo matching. Visual asymmetry disrupts stereo matching by affecting the crucial cost volume computation. To address this, we explore the matching cost distribution of two established cost volume construction methods in asymmetric stereo. We find that each cost volume experiences distinct information distortion, indicating that both should be comprehensively utilized to solve the issue. Based on this, we propose the two-phase Iterative Volume Fusion network for Asymmetric Stereo matching (IVF-AStereo). Initially, the aggregated concatenation volume refines the correlation volume. Subsequently, both volumes are fused to enhance fine details. Our method excels in asymmetric scenarios and shows robust performance against significant visual asymmetry. Extensive comparative experiments on benchmark datasets, along with ablation studies, confirm the effectiveness of our approach in asymmetric stereo with resolution and color degradation.
>
---
#### [new 095] AST-n: A Fast Sampling Approach for Low-Dose CT Reconstruction using Diffusion Models
- **分类: cs.CV**

- **简介: 论文提出AST-n框架，通过高阶ODE求解器加速低剂量CT重建，减少采样步骤并保持高PSNR/SSIM，显著缩短推理时间，推动扩散模型在临床中的应用。**

- **链接: [http://arxiv.org/pdf/2508.09943v1](http://arxiv.org/pdf/2508.09943v1)**

> **作者:** Tomás de la Sotta; José M. Saavedra; Héctor Henríquez; Violeta Chang; Aline Xavier
>
> **摘要:** Low-dose CT (LDCT) protocols reduce radiation exposure but increase image noise, compromising diagnostic confidence. Diffusion-based generative models have shown promise for LDCT denoising by learning image priors and performing iterative refinement. In this work, we introduce AST-n, an accelerated inference framework that initiates reverse diffusion from intermediate noise levels, and integrate high-order ODE solvers within conditioned models to further reduce sampling steps. We evaluate two acceleration paradigms--AST-n sampling and standard scheduling with high-order solvers -- on the Low Dose CT Grand Challenge dataset, covering head, abdominal, and chest scans at 10-25 % of standard dose. Conditioned models using only 25 steps (AST-25) achieve peak signal-to-noise ratio (PSNR) above 38 dB and structural similarity index (SSIM) above 0.95, closely matching standard baselines while cutting inference time from ~16 seg to under 1 seg per slice. Unconditional sampling suffers substantial quality loss, underscoring the necessity of conditioning. We also assess DDIM inversion, which yields marginal PSNR gains at the cost of doubling inference time, limiting its clinical practicality. Our results demonstrate that AST-n with high-order samplers enables rapid LDCT reconstruction without significant loss of image fidelity, advancing the feasibility of diffusion-based methods in clinical workflows.
>
---
#### [new 096] MOC: Meta-Optimized Classifier for Few-Shot Whole Slide Image Classification
- **分类: cs.CV**

- **简介: 论文提出一种Meta-Optimized Classifier (MOC)用于Few-Shot WSI分类，通过元学习优化分类器配置并整合分类器银行实现病理解释，显著提升数据稀缺场景下的分类性能。**

- **链接: [http://arxiv.org/pdf/2508.09967v1](http://arxiv.org/pdf/2508.09967v1)**

> **作者:** Tianqi Xiang; Yi Li; Qixiang Zhang; Xiaomeng Li
>
> **备注:** Accepted in MICCAI 2025
>
> **摘要:** Recent advances in histopathology vision-language foundation models (VLFMs) have shown promise in addressing data scarcity for whole slide image (WSI) classification via zero-shot adaptation. However, these methods remain outperformed by conventional multiple instance learning (MIL) approaches trained on large datasets, motivating recent efforts to enhance VLFM-based WSI classification through fewshot learning paradigms. While existing few-shot methods improve diagnostic accuracy with limited annotations, their reliance on conventional classifier designs introduces critical vulnerabilities to data scarcity. To address this problem, we propose a Meta-Optimized Classifier (MOC) comprising two core components: (1) a meta-learner that automatically optimizes a classifier configuration from a mixture of candidate classifiers and (2) a classifier bank housing diverse candidate classifiers to enable a holistic pathological interpretation. Extensive experiments demonstrate that MOC outperforms prior arts in multiple few-shot benchmarks. Notably, on the TCGA-NSCLC benchmark, MOC improves AUC by 10.4% over the state-of-the-art few-shot VLFM-based methods, with gains up to 26.25% under 1-shot conditions, offering a critical advancement for clinical deployments where diagnostic training data is severely limited. Code is available at https://github.com/xmed-lab/MOC.
>
---
#### [new 097] Plane Detection and Ranking via Model Information Optimization
- **分类: cs.CV; cs.RO**

- **简介: 本文提出基于模型信息优化的平面检测与排名框架，解决RANSAC假阳性问题，通过生成多模型并计算信息，选择最优，同时利用神经网络加速，提升平面参数准确性。**

- **链接: [http://arxiv.org/pdf/2508.09625v1](http://arxiv.org/pdf/2508.09625v1)**

> **作者:** Daoxin Zhong; Jun Li; Meng Yee Michael Chuah
>
> **备注:** Accepted as contributed paper in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Plane detection from depth images is a crucial subtask with broad robotic applications, often accomplished by iterative methods such as Random Sample Consensus (RANSAC). While RANSAC is a robust strategy with strong probabilistic guarantees, the ambiguity of its inlier threshold criterion makes it susceptible to false positive plane detections. This issue is particularly prevalent in complex real-world scenes, where the true number of planes is unknown and multiple planes coexist. In this paper, we aim to address this limitation by proposing a generalised framework for plane detection based on model information optimization. Building on previous works, we treat the observed depth readings as discrete random variables, with their probability distributions constrained by the ground truth planes. Various models containing different candidate plane constraints are then generated through repeated random sub-sampling to explain our observations. By incorporating the physics and noise model of the depth sensor, we can calculate the information for each model, and the model with the least information is accepted as the most likely ground truth. This information optimization process serves as an objective mechanism for determining the true number of planes and preventing false positive detections. Additionally, the quality of each detected plane can be ranked by summing the information reduction of inlier points for each plane. We validate these properties through experiments with synthetic data and find that our algorithm estimates plane parameters more accurately compared to the default Open3D RANSAC plane segmentation. Furthermore, we accelerate our algorithm by partitioning the depth map using neural network segmentation, which enhances its ability to generate more realistic plane parameters in real-world data.
>
---
#### [new 098] Harnessing Input-Adaptive Inference for Efficient VLN
- **分类: cs.CV; cs.LG**

- **简介: 论文提出输入自适应推理方法，优化视觉-语言导航（VLN）模型效率，通过空间、模型内、时间三层优化降低计算量，实验显示计算量减少2倍以上。**

- **链接: [http://arxiv.org/pdf/2508.09262v1](http://arxiv.org/pdf/2508.09262v1)**

> **作者:** Dongwoo Kang; Akhil Perincherry; Zachary Coalson; Aiden Gabriel; Stefan Lee; Sanghyun Hong
>
> **备注:** Accepted to ICCV 2025 [Poster]
>
> **摘要:** An emerging paradigm in vision-and-language navigation (VLN) is the use of history-aware multi-modal transformer models. Given a language instruction, these models process observation and navigation history to predict the most appropriate action for an agent. While they have significantly improved performance, the scale of these models can be a bottleneck in practical settings with limited computational resources. In this work, we propose a novel input-adaptive navigation method to enhance VLN model efficiency. We first show that existing input-adaptive mechanisms fail to reduce computations without substantial performance degradation. To address this, we introduce three adaptive algorithms, each deployed at a different level: (1) To improve spatial efficiency, we selectively process panoramic views at each observation of an agent. (2) To improve intra-model efficiency, we propose importance-based adaptive thresholding for the early-exit methods. (3) To improve temporal efficiency, we implement a caching mechanism that prevents reprocessing of views previously seen by the agent. In evaluations on seven VLN benchmarks, we demonstrate over a 2$\times$ reduction in computation across three off-the-shelf agents in both standard and continuous environments. Our code is publicly available at https://github.com/secure-ai-systems-group/adaptive-vision-and-language-navigation.
>
---
#### [new 099] IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出IAD-R1框架，针对工业异常检测中样本稀缺与VLMs性能不足的问题，通过两阶段训练增强感知与推理能力，实现零样本超越商业模型，提升7个VLMs在6个工业基准上的准确率43.3%。**

- **链接: [http://arxiv.org/pdf/2508.09178v1](http://arxiv.org/pdf/2508.09178v1)**

> **作者:** Yanhui Li; Yunkang Cao; Chengliang Liu; Yuan Xiong; Xinghui Dong; Chao Huang
>
> **摘要:** Industrial anomaly detection is a critical component of modern manufacturing, yet the scarcity of defective samples restricts traditional detection methods to scenario-specific applications. Although Vision-Language Models (VLMs) demonstrate significant advantages in generalization capabilities, their performance in industrial anomaly detection remains limited. To address this challenge, we propose IAD-R1, a universal post-training framework applicable to VLMs of different architectures and parameter scales, which substantially enhances their anomaly detection capabilities. IAD-R1 employs a two-stage training strategy: the Perception Activation Supervised Fine-Tuning (PA-SFT) stage utilizes a meticulously constructed high-quality Chain-of-Thought dataset (Expert-AD) for training, enhancing anomaly perception capabilities and establishing reasoning-to-answer correlations; the Structured Control Group Relative Policy Optimization (SC-GRPO) stage employs carefully designed reward functions to achieve a capability leap from "Anomaly Perception" to "Anomaly Interpretation". Experimental results demonstrate that IAD-R1 achieves significant improvements across 7 VLMs, attaining up to 43.3% enhancement in average accuracy on 6 industrial anomaly detection benchmark datasets. Notably, the 0.5B parameter model trained with IAD-R1 surpasses commercial models including GPT-4.1 and Claude-Sonnet-4 in zero-shot settings, demonstrating the effectiveness and superiority of IAD-R1. The dataset, code, and all model weights will be publicly available at https://github.com/Yanhui-Lee/IAD-R1.
>
---
#### [new 100] RASR: Retrieval-Augmented Super Resolution for Practical Reference-based Image Restoration
- **分类: cs.CV**

- **简介: 论文提出RASR，通过自动检索参考数据库中的高分辨率图像实现参考增强超分辨率，解决传统方法依赖手动配对的局限性，构建RASR-Flickr30数据集，并提出RASRNet结合语义检索与扩散生成器，提升图像修复性能。**

- **链接: [http://arxiv.org/pdf/2508.09449v1](http://arxiv.org/pdf/2508.09449v1)**

> **作者:** Jiaqi Yan; Shuning Xu; Xiangyu Chen; Dell Zhang; Jie Tang; Gangshan Wu; Jie Liu
>
> **摘要:** Reference-based Super Resolution (RefSR) improves upon Single Image Super Resolution (SISR) by leveraging high-quality reference images to enhance texture fidelity and visual realism. However, a critical limitation of existing RefSR approaches is their reliance on manually curated target-reference image pairs, which severely constrains their practicality in real-world scenarios. To overcome this, we introduce Retrieval-Augmented Super Resolution (RASR), a new and practical RefSR paradigm that automatically retrieves semantically relevant high-resolution images from a reference database given only a low-quality input. This enables scalable and flexible RefSR in realistic use cases, such as enhancing mobile photos taken in environments like zoos or museums, where category-specific reference data (e.g., animals, artworks) can be readily collected or pre-curated. To facilitate research in this direction, we construct RASR-Flickr30, the first benchmark dataset designed for RASR. Unlike prior datasets with fixed target-reference pairs, RASR-Flickr30 provides per-category reference databases to support open-world retrieval. We further propose RASRNet, a strong baseline that combines a semantic reference retriever with a diffusion-based RefSR generator. It retrieves relevant references based on semantic similarity and employs a diffusion-based generator enhanced with semantic conditioning. Experiments on RASR-Flickr30 demonstrate that RASRNet consistently improves over SISR baselines, achieving +0.38 dB PSNR and -0.0131 LPIPS, while generating more realistic textures. These findings highlight retrieval augmentation as a promising direction to bridge the gap between academic RefSR research and real-world applicability.
>
---
#### [new 101] Poaching Hotspot Identification Using Satellite Imagery
- **分类: cs.CV**

- **简介: 论文提出基于卫星图像的计算机视觉模型，用于自动识别偷猎热点，解决传统方法无法动态追踪问题，优化资源分配。**

- **链接: [http://arxiv.org/pdf/2508.09812v1](http://arxiv.org/pdf/2508.09812v1)**

> **作者:** Aryan Pandhi; Shrey Baid; Sanjali Jha
>
> **摘要:** Elephant Poaching in African countries has been a decade-old problem. So much so that African Forest Elephants are now listed as an endangered species, and African Savannah Elephants as critically endangered by the IUCN (International Union for Conservation of Nature). [1] Elephants are hunted primarily for their ivory tusks which caused many elephants to be born tuskless as a genetic modification for survival. [2] Data gathered by recent studies shows that though poaching methods remain the same, the poaching grounds are rather dynamic. Poachers have shifted to areas with less ranger patrols and several other factors like watering holes, seasons, altitude etc. cause constant shifts in poaching hotspot locations. [3] After a period of low poaching from 2000-2014, poaching numbers in African countries are now on the rise again -- WWF (World Wildlife Foundation) says there are 20,000 elephants poached annually [4]. In African countries, anti-poaching efforts are concentrated near towns, while a majority of poaching occurs in the deserted regions. All of these factors result in the need for a Computer Vision Model to identify poaching hotspots through locating the geographic indicators of favorable poaching regions. A CV model eliminates the need to manually track poachers and account for the environmental factors to deploy resources and its combination with satellite imagery allows us to survey large areas without disturbing local species or cross border aviation restrictions.
>
---
#### [new 102] Preacher: Paper-to-Video Agentic System
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Preacher系统，将论文转化为视频摘要，解决现有模型受限于上下文、时长、风格和领域知识的问题，通过自上而下分解与自下而上生成结合P-CoT迭代规划，实现高质量跨模态视频摘要。**

- **链接: [http://arxiv.org/pdf/2508.09632v1](http://arxiv.org/pdf/2508.09632v1)**

> **作者:** Jingwei Liu; Ling Yang; Hao Luo; Fan Wang Hongyan Li; Mengdi Wang
>
> **摘要:** The paper-to-video task converts a research paper into a structured video abstract, distilling key concepts, methods, and conclusions into an accessible, well-organized format. While state-of-the-art video generation models demonstrate potential, they are constrained by limited context windows, rigid video duration constraints, limited stylistic diversity, and an inability to represent domain-specific knowledge. To address these limitations, we introduce Preacher, the first paper-to-video agentic system. Preacher employs a top-down approach to decompose, summarize, and reformulate the paper, followed by bottom-up video generation, synthesizing diverse video segments into a coherent abstract. To align cross-modal representations, we define key scenes and introduce a Progressive Chain of Thought (P-CoT) for granular, iterative planning. Preacher successfully generates high-quality video abstracts across five research fields, demonstrating expertise beyond current video generation models. Code will be released at: https://github.com/GenVerse/Paper2Video
>
---
#### [new 103] Enhancing Diffusion Face Generation with Contrastive Embeddings and SegFormer Guidance
- **分类: cs.CV**

- **简介: 该论文提出一种基于对比嵌入和SegFormer的扩散模型，用于小规模人脸生成，解决受限数据下可控性与语义对齐问题，通过融合属性嵌入与分割编码器提升生成效果。**

- **链接: [http://arxiv.org/pdf/2508.09847v1](http://arxiv.org/pdf/2508.09847v1)**

> **作者:** Dhruvraj Singh Rawat; Enggen Sherpa; Rishikesan Kirupanantha; Tin Hoang
>
> **备注:** 10 pages, preprint
>
> **摘要:** We present a benchmark of diffusion models for human face generation on a small-scale CelebAMask-HQ dataset, evaluating both unconditional and conditional pipelines. Our study compares UNet and DiT architectures for unconditional generation and explores LoRA-based fine-tuning of pretrained Stable Diffusion models as a separate experiment. Building on the multi-conditioning approach of Giambi and Lisanti, which uses both attribute vectors and segmentation masks, our main contribution is the integration of an InfoNCE loss for attribute embedding and the adoption of a SegFormer-based segmentation encoder. These enhancements improve the semantic alignment and controllability of attribute-guided synthesis. Our results highlight the effectiveness of contrastive embedding learning and advanced segmentation encoding for controlled face generation in limited data settings.
>
---
#### [new 104] LIA-X: Interpretable Latent Portrait Animator
- **分类: cs.CV**

- **简介: 论文提出LIA-X模型，通过稀疏运动字典实现可解释的面部动态迁移，解决传统方法难以精确控制的问题，支持细粒度编辑并提升跨任务性能，适用于用户引导的图像/视频编辑及3D肖像处理。**

- **链接: [http://arxiv.org/pdf/2508.09959v1](http://arxiv.org/pdf/2508.09959v1)**

> **作者:** Yaohui Wang; Di Yang; Xinyuan Chen; Francois Bremond; Yu Qiao; Antitza Dantcheva
>
> **备注:** Project Page: https://wyhsirius.github.io/LIA-X-project/
>
> **摘要:** We introduce LIA-X, a novel interpretable portrait animator designed to transfer facial dynamics from a driving video to a source portrait with fine-grained control. LIA-X is an autoencoder that models motion transfer as a linear navigation of motion codes in latent space. Crucially, it incorporates a novel Sparse Motion Dictionary that enables the model to disentangle facial dynamics into interpretable factors. Deviating from previous 'warp-render' approaches, the interpretability of the Sparse Motion Dictionary allows LIA-X to support a highly controllable 'edit-warp-render' strategy, enabling precise manipulation of fine-grained facial semantics in the source portrait. This helps to narrow initial differences with the driving video in terms of pose and expression. Moreover, we demonstrate the scalability of LIA-X by successfully training a large-scale model with approximately 1 billion parameters on extensive datasets. Experimental results show that our proposed method outperforms previous approaches in both self-reenactment and cross-reenactment tasks across several benchmarks. Additionally, the interpretable and controllable nature of LIA-X supports practical applications such as fine-grained, user-guided image and video editing, as well as 3D-aware portrait video manipulation.
>
---
#### [new 105] DenoDet V2: Phase-Amplitude Cross Denoising for SAR Object Detection
- **分类: cs.CV**

- **简介: 论文提出一种基于相位-幅度交叉调制的SAR目标检测方法DenoDet V2，通过变换域特征解耦与联合增强解决相干噪声问题，显著提升检测精度并降低模型复杂度。**

- **链接: [http://arxiv.org/pdf/2508.09392v1](http://arxiv.org/pdf/2508.09392v1)**

> **作者:** Kang Ni; Minrui Zou; Yuxuan Li; Xiang Li; Kehua Guo; Ming-Ming Cheng; Yimian Dai
>
> **摘要:** One of the primary challenges in Synthetic Aperture Radar (SAR) object detection lies in the pervasive influence of coherent noise. As a common practice, most existing methods, whether handcrafted approaches or deep learning-based methods, employ the analysis or enhancement of object spatial-domain characteristics to achieve implicit denoising. In this paper, we propose DenoDet V2, which explores a completely novel and different perspective to deconstruct and modulate the features in the transform domain via a carefully designed attention architecture. Compared to DenoDet V1, DenoDet V2 is a major advancement that exploits the complementary nature of amplitude and phase information through a band-wise mutual modulation mechanism, which enables a reciprocal enhancement between phase and amplitude spectra. Extensive experiments on various SAR datasets demonstrate the state-of-the-art performance of DenoDet V2. Notably, DenoDet V2 achieves a significant 0.8\% improvement on SARDet-100K dataset compared to DenoDet V1, while reducing the model complexity by half. The code is available at https://github.com/GrokCV/GrokSAR.
>
---
#### [new 106] BridgeTA: Bridging the Representation Gap in Knowledge Distillation via Teacher Assistant for Bird's Eye View Map Segmentation
- **分类: cs.CV**

- **简介: 论文提出BridgeTA框架，针对鸟瞰图分割任务，解决相机-only与LC融合方法的性能差距问题，通过教师助手网络实现低成本知识蒸馏，提升4.2% mIoU。**

- **链接: [http://arxiv.org/pdf/2508.09599v1](http://arxiv.org/pdf/2508.09599v1)**

> **作者:** Beomjun Kim; Suhan Woo; Sejong Heo; Euntai Kim
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Bird's-Eye-View (BEV) map segmentation is one of the most important and challenging tasks in autonomous driving. Camera-only approaches have drawn attention as cost-effective alternatives to LiDAR, but they still fall behind LiDAR-Camera (LC) fusion-based methods. Knowledge Distillation (KD) has been explored to narrow this gap, but existing methods mainly enlarge the student model by mimicking the teacher's architecture, leading to higher inference cost. To address this issue, we introduce BridgeTA, a cost-effective distillation framework to bridge the representation gap between LC fusion and Camera-only models through a Teacher Assistant (TA) network while keeping the student's architecture and inference cost unchanged. A lightweight TA network combines the BEV representations of the teacher and student, creating a shared latent space that serves as an intermediate representation. To ground the framework theoretically, we derive a distillation loss using Young's Inequality, which decomposes the direct teacher-student distillation path into teacher-TA and TA-student dual paths, stabilizing optimization and strengthening knowledge transfer. Extensive experiments on the challenging nuScenes dataset demonstrate the effectiveness of our method, achieving an improvement of 4.2% mIoU over the Camera-only baseline, up to 45% higher than the improvement of other state-of-the-art KD methods.
>
---
#### [new 107] GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors
- **分类: cs.CV**

- **简介: 论文提出GSFixer，通过参考引导的视频扩散模型改进3DGS，解决稀疏输入下伪影问题，结合2D语义与3D几何特征提升重建质量。**

- **链接: [http://arxiv.org/pdf/2508.09667v1](http://arxiv.org/pdf/2508.09667v1)**

> **作者:** Xingyilang Yin; Qi Zhang; Jiahao Chang; Ying Feng; Qingnan Fan; Xi Yang; Chi-Man Pun; Huaqi Zhang; Xiaodong Cun
>
> **摘要:** Reconstructing 3D scenes using 3D Gaussian Splatting (3DGS) from sparse views is an ill-posed problem due to insufficient information, often resulting in noticeable artifacts. While recent approaches have sought to leverage generative priors to complete information for under-constrained regions, they struggle to generate content that remains consistent with input observations. To address this challenge, we propose GSFixer, a novel framework designed to improve the quality of 3DGS representations reconstructed from sparse inputs. The core of our approach is the reference-guided video restoration model, built upon a DiT-based video diffusion model trained on paired artifact 3DGS renders and clean frames with additional reference-based conditions. Considering the input sparse views as references, our model integrates both 2D semantic features and 3D geometric features of reference views extracted from the visual geometry foundation model, enhancing the semantic coherence and 3D consistency when fixing artifact novel views. Furthermore, considering the lack of suitable benchmarks for 3DGS artifact restoration evaluation, we present DL3DV-Res which contains artifact frames rendered using low-quality 3DGS. Extensive experiments demonstrate our GSFixer outperforms current state-of-the-art methods in 3DGS artifact restoration and sparse-view 3D reconstruction. Project page: https://github.com/GVCLab/GSFixer.
>
---
#### [new 108] IAG: Input-aware Backdoor Attack on VLMs for Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 论文提出输入感知的反向攻击方法IAG，针对VLMs在视觉接地任务中的安全问题，通过文本条件U-Net嵌入目标语义信息，利用重构损失降低攻击痕迹，实现对模型的隐蔽操控，实验显示其有效性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.09456v1](http://arxiv.org/pdf/2508.09456v1)**

> **作者:** Junxian Li; Beining Xu; Di Zhang
>
> **备注:** 13 pages, 13 Figures
>
> **摘要:** Vision-language models (VLMs) have shown significant advancements in tasks such as visual grounding, where they localize specific objects in images based on natural language queries and images. However, security issues in visual grounding tasks for VLMs remain underexplored, especially in the context of backdoor attacks. In this paper, we introduce a novel input-aware backdoor attack method, IAG, designed to manipulate the grounding behavior of VLMs. This attack forces the model to ground a specific target object in the input image, regardless of the user's query. We propose an adaptive trigger generator that embeds the semantic information of the attack target's description into the original image using a text-conditional U-Net, thereby overcoming the open-vocabulary attack challenge. To ensure the attack's stealthiness, we utilize a reconstruction loss to minimize visual discrepancies between poisoned and clean images. Additionally, we introduce a unified method for generating attack data. IAG is evaluated theoretically and empirically, demonstrating its feasibility and effectiveness. Notably, our ASR@0.5 on InternVL-2.5-8B reaches over 65\% on various testing sets. IAG also shows promising potential on manipulating Ferret-7B and LlaVA-1.5-7B with very little accuracy decrease on clean samples. Extensive specific experiments, such as ablation study and potential defense, also indicate the robustness and transferability of our attack.
>
---
#### [new 109] A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous Sign Language Recognition
- **分类: cs.CV; cs.AI; cs.IR; cs.LG**

- **简介: 论文提出双架构模型，解决连续手语识别中的签名人差异和新句法结构问题，通过Signer-Invariant Conformer和Multi-Scale Fusion Transformer提升性能，实验显示显著降低错误率并获挑战赛佳绩。**

- **链接: [http://arxiv.org/pdf/2508.09372v1](http://arxiv.org/pdf/2508.09372v1)**

> **作者:** Md Rezwanul Haque; Md. Milon Islam; S M Taslim Uddin Raju; Fakhri Karray
>
> **备注:** Accepted for the IEEE/CVF International Conference on Computer Vision (ICCV), Honolulu, Hawaii, USA. 1st MSLR Workshop 2025
>
> **摘要:** Continuous Sign Language Recognition (CSLR) faces multiple challenges, including significant inter-signer variability and poor generalization to novel sentence structures. Traditional solutions frequently fail to handle these issues efficiently. For overcoming these constraints, we propose a dual-architecture framework. For the Signer-Independent (SI) challenge, we propose a Signer-Invariant Conformer that combines convolutions with multi-head self-attention to learn robust, signer-agnostic representations from pose-based skeletal keypoints. For the Unseen-Sentences (US) task, we designed a Multi-Scale Fusion Transformer with a novel dual-path temporal encoder that captures both fine-grained posture dynamics, enabling the model's ability to comprehend novel grammatical compositions. Experiments on the challenging Isharah-1000 dataset establish a new standard for both CSLR benchmarks. The proposed conformer architecture achieves a Word Error Rate (WER) of 13.07% on the SI challenge, a reduction of 13.53% from the state-of-the-art. On the US task, the transformer model scores a WER of 47.78%, surpassing previous work. In the SignEval 2025 CSLR challenge, our team placed 2nd in the US task and 4th in the SI task, demonstrating the performance of these models. The findings validate our key hypothesis: that developing task-specific networks designed for the particular challenges of CSLR leads to considerable performance improvements and establishes a new baseline for further research. The source code is available at: https://github.com/rezwanh001/MSLR-Pose86K-CSLR-Isharah.
>
---
#### [new 110] X-UniMotion: Animating Human Images with Expressive, Unified and Identity-Agnostic Motion Latents
- **分类: cs.CV; cs.AI**

- **简介: 论文提出X-UniMotion，通过统一的隐式潜在表示生成高保真人体动画，解决跨身份动作迁移问题，采用解耦的四维潜在token和自监督框架，提升身份无关性和运动精度。**

- **链接: [http://arxiv.org/pdf/2508.09383v1](http://arxiv.org/pdf/2508.09383v1)**

> **作者:** Guoxian Song; Hongyi Xu; Xiaochen Zhao; You Xie; Tianpei Gu; Zenan Li; Chenxu Zhang; Linjie Luo
>
> **摘要:** We present X-UniMotion, a unified and expressive implicit latent representation for whole-body human motion, encompassing facial expressions, body poses, and hand gestures. Unlike prior motion transfer methods that rely on explicit skeletal poses and heuristic cross-identity adjustments, our approach encodes multi-granular motion directly from a single image into a compact set of four disentangled latent tokens -- one for facial expression, one for body pose, and one for each hand. These motion latents are both highly expressive and identity-agnostic, enabling high-fidelity, detailed cross-identity motion transfer across subjects with diverse identities, poses, and spatial configurations. To achieve this, we introduce a self-supervised, end-to-end framework that jointly learns the motion encoder and latent representation alongside a DiT-based video generative model, trained on large-scale, diverse human motion datasets. Motion-identity disentanglement is enforced via 2D spatial and color augmentations, as well as synthetic 3D renderings of cross-identity subject pairs under shared poses. Furthermore, we guide motion token learning with auxiliary decoders that promote fine-grained, semantically aligned, and depth-aware motion embeddings. Extensive experiments show that X-UniMotion outperforms state-of-the-art methods, producing highly expressive animations with superior motion fidelity and identity preservation.
>
---
#### [new 111] SegDAC: Segmentation-Driven Actor-Critic for Visual Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出SegDAC，针对视觉强化学习中高维输入与噪声奖励的挑战，通过SAM分解与YOLO-World语义接地，结合动态Transformer架构实现在线RL学习，提升视觉泛化与样本效率。**

- **链接: [http://arxiv.org/pdf/2508.09325v1](http://arxiv.org/pdf/2508.09325v1)**

> **作者:** Alexandre Brown; Glen Berseth
>
> **摘要:** Visual reinforcement learning (RL) is challenging due to the need to learn both perception and actions from high-dimensional inputs and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains unclear. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground segments semantically via text prompts. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks.
>
---
#### [new 112] KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging
- **分类: cs.CV**

- **简介: 论文提出一个模块化、可配置的深度学习框架KonfAI，用于医学影像分析，通过YAML配置管理训练、推理和评估流程，支持高级策略如patch-based learning和模型集成，提升效率与可重复性。**

- **链接: [http://arxiv.org/pdf/2508.09823v1](http://arxiv.org/pdf/2508.09823v1)**

> **作者:** Valentin Boussot; Jean-Louis Dillenseger
>
> **备注:** https://github.com/vboussot/KonfAI
>
> **摘要:** KonfAI is a modular, extensible, and fully configurable deep learning framework specifically designed for medical imaging tasks. It enables users to define complete training, inference, and evaluation workflows through structured YAML configuration files, without modifying the underlying code. This declarative approach enhances reproducibility, transparency, and experimental traceability while reducing development time. Beyond the capabilities of standard pipelines, KonfAI provides native abstractions for advanced strategies including patch-based learning, test-time augmentation, model ensembling, and direct access to intermediate feature representations for deep supervision. It also supports complex multi-model training setups such as generative adversarial architectures. Thanks to its modular and extensible architecture, KonfAI can easily accommodate custom models, loss functions, and data processing components. The framework has been successfully applied to segmentation, registration, and image synthesis tasks, and has contributed to top-ranking results in several international medical imaging challenges. KonfAI is open source and available at \href{https://github.com/vboussot/KonfAI}{https://github.com/vboussot/KonfAI}.
>
---
#### [new 113] Waymo-3DSkelMo: A Multi-Agent 3D Skeletal Motion Dataset for Pedestrian Interaction Modeling in Autonomous Driving
- **分类: cs.CV; cs.MM**

- **简介: 论文构建多智能体3D骨骼运动数据集，解决传统单目数据遮挡、时序缺失问题，提升行人交互建模精度。**

- **链接: [http://arxiv.org/pdf/2508.09404v1](http://arxiv.org/pdf/2508.09404v1)**

> **作者:** Guangxun Zhu; Shiyu Fan; Hang Dai; Edmond S. L. Ho
>
> **备注:** ACM Multimedia 2025 (Dataset Track) Paper
>
> **摘要:** Large-scale high-quality 3D motion datasets with multi-person interactions are crucial for data-driven models in autonomous driving to achieve fine-grained pedestrian interaction understanding in dynamic urban environments. However, existing datasets mostly rely on estimating 3D poses from monocular RGB video frames, which suffer from occlusion and lack of temporal continuity, thus resulting in unrealistic and low-quality human motion. In this paper, we introduce Waymo-3DSkelMo, the first large-scale dataset providing high-quality, temporally coherent 3D skeletal motions with explicit interaction semantics, derived from the Waymo Perception dataset. Our key insight is to utilize 3D human body shape and motion priors to enhance the quality of the 3D pose sequences extracted from the raw LiDRA point clouds. The dataset covers over 14,000 seconds across more than 800 real driving scenarios, including rich interactions among an average of 27 agents per scene (with up to 250 agents in the largest scene). Furthermore, we establish 3D pose forecasting benchmarks under varying pedestrian densities, and the results demonstrate its value as a foundational resource for future research on fine-grained human behavior understanding in complex urban environments. The dataset and code will be available at https://github.com/GuangxunZhu/Waymo-3DSkelMo
>
---
#### [new 114] Evolution of Low-Level and Texture Human-CLIP Alignment
- **分类: cs.CV**

- **简介: 论文研究CLIP训练中低级感知对齐与纹理偏倚的变化，分析形状-纹理偏倚和噪声影响，揭示特征学习演变及权衡优化。**

- **链接: [http://arxiv.org/pdf/2508.09814v1](http://arxiv.org/pdf/2508.09814v1)**

> **作者:** Pablo Hernández-Cámara; Jose Manuel Jaén-Lorites; Jorge Vila-Tomás; Jesus Malo; Valero Laparra
>
> **摘要:** During the training of multi-modal models like CLIP, we observed an intriguing phenomenon: the correlation with low-level human image quality assessments peaks in the early epochs before gradually declining. This study investigates this observation and seeks to understand its causes through two key factors: shape-texture bias alignment and classification accuracy drop under noise. Our findings suggest that CLIP initially learn low-level visual features, enhancing its alignment with low-level human perception but also increasing its sensitivity to noise and its texture bias. As training progresses, the model shifts toward more abstract shape-based representations, improving noise robustness but reducing alignment with low-level human perception. These results suggest that these factors shared an underlying learning mechanism and provide new insights into optimizing the trade-off between perceptual alignment and robustness in vision-language models.
>
---
#### [new 115] SpeechForensics: Audio-Visual Speech Representation Learning for Face Forgery Detection
- **分类: cs.CV**

- **简介: 论文提出一种基于音频-视觉语音表示的学习方法，解决人脸伪造检测中的跨数据集泛化与鲁棒性问题，通过自监督学习融合音频与视觉信息，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2508.09913v1](http://arxiv.org/pdf/2508.09913v1)**

> **作者:** Yachao Liang; Min Yu; Gang Li; Jianguo Jiang; Boquan Li; Feng Yu; Ning Zhang; Xiang Meng; Weiqing Huang
>
> **备注:** Accepted by NeurIPS 2024
>
> **摘要:** Detection of face forgery videos remains a formidable challenge in the field of digital forensics, especially the generalization to unseen datasets and common perturbations. In this paper, we tackle this issue by leveraging the synergy between audio and visual speech elements, embarking on a novel approach through audio-visual speech representation learning. Our work is motivated by the finding that audio signals, enriched with speech content, can provide precise information effectively reflecting facial movements. To this end, we first learn precise audio-visual speech representations on real videos via a self-supervised masked prediction task, which encodes both local and global semantic information simultaneously. Then, the derived model is directly transferred to the forgery detection task. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods in terms of cross-dataset generalization and robustness, without the participation of any fake video in model training. Code is available at https://github.com/Eleven4AI/SpeechForensics.
>
---
#### [new 116] Multi-Contrast Fusion Module: An attention mechanism integrating multi-contrast features for fetal torso plane classification
- **分类: cs.CV**

- **简介: 论文提出一种多对比度融合模块（MCFM），通过注意力机制整合多模态对比度特征，解决超声胎儿躯干平面分类中对比度低、纹理模糊的问题，提升识别精度与临床可靠性。**

- **链接: [http://arxiv.org/pdf/2508.09644v1](http://arxiv.org/pdf/2508.09644v1)**

> **作者:** Shengjun Zhu; Siyu Liu; Runqing Xiong; Liping Zheng; Duo Ma; Rongshang Chen; Jiaxin Cai
>
> **摘要:** Purpose: Prenatal ultrasound is a key tool in evaluating fetal structural development and detecting abnormalities, contributing to reduced perinatal complications and improved neonatal survival. Accurate identification of standard fetal torso planes is essential for reliable assessment and personalized prenatal care. However, limitations such as low contrast and unclear texture details in ultrasound imaging pose significant challenges for fine-grained anatomical recognition. Methods: We propose a novel Multi-Contrast Fusion Module (MCFM) to enhance the model's ability to extract detailed information from ultrasound images. MCFM operates exclusively on the lower layers of the neural network, directly processing raw ultrasound data. By assigning attention weights to image representations under different contrast conditions, the module enhances feature modeling while explicitly maintaining minimal parameter overhead. Results: The proposed MCFM was evaluated on a curated dataset of fetal torso plane ultrasound images. Experimental results demonstrate that MCFM substantially improves recognition performance, with a minimal increase in model complexity. The integration of multi-contrast attention enables the model to better capture subtle anatomical structures, contributing to higher classification accuracy and clinical reliability. Conclusions: Our method provides an effective solution for improving fetal torso plane recognition in ultrasound imaging. By enhancing feature representation through multi-contrast fusion, the proposed approach supports clinicians in achieving more accurate and consistent diagnoses, demonstrating strong potential for clinical adoption in prenatal screening. The codes are available at https://github.com/sysll/MCFM.
>
---
#### [new 117] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos
- **分类: cs.CV; cs.AI; cs.CE; cs.LG; cs.RO**

- **简介: 论文提出TRACE框架，通过将3D点建模为刚体粒子，直接学习其物理动力学，解决无标注视频中复杂场景运动建模问题，实现未来帧外推。**

- **链接: [http://arxiv.org/pdf/2508.09811v1](http://arxiv.org/pdf/2508.09811v1)**

> **作者:** Jinxi Li; Ziyang Song; Bo Yang
>
> **备注:** ICCV 2025. Code and data are available at: https://github.com/vLAR-group/TRACE
>
> **摘要:** In this paper, we aim to model 3D scene geometry, appearance, and physical information just from dynamic multi-view videos in the absence of any human labels. By leveraging physics-informed losses as soft constraints or integrating simple physics models into neural nets, existing works often fail to learn complex motion physics, or doing so requires additional labels such as object types or masks. We propose a new framework named TRACE to model the motion physics of complex dynamic 3D scenes. The key novelty of our method is that, by formulating each 3D point as a rigid particle with size and orientation in space, we directly learn a translation rotation dynamics system for each particle, explicitly estimating a complete set of physical parameters to govern the particle's motion over time. Extensive experiments on three existing dynamic datasets and one newly created challenging synthetic datasets demonstrate the extraordinary performance of our method over baselines in the task of future frame extrapolation. A nice property of our framework is that multiple objects or parts can be easily segmented just by clustering the learned physical parameters.
>
---
#### [new 118] OneVAE: Joint Discrete and Continuous Optimization Helps Discrete Video VAE Train Better
- **分类: cs.CV**

- **简介: 论文提出OneVAE，通过联合离散与连续优化，利用连续VAE先验提升离散VAE训练稳定性，改进重建质量，提出多token量化和增强第一帧重建，实现单网络双范式竞争性表现。**

- **链接: [http://arxiv.org/pdf/2508.09857v1](http://arxiv.org/pdf/2508.09857v1)**

> **作者:** Yupeng Zhou; Zhen Li; Ziheng Ouyang; Yuming Chen; Ruoyi Du; Daquan Zhou; Bin Fu; Yihao Liu; Peng Gao; Ming-Ming Cheng; Qibin Hou
>
> **摘要:** Encoding videos into discrete tokens could align with text tokens to facilitate concise and unified multi-modal LLMs, yet introducing significant spatiotemporal compression compared to continuous video representation. Previous discrete video VAEs experienced unstable training, long training time, and degraded reconstruction quality. Given the easier training and superior performance of continuous VAEs, an intuitive idea is to enhance discrete video VAEs by leveraging continuous VAEs. After rethinking the intrinsic link between discrete and continuous representations, we found that FSQ could effectively preserve pre-trained continuous VAE priors compared to other quantization methods. By leveraging continuous VAE priors, it converges several times faster than training from scratch and achieves superior performance at convergence. Meanwhile, two structural improvements are proposed. First, inspired by how continuous VAEs enhance reconstruction via enlarged latent dimensions, we introduce a multi-token quantization mechanism, which achieves nearly a 1 dB improvement in PSNR without compromising the token compression ratio. Second, to tackle reconstruction challenges in high-compression video VAEs, we strengthen first-frame reconstruction, enabling the causal VAE to leverage this information in subsequent frames and markedly improving the performance of 4 x 16 x 16 discrete VAEs. Furthermore, we propose a joint discrete-continuous optimization scheme that unifies the two paradigms and, for the first time, achieves competitive performance on both continuous and discrete representations within a single network. We name our method OneVAE to reflect this connection.
>
---
#### [new 119] Images Speak Louder Than Scores: Failure Mode Escape for Enhancing Generative Quality
- **分类: cs.CV**

- **简介: 论文提出FaME方法，通过图像质量评估识别低质生成并引导采样，提升视觉质量而不牺牲FID，解决扩散模型生成质量不足问题。**

- **链接: [http://arxiv.org/pdf/2508.09598v1](http://arxiv.org/pdf/2508.09598v1)**

> **作者:** Jie Shao; Ke Zhu; Minghao Fu; Guo-hua Wang; Jianxin Wu
>
> **摘要:** Diffusion models have achieved remarkable progress in class-to-image generation. However, we observe that despite impressive FID scores, state-of-the-art models often generate distorted or low-quality images, especially in certain classes. This gap arises because FID evaluates global distribution alignment, while ignoring the perceptual quality of individual samples. We further examine the role of CFG, a common technique used to enhance generation quality. While effective in improving metrics and suppressing outliers, CFG can introduce distribution shift and visual artifacts due to its misalignment with both training objectives and user expectations. In this work, we propose FaME, a training-free and inference-efficient method for improving perceptual quality. FaME uses an image quality assessment model to identify low-quality generations and stores their sampling trajectories. These failure modes are then used as negative guidance to steer future sampling away from poor-quality regions. Experiments on ImageNet demonstrate that FaME brings consistent improvements in visual quality without compromising FID. FaME also shows the potential to be extended to improve text-to-image generation.
>
---
#### [new 120] MeMoSORT: Memory-Assisted Filtering and Motion-Adaptive Association Metric for Multi-Person Tracking
- **分类: cs.CV**

- **简介: 论文提出MeMoSORT，解决多目标跟踪中运动建模与遮挡导致的误差问题，通过记忆辅助Kalman滤波和运动适应的IoU改进，实现高效轻量化的跟踪，HOTA成绩达67.9%。**

- **链接: [http://arxiv.org/pdf/2508.09796v1](http://arxiv.org/pdf/2508.09796v1)**

> **作者:** Yingjie Wang; Zhixing Wang; Le Zheng; Tianxiao Liu; Roujing Li; Xueyao Hu
>
> **摘要:** Multi-object tracking (MOT) in human-dominant scenarios, which involves continuously tracking multiple people within video sequences, remains a significant challenge in computer vision due to targets' complex motion and severe occlusions. Conventional tracking-by-detection methods are fundamentally limited by their reliance on Kalman filter (KF) and rigid Intersection over Union (IoU)-based association. The motion model in KF often mismatches real-world object dynamics, causing filtering errors, while rigid association struggles under occlusions, leading to identity switches or target loss. To address these issues, we propose MeMoSORT, a simple, online, and real-time MOT tracker with two key innovations. First, the Memory-assisted Kalman filter (MeKF) uses memory-augmented neural networks to compensate for mismatches between assumed and actual object motion. Second, the Motion-adaptive IoU (Mo-IoU) adaptively expands the matching space and incorporates height similarity to reduce the influence of detection errors and association failures, while remaining lightweight. Experiments on DanceTrack and SportsMOT show that MeMoSORT achieves state-of-the-art performance, with HOTA scores of 67.9\% and 82.1\%, respectively.
>
---
#### [new 121] RL-MoE: An Image-Based Privacy Preserving Approach In Intelligent Transportation System
- **分类: cs.CV; cs.AI**

- **简介: 论文提出RL-MoE框架，通过将图像转为文本描述实现隐私保护，解决传统方法在隐私与数据利用间的矛盾。结合MoE架构与强化学习，优化文本生成以兼顾语义准确性和隐私性，实验显示其攻击成功率降至9.4%，并生成更丰富的文本内容。**

- **链接: [http://arxiv.org/pdf/2508.09186v1](http://arxiv.org/pdf/2508.09186v1)**

> **作者:** Abdolazim Rezaei; Mehdi Sookhak; Mahboobeh Haghparast
>
> **摘要:** The proliferation of AI-powered cameras in Intelligent Transportation Systems (ITS) creates a severe conflict between the need for rich visual data and the fundamental right to privacy. Existing privacy-preserving mechanisms, such as blurring or encryption, are often insufficient, creating an undesirable trade-off where either privacy is compromised against advanced reconstruction attacks or data utility is critically degraded. To resolve this impasse, we propose RL-MoE, a novel framework that transforms sensitive visual data into privacy-preserving textual descriptions, eliminating the need for direct image transmission. RL-MoE uniquely combines a Mixture-of-Experts (MoE) architecture for nuanced, multi-aspect scene decomposition with a Reinforcement Learning (RL) agent that optimizes the generated text for a dual objective of semantic accuracy and privacy preservation. Extensive experiments demonstrate that RL-MoE provides superior privacy protection, reducing the success rate of replay attacks to just 9.4\% on the CFP-FP dataset, while simultaneously generating richer textual content than baseline methods. Our work provides a practical and scalable solution for building trustworthy AI systems in privacy-sensitive domains, paving the way for more secure smart city and autonomous vehicle networks.
>
---
#### [new 122] From Large Angles to Consistent Faces: Identity-Preserving Video Generation via Mixture of Facial Experts
- **分类: cs.CV**

- **简介: 论文提出基于MoFE模型的视频生成方法，解决大角度下身份丢失问题，通过融合三类面部专家（身份、语义、细节）及改进数据集提升生成质量，实现身份一致性。**

- **链接: [http://arxiv.org/pdf/2508.09476v1](http://arxiv.org/pdf/2508.09476v1)**

> **作者:** Yuji Wang; Moran Li; Xiaobin Hu; Ran Yi; Jiangning Zhang; Chengming Xu; Weijian Cao; Yabiao Wang; Chengjie Wang; Lizhuang Ma
>
> **摘要:** Current video generation models struggle with identity preservation under large facial angles, primarily facing two challenges: the difficulty in exploring an effective mechanism to integrate identity features into DiT structure, and the lack of targeted coverage of large facial angles in existing open-source video datasets. To address these, we present two key innovations. First, we introduce a Mixture of Facial Experts (MoFE) that dynamically combines complementary cues from three specialized experts, each designed to capture distinct but mutually reinforcing aspects of facial attributes. The identity expert captures cross-pose identity-sensitive features, the semantic expert extracts high-level visual semantxics, and the detail expert preserves pixel-level features (e.g., skin texture, color gradients). Furthermore, to mitigate dataset limitations, we have tailored a data processing pipeline centered on two key aspects: Face Constraints and Identity Consistency. Face Constraints ensure facial angle diversity and a high proportion of facial regions, while Identity Consistency preserves coherent person-specific features across temporal sequences, collectively addressing the scarcity of large facial angles and identity-stable training data in existing datasets. Leveraging this pipeline, we have curated and refined a Large Face Angles (LFA) Dataset from existing open-source human video datasets, comprising 460K video clips with annotated facial angles. Experimental results on the LFA benchmark demonstrate that our method, empowered by the LFA dataset, significantly outperforms prior SOTA methods in face similarity, face FID, and CLIP semantic alignment. The code and dataset will be made publicly available at https://github.com/rain152/LFA-Video-Generation.
>
---
#### [new 123] Zero-shot self-supervised learning of single breath-hold magnetic resonance cholangiopancreatography (MRCP) reconstruction
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出使用零样本自监督学习重建单次呼吸保持MRCP，减少呼吸保持时间，提升图像质量，通过预训练网络缩短训练时间。**

- **链接: [http://arxiv.org/pdf/2508.09200v1](http://arxiv.org/pdf/2508.09200v1)**

> **作者:** Jinho Kim; Marcel Dominik Nickel; Florian Knoll
>
> **备注:** 23 pages, 6 figures, 2 tabels
>
> **摘要:** Purpose: To investigate the feasibility of applying zero-shot self-supervised learning reconstruction to reduce breath-hold times in magnetic resonance cholangiopancreatography (MRCP). Methods: Breath-hold MRCP was acquired from 11 healthy volunteers on a 3T scanner using an incoherent k-space sampling pattern leading to a breath-hold duration of 14s. We evaluated zero-shot reconstruction of breath-hold MRCP against parallel imaging of respiratory-triggered MRCP acquired in 338s on average and compressed sensing reconstruction of breath-hold MRCP. To address the long computation times of zero-shot trainings, we used a training approach that leverages a pretrained network to reduce backpropagation depth during training. Results: Zero-shot learning reconstruction significantly improved visual image quality compared to compressed sensing reconstruction, particularly in terms of signal-to-noise ratio and ductal delineation, and reached a level of quality comparable to that of successful respiratory-triggered acquisitions with regular breathing patterns. Shallow training provided nearly equivalent reconstruction performance with a training time of 11 minutes in comparison to 271 minutes for a conventional zero-shot training. Conclusion: Zero-shot learning delivers high-fidelity MRCP reconstructions with reduced breath-hold times, and shallow training offers a practical solution for translation to time-constrained clinical workflows.
>
---
#### [new 124] Hybrid(Transformer+CNN)-based Polyp Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出一种基于Transformer与CNN的混合模型，解决结肠息肉分割中因尺寸、形状、光照及边界模糊导致的挑战，提升分割精度与抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2508.09189v1](http://arxiv.org/pdf/2508.09189v1)**

> **作者:** Madan Baduwal
>
> **备注:** 8 pages
>
> **摘要:** Colonoscopy is still the main method of detection and segmentation of colonic polyps, and recent advancements in deep learning networks such as U-Net, ResUNet, Swin-UNet, and PraNet have made outstanding performance in polyp segmentation. Yet, the problem is extremely challenging due to high variation in size, shape, endoscopy types, lighting, imaging protocols, and ill-defined boundaries (fluid, folds) of the polyps, rendering accurate segmentation a challenging and problematic task. To address these critical challenges in polyp segmentation, we introduce a hybrid (Transformer + CNN) model that is crafted to enhance robustness against evolving polyp characteristics. Our hybrid architecture demonstrates superior performance over existing solutions, particularly in addressing two critical challenges: (1) accurate segmentation of polyps with ill-defined margins through boundary-aware attention mechanisms, and (2) robust feature extraction in the presence of common endoscopic artifacts, including specular highlights, motion blur, and fluid occlusions. Quantitative evaluations reveal significant improvements in segmentation accuracy (Recall improved by 1.76%, i.e., 0.9555, accuracy improved by 0.07%, i.e., 0.9849) and artifact resilience compared to state-of-the-art polyp segmentation methods.
>
---
#### [new 125] MedPatch: Confidence-Guided Multi-Stage Fusion for Multimodal Clinical Data
- **分类: eess.IV; cs.CV**

- **简介: 论文提出MedPatch，通过信心引导的多阶段融合架构解决多模态临床数据异构性与稀疏性问题，提升临床预测性能。**

- **链接: [http://arxiv.org/pdf/2508.09182v1](http://arxiv.org/pdf/2508.09182v1)**

> **作者:** Baraa Al Jorf; Farah Shamout
>
> **摘要:** Clinical decision-making relies on the integration of information across various data modalities, such as clinical time-series, medical images and textual reports. Compared to other domains, real-world medical data is heterogeneous in nature, limited in size, and sparse due to missing modalities. This significantly limits model performance in clinical prediction tasks. Inspired by clinical workflows, we introduce MedPatch, a multi-stage multimodal fusion architecture, which seamlessly integrates multiple modalities via confidence-guided patching. MedPatch comprises three main components: (i) a multi-stage fusion strategy that leverages joint and late fusion simultaneously, (ii) a missingness-aware module that handles sparse samples with missing modalities, (iii) a joint fusion module that clusters latent token patches based on calibrated unimodal token-level confidence. We evaluated MedPatch using real-world data consisting of clinical time-series data, chest X-ray images, radiology reports, and discharge notes extracted from the MIMIC-IV, MIMIC-CXR, and MIMIC-Notes datasets on two benchmark tasks, namely in-hospital mortality prediction and clinical condition classification. Compared to existing baselines, MedPatch achieves state-of-the-art performance. Our work highlights the effectiveness of confidence-guided multi-stage fusion in addressing the heterogeneity of multimodal data, and establishes new state-of-the-art benchmark results for clinical prediction tasks.
>
---
#### [new 126] HiFi-Mamba: Dual-Stream W-Laplacian Enhanced Mamba for High-Fidelity MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 论文提出HiFi-Mamba，通过双流结构与W-Laplacian增强，解决MRI重建中高频细节缺失及冗余扫描问题，提升高保真图像重建精度。**

- **链接: [http://arxiv.org/pdf/2508.09179v1](http://arxiv.org/pdf/2508.09179v1)**

> **作者:** Hongli Chen; Pengcheng Fang; Yuxia Chen; Yingxuan Ren; Jing Hao; Fangfang Tang; Xiaohao Cai; Shanshan Shan; Feng Liu
>
> **摘要:** Reconstructing high-fidelity MR images from undersampled k-space data remains a challenging problem in MRI. While Mamba variants for vision tasks offer promising long-range modeling capabilities with linear-time complexity, their direct application to MRI reconstruction inherits two key limitations: (1) insensitivity to high-frequency anatomical details; and (2) reliance on redundant multi-directional scanning. To address these limitations, we introduce High-Fidelity Mamba (HiFi-Mamba), a novel dual-stream Mamba-based architecture comprising stacked W-Laplacian (WL) and HiFi-Mamba blocks. Specifically, the WL block performs fidelity-preserving spectral decoupling, producing complementary low- and high-frequency streams. This separation enables the HiFi-Mamba block to focus on low-frequency structures, enhancing global feature modeling. Concurrently, the HiFi-Mamba block selectively integrates high-frequency features through adaptive state-space modulation, preserving comprehensive spectral details. To eliminate the scanning redundancy, the HiFi-Mamba block adopts a streamlined unidirectional traversal strategy that preserves long-range modeling capability with improved computational efficiency. Extensive experiments on standard MRI reconstruction benchmarks demonstrate that HiFi-Mamba consistently outperforms state-of-the-art CNN-based, Transformer-based, and other Mamba-based models in reconstruction accuracy while maintaining a compact and efficient model design.
>
---
#### [new 127] Masked Training for Robust Arrhythmia Detection from Digitalized Multiple Layout ECG Images
- **分类: cs.LG; cs.CV**

- **简介: 本文提出基于掩码训练的PatchECG框架，解决异构ECG布局导致的鲁棒性不足问题，通过自动聚焦关键区域实现心律不齐检测。实验表明其在PTB-XL和真实数据集上表现优异，AUROC达0.835，优于传统方法及现有最优模型。**

- **链接: [http://arxiv.org/pdf/2508.09165v1](http://arxiv.org/pdf/2508.09165v1)**

> **作者:** Shanwei Zhang; Deyun Zhang; Yirao Tao; Kexin Wang; Shijia Geng; Jun Li; Qinghao Zhao; Xingpeng Liu; Yuxi Zhou; Shenda Hong
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Electrocardiogram (ECG) as an important tool for diagnosing cardiovascular diseases such as arrhythmia. Due to the differences in ECG layouts used by different hospitals, the digitized signals exhibit asynchronous lead time and partial blackout loss, which poses a serious challenge to existing models. To address this challenge, the study introduced PatchECG, a framework for adaptive variable block count missing representation learning based on a masking training strategy, which automatically focuses on key patches with collaborative dependencies between leads, thereby achieving key recognition of arrhythmia in ECGs with different layouts. Experiments were conducted on the PTB-XL dataset and 21388 asynchronous ECG images generated using ECG image kit tool, using the 23 Subclasses as labels. The proposed method demonstrated strong robustness under different layouts, with average Area Under the Receiver Operating Characteristic Curve (AUROC) of 0.835 and remained stable (unchanged with layout changes). In external validation based on 400 real ECG images data from Chaoyang Hospital, the AUROC for atrial fibrillation diagnosis reached 0.778; On 12 x 1 layout ECGs, AUROC reaches 0.893. This result is superior to various classic interpolation and baseline methods, and compared to the current optimal large-scale pre-training model ECGFounder, it has improved by 0.111 and 0.19.
>
---
#### [new 128] VisCodex: Unified Multimodal Code Generation via Merging Vision and Coding Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出VisCodex，通过融合视觉与编码模型解决多模态代码生成问题，构建MCD和InfiBench-V基准，实现开源MLLMs的代码生成突破。**

- **链接: [http://arxiv.org/pdf/2508.09945v1](http://arxiv.org/pdf/2508.09945v1)**

> **作者:** Lingjie Jiang; Shaohan Huang; Xun Wu; Yixia Li; Dongdong Zhang; Furu Wei
>
> **摘要:** Multimodal large language models (MLLMs) have significantly advanced the integration of visual and textual understanding. However, their ability to generate code from multimodal inputs remains limited. In this work, we introduce VisCodex, a unified framework that seamlessly merges vision and coding language models to empower MLLMs with strong multimodal code generation abilities. Leveraging a task vector-based model merging technique, we integrate a state-of-the-art coding LLM into a strong vision-language backbone, while preserving both visual comprehension and advanced coding skills. To support training and evaluation, we introduce the Multimodal Coding Dataset (MCD), a large-scale and diverse collection of 598k samples, including high-quality HTML code, chart image-code pairs, image-augmented StackOverflow QA, and algorithmic problems. Furthermore, we propose InfiBench-V, a novel and challenging benchmark specifically designed to assess models on visually-rich, real-world programming questions that demand a nuanced understanding of both textual and visual contexts. Extensive experiments show that VisCodex achieves state-of-the-art performance among open-source MLLMs and approaches proprietary models like GPT-4o, highlighting the effectiveness of our model merging strategy and new datasets.
>
---
#### [new 129] Perceptual Reality Transformer: Neural Architectures for Simulating Neurological Perception Conditions
- **分类: q-bio.NC; cs.AI; cs.CV; cs.NE**

- **简介: 论文提出Perceptual Reality Transformer框架，利用六种神经架构模拟八种神经感知条件，通过图像映射生成特定感知状态，评估其在ImageNet和CIFAR-10上的性能，建立系统基准。**

- **链接: [http://arxiv.org/pdf/2508.09852v1](http://arxiv.org/pdf/2508.09852v1)**

> **作者:** Baihan Lin
>
> **摘要:** Neurological conditions affecting visual perception create profound experiential divides between affected individuals and their caregivers, families, and medical professionals. We present the Perceptual Reality Transformer, a comprehensive framework employing six distinct neural architectures to simulate eight neurological perception conditions with scientifically-grounded visual transformations. Our system learns mappings from natural images to condition-specific perceptual states, enabling others to experience approximations of simultanagnosia, prosopagnosia, ADHD attention deficits, visual agnosia, depression-related changes, anxiety tunnel vision, and Alzheimer's memory effects. Through systematic evaluation across ImageNet and CIFAR-10 datasets, we demonstrate that Vision Transformer architectures achieve optimal performance, outperforming traditional CNN and generative approaches. Our work establishes the first systematic benchmark for neurological perception simulation, contributes novel condition-specific perturbation functions grounded in clinical literature, and provides quantitative metrics for evaluating simulation fidelity. The framework has immediate applications in medical education, empathy training, and assistive technology development, while advancing our fundamental understanding of how neural networks can model atypical human perception.
>
---
#### [new 130] Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations
- **分类: cs.IR; cs.CV**

- **简介: 论文提出一种零微调框架，利用多模态大语言模型生成视频文本描述，填补低级内容与用户意图间的语义鸿沟，提升视频推荐效果。**

- **链接: [http://arxiv.org/pdf/2508.09789v1](http://arxiv.org/pdf/2508.09789v1)**

> **作者:** Marco De Nadai; Andreas Damianou; Mounia Lalmas
>
> **摘要:** Existing video recommender systems rely primarily on user-defined metadata or on low-level visual and acoustic signals extracted by specialised encoders. These low-level features describe what appears on the screen but miss deeper semantics such as intent, humour, and world knowledge that make clips resonate with viewers. For example, is a 30-second clip simply a singer on a rooftop, or an ironic parody filmed amid the fairy chimneys of Cappadocia, Turkey? Such distinctions are critical to personalised recommendations yet remain invisible to traditional encoding pipelines. In this paper, we introduce a simple, recommendation system-agnostic zero-finetuning framework that injects high-level semantics into the recommendation pipeline by prompting an off-the-shelf Multimodal Large Language Model (MLLM) to summarise each clip into a rich natural-language description (e.g. "a superhero parody with slapstick fights and orchestral stabs"), bridging the gap between raw content and user intent. We use MLLM output with a state-of-the-art text encoder and feed it into standard collaborative, content-based, and generative recommenders. On the MicroLens-100K dataset, which emulates user interactions with TikTok-style videos, our framework consistently surpasses conventional video, audio, and metadata features in five representative models. Our findings highlight the promise of leveraging MLLMs as on-the-fly knowledge extractors to build more intent-aware video recommenders.
>
---
#### [new 131] MoLAN: A Unified Modality-Aware Noise Dynamic Editing Framework for Multimodal Sentiment Analysis
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 论文提出MoLAN框架，解决多模态情感分析中噪声与冗余信息干扰问题，通过动态分块去噪提升多模态信息融合效率，实现对关键信息的保留。**

- **链接: [http://arxiv.org/pdf/2508.09145v1](http://arxiv.org/pdf/2508.09145v1)**

> **作者:** Xingle Xu; Yongkang Liu; Dexian Cai; Shi Feng; Xiaocui Yang; Daling Wang; Yifei Zhang
>
> **摘要:** Multimodal Sentiment Analysis aims to integrate information from various modalities, such as audio, visual, and text, to make complementary predictions. However, it often struggles with irrelevant or misleading visual and auditory information. Most existing approaches typically treat the entire modality information (e.g., a whole image, audio segment, or text paragraph) as an independent unit for feature enhancement or denoising. They often suppress the redundant and noise information at the risk of losing critical information. To address this challenge, we propose MoLAN, a unified ModaLity-aware noise dynAmic editiNg framework. Specifically, MoLAN performs modality-aware blocking by dividing the features of each modality into multiple blocks. Each block is then dynamically assigned a distinct denoising strength based on its noise level and semantic relevance, enabling fine-grained noise suppression while preserving essential multimodal information. Notably, MoLAN is a unified and flexible framework that can be seamlessly integrated into a wide range of multimodal models. Building upon this framework, we further introduce MoLAN+, a new multimodal sentiment analysis approach. Experiments across five models and four datasets demonstrate the broad effectiveness of the MoLAN framework. Extensive evaluations show that MoLAN+ achieves the state-of-the-art performance. The code is publicly available at https://github.com/betterfly123/MoLAN-Framework.
>
---
#### [new 132] AMRG: Extend Vision Language Models for Automatic Mammography Report Generation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出AMRG框架，利用大视觉语言模型生成乳腺X光报告，解决多视角图像推理与非结构化语言问题，通过LoRA微调实现轻量级适配，取得ROUGE-L 0.5691等指标，建立首个可复现的乳腺报告生成基准。**

- **链接: [http://arxiv.org/pdf/2508.09225v1](http://arxiv.org/pdf/2508.09225v1)**

> **作者:** Nak-Jun Sung; Donghyun Lee; Bo Hwa Choi; Chae Jung Park
>
> **摘要:** Mammography report generation is a critical yet underexplored task in medical AI, characterized by challenges such as multiview image reasoning, high-resolution visual cues, and unstructured radiologic language. In this work, we introduce AMRG (Automatic Mammography Report Generation), the first end-to-end framework for generating narrative mammography reports using large vision-language models (VLMs). Building upon MedGemma-4B-it-a domain-specialized, instruction-tuned VLM-we employ a parameter-efficient fine-tuning (PEFT) strategy via Low-Rank Adaptation (LoRA), enabling lightweight adaptation with minimal computational overhead. We train and evaluate AMRG on DMID, a publicly available dataset of paired high-resolution mammograms and diagnostic reports. This work establishes the first reproducible benchmark for mammography report generation, addressing a longstanding gap in multimodal clinical AI. We systematically explore LoRA hyperparameter configurations and conduct comparative experiments across multiple VLM backbones, including both domain-specific and general-purpose models under a unified tuning protocol. Our framework demonstrates strong performance across both language generation and clinical metrics, achieving a ROUGE-L score of 0.5691, METEOR of 0.6152, CIDEr of 0.5818, and BI-RADS accuracy of 0.5582. Qualitative analysis further highlights improved diagnostic consistency and reduced hallucinations. AMRG offers a scalable and adaptable foundation for radiology report generation and paves the way for future research in multimodal medical AI.
>
---
#### [new 133] Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models: A Unified and Accurate Approach
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 论文提出LoD框架，用于检测大型视觉-语言模型的未知Jailbreak攻击，通过多模态安全概念激活向量与自动编码器，无需标签，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.09201v1](http://arxiv.org/pdf/2508.09201v1)**

> **作者:** Shuang Liang; Zhihao Xu; Jialing Tao; Hui Xue; Xiting Wang
>
> **摘要:** Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. Although recent detection works have shifted to internal representations due to their rich cross-modal information, most methods rely on heuristic rules rather than principled objectives, resulting in suboptimal performance. To address these limitations, we propose Learning to Detect (LoD), a novel unsupervised framework that formulates jailbreak detection as anomaly detection. LoD introduces two key components: Multi-modal Safety Concept Activation Vectors (MSCAV), which capture layer-wise safety-related representations across modalities, and the Safety Pattern Auto-Encoder, which models the distribution of MSCAV derived from safe inputs and detects anomalies via reconstruction errors. By training the auto-encoder (AE) solely on safe samples without attack labels, LoD naturally identifies jailbreak inputs as distributional anomalies, enabling accurate and unified detection of jailbreak attacks. Comprehensive experiments on three different LVLMs and five benchmarks demonstrate that LoD achieves state-of-the-art performance, with an average AUROC of 0.9951 and an improvement of up to 38.89% in the minimum AUROC over the strongest baselines.
>
---
#### [new 134] impuTMAE: Multi-modal Transformer with Masked Pre-training for Missing Modalities Imputation in Cancer Survival Prediction
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出一种多模态Transformer模型impuTMAE，通过掩码预训练填补癌症生存预测中的缺失模态，整合基因组、影像及临床数据，实现高效多模态建模，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2508.09195v1](http://arxiv.org/pdf/2508.09195v1)**

> **作者:** Maria Boyko; Aleksandra Beliaeva; Dmitriy Kornilov; Alexander Bernstein; Maxim Sharaev
>
> **摘要:** The use of diverse modalities, such as omics, medical images, and clinical data can not only improve the performance of prognostic models but also deepen an understanding of disease mechanisms and facilitate the development of novel treatment approaches. However, medical data are complex, often incomplete, and contains missing modalities, making effective handling its crucial for training multimodal models. We introduce impuTMAE, a novel transformer-based end-to-end approach with an efficient multimodal pre-training strategy. It learns inter- and intra-modal interactions while simultaneously imputing missing modalities by reconstructing masked patches. Our model is pre-trained on heterogeneous, incomplete data and fine-tuned for glioma survival prediction using TCGA-GBM/LGG and BraTS datasets, integrating five modalities: genetic (DNAm, RNA-seq), imaging (MRI, WSI), and clinical data. By addressing missing data during pre-training and enabling efficient resource utilization, impuTMAE surpasses prior multimodal approaches, achieving state-of-the-art performance in glioma patient survival prediction. Our code is available at https://github.com/maryjis/mtcp
>
---
#### [new 135] From Explainable to Explained AI: Ideas for Falsifying and Quantifying Explanations
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出基于人机-VLM的系统，用于验证和量化解释，解决解释有效性问题，通过滑动窗口实验和视觉语言模型量化预测性。**

- **链接: [http://arxiv.org/pdf/2508.09205v1](http://arxiv.org/pdf/2508.09205v1)**

> **作者:** Yoni Schirris; Eric Marcus; Jonas Teuwen; Hugo Horlings; Efstratios Gavves
>
> **备注:** 10 pages, 2 figures, 2 tables, submitted at MICCAI IMIMIC workshop
>
> **摘要:** Explaining deep learning models is essential for clinical integration of medical image analysis systems. A good explanation highlights if a model depends on spurious features that undermines generalization and harms a subset of patients or, conversely, may present novel biological insights. Although techniques like GradCAM can identify influential features, they are measurement tools that do not themselves form an explanation. We propose a human-machine-VLM interaction system tailored to explaining classifiers in computational pathology, including multi-instance learning for whole-slide images. Our proof of concept comprises (1) an AI-integrated slide viewer to run sliding-window experiments to test claims of an explanation, and (2) quantification of an explanation's predictiveness using general-purpose vision-language models. The results demonstrate that this allows us to qualitatively test claims of explanations and can quantifiably distinguish competing explanations. This offers a practical path from explainable AI to explained AI in digital pathology and beyond. Code and prompts are available at https://github.com/nki-ai/x2x.
>
---
#### [new 136] Speed Always Wins: A Survey on Efficient Architectures for Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文综述高效LLM架构，解决传统Transformer计算量大问题，提出线性稀疏模型、混合专家等技术，覆盖多模态应用并探讨资源优化方向。**

- **链接: [http://arxiv.org/pdf/2508.09834v1](http://arxiv.org/pdf/2508.09834v1)**

> **作者:** Weigao Sun; Jiaxi Hu; Yucheng Zhou; Jusen Du; Disen Lan; Kexin Wang; Tong Zhu; Xiaoye Qu; Yu Zhang; Xiaoyu Mo; Daizong Liu; Yuxuan Liang; Wenliang Chen; Guoqi Li; Yu Cheng
>
> **备注:** Survey, 82 pages, GitHub: https://github.com/weigao266/Awesome-Efficient-Arch
>
> **摘要:** Large Language Models (LLMs) have delivered impressive results in language understanding, generation, reasoning, and pushes the ability boundary of multimodal models. Transformer models, as the foundation of modern LLMs, offer a strong baseline with excellent scaling properties. However, the traditional transformer architecture requires substantial computations and poses significant obstacles for large-scale training and practical deployment. In this survey, we offer a systematic examination of innovative LLM architectures that address the inherent limitations of transformers and boost the efficiency. Starting from language modeling, this survey covers the background and technical details of linear and sparse sequence modeling methods, efficient full attention variants, sparse mixture-of-experts, hybrid model architectures incorporating the above techniques, and emerging diffusion LLMs. Additionally, we discuss applications of these techniques to other modalities and consider their wider implications for developing scalable, resource-aware foundation models. By grouping recent studies into the above category, this survey presents a blueprint of modern efficient LLM architectures, and we hope this could help motivate future research toward more efficient, versatile AI systems.
>
---
#### [new 137] Noise Hypernetworks: Amortizing Test-Time Compute in Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 论文提出噪声超网络框架，通过替代传统测试时扩展方法，减少计算开销，保留性能提升，实现低成本高质的测试时扩展。**

- **链接: [http://arxiv.org/pdf/2508.09968v1](http://arxiv.org/pdf/2508.09968v1)**

> **作者:** Luca Eyring; Shyamgopal Karthik; Alexey Dosovitskiy; Nataniel Ruiz; Zeynep Akata
>
> **备注:** Project page: https://noisehypernetworks.github.io/
>
> **摘要:** The new paradigm of test-time scaling has yielded remarkable breakthroughs in Large Language Models (LLMs) (e.g. reasoning models) and in generative vision models, allowing models to allocate additional computation during inference to effectively tackle increasingly complex problems. Despite the improvements of this approach, an important limitation emerges: the substantial increase in computation time makes the process slow and impractical for many applications. Given the success of this paradigm and its growing usage, we seek to preserve its benefits while eschewing the inference overhead. In this work we propose one solution to the critical problem of integrating test-time scaling knowledge into a model during post-training. Specifically, we replace reward guided test-time noise optimization in diffusion models with a Noise Hypernetwork that modulates initial input noise. We propose a theoretically grounded framework for learning this reward-tilted distribution for distilled generators, through a tractable noise-space objective that maintains fidelity to the base model while optimizing for desired characteristics. We show that our approach recovers a substantial portion of the quality gains from explicit test-time optimization at a fraction of the computational cost. Code is available at https://github.com/ExplainableML/HyperNoise
>
---
#### [new 138] Real-time deep learning phase imaging flow cytometer reveals blood cell aggregate biomarkers for haematology diagnostics
- **分类: q-bio.QM; cs.AI; cs.CV; cs.LG; eess.IV**

- **简介: 论文提出实时深度学习框架RT-HAD，用于检测血液细胞聚集的生物标志物，解决传统方法无法识别聚集的问题，实现高效、准确的实时诊断。**

- **链接: [http://arxiv.org/pdf/2508.09215v1](http://arxiv.org/pdf/2508.09215v1)**

> **作者:** Kerem Delikoyun; Qianyu Chen; Liu Wei; Si Ko Myo; Johannes Krell; Martin Schlegel; Win Sen Kuan; John Tshon Yit Soong; Gerhard Schneider; Clarissa Prazeres da Costa; Percy A. Knolle; Laurent Renia; Matthew Edward Cove; Hwee Kuan Lee; Klaus Diepold; Oliver Hayden
>
> **摘要:** While analysing rare blood cell aggregates remains challenging in automated haematology, they could markedly advance label-free functional diagnostics. Conventional flow cytometers efficiently perform cell counting with leukocyte differentials but fail to identify aggregates with flagged results, requiring manual reviews. Quantitative phase imaging flow cytometry captures detailed aggregate morphologies, but clinical use is hampered by massive data storage and offline processing. Incorporating hidden biomarkers into routine haematology panels would significantly improve diagnostics without flagged results. We present RT-HAD, an end-to-end deep learning-based image and data processing framework for off-axis digital holographic microscopy (DHM), which combines physics-consistent holographic reconstruction and detection, representing each blood cell in a graph to recognize aggregates. RT-HAD processes >30 GB of image data on-the-fly with turnaround time of <1.5 min and error rate of 8.9% in platelet aggregate detection, which matches acceptable laboratory error rates of haematology biomarkers and solves the big data challenge for point-of-care diagnostics.
>
---
#### [new 139] Generative Artificial Intelligence in Medical Imaging: Foundations, Progress, and Clinical Translation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文综述生成式AI在医学影像中的应用，涵盖模型、临床应用及挑战，提出评估框架和未来方向。**

- **链接: [http://arxiv.org/pdf/2508.09177v1](http://arxiv.org/pdf/2508.09177v1)**

> **作者:** Xuanru Zhou; Cheng Li; Shuqiang Wang; Ye Li; Tao Tan; Hairong Zheng; Shanshan Wang
>
> **摘要:** Generative artificial intelligence (AI) is rapidly transforming medical imaging by enabling capabilities such as data synthesis, image enhancement, modality translation, and spatiotemporal modeling. This review presents a comprehensive and forward-looking synthesis of recent advances in generative modeling including generative adversarial networks (GANs), variational autoencoders (VAEs), diffusion models, and emerging multimodal foundation architectures and evaluates their expanding roles across the clinical imaging continuum. We systematically examine how generative AI contributes to key stages of the imaging workflow, from acquisition and reconstruction to cross-modality synthesis, diagnostic support, and treatment planning. Emphasis is placed on both retrospective and prospective clinical scenarios, where generative models help address longstanding challenges such as data scarcity, standardization, and integration across modalities. To promote rigorous benchmarking and translational readiness, we propose a three-tiered evaluation framework encompassing pixel-level fidelity, feature-level realism, and task-level clinical relevance. We also identify critical obstacles to real-world deployment, including generalization under domain shift, hallucination risk, data privacy concerns, and regulatory hurdles. Finally, we explore the convergence of generative AI with large-scale foundation models, highlighting how this synergy may enable the next generation of scalable, reliable, and clinically integrated imaging systems. By charting technical progress and translational pathways, this review aims to guide future research and foster interdisciplinary collaboration at the intersection of AI, medicine, and biomedical engineering.
>
---
#### [new 140] Robustness analysis of Deep Sky Objects detection models on HPC
- **分类: astro-ph.IM; cs.CV**

- **简介: 论文分析HPC环境下深空对象检测模型的鲁棒性，解决因信号弱和背景复杂导致的检测难题，通过对比YOLO和RET-DETR模型实现高效并行计算。**

- **链接: [http://arxiv.org/pdf/2508.09831v1](http://arxiv.org/pdf/2508.09831v1)**

> **作者:** Olivier Parisot; Diogo Ramalho Fernandes
>
> **备注:** 11 pages, 4 figures, NEOD project
>
> **摘要:** Astronomical surveys and the growing involvement of amateur astronomers are producing more sky images than ever before, and this calls for automated processing methods that are accurate and robust. Detecting Deep Sky Objects -- such as galaxies, nebulae, and star clusters -- remains challenging because of their faint signals and complex backgrounds. Advances in Computer Vision and Deep Learning now make it possible to improve and automate this process. In this paper, we present the training and comparison of different detection models (YOLO, RET-DETR) on smart telescope images, using High-Performance Computing (HPC) to parallelise computations, in particular for robustness testing.
>
---
#### [new 141] T-CACE: A Time-Conditioned Autoregressive Contrast Enhancement Multi-Task Framework for Contrast-Free Liver MRI Synthesis, Segmentation, and Diagnosis
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出T-CACE框架，用于无对比剂肝脏MRI合成、分割与诊断，解决传统MRI对比剂风险、手动评估和数据不足问题，通过CTE、DTAM和TCC提升性能，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.09919v1](http://arxiv.org/pdf/2508.09919v1)**

> **作者:** Xiaojiao Xiao; Jianfeng Zhao; Qinmin Vivian Hu; Guanghui Wang
>
> **备注:** IEEE Journal of Biomedical and Health Informatics, 2025
>
> **摘要:** Magnetic resonance imaging (MRI) is a leading modality for the diagnosis of liver cancer, significantly improving the classification of the lesion and patient outcomes. However, traditional MRI faces challenges including risks from contrast agent (CA) administration, time-consuming manual assessment, and limited annotated datasets. To address these limitations, we propose a Time-Conditioned Autoregressive Contrast Enhancement (T-CACE) framework for synthesizing multi-phase contrast-enhanced MRI (CEMRI) directly from non-contrast MRI (NCMRI). T-CACE introduces three core innovations: a conditional token encoding (CTE) mechanism that unifies anatomical priors and temporal phase information into latent representations; and a dynamic time-aware attention mask (DTAM) that adaptively modulates inter-phase information flow using a Gaussian-decayed attention mechanism, ensuring smooth and physiologically plausible transitions across phases. Furthermore, a constraint for temporal classification consistency (TCC) aligns the lesion classification output with the evolution of the physiological signal, further enhancing diagnostic reliability. Extensive experiments on two independent liver MRI datasets demonstrate that T-CACE outperforms state-of-the-art methods in image synthesis, segmentation, and lesion classification. This framework offers a clinically relevant and efficient alternative to traditional contrast-enhanced imaging, improving safety, diagnostic efficiency, and reliability for the assessment of liver lesion. The implementation of T-CACE is publicly available at: https://github.com/xiaojiao929/T-CACE.
>
---
#### [new 142] SVGen: Interpretable Vector Graphics Generation with Large Language Models
- **分类: cs.LG; cs.CV**

- **简介: 论文提出SVGen，利用大语言模型生成可解释的矢量图形，解决自然语言到SVG转换效率低的问题，通过构建SVG-1M数据集和优化模型实现高效生成。**

- **链接: [http://arxiv.org/pdf/2508.09168v1](http://arxiv.org/pdf/2508.09168v1)**

> **作者:** Feiyu Wang; Zhiyuan Zhao; Yuandong Liu; Da Zhang; Junyu Gao; Hao Sun; Xuelong Li
>
> **摘要:** Scalable Vector Graphics (SVG) is widely used in front-end development and UI/UX design due to its scalability, editability, and rendering efficiency. However, turning creative ideas into precise vector graphics remains a time-consuming challenge. To address this, we introduce SVG-1M, a large-scale dataset of high-quality SVGs paired with natural language descriptions. Through advanced data augmentation and annotation, we create well-aligned Text to SVG training pairs, including a subset with Chain of Thought annotations for enhanced semantic guidance. Based on this dataset, we propose SVGen, an end-to-end model that generates SVG code from natural language inputs. Our approach ensures semantic accuracy and structural completeness, supported by curriculum learning and reinforcement learning optimization. Experiments show that SVGen outperforms general large models and traditional rendering methods in both effectiveness and efficiency. Code, model, and dataset are available on GitHub.
>
---
#### [new 143] Multimodal RAG Enhanced Visual Description
- **分类: cs.LG; cs.AI; cs.CV; cs.IR**

- **简介: 论文提出轻量级无训练方法，利用RAG和线性映射解决多模态视觉描述中的模态间隙问题，通过检索与指令生成优化。**

- **链接: [http://arxiv.org/pdf/2508.09170v1](http://arxiv.org/pdf/2508.09170v1)**

> **作者:** Amit Kumar Jaiswal; Haiming Liu; Ingo Frommholz
>
> **备注:** Accepted by ACM CIKM 2025. 5 pages, 2 figures
>
> **摘要:** Textual descriptions for multimodal inputs entail recurrent refinement of queries to produce relevant output images. Despite efforts to address challenges such as scaling model size and data volume, the cost associated with pre-training and fine-tuning remains substantial. However, pre-trained large multimodal models (LMMs) encounter a modality gap, characterised by a misalignment between textual and visual representations within a common embedding space. Although fine-tuning can potentially mitigate this gap, it is typically expensive and impractical due to the requirement for extensive domain-driven data. To overcome this challenge, we propose a lightweight training-free approach utilising Retrieval-Augmented Generation (RAG) to extend across the modality using a linear mapping, which can be computed efficiently. During inference, this mapping is applied to images embedded by an LMM enabling retrieval of closest textual descriptions from the training set. These textual descriptions, in conjunction with an instruction, cater as an input prompt for the language model to generate new textual descriptions. In addition, we introduce an iterative technique for distilling the mapping by generating synthetic descriptions via the language model facilitating optimisation for standard utilised image description measures. Experimental results on two benchmark multimodal datasets demonstrate significant improvements.
>
---
#### [new 144] Toward Human-Robot Teaming: Learning Handover Behaviors from 3D Scenes
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文提出一种基于RGB图像和3D场景重建的机器人手部交接学习方法，解决模拟与真实环境视觉域差距及数据采集成本高的问题，通过相机位姿映射生成机器人演示，提升HRT的稳定性与安全性。**

- **链接: [http://arxiv.org/pdf/2508.09855v1](http://arxiv.org/pdf/2508.09855v1)**

> **作者:** Yuekun Wu; Yik Lung Pang; Andrea Cavallaro; Changjae Oh
>
> **备注:** 3 pages, 3 figures
>
> **摘要:** Human-robot teaming (HRT) systems often rely on large-scale datasets of human and robot interactions, especially for close-proximity collaboration tasks such as human-robot handovers. Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although simulation training offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. We introduce a method for training HRT policies, focusing on human-to-robot handovers, solely from RGB images without the need for real-robot training or real-robot data collection. The goal is to enable the robot to reliably receive objects from a human with stable grasping while avoiding collisions with the human hand. The proposed policy learner leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. Experiments in both Gaussian Splatting reconstructed scene and real-world human-to-robot handover experiments demonstrate that our method serves as a new and effective representation for the human-to-robot handover task, contributing to more seamless and robust HRT.
>
---
#### [new 145] Combating Noisy Labels via Dynamic Connection Masking
- **分类: cs.LG; cs.CV**

- **简介: 论文提出动态连接掩码（DCM）机制，通过自适应掩码不重要边提升深度网络对噪声标签的鲁棒性，结合KANs稀疏正则化思想，解决噪声标签导致的性能退化问题，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.09697v1](http://arxiv.org/pdf/2508.09697v1)**

> **作者:** Xinlei Zhang; Fan Liu; Chuanyi Zhang; Fan Cheng; Yuhui Zheng
>
> **摘要:** Noisy labels are inevitable in real-world scenarios. Due to the strong capacity of deep neural networks to memorize corrupted labels, these noisy labels can cause significant performance degradation. Existing research on mitigating the negative effects of noisy labels has mainly focused on robust loss functions and sample selection, with comparatively limited exploration of regularization in model architecture. Inspired by the sparsity regularization used in Kolmogorov-Arnold Networks (KANs), we propose a Dynamic Connection Masking (DCM) mechanism for both Multi-Layer Perceptron Networks (MLPs) and KANs to enhance the robustness of classifiers against noisy labels. The mechanism can adaptively mask less important edges during training by evaluating their information-carrying capacity. Through theoretical analysis, we demonstrate its efficiency in reducing gradient error. Our approach can be seamlessly integrated into various noise-robust training methods to build more robust deep networks, including robust loss functions, sample selection strategies, and regularization techniques. Extensive experiments on both synthetic and real-world benchmarks demonstrate that our method consistently outperforms state-of-the-art (SOTA) approaches. Furthermore, we are also the first to investigate KANs as classifiers against noisy labels, revealing their superior noise robustness over MLPs in real-world noisy scenarios. Our code will soon be publicly available.
>
---
#### [new 146] DAgger Diffusion Navigation: DAgger Boosted Diffusion Policy for Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种基于扩散模型的视觉-语言导航方法DifNav，解决传统两阶段框架的全局子优化和依赖起点的问题，通过单阶段扩散政策直接建模多模态动作分布，结合DAgger实现在线训练与轨迹增强，显著提升导航性能。**

- **链接: [http://arxiv.org/pdf/2508.09444v1](http://arxiv.org/pdf/2508.09444v1)**

> **作者:** Haoxiang Shi; Xiang Deng; Zaijing Li; Gongwei Chen; Yaowei Wang; Liqiang Nie
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to follow natural language instructions through free-form 3D spaces. Existing VLN-CE approaches typically use a two-stage waypoint planning framework, where a high-level waypoint predictor generates the navigable waypoints, and then a navigation planner suggests the intermediate goals in the high-level action space. However, this two-stage decomposition framework suffers from: (1) global sub-optimization due to the proxy objective in each stage, and (2) a performance bottleneck caused by the strong reliance on the quality of the first-stage predicted waypoints. To address these limitations, we propose DAgger Diffusion Navigation (DifNav), an end-to-end optimized VLN-CE policy that unifies the traditional two stages, i.e. waypoint generation and planning, into a single diffusion policy. Notably, DifNav employs a conditional diffusion policy to directly model multi-modal action distributions over future actions in continuous navigation space, eliminating the need for a waypoint predictor while enabling the agent to capture multiple possible instruction-following behaviors. To address the issues of compounding error in imitation learning and enhance spatial reasoning in long-horizon navigation tasks, we employ DAgger for online policy training and expert trajectory augmentation, and use the aggregated data to further fine-tune the policy. This approach significantly improves the policy's robustness and its ability to recover from error states. Extensive experiments on benchmark datasets demonstrate that, even without a waypoint predictor, the proposed method substantially outperforms previous state-of-the-art two-stage waypoint-based models in terms of navigation performance. Our code is available at: https://github.com/Tokishx/DifNav.
>
---
#### [new 147] Dynamic Survival Prediction using Longitudinal Images based on Transformer
- **分类: eess.IV; cs.CV; stat.AP; stat.OT**

- **简介: 论文提出基于Transformer的SurLonFormer模型，用于生存预测，解决删失数据、纵向关联及可解释性问题，通过整合视觉与序列编码器提升性能。**

- **链接: [http://arxiv.org/pdf/2508.09328v1](http://arxiv.org/pdf/2508.09328v1)**

> **作者:** Bingfan Liu; Haolun Shi; Jiguo Cao
>
> **摘要:** Survival analysis utilizing multiple longitudinal medical images plays a pivotal role in the early detection and prognosis of diseases by providing insight beyond single-image evaluations. However, current methodologies often inadequately utilize censored data, overlook correlations among longitudinal images measured over multiple time points, and lack interpretability. We introduce SurLonFormer, a novel Transformer-based neural network that integrates longitudinal medical imaging with structured data for survival prediction. Our architecture comprises three key components: a Vision Encoder for extracting spatial features, a Sequence Encoder for aggregating temporal information, and a Survival Encoder based on the Cox proportional hazards model. This framework effectively incorporates censored data, addresses scalability issues, and enhances interpretability through occlusion sensitivity analysis and dynamic survival prediction. Extensive simulations and a real-world application in Alzheimer's disease analysis demonstrate that SurLonFormer achieves superior predictive performance and successfully identifies disease-related imaging biomarkers.
>
---
#### [new 148] FIVA: Federated Inverse Variance Averaging for Universal CT Segmentation with Uncertainty Estimation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出一种联邦学习方法FIVA，解决跨异构CT数据分割隐私问题，通过模型与预测不确定性聚合提升分割质量，采用逆方差加权和不确定性传播实现有效协作。**

- **链接: [http://arxiv.org/pdf/2508.09196v1](http://arxiv.org/pdf/2508.09196v1)**

> **作者:** Asim Ukaye; Numan Saeed; Karthik Nandakumar
>
> **备注:** 17 pages, 5 figures, Machine Learning for Healthcare Conference
>
> **摘要:** Different CT segmentation datasets are typically obtained from different scanners under different capture settings and often provide segmentation labels for a limited and often disjoint set of organs. Using these heterogeneous data effectively while preserving patient privacy can be challenging. This work presents a novel federated learning approach to achieve universal segmentation across diverse abdominal CT datasets by utilizing model uncertainty for aggregation and predictive uncertainty for inference. Our approach leverages the inherent noise in stochastic mini-batch gradient descent to estimate a distribution over the model weights to provide an on-the-go uncertainty over the model parameters at the client level. The parameters are then aggregated at the server using the additional uncertainty information using a Bayesian-inspired inverse-variance aggregation scheme. Furthermore, the proposed method quantifies prediction uncertainty by propagating the uncertainty from the model weights, providing confidence measures essential for clinical decision-making. In line with recent work shown, predictive uncertainty is utilized in the inference stage to improve predictive performance. Experimental evaluations demonstrate the effectiveness of this approach in improving both the quality of federated aggregation and uncertainty-weighted inference compared to previously established baselines. The code for this work is made available at: https://github.com/asimukaye/fiva
>
---
#### [new 149] MoQE: Improve Quantization Model performance via Mixture of Quantization Experts
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 论文提出基于MoE架构的MoQE框架，通过多专家动态路由提升量化模型性能，解决量化精度下降问题，实验表明效果接近SOTA。**

- **链接: [http://arxiv.org/pdf/2508.09204v1](http://arxiv.org/pdf/2508.09204v1)**

> **作者:** Jinhao Zhang; Yunquan Zhang; Boyang Zhang; Zeyu Liu; Daning Cheng
>
> **摘要:** Quantization method plays a crucial role in improving model efficiency and reducing deployment costs, enabling the widespread application of deep learning models on resource-constrained devices. However, the quantization process inevitably introduces accuracy degradation. In this paper, we propose Mixture of Quantization Experts( abbr. MoQE), a quantization inference framework based on the Mixture-of-Experts (MoE) architecture, aiming to jointly improve the performance of quantization models. MoQE combines multiple quantization variants of one full-precision model as specialized "quantization experts" and dynamically routes input data to the most suitable expert based on its characteristics. MoQE alleviates the performance degradation commonly seen in single quantization models through specialization quantization expert models. We design lightweight, structure-aware router models tailored for both CV and NLP tasks. Experimental evaluations on ResNet, LLaMA, and Qwen model families across benchmark datasets including ImageNet, WikiText, C4, and OpenWebText demonstrate that MoQE achieves performance comparable to SOTA quantization model, without incurring significant increases in inference latency.
>
---
## 更新

#### [replaced 001] BigTokDetect: A Clinically-Informed Vision-Language Modeling Framework for Detecting Pro-Bigorexia Videos on TikTok
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06515v2](http://arxiv.org/pdf/2508.06515v2)**

> **作者:** Minh Duc Chu; Kshitij Pawar; Zihao He; Roxanna Sharifi; Ross Sonnenblick; Magdalayna Curry; Laura D'Adamo; Lindsay Young; Stuart B Murray; Kristina Lerman
>
> **摘要:** Social media platforms increasingly struggle to detect harmful content that promotes muscle dysmorphic behaviors, particularly pro-bigorexia content that disproportionately affects adolescent males. Unlike traditional eating disorder detection focused on the "thin ideal," pro-bigorexia material masquerades as legitimate fitness content through complex multimodal combinations of visual displays, coded language, and motivational messaging that evade text-based detection systems. We address this challenge by developing BigTokDetect, a clinically-informed detection framework for identifying pro-bigorexia content on TikTok. We introduce BigTok, the first expert-annotated multimodal dataset of over 2,200 TikTok videos labeled by clinical psychologists and psychiatrists across five primary categories spanning body image, nutrition, exercise, supplements, and masculinity. Through a comprehensive evaluation of state-of-the-art vision language models, we achieve 82.9% accuracy on primary category classification and 69.0% on subcategory detection via domain-specific finetuning. Our ablation studies demonstrate that multimodal fusion improves performance by 5-10% over text-only approaches, with video features providing the most discriminative signals. These findings establish new benchmarks for multimodal harmful content detection and provide both the computational tools and methodological framework needed for scalable content moderation in specialized mental health domains.
>
---
#### [replaced 002] Advancing Reliable Test-Time Adaptation of Vision-Language Models under Visual Variations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09500v2](http://arxiv.org/pdf/2507.09500v2)**

> **作者:** Yiwen Liang; Hui Chen; Yizhe Xiong; Zihan Zhou; Mengyao Lyu; Zijia Lin; Shuaicheng Niu; Sicheng Zhao; Jungong Han; Guiguang Ding
>
> **备注:** Accepted at the 33rd ACM International Conference on Multimedia(ACM MM 2025)
>
> **摘要:** Vision-language models (VLMs) exhibit remarkable zero-shot capabilities but struggle with distribution shifts in downstream tasks when labeled data is unavailable, which has motivated the development of Test-Time Adaptation (TTA) to improve VLMs' performance during inference without annotations. Among various TTA approaches, cache-based methods show promise by preserving historical knowledge from low-entropy samples in a dynamic cache and fostering efficient adaptation. However, these methods face two critical reliability challenges: (1) entropy often becomes unreliable under distribution shifts, causing error accumulation in the cache and degradation in adaptation performance; (2) the final predictions may be unreliable due to inflexible decision boundaries that fail to accommodate large downstream shifts. To address these challenges, we propose a Reliable Test-time Adaptation (ReTA) method that integrates two complementary strategies to enhance reliability from two perspectives. First, to mitigate the unreliability of entropy as a sample selection criterion for cache construction, we introduce Consistency-aware Entropy Reweighting (CER), which incorporates consistency constraints to weight entropy during cache updating. While conventional approaches rely solely on low entropy for cache prioritization and risk introducing noise, our method leverages predictive consistency to maintain a high-quality cache and facilitate more robust adaptation. Second, we present Diversity-driven Distribution Calibration (DDC), which models class-wise text embeddings as multivariate Gaussian distributions, enabling adaptive decision boundaries for more accurate predictions across visually diverse content. Extensive experiments demonstrate that ReTA consistently outperforms state-of-the-art methods, particularly under real-world distribution shifts. Code: https://github.com/Evelyn1ywliang/ReTA.
>
---
#### [replaced 003] GLM-4.1V-Thinking and GLM-4.5V: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01006v3](http://arxiv.org/pdf/2507.01006v3)**

> **作者:** GLM-V Team; :; Wenyi Hong; Wenmeng Yu; Xiaotao Gu; Guo Wang; Guobing Gan; Haomiao Tang; Jiale Cheng; Ji Qi; Junhui Ji; Lihang Pan; Shuaiqi Duan; Weihan Wang; Yan Wang; Yean Cheng; Zehai He; Zhe Su; Zhen Yang; Ziyang Pan; Aohan Zeng; Baoxu Wang; Bin Chen; Boyan Shi; Changyu Pang; Chenhui Zhang; Da Yin; Fan Yang; Guoqing Chen; Jiazheng Xu; Jiale Zhu; Jiali Chen; Jing Chen; Jinhao Chen; Jinghao Lin; Jinjiang Wang; Junjie Chen; Leqi Lei; Letian Gong; Leyi Pan; Mingdao Liu; Mingzhi Zhang; Qinkai Zheng; Sheng Yang; Shi Zhong; Shiyu Huang; Shuyuan Zhao; Siyan Xue; Shangqin Tu; Shengbiao Meng; Tianshu Zhang; Tianwei Luo; Tianxiang Hao; Wenkai Li; Wei Jia; Xiao Liu; Xiaohan Zhang; Xin Lyu; Xuancheng Huang; Yanling Wang; Yadong Xue; Yanfeng Wang; Yanzi Wang; Yifan An; Yifan Du; Yiming Shi; Yiheng Huang; Yilin Niu; Yuan Wang; Yuanchang Yue; Yuchen Li; Yutao Zhang; Yuting Wang; Yu Wang; Yuxuan Zhang; Zhanxiao Du; Zhenyu Hou; Zhao Xue; Zhengxiao Du; Zihan Wang; Peng Zhang; Debing Liu; Bin Xu; Juanzi Li; Minlie Huang; Yuxiao Dong; Jie Tang
>
> **摘要:** We present GLM-4.1V-Thinking and GLM-4.5V, a family of vision-language models (VLMs) designed to advance general-purpose multimodal understanding and reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. We then propose Reinforcement Learning with Curriculum Sampling (RLCS) to unlock the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document interpretation. In a comprehensive evaluation across 42 public benchmarks, GLM-4.5V achieves state-of-the-art performance on nearly all tasks among open-source models of similar size, and demonstrates competitive or even superior results compared to closed-source models such as Gemini-2.5-Flash on challenging tasks including Coding and GUI Agents. Meanwhile, the smaller GLM-4.1V-9B-Thinking remains highly competitive-achieving superior results to the much larger Qwen2.5-VL-72B on 29 benchmarks. We open-source both GLM-4.1V-9B-Thinking and GLM-4.5V. Code, models and more information are released at https://github.com/zai-org/GLM-V.
>
---
#### [replaced 004] Towards Black-Box Membership Inference Attack for Diffusion Models
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.20771v5](http://arxiv.org/pdf/2405.20771v5)**

> **作者:** Jingwei Li; Jing Dong; Tianxing He; Jingzhao Zhang
>
> **摘要:** Given the rising popularity of AI-generated art and the associated copyright concerns, identifying whether an artwork was used to train a diffusion model is an important research topic. The work approaches this problem from the membership inference attack (MIA) perspective. We first identify the limitation of applying existing MIA methods for proprietary diffusion models: the required access of internal U-nets. To address the above problem, we introduce a novel membership inference attack method that uses only the image-to-image variation API and operates without access to the model's internal U-net. Our method is based on the intuition that the model can more easily obtain an unbiased noise prediction estimate for images from the training set. By applying the API multiple times to the target image, averaging the outputs, and comparing the result to the original image, our approach can classify whether a sample was part of the training set. We validate our method using DDIM and Stable Diffusion setups and further extend both our approach and existing algorithms to the Diffusion Transformer architecture. Our experimental results consistently outperform previous methods.
>
---
#### [replaced 005] Revisiting 3D Medical Scribble Supervision: Benchmarking Beyond Cardiac Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.12834v2](http://arxiv.org/pdf/2403.12834v2)**

> **作者:** Karol Gotkowski; Klaus H. Maier-Hein; Fabian Isensee
>
> **备注:** accepted at MICCAI2025
>
> **摘要:** Scribble supervision has emerged as a promising approach for reducing annotation costs in medical 3D segmentation by leveraging sparse annotations instead of voxel-wise labels. While existing methods report strong performance, a closer analysis reveals that the majority of research is confined to the cardiac domain, predominantly using ACDC and MSCMR datasets. This over-specialization has resulted in severe overfitting, misleading claims of performance improvements, and a lack of generalization across broader segmentation tasks. In this work, we formulate a set of key requirements for practical scribble supervision and introduce ScribbleBench, a comprehensive benchmark spanning over seven diverse medical imaging datasets, to systematically evaluate the fulfillment of these requirements. Consequently, we uncover a general failure of methods to generalize across tasks and that many widely used novelties degrade performance outside of the cardiac domain, whereas simpler overlooked approaches achieve superior generalization. Finally, we raise awareness for a strong yet overlooked baseline, nnU-Net coupled with a partial loss, which consistently outperforms specialized methods across a diverse range of tasks. By identifying fundamental limitations in existing research and establishing a new benchmark-driven evaluation standard, this work aims to steer scribble supervision toward more practical, robust, and generalizable methodologies for medical image segmentation.
>
---
#### [replaced 006] PrAViC: Probabilistic Adaptation Framework for Real-Time Video Classification
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.11443v2](http://arxiv.org/pdf/2406.11443v2)**

> **作者:** Magdalena Trędowicz; Marcin Mazur; Szymon Janusz; Arkadiusz Lewicki; Jacek Tabor; Łukasz Struski
>
> **备注:** The paper was accepted at ECAI 2025
>
> **摘要:** Video processing is generally divided into two main categories: processing of the entire video, which typically yields optimal classification outcomes, and real-time processing, where the objective is to make a decision as promptly as possible. Although the models dedicated to the processing of entire videos are typically well-defined and clearly presented in the literature, this is not the case for online processing, where a~plethora of hand-devised methods exist. To address this issue, we present PrAViC, a novel, unified, and theoretically-based adaptation framework for tackling the online classification problem in video data. The initial phase of our study is to establish a mathematical background for the classification of sequential data, with the potential to make a decision at an early stage. This allows us to construct a natural function that encourages the model to return a result much faster. The subsequent phase is to present a straightforward and readily implementable method for adapting offline models to the online setting using recurrent operations. Finally, PrAViC is evaluated by comparing it with existing state-of-the-art offline and online models and datasets. This enables the network to significantly reduce the time required to reach classification decisions while maintaining, or even enhancing, accuracy.
>
---
#### [replaced 007] CAS-IQA: Teaching Vision-Language Models for Synthetic Angiography Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17619v2](http://arxiv.org/pdf/2505.17619v2)**

> **作者:** Bo Wang; De-Xing Huang; Xiao-Hu Zhou; Mei-Jiang Gui; Nu-Fang Xiao; Jian-Long Hao; Ming-Yuan Liu; Zeng-Guang Hou
>
> **备注:** Camera ready version for ICONIP 2025
>
> **摘要:** Synthetic X-ray angiographies generated by modern generative models hold great potential to reduce the use of contrast agents in vascular interventional procedures. However, low-quality synthetic angiographies can significantly increase procedural risk, underscoring the need for reliable image quality assessment (IQA) methods. Existing IQA models, however, fail to leverage auxiliary images as references during evaluation and lack fine-grained, task-specific metrics necessary for clinical relevance. To address these limitations, this paper proposes CAS-IQA, a vision-language model (VLM)-based framework that predicts fine-grained quality scores by effectively incorporating auxiliary information from related images. In the absence of angiography datasets, CAS-3K is constructed, comprising 3,565 synthetic angiographies along with score annotations. To ensure clinically meaningful assessment, three task-specific evaluation metrics are defined. Furthermore, a Multi-path featUre fuSion and rouTing (MUST) module is designed to enhance image representations by adaptively fusing and routing visual tokens to metric-specific branches. Extensive experiments on the CAS-3K dataset demonstrate that CAS-IQA significantly outperforms state-of-the-art IQA methods by a considerable margin.
>
---
#### [replaced 008] ProbRadarM3F: mmWave Radar based Human Skeletal Pose Estimation with Probability Map Guided Multi-Format Feature Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.05164v5](http://arxiv.org/pdf/2405.05164v5)**

> **作者:** Bing Zhu; Zixin He; Weiyi Xiong; Guanhua Ding; Tao Huang; Wei Xiang
>
> **摘要:** Millimeter wave (mmWave) radar is a non-intrusive privacy and relatively convenient and inexpensive device, which has been demonstrated to be applicable in place of RGB cameras in human indoor pose estimation tasks. However, mmWave radar relies on the collection of reflected signals from the target, and the radar signals containing information is difficult to be fully applied. This has been a long-standing hindrance to the improvement of pose estimation accuracy. To address this major challenge, this paper introduces a probability map guided multi-format feature fusion model, ProbRadarM3F. This is a novel radar feature extraction framework using a traditional FFT method in parallel with a probability map based positional encoding method. ProbRadarM3F fuses the traditional heatmap features and the positional features, then effectively achieves the estimation of 14 keypoints of the human body. Experimental evaluation on the HuPR dataset proves the effectiveness of the model proposed in this paper, outperforming other methods experimented on this dataset with an AP of 69.9 %. The emphasis of our study is focusing on the position information that is not exploited before in radar singal. This provides direction to investigate other potential non-redundant information from mmWave rader.
>
---
#### [replaced 009] OC-SOP: Enhancing Vision-Based 3D Semantic Occupancy Prediction by Object-Centric Awareness
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18798v2](http://arxiv.org/pdf/2506.18798v2)**

> **作者:** Helin Cao; Sven Behnke
>
> **备注:** 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Vienna, Austria, Oct 2025
>
> **摘要:** Autonomous driving perception faces significant challenges due to occlusions and incomplete scene data in the environment. To overcome these issues, the task of semantic occupancy prediction (SOP) is proposed, which aims to jointly infer both the geometry and semantic labels of a scene from images. However, conventional camera-based methods typically treat all categories equally and primarily rely on local features, leading to suboptimal predictions, especially for dynamic foreground objects. To address this, we propose Object-Centric SOP (OC-SOP), a framework that integrates high-level object-centric cues extracted via a detection branch into the semantic occupancy prediction pipeline. This object-centric integration significantly enhances the prediction accuracy for foreground objects and achieves state-of-the-art performance among all categories on SemanticKITTI.
>
---
#### [replaced 010] Simulating the Real World: A Unified Survey of Multimodal Generative Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04641v2](http://arxiv.org/pdf/2503.04641v2)**

> **作者:** Yuqi Hu; Longguang Wang; Xian Liu; Ling-Hao Chen; Yuwei Guo; Yukai Shi; Ce Liu; Anyi Rao; Zeyu Wang; Hui Xiong
>
> **备注:** Repository for the related papers at https://github.com/ALEEEHU/World-Simulator
>
> **摘要:** Understanding and replicating the real world is a critical challenge in Artificial General Intelligence (AGI) research. To achieve this, many existing approaches, such as world models, aim to capture the fundamental principles governing the physical world, enabling more accurate simulations and meaningful interactions. However, current methods often treat different modalities, including 2D (images), videos, 3D, and 4D representations, as independent domains, overlooking their interdependencies. Additionally, these methods typically focus on isolated dimensions of reality without systematically integrating their connections. In this survey, we present a unified survey for multimodal generative models that investigate the progression of data dimensionality in real-world simulation. Specifically, this survey starts from 2D generation (appearance), then moves to video (appearance+dynamics) and 3D generation (appearance+geometry), and finally culminates in 4D generation that integrate all dimensions. To the best of our knowledge, this is the first attempt to systematically unify the study of 2D, video, 3D and 4D generation within a single framework. To guide future research, we provide a comprehensive review of datasets, evaluation metrics and future directions, and fostering insights for newcomers. This survey serves as a bridge to advance the study of multimodal generative models and real-world simulation within a unified framework.
>
---
#### [replaced 011] Cyc3D: Fine-grained Controllable 3D Generation via Cycle Consistency Regularization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14975v2](http://arxiv.org/pdf/2504.14975v2)**

> **作者:** Hongbin Xu; Chaohui Yu; Feng Xiao; Jiazheng Xing; Hai Ci; Weitao Chen; Fan Wang; Ming Li
>
> **备注:** Preprint version. Update with new experimental results
>
> **摘要:** Despite the remarkable progress of 3D generation, achieving controllability, i.e., ensuring consistency between generated 3D content and input conditions like edge and depth, remains a significant challenge. Existing methods often struggle to maintain accurate alignment, leading to noticeable discrepancies. To address this issue, we propose \name{}, a new framework that enhances controllable 3D generation by explicitly encouraging cyclic consistency between the second-order 3D content, generated based on extracted signals from the first-order generation, and its original input controls. Specifically, we employ an efficient feed-forward backbone that can generate a 3D object from an input condition and a text prompt. Given an initial viewpoint and a control signal, a novel view is rendered from the generated 3D content, from which the extracted condition is used to regenerate the 3D content. This re-generated output is then rendered back to the initial viewpoint, followed by another round of control signal extraction, forming a cyclic process with two consistency constraints. \emph{View consistency} ensures coherence between the two generated 3D objects, measured by semantic similarity to accommodate generative diversity. \emph{Condition consistency} aligns the final extracted signal with the original input control, preserving structural or geometric details throughout the process. Extensive experiments on popular benchmarks demonstrate that \name{} significantly improves controllability, especially for fine-grained details, outperforming existing methods across various conditions (e.g., +14.17\% PSNR for edge, +6.26\% PSNR for sketch).
>
---
#### [replaced 012] Ear-Keeper: A Cross-Platform AI System for Rapid and Accurate Ear Disease Diagnosis
- **分类: cs.CV; cs.SE**

- **链接: [http://arxiv.org/pdf/2308.10610v5](http://arxiv.org/pdf/2308.10610v5)**

> **作者:** Feiyan Lu; Yubiao Yue; Zhenzhang Li; Meiping Zhang; Wen Luo; Fan Zhang; Tong Liu; Jingyong Shi; Guang Wang; Xinyu Zeng
>
> **备注:** 18 pages,8 figures
>
> **摘要:** Early and accurate detection systems for ear diseases, powered by deep learning, are essential for preventing hearing impairment and improving population health. However, the limited diversity of existing otoendoscopy datasets and the poor balance between diagnostic accuracy, computational efficiency, and model size have hindered the translation of artificial intelligence (AI) algorithms into healthcare applications. In this study, we constructed a large-scale, multi-center otoendoscopy dataset covering eight common ear diseases and healthy cases. Building upon this resource, we developed Best-EarNet, an ultrafast and lightweight deep learning architecture integrating a novel Local-Global Spatial Feature Fusion Module with a multi-scale supervision strategy, enabling real-time and accurate classification of ear conditions. Leveraging transfer learning, Best-EarNet, with a model size of only 2.94 MB, achieved diagnostic accuracies of 95.23% on an internal test set (22,581 images) and 92.14% on an external test set (1,652 images), while requiring only 0.0125 seconds (80 frames per second) to process a single image on a standard CPU. Further subgroup analysis by gender and age showed consistently excellent performance of Best-EarNet across all demographic groups. To enhance clinical interpretability and user trust, we incorporated Grad-CAM-based visualization, highlighting the specific abnormal ear regions contributing to AI predictions. Most importantly, we developed Ear-Keeper, a cross-platform intelligent diagnosis system built upon Best-EarNet, deployable on smartphones, tablets, and personal computers. Ear-Keeper enables public users and healthcare providers to perform comprehensive real-time video-based ear canal screening, supporting early detection and timely intervention of ear diseases.
>
---
#### [replaced 013] LM-MCVT: A Lightweight Multi-modal Multi-view Convolutional-Vision Transformer Approach for 3D Object Recognition
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19256v3](http://arxiv.org/pdf/2504.19256v3)**

> **作者:** Songsong Xiong; Hamidreza Kasaei
>
> **摘要:** In human-centered environments such as restaurants, homes, and warehouses, robots often face challenges in accurately recognizing 3D objects. These challenges stem from the complexity and variability of these environments, including diverse object shapes. In this paper, we propose a novel Lightweight Multi-modal Multi-view Convolutional-Vision Transformer network (LM-MCVT) to enhance 3D object recognition in robotic applications. Our approach leverages the Globally Entropy-based Embeddings Fusion (GEEF) method to integrate multi-views efficiently. The LM-MCVT architecture incorporates pre- and mid-level convolutional encoders and local and global transformers to enhance feature extraction and recognition accuracy. We evaluate our method on the synthetic ModelNet40 dataset and achieve a recognition accuracy of 95.6% using a four-view setup, surpassing existing state-of-the-art methods. To further validate its effectiveness, we conduct 5-fold cross-validation on the real-world OmniObject3D dataset using the same configuration. Results consistently show superior performance, demonstrating the method's robustness in 3D object recognition across synthetic and real-world 3D data.
>
---
#### [replaced 014] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06476v3](http://arxiv.org/pdf/2502.06476v3)**

> **作者:** Vlad Hosu; Lorenzo Agnolucci; Daisuke Iso; Dietmar Saupe
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** Image Quality Assessment (IQA) measures and predicts perceived image quality by human observers. Although recent studies have highlighted the critical influence that variations in the scale of an image have on its perceived quality, this relationship has not been systematically quantified. To bridge this gap, we introduce the Image Intrinsic Scale (IIS), defined as the largest scale where an image exhibits its highest perceived quality. We also present the Image Intrinsic Scale Assessment (IISA) task, which involves subjectively measuring and predicting the IIS based on human judgments. We develop a subjective annotation methodology and create the IISA-DB dataset, comprising 785 image-IIS pairs annotated by experts in a rigorously controlled crowdsourcing study. Furthermore, we propose WIISA (Weak-labeling for Image Intrinsic Scale Assessment), a strategy that leverages how the IIS of an image varies with downscaling to generate weak labels. Experiments show that applying WIISA during the training of several IQA methods adapted for IISA consistently improves the performance compared to using only ground-truth labels. The code, dataset, and pre-trained models are available at https://github.com/SonyResearch/IISA.
>
---
#### [replaced 015] MIND: A Noise-Adaptive Denoising Framework for Medical Images Integrating Multi-Scale Transformer
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.07817v2](http://arxiv.org/pdf/2508.07817v2)**

> **作者:** Tao Tang; Chengxu Yang
>
> **备注:** Accepted by the 7th International Conference on Intelligent Control, Measurement and Signal Processing (ICMSP 2025). 6 pages, 6 figures
>
> **摘要:** The core role of medical images in disease diagnosis makes their quality directly affect the accuracy of clinical judgment. However, due to factors such as low-dose scanning, equipment limitations and imaging artifacts, medical images are often accompanied by non-uniform noise interference, which seriously affects structure recognition and lesion detection. This paper proposes a medical image adaptive denoising model (MI-ND) that integrates multi-scale convolutional and Transformer architecture, introduces a noise level estimator (NLE) and a noise adaptive attention module (NAAB), and realizes channel-spatial attention regulation and cross-modal feature fusion driven by noise perception. Systematic testing is carried out on multimodal public datasets. Experiments show that this method significantly outperforms the comparative methods in image quality indicators such as PSNR, SSIM, and LPIPS, and improves the F1 score and ROC-AUC in downstream diagnostic tasks, showing strong prac-tical value and promotional potential. The model has outstanding benefits in structural recovery, diagnostic sensitivity, and cross-modal robustness, and provides an effective solution for medical image enhancement and AI-assisted diagnosis and treatment.
>
---
#### [replaced 016] Learning Adaptive Node Selection with External Attention for Human Interaction Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03936v2](http://arxiv.org/pdf/2507.03936v2)**

> **作者:** Chen Pang; Xuequan Lu; Qianyu Zhou; Lei Lyu
>
> **备注:** Accepted by ACM MM25
>
> **摘要:** Most GCN-based methods model interacting individuals as independent graphs, neglecting their inherent inter-dependencies. Although recent approaches utilize predefined interaction adjacency matrices to integrate participants, these matrices fail to adaptively capture the dynamic and context-specific joint interactions across different actions. In this paper, we propose the Active Node Selection with External Attention Network (ASEA), an innovative approach that dynamically captures interaction relationships without predefined assumptions. Our method models each participant individually using a GCN to capture intra-personal relationships, facilitating a detailed representation of their actions. To identify the most relevant nodes for interaction modeling, we introduce the Adaptive Temporal Node Amplitude Calculation (AT-NAC) module, which estimates global node activity by combining spatial motion magnitude with adaptive temporal weighting, thereby highlighting salient motion patterns while reducing irrelevant or redundant information. A learnable threshold, regularized to prevent extreme variations, is defined to selectively identify the most informative nodes for interaction modeling. To capture interactions, we design the External Attention (EA) module to operate on active nodes, effectively modeling the interaction dynamics and semantic relationships between individuals. Extensive evaluations show that our method captures interaction relationships more effectively and flexibly, achieving state-of-the-art performance.
>
---
#### [replaced 017] MGDFIS: Multi-scale Global-detail Feature Integration Strategy for Small Object Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.12697v2](http://arxiv.org/pdf/2506.12697v2)**

> **作者:** Yuxiang Wang; Xuecheng Bai; Boyu Hu; Chuanzhi Xu; Haodong Chen; Vera Chung; Tingxue Li; Xiaoming Chen
>
> **备注:** 9 pages, 5 figures, 3 tables
>
> **摘要:** Small object detection in UAV imagery is crucial for applications such as search-and-rescue, traffic monitoring, and environmental surveillance, but it is hampered by tiny object size, low signal-to-noise ratios, and limited feature extraction. Existing multi-scale fusion methods help, but add computational burden and blur fine details, making small object detection in cluttered scenes difficult. To overcome these challenges, we propose the Multi-scale Global-detail Feature Integration Strategy (MGDFIS), a unified fusion framework that tightly couples global context with local detail to boost detection performance while maintaining efficiency. MGDFIS comprises three synergistic modules: the FusionLock-TSS Attention Module, which marries token-statistics self-attention with DynamicTanh normalization to highlight spectral and spatial cues at minimal cost; the Global-detail Integration Module, which fuses multi-scale context via directional convolution and parallel attention while preserving subtle shape and texture variations; and the Dynamic Pixel Attention Module, which generates pixel-wise weighting maps to rebalance uneven foreground and background distributions and sharpen responses to true object regions. Extensive experiments on the VisDrone benchmark demonstrate that MGDFIS consistently outperforms state-of-the-art methods across diverse backbone architectures and detection frameworks, achieving superior precision and recall with low inference time. By striking an optimal balance between accuracy and resource usage, MGDFIS provides a practical solution for small-object detection on resource-constrained UAV platforms.
>
---
#### [replaced 018] LiteFat: Lightweight Spatio-Temporal Graph Learning for Real-Time Driver Fatigue Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.21756v2](http://arxiv.org/pdf/2507.21756v2)**

> **作者:** Jing Ren; Suyu Ma; Hong Jia; Xiwei Xu; Ivan Lee; Haytham Fayek; Xiaodong Li; Feng Xia
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Detecting driver fatigue is critical for road safety, as drowsy driving remains a leading cause of traffic accidents. Many existing solutions rely on computationally demanding deep learning models, which result in high latency and are unsuitable for embedded robotic devices with limited resources (such as intelligent vehicles/cars) where rapid detection is necessary to prevent accidents. This paper introduces LiteFat, a lightweight spatio-temporal graph learning model designed to detect driver fatigue efficiently while maintaining high accuracy and low computational demands. LiteFat involves converting streaming video data into spatio-temporal graphs (STG) using facial landmark detection, which focuses on key motion patterns and reduces unnecessary data processing. LiteFat uses MobileNet to extract facial features and create a feature matrix for the STG. A lightweight spatio-temporal graph neural network is then employed to identify signs of fatigue with minimal processing and low latency. Experimental results on benchmark datasets show that LiteFat performs competitively while significantly decreasing computational complexity and latency as compared to current state-of-the-art methods. This work enables the development of real-time, resource-efficient human fatigue detection systems that can be implemented upon embedded robotic devices.
>
---
#### [replaced 019] Towards Synthesized and Editable Motion In-Betweening Through Part-Wise Phase Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08180v3](http://arxiv.org/pdf/2503.08180v3)**

> **作者:** Minyue Dai; Ke Fan; Bin Ji; Haoran Xu; Haoyu Zhao; Junting Dong; Jingbo Wang; Bo Dai
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Styled motion in-betweening is crucial for computer animation and gaming. However, existing methods typically encode motion styles by modeling whole-body motions, often overlooking the representation of individual body parts. This limitation reduces the flexibility of infilled motion, particularly in adjusting the motion styles of specific limbs independently. To overcome this challenge, we propose a novel framework that models motion styles at the body-part level, enhancing both the diversity and controllability of infilled motions. Our approach enables more nuanced and expressive animations by allowing precise modifications to individual limb motions while maintaining overall motion coherence. Leveraging phase-related insights, our framework employs periodic autoencoders to automatically extract the phase of each body part, capturing distinctive local style features. Additionally, we effectively decouple the motion source from synthesis control by integrating motion manifold learning and conditional generation techniques from both image and motion domains. This allows the motion source to generate high-quality motions across various styles, with extracted motion and style features readily available for controlled synthesis in subsequent tasks. Comprehensive evaluations demonstrate that our method achieves superior speed, robust generalization, and effective generation of extended motion sequences.
>
---
#### [replaced 020] A Simple yet Powerful Instance-Aware Prompting Framework for Training-free Camouflaged Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06904v2](http://arxiv.org/pdf/2508.06904v2)**

> **作者:** Chao Yin; Jide Li; Xiaoqiang Li
>
> **备注:** under review
>
> **摘要:** Camouflaged Object Segmentation (COS) remains highly challenging due to the intrinsic visual similarity between target objects and their surroundings. While training-based COS methods achieve good performance, their performance degrades rapidly with increased annotation sparsity. To circumvent this limitation, recent studies have explored training-free COS methods, leveraging the Segment Anything Model (SAM) by automatically generating visual prompts from a single task-generic prompt (\textit{e.g.}, "\textit{camouflaged animal}") uniformly applied across all test images. However, these methods typically produce only semantic-level visual prompts, causing SAM to output coarse semantic masks and thus failing to handle scenarios with multiple discrete camouflaged instances effectively. To address this critical limitation, we propose a simple yet powerful \textbf{I}nstance-\textbf{A}ware \textbf{P}rompting \textbf{F}ramework (IAPF), the first training-free COS pipeline that explicitly converts a task-generic prompt into fine-grained instance masks. Specifically, the IAPF comprises three steps: (1) Text Prompt Generator, utilizing task-generic queries to prompt a Multimodal Large Language Model (MLLM) for generating image-specific foreground and background tags; (2) \textbf{Instance Mask Generator}, leveraging Grounding DINO to produce precise instance-level bounding box prompts, alongside the proposed Single-Foreground Multi-Background Prompting strategy to sample region-constrained point prompts within each box, enabling SAM to yield a candidate instance mask; (3) Self-consistency Instance Mask Voting, which selects the final COS prediction by identifying the candidate mask most consistent across multiple candidate instance masks. Extensive evaluations on standard COS benchmarks demonstrate that the proposed IAPF significantly surpasses existing state-of-the-art training-free COS methods.
>
---
#### [replaced 021] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01367v2](http://arxiv.org/pdf/2507.01367v2)**

> **作者:** Tianrui Lou; Xiaojun Jia; Siyuan Liang; Jiawei Liang; Ming Zhang; Yanjun Xiao; Xiaochun Cao
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.
>
---
#### [replaced 022] Depth-Guided Self-Supervised Human Keypoint Detection via Cross-Modal Distillation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.14700v2](http://arxiv.org/pdf/2410.14700v2)**

> **作者:** Aman Anand; Elyas Rashno; Amir Eskandari; Farhana Zulkernine
>
> **摘要:** Existing unsupervised keypoint detection methods apply artificial deformations to images such as masking a significant portion of images and using reconstruction of original image as a learning objective to detect keypoints. However, this approach lacks depth information in the image and often detects keypoints on the background. To address this, we propose Distill-DKP, a novel cross-modal knowledge distillation framework that leverages depth maps and RGB images for keypoint detection in a self-supervised setting. During training, Distill-DKP extracts embedding-level knowledge from a depth-based teacher model to guide an image-based student model with inference restricted to the student. Experiments show that Distill-DKP significantly outperforms previous unsupervised methods by reducing mean L2 error by 47.15% on Human3.6M, mean average error by 5.67% on Taichi, and improving keypoints accuracy by 1.3% on DeepFashion dataset. Detailed ablation studies demonstrate the sensitivity of knowledge distillation across different layers of the network. Project Page: https://23wm13.github.io/distill-dkp/
>
---
#### [replaced 023] Training-Free Text-Guided Color Editing with Multi-Modal Diffusion Transformer
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09131v2](http://arxiv.org/pdf/2508.09131v2)**

> **作者:** Zixin Yin; Xili Dai; Ling-Hao Chen; Deyu Zhou; Jianan Wang; Duomin Wang; Gang Yu; Lionel M. Ni; Lei Zhang; Heung-Yeung Shum
>
> **摘要:** Text-guided color editing in images and videos is a fundamental yet unsolved problem, requiring fine-grained manipulation of color attributes, including albedo, light source color, and ambient lighting, while preserving physical consistency in geometry, material properties, and light-matter interactions. Existing training-free methods offer broad applicability across editing tasks but struggle with precise color control and often introduce visual inconsistency in both edited and non-edited regions. In this work, we present ColorCtrl, a training-free color editing method that leverages the attention mechanisms of modern Multi-Modal Diffusion Transformers (MM-DiT). By disentangling structure and color through targeted manipulation of attention maps and value tokens, our method enables accurate and consistent color editing, along with word-level control of attribute intensity. Our method modifies only the intended regions specified by the prompt, leaving unrelated areas untouched. Extensive experiments on both SD3 and FLUX.1-dev demonstrate that ColorCtrl outperforms existing training-free approaches and achieves state-of-the-art performances in both edit quality and consistency. Furthermore, our method surpasses strong commercial models such as FLUX.1 Kontext Max and GPT-4o Image Generation in terms of consistency. When extended to video models like CogVideoX, our approach exhibits greater advantages, particularly in maintaining temporal coherence and editing stability. Finally, our method also generalizes to instruction-based editing diffusion models such as Step1X-Edit and FLUX.1 Kontext dev, further demonstrating its versatility.
>
---
#### [replaced 024] HVL: Semi-Supervised Segmentation leveraging Hierarchical Vision-Language Synergy with Dynamic Text-Spatial Query Alignment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13925v2](http://arxiv.org/pdf/2506.13925v2)**

> **作者:** Numair Nadeem; Saeed Anwar; Muhammad Hamza Asad; Abdul Bais
>
> **摘要:** In this paper, we address Semi-supervised Semantic Segmentation (SSS) under domain shift by leveraging domain-invariant semantic knowledge from text embeddings of Vision-Language Models (VLMs). We propose a unified Hierarchical Vision-Language framework (HVL) that integrates domain-invariant text embeddings as object queries in a transformer-based segmentation network to improve generalization and reduce misclassification under limited supervision. The mentioned textual queries are used for grouping pixels with shared semantics under SSS. HVL is designed to (1) generate textual queries that maximally encode domain-invariant semantics from VLM while capturing intra-class variations; (2) align these queries with spatial visual features to enhance their segmentation ability and improve the semantic clarity of visual features. We also introduce targeted regularization losses that maintain vision--language alignment throughout training to reinforce semantic understanding. HVL establishes a novel state-of-the-art by achieving a +9.3% improvement in mean Intersection over Union (mIoU) on COCO, utilizing 232 labelled images, +3.1% on Pascal VOC employing 92 labels, +4.8% on ADE20 using 316 labels, and +3.4% on Cityscapes with 100 labels, demonstrating superior performance with less than 1% supervision on four benchmark datasets. Our results show that language-guided segmentation bridges the label efficiency gap and enables new levels of fine-grained generalization.
>
---
#### [replaced 025] RAGAR: Retrieval Augmented Personalized Image Generation Guided by Recommendation
- **分类: cs.IR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01657v2](http://arxiv.org/pdf/2505.01657v2)**

> **作者:** Run Ling; Wenji Wang; Yuting Liu; Guibing Guo; Haowei Liu; Jian Lu; Quanwei Zhang; Yexing Xu; Shuo Lu; Yun Wang; Yihua Shao; Zhanjie Zhang; Ao Ma; Linying Jiang; Xingwei Wang
>
> **摘要:** Personalized image generation is crucial for improving the user experience, as it renders reference images into preferred ones according to user visual preferences. Although effective, existing methods face two main issues. First, existing methods treat all items in the user historical sequence equally when extracting user preferences, overlooking the varying semantic similarities between historical items and the reference item. Disproportionately high weights for low-similarity items distort users' visual preferences for the reference item. Second, existing methods heavily rely on consistency between generated and reference images to optimize the generation, which leads to underfitting user preferences and hinders personalization. To address these issues, we propose Retrieval Augment Personalized Image GenerAtion guided by Recommendation (RAGAR). Our approach uses a retrieval mechanism to assign different weights to historical items according to their similarities to the reference item, thereby extracting more refined users' visual preferences for the reference item. Then we introduce a novel rank task based on the multi-modal ranking model to optimize the personalization of the generated images instead of forcing depend on consistency. Extensive experiments and human evaluations on three real-world datasets demonstrate that RAGAR achieves significant improvements in both personalization and semantic metrics compared to five baselines.
>
---
#### [replaced 026] MoSE: Skill-by-Skill Mixture-of-Experts Learning for Embodied Autonomous Machines
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.07818v2](http://arxiv.org/pdf/2507.07818v2)**

> **作者:** Lu Xu; Jiaqian Yu; Xiongfeng Peng; Yiwei Chen; Weiming Li; Jaewook Yoo; Sunghyun Chunag; Dongwook Lee; Daehyun Ji; Chao Zhang
>
> **摘要:** To meet the growing demand for smarter, faster, and more efficient embodied AI solutions, we introduce a novel Mixture-of-Expert (MoE) method that significantly boosts reasoning and learning efficiency for embodied autonomous systems. General MoE models demand extensive training data and complex optimization, which limits their applicability in embodied AI such as autonomous driving (AD) and robotic manipulation. In this work, we propose a skill-oriented MoE called MoSE, which mimics the human learning and reasoning process skill-by-skill, step-by-step. We introduce a skill-oriented routing mechanism that begins with defining and annotating specific skills, enabling experts to identify the necessary competencies for various scenarios and reasoning tasks, thereby facilitating skill-by-skill learning. To better align with multi-step planning in human reasoning and in end-to-end driving models, we build a hierarchical skill dataset and pretrain the router to encourage the model to think step-by-step. Unlike other multi-round dialogues, MoSE integrates valuable auxiliary tasks (e.g. perception-prediction-planning for AD, and high-level and low-level planning for robots) in one single forward process without introducing any extra computational cost. With less than 3B sparsely activated parameters, our model effectively grows more diverse expertise and outperforms models on both AD corner-case reasoning tasks and robot reasoning tasks with less than 40% of the parameters.
>
---
#### [replaced 027] GenAI Confessions: Black-box Membership Inference for Generative Image Models
- **分类: cs.CV; cs.AI; cs.CR; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.06399v2](http://arxiv.org/pdf/2501.06399v2)**

> **作者:** Matyas Bohacek; Hany Farid
>
> **备注:** https://genai-confessions.github.io
>
> **摘要:** From a simple text prompt, generative-AI image models can create stunningly realistic and creative images bounded, it seems, by only our imagination. These models have achieved this remarkable feat thanks, in part, to the ingestion of billions of images collected from nearly every corner of the internet. Many creators have understandably expressed concern over how their intellectual property has been ingested without their permission or a mechanism to opt out of training. As a result, questions of fair use and copyright infringement have quickly emerged. We describe a method that allows us to determine if a model was trained on a specific image or set of images. This method is computationally efficient and assumes no explicit knowledge of the model architecture or weights (so-called black-box membership inference). We anticipate that this method will be crucial for auditing existing models and, looking ahead, ensuring the fairer development and deployment of generative AI models.
>
---
#### [replaced 028] Grounding Emotion Recognition with Visual Prototypes: VEGA -- Revisiting CLIP in MERC
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06564v2](http://arxiv.org/pdf/2508.06564v2)**

> **作者:** Guanyu Hu; Dimitrios Kollias; Xinyu Yang
>
> **备注:** accepted for publication at ACM Multimedia (ACM MM) 2025
>
> **摘要:** Multimodal Emotion Recognition in Conversations remains a challenging task due to the complex interplay of textual, acoustic and visual signals. While recent models have improved performance via advanced fusion strategies, they often lack psychologically meaningful priors to guide multimodal alignment. In this paper, we revisit the use of CLIP and propose a novel Visual Emotion Guided Anchoring (VEGA) mechanism that introduces class-level visual semantics into the fusion and classification process. Distinct from prior work that primarily utilizes CLIP's textual encoder, our approach leverages its image encoder to construct emotion-specific visual anchors based on facial exemplars. These anchors guide unimodal and multimodal features toward a perceptually grounded and psychologically aligned representation space, drawing inspiration from cognitive theories (prototypical emotion categories and multisensory integration). A stochastic anchor sampling strategy further enhances robustness by balancing semantic stability and intra-class diversity. Integrated into a dual-branch architecture with self-distillation, our VEGA-augmented model achieves sota performance on IEMOCAP and MELD. Code is available at: https://github.com/dkollias/VEGA.
>
---
#### [replaced 029] Human Motion Capture from Loose and Sparse Inertial Sensors with Garment-aware Diffusion Models
- **分类: cs.GR; cs.AI; cs.CV; cs.HC; 68T07, 68T45, 68U01; I.2; I.3; I.4; I.5**

- **链接: [http://arxiv.org/pdf/2506.15290v2](http://arxiv.org/pdf/2506.15290v2)**

> **作者:** Andela Ilic; Jiaxi Jiang; Paul Streli; Xintong Liu; Christian Holz
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Motion capture using sparse inertial sensors has shown great promise due to its portability and lack of occlusion issues compared to camera-based tracking. Existing approaches typically assume that IMU sensors are tightly attached to the human body. However, this assumption often does not hold in real-world scenarios. In this paper, we present Garment Inertial Poser (GaIP), a method for estimating full-body poses from sparse and loosely attached IMU sensors. We first simulate IMU recordings using an existing garment-aware human motion dataset. Our transformer-based diffusion models synthesize loose IMU data and estimate human poses from this challenging loose IMU data. We also demonstrate that incorporating garment-related parameters during training on loose IMU data effectively maintains expressiveness and enhances the ability to capture variations introduced by looser or tighter garments. Our experiments show that our diffusion methods trained on simulated and synthetic data outperform state-of-the-art inertial full-body pose estimators, both quantitatively and qualitatively, opening up a promising direction for future research on motion capture from such realistic sensor placements.
>
---
#### [replaced 030] FROST-BRDF: A Fast and Robust Optimal Sampling Technique for BRDF Acquisition
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.07283v2](http://arxiv.org/pdf/2401.07283v2)**

> **作者:** Ehsan Miandji; Tanaboon Tongbuasirilai; Saghi Hajisharif; Behnaz Kavoosighafi; Jonas Unger
>
> **备注:** Submitted to IEEE Transactions on Visualization and Computer Graphics (IEEE TVCG)
>
> **摘要:** Efficient and accurate BRDF acquisition of real world materials is a challenging research problem that requires sampling millions of incident light and viewing directions. To accelerate the acquisition process, one needs to find a minimal set of sampling directions such that the recovery of the full BRDF is accurate and robust given such samples. In this paper, we formulate BRDF acquisition as a compressed sensing problem, where the sensing operator is one that performs sub-sampling of the BRDF signal according to a set of optimal sample directions. To solve this problem, we propose the Fast and Robust Optimal Sampling Technique (FROST) for designing a provably optimal sub-sampling operator that places light-view samples such that the recovery error is minimized. FROST casts the problem of designing an optimal sub-sampling operator for compressed sensing into a sparse representation formulation under the Multiple Measurement Vector (MMV) signal model. The proposed reformulation is exact, i.e. without any approximations, hence it converts an intractable combinatorial problem into one that can be solved with standard optimization techniques. As a result, FROST is accompanied by strong theoretical guarantees from the field of compressed sensing. We perform a thorough analysis of FROST-BRDF using a 10-fold cross-validation with publicly available BRDF datasets and show significant advantages compared to the state-of-the-art with respect to reconstruction quality. Finally, FROST is simple, both conceptually and in terms of implementation, it produces consistent results at each run, and it is at least two orders of magnitude faster than the prior art.
>
---
#### [replaced 031] Debiased Fine-Tuning for Vision-language Models by Prompt Regularization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2301.12429v3](http://arxiv.org/pdf/2301.12429v3)**

> **作者:** Beier Zhu; Yulei Niu; Saeil Lee; Minhoe Hur; Hanwang Zhang
>
> **备注:** AAAI2023 accepted
>
> **摘要:** We present a new paradigm for fine-tuning large-scale visionlanguage pre-trained models on downstream task, dubbed Prompt Regularization (ProReg). Different from traditional fine-tuning which easily overfits to the downstream task data, ProReg uses the prediction by prompting the pretrained model to regularize the fine-tuning. The motivation is: by prompting the large model "a photo of a [CLASS]", the fil-lin answer is only dependent on the pretraining encyclopedic knowledge while independent of the task data distribution, which is usually biased. Specifically, given a training sample prediction during fine-tuning, we first calculate its KullbackLeibler loss of the prompt prediction and Cross-Entropy loss of the ground-truth label, and then combine them with a proposed sample-wise adaptive trade-off weight, which automatically adjusts the transfer between the pretrained and downstream domains. On various out-of-distribution benchmarks, we show the consistently strong performance of ProReg compared with conventional fine-tuning, zero-shot prompt, prompt tuning, and other state-of-the-art methods.
>
---
#### [replaced 032] UltraRay: Introducing Full-Path Ray Tracing in Physics-Based Ultrasound Simulation
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2501.05828v2](http://arxiv.org/pdf/2501.05828v2)**

> **作者:** Felix Duelmer; Mohammad Farid Azampour; Magdalena Wysocki; Nassir Navab
>
> **摘要:** Traditional ultrasound simulators solve the wave equation to model pressure distribution fields, achieving high accuracy but requiring significant computational time and resources. To address this, ray tracing approaches have been introduced, modeling wave propagation as rays interacting with boundaries and scatterers. However, existing models simplify ray propagation, generating echoes at interaction points without considering return paths to the sensor. This can result in unrealistic artifacts and necessitates careful scene tuning for plausible results. We propose a novel ultrasound simulation pipeline that utilizes a ray tracing algorithm to generate echo data, tracing each ray from the transducer through the scene and back to the sensor. To replicate advanced ultrasound imaging, we introduce a ray emission scheme optimized for plane wave imaging, incorporating delay and steering capabilities. Furthermore, we integrate a standard signal processing pipeline to simulate end-to-end ultrasound image formation. We showcase the efficacy of the proposed pipeline by modeling synthetic scenes featuring highly reflective objects, such as bones. In doing so, our proposed approach, UltraRay, not only enhances the overall visual quality but also improves the realism of the simulated images by accurately capturing secondary reflections and reducing unnatural artifacts. By building on top of a differentiable framework, the proposed pipeline lays the groundwork for a fast and differentiable ultrasound simulation tool necessary for gradient-based optimization, enabling advanced ultrasound beamforming strategies, neural network integration, and accurate inverse scene reconstruction.
>
---
#### [replaced 033] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.03012v2](http://arxiv.org/pdf/2501.03012v2)**

> **作者:** Pegah Khayatan; Mustafa Shukor; Jayneel Parekh; Arnaud Dapogny; Matthieu Cord
>
> **备注:** ICCV 2025. The first three authors contributed equally. Project page and code: https://pegah- kh.github.io/projects/lmm-finetuning-analysis-and-steering/
>
> **摘要:** Multimodal LLMs (MLLMs) have reached remarkable levels of proficiency in understanding multimodal inputs. However, understanding and interpreting the behavior of such complex models is a challenging task, not to mention the dynamic shifts that may occur during fine-tuning, or due to covariate shift between datasets. In this work, we apply concept-level analysis towards MLLM understanding. More specifically, we propose to map hidden states to interpretable visual and textual concepts. This enables us to more efficiently compare certain semantic dynamics, such as the shift from an original and fine-tuned model, revealing concept alteration and potential biases that may occur during fine-tuning. We also demonstrate the use of shift vectors to capture these concepts changes. These shift vectors allow us to recover fine-tuned concepts by applying simple, computationally inexpensive additive concept shifts in the original model. Finally, our findings also have direct applications for MLLM steering, which can be used for model debiasing as well as enforcing safety in MLLM output. All in all, we propose a novel, training-free, ready-to-use framework for MLLM behavior interpretability and control. Our implementation is publicly available.
>
---
#### [replaced 034] STAC: Leveraging Spatio-Temporal Data Associations For Efficient Cross-Camera Streaming and Analytics
- **分类: cs.CV; cs.MM; cs.NI; I.4.2; I.4.0; C.2.2; C.2.0**

- **链接: [http://arxiv.org/pdf/2401.15288v2](http://arxiv.org/pdf/2401.15288v2)**

> **作者:** Ragini Gupta; Lingzhi Zhao; Jiaxi Li; Volodymyr Vakhniuk; Claudiu Danilov; Josh Eckhardt; Keyshla Bernard; Klara Nahrstedt
>
> **摘要:** In IoT based distributed network of cameras, real-time multi-camera video analytics is challenged by high bandwidth demands and redundant visual data, creating a fundamental tension where reducing data saves network overhead but can degrade model performance, and vice versa. We present STAC, a cross-cameras surveillance system that leverages spatio-temporal associations for efficient object tracking under constrained network conditions. STAC integrates multi-resolution feature learning, ensuring robustness under variable networked system level optimizations such as frame filtering, FFmpeg-based compression, and Region-of-Interest (RoI) masking, to eliminate redundant content across distributed video streams while preserving downstream model accuracy for object identification and tracking. Evaluated on NVIDIA's AICity Challenge dataset, STAC achieves a 76\% improvement in tracking accuracy and an 8.6x reduction in inference latency over a standard multi-object multi-camera tracking baseline (using YOLOv4 and DeepSORT). Furthermore, 29\% of redundant frames are filtered, significantly reducing data volume without compromising inference quality.
>
---
#### [replaced 035] Cryo-em images are intrinsically low dimensional
- **分类: q-bio.QM; cs.CV; cs.LG; q-bio.BM; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.11249v2](http://arxiv.org/pdf/2504.11249v2)**

> **作者:** Luke Evans; Octavian-Vlad Murad; Lars Dingeldein; Pilar Cossio; Roberto Covino; Marina Meila
>
> **摘要:** Simulation-based inference provides a powerful framework for cryo-electron microscopy, employing neural networks in methods like CryoSBI to infer biomolecular conformations via learned latent representations. This latent space represents a rich opportunity, encoding valuable information about the physical system and the inference process. Harnessing this potential hinges on understanding the underlying geometric structure of these representations. We investigate this structure by applying manifold learning techniques to CryoSBI representations of hemagglutinin (simulated and experimental). We reveal that these high-dimensional data inherently populate low-dimensional, smooth manifolds, with simulated data effectively covering the experimental counterpart. By characterizing the manifold's geometry using Diffusion Maps and identifying its principal axes of variation via coordinate interpretation methods, we establish a direct link between the latent structure and key physical parameters. Discovering this intrinsic low-dimensionality and interpretable geometric organization not only validates the CryoSBI approach but enables us to learn more from the data structure and provides opportunities for improving future inference strategies by exploiting this revealed manifold geometry.
>
---
#### [replaced 036] SLTNet: Efficient Event-based Semantic Segmentation with Spike-driven Lightweight Transformer-based Networks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12843v3](http://arxiv.org/pdf/2412.12843v3)**

> **作者:** Xianlei Long; Xiaxin Zhu; Fangming Guo; Wanyi Zhang; Qingyi Gu; Chao Chen; Fuqiang Gu
>
> **备注:** Accepted by IROS 2025 (2025 IEEE/RSJ International Conference on Intelligent Robots and Systems)
>
> **摘要:** Event-based semantic segmentation has great potential in autonomous driving and robotics due to the advantages of event cameras, such as high dynamic range, low latency, and low power cost. Unfortunately, current artificial neural network (ANN)-based segmentation methods suffer from high computational demands, the requirements for image frames, and massive energy consumption, limiting their efficiency and application on resource-constrained edge/mobile platforms. To address these problems, we introduce SLTNet, a spike-driven lightweight transformer-based network designed for event-based semantic segmentation. Specifically, SLTNet is built on efficient spike-driven convolution blocks (SCBs) to extract rich semantic features while reducing the model's parameters. Then, to enhance the long-range contextural feature interaction, we propose novel spike-driven transformer blocks (STBs) with binary mask operations. Based on these basic blocks, SLTNet employs a high-efficiency single-branch architecture while maintaining the low energy consumption of the Spiking Neural Network (SNN). Finally, extensive experiments on DDD17 and DSEC-Semantic datasets demonstrate that SLTNet outperforms state-of-the-art (SOTA) SNN-based methods by at most 9.06% and 9.39% mIoU, respectively, with extremely 4.58x lower energy consumption and 114 FPS inference speed. Our code is open-sourced and available at https://github.com/longxianlei/SLTNet-v1.0.
>
---
#### [replaced 037] NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23162v2](http://arxiv.org/pdf/2503.23162v2)**

> **作者:** Zhenyu Tang; Chaoran Feng; Xinhua Cheng; Wangbo Yu; Junwu Zhang; Yuan Liu; Xiaoxiao Long; Wenping Wang; Li Yuan
>
> **备注:** Project page: https://pku-yuangroup.github.io/NeuralGS/
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves impressive quality and rendering speed, but with millions of 3D Gaussians and significant storage and transmission costs. In this paper, we aim to develop a simple yet effective method called NeuralGS that compresses the original 3DGS into a compact representation. Our observation is that neural fields like NeRF can represent complex 3D scenes with Multi-Layer Perceptron (MLP) neural networks using only a few megabytes. Thus, NeuralGS effectively adopts the neural field representation to encode the attributes of 3D Gaussians with MLPs, only requiring a small storage size even for a large-scale scene. To achieve this, we adopt a clustering strategy and fit the Gaussians within each cluster using different tiny MLPs, based on importance scores of Gaussians as fitting weights. We experiment on multiple datasets, achieving a 91-times average model size reduction without harming the visual quality.
>
---
#### [replaced 038] ViewDelta: Scaling Scene Change Detection through Text-Conditioning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07612v3](http://arxiv.org/pdf/2412.07612v3)**

> **作者:** Subin Varghese; Joshua Gao; Vedhus Hoskere
>
> **摘要:** We introduce a generalized framework for Scene Change Detection (SCD) that addresses the core ambiguity of distinguishing "relevant" from "nuisance" changes, enabling effective joint training of a single model across diverse domains and applications. Existing methods struggle to generalize due to differences in dataset labeling, where changes such as vegetation growth or lane marking alterations may be labeled as relevant in one dataset and irrelevant in another. To resolve this ambiguity, we propose ViewDelta, a text conditioned change detection framework that uses natural language prompts to define relevant changes precisely, such as a single attribute, a specific set of classes, or all observable differences. To facilitate training in this paradigm, we release the Conditional Change Segmentation dataset (CSeg), the first large-scale synthetic dataset for text conditioned SCD, consisting of over 500,000 image pairs with more than 300,000 unique textual prompts describing relevant changes. Experiments demonstrate that a single ViewDelta model trained jointly on CSeg, SYSU-CD, PSCD, VL-CMU-CD, and their unaligned variants achieves performance competitive with or superior to dataset specific models, highlighting text conditioning as a powerful approach for generalizable SCD. Our code and dataset are available at https://joshuakgao.github.io/viewdelta/.
>
---
#### [replaced 039] Joint multi-dimensional dynamic attention and transformer for general image restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07893v2](http://arxiv.org/pdf/2411.07893v2)**

> **作者:** Huan Zhang; Xu Zhang; Nian Cai; Jianglei Di; Yun Zhang
>
> **摘要:** Outdoor images often suffer from severe degradation due to rain, haze, and noise, impairing image quality and challenging high-level tasks. Current image restoration methods struggle to handle complex degradation while maintaining efficiency. This paper introduces a novel image restoration architecture that combines multi-dimensional dynamic attention and self-attention within a U-Net framework. To leverage the global modeling capabilities of transformers and the local modeling capabilities of convolutions, we integrate sole CNNs in the encoder-decoder and sole transformers in the latent layer. Additionally, we design convolutional kernels with selected multi-dimensional dynamic attention to capture diverse degraded inputs efficiently. A transformer block with transposed self-attention further enhances global feature extraction while maintaining efficiency. Extensive experiments demonstrate that our method achieves a better balance between performance and computational complexity across five image restoration tasks: deraining, deblurring, denoising, dehazing, and enhancement, as well as superior performance for high-level vision tasks. The source code will be available at https://github.com/House-yuyu/MDDA-former.
>
---
#### [replaced 040] Follow-Your-Motion: Video Motion Transfer via Efficient Spatial-Temporal Decoupled Finetuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05207v2](http://arxiv.org/pdf/2506.05207v2)**

> **作者:** Yue Ma; Yulong Liu; Qiyuan Zhu; Ayden Yang; Kunyu Feng; Xinhua Zhang; Zhifeng Li; Sirui Han; Chenyang Qi; Qifeng Chen
>
> **备注:** project page: https://follow-your-motion.github.io/
>
> **摘要:** Recently, breakthroughs in the video diffusion transformer have shown remarkable capabilities in diverse motion generations. As for the motion-transfer task, current methods mainly use two-stage Low-Rank Adaptations (LoRAs) finetuning to obtain better performance. However, existing adaptation-based motion transfer still suffers from motion inconsistency and tuning inefficiency when applied to large video diffusion transformers. Naive two-stage LoRA tuning struggles to maintain motion consistency between generated and input videos due to the inherent spatial-temporal coupling in the 3D attention operator. Additionally, they require time-consuming fine-tuning processes in both stages. To tackle these issues, we propose Follow-Your-Motion, an efficient two-stage video motion transfer framework that finetunes a powerful video diffusion transformer to synthesize complex motion. Specifically, we propose a spatial-temporal decoupled LoRA to decouple the attention architecture for spatial appearance and temporal motion processing. During the second training stage, we design the sparse motion sampling and adaptive RoPE to accelerate the tuning speed. To address the lack of a benchmark for this field, we introduce MotionBench, a comprehensive benchmark comprising diverse motion, including creative camera motion, single object motion, multiple object motion, and complex human motion. We show extensive evaluations on MotionBench to verify the superiority of Follow-Your-Motion.
>
---
#### [replaced 041] Calibrated Self-supervised Vision Transformers Improve Intracranial Arterial Calcification Segmentation from Clinical CT Head Scans
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01744v2](http://arxiv.org/pdf/2507.01744v2)**

> **作者:** Benjamin Jin; Grant Mair; Joanna M. Wardlaw; Maria del C. Valdés Hernández
>
> **备注:** Accepted at the 3rd Data Engineering in Medical Imaging workshop @ MICCAI 2025
>
> **摘要:** Vision Transformers (ViTs) have gained significant popularity in the natural image domain but have been less successful in 3D medical image segmentation. Nevertheless, 3D ViTs are particularly interesting for large medical imaging volumes due to their efficient self-supervised training within the masked autoencoder (MAE) framework, which enables the use of imaging data without the need for expensive manual annotations. Intracranial arterial calcification (IAC) is an imaging biomarker visible on routinely acquired CT scans linked to neurovascular diseases such as stroke and dementia, and automated IAC quantification could enable their large-scale risk assessment. We pre-train ViTs with MAE and fine-tune them for IAC segmentation for the first time. To develop our models, we use highly heterogeneous data from a large clinical trial, the third International Stroke Trial (IST-3). We evaluate key aspects of MAE pre-trained ViTs in IAC segmentation, and analyse the clinical implications. We show: 1) our calibrated self-supervised ViT beats a strong supervised nnU-Net baseline by 3.2 Dice points, 2) low patch sizes are crucial for ViTs for IAC segmentation and interpolation upsampling with regular convolutions is preferable to transposed convolutions for ViT-based models, and 3) our ViTs increase robustness to higher slice thicknesses and improve risk group classification in a clinical scenario by 46%. Our code is available online.
>
---
#### [replaced 042] Integrating Clinical Knowledge Graphs and Gradient-Based Neural Systems for Enhanced Melanoma Diagnosis via the 7-Point Checklist
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.16822v2](http://arxiv.org/pdf/2407.16822v2)**

> **作者:** Yuheng Wang; Tianze Yu; Jiayue Cai; Sunil Kalia; Harvey Lui; Z. Jane Wang; Tim K. Lee
>
> **备注:** The paper was officially accepted for publication in IEEE Transactions on Neural Networks and Learning Systems in August 2025
>
> **摘要:** The 7-point checklist (7PCL) is a widely used diagnostic tool in dermoscopy for identifying malignant melanoma by assigning point values to seven specific attributes. However, the traditional 7PCL is limited to distinguishing between malignant melanoma and melanocytic Nevi, and falls short in scenarios where multiple skin diseases with appearances similar to melanoma coexist. To address this limitation, we propose a novel diagnostic framework that integrates a clinical knowledge-based topological graph (CKTG) with a gradient diagnostic strategy featuring a data-driven weighting system (GD-DDW). The CKTG captures both the internal and external relationships among the 7PCL attributes, while the GD-DDW emulates dermatologists' diagnostic processes, prioritizing visual observation before making predictions. Additionally, we introduce a multimodal feature extraction approach leveraging a dual-attention mechanism to enhance feature extraction through cross-modal interaction and unimodal collaboration. This method incorporates meta-information to uncover interactions between clinical data and image features, ensuring more accurate and robust predictions. Our approach, evaluated on the EDRA dataset, achieved an average AUC of 88.6%, demonstrating superior performance in melanoma detection and feature prediction. This integrated system provides data-driven benchmarks for clinicians, significantly enhancing the precision of melanoma diagnosis.
>
---
#### [replaced 043] Scaling Vision Mamba Across Resolutions via Fractal Traversal
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14062v2](http://arxiv.org/pdf/2505.14062v2)**

> **作者:** Bo Li; Haoke Xiao; Lv Tang
>
> **摘要:** Vision Mamba has recently emerged as a promising alternative to Transformer-based architectures, offering linear complexity in sequence length while maintaining strong modeling capacity. However, its adaptation to visual inputs is hindered by challenges in 2D-to-1D patch serialization and weak scalability across input resolutions. Existing serialization strategies such as raster scanning disrupt local spatial continuity and limit the model's ability to generalize across scales. In this paper, we propose FractalMamba++, a robust vision backbone that leverages fractal-based patch serialization via Hilbert curves to preserve spatial locality and enable seamless resolution adaptability. To address long-range dependency fading in high-resolution inputs, we further introduce a Cross-State Routing (CSR) mechanism that enhances global context propagation through selective state reuse. Additionally, we propose a Positional-Relation Capture (PRC) module to recover local adjacency disrupted by curve inflection points. Extensive experiments across diverse downstream tasks, including image classification, semantic segmentation and object detection, demonstrate that FractalMamba++ consistently outperforms previous Mamba-based backbones, with particularly notable gains under high-resolution settings.
>
---
#### [replaced 044] Transferable Model-agnostic Vision-Language Model Adaptation for Efficient Weak-to-Strong Generalization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.08604v2](http://arxiv.org/pdf/2508.08604v2)**

> **作者:** Jihwan Park; Taehoon song; Sanghyeok Lee; Miso Choi; Hyunwoo J. Kim
>
> **摘要:** Vision-Language Models (VLMs) have been widely used in various visual recognition tasks due to their remarkable generalization capabilities. As these models grow in size and complexity, fine-tuning becomes costly, emphasizing the need to reuse adaptation knowledge from 'weaker' models to efficiently enhance 'stronger' ones. However, existing adaptation transfer methods exhibit limited transferability across models due to their model-specific design and high computational demands. To tackle this, we propose Transferable Model-agnostic adapter (TransMiter), a light-weight adapter that improves vision-language models 'without backpropagation'. TransMiter captures the knowledge gap between pre-trained and fine-tuned VLMs, in an 'unsupervised' manner. Once trained, this knowledge can be seamlessly transferred across different models without the need for backpropagation. Moreover, TransMiter consists of only a few layers, inducing a negligible additional inference cost. Notably, supplementing the process with a few labeled data further yields additional performance gain, often surpassing a fine-tuned stronger model, with a marginal training cost. Experimental results and analyses demonstrate that TransMiter effectively and efficiently transfers adaptation knowledge while preserving generalization abilities across VLMs of different sizes and architectures in visual recognition tasks.
>
---
#### [replaced 045] Explaining Caption-Image Interactions in CLIP Models with Second-Order Attributions
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.14153v4](http://arxiv.org/pdf/2408.14153v4)**

> **作者:** Lucas Möller; Pascal Tilli; Ngoc Thang Vu; Sebastian Padó
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Dual encoder architectures like Clip models map two types of inputs into a shared embedding space and predict similarities between them. Despite their wide application, it is, however, not understood how these models compare their two inputs. Common first-order feature-attribution methods explain importances of individual features and can, thus, only provide limited insights into dual encoders, whose predictions depend on interactions between features. In this paper, we first derive a second-order method enabling the attribution of predictions by any differentiable dual encoder onto feature-interactions between its inputs. Second, we apply our method to Clip models and show that they learn fine-grained correspondences between parts of captions and regions in images. They match objects across input modes and also account for mismatches. This intrinsic visual-linguistic grounding ability, however, varies heavily between object classes, exhibits pronounced out-of-domain effects and we can identify individual errors as well as systematic failure categories. Code is publicly available: https://github.com/lucasmllr/exCLIP
>
---
#### [replaced 046] SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07966v3](http://arxiv.org/pdf/2506.07966v3)**

> **作者:** Ziyang Gong; Wenhao Li; Oliver Ma; Songyuan Li; Jiayi Ji; Xue Yang; Gen Luo; Junchi Yan; Rongrong Ji
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in various multimodal tasks. To pursue higher intelligence in space, MLLMs require integrating multiple atomic spatial capabilities to handle complex and dynamic tasks. However, existing benchmarks struggle to comprehensively evaluate the spatial intelligence of common MLLMs from the atomic level to the compositional level. To fill this gap, we present SpaCE-10, a comprehensive benchmark for compositional spatial evaluations. In SpaCE-10, we define 10 atomic spatial capabilities, which are combined to form 8 compositional capabilities. Based on these definitions, we propose a novel hierarchical annotation pipeline to generate high-quality and diverse question-answer (QA) pairs. With over 150+ hours of human expert effort, we obtain over 5k QA pairs for 811 real indoor scenes in SpaCE-10, which covers various evaluation settings like point cloud input and multi-choice QA. We conduct an extensive evaluation of common MLLMs on SpaCE-10 and find that even the most advanced MLLM still lags behind humans by large margins. Through our careful study, we also draw several significant findings that benefit the MLLM community. For example, we reveal that the shortcoming of counting capability greatly limits the compositional spatial capabilities of existing MLLMs. The evaluation code and benchmark datasets are available at https://github.com/Cuzyoung/SpaCE-10.
>
---
#### [replaced 047] GranQ: Granular Zero-Shot Quantization with Channel-Wise Activation Scaling in QAT
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18339v5](http://arxiv.org/pdf/2503.18339v5)**

> **作者:** Inpyo Hong; Youngwan Jo; Hyojeong Lee; Sunghyun Ahn; Kijung Lee; Sanghyun Park
>
> **摘要:** Zero-shot quantization (ZSQ) enables neural network compression without original training data, making it a promising solution for restricted data access scenarios. To compensate for the lack of data, recent ZSQ methods typically rely on synthetic inputs generated from the full-precision model. However, these synthetic inputs often lead to activation distortion, especially under low-bit settings. To mitigate this, existing methods typically employ per-channel scaling, but they still struggle due to the severe computational overhead during the accumulation process. To overcome this critical bottleneck, we propose GranQ, a novel activation quantization framework that introduces an efficient pre-scaling strategy. Unlike conventional channel-wise methods that repeatedly perform scaling operations during accumulation, GranQ applies scaling factors in a pre-scaling step through fully vectorized computation, eliminating runtime scaling overhead. This design enables GranQ to maintain fine-grained quantization accuracy while significantly reducing computational burden, particularly in low-bit quantization settings. Extensive experiments under quantization-aware training (QAT) settings demonstrate that GranQ consistently outperforms state-of-the-art ZSQ methods across CIFAR and ImageNet. In particular, our method achieves up to 5.45% higher accuracy in the 3-bit setting on CIFAR-100 and even surpasses the full-precision baseline on CIFAR-10. Furthermore, GranQ achieves significant speedup in quantization latency over conventional per-channel methods, demonstrating improved efficiency. With these findings, we anticipate that GranQ will inspire future research beyond conventional ZSQ approaches centered on data generation and model fine-tuning. The official code is available at https://github.com/anonymus-orange/GranQ.
>
---
#### [replaced 048] Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07701v2](http://arxiv.org/pdf/2508.07701v2)**

> **作者:** Bo Jia; Yanan Guo; Ying Chang; Benkui Zhang; Ying Xie; Kangning Du; Lin Cao
>
> **备注:** This paper has been accepted by IROS 2025. Code: https://github.com/Bistu3DV/MND-GS/
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves remarkable results in the field of surface reconstruction. However, when Gaussian normal vectors are aligned within the single-view projection plane, while the geometry appears reasonable in the current view, biases may emerge upon switching to nearby views. To address the distance and global matching challenges in multi-view scenes, we design multi-view normal and distance-guided Gaussian splatting. This method achieves geometric depth unification and high-accuracy reconstruction by constraining nearby depth maps and aligning 3D normals. Specifically, for the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multi-view normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss. Extensive experimental results demonstrate that our method outperforms the baseline in both quantitative and qualitative evaluations, significantly enhancing the surface reconstruction capability of 3DGS. Our code will be made publicly available at (https://github.com/Bistu3DV/MND-GS/).
>
---
#### [replaced 049] See the Forest and the Trees: A Synergistic Reasoning Framework for Knowledge-Based Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17659v3](http://arxiv.org/pdf/2507.17659v3)**

> **作者:** Junjie Wang; Yunhan Tang; Yijie Wang; Zhihao Yuan; Huan Wang; Yangfan He; Bin Li
>
> **备注:** We are withdrawing this preprint because it is undergoing a major revision and restructuring. We feel that the current version does not convey our core contributions and methodology with sufficient clarity and accuracy
>
> **摘要:** Multimodal Large Language Models (MLLMs) have pushed the frontiers of Knowledge-Based Visual Question Answering (KBVQA), yet their reasoning is fundamentally bottlenecked by a reliance on uni-dimensional evidence. This "seeing only the trees, but not the forest" approach prevents robust, multi-faceted understanding. Inspired by the principle of seeing both the forest and trees, we propose Synergos-VQA, a novel synergistic reasoning framework. At its core, Synergos-VQA concurrently generates and fuses three complementary evidence streams at inference time: (1) Holistic Evidence to perceive the entire scene (the "forest"), (2) Structural Evidence from a prototype-driven module to identify key objects (the "trees"), and (3) Causal Evidence from a counterfactual probe to ensure the reasoning is robustly grounded. By synergistically fusing this multi-faceted evidence, our framework achieves a more comprehensive and reliable reasoning process. Extensive experiments show that Synergos-VQA decisively establishes a new state-of-the-art on three challenging benchmarks, including OK-VQA and A-OKVQA. Furthermore, our approach demonstrates strong plug-and-play capabilities, significantly boosting various open-source MLLMs and proving that superior methodological design can outperform sheer model scale.
>
---
#### [replaced 050] HRSeg: High-Resolution Visual Perception and Enhancement for Reasoning Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12883v2](http://arxiv.org/pdf/2507.12883v2)**

> **作者:** Weihuang Lin; Yiwei Ma; Xiaoshuai Sun; Shuting He; Jiayi Ji; Liujuan Cao; Rongrong Ji
>
> **备注:** 10 pages, 4 figures, ACM MM25
>
> **摘要:** The reasoning segmentation task involves segmenting objects within an image by interpreting implicit user instructions, which may encompass subtleties such as contextual cues and open-world knowledge. Despite significant advancements made by existing approaches, they remain constrained by low perceptual resolution, as visual encoders are typically pre-trained at lower resolutions. Furthermore, simply interpolating the positional embeddings of visual encoders to enhance perceptual resolution yields only marginal performance improvements while incurring substantial computational costs. To address this, we propose HRSeg, an efficient model with high-resolution fine-grained perception. It features two key innovations: High-Resolution Perception (HRP) and High-Resolution Enhancement (HRE). The HRP module processes high-resolution images through cropping, integrating local and global features for multi-granularity quality. The HRE module enhances mask features by integrating fine-grained information from high-resolution images, refining their alignment with text features for precise segmentation. Extensive ablation studies validate the effectiveness of our modules, while comprehensive experiments on multiple benchmark datasets demonstrate HRSeg's superior performance.
>
---
#### [replaced 051] PAD-F: Prior-Aware Debiasing Framework for Long-Tailed X-ray Prohibited Item Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18078v4](http://arxiv.org/pdf/2411.18078v4)**

> **作者:** Haoyu Wang; Renshuai Tao; Wei Wang; Yunchao Wei
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Detecting prohibited items in X-ray security imagery is a challenging yet crucial task. With the rapid advancement of deep learning, object detection algorithms have been widely applied in this area. However, the distribution of object classes in real-world prohibited item detection scenarios often exhibits a distinct long-tailed distribution. Due to the unique principles of X-ray imaging, conventional methods for long-tailed object detection are often ineffective in this domain. To tackle these challenges, we introduce the Prior-Aware Debiasing Framework (PAD-F), a novel approach that employs a two-pronged strategy leveraging both material and co-occurrence priors. At the data level, our Explicit Material-Aware Augmentation (EMAA) component generates numerous challenging training samples for tail classes. It achieves this through a placement strategy guided by material-specific absorption rates and a gradient-based Poisson blending technique. At the feature level, the Implicit Co-occurrence Aggregator (ICA) acts as a plug-in module that enhances features for ambiguous objects by implicitly learning and aggregating statistical co-occurrence relationships within the image. Extensive experiments on the HiXray and PIDray datasets demonstrate that PAD-F significantly boosts the performance of multiple popular detectors. It achieves an absolute improvement of up to +17.2% in AP50 for tail classes and comprehensively outperforms existing state-of-the-art methods. Our work provides an effective and versatile solution to the critical problem of long-tailed detection in X-ray security.
>
---
#### [replaced 052] Improving Multimodal Large Language Models Using Continual Learning
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.19925v2](http://arxiv.org/pdf/2410.19925v2)**

> **作者:** Shikhar Srivastava; Md Yousuf Harun; Robik Shrestha; Christopher Kanan
>
> **备注:** CoLLAs 2025 and Scalable Continual Learning for Lifelong Foundation Models, NeurIPS 2024
>
> **摘要:** Generative large language models (LLMs) exhibit impressive capabilities, which can be further augmented by integrating a pre-trained vision model into the original LLM to create a multimodal LLM (MLLM). However, this integration often significantly decreases performance on natural language understanding and generation tasks, compared to the original LLM. This study investigates this issue using the LLaVA MLLM, treating the integration as a continual learning problem. We evaluate five continual learning methods to mitigate forgetting and identify a technique that enhances visual understanding while minimizing linguistic performance loss. Our approach reduces linguistic performance degradation by up to 15% over the LLaVA recipe, while maintaining high multimodal accuracy. We also demonstrate the robustness of our method through continual learning on a sequence of vision-language tasks, effectively preserving linguistic skills while acquiring new multimodal capabilities. Project webpage: https://shikhar-srivastava.github.io/cl-for-improving-mllms
>
---
#### [replaced 053] From Few to More: Scribble-based Medical Image Segmentation via Masked Context Modeling and Continuous Pseudo Labels
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.12814v2](http://arxiv.org/pdf/2408.12814v2)**

> **作者:** Zhisong Wang; Yiwen Ye; Ziyang Chen; Minglei Shu; Yanning Zhang; Yong Xia
>
> **备注:** 13 pages, 10 figures, 10 tables, JBHI
>
> **摘要:** Scribble-based weakly supervised segmentation methods have shown promising results in medical image segmentation, significantly reducing annotation costs. However, existing approaches often rely on auxiliary tasks to enforce semantic consistency and use hard pseudo labels for supervision, overlooking the unique challenges faced by models trained with sparse annotations. These models must predict pixel-wise segmentation maps from limited data, making it crucial to handle varying levels of annotation richness effectively. In this paper, we propose MaCo, a weakly supervised model designed for medical image segmentation, based on the principle of "from few to more." MaCo leverages Masked Context Modeling (MCM) and Continuous Pseudo Labels (CPL). MCM employs an attention-based masking strategy to perturb the input image, ensuring that the model's predictions align with those of the original image. CPL converts scribble annotations into continuous pixel-wise labels by applying an exponential decay function to distance maps, producing confidence maps that represent the likelihood of each pixel belonging to a specific category, rather than relying on hard pseudo labels. We evaluate MaCo on three public datasets, comparing it with other weakly supervised methods. Our results show that MaCo outperforms competing methods across all datasets, establishing a new record in weakly supervised medical image segmentation.
>
---
#### [replaced 054] DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01950v3](http://arxiv.org/pdf/2506.01950v3)**

> **作者:** Jiajun Jiang; Yiming Zhu; Zirui Wu; Jie Song
>
> **备注:** 14 pages, 14 figures. Code: https://github.com/Eku127/DualMap Project page: https://eku127.github.io/DualMap/
>
> **摘要:** We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation.Project page: https://eku127.github.io/DualMap/
>
---
#### [replaced 055] CD-TVD: Contrastive Diffusion for 3D Super-Resolution with Scarce High-Resolution Time-Varying Data
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.08173v2](http://arxiv.org/pdf/2508.08173v2)**

> **作者:** Chongke Bi; Xin Gao; Jiangkang Deng; Guan Li; Jun Han
>
> **备注:** Accepted to IEEE VIS 2025
>
> **摘要:** Large-scale scientific simulations require significant resources to generate high-resolution time-varying data (TVD). While super-resolution is an efficient post-processing strategy to reduce costs, existing methods rely on a large amount of HR training data, limiting their applicability to diverse simulation scenarios. To address this constraint, we proposed CD-TVD, a novel framework that combines contrastive learning and an improved diffusion-based super-resolution model to achieve accurate 3D super-resolution from limited time-step high-resolution data. During pre-training on historical simulation data, the contrastive encoder and diffusion superresolution modules learn degradation patterns and detailed features of high-resolution and low-resolution samples. In the training phase, the improved diffusion model with a local attention mechanism is fine-tuned using only one newly generated high-resolution timestep, leveraging the degradation knowledge learned by the encoder. This design minimizes the reliance on large-scale high-resolution datasets while maintaining the capability to recover fine-grained details. Experimental results on fluid and atmospheric simulation datasets confirm that CD-TVD delivers accurate and resource-efficient 3D super-resolution, marking a significant advancement in data augmentation for large-scale scientific simulations. The code is available at https://github.com/Xin-Gao-private/CD-TVD.
>
---
#### [replaced 056] RoHOI: Robustness Benchmark for Human-Object Interaction Detection
- **分类: cs.CV; cs.HC; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.09111v2](http://arxiv.org/pdf/2507.09111v2)**

> **作者:** Di Wen; Kunyu Peng; Kailun Yang; Yufan Chen; Ruiping Liu; Junwei Zheng; Alina Roitberg; Danda Pani Paudel; Luc Van Gool; Rainer Stiefelhagen
>
> **备注:** Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI
>
> **摘要:** Human-Object Interaction (HOI) detection is crucial for robot-human assistance, enabling context-aware support. However, models trained on clean datasets degrade in real-world conditions due to unforeseen corruptions, leading to inaccurate prediction. To address this, we introduce the first robustness benchmark for HOI detection, evaluating model resilience under diverse challenges. Despite advances, current models struggle with environmental variability, occlusions, and noise. Our benchmark, RoHOI, includes 20 corruption types based on the HICO-DET and V-COCO datasets and a new robustness-focused metric. We systematically analyze existing models in the HOI field, revealing significant performance drops under corruptions. To improve robustness, we propose a Semantic-Aware Masking-based Progressive Learning (SAMPL) strategy to guide the model to be optimized based on holistic and partial cues, thus dynamically adjusting the model's optimization to enhance robust feature learning. Extensive experiments show that our approach outperforms state-of-the-art methods, setting a new standard for robust HOI detection. Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI.
>
---
#### [replaced 057] MultiFormer: A Multi-Person Pose Estimation System Based on CSI and Attention Mechanism
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.22555v2](http://arxiv.org/pdf/2505.22555v2)**

> **作者:** Yanyi Qu; Haoyang Ma; Wenhui Xiong
>
> **摘要:** Human pose estimation based on Channel State Information (CSI) has emerged as a promising approach for non-intrusive and precise human activity monitoring, yet faces challenges including accurate multi-person pose recognition and effective CSI feature learning. This paper presents MultiFormer, a wireless sensing system that accurately estimates human pose through CSI. The proposed system adopts a Transformer based time-frequency dual-token feature extractor with multi-head self-attention. This feature extractor is able to model inter-subcarrier correlations and temporal dependencies of the CSI. The extracted CSI features and the pose probability heatmaps are then fused by Multi-Stage Feature Fusion Network (MSFN) to enforce the anatomical constraints. Extensive experiments conducted on on the public MM-Fi dataset and our self-collected dataset show that the MultiFormer achieves higher accuracy over state-of-the-art approaches, especially for high-mobility keypoints (wrists, elbows) that are particularly difficult for previous methods to accurately estimate.
>
---
#### [replaced 058] MoCA: Identity-Preserving Text-to-Video Generation via Mixture of Cross Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03034v2](http://arxiv.org/pdf/2508.03034v2)**

> **作者:** Qi Xie; Yongjia Ma; Donglin Di; Xuehao Gao; Xun Yang
>
> **摘要:** Achieving ID-preserving text-to-video (T2V) generation remains challenging despite recent advances in diffusion-based models. Existing approaches often fail to capture fine-grained facial dynamics or maintain temporal identity coherence. To address these limitations, we propose MoCA, a novel Video Diffusion Model built on a Diffusion Transformer (DiT) backbone, incorporating a Mixture of Cross-Attention mechanism inspired by the Mixture-of-Experts paradigm. Our framework improves inter-frame identity consistency by embedding MoCA layers into each DiT block, where Hierarchical Temporal Pooling captures identity features over varying timescales, and Temporal-Aware Cross-Attention Experts dynamically model spatiotemporal relationships. We further incorporate a Latent Video Perceptual Loss to enhance identity coherence and fine-grained details across video frames. To train this model, we collect CelebIPVid, a dataset of 10,000 high-resolution videos from 1,000 diverse individuals, promoting cross-ethnicity generalization. Extensive experiments on CelebIPVid show that MoCA outperforms existing T2V methods by over 5% across Face similarity.
>
---
#### [replaced 059] BridgeDepth: Bridging Monocular and Stereo Reasoning with Latent Alignment
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.04611v2](http://arxiv.org/pdf/2508.04611v2)**

> **作者:** Tongfan Guan; Jiaxin Guo; Chen Wang; Yun-Hui Liu
>
> **备注:** ICCV 2025 Highlight
>
> **摘要:** Monocular and stereo depth estimation offer complementary strengths: monocular methods capture rich contextual priors but lack geometric precision, while stereo approaches leverage epipolar geometry yet struggle with ambiguities such as reflective or textureless surfaces. Despite post-hoc synergies, these paradigms remain largely disjoint in practice. We introduce a unified framework that bridges both through iterative bidirectional alignment of their latent representations. At its core, a novel cross-attentive alignment mechanism dynamically synchronizes monocular contextual cues with stereo hypothesis representations during stereo reasoning. This mutual alignment resolves stereo ambiguities (e.g., specular surfaces) by injecting monocular structure priors while refining monocular depth with stereo geometry within a single network. Extensive experiments demonstrate state-of-the-art results: \textbf{it reduces zero-shot generalization error by $\!>\!40\%$ on Middlebury and ETH3D}, while addressing longstanding failures on transparent and reflective surfaces. By harmonizing multi-view geometry with monocular context, our approach enables robust 3D perception that transcends modality-specific limitations. Codes available at https://github.com/aeolusguan/BridgeDepth.
>
---
#### [replaced 060] Lung-DDPM: Semantic Layout-guided Diffusion Models for Thoracic CT Image Synthesis
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15204v2](http://arxiv.org/pdf/2502.15204v2)**

> **作者:** Yifan Jiang; Yannick Lemaréchal; Sophie Plante; Josée Bafaro; Jessica Abi-Rjeile; Philippe Joubert; Philippe Després; Venkata Manem
>
> **备注:** Accepted by IEEE Transactions on Biomedical Engineering (TBME)
>
> **摘要:** With the rapid development of artificial intelligence (AI), AI-assisted medical imaging analysis demonstrates remarkable performance in early lung cancer screening. However, the costly annotation process and privacy concerns limit the construction of large-scale medical datasets, hampering the further application of AI in healthcare. To address the data scarcity in lung cancer screening, we propose Lung-DDPM, a thoracic CT image synthesis approach that effectively generates high-fidelity 3D synthetic CT images, which prove helpful in downstream lung nodule segmentation tasks. Our method is based on semantic layout-guided denoising diffusion probabilistic models (DDPM), enabling anatomically reasonable, seamless, and consistent sample generation even from incomplete semantic layouts. Our results suggest that the proposed method outperforms other state-of-the-art (SOTA) generative models in image quality evaluation and downstream lung nodule segmentation tasks. Specifically, Lung-DDPM achieved superior performance on our large validation cohort, with a Fr\'echet inception distance (FID) of 0.0047, maximum mean discrepancy (MMD) of 0.0070, and mean squared error (MSE) of 0.0024. These results were 7.4$\times$, 3.1$\times$, and 29.5$\times$ better than the second-best competitors, respectively. Furthermore, the lung nodule segmentation model, trained on a dataset combining real and Lung-DDPM-generated synthetic samples, attained a Dice Coefficient (Dice) of 0.3914 and sensitivity of 0.4393. This represents 8.8% and 18.6% improvements in Dice and sensitivity compared to the model trained solely on real samples. The experimental results highlight Lung-DDPM's potential for a broader range of medical imaging applications, such as general tumor segmentation, cancer survival estimation, and risk prediction. The code and pretrained models are available at https://github.com/Manem-Lab/Lung-DDPM/.
>
---
#### [replaced 061] Prompt-aligned Gradient for Prompt Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2205.14865v4](http://arxiv.org/pdf/2205.14865v4)**

> **作者:** Beier Zhu; Yulei Niu; Yucheng Han; Yue Wu; Hanwang Zhang
>
> **备注:** ICCV2023
>
> **摘要:** Thanks to the large pre-trained vision-language models (VLMs) like CLIP, we can craft a zero-shot classifier by "prompt", e.g., the confidence score of an image being "[CLASS]" can be obtained by using the VLM provided similarity measure between the image and the prompt sentence "a photo of a [CLASS]". Therefore, prompt shows a great potential for fast adaptation of VLMs to downstream tasks if we fine-tune the prompt-based similarity measure. However, we find a common failure that improper fine-tuning may not only undermine the prompt's inherent prediction for the task-related classes, but also for other classes in the VLM vocabulary. Existing methods still address this problem by using traditional anti-overfitting techniques such as early stopping and data augmentation, which lack a principled solution specific to prompt. We present Prompt-aligned Gradient, dubbed ProGrad, to prevent prompt tuning from forgetting the the general knowledge learned from VLMs. In particular, ProGrad only updates the prompt whose gradient is aligned (or non-conflicting) to the "general direction", which is represented as the gradient of the KL loss of the pre-defined prompt prediction. Extensive experiments demonstrate the stronger few-shot generalization ability of ProGrad over state-of-the-art prompt tuning methods. Codes are available at https://github.com/BeierZhu/Prompt-align.
>
---
#### [replaced 062] Video SimpleQA: Towards Factuality Evaluation in Large Video Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18923v2](http://arxiv.org/pdf/2503.18923v2)**

> **作者:** Meng Cao; Pengfei Hu; Yingyao Wang; Jihao Gu; Haoran Tang; Haoze Zhao; Chen Wang; Jiahua Dong; Wangbo Yu; Ge Zhang; Jun Song; Xiang Li; Bo Zheng; Ian Reid; Xiaodan Liang
>
> **摘要:** Recent advancements in Large Video Language Models (LVLMs) have highlighted their potential for multi-modal understanding, yet evaluating their factual grounding in videos remains a critical unsolved challenge. To address this gap, we introduce Video SimpleQA, the first comprehensive benchmark tailored for factuality evaluation in video contexts. Our work differs from existing video benchmarks through the following key features: 1) Knowledge required: demanding integration of external knowledge beyond the video's explicit narrative; 2) Multi-hop fact-seeking question: Each question involves multiple explicit facts and requires strict factual grounding without hypothetical or subjective inferences. We also include per-hop single-fact-based sub-QAs alongside final QAs to enable fine-grained, stepby-step evaluation; 3) Short-form definitive answer: Answers are crafted as unambiguous and definitively correct in a short format with minimal scoring variance; 4) Temporal grounded required: Requiring answers to rely on one or more temporal segments in videos, rather than single frames. We extensively evaluate 33 state-of-the-art LVLMs and summarize key findings as follows: 1) Current LVLMs exhibit notable deficiencies in factual adherence, with the best-performing model o3 merely achieving an F-score of 66.3%; 2) Most LVLMs are overconfident in what they generate, with self-stated confidence exceeding actual accuracy; 3) Retrieval-augmented generation demonstrates consistent improvements at the cost of additional inference time overhead; 4) Multi-hop QA demonstrates substantially degraded performance compared to single-hop sub-QAs, with first-hop object or event recognition emerging as the primary bottleneck. We position Video SimpleQA as the cornerstone benchmark for video factuality assessment, aiming to steer LVLM development toward verifiable grounding in real-world contexts.
>
---
#### [replaced 063] When Deepfakes Look Real: Detecting AI-Generated Faces with Unlabeled Data due to Annotation Challenges
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09022v2](http://arxiv.org/pdf/2508.09022v2)**

> **作者:** Zhiqiang Yang; Renshuai Tao; Xiaolong Zheng; Guodong Yang; Chunjie Zhang
>
> **备注:** 10pages,5figures
>
> **摘要:** Existing deepfake detection methods heavily depend on labeled training data. However, as AI-generated content becomes increasingly realistic, even \textbf{human annotators struggle to distinguish} between deepfakes and authentic images. This makes the labeling process both time-consuming and less reliable. Specifically, there is a growing demand for approaches that can effectively utilize large-scale unlabeled data from online social networks. Unlike typical unsupervised learning tasks, where categories are distinct, AI-generated faces closely mimic real image distributions and share strong similarities, causing performance drop in conventional strategies. In this paper, we introduce the Dual-Path Guidance Network (DPGNet), to tackle two key challenges: (1) bridging the domain gap between faces from different generation models, and (2) utilizing unlabeled image samples. The method features two core modules: text-guided cross-domain alignment, which uses learnable prompts to unify visual and textual embeddings into a domain-invariant feature space, and curriculum-driven pseudo label generation, which dynamically exploit more informative unlabeled samples. To prevent catastrophic forgetting, we also facilitate bridging between domains via cross-domain knowledge distillation. Extensive experiments on \textbf{11 popular datasets}, show that DPGNet outperforms SoTA approaches by \textbf{6.3\%}, highlighting its effectiveness in leveraging unlabeled data to address the annotation challenges posed by the increasing realism of deepfakes.
>
---
#### [replaced 064] Towards flexible perception with visual memory
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.08172v3](http://arxiv.org/pdf/2408.08172v3)**

> **作者:** Robert Geirhos; Priyank Jaini; Austin Stone; Sourabh Medapati; Xi Yi; George Toderici; Abhijit Ogale; Jonathon Shlens
>
> **备注:** ICML 2025 camera ready version
>
> **摘要:** Training a neural network is a monolithic endeavor, akin to carving knowledge into stone: once the process is completed, editing the knowledge in a network is hard, since all information is distributed across the network's weights. We here explore a simple, compelling alternative by marrying the representational power of deep neural networks with the flexibility of a database. Decomposing the task of image classification into image similarity (from a pre-trained embedding) and search (via fast nearest neighbor retrieval from a knowledge database), we build on well-established components to construct a simple and flexible visual memory that has the following key capabilities: (1.) The ability to flexibly add data across scales: from individual samples all the way to entire classes and billion-scale data; (2.) The ability to remove data through unlearning and memory pruning; (3.) An interpretable decision-mechanism on which we can intervene to control its behavior. Taken together, these capabilities comprehensively demonstrate the benefits of an explicit visual memory. We hope that it might contribute to a conversation on how knowledge should be represented in deep vision models -- beyond carving it in "stone" weights.
>
---
#### [replaced 065] LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.09864v3](http://arxiv.org/pdf/2406.09864v3)**

> **作者:** Grigor Bezirganyan; Sana Sellami; Laure Berti-Équille; Sébastien Fournier
>
> **备注:** SIGIR 2025
>
> **摘要:** Multimodal Deep Learning enhances decision-making by integrating diverse information sources, such as texts, images, audio, and videos. To develop trustworthy multimodal approaches, it is essential to understand how uncertainty impacts these models. We propose LUMA, a unique multimodal dataset, featuring audio, image, and textual data from 50 classes, specifically designed for learning from uncertain data. It extends the well-known CIFAR 10/100 dataset with audio samples extracted from three audio corpora, and text data generated using the Gemma-7B Large Language Model (LLM). The LUMA dataset enables the controlled injection of varying types and degrees of uncertainty to achieve and tailor specific experiments and benchmarking initiatives. LUMA is also available as a Python package including the functions for generating multiple variants of the dataset with controlling the diversity of the data, the amount of noise for each modality, and adding out-of-distribution samples. A baseline pre-trained model is also provided alongside three uncertainty quantification methods: Monte-Carlo Dropout, Deep Ensemble, and Reliable Conflictive Multi-View Learning. This comprehensive dataset and its tools are intended to promote and support the development, evaluation, and benchmarking of trustworthy and robust multimodal deep learning approaches. We anticipate that the LUMA dataset will help the research community to design more trustworthy and robust machine learning approaches for safety critical applications. The code and instructions for downloading and processing the dataset can be found at: https://github.com/bezirganyan/LUMA/ .
>
---
#### [replaced 066] HERMES: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.14729v3](http://arxiv.org/pdf/2501.14729v3)**

> **作者:** Xin Zhou; Dingkang Liang; Sifan Tu; Xiwu Chen; Yikang Ding; Dingyuan Zhang; Feiyang Tan; Hengshuang Zhao; Xiang Bai
>
> **备注:** Accepted by ICCV 2025. The code is available at https://github.com/LMD0311/HERMES
>
> **摘要:** Driving World Models (DWMs) have become essential for autonomous driving by enabling future scene prediction. However, existing DWMs are limited to scene generation and fail to incorporate scene understanding, which involves interpreting and reasoning about the driving environment. In this paper, we present a unified Driving World Model named HERMES. We seamlessly integrate 3D scene understanding and future scene evolution (generation) through a unified framework in driving scenarios. Specifically, HERMES leverages a Bird's-Eye View (BEV) representation to consolidate multi-view spatial information while preserving geometric relationships and interactions. We also introduce world queries, which incorporate world knowledge into BEV features via causal attention in the Large Language Model, enabling contextual enrichment for understanding and generation tasks. We conduct comprehensive studies on nuScenes and OmniDrive-nuScenes datasets to validate the effectiveness of our method. HERMES achieves state-of-the-art performance, reducing generation error by 32.4% and improving understanding metrics such as CIDEr by 8.0%. The model and code will be publicly released at https://github.com/LMD0311/HERMES.
>
---
#### [replaced 067] SpectralEarth: Training Hyperspectral Foundation Models at Scale
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.08447v2](http://arxiv.org/pdf/2408.08447v2)**

> **作者:** Nassim Ait Ali Braham; Conrad M Albrecht; Julien Mairal; Jocelyn Chanussot; Yi Wang; Xiao Xiang Zhu
>
> **摘要:** Foundation models have triggered a paradigm shift in computer vision and are increasingly being adopted in remote sensing, particularly for multispectral imagery. Yet, their potential in hyperspectral imaging (HSI) remains untapped due to the absence of comprehensive and globally representative hyperspectral datasets. To close this gap, we introduce SpectralEarth, a large-scale multitemporal dataset designed to pretrain hyperspectral foundation models leveraging data from the environmental mapping and analysis program (EnMAP). SpectralEarth comprises 538 974 image patches covering 415 153 unique locations from 11 636 globally distributed EnMAP scenes spanning two years of archive. In addition, 17.5% of these locations include multiple timestamps, enabling multitemporal HSI analysis. Utilizing state-of-the-art self-supervised learning algorithms, we pretrain a series of foundation models on SpectralEarth, integrating a spectral adapter into classical vision backbones to accommodate the unique characteristics of HSI. In tandem, we construct nine downstream datasets for land-cover, crop-type mapping, and tree-species classification, providing benchmarks for model evaluation. Experimental results support the versatility of our models and their generalizability across different tasks and sensors. We also highlight computational efficiency during model fine-tuning.
>
---
#### [replaced 068] PiT: Progressive Diffusion Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13219v4](http://arxiv.org/pdf/2505.13219v4)**

> **作者:** Jiafu Wu; Yabiao Wang; Jian Li; Jinlong Peng; Yun Cao; Chengjie Wang; Jiangning Zhang
>
> **摘要:** Diffusion Transformers (DiTs) achieve remarkable performance within image generation via the transformer architecture. Conventionally, DiTs are constructed by stacking serial isotropic global modeling transformers, which face significant quadratic computational cost. However, through empirical analysis, we find that DiTs do not rely as heavily on global information as previously believed. In fact, most layers exhibit significant redundancy in global computation. Additionally, conventional attention mechanisms suffer from low-frequency inertia, limiting their efficiency. To address these issues, we propose Pseudo Shifted Window Attention (PSWA), which fundamentally mitigates global attention redundancy. PSWA achieves moderate global-local information through window attention. It further utilizes a high-frequency bridging branch to simulate shifted window operations, which both enrich the high-frequency information and strengthen inter-window connections. Furthermore, we propose the Progressive Coverage Channel Allocation (PCCA) strategy that captures high-order attention without additional computational cost. Based on these innovations, we propose a series of Pseudo Progressive Diffusion Transformer (PiT). Our extensive experiments show their superior performance; for example, our proposed PiT-L achieves 54% FID improvement over DiT-XL/2 while using less computation.
>
---
#### [replaced 069] Are you Struggling? Dataset and Baselines for Struggle Determination in Assembly Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.11057v5](http://arxiv.org/pdf/2402.11057v5)**

> **作者:** Shijia Feng; Michael Wray; Brian Sullivan; Youngkyoon Jang; Casimir Ludwig; Iain Gilchrist; Walterio Mayol-Cuevas
>
> **备注:** Accepted by International Journal of Computer Vision (IJCV, 2025)
>
> **摘要:** Determining when people are struggling allows for a finer-grained understanding of actions that complements conventional action classification and error detection. Struggle detection, as defined in this paper, is a distinct and important task that can be identified without explicit step or activity knowledge. We introduce the first struggle dataset with three real-world problem-solving activities that are labelled by both expert and crowd-source annotators. Video segments were scored w.r.t. their level of struggle using a forced choice 4-point scale. This dataset contains 5.1 hours of video from 73 participants. We conducted a series of experiments to identify the most suitable modelling approaches for struggle determination. Additionally, we compared various deep learning models, establishing baseline results for struggle classification, struggle regression, and struggle label distribution learning. Our results indicate that struggle detection in video can achieve up to $88.24\%$ accuracy in binary classification, while detecting the level of struggle in a four-way classification setting performs lower, with an overall accuracy of $52.45\%$. Our work is motivated toward a more comprehensive understanding of action in video and potentially the improvement of assistive systems that analyse struggle and can better support users during manual activities.
>
---
#### [replaced 070] Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12207v3](http://arxiv.org/pdf/2505.12207v3)**

> **作者:** Qingmei Li; Yang Zhang; Zurong Mai; Yuhang Chen; Shuohong Lou; Henglian Huang; Jiarui Zhang; Zhiwei Zhang; Yibin Wen; Weijia Li; Haohuan Fu; Jianxi Huang; Juepeng Zheng
>
> **摘要:** Large Multimodal Models (LMMs) has demonstrated capabilities across various domains, but comprehensive benchmarks for agricultural remote sensing (RS) remain scarce. Existing benchmarks designed for agricultural RS scenarios exhibit notable limitations, primarily in terms of insufficient scene diversity in the dataset and oversimplified task design. To bridge this gap, we introduce AgroMind, a comprehensive agricultural remote sensing benchmark covering four task dimensions: spatial perception, object understanding, scene understanding, and scene reasoning, with a total of 13 task types, ranging from crop identification and health monitoring to environmental analysis. We curate a high-quality evaluation set by integrating eight public datasets and one private farmland plot dataset, containing 27,247 QA pairs and 19,615 images. The pipeline begins with multi-source data pre-processing, including collection, format standardization, and annotation refinement. We then generate a diverse set of agriculturally relevant questions through the systematic definition of tasks. Finally, we employ LMMs for inference, generating responses, and performing detailed examinations. We evaluated 20 open-source LMMs and 4 closed-source models on AgroMind. Experiments reveal significant performance gaps, particularly in spatial reasoning and fine-grained recognition, it is notable that human performance lags behind several leading LMMs. By establishing a standardized evaluation framework for agricultural RS, AgroMind reveals the limitations of LMMs in domain knowledge and highlights critical challenges for future work. Data and code can be accessed at https://rssysu.github.io/AgroMind/.
>
---
#### [replaced 071] Pediatric brain tumor classification using digital histopathology and deep learning: evaluation of SOTA methods on a multi-center Swedish cohort
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.01330v2](http://arxiv.org/pdf/2409.01330v2)**

> **作者:** Iulian Emil Tampu; Per Nyman; Christoforos Spyretos; Ida Blystad; Alia Shamikh; Gabriela Prochazka; Teresita Díaz de Ståhl; Johanna Sandgren; Peter Lundberg; Neda Haj-Hosseini
>
> **摘要:** Brain tumors are the most common solid tumors in children and young adults, but the scarcity of large histopathology datasets has limited the application of computational pathology in this group. This study implements two weakly supervised multiple-instance learning (MIL) approaches on patch-features obtained from state-of-the-art histology-specific foundation models to classify pediatric brain tumors in hematoxylin and eosin whole slide images (WSIs) from a multi-center Swedish cohort. WSIs from 540 subjects (age 8.5$\pm$4.9 years) diagnosed with brain tumor were gathered from the six Swedish university hospitals. Instance (patch)-level features were obtained from WSIs using three pre-trained feature extractors: ResNet50, UNI, and CONCH. Instances were aggregated using attention-based MIL (ABMIL) or clustering-constrained attention MIL (CLAM) for patient-level classification. Models were evaluated on three classification tasks based on the hierarchical classification of pediatric brain tumors: tumor category, family, and type. Model generalization was assessed by training on data from two of the centers and testing on data from four other centers. Model interpretability was evaluated through attention mapping. The highest classification performance was achieved using UNI features and ABMIL aggregation, with Matthew's correlation coefficient of 0.76$\pm$0.04, 0.63$\pm$0.04, and 0.60$\pm$0.05 for tumor category, family, and type classification, respectively. When evaluating generalization, models utilizing UNI and CONCH features outperformed those using ResNet50. However, the drop in performance from the in-site to out-of-site testing was similar across feature extractors. These results show the potential of state-of-the-art computational pathology methods in diagnosing pediatric brain tumors at different hierarchical levels with fair generalizability on a multi-center national dataset.
>
---
#### [replaced 072] Modulate and Reconstruct: Learning Hyperspectral Imaging from Misaligned Smartphone Views
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01835v2](http://arxiv.org/pdf/2507.01835v2)**

> **作者:** Daniil Reutsky; Daniil Vladimirov; Yasin Mamedov; Georgy Perevozchikov; Nancy Mehta; Egor Ershov; Radu Timofte
>
> **摘要:** Hyperspectral reconstruction (HSR) from RGB images is a fundamentally ill-posed problem due to severe spectral information loss. Existing approaches typically rely on a single RGB image, limiting reconstruction accuracy. In this work, we propose a novel multi-image-to-hyperspectral reconstruction (MI-HSR) framework that leverages a triple-camera smartphone system, where two lenses are equipped with carefully selected spectral filters. Our configuration, grounded in theoretical and empirical analysis, enables richer and more diverse spectral observations than conventional single-camera setups. To support this new paradigm, we introduce Doomer, the first dataset for MI-HSR, comprising aligned images from three smartphone cameras and a hyperspectral reference camera across diverse scenes. We show that the proposed HSR model achieves consistent improvements over existing methods on the newly proposed benchmark. In a nutshell, our setup allows 30% towards more accurately estimated spectra compared to an ordinary RGB camera. Our findings suggest that multi-view spectral filtering with commodity hardware can unlock more accurate and practical hyperspectral imaging solutions.
>
---
#### [replaced 073] Audio-3DVG: Unified Audio -- Point Cloud Fusion for 3D Visual Grounding
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00669v2](http://arxiv.org/pdf/2507.00669v2)**

> **作者:** Duc Cao-Dinh; Khai Le-Duc; Anh Dao; Bach Phan Tat; Chris Ngo; Duy M. H. Nguyen; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** Preprint, 51 pages
>
> **摘要:** 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce (i) Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an (ii) Audio-Guided Attention module that models the interactions between target candidates and mentioned objects, enhancing discrimination in cluttered 3D environments. To support benchmarking, we (iii) synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods, highlight the promise of integrating spoken language into 3D vision tasks.
>
---
#### [replaced 074] DRWKV: Focusing on Object Edges for Low-Light Image Enhancement
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18594v2](http://arxiv.org/pdf/2507.18594v2)**

> **作者:** Xuecheng Bai; Yuxiang Wang; Boyu Hu; Qinyuan Jie; Chuanzhi Xu; Hongru Xiao; Kechen Li; Vera Chung
>
> **摘要:** Low-light image enhancement remains a challenging task, particularly in preserving object edge continuity and fine structural details under extreme illumination degradation. In this paper, we propose a novel model, DRWKV (Detailed Receptance Weighted Key Value), which integrates our proposed Global Edge Retinex (GER) theory, enabling effective decoupling of illumination and edge structures for enhanced edge fidelity. Secondly, we introduce Evolving WKV Attention, a spiral-scanning mechanism that captures spatial edge continuity and models irregular structures more effectively. Thirdly, we design the Bilateral Spectrum Aligner (Bi-SAB) and a tailored MS2-Loss to jointly align luminance and chrominance features, improving visual naturalness and mitigating artifacts. Extensive experiments on five LLIE benchmarks demonstrate that DRWKV achieves leading performance in PSNR, SSIM, and NIQE while maintaining low computational complexity. Furthermore, DRWKV enhances downstream performance in low-light multi-object tracking tasks, validating its generalization capabilities.
>
---
#### [replaced 075] Yan: Foundational Interactive Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08601v2](http://arxiv.org/pdf/2508.08601v2)**

> **作者:** Deheng Ye; Fangyun Zhou; Jiacheng Lv; Jianqi Ma; Jun Zhang; Junyan Lv; Junyou Li; Minwen Deng; Mingyu Yang; Qiang Fu; Wei Yang; Wenkai Lv; Yangbin Yu; Yewen Wang; Yonghang Guan; Zhihao Hu; Zhongbin Fang; Zhongqian Sun
>
> **摘要:** We present Yan, a foundational framework for interactive video generation, covering the entire pipeline from simulation and generation to editing. Specifically, Yan comprises three core modules. AAA-level Simulation: We design a highly-compressed, low-latency 3D-VAE coupled with a KV-cache-based shift-window denoising inference process, achieving real-time 1080P/60FPS interactive simulation. Multi-Modal Generation: We introduce a hierarchical autoregressive caption method that injects game-specific knowledge into open-domain multi-modal video diffusion models (VDMs), then transforming the VDM into a frame-wise, action-controllable, real-time infinite interactive video generator. Notably, when the textual and visual prompts are sourced from different domains, the model demonstrates strong generalization, allowing it to blend and compose the style and mechanics across domains flexibly according to user prompts. Multi-Granularity Editing: We propose a hybrid model that explicitly disentangles interactive mechanics simulation from visual rendering, enabling multi-granularity video content editing during interaction through text. Collectively, Yan offers an integration of these modules, pushing interactive video generation beyond isolated capabilities toward a comprehensive AI-driven interactive creation paradigm, paving the way for the next generation of creative tools, media, and entertainment. The project page is: https://greatx3.github.io/Yan/.
>
---
#### [replaced 076] GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06866v2](http://arxiv.org/pdf/2504.06866v2)**

> **作者:** Seunghyeok Back; Joosoon Lee; Kangmin Kim; Heeseon Rho; Geonhyup Lee; Raeyoung Kang; Sangbeom Lee; Sangjun Noh; Youngjin Lee; Taeyeop Lee; Kyoobin Lee
>
> **摘要:** Robust grasping in cluttered environments remains an open challenge in robotics. While benchmark datasets have significantly advanced deep learning methods, they mainly focus on simplistic scenes with light occlusion and insufficient diversity, limiting their applicability to practical scenarios. We present GraspClutter6D, a large-scale real-world grasping dataset featuring: (1) 1,000 highly cluttered scenes with dense arrangements (14.1 objects/scene, 62.6\% occlusion), (2) comprehensive coverage across 200 objects in 75 environment configurations (bins, shelves, and tables) captured using four RGB-D cameras from multiple viewpoints, and (3) rich annotations including 736K 6D object poses and 9.3B feasible robotic grasps for 52K RGB-D images. We benchmark state-of-the-art segmentation, object pose estimation, and grasp detection methods to provide key insights into challenges in cluttered environments. Additionally, we validate the dataset's effectiveness as a training resource, demonstrating that grasping networks trained on GraspClutter6D significantly outperform those trained on existing datasets in both simulation and real-world experiments. The dataset, toolkit, and annotation tools are publicly available on our project website: https://sites.google.com/view/graspclutter6d.
>
---
#### [replaced 077] HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21809v2](http://arxiv.org/pdf/2507.21809v2)**

> **作者:** HunyuanWorld Team; Zhenwei Wang; Yuhao Liu; Junta Wu; Zixiao Gu; Haoyuan Wang; Xuhui Zuo; Tianyu Huang; Wenhuan Li; Sheng Zhang; Yihang Lian; Yulin Tsai; Lifu Wang; Sicong Liu; Puhua Jiang; Xianghui Yang; Dongyuan Guo; Yixuan Tang; Xinyue Mao; Jiaao Yu; Junlin Yu; Jihong Zhang; Meng Chen; Liang Dong; Yiwen Jia; Chao Zhang; Yonghao Tan; Hao Zhang; Zheng Ye; Peng He; Runzhou Wu; Minghui Chen; Zhan Li; Wangchen Qin; Lei Wang; Yifu Sun; Lin Niu; Xiang Yuan; Xiaofeng Yang; Yingping He; Jie Xiao; Yangyu Tao; Jianchen Zhu; Jinbao Xue; Kai Liu; Chongqing Zhao; Xinming Wu; Tian Liu; Peng Chen; Di Wang; Yuhong Liu; Linus; Jie Jiang; Tengfei Wang; Chunchao Guo
>
> **备注:** Technical Report; Project Page: https://3d-models.hunyuan.tencent.com/world/
>
> **摘要:** Creating immersive and playable 3D worlds from texts or images remains a fundamental challenge in computer vision and graphics. Existing world generation approaches typically fall into two categories: video-based methods that offer rich diversity but lack 3D consistency and rendering efficiency, and 3D-based methods that provide geometric consistency but struggle with limited training data and memory-inefficient representations. To address these limitations, we present HunyuanWorld 1.0, a novel framework that combines the best of both worlds for generating immersive, explorable, and interactive 3D scenes from text and image conditions. Our approach features three key advantages: 1) 360{\deg} immersive experiences via panoramic world proxies; 2) mesh export capabilities for seamless compatibility with existing computer graphics pipelines; 3) disentangled object representations for augmented interactivity. The core of our framework is a semantically layered 3D mesh representation that leverages panoramic images as 360{\deg} world proxies for semantic-aware world decomposition and reconstruction, enabling the generation of diverse 3D worlds. Extensive experiments demonstrate that our method achieves state-of-the-art performance in generating coherent, explorable, and interactive 3D worlds while enabling versatile applications in virtual reality, physical simulation, game development, and interactive content creation.
>
---
#### [replaced 078] On the Reliability of Vision-Language Models Under Adversarial Frequency-Domain Perturbations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22398v3](http://arxiv.org/pdf/2507.22398v3)**

> **作者:** Jordan Vice; Naveed Akhtar; Yansong Gao; Richard Hartley; Ajmal Mian
>
> **备注:** Keywords: Vision-Language Models, Frequency-Domain Perturbations, Adversarial Robustness, Image Authenticity, Reliability
>
> **摘要:** Vision-Language Models (VLMs) are increasingly used as perceptual modules for visual content reasoning, including through captioning and DeepFake detection. In this work, we expose a critical vulnerability of VLMs when exposed to subtle, structured perturbations in the frequency domain. Specifically, we highlight how these feature transformations undermine authenticity/DeepFake detection and automated image captioning tasks. We design targeted image transformations, operating in the frequency domain to systematically adjust VLM outputs when exposed to frequency-perturbed real and synthetic images. We demonstrate that the perturbation injection method generalizes across five state-of-the-art VLMs which includes different-parameter Qwen2/2.5 and BLIP models. Experimenting across ten real and generated image datasets reveals that VLM judgments are sensitive to frequency-based cues and may not wholly align with semantic content. Crucially, we show that visually-imperceptible spatial frequency transformations expose the fragility of VLMs deployed for automated image captioning and authenticity detection tasks. Our findings under realistic, black-box constraints challenge the reliability of VLMs, underscoring the need for robust multimodal perception systems.
>
---
#### [replaced 079] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01105v3](http://arxiv.org/pdf/2502.01105v3)**

> **作者:** Yiren Song; Danze Chen; Mike Zheng Shou
>
> **摘要:** Generating cognitive-aligned layered SVGs remains challenging due to existing methods' tendencies toward either oversimplified single-layer outputs or optimization-induced shape redundancies. We propose LayerTracer, a diffusion transformer based framework that bridges this gap by learning designers' layered SVG creation processes from a novel dataset of sequential design operations. Our approach operates in two phases: First, a text-conditioned DiT generates multi-phase rasterized construction blueprints that simulate human design workflows. Second, layer-wise vectorization with path deduplication produces clean, editable SVGs. For image vectorization, we introduce a conditional diffusion mechanism that encodes reference images into latent tokens, guiding hierarchical reconstruction while preserving structural integrity. Extensive experiments demonstrate LayerTracer's superior performance against optimization-based and neural baselines in both generation quality and editability, effectively aligning AI-generated vectors with professional design cognition.
>
---
#### [replaced 080] ViCToR: Improving Visual Comprehension via Token Reconstruction for Pretraining LMMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.14332v4](http://arxiv.org/pdf/2410.14332v4)**

> **作者:** Yin Xie; Kaicheng Yang; Peirou Liang; Xiang An; Yongle Zhao; Yumeng Wang; Ziyong Feng; Roy Miles; Ismail Elezi; Jiankang Deng
>
> **备注:** 10 pages, 6 figures, 5 tables
>
> **摘要:** Large Multimodal Models (LMMs) often face a modality representation gap during pretraining: while language embeddings remain stable, visual representations are highly sensitive to contextual noise (e.g., background clutter). To address this issue, we introduce a visual comprehension stage, which we call ViCToR (Visual Comprehension via Token Reconstruction), a novel pretraining framework for LMMs. ViCToR employs a learnable visual token pool and utilizes the Hungarian matching algorithm to select semantically relevant tokens from this pool for visual token replacement. Furthermore, by integrating a visual token reconstruction loss with dense semantic supervision, ViCToR can learn tokens which retain high visual detail, thereby enhancing the large language model's (LLM's) understanding of visual information. After pretraining on 3 million publicly accessible images and captions, ViCToR achieves state-of-the-art results, improving over LLaVA-NeXT-8B by 10.4%, 3.2%, and 7.2% on the MMStar, SEED$^I$, and RealWorldQA benchmarks, respectively. Code is available at https://github.com/deepglint/Victor.
>
---
#### [replaced 081] SWA-SOP: Spatially-aware Window Attention for Semantic Occupancy Prediction in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18785v2](http://arxiv.org/pdf/2506.18785v2)**

> **作者:** Helin Cao; Rafael Materla; Sven Behnke
>
> **备注:** 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Vienna, Austria, Oct 2025
>
> **摘要:** Perception systems in autonomous driving rely on sensors such as LiDAR and cameras to perceive the 3D environment. However, due to occlusions and data sparsity, these sensors often fail to capture complete information. Semantic Occupancy Prediction (SOP) addresses this challenge by inferring both occupancy and semantics of unobserved regions. Existing transformer-based SOP methods lack explicit modeling of spatial structure in attention computation, resulting in limited geometric awareness and poor performance in sparse or occluded areas. To this end, we propose Spatially-aware Window Attention (SWA), a novel mechanism that incorporates local spatial context into attention. SWA significantly improves scene completion and achieves state-of-the-art results on LiDAR-based SOP benchmarks. We further validate its generality by integrating SWA into a camera-based SOP pipeline, where it also yields consistent gains across modalities.
>
---
#### [replaced 082] Emotion-Qwen: A Unified Framework for Emotion and Vision Understanding
- **分类: cs.MM; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06685v3](http://arxiv.org/pdf/2505.06685v3)**

> **作者:** Dawei Huang; Qing Li; Chuan Yan; Zebang Cheng; Zihao Han; Yurong Huang; Xiang Li; Bin Li; Xiaohui Wang; Zheng Lian; Zhi-Qi Cheng; Xiaojiang Peng
>
> **摘要:** Accurate emotion understanding in videos necessitates effectively recognizing and interpreting emotional states by integrating visual, textual, auditory, and contextual cues. Although recent Large Multimodal Models (LMMs) have exhibited significant progress in general vision-language (VL) tasks, their performance often deteriorates in emotion-specific scenarios, exhibiting catastrophic forgetting when fine-tuned on emotion-centric tasks. To overcome these limitations, we propose Emotion-Qwen, a unified multimodal framework designed to simultaneously enable robust emotion understanding and preserve general VL reasoning capabilities. Emotion-Qwen introduces a novel Hybrid Compressor based on a Mixture-of-Experts (MoE) architecture, dynamically routing inputs to optimally balance emotion-specific processing and general multimodal reasoning. We further propose a carefully structured three-stage pre-training pipeline, leveraging extensive general and emotion-focused datasets to strengthen multimodal representation robustness and model adaptability. Additionally, we develop the Video Emotion Reasoning (VER) dataset, a large-scale bilingual resource containing over 40K video clips annotated with detailed context-aware emotional descriptions, significantly facilitating research on fine-grained emotional reasoning. Extensive experiments confirm that Emotion-Qwen achieves state-of-the-art performance across multiple emotion recognition and reasoning benchmarks, while maintaining highly competitive results in general VL tasks.
>
---
#### [replaced 083] Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13265v2](http://arxiv.org/pdf/2506.13265v2)**

> **作者:** Rohit Mohan; Julia Hindel; Florian Drews; Claudius Gläser; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods.
>
---
#### [replaced 084] CoherenDream: Boosting Holistic Text Coherence in 3D Generation via Multimodal Large Language Models Feedback
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19860v3](http://arxiv.org/pdf/2504.19860v3)**

> **作者:** Chenhan Jiang; Yihan Zeng; Dit-Yan Yeung
>
> **摘要:** Score Distillation Sampling (SDS) has achieved remarkable success in text-to-3D content generation. However, SDS-based methods struggle to maintain semantic fidelity for user prompts, particularly when involving multiple objects with intricate interactions. While existing approaches often address 3D consistency through multiview diffusion model fine-tuning on 3D datasets, this strategy inadvertently exacerbates text-3D alignment degradation. The limitation stems from SDS's inherent accumulation of view-independent biases during optimization, which progressively diverges from the ideal text alignment direction. To alleviate this limitation, we propose a novel SDS objective, dubbed as Textual Coherent Score Distillation (TCSD), which integrates alignment feedback from multimodal large language models (MLLMs). Our TCSD leverages cross-modal understanding capabilities of MLLMs to assess and guide the text-3D correspondence during the optimization. We further develop 3DLLaVA-CRITIC - a fine-tuned MLLM specialized for evaluating multiview text alignment in 3D generations. Additionally, we introduce an LLM-layout initialization that significantly accelerates optimization convergence through semantic-aware spatial configuration. Our framework, CoherenDream, achieves consistent improvement across multiple metrics on TIFA subset.As the first study to incorporate MLLMs into SDS optimization, we also conduct extensive ablation studies to explore optimal MLLM adaptations for 3D generation tasks.
>
---
