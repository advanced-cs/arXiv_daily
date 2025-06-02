# 计算机视觉 cs.CV

- **最新发布 177 篇**

- **更新 86 篇**

## 最新发布

#### [new 001] ControlTac: Force- and Position-Controlled Tactile Data Augmentation with a Single Reference Image
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于触觉数据增强任务，旨在解决触觉数据采集成本高及现有生成方法不真实、迁移性差的问题。提出ControlTac框架，通过单参考触觉图像、接触力和位置物理先验，生成真实可控的触觉图像，提升下游任务效果。实验验证其数据增强的有效性。**

- **链接: [http://arxiv.org/pdf/2505.20498v1](http://arxiv.org/pdf/2505.20498v1)**

> **作者:** Dongyu Luo; Kelin Yu; Amir-Hossein Shahidzadeh; Cornelia Fermüller; Yiannis Aloimonos
>
> **备注:** 22 pages, 11 figures, 7 tables
>
> **摘要:** Vision-based tactile sensing has been widely used in perception, reconstruction, and robotic manipulation. However, collecting large-scale tactile data remains costly due to the localized nature of sensor-object interactions and inconsistencies across sensor instances. Existing approaches to scaling tactile data, such as simulation and free-form tactile generation, often suffer from unrealistic output and poor transferability to downstream tasks.To address this, we propose ControlTac, a two-stage controllable framework that generates realistic tactile images conditioned on a single reference tactile image, contact force, and contact position. With those physical priors as control input, ControlTac generates physically plausible and varied tactile images that can be used for effective data augmentation. Through experiments on three downstream tasks, we demonstrate that ControlTac can effectively augment tactile datasets and lead to consistent gains. Our three real-world experiments further validate the practical utility of our approach. Project page: https://dongyuluo.github.io/controltac.
>
---
#### [new 002] VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving Visual-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于具身视觉追踪（EVT）任务，解决现有系统在动态环境中的跟踪失败恢复问题。提出结合视觉语言模型（VLM）与主动跟踪方法的自改进框架，通过记忆增强的自我反思机制提升3D空间推理能力，实验显示较传统方法成功率显著提升。**

- **链接: [http://arxiv.org/pdf/2505.20718v1](http://arxiv.org/pdf/2505.20718v1)**

> **作者:** Kui Wu; Shuhang Xu; Hao Chen; Churan Wang; Zhoujun Li; Yizhou Wang; Fangwei Zhong
>
> **摘要:** We introduce a novel self-improving framework that enhances Embodied Visual Tracking (EVT) with Visual-Language Models (VLMs) to address the limitations of current active visual tracking systems in recovering from tracking failure. Our approach combines the off-the-shelf active tracking methods with VLMs' reasoning capabilities, deploying a fast visual policy for normal tracking and activating VLM reasoning only upon failure detection. The framework features a memory-augmented self-reflection mechanism that enables the VLM to progressively improve by learning from past experiences, effectively addressing VLMs' limitations in 3D spatial reasoning. Experimental results demonstrate significant performance improvements, with our framework boosting success rates by $72\%$ with state-of-the-art RL-based approaches and $220\%$ with PID-based methods in challenging environments. This work represents the first integration of VLM-based reasoning to assist EVT agents in proactive failure recovery, offering substantial advances for real-world robotic applications that require continuous target monitoring in dynamic, unstructured environments. Project website: https://sites.google.com/view/evt-recovery-assistant.
>
---
#### [new 003] Intern-GS: Vision Model Guided Sparse-View 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于稀疏视角3D场景重建任务，解决因观测数据有限导致的传统方法重建质量差的问题。提出Intern-GS方法，利用视觉基础模型引导3D高斯散射的初始化（通过DUSt3R生成高效点云）与优化（预测未观测区域的深度和外观），提升重建精度，实验显示达 state-of-the-art效果。**

- **链接: [http://arxiv.org/pdf/2505.20729v1](http://arxiv.org/pdf/2505.20729v1)**

> **作者:** Xiangyu Sun; Runnan Chen; Mingming Gong; Dong Xu; Tongliang Liu
>
> **摘要:** Sparse-view scene reconstruction often faces significant challenges due to the constraints imposed by limited observational data. These limitations result in incomplete information, leading to suboptimal reconstructions using existing methodologies. To address this, we present Intern-GS, a novel approach that effectively leverages rich prior knowledge from vision foundation models to enhance the process of sparse-view Gaussian Splatting, thereby enabling high-quality scene reconstruction. Specifically, Intern-GS utilizes vision foundation models to guide both the initialization and the optimization process of 3D Gaussian splatting, effectively addressing the limitations of sparse inputs. In the initialization process, our method employs DUSt3R to generate a dense and non-redundant gaussian point cloud. This approach significantly alleviates the limitations encountered by traditional structure-from-motion (SfM) methods, which often struggle under sparse-view constraints. During the optimization process, vision foundation models predict depth and appearance for unobserved views, refining the 3D Gaussians to compensate for missing information in unseen regions. Extensive experiments demonstrate that Intern-GS achieves state-of-the-art rendering quality across diverse datasets, including both forward-facing and large-scale scenes, such as LLFF, DTU, and Tanks and Temples.
>
---
#### [new 004] Robust Video-Based Pothole Detection and Area Estimation for Intelligent Vehicles with Depth Map and Kalman Smoothing
- **分类: cs.CV**

- **简介: 该论文提出基于视频的鲁棒坑洞检测与面积估计框架，解决现有方法受相机角度及平坦路面假设限制的问题。通过改进YOLOv8（ACSH-YOLOv8）增强小坑洞检测，结合深度图、BoT-SORT跟踪及CDKF优化，提升估计精度与连续性。**

- **链接: [http://arxiv.org/pdf/2505.21049v1](http://arxiv.org/pdf/2505.21049v1)**

> **作者:** Dehao Wang; Haohang Zhu; Yiwen Xu; Kaiqi Liu
>
> **摘要:** Road potholes pose a serious threat to driving safety and comfort, making their detection and assessment a critical task in fields such as autonomous driving. When driving vehicles, the operators usually avoid large potholes and approach smaller ones at reduced speeds to ensure safety. Therefore, accurately estimating pothole area is of vital importance. Most existing vision-based methods rely on distance priors to construct geometric models. However, their performance is susceptible to variations in camera angles and typically relies on the assumption of a flat road surface, potentially leading to significant errors in complex real-world environments. To address these problems, a robust pothole area estimation framework that integrates object detection and monocular depth estimation in a video stream is proposed in this paper. First, to enhance pothole feature extraction and improve the detection of small potholes, ACSH-YOLOv8 is proposed with ACmix module and the small object detection head. Then, the BoT-SORT algorithm is utilized for pothole tracking, while DepthAnything V2 generates depth maps for each frame. With the obtained depth maps and potholes labels, a novel Minimum Bounding Triangulated Pixel (MBTP) method is proposed for pothole area estimation. Finally, Kalman Filter based on Confidence and Distance (CDKF) is developed to maintain consistency of estimation results across consecutive frames. The results show that ACSH-YOLOv8 model achieves an AP(50) of 76.6%, representing a 7.6% improvement over YOLOv8. Through CDKF optimization across consecutive frames, pothole predictions become more robust, thereby enhancing the method's practical applicability.
>
---
#### [new 005] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于Wild Test-Time Adaptation（WTTA）任务，针对传统熵最小化方法优化低效、适应能力受限的问题，提出ReCAP方法。通过概率区域建模捕捉语义变化，并采用渐近近似将区域置信度转化为可计算的代理，提升模型在数据稀缺场景下的跨域适应效率，实验验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.20704v1](http://arxiv.org/pdf/2505.20704v1)**

> **作者:** Zixuan Hu; Yichun Hu; Xiaotong Li; Shixiang Tang; Ling-Yu Duan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Wild Test-Time Adaptation (WTTA) is proposed to adapt a source model to unseen domains under extreme data scarcity and multiple shifts. Previous approaches mainly focused on sample selection strategies, while overlooking the fundamental problem on underlying optimization. Initially, we critically analyze the widely-adopted entropy minimization framework in WTTA and uncover its significant limitations in noisy optimization dynamics that substantially hinder adaptation efficiency. Through our analysis, we identify region confidence as a superior alternative to traditional entropy, however, its direct optimization remains computationally prohibitive for real-time applications. In this paper, we introduce a novel region-integrated method ReCAP that bypasses the lengthy process. Specifically, we propose a probabilistic region modeling scheme that flexibly captures semantic changes in embedding space. Subsequently, we develop a finite-to-infinite asymptotic approximation that transforms the intractable region confidence into a tractable and upper-bounded proxy. These innovations significantly unlock the overlooked potential dynamics in local region in a concise solution. Our extensive experiments demonstrate the consistent superiority of ReCAP over existing methods across various datasets and wild scenarios.
>
---
#### [new 006] Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航（VLN）任务，旨在解决现有方法依赖视觉合成导致的高计算成本和冗余问题。提出Adaptive Text Dreamer（ATD），基于LLM构建左右脑架构：左脑逻辑整合指令，右脑预测未来场景语义，通过轻量微调和交叉交互机制结合导航模型，实现高效精准导航，在R2R数据集达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.20897v1](http://arxiv.org/pdf/2505.20897v1)**

> **作者:** Pingrui Zhang; Yifei Su; Pengyuan Wu; Dong An; Li Zhang; Zhigang Wang; Dong Wang; Yan Ding; Bin Zhao; Xuelong Li
>
> **摘要:** Vision-and-Language Navigation (VLN) requires the agent to navigate by following natural instructions under partial observability, making it difficult to align perception with language. Recent methods mitigate this by imagining future scenes, yet they rely on vision-based synthesis, leading to high computational cost and redundant details. To this end, we propose to adaptively imagine key environmental semantics via \textit{language} form, enabling a more reliable and efficient strategy. Specifically, we introduce a novel Adaptive Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a large language model (LLM). ATD is designed with a human-like left-right brain architecture, where the left brain focuses on logical integration, and the right brain is responsible for imaginative prediction of future scenes. To achieve this, we fine-tune only the Q-former within both brains to efficiently activate domain-specific knowledge in the LLM, enabling dynamic updates of logical reasoning and imagination during navigation. Furthermore, we introduce a cross-interaction mechanism to regularize the imagined outputs and inject them into a navigation expert module, allowing ATD to jointly exploit both the reasoning capacity of the LLM and the expertise of the navigation model. We conduct extensive experiments on the R2R benchmark, where ATD achieves state-of-the-art performance with fewer parameters. The code is \href{https://github.com/zhangpingrui/Adaptive-Text-Dreamer}{here}.
>
---
#### [new 007] MagicTryOn: Harnessing Diffusion Transformer for Garment-Preserving Video Virtual Try-on
- **分类: cs.CV**

- **简介: 该论文属于视频虚拟试穿任务，旨在解决现有方法在时空连续性、衣物细节保存及动态一致性上的不足。提出MagicTryOn框架，采用扩散Transformer替代U-Net，通过全自注意力建模时空连续性，并设计粗细结合的衣物保留策略（嵌入衣物标记、语义/纹理等条件）及掩码损失优化，提升合成效果的逼真度与稳定性。**

- **链接: [http://arxiv.org/pdf/2505.21325v1](http://arxiv.org/pdf/2505.21325v1)**

> **作者:** Guangyuan Li; Siming Zheng; Hao Zhang; Jinwei Chen; Junsheng Luan; Binkai Ou; Lei Zhao; Bo Li; Peng-Tao Jiang
>
> **摘要:** Video Virtual Try-On (VVT) aims to simulate the natural appearance of garments across consecutive video frames, capturing their dynamic variations and interactions with human body motion. However, current VVT methods still face challenges in terms of spatiotemporal consistency and garment content preservation. First, they use diffusion models based on the U-Net, which are limited in their expressive capability and struggle to reconstruct complex details. Second, they adopt a separative modeling approach for spatial and temporal attention, which hinders the effective capture of structural relationships and dynamic consistency across frames. Third, their expression of garment details remains insufficient, affecting the realism and stability of the overall synthesized results, especially during human motion. To address the above challenges, we propose MagicTryOn, a video virtual try-on framework built upon the large-scale video diffusion Transformer.We replace the U-Net architecture with a diffusion Transformer and combine full self-attention to jointly model the spatiotemporal consistency of videos. We design a coarse-to-fine garment preservation strategy. The coarse strategy integrates garment tokens during the embedding stage, while the fine strategy incorporates multiple garment-based conditions, such as semantics, textures, and contour lines during the denoising stage. Moreover, we introduce a mask-aware loss to further optimize garment region fidelity. Extensive experiments on both image and video try-on datasets demonstrate that our method outperforms existing SOTA methods in comprehensive evaluations and generalizes to in-the-wild scenarios.
>
---
#### [new 008] Normalized Attention Guidance: Universal Negative Guidance for Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出Normalized Attention Guidance（NAG），解决扩散模型中负向引导在少步采样时失效的问题。针对现有Classifier-Free Guidance（CFG）因正负分支预测发散导致效果下降的缺陷，NAG通过注意力空间的L1归一化与外推，在无需训练情况下实现跨架构/模态的通用负向引导，提升生成质量与文本对齐度。**

- **链接: [http://arxiv.org/pdf/2505.21179v1](http://arxiv.org/pdf/2505.21179v1)**

> **作者:** Dar-Yen Chen; Hmrishav Bandyopadhyay; Kai Zou; Yi-Zhe Song
>
> **摘要:** Negative guidance -- explicitly suppressing unwanted attributes -- remains a fundamental challenge in diffusion models, particularly in few-step sampling regimes. While Classifier-Free Guidance (CFG) works well in standard settings, it fails under aggressive sampling step compression due to divergent predictions between positive and negative branches. We present Normalized Attention Guidance (NAG), an efficient, training-free mechanism that applies extrapolation in attention space with L1-based normalization and refinement. NAG restores effective negative guidance where CFG collapses while maintaining fidelity. Unlike existing approaches, NAG generalizes across architectures (UNet, DiT), sampling regimes (few-step, multi-step), and modalities (image, video), functioning as a \textit{universal} plug-in with minimal computational overhead. Through extensive experimentation, we demonstrate consistent improvements in text alignment (CLIP Score), fidelity (FID, PFID), and human-perceived quality (ImageReward). Our ablation studies validate each design component, while user studies confirm significant preference for NAG-guided outputs. As a model-agnostic inference-time approach requiring no retraining, NAG provides effortless negative guidance for all modern diffusion frameworks -- pseudocode in the Appendix!
>
---
#### [new 009] Uni3D-MoE: Scalable Multimodal 3D Scene Understanding via Mixture of Experts
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决现有方法模态覆盖不全及统一处理导致精度不足的问题。提出Uni3D-MoE模型，整合多模态数据（多视角RGB/深度图、BEV、点云、体素），通过稀疏MoE的动态路由机制，按需选择专家处理不同模态token，提升场景表征与任务适应性。**

- **链接: [http://arxiv.org/pdf/2505.21079v1](http://arxiv.org/pdf/2505.21079v1)**

> **作者:** Yue Zhang; Yingzhao Jian; Hehe Fan; Yi Yang; Roger Zimmermann
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have demonstrated considerable potential for comprehensive 3D scene understanding. However, existing approaches typically utilize only one or a limited subset of 3D modalities, resulting in incomplete representations of 3D scenes and reduced interpretive accuracy. Furthermore, different types of queries inherently depend on distinct modalities, indicating that uniform processing of all modality tokens may fail to effectively capture query-specific context. To address these challenges, we propose Uni3D-MoE, a sparse Mixture-of-Experts (MoE)-based 3D MLLM designed to enable adaptive 3D multimodal fusion. Specifically, Uni3D-MoE integrates a comprehensive set of 3D modalities, including multi-view RGB and depth images, bird's-eye-view (BEV) maps, point clouds, and voxel representations. At its core, our framework employs a learnable routing mechanism within the sparse MoE-based large language model, dynamically selecting appropriate experts at the token level. Each expert specializes in processing multimodal tokens based on learned modality preferences, thus facilitating flexible collaboration tailored to diverse task-specific requirements. Extensive evaluations on standard 3D scene understanding benchmarks and specialized datasets demonstrate the efficacy of Uni3D-MoE.
>
---
#### [new 010] MultLFG: Training-free Multi-LoRA composition using Frequency-domain Guidance
- **分类: cs.CV**

- **简介: 该论文属于生成模型微调任务，解决多LoRA适配器无训练融合效果差的问题。提出MultLFG框架，通过时序和频段自适应策略选择性激活相关LoRA，提升合成图像的空间一致性和质量，在ComposLoRA基准测试中表现最优。**

- **链接: [http://arxiv.org/pdf/2505.20525v1](http://arxiv.org/pdf/2505.20525v1)**

> **作者:** Aniket Roy; Maitreya Suin; Ketul Shah; Rama Chellappa
>
> **摘要:** Low-Rank Adaptation (LoRA) has gained prominence as a computationally efficient method for fine-tuning generative models, enabling distinct visual concept synthesis with minimal overhead. However, current methods struggle to effectively merge multiple LoRA adapters without training, particularly in complex compositions involving diverse visual elements. We introduce MultLFG, a novel framework for training-free multi-LoRA composition that utilizes frequency-domain guidance to achieve adaptive fusion of multiple LoRAs. Unlike existing methods that uniformly aggregate concept-specific LoRAs, MultLFG employs a timestep and frequency subband adaptive fusion strategy, selectively activating relevant LoRAs based on content relevance at specific timesteps and frequency bands. This frequency-sensitive guidance not only improves spatial coherence but also provides finer control over multi-LoRA composition, leading to more accurate and consistent results. Experimental evaluations on the ComposLoRA benchmark reveal that MultLFG substantially enhances compositional fidelity and image quality across various styles and concept sets, outperforming state-of-the-art baselines in multi-concept generation tasks. Code will be released.
>
---
#### [new 011] LPOI: Listwise Preference Optimization for Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LPOI方法，通过列表式偏好优化减少视觉语言模型(VLMs)的幻觉。针对RLHF/DPO等方法过拟合文本或加剧幻觉的问题，其通过掩码关键物体并插值生成渐进图像序列，训练模型按物体可见性排序，提升性能且无需额外标注。**

- **链接: [http://arxiv.org/pdf/2505.21061v1](http://arxiv.org/pdf/2505.21061v1)**

> **作者:** Fatemeh Pesaran Zadeh; Yoojin Oh; Gunhee Kim
>
> **备注:** ACL 2025 Main. Code is released at https://github.com/fatemehpesaran310/lpoi
>
> **摘要:** Aligning large VLMs with human preferences is a challenging task, as methods like RLHF and DPO often overfit to textual information or exacerbate hallucinations. Although augmenting negative image samples partially addresses these pitfalls, no prior work has employed listwise preference optimization for VLMs, due to the complexity and cost of constructing listwise image samples. In this work, we propose LPOI, the first object-aware listwise preference optimization developed for reducing hallucinations in VLMs. LPOI identifies and masks a critical object in the image, and then interpolates the masked region between the positive and negative images to form a sequence of incrementally more complete images. The model is trained to rank these images in ascending order of object visibility, effectively reducing hallucinations while retaining visual fidelity. LPOI requires no extra annotations beyond standard pairwise preference data, as it automatically constructs the ranked lists through object masking and interpolation. Comprehensive experiments on MMHalBench, AMBER, and Object HalBench confirm that LPOI outperforms existing preference optimization methods in reducing hallucinations and enhancing VLM performance. We make the code available at https://github.com/fatemehpesaran310/lpoi.
>
---
#### [new 012] ISAC: Training-Free Instance-to-Semantic Attention Control for Improving Multi-Instance Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决多实例生成中物体合并/遗漏及语义纠缠问题。提出无训练的ISAC方法，通过实例优先建模与树结构提示机制，分离并精准对齐实例与语义标签，提升生成质量，无需外部模型。**

- **链接: [http://arxiv.org/pdf/2505.20935v1](http://arxiv.org/pdf/2505.20935v1)**

> **作者:** Sanghyun Jo; Wooyeol Lee; Ziseok Lee; Kyungsu Kim
>
> **备注:** 34 pages
>
> **摘要:** Text-to-image diffusion models excel at generating single-instance scenes but struggle with multi-instance scenarios, often merging or omitting objects. Unlike previous training-free approaches that rely solely on semantic-level guidance without addressing instance individuation, our training-free method, Instance-to-Semantic Attention Control (ISAC), explicitly resolves incomplete instance formation and semantic entanglement through an instance-first modeling approach. This enables ISAC to effectively leverage a hierarchical, tree-structured prompt mechanism, disentangling multiple object instances and individually aligning them with their corresponding semantic labels. Without employing any external models, ISAC achieves up to 52% average multi-class accuracy and 83% average multi-instance accuracy by effectively forming disentangled instances. The code will be made available upon publication.
>
---
#### [new 013] Think Twice, Act Once: Token-Aware Compression and Action Reuse for Efficient Inference in Vision-Language-Action Models
- **分类: cs.CV**

- **简介: 该论文属于Vision-Language-Action（VLA）模型加速任务，旨在解决其高推理成本问题。提出FlashVLA框架，通过动作复用机制减少连续动作的冗余解码，结合视觉token选择策略剔除低贡献视觉信息，实现推理效率提升（FLOPs降55.7%，延迟降36%），且精度损失小。**

- **链接: [http://arxiv.org/pdf/2505.21200v1](http://arxiv.org/pdf/2505.21200v1)**

> **作者:** Xudong Tan; Yaoxin Yang; Peng Ye; Jialin Zheng; Bizhe Bai; Xinyi Wang; Jia Hao; Tao Chen
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful paradigm for general-purpose robot control through natural language instructions. However, their high inference cost-stemming from large-scale token computation and autoregressive decoding-poses significant challenges for real-time deployment and edge applications. While prior work has primarily focused on architectural optimization, we take a different perspective by identifying a dual form of redundancy in VLA models: (i) high similarity across consecutive action steps, and (ii) substantial redundancy in visual tokens. Motivated by these observations, we propose FlashVLA, the first training-free and plug-and-play acceleration framework that enables action reuse in VLA models. FlashVLA improves inference efficiency through a token-aware action reuse mechanism that avoids redundant decoding across stable action steps, and an information-guided visual token selection strategy that prunes low-contribution tokens. Extensive experiments on the LIBERO benchmark show that FlashVLA reduces FLOPs by 55.7% and latency by 36.0%, with only a 0.7% drop in task success rate. These results demonstrate the effectiveness of FlashVLA in enabling lightweight, low-latency VLA inference without retraining.
>
---
#### [new 014] Frame In-N-Out: Unbounded Controllable Image-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文提出Frame In-N-Out技术，属于可控图像到视频生成任务。旨在解决可控性、时间连贯性及细节合成问题。通过构建半自动数据集、新评估协议及身份保持运动可控扩散Transformer模型，支持用户通过轨迹控制物体进出场景，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21491v1](http://arxiv.org/pdf/2505.21491v1)**

> **作者:** Boyang Wang; Xuweiyi Chen; Matheus Gadelha; Zezhou Cheng
>
> **摘要:** Controllability, temporal coherence, and detail synthesis remain the most critical challenges in video generation. In this paper, we focus on a commonly used yet underexplored cinematic technique known as Frame In and Frame Out. Specifically, starting from image-to-video generation, users can control the objects in the image to naturally leave the scene or provide breaking new identity references to enter the scene, guided by user-specified motion trajectory. To support this task, we introduce a new dataset curated semi-automatically, a comprehensive evaluation protocol targeting this setting, and an efficient identity-preserving motion-controllable video Diffusion Transformer architecture. Our evaluation shows that our proposed approach significantly outperforms existing baselines.
>
---
#### [new 015] ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ViewSpatial-Bench基准，评估视觉语言模型（VLMs）在多视角空间定位任务中的表现。针对现有模型擅长相机视角但无法有效处理他人称视角的问题，设计五类任务及3D标注系统，揭示模型性能差距，并通过微调提升46.24%，推动具身AI的空间智能研究。**

- **链接: [http://arxiv.org/pdf/2505.21500v1](http://arxiv.org/pdf/2505.21500v1)**

> **作者:** Dingming Li; Hongxing Li; Zixuan Wang; Yuchen Yan; Hang Zhang; Siqi Chen; Guiyang Hou; Shengpei Jiang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Yueting Zhuang
>
> **备注:** Project: https://zju-real.github.io/ViewSpatial-Page/
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and reasoning about visual content, but significant challenges persist in tasks requiring cross-viewpoint understanding and spatial reasoning. We identify a critical limitation: current VLMs excel primarily at egocentric spatial reasoning (from the camera's perspective) but fail to generalize to allocentric viewpoints when required to adopt another entity's spatial frame of reference. We introduce ViewSpatial-Bench, the first comprehensive benchmark designed specifically for multi-viewpoint spatial localization recognition evaluation across five distinct task types, supported by an automated 3D annotation pipeline that generates precise directional labels. Comprehensive evaluation of diverse VLMs on ViewSpatial-Bench reveals a significant performance disparity: models demonstrate reasonable performance on camera-perspective tasks but exhibit reduced accuracy when reasoning from a human viewpoint. By fine-tuning VLMs on our multi-perspective spatial dataset, we achieve an overall performance improvement of 46.24% across tasks, highlighting the efficacy of our approach. Our work establishes a crucial benchmark for spatial intelligence in embodied AI systems and provides empirical evidence that modeling 3D spatial relationships enhances VLMs' corresponding spatial comprehension capabilities.
>
---
#### [new 016] ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding
- **分类: cs.CV**

- **简介: 该论文属于点云自监督学习任务，旨在解决传统PointMamba模型因复杂排序和随机掩码导致的空间连续性破坏及语义关联弱化问题。提出ZigzagPointMamba，采用zigzag扫描路径保持空间连续性，并引入Semantic-Siamese Masking Strategy通过掩码语义相似token增强局部特征整合，提升全局语义建模，实验显示在多个点云任务中性能显著提升。**

- **链接: [http://arxiv.org/pdf/2505.21381v1](http://arxiv.org/pdf/2505.21381v1)**

> **作者:** Linshuang Diao; Dayong Ren; Sensen Song; Yurong Qian
>
> **摘要:** State Space models (SSMs) such as PointMamba enable efficient feature extraction for point cloud self-supervised learning with linear complexity, outperforming Transformers in computational efficiency. However, existing PointMamba-based methods depend on complex token ordering and random masking, which disrupt spatial continuity and local semantic correlations. We propose ZigzagPointMamba to tackle these challenges. The core of our approach is a simple zigzag scan path that globally sequences point cloud tokens, enhancing spatial continuity by preserving the proximity of spatially adjacent point tokens. Nevertheless, random masking undermines local semantic modeling in self-supervised learning. To address this, we introduce a Semantic-Siamese Masking Strategy (SMS), which masks semantically similar tokens to facilitate reconstruction by integrating local features of original and similar tokens. This overcomes the dependence on isolated local features and enables robust global semantic modeling. Our pre-trained ZigzagPointMamba weights significantly improve downstream tasks, achieving a 1.59% mIoU gain on ShapeNetPart for part segmentation, a 0.4% higher accuracy on ModelNet40 for classification, and 0.19%, 1.22%, and 0.72% higher accuracies respectively for the classification tasks on the OBJ-BG, OBJ-ONLY, and PB-T50-RS subsets of ScanObjectNN. The code is available at: https://anonymous.4open.science/r/ZigzagPointMamba-1800/
>
---
#### [new 017] GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution
- **分类: cs.CV**

- **简介: 该论文提出GeoLLaVA-8K，解决超分辨率遥感图像处理中数据稀缺和token爆炸问题。构建了最高分辨率的SuperRS-VQA和HighRS-VQA数据集，并提出背景token修剪和锚定选择策略，减少冗余以提升模型效率。该模型支持8K输入，达地球观测任务新标杆。**

- **链接: [http://arxiv.org/pdf/2505.21375v1](http://arxiv.org/pdf/2505.21375v1)**

> **作者:** Fengxiang Wang; Mingshuo Chen; Yueying Li; Di Wang; Haotian Wang; Zonghao Guo; Zefan Wang; Boqi Shan; Long Lan; Yulin Wang; Hongzhen Wang; Wenjing Yang; Bo Du; Jing Zhang
>
> **摘要:** Ultra-high-resolution (UHR) remote sensing (RS) imagery offers valuable data for Earth observation but pose challenges for existing multimodal foundation models due to two key bottlenecks: (1) limited availability of UHR training data, and (2) token explosion caused by the large image size. To address data scarcity, we introduce SuperRS-VQA (avg. 8,376$\times$8,376) and HighRS-VQA (avg. 2,000$\times$1,912), the highest-resolution vision-language datasets in RS to date, covering 22 real-world dialogue tasks. To mitigate token explosion, our pilot studies reveal significant redundancy in RS images: crucial information is concentrated in a small subset of object-centric tokens, while pruning background tokens (e.g., ocean or forest) can even improve performance. Motivated by these findings, we propose two strategies: Background Token Pruning and Anchored Token Selection, to reduce the memory footprint while preserving key semantics.Integrating these techniques, we introduce GeoLLaVA-8K, the first RS-focused multimodal large language model capable of handling inputs up to 8K$\times$8K resolution, built on the LLaVA framework. Trained on SuperRS-VQA and HighRS-VQA, GeoLLaVA-8K sets a new state-of-the-art on the XLRS-Bench.
>
---
#### [new 018] MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition
- **分类: cs.CV**

- **简介: 该论文提出MoPFormer，针对可穿戴传感器人类活动识别中可解释性不足及跨数据集泛化难题，通过将传感器信号分割为语义运动原语，结合Transformer学习时序特征，并采用自监督预训练。该方法提升可解释性与跨数据集性能，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20744v1](http://arxiv.org/pdf/2505.20744v1)**

> **作者:** Hao Zhang; Zhan Zhuang; Xuehao Wang; Xiaodong Yang; Yu Zhang
>
> **摘要:** Human Activity Recognition (HAR) with wearable sensors is challenged by limited interpretability, which significantly impacts cross-dataset generalization. To address this challenge, we propose Motion-Primitive Transformer (MoPFormer), a novel self-supervised framework that enhances interpretability by tokenizing inertial measurement unit signals into semantically meaningful motion primitives and leverages a Transformer architecture to learn rich temporal representations. MoPFormer comprises two-stages. first stage is to partition multi-channel sensor streams into short segments and quantizing them into discrete "motion primitive" codewords, while the second stage enriches those tokenized sequences through a context-aware embedding module and then processes them with a Transformer encoder. The proposed MoPFormer can be pre-trained using a masked motion-modeling objective that reconstructs missing primitives, enabling it to develop robust representations across diverse sensor configurations. Experiments on six HAR benchmarks demonstrate that MoPFormer not only outperforms state-of-the-art methods but also successfully generalizes across multiple datasets. Most importantly, the learned motion primitives significantly enhance both interpretability and cross-dataset performance by capturing fundamental movement patterns that remain consistent across similar activities regardless of dataset origin.
>
---
#### [new 019] Beyond Accuracy: Uncovering the Role of Similarity Perception and its Alignment with Semantics in Supervised Learning
- **分类: cs.CV**

- **简介: 该论文研究深度视觉模型（CNN/ViT）在训练中相似性感知的形成及其与语义的对齐问题。提出DSI框架，揭示模型经历相似性激增、细化、稳定三阶段，并发现两类模型存在差异及错误修正现象。任务聚焦监督学习中相似性机制分析，解决模型内部表征与语义关联的系统化研究不足问题。**

- **链接: [http://arxiv.org/pdf/2505.21338v1](http://arxiv.org/pdf/2505.21338v1)**

> **作者:** Katarzyna Filus; Mateusz Żarski
>
> **摘要:** Similarity manifests in various forms, including semantic similarity that is particularly important, serving as an approximation of human object categorization based on e.g. shared functionalities and evolutionary traits. It also offers practical advantages in computational modeling via lexical structures such as WordNet with constant and interpretable similarity. As in the domain of deep vision, there is still not enough focus on the phenomena regarding the similarity perception emergence. We introduce Deep Similarity Inspector (DSI) -- a systematic framework to inspect how deep vision networks develop their similarity perception and its alignment with semantic similarity. Our experiments show that both Convolutional Neural Networks' (CNNs) and Vision Transformers' (ViTs) develop a rich similarity perception during training with 3 phases (initial similarity surge, refinement, stabilization), with clear differences between CNNs and ViTs. Besides the gradual mistakes elimination, the mistakes refinement phenomenon can be observed.
>
---
#### [new 020] Create Anything Anywhere: Layout-Controllable Personalized Diffusion Model for Multiple Subjects
- **分类: cs.CV**

- **简介: 该论文提出LCP-Diffusion模型，解决现有个性化扩散模型布局控制不足和动态特征利用缺失问题。通过动态静态互补视觉优化模块与双布局控制机制，实现无调参的身份保持与灵活空间布局，提升生成图像的保真度与可控性。**

- **链接: [http://arxiv.org/pdf/2505.20909v1](http://arxiv.org/pdf/2505.20909v1)**

> **作者:** Wei Li; Hebei Li; Yansong Peng; Siying Wu; Yueyi Zhang; Xiaoyan Sun
>
> **备注:** ICME 2025
>
> **摘要:** Diffusion models have significantly advanced text-to-image generation, laying the foundation for the development of personalized generative frameworks. However, existing methods lack precise layout controllability and overlook the potential of dynamic features of reference subjects in improving fidelity. In this work, we propose Layout-Controllable Personalized Diffusion (LCP-Diffusion) model, a novel framework that integrates subject identity preservation with flexible layout guidance in a tuning-free approach. Our model employs a Dynamic-Static Complementary Visual Refining module to comprehensively capture the intricate details of reference subjects, and introduces a Dual Layout Control mechanism to enforce robust spatial control across both training and inference stages. Extensive experiments validate that LCP-Diffusion excels in both identity preservation and layout controllability. To the best of our knowledge, this is a pioneering work enabling users to "create anything anywhere".
>
---
#### [new 021] Not All Thats Rare Is Lost: Causal Paths to Rare Concept Synthesis
- **分类: cs.CV**

- **简介: 该论文提出RAP框架，解决扩散模型生成罕见概念图像效果差的问题。通过构建因果路径，用语义相关常见提示近似罕见提示，动态切换提示并采用二阶去噪机制，引导生成过程从常见概念平滑过渡到罕见目标，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20808v1](http://arxiv.org/pdf/2505.20808v1)**

> **作者:** Bo-Kai Ruan; Zi-Xiang Ni; Bo-Lun Huang; Teng-Fang Hsiao; Hong-Han Shuai
>
> **摘要:** Diffusion models have shown strong capabilities in high-fidelity image generation but often falter when synthesizing rare concepts, i.e., prompts that are infrequently observed in the training distribution. In this paper, we introduce RAP, a principled framework that treats rare concept generation as navigating a latent causal path: a progressive, model-aligned trajectory through the generative space from frequent concepts to rare targets. Rather than relying on heuristic prompt alternation, we theoretically justify that rare prompt guidance can be approximated by semantically related frequent prompts. We then formulate prompt switching as a dynamic process based on score similarity, enabling adaptive stage transitions. Furthermore, we reinterpret prompt alternation as a second-order denoising mechanism, promoting smooth semantic progression and coherent visual synthesis. Through this causal lens, we align input scheduling with the model's internal generative dynamics. Experiments across diverse diffusion backbones demonstrate that RAP consistently enhances rare concept generation, outperforming strong baselines in both automated evaluations and human studies.
>
---
#### [new 022] DiMoSR: Feature Modulation via Multi-Branch Dilated Convolutions for Efficient Image Super-Resolution
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于轻量级单图像超分辨率（SISR）任务，旨在解决模型效率与重建质量的平衡问题。提出DiMoSR架构，通过多分支空洞卷积实现特征调制，在扩大感受野捕获上下文信息的同时保持高效计算，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21262v1](http://arxiv.org/pdf/2505.21262v1)**

> **作者:** M. Akin Yilmaz; Ahmet Bilican; A. Murat Tekalp
>
> **摘要:** Balancing reconstruction quality versus model efficiency remains a critical challenge in lightweight single image super-resolution (SISR). Despite the prevalence of attention mechanisms in recent state-of-the-art SISR approaches that primarily emphasize or suppress feature maps, alternative architectural paradigms warrant further exploration. This paper introduces DiMoSR (Dilated Modulation Super-Resolution), a novel architecture that enhances feature representation through modulation to complement attention in lightweight SISR networks. The proposed approach leverages multi-branch dilated convolutions to capture rich contextual information over a wider receptive field while maintaining computational efficiency. Experimental results demonstrate that DiMoSR outperforms state-of-the-art lightweight methods across diverse benchmark datasets, achieving superior PSNR and SSIM metrics with comparable or reduced computational complexity. Through comprehensive ablation studies, this work not only validates the effectiveness of DiMoSR but also provides critical insights into the interplay between attention mechanisms and feature modulation to guide future research in efficient network design. The code and model weights to reproduce our results are available at: https://github.com/makinyilmaz/DiMoSR
>
---
#### [new 023] Vision Transformers with Self-Distilled Registers
- **分类: cs.CV**

- **简介: 该论文针对Vision Transformer中异常artifact tokens导致定位与结构任务性能下降的问题，提出Post Hoc Registers方法。通过自蒸馏在预训练模型中添加可训练的register tokens，利用教师网络的增强输入优化少量参数，提升分割和深度预测性能，无需重训。**

- **链接: [http://arxiv.org/pdf/2505.21501v1](http://arxiv.org/pdf/2505.21501v1)**

> **作者:** Yinjie Chen; Zipeng Yan; Chong Zhou; Bo Dai; Andrew F. Luo
>
> **备注:** 27 pages, 14 figures
>
> **摘要:** Vision Transformers (ViTs) have emerged as the dominant architecture for visual processing tasks, demonstrating excellent scalability with increased training data and model size. However, recent work has identified the emergence of artifact tokens in ViTs that are incongruous with the local semantics. These anomalous tokens degrade ViT performance in tasks that require fine-grained localization or structural coherence. An effective mitigation of this issue is to the addition of register tokens to ViTs, which implicitly "absorb" the artifact term during training. Given the availability of various large-scale pre-trained ViTs, in this paper we aim at equipping them with such register tokens without the need of re-training them from scratch, which is infeasible considering their size. Specifically, we propose Post Hoc Registers (PH-Reg), an efficient self-distillation method that integrates registers into an existing ViT without requiring additional labeled data and full retraining. PH-Reg initializes both teacher and student networks from the same pre-trained ViT. The teacher remains frozen and unmodified, while the student is augmented with randomly initialized register tokens. By applying test-time augmentation to the teacher's inputs, we generate denoised dense embeddings free of artifacts, which are then used to optimize only a small subset of unlocked student weights. We show that our approach can effectively reduce the number of artifact tokens, improving the segmentation and depth prediction of the student ViT under zero-shot and linear probing.
>
---
#### [new 024] RainFusion: Adaptive Video Generation Acceleration via Multi-Dimensional Visual Redundancy
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成加速任务，旨在解决扩散模型中3D注意力计算资源消耗过高的问题。提出RainFusion方法，通过识别视频生成中的空间、时间、纹理三种视觉冗余模式，利用ARM模块在线生成稀疏注意力模式，实现无需训练的加速。实验显示其计算速度提升超2倍，且视频质量影响微小。**

- **链接: [http://arxiv.org/pdf/2505.21036v1](http://arxiv.org/pdf/2505.21036v1)**

> **作者:** Aiyue Chen; Bin Dong; Jingru Li; Jing Lin; Yiwu Yao; Gongyi Wang
>
> **摘要:** Video generation using diffusion models is highly computationally intensive, with 3D attention in Diffusion Transformer (DiT) models accounting for over 80\% of the total computational resources. In this work, we introduce {\bf RainFusion}, a novel training-free sparse attention method that exploits inherent sparsity nature in visual data to accelerate attention computation while preserving video quality. Specifically, we identify three unique sparse patterns in video generation attention calculations--Spatial Pattern, Temporal Pattern and Textural Pattern. The sparse pattern for each attention head is determined online with negligible overhead (\textasciitilde\,0.2\%) with our proposed {\bf ARM} (Adaptive Recognition Module) during inference. Our proposed {\bf RainFusion} is a plug-and-play method, that can be seamlessly integrated into state-of-the-art 3D-attention video generation models without additional training or calibration. We evaluate our method on leading open-sourced models including HunyuanVideo, OpenSoraPlan-1.2 and CogVideoX-5B, demonstrating its broad applicability and effectiveness. Experimental results show that RainFusion achieves over {\bf 2\(\times\)} speedup in attention computation while maintaining video quality, with only a minimal impact on VBench scores (-0.2\%).
>
---
#### [new 025] TACO: Think-Answer Consistency for Optimized Long-Chain Reasoning and Efficient Data Learning via Reinforcement Learning in LVLMs
- **分类: cs.CV**

- **简介: 该论文提出TACO算法，针对多模态任务（如REC和VQA）中长链推理不一致、模型不稳定及数据效率低的问题，通过强化学习引入思维-答案一致性机制、回滚重采样策略、自适应学习计划及分辨率缩放方案，提升模型推理性能与稳定性。**

- **链接: [http://arxiv.org/pdf/2505.20777v1](http://arxiv.org/pdf/2505.20777v1)**

> **作者:** Zhehan Kan; Yanlin Liu; Kun Yin; Xinghua Jiang; Xin Li; Haoyu Cao; Yinsong Liu; Deqiang Jiang; Xing Sun; Qingmin Liao; Wenming Yang
>
> **摘要:** DeepSeek R1 has significantly advanced complex reasoning for large language models (LLMs). While recent methods have attempted to replicate R1's reasoning capabilities in multimodal settings, they face limitations, including inconsistencies between reasoning and final answers, model instability and crashes during long-chain exploration, and low data learning efficiency. To address these challenges, we propose TACO, a novel reinforcement learning algorithm for visual reasoning. Building on Generalized Reinforcement Policy Optimization (GRPO), TACO introduces Think-Answer Consistency, which tightly couples reasoning with answer consistency to ensure answers are grounded in thoughtful reasoning. We also introduce the Rollback Resample Strategy, which adaptively removes problematic samples and reintroduces them to the sampler, enabling stable long-chain exploration and future learning opportunities. Additionally, TACO employs an adaptive learning schedule that focuses on moderate difficulty samples to optimize data efficiency. Furthermore, we propose the Test-Time-Resolution-Scaling scheme to address performance degradation due to varying resolutions during reasoning while balancing computational overhead. Extensive experiments on in-distribution and out-of-distribution benchmarks for REC and VQA tasks show that fine-tuning LVLMs leads to significant performance improvements.
>
---
#### [new 026] Mamba-Driven Topology Fusion for Monocular 3-D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文针对单目3D人体姿态估计任务，解决Mamba模型在处理关节拓扑关系和局部依赖上的不足。提出Bone Aware模块通过球面坐标建模骨骼拓扑，改进双向图卷积捕捉局部关系，并设计时空优化模块，提升结构建模能力，实现实时性与精度的提升。**

- **链接: [http://arxiv.org/pdf/2505.20611v1](http://arxiv.org/pdf/2505.20611v1)**

> **作者:** Zenghao Zheng; Lianping Yang; Jinshan Pan; Hegui Zhu
>
> **摘要:** Transformer-based methods for 3-D human pose estimation face significant computational challenges due to the quadratic growth of self-attention mechanism complexity with sequence length. Recently, the Mamba model has substantially reduced computational overhead and demonstrated outstanding performance in modeling long sequences by leveraging state space model (SSM). However, the ability of SSM to process sequential data is not suitable for 3-D joint sequences with topological structures, and the causal convolution structure in Mamba also lacks insight into local joint relationships. To address these issues, we propose the Mamba-Driven Topology Fusion framework in this paper. Specifically, the proposed Bone Aware Module infers the direction and length of bone vectors in the spherical coordinate system, providing effective topological guidance for the Mamba model in processing joint sequences. Furthermore, we enhance the convolutional structure within the Mamba model by integrating forward and backward graph convolutional network, enabling it to better capture local joint dependencies. Finally, we design a Spatiotemporal Refinement Module to model both temporal and spatial relationships within the sequence. Through the incorporation of skeletal topology, our approach effectively alleviates Mamba's limitations in capturing human structural relationships. We conduct extensive experiments on the Human3.6M and MPI-INF-3DHP datasets for testing and comparison, and the results show that the proposed method greatly reduces computational cost while achieving higher accuracy. Ablation studies further demonstrate the effectiveness of each proposed module. The code and models will be released.
>
---
#### [new 027] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter
- **分类: cs.CV**

- **简介: 该论文属于3D感知中的点云理解任务，旨在解决现有方法仅利用预训练模型最终输出而忽略中间层互补信息的问题。提出Point Mamba Adapter（PMA）通过构建多层有序特征序列并融合语义，结合几何约束门控生成器（G2PG）动态优化空间顺序，提升点云理解效果。**

- **链接: [http://arxiv.org/pdf/2505.20941v1](http://arxiv.org/pdf/2505.20941v1)**

> **作者:** Yaohua Zha; Yanzi Wang; Hang Guo; Jinpeng Wang; Tao Dai; Bin Chen; Zhihao Ouyang; Xue Yuerong; Ke Chen; Shu-Tao Xia
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Applying pre-trained models to assist point cloud understanding has recently become a mainstream paradigm in 3D perception. However, existing application strategies are straightforward, utilizing only the final output of the pre-trained model for various task heads. It neglects the rich complementary information in the intermediate layer, thereby failing to fully unlock the potential of pre-trained models. To overcome this limitation, we propose an orthogonal solution: Point Mamba Adapter (PMA), which constructs an ordered feature sequence from all layers of the pre-trained model and leverages Mamba to fuse all complementary semantics, thereby promoting comprehensive point cloud understanding. Constructing this ordered sequence is non-trivial due to the inherent isotropy of 3D space. Therefore, we further propose a geometry-constrained gate prompt generator (G2PG) shared across different layers, which applies shared geometric constraints to the output gates of the Mamba and dynamically optimizes the spatial order, thus enabling more effective integration of multi-layer information. Extensive experiments conducted on challenging point cloud datasets across various tasks demonstrate that our PMA elevates the capability for point cloud understanding to a new level by fusing diverse complementary intermediate features. Code is available at https://github.com/zyh16143998882/PMA.
>
---
#### [new 028] Mentor3AD: Feature Reconstruction-based 3D Anomaly Detection via Multi-modality Mentor Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D异常检测任务，解决多模态特征融合不足的问题。提出Mentor3AD方法，通过多模态导师学习融合RGB与3D特征，设计MFM、MGM和VM模块提升特征区分与重建，最终生成准确异常分数。实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.21420v1](http://arxiv.org/pdf/2505.21420v1)**

> **作者:** Jinbao Wang; Hanzhe Liang; Can Gao; Chenxi Hu; Jie Zhou; Yunkang Cao; Linlin Shen; Weiming Shen
>
> **备注:** 10 Pages, 6 Figures, 7 Tables
>
> **摘要:** Multimodal feature reconstruction is a promising approach for 3D anomaly detection, leveraging the complementary information from dual modalities. We further advance this paradigm by utilizing multi-modal mentor learning, which fuses intermediate features to further distinguish normal from feature differences. To address these challenges, we propose a novel method called Mentor3AD, which utilizes multi-modal mentor learning. By leveraging the shared features of different modalities, Mentor3AD can extract more effective features and guide feature reconstruction, ultimately improving detection performance. Specifically, Mentor3AD includes a Mentor of Fusion Module (MFM) that merges features extracted from RGB and 3D modalities to create a mentor feature. Additionally, we have designed a Mentor of Guidance Module (MGM) to facilitate cross-modal reconstruction, supported by the mentor feature. Lastly, we introduce a Voting Module (VM) to more accurately generate the final anomaly score. Extensive comparative and ablation studies on MVTec 3D-AD and Eyecandies have verified the effectiveness of the proposed method.
>
---
#### [new 029] Open-Det: An Efficient Learning Framework for Open-Ended Detection
- **分类: cs.CV**

- **简介: 该论文提出Open-Det框架，解决开放集检测（OED）任务中数据需求大、训练慢、性能不足的问题。通过重构检测器与生成器、设计视觉-语言对齐器及知识蒸馏、引入新型损失函数，实现高效训练，在更少资源下提升检测与命名性能。**

- **链接: [http://arxiv.org/pdf/2505.20639v1](http://arxiv.org/pdf/2505.20639v1)**

> **作者:** Guiping Cao; Tao Wang; Wenjian Huang; Xiangyuan Lan; Jianguo Zhang; Dongmei Jiang
>
> **备注:** ICML 2025
>
> **摘要:** Open-Ended object Detection (OED) is a novel and challenging task that detects objects and generates their category names in a free-form manner, without requiring additional vocabularies during inference. However, the existing OED models, such as GenerateU, require large-scale datasets for training, suffer from slow convergence, and exhibit limited performance. To address these issues, we present a novel and efficient Open-Det framework, consisting of four collaborative parts. Specifically, Open-Det accelerates model training in both the bounding box and object name generation process by reconstructing the Object Detector and the Object Name Generator. To bridge the semantic gap between Vision and Language modalities, we propose a Vision-Language Aligner with V-to-L and L-to-V alignment mechanisms, incorporating with the Prompts Distiller to transfer knowledge from the VLM into VL-prompts, enabling accurate object name generation for the LLM. In addition, we design a Masked Alignment Loss to eliminate contradictory supervision and introduce a Joint Loss to enhance classification, resulting in more efficient training. Compared to GenerateU, Open-Det, using only 1.5% of the training data (0.077M vs. 5.077M), 20.8% of the training epochs (31 vs. 149), and fewer GPU resources (4 V100 vs. 16 A100), achieves even higher performance (+1.0% in APr). The source codes are available at: https://github.com/Med-Process/Open-Det.
>
---
#### [new 030] HuMoCon: Concept Discovery for Human Motion Understanding
- **分类: cs.CV; 68T07; I.2.10; I.2.7**

- **简介: 该论文属于人类运动理解任务，旨在解决多模态特征对齐不足及高频率信息丢失问题。提出HuMoCon框架，通过视频上下文与运动细节的特征对齐策略，结合速度重建机制，提升运动概念发现的语义表达与泛化能力，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20920v1](http://arxiv.org/pdf/2505.20920v1)**

> **作者:** Qihang Fang; Chengcheng Tang; Bugra Tekin; Shugao Ma; Yanchao Yang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** We present HuMoCon, a novel motion-video understanding framework designed for advanced human behavior analysis. The core of our method is a human motion concept discovery framework that efficiently trains multi-modal encoders to extract semantically meaningful and generalizable features. HuMoCon addresses key challenges in motion concept discovery for understanding and reasoning, including the lack of explicit multi-modality feature alignment and the loss of high-frequency information in masked autoencoding frameworks. Our approach integrates a feature alignment strategy that leverages video for contextual understanding and motion for fine-grained interaction modeling, further with a velocity reconstruction mechanism to enhance high-frequency feature expression and mitigate temporal over-smoothing. Comprehensive experiments on standard benchmarks demonstrate that HuMoCon enables effective motion concept discovery and significantly outperforms state-of-the-art methods in training large models for human motion understanding. We will open-source the associated code with our paper.
>
---
#### [new 031] Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visual Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于增强音频-视觉大语言模型（AV-LLMs）的多模态理解任务，旨在解决模态偏差问题（模型过度依赖单一模态）。提出Fork-Merge Decoding方法：推理时先分别处理音频/视频特征（叉阶段），再合并结果联合推理（合阶段），平衡模态贡献并提升跨模态互补性。**

- **链接: [http://arxiv.org/pdf/2505.20873v1](http://arxiv.org/pdf/2505.20873v1)**

> **作者:** Chaeyoung Jung; Youngjoon Jang; Jongmin Choi; Joon Son Chung
>
> **摘要:** The goal of this work is to enhance balanced multimodal understanding in audio-visual large language models (AV-LLMs) by addressing modality bias without requiring additional training. In current AV-LLMs, audio and video features are typically processed jointly in the decoder. While this strategy facilitates unified multimodal understanding, it may introduce modality bias, where the model tends to over-rely on one modality due to imbalanced training signals. To mitigate this, we propose Fork-Merge Decoding (FMD), a simple yet effective inference-time strategy that requires no additional training or architectural modifications. FMD first performs modality-specific reasoning by processing audio-only and video-only inputs through the early decoder layers (a fork phase), and then merges the resulting hidden states for joint reasoning in the remaining layers (a merge phase). This approach promotes balanced modality contributions and leverages complementary information across modalities. We evaluate our method on two representative AV-LLMs, VideoLLaMA2 and video-SALMONN, using three benchmark datasets. Experimental results demonstrate consistent performance improvements on tasks focused on audio, video, and combined audio-visual reasoning, demonstrating the effectiveness of inference-time interventions for robust multimodal understanding.
>
---
#### [new 032] Instance Data Condensation for Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率（ISR）任务，旨在解决其依赖大规模数据导致的高计算/存储成本问题。提出Instance Data Condensation（IDC）框架，通过随机局部傅里叶特征提取与多级特征分布匹配，生成高质量合成数据。在DIV2K数据集上实现10%凝缩率，合成数据性能媲美原数据，为首次实现超分辨率领域小规模合成数据的有效应用。**

- **链接: [http://arxiv.org/pdf/2505.21099v1](http://arxiv.org/pdf/2505.21099v1)**

> **作者:** Tianhao Peng; Ho Man Kwan; Yuxuan Jiang; Ge Gao; Fan Zhang; Xiaozhong Xu; Shan Liu; David Bull
>
> **摘要:** Deep learning based image Super-Resolution (ISR) relies on large training datasets to optimize model generalization; this requires substantial computational and storage resources during training. While dataset condensation has shown potential in improving data efficiency and privacy for high-level computer vision tasks, it has not yet been fully exploited for ISR. In this paper, we propose a novel Instance Data Condensation (IDC) framework specifically for ISR, which achieves instance-level data condensation through Random Local Fourier Feature Extraction and Multi-level Feature Distribution Matching. This aims to optimize feature distributions at both global and local levels and obtain high-quality synthesized training content with fine detail. This framework has been utilized to condense the most commonly used training dataset for ISR, DIV2K, with a 10% condensation rate. The resulting synthetic dataset offers comparable or (in certain cases) even better performance compared to the original full dataset and excellent training stability when used to train various popular ISR models. To the best of our knowledge, this is the first time that a condensed/synthetic dataset (with a 10% data volume) has demonstrated such performance. The source code and the synthetic dataset have been made available at https://github.com/.
>
---
#### [new 033] AgriFM: A Multi-source Temporal Remote Sensing Foundation Model for Crop Mapping
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业作物制图任务，旨在解决现有遥感模型忽略多尺度时空特征及时序信息的问题。提出AgriFM模型，改进Video Swin Transformer同步时空下采样，融合MODIS、Landsat-8/9和Sentinel-2的多源时序卫星数据，通过预训练和动态解码器提升作物制图性能，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21357v1](http://arxiv.org/pdf/2505.21357v1)**

> **作者:** Wenyuan Li; Shunlin Liang; Keyan Chen; Yongzhe Chen; Han Ma; Jianglei Xu; Yichuan Ma; Shikang Guan; Husheng Fang; Zhenwei Shi
>
> **摘要:** Accurate crop mapping fundamentally relies on modeling multi-scale spatiotemporal patterns, where spatial scales range from individual field textures to landscape-level context, and temporal scales capture both short-term phenological transitions and full growing-season dynamics. Transformer-based remote sensing foundation models (RSFMs) offer promising potential for crop mapping due to their innate ability for unified spatiotemporal processing. However, current RSFMs remain suboptimal for crop mapping: they either employ fixed spatiotemporal windows that ignore the multi-scale nature of crop systems or completely disregard temporal information by focusing solely on spatial patterns. To bridge these gaps, we present AgriFM, a multi-source remote sensing foundation model specifically designed for agricultural crop mapping. Our approach begins by establishing the necessity of simultaneous hierarchical spatiotemporal feature extraction, leading to the development of a modified Video Swin Transformer architecture where temporal down-sampling is synchronized with spatial scaling operations. This modified backbone enables efficient unified processing of long time-series satellite inputs. AgriFM leverages temporally rich data streams from three satellite sources including MODIS, Landsat-8/9 and Sentinel-2, and is pre-trained on a global representative dataset comprising over 25 million image samples supervised by land cover products. The resulting framework incorporates a versatile decoder architecture that dynamically fuses these learned spatiotemporal representations, supporting diverse downstream tasks. Comprehensive evaluations demonstrate AgriFM's superior performance over conventional deep learning approaches and state-of-the-art general-purpose RSFMs across all downstream tasks. Codes will be available at urlhttps://github.com/flyakon/AgriFM.
>
---
#### [new 034] DSOcc: Leveraging Depth Awareness and Semantic Aid to Boost Camera-Based 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文提出DSOcc方法，针对基于相机的3D语义占用预测任务。旨在解决现有方法因显式占用状态推断导致的特征分配错误及样本不足问题。通过融合深度感知（非学习计算软置信度，增强体素深度意识）和语义辅助（利用预训练分割模型及多帧概率融合），实现联合状态与类别推断，提升预测鲁棒性，在SemanticKITTI数据集达SOTA。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20951v1](http://arxiv.org/pdf/2505.20951v1)**

> **作者:** Naiyu Fang; Zheyuan Zhou; Kang Wang; Ruibo Li; Lemiao Qiu; Shuyou Zhang; Zhe Wang; Guosheng Lin
>
> **摘要:** Camera-based 3D semantic occupancy prediction offers an efficient and cost-effective solution for perceiving surrounding scenes in autonomous driving. However, existing works rely on explicit occupancy state inference, leading to numerous incorrect feature assignments, and insufficient samples restrict the learning of occupancy class inference. To address these challenges, we propose leveraging Depth awareness and Semantic aid to boost camera-based 3D semantic Occupancy prediction (DSOcc). We jointly perform occupancy state and occupancy class inference, where soft occupancy confidence is calculated through non-learning method and multiplied with image features to make the voxel representation aware of depth, enabling adaptive implicit occupancy state inference. Rather than focusing on improving feature learning, we directly utilize well-trained image semantic segmentation and fuse multiple frames with their occupancy probabilities to aid occupancy class inference, thereby enhancing robustness. Experimental results demonstrate that DSOcc achieves state-of-the-art performance on the SemanticKITTI dataset among camera-based methods.
>
---
#### [new 035] Spectral Compression Transformer with Line Pose Graph for Monocular 3D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文针对单目3D人体姿态估计任务，解决Transformer模型计算成本高及序列冗余问题。提出Spectral Compression Transformer通过离散余弦变换压缩序列长度，结合Line Pose Graph补充骨骼结构信息，并设计双流网络建模空间关系与运动轨迹，在保持高性能（Human3.6M达37.7mm MPJPE）的同时提升效率。**

- **链接: [http://arxiv.org/pdf/2505.21309v1](http://arxiv.org/pdf/2505.21309v1)**

> **作者:** Zenghao Zheng; Lianping Yang; Hegui Zhu; Mingrui Ye
>
> **摘要:** Transformer-based 3D human pose estimation methods suffer from high computational costs due to the quadratic complexity of self-attention with respect to sequence length. Additionally, pose sequences often contain significant redundancy between frames. However, recent methods typically fail to improve model capacity while effectively eliminating sequence redundancy. In this work, we introduce the Spectral Compression Transformer (SCT) to reduce sequence length and accelerate computation. The SCT encoder treats hidden features between blocks as Temporal Feature Signals (TFS) and applies the Discrete Cosine Transform, a Fourier transform-based technique, to determine the spectral components to be retained. By filtering out certain high-frequency noise components, SCT compresses the sequence length and reduces redundancy. To further enrich the input sequence with prior structural information, we propose the Line Pose Graph (LPG) based on line graph theory. The LPG generates skeletal position information that complements the input 2D joint positions, thereby improving the model's performance. Finally, we design a dual-stream network architecture to effectively model spatial joint relationships and the compressed motion trajectory within the pose sequence. Extensive experiments on two benchmark datasets (i.e., Human3.6M and MPI-INF-3DHP) demonstrate that our model achieves state-of-the-art performance with improved computational efficiency. For example, on the Human3.6M dataset, our method achieves an MPJPE of 37.7mm while maintaining a low computational cost. Furthermore, we perform ablation studies on each module to assess its effectiveness. The code and models will be released.
>
---
#### [new 036] Policy Optimized Text-to-Image Pipeline Design
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成流水线设计优化任务，解决现有方法计算成本高和泛化能力差的问题。提出基于强化学习的框架，训练奖励模型预测图像质量，采用两阶段策略（词汇训练与GRPO优化），结合分类器自由引导增强技术，提升生成多样性和图像质量。**

- **链接: [http://arxiv.org/pdf/2505.21478v1](http://arxiv.org/pdf/2505.21478v1)**

> **作者:** Uri Gadot; Rinon Gal; Yftah Ziser; Gal Chechik; Shie Mannor
>
> **摘要:** Text-to-image generation has evolved beyond single monolithic models to complex multi-component pipelines. These combine fine-tuned generators, adapters, upscaling blocks and even editing steps, leading to significant improvements in image quality. However, their effective design requires substantial expertise. Recent approaches have shown promise in automating this process through large language models (LLMs), but they suffer from two critical limitations: extensive computational requirements from generating images with hundreds of predefined pipelines, and poor generalization beyond memorized training examples. We introduce a novel reinforcement learning-based framework that addresses these inefficiencies. Our approach first trains an ensemble of reward models capable of predicting image quality scores directly from prompt-workflow combinations, eliminating the need for costly image generation during training. We then implement a two-phase training strategy: initial workflow vocabulary training followed by GRPO-based optimization that guides the model toward higher-performing regions of the workflow space. Additionally, we incorporate a classifier-free guidance based enhancement technique that extrapolates along the path between the initial and GRPO-tuned models, further improving output quality. We validate our approach through a set of comparisons, showing that it can successfully create new flows with greater diversity and lead to superior image quality compared to existing baselines.
>
---
#### [new 037] MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文提出MetaWriter，针对个性化手写文本识别任务，解决跨写作风格鲁棒性差及传统方法参数调整低效的问题。通过元学习优化提示调整结合自监督图像重建，利用无标签测试数据微调模型，仅更新<1%参数，无需标注。在RIMES和IAM数据集超越SOTA，参数减少20倍。**

- **链接: [http://arxiv.org/pdf/2505.20513v1](http://arxiv.org/pdf/2505.20513v1)**

> **作者:** Wenhao Gu; Li Gu; Ching Yee Suen; Yang Wang
>
> **备注:** CVPR2025
>
> **摘要:** Recent advancements in handwritten text recognition (HTR) have enabled the effective conversion of handwritten text to digital formats. However, achieving robust recognition across diverse writing styles remains challenging. Traditional HTR methods lack writer-specific personalization at test time due to limitations in model architecture and training strategies. Existing attempts to bridge this gap, through gradient-based meta-learning, still require labeled examples and suffer from parameter-inefficient fine-tuning, leading to substantial computational and memory overhead. To overcome these challenges, we propose an efficient framework that formulates personalization as prompt tuning, incorporating an auxiliary image reconstruction task with a self-supervised loss to guide prompt adaptation with unlabeled test-time examples. To ensure self-supervised loss effectively minimizes text recognition error, we leverage meta-learning to learn the optimal initialization of the prompts. As a result, our method allows the model to efficiently capture unique writing styles by updating less than 1% of its parameters and eliminating the need for time-intensive annotation processes. We validate our approach on the RIMES and IAM Handwriting Database benchmarks, where it consistently outperforms previous state-of-the-art methods while using 20x fewer parameters. We believe this represents a significant advancement in personalized handwritten text recognition, paving the way for more reliable and practical deployment in resource-constrained scenarios.
>
---
#### [new 038] Total-Editing: Head Avatar with Editable Appearance, Motion, and Lighting
- **分类: cs.CV**

- **简介: 该论文提出Total-Editing框架，统一处理头像的外观、动作及光照编辑任务。针对传统方法中面部重演与光照调整独立导致的协同性不足问题，设计神经辐射场解码器分解光照信息，并结合变形场增强运动与光照一致性，实现多维度精准控制，提升编辑效果并支持灵活应用。**

- **链接: [http://arxiv.org/pdf/2505.20582v1](http://arxiv.org/pdf/2505.20582v1)**

> **作者:** Yizhou Zhao; Chunjiang Liu; Haoyu Chen; Bhiksha Raj; Min Xu; Tadas Baltrusaitis; Mitch Rundle; HsiangTao Wu; Kamran Ghasedi
>
> **摘要:** Face reenactment and portrait relighting are essential tasks in portrait editing, yet they are typically addressed independently, without much synergy. Most face reenactment methods prioritize motion control and multiview consistency, while portrait relighting focuses on adjusting shading effects. To take advantage of both geometric consistency and illumination awareness, we introduce Total-Editing, a unified portrait editing framework that enables precise control over appearance, motion, and lighting. Specifically, we design a neural radiance field decoder with intrinsic decomposition capabilities. This allows seamless integration of lighting information from portrait images or HDR environment maps into synthesized portraits. We also incorporate a moving least squares based deformation field to enhance the spatiotemporal coherence of avatar motion and shading effects. With these innovations, our unified framework significantly improves the quality and realism of portrait editing results. Further, the multi-source nature of Total-Editing supports more flexible applications, such as illumination transfer from one portrait to another, or portrait animation with customized backgrounds.
>
---
#### [new 039] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文针对大视觉语言模型中的物体幻觉问题，提出RVCD方法。通过在logit层面结合正负图像（含AI生成的单概念图像）进行对比解码，有效抑制虚假对象生成，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20569v1](http://arxiv.org/pdf/2505.20569v1)**

> **作者:** Jihoon Lee; Min Song
>
> **备注:** ACL Findings camera-ready version. Code is released at https://github.com/JiHoonLee9898/RVCD
>
> **摘要:** Despite significant advancements in Large Vision-Language Models, Object Hallucination (OH) remains a persistent challenge. Building upon prior studies on contrastive decoding that address this issue without requiring additional model training, we introduce RVCD (Retrieval Visual Contrastive Decoding), an advanced method to suppress OH. RVCD leverages both negative and positive images at the logit level, explicitly referencing AI-generated images designed to represent a single concept. Our approach demonstrates substantial improvements over existing decoding-based methods.
>
---
#### [new 040] Intelligent Incident Hypertension Prediction in Obstructive Sleep Apnea
- **分类: cs.CV; cs.LG**

- **简介: 该研究针对OSA患者五年内高血压预测难题，提出基于DCT与迁移学习的深度学习方法，首次整合多导睡眠图全信号，转化为2D特征并优化频域特征提取，提升模型准确率（AUC 72.88%）。**

- **链接: [http://arxiv.org/pdf/2505.20615v1](http://arxiv.org/pdf/2505.20615v1)**

> **作者:** Omid Halimi Milani; Ahmet Enis Cetin; Bharati Prasad
>
> **备注:** Accepted at EUSIPCO 2025. Camera-ready due June 20, 2025
>
> **摘要:** Obstructive sleep apnea (OSA) is a significant risk factor for hypertension, primarily due to intermittent hypoxia and sleep fragmentation. Predicting whether individuals with OSA will develop hypertension within five years remains a complex challenge. This study introduces a novel deep learning approach that integrates Discrete Cosine Transform (DCT)-based transfer learning to enhance prediction accuracy. We are the first to incorporate all polysomnography signals together for hypertension prediction, leveraging their collective information to improve model performance. Features were extracted from these signals and transformed into a 2D representation to utilize pre-trained 2D neural networks such as MobileNet, EfficientNet, and ResNet variants. To further improve feature learning, we introduced a DCT layer, which transforms input features into a frequency-based representation, preserving essential spectral information, decorrelating features, and enhancing robustness to noise. This frequency-domain approach, coupled with transfer learning, is especially beneficial for limited medical datasets, as it leverages rich representations from pre-trained networks to improve generalization. By strategically placing the DCT layer at deeper truncation depths within EfficientNet, our model achieved a best area under the curve (AUC) of 72.88%, demonstrating the effectiveness of frequency-domain feature extraction and transfer learning in predicting hypertension risk in OSA patients over a five-year period.
>
---
#### [new 041] Supervised and self-supervised land-cover segmentation & classification of the Biesbosch wetlands
- **分类: cs.CV; eess.IV; 68; I.4.6**

- **简介: 该论文属于湿地土地覆盖分割与分类任务，旨在解决高分辨率卫星图像标注数据稀缺的问题。提出结合监督学习与自监督预训练（U-Net模型+自编码器），在荷兰湿地Sentinel-2数据上实现88.23%精度，并开发高分辨率标注扩展至中分辨率的框架，同时发布公开数据集。**

- **链接: [http://arxiv.org/pdf/2505.21269v1](http://arxiv.org/pdf/2505.21269v1)**

> **作者:** Eva Gmelich Meijling; Roberto Del Prete; Arnoud Visser
>
> **备注:** 12 pages, presented at the Netherlands Conference on Computer Vision (NCCV), Utrecht, May 2025
>
> **摘要:** Accurate wetland land-cover classification is essential for environmental monitoring, biodiversity assessment, and sustainable ecosystem management. However, the scarcity of annotated data, especially for high-resolution satellite imagery, poses a significant challenge for supervised learning approaches. To tackle this issue, this study presents a methodology for wetland land-cover segmentation and classification that adopts both supervised and self-supervised learning (SSL). We train a U-Net model from scratch on Sentinel-2 imagery across six wetland regions in the Netherlands, achieving a baseline model accuracy of 85.26%. Addressing the limited availability of labeled data, the results show that SSL pretraining with an autoencoder can improve accuracy, especially for the high-resolution imagery where it is more difficult to obtain labeled data, reaching an accuracy of 88.23%. Furthermore, we introduce a framework to scale manually annotated high-resolution labels to medium-resolution inputs. While the quantitative performance between resolutions is comparable, high-resolution imagery provides significantly sharper segmentation boundaries and finer spatial detail. As part of this work, we also contribute a curated Sentinel-2 dataset with Dynamic World labels, tailored for wetland classification tasks and made publicly available.
>
---
#### [new 042] QwT-v2: Practical, Effective and Efficient Post-Training Quantization
- **分类: cs.CV**

- **简介: 该论文属于深度学习模型量化任务，旨在解决QwT方法存在的额外参数、计算延迟及硬件兼容性问题。提出QwT-v2通过轻量级通道仿射补偿模块，在减少资源消耗的同时提升量化精度，并实现硬件友好部署。**

- **链接: [http://arxiv.org/pdf/2505.20932v1](http://arxiv.org/pdf/2505.20932v1)**

> **作者:** Ningyuan Tang; Minghao Fu; Hao Yu; Jianxin Wu
>
> **摘要:** Network quantization is arguably one of the most practical network compression approaches for reducing the enormous resource consumption of modern deep neural networks. They usually require diverse and subtle design choices for specific architecture and tasks. Instead, the QwT method is a simple and general approach which introduces lightweight additional structures to improve quantization. But QwT incurs extra parameters and latency. More importantly, QwT is not compatible with many hardware platforms. In this paper, we propose QwT-v2, which not only enjoys all advantages of but also resolves major defects of QwT. By adopting a very lightweight channel-wise affine compensation (CWAC) module, QwT-v2 introduces significantly less extra parameters and computations compared to QwT, and at the same time matches or even outperforms QwT in accuracy. The compensation module of QwT-v2 can be integrated into quantization inference engines with little effort, which not only effectively removes the extra costs but also makes it compatible with most existing hardware platforms.
>
---
#### [new 043] Incorporating Flexible Image Conditioning into Text-to-Video Diffusion Models without Training
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本到视频生成任务，解决现有方法依赖资源密集微调且视觉条件受限的问题。提出FlexTI2V，通过将图像逆向为潜空间噪声、随机块交换融合特征及动态调整条件强度，实现任意位置/数量图像的无训练灵活视频生成。**

- **链接: [http://arxiv.org/pdf/2505.20629v1](http://arxiv.org/pdf/2505.20629v1)**

> **作者:** Bolin Lai; Sangmin Lee; Xu Cao; Xiang Li; James M. Rehg
>
> **备注:** 21 pages, 11 figures, 4 tables
>
> **摘要:** Text-image-to-video (TI2V) generation is a critical problem for controllable video generation using both semantic and visual conditions. Most existing methods typically add visual conditions to text-to-video (T2V) foundation models by finetuning, which is costly in resources and only limited to a few predefined conditioning settings. To tackle this issue, we introduce a unified formulation for TI2V generation with flexible visual conditioning. Furthermore, we propose an innovative training-free approach, dubbed FlexTI2V, that can condition T2V foundation models on an arbitrary amount of images at arbitrary positions. Specifically, we firstly invert the condition images to noisy representation in a latent space. Then, in the denoising process of T2V models, our method uses a novel random patch swapping strategy to incorporate visual features into video representations through local image patches. To balance creativity and fidelity, we use a dynamic control mechanism to adjust the strength of visual conditioning to each video frame. Extensive experiments validate that our method surpasses previous training-free image conditioning methods by a notable margin. We also show more insights of our method by detailed ablation study and analysis.
>
---
#### [new 044] Photography Perspective Composition: Towards Aesthetic Perspective Recommendation
- **分类: cs.CV**

- **简介: 该论文属于摄影构图优化任务，针对传统2D裁剪在处理主体排列不佳场景时的不足，提出摄影透视构图（PPC）方法。通过构建自动化数据集框架、生成优化视角视频及设计基于人类评价的透视质量评估模型，解决数据稀缺与评估标准缺失问题，辅助用户提升构图技能。**

- **链接: [http://arxiv.org/pdf/2505.20655v1](http://arxiv.org/pdf/2505.20655v1)**

> **作者:** Lujian Yao; Siming Zheng; Xinbin Yuan; Zhuoxuan Cai; Pu Wu; Jinwei Chen; Bo Li; Peng-Tao Jiang
>
> **摘要:** Traditional photography composition approaches are dominated by 2D cropping-based methods. However, these methods fall short when scenes contain poorly arranged subjects. Professional photographers often employ perspective adjustment as a form of 3D recomposition, modifying the projected 2D relationships between subjects while maintaining their actual spatial positions to achieve better compositional balance. Inspired by this artistic practice, we propose photography perspective composition (PPC), extending beyond traditional cropping-based methods. However, implementing the PPC faces significant challenges: the scarcity of perspective transformation datasets and undefined assessment criteria for perspective quality. To address these challenges, we present three key contributions: (1) An automated framework for building PPC datasets through expert photographs. (2) A video generation approach that demonstrates the transformation process from suboptimal to optimal perspectives. (3) A perspective quality assessment (PQA) model constructed based on human performance. Our approach is concise and requires no additional prompt instructions or camera trajectories, helping and guiding ordinary users to enhance their composition skills.
>
---
#### [new 045] MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属视频时间理解任务，针对多模态大语言模型（MLLMs）在细粒度时间推理上的不足，提出MUSEG方法。通过时间戳感知的多片段接地机制及分阶段奖励的强化学习框架，提升模型对视频时间信息的推理能力，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20715v1](http://arxiv.org/pdf/2505.20715v1)**

> **作者:** Fuwen Luo; Shengfeng Lou; Chi Chen; Ziyue Wang; Chenliang Li; Weizhou Shen; Jiyue Guo; Peng Li; Ming Yan; Ji Zhang; Fei Huang; Yang Liu
>
> **摘要:** Video temporal understanding is crucial for multimodal large language models (MLLMs) to reason over events in videos. Despite recent advances in general video understanding, current MLLMs still struggle with fine-grained temporal reasoning. While reinforcement learning (RL) has been explored to address this issue recently, existing RL approaches remain limited in effectiveness. In this work, we propose MUSEG, a novel RL-based method that enhances temporal understanding by introducing timestamp-aware multi-segment grounding. MUSEG enables MLLMs to align queries with multiple relevant video segments, promoting more comprehensive temporal reasoning. To facilitate effective learning, we design a customized RL training recipe with phased rewards that progressively guides the model toward temporally grounded reasoning. Extensive experiments on temporal grounding and time-sensitive video QA tasks demonstrate that MUSEG significantly outperforms existing methods and generalizes well across diverse temporal understanding scenarios. View our project at https://github.com/THUNLP-MT/MUSEG.
>
---
#### [new 046] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness
- **分类: cs.CV**

- **简介: 该论文提出MMPerspective基准，评估多模态大模型对透视几何的理解。通过10项任务（分感知、推理、鲁棒性三维度）及2711个图像样本，测试43个模型，发现其表面感知能力较强但组合推理与空间一致性不足，揭示模型结构与尺度对透视能力的影响。**

- **链接: [http://arxiv.org/pdf/2505.20426v1](http://arxiv.org/pdf/2505.20426v1)**

> **作者:** Yunlong Tang; Pinxin Liu; Mingqian Feng; Zhangyun Tan; Rui Mao; Chao Huang; Jing Bi; Yunzhong Xiao; Susan Liang; Hang Hua; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Chenliang Xu
>
> **摘要:** Understanding perspective is fundamental to human visual perception, yet the extent to which multimodal large language models (MLLMs) internalize perspective geometry remains unclear. We introduce MMPerspective, the first benchmark specifically designed to systematically evaluate MLLMs' understanding of perspective through 10 carefully crafted tasks across three complementary dimensions: Perspective Perception, Reasoning, and Robustness. Our benchmark comprises 2,711 real-world and synthetic image instances with 5,083 question-answer pairs that probe key capabilities, such as vanishing point perception and counting, perspective type reasoning, line relationship understanding in 3D space, invariance to perspective-preserving transformations, etc. Through a comprehensive evaluation of 43 state-of-the-art MLLMs, we uncover significant limitations: while models demonstrate competence on surface-level perceptual tasks, they struggle with compositional reasoning and maintaining spatial consistency under perturbations. Our analysis further reveals intriguing patterns between model architecture, scale, and perspective capabilities, highlighting both robustness bottlenecks and the benefits of chain-of-thought prompting. MMPerspective establishes a valuable testbed for diagnosing and advancing spatial understanding in vision-language systems. Resources available at: https://yunlong10.github.io/MMPerspective/
>
---
#### [new 047] Causality and "In-the-Wild" Video-Based Person Re-ID: A Survey
- **分类: cs.CV**

- **简介: 该论文属于视频Re-ID任务，针对现有模型依赖易变表面特征导致跨场景泛化差的问题，提出因果推理方法分类（生成解缠、领域不变建模、因果Transformer），分析方法、评估指标与挑战（如公平性、可解释性），并指明未来方向，推动因果建模与自监督学习的融合。**

- **链接: [http://arxiv.org/pdf/2505.20540v1](http://arxiv.org/pdf/2505.20540v1)**

> **作者:** Md Rashidunnabi; Kailash Hambarde; Hugo Proença
>
> **备注:** 30 pages, 9 figures
>
> **摘要:** Video-based person re-identification (Re-ID) remains brittle in real-world deployments despite impressive benchmark performance. Most existing models rely on superficial correlations such as clothing, background, or lighting that fail to generalize across domains, viewpoints, and temporal variations. This survey examines the emerging role of causal reasoning as a principled alternative to traditional correlation-based approaches in video-based Re-ID. We provide a structured and critical analysis of methods that leverage structural causal models, interventions, and counterfactual reasoning to isolate identity-specific features from confounding factors. The survey is organized around a novel taxonomy of causal Re-ID methods that spans generative disentanglement, domain-invariant modeling, and causal transformers. We review current evaluation metrics and introduce causal-specific robustness measures. In addition, we assess practical challenges of scalability, fairness, interpretability, and privacy that must be addressed for real-world adoption. Finally, we identify open problems and outline future research directions that integrate causal modeling with efficient architectures and self-supervised learning. This survey aims to establish a coherent foundation for causal video-based person Re-ID and to catalyze the next phase of research in this rapidly evolving domain.
>
---
#### [new 048] Frequency Composition for Compressed and Domain-Adaptive Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CoDA框架，联合优化模型压缩与领域自适应。针对资源受限设备需应对领域迁移的挑战，通过量化训练学习低频通用特征，测试时利用高频信息无源域适应，提升CIFAR10-C和ImageNet-C表现。**

- **链接: [http://arxiv.org/pdf/2505.20890v1](http://arxiv.org/pdf/2505.20890v1)**

> **作者:** Yoojin Kwon; Hongjun Suh; Wooseok Lee; Taesik Gong; Songyi Han; Hyung-Sin Kim
>
> **备注:** Work in progress
>
> **摘要:** Modern on-device neural network applications must operate under resource constraints while adapting to unpredictable domain shifts. However, this combined challenge-model compression and domain adaptation-remains largely unaddressed, as prior work has tackled each issue in isolation: compressed networks prioritize efficiency within a fixed domain, whereas large, capable models focus on handling domain shifts. In this work, we propose CoDA, a frequency composition-based framework that unifies compression and domain adaptation. During training, CoDA employs quantization-aware training (QAT) with low-frequency components, enabling a compressed model to selectively learn robust, generalizable features. At test time, it refines the compact model in a source-free manner (i.e., test-time adaptation, TTA), leveraging the full-frequency information from incoming data to adapt to target domains while treating high-frequency components as domain-specific cues. LFC are aligned with the trained distribution, while HFC unique to the target distribution are solely utilized for batch normalization. CoDA can be integrated synergistically into existing QAT and TTA methods. CoDA is evaluated on widely used domain-shift benchmarks, including CIFAR10-C and ImageNet-C, across various model architectures. With significant compression, it achieves accuracy improvements of 7.96%p on CIFAR10-C and 5.37%p on ImageNet-C over the full-precision TTA baseline.
>
---
#### [new 049] DriveRX: A Vision-Language Reasoning Model for Cross-Task Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶跨任务推理任务，旨在解决传统模型泛化不足及现有视觉语言模型模块孤立、静态监督限制多阶段决策的问题。提出AutoDriveRL框架，将驾驶分解为四核心任务，以视觉语言问答形式优化，训练DriveRX模型，实现复杂场景下的鲁棒决策，优于GPT-4o。**

- **链接: [http://arxiv.org/pdf/2505.20665v1](http://arxiv.org/pdf/2505.20665v1)**

> **作者:** Muxi Diao; Lele Yang; Hongbo Yin; Zhexu Wang; Yejie Wang; Daxin Tian; Kongming Liang; Zhanyu Ma
>
> **摘要:** Autonomous driving requires real-time, robust reasoning across perception, prediction, planning, and behavior. However, conventional end-to-end models fail to generalize in complex scenarios due to the lack of structured reasoning. Recent vision-language models (VLMs) have been applied to driving tasks, but they typically rely on isolated modules and static supervision, limiting their ability to support multi-stage decision-making. We present AutoDriveRL, a unified training framework that formulates autonomous driving as a structured reasoning process over four core tasks. Each task is independently modeled as a vision-language question-answering problem and optimized using task-specific reward models, enabling fine-grained reinforcement signals at different reasoning stages. Within this framework, we train DriveRX, a cross-task reasoning VLM designed for real-time decision-making. DriveRX achieves strong performance on a public benchmark, outperforming GPT-4o in behavior reasoning and demonstrating robustness under complex or corrupted driving conditions. Our analysis further highlights the impact of vision encoder design and reward-guided reasoning compression. We will release the AutoDriveRL framework and the DriveRX model to support future research.
>
---
#### [new 050] Breaking Dataset Boundaries: Class-Agnostic Targeted Adversarial Attacks
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击任务，解决传统多目标攻击仅限训练类别且依赖黑盒模型数据的问题。提出CD-MTA方法，用图像条件输入替代类别嵌入，并设计类无关损失，使对抗样本可泛化到未见类别，无需访问目标数据。**

- **链接: [http://arxiv.org/pdf/2505.20782v1](http://arxiv.org/pdf/2505.20782v1)**

> **作者:** Taïga Gonçalves; Tomo Miyazaki; Shinichiro Omachi
>
> **摘要:** We present Cross-Domain Multi-Targeted Attack (CD-MTA), a method for generating adversarial examples that mislead image classifiers toward any target class, including those not seen during training. Traditional targeted attacks are limited to one class per model, requiring expensive retraining for each target. Multi-targeted attacks address this by introducing a perturbation generator with a conditional input to specify the target class. However, existing methods are constrained to classes observed during training and require access to the black-box model's training data--introducing a form of data leakage that undermines realistic evaluation in practical black-box scenarios. We identify overreliance on class embeddings as a key limitation, leading to overfitting and poor generalization to unseen classes. To address this, CD-MTA replaces class-level supervision with an image-based conditional input and introduces class-agnostic losses that align the perturbed and target images in the feature space. This design removes dependence on class semantics, thereby enabling generalization to unseen classes across datasets. Experiments on ImageNet and seven other datasets show that CD-MTA outperforms prior multi-targeted attacks in both standard and cross-domain settings--without accessing the black-box model's training data.
>
---
#### [new 051] Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals
- **分类: cs.CV**

- **简介: 该论文提出逆向虚拟试穿（VTOFF）任务，解决从真人着装图生成多品类标准化服装产品图的难题，针对遮挡干扰和单品类限制，设计了双DiT骨干网络与多模态注意力机制，并添加对齐模块优化细节，实现多模态信息融合提升生成质量。**

- **链接: [http://arxiv.org/pdf/2505.21062v1](http://arxiv.org/pdf/2505.21062v1)**

> **作者:** Davide Lobba; Fulvio Sanguigni; Bin Ren; Marcella Cornia; Rita Cucchiara; Nicu Sebe
>
> **摘要:** While virtual try-on (VTON) systems aim to render a garment onto a target person image, this paper tackles the novel task of virtual try-off (VTOFF), which addresses the inverse problem: generating standardized product images of garments from real-world photos of clothed individuals. Unlike VTON, which must resolve diverse pose and style variations, VTOFF benefits from a consistent and well-defined output format -- typically a flat, lay-down-style representation of the garment -- making it a promising tool for data generation and dataset enhancement. However, existing VTOFF approaches face two major limitations: (i) difficulty in disentangling garment features from occlusions and complex poses, often leading to visual artifacts, and (ii) restricted applicability to single-category garments (e.g., upper-body clothes only), limiting generalization. To address these challenges, we present Text-Enhanced MUlti-category Virtual Try-Off (TEMU-VTOFF), a novel architecture featuring a dual DiT-based backbone with a modified multimodal attention mechanism for robust garment feature extraction. Our architecture is designed to receive garment information from multiple modalities like images, text, and masks to work in a multi-category setting. Finally, we propose an additional alignment module to further refine the generated visual details. Experiments on VITON-HD and Dress Code datasets show that TEMU-VTOFF sets a new state-of-the-art on the VTOFF task, significantly improving both visual quality and fidelity to the target garments.
>
---
#### [new 052] Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment
- **分类: cs.CV**

- **简介: 该论文提出FOA-Attack方法，针对闭源多模态大语言模型（MLLMs）的对抗攻击任务。解决现有方法因忽略局部特征导致对抗样本迁移性差的问题。通过全局余弦相似度对齐与局部最优传输对齐结合动态模型加权策略，提升对抗攻击的跨模型迁移能力，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21494v1](http://arxiv.org/pdf/2505.21494v1)**

> **作者:** Xiaojun Jia; Sensen Gao; Simeng Qin; Tianyu Pang; Chao Du; Yihao Huang; Xinfeng Li; Yiming Li; Bo Li; Yang Liu
>
> **摘要:** Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.
>
---
#### [new 053] OccLE: Label-Efficient 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文提出OccLE方法，针对3D语义占用预测任务中全监督需大量标注、自监督效果差的问题，通过解耦语义与几何学习分支：语义分支利用2D模型生成伪标签，几何分支融合图像-LiDAR数据并采用半监督，最终融合特征网格提升性能。实验显示仅用10%标注达16.59%mIoU，实现标注高效。**

- **链接: [http://arxiv.org/pdf/2505.20617v1](http://arxiv.org/pdf/2505.20617v1)**

> **作者:** Naiyu Fang; Zheyuan Zhou; Fayao Liu; Xulei Yang; Jiacheng Wei; Lemiao Qiu; Guosheng Lin
>
> **摘要:** 3D semantic occupancy prediction offers an intuitive and efficient scene understanding and has attracted significant interest in autonomous driving perception. Existing approaches either rely on full supervision, which demands costly voxel-level annotations, or on self-supervision, which provides limited guidance and yields suboptimal performance. To address these challenges, we propose OccLE, a Label-Efficient 3D Semantic Occupancy Prediction that takes images and LiDAR as inputs and maintains high performance with limited voxel annotations. Our intuition is to decouple the semantic and geometric learning tasks and then fuse the learned feature grids from both tasks for the final semantic occupancy prediction. Therefore, the semantic branch distills 2D foundation model to provide aligned pseudo labels for 2D and 3D semantic learning. The geometric branch integrates image and LiDAR inputs in cross-plane synergy based on their inherency, employing semi-supervision to enhance geometry learning. We fuse semantic-geometric feature grids through Dual Mamba and incorporate a scatter-accumulated projection to supervise unannotated prediction with aligned pseudo labels. Experiments show that OccLE achieves competitive performance with only 10% of voxel annotations, reaching a mIoU of 16.59% on the SemanticKITTI validation set.
>
---
#### [new 054] Facial Attribute Based Text Guided Face Anonymization
- **分类: cs.CV; I.4.9; I.2.10; I.4.8**

- **简介: 该论文属于面部匿名化任务，解决隐私法规限制下难以获取合规人脸数据的问题。提出基于扩散模型BrushNet的三阶段流程：RetinaNet检测人脸、VGG-Face提取特征、结合文本属性（年龄/性别等）生成自然不可识别的面部，实现隐私保护数据集构建。**

- **链接: [http://arxiv.org/pdf/2505.21002v1](http://arxiv.org/pdf/2505.21002v1)**

> **作者:** Mustafa İzzet Muştu; Hazım Kemal Ekenel
>
> **备注:** 6 pages, 5 figures, published in the Proceedings of the Joint visuAAL-GoodBrother Conference on Trustworthy Video- and Audio-Based Assistive Technologies
>
> **摘要:** The increasing prevalence of computer vision applications necessitates handling vast amounts of visual data, often containing personal information. While this technology offers significant benefits, it should not compromise privacy. Data privacy regulations emphasize the need for individual consent for processing personal data, hindering researchers' ability to collect high-quality datasets containing the faces of the individuals. This paper presents a deep learning-based face anonymization pipeline to overcome this challenge. Unlike most of the existing methods, our method leverages recent advancements in diffusion-based inpainting models, eliminating the need for training Generative Adversarial Networks. The pipeline employs a three-stage approach: face detection with RetinaNet, feature extraction with VGG-Face, and realistic face generation using the state-of-the-art BrushNet diffusion model. BrushNet utilizes the entire image, face masks, and text prompts specifying desired facial attributes like age, ethnicity, gender, and expression. This enables the generation of natural-looking images with unrecognizable individuals, facilitating the creation of privacy-compliant datasets for computer vision research.
>
---
#### [new 055] Advancing high-fidelity 3D and Texture Generation with 2.5D latents
- **分类: cs.CV**

- **简介: 该论文属于高保真3D几何与纹理联合生成任务，解决现有方法分离生成导致的不连贯与质量不均问题。提出2.5D latent统一表征，整合多视角图像数据，利用2D预训练模型生成2.5D中间结果，并通过轻量级转换模块高效生成精细3D模型，提升结构与纹理一致性。**

- **链接: [http://arxiv.org/pdf/2505.21050v1](http://arxiv.org/pdf/2505.21050v1)**

> **作者:** Xin Yang; Jiantao Lin; Yingjie Xu; Haodong Li; Yingcong Chen
>
> **摘要:** Despite the availability of large-scale 3D datasets and advancements in 3D generative models, the complexity and uneven quality of 3D geometry and texture data continue to hinder the performance of 3D generation techniques. In most existing approaches, 3D geometry and texture are generated in separate stages using different models and non-unified representations, frequently leading to unsatisfactory coherence between geometry and texture. To address these challenges, we propose a novel framework for joint generation of 3D geometry and texture. Specifically, we focus in generate a versatile 2.5D representations that can be seamlessly transformed between 2D and 3D. Our approach begins by integrating multiview RGB, normal, and coordinate images into a unified representation, termed as 2.5D latents. Next, we adapt pre-trained 2D foundation models for high-fidelity 2.5D generation, utilizing both text and image conditions. Finally, we introduce a lightweight 2.5D-to-3D refiner-decoder framework that efficiently generates detailed 3D representations from 2.5D images. Extensive experiments demonstrate that our model not only excels in generating high-quality 3D objects with coherent structure and color from text and image inputs but also significantly outperforms existing methods in geometry-conditioned texture generation.
>
---
#### [new 056] A Feature-level Bias Evaluation Framework for Facial Expression Recognition Models
- **分类: cs.CV**

- **简介: 该论文提出特征级偏见评估框架，解决面部表情识别模型在无测试集人口统计标签时的偏见分析问题，替代伪标签方法；引入统计模块确保结果显著性，实验显示更有效，揭示年龄、性别、种族等属性的显著偏见，助力公平模型选择。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20512v1](http://arxiv.org/pdf/2505.20512v1)**

> **作者:** Tangzheng Lian; Oya Celiktutan
>
> **备注:** Submitted to IEEE Transactions on Affective Computing
>
> **摘要:** Recent studies on fairness have shown that Facial Expression Recognition (FER) models exhibit biases toward certain visually perceived demographic groups. However, the limited availability of human-annotated demographic labels in public FER datasets has constrained the scope of such bias analysis. To overcome this limitation, some prior works have resorted to pseudo-demographic labels, which may distort bias evaluation results. Alternatively, in this paper, we propose a feature-level bias evaluation framework for evaluating demographic biases in FER models under the setting where demographic labels are unavailable in the test set. Extensive experiments demonstrate that our method more effectively evaluates demographic biases compared to existing approaches that rely on pseudo-demographic labels. Furthermore, we observe that many existing studies do not include statistical testing in their bias evaluations, raising concerns that some reported biases may not be statistically significant but rather due to randomness. To address this issue, we introduce a plug-and-play statistical module to ensure the statistical significance of biased evaluation results. A comprehensive bias analysis based on the proposed module is then conducted across three sensitive attributes (age, gender, and race), seven facial expressions, and multiple network architectures on a large-scale dataset, revealing the prominent demographic biases in FER and providing insights on selecting a fairer network architecture.
>
---
#### [new 057] Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对扩散模型求解逆问题时计算量大、收敛不佳的问题，提出DMILO和DMILO-PGD方法。前者用中间层优化减少内存消耗并引入稀疏偏差扩展模型范围；后者结合投影梯度下降避免次优收敛。实验验证了方法有效性，优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.20789v1](http://arxiv.org/pdf/2505.20789v1)**

> **作者:** Yang Zheng; Wen Li; Zhaoqiang Liu
>
> **备注:** ICML 2025
>
> **摘要:** Inverse problems (IPs) involve reconstructing signals from noisy observations. Traditional approaches often rely on handcrafted priors, which can fail to capture the complexity of real-world data. The advent of pre-trained generative models has introduced new paradigms, offering improved reconstructions by learning rich priors from data. Among these, diffusion models (DMs) have emerged as a powerful framework, achieving remarkable reconstruction performance across numerous IPs. However, existing DM-based methods frequently encounter issues such as heavy computational demands and suboptimal convergence. In this work, building upon the idea of the recent work DMPlug~\cite{wang2024dmplug}, we propose two novel methods, DMILO and DMILO-PGD, to address these challenges. Our first method, DMILO, employs intermediate layer optimization (ILO) to alleviate the memory burden inherent in DMPlug. Additionally, by introducing sparse deviations, we expand the range of DMs, enabling the exploration of underlying signals that may lie outside the range of the diffusion model. We further propose DMILO-PGD, which integrates ILO with projected gradient descent (PGD), thereby reducing the risk of suboptimal convergence. We provide an intuitive theoretical analysis of our approach under appropriate conditions and validate its superiority through extensive experiments on diverse image datasets, encompassing both linear and nonlinear IPs. Our results demonstrate significant performance gains over state-of-the-art methods, highlighting the effectiveness of DMILO and DMILO-PGD in addressing common challenges in DM-based IP solvers.
>
---
#### [new 058] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval
- **分类: cs.CV; cs.LG**

- **简介: 论文提出ConText-CIR框架，针对组合图像检索任务，解决现有方法在图像与文本修改语义表示上的不足。通过文本概念一致性损失函数和合成数据生成，提升图像与文本关键区域的注意力关联，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.20764v1](http://arxiv.org/pdf/2505.20764v1)**

> **作者:** Eric Xing; Pranavi Kolouju; Robert Pless; Abby Stylianou; Nathan Jacobs
>
> **备注:** 15 pages, 8 figures, 6 tables. CVPR 2025
>
> **摘要:** Composed image retrieval (CIR) is the task of retrieving a target image specified by a query image and a relative text that describes a semantic modification to the query image. Existing methods in CIR struggle to accurately represent the image and the text modification, resulting in subpar performance. To address this limitation, we introduce a CIR framework, ConText-CIR, trained with a Text Concept-Consistency loss that encourages the representations of noun phrases in the text modification to better attend to the relevant parts of the query image. To support training with this loss function, we also propose a synthetic data generation pipeline that creates training data from existing CIR datasets or unlabeled images. We show that these components together enable stronger performance on CIR tasks, setting a new state-of-the-art in composed image retrieval in both the supervised and zero-shot settings on multiple benchmark datasets, including CIRR and CIRCO. Source code, model checkpoints, and our new datasets are available at https://github.com/mvrl/ConText-CIR.
>
---
#### [new 059] DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response
- **分类: cs.CV; I.4.9**

- **简介: 该论文提出DisasterM3数据集，用于灾害评估与响应。针对复杂灾害场景下视觉语言模型（VLM）的不足，构建包含多灾害类型（36事件/10类）、多传感器（光学与SAR）和多任务（9项灾害相关任务）的遥感数据集，通过微调VLM提升其跨传感器及灾害泛化能力，解决现有模型在灾害分析中的性能缺陷。**

- **链接: [http://arxiv.org/pdf/2505.21089v1](http://arxiv.org/pdf/2505.21089v1)**

> **作者:** Junjue Wang; Weihao Xuan; Heli Qi; Zhihao Liu; Kunyi Liu; Yuhan Wu; Hongruixuan Chen; Jian Song; Junshi Xia; Zhuo Zheng; Naoto Yokoya
>
> **备注:** A multi-hazard, multi-sensor, and multi-task vision-language dataset for global-scale disaster assessment and response
>
> **摘要:** Large vision-language models (VLMs) have made great achievements in Earth vision. However, complex disaster scenes with diverse disaster types, geographic regions, and satellite sensors have posed new challenges for VLM applications. To fill this gap, we curate a remote sensing vision-language dataset (DisasterM3) for global-scale disaster assessment and response. DisasterM3 includes 26,988 bi-temporal satellite images and 123k instruction pairs across 5 continents, with three characteristics: 1) Multi-hazard: DisasterM3 involves 36 historical disaster events with significant impacts, which are categorized into 10 common natural and man-made disasters. 2)Multi-sensor: Extreme weather during disasters often hinders optical sensor imaging, making it necessary to combine Synthetic Aperture Radar (SAR) imagery for post-disaster scenes. 3) Multi-task: Based on real-world scenarios, DisasterM3 includes 9 disaster-related visual perception and reasoning tasks, harnessing the full potential of VLM's reasoning ability with progressing from disaster-bearing body recognition to structural damage assessment and object relational reasoning, culminating in the generation of long-form disaster reports. We extensively evaluated 14 generic and remote sensing VLMs on our benchmark, revealing that state-of-the-art models struggle with the disaster tasks, largely due to the lack of a disaster-specific corpus, cross-sensor gap, and damage object counting insensitivity. Focusing on these issues, we fine-tune four VLMs using our dataset and achieve stable improvements across all tasks, with robust cross-sensor and cross-disaster generalization capabilities.
>
---
#### [new 060] Efficient Leaf Disease Classification and Segmentation using Midpoint Normalization Technique and Attention Mechanism
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出基于Midpoint Normalization（MPN）和注意力机制的两阶段方法，用于植物叶片疾病分类与分割。针对数据稀缺和复杂背景问题，MPN优化图像预处理，结合SE块提升分类准确率至93%，并改进U-Net分割性能（Dice 72.44%，IoU 58.54%），实现高效轻量模型。**

- **链接: [http://arxiv.org/pdf/2505.21316v1](http://arxiv.org/pdf/2505.21316v1)**

> **作者:** Enam Ahmed Taufik; Antara Firoz Parsa; Seraj Al Mahmud Mostafa
>
> **备注:** Accepted in 2025 IEEE International Conference on Image Processing (ICIP)
>
> **摘要:** Enhancing plant disease detection from leaf imagery remains a persistent challenge due to scarce labeled data and complex contextual factors. We introduce a transformative two-stage methodology, Mid Point Normalization (MPN) for intelligent image preprocessing, coupled with sophisticated attention mechanisms that dynamically recalibrate feature representations. Our classification pipeline, merging MPN with Squeeze-and-Excitation (SE) blocks, achieves remarkable 93% accuracy while maintaining exceptional class-wise balance. The perfect F1 score attained for our target class exemplifies attention's power in adaptive feature refinement. For segmentation tasks, we seamlessly integrate identical attention blocks within U-Net architecture using MPN-enhanced inputs, delivering compelling performance gains with 72.44% Dice score and 58.54% IoU, substantially outperforming baseline implementations. Beyond superior accuracy metrics, our approach yields computationally efficient, lightweight architectures perfectly suited for real-world computer vision applications.
>
---
#### [new 061] Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers
- **分类: cs.CV; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出Paper2Poster系统，解决学术海报自动化生成任务。针对长论文压缩为视觉连贯海报的挑战，团队构建首个评估基准（含视觉语义、文本连贯性等指标），设计多阶段Pipeline（解析论文、二叉树布局规划、渲染优化），开源方案性能超GPT-4o，成本仅$0.005。**

- **链接: [http://arxiv.org/pdf/2505.21497v1](http://arxiv.org/pdf/2505.21497v1)**

> **作者:** Wei Pang; Kevin Qinghong Lin; Xiangru Jian; Xi He; Philip Torr
>
> **备注:** Project Page: https://github.com/Paper2Poster/Paper2Poster
>
> **摘要:** Academic poster generation is a crucial yet challenging task in scientific communication, requiring the compression of long-context interleaved documents into a single, visually coherent page. To address this challenge, we introduce the first benchmark and metric suite for poster generation, which pairs recent conference papers with author-designed posters and evaluates outputs on (i)Visual Quality-semantic alignment with human posters, (ii)Textual Coherence-language fluency, (iii)Holistic Assessment-six fine-grained aesthetic and informational criteria scored by a VLM-as-judge, and notably (iv)PaperQuiz-the poster's ability to convey core paper content as measured by VLMs answering generated quizzes. Building on this benchmark, we propose PosterAgent, a top-down, visual-in-the-loop multi-agent pipeline: the (a)Parser distills the paper into a structured asset library; the (b)Planner aligns text-visual pairs into a binary-tree layout that preserves reading order and spatial balance; and the (c)Painter-Commenter loop refines each panel by executing rendering code and using VLM feedback to eliminate overflow and ensure alignment. In our comprehensive evaluation, we find that GPT-4o outputs-though visually appealing at first glance-often exhibit noisy text and poor PaperQuiz scores, and we find that reader engagement is the primary aesthetic bottleneck, as human-designed posters rely largely on visual semantics to convey meaning. Our fully open-source variants (e.g. based on the Qwen-2.5 series) outperform existing 4o-driven multi-agent systems across nearly all metrics, while using 87% fewer tokens. It transforms a 22-page paper into a finalized yet editable .pptx poster - all for just $0.005. These findings chart clear directions for the next generation of fully automated poster-generation models. The code and datasets are available at https://github.com/Paper2Poster/Paper2Poster.
>
---
#### [new 062] Electrolyzers-HSI: Close-Range Multi-Scene Hyperspectral Imaging Benchmark Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Electrolyzers-HSI数据集，旨在通过高光谱成像加速电解槽材料分类，解决可持续回收中关键材料识别效率低的问题。工作包括构建含RGB与HSI数据的多模态基准集，评估多种机器学习与Transformer模型，并开源数据支持智能回收研究。**

- **链接: [http://arxiv.org/pdf/2505.20507v1](http://arxiv.org/pdf/2505.20507v1)**

> **作者:** Elias Arbash; Ahmed Jamal Afifi; Ymane Belahsen; Margret Fuchs; Pedram Ghamisi; Paul Scheunders; Richard Gloaguen
>
> **摘要:** The global challenge of sustainable recycling demands automated, fast, and accurate, state-of-the-art (SOTA) material detection systems that act as a bedrock for a circular economy. Democratizing access to these cutting-edge solutions that enable real-time waste analysis is essential for scaling up recycling efforts and fostering the Green Deal. In response, we introduce \textbf{Electrolyzers-HSI}, a novel multimodal benchmark dataset designed to accelerate the recovery of critical raw materials through accurate electrolyzer materials classification. The dataset comprises 55 co-registered high-resolution RGB images and hyperspectral imaging (HSI) data cubes spanning the 400--2500 nm spectral range, yielding over 4.2 million pixel vectors and 424,169 labeled ones. This enables non-invasive spectral analysis of shredded electrolyzer samples, supporting quantitative and qualitative material classification and spectral properties investigation. We evaluate a suite of baseline machine learning (ML) methods alongside SOTA transformer-based deep learning (DL) architectures, including Vision Transformer, SpectralFormer, and the Multimodal Fusion Transformer, to investigate architectural bottlenecks for further efficiency optimisation when deploying transformers in material identification. We implement zero-shot detection techniques and majority voting across pixel-level predictions to establish object-level classification robustness. In adherence to the FAIR data principles, the electrolyzers-HSI dataset and accompanying codebase are openly available at https://github.com/hifexplo/Electrolyzers-HSI and https://rodare.hzdr.de/record/3668, supporting reproducible research and facilitating the broader adoption of smart and sustainable e-waste recycling solutions.
>
---
#### [new 063] Generalizable and Relightable Gaussian Splatting for Human Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文提出GRGS框架，用于3D人体新视角合成任务。解决现有方法在复杂光照下泛化性差、物理真实性不足的问题。通过光照感知几何优化模块（LGR）重建稳定几何，结合物理渲染模块（PGNR）支持可编辑光照，并设计2D-3D投影训练降低计算成本，实现高质量、可重光照的跨场景重建。**

- **链接: [http://arxiv.org/pdf/2505.21502v1](http://arxiv.org/pdf/2505.21502v1)**

> **作者:** Yipengjing Sun; Chenyang Wang; Shunyuan Zheng; Zonglin Li; Shengping Zhang; Xiangyang Ji
>
> **备注:** Project Webpage: https://sypj-98.github.io/grgs/
>
> **摘要:** We propose GRGS, a generalizable and relightable 3D Gaussian framework for high-fidelity human novel view synthesis under diverse lighting conditions. Unlike existing methods that rely on per-character optimization or ignore physical constraints, GRGS adopts a feed-forward, fully supervised strategy that projects geometry, material, and illumination cues from multi-view 2D observations into 3D Gaussian representations. Specifically, to reconstruct lighting-invariant geometry, we introduce a Lighting-aware Geometry Refinement (LGR) module trained on synthetically relit data to predict accurate depth and surface normals. Based on the high-quality geometry, a Physically Grounded Neural Rendering (PGNR) module is further proposed to integrate neural prediction with physics-based shading, supporting editable relighting with shadows and indirect illumination. Besides, we design a 2D-to-3D projection training scheme that leverages differentiable supervision from ambient occlusion, direct, and indirect lighting maps, which alleviates the computational cost of explicit ray tracing. Extensive experiments demonstrate that GRGS achieves superior visual quality, geometric consistency, and generalization across characters and lighting conditions.
>
---
#### [new 064] RoBiS: Robust Binary Segmentation for High-Resolution Industrial Images
- **分类: cs.CV**

- **简介: 该论文提出RoBiS框架，用于高分辨率工业图像的无监督异常检测。针对MVTec AD2基准中复杂场景导致的性能下降问题，其通过Swin-Cropping预处理保留小异常信息、数据增强提升鲁棒性，并结合统计方法与MEBin实现自适应二值化，最终用SAM优化分割结果，显著提升SegF1指标。**

- **链接: [http://arxiv.org/pdf/2505.21152v1](http://arxiv.org/pdf/2505.21152v1)**

> **作者:** Xurui Li; Zhonesheng Jiang; Tingxuan Ai; Yu Zhou
>
> **摘要:** Robust unsupervised anomaly detection (AD) in real-world scenarios is an important task. Current methods exhibit severe performance degradation on the MVTec AD 2 benchmark due to its complex real-world challenges. To solve this problem, we propose a robust framework RoBiS, which consists of three core modules: (1) Swin-Cropping, a high-resolution image pre-processing strategy to preserve the information of small anomalies through overlapping window cropping. (2) The data augmentation of noise addition and lighting simulation is carried out on the training data to improve the robustness of AD model. We use INP-Former as our baseline, which could generate better results on the various sub-images. (3) The traditional statistical-based binarization strategy (mean+3std) is combined with our previous work, MEBin (published in CVPR2025), for joint adaptive binarization. Then, SAM is further employed to refine the segmentation results. Compared with some methods reported by the MVTec AD 2, our RoBiS achieves a 29.2% SegF1 improvement (from 21.8% to 51.00%) on Test_private and 29.82% SegF1 gains (from 16.7% to 46.52%) on Test_private_mixed. Code is available at https://github.com/xrli-U/RoBiS.
>
---
#### [new 065] RetroMotion: Retrocausal Motion Forecasting Models are Instructable
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出RetroMotion模型，属交互式运动预测任务，解决复杂场景下多代理轨迹预测的不确定性。采用多任务学习结合逆因果信息流，利用Transformer重新编码边际分布生成联合轨迹分布，并支持通过修改轨迹响应指令。在Waymo和Argoverse数据集达SOTA，可适应场景指令。**

- **链接: [http://arxiv.org/pdf/2505.20414v1](http://arxiv.org/pdf/2505.20414v1)**

> **作者:** Royden Wagner; Omer Sahin Tas; Felix Hauser; Marlon Steiner; Dominik Strutz; Abhishek Vivekanandan; Carlos Fernandez; Christoph Stiller
>
> **摘要:** Motion forecasts of road users (i.e., agents) vary in complexity as a function of scene constraints and interactive behavior. We address this with a multi-task learning method for motion forecasting that includes a retrocausal flow of information. The corresponding tasks are to forecast (1) marginal trajectory distributions for all modeled agents and (2) joint trajectory distributions for interacting agents. Using a transformer model, we generate the joint distributions by re-encoding marginal distributions followed by pairwise modeling. This incorporates a retrocausal flow of information from later points in marginal trajectories to earlier points in joint trajectories. Per trajectory point, we model positional uncertainty using compressed exponential power distributions. Notably, our method achieves state-of-the-art results in the Waymo Interaction Prediction dataset and generalizes well to the Argoverse 2 dataset. Additionally, our method provides an interface for issuing instructions through trajectory modifications. Our experiments show that regular training of motion forecasting leads to the ability to follow goal-based instructions and to adapt basic directional instructions to the scene context. Code: https://github.com/kit-mrt/future-motion
>
---
#### [new 066] Empowering Vector Graphics with Consistently Arbitrary Viewing and View-dependent Visibility
- **分类: cs.CV**

- **简介: 该论文属于文本生成3D矢量图形任务，解决跨视角一致性和视点相关遮挡问题。提出双分支优化框架Dream3DVG，通过3D高斯散射分支弥合文本与矢量图形的鸿沟，利用渐进式无分类器引导控制细节优化，并设计可见性感知渲染模块处理遮挡，提升多视角一致性与细节表现。**

- **链接: [http://arxiv.org/pdf/2505.21377v1](http://arxiv.org/pdf/2505.21377v1)**

> **作者:** Yidi Li; Jun Xiao; Zhengda Lu; Yiqun Wang; Haiyong Jiang
>
> **备注:** CVPR 2025
>
> **摘要:** This work presents a novel text-to-vector graphics generation approach, Dream3DVG, allowing for arbitrary viewpoint viewing, progressive detail optimization, and view-dependent occlusion awareness. Our approach is a dual-branch optimization framework, consisting of an auxiliary 3D Gaussian Splatting optimization branch and a 3D vector graphics optimization branch. The introduced 3DGS branch can bridge the domain gaps between text prompts and vector graphics with more consistent guidance. Moreover, 3DGS allows for progressive detail control by scheduling classifier-free guidance, facilitating guiding vector graphics with coarse shapes at the initial stages and finer details at later stages. We also improve the view-dependent occlusions by devising a visibility-awareness rendering module. Extensive results on 3D sketches and 3D iconographies, demonstrate the superiority of the method on different abstraction levels of details, cross-view consistency, and occlusion-aware stroke culling.
>
---
#### [new 067] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D语义理解任务，旨在解决2D先验方法因遮挡、模糊等导致的跨视角语义不一致问题。提出CCL-LGS框架，通过零样本跟踪对齐多视角SAM掩码，结合CLIP提取语义编码，并采用对比代码本学习模块优化语义特征，提升3D高斯语义场质量，减少渲染伪影。**

- **链接: [http://arxiv.org/pdf/2505.20469v1](http://arxiv.org/pdf/2505.20469v1)**

> **作者:** Lei Tian; Xiaomin Li; Liqian Ma; Hefei Huang; Zirui Zheng; Hao Yin; Taiqing Li; Huchuan Lu; Xu Jia
>
> **摘要:** Recent advances in 3D reconstruction techniques and vision-language models have fueled significant progress in 3D semantic understanding, a capability critical to robotics, autonomous driving, and virtual/augmented reality. However, methods that rely on 2D priors are prone to a critical challenge: cross-view semantic inconsistencies induced by occlusion, image blur, and view-dependent variations. These inconsistencies, when propagated via projection supervision, deteriorate the quality of 3D Gaussian semantic fields and introduce artifacts in the rendered outputs. To mitigate this limitation, we propose CCL-LGS, a novel framework that enforces view-consistent semantic supervision by integrating multi-view semantic cues. Specifically, our approach first employs a zero-shot tracker to align a set of SAM-generated 2D masks and reliably identify their corresponding categories. Next, we utilize CLIP to extract robust semantic encodings across views. Finally, our Contrastive Codebook Learning (CCL) module distills discriminative semantic features by enforcing intra-class compactness and inter-class distinctiveness. In contrast to previous methods that directly apply CLIP to imperfect masks, our framework explicitly resolves semantic conflicts while preserving category discriminability. Extensive experiments demonstrate that CCL-LGS outperforms previous state-of-the-art methods. Our project page is available at https://epsilontl.github.io/CCL-LGS/.
>
---
#### [new 068] Supervised Contrastive Learning for Ordinal Engagement Measurement
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出基于监督对比学习的视频学生参与度等级分类方法，解决类别不平衡与顺序信息建模问题。通过提取多模态特征、时间序列增强及顺序分类器框架，在DAiSEE数据集验证有效。**

- **链接: [http://arxiv.org/pdf/2505.20676v1](http://arxiv.org/pdf/2505.20676v1)**

> **作者:** Sadaf Safa; Ali Abedi; Shehroz S. Khan
>
> **备注:** 9 pages, 1 figure, 5 tables
>
> **摘要:** Student engagement plays a crucial role in the successful delivery of educational programs. Automated engagement measurement helps instructors monitor student participation, identify disengagement, and adapt their teaching strategies to enhance learning outcomes effectively. This paper identifies two key challenges in this problem: class imbalance and incorporating order into engagement levels rather than treating it as mere categories. Then, a novel approach to video-based student engagement measurement in virtual learning environments is proposed that utilizes supervised contrastive learning for ordinal classification of engagement. Various affective and behavioral features are extracted from video samples and utilized to train ordinal classifiers within a supervised contrastive learning framework (with a sequential classifier as the encoder). A key step involves the application of diverse time-series data augmentation techniques to these feature vectors, enhancing model training. The effectiveness of the proposed method was evaluated using a publicly available dataset for engagement measurement, DAiSEE, containing videos of students who participated in virtual learning programs. The results demonstrate the robust ability of the proposed method for the classification of the engagement level. This approach promises a significant contribution to understanding and enhancing student engagement in virtual learning environments.
>
---
#### [new 069] YOLO-SPCI: Enhancing Remote Sensing Object Detection via Selective-Perspective-Class Integration
- **分类: cs.CV**

- **简介: 该论文针对遥感图像目标检测中尺度变化大、目标密集及背景复杂的问题，提出YOLO-SPCI模型。通过集成选择性视角-类别融合模块（SPCI），改进YOLOv8的多尺度特征表达能力，在NWPU VHR-10数据集上取得更优检测效果。**

- **链接: [http://arxiv.org/pdf/2505.21370v1](http://arxiv.org/pdf/2505.21370v1)**

> **作者:** Xinyuan Wang; Lian Peng; Xiangcheng Li; Yilin He; KinTak U
>
> **摘要:** Object detection in remote sensing imagery remains a challenging task due to extreme scale variation, dense object distributions, and cluttered backgrounds. While recent detectors such as YOLOv8 have shown promising results, their backbone architectures lack explicit mechanisms to guide multi-scale feature refinement, limiting performance on high-resolution aerial data. In this work, we propose YOLO-SPCI, an attention-enhanced detection framework that introduces a lightweight Selective-Perspective-Class Integration (SPCI) module to improve feature representation. The SPCI module integrates three components: a Selective Stream Gate (SSG) for adaptive regulation of global feature flow, a Perspective Fusion Module (PFM) for context-aware multi-scale integration, and a Class Discrimination Module (CDM) to enhance inter-class separability. We embed two SPCI blocks into the P3 and P5 stages of the YOLOv8 backbone, enabling effective refinement while preserving compatibility with the original neck and head. Experiments on the NWPU VHR-10 dataset demonstrate that YOLO-SPCI achieves superior performance compared to state-of-the-art detectors.
>
---
#### [new 070] See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文提出LIAR框架，解决夜间光照不足导致的占用预测问题。通过选择性低光增强(SLLIE)区分真实黑暗与足够光照，结合2D光照引导采样(2D-IGS)缓解欠曝，及3D光照驱动投影(3D-IDP)优化过曝区域，提升夜间3D占用预测性能。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20641v1](http://arxiv.org/pdf/2505.20641v1)**

> **作者:** Yuan Wu; Zhiqiang Yan; Yigong Zhang; Xiang Li; ian Yang
>
> **摘要:** Occupancy prediction aims to estimate the 3D spatial distribution of occupied regions along with their corresponding semantic labels. Existing vision-based methods perform well on daytime benchmarks but struggle in nighttime scenarios due to limited visibility and challenging lighting conditions. To address these challenges, we propose \textbf{LIAR}, a novel framework that learns illumination-affined representations. LIAR first introduces Selective Low-light Image Enhancement (SLLIE), which leverages the illumination priors from daytime scenes to adaptively determine whether a nighttime image is genuinely dark or sufficiently well-lit, enabling more targeted global enhancement. Building on the illumination maps generated by SLLIE, LIAR further incorporates two illumination-aware components: 2D Illumination-guided Sampling (2D-IGS) and 3D Illumination-driven Projection (3D-IDP), to respectively tackle local underexposure and overexposure. Specifically, 2D-IGS modulates feature sampling positions according to illumination maps, assigning larger offsets to darker regions and smaller ones to brighter regions, thereby alleviating feature degradation in underexposed areas. Subsequently, 3D-IDP enhances semantic understanding in overexposed regions by constructing illumination intensity fields and supplying refined residual queries to the BEV context refinement process. Extensive experiments on both real and synthetic datasets demonstrate the superior performance of LIAR under challenging nighttime scenarios. The source code and pretrained models are available \href{https://github.com/yanzq95/LIAR}{here}.
>
---
#### [new 071] Making Every Event Count: Balancing Data Efficiency and Accuracy in Event Camera Subsampling
- **分类: cs.CV**

- **简介: 该论文属于事件相机数据下采样优化任务，旨在解决高事件率导致的数据传输与处理效率问题，同时保持视觉任务精度。研究评估了六种硬件友好型下采样方法，提出基于事件密度的因果下采样策略，提升稀疏场景下的分类准确率，并分析关键影响因素如超参数敏感性。**

- **链接: [http://arxiv.org/pdf/2505.21187v1](http://arxiv.org/pdf/2505.21187v1)**

> **作者:** Hesam Araghi; Jan van Gemert; Nergis Tomen
>
> **摘要:** Event cameras offer high temporal resolution and power efficiency, making them well-suited for edge AI applications. However, their high event rates present challenges for data transmission and processing. Subsampling methods provide a practical solution, but their effect on downstream visual tasks remains underexplored. In this work, we systematically evaluate six hardware-friendly subsampling methods using convolutional neural networks for event video classification on various benchmark datasets. We hypothesize that events from high-density regions carry more task-relevant information and are therefore better suited for subsampling. To test this, we introduce a simple causal density-based subsampling method, demonstrating improved classification accuracy in sparse regimes. Our analysis further highlights key factors affecting subsampling performance, including sensitivity to hyperparameters and failure cases in scenarios with large event count variance. These findings provide insights for utilization of hardware-efficient subsampling strategies that balance data efficiency and task accuracy. The code for this paper will be released at: https://github.com/hesamaraghi/event-camera-subsampling-methods.
>
---
#### [new 072] Hierarchical Instruction-aware Embodied Visual Tracking
- **分类: cs.CV**

- **简介: 该论文属于用户为中心的具身视觉跟踪（UC-EVT）任务，旨在解决高阶用户指令与低阶智能体动作间的鸿沟。提出分层指令感知方法HIEVT，通过LLM生成空间目标中介指令语义，再用RL策略实现目标追踪，提升速度与泛化性，经百万轨迹数据验证其环境适应性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20710v1](http://arxiv.org/pdf/2505.20710v1)**

> **作者:** Kui Wu; Hao Chen; Churan Wang; Fakhri Karray; Zhoujun Li; Yizhou Wang; Fangwei Zhong
>
> **摘要:** User-Centric Embodied Visual Tracking (UC-EVT) presents a novel challenge for reinforcement learning-based models due to the substantial gap between high-level user instructions and low-level agent actions. While recent advancements in language models (e.g., LLMs, VLMs, VLAs) have improved instruction comprehension, these models face critical limitations in either inference speed (LLMs, VLMs) or generalizability (VLAs) for UC-EVT tasks. To address these challenges, we propose \textbf{Hierarchical Instruction-aware Embodied Visual Tracking (HIEVT)} agent, which bridges instruction comprehension and action generation using \textit{spatial goals} as intermediaries. HIEVT first introduces \textit{LLM-based Semantic-Spatial Goal Aligner} to translate diverse human instructions into spatial goals that directly annotate the desired spatial position. Then the \textit{RL-based Adaptive Goal-Aligned Policy}, a general offline policy, enables the tracker to position the target as specified by the spatial goal. To benchmark UC-EVT tasks, we collect over ten million trajectories for training and evaluate across one seen environment and nine unseen challenging environments. Extensive experiments and real-world deployments demonstrate the robustness and generalizability of HIEVT across diverse environments, varying target dynamics, and complex instruction combinations. The complete project is available at https://sites.google.com/view/hievt.
>
---
#### [new 073] FastFace: Tuning Identity Preservation in Distilled Diffusion via Guidance and Attention
- **分类: cs.CV**

- **简介: 该论文属于个性化生成任务，旨在解决现有身份保持适配器需联合训练导致推理速度慢的问题。提出FastFace框架，通过重新设计分类器自由引导和注意力机制，实现无训练适配蒸馏加速的扩散模型，提升身份相似度与生成质量，并开发了独立评估协议。**

- **链接: [http://arxiv.org/pdf/2505.21144v1](http://arxiv.org/pdf/2505.21144v1)**

> **作者:** Sergey Karpukhin; Vadim Titov; Andrey Kuznetsov; Aibek Alanov
>
> **备注:** code available at https://github.com/shredder67/fastface
>
> **摘要:** In latest years plethora of identity-preserving adapters for a personalized generation with diffusion models have been released. Their main disadvantage is that they are dominantly trained jointly with base diffusion models, which suffer from slow multi-step inference. This work aims to tackle the challenge of training-free adaptation of pretrained ID-adapters to diffusion models accelerated via distillation - through careful re-design of classifier-free guidance for few-step stylistic generation and attention manipulation mechanisms in decoupled blocks to improve identity similarity and fidelity, we propose universal FastFace framework. Additionally, we develop a disentangled public evaluation protocol for id-preserving adapters.
>
---
#### [new 074] IndustryEQA: Pushing the Frontiers of Embodied Question Answering in Industrial Scenarios
- **分类: cs.CV**

- **简介: 该论文提出IndustryEQA，首个针对工业场景安全的具身问答基准，解决现有EQA方法忽略工业环境安全与复杂推理的问题。基于NVIDIA平台构建含高保真视频、六类标注（设备/人类安全、物体识别等）及1344个问答对的工业仓库数据集，并提供评估框架与基线模型，推动安全可靠的工业智能代理研发。**

- **链接: [http://arxiv.org/pdf/2505.20640v1](http://arxiv.org/pdf/2505.20640v1)**

> **作者:** Yifan Li; Yuhang Chen; Anh Dao; Lichi Li; Zhongyi Cai; Zhen Tan; Tianlong Chen; Yu Kong
>
> **备注:** v1.0
>
> **摘要:** Existing Embodied Question Answering (EQA) benchmarks primarily focus on household environments, often overlooking safety-critical aspects and reasoning processes pertinent to industrial settings. This drawback limits the evaluation of agent readiness for real-world industrial applications. To bridge this, we introduce IndustryEQA, the first benchmark dedicated to evaluating embodied agent capabilities within safety-critical warehouse scenarios. Built upon the NVIDIA Isaac Sim platform, IndustryEQA provides high-fidelity episodic memory videos featuring diverse industrial assets, dynamic human agents, and carefully designed hazardous situations inspired by real-world safety guidelines. The benchmark includes rich annotations covering six categories: equipment safety, human safety, object recognition, attribute recognition, temporal understanding, and spatial understanding. Besides, it also provides extra reasoning evaluation based on these categories. Specifically, it comprises 971 question-answer pairs generated from small warehouse and 373 pairs from large ones, incorporating scenarios with and without human. We further propose a comprehensive evaluation framework, including various baseline models, to assess their general perception and reasoning abilities in industrial environments. IndustryEQA aims to steer EQA research towards developing more robust, safety-aware, and practically applicable embodied agents for complex industrial environments. Benchmark and codes are available.
>
---
#### [new 075] HoliTom: Holistic Token Merging for Fast Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频大语言模型（video LLMs）优化任务，旨在解决冗余视频标记导致的计算低效问题。现有方法或因内在计算开销大，或仅处理局部时空冗余，忽略全局时序关联。论文提出HoliTom框架，结合外层全局时序分段与内层标记相似度合并，减少90%视觉标记，使FLOPs降至6.9%且性能保持99.1%，加速推理速度。**

- **链接: [http://arxiv.org/pdf/2505.21334v1](http://arxiv.org/pdf/2505.21334v1)**

> **作者:** Kele Shao; Keda Tao; Can Qin; Haoxuan You; Yang Sui; Huan Wang
>
> **摘要:** Video large language models (video LLMs) excel at video comprehension but face significant computational inefficiency due to redundant video tokens. Existing token pruning methods offer solutions. However, approaches operating within the LLM (inner-LLM pruning), such as FastV, incur intrinsic computational overhead in shallow layers. In contrast, methods performing token pruning before the LLM (outer-LLM pruning) primarily address spatial redundancy within individual frames or limited temporal windows, neglecting the crucial global temporal dynamics and correlations across longer video sequences. This leads to sub-optimal spatio-temporal reduction and does not leverage video compressibility fully. Crucially, the synergistic potential and mutual influence of combining these strategies remain unexplored. To further reduce redundancy, we introduce HoliTom, a novel training-free holistic token merging framework. HoliTom employs outer-LLM pruning through global redundancy-aware temporal segmentation, followed by spatial-temporal merging to reduce visual tokens by over 90%, significantly alleviating the LLM's computational burden. Complementing this, we introduce a robust inner-LLM token similarity-based merging approach, designed for superior performance and compatibility with outer-LLM pruning. Evaluations demonstrate our method's promising efficiency-performance trade-off on LLaVA-OneVision-7B, reducing computational costs to 6.9% of FLOPs while maintaining 99.1% of the original performance. Furthermore, we achieve a 2.28x reduction in Time-To-First-Token (TTFT) and a 1.32x acceleration in decoding throughput, highlighting the practical benefits of our integrated pruning approach for efficient video LLMs inference.
>
---
#### [new 076] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文提出OmniSync框架，解决跨场景唇同步任务，针对传统方法依赖参考帧、受姿态/遮挡限制及音频弱条件导致的唇形泄漏问题。其创新包括无遮罩扩散Transformer直接编辑帧、流匹配噪声初始化保持身份/姿态一致性，及动态时空引导增强音频指导，并建立首个AI生成视频评估基准。**

- **链接: [http://arxiv.org/pdf/2505.21448v1](http://arxiv.org/pdf/2505.21448v1)**

> **作者:** Ziqiao Peng; Jiwen Liu; Haoxian Zhang; Xiaoqiang Liu; Songlin Tang; Pengfei Wan; Di Zhang; Hongyan Liu; Jun He
>
> **备注:** https://ziqiaopeng.github.io/OmniSync/
>
> **摘要:** Lip synchronization is the task of aligning a speaker's lip movements in video with corresponding speech audio, and it is essential for creating realistic, expressive video content. However, existing methods often rely on reference frames and masked-frame inpainting, which limit their robustness to identity consistency, pose variations, facial occlusions, and stylized content. In addition, since audio signals provide weaker conditioning than visual cues, lip shape leakage from the original video will affect lip sync quality. In this paper, we present OmniSync, a universal lip synchronization framework for diverse visual scenarios. Our approach introduces a mask-free training paradigm using Diffusion Transformer models for direct frame editing without explicit masks, enabling unlimited-duration inference while maintaining natural facial dynamics and preserving character identity. During inference, we propose a flow-matching-based progressive noise initialization to ensure pose and identity consistency, while allowing precise mouth-region editing. To address the weak conditioning signal of audio, we develop a Dynamic Spatiotemporal Classifier-Free Guidance (DS-CFG) mechanism that adaptively adjusts guidance strength over time and space. We also establish the AIGC-LipSync Benchmark, the first evaluation suite for lip synchronization in diverse AI-generated videos. Extensive experiments demonstrate that OmniSync significantly outperforms prior methods in both visual quality and lip sync accuracy, achieving superior results in both real-world and AI-generated videos.
>
---
#### [new 077] VisAlgae 2023: A Dataset and Challenge for Algae Detection in Microscopy Images
- **分类: cs.CV**

- **简介: 该论文介绍VisAlgae2023挑战赛及配套数据集，旨在通过竞赛推动显微图像中微藻检测技术发展，解决小目标检测、运动模糊及复杂背景干扰问题。工作包含构建含1000张六类藻类图像的数据集，吸引369队参与，分析优胜方法以提升检测精度，促进生态与技术应用。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20687v1](http://arxiv.org/pdf/2505.20687v1)**

> **作者:** Mingxuan Sun; Juntao Jiang; Zhiqiang Yang; Shenao Kong; Jiamin Qi; Jianru Shang; Shuangling Luo; Wanfa Sun; Tianyi Wang; Yanqi Wang; Qixuan Wang; Tingjian Dai; Tianxiang Chen; Jinming Zhang; Xuerui Zhang; Yuepeng He; Pengcheng Fu; Qiu Guan; Shizheng Zhou; Yanbo Yu; Qigui Jiang; Teng Zhou; Liuyong Shi; Hong Yan
>
> **摘要:** Microalgae, vital for ecological balance and economic sectors, present challenges in detection due to their diverse sizes and conditions. This paper summarizes the second "Vision Meets Algae" (VisAlgae 2023) Challenge, aiming to enhance high-throughput microalgae cell detection. The challenge, which attracted 369 participating teams, includes a dataset of 1000 images across six classes, featuring microalgae of varying sizes and distinct features. Participants faced tasks such as detecting small targets, handling motion blur, and complex backgrounds. The top 10 methods, outlined here, offer insights into overcoming these challenges and maximizing detection accuracy. This intersection of algae research and computer vision offers promise for ecological understanding and technological advancement. The dataset can be accessed at: https://github.com/juntaoJianggavin/Visalgae2023/.
>
---
#### [new 078] Scan-and-Print: Patch-level Data Summarization and Augmentation for Content-aware Layout Generation in Poster Design
- **分类: cs.CV**

- **简介: 该论文属于海报设计中的内容感知布局生成任务。针对现有方法因参数过多导致实时性和泛化性差的问题，提出Scan-and-Print方法：通过"扫描"筛选适配元素顶点的图像块，"打印"混合多图像块生成新样本，并引入顶点布局表示，使计算效率提升95.2%且保持高质量。**

- **链接: [http://arxiv.org/pdf/2505.20649v1](http://arxiv.org/pdf/2505.20649v1)**

> **作者:** HsiaoYuan Hsu; Yuxin Peng
>
> **备注:** Accepted to IJCAI 2025 (AI, Arts and Creativity). Project page is at https://thekinsley.github.io/Scan-and-Print/
>
> **摘要:** In AI-empowered poster design, content-aware layout generation is crucial for the on-image arrangement of visual-textual elements, e.g., logo, text, and underlay. To perceive the background images, existing work demanded a high parameter count that far exceeds the size of available training data, which has impeded the model's real-time performance and generalization ability. To address these challenges, we proposed a patch-level data summarization and augmentation approach, vividly named Scan-and-Print. Specifically, the scan procedure selects only the patches suitable for placing element vertices to perform fine-grained perception efficiently. Then, the print procedure mixes up the patches and vertices across two image-layout pairs to synthesize over 100% new samples in each epoch while preserving their plausibility. Besides, to facilitate the vertex-level operations, a vertex-based layout representation is introduced. Extensive experimental results on widely used benchmarks demonstrated that Scan-and-Print can generate visually appealing layouts with state-of-the-art quality while dramatically reducing computational bottleneck by 95.2%.
>
---
#### [new 079] Visual Product Graph: Bridging Visual Products And Composite Images For End-to-End Style Recommendations
- **分类: cs.CV**

- **简介: 该论文属于视觉推荐任务，旨在解决检索语义相似但视觉不同的产品并提供风格搭配的问题。提出Visual Product Graph（VPG）系统，通过物体检测、视觉嵌入等技术，构建产品与复合场景的关联网络，实现端到端风格推荐，部署于Pinterest提升用户参与度。**

- **链接: [http://arxiv.org/pdf/2505.21454v1](http://arxiv.org/pdf/2505.21454v1)**

> **作者:** Yue Li Du; Ben Alexander; Mikhail Antonenka; Rohan Mahadev; Hao-yu Wu; Dmitry Kislyuk
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Retrieving semantically similar but visually distinct contents has been a critical capability in visual search systems. In this work, we aim to tackle this problem with Visual Product Graph (VPG), leveraging high-performance infrastructure for storage and state-of-the-art computer vision models for image understanding. VPG is built to be an online real-time retrieval system that enables navigation from individual products to composite scenes containing those products, along with complementary recommendations. Our system not only offers contextual insights by showcasing how products can be styled in a context, but also provides recommendations for complementary products drawn from these inspirations. We discuss the essential components for building the Visual Product Graph, along with the core computer vision model improvements across object detection, foundational visual embeddings, and other visual signals. Our system achieves a 78.8% extremely similar@1 in end-to-end human relevance evaluations, and a 6% module engagement rate. The "Ways to Style It" module, powered by the Visual Product Graph technology, is deployed in production at Pinterest.
>
---
#### [new 080] Differentiable Solver Search for Fast Diffusion Sampling
- **分类: cs.CV**

- **简介: 该论文针对扩散模型采样步骤多、计算成本高的问题，提出基于可微搜索的求解器优化方法。通过分析传统多步法局限性，构建包含时间步与系数的搜索空间，搜索最优求解器。实验显示，新求解器使多种模型在10步内达2.33-2.40 FID，显著优于传统方法，且具通用性。**

- **链接: [http://arxiv.org/pdf/2505.21114v1](http://arxiv.org/pdf/2505.21114v1)**

> **作者:** Shuai Wang; Zexian Li; Qipeng zhang; Tianhui Song; Xubin Li; Tiezheng Ge; Bo Zheng; Limin Wang
>
> **备注:** accpeted on ICML25
>
> **摘要:** Diffusion models have demonstrated remarkable generation quality but at the cost of numerous function evaluations. Recently, advanced ODE-based solvers have been developed to mitigate the substantial computational demands of reverse-diffusion solving under limited sampling steps. However, these solvers, heavily inspired by Adams-like multistep methods, rely solely on t-related Lagrange interpolation. We show that t-related Lagrange interpolation is suboptimal for diffusion model and reveal a compact search space comprised of time steps and solver coefficients. Building on our analysis, we propose a novel differentiable solver search algorithm to identify more optimal solver. Equipped with the searched solver, rectified-flow models, e.g., SiT-XL/2 and FlowDCN-XL/2, achieve FID scores of 2.40 and 2.35, respectively, on ImageNet256 with only 10 steps. Meanwhile, DDPM model, DiT-XL/2, reaches a FID score of 2.33 with only 10 steps. Notably, our searched solver outperforms traditional solvers by a significant margin. Moreover, our searched solver demonstrates generality across various model architectures, resolutions, and model sizes.
>
---
#### [new 081] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于任务导向的部件分割任务，旨在解决现有视觉语言模型（VLMs）难以理解物体部件及其功能的问题。作者构建了含部件分割标注和任务指令的新基准InstructPart，提出通过微调提升性能的基线模型，推动VLMs在机器人等领域的应用。**

- **链接: [http://arxiv.org/pdf/2505.18291v1](http://arxiv.org/pdf/2505.18291v1)**

> **作者:** Zifu Wan; Yaqi Xie; Ce Zhang; Zhiqiu Lin; Zihan Wang; Simon Stepputtis; Deva Ramanan; Katia Sycara
>
> **备注:** Accepted by ACL 2025 Main. Project page: https://zifuwan.github.io/InstructPart/
>
> **摘要:** Large multimodal foundation models, particularly in the domains of language and vision, have significantly advanced various tasks, including robotics, autonomous driving, information retrieval, and grounding. However, many of these models perceive objects as indivisible, overlooking the components that constitute them. Understanding these components and their associated affordances provides valuable insights into an object's functionality, which is fundamental for performing a wide range of tasks. In this work, we introduce a novel real-world benchmark, InstructPart, comprising hand-labeled part segmentation annotations and task-oriented instructions to evaluate the performance of current models in understanding and executing part-level tasks within everyday contexts. Through our experiments, we demonstrate that task-oriented part segmentation remains a challenging problem, even for state-of-the-art Vision-Language Models (VLMs). In addition to our benchmark, we introduce a simple baseline that achieves a twofold performance improvement through fine-tuning with our dataset. With our dataset and benchmark, we aim to facilitate research on task-oriented part segmentation and enhance the applicability of VLMs across various domains, including robotics, virtual reality, information retrieval, and other related fields. Project website: https://zifuwan.github.io/InstructPart/.
>
---
#### [new 082] Understand, Think, and Answer: Advancing Visual Reasoning with Large Multimodal Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉推理任务，针对大语言模型在复杂组合推理中的不足，提出模仿人类"理解-思考-回答"的统一推理机制，通过单次前向传播实现端到端推理，无需多次交互或工具。构建334K视觉指令数据集，训练Griffon-R模型，在VSR、CLEVR等基准测试中取得先进性能。**

- **链接: [http://arxiv.org/pdf/2505.20753v1](http://arxiv.org/pdf/2505.20753v1)**

> **作者:** Yufei Zhan; Hongyin Zhao; Yousong Zhu; Shurong Zheng; Fan Yang; Ming Tang; Jinqiao Wang
>
> **备注:** Tech report
>
> **摘要:** Large Multimodal Models (LMMs) have recently demonstrated remarkable visual understanding performance on both vision-language and vision-centric tasks. However, they often fall short in integrating advanced, task-specific capabilities for compositional reasoning, which hinders their progress toward truly competent general vision models. To address this, we present a unified visual reasoning mechanism that enables LMMs to solve complicated compositional problems by leveraging their intrinsic capabilities (e.g. grounding and visual understanding capabilities). Different from the previous shortcut learning mechanism, our approach introduces a human-like understanding-thinking-answering process, allowing the model to complete all steps in a single pass forwarding without the need for multiple inferences or external tools. This design bridges the gap between foundational visual capabilities and general question answering, encouraging LMMs to generate faithful and traceable responses for complex visual reasoning. Meanwhile, we curate 334K visual instruction samples covering both general scenes and text-rich scenes and involving multiple foundational visual capabilities. Our trained model, Griffon-R, has the ability of end-to-end automatic understanding, self-thinking, and reasoning answers. Comprehensive experiments show that Griffon-R not only achieves advancing performance on complex visual reasoning benchmarks including VSR and CLEVR, but also enhances multimodal capabilities across various benchmarks like MMBench and ScienceQA. Data, models, and codes will be release at https://github.com/jefferyZhan/Griffon/tree/master/Griffon-R soon.
>
---
#### [new 083] ConsiStyle: Style Diversity in Training-Free Consistent T2I Generation
- **分类: cs.CV**

- **简介: 该论文提出无训练方法ConsiStyle，解决文本到图像生成中不同风格提示下保持主体一致性的问题。通过分离锚图像的查询/键与非锚值，扩展注意力机制并统计对齐，实现风格与主体解耦，生成跨风格一致且符合文本的图像。（98字）**

- **链接: [http://arxiv.org/pdf/2505.20626v1](http://arxiv.org/pdf/2505.20626v1)**

> **作者:** Yohai Mazuz; Janna Bruner; Lior Wolf
>
> **摘要:** In text-to-image models, consistent character generation is the task of achieving text alignment while maintaining the subject's appearance across different prompts. However, since style and appearance are often entangled, the existing methods struggle to preserve consistent subject characteristics while adhering to varying style prompts. Current approaches for consistent text-to-image generation typically rely on large-scale fine-tuning on curated image sets or per-subject optimization, which either fail to generalize across prompts or do not align well with textual descriptions. Meanwhile, training-free methods often fail to maintain subject consistency across different styles. In this work, we introduce a training-free method that achieves both style alignment and subject consistency. The attention matrices are manipulated such that Queries and Keys are obtained from the anchor image(s) that are used to define the subject, while the Values are imported from a parallel copy that is not subject-anchored. Additionally, cross-image components are added to the self-attention mechanism by expanding the Key and Value matrices. To do without shifting from the target style, we align the statistics of the Value matrices. As is demonstrated in a comprehensive battery of qualitative and quantitative experiments, our method effectively decouples style from subject appearance and enables faithful generation of text-aligned images with consistent characters across diverse styles.
>
---
#### [new 084] In Context Learning with Vision Transformers: Case Study
- **分类: cs.CV; cs.AI; cs.LG; I.2.6; I.2.10; I.4.8**

- **简介: 该论文研究视觉Transformer的上下文学习能力，旨在扩展其在图像空间中学习复杂函数（如CNN）的任务。通过实验分析模型在图像任务中的少/零样本学习表现，评估其泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.20872v1](http://arxiv.org/pdf/2505.20872v1)**

> **作者:** Antony Zhao; Alex Proshkin; Fergal Hennessy; Francesco Crivelli
>
> **备注:** 12 pages, 16 figures. UC Berkeley research project
>
> **摘要:** Large transformer models have been shown to be capable of performing in-context learning. By using examples in a prompt as well as a query, they are capable of performing tasks such as few-shot, one-shot, or zero-shot learning to output the corresponding answer to this query. One area of interest to us is that these transformer models have been shown to be capable of learning the general class of certain functions, such as linear functions and small 2-layer neural networks, on random data (Garg et al, 2023). We aim to extend this to the image space to analyze their capability to in-context learn more complex functions on the image space, such as convolutional neural networks and other methods.
>
---
#### [new 085] Rendering-Aware Reinforcement Learning for Vector Graphics Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于矢量图形生成任务，旨在解决现有视觉语言模型（VLMs）生成SVG时准确性差、效率低的问题。提出RLRF方法，通过强化学习利用渲染后的SVG与原图的对比反馈，优化生成过程，提升输出的视觉保真度和结构合理性。**

- **链接: [http://arxiv.org/pdf/2505.20793v1](http://arxiv.org/pdf/2505.20793v1)**

> **作者:** Juan A. Rodriguez; Haotian Zhang; Abhay Puri; Aarash Feizi; Rishav Pramanik; Pascal Wichmann; Arnab Mondal; Mohammad Reza Samsami; Rabiul Awal; Perouz Taslakian; Spandana Gella; Sai Rajeswar; David Vazquez; Christopher Pal; Marco Pedersoli
>
> **摘要:** Scalable Vector Graphics (SVG) offer a powerful format for representing visual designs as interpretable code. Recent advances in vision-language models (VLMs) have enabled high-quality SVG generation by framing the problem as a code generation task and leveraging large-scale pretraining. VLMs are particularly suitable for this task as they capture both global semantics and fine-grained visual patterns, while transferring knowledge across vision, natural language, and code domains. However, existing VLM approaches often struggle to produce faithful and efficient SVGs because they never observe the rendered images during training. Although differentiable rendering for autoregressive SVG code generation remains unavailable, rendered outputs can still be compared to original inputs, enabling evaluative feedback suitable for reinforcement learning (RL). We introduce RLRF(Reinforcement Learning from Rendering Feedback), an RL method that enhances SVG generation in autoregressive VLMs by leveraging feedback from rendered SVG outputs. Given an input image, the model generates SVG roll-outs that are rendered and compared to the original image to compute a reward. This visual fidelity feedback guides the model toward producing more accurate, efficient, and semantically coherent SVGs. RLRF significantly outperforms supervised fine-tuning, addressing common failure modes and enabling precise, high-quality SVG generation with strong structural understanding and generalization.
>
---
#### [new 086] What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于图像编辑评估任务，针对现有指标与人类判断不一致的问题，提出DICE模型，通过差异检测器和连贯性评估器，结合多模态大模型与混合训练策略，有效识别并评估指令引导的图像修改效果，实验显示与人类判断强相关。**

- **链接: [http://arxiv.org/pdf/2505.20405v1](http://arxiv.org/pdf/2505.20405v1)**

> **作者:** Lorenzo Baraldi; Davide Bucciarelli; Federico Betti; Marcella Cornia; Lorenzo Baraldi; Nicu Sebe; Rita Cucchiara
>
> **摘要:** Instruction-based image editing models offer increased personalization opportunities in generative tasks. However, properly evaluating their results is challenging, and most of the existing metrics lag in terms of alignment with human judgment and explainability. To tackle these issues, we introduce DICE (DIfference Coherence Estimator), a model designed to detect localized differences between the original and the edited image and to assess their relevance to the given modification request. DICE consists of two key components: a difference detector and a coherence estimator, both built on an autoregressive Multimodal Large Language Model (MLLM) and trained using a strategy that leverages self-supervision, distillation from inpainting networks, and full supervision. Through extensive experiments, we evaluate each stage of our pipeline, comparing different MLLMs within the proposed framework. We demonstrate that DICE effectively identifies coherent edits, effectively evaluating images generated by different editing models with a strong correlation with human judgment. We publicly release our source code, models, and data.
>
---
#### [new 087] Stereo Radargrammetry Using Deep Learning from Airborne SAR Images
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出基于深度学习的机载SAR立体雷达测高方法，解决SAR图像缺乏公开数据集及几何畸变影响的问题。工作包括构建SAR数据集、改进图像配准算法（通过像素插值和图像分块处理），实现更广范围和更高精度的高程测量。**

- **链接: [http://arxiv.org/pdf/2505.20876v1](http://arxiv.org/pdf/2505.20876v1)**

> **作者:** Tatsuya Sasayama; Shintaro Ito; Koichi Ito; Takafumi Aoki
>
> **备注:** 5 pages, 5 figures, conference IGARSS2025
>
> **摘要:** In this paper, we propose a stereo radargrammetry method using deep learning from airborne Synthetic Aperture Radar (SAR) images.Deep learning-based methods are considered to suffer less from geometric image modulation, while there is no public SAR image dataset used to train such methods.We create a SAR image dataset and perform fine-tuning of a deep learning-based image correspondence method.The proposed method suppresses the degradation of image quality by pixel interpolation without ground projection of the SAR image and divides the SAR image into patches for processing, which makes it possible to apply deep learning.Through a set of experiments, we demonstrate that the proposed method exhibits a wider range and more accurate elevation measurements compared to conventional methods.
>
---
#### [new 088] Unified Alignment Protocol: Making Sense of the Unlabeled Data in New Domains
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对半监督联邦学习（SSFL）在领域偏移下的泛化问题，提出Unified Alignment Protocol（UAP）。传统SSFL假设数据分布一致，但实际中领域偏移频繁。UAP通过两阶段训练对齐服务器与客户端特征分布，提升跨领域性能，在基准测试中达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.21010v1](http://arxiv.org/pdf/2505.21010v1)**

> **作者:** Sabbir Ahmed; Mamshad Nayeem Rizve; Abdullah Al Arafat; Jacqueline Liu; Rahim Hossain; Mohaiminul Al Nahian; Adnan Siraj Rakin
>
> **摘要:** Semi-Supervised Federated Learning (SSFL) is gaining popularity over conventional Federated Learning in many real-world applications. Due to the practical limitation of limited labeled data on the client side, SSFL considers that participating clients train with unlabeled data, and only the central server has the necessary resources to access limited labeled data, making it an ideal fit for real-world applications (e.g., healthcare). However, traditional SSFL assumes that the data distributions in the training phase and testing phase are the same. In practice, however, domain shifts frequently occur, making it essential for SSFL to incorporate generalization capabilities and enhance their practicality. The core challenge is improving model generalization to new, unseen domains while the client participate in SSFL. However, the decentralized setup of SSFL and unsupervised client training necessitates innovation to achieve improved generalization across domains. To achieve this, we propose a novel framework called the Unified Alignment Protocol (UAP), which consists of an alternating two-stage training process. The first stage involves training the server model to learn and align the features with a parametric distribution, which is subsequently communicated to clients without additional communication overhead. The second stage proposes a novel training algorithm that utilizes the server feature distribution to align client features accordingly. Our extensive experiments on standard domain generalization benchmark datasets across multiple model architectures reveal that proposed UAP successfully achieves SOTA generalization performance in SSFL setting.
>
---
#### [new 089] Causality-Driven Infrared and Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于红外与可见光图像融合任务，旨在解决现有方法因数据集场景偏差导致模型学习伪相关关系的问题。通过构建因果图分离变量间因果关系，并提出Back-door调整的特征融合模块（BAFFM），消除混杂因素以学习真实因果效应，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.20830v1](http://arxiv.org/pdf/2505.20830v1)**

> **作者:** Linli Ma; Suzhen Lin; Jianchao Zeng; Zanxia Jin; Yanbo Wang; Fengyuan Li; Yubing Luo
>
> **摘要:** Image fusion aims to combine complementary information from multiple source images to generate more comprehensive scene representations. Existing methods primarily rely on the stacking and design of network architectures to enhance the fusion performance, often ignoring the impact of dataset scene bias on model training. This oversight leads the model to learn spurious correlations between specific scenes and fusion weights under conventional likelihood estimation framework, thereby limiting fusion performance. To solve the above problems, this paper first re-examines the image fusion task from the causality perspective, and disentangles the model from the impact of bias by constructing a tailored causal graph to clarify the causalities among the variables in image fusion task. Then, the Back-door Adjustment based Feature Fusion Module (BAFFM) is proposed to eliminate confounder interference and enable the model to learn the true causal effect. Finally, Extensive experiments on three standard datasets prove that the proposed method significantly surpasses state-of-the-art methods in infrared and visible image fusion.
>
---
#### [new 090] FeatInv: Spatially resolved mapping from feature space to input space using conditional diffusion models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出FeatInv方法，通过条件扩散模型将特征空间映射到输入空间，解决现有方法对深度模型内部表示解释不足的问题。利用预训练扩散模型基于空间特征图进行概率重建，在CNN和ViT等模型上验证了高保真映射能力，并应用于概念可视化和特征复合性分析，提升计算机视觉模型的可解释性。**

- **链接: [http://arxiv.org/pdf/2505.21032v1](http://arxiv.org/pdf/2505.21032v1)**

> **作者:** Nils Neukirch; Johanna Vielhaben; Nils Strodthoff
>
> **备注:** 15 pages, 10 figures, code is available at https://github.com/AI4HealthUOL/FeatInv
>
> **摘要:** Internal representations are crucial for understanding deep neural networks, such as their properties and reasoning patterns, but remain difficult to interpret. While mapping from feature space to input space aids in interpreting the former, existing approaches often rely on crude approximations. We propose using a conditional diffusion model - a pretrained high-fidelity diffusion model conditioned on spatially resolved feature maps - to learn such a mapping in a probabilistic manner. We demonstrate the feasibility of this approach across various pretrained image classifiers from CNNs to ViTs, showing excellent reconstruction capabilities. Through qualitative comparisons and robustness analysis, we validate our method and showcase possible applications, such as the visualization of concept steering in input space or investigations of the composite nature of the feature space. This approach has broad potential for improving feature space understanding in computer vision models.
>
---
#### [new 091] Occlusion Boundary and Depth: Mutual Enhancement via Multi-Task Learning
- **分类: cs.CV**

- **简介: 该论文提出MoDOT网络，通过多任务学习联合估计遮挡边界(OB)与单目深度，利用CASM模块和OBDCL损失函数实现两者相互增强，解决复杂场景下遮挡边界与深度预测的相互依赖问题，在多个数据集上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2505.21231v1](http://arxiv.org/pdf/2505.21231v1)**

> **作者:** Lintao Xu; Yinghao Wang; Chaohui Wang
>
> **备注:** 7 pages, 4 tables, 4 figures
>
> **摘要:** Occlusion Boundary Estimation (OBE) identifies boundaries arising from both inter-object occlusions and self-occlusion within individual objects, distinguishing intrinsic object edges from occlusion-induced contours to improve scene understanding and 3D reconstruction capacity. This is closely related to Monocular Depth Estimation (MDE), which infers depth from a single image, as occlusion boundaries provide critical geometric cues for resolving depth ambiguities, while depth priors can conversely refine occlusion reasoning in complex scenes. In this paper, we propose a novel network, MoDOT, that first jointly estimates depth and OBs. We propose CASM, a cross-attention multi-scale strip convolution module, leverages mid-level OB features to significantly enhance depth prediction. Additionally, we introduce an occlusion-aware loss function, OBDCL, which encourages sharper and more accurate depth boundaries. Extensive experiments on both real and synthetic datasets demonstrate the mutual benefits of jointly estimating depth and OB, and highlight the effectiveness of our model design. Our method achieves the state-of-the-art (SOTA) on both our proposed synthetic datasets and one popular real dataset, NYUD-v2, significantly outperforming multi-task baselines. Besides, without domain adaptation, results on real-world depth transfer are comparable to the competitors, while preserving sharp occlusion boundaries for geometric fidelity. We will release our code, pre-trained models, and datasets to support future research in this direction.
>
---
#### [new 092] TrustSkin: A Fairness Pipeline for Trustworthy Facial Affect Analysis Across Skin Tone
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于面部情感分析（FAA）公平性评估任务，旨在解决肤色差异导致的模型性能不平等问题。研究对比了两种肤色分类方法（ITA与L*-H*），发现深色皮肤样本严重不足且性能差异显著（F1差0.08，TPR差0.11），提出基于L*-H*的公平感知pipeline，集成肤色估计、模型解释和公平评估模块，以改进跨肤色群体的FAA可靠性。**

- **链接: [http://arxiv.org/pdf/2505.20637v1](http://arxiv.org/pdf/2505.20637v1)**

> **作者:** Ana M. Cabanas; Alma Pedro; Domingo Mery
>
> **备注:** 10 pages
>
> **摘要:** Understanding how facial affect analysis (FAA) systems perform across different demographic groups requires reliable measurement of sensitive attributes such as ancestry, often approximated by skin tone, which itself is highly influenced by lighting conditions. This study compares two objective skin tone classification methods: the widely used Individual Typology Angle (ITA) and a perceptually grounded alternative based on Lightness ($L^*$) and Hue ($H^*$). Using AffectNet and a MobileNet-based model, we assess fairness across skin tone groups defined by each method. Results reveal a severe underrepresentation of dark skin tones ($\sim 2 \%$), alongside fairness disparities in F1-score (up to 0.08) and TPR (up to 0.11) across groups. While ITA shows limitations due to its sensitivity to lighting, the $H^*$-$L^*$ method yields more consistent subgrouping and enables clearer diagnostics through metrics such as Equal Opportunity. Grad-CAM analysis further highlights differences in model attention patterns by skin tone, suggesting variation in feature encoding. To support future mitigation efforts, we also propose a modular fairness-aware pipeline that integrates perceptual skin tone estimation, model interpretability, and fairness evaluation. These findings emphasize the relevance of skin tone measurement choices in fairness assessment and suggest that ITA-based evaluations may overlook disparities affecting darker-skinned individuals.
>
---
#### [new 093] Exploring Timeline Control for Facial Motion Generation
- **分类: cs.CV**

- **简介: 该论文提出基于时间线控制的面部动作生成方法，解决现有方法时序控制不足的问题。通过帧级标注自然面部动作（结合Toeplitz聚类降本），提出扩散模型生成自然且精准对齐时间线的面部运动，并利用ChatGPT将文本转为时间线，实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.20861v1](http://arxiv.org/pdf/2505.20861v1)**

> **作者:** Yifeng Ma; Jinwei Qi; Chaonan Ji; Peng Zhang; Bang Zhang; Zhidong Deng; Liefeng Bo
>
> **备注:** Accepted by CVPR 2025, Project Page: https://humanaigc.github.io/facial-motion-timeline-control/
>
> **摘要:** This paper introduces a new control signal for facial motion generation: timeline control. Compared to audio and text signals, timelines provide more fine-grained control, such as generating specific facial motions with precise timing. Users can specify a multi-track timeline of facial actions arranged in temporal intervals, allowing precise control over the timing of each action. To model the timeline control capability, We first annotate the time intervals of facial actions in natural facial motion sequences at a frame-level granularity. This process is facilitated by Toeplitz Inverse Covariance-based Clustering to minimize human labor. Based on the annotations, we propose a diffusion-based generation model capable of generating facial motions that are natural and accurately aligned with input timelines. Our method supports text-guided motion generation by using ChatGPT to convert text into timelines. Experimental results show that our method can annotate facial action intervals with satisfactory accuracy, and produces natural facial motions accurately aligned with timelines.
>
---
#### [new 094] DetailFlow: 1D Coarse-to-Fine Autoregressive Image Generation via Next-Detail Prediction
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决传统自回归（AR）模型生成效率低、tokens需求高的问题。提出DetailFlow方法，通过1D分辨率感知token序列和"由粗到细"的细节预测策略，结合并行推理与自纠错机制，在ImageNet 256x256上以128 tokens实现2.96 gFID，速度比VAR快2倍，显著提升生成效率与质量。**

- **链接: [http://arxiv.org/pdf/2505.21473v1](http://arxiv.org/pdf/2505.21473v1)**

> **作者:** Yiheng Liu; Liao Qu; Huichao Zhang; Xu Wang; Yi Jiang; Yiming Gao; Hu Ye; Xian Li; Shuai Wang; Daniel K. Du; Shu Cheng; Zehuan Yuan; Xinglong Wu
>
> **摘要:** This paper presents DetailFlow, a coarse-to-fine 1D autoregressive (AR) image generation method that models images through a novel next-detail prediction strategy. By learning a resolution-aware token sequence supervised with progressively degraded images, DetailFlow enables the generation process to start from the global structure and incrementally refine details. This coarse-to-fine 1D token sequence aligns well with the autoregressive inference mechanism, providing a more natural and efficient way for the AR model to generate complex visual content. Our compact 1D AR model achieves high-quality image synthesis with significantly fewer tokens than previous approaches, i.e. VAR/VQGAN. We further propose a parallel inference mechanism with self-correction that accelerates generation speed by approximately 8x while reducing accumulation sampling error inherent in teacher-forcing supervision. On the ImageNet 256x256 benchmark, our method achieves 2.96 gFID with 128 tokens, outperforming VAR (3.3 FID) and FlexVAR (3.05 FID), which both require 680 tokens in their AR models. Moreover, due to the significantly reduced token count and parallel inference mechanism, our method runs nearly 2x faster inference speed compared to VAR and FlexVAR. Extensive experimental results demonstrate DetailFlow's superior generation quality and efficiency compared to existing state-of-the-art methods.
>
---
#### [new 095] HTMNet: A Hybrid Network with Transformer-Mamba Bottleneck Multimodal Fusion for Transparent and Reflective Objects Depth Completion
- **分类: cs.CV**

- **简介: 该论文属于透明/反射物体深度补全任务，解决其导致的深度传感器数据不完整问题。提出HTMNet，结合Transformer、CNN与Mamba架构，设计基于自注意力及状态模型的多模态融合模块（首次应用Mamba于此领域），并采用多尺度特征融合解码器，实验达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.20904v1](http://arxiv.org/pdf/2505.20904v1)**

> **作者:** Guanghu Xie; Yonglong Zhang; Zhiduo Jiang; Yang Liu; Zongwu Xie; Baoshi Cao; Hong Liu
>
> **摘要:** Transparent and reflective objects pose significant challenges for depth sensors, resulting in incomplete depth information that adversely affects downstream robotic perception and manipulation tasks. To address this issue, we propose HTMNet, a novel hybrid model integrating Transformer, CNN, and Mamba architectures. The encoder is constructed based on a dual-branch Transformer-CNN framework, while the multi-scale fusion module leverages a Transformer-Mamba architecture, which also serves as the foundation for the decoder design. We introduce a novel multimodal fusion module grounded in self-attention mechanisms and state space models, marking the first application of the Mamba architecture in the field of transparent object depth completion and revealing its promising potential. Additionally, we design an innovative multi-scale fusion module for the decoder that combines channel attention, spatial attention, and multi-scale feature extraction techniques to effectively integrate multi-scale features through a down-fusion strategy. Extensive evaluations on multiple public datasets demonstrate that our model achieves state-of-the-art(SOTA) performance, validating the effectiveness of our approach.
>
---
#### [new 096] Fully Spiking Neural Networks for Unified Frame-Event Object Tracking
- **分类: cs.CV; cs.NE**

- **简介: 该论文属于目标跟踪任务，旨在解决现有帧-事件融合方法计算效率低、难以有效利用事件流稀疏异步信息的问题。提出首个全脉冲神经网络框架SpikeFET，通过融合卷积局部特征与Transformer全局建模，并设计随机拼块模块（RPM）消除位置偏差，及时空正则化策略（STR）提升特征一致性，实现高精度低功耗跟踪。**

- **链接: [http://arxiv.org/pdf/2505.20834v1](http://arxiv.org/pdf/2505.20834v1)**

> **作者:** Jingjun Yang; Liangwei Fan; Jinpu Zhang; Xiangkai Lian; Hui Shen; Dewen Hu
>
> **备注:** 13 pages,6 figures,4 tables
>
> **摘要:** The integration of image and event streams offers a promising approach for achieving robust visual object tracking in complex environments. However, current fusion methods achieve high performance at the cost of significant computational overhead and struggle to efficiently extract the sparse, asynchronous information from event streams, failing to leverage the energy-efficient advantages of event-driven spiking paradigms. To address this challenge, we propose the first fully Spiking Frame-Event Tracking framework called SpikeFET. This network achieves synergistic integration of convolutional local feature extraction and Transformer-based global modeling within the spiking paradigm, effectively fusing frame and event data. To overcome the degradation of translation invariance caused by convolutional padding, we introduce a Random Patchwork Module (RPM) that eliminates positional bias through randomized spatial reorganization and learnable type encoding while preserving residual structures. Furthermore, we propose a Spatial-Temporal Regularization (STR) strategy that overcomes similarity metric degradation from asymmetric features by enforcing spatio-temporal consistency among temporal template features in latent space. Extensive experiments across multiple benchmarks demonstrate that the proposed framework achieves superior tracking accuracy over existing methods while significantly reducing power consumption, attaining an optimal balance between performance and efficiency. The code will be released.
>
---
#### [new 097] Boosting Adversarial Transferability via High-Frequency Augmentation and Hierarchical-Gradient Fusion
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于对抗攻击任务，旨在提升黑盒模型的对抗迁移性。针对现有方法依赖空间域的局限，提出FSA框架，融合高频增强（傅里叶变换放大高频成分）与层次梯度融合（多尺度分解优化扰动），实验显示在八种黑盒防御模型上攻击成功率平均提升23.6%。**

- **链接: [http://arxiv.org/pdf/2505.21181v1](http://arxiv.org/pdf/2505.21181v1)**

> **作者:** Yayin Zheng; Chen Wan; Zihong Guo; Hailing Kuang; Xiaohai Lu
>
> **摘要:** Adversarial attacks have become a significant challenge in the security of machine learning models, particularly in the context of black-box defense strategies. Existing methods for enhancing adversarial transferability primarily focus on the spatial domain. This paper presents Frequency-Space Attack (FSA), a new adversarial attack framework that effectively integrates frequency-domain and spatial-domain transformations. FSA combines two key techniques: (1) High-Frequency Augmentation, which applies Fourier transform with frequency-selective amplification to diversify inputs and emphasize the critical role of high-frequency components in adversarial attacks, and (2) Hierarchical-Gradient Fusion, which merges multi-scale gradient decomposition and fusion to capture both global structures and fine-grained details, resulting in smoother perturbations. Our experiment demonstrates that FSA consistently outperforms state-of-the-art methods across various black-box models. Notably, our proposed FSA achieves an average attack success rate increase of 23.6% compared with BSR (CVPR 2024) on eight black-box defense models.
>
---
#### [new 098] RoGA: Towards Generalizable Deepfake Detection through Robust Gradient Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造检测的领域泛化任务，旨在解决现有方法因添加正则化模块导致模型性能下降的问题。提出RoGA方法，通过扰动模型参数对齐跨领域梯度更新，提升模型对领域变化的鲁棒性，无需额外正则化，在多个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20653v1](http://arxiv.org/pdf/2505.20653v1)**

> **作者:** Lingyu Qiu; Ke Jiang; Xiaoyang Tan
>
> **备注:** Accepted to ICME2025
>
> **摘要:** Recent advancements in domain generalization for deepfake detection have attracted significant attention, with previous methods often incorporating additional modules to prevent overfitting to domain-specific patterns. However, such regularization can hinder the optimization of the empirical risk minimization (ERM) objective, ultimately degrading model performance. In this paper, we propose a novel learning objective that aligns generalization gradient updates with ERM gradient updates. The key innovation is the application of perturbations to model parameters, aligning the ascending points across domains, which specifically enhances the robustness of deepfake detection models to domain shifts. This approach effectively preserves domain-invariant features while managing domain-specific characteristics, without introducing additional regularization. Experimental results on multiple challenging deepfake detection datasets demonstrate that our gradient alignment strategy outperforms state-of-the-art domain generalization techniques, confirming the efficacy of our method. The code is available at https://github.com/Lynn0925/RoGA.
>
---
#### [new 099] ID-Align: RoPE-Conscious Position Remapping for Dynamic High-Resolution Adaptation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉语言模型（VLM）中高分辨率图像与文本交互不足的问题，提出ID-Align方法。通过重新映射位置ID，使高分辨率图像token继承缩略图token的位置信息并限制索引扩张，缓解RoPE位置嵌入的衰减效应，提升跨模态交互，在MMBench等任务中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.21465v1](http://arxiv.org/pdf/2505.21465v1)**

> **作者:** Bozhou Li; Wentao Zhang
>
> **摘要:** Currently, a prevalent approach for enhancing Vision-Language Models (VLMs) performance is to encode both the high-resolution version and the thumbnail of an image simultaneously. While effective, this method generates a large number of image tokens. When combined with the widely used Rotary Position Embedding (RoPE), its long-term decay property hinders the interaction between high-resolution tokens and thumbnail tokens, as well as between text and image. To address these issues, we propose ID-Align, which alleviates these problems by reordering position IDs. In this method, high-resolution tokens inherit IDs from their corresponding thumbnail token while constraining the overexpansion of positional indices. Our experiments conducted within the LLaVA-Next framework demonstrate that ID-Align achieves significant improvements, including a 6.09% enhancement on MMBench's relation reasoning tasks and notable gains across multiple benchmarks. Our code is available at the following link: https://github.com/zooblastlbz/ID-Align.
>
---
#### [new 100] OmniIndoor3D: Comprehensive Indoor 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OmniIndoor3D框架，解决室内3D重建中几何精度与外观优化冲突问题。通过RGB-D图像初始化高斯表示，引入轻量级MLP分离几何与外观优化，并采用全景先验引导的高斯密集化策略，实现精准几何、逼真外观及全景联合重建。**

- **链接: [http://arxiv.org/pdf/2505.20610v1](http://arxiv.org/pdf/2505.20610v1)**

> **作者:** Xiaobao Wei; Xiaoan Zhang; Hao Wang; Qingpo Wuwu; Ming Lu; Wenzhao Zheng; Shanghang Zhang
>
> **摘要:** We propose a novel framework for comprehensive indoor 3D reconstruction using Gaussian representations, called OmniIndoor3D. This framework enables accurate appearance, geometry, and panoptic reconstruction of diverse indoor scenes captured by a consumer-level RGB-D camera. Since 3DGS is primarily optimized for photorealistic rendering, it lacks the precise geometry critical for high-quality panoptic reconstruction. Therefore, OmniIndoor3D first combines multiple RGB-D images to create a coarse 3D reconstruction, which is then used to initialize the 3D Gaussians and guide the 3DGS training. To decouple the optimization conflict between appearance and geometry, we introduce a lightweight MLP that adjusts the geometric properties of 3D Gaussians. The introduced lightweight MLP serves as a low-pass filter for geometry reconstruction and significantly reduces noise in indoor scenes. To improve the distribution of Gaussian primitives, we propose a densification strategy guided by panoptic priors to encourage smoothness on planar surfaces. Through the joint optimization of appearance, geometry, and panoptic reconstruction, OmniIndoor3D provides comprehensive 3D indoor scene understanding, which facilitates accurate and robust robotic navigation. We perform thorough evaluations across multiple datasets, and OmniIndoor3D achieves state-of-the-art results in appearance, geometry, and panoptic reconstruction. We believe our work bridges a critical gap in indoor 3D reconstruction. The code will be released at: https://ucwxb.github.io/OmniIndoor3D/
>
---
#### [new 101] AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding
- **分类: cs.CV**

- **简介: 该论文针对视听大语言模型（AV-LLMs）中由多模态交互引发的幻觉问题，提出AVCD框架：通过动态分析注意力分布识别次要模态、生成扰动输出，并结合熵引导解码抑制幻觉，提升模型准确率（在AVHBench上提升6-11%）。属于多模态模型解码优化任务。**

- **链接: [http://arxiv.org/pdf/2505.20862v1](http://arxiv.org/pdf/2505.20862v1)**

> **作者:** Chaeyoung Jung; Youngjoon Jang; Joon Son Chung
>
> **摘要:** Hallucination remains a major challenge in multimodal large language models (MLLMs). To address this, various contrastive decoding (CD) methods have been proposed that contrasts original logits with hallucinated logits generated from perturbed inputs. While CD has shown promise in vision-language models (VLMs), it is not well-suited for AV-LLMs, where hallucinations often emerge from both unimodal and cross-modal combinations involving audio, video, and language. These intricate interactions call for a more adaptive and modality-aware decoding strategy. In this paper, we propose Audio-Visual Contrastive Decoding (AVCD)-a novel, training-free decoding framework designed to model trimodal interactions and suppress modality-induced hallucinations in AV-LLMs. Unlike previous CD methods in VLMs that corrupt a fixed modality, AVCD leverages attention distributions to dynamically identify less dominant modalities and applies attentive masking to generate perturbed output logits. To support CD in a trimodal setting, we also reformulate the original CD framework to jointly handle audio, visual, and textual inputs. Finally, to improve efficiency, we introduce entropy-guided adaptive decoding, which selectively skips unnecessary decoding steps based on the model's confidence in its predictions. Extensive experiments demonstrate that AVCD consistently outperforms existing decoding methods. Especially, on the AVHBench dataset, it improves accuracy by 6% for VideoLLaMA2 and 11% for video-SALMONN, demonstrating strong robustness and generalizability.
>
---
#### [new 102] Assessing the Use of Face Swapping Methods as Face Anonymizers in Videos
- **分类: cs.CV**

- **简介: 该论文评估换脸技术在视频匿名化中的潜力，旨在平衡隐私保护与数据质量。通过测试时间一致性、匿名强度和视觉保真度，验证换脸技术能有效隐藏身份并保持视频连贯性，为隐私保护应用提供技术基础。**

- **链接: [http://arxiv.org/pdf/2505.20985v1](http://arxiv.org/pdf/2505.20985v1)**

> **作者:** Mustafa İzzet Muştu; Hazım Kemal Ekenel
>
> **备注:** Accepted to the 2025 25th International Conference on Digital Signal Processing (DSP 2025)
>
> **摘要:** The increasing demand for large-scale visual data, coupled with strict privacy regulations, has driven research into anonymization methods that hide personal identities without seriously degrading data quality. In this paper, we explore the potential of face swapping methods to preserve privacy in video data. Through extensive evaluations focusing on temporal consistency, anonymity strength, and visual fidelity, we find that face swapping techniques can produce consistent facial transitions and effectively hide identities. These results underscore the suitability of face swapping for privacy-preserving video applications and lay the groundwork for future advancements in anonymization focused face-swapping models.
>
---
#### [new 103] DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data
- **分类: cs.CV**

- **简介: 论文提出DIPO框架，通过双状态图像生成可控3D关节物体，解决单图难以捕捉运动关系的问题。方法包含双图扩散模型、CoT推理模块、数据增强pipeline LEGO-Art及PM-X数据集，提升复杂物体生成与泛化。**

- **链接: [http://arxiv.org/pdf/2505.20460v1](http://arxiv.org/pdf/2505.20460v1)**

> **作者:** Ruqi Wu; Xinjie Wang; Liu Liu; Chunle Guo; Jiaxiong Qiu; Chongyi Li; Lichao Huang; Zhizhong Su; Ming-Ming Cheng
>
> **摘要:** We present DIPO, a novel framework for the controllable generation of articulated 3D objects from a pair of images: one depicting the object in a resting state and the other in an articulated state. Compared to the single-image approach, our dual-image input imposes only a modest overhead for data collection, but at the same time provides important motion information, which is a reliable guide for predicting kinematic relationships between parts. Specifically, we propose a dual-image diffusion model that captures relationships between the image pair to generate part layouts and joint parameters. In addition, we introduce a Chain-of-Thought (CoT) based graph reasoner that explicitly infers part connectivity relationships. To further improve robustness and generalization on complex articulated objects, we develop a fully automated dataset expansion pipeline, name LEGO-Art, that enriches the diversity and complexity of PartNet-Mobility dataset. We propose PM-X, a large-scale dataset of complex articulated 3D objects, accompanied by rendered images, URDF annotations, and textual descriptions. Extensive experiments demonstrate that DIPO significantly outperforms existing baselines in both the resting state and the articulated state, while the proposed PM-X dataset further enhances generalization to diverse and structurally complex articulated objects. Our code and dataset will be released to the community upon publication.
>
---
#### [new 104] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PARTONOMY基准与PLUM模型，解决大模型在部件级视觉理解（如细粒度分割、部件关系推理）上的不足。通过构建含862部件标签的数据集，揭示现有模型性能缺陷，并改进架构（span标记替代SEG token、预测反馈机制），提升分割与推理能力。**

- **链接: [http://arxiv.org/pdf/2505.20759v1](http://arxiv.org/pdf/2505.20759v1)**

> **作者:** Ansel Blume; Jeonghwan Kim; Hyeonjeong Ha; Elen Chatikyan; Xiaomeng Jin; Khanh Duy Nguyen; Nanyun Peng; Kai-Wei Chang; Derek Hoiem; Heng Ji
>
> **备注:** 18 pages
>
> **摘要:** Real-world objects are composed of distinctive, object-specific parts. Identifying these parts is key to performing fine-grained, compositional reasoning-yet, large multimodal models (LMMs) struggle to perform this seemingly straightforward task. In this work, we introduce PARTONOMY, an LMM benchmark designed for pixel-level part grounding. We construct PARTONOMY from existing part datasets and our own rigorously annotated set of images, encompassing 862 part labels and 534 object labels for evaluation. Unlike existing datasets that simply ask models to identify generic parts, PARTONOMY uses specialized concepts (e.g., agricultural airplane), and challenges models to compare objects' parts, consider part-whole relationships, and justify textual predictions with visual segmentations. Our experiments demonstrate significant limitations in state-of-the-art LMMs (e.g., LISA-13B achieves only 5.9% gIoU), highlighting a critical gap in their part grounding abilities. We note that existing segmentation-enabled LMMs (segmenting LMMs) have two key architectural shortcomings: they use special [SEG] tokens not seen during pretraining which induce distribution shift, and they discard predicted segmentations instead of using past predictions to guide future ones. To address these deficiencies, we train several part-centric LMMs and propose PLUM, a novel segmenting LMM that uses span tagging instead of segmentation tokens and that conditions on prior predictions in a feedback loop. We find that pretrained PLUM outperforms existing segmenting LMMs on reasoning segmentation, VQA, and visual hallucination benchmarks. In addition, PLUM finetuned on our proposed Explanatory Part Segmentation task is competitive with segmenting LMMs trained on significantly more segmentation data. Our work opens up new avenues towards enabling fine-grained, grounded visual understanding in LMMs.
>
---
#### [new 105] 3D-UIR: 3D Gaussian for Underwater 3D Scene Reconstruction via Physics-Based Appearance-Medium Decouplin
- **分类: cs.CV**

- **简介: 该论文属于水下3D场景重建任务，针对水体散射/吸收导致的介质不均匀性干扰问题，提出基于物理的外观-介质分离框架。通过3D高斯建模分离物体外观与介质效应，结合伪深度图优化策略，提升水下新型视图合成的渲染质量和几何准确性。**

- **链接: [http://arxiv.org/pdf/2505.21238v1](http://arxiv.org/pdf/2505.21238v1)**

> **作者:** Jieyu Yuan; Yujun Li; Yuanlin Zhang; Chunle Guo; Xiongxin Tang; Ruixing Wang; Chongyi Li
>
> **摘要:** Novel view synthesis for underwater scene reconstruction presents unique challenges due to complex light-media interactions. Optical scattering and absorption in water body bring inhomogeneous medium attenuation interference that disrupts conventional volume rendering assumptions of uniform propagation medium. While 3D Gaussian Splatting (3DGS) offers real-time rendering capabilities, it struggles with underwater inhomogeneous environments where scattering media introduce artifacts and inconsistent appearance. In this study, we propose a physics-based framework that disentangles object appearance from water medium effects through tailored Gaussian modeling. Our approach introduces appearance embeddings, which are explicit medium representations for backscatter and attenuation, enhancing scene consistency. In addition, we propose a distance-guided optimization strategy that leverages pseudo-depth maps as supervision with depth regularization and scale penalty terms to improve geometric fidelity. By integrating the proposed appearance and medium modeling components via an underwater imaging model, our approach achieves both high-quality novel view synthesis and physically accurate scene restoration. Experiments demonstrate our significant improvements in rendering quality and restoration accuracy over existing methods. The project page is available at \href{https://bilityniu.github.io/3D-UIR}{https://bilityniu.github.io/3D-UIR
>
---
#### [new 106] CROP: Contextual Region-Oriented Visual Token Pruning
- **分类: cs.CV**

- **简介: 该论文针对基于VLM的视觉问答（VQA）任务中冗余视觉token导致的高计算与内存消耗问题，提出CROP框架。通过先定位问题相关区域，再采用自适应压缩（PLC）和无训练层内剪枝（ILP）策略，有效减少无关token，提升效率，实验达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.21233v1](http://arxiv.org/pdf/2505.21233v1)**

> **作者:** Jiawei Guo; Feifei Zhai; Pu Jian; Qianrun Wei; Yu Zhou
>
> **摘要:** Current VLM-based VQA methods often process entire images, leading to excessive visual tokens that include redundant information irrelevant to the posed question. This abundance of unnecessary image details creates numerous visual tokens, drastically increasing memory and computational requirements in VLMs. To address this, we propose Contextual Region-Oriented Visual Token Pruning (CROP), a novel framework to compress visual tokens through a two-step process: Localization and Pruning. Specifically, CROP first employs an efficient model to identify the contextual region relevant to the input query. Subsequently, two distinct strategies are introduced for pruning: (1) Pre-LLM Compression (PLC), which adaptively compresses different image regions with varying ratios, and (2) Inner-LLM Pruning (ILP), a training-free method that prunes tokens within early LLM layers guided by the identified contextual region. Extensive experiments on a wide range of VQA tasks demonstrate that CROP significantly outperforms existing visual token pruning methods and achieves state-of-the-art performance. Our code and datasets will be made available.
>
---
#### [new 107] YOLO-FireAD: Efficient Fire Detection via Attention-Guided Inverted Residual Learning and Dual-Pooling Feature Preservation
- **分类: cs.CV**

- **简介: 该论文属于实时火灾检测任务，旨在解决动态环境下光照干扰、误检/漏检及效率-精度失衡问题。提出YOLO-FireAD模型，通过注意力引导倒残差模块(AIR)增强火特征并抑制噪声，以及双池化融合模块(DPDF)保留多尺度特征，减少参数（比YOLOv8n少51.8%）且mAP提升1.3-5.5%。**

- **链接: [http://arxiv.org/pdf/2505.20884v1](http://arxiv.org/pdf/2505.20884v1)**

> **作者:** Weichao Pan; Bohan Xu; Xu Wang; Chengze Lv; Shuoyang Wang; Zhenke Duan
>
> **摘要:** Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO-based models, this study propose You Only Look Once for Fire Detection with Attention-guided Inverted Residual and Dual-pooling Downscale Fusion (YOLO-FireAD) with two core innovations: (1) Attention-guided Inverted Residual Block (AIR) integrates hybrid channel-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; (2) Dual Pool Downscale Fusion Block (DPDF) preserves multi-scale fire patterns through learnable fusion of max-average pooling outputs, mitigating small-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters (1.45M, 51.8% lower than YOLOv8n) (4.6G, 43.2% lower than YOLOv8n), and mAP75 is higher than the mainstream real-time object detection models YOLOv8n, YOL-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3-5.5%.
>
---
#### [new 108] MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于对象中心学习任务，针对传统方法因固定插槽数量导致物体被分割的问题，提出MetaSlot：通过维护对象原型代码本去除冗余插槽、渐弱噪声加速聚合，实现自适应变物体数量的可插拔插槽注意力机制，提升模型性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2505.20772v1](http://arxiv.org/pdf/2505.20772v1)**

> **作者:** Hongjia Liu; Rongzhen Zhao; Haohan Chen; Joni Pajarinen
>
> **摘要:** Learning object-level, structured representations is widely regarded as a key to better generalization in vision and underpins the design of next-generation Pre-trained Vision Models (PVMs). Mainstream Object-Centric Learning (OCL) methods adopt Slot Attention or its variants to iteratively aggregate objects' super-pixels into a fixed set of query feature vectors, termed slots. However, their reliance on a static slot count leads to an object being represented as multiple parts when the number of objects varies. We introduce MetaSlot, a plug-and-play Slot Attention variant that adapts to variable object counts. MetaSlot (i) maintains a codebook that holds prototypes of objects in a dataset by vector-quantizing the resulting slot representations; (ii) removes duplicate slots from the traditionally aggregated slots by quantizing them with the codebook; and (iii) injects progressively weaker noise into the Slot Attention iterations to accelerate and stabilize the aggregation. MetaSlot is a general Slot Attention variant that can be seamlessly integrated into existing OCL architectures. Across multiple public datasets and tasks--including object discovery and recognition--models equipped with MetaSlot achieve significant performance gains and markedly interpretable slot representations, compared with existing Slot Attention variants.
>
---
#### [new 109] Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Roboflow100-VL基准，针对视觉语言模型（VLMs）在分布外场景（如医疗影像）目标检测泛化差的问题，构建100个多模态数据集并评估模型在不同训练模式下的表现，证明需通过少量样本与文本描述对齐提升性能。**

- **链接: [http://arxiv.org/pdf/2505.20612v1](http://arxiv.org/pdf/2505.20612v1)**

> **作者:** Peter Robicheaux; Matvei Popov; Anish Madan; Isaac Robinson; Joseph Nelson; Deva Ramanan; Neehar Peri
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Our code and dataset are available at https://github.com/roboflow/rf100-vl/ and https://universe.roboflow.com/rf100-vl/
>
---
#### [new 110] Is Hyperbolic Space All You Need for Medical Anomaly Detection?
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学异常检测任务，旨在解决欧氏空间无法有效捕捉特征层级关系导致检测性能不足的问题。提出将特征投影至双曲空间并按置信度聚合分类，实验显示其在多数据集的AUROC更高，且对参数变化和小样本场景鲁棒。**

- **链接: [http://arxiv.org/pdf/2505.21228v1](http://arxiv.org/pdf/2505.21228v1)**

> **作者:** Alvaro Gonzalez-Jimenez; Simone Lionetti; Ludovic Amruthalingam; Philippe Gottfrois; Fabian Gröger; Marc Pouly; Alexander A. Navarini
>
> **备注:** Provisionally Accepted at MICCAI 2025
>
> **摘要:** Medical anomaly detection has emerged as a promising solution to challenges in data availability and labeling constraints. Traditional methods extract features from different layers of pre-trained networks in Euclidean space; however, Euclidean representations fail to effectively capture the hierarchical relationships within these features, leading to suboptimal anomaly detection performance. We propose a novel yet simple approach that projects feature representations into hyperbolic space, aggregates them based on confidence levels, and classifies samples as healthy or anomalous. Our experiments demonstrate that hyperbolic space consistently outperforms Euclidean-based frameworks, achieving higher AUROC scores at both image and pixel levels across multiple medical benchmark datasets. Additionally, we show that hyperbolic space exhibits resilience to parameter variations and excels in few-shot scenarios, where healthy images are scarce. These findings underscore the potential of hyperbolic space as a powerful alternative for medical anomaly detection. The project website can be found at https://hyperbolic-anomalies.github.io
>
---
#### [new 111] Sci-Fi: Symmetric Constraint for Frame Inbetweening
- **分类: cs.CV**

- **简介: 该论文针对帧插值任务中起始与结束帧约束不对称导致生成不协调的问题，提出Sci-Fi框架。通过EF-Net模块增强结束帧约束强度，实现对称控制，提升视频过渡的连贯性与质量。**

- **链接: [http://arxiv.org/pdf/2505.21205v1](http://arxiv.org/pdf/2505.21205v1)**

> **作者:** Liuhan Chen; Xiaodong Cun; Xiaoyu Li; Xianyi He; Shenghai Yuan; Jie Chen; Ying Shan; Li Yuan
>
> **备注:** 22 pages, 9 figures, submitted to NeurIPS2025, under reviewering
>
> **摘要:** Frame inbetweening aims to synthesize intermediate video sequences conditioned on the given start and end frames. Current state-of-the-art methods mainly extend large-scale pre-trained Image-to-Video Diffusion models (I2V-DMs) by incorporating end-frame constraints via directly fine-tuning or omitting training. We identify a critical limitation in their design: Their injections of the end-frame constraint usually utilize the same mechanism that originally imposed the start-frame (single image) constraint. However, since the original I2V-DMs are adequately trained for the start-frame condition in advance, naively introducing the end-frame constraint by the same mechanism with much less (even zero) specialized training probably can't make the end frame have a strong enough impact on the intermediate content like the start frame. This asymmetric control strength of the two frames over the intermediate content likely leads to inconsistent motion or appearance collapse in generated frames. To efficiently achieve symmetric constraints of start and end frames, we propose a novel framework, termed Sci-Fi, which applies a stronger injection for the constraint of a smaller training scale. Specifically, it deals with the start-frame constraint as before, while introducing the end-frame constraint by an improved mechanism. The new mechanism is based on a well-designed lightweight module, named EF-Net, which encodes only the end frame and expands it into temporally adaptive frame-wise features injected into the I2V-DM. This makes the end-frame constraint as strong as the start-frame constraint, enabling our Sci-Fi to produce more harmonious transitions in various scenarios. Extensive experiments prove the superiority of our Sci-Fi compared with other baselines.
>
---
#### [new 112] Automatically Identify and Rectify: Robust Deep Contrastive Multi-view Clustering in Noisy Scenarios
- **分类: cs.CV**

- **简介: 该论文属于多视角聚类任务，解决噪声场景下性能退化问题。提出AIRMVC框架，通过GMM识别噪声，设计混合修复策略与鲁棒对比机制，生成去噪表示，理论证明与实验验证其在噪声场景中的优越性。**

- **链接: [http://arxiv.org/pdf/2505.21387v1](http://arxiv.org/pdf/2505.21387v1)**

> **作者:** Xihong Yang; Siwei Wang; Fangdi Wang; Jiaqi Jin; Suyuan Liu; Yue Liu; En Zhu; Xinwang Liu; Yueming Jin
>
> **摘要:** Leveraging the powerful representation learning capabilities, deep multi-view clustering methods have demonstrated reliable performance by effectively integrating multi-source information from diverse views in recent years. Most existing methods rely on the assumption of clean views. However, noise is pervasive in real-world scenarios, leading to a significant degradation in performance. To tackle this problem, we propose a novel multi-view clustering framework for the automatic identification and rectification of noisy data, termed AIRMVC. Specifically, we reformulate noisy identification as an anomaly identification problem using GMM. We then design a hybrid rectification strategy to mitigate the adverse effects of noisy data based on the identification results. Furthermore, we introduce a noise-robust contrastive mechanism to generate reliable representations. Additionally, we provide a theoretical proof demonstrating that these representations can discard noisy information, thereby improving the performance of downstream tasks. Extensive experiments on six benchmark datasets demonstrate that AIRMVC outperforms state-of-the-art algorithms in terms of robustness in noisy scenarios. The code of AIRMVC are available at https://github.com/xihongyang1999/AIRMVC on Github.
>
---
#### [new 113] Continual Learning on CLIP via Incremental Prompt Tuning with Intrinsic Textual Anchors
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，针对CLIP模型在增量任务中知识遗忘和方法复杂度高的问题，提出TPPT方法：通过文本原型作为稳定锚点双向引导视觉-文本提示调整，并加入正则化防止嵌入空间坍塌，提升持续学习效果。**

- **链接: [http://arxiv.org/pdf/2505.20680v1](http://arxiv.org/pdf/2505.20680v1)**

> **作者:** Haodong Lu; Xinyu Zhang; Kristen Moore; Jason Xue; Lina Yao; Anton van den Hengel; Dong Gong
>
> **备注:** Preprint
>
> **摘要:** Continual learning (CL) enables deep networks to acquire new knowledge while avoiding catastrophic forgetting. The powerful generalization ability of pre-trained models (PTMs), such as the Contrastive Language-Image Pre-training (CLIP) model, has inspired a range of CL methods targeting new and specialized tasks, providing rich multi-modal embeddings that support lightweight, incremental prompt tuning. Existing methods often rely on complex designs built upon specific assumptions, such as intricate regularization schemes for prompt pools, specialized routing mechanisms, or multi-stage incrementations, that introduce additional-and possibly unnecessary-complexity, underutilizing CLIP's intrinsic capabilities. In this paper, we propose a concise CL approach for CLIP based on incremental prompt tuning that fully exploits its multi-modal structure and the stability of textual representations. Our method, Textual Prototype-guided Prompt Tuning (TPPT), introduces textual prototypes not merely as static classifiers, as in existing methods, but as stable anchors to guide the learning of visual prompts, thereby shaping the embedding space (i.e., TPPT-V). We show that our bidirectional supervision strategy enables more effective learning of new knowledge while reducing forgetting. To further close the vision-language gap during CL, we jointly optimizes visual and textual prompts (i.e., TPPT-VT). We also introduce a relational diversity regularization on the textual anchors to prevent embedding space collapse and mitigate correlated forgetting. Extensive experiments and analyses demonstrate the effectiveness of our proposed approach, highlighting the benefits of leveraging CLIP's intrinsic guidance for continual adaptation.
>
---
#### [new 114] OrienText: Surface Oriented Textual Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，针对现有模型在复杂表面（如建筑、横幅）和多视角下难以准确添加文本的问题，提出OrienText方法。通过引入区域表面法线作为条件输入优化扩散模型，实现文本精准定位与方向校正。在自建数据集对比实验中验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.20958v1](http://arxiv.org/pdf/2505.20958v1)**

> **作者:** Shubham Singh Paliwal; Arushi Jain; Monika Sharma; Vikram Jamwal; Lovekesh Vig
>
> **备注:** 4 pages, SIGGRAPH Asia 2024 Technical Communications
>
> **摘要:** Textual content in images is crucial in e-commerce sectors, particularly in marketing campaigns, product imaging, advertising, and the entertainment industry. Current text-to-image (T2I) generation diffusion models, though proficient at producing high-quality images, often struggle to incorporate text accurately onto complex surfaces with varied perspectives, such as angled views of architectural elements like buildings, banners, or walls. In this paper, we introduce the Surface Oriented Textual Image Generation (OrienText) method, which leverages region-specific surface normals as conditional input to T2I generation diffusion model. Our approach ensures accurate rendering and correct orientation of the text within the image context. We demonstrate the effectiveness of the OrienText method on a self-curated dataset of images and compare it against the existing textual image generation methods.
>
---
#### [new 115] Plenodium: UnderWater 3D Scene Reconstruction with Plenoptic Medium Representation
- **分类: cs.CV**

- **简介: 该论文属于水下3D场景重建任务，解决现有方法无法同时建模物体与散射介质及初始化困难的问题。提出Plenodium框架，采用球面调和编码整合方向与位置信息，并通过伪深度高斯补充和深度排序正则化损失优化几何一致性，提升水下重建精度。**

- **链接: [http://arxiv.org/pdf/2505.21258v1](http://arxiv.org/pdf/2505.21258v1)**

> **作者:** Changguanng Wu; Jiangxin Dong; Chengjian Li; Jinhui Tang
>
> **摘要:** We present Plenodium (plenoptic medium), an effective and efficient 3D representation framework capable of jointly modeling both objects and participating media. In contrast to existing medium representations that rely solely on view-dependent modeling, our novel plenoptic medium representation incorporates both directional and positional information through spherical harmonics encoding, enabling highly accurate underwater scene reconstruction. To address the initialization challenge in degraded underwater environments, we propose the pseudo-depth Gaussian complementation to augment COLMAP-derived point clouds with robust depth priors. In addition, a depth ranking regularized loss is developed to optimize the geometry of the scene and improve the ordinal consistency of the depth maps. Extensive experiments on real-world underwater datasets demonstrate that our method achieves significant improvements in 3D reconstruction. Furthermore, we conduct a simulated dataset with ground truth and the controllable scattering medium to demonstrate the restoration capability of our method in underwater scenarios. Our code and dataset are available at https://plenodium.github.io/.
>
---
#### [new 116] Good Enough: Is it Worth Improving your Label Quality?
- **分类: cs.CV**

- **简介: 论文评估医学图像分割中标签质量提升的影响，探讨其成本效益。通过生成不同质量的伪标签CT数据集（如nnU-Net等），发现高质标签仅在超过阈值时显著提升性能，预训练时标签质量影响小，表明模型迁移通用概念为主。研究为是否值得改进标签提供依据。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20928v1](http://arxiv.org/pdf/2505.20928v1)**

> **作者:** Alexander Jaus; Zdravko Marinov; Constantin Seibold; Simon Reiß; Jens Kleesiek; Rainer Stiefelhagen
>
> **摘要:** Improving label quality in medical image segmentation is costly, but its benefits remain unclear. We systematically evaluate its impact using multiple pseudo-labeled versions of CT datasets, generated by models like nnU-Net, TotalSegmentator, and MedSAM. Our results show that while higher-quality labels improve in-domain performance, gains remain unclear if below a small threshold. For pre-training, label quality has minimal impact, suggesting that models rather transfer general concepts than detailed annotations. These findings provide guidance on when improving label quality is worth the effort.
>
---
#### [new 117] Minute-Long Videos with Dual Parallelisms
- **分类: cs.CV**

- **简介: 该论文提出DualParal方法，解决DiT模型生成长视频时的高延迟和内存问题。通过时空双路并行、分块渐降噪声、特征缓存及协同噪声初始化策略，实现在多GPU上高效生成高质量长视频，降低6.54倍延迟和1.48倍内存成本。**

- **链接: [http://arxiv.org/pdf/2505.21070v1](http://arxiv.org/pdf/2505.21070v1)**

> **作者:** Zeqing Wang; Bowen Zheng; Xingyi Yang; Yuecong Xu; Xinchao Wang
>
> **备注:** The code is available at https://github.com/DualParal-Project/DualParal
>
> **摘要:** Diffusion Transformer (DiT)-based video diffusion models generate high-quality videos at scale but incur prohibitive processing latency and memory costs for long videos. To address this, we propose a novel distributed inference strategy, termed DualParal. The core idea is that, instead of generating an entire video on a single GPU, we parallelize both temporal frames and model layers across GPUs. However, a naive implementation of this division faces a key limitation: since diffusion models require synchronized noise levels across frames, this implementation leads to the serialization of original parallelisms. We leverage a block-wise denoising scheme to handle this. Namely, we process a sequence of frame blocks through the pipeline with progressively decreasing noise levels. Each GPU handles a specific block and layer subset while passing previous results to the next GPU, enabling asynchronous computation and communication. To further optimize performance, we incorporate two key enhancements. Firstly, a feature cache is implemented on each GPU to store and reuse features from the prior block as context, minimizing inter-GPU communication and redundant computation. Secondly, we employ a coordinated noise initialization strategy, ensuring globally consistent temporal dynamics by sharing initial noise patterns across GPUs without extra resource costs. Together, these enable fast, artifact-free, and infinitely long video generation. Applied to the latest diffusion transformer video generator, our method efficiently produces 1,025-frame videos with up to 6.54$\times$ lower latency and 1.48$\times$ lower memory cost on 8$\times$RTX 4090 GPUs.
>
---
#### [new 118] CPathAgent: An Agent-based Foundation Model for Interpretable High-Resolution Pathology Image Analysis Mimicking Pathologists' Diagnostic Logic
- **分类: cs.CV**

- **简介: 该论文提出CPathAgent模型，解决现有病理图像分析方法无法模仿病理学家逐步缩放、系统诊断的问题。通过代理模型自主执行缩放/导航操作，结合多阶段训练策略整合多尺度分析能力，并构建新基准PathMMU-HR²，实现更可解释的诊断报告生成。**

- **链接: [http://arxiv.org/pdf/2505.20510v1](http://arxiv.org/pdf/2505.20510v1)**

> **作者:** Yuxuan Sun; Yixuan Si; Chenglu Zhu; Kai Zhang; Zhongyi Shui; Bowen Ding; Tao Lin; Lin Yang
>
> **备注:** 49 pages, 33 figures
>
> **摘要:** Recent advances in computational pathology have led to the emergence of numerous foundation models. However, these approaches fail to replicate the diagnostic process of pathologists, as they either simply rely on general-purpose encoders with multi-instance learning for classification or directly apply multimodal models to generate reports from images. A significant limitation is their inability to emulate the diagnostic logic employed by pathologists, who systematically examine slides at low magnification for overview before progressively zooming in on suspicious regions to formulate comprehensive diagnoses. To address this gap, we introduce CPathAgent, an innovative agent-based model that mimics pathologists' reasoning processes by autonomously executing zoom-in/out and navigation operations across pathology images based on observed visual features. To achieve this, we develop a multi-stage training strategy unifying patch-level, region-level, and whole-slide capabilities within a single model, which is essential for mimicking pathologists, who require understanding and reasoning capabilities across all three scales. This approach generates substantially more detailed and interpretable diagnostic reports compared to existing methods, particularly for huge region understanding. Additionally, we construct an expert-validated PathMMU-HR$^{2}$, the first benchmark for huge region analysis, a critical intermediate scale between patches and whole slides, as diagnosticians typically examine several key regions rather than entire slides at once. Extensive experiments demonstrate that CPathAgent consistently outperforms existing approaches across three scales of benchmarks, validating the effectiveness of our agent-based diagnostic approach and highlighting a promising direction for the future development of computational pathology.
>
---
#### [new 119] Geometry-Editable and Appearance-Preserving Object Compositon
- **分类: cs.CV**

- **简介: 该论文属于通用物体合成任务，解决几何编辑时外观细节丢失问题。提出DGAD模型，通过分离几何编辑与外观保留，先用语义嵌入隐式建模几何变换，再以交叉注意力机制对齐外观特征，实现精准编辑与保真。**

- **链接: [http://arxiv.org/pdf/2505.20914v1](http://arxiv.org/pdf/2505.20914v1)**

> **作者:** Jianman Lin; Haojie Li; Chunmei Qing; Zhijing Yang; Liang Lin; Tianshui Chen
>
> **摘要:** General object composition (GOC) aims to seamlessly integrate a target object into a background scene with desired geometric properties, while simultaneously preserving its fine-grained appearance details. Recent approaches derive semantic embeddings and integrate them into advanced diffusion models to enable geometry-editable generation. However, these highly compact embeddings encode only high-level semantic cues and inevitably discard fine-grained appearance details. We introduce a Disentangled Geometry-editable and Appearance-preserving Diffusion (DGAD) model that first leverages semantic embeddings to implicitly capture the desired geometric transformations and then employs a cross-attention retrieval mechanism to align fine-grained appearance features with the geometry-edited representation, facilitating both precise geometry editing and faithful appearance preservation in object composition. Specifically, DGAD builds on CLIP/DINO-derived and reference networks to extract semantic embeddings and appearance-preserving representations, which are then seamlessly integrated into the encoding and decoding pipelines in a disentangled manner. We first integrate the semantic embeddings into pre-trained diffusion models that exhibit strong spatial reasoning capabilities to implicitly capture object geometry, thereby facilitating flexible object manipulation and ensuring effective editability. Then, we design a dense cross-attention mechanism that leverages the implicitly learned object geometry to retrieve and spatially align appearance features with their corresponding regions, ensuring faithful appearance consistency. Extensive experiments on public benchmarks demonstrate the effectiveness of the proposed DGAD framework.
>
---
#### [new 120] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios
- **分类: cs.CV**

- **简介: 该论文评估多模态大模型在视频OCR中的能力，针对其在动态场景（如运动模糊、时序变化）中表现不佳的问题，构建包含10类25项任务、1,464个视频及2,000问答对的MME-VideoOCR基准。测试18个模型发现，最优模型准确率仅73.7%，凸显在时空推理、跨帧整合等任务中的不足，并强调高分辨率与时间覆盖的重要性。**

- **链接: [http://arxiv.org/pdf/2505.21333v1](http://arxiv.org/pdf/2505.21333v1)**

> **作者:** Yang Shi; Huanqian Wang; Wulin Xie; Huanyao Zhang; Lijie Zhao; Yi-Fan Zhang; Xinfeng Li; Chaoyou Fu; Zhuoer Wen; Wenting Liu; Zhuoran Zhang; Xinlong Chen; Bohan Zeng; Sihan Yang; Yuanxing Zhang; Pengfei Wan; Haotian Wang; Wenjing Yang
>
> **备注:** preprint
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved considerable accuracy in Optical Character Recognition (OCR) from static images. However, their efficacy in video OCR is significantly diminished due to factors such as motion blur, temporal variations, and visual effects inherent in video content. To provide clearer guidance for training practical MLLMs, we introduce the MME-VideoOCR benchmark, which encompasses a comprehensive range of video OCR application scenarios. MME-VideoOCR features 10 task categories comprising 25 individual tasks and spans 44 diverse scenarios. These tasks extend beyond text recognition to incorporate deeper comprehension and reasoning of textual content within videos. The benchmark consists of 1,464 videos with varying resolutions, aspect ratios, and durations, along with 2,000 meticulously curated, manually annotated question-answer pairs. We evaluate 18 state-of-the-art MLLMs on MME-VideoOCR, revealing that even the best-performing model (Gemini-2.5 Pro) achieves an accuracy of only 73.7%. Fine-grained analysis indicates that while existing MLLMs demonstrate strong performance on tasks where relevant texts are contained within a single or few frames, they exhibit limited capability in effectively handling tasks that demand holistic video comprehension. These limitations are especially evident in scenarios that require spatio-temporal reasoning, cross-frame information integration, or resistance to language prior bias. Our findings also highlight the importance of high-resolution visual input and sufficient temporal coverage for reliable OCR in dynamic video scenarios.
>
---
#### [new 121] DynamicVL: Benchmarking Multimodal Large Language Models for Dynamic City Understanding
- **分类: cs.CV**

- **简介: 该论文提出DVL-Suite框架（含DVL-Bench和DVL-Instruct），通过15,063幅多时相遥感图像，评估17个模型在长期城市动态分析（如扩张、灾害评估）中的能力，揭示其在长期时序和定量分析上的局限，并开发DVLChat作为基准模型。任务为多模态模型动态城市理解基准测试，解决其在长时间地球观测分析中的能力不足问题。**

- **链接: [http://arxiv.org/pdf/2505.21076v1](http://arxiv.org/pdf/2505.21076v1)**

> **作者:** Weihao Xuan; Junjue Wang; Heli Qi; Zihang Chen; Zhuo Zheng; Yanfei Zhong; Junshi Xia; Naoto Yokoya
>
> **摘要:** Multimodal large language models have demonstrated remarkable capabilities in visual understanding, but their application to long-term Earth observation analysis remains limited, primarily focusing on single-temporal or bi-temporal imagery. To address this gap, we introduce DVL-Suite, a comprehensive framework for analyzing long-term urban dynamics through remote sensing imagery. Our suite comprises 15,063 high-resolution (1.0m) multi-temporal images spanning 42 megacities in the U.S. from 2005 to 2023, organized into two components: DVL-Bench and DVL-Instruct. The DVL-Bench includes seven urban understanding tasks, from fundamental change detection (pixel-level) to quantitative analyses (regional-level) and comprehensive urban narratives (scene-level), capturing diverse urban dynamics including expansion/transformation patterns, disaster assessment, and environmental challenges. We evaluate 17 state-of-the-art multimodal large language models and reveal their limitations in long-term temporal understanding and quantitative analysis. These challenges motivate the creation of DVL-Instruct, a specialized instruction-tuning dataset designed to enhance models' capabilities in multi-temporal Earth observation. Building upon this dataset, we develop DVLChat, a baseline model capable of both image-level question-answering and pixel-level segmentation, facilitating a comprehensive understanding of city dynamics through language interactions.
>
---
#### [new 122] DreamBoothDPO: Improving Personalized Generation using Direct Preference Optimization
- **分类: cs.CV**

- **简介: 该论文属于个性化文本到图像生成任务，旨在解决用户定义概念的保真度与上下文一致性平衡问题。提出基于RL的DreamBoothDPO方法，通过合成优劣图像对进行偏好优化，无需人工标注，灵活调整图像质量与文本匹配的权衡，提升生成效果和收敛速度。**

- **链接: [http://arxiv.org/pdf/2505.20975v1](http://arxiv.org/pdf/2505.20975v1)**

> **作者:** Shamil Ayupov; Maksim Nakhodnov; Anastasia Yaschenko; Andrey Kuznetsov; Aibek Alanov
>
> **备注:** The first two authors contributed equally. The source code can be found at https://github.com/ControlGenAI/DreamBoothDPO
>
> **摘要:** Personalized diffusion models have shown remarkable success in Text-to-Image (T2I) generation by enabling the injection of user-defined concepts into diverse contexts. However, balancing concept fidelity with contextual alignment remains a challenging open problem. In this work, we propose an RL-based approach that leverages the diverse outputs of T2I models to address this issue. Our method eliminates the need for human-annotated scores by generating a synthetic paired dataset for DPO-like training using external quality metrics. These better-worse pairs are specifically constructed to improve both concept fidelity and prompt adherence. Moreover, our approach supports flexible adjustment of the trade-off between image fidelity and textual alignment. Through multi-step training, our approach outperforms a naive baseline in convergence speed and output quality. We conduct extensive qualitative and quantitative analysis, demonstrating the effectiveness of our method across various architectures and fine-tuning techniques. The source code can be found at https://github.com/ControlGenAI/DreamBoothDPO.
>
---
#### [new 123] Contrastive Desensitization Learning for Cross Domain Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文属于跨领域人脸伪造检测任务，旨在降低不同或未知伪造方法下的高误报率。提出Contrastive Desensitization Network（CDN），通过对比学习真实图像对的领域转换特征，增强模型鲁棒性，实验显示误报更低且检测更准。**

- **链接: [http://arxiv.org/pdf/2505.20675v1](http://arxiv.org/pdf/2505.20675v1)**

> **作者:** Lingyu Qiu; Ke Jiang; Xiaoyang Tan
>
> **摘要:** In this paper, we propose a new cross-domain face forgery detection method that is insensitive to different and possibly unseen forgery methods while ensuring an acceptable low false positive rate. Although existing face forgery detection methods are applicable to multiple domains to some degree, they often come with a high false positive rate, which can greatly disrupt the usability of the system. To address this issue, we propose an Contrastive Desensitization Network (CDN) based on a robust desensitization algorithm, which captures the essential domain characteristics through learning them from domain transformation over pairs of genuine face images. One advantage of CDN lies in that the learnt face representation is theoretical justified with regard to the its robustness against the domain changes. Extensive experiments over large-scale benchmark datasets demonstrate that our method achieves a much lower false alarm rate with improved detection accuracy compared to several state-of-the-art methods.
>
---
#### [new 124] ReassembleNet: Learnable Keypoints and Diffusion for 2D Fresco Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出ReassembleNet，用于2D壁画碎片重组。针对现有方法在可扩展性、多模态处理及现实复杂场景（如不规则形状、侵蚀）上的不足，其通过图神经网络选择轮廓关键点并融合多模态特征，结合扩散模型优化姿态，显著提升旋转和位移精度（RMSE提升55%及86%）。**

- **链接: [http://arxiv.org/pdf/2505.21117v1](http://arxiv.org/pdf/2505.21117v1)**

> **作者:** Adeela Islam; Stefano Fiorini; Stuart James; Pietro Morerio; Alessio Del Bue
>
> **摘要:** The task of reassembly is a significant challenge across multiple domains, including archaeology, genomics, and molecular docking, requiring the precise placement and orientation of elements to reconstruct an original structure. In this work, we address key limitations in state-of-the-art Deep Learning methods for reassembly, namely i) scalability; ii) multimodality; and iii) real-world applicability: beyond square or simple geometric shapes, realistic and complex erosion, or other real-world problems. We propose ReassembleNet, a method that reduces complexity by representing each input piece as a set of contour keypoints and learning to select the most informative ones by Graph Neural Networks pooling inspired techniques. ReassembleNet effectively lowers computational complexity while enabling the integration of features from multiple modalities, including both geometric and texture data. Further enhanced through pretraining on a semi-synthetic dataset. We then apply diffusion-based pose estimation to recover the original structure. We improve on prior methods by 55% and 86% for RMSE Rotation and Translation, respectively.
>
---
#### [new 125] MV-CoLight: Efficient Object Compositing with Consistent Lighting and Shadow Generation
- **分类: cs.CV**

- **简介: 该论文提出MV-CoLight框架，针对AR和3D场景中对象合成的光照一致性和多视角不一致问题。通过两阶段前馈架构直接建模光照与阴影，结合希尔伯特曲线映射实现2D-3D对齐，并构建大规模数据集。解决了现有方法在复杂场景、多变光照下的效率与一致性局限，实验验证其优越性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21483v1](http://arxiv.org/pdf/2505.21483v1)**

> **作者:** Kerui Ren; Jiayang Bai; Linning Xu; Lihan Jiang; Jiangmiao Pang; Mulin Yu; Bo Dai
>
> **摘要:** Object compositing offers significant promise for augmented reality (AR) and embodied intelligence applications. Existing approaches predominantly focus on single-image scenarios or intrinsic decomposition techniques, facing challenges with multi-view consistency, complex scenes, and diverse lighting conditions. Recent inverse rendering advancements, such as 3D Gaussian and diffusion-based methods, have enhanced consistency but are limited by scalability, heavy data requirements, or prolonged reconstruction time per scene. To broaden its applicability, we introduce MV-CoLight, a two-stage framework for illumination-consistent object compositing in both 2D images and 3D scenes. Our novel feed-forward architecture models lighting and shadows directly, avoiding the iterative biases of diffusion-based methods. We employ a Hilbert curve-based mapping to align 2D image inputs with 3D Gaussian scene representations seamlessly. To facilitate training and evaluation, we further introduce a large-scale 3D compositing dataset. Experiments demonstrate state-of-the-art harmonized results across standard benchmarks and our dataset, as well as casually captured real-world scenes demonstrate the framework's robustness and wide generalization.
>
---
#### [new 126] Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles
- **分类: cs.CV**

- **简介: 该论文提出Styl3R，解决快速3D风格化重建问题。现有方法计算量大且需密集输入，导致结构失真。Styl3R采用分叉架构分离结构与外观，结合身份损失预训练，通过新型视图合成任务保持重建能力，实现秒级处理稀疏图像，兼顾风格融合与多视角一致性，优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21060v1](http://arxiv.org/pdf/2505.21060v1)**

> **作者:** Peng Wang; Xiang Liu; Peidong Liu
>
> **备注:** Project page: https://nickisdope.github.io/Styl3R
>
> **摘要:** Stylizing 3D scenes instantly while maintaining multi-view consistency and faithfully resembling a style image remains a significant challenge. Current state-of-the-art 3D stylization methods typically involve computationally intensive test-time optimization to transfer artistic features into a pretrained 3D representation, often requiring dense posed input images. In contrast, leveraging recent advances in feed-forward reconstruction models, we demonstrate a novel approach to achieve direct 3D stylization in less than a second using unposed sparse-view scene images and an arbitrary style image. To address the inherent decoupling between reconstruction and stylization, we introduce a branched architecture that separates structure modeling and appearance shading, effectively preventing stylistic transfer from distorting the underlying 3D scene structure. Furthermore, we adapt an identity loss to facilitate pre-training our stylization model through the novel view synthesis task. This strategy also allows our model to retain its original reconstruction capabilities while being fine-tuned for stylization. Comprehensive evaluations, using both in-domain and out-of-domain datasets, demonstrate that our approach produces high-quality stylized 3D content that achieve a superior blend of style and scene appearance, while also outperforming existing methods in terms of multi-view consistency and efficiency.
>
---
#### [new 127] WeatherEdit: Controllable Weather Editing with 4D Gaussian Field
- **分类: cs.CV; cs.AI; cs.ET; cs.LG; cs.RO**

- **简介: WeatherEdit提出可控天气编辑任务，解决3D场景中生成多样且可调节强度的天气效果问题。方法分两阶段：1）通过适配器整合预训练扩散模型实现多天气风格背景生成，并设计TV注意力确保时空一致性；2）基于3D重建场景，用4D高斯场结合物理模拟生成动态天气粒子（雨/雪/雾），实现强度可控的真实渲染。应用于自动驾驶模拟。**

- **链接: [http://arxiv.org/pdf/2505.20471v1](http://arxiv.org/pdf/2505.20471v1)**

> **作者:** Chenghao Qian; Wenjing Li; Yuhu Guo; Gustav Markkula
>
> **摘要:** In this work, we present WeatherEdit, a novel weather editing pipeline for generating realistic weather effects with controllable types and severity in 3D scenes. Our approach is structured into two key components: weather background editing and weather particle construction. For weather background editing, we introduce an all-in-one adapter that integrates multiple weather styles into a single pretrained diffusion model, enabling the generation of diverse weather effects in 2D image backgrounds. During inference, we design a Temporal-View (TV-) attention mechanism that follows a specific order to aggregate temporal and spatial information, ensuring consistent editing across multi-frame and multi-view images. To construct the weather particles, we first reconstruct a 3D scene using the edited images and then introduce a dynamic 4D Gaussian field to generate snowflakes, raindrops and fog in the scene. The attributes and dynamics of these particles are precisely controlled through physical-based modelling and simulation, ensuring realistic weather representation and flexible severity adjustments. Finally, we integrate the 4D Gaussian field with the 3D scene to render consistent and highly realistic weather effects. Experiments on multiple driving datasets demonstrate that WeatherEdit can generate diverse weather effects with controllable condition severity, highlighting its potential for autonomous driving simulation in adverse weather. See project page: https://jumponthemoon.github.io/w-edit
>
---
#### [new 128] LeDiFlow: Learned Distribution-guided Flow Matching to Accelerate Image Generation
- **分类: cs.CV; cs.LG; I.4; I.2**

- **简介: 该论文提出LeDiFlow方法，旨在加速基于Flow Matching的图像生成。针对传统高斯先验导致弯曲路径、增加计算步骤的问题，通过学习更接近数据分布的先验，优化路径以减少推理步骤。实验显示像素空间下加速3.75倍，图像质量提升1.32倍。**

- **链接: [http://arxiv.org/pdf/2505.20723v1](http://arxiv.org/pdf/2505.20723v1)**

> **作者:** Pascal Zwick; Nils Friederich; Maximilian Beichter; Lennart Hilbert; Ralf Mikut; Oliver Bringmann
>
> **摘要:** Enhancing the efficiency of high-quality image generation using Diffusion Models (DMs) is a significant challenge due to the iterative nature of the process. Flow Matching (FM) is emerging as a powerful generative modeling paradigm based on a simulation-free training objective instead of a score-based one used in DMs. Typical FM approaches rely on a Gaussian distribution prior, which induces curved, conditional probability paths between the prior and target data distribution. These curved paths pose a challenge for the Ordinary Differential Equation (ODE) solver, requiring a large number of inference calls to the flow prediction network. To address this issue, we present Learned Distribution-guided Flow Matching (LeDiFlow), a novel scalable method for training FM-based image generation models using a better-suited prior distribution learned via a regression-based auxiliary model. By initializing the ODE solver with a prior closer to the target data distribution, LeDiFlow enables the learning of more computationally tractable probability paths. These paths directly translate to fewer solver steps needed for high-quality image generation at inference time. Our method utilizes a State-Of-The-Art (SOTA) transformer architecture combined with latent space sampling and can be trained on a consumer workstation. We empirically demonstrate that LeDiFlow remarkably outperforms the respective FM baselines. For instance, when operating directly on pixels, our model accelerates inference by up to 3.75x compared to the corresponding pixel-space baseline. Simultaneously, our latent FM model enhances image quality on average by 1.32x in CLIP Maximum Mean Discrepancy (CMMD) metric against its respective baseline.
>
---
#### [new 129] Be Decisive: Noise-Induced Layouts for Multi-Subject Generation
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决多主体生成中的布局泄漏与不准确问题。提出通过分析初始噪声预测动态布局，并在去噪过程中用神经网络优化，使布局与模型先验一致，提升生成稳定性与主体边界清晰度，同时保持多样性。**

- **链接: [http://arxiv.org/pdf/2505.21488v1](http://arxiv.org/pdf/2505.21488v1)**

> **作者:** Omer Dahary; Yehonathan Cohen; Or Patashnik; Kfir Aberman; Daniel Cohen-Or
>
> **备注:** SIGGRAPH 2025. Project page: https://omer11a.github.io/be-decisive/
>
> **摘要:** Generating multiple distinct subjects remains a challenge for existing text-to-image diffusion models. Complex prompts often lead to subject leakage, causing inaccuracies in quantities, attributes, and visual features. Preventing leakage among subjects necessitates knowledge of each subject's spatial location. Recent methods provide these spatial locations via an external layout control. However, enforcing such a prescribed layout often conflicts with the innate layout dictated by the sampled initial noise, leading to misalignment with the model's prior. In this work, we introduce a new approach that predicts a spatial layout aligned with the prompt, derived from the initial noise, and refines it throughout the denoising process. By relying on this noise-induced layout, we avoid conflicts with externally imposed layouts and better preserve the model's prior. Our method employs a small neural network to predict and refine the evolving noise-induced layout at each denoising step, ensuring clear boundaries between subjects while maintaining consistency. Experimental results show that this noise-aligned strategy achieves improved text-image alignment and more stable multi-subject generation compared to existing layout-guided techniques, while preserving the rich diversity of the model's original distribution.
>
---
#### [new 130] RF4D:Neural Radar Fields for Novel View Synthesis in Outdoor Dynamic Scenes
- **分类: cs.CV**

- **简介: 该论文提出RF4D框架，基于雷达数据进行户外动态场景的新型视图合成。针对RGB/LiDAR在恶劣天气脆弱及动态建模不足的问题，其整合时空信息，设计特征流模块预测时序偏移，并提出雷达物理渲染方法，提升动态场景合成精度与一致性。**

- **链接: [http://arxiv.org/pdf/2505.20967v1](http://arxiv.org/pdf/2505.20967v1)**

> **作者:** Jiarui Zhang; Zhihao Li; Chong Wang; Bihan Wen
>
> **摘要:** Neural fields (NFs) have demonstrated remarkable performance in scene reconstruction, powering various tasks such as novel view synthesis. However, existing NF methods relying on RGB or LiDAR inputs often exhibit severe fragility to adverse weather, particularly when applied in outdoor scenarios like autonomous driving. In contrast, millimeter-wave radar is inherently robust to environmental changes, while unfortunately, its integration with NFs remains largely underexplored. Besides, as outdoor driving scenarios frequently involve moving objects, making spatiotemporal modeling essential for temporally consistent novel view synthesis. To this end, we introduce RF4D, a radar-based neural field framework specifically designed for novel view synthesis in outdoor dynamic scenes. RF4D explicitly incorporates temporal information into its representation, significantly enhancing its capability to model moving objects. We further introduce a feature-level flow module that predicts latent temporal offsets between adjacent frames, enforcing temporal coherence in dynamic scene modeling. Moreover, we propose a radar-specific power rendering formulation closely aligned with radar sensing physics, improving synthesis accuracy and interoperability. Extensive experiments on public radar datasets demonstrate the superior performance of RF4D in terms of radar measurement synthesis quality and occupancy estimation accuracy, achieving especially pronounced improvements in dynamic outdoor scenarios.
>
---
#### [new 131] ReaMOT: A Benchmark and Framework for Reasoning-based Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文提出ReaMOT任务，解决现有RMOT方法无法处理含推理需求语言指令的问题。构建了基于12个数据集的基准测试（含分级推理指令和图像-语言对），提出评估指标及无需训练的ReaTrack框架（基于LVLM和SAM2），实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.20381v1](http://arxiv.org/pdf/2505.20381v1)**

> **作者:** Sijia Chen; Yanqiu Yu; En Yu; Wenbing Tao
>
> **备注:** 19 pages, 11 figures, 6 tables
>
> **摘要:** Referring Multi-object tracking (RMOT) is an important research field in computer vision. Its task form is to guide the models to track the objects that conform to the language instruction. However, the RMOT task commonly requires clear language instructions, such methods often fail to work when complex language instructions with reasoning characteristics appear. In this work, we propose a new task, called Reasoning-based Multi-Object Tracking (ReaMOT). ReaMOT is a more challenging task that requires accurate reasoning about objects that match the language instruction with reasoning characteristic and tracking the objects' trajectories. To advance the ReaMOT task and evaluate the reasoning capabilities of tracking models, we construct ReaMOT Challenge, a reasoning-based multi-object tracking benchmark built upon 12 datasets. Specifically, it comprises 1,156 language instructions with reasoning characteristic, 423,359 image-language pairs, and 869 diverse scenes, which is divided into three levels of reasoning difficulty. In addition, we propose a set of evaluation metrics tailored for the ReaMOT task. Furthermore, we propose ReaTrack, a training-free framework for reasoning-based multi-object tracking based on large vision-language models (LVLM) and SAM2, as a baseline for the ReaMOT task. Extensive experiments on the ReaMOT Challenge benchmark demonstrate the effectiveness of our ReaTrack framework.
>
---
#### [new 132] Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO
- **分类: cs.CV; cs.AI**

- **简介: 论文提出ACTIVE-O3框架，基于GRPO强化学习，解决多模态大语言模型（MLLM）主动感知效率低和区域选择不准的问题。系统定义任务，建立综合基准，验证其在小物体检测、自动驾驶等场景中的有效性，并展现零样本推理能力。**

- **链接: [http://arxiv.org/pdf/2505.21457v1](http://arxiv.org/pdf/2505.21457v1)**

> **作者:** Muzhi Zhu; Hao Zhong; Canyu Zhao; Zongze Du; Zheng Huang; Mingyu Liu; Hao Chen; Cheng Zou; Jingdong Chen; Ming Yang; Chunhua Shen
>
> **备注:** Project Page: https://aim-uofa.github.io/ACTIVE-o3
>
> **摘要:** Active vision, also known as active perception, refers to the process of actively selecting where and how to look in order to gather task-relevant information. It is a critical component of efficient perception and decision-making in humans and advanced embodied agents. Recently, the use of Multimodal Large Language Models (MLLMs) as central planning and decision-making modules in robotic systems has gained extensive attention. However, despite the importance of active perception in embodied intelligence, there is little to no exploration of how MLLMs can be equipped with or learn active perception capabilities. In this paper, we first provide a systematic definition of MLLM-based active perception tasks. We point out that the recently proposed GPT-o3 model's zoom-in search strategy can be regarded as a special case of active perception; however, it still suffers from low search efficiency and inaccurate region selection. To address these issues, we propose ACTIVE-O3, a purely reinforcement learning based training framework built on top of GRPO, designed to equip MLLMs with active perception capabilities. We further establish a comprehensive benchmark suite to evaluate ACTIVE-O3 across both general open-world tasks, such as small-object and dense object grounding, and domain-specific scenarios, including small object detection in remote sensing and autonomous driving, as well as fine-grained interactive segmentation. In addition, ACTIVE-O3 also demonstrates strong zero-shot reasoning abilities on the V* Benchmark, without relying on any explicit reasoning data. We hope that our work can provide a simple codebase and evaluation protocol to facilitate future research on active perception in MLLMs.
>
---
#### [new 133] ProBA: Probabilistic Bundle Adjustment with the Bhattacharyya Coefficient
- **分类: cs.CV**

- **简介: 该论文提出ProBA方法，改进传统Bundle Adjustment（BA）对初始估计和已知相机内参的依赖问题。通过3D高斯分布建模场景不确定性和Bhattacharyya系数保持几何一致性，实现无需初始化的鲁棒优化，提升SLAM在复杂环境中的可靠性。**

- **链接: [http://arxiv.org/pdf/2505.20858v1](http://arxiv.org/pdf/2505.20858v1)**

> **作者:** Jason Chui; Daniel Cremers
>
> **备注:** 15 pages, 14 figures, 5 tables
>
> **摘要:** Classical Bundle Adjustment (BA) methods require accurate initial estimates for convergence and typically assume known camera intrinsics, which limits their applicability when such information is uncertain or unavailable. We propose a novel probabilistic formulation of BA (ProBA) that explicitly models and propagates uncertainty in both the 2D observations and the 3D scene structure, enabling optimization without any prior knowledge of camera poses or focal length. Our method uses 3D Gaussians instead of point-like landmarks and we introduce uncertainty-aware reprojection losses by projecting the 3D Gaussians onto the 2D image space, and enforce geometric consistency across multiple 3D Gaussians using the Bhattacharyya coefficient to encourage overlap between their corresponding Gaussian distributions. This probabilistic framework leads to more robust and reliable optimization, even in the presence of outliers in the correspondence set, reducing the likelihood of converging to poor local minima. Experimental results show that \textit{ProBA} outperforms traditional methods in challenging real-world conditions. By removing the need for strong initialization and known intrinsics, ProBA enhances the practicality of SLAM systems deployed in unstructured environments.
>
---
#### [new 134] Mitigating Hallucination in Large Vision-Language Models via Adaptive Attention Calibration
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（LVLM）幻觉 mitigation 任务。针对LVLM在多模态生成中虚构图像不存在内容的问题，提出CAAC框架：通过视觉令牌校准（VTC）平衡图像注意力，利用自适应注意力重标定（AAR）基于模型置信度强化视觉 grounding，减少长文本生成中的幻觉。**

- **链接: [http://arxiv.org/pdf/2505.21472v1](http://arxiv.org/pdf/2505.21472v1)**

> **作者:** Mehrdad Fazli; Bowen Wei; Ziwei Zhu
>
> **摘要:** Large vision-language models (LVLMs) achieve impressive performance on multimodal tasks but often suffer from hallucination, and confidently describe objects or attributes not present in the image. Current inference-time interventions, while training-free, struggle to maintain accuracy in open-ended and long-form generation scenarios. We introduce the Confidence-Aware Attention Calibration (CAAC) framework to address this challenge by targeting two key biases: spatial perception bias, which distributes attention disproportionately across image tokens, and modality bias, which shifts focus from visual to textual inputs over time. CAAC employs a two-step approach: Visual-Token Calibration (VTC) to balance attention across visual tokens, and Adaptive Attention Re-Scaling (AAR) to reinforce visual grounding based on the model's confidence. This confidence-driven adjustment ensures consistent visual alignment during generation. Experiments on CHAIR, AMBER, and POPE benchmarks demonstrate that CAAC outperforms baselines, particularly in long-form generations, effectively reducing hallucination.
>
---
#### [new 135] Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?
- **分类: cs.CV**

- **简介: 该论文提出Video-Holmes基准，评估多模态大语言模型（MLLM）的复杂视频推理能力。针对现有视频任务仅需简单视觉感知的问题，设计需整合多线索的推理任务，包含1837个问题及270部短片，揭示模型在信息关联上的不足（最佳模型准确率45%），推动多模态推理研究。**

- **链接: [http://arxiv.org/pdf/2505.21374v1](http://arxiv.org/pdf/2505.21374v1)**

> **作者:** Junhao Cheng; Yuying Ge; Teng Wang; Yixiao Ge; Jing Liao; Ying Shan
>
> **备注:** Homepage: https://github.com/TencentARC/Video-Holmes
>
> **摘要:** Recent advances in CoT reasoning and RL post-training have been reported to enhance video reasoning capabilities of MLLMs. This progress naturally raises a question: can these models perform complex video reasoning in a manner comparable to human experts? However, existing video benchmarks primarily evaluate visual perception and grounding abilities, with questions that can be answered based on explicit prompts or isolated visual cues. Such benchmarks do not fully capture the intricacies of real-world reasoning, where humans must actively search for, integrate, and analyze multiple clues before reaching a conclusion. To address this issue, we present Video-Holmes, a benchmark inspired by the reasoning process of Sherlock Holmes, designed to evaluate the complex video reasoning capabilities of MLLMs. Video-Holmes consists of 1,837 questions derived from 270 manually annotated suspense short films, which spans seven carefully designed tasks. Each task is constructed by first identifying key events and causal relationships within films, and then designing questions that require models to actively locate and connect multiple relevant visual clues scattered across different video segments. Our comprehensive evaluation of state-of-the-art MLLMs reveals that, while these models generally excel at visual perception, they encounter substantial difficulties with integrating information and often miss critical clues. For example, the best-performing model, Gemini-2.5-Pro, achieves an accuracy of only 45%, with most models scoring below 40%. We aim that Video-Holmes can serve as a "Holmes-test" for multimodal reasoning, motivating models to reason more like humans and emphasizing the ongoing challenges in this field. The benchmark is released in https://github.com/TencentARC/Video-Holmes.
>
---
#### [new 136] HCQA-1.5 @ Ego4D EgoSchema Challenge 2025
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对Ego4D挑战赛中的第一人称视频问答任务，提出HCQA-1.5框架。通过多源聚合策略生成多样预测，结合置信度筛选机制选择高置信答案，对低置信案例引入细粒度视觉-上下文推理模块，提升预测可靠性，实现77%测试准确率。**

- **链接: [http://arxiv.org/pdf/2505.20644v1](http://arxiv.org/pdf/2505.20644v1)**

> **作者:** Haoyu Zhang; Yisen Feng; Qiaohui Chu; Meng Liu; Weili Guan; Yaowei Wang; Liqiang Nie
>
> **备注:** The third-place solution for the Ego4D EgoSchema Challenge at the CVPR EgoVis Workshop 2025
>
> **摘要:** In this report, we present the method that achieves third place for Ego4D EgoSchema Challenge in CVPR 2025. To improve the reliability of answer prediction in egocentric video question answering, we propose an effective extension to the previously proposed HCQA framework. Our approach introduces a multi-source aggregation strategy to generate diverse predictions, followed by a confidence-based filtering mechanism that selects high-confidence answers directly. For low-confidence cases, we incorporate a fine-grained reasoning module that performs additional visual and contextual analysis to refine the predictions. Evaluated on the EgoSchema blind test set, our method achieves 77% accuracy on over 5,000 human-curated multiple-choice questions, outperforming last year's winning solution and the majority of participating teams. Our code will be added at https://github.com/Hyu-Zhang/HCQA.
>
---
#### [new 137] RefAV: Towards Planning-Centric Scenario Mining
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于自动驾驶场景挖掘任务，旨在解决从原始驾驶日志中高效识别复杂安全场景的难题。提出RefAV数据集，利用视觉语言模型（VLM）通过自然语言查询定位多智能体交互场景，验证了现有VLM直接应用效果不佳，凸显了场景挖掘的独特挑战。**

- **链接: [http://arxiv.org/pdf/2505.20981v1](http://arxiv.org/pdf/2505.20981v1)**

> **作者:** Cainan Davidson; Deva Ramanan; Neehar Peri
>
> **摘要:** Autonomous Vehicles (AVs) collect and pseudo-label terabytes of multi-modal data localized to HD maps during normal fleet testing. However, identifying interesting and safety-critical scenarios from uncurated driving logs remains a significant challenge. Traditional scenario mining techniques are error-prone and prohibitively time-consuming, often relying on hand-crafted structured queries. In this work, we revisit spatio-temporal scenario mining through the lens of recent vision-language models (VLMs) to detect whether a described scenario occurs in a driving log and, if so, precisely localize it in both time and space. To address this problem, we introduce RefAV, a large-scale dataset of 10,000 diverse natural language queries that describe complex multi-agent interactions relevant to motion planning derived from 1000 driving logs in the Argoverse 2 Sensor dataset. We evaluate several referential multi-object trackers and present an empirical analysis of our baselines. Notably, we find that naively repurposing off-the-shelf VLMs yields poor performance, suggesting that scenario mining presents unique challenges. Our code and dataset are available at https://github.com/CainanD/RefAV/ and https://argoverse.github.io/user-guide/tasks/scenario_mining.html
>
---
#### [new 138] Frame-Level Captions for Long Video Generation with Complex Multi Scenes
- **分类: cs.CV**

- **简介: 该论文属于长视频生成任务，旨在解决复杂多场景视频生成中的误差累积和场景单一问题。提出帧级标注方法与注意力机制，结合扩散模型强制训练，使模型能精准匹配多场景文本提示，生成高质量长视频。**

- **链接: [http://arxiv.org/pdf/2505.20827v1](http://arxiv.org/pdf/2505.20827v1)**

> **作者:** Guangcong Zheng; Jianlong Yuan; Bo Wang; Haoyang Huang; Guoqing Ma; Nan Duan
>
> **摘要:** Generating long videos that can show complex stories, like movie scenes from scripts, has great promise and offers much more than short clips. However, current methods that use autoregression with diffusion models often struggle because their step-by-step process naturally leads to a serious error accumulation (drift). Also, many existing ways to make long videos focus on single, continuous scenes, making them less useful for stories with many events and changes. This paper introduces a new approach to solve these problems. First, we propose a novel way to annotate datasets at the frame-level, providing detailed text guidance needed for making complex, multi-scene long videos. This detailed guidance works with a Frame-Level Attention Mechanism to make sure text and video match precisely. A key feature is that each part (frame) within these windows can be guided by its own distinct text prompt. Our training uses Diffusion Forcing to provide the model with the ability to handle time flexibly. We tested our approach on difficult VBench 2.0 benchmarks ("Complex Plots" and "Complex Landscapes") based on the WanX2.1-T2V-1.3B model. The results show our method is better at following instructions in complex, changing scenes and creates high-quality long videos. We plan to share our dataset annotation methods and trained models with the research community. Project page: https://zgctroy.github.io/frame-level-captions .
>
---
#### [new 139] Temporal Saliency-Guided Distillation: A Scalable Framework for Distilling Video Datasets
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出单级视频数据集蒸馏框架，解决视频压缩中计算成本高及时间动态丢失问题。通过时序显著性引导过滤机制，利用帧间差异优化合成视频，抑制冗余并保留关键运动信息，实现高效视频压缩与性能最优。**

- **链接: [http://arxiv.org/pdf/2505.20694v1](http://arxiv.org/pdf/2505.20694v1)**

> **作者:** Xulin Gu; Xinhao Zhong; Zhixing Wei; Yimin Zhou; Shuoyang Sun; Bin Chen; Hongpeng Wang; Yuan Luo
>
> **摘要:** Dataset distillation (DD) has emerged as a powerful paradigm for dataset compression, enabling the synthesis of compact surrogate datasets that approximate the training utility of large-scale ones. While significant progress has been achieved in distilling image datasets, extending DD to the video domain remains challenging due to the high dimensionality and temporal complexity inherent in video data. Existing video distillation (VD) methods often suffer from excessive computational costs and struggle to preserve temporal dynamics, as na\"ive extensions of image-based approaches typically lead to degraded performance. In this paper, we propose a novel uni-level video dataset distillation framework that directly optimizes synthetic videos with respect to a pre-trained model. To address temporal redundancy and enhance motion preservation, we introduce a temporal saliency-guided filtering mechanism that leverages inter-frame differences to guide the distillation process, encouraging the retention of informative temporal cues while suppressing frame-level redundancy. Extensive experiments on standard video benchmarks demonstrate that our method achieves state-of-the-art performance, bridging the gap between real and distilled video data and offering a scalable solution for video dataset compression.
>
---
#### [new 140] Generative Image Compression by Estimating Gradients of the Rate-variable Feature Distribution
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于扩散模型的生成式图像压缩方法，将压缩过程建模为随机微分方程驱动的前向扩散，通过反向神经网络直接逆过程重建图像，无需噪声初始化。解决了传统方法效率低、重建质量差的问题，实现平滑率调整与高保真重建，实验显示其性能优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.20984v1](http://arxiv.org/pdf/2505.20984v1)**

> **作者:** Minghao Han; Weiyi You; Jinhua Zhang; Leheng Zhang; Ce Zhu; Shuhang Gu
>
> **摘要:** While learned image compression (LIC) focuses on efficient data transmission, generative image compression (GIC) extends this framework by integrating generative modeling to produce photo-realistic reconstructed images. In this paper, we propose a novel diffusion-based generative modeling framework tailored for generative image compression. Unlike prior diffusion-based approaches that indirectly exploit diffusion modeling, we reinterpret the compression process itself as a forward diffusion path governed by stochastic differential equations (SDEs). A reverse neural network is trained to reconstruct images by reversing the compression process directly, without requiring Gaussian noise initialization. This approach achieves smooth rate adjustment and photo-realistic reconstructions with only a minimal number of sampling steps. Extensive experiments on benchmark datasets demonstrate that our method outperforms existing generative image compression approaches across a range of metrics, including perceptual distortion, statistical fidelity, and no-reference quality assessments.
>
---
#### [new 141] LazyVLM: Neuro-Symbolic Approach to Video Analytics
- **分类: cs.DB; cs.AI; cs.CV; cs.IR; cs.MM**

- **简介: 论文提出LazyVLM系统，解决视频分析中VLMs计算成本高与神经符号方法依赖人工设计的局限。通过分解多帧查询为细粒度操作，结合关系查询与向量搜索，实现高效、用户友好的大规模视频数据交互式分析。**

- **链接: [http://arxiv.org/pdf/2505.21459v1](http://arxiv.org/pdf/2505.21459v1)**

> **作者:** Xiangru Jian; Wei Pang; Zhengyuan Dong; Chao Zhang; M. Tamer Özsu
>
> **备注:** 5 pages, 2 figures, Working paper
>
> **摘要:** Current video analytics approaches face a fundamental trade-off between flexibility and efficiency. End-to-end Vision Language Models (VLMs) often struggle with long-context processing and incur high computational costs, while neural-symbolic methods depend heavily on manual labeling and rigid rule design. In this paper, we introduce LazyVLM, a neuro-symbolic video analytics system that provides a user-friendly query interface similar to VLMs, while addressing their scalability limitation. LazyVLM enables users to effortlessly drop in video data and specify complex multi-frame video queries using a semi-structured text interface for video analytics. To address the scalability limitations of VLMs, LazyVLM decomposes multi-frame video queries into fine-grained operations and offloads the bulk of the processing to efficient relational query execution and vector similarity search. We demonstrate that LazyVLM provides a robust, efficient, and user-friendly solution for querying open-domain video data at scale.
>
---
#### [new 142] Multitemporal Latent Dynamical Framework for Hyperspectral Images Unmixing
- **分类: eess.IV; cs.CV; 68T07; I.4.10**

- **简介: 该论文提出多时相潜在动力学（MiLD）框架，解决高光谱图像解混中丰度动态建模缺失的问题。通过神经ODE建模丰度随时间演变，构建问题定义、动态离散化模型、求解算法及理论验证，确保方法一致性、收敛性与稳定性，实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.20902v1](http://arxiv.org/pdf/2505.20902v1)**

> **作者:** Ruiying Li; Bin Pan; Lan Ma; Xia Xu; Zhenwei Shi
>
> **备注:** 11 Pages,8 figures
>
> **摘要:** Multitemporal hyperspectral unmixing can capture dynamical evolution of materials. Despite its capability, current methods emphasize variability of endmembers while neglecting dynamics of abundances, which motivates our adoption of neural ordinary differential equations to model abundances temporally. However, this motivation is hindered by two challenges: the inherent complexity in defining, modeling and solving problem, and the absence of theoretical support. To address above challenges, in this paper, we propose a multitemporal latent dynamical (MiLD) unmixing framework by capturing dynamical evolution of materials with theoretical validation. For addressing multitemporal hyperspectral unmixing, MiLD consists of problem definition, mathematical modeling, solution algorithm and theoretical support. We formulate multitemporal unmixing problem definition by conducting ordinary differential equations and developing latent variables. We transfer multitemporal unmixing to mathematical model by dynamical discretization approaches, which describe the discreteness of observed sequence images with mathematical expansions. We propose algorithm to solve problem and capture dynamics of materials, which approximates abundance evolution by neural networks. Furthermore, we provide theoretical support by validating the crucial properties, which verifies consistency, convergence and stability theorems. The major contributions of MiLD include defining problem by ordinary differential equations, modeling problem by dynamical discretization approach, solving problem by multitemporal unmixing algorithm, and presenting theoretical support. Our experiments on both synthetic and real datasets have validated the utility of our work
>
---
#### [new 143] Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文是系统综述，探讨基础模型（如LLMs、多模态模型）在移动服务机器人中的应用，解决多模态融合、实时决策、任务泛化及人机交互等挑战。通过分析基础模型在传感器融合、语言控制和自适应任务执行中的作用，综述其在家庭、医疗和自动化领域的应用，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.20503v1](http://arxiv.org/pdf/2505.20503v1)**

> **作者:** Matthew Lisondra; Beno Benhabib; Goldie Nejat
>
> **摘要:** Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interactions, robots can improve understanding, adapt to, and execute complex tasks in dynamic real-world environments. However, embodied AI in mobile service robots continues to face key challenges, including multimodal sensor fusion, real-time decision-making under uncertainty, task generalization, and effective human-robot interactions (HRI). In this paper, we present the first systematic review of the integration of foundation models in mobile service robotics, identifying key open challenges in embodied AI and examining how foundation models can address them. Namely, we explore the role of such models in enabling real-time sensor fusion, language-conditioned control, and adaptive task execution. Furthermore, we discuss real-world applications in the domestic assistance, healthcare, and service automation sectors, demonstrating the transformative impact of foundation models on service robotics. We also include potential future research directions, emphasizing the need for predictive scaling laws, autonomous long-term adaptation, and cross-embodiment generalization to enable scalable, efficient, and robust deployment of foundation models in human-centric robotic systems.
>
---
#### [new 144] Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Ui2i模型，用于无配对图像翻译任务，解决跨域风格转换中结构保真难题。针对生物医学场景（如免疫组化核分割、免疫荧光信号分离），改进CycleGAN架构：采用U-Net结构传递局部特征，使用双向频谱归一化增强稳定性，并整合注意力机制，提升内容完整性。实验验证其在结构敏感任务中的优势。**

- **链接: [http://arxiv.org/pdf/2505.20746v1](http://arxiv.org/pdf/2505.20746v1)**

> **作者:** Nikola Andrejic; Milica Spasic; Igor Mihajlovic; Petra Milosavljevic; Djordje Pavlovic; Filip Milisavljevic; Uros Milivojevic; Danilo Delibasic; Ivana Mikic; Sinisa Todorovic
>
> **备注:** submitted to NeurIPs 2025
>
> **摘要:** This work introduces Ui2i, a novel model for unpaired image-to-image translation, trained on content-wise unpaired datasets to enable style transfer across domains while preserving content. Building on CycleGAN, Ui2i incorporates key modifications to better disentangle content and style features, and preserve content integrity. Specifically, Ui2i employs U-Net-based generators with skip connections to propagate localized shallow features deep into the generator. Ui2i removes feature-based normalization layers from all modules and replaces them with approximate bidirectional spectral normalization -- a parameter-based alternative that enhances training stability. To further support content preservation, channel and spatial attention mechanisms are integrated into the generators. Training is facilitated through image scale augmentation. Evaluation on two biomedical tasks -- domain adaptation for nuclear segmentation in immunohistochemistry (IHC) images and unmixing of biological structures superimposed in single-channel immunofluorescence (IF) images -- demonstrates Ui2i's ability to preserve content fidelity in settings that demand more accurate structural preservation than typical translation tasks. To the best of our knowledge, Ui2i is the first approach capable of separating superimposed signals in IF images using real, unpaired training data.
>
---
#### [new 145] Cultural Awareness in Vision-Language Models: A Cross-Country Exploration
- **分类: cs.CY; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型（VLMs）偏见评估任务，旨在探究VLMs在跨国家场景中编码文化差异及种族、性别、身体特征偏见的问题。研究提出三类检索任务（种族-国家、特质-国家、体征-国家关联分析），揭示VLMs存在强化社会刻板印象的持续偏见。**

- **链接: [http://arxiv.org/pdf/2505.20326v1](http://arxiv.org/pdf/2505.20326v1)**

> **作者:** Avinash Madasu; Vasudev Lal; Phillip Howard
>
> **摘要:** Vision-Language Models (VLMs) are increasingly deployed in diverse cultural contexts, yet their internal biases remain poorly understood. In this work, we propose a novel framework to systematically evaluate how VLMs encode cultural differences and biases related to race, gender, and physical traits across countries. We introduce three retrieval-based tasks: (1) Race to Country retrieval, which examines the association between individuals from specific racial groups (East Asian, White, Middle Eastern, Latino, South Asian, and Black) and different countries; (2) Personal Traits to Country retrieval, where images are paired with trait-based prompts (e.g., Smart, Honest, Criminal, Violent) to investigate potential stereotypical associations; and (3) Physical Characteristics to Country retrieval, focusing on visual attributes like skinny, young, obese, and old to explore how physical appearances are culturally linked to nations. Our findings reveal persistent biases in VLMs, highlighting how visual representations may inadvertently reinforce societal stereotypes.
>
---
#### [new 146] Vision-Based Risk Aware Emergency Landing for UAVs in Complex Urban Environments
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于无人机(UAV)紧急着陆任务，解决复杂城市环境中动态障碍物和视觉挑战下的安全着陆问题。提出基于语义分割的实时风险评估方法，通过深度神经网络生成像素级风险图，动态识别稳定着陆区，并设计高度自适应的控制策略确保安全轨迹，实验显示其在高风险场景下成功率超90%。**

- **链接: [http://arxiv.org/pdf/2505.20423v1](http://arxiv.org/pdf/2505.20423v1)**

> **作者:** Julio de la Torre-Vanegas; Miguel Soriano-Garcia; Israel Becerra; Diego Mercado-Ravell
>
> **摘要:** Landing safely in crowded urban environments remains an essential yet challenging endeavor for Unmanned Aerial Vehicles (UAVs), especially in emergency situations. In this work, we propose a risk-aware approach that harnesses semantic segmentation to continuously evaluate potential hazards in the drone's field of view. By using a specialized deep neural network to assign pixel-level risk values and applying an algorithm based on risk maps, our method adaptively identifies a stable Safe Landing Zone (SLZ) despite moving critical obstacles such as vehicles, people, etc., and other visual challenges like shifting illumination. A control system then guides the UAV toward this low-risk region, employing altitude-dependent safety thresholds and temporal landing point stabilization to ensure robust descent trajectories. Experimental validation in diverse urban environments demonstrates the effectiveness of our approach, achieving over 90% landing success rates in very challenging real scenarios, showing significant improvements in various risk metrics. Our findings suggest that risk-oriented vision methods can effectively help reduce the risk of accidents in emergency landing situations, particularly in complex, unstructured, urban scenarios, densely populated with moving risky obstacles, while potentiating the true capabilities of UAVs in complex urban operations.
>
---
#### [new 147] Stochastic Preconditioning for Neural Field Optimization
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于神经场优化任务，旨在解决训练中收敛慢和依赖复杂结构的问题。提出通过空间随机采样（高斯偏移）隐式模糊场的方法，替代现有层次化设计，提升优化鲁棒性。实验验证其在多种模型和视觉任务中的有效性，简单且性能优越。**

- **链接: [http://arxiv.org/pdf/2505.20473v1](http://arxiv.org/pdf/2505.20473v1)**

> **作者:** Selena Ling; Merlin Nimier-David; Alec Jacobson; Nicholas Sharp
>
> **备注:** 15 pages, 11 figures, SIGGRAPH 2025 (Journal track)
>
> **摘要:** Neural fields are a highly effective representation across visual computing. This work observes that fitting these fields is greatly improved by incorporating spatial stochasticity during training, and that this simple technique can replace or even outperform custom-designed hierarchies and frequency space constructions. The approach is formalized as implicitly operating on a blurred version of the field, evaluated in-expectation by sampling with Gaussian-distributed offsets. Querying the blurred field during optimization greatly improves convergence and robustness, akin to the role of preconditioners in numerical linear algebra. This implicit, sampling-based perspective fits naturally into the neural field paradigm, comes at no additional cost, and is extremely simple to implement. We describe the basic theory of this technique, including details such as handling boundary conditions, and extending to a spatially-varying blur. Experiments demonstrate this approach on representations including coordinate MLPs, neural hashgrids, triplanes, and more, across tasks including surface reconstruction and radiance fields. In settings where custom-designed hierarchies have already been developed, stochastic preconditioning nearly matches or improves their performance with a simple and unified approach; in settings without existing hierarchies it provides an immediate boost to quality and robustness.
>
---
#### [new 148] Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于单步扩散模型蒸馏与生成任务，旨在解决现有方法理论分散、生成效果受限的问题。提出Uni-Instruct框架，通过统一f-散度理论，设计可计算损失函数，实现高效训练，提升生成质量与多样性，在CIFAR10、ImageNet等数据集及文本到3D生成中取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2505.20755v1](http://arxiv.org/pdf/2505.20755v1)**

> **作者:** Yifei Wang; Weimin Bai; Colin Zhang; Debing Zhang; Weijian Luo; He Sun
>
> **摘要:** In this paper, we unify more than 10 existing one-step diffusion distillation approaches, such as Diff-Instruct, DMD, SIM, SiD, $f$-distill, etc, inside a theory-driven framework which we name the \textbf{\emph{Uni-Instruct}}. Uni-Instruct is motivated by our proposed diffusion expansion theory of the $f$-divergence family. Then we introduce key theories that overcome the intractability issue of the original expanded $f$-divergence, resulting in an equivalent yet tractable loss that effectively trains one-step diffusion models by minimizing the expanded $f$-divergence family. The novel unification introduced by Uni-Instruct not only offers new theoretical contributions that help understand existing approaches from a high-level perspective but also leads to state-of-the-art one-step diffusion generation performances. On the CIFAR10 generation benchmark, Uni-Instruct achieves record-breaking Frechet Inception Distance (FID) values of \textbf{\emph{1.46}} for unconditional generation and \textbf{\emph{1.38}} for conditional generation. On the ImageNet-$64\times 64$ generation benchmark, Uni-Instruct achieves a new SoTA one-step generation FID of \textbf{\emph{1.02}}, which outperforms its 79-step teacher diffusion with a significant improvement margin of 1.33 (1.02 vs 2.35). We also apply Uni-Instruct on broader tasks like text-to-3D generation. For text-to-3D generation, Uni-Instruct gives decent results, which slightly outperforms previous methods, such as SDS and VSD, in terms of both generation quality and diversity. Both the solid theoretical and empirical contributions of Uni-Instruct will potentially help future studies on one-step diffusion distillation and knowledge transferring of diffusion models.
>
---
#### [new 149] IKMo: Image-Keyframed Motion Generation with Trajectory-Pose Conditioned Motion Diffusion Model
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出IKMo方法，针对现有轨迹与姿态全局处理导致的运动生成质量不足问题。通过解耦轨迹-姿态输入，采用两阶段处理（优化模块+并行编码器）及Motion ControlNet引导生成，提升运动的时空保真度和可控性。实验显示优于现有方法，并结合MLLM代理预处理用户输入，增强生成效果与用户预期匹配度。**

- **链接: [http://arxiv.org/pdf/2505.21146v1](http://arxiv.org/pdf/2505.21146v1)**

> **作者:** Yang Zhao; Yan Zhang; Xubo Yang
>
> **摘要:** Existing human motion generation methods with trajectory and pose inputs operate global processing on both modalities, leading to suboptimal outputs. In this paper, we propose IKMo, an image-keyframed motion generation method based on the diffusion model with trajectory and pose being decoupled. The trajectory and pose inputs go through a two-stage conditioning framework. In the first stage, the dedicated optimization module is applied to refine inputs. In the second stage, trajectory and pose are encoded via a Trajectory Encoder and a Pose Encoder in parallel. Then, motion with high spatial and semantic fidelity is guided by a motion ControlNet, which processes the fused trajectory and pose data. Experiment results based on HumanML3D and KIT-ML datasets demonstrate that the proposed method outperforms state-of-the-art on all metrics under trajectory-keyframe constraints. In addition, MLLM-based agents are implemented to pre-process model inputs. Given texts and keyframe images from users, the agents extract motion descriptions, keyframe poses, and trajectories as the optimized inputs into the motion generation model. We conducts a user study with 10 participants. The experiment results prove that the MLLM-based agents pre-processing makes generated motion more in line with users' expectation. We believe that the proposed method improves both the fidelity and controllability of motion generation by the diffusion model.
>
---
#### [new 150] FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation
- **分类: cs.LG; cs.AI; cs.CV; cs.MM; cs.PF**

- **简介: 该论文属于加速扩散Transformer（DiT）推理任务。针对其计算密集、迭代结构及深层Transformer堆栈导致的效率问题，提出FastCache框架：1）采用空间感知的冗余标记过滤机制，基于隐藏状态重要性选择关键信息；2）通过统计检验在时间步间复用潜变量激活，减少冗余计算。理论分析证明误差有界，实验表明显著降低延迟和内存使用，且生成质量最优。**

- **链接: [http://arxiv.org/pdf/2505.20353v1](http://arxiv.org/pdf/2505.20353v1)**

> **作者:** Dong Liu; Jiayi Zhang; Yifan Li; Yanxuan Yu; Ben Lengerich; Ying Nian Wu
>
> **摘要:** Diffusion Transformers (DiT) are powerful generative models but remain computationally intensive due to their iterative structure and deep transformer stacks. To alleviate this inefficiency, we propose FastCache, a hidden-state-level caching and compression framework that accelerates DiT inference by exploiting redundancy within the model's internal representations. FastCache introduces a dual strategy: (1) a spatial-aware token selection mechanism that adaptively filters redundant tokens based on hidden state saliency, and (2) a transformer-level cache that reuses latent activations across timesteps when changes are statistically insignificant. These modules work jointly to reduce unnecessary computation while preserving generation fidelity through learnable linear approximation. Theoretical analysis shows that FastCache maintains bounded approximation error under a hypothesis-testing-based decision rule. Empirical evaluations across multiple DiT variants demonstrate substantial reductions in latency and memory usage, with best generation output quality compared to other cache methods, as measured by FID and t-FID. Code implementation of FastCache is available on GitHub at https://github.com/NoakLiu/FastCache-xDiT.
>
---
#### [new 151] Bi-Level Unsupervised Feature Selection
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于无监督特征选择任务，旨在解决现有方法无法同时评估特征重要性与保持数据结构的问题。提出双层框架BLUFS，结合谱聚类生成伪标签与ℓ₂₀-范数约束，通过PAM算法优化模型，实验验证其在聚类与分类中的优势。**

- **链接: [http://arxiv.org/pdf/2505.20563v1](http://arxiv.org/pdf/2505.20563v1)**

> **作者:** Jingjing Liu; Xiansen Ju; Xianchao Xiu; Wanquan Liu
>
> **摘要:** Unsupervised feature selection (UFS) is an important task in data engineering. However, most UFS methods construct models from a single perspective and often fail to simultaneously evaluate feature importance and preserve their inherent data structure, thus limiting their performance. To address this challenge, we propose a novel bi-level unsupervised feature selection (BLUFS) method, including a clustering level and a feature level. Specifically, at the clustering level, spectral clustering is used to generate pseudo-labels for representing the data structure, while a continuous linear regression model is developed to learn the projection matrix. At the feature level, the $\ell_{2,0}$-norm constraint is imposed on the projection matrix for more effectively selecting features. To the best of our knowledge, this is the first work to combine a bi-level framework with the $\ell_{2,0}$-norm. To solve the proposed bi-level model, we design an efficient proximal alternating minimization (PAM) algorithm, whose subproblems either have explicit solutions or can be computed by fast solvers. Furthermore, we establish the convergence result and computational complexity. Finally, extensive experiments on two synthetic datasets and eight real datasets demonstrate the superiority of BLUFS in clustering and classification tasks.
>
---
#### [new 152] CityGo: Lightweight Urban Modeling and Rendering with Proxy Buildings and Residual Gaussians
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出CityGo框架，针对大规模城市建模与渲染任务，解决航拍重建中的遮挡、内存占用高及效率低问题。通过结合代理建筑网格与残差高斯，提取建筑网格并利用高斯捕捉细节，优化纹理与参数，实现实时渲染，减少训练时间与内存消耗，适用于移动设备。**

- **链接: [http://arxiv.org/pdf/2505.21041v1](http://arxiv.org/pdf/2505.21041v1)**

> **作者:** Weihang Liu; Yuhui Zhong; Yuke Li; Xi Chen; Jiadi Cui; Honglong Zhang; Lan Xu; Xin Lou; Yujiao Shi; Jingyi Yu; Yingliang Zhang
>
> **摘要:** Accurate and efficient modeling of large-scale urban scenes is critical for applications such as AR navigation, UAV based inspection, and smart city digital twins. While aerial imagery offers broad coverage and complements limitations of ground-based data, reconstructing city-scale environments from such views remains challenging due to occlusions, incomplete geometry, and high memory demands. Recent advances like 3D Gaussian Splatting (3DGS) improve scalability and visual quality but remain limited by dense primitive usage, long training times, and poor suit ability for edge devices. We propose CityGo, a hybrid framework that combines textured proxy geometry with residual and surrounding 3D Gaussians for lightweight, photorealistic rendering of urban scenes from aerial perspectives. Our approach first extracts compact building proxy meshes from MVS point clouds, then uses zero order SH Gaussians to generate occlusion-free textures via image-based rendering and back-projection. To capture high-frequency details, we introduce residual Gaussians placed based on proxy-photo discrepancies and guided by depth priors. Broader urban context is represented by surrounding Gaussians, with importance-aware downsampling applied to non-critical regions to reduce redundancy. A tailored optimization strategy jointly refines proxy textures and Gaussian parameters, enabling real-time rendering of complex urban scenes on mobile GPUs with significantly reduced training and memory requirements. Extensive experiments on real-world aerial datasets demonstrate that our hybrid representation significantly reduces training time, achieving on average 1.4x speedup, while delivering comparable visual fidelity to pure 3D Gaussian Splatting approaches. Furthermore, CityGo enables real-time rendering of large-scale urban scenes on mobile consumer GPUs, with substantially reduced memory usage and energy consumption.
>
---
#### [new 153] Object-Centric Action-Enhanced Representations for Robot Visuo-Motor Policy Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉-运动策略学习任务，旨在解决现有方法中语义分割与视觉编码分离及依赖标注数据的问题。提出集成Slot Attention和SOLV模型的object-centric编码器，通过外部预训练及人类动作数据微调，提升强化/模仿学习效果，减少对机器人专用数据的依赖。**

- **链接: [http://arxiv.org/pdf/2505.20962v1](http://arxiv.org/pdf/2505.20962v1)**

> **作者:** Nikos Giannakakis; Argyris Manetas; Panagiotis P. Filntisis; Petros Maragos; George Retsinas
>
> **摘要:** Learning visual representations from observing actions to benefit robot visuo-motor policy generation is a promising direction that closely resembles human cognitive function and perception. Motivated by this, and further inspired by psychological theories suggesting that humans process scenes in an object-based fashion, we propose an object-centric encoder that performs semantic segmentation and visual representation generation in a coupled manner, unlike other works, which treat these as separate processes. To achieve this, we leverage the Slot Attention mechanism and use the SOLV model, pretrained in large out-of-domain datasets, to bootstrap fine-tuning on human action video data. Through simulated robotic tasks, we demonstrate that visual representations can enhance reinforcement and imitation learning training, highlighting the effectiveness of our integrated approach for semantic segmentation and encoding. Furthermore, we show that exploiting models pretrained on out-of-domain datasets can benefit this process, and that fine-tuning on datasets depicting human actions -- although still out-of-domain -- , can significantly improve performance due to close alignment with robotic tasks. These findings show the capability to reduce reliance on annotated or robot-specific action datasets and the potential to build on existing visual encoders to accelerate training and improve generalizability.
>
---
#### [new 154] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于LLM行为控制任务，旨在解决参数纠缠导致的控制精度不足和副作用问题。提出Steering Target Atoms（STA）方法，通过分离并操纵知识组件实现精准安全控制，在对抗场景及复杂推理中验证了其鲁棒性与灵活性。**

- **链接: [http://arxiv.org/pdf/2505.20322v1](http://arxiv.org/pdf/2505.20322v1)**

> **作者:** Mengru Wang; Ziwen Xu; Shengyu Mao; Shumin Deng; Zhaopeng Tu; Huajun Chen; Ningyu Zhang
>
> **摘要:** Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control.
>
---
#### [new 155] HoPE: Hybrid of Position Embedding for Length Generalization in Vision-Language Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉语言模型（VLM）的长上下文处理任务，旨在解决其在长视频等场景中性能下降的问题。现有方法通过启发式分配RoPE的3D频率但效果不佳。论文提出HoPE，结合混合频率分配策略与动态时间缩放机制，提升长视频理解与检索能力，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20444v1](http://arxiv.org/pdf/2505.20444v1)**

> **作者:** Haoran Li; Yingjie Qin; Baoyuan Ou; Lai Xu; Ruiwen Xu
>
> **摘要:** Vision-Language Models (VLMs) have made significant progress in multimodal tasks. However, their performance often deteriorates in long-context scenarios, particularly long videos. While Rotary Position Embedding (RoPE) has been widely adopted for length generalization in Large Language Models (LLMs), extending vanilla RoPE to capture the intricate spatial-temporal dependencies in videos remains an unsolved challenge. Existing methods typically allocate different frequencies within RoPE to encode 3D positional information. However, these allocation strategies mainly rely on heuristics, lacking in-depth theoretical analysis. In this paper, we first study how different allocation strategies impact the long-context capabilities of VLMs. Our analysis reveals that current multimodal RoPEs fail to reliably capture semantic similarities over extended contexts. To address this issue, we propose HoPE, a Hybrid of Position Embedding designed to improve the long-context capabilities of VLMs. HoPE introduces a hybrid frequency allocation strategy for reliable semantic modeling over arbitrarily long context, and a dynamic temporal scaling mechanism to facilitate robust learning and flexible inference across diverse context lengths. Extensive experiments across four video benchmarks on long video understanding and retrieval tasks demonstrate that HoPE consistently outperforms existing methods, confirming its effectiveness. Code is available at https://github.com/hrlics/HoPE.
>
---
#### [new 156] Prostate Cancer Screening with Artificial Intelligence-Enhanced Micro-Ultrasound: A Comparative Study with Traditional Methods
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于前列腺癌筛查任务，旨在比较AI增强微超声与传统PSA/DRE方法的诊断效能。研究通过AI模型分析微超声图像特征，结合随机森林分类器预测癌症，并与临床模型对比。结果显示AI模型（AUROC 0.871）特异性（68.1%）显著优于传统方法（27.3%），同时保持高敏感性，可减少不必要的活检。**

- **链接: [http://arxiv.org/pdf/2505.21355v1](http://arxiv.org/pdf/2505.21355v1)**

> **作者:** Muhammad Imran; Wayne G. Brisbane; Li-Ming Su; Jason P. Joseph; Wei Shao
>
> **摘要:** Background and objective: Micro-ultrasound (micro-US) is a novel imaging modality with diagnostic accuracy comparable to MRI for detecting clinically significant prostate cancer (csPCa). We investigated whether artificial intelligence (AI) interpretation of micro-US can outperform clinical screening methods using PSA and digital rectal examination (DRE). Methods: We retrospectively studied 145 men who underwent micro-US guided biopsy (79 with csPCa, 66 without). A self-supervised convolutional autoencoder was used to extract deep image features from 2D micro-US slices. Random forest classifiers were trained using five-fold cross-validation to predict csPCa at the slice level. Patients were classified as csPCa-positive if 88 or more consecutive slices were predicted positive. Model performance was compared with a classifier using PSA, DRE, prostate volume, and age. Key findings and limitations: The AI-based micro-US model and clinical screening model achieved AUROCs of 0.871 and 0.753, respectively. At a fixed threshold, the micro-US model achieved 92.5% sensitivity and 68.1% specificity, while the clinical model showed 96.2% sensitivity but only 27.3% specificity. Limitations include a retrospective single-center design and lack of external validation. Conclusions and clinical implications: AI-interpreted micro-US improves specificity while maintaining high sensitivity for csPCa detection. This method may reduce unnecessary biopsies and serve as a low-cost alternative to PSA-based screening. Patient summary: We developed an AI system to analyze prostate micro-ultrasound images. It outperformed PSA and DRE in detecting aggressive cancer and may help avoid unnecessary biopsies.
>
---
#### [new 157] Leaner Transformers: More Heads, Less Depth
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出优化Transformer模型结构，挑战"越大越好"理念。通过理论分析发现多头注意力能改善模型条件数，从而重新设计架构：增加注意力头数并减少深度，使参数减少30-50%同时保持精度，适用于CV和语言任务。**

- **链接: [http://arxiv.org/pdf/2505.20802v1](http://arxiv.org/pdf/2505.20802v1)**

> **作者:** Hemanth Saratchandran; Damien Teney; Simon Lucey
>
> **摘要:** Transformers have reshaped machine learning by utilizing attention mechanisms to capture complex patterns in large datasets, leading to significant improvements in performance. This success has contributed to the belief that "bigger means better", leading to ever-increasing model sizes. This paper challenge this ideology by showing that many existing transformers might be unnecessarily oversized. We discover a theoretical principle that redefines the role of multi-head attention. An important benefit of the multiple heads is in improving the conditioning of the attention block. We exploit this theoretical insight and redesign popular architectures with an increased number of heads. The improvement in the conditioning proves so significant in practice that model depth can be decreased, reducing the parameter count by up to 30-50% while maintaining accuracy. We obtain consistent benefits across a variety of transformer-based architectures of various scales, on tasks in computer vision (ImageNet-1k) as well as language and sequence modeling (GLUE benchmark, TinyStories, and the Long-Range Arena benchmark).
>
---
#### [new 158] Topological Deep Learning for Speech Data
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升低噪声场景及跨领域适应性。通过拓扑数据分析设计拓扑感知卷积核，理论分解矩阵空间为纤维丛，并提出正交特征层，优化神经网络，显著提升音素识别性能。**

- **链接: [http://arxiv.org/pdf/2505.21173v1](http://arxiv.org/pdf/2505.21173v1)**

> **作者:** Zhiwang Yu
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Topological data analysis (TDA) offers novel mathematical tools for deep learning. Inspired by Carlsson et al., this study designs topology-aware convolutional kernels that significantly improve speech recognition networks. Theoretically, by investigating orthogonal group actions on kernels, we establish a fiber-bundle decomposition of matrix spaces, enabling new filter generation methods. Practically, our proposed Orthogonal Feature (OF) layer achieves superior performance in phoneme recognition, particularly in low-noise scenarios, while demonstrating cross-domain adaptability. This work reveals TDA's potential in neural network optimization, opening new avenues for mathematics-deep learning interdisciplinary studies.
>
---
#### [new 159] Spatial RoboGrasp: Generalized Robotic Grasping Control Policy
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取控制任务，旨在解决现有方法在复杂环境（如光照、遮挡变化）下抓取泛化性和精度不足的问题。提出融合领域随机化增强、单目深度估计与6-DoF抓取提示的统一框架，并采用扩散模型策略生成动作序列，提升抓取成功率。**

- **链接: [http://arxiv.org/pdf/2505.20814v1](http://arxiv.org/pdf/2505.20814v1)**

> **作者:** Yiqi Huang; Travis Davies; Jiahuan Yan; Jiankai Sun; Xiang Chen; Luhui Hu
>
> **摘要:** Achieving generalizable and precise robotic manipulation across diverse environments remains a critical challenge, largely due to limitations in spatial perception. While prior imitation-learning approaches have made progress, their reliance on raw RGB inputs and handcrafted features often leads to overfitting and poor 3D reasoning under varied lighting, occlusion, and object conditions. In this paper, we propose a unified framework that couples robust multimodal perception with reliable grasp prediction. Our architecture fuses domain-randomized augmentation, monocular depth estimation, and a depth-aware 6-DoF Grasp Prompt into a single spatial representation for downstream action planning. Conditioned on this encoding and a high-level task prompt, our diffusion-based policy yields precise action sequences, achieving up to 40% improvement in grasp success and 45% higher task success rates under environmental variation. These results demonstrate that spatially grounded perception, paired with diffusion-based imitation learning, offers a scalable and robust solution for general-purpose robotic grasping.
>
---
#### [new 160] Learning Annotation Consensus for Continuous Emotion Recognition
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于连续情绪识别任务，旨在解决多标注者数据不一致导致信息丢失的问题。提出多标注者训练方法，通过共识网络聚合标注信息指导情绪预测，提升模型对集体输入的反映能力，在RECOLA和COGNIMUSE数据集上验证了效果。**

- **链接: [http://arxiv.org/pdf/2505.21196v1](http://arxiv.org/pdf/2505.21196v1)**

> **作者:** Ibrahim Shoer; Engin Erzin
>
> **摘要:** In affective computing, datasets often contain multiple annotations from different annotators, which may lack full agreement. Typically, these annotations are merged into a single gold standard label, potentially losing valuable inter-rater variability. We propose a multi-annotator training approach for continuous emotion recognition (CER) that seeks a consensus across all annotators rather than relying on a single reference label. Our method employs a consensus network to aggregate annotations into a unified representation, guiding the main arousal-valence predictor to better reflect collective inputs. Tested on the RECOLA and COGNIMUSE datasets, our approach outperforms traditional methods that unify annotations into a single label. This underscores the benefits of fully leveraging multi-annotator data in emotion recognition and highlights its applicability across various fields where annotations are abundant yet inconsistent.
>
---
#### [new 161] The Role of AI in Early Detection of Life-Threatening Diseases: A Retinal Imaging Perspective
- **分类: eess.IV; cs.CV**

- **简介: 该论文综述AI在视网膜成像早期检测系统性疾病的应用，解决技术分散、模型验证不足及临床整合难题。系统分析了OCT/AO成像、AI算法及移动医疗进展，量化诊断效能，并提出标准化协议和临床路径优化方案，推动精准预防与早期干预。**

- **链接: [http://arxiv.org/pdf/2505.20810v1](http://arxiv.org/pdf/2505.20810v1)**

> **作者:** Tariq M Khan; Toufique Ahmed Soomro; Imran Razzak
>
> **摘要:** Retinal imaging has emerged as a powerful, non-invasive modality for detecting and quantifying biomarkers of systemic diseases-ranging from diabetes and hypertension to Alzheimer's disease and cardiovascular disorders but current insights remain dispersed across platforms and specialties. Recent technological advances in optical coherence tomography (OCT/OCTA) and adaptive optics (AO) now deliver ultra-high-resolution scans (down to 5 {\mu}m ) with superior contrast and spatial integration, allowing early identification of microvascular abnormalities and neurodegenerative changes. At the same time, AI-driven and machine learning (ML) algorithms have revolutionized the analysis of large-scale retinal datasets, increasing sensitivity and specificity; for example, deep learning models achieve > 90 \% sensitivity for diabetic retinopathy and AUC = 0.89 for the prediction of cardiovascular risk from fundus photographs. The proliferation of mobile health technologies and telemedicine platforms further extends access, reduces costs, and facilitates community-based screening and longitudinal monitoring. Despite these breakthroughs, translation into routine practice is hindered by heterogeneous imaging protocols, limited external validation of AI models, and integration challenges within clinical workflows. In this review, we systematically synthesize the latest OCT/OCT and AO developments, AI/ML approaches, and mHealth/Tele-ophthalmology initiatives and quantify their diagnostic performance across disease domains. Finally, we propose a roadmap for multicenter protocol standardization, prospective validation trials, and seamless incorporation of retinal screening into primary and specialty care pathways-paving the way for precision prevention, early intervention, and ongoing treatment of life-threatening systemic diseases.
>
---
#### [new 162] ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出ART-DECO模型，属于文本引导的3D细节生成任务。旨在解决快速生成高质量结构可控的3D模型问题。工作包括：通过两阶段蒸馏训练，将预训练扩散模型知识迁移到轻量detailizer，实现1秒内基于粗糙形状和文本提示生成细节一致的3D资产，支持交互式编辑与创意结构输出。**

- **链接: [http://arxiv.org/pdf/2505.20431v1](http://arxiv.org/pdf/2505.20431v1)**

> **作者:** Qimin Chen; Yuezhi Yang; Yifang Wang; Vladimir G. Kim; Siddhartha Chaudhuri; Hao Zhang; Zhiqin Chen
>
> **摘要:** We introduce a 3D detailizer, a neural model which can instantaneously (in <1s) transform a coarse 3D shape proxy into a high-quality asset with detailed geometry and texture as guided by an input text prompt. Our model is trained using the text prompt, which defines the shape class and characterizes the appearance and fine-grained style of the generated details. The coarse 3D proxy, which can be easily varied and adjusted (e.g., via user editing), provides structure control over the final shape. Importantly, our detailizer is not optimized for a single shape; it is the result of distilling a generative model, so that it can be reused, without retraining, to generate any number of shapes, with varied structures, whose local details all share a consistent style and appearance. Our detailizer training utilizes a pretrained multi-view image diffusion model, with text conditioning, to distill the foundational knowledge therein into our detailizer via Score Distillation Sampling (SDS). To improve SDS and enable our detailizer architecture to learn generalizable features over complex structures, we train our model in two training stages to generate shapes with increasing structural complexity. Through extensive experiments, we show that our method generates shapes of superior quality and details compared to existing text-to-3D models under varied structure control. Our detailizer can refine a coarse shape in less than a second, making it possible to interactively author and adjust 3D shapes. Furthermore, the user-imposed structure control can lead to creative, and hence out-of-distribution, 3D asset generations that are beyond the current capabilities of leading text-to-3D generative models. We demonstrate an interactive 3D modeling workflow our method enables, and its strong generalizability over styles, structures, and object categories.
>
---
#### [new 163] UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出UI-Genie框架，针对移动GUI代理中轨迹验证困难和数据不足问题，设计了融合图像文本的奖励模型UI-Genie-RM，并构建自我改进 pipeline，通过奖励引导和动态环境验证迭代提升模型性能，生成高质量合成数据集，实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.21496v1](http://arxiv.org/pdf/2505.21496v1)**

> **作者:** Han Xiao; Guozhi Wang; Yuxiang Chai; Zimu Lu; Weifeng Lin; Hao He; Lue Fan; Liuyang Bian; Rui Hu; Liang Liu; Shuai Ren; Yafei Wen; Xiaoxin Chen; Aojun Zhou; Hongsheng Li
>
> **备注:** https://github.com/Euphoria16/UI-Genie
>
> **摘要:** In this paper, we introduce UI-Genie, a self-improving framework addressing two key challenges in GUI agents: verification of trajectory outcome is challenging and high-quality training data are not scalable. These challenges are addressed by a reward model and a self-improving pipeline, respectively. The reward model, UI-Genie-RM, features an image-text interleaved architecture that efficiently pro- cesses historical context and unifies action-level and task-level rewards. To sup- port the training of UI-Genie-RM, we develop deliberately-designed data genera- tion strategies including rule-based verification, controlled trajectory corruption, and hard negative mining. To address the second challenge, a self-improvement pipeline progressively expands solvable complex GUI tasks by enhancing both the agent and reward models through reward-guided exploration and outcome verification in dynamic environments. For training the model, we generate UI- Genie-RM-517k and UI-Genie-Agent-16k, establishing the first reward-specific dataset for GUI agents while demonstrating high-quality synthetic trajectory gen- eration without manual annotation. Experimental results show that UI-Genie achieves state-of-the-art performance across multiple GUI agent benchmarks with three generations of data-model self-improvement. We open-source our complete framework implementation and generated datasets to facilitate further research in https://github.com/Euphoria16/UI-Genie.
>
---
#### [new 164] Learning Single Index Models with Diffusion Priors
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于信号恢复任务，针对扩散模型在非线性测量模型（含不连续/未知链接函数）中的局限性，提出仅需一次无条件采样与扩散模型部分逆运算的高效方法，通过理论分析和实验验证其在半参数单指数模型重建中的更高精度与计算效率。**

- **链接: [http://arxiv.org/pdf/2505.21135v1](http://arxiv.org/pdf/2505.21135v1)**

> **作者:** Anqi Tang; Youming Chen; Shuchen Xue; Zhaoqiang Liu
>
> **备注:** ICML 2025
>
> **摘要:** Diffusion models (DMs) have demonstrated remarkable ability to generate diverse and high-quality images by efficiently modeling complex data distributions. They have also been explored as powerful generative priors for signal recovery, resulting in a substantial improvement in the quality of reconstructed signals. However, existing research on signal recovery with diffusion models either focuses on specific reconstruction problems or is unable to handle nonlinear measurement models with discontinuous or unknown link functions. In this work, we focus on using DMs to achieve accurate recovery from semi-parametric single index models, which encompass a variety of popular nonlinear models that may have {\em discontinuous} and {\em unknown} link functions. We propose an efficient reconstruction method that only requires one round of unconditional sampling and (partial) inversion of DMs. Theoretical analysis on the effectiveness of the proposed methods has been established under appropriate conditions. We perform numerical experiments on image datasets for different nonlinear measurement models. We observe that compared to competing methods, our approach can yield more accurate reconstructions while utilizing significantly fewer neural function evaluations.
>
---
#### [new 165] VoxAging: Continuously Tracking Speaker Aging with a Large-Scale Longitudinal Dataset in English and Mandarin
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 论文提出VoxAging数据集，解决说话人验证中年龄影响的难题。该数据集包含293名英、汉双语者多年每周录音（最长17年），用于研究说话人老化对系统性能的影响，分析个体老化差异及年龄、性别等因素的作用。（98字）**

- **链接: [http://arxiv.org/pdf/2505.21445v1](http://arxiv.org/pdf/2505.21445v1)**

> **作者:** Zhiqi Ai; Meixuan Bao; Zhiyong Chen; Zhi Yang; Xinnuo Li; Shugong Xu
>
> **备注:** 5 pages, 4 figures, Accepted by Interspeech 2025
>
> **摘要:** The performance of speaker verification systems is adversely affected by speaker aging. However, due to challenges in data collection, particularly the lack of sustained and large-scale longitudinal data for individuals, research on speaker aging remains difficult. In this paper, we present VoxAging, a large-scale longitudinal dataset collected from 293 speakers (226 English speakers and 67 Mandarin speakers) over several years, with the longest time span reaching 17 years (approximately 900 weeks). For each speaker, the data were recorded at weekly intervals. We studied the phenomenon of speaker aging and its effects on advanced speaker verification systems, analyzed individual speaker aging processes, and explored the impact of factors such as age group and gender on speaker aging research.
>
---
#### [new 166] Predicting Implicit Arguments in Procedural Video Instructions
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对程序式视频指令中隐式参数预测问题，提出Implicit-VidSRL多模态数据集，通过构建需结合视觉与上下文推断隐式"what/where"参数的基准任务，揭示现有模型局限并提出iSRL-Qwen2-VL方法，显著提升隐式语义角色识别效果。**

- **链接: [http://arxiv.org/pdf/2505.21068v1](http://arxiv.org/pdf/2505.21068v1)**

> **作者:** Anil Batra; Laura Sevilla-Lara; Marcus Rohrbach; Frank Keller
>
> **备注:** ACL 2025 Main
>
> **摘要:** Procedural texts help AI enhance reasoning about context and action sequences. Transforming these into Semantic Role Labeling (SRL) improves understanding of individual steps by identifying predicate-argument structure like {verb,what,where/with}. Procedural instructions are highly elliptic, for instance, (i) add cucumber to the bowl and (ii) add sliced tomatoes, the second step's where argument is inferred from the context, referring to where the cucumber was placed. Prior SRL benchmarks often miss implicit arguments, leading to incomplete understanding. To address this, we introduce Implicit-VidSRL, a dataset that necessitates inferring implicit and explicit arguments from contextual information in multimodal cooking procedures. Our proposed dataset benchmarks multimodal models' contextual reasoning, requiring entity tracking through visual changes in recipes. We study recent multimodal LLMs and reveal that they struggle to predict implicit arguments of what and where/with from multi-modal procedural data given the verb. Lastly, we propose iSRL-Qwen2-VL, which achieves a 17% relative improvement in F1-score for what-implicit and a 14.7% for where/with-implicit semantic roles over GPT-4o.
>
---
#### [new 167] Detecting Informative Channels: ActionFormer
- **分类: cs.LG; cs.CV**

- **简介: 论文提出改进ActionFormer用于传感器信号的人类活动识别，解决其高时间动态性和时空特征依赖导致的性能不足问题，通过Sequence-and-Excitation策略与swish激活函数优化模型，实验提升16.01% mAP。**

- **链接: [http://arxiv.org/pdf/2505.20739v1](http://arxiv.org/pdf/2505.20739v1)**

> **作者:** Kunpeng Zhao; Asahi Miyazaki; Tsuyoshi Okita
>
> **摘要:** Human Activity Recognition (HAR) has recently witnessed advancements with Transformer-based models. Especially, ActionFormer shows us a new perspectives for HAR in the sense that this approach gives us additional outputs which detect the border of the activities as well as the activity labels. ActionFormer was originally proposed with its input as image/video. However, this was converted to with its input as sensor signals as well. We analyze this extensively in terms of deep learning architectures. Based on the report of high temporal dynamics which limits the model's ability to capture subtle changes effectively and of the interdependencies between the spatial and temporal features. We propose the modified ActionFormer which will decrease these defects for sensor signals. The key to our approach lies in accordance with the Sequence-and-Excitation strategy to minimize the increase in additional parameters and opt for the swish activation function to retain the information about direction in the negative range. Experiments on the WEAR dataset show that our method achieves substantial improvement of a 16.01\% in terms of average mAP for inertial data.
>
---
#### [new 168] Music's Multimodal Complexity in AVQA: Why We Need More than General Multimodal LLMs
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于音乐视听问答（Music AVQA）任务，针对通用多模态模型在处理音乐连续音频-视觉内容、复杂时序动态及领域知识时的不足，通过分析现有数据集与方法，提出需专用输入处理、时空架构及音乐建模策略，并指明融合音乐先验知识的未来方向，提供相关研究资源库。**

- **链接: [http://arxiv.org/pdf/2505.20638v1](http://arxiv.org/pdf/2505.20638v1)**

> **作者:** Wenhao You; Xingjian Diao; Chunhui Zhang; Keyi Kong; Weiyi Wu; Zhongyu Ouyang; Chiyu Ma; Tingxuan Wu; Noah Wei; Zong Ke; Ming Cheng; Soroush Vosoughi; Jiang Gui
>
> **摘要:** While recent Multimodal Large Language Models exhibit impressive capabilities for general multimodal tasks, specialized domains like music necessitate tailored approaches. Music Audio-Visual Question Answering (Music AVQA) particularly underscores this, presenting unique challenges with its continuous, densely layered audio-visual content, intricate temporal dynamics, and the critical need for domain-specific knowledge. Through a systematic analysis of Music AVQA datasets and methods, this position paper identifies that specialized input processing, architectures incorporating dedicated spatial-temporal designs, and music-specific modeling strategies are critical for success in this domain. Our study provides valuable insights for researchers by highlighting effective design patterns empirically linked to strong performance, proposing concrete future directions for incorporating musical priors, and aiming to establish a robust foundation for advancing multimodal musical understanding. This work is intended to inspire broader attention and further research, supported by a continuously updated anonymous GitHub repository of relevant papers: https://github.com/xid32/Survey4MusicAVQA.
>
---
#### [new 169] SageAttention2++: A More Efficient Implementation of SageAttention2
- **分类: cs.LG; cs.AI; cs.AR; cs.CV**

- **简介: 该论文提出SageAttention2++，优化注意力机制计算效率。针对长序列下二次时间复杂度问题，在SageAttention2量化加速基础上，采用FP8 Matmul在FP16中累积运算，实现比FlashAttention快3.9倍且保持精度，适用于多种生成模型。**

- **链接: [http://arxiv.org/pdf/2505.21136v1](http://arxiv.org/pdf/2505.21136v1)**

> **作者:** Jintao Zhang; Xiaoming Xu; Jia Wei; Haofeng Huang; Pengle Zhang; Chendong Xiang; Jun Zhu; Jianfei Chen
>
> **摘要:** The efficiency of attention is critical because its time complexity grows quadratically with sequence length. SageAttention2 addresses this by utilizing quantization to accelerate matrix multiplications (Matmul) in attention. To further accelerate SageAttention2, we propose to utilize the faster instruction of FP8 Matmul accumulated in FP16. The instruction is 2x faster than the FP8 Matmul used in SageAttention2. Our experiments show that SageAttention2++ achieves a 3.9x speedup over FlashAttention while maintaining the same attention accuracy as SageAttention2. This means SageAttention2++ effectively accelerates various models, including those for language, image, and video generation, with negligible end-to-end metrics loss. The code will be available at https://github.com/thu-ml/SageAttention.
>
---
#### [new 170] PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出PreP-OCR管道，针对退化历史文档的OCR任务，通过两阶段方法提升文本提取：首先用合成数据训练图像恢复模型优化清晰度，其次基于ByT5的纠错模型修正语义错误，显著降低字符错误率（63.9-70.3%）。属于文档图像恢复与OCR优化，解决退化图像导致的高误差问题。**

- **链接: [http://arxiv.org/pdf/2505.20429v1](http://arxiv.org/pdf/2505.20429v1)**

> **作者:** Shuhao Guan; Moule Lin; Cheng Xu; Xinyi Liu; Jinman Zhao; Jiexin Fan; Qi Xu; Derek Greene
>
> **备注:** ACL 2025 main
>
> **摘要:** This paper introduces PreP-OCR, a two-stage pipeline that combines document image restoration with semantic-aware post-OCR correction to improve text extraction from degraded historical documents. Our key innovation lies in jointly optimizing image clarity and linguistic consistency. First, we generate synthetic image pairs with randomized text fonts, layouts, and degradations. An image restoration model is trained on this synthetic data, using multi-directional patch extraction and fusion to process large images. Second, a ByT5 post-corrector, fine-tuned on synthetic historical text training pairs, addresses any remaining OCR errors. Detailed experiments on 13,831 pages of real historical documents in English, French, and Spanish show that PreP-OCR pipeline reduces character error rates by 63.9-70.3\% compared to OCR on raw images. Our pipeline demonstrates the potential of integrating image restoration with linguistic error correction for digitizing historical archives.
>
---
#### [new 171] CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects
- **分类: cs.GR; cs.CV; cs.RO**

- **简介: 该论文提出CoDA框架，解决机器人与虚拟人类全身操纵关节物体的协调与精准控制问题。通过三个扩散模型分别优化身体、双手动作，结合梯度流协调及BPS统一表示提升精度，实验验证其动作质量与物理合理性优势。**

- **链接: [http://arxiv.org/pdf/2505.21437v1](http://arxiv.org/pdf/2505.21437v1)**

> **作者:** Huaijin Pi; Zhi Cen; Zhiyang Dou; Taku Komura
>
> **备注:** Project page: https://phj128.github.io/page/CoDA/index.html
>
> **摘要:** Synthesizing whole-body manipulation of articulated objects, including body motion, hand motion, and object motion, is a critical yet challenging task with broad applications in virtual humans and robotics. The core challenges are twofold. First, achieving realistic whole-body motion requires tight coordination between the hands and the rest of the body, as their movements are interdependent during manipulation. Second, articulated object manipulation typically involves high degrees of freedom and demands higher precision, often requiring the fingers to be placed at specific regions to actuate movable parts. To address these challenges, we propose a novel coordinated diffusion noise optimization framework. Specifically, we perform noise-space optimization over three specialized diffusion models for the body, left hand, and right hand, each trained on its own motion dataset to improve generalization. Coordination naturally emerges through gradient flow along the human kinematic chain, allowing the global body posture to adapt in response to hand motion objectives with high fidelity. To further enhance precision in hand-object interaction, we adopt a unified representation based on basis point sets (BPS), where end-effector positions are encoded as distances to the same BPS used for object geometry. This unified representation captures fine-grained spatial relationships between the hand and articulated object parts, and the resulting trajectories serve as targets to guide the optimization of diffusion noise, producing highly accurate interaction motion. We conduct extensive experiments demonstrating that our method outperforms existing approaches in motion quality and physical plausibility, and enables various capabilities such as object pose control, simultaneous walking and manipulation, and whole-body generation from hand-only data.
>
---
#### [new 172] efunc: An Efficient Function Representation without Neural Networks
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于函数逼近任务，旨在解决神经网络参数过多导致的效率问题。提出基于径向基函数插值的多项式表示框架，无需神经网络或复杂数据结构，并开发CUDA优化算法，实验证明在3D SDF中参数更少且性能更优。**

- **链接: [http://arxiv.org/pdf/2505.21319v1](http://arxiv.org/pdf/2505.21319v1)**

> **作者:** Biao Zhang; Peter Wonka
>
> **备注:** Project website: https://efunc.github.io/efunc/
>
> **摘要:** Function fitting/approximation plays a fundamental role in computer graphics and other engineering applications. While recent advances have explored neural networks to address this task, these methods often rely on architectures with many parameters, limiting their practical applicability. In contrast, we pursue high-quality function approximation using parameter-efficient representations that eliminate the dependency on neural networks entirely. We first propose a novel framework for continuous function modeling. Most existing works can be formulated using this framework. We then introduce a compact function representation, which is based on polynomials interpolated using radial basis functions, bypassing both neural networks and complex/hierarchical data structures. We also develop memory-efficient CUDA-optimized algorithms that reduce computational time and memory consumption to less than 10% compared to conventional automatic differentiation frameworks. Finally, we validate our representation and optimization pipeline through extensive experiments on 3D signed distance functions (SDFs). The proposed representation achieves comparable or superior performance to state-of-the-art techniques (e.g., octree/hash-grid techniques) with significantly fewer parameters.
>
---
#### [new 173] Avoid Forgetting by Preserving Global Knowledge Gradients in Federated Learning with Non-IID Data
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; cs.PF**

- **简介: 该论文针对联邦学习中非IID数据导致客户端遗忘全局决策边界的问题，提出FedProj框架。通过设计服务器端集成知识转移损失和公共未标记数据的记忆机制，保留全局知识梯度，避免本地训练时遗忘。实验显示其显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20485v1](http://arxiv.org/pdf/2505.20485v1)**

> **作者:** Abhijit Chunduru; Majid Morafah; Mahdi Morafah; Vishnu Pandi Chellapandi; Ang Li
>
> **摘要:** The inevitable presence of data heterogeneity has made federated learning very challenging. There are numerous methods to deal with this issue, such as local regularization, better model fusion techniques, and data sharing. Though effective, they lack a deep understanding of how data heterogeneity can affect the global decision boundary. In this paper, we bridge this gap by performing an experimental analysis of the learned decision boundary using a toy example. Our observations are surprising: (1) we find that the existing methods suffer from forgetting and clients forget the global decision boundary and only learn the perfect local one, and (2) this happens regardless of the initial weights, and clients forget the global decision boundary even starting from pre-trained optimal weights. In this paper, we present FedProj, a federated learning framework that robustly learns the global decision boundary and avoids its forgetting during local training. To achieve better ensemble knowledge fusion, we design a novel server-side ensemble knowledge transfer loss to further calibrate the learned global decision boundary. To alleviate the issue of learned global decision boundary forgetting, we further propose leveraging an episodic memory of average ensemble logits on a public unlabeled dataset to regulate the gradient updates at each step of local training. Experimental results demonstrate that FedProj outperforms state-of-the-art methods by a large margin.
>
---
#### [new 174] MME-Reasoning: A Comprehensive Benchmark for Logical Reasoning in MLLMs
- **分类: cs.AI; cs.CV**

- **简介: 论文提出MME-Reasoning基准，评估多模态大语言模型的逻辑推理能力。针对现有基准未分类推理类型且评估不全面的问题，该工作覆盖归纳、演绎、溯因三种推理，精选数据排除感知/知识依赖，并分析模型局限性及改进方法。**

- **链接: [http://arxiv.org/pdf/2505.21327v1](http://arxiv.org/pdf/2505.21327v1)**

> **作者:** Jiakang Yuan; Tianshuo Peng; Yilei Jiang; Yiting Lu; Renrui Zhang; Kaituo Feng; Chaoyou Fu; Tao Chen; Lei Bai; Bo Zhang; Xiangyu Yue
>
> **摘要:** Logical reasoning is a fundamental aspect of human intelligence and an essential capability for multimodal large language models (MLLMs). Despite the significant advancement in multimodal reasoning, existing benchmarks fail to comprehensively evaluate their reasoning abilities due to the lack of explicit categorization for logical reasoning types and an unclear understanding of reasoning. To address these issues, we introduce MME-Reasoning, a comprehensive benchmark designed to evaluate the reasoning ability of MLLMs, which covers all three types of reasoning (i.e., inductive, deductive, and abductive) in its questions. We carefully curate the data to ensure that each question effectively evaluates reasoning ability rather than perceptual skills or knowledge breadth, and extend the evaluation protocols to cover the evaluation of diverse questions. Our evaluation reveals substantial limitations of state-of-the-art MLLMs when subjected to holistic assessments of logical reasoning capabilities. Even the most advanced MLLMs show limited performance in comprehensive logical reasoning, with notable performance imbalances across reasoning types. In addition, we conducted an in-depth analysis of approaches such as ``thinking mode'' and Rule-based RL, which are commonly believed to enhance reasoning abilities. These findings highlight the critical limitations and performance imbalances of current MLLMs in diverse logical reasoning scenarios, providing comprehensive and systematic insights into the understanding and evaluation of reasoning capabilities.
>
---
#### [new 175] Structure from Collision
- **分类: cs.GR; cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出"结构从碰撞"(SfC)任务，通过碰撞时的外观变化重建物体内外结构。针对传统方法无法捕捉内部结构的问题，提出SfC-NeRF模型，结合物理约束与体积退火优化，有效解决内部结构估计，实验验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.21335v1](http://arxiv.org/pdf/2505.21335v1)**

> **作者:** Takuhiro Kaneko
>
> **备注:** Accepted to CVPR 2025 (Highlight). Project page: https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/sfc/
>
> **摘要:** Recent advancements in neural 3D representations, such as neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS), have enabled the accurate estimation of 3D structures from multiview images. However, this capability is limited to estimating the visible external structure, and identifying the invisible internal structure hidden behind the surface is difficult. To overcome this limitation, we address a new task called Structure from Collision (SfC), which aims to estimate the structure (including the invisible internal structure) of an object from appearance changes during collision. To solve this problem, we propose a novel model called SfC-NeRF that optimizes the invisible internal structure of an object through a video sequence under physical, appearance (i.e., visible external structure)-preserving, and keyframe constraints. In particular, to avoid falling into undesirable local optima owing to its ill-posed nature, we propose volume annealing; that is, searching for global optima by repeatedly reducing and expanding the volume. Extensive experiments on 115 objects involving diverse structures (i.e., various cavity shapes, locations, and sizes) and material properties revealed the properties of SfC and demonstrated the effectiveness of the proposed SfC-NeRF.
>
---
#### [new 176] A False Discovery Rate Control Method Using a Fully Connected Hidden Markov Random Field for Neuroimaging Data
- **分类: stat.ML; cs.CV; cs.LG; stat.ME**

- **简介: 该论文提出fcHMRF-LIS方法，解决神经影像学中多重检验的FDR控制问题，通过结合LIS检验与全连接隐马尔可夫随机场模型，有效建模空间依赖、降低FDR/FNR变异性并提升计算效率，应用于阿尔茨海默病PET数据验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.20688v1](http://arxiv.org/pdf/2505.20688v1)**

> **作者:** Taehyo Kim; Qiran Jia; Mony J. de Leon; Hai Shu
>
> **摘要:** False discovery rate (FDR) control methods are essential for voxel-wise multiple testing in neuroimaging data analysis, where hundreds of thousands or even millions of tests are conducted to detect brain regions associated with disease-related changes. Classical FDR control methods (e.g., BH, q-value, and LocalFDR) assume independence among tests and often lead to high false non-discovery rates (FNR). Although various spatial FDR control methods have been developed to improve power, they still fall short in jointly addressing three major challenges in neuroimaging applications: capturing complex spatial dependencies, maintaining low variability in both false discovery proportion (FDP) and false non-discovery proportion (FNP) across replications, and achieving computational scalability for high-resolution data. To address these challenges, we propose fcHMRF-LIS, a powerful, stable, and scalable spatial FDR control method for voxel-wise multiple testing. It integrates the local index of significance (LIS)-based testing procedure with a novel fully connected hidden Markov random field (fcHMRF) designed to model complex spatial structures using a parsimonious parameterization. We develop an efficient expectation-maximization algorithm incorporating mean-field approximation, the Conditional Random Fields as Recurrent Neural Networks (CRF-RNN) technique, and permutohedral lattice filtering, reducing the computational complexity from quadratic to linear in the number of tests. Extensive simulations demonstrate that fcHMRF-LIS achieves accurate FDR control, lower FNR, reduced variability in FDP and FNP, and a higher number of true positives compared to existing methods. Applied to an FDG-PET dataset from the Alzheimer's Disease Neuroimaging Initiative, fcHMRF-LIS identifies neurobiologically relevant brain regions and offers notable advantages in computational efficiency.
>
---
#### [new 177] Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling
- **分类: cs.LG; cs.AI; cs.CR; cs.CV; stat.ML**

- **简介: 该论文属于AI安全红队测试任务，旨在解决黑盒T2I模型安全性评估中如何绕过未知防御机制的问题。提出RPG-RT方法，通过迭代利用LLM修改提示词，并结合规则模型解析粗粒度反馈，动态适应不同安全机制，实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.21074v1](http://arxiv.org/pdf/2505.21074v1)**

> **作者:** Yichuan Cao; Yibo Miao; Xiao-Shan Gao; Yinpeng Dong
>
> **摘要:** Text-to-image (T2I) models raise ethical and safety concerns due to their potential to generate inappropriate or harmful images. Evaluating these models' security through red-teaming is vital, yet white-box approaches are limited by their need for internal access, complicating their use with closed-source models. Moreover, existing black-box methods often assume knowledge about the model's specific defense mechanisms, limiting their utility in real-world commercial API scenarios. A significant challenge is how to evade unknown and diverse defense mechanisms. To overcome this difficulty, we propose a novel Rule-based Preference modeling Guided Red-Teaming (RPG-RT), which iteratively employs LLM to modify prompts to query and leverages feedback from T2I systems for fine-tuning the LLM. RPG-RT treats the feedback from each iteration as a prior, enabling the LLM to dynamically adapt to unknown defense mechanisms. Given that the feedback is often labeled and coarse-grained, making it difficult to utilize directly, we further propose rule-based preference modeling, which employs a set of rules to evaluate desired or undesired feedback, facilitating finer-grained control over the LLM's dynamic adaptation process. Extensive experiments on nineteen T2I systems with varied safety mechanisms, three online commercial API services, and T2V models verify the superiority and practicality of our approach.
>
---
## 更新

#### [replaced 001] Training-free Stylized Text-to-Image Generation with Fast Inference
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19063v2](http://arxiv.org/pdf/2505.19063v2)**

> **作者:** Xin Ma; Yaohui Wang; Xinyuan Chen; Tien-Tsin Wong; Cunjian Chen
>
> **备注:** Project Page: https://maxin-cn.github.io/omnipainter_project
>
> **摘要:** Although diffusion models exhibit impressive generative capabilities, existing methods for stylized image generation based on these models often require textual inversion or fine-tuning with style images, which is time-consuming and limits the practical applicability of large-scale diffusion models. To address these challenges, we propose a novel stylized image generation method leveraging a pre-trained large-scale diffusion model without requiring fine-tuning or any additional optimization, termed as OmniPainter. Specifically, we exploit the self-consistency property of latent consistency models to extract the representative style statistics from reference style images to guide the stylization process. Additionally, we then introduce the norm mixture of self-attention, which enables the model to query the most relevant style patterns from these statistics for the intermediate output content features. This mechanism also ensures that the stylized results align closely with the distribution of the reference style images. Our qualitative and quantitative experimental results demonstrate that the proposed method outperforms state-of-the-art approaches.
>
---
#### [replaced 002] Cognitive Disentanglement for Referring Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11496v4](http://arxiv.org/pdf/2503.11496v4)**

> **作者:** Shaofeng Liang; Runwei Guan; Wangwang Lian; Daizong Liu; Xiaolou Sun; Dongming Wu; Yutao Yue; Weiping Ding; Hui Xiong
>
> **备注:** 27 pages, 12 figures
>
> **摘要:** As a significant application of multi-source information fusion in intelligent transportation perception systems, Referring Multi-Object Tracking (RMOT) involves localizing and tracking specific objects in video sequences based on language references. However, existing RMOT approaches often treat language descriptions as holistic embeddings and struggle to effectively integrate the rich semantic information contained in language expressions with visual features. This limitation is especially apparent in complex scenes requiring comprehensive understanding of both static object attributes and spatial motion information. In this paper, we propose a Cognitive Disentanglement for Referring Multi-Object Tracking (CDRMT) framework that addresses these challenges. It adapts the "what" and "where" pathways from the human visual processing system to RMOT tasks. Specifically, our framework first establishes cross-modal connections while preserving modality-specific characteristics. It then disentangles language descriptions and hierarchically injects them into object queries, refining object understanding from coarse to fine-grained semantic levels. Finally, we reconstruct language representations based on visual features, ensuring that tracked objects faithfully reflect the referring expression. Extensive experiments on different benchmark datasets demonstrate that CDRMT achieves substantial improvements over state-of-the-art methods, with average gains of 6.0% in HOTA score on Refer-KITTI and 3.2% on Refer-KITTI-V2. Our approach advances the state-of-the-art in RMOT while simultaneously providing new insights into multi-source information fusion.
>
---
#### [replaced 003] Lean classical-quantum hybrid neural network model for image classification
- **分类: quant-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02059v3](http://arxiv.org/pdf/2412.02059v3)**

> **作者:** Ao Liu; Cuihong Wen; Jieci Wang
>
> **备注:** 18 pages,8 figures
>
> **摘要:** The integration of algorithms from quantum information with neural networks has enabled unprecedented advancements in various domains. Nonetheless, the application of quantum machine learning algorithms for image classification predominantly relies on traditional architectures such as variational quantum circuits. The performance of these models is closely tied to the scale of their parameters, with the substantial demand for parameters potentially leading to limitations in computational resources and a significant increase in computation time. In this paper, we introduce a Lean Classical-Quantum Hybrid Neural Network (LCQHNN), which achieves efficient classification performance with only four layers of variational circuits, thereby substantially reducing computational costs. Our experiments demonstrate that LCQHNN achieves 100\%, 99.02\%, and 85.55\% classification accuracy on MNIST, FashionMNIST, and CIFAR-10 datasets. Under the same parameter conditions, the convergence speed of this method is also faster than that of traditional models. Furthermore, through visualization studies, it is found that the model effectively captures key data features during training and establishes a clear association between these features and their corresponding categories. This study confirms that the employment of quantum algorithms enhances the model's ability to handle complex classification problems.
>
---
#### [replaced 004] MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.20791v2](http://arxiv.org/pdf/2405.20791v2)**

> **作者:** Yumeng He; Yunbo Wang; Xiaokang Yang
>
> **摘要:** Out-of-distribution (OOD) 3D relighting requires novel view synthesis under unseen lighting conditions that differ significantly from the observed images. Existing relighting methods, which assume consistent light source distributions between training and testing, often degrade in OOD scenarios. We introduce MetaGS to tackle this challenge from two perspectives. First, we propose a meta-learning approach to train 3D Gaussian splatting, which explicitly promotes learning generalizable Gaussian geometries and appearance attributes across diverse lighting conditions, even with biased training data. Second, we embed fundamental physical priors from the Blinn-Phong reflection model into Gaussian splatting, which enhances the decoupling of shading components and leads to more accurate 3D scene reconstruction. Results on both synthetic and real-world datasets demonstrate the effectiveness of MetaGS in challenging OOD relighting tasks, supporting efficient point-light relighting and generalizing well to unseen environment lighting maps.
>
---
#### [replaced 005] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13055v3](http://arxiv.org/pdf/2504.13055v3)**

> **作者:** Xiangyan Liu; Jinjie Ni; Zijian Wu; Chao Du; Longxu Dou; Haonan Wang; Tianyu Pang; Michael Qizhe Shieh
>
> **备注:** Technical Report
>
> **摘要:** Recent advances in reinforcement learning (RL) have strengthened the reasoning capabilities of vision-language models (VLMs). However, enhancing policy exploration to better scale test-time compute remains largely underexplored. In addition, VLMs continue to struggle with imperfect visual perception, which in turn affects the subsequent reasoning process. To this end, we propose NoisyRollout, a simple yet effective data augmentation method that mixes trajectories from both clean and moderately distorted images during RL training. By injecting targeted diversity in visual perception and the resulting reasoning patterns, NoisyRollout promotes better policy exploration through vision-oriented inductive biases, ultimately leading to more robust reasoning behaviors. We further adopt a noise annealing schedule that gradually reduces distortion strength over training, leveraging noisy signals early on while ensuring training stability in later stages. Crucially, our method is easy-to-adopt--requiring no additional training cost and no modifications to the RL objective. Extensive experiments on $2$ distinct training datasets demonstrate that NoisyRollout achieves state-of-the-art performance among open-source RL-tuned models across $5$ out-of-domain reasoning and perception benchmarks. Furthermore, we validate the effectiveness of NoisyRollout across model sizes ($7$B and $32$B) and data scales (from $1$K to $6$K), highlighting its generalizability and scalability.
>
---
#### [replaced 006] Dynamic-I2V: Exploring Image-to-Video Generation Models via Multimodal LLM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19901v2](http://arxiv.org/pdf/2505.19901v2)**

> **作者:** Peng Liu; Xiaoming Ren; Fengkai Liu; Qingsong Xie; Quanlong Zheng; Yanhao Zhang; Haonan Lu; Yujiu Yang
>
> **摘要:** Recent advancements in image-to-video (I2V) generation have shown promising performance in conventional scenarios. However, these methods still encounter significant challenges when dealing with complex scenes that require a deep understanding of nuanced motion and intricate object-action relationships. To address these challenges, we present Dynamic-I2V, an innovative framework that integrates Multimodal Large Language Models (MLLMs) to jointly encode visual and textual conditions for a diffusion transformer (DiT) architecture. By leveraging the advanced multimodal understanding capabilities of MLLMs, our model significantly improves motion controllability and temporal coherence in synthesized videos. The inherent multimodality of Dynamic-I2V further enables flexible support for diverse conditional inputs, extending its applicability to various downstream generation tasks. Through systematic analysis, we identify a critical limitation in current I2V benchmarks: a significant bias towards favoring low-dynamic videos, stemming from an inadequate balance between motion complexity and visual quality metrics. To resolve this evaluation gap, we propose DIVE - a novel assessment benchmark specifically designed for comprehensive dynamic quality measurement in I2V generation. In conclusion, extensive quantitative and qualitative experiments confirm that Dynamic-I2V attains state-of-the-art performance in image-to-video generation, particularly revealing significant improvements of 42.5%, 7.9%, and 11.8% in dynamic range, controllability, and quality, respectively, as assessed by the DIVE metric in comparison to existing methods.
>
---
#### [replaced 007] Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02826v4](http://arxiv.org/pdf/2504.02826v4)**

> **作者:** Xiangyu Zhao; Peiyuan Zhang; Kexian Tang; Xiaorong Zhu; Hao Li; Wenhao Chai; Zicheng Zhang; Renqiu Xia; Guangtao Zhai; Junchi Yan; Hua Yang; Xue Yang; Haodong Duan
>
> **摘要:** Large Multi-modality Models (LMMs) have made significant progress in visual understanding and generation, but they still face challenges in General Visual Editing, particularly in following complex instructions, preserving appearance consistency, and supporting flexible input formats. To study this gap, we introduce RISEBench, the first benchmark for evaluating Reasoning-Informed viSual Editing (RISE). RISEBench focuses on four key reasoning categories: Temporal, Causal, Spatial, and Logical Reasoning. We curate high-quality test cases for each category and propose an robust evaluation framework that assesses Instruction Reasoning, Appearance Consistency, and Visual Plausibility with both human judges and the LMM-as-a-judge approach. We conducted experiments evaluating nine prominent visual editing models, comprising both open-source and proprietary models. The evaluation results demonstrate that current models face significant challenges in reasoning-based editing tasks. Even the most powerful model evaluated, GPT-4o-Image, achieves an accuracy of merely 28.8%. RISEBench effectively highlights the limitations of contemporary editing models, provides valuable insights, and indicates potential future directions for the field of reasoning-aware visual editing. Our code and data have been released at https://github.com/PhoenixZ810/RISEBench.
>
---
#### [replaced 008] Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training
- **分类: cs.AI; cs.CL; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14681v2](http://arxiv.org/pdf/2505.14681v2)**

> **作者:** Mengru Wang; Xingyu Chen; Yue Wang; Zhiwei He; Jiahao Xu; Tian Liang; Qiuzhi Liu; Yunzhi Yao; Wenxuan Wang; Ruotian Ma; Haitao Mi; Ningyu Zhang; Zhaopeng Tu; Xiaolong Li; Dong Yu
>
> **备注:** Work in progress
>
> **摘要:** Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models.
>
---
#### [replaced 009] EDmamba: A Simple yet Effective Event Denoising Method with State Space Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05391v2](http://arxiv.org/pdf/2505.05391v2)**

> **作者:** Ciyu Ruan; Zihang Gong; Ruishan Guo; Jingao Xu; Xinlei Chen
>
> **摘要:** Event cameras excel in high-speed vision due to their high temporal resolution, high dynamic range, and low power consumption. However, as dynamic vision sensors, their output is inherently noisy, making efficient denoising essential to preserve their ultra-low latency and real-time processing capabilities. Existing event denoising methods struggle with a critical dilemma: computationally intensive approaches compromise the sensor's high-speed advantage, while lightweight methods often lack robustness across varying noise levels. To address this, we propose a novel event denoising framework based on State Space Models (SSMs). Our approach represents events as 4D event clouds and includes a Coarse Feature Extraction (CFE) module that extracts embedding features from both geometric and polarity-aware subspaces. The model is further composed of two essential components: A Spatial Mamba (S-SSM) that models local geometric structures and a Temporal Mamba (T-SSM) that captures global temporal dynamics, efficiently propagating spatiotemporal features across events. Experiments demonstrate that our method achieves state-of-the-art accuracy and efficiency, with 88.89K parameters, 0.0685s per 100K events inference time, and a 0.982 accuracy score, outperforming Transformer-based methods by 2.08% in denoising accuracy and 36X faster.
>
---
#### [replaced 010] Exploring Disentangled and Controllable Human Image Synthesis: From End-to-End to Stage-by-Stage
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19486v2](http://arxiv.org/pdf/2503.19486v2)**

> **作者:** Zhengwentai Sun; Chenghong Li; Hongjie Liao; Xihe Yang; Keru Zheng; Heyuan Li; Yihao Zhi; Shuliang Ning; Shuguang Cui; Xiaoguang Han
>
> **摘要:** Achieving fine-grained controllability in human image synthesis is a long-standing challenge in computer vision. Existing methods primarily focus on either facial synthesis or near-frontal body generation, with limited ability to simultaneously control key factors such as viewpoint, pose, clothing, and identity in a disentangled manner. In this paper, we introduce a new disentangled and controllable human synthesis task, which explicitly separates and manipulates these four factors within a unified framework. We first develop an end-to-end generative model trained on MVHumanNet for factor disentanglement. However, the domain gap between MVHumanNet and in-the-wild data produces unsatisfactory results, motivating the exploration of virtual try-on (VTON) dataset as a potential solution. Through experiments, we observe that simply incorporating the VTON dataset as additional data to train the end-to-end model degrades performance, primarily due to the inconsistency in data forms between the two datasets, which disrupts the disentanglement process. To better leverage both datasets, we propose a stage-by-stage framework that decomposes human image generation into three sequential steps: clothed A-pose generation, back-view synthesis, and pose and view control. This structured pipeline enables better dataset utilization at different stages, significantly improving controllability and generalization, especially for in-the-wild scenarios. Extensive experiments demonstrate that our stage-by-stage approach outperforms end-to-end models in both visual fidelity and disentanglement quality, offering a scalable solution for real-world tasks. Additional demos are available on the project page: https://taited.github.io/discohuman-project/.
>
---
#### [replaced 011] Efficient Robotic Policy Learning via Latent Space Backward Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06861v2](http://arxiv.org/pdf/2505.06861v2)**

> **作者:** Dongxiu Liu; Haoyi Niu; Zhihao Wang; Jinliang Zheng; Yinan Zheng; Zhonghong Ou; Jianming Hu; Jianxiong Li; Xianyuan Zhan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Current robotic planning methods often rely on predicting multi-frame images with full pixel details. While this fine-grained approach can serve as a generic world model, it introduces two significant challenges for downstream policy learning: substantial computational costs that hinder real-time deployment, and accumulated inaccuracies that can mislead action extraction. Planning with coarse-grained subgoals partially alleviates efficiency issues. However, their forward planning schemes can still result in off-task predictions due to accumulation errors, leading to misalignment with long-term goals. This raises a critical question: Can robotic planning be both efficient and accurate enough for real-time control in long-horizon, multi-stage tasks? To address this, we propose a Latent Space Backward Planning scheme (LBP), which begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating on-task prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance. Project Page: https://lbp-authors.github.io
>
---
#### [replaced 012] Towards Training One-Step Diffusion Models Without Distillation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08005v3](http://arxiv.org/pdf/2502.08005v3)**

> **作者:** Mingtian Zhang; Wenlin Chen; Jiajun He; Zijing Ou; José Miguel Hernández-Lobato; Bernhard Schölkopf; David Barber
>
> **备注:** 21 pages, 8 figures, 3 tables, 2 algorithms
>
> **摘要:** Recent advances in training one-step diffusion models typically follow a two-stage pipeline: first training a teacher diffusion model and then distilling it into a one-step student model. This process often depends on both the teacher's score function for supervision and its weights for initializing the student model. In this paper, we explore whether one-step diffusion models can be trained directly without this distillation procedure. We introduce a family of new training methods that entirely forgo teacher score supervision, yet outperforms most teacher-guided distillation approaches. This suggests that score supervision is not essential for effective training of one-step diffusion models. However, we find that initializing the student model with the teacher's weights remains critical. Surprisingly, the key advantage of teacher initialization is not due to better latent-to-output mappings, but rather the rich set of feature representations across different noise levels that the teacher diffusion model provides. These insights take us one step closer towards training one-step diffusion models without distillation and provide a better understanding of the roles of teacher supervision and initialization in the distillation process.
>
---
#### [replaced 013] RemoteSAM: Towards Segment Anything for Earth Observation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18022v2](http://arxiv.org/pdf/2505.18022v2)**

> **作者:** Liang Yao; Fan Liu; Delong Chen; Chuanyi Zhang; Yijun Wang; Ziyun Chen; Wei Xu; Shimin Di; Yuhui Zheng
>
> **摘要:** We aim to develop a robust yet flexible visual foundation model for Earth observation. It should possess strong capabilities in recognizing and localizing diverse visual targets while providing compatibility with various input-output interfaces required across different task scenarios. Current systems cannot meet these requirements, as they typically utilize task-specific architecture trained on narrow data domains with limited semantic coverage. Our study addresses these limitations from two aspects: data and modeling. We first introduce an automatic data engine that enjoys significantly better scalability compared to previous human annotation or rule-based approaches. It has enabled us to create the largest dataset of its kind to date, comprising 270K image-text-mask triplets covering an unprecedented range of diverse semantic categories and attribute specifications. Based on this data foundation, we further propose a task unification paradigm that centers around referring expression segmentation. It effectively handles a wide range of vision-centric perception tasks, including classification, detection, segmentation, grounding, etc, using a single model without any task-specific heads. Combining these innovations on data and modeling, we present RemoteSAM, a foundation model that establishes new SoTA on several earth observation perception benchmarks, outperforming other foundation models such as Falcon, GeoChat, and LHRS-Bot with significantly higher efficiency. Models and data are publicly available at https://github.com/1e12Leon/RemoteSAM.
>
---
#### [replaced 014] Corrupted but Not Broken: Understanding and Mitigating the Negative Impacts of Corrupted Data in Visual Instruction Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12635v3](http://arxiv.org/pdf/2502.12635v3)**

> **作者:** Yunhao Gou; Hansi Yang; Zhili Liu; Kai Chen; Yihan Zeng; Lanqing Hong; Zhenguo Li; Qun Liu; Bo Han; James T. Kwok; Yu Zhang
>
> **摘要:** Visual Instruction Tuning (VIT) aims to enhance Multimodal Large Language Models (MLLMs), yet its effectiveness is often compromised by corrupted datasets with issues such as hallucinated content, incorrect responses, and poor OCR quality. Previous approaches to address these challenges have focused on refining datasets through high-quality data collection or rule-based filtering that can be costly or limited in scope. In this paper, we conduct a systematic investigation into the impact of corrupted data on MLLMs and discover that, although corrupted data degrade model performance, such adverse effects are largely reversible, and MLLMs are {\bf corrupted but not broken}. Specifically, we find that disabling a small subset of parameters can almost fully restore performance. Moreover, corrupted MLLMs inherently possess the capability to differentiate between clean and corrupted samples, facilitating dataset cleaning without external intervention. Building on these insights, we introduce a corruption-robust training paradigm that significantly surpasses existing strategies for mitigating the effects of corrupted data.
>
---
#### [replaced 015] ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13107v2](http://arxiv.org/pdf/2503.13107v2)**

> **作者:** Hao Yin; Guangzong Si; Zilei Wang
>
> **摘要:** Contrastive decoding strategies are widely used to mitigate object hallucinations in multimodal large language models (MLLMs). By reducing over-reliance on language priors, these strategies ensure that generated content remains closely grounded in visual inputs, producing contextually accurate outputs. Since contrastive decoding requires no additional training or external tools, it offers both computational efficiency and versatility, making it highly attractive. However, these methods present two main limitations: (1) bluntly suppressing language priors can compromise coherence and accuracy of generated content, and (2) processing contrastive inputs adds computational load, significantly slowing inference speed. To address these challenges, we propose Visual Amplification Fusion (VAF), a plug-and-play technique that enhances attention to visual signals within the model's middle layers, where modality fusion predominantly occurs. This approach enables more effective capture of visual features, reducing the model's bias toward language modality. Experimental results demonstrate that VAF significantly reduces hallucinations across various MLLMs without affecting inference speed, while maintaining coherence and accuracy in generated outputs.
>
---
#### [replaced 016] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.09403v4](http://arxiv.org/pdf/2410.09403v4)**

> **作者:** Haoyang Su; Renqi Chen; Shixiang Tang; Zhenfei Yin; Xinzhe Zheng; Jinzhe Li; Biqing Qi; Qi Wu; Hui Li; Wanli Ouyang; Philip Torr; Bowen Zhou; Nanqing Dong
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** The rapid advancement of scientific progress requires innovative tools that can accelerate knowledge discovery. Although recent AI methods, particularly large language models (LLMs), have shown promise in tasks such as hypothesis generation and experimental design, they fall short of replicating the collaborative nature of real-world scientific practices, where diverse experts work together in teams to tackle complex problems. To address the limitations, we propose an LLM-based multi-agent system, i.e., Virtual Scientists (VirSci), designed to mimic the teamwork inherent in scientific research. VirSci organizes a team of agents to collaboratively generate, evaluate, and refine research ideas. Through comprehensive experiments, we demonstrate that this multi-agent approach outperforms the state-of-the-art method in producing novel scientific ideas. We further investigate the collaboration mechanisms that contribute to its tendency to produce ideas with higher novelty, offering valuable insights to guide future research and illuminating pathways toward building a robust system for autonomous scientific discovery. The code is available at https://github.com/open-sciencelab/Virtual-Scientists.
>
---
#### [replaced 017] Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.02857v4](http://arxiv.org/pdf/2503.02857v4)**

> **作者:** Nuria Alina Chandra; Ryan Murtfeldt; Lin Qiu; Arnab Karmakar; Hannah Lee; Emmanuel Tanumihardja; Kevin Farhat; Ben Caffee; Sejin Paik; Changyeon Lee; Jongwook Choi; Aerin Kim; Oren Etzioni
>
> **摘要:** In the age of increasingly realistic generative AI, robust deepfake detection is essential for mitigating fraud and disinformation. While many deepfake detectors report high accuracy on academic datasets, we show that these academic benchmarks are out of date and not representative of real-world deepfakes. We introduce Deepfake-Eval-2024, a new deepfake detection benchmark consisting of in-the-wild deepfakes collected from social media and deepfake detection platform users in 2024. Deepfake-Eval-2024 consists of 45 hours of videos, 56.5 hours of audio, and 1,975 images, encompassing the latest manipulation technologies. The benchmark contains diverse media content from 88 different websites in 52 different languages. We find that the performance of open-source state-of-the-art deepfake detection models drops precipitously when evaluated on Deepfake-Eval-2024, with AUC decreasing by 50% for video, 48% for audio, and 45% for image models compared to previous benchmarks. We also evaluate commercial deepfake detection models and models finetuned on Deepfake-Eval-2024, and find that they have superior performance to off-the-shelf open-source models, but do not yet reach the accuracy of deepfake forensic analysts. The dataset is available at https://github.com/nuriachandra/Deepfake-Eval-2024.
>
---
#### [replaced 018] Exploring Out-of-distribution Detection for Sparse-view Computed Tomography with Diffusion Models
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.06308v2](http://arxiv.org/pdf/2411.06308v2)**

> **作者:** Ezgi Demircan-Tureyen; Felix Lucka; Tristan van Leeuwen
>
> **摘要:** Recent works demonstrate the effectiveness of diffusion models as unsupervised solvers for inverse imaging problems. Sparse-view computed tomography (CT) has greatly benefited from these advancements, achieving improved generalization without reliance on measurement parameters. However, this comes at the cost of potential hallucinations, especially when handling out-of-distribution (OOD) data. To ensure reliability, it is essential to study OOD detection for CT reconstruction across both clinical and industrial applications. This need further extends to enabling the OOD detector to function effectively as an anomaly inspection tool. In this paper, we explore the use of a diffusion model, trained to capture the target distribution for CT reconstruction, as an in-distribution prior. Building on recent research, we employ the model to reconstruct partially diffused input images and assess OOD-ness through multiple reconstruction errors. Adapting this approach for sparse-view CT requires redefining the notions of ``input'' and ``reconstruction error''. Here, we use filtered backprojection (FBP) reconstructions as input and investigate various definitions of reconstruction error. Our proof-of-concept experiments on the MNIST dataset highlight both successes and failures, demonstrating the potential and limitations of integrating such an OOD detector into a CT reconstruction system. Our findings suggest that effective OOD detection can be achieved by comparing measurements with forward-projected reconstructions, provided that reconstructions from noisy FBP inputs are conditioned on the measurements. However, conditioning can sometimes lead the OOD detector to inadvertently reconstruct OOD images well. To counter this, we introduce a weighting approach that improves robustness against highly informative OOD measurements, albeit with a trade-off in performance in certain cases.
>
---
#### [replaced 019] WildDoc: How Far Are We from Achieving Comprehensive and Robust Document Understanding in the Wild?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11015v2](http://arxiv.org/pdf/2505.11015v2)**

> **作者:** An-Lan Wang; Jingqun Tang; Liao Lei; Hao Feng; Qi Liu; Xiang Fei; Jinghui Lu; Han Wang; Weiwei Liu; Hao Liu; Yuliang Liu; Xiang Bai; Can Huang
>
> **摘要:** The rapid advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced capabilities in Document Understanding. However, prevailing benchmarks like DocVQA and ChartQA predominantly comprise \textit{scanned or digital} documents, inadequately reflecting the intricate challenges posed by diverse real-world scenarios, such as variable illumination and physical distortions. This paper introduces WildDoc, the inaugural benchmark designed specifically for assessing document understanding in natural environments. WildDoc incorporates a diverse set of manually captured document images reflecting real-world conditions and leverages document sources from established benchmarks to facilitate comprehensive comparisons with digital or scanned documents. Further, to rigorously evaluate model robustness, each document is captured four times under different conditions. Evaluations of state-of-the-art MLLMs on WildDoc expose substantial performance declines and underscore the models' inadequate robustness compared to traditional benchmarks, highlighting the unique challenges posed by real-world document understanding. Our project homepage is available at https://bytedance.github.io/WildDoc.
>
---
#### [replaced 020] SuperAD: A Training-free Anomaly Classification and Segmentation Method for CVPR 2025 VAND 3.0 Workshop Challenge Track 1: Adapt & Detect
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19750v2](http://arxiv.org/pdf/2505.19750v2)**

> **作者:** Huaiyuan Zhang; Hang Chen; Yu Cheng; Shunyi Wu; Linghao Sun; Linao Han; Zeyu Shi; Lei Qi
>
> **摘要:** In this technical report, we present our solution to the CVPR 2025 Visual Anomaly and Novelty Detection (VAND) 3.0 Workshop Challenge Track 1: Adapt & Detect: Robust Anomaly Detection in Real-World Applications. In real-world industrial anomaly detection, it is crucial to accurately identify anomalies with physical complexity, such as transparent or reflective surfaces, occlusions, and low-contrast contaminations. The recently proposed MVTec AD 2 dataset significantly narrows the gap between publicly available benchmarks and anomalies found in real-world industrial environments. To address the challenges posed by this dataset--such as complex and varying lighting conditions and real anomalies with large scale differences--we propose a fully training-free anomaly detection and segmentation method based on feature extraction using the DINOv2 model named SuperAD. Our method carefully selects a small number of normal reference images and constructs a memory bank by leveraging the strong representational power of DINOv2. Anomalies are then segmented by performing nearest neighbor matching between test image features and the memory bank. Our method achieves competitive results on both test sets of the MVTec AD 2 dataset.
>
---
#### [replaced 021] EmoNet-Face: An Expert-Annotated Benchmark for Synthetic Emotion Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20033v2](http://arxiv.org/pdf/2505.20033v2)**

> **作者:** Christoph Schuhmann; Robert Kaczmarczyk; Gollam Rabby; Felix Friedrich; Maurice Kraus; Krishna Kalyan; Kourosh Nadi; Huu Nguyen; Kristian Kersting; Sören Auer
>
> **摘要:** Effective human-AI interaction relies on AI's ability to accurately perceive and interpret human emotions. Current benchmarks for vision and vision-language models are severely limited, offering a narrow emotional spectrum that overlooks nuanced states (e.g., bitterness, intoxication) and fails to distinguish subtle differences between related feelings (e.g., shame vs. embarrassment). Existing datasets also often use uncontrolled imagery with occluded faces and lack demographic diversity, risking significant bias. To address these critical gaps, we introduce EmoNet Face, a comprehensive benchmark suite. EmoNet Face features: (1) A novel 40-category emotion taxonomy, meticulously derived from foundational research to capture finer details of human emotional experiences. (2) Three large-scale, AI-generated datasets (EmoNet HQ, Binary, and Big) with explicit, full-face expressions and controlled demographic balance across ethnicity, age, and gender. (3) Rigorous, multi-expert annotations for training and high-fidelity evaluation. (4) We built EmpathicInsight-Face, a model achieving human-expert-level performance on our benchmark. The publicly released EmoNet Face suite - taxonomy, datasets, and model - provides a robust foundation for developing and evaluating AI systems with a deeper understanding of human emotions.
>
---
#### [replaced 022] Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18600v2](http://arxiv.org/pdf/2505.18600v2)**

> **作者:** Bryan Sangwoo Kim; Jeongsol Kim; Jong Chul Ye
>
> **备注:** Project Page: https://bryanswkim.github.io/chain-of-zoom/
>
> **摘要:** Modern single-image super-resolution (SISR) models deliver photo-realistic results at the scale factors on which they are trained, but collapse when asked to magnify far beyond that regime. We address this scalability bottleneck with Chain-of-Zoom (CoZ), a model-agnostic framework that factorizes SISR into an autoregressive chain of intermediate scale-states with multi-scale-aware prompts. CoZ repeatedly re-uses a backbone SR model, decomposing the conditional probability into tractable sub-problems to achieve extreme resolutions without additional training. Because visual cues diminish at high magnifications, we augment each zoom step with multi-scale-aware text prompts generated by a vision-language model (VLM). The prompt extractor itself is fine-tuned using Generalized Reward Policy Optimization (GRPO) with a critic VLM, aligning text guidance towards human preference. Experiments show that a standard 4x diffusion SR model wrapped in CoZ attains beyond 256x enlargement with high perceptual quality and fidelity. Project Page: https://bryanswkim.github.io/chain-of-zoom/ .
>
---
#### [replaced 023] SC-Pro: Training-Free Framework for Defending Unsafe Image Synthesis Attack
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.05359v2](http://arxiv.org/pdf/2501.05359v2)**

> **作者:** Junha Park; Jaehui Hwang; Ian Ryu; Hyungkeun Park; Jiyoon Kim; Jong-Seok Lee
>
> **摘要:** With advances in diffusion models, image generation has shown significant performance improvements. This raises concerns about the potential abuse of image generation, such as the creation of explicit or violent images, commonly referred to as Not Safe For Work (NSFW) content. To address this, the Stable Diffusion model includes several safety checkers to censor initial text prompts and final output images generated from the model. However, recent research has shown that these safety checkers have vulnerabilities against adversarial attacks, allowing them to generate NSFW images. In this paper, we find that these adversarial attacks are not robust to small changes in text prompts or input latents. Based on this, we propose SC-Pro (Spherical or Circular Probing), a training-free framework that easily defends against adversarial attacks generating NSFW images. Moreover, we develop an approach that utilizes one-step diffusion models for efficient NSFW detection (SC-Pro-o), further reducing computational resources. We demonstrate the superiority of our method in terms of performance and applicability.
>
---
#### [replaced 024] BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation
- **分类: astro-ph.GA; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08528v3](http://arxiv.org/pdf/2502.08528v3)**

> **作者:** Ao liu; Zelin Zhang; Songbai Chen; Cuihong Wen; Jieci Wang
>
> **备注:** 20 pages, 10figures
>
> **摘要:** The properties of black holes and accretion flows can be inferred by fitting Event Horizon Telescope (EHT) data to simulated images generated through general relativistic ray tracing (GRRT). However, due to the computationally intensive nature of GRRT, the efficiency of generating specific radiation flux images needs to be improved. This paper introduces the Branch Correction Denoising Diffusion Model (BCDDM), a deep learning framework that synthesizes black hole images directly from physical parameters. The model incorporates a branch correction mechanism and a weighted mixed loss function to enhance accuracy and stability. We have constructed a dataset of 2,157 GRRT-simulated images for training the BCDDM, which spans seven key physical parameters of the radiatively inefficient accretion flow (RIAF) model. Our experiments show a strong correlation between the generated images and their physical parameters. By enhancing the GRRT dataset with BCDDM-generated images and using ResNet50 for parameter regression, we achieve significant improvements in parameter prediction performance. BCDDM offers a novel approach to reducing the computational costs of black hole image generation, providing a faster and more efficient pathway for dataset augmentation, parameter estimation, and model fitting.
>
---
#### [replaced 025] TAPIP3D: Tracking Any Point in Persistent 3D Geometry
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.14717v2](http://arxiv.org/pdf/2504.14717v2)**

> **作者:** Bowei Zhang; Lei Ke; Adam W. Harley; Katerina Fragkiadaki
>
> **备注:** Long-term feed-forward 3D point tracking in persistent 3D point maps. Code:https://github.com/zbw001/TAPIP3D
>
> **摘要:** We introduce TAPIP3D, a novel approach for long-term 3D point tracking in monocular RGB and RGB-D videos. TAPIP3D represents videos as camera-stabilized spatio-temporal feature clouds, leveraging depth and camera motion information to lift 2D video features into a 3D world space where camera movement is effectively canceled out. Within this stabilized 3D representation, TAPIP3D iteratively refines multi-frame motion estimates, enabling robust point tracking over long time horizons. To handle the irregular structure of 3D point distributions, we propose a 3D Neighborhood-to-Neighborhood (N2N) attention mechanism - a 3D-aware contextualization strategy that builds informative, spatially coherent feature neighborhoods to support precise trajectory estimation. Our 3D-centric formulation significantly improves performance over existing 3D point tracking methods and even surpasses state-of-the-art 2D pixel trackers in accuracy when reliable depth is available. The model supports inference in both camera-centric (unstabilized) and world-centric (stabilized) coordinates, with experiments showing that compensating for camera motion leads to substantial gains in tracking robustness. By replacing the conventional 2D square correlation windows used in prior 2D and 3D trackers with a spatially grounded 3D attention mechanism, TAPIP3D achieves strong and consistent results across multiple 3D point tracking benchmarks. Project Page: https://tapip3d.github.io
>
---
#### [replaced 026] Quantum autoencoders for image classification
- **分类: quant-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15254v2](http://arxiv.org/pdf/2502.15254v2)**

> **作者:** Hinako Asaoka; Kazue Kudo
>
> **摘要:** Classical machine learning often struggles with complex, high-dimensional data. Quantum machine learning offers a potential solution, promising more efficient processing. The quantum convolutional neural network (QCNN), a hybrid algorithm, fits current noisy intermediate-scale quantum hardware. However, its training depends largely on classical computation. Future gate-based quantum computers may realize full quantum advantages. In contrast to QCNNs, quantum autoencoders (QAEs) leverage classical optimization solely for parameter tuning. Data compression and reconstruction are handled entirely within quantum circuits, enabling purely quantum-based feature extraction. This study introduces a novel image-classification approach using QAEs, achieving classification without requiring additional qubits compared with conventional QAE implementations. The quantum circuit structure significantly impacts classification accuracy. Unlike hybrid methods such as QCNN, QAE-based classification emphasizes quantum computation. Our experiments demonstrate high accuracy in a four-class classification task, evaluating various quantum-gate configurations to understand the impact of different parameterized quantum circuit structures on classification performance. Specifically, noise-free conditions are considered, and simulations are performed using a statevector simulator to model the quantum system with full amplitude precision. Our results reveal that specific ansatz structures achieve superior accuracy. Moreover, the proposed approach achieves performance comparable to that of conventional machine-learning methods while significantly reducing the number of parameters requiring optimization. These findings indicate that QAEs can serve as efficient classification models with fewer parameters and highlight the potential of utilizing quantum circuits for complete end-to-end learning.
>
---
#### [replaced 027] Panoramic Distortion-Aware Tokenization for Person Detection and Localization Using Transformers in Overhead Fisheye Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.14228v2](http://arxiv.org/pdf/2503.14228v2)**

> **作者:** Nobuhiko Wakai; Satoshi Sato; Yasunori Ishii; Takayoshi Yamashita
>
> **摘要:** Person detection methods are used widely in applications including visual surveillance, pedestrian detection, and robotics. However, accurate detection of persons from overhead fisheye images remains an open challenge because of factors including person rotation and small-sized persons. To address the person rotation problem, we convert the fisheye images into panoramic images. For smaller people, we focused on the geometry of the panoramas. Conventional detection methods tend to focus on larger people because these larger people yield large significant areas for feature maps. In equirectangular panoramic images, we find that a person's height decreases linearly near the top of the images. Using this finding, we leverage the significance values and aggregate tokens that are sorted based on these values to balance the significant areas. In this leveraging process, we introduce panoramic distortion-aware tokenization. This tokenization procedure divides a panoramic image using self-similarity figures that enable determination of optimal divisions without gaps, and we leverage the maximum significant values in each tile of token groups to preserve the significant areas of smaller people. To achieve higher detection accuracy, we propose a person detection and localization method that combines panoramic-image remapping and the tokenization procedure. Extensive experiments demonstrated that our method outperforms conventional methods when applied to large-scale datasets.
>
---
#### [replaced 028] Bringing Objects to Life: training-free 4D generation from 3D objects through view consistent noise
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20422v2](http://arxiv.org/pdf/2412.20422v2)**

> **作者:** Ohad Rahamim; Ori Malca; Dvir Samuel; Gal Chechik
>
> **摘要:** Recent advancements in generative models have enabled the creation of dynamic 4D content - 3D objects in motion - based on text prompts, which holds potential for applications in virtual worlds, media, and gaming. Existing methods provide control over the appearance of generated content, including the ability to animate 3D objects. However, their ability to generate dynamics is limited to the mesh datasets they were trained on, lacking any growth or structural development capability. In this work, we introduce a training-free method for animating 3D objects by conditioning on textual prompts to guide 4D generation, enabling custom general scenes while maintaining the original object's identity. We first convert a 3D mesh into a static 4D Neural Radiance Field (NeRF) that preserves the object's visual attributes. Then, we animate the object using an Image-to-Video diffusion model driven by text. To improve motion realism, we introduce a view-consistent noising protocol that aligns object perspectives with the noising process to promote lifelike movement, and a masked Score Distillation Sampling (SDS) loss that leverages attention maps to focus optimization on relevant regions, better preserving the original object. We evaluate our model on two different 3D object datasets for temporal coherence, prompt adherence, and visual fidelity, and find that our method outperforms the baseline based on multiview training, achieving better consistency with the textual prompt in hard scenarios.
>
---
#### [replaced 029] Efficient LiDAR Reflectance Compression via Scanning Serialization
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.09433v2](http://arxiv.org/pdf/2505.09433v2)**

> **作者:** Jiahao Zhu; Kang You; Dandan Ding; Zhan Ma
>
> **摘要:** Reflectance attributes in LiDAR point clouds provide essential information for downstream tasks but remain underexplored in neural compression methods. To address this, we introduce SerLiC, a serialization-based neural compression framework to fully exploit the intrinsic characteristics of LiDAR reflectance. SerLiC first transforms 3D LiDAR point clouds into 1D sequences via scan-order serialization, offering a device-centric perspective for reflectance analysis. Each point is then tokenized into a contextual representation comprising its sensor scanning index, radial distance, and prior reflectance, for effective dependencies exploration. For efficient sequential modeling, Mamba is incorporated with a dual parallelization scheme, enabling simultaneous autoregressive dependency capture and fast processing. Extensive experiments demonstrate that SerLiC attains over 2x volume reduction against the original reflectance data, outperforming the state-of-the-art method by up to 22% reduction of compressed bits while using only 2% of its parameters. Moreover, a lightweight version of SerLiC achieves > 10 fps (frames per second) with just 111K parameters, which is attractive for real-world applications.
>
---
#### [replaced 030] Auto-nnU-Net: Towards Automated Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16561v3](http://arxiv.org/pdf/2505.16561v3)**

> **作者:** Jannis Becktepe; Leona Hennig; Steffen Oeltze-Jafra; Marius Lindauer
>
> **备注:** 31 pages, 19 figures. Accepted for publication at AutoML 2025
>
> **摘要:** Medical Image Segmentation (MIS) includes diverse tasks, from bone to organ segmentation, each with its own challenges in finding the best segmentation model. The state-of-the-art AutoML-related MIS-framework nnU-Net automates many aspects of model configuration but remains constrained by fixed hyperparameters and heuristic design choices. As a full-AutoML framework for MIS, we propose Auto-nnU-Net, a novel nnU-Net variant enabling hyperparameter optimization (HPO), neural architecture search (NAS), and hierarchical NAS (HNAS). Additionally, we propose Regularized PriorBand to balance model accuracy with the computational resources required for training, addressing the resource constraints often faced in real-world medical settings that limit the feasibility of extensive training procedures. We evaluate our approach across diverse MIS datasets from the well-established Medical Segmentation Decathlon, analyzing the impact of AutoML techniques on segmentation performance, computational efficiency, and model design choices. The results demonstrate that our AutoML approach substantially improves the segmentation performance of nnU-Net on 6 out of 10 datasets and is on par on the other datasets while maintaining practical resource requirements. Our code is available at https://github.com/automl/AutoNNUnet.
>
---
#### [replaced 031] Denoising Mutual Knowledge Distillation in Bi-Directional Multiple Instance Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12074v2](http://arxiv.org/pdf/2505.12074v2)**

> **作者:** Chen Shu; Boyu Fu; Yiman Li; Ting Yin; Wenchuan Zhang; Jie Chen; Yuhao Yi; Hong Bu
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Multiple Instance Learning is the predominant method for Whole Slide Image classification in digital pathology, enabling the use of slide-level labels to supervise model training. Although MIL eliminates the tedious fine-grained annotation process for supervised learning, whether it can learn accurate bag- and instance-level classifiers remains a question. To address the issue, instance-level classifiers and instance masks were incorporated to ground the prediction on supporting patches. These methods, while practically improving the performance of MIL methods, may potentially introduce noisy labels. We propose to bridge the gap between commonly used MIL and fully supervised learning by augmenting both the bag- and instance-level learning processes with pseudo-label correction capabilities elicited from weak to strong generalization techniques. The proposed algorithm improves the performance of dual-level MIL algorithms on both bag- and instance-level predictions. Experiments on public pathology datasets showcase the advantage of the proposed methods.
>
---
#### [replaced 032] ACT-R: Adaptive Camera Trajectories for Single View 3D Reconstruction
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08239v2](http://arxiv.org/pdf/2505.08239v2)**

> **作者:** Yizhi Wang; Mingrui Zhao; Ali Mahdavi-Amiri; Hao Zhang
>
> **摘要:** We introduce the simple idea of adaptive view planning to multi-view synthesis, aiming to improve both occlusion revelation and 3D consistency for single-view 3D reconstruction. Instead of producing an unordered set of views independently or simultaneously, we generate a sequence of views, leveraging temporal consistency to enhance 3D coherence. More importantly, our view sequence is not determined by a pre-determined and fixed camera setup. Instead, we compute an adaptive camera trajectory (ACT), forming an orbit, which seeks to maximize the visibility of occluded regions of the 3D object to be reconstructed. Once the best orbit is found, we feed it to a video diffusion model to generate novel views around the orbit, which can then be passed to any multi-view 3D reconstruction model to obtain the final result. Our multi-view synthesis pipeline is quite efficient since it involves no run-time training/optimization, only forward inferences by applying pre-trained models for occlusion analysis and multi-view synthesis. Our method predicts camera trajectories that reveal occlusions effectively and produce consistent novel views, significantly improving 3D reconstruction over SOTA alternatives on the unseen GSO dataset.
>
---
#### [replaced 033] Shaping a Stabilized Video by Mitigating Unintended Changes for Concept-Augmented Video Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.12526v2](http://arxiv.org/pdf/2410.12526v2)**

> **作者:** Mingce Guo; Jingxuan He; Shengeng Tang; Zhangye Wang; Lechao Cheng
>
> **摘要:** Text-driven video editing utilizing generative diffusion models has garnered significant attention due to their potential applications. However, existing approaches are constrained by the limited word embeddings provided in pre-training, which hinders nuanced editing targeting open concepts with specific attributes. Directly altering the keywords in target prompts often results in unintended disruptions to the attention mechanisms. To achieve more flexible editing easily, this work proposes an improved concept-augmented video editing approach that generates diverse and stable target videos flexibly by devising abstract conceptual pairs. Specifically, the framework involves concept-augmented textual inversion and a dual prior supervision mechanism. The former enables plug-and-play guidance of stable diffusion for video editing, effectively capturing target attributes for more stylized results. The dual prior supervision mechanism significantly enhances video stability and fidelity. Comprehensive evaluations demonstrate that our approach generates more stable and lifelike videos, outperforming state-of-the-art methods.
>
---
#### [replaced 034] UOD: Unseen Object Detection in 3D Point Cloud
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.03846v2](http://arxiv.org/pdf/2401.03846v2)**

> **作者:** Hyunjun Choi; Daeho Um; Hawook Jeong
>
> **备注:** Under review
>
> **摘要:** Existing 3D object detectors encounter extreme challenges in localizing unseen 3D objects and recognizing them as unseen, which is a crucial technology in autonomous driving in the wild. To address these challenges, we propose practical methods to enhance the performance of 3D detection and Out-Of-Distribution (OOD) classification for unseen objects. The proposed methods include anomaly sample augmentation, learning of universal objectness, learning of detecting unseen objects, and learning of distinguishing unseen objects. To demonstrate the effectiveness of our approach, we propose the KITTI Misc benchmark and two additional synthetic OOD benchmarks: the Nuscenes OOD benchmark and the SUN-RGBD OOD benchmark. The proposed methods consistently enhance performance by a large margin across all existing methods, giving insight for future work on unseen 3D object detection in the wild.
>
---
#### [replaced 035] PUSSM: Point Cloud Upsampling as Implicit Statistical Shape Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16716v3](http://arxiv.org/pdf/2501.16716v3)**

> **作者:** Tongxu Zhang; Bei Wang
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** This paper proposes a framework for high-fidelity reconstruction of pelvic structures by integrating medical image segmentation and point cloud upsampling. By point cloud upsampling to learn shape priors from MedShapePelvic without requiring landmarks or PCA, our method functions as an implicit statistical shape model. Evaluations on Pelvic1k show significant improvements in surface quality and anatomical accuracy. This approach is generalizable and applicable to other skeletal regions.
>
---
#### [replaced 036] Recent Advances in Diffusion Models for Hyperspectral Image Processing and Analysis: A Review
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11158v2](http://arxiv.org/pdf/2505.11158v2)**

> **作者:** Xing Hu; Xiangcheng Liu; Danfeng Hong; Qianqian Duan; Linghua Jiang; Haima Yang; Dawei Zhan
>
> **摘要:** Hyperspectral image processing and analysis has important application value in remote sensing, agriculture and environmental monitoring, but its high dimensionality, data redundancy and noise interference etc. bring great challenges to the analysis. Traditional models have limitations in dealing with these complex data, and it is difficult to meet the increasing demand for analysis. In recent years, Diffusion models, as a class of emerging generative approaches, have demonstrated promising capabilities in hyperspectral image (HSI) processing tasks. By simulating the diffusion process of data in time, the Diffusion Model are capable of modeling high-dimensional spectral structures, generate high-quality samples, and achieve competitive performance in spectral-spatial denoising tasks and data enhancement. In this paper, we review the recent research advances in diffusion modeling for hyperspectral image processing and analysis, and discuss its applications in tasks such as high-dimensional data processing, noise removal, classification, and anomaly detection. The performance of diffusion-based models on image processing is compared and the challenges are summarized. It is shown that the diffusion model can significantly improve the accuracy and efficiency of hyperspectral image analysis, providing a new direction for future research.
>
---
#### [replaced 037] HWA-UNETR: Hierarchical Window Aggregate UNETR for 3D Multimodal Gastric Lesion Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10464v3](http://arxiv.org/pdf/2505.10464v3)**

> **作者:** Jiaming Liang; Lihuan Dai; Xiaoqi Sheng; Xiangguang Chen; Chun Yao; Guihua Tao; Qibin Leng; Hongmin Cai; Xi Zhong
>
> **备注:** This work has been provisionally accepted for MICCAI 2025
>
> **摘要:** Multimodal medical image segmentation faces significant challenges in the context of gastric cancer lesion analysis. This clinical context is defined by the scarcity of independent multimodal datasets and the imperative to amalgamate inherently misaligned modalities. As a result, algorithms are constrained to train on approximate data and depend on application migration, leading to substantial resource expenditure and a potential decline in analysis accuracy. To address those challenges, we have made two major contributions: First, we publicly disseminate the GCM 2025 dataset, which serves as the first large-scale, open-source collection of gastric cancer multimodal MRI scans, featuring professionally annotated FS-T2W, CE-T1W, and ADC images from 500 patients. Second, we introduce HWA-UNETR, a novel 3D segmentation framework that employs an original HWA block with learnable window aggregation layers to establish dynamic feature correspondences between different modalities' anatomical structures, and leverages the innovative tri-orientated fusion mamba mechanism for context modeling and capturing long-range spatial dependencies. Extensive experiments on our GCM 2025 dataset and the publicly BraTS 2021 dataset validate the performance of our framework, demonstrating that the new approach surpasses existing methods by up to 1.68\% in the Dice score while maintaining solid robustness. The dataset and code are public via https://github.com/JeMing-creater/HWA-UNETR.
>
---
#### [replaced 038] PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16023v2](http://arxiv.org/pdf/2504.16023v2)**

> **作者:** Song Wang; Xiaolu Liu; Lingdong Kong; Jianyun Xu; Chunyong Hu; Gongfan Fang; Wentong Li; Jianke Zhu; Xinchao Wang
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Self-supervised representation learning for point cloud has demonstrated effectiveness in improving pre-trained model performance across diverse tasks. However, as pre-trained models grow in complexity, fully fine-tuning them for downstream applications demands substantial computational and storage resources. Parameter-efficient fine-tuning (PEFT) methods offer a promising solution to mitigate these resource requirements, yet most current approaches rely on complex adapter and prompt mechanisms that increase tunable parameters. In this paper, we propose PointLoRA, a simple yet effective method that combines low-rank adaptation (LoRA) with multi-scale token selection to efficiently fine-tune point cloud models. Our approach embeds LoRA layers within the most parameter-intensive components of point cloud transformers, reducing the need for tunable parameters while enhancing global feature capture. Additionally, multi-scale token selection extracts critical local information to serve as prompts for downstream fine-tuning, effectively complementing the global context captured by LoRA. The experimental results across various pre-trained models and three challenging public datasets demonstrate that our approach achieves competitive performance with only 3.43% of the trainable parameters, making it highly effective for resource-constrained applications. Source code is available at: https://github.com/songw-zju/PointLoRA.
>
---
#### [replaced 039] Multiple Different Black Box Explanations for Image Classifiers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2309.14309v4](http://arxiv.org/pdf/2309.14309v4)**

> **作者:** Hana Chockler; David A. Kelly; Daniel Kroening
>
> **摘要:** Existing explanation tools for image classifiers usually give only a single explanation for an image's classification. For many images, however, image classifiers accept more than one explanation for the image label. These explanations are useful for analyzing the decision process of the classifier and for detecting errors. Thus, restricting the number of explanations to just one severely limits insight into the behavior of the classifier. In this paper, we describe an algorithm and a tool, MultEX, for computing multiple explanations as the output of a black-box image classifier for a given image. Our algorithm uses a principled approach based on actual causality. We analyze its theoretical complexity and evaluate MultEX against the state-of-the-art across three different models and three different datasets. We find that MultEX finds more explanations and that these explanations are of higher quality.
>
---
#### [replaced 040] From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22976v5](http://arxiv.org/pdf/2503.22976v5)**

> **作者:** Jiahui Zhang; Yurui Chen; Yanpeng Zhou; Yueming Xu; Ze Huang; Jilin Mei; Junhui Chen; Yu-Jie Yuan; Xinyue Cai; Guowei Huang; Xingyue Quan; Hang Xu; Li Zhang
>
> **备注:** Project page: https://fudan-zvg.github.io/spar
>
> **摘要:** Recent advances in LVLMs have improved vision-language understanding, but they still struggle with spatial perception, limiting their ability to reason about complex 3D scenes. Unlike previous approaches that incorporate 3D representations into models to improve spatial understanding, we aim to unlock the potential of VLMs by leveraging spatially relevant image data. To this end, we introduce a novel 2D spatial data generation and annotation pipeline built upon scene data with 3D ground-truth. This pipeline enables the creation of a diverse set of spatial tasks, ranging from basic perception tasks to more complex reasoning tasks. Leveraging this pipeline, we construct SPAR-7M, a large-scale dataset generated from thousands of scenes across multiple public datasets. In addition, we introduce SPAR-Bench, a benchmark designed to offer a more comprehensive evaluation of spatial capabilities compared to existing spatial benchmarks, supporting both single-view and multi-view inputs. Training on both SPAR-7M and large-scale 2D datasets enables our models to achieve state-of-the-art performance on 2D spatial benchmarks. Further fine-tuning on 3D task-specific datasets yields competitive results, underscoring the effectiveness of our dataset in enhancing spatial reasoning.
>
---
#### [replaced 041] CLEVRER-Humans: Describing Physical and Causal Events the Human Way
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2310.03635v2](http://arxiv.org/pdf/2310.03635v2)**

> **作者:** Jiayuan Mao; Xuelin Yang; Xikun Zhang; Noah D. Goodman; Jiajun Wu
>
> **备注:** Version 3. NeurIPS 2022 (Dataset and Benchmark Track). First two authors contributed equally. Project page: https://sites.google.com/stanford.edu/clevrer-humans/home
>
> **摘要:** Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models. We convert the collected CEGs into questions and answers to be consistent with prior work. Finally, we study a collection of baseline approaches for CLEVRER-Humans question-answering, highlighting the great challenges set forth by our benchmark.
>
---
#### [replaced 042] How Do Multimodal Large Language Models Handle Complex Multimodal Reasoning? Placing Them in An Extensible Escape Game
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10042v3](http://arxiv.org/pdf/2503.10042v3)**

> **作者:** Ziyue Wang; Yurui Dong; Fuwen Luo; Minyuan Ruan; Zhili Cheng; Chi Chen; Peng Li; Yang Liu
>
> **摘要:** The rapid advancing of Multimodal Large Language Models (MLLMs) has spurred interest in complex multimodal reasoning tasks in the real-world and virtual environment, which require coordinating multiple abilities, including visual perception, visual reasoning, spatial awareness, and target deduction. However, existing evaluations primarily assess the final task completion, often degrading assessments to isolated abilities such as visual grounding and visual question answering. Less attention is given to comprehensively and quantitatively analyzing reasoning process in multimodal environments, which is crucial for understanding model behaviors and underlying reasoning mechanisms beyond merely task success. To address this, we introduce MM-Escape, an extensible benchmark for investigating multimodal reasoning, inspired by real-world escape games. MM-Escape emphasizes intermediate model behaviors alongside final task completion. To achieve this, we develop EscapeCraft, a customizable and open environment that enables models to engage in free-form exploration for assessing multimodal reasoning. Extensive experiments show that MLLMs, regardless of scale, can successfully complete the simplest room escape tasks, with some exhibiting human-like exploration strategies. Yet, performance dramatically drops as task difficulty increases. Moreover, we observe that performance bottlenecks vary across models, revealing distinct failure modes and limitations in their multimodal reasoning abilities, such as repetitive trajectories without adaptive exploration, getting stuck in corners due to poor visual spatial awareness, and ineffective use of acquired props, such as the key. We hope our work sheds light on new challenges in multimodal reasoning, and uncovers potential improvements in MLLMs capabilities.
>
---
#### [replaced 043] When Are Concepts Erased From Diffusion Models?
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17013v3](http://arxiv.org/pdf/2505.17013v3)**

> **作者:** Kevin Lu; Nicky Kriplani; Rohit Gandikota; Minh Pham; David Bau; Chinmay Hegde; Niv Cohen
>
> **备注:** Project Page: https://nyu-dice-lab.github.io/when-are-concepts-erased/
>
> **摘要:** Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.
>
---
#### [replaced 044] ZeroPur: Succinct Training-Free Adversarial Purification
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2406.03143v3](http://arxiv.org/pdf/2406.03143v3)**

> **作者:** Erhu Liu; Zonglin Yang; Bo Liu; Bin Xiao; Xiuli Bi
>
> **备注:** 17 pages, 7 figures, under review
>
> **摘要:** Adversarial purification is a kind of defense technique that can defend against various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold, and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.
>
---
#### [replaced 045] SHARDeg: A Benchmark for Skeletal Human Action Recognition in Degraded Scenarios
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18048v2](http://arxiv.org/pdf/2505.18048v2)**

> **作者:** Simon Malzard; Nitish Mital; Richard Walters; Victoria Nockles; Raghuveer Rao; Celso M. De Melo
>
> **备注:** 19 pages, 2 images, updated acknowledgements versus previous versions to be compliant with funders
>
> **摘要:** Computer vision (CV) models for detection, prediction or classification tasks operate on video data-streams that are often degraded in the real world, due to deployment in real-time or on resource-constrained hardware. It is therefore critical that these models are robust to degraded data, but state of the art (SoTA) models are often insufficiently assessed with these real-world constraints in mind. This is exemplified by Skeletal Human Action Recognition (SHAR), which is critical in many CV pipelines operating in real-time and at the edge, but robustness to degraded data has previously only been shallowly and inconsistently assessed. Here we address this issue for SHAR by providing an important first data degradation benchmark on the most detailed and largest 3D open dataset, NTU-RGB+D-120, and assess the robustness of five leading SHAR models to three forms of degradation that represent real-world issues. We demonstrate the need for this benchmark by showing that the form of degradation, which has not previously been considered, has a large impact on model accuracy; at the same effective frame rate, model accuracy can vary by >40% depending on degradation type. We also identify that temporal regularity of frames in degraded SHAR data is likely a major driver of differences in model performance, and harness this to improve performance of existing models by up to >40%, through employing a simple mitigation approach based on interpolation. Finally, we highlight how our benchmark has helped identify an important degradation-resistant SHAR model based in Rough Path Theory; the LogSigRNN SHAR model outperforms the SoTA DeGCN model in five out of six cases at low frame rates by an average accuracy of 6%, despite trailing the SoTA model by 11-12% on un-degraded data at high frame rates (30 FPS).
>
---
#### [replaced 046] CDPDNet: Integrating Text Guidance with Hybrid Vision Encoders for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18958v2](http://arxiv.org/pdf/2505.18958v2)**

> **作者:** Jiong Wu; Yang Xing; Boxiao Yu; Wei Shao; Kuang Gong
>
> **摘要:** Most publicly available medical segmentation datasets are only partially labeled, with annotations provided for a subset of anatomical structures. When multiple datasets are combined for training, this incomplete annotation poses challenges, as it limits the model's ability to learn shared anatomical representations among datasets. Furthermore, vision-only frameworks often fail to capture complex anatomical relationships and task-specific distinctions, leading to reduced segmentation accuracy and poor generalizability to unseen datasets. In this study, we proposed a novel CLIP-DINO Prompt-Driven Segmentation Network (CDPDNet), which combined a self-supervised vision transformer with CLIP-based text embedding and introduced task-specific text prompts to tackle these challenges. Specifically, the framework was constructed upon a convolutional neural network (CNN) and incorporated DINOv2 to extract both fine-grained and global visual features, which were then fused using a multi-head cross-attention module to overcome the limited long-range modeling capability of CNNs. In addition, CLIP-derived text embeddings were projected into the visual space to help model complex relationships among organs and tumors. To further address the partial label challenge and enhance inter-task discriminative capability, a Text-based Task Prompt Generation (TTPG) module that generated task-specific prompts was designed to guide the segmentation. Extensive experiments on multiple medical imaging datasets demonstrated that CDPDNet consistently outperformed existing state-of-the-art segmentation methods. Code and pretrained model are available at: https://github.com/wujiong-hub/CDPDNet.git.
>
---
#### [replaced 047] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10610v2](http://arxiv.org/pdf/2505.10610v2)**

> **作者:** Zhaowei Wang; Wenhao Yu; Xiyu Ren; Jipeng Zhang; Yu Zhao; Rohit Saxena; Liang Cheng; Ginny Wong; Simon See; Pasquale Minervini; Yangqiu Song; Mark Steedman
>
> **备注:** Work in progress
>
> **摘要:** The rapid extension of context windows in large vision-language models has given rise to long-context vision-language models (LCVLMs), which are capable of handling hundreds of images with interleaved text tokens in a single forward pass. In this work, we introduce MMLongBench, the first benchmark covering a diverse set of long-context vision-language tasks, to evaluate LCVLMs effectively and thoroughly. MMLongBench is composed of 13,331 examples spanning five different categories of downstream tasks, such as Visual RAG and Many-Shot ICL. It also provides broad coverage of image types, including various natural and synthetic images. To assess the robustness of the models to different input lengths, all examples are delivered at five standardized input lengths (8K-128K tokens) via a cross-modal tokenization scheme that combines vision patches and text tokens. Through a thorough benchmarking of 46 closed-source and open-source LCVLMs, we provide a comprehensive analysis of the current models' vision-language long-context ability. Our results show that: i) performance on a single task is a weak proxy for overall long-context capability; ii) both closed-source and open-source models face challenges in long-context vision-language tasks, indicating substantial room for future improvement; iii) models with stronger reasoning ability tend to exhibit better long-context performance. By offering wide task coverage, various image types, and rigorous length control, MMLongBench provides the missing foundation for diagnosing and advancing the next generation of LCVLMs.
>
---
#### [replaced 048] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17982v2](http://arxiv.org/pdf/2505.17982v2)**

> **作者:** Bryan Wong; Jong Woo Kim; Huazhu Fu; Mun Yong Yi
>
> **摘要:** Vision-language models (VLMs) have recently been integrated into multiple instance learning (MIL) frameworks to address the challenge of few-shot, weakly supervised classification of whole slide images (WSIs). A key trend involves leveraging multi-scale information to better represent hierarchical tissue structures. However, existing methods often face two key limitations: (1) insufficient modeling of interactions within the same modalities across scales (e.g., 5x and 20x) and (2) inadequate alignment between visual and textual modalities on the same scale. To address these gaps, we propose HiVE-MIL, a hierarchical vision-language framework that constructs a unified graph consisting of (1) parent-child links between coarse (5x) and fine (20x) visual/textual nodes to capture hierarchical relationships, and (2) heterogeneous intra-scale edges linking visual and textual nodes on the same scale. To further enhance semantic consistency, HiVE-MIL incorporates a two-stage, text-guided dynamic filtering mechanism that removes weakly correlated patch-text pairs, and introduces a hierarchical contrastive loss to align textual semantics across scales. Extensive experiments on TCGA breast, lung, and kidney cancer datasets demonstrate that HiVE-MIL consistently outperforms both traditional MIL and recent VLM-based MIL approaches, achieving gains of up to 4.1% in macro F1 under 16-shot settings. Our results demonstrate the value of jointly modeling hierarchical structure and multimodal alignment for efficient and scalable learning from limited pathology data. The code is available at https://github.com/bryanwong17/HiVE-MIL
>
---
#### [replaced 049] Selftok: Discrete Visual Tokens of Autoregression, by Diffusion, and for Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07538v3](http://arxiv.org/pdf/2505.07538v3)**

> **作者:** Bohan Wang; Zhongqi Yue; Fengda Zhang; Shuo Chen; Li'an Bi; Junzhe Zhang; Xue Song; Kennard Yanting Chan; Jiachun Pan; Weijia Wu; Mingze Zhou; Wang Lin; Kaihang Pan; Saining Zhang; Liyu Jia; Wentao Hu; Wei Zhao; Hanwang Zhang
>
> **摘要:** We completely discard the conventional spatial prior in image representation and introduce a novel discrete visual tokenizer: Self-consistency Tokenizer (Selftok). At its design core, we compose an autoregressive (AR) prior -- mirroring the causal structure of language -- into visual tokens by using the reverse diffusion process of image generation. The AR property makes Selftok fundamentally distinct from traditional spatial tokens in the following two key ways: - Selftok offers an elegant and minimalist approach to unify diffusion and AR for vision-language models (VLMs): By representing images with Selftok tokens, we can train a VLM using a purely discrete autoregressive architecture -- like that in LLMs -- without requiring additional modules or training objectives. - We theoretically show that the AR prior satisfies the Bellman equation, whereas the spatial prior does not. Therefore, Selftok supports reinforcement learning (RL) for visual generation with effectiveness comparable to that achieved in LLMs. Besides the AR property, Selftok is also a SoTA tokenizer that achieves a favorable trade-off between high-quality reconstruction and compression rate. We use Selftok to build a pure AR VLM for both visual comprehension and generation tasks. Impressively, without using any text-image training pairs, a simple policy gradient RL working in the visual tokens can significantly boost the visual generation benchmark, surpassing all the existing models by a large margin. Therefore, we believe that Selftok effectively addresses the long-standing challenge that visual tokens cannot support effective RL. When combined with the well-established strengths of RL in LLMs, this brings us one step closer to realizing a truly multimodal LLM. Project Page: https://selftok-team.github.io/report/.
>
---
#### [replaced 050] SAIL: Self-supervised Albedo Estimation from Real Images with a Latent Diffusion Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19751v2](http://arxiv.org/pdf/2505.19751v2)**

> **作者:** Hala Djeghim; Nathan Piasco; Luis Roldão; Moussab Bennehar; Dzmitry Tsishkou; Céline Loscos; Désiré Sidibé
>
> **备注:** Project page: https://hala-djeghim.github.io/SAIL/
>
> **摘要:** Intrinsic image decomposition aims at separating an image into its underlying albedo and shading components, isolating the base color from lighting effects to enable downstream applications such as virtual relighting and scene editing. Despite the rise and success of learning-based approaches, intrinsic image decomposition from real-world images remains a significant challenging task due to the scarcity of labeled ground-truth data. Most existing solutions rely on synthetic data as supervised setups, limiting their ability to generalize to real-world scenes. Self-supervised methods, on the other hand, often produce albedo maps that contain reflections and lack consistency under different lighting conditions. To address this, we propose SAIL, an approach designed to estimate albedo-like representations from single-view real-world images. We repurpose the prior knowledge of a latent diffusion model for unconditioned scene relighting as a surrogate objective for albedo estimation. To extract the albedo, we introduce a novel intrinsic image decomposition fully formulated in the latent space. To guide the training of our latent diffusion model, we introduce regularization terms that constrain both the lighting-dependent and independent components of our latent image decomposition. SAIL predicts stable albedo under varying lighting conditions and generalizes to multiple scenes, using only unlabeled multi-illumination data available online.
>
---
#### [replaced 051] Parameter Efficient Continual Learning with Dynamic Low-Rank Adaptation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11998v2](http://arxiv.org/pdf/2505.11998v2)**

> **作者:** Prashant Shivaram Bhat; Shakib Yazdani; Elahe Arani; Bahram Zonooz
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Catastrophic forgetting has remained a critical challenge for deep neural networks in Continual Learning (CL) as it undermines consolidated knowledge when learning new tasks. Parameter efficient fine tuning CL techniques are gaining traction for their effectiveness in addressing catastrophic forgetting with a lightweight training schedule while avoiding degradation of consolidated knowledge in pre-trained models. However, low rank adapters (LoRA) in these approaches are highly sensitive to rank selection which can lead to sub-optimal resource allocation and performance. To this end, we introduce PEARL, a rehearsal-free CL framework that entails dynamic rank allocation for LoRA components during CL training. Specifically, PEARL leverages reference task weights and adaptively determines the rank of task-specific LoRA components based on the current tasks' proximity to reference task weights in parameter space. To demonstrate the versatility of PEARL, we evaluate it across three vision architectures (ResNet, Separable Convolutional Network and Vision Transformer) and a multitude of CL scenarios, and show that PEARL outperforms all considered baselines by a large margin.
>
---
#### [replaced 052] Multi-Granularity Class Prototype Topology Distillation for Class-Incremental Source-Free Unsupervised Domain Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16064v4](http://arxiv.org/pdf/2411.16064v4)**

> **作者:** Peihua Deng; Jiehua Zhang; Xichun Sheng; Chenggang Yan; Yaoqi Sun; Ying Fu; Liang Li
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** This paper explores the Class-Incremental Source-Free Unsupervised Domain Adaptation (CI-SFUDA) problem, where the unlabeled target data come incrementally without access to labeled source instances. This problem poses two challenges, the interference of similar source-class knowledge in target-class representation learning and the shocks of new target knowledge to old ones. To address them, we propose the Multi-Granularity Class Prototype Topology Distillation (GROTO) algorithm, which effectively transfers the source knowledge to the class-incremental target domain. Concretely, we design the multi-granularity class prototype self-organization module and the prototype topology distillation module. First, we mine the positive classes by modeling accumulation distributions. Next, we introduce multi-granularity class prototypes to generate reliable pseudo-labels, and exploit them to promote the positive-class target feature self-organization. Second, the positive-class prototypes are leveraged to construct the topological structures of source and target feature spaces. Then, we perform the topology distillation to continually mitigate the shocks of new target knowledge to old ones. Extensive experiments demonstrate that our proposed method achieves state-of-the-art performance on three public datasets. Code is available at https://github.com/dengpeihua/GROTO.
>
---
#### [replaced 053] UltraBones100k: A reliable automated labeling method and large-scale dataset for ultrasound-based bone surface extraction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03783v3](http://arxiv.org/pdf/2502.03783v3)**

> **作者:** Luohong Wu; Nicola A. Cavalcanti; Matthias Seibold; Giuseppe Loggia; Lisa Reissner; Jonas Hein; Silvan Beeler; Arnd Viehöfer; Stephan Wirth; Lilian Calvet; Philipp Fürnstahl
>
> **摘要:** Ultrasound-based bone surface segmentation is crucial in computer-assisted orthopedic surgery. However, ultrasound images have limitations, including a low signal-to-noise ratio, and acoustic shadowing, which make interpretation difficult. Existing deep learning models for bone segmentation rely primarily on costly manual labeling by experts, limiting dataset size and model generalizability. Additionally, the complexity of ultrasound physics and acoustic shadow makes the images difficult for humans to interpret, leading to incomplete labels in anechoic regions and limiting model performance. To advance ultrasound bone segmentation and establish effective model benchmarks, larger and higher-quality datasets are needed. We propose a methodology for collecting ex-vivo ultrasound datasets with automatically generated bone labels, including anechoic regions. The proposed labels are derived by accurately superimposing tracked bone CT models onto the tracked ultrasound images. These initial labels are refined to account for ultrasound physics. A clinical evaluation is conducted by an expert physician specialized on orthopedic sonography to assess the quality of the generated bone labels. A neural network for bone segmentation is trained on the collected dataset and its predictions are compared to expert manual labels, evaluating accuracy, completeness, and F1-score. We collected the largest known dataset of 100k ultrasound images of human lower limbs with bone labels, called UltraBones100k. A Wilcoxon signed-rank test with Bonferroni correction confirmed that the bone alignment after our method significantly improved the quality of bone labeling (p < 0.001). The model trained on UltraBones100k consistently outperforms manual labeling in all metrics, particularly in low-intensity regions (320% improvement in completeness at a distance threshold of 0.5 mm).
>
---
#### [replaced 054] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12363v2](http://arxiv.org/pdf/2505.12363v2)**

> **作者:** Qi Feng
>
> **备注:** 26 pages, 19 figures, 4 tables. Code, models, and datasets are available at our project page: https://github.com/nkkbr/ViCA. This is a draft technical report. At the request of Professor Hidetoshi Shimodaira, his name has been removed from the author list
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [replaced 055] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12312v2](http://arxiv.org/pdf/2505.12312v2)**

> **作者:** Qi Feng
>
> **备注:** 31 pages, 10 figures, 6 tables. The implementation and fine-tuned model (ViCA-7B), along with detailed documentation, are publicly available at https://huggingface.co/nkkbr/ViCA. This is a draft technical report. At Professor Hidetoshi Shimodaira's request, his name has been removed from the author list
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [replaced 056] Modality Curation: Building Universal Embeddings for Advanced Multimodal Information Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.19650v2](http://arxiv.org/pdf/2505.19650v2)**

> **作者:** Fanheng Kong; Jingyuan Zhang; Yahui Liu; Hongzhi Zhang; Shi Feng; Xiaocui Yang; Daling Wang; Yu Tian; Victoria W.; Fuzheng Zhang; Guorui Zhou
>
> **备注:** 26 pages, project page: https://friedrichor.github.io/projects/UNITE
>
> **摘要:** Multimodal information retrieval (MIR) faces inherent challenges due to the heterogeneity of data sources and the complexity of cross-modal alignment. While previous studies have identified modal gaps in feature spaces, a systematic approach to address these challenges remains unexplored. In this work, we introduce UNITE, a universal framework that tackles these challenges through two critical yet underexplored aspects: data curation and modality-aware training configurations. Our work provides the first comprehensive analysis of how modality-specific data properties influence downstream task performance across diverse scenarios. Moreover, we propose Modal-Aware Masked Contrastive Learning (MAMCL) to mitigate the competitive relationships among the instances of different modalities. Our framework achieves state-of-the-art results on multiple multimodal retrieval benchmarks, outperforming existing methods by notable margins. Through extensive experiments, we demonstrate that strategic modality curation and tailored training protocols are pivotal for robust cross-modal representation learning. This work not only advances MIR performance but also provides a foundational blueprint for future research in multimodal systems. Our project is available at https://friedrichor.github.io/projects/UNITE.
>
---
#### [replaced 057] Minimal Interaction Separated Tuning: A New Paradigm for Visual Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.17559v3](http://arxiv.org/pdf/2406.17559v3)**

> **作者:** Ningyuan Tang; Minghao Fu; Jianxin Wu
>
> **摘要:** The rapid scaling of large vision pretrained models makes fine-tuning tasks more and more difficult on devices with low computational resources. We explore a new visual adaptation paradigm called separated tuning, which treats large pretrained models as standalone feature extractors that run on powerful cloud servers. The fine-tuning carries out on devices which possess only low computational resources (slow CPU, no GPU, small memory, etc.) Existing methods that are potentially suitable for our separated tuning paradigm are discussed. But, three major drawbacks hinder their application in separated tuning: low adaptation capability, large adapter network, and in particular, high information transfer overhead. To address these issues, we propose Minimal Interaction Separated Tuning, or MIST, which reveals that the sum of intermediate features from pretrained models not only has minimal information transfer but also has high adaptation capability. With a lightweight attention-based adaptor network, MIST achieves information transfer efficiency, parameter efficiency, computational and memory efficiency, and at the same time demonstrates competitive results on various visual adaptation benchmarks.
>
---
#### [replaced 058] Can Large Language Models Understand Symbolic Graphics Programs?
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.08313v4](http://arxiv.org/pdf/2408.08313v4)**

> **作者:** Zeju Qiu; Weiyang Liu; Haiwen Feng; Zhen Liu; Tim Z. Xiao; Katherine M. Collins; Joshua B. Tenenbaum; Adrian Weller; Michael J. Black; Bernhard Schölkopf
>
> **备注:** ICLR 2025 Spotlight (v4: 47 pages, 26 figures, project page: https://sgp-bench.github.io/)
>
> **摘要:** Against the backdrop of enthusiasm for large language models (LLMs), there is a growing need to scientifically assess their capabilities and shortcomings. This is nontrivial in part because it is difficult to find tasks which the models have not encountered during training. Utilizing symbolic graphics programs, we propose a domain well-suited to test multiple spatial-semantic reasoning skills of LLMs. Popular in computer graphics, these programs procedurally generate visual data. While LLMs exhibit impressive skills in general program synthesis and analysis, symbolic graphics programs offer a new layer of evaluation: they allow us to test an LLM's ability to answer semantic questions about the images or 3D geometries without a vision encoder. To semantically understand the symbolic programs, LLMs would need to possess the ability to "imagine" and reason how the corresponding graphics content would look with only the symbolic description of the local curvatures and strokes. We use this task to evaluate LLMs by creating a large benchmark for the semantic visual understanding of symbolic graphics programs, built procedurally with minimal human effort. Particular emphasis is placed on transformations of images that leave the image level semantics invariant while introducing significant changes to the underlying program. We evaluate commercial and open-source LLMs on our benchmark to assess their ability to reason about visual output of programs, finding that LLMs considered stronger at reasoning generally perform better. Lastly, we introduce a novel method to improve this ability -- Symbolic Instruction Tuning (SIT), in which the LLM is finetuned with pre-collected instruction data on symbolic graphics programs. Interestingly, we find that SIT not only improves LLM's understanding on symbolic programs, but it also improves general reasoning ability on various other benchmarks.
>
---
#### [replaced 059] Towards Generalized Proactive Defense against Face Swapping with Contour-Hybrid Watermark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19081v2](http://arxiv.org/pdf/2505.19081v2)**

> **作者:** Ruiyang Xia; Dawei Zhou; Decheng Liu; Lin Yuan; Jie Li; Nannan Wang; Xinbo Gao
>
> **备注:** 16 pages, 11 figures, under review
>
> **摘要:** Face swapping, recognized as a privacy and security concern, has prompted considerable defensive research. With the advancements in AI-generated content, the discrepancies between the real and swapped faces have become nuanced. Considering the difficulty of forged traces detection, we shift the focus to the face swapping purpose and proactively embed elaborate watermarks against unknown face swapping techniques. Given that the constant purpose is to swap the original face identity while preserving the background, we concentrate on the regions surrounding the face to ensure robust watermark generation, while embedding the contour texture and face identity information to achieve progressive image determination. The watermark is located in the facial contour and contains hybrid messages, dubbed the contour-hybrid watermark (CMark). Our approach generalizes face swapping detection without requiring any swapping techniques during training and the storage of large-scale messages in advance. Experiments conducted across 8 face swapping techniques demonstrate the superiority of our approach compared with state-of-the-art passive and proactive detectors while achieving a favorable balance between the image quality and watermark robustness.
>
---
#### [replaced 060] Semantic Correspondence: Unified Benchmarking and a Strong Baseline
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18060v3](http://arxiv.org/pdf/2505.18060v3)**

> **作者:** Kaiyan Zhang; Xinghui Li; Jingyi Lu; Kai Han
>
> **摘要:** Establishing semantic correspondence is a challenging task in computer vision, aiming to match keypoints with the same semantic information across different images. Benefiting from the rapid development of deep learning, remarkable progress has been made over the past decade. However, a comprehensive review and analysis of this task remains absent. In this paper, we present the first extensive survey of semantic correspondence methods. We first propose a taxonomy to classify existing methods based on the type of their method designs. These methods are then categorized accordingly, and we provide a detailed analysis of each approach. Furthermore, we aggregate and summarize the results of methods in literature across various benchmarks into a unified comparative table, with detailed configurations to highlight performance variations. Additionally, to provide a detailed understanding on existing methods for semantic matching, we thoroughly conduct controlled experiments to analyse the effectiveness of the components of different methods. Finally, we propose a simple yet effective baseline that achieves state-of-the-art performance on multiple benchmarks, providing a solid foundation for future research in this field. We hope this survey serves as a comprehensive reference and consolidated baseline for future development. Code is publicly available at: https://github.com/Visual-AI/Semantic-Correspondence.
>
---
#### [replaced 061] Regularized Personalization of Text-to-Image Diffusion Models without Distributional Drift
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19519v2](http://arxiv.org/pdf/2505.19519v2)**

> **作者:** Gihoon Kim; Hyungjin Park; Taesup Kim
>
> **摘要:** Personalization using text-to-image diffusion models involves adapting a pretrained model to novel subjects with only a few image examples. This task presents a fundamental challenge, as the model must not only learn the new subject effectively but also preserve its ability to generate diverse and coherent outputs across a wide range of prompts. In other words, successful personalization requires integrating new concepts without forgetting previously learned generative capabilities. Forgetting denotes unintended distributional drift, where the model's output distribution deviates from that of the original pretrained model. In this paper, we provide an analysis of this issue and identify a mismatch between standard training objectives and the goals of personalization. To address this, we propose a new training objective based on a Lipschitz-bounded formulation that explicitly constrains deviation from the pretrained distribution. Our method provides improved control over distributional drift and performs well even in data-scarce scenarios. Experimental results demonstrate that our approach consistently outperforms existing personalization methods, achieving higher CLIP-T, CLIP-I, and DINO scores.
>
---
#### [replaced 062] SoftPQ: Robust Instance Segmentation Evaluation via Soft Matching and Tunable Thresholds
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12155v2](http://arxiv.org/pdf/2505.12155v2)**

> **作者:** Ranit Karmakar; Simon F. Nørrelykke
>
> **摘要:** Segmentation evaluation metrics traditionally rely on binary decision logic: predictions are either correct or incorrect, based on rigid IoU thresholds. Detection--based metrics such as F1 and mAP determine correctness at the object level using fixed overlap cutoffs, while overlap--based metrics like Intersection over Union (IoU) and Dice operate at the pixel level, often overlooking instance--level structure. Panoptic Quality (PQ) attempts to unify detection and segmentation assessment, but it remains dependent on hard-threshold matching--treating predictions below the threshold as entirely incorrect. This binary framing obscures important distinctions between qualitatively different errors and fails to reward gradual model improvements. We propose SoftPQ, a flexible and interpretable instance segmentation metric that redefines evaluation as a graded continuum rather than a binary classification. SoftPQ introduces tunable upper and lower IoU thresholds to define a partial matching region and applies a sublinear penalty function to ambiguous or fragmented predictions. These extensions allow SoftPQ to exhibit smoother score behavior, greater robustness to structural segmentation errors, and more informative feedback for model development and evaluation. Through controlled perturbation experiments, we show that SoftPQ captures meaningful differences in segmentation quality that existing metrics overlook, making it a practical and principled alternative for both benchmarking and iterative model refinement.
>
---
#### [replaced 063] PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.09866v3](http://arxiv.org/pdf/2312.09866v3)**

> **作者:** Tianchen Deng; Guole Shen; Tong Qin; Jianyu Wang; Wentao Zhao; Jingchuan Wang; Danwei Wang; Weidong Chen
>
> **备注:** Accepted by CVPR 2024
>
> **摘要:** Neural implicit scene representations have recently shown encouraging results in dense visual SLAM. However, existing methods produce low-quality scene reconstruction and low-accuracy localization performance when scaling up to large indoor scenes and long sequences. These limitations are mainly due to their single, global radiance field with finite capacity, which does not adapt to large scenarios. Their end-to-end pose networks are also not robust enough with the growth of cumulative errors in large scenes. To this end, we introduce PLGSLAM, a neural visual SLAM system capable of high-fidelity surface reconstruction and robust camera tracking in real-time. To handle large-scale indoor scenes, PLGSLAM proposes a progressive scene representation method which dynamically allocates new local scene representation trained with frames within a local sliding window. This allows us to scale up to larger indoor scenes and improves robustness (even under pose drifts). In local scene representation, PLGSLAM utilizes tri-planes for local high-frequency features with multi-layer perceptron (MLP) networks for the low-frequency feature, achieving smoothness and scene completion in unobserved areas. Moreover, we propose local-to-global bundle adjustment method with a global keyframe database to address the increased pose drifts on long sequences. Experimental results demonstrate that PLGSLAM achieves state-of-the-art scene reconstruction results and tracking performance across various datasets and scenarios (both in small and large-scale indoor environments). The code is open-sourced at https://github.com/dtc111111/plgslam.
>
---
#### [replaced 064] OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17473v2](http://arxiv.org/pdf/2505.17473v2)**

> **作者:** Jiangning Zhu; Yuxing Zhou; Zheng Wang; Juntao Yao; Yima Gu; Yuhui Yuan; Shixia Liu
>
> **摘要:** Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection.
>
---
#### [replaced 065] OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20292v2](http://arxiv.org/pdf/2505.20292v2)**

> **作者:** Shenghai Yuan; Xianyi He; Yufan Deng; Yang Ye; Jinfa Huang; Bin Lin; Chongyang Ma; Jiebo Luo; Li Yuan
>
> **备注:** Code and Dataset: https://github.com/PKU-YuanGroup/OpenS2V-Nexus
>
> **摘要:** Subject-to-Video (S2V) generation aims to create videos that faithfully incorporate reference content, providing enhanced flexibility in the production of videos. To establish the infrastructure for S2V generation, we propose OpenS2V-Nexus, consisting of (i) OpenS2V-Eval, a fine-grained benchmark, and (ii) OpenS2V-5M, a million-scale dataset. In contrast to existing S2V benchmarks inherited from VBench that focus on global and coarse-grained assessment of generated videos, OpenS2V-Eval focuses on the model's ability to generate subject-consistent videos with natural subject appearance and identity fidelity. For these purposes, OpenS2V-Eval introduces 180 prompts from seven major categories of S2V, which incorporate both real and synthetic test data. Furthermore, to accurately align human preferences with S2V benchmarks, we propose three automatic metrics, NexusScore, NaturalScore and GmeScore, to separately quantify subject consistency, naturalness, and text relevance in generated videos. Building on this, we conduct a comprehensive evaluation of 16 representative S2V models, highlighting their strengths and weaknesses across different content. Moreover, we create the first open-source large-scale S2V generation dataset OpenS2V-5M, which consists of five million high-quality 720P subject-text-video triples. Specifically, we ensure subject-information diversity in our dataset by (1) segmenting subjects and building pairing information via cross-video associations and (2) prompting GPT-Image-1 on raw frames to synthesize multi-view representations. Through OpenS2V-Nexus, we deliver a robust infrastructure to accelerate future S2V generation research.
>
---
#### [replaced 066] Toward Unified Practices in Trajectory Prediction Research on Bird's-Eye-View Datasets
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.00604v4](http://arxiv.org/pdf/2405.00604v4)**

> **作者:** Theodor Westny; Björn Olofsson; Erik Frisk
>
> **备注:** https://github.com/westny/dronalize
>
> **摘要:** The availability of high-quality datasets is crucial for the development of behavior prediction algorithms in autonomous vehicles. This paper highlights the need to standardize the use of certain datasets for motion forecasting research to simplify comparative analysis and proposes a set of tools and practices to achieve this. Drawing on extensive experience and a comprehensive review of current literature, we summarize our proposals for preprocessing, visualization, and evaluation in the form of an open-sourced toolbox designed for researchers working on trajectory prediction problems. The clear specification of necessary preprocessing steps and evaluation metrics is intended to alleviate development efforts and facilitate the comparison of results across different studies. The toolbox is available at: https://github.com/westny/dronalize.
>
---
#### [replaced 067] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis on the role of model complexity
- **分类: stat.ML; cs.AI; cs.CV; cs.LG; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2411.02184v2](http://arxiv.org/pdf/2411.02184v2)**

> **作者:** Mouïn Ben Ammar; David Brellmann; Arturo Mendoza; Antoine Manzanera; Gianni Franchi
>
> **摘要:** Out-of-distribution (OOD) detection is essential for ensuring the reliability and safety of machine learning systems. In recent years, it has received increasing attention, particularly through post-hoc detection and training-based methods. In this paper, we focus on post-hoc OOD detection, which enables identifying OOD samples without altering the model's training procedure or objective. Our primary goal is to investigate the relationship between model capacity and its OOD detection performance. Specifically, we aim to answer the following question: Does the Double Descent phenomenon manifest in post-hoc OOD detection? This question is crucial, as it can reveal whether overparameterization, which is already known to benefit generalization, can also enhance OOD detection. Despite the growing interest in these topics by the classic supervised machine learning community, this intersection remains unexplored for OOD detection. We empirically demonstrate that the Double Descent effect does indeed appear in post-hoc OOD detection. Furthermore, we provide theoretical insights to explain why this phenomenon emerges in such setting. Finally, we show that the overparameterized regime does not yield superior results consistently, and we propose a method to identify the optimal regime for OOD detection based on our observations.
>
---
#### [replaced 068] Structure-Accurate Medical Image Translation via Dynamic Frequency Balance and Knowledge Guidance
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.09441v2](http://arxiv.org/pdf/2504.09441v2)**

> **作者:** Jiahua Xu; Dawei Zhou; Lei Hu; Zaiyi Liu; Nannan Wang; Xinbo Gao
>
> **备注:** Medical image translation, Diffusion model, 16 pages
>
> **摘要:** Multimodal medical images play a crucial role in the precise and comprehensive clinical diagnosis. Diffusion model is a powerful strategy to synthesize the required medical images. However, existing approaches still suffer from the problem of anatomical structure distortion due to the overfitting of high-frequency information and the weakening of low-frequency information. Thus, we propose a novel method based on dynamic frequency balance and knowledge guidance. Specifically, we first extract the low-frequency and high-frequency components by decomposing the critical features of the model using wavelet transform. Then, a dynamic frequency balance module is designed to adaptively adjust frequency for enhancing global low-frequency features and effective high-frequency details as well as suppressing high-frequency noise. To further overcome the challenges posed by the large differences between different medical modalities, we construct a knowledge-guided mechanism that fuses the prior clinical knowledge from a visual language model with visual features, to facilitate the generation of accurate anatomical structures. Experimental evaluations on multiple datasets show the proposed method achieves significant improvements in qualitative and quantitative assessments, verifying its effectiveness and superiority.
>
---
#### [replaced 069] EventEgoHands: Event-based Egocentric 3D Hand Mesh Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19169v2](http://arxiv.org/pdf/2505.19169v2)**

> **作者:** Ryosei Hara; Wataru Ikeda; Masashi Hatano; Mariko Isogawa
>
> **备注:** IEEE International Conference on Image Processing 2025, Project Page: https://ryhara.github.io/EventEgoHands/
>
> **摘要:** Reconstructing 3D hand mesh is challenging but an important task for human-computer interaction and AR/VR applications. In particular, RGB and/or depth cameras have been widely used in this task. However, methods using these conventional cameras face challenges in low-light environments and during motion blur. Thus, to address these limitations, event cameras have been attracting attention in recent years for their high dynamic range and high temporal resolution. Despite their advantages, event cameras are sensitive to background noise or camera motion, which has limited existing studies to static backgrounds and fixed cameras. In this study, we propose EventEgoHands, a novel method for event-based 3D hand mesh reconstruction in an egocentric view. Our approach introduces a Hand Segmentation Module that extracts hand regions, effectively mitigating the influence of dynamic background events. We evaluated our approach and demonstrated its effectiveness on the N-HOT3D dataset, improving MPJPE by approximately more than 4.5 cm (43%).
>
---
#### [replaced 070] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14359v2](http://arxiv.org/pdf/2505.14359v2)**

> **作者:** Ruoxin Chen; Junwei Xi; Zhiyuan Yan; Ke-Yue Zhang; Shuang Wu; Jingyi Xie; Xu Chen; Lei Xu; Isabel Guan; Taiping Yao; Shouhong Ding
>
> **备注:** 12 Pages, 9 figures
>
> **摘要:** Existing detectors are often trained on biased datasets, leading to the possibility of overfitting on non-causal image attributes that are spuriously correlated with real/synthetic labels. While these biased features enhance performance on the training data, they result in substantial performance degradation when applied to unbiased datasets. One common solution is to perform dataset alignment through generative reconstruction, matching the semantic content between real and synthetic images. However, we revisit this approach and show that pixel-level alignment alone is insufficient. The reconstructed images still suffer from frequency-level misalignment, which can perpetuate spurious correlations. To illustrate, we observe that reconstruction models tend to restore the high-frequency details lost in real images (possibly due to JPEG compression), inadvertently creating a frequency-level misalignment, where synthetic images appear to have richer high-frequency content than real ones. This misalignment leads to models associating high-frequency features with synthetic labels, further reinforcing biased cues. To resolve this, we propose Dual Data Alignment (DDA), which aligns both the pixel and frequency domains. Moreover, we introduce two new test sets: DDA-COCO, containing DDA-aligned synthetic images for testing detector performance on the most aligned dataset, and EvalGEN, featuring the latest generative models for assessing detectors under new generative architectures such as visual auto-regressive generators. Finally, our extensive evaluations demonstrate that a detector trained exclusively on DDA-aligned MSCOCO could improve across 8 diverse benchmarks by a non-trivial margin, showing a +7.2% on in-the-wild benchmarks, highlighting the improved generalizability of unbiased detectors.
>
---
#### [replaced 071] SPF-Portrait: Towards Pure Text-to-Portrait Customization with Semantic Pollution-Free Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00396v3](http://arxiv.org/pdf/2504.00396v3)**

> **作者:** Xiaole Xian; Zhichao Liao; Qingyu Li; Wenyu Qin; Pengfei Wan; Weicheng Xie; Long Zeng; Linlin Shen; Pingfa Feng
>
> **摘要:** Fine-tuning a pre-trained Text-to-Image (T2I) model on a tailored portrait dataset is the mainstream method for text-to-portrait customization. However, existing methods often severely impact the original model's behavior (e.g., changes in ID, layout, etc.) while customizing portrait attributes. To address this issue, we propose SPF-Portrait, a pioneering work to purely understand customized target semantics and minimize disruption to the original model. In our SPF-Portrait, we design a dual-path contrastive learning pipeline, which introduces the original model as a behavioral alignment reference for the conventional fine-tuning path. During the contrastive learning, we propose a novel Semantic-Aware Fine Control Map that indicates the intensity of response regions of the target semantics, to spatially guide the alignment process between the contrastive paths. It adaptively balances the behavioral alignment across different regions and the responsiveness of the target semantics. Furthermore, we propose a novel response enhancement mechanism to reinforce the presentation of target semantics, while mitigating representation discrepancy inherent in direct cross-modal supervision. Through the above strategies, we achieve incremental learning of customized target semantics for pure text-to-portrait customization. Extensive experiments show that SPF-Portrait achieves state-of-the-art performance. Project page: https://spf-portrait.github.io/SPF-Portrait/
>
---
#### [replaced 072] Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.08105v4](http://arxiv.org/pdf/2408.08105v4)**

> **作者:** Zhiyuan Li; Heng Wang; Dongnan Liu; Chaoyi Zhang; Ao Ma; Jieting Long; Weidong Cai
>
> **备注:** ACL2025 Findings
>
> **摘要:** Multimodal Large Language Models (MLLMs) have showcased exceptional Chain-of-Thought (CoT) reasoning ability in complex textual inference tasks including causal reasoning. However, will these causalities remain straightforward when crucial hints hide in visual details? If not, what factors might influence cross-modal generalization? Whether we can effectively enhance their capacity for robust causal inference across both text and vision? Motivated by these, we introduce MuCR - a novel Multimodal Causal Reasoning benchmark that leverages synthetic siamese images and text pairs to challenge MLLMs. Additionally, we develop tailored metrics from multiple perspectives, including image-level match, phrase-level understanding, and sentence-level explanation, to comprehensively assess MLLMs' comprehension abilities. Our experiments reveal that current MLLMs fall short in multimodal causal reasoning compared to their performance in purely textual settings. Additionally, we find that identifying visual cues across images is key to effective cross-modal generalization. Finally, we propose a VcCoT strategy that better highlights visual cues, and our results confirm its efficacy in enhancing multimodal causal reasoning. The project is available at: https://github.com/Zhiyuan-Li-John/MuCR
>
---
#### [replaced 073] Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17659v2](http://arxiv.org/pdf/2505.17659v2)**

> **作者:** Xiaolong Tang; Meina Kan; Shiguang Shan; Xilin Chen
>
> **摘要:** Safe and feasible trajectory planning is essential for real-world autonomous driving systems. However, existing learning-based planning methods often rely on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting unsafe behaviors such as speeding from suboptimal human driving data. Inspired by the success of large language models, we propose Plan-R1, a novel two-stage trajectory planning framework that formulates trajectory planning as a sequential prediction task, guided by explicit planning principles such as safety, comfort, and traffic rule compliance. In the first stage, we train an autoregressive trajectory predictor via next motion token prediction on expert data. In the second stage, we design rule-based rewards (e.g., collision avoidance, speed limits) and fine-tune the model using Group Relative Policy Optimization (GRPO), a reinforcement learning strategy, to align its predictions with these planning principles. Experiments on the nuPlan benchmark demonstrate that our Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance. Our code will be made public soon.
>
---
#### [replaced 074] DADM: Dual Alignment of Domain and Modality for Face Anti-spoofing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00429v2](http://arxiv.org/pdf/2503.00429v2)**

> **作者:** Jingyi Yang; Xun Lin; Zitong Yu; Liepiao Zhang; Xin Liu; Hui Li; Xiaochen Yuan; Xiaochun Cao
>
> **备注:** 18 pages, 9 figures, Code: https://github.com/yjyddq/DADM
>
> **摘要:** With the availability of diverse sensor modalities (i.e., RGB, Depth, Infrared) and the success of multi-modal learning, multi-modal face anti-spoofing (FAS) has emerged as a prominent research focus. The intuition behind it is that leveraging multiple modalities can uncover more intrinsic spoofing traces. However, this approach presents more risk of misalignment. We identify two main types of misalignment: (1) \textbf{Intra-domain modality misalignment}, where the importance of each modality varies across different attacks. For instance, certain modalities (e.g., Depth) may be non-defensive against specific attacks (e.g., 3D mask), indicating that each modality has unique strengths and weaknesses in countering particular attacks. Consequently, simple fusion strategies may fall short. (2) \textbf{Inter-domain modality misalignment}, where the introduction of additional modalities exacerbates domain shifts, potentially overshadowing the benefits of complementary fusion. To tackle (1), we propose a alignment module between modalities based on mutual information, which adaptively enhances favorable modalities while suppressing unfavorable ones. To address (2), we employ a dual alignment optimization method that aligns both sub-domain hyperplanes and modality angle margins, thereby mitigating domain gaps. Our method, dubbed \textbf{D}ual \textbf{A}lignment of \textbf{D}omain and \textbf{M}odality (DADM), achieves state-of-the-art performance in extensive experiments across four challenging protocols demonstrating its robustness in multi-modal domain generalization scenarios. The codes will be released soon.
>
---
#### [replaced 075] H2VU-Benchmark: A Comprehensive Benchmark for Hierarchical Holistic Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.24008v2](http://arxiv.org/pdf/2503.24008v2)**

> **作者:** Qi Wu; Quanlong Zheng; Yanhao Zhang; Junlin Xie; Jinguo Luo; Kuo Wang; Peng Liu; Qingsong Xie; Ru Zhen; Zhenyu Yang; Haonan Lu
>
> **摘要:** With the rapid development of multimodal models, the demand for assessing video understanding capabilities has been steadily increasing. However, existing benchmarks for evaluating video understanding exhibit significant limitations in coverage, task diversity, and scene adaptability. These shortcomings hinder the accurate assessment of models' comprehensive video understanding capabilities. To tackle this challenge, we propose a hierarchical and holistic video understanding (H2VU) benchmark designed to evaluate both general video and online streaming video comprehension. This benchmark contributes three key features: Extended video duration: Spanning videos from brief 3-second clips to comprehensive 1.5-hour recordings, thereby bridging the temporal gaps found in current benchmarks. Comprehensive assessment tasks: Beyond traditional perceptual and reasoning tasks, we have introduced modules for countercommonsense comprehension and trajectory state tracking. These additions test the models' deep understanding capabilities beyond mere prior knowledge. Enriched video data: To keep pace with the rapid evolution of current AI agents, we have expanded first-person streaming video datasets. This expansion allows for the exploration of multimodal models' performance in understanding streaming videos from a first-person perspective. Extensive results from H2VU reveal that existing multimodal large language models (MLLMs) possess substantial potential for improvement in our newly proposed evaluation tasks. We expect that H2VU will facilitate advancements in video understanding research by offering a comprehensive and in-depth analysis of MLLMs.
>
---
#### [replaced 076] DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19381v2](http://arxiv.org/pdf/2505.19381v2)**

> **作者:** Anqing Jiang; Yu Gao; Zhigang Sun; Yiru Wang; Jijun Wang; Jinghao Chai; Qian Cao; Yuweng Heng; Hao Jiang; Zongzheng Zhang; Xianda Guo; Hao Sun; Hao Zhao
>
> **备注:** 4pages
>
> **摘要:** Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS.
>
---
#### [replaced 077] HA-VLN: A Benchmark for Human-Aware Navigation in Discrete-Continuous Environments with Dynamic Multi-Human Interactions, Real-World Validation, and an Open Leaderboard
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.14229v2](http://arxiv.org/pdf/2503.14229v2)**

> **作者:** Yifei Dong; Fengyi Wu; Qi He; Heng Li; Minghan Li; Zebang Cheng; Yuxuan Zhou; Jingdong Sun; Qi Dai; Zhi-Qi Cheng; Alexander G Hauptmann
>
> **备注:** 27 pages, 21 figures, with added experiments and analysis, website: https://ha-vln-project.vercel.app/
>
> **摘要:** Vision-and-Language Navigation (VLN) systems often focus on either discrete (panoramic) or continuous (free-motion) paradigms alone, overlooking the complexities of human-populated, dynamic environments. We introduce a unified Human-Aware VLN (HA-VLN) benchmark that merges these paradigms under explicit social-awareness constraints. Our contributions include: 1. A standardized task definition that balances discrete-continuous navigation with personal-space requirements; 2. An enhanced human motion dataset (HAPS 2.0) and upgraded simulators capturing realistic multi-human interactions, outdoor contexts, and refined motion-language alignment; 3. Extensive benchmarking on 16,844 human-centric instructions, revealing how multi-human dynamics and partial observability pose substantial challenges for leading VLN agents; 4. Real-world robot tests validating sim-to-real transfer in crowded indoor spaces; and 5. A public leaderboard supporting transparent comparisons across discrete and continuous tasks. Empirical results show improved navigation success and fewer collisions when social context is integrated, underscoring the need for human-centric design. By releasing all datasets, simulators, agent code, and evaluation tools, we aim to advance safer, more capable, and socially responsible VLN research.
>
---
#### [replaced 078] Unforgettable Lessons from Forgettable Images: Intra-Class Memorability Matters in Computer Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20761v4](http://arxiv.org/pdf/2412.20761v4)**

> **作者:** Jie Jing; Qing Lin; Shuangpeng Han; Lucia Schiatti; Yen-Ling Kuo; Mengmi Zhang
>
> **摘要:** We introduce intra-class memorability, where certain images within the same class are more memorable than others despite shared category characteristics. To investigate what features make one object instance more memorable than others, we design and conduct human behavior experiments, where participants are shown a series of images, and they must identify when the current image matches the image presented a few steps back in the sequence. To quantify memorability, we propose the Intra-Class Memorability score (ICMscore), a novel metric that incorporates the temporal intervals between repeated image presentations into its calculation. Furthermore, we curate the Intra-Class Memorability Dataset (ICMD), comprising over 5,000 images across ten object classes with their ICMscores derived from 2,000 participants' responses. Subsequently, we demonstrate the usefulness of ICMD by training AI models on this dataset for various downstream tasks: memorability prediction, image recognition, continual learning, and memorability-controlled image editing. Surprisingly, high-ICMscore images impair AI performance in image recognition and continual learning tasks, while low-ICMscore images improve outcomes in these tasks. Additionally, we fine-tune a state-of-the-art image diffusion model on ICMD image pairs with and without masked semantic objects. The diffusion model can successfully manipulate image elements to enhance or reduce memorability. Our contributions open new pathways in understanding intra-class memorability by scrutinizing fine-grained visual features behind the most and least memorable images and laying the groundwork for real-world applications in computer vision. We will release all code, data, and models publicly.
>
---
#### [replaced 079] QUART-Online: Latency-Free Large Multimodal Language Model for Quadruped Robot Learning
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.15576v5](http://arxiv.org/pdf/2412.15576v5)**

> **作者:** Xinyang Tong; Pengxiang Ding; Yiguo Fan; Donglin Wang; Wenjie Zhang; Can Cui; Mingyang Sun; Han Zhao; Hongyin Zhang; Yonghao Dang; Siteng Huang; Shangke Lyu
>
> **备注:** Accepted to ICRA 2025; Github page: https://quart-online.github.io
>
> **摘要:** This paper addresses the inherent inference latency challenges associated with deploying multimodal large language models (MLLM) in quadruped vision-language-action (QUAR-VLA) tasks. Our investigation reveals that conventional parameter reduction techniques ultimately impair the performance of the language foundation model during the action instruction tuning phase, making them unsuitable for this purpose. We introduce a novel latency-free quadruped MLLM model, dubbed QUART-Online, designed to enhance inference efficiency without degrading the performance of the language foundation model. By incorporating Action Chunk Discretization (ACD), we compress the original action representation space, mapping continuous action values onto a smaller set of discrete representative vectors while preserving critical information. Subsequently, we fine-tune the MLLM to integrate vision, language, and compressed actions into a unified semantic space. Experimental results demonstrate that QUART-Online operates in tandem with the existing MLLM system, achieving real-time inference in sync with the underlying controller frequency, significantly boosting the success rate across various tasks by 65%. Our project page is https://quart-online.github.io.
>
---
#### [replaced 080] One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13358v2](http://arxiv.org/pdf/2503.13358v2)**

> **作者:** Daniil Selikhanovych; David Li; Aleksei Leonov; Nikita Gushchin; Sergei Kushneriuk; Alexander Filippov; Evgeny Burnaev; Iaroslav Koshelev; Alexander Korotin
>
> **摘要:** Diffusion models for super-resolution (SR) produce high-quality visual results but require expensive computational costs. Despite the development of several methods to accelerate diffusion-based SR models, some (e.g., SinSR) fail to produce realistic perceptual details, while others (e.g., OSEDiff) may hallucinate non-existent structures. To overcome these issues, we present RSD, a new distillation method for ResShift, one of the top diffusion-based SR models. Our method is based on training the student network to produce such images that a new fake ResShift model trained on them will coincide with the teacher model. RSD achieves single-step restoration and outperforms the teacher by a large margin. We show that our distillation method can surpass the other distillation-based method for ResShift - SinSR - making it on par with state-of-the-art diffusion-based SR distillation methods. Compared to SR methods based on pre-trained text-to-image models, RSD produces competitive perceptual quality, provides images with better alignment to degraded input images, and requires fewer parameters and GPU memory. We provide experimental results on various real-world and synthetic datasets, including RealSR, RealSet65, DRealSR, ImageNet, and DIV2K.
>
---
#### [replaced 081] LIB-KD: Learning Inductive Bias, Not Just Parameters A New Perspective on Knowledge Distillations
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.00369v3](http://arxiv.org/pdf/2310.00369v3)**

> **作者:** Gousia Habib; Tausifa Jan Saleem; Ishfaq Ahmad Malik; Brejesh Lall
>
> **摘要:** With the rapid development of computer vision, Vision Transformers (ViTs) offer the tantalizing prospect of unified information processing across visual and textual domains. But due to the lack of inherent inductive biases in ViTs, they require enormous amount of data for training. To make their applications practical, we introduce an innovative ensemble-based distillation approach distilling inductive bias from complementary lightweight teacher models. Prior systems relied solely on convolution-based teaching. However, this method incorporates an ensemble of light teachers with different architectural tendencies, such as convolution and involution, to instruct the student transformer jointly. Because of these unique inductive biases, instructors can accumulate a wide range of knowledge, even from readily identifiable stored datasets, which leads to enhanced student performance. Our proposed framework also involves precomputing and storing logits in advance, essentially the unnormalized predictions of the model. This optimization can accelerate the distillation process by eliminating the need for repeated forward passes during knowledge distillation, significantly reducing the computational burden and enhancing efficiency.
>
---
#### [replaced 082] Restoring Real-World Images with an Internal Detail Enhancement Diffusion Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18674v2](http://arxiv.org/pdf/2505.18674v2)**

> **作者:** Peng Xiao; Hongbo Zhao; Yijun Wang; Jianxin Lin
>
> **摘要:** Restoring real-world degraded images, such as old photographs or low-resolution images, presents a significant challenge due to the complex, mixed degradations they exhibit, such as scratches, color fading, and noise. Recent data-driven approaches have struggled with two main challenges: achieving high-fidelity restoration and providing object-level control over colorization. While diffusion models have shown promise in generating high-quality images with specific controls, they often fail to fully preserve image details during restoration. In this work, we propose an internal detail-preserving diffusion model for high-fidelity restoration of real-world degraded images. Our method utilizes a pre-trained Stable Diffusion model as a generative prior, eliminating the need to train a model from scratch. Central to our approach is the Internal Image Detail Enhancement (IIDE) technique, which directs the diffusion model to preserve essential structural and textural information while mitigating degradation effects. The process starts by mapping the input image into a latent space, where we inject the diffusion denoising process with degradation operations that simulate the effects of various degradation factors. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art models in both qualitative assessments and perceptual quantitative evaluations. Additionally, our approach supports text-guided restoration, enabling object-level colorization control that mimics the expertise of professional photo editing.
>
---
#### [replaced 083] SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13112v3](http://arxiv.org/pdf/2411.13112v3)**

> **作者:** Xianda Guo; Ruijun Zhang; Yiqun Duan; Yuhang He; Dujun Nie; Wenke Huang; Chenming Zhang; Shuai Liu; Hao Zhao; Long Chen
>
> **摘要:** Accurate spatial reasoning in outdoor environments - covering geometry, object pose, and inter-object relationships - is fundamental to downstream tasks such as mapping, motion forecasting, and high-level planning in autonomous driving. We introduce SURDS, a large-scale benchmark designed to systematically evaluate the spatial reasoning capabilities of vision language models (VLMs). Built on the nuScenes dataset, SURDS comprises 41,080 vision-question-answer training instances and 9,250 evaluation samples, spanning six spatial categories: orientation, depth estimation, pixel-level localization, pairwise distance, lateral ordering, and front-behind relations. We benchmark leading general-purpose VLMs, including GPT, Gemini, and Qwen, revealing persistent limitations in fine-grained spatial understanding. To address these deficiencies, we go beyond static evaluation and explore whether alignment techniques can improve spatial reasoning performance. Specifically, we propose a reinforcement learning-based alignment scheme leveraging spatially grounded reward signals - capturing both perception-level accuracy (location) and reasoning consistency (logic). We further incorporate final-answer correctness and output-format rewards to guide fine-grained policy adaptation. Our GRPO-aligned variant achieves an overall score of 40.80 in the SURDS benchmark. Notably, it outperforms proprietary systems such as GPT-4o (13.30) and Gemini-2.0-flash (35.71). To our best knowledge, this is the first study to demonstrate that reinforcement learning-based alignment can significantly and consistently enhance the spatial reasoning capabilities of VLMs in real-world driving contexts. We release the SURDS benchmark, evaluation toolkit, and GRPO alignment code through: https://github.com/XiandaGuo/Drive-MLLM.
>
---
#### [replaced 084] diffDemorph: Extending Reference-Free Demorphing to Unseen Faces
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14527v2](http://arxiv.org/pdf/2505.14527v2)**

> **作者:** Nitish Shukla; Arun Ross
>
> **摘要:** A face morph is created by combining two (or more) face images corresponding to two (or more) identities to produce a composite that successfully matches the constituent identities. Reference-free (RF) demorphing reverses this process using only the morph image, without the need for additional reference images. Previous RF demorphing methods were overly constrained, as they rely on assumptions about the distributions of training and testing morphs such as the morphing technique used, face style, and images used to create the morph. In this paper, we introduce a novel diffusion-based approach that effectively disentangles component images from a composite morph image with high visual fidelity. Our method is the first to generalize across morph techniques and face styles, beating the current state of the art by $\geq 59.46\%$ under a common training protocol across all datasets tested. We train our method on morphs created using synthetically generated face images and test on real morphs, thereby enhancing the practicality of the technique. Experiments on six datasets and two face matchers establish the utility and efficacy of our method.
>
---
#### [replaced 085] GarmentDiffusion: 3D Garment Sewing Pattern Generation with Multimodal Diffusion Transformers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21476v3](http://arxiv.org/pdf/2504.21476v3)**

> **作者:** Xinyu Li; Qi Yao; Yuanda Wang
>
> **备注:** The 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** Garment sewing patterns are fundamental design elements that bridge the gap between design concepts and practical manufacturing. The generative modeling of sewing patterns is crucial for creating diversified garments. However, existing approaches are limited either by reliance on a single input modality or by suboptimal generation efficiency. In this work, we present GarmentDiffusion, a new generative model capable of producing centimeter-precise, vectorized 3D sewing patterns from multimodal inputs (text, image, and incomplete sewing pattern). Our method efficiently encodes 3D sewing pattern parameters into compact edge token representations, achieving a sequence length that is 10 times shorter than that of the autoregressive SewingGPT in DressCode. By employing a diffusion transformer, we simultaneously denoise all edge tokens along the temporal axis, while maintaining a constant number of denoising steps regardless of dataset-specific edge and panel statistics. With all combination of designs of our model, the sewing pattern generation speed is accelerated by 100 times compared to SewingGPT. We achieve new state-of-the-art results on DressCodeData, as well as on the largest sewing pattern dataset, namely GarmentCodeData. The project website is available at https://shenfu-research.github.io/Garment-Diffusion/.
>
---
#### [replaced 086] Rebalancing Contrastive Alignment with Learnable Semantic Gaps in Text-Video Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.12499v3](http://arxiv.org/pdf/2505.12499v3)**

> **作者:** Jian Xiao; Zijie Song; Jialong Hu; Hao Cheng; Zhenzhen Hu; Jia Li; Richang Hong
>
> **摘要:** Recent advances in text-video retrieval have been largely driven by contrastive learning frameworks. However, existing methods overlook a key source of optimization tension: the separation between text and video distributions in the representation space (referred to as the modality gap), and the prevalence of false negatives in batch sampling. These factors lead to conflicting gradients under the InfoNCE loss, impeding stable alignment. To mitigate this, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment Delta_ij between text t_i and video v_j to offload the tension from the global anchor representation. We first derive the ideal form of Delta_ij via a coupled multivariate first-order Taylor approximation of the InfoNCE loss under a trust-region constraint, revealing it as a mechanism for resolving gradient conflicts by guiding updates along a locally optimal descent direction. Due to the high cost of directly computing Delta_ij, we introduce a lightweight neural module conditioned on the semantic gap between each video-text pair, enabling structure-aware correction guided by gradient supervision. To further stabilize learning and promote interpretability, we regularize Delta using three components: a trust-region constraint to prevent oscillation, a directional diversity term to promote semantic coverage, and an information bottleneck to limit redundancy. Experiments across four retrieval benchmarks show that GARE consistently improves alignment accuracy and robustness to noisy supervision, confirming the effectiveness of gap-aware tension mitigation.
>
---
