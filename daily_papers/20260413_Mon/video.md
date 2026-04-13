# 计算机视觉 cs.CV

- **最新发布 146 篇**

- **更新 78 篇**

## 最新发布

#### [new 001] FaceLiVTv2: An Improved Hybrid Architecture for Efficient Mobile Face Recognition
- **分类: cs.CV**

- **简介: 该论文属于轻量级人脸识别任务，旨在提升移动端的识别效率与准确性。提出FaceLiVTv2架构，通过改进模块优化全局与局部特征交互，实现更优的性能与效率平衡。**

- **链接: [https://arxiv.org/pdf/2604.09127](https://arxiv.org/pdf/2604.09127)**

> **作者:** Novendra Setyawan; Chi-Chia Sun; Mao-Hsiu Hsu; Wen-Kai Kuo; Jun-Wei Hsieh
>
> **摘要:** Lightweight face recognition is increasingly important for deployment on edge and mobile devices, where strict constraints on latency, memory, and energy consumption must be met alongside reliable accuracy. Although recent hybrid CNN-Transformer architectures have advanced global context modeling, striking an effective balance between recognition performance and computational efficiency remains an open challenge. In this work, we present FaceLiVTv2, an improved version of our FaceLiVT hybrid architecture designed for efficient global--local feature interaction in mobile face recognition. At its core is Lite MHLA, a lightweight global token interaction module that replaces the original multi-layer attention design with multi-head linear token projections and affine rescale transformations, reducing redundancy while preserving representational diversity across heads. We further integrate Lite MHLA into a unified RepMix block that coordinates local and global feature interactions and adopts global depthwise convolution for adaptive spatial aggregation in the embedding stage. Under our experimental setup, results on LFW, CA-LFW, CP-LFW, CFP-FP, AgeDB-30, and IJB show that FaceLiVTv2 consistently improves the accuracy-efficiency trade-off over existing lightweight methods. Notably, FaceLiVTv2 reduces mobile inference latency by 22% relative to FaceLiVTv1, achieves speedups of up to 30.8% over GhostFaceNets on mobile devices, and delivers 20-41% latency improvements over EdgeFace and KANFace across platforms while maintaining higher recognition accuracy. These results demonstrate that FaceLiVTv2 offers a practical and deployable solution for real-time face recognition. Code is available at this https URL.
>
---
#### [new 002] Adaptive Dual Residual U-Net with Attention Gate and Multiscale Spatial Attention Mechanisms (ADRUwAMS)
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决脑肿瘤自动分割难题。提出ADRUwAMS模型，结合残差网络、注意力机制和多尺度空间注意力，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.08893](https://arxiv.org/pdf/2604.08893)**

> **作者:** Mohsen Yaghoubi Suraki
>
> **摘要:** Glioma is a harmful brain tumor that requires early detection to ensure better health results. Early detection of this tumor is key for effective treatment and requires an automated segmentation process. However, it is a challenging task to find tumors due to tumor characteristics like location and size. A reliable method to accurately separate tumor zones from healthy tissues is deep learning models, which have shown promising results over the last few years. In this research, an Adaptive Dual Residual U-Net with Attention Gate and Multiscale Spatial Attention Mechanisms (ADRUwAMS) is introduced. This model is an innovative combination of adaptive dual residual networks, attention mechanisms, and multiscale spatial attention. The dual adaptive residual network architecture captures high-level semantic and intricate low-level details from brain images, ensuring precise segmentation of different tumor parts, types, and hard regions. The attention gates use gating and input signals to compute attention coefficients for the input features, and multiscale spatial attention generates scaled attention maps and combines these features to hold the most significant information about the brain tumor. We trained the model for 200 epochs using the ReLU activation function on BraTS 2020 and BraTS 2019 datasets. These improvements resulted in high accuracy for tumor detection and segmentation on BraTS 2020, achieving dice scores of 0.9229 for the whole tumor, 0.8432 for the tumor core, and 0.8004 for the enhancing tumor.
>
---
#### [new 003] Skill-Conditioned Visual Geolocation for Vision-Language
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉定位任务，旨在解决VLM在地理推理和自我进化上的不足。提出GeoSkill框架，通过技能图实现持续学习与优化，提升定位准确性和推理可靠性。**

- **链接: [https://arxiv.org/pdf/2604.09025](https://arxiv.org/pdf/2604.09025)**

> **作者:** Chenjie Yang; Yutian Jiang; Chenyu Wu
>
> **摘要:** Vision-language models (VLMs) have shown a promising ability in image geolocation, but they still lack structured geographic reasoning and the capacity for autonomous self-evolution. Existing methods predominantly rely on implicit parametric memory, which often exploits outdated knowledge and generates hallucinated reasoning. Furthermore, current inference is a "one-off" process, lacking the feedback loops necessary for self-evolution based on reasoning outcomes. To address these issues, we propose GeoSkill, a training-free framework based on an evolving Skill-Graph. We first initialize the graph by refining human expert trajectories into atomic, natural-language skills. For execution, GeoSkill employs an inference model to perform direct reasoning guided by the current Skill-Graph. For continuous growth, an Autonomous Evolution mechanism leverages a larger model to conduct multiple reasoning rollouts on image-coordinate pairs sourced from web-scale data and verified real-world reasoning. By analyzing both successful and failed trajectories from these rollouts, the mechanism iteratively synthesizes and prunes skills, effectively expanding the Skill-Graph and correcting geographic biases without any parameter updates. Experiments demonstrate that GeoSkill achieves promising performance in both geolocation accuracy and reasoning faithfulness on GeoRC, while maintaining superior generalization across diverse external datasets. Furthermore, our autonomous evolution fosters the emergence of novel, verifiable skills, significantly enhancing the system's cognition of real-world geographic knowledge beyond isolated case studies.
>
---
#### [new 004] SIC3D: Style Image Conditioned Text-to-3D Gaussian Splatting Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到3D生成任务，旨在解决生成可控性差和纹理模糊的问题。提出SIC3D框架，结合3D高斯点云与图像风格迁移，提升几何精度与风格一致性。**

- **链接: [https://arxiv.org/pdf/2604.08760](https://arxiv.org/pdf/2604.08760)**

> **作者:** Ming He; Zhixiang Chen; Steve Maddock
>
> **摘要:** Recent progress in text-to-3D object generation enables the synthesis of detailed geometry from text input by leveraging 2D diffusion models and differentiable 3D representations. However, the approaches often suffer from limited controllability and texture ambiguity due to the limitation of the text modality. To address this, we present SIC3D, a controllable image-conditioned text-to-3D generation pipeline with 3D Gaussian Splatting (3DGS). There are two stages in SIC3D. The first stage generates the 3D object content from text with a text-to-3DGS generation model. The second stage transfers style from a reference image to the 3DGS. Within this stylization stage, we introduce a novel Variational Stylized Score Distillation (VSSD) loss to effectively capture both global and local texture patterns while mitigating conflicts between geometry and appearance. A scaling regularization is further applied to prevent the emergence of artifacts and preserve the pattern from the style image. Extensive experiments demonstrate that SIC3D enhances geometric fidelity and style adherence, outperforming prior approaches in both qualitative and quantitative evaluations.
>
---
#### [new 005] Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种联合学习视频与相机轨迹的模型，解决图像覆盖稀疏或姿态模糊时任务分离的问题。通过密集光线像素和注意力机制，实现相机轨迹预测与视频生成。**

- **链接: [https://arxiv.org/pdf/2604.09429](https://arxiv.org/pdf/2604.09429)**

> **作者:** Wonbong Jang; Shikun Liu; Soubhik Sanyal; Juan Camilo Perez; Kam Woh Ng; Sanskar Agrawal; Juan-Manuel Perez-Rua; Yiannis Douratsos; Tao Xiang
>
> **备注:** 9 pages, 6 figures, 4 tables. Project page: this https URL
>
> **摘要:** Recovering camera parameters from images and rendering scenes from novel viewpoints have long been treated as separate tasks in computer vision and graphics. This separation breaks down when image coverage is sparse or poses are ambiguous, since each task needs what the other produces. We propose Rays as Pixels, a Video Diffusion Model (VDM) that learns a joint distribution over videos and camera trajectories. We represent each camera as dense ray pixels (raxels) and denoise them jointly with video frames through Decoupled Self-Cross Attention mechanism. A single trained model handles three tasks: predicting camera trajectories from video, jointly generating video and camera trajectory from input images, and generating video from input images along a target camera trajectory. Because the model can both predict trajectories from a video and generate views conditioned on its own predictions, we evaluate it through a closed-loop self-consistency test, demonstrating that its forward and inverse predictions agree. Notably, trajectory prediction requires far fewer denoising steps than video generation, even a few denoising steps suffice for self-consistency. We report results on pose estimation and camera-controlled video generation.
>
---
#### [new 006] What Matters in Virtual Try-Off? Dual-UNet Diffusion Model For Garment Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于虚拟试衣任务中的逆问题——从穿在身上的图像重建原始衣物。研究提出Dual-UNet扩散模型，探索生成骨干、条件设计及损失策略，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.08716](https://arxiv.org/pdf/2604.08716)**

> **作者:** Loc-Phat Truong; Meysam Madadi; Sergio Escalera
>
> **摘要:** Virtual Try-On (VTON) has seen rapid advancements, providing a strong foundation for generative fashion tasks. However, the inverse problem, Virtual Try-Off (VTOFF)-aimed at reconstructing the canonical garment from a draped-on image-remains a less understood domain, distinct from the heavily researched field of VTON. In this work, we seek to establish a robust architectural foundation for VTOFF by studying and adapting various diffusion-based strategies from VTON and general Latent Diffusion Models (LDMs). We focus our investigation on the Dual-UNet Diffusion Model architecture and analyze three axes of design: (i) Generation Backbone: comparing Stable Diffusion variants; (ii) Conditioning: ablating different mask designs, masked/unmasked inputs for image conditioning, and the utility of high-level semantic features; and (iii) Losses and Training Strategies: evaluating the impact of the auxiliary attention-based loss, perceptual objectives and multi-stage curriculum schedules. Extensive experiments reveal trade-offs across various configuration options. Evaluated on VITON-HD and DressCode datasets, our framework achieves state-of-the-art performance with a drop of 9.5\% on the primary metric DISTS and competitive performance on LPIPS, FID, KID, and SSIM, providing both stronger baselines and insights to guide future Virtual Try-Off research.
>
---
#### [new 007] SCoRe: Clean Image Generation from Diffusion Models Trained on Noisy Images
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型在噪声数据上训练后生成质量下降的问题。提出SCoRe方法，在生成阶段通过频域处理提升图像清晰度。**

- **链接: [https://arxiv.org/pdf/2604.09436](https://arxiv.org/pdf/2604.09436)**

> **作者:** Yuta Matsuzaki; Seiichi Uchida; Shumpei Takezaki
>
> **备注:** Accepted at IJCNN2026
>
> **摘要:** Diffusion models trained on noisy datasets often reproduce high-frequency training artifacts, significantly degrading generation quality. To address this, we propose SCoRe (Spectral Cutoff Regeneration), a training-free, generation-time spectral regeneration method for clean image generation from diffusion models trained on noisy images. Leveraging the spectral bias of diffusion models, which infer high-frequency details from low-frequency cues, SCoRe suppresses corrupted high-frequency components of a generated image via a frequency cutoff and regenerates them via SDEdit. Crucially, we derive a theoretical mapping between the cutoff frequency and the SDEdit initialization timestep based on Radially Averaged Power Spectral Density (RAPSD), which prevents excessive noise injection during regeneration. Experiments on synthetic (CIFAR-10) and real-world (SIDD) noisy datasets demonstrate that SCoRe substantially outperforms post-processing and noise-robust baselines, restoring samples closer to clean image distributions without any retraining or fine-tuning.
>
---
#### [new 008] EpiAgent: An Agent-Centric System for Ancient Inscription Restoration
- **分类: cs.CV**

- **简介: 该论文属于古文字修复任务，旨在解决复杂退化古铭文的恢复问题。提出EpiAgent系统，通过多模态协作和迭代优化实现更高质量的修复。**

- **链接: [https://arxiv.org/pdf/2604.09367](https://arxiv.org/pdf/2604.09367)**

> **作者:** Shipeng Zhu; Ang Chen; Na Nie; Pengfei Fang; Min-Ling Zhang; Hui Xue
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Ancient inscriptions, as repositories of cultural memory, have suffered from centuries of environmental and human-induced degradation. Restoring their intertwined visual and textual integrity poses one of the most demanding challenges in digital heritage preservation. However, existing AI-based approaches often rely on rigid pipelines, struggling to generalize across such complex and heterogeneous real-world degradations. Inspired by the skill-coordinated workflow of human epigraphers, we propose EpiAgent, an agent-centric system that formulates inscription restoration as a hierarchical planning problem. Following an Observe-Conceive-Execute-Reevaluate paradigm, an LLM-based central planner orchestrates collaboration among multimodal analysis, historical experience, specialized restoration tools, and iterative self-refinement. This agent-centric coordination enables a flexible and adaptive restoration process beyond conventional single-pass methods. Across real-world degraded inscriptions, EpiAgent achieves superior restoration quality and stronger generalization compared to existing methods. Our work marks an important step toward expert-level agent-driven restoration of cultural heritage. The code is available at this https URL.
>
---
#### [new 009] FashionStylist: An Expert Knowledge-enhanced Multimodal Dataset for Fashion Understanding
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出FashionStylist，一个用于时尚理解的多模态数据集，解决现有数据集碎片化问题。支持 outfit-to-item 、补全和评估任务，提升时尚系统中的语义理解能力。**

- **链接: [https://arxiv.org/pdf/2604.09249](https://arxiv.org/pdf/2604.09249)**

> **作者:** Kaidong Feng; Zhuoxuan Huang; Huizhong Guo; Yuting Jin; Xinyu Chen; Yue Liang; Yifei Gai; Li Zhou; Yunshan Ma; Zhu Sun
>
> **摘要:** Fashion understanding requires both visual perception and expert-level reasoning about style, occasion, compatibility, and outfit rationale. However, existing fashion datasets remain fragmented and task-specific, often focusing on item attributes, outfit co-occurrence, or weak textual supervision, and thus provide limited support for holistic outfit understanding. In this paper, we introduce FashionStylist, an expert-annotated benchmark for holistic and expert-level fashion understanding. Constructed through a dedicated fashion-expert annotation pipeline, FashionStylist provides professionally grounded annotations at both the item and outfit levels. It supports three representative tasks: outfit-to-item grounding, outfit completion, and outfit evaluation. These tasks cover realistic item recovery from complex outfits with layering and accessories, compatibility-aware composition beyond co-occurrence matching, and expert-level assessment of style, season, occasion, and overall coherence. Experimental results show that FashionStylist serves not only as a unified benchmark for multiple fashion tasks, but also as an effective training resource for improving grounding, completion, and outfit-level semantic evaluation in MLLM-based fashion systems.
>
---
#### [new 010] Frequency-Enhanced Diffusion Models: Curriculum-Guided Semantic Alignment for Zero-Shot Skeleton Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于零样本骨架动作识别任务，解决扩散模型在高频率动态上的过度平滑问题。提出FDSM方法，通过语义引导和课程学习提升动作识别性能。**

- **链接: [https://arxiv.org/pdf/2604.09063](https://arxiv.org/pdf/2604.09063)**

> **作者:** Yuxi Zhou; Zhengbo Zhang; Jingyu Pan; Zhiyu Lin; Zhigang Tu
>
> **摘要:** Human action recognition is pivotal in computer vision, with applications ranging from surveillance to human-robot interaction. Despite the effectiveness of supervised skeleton-based methods, their reliance on exhaustive annotation limits generalization to novel actions. Zero-Shot Skeleton Action Recognition (ZSAR) emerges as a promising paradigm, yet it faces challenges due to the spectral bias of diffusion models, which oversmooth high-frequency dynamics. Here, we propose Frequency-Aware Diffusion for Skeleton-Text Matching (FDSM), integrating a Semantic-Guided Spectral Residual Module, a Timestep-Adaptive Spectral Loss, and Curriculum-based Semantic Abstraction to address these challenges. Our approach effectively recovers fine-grained motion details, achieving state-of-the-art performance on NTU RGB+D, PKU-MMD, and Kinetics-skeleton datasets. Code has been made available at this https URL. Project homepage: this https URL
>
---
#### [new 011] MASS: Mesh-inellipse Aligned Deformable Surfel Splatting for Hand Reconstruction and Rendering from Egocentric Monocular Video
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手部重建与渲染任务，解决单目视频中高精度手部建模和实时渲染问题。提出MASS方法，通过可变形高斯表面点实现高效、逼真的手部重建。**

- **链接: [https://arxiv.org/pdf/2604.08943](https://arxiv.org/pdf/2604.08943)**

> **作者:** Haoyu Zhu; Yi Zhang; Lei Yao; Lap-pui Chau; Yi Wang
>
> **备注:** This paper has been accepted to CVM 2026 Journal Track and is under consideration for publication in IEEE TVCG
>
> **摘要:** Reconstructing high-fidelity 3D hands from egocentric monocular videos remains a challenge due to the limitations in capturing high-resolution geometry, hand-object interactions, and complex objects on hands. Additionally, existing methods often incur high computational costs, making them impractical for real-time applications. In this work, we propose Mesh-inellipse Aligned deformable Surfel Splatting (MASS) to address these challenges by leveraging a deformable 2D Gaussian Surfel representation. We introduce the mesh-aligned Steiner Inellipse and fractal densification for mesh-to-surfel conversion that initiates high-resolution 2D Gaussian surfels from coarse parametric hand meshes, providing surface representation with photorealistic rendering potential. Second, we propose Gaussian Surfel Deformation, which enables efficient modeling of hand deformations and personalized features by predicting residual updates to surfel attributes and introducing an opacity mask to refine geometry and texture without adaptive density control. In addition, we propose a two-stage training strategy and a novel binding loss to improve the optimization robustness and reconstruction quality. Extensive experiments on the ARCTIC dataset, the Hand Appearance dataset, and the Interhand2.6M dataset demonstrate that our model achieves superior reconstruction performance compared to state-of-the-art methods.
>
---
#### [new 012] Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory
- **分类: cs.CV**

- **简介: 该论文提出Matrix-Game 3.0，解决实时长序列视频生成中长期记忆与高分辨率的矛盾。通过数据、模型和推理优化，实现720p实时生成与稳定记忆一致性。**

- **链接: [https://arxiv.org/pdf/2604.08995](https://arxiv.org/pdf/2604.08995)**

> **作者:** Zile Wang; Zexiang Liu; Jaixing Li; Kaichen Huang; Baixin Xu; Fei Kang; Mengyin An; Peiyu Wang; Biao Jiang; Yichen Wei; Yidan Xietian; Jiangbo Pei; Liang Hu; Boyi Jiang; Hua Xue; Zidong Wang; Haofeng Sun; Wei Li; Wanli Ouyang; Xianglong He; Yang Liu; Yangguang Li; Yahui Zhou
>
> **备注:** Project page: this https URL
>
> **摘要:** With the advancement of interactive video generation, diffusion models have increasingly demonstrated their potential as world models. However, existing approaches still struggle to simultaneously achieve memory-enabled long-term temporal consistency and high-resolution real-time generation, limiting their applicability in real-world scenarios. To address this, we present Matrix-Game 3.0, a memory-augmented interactive world model designed for 720p real-time longform video generation. Building upon Matrix-Game 2.0, we introduce systematic improvements across data, model, and inference. First, we develop an upgraded industrial-scale infinite data engine that integrates Unreal Engine-based synthetic data, large-scale automated collection from AAA games, and real-world video augmentation to produce high-quality Video-Pose-Action-Prompt quadruplet data at scale. Second, we propose a training framework for long-horizon consistency: by modeling prediction residuals and re-injecting imperfect generated frames during training, the base model learns self-correction; meanwhile, camera-aware memory retrieval and injection enable the base model to achieve long horizon spatiotemporal consistency. Third, we design a multi-segment autoregressive distillation strategy based on Distribution Matching Distillation (DMD), combined with model quantization and VAE decoder pruning, to achieve efficient real-time inference. Experimental results show that Matrix-Game 3.0 achieves up to 40 FPS real-time generation at 720p resolution with a 5B model, while maintaining stable memory consistency over minute-long sequences. Scaling up to a 2x14B model further improves generation quality, dynamics, and generalization. Our approach provides a practical pathway toward industrial-scale deployable world models.
>
---
#### [new 013] Structure-Aware Fine-Grained Gaussian Splatting for Expressive Avatar Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D人体重建任务，旨在解决单目视频中精细表情和手势重建的问题。提出SFGS方法，结合时空特征与结构感知模块，提升重建精度与细节表现。**

- **链接: [https://arxiv.org/pdf/2604.09324](https://arxiv.org/pdf/2604.09324)**

> **作者:** Yuze Su; Hongsong Wang; Jie Gui; Liang Wang
>
> **备注:** The code is on Github: this https URL
>
> **摘要:** Reconstructing photorealistic and topology-aware human avatars from monocular videos remains a significant challenge in the fields of computer vision and graphics. While existing 3D human avatar modeling approaches can effectively capture body motion, they often fail to accurately model fine details such as hand movements and facial expressions. To address this, we propose Structure-aware Fine-grained Gaussian Splatting (SFGS), a novel method for reconstructing expressive and coherent full-body 3D human avatars from a monocular video sequence. The SFGS use both spatial-only triplane and time-aware hexplane to capture dynamic features across consecutive frames. A structure-aware gaussian module is designed to capture pose-dependent details in a spatially coherent manner and improve pose and texture expression. To better model hand deformations, we also propose a residual refinement module based on fine-grained hand reconstruction. Our method requires only a single-stage training and outperforms state-of-the-art baselines in both quantitative and qualitative evaluations, generating high-fidelity avatars with natural motion and fine details. The code is on Github: this https URL
>
---
#### [new 014] Off-the-shelf Vision Models Benefit Image Manipulation Localization
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文属于图像篡改定位任务，旨在解决传统方法与语义特征分离的问题。通过引入可训练适配器ReVi，利用现成视觉模型提升篡改定位效果。**

- **链接: [https://arxiv.org/pdf/2604.09096](https://arxiv.org/pdf/2604.09096)**

> **作者:** Zhengxuan Zhang; Keji Song; Junmin Hu; Ao Luo; Yuezun Li
>
> **摘要:** Image manipulation localization (IML) and general vision tasks are typically treated as two separate research directions due to the fundamental differences between manipulation-specific and semantic features. In this paper, however, we bridge this gap by introducing a fresh perspective: these two directions are intrinsically connected, and general semantic priors can benefit IML. Building on this insight, we propose a novel trainable adapter (named ReVi) that repurposes existing off-the-shelf general-purpose vision models (e.g., image generation and segmentation networks) for IML. Inspired by robust principal component analysis, the adapter disentangles semantic redundancy from manipulation-specific information embedded in these models and selectively enhances the latter. Unlike existing IML methods that require extensive model redesign and full retraining, our method relies on the off-the-shelf vision models with frozen parameters and only fine-tunes the proposed adapter. The experimental results demonstrate the superiority of our method, showing the potential for scalable IML frameworks.
>
---
#### [new 015] Online3R: Online Learning for Consistent Sequential Reconstruction Based on Geometry Foundation Model
- **分类: cs.CV**

- **简介: 该论文提出Online3R，解决序列重建中的不一致性问题。通过在线学习调整视觉提示，在保持基础模型能力的同时适应新场景。**

- **链接: [https://arxiv.org/pdf/2604.09480](https://arxiv.org/pdf/2604.09480)**

> **作者:** Shunkai Zhou; Zike Yan; Fei Xue; Dong Wu; Yuchen Deng; Hongbin Zha
>
> **摘要:** We present Online3R, a new sequential reconstruction framework that is capable of adapting to new scenes through online learning, effectively resolving inconsistency issues. Specifically, we introduce a set of learnable lightweight visual prompts into a pretrained, frozen geometry foundation model to capture the knowledge of new environments while preserving the fundamental capability of the foundation model for geometry prediction. To solve the problems of missing groundtruth and the requirement of high efficiency when updating these visual prompts at test time, we introduce a local-global self-supervised learning strategy by enforcing the local and global consistency constraints on predictions. The local consistency constraints are conducted on intermediate and previously local fused results, enabling the model to be trained with high-quality pseudo groundtruth signals; the global consistency constraints are operated on sparse keyframes spanning long distances rather than per frame, allowing the model to learn from a consistent prediction over a long trajectory in an efficient way. Our experiments demonstrate that Online3R outperforms previous state-of-the-art methods on various benchmarks. Project page: this https URL
>
---
#### [new 016] Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于红外-可见光图像融合任务，旨在解决多任务适应性不足的问题。提出CLDyN网络，通过闭环优化和语义补偿实现任务定制化融合。**

- **链接: [https://arxiv.org/pdf/2604.08924](https://arxiv.org/pdf/2604.08924)**

> **作者:** Zengyi Yang; Yu Liu; Juan Cheng; Zhiqin Zhu; Yafei Zhang; Huafeng Li
>
> **备注:** This paper has been accepted by CVPR 2026
>
> **摘要:** Infrared-visible image fusion aims to integrate complementary information for robust visual understanding, but existing fusion methods struggle with simultaneously adapting to multiple downstream tasks. To address this issue, we propose a Closed-Loop Dynamic Network (CLDyN) that can adaptively respond to the semantic requirements of diverse downstream tasks for task-customized image fusion. Specifically, CLDyN introduces a closed-loop optimization mechanism that establishes a semantic transmission chain to achieve explicit feedback from downstream tasks to the fusion network through a Requirement-driven Semantic Compensation (RSC) module. The RSC module leverages a Basis Vector Bank (BVB) and an Architecture-Adaptive Semantic Injection (A2SI) block to customize the network architecture according to task requirements, thereby enabling task-specific semantic compensation and allowing the fusion network to actively adapt to diverse tasks without retraining. To promote semantic compensation, a reward-penalty strategy is introduced to reward or penalize the RSC module based on task performance variations. Experiments on the M3FD, FMB, and VT5000 datasets demonstrate that CLDyN not only maintains high fusion quality but also exhibits strong multi-task adaptability. The code is available at this https URL.
>
---
#### [new 017] Detecting Diffusion-generated Images via Dynamic Assembly ForestsDetecting Diffusion-generated Images via Dynamic Assembly Forests
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像检测任务，旨在解决扩散模型生成图像的检测问题。提出一种新型动态组装森林模型（DAF），在参数少、计算成本低的前提下实现有效检测。**

- **链接: [https://arxiv.org/pdf/2604.09106](https://arxiv.org/pdf/2604.09106)**

> **作者:** Mengxin Fu; Yuezun Li
>
> **摘要:** Diffusion models are known for generating high-quality images, causing serious security concerns. To combat this, most efforts rely on deep neural networks (e.g., CNNs and Transformers), while largely overlooking the potential of traditional machine learning models. In this paper, we freshly investigate such alternatives and proposes a novel Dynamic Assembly Forest model (DAF) to detect diffusion-generated images. Built upon the deep forest paradigm, DAF addresses the inherent limitations in feature learning and scalable training, making it an effective diffusion-generated image detector. Compared to existing DNN-based methods, DAF has significantly fewer parameters, much lower computational cost, and can be deployed without GPUs, while achieving competitive performance under standard evaluation protocols. These results highlight the strong potential of the proposed method as a practical substitute for heavyweight DNN models in resource-constrained scenarios. Our code and models are available at this https URL.
>
---
#### [new 018] Globally Optimal Pose from Orthographic Silhouettes
- **分类: cs.CV**

- **简介: 该论文属于三维姿态估计任务，解决从轮廓中确定已知形状的全局最优姿态问题。通过分析轮廓面积和椭圆比，提出一种无需对应点的高效方法。**

- **链接: [https://arxiv.org/pdf/2604.09199](https://arxiv.org/pdf/2604.09199)**

> **作者:** Agniva Sengupta; Dilara Kuş; Jianning Li; Stefan Zachow
>
> **摘要:** We solve the problem of determining the pose of known shapes in $\mathbb{R}^3$ from their unoccluded silhouettes. The pose is determined up to global optimality using a simple yet under-explored property of the area-of-silhouette: its continuity w.r.t trajectories in the rotation space. The proposed method utilises pre-computed silhouette-signatures, modelled as a response surface of the area-of-silhouettes. Querying this silhouette-signature response surface for pose estimation leads to a strong branching of the rotation search space, making resolution-guided candidate search feasible. Additionally, we utilise the aspect ratio of 2D ellipses fitted to projected silhouettes as an auxiliary global shape signature to accelerate the pose search. This combined strategy forms the first method to efficiently estimate globally optimal pose from just the silhouettes, without being guided by correspondences, for any shape, irrespective of its convexity and genus. We validate our method on synthetic and real examples, demonstrating significantly improved accuracy against comparable approaches. Code, data, and supplementary in: this https URL
>
---
#### [new 019] MV3DIS: Multi-View Mask Matching via 3D Guides for Zero-Shot 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D实例分割任务，解决零样本下依赖3D标注的问题。提出MV3DIS框架，结合3D先验和多视角一致性，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.08916](https://arxiv.org/pdf/2604.08916)**

> **作者:** Yibo Zhao; Yigong Zhang; Jin Xie
>
> **摘要:** Conventional 3D instance segmentation methods rely on labor-intensive 3D annotations for supervised training, which limits their scalability and generalization to novel objects. Recent approaches leverage multi-view 2D masks from the Segment Anything Model (SAM) to guide the merging of 3D geometric primitives, thereby enabling zero-shot 3D instance segmentation. However, these methods typically process each frame independently and rely solely on 2D metrics, such as SAM prediction scores, to produce segmentation maps. This design overlooks multi-view correlations and inherent 3D priors, leading to inconsistent 2D masks across views and ultimately fragmented 3D segmentation. In this paper, we propose MV3DIS, a coarse-to-fine framework for zero-shot 3D instance segmentation that explicitly incorporates 3D priors. Specifically, we introduce a 3D-guided mask matching strategy that uses coarse 3D segments as a common reference to match 2D masks across views and consolidates multi-view mask consistency via 3D coverage distributions. Guided by these view-consistent 2D masks, the coarse 3D segments are further refined into precise 3D instances. Additionally, we introduce a depth consistency weighting scheme that quantifies projection reliability to suppress ambiguities from inter-object occlusions, thereby improving the robustness of 3D-to-2D correspondence. Extensive experiments on the ScanNetV2, ScanNet200, ScanNet++, Replica, and Matterport3D datasets demonstrate the effectiveness of MV3DIS, which achieves superior performance over previous methods
>
---
#### [new 020] Nested Radially Monotone Polar Occupancy Estimation: Clinically-Grounded Optic Disc and Cup Segmentation for Glaucoma Screening
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决青光眼筛查中视盘和视杯分割的临床有效性问题。提出NPS-Net框架，确保分割结果符合解剖结构要求并提升精度。**

- **链接: [https://arxiv.org/pdf/2604.09062](https://arxiv.org/pdf/2604.09062)**

> **作者:** Rimsa Goperma; Rojan Basnet; Liang Zhao
>
> **摘要:** Valid segmentation of the optic disc (OD) and optic cup (OC) from fundus photographs is essential for glaucoma screening. Unfortunately, existing deep learning methods do not guarantee clinical validness including star-convexity and nested structure of OD and OC, resulting corruption in diagnostic metric, especially under cross-dataset domain shift. To adress this issue, this paper proposed NPS-Net (Nested Polar Shape Network), the first framework that formulates the OD/OC segmentation as nested radially monotone polar occupancy this http URL output representation can guarantee the aforementioned clinical validness and achieve high accuracy. Evaluated across seven public datasets, NPS-Net shows strong zero-shot generalization. On RIM-ONE, it maintains 100% anatomical validity and improves Cup Dice by 12.8% absolute over the best baseline, reducing vCDR MAE by over 56%. On PAPILA, it achieves Disc Dice of 0.9438 and Disc HD95 of 2.78 px, an 83% reduction over the best competing method.
>
---
#### [new 021] LMGenDrive: Bridging Multimodal Understanding and Generative World Modeling for End-to-End Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决长尾和开放场景下的泛化问题。通过融合多模态理解和生成模型，提出LMGenDrive框架，实现端到端驾驶决策。**

- **链接: [https://arxiv.org/pdf/2604.08719](https://arxiv.org/pdf/2604.08719)**

> **作者:** Hao Shao; Letian Wang; Yang Zhou; Yuxuan Hu; Zhuofan Zong; Steven L. Waslander; Wei Zhan; Hongsheng Li
>
> **摘要:** Recent years have seen remarkable progress in autonomous driving, yet generalization to long-tail and open-world scenarios remains a major bottleneck for large-scale deployment. To address this challenge, some works use LLMs and VLMs for vision-language understanding and reasoning, enabling vehicles to interpret rare and safety-critical situations when generating actions. Others study generative world models to capture the spatio-temporal evolution of driving scenes, allowing agents to imagine possible futures before acting. Inspired by human intelligence, which unifies understanding and imagination, we explore a unified model for autonomous driving. We present LMGenDrive, the first framework that combines LLM-based multimodal understanding with generative world models for end-to-end closed-loop driving. Given multi-view camera inputs and natural-language instructions, LMGenDrive generates both future driving videos and control signals. This design provides complementary benefits: video prediction improves spatio-temporal scene modeling, while the LLM contributes strong semantic priors and instruction grounding from large-scale pretraining. We further propose a progressive three-stage training strategy, from vision pretraining to multi-step long-horizon driving, to improve stability and performance. LMGenDrive supports both low-latency online planning and autoregressive offline video generation. Experiments show that it significantly outperforms prior methods on challenging closed-loop benchmarks, with clear gains in instruction following, spatio-temporal understanding, and robustness to rare scenarios. These results suggest that unifying multimodal understanding and generation is a promising direction for more generalizable and robust embodied decision-making systems.
>
---
#### [new 022] InstrAct: Towards Action-Centric Understanding in Instructional Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决 instructional 视频中动作识别与时序关系建模的问题。提出 InstrAct 框架，通过对比学习、动态时间对齐和掩码动作建模提升动作中心表示。**

- **链接: [https://arxiv.org/pdf/2604.08762](https://arxiv.org/pdf/2604.08762)**

> **作者:** Zhuoyi Yang; Jiapeng Yu; Reuben Tan; Boyang Li; Huijuan Xu
>
> **摘要:** Understanding instructional videos requires recognizing fine-grained actions and modeling their temporal relations, which remains challenging for current Video Foundation Models (VFMs). This difficulty stems from noisy web supervision and a pervasive "static bias", where models rely on objects rather than motion cues. To address this, we propose InstrAction, a pretraining framework for instructional videos' action-centric representations. We first introduce a data-driven strategy, which filters noisy captions and generates action-centric hard negatives to disentangle actions from objects during contrastive learning. At the visual feature level, an Action Perceiver extracts motion-relevant tokens from redundant video encodings. Beyond contrastive learning, we introduce two auxiliary objectives: Dynamic Time Warping alignment (DTW-Align) for modeling sequential temporal structure, and Masked Action Modeling (MAM) for strengthening cross-modal grounding. Finally, we introduce the InstrAct Bench to evaluate action-centric understanding, where our method consistently outperforms state-of-the-art VFMs on semantic reasoning, procedural logic, and fine-grained retrieval tasks.
>
---
#### [new 023] EgoTL: Egocentric Think-Aloud Chains for Long-Horizon Tasks
- **分类: cs.CV**

- **简介: 该论文提出EgoTL，用于解决长时序任务中视觉语言模型的推理与空间定位问题，通过收集精确的思维链和空间信息提升模型表现。**

- **链接: [https://arxiv.org/pdf/2604.09535](https://arxiv.org/pdf/2604.09535)**

> **作者:** Lulin Liu; Dayou Li; Yiqing Liang; Sicong Jiang; Hitesh Vijay; Hezhen Hu; Xuhai Xu; Zirui Liu; Srinivas Shakkottai; Manling Li; Zhiwen Fan
>
> **备注:** this https URL
>
> **摘要:** Large foundation models have made significant advances in embodied intelligence, enabling synthesis and reasoning over egocentric input for household tasks. However, VLM-based auto-labeling is often noisy because the primary data sources lack accurate human action labels, chain-of-thought (CoT), and spatial annotations; these errors are amplified during long-horizon spatial instruction following. These issues stem from insufficient coverage of minute-long, daily household planning tasks and from inaccurate spatial grounding. As a result, VLM reasoning chains and world-model synthesis can hallucinate objects, skip steps, or fail to respect real-world physical attributes. To address these gaps, we introduce EgoTL. EgoTL builds a think-aloud capture pipeline for egocentric data. It uses a say-before-act protocol to record step-by-step goals and spoken reasoning with word-level timestamps, then calibrates physical properties with metric-scale spatial estimators, a memory-bank walkthrough for scene context, and clip-level tags for navigation instructions and detailed manipulation actions. With EgoTL, we are able to benchmark VLMs and World Models on six task dimensions from three layers and long-horizon generation over minute-long sequences across over 100 daily household tasks. We find that foundation models still fall short as egocentric assistants or open-world simulators. Finally, we finetune foundation models with human CoT aligned with metric labels on the training split of EgoTL, which improves long-horizon planning and reasoning, step-wise reasoning, instruction following, and spatial grounding.
>
---
#### [new 024] Efficient Spatial-Temporal Focal Adapter with SSM for Temporal Action Detection
- **分类: cs.CV**

- **简介: 该论文属于视频动作检测任务，解决长视频中特征冗余和全局依赖建模不足的问题。提出ESTF适配器与TB-SSM模块，提升检测性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.09164](https://arxiv.org/pdf/2604.09164)**

> **作者:** Yicheng Qiu; Keiji Yanai
>
> **备注:** ICME2026
>
> **摘要:** Temporal human action detection aims to identify and localize action segments within untrimmed videos, serving as a pivotal task in video understanding. Despite the progress achieved by prior architectures like CNN and Transformer models, these continue to struggle with feature redundancy and degraded global dependency modeling capabilities when applied to long video sequences. These limitations severely constrain their scalability in real-world video analysis. State Space Models (SSMs) offer a promising alternative with linear long-term modeling and robust global temporal reasoning capabilities. Rethinking the application of SSMs in temporal modeling, this research constructs a novel framework for video human action detection. Specifically, we introduce the Efficient Spatial-Temporal Focal (ESTF) Adapter into the pre-trained layers. This module integrates the advantages of our proposed Temporal Boundary-aware SSM(TB-SSM) for temporal feature modeling with efficient processing of spatial features. We perform comprehensive and quantitative analyses across multiple benchmarks, comparing our proposed method against previous SSM-based and other structural methods. Extensive experiments demonstrate that our improved strategy significantly enhances both localization performance and robustness, validating the effectiveness of our proposed method.
>
---
#### [new 025] ViSAGE @ NTIRE 2026 Challenge on Video Saliency Prediction
- **分类: cs.CV**

- **简介: 该论文属于视频显著性预测任务，旨在提升视频中显著区域的检测效果。提出ViSAGE框架，通过多专家集成融合不同先验偏差，增强模型性能。**

- **链接: [https://arxiv.org/pdf/2604.08613](https://arxiv.org/pdf/2604.08613)**

> **作者:** Kun Wang; Yupeng Hu; Zhiran Li; Hao Liu; Qianlong Xiang; Liqiang Nie
>
> **摘要:** In this report, we present our champion solution for the NTIRE 2026 Challenge on Video Saliency Prediction held in conjunction with CVPR 2026. To exploit complementary inductive biases for video saliency, we propose Video Saliency with Adaptive Gated Experts (ViSAGE), a multi-expert ensemble framework. Each specialized decoder performs adaptive gating and modulation to refine spatio-temporal features. The complementary predictions from different experts are then fused at inference. ViSAGE thereby aggregates diverse inductive biases to capture complex spatio-temporal saliency cues in videos. On the Private Test set, ViSAGE ranked first on two out of four evaluation metrics, and outperformed most competing solutions on the other two metrics, demonstrating its effectiveness and generalization ability. Our code has been released at this https URL.
>
---
#### [new 026] Harnessing Weak Pair Uncertainty for Text-based Person Search
- **分类: cs.CV**

- **简介: 该论文属于文本检索人物任务，解决弱正样本对匹配问题。提出一种考虑不确定性的方法，提升模型对弱正样本的利用效果。**

- **链接: [https://arxiv.org/pdf/2604.08877](https://arxiv.org/pdf/2604.08877)**

> **作者:** Jintao Sun; Zhedong Zheng; Gangyi Ding
>
> **备注:** 39 pages, 15 tables, 7 figures
>
> **摘要:** In this paper, we study the text-based person search, which is to retrieve the person of interest via natural language description. Prevailing methods usually focus on the strict one-to-one correspondence pair matching between the visual and textual modality, such as contrastive learning. However, such a paradigm unintentionally disregards the weak positive image-text pairs, which are of the same person but the text descriptions are annotated from different views (cameras). To take full use of weak positives, we introduce an uncertainty-aware method to explicitly estimate image-text pair uncertainty, and incorporate the uncertainty into the optimization procedure in a smooth manner. Specifically, our method contains two modules: uncertainty estimation and uncertainty regularization. (1) Uncertainty estimation is to obtain the relative confidence on the given positive pairs; (2) Based on the predicted uncertainty, we propose the uncertainty regularization to adaptively adjust loss weight. Additionally, we introduce a group-wise image-text matching loss to further facilitate the representation space among the weak pairs. Compared with existing methods, the proposed method explicitly prevents the model from pushing away potentially weak positive candidates. Extensive experiments on three widely-used datasets, .e.g, CUHK-PEDES, RSTPReid and ICFG-PEDES, verify the mAP improvement of our method against existing competitive methods +3.06%, +3.55% and +6.94%, respectively.
>
---
#### [new 027] BIAS: A Biologically Inspired Algorithm for Video Saliency Detection
- **分类: cs.CV**

- **简介: 该论文提出BIAS算法，用于视频显著性检测任务，解决动态视觉注意力分配问题。通过结合静态与运动信息，实现快速、生物启发的显著区域识别。**

- **链接: [https://arxiv.org/pdf/2604.08858](https://arxiv.org/pdf/2604.08858)**

> **作者:** Zhao-ji Zhang; Ya-tang Li
>
> **摘要:** We present BIAS, a fast, biologically inspired model for dynamic visual saliency detection in continuous video streams. Building on the Itti--Koch framework, BIAS incorporates a retina-inspired motion detector to extract temporal features, enabling the generation of saliency maps that integrate both static and motion information. Foci of attention (FOAs) are identified using a greedy multi-Gaussian peak-fitting algorithm that balances winner-take-all competition with information maximization. BIAS detects salient regions with millisecond-scale latency and outperforms heuristic-based approaches and several deep-learning models on the DHF1K dataset, particularly in videos dominated by bottom-up attention. Applied to traffic accident analysis, BIAS demonstrates strong real-world utility, achieving state-of-the-art performance in cause-effect recognition and anticipating accidents up to 0.72 seconds before manual annotation with reliable accuracy. Overall, BIAS bridges biological plausibility and computational efficiency to achieve interpretable, high-speed dynamic saliency detection.
>
---
#### [new 028] Towards Responsible Multimodal Medical Reasoning via Context-Aligned Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于医学多模态推理任务，旨在解决模型依赖单一模态导致结论不准确的问题。通过引入上下文对齐框架，整合多种辅助信号提升诊断可靠性。**

- **链接: [https://arxiv.org/pdf/2604.08815](https://arxiv.org/pdf/2604.08815)**

> **作者:** Sumra Khan; Sagar Chhabriya; Aizan Zafar; Sheeraz Arif; Amgad Muneer; Anas Zafar; Shaina Raza; Rizwan Qureshi
>
> **摘要:** Medical vision-language models (VLMs) show strong performance on radiology tasks but often produce fluent yet weakly grounded conclusions due to over-reliance on a dominant modality. We introduce a context-aligned reasoning framework that enforces agreement across heterogeneous clinical evidence before generating diagnostic conclusions. The proposed approach augments a frozen VLM with structured contextual signals derived from radiomic statistics, explainability activations, and vocabulary-grounded semantic cues. Instead of producing free-form responses, the model generates structured outputs containing supporting evidence, uncertainty estimates, limitations, and safety notes. We observe that auxiliary signals alone provide limited benefit; performance gains emerge only when these signals are integrated through contextual verification. Experiments on chest X-ray datasets demonstrate that context alignment improves discriminative performance (AUC 0.918 to 0.925) while maintaining calibrated uncertainty. The framework also substantially reduces hallucinated keywords (1.14 to 0.25) and produces more concise reasoning explanations (19.4 to 15.3 words) without increasing model confidence (0.70 to 0.68). Cross-dataset evaluation on CheXpert further reveals that modality informativeness significantly influences reasoning behavior. These results suggest that enforcing multi-evidence agreement improves both reliability and trustworthiness in medical multimodal reasoning, while preserving the underlying model architecture.
>
---
#### [new 029] StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决流式视频中代理记忆存储成本高的问题。通过提出StreamMeCo框架，实现记忆压缩，提升检索效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2604.09000](https://arxiv.org/pdf/2604.09000)**

> **作者:** Junxi Wang; Te Sun; Jiayi Zhu; Junxian Li; Haowen Xu; Zichen Wen; Xuming Hu; Zhiyu Li; Linfeng Zhang
>
> **备注:** 2026ACL Findings
>
> **摘要:** Vision agent memory has shown remarkable effectiveness in streaming video understanding. However, storing such memory for videos incurs substantial memory overhead, leading to high costs in both storage and computation. To address this issue, we propose StreamMeCo, an efficient Stream Agent Memory Compression framework. Specifically, based on the connectivity of the memory graph, StreamMeCo introduces edge-free minmax sampling for the isolated nodes and an edge-aware weight pruning for connected nodes, evicting the redundant memory nodes while maintaining the accuracy. In addition, we introduce a time-decay memory retrieval mechanism to further eliminate the performance degradation caused by memory compression. Extensive experiments on three challenging benchmark datasets (M3-Bench-robot, M3-Bench-web and Video-MME-Long) demonstrate that under 70% memory graph compression, StreamMeCo achieves a 1.87* speedup in memory retrieval while delivering an average accuracy improvement of 1.0%. Our code is available at this https URL.
>
---
#### [new 030] MixFlow: Mixed Source Distributions Improve Rectified Flows
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型采样慢的问题。通过引入混合源分布，减少生成路径曲率，提升采样效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2604.09181](https://arxiv.org/pdf/2604.09181)**

> **作者:** Nazir Nayal; Christopher Wewer; Jan Eric Lenssen
>
> **摘要:** Diffusion models and their variations, such as rectified flows, generate diverse and high-quality images, but they are still hindered by slow iterative sampling caused by the highly curved generative paths they learn. An important cause of high curvature, as shown by previous work, is independence between the source distribution (standard Gaussian) and the data distribution. In this work, we tackle this limitation by two complementary contributions. First, we attempt to break away from the standard Gaussian assumption by introducing $\kappa\texttt{-FC}$, a general formulation that conditions the source distribution on an arbitrary signal $\kappa$ that aligns it better with the data distribution. Then, we present MixFlow, a simple but effective training strategy that reduces the generative path curvatures and considerably improves sampling efficiency. MixFlow trains a flow model on linear mixtures of a fixed unconditional distribution and a $\kappa\texttt{-FC}$-based distribution. This simple mixture improves the alignment between the source and data, provides better generation quality with less required sampling steps, and accelerates the training convergence considerably. On average, our training procedure improves the generation quality by 12\% in FID compared to standard rectified flow and 7\% compared to previous baselines under a fixed sampling budget. Code available at: $\href{this https URL}{this https URL}$
>
---
#### [new 031] TinyNeRV: Compact Neural Video Representations via Capacity Scaling, Distillation, and Low-Precision Inference
- **分类: cs.CV**

- **简介: 该论文属于视频压缩任务，旨在解决紧凑神经视频表示的效率问题。通过架构缩放、知识蒸馏和低精度推理，提升小型模型的性能与实用性。**

- **链接: [https://arxiv.org/pdf/2604.09220](https://arxiv.org/pdf/2604.09220)**

> **作者:** Muhammad Hannan Akhtar; Ihab Amer; Tamer Shanableh
>
> **备注:** Submitted to "Computers and Electrical Engineering", Elsevier
>
> **摘要:** Implicit neural video representations encode entire video sequences within the parameters of a neural network and enable constant time frame reconstruction. Recent work on Neural Representations for Videos (NeRV) has demonstrated competitive reconstruction performance while avoiding the sequential decoding process of conventional video codecs. However, most existing studies focus on moderate or high capacity models, leaving the behavior of extremely compact configurations required for constrained environments insufficiently explored. This paper presents a systematic study of tiny NeRV architectures designed for efficient deployment. Two lightweight configurations, NeRV-T and NeRV-T+, are introduced and evaluated across multiple video datasets in order to analyze how aggressive capacity reduction affects reconstruction quality, computational complexity, and decoding throughput. Beyond architectural scaling, the work investigates strategies for improving the performance of compact models without increasing inference cost. Knowledge distillation with frequency-aware focal supervision is explored to enhance reconstruction fidelity in low-capacity networks. In addition, the impact of lowprecision inference is examined through both post training quantization and quantization aware training to study the robustness of tiny models under reduced numerical precision. Experimental results demonstrate that carefully designed tiny NeRV variants can achieve favorable quality efficiency trade offs while substantially reducing parameter count, computational cost, and memory requirements. These findings provide insight into the practical limits of compact neural video representations and offer guidance for deploying NeRV style models in resource constrained and real-time environments. The official implementation is available at https: //github.com/HannanAkhtar/TinyNeRV-Implementation.
>
---
#### [new 032] UniSemAlign: Text-Prototype Alignment with a Foundation Encoder for Semi-Supervised Histopathology Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督病理分割任务，旨在解决标注数据少和伪标签不可靠的问题。提出UniSemAlign框架，通过文本与视觉对齐增强分割效果。**

- **链接: [https://arxiv.org/pdf/2604.09169](https://arxiv.org/pdf/2604.09169)**

> **作者:** Le-Van Thai; Tien Dat Nguyen; Hoai Nhan Pham; Lan Anh Dinh Thi; Duy-Dong Nguyen; Ngoc Lam Quang Bui
>
> **备注:** Accepted at CVPR 2026 Workshop. 11 pages, 5 figures, 4 tables
>
> **摘要:** Semi-supervised semantic segmentation in computational pathology remains challenging due to scarce pixel-level annotations and unreliable pseudo-label supervision. We propose UniSemAlign, a dual-modal semantic alignment framework that enhances visual segmentation by injecting explicit class-level structure into pixel-wise learning. Built upon a pathology-pretrained Transformer encoder, UniSemAlign introduces complementary prototype-level and text-level alignment branches in a shared embedding space, providing structured guidance that reduces class ambiguity and stabilizes pseudo-label refinement. The aligned representations are fused with visual predictions to generate more reliable supervision for unlabeled histopathology images. The framework is trained end-to-end with supervised segmentation, cross-view consistency, and cross-modal alignment objectives. Extensive experiments on the GlaS and CRAG datasets demonstrate that UniSemAlign substantially outperforms recent semi-supervised baselines under limited supervision, achieving Dice improvements of up to 2.6% on GlaS and 8.6% on CRAG with only 10% labeled data, and strong improvements at 20% supervision. Code is available at: this https URL
>
---
#### [new 033] Hitem3D 2.0: Multi-View Guided Native 3D Texture Generation
- **分类: cs.CV**

- **简介: 该论文属于3D纹理生成任务，解决纹理覆盖不全、视角不一致和几何与纹理错位问题。提出Hitem3D 2.0框架，结合多视角生成先验和原生3D纹理表示，提升纹理质量与一致性。**

- **链接: [https://arxiv.org/pdf/2604.09231](https://arxiv.org/pdf/2604.09231)**

> **作者:** Huiang He; Shengchu Zhao; Jianwen Huang; Jie Li; Jiaqi Wu; Hu Zhang; Pei Tang; Heliang Zheng; Yukun Li; Rongfei Jia
>
> **备注:** 13 pages
>
> **摘要:** Although recent advances have improved the quality of 3D texture generation, existing methods still struggle with incomplete texture coverage, cross-view inconsistency, and misalignment between geometry and texture. To address these limitations, we propose Hitem3D 2.0, a multi-view guided native 3D texture generation framework that enhances texture quality through the integration of 2D multi-view generation priors and native 3D texture representations. Hitem3D 2.0 comprises two key components: a multi-view synthesis framework and a native 3D texture generation model. The multi-view generation is built upon a pre-trained image editing backbone and incorporates plug-and-play modules that explicitly promote geometric alignment, cross-view consistency, and illumination uniformity, thereby enabling the synthesis of high-fidelity multi-view images. Conditioned on the generated views and 3D geometry, the native 3D texture generation model projects multi-view textures onto 3D surfaces while plausibly completing textures in unseen regions. Through the integration of multi-view consistency constraints with native 3D texture modeling, Hitem3D 2.0 significantly improves texture completeness, cross-view coherence, and geometric alignment. Experimental results demonstrate that Hitem3D 2.0 outperforms existing methods in terms of texture detail, fidelity, consistency, coherence, and alignment.
>
---
#### [new 034] CatalogStitch: Dimension-Aware and Occlusion-Preserving Object Compositing for Catalog Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，解决 catalog 图像生成中因产品尺寸差异和遮挡导致的需手动调整问题。提出 CatalogStitch 方法，自动处理尺寸适配和遮挡恢复。**

- **链接: [https://arxiv.org/pdf/2604.08836](https://arxiv.org/pdf/2604.08836)**

> **作者:** Sanyam Jain; Pragya Kandari; Manit Singhal; He Zhang; Soo Ye Kim
>
> **备注:** CVPR 2026 HiGen Workshop. Project page, this https URL
>
> **摘要:** Generative object compositing methods have shown remarkable ability to seamlessly insert objects into scenes. However, when applied to real-world catalog image generation, these methods require tedious manual intervention: users must carefully adjust masks when product dimensions differ, and painstakingly restore occluded elements post-generation. We present CatalogStitch, a set of model-agnostic techniques that automate these corrections, enabling user-friendly content creation. Our dimension-aware mask computation algorithm automatically adapts the target region to accommodate products with different dimensions; users simply provide a product image and background, without manual mask adjustments. Our occlusion-aware hybrid restoration method guarantees pixel-perfect preservation of occluding elements, eliminating post-editing workflows. We additionally introduce CatalogStitch-Eval, a 58-example benchmark covering aspect-ratio mismatch and occlusion-heavy catalog scenarios, together with supplementary PDF and HTML viewers. We evaluate our techniques with three state-of-the-art compositing models (ObjectStitch, OmniPaint, and InsertAnything), demonstrating consistent improvements across diverse catalog scenarios. By reducing manual intervention and automating tedious corrections, our approach transforms generative compositing into a practical, human-friendly tool for production catalog workflows.
>
---
#### [new 035] Vision Transformers for Preoperative CT-Based Prediction of Histopathologic Chemotherapy Response Score in High-Grade Serous Ovarian Carcinoma
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在通过CT图像和临床数据预判卵巢癌化疗反应评分，解决术前评估治疗效果的问题。研究提出一种多模态深度学习框架进行预测。**

- **链接: [https://arxiv.org/pdf/2604.09197](https://arxiv.org/pdf/2604.09197)**

> **作者:** Francesca Fati; Felipe Coutinho; Marika Reinius; Marina Rosanu; Gabriel Funingana; Luigi De Vitis; Gabriella Schivardi; Hannah Clayton; Alice Traversa; Zeyu Gao; Guilherme Penteado; Shangqi Gao; Francesco Pastori; Ramona Woitek; Maria Cristina Ghioni; Giovanni Damiano Aletti; Mercedes Jimenez-Linan; Sarah Burge; Nicoletta Colombo; Evis Sala; Maria Francesca Spadea; Timothy L. Kline; James D. Brenton; Jaime Cardoso; Francesco Multinu; Elena De Momi; Mireia Crispin-Ortuzar; Ines P. Machado
>
> **摘要:** Purpose. High-grade serous ovarian carcinoma (HGSOC) is characterized by pronounced biological and spatial heterogeneity and is frequently diagnosed at an advanced stage. Neoadjuvant chemotherapy (NACT) followed by delayed primary surgery is commonly employed in patients unsuitable for primary cytoreduction. The Chemotherapy Response Score (CRS) is a validated histopathological biomarker of response to NACT, but it is only available postoperatively. In this study, we investigate whether pre-treatment computed tomography (CT) imaging and clinical data can be used to predict CRS as an investigational decision-support adjunct to inform multidisciplinary team (MDT) discussions regarding expected treatment response. Methods. We proposed a 2.5D multimodal deep learning framework that processes lesion-dense omental slices using a pre-trained Vision Transformer encoder and integrates the resulting visual representations with clinical variables through an intermediate fusion module to predict CRS. Results. Our multimodal model, integrating imaging and clinical data, achieved a ROC-AUC of 0.95 alongside 95% accuracy and 80% precision on the internal test cohort (IEO, n=41 patients). On the external test set (OV04, n=70 patients), it achieved a ROC-AUC of 0.68, alongside 67% accuracy and 75% precision. Conclusion. These preliminary results demonstrate the feasibility of transformer-based deep learning for preoperative prediction of CRS in HGSOC using routine clinical data and CT imaging. As an investigational, pre-treatment decision-support tool, this approach may assist MDT discussions by providing early, non-invasive estimates of treatment response.
>
---
#### [new 036] Geometry Reinforced Efficient Attention Tuning Equipped with Normals for Robust Stereo Matching
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在解决合成到真实场景的泛化问题。通过引入法线信息和注意力机制，提升模型在复杂区域的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2604.09142](https://arxiv.org/pdf/2604.09142)**

> **作者:** Jiahao Li; Xinhong Chen; Zhengmin Jiang; Cheng Huang; Yung-Hui Li; Jianping Wang
>
> **摘要:** Despite remarkable advances in image-driven stereo matching over the past decade, Synthetic-to-Realistic Zero-Shot (Syn-to-Real) generalization remains an open challenge. This suboptimal generalization performance mainly stems from cross-domain shifts and ill-posed ambiguities inherent in image textures, particularly in occluded, textureless, repetitive, and non-Lambertian (specular/transparent) regions. To improve Syn-to-Real generalization, we propose GREATEN, a framework that incorporates surface normals as domain-invariant, object-intrinsic, and discriminative geometric cues to compensate for the limitations of image textures. The proposed framework consists of three key components. First, a Gated Contextual-Geometric Fusion (GCGF) module adaptively suppresses unreliable contextual cues in image features and fuses the filtered image features with normal-driven geometric features to construct domain-invariant and discriminative contextual-geometric representations. Second, a Specular-Transparent Augmentation (STA) strategy improves the robustness of GCGF against misleading visual cues in non-Lambertian regions. Third, sparse attention designs preserve the fine-grained global feature extraction capability of GREAT-Stereo for handling occlusion and texture-related ambiguities while substantially reducing computational overhead, including Sparse Spatial (SSA), Sparse Dual-Matching (SDMA), and Simple Volume (SVA) attentions. Trained exclusively on synthetic data such as SceneFlow, GREATEN-IGEV achieves outstanding Syn-to-Real performance. Specifically, it reduces errors by 30% on ETH3D, 8.5% on the non-Lambertian Booster, and 14.1% on KITTI-2015, compared to FoundationStereo, Monster-Stereo, and DEFOM-Stereo, respectively. In addition, GREATEN-IGEV runs 19.2% faster than GREAT-IGEV and supports high-resolution (3K) inference on Middlebury with disparity ranges up to 768.
>
---
#### [new 037] LuMon: A Comprehensive Benchmark and Development Suite with Novel Datasets for Lunar Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于月球单目深度估计任务，旨在解决地球模型在月球环境中的适应性问题。提出LuMon基准和新数据集，评估并分析现有方法的局限性。**

- **链接: [https://arxiv.org/pdf/2604.09352](https://arxiv.org/pdf/2604.09352)**

> **作者:** Aytaç Sekmen; Fatih Emre Gunes; Furkan Horoz; Hüseyin Umut Işık; Mehmet Alp Ozaydin; Onur Altay Topaloglu; Şahin Umutcan Üstündaş; Yurdasen Alp Yeni; Halil Ersin Soken; Erol Sahin; Ramazan Gokberk Cinbis; Sinan Kalkan
>
> **备注:** This paper will be published in CVPRW2026
>
> **摘要:** Monocular Depth Estimation (MDE) is crucial for autonomous lunar rover navigation using electro-optical cameras. However, deploying terrestrial MDE networks to the Moon brings a severe domain gap due to harsh shadows, textureless regolith, and zero atmospheric scattering. Existing evaluations rely on analogs that fail to replicate these conditions and lack actual metric ground truth. To address this, we present LuMon, a comprehensive benchmarking framework to evaluate MDE methods for lunar exploration. We introduce novel datasets featuring high-quality stereo ground truth depth from the real Chang'e-3 mission and the CHERI dark analog dataset. Utilizing this framework, we conduct a systematic zero-shot evaluation of state-of-the-art architectures across synthetic, analog, and real datasets. We rigorously assess performance against mission critical challenges like craters, rocks, extreme shading, and varying depth ranges. Furthermore, we establish a sim-to-real domain adaptation baseline by fine tuning a foundation model on synthetic data. While this adaptation yields drastic in-domain performance gains, it exhibits minimal generalization to authentic lunar imagery, highlighting a persistent cross-domain transfer gap. Our extensive analysis reveals the inherent limitations of current networks and sets a standard foundation to guide future advancements in extraterrestrial perception and domain adaptation.
>
---
#### [new 038] From Frames to Events: Rethinking Evaluation in Human-Centric Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决传统帧级评估与实际事件检测不匹配的问题。通过引入事件级评估标准和新方法提升异常事件的检测精度。**

- **链接: [https://arxiv.org/pdf/2604.09327](https://arxiv.org/pdf/2604.09327)**

> **作者:** Narges Rashvand; Shanle Yao; Armin Danesh Pazho; Babak Rahimi Ardabili; Hamed Tabkhi
>
> **摘要:** Pose-based Video Anomaly Detection (VAD) has gained significant attention for its privacy-preserving nature and robustness to environmental variations. However, traditional frame-level evaluations treat video as a collection of isolated frames, fundamentally misaligned with how anomalies manifest and are acted upon in the real world. In operational surveillance systems, what matters is not the flagging of individual frames, but the reliable detection, localization, and reporting of a coherent anomalous event, a contiguous temporal episode with an identifiable onset and duration. Frame-level metrics are blind to this distinction, and as a result, they systematically overestimate model performance for any deployment that requires actionable, event-level alerts. In this work, we propose a shift toward an event-centric perspective in VAD. We first audit widely used VAD benchmarks, including SHT[19], CHAD[6], NWPUC[4], and HuVAD[25], to characterize their event structure. We then introduce two strategies for temporal event localization: a score-refinement pipeline with hierarchical Gaussian smoothing and adaptive binarization, and an end-to-end Dual-Branch Model that directly generates event-level detections. Finally, we establish the first event-based evaluation standard for VAD by adapting Temporal Action Localization metrics, including tIoU-based event matching and multi-threshold F1 evaluation. Our results quantify a substantial performance gap: while all SoTA models achieve frame-level AUC-ROC exceeding 52% on the NWPUC[4], their event-level localization precision falls below 10% even at a minimal tIoU=0.2, with an average event-level F1 of only 0.11 across all thresholds. The code base for this work is available at this https URL.
>
---
#### [new 039] SynFlow: Scaling Up LiDAR Scene Flow Estimation with Synthetic Data
- **分类: cs.CV**

- **简介: 该论文属于3D动态感知任务，解决真实场景运动标注稀缺问题。通过合成数据生成方法SynFlow，构建大规模LiDAR场景流数据集，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.09411](https://arxiv.org/pdf/2604.09411)**

> **作者:** Qingwen Zhang; Xiaomeng Zhu; Chenhan Jiang; Patric Jensfelt
>
> **摘要:** Reliable 3D dynamic perception requires models that can anticipate motion beyond predefined categories, yet progress is hindered by the scarcity of dense, high-quality motion annotations. While self-supervision on unlabeled real data offers a path forward, empirical evidence suggests that scaling unlabeled data fails to close the performance gap due to noisy proxy signals. In this paper, we propose a shift in paradigm: learning robust real-world motion priors entirely from scalable simulation. We introduce SynFlow, a data generation pipeline that generates large-scale synthetic dataset specifically designed for LiDAR scene flow. Unlike prior works that prioritize sensor-specific realism, SynFlow employs a motion-oriented strategy to synthesize diverse kinematic patterns across 4,000 sequences ($\sim$940k frames), termed SynFlow-4k. This represents a 34x scale-up in annotated volume over existing real-world benchmarks. Our experiments demonstrate that SynFlow-4k provides a highly domain-invariant motion prior. In a zero-shot regime, models trained exclusively on our synthetic data generalize across multiple real-world benchmarks, rivaling in-domain supervised baselines on nuScenes and outperforming state-of-the-art methods on TruckScenes by 31.8%. Furthermore, SynFlow-4k serves as a label-efficient foundation: fine-tuning with only 5% of real-world labels surpasses models trained from scratch on the full available budget. We open-source the pipeline and dataset to facilitate research in generalizable 3D motion estimation. More detail can be found at this https URL.
>
---
#### [new 040] DeFakeQ: Enabling Real-Time Deepfake Detection on Edge Devices via Adaptive Bidirectional Quantization
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决模型在边缘设备上实时部署的问题。提出DefakeQ框架，通过自适应双向量化策略，在保持检测性能的同时压缩模型。**

- **链接: [https://arxiv.org/pdf/2604.08847](https://arxiv.org/pdf/2604.08847)**

> **作者:** Xiangyu Li; Yujing Sun; Yuhang Zheng; Yuexin Ma; Kwok-Yan Lam
>
> **摘要:** Deepfake detection has become a fundamental component of modern media forensics. Despite significant progress in detection accuracy, most existing methods remain computationally intensive and parameter-heavy, limiting their deployment on resource-constrained edge devices that require real-time, on-site inference. This limitation is particularly critical in an era where mobile devices are extensively used for media-centric applications, including online payments, virtual meetings, and social networking. Meanwhile, due to the unique requirement of capturing extremely subtle forgery artifacts for deepfake detection, state-of-the-art quantization techniques usually underperform for such a challenging task. These fine-grained cues are highly sensitive to model compression and can be easily degraded during quantization, leading to noticeable performance drops. This challenge highlights the need for quantization strategies specifically designed to preserve the discriminative features essential for reliable deepfake detection. To address this gap, we propose DefakeQ, the first quantization framework tailored for deepfake detectors, enabling real-time deployment on edge devices. Our approach introduces a novel adaptive bidirectional compression strategy that simultaneously leverages feature correlations and eliminates redundancy, achieving an effective balance between model compactness and detection performance. Extensive experiments across five benchmark datasets and eleven state-of-the-art backbone detectors demonstrate that DeFakeQ consistently surpasses existing quantization and model compression baselines. Furthermore, we deploy DefakeQ on mobile devices in real-world scenarios, demonstrating its capability for real-time deepfake detection and its practical applicability in edge environments.
>
---
#### [new 041] LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification
- **分类: cs.CV**

- **简介: 该论文属于车牌可读性分类任务，解决真实场景下车牌识别困难的问题。通过扩展数据集、优化标注和引入新标签，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.08741](https://arxiv.org/pdf/2604.08741)**

> **作者:** Lucas Wojcik; Eduardo A. F. Machoski; Eduil Nascimento Jr.; Rayson Laroca; David Menotti
>
> **摘要:** Modern Automatic License Plate Recognition (ALPR) systems achieve outstanding performance in controlled, well-defined scenarios. However, large-scale real-world usage remains challenging due to low-quality imaging devices, compression artifacts, and suboptimal camera installation. Identifying illegible license plates (LPs) has recently become feasible through a dedicated benchmark; however, its impact has been limited by its small size and annotation errors. In this work, we expand the original benchmark to over three times the size with two extra capture days, revise its annotations and introduce novel labels. LP-level annotations include bounding boxes, text, and legibility level, while vehicle-level annotations comprise make, model, type, and color. Image-level annotations feature camera identity, capture conditions (e.g., rain and faulty cameras), acquisition time, and day ID. We present a novel training procedure featuring an Exponential Moving Average-based loss function and a refined learning rate scheduler, addressing common mistakes in testing. These improvements enable a baseline model to achieve an 89.5% F1-score on the test set, considerably surpassing the previous state of the art. We further introduce a novel protocol to explicitly addresses camera contamination between training and evaluation splits, where results show a small impact. Dataset and code are publicly available at this https URL.
>
---
#### [new 042] FIRE-CIR: Fine-grained Reasoning for Composed Fashion Image Retrieval
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于时尚图像检索任务，解决传统方法无法准确理解修改描述的问题。提出FIRE-CIR模型，通过视觉问答进行细粒度推理，提升检索准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.09114](https://arxiv.org/pdf/2604.09114)**

> **作者:** François Gardères; Camille-Sovanneary Gauthier; Jean Ponce; Shizhe Chen
>
> **摘要:** Composed image retrieval (CIR) aims to retrieve a target image that depicts a reference image modified by a textual description. While recent vision-language models (VLMs) achieve promising CIR performance by embedding images and text into a shared space for retrieval, they often fail to reason about what to preserve and what to change. This limitation hinders interpretability and yields suboptimal results, particularly in fine-grained domains like fashion. In this paper, we introduce FIRE-CIR, a model that brings compositional reasoning and interpretability to fashion CIR. Instead of relying solely on embedding similarity, FIRE-CIR performs question-driven visual reasoning: it automatically generates attribute-focused visual questions derived from the modification text, and verifies the corresponding visual evidence in both reference and candidate images. To train such a reasoning system, we automatically construct a large-scale fashion-specific visual question answering dataset, containing questions requiring either single- or dual-image analysis. During retrieval, our model leverages this explicit reasoning to re-rank candidate results, filtering out images inconsistent with the intended modifications. Experimental results on the Fashion IQ benchmark show that FIRE-CIR outperforms state-of-the-art methods in retrieval accuracy. It also provides interpretable, attribute-level insights into retrieval decisions.
>
---
#### [new 043] Physically Grounded 3D Generative Reconstruction under Hand Occlusion using Proprioception and Multi-Contact Touch
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D物体重建任务，解决手部遮挡下的物体结构与位姿估计问题。通过融合本体感觉和多点触觉信息，提升遮挡区域的重建精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2604.09100](https://arxiv.org/pdf/2604.09100)**

> **作者:** Gabriele Mario Caddeo; Pasquale Marra; Lorenzo Natale
>
> **备注:** 27 pages, 10 figures, under review
>
> **摘要:** We propose a multimodal, physically grounded approach for metric-scale amodal object reconstruction and pose estimation under severe hand occlusion. Unlike prior occlusion-aware 3D generation methods that rely only on vision, we leverage physical interaction signals: proprioception provides the posed hand geometry, and multi-contact touch constrains where the object surface must lie, reducing ambiguity in occluded regions. We represent object structure as a pose-aware, camera-aligned signed distance field (SDF) and learn a compact latent space with a Structure-VAE. In this latent space, we train a conditional flow-matching diffusion model, pretraining on vision-only images and finetuning on occluded manipulation scenes while conditioning on visible RGB evidence, occluder/visibility masks, the hand latent representation, and tactile information. Crucially, we incorporate physics-based objectives and differentiable decoder-guidance during finetuning and inference to reduce hand--object interpenetration and to align the reconstructed surface with contact observations. Because our method produces a metric, physically consistent structure estimate, it integrates naturally into existing two-stage reconstruction pipelines, where a downstream module refines geometry and predicts appearance. Experiments in simulation show that adding proprioception and touch substantially improves completion under occlusion and yields physically plausible reconstructions at correct real-world scale compared to vision-only baselines; we further validate transfer by deploying the model on a real humanoid robot with an end-effector different from those used during training.
>
---
#### [new 044] AI Driven Soccer Analysis Using Computer Vision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉在体育分析中的应用，解决足球比赛中球员和关键点的定位问题。通过目标检测、分割和坐标转换，提取战术数据以辅助教练决策。**

- **链接: [https://arxiv.org/pdf/2604.08722](https://arxiv.org/pdf/2604.08722)**

> **作者:** Adrian Manchado; Tanner Cellio; Jonathan Keane; Yiyang Wang
>
> **摘要:** Sport analysis is crucial for team performance since it provides actionable data that can inform coaching decisions, improve player performance, and enhance team strategies. To analyze more complex features from game footage, a computer vision model can be used to identify and track key entities from the field. We propose the use of an object detection and tracking system to predict player positioning throughout the game. To translate this to positioning in relation to the field dimensions, we use a point prediction model to identify key points on the field and combine these with known field dimensions to extract actual distances. For the player-identification model, object detection models like YOLO and Faster R-CNN are evaluated on the accuracy of our custom video footage using multiple different evaluation metrics. The goal is to identify the best model for object identification to obtain the most accurate results when paired with SAM2 (Segment Anything Model 2) for segmentation and tracking. For the key point detection model, we use a CNN model to find consistent locations in the soccer field. Through homography, the positions of points and objects in the camera perspective will be transformed to a real-ground perspective. The segmented player masks from SAM2 are transformed from camera perspective to real-world field coordinates through homography, regardless of camera angle or movement. The transformed real-world coordinates can be used to calculate valuable tactical insights including player speed, distance covered, positioning heatmaps, and more complex team statistics, providing coaches and players with actionable performance data previously unavailable from standard video analysis.
>
---
#### [new 045] VAGNet: Vision-based accident anticipation with global features
- **分类: cs.CV**

- **简介: 该论文提出VAGNet，用于视觉事故预判任务，解决实时、高效预测交通事故的问题。通过全局特征而非物体级特征进行学习，提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.09305](https://arxiv.org/pdf/2604.09305)**

> **作者:** Vipooshan Vipulananthan; Charith D. Chitraranjan
>
> **摘要:** Traffic accidents are a leading cause of fatalities and injuries across the globe. Therefore, the ability to anticipate hazardous situations in advance is essential. Automated accident anticipation enables timely intervention through driver alerts and collision avoidance maneuvers, forming a key component of advanced driver assistance systems. In autonomous driving, such predictive capabilities support proactive safety behaviors, such as initiating defensive driving and human takeover when required. Using dashcam video as input offers a cost-effective solution, but it is challenging due to the complexity of real-world driving scenes. Accident anticipation systems need to operate in real-time. However, current methods involve extracting features from each detected object, which is computationally intensive. We propose VAGNet, a deep neural network that learns to predict accidents from dash-cam video using global features of traffic scenes without requiring explicit object-level features. The network consists of transformer and graph modules, and we use the vision foundation model VideoMAE-V2 for global feature extraction. Experiments on four benchmark datasets (DAD, DoTA, DADA, and Nexar) show that our method anticipates accidents with higher average precision and mean time-to-accident while being computationally more efficient compared to existing methods.
>
---
#### [new 046] R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于电路设计任务，解决GNN在物理设计中因表示不一致导致的评估困难。构建了R2G基准库，提供多视图标准化数据，提升模型评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.08810](https://arxiv.org/pdf/2604.08810)**

> **作者:** Zewei Zhou; Jiajun Zou; Jiajia Zhang; Ao Yang; Ruichao He; Haozheng Zhou; Ao Liu; Jiawei Liu; Leilei Jin; Shan Shen; Daying Sun
>
> **备注:** Accepted as a poster by CVPR2026
>
> **摘要:** Graph neural networks (GNNs) are increasingly applied to physical design tasks such as congestion prediction and wirelength estimation, yet progress is hindered by inconsistent circuit representations and the absence of controlled evaluation protocols. We present R2G (RTL-to-GDSII), a multi-view circuit-graph benchmark suite that standardizes five stage-aware views with information parity (every view encodes the same attribute set, differing only in where features attach) over 30 open-source IP cores (up to $10^6$ nodes/edges). R2G provides an end-to-end DEF-to-graph pipeline spanning synthesis, placement, and routing stages, together with loaders, unified splits, domain metrics, and reproducible baselines. By decoupling representation choice from model choice, R2G isolates a confound that prior EDA and graph-ML benchmarks leave uncontrolled. In systematic studies with GINE, GAT, and ResGatedGCN, we find: (i) view choice dominates model choice, with Test R$^2$ varying by more than 0.3 across representations for a fixed GNN; (ii) node-centric views generalize best across both placement and routing; and (iii) decoder-head depth (3--4 layers) is the primary accuracy driver, turning divergent training into near-perfect predictions (R$^2$$>$0.99). Code and datasets are available at this https URL.
>
---
#### [new 047] NTIRE 2026 The 3rd Restore Any Image Model (RAIM) Challenge: Multi-Exposure Image Fusion in Dynamic Scenes (Track 2)
- **分类: cs.CV**

- **简介: 本文是NTIRE 2026挑战赛的论文，聚焦于动态场景下的多曝光图像融合任务，解决因运动、光照变化和相机抖动导致的对齐困难与伪影问题。**

- **链接: [https://arxiv.org/pdf/2604.09030](https://arxiv.org/pdf/2604.09030)**

> **作者:** Lishen Qu; Yao Liu; Jie Liang; Hui Zeng; Wen Dai; Guanyi Qin; Ya-nan Guan; Shihao Zhou; Jufeng Yang; Lei Zhang; Radu Timofte; Xiyuan Yuan; Wanjie Sun; Shihang Li; Bo Zhang; Bin Chen; Jiannan Lin; Yuxu Chen; Qinquan Gao; Tong Tong; Song Gao; Jiacong Tang; Tao Hu; Xiaowen Ma; Qingsen Yan; Sunhan Xu; Juan Wang; Xinyu Sun; Lei Qi; He Xu; Jiachen Tu; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yaokun Shi
>
> **备注:** Accepted by CVPRW 2026
>
> **摘要:** This paper presents NTIRE 2026, the 3rd Restore Any Image Model (RAIM) challenge on multi-exposure image fusion in dynamic scenes. We introduce a benchmark that targets a practical yet difficult HDR imaging setting, where exposure bracketing must be fused under scene motion, illumination variation, and handheld camera jitter. The challenge data contains 100 training sequences with 7 exposure levels and 100 test sequences with 5 exposure levels, reflecting real-world scenarios that frequently cause misalignment and ghosting artefacts. We evaluate submissions with a leaderboard score derived from PSNR, SSIM, and LPIPS, while also considering perceptual quality, efficiency, and reproducibility during the final review. This track attracted 114 participating teams and received 987 submissions. The winning methods significantly improved the ability to remove artifacts from multi-exposure fusion and recover fine details. The dataset and the code of each team can be found at the repository: this https URL.
>
---
#### [new 048] MARINER: A 3E-Driven Benchmark for Fine-Grained Perception and Complex Reasoning in Open-Water Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MARINER基准，解决开放水域中细粒度感知与复杂推理问题，涵盖分类、检测和问答任务，推动多模态视觉语言模型研究。**

- **链接: [https://arxiv.org/pdf/2604.08615](https://arxiv.org/pdf/2604.08615)**

> **作者:** Xingming Liao; Ning Chen; Muying Shu; Yunpeng Yin; Peijian Zeng; Zhuowei Wang; Nankai Lin; Lianglun Cheng
>
> **摘要:** Fine-grained visual understanding and high-level reasoning in real-world open-water environments remain under-explored due to the lack of dedicated benchmarks. We introduce MARINER, a comprehensive benchmark built under the novel Entity-Environment-Event (3E) paradigm. MARINER contains 16,629 multi-source maritime images with 63 fine-grained vessel categories, diverse adverse environments, and 5 typical dynamic maritime incidents, covering fine-grained classification, object detection, and visual question answering tasks. We conduct extensive evaluations on mainstream Multimodal Large language models (MLLMs) and establish baselines, revealing that even advanced models struggle with fine-grained discrimination and causal reasoning in complex marine scenes. As a dedicated maritime benchmark, MARINER fills the gap of realistic and cognitive-level evaluation for maritime multimodal understanding, and promotes future research on robust vision-language models for open-water applications. Appendix and supplementary materials are available at this https URL.
>
---
#### [new 049] On Semiotic-Grounded Interpretive Evaluation of Generative Art
- **分类: cs.CV; cs.AI; cs.HC; cs.MM**

- **简介: 该论文属于艺术评价任务，旨在解决GenArt评估缺乏对深层象征意义的分析问题。提出SemJudge，通过符号学框架提升对艺术意义的解读能力。**

- **链接: [https://arxiv.org/pdf/2604.08641](https://arxiv.org/pdf/2604.08641)**

> **作者:** Ruixiang Jiang; Changwen Chen
>
> **摘要:** Interpretation is essential to deciphering the language of art: audiences communicate with artists by recovering meaning from visual artifacts. However, current Generative Art (GenArt) evaluators remain fixated on surface-level image quality or literal prompt adherence, failing to assess the deeper symbolic or abstract meaning intended by the creator. We address this gap by formalizing a Peircean computational semiotic theory that models Human-GenArt Interaction (HGI) as cascaded semiosis. This framework reveals that artistic meaning is conveyed through three modes - iconic, symbolic, and indexical - yet existing evaluators operate heavily within the iconic mode, remaining structurally blind to the latter two. To overcome this structural blindness, we propose SemJudge. This evaluator explicitly assesses symbolic and indexical meaning in HGI via a Hierarchical Semiosis Graph (HSG) that reconstructs the meaning-making process from prompt to generated artifact. Extensive quantitative experiments show that SemJudge aligns more closely with human judgments than prior evaluators on an interpretation-intensive fine-art benchmark. User studies further demonstrate that SemJudge produces deeper, more insightful artistic interpretations, thereby paving the way for GenArt to move beyond the generation of "pretty" images toward a medium capable of expressing complex human experience. Project page: this https URL.
>
---
#### [new 050] Scene-Agnostic Object-Centric Representation Learning for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决现有方法依赖场景相关监督信号的问题。通过引入无监督的物体中心学习，构建跨场景一致的物体表示，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.09045](https://arxiv.org/pdf/2604.09045)**

> **作者:** Tsuheng Hsu; Guiyu Liu; Juho Kannala; Janne Heikkilä
>
> **摘要:** Recent works on 3D scene understanding leverage 2D masks from visual foundation models (VFMs) to supervise radiance fields, enabling instance-level 3D segmentation. However, the supervision signals from foundation models are not fundamentally object-centric and often require additional mask pre/post-processing or specialized training and loss design to resolve mask identity conflicts across views. The learned identity of the 3D scene is scene-dependent, limiting generalizability across scenes. Therefore, we propose a dataset-level, object-centric supervision scheme to learn object representations in 3D Gaussian Splatting (3DGS). Building on a pre-trained slot attention-based Global Object Centric Learning (GOCL) module, we learn a scene-agnostic object codebook that provides consistent, identity-anchored representations across views and scenes. By coupling the codebook with the module's unsupervised object masks, we can directly supervise the identity features of 3D Gaussians without additional mask pre-/post-processing or explicit multi-view alignment. The learned scene-agnostic codebook enables object supervision and identification without per-scene fine-tuning or retraining. Our method thus introduces unsupervised object-centric learning (OCL) into 3DGS, yielding more structured representations and better generalization for downstream tasks such as robotic interaction, scene understanding, and cross-scene generalization.
>
---
#### [new 051] Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在视觉-语言冲突中的决策机制，探讨其是否因感知缺陷或仲裁失败导致错误。通过分析模型层间信号竞争，发现错误源于仲裁而非感知，提出干预方法提升视觉定位能力。**

- **链接: [https://arxiv.org/pdf/2604.09364](https://arxiv.org/pdf/2604.09364)**

> **作者:** Farhad Nooralahzadeh; Omid Rohanian; Yi Zhang; Jonathan Fürst; Kurt Stockinger
>
> **摘要:** When a Vision-Language Model (VLM) sees a blue banana and answers "yellow", is the problem of perception or arbitration? We explore the question in ten VLMs with various sizes and reveal an Encoding--Grounding Dissociation: models that fail to report what they see (and thus provide a wrong answer) still encode the visual evidence as strongly as models that provide the correct answer. Using Multimodal Arbitration Crossover (MAC) analysis with layer-by-layer Logit Lens probing, we track the competition between visual and prior signals across every layer of each model. We show that visual attributes can be linearly decodable from early layers (AUC > 0.86). The accuracy remains nearly identical for both successful and failed samples. However, the gap in the final-layer logit -- not the strength of encoding -- better predicts grounding outcomes with a correlation of . After having studied when VLMs base their answers on image clues rather than prior knowledge, we want to understand the causal relationships. We establish causality through full-sequence activation patching. The standard last-token interventions in LLM interpretability do not affect VLMs. In contrast, replacing the full token sequence at layers identified by MAC alters 60 to 84% of outputs. Partial-token decomposition shows that image tokens carry almost all of the causal impact, while text tokens have none. Scaling addresses the remaining architectural differences to achieve perfect retention. Moving from diagnosis to intervention, we show that training-free activation steering -- both linear and sparse autoencoder-guided -- in early layers can improve visual grounding by up to +3.8% with degrading performance in some setups. Overall, these findings lead to a clear conclusion: VLMs already see well, but the challenge is acting on what they see. Targeted interventions can help to bridge this gap.
>
---
#### [new 052] Incremental Semantics-Aided Meshing from LiDAR-Inertial Odometry and RGB Direct Label Transfer
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维重建任务，解决室内大场景中点云稀疏导致的几何失真问题。通过融合RGB语义与LiDAR数据，提升网格重建质量。**

- **链接: [https://arxiv.org/pdf/2604.09478](https://arxiv.org/pdf/2604.09478)**

> **作者:** Muhammad Affan; Ville Lehtola; George Vosselman
>
> **备注:** 8 pages, 5 figures, 2 tables. Accepted in ISPRS Archives 2026
>
> **摘要:** Geometric high-fidelity mesh reconstruction from LiDAR-inertial scans remains challenging in large, complex indoor environments -- such as cultural buildings -- where point cloud sparsity, geometric drift, and fixed fusion parameters produce holes, over-smoothing, and spurious surfaces at structural boundaries. We propose a modular, incremental RGB+LiDAR pipeline that generates incremental semantics-aided high-quality meshes from indoor scans through scan frame-based direct label transfer. A vision foundation model labels each incoming RGB frame; labels are incrementally projected and fused onto a LiDAR-inertial odometry map; and an incremental semantics-aware Truncated Signed Distance Function (TSDF) fusion step produces the final mesh via marching cubes. This frame-level fusion strategy preserves the geometric fidelity of LiDAR while leveraging rich visual semantics to resolve geometric ambiguities at reconstruction boundaries caused by LiDAR point-cloud sparsity and geometric drift. We demonstrate that semantic guidance improves geometric reconstruction quality; quantitative evaluation is therefore performed using geometric metrics on the Oxford Spires dataset, while results from the NTU VIRAL dataset are analyzed qualitatively. The proposed method outperforms state-of-the-art geometric baselines ImMesh and Voxblox, demonstrating the benefit of semantics-aided fusion for geometric mesh quality. The resulting semantically labelled meshes are of value when reconstructing Universal Scene Description (USD) assets, offering a path from indoor LiDAR scanning to XR and digital modeling.
>
---
#### [new 053] Deep Learning-Based Tracking and Lineage Reconstruction of Ligament Breakup
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于液体破碎的跟踪与谱系重建任务，解决多尺度动态识别与追踪问题。提出两阶段深度学习框架，实现 ligament 和 droplet 的检测与碎片化关系建模。**

- **链接: [https://arxiv.org/pdf/2604.08711](https://arxiv.org/pdf/2604.08711)**

> **作者:** Vrushank Ahire; Vivek Kurumanghat; Mudasir Ganaie; Lipika Kabiraj
>
> **摘要:** The disintegration of liquid sheets into ligaments and droplets involves highly transient, multi-scale dynamics that are difficult to quantify from high-speed shadowgraphy images. Identifying droplets, ligaments, and blobs formed during breakup, along with tracking across frames, is essential for spray analysis. However, conventional multi-object tracking frameworks impose strict one-to-one temporal associations and cannot represent one-to-many fragmentation events. In this study, we present a two-stage deep learning framework for object detection and temporal relationship modeling across frames. The framework captures ligament deformation, fragmentation, and parent-child lineage during liquid sheet disintegration. In the first stage, a Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Network detects and classifies ligaments and droplets in high-speed shadowgraphy recordings of an impinging Carbopol gel jet. A morphology-preserving synthetic data generation strategy augments the training set without introducing physically implausible configurations, achieving a held-out F1 score of up to 0.872 across fourteen original-to-synthetic configurations. In the second stage, a Transformer-augmented multilayer perceptron classifies inter-frame associations into continuation, fragmentation (one-to-many), and non-association using physics-informed geometric features. Despite severe class imbalance, the model achieves 86.1% accuracy, 93.2% precision, and perfect recall (1.00) for fragmentation events. Together, the framework enables automated reconstruction of fragmentation trees, preservation of parent-child lineage, and extraction of breakup statistics such as fragment multiplicity and droplet size distributions. By explicitly identifying children droplets formed from ligament fragmentation, the framework provides automated analysis of the primary atomization mode.
>
---
#### [new 054] CAD 100K: A Comprehensive Multi-Task Dataset for Car Related Visual Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出CAD 100K数据集，用于汽车相关多任务视觉异常检测。解决现有方法任务专用、缺乏统一评估的问题，通过多任务学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09023](https://arxiv.org/pdf/2604.09023)**

> **作者:** Jiahua Pang; Ying Li; Dongpu Cao; Jingcai Luo; Yanuo Zheng; Bao Yunfan; Yujie Lei; Rui Yuan; Yuxi Tian; Guojin Yuan; Hongchang Chen; Zhi Zheng; Yongchun Liu
>
> **摘要:** Multi-task visual anomaly detection is critical for car-related manufacturing quality assessment. However, existing methods remain task-specific, hindered by the absence of a unified benchmark for multi-task evaluation. To fill in this gap, We present the CAD Dataset, a large-scale and comprehensive benchmark designed for car-related multi-task visual anomaly detection. The dataset contains over 100 images crossing 7 vehicle domains and 3 tasks, providing models a comprehensive view for car-related anomaly detection. It is the first car-related anomaly dataset specialized for multi-task learning(MTL), while combining synthesis data augmentation for few-shot anomaly images. We implement a multi-task baseline and conduct extensive empirical studies. Results show MTL promotes task interaction and knowledge transfer, while also exposing challenging conflicts between tasks. The CAD dataset serves as a standardized platform to drive future advances in car-related multi-task visual anomaly detection.
>
---
#### [new 055] Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像云分割任务，解决视觉-语言模型在领域迁移中的适应问题。通过对比提示和微调方法，证明少量标注数据的监督微调优于提示方法。**

- **链接: [https://arxiv.org/pdf/2604.08956](https://arxiv.org/pdf/2604.08956)**

> **作者:** Harshith Kethavath; Weiming Hu
>
> **备注:** 10 pages, 6 figures, to be published in EarthVision @ CVPR 2026
>
> **摘要:** Adapting vision-language models to remote sensing imagery presents a fundamental challenge: both the visual and linguistic distributions of satellite data lie far outside natural image pretraining corpora. Despite this, prompting remains the dominant deployment paradigm, driven by the assumption that domain-specific language can guide frozen model representations toward specialized tasks. We test this assumption directly on a domain where the mismatch is prominent: cloud segmentation for satellite imagery. Using CLIPSeg on the CloudSEN12+ cloud segmentation benchmark, we evaluate 60 prompt variants spanning simple labels, domain terminology, appearance descriptors, and contextual cues, finding that every variant underperforms the zero-shot baseline (0.255 mIoU), with engineered prompts scoring as low as 0.07 mIoU. No amount of linguistic refinement bridges the gap between CLIP's natural image representations and satellite spectral imagery. In contrast, supervised fine-tuning with just 0.1% labeled data (~8 images) surpasses zero-shot performance overall, and 5-10% data recovers ~85% of maximum achievable mIoU. Full fine-tuning consistently outperforms low-rank adaptation by 0.03-0.09 mIoU, with the largest gaps for spectrally ambiguous classes, and at 0.5 to 1% labeled data, fine-tuning temporarily degrades performance on these classes before recovering, a supervision dip that aggregate mIoU can mask. For practitioners adapting vision-language models to specialized imagery, our results deliver a clear message: labeled data is not the expensive alternative to prompting; it is the worthwhile path.
>
---
#### [new 056] Visually-Guided Policy Optimization for Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决VLMs视觉忠实度不足的问题。提出VGPO框架，通过视觉引导策略增强视觉关注与记忆。**

- **链接: [https://arxiv.org/pdf/2604.09349](https://arxiv.org/pdf/2604.09349)**

> **作者:** Zengbin Wang; Feng Xiong; Liang Lin; Xuecai Hu; Yong Wang; Yanlin Wang; Man Zhang; Xiangxiang Chu
>
> **备注:** ACL 2026
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning ability of vision-language models (VLMs). However, the inherent text-dominated nature of VLMs often leads to insufficient visual faithfulness, characterized by sparse attention activation to visual tokens. More importantly, our empirical analysis reveals that temporal visual forgetting along reasoning steps exacerbates this deficiency. To bridge this gap, we propose Visually-Guided Policy Optimization (VGPO), a novel framework to reinforce visual focus during policy optimization. Specifically, VGPO initially introduces a Visual Attention Compensation mechanism that leverages visual similarity to localize and amplify visual cues, while progressively elevating visual expectations in later steps to counteract visual forgetting. Building on this mechanism, we implement a dual-grained advantage re-weighting strategy: the intra-trajectory level highlights tokens exhibiting relatively high visual activation, while the inter-trajectory level prioritizes trajectories demonstrating superior visual accumulation. Extensive experiments demonstrate that VGPO achieves better visual activation and superior performance in mathematical multimodal reasoning and visual-dependent tasks.
>
---
#### [new 057] M-IDoL: Information Decomposition for Modality-Specific and Diverse Representation Learning in Medical Foundation Model
- **分类: cs.CV**

- **简介: 该论文属于医学基础模型领域，旨在解决多模态医学图像表示学习中的信息模糊问题。通过信息分解方法，提升模态特异性和多样性。**

- **链接: [https://arxiv.org/pdf/2604.08936](https://arxiv.org/pdf/2604.08936)**

> **作者:** Yihang Liu; Ying Wen; Jiaxiong Yang; Longzhen Yang; Lianghua He; Heng Tao Shen
>
> **摘要:** Medical foundation models (MFMs) aim to learn universal representations from multimodal medical images that can generalize effectively to diverse downstream clinical tasks. However, most existing MFMs suffer from information ambiguity that blend multimodal representations in a single embedding space, leading to the degradation of modality specificity and diversity. In this paper, we propose M-IDoL, a self-supervised \underline{\textit{M}}FM that introduces Information Decomposition for multimodal representation Learning via two objectives: i) maximize inter-modality entropy by dispersing multimodal representation into separable Mixture-of-Experts (MoE) subspaces to achieve representation specificity across modalities; and ii) minimize intra-modality uncertainty by performing fine-grained semantic discrimination within each MoE subspace to enrich representation diversity per modality. By pre-training on 1.15 million medical images, M-IDoL i) delivers superior generalization across 21 downstream clinical tasks, outperforming 20 foundation models on five imaging modalities (e.g., X-ray, fundus, OCT, dermoscopy and pathology), and ii) learns modality-specific and diverse representations, showing clearer separation of feature cluster across modalities and finer-grained feature discrimination within each modality.
>
---
#### [new 058] State Space Models are Effective Sign Language Learners: Exploiting Phonological Compositionality for Vocabulary-Scale Recognition
- **分类: cs.CV**

- **简介: 该论文属于手语识别任务，解决小词汇量模型在大规模词汇上失效的问题。通过引入PHONSSM，利用手语的音系结构进行分解，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.08761](https://arxiv.org/pdf/2604.08761)**

> **作者:** Bryan Cheng; Austin Jin; Jasper Zhang
>
> **备注:** 8 pages, 3 figures. Accepted to workshop on Algorithmic Fairness Across Alignment Procedures and Agentic Systems at ICLR 2026
>
> **摘要:** Sign language recognition suffers from catastrophic scaling failure: models achieving high accuracy on small vocabularies collapse at realistic sizes. Existing architectures treat signs as atomic visual patterns, learning flat representations that cannot exploit the compositional structure of sign languages-systematically organized from discrete phonological parameters (handshape, location, movement, orientation) reused across the vocabulary. We introduce PHONSSM, enforcing phonological decomposition through anatomically-grounded graph attention, explicit factorization into orthogonal subspaces, and prototypical classification enabling few-shot transfer. Using skeleton data alone on the largest ASL dataset ever assembled (5,565 signs), PHONSSM achieves 72.1% on WLASL2000 (+18.4pp over skeleton SOTA), surpassing most RGB methods without video input. Gains are most dramatic in the few-shot regime (+225% relative), and the model transfers zero-shot to ASL Citizen, exceeding supervised RGB baselines. The vocabulary scaling bottleneck is fundamentally a representation learning problem, solvable through compositional inductive biases mirroring linguistic structure.
>
---
#### [new 059] Cross-Modal Knowledge Distillation from Spatial Transcriptomics to Histology
- **分类: cs.CV**

- **简介: 该论文属于跨模态知识蒸馏任务，旨在将空间转录组学的结构信息转移到组织病理学模型中，解决数据稀缺与分辨率不足的问题。通过联合训练提升模型对组织结构的准确性。**

- **链接: [https://arxiv.org/pdf/2604.09076](https://arxiv.org/pdf/2604.09076)**

> **作者:** Arbel Hizmi; Artemii Bakulin; Shai Bagon; Nir Yosef
>
> **备注:** Accepted to the CVMI Workshop at CVPR 2026. Project page: this https URL
>
> **摘要:** Spatial transcriptomics provides a molecularly rich description of tissue organization, enabling unsupervised discovery of tissue niches -- spatially coherent regions of distinct cell-type composition and function that are relevant to both biological research and clinical interpretation. However, spatial transcriptomics remains costly and scarce, while H&E histology is abundant but carries a less granular signal. We propose to leverage paired spatial transcriptomics and H&E data to transfer transcriptomics-derived niche structure to a histology-only model via cross-modal distillation. Across multiple tissue types and disease contexts, the distilled model achieves substantially higher agreement with transcriptomics-derived niche structure than unsupervised morphology-based baselines trained on identical image features, and recovers biologically meaningful neighborhood composition as confirmed by cell-type analysis. The resulting framework leverages paired spatial transcriptomic and H&E data during training, and can then be applied to held-out tissue regions using histology alone, without any transcriptomic input at inference time.
>
---
#### [new 060] Few-Shot Personalized Age Estimation
- **分类: cs.CV**

- **简介: 该论文属于个性化年龄估计任务，解决个体差异导致的年龄预测不准确问题。通过引入OpenPAE基准，提出多种个性化方法，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2604.09125](https://arxiv.org/pdf/2604.09125)**

> **作者:** Jakub Paplhám; Vojtěch Franc; Artem Moroz
>
> **摘要:** Existing age estimation methods treat each face as an independent sample, learning a global mapping from appearance to age. This ignores a well-documented phenomenon: individuals age at different rates due to genetics, lifestyle, and health, making the mapping from face to age identity-dependent. When reference images of the same person with known ages are available, we can exploit this context to personalize the estimate. The only existing benchmark for this task (NIST FRVT) is closed-source and limited to a single reference image. In this work, we introduce OpenPAE, the first open benchmark for $N$-shot personalized age estimation with strict evaluation protocols. We establish a hierarchy of increasingly sophisticated baselines: from arithmetic offset, through closed-form Bayesian linear regression, to a conditional attentive neural process. Our experiments show that personalization consistently improves performance, that the gains are not merely domain adaptation, and that nonlinear methods significantly outperform simpler alternatives. We release all models, code, protocols, and evaluation splits.
>
---
#### [new 061] Accelerating Transformer-Based Monocular SLAM via Geometric Utility Scoring
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于单目SLAM任务，旨在解决GFM部署中的计算冗余问题。提出LeanGate网络，在特征提取前预测几何价值，减少冗余帧处理，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2604.08718](https://arxiv.org/pdf/2604.08718)**

> **作者:** Xinmiao Xiong; Bangya Liu; Hao Wang; Dayou Li; Nuo Chen; Andrew Feng; Mingyu Ding; Suman Banerjee; Yang Zhou; Zhiwen Fan
>
> **摘要:** Geometric Foundation Models (GFMs) have recently advanced monocular SLAM by providing robust, calibration-free 3D priors. However, deploying these models on dense video streams introduces significant computational redundancy. Current GFM-based SLAM systems typically rely on post hoc keyframe selection. Because of this, they must perform expensive dense geometric decoding simply to determine whether a frame contains novel geometry, resulting in late rejection and wasted computation. To mitigate this inefficiency, we propose LeanGate, a lightweight feed-forward frame-gating network. LeanGate predicts a geometric utility score to assess a frame's mapping value prior to the heavy GFM feature extraction and matching stages. As a predictive plug-and-play module, our approach bypasses over 90% of redundant frames. Evaluations on standard SLAM benchmarks demonstrate that LeanGate reduces tracking FLOPs by more than 85% and achieves a 5x end-to-end throughput speedup. Furthermore, it maintains the tracking and mapping accuracy of dense baselines.
>
---
#### [new 062] Large-Scale Universal Defect Generation: Foundation Models and Datasets
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于缺陷生成任务，解决现有方法泛化能力差、依赖少量样本的问题。提出UDG数据集和UniDG模型，实现通用缺陷生成与编辑。**

- **链接: [https://arxiv.org/pdf/2604.08915](https://arxiv.org/pdf/2604.08915)**

> **作者:** Yuanting Fan; Jun Liu; Bin-Bin Gao; Xiaochen Chen; Yuhuan Lin; Zhewei Dai; Jiawei Zhan; Chengjie Wang
>
> **备注:** 25 pages, 13 figures, preprint
>
> **摘要:** Existing defect/anomaly generation methods often rely on few-shot learning, which overfits to specific defect categories due to the lack of large-scale paired defect editing data. This issue is aggravated by substantial variations in defect scale and morphology, resulting in limited generalization, degraded realism, and category consistency. We address these challenges by introducing UDG, a large-scale dataset of 300K normal-abnormal-mask-caption quadruplets spanning diverse domains, and by presenting UniDG, a universal defect generation foundation model that supports both reference-based defect generation and text instruction-based defect editing without per-category fine-tuning. UniDG performs Defect-Context Editing via adaptive defect cropping and structured diptych input format, and fuses reference and target conditions through MM-DiT multimodal attention. A two-stage training strategy, Diversity-SFT followed by Consistency-RFT, further improves diversity while enhancing realism and reference consistency. Extensive experiments on MVTec-AD and VisA show that UniDG outperforms prior few-shot anomaly generation and image insertion/editing baselines in synthesis quality and downstream single- and multi-class anomaly detection/localization. Code will be available at this https URL.
>
---
#### [new 063] RS-OVC: Open-Vocabulary Counting for Remote-Sensing Data
- **分类: cs.CV**

- **简介: 该论文属于遥感目标计数任务，解决传统方法仅支持预定义类别的问题。提出RS-OVC模型，实现基于文本或视觉条件的开放词汇计数。**

- **链接: [https://arxiv.org/pdf/2604.08704](https://arxiv.org/pdf/2604.08704)**

> **作者:** Tamir Shor; George Leifman; Genady Beryozkin
>
> **摘要:** Object-Counting for remote-sensing (RS) imagery is attracting increasing research interest due to its crucial role in a wide and diverse set of applications. While several promising methods for RS object-counting have been proposed, existing methods focus on a closed, pre-defined set of object classes. This limitation necessitates costly re-annotation and model re-training to adapt current approaches for counting of novel objects that have not been seen during training, and severely inhibits their application in dynamic, real-world monitoring scenarios. To address this gap, in this work we propose RS-OVC - the first Open Vocabulary Counting (OVC) model for Remote-Sensing and aerial imagery. We show that our model is capable of accurate counting of novel object classes, that were unseen during training, based solely on textual and/or visual conditioning.
>
---
#### [new 064] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的3D感知任务，旨在解决长距离协同感知中的计算复杂度高和特征匹配不稳定问题。提出Long-SCOPE框架，采用稀疏表示和两个创新模块提升性能。**

- **链接: [https://arxiv.org/pdf/2604.09206](https://arxiv.org/pdf/2604.09206)**

> **作者:** Jiahao Wang; Zikun Xu; Yuner Zhang; Zhongwei Jiang; Chenyang Lu; Shuocheng Yang; Yuxuan Wang; Jiaru Zhong; Chuang Zhang; Shaobing Xu; Jianqiang Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Cooperative 3D perception via Vehicle-to-Everything communication is a promising paradigm for enhancing autonomous driving, offering extended sensing horizons and occlusion resolution. However, the practical deployment of existing methods is hindered at long distances by two critical bottlenecks: the quadratic computational scaling of dense BEV representations and the fragility of feature association mechanisms under significant observation and alignment errors. To overcome these limitations, we introduce Long-SCOPE, a fully sparse framework designed for robust long-distance cooperative 3D perception. Our method features two novel components: a Geometry-guided Query Generation module to accurately detect small, distant objects, and a learnable Context-Aware Association module that robustly matches cooperative queries despite severe positional noise. Experiments on the V2X-Seq and Griffin datasets validate that Long-SCOPE achieves state-of-the-art performance, particularly in challenging 100-150 m long-range settings, while maintaining highly competitive computation and communication costs.
>
---
#### [new 065] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms
- **分类: cs.CV**

- **简介: 该论文属于视频时序定位任务，旨在解决输出范式对系统效率与精度的影响问题。通过对比三种输出方法，评估其在不同模型和数据集上的表现，提出高效部署的方案。**

- **链接: [https://arxiv.org/pdf/2604.08966](https://arxiv.org/pdf/2604.08966)**

> **作者:** Shengji Jin; Yuanhao Zou; Victor Zhu; Zhengping Ji; Chen Chen
>
> **备注:** CVPR 2026 Workshop Paper
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have advanced Video Temporal Grounding (VTG), existing methods often couple output paradigms with different backbones, datasets, and training protocols. This makes it challenging to isolate the specific impact of the output design. Additionally, as VTG systems are increasingly considered for resource-constrained edge deployment, the trade-off between output formulation and system-level efficiency requires systematic investigation. In this paper, we present a controlled empirical study comparing three dominant VTG output paradigms: Text Numeral Generation, Temporal Token Generation, and Continuous Temporal Decoding. We evaluate these paradigms across identical compact VLMs (SmolVLM2, FastVLM, and Molmo2) using consistent datasets and LoRA fine-tuning protocols. Evaluations on Charades-STA, QVHighlights, and YouCook2 measure both localization accuracy and system efficiency, including inference latency, training throughput, and parameter overhead. Our results demonstrate that the choice of output formulation significantly affects both grounding accuracy and computational cost, independent of model scale. Specifically, the continuous distribution paradigm consistently achieves the most favorable efficiency-accuracy trade-off on the Pareto frontier, delivering robust localization with minimal latency overhead. These findings provide objective empirical guidelines for designing efficient, deployment-ready VTG systems.
>
---
#### [new 066] ELT: Elastic Looped Transformers for Visual Generation
- **分类: cs.CV**

- **简介: 该论文提出ELT模型，用于视觉生成任务，解决参数效率与生成质量的平衡问题。通过循环Transformer结构和ILSD方法，在减少参数的同时保持高质量生成。**

- **链接: [https://arxiv.org/pdf/2604.09168](https://arxiv.org/pdf/2604.09168)**

> **作者:** Sahil Goyal; Swayam Agrawal; Gautham Govind Anil; Prateek Jain; Sujoy Paul; Aditya Kusupati
>
> **摘要:** We introduce Elastic Looped Transformers (ELT), a highly parameter-efficient class of visual generative models based on a recurrent transformer architecture. While conventional generative models rely on deep stacks of unique transformer layers, our approach employs iterative, weight-shared transformer blocks to drastically reduce parameter counts while maintaining high synthesis quality. To effectively train these models for image and video generation, we propose the idea of Intra-Loop Self Distillation (ILSD), where student configurations (intermediate loops) are distilled from the teacher configuration (maximum training loops) to ensure consistency across the model's depth in a single training step. Our framework yields a family of elastic models from a single training run, enabling Any-Time inference capability with dynamic trade-offs between computational cost and generation quality, with the same parameter count. ELT significantly shifts the efficiency frontier for visual synthesis. With $4\times$ reduction in parameter count under iso-inference-compute settings, ELT achieves a competitive FID of $2.0$ on class-conditional ImageNet $256 \times 256$ and FVD of $72.8$ on class-conditional UCF-101.
>
---
#### [new 067] Fine-Grained Action Segmentation for Renorrhaphy in Robot-Assisted Partial Nephrectomy
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于动作分割任务，解决机器人辅助部分肾切除术中精细动作识别问题。通过对比不同模型在基准数据集上的性能，提升动作识别的准确性。**

- **链接: [https://arxiv.org/pdf/2604.09051](https://arxiv.org/pdf/2604.09051)**

> **作者:** Jiaheng Dai; Huanrong Liu; Tailai Zhou; Tongyu Jia; Qin Liu; Yutong Ban; Zeju Li; Yu Gao; Xin Ma; Qingbiao Li
>
> **摘要:** Fine-grained action segmentation during renorrhaphy in robot-assisted partial nephrectomy requires frame-level recognition of visually similar suturing gestures with variable duration and substantial class imbalance. The SIA-RAPN benchmark defines this problem on 50 clinical videos acquired with the da Vinci Xi system and annotated with 12 frame-level labels. The benchmark compares four temporal models built on I3D features: MS-TCN++, AsFormer, TUT, and DiffAct. Evaluation uses balanced accuracy, edit score, segmental F1 at overlap thresholds of 10, 25, and 50, frame-wise accuracy, and frame-wise mean average precision. In addition to the primary evaluation across five released split configurations on SIA-RAPN, the benchmark reports cross-domain results on a separate single-port RAPN dataset. Across the strongest reported values over those five runs on the primary dataset, DiffAct achieves the highest F1, frame-wise accuracy, edit score, and frame mAP, while MS-TCN++ attains the highest balanced accuracy.
>
---
#### [new 068] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation
- **分类: cs.CV**

- **简介: 该论文属于高效迁移学习任务，旨在解决侧网络增加推理开销的问题。通过提出MDPD方法，在保持参数和内存效率的同时加速推理并提升准确率。**

- **链接: [https://arxiv.org/pdf/2604.09088](https://arxiv.org/pdf/2604.09088)**

> **作者:** Yutong Zhang; Jiaxin Chen; Honglin Chen; Kaiqi Zheng; Shengcai Liao; Hanwen Zhong; Weixin Li; Yunhong Wang
>
> **备注:** CVPR2026 Accepted
>
> **摘要:** Memory-efficient transfer learning (METL) approaches have recently achieved promising performance in adapting pre-trained models to downstream tasks. They avoid applying gradient backpropagation in large backbones, thus significantly reducing the number of trainable parameters and high memory consumption during fine-tuning. However, since they typically employ a lightweight and learnable side network, these methods inevitably introduce additional memory and time overhead during inference, which contradicts the ultimate goal of efficient transfer learning. To address the above issue, we propose a novel approach dubbed Masked Dual Path Distillation (MDPD) to accelerate inference while retaining parameter and memory efficiency in fine-tuning with fading side networks. Specifically, MDPD develops a framework that enhances the performance by mutually distilling the frozen backbones and learnable side networks in fine-tuning, and discard the side network during inference without sacrificing accuracy. Moreover, we design a novel feature-based knowledge distillation method for the encoder structure with multiple layers. Extensive experiments on distinct backbones across vision/language-only and vision-and-language tasks demonstrate that our method not only accelerates inference by at least 25.2\% while keeping parameter and memory consumption comparable, but also remarkably promotes the accuracy compared to SOTA approaches. The source code is available at this https URL.
>
---
#### [new 069] WildDet3D: Scaling Promptable 3D Detection in the Wild
- **分类: cs.CV**

- **简介: 该论文提出WildDet3D，解决开放世界下的单目3D目标检测问题，支持多种提示类型并融合几何信息，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.08626](https://arxiv.org/pdf/2604.08626)**

> **作者:** Weikai Huang; Jieyu Zhang; Sijun Li; Taoyang Jia; Jiafei Duan; Yunqian Cheng; Jaemin Cho; Mattew Wallingford; Rustin Soraki; Chris Dongjoo Kim; Donovan Clay; Taira Anderson; Winson Han; Ali Farhadi; Bharath Hariharan; Zhongzheng Ren; Ranjay Krishna
>
> **摘要:** Understanding objects in 3D from a single image is a cornerstone of spatial intelligence. A key step toward this goal is monocular 3D object detection--recovering the extent, location, and orientation of objects from an input RGB image. To be practical in the open world, such a detector must generalize beyond closed-set categories, support diverse prompt modalities, and leverage geometric cues when available. Progress is hampered by two bottlenecks: existing methods are designed for a single prompt type and lack a mechanism to incorporate additional geometric cues, and current 3D datasets cover only narrow categories in controlled environments, limiting open-world transfer. In this work we address both gaps. First, we introduce WildDet3D, a unified geometry-aware architecture that natively accepts text, point, and box prompts and can incorporate auxiliary depth signals at inference time. Second, we present WildDet3D-Data, the largest open 3D detection dataset to date, constructed by generating candidate 3D boxes from existing 2D annotations and retaining only human-verified ones, yielding over 1M images across 13.5K categories in diverse real-world scenes. WildDet3D establishes a new state-of-the-art across multiple benchmarks and settings. In the open-world setting, it achieves 22.6/24.8 AP3D on our newly introduced WildDet3D-Bench with text and box prompts. On Omni3D, it reaches 34.2/36.4 AP3D with text and box prompts, respectively. In zero-shot evaluation, it achieves 40.3/48.9 ODS on Argoverse 2 and ScanNet. Notably, incorporating depth cues at inference time yields substantial additional gains (+20.7 AP on average across settings).
>
---
#### [new 070] Realizing Immersive Volumetric Video: A Multimodal Framework for 6-DoF VR Engagement
- **分类: cs.CV**

- **简介: 该论文属于虚拟现实任务，旨在解决6-DoF沉浸式视频的构建问题。提出Immersive Volumetric Video新格式及ImViD数据集，并开发动态光场重建框架与声音场重建方法。**

- **链接: [https://arxiv.org/pdf/2604.09473](https://arxiv.org/pdf/2604.09473)**

> **作者:** Zhengxian Yang; Shengqi Wang; Shi Pan; Hongshuai Li; Haoxiang Wang; Lin Li; Guanjun Li; Zhengqi Wen; Borong Lin; Jianhua Tao; Tao Yu
>
> **备注:** Journal extension of CVPR 2025. See also arXiv:2503.14359 . Project page and code: this https URL
>
> **摘要:** Fully immersive experiences that tightly integrate 6-DoF visual and auditory interaction are essential for virtual and augmented reality. While such experiences can be achieved through computer-generated content, constructing them directly from real-world captured videos remains largely unexplored. We introduce Immersive Volumetric Videos, a new volumetric media format designed to provide large 6-DoF interaction spaces, audiovisual feedback, and high-resolution, high-frame-rate dynamic content. To support IVV construction, we present ImViD, a multi-view, multi-modal dataset built upon a space-oriented capture philosophy. Our custom capture rig enables synchronized multi-view video-audio acquisition during motion, facilitating efficient capture of complex indoor and outdoor scenes with rich foreground--background interactions and challenging dynamics. The dataset provides 5K-resolution videos at 60 FPS with durations of 1-5 minutes, offering richer spatial, temporal, and multimodal coverage than existing benchmarks. Leveraging this dataset, we develop a dynamic light field reconstruction framework built upon a Gaussian-based spatio-temporal representation, incorporating flow-guided sparse initialization, joint camera temporal calibration, and multi-term spatio-temporal supervision for robust and accurate modeling of complex motion. We further propose, to our knowledge, the first method for sound field reconstruction from such multi-view audiovisual data. Together, these components form a unified pipeline for immersive volumetric video production. Extensive benchmarks and immersive VR experiments demonstrate that our pipeline generates high-quality, temporally stable audiovisual volumetric content with large 6-DoF interaction spaces. This work provides both a foundational definition and a practical construction methodology for immersive volumetric videos.
>
---
#### [new 071] VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VISOR框架，解决长视界视觉推理中的证据稀疏和搜索漂移问题，提升视觉增强生成系统的效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.09508](https://arxiv.org/pdf/2604.09508)**

> **作者:** Yucheng Shen; Jiulong Wu; Jizhou Huang; Dawei Yin; Lingyong Yan; Min Cao
>
> **摘要:** Visual Retrieval-Augmented Generation (VRAG) empowers Vision-Language Models to retrieve and reason over visually rich documents. To tackle complex queries requiring multi-step reasoning, agentic VRAG systems interleave reasoning with iterative retrieval.. However, existing agentic VRAG faces two critical bottlenecks. (1) Visual Evidence Sparsity: key evidence is scattered across pages yet processed in isolation, hindering cross-page reasoning; moreover, fine-grained intra-image evidence often requires precise visual actions, whose misuse degrades retrieval quality; (2) Search Drift in Long Horizons: the accumulation of visual tokens across retrieved pages dilutes context and causes cognitive overload, leading agents to deviate from their search objective. To address these challenges, we propose VISOR (Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning), a unified single-agent framework. VISOR features a structured Evidence Space for progressive cross-page reasoning, coupled with a Visual Action Evaluation and Correction mechanism to manage visual actions. Additionally, we introduce a Dynamic Trajectory with Sliding Window and Intent Injection to mitigate search drift. They anchor the evidence space while discarding earlier raw interactions, preventing context from being overwhelmed by visual tokens. We train VISOR using a Group Relative Policy Optimization-based Reinforcement Learning (GRPO-based RL) pipeline with state masking and credit assignment tailored for dynamic context reconstruction. Extensive experiments on ViDoSeek, SlideVQA, and MMLongBench demonstrate that VISOR achieves state-of-the-art performance with superior efficiency for long-horizon visual reasoning tasks.
>
---
#### [new 072] CT-1: Vision-Language-Camera Models Transfer Spatial Reasoning Knowledge to Camera-Controllable Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决相机控制不精准的问题。提出CT-1模型，通过空间推理提升相机轨迹生成精度。**

- **链接: [https://arxiv.org/pdf/2604.09201](https://arxiv.org/pdf/2604.09201)**

> **作者:** Haoyu Zhao; Zihao Zhang; Jiaxi Gu; Haoran Chen; Qingping Zheng; Pin Tang; Yeyin Jin; Yuang Zhang; Junqi Cheng; Zenghui Lu; Peng Shu; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Camera-controllable video generation aims to synthesize videos with flexible and physically plausible camera movements. However, existing methods either provide imprecise camera control from text prompts or rely on labor-intensive manual camera trajectory parameters, limiting their use in automated scenarios. To address these issues, we propose a novel Vision-Language-Camera model, termed CT-1 (Camera Transformer 1), a specialized model designed to transfer spatial reasoning knowledge to video generation by accurately estimating camera trajectories. Built upon vision-language modules and a Diffusion Transformer model, CT-1 employs a Wavelet-based Regularization Loss in the frequency domain to effectively learn complex camera trajectory distributions. These trajectories are integrated into a video diffusion model to enable spatially aware camera control that aligns with user intentions. To facilitate the training of CT-1, we design a dedicated data curation pipeline and construct CT-200K, a large-scale dataset containing over 47M frames. Experimental results demonstrate that our framework successfully bridges the gap between spatial reasoning and video synthesis, yielding faithful and high-quality camera-controllable videos and improving camera control accuracy by 25.7% over prior methods.
>
---
#### [new 073] Text-Conditioned Multi-Expert Regression Framework for Fully Automated Multi-Abutment Design
- **分类: cs.CV**

- **简介: 该论文属于牙科种植体基台自动化设计任务，解决手动设计耗时且难以扩展的问题。提出TEMAD框架，实现多基台的全自动设计。**

- **链接: [https://arxiv.org/pdf/2604.09047](https://arxiv.org/pdf/2604.09047)**

> **作者:** Mianjie Zheng; Xinquan Yang; Xuefen Liu; Xuguang Li; Kun Tang; He Meng; Linlin Shen
>
> **摘要:** Dental implant abutments serve as the geometric and biomechanical interface between the implant fixture and the prosthetic crown, yet their design relies heavily on manual effort and is time-consuming. Although deep neural networks have been proposed to assist dentists in designing abutments, most existing approaches remain largely manual or semi-automated, requiring substantial clinician intervention and lacking scalability in multi-abutment scenarios. To address these limitations, we propose TEMAD, a fully automated, text-conditioned multi-expert architecture for multi-abutment design. This framework integrates implant site localization and implant system, compatible abutment parameter regression into a unified pipeline. Specifically, we introduce an Implant Site Identification Network (ISIN) to automatically localize implant sites and provide this information to the subsequent multi-abutment regression network. We further design a Tooth-Conditioned Feature-wise Linear Modulation (TC-FiLM) module, which adaptively calibrates mesh representations using tooth embeddings to enable position-specific feature modulation. Additionally, a System-Prompted Mixture-of-Experts (SPMoE) mechanism leverages implant system prompts to guide expert selection, ensuring system-aware regression. Extensive experiments on a large-scale abutment design dataset show that TEMAD achieves state-of-the-art performance compared to existing methods, particularly in multi-abutment settings, validating its effectiveness for fully automated dental implant planning.
>
---
#### [new 074] Adding Another Dimension to Image-based Animal Detection
- **分类: cs.CV**

- **简介: 该论文属于单目3D动物检测任务，解决2D检测缺乏方向信息的问题，通过估计3D边界框并投影到2D图像，构建标注数据。**

- **链接: [https://arxiv.org/pdf/2604.09210](https://arxiv.org/pdf/2604.09210)**

> **作者:** Vandita Shukla; Fabio Remondino; Benjamin Risse
>
> **备注:** CV4Animals Workshop 2025
>
> **摘要:** Monocular imaging of animals inherently reduces 3D structures to 2D projections. Detection algorithms lead to 2D bounding boxes that lack information about animal's orientation relative to the camera. To build 3D detection methods for RGB animal images, there is a lack of labeled datasets; such labeling processes require 3D input streams along with RGB data. We present a pipeline that utilises Skinned Multi Animal Linear models to estimate 3D bounding boxes and to project them as robust labels into 2D image space using a dedicated camera pose refinement algorithm. To assess which sides of the animal are captured, cuboid face visibility metrics are computed. These 3D bounding boxes and metrics form a crucial step toward developing and benchmarking future monocular 3D animal detection algorithms. We evaluate our method on the Animal3D dataset, demonstrating accurate performance across species and settings.
>
---
#### [new 075] Tango: Taming Visual Signals for Efficient Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频大模型优化任务，旨在解决token剪枝效率低的问题。提出Tango框架，提升注意力选择和时空位置编码，提高模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.09547](https://arxiv.org/pdf/2604.09547)**

> **作者:** Shukang Yin; Sirui Zhao; Hanchao Wang; Baozhi Jia; Xianquan Wang; Chaoyou Fu; Enhong Chen
>
> **备注:** Code is available at this https URL
>
> **摘要:** Token pruning has emerged as a mainstream approach for developing efficient Video Large Language Models (Video LLMs). This work revisits and advances the two predominant token-pruning paradigms: attention-based selection and similarity-based clustering. Our study reveals two critical limitations in existing methods: (1) conventional top-k selection strategies fail to fully account for the attention distribution, which is often spatially multi-modal and long-tailed in magnitude; and (2) direct similarity-based clustering frequently generates fragmented clusters, resulting in distorted representations after pooling. To address these bottlenecks, we propose Tango, a novel framework designed to optimize the utilization of visual signals. Tango integrates a diversity-driven strategy to enhance attention-based token selection, and introduces Spatio-temporal Rotary Position Embedding (ST-RoPE) to preserve geometric structure via locality priors. Comprehensive experiments across various Video LLMs and video understanding benchmarks demonstrate the effectiveness and generalizability of our approach. Notably, when retaining only 10% of the video tokens, Tango preserves 98.9% of the original performance on LLaVA-OV while delivering a 1.88x inference speedup.
>
---
#### [new 076] A Semi-Automated Framework for 3D Reconstruction of Medieval Manuscript Miniatures
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在将中世纪手稿插图转换为三维模型。通过评估多种方法，提出一个半自动框架，解决几何精度与体积扩展的平衡问题，并应用于不同艺术风格的案例。**

- **链接: [https://arxiv.org/pdf/2604.08610](https://arxiv.org/pdf/2604.08610)**

> **作者:** Riccardo Pallotto; Pierluigi Feliciati; Tiberio Uricchio
>
> **摘要:** This paper presents a semi-automated framework for transforming two-dimensional miniatures from medieval manuscripts into three-dimensional digital models suitable for extended reality (XR), tactile 3D~printing, and web-based visualization. We evaluate seven image-to-3D methods (TripoSR, SF3D, SPAR3D, TRELLIS, Wonder3D, SAM~3D, Hi3DGen) on 69~manuscript figures from two collections using rendering-based metrics (Silhouette IoU, LPIPS, CLIP~Score) and volumetric measures (Depth Range Ratio, watertight percentage), revealing a trade-off between volumetric expansion and geometric fidelity. Hi3DGen balances topological quality with rich surface detail through its normal bridging approach, making it a good starting point for expert refinement. Our pipeline combines SAM segmentation, Hi3DGen mesh generation, expert refinement in ZBrush, and AI-assisted texturing. Two case studies on Gothic illuminations from the Decretum Gratiani (Vatican Library) and Renaissance miniatures by Giulio Clovio demonstrate applicability across artistic traditions. The resulting models can support WebXR visualization, AR overlay on physical manuscripts, and tactile 3D~prints for visually impaired users.
>
---
#### [new 077] 3D-VCD: Hallucination Mitigation in 3D-LLM Embodied Agents through Visual Contrastive Decoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多模态推理任务，旨在解决3D大模型在具身代理中的幻觉问题。通过构建扭曲的3D场景图进行对比解码，提升 grounded 推理可靠性。**

- **链接: [https://arxiv.org/pdf/2604.08645](https://arxiv.org/pdf/2604.08645)**

> **作者:** Makanjuola Ogunleye; Eman Abdelrahman; Ismini Lourentzou
>
> **备注:** 8 pages, 6 figures, Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Large multimodal models are increasingly used as the reasoning core of embodied agents operating in 3D environments, yet they remain prone to hallucinations that can produce unsafe and ungrounded decisions. Existing inference-time hallucination mitigation methods largely target 2D vision-language settings and do not transfer to embodied 3D reasoning, where failures arise from object presence, spatial layout, and geometric grounding rather than pixel-level inconsistencies. We introduce 3D-VCD, the first inference-time visual contrastive decoding framework for hallucination mitigation in 3D embodied agents. 3D-VCD constructs a distorted 3D scene graph by applying semantic and geometric perturbations to object-centric representations, such as category substitutions and coordinate or extent corruption. By contrasting predictions under the original and distorted 3D contexts, our method suppresses tokens that are insensitive to grounded scene evidence and are therefore likely driven by language priors. We evaluate 3D-VCD on the 3D-POPE and HEAL benchmarks and show that it consistently improves grounded reasoning without any retraining, establishing inference-time contrastive decoding over structured 3D representations as an effective and practical route to more reliable embodied intelligence.
>
---
#### [new 078] SHIFT: Steering Hidden Intermediates in Flow Transformers
- **分类: cs.CV**

- **简介: 该论文提出SHIFT框架，用于在DiT扩散模型中控制生成内容，解决概念移除与风格迁移问题，通过调节中间激活实现高效控制。**

- **链接: [https://arxiv.org/pdf/2604.09213](https://arxiv.org/pdf/2604.09213)**

> **作者:** Nina Konovalova; Andrey Kuznetsov; Aibek Alanov
>
> **摘要:** Diffusion models have become leading approaches for high-fidelity image generation. Recent DiT-based diffusion models, in particular, achieve strong prompt adherence while producing high-quality samples. We propose SHIFT, a simple but effective and lightweight framework for concept removal in DiT diffusion models via targeted manipulation of intermediate activations at inference time, inspired by activation steering in large language models. SHIFT learns steering vectors that are dynamically applied to selected layers and timesteps to suppress unwanted visual concepts while preserving the prompt's remaining content and overall image quality. Beyond suppression, the same mechanism can shift generations into a desired \emph{style domain} or bias samples toward adding or changing target objects. We demonstrate that SHIFT provides effective and flexible control over DiT generation across diverse prompts and targets without time-consuming retraining.
>
---
#### [new 079] Dynamic Class-Aware Active Learning for Unbiased Satellite Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于卫星图像分割任务，解决标注成本高和类别不平衡问题。提出DCAU-AL方法，动态关注性能较差的类别，提升分割效果与标注效率。**

- **链接: [https://arxiv.org/pdf/2604.08965](https://arxiv.org/pdf/2604.08965)**

> **作者:** Gadi Hemanth Kumar; Athira Nambiar; Pankaj Bodani
>
> **摘要:** Semantic segmentation of satellite imagery plays a vital role in land cover mapping and environmental monitoring. However, annotating large-scale, high-resolution satellite datasets is costly and time consuming, especially when covering vast geographic regions. Instead of randomly labeling data or exhaustively annotating entire datasets, Active Learning (AL) offers an efficient alternative by intelligently selecting the most informative samples for annotation with the help of Human-in-the-loop (HITL), thereby reducing labeling costs while maintaining high model performance. AL is particularly beneficial for large-scale or resource-constrained satellite applications, as it enables high segmentation accuracy with significantly fewer labeled samples. Despite these advantages, standard AL strategies typically rely on global uncertainty or diversity measures and lack the adaptability to target underperforming or rare classes as training progresses, leading to bias in the system. To overcome these limitations, we propose a novel adaptive acquisition function, Dynamic Class-Aware Uncertainty based Active learning (DCAU-AL) that prioritizes sample selection based on real-time class-wise performance gaps, thereby overcoming class-imbalance issue. The proposed DCAU-AL mechanism continuously tracks the performance of the segmentation per class and dynamically adjusts the sampling weights to focus on poorly performing or underrepresented classes throughout the active learning process. Extensive experiments on the OpenEarth land cover dataset show that DCAU-AL significantly outperforms existing AL methods, especially under severe class imbalance, delivering superior per-class IoU and improved annotation efficiency.
>
---
#### [new 080] Robust by Design: A Continuous Monitoring and Data Integration Framework for Medical AI
- **分类: cs.CV**

- **简介: 该论文属于医学AI任务，解决动态环境中模型性能下降问题。提出框架通过持续监控和数据集成，保持模型鲁棒性，防止性能退化。**

- **链接: [https://arxiv.org/pdf/2604.09009](https://arxiv.org/pdf/2604.09009)**

> **作者:** Mohammad Daouk; Jan Ulrich Becker; Neeraja Kambham; Anthony Chang; Chandra Mohan; Hien Van Nguyen
>
> **备注:** Accepted at IEEE ISBI 2026. Chandra Mohan and Hien Van Nguyen jointly supervised this work
>
> **摘要:** Adaptive medical AI models often face performance drops in dynamic clinical environments due to data drift. We propose an autonomous continuous monitoring and data integration framework that maintains robust performance over time. Focusing on glomerular pathology image classification (proliferative vs. non-proliferative lupus nephritis), our three-stage method uses multi-metric feature analysis and Monte Carlo dropout-based uncertainty gating to decide when to retrain on new data. Only images statistically similar to the training distribution (via Euclidean, cosine, Mahalanobis metrics) and with low predictive entropy are integrated. The model is then incrementally retrained with these images under strict performance safeguards (no metric degradation >5%). In experiments with a ResNet18 ensemble on a multi-center dataset, the framework prevents performance degradation: new images were added without significant change in AUC (~0.92) or accuracy (~89%). This approach addresses data shift and avoids catastrophic forgetting, enabling sustained learning in medical imaging AI.
>
---
#### [new 081] Unified Multimodal Uncertain Inference
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出UMUI任务，解决多模态不确定性推理问题。构建了包含音频、视觉和视听概率标注的数据集，并引入CLUE方法提升模型预测校准性。**

- **链接: [https://arxiv.org/pdf/2604.08701](https://arxiv.org/pdf/2604.08701)**

> **作者:** Dengjia Zhang; Alexander Martin; William Jurayj; Kenton Murray; Benjamin Van Durme; Reno Kriz
>
> **摘要:** We introduce Unified Multimodal Uncertain Inference (UMUI), a multimodal inference task spanning text, audio, and video, where models must produce calibrated probability estimates of hypotheses conditioned on a premise in any modality or combination. While uncertain inference has been explored in text, extension to other modalities has been limited to single-modality binary entailment judgments, leaving no framework for fine-grained probabilistic reasoning in or across other modalities. To address this, we curate a human-annotated evaluation set with scalar probability judgments across audio, visual, and audiovisual settings, and additionally evaluate on existing text and audio benchmarks. We introduce CLUE (Calibrated Latent Uncertainty Estimation), which combines self-consistent teacher calibration and distribution-based confidence probing to produce calibrated predictions. We demonstrate that our 3B-parameter model achieves equivalent or stronger performance than baselines up to 32B parameters across all modalities.
>
---
#### [new 082] Benchmarking CNN- and Transformer-Based Models for Surgical Instrument Segmentation in Robotic-Assisted Surgery
- **分类: cs.CV; nlin.PS**

- **简介: 该论文属于医学图像分割任务，旨在提升机器人手术中手术器械的分割精度。通过对比多种模型，探讨其在复杂手术场景中的表现与优劣。**

- **链接: [https://arxiv.org/pdf/2604.09151](https://arxiv.org/pdf/2604.09151)**

> **作者:** Sara Ameli
>
> **摘要:** Accurate segmentation of surgical instruments in robotic-assisted surgery is critical for enabling context-aware computer-assisted interventions, such as tool tracking, workflow analysis, and autonomous decision-making. In this study, we benchmark five deep learning architectures-UNet, UNet, DeepLabV3, Attention UNet, and SegFormer on the SAR-RARP50 dataset for multi-class semantic segmentation of surgical instruments in real-world radical prostatectomy videos. The models are trained with a compound loss function combining Cross Entropy and Dice loss to address class imbalance and capture fine object boundaries. Our experiments reveal that while convolutional models such as UNet and Attention UNet provide strong baseline performance, DeepLabV3 achieves results comparable to SegFormer, demonstrating the effectiveness of atrous convolution and multi-scale context aggregation in capturing complex surgical scenes. Transformer-based architectures like SegFormer further enhance global contextual understanding, leading to improved generalization across varying instrument appearances and surgical conditions. This work provides a comprehensive comparison and practical insights for selecting segmentation models in surgical AI applications, highlighting the trade-offs between convolutional and transformer-based approaches.
>
---
#### [new 083] RIRF: Reasoning Image Restoration Framework
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决通用图像恢复中缺乏明确诊断推理的问题。提出R&R框架，结合结构化思维链与修复过程，提升恢复质量并增强可解释性。**

- **链接: [https://arxiv.org/pdf/2604.09511](https://arxiv.org/pdf/2604.09511)**

> **作者:** Wending Yan; Rongkai Zhang; Kaihua Tang; Yu Cheng; Qiankun Liu
>
> **摘要:** Universal image restoration (UIR) aims to recover clean images from diverse and unknown degradations using a unified model. Existing UIR methods primarily focus on pixel reconstruction and often lack explicit diagnostic reasoning over degradation composition, severity, and scene semantics prior to restoration. We propose Reason and Restore (R\&R), a novel framework that integrates structured Chain-of-Thought (CoT) reasoning into the image restoration pipeline. R\&R introduces an explicit reasoner, implemented by fine-tuning Qwen3-VL, to diagnose degradation types, quantify degradation severity, infer key degradation-related factors, and describe relevant scene and object semantics. The resulting structured reasoning provides interpretable and fine-grained diagnostic priors for the restorer. To further improve restoration quality, the quantified degradation severity produced by the reasoner is leveraged as reinforcement learning (RL) signals to guide and strengthen the restorer. Unlike existing multimodal LLM-based agentic systems that decouple reasoning from low-level vision tasks, R\&R tightly couples semantic diagnostic reasoning with pixel-level restoration in a unified framework. Extensive experiments across diverse UIR benchmarks demonstrate that R\&R achieves state-of-the-art performance while offering unique interpretability into the restoration process.
>
---
#### [new 084] GeRM: A Generative Rendering Model From Physically Realistic to Photorealistic
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决PBR与PRR之间的差距问题。通过构建GeRM模型，结合物理属性和文本提示，实现从物理真实到照片真实的可控生成。**

- **链接: [https://arxiv.org/pdf/2604.09304](https://arxiv.org/pdf/2604.09304)**

> **作者:** Jiayuan Lu; Rengan Xie; Xuancheng Jin; Zhizhen Wu; Qi Ye; Tian Xie; Hujun Bao; Rui Wang. Yuchi Huo
>
> **摘要:** For decades, Physically-Based Rendering (PBR) is the fundation of synthesizing photorealisitic images, and therefore sometimes roughly referred as Photorealistic Rendering (PRR). While PBR is indeed a mathematical simulation of light transport that guarantees physical reality, photorealism has additional reliance on the realistic digital model of geometry and appearance of the real world, leaving a barely explored gap from PBR to PRR (P2P). Consequently, the path toward photorealism faces a critical dilemma: the explicit simulation of PRR encumbered by unreachable realistic digital models for real-world existence, while implicit generation models sacrifice controllability and geometric consistency. Based on this insight, this paper presents the problem, data, and approach of mitigating P2P gap, followed by the first multi-modal generative rendering model, dubbed GeRM, to unify PBR and PRR. GeRM integrates physical attributes like G-buffers with text prompts, and progressive incremental injection to generate controllable photorealistic images, allowing users to fluidly navigate the continuum between strict physical fidelity and perceptual photorealism. Technically, we model the transition between PBR and PRR images as a distribution transfer and aim to learn a distribution transfer vector field (DTV Field) to guide this process. To define the learning objective, we first leverage a multi-agent VLM framework to construct an expert-guided pairwise P2P transfer dataset, named P2P-50K, where each paired sample in the dataset corresponds to a transfer vector in the DTV Field. Subsequently, we propose a multi-condition ControlNet to learn the DTV Field, which synthesizes PBR images and progressively transitions them into PRR images, guided by G-buffers, text prompts, and cues for enhanced regions.
>
---
#### [new 085] Seeing is Believing: Robust Vision-Guided Cross-Modal Prompt Learning under Label Noise
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型任务，旨在解决标签噪声下的提示学习鲁棒性问题。通过引入视觉引导的提示框架，提升模型对噪声标签的抵抗能力。**

- **链接: [https://arxiv.org/pdf/2604.09532](https://arxiv.org/pdf/2604.09532)**

> **作者:** Zibin Geng; Xuefeng Jiang; Jia Li; Zheng Li; Tian Wen; Lvhua Wu; Sheng Sun; Yuwei Wang; Min Liu
>
> **摘要:** Prompt learning is a parameter-efficient approach for vision-language models, yet its robustness under label noise is less investigated. Visual content contains richer and more reliable semantic information, which remains more robust under label noise. However, the prompt itself is highly susceptible to label noise. Motivated by this intuition, we propose VisPrompt, a lightweight and robust vision-guided prompt learning framework for noisy-label settings. Specifically, we exploit a cross-modal attention mechanism to reversely inject visual semantics into prompt representations. This enables the prompt tokens to selectively aggregate visual information relevant to the current sample, thereby improving robustness by anchoring prompt learning to stable instance-level visual evidence and reducing the influence of noisy supervision. To address the instability caused by using the same way of injecting visual information for all samples, despite differences in the quality of their visual cues, we further introduce a lightweight conditional modulation mechanism to adaptively control the strength of visual information injection, which strikes a more robust balance between text-side semantic priors and image-side instance evidence. The proposed framework effectively suppresses the noise-induced disturbances, reduce instability in prompt updates, and alleviate memorization of mislabeled samples. VisPrompt significantly improves robustness while keeping the pretrained VLM backbone frozen and introducing only a small amount of additional trainable parameters. Extensive experiments under synthetic and real-world label noise demonstrate that VisPrompt generally outperforms existing baselines on seven benchmark datasets and achieves stronger robustness. Our code is publicly available at this https URL.
>
---
#### [new 086] VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出VisionFoundry，通过合成数据提升VLM的视觉感知能力。解决VLM在空间理解和视角识别上的不足，通过任务驱动生成数据并验证效果。**

- **链接: [https://arxiv.org/pdf/2604.09531](https://arxiv.org/pdf/2604.09531)**

> **作者:** Guanyu Zhou; Yida Yin; Wenhao Chai; Shengbang Tong; Xingyu Fu; Zhuang Liu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Vision-language models (VLMs) still struggle with visual perception tasks such as spatial understanding and viewpoint recognition. One plausible contributing factor is that natural image datasets provide limited supervision for low-level visual skills. This motivates a practical question: can targeted synthetic supervision, generated from only a task keyword such as Depth Order, address these weaknesses? To investigate this question, we introduce VisionFoundry, a task-aware synthetic data generation pipeline that takes only the task name as input and uses large language models (LLMs) to generate questions, answers, and text-to-image (T2I) prompts, then synthesizes images with T2I models and verifies consistency with a proprietary VLM, requiring no reference images or human annotation. Using VisionFoundry, we construct VisionFoundry-10K, a synthetic visual question answering (VQA) dataset containing 10k image-question-answer triples spanning 10 tasks. Models trained on VisionFoundry-10K achieve substantial improvements on visual perception benchmarks: +7% on MMVP and +10% on CV-Bench-3D, while preserving broader capabilities and showing favorable scaling behavior as data size increases. Our results suggest that limited task-targeted supervision is an important contributor to this bottleneck and that synthetic supervision is a promising path toward more systematic training for VLMs.
>
---
#### [new 087] ActFER: Agentic Facial Expression Recognition via Active Tool-Augmented Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文属于面部表情识别任务，解决传统方法被动感知的问题，提出ActFER框架，通过主动视觉证据获取和多模态推理提升识别效果。**

- **链接: [https://arxiv.org/pdf/2604.08990](https://arxiv.org/pdf/2604.08990)**

> **作者:** Shifeng Liu; Zhengye Zhang; Sirui Zhao; Xinglong Mao; Zhehan Kan; Zhixiang Wei; Shiwei Wu; Chaoyou Fu; Tong Xu; Enhong Chen
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have created new opportunities for facial expression recognition (FER), moving it beyond pure label prediction toward reasoning-based affect understanding. However, existing MLLM-based FER methods still follow a passive paradigm: they rely on externally prepared facial inputs and perform single-pass reasoning over fixed visual evidence, without the capability for active facial perception. To address this limitation, we propose ActFER, an agentic framework that reformulates FER as active visual evidence acquisition followed by multimodal reasoning. Specifically, ActFER dynamically invokes tools for face detection and alignment, selectively zooms into informative local regions, and reasons over facial Action Units (AUs) and emotions through a visual Chain-of-Thought. To realize such behavior, we further develop Utility-Calibrated GRPO (UC-GRPO), a reinforcement learning algorithm tailored to agentic FER. UC-GRPO uses AU-grounded multi-level verifiable rewards to densify supervision, query-conditional contrastive utility estimation to enable sample-aware dynamic credit assignment for local inspection, and emotion-aware EMA calibration to reduce noisy utility estimates while capturing emotion-wise inspection tendencies. This algorithm enables ActFER to learn both when local inspection is beneficial and how to reason over the acquired evidence. Comprehensive experiments show that ActFER trained with UC-GRPO consistently outperforms passive MLLM-based FER baselines and substantially improves AU prediction accuracy.
>
---
#### [new 088] Precise Shield: Explaining and Aligning VLLM Safety via Neuron-Level Guidance
- **分类: cs.CV**

- **简介: 该论文属于模型安全任务，旨在解决VLLM在多语言和多模态攻击下的安全漏洞。通过识别关键安全神经元并限制参数更新，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2604.08881](https://arxiv.org/pdf/2604.08881)**

> **作者:** Enyi Shi; Fei Shen; Shuyi Miao; Linxia Zhu; Pengyang Shao; Jinhui Tang; Tat-Seng Chua
>
> **摘要:** In real-world deployments, Vision-Language Large Models (VLLMs) face critical challenges from multilingual and multimodal composite attacks: harmful images paired with low-resource language texts can easily bypass defenses designed for high-resource language scenarios, exposing structural blind spots in current cross-lingual and cross-modal safety methods. This raises a mechanistic question: where is safety capability instantiated within the model, and how is it distributed across languages and modalities? Prior studies on pure-text LLMs have identified cross-lingual shared safety neurons, suggesting that safety may be governed by a small subset of critical neurons. Leveraging this insight, we propose Precise Shield, a two-stage framework that first identifies safety neurons by contrasting activation patterns between harmful and benign inputs, and then constrains parameter updates strictly within this subspace via gradient masking with affecting fewer than 0.03% of parameters. This strategy substantially improves safety while preserving multilingual and multimodal generalization. Further analysis reveals a moderate overlap of safety neurons across languages and modalities, enabling zero-shot cross-lingual and cross-modal transfer of safety capabilities, and offering a new direction for neuron-level, transfer-based safety enhancement.
>
---
#### [new 089] Do Vision Language Models Need to Process Image Tokens?
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型研究，探讨图像标记是否必要。通过分析视觉表示的演化，发现深层处理并非总是必需，为模型优化提供新思路。**

- **链接: [https://arxiv.org/pdf/2604.09425](https://arxiv.org/pdf/2604.09425)**

> **作者:** Sambit Ghosh; R. Venkatesh Babu; Chirag Agarwal
>
> **备注:** Accepted (Oral) at TRUE-V Workshop CVPR 2026
>
> **摘要:** Vision Language Models (VLMs) have achieved remarkable success by integrating visual encoders with large language models (LLMs). While VLMs process dense image tokens across deep transformer stacks (incurring substantial computational overhead), it remains fundamentally unclear whether sustained image-token processing is necessary for their performance or visual representations meaningfully evolve from early to later layers. In this work, we systematically investigate the functional role of image tokens in VLMs and show that visual representations rapidly converge to a bounded-complexity regime, \ie their entropy stabilizes, intrinsic dimensionality compresses, and trajectory curvature approaches a near-constant profile. In contrast, textual representations continue to undergo substantial restructuring across depth. Once stabilized, visual representations become largely interchangeable between layers, indicating limited additional transformation in deeper stages. Further, depth-wise visual truncation reveals that the necessity of visual processing is task-dependent, where single-token predictions remain comparatively robust to truncated visual depth, but multi-token generation require sustained access to visual representations. Under deterministic decoding, reducing visual depth perturbs intermediate reasoning trajectories more strongly than final outputs, suggesting that image tokens influence the structure of reasoning more than the ultimate conclusions. Collectively, these findings \textbf{question the assumption} that deeper visual processing is uniformly essential in VLMs, challenging the current paradigm of multimodal LLM architectures.
>
---
#### [new 090] SiMing-Bench: Evaluating Procedural Correctness from Continuous Interactions in Clinical Skill Videos
- **分类: cs.CV; cs.CL; cs.HC**

- **简介: 该论文属于医疗视频分析任务，旨在评估模型对临床操作流程的正确性判断能力。研究提出SiMing-Bench基准，解决现有模型在流程状态跟踪和步骤正确性判断上的不足。**

- **链接: [https://arxiv.org/pdf/2604.09037](https://arxiv.org/pdf/2604.09037)**

> **作者:** Xiyang Huang; Jiawei Lin; Keying Wu; Jiaxin Huang; Kailai Yang; Renxiong Wei; Cheng zeng; Jiayi Xiang; Ziyan Kuang; Min Peng; Qianqian Xie; Sophia Ananiadou
>
> **摘要:** Current video benchmarks for multimodal large language models (MLLMs) focus on event recognition, temporal ordering, and long-context recall, but overlook a harder capability required for expert procedural judgment: tracking how ongoing interactions update the procedural state and thereby determine the correctness of later actions. We introduce SiMing-Bench, the first benchmark for evaluating this capability from full-length clinical skill videos. It targets rubric-grounded process-level judgment of whether interaction-driven state updates preserve procedural correctness across an entire workflow. SiMing-Bench is instantiated with SiMing-Score, a physician-annotated dataset of real clinical skill examination videos spanning cardiopulmonary resuscitation, automated external defibrillator operation, and bag-mask ventilation, each paired with a standardized step-wise rubric and dual-expert labels. Across diverse open- and closed-source MLLMs, we observe consistently weak agreement with physician judgments. Moreover, weak performance on rubric-defined intermediate steps persists even when overall procedure-level correlation appears acceptable, suggesting that coarse global assessment substantially overestimates current models' procedural judgment ability. Additional analyses with binary step judgment and step-aligned clips indicate that the bottleneck is not merely fine-grained scoring or temporal localization, but modeling how continuous interactions update procedural state over time.
>
---
#### [new 091] Region-Constrained Group Relative Policy Optimization for Flow-Based Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决指令引导下的区域精准编辑问题。针对现有方法的噪声信用分配问题，提出RC-GRPO-Editing框架，提升编辑区域遵循度与非目标区域保真。**

- **链接: [https://arxiv.org/pdf/2604.09386](https://arxiv.org/pdf/2604.09386)**

> **作者:** Zhuohan Ouyang; Zhe Qian; Wenhuo Cui; Chaoqun Wang
>
> **摘要:** Instruction-guided image editing requires balancing target modification with non-target preservation. Recently, flow-based models have emerged as a strong and increasingly adopted backbone for instruction-guided image editing, thanks to their high fidelity and efficient deterministic ODE sampling. Building on this foundation, GRPO-based reward-driven post-training has been explored to directly optimize editing-specific rewards, improving instruction following and editing consistency. However, existing methods often suffer from noisy credit assignment: global exploration also perturbs non-target regions, inflating within-group reward variance and yielding noisy GRPO advantages. To address this, we propose RC-GRPO-Editing, a region-constrained GRPO post-training framework for flow-based image editing under deterministic ODE sampling. It suppresses background-induced nuisance variance to enable cleaner localized credit assignment, improving editing region instruction adherence while preserving non-target content. Concretely, we localize exploration via region-decoupled initial noise perturbations to reduce background-induced reward variance and stabilize GRPO advantages, and introduce an attention concentration reward that aligns cross-attention with the intended editing region throughout the rollout, reducing unintended changes in non-target regions. Experiments on CompBench show consistent improvements in editing region instruction adherence and non-target preservation.
>
---
#### [new 092] Robust 4D Visual Geometry Transformer with Uncertainty-Aware Priors
- **分类: cs.CV**

- **简介: 该论文属于4D场景重建任务，旨在解决动态序列中几何模糊问题。通过引入不确定性感知机制，有效分离动态与静态成分，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2604.09366](https://arxiv.org/pdf/2604.09366)**

> **作者:** Ying Zang; Yidong Han; Chaotao Ding; Yuanqi Hu; Deyi Ji; Qi Zhu; Xuanfu Li; Jin Ma; Lingyun Sun; Tianrun Chen; Lanyun Zhu
>
> **摘要:** Reconstructing dynamic 4D scenes is an important yet challenging task. While 3D foundation models like VGGT excel in static settings, they often struggle with dynamic sequences where motion causes significant geometric ambiguity. To address this, we present a framework designed to disentangle dynamic and static components by modeling uncertainty across different stages of the reconstruction process. Our approach introduces three synergistic mechanisms: (1) Entropy-Guided Subspace Projection, which leverages information-theoretic weighting to adaptively aggregate multi-head attention distributions, effectively isolating dynamic motion cues from semantic noise; (2) Local-Consistency Driven Geometry Purification, which enforces spatial continuity via radius-based neighborhood constraints to eliminate structural outliers; and (3) Uncertainty-Aware Cross-View Consistency, which formulates multi-view projection refinement as a heteroscedastic maximum likelihood estimation problem, utilizing depth confidence as a probabilistic weight. Experiments on dynamic benchmarks show that our approach outperforms current state-of-the-art methods, reducing Mean Accuracy error by 13.43\% and improving segmentation F-measure by 10.49\%. Our framework maintains the efficiency of feed-forward inference and requires no task-specific fine-tuning or per-scene optimization.
>
---
#### [new 093] SenBen: Sensitive Scene Graphs for Explainable Content Moderation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于内容审核任务，旨在提升图像敏感行为的解释性。通过构建SenBen基准和优化模型，解决检测结果不透明的问题。**

- **链接: [https://arxiv.org/pdf/2604.08819](https://arxiv.org/pdf/2604.08819)**

> **作者:** Fatih Cagatay Akyon; Alptekin Temizel
>
> **备注:** Accepted at CVPRW 2026
>
> **摘要:** Content moderation systems classify images as safe or unsafe but lack spatial grounding and interpretability: they cannot explain what sensitive behavior was detected, who is involved, or where it occurs. We introduce the Sensitive Benchmark (SenBen), the first large-scale scene graph benchmark for sensitive content, comprising 13,999 frames from 157 movies annotated with Visual Genome-style scene graphs (25 object classes, 28 attributes including affective states such as pain, fear, aggression, and distress, 14 predicates) and 16 sensitivity tags across 5 categories. We distill a frontier VLM into a compact 241M student model using a multi-task recipe that addresses vocabulary imbalance in autoregressive scene graph generation through suffix-based object identity, Vocabulary-Aware Recall (VAR) Loss, and a decoupled Query2Label tag head with asymmetric loss, yielding a +6.4 percentage point improvement in SenBen Recall over standard cross-entropy training. On grounded scene graph metrics, our student model outperforms all evaluated VLMs except Gemini models and all commercial safety APIs, while achieving the highest object detection and captioning scores across all models, at $7.6\times$ faster inference and $16\times$ less GPU memory.
>
---
#### [new 094] Degradation-Robust Fusion: An Efficient Degradation-Aware Diffusion Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决复杂退化条件下的融合问题。提出一种高效退化感知扩散框架，通过隐式去噪和联合观测模型修正，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2604.08922](https://arxiv.org/pdf/2604.08922)**

> **作者:** Yu Shi; Yu Liu; Zhong-Cheng Wu; Juan Cheng; Huafeng Li; Xun Chen
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Complex degradations like noise, blur, and low resolution are typical challenges in real world image fusion tasks, limiting the performance and practicality of existing methods. End to end neural network based approaches are generally simple to design and highly efficient in inference, but their black-box nature leads to limited interpretability. Diffusion based methods alleviate this to some extent by providing powerful generative priors and a more structured inference process. However, they are trained to learn a single domain target distribution, whereas fusion lacks natural fused data and relies on modeling complementary information from multiple sources, making diffusion hard to apply directly in practice. To address these challenges, this paper proposes an efficient degradation aware diffusion framework for image fusion under arbitrary degradation scenarios. Specifically, instead of explicitly predicting noise as in conventional diffusion models, our method performs implicit denoising by directly regressing the fused image, enabling flexible adaptation to diverse fusion tasks under complex degradations with limited steps. Moreover, we design a joint observation model correction mechanism that simultaneously imposes degradation and fusion constraints during sampling to ensure high reconstruction accuracy. Experiments on diverse fusion tasks and degradation configurations demonstrate the superiority of the proposed method under complex degradation scenarios.
>
---
#### [new 095] HM-Bench: A Comprehensive Benchmark for Multimodal Large Language Models in Hyperspectral Remote Sensing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大语言模型在高光谱遥感中的应用任务，旨在解决MLLM在高光谱图像理解上的不足。工作包括构建HM-Bench基准和双模态评估框架，以评估模型性能。**

- **链接: [https://arxiv.org/pdf/2604.08884](https://arxiv.org/pdf/2604.08884)**

> **作者:** Xinyu Zhang; Zurong Mai; Qingmei Li; Zjin Liao; Yibin Wen; Yuhang Chen; Xiaoya Fan; Chan Tsz Ho; Bi Tianyuan; Haoyuan Liang; Ruifeng Su; Zihao Qian; Juepeng Zheng; Jianxi Huang; Yutong Lu; Haohuan Fu
>
> **摘要:** While multimodal large language models (MLLMs) have made significant strides in natural image understanding, their ability to perceive and reason over hyperspectral image (HSI) remains underexplored, which is a vital modality in remote sensing. The high dimensionality and intricate spectral-spatial properties of HSI pose unique challenges for models primarily trained on RGB this http URL address this gap, we introduce Hyperspectral Multimodal Benchmark (HM-Bench), the first benchmark designed specifically to evaluate MLLMs in HSI understanding. We curate a large-scale dataset of 19,337 question-answer pairs across 13 task categories, ranging from basic perception to spectral reasoning. Given that existing MLLMs are not equipped to process raw hyperspectral cubes natively, we propose a dual-modality evaluation framework that transforms HSI data into two complementary representations: PCA-based composite images and structured textual reports. This approach facilitates a systematic comparison of different representation for model performance. Extensive evaluations on 18 representative MLLMs reveal significant difficulties in handling complex spatial-spectral reasoning tasks. Furthermore, our results demonstrate that visual inputs generally outperform textual inputs, highlighting the importance of grounding in spectral-spatial evidence for effective HSI understanding. Dataset and appendix can be accessed at this https URL.
>
---
#### [new 096] EfficientSign: An Attention-Enhanced Lightweight Architecture for Indian Sign Language Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于手语识别任务，旨在开发轻量级模型高效识别印度手语。提出EfficientSign，结合注意力机制，在保持高准确率的同时减少参数量。**

- **链接: [https://arxiv.org/pdf/2604.08694](https://arxiv.org/pdf/2604.08694)**

> **作者:** Rishabh Gupta; Shravya R. Nalla
>
> **备注:** Submitted to IEEE Transactions on Human-Machine Systems
>
> **摘要:** How do you build a sign language recognizer that works on a phone? That question drove this work. We built EfficientSign, a lightweight model which takes EfficientNet-B0 and focuses on two attention modules (Squeeze-and-Excitation for channel focus, and a spatial attention layer that focuses on the hand gestures). We tested it against five other approaches on 12,637 images of Indian Sign Language alphabets, all 26 classes, using 5-fold cross-validation. EfficientSign achieves the accuracy of 99.94% (+/-0.05%), which matches the performance of ResNet18's 99.97% accuracy, but with 62% fewer parameters (4.2M vs 11.2M). We also experimented with feeding deep features (1,280-dimensional vectors pulled from EfficientNet-B0's pooling layer) into classical classifiers. SVM achieved the accuracy of 99.63%, Logistic Regression achieved the accuracy of 99.03% and KNN achieved accuracy of 96.33%. All of these blow past the 92% that SURF-based methods managed on a similar dataset back in 2015. Our results show that attention-enhanced learning model provides an efficient and deployable solution for ISL recognition without requiring a massive model or hand-tuned feature pipelines anymore.
>
---
#### [new 097] Strips as Tokens: Artist Mesh Generation with Native UV Segmentation
- **分类: cs.CV; cs.CG; cs.GR**

- **简介: 该论文属于三维网格生成任务，解决传统方法在令牌排序上效率低、破坏结构连续性的问题。提出SATO框架，通过三角形条带方式构建序列，提升几何质量和UV分割效果。**

- **链接: [https://arxiv.org/pdf/2604.09132](https://arxiv.org/pdf/2604.09132)**

> **作者:** Rui Xu; Dafei Qin; Kaichun Qiao; Qiujie Dong; Huaijin Pi; Qixuan Zhang; Longwen Zhang; Lan Xu; Jingyi Yu; Wenping Wang; Taku Komura
>
> **摘要:** Recent advancements in autoregressive transformers have demonstrated remarkable potential for generating artist-quality meshes. However, the token ordering strategies employed by existing methods typically fail to meet professional artist standards, where coordinate-based sorting yields inefficiently long sequences, and patch-based heuristics disrupt the continuous edge flow and structural regularity essential for high-quality modeling. To address these limitations, we propose Strips as Tokens (SATO), a novel framework with a token ordering strategy inspired by triangle strips. By constructing the sequence as a connected chain of faces that explicitly encodes UV boundaries, our method naturally preserves the organized edge flow and semantic layout characteristic of artist-created meshes. A key advantage of this formulation is its unified representation, enabling the same token sequence to be decoded into either a triangle or quadrilateral mesh. This flexibility facilitates joint training on both data types: large-scale triangle data provides fundamental structural priors, while high-quality quad data enhances the geometric regularity of the outputs. Extensive experiments demonstrate that SATO consistently outperforms prior methods in terms of geometric quality, structural coherence, and UV segmentation.
>
---
#### [new 098] Tora3: Trajectory-Guided Audio-Video Generation with Physical Coherence
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音视频生成任务，旨在解决运动与声音关系不协调的问题。提出Tora3框架，利用物体轨迹提升物理一致性，增强运动与声音同步。**

- **链接: [https://arxiv.org/pdf/2604.09057](https://arxiv.org/pdf/2604.09057)**

> **作者:** Junchao Liao; Zhenghao Zhang; Xiangyu Meng; Litao Li; Ziying Zhang; Siyu Zhu; Long Qin; Weizhi Wang
>
> **摘要:** Audio-video (AV) generation has recently made strong progress in perceptual quality and multimodal coherence, yet generating content with plausible motion-sound relations remains challenging. Existing methods often produce object motions that are visually unstable and sounds that are only loosely aligned with salient motion or contact events, largely because they lack an explicit motion-aware structure shared by video and audio generation. We present Tora3, a trajectory-guided AV generation framework that improves physical coherence by using object trajectories as a shared kinematic prior. Rather than treating trajectories as a video-only control signal, Tora3 uses them to jointly guide visual motion and acoustic events. Specifically, we design a trajectory-aligned motion representation for video, a kinematic-audio alignment module driven by trajectory-derived second-order kinematic states, and a hybrid flow matching scheme that preserves trajectory fidelity in trajectory-conditioned regions while maintaining local coherence elsewhere. We further curate PAV, a large-scale AV dataset emphasizing motion-relevant patterns with automatically extracted motion annotations. Extensive experiments show that Tora3 improves motion realism, motion-sound synchronization, and overall AV generation quality over strong open-source baselines.
>
---
#### [new 099] BlendFusion -- Scalable Synthetic Data Generation for Diffusion Model Training
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型训练中合成数据质量低的问题。提出BlendFusion框架，通过3D场景生成高质量图像与标题对，提升数据多样性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.09022](https://arxiv.org/pdf/2604.09022)**

> **作者:** Thejas Venkatesh; Suguna Varshini Velury
>
> **摘要:** With the rapid adoption of diffusion models, synthetic data generation has emerged as a promising approach for addressing the growing demand for large-scale image datasets. However, images generated purely by diffusion models often exhibit visual inconsistencies, and training models on such data can create an autophagous feedback loop that leads to model collapse, commonly referred to as Model Autophagy Disorder (MAD). To address these challenges, we propose BlendFusion, a scalable framework for synthetic data generation from 3D scenes using path tracing. Our pipeline incorporates an object-centric camera placement strategy, robust filtering mechanisms, and automatic captioning to produce high-quality image-caption pairs. Using this pipeline, we curate FineBLEND, an image-caption dataset constructed from a diverse set of 3D scenes. We empirically analyze the quality of FineBLEND and compare it to several widely used image-caption datasets. We also demonstrate the effectiveness of our object-centric camera placement strategy relative to object-agnostic sampling approaches. Our open-source framework is designed for high configurability, enabling the community to create their own datasets from 3D scenes.
>
---
#### [new 100] Learning Vision-Language-Action World Models for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自主驾驶任务，旨在解决VLA模型缺乏时间动态和全局一致性的问题。提出VLA-World模型，结合预测与推理，提升驾驶预见性。**

- **链接: [https://arxiv.org/pdf/2604.09059](https://arxiv.org/pdf/2604.09059)**

> **作者:** Guoqing Wang; Pin Tang; Xiangxuan Ren; Guodongfang Zhao; Bailan Feng; Chao Ma
>
> **备注:** Accepted by CVPR2026 findings
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved notable progress in end-to-end autonomous driving by integrating perception, reasoning, and control within a unified multimodal framework. However, they often lack explicit modeling of temporal dynamics and global world consistency, which limits their foresight and safety. In contrast, world models can simulate plausible future scenes but generally struggle to reason about or evaluate the imagined future they generate. In this work, we present VLA-World, a simple yet effective VLA world model that unifies predictive imagination with reflective reasoning to improve driving foresight. VLA-World first uses an action-derived feasible trajectory to guide the generation of the next-frame image, capturing rich spatial and temporal cues that describe how the surrounding environment evolves. The model then reasons over this self-generated future imagined frame to refine the predicted trajectory, achieving higher performance and better interpretability. To support this pipeline, we curate nuScenes-GR-20K, a generative reasoning dataset derived from nuScenes, and employ a three-stage training strategy that includes pretraining, supervised fine-tuning, and reinforcement learning. Extensive experiments demonstrate that VLA-World consistently surpasses state-of-the-art VLA and world-model baselines on both planning and future-generation benchmarks. Project page: this https URL
>
---
#### [new 101] AsymLoc: Towards Asymmetric Feature Matching for Efficient Visual Localization
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，旨在解决资源受限设备上高效精确定位的问题。通过师生模型的异构匹配，提出AsymLoc框架，在减少计算量的同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2604.09445](https://arxiv.org/pdf/2604.09445)**

> **作者:** Mohammad Omama; Gabriele Berton; Eric Foxlin; Yelin Kim
>
> **摘要:** Precise and real-time visual localization is critical for applications like AR/VR and robotics, especially on resource-constrained edge devices such as smart glasses, where battery life and heat dissipation can be a primary concerns. While many efficient models exist, further reducing compute without sacrificing accuracy is essential for practical deployment. To address this, we propose asymmetric visual localization: a large Teacher model processes pre-mapped database images offline, while a lightweight Student model processes the query image online. This creates a challenge in matching features from two different models without resorting to heavy, learned matchers. We introduce AsymLoc, a novel distillation framework that aligns a Student to its Teacher through a combination of a geometry-driven matching objective and a joint detector-descriptor distillation objective, enabling fast, parameter-less nearest-neighbor matching. Extensive experiments on HPatches, ScanNet, IMC2022, and Aachen show that AsymLoc achieves up to 95% of the teacher's localization accuracy using an order of magnitude smaller models, significantly outperforming existing baselines and establishing a new state-of-the-art efficiency-accuracy trade-off.
>
---
#### [new 102] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Images via Visual Prompt Injection
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于隐私保护任务，旨在防止多模态大语言模型分析图像。通过嵌入视觉提示扰动，使模型拒绝回答，有效保护用户图像隐私。**

- **链接: [https://arxiv.org/pdf/2604.09024](https://arxiv.org/pdf/2604.09024)**

> **作者:** Zedian Shao; Hongbin Liu; Yuepeng Hu; Neil Zhenqiang Gong
>
> **备注:** Appeared in ACL 2026 main conference
>
> **摘要:** Multi-modal large language models (MLLMs) have emerged as powerful tools for analyzing Internet-scale image data, offering significant benefits but also raising critical safety and societal concerns. In particular, open-weight MLLMs may be misused to extract sensitive information from personal images at scale, such as identities, locations, or other private details. In this work, we propose ImageProtector, a user-side method that proactively protects images before sharing by embedding a carefully crafted, nearly imperceptible perturbation that acts as a visual prompt injection attack on MLLMs. As a result, when an adversary analyzes a protected image with an MLLM, the MLLM is consistently induced to generate a refusal response such as "I'm sorry, I can't help with that request." We empirically demonstrate the effectiveness of ImageProtector across six MLLMs and four datasets. Additionally, we evaluate three potential countermeasures, Gaussian noise, DiffPure, and adversarial training, and show that while they partially mitigate the impact of ImageProtector, they simultaneously degrade model accuracy and/or efficiency. Our study focuses on the practically important setting of open-weight MLLMs and large-scale automated image analysis, and highlights both the promise and the limitations of perturbation-based privacy protection.
>
---
#### [new 103] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的置信度校准任务，旨在解决LVLMs中高置信度错误回答的问题。通过分离视觉与推理置信度，提升模型校准效果和视觉推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.09529](https://arxiv.org/pdf/2604.09529)**

> **作者:** Wenyi Xiao; Xinchi Xu; Leilei Gan
>
> **备注:** 24 pages, ACL 2026 Main. Repository: this https URL
>
> **摘要:** Large Vision Language Models (LVLMs) achieve strong multimodal reasoning but frequently exhibit hallucinations and incorrect responses with high certainty, which hinders their usage in high-stakes domains. Existing verbalized confidence calibration methods, largely developed for text-only LLMs, typically optimize a single holistic confidence score using binary answer-level correctness. This design is mismatched to LVLMs: an incorrect prediction may arise from perceptual failures or from reasoning errors given correct perception, and a single confidence conflates these sources while visual uncertainty is often dominated by language priors. To address these issues, we propose VL-Calibration, a reinforcement learning framework that explicitly decouples confidence into visual and reasoning confidence. To supervise visual confidence without ground-truth perception labels, we introduce an intrinsic visual certainty estimation that combines (i) visual grounding measured by KL-divergence under image perturbations and (ii) internal certainty measured by token entropy. We further propose token-level advantage reweighting to focus optimization on tokens based on visual certainty, suppressing ungrounded hallucinations while preserving valid perception. Experiments on thirteen benchmarks show that VL-Calibration effectively improves calibration while boosting visual reasoning accuracy, and it generalizes to out-of-distribution benchmarks across model scales and architectures.
>
---
#### [new 104] InsEdit: Towards Instruction-based Visual Editing via Data-Efficient Video Diffusion Models Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在解决视频生成模型转为编辑器时数据需求大的问题。通过少量数据和架构改进，实现高效视频编辑。**

- **链接: [https://arxiv.org/pdf/2604.08646](https://arxiv.org/pdf/2604.08646)**

> **作者:** Zhefan Rao; Bin Zou; Haoxuan Che; Xuanhua He; Chong Hou Choi; Yanheng Li; Rui Liu; Qifeng Chen
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** Instruction-based video editing is a natural way to control video content with text, but adapting a video generation model into an editor usually appears data-hungry. At the same time, high-quality video editing data remains scarce. In this paper, we show that a video generation backbone can become a strong video editor without large scale video editing data. We present InsEdit, an instruction-based editing model built on HunyuanVideo-1.5. InsEdit combines a visual editing architecture with a video data pipeline based on Mutual Context Attention (MCA), which creates aligned video pairs where edits can begin in the middle of a clip rather than only from the first frame. With only O(100)K video editing data, InsEdit achieves state-of-the-art results among open-source methods on our video instruction editing benchmarks. In addition, because our training recipe also includes image editing data, the final model supports image editing without any modification.
>
---
#### [new 105] Beyond Segmentation: Structurally Informed Facade Parsing from Imperfect Images
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于建筑立面解析任务，解决结构不连贯问题。通过改进YOLOv8的损失函数，提升边界框的几何一致性，增强结构规整性。**

- **链接: [https://arxiv.org/pdf/2604.09260](https://arxiv.org/pdf/2604.09260)**

> **作者:** Maciej Janicki; Aleksander Plocharski; Przemyslaw Musialski
>
> **备注:** 4 pages, 4 figures, EUROGRAPHICS 2026 Short Paper
>
> **摘要:** Standard object detectors typically treat architectural elements independently, often resulting in facade parsings that lack the structural coherence required for downstream procedural reconstruction. We address this limitation by augmenting the YOLOv8 training objective with a custom lightweight alignment loss. This regularization encourages grid-consistent arrangements of bounding boxes during training, effectively injecting geometric priors without altering the standard inference pipeline. Experiments on the CMP dataset demonstrate that our method successfully improves structural regularity, correcting alignment errors caused by perspective and occlusion while maintaining a controllable trade-off with standard detection accuracy.
>
---
#### [new 106] Domain-generalizable Face Anti-Spoofing with Patch-based Multi-tasking and Artifact Pattern Conversion
- **分类: cs.CV**

- **简介: 该论文属于人脸反欺骗任务，解决FAS在未知域和攻击方法下的泛化能力不足问题。提出PCGAN模型，通过生成多样伪影和多任务学习提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.09018](https://arxiv.org/pdf/2604.09018)**

> **作者:** Seungjin Jung; Yonghyun Jeong; Minha Kim; Jimin Min; Youngjoon Yoo; Jongwon Choi
>
> **备注:** The published version is available at DOI: this https URL
>
> **摘要:** Face Anti-Spoofing (FAS) algorithms, designed to secure face recognition systems against spoofing, struggle with limited dataset diversity, impairing their ability to handle unseen visual domains and spoofing methods. We introduce the Pattern Conversion Generative Adversarial Network (PCGAN) to enhance domain generalization in FAS. PCGAN effectively disentangles latent vectors for spoof artifacts and facial features, allowing to generate images with diverse artifacts. We further incorporate patch-based and multi-task learning to tackle partial attacks and overfitting issues to facial features. Our extensive experiments validate PCGAN's effectiveness in domain generalization and detecting partial attacks, giving a substantial improvement in facial recognition security.
>
---
#### [new 107] Fast Model-guided Instance-wise Adaptation Framework for Real-world Pansharpening with Fidelity Constraints
- **分类: cs.CV**

- **简介: 该论文属于遥感图像处理中的 pansharpening 任务，旨在解决传统方法训练成本高、泛化能力差的问题。提出 FMG-Pan 框架，实现快速且高质量的图像融合。**

- **链接: [https://arxiv.org/pdf/2604.08903](https://arxiv.org/pdf/2604.08903)**

> **作者:** Zhiqi Yang; Jin-Liang Xiao; Shan Yin; Liang-Jian Deng; Gemine Vivone
>
> **摘要:** Pansharpening aims to generate high-resolution multispectral (HRMS) images by fusing low-resolution multispectral (LRMS) and high-resolution panchromatic (PAN) images while preserving both spectral and spatial information. Although deep learning (DL)-based pansharpening methods achieve impressive performance, they require high training cost and large datasets, and often degrade when the test distribution differs from training, limiting generalization. Recent zero-shot methods, trained on a single PAN/LRMS pair, offer strong generalization but suffer from limited fusion quality, high computational overhead, and slow convergence. To address these issues, we propose FMG-Pan, a fast and generalizable model-guided instance-wise adaptation framework for real-world pansharpening, achieving both cross-sensor generality and rapid training-inference. The framework leverages a pretrained model to guide a lightweight adaptive network through joint optimization with spectral and physical fidelity constraints. We further design a novel physical fidelity term to enhance spatial detail preservation. Extensive experiments on real-world datasets under both intra- and cross-sensor settings demonstrate state-of-the-art performance. On the WorldView-3 dataset, FMG-Pan completes training and inference for a 512x512x8 image within 3 seconds on an RTX 3090 GPU, significantly faster than existing zero-shot methods, making it suitable for practical deployment.
>
---
#### [new 108] Deep Light Pollution Removal in Night Cityscape Photographs
- **分类: cs.CV**

- **简介: 该论文属于夜间图像修复任务，旨在解决光污染导致的夜景失真问题。通过构建物理退化模型和训练策略，有效减少光污染 artifacts，恢复真实夜景。**

- **链接: [https://arxiv.org/pdf/2604.09145](https://arxiv.org/pdf/2604.09145)**

> **作者:** Hao Wang; Xiaolin Wu; Xi Zhang; Baoqing Sun
>
> **备注:** 17 pages, supplementary material included
>
> **摘要:** Nighttime photography is severely degraded by light pollution induced by pervasive artificial lighting in urban environments. After long-range scattering and spatial diffusion, unwanted artificial light overwhelms natural night luminance, generates skyglow that washes out the view of stars and celestial objects and produces halos and glow artifacts around light sources. Unlike nighttime dehazing, which aims to improve detail legibility through thick air, the objective of light pollution removal is to restore the pristine night appearance by neutralizing the radiative footprint of ground lighting. In this paper we introduce a physically-based degradation model that adds to the previous ones for nighttime dehazing two critical aspects; (i) anisotropic spread of directional light sources, and (ii) skyglow caused by invisible surface lights behind skylines. In addition, we construct a training strategy that leverages large generative model and synthetic-real coupling to compensate for the scarcity of paired real data and enhance generalization. Extensive experiments demonstrate that the proposed formulation and learning framework substantially reduce light pollution artifacts and better recover authentic night imagery than prior nighttime restoration methods.
>
---
#### [new 109] Mosaic: Multimodal Jailbreak against Closed-Source VLMs via Multi-View Ensemble Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全攻击任务，旨在解决封闭式VLMs的多模态越狱问题。提出Mosaic框架，通过多视角优化提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2604.09253](https://arxiv.org/pdf/2604.09253)**

> **作者:** Yuqin Lan; Gen Li; Yuanze Hu; Weihao Shen; Zhaoxin Fan; Faguo Wu; Xiao Zhang; Laurence T. Yang; Zhiming Zheng
>
> **备注:** 14pages, 9 figures
>
> **摘要:** Vision-Language Models (VLMs) are powerful but remain vulnerable to multimodal jailbreak attacks. Existing attacks mainly rely on either explicit visual prompt attacks or gradient-based adversarial optimization. While the former is easier to detect, the latter produces subtle perturbations that are less perceptible, but is usually optimized and evaluated under homogeneous open-source surrogate-target settings, leaving its effectiveness on commercial closed-source VLMs under heterogeneous settings unclear. To examine this issue, we study different surrogate-target settings and observe a consistent gap between homogeneous and heterogeneous settings, a phenomenon we term surrogate dependency. Motivated by this finding, we propose Mosaic, a Multi-view ensemble optimization framework for multimodal jailbreak against closed-source VLMs, which alleviates surrogate dependency under heterogeneous surrogate-target settings by reducing over-reliance on any single surrogate model and visual view. Specifically, Mosaic incorporates three core components: a Text-Side Transformation module, which perturbs refusal-sensitive lexical patterns; a Multi-View Image Optimization module, which updates perturbations under diverse cropped views to avoid overfitting to a single visual view; and a Surrogate Ensemble Guidance module, which aggregates optimization signals from multiple surrogate VLMs to reduce surrogate-specific bias. Extensive experiments on safety benchmarks demonstrate that Mosaic achieves state-of-the-art Attack Success Rate and Average Toxicity against commercial closed-source VLMs.
>
---
#### [new 110] MAG-3D: Multi-Agent Grounded Reasoning for 3D Understanding
- **分类: cs.CV; cs.MA**

- **简介: 该论文提出MAG-3D，解决3D场景中基于视觉语言模型的接地推理问题，通过多智能体协作实现无需训练的灵活推理。**

- **链接: [https://arxiv.org/pdf/2604.09167](https://arxiv.org/pdf/2604.09167)**

> **作者:** Henry Zheng; Chenyue Fang; Rui Huang; Siyuan Wei; Xiao Liu; Gao Huang
>
> **摘要:** Vision-language models (VLMs) have achieved strong performance in multimodal understanding and reasoning, yet grounded reasoning in 3D scenes remains underexplored. Effective 3D reasoning hinges on accurate grounding: to answer open-ended queries, a model must first identify query-relevant objects and regions in a complex scene, and then reason about their spatial and geometric relationships. Recent approaches have demonstrated strong potential for grounded 3D reasoning. However, they often rely on in-domain tuning or hand-crafted reasoning pipelines, which limit their flexibility and zero-shot generalization to novel environments. In this work, we present MAG-3D, a training-free multi-agent framework for grounded 3D reasoning with off-the-shelf VLMs. Instead of relying on task-specific training or fixed reasoning procedures, MAG-3D dynamically coordinates expert agents to address the key challenges of 3D reasoning. Specifically, we propose a planning agent that decomposes the task and orchestrates the overall reasoning process, a grounding agent that performs free-form 3D grounding and relevant frame retrieval from extensive 3D scene observations, and a coding agent that conducts flexible geometric reasoning and explicit verification through executable programs. This multi-agent collaborative design enables flexible training-free 3D grounded reasoning across diverse scenes and achieves state-of-the-art performance on challenging benchmarks.
>
---
#### [new 111] EGLOCE: Training-Free Energy-Guided Latent Optimization for Concept Erasure
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决概念擦除问题。提出EGLOCE方法，在推理阶段通过能量引导优化，无需训练即可有效移除特定概念，保持图像质量和提示一致性。**

- **链接: [https://arxiv.org/pdf/2604.09405](https://arxiv.org/pdf/2604.09405)**

> **作者:** Junyeong Ahn; Seojin Yoon; Sungyong Baik
>
> **摘要:** As text-to-image diffusion models grow increasingly prevalent, the ability to remove specific concepts-mostly explicit content and many copyrighted characters or styles-has become essential for safety and compliance. Existing unlearning approaches often require costly re-training, modify parameters at the cost of degradation of unrelated concept fidelity, or depend on indirect inference-time adjustment that compromise the effectiveness of concept erasure. Inspired by the success of energy-guided sampling for preservation of the condition of diffusion models, we introduce Energy-Guided Latent Optimization for Concept Erasure (EGLOCE), a training-free approach that removes unwanted concepts by re-directing noisy latent during inference. Our method employs a dual-objective framework: a repulsion energy that steers generation away from target concepts via gradient descent in latent space, and a retention energy that preserves semantic alignment to the original prompt. Combined with previous approaches that either require erroneous modified model weights or provide weak inference-time guidance, EGLOCE operates entirely at inference and enhances erasure performance, enabling plug-and-play integration. Extensive experiments demonstrate that EGLOCE improves concept removal while maintaining image quality and prompt alignment across baselines, even with adversarial attacks. To the best of our knowledge, our work is the first to establish a new paradigm for safe and controllable image generation through dual energy-based guidance during sampling.
>
---
#### [new 112] PhysInOne: Visual Physics Learning and Reasoning in One Suite
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出PhysInOne，一个大规模合成数据集，解决AI缺乏物理基础训练数据的问题。用于物理学习与推理任务，涵盖多种物理现象，提升模型的物理合理性。**

- **链接: [https://arxiv.org/pdf/2604.09415](https://arxiv.org/pdf/2604.09415)**

> **作者:** Siyuan Zhou; Hejun Wang; Hu Cheng; Jinxi Li; Dongsheng Wang; Junwei Jiang; Yixiao Jin; Jiayue Huang; Shiwei Mao; Shangjia Liu; Yafei Yang; Hongkang Song; Shenxing Wei; Zihui Zhang; Peng Huang; Shijie Liu; Zhengli Hao; Hao Li; Yitian Li; Wenqi Zhou; Zhihan Zhao; Zongqi He; Hongtao Wen; Shouwang Huang; Peng Yun; Bowen Cheng; Pok Kazaf Fu; Wai Kit Lai; Jiahao Chen; Kaiyuan Wang; Zhixuan Sun; Ziqi Li; Haochen Hu; Di Zhang; Chun Ho Yuen; Bing Wang; Zhihua Wang; Chuhang Zou; Bo Yang
>
> **备注:** CVPR 2026. Siyuan, Hejun, Hu, Jinxi, Dongsheng, Junwei, Yixiao, Jiayue, and Shiwei are co-first authors. Project page: this https URL
>
> **摘要:** We present PhysInOne, a large-scale synthetic dataset addressing the critical scarcity of physically-grounded training data for AI systems. Unlike existing datasets limited to merely hundreds or thousands of examples, PhysInOne provides 2 million videos across 153,810 dynamic 3D scenes, covering 71 basic physical phenomena in mechanics, optics, fluid dynamics, and magnetism. Distinct from previous works, our scenes feature multiobject interactions against complex backgrounds, with comprehensive ground-truth annotations including 3D geometry, semantics, dynamic motion, physical properties, and text descriptions. We demonstrate PhysInOne's efficacy across four emerging applications: physics-aware video generation, long-/short-term future frame prediction, physical property estimation, and motion transfer. Experiments show that fine-tuning foundation models on PhysInOne significantly enhances physical plausibility, while also exposing critical gaps in modeling complex physical dynamics and estimating intrinsic properties. As the largest dataset of its kind, orders of magnitude beyond prior works, PhysInOne establishes a new benchmark for advancing physics-grounded world models in generation, simulation, and embodied AI.
>
---
#### [new 113] TouchAnything: Diffusion-Guided 3D Reconstruction from Sparse Robot Touches
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决从稀疏触觉数据中准确重建物体几何的问题。通过迁移视觉扩散模型的先验知识，提升触觉重建精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.08945](https://arxiv.org/pdf/2604.08945)**

> **作者:** Langzhe Gu; Hung-Jui Huang; Mohamad Qadri; Michael Kaess; Wenzhen Yuan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Accurate object geometry estimation is essential for many downstream tasks, including robotic manipulation and physical interaction. Although vision is the dominant modality for shape perception, it becomes unreliable under occlusions or challenging lighting conditions. In such scenarios, tactile sensing provides direct geometric information through physical contact. However, reconstructing global 3D geometry from sparse local touches alone is fundamentally underconstrained. We present TouchAnything, a framework that leverages a pretrained large-scale 2D vision diffusion model as a semantic and geometric prior for 3D reconstruction from sparse tactile measurements. Unlike prior work that trains category-specific reconstruction networks or learns diffusion models directly from tactile data, we transfer the geometric knowledge encoded in pretrained visual diffusion models to the tactile domain. Given sparse contact constraints and a coarse class-level description of the object, we formulate reconstruction as an optimization problem that enforces tactile consistency while guiding solutions toward shapes consistent with the diffusion prior. Our method reconstructs accurate geometries from only a few touches, outperforms existing baselines, and enables open-world 3D reconstruction of previously unseen object instances. Our project page is this https URL .
>
---
#### [new 114] Neural Distribution Prior for LiDAR Out-of-Distribution Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于LiDAR的OOD检测任务，解决开放世界中模型无法识别异常对象的问题。提出NDP框架，通过建模预测分布和生成辅助样本提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.09232](https://arxiv.org/pdf/2604.09232)**

> **作者:** Zizhao Li; Zhengkang Xiang; Jiayang Ao; Feng Liu; Joseph West; Kourosh Khoshelham
>
> **备注:** CVPR 2026
>
> **摘要:** LiDAR-based perception is critical for autonomous driving due to its robustness to poor lighting and visibility conditions. Yet, current models operate under the closed-set assumption and often fail to recognize unexpected out-of-distribution (OOD) objects in the open world. Existing OOD scoring functions exhibit limited performance because they ignore the pronounced class imbalance inherent in LiDAR OOD detection and assume a uniform class distribution. To address this limitation, we propose the Neural Distribution Prior (NDP), a framework that models the distributional structure of network predictions and adaptively reweights OOD scores based on alignment with a learned distribution prior. NDP dynamically captures the logit distribution patterns of training data and corrects class-dependent confidence bias through an attention-based module. We further introduce a Perlin noise-based OOD synthesis strategy that generates diverse auxiliary OOD samples from input scans, enabling robust OOD training without external datasets. Extensive experiments on the SemanticKITTI and STU benchmarks demonstrate that NDP substantially improves OOD detection performance, achieving a point-level AP of 61.31\% on the STU test set, which is more than 10$\times$ higher than the previous best result. Our framework is compatible with various existing OOD scoring formulations, providing an effective solution for open-world LiDAR perception.
>
---
#### [new 115] Detection of Hate and Threat in Digital Forensics: A Case-Driven Multimodal Approach
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于数字取证中的仇恨与威胁检测任务，解决多模态证据中准确识别有害内容的问题。提出一种案例驱动的多模态方法，结合文本分析与视觉语言模型，提升检测的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.08609](https://arxiv.org/pdf/2604.08609)**

> **作者:** Ponkoj Chandra Shill
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Digital forensic investigations increasingly rely on heterogeneous evidence such as images, scanned documents, and contextual reports. These artifacts may contain explicit or implicit expressions of harm, hate, threat, violence, or intimidation, yet existing automated approaches often assume clean text input or apply vision models without forensic justification. This paper presents a case-driven multimodal approach for hate and threat detection in forensic analysis. The proposed framework explicitly determines the presence and source of textual evidence, distinguishing between embedded text, associated contextual text, and image-only evidence. Based on the identified evidence configuration, the framework selectively applies text analysis, multimodal fusion, or image-only semantic reasoning using vision language models with vision transformer backbones (ViT). By conditioning inference on evidence availability, the approach mirrors forensic decision-making, improves evidentiary traceability, and avoids unjustified modality assumptions. Experimental evaluation on forensic-style image evidence demonstrates consistent and interpretable behavior across heterogeneous evidence scenarios.
>
---
#### [new 116] TAIHRI: Task-Aware 3D Human Keypoints Localization for Close-Range Human-Robot Interaction
- **分类: cs.CV**

- **简介: 该论文提出TAIHRI，解决近距离人机交互中关键人体部位的3D定位问题，通过视觉语言模型实现任务导向的精准定位。**

- **链接: [https://arxiv.org/pdf/2604.08921](https://arxiv.org/pdf/2604.08921)**

> **作者:** Ao Li; Yonggen Ling; Yiyang Lin; Yuji Wang; Yong Deng; Yansong Tang
>
> **摘要:** Accurate 3D human keypoints localization is a critical technology enabling robots to achieve natural and safe physical interaction with users. Conventional 3D human keypoints estimation methods primarily focus on the whole-body reconstruction quality relative to the root joint. However, in practical human-robot interaction (HRI) scenarios, robots are more concerned with the precise metric-scale spatial localization of task-relevant body parts under the egocentric camera 3D coordinate. We propose TAIHRI, the first Vision-Language Model (VLM) tailored for close-range HRI perception, capable of understanding users' motion commands and directing the robot's attention to the most task-relevant keypoints. By quantizing 3D keypoints into a finite interaction space, TAIHRI precisely localize the 3D spatial coordinates of critical body parts by 2D keypoint reasoning via next token prediction, and seamlessly adapt to downstream tasks such as natural language control or global space human mesh recovery. Experiments on egocentric interaction benchmarks demonstrate that TAIHRI achieves superior estimation accuracy for task-critical body parts. We believe TAIHRI opens new research avenues in the field of embodied human-robot interaction. Code is available at: this https URL.
>
---
#### [new 117] PinpointQA: A Dataset and Benchmark for Small Object-Centric Spatial Understanding in Indoor Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PinpointQA，解决室内视频中小物体空间理解问题，包含四个渐进任务，用于评估和提升模型定位能力。**

- **链接: [https://arxiv.org/pdf/2604.08991](https://arxiv.org/pdf/2604.08991)**

> **作者:** Zhiyu Zhou; Peilin Liu; Ruoxuan Zhang; Luyang Zhang; Cheng Zhang; Hongxia Xie; Wen-Huang Cheng
>
> **摘要:** Small object-centric spatial understanding in indoor videos remains a significant challenge for multimodal large language models (MLLMs), despite its practical value for object search and assistive applications. Although existing benchmarks have advanced video spatial intelligence, embodied reasoning, and diagnostic perception, no existing benchmark directly evaluates whether a model can localize a target object in video and express its position with sufficient precision for downstream use. In this work, we introduce PinpointQA, the first dataset and benchmark for small object-centric spatial understanding in indoor videos. Built from ScanNet++ and ScanNet200, PinpointQA comprises 1,024 scenes and 10,094 QA pairs organized into four progressively challenging tasks: Target Presence Verification (TPV), Nearest Reference Identification (NRI), Fine-Grained Spatial Description (FSD), and Structured Spatial Prediction (SSP). The dataset is built from intermediate spatial representations, with QA pairs generated automatically and further refined through quality control. Experiments on representative MLLMs reveal a consistent capability gap along the progressive chain, with SSP remaining particularly difficult. Supervised fine-tuning on PinpointQA yields substantial gains, especially on the harder tasks, demonstrating that PinpointQA serves as both a diagnostic benchmark and an effective training dataset. The dataset and project page are available at this https URL.
>
---
#### [new 118] Envisioning the Future, One Step at a Time
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于未来场景预测任务，解决长时序、多模态运动预测问题。提出基于稀疏点轨迹的扩散模型，实现高效、多样化的未来场景生成。**

- **链接: [https://arxiv.org/pdf/2604.09527](https://arxiv.org/pdf/2604.09527)**

> **作者:** Stefan Andreas Baumann; Jannik Wiese; Tommaso Martorella; Mahdi M. Kalayeh; Björn Ommer
>
> **备注:** CVPR 2026. For code and models, see this http URL
>
> **摘要:** Accurately anticipating how complex, diverse scenes will evolve requires models that represent uncertainty, simulate along extended interaction chains, and efficiently explore many plausible futures. Yet most existing approaches rely on dense video or latent-space prediction, expending substantial capacity on dense appearance rather than on the underlying sparse trajectories of points in the scene. This makes large-scale exploration of future hypotheses costly and limits performance when long-horizon, multi-modal motion is essential. We address this by formulating the prediction of open-set future scene dynamics as step-wise inference over sparse point trajectories. Our autoregressive diffusion model advances these trajectories through short, locally predictable transitions, explicitly modeling the growth of uncertainty over time. This dynamics-centric representation enables fast rollout of thousands of diverse futures from a single image, optionally guided by initial constraints on motion, while maintaining physical plausibility and long-range coherence. We further introduce OWM, a benchmark for open-set motion prediction based on diverse in-the-wild videos, to evaluate accuracy and variability of predicted trajectory distributions under real-world uncertainty. Our method matches or surpasses dense simulators in predictive accuracy while achieving orders-of-magnitude higher sampling speed, making open-set future prediction both scalable and practical. Project page: this http URL.
>
---
#### [new 119] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于地理科学与遥感领域的多模态任务，旨在解决领域知识不足、感知与推理能力弱的问题。提出GeoMMBench评估基准和GeoMMAgent框架，提升专家级地理空间解析能力。**

- **链接: [https://arxiv.org/pdf/2604.08896](https://arxiv.org/pdf/2604.08896)**

> **作者:** Aoran Xiao; Shihao Cheng; Yonghao Xu; Yexian Ren; Hongruixuan Chen; Naoto Yokoya
>
> **备注:** CVPR 2026 Highlight paper
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have accelerated progress in domain-oriented AI, yet their development in geoscience and remote sensing (RS) remains constrained by distinctive challenges: wide-ranging disciplinary knowledge, heterogeneous sensor modalities, and a fragmented spectrum of tasks. To bridge these gaps, we introduce GeoMMBench, a comprehensive multimodal question-answering benchmark covering diverse RS disciplines, sensors, and tasks, enabling broader and more rigorous evaluation than prior benchmarks. Using GeoMMBench, we assess 36 open-source and proprietary large language models, uncovering systematic deficiencies in domain knowledge, perceptual grounding, and reasoning--capabilities essential for expert-level geospatial interpretation. Beyond evaluation, we propose GeoMMAgent, a multi-agent framework that strategically integrates retrieval, perception, and reasoning through domain-specific RS models and tools. Extensive experimental results demonstrate that GeoMMAgent significantly outperforms standalone LLMs, underscoring the importance of tool-augmented agents for dynamically tackling complex geoscience and RS challenges.
>
---
#### [new 120] MedFormer-UR: Uncertainty-Routed Transformer for Medical Image Classification
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，解决模型过自信和透明度不足的问题。通过引入不确定性路由机制和原型学习，提升模型校准性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.08868](https://arxiv.org/pdf/2604.08868)**

> **作者:** Mohammed Maaz Sibhai; Abedalrhman Alkhateeb; Saad B. Ahmed
>
> **摘要:** To ensure safe clinical integration, deep learning models must provide more than just high accuracy; they require dependable uncertainty quantification. While current Medical Vision Transformers perform well, they frequently struggle with overconfident predictions and a lack of transparency, issues that are magnified by the noisy and imbalanced nature of clinical data. To address this, we enhanced the modified Medical Transformer (MedFormer) that incorporates prototype-based learning and uncertainty-guided routing, by utilizing a Dirichlet distribution for per-token evidential uncertainty, our framework can quantify and localize ambiguity in real-time. This uncertainty is not just an output but an active participant in the training process, filtering out unreliable feature updates. Furthermore, the use of class-specific prototypes ensures the embedding space remains structured, allowing for decisions based on visual similarity. Testing across four modalities (mammography, ultrasound, MRI, and histopathology) confirms that our approach significantly enhances model calibration, reducing expected calibration error (ECE) by up to 35%, and improves selective prediction, even when accuracy gains are modest.
>
---
#### [new 121] Multimodal Anomaly Detection for Human-Robot Interaction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机交互中的异常检测任务，旨在提升机器人对异常事件的识别能力。通过多模态特征重构，融合视觉与传感器数据，提高检测效果。**

- **链接: [https://arxiv.org/pdf/2604.09326](https://arxiv.org/pdf/2604.09326)**

> **作者:** Guilherme Ribeiro; Iordanis Antypas; Leonardo Bizzaro; João Bimbo; Nuno Cruz Garcia
>
> **摘要:** Ensuring safety and reliability in human-robot interaction (HRI) requires the timely detection of unexpected events that could lead to system failures or unsafe behaviours. Anomaly detection thus plays a critical role in enabling robots to recognize and respond to deviations from normal operation during collaborative tasks. While reconstruction models have been actively explored in HRI, approaches that operate directly on feature vectors remain largely unexplored. In this work, we propose MADRI, a framework that first transforms video streams into semantically meaningful feature vectors before performing reconstruction-based anomaly detection. Additionally, we augment these visual feature vectors with the robot's internal sensors' readings and a Scene Graph, enabling the model to capture both external anomalies in the visual environment and internal failures within the robot itself. To evaluate our approach, we collected a custom dataset consisting of a simple pick-and-place robotic task under normal and anomalous conditions. Experimental results demonstrate that reconstruction on vision-based feature vectors alone is effective for detecting anomalies, while incorporating other modalities further improves detection performance, highlighting the benefits of multimodal feature reconstruction for robust anomaly detection in human-robot collaboration.
>
---
#### [new 122] Silhouette Loss: Differentiable Global Structure Learning for Deep Representations
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于分类任务，旨在解决深度表示中几何结构不明确的问题。提出Soft Silhouette Loss，增强类内紧凑性和类间分离性，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.08573](https://arxiv.org/pdf/2604.08573)**

> **作者:** Matheus Vinícius Todescato; Joel Luís Carbonera
>
> **摘要:** Learning discriminative representations is a central goal of supervised deep learning. While cross-entropy (CE) remains the dominant objective for classification, it does not explicitly enforce desirable geometric properties in the embedding space, such as intra-class compactness and inter-class separation. Existing metric learning approaches, including supervised contrastive learning (SupCon) and proxy-based methods, address this limitation by operating on pairwise or proxy-based relationships, but often increase computational cost and complexity. In this work, we introduce Soft Silhouette Loss, a novel differentiable objective inspired by the classical silhouette coefficient from clustering analysis. Unlike pairwise objectives, our formulation evaluates each sample against all classes in the batch, providing a batch-level notion of global structure. The proposed loss directly encourages samples to be closer to their own class than to competing classes, while remaining lightweight. Soft Silhouette Loss can be seamlessly combined with cross-entropy, and is also complementary to supervised contrastive learning. We propose a hybrid objective that integrates them, jointly optimizing local pairwise consistency and global cluster structure. Extensive experiments on seven diverse datasets demonstrate that: (i) augmenting CE with Soft Silhouette Loss consistently improves over CE and other metric learning baselines; (ii) the hybrid formulation outperforms SupCon alone; and (iii) the combined method achieves the best performance, improving average top-1 accuracy from 36.71% (CE) and 37.85% (SupCon2) to 39.08%, while incurring substantially lower computational overhead. These results suggest that classical clustering principles can be reinterpreted as differentiable objectives for deep learning, enabling efficient optimization of both local and global structure in representation spaces.
>
---
#### [new 123] Pretrain-then-Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于文本检索人物搜索任务，解决数据稀缺与领域迁移问题。提出UATTA框架，通过无监督测试时适应提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.08598](https://arxiv.org/pdf/2604.08598)**

> **作者:** Jiahao Zhang; Shaofei Huang; Yaxiong Wang; Zhedong Zheng
>
> **备注:** Accepted to ACM SIGIR 2026
>
> **摘要:** Text-based person search faces inherent limitations due to data scarcity, driven by stringent privacy constraints and the high cost of manual annotation. To mitigate this, existing methods usually rely on a Pretrain-then-Finetune paradigm, where models are first pretrained on synthetic person-caption data to establish cross-modal alignment, followed by fine-tuning on labeled real-world datasets. However, this paradigm lacks practicality in real-world deployment scenarios, where large-scale annotated target-domain data is typically inaccessible. In this work, we propose a new Pretrain-then-Adapt paradigm that eliminates reliance on extensive target-domain supervision through an offline test-time adaptation manner, enabling dynamic model adaptation using only unlabeled test data with minimal post-train time cost. To mitigate overconfidence with false positives of previous entropy-based test-time adaptation, we propose an Uncertainty-Aware Test-Time Adaptation (UATTA) framework, which introduces a bidirectional retrieval disagreement mechanism to estimate uncertainty, i.e., low uncertainty is assigned when an image-text pair ranks highly in both image-to-text and text-to-image retrieval, indicating high alignment; otherwise, high uncertainty is detected. This indicator drives offline test-time model recalibration without labels, effectively mitigating domain shift. We validate UATTA on four benchmarks, i.e., CUHK-PEDES, ICFG-PEDES, RSTPReid, and PAB, showing consistent improvements across both CLIP-based (one-stage) and XVLM-based (two-stage) frameworks. Ablation studies confirm that UATTA outperforms existing offline test-time adaptation strategies, establishing a new benchmark for label-efficient, deployable person search systems. Our code is available at this https URL.
>
---
#### [new 124] MeshOn: Intersection-Free Mesh-to-Mesh Composition
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出MeshOn，解决mesh-to-mesh组合问题，通过优化框架实现无交集的物理与语义合理装配。**

- **链接: [https://arxiv.org/pdf/2604.08799](https://arxiv.org/pdf/2604.08799)**

> **作者:** Hyunwoo Kim; Itai Lang; Hadar Averbuch-Elor; Silvia Sellán; Rana Hanocka
>
> **备注:** Project page: \hyperlink{this https URL}{this https URL}
>
> **摘要:** We propose MeshOn, a method that finds physically and semantically realistic compositions of two input meshes. Given an accessory, a base mesh with a user-defined target region, and optional text strings for both meshes, MeshOn uses a multi-step optimization framework to realistically fit the meshes onto each other while preventing intersections. We initialize the shapes' rigid configuration via a structured alignment scheme using Vision-to-Language Models, which we then optimize using a combination of attractive geometric losses, and a physics-inspired barrier loss that prevents surface intersections. We then obtain a final deformation of the object, assisted by a diffusion prior. Our method successfully fits accessories of various materials over a breadth of target regions, and is designed to fit directly into existing digital artist workflows. We demonstrate the robustness and accuracy of our pipeline by comparing it with generative approaches and traditional registration algorithms.
>
---
#### [new 125] AniGen: Unified $S^3$ Fields for Animatable 3D Asset Generation
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出AniGen，解决生成可动画3D资产的问题。通过统一的S³场结构，直接从单张图像生成包含骨架和皮肤权重的动态3D模型，提升动画质量和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.08746](https://arxiv.org/pdf/2604.08746)**

> **作者:** Yi-Hua Huang; Zi-Xin Zou; Yuting He; Chirui Chang; Cheng-Feng Pu; Ziyi Yang; Yuan-Chen Guo; Yan-Pei Cao; Xiaojuan Qi
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Animatable 3D assets, defined as geometry equipped with an articulated skeleton and skinning weights, are fundamental to interactive graphics, embodied agents, and animation production. While recent 3D generative models can synthesize visually plausible shapes from images, the results are typically static. Obtaining usable rigs via post-hoc auto-rigging is brittle and often produces skeletons that are topologically inconsistent with the generated geometry. We present AniGen, a unified framework that directly generates animate-ready 3D assets conditioned on a single image. Our key insight is to represent shape, skeleton, and skinning as mutually consistent $S^3$ Fields (Shape, Skeleton, Skin) defined over a shared spatial domain. To enable the robust learning of these fields, we introduce two technical innovations: (i) a confidence-decaying skeleton field that explicitly handles the geometric ambiguity of bone prediction at Voronoi boundaries, and (ii) a dual skin feature field that decouples skinning weights from specific joint counts, allowing a fixed-architecture network to predict rigs of arbitrary complexity. Built upon a two-stage flow-matching pipeline, AniGen first synthesizes a sparse structural scaffold and then generates dense geometry and articulation in a structured latent space. Extensive experiments demonstrate that AniGen substantially outperforms state-of-the-art sequential baselines in rig validity and animation quality, generalizing effectively to in-the-wild images across diverse categories including animals, humanoids, and machinery. Homepage: this https URL
>
---
#### [new 126] Training-free, Perceptually Consistent Low-Resolution Previews with High-Resolution Image for Efficient Workflows of Diffusion Models
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决低分辨率预览与高分辨率图像的感知一致性问题。通过提出无训练方案，实现高效工作流，减少计算成本。**

- **链接: [https://arxiv.org/pdf/2604.09227](https://arxiv.org/pdf/2604.09227)**

> **作者:** Wongi Jeong; Hoigi Seo; Se Young Chun
>
> **摘要:** Image generative models have become indispensable tools to yield exquisite high-resolution (HR) images for everyone, ranging from general users to professional designers. However, a desired outcome often requires generating a large number of HR images with different prompts and seeds, resulting in high computational cost for both users and service providers. Generating low-resolution (LR) images first could alleviate computational burden, but it is not straightforward how to generate LR images that are perceptually consistent with their HR counterparts. Here, we consider the task of generating high-fidelity LR images, called Previews, that preserve perceptual similarity of their HR counterparts for an efficient workflow, allowing users to identify promising candidates before generating the final HR image. We propose the commutator-zero condition to ensure the LR-HR perceptual consistency for flow matching models, leading to the proposed training-free solution with downsampling matrix selection and commutator-zero guidance. Extensive experiments show that our method can generate LR images with up to 33\% computation reduction while maintaining HR perceptual consistency. When combined with existing acceleration techniques, our method achieves up to 3$\times$ speedup. Moreover, our formulation can be extended to image manipulations, such as warping and translation, demonstrating its generalizability.
>
---
#### [new 127] Characterizing Lidar Range-Measurement Ambiguity due to Multiple Returns
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于激光雷达感知任务，旨在解决多回波导致的距离测量模糊问题，通过分析数据集并提出累积分布函数来评估其对定位的影响。**

- **链接: [https://arxiv.org/pdf/2604.09282](https://arxiv.org/pdf/2604.09282)**

> **作者:** Jason H. Rife; Yifan Li
>
> **备注:** Proceedings of the 38th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2025), Baltimore, Maryland, September 2025, pp. 1949-1963
>
> **摘要:** Reliable position and attitude sensing is critical for highly automated vehicles that operate on conventional roadways. Lidar sensors are increasingly incorporated into pose-estimation systems. Despite its great utility, lidar is a complex sensor, and its performance in roadway environments is not yet well understood. For instance, it is often assumed in lidar-localization algorithms that a lidar will always identify a unique surface along a given raypath. However, this assumption is not always true, as ample prior evidence exists to suggest that lidar units may generate measurements probabilistically when more than one scattering surface appears within the lidar's conical beam. In this paper, we analyze lidar datasets to characterize cases with probabilistic returns along particular raypaths. Our contribution is to present representative cumulative distribution functions (CDFs) for raypaths observed by two different mechanically rotating lidar units with stationary bases. In subsequent discussion, we outline a qualitative methodology to assess the effect of probabilistic multi-return cases on lidar-based localization.
>
---
#### [new 128] Ge$^\text{2}$mS-T: Multi-Dimensional Grouping for Ultra-High Energy Efficiency in Spiking Transformer
- **分类: cs.NE; cs.AI; cs.CV**

- **简介: 该论文属于视觉Transformer优化任务，解决S-ViTs在训练和推理中的能效与性能问题。提出Ge$^\text{2}$mS-T架构，通过多维分组计算提升能效与精度。**

- **链接: [https://arxiv.org/pdf/2604.08894](https://arxiv.org/pdf/2604.08894)**

> **作者:** Zecheng Hao; Shenghao Xie; Kang Chen; Wenxuan Liu; Zhaofei Yu; Tiejun Huang
>
> **摘要:** Spiking Neural Networks (SNNs) offer superior energy efficiency over Artificial Neural Networks (ANNs). However, they encounter significant deficiencies in training and inference metrics when applied to Spiking Vision Transformers (S-ViTs). Existing paradigms including ANN-SNN Conversion and Spatial-Temporal Backpropagation (STBP) suffer from inherent limitations, precluding concurrent optimization of memory, accuracy and energy consumption. To address these issues, we propose Ge$^\text{2}$mS-T, a novel architecture implementing grouped computation across temporal, spatial and network structure dimensions. Specifically, we introduce the Grouped-Exponential-Coding-based IF (ExpG-IF) model, enabling lossless conversion with constant training overhead and precise regulation for spike patterns. Additionally, we develop Group-wise Spiking Self-Attention (GW-SSA) to reduce computational complexity via multi-scale token grouping and multiplication-free operations within a hybrid attention-convolution framework. Experiments confirm that our method can achieve superior performance with ultra-high energy efficiency on challenging benchmarks. To our best knowledge, this is the first work to systematically establish multi-dimensional grouped computation for resolving the triad of memory overhead, learning capability and energy budget in S-ViTs.
>
---
#### [new 129] Efficient Unlearning through Maximizing Relearning Convergence Delay
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在有效移除模型中不良数据的影响。通过引入重学习收敛延迟指标，提出一种新的遗忘框架，提升遗忘效果并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09391](https://arxiv.org/pdf/2604.09391)**

> **作者:** Khoa Tran; Simon S. Woo
>
> **摘要:** Machine unlearning poses challenges in removing mislabeled, contaminated, or problematic data from a pretrained model. Current unlearning approaches and evaluation metrics are solely focused on model predictions, which limits insight into the model's true underlying data characteristics. To address this issue, we introduce a new metric called relearning convergence delay, which captures both changes in weight space and prediction space, providing a more comprehensive assessment of the model's understanding of the forgotten dataset. This metric can be used to assess the risk of forgotten data being recovered from the unlearned model. Based on this, we propose the Influence Eliminating Unlearning framework, which removes the influence of the forgetting set by degrading its performance and incorporates weight decay and injecting noise into the model's weights, while maintaining accuracy on the retaining set. Extensive experiments show that our method outperforms existing metrics and our proposed relearning convergence delay metric, approaching ideal unlearning performance. We provide theoretical guarantees, including exponential convergence and upper bounds, as well as empirical evidence of strong retention and resistance to relearning in both classification and generative unlearning tasks.
>
---
#### [new 130] Multi-task Just Recognizable Difference for Video Coding for Machines: Database, Model, and Coding Application
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于视频编码任务，旨在解决机器视觉中单任务JRD的局限性。提出多任务JRD数据集和模型，提升预测精度与编码效率。**

- **链接: [https://arxiv.org/pdf/2604.09421](https://arxiv.org/pdf/2604.09421)**

> **作者:** Junqi Liu; Yun Zhang; Xiaoxia Huang; Long Xu; Weisi Lin
>
> **备注:** Submitted to IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Just Recognizable Difference (JRD) boosts coding efficiency for machine vision through visibility threshold modeling, but is currently limited to a single-task scenario. To address this issue, we propose a Multi-Task JRD (MT-JRD) dataset and an Attribute-assisted MT-JRD (AMT-JRD) model for Video Coding for Machines (VCM), enhancing both prediction accuracy and coding efficiency. First, we construct a dataset comprising 27,264 JRD annotations from machines, supporting three representative tasks including object detection, instance segmentation, and keypoint detection. Secondly, we propose the AMT-JRD prediction model, which integrates Generalized Feature Extraction Module (GFEM) and Specialized Feature Extraction Module (SFEM) to facilitate joint learning across multiple tasks. Thirdly, we innovatively incorporate object attribute information into object-wise JRD prediction through the Attribute Feature Fusion Module (AFFM), which introduces prior knowledge about object size and location. This design effectively compensates for the limitations of relying solely on image features and enhances the model's capacity to represent the perceptual mechanisms of machine vision. Finally, we apply the AMT-JRD model to VCM, where the accurately predicted JRDs are applied to reduce the coding bit rate while preserving accuracy across multiple machine vision tasks. Extensive experimental results demonstrate that AMT-JRD achieves precise and robust multi-task prediction with a mean absolute error of 3.781 and error variance of 5.332 across three tasks, outperforming the state-of-the-art single-task prediction model by 6.7% and 6.3%, respectively. Coding experiments further reveal that compared to the baseline VVC and JPEG, the AMT-JRD-based VCM improves an average of 3.861% and 7.886% Bjontegaard Delta-mean Average Precision (BD-mAP), respectively.
>
---
#### [new 131] CLIP-Inspector: Model-Level Backdoor Detection for Prompt-Tuned CLIP via OOD Trigger Inversion
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于模型安全任务，解决prompt-tuned CLIP模型中的后门检测问题。提出CLIP-Inspector方法，通过OOD触发器逆向重建实现模型级后门检测与修复。**

- **链接: [https://arxiv.org/pdf/2604.09101](https://arxiv.org/pdf/2604.09101)**

> **作者:** Akshit Jindal; Saket Anand; Chetan Arora; Vikram Goyal
>
> **备注:** 17 pages (8 main + 2 references + 7 supplementary), Accepted to CVPR Findings 2026
>
> **摘要:** Organisations with limited data and computational resources increasingly outsource model training to Machine Learning as a Service (MLaaS) providers, who adapt vision-language models (VLMs) such as CLIP to downstream tasks via prompt tuning rather than training from scratch. This semi-honest setting creates a security risk where a malicious provider can follow the prompt-tuning protocol yet implant a backdoor, forcing triggered inputs to be classified into an attacker-chosen class, even for out-of-distribution (OOD) data. Such backdoors leave encoders untouched, making them undetectable to existing methods that focus on encoder corruption. Other data-level methods that sanitize data before training or during inference, also fail to answer the critical question, "Is the delivered model backdoored or not?" To address this model-level verification problem, we introduce CLIP-Inspector (CI), a backdoor detection method designed for prompt-tuned CLIP models. Assuming white-box access to the delivered model and a pool of unlabeled OOD images, CI reconstructs possible triggers for each class to determine if the model exhibits backdoor behaviour or not. Additionally, we demonstrate that using CI's reconstructed trigger for fine-tuning on correctly labeled triggered inputs enables us to re-align the model and reduce backdoor effectiveness. Through extensive experiments across ten datasets and four backdoor attacks, we demonstrate that CI can reconstruct effective triggers in a single epoch using only 1,000 OOD images, achieving a 94% detection accuracy (47/50 models). Compared to adapted trigger-inversion baselines, CI yields a markedly higher AUROC score (0.973 vs 0.495/0.687), thus enabling the vetting and post-hoc repair of prompt-tuned CLIP models to ensure safe deployment.
>
---
#### [new 132] From Selection to Scheduling: Federated Geometry-Aware Correction Makes Exemplar Replay Work Better under Continual Dynamic Heterogeneity
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于联邦持续学习任务，旨在解决动态异质性下的灾难性遗忘问题。通过提出FEAT方法，优化样本重放的利用，提升模型在类别不平衡下的性能。**

- **链接: [https://arxiv.org/pdf/2604.08617](https://arxiv.org/pdf/2604.08617)**

> **作者:** Zhuang Qi; Ying-Peng Tang; Lei Meng; Guoqing Chao; Lei Wu; Han Yu; Xiangxu Meng
>
> **备注:** CVPR 2026 accepted
>
> **摘要:** Exemplar replay has become an effective strategy for mitigating catastrophic forgetting in federated continual learning (FCL) by retaining representative samples from past tasks. Existing studies focus on designing sample-importance estimation mechanisms to identify information-rich samples. However, they typically overlook strategies for effectively utilizing the selected exemplars, which limits their performance under continual dynamic heterogeneity across clients and tasks. To address this issue, this paper proposes a Federated gEometry-Aware correcTion method, termed FEAT, which alleviates imbalance-induced representation collapse that drags rare-class features toward frequent classes across clients. Specifically, it consists of two key modules: 1) the Geometric Structure Alignment module performs structural knowledge distillation by aligning the pairwise angular similarities between feature representations and their corresponding Equiangular Tight Frame prototypes, which are fixed and shared across clients to serve as a class-discriminative reference structure. This encourages geometric consistency across tasks and helps mitigate representation drift; 2) the Energy-based Geometric Correction module removes task-irrelevant directional components from feature embeddings, which reduces prediction bias toward majority classes. This improves sensitivity to minority classes and enhances the model's robustness under class-imbalanced distributions.
>
---
#### [new 133] UHD Low-Light Image Enhancement via Real-Time Enhancement Methods with Clifford Information Fusion
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决UHD图像实时处理效率低的问题。通过 Clifford 代数融合几何特征，提升图像质量并实现毫秒级推理。**

- **链接: [https://arxiv.org/pdf/2604.09321](https://arxiv.org/pdf/2604.09321)**

> **作者:** Xiaohan Wang; Chen Wu; Dawei Zhao; Guangwei Gao; Dianjie Lu; Guijuan Zhang; Linwei Fan; Xu Lu; Shuai Wu; Hang Wei; Zhuoran Zheng
>
> **摘要:** Considering efficiency, ultra-high-definition (UHD) low-light image restoration is extremely challenging. Existing methods based on Transformer architectures or high-dimensional complex convolutional neural networks often suffer from the "memory wall" bottleneck, failing to achieve millisecond-level inference on edge devices. To address this issue, we propose a novel real-time UHD low-light enhancement network based on geometric feature fusion using Clifford algebra in 2D Euclidean space. First, we construct a four-layer feature pyramid with gradually increasing resolution, which decomposes input images into low-frequency and high-frequency structural components via a Gaussian blur kernel, and adopts a lightweight U-Net based on depthwise separable convolution for dual-branch feature extraction. Second, to resolve structural information loss and artifacts from traditional high-low frequency feature fusion, we introduce spatially aware Clifford algebra, which maps feature tensors to a multivector space (scalars, vectors, bivectors) and uses Clifford similarity to aggregate features while suppressing noise and preserving textures. In the reconstruction stage, the network outputs adaptive Gamma and Gain maps, which perform physically constrained non-linear brightness adjustment via Retinex theory. Integrated with FP16 mixed-precision computation and dynamic operator fusion, our method achieves millisecond-level inference for 4K/8K images on a single consumer-grade device, while outperforming state-of-the-art (SOTA) models on several restoration metrics.
>
---
#### [new 134] Ranked Activation Shift for Post-Hoc Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于分布外检测任务，解决现有方法性能不稳定的问题。提出一种无需超参数的方法，通过替换激活幅度为固定参考模式，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.08572](https://arxiv.org/pdf/2604.08572)**

> **作者:** Gianluca Guglielmo; Marc Masana
>
> **备注:** Code is available at this https URL
>
> **摘要:** State-of-the-art post-hoc out-of-distribution detection methods rely on intermediate layer activation editing. However, they exhibit inconsistent performance across datasets and models. We show that this instability is driven by differences in the activation distributions, and identify a failure mode of scaling-based methods that arises when penultimate layer activations are not rectified. Motivated by this analysis, we propose \ours, a hyperparameter-free post-hoc method that replaces sorted activation magnitudes with a fixed in-distribution reference profile. Our simple plug-and-play method shows strong and consistent performance across datasets and architectures without assumptions on the penultimate layer activation function, and without requiring any hyperparameter tuning, while preserving in-distribution classification accuracy by construction. We further analyze what drives the improvement, showing that both inhibiting and exciting activation shifts independently contribute to better out-of-distribution discrimination.
>
---
#### [new 135] PSIRNet: Deep Learning-based Free-breathing Rapid Acquisition Late Enhancement Imaging
- **分类: eess.IV; cs.AI; cs.CV; eess.SP; physics.med-ph**

- **简介: 该论文提出PSIRNet，用于快速获取高质量心脏MRI图像，解决传统方法耗时长的问题，通过深度学习实现单次采集诊断级图像。**

- **链接: [https://arxiv.org/pdf/2604.08781](https://arxiv.org/pdf/2604.08781)**

> **作者:** Arda Atalik; Hui Xue; Rhodri H. Davies; Thomas A. Treibel; Daniel K. Sodickson; Michael S. Hansen; Peter Kellman
>
> **备注:** 25 pages, 5 figures, 4 tables
>
> **摘要:** Purpose: To develop and evaluate a deep learning (DL) method for free-breathing phase-sensitive inversion recovery (PSIR) late gadolinium enhancement (LGE) cardiac MRI that produces diagnostic-quality images from a single acquisition over two heartbeats, eliminating the need for 8 to 24 motion-corrected (MOCO) signal averages. Materials and Methods: Raw data comprising 800,653 slices from 55,917 patients, acquired on 1.5T and 3T scanners across multiple sites from 2016 to 2024, were used in this retrospective study. Data were split by patient: 640,000 slices (42,822 patients) for training and the remainder for validation and testing, without overlap. The training and testing data were from different institutions. PSIRNet, a physics-guided DL network with 845 million parameters, was trained end-to-end to reconstruct PSIR images with surface coil correction from a single interleaved IR/PD acquisition over two heartbeats. Reconstruction quality was evaluated using SSIM, PSNR, and NRMSE against MOCO PSIR references. Two expert cardiologists performed an independent qualitative assessment, scoring image quality on a 5-point Likert scale across bright blood, dark blood, and wideband LGE variants. Paired superiority and equivalence (margin = 0.25 Likert points) were tested using exact Wilcoxon signed-rank tests at a significance level of 0.05 using R version 4.5.2. Results: Both readers rated single-average PSIRNet reconstructions superior to MOCO PSIR for dark blood LGE (conservative P = .002); for bright blood and wideband, one reader rated it superior and the other confirmed equivalence (all P < .001). Inference required approximately 100 msec per slice versus more than 5 sec for MOCO PSIR. Conclusion: PSIRNet produces diagnostic-quality free-breathing PSIR LGE images from a single acquisition, enabling 8- to 24-fold reduction in acquisition time.
>
---
#### [new 136] Through Their Eyes: Fixation-aligned Tuning for Personalized User Emulation
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于推荐系统评估任务，旨在解决用户模拟器缺乏视觉注意力模拟的问题。通过将视觉语言模型的注意力与用户眼动模式对齐，提升模拟精度。**

- **链接: [https://arxiv.org/pdf/2604.09368](https://arxiv.org/pdf/2604.09368)**

> **作者:** Lingfeng Huang; Huizhong Guo; Tianjun Wei; Yingpeng Du; Zhu Sun
>
> **摘要:** Large language model (LLM) agents are increasingly deployed as scalable user simulators for recommender system evaluation. Yet existing simulators perceive recommendations through text or structured metadata rather than the visual interfaces real users browse-a critical gap, since attention over recommendation layouts is both visually driven and highly personalized. We investigate whether aligning a vision-language model's (VLM's) visual attention with user-specific gaze patterns can improve simulation fidelity. Analysis of a real-world eye-tracking dataset collected in a carousel-based recommendation setting reveals that users exhibit stable individual gaze patterns strongly predictive of click behavior. Building on this finding, we propose Fixation-Aligned Tuning for user Emulation (FixATE). Our approach first probes the VLM's internal visual attention via interpretability operators to obtain a slot-level relevance distribution comparable with human fixation, and then learns personalized soft prompts to steer the model's attention toward each user's characteristic fixation pattern. Experiments across three interpretability-based probing operators and two architecturally distinct VLM backbones demonstrate consistent improvements in both attention alignment and click prediction accuracy. These results suggest that making the model "see like the user" is a viable path toward simulators that more faithfully reproduce how users perceive and act in recommendation interfaces.
>
---
#### [new 137] AMO-ENE: Attention-based Multi-Omics Fusion Model for Outcome Prediction in Extra Nodal Extension and HPV-associated Oropharyngeal Cancer
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于癌症预后预测任务，旨在解决HPV相关口咽癌的转移性复发预测问题。通过整合影像和临床数据，构建多模态预测模型，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.09280](https://arxiv.org/pdf/2604.09280)**

> **作者:** Gautier Hénique; William Le; Gabriel Dayan; Coralie Brodeur; Kristoff Nelson; Apostolos Christopoulos; Edith Filion; Phuc-Felix Nguyen-Tan; Laurent Letourneau-Guillon; Houda Bahig; Samuel Kadoury
>
> **摘要:** Extranodal extension (ENE) is an emerging prognostic factor in human papillomavirus (HPV)-associated oropharyngeal cancer (OPC), although it is currently omitted as a clinical staging criteria. Recent works have advocated for the inclusion of iENE as a prognostic marker in HPV-positive OPC staging. However, several practical limitations continue to hinder its clinical integration, including inconsistencies in segmentation, low contrast in the periphery of metastatic lymph nodes on CT imaging, and laborious manual annotations. To address these limitations, we propose a fully automated end-to-end pipeline that uses computed tomography (CT) images with clinical data to assess the status of nodal ENE and predict treatment outcomes. Our approach includes a hierarchical 3D semi-supervised segmentation model designed to detect and delineate relevant iENE from radiotherapy planning CT scans. From these segmentations, a set of radiomics and deep features are extracted to train an imaging-detected ENE grading classifier. The predicted ENE status is then evaluated for its prognostic value and compared with existing staging criteria. Furthermore, we integrate these nodal features with primary tumor characteristics in a multimodal, attention-based outcome prediction model, providing a dynamic framework for outcome prediction. Our method is validated in an internal cohort of 397 HPV-positive OPC patients treated with radiation therapy or chemoradiotherapy between 2009 and 2020. For outcome prediction at the 2-year mark, our pipeline surpassed baseline models with 88.2% (4.8) in AUC for metastatic recurrence, 79.2% (7.4) for overall survival, and 78.1% (8.6) for disease-free survival. We also obtain a concordance index of 83.3% (6.5) for metastatic recurrence, 71.3% (8.9) for overall survival, and 70.0% (8.1) for disease-free survival, making it feasible for clinical decision making.
>
---
#### [new 138] Cluster-First Labelling: An Automated Pipeline for Segmentation and Morphological Clustering in Histology Whole Slide Images
- **分类: q-bio.QM; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决组织切片中结构标注耗时的问题。通过自动化流程实现分割与聚类，显著减少人工标注工作量。**

- **链接: [https://arxiv.org/pdf/2604.09370](https://arxiv.org/pdf/2604.09370)**

> **作者:** Muhammad Haseeb Ahmad; Sharmila Rajendran; Damion Young; Jon Mason
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Labelling tissue components in histology whole slide images (WSIs) is prohibitively labour-intensive: a single slide may contain tens of thousands of structures--cells, nuclei, and other morphologically distinct objects--each requiring manual boundary delineation and classification. We present a cloudnative, end-to-end pipeline that automates this process through a cluster-first paradigm. Our system tiles WSIs, filters out tiles deemed unlikely to contain valuable information, segments tissue components with Cellpose-SAM (including cells, nuclei, and other morphologically similar structures), extracts neural embeddings via a pretrained ResNet-50, reduces dimensionality with UMAP, and groups morphologically similar objects using DBSCAN clustering. Under this paradigm, a human annotator labels representative clusters rather than individual objects, reducing annotation effort by orders of magnitude. We evaluate the pipeline on 3,696 tissue components across 13 diverse tissue types from three species (human, rat, rabbit), measuring how well unsupervised clusters align with independent human labels via per-tile Hungarian-algorithm matching. Our system achieves a weighted cluster-label alignment accuracy of 96.8%, with 7 of 13 tissue types reaching perfect agreement. The pipeline, a companion labelling web application, and all evaluation code are released as open-source software.
>
---
#### [new 139] VOLTA: The Surprising Ineffectiveness of Auxiliary Losses for Calibrated Deep Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于深度学习中的不确定性量化任务，旨在提升模型校准性。通过对比实验，提出VOLTA方法，在多个数据集上表现出色。**

- **链接: [https://arxiv.org/pdf/2604.08639](https://arxiv.org/pdf/2604.08639)**

> **作者:** Rahul D Ray; Utkarsh Srivastava
>
> **摘要:** Uncertainty quantification (UQ) is essential for deploying deep learning models in safety critical applications, yet no consensus exists on which UQ method performs best across different data modalities and distribution shifts. This paper presents a comprehensive benchmark of ten widely used UQ baselines including MC Dropout, SWAG, ensemble methods, temperature scaling, energy based OOD, Mahalanobis, hyperbolic classifiers, ENN, Taylor Sensus, and split conformal prediction against a simplified yet highly effective variant of VOLTA that retains only a deep encoder, learnable prototypes, cross entropy loss, and post hoc temperature scaling. We evaluate all methods on CIFAR 10 (in distribution), CIFAR 100, SVHN, uniform noise (out of distribution), CIFAR 10 C (corruptions), and Tiny ImageNet features (tabular). VOLTA achieves competitive or superior accuracy (up to 0.864 on CIFAR 10), significantly lower expected calibration error (0.010 vs. 0.044 to 0.102 for baselines), and strong OOD detection (AUROC 0.802). Statistical testing over three random seeds shows that VOLTA matches or outperforms most baselines, with ablation studies confirming the importance of adaptive temperature and deep encoders. Our results establish VOLTA as a lightweight, deterministic, and well calibrated alternative to more complex UQ approaches.
>
---
#### [new 140] VAG: Dual-Stream Video-Action Generation for Embodied Data Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VAG框架，解决机器人合成数据生成中视频与动作对齐问题，通过双流结构联合生成视频和动作，提升跨模态一致性。**

- **链接: [https://arxiv.org/pdf/2604.09330](https://arxiv.org/pdf/2604.09330)**

> **作者:** Xiaolei Lang; Yang Wang; Yukun Zhou; Chaojun Ni; Kerui Li; Jiagang Zhu; Tianze Liu; Jiajun Lv; Xingxing Zuo; Yun Ye; Guan Huang; Xiaofeng Wang; Zheng Zhu
>
> **摘要:** Recent advances in robot foundation models trained on large-scale human teleoperation data have enabled robots to perform increasingly complex real-world tasks. However, scaling these systems remains difficult because collecting task-specific demonstrations is expensive and labor-intensive. Synthetic data, especially generated videos, offer a promising direction, but existing World Models (WMs) are not directly suitable for policy learning since they do not provide paired action trajectories. World-Action (WA) models partially address this by predicting actions with visual outputs, yet often lack strong video-action alignment, while two-stage pipelines that generate video first and then infer actions introduce inefficiency and error accumulation. To address these limitations, we propose VAG, a unified flow-matching-based dual-stream framework that jointly generates video and action under visual and language conditioning. By synchronizing denoising in both branches and using an adaptive 3D pooling mechanism to transfer compact global video context to the action branch, VAG improves cross-modal consistency during generation. Across both simulated and real-world settings, VAG produces aligned video-action pairs with competitive prediction quality, supports executable trajectory replay, and provides useful synthetic pretraining data that improves downstream policy generalization, indicating its potential as a practical world-action model for embodied data synthesis.
>
---
#### [new 141] Compositional-Degradation UAV Image Restoration: Conditional Decoupled MoE Network and A Benchmark
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于无人机图像修复任务，解决多因素退化影响下的图像质量下降问题。提出DAME-Net模型，通过解耦退化感知与重建，提升修复效果。**

- **链接: [https://arxiv.org/pdf/2604.09313](https://arxiv.org/pdf/2604.09313)**

> **作者:** Jinquan Yan; Zhicheng Zhao; Zhengzheng Tu; Chenglong Li; Jin Tang; Bin Luo
>
> **摘要:** UAV images are critical for applications such as large-area mapping, infrastructure inspection, and emergency response. However, in real-world flight environments, a single image is often affected by multiple degradation factors, including rain, haze, and noise, undermining downstream task performance. Current unified restoration approaches typically rely on implicit degradation representations that entangle multiple factors into a single condition, causing mutual interference among heterogeneous corrections. To this end, we propose DAME-Net, a Degradation-Aware Mixture-of-Experts Network that decouples explicit degradation perception from degradation-conditioned reconstruction for compositional UAV image restoration. Specifically, we design a Factor-wise Degradation Perception module(FDPM) to provide explicit per-factor degradation cues for the restoration stage through multi-label prediction with label-similarity-guided soft alignment, replacing implicit entangled conditions with interpretable and generalizable degradation descriptions. Moreover, we develop a Conditioned Decoupled MoE module(CDMM) that leverages these cues for stage-wise conditioning, spatial-frequency hybrid processing, and mask-constrained decoupled expert routing, enabling selective factor-specific correction while suppressing irrelevant interference. In addition, we construct the Multi-Degradation UAV Restoration benchmark (MDUR), the first large-scale UAV benchmark for compositional UAV image restoration, with 43 degradation configurations from single degradations to four-factor composites and standardized seen/unseen this http URL experiments on MDUR demonstrate consistent improvements over representative unified restoration methods, with greater gains on unseen and higher-order composite degradations. Downstream experiments further validate benefits for UAV object detection.
>
---
#### [new 142] Post-Hoc Guidance for Consistency Models by Joint Flow Distribution Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像生成任务，解决Consistency Models（CMs）缺乏有效后置引导的问题。提出JFDL方法，使CM在无需DM教师的情况下实现类似CFG的可调引导生成。**

- **链接: [https://arxiv.org/pdf/2604.08828](https://arxiv.org/pdf/2604.08828)**

> **作者:** Chia-Hong Hsu; Randall Balestriero
>
> **摘要:** Classifier-free Guidance (CFG) lets practitioners trade-off fidelity against diversity in Diffusion Models (DMs). The practicality of CFG is however hindered by DMs sampling cost. On the other hand, Consistency Models (CMs) generate images in one or a few steps, but existing guidance methods require knowledge distillation from a separate DM teacher, limiting CFG to Consistency Distillation (CD) methods. We propose Joint Flow Distribution Learning (JFDL), a lightweight alignment method enabling guidance in a pre-trained CM. With a pre-trained CM as an ordinary differential equation (ODE) solver, we verify with normality tests that the variance-exploding noise implied by the velocity fields from unconditional and conditional distributions is Gaussian. In practice, JFDL equips CMs with the familiar adjustable guidance knob, yielding guided images with similar characteristics to CFG. Applied to an original Consistency Trained (CT) CM that could only do conditional sampling, JFDL unlocks guided generation and reduces FID on both CIFAR-10 and ImageNet 64x64 datasets. This is the first time that CMs are able to receive effective guidance post-hoc without a DM teacher, thus, bridging a key gap in current methods for CMs.
>
---
#### [new 143] Towards Lifelong Aerial Autonomy: Geometric Memory Management for Continual Visual Place Recognition in Dynamic Environments
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉定位任务，解决动态环境中持续学习的灾难性遗忘问题。提出一种异构记忆框架，结合静态地理锚点与动态经验回放，提升长期空中自主能力。**

- **链接: [https://arxiv.org/pdf/2604.09038](https://arxiv.org/pdf/2604.09038)**

> **作者:** Xingyu Shao; Zhiqiang Yan; Liangzheng Sun; Mengfan He; Chao Chen; Jinhui Zhang; Chunyu Li; Ziyang Meng
>
> **摘要:** Robust geo-localization in changing environmental conditions is critical for long-term aerial autonomy. While visual place recognition (VPR) models perform well when airborne views match the training domain, adapting them to shifting distributions during sequential missions triggers catastrophic forgetting. Existing continual learning (CL) methods often fail here because geographic features exhibit severe intra-class variations. In this work, we formulate aerial VPR as a mission-based domain-incremental learning (DIL) problem and propose a novel heterogeneous memory framework. To respect strict onboard storage constraints, our "Learn-and-Dispose" pipeline decouples geographic knowledge into static satellite anchors (preserving global geometric priors) and a dynamic experience replay buffer (retaining domain-specific features). We introduce a spatially-constrained allocation strategy that optimizes buffer selection based on sample difficulty or feature space diversity. To facilitate systematic assessment, we provide three evaluation criteria and a comprehensive benchmark derived from 21 diverse mission sequences. Extensive experiments demonstrate that our architecture significantly boosts spatial generalization; our diversity-driven buffer selection outperforms the random baseline by 7.8% in knowledge retention. Unlike class-mean preservation methods that fail in unstructured environments, maximizing structural diversity achieves a superior plasticity-stability balance and ensures order-agnostic robustness across randomized sequences. These results prove that maintaining structural feature coverage is more critical than sample difficulty for resolving catastrophic forgetting in lifelong aerial autonomy.
>
---
#### [new 144] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型安全任务，旨在解决模型对恶意查询的脆弱性问题。通过构建概念字典和稀疏自编码器，实现对模型激活的精准控制，提升安全性。**

- **链接: [https://arxiv.org/pdf/2604.08846](https://arxiv.org/pdf/2604.08846)**

> **作者:** Jinqi Luo; Jinyu Yang; Tal Neiman; Lei Fan; Bing Yin; Son Tran; Mubarak Shah; René Vidal
>
> **备注:** Accepted in CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have been shown to be vulnerable to malicious queries that can elicit unsafe responses. Recent work uses prompt engineering, response classification, or finetuning to improve MLLM safety. Nevertheless, such approaches are often ineffective against evolving malicious patterns, may require rerunning the query, or demand heavy computational resources. Steering the activations of a frozen model at inference time has recently emerged as a flexible and effective solution. However, existing steering methods for MLLMs typically handle only a narrow set of safety-related concepts or struggle to adjust specific concepts without affecting others. To address these challenges, we introduce Dictionary-Aligned Concept Control (DACO), a framework that utilizes a curated concept dictionary and a Sparse Autoencoder (SAE) to provide granular control over MLLM activations. First, we curate a dictionary of 15,000 multimodal concepts by retrieving over 400,000 caption-image stimuli and summarizing their activations into concept directions. We name the dataset DACO-400K. Second, we show that the curated dictionary can be used to intervene activations via sparse coding. Third, we propose a new steering approach that uses our dictionary to initialize the training of an SAE and automatically annotate the semantics of the SAE atoms for safeguarding MLLMs. Experiments on multiple MLLMs (e.g., QwenVL, LLaVA, InternVL) across safety benchmarks (e.g., MM-SafetyBench, JailBreakV) show that DACO significantly improves MLLM safety while maintaining general-purpose capabilities.
>
---
#### [new 145] DSVTLA: Deep Swin Vision Transformer-Based Transfer Learning Architecture for Multi-Type Cancer Histopathological Cancer Image Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种基于Swin-Vision Transformer的迁移学习架构，用于多类型癌症病理图像分类，解决跨类型癌症图像准确分类问题，通过融合Transformer与ResNet50提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09468](https://arxiv.org/pdf/2604.09468)**

> **作者:** Muazzem Hussain Khan; Tasdid Hasnain; Md. Jamil khan; Ruhul Amin; Md. Shamim Reza; Md. Al Mehedi Hasan; Md Ashad Alam
>
> **备注:** 25 [ages. 9 Figures
>
> **摘要:** In this study, we proposed a deep Swin-Vision Transformer-based transfer learning architecture for robust multi-cancer histopathological image classification. The proposed framework integrates a hierarchical Swin Transformer with ResNet50-based convolution features extraction, enabling the model to capture both long-range contextual dependencies and fine-grained local morphological patterns within histopathological images. To validate the efficiency of the proposed architecture, an extensive experiment was executed on a comprehensive multi-cancer dataset including Breast Cancer, Oral Cancer, Lung and Colon Cancer, Kidney Cancer, and Acute Lymphocytic Leukemia (ALL), including both original and segmented images were analyzed to assess model robustness across heterogeneous clinical imaging conditions. Our approach is benchmarked alongside several state-of-the-art CNN and transfer models, including DenseNet121, DenseNet201, InceptionV3, ResNet50, EfficientNetB3, multiple ViT variants, and Swin Transformer models. However, all models were trained and validated using a unified pipeline, incorporating balanced data preprocessing, transfer learning, and fine-tuning strategies. The experimental results demonstrated that our proposed architecture consistently gained superior performance, reaching 100% test accuracy for lung-colon cancer, segmented leukemia datasets, and up to 99.23% accuracy for breast cancer classification. The model also achieved near-perfect precision, f1 score, and recall, indicating highly stable scores across divers cancer types. Overall, the proposed model establishes a highly accurate, interpretable, and also robust multi-cancer classification system, demonstrating strong benchmark for future research and provides a unified comparative assessment useful for designing reliable AI-assisted histopathological diagnosis and clinical decision-making.
>
---
#### [new 146] 2D or 3D: Who Governs Salience in VLA Models? -- Tri-Stage Token Pruning Framework with Modality Salience Awareness
- **分类: cs.MM; cs.CV; cs.RO**

- **简介: 该论文针对多模态视觉-语言-动作模型中的token剪枝问题，提出一种考虑2D/3D模态显著性的三阶段剪枝框架，以提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2604.09244](https://arxiv.org/pdf/2604.09244)**

> **作者:** Zihao Zheng; Sicheng Tian; Zhihao Mao; Lingyue Zhang; Chenyue Li; Ziyun Zhang; Hong Gao; Yuchen Huang; Yutong Xu; Guojie Luo; Xiang Chen
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as the mainstream of embodied intelligence. Recent VLA models have expanded their input modalities from 2D-only to 2D+3D paradigms, forming multi-visual-modal VLA (MVLA) models. Despite achieving improved spatial perception, MVLA faces a greater acceleration demand due to the increased number of input tokens caused by modal expansion. Token pruning is an effective optimization methods tailored to MVLA models. However, existing token pruning schemes are designed for 2D-only VLA models, ignoring 2D/3D modality salience differences. In this paper, we follow the application process of multi-modal data in MVLA models and develop a tri-stage analysis to capture the discrepancy and dynamics of 2D/3D modality salience. Based on these, we propose a corresponding tri-stage token pruning framework for MVLA models to achieve optimal 2D/3D token selection and efficient pruning. Experiments show that our framework achieves up to a 2.55x inference speedup with minimal accuracy loss, while only costing 5.8% overhead. Our Code is coming soon.
>
---
## 更新

#### [replaced 001] EmoCtrl: Controllable Emotional Image Content Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.22437](https://arxiv.org/pdf/2512.22437)**

> **作者:** Jingyuan Yang; Weibin Luo; Hui Huang
>
> **摘要:** An image conveys meaning through both its visual content and emotional tone, jointly shaping human perception. We introduce Controllable Emotional Image Content Generation (C-EICG), which aims to generate images that remain faithful to a given content description while expressing a target emotion. Existing text-to-image models ensure content consistency but lack emotional awareness, whereas emotion-driven models generate affective results at the cost of content distortion. To address this gap, we propose EmoCtrl, supported by a dataset annotated with content, emotion, and affective prompts, bridging abstract emotions to visual cues. EmoCtrl incorporates textual and visual emotion enhancement modules that enrich affective expression via descriptive semantics and perceptual cues. To align with human preference, we further introduce an emotion-driven preference optimization with specifically designed emotion reward. Comprehensive experiments demonstrate that EmoCtrl achieves faithful content and expressive emotion control, outperforming existing methods. User studies confirm EmoCtrl's strong alignment with human preference. Moreover, EmoCtrl generalizes well to creative applications, further demonstrating the robustness and adaptability of the learned emotion tokens.
>
---
#### [replaced 002] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.02231](https://arxiv.org/pdf/2512.02231)**

> **作者:** Le Thien Phuc Nguyen; Zhuoran Yu; Samuel Low Yu Hang; Subin An; Jeongik Lee; Yohan Ban; SeungEun Chung; Thanh-Huy Nguyen; JuWan Maeng; Soochahn Lee; Yong Jae Lee
>
> **备注:** Findings of CVPR 2026
>
> **摘要:** Multimodal large language models (MLLMs) are expected to jointly interpret vision, audio, and language, yet existing video benchmarks rarely assess fine-grained reasoning about human speech. Many tasks remain visually solvable or only coarsely evaluate speech, offering limited insight into whether models can align who speaks, what is said, and when it occurs. We introduce AV-SpeakerBench, a curated benchmark of 3,212 multiple-choice questions focused on speaker-centric audiovisual reasoning in real-world videos. It features: (1) a speaker-centered formulation that treats speakers-not scenes-as the core reasoning unit; (2) fusion-grounded question design embedding audiovisual dependencies into question semantics; and (3) expert-curated annotations ensuring temporal precision and cross-modal validity. Comprehensive evaluations show that the Gemini family consistently outperforms open-source systems, with Gemini 2.5 Pro achieving the best results. Among open models, Qwen3-Omni-30B approaches Gemini 2.0 Flash but remains far behind Gemini 2.5 Pro, primarily due to weaker audiovisual fusion rather than visual perception. We believe AV-SpeakerBench establishes a rigorous foundation for advancing fine-grained audiovisual reasoning in future multimodal systems.
>
---
#### [replaced 003] LADR: Locality-Aware Dynamic Rescue for Efficient Text-to-Image Generation with Diffusion Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型推理速度慢的问题。提出LADR方法，通过利用图像空间特性加速生成，提升效率同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2603.13450](https://arxiv.org/pdf/2603.13450)**

> **作者:** Chenglin Wang; Yucheng Zhou; Shawn Chen; Tao Wang; Kai Zhang
>
> **备注:** ACL2026 Main Conference
>
> **摘要:** Discrete Diffusion Language Models have emerged as a compelling paradigm for unified multimodal generation, yet their deployment is hindered by high inference latency arising from iterative decoding. Existing acceleration strategies often require expensive re-training or fail to leverage the 2D spatial redundancy inherent in visual data. To address this, we propose Locality-Aware Dynamic Rescue (LADR), a training-free method that expedites inference by exploiting the spatial Markov property of images. LADR prioritizes the recovery of tokens at the ''generation frontier'', regions spatially adjacent to observed pixels, thereby maximizing information gain. Specifically, our method integrates morphological neighbor identification to locate candidate tokens, employs a risk-bounded filtering mechanism to prevent error propagation, and utilizes manifold-consistent inverse scheduling to align the diffusion trajectory with the accelerated mask density. Extensive experiments on four text-to-image generation benchmarks demonstrate that our LADR achieves an approximate 4 x speedup over standard baselines. Remarkably, it maintains or even enhances generative fidelity, particularly in spatial reasoning tasks, offering a state-of-the-art trade-off between efficiency and quality.
>
---
#### [replaced 004] RetinexDualV2: Physically-Grounded Dual Retinex for Generalized UHD Image Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27979](https://arxiv.org/pdf/2603.27979)**

> **作者:** Mohab Kishawy; Jun Chen
>
> **摘要:** We propose RetinexDualV2, a unified, physically grounded dual-branch framework for diverse Ultra-High-Definition (UHD) image restoration. Unlike generic models, our method employs a Task-Specific Physical Grounding Module (TS-PGM) to extract degradation-aware priors (e.g., rain masks and dark channels). These explicitly guide a Retinex decomposition network via a novel Physical-Conditioned Multi-head Self-Attention (PC-MSA) mechanism, enabling robust reflection and illumination correction. This physical conditioning allows a single architecture to handle various complex degradations seamlessly, without task-specific structural modifications. RetinexDualV2 demonstrates exceptional generalizability, securing 4th place in the NTIRE 2026 Day and Night Raindrop Removal Challenge and 5th place in the Joint Noise Low-light Enhancement (JNLLIE) Challenge. Extensive experiments confirm the state-of-the-art performance and efficiency of our physically motivated approach. Code is available at this https URL
>
---
#### [replaced 005] PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20068](https://arxiv.org/pdf/2511.20068)**

> **作者:** Simon Damm; Jonas Ricker; Henning Petzka; Asja Fischer
>
> **备注:** 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition - Findings Track (CVPRF 2026)
>
> **摘要:** Autoregressive (AR) image generation has recently emerged as a powerful paradigm for image synthesis. Leveraging the generation principle of large language models, they allow for efficiently generating deceptively real-looking images, further increasing the need for reliable detection methods. However, to date there is a lack of work specifically targeting the detection of images generated by AR image generators. In this work, we present PRADA (Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images), a simple and interpretable approach that can reliably detect AR-generated images and attribute them to their respective source model. The key idea is to inspect the ratio of a model's conditional and unconditional probability for the autoregressive token sequence representing a given image. Whenever an image is generated by a particular model, its probability ratio shows unique characteristics which are not present for images generated by other models or real images. We exploit these characteristics for threshold-based attribution and detection by calibrating a simple, model-specific score function. Our experimental evaluation shows that PRADA is highly effective against eight class-to-image and four text-to-image models. We release our code and data at this http URL.
>
---
#### [replaced 006] Beyond Flicker: Detecting Kinematic Inconsistencies for Generalizable Deepfake Video Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04175](https://arxiv.org/pdf/2512.04175)**

> **作者:** Alejandro Cobo; Roberto Valle; José Miguel Buenaposada; Luis Baumela
>
> **摘要:** Generalizing deepfake detection to unseen manipulations remains a key challenge. A recent approach to tackle this issue is to train a network with pristine face images that have been manipulated with hand-crafted artifacts to extract more generalizable clues. While effective for static images, extending this to the video domain is an open issue. Existing methods model temporal artifacts as frame-to-frame instabilities, overlooking a key vulnerability: the violation of natural motion dependencies between different facial regions. In this paper, we propose a synthetic video generation method that creates training data with subtle kinematic inconsistencies. We train an autoencoder to decompose facial landmark configurations into motion bases. By manipulating these bases, we selectively break the natural correlations in facial movements and introduce these artifacts into pristine videos via face morphing. A network trained on our data learns to spot these sophisticated biomechanical flaws, achieving state-of-the-art generalization results on several popular benchmarks.
>
---
#### [replaced 007] Dejavu: Towards Experience Feedback Learning for Embodied Intelligence
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出Dejavu框架，解决部署后智能体无法持续学习的问题。通过经验反馈网络，增强冻结策略，提升任务适应性和成功率。属于机器人学习任务。**

- **链接: [https://arxiv.org/pdf/2510.10181](https://arxiv.org/pdf/2510.10181)**

> **作者:** Shaokai Wu; Yanbiao Ji; Qiuchang Li; Zhiyi Zhang; Qichen He; Wenyuan Xie; Guodong Zhang; Bayram Bayramli; Yue Ding; Hongtao Lu
>
> **摘要:** Embodied agents face a fundamental limitation: once deployed in real-world environments, they cannot easily acquire new knowledge to improve task performance. In this paper, we propose Dejavu, a general post-deployment learning framework that augments a frozen Vision-Language-Action (VLA) policy with retrieved execution memories through an Experience Feedback Network (EFN). EFN identifies contextually relevant prior action experiences and conditions action prediction on the retrieved guidance. We train EFN with reinforcement learning and semantic similarity rewards, encouraging the predicted actions to align with past behaviors under the current observation. During deployment, EFN continually expands its memory with new trajectories, enabling the agent to exhibit ``learning from experience.'' Experiments across diverse embodied tasks show that EFN improves adaptability, robustness, and success rates over frozen baselines. Our Project Page is this https URL.
>
---
#### [replaced 008] HD-VGGT: High-Resolution Visual Geometry Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27222](https://arxiv.org/pdf/2603.27222)**

> **作者:** Tianrun Chen; Yuanqi Hu; Yidong Han; Hanjie Xu; Deyi Ji; Qi Zhu; Chunan Yu; Xin Zhang; Cheng Chen; Chaotao Ding; Ying Zang; Xuanfu Li; Jin Ma; Lanyun Zhu
>
> **摘要:** High-resolution imagery is essential for accurate 3D reconstruction, as many geometric details only emerge at fine spatial scales. Recent feed-forward approaches, such as the Visual Geometry Grounded Transformer (VGGT), have demonstrated the ability to infer scene geometry from large collections of images in a single forward pass. However, scaling these models to high-resolution inputs remains challenging: the number of tokens in transformer architectures grows rapidly with both image resolution and the number of views, leading to prohibitive computational and memory costs. Moreover, we observe that visually ambiguous regions, such as repetitive patterns, weak textures, or specular surfaces, often produce unstable feature tokens that degrade geometric inference, especially at higher resolutions. We introduce HD-VGGT, a dual-branch architecture for efficient and robust high-resolution 3D reconstruction. A low-resolution branch predicts a coarse, globally consistent geometry, while a high-resolution branch refines details via a learned feature upsampling module. To handle unstable tokens, we propose Feature Modulation, which suppresses unreliable features early in the transformer. HD-VGGT leverages high-resolution images and supervision without full-resolution transformer costs, achieving state-of-the-art reconstruction quality.
>
---
#### [replaced 009] Streaming Video Instruction Tuning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21334](https://arxiv.org/pdf/2512.21334)**

> **作者:** Jiaer Xia; Peixian Chen; Mengdan Zhang; Xing Sun; Kaiyang Zhou
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** We present Streamo, a real-time streaming video LLM that serves as a general-purpose interactive assistant. Unlike existing online video models that focus narrowly on question answering or captioning, Streamo performs a broad spectrum of streaming video tasks, including real-time narration, action understanding, event captioning, temporal event grounding, and time-sensitive question answering. To develop such versatility, we construct Streamo-Instruct-465K, a large-scale instruction-following dataset tailored for streaming video understanding. The dataset covers diverse temporal contexts and multi-task supervision, enabling unified training across heterogeneous streaming tasks. After training end-to-end on the instruction-following dataset through a streamlined pipeline, Streamo exhibits strong temporal reasoning, responsive interaction, and broad generalization across a variety of streaming benchmarks. Extensive experiments show that Streamo bridges the gap between offline video perception models and real-time multimodal assistants, making a step toward unified, intelligent video understanding in continuous video streams.
>
---
#### [replaced 010] Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.07928](https://arxiv.org/pdf/2604.07928)**

> **作者:** Tao Han; Zhibin Wen; Zhenghao Chen; Fenghua Lin; Junyu Gao; Song Guo; Lei Bai
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: this https URL.
>
---
#### [replaced 011] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于音乐音频-视觉问答任务，旨在解决音乐领域中多模态理解的特殊挑战。通过分析数据集和方法，提出针对性的模型设计与策略。**

- **链接: [https://arxiv.org/pdf/2505.20638](https://arxiv.org/pdf/2505.20638)**

> **作者:** Wenhao You; Xingjian Diao; Wenjun Huang; Chunhui Zhang; Keyi Kong; Weiyi Wu; Chiyu Ma; Zhongyu Ouyang; Tingxuan Wu; Ming Cheng; Soroush Vosoughi; Jiang Gui
>
> **备注:** Accepted to Annual Meeting of the Association for Computational Linguistics (ACL 2026). The first two authors contributed equally
>
> **摘要:** While recent Multimodal Large Language Models exhibit impressive capabilities for general multimodal tasks, specialized domains like music necessitate tailored approaches. Music Audio-Visual Question Answering (Music AVQA) particularly underscores this, presenting unique challenges with its continuous, densely layered audio-visual content, intricate temporal dynamics, and the critical need for domain-specific knowledge. Through a systematic analysis of Music AVQA datasets and methods, this paper identifies that specialized input processing, architectures incorporating dedicated spatial-temporal designs, and music-specific modeling strategies are critical for success in this domain. Our study provides valuable insights for researchers by highlighting effective design patterns empirically linked to strong performance, proposing concrete future directions for incorporating musical priors, and aiming to establish a robust foundation for advancing multimodal musical understanding. We aim to encourage further research in this area and provide a GitHub repository of relevant works: this https URL.
>
---
#### [replaced 012] FlashLips: 100-FPS Mask-Free Latent Lip-Sync using Reconstruction Instead of Diffusion or GANs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20033](https://arxiv.org/pdf/2512.20033)**

> **作者:** Andreas Zinonos; Michał Stypułkowski; Antoni Bigata; Stavros Petridis; Maja Pantic; Nikita Drobyshev
>
> **摘要:** We present FlashLips, a two-stage, mask-free lip-sync system that decouples lips control from rendering and achieves real-time performance, with our U-Net variant running at over 100 FPS on a single GPU, while matching the visual quality of larger state-of-the-art models. Stage 1 is a compact, one-step latent-space editor that reconstructs an image using a reference identity, a masked target frame, and a low-dimensional lips-pose vector, trained purely with reconstruction losses - no GANs or diffusion. To remove explicit masks at inference, we use self-supervision via mouth-altered target variants as pseudo ground truth, teaching the network to localize lip edits while preserving the rest. Stage 2 is an audio-to-pose transformer trained with a flow-matching objective to predict lips-pose vectors from speech. Together, these stages form a simple and stable pipeline that combines deterministic reconstruction with robust audio control, delivering high perceptual quality and faster-than-real-time speed.
>
---
#### [replaced 013] Plug-and-Play Logit Fusion for Heterogeneous Pathology Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07779](https://arxiv.org/pdf/2604.07779)**

> **作者:** Gexin Huang; Anqi Li; Yusheng Tan; Beidi Zhao; Gang Wang; Zu-Hua Gao; Xiaoxiao Li
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Pathology foundation models (FMs) have become central to computational histopathology, offering strong transfer performance across a wide range of diagnostic and prognostic tasks. The rapid proliferation of pathology foundation models creates a model-selection bottleneck: no single model is uniformly best, yet exhaustively adapting and validating many candidates for each downstream endpoint is prohibitively expensive. We address this challenge with a lightweight and novel model fusion strategy, LogitProd, which treats independently trained FM-based predictors as fixed experts and learns sample-adaptive fusion weights over their slide-level outputs. The fusion operates purely on logits, requiring no encoder retraining and no feature-space alignment across heterogeneous backbones. We further provide a theoretical analysis showing that the optimal weighted product fusion is guaranteed to perform at least as well as the best individual expert under the training objective. We systematically evaluate LogitProd on \textbf{22} benchmarks spanning WSI-level classification, tile-level classification, gene mutation prediction, and discrete-time survival modeling. LogitProd ranks first on 20/22 tasks and improves the average performance across all tasks by ~3% over the strongest single expert. LogitProd enables practitioners to upgrade heterogeneous FM-based pipelines in a plug-and-play manner, achieving multi-expert gains with $\sim$12$\times$ lower training cost than feature-fusion alternatives.
>
---
#### [replaced 014] All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.26641](https://arxiv.org/pdf/2510.26641)**

> **作者:** Sayed Pedram Haeri Boroujeni; Niloufar Mehrabi; Hazim Alzorgan; Mahlagha Fazeli; Abolfazl Razi
>
> **摘要:** Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.
>
---
#### [replaced 015] RAM: Recover Any 3D Human Motion in-the-Wild
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.19929](https://arxiv.org/pdf/2603.19929)**

> **作者:** Sen Jia; Ning Zhu; Jinqin Zhong; Jiale Zhou; Huaping Zhang; Jenq-Neng Hwang; Lei Li
>
> **备注:** Accepted by CVPR2026!
>
> **摘要:** RAM incorporates a motion-aware semantic tracker with adaptive Kalman filtering to achieve robust identity association under severe occlusions and dynamic interactions. A memory-augmented Temporal HMR module further enhances human motion reconstruction by injecting spatio-temporal priors for consistent and smooth motion estimation. Moreover, a lightweight Predictor module forecasts future poses to maintain reconstruction continuity, while a gated combiner adaptively fuses reconstructed and predicted features to ensure coherence and robustness. Experiments on in-the-wild multi-person benchmarks such as PoseTrack and 3DPW, demonstrate that RAM substantially outperforms previous state-of-the-art in both Zero-shot tracking stability and 3D accuracy, offering a generalizable paradigm for markerless 3D human motion capture in-the-wild.
>
---
#### [replaced 016] Relational Visual Similarity
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.07833](https://arxiv.org/pdf/2512.07833)**

> **作者:** Thao Nguyen; Sicheng Mo; Krishna Kumar Singh; Yilin Wang; Jing Shi; Nicholas Kolkin; Eli Shechtman; Yong Jae Lee; Yuheng Li
>
> **备注:** CVPR 2026 camera-ready; Project page, data, and code: this https URL
>
> **摘要:** Humans do not just see attribute similarity -- we also see relational similarity. An apple is like a peach because both are reddish fruit, but the Earth is also like a peach: its crust, mantle, and core correspond to the peach's skin, flesh, and pit. This ability to perceive and recognize relational similarity, is arguable by cognitive scientist to be what distinguishes humans from other species. Yet, all widely used visual similarity metrics today (e.g., LPIPS, CLIP, DINO) focus solely on perceptual attribute similarity and fail to capture the rich, often surprising relational similarities that humans perceive. How can we go beyond the visible content of an image to capture its relational properties? How can we bring images with the same relational logic closer together in representation space? To answer these questions, we first formulate relational image similarity as a measurable problem: two images are relationally similar when their internal relations or functions among visual elements correspond, even if their visual attributes differ. We then curate 114k image-caption dataset in which the captions are anonymized -- describing the underlying relational logic of the scene rather than its surface content. Using this dataset, we finetune a Vision-Language model to measure the relational similarity between images. This model serves as the first step toward connecting images by their underlying relational structure rather than their visible appearance. Our study shows that while relational similarity has a lot of real-world applications, existing image similarity models fail to capture it -- revealing a critical gap in visual computing.
>
---
#### [replaced 017] P3P Made Easy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01312](https://arxiv.org/pdf/2508.01312)**

> **作者:** Seong Hun Lee; Patrick Vandewalle; Javier Civera
>
> **摘要:** We revisit the classical Perspective-Three-Point (P3P) problem, which aims to recover the absolute pose of a calibrated camera from three 2D-3D correspondences. It has long been known that P3P can be reduced to a quartic polynomial with analytically simple and computationally efficient coefficients. However, this elegant formulation has been largely overlooked in modern literature. Building on the theoretical foundation that traces back to Grunert's work in 1841, we propose a compact algebraic solver that achieves accuracy and runtime comparable to state-of-the-art methods. Our results show that this classical formulation remains highly competitive when implemented with modern insights, offering an excellent balance between simplicity, efficiency, and accuracy.
>
---
#### [replaced 018] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05091](https://arxiv.org/pdf/2508.05091)**

> **作者:** Jingxuan He; Busheng Su; Finn Wong
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Generating temporally coherent, long-duration videos with precise control over subject identity and movement remains a fundamental challenge for contemporary diffusion-based models, which often suffer from identity drift and are limited to short video length. We present PoseGen, a novel framework that generates human videos of extended duration from a single reference image and a driving video. Our contributions include an in-context LoRA finetuning design that injects subject appearance at the token level for identity preservation, while simultaneously conditioning on pose information at the channel level for fine-grained motion control. To overcome duration limits, we introduce a segment-interleaved generation strategy, where non-overlapping segments are first generated with improved background consistency through a shared KV-cache mechanism, and then stitched into a continuous sequence via pose-aware interpolated generation. Despite being trained on a remarkably small 33-hour video dataset, PoseGen demonstrates superior performance over state-of-the-art baselines in identity fidelity, pose accuracy, and temporal consistency. Code is available at this https URL .
>
---
#### [replaced 019] SAT: Selective Aggregation Transformer for Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07994](https://arxiv.org/pdf/2604.07994)**

> **作者:** Dinh Phu Tran; Thao Do; Saad Wazir; Seongah Kim; Seon Kwon Kim; Daeyoung Kim
>
> **备注:** Accepted to CVPR2026 (Findings Track)
>
> **摘要:** Transformer-based approaches have revolutionized image super-resolution by modeling long-range dependencies. However, the quadratic computational complexity of vanilla self-attention mechanisms poses significant challenges, often leading to compromises between efficiency and global context exploitation. Recent window-based attention methods mitigate this by localizing computations, but they often yield restricted receptive fields. To mitigate these limitations, we propose Selective Aggregation Transformer (SAT). This novel transformer efficiently captures long-range dependencies, leading to an enlarged model receptive field by selectively aggregating key-value matrices (reducing the number of tokens by 97\%) via our Density-driven Token Aggregation algorithm while maintaining the full resolution of the query matrix. This design significantly reduces computational costs, resulting in lower complexity and enabling scalable global interactions without compromising reconstruction fidelity. SAT identifies and represents each cluster with a single aggregation token, utilizing density and isolation metrics to ensure that critical high-frequency details are preserved. Experimental results demonstrate that SAT outperforms the state-of-the-art method PFT by up to 0.22dB, while the total number of FLOPs can be reduced by up to 27\%.
>
---
#### [replaced 020] Shortcut Learning in Glomerular AI: Adversarial Penalties Hurt, Entropy Helps
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07936](https://arxiv.org/pdf/2604.07936)**

> **作者:** Mohammad Daouk; Jan Ulrich Becker; Neeraja Kambham; Anthony Chang; Hien Van Nguyen; Chandra Mohan
>
> **备注:** Accepted at IEEE ISBI 2026. Hien Nguyen and Chandra Mohan jointly supervised this work
>
> **摘要:** Stain variability is a pervasive source of distribution shift and potential shortcut learning in renal pathology AI. We ask whether lupus nephritis glomerular lesion classifiers exploit stain as a shortcut, and how to mitigate such bias without stain or site labels. We curate a multi-center, multi-stain dataset of 9,674 glomerular patches (224$\times$224) from 365 WSIs across three centers and four stains (PAS, H&E, Jones, Trichrome), labeled as proliferative vs. non-proliferative. We evaluate Bayesian CNN and ViT backbones with Monte Carlo dropout in three settings: (1) stain-only classification; (2) a dual-head model jointly predicting lesion and stain with supervised stain loss; and (3) a dual-head model with label-free stain regularization via entropy maximization on the stain head. In (1), stain identity is trivially learnable, confirming a strong candidate shortcut. In (2), varying the strength and sign of stain supervision strongly modulates stain performance but leaves lesion metrics essentially unchanged, indicating no measurable stain-driven shortcut learning on this multi-stain, multi-center dataset, while overly adversarial stain penalties inflate predictive uncertainty. In (3), entropy-based regularization holds stain predictions near chance without degrading lesion accuracy or calibration. Overall, a carefully curated multi-stain dataset can be inherently robust to stain shortcuts, and a Bayesian dual-head architecture with label-free entropy regularization offers a simple, deployment-friendly safeguard against potential stain-related drift in glomerular AI.
>
---
#### [replaced 021] ETCH-X: Robustify Expressive Body Fitting to Clothed Humans with Composable Datasets
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08548](https://arxiv.org/pdf/2604.08548)**

> **作者:** Xiaoben Li; Jingyi Wu; Zeyu Cai; Siyuan Yu; Boqian Li; Yuliang Xiu
>
> **备注:** Page: this https URL, Code: this https URL
>
> **摘要:** Human body fitting, which aligns parametric body models such as SMPL to raw 3D point clouds of clothed humans, serves as a crucial first step for downstream tasks like animation and texturing. An effective fitting method should be both locally expressive-capturing fine details such as hands and facial features-and globally robust to handle real-world challenges, including clothing dynamics, pose variations, and noisy or partial inputs. Existing approaches typically excel in only one aspect, lacking an all-in-one solution. We upgrade ETCH to ETCH-X, which leverages a tightness-aware fitting paradigm to filter out clothing dynamics ("undress"), extends expressiveness with SMPL-X, and replaces explicit sparse markers (which are highly sensitive to partial data) with implicit dense correspondences ("dense fit") for more robust and fine-grained body fitting. Our disentangled "undress" and "dense fit" modular stages enable separate and scalable training on composable data sources, including diverse simulated garments (CLOTH3D), large-scale full-body motions (AMASS), and fine-grained hand gestures (InterHand2.6M), improving outfit generalization and pose robustness of both bodies and hands. Our approach achieves robust and expressive fitting across diverse clothing, poses, and levels of input completeness, delivering a substantial performance improvement over ETCH on both: 1) seen data, such as 4D-Dress (MPJPE-All, 33.0% ) and CAPE (V2V-Hands, 35.8% ), and 2) unseen data, such as BEDLAM2.0 (MPJPE-All, 80.8% ; V2V-All, 80.5% ). Code and models will be released at this https URL.
>
---
#### [replaced 022] Out-of-the-box: Black-box Causal Attacks on Object Detectors
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.03730](https://arxiv.org/pdf/2512.03730)**

> **作者:** Melane Navaratnarajah; David A. Kelly; Hana Chockler
>
> **备注:** 14 pages, 12 pages of appendices
>
> **摘要:** Adversarial perturbations are a useful way to expose vulnerabilities in object detectors. Existing perturbation methods are frequently white-box, architecture specific and use a loss function. More importantly, while they are often successful, it is rarely clear why they work. Insights into the mechanism of this success would allow developers to understand and analyze these attacks, as well as fine-tune the model to prevent them. This paper presents BlackCAtt, a black-box algorithm and tool, which uses minimal, causally sufficient pixel sets to construct explainable, imperceptible, reproducible, architecture-agnostic attacks on object detectors. We evaluate BlackCAtt on standard benchmarks and compare it to other black-box adversarial attacks methods. When BlackCAtt has access only to the position and label of a bounding box, it produces attacks that are comparable or better to those produced by other black-box methods. When BlackCAtt has access to the model confidence as well, it can work as a meta-algorithm, improving the ability of standard black-box techniques to construct smaller, less perceptible attacks. As BlackCAtt attacks manipulate causes only, the attacks become fully explainable. We compare the performance of BlackCAtt with other black-box attack methods and show that targeting causal pixels leads to smaller and less perceptible attacks. For example, when using BlackCAtt with SquareAttack, it reduces the average distance ($L_0$ norm) of the attack from the original input from $0.987$ to $0.072$, while maintaining a similar success rate. We perform ablation studies on the BlackCAtt algorithm and analyze the effect of different components on its performance.
>
---
#### [replaced 023] Mitigating Domain Drift in Multi Species Segmentation with DINOv2: A Cross-Domain Evaluation in Herbicide Research Trials
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.07514](https://arxiv.org/pdf/2508.07514)**

> **作者:** Artzai Picon; Itziar Eguskiza; Daniel Mugica; Javier Romero; Carlos Javier Jimenez; Eric White; Gabriel Do-Lago-Junqueira; Christian Klukas; Ramon Navarra-Mestre
>
> **摘要:** Reliable plant species and damage segmentation for herbicide field research trials requires models that can withstand substantial real-world variation across seasons, geographies, devices, and sensing modalities. Most deep learning approaches trained on controlled datasets fail to generalize under these domain shifts, limiting their suitability for operational phenotyping pipelines. This study evaluates a segmentation framework that integrates vision foundation models (DINOv2) with hierarchical taxonomic inference to improve robustness across heterogeneous agricultural conditions. We train on a large, multi-year dataset collected in Germany and Spain (2018-2020), comprising 14 plant species and 4 herbicide damage classes, and assess generalization under increasingly challenging shifts: temporal and device changes (2023), geographic transfer to the United States, and extreme sensor shift to drone imagery (2024). Results show that the foundation-model backbone consistently outperforms prior baselines, improving species-level F1 from 0.52 to 0.87 on in-distribution data and maintaining significant advantages under moderate (0.77 vs. 0.24) and extreme (0.44 vs. 0.14) shift conditions. Hierarchical inference provides an additional layer of robustness, enabling meaningful predictions even when fine-grained species classification degrades (family F1: 0.68, class F1: 0.88 on aerial imagery). Error analysis reveals that failures under severe shift stem primarily from vegetation-soil confusion, suggesting that taxonomic distinctions remain preserved despite background and viewpoint variability. The system is now deployed within BASF's phenotyping workflow for herbicide research trials across multiple regions, illustrating the practical viability of combining foundation models with structured biological hierarchies for scalable, shift-resilient agricultural monitoring.
>
---
#### [replaced 024] PolySLGen: Online Multimodal Speaking-Listening Reaction Generation in Polyadic Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08125](https://arxiv.org/pdf/2604.08125)**

> **作者:** Zhi-Yi Lin; Thomas Markhorst; Jouh Yeong Chew; Xucong Zhang
>
> **摘要:** Human-like multimodal reaction generation is essential for natural group interactions between humans and embodied AI. However, existing approaches are limited to single-modality or speaking-only responses in dyadic interactions, making them unsuitable for realistic social scenarios. Many also overlook nonverbal cues and complex dynamics of polyadic interactions, both critical for engagement and conversational coherence. In this work, we present PolySLGen, an online framework for Polyadic multimodal Speaking and Listening reaction Generation. Given past conversation and motion from all participants, PolySLGen generates a future speaking or listening reaction for a target participant, including speech, body motion, and speaking state score. To model group interactions effectively, we propose a pose fusion module and a social cue encoder that jointly aggregate motion and social signals from the group. Extensive experiments, along with quantitative and qualitative evaluations, show that PolySLGen produces contextually appropriate and temporally coherent multi-modal reactions, outperforming several adapted and state-of-the-art baselines in motion quality, motion-speech alignment, speaking state prediction, and human-perceived realism.
>
---
#### [replaced 025] Revisiting Image Manipulation Localization under Realistic Manipulation Scenarios
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20006](https://arxiv.org/pdf/2509.20006)**

> **作者:** Xuekang Zhu; Ji-Zhe Zhou; Kaiwen Feng; Chenfan Qu; Xiwen Wang; Yunfei Wang; Liting Zhou; Jian Liu
>
> **摘要:** With the large models easing the labor-intensive manipulation process, image manipulations in today's real scenarios often entail a complex manipulation process, comprising a series of editing operations to create a deceptive image. However, existing IML methods remain manipulation-process-agnostic, directly producing localization masks in a one-shot prediction paradigm without modeling the underlying editing steps. This one-shot paradigm compresses the high-dimensional compositional space into a single binary mask, inducing severe dimensional collapse, which forces the model to discard essential structural cues and ultimately leads to overfitting and degraded generalization. To address this, we are the first to reformulate image manipulation localization as a conditional sequence prediction task, proposing the RITA framework. RITA predicts manipulated regions layer-by-layer in an ordered manner, using each step's prediction as the condition for the next, thereby explicitly modeling temporal dependencies and hierarchical structures among editing operations. To enable training and evaluation, we synthesize multi-step manipulation data and construct a new benchmark HSIM. We further propose the HSS metric to assess sequential order and hierarchical alignment. Extensive experiments show that: 1) RITA achieves SOTA generalization and robustness on traditional benchmarks; 2) it remains computationally efficient despite explicitly modeling multi-step sequences; and 3) it establishes a viable foundation for hierarchical, process-aware manipulation localization. Code and dataset are available at this https URL.
>
---
#### [replaced 026] SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文聚焦于柔性物体的机器人操作任务，解决仿真与现实数据不匹配的问题。提出SIM1系统，通过物理对齐实现高效数据生成与策略学习。**

- **链接: [https://arxiv.org/pdf/2604.08544](https://arxiv.org/pdf/2604.08544)**

> **作者:** Yunsong Zhou; Hangxu Liu; Xuekun Jiang; Xing Shen; Yuanzhen Zhou; Hui Wang; Baole Fang; Yang Tian; Mulin Yu; Qiaojun Yu; Li Ma; Hengjie Li; Hanqing Wang; Jia Zeng; Jiangmiao Pang
>
> **备注:** Website: this https URL
>
> **摘要:** Robotic manipulation with deformable objects represents a data-intensive regime in embodied learning, where shape, contact, and topology co-evolve in ways that far exceed the variability of rigids. Although simulation promises relief from the cost of real-world data acquisition, prevailing sim-to-real pipelines remain rooted in rigid-body abstractions, producing mismatched geometry, fragile soft dynamics, and motion primitives poorly suited for cloth interaction. We posit that simulation fails not for being synthetic, but for being ungrounded. To address this, we introduce SIM1, a physics-aligned real-to-sim-to-real data engine that grounds simulation in the physical world. Given limited demonstrations, the system digitizes scenes into metric-consistent twins, calibrates deformable dynamics through elastic modeling, and expands behaviors via diffusion-based trajectory generation with quality filtering. This pipeline transforms sparse observations into scaled synthetic supervision with near-demonstration fidelity. Experiments show that policies trained on purely synthetic data achieve parity with real-data baselines at a 1:15 equivalence ratio, while delivering 90% zero-shot success and 50% generalization gains in real-world deployment. These results validate physics-aligned simulation as scalable supervision for deformable manipulation and a practical pathway for data-efficient policy learning.
>
---
#### [replaced 027] Needle in a Haystack: One-Class Representation Learning for Detecting Rare Malignant Cells in Computational Cytology
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.07722](https://arxiv.org/pdf/2604.07722)**

> **作者:** Swarnadip Chatterjee; Vladimir Basic; Arrigo Capitanio; Orcun Goksel; Joakim Lindblad
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** In computational cytology, detecting malignancy on whole-slide images is difficult because malignant cells are morphologically diverse yet vanishingly rare amid a vast background of normal cells. Accurate detection of these extremely rare malignant cells remains challenging due to large class imbalance and limited annotations. Conventional weakly supervised approaches, such as multiple instance learning (MIL), often fail to generalize at the instance level, especially when the fraction of malignant cells (witness rate) is exceedingly low. In this study, we explore the use of one-class representation learning techniques for detecting malignant cells in low-witness-rate scenarios. These methods are trained exclusively on slide-negative patches, without requiring any instance-level supervision. Specifically, we evaluate two OCC approaches, DSVDD and DROC, and compare them with FS-SIL, WS-SIL, and the recent ItS2CLR method. The one-class methods learn compact representations of normality and detect deviations at test time. Experiments on a publicly available bone marrow cytomorphology dataset (TCIA) and an in-house oral cancer cytology dataset show that DSVDD achieves state-of-the-art performance in instance-level abnormality ranking, particularly in ultra-low witness-rate regimes ($\leq 1\%$) and, in some cases, even outperforming fully supervised learning, which is typically not a practical option in whole-slide cytology due to the infeasibility of exhaustive instance-level annotations. DROC is also competitive under extreme rarity, benefiting from distribution-augmented contrastive learning. These findings highlight one-class representation learning as a robust and interpretable superior choice to MIL for malignant cell detection under extreme rarity.
>
---
#### [replaced 028] Adversarial Concept Distillation for One-Step Diffusion Personalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20512](https://arxiv.org/pdf/2510.20512)**

> **作者:** Yixiong Yang; Tao Wu; Senmao Li; Shiqi Yang; Yaxing Wang; Joost van de Weijer; Kai Wang
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Recent progress in accelerating text-to-image diffusion models enables high-fidelity synthesis within a single denoising step. However, customizing the fast one-step models remains challenging, as existing methods consistently fail to produce acceptable results, underscoring the need for new methodologies to personalize one-step models. Therefore, we propose One-step Personalized Adversarial Distillation (OPAD), a framework that combines teacher-student distillation with adversarial supervision. A multi-step diffusion model serves as the teacher, while a one-step student model is jointly trained with it. The student learns from alignment losses that preserve consistency with the teacher and from adversarial losses that align its output with real image distributions. Beyond one-step personalization, we further observe that the student's efficient generation and adversarially enriched representations provide valuable feedback to improve the teacher model, forming a collaborative learning stage. Extensive experiments demonstrate that OPAD is the first approach to deliver reliable, high-quality personalization for one-step diffusion models; in contrast, prior methods largely fail and produce severe failure cases, while OPAD preserves single-step efficiency.
>
---
#### [replaced 029] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01400](https://arxiv.org/pdf/2603.01400)**

> **作者:** Jinlong Li; Liyuan Jiang; Haonan Zhang; Nicu Sebe
>
> **备注:** CVPR2026, Project webpage: this https URL
>
> **摘要:** Video Large Language Models (VLLMs) demonstrate strong video understanding but suffer from inefficiency due to redundant visual tokens. Existing pruning primary targets intra-frame spatial redundancy or prunes inside the LLM with shallow-layer overhead, yielding suboptimal spatiotemporal reduction and underutilizing long-context compressibility. All of them often discard subtle yet informative context from merged or pruned tokens. In this paper, we propose a new perspective that elaborates token \textbf{A}nchors within intra-frame and inter-frame to comprehensively aggregate the informative contexts via local-global \textbf{O}ptimal \textbf{T}ransport (\textbf{AOT}). Specifically, we first establish local- and global-aware token anchors within each frame under the attention guidance, which then optimal transport aggregates the informative contexts from pruned tokens, constructing intra-frame token anchors. Then, building on the temporal frame clips, the first frame within each clip will be considered as the keyframe anchors to ensemble similar information from consecutive frames through optimal transport, while keeping distinct tokens to represent temporal dynamics, leading to efficient token reduction in a training-free manner. Extensive evaluations show that our proposed AOT obtains competitive performances across various short- and long-video benchmarks on leading video LLMs, obtaining substantial computational efficiency while preserving temporal and visual fidelity. Project webpage: this https URL.
>
---
#### [replaced 030] LoBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.01767](https://arxiv.org/pdf/2510.01767)**

> **作者:** Sheng-Hsiang Hung; Ting-Yu Yen; Wei-Fang Sun; Simon See; Shih-Hsuan Hung; Hung-Kuo Chu
>
> **摘要:** 3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks and training on multiple, non-communicating GPUs, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. Specifically, LoBE-GS introduces a load-balanced KD-tree scene partitioning scheme with optimized cutlines that balance per-block camera counts. To accelerate preprocessing, it employs depth-based back-projection for fast camera assignment, reducing processing time from hours to minutes. It further reduces training cost through two lightweight techniques: visibility cropping and selective densification. Evaluations on large-scale urban and outdoor datasets show that LoBE-GS consistently achieves up to 2 times faster end-to-end training time than state-of-the-art baselines, while maintaining reconstruction quality and enabling scalability to scenes infeasible with vanilla 3DGS.
>
---
#### [replaced 031] 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17012](https://arxiv.org/pdf/2512.17012)**

> **作者:** Chiao-An Yang; Ryo Hachiuma; Sifei Liu; Subhashree Radhakrishnan; Raymond A. Yeh; Yu-Chiang Frank Wang; Min-Hung Chen
>
> **备注:** CVPR 2026 (Highlight). Project page: this https URL. GitHub: this https URL. Dataset: this https URL
>
> **摘要:** Despite advances in Multimodal LLMs (MLLMs), their ability to reason over 3D structures and temporal dynamics remains limited, constrained by weak 4D perception and temporal understanding. Existing 3D and 4D Video Question Answering (VQA) benchmarks also emphasize static scenes and lack region-level prompting. We tackle these issues by introducing: (a) 4D-RGPT, a specialized MLLM designed to capture 4D representations from video inputs with enhanced temporal perception; (b) Perceptual 4D Distillation (P4D), a training framework that transfers 4D representations from a frozen expert model into 4D-RGPT for comprehensive 4D perception; and (c) R4D-Bench, a benchmark for depth-aware dynamic scenes with region-level prompting, built via a hybrid automated and human-verified pipeline. Our 4D-RGPT achieves notable improvements on both existing 4D VQA benchmarks and the proposed R4D-Bench benchmark.
>
---
#### [replaced 032] Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.22990](https://arxiv.org/pdf/2601.22990)**

> **作者:** Yinsong Wang; Thomas Fletcher; Xinzhe Luo; Aine Travers Dineen; Rhodri Cusack; Chen Qin
>
> **摘要:** Reconstructing 3D fetal MR volumes from motion-corrupted stacks of 2D slices is a crucial and challenging task. Conventional slice-to-volume reconstruction (SVR) methods are time-consuming and require multiple orthogonal stacks for reconstruction. While learning-based SVR approaches have significantly reduced the time required at the inference stage, they heavily rely on ground truth information for training, which is inaccessible in practice. To address these challenges, we propose GaussianSVR, a self-supervised framework for slice-to-volume reconstruction. GaussianSVR represents the target volume using 3D Gaussian representations to achieve high-fidelity reconstruction. It leverages a simulated forward slice acquisition model to enable self-supervised training, alleviating the need for ground-truth volumes. Furthermore, to enhance both accuracy and efficiency, we introduce a multi-resolution training strategy that jointly optimizes Gaussian parameters and spatial transformations across different resolution levels. Experiments show that GaussianSVR outperforms the baseline methods on fetal MR volumetric reconstruction. Code is available at this https URL.
>
---
#### [replaced 033] RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19704](https://arxiv.org/pdf/2511.19704)**

> **作者:** Omar Alama; Darshil Jariwala; Avigyan Bhattacharya; Seungchan Kim; Wenshan Wang; Sebastian Scherer
>
> **备注:** Accepted to CVPR'26 Findings Code at this https URL
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) underpins many vision and robotics tasks that require generalizable semantic understanding. Existing approaches either rely on limited segmentation training data, which hinders generalization, or apply zero-shot heuristics to vision-language models (e.g CLIP), while the most competitive approaches combine multiple models to improve performance at the cost of high computational and memory demands. In this work, we leverage an overlooked agglomerative vision foundation model, RADIO, to improve zero-shot OVSS along three key axes simultaneously: mIoU, latency, and parameter efficiency. We present the first comprehensive study of RADIO for zero-shot OVSS and enhance its performance through self-correlating recursive attention, self-correlating global aggregation, and computationally efficient RADIO SAM mask refinement. Our approach, RADSeg, achieves 6-30% mIoU improvement in the base ViT class while being 3.95x faster and using 2.5x fewer parameters. Surprisingly, RADSeg-base (106M) outperforms previous combinations of huge vision models (850-1350M) in mIoU, achieving state-of-the-art accuracy with substantially lower computational and memory cost.
>
---
#### [replaced 034] FDIF: Formula-Driven supervised Learning with Implicit Functions for 3D Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23199](https://arxiv.org/pdf/2603.23199)**

> **作者:** Yukinori Yamamoto; Kazuya Nishimura; Tsukasa Fukusato; Hirokazu Nosato; Tetsuya Ogata; Hirokatsu Kataoka
>
> **摘要:** Deep learning-based 3D medical image segmentation methods relies on large-scale labeled datasets, yet acquiring such data is difficult due to privacy constraints and the high cost of expert annotation. Formula-Driven Supervised Learning (FDSL) offers an appealing alternative by generating training data and labels directly from mathematical formulas. However, existing voxel-based approaches are limited in geometric expressiveness and cannot synthesize realistic textures. We introduce Formula-Driven supervised learning with Implicit Functions (FDIF), a framework that enables scalable pre-training without using any real data and medical expert annotations. FDIF introduces an implicit-function representation based on signed distance functions (SDFs), enabling compact modeling of complex geometries while exploiting the surface representation of SDFs to support controllable synthesis of both geometric and intensity textures. Across three medical image segmentation benchmarks (AMOS, ACDC, and KiTS) and three architectures (SwinUNETR, nnUNet ResEnc-L, and nnUNet Primus-M), FDIF consistently improves over a formula-driven method, and achieves performance comparable to self-supervised approaches pre-trained on large-scale real datasets. We further show that FDIF pre-training also benefits 3D classification tasks, highlighting implicit-function-based formula supervision as a promising paradigm for data-free representation learning. Code is available at this https URL.
>
---
#### [replaced 035] Measurement-Consistent Langevin Corrector for Stabilizing Latent Diffusion Inverse Problem Solvers
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.04791](https://arxiv.org/pdf/2601.04791)**

> **作者:** Lee Hyoseok; Sohwi Lim; Eunju Cha; Tae-Hyun Oh
>
> **备注:** Under Review
>
> **摘要:** While latent diffusion models (LDMs) have emerged as powerful priors for inverse problems, existing LDM-based solvers frequently suffer from instability. In this work, we first identify the instability as a discrepancy between the solver dynamics and stable reverse diffusion dynamics learned by the diffusion model, and show that reducing this gap stabilizes the solver. Building on this, we introduce \textit{Measurement-Consistent Langevin Corrector (MCLC)}, a theoretically grounded plug-and-play stabilization module that remedies the LDM-based inverse problem solvers through measurement-consistent Langevin updates. Compared to prior approaches that rely on linear manifold assumptions, which often fail to hold in latent space, MCLC provides a principled stabilization mechanism, leading to more stable and reliable behavior in latent space.
>
---
#### [replaced 036] Enhanced Self-Supervised Multi-Image Super-Resolution for Camera Array Images
- **分类: physics.optics; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.06816](https://arxiv.org/pdf/2604.06816)**

> **作者:** Yating Chen; Feng Huang; Xianyu Wu; Jing Wu; Ying Shen
>
> **摘要:** Conventional multi-image super-resolution (MISR) methods, such as burst and video SR, rely on sequential frames from a single camera. Consequently, they suffer from complex image degradation and severe occlusion, increasing the difficulty of accurate image restoration. In contrast, multi-aperture camera-array imaging captures spatially distributed views with sampling offsets forming a stable disk-like distribution, which enhances the non-redundancy of observed data. Existing MISR algorithms fail to fully exploit these unique properties. Supervised MISR methods tend to overfit the degradation patterns in training data, and current self-supervised learning (SSL) techniques struggle to recover fine-grained details. To address these issues, this paper thoroughly investigates the strengths, limitations and applicability boundaries of multi-image-to-single-image (Multi-to-Single) and multi-image-to-multi-image (Multi-to-Multi) SSL methods. We propose the Multi-to-Single-Guided Multi-to-Multi SSL framework that combines the advantages of Multi-to-Single and Multi-to-Multi to generate visually appealing and high-fidelity images rich in texture details. The Multi-to-Single-Guided Multi-to-Multi SSL framework provides a new paradigm for integrating deep neural network with classical physics-based variational methods. To enhance the ability of MISR network to recover high-frequency details from aliased artifacts, this paper proposes a novel camera-array SR network called dual Transformer suitable for SSL. Experiments on synthetic and real-world datasets demonstrate the superiority of the proposed method.
>
---
#### [replaced 037] Generative View Stitching
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.24718](https://arxiv.org/pdf/2510.24718)**

> **作者:** Chonghyuk Song; Michal Stary; Boyuan Chen; George Kopanas; Vincent Sitzmann
>
> **备注:** Published at ICLR 2026. Camera-ready Submission. Project website: this https URL
>
> **摘要:** Autoregressive video diffusion models are capable of long rollouts that are stable and consistent with history, but they are unable to guide the current generation with conditioning from the future. In camera-guided video generation with a predefined camera trajectory, this limitation leads to collisions with the generated scene, after which autoregression quickly collapses. To address this, we propose Generative View Stitching (GVS), which samples the entire sequence in parallel such that the generated scene is faithful to every part of the predefined camera trajectory. Our main contribution is a sampling algorithm that extends prior work on diffusion stitching for robot planning to video generation. While such stitching methods usually require a specially trained model, GVS is compatible with any off-the-shelf video model trained with Diffusion Forcing, a prevalent sequence diffusion framework that we show already provides the affordances necessary for stitching. We then introduce Omni Guidance, a technique that enhances the temporal consistency in stitching by conditioning on both the past and future, and that enables our proposed loop-closing mechanism for delivering long-range coherence. Overall, GVS achieves camera-guided video generation that is stable, collision-free, frame-to-frame consistent, and closes loops for a variety of predefined camera paths, including Oscar Reutersvärd's Impossible Staircase. Results are best viewed as videos at this https URL.
>
---
#### [replaced 038] When & How to Write for Personalized Demand-aware Query Rewriting in Video Search
- **分类: cs.IR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.17667](https://arxiv.org/pdf/2602.17667)**

> **作者:** Cheng cheng; Chenxing Wang; Aolin Li; Haijun Wu; Huiyun Hu; Juyuan Wang
>
> **摘要:** In video search systems, user historical behaviors provide rich context for identifying search intent and resolving ambiguity. However, traditional methods utilizing implicit history features often suffer from signal dilution and delayed feedback. To address these challenges, we propose WeWrite, a novel Personalized Demand-aware Query Rewriting framework. Specifically, WeWrite tackles three key challenges: (1) When to Write: An automated posterior-based mining strategy extracts high-quality samples from user logs, identifying scenarios where personalization is strictly necessary; (2) How to Write: A hybrid training paradigm combines Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to align the LLM's output style with the retrieval system; (3) Deployment: A parallel "Fake Recall" architecture ensures low latency. Online A/B testing on a large-scale video platform demonstrates that WeWrite improves the Click-Through Video Volume (VV$>$10s) by 1.07% and reduces the Query Reformulation Rate by 2.97%.
>
---
#### [replaced 039] Growing a Multi-head Twig via Distillation and Reinforcement Learning to Accelerate Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型加速任务，旨在解决模型计算开销大和生成速度慢的问题。通过引入轻量级模块和优化策略，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.14075](https://arxiv.org/pdf/2503.14075)**

> **作者:** Zhenwei Shao; Mingyang Wang; Weijun Zhang; Zhou Yu; Wenwen Pan; Yan Yang; Tao Wei; Hongyuan Zhang; Jun Yu
>
> **备注:** An extended version of our ICCV paper at this https URL
>
> **摘要:** Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (e.g., 30 tokens). To address the limitations above, we present TwigVLM -- a simple and general architecture by growing a lightweight module, named twig, upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves better accuracy retention by employing a twig-guided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of the visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods. Moreover, we extend TwigVLM to an improved TwigVLM++ variant by introducing a novel multi-head twig architecture with a specialized pruning head. TwigVLM++ improves pruning quality via a two-stage training paradigm combining a distillation learning stage and a pruning-oriented reinforcement learning stage, and further accelerates inference via a tree-based SSD strategy.
>
---
#### [replaced 040] OmniPrism: Learning Disentangled Visual Concept for Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.12242](https://arxiv.org/pdf/2412.12242)**

> **作者:** Yangyang Li; Daqing Liu; Wu Liu; Allen He; Xinchen Liu; Yongdong Zhang; Guoqing Jin
>
> **备注:** WebPage available at this https URL
>
> **摘要:** Creative visual concept generation often draws inspiration from specific concepts in a reference image to produce relevant outcomes. However, existing methods are typically constrained to single-aspect concept generation or are easily disrupted by irrelevant concepts in multi-aspect concept scenarios, leading to concept confusion and hindering creative generation. To address this, we propose OmniPrism, a visual concept disentangling approach for creative image generation. Our method learns disentangled concept representations guided by natural language and trains a diffusion model to incorporate these concepts. We utilize the rich semantic space of a multimodal extractor to achieve concept disentanglement from given images and concept guidance. To disentangle concepts with different semantics, we construct a paired concept disentangled dataset (PCD-200K), where each pair shares the same concept such as content, style, and composition. We learn disentangled concept representations through our contrastive orthogonal disentangled (COD) training pipeline, which are then injected into additional diffusion cross-attention layers for generation. A set of block embeddings is designed to adapt each block's concept domain in the diffusion models. Extensive experiments demonstrate that our method can generate high-quality, concept-disentangled results with high fidelity to text prompts and desired concepts.
>
---
#### [replaced 041] Zero-Shot Generative De-identification: Inversion-Free Flow for Privacy-Preserving Skin Image Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.00821](https://arxiv.org/pdf/2602.00821)**

> **作者:** Konstantinos Moutselos; Ilias Maglogiannis
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** The secure analysis of dermatological images in clinical environments is fundamentally restricted by the critical trade-off between patient privacy and the preservation of diagnostic fidelity. Traditional de-identification techniques often degrade essential pathological markers, while state-of-the-art generative approaches typically require computationally intensive inversion processes or extensive task-specific fine-tuning, limiting their feasibility for real-time deployment. This study introduces a zero-shot generative de-identification framework that utilizes an inversion-free pipeline for privacy-preserving medical image analysis. By leveraging Rectified Flow Transformers (FlowEdit), the proposed method achieves high-fidelity identity transformation in less than 20 seconds without requiring pathology-specific training or labeled datasets. We introduce a novel "segment-by-synthesis" mechanism that generates counterfactual "healthy" and "pathological" digital twin pairs to isolate clinical signals from biometric identifiers in a zero-shot manner. Our approach specifically utilizes the CIELAB color space to decouple erythema-related pathological signals from semantic noise and individual skin characteristics. Pilot validation on high-resolution clinical samples demonstrates robust stability in preserving pathological features, achieving an Intersection over Union (IoU) stability exceeding 0.67, while ensuring rigorous de-identification. These results suggest that the proposed zero-shot, inversion-free approach provides a scalable and efficient solution for secure data sharing and collaborative biomedical research, bypassing the need for large-scale annotated medical datasets while aligning with data protection standards.
>
---
#### [replaced 042] Intrinsic Concept Extraction Based on Compositional Interpretability
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11795](https://arxiv.org/pdf/2603.11795)**

> **作者:** Hanyu Shi; Hong Tao; Guoheng Huang; Jianbin Jiang; Xuhang Chen; Chi-Man Pun; Shanhu Wang; Pan Pan
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Unsupervised Concept Extraction aims to extract concepts from a single image; however, existing methods suffer from the inability to extract composable intrinsic concepts. To address this, this paper introduces a new task called Compositional and Interpretable Intrinsic Concept Extraction (CI-ICE). The CI-ICE task aims to leverage diffusion-based text-to-image models to extract composable object-level and attribute-level concepts from a single image, such that the original concept can be reconstructed through the combination of these concepts. To achieve this goal, we propose a method called HyperExpress, which addresses the CI-ICE task through two core aspects. Specifically, first, we propose a concept learning approach that leverages the inherent hierarchical modeling capability of hyperbolic space to achieve accurate concept disentanglement while preserving the hierarchical structure and relational dependencies among concepts; second, we introduce a concept-wise optimization method that maps the concept embedding space to maintain complex inter-concept relationships while ensuring concept composability. Our method demonstrates outstanding performance in extracting compositionally interpretable intrinsic concepts from a single image.
>
---
#### [replaced 043] CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.18561](https://arxiv.org/pdf/2603.18561)**

> **作者:** Jiacheng Tang; Zhiyuan Zhou; Zhuolin He; Jia Zhang; Kai Zhang; Jian Pu
>
> **备注:** Accepted to CVPR 2026 (Highlight)
>
> **摘要:** Planning-oriented end-to-end driving models show great promise, yet they fundamentally learn statistical correlations instead of true causal relationships. This vulnerability leads to causal confusion, where models exploit dataset biases as shortcuts, critically harming their reliability and safety in complex scenarios. To address this, we introduce CausalVAD, a de-confounding training framework that leverages causal intervention. At its core, we design the sparse causal intervention scheme (SCIS), a lightweight, plug-and-play module to instantiate the backdoor adjustment theory in neural networks. SCIS constructs a dictionary of prototypes representing latent driving contexts. It then uses this dictionary to intervene on the model's sparse vectorized queries. This step actively eliminates spurious associations induced by confounders, thereby eliminating spurious factors from the representations for downstream tasks. Extensive experiments on benchmarks like nuScenes show CausalVAD achieves state-of-the-art planning accuracy and safety. Furthermore, our method demonstrates superior robustness against both data bias and noisy scenarios configured to induce causal confusion.
>
---
#### [replaced 044] REACT3D: Recovering Articulations for Interactive Physical 3D Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出REACT3D，解决静态3D场景交互化问题，通过检测、分割、估计关节等步骤生成可模拟的互动场景，提升场景理解研究效率。**

- **链接: [https://arxiv.org/pdf/2510.11340](https://arxiv.org/pdf/2510.11340)**

> **作者:** Zhao Huang; Boyang Sun; Alexandros Delitzas; Jiaqi Chen; Marc Pollefeys
>
> **备注:** Accepted at IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is this https URL
>
---
#### [replaced 045] ShelfGaussian: Shelf-Supervised Open-Vocabulary Gaussian-based 3D Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03370](https://arxiv.org/pdf/2512.03370)**

> **作者:** Lingjun Zhao; Yandong Luo; James Hays; Lu Gan
>
> **摘要:** We introduce ShelfGaussian, an open-vocabulary multi-modal Gaussian-based 3D scene understanding framework supervised by off-the-shelf vision foundation models (VFMs). Gaussian-based methods have demonstrated superior performance and computational efficiency across a wide range of scene understanding tasks. However, existing methods either model objects as closed-set semantic Gaussians supervised by annotated 3D labels, neglecting their rendering ability, or learn open-set Gaussian representations via purely 2D self-supervision, leading to degraded geometry and limited to camera-only settings. To fully exploit the potential of Gaussians, we propose a Multi-Modal Gaussian Transformer that enables Gaussians to query features from diverse sensor modalities, and a Shelf-Supervised Learning Paradigm that efficiently optimizes Gaussians with VFM features jointly at 2D image and 3D scene levels. We evaluate ShelfGaussian on various perception and planning tasks. Experiments on Occ3D-nuScenes demonstrate its state-of-the-art zero-shot semantic occupancy prediction performance. ShelfGaussian is further evaluated on an unmanned ground vehicle (UGV) to assess its in the-wild performance across diverse urban scenarios. Project website: this https URL.
>
---
#### [replaced 046] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作任务，解决传统方法忽略历史信息的问题。提出AVA-VLA框架，通过递归状态和主动视觉注意力提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18960](https://arxiv.org/pdf/2511.18960)**

> **作者:** Lei Xiao; Jifeng Li; Juntao Gao; Feiyang Ye; Yan Jin; Jingjing Qian; Jing Zhang; Yong Wu; Xiaoyuan Yu
>
> **备注:** Accepted at CVPR 2026 (Highlight)
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable progress in embodied tasks recently, but most methods process visual observations independently at each timestep. This history-agnostic design treats robot manipulation as a Markov Decision Process, even though real-world robotic control is inherently partially observable and requires reasoning over past interactions. To address this mismatch, we reformulate VLA policy learning from a Partially Observable Markov Decision Process perspective and propose AVA-VLA, a framework that conditions action generation on a recurrent state that serves as a neural approximation to the agent's belief over task history. Built on this recurrent state, we introduce Active Visual Attention (AVA), which dynamically reweights visual tokens in the current observation to focus on regions most relevant given both the instruction and execution history. Extensive experiments show that AVA-VLA achieves state-of-the-art performance on standard robotic benchmarks, including LIBERO and CALVIN, and transfers effectively to real-world dual-arm manipulation tasks. These results demonstrate the effectiveness of temporally grounded active visual processing for improving VLA performance in robotic sequential decision-making. The project page is available at this https URL.
>
---
#### [replaced 047] ParseBench: A Document Parsing Benchmark for AI Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08538](https://arxiv.org/pdf/2604.08538)**

> **作者:** Boyang Zhang; Sebastián G. Acosta; Preston Carlson; Sacha Bron; Pierre-Loïc Doulcet; Daniel B. Ospina; Simon Suo
>
> **摘要:** AI agents are changing the requirements for document parsing. What matters is \emph{semantic correctness}: parsed output must preserve the structure and meaning needed for autonomous decisions, including correct table structure, precise chart data, semantically meaningful formatting, and visual grounding. Existing benchmarks do not fully capture this setting for enterprise automation, relying on narrow document distributions and text-similarity metrics that miss agent-critical failures. We introduce \textbf{ParseBench}, a benchmark of ${\sim}2{,}000$ human-verified pages from enterprise documents spanning insurance, finance, and government, organized around five capability dimensions: tables, charts, content faithfulness, semantic formatting, and visual grounding. Across 14 methods spanning vision-language models, specialized document parsers, and LlamaParse, the benchmark reveals a fragmented capability landscape: no method is consistently strong across all five dimensions. LlamaParse Agentic achieves the highest overall score at \agenticoverall\%, and the benchmark highlights the remaining capability gaps across current systems. Dataset and evaluation code are available on this https URL and this https URL.
>
---
#### [replaced 048] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06656](https://arxiv.org/pdf/2508.06656)**

> **作者:** Denis Lukovnikov; Andreas Müller; Erwin Quiring; Asja Fischer
>
> **备注:** CVPR 2026
>
> **摘要:** In-generation watermarking for latent diffusion models has recently shown high robustness in marking generated images for easier detection and attribution. However, its application to autoregressive (AR) image models is underexplored. Autoregressive models generate images by autoregressively predicting a sequence of visual tokens that are then decoded into pixels using a VQ-VAE decoder. Inspired by KGW watermarking for large language models, we examine token-level watermarking schemes that bias the next-token prediction based on prior tokens. We find that a direct transfer of these schemes works in principle, but the detectability of the watermarks decreases considerably under common image perturbations. As a remedy, we propose a watermarking approach based on visual token clustering, which assigns similar tokens to the same set (red or green). We investigate token clustering in a training-free setting, as well as in combination with a more accurate fine-tuned token or cluster predictor. Overall, our experiments show that cluster-based watermarks greatly improve robustness against perturbations and regeneration attacks while preserving image quality, outperforming a set of baselines and concurrent works. Moreover, our methods offer fast verification runtime, comparable to lightweight post-hoc watermarking techniques.
>
---
#### [replaced 049] Chronological Contrastive Learning: Few-Shot Progression Assessment in Irreversible Diseases
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.21935](https://arxiv.org/pdf/2603.21935)**

> **作者:** Clemens Watzenböck; Daniel Aletaha; Michaël Deman; Thomas Deimel; Jana Eder; Ivana Janickova; Robert Janiczek; Peter Mandl; Philipp Seeböck; Gabriela Supp; Paul Weiser; Georg Langs
>
> **备注:** Accepted for MIDL 2026; Reviews available at this https URL
>
> **摘要:** Quantitative disease severity scoring in medical imaging is costly, time-consuming, and subject to inter-reader variability. At the same time, clinical archives contain far more longitudinal imaging data than expert-annotated severity scores. Existing self-supervised methods typically ignore this chronological structure. We introduce ChronoCon, a contrastive learning approach that replaces label-based ranking losses with rankings derived solely from the visitation order of a patient's longitudinal scans. Under the clinically plausible assumption of monotonic progression in irreversible diseases, the method learns disease-relevant representations without using any expert labels. This generalizes the idea of Rank-N-Contrast from label distances to temporal ordering. Evaluated on rheumatoid arthritis radiographs for severity assessment, the learned representations substantially improve label efficiency. In low-label settings, ChronoCon significantly outperforms a fully supervised baseline initialized from ImageNet weights. In a few-shot learning experiment, fine-tuning ChronoCon on expert scores from only five patients yields an intraclass correlation coefficient of 86% for severity score prediction. These results demonstrate the potential of chronological contrastive learning to exploit routinely available imaging metadata to reduce annotation requirements in the irreversible disease domain. Code is available at this https URL.
>
---
#### [replaced 050] A Compact Hybrid Convolution--Frequency State Space Network for Learned Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20151](https://arxiv.org/pdf/2511.20151)**

> **作者:** Haodong Pan; Hao Wei; Yusong Wang; Nanning Zheng; Caigui Jiang
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Learned image compression (LIC) has recently benefited from Transformer- and state space models (SSM)- based backbones for modeling long-range dependencies. However, the former typically incurs quadratic complexity, whereas the latter often disrupts neighborhood continuity by flattening 2D features into 1D sequences. To address these issues, we propose a compact Hybrid Convolution and Frequency State Space Network (HCFSSNet) for LIC. HCFSSNet combines convolutional layers for local detail modeling with a Vision Frequency State Space (VFSS) block for complementary long-range contextual aggregation. Specifically, the VFSS block consists of a Vision Omni-directional Neighborhood State Space (VONSS) module, which scans features along horizontal, vertical, and diagonal directions to better preserve 2D neighborhood relations, and an Adaptive Frequency Modulation Module (AFMM), which performs discrete cosine transform-based adaptive reweighting of frequency components. In addition, we introduce a Frequency Swin Transformer Attention Module (FSTAM) in the hyperprior path to enhance frequency-aware side information modeling. Experiments on the benchmark datasets show that the proposed HCFSSNet achieves a competitive rate-distortion performance against recent LIC codecs. The source code and models will be made publicly available.
>
---
#### [replaced 051] CoMoVi: Co-Generation of 3D Human Motions and Realistic Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.10632](https://arxiv.org/pdf/2601.10632)**

> **作者:** Chengfeng Zhao; Jiazhi Shu; Yubo Zhao; Tianyu Huang; Jiahao Lu; Zekai Gu; Chengwei Ren; Zhiyang Dou; Qing Shuai; Yuan Liu
>
> **备注:** Project Page: this https URL
>
> **摘要:** In this paper, we find that the generation of 3D human motions and 2D human videos is intrinsically coupled. 3D motions provide the structural prior for plausibility and consistency in videos, while pre-trained video models offer strong generalization capabilities for motions. Based on this, we present CoMoVi, a co-generative framework that generates 3D human motions and videos synchronously within a single diffusion denoising loop. However, since the 3D human motions and the 2D human-centric videos have a modality gap between each other, we propose to project the 3D human motion into an effective 2D human motion representation that effectively aligns with the 2D videos. Then, we design a dual-branch diffusion model to couple human motion and the video generation process with mutual feature interaction and 3D-2D cross attentions. To train and evaluate our model, we curate CoMoVi-Dataset, a large-scale real-world human video dataset with text and motion annotations, covering diverse and challenging human motions. Extensive experiments demonstrate that our method generates high-quality 3D human motion with a better generalization ability and that our method can generate high-quality human-centric videos without external motion references.
>
---
#### [replaced 052] VSI: Visual Subtitle Integration for Keyframe Selection to enhance Long Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06869](https://arxiv.org/pdf/2508.06869)**

> **作者:** Jianxiang He; Meisheng Hong; Jungang Li; Weiyu Guo; Xuming Hu; Hui Xiong
>
> **备注:** Accepted to CVPR 2026 Findings, 10 pages
>
> **摘要:** Multimodal large language models (MLLMs) demonstrate exceptional performance in vision-language tasks, yet their processing of long videos is constrained by input context length and high computational costs. Sparse frame sampling thus becomes a necessary preprocessing step, with sampled frame quality directly impacting downstream performance. Existing keyframe search algorithms achieve a balance between efficiency and sampled frame quality but heavily rely on the visual modality alone. This makes them difficult to adapt to text-related tasks and often leads to retrieval results deviating from core semantic content. To address this, we propose the VISUAL-SUBTITLE INTEGRATION (VSI), a multimodal keyframe retrieval framework. It employs a dual-branch collaborative retrieval approach combining Video Search and Subtitle Match to fuse complementary visual and textual information for precise localization. Experiments on LongVideoBench and VideoMME demonstrate that VSI achieves state-of-the-art accuracy in keyframe retrieval while delivering breakthrough performance in text-related tasks and exhibiting strong generalization across other tasks.
>
---
#### [replaced 053] VisionLaw: Inferring Interpretable Intrinsic Dynamics from Visual Observations via Bilevel Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13792](https://arxiv.org/pdf/2508.13792)**

> **作者:** Jiajing Lin; Shu Jiang; Qingyuan Zeng; Zhenzhong Wang; Min Jiang
>
> **备注:** Accepted by ICLR 2026; Project Page: this https URL
>
> **摘要:** The intrinsic dynamics of an object governs its physical behavior in the real world, playing a critical role in enabling physically plausible interactive simulation with 3D assets. Existing methods have attempted to infer the intrinsic dynamics of objects from visual observations, but generally face two major challenges: one line of work relies on manually defined constitutive priors, making it difficult to align with actual intrinsic dynamics; the other models intrinsic dynamics using neural networks, resulting in limited interpretability and poor generalization. To address these challenges, we propose VisionLaw, a bilevel optimization framework that infers interpretable expressions of intrinsic dynamics from visual observations. At the upper level, we introduce an LLMs-driven decoupled constitutive evolution strategy, where LLMs are prompted to act as physics experts to generate and revise constitutive laws, with a built-in decoupling mechanism that substantially reduces the search complexity of LLMs. At the lower level, we introduce a vision-guided constitutive evaluation mechanism, which utilizes visual simulation to evaluate the consistency between the generated constitutive law and the underlying intrinsic dynamics, thereby guiding the upper-level evolution. Experiments on both synthetic and real-world datasets demonstrate that VisionLaw can effectively infer interpretable intrinsic dynamics from visual observations. It significantly outperforms existing state-of-the-art methods and exhibits strong generalization for interactive simulation in novel scenarios.
>
---
#### [replaced 054] B-MoE: A Body-Part-Aware Mixture-of-Experts "All Parts Matter" Approach to Micro-Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24245](https://arxiv.org/pdf/2603.24245)**

> **作者:** Nishit Poddar; Aglind Reka; Diana-Laura Borza; Snehashis Majhi; Michal Balazia; Abhijit Das; Francois Bremond
>
> **摘要:** Micro-actions, fleeting and low-amplitude motions, such as glances, nods, or minor posture shifts, carry rich social meaning but remain difficult for current action recognition models to recognize due to their subtlety, short duration, and high inter-class ambiguity. In this paper, we introduce B-MoE, a Body-part-aware Mixture-of-Experts framework designed to explicitly model the structured nature of human motion. In B-MoE, each expert specializes in a distinct body region (head, body, upper limbs, lower limbs), and is based on the lightweight Macro-Micro Motion Encoder (M3E) that captures long-range contextual structure and fine-grained local motion. A cross-attention routing mechanism learns inter-region relationships and dynamically selects the most informative regions for each micro-action. B-MoE uses a dual-stream encoder that fuses these region-specific semantic cues with global motion features to jointly capture spatially localized cues and temporally subtle variations that characterize micro-actions. Experiments on three challenging benchmarks (MA-52, SocialGesture, and MPII-GroupInteraction) show consistent state-of-theart gains, with improvements in ambiguous, underrepresented, and low amplitude classes.
>
---
#### [replaced 055] CAMotion: A High-Quality Benchmark for Camouflaged Moving Object Detection in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08287](https://arxiv.org/pdf/2604.08287)**

> **作者:** Siyuan Yao; Hao Sun; Ruiqi Yu; Xiwei Jiang; Wenqi Ren; Xiaochun Cao
>
> **备注:** Under review
>
> **摘要:** Discovering camouflaged objects is a challenging task in computer vision due to the high similarity between camouflaged objects and their surroundings. While the problem of camouflaged object detection over sequential video frames has received increasing attention, the scale and diversity of existing video camouflaged object detection (VCOD) datasets are greatly limited, which hinders the deeper analysis and broader evaluation of recent deep learning-based algorithms with data-hungry training strategy. To break this bottleneck, in this paper, we construct CAMotion, a high-quality benchmark covers a wide range of species for camouflaged moving object detection in the wild. CAMotion comprises various sequences with multiple challenging attributes such as uncertain edge, occlusion, motion blur, and shape complexity, etc. The sequence annotation details and statistical distribution are presented from various perspectives, allowing CAMotion to provide in-depth analyses on the camouflaged object's motion characteristics in different challenging scenarios. Additionally, we conduct a comprehensive evaluation of existing SOTA models on CAMotion, and discuss the major challenges in VCOD task. The benchmark is available at this https URL, we hope that our CAMotion can lead to further advancements in the research community.
>
---
#### [replaced 056] Listener-Rewarded Thinking in VLMs for Image Preferences
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.22832](https://arxiv.org/pdf/2506.22832)**

> **作者:** Alexander Gambashidze; Li Pengyi; Matvey Skripkin; Andrey Galichin; Anton Gusarov; Konstantin Sobolev; Andrey Kuznetsov; Ivan Oseledets
>
> **备注:** part of a different work
>
> **摘要:** Training robust and generalizable reward models for human visual preferences is essential for aligning text-to-image and text-to-video generative models with human intent. However, current reward models often fail to generalize, and supervised fine-tuning leads to memorization, demanding complex annotation pipelines. While reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO), improves generalization, we uncover a key failure mode: a significant drop in reasoning accuracy occurs when a model's reasoning trace contradicts that of an independent, frozen vision-language model ("listener") evaluating the same output. To address this, we introduce a listener-augmented GRPO framework. Here, the listener re-evaluates the reasoner's chain-of-thought to provide a dense, calibrated confidence score, shaping the RL reward signal. This encourages the reasoner not only to answer correctly, but to produce explanations that are persuasive to an independent model. Our listener-shaped reward scheme achieves best accuracy on the ImageReward benchmark (67.4%), significantly improves out-of-distribution (OOD) performance on a large-scale human preference dataset (1.2M votes, up to +6% over naive reasoner), and reduces reasoning contradictions compared to strong GRPO and SFT baselines. These results demonstrate that listener-based rewards provide a scalable, data-efficient path to aligning vision-language models with nuanced human preferences. We will release our reasoning model here: this https URL.
>
---
#### [replaced 057] Enhancing the Safety of Medical Vision-Language Models by Synthetic Demonstrations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.09067](https://arxiv.org/pdf/2506.09067)**

> **作者:** Zhiyu Xue; Reza Abbasi-Asl; Ramtin Pedarsani
>
> **摘要:** Generative medical vision-language models~(Med-VLMs) are primarily designed to generate complex textual information~(e.g., diagnostic reports) from multimodal inputs including vision modality~(e.g., medical images) and language modality~(e.g., clinical queries). However, their security vulnerabilities remain underexplored. Med-VLMs should be capable of rejecting harmful queries, such as \textit{Provide detailed instructions for using this CT scan for insurance fraud}. At the same time, addressing security concerns introduces the risk of over-defense, where safety-enhancing mechanisms may degrade general performance, causing Med-VLMs to reject benign clinical queries. In this paper, we propose a novel inference-time defense strategy to mitigate harmful queries, enabling defense against visual and textual jailbreak attacks. Using diverse medical imaging datasets collected from nine modalities, we demonstrate that our defense strategy based on synthetic clinical demonstrations enhances model safety without significantly compromising performance. Additionally, we find that increasing the demonstration budget alleviates the over-defense issue. We then introduce a mixed demonstration strategy as a trade-off solution for balancing security and performance under few-shot demonstration budget constraints.
>
---
#### [replaced 058] Gen-n-Val: Agentic Image Data Generation and Validation
- **分类: cs.CV; cs.AI; cs.LG; cs.MA**

- **链接: [https://arxiv.org/pdf/2506.04676](https://arxiv.org/pdf/2506.04676)**

> **作者:** Jing-En Huang; I-Sheng Fang; Tzuhsuan Huang; Yu-Lun Liu; Chih-Yu Wang; Jun-Cheng Chen
>
> **备注:** Accepted to the CVPR 2026 Findings track
>
> **摘要:** The data scarcity, label noise, and long-tailed category imbalance remain important and unresolved challenges in many computer vision tasks, such as object detection and instance segmentation, especially on large-vocabulary benchmarks like LVIS, where most categories appear in only a few images. Current synthetic data generation methods still suffer from multiple objects per mask, inaccurate segmentation, incorrect category labels, and other issues, limiting their effectiveness. To address these issues, we introduce Gen-n-Val, a novel agentic data generation framework that leverages Layer Diffusion (LD), a Large Language Model (LLM), and a Vision Large Language Model (VLLM) to produce high-quality and diverse instance masks and images for object detection and instance segmentation. Gen-n-Val consists of two agents: (1) the LD prompt agent, an LLM, optimizes rompts to encourage LD to generate high-quality foreground single-object images and corresponding segmentation masks; and (2) the data validation agent, a VLLM, filters out low-quality synthetic instance images. The system prompts for both agents are optimized by TextGrad. Compared to state-of-the-art synthetic data approaches like MosaicFusion, our approach reduces invalid synthetic data from 50% to 7% and improves performance by 7.6% on rare classes in LVIS instance segmentation with Mask R-CNN, and by 3.6% mAP on rare classes in COCO instance segmentation with YOLOv9c and YOLO11m. Furthermore, Gen-n-Val shows significant improvements (7.1% mAP) over YOLO-Worldv2-M in open-vocabulary object detection benchmarks with YOLO11m. Moreover, Gen-n-Val has scalability in model capacity and dataset size. The code is available at this https URL.
>
---
#### [replaced 059] Unmasking Puppeteers: Leveraging Biometric Leakage to Disarm Impersonation in AI-based Videoconferencing
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.03548](https://arxiv.org/pdf/2510.03548)**

> **作者:** Danial Samadi Vahdati; Tai Duc Nguyen; Ekta Prashnani; Koki Nagano; David Luebke; Orazio Gallo; Matthew Stamm
>
> **摘要:** AI-based talking-head videoconferencing systems reduce bandwidth by sending a compact pose-expression latent and re-synthesizing RGB at the receiver, but this latent can be puppeteered, letting an attacker hijack a victim's likeness in real time. Because every frame is synthetic, deepfake and synthetic video detectors fail outright. To address this security problem, we exploit a key observation: the pose-expression latent inherently contains biometric information of the driving identity. Therefore, we introduce the first biometric leakage defense without ever looking at the reconstructed RGB video: a pose-conditioned, large-margin contrastive encoder that isolates persistent identity cues inside the transmitted latent while cancelling transient pose and expression. A simple cosine test on this disentangled embedding flags illicit identity swaps as the video is rendered. Our experiments on multiple talking-head generation models show that our method consistently outperforms existing puppeteering defenses, operates in real-time, and shows strong generalization to out-of-distribution scenarios.
>
---
#### [replaced 060] DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2604.06161](https://arxiv.org/pdf/2604.06161)**

> **作者:** Zhengming Yu; Li Ma; Mingming He; Leo Isikdogan; Yuancheng Xu; Dmitriy Smirnov; Pablo Salamanca; Dao Mi; Pablo Delgado; Ning Yu; Julien Philip; Xin Li; Wenping Wang; Paul Debevec
>
> **备注:** 28 pages, 13 figures
>
> **摘要:** Most digital videos are stored in 8-bit low dynamic range (LDR) formats, where much of the original high dynamic range (HDR) scene radiance is lost due to saturation and quantization. This loss of highlight and shadow detail precludes mapping accurate luminance to HDR displays and limits meaningful re-exposure in post-production workflows. Although techniques have been proposed to convert LDR images to HDR through dynamic range expansion, they struggle to restore realistic detail in the over- and underexposed regions. To address this, we present DiffHDR, a framework that formulates LDR-to-HDR conversion as a generative radiance inpainting task within the latent space of a video diffusion model. By operating in Log-Gamma color space, DiffHDR leverages spatio-temporal generative priors from a pretrained video diffusion model to synthesize plausible HDR radiance in over- and underexposed regions while recovering the continuous scene radiance of the quantized pixels. Our framework further enables controllable LDR-to-HDR video conversion guided by text prompts or reference images. To address the scarcity of paired HDR video data, we develop a pipeline that synthesizes high-quality HDR video training data from static HDRI maps. Extensive experiments demonstrate that DiffHDR significantly outperforms state-of-the-art approaches in radiance fidelity and temporal stability, producing realistic HDR videos with considerable latitude for re-exposure.
>
---
#### [replaced 061] Adversarial Evasion Attacks on Computer Vision using SHAP Values
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.10587](https://arxiv.org/pdf/2601.10587)**

> **作者:** Frank Mollard; Marcus Becker; Florian Roehrbein
>
> **备注:** 10th bwHPC Symposium - September 25th & 26th, 2024
>
> **摘要:** The paper introduces a white-box attack on computer vision models using SHAP values. It demonstrates how adversarial evasion attacks can compromise the performance of deep learning models by reducing output confidence or inducing misclassifications. Such attacks are particularly insidious as they can deceive the perception of an algorithm while eluding human perception due to their imperceptibility to the human eye. The proposed attack leverages SHAP values to quantify the significance of individual inputs to the output at the inference stage. A comparison is drawn between the SHAP attack and the well-known Fast Gradient Sign Method. We find evidence that SHAP attacks are more robust in generating misclassifications particularly in gradient hiding scenarios.
>
---
#### [replaced 062] SelfHVD: Self-Supervised Handheld Video Deblurring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.08605](https://arxiv.org/pdf/2508.08605)**

> **作者:** Honglei Xu; Zhilu Zhang; Junjie Fan; Xiaohe Wu; Wangmeng Zuo
>
> **备注:** CVPR 2026
>
> **摘要:** Shooting video with handheld shooting devices often results in blurry frames due to shaking hands and other instability factors. Although previous video deblurring methods have achieved impressive progress, they still struggle to perform satisfactorily on real-world handheld video due to the blur domain gap between training and testing data. To address the issue, we propose a self-supervised method for handheld video deblurring, which is driven by sharp clues in the video. First, to train the deblurring model, we extract the sharp clues from the video and take them as misalignment labels of neighboring blurry frames. Second, to improve the deblurring ability of the model, we propose a novel Self-Enhanced Video Deblurring (SEVD) method to create higher-quality paired video data. Third, we propose a Self-Constrained Spatial Consistency Maintenance (SCSCM) method to regularize the model, preventing position shifts between the output and input frames. Moreover, we construct synthetic and real-world handheld video datasets for handheld video deblurring. Extensive experiments on these and other common real-world datasets demonstrate that our method significantly outperforms existing self-supervised ones. The code and datasets are publicly available at this https URL.
>
---
#### [replaced 063] Towards Context-Aware Image Anonymization with Multi-Agent Reasoning
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2603.27817](https://arxiv.org/pdf/2603.27817)**

> **作者:** Robert Aufschläger; Jakob Folz; Gautam Savaliya; Manjitha D Vidanalage; Michael Heigl; Martin Schramm
>
> **备注:** Accepted to IEEE CVPR 2026 GRAIL-V Workshop
>
> **摘要:** Street-level imagery contains personally identifiable information (PII), some of which is context-dependent. Existing anonymization methods either over-process images or miss subtle identifiers, while API-based solutions compromise data sovereignty. We present an agentic framework CAIAMAR (\underline{C}ontext-\underline{A}ware \underline{I}mage \underline{A}nonymization with \underline{M}ulti-\underline{A}gent \underline{R}easoning) for context-aware PII segmentation with diffusion-based anonymization, combining pre-defined processing for high-confidence cases with multi-agent reasoning for indirect identifiers. Three specialized agents coordinate via round-robin speaker selection in a Plan-Do-Check-Act (PDCA) cycle, enabling large vision-language models to classify PII based on spatial context (private vs. public property) rather than rigid category rules. The agents implement spatially-filtered coarse-to-fine detection where a scout-and-zoom strategy identifies candidates, open-vocabulary segmentation processes localized crops, and $IoU$-based deduplication ($30\%$ threshold) prevents redundant processing. Modal-specific diffusion guidance with appearance decorrelation substantially reduces re-identification (Re-ID) risks. On CUHK03-NP, our method reduces person Re-ID risk by $73\%$ ($R1$: $16.9\%$ vs. $62.4\%$ baseline). For image quality preservation on CityScapes, we achieve KID: $0.001$, and FID: $9.1$, significantly outperforming existing anonymization. The agentic workflow detects non-direct PII instances across object categories, and downstream semantic segmentation is preserved. Operating entirely on-premise with open-source models, the framework generates human-interpretable audit trails supporting EU's GDPR transparency requirements while flagging failed cases for human review.
>
---
#### [replaced 064] SyncBreaker:Stage-Aware Multimodal Adversarial Attacks on Audio-Driven Talking Head Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08405](https://arxiv.org/pdf/2604.08405)**

> **作者:** Wenli Zhang; Xianglong Shi; Sirui Zhao; Xinqi Chen; Guo Cheng; Yifan Xu; Tong Xu; Yong Liao
>
> **摘要:** Diffusion-based audio-driven talking-head generation enables realistic portrait animation, but also introduces risks of misuse, such as fraud and misinformation. Existing protection methods are largely limited to a single modality, and neither image-only nor audio-only attacks can effectively suppress speech-driven facial dynamics. To address this gap, we propose SyncBreaker, a stage-aware multimodal protection framework that jointly perturbs portrait and audio inputs under modality-specific perceptual constraints. Our key contributions are twofold. First, for the image stream, we introduce nullifying supervision with Multi-Interval Sampling (MIS) across diffusion stages to steer the generation toward the static reference portrait by aggregating guidance from multiple denoising intervals. Second, for the audio stream, we propose Cross-Attention Fooling (CAF), which suppresses interval-specific audio-conditioned cross-attention responses. Both streams are optimized independently and combined at inference time to enable flexible deployment. We evaluate SyncBreaker in a white-box proactive protection setting. Extensive experiments demonstrate that SyncBreaker more effectively degrades lip synchronization and facial dynamics than strong single-modality baselines, while preserving input perceptual quality and remaining robust under purification. Code: this https URL.
>
---
#### [replaced 065] TurPy: a physics-based and differentiable optical turbulence simulator for algorithmic development and system optimization
- **分类: physics.optics; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07248](https://arxiv.org/pdf/2604.07248)**

> **作者:** Joseph L. Greene; Alfred Moore; Iris Ochoa; Emily Kwan; Patrick Marano; Christopher R. Valenta
>
> **备注:** 19 pages, 7 figures, 1 table. Presented at 2026 SPIE DS Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications IV
>
> **摘要:** Developing optical systems for free-space applications requires simulation tools that accurately capture turbulence-induced wavefront distortions and support gradient-based optimization. Here we introduce TurPy, a GPU-accelerated, fully differentiable wave optics turbulence simulator to bridge high fidelity simulation with end-to-end optical system design. TurPy incorporates subharmonic phase screen generation, autoregressive temporal evolution, and an automated screen placement routine balancing Fourier aliasing constraints and weak-turbulence approximations into a unified, user-ready framework. Because TurPy's phase screen generation is parameterized through a media-specific power spectral density, the framework extends to atmospheric, oceanic, and biological propagation environments with minimal modification. We validate TurPy against established atmospheric turbulence theory by matching 2nd order Gaussian beam broadening and 4th order plane wave scintillation to closed-form models with 98% accuracy across weak to strong turbulence regimes, requiring only the medium's refractive index structure constant and power spectral density as inputs. To demonstrate TurPy as a gradient-based training platform, we optimize a dual-domain diffractive deep neural network (D2NN) in a two-mask dual-domain architecture to recover a Gaussian beam from a weakly turbulent path and achieving over 20x reduction in scintillation relative to an uncompensated receiver in simulation. TurPy is released as an open-source package to support synthetic data generation, turbulence-informed algorithm development, and the end-to-end design of optical platforms operating in turbulent environments.
>
---
#### [replaced 066] CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出CrashSight，一个用于交通碰撞场景理解的视觉语言基准，解决自动驾驶中基础设施视角下的事故分析问题。通过真实路侧视频数据，评估模型在时间与因果推理上的能力。**

- **链接: [https://arxiv.org/pdf/2604.08457](https://arxiv.org/pdf/2604.08457)**

> **作者:** Rui Gan; Junyi Ma; Pei Li; Xingyou Yang; Kai Chen; Sikai Chen; Bin Ran
>
> **摘要:** Cooperative autonomous driving requires traffic scene understanding from both vehicle and infrastructure perspectives. While vision-language models (VLMs) show strong general reasoning capabilities, their performance in safety-critical traffic scenarios remains insufficiently evaluated due to the ego-vehicle focus of existing benchmarks. To bridge this gap, we present \textbf{CrashSight}, a large-scale vision-language benchmark for roadway crash understanding using real-world roadside camera data. The dataset comprises 250 crash videos, annotated with 13K multiple-choice question-answer pairs organized under a two-tier taxonomy. Tier 1 evaluates the visual grounding of scene context and involved parties, while Tier 2 probes higher-level reasoning, including crash mechanics, causal attribution, temporal progression, and post-crash outcomes. We benchmark 8 state-of-the-art VLMs and show that, despite strong scene description capabilities, current models struggle with temporal and causal reasoning in safety-critical scenarios. We provide a detailed analysis of failure scenarios and discuss directions for improving VLM crash understanding. The benchmark provides a standardized evaluation framework for infrastructure-assisted perception in cooperative autonomous driving. The CrashSight benchmark, including the full dataset and code, is accessible at this https URL.
>
---
#### [replaced 067] Better Eyes, Better Thoughts: Why Vision Chain-of-Thought Fails in Medicine
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.06665](https://arxiv.org/pdf/2603.06665)**

> **作者:** Yuan Wu; Zongxian Yang; Jiayu Qian; Songpan Gao; Guanxing Chen; Qiankun Li; Yu-An Huang; Zhi-An Huang
>
> **摘要:** Large vision-language models (VLMs) often benefit from chain-of-thought (CoT) prompting in general domains, yet its efficacy in medical vision-language tasks remains underexplored. We report a counter-intuitive trend: on medical visual question answering, CoT frequently underperforms direct answering (DirA) across general-purpose and medical-specific models. We attribute this to a \emph{medical perception bottleneck}: subtle, domain-specific cues can weaken visual grounding, and CoT may compound early perceptual uncertainty rather than correct it. To probe this hypothesis, we introduce two training-free, inference-time grounding interventions: (i) \emph{perception anchoring} via region-of-interest cues and (ii) \emph{description grounding} via high-quality textual guidance. Across multiple benchmarks and model families, these interventions improve accuracy, mitigate CoT degradation, and in several settings reverse the CoT--DirA inversion. Our findings suggest that reliable clinical VLMs require robust visual grounding and cross-modal alignment, beyond extending text-driven reasoning chains. Code is available \href{this https URL}{here}.
>
---
#### [replaced 068] Bharat Scene Text: A Novel Comprehensive Dataset and Benchmark for Indian Language Scene Text Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于印度语言场景文本识别任务，针对数据不足和模型缺失的问题，构建了Bharat Scene Text数据集，涵盖11种印度语言，支持多种文本处理任务。**

- **链接: [https://arxiv.org/pdf/2511.23071](https://arxiv.org/pdf/2511.23071)**

> **作者:** Anik De; Abhirama Subramanyam Penamakuri; Rajeev Yadav; Aditya Rathore; Harshiv Shah; Devesh Sharma; Sagar Agarwal; Pravin Kumar; Anand Mishra
>
> **备注:** Accepted in International Journal on Document Analysis and Recognition (IJDAR)
>
> **摘要:** Reading scene text, that is, text appearing in images, has numerous application areas, including assistive technology, search, and e-commerce. Although scene text recognition in English has advanced significantly and is often considered nearly a solved problem, Indian language scene text recognition remains an open challenge. This is due to script diversity, non-standard fonts, and varying writing styles, and, more importantly, the lack of high-quality datasets and open-source models. To address these gaps, we introduce the Bharat Scene Text Dataset (BSTD) - a large-scale and comprehensive benchmark for studying Indian Language Scene Text Recognition. It comprises more than 100K words that span 11 Indian languages and English, sourced from over 6,500 scene images captured across various linguistic regions of India. The dataset is meticulously annotated and supports multiple scene text tasks, including: (i) Scene Text Detection, (ii) Script Identification, (iii) Cropped Word Recognition, and (iv) End-to-End Scene Text Recognition. We evaluated state-of-the-art models originally developed for English by adapting (fine-tuning) them for Indian languages. Our results highlight the challenges and opportunities in Indian language scene text recognition. We believe that this dataset represents a significant step toward advancing research in this domain. All our models and data are open source.
>
---
#### [replaced 069] HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.06165](https://arxiv.org/pdf/2604.06165)**

> **作者:** Reihaneh Zohrabi; Hosein Hasani; Akshita Gupta; Mahdieh Soleymani Baghshah; Anna Rohrbach; Marcus Rohrbach
>
> **摘要:** Large vision-language models can produce object hallucinations in image descriptions, highlighting the need for effective detection and mitigation strategies. Prior work commonly relies on the model's attention weights on visual tokens as a detection signal. We reveal that coarse-grained attention-based analysis is unreliable due to hidden confounders, specifically token position and object repetition in a description. This leads to Simpson's paradox: the attention trends reverse or disappear when statistics are aggregated. Based on this observation, we introduce HaloProbe, a Bayesian framework that factorizes external description statistics and internal decoding signals to estimate token-level hallucination probabilities. HaloProbe uses balanced training to isolate internal evidence and combines it with a learned prior over external features to recover the true posterior. While intervention-based mitigation methods often degrade utility or fluency by modifying models' internals, we use HaloProbe as an external scoring signal for non-invasive mitigation. Our experiments show that HaloProbe-guided decoding reduces hallucinations more effectively than state-of-the-art intervention-based methods while preserving utility.
>
---
#### [replaced 070] Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.18600](https://arxiv.org/pdf/2505.18600)**

> **作者:** Bryan Sangwoo Kim; Jeongsol Kim; Jong Chul Ye
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Modern single-image super-resolution (SISR) models deliver photo-realistic results at the scale factors on which they are trained, but collapse when asked to magnify far beyond that regime. We address this scalability bottleneck with Chain-of-Zoom (CoZ), a model-agnostic framework that factorizes SISR into an autoregressive chain of intermediate scale-states with multi-scale-aware prompts. CoZ repeatedly re-uses a backbone SR model, decomposing the conditional probability into tractable sub-problems to achieve extreme resolutions without additional training. Because visual cues diminish at high magnifications, we augment each zoom step with multi-scale-aware text prompts generated by a vision-language model (VLM). The prompt extractor itself is fine-tuned using Generalized Reward Policy Optimization (GRPO) with a critic VLM, aligning text guidance towards human preference. Experiments show that a standard 4x diffusion SR model wrapped in CoZ attains beyond 256x enlargement with high perceptual quality and fidelity. Project Page: this https URL.
>
---
#### [replaced 071] Tiled Prompts: Overcoming Prompt Misguidance in Image and Video Super-Resolution
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.03342](https://arxiv.org/pdf/2602.03342)**

> **作者:** Bryan Sangwoo Kim; Jonghyun Park; Jong Chul Ye
>
> **备注:** 29 pages, 8 figures
>
> **摘要:** Text-conditioned diffusion models have advanced image and video super-resolution by using prompts as semantic priors, and modern super-resolution pipelines typically rely on latent tiling to scale to high resolutions. In practice, a single global caption is used with the latent tiling, often causing prompt misguidance. Specifically, a coarse global prompt often misses localized details (errors of omission) and provides locally irrelevant guidance (errors of commission) which leads to substandard results at the tile level. To solve this, we propose Tiled Prompts, a unified framework for image and video super-resolution that generates a tile-specific prompt for each latent tile and performs super-resolution under locally text-conditioned posteriors to resolve prompt misguidance with minimal overhead. Our experiments on high resolution real-world images and videos show that tiled prompts bring consistent gains in perceptual quality and fidelity, while reducing hallucinations and tile-level artifacts that can be found in global-prompt baselines. Project Page: this https URL.
>
---
#### [replaced 072] How Noise Benefits AI-generated Image Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16136](https://arxiv.org/pdf/2511.16136)**

> **作者:** Ziqiang Li; Jiazhen Yan; Fan Wang; Kai Zeng; Zhangjie Fu
>
> **摘要:** The rapid advancement of generative models has made real and synthetic images increasingly indistinguishable. Although extensive efforts have been devoted to detecting AI-generated images, out-of-distribution generalization remains a persistent challenge. We trace this weakness to spurious shortcuts exploited during training and we also observe that small feature-space perturbations can mitigate shortcut dominance. To address this problem in a more controllable manner, we propose the Positive-Incentive Noise for CLIP (PiN-CLIP), which jointly trains a noise generator and a detection network under a variational positive-incentive principle. Specifically, we construct positive-incentive noise in the feature space via cross-attention fusion of visual and categorical semantic features. During optimization, the noise is injected into the feature space to fine-tune the visual encoder, suppressing shortcut-sensitive directions while amplifying stable forensic cues, thereby enabling the extraction of more robust and generalized artifact representations. Comparative experiments are conducted on an open-world dataset comprising synthetic images generated by 42 distinct generative models. Our method achieves new state-of-the-art performance, with notable improvements of 5.4 in average accuracy over existing approaches.
>
---
#### [replaced 073] Another BRIXEL in the Wall: Towards Cheaper Dense Features
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.05168](https://arxiv.org/pdf/2511.05168)**

> **作者:** Alexander Lappe; Martin A. Giese
>
> **摘要:** Vision foundation models achieve strong performance on both global and locally dense downstream tasks. Pretrained on large images, the recent DINOv3 model family is able to produce very fine-grained dense feature maps, enabling state-of-the-art performance. However, computing these feature maps requires the input image to be available at very high resolution, as well as large amounts of compute due to the squared complexity of the transformer architecture. To address these issues, we propose BRIXEL, a simple knowledge distillation approach that has the student learn to reproduce its own feature maps at higher resolution. Despite its simplicity, BRIXEL outperforms the baseline DINOv3 models by large margins on downstream tasks when the resolution is kept fixed. We also apply BRIXEL to other recent dense-feature extractors and show that it yields substantial performance gains across model families. Code and model weights are available at this https URL.
>
---
#### [replaced 074] OV-Stitcher: A Global Context-Aware Framework for Training-Free Open-Vocabulary Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.08110](https://arxiv.org/pdf/2604.08110)**

> **作者:** Seungjae Moon; Seunghyun Oh; Youngmin Ro
>
> **摘要:** Training-free open-vocabulary semantic segmentation(TF-OVSS) has recently attracted attention for its ability to perform dense prediction by leveraging the pretrained knowledge of large vision and vision-language models, without requiring additional training. However, due to the limited input resolution of these pretrained encoders, existing TF-OVSS methods commonly adopt a sliding-window strategy that processes cropped sub-images independently. While effective for managing high-resolution inputs, this approach prevents global attention over the full image, leading to fragmented feature representations and limited contextual reasoning. We propose OV-Stitcher, a training-free framework that addresses this limitation by stitching fragmented sub-image features directly within the final encoder block. By reconstructing attention representations from fragmented sub-image features, OV-Stitcher enables global attention within the final encoder block, producing coherent context aggregation and spatially consistent, semantically aligned segmentation maps. Extensive evaluations across eight benchmarks demonstrate that OV-Stitcher establishes a scalable and effective solution for open-vocabulary segmentation, achieving a notable improvement in mean Intersection over Union(mIoU) from 48.7 to 50.7 compared with prior training-free baselines.
>
---
#### [replaced 075] Descriptor: Parasitoid Wasps and Associated Hymenoptera Dataset (DAPWH)
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20028](https://arxiv.org/pdf/2602.20028)**

> **作者:** Joao Manoel Herrera Pinheiro; Gabriela Do Nascimento Herrera; Luciana Bueno Dos Reis Fernandes; Alvaro Doria Dos Santos; Ricardo V. Godoy; Eduardo A. B. Almeida; Helena Carolina Onody; Marcelo Andrade Da Costa Vieira; Angelica Maria Penteado-Dias; Marcelo Becker
>
> **摘要:** Accurate taxonomic identification is the cornerstone of biodiversity monitoring and agricultural management, particularly for the hyper-diverse superfamily Ichneumonoidea. Comprising the families Ichneumonidae and Braconidae, these parasitoid wasps are ecologically critical for regulating insect populations, yet they remain one of the most taxonomically challenging groups due to their cryptic morphology and vast number of undescribed species. To address the scarcity of robust digital resources for these key groups, we present a curated image dataset designed to advance automated identification systems. The dataset contains 3,556 high-resolution images, primarily focused on Neotropical Ichneumonidae and Braconidae, while also including supplementary families such as Andrenidae, Apidae, Bethylidae, Chrysididae, Colletidae, Halictidae, Megachilidae, Pompilidae, and Vespidae to improve model robustness. Crucially, a subset of 1,739 images is annotated in COCO format, featuring multi-class bounding boxes for the full insect body, wing venation, and scale bars. This resource provides a foundation for developing computer vision models capable of identifying these families.
>
---
#### [replaced 076] SimScale: Learning to Drive via Real-World Simulation at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决真实数据不足问题。通过SimScale框架生成大量模拟数据，并结合真实数据训练，提升模型鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.23369](https://arxiv.org/pdf/2511.23369)**

> **作者:** Haochen Tian; Tianyu Li; Haochen Liu; Jiazhi Yang; Yihang Qiu; Guang Li; Junli Wang; Yinfeng Gao; Zhang Zhang; Liang Wang; Hangjun Ye; Tieniu Tan; Long Chen; Hongyang Li
>
> **备注:** CVPR 2026 Oral. Project page: this https URL
>
> **摘要:** Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Simulation data and code have been released at this https URL.
>
---
#### [replaced 077] R3PM-Net: Real-time, Robust, Real-world Point Matching Network
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.05060](https://arxiv.org/pdf/2604.05060)**

> **作者:** Yasaman Kashefbahrami; Erkut Akdag; Panagiotis Meletis; Evgeniya Balmashnova; Dip Goswami; Egor Bondarau
>
> **备注:** Accepted to CVPRw 2026 (Oral), Code and datasets at this https URL
>
> **摘要:** Accurate Point Cloud Registration (PCR) is an important task in 3D data processing, involving the estimation of a rigid transformation between two point clouds. While deep-learning methods have addressed key limitations of traditional non-learning approaches, such as sensitivity to noise, outliers, occlusion, and initialization, they are developed and evaluated on clean, dense, synthetic datasets (limiting their generalizability to real-world industrial scenarios). This paper introduces R3PM-Net, a lightweight, global-aware, object-level point matching network designed to bridge this gap by prioritizing both generalizability and real-time efficiency. To support this transition, two datasets, Sioux-Cranfield and Sioux-Scans, are proposed. They provide an evaluation ground for registering imperfect photogrammetric and event-camera scans to digital CAD models, and have been made publicly available. Extensive experiments demonstrate that R3PM-Net achieves competitive accuracy with unmatched speed. On ModelNet40, it reaches a perfect fitness score of $1$ and inlier RMSE of $0.029$ cm in only $0.007$s, approximately 7 times faster than the state-of-the-art method RegTR. This performance carries over to the Sioux-Cranfield dataset, maintaining a fitness of $1$ and inlier RMSE of $0.030$ cm with similarly low latency. Furthermore, on the highly challenging Sioux-Scans dataset, R3PM-Net successfully resolves edge cases in under 50 ms. These results confirm that R3PM-Net offers a robust, high-speed solution for critical industrial applications, where precision and real-time performance are indispensable. The code and datasets are available at this https URL.
>
---
#### [replaced 078] UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉跟踪任务，解决动态环境中目标跟踪的精度与效率问题。提出UAV-Track VLA模型，提升长距离跟踪性能和实时性。**

- **链接: [https://arxiv.org/pdf/2604.02241](https://arxiv.org/pdf/2604.02241)**

> **作者:** Qiyao Zhang; Shuhua Zheng; Jianli Sun; Chengxiang Li; Xianke Wu; Zihan Song; Zhiyong Cui; Yisheng Lv; Yonglin Tian
>
> **摘要:** Embodied visual tracking is crucial for Unmanned Aerial Vehicles (UAVs) executing complex real-world tasks. In dynamic urban scenarios with complex semantic requirements, Vision-Language-Action (VLA) models show great promise due to their cross-modal fusion and continuous action generation capabilities. To benchmark multimodal tracking in such environments, we construct a dedicated evaluation benchmark and a large-scale dataset encompassing over 890K frames, 176 tasks, and 85 diverse objects. Furthermore, to address temporal feature redundancy and the lack of spatial geometric priors in existing VLA models, we propose an improved VLA tracking model, UAV-Track VLA. Built upon the $\pi_{0.5}$ architecture, our model introduces a temporal compression net to efficiently capture inter-frame dynamics. Additionally, a parallel dual-branch decoder comprising a spatial-aware auxiliary grounding head and a flow matching action expert is designed to decouple cross-modal features and generate fine-grained continuous actions. Systematic experiments in the CARLA simulator validate the superior end-to-end performance of our method. Notably, in challenging long-distance pedestrian tracking tasks, UAV-Track VLA achieves a 61.76\% success rate and 269.65 average tracking frames, significantly outperforming existing baselines. Furthermore, it demonstrates robust zero-shot generalization in unseen environments and reduces single-step inference latency by 33.4\% (to 0.0571s) compared to the original $\pi_{0.5}$, enabling highly efficient, real-time UAV control. Data samples and demonstration videos are available at: this https URL.
>
---
