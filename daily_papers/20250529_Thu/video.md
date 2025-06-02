# 计算机视觉 cs.CV

- **最新发布 206 篇**

- **更新 91 篇**

## 最新发布

#### [new 001] SridBench: Benchmark of Scientific Research Illustration Drawing of Image Generation Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SridBench，首个科学插图生成基准测试，解决AI缺乏科学图生成评估标准的问题。通过收集13学科1120个实例，从语义、结构等六维度评估模型，发现顶级模型在准确性和清晰度上仍显著低于人类，强调需提升推理驱动的生成能力。**

- **链接: [http://arxiv.org/pdf/2505.22126v1](http://arxiv.org/pdf/2505.22126v1)**

> **作者:** Yifan Chang; Yukang Feng; Jianwen Sun; Jiaxin Ai; Chuanhao Li; S. Kevin Zhou; Kaipeng Zhang
>
> **摘要:** Recent years have seen rapid advances in AI-driven image generation. Early diffusion models emphasized perceptual quality, while newer multimodal models like GPT-4o-image integrate high-level reasoning, improving semantic understanding and structural composition. Scientific illustration generation exemplifies this evolution: unlike general image synthesis, it demands accurate interpretation of technical content and transformation of abstract ideas into clear, standardized visuals. This task is significantly more knowledge-intensive and laborious, often requiring hours of manual work and specialized tools. Automating it in a controllable, intelligent manner would provide substantial practical value. Yet, no benchmark currently exists to evaluate AI on this front. To fill this gap, we introduce SridBench, the first benchmark for scientific figure generation. It comprises 1,120 instances curated from leading scientific papers across 13 natural and computer science disciplines, collected via human experts and MLLMs. Each sample is evaluated along six dimensions, including semantic fidelity and structural accuracy. Experimental results reveal that even top-tier models like GPT-4o-image lag behind human performance, with common issues in text/visual clarity and scientific correctness. These findings highlight the need for more advanced reasoning-driven visual generation capabilities.
>
---
#### [new 002] Do We Need All the Synthetic Data? Towards Targeted Synthetic Image Augmentation via Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对图像分类任务中合成数据增强效率低的问题，提出通过扩散模型对训练早期未充分学习的数据进行针对性增强，促进特征学习速度均匀性，提升模型泛化。实验表明仅增强30%-40%数据即可提升2.8%性能，优于现有方法且兼容其他增强策略。**

- **链接: [http://arxiv.org/pdf/2505.21574v1](http://arxiv.org/pdf/2505.21574v1)**

> **作者:** Dang Nguyen; Jiping Li; Jinghao Zheng; Baharan Mirzasoleiman
>
> **摘要:** Synthetically augmenting training datasets with diffusion models has been an effective strategy for improving generalization of image classifiers. However, existing techniques struggle to ensure the diversity of generation and increase the size of the data by up to 10-30x to improve the in-distribution performance. In this work, we show that synthetically augmenting part of the data that is not learned early in training outperforms augmenting the entire dataset. By analyzing a two-layer CNN, we prove that this strategy improves generalization by promoting homogeneity in feature learning speed without amplifying noise. Our extensive experiments show that by augmenting only 30%-40% of the data, our method boosts the performance by up to 2.8% in a variety of scenarios, including training ResNet, ViT and DenseNet on CIFAR-10, CIFAR-100, and TinyImageNet, with a range of optimizers including SGD and SAM. Notably, our method applied with SGD outperforms the SOTA optimizer, SAM, on CIFAR-100 and TinyImageNet. It can also easily stack with existing weak and strong augmentation strategies to further boost the performance.
>
---
#### [new 003] Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation
- **分类: cs.CV**

- **简介: 该论文提出多人员对话视频生成任务，解决现有方法在多人音频绑定错误及指令遵循不足的问题。通过MultiTalk框架，采用Label Rotary Position Embedding（L-RoPE）解决音频-人物绑定，结合部分参数与多任务训练策略，实现高质量多人对话视频生成，在多个数据集表现优异。**

- **链接: [http://arxiv.org/pdf/2505.22647v1](http://arxiv.org/pdf/2505.22647v1)**

> **作者:** Zhe Kong; Feng Gao; Yong Zhang; Zhuoliang Kang; Xiaoming Wei; Xunliang Cai; Guanying Chen; Wenhan Luo
>
> **备注:** Homepage: https://meigen-ai.github.io/multi-talk Github: https://github.com/MeiGen-AI/MultiTalk
>
> **摘要:** Audio-driven human animation methods, such as talking head and talking body generation, have made remarkable progress in generating synchronized facial movements and appealing visual quality videos. However, existing methods primarily focus on single human animation and struggle with multi-stream audio inputs, facing incorrect binding problems between audio and persons. Additionally, they exhibit limitations in instruction-following capabilities. To solve this problem, in this paper, we propose a novel task: Multi-Person Conversational Video Generation, and introduce a new framework, MultiTalk, to address the challenges during multi-person generation. Specifically, for audio injection, we investigate several schemes and propose the Label Rotary Position Embedding (L-RoPE) method to resolve the audio and person binding problem. Furthermore, during training, we observe that partial parameter training and multi-task training are crucial for preserving the instruction-following ability of the base model. MultiTalk achieves superior performance compared to other methods on several datasets, including talking head, talking body, and multi-person datasets, demonstrating the powerful generation capabilities of our approach.
>
---
#### [new 004] Diffusion Model-based Activity Completion for AI Motion Capture from Videos
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出基于扩散模型的AI运动捕捉补全方法（MDC-Net），解决传统AI动作捕捉仅依赖观察视频、无法生成未预定义动作的问题。通过引入门控模块和位置-时间嵌入模块，生成连贯动作序列，在Human3.6M数据集上表现优于现有方法，模型更轻量且生成效果更自然。**

- **链接: [http://arxiv.org/pdf/2505.21566v1](http://arxiv.org/pdf/2505.21566v1)**

> **作者:** Gao Huayu; Huang Tengjiu; Ye Xiaolong; Tsuyoshi Okita
>
> **备注:** 32 pages, 16 figures
>
> **摘要:** AI-based motion capture is an emerging technology that offers a cost-effective alternative to traditional motion capture systems. However, current AI motion capture methods rely entirely on observed video sequences, similar to conventional motion capture. This means that all human actions must be predefined, and movements outside the observed sequences are not possible. To address this limitation, we aim to apply AI motion capture to virtual humans, where flexible actions beyond the observed sequences are required. We assume that while many action fragments exist in the training data, the transitions between them may be missing. To bridge these gaps, we propose a diffusion-model-based action completion technique that generates complementary human motion sequences, ensuring smooth and continuous movements. By introducing a gate module and a position-time embedding module, our approach achieves competitive results on the Human3.6M dataset. Our experimental results show that (1) MDC-Net outperforms existing methods in ADE, FDE, and MMADE but is slightly less accurate in MMFDE, (2) MDC-Net has a smaller model size (16.84M) compared to HumanMAC (28.40M), and (3) MDC-Net generates more natural and coherent motion sequences. Additionally, we propose a method for extracting sensor data, including acceleration and angular velocity, from human motion sequences.
>
---
#### [new 005] LiDARDustX: A LiDAR Dataset for Dusty Unstructured Road Environments
- **分类: cs.CV**

- **简介: 该论文提出LiDARDustX数据集，针对高粉尘不规则道路环境的感知任务。解决现有数据集缺乏此类场景的问题，通过6种LiDAR采集3万帧数据（80%含粉尘），提供3D标注和语义分割，建立算法评估基准并分析粉尘影响。**

- **链接: [http://arxiv.org/pdf/2505.21914v1](http://arxiv.org/pdf/2505.21914v1)**

> **作者:** Chenfeng Wei; Qi Wu; Si Zuo; Jiahua Xu; Boyang Zhao; Zeyu Yang; Guotao Xie; Shenhong Wang
>
> **摘要:** Autonomous driving datasets are essential for validating the progress of intelligent vehicle algorithms, which include localization, perception, and prediction. However, existing datasets are predominantly focused on structured urban environments, which limits the exploration of unstructured and specialized scenarios, particularly those characterized by significant dust levels. This paper introduces the LiDARDustX dataset, which is specifically designed for perception tasks under high-dust conditions, such as those encountered in mining areas. The LiDARDustX dataset consists of 30,000 LiDAR frames captured by six different LiDAR sensors, each accompanied by 3D bounding box annotations and point cloud semantic segmentation. Notably, over 80% of the dataset comprises dust-affected scenes. By utilizing this dataset, we have established a benchmark for evaluating the performance of state-of-the-art 3D detection and segmentation algorithms. Additionally, we have analyzed the impact of dust on perception accuracy and delved into the causes of these effects. The data and further information can be accessed at: https://github.com/vincentweikey/LiDARDustX.
>
---
#### [new 006] Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation
- **分类: cs.CV**

- **简介: 该论文提出DiffPhy框架，通过结合LLM生成物理上下文指导视频扩散模型，解决生成视频中物理效果不准确的问题。任务为物理正确的逼真视频生成，方法包括利用多模态LLM设计监督信号、新型训练目标及物理视频数据集微调模型。**

- **链接: [http://arxiv.org/pdf/2505.21653v1](http://arxiv.org/pdf/2505.21653v1)**

> **作者:** Ke Zhang; Cihan Xiao; Yiqun Mei; Jiacong Xu; Vishal M. Patel
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Recent video diffusion models have demonstrated their great capability in generating visually-pleasing results, while synthesizing the correct physical effects in generated videos remains challenging. The complexity of real-world motions, interactions, and dynamics introduce great difficulties when learning physics from data. In this work, we propose DiffPhy, a generic framework that enables physically-correct and photo-realistic video generation by fine-tuning a pre-trained video diffusion model. Our method leverages large language models (LLMs) to explicitly reason a comprehensive physical context from the text prompt and use it to guide the generation. To incorporate physical context into the diffusion model, we leverage a Multimodal large language model (MLLM) as a supervisory signal and introduce a set of novel training objectives that jointly enforce physical correctness and semantic consistency with the input text. We also establish a high-quality physical video dataset containing diverse phyiscal actions and events to facilitate effective finetuning. Extensive experiments on public benchmarks demonstrate that DiffPhy is able to produce state-of-the-art results across diverse physics-related scenarios. Our project page is available at https://bwgzk-keke.github.io/DiffPhy/
>
---
#### [new 007] From Controlled Scenarios to Real-World: Cross-Domain Degradation Pattern Matching for All-in-One Image Restoration
- **分类: cs.CV**

- **简介: 该论文提出UDAIR框架，解决All-in-One图像修复（AiOIR）在跨域场景下的性能下降问题。通过离散嵌入编码学习退化模式，结合跨样本对比学习和动态领域自适应策略，缩小源域与目标域的数据分布差距，提升真实场景的泛化能力，实验显示其效果达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.22284v1](http://arxiv.org/pdf/2505.22284v1)**

> **作者:** Junyu Fan; Chuanlin Liao; Yi Lin
>
> **摘要:** As a fundamental imaging task, All-in-One Image Restoration (AiOIR) aims to achieve image restoration caused by multiple degradation patterns via a single model with unified parameters. Although existing AiOIR approaches obtain promising performance in closed and controlled scenarios, they still suffered from considerable performance reduction in real-world scenarios since the gap of data distributions between the training samples (source domain) and real-world test samples (target domain) can lead inferior degradation awareness ability. To address this issue, a Unified Domain-Adaptive Image Restoration (UDAIR) framework is proposed to effectively achieve AiOIR by leveraging the learned knowledge from source domain to target domain. To improve the degradation identification, a codebook is designed to learn a group of discrete embeddings to denote the degradation patterns, and the cross-sample contrastive learning mechanism is further proposed to capture shared features from different samples of certain degradation. To bridge the data gap, a domain adaptation strategy is proposed to build the feature projection between the source and target domains by dynamically aligning their codebook embeddings, and a correlation alignment-based test-time adaptation mechanism is designed to fine-tune the alignment discrepancies by tightening the degradation embeddings to the corresponding cluster center in the source domain. Experimental results on 10 open-source datasets demonstrate that UDAIR achieves new state-of-the-art performance for the AiOIR task. Most importantly, the feature cluster validate the degradation identification under unknown conditions, and qualitative comparisons showcase robust generalization to real-world scenarios.
>
---
#### [new 008] NFR: Neural Feature-Guided Non-Rigid Shape Registration
- **分类: cs.CV; cs.AI; I.4.m; I.2.6**

- **简介: 该论文提出NFR框架，解决非刚性形状配准任务，针对显著变形和部分形状匹配难题。通过融合神经特征与迭代几何配准，动态更新对应关系并过滤不一致匹配，无需标注实现高精度配准，在多个基准测试中达当前最优。**

- **链接: [http://arxiv.org/pdf/2505.22445v1](http://arxiv.org/pdf/2505.22445v1)**

> **作者:** Puhua Jiang; Zhangquan Chen; Mingze Sun; Ruqi Huang
>
> **备注:** 20 pages, 9 figures. arXiv admin note: substantial text overlap with arXiv:2311.04494
>
> **摘要:** In this paper, we propose a novel learning-based framework for 3D shape registration, which overcomes the challenges of significant non-rigid deformation and partiality undergoing among input shapes, and, remarkably, requires no correspondence annotation during training. Our key insight is to incorporate neural features learned by deep learning-based shape matching networks into an iterative, geometric shape registration pipeline. The advantage of our approach is two-fold -- On one hand, neural features provide more accurate and semantically meaningful correspondence estimation than spatial features (e.g., coordinates), which is critical in the presence of large non-rigid deformations; On the other hand, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching and partial shape matching across varying settings, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic deformations, in which case neither traditional registration methods nor intrinsic methods work.
>
---
#### [new 009] One-Way Ticket:Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本到图像扩散模型推理速度与质量的权衡问题，提出时间无关统一编码器TiUE。通过共享编码器特征至多时间步解码器，实现并行采样，减少计算；引入KL散度正则化提升生成图像的多样性和真实感，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21960v1](http://arxiv.org/pdf/2505.21960v1)**

> **作者:** Senmao Li; Lei Wang; Kai Wang; Tao Liu; Jiehang Xie; Joost van de Weijer; Fahad Shahbaz Khan; Shiqi Yang; Yaxing Wang; Jian Yang
>
> **备注:** Accepted at CVPR2025, Code: https://github.com/sen-mao/Loopfree
>
> **摘要:** Text-to-Image (T2I) diffusion models have made remarkable advancements in generative modeling; however, they face a trade-off between inference speed and image quality, posing challenges for efficient deployment. Existing distilled T2I models can generate high-fidelity images with fewer sampling steps, but often struggle with diversity and quality, especially in one-step models. From our analysis, we observe redundant computations in the UNet encoders. Our findings suggest that, for T2I diffusion models, decoders are more adept at capturing richer and more explicit semantic information, while encoders can be effectively shared across decoders from diverse time steps. Based on these observations, we introduce the first Time-independent Unified Encoder TiUE for the student model UNet architecture, which is a loop-free image generation approach for distilling T2I diffusion models. Using a one-pass scheme, TiUE shares encoder features across multiple decoder time steps, enabling parallel sampling and significantly reducing inference time complexity. In addition, we incorporate a KL divergence term to regularize noise prediction, which enhances the perceptual realism and diversity of the generated images. Experimental results demonstrate that TiUE outperforms state-of-the-art methods, including LCM, SD-Turbo, and SwiftBrushv2, producing more diverse and realistic results while maintaining the computational efficiency.
>
---
#### [new 010] EvidenceMoE: A Physics-Guided Mixture-of-Experts with Evidential Critics for Advancing Fluorescence Light Detection and Ranging in Scattering Media
- **分类: cs.CV; cs.AI; cs.LG; physics.optics**

- **简介: 该论文提出EvidenceMoE框架，解决荧光LiDAR在散射介质中难以分离光子飞行时间与荧光寿命的问题。其结合物理引导的专家模型（基于辐射输运方程）和证据批评者（EDC）评估可靠性，通过决策网络动态融合预测。在模拟组织数据中实现高精度深度（NRMSE 0.03）和荧光寿命（0.074）估计，提升非侵入性癌症检测性能。**

- **链接: [http://arxiv.org/pdf/2505.21532v1](http://arxiv.org/pdf/2505.21532v1)**

> **作者:** Ismail Erbas; Ferhat Demirkiran; Karthik Swaminathan; Naigang Wang; Navid Ibtehaj Nizam; Stefan T. Radev; Kaoutar El Maghraoui; Xavier Intes; Vikas Pandey
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Fluorescence LiDAR (FLiDAR), a Light Detection and Ranging (LiDAR) technology employed for distance and depth estimation across medical, automotive, and other fields, encounters significant computational challenges in scattering media. The complex nature of the acquired FLiDAR signal, particularly in such environments, makes isolating photon time-of-flight (related to target depth) and intrinsic fluorescence lifetime exceptionally difficult, thus limiting the effectiveness of current analytical and computational methodologies. To overcome this limitation, we present a Physics-Guided Mixture-of-Experts (MoE) framework tailored for specialized modeling of diverse temporal components. In contrast to the conventional MoE approaches our expert models are informed by underlying physics, such as the radiative transport equation governing photon propagation in scattering media. Central to our approach is EvidenceMoE, which integrates Evidence-Based Dirichlet Critics (EDCs). These critic models assess the reliability of each expert's output by providing per-expert quality scores and corrective feedback. A Decider Network then leverages this information to fuse expert predictions into a robust final estimate adaptively. We validate our method using realistically simulated Fluorescence LiDAR (FLiDAR) data for non-invasive cancer cell depth detection generated from photon transport models in tissue. Our framework demonstrates strong performance, achieving a normalized root mean squared error (NRMSE) of 0.030 for depth estimation and 0.074 for fluorescence lifetime.
>
---
#### [new 011] VME: A Satellite Imagery Dataset and Benchmark for Detecting Vehicles in the Middle East and Beyond
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦卫星图像车辆检测任务，针对现有数据集地理偏差导致中东地区检测效果差的问题，构建了包含12国54城的VME数据集（超4000图像、10万车辆）及全球基准CDSI，实验显示其显著提升中东及全球车辆检测精度。**

- **链接: [http://arxiv.org/pdf/2505.22353v1](http://arxiv.org/pdf/2505.22353v1)**

> **作者:** Noora Al-Emadi; Ingmar Weber; Yin Yang; Ferda Ofli
>
> **摘要:** Detecting vehicles in satellite images is crucial for traffic management, urban planning, and disaster response. However, current models struggle with real-world diversity, particularly across different regions. This challenge is amplified by geographic bias in existing datasets, which often focus on specific areas and overlook regions like the Middle East. To address this gap, we present the Vehicles in the Middle East (VME) dataset, designed explicitly for vehicle detection in high-resolution satellite images from Middle Eastern countries. Sourced from Maxar, the VME dataset spans 54 cities across 12 countries, comprising over 4,000 image tiles and more than 100,000 vehicles, annotated using both manual and semi-automated methods. Additionally, we introduce the largest benchmark dataset for Car Detection in Satellite Imagery (CDSI), combining images from multiple sources to enhance global car detection. Our experiments demonstrate that models trained on existing datasets perform poorly on Middle Eastern images, while the VME dataset significantly improves detection accuracy in this region. Moreover, state-of-the-art models trained on CDSI achieve substantial improvements in global car detection.
>
---
#### [new 012] Deep Learning-Based BMD Estimation from Radiographs with Conformal Uncertainty Quantification
- **分类: cs.CV; stat.AP**

- **简介: 该论文提出基于深度学习的膝关节X光骨密度（BMD）估计方法，解决DXA设备不足导致的骨质疏松筛查问题。采用EfficientNet模型，利用OAI数据集训练，对比传统与多样本TTA方法，并通过Split Conformal Prediction生成统计可靠预测区间。结果显示方法有效，为可信AI辅助BMD筛查奠定基础。（98字）**

- **链接: [http://arxiv.org/pdf/2505.22551v1](http://arxiv.org/pdf/2505.22551v1)**

> **作者:** Long Hui; Wai Lok Yeung
>
> **摘要:** Limited DXA access hinders osteoporosis screening. This proof-of-concept study proposes using widely available knee X-rays for opportunistic Bone Mineral Density (BMD) estimation via deep learning, emphasizing robust uncertainty quantification essential for clinical use. An EfficientNet model was trained on the OAI dataset to predict BMD from bilateral knee radiographs. Two Test-Time Augmentation (TTA) methods were compared: traditional averaging and a multi-sample approach. Crucially, Split Conformal Prediction was implemented to provide statistically rigorous, patient-specific prediction intervals with guaranteed coverage. Results showed a Pearson correlation of 0.68 (traditional TTA). While traditional TTA yielded better point predictions, the multi-sample approach produced slightly tighter confidence intervals (90%, 95%, 99%) while maintaining coverage. The framework appropriately expressed higher uncertainty for challenging cases. Although anatomical mismatch between knee X-rays and standard DXA limits immediate clinical use, this method establishes a foundation for trustworthy AI-assisted BMD screening using routine radiographs, potentially improving early osteoporosis detection.
>
---
#### [new 013] UAVPairs: A Challenging Benchmark for Match Pair Retrieval of Large-scale UAV Images
- **分类: cs.CV**

- **简介: 该论文提出UAVPairs数据集及训练流程，解决大规模无人机图像匹配检索任务。针对现有数据集缺乏真实匹配对及训练效率低的问题，构建21,622张图像的基准数据集，利用几何相似性定义有效匹配对；提出批次非trivial样本挖掘策略降低负例挖掘成本，并设计排名列表损失提升模型区分能力。实验表明方法显著提升检索精度和3D重建鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.22098v1](http://arxiv.org/pdf/2505.22098v1)**

> **作者:** Junhuan Liu; San Jiang; Wei Ge; Wei Huang; Bingxuan Guo; Qingquan Li
>
> **摘要:** The primary contribution of this paper is a challenging benchmark dataset, UAVPairs, and a training pipeline designed for match pair retrieval of large-scale UAV images. First, the UAVPairs dataset, comprising 21,622 high-resolution images across 30 diverse scenes, is constructed; the 3D points and tracks generated by SfM-based 3D reconstruction are employed to define the geometric similarity of image pairs, ensuring genuinely matchable image pairs are used for training. Second, to solve the problem of expensive mining cost for global hard negative mining, a batched nontrivial sample mining strategy is proposed, leveraging the geometric similarity and multi-scene structure of the UAVPairs to generate training samples as to accelerate training. Third, recognizing the limitation of pair-based losses, the ranked list loss is designed to improve the discrimination of image retrieval models, which optimizes the global similarity structure constructed from the positive set and negative set. Finally, the effectiveness of the UAVPairs dataset and training pipeline is validated through comprehensive experiments on three distinct large-scale UAV datasets. The experiment results demonstrate that models trained with the UAVPairs dataset and the ranked list loss achieve significantly improved retrieval accuracy compared to models trained on existing datasets or with conventional losses. Furthermore, these improvements translate to enhanced view graph connectivity and higher quality of reconstructed 3D models. The models trained by the proposed approach perform more robustly compared with hand-crafted global features, particularly in challenging repetitively textured scenes and weakly textured scenes. For match pair retrieval of large-scale UAV images, the trained image retrieval models offer an effective solution. The dataset would be made publicly available at https://github.com/json87/UAVPairs.
>
---
#### [new 014] Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对点云分类模型的对抗攻击评估，提出WAAttack和SubAttack方法，解决传统梯度攻击扰动过大和不自然的问题，通过加权梯度、自适应步长及关键区域聚焦提升攻击效果与隐蔽性。**

- **链接: [http://arxiv.org/pdf/2505.21854v1](http://arxiv.org/pdf/2505.21854v1)**

> **作者:** Jun Chen; Xinke Li; Mingyue Xu; Tianrui Li; Chongshou Li
>
> **摘要:** Gradient-based adversarial attacks have become a dominant approach for evaluating the robustness of point cloud classification models. However, existing methods often rely on uniform update rules that fail to consider the heterogeneous nature of point clouds, resulting in excessive and perceptible perturbations. In this paper, we rethink the design of gradient-based attacks by analyzing the limitations of conventional gradient update mechanisms and propose two new strategies to improve both attack effectiveness and imperceptibility. First, we introduce WAAttack, a novel framework that incorporates weighted gradients and an adaptive step-size strategy to account for the non-uniform contribution of points during optimization. This approach enables more targeted and subtle perturbations by dynamically adjusting updates according to the local structure and sensitivity of each point. Second, we propose SubAttack, a complementary strategy that decomposes the point cloud into subsets and focuses perturbation efforts on structurally critical regions. Together, these methods represent a principled rethinking of gradient-based adversarial attacks for 3D point cloud classification. Extensive experiments demonstrate that our approach outperforms state-of-the-art baselines in generating highly imperceptible adversarial examples. Code will be released upon paper acceptance.
>
---
#### [new 015] Learning World Models for Interactive Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于交互式视频生成任务，旨在解决现有模型在长期视频生成中因累积误差和记忆机制不足导致的时空不一致问题。提出视频检索增强生成（VRAG）方法，通过显式全局状态条件减少长期误差并提升世界模型的时空一致性，建立基准以改进具有内部世界建模能力的视频生成模型。**

- **链接: [http://arxiv.org/pdf/2505.21996v1](http://arxiv.org/pdf/2505.21996v1)**

> **作者:** Taiye Chen; Xun Hu; Zihan Ding; Chi Jin
>
> **摘要:** Foundational world models must be both interactive and preserve spatiotemporal coherence for effective future planning with action choices. However, present models for long video generation have limited inherent world modeling capabilities due to two main challenges: compounding errors and insufficient memory mechanisms. We enhance image-to-video models with interactive capabilities through additional action conditioning and autoregressive framework, and reveal that compounding error is inherently irreducible in autoregressive video generation, while insufficient memory mechanism leads to incoherence of world models. We propose video retrieval augmented generation (VRAG) with explicit global state conditioning, which significantly reduces long-term compounding errors and increases spatiotemporal consistency of world models. In contrast, naive autoregressive generation with extended context windows and retrieval-augmented generation prove less effective for video generation, primarily due to the limited in-context learning capabilities of current video models. Our work illuminates the fundamental challenges in video world models and establishes a comprehensive benchmark for improving video generation models with internal world modeling capabilities.
>
---
#### [new 016] Identity-Preserving Text-to-Image Generation via Dual-Level Feature Decoupling and Expert-Guided Fusion
- **分类: cs.CV**

- **简介: 该论文属于身份保持的文本到图像生成任务，旨在解决现有方法难以分离身份相关与无关特征导致生成图像身份失真或过拟合的问题。提出双阶段解耦模块（IEDM结合隐式特征解耦与显式前景-背景分离）和专家引导融合模块（FFM动态整合特征），并设计三类损失函数优化特征分离与融合，提升生成图像质量、场景适应性及多样性。**

- **链接: [http://arxiv.org/pdf/2505.22360v1](http://arxiv.org/pdf/2505.22360v1)**

> **作者:** Kewen Chen; Xiaobin Hu; Wenqi Ren
>
> **摘要:** Recent advances in large-scale text-to-image generation models have led to a surge in subject-driven text-to-image generation, which aims to produce customized images that align with textual descriptions while preserving the identity of specific subjects. Despite significant progress, current methods struggle to disentangle identity-relevant information from identity-irrelevant details in the input images, resulting in overfitting or failure to maintain subject identity. In this work, we propose a novel framework that improves the separation of identity-related and identity-unrelated features and introduces an innovative feature fusion mechanism to improve the quality and text alignment of generated images. Our framework consists of two key components: an Implicit-Explicit foreground-background Decoupling Module (IEDM) and a Feature Fusion Module (FFM) based on a Mixture of Experts (MoE). IEDM combines learnable adapters for implicit decoupling at the feature level with inpainting techniques for explicit foreground-background separation at the image level. FFM dynamically integrates identity-irrelevant features with identity-related features, enabling refined feature representations even in cases of incomplete decoupling. In addition, we introduce three complementary loss functions to guide the decoupling process. Extensive experiments demonstrate the effectiveness of our proposed method in enhancing image generation quality, improving flexibility in scene adaptation, and increasing the diversity of generated outputs across various textual descriptions.
>
---
#### [new 017] Corruption-Aware Training of Latent Video Diffusion Models for Robust Text-to-Video Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本到视频生成任务，旨在解决潜在视频扩散模型（LVDM）对噪声敏感导致的语义偏移和时间不连贯问题。提出CAT-LVDM框架，包含批内语义噪声注入（BCNI）和频谱感知噪声（SACN），通过结构化噪声提升模型鲁棒性，在多个数据集上显著降低视频质量评估指标。**

- **链接: [http://arxiv.org/pdf/2505.21545v1](http://arxiv.org/pdf/2505.21545v1)**

> **作者:** Chika Maduabuchi; Hao Chen; Yujin Han; Jindong Wang
>
> **备注:** Code: https://github.com/chikap421/catlvdm Models: https://huggingface.co/Chikap421/catlvdm-checkpoints/tree/main
>
> **摘要:** Latent Video Diffusion Models (LVDMs) achieve high-quality generation but are sensitive to imperfect conditioning, which causes semantic drift and temporal incoherence on noisy, web-scale video-text datasets. We introduce CAT-LVDM, the first corruption-aware training framework for LVDMs that improves robustness through structured, data-aligned noise injection. Our method includes Batch-Centered Noise Injection (BCNI), which perturbs embeddings along intra-batch semantic directions to preserve temporal consistency. BCNI is especially effective on caption-rich datasets like WebVid-2M, MSR-VTT, and MSVD. We also propose Spectrum-Aware Contextual Noise (SACN), which injects noise along dominant spectral directions to improve low-frequency smoothness, showing strong results on UCF-101. On average, BCNI reduces FVD by 31.9% across WebVid-2M, MSR-VTT, and MSVD, while SACN yields a 12.3% improvement on UCF-101. Ablation studies confirm the benefit of low-rank, data-aligned noise. Our theoretical analysis further explains how such perturbations tighten entropy, Wasserstein, score-drift, mixing-time, and generalization bounds. CAT-LVDM establishes a principled, scalable training approach for robust video diffusion under multimodal noise. Code and models: https://github.com/chikap421/catlvdm
>
---
#### [new 018] RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像重描述任务，旨在解决现有方法因幻觉和细节缺失导致的描述不准确和不完整问题。提出RICO框架，通过文本到图像模型重建图像并迭代修正描述，同时开发RICO-Flash以降低计算成本，实验显示其在准确性和完整性上超越基线约10%。**

- **链接: [http://arxiv.org/pdf/2505.22613v1](http://arxiv.org/pdf/2505.22613v1)**

> **作者:** Yuchi Wang; Yishuo Cai; Shuhuai Ren; Sihan Yang; Linli Yao; Yuanxin Liu; Yuanxing Zhang; Pengfei Wan; Xu Sun
>
> **备注:** code: https://github.com/wangyuchi369/RICO
>
> **摘要:** Image recaptioning is widely used to generate training datasets with enhanced quality for various multimodal tasks. Existing recaptioning methods typically rely on powerful multimodal large language models (MLLMs) to enhance textual descriptions, but often suffer from inaccuracies due to hallucinations and incompleteness caused by missing fine-grained details. To address these limitations, we propose RICO, a novel framework that refines captions through visual reconstruction. Specifically, we leverage a text-to-image model to reconstruct a caption into a reference image, and prompt an MLLM to identify discrepancies between the original and reconstructed images to refine the caption. This process is performed iteratively, further progressively promoting the generation of more faithful and comprehensive descriptions. To mitigate the additional computational cost induced by the iterative process, we introduce RICO-Flash, which learns to generate captions like RICO using DPO. Extensive experiments demonstrate that our approach significantly improves caption accuracy and completeness, outperforms most baselines by approximately 10% on both CapsBench and CompreCap. Code released at https://github.com/wangyuchi369/RICO.
>
---
#### [new 019] Learnable Burst-Encodable Time-of-Flight Imaging for High-Fidelity Long-Distance Depth Sensing
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出Burst-Encodable ToF（BE-ToF）方法，解决长距深度成像中相位包裹和低信噪比问题。通过burst脉冲模式避免相位折叠，结合端到端可学习框架优化编码与重建，并加入硬件约束项，实现实验验证的高精度远距成像。**

- **链接: [http://arxiv.org/pdf/2505.22025v1](http://arxiv.org/pdf/2505.22025v1)**

> **作者:** Manchao Bao; Shengjiang Fang; Tao Yue; Xuemei Hu
>
> **摘要:** Long-distance depth imaging holds great promise for applications such as autonomous driving and robotics. Direct time-of-flight (dToF) imaging offers high-precision, long-distance depth sensing, yet demands ultra-short pulse light sources and high-resolution time-to-digital converters. In contrast, indirect time-of-flight (iToF) imaging often suffers from phase wrapping and low signal-to-noise ratio (SNR) as the sensing distance increases. In this paper, we introduce a novel ToF imaging paradigm, termed Burst-Encodable Time-of-Flight (BE-ToF), which facilitates high-fidelity, long-distance depth imaging. Specifically, the BE-ToF system emits light pulses in burst mode and estimates the phase delay of the reflected signal over the entire burst period, thereby effectively avoiding the phase wrapping inherent to conventional iToF systems. Moreover, to address the low SNR caused by light attenuation over increasing distances, we propose an end-to-end learnable framework that jointly optimizes the coding functions and the depth reconstruction network. A specialized double well function and first-order difference term are incorporated into the framework to ensure the hardware implementability of the coding functions. The proposed approach is rigorously validated through comprehensive simulations and real-world prototype experiments, demonstrating its effectiveness and practical applicability.
>
---
#### [new 020] Concentrate on Weakness: Mining Hard Prototypes for Few-Shot Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于少样本医学图像分割任务，针对现有方法因随机采样导致边界模糊的问题，提出通过挖掘"弱特征"优化原型生成。方法包括：SSP模块识别关键弱特征，HPG模块生成硬原型，MSMF模块双路径融合相似性图，并引入边界损失约束边缘，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2505.21897v1](http://arxiv.org/pdf/2505.21897v1)**

> **作者:** Jianchao Jiang; Haofeng Zhang
>
> **备注:** 12 pages, 9 figures, 9 tables, accepted by IJCAI 2025
>
> **摘要:** Few-Shot Medical Image Segmentation (FSMIS) has been widely used to train a model that can perform segmentation from only a few annotated images. However, most existing prototype-based FSMIS methods generate multiple prototypes from the support image solely by random sampling or local averaging, which can cause particularly severe boundary blurring due to the tendency for normal features accounting for the majority of features of a specific category. Consequently, we propose to focus more attention to those weaker features that are crucial for clear segmentation boundary. Specifically, we design a Support Self-Prediction (SSP) module to identify such weak features by comparing true support mask with one predicted by global support prototype. Then, a Hard Prototypes Generation (HPG) module is employed to generate multiple hard prototypes based on these weak features. Subsequently, a Multiple Similarity Maps Fusion (MSMF) module is devised to generate final segmenting mask in a dual-path fashion to mitigate the imbalance between foreground and background in medical images. Furthermore, we introduce a boundary loss to further constraint the edge of segmentation. Extensive experiments on three publicly available medical image datasets demonstrate that our method achieves state-of-the-art performance. Code is available at https://github.com/jcjiang99/CoW.
>
---
#### [new 021] FaceEditTalker: Interactive Talking Head Generation with Facial Attribute Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于可控 talking head 生成任务，解决现有方法忽略面部属性编辑的问题。提出FaceEditTalker框架，通过图像特征编辑模块（控制表情、发型等）与音频驱动视频生成模块（融合特征与语音 landmark）的结合，在保证 lip-sync 和身份一致性的前提下实现面部属性可控编辑。**

- **链接: [http://arxiv.org/pdf/2505.22141v1](http://arxiv.org/pdf/2505.22141v1)**

> **作者:** Guanwen Feng; Zhiyuan Ma; Yunan Li; Junwei Jing; Jiahao Yang; Qiguang Miao
>
> **摘要:** Recent advances in audio-driven talking head generation have achieved impressive results in lip synchronization and emotional expression. However, they largely overlook the crucial task of facial attribute editing. This capability is crucial for achieving deep personalization and expanding the range of practical applications, including user-tailored digital avatars, engaging online education content, and brand-specific digital customer service. In these key domains, the flexible adjustment of visual attributes-such as hairstyle, accessories, and subtle facial features is essential for aligning with user preferences, reflecting diverse brand identities, and adapting to varying contextual demands. In this paper, we present FaceEditTalker, a unified framework that enables controllable facial attribute manipulation while generating high-quality, audio-synchronized talking head videos. Our method consists of two key components: an image feature space editing module, which extracts semantic and detail features and allows flexible control over attributes like expression, hairstyle, and accessories; and an audio-driven video generation module, which fuses these edited features with audio-guided facial landmarks to drive a diffusion-based generator. This design ensures temporal coherence, visual fidelity, and identity preservation across frames. Extensive experiments on public datasets demonstrate that our method outperforms state-of-the-art approaches in lip-sync accuracy, video quality, and attribute controllability. Project page: https://peterfanfan.github.io/FaceEditTalker/
>
---
#### [new 022] The Meeseeks Mesh: Spatially Consistent 3D Adversarial Objects for BEV Detector
- **分类: cs.CV**

- **简介: 该论文属于3D对抗攻击任务，针对BEV检测器脆弱性，提出生成空间一致的3D对抗对象。通过可微渲染建模空间关系、遮挡模块增强视觉一致性及BEV特征优化策略，解决模型对非侵入式攻击的鲁棒性评估问题。（98字）**

- **链接: [http://arxiv.org/pdf/2505.22499v1](http://arxiv.org/pdf/2505.22499v1)**

> **作者:** Aixuan Li; Mochu Xiang; Jing Zhang; Yuchao Dai
>
> **摘要:** 3D object detection is a critical component in autonomous driving systems. It allows real-time recognition and detection of vehicles, pedestrians and obstacles under varying environmental conditions. Among existing methods, 3D object detection in the Bird's Eye View (BEV) has emerged as the mainstream framework. To guarantee a safe, robust and trustworthy 3D object detection, 3D adversarial attacks are investigated, where attacks are placed in 3D environments to evaluate the model performance, e.g., putting a film on a car, clothing a pedestrian. The vulnerability of 3D object detection models to 3D adversarial attacks serves as an important indicator to evaluate the robustness of the model against perturbations. To investigate this vulnerability, we generate non-invasive 3D adversarial objects tailored for real-world attack scenarios. Our method verifies the existence of universal adversarial objects that are spatially consistent across time and camera views. Specifically, we employ differentiable rendering techniques to accurately model the spatial relationship between adversarial objects and the target vehicle. Furthermore, we introduce an occlusion-aware module to enhance visual consistency and realism under different viewpoints. To maintain attack effectiveness across multiple frames, we design a BEV spatial feature-guided optimization strategy. Experimental results demonstrate that our approach can reliably suppress vehicle predictions from state-of-the-art 3D object detectors, serving as an important tool to test robustness of 3D object detection models before deployment. Moreover, the generated adversarial objects exhibit strong generalization capabilities, retaining its effectiveness at various positions and distances in the scene.
>
---
#### [new 023] Prototype Embedding Optimization for Human-Object Interaction Detection in Livestreaming
- **分类: cs.CV**

- **简介: 该论文针对直播中人类-物体交互检测（HOI）的物体偏置问题，提出PeO-HOI方法。通过HO对特征提取、原型嵌入优化及时空建模，提升交互检测精度，在VidHOI和BJUT-HOI数据集上取得更高准确率。任务为HOI检测，解决物体主导导致的交互忽视，方法包含预处理、偏置优化与预测建模。**

- **链接: [http://arxiv.org/pdf/2505.22011v1](http://arxiv.org/pdf/2505.22011v1)**

> **作者:** Menghui Zhang; Jing Zhang; Lin Chen; Li Zhuo
>
> **摘要:** Livestreaming often involves interactions between streamers and objects, which is critical for understanding and regulating web content. While human-object interaction (HOI) detection has made some progress in general-purpose video downstream tasks, when applied to recognize the interaction behaviors between a streamer and different objects in livestreaming, it tends to focuses too much on the objects and neglects their interactions with the streamer, which leads to object bias. To solve this issue, we propose a prototype embedding optimization for human-object interaction detection (PeO-HOI). First, the livestreaming is preprocessed using object detection and tracking techniques to extract features of the human-object (HO) pairs. Then, prototype embedding optimization is adopted to mitigate the effect of object bias on HOI. Finally, after modelling the spatio-temporal context between HO pairs, the HOI detection results are obtained by the prediction head. The experimental results show that the detection accuracy of the proposed PeO-HOI method has detection accuracies of 37.19%@full, 51.42%@non-rare, 26.20%@rare on the publicly available dataset VidHOI, 45.13%@full, 62.78%@non-rare and 30.37%@rare on the self-built dataset BJUT-HOI, which effectively improves the HOI detection performance in livestreaming.
>
---
#### [new 024] Scaling-up Perceptual Video Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频质量评估（VQA）任务，旨在解决因标注数据稀缺和数据集规模不足导致的模型性能瓶颈。提出OmniVQA框架，构建了大规模多模态指令数据库OmniVQA-Chat-400K及定量评估数据集OmniVQA-MOS-20K，设计互补训练策略与细粒度评估基准，显著提升模型在质量理解和评分任务中的性能，达当前最优。**

- **链接: [http://arxiv.org/pdf/2505.22543v1](http://arxiv.org/pdf/2505.22543v1)**

> **作者:** Ziheng Jia; Zicheng Zhang; Zeyu Zhang; Yingji Liang; Xiaorong Zhu; Chunyi Li; Jinliang Han; Haoning Wu; Bin Wang; Haoran Zhang; Guanyu Zhu; Qiyong Zhao; Xiaohong Liu; Guangtao Zhai; Xiongkuo Min
>
> **摘要:** The data scaling law has been shown to significantly enhance the performance of large multi-modal models (LMMs) across various downstream tasks. However, in the domain of perceptual video quality assessment (VQA), the potential of scaling law remains unprecedented due to the scarcity of labeled resources and the insufficient scale of datasets. To address this, we propose \textbf{OmniVQA}, an efficient framework designed to efficiently build high-quality, human-in-the-loop VQA multi-modal instruction databases (MIDBs). We then scale up to create \textbf{OmniVQA-Chat-400K}, the largest MIDB in the VQA field concurrently. Our focus is on the technical and aesthetic quality dimensions, with abundant in-context instruction data to provide fine-grained VQA knowledge. Additionally, we have built the \textbf{OmniVQA-MOS-20K} dataset to enhance the model's quantitative quality rating capabilities. We then introduce a \textbf{complementary} training strategy that effectively leverages the knowledge from datasets for quality understanding and quality rating tasks. Furthermore, we propose the \textbf{OmniVQA-FG (fine-grain)-Benchmark} to evaluate the fine-grained performance of the models. Our results demonstrate that our models achieve state-of-the-art performance in both quality understanding and rating tasks.
>
---
#### [new 025] OmniResponse: Online Multimodal Conversational Response Generation in Dyadic Interactions
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文提出在线多模态对话响应生成（OMCRG）任务，旨在实时生成与说话者输入同步的听者语言及面部反馈。针对音频-视觉同步难题，提出多模态大模型OmniResponse，通过文本中间表征连接音频/面部生成，包含时间锚定文本模块Chrono-Text和可控语音合成模块TempoVoice，并构建含696个交互案例的ResponseNet数据集，实验验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.21724v1](http://arxiv.org/pdf/2505.21724v1)**

> **作者:** Cheng Luo; Jianghui Wang; Bing Li; Siyang Song; Bernard Ghanem
>
> **备注:** 23 pages, 9 figures
>
> **摘要:** In this paper, we introduce Online Multimodal Conversational Response Generation (OMCRG), a novel task that aims to online generate synchronized verbal and non-verbal listener feedback, conditioned on the speaker's multimodal input. OMCRG reflects natural dyadic interactions and poses new challenges in achieving synchronization between the generated audio and facial responses of the listener. To address these challenges, we innovatively introduce text as an intermediate modality to bridge the audio and facial responses. We hence propose OmniResponse, a Multimodal Large Language Model (MLLM) that autoregressively generates high-quality multi-modal listener responses. OmniResponse leverages a pretrained LLM enhanced with two novel components: Chrono-Text, which temporally anchors generated text tokens, and TempoVoice, a controllable online TTS module that produces speech synchronized with facial reactions. To support further OMCRG research, we present ResponseNet, a new dataset comprising 696 high-quality dyadic interactions featuring synchronized split-screen videos, multichannel audio, transcripts, and facial behavior annotations. Comprehensive evaluations conducted on ResponseNet demonstrate that OmniResponse significantly outperforms baseline models in terms of semantic speech content, audio-visual synchronization, and generation quality.
>
---
#### [new 026] Self-Reflective Reinforcement Learning for Diffusion-based Image Reasoning Generation
- **分类: cs.CV**

- **简介: 该论文属于扩散模型驱动的图像逻辑推理生成任务。针对现有方法在逻辑中心图像生成中推理能力不足的问题，提出SRRL算法：通过自反思强化学习，结合多轮反思去噪过程与条件引导前向过程，在扩散轨迹中引入类似LLM的CoT推理机制，实现符合物理规律的图像生成，实验显示其超越GPT-4o。**

- **链接: [http://arxiv.org/pdf/2505.22407v1](http://arxiv.org/pdf/2505.22407v1)**

> **作者:** Jiadong Pan; Zhiyuan Ma; Kaiyan Zhang; Ning Ding; Bowen Zhou
>
> **摘要:** Diffusion models have recently demonstrated exceptional performance in image generation task. However, existing image generation methods still significantly suffer from the dilemma of image reasoning, especially in logic-centered image generation tasks. Inspired by the success of Chain of Thought (CoT) and Reinforcement Learning (RL) in LLMs, we propose SRRL, a self-reflective RL algorithm for diffusion models to achieve reasoning generation of logical images by performing reflection and iteration across generation trajectories. The intermediate samples in the denoising process carry noise, making accurate reward evaluation difficult. To address this challenge, SRRL treats the entire denoising trajectory as a CoT step with multi-round reflective denoising process and introduces condition guided forward process, which allows for reflective iteration between CoT steps. Through SRRL-based iterative diffusion training, we introduce image reasoning through CoT into generation tasks adhering to physical laws and unconventional physical phenomena for the first time. Notably, experimental results of case study exhibit that the superior performance of our SRRL algorithm even compared with GPT-4o. The project page is https://jadenpan0.github.io/srrl.github.io/.
>
---
#### [new 027] Towards Scalable Language-Image Pre-training for 3D Medical Imaging
- **分类: cs.CV**

- **简介: 该论文提出HLIP框架，针对3D医学影像（CT/MRI）语言-图像预训练的高计算需求问题，通过分层注意力机制建模放射学数据层级（切片/扫描/研究），实现高效训练与泛化，在220K患者脑MRI和240K例头CT数据上达SOTA，验证了直接使用大规模未整理临床数据的可行性。**

- **链接: [http://arxiv.org/pdf/2505.21862v1](http://arxiv.org/pdf/2505.21862v1)**

> **作者:** Chenhui Zhao; Yiwei Lyu; Asadur Chowdury; Edward Harake; Akhil Kondepudi; Akshay Rao; Xinhai Hou; Honglak Lee; Todd Hollon
>
> **摘要:** Language-image pre-training has demonstrated strong performance in 2D medical imaging, but its success in 3D modalities such as CT and MRI remains limited due to the high computational demands of volumetric data, which pose a significant barrier to training on large-scale, uncurated clinical studies. In this study, we introduce Hierarchical attention for Language-Image Pre-training (HLIP), a scalable pre-training framework for 3D medical imaging. HLIP adopts a lightweight hierarchical attention mechanism inspired by the natural hierarchy of radiology data: slice, scan, and study. This mechanism exhibits strong generalizability, e.g., +4.3% macro AUC on the Rad-ChestCT benchmark when pre-trained on CT-RATE. Moreover, the computational efficiency of HLIP enables direct training on uncurated datasets. Trained on 220K patients with 3.13 million scans for brain MRI and 240K patients with 1.44 million scans for head CT, HLIP achieves state-of-the-art performance, e.g., +32.4% balanced ACC on the proposed publicly available brain MRI benchmark Pub-Brain-5; +1.4% and +6.9% macro AUC on head CT benchmarks RSNA and CQ500, respectively. These results demonstrate that, with HLIP, directly pre-training on uncurated clinical datasets is a scalable and effective direction for language-image pre-training in 3D medical imaging. The code is available at https://github.com/Zch0414/hlip
>
---
#### [new 028] Scalable Segmentation for Ultra-High-Resolution Brain MR Images
- **分类: cs.CV**

- **简介: 该论文针对超高分辨率脑MRI分割任务，解决标注数据不足与计算需求高的问题。提出利用低分辨率粗标注作为空间参考，通过回归符号距离图实现边界感知监督，并采用逐类条件分割策略，降低内存消耗并提升泛化性，实验证明其高效性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2505.21697v1](http://arxiv.org/pdf/2505.21697v1)**

> **作者:** Xiaoling Hu; Peirong Liu; Dina Zemlyanker; Jonathan Williams Ramirez; Oula Puonti; Juan Eugenio Iglesias
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Although deep learning has shown great success in 3D brain MRI segmentation, achieving accurate and efficient segmentation of ultra-high-resolution brain images remains challenging due to the lack of labeled training data for fine-scale anatomical structures and high computational demands. In this work, we propose a novel framework that leverages easily accessible, low-resolution coarse labels as spatial references and guidance, without incurring additional annotation cost. Instead of directly predicting discrete segmentation maps, our approach regresses per-class signed distance transform maps, enabling smooth, boundary-aware supervision. Furthermore, to enhance scalability, generalizability, and efficiency, we introduce a scalable class-conditional segmentation strategy, where the model learns to segment one class at a time conditioned on a class-specific input. This novel design not only reduces memory consumption during both training and testing, but also allows the model to generalize to unseen anatomical classes. We validate our method through comprehensive experiments on both synthetic and real-world datasets, demonstrating its superior performance and scalability compared to conventional segmentation approaches.
>
---
#### [new 029] PrismLayers: Open Data for High-Quality Multi-Layer Transparent Image Generative Models
- **分类: cs.CV**

- **简介: 该论文聚焦多层透明图像生成任务，针对缺乏高质量数据集的问题，提出三方面工作：发布含20万张带精准透明通道的PrismLayersPro数据集；设计无训练需求的合成pipeline；开源ART+模型，通过LayerFLUX生成单层图像并用MultiLayerFLUX合成，经人工筛选提升质量，性能超原有模型。**

- **链接: [http://arxiv.org/pdf/2505.22523v1](http://arxiv.org/pdf/2505.22523v1)**

> **作者:** Junwen Chen; Heyang Jiang; Yanbin Wang; Keming Wu; Ji Li; Chao Zhang; Keiji Yanai; Dong Chen; Yuhui Yuan
>
> **备注:** Homepage: https://prism-layers.github.io/
>
> **摘要:** Generating high-quality, multi-layer transparent images from text prompts can unlock a new level of creative control, allowing users to edit each layer as effortlessly as editing text outputs from LLMs. However, the development of multi-layer generative models lags behind that of conventional text-to-image models due to the absence of a large, high-quality corpus of multi-layer transparent data. In this paper, we address this fundamental challenge by: (i) releasing the first open, ultra-high-fidelity PrismLayers (PrismLayersPro) dataset of 200K (20K) multilayer transparent images with accurate alpha mattes, (ii) introducing a trainingfree synthesis pipeline that generates such data on demand using off-the-shelf diffusion models, and (iii) delivering a strong, open-source multi-layer generation model, ART+, which matches the aesthetics of modern text-to-image generation models. The key technical contributions include: LayerFLUX, which excels at generating high-quality single transparent layers with accurate alpha mattes, and MultiLayerFLUX, which composes multiple LayerFLUX outputs into complete images, guided by human-annotated semantic layout. To ensure higher quality, we apply a rigorous filtering stage to remove artifacts and semantic mismatches, followed by human selection. Fine-tuning the state-of-the-art ART model on our synthetic PrismLayersPro yields ART+, which outperforms the original ART in 60% of head-to-head user study comparisons and even matches the visual quality of images generated by the FLUX.1-[dev] model. We anticipate that our work will establish a solid dataset foundation for the multi-layer transparent image generation task, enabling research and applications that require precise, editable, and visually compelling layered imagery.
>
---
#### [new 030] FPAN: Mitigating Replication in Diffusion Models through the Fine-Grained Probabilistic Addition of Noise to Token Embeddings
- **分类: cs.CV**

- **简介: 该论文属于扩散模型隐私保护任务，旨在解决训练数据复制引发的隐私风险。提出FPAN方法，通过向token嵌入以概率方式添加更大噪声，减少复制（较基线降低28.78%），且不影响图像质量，结合其他方法可进一步降低复制率达16.82%。**

- **链接: [http://arxiv.org/pdf/2505.21848v1](http://arxiv.org/pdf/2505.21848v1)**

> **作者:** Jingqi Xu; Chenghao Li; Yuke Zhang; Peter A. Beerel
>
> **摘要:** Diffusion models have demonstrated remarkable potential in generating high-quality images. However, their tendency to replicate training data raises serious privacy concerns, particularly when the training datasets contain sensitive or private information. Existing mitigation strategies primarily focus on reducing image duplication, modifying the cross-attention mechanism, and altering the denoising backbone architecture of diffusion models. Moreover, recent work has shown that adding a consistent small amount of noise to text embeddings can reduce replication to some degree. In this work, we begin by analyzing the impact of adding varying amounts of noise. Based on our analysis, we propose a fine-grained noise injection technique that probabilistically adds a larger amount of noise to token embeddings. We refer to our method as Fine-grained Probabilistic Addition of Noise (FPAN). Through our extensive experiments, we show that our proposed FPAN can reduce replication by an average of 28.78% compared to the baseline diffusion model without significantly impacting image quality, and outperforms the prior consistent-magnitude-noise-addition approach by 26.51%. Moreover, when combined with other existing mitigation methods, our FPAN approach can further reduce replication by up to 16.82% with similar, if not improved, image quality.
>
---
#### [new 031] Can NeRFs See without Cameras?
- **分类: cs.CV; cs.AI**

- **简介: 论文提出将NeRF扩展至处理多径RF信号，解决从稀疏WiFi测量反推室内布局的逆问题，实现环境隐式建模，支持信号预测等应用。**

- **链接: [http://arxiv.org/pdf/2505.22441v1](http://arxiv.org/pdf/2505.22441v1)**

> **作者:** Chaitanya Amballa; Sattwik Basu; Yu-Lin Wei; Zhijian Yang; Mehmet Ergezer; Romit Roy Choudhury
>
> **摘要:** Neural Radiance Fields (NeRFs) have been remarkably successful at synthesizing novel views of 3D scenes by optimizing a volumetric scene function. This scene function models how optical rays bring color information from a 3D object to the camera pixels. Radio frequency (RF) or audio signals can also be viewed as a vehicle for delivering information about the environment to a sensor. However, unlike camera pixels, an RF/audio sensor receives a mixture of signals that contain many environmental reflections (also called "multipath"). Is it still possible to infer the environment using such multipath signals? We show that with redesign, NeRFs can be taught to learn from multipath signals, and thereby "see" the environment. As a grounding application, we aim to infer the indoor floorplan of a home from sparse WiFi measurements made at multiple locations inside the home. Although a difficult inverse problem, our implicitly learnt floorplans look promising, and enables forward applications, such as indoor signal prediction and basic ray tracing.
>
---
#### [new 032] Do DeepFake Attribution Models Generalize?
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究DeepFake归因模型的跨数据集泛化能力。针对现有检测模型无法区分不同生成方法的问题，通过对比二分类与多分类模型、分析数据分布变化及测试对比学习方法，评估归因模型性能。发现二分类模型泛化更好，而归因模型可通过大模型、对比方法和高质量数据提升效果。**

- **链接: [http://arxiv.org/pdf/2505.21520v1](http://arxiv.org/pdf/2505.21520v1)**

> **作者:** Spiros Baxavanakis; Manos Schinas; Symeon Papadopoulos
>
> **摘要:** Recent advancements in DeepFake generation, along with the proliferation of open-source tools, have significantly lowered the barrier for creating synthetic media. This trend poses a serious threat to the integrity and authenticity of online information, undermining public trust in institutions and media. State-of-the-art research on DeepFake detection has primarily focused on binary detection models. A key limitation of these models is that they treat all manipulation techniques as equivalent, despite the fact that different methods introduce distinct artifacts and visual cues. Only a limited number of studies explore DeepFake attribution models, although such models are crucial in practical settings. By providing the specific manipulation method employed, these models could enhance both the perceived trustworthiness and explainability for end users. In this work, we leverage five state-of-the-art backbone models and conduct extensive experiments across six DeepFake datasets. First, we compare binary and multi-class models in terms of cross-dataset generalization. Second, we examine the accuracy of attribution models in detecting seen manipulation methods in unknown datasets, hence uncovering data distribution shifts on the same DeepFake manipulations. Last, we assess the effectiveness of contrastive methods in improving cross-dataset generalization performance. Our findings indicate that while binary models demonstrate better generalization abilities, larger models, contrastive methods, and higher data quality can lead to performance improvements in attribution models. The code of this work is available on GitHub.
>
---
#### [new 033] Thinking with Generated Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出一种多模态视觉推理新范式，使大型模型能通过生成中间视觉步骤（如分解任务、自检优化）进行主动思考，突破仅依赖输入图像或纯文本推理的限制。通过生成视觉子目标及自我批判机制，显著提升复杂场景处理能力（如生物、建筑、刑侦领域），实现50%性能提升。任务属多模态视觉推理，解决模型视觉思维能力不足问题。**

- **链接: [http://arxiv.org/pdf/2505.22525v1](http://arxiv.org/pdf/2505.22525v1)**

> **作者:** Ethan Chern; Zhulin Hu; Steffi Chern; Siqi Kou; Jiadi Su; Yan Ma; Zhijie Deng; Pengfei Liu
>
> **摘要:** We present Thinking with Generated Images, a novel paradigm that fundamentally transforms how large multimodal models (LMMs) engage with visual reasoning by enabling them to natively think across text and vision modalities through spontaneous generation of intermediate visual thinking steps. Current visual reasoning with LMMs is constrained to either processing fixed user-provided images or reasoning solely through text-based chain-of-thought (CoT). Thinking with Generated Images unlocks a new dimension of cognitive capability where models can actively construct intermediate visual thoughts, critique their own visual hypotheses, and refine them as integral components of their reasoning process. We demonstrate the effectiveness of our approach through two complementary mechanisms: (1) vision generation with intermediate visual subgoals, where models decompose complex visual tasks into manageable components that are generated and integrated progressively, and (2) vision generation with self-critique, where models generate an initial visual hypothesis, analyze its shortcomings through textual reasoning, and produce refined outputs based on their own critiques. Our experiments on vision generation benchmarks show substantial improvements over baseline approaches, with our models achieving up to 50% (from 38% to 57%) relative improvement in handling complex multi-object scenarios. From biochemists exploring novel protein structures, and architects iterating on spatial designs, to forensic analysts reconstructing crime scenes, and basketball players envisioning strategic plays, our approach enables AI models to engage in the kind of visual imagination and iterative refinement that characterizes human creative, analytical, and strategic thinking. We release our open-source suite at https://github.com/GAIR-NLP/thinking-with-generated-images.
>
---
#### [new 034] Universal Domain Adaptation for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于无监督域适应语义分割（UDA-SS）任务，解决源域与目标域类别设置未知导致的私有类干扰问题。提出UniMAP框架，通过领域特定原型区分（DSPD）分离公共/私有类特征，结合目标图像匹配（TIM）优化公共类学习，实现无先验知识的鲁棒域适应。**

- **链接: [http://arxiv.org/pdf/2505.22458v1](http://arxiv.org/pdf/2505.22458v1)**

> **作者:** Seun-An Choe; Keon-Hee Park; Jinwoo Choi; Gyeong-Moon Park
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Unsupervised domain adaptation for semantic segmentation (UDA-SS) aims to transfer knowledge from labeled source data to unlabeled target data. However, traditional UDA-SS methods assume that category settings between source and target domains are known, which is unrealistic in real-world scenarios. This leads to performance degradation if private classes exist. To address this limitation, we propose Universal Domain Adaptation for Semantic Segmentation (UniDA-SS), achieving robust adaptation even without prior knowledge of category settings. We define the problem in the UniDA-SS scenario as low confidence scores of common classes in the target domain, which leads to confusion with private classes. To solve this problem, we propose UniMAP: UniDA-SS with Image Matching and Prototype-based Distinction, a novel framework composed of two key components. First, Domain-Specific Prototype-based Distinction (DSPD) divides each class into two domain-specific prototypes, enabling finer separation of domain-specific features and enhancing the identification of common classes across domains. Second, Target-based Image Matching (TIM) selects a source image containing the most common-class pixels based on the target pseudo-label and pairs it in a batch to promote effective learning of common classes. We also introduce a new UniDA-SS benchmark and demonstrate through various experiments that UniMAP significantly outperforms baselines. The code is available at \href{https://github.com/KU-VGI/UniMAP}{this https URL}.
>
---
#### [new 035] Enhancing Vision Transformer Explainability Using Artificial Astrocytes
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于可解释性AI任务，旨在提升Vision Transformer（ViT）的决策透明度。针对复杂模型解释不足的问题，提出ViTA方法：通过模拟神经胶质细胞（artificial astrocytes）增强预训练模型的推理，无需重新训练。实验显示，其生成的热图与人类标注更一致，显著优于标准ViT。**

- **链接: [http://arxiv.org/pdf/2505.21513v1](http://arxiv.org/pdf/2505.21513v1)**

> **作者:** Nicolas Echevarrieta-Catalan; Ana Ribas-Rodriguez; Francisco Cedron; Odelia Schwartz; Vanessa Aguiar-Pulido
>
> **备注:** LXCV Workshop at IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR) 2025
>
> **摘要:** Machine learning models achieve high precision, but their decision-making processes often lack explainability. Furthermore, as model complexity increases, explainability typically decreases. Existing efforts to improve explainability primarily involve developing new eXplainable artificial intelligence (XAI) techniques or incorporating explainability constraints during training. While these approaches yield specific improvements, their applicability remains limited. In this work, we propose the Vision Transformer with artificial Astrocytes (ViTA). This training-free approach is inspired by neuroscience and enhances the reasoning of a pretrained deep neural network to generate more human-aligned explanations. We evaluated our approach employing two well-known XAI techniques, Grad-CAM and Grad-CAM++, and compared it to a standard Vision Transformer (ViT). Using the ClickMe dataset, we quantified the similarity between the heatmaps produced by the XAI techniques and a (human-aligned) ground truth. Our results consistently demonstrate that incorporating artificial astrocytes enhances the alignment of model explanations with human perception, leading to statistically significant improvements across all XAI techniques and metrics utilized.
>
---
#### [new 036] Benign-to-Toxic Jailbreaking: Inducing Harmful Responses from Harmless Prompts
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态模型安全漏洞研究，旨在解决现有对抗攻击难以在无明显有害输入时诱导模型输出有害内容的问题。提出Benign-to-Toxic（B2T）新范式，通过优化对抗图像使无害提示生成有害响应，突破模型安全机制，验证了其有效性及黑盒迁移性，揭示多模态系统潜在漏洞。**

- **链接: [http://arxiv.org/pdf/2505.21556v1](http://arxiv.org/pdf/2505.21556v1)**

> **作者:** Hee-Seon Kim; Minbeom Kim; Wonjun Lee; Kihyun Kim; Changick Kim
>
> **备注:** LVLM, Jailbreak
>
> **摘要:** Optimization-based jailbreaks typically adopt the Toxic-Continuation setting in large vision-language models (LVLMs), following the standard next-token prediction objective. In this setting, an adversarial image is optimized to make the model predict the next token of a toxic prompt. However, we find that the Toxic-Continuation paradigm is effective at continuing already-toxic inputs, but struggles to induce safety misalignment when explicit toxic signals are absent. We propose a new paradigm: Benign-to-Toxic (B2T) jailbreak. Unlike prior work, we optimize adversarial images to induce toxic outputs from benign conditioning. Since benign conditioning contains no safety violations, the image alone must break the model's safety mechanisms. Our method outperforms prior approaches, transfers in black-box settings, and complements text-based jailbreaks. These results reveal an underexplored vulnerability in multimodal alignment and introduce a fundamentally new direction for jailbreak approaches.
>
---
#### [new 037] MMTBENCH: A Unified Benchmark for Complex Multimodal Table Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MMTBENCH基准，针对复杂多模态表格推理任务，解决现有视觉语言模型在整合表格数据与视觉元素（如图表）上的不足。通过构建含500个真实表格、4021个问题的答案数据集，分类问题类型、推理类型及表格类型，评估模型性能，揭示视觉推理与多步推断的短板，推动多模态模型架构优化。**

- **链接: [http://arxiv.org/pdf/2505.21771v1](http://arxiv.org/pdf/2505.21771v1)**

> **作者:** Prasham Yatinkumar Titiya; Jainil Trivedi; Chitta Baral; Vivek Gupta
>
> **摘要:** Multimodal tables those that integrate semi structured data with visual elements such as charts and maps are ubiquitous across real world domains, yet they pose a formidable challenge to current vision language models (VLMs). While Large Language models (LLMs) and VLMs have demonstrated strong capabilities in text and image understanding, their performance on complex, real world multimodal table reasoning remains unexplored. To bridge this gap, we introduce MMTBENCH (Multimodal Table Benchmark), a benchmark consisting of 500 real world multimodal tables drawn from diverse real world sources, with a total of 4021 question answer pairs. MMTBENCH questions cover four question types (Explicit, Implicit, Answer Mention, and Visual Based), five reasoning types (Mathematical, Extrema Identification, Fact Verification, Vision Based, and Others), and eight table types (Single/Multiple Entity, Maps and Charts with Entities, Single/Multiple Charts, Maps, and Visualizations). Extensive evaluation of state of the art models on all types reveals substantial performance gaps, particularly on questions requiring visual-based reasoning and multi-step inference. These findings show the urgent need for improved architectures that more tightly integrate vision and language processing. By providing a challenging, high-quality resource that mirrors the complexity of real-world tasks, MMTBENCH underscores its value as a resource for future research on multimodal tables.
>
---
#### [new 038] Beyond Perception: Evaluating Abstract Visual Reasoning through Multi-Stage Task
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦抽象视觉推理（AVR）评估任务，针对现有方法忽视多阶段推理过程及中间步骤评估的问题，提出MultiStAR多阶段基准和MSEval指标，通过实验揭示MLLMs在复杂规则检测阶段的局限性。**

- **链接: [http://arxiv.org/pdf/2505.21850v1](http://arxiv.org/pdf/2505.21850v1)**

> **作者:** Yanbei Jiang; Yihao Ding; Chao Lei; Jiayang Ao; Jey Han Lau; Krista A. Ehinger
>
> **备注:** Accepted at ACL Findings
>
> **摘要:** Current Multimodal Large Language Models (MLLMs) excel in general visual reasoning but remain underexplored in Abstract Visual Reasoning (AVR), which demands higher-order reasoning to identify abstract rules beyond simple perception. Existing AVR benchmarks focus on single-step reasoning, emphasizing the end result but neglecting the multi-stage nature of reasoning process. Past studies found MLLMs struggle with these benchmarks, but it doesn't explain how they fail. To address this gap, we introduce MultiStAR, a Multi-Stage AVR benchmark, based on RAVEN, designed to assess reasoning across varying levels of complexity. Additionally, existing metrics like accuracy only focus on the final outcomes while do not account for the correctness of intermediate steps. Therefore, we propose a novel metric, MSEval, which considers the correctness of intermediate steps in addition to the final outcomes. We conduct comprehensive experiments on MultiStAR using 17 representative close-source and open-source MLLMs. The results reveal that while existing MLLMs perform adequately on basic perception tasks, they continue to face challenges in more complex rule detection stages.
>
---
#### [new 039] PacTure: Efficient PBR Texture Generation on Packed Views with Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 论文提出PacTure，用于从3D网格、文本及可选图像生成PBR材质纹理。针对现有方法效率低、全局不一致或分辨率受限问题，其创新性地采用view packing技术优化多视角排列提升分辨率，结合自回归框架实现高效多领域生成，兼顾质量和推理效率。**

- **链接: [http://arxiv.org/pdf/2505.22394v1](http://arxiv.org/pdf/2505.22394v1)**

> **作者:** Fan Fei; Jiajun Tang; Fei-Peng Tian; Boxin Shi; Ping Tan
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** We present PacTure, a novel framework for generating physically-based rendering (PBR) material textures from an untextured 3D mesh, a text description, and an optional image prompt. Early 2D generation-based texturing approaches generate textures sequentially from different views, resulting in long inference times and globally inconsistent textures. More recent approaches adopt multi-view generation with cross-view attention to enhance global consistency, which, however, limits the resolution for each view. In response to these weaknesses, we first introduce view packing, a novel technique that significantly increases the effective resolution for each view during multi-view generation without imposing additional inference cost, by formulating the arrangement of multi-view maps as a 2D rectangle bin packing problem. In contrast to UV mapping, it preserves the spatial proximity essential for image generation and maintains full compatibility with current 2D generative models. To further reduce the inference cost, we enable fine-grained control and multi-domain generation within the next-scale prediction autoregressive framework to create an efficient multi-view multi-domain generative backbone. Extensive experiments show that PacTure outperforms state-of-the-art methods in both quality of generated PBR textures and efficiency in training and inference.
>
---
#### [new 040] Cross-DINO: Cross the Deep MLP and Transformer for Small Object Detection
- **分类: cs.CV**

- **简介: 该论文属于小物体检测任务，针对Transformer模型在小目标特征捕捉、注意力模糊及分类分数低的问题，提出Cross-DINO：融合深度MLP与Transformer，设计CCTM模块增强细节，引入类别-尺寸软标签及Boost Loss优化分类，实现参数更少、性能更优的检测效果。**

- **链接: [http://arxiv.org/pdf/2505.21868v1](http://arxiv.org/pdf/2505.21868v1)**

> **作者:** Guiping Cao; Wenjian Huang; Xiangyuan Lan; Jianguo Zhang; Dongmei Jiang; Yaowei Wang
>
> **备注:** IEEE TRANSACTIONS ON MULTIMEDIA
>
> **摘要:** Small Object Detection (SOD) poses significant challenges due to limited information and the model's low class prediction score. While Transformer-based detectors have shown promising performance, their potential for SOD remains largely unexplored. In typical DETR-like frameworks, the CNN backbone network, specialized in aggregating local information, struggles to capture the necessary contextual information for SOD. The multiple attention layers in the Transformer Encoder face difficulties in effectively attending to small objects and can also lead to blurring of features. Furthermore, the model's lower class prediction score of small objects compared to large objects further increases the difficulty of SOD. To address these challenges, we introduce a novel approach called Cross-DINO. This approach incorporates the deep MLP network to aggregate initial feature representations with both short and long range information for SOD. Then, a new Cross Coding Twice Module (CCTM) is applied to integrate these initial representations to the Transformer Encoder feature, enhancing the details of small objects. Additionally, we introduce a new kind of soft label named Category-Size (CS), integrating the Category and Size of objects. By treating CS as new ground truth, we propose a new loss function called Boost Loss to improve the class prediction score of the model. Extensive experimental results on COCO, WiderPerson, VisDrone, AI-TOD, and SODA-D datasets demonstrate that Cross-DINO efficiently improves the performance of DETR-like models on SOD. Specifically, our model achieves 36.4% APs on COCO for SOD with only 45M parameters, outperforming the DINO by +4.4% APS (36.4% vs. 32.0%) with fewer parameters and FLOPs, under 12 epochs training setting. The source codes will be available at https://github.com/Med-Process/Cross-DINO.
>
---
#### [new 041] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network
- **分类: cs.CV**

- **简介: 论文提出RC-AutoCalib，首个在线雷达-相机自动校准方法。针对雷达高度数据稀疏与不确定性问题，设计双视角特征融合机制（前视与鸟瞰），结合跨模态注意力与噪声鲁棒匹配器，提升校准精度。实验表明优于现有方法，创基准。**

- **链接: [http://arxiv.org/pdf/2505.22427v1](http://arxiv.org/pdf/2505.22427v1)**

> **作者:** Van-Tin Luu; Yon-Lin Cai; Vu-Hoang Tran; Wei-Chen Chiu; Yi-Ting Chen; Ching-Chun Huang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** This paper presents a groundbreaking approach - the first online automatic geometric calibration method for radar and camera systems. Given the significant data sparsity and measurement uncertainty in radar height data, achieving automatic calibration during system operation has long been a challenge. To address the sparsity issue, we propose a Dual-Perspective representation that gathers features from both frontal and bird's-eye views. The frontal view contains rich but sensitive height information, whereas the bird's-eye view provides robust features against height uncertainty. We thereby propose a novel Selective Fusion Mechanism to identify and fuse reliable features from both perspectives, reducing the effect of height uncertainty. Moreover, for each view, we incorporate a Multi-Modal Cross-Attention Mechanism to explicitly find location correspondences through cross-modal matching. During the training phase, we also design a Noise-Resistant Matcher to provide better supervision and enhance the robustness of the matching mechanism against sparsity and height uncertainty. Our experimental results, tested on the nuScenes dataset, demonstrate that our method significantly outperforms previous radar-camera auto-calibration methods, as well as existing state-of-the-art LiDAR-camera calibration techniques, establishing a new benchmark for future research. The code is available at https://github.com/nycu-acm/RC-AutoCalib.
>
---
#### [new 042] GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GeoDrive，针对自动驾驶中世界模型在3D几何一致性和遮挡处理上的不足，通过整合3D几何约束与动态编辑模块，提升场景建模精度与动作控制能力，实现更安全可靠的自动驾驶规划与仿真。**

- **链接: [http://arxiv.org/pdf/2505.22421v1](http://arxiv.org/pdf/2505.22421v1)**

> **作者:** Anthony Chen; Wenzhao Zheng; Yida Wang; Xueyang Zhang; Kun Zhan; Peng Jia; Kurt Keutzer; Shangbang Zhang
>
> **备注:** code will be released at https://github.com/antonioo-c/GeoDrive
>
> **摘要:** Recent advancements in world models have revolutionized dynamic environment simulation, allowing systems to foresee future states and assess potential actions. In autonomous driving, these capabilities help vehicles anticipate the behavior of other road users, perform risk-aware planning, accelerate training in simulation, and adapt to novel scenarios, thereby enhancing safety and reliability. Current approaches exhibit deficiencies in maintaining robust 3D geometric consistency or accumulating artifacts during occlusion handling, both critical for reliable safety assessment in autonomous navigation tasks. To address this, we introduce GeoDrive, which explicitly integrates robust 3D geometry conditions into driving world models to enhance spatial understanding and action controllability. Specifically, we first extract a 3D representation from the input frame and then obtain its 2D rendering based on the user-specified ego-car trajectory. To enable dynamic modeling, we propose a dynamic editing module during training to enhance the renderings by editing the positions of the vehicles. Extensive experiments demonstrate that our method significantly outperforms existing models in both action accuracy and 3D spatial awareness, leading to more realistic, adaptable, and reliable scene modeling for safer autonomous driving. Additionally, our model can generalize to novel trajectories and offers interactive scene editing capabilities, such as object editing and object trajectory control.
>
---
#### [new 043] Caption This, Reason That: VLMs Caught in the Middle
- **分类: cs.CV; cs.AI**

- **简介: 该论文评估视觉语言模型（VLM）在认知任务中的表现，分析其在感知、注意力和记忆上的局限性，发现空间推理和选择性注意力存在明显不足。通过解耦分析和微调实验，提出改进Chain-of-Thought能力及针对性训练可提升核心认知能力，揭示VLM在感知与推理协同中的瓶颈，并提供解决方案。**

- **链接: [http://arxiv.org/pdf/2505.21538v1](http://arxiv.org/pdf/2505.21538v1)**

> **作者:** Zihan Weng; Lucas Gomez; Taylor Whittington Webb; Pouya Bashivan
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable progress in visual understanding in recent years. Yet, they still lag behind human capabilities in specific visual tasks such as counting or relational reasoning. To understand the underlying limitations, we adopt methodologies from cognitive science, analyzing VLM performance along core cognitive axes: Perception, Attention, and Memory. Using a suite of tasks targeting these abilities, we evaluate state-of-the-art VLMs, including GPT-4o. Our analysis reveals distinct cognitive profiles: while advanced models approach ceiling performance on some tasks (e.g. category identification), a significant gap persists, particularly in tasks requiring spatial understanding or selective attention. Investigating the source of these failures and potential methods for improvement, we employ a vision-text decoupling analysis, finding that models struggling with direct visual reasoning show marked improvement when reasoning over their own generated text captions. These experiments reveal a strong need for improved VLM Chain-of-Thought (CoT) abilities, even in models that consistently exceed human performance. Furthermore, we demonstrate the potential of targeted fine-tuning on composite visual reasoning tasks and show that fine-tuning smaller VLMs substantially improves core cognitive abilities. While this improvement does not translate to large enhancements on challenging, out-of-distribution benchmarks, we show broadly that VLM performance on our datasets strongly correlates with performance on these other benchmarks. Our work provides a detailed analysis of VLM cognitive strengths and weaknesses and identifies key bottlenecks in simultaneous perception and reasoning while also providing an effective and simple solution.
>
---
#### [new 044] Improving Brain-to-Image Reconstruction via Fine-Grained Text Bridging
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于脑-图像重建任务，旨在解决现有方法重建图像细节缺失和语义不一致的问题。提出FgB2I方法，通过细粒度文本作为桥梁，分三阶段工作：利用视觉语言模型增强细节、基于fMRI信号解码细粒度文本（采用三指标优化），最后整合文本进行图像重建，提升重建精度。**

- **链接: [http://arxiv.org/pdf/2505.22150v1](http://arxiv.org/pdf/2505.22150v1)**

> **作者:** Runze Xia; Shuo Feng; Renzhi Wang; Congchi Yin; Xuyun Wen; Piji Li
>
> **备注:** CogSci2025
>
> **摘要:** Brain-to-Image reconstruction aims to recover visual stimuli perceived by humans from brain activity. However, the reconstructed visual stimuli often missing details and semantic inconsistencies, which may be attributed to insufficient semantic information. To address this issue, we propose an approach named Fine-grained Brain-to-Image reconstruction (FgB2I), which employs fine-grained text as bridge to improve image reconstruction. FgB2I comprises three key stages: detail enhancement, decoding fine-grained text descriptions, and text-bridged brain-to-image reconstruction. In the detail-enhancement stage, we leverage large vision-language models to generate fine-grained captions for visual stimuli and experimentally validate its importance. We propose three reward metrics (object accuracy, text-image semantic similarity, and image-image semantic similarity) to guide the language model in decoding fine-grained text descriptions from fMRI signals. The fine-grained text descriptions can be integrated into existing reconstruction methods to achieve fine-grained Brain-to-Image reconstruction.
>
---
#### [new 045] What is Adversarial Training for Diffusion Models?
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究扩散模型的对抗训练方法，旨在增强其对噪声、异常数据及对抗攻击的鲁棒性。提出通过等方差约束扩散过程，无需假设噪声模型，集成随机/对抗噪声提升平滑性。在合成数据集及CIFAR-10等基准上验证，显著提升抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2505.21742v1](http://arxiv.org/pdf/2505.21742v1)**

> **作者:** Briglia Maria Rosaria; Mujtaba Hussain Mirza; Giuseppe Lisanti; Iacopo Masi
>
> **备注:** 40 pages
>
> **摘要:** We answer the question in the title, showing that adversarial training (AT) for diffusion models (DMs) fundamentally differs from classifiers: while AT in classifiers enforces output invariance, AT in DMs requires equivariance to keep the diffusion process aligned with the data distribution. AT is a way to enforce smoothness in the diffusion flow, improving robustness to outliers and corrupted data. Unlike prior art, our method makes no assumptions about the noise model and integrates seamlessly into diffusion training by adding random noise, similar to randomized smoothing, or adversarial noise, akin to AT. This enables intrinsic capabilities such as handling noisy data, dealing with extreme variability such as outliers, preventing memorization, and improving robustness. We rigorously evaluate our approach with proof-of-concept datasets with known distributions in low- and high-dimensional space, thereby taking a perfect measure of errors; we further evaluate on standard benchmarks such as CIFAR-10, CelebA and LSUN Bedroom, showing strong performance under severe noise, data corruption, and iterative adversarial attacks.
>
---
#### [new 046] CAST: Contrastive Adaptation and Distillation for Semi-Supervised Instance Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CAST框架，用于半监督实例分割任务。针对传统方法依赖大量标注数据和大模型的问题，通过三阶段流程（对比自适应、多目标蒸馏、微调），将预训练大模型压缩为紧凑学生模型，核心为实例感知对比损失，提升未标注数据利用效率。在Cityscapes等数据集上，学生模型性能超越教师模型及现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21904v1](http://arxiv.org/pdf/2505.21904v1)**

> **作者:** Pardis Taghavi; Tian Liu; Renjie Li; Reza Langari; Zhengzhong Tu
>
> **摘要:** Instance segmentation demands costly per-pixel annotations and large models. We introduce CAST, a semi-supervised knowledge distillation (SSKD) framework that compresses pretrained vision foundation models (VFM) into compact experts using limited labeled and abundant unlabeled data. CAST unfolds in three stages: (1) domain adaptation of the VFM teacher(s) via self-training with contrastive pixel calibration, (2) distillation into a compact student via a unified multi-objective loss that couples standard supervision and pseudo-labels with our instance-aware pixel-wise contrastive term, and (3) fine-tuning on labeled data to remove residual pseudo-label bias. Central to CAST is an \emph{instance-aware pixel-wise contrastive loss} that fuses mask and class scores to mine informative negatives and enforce clear inter-instance margins. By maintaining this contrastive signal across both adaptation and distillation, we align teacher and student embeddings and fully leverage unlabeled images. On Cityscapes and ADE20K, our ~11X smaller student surpasses its adapted VFM teacher(s) by +3.4 AP (33.9 vs. 30.5) and +1.5 AP (16.7 vs. 15.2) and outperforms state-of-the-art semi-supervised approaches.
>
---
#### [new 047] UniTalk: Towards Universal Active Speaker Detection in Real World Scenarios
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UniTalk数据集，旨在提升现实场景下的主动说话者检测（ASD）任务。针对现有数据集（如AVA）因依赖老电影导致的领域差距问题，UniTalk包含多语言、嘈杂环境、多人同时说话等复杂场景，覆盖48,693个身份和44.5小时视频。实验表明现有模型在UniTalk表现不佳，但基于其训练的模型泛化能力更强，建立了新基准。**

- **链接: [http://arxiv.org/pdf/2505.21954v1](http://arxiv.org/pdf/2505.21954v1)**

> **作者:** Le Thien Phuc Nguyen; Zhuoran Yu; Khoa Quang Nhat Cao; Yuwei Guo; Tu Ho Manh Pham; Tuan Tai Nguyen; Toan Ngo Duc Vo; Lucas Poon; Soochahn Lee; Yong Jae Lee
>
> **摘要:** We present UniTalk, a novel dataset specifically designed for the task of active speaker detection, emphasizing challenging scenarios to enhance model generalization. Unlike previously established benchmarks such as AVA, which predominantly features old movies and thus exhibits significant domain gaps, UniTalk focuses explicitly on diverse and difficult real-world conditions. These include underrepresented languages, noisy backgrounds, and crowded scenes - such as multiple visible speakers speaking concurrently or in overlapping turns. It contains over 44.5 hours of video with frame-level active speaker annotations across 48,693 speaking identities, and spans a broad range of video types that reflect real-world conditions. Through rigorous evaluation, we show that state-of-the-art models, while achieving nearly perfect scores on AVA, fail to reach saturation on UniTalk, suggesting that the ASD task remains far from solved under realistic conditions. Nevertheless, models trained on UniTalk demonstrate stronger generalization to modern "in-the-wild" datasets like Talkies and ASW, as well as to AVA. UniTalk thus establishes a new benchmark for active speaker detection, providing researchers with a valuable resource for developing and evaluating versatile and resilient models. Dataset: https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD Code: https://github.com/plnguyen2908/UniTalk-ASD-code
>
---
#### [new 048] Moment kernels: a simple and scalable approach for equivariance to rotations and reflections in deep convolutional networks
- **分类: cs.CV; cs.LG**

- **简介: 论文提出"moment kernels"方法，通过简单卷积核实现深度网络对旋转和平移的等变性，解决传统方法依赖复杂数学导致未被广泛采用的问题。证明其形式的必然性，并应用于生物医学图像分类、3D配准和细胞分割任务。**

- **链接: [http://arxiv.org/pdf/2505.21736v1](http://arxiv.org/pdf/2505.21736v1)**

> **作者:** Zachary Schlamowitz; Andrew Bennecke; Daniel J. Tward
>
> **摘要:** The principle of translation equivariance (if an input image is translated an output image should be translated by the same amount), led to the development of convolutional neural networks that revolutionized machine vision. Other symmetries, like rotations and reflections, play a similarly critical role, especially in biomedical image analysis, but exploiting these symmetries has not seen wide adoption. We hypothesize that this is partially due to the mathematical complexity of methods used to exploit these symmetries, which often rely on representation theory, a bespoke concept in differential geometry and group theory. In this work, we show that the same equivariance can be achieved using a simple form of convolution kernels that we call ``moment kernels,'' and prove that all equivariant kernels must take this form. These are a set of radially symmetric functions of a spatial position $x$, multiplied by powers of the components of $x$ or the identity matrix. We implement equivariant neural networks using standard convolution modules, and provide architectures to execute several biomedical image analysis tasks that depend on equivariance principles: classification (outputs are invariant under orthogonal transforms), 3D image registration (outputs transform like a vector), and cell segmentation (quadratic forms defining ellipses transform like a matrix).
>
---
#### [new 049] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于无监督视觉表征学习任务，旨在解决传统原型方法单一表征不足的问题。提出SOP策略，通过多语义相似支持嵌入（SEs）互补表征数据区域，并设计非参数损失函数及SOP-MIM任务。实验显示其在检索等任务中达SOTA，并随模型复杂度提升性能。**

- **链接: [http://arxiv.org/pdf/2505.21533v1](http://arxiv.org/pdf/2505.21533v1)**

> **作者:** Thalles Silva; Helio Pedrini; Adín Ramírez Rivera
>
> **备注:** Accepted at ICML 2025, code at https://github.com/sthalles/sop
>
> **摘要:** We present Self-Organizing Visual Prototypes (SOP), a new training technique for unsupervised visual feature learning. Unlike existing prototypical self-supervised learning (SSL) methods that rely on a single prototype to encode all relevant features of a hidden cluster in the data, we propose the SOP strategy. In this strategy, a prototype is represented by many semantically similar representations, or support embeddings (SEs), each containing a complementary set of features that together better characterize their region in space and maximize training performance. We reaffirm the feasibility of non-parametric SSL by introducing novel non-parametric adaptations of two loss functions that implement the SOP strategy. Notably, we introduce the SOP Masked Image Modeling (SOP-MIM) task, where masked representations are reconstructed from the perspective of multiple non-parametric local SEs. We comprehensively evaluate the representations learned using the SOP strategy on a range of benchmarks, including retrieval, linear evaluation, fine-tuning, and object detection. Our pre-trained encoders achieve state-of-the-art performance on many retrieval benchmarks and demonstrate increasing performance gains with more complex encoders.
>
---
#### [new 050] EaqVLA: Encoding-aligned Quantization for Vision-Language-Action Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出EaqVLA框架，针对视觉-语言-动作（VLA）模型量化中的编码对齐问题，通过分析多粒度对齐偏差并设计混合精度对齐量化方法，在减少端到端控制性能损失的同时实现加速。**

- **链接: [http://arxiv.org/pdf/2505.21567v1](http://arxiv.org/pdf/2505.21567v1)**

> **作者:** Feng Jiang; Zihao Zheng; Xiuping Cui; Maoliang Li; JIayu Chen; Xiang Chen
>
> **摘要:** With the development of Embodied Artificial intelligence, the end-to-end control policy such as Vision-Language-Action (VLA) model has become the mainstream. Existing VLA models faces expensive computing/storage cost, which need to be optimized. Quantization is considered as the most effective method which can not only reduce the memory cost but also achieve computation acceleration. However, we find the token alignment of VLA models hinders the application of existing quantization methods. To address this, we proposed an optimized framework called EaqVLA, which apply encoding-aligned quantization to VLA models. Specifically, we propose an complete analysis method to find the misalignment in various granularity. Based on the analysis results, we propose a mixed precision quantization with the awareness of encoding alignment. Experiments shows that the porposed EaqVLA achieves better quantization performance (with the minimal quantization loss for end-to-end action control and xxx times acceleration) than existing quantization methods.
>
---
#### [new 051] Frugal Incremental Generative Modeling using Variational Autoencoders
- **分类: cs.CV**

- **简介: 该论文属于增量生成建模任务，解决数据增长导致的扩展性及灾难性遗忘问题。提出基于VAE的无重播方法，设计多模态潜在空间与正交准则，采用静态/动态VAE变体控制参数增长，实现高精度且内存效率提升一个数量级。**

- **链接: [http://arxiv.org/pdf/2505.22408v1](http://arxiv.org/pdf/2505.22408v1)**

> **作者:** Victor Enescu; Hichem Sahbi
>
> **摘要:** Continual or incremental learning holds tremendous potential in deep learning with different challenges including catastrophic forgetting. The advent of powerful foundation and generative models has propelled this paradigm even further, making it one of the most viable solution to train these models. However, one of the persisting issues lies in the increasing volume of data particularly with replay-based methods. This growth introduces challenges with scalability since continuously expanding data becomes increasingly demanding as the number of tasks grows. In this paper, we attenuate this issue by devising a novel replay-free incremental learning model based on Variational Autoencoders (VAEs). The main contribution of this work includes (i) a novel incremental generative modelling, built upon a well designed multi-modal latent space, and also (ii) an orthogonality criterion that mitigates catastrophic forgetting of the learned VAEs. The proposed method considers two variants of these VAEs: static and dynamic with no (or at most a controlled) growth in the number of parameters. Extensive experiments show that our method is (at least) an order of magnitude more ``memory-frugal'' compared to the closely related works while achieving SOTA accuracy scores.
>
---
#### [new 052] StateSpaceDiffuser: Bringing Long Context to Diffusion World Models
- **分类: cs.CV**

- **简介: 该论文属于视觉预测任务，旨在解决扩散世界模型因缺乏长期环境状态而难以保持视觉一致性的难题。提出StateSpaceDiffuser，将状态空间模型（Mamba）与扩散模型结合，通过整合交互历史的序列表示，实现长上下文高保真生成，在2D/3D环境中显著提升长期视觉连贯性。**

- **链接: [http://arxiv.org/pdf/2505.22246v1](http://arxiv.org/pdf/2505.22246v1)**

> **作者:** Nedko Savov; Naser Kazemi; Deheng Zhang; Danda Pani Paudel; Xi Wang; Luc Van Gool
>
> **摘要:** World models have recently become promising tools for predicting realistic visuals based on actions in complex environments. However, their reliance on a short sequence of observations causes them to quickly lose track of context. As a result, visual consistency breaks down after just a few steps, and generated scenes no longer reflect information seen earlier. This limitation of the state-of-the-art diffusion-based world models comes from their lack of a lasting environment state. To address this problem, we introduce StateSpaceDiffuser, where a diffusion model is enabled to perform on long-context tasks by integrating a sequence representation from a state-space model (Mamba), representing the entire interaction history. This design restores long-term memory without sacrificing the high-fidelity synthesis of diffusion models. To rigorously measure temporal consistency, we develop an evaluation protocol that probes a model's ability to reinstantiate seen content in extended rollouts. Comprehensive experiments show that StateSpaceDiffuser significantly outperforms a strong diffusion-only baseline, maintaining a coherent visual context for an order of magnitude more steps. It delivers consistent views in both a 2D maze navigation and a complex 3D environment. These results establish that bringing state-space representations into diffusion models is highly effective in demonstrating both visual details and long-term memory.
>
---
#### [new 053] Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于视频生成模型量化与知识蒸馏任务。针对DiT模型参数多、计算复杂且现有量化方法不适用视频生成的问题，提出Q-VDiT框架：通过TQE补偿量化误差，利用TMD保持帧间时空关联，实现高效压缩与性能提升，场景一致性达23.40，超越现有方法1.9倍。**

- **链接: [http://arxiv.org/pdf/2505.22167v1](http://arxiv.org/pdf/2505.22167v1)**

> **作者:** Weilun Feng; Chuanguang Yang; Haotong Qin; Xiangqi Li; Yu Wang; Zhulin An; Libo Huang; Boyu Diao; Zixiang Zhao; Yongjun Xu; Michele Magno
>
> **备注:** Accepted to ICML2025
>
> **摘要:** Diffusion transformers (DiT) have demonstrated exceptional performance in video generation. However, their large number of parameters and high computational complexity limit their deployment on edge devices. Quantization can reduce storage requirements and accelerate inference by lowering the bit-width of model parameters. Yet, existing quantization methods for image generation models do not generalize well to video generation tasks. We identify two primary challenges: the loss of information during quantization and the misalignment between optimization objectives and the unique requirements of video generation. To address these challenges, we present Q-VDiT, a quantization framework specifically designed for video DiT models. From the quantization perspective, we propose the Token-aware Quantization Estimator (TQE), which compensates for quantization errors in both the token and feature dimensions. From the optimization perspective, we introduce Temporal Maintenance Distillation (TMD), which preserves the spatiotemporal correlations between frames and enables the optimization of each frame with respect to the overall video context. Our W3A6 Q-VDiT achieves a scene consistency of 23.40, setting a new benchmark and outperforming current state-of-the-art quantization methods by 1.9$\times$. Code will be available at https://github.com/cantbebetter2/Q-VDiT.
>
---
#### [new 054] LatentMove: Towards Complex Human Movement Video Generation
- **分类: cs.CV**

- **简介: 该论文属于图像到视频（I2V）生成任务，旨在解决复杂非重复人体动作生成中出现的不自然变形问题。提出LatentMove框架，采用DiT架构结合条件控制分支和可学习的人脸/身体标记，提升动态人体动画的时序与细节一致性，并构建CHV数据集及新型评估指标，实验显示其显著改善了快速复杂动作的生成质量。**

- **链接: [http://arxiv.org/pdf/2505.22046v1](http://arxiv.org/pdf/2505.22046v1)**

> **作者:** Ashkan Taghipour; Morteza Ghahremani; Mohammed Bennamoun; Farid Boussaid; Aref Miri Rekavandi; Zinuo Li; Qiuhong Ke; Hamid Laga
>
> **备注:** 12 pages
>
> **摘要:** Image-to-video (I2V) generation seeks to produce realistic motion sequences from a single reference image. Although recent methods exhibit strong temporal consistency, they often struggle when dealing with complex, non-repetitive human movements, leading to unnatural deformations. To tackle this issue, we present LatentMove, a DiT-based framework specifically tailored for highly dynamic human animation. Our architecture incorporates a conditional control branch and learnable face/body tokens to preserve consistency as well as fine-grained details across frames. We introduce Complex-Human-Videos (CHV), a dataset featuring diverse, challenging human motions designed to benchmark the robustness of I2V systems. We also introduce two metrics to assess the flow and silhouette consistency of generated videos with their ground truth. Experimental results indicate that LatentMove substantially improves human animation quality--particularly when handling rapid, intricate movements--thereby pushing the boundaries of I2V generation. The code, the CHV dataset, and the evaluation metrics will be available at https://github.com/ --.
>
---
#### [new 055] DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DiffDecompose，一种基于扩散Transformer的框架，解决半透明/透明层分解难题。针对现有方法受掩码先验、静态假设和数据不足限制，提出首个大规模数据集AlphaBlend，并通过In-Context分解和位置编码克隆实现无监督层分解，提升像素对应性。**

- **链接: [http://arxiv.org/pdf/2505.21541v1](http://arxiv.org/pdf/2505.21541v1)**

> **作者:** Zitong Wang; Hang Zhao; Qianyu Zhou; Xuequan Lu; Xiangtai Li; Yiren Song
>
> **摘要:** Diffusion models have recently motivated great success in many generation tasks like object removal. Nevertheless, existing image decomposition methods struggle to disentangle semi-transparent or transparent layer occlusions due to mask prior dependencies, static object assumptions, and the lack of datasets. In this paper, we delve into a novel task: Layer-Wise Decomposition of Alpha-Composited Images, aiming to recover constituent layers from single overlapped images under the condition of semi-transparent/transparent alpha layer non-linear occlusion. To address challenges in layer ambiguity, generalization, and data scarcity, we first introduce AlphaBlend, the first large-scale and high-quality dataset for transparent and semi-transparent layer decomposition, supporting six real-world subtasks (e.g., translucent flare removal, semi-transparent cell decomposition, glassware decomposition). Building on this dataset, we present DiffDecompose, a diffusion Transformer-based framework that learns the posterior over possible layer decompositions conditioned on the input image, semantic prompts, and blending type. Rather than regressing alpha mattes directly, DiffDecompose performs In-Context Decomposition, enabling the model to predict one or multiple layers without per-layer supervision, and introduces Layer Position Encoding Cloning to maintain pixel-level correspondence across layers. Extensive experiments on the proposed AlphaBlend dataset and public LOGO dataset verify the effectiveness of DiffDecompose. The code and dataset will be available upon paper acceptance. Our code will be available at: https://github.com/Wangzt1121/DiffDecompose.
>
---
#### [new 056] FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决模型在多模态分布偏移下的鲁棒性不足问题。作者构建了FRAMES-VQA基准，整合10个数据集分类为ID、近/远OOD场景，通过马氏距离量化分布偏移，分析模态间交互及重要性，为开发鲁棒微调方法提供指导。**

- **链接: [http://arxiv.org/pdf/2505.21755v1](http://arxiv.org/pdf/2505.21755v1)**

> **作者:** Chengyue Huang; Brisa Maneechotesuwan; Shivang Chopra; Zsolt Kira
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Visual question answering (VQA) systems face significant challenges when adapting to real-world data shifts, especially in multi-modal contexts. While robust fine-tuning strategies are essential for maintaining performance across in-distribution (ID) and out-of-distribution (OOD) scenarios, current evaluation settings are primarily unimodal or particular to some types of OOD, offering limited insight into the complexities of multi-modal contexts. In this work, we propose a new benchmark FRAMES-VQA (Fine-Tuning Robustness across Multi-Modal Shifts in VQA) for evaluating robust fine-tuning for VQA tasks. We utilize ten existing VQA benchmarks, including VQAv2, IV-VQA, VQA-CP, OK-VQA and others, and categorize them into ID, near and far OOD datasets covering uni-modal, multi-modal and adversarial distribution shifts. We first conduct a comprehensive comparison of existing robust fine-tuning methods. We then quantify the distribution shifts by calculating the Mahalanobis distance using uni-modal and multi-modal embeddings extracted from various models. Further, we perform an extensive analysis to explore the interactions between uni- and multi-modal shifts as well as modality importance for ID and OOD samples. These analyses offer valuable guidance on developing more robust fine-tuning methods to handle multi-modal distribution shifts. The code is available at https://github.com/chengyuehuang511/FRAMES-VQA .
>
---
#### [new 057] 3D Question Answering via only 2D Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决仅用2D视觉语言模型处理3D-QA的问题。针对3D数据稀缺，提出cdViews方法：通过viewSelector筛选关键视图，viewNMS去冗余，提升2D模型在3D任务的零样本推理效果，在ScanQA/SQA达SOTA，证明2D模型优于需大量资源的3D模型。**

- **链接: [http://arxiv.org/pdf/2505.22143v1](http://arxiv.org/pdf/2505.22143v1)**

> **作者:** Fengyun Wang; Sicheng Yu; Jiawei Wu; Jinhui Tang; Hanwang Zhang; Qianru Sun
>
> **备注:** ICML2025
>
> **摘要:** Large vision-language models (LVLMs) have significantly advanced numerous fields. In this work, we explore how to harness their potential to address 3D scene understanding tasks, using 3D question answering (3D-QA) as a representative example. Due to the limited training data in 3D, we do not train LVLMs but infer in a zero-shot manner. Specifically, we sample 2D views from a 3D point cloud and feed them into 2D models to answer a given question. When the 2D model is chosen, e.g., LLAVA-OV, the quality of sampled views matters the most. We propose cdViews, a novel approach to automatically selecting critical and diverse Views for 3D-QA. cdViews consists of two key components: viewSelector prioritizing critical views based on their potential to provide answer-specific information, and viewNMS enhancing diversity by removing redundant views based on spatial overlap. We evaluate cdViews on the widely-used ScanQA and SQA benchmarks, demonstrating that it achieves state-of-the-art performance in 3D-QA while relying solely on 2D models without fine-tuning. These findings support our belief that 2D LVLMs are currently the most effective alternative (of the resource-intensive 3D LVLMs) for addressing 3D tasks.
>
---
#### [new 058] Compositional Scene Understanding through Inverse Generative Modeling
- **分类: cs.CV**

- **简介: 该论文属于场景理解任务，旨在通过逆向生成建模解决复杂场景的结构解析与泛化问题。提出将生成模型分解为场景部件的小模型，通过逆向推理从图像反推生成条件参数，实现对物体集合及全局场景因素的推断，并扩展至预训练文本到图像模型的零样本多物体感知，提升新场景泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.21780v1](http://arxiv.org/pdf/2505.21780v1)**

> **作者:** Yanbo Wang; Justin Dauwels; Yilun Du
>
> **备注:** ICML 2025, Webpage: https://energy-based-model.github.io/compositional-inference/
>
> **摘要:** Generative models have demonstrated remarkable abilities in generating high-fidelity visual content. In this work, we explore how generative models can further be used not only to synthesize visual content but also to understand the properties of a scene given a natural image. We formulate scene understanding as an inverse generative modeling problem, where we seek to find conditional parameters of a visual generative model to best fit a given natural image. To enable this procedure to infer scene structure from images substantially different than those seen during training, we further propose to build this visual generative model compositionally from smaller models over pieces of a scene. We illustrate how this procedure enables us to infer the set of objects in a scene, enabling robust generalization to new test scenes with an increased number of objects of new shapes. We further illustrate how this enables us to infer global scene factors, likewise enabling robust generalization to new scenes. Finally, we illustrate how this approach can be directly applied to existing pretrained text-to-image generative models for zero-shot multi-object perception. Code and visualizations are at \href{https://energy-based-model.github.io/compositional-inference}{https://energy-based-model.github.io/compositional-inference}.
>
---
#### [new 059] Universal Visuo-Tactile Video Understanding for Embodied Interaction
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VTV-LLM，首个多模态大模型，解决触觉与视觉语言融合不足的问题。构建含15万帧触觉视频的VTV150K数据集，设计三阶段训练方法（增强、对齐、微调），实现触觉推理（属性评估、比较、决策），提升人机交互。**

- **链接: [http://arxiv.org/pdf/2505.22566v1](http://arxiv.org/pdf/2505.22566v1)**

> **作者:** Yifan Xie; Mingyang Li; Shoujie Li; Xingting Li; Guangyu Chen; Fei Ma; Fei Richard Yu; Wenbo Ding
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Tactile perception is essential for embodied agents to understand physical attributes of objects that cannot be determined through visual inspection alone. While existing approaches have made progress in visual and language modalities for physical understanding, they fail to effectively incorporate tactile information that provides crucial haptic feedback for real-world interaction. In this paper, we present VTV-LLM, the first multi-modal large language model for universal Visuo-Tactile Video (VTV) understanding that bridges the gap between tactile perception and natural language. To address the challenges of cross-sensor and cross-modal integration, we contribute VTV150K, a comprehensive dataset comprising 150,000 video frames from 100 diverse objects captured across three different tactile sensors (GelSight Mini, DIGIT, and Tac3D), annotated with four fundamental tactile attributes (hardness, protrusion, elasticity, and friction). We develop a novel three-stage training paradigm that includes VTV enhancement for robust visuo-tactile representation, VTV-text alignment for cross-modal correspondence, and text prompt finetuning for natural language generation. Our framework enables sophisticated tactile reasoning capabilities including feature assessment, comparative analysis, scenario-based decision making and so on. Experimental evaluations demonstrate that VTV-LLM achieves superior performance in tactile video understanding tasks, establishing a foundation for more intuitive human-machine interaction in tactile domains.
>
---
#### [new 060] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective
- **分类: cs.CV**

- **简介: 该论文属于模型微调任务，针对SAM在专业领域表现不佳及现有方法忽略预训练知识的问题，提出InfoSAM。通过信息论中的互信息最大化，设计双目标蒸馏框架，压缩领域不变关系并增强师生模型知识传递，提升SAM在特定场景的适应性与性能。**

- **链接: [http://arxiv.org/pdf/2505.21920v1](http://arxiv.org/pdf/2505.21920v1)**

> **作者:** Yuanhong Zhang; Muyao Yuan; Weizhan Zhang; Tieliang Gong; Wen Wen; Jiangyong Ying; Weijie Shi
>
> **备注:** Accepted by ICML 2025 (Highlight)
>
> **摘要:** The Segment Anything Model (SAM), a vision foundation model, exhibits impressive zero-shot capabilities in general tasks but struggles in specialized domains. Parameter-efficient fine-tuning (PEFT) is a promising approach to unleash the potential of SAM in novel scenarios. However, existing PEFT methods for SAM neglect the domain-invariant relations encoded in the pre-trained model. To bridge this gap, we propose InfoSAM, an information-theoretic approach that enhances SAM fine-tuning by distilling and preserving its pre-trained segmentation knowledge. Specifically, we formulate the knowledge transfer process as two novel mutual information-based objectives: (i) to compress the domain-invariant relation extracted from pre-trained SAM, excluding pseudo-invariant information as possible, and (ii) to maximize mutual information between the relational knowledge learned by the teacher (pre-trained SAM) and the student (fine-tuned model). The proposed InfoSAM establishes a robust distillation framework for PEFT of SAM. Extensive experiments across diverse benchmarks validate InfoSAM's effectiveness in improving SAM family's performance on real-world tasks, demonstrating its adaptability and superiority in handling specialized scenarios.
>
---
#### [new 061] Knowledge Distillation Approach for SOS Fusion Staging: Towards Fully Automated Skeletal Maturity Assessment
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出基于知识蒸馏的SOS融合分期框架，解决骨骼成熟度评估中依赖人工裁剪和预处理的自动化难题。通过教师模型迁移空间特征至学生模型，结合新型损失函数与梯度注意力机制，实现全图直接分析，提升临床评估效率与一致性。**

- **链接: [http://arxiv.org/pdf/2505.21561v1](http://arxiv.org/pdf/2505.21561v1)**

> **作者:** Omid Halimi Milani; Amanda Nikho; Marouane Tliba; Lauren Mills; Ahmet Enis Cetin; Mohammed H Elnagar
>
> **备注:** This paper has been accepted to the CVPR Workshop 2025, to be held in Nashville, Tennessee
>
> **摘要:** We introduce a novel deep learning framework for the automated staging of spheno-occipital synchondrosis (SOS) fusion, a critical diagnostic marker in both orthodontics and forensic anthropology. Our approach leverages a dual-model architecture wherein a teacher model, trained on manually cropped images, transfers its precise spatial understanding to a student model that operates on full, uncropped images. This knowledge distillation is facilitated by a newly formulated loss function that aligns spatial logits as well as incorporates gradient-based attention spatial mapping, ensuring that the student model internalizes the anatomically relevant features without relying on external cropping or YOLO-based segmentation. By leveraging expert-curated data and feedback at each step, our framework attains robust diagnostic accuracy, culminating in a clinically viable end-to-end pipeline. This streamlined approach obviates the need for additional pre-processing tools and accelerates deployment, thereby enhancing both the efficiency and consistency of skeletal maturation assessment in diverse clinical settings.
>
---
#### [new 062] Learning Shared Representations from Unpaired Data
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文属于跨模态表示学习任务，旨在解决现有方法依赖配对数据的问题。提出仅用未配对数据构建共享嵌入空间，通过单模态随机游走矩阵的谱嵌入实现跨模态对齐。实验显示其在检索、生成等任务中有效，为首个仅用未配对数据实现通用跨模态嵌入的方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21524v1](http://arxiv.org/pdf/2505.21524v1)**

> **作者:** Amitai Yacobi; Nir Ben-Ari; Ronen Talmon; Uri Shaham
>
> **摘要:** Learning shared representations is a primary area of multimodal representation learning. The current approaches to achieve a shared embedding space rely heavily on paired samples from each modality, which are significantly harder to obtain than unpaired ones. In this work, we demonstrate that shared representations can be learned almost exclusively from unpaired data. Our arguments are grounded in the spectral embeddings of the random walk matrices constructed independently from each unimodal representation. Empirical results in computer vision and natural language processing domains support its potential, revealing the effectiveness of unpaired data in capturing meaningful cross-modal relations, demonstrating high capabilities in retrieval tasks, generation, arithmetics, zero-shot, and cross-domain classification. This work, to the best of our knowledge, is the first to demonstrate these capabilities almost exclusively from unpaired samples, giving rise to a cross-modal embedding that could be viewed as universal, i.e., independent of the specific modalities of the data. Our code IS publicly available at https://github.com/shaham-lab/SUE.
>
---
#### [new 063] Investigating Mechanisms for In-Context Vision Language Binding
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型（VLMs）的跨模态绑定机制，探究其如何关联图像与文本信息。通过合成数据集验证，发现VLMs为图像对象及其文本描述分配共享的Binding ID，实现跨模态绑定，解决图像-文本对应关系建模问题。**

- **链接: [http://arxiv.org/pdf/2505.22200v1](http://arxiv.org/pdf/2505.22200v1)**

> **作者:** Darshana Saravanan; Makarand Tapaswi; Vineet Gandhi
>
> **备注:** Accepted to MIV at CVPRW 2025 (Oral)
>
> **摘要:** To understand a prompt, Vision-Language models (VLMs) must perceive the image, comprehend the text, and build associations within and across both modalities. For instance, given an 'image of a red toy car', the model should associate this image to phrases like 'car', 'red toy', 'red object', etc. Feng and Steinhardt propose the Binding ID mechanism in LLMs, suggesting that the entity and its corresponding attribute tokens share a Binding ID in the model activations. We investigate this for image-text binding in VLMs using a synthetic dataset and task that requires models to associate 3D objects in an image with their descriptions in the text. Our experiments demonstrate that VLMs assign a distinct Binding ID to an object's image tokens and its textual references, enabling in-context association.
>
---
#### [new 064] Reference-Guided Identity Preserving Face Restoration
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于人脸修复任务，旨在解决扩散模型中身份保持不足的问题。提出三方法：Composite Context融合多层级参考信息、Hard Example Identity Loss优化身份学习、训练-free多参考适配策略，提升修复质量与身份保真度，在FFHQ-Ref等数据集达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.21905v1](http://arxiv.org/pdf/2505.21905v1)**

> **作者:** Mo Zhou; Keren Ye; Viraj Shah; Kangfu Mei; Mauricio Delbracio; Peyman Milanfar; Vishal M. Patel; Hossein Talebi
>
> **摘要:** Preserving face identity is a critical yet persistent challenge in diffusion-based image restoration. While reference faces offer a path forward, existing reference-based methods often fail to fully exploit their potential. This paper introduces a novel approach that maximizes reference face utility for improved face restoration and identity preservation. Our method makes three key contributions: 1) Composite Context, a comprehensive representation that fuses multi-level (high- and low-level) information from the reference face, offering richer guidance than prior singular representations. 2) Hard Example Identity Loss, a novel loss function that leverages the reference face to address the identity learning inefficiencies found in the existing identity loss. 3) A training-free method to adapt the model to multi-reference inputs during inference. The proposed method demonstrably restores high-quality faces and achieves state-of-the-art identity preserving restoration on benchmarks such as FFHQ-Ref and CelebA-Ref-Test, consistently outperforming previous work.
>
---
#### [new 065] On the Transferability and Discriminability of Repersentation Learning in Unsupervised Domain Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于无监督领域自适应(UDA)任务，针对传统方法仅依赖分布对齐导致目标域特征判别性不足的问题，提出RLGLC框架，结合领域对齐与目标域判别性约束，采用AR-WWD处理类别不平衡并保留局部特征细节，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22099v1](http://arxiv.org/pdf/2505.22099v1)**

> **作者:** Wenwen Qiang; Ziyin Gu; Lingyu Si; Jiangmeng Li; Changwen Zheng; Fuchun Sun; Hui Xiong
>
> **摘要:** In this paper, we addressed the limitation of relying solely on distribution alignment and source-domain empirical risk minimization in Unsupervised Domain Adaptation (UDA). Our information-theoretic analysis showed that this standard adversarial-based framework neglects the discriminability of target-domain features, leading to suboptimal performance. To bridge this theoretical-practical gap, we defined "good representation learning" as guaranteeing both transferability and discriminability, and proved that an additional loss term targeting target-domain discriminability is necessary. Building on these insights, we proposed a novel adversarial-based UDA framework that explicitly integrates a domain alignment objective with a discriminability-enhancing constraint. Instantiated as Domain-Invariant Representation Learning with Global and Local Consistency (RLGLC), our method leverages Asymmetrically-Relaxed Wasserstein of Wasserstein Distance (AR-WWD) to address class imbalance and semantic dimension weighting, and employs a local consistency mechanism to preserve fine-grained target-domain discriminative information. Extensive experiments across multiple benchmark datasets demonstrate that RLGLC consistently surpasses state-of-the-art methods, confirming the value of our theoretical perspective and underscoring the necessity of enforcing both transferability and discriminability in adversarial-based UDA.
>
---
#### [new 066] D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples
- **分类: cs.CV**

- **简介: 该论文属于扩散模型优化任务，旨在解决生成图像与文本提示对齐不足的问题。针对DPO训练中视觉差异过大的缺陷，提出D-Fusion方法：通过掩码引导自注意力融合生成视觉一致的优质样本，并保留其去噪轨迹，提升模型对齐能力。实验表明其有效提升多种强化学习算法的对齐效果。**

- **链接: [http://arxiv.org/pdf/2505.22002v1](http://arxiv.org/pdf/2505.22002v1)**

> **作者:** Zijing Hu; Fengda Zhang; Kun Kuang
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** The practical applications of diffusion models have been limited by the misalignment between generated images and corresponding text prompts. Recent studies have introduced direct preference optimization (DPO) to enhance the alignment of these models. However, the effectiveness of DPO is constrained by the issue of visual inconsistency, where the significant visual disparity between well-aligned and poorly-aligned images prevents diffusion models from identifying which factors contribute positively to alignment during fine-tuning. To address this issue, this paper introduces D-Fusion, a method to construct DPO-trainable visually consistent samples. On one hand, by performing mask-guided self-attention fusion, the resulting images are not only well-aligned, but also visually consistent with given poorly-aligned images. On the other hand, D-Fusion can retain the denoising trajectories of the resulting images, which are essential for DPO training. Extensive experiments demonstrate the effectiveness of D-Fusion in improving prompt-image alignment when applied to different reinforcement learning algorithms.
>
---
#### [new 067] AlignGen: Boosting Personalized Image Generation with Cross-Modality Prior Alignment
- **分类: cs.CV**

- **简介: 该论文属个性化图像生成任务，解决提示与参考图像不匹配时生成结果偏重文本、丢失视觉信息的问题。提出AlignGen方法，通过可学习token连接文本/视觉先验、鲁棒训练策略及选择性注意力掩码对齐跨模态先验，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2505.21911v1](http://arxiv.org/pdf/2505.21911v1)**

> **作者:** Yiheng Lin; Shifang Zhao; Ting Liu; Xiaochao Qu; Luoqi Liu; Yao Zhao; Yunchao Wei
>
> **摘要:** Personalized image generation aims to integrate user-provided concepts into text-to-image models, enabling the generation of customized content based on a given prompt. Recent zero-shot approaches, particularly those leveraging diffusion transformers, incorporate reference image information through multi-modal attention mechanism. This integration allows the generated output to be influenced by both the textual prior from the prompt and the visual prior from the reference image. However, we observe that when the prompt and reference image are misaligned, the generated results exhibit a stronger bias toward the textual prior, leading to a significant loss of reference content. To address this issue, we propose AlignGen, a Cross-Modality Prior Alignment mechanism that enhances personalized image generation by: 1) introducing a learnable token to bridge the gap between the textual and visual priors, 2) incorporating a robust training strategy to ensure proper prior alignment, and 3) employing a selective cross-modal attention mask within the multi-modal attention mechanism to further align the priors. Experimental results demonstrate that AlignGen outperforms existing zero-shot methods and even surpasses popular test-time optimization approaches.
>
---
#### [new 068] Fast Feature Matching of UAV Images via Matrix Band Reduction-based GPU Data Schedule
- **分类: cs.CV**

- **简介: 该论文提出基于矩阵带缩减的GPU数据调度算法，解决无人机图像结构从运动中特征匹配效率低的问题。通过图像分块减少冗余IO、利用GPU加速级联哈希匹配，并行优化计算，实现77-100倍加速且保持精度。**

- **链接: [http://arxiv.org/pdf/2505.22089v1](http://arxiv.org/pdf/2505.22089v1)**

> **作者:** San Jiang; Kan You; Wanshou Jiang; Qingquan Li
>
> **摘要:** Feature matching dominats the time costs in structure from motion (SfM). The primary contribution of this study is a GPU data schedule algorithm for efficient feature matching of Unmanned aerial vehicle (UAV) images. The core idea is to divide the whole dataset into blocks based on the matrix band reduction (MBR) and achieve efficient feature matching via GPU-accelerated cascade hashing. First, match pairs are selected by using an image retrieval technique, which converts images into global descriptors and searches high-dimension nearest neighbors with graph indexing. Second, compact image blocks are iteratively generated from a MBR-based data schedule strategy, which exploits image connections to avoid redundant data IO (input/output) burden and increases the usage of GPU computing power. Third, guided by the generated image blocks, feature matching is executed sequentially within the framework of GPU-accelerated cascade hashing, and initial candidate matches are refined by combining a local geometric constraint and RANSAC-based global verification. For further performance improvement, these two seps are designed to execute parallelly in GPU and CPU. Finally, the performance of the proposed solution is evaluated by using large-scale UAV datasets. The results demonstrate that it increases the efficiency of feature matching with speedup ratios ranging from 77.0 to 100.0 compared with KD-Tree based matching methods, and achieves comparable accuracy in relative and absolute bundle adjustment (BA). The proposed algorithm is an efficient solution for feature matching of UAV images.
>
---
#### [new 069] Distance Transform Guided Mixup for Alzheimer's Detection
- **分类: cs.CV**

- **简介: 该论文针对阿尔茨海默病检测中数据不平衡与多样性不足导致的模型泛化问题，提出基于距离变换引导的Mixup方法。通过分层混合MRI图像，生成结构保留的增强数据，提升单领域泛化性能，在ADNI和AIBL数据集上验证有效。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22434v1](http://arxiv.org/pdf/2505.22434v1)**

> **作者:** Zobia Batool; Huseyin Ozkan; Erchan Aptoula
>
> **摘要:** Alzheimer's detection efforts aim to develop accurate models for early disease diagnosis. Significant advances have been achieved with convolutional neural networks and vision transformer based approaches. However, medical datasets suffer heavily from class imbalance, variations in imaging protocols, and limited dataset diversity, which hinder model generalization. To overcome these challenges, this study focuses on single-domain generalization by extending the well-known mixup method. The key idea is to compute the distance transform of MRI scans, separate them spatially into multiple layers and then combine layers stemming from distinct samples to produce augmented images. The proposed approach generates diverse data while preserving the brain's structure. Experimental results show generalization performance improvement across both ADNI and AIBL datasets.
>
---
#### [new 070] HDRSDR-VQA: A Subjective Video Quality Dataset for HDR and SDR Comparative Evaluation
- **分类: cs.CV**

- **简介: 该论文属于视频质量评估任务，旨在解决HDR与SDR格式直接比较缺乏大规模数据的问题。构建了含960个视频（54组源，9级失真）的HDRSDR-VQA数据集，通过145名参与者在6台HDR电视上完成22,000次主观比较，生成JOD分数，支持动态范围格式的优劣分析及模型开发。**

- **链接: [http://arxiv.org/pdf/2505.21831v1](http://arxiv.org/pdf/2505.21831v1)**

> **作者:** Bowen Chen; Cheng-han Lee; Yixu Chen; Zaixi Shang; Hai Wei; Alan C. Bovik
>
> **摘要:** We introduce HDRSDR-VQA, a large-scale video quality assessment dataset designed to facilitate comparative analysis between High Dynamic Range (HDR) and Standard Dynamic Range (SDR) content under realistic viewing conditions. The dataset comprises 960 videos generated from 54 diverse source sequences, each presented in both HDR and SDR formats across nine distortion levels. To obtain reliable perceptual quality scores, we conducted a comprehensive subjective study involving 145 participants and six consumer-grade HDR-capable televisions. A total of over 22,000 pairwise comparisons were collected and scaled into Just-Objectionable-Difference (JOD) scores. Unlike prior datasets that focus on a single dynamic range format or use limited evaluation protocols, HDRSDR-VQA enables direct content-level comparison between HDR and SDR versions, supporting detailed investigations into when and why one format is preferred over the other. The open-sourced part of the dataset is publicly available to support further research in video quality assessment, content-adaptive streaming, and perceptual model development.
>
---
#### [new 071] MultiFormer: A Multi-Person Pose Estimation System Based on CSI and Attention Mechanism
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于基于CSI的多人姿态估计任务，旨在解决多目标识别不准确及CSI特征学习效果差的问题。提出MultiFormer系统，采用Transformer的时频双token特征提取器建模CSI的子载波相关性和时序依赖，并通过多阶段特征融合网络（MSFN）强化解剖约束，实验显示其在高移动性关键点（手腕、肘部）的精度优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22555v1](http://arxiv.org/pdf/2505.22555v1)**

> **作者:** Yanyi Qu; Haoyang Ma; Wenhui Xiong
>
> **摘要:** Human pose estimation based on Channel State Information (CSI) has emerged as a promising approach for non-intrusive and precise human activity monitoring, yet faces challenges including accurate multi-person pose recognition and effective CSI feature learning. This paper presents MultiFormer, a wireless sensing system that accurately estimates human pose through CSI. The proposed system adopts a Transformer based time-frequency dual-token feature extractor with multi-head self-attention. This feature extractor is able to model inter-subcarrier correlations and temporal dependencies of the CSI. The extracted CSI features and the pose probability heatmaps are then fused by Multi-Stage Feature Fusion Network (MSFN) to enforce the anatomical constraints. Extensive experiments conducted on on the public MM-Fi dataset and our self-collected dataset show that the MultiFormer achieves higher accuracy over state-of-the-art approaches, especially for high-mobility keypoints (wrists, elbows) that are particularly difficult for previous methods to accurately estimate.
>
---
#### [new 072] ObjectClear: Complete Object Removal via Object-Effect Attention
- **分类: cs.CV**

- **简介: 该论文属于物体完全移除任务，针对现有方法难以精准消除物体及其阴影、反射等效果且易产生伪影和背景失真的问题，提出OBER数据集与ObjectClear框架。通过对象-效果注意力机制分离前景去除与背景重建，并采用注意力引导融合策略，提升复杂场景下物体效果去除质量与背景保真度。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22636v1](http://arxiv.org/pdf/2505.22636v1)**

> **作者:** Jixin Zhao; Shangchen Zhou; Zhouxia Wang; Peiqing Yang; Chen Change Loy
>
> **备注:** Project page: https://zjx0101.github.io/projects/ObjectClear/
>
> **摘要:** Object removal requires eliminating not only the target object but also its effects, such as shadows and reflections. However, diffusion-based inpainting methods often produce artifacts, hallucinate content, alter background, and struggle to remove object effects accurately. To address this challenge, we introduce a new dataset for OBject-Effect Removal, named OBER, which provides paired images with and without object effects, along with precise masks for both objects and their associated visual artifacts. The dataset comprises high-quality captured and simulated data, covering diverse object categories and complex multi-object scenes. Building on OBER, we propose a novel framework, ObjectClear, which incorporates an object-effect attention mechanism to guide the model toward the foreground removal regions by learning attention masks, effectively decoupling foreground removal from background reconstruction. Furthermore, the predicted attention map enables an attention-guided fusion strategy during inference, greatly preserving background details. Extensive experiments demonstrate that ObjectClear outperforms existing methods, achieving improved object-effect removal quality and background fidelity, especially in complex scenarios.
>
---
#### [new 073] How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文评估大型语言模型（LLMs）在3D虚拟形象运动控制中的能力，研究其对人类运动知识的掌握。通过设计20个动作指令，测试LLMs的高阶动作规划（如分步骤分解动作）与低阶身体部位定位能力，发现其擅长理解高层次动作但难以精确控制高自由度身体部位，适合创意动作设计但不适用于精准时空参数生成。**

- **链接: [http://arxiv.org/pdf/2505.21531v1](http://arxiv.org/pdf/2505.21531v1)**

> **作者:** Kunhang Li; Jason Naradowsky; Yansong Feng; Yusuke Miyao
>
> **摘要:** We explore Large Language Models (LLMs)' human motion knowledge through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations as a clear verification lens for human evaluators. Through carefully designed 20 representative motion instructions with full coverage of basic movement primitives and balanced body part usage, we conduct comprehensive evaluations including human assessment of both generated animations and high-level movement plans, as well as automatic comparison with oracle positions in low-level planning. We find that LLMs are strong at interpreting the high-level body movements but struggle with precise body part positioning. While breaking down motion queries into atomic components improves planning performance, LLMs have difficulty with multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximation for general spatial descriptions, but fail to handle precise spatial specifications in text, and the precise spatial-temporal parameters needed for avatar control. Notably, LLMs show promise in conceptualizing creative motions and distinguishing culturally-specific motion patterns.
>
---
#### [new 074] A Survey on Training-free Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于无训练的开放词汇语义分割任务，旨在解决传统方法依赖大量标注数据且无法处理未知类别的问题。通过综述30余种方法，将其分为CLIP基、视觉基础模型辅助及生成方法三类，分析技术路线、局限性并提出未来方向，为新研究者提供入门指南。**

- **链接: [http://arxiv.org/pdf/2505.22209v1](http://arxiv.org/pdf/2505.22209v1)**

> **作者:** Naomi Kombol; Ivan Martinović; Siniša Šegvić
>
> **摘要:** Semantic segmentation is one of the most fundamental tasks in image understanding with a long history of research, and subsequently a myriad of different approaches. Traditional methods strive to train models up from scratch, requiring vast amounts of computational resources and training data. In the advent of moving to open-vocabulary semantic segmentation, which asks models to classify beyond learned categories, large quantities of finely annotated data would be prohibitively expensive. Researchers have instead turned to training-free methods where they leverage existing models made for tasks where data is more easily acquired. Specifically, this survey will cover the history, nuance, idea development and the state-of-the-art in training-free open-vocabulary semantic segmentation that leverages existing multi-modal classification models. We will first give a preliminary on the task definition followed by an overview of popular model archetypes and then spotlight over 30 approaches split into broader research branches: purely CLIP-based, those leveraging auxiliary visual foundation models and ones relying on generative methods. Subsequently, we will discuss the limitations and potential problems of current research, as well as provide some underexplored ideas for future study. We believe this survey will serve as a good onboarding read to new researchers and spark increased interest in the area.
>
---
#### [new 075] Tell me Habibi, is it Real or Fake?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多语言深度伪造检测任务，针对现有方法在处理阿拉伯-英语混合语言及方言时的不足，构建了首个包含38.7万视频的ArEnAV数据集，集成文本到语音和唇动模型，通过基准测试推动多模态检测研究。**

- **链接: [http://arxiv.org/pdf/2505.22581v1](http://arxiv.org/pdf/2505.22581v1)**

> **作者:** Kartik Kuckreja; Parul Gupta; Injy Hamed; Thamar Solorio; Muhammad Haris Khan; Abhinav Dhall
>
> **备注:** 9 pages, 2 figures, 12 tables
>
> **摘要:** Deepfake generation methods are evolving fast, making fake media harder to detect and raising serious societal concerns. Most deepfake detection and dataset creation research focuses on monolingual content, often overlooking the challenges of multilingual and code-switched speech, where multiple languages are mixed within the same discourse. Code-switching, especially between Arabic and English, is common in the Arab world and is widely used in digital communication. This linguistic mixing poses extra challenges for deepfake detection, as it can confuse models trained mostly on monolingual data. To address this, we introduce \textbf{ArEnAV}, the first large-scale Arabic-English audio-visual deepfake dataset featuring intra-utterance code-switching, dialectal variation, and monolingual Arabic content. It \textbf{contains 387k videos and over 765 hours of real and fake videos}. Our dataset is generated using a novel pipeline integrating four Text-To-Speech and two lip-sync models, enabling comprehensive analysis of multilingual multimodal deepfake detection. We benchmark our dataset against existing monolingual and multilingual datasets, state-of-the-art deepfake detection models, and a human evaluation, highlighting its potential to advance deepfake research. The dataset can be accessed \href{https://huggingface.co/datasets/kartik060702/ArEnAV-Full}{here}.
>
---
#### [new 076] A Novel Convolutional Neural Network-Based Framework for Complex Multiclass Brassica Seed Classification
- **分类: cs.CV; cs.AI; cs.LG; na**

- **简介: 该论文提出基于CNN的油菜种子分类框架，解决多类种子因纹理相似导致的分类难题。通过定制网络架构优化预训练模型，实现10类种子识别，在自建数据集上达93%准确率，提升种子质量监控与农业生产效率。**

- **链接: [http://arxiv.org/pdf/2505.21558v1](http://arxiv.org/pdf/2505.21558v1)**

> **作者:** Elhoucine Elfatimia; Recep Eryigitb; Lahcen Elfatimi
>
> **备注:** 11 Figure
>
> **摘要:** Agricultural research has accelerated in recent years, yet farmers often lack the time and resources for on-farm research due to the demands of crop production and farm operations. Seed classification offers valuable insights into quality control, production efficiency, and impurity detection. Early identification of seed types is critical to reducing the cost and risk associated with field emergence, which can lead to yield losses or disruptions in downstream processes like harvesting. Seed sampling supports growers in monitoring and managing seed quality, improving precision in determining seed purity levels, guiding management adjustments, and enhancing yield estimations. This study proposes a novel convolutional neural network (CNN)-based framework for the efficient classification of ten common Brassica seed types. The approach addresses the inherent challenge of texture similarity in seed images using a custom-designed CNN architecture. The model's performance was evaluated against several pre-trained state-of-the-art architectures, with adjustments to layer configurations for optimized classification. Experimental results using our collected Brassica seed dataset demonstrate that the proposed model achieved a high accuracy rate of 93 percent.
>
---
#### [new 077] Multi-instance Learning as Downstream Task of Self-Supervised Learning-based Pre-trained Model
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多实例学习任务，针对脑出血CT图像中bag实例数过多（如256）导致模型性能下降的问题，提出以自监督学习预训练模型作为下游任务，缓解假相关问题，提升分类准确率5%-13%及F1值40%-55%。**

- **链接: [http://arxiv.org/pdf/2505.21564v1](http://arxiv.org/pdf/2505.21564v1)**

> **作者:** Koki Matsuishi; Tsuyoshi Okita
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** In deep multi-instance learning, the number of applicable instances depends on the data set. In histopathology images, deep learning multi-instance learners usually assume there are hundreds to thousands instances in a bag. However, when the number of instances in a bag increases to 256 in brain hematoma CT, learning becomes extremely difficult. In this paper, we address this drawback. To overcome this problem, we propose using a pre-trained model with self-supervised learning for the multi-instance learner as a downstream task. With this method, even when the original target task suffers from the spurious correlation problem, we show improvements of 5% to 13% in accuracy and 40% to 55% in the F1 measure for the hypodensity marker classification of brain hematoma CT.
>
---
#### [new 078] Enjoying Information Dividend: Gaze Track-based Medical Weakly Supervised Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像弱监督分割任务，针对现有方法未充分挖掘眼动数据信息的问题，提出GradTrack框架。其利用医生眼动轨迹的注视点、时序及持续时间，通过眼动图生成与轨迹注意力机制实现多级监督，提升特征优化，实验显示Dice分数提升并缩小与全监督模型的差距。**

- **链接: [http://arxiv.org/pdf/2505.22230v1](http://arxiv.org/pdf/2505.22230v1)**

> **作者:** Zhisong Wang; Yiwen Ye; Ziyang Chen; Yong Xia
>
> **备注:** 10 pages, 4 figures, MICCAI 2025 (Early Accept)
>
> **摘要:** Weakly supervised semantic segmentation (WSSS) in medical imaging struggles with effectively using sparse annotations. One promising direction for WSSS leverages gaze annotations, captured via eye trackers that record regions of interest during diagnostic procedures. However, existing gaze-based methods, such as GazeMedSeg, do not fully exploit the rich information embedded in gaze data. In this paper, we propose GradTrack, a framework that utilizes physicians' gaze track, including fixation points, durations, and temporal order, to enhance WSSS performance. GradTrack comprises two key components: Gaze Track Map Generation and Track Attention, which collaboratively enable progressive feature refinement through multi-level gaze supervision during the decoding process. Experiments on the Kvasir-SEG and NCI-ISBI datasets demonstrate that GradTrack consistently outperforms existing gaze-based methods, achieving Dice score improvements of 3.21\% and 2.61\%, respectively. Moreover, GradTrack significantly narrows the performance gap with fully supervised models such as nnUNet.
>
---
#### [new 079] Analytical Calculation of Weights Convolutional Neural Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出无需训练的CNN参数分析计算方法，解决传统CNN依赖训练数据的问题。通过10张MNIST样本图像，推导权重、阈值及通道数，实现快速分类。实验显示其模型可识别超半数测试图像，且推理高效。**

- **链接: [http://arxiv.org/pdf/2505.21557v1](http://arxiv.org/pdf/2505.21557v1)**

> **作者:** Polad Geidarov
>
> **摘要:** This paper presents an algorithm for analytically calculating the weights and thresholds of convolutional neural networks (CNNs) without using standard training procedures. The algorithm enables the determination of CNN parameters based on just 10 selected images from the MNIST dataset, each representing a digit from 0 to 9. As part of the method, the number of channels in CNN layers is also derived analytically. A software module was implemented in C++ Builder, and a series of experiments were conducted using the MNIST dataset. Results demonstrate that the analytically computed CNN can recognize over half of 1000 handwritten digit images without any training, achieving inference in fractions of a second. These findings suggest that CNNs can be constructed and applied directly for classification tasks without training, using purely analytical computation of weights.
>
---
#### [new 080] Image Tokens Matter: Mitigating Hallucination in Discrete Tokenizer-based Large Vision-Language Models via Latent Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于减少视觉语言模型幻觉的任务。针对离散图像分词器LVLMs因token共现关联而虚构物体的问题，作者构建共现图并用GNN聚类分析，发现幻觉与缺失token簇相关，提出潜隐编辑方法抑制其影响，减少幻觉同时保持模型表达力。**

- **链接: [http://arxiv.org/pdf/2505.21547v1](http://arxiv.org/pdf/2505.21547v1)**

> **作者:** Weixing Wang; Zifeng Ding; Jindong Gu; Rui Cao; Christoph Meinel; Gerard de Melo; Haojin Yang
>
> **摘要:** Large Vision-Language Models (LVLMs) with discrete image tokenizers unify multimodal representations by encoding visual inputs into a finite set of tokens. Despite their effectiveness, we find that these models still hallucinate non-existent objects. We hypothesize that this may be due to visual priors induced during training: When certain image tokens frequently co-occur in the same spatial regions and represent shared objects, they become strongly associated with the verbalizations of those objects. As a result, the model may hallucinate by evoking visually absent tokens that often co-occur with present ones. To test this assumption, we construct a co-occurrence graph of image tokens using a segmentation dataset and employ a Graph Neural Network (GNN) with contrastive learning followed by a clustering method to group tokens that frequently co-occur in similar visual contexts. We find that hallucinations predominantly correspond to clusters whose tokens dominate the input, and more specifically, that the visually absent tokens in those clusters show much higher correlation with hallucinated objects compared to tokens present in the image. Based on this observation, we propose a hallucination mitigation method that suppresses the influence of visually absent tokens by modifying latent image embeddings during generation. Experiments show our method reduces hallucinations while preserving expressivity. Code is available at https://github.com/weixingW/CGC-VTD/tree/main
>
---
#### [new 081] Fostering Video Reasoning via Next-Event Prediction
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出通过预测下一事件（NEP）提升视频时间推理。针对现有任务依赖标注或混杂空间信息的问题，该工作将视频分为过去和未来帧，训练模型预测未来事件总结，并构建V1-33K数据集及评估基准FutureBench，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22457v1](http://arxiv.org/pdf/2505.22457v1)**

> **作者:** Haonan Wang; Hongfu Liu; Xiangyan Liu; Chao Du; Kenji Kawaguchi; Ye Wang; Tianyu Pang
>
> **摘要:** Next-token prediction serves as the foundational learning task enabling reasoning in LLMs. But what should the learning task be when aiming to equip MLLMs with temporal reasoning capabilities over video inputs? Existing tasks such as video question answering often rely on annotations from humans or much stronger MLLMs, while video captioning tends to entangle temporal reasoning with spatial information. To address this gap, we propose next-event prediction (NEP), a learning task that harnesses future video segments as a rich, self-supervised signal to foster temporal reasoning. We segment each video into past and future frames: the MLLM takes the past frames as input and predicts a summary of events derived from the future frames, thereby encouraging the model to reason temporally in order to complete the task. To support this task, we curate V1-33K, a dataset comprising 33,000 automatically extracted video segments spanning diverse real-world scenarios. We further explore a range of video instruction-tuning strategies to study their effects on temporal reasoning. To evaluate progress, we introduce FutureBench to assess coherence in predicting unseen future events. Experiments validate that NEP offers a scalable and effective training paradigm for fostering temporal reasoning in MLLMs.
>
---
#### [new 082] GL-PGENet: A Parameterized Generation Framework for Robust Document Image Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文档图像增强任务，旨在解决现有方法仅适用于单退化或灰度图像的问题。提出GL-PGENet，通过分层增强框架、双分支参数生成网络及改进的NestUNet结构，结合两阶段训练策略，提升多退化彩色文档图像处理的鲁棒性与效率，实现跨领域适应和高分辨率计算效率。**

- **链接: [http://arxiv.org/pdf/2505.22021v1](http://arxiv.org/pdf/2505.22021v1)**

> **作者:** Zhihong Tang; Yang Li
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Document Image Enhancement (DIE) serves as a critical component in Document AI systems, where its performance substantially determines the effectiveness of downstream tasks. To address the limitations of existing methods confined to single-degradation restoration or grayscale image processing, we present Global with Local Parametric Generation Enhancement Network (GL-PGENet), a novel architecture designed for multi-degraded color document images, ensuring both efficiency and robustness in real-world scenarios. Our solution incorporates three key innovations: First, a hierarchical enhancement framework that integrates global appearance correction with local refinement, enabling coarse-to-fine quality improvement. Second, a Dual-Branch Local-Refine Network with parametric generation mechanisms that replaces conventional direct prediction, producing enhanced outputs through learned intermediate parametric representations rather than pixel-wise mapping. This approach enhances local consistency while improving model generalization. Finally, a modified NestUNet architecture incorporating dense block to effectively fuse low-level pixel features and high-level semantic features, specifically adapted for document image characteristics. In addition, to enhance generalization performance, we adopt a two-stage training strategy: large-scale pretraining on a synthetic dataset of 500,000+ samples followed by task-specific fine-tuning. Extensive experiments demonstrate the superiority of GL-PGENet, achieving state-of-the-art SSIM scores of 0.7721 on DocUNet and 0.9480 on RealDAE. The model also exhibits remarkable cross-domain adaptability and maintains computational efficiency for high-resolution images without performance degradation, confirming its practical utility in real-world scenarios.
>
---
#### [new 083] Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇语义分割（OVSS）的测试时适应（TTA）任务，解决其在密集预测中缺乏有效TTA方法的问题。提出多层级多提示（MLMP）熵最小化方法，整合视觉编码器中间层特征，采用多文本提示模板在全局和像素级优化，实现即插即用且无需额外数据，同时建立包含多数据集的基准测试套件验证效果。**

- **链接: [http://arxiv.org/pdf/2505.21844v1](http://arxiv.org/pdf/2505.21844v1)**

> **作者:** Mehrdad Noori; David Osowiechi; Gustavo Adolfo Vargas Hakim; Ali Bahri; Moslem Yazdanpanah; Sahar Dastani; Farzad Beizaee; Ismail Ben Ayed; Christian Desrosiers
>
> **摘要:** Recently, test-time adaptation has attracted wide interest in the context of vision-language models for image classification. However, to the best of our knowledge, the problem is completely overlooked in dense prediction tasks such as Open-Vocabulary Semantic Segmentation (OVSS). In response, we propose a novel TTA method tailored to adapting VLMs for segmentation during test time. Unlike TTA methods for image classification, our Multi-Level and Multi-Prompt (MLMP) entropy minimization integrates features from intermediate vision-encoder layers and is performed with different text-prompt templates at both the global CLS token and local pixel-wise levels. Our approach could be used as plug-and-play for any segmentation network, does not require additional training data or labels, and remains effective even with a single test sample. Furthermore, we introduce a comprehensive OVSS TTA benchmark suite, which integrates a rigorous evaluation protocol, seven segmentation datasets, and 15 common corruptions, with a total of 82 distinct test scenarios, establishing a standardized and comprehensive testbed for future TTA research in open-vocabulary segmentation. Our experiments on this suite demonstrate that our segmentation-tailored method consistently delivers significant gains over direct adoption of TTA classification baselines.
>
---
#### [new 084] Autoregression-free video prediction using diffusion model for mitigating error propagation
- **分类: cs.CV**

- **简介: 该论文属于视频预测任务，旨在解决自回归模型长期预测中的误差传播问题。提出无自回归框架ARFree，利用扩散模型直接预测未来帧，包含运动预测模块与提升连续性的训练方法，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22111v1](http://arxiv.org/pdf/2505.22111v1)**

> **作者:** Woonho Ko; Jin Bok Park; Il Yong Chun
>
> **备注:** 6 pages, 4 figures, 2 tables
>
> **摘要:** Existing long-term video prediction methods often rely on an autoregressive video prediction mechanism. However, this approach suffers from error propagation, particularly in distant future frames. To address this limitation, this paper proposes the first AutoRegression-Free (ARFree) video prediction framework using diffusion models. Different from an autoregressive video prediction mechanism, ARFree directly predicts any future frame tuples from the context frame tuple. The proposed ARFree consists of two key components: 1) a motion prediction module that predicts a future motion using motion feature extracted from the context frame tuple; 2) a training method that improves motion continuity and contextual consistency between adjacent future frame tuples. Our experiments with two benchmark datasets show that the proposed ARFree video prediction framework outperforms several state-of-the-art video prediction methods.
>
---
#### [new 085] SHTOcc: Effective 3D Occupancy Prediction with Sparse Head and Tail Voxels
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶领域的3D占用预测任务，旨在解决体素类间长尾分布与几何分布导致的性能不足问题。提出SHTOcc方法，通过稀疏头尾体素构造平衡关键体素，并采用解耦学习减少类别偏差，提升精度同时降低42.2%内存与提升58.6%推理速度。**

- **链接: [http://arxiv.org/pdf/2505.22461v1](http://arxiv.org/pdf/2505.22461v1)**

> **作者:** Qiucheng Yu; Yuan Xie; Xin Tan
>
> **摘要:** 3D occupancy prediction has attracted much attention in the field of autonomous driving due to its powerful geometric perception and object recognition capabilities. However, existing methods have not explored the most essential distribution patterns of voxels, resulting in unsatisfactory results. This paper first explores the inter-class distribution and geometric distribution of voxels, thereby solving the long-tail problem caused by the inter-class distribution and the poor performance caused by the geometric distribution. Specifically, this paper proposes SHTOcc (Sparse Head-Tail Occupancy), which uses sparse head-tail voxel construction to accurately identify and balance key voxels in the head and tail classes, while using decoupled learning to reduce the model's bias towards the dominant (head) category and enhance the focus on the tail class. Experiments show that significant improvements have been made on multiple baselines: SHTOcc reduces GPU memory usage by 42.2%, increases inference speed by 58.6%, and improves accuracy by about 7%, verifying its effectiveness and efficiency. The code is available at https://github.com/ge95net/SHTOcc
>
---
#### [new 086] A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding
- **分类: cs.CV**

- **简介: 该论文属于无人机视角下的异常检测与理解任务，旨在解决现有方法在动态场景、尺度变化和复杂环境中的性能不足问题。提出A2Seek数据集（含多场景标注）及A2Seek-R1框架，通过图思维引导、优化策略和注意力机制提升异常定位与因果推理能力，显著提高检测精度和泛化性。**

- **链接: [http://arxiv.org/pdf/2505.21962v1](http://arxiv.org/pdf/2505.21962v1)**

> **作者:** Mengjingcheng Mo; Xinyang Tong; Jiaxu Leng; Mingpi Tan; Jiankang Zheng; Yiran Liu; Haosheng Chen; Ji Gan; Weisheng Li; Xinbo Gao
>
> **摘要:** While unmanned aerial vehicles (UAVs) offer wide-area, high-altitude coverage for anomaly detection, they face challenges such as dynamic viewpoints, scale variations, and complex scenes. Existing datasets and methods, mainly designed for fixed ground-level views, struggle to adapt to these conditions, leading to significant performance drops in drone-view scenarios. To bridge this gap, we introduce A2Seek (Aerial Anomaly Seek), a large-scale, reasoning-centric benchmark dataset for aerial anomaly understanding. This dataset covers various scenarios and environmental conditions, providing high-resolution real-world aerial videos with detailed annotations, including anomaly categories, frame-level timestamps, region-level bounding boxes, and natural language explanations for causal reasoning. Building on this dataset, we propose A2Seek-R1, a novel reasoning framework that generalizes R1-style strategies to aerial anomaly understanding, enabling a deeper understanding of "Where" anomalies occur and "Why" they happen in aerial frames. To this end, A2Seek-R1 first employs a graph-of-thought (GoT)-guided supervised fine-tuning approach to activate the model's latent reasoning capabilities on A2Seek. Then, we introduce Aerial Group Relative Policy Optimization (A-GRPO) to design rule-based reward functions tailored to aerial scenarios. Furthermore, we propose a novel "seeking" mechanism that simulates UAV flight behavior by directing the model's attention to informative regions. Extensive experiments demonstrate that A2Seek-R1 achieves up to a 22.04% improvement in AP for prediction accuracy and a 13.9% gain in mIoU for anomaly localization, exhibiting strong generalization across complex environments and out-of-distribution scenarios. Our dataset and code will be released at https://hayneyday.github.io/A2Seek/.
>
---
#### [new 087] Visual Loop Closure Detection Through Deep Graph Consensus
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LoopGNN，通过图神经网络分析多关键帧邻域的视觉相似性，解决传统视觉环闭检测计算成本高、误报多的问题，提升SLAM场景下的精度与效率，实验显示优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.21754v1](http://arxiv.org/pdf/2505.21754v1)**

> **作者:** Martin Büchner; Liza Dahiya; Simon Dorer; Vipul Ramtekkar; Kenji Nishimiya; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Visual loop closure detection traditionally relies on place recognition methods to retrieve candidate loops that are validated using computationally expensive RANSAC-based geometric verification. As false positive loop closures significantly degrade downstream pose graph estimates, verifying a large number of candidates in online simultaneous localization and mapping scenarios is constrained by limited time and compute resources. While most deep loop closure detection approaches only operate on pairs of keyframes, we relax this constraint by considering neighborhoods of multiple keyframes when detecting loops. In this work, we introduce LoopGNN, a graph neural network architecture that estimates loop closure consensus by leveraging cliques of visually similar keyframes retrieved through place recognition. By propagating deep feature encodings among nodes of the clique, our method yields high-precision estimates while maintaining high recall. Extensive experimental evaluations on the TartanDrive 2.0 and NCLT datasets demonstrate that LoopGNN outperforms traditional baselines. Additionally, an ablation study across various keypoint extractors demonstrates that our method is robust, regardless of the type of deep feature encodings used, and exhibits higher computational efficiency compared to classical geometric verification baselines. We release our code, supplementary material, and keyframe data at https://loopgnn.cs.uni-freiburg.de.
>
---
#### [new 088] Real-Time Blind Defocus Deblurring for Earth Observation: The IMAGIN-e Mission Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种实时盲散焦去模糊方法，解决国际空间站IMAGIN-e任务中地球观测图像的机械散焦问题。基于GAN框架，利用Sentinel-2数据估计模糊核并训练模型，无需参考图像，在合成数据提升SSIM/PSNR，真实数据优化感知指标，已部署实现资源受限下的图像复原与应用。**

- **链接: [http://arxiv.org/pdf/2505.22128v1](http://arxiv.org/pdf/2505.22128v1)**

> **作者:** Alejandro D. Mousist
>
> **摘要:** This work addresses mechanical defocus in Earth observation images from the IMAGIN-e mission aboard the ISS, proposing a blind deblurring approach adapted to space-based edge computing constraints. Leveraging Sentinel-2 data, our method estimates the defocus kernel and trains a restoration model within a GAN framework, effectively operating without reference images. On Sentinel-2 images with synthetic degradation, SSIM improved by 72.47% and PSNR by 25.00%, confirming the model's ability to recover lost details when the original clean image is known. On IMAGIN-e, where no reference images exist, perceptual quality metrics indicate a substantial enhancement, with NIQE improving by 60.66% and BRISQUE by 48.38%, validating real-world onboard restoration. The approach is currently deployed aboard the IMAGIN-e mission, demonstrating its practical application in an operational space environment. By efficiently handling high-resolution images under edge computing constraints, the method enables applications such as water body segmentation and contour detection while maintaining processing viability despite resource limitations.
>
---
#### [new 089] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D场景分割任务，解决参数高效微调（PEFT）在3D点云模型中因忽视几何结构导致效果差的问题。提出GEM模块，融合局部位置编码与轻量注意力机制，捕捉几何上下文，仅更新1.6%参数即达全量微调效果，提升效率。**

- **链接: [http://arxiv.org/pdf/2505.22444v1](http://arxiv.org/pdf/2505.22444v1)**

> **作者:** Liyao Tang; Zhe Chen; Dacheng Tao
>
> **摘要:** The emergence of large-scale pre-trained point cloud models has significantly advanced 3D scene understanding, but adapting these models to specific downstream tasks typically demands full fine-tuning, incurring high computational and storage costs. Parameter-efficient fine-tuning (PEFT) techniques, successful in natural language processing and 2D vision tasks, would underperform when naively applied to 3D point cloud models due to significant geometric and spatial distribution shifts. Existing PEFT methods commonly treat points as orderless tokens, neglecting important local spatial structures and global geometric contexts in 3D modeling. To bridge this gap, we introduce the Geometric Encoding Mixer (GEM), a novel geometry-aware PEFT module specifically designed for 3D point cloud transformers. GEM explicitly integrates fine-grained local positional encodings with a lightweight latent attention mechanism to capture comprehensive global context, thereby effectively addressing the spatial and geometric distribution mismatch. Extensive experiments demonstrate that GEM achieves performance comparable to or sometimes even exceeding full fine-tuning, while only updating 1.6% of the model's parameters, fewer than other PEFT methods. With significantly reduced training time and memory requirements, our approach thus sets a new benchmark for efficient, scalable, and geometry-aware fine-tuning of large-scale 3D point cloud models. Code will be released.
>
---
#### [new 090] Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting
- **分类: cs.CV**

- **简介: 该论文提出点到区域(P2R)损失，用于半监督基于点的人群计数任务。针对传统P2P监督需大量标注及伪标签置信度无法传播背景的问题，通过设计点特异性激活图(PSAM)发现特征图过激活现象，进而用区域监督替代点监督，使区域像素共享置信度，提升半监督训练效果。**

- **链接: [http://arxiv.org/pdf/2505.21943v1](http://arxiv.org/pdf/2505.21943v1)**

> **作者:** Wei Lin; Chenyang Zhao; Antoni B. Chan
>
> **备注:** accepted by CVPR-2025(highlight)
>
> **摘要:** Point detection has been developed to locate pedestrians in crowded scenes by training a counter through a point-to-point (P2P) supervision scheme. Despite its excellent localization and counting performance, training a point-based counter still faces challenges concerning annotation labor: hundreds to thousands of points are required to annotate a single sample capturing a dense crowd. In this paper, we integrate point-based methods into a semi-supervised counting framework based on pseudo-labeling, enabling the training of a counter with only a few annotated samples supplemented by a large volume of pseudo-labeled data. However, during implementation, the training encounters issues as the confidence for pseudo-labels fails to be propagated to background pixels via the P2P. To tackle this challenge, we devise a point-specific activation map (PSAM) to visually interpret the phenomena occurring during the ill-posed training. Observations from the PSAM suggest that the feature map is excessively activated by the loss for unlabeled data, causing the decoder to misinterpret these over-activations as pedestrians. To mitigate this issue, we propose a point-to-region (P2R) scheme to substitute P2P, which segments out local regions rather than detects a point corresponding to a pedestrian for supervision. Consequently, pixels in the local region can share the same confidence with the corresponding pseudo points. Experimental results in both semi-supervised counting and unsupervised domain adaptation highlight the advantages of our method, illustrating P2R can resolve issues identified in PSAM. The code is available at https://github.com/Elin24/P2RLoss.
>
---
#### [new 091] PS4PRO: Pixel-to-pixel Supervision for Photorealistic Rendering and Optimization
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于神经渲染任务，旨在解决输入视图不足导致的3D重建质量限制，尤其在复杂动态场景中。提出PS4PRO模型，通过视频帧插值增强数据，隐式建模相机运动与3D几何，提升照片级监督，从而优化重建效果，在动静态场景中均取得改进。**

- **链接: [http://arxiv.org/pdf/2505.22616v1](http://arxiv.org/pdf/2505.22616v1)**

> **作者:** Yezhi Shen; Qiuchen Zhai; Fengqing Zhu
>
> **备注:** Accepted to the CVPR 2025 Workshop on Autonomous Driving (WAD)
>
> **摘要:** Neural rendering methods have gained significant attention for their ability to reconstruct 3D scenes from 2D images. The core idea is to take multiple views as input and optimize the reconstructed scene by minimizing the uncertainty in geometry and appearance across the views. However, the reconstruction quality is limited by the number of input views. This limitation is further pronounced in complex and dynamic scenes, where certain angles of objects are never seen. In this paper, we propose to use video frame interpolation as the data augmentation method for neural rendering. Furthermore, we design a lightweight yet high-quality video frame interpolation model, PS4PRO (Pixel-to-pixel Supervision for Photorealistic Rendering and Optimization). PS4PRO is trained on diverse video datasets, implicitly modeling camera movement as well as real-world 3D geometry. Our model performs as an implicit world prior, enriching the photo supervision for 3D reconstruction. By leveraging the proposed method, we effectively augment existing datasets for neural rendering methods. Our experimental results indicate that our method improves the reconstruction performance on both static and dynamic scenes.
>
---
#### [new 092] IKIWISI: An Interactive Visual Pattern Generator for Evaluating the Reliability of Vision-Language Models Without Ground Truth
- **分类: cs.CV**

- **简介: 论文提出IKIWISI工具，用于无ground truth时评估视觉语言模型的可靠性。通过生成二进制热图（绿/红标记物体存在/缺失）和引入"spy objects"（已知缺失的对抗实例），辅助人类识别模型幻觉。实验表明用户可快速有效评估模型，方法补充传统评估并促进人机理解对齐。**

- **链接: [http://arxiv.org/pdf/2505.22305v1](http://arxiv.org/pdf/2505.22305v1)**

> **作者:** Md Touhidul Islam; Imran Kabir; Md Alimoor Reza; Syed Masum Billah
>
> **备注:** Accepted at DIS'25 (Funchal, Portugal)
>
> **摘要:** We present IKIWISI ("I Know It When I See It"), an interactive visual pattern generator for assessing vision-language models in video object recognition when ground truth is unavailable. IKIWISI transforms model outputs into a binary heatmap where green cells indicate object presence and red cells indicate object absence. This visualization leverages humans' innate pattern recognition abilities to evaluate model reliability. IKIWISI introduces "spy objects": adversarial instances users know are absent, to discern models hallucinating on nonexistent items. The tool functions as a cognitive audit mechanism, surfacing mismatches between human and machine perception by visualizing where models diverge from human understanding. Our study with 15 participants found that users considered IKIWISI easy to use, made assessments that correlated with objective metrics when available, and reached informed conclusions by examining only a small fraction of heatmap cells. This approach not only complements traditional evaluation methods through visual assessment of model behavior with custom object sets, but also reveals opportunities for improving alignment between human perception and machine understanding in vision-language systems.
>
---
#### [new 093] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型（LVLM）压缩任务，旨在解决现有图像token剪枝方法仅关注局部影响导致效率低下的问题。提出平衡剪枝（BTP）方法，通过分阶段校准全局层间影响与局部输出一致性，实现78%压缩率同时保持96.7%原始性能。**

- **链接: [http://arxiv.org/pdf/2505.22038v1](http://arxiv.org/pdf/2505.22038v1)**

> **作者:** Kaiyuan Li; Xiaoyue Chen; Chen Gao; Yong Li; Xinlei Chen
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown impressive performance across multi-modal tasks by encoding images into thousands of tokens. However, the large number of image tokens results in significant computational overhead, and the use of dynamic high-resolution inputs further increases this burden. Previous approaches have attempted to reduce the number of image tokens through token pruning, typically by selecting tokens based on attention scores or image token diversity. Through empirical studies, we observe that existing methods often overlook the joint impact of pruning on both the current layer's output (local) and the outputs of subsequent layers (global), leading to suboptimal pruning decisions. To address this challenge, we propose Balanced Token Pruning (BTP), a plug-and-play method for pruning vision tokens. Specifically, our method utilizes a small calibration set to divide the pruning process into multiple stages. In the early stages, our method emphasizes the impact of pruning on subsequent layers, whereas in the deeper stages, the focus shifts toward preserving the consistency of local outputs. Extensive experiments across various LVLMs demonstrate the broad effectiveness of our approach on multiple benchmarks. Our method achieves a 78% compression rate while preserving 96.7% of the original models' performance on average.
>
---
#### [new 094] Zero-Shot 3D Visual Grounding from Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于零样本3D视觉定位任务，解决现有方法依赖3D标注数据和固定类别的局限。提出SeeGround框架，利用2D视觉语言模型，通过混合输入格式（渲染视图+空间文本）及视角适应、融合对齐模块，弥合模态差距，实现在ScanRefer/Nr3D数据集显著超越零样本基线，接近监督方法。**

- **链接: [http://arxiv.org/pdf/2505.22429v1](http://arxiv.org/pdf/2505.22429v1)**

> **作者:** Rong Li; Shijie Li; Lingdong Kong; Xulei Yang; Junwei Liang
>
> **备注:** 3D-LLM/VLA @ CVPR 2025; Project Page at https://seeground.github.io/
>
> **摘要:** 3D Visual Grounding (3DVG) seeks to locate target objects in 3D scenes using natural language descriptions, enabling downstream applications such as augmented reality and robotics. Existing approaches typically rely on labeled 3D data and predefined categories, limiting scalability to open-world settings. We present SeeGround, a zero-shot 3DVG framework that leverages 2D Vision-Language Models (VLMs) to bypass the need for 3D-specific training. To bridge the modality gap, we introduce a hybrid input format that pairs query-aligned rendered views with spatially enriched textual descriptions. Our framework incorporates two core components: a Perspective Adaptation Module that dynamically selects optimal viewpoints based on the query, and a Fusion Alignment Module that integrates visual and spatial signals to enhance localization precision. Extensive evaluations on ScanRefer and Nr3D confirm that SeeGround achieves substantial improvements over existing zero-shot baselines -- outperforming them by 7.7% and 7.1%, respectively -- and even rivals fully supervised alternatives, demonstrating strong generalization under challenging conditions.
>
---
#### [new 095] Hyperspectral Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于高光谱3D重建与视图合成任务，旨在解决传统NeRF方法训练慢、渲染速度低的问题。提出HS-GS框架，结合3DGS与扩散模型，引入波长编码器和KL散度损失，实现高光谱场景的精细重建与去噪，达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.21890v1](http://arxiv.org/pdf/2505.21890v1)**

> **作者:** Sunil Kumar Narayanan; Lingjun Zhao; Lu Gan; Yongsheng Chen
>
> **摘要:** Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements in samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Hyperspectral Gaussian Splatting (HS-GS), which combines the state-of-the-art 3D Gaussian Splatting (3DGS) with a diffusion model to enable 3D explicit reconstruction of the hyperspectral scenes and novel view synthesis for the entire spectral range. To enhance the model's ability to capture fine-grained reflectance variations across the light spectrum and leverage correlations between adjacent wavelengths for denoising, we introduce a wavelength encoder to generate wavelength-specific spherical harmonics offsets. We also introduce a novel Kullback--Leibler divergence-based loss to mitigate the spectral distribution gap between the rendered image and the ground truth. A diffusion model is further applied for denoising the rendered images and generating photorealistic hyperspectral images. We present extensive evaluations on five diverse hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of our proposed HS-GS framework. The results demonstrate that HS-GS achieves new state-of-the-art performance among all previously published methods. Code will be released upon publication.
>
---
#### [new 096] Hadaptive-Net: Efficient Vision Models via Adaptive Cross-Hadamard Synergy
- **分类: cs.CV**

- **简介: 该论文属于视觉任务，旨在解决Hadamard乘积未充分应用于提升网络效率与表示能力的问题。提出Adaptive Cross-Hadamard模块，通过自适应跨通道Hadamard乘积实现高效通道扩展，并构建Hadaptive-Net网络，在推理速度与准确率间达到平衡。**

- **链接: [http://arxiv.org/pdf/2505.22226v1](http://arxiv.org/pdf/2505.22226v1)**

> **作者:** Xuyang Zhang; Xi Zhang; Liang Chen; Hao Shi; Qingshan Guo
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent studies have revealed the immense potential of Hadamard product in enhancing network representational capacity and dimensional compression. However, despite its theoretical promise, this technique has not been systematically explored or effectively applied in practice, leaving its full capabilities underdeveloped. In this work, we first analyze and identify the advantages of Hadamard product over standard convolutional operations in cross-channel interaction and channel expansion. Building upon these insights, we propose a computationally efficient module: Adaptive Cross-Hadamard (ACH), which leverages adaptive cross-channel Hadamard products for high-dimensional channel expansion. Furthermore, we introduce Hadaptive-Net (Hadamard Adaptive Network), a lightweight network backbone for visual tasks, which is demonstrated through experiments that it achieves an unprecedented balance between inference speed and accuracy through our proposed module.
>
---
#### [new 097] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出SAM-R1框架，针对多模态图像分割任务，解决依赖昂贵标注数据的问题。通过强化学习整合SAM模型作为奖励提供者，首次在训练中引入细粒度分割设置，优化模型推理与分割对齐，仅用3k样本实现高性能，减少对标注数据的依赖。**

- **链接: [http://arxiv.org/pdf/2505.22596v1](http://arxiv.org/pdf/2505.22596v1)**

> **作者:** Jiaqi Huang; Zunnan Xu; Jun Zhou; Ting Liu; Yicheng Xiao; Mingwen Ou; Bowen Ji; Xiu Li; Kehong Yuan
>
> **摘要:** Leveraging multimodal large models for image segmentation has become a prominent research direction. However, existing approaches typically rely heavily on manually annotated datasets that include explicit reasoning processes, which are costly and time-consuming to produce. Recent advances suggest that reinforcement learning (RL) can endow large models with reasoning capabilities without requiring such reasoning-annotated data. In this paper, we propose SAM-R1, a novel framework that enables multimodal large models to perform fine-grained reasoning in image understanding tasks. Our approach is the first to incorporate fine-grained segmentation settings during the training of multimodal reasoning models. By integrating task-specific, fine-grained rewards with a tailored optimization objective, we further enhance the model's reasoning and segmentation alignment. We also leverage the Segment Anything Model (SAM) as a strong and flexible reward provider to guide the learning process. With only 3k training samples, SAM-R1 achieves strong performance across multiple benchmarks, demonstrating the effectiveness of reinforcement learning in equipping multimodal models with segmentation-oriented reasoning capabilities.
>
---
#### [new 098] DAM: Domain-Aware Module for Multi-Domain Dataset Condensation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于数据集凝练任务，解决现有方法忽视多域数据泛化的问题。提出Domain-Aware Module（DAM），通过可学习空间掩码嵌入域特征，并采用基于频率的伪域标签，在保持图像数量前提下提升跨域与跨模型性能。**

- **链接: [http://arxiv.org/pdf/2505.22387v1](http://arxiv.org/pdf/2505.22387v1)**

> **作者:** Jaehyun Choi; Gyojin Han; Dong-Jae Lee; Sunghyun Baek; Junmo Kim
>
> **摘要:** Dataset Condensation (DC) has emerged as a promising solution to mitigate the computational and storage burdens associated with training deep learning models. However, existing DC methods largely overlook the multi-domain nature of modern datasets, which are increasingly composed of heterogeneous images spanning multiple domains. In this paper, we extend DC and introduce Multi-Domain Dataset Condensation (MDDC), which aims to condense data that generalizes across both single-domain and multi-domain settings. To this end, we propose the Domain-Aware Module (DAM), a training-time module that embeds domain-related features into each synthetic image via learnable spatial masks. As explicit domain labels are mostly unavailable in real-world datasets, we employ frequency-based pseudo-domain labeling, which leverages low-frequency amplitude statistics. DAM is only active during the condensation process, thus preserving the same images per class (IPC) with prior methods. Experiments show that DAM consistently improves in-domain, out-of-domain, and cross-architecture performance over baseline dataset condensation methods.
>
---
#### [new 099] Learning to Infer Parameterized Representations of Plants from 3D Scans
- **分类: cs.CV**

- **简介: 该论文提出一种基于3D扫描的植物参数化表示统一框架，解决复杂植物结构重建、分割等任务分散的问题。通过递归神经网络学习L系统生成的虚拟植物数据，直接推断树状参数化表示，实现多任务处理，实验验证其效果与前沿方法相当。**

- **链接: [http://arxiv.org/pdf/2505.22337v1](http://arxiv.org/pdf/2505.22337v1)**

> **作者:** Samara Ghrer; Christophe Godin; Stefanie Wuhrer
>
> **摘要:** Reconstructing faithfully the 3D architecture of plants from unstructured observations is a challenging task. Plants frequently contain numerous organs, organized in branching systems in more or less complex spatial networks, leading to specific computational issues due to self-occlusion or spatial proximity between organs. Existing works either consider inverse modeling where the aim is to recover the procedural rules that allow to simulate virtual plants, or focus on specific tasks such as segmentation or skeletonization. We propose a unified approach that, given a 3D scan of a plant, allows to infer a parameterized representation of the plant. This representation describes the plant's branching structure, contains parametric information for each plant organ, and can therefore be used directly in a variety of tasks. In this data-driven approach, we train a recursive neural network with virtual plants generated using an L-systems-based procedural model. After training, the network allows to infer a parametric tree-like representation based on an input 3D point cloud. Our method is applicable to any plant that can be represented as binary axial tree. We evaluate our approach on Chenopodium Album plants, using experiments on synthetic plants to show that our unified framework allows for different tasks including reconstruction, segmentation and skeletonization, while achieving results on-par with state-of-the-art for each task.
>
---
#### [new 100] Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language
- **分类: cs.CV; cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文提出通过低维视觉-语言属性对齐进行灵活工具选择的框架，解决计算模型在模拟人类工具认知能力上的不足。构建ToolNet数据集，利用视觉模型提取工具图像属性，语言模型解析任务需求属性，通过关键操作属性（如握持性）匹配实现74%选择准确率，参数效率高且性能接近大模型。**

- **链接: [http://arxiv.org/pdf/2505.22146v1](http://arxiv.org/pdf/2505.22146v1)**

> **作者:** Guangfu Hao; Haojie Wen; Liangxuna Guo; Yang Chen; Yanchao Bi; Shan Yu
>
> **摘要:** Flexible tool selection reflects a complex cognitive ability that distinguishes humans from other species, yet computational models that capture this ability remain underdeveloped. We developed a framework using low-dimensional attribute representations to bridge visual tool perception and linguistic task understanding. We constructed a comprehensive dataset (ToolNet) containing 115 common tools labeled with 13 carefully designed attributes spanning physical, functional, and psychological properties, paired with natural language scenarios describing tool usage. Visual encoders (ResNet or ViT) extract attributes from tool images while fine-tuned language models (GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our approach achieves 74% accuracy in tool selection tasks-significantly outperforming direct tool matching (20%) and smaller multimodal models (21%-58%), while approaching performance of much larger models like GPT-4o (73%) with substantially fewer parameters. Ablation studies revealed that manipulation-related attributes (graspability, hand-relatedness, elongation) consistently prove most critical across modalities. This work provides a parameter-efficient, interpretable solution that mimics human-like tool cognition, advancing both cognitive science understanding and practical applications in tool selection tasks.
>
---
#### [new 101] Bringing CLIP to the Clinic: Dynamic Soft Labels and Negation-Aware Learning for Medical Analysis
- **分类: cs.CV**

- **简介: 该论文属于医疗视觉语言处理任务，针对CLIP模型在医疗影像分析中处理否定和数据不平衡的问题，提出动态软标签、否定感知学习及否定硬负例方法，并构建CXR-Align基准，提升零样本分类、报告检索等任务性能。**

- **链接: [http://arxiv.org/pdf/2505.22079v1](http://arxiv.org/pdf/2505.22079v1)**

> **作者:** Hanbin Ko; Chang-Min Park
>
> **备注:** 16 pages (8 main, 2 references, 6 appendix), 13 figures. Accepted to CVPR 2025. This author-accepted manuscript includes an expanded ethics/data user agreement section. The final version will appear in the Proceedings of CVPR 2025
>
> **摘要:** The development of large-scale image-text pair datasets has significantly advanced self-supervised learning in Vision-Language Processing (VLP). However, directly applying general-domain architectures such as CLIP to medical data presents challenges, particularly in handling negations and addressing the inherent data imbalance of medical datasets. To address these issues, we propose a novel approach that integrates clinically-enhanced dynamic soft labels and medical graphical alignment, thereby improving clinical comprehension and the applicability of contrastive loss in medical contexts. Furthermore, we introduce negation-based hard negatives to deepen the model's understanding of the complexities of clinical language. Our approach is easily integrated into the medical CLIP training pipeline and achieves state-of-the-art performance across multiple tasks, including zero-shot, fine-tuned classification, and report retrieval. To comprehensively evaluate our model's capacity for understanding clinical language, we introduce CXR-Align, a benchmark uniquely designed to evaluate the understanding of negation and clinical information within chest X-ray (CXR) datasets. Experimental results demonstrate that our proposed methods are straightforward to implement and generalize effectively across contrastive learning frameworks, enhancing medical VLP capabilities and advancing clinical language understanding in medical imaging.
>
---
#### [new 102] Domain Adaptation of Attention Heads for Zero-shot Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出HeadCLIP，针对零样本图像异常检测任务，解决现有方法在领域适应上的不足。通过可学习提示扩展文本编码器的正常/异常概念，图像编码器引入可学习注意力头权重，结合像素级信息优化跨领域检测效果，实验显示其在工业和医疗领域表现更优。**

- **链接: [http://arxiv.org/pdf/2505.22259v1](http://arxiv.org/pdf/2505.22259v1)**

> **作者:** Kiyoon Jeong; Jaehyuk Heo; Junyeong Son; Pilsung Kang
>
> **摘要:** Zero-shot anomaly detection (ZSAD) in images is an approach that can detect anomalies without access to normal samples, which can be beneficial in various realistic scenarios where model training is not possible. However, existing ZSAD research has shown limitations by either not considering domain adaptation of general-purpose backbone models to anomaly detection domains or by implementing only partial adaptation to some model components. In this paper, we propose HeadCLIP to overcome these limitations by effectively adapting both text and image encoders to the domain. HeadCLIP generalizes the concepts of normality and abnormality through learnable prompts in the text encoder, and introduces learnable head weights to the image encoder to dynamically adjust the features held by each attention head according to domain characteristics. Additionally, we maximize the effect of domain adaptation by introducing a joint anomaly score that utilizes domain-adapted pixel-level information for image-level anomaly detection. Experimental results using multiple real datasets in both industrial and medical domains show that HeadCLIP outperforms existing ZSAD techniques at both pixel and image levels. In the industrial domain, improvements of up to 4.9%p in pixel-level mean anomaly detection score (mAD) and up to 3.0%p in image-level mAD were achieved, with similar improvements (3.2%p, 3.1%p) in the medical domain.
>
---
#### [new 103] Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Thickness-aware E(3)-Equivariant 3D Mesh Neural Network（T-EMNN），解决传统3D网格分析方法忽略物体厚度导致精度不足的问题。通过引入集成厚度信息的E(3)协变坐标系统，提升节点级变形预测精度，同时保持计算效率。**

- **链接: [http://arxiv.org/pdf/2505.21572v1](http://arxiv.org/pdf/2505.21572v1)**

> **作者:** Sungwon Kim; Namkyeong Lee; Yunyoung Doh; Seungmin Shin; Guimok Cho; Seung-Won Jeon; Sangkook Kim; Chanyoung Park
>
> **备注:** ICML 2025
>
> **摘要:** Mesh-based 3D static analysis methods have recently emerged as efficient alternatives to traditional computational numerical solvers, significantly reducing computational costs and runtime for various physics-based analyses. However, these methods primarily focus on surface topology and geometry, often overlooking the inherent thickness of real-world 3D objects, which exhibits high correlations and similar behavior between opposing surfaces. This limitation arises from the disconnected nature of these surfaces and the absence of internal edge connections within the mesh. In this work, we propose a novel framework, the Thickness-aware E(3)-Equivariant 3D Mesh Neural Network (T-EMNN), that effectively integrates the thickness of 3D objects while maintaining the computational efficiency of surface meshes. Additionally, we introduce data-driven coordinates that encode spatial information while preserving E(3)-equivariance or invariance properties, ensuring consistent and robust analysis. Evaluations on a real-world industrial dataset demonstrate the superior performance of T-EMNN in accurately predicting node-level 3D deformations, effectively capturing thickness effects while maintaining computational efficiency.
>
---
#### [new 104] Learning A Robust RGB-Thermal Detector for Extreme Modality Imbalance
- **分类: cs.CV**

- **简介: 该论文属于RGB-T目标检测任务，解决模态失衡导致的检测性能下降问题。提出基-辅助检测器架构，包含模态交互模块自适应加权数据质量，并通过伪降级训练数据模拟现实失衡，提升极端条件下的检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.22154v1](http://arxiv.org/pdf/2505.22154v1)**

> **作者:** Chao Tian; Chao Yang; Guoqing Zhu; Qiang Wang; Zhenyu He
>
> **摘要:** RGB-Thermal (RGB-T) object detection utilizes thermal infrared (TIR) images to complement RGB data, improving robustness in challenging conditions. Traditional RGB-T detectors assume balanced training data, where both modalities contribute equally. However, in real-world scenarios, modality degradation-due to environmental factors or technical issues-can lead to extreme modality imbalance, causing out-of-distribution (OOD) issues during testing and disrupting model convergence during training. This paper addresses these challenges by proposing a novel base-and-auxiliary detector architecture. We introduce a modality interaction module to adaptively weigh modalities based on their quality and handle imbalanced samples effectively. Additionally, we leverage modality pseudo-degradation to simulate real-world imbalances in training data. The base detector, trained on high-quality pairs, provides a consistency constraint for the auxiliary detector, which receives degraded samples. This framework enhances model robustness, ensuring reliable performance even under severe modality degradation. Experimental results demonstrate the effectiveness of our method in handling extreme modality imbalances~(decreasing the Missing Rate by 55%) and improving performance across various baseline detectors.
>
---
#### [new 105] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频数据集缩合任务，旨在解决视频中空间内容与动态变化的复杂关联难以保留的问题。提出PRISM方法，通过渐进式优化与帧插入，保持动静元素间的依赖关系，提升动作表示效率，减少存储，在动作识别任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22564v1](http://arxiv.org/pdf/2505.22564v1)**

> **作者:** Jaehyun Choi; Jiwan Hur; Gyojin Han; Jaemyung Yu; Junmo Kim
>
> **摘要:** Video dataset condensation has emerged as a critical technique for addressing the computational challenges associated with large-scale video data processing in deep learning applications. While significant progress has been made in image dataset condensation, the video domain presents unique challenges due to the complex interplay between spatial content and temporal dynamics. This paper introduces PRISM, Progressive Refinement and Insertion for Sparse Motion, for video dataset condensation, a novel approach that fundamentally reconsiders how video data should be condensed. Unlike the previous method that separates static content from dynamic motion, our method preserves the essential interdependence between these elements. Our approach progressively refines and inserts frames to fully accommodate the motion in an action while achieving better performance but less storage, considering the relation of gradients for each frame. Extensive experiments across standard video action recognition benchmarks demonstrate that PRISM outperforms existing disentangled approaches while maintaining compact representations suitable for resource-constrained environments.
>
---
#### [new 106] BD Open LULC Map: High-resolution land use land cover mapping & benchmarking for urban development in Dhaka, Bangladesh
- **分类: cs.CV**

- **简介: 该论文属于高分辨率土地利用覆盖（LULC）制图及基准测试任务。针对南/东亚发展中国家标注卫星数据稀缺问题，构建覆盖达卡地区11类LULC的BOLM数据集（4392平方公里），通过三阶段GIS专家验证确保准确性。基于Bing和Sentinel-2A影像，采用DeepLab V3+进行模型性能对比，旨在填补区域数据缺口并支持深度学习应用。**

- **链接: [http://arxiv.org/pdf/2505.21915v1](http://arxiv.org/pdf/2505.21915v1)**

> **作者:** Mir Sazzat Hossain; Ovi Paul; Md Akil Raihan Iftee; Rakibul Hasan Rajib; Abu Bakar Siddik Nayem; Anis Sarker; Arshad Momen; Md. Ashraful Amin; Amin Ahsan Ali; AKM Mahbubur Rahman
>
> **备注:** 6 pages, 5 figures, 3 tables, Accepted In ICIP 2025
>
> **摘要:** Land Use Land Cover (LULC) mapping using deep learning significantly enhances the reliability of LULC classification, aiding in understanding geography, socioeconomic conditions, poverty levels, and urban sprawl. However, the scarcity of annotated satellite data, especially in South/East Asian developing countries, poses a major challenge due to limited funding, diverse infrastructures, and dense populations. In this work, we introduce the BD Open LULC Map (BOLM), providing pixel-wise LULC annotations across eleven classes (e.g., Farmland, Water, Forest, Urban Structure, Rural Built-Up) for Dhaka metropolitan city and its surroundings using high-resolution Bing satellite imagery (2.22 m/pixel). BOLM spans 4,392 sq km (891 million pixels), with ground truth validated through a three-stage process involving GIS experts. We benchmark LULC segmentation using DeepLab V3+ across five major classes and compare performance on Bing and Sentinel-2A imagery. BOLM aims to support reliable deep models and domain adaptation tasks, addressing critical LULC dataset gaps in South/East Asia.
>
---
#### [new 107] Event-based Egocentric Human Pose Estimation in Dynamic Environment
- **分类: cs.CV**

- **简介: 该论文提出D-EventEgo框架，解决动态环境中基于事件相机的自我中心人体姿态估计问题。任务通过头姿态估计引导身体姿态预测，创新性加入运动分割模块剔除动态干扰，实验显示其在合成数据集上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.22007v1](http://arxiv.org/pdf/2505.22007v1)**

> **作者:** Wataru Ikeda; Masashi Hatano; Ryosei Hara; Mariko Isogawa
>
> **备注:** Accepted at ICIP 2025, Project Page: https://wataru823.github.io/D-EventEgo/
>
> **摘要:** Estimating human pose using a front-facing egocentric camera is essential for applications such as sports motion analysis, VR/AR, and AI for wearable devices. However, many existing methods rely on RGB cameras and do not account for low-light environments or motion blur. Event-based cameras have the potential to address these challenges. In this work, we introduce a novel task of human pose estimation using a front-facing event-based camera mounted on the head and propose D-EventEgo, the first framework for this task. The proposed method first estimates the head poses, and then these are used as conditions to generate body poses. However, when estimating head poses, the presence of dynamic objects mixed with background events may reduce head pose estimation accuracy. Therefore, we introduce the Motion Segmentation Module to remove dynamic objects and extract background information. Extensive experiments on our synthetic event-based dataset derived from EgoBody, demonstrate that our approach outperforms our baseline in four out of five evaluation metrics in dynamic environments.
>
---
#### [new 108] Adapting Segment Anything Model for Power Transmission Corridor Hazard Segmentation
- **分类: cs.CV**

- **简介: 该论文属于电力输电走廊危险区域分割任务，旨在解决Segment Anything Model（SAM）在复杂场景中对精细结构目标分割效果不佳的问题。提出ELE-SAM模型，通过上下文感知提示适配器和高保真掩码解码器优化，结合新构建的ELE-40K数据集训练，显著提升分割精度。**

- **链接: [http://arxiv.org/pdf/2505.22105v1](http://arxiv.org/pdf/2505.22105v1)**

> **作者:** Hang Chen; Maoyuan Ye; Peng Yang; Haibin He; Juhua Liu; Bo Du
>
> **摘要:** Power transmission corridor hazard segmentation (PTCHS) aims to separate transmission equipment and surrounding hazards from complex background, conveying great significance to maintaining electric power transmission safety. Recently, the Segment Anything Model (SAM) has emerged as a foundational vision model and pushed the boundaries of segmentation tasks. However, SAM struggles to deal with the target objects in complex transmission corridor scenario, especially those with fine structure. In this paper, we propose ELE-SAM, adapting SAM for the PTCHS task. Technically, we develop a Context-Aware Prompt Adapter to achieve better prompt tokens via incorporating global-local features and focusing more on key regions. Subsequently, to tackle the hazard objects with fine structure in complex background, we design a High-Fidelity Mask Decoder by leveraging multi-granularity mask features and then scaling them to a higher resolution. Moreover, to train ELE-SAM and advance this field, we construct the ELE-40K benchmark, the first large-scale and real-world dataset for PTCHS including 44,094 image-mask pairs. Experimental results for ELE-40K demonstrate the superior performance that ELE-SAM outperforms the baseline model with the average 16.8% mIoU and 20.6% mBIoU performance improvement. Moreover, compared with the state-of-the-art method on HQSeg-44K, the average 2.9% mIoU and 3.8% mBIoU absolute improvements further validate the effectiveness of our method on high-quality generic object segmentation. The source code and dataset are available at https://github.com/Hhaizee/ELE-SAM.
>
---
#### [new 109] Single Domain Generalization for Alzheimer's Detection from 3D MRIs with Pseudo-Morphological Augmentations and Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于单领域泛化任务，旨在提升阿尔茨海默病3D MRI检测模型在新分布数据上的泛化能力，解决类不平衡和成像协议差异问题。提出伪形态增强模块生成解剖学相关的形状增强数据，并结合监督对比学习提取鲁棒特征，实验显示跨数据集性能提升。**

- **链接: [http://arxiv.org/pdf/2505.22465v1](http://arxiv.org/pdf/2505.22465v1)**

> **作者:** Zobia Batool; Huseyin Ozkan; Erchan Aptoula
>
> **摘要:** Although Alzheimer's disease detection via MRIs has advanced significantly thanks to contemporary deep learning models, challenges such as class imbalance, protocol variations, and limited dataset diversity often hinder their generalization capacity. To address this issue, this article focuses on the single domain generalization setting, where given the data of one domain, a model is designed and developed with maximal performance w.r.t. an unseen domain of distinct distribution. Since brain morphology is known to play a crucial role in Alzheimer's diagnosis, we propose the use of learnable pseudo-morphological modules aimed at producing shape-aware, anatomically meaningful class-specific augmentations in combination with a supervised contrastive learning module to extract robust class-specific representations. Experiments conducted across three datasets show improved performance and generalization capacity, especially under class imbalance and imaging protocol variations. The source code will be made available upon acceptance at https://github.com/zobia111/SDG-Alzheimer.
>
---
#### [new 110] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
- **分类: cs.CV; cs.LG**

- **简介: 论文提出RiverMamba模型，解决现有方法在空间连接上的局限，通过高效Mamba块捕捉全球水道网络并整合气象预报数据，实现7天全球河流流量及洪水预测，精度超越现有AI和物理模型。**

- **链接: [http://arxiv.org/pdf/2505.22535v1](http://arxiv.org/pdf/2505.22535v1)**

> **作者:** Mohamad Hakam Shams Eddin; Yikui Zahng; Stefan Kollet; Juergen Gall
>
> **备注:** Main paper 10 pages, Appendix 53 pages
>
> **摘要:** Recent deep learning approaches for river discharge forecasting have improved the accuracy and efficiency in flood forecasting, enabling more reliable early warning systems for risk management. Nevertheless, existing deep learning approaches in hydrology remain largely confined to local-scale applications and do not leverage the inherent spatial connections of bodies of water. Thus, there is a strong need for new deep learning methodologies that are capable of modeling spatio-temporal relations to improve river discharge and flood forecasting for scientific and operational applications. To address this, we present RiverMamba, a novel deep learning model that is pretrained with long-term reanalysis data and that can forecast global river discharge and floods on a $0.05^\circ$ grid up to 7 days lead time, which is of high relevance in early warning. To achieve this, RiverMamba leverages efficient Mamba blocks that enable the model to capture global-scale channel network routing and enhance its forecast capability for longer lead times. The forecast blocks integrate ECMWF HRES meteorological forecasts, while accounting for their inaccuracies through spatio-temporal modeling. Our analysis demonstrates that RiverMamba delivers reliable predictions of river discharge, including extreme floods across return periods and lead times, surpassing both operational AI- and physics-based models.
>
---
#### [new 111] Progressive Data Dropout: An Embarrassingly Simple Approach to Faster Training
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于训练加速任务，旨在解决大数据集训练耗时高问题。提出Progressive Data Dropout方法，通过渐进式数据筛选减少有效epoch至基线12.4%，同时提升精度4.82%，无需修改模型或优化器，可广泛应用于标准训练流程。**

- **链接: [http://arxiv.org/pdf/2505.22342v1](http://arxiv.org/pdf/2505.22342v1)**

> **作者:** Shriram M S; Xinyue Hao; Shihao Hou; Yang Lu; Laura Sevilla-Lara; Anurag Arnab; Shreyank N Gowda
>
> **摘要:** The success of the machine learning field has reliably depended on training on large datasets. While effective, this trend comes at an extraordinary cost. This is due to two deeply intertwined factors: the size of models and the size of datasets. While promising research efforts focus on reducing the size of models, the other half of the equation remains fairly mysterious. Indeed, it is surprising that the standard approach to training remains to iterate over and over, uniformly sampling the training dataset. In this paper we explore a series of alternative training paradigms that leverage insights from hard-data-mining and dropout, simple enough to implement and use that can become the new training standard. The proposed Progressive Data Dropout reduces the number of effective epochs to as little as 12.4% of the baseline. This savings actually do not come at any cost for accuracy. Surprisingly, the proposed method improves accuracy by up to 4.82%. Our approach requires no changes to model architecture or optimizer, and can be applied across standard training pipelines, thus posing an excellent opportunity for wide adoption. Code can be found here: https://github.com/bazyagami/LearningWithRevision
>
---
#### [new 112] Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像-文本检索任务，旨在解决CLIP模型因固定图像分辨率和有限上下文导致的细粒度跨模态理解不足问题。通过教师-学生蒸馏框架，利用YOLO提取的图像区域与文本片段的双向注意力生成增强嵌入，并采用混合损失函数训练轻量学生模型，在小规模数据下显著提升检索指标，同时保留94%的零样本分类性能。**

- **链接: [http://arxiv.org/pdf/2505.21549v1](http://arxiv.org/pdf/2505.21549v1)**

> **作者:** Daniel Csizmadia; Andrei Codreanu; Victor Sim; Vighnesh Prabeau; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** We present Distill CLIP (DCLIP), a fine-tuned variant of the CLIP model that enhances multimodal image-text retrieval while preserving the original model's strong zero-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine-grained cross-modal understanding. DCLIP addresses these challenges through a meta teacher-student distillation framework, where a cross-modal transformer teacher is fine-tuned to produce enriched embeddings via bidirectional cross-attention between YOLO-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions-just a fraction of CLIP's original dataset-DCLIP significantly improves image-text retrieval metrics (Recall@K, MAP), while retaining approximately 94% of CLIP's zero-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade-off between task specialization and generalization, offering a resource-efficient, domain-adaptive, and detail-sensitive solution for advanced vision-language tasks. Code available at https://anonymous.4open.science/r/DCLIP-B772/README.md.
>
---
#### [new 113] RePaViT: Scalable Vision Transformer Acceleration via Structural Reparameterization on Feedforward Network Layers
- **分类: cs.CV; cs.AI**

- **简介: 论文提出RePaViT，通过FFN层结构重参数化加速Vision Transformer。针对FFN层成为大模型推理瓶颈的问题，设计通道空闲机制，使部分通道跳过非线性激活，形成线性路径。实验显示，该方法在大模型中显著提升速度（如RePa-ViT-Huge加速68.7%），且准确率提高，为高效ViT提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.21847v1](http://arxiv.org/pdf/2505.21847v1)**

> **作者:** Xuwei Xu; Yang Li; Yudong Chen; Jiajun Liu; Sen Wang
>
> **备注:** Accepted to ICML2025
>
> **摘要:** We reveal that feedforward network (FFN) layers, rather than attention layers, are the primary contributors to Vision Transformer (ViT) inference latency, with their impact signifying as model size increases. This finding highlights a critical opportunity for optimizing the efficiency of large-scale ViTs by focusing on FFN layers. In this work, we propose a novel channel idle mechanism that facilitates post-training structural reparameterization for efficient FFN layers during testing. Specifically, a set of feature channels remains idle and bypasses the nonlinear activation function in each FFN layer, thereby forming a linear pathway that enables structural reparameterization during inference. This mechanism results in a family of ReParameterizable Vision Transformers (RePaViTs), which achieve remarkable latency reductions with acceptable sacrifices (sometimes gains) in accuracy across various ViTs. The benefits of our method scale consistently with model sizes, demonstrating greater speed improvements and progressively narrowing accuracy gaps or even higher accuracies on larger models. In particular, RePa-ViT-Large and RePa-ViT-Huge enjoy 66.8% and 68.7% speed-ups with +1.7% and +1.1% higher top-1 accuracies under the same training strategy, respectively. RePaViT is the first to employ structural reparameterization on FFN layers to expedite ViTs to our best knowledge, and we believe that it represents an auspicious direction for efficient ViTs. Source code is available at https://github.com/Ackesnal/RePaViT.
>
---
#### [new 114] Do you see what I see? An Ambiguous Optical Illusion Dataset exposing limitations of Explainable AI
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文构建了一个含动物混杂图像的光学 illusion 数据集，用于揭示可解释AI在模糊视觉任务中的局限性。通过分析 gaze direction 等视觉线索对模型的影响，研究人机视觉偏差与对齐问题，为 bias 缓解提供基准数据。**

- **链接: [http://arxiv.org/pdf/2505.21589v1](http://arxiv.org/pdf/2505.21589v1)**

> **作者:** Carina Newen; Luca Hinkamp; Maria Ntonti; Emmanuel Müller
>
> **备注:** 19 pages, 18 figures
>
> **摘要:** From uncertainty quantification to real-world object detection, we recognize the importance of machine learning algorithms, particularly in safety-critical domains such as autonomous driving or medical diagnostics. In machine learning, ambiguous data plays an important role in various machine learning domains. Optical illusions present a compelling area of study in this context, as they offer insight into the limitations of both human and machine perception. Despite this relevance, optical illusion datasets remain scarce. In this work, we introduce a novel dataset of optical illusions featuring intermingled animal pairs designed to evoke perceptual ambiguity. We identify generalizable visual concepts, particularly gaze direction and eye cues, as subtle yet impactful features that significantly influence model accuracy. By confronting models with perceptual ambiguity, our findings underscore the importance of concepts in visual learning and provide a foundation for studying bias and alignment between human and machine vision. To make this dataset useful for general purposes, we generate optical illusions systematically with different concepts discussed in our bias mitigation section. The dataset is accessible in Kaggle via https://kaggle.com/datasets/693bf7c6dd2cb45c8a863f9177350c8f9849a9508e9d50526e2ffcc5559a8333. Our source code can be found at https://github.com/KDD-OpenSource/Ambivision.git.
>
---
#### [new 115] QuARI: Query Adaptive Retrieval Improvement
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于大规模图像实例检索任务，解决视觉语言模型在处理超大规模图像集合时性能不足的问题。提出QuARI方法，通过学习将查询动态映射到特定的线性特征变换空间，实现计算高效且适应性更强的检索，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21647v1](http://arxiv.org/pdf/2505.21647v1)**

> **作者:** Eric Xing; Abby Stylianou; Robert Pless; Nathan Jacobs
>
> **备注:** 13 pages, 4 figures, 4 tables
>
> **摘要:** Massive-scale pretraining has made vision-language models increasingly popular for image-to-image and text-to-image retrieval across a broad collection of domains. However, these models do not perform well when used for challenging retrieval tasks, such as instance retrieval in very large-scale image collections. Recent work has shown that linear transformations of VLM features trained for instance retrieval can improve performance by emphasizing subspaces that relate to the domain of interest. In this paper, we explore a more extreme version of this specialization by learning to map a given query to a query-specific feature space transformation. Because this transformation is linear, it can be applied with minimal computational cost to millions of image embeddings, making it effective for large-scale retrieval or re-ranking. Results show that this method consistently outperforms state-of-the-art alternatives, including those that require many orders of magnitude more computation at query time.
>
---
#### [new 116] UniDB++: Fast Sampling of Unified Diffusion Bridge
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于扩散模型加速任务，针对UniDB框架采样效率低、误差积累大的问题，提出UniDB++：通过反向SDE的精确闭式解、数据预测模型及SDE校正机制，减少采样步骤（最多20倍），提升生成质量与速度，在图像修复任务中达最优性能。**

- **链接: [http://arxiv.org/pdf/2505.21528v1](http://arxiv.org/pdf/2505.21528v1)**

> **作者:** Mokai Pan; Kaizhen Zhu; Yuexin Ma; Yanwei Fu; Jingyi Yu; Jingya Wang; Ye Shi
>
> **摘要:** Diffusion Bridges enable transitions between arbitrary distributions, with the Unified Diffusion Bridge (UniDB) framework achieving high-fidelity image generation via a Stochastic Optimal Control (SOC) formulation. However, UniDB's reliance on iterative Euler sampling methods results in slow, computationally expensive inference, while existing acceleration techniques for diffusion or diffusion bridge models fail to address its unique challenges: missing terminal mean constraints and SOC-specific penalty coefficients in its SDEs. We present UniDB++, a training-free sampling algorithm that significantly improves upon these limitations. The method's key advancement comes from deriving exact closed-form solutions for UniDB's reverse-time SDEs, effectively reducing the error accumulation inherent in Euler approximations and enabling high-quality generation with up to 20$\times$ fewer sampling steps. This method is further complemented by replacing conventional noise prediction with a more stable data prediction model, along with an SDE-Corrector mechanism that maintains perceptual quality for low-step regimes (5-10 steps). Additionally, we demonstrate that UniDB++ aligns with existing diffusion bridge acceleration methods by evaluating their update rules, and UniDB++ can recover DBIMs as special cases under some theoretical conditions. Experiments demonstrate UniDB++'s state-of-the-art performance in image restoration tasks, outperforming Euler-based methods in fidelity and speed while reducing inference time significantly. This work bridges the gap between theoretical generality and practical efficiency in SOC-driven diffusion bridge models. Our code is available at https://github.com/2769433owo/UniDB-plusplus.
>
---
#### [new 117] MedBridge: Bridging Foundation Vision-Language Models to Medical Image Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医疗影像诊断任务，解决基础视觉语言模型（VLM）在医学领域表现差及训练资源消耗大的问题。提出MedBridge框架，通过Focal Sampling提取局部特征、Query Encoder对齐医学语义、Mixture of Experts整合多模型，高效适配VLM实现精准诊断，提升数据效率。**

- **链接: [http://arxiv.org/pdf/2505.21698v1](http://arxiv.org/pdf/2505.21698v1)**

> **作者:** Yitong Li; Morteza Ghahremani; Christian Wachinger
>
> **摘要:** Recent vision-language foundation models deliver state-of-the-art results on natural image classification but falter on medical images due to pronounced domain shifts. At the same time, training a medical foundation model requires substantial resources, including extensive annotated data and high computational capacity. To bridge this gap with minimal overhead, we introduce MedBridge, a lightweight multimodal adaptation framework that re-purposes pretrained VLMs for accurate medical image diagnosis. MedBridge comprises three key components. First, a Focal Sampling module that extracts high-resolution local regions to capture subtle pathological features and compensate for the limited input resolution of general-purpose VLMs. Second, a Query Encoder (QEncoder) injects a small set of learnable queries that attend to the frozen feature maps of VLM, aligning them with medical semantics without retraining the entire backbone. Third, a Mixture of Experts mechanism, driven by learnable queries, harnesses the complementary strength of diverse VLMs to maximize diagnostic performance. We evaluate MedBridge on five medical imaging benchmarks across three key adaptation tasks, demonstrating its superior performance in both cross-domain and in-domain adaptation settings, even under varying levels of training data availability. Notably, MedBridge achieved over 6-15% improvement in AUC compared to state-of-the-art VLM adaptation methods in multi-label thoracic disease diagnosis, underscoring its effectiveness in leveraging foundation models for accurate and data-efficient medical diagnosis. Our code is available at https://github.com/ai-med/MedBridge.
>
---
#### [new 118] Adversarially Robust AI-Generated Image Detection for Free: An Information Theoretic Perspective
- **分类: cs.CV**

- **简介: 论文提出TRIM方法，解决AI生成图像检测中对抗攻击防御难题。发现对抗训练因特征纠缠导致性能崩溃，通过信息论分析特征分离问题，利用预测熵和KL散度量化特征偏移，实现无训练鲁棒检测，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22604v1](http://arxiv.org/pdf/2505.22604v1)**

> **作者:** Ruixuan Zhang; He Wang; Zhengyu Zhao; Zhiqing Guo; Xun Yang; Yunfeng Diao; Meng Wang
>
> **摘要:** Rapid advances in Artificial Intelligence Generated Images (AIGI) have facilitated malicious use, such as forgery and misinformation. Therefore, numerous methods have been proposed to detect fake images. Although such detectors have been proven to be universally vulnerable to adversarial attacks, defenses in this field are scarce. In this paper, we first identify that adversarial training (AT), widely regarded as the most effective defense, suffers from performance collapse in AIGI detection. Through an information-theoretic lens, we further attribute the cause of collapse to feature entanglement, which disrupts the preservation of feature-label mutual information. Instead, standard detectors show clear feature separation. Motivated by this difference, we propose Training-free Robust Detection via Information-theoretic Measures (TRIM), the first training-free adversarial defense for AIGI detection. TRIM builds on standard detectors and quantifies feature shifts using prediction entropy and KL divergence. Extensive experiments across multiple datasets and attacks validate the superiority of our TRIM, e.g., outperforming the state-of-the-art defense by 33.88% (28.91%) on ProGAN (GenImage), while well maintaining original accuracy.
>
---
#### [new 119] Learning to See More: UAS-Guided Super-Resolution of Satellite Imagery for Precision Agriculture
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出融合卫星与无人机图像的超分辨率框架，解决两者数据分辨率与覆盖的trade-off。通过光谱和空间扩展，提升作物生物量及氮含量估测精度（分别提高18%和31%），减少无人机采集频率，实现精准农业低成本高精度应用。**

- **链接: [http://arxiv.org/pdf/2505.21746v1](http://arxiv.org/pdf/2505.21746v1)**

> **作者:** Arif Masrur; Peder A. Olsen; Paul R. Adler; Carlan Jackson; Matthew W. Myers; Nathan Sedghi; Ray R. Weil
>
> **摘要:** Unmanned Aircraft Systems (UAS) and satellites are key data sources for precision agriculture, yet each presents trade-offs. Satellite data offer broad spatial, temporal, and spectral coverage but lack the resolution needed for many precision farming applications, while UAS provide high spatial detail but are limited by coverage and cost, especially for hyperspectral data. This study presents a novel framework that fuses satellite and UAS imagery using super-resolution methods. By integrating data across spatial, spectral, and temporal domains, we leverage the strengths of both platforms cost-effectively. We use estimation of cover crop biomass and nitrogen (N) as a case study to evaluate our approach. By spectrally extending UAS RGB data to the vegetation red edge and near-infrared regions, we generate high-resolution Sentinel-2 imagery and improve biomass and N estimation accuracy by 18% and 31%, respectively. Our results show that UAS data need only be collected from a subset of fields and time points. Farmers can then 1) enhance the spectral detail of UAS RGB imagery; 2) increase the spatial resolution by using satellite data; and 3) extend these enhancements spatially and across the growing season at the frequency of the satellite flights. Our SRCNN-based spectral extension model shows considerable promise for model transferability over other cropping systems in the Upper and Lower Chesapeake Bay regions. Additionally, it remains effective even when cloud-free satellite data are unavailable, relying solely on the UAS RGB input. The spatial extension model produces better biomass and N predictions than models built on raw UAS RGB images. Once trained with targeted UAS RGB data, the spatial extension model allows farmers to stop repeated UAS flights. While we introduce super-resolution advances, the core contribution is a lightweight and scalable system for affordable on-farm use.
>
---
#### [new 120] BaryIR: Learning Multi-Source Unified Representation in Continuous Barycenter Space for Generalizable All-in-One Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决现有All-in-One图像修复（AIR）方法对分布外退化和图像泛化能力不足的问题。提出BaryIR框架，通过分解多源退化图像的潜空间为连续重心空间（统一退化无关表征）和特定子空间（捕捉退化特异性），利用多源最优传输重心学习紧凑表征，提升泛化性。**

- **链接: [http://arxiv.org/pdf/2505.21637v1](http://arxiv.org/pdf/2505.21637v1)**

> **作者:** Xiaole Tang; Xiaoyi He; Xiang Gu; Jian Sun
>
> **摘要:** Despite remarkable advances made in all-in-one image restoration (AIR) for handling different types of degradations simultaneously, existing methods remain vulnerable to out-of-distribution degradations and images, limiting their real-world applicability. In this paper, we propose a multi-source representation learning framework BaryIR, which decomposes the latent space of multi-source degraded images into a continuous barycenter space for unified feature encoding and source-specific subspaces for specific semantic encoding. Specifically, we seek the multi-source unified representation by introducing a multi-source latent optimal transport barycenter problem, in which a continuous barycenter map is learned to transport the latent representations to the barycenter space. The transport cost is designed such that the representations from source-specific subspaces are contrasted with each other while maintaining orthogonality to those from the barycenter space. This enables BaryIR to learn compact representations with unified degradation-agnostic information from the barycenter space, as well as degradation-specific semantics from source-specific subspaces, capturing the inherent geometry of multi-source data manifold for generalizable AIR. Extensive experiments demonstrate that BaryIR achieves competitive performance compared to state-of-the-art all-in-one methods. Particularly, BaryIR exhibits superior generalization ability to real-world data and unseen degradations. The code will be publicly available at https://github.com/xl-tang3/BaryIR.
>
---
#### [new 121] VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决长视觉token序列导致的计算开销问题。提出VScan框架：在视觉编码阶段通过全局/局部扫描合并冗余token，语言解码阶段在中间层进行剪枝，实现加速（2.91倍预填充速度，10倍FLOPs减少）同时保留95.4%原性能。**

- **链接: [http://arxiv.org/pdf/2505.22654v1](http://arxiv.org/pdf/2505.22654v1)**

> **作者:** Ce Zhang; Kaixin Ma; Tianqing Fang; Wenhao Yu; Hongming Zhang; Zhisong Zhang; Yaqi Xie; Katia Sycara; Haitao Mi; Dong Yu
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) have advanced multi-modal understanding by incorporating finer-grained visual perception and encoding. However, such methods incur significant computational costs due to longer visual token sequences, posing challenges for real-time deployment. To mitigate this, prior studies have explored pruning unimportant visual tokens either at the output layer of the visual encoder or at the early layers of the language model. In this work, we revisit these design choices and reassess their effectiveness through comprehensive empirical studies of how visual tokens are processed throughout the visual encoding and language decoding stages. Guided by these insights, we propose VScan, a two-stage visual token reduction framework that addresses token redundancy by: (1) integrating complementary global and local scans with token merging during visual encoding, and (2) introducing pruning at intermediate layers of the language model. Extensive experimental results across four LVLMs validate the effectiveness of VScan in accelerating inference and demonstrate its superior performance over current state-of-the-arts on sixteen benchmarks. Notably, when applied to LLaVA-NeXT-7B, VScan achieves a 2.91$\times$ speedup in prefilling and a 10$\times$ reduction in FLOPs, while retaining 95.4% of the original performance.
>
---
#### [new 122] YH-MINER: Multimodal Intelligent System for Natural Ecological Reef Metric Extraction
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文提出YH-OSI系统，解决珊瑚礁监测中人工分析效率低和复杂水下场景分割精度不足的问题。通过多模态大模型整合目标检测（mAP@0.5=0.78）、语义分割及分类模块，实现低光/遮挡下的像素级分割（属级分类88%），并支持生态指标提取与自动化流程扩展。**

- **链接: [http://arxiv.org/pdf/2505.22250v1](http://arxiv.org/pdf/2505.22250v1)**

> **作者:** Mingzhuang Wang; Yvyang Li; Xiyang Zhang; Fei Tan; Qi Shi; Guotao Zhang; Siqi Chen; Yufei Liu; Lei Lei; Ming Zhou; Qiang Lin; Hongqiang Yang
>
> **摘要:** Coral reefs, crucial for sustaining marine biodiversity and ecological processes (e.g., nutrient cycling, habitat provision), face escalating threats, underscoring the need for efficient monitoring. Coral reef ecological monitoring faces dual challenges of low efficiency in manual analysis and insufficient segmentation accuracy in complex underwater scenarios. This study develops the YH-OSI system, establishing an intelligent framework centered on the Multimodal Large Model (MLLM) for "object detection-semantic segmentation-prior input". The system uses the object detection module (mAP@0.5=0.78) to generate spatial prior boxes for coral instances, driving the segment module to complete pixel-level segmentation in low-light and densely occluded scenarios. The segmentation masks and finetuned classification instructions are fed into the Qwen2-VL-based multimodal model as prior inputs, achieving a genus-level classification accuracy of 88% and simultaneously extracting core ecological metrics. Meanwhile, the system retains the scalability of the multimodal model through standardized interfaces, laying a foundation for future integration into multimodal agent-based underwater robots and supporting the full-process automation of "image acquisition-prior generation-real-time analysis."
>
---
#### [new 123] Training Free Stylized Abstraction
- **分类: cs.CV**

- **简介: 该论文提出一种无需训练的风格化抽象生成方法，解决在保持语义准确性的同时实现视觉夸张化的问题。通过视觉语言模型的推理时特征提取与跨域流逆向重构策略，动态平衡身份识别与风格变形，支持多轮生成且适用于未见身份和风格。任务属图像风格化领域，方法包含新型逆向 flow 算法与评估指标 StyleBench。**

- **链接: [http://arxiv.org/pdf/2505.22663v1](http://arxiv.org/pdf/2505.22663v1)**

> **作者:** Aimon Rahman; Kartik Narayan; Vishal M. Patel
>
> **备注:** Project Page: https://kartik-3004.github.io/TF-SA/
>
> **摘要:** Stylized abstraction synthesizes visually exaggerated yet semantically faithful representations of subjects, balancing recognizability with perceptual distortion. Unlike image-to-image translation, which prioritizes structural fidelity, stylized abstraction demands selective retention of identity cues while embracing stylistic divergence, especially challenging for out-of-distribution individuals. We propose a training-free framework that generates stylized abstractions from a single image using inference-time scaling in vision-language models (VLLMs) to extract identity-relevant features, and a novel cross-domain rectified flow inversion strategy that reconstructs structure based on style-dependent priors. Our method adapts structural restoration dynamically through style-aware temporal scheduling, enabling high-fidelity reconstructions that honor both subject and style. It supports multi-round abstraction-aware generation without fine-tuning. To evaluate this task, we introduce StyleBench, a GPT-based human-aligned metric suited for abstract styles where pixel-level similarity fails. Experiments across diverse abstraction (e.g., LEGO, knitted dolls, South Park) show strong generalization to unseen identities and styles in a fully open-source setup.
>
---
#### [new 124] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文属于少样本分割任务，针对SAM2因优化跟踪任务导致语义表示混乱的问题，提出SANSA框架，通过显式语义对齐释放其隐藏语义，实现更优性能，速度快且模型紧凑。**

- **链接: [http://arxiv.org/pdf/2505.21795v1](http://arxiv.org/pdf/2505.21795v1)**

> **作者:** Claudia Cuttano; Gabriele Trivigno; Giuseppe Averta; Carlo Masone
>
> **备注:** Code: https://github.com/ClaudiaCuttano/SANSA
>
> **摘要:** Few-shot segmentation aims to segment unseen object categories from just a handful of annotated examples. This requires mechanisms that can both identify semantically related objects across images and accurately produce segmentation masks. We note that Segment Anything 2 (SAM2), with its prompt-and-propagate mechanism, offers both strong segmentation capabilities and a built-in feature matching process. However, we show that its representations are entangled with task-specific cues optimized for object tracking, which impairs its use for tasks requiring higher level semantic understanding. Our key insight is that, despite its class-agnostic pretraining, SAM2 already encodes rich semantic structure in its features. We propose SANSA (Semantically AligNed Segment Anything 2), a framework that makes this latent structure explicit, and repurposes SAM2 for few-shot segmentation through minimal task-specific modifications. SANSA achieves state-of-the-art performance on few-shot segmentation benchmarks specifically designed to assess generalization, outperforms generalist methods in the popular in-context setting, supports various prompts flexible interaction via points, boxes, or scribbles, and remains significantly faster and more compact than prior approaches. Code is available at https://github.com/ClaudiaCuttano/SANSA.
>
---
#### [new 125] S2AFormer: Strip Self-Attention for Efficient Vision Transformer
- **分类: cs.CV**

- **简介: 该论文属于高效视觉Transformer任务，旨在解决自注意力计算开销大的问题。提出S2AFormer，通过Strip Self-Attention（SSA）和混合感知块（HPBs），压缩K/V的空间维度及Q/K的通道维度，在ImageNet等多任务中实现效率与精度的平衡。**

- **链接: [http://arxiv.org/pdf/2505.22195v1](http://arxiv.org/pdf/2505.22195v1)**

> **作者:** Guoan Xu; Wenfeng Huang; Wenjing Jia; Jiamao Li; Guangwei Gao; Guo-Jun Qi
>
> **备注:** 12 pages, 6 figures, 8 tables
>
> **摘要:** Vision Transformer (ViT) has made significant advancements in computer vision, thanks to its token mixer's sophisticated ability to capture global dependencies between all tokens. However, the quadratic growth in computational demands as the number of tokens increases limits its practical efficiency. Although recent methods have combined the strengths of convolutions and self-attention to achieve better trade-offs, the expensive pairwise token affinity and complex matrix operations inherent in self-attention remain a bottleneck. To address this challenge, we propose S2AFormer, an efficient Vision Transformer architecture featuring novel Strip Self-Attention (SSA). We design simple yet effective Hybrid Perception Blocks (HPBs) to effectively integrate the local perception capabilities of CNNs with the global context modeling of Transformer's attention mechanisms. A key innovation of SSA lies in its reducing the spatial dimensions of $K$ and $V$ while compressing the channel dimensions of $Q$ and $K$. This design significantly reduces computational overhead while preserving accuracy, striking an optimal balance between efficiency and effectiveness. We evaluate the robustness and efficiency of S2AFormer through extensive experiments on multiple vision benchmarks, including ImageNet-1k for image classification, ADE20k for semantic segmentation, and COCO for object detection and instance segmentation. Results demonstrate that S2AFormer achieves significant accuracy gains with superior efficiency in both GPU and non-GPU environments, making it a strong candidate for efficient vision Transformers.
>
---
#### [new 126] Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for Chest X-ray Report Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学影像分析任务，旨在解决多模态大语言模型生成胸片报告时存在的幻觉和临床错误问题。提出Look & Mark方法，通过整合放射科医生眼动数据和边界框标注到LLM的提示框架中，利用上下文学习提升模型性能，减少临床错误，无需重新训练即显著提高报告准确性。**

- **链接: [http://arxiv.org/pdf/2505.22222v1](http://arxiv.org/pdf/2505.22222v1)**

> **作者:** Yunsoo Kim; Jinge Wu; Su-Hwan Kim; Pardeep Vasudev; Jiashu Shen; Honghan Wu
>
> **摘要:** Recent advancements in multimodal Large Language Models (LLMs) have significantly enhanced the automation of medical image analysis, particularly in generating radiology reports from chest X-rays (CXR). However, these models still suffer from hallucinations and clinically significant errors, limiting their reliability in real-world applications. In this study, we propose Look & Mark (L&M), a novel grounding fixation strategy that integrates radiologist eye fixations (Look) and bounding box annotations (Mark) into the LLM prompting framework. Unlike conventional fine-tuning, L&M leverages in-context learning to achieve substantial performance gains without retraining. When evaluated across multiple domain-specific and general-purpose models, L&M demonstrates significant gains, including a 1.2% improvement in overall metrics (A.AVG) for CXR-LLaVA compared to baseline prompting and a remarkable 9.2% boost for LLaVA-Med. General-purpose models also benefit from L&M combined with in-context learning, with LLaVA-OV achieving an 87.3% clinical average performance (C.AVG)-the highest among all models, even surpassing those explicitly trained for CXR report generation. Expert evaluations further confirm that L&M reduces clinically significant errors (by 0.43 average errors per report), such as false predictions and omissions, enhancing both accuracy and reliability. These findings highlight L&M's potential as a scalable and efficient solution for AI-assisted radiology, paving the way for improved diagnostic workflows in low-resource clinical settings.
>
---
#### [new 127] UniMoGen: Universal Motion Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出UniMoGen，一种基于UNet的扩散模型，解决现有运动生成方法依赖特定骨骼结构的问题。其通过动态处理不同角色必要关节，实现骨架无关且高效的运动生成，支持风格/轨迹控制及延续生成。实验显示其在多数据集上超越现有方法，提升跨角色动画生成的灵活性与效率。**

- **链接: [http://arxiv.org/pdf/2505.21837v1](http://arxiv.org/pdf/2505.21837v1)**

> **作者:** Aliasghar Khani; Arianna Rampini; Evan Atherton; Bruno Roy
>
> **摘要:** Motion generation is a cornerstone of computer graphics, animation, gaming, and robotics, enabling the creation of realistic and varied character movements. A significant limitation of existing methods is their reliance on specific skeletal structures, which restricts their versatility across different characters. To overcome this, we introduce UniMoGen, a novel UNet-based diffusion model designed for skeleton-agnostic motion generation. UniMoGen can be trained on motion data from diverse characters, such as humans and animals, without the need for a predefined maximum number of joints. By dynamically processing only the necessary joints for each character, our model achieves both skeleton agnosticism and computational efficiency. Key features of UniMoGen include controllability via style and trajectory inputs, and the ability to continue motions from past frames. We demonstrate UniMoGen's effectiveness on the 100style dataset, where it outperforms state-of-the-art methods in diverse character motion generation. Furthermore, when trained on both the 100style and LAFAN1 datasets, which use different skeletons, UniMoGen achieves high performance and improved efficiency across both skeletons. These results highlight UniMoGen's potential to advance motion generation by providing a flexible, efficient, and controllable solution for a wide range of character animations.
>
---
#### [new 128] Equivariant Flow Matching for Point Cloud Assembly
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云装配任务，旨在通过多点云片段对齐重建完整3D形状。针对传统方法在数据效率和非重叠输入处理上的不足，提出equivariant diffusion assembly（Eda）模型。其基于流匹配理论学习等变向量场，并构建等变路径提升训练效率，有效解决复杂装配问题，尤其在非重叠场景表现优异。**

- **链接: [http://arxiv.org/pdf/2505.21539v1](http://arxiv.org/pdf/2505.21539v1)**

> **作者:** Ziming Wang; Nan Xue; Rebecka Jörnsten
>
> **摘要:** The goal of point cloud assembly is to reconstruct a complete 3D shape by aligning multiple point cloud pieces. This work presents a novel equivariant solver for assembly tasks based on flow matching models. We first theoretically show that the key to learning equivariant distributions via flow matching is to learn related vector fields. Based on this result, we propose an assembly model, called equivariant diffusion assembly (Eda), which learns related vector fields conditioned on the input pieces. We further construct an equivariant path for Eda, which guarantees high data efficiency of the training process. Our numerical results show that Eda is highly competitive on practical datasets, and it can even handle the challenging situation where the input pieces are non-overlapped.
>
---
#### [new 129] GoMatching++: Parameter- and Data-Efficient Arbitrary-Shaped Video Text Spotting and Benchmarking
- **分类: cs.CV**

- **简介: 该论文属于视频文本检测任务，旨在解决现有方法参数与数据效率低、视频文本识别性能不足的问题。提出GoMatching++方法，通过冻结图像检测器并添加轻量跟踪器、域间隙重新评分及LST-Matcher组件，提升视频文本处理能力且降低训练成本；同时构建含弯曲文本的新数据集ArTVideo推动研究。**

- **链接: [http://arxiv.org/pdf/2505.22228v1](http://arxiv.org/pdf/2505.22228v1)**

> **作者:** Haibin He; Jing Zhang; Maoyuan Ye; Juhua Liu; Bo Du; Dacheng Tao
>
> **备注:** arXiv admin note: text overlap with arXiv:2401.07080
>
> **摘要:** Video text spotting (VTS) extends image text spotting (ITS) by adding text tracking, significantly increasing task complexity. Despite progress in VTS, existing methods still fall short of the performance seen in ITS. This paper identifies a key limitation in current video text spotters: limited recognition capability, even after extensive end-to-end training. To address this, we propose GoMatching++, a parameter- and data-efficient method that transforms an off-the-shelf image text spotter into a video specialist. The core idea lies in freezing the image text spotter and introducing a lightweight, trainable tracker, which can be optimized efficiently with minimal training data. Our approach includes two key components: (1) a rescoring mechanism to bridge the domain gap between image and video data, and (2) the LST-Matcher, which enhances the frozen image text spotter's ability to handle video text. We explore various architectures for LST-Matcher to ensure efficiency in both parameters and training data. As a result, GoMatching++ sets new performance records on challenging benchmarks such as ICDAR15-video, DSText, and BOVText, while significantly reducing training costs. To address the lack of curved text datasets in VTS, we introduce ArTVideo, a new benchmark featuring over 30% curved text with detailed annotations. We also provide a comprehensive statistical analysis and experimental results for ArTVideo. We believe that GoMatching++ and the ArTVideo benchmark will drive future advancements in video text spotting. The source code, models and dataset are publicly available at https://github.com/Hxyz-123/GoMatching.
>
---
#### [new 130] EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像到视频（I2V）的相机控制任务，旨在解决现有方法依赖点云估计导致锚视频误差大及需大量相机轨迹标注的问题。提出EPiC框架：通过第一帧可见性掩码自动生成高质量锚视频，并设计轻量Anchor-ControlNet模块（参数<1%），实现无需轨迹标注、高效训练且泛化性强的3D相机控制，达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.21876v1](http://arxiv.org/pdf/2505.21876v1)**

> **作者:** Zun Wang; Jaemin Cho; Jialu Li; Han Lin; Jaehong Yoon; Yue Zhang; Mohit Bansal
>
> **备注:** Project website: https://zunwang1.github.io/Epic
>
> **摘要:** Recent approaches on 3D camera control in video diffusion models (VDMs) often create anchor videos to guide diffusion models as a structured prior by rendering from estimated point clouds following annotated camera trajectories. However, errors inherent in point cloud estimation often lead to inaccurate anchor videos. Moreover, the requirement for extensive camera trajectory annotations further increases resource demands. To address these limitations, we introduce EPiC, an efficient and precise camera control learning framework that automatically constructs high-quality anchor videos without expensive camera trajectory annotations. Concretely, we create highly precise anchor videos for training by masking source videos based on first-frame visibility. This approach ensures high alignment, eliminates the need for camera trajectory annotations, and thus can be readily applied to any in-the-wild video to generate image-to-video (I2V) training pairs. Furthermore, we introduce Anchor-ControlNet, a lightweight conditioning module that integrates anchor video guidance in visible regions to pretrained VDMs, with less than 1% of backbone model parameters. By combining the proposed anchor video data and ControlNet module, EPiC achieves efficient training with substantially fewer parameters, training steps, and less data, without requiring modifications to the diffusion model backbone typically needed to mitigate rendering misalignments. Although being trained on masking-based anchor videos, our method generalizes robustly to anchor videos made with point clouds during inference, enabling precise 3D-informed camera control. EPiC achieves SOTA performance on RealEstate10K and MiraData for I2V camera control task, demonstrating precise and robust camera control ability both quantitatively and qualitatively. Notably, EPiC also exhibits strong zero-shot generalization to video-to-video scenarios.
>
---
#### [new 131] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视角场景理解任务，旨在解决第一人称视角（如头戴相机）因视野狭窄和缺乏全局信息导致的推理局限。提出结合第三人称视角的框架，构建E3VQA基准数据集，并设计M3CoT提示方法，通过融合多视角场景图提升大模型跨视角推理能力，显著提高GPT-4o等模型性能。**

- **链接: [http://arxiv.org/pdf/2505.21955v1](http://arxiv.org/pdf/2505.21955v1)**

> **作者:** Insu Lee; Wooje Park; Jaeyun Jang; Minyoung Noh; Kyuhong Shim; Byonghyo Shim
>
> **摘要:** Large vision-language models (LVLMs) are increasingly deployed in interactive applications such as virtual and augmented reality, where first-person (egocentric) view captured by head-mounted cameras serves as key input. While this view offers fine-grained cues about user attention and hand-object interactions, their narrow field of view and lack of global context often lead to failures on spatially or contextually demanding queries. To address this, we introduce a framework that augments egocentric inputs with third-person (exocentric) views, providing complementary information such as global scene layout and object visibility to LVLMs. We present E3VQA, the first benchmark for multi-view question answering with 4K high-quality question-answer pairs grounded in synchronized ego-exo image pairs. Additionally, we propose M3CoT, a training-free prompting technique that constructs a unified scene representation by integrating scene graphs from three complementary perspectives. M3CoT enables LVLMs to reason more effectively across views, yielding consistent performance gains (4.84% for GPT-4o and 5.94% for Gemini 2.0 Flash) over a recent CoT baseline. Our extensive evaluation reveals key strengths and limitations of LVLMs in multi-view reasoning and highlights the value of leveraging both egocentric and exocentric inputs.
>
---
#### [new 132] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像上下文理解任务，旨在解决现有方法难以准确提取事件相关图像的深层时空与地理信息的问题。提出GETReason框架，通过分层多智能体推理整合全局事件、时间及地理数据，并设计GREAT评估指标，提升图像重要性与事件背景的关联分析。**

- **链接: [http://arxiv.org/pdf/2505.21863v1](http://arxiv.org/pdf/2505.21863v1)**

> **作者:** Shikhhar Siingh; Abhinav Rawat; Vivek Gupta; Chitta Baral
>
> **摘要:** Publicly significant images from events hold valuable contextual information, crucial for journalism and education. However, existing methods often struggle to extract this relevance accurately. To address this, we introduce GETReason (Geospatial Event Temporal Reasoning), a framework that moves beyond surface-level image descriptions to infer deeper contextual meaning. We propose that extracting global event, temporal, and geospatial information enhances understanding of an image's significance. Additionally, we introduce GREAT (Geospatial Reasoning and Event Accuracy with Temporal Alignment), a new metric for evaluating reasoning-based image understanding. Our layered multi-agent approach, assessed using a reasoning-weighted metric, demonstrates that meaningful insights can be inferred, effectively linking images to their broader event context.
>
---
#### [new 133] Object Concepts Emerge from Motion
- **分类: cs.CV**

- **简介: 该论文属于无监督视觉表征学习任务，旨在通过运动信息自动学习物体实例表征。针对如何在无标注视频中捕捉物体边界的问题，提出利用光流和聚类生成伪实例掩码，结合对比学习训练视觉编码器。方法无需标注和相机参数，通过运动边界驱动物体级分组，在多个视觉任务中超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21635v1](http://arxiv.org/pdf/2505.21635v1)**

> **作者:** Haoqian Liang; Xiaohui Wang; Zhichao Li; Ya Yang; Naiyan Wang
>
> **摘要:** Object concepts play a foundational role in human visual cognition, enabling perception, memory, and interaction in the physical world. Inspired by findings in developmental neuroscience - where infants are shown to acquire object understanding through observation of motion - we propose a biologically inspired framework for learning object-centric visual representations in an unsupervised manner. Our key insight is that motion boundary serves as a strong signal for object-level grouping, which can be used to derive pseudo instance supervision from raw videos. Concretely, we generate motion-based instance masks using off-the-shelf optical flow and clustering algorithms, and use them to train visual encoders via contrastive learning. Our framework is fully label-free and does not rely on camera calibration, making it scalable to large-scale unstructured video data. We evaluate our approach on three downstream tasks spanning both low-level (monocular depth estimation) and high-level (3D object detection and occupancy prediction) vision. Our models outperform previous supervised and self-supervised baselines and demonstrate strong generalization to unseen scenes. These results suggest that motion-induced object representations offer a compelling alternative to existing vision foundation models, capturing a crucial but overlooked level of abstraction: the visual instance. The corresponding code will be released upon paper acceptance.
>
---
#### [new 134] ALTER: All-in-One Layer Pruning and Temporal Expert Routing for Efficient Diffusion Generation
- **分类: cs.CV**

- **简介: 该论文属于扩散模型加速任务。解决现有方法无法捕捉生成过程时序变化及剪枝与微调不匹配的问题。提出ALTER框架，通过可训练超网络统一实现层剪枝、时序路由与模型微调，动态优化计算路径，使模型在20步推理中以25.9%计算量达成原50步模型的生成质量，速度提升3.64倍。**

- **链接: [http://arxiv.org/pdf/2505.21817v1](http://arxiv.org/pdf/2505.21817v1)**

> **作者:** Xiaomeng Yang; Lei Lu; Qihui Fan; Changdi Yang; Juyi Lin; Yanzhi Wang; Xuan Zhang; Shangqian Gao
>
> **摘要:** Diffusion models have demonstrated exceptional capabilities in generating high-fidelity images. However, their iterative denoising process results in significant computational overhead during inference, limiting their practical deployment in resource-constrained environments. Existing acceleration methods often adopt uniform strategies that fail to capture the temporal variations during diffusion generation, while the commonly adopted sequential pruning-then-fine-tuning strategy suffers from sub-optimality due to the misalignment between pruning decisions made on pretrained weights and the model's final parameters. To address these limitations, we introduce ALTER: All-in-One Layer Pruning and Temporal Expert Routing, a unified framework that transforms diffusion models into a mixture of efficient temporal experts. ALTER achieves a single-stage optimization that unifies layer pruning, expert routing, and model fine-tuning by employing a trainable hypernetwork, which dynamically generates layer pruning decisions and manages timestep routing to specialized, pruned expert sub-networks throughout the ongoing fine-tuning of the UNet. This unified co-optimization strategy enables significant efficiency gains while preserving high generative quality. Specifically, ALTER achieves same-level visual fidelity to the original 50-step Stable Diffusion v2.1 model while utilizing only 25.9% of its total MACs with just 20 inference steps and delivering a 3.64x speedup through 35% sparsity.
>
---
#### [new 135] OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测与分析任务，旨在解决现有方法无法生成结合领域知识的细粒度异常解释问题。提出OmniAD框架，融合视觉（Text-as-Mask编码异常检测）与文本推理（视觉引导分析），并通过混合监督微调与强化学习提升少样本泛化能力，在多基准测试中取得最优性能。**

- **链接: [http://arxiv.org/pdf/2505.22039v1](http://arxiv.org/pdf/2505.22039v1)**

> **作者:** Shifang Zhao; Yiheng Lin; Lu Han; Yao Zhao; Yunchao Wei
>
> **摘要:** While anomaly detection has made significant progress, generating detailed analyses that incorporate industrial knowledge remains a challenge. To address this gap, we introduce OmniAD, a novel framework that unifies anomaly detection and understanding for fine-grained analysis. OmniAD is a multimodal reasoner that combines visual and textual reasoning processes. The visual reasoning provides detailed inspection by leveraging Text-as-Mask Encoding to perform anomaly detection through text generation without manually selected thresholds. Following this, Visual Guided Textual Reasoning conducts comprehensive analysis by integrating visual perception. To enhance few-shot generalization, we employ an integrated training strategy that combines supervised fine-tuning (SFT) with reinforcement learning (GRPO), incorporating three sophisticated reward functions. Experimental results demonstrate that OmniAD achieves a performance of 79.1 on the MMAD benchmark, surpassing models such as Qwen2.5-VL-7B and GPT-4o. It also shows strong results across multiple anomaly detection benchmarks. These results highlight the importance of enhancing visual perception for effective reasoning in anomaly understanding. All codes and models will be publicly available.
>
---
#### [new 136] Learning Fine-Grained Geometry for Sparse-View Splatting via Cascade Depth Loss
- **分类: cs.CV**

- **简介: 该论文属于3D视图合成任务，解决稀疏视角下几何不准确导致的重建模糊和结构伪影问题。提出HDGS框架，通过级联深度损失多尺度优化深度一致性，提升几何精度与结构保真度，在LLFF/DTU数据集上实现稀疏场景下的最优效果。**

- **链接: [http://arxiv.org/pdf/2505.22279v1](http://arxiv.org/pdf/2505.22279v1)**

> **作者:** Wenjun Lu; Haodong Chen; Anqi Yi; Yuk Ying Chung; Zhiyong Wang; Kun Hu
>
> **摘要:** Novel view synthesis is a fundamental task in 3D computer vision that aims to reconstruct realistic images from a set of posed input views. However, reconstruction quality degrades significantly under sparse-view conditions due to limited geometric cues. Existing methods, such as Neural Radiance Fields (NeRF) and the more recent 3D Gaussian Splatting (3DGS), often suffer from blurred details and structural artifacts when trained with insufficient views. Recent works have identified the quality of rendered depth as a key factor in mitigating these artifacts, as it directly affects geometric accuracy and view consistency. In this paper, we address these challenges by introducing Hierarchical Depth-Guided Splatting (HDGS), a depth supervision framework that progressively refines geometry from coarse to fine levels. Central to HDGS is a novel Cascade Pearson Correlation Loss (CPCL), which aligns rendered and estimated monocular depths across multiple spatial scales. By enforcing multi-scale depth consistency, our method substantially improves structural fidelity in sparse-view scenarios. Extensive experiments on the LLFF and DTU benchmarks demonstrate that HDGS achieves state-of-the-art performance under sparse-view settings while maintaining efficient and high-quality rendering
>
---
#### [new 137] Neural Restoration of Greening Defects in Historical Autochrome Photographs Based on Purely Synthetic Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于合成数据与生成对抗网络的神经修复方法，解决历史Autochrome彩色照片的绿化缺陷问题。通过构建模拟退化缺陷的合成数据集，并改进ChaIR方法的加权损失函数以平衡颜色偏差，实现自动高效修复，减少人工干预。**

- **链接: [http://arxiv.org/pdf/2505.22291v1](http://arxiv.org/pdf/2505.22291v1)**

> **作者:** Saptarshi Neil Sinha; P. Julius Kuehn; Johannes Koppe; Arjan Kuijper; Michael Weinmann
>
> **摘要:** The preservation of early visual arts, particularly color photographs, is challenged by deterioration caused by aging and improper storage, leading to issues like blurring, scratches, color bleeding, and fading defects. In this paper, we present the first approach for the automatic removal of greening color defects in digitized autochrome photographs. Our main contributions include a method based on synthetic dataset generation and the use of generative AI with a carefully designed loss function for the restoration of visual arts. To address the lack of suitable training datasets for analyzing greening defects in damaged autochromes, we introduce a novel approach for accurately simulating such defects in synthetic data. We also propose a modified weighted loss function for the ChaIR method to account for color imbalances between defected and non-defected areas. While existing methods struggle with accurately reproducing original colors and may require significant manual effort, our method allows for efficient restoration with reduced time requirements.
>
---
#### [new 138] PanoWan: Lifting Diffusion Video Generation Models to 360° with Latitude/Longitude-aware Mechanisms
- **分类: cs.CV**

- **简介: 该论文属于全景视频生成任务，解决现有模型难以利用预训练文本-视频模型生成高质量全景视频的问题。提出PanoWan模型，通过纬度采样避免失真、旋转语义去噪及填充解码实现经向无缝衔接，并构建PanoVid数据集，提升生成效果与零样本任务鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.22016v1](http://arxiv.org/pdf/2505.22016v1)**

> **作者:** Yifei Xia; Shuchen Weng; Siqi Yang; Jingqi Liu; Chengxuan Zhu; Minggui Teng; Zijian Jia; Han Jiang; Boxin Shi
>
> **摘要:** Panoramic video generation enables immersive 360{\deg} content creation, valuable in applications that demand scene-consistent world exploration. However, existing panoramic video generation models struggle to leverage pre-trained generative priors from conventional text-to-video models for high-quality and diverse panoramic videos generation, due to limited dataset scale and the gap in spatial feature representations. In this paper, we introduce PanoWan to effectively lift pre-trained text-to-video models to the panoramic domain, equipped with minimal modules. PanoWan employs latitude-aware sampling to avoid latitudinal distortion, while its rotated semantic denoising and padded pixel-wise decoding ensure seamless transitions at longitude boundaries. To provide sufficient panoramic videos for learning these lifted representations, we contribute PanoVid, a high-quality panoramic video dataset with captions and diverse scenarios. Consequently, PanoWan achieves state-of-the-art performance in panoramic video generation and demonstrates robustness for zero-shot downstream tasks.
>
---
#### [new 139] Zero-Shot Vision Encoder Grafting via LLM Surrogates
- **分类: cs.CV**

- **简介: 该论文提出零样本视觉编码器移植方法，优化视觉语言模型（VLM）训练。针对大语言模型（如Llama-70B）解码器导致的高计算成本问题，构建共享嵌入空间的小代理模型（继承大模型浅层），先训练视觉编码器再直接移植至大模型，使成本降低45%，性能媲美全训练。**

- **链接: [http://arxiv.org/pdf/2505.22664v1](http://arxiv.org/pdf/2505.22664v1)**

> **作者:** Kaiyu Yue; Vasu Singla; Menglin Jia; John Kirchenbauer; Rifaa Qadri; Zikui Cai; Abhinav Bhatele; Furong Huang; Tom Goldstein
>
> **备注:** 15 pages
>
> **摘要:** Vision language models (VLMs) typically pair a modestly sized vision encoder with a large language model (LLM), e.g., Llama-70B, making the decoder the primary computational burden during training. To reduce costs, a potential promising strategy is to first train the vision encoder using a small language model before transferring it to the large one. We construct small "surrogate models" that share the same embedding space and representation language as the large target LLM by directly inheriting its shallow layers. Vision encoders trained on the surrogate can then be directly transferred to the larger model, a process we call zero-shot grafting -- when plugged directly into the full-size target LLM, the grafted pair surpasses the encoder-surrogate pair and, on some benchmarks, even performs on par with full decoder training with the target LLM. Furthermore, our surrogate training approach reduces overall VLM training costs by ~45% when using Llama-70B as the decoder.
>
---
#### [new 140] Zooming from Context to Cue: Hierarchical Preference Optimization for Multi-Image MLLMs
- **分类: cs.CV**

- **简介: 该论文针对多图像多模态大模型（MLLM）的跨模态对齐问题，提出分层优化框架CcDPO。通过上下文级全局偏好优化和针尖级局部细节监督，解决多图理解中的上下文遗漏、混淆及误判问题，并构建MultiScope-42k数据集支持训练，实验显示显著减少幻觉并提升多图任务表现。**

- **链接: [http://arxiv.org/pdf/2505.22396v1](http://arxiv.org/pdf/2505.22396v1)**

> **作者:** Xudong Li; Mengdan Zhang; Peixian Chen; Xiawu Zheng; Yan Zhang; Jingyuan Zheng; Yunhang Shen; Ke Li; Chaoyou Fu; Xing Sun; Rongrong Ji
>
> **摘要:** Multi-modal Large Language Models (MLLMs) excel at single-image tasks but struggle with multi-image understanding due to cross-modal misalignment, leading to hallucinations (context omission, conflation, and misinterpretation). Existing methods using Direct Preference Optimization (DPO) constrain optimization to a solitary image reference within the input sequence, neglecting holistic context modeling. We propose Context-to-Cue Direct Preference Optimization (CcDPO), a multi-level preference optimization framework that enhances per-image perception in multi-image settings by zooming into visual clues -- from sequential context to local details. It features: (i) Context-Level Optimization : Re-evaluates cognitive biases underlying MLLMs' multi-image context comprehension and integrates a spectrum of low-cost global sequence preferences for bias mitigation. (ii) Needle-Level Optimization : Directs attention to fine-grained visual details through region-targeted visual prompts and multimodal preference supervision. To support scalable optimization, we also construct MultiScope-42k, an automatically generated dataset with high-quality multi-level preference pairs. Experiments show that CcDPO significantly reduces hallucinations and yields consistent performance gains across general single- and multi-image tasks.
>
---
#### [new 141] CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction
- **分类: cs.CV**

- **简介: 该论文提出CADReview任务及ReCAD框架，解决CAD程序与参考图像不一致的自动检测与修正问题。针对多模态模型在几何组件识别和空间操作上的不足，开发了错误检测与修复框架，并构建2万+程序-图像数据集，实验显示显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2505.22304v1](http://arxiv.org/pdf/2505.22304v1)**

> **作者:** Jiali Chen; Xusen Hei; HongFei Liu; Yuancheng Wei; Zikun Deng; Jiayuan Xie; Yi Cai; Li Qing
>
> **备注:** ACL 2025 main conference
>
> **摘要:** Computer-aided design (CAD) is crucial in prototyping 3D objects through geometric instructions (i.e., CAD programs). In practical design workflows, designers often engage in time-consuming reviews and refinements of these prototypes by comparing them with reference images. To bridge this gap, we introduce the CAD review task to automatically detect and correct potential errors, ensuring consistency between the constructed 3D objects and reference images. However, recent advanced multimodal large language models (MLLMs) struggle to recognize multiple geometric components and perform spatial geometric operations within the CAD program, leading to inaccurate reviews. In this paper, we propose the CAD program repairer (ReCAD) framework to effectively detect program errors and provide helpful feedback on error correction. Additionally, we create a dataset, CADReview, consisting of over 20K program-image pairs, with diverse errors for the CAD review task. Extensive experiments demonstrate that our ReCAD significantly outperforms existing MLLMs, which shows great potential in design applications.
>
---
#### [new 142] CIM-NET: A Video Denoising Deep Neural Network Model Optimized for Computing-in-Memory Architectures
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于视频去噪任务，针对边缘设备部署中实时与能效挑战，提出CIM-NET模型及CIM-CONV操作符，优化计算内存架构，减少MVM操作至1/77，提升推理速度同时保持性能（PSNR 35.11dB）。通过协同设计适配CIM硬件，解决传统模型未考虑架构约束导致的加速受限问题。**

- **链接: [http://arxiv.org/pdf/2505.21522v1](http://arxiv.org/pdf/2505.21522v1)**

> **作者:** Shan Gao; Zhiqiang Wu; Yawen Niu; Xiaotao Li; Qingqing Xu
>
> **摘要:** While deep neural network (DNN)-based video denoising has demonstrated significant performance, deploying state-of-the-art models on edge devices remains challenging due to stringent real-time and energy efficiency requirements. Computing-in-Memory (CIM) chips offer a promising solution by integrating computation within memory cells, enabling rapid matrix-vector multiplication (MVM). However, existing DNN models are often designed without considering CIM architectural constraints, thus limiting their acceleration potential during inference. To address this, we propose a hardware-algorithm co-design framework incorporating two innovations: (1) a CIM-Aware Architecture, CIM-NET, optimized for large receptive field operation and CIM's crossbar-based MVM acceleration; and (2) a pseudo-convolutional operator, CIM-CONV, used within CIM-NET to integrate slide-based processing with fully connected transformations for high-quality feature extraction and reconstruction. This framework significantly reduces the number of MVM operations, improving inference speed on CIM chips while maintaining competitive performance. Experimental results indicate that, compared to the conventional lightweight model FastDVDnet, CIM-NET substantially reduces MVM operations with a slight decrease in denoising performance. With a stride value of 8, CIM-NET reduces MVM operations to 1/77th of the original, while maintaining competitive PSNR (35.11 dB vs. 35.56 dB
>
---
#### [new 143] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?
- **分类: cs.CV**

- **简介: 该论文研究文本生成360度全景图任务，解决视角图像与全景图的领域差距问题。通过分析扩散模型注意力模块，发现值/输出权重矩阵对全景适应更关键，提出UniPano框架，提升性能并减少资源消耗，建立高效基准。**

- **链接: [http://arxiv.org/pdf/2505.22129v1](http://arxiv.org/pdf/2505.22129v1)**

> **作者:** Jinhong Ni; Chang-Bin Zhang; Qiang Zhang; Jing Zhang
>
> **摘要:** Recent prosperity of text-to-image diffusion models, e.g. Stable Diffusion, has stimulated research to adapt them to 360-degree panorama generation. Prior work has demonstrated the feasibility of using conventional low-rank adaptation techniques on pre-trained diffusion models to generate panoramic images. However, the substantial domain gap between perspective and panoramic images raises questions about the underlying mechanisms enabling this empirical success. We hypothesize and examine that the trainable counterparts exhibit distinct behaviors when fine-tuned on panoramic data, and such an adaptation conceals some intrinsic mechanism to leverage the prior knowledge within the pre-trained diffusion models. Our analysis reveals the following: 1) the query and key matrices in the attention modules are responsible for common information that can be shared between the panoramic and perspective domains, thus are less relevant to panorama generation; and 2) the value and output weight matrices specialize in adapting pre-trained knowledge to the panoramic domain, playing a more critical role during fine-tuning for panorama generation. We empirically verify these insights by introducing a simple framework called UniPano, with the objective of establishing an elegant baseline for future research. UniPano not only outperforms existing methods but also significantly reduces memory usage and training time compared to prior dual-branch approaches, making it scalable for end-to-end panorama generation with higher resolution. The code will be released.
>
---
#### [new 144] Is Attention Required for Transformer Inference? Explore Function-preserving Attention Replacement
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于模型优化任务，旨在解决Transformer推理时注意力机制效率低下的问题。提出FAR框架，用可学习的LSTM模块替代预训练模型中的所有注意力层，通过蒸馏和剪枝优化，生成高效LSTM模型。在DeiT上验证，保持ImageNet等任务精度的同时减少参数和延迟。**

- **链接: [http://arxiv.org/pdf/2505.21535v1](http://arxiv.org/pdf/2505.21535v1)**

> **作者:** Yuxin Ren; Maxwell D Collins; Miao Hu; Huanrui Yang
>
> **备注:** 12 pages main paper + 6 pages appendix, 14 figures; submitted to NeurIPS 2025
>
> **摘要:** While transformers excel across vision and language pretraining tasks, their reliance on attention mechanisms poses challenges for inference efficiency, especially on edge and embedded accelerators with limited parallelism and memory bandwidth. Hinted by the observed redundancy of attention at inference time, we hypothesize that though the model learns complicated token dependency through pretraining, the inference-time sequence-to-sequence mapping in each attention layer is actually ''simple'' enough to be represented with a much cheaper function. In this work, we explore FAR, a Function-preserving Attention Replacement framework that replaces all attention blocks in pretrained transformers with learnable sequence-to-sequence modules, exemplified by an LSTM. FAR optimize a multi-head LSTM architecture with a block-wise distillation objective and a global structural pruning framework to achieve a family of efficient LSTM-based models from pretrained transformers. We validate FAR on the DeiT vision transformer family and demonstrate that it matches the accuracy of the original models on ImageNet and multiple downstream tasks with reduced parameters and latency. Further analysis shows that FAR preserves the semantic token relationships and the token-to-token correlation learned in the transformer's attention module.
>
---
#### [new 145] ProCrop: Learning Aesthetic Image Cropping from Professional Compositions
- **分类: cs.CV**

- **简介: 该论文属于图像美学裁剪任务，旨在解决现有方法多样性不足或依赖标注数据的问题。提出ProCrop方法，通过融合专业摄影作品与输入图像特征指导裁剪，并构建24万弱标注数据集，提升裁剪质量和多样性，实验显示其性能超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22490v1](http://arxiv.org/pdf/2505.22490v1)**

> **作者:** Ke Zhang; Tianyu Ding; Jiachen Jiang; Tianyi Chen; Ilya Zharkov; Vishal M. Patel; Luming Liang
>
> **备注:** 16 pages, 15 figures
>
> **摘要:** Image cropping is crucial for enhancing the visual appeal and narrative impact of photographs, yet existing rule-based and data-driven approaches often lack diversity or require annotated training data. We introduce ProCrop, a retrieval-based method that leverages professional photography to guide cropping decisions. By fusing features from professional photographs with those of the query image, ProCrop learns from professional compositions, significantly boosting performance. Additionally, we present a large-scale dataset of 242K weakly-annotated images, generated by out-painting professional images and iteratively refining diverse crop proposals. This composition-aware dataset generation offers diverse high-quality crop proposals guided by aesthetic principles and becomes the largest publicly available dataset for image cropping. Extensive experiments show that ProCrop significantly outperforms existing methods in both supervised and weakly-supervised settings. Notably, when trained on the new dataset, our ProCrop surpasses previous weakly-supervised methods and even matches fully supervised approaches. Both the code and dataset will be made publicly available to advance research in image aesthetics and composition analysis.
>
---
#### [new 146] Any-to-Bokeh: One-Step Video Bokeh via Multi-Plane Image Guided Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Any-to-Bokeh框架，属于视频景深生成任务。解决现有方法无法控制焦点平面、调整虚化强度及视频时间闪烁问题。通过多平面图像引导的单步扩散模型，结合渐进训练和预训练三维先验，实现高质量、时序一致的深度感知视频虚化效果。**

- **链接: [http://arxiv.org/pdf/2505.21593v1](http://arxiv.org/pdf/2505.21593v1)**

> **作者:** Yang Yang; Siming Zheng; Jinwei Chen; Boxi Wu; Xiaofei He; Deng Cai; Bo Li; Peng-Tao Jiang
>
> **备注:** project page: https://vivocameraresearch.github.io/any2bokeh/
>
> **摘要:** Recent advances in diffusion based editing models have enabled realistic camera simulation and image-based bokeh, but video bokeh remains largely unexplored. Existing video editing models cannot explicitly control focus planes or adjust bokeh intensity, limiting their applicability for controllable optical effects. Moreover, naively extending image-based bokeh methods to video often results in temporal flickering and unsatisfactory edge blur transitions due to the lack of temporal modeling and generalization capability. To address these challenges, we propose a novel one-step video bokeh framework that converts arbitrary input videos into temporally coherent, depth-aware bokeh effects. Our method leverages a multi-plane image (MPI) representation constructed through a progressively widening depth sampling function, providing explicit geometric guidance for depth-dependent blur synthesis. By conditioning a single-step video diffusion model on MPI layers and utilizing the strong 3D priors from pre-trained models such as Stable Video Diffusion, our approach achieves realistic and consistent bokeh effects across diverse scenes. Additionally, we introduce a progressive training strategy to enhance temporal consistency, depth robustness, and detail preservation. Extensive experiments demonstrate that our method produces high-quality, controllable bokeh effects and achieves state-of-the-art performance on multiple evaluation benchmarks.
>
---
#### [new 147] Sherlock: Self-Correcting Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型（VLM）推理任务，旨在解决其推理易出错、依赖大量标注数据及泛化能力差的问题。提出Sherlock框架，通过轨迹级自纠目标、视觉扰动构建偏好数据及动态β调参，使模型仅用20k标注数据获得自纠能力并持续自我优化，在8个基准测试中超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.22651v1](http://arxiv.org/pdf/2505.22651v1)**

> **作者:** Yi Ding; Ruqi Zhang
>
> **备注:** 27 pages
>
> **摘要:** Reasoning Vision-Language Models (VLMs) have shown promising performance on complex multimodal tasks. However, they still face significant challenges: they are highly sensitive to reasoning errors, require large volumes of annotated data or accurate verifiers, and struggle to generalize beyond specific domains. To address these limitations, we explore self-correction as a strategy to enhance reasoning VLMs. We first conduct an in-depth analysis of reasoning VLMs' self-correction abilities and identify key gaps. Based on our findings, we introduce Sherlock, a self-correction and self-improvement training framework. Sherlock introduces a trajectory-level self-correction objective, a preference data construction method based on visual perturbation, and a dynamic $\beta$ for preference tuning. Once the model acquires self-correction capabilities using only 20k randomly sampled annotated data, it continues to self-improve without external supervision. Built on the Llama3.2-Vision-11B model, Sherlock achieves remarkable results across eight benchmarks, reaching an average accuracy of 64.1 with direct generation and 65.4 after self-correction. It outperforms LLaVA-CoT (63.2), Mulberry (63.9), and LlamaV-o1 (63.4) while using less than 20% of the annotated data.
>
---
#### [new 148] Vision Meets Language: A RAG-Augmented YOLOv8 Framework for Coffee Disease Diagnosis and Farmer Assistance
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出结合YOLOv8与RAG增强的LLM框架，用于咖啡叶病害诊断及农民支持。针对传统农业低效和LLM幻觉问题，通过视觉检测病害、语言模型生成精准诊断及环保治疗方案，减少农药使用，提供用户友好界面，推动精准农业应用。**

- **链接: [http://arxiv.org/pdf/2505.21544v1](http://arxiv.org/pdf/2505.21544v1)**

> **作者:** Semanto Mondal
>
> **备注:** There are 14 pages, 8 figures
>
> **摘要:** As a social being, we have an intimate bond with the environment. A plethora of things in human life, such as lifestyle, health, and food are dependent on the environment and agriculture. It comes under our responsibility to support the environment as well as agriculture. However, traditional farming practices often result in inefficient resource use and environmental challenges. To address these issues, precision agriculture has emerged as a promising approach that leverages advanced technologies to optimise agricultural processes. In this work, a hybrid approach is proposed that combines the three different potential fields of model AI: object detection, large language model (LLM), and Retrieval-Augmented Generation (RAG). In this novel framework, we have tried to combine the vision and language models to work together to identify potential diseases in the tree leaf. This study introduces a novel AI-based precision agriculture system that uses Retrieval Augmented Generation (RAG) to provide context-aware diagnoses and natural language processing (NLP) and YOLOv8 for crop disease detection. The system aims to tackle major issues with large language models (LLMs), especially hallucinations and allows for adaptive treatment plans and real-time disease detection. The system provides an easy-to-use interface to the farmers, which they can use to detect the different diseases related to coffee leaves by just submitting the image of the affected leaf the model will detect the diseases as well as suggest potential remediation methodologies which aim to lower the use of pesticides, preserving livelihoods, and encouraging environmentally friendly methods. With an emphasis on scalability, dependability, and user-friendliness, the project intends to improve RAG-integrated object detection systems for wider agricultural applications in the future.
>
---
#### [new 149] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于LiDAR 3D场景生成任务，针对现有范围视图方法无法生成语义标签且存在跨模态不一致问题，提出SPIRAL模型，通过扩散模型同步生成深度、反射率及语义图，并设计新评估指标。实验显示其参数最少且性能领先，生成数据可有效用于分割任务的数据增强。**

- **链接: [http://arxiv.org/pdf/2505.22643v1](http://arxiv.org/pdf/2505.22643v1)**

> **作者:** Dekai Zhu; Yixuan Hu; Youquan Liu; Dongyue Lu; Lingdong Kong; Slobodan Ilic
>
> **摘要:** Leveraging recent diffusion models, LiDAR-based large-scale 3D scene generation has achieved great success. While recent voxel-based approaches can generate both geometric structures and semantic labels, existing range-view methods are limited to producing unlabeled LiDAR scenes. Relying on pretrained segmentation models to predict the semantic maps often results in suboptimal cross-modal consistency. To address this limitation while preserving the advantages of range-view representations, such as computational efficiency and simplified network design, we propose Spiral, a novel range-view LiDAR diffusion model that simultaneously generates depth, reflectance images, and semantic maps. Furthermore, we introduce novel semantic-aware metrics to evaluate the quality of the generated labeled range-view data. Experiments on the SemanticKITTI and nuScenes datasets demonstrate that Spiral achieves state-of-the-art performance with the smallest parameter size, outperforming two-step methods that combine the generative and segmentation models. Additionally, we validate that range images generated by Spiral can be effectively used for synthetic data augmentation in the downstream segmentation training, significantly reducing the labeling effort on LiDAR data.
>
---
#### [new 150] AquaMonitor: A multimodal multi-view image sequence dataset for real-life aquatic invertebrate biodiversity monitoring
- **分类: cs.CV**

- **简介: 该论文提出AquaMonitor数据集，首个标准化采集的大型水生无脊椎动物多模态图像数据集，解决现有数据集缺乏标准化及专注水生生物的问题。包含270万图像、DNA等多模态数据，定义监测、分类、少样本三个基准任务，为自动化生物多样性评估提供基线方法。**

- **链接: [http://arxiv.org/pdf/2505.22065v1](http://arxiv.org/pdf/2505.22065v1)**

> **作者:** Mikko Impiö; Philipp M. Rehsen; Tiina Laamanen; Arne J. Beermann; Florian Leese; Jenni Raitoharju
>
> **摘要:** This paper presents the AquaMonitor dataset, the first large computer vision dataset of aquatic invertebrates collected during routine environmental monitoring. While several large species identification datasets exist, they are rarely collected using standardized collection protocols, and none focus on aquatic invertebrates, which are particularly laborious to collect. For AquaMonitor, we imaged all specimens from two years of monitoring whenever imaging was possible given practical limitations. The dataset enables the evaluation of automated identification methods for real-life monitoring purposes using a realistically challenging and unbiased setup. The dataset has 2.7M images from 43,189 specimens, DNA sequences for 1358 specimens, and dry mass and size measurements for 1494 specimens, making it also one of the largest biological multi-view and multimodal datasets to date. We define three benchmark tasks and provide strong baselines for these: 1) Monitoring benchmark, reflecting real-life deployment challenges such as open-set recognition, distribution shift, and extreme class imbalance, 2) Classification benchmark, which follows a standard fine-grained visual categorization setup, and 3) Few-shot benchmark, which targets classes with only few training examples from very fine-grained categories. Advancements on the Monitoring benchmark can directly translate to improvement of aquatic biodiversity monitoring, which is an important component of regular legislative water quality assessment in many countries.
>
---
#### [new 151] DvD: Unleashing a Generative Paradigm for Document Dewarping via Coordinates-based Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出DvD模型，针对文档去扭曲任务，解决复杂文档结构保持难题。通过坐标级扩散模型生成变形映射，并设计时间动态条件优化机制提升结构保真度，同时构建大规模基准AnyPhotoDoc6300，实验显示其效率与性能优势。**

- **链接: [http://arxiv.org/pdf/2505.21975v1](http://arxiv.org/pdf/2505.21975v1)**

> **作者:** Weiguang Zhang; Huangcheng Lu; Maizhen Ning; Xiaowei Huang; Wei Wang; Kaizhu Huang; Qiufeng Wang
>
> **摘要:** Document dewarping aims to rectify deformations in photographic document images, thus improving text readability, which has attracted much attention and made great progress, but it is still challenging to preserve document structures. Given recent advances in diffusion models, it is natural for us to consider their potential applicability to document dewarping. However, it is far from straightforward to adopt diffusion models in document dewarping due to their unfaithful control on highly complex document images (e.g., 2000$\times$3000 resolution). In this paper, we propose DvD, the first generative model to tackle document \textbf{D}ewarping \textbf{v}ia a \textbf{D}iffusion framework. To be specific, DvD introduces a coordinate-level denoising instead of typical pixel-level denoising, generating a mapping for deformation rectification. In addition, we further propose a time-variant condition refinement mechanism to enhance the preservation of document structures. In experiments, we find that current document dewarping benchmarks can not evaluate dewarping models comprehensively. To this end, we present AnyPhotoDoc6300, a rigorously designed large-scale document dewarping benchmark comprising 6,300 real image pairs across three distinct domains, enabling fine-grained evaluation of dewarping models. Comprehensive experiments demonstrate that our proposed DvD can achieve state-of-the-art performance with acceptable computational efficiency on multiple metrics across various benchmarks including DocUNet, DIR300, and AnyPhotoDoc6300. The new benchmark and code will be publicly available.
>
---
#### [new 152] 3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于具身智能任务，解决大语言模型在3D动态环境中的长期时空记忆不足问题。提出3DMem-Bench基准测试和3DLLM-Mem模型，通过工作记忆令牌查询与选择性融合 episodic记忆中的时空特征，提升复杂环境中的任务执行效率，成功率达16.5%提升。**

- **链接: [http://arxiv.org/pdf/2505.22657v1](http://arxiv.org/pdf/2505.22657v1)**

> **作者:** Wenbo Hu; Yining Hong; Yanjun Wang; Leison Gao; Zibu Wei; Xingcheng Yao; Nanyun Peng; Yonatan Bitton; Idan Szpektor; Kai-Wei Chang
>
> **备注:** demos at: https://3dllm-mem.github.io
>
> **摘要:** Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks.
>
---
#### [new 153] Geometric Feature Prompting of Image Segmentation Models
- **分类: cs.CV**

- **简介: 该论文提出几何特征驱动的提示生成方法（GeomPrompt），用于改进SAM模型在植物根系图像分割任务中的自动化处理。针对传统根系分割依赖人工标注、效率低且主观的问题，该工作设计基于局部脊线特征的提示点生成器，通过少量几何提示提升分割精度，并开源工具实现。**

- **链接: [http://arxiv.org/pdf/2505.21644v1](http://arxiv.org/pdf/2505.21644v1)**

> **作者:** Kenneth Ball; Erin Taylor; Nirav Patel; Andrew Bartels; Gary Koplik; James Polly; Jay Hineman
>
> **摘要:** Advances in machine learning, especially the introduction of transformer architectures and vision transformers, have led to the development of highly capable computer vision foundation models. The segment anything model (known colloquially as SAM and more recently SAM 2), is a highly capable foundation model for segmentation of natural images and has been further applied to medical and scientific image segmentation tasks. SAM relies on prompts -- points or regions of interest in an image -- to generate associated segmentations. In this manuscript we propose the use of a geometrically motivated prompt generator to produce prompt points that are colocated with particular features of interest. Focused prompting enables the automatic generation of sensitive and specific segmentations in a scientific image analysis task using SAM with relatively few point prompts. The image analysis task examined is the segmentation of plant roots in rhizotron or minirhizotron images, which has historically been a difficult task to automate. Hand annotation of rhizotron images is laborious and often subjective; SAM, initialized with GeomPrompt local ridge prompts has the potential to dramatically improve rhizotron image processing. The authors have concurrently released an open source software suite called geomprompt https://pypi.org/project/geomprompt/ that can produce point prompts in a format that enables direct integration with the segment-anything package.
>
---
#### [new 154] Guess the Age of Photos: An Interactive Web Platform for Historical Image Age Estimation
- **分类: cs.CV**

- **简介: 该论文开发了Guess the Age of Photos互动平台，通过游戏化模式（单图年份预测和双图年代对比）让用户参与历史照片年代估计，解决公众参与历史图像分析及计算机视觉数据收集问题。基于10,150张图像数据集，采用Flask等技术构建平台，实验显示用户相对比较准确率（65.9%）显著高于绝对年份猜测（25.6%），兼具历史教育与视觉算法研究价值。**

- **链接: [http://arxiv.org/pdf/2505.22031v1](http://arxiv.org/pdf/2505.22031v1)**

> **作者:** Hasan Yucedag; Adam Jatowt
>
> **备注:** 4 Pages,4 figures and 1 system architect
>
> **摘要:** This paper introduces Guess the Age of Photos, a web platform engaging users in estimating the years of historical photographs through two gamified modes: Guess the Year (predicting a single image's year) and Timeline Challenge (comparing two images to identify the older). Built with Python, Flask, Bootstrap, and PostgreSQL, it uses a 10,150-image subset of the Date Estimation in the Wild dataset (1930-1999). Features like dynamic scoring and leaderboards boost engagement. Evaluated with 113 users and 15,473 gameplays, the platform earned a 4.25/5 satisfaction rating. Users excelled in relative comparisons (65.9% accuracy) over absolute year guesses (25.6% accuracy), with older decades easier to identify. The platform serves as an educational tool, fostering historical awareness and analytical skills via interactive exploration of visual heritage. Furthermore, the platform provides a valuable resource for studying human perception of temporal cues in images and could be used to generate annotated data for training and evaluating computer vision models.
>
---
#### [new 155] PathFL: Multi-Alignment Federated Learning for Pathology Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出PathFL框架，针对多中心病理图像分割中的异构性问题（如成像模态、器官、设备差异），通过图像风格增强、特征对齐及分层模型聚合三层对齐策略，提升模型泛化能力，在多个异构数据集上验证了有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22522v1](http://arxiv.org/pdf/2505.22522v1)**

> **作者:** Yuan Zhang; Feng Chen; Yaolei Qi; Guanyu Yang; Huazhu Fu
>
> **备注:** 17 pages, 5 figures; Accepted by MedIA
>
> **摘要:** Pathology image segmentation across multiple centers encounters significant challenges due to diverse sources of heterogeneity including imaging modalities, organs, and scanning equipment, whose variability brings representation bias and impedes the development of generalizable segmentation models. In this paper, we propose PathFL, a novel multi-alignment Federated Learning framework for pathology image segmentation that addresses these challenges through three-level alignment strategies of image, feature, and model aggregation. Firstly, at the image level, a collaborative style enhancement module aligns and diversifies local data by facilitating style information exchange across clients. Secondly, at the feature level, an adaptive feature alignment module ensures implicit alignment in the representation space by infusing local features with global insights, promoting consistency across heterogeneous client features learning. Finally, at the model aggregation level, a stratified similarity aggregation strategy hierarchically aligns and aggregates models on the server, using layer-specific similarity to account for client discrepancies and enhance global generalization. Comprehensive evaluations on four sets of heterogeneous pathology image datasets, encompassing cross-source, cross-modality, cross-organ, and cross-scanner variations, validate the effectiveness of our PathFL in achieving better performance and robustness against data heterogeneity.
>
---
#### [new 156] MObyGaze: a film dataset of multimodal objectification densely annotated by experts
- **分类: cs.CV**

- **简介: 该论文提出MObyGaze数据集，通过专家标注多模态影视片段，量化性别客体化现象。任务为分析视觉、语音、音频的时序模式以识别客体化，解决影视中性别刻板印象的量化难题。工作包括构建含6072个标注片段的数据集，定义5个子维度，评估多模态模型并开源资源。**

- **链接: [http://arxiv.org/pdf/2505.22084v1](http://arxiv.org/pdf/2505.22084v1)**

> **作者:** Julie Tores; Elisa Ancarani; Lucile Sassatelli; Hui-Yin Wu; Clement Bergman; Lea Andolfi; Victor Ecrement; Remy Sun; Frederic Precioso; Thierry Devars; Magali Guaresi; Virginie Julliard; Sarah Lecossais
>
> **摘要:** Characterizing and quantifying gender representation disparities in audiovisual storytelling contents is necessary to grasp how stereotypes may perpetuate on screen. In this article, we consider the high-level construct of objectification and introduce a new AI task to the ML community: characterize and quantify complex multimodal (visual, speech, audio) temporal patterns producing objectification in films. Building on film studies and psychology, we define the construct of objectification in a structured thesaurus involving 5 sub-constructs manifesting through 11 concepts spanning 3 modalities. We introduce the Multimodal Objectifying Gaze (MObyGaze) dataset, made of 20 movies annotated densely by experts for objectification levels and concepts over freely delimited segments: it amounts to 6072 segments over 43 hours of video with fine-grained localization and categorization. We formulate different learning tasks, propose and investigate best ways to learn from the diversity of labels among a low number of annotators, and benchmark recent vision, text and audio models, showing the feasibility of the task. We make our code and our dataset available to the community and described in the Croissant format: https://anonymous.4open.science/r/MObyGaze-F600/.
>
---
#### [new 157] Right Side Up? Disentangling Orientation Understanding in MLLMs with Fine-grained Multi-axis Perception Tasks
- **分类: cs.CV**

- **简介: 该论文提出DORI基准，评估多模态模型在物体方向理解（如对齐、旋转、相对方向、典型方向）上的能力。针对现有方法未明确区分方向与场景理解的问题，通过多任务测试揭示模型在角度估计、视角变化等任务中存在系统性不足，表明需改进3D空间表征。**

- **链接: [http://arxiv.org/pdf/2505.21649v1](http://arxiv.org/pdf/2505.21649v1)**

> **作者:** Keanu Nichols; Nazia Tasnim; Yan Yuting; Nicholas Ikechukwu; Elva Zou; Deepti Ghadiyaram; Bryan Plummer
>
> **摘要:** Object orientation understanding represents a fundamental challenge in visual perception critical for applications like robotic manipulation and augmented reality. Current vision-language benchmarks fail to isolate this capability, often conflating it with positional relationships and general scene understanding. We introduce DORI (Discriminative Orientation Reasoning Intelligence), a comprehensive benchmark establishing object orientation perception as a primary evaluation target. DORI assesses four dimensions of orientation comprehension: frontal alignment, rotational transformations, relative directional relationships, and canonical orientation understanding. Through carefully curated tasks from 11 datasets spanning 67 object categories across synthetic and real-world scenarios, DORI provides insights on how multi-modal systems understand object orientations. Our evaluation of 15 state-of-the-art vision-language models reveals critical limitations: even the best models achieve only 54.2% accuracy on coarse tasks and 33.0% on granular orientation judgments, with performance deteriorating for tasks requiring reference frame shifts or compound rotations. These findings demonstrate the need for dedicated orientation representation mechanisms, as models show systematic inability to perform precise angular estimations, track orientation changes across viewpoints, and understand compound rotations - suggesting limitations in their internal 3D spatial representations. As the first diagnostic framework specifically designed for orientation awareness in multimodal systems, DORI offers implications for improving robotic control, 3D scene reconstruction, and human-AI interaction in physical environments. DORI data: https://huggingface.co/datasets/appledora/DORI-Benchmark
>
---
#### [new 158] Task-Driven Implicit Representations for Automated Design of LiDAR Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出基于任务驱动的隐式表示方法，解决LiDAR系统设计复杂、耗时且依赖人工的问题。通过构建六维连续设计空间，利用流式生成模型学习任务导向的传感器配置密度，并用EM算法优化参数，实现高效自动化设计，验证于3D视觉任务如人脸扫描、机器人追踪等。**

- **链接: [http://arxiv.org/pdf/2505.22344v1](http://arxiv.org/pdf/2505.22344v1)**

> **作者:** Nikhil Behari; Aaron Young; Akshat Dave; Ramesh Raskar
>
> **摘要:** Imaging system design is a complex, time-consuming, and largely manual process; LiDAR design, ubiquitous in mobile devices, autonomous vehicles, and aerial imaging platforms, adds further complexity through unique spatial and temporal sampling requirements. In this work, we propose a framework for automated, task-driven LiDAR system design under arbitrary constraints. To achieve this, we represent LiDAR configurations in a continuous six-dimensional design space and learn task-specific implicit densities in this space via flow-based generative modeling. We then synthesize new LiDAR systems by modeling sensors as parametric distributions in 6D space and fitting these distributions to our learned implicit density using expectation-maximization, enabling efficient, constraint-aware LiDAR system design. We validate our method on diverse tasks in 3D vision, enabling automated LiDAR system design across real-world-inspired applications in face scanning, robotic tracking, and object detection.
>
---
#### [new 159] From Failures to Fixes: LLM-Driven Scenario Repair for Self-Evolving Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶系统自进化任务，旨在解决现有方法在修复复杂场景失败案例时的适应性与语义相关性不足问题。提出SERA框架，利用LLM分析失败日志、检索匹配场景并优化推荐，通过小样本微调提升系统性能，实验验证其在安全关键场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22067v1](http://arxiv.org/pdf/2505.22067v1)**

> **作者:** Xinyu Xia; Xingjun Ma; Yunfeng Hu; Ting Qu; Hong Chen; Xun Gong
>
> **摘要:** Ensuring robust and generalizable autonomous driving requires not only broad scenario coverage but also efficient repair of failure cases, particularly those related to challenging and safety-critical scenarios. However, existing scenario generation and selection methods often lack adaptivity and semantic relevance, limiting their impact on performance improvement. In this paper, we propose \textbf{SERA}, an LLM-powered framework that enables autonomous driving systems to self-evolve by repairing failure cases through targeted scenario recommendation. By analyzing performance logs, SERA identifies failure patterns and dynamically retrieves semantically aligned scenarios from a structured bank. An LLM-based reflection mechanism further refines these recommendations to maximize relevance and diversity. The selected scenarios are used for few-shot fine-tuning, enabling targeted adaptation with minimal data. Experiments on the benchmark show that SERA consistently improves key metrics across multiple autonomous driving baselines, demonstrating its effectiveness and generalizability under safety-critical conditions.
>
---
#### [new 160] Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Cross-modal RAG框架，针对复杂文本到图像生成中现有方法无法整合多图像元素的问题，通过子维度分解查询与图像，结合稀疏与密集检索策略获取互补图像，并指导模型生成。实验显示其在质量和效率上优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21956v1](http://arxiv.org/pdf/2505.21956v1)**

> **作者:** Mengdan Zhu; Senhao Cheng; Guangji Bai; Yifei Zhang; Liang Zhao
>
> **摘要:** Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy - combining a sub-dimensional sparse retriever with a dense retriever - to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to specific subqueries, ensuring subquery-aware image synthesis. Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal RAG significantly outperforms existing baselines in both retrieval and generation quality, while maintaining high efficiency.
>
---
#### [new 161] ImageReFL: Balancing Quality and Diversity in Human-Aligned Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于扩散模型优化任务，旨在解决奖励微调导致图像多样性下降的问题。提出"结合生成"策略（后期应用奖励模型，保留早期基础模型结构）和ImageReFL方法（基于真实图像与正则化训练），平衡生成质量与多样性，实验表明优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.22569v1](http://arxiv.org/pdf/2505.22569v1)**

> **作者:** Dmitrii Sorokin; Maksim Nakhodnov; Andrey Kuznetsov; Aibek Alanov
>
> **备注:** The source code can be found at https://github.com/ControlGenAI/ImageReFL
>
> **摘要:** Recent advances in diffusion models have led to impressive image generation capabilities, but aligning these models with human preferences remains challenging. Reward-based fine-tuning using models trained on human feedback improves alignment but often harms diversity, producing less varied outputs. In this work, we address this trade-off with two contributions. First, we introduce \textit{combined generation}, a novel sampling strategy that applies a reward-tuned diffusion model only in the later stages of the generation process, while preserving the base model for earlier steps. This approach mitigates early-stage overfitting and helps retain global structure and diversity. Second, we propose \textit{ImageReFL}, a fine-tuning method that improves image diversity with minimal loss in quality by training on real images and incorporating multiple regularizers, including diffusion and ReFL losses. Our approach outperforms conventional reward tuning methods on standard quality and diversity metrics. A user study further confirms that our method better balances human preference alignment and visual diversity. The source code can be found at https://github.com/ControlGenAI/ImageReFL .
>
---
#### [new 162] Multipath cycleGAN for harmonization of paired and unpaired low-dose lung computed tomography reconstruction kernels
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出多路径cycleGAN模型，解决CT重建核差异导致的定量测量偏差问题。通过混合配对/不配对低剂量肺CT数据训练，协调不同核图像，提升肺气肿量化一致性并保持解剖结构，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.22568v1](http://arxiv.org/pdf/2505.22568v1)**

> **作者:** Aravind R. Krishnan; Thomas Z. Li; Lucas W. Remedios; Michael E. Kim; Chenyu Gao; Gaurav Rudravaram; Elyssa M. McMaster; Adam M. Saunders; Shunxing Bao; Kaiwen Xu; Lianrui Zuo; Kim L. Sandler; Fabien Maldonado; Yuankai Huo; Bennett A. Landman
>
> **摘要:** Reconstruction kernels in computed tomography (CT) affect spatial resolution and noise characteristics, introducing systematic variability in quantitative imaging measurements such as emphysema quantification. Choosing an appropriate kernel is therefore essential for consistent quantitative analysis. We propose a multipath cycleGAN model for CT kernel harmonization, trained on a mixture of paired and unpaired data from a low-dose lung cancer screening cohort. The model features domain-specific encoders and decoders with a shared latent space and uses discriminators tailored for each domain.We train the model on 42 kernel combinations using 100 scans each from seven representative kernels in the National Lung Screening Trial (NLST) dataset. To evaluate performance, 240 scans from each kernel are harmonized to a reference soft kernel, and emphysema is quantified before and after harmonization. A general linear model assesses the impact of age, sex, smoking status, and kernel on emphysema. We also evaluate harmonization from soft kernels to a reference hard kernel. To assess anatomical consistency, we compare segmentations of lung vessels, muscle, and subcutaneous adipose tissue generated by TotalSegmentator between harmonized and original images. Our model is benchmarked against traditional and switchable cycleGANs. For paired kernels, our approach reduces bias in emphysema scores, as seen in Bland-Altman plots (p<0.05). For unpaired kernels, harmonization eliminates confounding differences in emphysema (p>0.05). High Dice scores confirm preservation of muscle and fat anatomy, while lung vessel overlap remains reasonable. Overall, our shared latent space multipath cycleGAN enables robust harmonization across paired and unpaired CT kernels, improving emphysema quantification and preserving anatomical fidelity.
>
---
#### [new 163] Efficiently Enhancing General Agents With Hierarchical-categorical Memory
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于构建通用多模态代理任务，解决现有方法计算成本高或无法持续学习的问题。提出EHC模型，通过分层记忆检索模块（HMR）快速存储/检索信息，及任务导向经验学习模块（TOEL）分类经验提取模式，实现无需参数更新的持续学习，提升复杂任务处理能力。**

- **链接: [http://arxiv.org/pdf/2505.22006v1](http://arxiv.org/pdf/2505.22006v1)**

> **作者:** Changze Qiao; Mingming Lu
>
> **摘要:** With large language models (LLMs) demonstrating remarkable capabilities, there has been a surge in research on leveraging LLMs to build general-purpose multi-modal agents. However, existing approaches either rely on computationally expensive end-to-end training using large-scale multi-modal data or adopt tool-use methods that lack the ability to continuously learn and adapt to new environments. In this paper, we introduce EHC, a general agent capable of learning without parameter updates. EHC consists of a Hierarchical Memory Retrieval (HMR) module and a Task-Category Oriented Experience Learning (TOEL) module. The HMR module facilitates rapid retrieval of relevant memories and continuously stores new information without being constrained by memory capacity. The TOEL module enhances the agent's comprehension of various task characteristics by classifying experiences and extracting patterns across different categories. Extensive experiments conducted on multiple standard datasets demonstrate that EHC outperforms existing methods, achieving state-of-the-art performance and underscoring its effectiveness as a general agent for handling complex multi-modal tasks.
>
---
#### [new 164] CogAD: Cognitive-Hierarchy Guided End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于端到端自动驾驶任务，旨在解决现有方法与人类认知不一致的问题。提出CogAD模型，通过分层感知（全局-局部环境理解）和意图驱动的多模式轨迹规划，结合双层不确定性建模，提升复杂场景的泛化与规划能力，实验显示其性能优越。**

- **链接: [http://arxiv.org/pdf/2505.21581v1](http://arxiv.org/pdf/2505.21581v1)**

> **作者:** Zhennan Wang; Jianing Teng; Canqun Xiang; Kangliang Chen; Xing Pan; Lu Deng; Weihao Gu
>
> **摘要:** While end-to-end autonomous driving has advanced significantly, prevailing methods remain fundamentally misaligned with human cognitive principles in both perception and planning. In this paper, we propose CogAD, a novel end-to-end autonomous driving model that emulates the hierarchical cognition mechanisms of human drivers. CogAD implements dual hierarchical mechanisms: global-to-local context processing for human-like perception and intent-conditioned multi-mode trajectory generation for cognitively-inspired planning. The proposed method demonstrates three principal advantages: comprehensive environmental understanding through hierarchical perception, robust planning exploration enabled by multi-level planning, and diverse yet reasonable multi-modal trajectory generation facilitated by dual-level uncertainty modeling. Extensive experiments on nuScenes and Bench2Drive demonstrate that CogAD achieves state-of-the-art performance in end-to-end planning, exhibiting particular superiority in long-tail scenarios and robust generalization to complex real-world driving conditions.
>
---
#### [new 165] Detecting Cultural Differences in News Video Thumbnails via Computational Aesthetics
- **分类: cs.CY; cs.CV**

- **简介: 该论文通过计算美学分析检测中美新闻视频缩略图的文化差异，提出两步法：先按内容聚类视觉主题，再比较美学特征。基于2400张中美频道缩略图（涉及疫情和乌克兰冲突），发现中美在色彩、构图等风格上的显著差异，揭示文化偏好，并为视觉 propaganda 分析提供基准。**

- **链接: [http://arxiv.org/pdf/2505.21912v1](http://arxiv.org/pdf/2505.21912v1)**

> **作者:** Marvin Limpijankit; John Kender
>
> **摘要:** We propose a two-step approach for detecting differences in the style of images across sources of differing cultural affinity, where images are first clustered into finer visual themes based on content before their aesthetic features are compared. We test this approach on 2,400 YouTube video thumbnails taken equally from two U.S. and two Chinese YouTube channels, and relating equally to COVID-19 and the Ukraine conflict. Our results suggest that while Chinese thumbnails are less formal and more candid, U.S. channels tend to use more deliberate, proper photographs as thumbnails. In particular, U.S. thumbnails are less colorful, more saturated, darker, more finely detailed, less symmetric, sparser, less varied, and more up close and personal than Chinese thumbnails. We suggest that most of these differences reflect cultural preferences, and that our methods and observations can serve as a baseline against which suspected visual propaganda can be computed and compared.
>
---
#### [new 166] Cascaded 3D Diffusion Models for Whole-body 3D 18-F FDG PET/CT synthesis from Demographics
- **分类: eess.IV; cs.CV; cs.GR**

- **简介: 该论文提出级联3D扩散模型，从人口统计学数据生成全身PET/CT图像。旨在解决传统确定性模板生成的局限性，提升合成影像的解剖和代谢准确性。方法分为两阶段：先生成低分辨率影像，再通过超分辨率细化。实验显示合成数据与真实数据高度一致（误差<5%），为临床和研究提供可扩展的合成影像方案。**

- **链接: [http://arxiv.org/pdf/2505.22489v1](http://arxiv.org/pdf/2505.22489v1)**

> **作者:** Siyeop Yoon; Sifan Song; Pengfei Jin; Matthew Tivnan; Yujin Oh; Sekeun Kim; Dufan Wu; Xiang Li; Quanzheng Li
>
> **备注:** MICCAI2025 Submitted version
>
> **摘要:** We propose a cascaded 3D diffusion model framework to synthesize high-fidelity 3D PET/CT volumes directly from demographic variables, addressing the growing need for realistic digital twins in oncologic imaging, virtual trials, and AI-driven data augmentation. Unlike deterministic phantoms, which rely on predefined anatomical and metabolic templates, our method employs a two-stage generative process. An initial score-based diffusion model synthesizes low-resolution PET/CT volumes from demographic variables alone, providing global anatomical structures and approximate metabolic activity. This is followed by a super-resolution residual diffusion model that refines spatial resolution. Our framework was trained on 18-F FDG PET/CT scans from the AutoPET dataset and evaluated using organ-wise volume and standardized uptake value (SUV) distributions, comparing synthetic and real data between demographic subgroups. The organ-wise comparison demonstrated strong concordance between synthetic and real images. In particular, most deviations in metabolic uptake values remained within 3-5% of the ground truth in subgroup analysis. These findings highlight the potential of cascaded 3D diffusion models to generate anatomically and metabolically accurate PET/CT images, offering a robust alternative to traditional phantoms and enabling scalable, population-informed synthetic imaging for clinical and research applications.
>
---
#### [new 167] Large-Area Fabrication-aware Computational Diffractive Optics
- **分类: physics.optics; cs.CV; cs.ET; cs.GR**

- **简介: 该论文属于计算衍射光学设计任务，旨在解决模拟与实际制造差异导致的光学系统应用局限。提出包含神经光刻预测模型和分布式计算框架的制造感知设计流程，实现大规模（厘米级）衍射光学元件的精准端到端优化，提升全息成像等应用的实用性。**

- **链接: [http://arxiv.org/pdf/2505.22313v1](http://arxiv.org/pdf/2505.22313v1)**

> **作者:** Kaixuan Wei; Hector A. Jimenez-Romero; Hadi Amata; Jipeng Sun; Qiang Fu; Felix Heide; Wolfgang Heidrich
>
> **摘要:** Differentiable optics, as an emerging paradigm that jointly optimizes optics and (optional) image processing algorithms, has made innovative optical designs possible across a broad range of applications. Many of these systems utilize diffractive optical components (DOEs) for holography, PSF engineering, or wavefront shaping. Existing approaches have, however, mostly remained limited to laboratory prototypes, owing to a large quality gap between simulation and manufactured devices. We aim at lifting the fundamental technical barriers to the practical use of learned diffractive optical systems. To this end, we propose a fabrication-aware design pipeline for diffractive optics fabricated by direct-write grayscale lithography followed by nano-imprinting replication, which is directly suited for inexpensive mass production of large area designs. We propose a super-resolved neural lithography model that can accurately predict the 3D geometry generated by the fabrication process. This model can be seamlessly integrated into existing differentiable optics frameworks, enabling fabrication-aware, end-to-end optimization of computational optical systems. To tackle the computational challenges, we also devise tensor-parallel compute framework centered on distributing large-scale FFT computation across many GPUs. As such, we demonstrate large scale diffractive optics designs up to 32.16 mm $\times$ 21.44 mm, simulated on grids of up to 128,640 by 85,760 feature points. We find adequate agreement between simulation and fabricated prototypes for applications such as holography and PSF engineering. We also achieve high image quality from an imaging system comprised only of a single DOE, with images processed only by a Wiener filter utilizing the simulation PSF. We believe our findings lift the fabrication limitations for real-world applications of diffractive optics and differentiable optical design.
>
---
#### [new 168] Risk-Sensitive Conformal Prediction for Catheter Placement Detection in Chest X-rays
- **分类: eess.IV; cs.CV; stat.AP**

- **简介: 该论文提出结合多任务学习与风险敏感符合预测，用于胸片中导管位置检测。旨在解决临床部署中的高可靠性需求，避免危险误判。工作包括同时进行分类、分割和关键点检测，利用任务协同提升性能，并通过风险敏感方法确保关键发现的高覆盖（99.29%）及零高风险错误，适用于医疗应用。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22496v1](http://arxiv.org/pdf/2505.22496v1)**

> **作者:** Long Hui
>
> **摘要:** This paper presents a novel approach to catheter and line position detection in chest X-rays, combining multi-task learning with risk-sensitive conformal prediction to address critical clinical requirements. Our model simultaneously performs classification, segmentation, and landmark detection, leveraging the synergistic relationship between these tasks to improve overall performance. We further enhance clinical reliability through risk-sensitive conformal prediction, which provides statistically guaranteed prediction sets with higher reliability for clinically critical findings. Experimental results demonstrate excellent performance with 90.68\% overall empirical coverage and 99.29\% coverage for critical conditions, while maintaining remarkable precision in prediction sets. Most importantly, our risk-sensitive approach achieves zero high-risk mispredictions (cases where the system dangerously declares problematic tubes as confidently normal), making the system particularly suitable for clinical deployment. This work offers both accurate predictions and reliably quantified uncertainty -- essential features for life-critical medical applications.
>
---
#### [new 169] LiDAR Based Semantic Perception for Forklifts in Outdoor Environments
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出基于双LiDAR的语义分割框架，解决叉车在复杂户外环境中的实时障碍物感知问题。通过整合前向与下视角LiDAR的3D点云，实现对行人、车辆等安全关键目标及道路、建筑等环境要素的高精度分割，满足自主叉车导航的实时性要求。**

- **链接: [http://arxiv.org/pdf/2505.22258v1](http://arxiv.org/pdf/2505.22258v1)**

> **作者:** Benjamin Serfling; Hannes Reichert; Lorenzo Bayerlein; Konrad Doll; Kati Radkhah-Lens
>
> **摘要:** In this study, we present a novel LiDAR-based semantic segmentation framework tailored for autonomous forklifts operating in complex outdoor environments. Central to our approach is the integration of a dual LiDAR system, which combines forward-facing and downward-angled LiDAR sensors to enable comprehensive scene understanding, specifically tailored for industrial material handling tasks. The dual configuration improves the detection and segmentation of dynamic and static obstacles with high spatial precision. Using high-resolution 3D point clouds captured from two sensors, our method employs a lightweight yet robust approach that segments the point clouds into safety-critical instance classes such as pedestrians, vehicles, and forklifts, as well as environmental classes such as driveable ground, lanes, and buildings. Experimental validation demonstrates that our approach achieves high segmentation accuracy while satisfying strict runtime requirements, establishing its viability for safety-aware, fully autonomous forklift navigation in dynamic warehouse and yard environments.
>
---
#### [new 170] Spatial Knowledge Graph-Guided Multimodal Synthesis
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 论文提出SKG2Data方法，属于多模态数据合成任务，解决多模态大模型（MLLMs）空间感知不足问题。通过构建空间知识图（SKG）模拟人类对方向和距离的感知，生成符合空间常识的合成数据，实验表明有效提升模型空间推理与泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.22633v1](http://arxiv.org/pdf/2505.22633v1)**

> **作者:** Yida Xue; Zhen Bi; Jinnan Yang; Jungang Lou; Huajun Chen; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have significantly enhanced their capabilities; however, their spatial perception abilities remain a notable limitation. To address this challenge, multimodal data synthesis offers a promising solution. Yet, ensuring that synthesized data adhere to spatial common sense is a non-trivial task. In this work, we introduce SKG2Data, a novel multimodal synthesis approach guided by spatial knowledge graphs, grounded in the concept of knowledge-to-data generation. SKG2Data automatically constructs a Spatial Knowledge Graph (SKG) to emulate human-like perception of spatial directions and distances, which is subsequently utilized to guide multimodal data synthesis. Extensive experiments demonstrate that data synthesized from diverse types of spatial knowledge, including direction and distance, not only enhance the spatial perception and reasoning abilities of MLLMs but also exhibit strong generalization capabilities. We hope that the idea of knowledge-based data synthesis can advance the development of spatial intelligence.
>
---
#### [new 171] Subspecialty-Specific Foundation Model for Intelligent Gastrointestinal Pathology
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出Digepath模型，针对胃肠病理诊断的主观性和可变性问题，通过预训练与精细筛选的双阶段优化策略，利用353 million图像块训练，实现33项任务的顶尖性能，并在早期胃癌筛查中达99.6%敏感度，推动精准病理学发展。**

- **链接: [http://arxiv.org/pdf/2505.21928v1](http://arxiv.org/pdf/2505.21928v1)**

> **作者:** Lianghui Zhu; Xitong Ling; Minxi Ouyang; Xiaoping Liu; Mingxi Fu; Tian Guan; Fanglei Fu; Xuanyu Wang; Maomao Zeng; Mingxi Zhu; Yibo Jin; Liming Liu; Song Duan; Qiming He; Yizhi Wang; Luxi Xie; Houqiang Li; Yonghong He; Sufang Tian
>
> **摘要:** Gastrointestinal (GI) diseases represent a clinically significant burden, necessitating precise diagnostic approaches to optimize patient outcomes. Conventional histopathological diagnosis, heavily reliant on the subjective interpretation of pathologists, suffers from limited reproducibility and diagnostic variability. To overcome these limitations and address the lack of pathology-specific foundation models for GI diseases, we develop Digepath, a specialized foundation model for GI pathology. Our framework introduces a dual-phase iterative optimization strategy combining pretraining with fine-screening, specifically designed to address the detection of sparsely distributed lesion areas in whole-slide images. Digepath is pretrained on more than 353 million image patches from over 200,000 hematoxylin and eosin-stained slides of GI diseases. It attains state-of-the-art performance on 33 out of 34 tasks related to GI pathology, including pathological diagnosis, molecular prediction, gene mutation prediction, and prognosis evaluation, particularly in diagnostically ambiguous cases and resolution-agnostic tissue classification.We further translate the intelligent screening module for early GI cancer and achieve near-perfect 99.6% sensitivity across 9 independent medical institutions nationwide. The outstanding performance of Digepath highlights its potential to bridge critical gaps in histopathological practice. This work not only advances AI-driven precision pathology for GI diseases but also establishes a transferable paradigm for other pathology subspecialties.
>
---
#### [new 172] Comparative Analysis of Machine Learning Models for Lung Cancer Mutation Detection and Staging Using 3D CT Scans
- **分类: eess.IV; cs.CV**

- **简介: 该论文比较两种机器学习模型（FMCIB+XGBoost与Dinov2+ABMIL）在3D CT肺癌突变检测与分期中的性能。针对非侵入性检测需求，通过斯坦福和Lung-CT-PT-Dx数据集验证，发现监督模型更优于突变检测（KRAS 84.6%、EGFR 88.3%），而自监督模型在T分期泛化性（79.7%）表现突出，为临床提供模型选择依据。**

- **链接: [http://arxiv.org/pdf/2505.22592v1](http://arxiv.org/pdf/2505.22592v1)**

> **作者:** Yiheng Li; Francisco Carrillo-Perez; Mohammed Alawad; Olivier Gevaert
>
> **摘要:** Lung cancer is the leading cause of cancer mortality worldwide, and non-invasive methods for detecting key mutations and staging are essential for improving patient outcomes. Here, we compare the performance of two machine learning models - FMCIB+XGBoost, a supervised model with domain-specific pretraining, and Dinov2+ABMIL, a self-supervised model with attention-based multiple-instance learning - on 3D lung nodule data from the Stanford Radiogenomics and Lung-CT-PT-Dx cohorts. In the task of KRAS and EGFR mutation detection, FMCIB+XGBoost consistently outperformed Dinov2+ABMIL, achieving accuracies of 0.846 and 0.883 for KRAS and EGFR mutations, respectively. In cancer staging, Dinov2+ABMIL demonstrated competitive generalization, achieving an accuracy of 0.797 for T-stage prediction in the Lung-CT-PT-Dx cohort, suggesting SSL's adaptability across diverse datasets. Our results emphasize the clinical utility of supervised models in mutation detection and highlight the potential of SSL to improve staging generalization, while identifying areas for enhancement in mutation sensitivity.
>
---
#### [new 173] Optimizing Deep Learning for Skin Cancer Classification: A Computationally Efficient CNN with Minimal Accuracy Trade-Off
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于皮肤癌图像分类任务，针对资源受限场景下深度模型计算效率低的问题，提出轻量级CNN。通过参数压缩（2390万→69.2万）将FLOPs从40亿降至3040万，在HAM10000数据集上保持96.7%参数减少且准确率仅降0.022%，优化移动端/边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2505.21597v1](http://arxiv.org/pdf/2505.21597v1)**

> **作者:** Abdullah Al Mamun; Pollob Chandra Ray; Md Rahat Ul Nasib; Akash Das; Jia Uddin; Md Nurul Absur
>
> **备注:** 6 pages, & 7 Images
>
> **摘要:** The rapid advancement of deep learning in medical image analysis has greatly enhanced the accuracy of skin cancer classification. However, current state-of-the-art models, especially those based on transfer learning like ResNet50, come with significant computational overhead, rendering them impractical for deployment in resource-constrained environments. This study proposes a custom CNN model that achieves a 96.7\% reduction in parameters (from 23.9 million in ResNet50 to 692,000) while maintaining a classification accuracy deviation of less than 0.022\%. Our empirical analysis of the HAM10000 dataset reveals that although transfer learning models provide a marginal accuracy improvement of approximately 0.022\%, they result in a staggering 13,216.76\% increase in FLOPs, considerably raising computational costs and inference latency. In contrast, our lightweight CNN architecture, which encompasses only 30.04 million FLOPs compared to ResNet50's 4.00 billion, significantly reduces energy consumption, memory footprint, and inference time. These findings underscore the trade-off between the complexity of deep models and their real-world feasibility, positioning our optimized CNN as a practical solution for mobile and edge-based skin cancer diagnostics.
>
---
#### [new 174] More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态推理任务，研究推理链延长导致模型过度依赖语言先验、忽视视觉信息的幻觉问题。提出RH-AUC指标量化推理长度对感知准确性的影响，并构建RH-Bench基准测试，揭示模型规模及训练数据类型比数据量更影响推理与视觉接地的平衡。**

- **链接: [http://arxiv.org/pdf/2505.21523v1](http://arxiv.org/pdf/2505.21523v1)**

> **作者:** Chengzhi Liu; Zhongxing Xu; Qingyue Wei; Juncheng Wu; James Zou; Xin Eric Wang; Yuyin Zhou; Sheng Liu
>
> **摘要:** Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more heavily on language priors. Attention analysis shows that longer reasoning chains lead to reduced focus on visual inputs, which contributes to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model's perception accuracy changes with reasoning length, allowing us to evaluate whether the model preserves visual grounding during reasoning. We also release RH-Bench, a diagnostic benchmark that spans a variety of multimodal tasks, designed to assess the trade-off between reasoning ability and hallucination. Our analysis reveals that (i) larger models typically achieve a better balance between reasoning and perception, and (ii) this balance is influenced more by the types and domains of training data than by its overall volume. These findings underscore the importance of evaluation frameworks that jointly consider both reasoning quality and perceptual fidelity.
>
---
#### [new 175] Mitigating Audiovisual Mismatch in Visual-Guide Audio Captioning
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视觉引导音频描述任务，旨在解决视听不匹配问题（如配音或画外音）。提出熵感知门控融合框架，通过注意力熵分析抑制误导视觉信息，并开发批量视听混排技术增强模型鲁棒性，在AudioCaps基准上超越基线，速度提升6倍。**

- **链接: [http://arxiv.org/pdf/2505.22045v1](http://arxiv.org/pdf/2505.22045v1)**

> **作者:** Le Xu; Chenxing Li; Yong Ren; Yujie Chen; Yu Gu; Ruibo Fu; Shan Yang; Dong Yu
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Current vision-guided audio captioning systems frequently fail to address audiovisual misalignment in real-world scenarios, such as dubbed content or off-screen sounds. To bridge this critical gap, we present an entropy-aware gated fusion framework that dynamically modulates visual information flow through cross-modal uncertainty quantification. Our novel approach employs attention entropy analysis in cross-attention layers to automatically identify and suppress misleading visual cues during modal fusion. Complementing this architecture, we develop a batch-wise audiovisual shuffling technique that generates synthetic mismatched training pairs, greatly enhancing model resilience against alignment noise. Evaluations on the AudioCaps benchmark demonstrate our system's superior performance over existing baselines, especially in mismatched modality scenarios. Furthermore, our solution demonstrates an approximately 6x improvement in inference speed compared to the baseline.
>
---
#### [new 176] High-Fidelity Functional Ultrasound Reconstruction via A Visual Auto-Regressive Framework
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出一种视觉自回归框架，旨在解决功能超声成像（fUS）因数据稀缺（伦理限制）和颅骨导致的信号衰减问题，通过高保真重建提升神经血管映射的图像质量，增强机器学习模型的公平性和数据多样性。**

- **链接: [http://arxiv.org/pdf/2505.21530v1](http://arxiv.org/pdf/2505.21530v1)**

> **作者:** Xuhang Chen; Zhuo Li; Yanyan Shen; Mufti Mahmud; Hieu Pham; Chi-Man Pun; Shuqiang Wang
>
> **摘要:** Functional ultrasound (fUS) imaging provides exceptional spatiotemporal resolution for neurovascular mapping, yet its practical application is significantly hampered by critical challenges. Foremost among these are data scarcity, arising from ethical considerations and signal degradation through the cranium, which collectively limit dataset diversity and compromise the fairness of downstream machine learning models.
>
---
#### [new 177] Chest Disease Detection In X-Ray Images Using Deep Learning Classification Method
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于胸部X光图像分类任务，旨在通过深度学习区分COVID-19、肺炎、肺结核与正常病例。采用迁移学习微调预训练CNN模型，评估分类性能（如准确率、F1值），并利用Grad-CAM提升模型可解释性，以增强临床应用可靠性。**

- **链接: [http://arxiv.org/pdf/2505.22609v1](http://arxiv.org/pdf/2505.22609v1)**

> **作者:** Alanna Hazlett; Naomi Ohashi; Timothy Rodriguez; Sodiq Adewole
>
> **摘要:** In this work, we investigate the performance across multiple classification models to classify chest X-ray images into four categories of COVID-19, pneumonia, tuberculosis (TB), and normal cases. We leveraged transfer learning techniques with state-of-the-art pre-trained Convolutional Neural Networks (CNNs) models. We fine-tuned these pre-trained architectures on a labeled medical x-ray images. The initial results are promising with high accuracy and strong performance in key classification metrics such as precision, recall, and F1 score. We applied Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability to provide visual explanations for classification decisions, improving trust and transparency in clinical applications.
>
---
#### [new 178] Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-行动(VLA)模型任务，旨在解决现有VLA模型在微调过程中丢失预训练视觉语言模型(VLM)核心能力的问题。提出ChatVLA-2混合专家模型及三阶段训练 pipeline，保留VLM的开放世界推理（如数学、空间认知）并转化为机器人动作。通过数学题解答和新型空间指令任务验证，其推理能力超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21906v1](http://arxiv.org/pdf/2505.21906v1)**

> **作者:** Zhongyi Zhou; Yichen Zhu; Junjie Wen; Chaomin Shen; Yi Xu
>
> **备注:** Project page: https://chatvla-2.github.io/
>
> **摘要:** Vision-language-action (VLA) models have emerged as the next generation of models in robotics. However, despite leveraging powerful pre-trained Vision-Language Models (VLMs), existing end-to-end VLA systems often lose key capabilities during fine-tuning as the model adapts to specific robotic tasks. We argue that a generalizable VLA model should retain and expand upon the VLM's core competencies: 1) Open-world embodied reasoning - the VLA should inherit the knowledge from VLM, i.e., recognize anything that the VLM can recognize, capable of solving math problems, possessing visual-spatial intelligence, 2) Reasoning following - effectively translating the open-world reasoning into actionable steps for the robot. In this work, we introduce ChatVLA-2, a novel mixture-of-expert VLA model coupled with a specialized three-stage training pipeline designed to preserve the VLM's original strengths while enabling actionable reasoning. To validate our approach, we design a math-matching task wherein a robot interprets math problems written on a whiteboard and picks corresponding number cards from a table to solve equations. Remarkably, our method exhibits exceptional mathematical reasoning and OCR capabilities, despite these abilities not being explicitly trained within the VLA. Furthermore, we demonstrate that the VLA possesses strong spatial reasoning skills, enabling it to interpret novel directional instructions involving previously unseen objects. Overall, our method showcases reasoning and comprehension abilities that significantly surpass state-of-the-art imitation learning methods such as OpenVLA, DexVLA, and pi-zero. This work represents a substantial advancement toward developing truly generalizable robotic foundation models endowed with robust reasoning capacities.
>
---
#### [new 179] Learning Compositional Behaviors from Demonstration and Language
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出BLADE框架，属于长期机器人操作任务，解决复杂场景下机器人泛化执行新任务的挑战。通过整合模仿学习与模型规划，利用语言标注的示范数据，自动提取结构化高阶动作表示（含视觉感知前提/效果及神经网络策略），无需人工标注，实现在模拟与真实机器人上的复杂操作验证。**

- **链接: [http://arxiv.org/pdf/2505.21981v1](http://arxiv.org/pdf/2505.21981v1)**

> **作者:** Weiyu Liu; Neil Nie; Ruohan Zhang; Jiayuan Mao; Jiajun Wu
>
> **备注:** Presented at CoRL 2024 and as an Oral Presentation at the 2024 CoRL LEAP Workshop. The first two authors contributed equally. The last two authors jointly advised the project. For videos and additional results, visit: https://blade-bot.github.io/
>
> **摘要:** We introduce Behavior from Language and Demonstration (BLADE), a framework for long-horizon robotic manipulation by integrating imitation learning and model-based planning. BLADE leverages language-annotated demonstrations, extracts abstract action knowledge from large language models (LLMs), and constructs a library of structured, high-level action representations. These representations include preconditions and effects grounded in visual perception for each high-level action, along with corresponding controllers implemented as neural network-based policies. BLADE can recover such structured representations automatically, without manually labeled states or symbolic definitions. BLADE shows significant capabilities in generalizing to novel situations, including novel initial states, external state perturbations, and novel goals. We validate the effectiveness of our approach both in simulation and on real robots with a diverse set of objects with articulated parts, partial observability, and geometric constraints.
>
---
#### [new 180] tenSVD algorithm for compression
- **分类: stat.CO; cs.CV; cs.LG**

- **简介: 该论文提出基于张量Tucker分解的tenSVD算法，旨在高效压缩高维数据以减少存储、传输带宽和能耗。通过将数据组织为高阶张量并建立Tucker模型，对比基准算法评估压缩效率与信息保真度，使用定量指标分析真实/模拟数据结果，并关注算法能耗可持续性。**

- **链接: [http://arxiv.org/pdf/2505.21686v1](http://arxiv.org/pdf/2505.21686v1)**

> **作者:** Michele Gallo
>
> **摘要:** Tensors provide a robust framework for managing high-dimensional data. Consequently, tensor analysis has emerged as an active research area in various domains, including machine learning, signal processing, computer vision, graph analysis, and data mining. This study introduces an efficient image storage approach utilizing tensors, aiming to minimize memory to store, bandwidth to transmit and energy to processing. The proposed method organizes original data into a higher-order tensor and applies the Tucker model for compression. Implemented in R, this method is compared to a baseline algorithm. The evaluation focuses on efficient of algorithm measured in term of computational time and the quality of information preserved, using both simulated and real datasets. A detailed analysis of the results is conducted, employing established quantitative metrics, with significant attention paid to sustainability in terms of energy consumption across algorithms.
>
---
#### [new 181] Privacy-Preserving Chest X-ray Report Generation via Multimodal Federated Learning with ViT and GPT-2
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决传统中心化模型导致的隐私泄露问题。通过多模态联邦学习框架（ViT编码X光片，GPT-2生成报告），在不共享原始数据下训练模型，对比三种聚合策略，验证了Krum方法在保持隐私前提下生成高质量报告的有效性。**

- **链接: [http://arxiv.org/pdf/2505.21715v1](http://arxiv.org/pdf/2505.21715v1)**

> **作者:** Md. Zahid Hossain; Mustofa Ahmed; Most. Sharmin Sultana Samu; Md. Rakibul Islam
>
> **备注:** Preprint, manuscript under-review
>
> **摘要:** The automated generation of radiology reports from chest X-ray images holds significant promise in enhancing diagnostic workflows while preserving patient privacy. Traditional centralized approaches often require sensitive data transfer, posing privacy concerns. To address this, the study proposes a Multimodal Federated Learning framework for chest X-ray report generation using the IU-Xray dataset. The system utilizes a Vision Transformer (ViT) as the encoder and GPT-2 as the report generator, enabling decentralized training without sharing raw data. Three Federated Learning (FL) aggregation strategies: FedAvg, Krum Aggregation and a novel Loss-aware Federated Averaging (L-FedAvg) were evaluated. Among these, Krum Aggregation demonstrated superior performance across lexical and semantic evaluation metrics such as ROUGE, BLEU, BERTScore and RaTEScore. The results show that FL can match or surpass centralized models in generating clinically relevant and semantically rich radiology reports. This lightweight and privacy-preserving framework paves the way for collaborative medical AI development without compromising data confidentiality.
>
---
#### [new 182] STDR: Spatio-Temporal Decoupling for Real-Time Dynamic Scene Rendering
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于动态场景重建任务，旨在解决3DGS方法初始化时因时空不一致导致的动态建模不准问题。提出STDR模块，通过时空掩码、分离变形场及一致性正则化解耦空间-时间信息，提升重建质量和时空一致性。**

- **链接: [http://arxiv.org/pdf/2505.22400v1](http://arxiv.org/pdf/2505.22400v1)**

> **作者:** Zehao Li; Hao Jiang; Yujun Cai; Jianing Chen; Baolong Bi; Shuqin Gao; Honglong Zhao; Yiwei Wang; Tianlu Mao; Zhaoqi Wang
>
> **摘要:** Although dynamic scene reconstruction has long been a fundamental challenge in 3D vision, the recent emergence of 3D Gaussian Splatting (3DGS) offers a promising direction by enabling high-quality, real-time rendering through explicit Gaussian primitives. However, existing 3DGS-based methods for dynamic reconstruction often suffer from \textit{spatio-temporal incoherence} during initialization, where canonical Gaussians are constructed by aggregating observations from multiple frames without temporal distinction. This results in spatio-temporally entangled representations, making it difficult to model dynamic motion accurately. To overcome this limitation, we propose \textbf{STDR} (Spatio-Temporal Decoupling for Real-time rendering), a plug-and-play module that learns spatio-temporal probability distributions for each Gaussian. STDR introduces a spatio-temporal mask, a separated deformation field, and a consistency regularization to jointly disentangle spatial and temporal patterns. Extensive experiments demonstrate that incorporating our module into existing 3DGS-based dynamic scene reconstruction frameworks leads to notable improvements in both reconstruction quality and spatio-temporal consistency across synthetic and real-world benchmarks.
>
---
#### [new 183] Taming Transformer Without Using Learning Rate Warmup
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出无需学习率预热训练Transformer的新方法。针对训练中因Wq/Wk矩阵的"光谱能量集中"引发的模型崩溃问题，基于Weyl不等式设计优化策略，通过限制权重更新的奇异值比值防止方向能量聚集，成功在ViT/Swin/GPT中实现稳定训练。**

- **链接: [http://arxiv.org/pdf/2505.21910v1](http://arxiv.org/pdf/2505.21910v1)**

> **作者:** Xianbiao Qi; Yelin He; Jiaquan Ye; Chun-Guang Li; Bojia Zi; Xili Dai; Qin Zou; Rong Xiao
>
> **备注:** This paper is published as a conference paper at ICLR 2025
>
> **摘要:** Scaling Transformer to a large scale without using some technical tricks such as learning rate warump and using an obviously lower learning rate is an extremely challenging task, and is increasingly gaining more attention. In this paper, we provide a theoretical analysis for the process of training Transformer and reveal the rationale behind the model crash phenomenon in the training process, termed \textit{spectral energy concentration} of ${\bW_q}^{\top} \bW_k$, which is the reason for a malignant entropy collapse, where ${\bW_q}$ and $\bW_k$ are the projection matrices for the query and the key in Transformer, respectively. To remedy this problem, motivated by \textit{Weyl's Inequality}, we present a novel optimization strategy, \ie, making the weight updating in successive steps smooth -- if the ratio $\frac{\sigma_{1}(\nabla \bW_t)}{\sigma_{1}(\bW_{t-1})}$ is larger than a threshold, we will automatically bound the learning rate to a weighted multiple of $\frac{\sigma_{1}(\bW_{t-1})}{\sigma_{1}(\nabla \bW_t)}$, where $\nabla \bW_t$ is the updating quantity in step $t$. Such an optimization strategy can prevent spectral energy concentration to only a few directions, and thus can avoid malignant entropy collapse which will trigger the model crash. We conduct extensive experiments using ViT, Swin-Transformer and GPT, showing that our optimization strategy can effectively and stably train these Transformers without using learning rate warmup.
>
---
#### [new 184] Chain-of-Talkers (CoTalk): Fast Human Annotation of Dense Image Captions
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出CoTalk方法，优化密集图像标注任务，通过顺序标注减少冗余、多模态界面提升效率，在固定预算下提高标注数量与全面性。实验显示其速度（0.42 vs 0.30单位/秒）和检索性能（41.13% vs 40.52%）优于并行方法。**

- **链接: [http://arxiv.org/pdf/2505.22627v1](http://arxiv.org/pdf/2505.22627v1)**

> **作者:** Yijun Shen; Delong Chen; Fan Liu; Xingyu Wang; Chuanyi Zhang; Liang Yao; Yuhui Zheng
>
> **摘要:** While densely annotated image captions significantly facilitate the learning of robust vision-language alignment, methodologies for systematically optimizing human annotation efforts remain underexplored. We introduce Chain-of-Talkers (CoTalk), an AI-in-the-loop methodology designed to maximize the number of annotated samples and improve their comprehensiveness under fixed budget constraints (e.g., total human annotation time). The framework is built upon two key insights. First, sequential annotation reduces redundant workload compared to conventional parallel annotation, as subsequent annotators only need to annotate the ``residual'' -- the missing visual information that previous annotations have not covered. Second, humans process textual input faster by reading while outputting annotations with much higher throughput via talking; thus a multimodal interface enables optimized efficiency. We evaluate our framework from two aspects: intrinsic evaluations that assess the comprehensiveness of semantic units, obtained by parsing detailed captions into object-attribute trees and analyzing their effective connections; extrinsic evaluation measures the practical usage of the annotated captions in facilitating vision-language alignment. Experiments with eight participants show our Chain-of-Talkers (CoTalk) improves annotation speed (0.42 vs. 0.30 units/sec) and retrieval performance (41.13\% vs. 40.52\%) over the parallel method.
>
---
#### [new 185] Physics-inspired Generative AI models via real hardware-based noisy quantum diffusion
- **分类: quant-ph; cond-mat.dis-nn; cs.AI; cs.CV; cs.LG; 81P68, 81P40, 81P47, 68Q12, 68T07,; I.2.6; I.3.3; J.2**

- **简介: 该论文属于量子生成AI任务，旨在解决现有量子扩散模型（QDMs）因硬件限制难以扩展的问题。提出两种物理启发方法：1）量子随机行走结合经典动力学，提升MNIST图像生成质量（降低FID）；2）利用IBM四量子位硬件的内在噪声生成图像。通过利用量子噪声而非纠错，为大规模量子生成AI奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.22193v1](http://arxiv.org/pdf/2505.22193v1)**

> **作者:** Marco Parigi; Stefano Martina; Francesco Aldo Venturelli; Filippo Caruso
>
> **备注:** 17 pages, 9 figures. Supplementary materials: 2 pages, 2 figures
>
> **摘要:** Quantum Diffusion Models (QDMs) are an emerging paradigm in Generative AI that aims to use quantum properties to improve the performances of their classical counterparts. However, existing algorithms are not easily scalable due to the limitations of near-term quantum devices. Following our previous work on QDMs, here we propose and implement two physics-inspired protocols. In the first, we use the formalism of quantum stochastic walks, showing that a specific interplay of quantum and classical dynamics in the forward process produces statistically more robust models generating sets of MNIST images with lower Fr\'echet Inception Distance (FID) than using totally classical dynamics. In the second approach, we realize an algorithm to generate images by exploiting the intrinsic noise of real IBM quantum hardware with only four qubits. Our work could be a starting point to pave the way for new scenarios for large-scale algorithms in quantum Generative AI, where quantum noise is neither mitigated nor corrected, but instead exploited as a useful resource.
>
---
#### [new 186] VideoMarkBench: Benchmarking Robustness of Video Watermarking
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视频水印鲁棒性评估任务，旨在解决现有方法对抗常见/恶意攻击的脆弱性问题。研究构建了首个系统基准VideoMarkBench，通过测试3种生成模型、4种水印方法、7种检测策略在12类扰动（含白/黑/无盒攻击）下的表现，揭示当前技术的不足。**

- **链接: [http://arxiv.org/pdf/2505.21620v1](http://arxiv.org/pdf/2505.21620v1)**

> **作者:** Zhengyuan Jiang; Moyang Guo; Kecen Li; Yuepeng Hu; Yupu Wang; Zhicong Huang; Cheng Hong; Neil Zhenqiang Gong
>
> **摘要:** The rapid development of video generative models has led to a surge in highly realistic synthetic videos, raising ethical concerns related to disinformation and copyright infringement. Recently, video watermarking has been proposed as a mitigation strategy by embedding invisible marks into AI-generated videos to enable subsequent detection. However, the robustness of existing video watermarking methods against both common and adversarial perturbations remains underexplored. In this work, we introduce VideoMarkBench, the first systematic benchmark designed to evaluate the robustness of video watermarks under watermark removal and watermark forgery attacks. Our study encompasses a unified dataset generated by three state-of-the-art video generative models, across three video styles, incorporating four watermarking methods and seven aggregation strategies used during detection. We comprehensively evaluate 12 types of perturbations under white-box, black-box, and no-box threat models. Our findings reveal significant vulnerabilities in current watermarking approaches and highlight the urgent need for more robust solutions. Our code is available at https://github.com/zhengyuan-jiang/VideoMarkBench.
>
---
#### [new 187] Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文聚焦多模态推理任务，旨在解决强化学习（RL）冷启动阶段推理性能不足的问题。提出两阶段方法：先通过监督微调（SFT）建立结构化推理模式，再结合GRPO强化学习优化，显著提升模型性能。实验显示其开源模型在MathVista等基准测试中达SOTA，7B模型性能提升超7%，3B模型表现媲美7B模型。**

- **链接: [http://arxiv.org/pdf/2505.22334v1](http://arxiv.org/pdf/2505.22334v1)**

> **作者:** Lai Wei; Yuting Li; Kaipeng Zheng; Chen Wang; Yue Wang; Linghe Kong; Lichao Sun; Weiran Huang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at https://github.com/waltonfuture/RL-with-Cold-Start.
>
---
#### [new 188] MAMBO-NET: Multi-Causal Aware Modeling Backdoor-Intervention Optimization for Medical Image Segmentation Network
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医疗图像分割任务，旨在解决复杂解剖变异和成像限制等混淆因素导致的分割偏差问题。提出MAMBO-NET方法，通过多高斯分布建模混淆因素，结合因果干预与后门优化技术，减少其对分割的影响，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.21874v1](http://arxiv.org/pdf/2505.21874v1)**

> **作者:** Ruiguo Yu; Yiyang Zhang; Yuan Tian; Yujie Diao; Di Jin; Witold Pedrycz
>
> **摘要:** Medical image segmentation methods generally assume that the process from medical image to segmentation is unbiased, and use neural networks to establish conditional probability models to complete the segmentation task. This assumption does not consider confusion factors, which can affect medical images, such as complex anatomical variations and imaging modality limitations. Confusion factors obfuscate the relevance and causality of medical image segmentation, leading to unsatisfactory segmentation results. To address this issue, we propose a multi-causal aware modeling backdoor-intervention optimization (MAMBO-NET) network for medical image segmentation. Drawing insights from causal inference, MAMBO-NET utilizes self-modeling with multi-Gaussian distributions to fit the confusion factors and introduce causal intervention into the segmentation process. Moreover, we design appropriate posterior probability constraints to effectively train the distributions of confusion factors. For the distributions to effectively guide the segmentation and mitigate and eliminate the Impact of confusion factors on the segmentation, we introduce classical backdoor intervention techniques and analyze their feasibility in the segmentation task. To evaluate the effectiveness of our approach, we conducted extensive experiments on five medical image datasets. The results demonstrate that our method significantly reduces the influence of confusion factors, leading to enhanced segmentation accuracy.
>
---
#### [new 189] Neural Face Skinning for Mesh-agnostic Facial Expression Cloning
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于面部动画重目标任务，解决全局方法细节不足与局部方法控制复杂的问题。提出结合全局潜码与局部变形的神经皮肤模型，通过预测顶点权重定位潜码影响区域，并用FACS混合形状监督，实现精细可控的表情克隆，适应多样网格结构。**

- **链接: [http://arxiv.org/pdf/2505.22416v1](http://arxiv.org/pdf/2505.22416v1)**

> **作者:** Sihun Cha; Serin Yoon; Kwanggyoon Seo; Junyong Noh
>
> **摘要:** Accurately retargeting facial expressions to a face mesh while enabling manipulation is a key challenge in facial animation retargeting. Recent deep-learning methods address this by encoding facial expressions into a global latent code, but they often fail to capture fine-grained details in local regions. While some methods improve local accuracy by transferring deformations locally, this often complicates overall control of the facial expression. To address this, we propose a method that combines the strengths of both global and local deformation models. Our approach enables intuitive control and detailed expression cloning across diverse face meshes, regardless of their underlying structures. The core idea is to localize the influence of the global latent code on the target mesh. Our model learns to predict skinning weights for each vertex of the target face mesh through indirect supervision from predefined segmentation labels. These predicted weights localize the global latent code, enabling precise and region-specific deformations even for meshes with unseen shapes. We supervise the latent code using Facial Action Coding System (FACS)-based blendshapes to ensure interpretability and allow straightforward editing of the generated animation. Through extensive experiments, we demonstrate improved performance over state-of-the-art methods in terms of expression fidelity, deformation transfer accuracy, and adaptability across diverse mesh structures.
>
---
#### [new 190] Image denoising as a conditional expectation
- **分类: eess.IV; cs.CV; math.OC**

- **简介: 该论文属于图像去噪任务，针对传统投影方法的偏差与收敛性问题，提出基于条件期望的框架，通过核积分算子在RKHS中构建最小二乘解，并证明收敛性以优化参数选择。**

- **链接: [http://arxiv.org/pdf/2505.21546v1](http://arxiv.org/pdf/2505.21546v1)**

> **作者:** Sajal Chakroborty; Suddhasattwa Das
>
> **摘要:** All techniques for denoising involve a notion of a true (noise-free) image, and a hypothesis space. The hypothesis space may reconstruct the image directly as a grayscale valued function, or indirectly by its Fourier or wavelet spectrum. Most common techniques estimate the true image as a projection to some subspace. We propose an interpretation of a noisy image as a collection of samples drawn from a certain probability space. Within this interpretation, projection based approaches are not guaranteed to be unbiased and convergent. We present a data-driven denoising method in which the true image is recovered as a conditional expectation. Although the probability space is unknown apriori, integrals on this space can be estimated by kernel integral operators. The true image is reformulated as the least squares solution to a linear equation in a reproducing kernel Hilbert space (RKHS), and involving various kernel integral operators as linear transforms. Assuming the true image to be a continuous function on a compact planar domain, the technique is shown to be convergent as the number of pixels goes to infinity. We also show that for a picture with finite number of pixels, the convergence result can be used to choose the various parameters for an optimum denoising result.
>
---
#### [new 191] Synonymous Variational Inference for Perceptual Image Compression
- **分类: cs.IT; cs.AI; cs.CV; cs.LG; eess.IV; math.IT**

- **简介: 该论文属于感知图像压缩任务，旨在优化率-失真-感知三重权衡。提出同义变分推断（SVI）方法，利用感知相似性构建语义集并近似潜在表征，证明其理论基础，并开发对应渐进式压缩编解码器，实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.22438v1](http://arxiv.org/pdf/2505.22438v1)**

> **作者:** Zijian Liang; Kai Niu; Changshuo Wang; Jin Xu; Ping Zhang
>
> **备注:** 31 pages, 20 figures. This paper is accepted by Proceedings of the 42nd International Conference on Machine Learning (ICML 2025) Poster
>
> **摘要:** Recent contributions of semantic information theory reveal the set-element relationship between semantic and syntactic information, represented as synonymous relationships. In this paper, we propose a synonymous variational inference (SVI) method based on this synonymity viewpoint to re-analyze the perceptual image compression problem. It takes perceptual similarity as a typical synonymous criterion to build an ideal synonymous set (Synset), and approximate the posterior of its latent synonymous representation with a parametric density by minimizing a partial semantic KL divergence. This analysis theoretically proves that the optimization direction of perception image compression follows a triple tradeoff that can cover the existing rate-distortion-perception schemes. Additionally, we introduce synonymous image compression (SIC), a new image compression scheme that corresponds to the analytical process of SVI, and implement a progressive SIC codec to fully leverage the model's capabilities. Experimental results demonstrate comparable rate-distortion-perception performance using a single progressive SIC codec, thus verifying the effectiveness of our proposed analysis method.
>
---
#### [new 192] UP-SLAM: Adaptively Structured Gaussian SLAM with Uncertainty Prediction in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: UP-SLAM是动态环境下的RGB-D SLAM系统。针对传统方法实时性差、动态敏感问题，提出并行框架解耦跟踪与建图，采用概率八叉树自适应管理高斯原语，设计无训练不确定性估计器处理动态对象，结合时间编码器与DINO特征提升渲染和定位精度。实验显示其定位准确度提升59.8%，渲染PSNR提高4.57dB，实时生成无瑕疵静态地图。**

- **链接: [http://arxiv.org/pdf/2505.22335v1](http://arxiv.org/pdf/2505.22335v1)**

> **作者:** Wancai Zheng; Linlin Ou; Jiajie He; Libo Zhou; Xinyi Yu; Yan Wei
>
> **摘要:** Recent 3D Gaussian Splatting (3DGS) techniques for Visual Simultaneous Localization and Mapping (SLAM) have significantly progressed in tracking and high-fidelity mapping. However, their sequential optimization framework and sensitivity to dynamic objects limit real-time performance and robustness in real-world scenarios. We present UP-SLAM, a real-time RGB-D SLAM system for dynamic environments that decouples tracking and mapping through a parallelized framework. A probabilistic octree is employed to manage Gaussian primitives adaptively, enabling efficient initialization and pruning without hand-crafted thresholds. To robustly filter dynamic regions during tracking, we propose a training-free uncertainty estimator that fuses multi-modal residuals to estimate per-pixel motion uncertainty, achieving open-set dynamic object handling without reliance on semantic labels. Furthermore, a temporal encoder is designed to enhance rendering quality. Concurrently, low-dimensional features are efficiently transformed via a shallow multilayer perceptron to construct DINO features, which are then employed to enrich the Gaussian field and improve the robustness of uncertainty prediction. Extensive experiments on multiple challenging datasets suggest that UP-SLAM outperforms state-of-the-art methods in both localization accuracy (by 59.8%) and rendering quality (by 4.57 dB PSNR), while maintaining real-time performance and producing reusable, artifact-free static maps in dynamic environments.The project: https://aczheng-cai.github.io/up_slam.github.io/
>
---
#### [new 193] Laparoscopic Image Desmoking Using the U-Net with New Loss Function and Integrated Differentiable Wiener Filter
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出ULW方法，通过改进U-Net结合新损失函数（融合像素、结构相似性和感知损失）及可学习Wiener滤波，解决腹腔镜手术中烟雾导致的图像模糊问题。实验表明其有效提升图像清晰度，为实时手术影像增强提供新方案。**

- **链接: [http://arxiv.org/pdf/2505.21634v1](http://arxiv.org/pdf/2505.21634v1)**

> **作者:** Chengyu Yang; Chengjun Liu
>
> **摘要:** Laparoscopic surgeries often suffer from reduced visual clarity due to the presence of surgical smoke originated by surgical instruments, which poses significant challenges for both surgeons and vision based computer-assisted technologies. In order to remove the surgical smoke, a novel U-Net deep learning with new loss function and integrated differentiable Wiener filter (ULW) method is presented. Specifically, the new loss function integrates the pixel, structural, and perceptual properties. Thus, the new loss function, which combines the structural similarity index measure loss, the perceptual loss, as well as the mean squared error loss, is able to enhance the quality and realism of the reconstructed images. Furthermore, the learnable Wiener filter is capable of effectively modelling the degradation process caused by the surgical smoke. The effectiveness of the proposed ULW method is evaluated using the publicly available paired laparoscopic smoke and smoke-free image dataset, which provides reliable benchmarking and quantitative comparisons. Experimental results show that the proposed ULW method excels in both visual clarity and metric-based evaluation. As a result, the proposed ULW method offers a promising solution for real-time enhancement of laparoscopic imagery. The code is available at https://github.com/chengyuyang-njit/ImageDesmoke.
>
---
#### [new 194] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Surf2CT框架，首次通过人体表面扫描和人口统计数据生成 torso CT图像。任务是无创内部解剖成像，解决仅凭外部信息合成高精度CT的难题。方法分三阶段：表面补全、粗CT生成及超分辨率重建，均采用3D流匹配模型，训练于3,198例CT数据，验证显示器官体积误差小、解剖保真度高，为居家医疗和个性化诊断提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.22511v1](http://arxiv.org/pdf/2505.22511v1)**

> **作者:** Siyeop Yoon; Yujin Oh; Pengfei Jin; Sifan Song; Matthew Tivnan; Dufan Wu; Xiang Li; Quanzheng Li
>
> **备注:** Neurips 2025 submitted
>
> **摘要:** We present Surf2CT, a novel cascaded flow matching framework that synthesizes full 3D computed tomography (CT) volumes of the human torso from external surface scans and simple demographic data (age, sex, height, weight). This is the first approach capable of generating realistic volumetric internal anatomy images solely based on external body shape and demographics, without any internal imaging. Surf2CT proceeds through three sequential stages: (1) Surface Completion, reconstructing a complete signed distance function (SDF) from partial torso scans using conditional 3D flow matching; (2) Coarse CT Synthesis, generating a low-resolution CT volume from the completed SDF and demographic information; and (3) CT Super-Resolution, refining the coarse volume into a high-resolution CT via a patch-wise conditional flow model. Each stage utilizes a 3D-adapted EDM2 backbone trained via flow matching. We trained our model on a combined dataset of 3,198 torso CT scans (approximately 1.13 million axial slices) sourced from Massachusetts General Hospital (MGH) and the AutoPET challenge. Evaluation on 700 paired torso surface-CT cases demonstrated strong anatomical fidelity: organ volumes exhibited small mean percentage differences (range from -11.1% to 4.4%), and muscle/fat body composition metrics matched ground truth with strong correlation (range from 0.67 to 0.96). Lung localization had minimal bias (mean difference -2.5 mm), and surface completion significantly improved metrics (Chamfer distance: from 521.8 mm to 2.7 mm; Intersection-over-Union: from 0.87 to 0.98). Surf2CT establishes a new paradigm for non-invasive internal anatomical imaging using only external data, opening opportunities for home-based healthcare, preventive medicine, and personalized clinical assessments without the risks associated with conventional imaging techniques.
>
---
#### [new 195] VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出VRAG-RL框架，针对视觉丰富信息理解任务中传统方法处理视觉信息不足、检索查询效果差的问题，通过强化学习优化视觉语言模型与搜索引擎的交互，设计视觉操作动作（如裁剪/缩放）和奖励机制，提升多模态推理能力。**

- **链接: [http://arxiv.org/pdf/2505.22019v1](http://arxiv.org/pdf/2505.22019v1)**

> **作者:** Qiuchen Wang; Ruixue Ding; Yu Zeng; Zehui Chen; Lin Chen; Shihang Wang; Pengjun Xie; Fei Huang; Feng Zhao
>
> **摘要:** Effectively retrieving, reasoning and understanding visually rich information remains a challenge for RAG methods. Traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As RL has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing continual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incorporate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search engines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse-to-fine perspective. Furthermore, to bridge the gap between users' original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewriting and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. The code is available at \hyperlink{https://github.com/Alibaba-NLP/VRAG}{https://github.com/Alibaba-NLP/VRAG}.
>
---
#### [new 196] A Closer Look at Multimodal Representation Collapse
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究多模态融合中的模态崩溃问题，即模型仅依赖部分模态忽略其他模态。通过分析发现噪声特征与预测特征在神经元共享时引发崩溃，提出基于知识蒸馏的显式基底重分配算法，解除特征纠缠并提升鲁棒性，实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.22483v1](http://arxiv.org/pdf/2505.22483v1)**

> **作者:** Abhra Chaudhuri; Anjan Dutta; Tu Bui; Serban Georgescu
>
> **备注:** International Conference on Machine Learning (ICML) 2025 (Spotlight)
>
> **摘要:** We aim to develop a fundamental understanding of modality collapse, a recently observed empirical phenomenon wherein models trained for multimodal fusion tend to rely only on a subset of the modalities, ignoring the rest. We show that modality collapse happens when noisy features from one modality are entangled, via a shared set of neurons in the fusion head, with predictive features from another, effectively masking out positive contributions from the predictive features of the former modality and leading to its collapse. We further prove that cross-modal knowledge distillation implicitly disentangles such representations by freeing up rank bottlenecks in the student encoder, denoising the fusion-head outputs without negatively impacting the predictive features from either modality. Based on the above findings, we propose an algorithm that prevents modality collapse through explicit basis reallocation, with applications in dealing with missing modalities. Extensive experiments on multiple multimodal benchmarks validate our theoretical claims. Project page: https://abhrac.github.io/mmcollapse/.
>
---
#### [new 197] STA-Risk: A Deep Dive of Spatio-Temporal Asymmetries for Breast Cancer Risk Prediction
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出STA-Risk模型，通过捕捉乳腺X光片的双侧空间与时间不对称性，解决现有模型忽视影像时空演变细节导致的乳腺癌风险预测精度不足问题。基于Transformer架构，采用侧向与时间编码及定制不对称损失函数，在两个数据集上实现1-5年风险预测的SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.21699v1](http://arxiv.org/pdf/2505.21699v1)**

> **作者:** Zhengbo Zhou; Dooman Arefan; Margarita Zuley; Jules Sumkin; Shandong Wu
>
> **摘要:** Predicting the risk of developing breast cancer is an important clinical tool to guide early intervention and tailoring personalized screening strategies. Early risk models have limited performance and recently machine learning-based analysis of mammogram images showed encouraging risk prediction effects. These models however are limited to the use of a single exam or tend to overlook nuanced breast tissue evolvement in spatial and temporal details of longitudinal imaging exams that are indicative of breast cancer risk. In this paper, we propose STA-Risk (Spatial and Temporal Asymmetry-based Risk Prediction), a novel Transformer-based model that captures fine-grained mammographic imaging evolution simultaneously from bilateral and longitudinal asymmetries for breast cancer risk prediction. STA-Risk is innovative by the side encoding and temporal encoding to learn spatial-temporal asymmetries, regulated by a customized asymmetry loss. We performed extensive experiments with two independent mammogram datasets and achieved superior performance than four representative SOTA models for 1- to 5-year future risk prediction. Source codes will be released upon publishing of the paper.
>
---
#### [new 198] Temporal Restoration and Spatial Rewiring for Source-Free Multivariate Time Series Domain Adaptation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出TERSE方法，针对无源域多变量时间序列领域自适应任务，解决现有方法忽视空间相关性导致的跨域迁移效果差问题。通过设计时空特征编码器及时间恢复、空间重连任务，重建时空依赖关系，实现跨域特征对齐，实验验证其有效性与扩展性。**

- **链接: [http://arxiv.org/pdf/2505.21525v1](http://arxiv.org/pdf/2505.21525v1)**

> **作者:** Peiliang Gong; Yucheng Wang; Min Wu; Zhenghua Chen; Xiaoli Li; Daoqiang Zhang
>
> **摘要:** Source-Free Domain Adaptation (SFDA) aims to adapt a pre-trained model from an annotated source domain to an unlabelled target domain without accessing the source data, thereby preserving data privacy. While existing SFDA methods have proven effective in reducing reliance on source data, they struggle to perform well on multivariate time series (MTS) due to their failure to consider the intrinsic spatial correlations inherent in MTS data. These spatial correlations are crucial for accurately representing MTS data and preserving invariant information across domains. To address this challenge, we propose Temporal Restoration and Spatial Rewiring (TERSE), a novel and concise SFDA method tailored for MTS data. Specifically, TERSE comprises a customized spatial-temporal feature encoder designed to capture the underlying spatial-temporal characteristics, coupled with both temporal restoration and spatial rewiring tasks to reinstate latent representations of the temporally masked time series and the spatially masked correlated structures. During the target adaptation phase, the target encoder is guided to produce spatially and temporally consistent features with the source domain by leveraging the source pre-trained temporal restoration and spatial rewiring networks. Therefore, TERSE can effectively model and transfer spatial-temporal dependencies across domains, facilitating implicit feature alignment. In addition, as the first approach to simultaneously consider spatial-temporal consistency in MTS-SFDA, TERSE can also be integrated as a versatile plug-and-play module into established SFDA methods. Extensive experiments on three real-world time series datasets demonstrate the effectiveness and versatility of our approach.
>
---
#### [new 199] RESOUND: Speech Reconstruction from Silent Videos via Acoustic-Semantic Decomposed Modeling
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于唇动到语音合成（L2S）任务，旨在解决无声视频语音重建的准确性与自然度不足问题。提出RESOUND系统，通过声学-语义分解建模分离韵律与语言特征，结合语音单元与梅尔频谱生成波形，提升可懂度与表达自然度，实验验证有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22024v1](http://arxiv.org/pdf/2505.22024v1)**

> **作者:** Long-Khanh Pham; Thanh V. T. Tran; Minh-Tan Pham; Van Nguyen
>
> **备注:** accepted in Interspeech 2025
>
> **摘要:** Lip-to-speech (L2S) synthesis, which reconstructs speech from visual cues, faces challenges in accuracy and naturalness due to limited supervision in capturing linguistic content, accents, and prosody. In this paper, we propose RESOUND, a novel L2S system that generates intelligible and expressive speech from silent talking face videos. Leveraging source-filter theory, our method involves two components: an acoustic path to predict prosody and a semantic path to extract linguistic features. This separation simplifies learning, allowing independent optimization of each representation. Additionally, we enhance performance by integrating speech units, a proven unsupervised speech representation technique, into waveform generation alongside mel-spectrograms. This allows RESOUND to synthesize prosodic speech while preserving content and speaker identity. Experiments conducted on two standard L2S benchmarks confirm the effectiveness of the proposed method across various metrics.
>
---
#### [new 200] Taylor expansion-based Kolmogorov-Arnold network for blind image quality assessment
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出TaylorKAN模型用于盲图像质量评估，解决高维特征处理中的性能与效率问题。通过泰勒展开作为激活函数增强局部逼近能力，并结合网络深度压缩和特征降维，提升计算效率。实验表明其优于传统KAN模型，在多数据库测试中表现更优。**

- **链接: [http://arxiv.org/pdf/2505.21592v1](http://arxiv.org/pdf/2505.21592v1)**

> **作者:** Ze Chen; Shaode Yu
>
> **备注:** under review
>
> **摘要:** Kolmogorov-Arnold Network (KAN) has attracted growing interest for its strong function approximation capability. In our previous work, KAN and its variants were explored in score regression for blind image quality assessment (BIQA). However, these models encounter challenges when processing high-dimensional features, leading to limited performance gains and increased computational cost. To address these issues, we propose TaylorKAN that leverages the Taylor expansions as learnable activation functions to enhance local approximation capability. To improve the computational efficiency, network depth reduction and feature dimensionality compression are integrated into the TaylorKAN-based score regression pipeline. On five databases (BID, CLIVE, KonIQ, SPAQ, and FLIVE) with authentic distortions, extensive experiments demonstrate that TaylorKAN consistently outperforms the other KAN-related models, indicating that the local approximation via Taylor expansions is more effective than global approximation using orthogonal functions. Its generalization capacity is validated through inter-database experiments. The findings highlight the potential of TaylorKAN as an efficient and robust model for high-dimensional score regression.
>
---
#### [new 201] RenderFormer: Transformer-based Neural Rendering of Triangle Meshes with Global Illumination
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文提出RenderFormer，一种基于Transformer的神经渲染方法，通过三角网格直接生成含全局光照的图像，无需场景级训练。旨在解决传统渲染计算复杂或依赖场景数据的问题，其两阶段架构（光照传输建模与视光转换）实现高效视图依赖渲染。**

- **链接: [http://arxiv.org/pdf/2505.21925v1](http://arxiv.org/pdf/2505.21925v1)**

> **作者:** Chong Zeng; Yue Dong; Pieter Peers; Hongzhi Wu; Xin Tong
>
> **备注:** Accepted to SIGGRAPH 2025. Project page: https://microsoft.github.io/renderformer
>
> **摘要:** We present RenderFormer, a neural rendering pipeline that directly renders an image from a triangle-based representation of a scene with full global illumination effects and that does not require per-scene training or fine-tuning. Instead of taking a physics-centric approach to rendering, we formulate rendering as a sequence-to-sequence transformation where a sequence of tokens representing triangles with reflectance properties is converted to a sequence of output tokens representing small patches of pixels. RenderFormer follows a two stage pipeline: a view-independent stage that models triangle-to-triangle light transport, and a view-dependent stage that transforms a token representing a bundle of rays to the corresponding pixel values guided by the triangle-sequence from the view-independent stage. Both stages are based on the transformer architecture and are learned with minimal prior constraints. We demonstrate and evaluate RenderFormer on scenes with varying complexity in shape and light transport.
>
---
#### [new 202] Higher-Order Group Synchronization
- **分类: stat.ML; cs.CV; cs.LG; math.CO; math.OC**

- **简介: 该论文提出高阶群同步任务，旨在超图上通过同步高阶边测量获取节点全局估计，解决传统成对同步方法在复杂关系建模中的局限。工作包括建立同步性条件、提出基于消息传递的计算框架，并验证其在抗噪和处理旋转同步等任务中的优势。**

- **链接: [http://arxiv.org/pdf/2505.21932v1](http://arxiv.org/pdf/2505.21932v1)**

> **作者:** Adriana L. Duncan; Joe Kileel
>
> **备注:** 40 pages
>
> **摘要:** Group synchronization is the problem of determining reliable global estimates from noisy local measurements on networks. The typical task for group synchronization is to assign elements of a group to the nodes of a graph in a way that respects group elements given on the edges which encode information about local pairwise relationships between the nodes. In this paper, we introduce a novel higher-order group synchronization problem which operates on a hypergraph and seeks to synchronize higher-order local measurements on the hyperedges to obtain global estimates on the nodes. Higher-order group synchronization is motivated by applications to computer vision and image processing, among other computational problems. First, we define the problem of higher-order group synchronization and discuss its mathematical foundations. Specifically, we give necessary and sufficient synchronizability conditions which establish the importance of cycle consistency in higher-order group synchronization. Then, we propose the first computational framework for general higher-order group synchronization; it acts globally and directly on higher-order measurements using a message passing algorithm. We discuss theoretical guarantees for our framework, including convergence analyses under outliers and noise. Finally, we show potential advantages of our method through numerical experiments. In particular, we show that in certain cases our higher-order method applied to rotational and angular synchronization outperforms standard pairwise synchronization methods and is more robust to outliers. We also show that our method has comparable performance on simulated cryo-electron microscopy (cryo-EM) data compared to a standard cryo-EM reconstruction package.
>
---
#### [new 203] From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究机器学习模型的unlearning任务，解决现有方法易受relearning攻击的问题。发现unlearned数据的知识可通过微调保留集恢复，提出基于权重空间正则化的抗攻击方法，通过L2距离等属性优化模型，提升抵抗relearning能力。**

- **链接: [http://arxiv.org/pdf/2505.22310v1](http://arxiv.org/pdf/2505.22310v1)**

> **作者:** Shoaib Ahmed Siddiqui; Adrian Weller; David Krueger; Gintare Karolina Dziugaite; Michael Curtis Mozer; Eleni Triantafillou
>
> **摘要:** Recent unlearning methods for LLMs are vulnerable to relearning attacks: knowledge believed-to-be-unlearned re-emerges by fine-tuning on a small set of (even seemingly-unrelated) examples. We study this phenomenon in a controlled setting for example-level unlearning in vision classifiers. We make the surprising discovery that forget-set accuracy can recover from around 50% post-unlearning to nearly 100% with fine-tuning on just the retain set -- i.e., zero examples of the forget set. We observe this effect across a wide variety of unlearning methods, whereas for a model retrained from scratch excluding the forget set (gold standard), the accuracy remains at 50%. We observe that resistance to relearning attacks can be predicted by weight-space properties, specifically, $L_2$-distance and linear mode connectivity between the original and the unlearned model. Leveraging this insight, we propose a new class of methods that achieve state-of-the-art resistance to relearning attacks.
>
---
#### [new 204] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人接触密集操作任务，旨在解决现有VLA模型在力控精细操作（如插拔）中因视觉遮挡或动态不确定性导致的性能不足问题。提出ForceVLA框架，通过FVLMoE模块融合视觉、语言和实时六轴力反馈，并构建多模态数据集ForceVLA-Data，提升操作成功率至80%，超越基线23.2%。**

- **链接: [http://arxiv.org/pdf/2505.22159v1](http://arxiv.org/pdf/2505.22159v1)**

> **作者:** Jiawen Yu; Hairuo Liu; Qiaojun Yu; Jieji Ren; Ce Hao; Haitong Ding; Guangyu Huang; Guofan Huang; Yan Song; Panpan Cai; Cewu Lu; Wenqiang Zhang
>
> **摘要:** Vision-Language-Action (VLA) models have advanced general-purpose robotic manipulation by leveraging pretrained visual and linguistic representations. However, they struggle with contact-rich tasks that require fine-grained control involving force, especially under visual occlusion or dynamic uncertainty. To address these limitations, we propose \textbf{ForceVLA}, a novel end-to-end manipulation framework that treats external force sensing as a first-class modality within VLA systems. ForceVLA introduces \textbf{FVLMoE}, a force-aware Mixture-of-Experts fusion module that dynamically integrates pretrained visual-language embeddings with real-time 6-axis force feedback during action decoding. This enables context-aware routing across modality-specific experts, enhancing the robot's ability to adapt to subtle contact dynamics. We also introduce \textbf{ForceVLA-Data}, a new dataset comprising synchronized vision, proprioception, and force-torque signals across five contact-rich manipulation tasks. ForceVLA improves average task success by 23.2\% over strong $\pi_0$-based baselines, achieving up to 80\% success in tasks such as plug insertion. Our approach highlights the importance of multimodal integration for dexterous manipulation and sets a new benchmark for physically intelligent robotic control. Code and data will be released at https://sites.google.com/view/forcevla2025.
>
---
#### [new 205] Understanding Adversarial Training with Energy-based Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于对抗训练与生成建模任务，旨在解决对抗训练中的过拟合（CO/RO）及鲁棒分类器生成能力不足问题。通过能量视角分析对抗样本能量差异，提出Delta Energy Regularizer（DER）缓解过拟合，并基于局部PCA和能量指导提升生成多样性与质量，实现竞争力生成效果。**

- **链接: [http://arxiv.org/pdf/2505.22486v1](http://arxiv.org/pdf/2505.22486v1)**

> **作者:** Mujtaba Hussain Mirza; Maria Rosaria Briglia; Filippo Bartolucci; Senad Beadini; Giuseppe Lisanti; Iacopo Masi
>
> **备注:** Under review for TPAMI
>
> **摘要:** We aim at using Energy-based Model (EBM) framework to better understand adversarial training (AT) in classifiers, and additionally to analyze the intrinsic generative capabilities of robust classifiers. By viewing standard classifiers through an energy lens, we begin by analyzing how the energies of adversarial examples, generated by various attacks, differ from those of the natural samples. The central focus of our work is to understand the critical phenomena of Catastrophic Overfitting (CO) and Robust Overfitting (RO) in AT from an energy perspective. We analyze the impact of existing AT approaches on the energy of samples during training and observe that the behavior of the ``delta energy' -- change in energy between original sample and its adversarial counterpart -- diverges significantly when CO or RO occurs. After a thorough analysis of these energy dynamics and their relationship with overfitting, we propose a novel regularizer, the Delta Energy Regularizer (DER), designed to smoothen the energy landscape during training. We demonstrate that DER is effective in mitigating both CO and RO across multiple benchmarks. We further show that robust classifiers, when being used as generative models, have limits in handling trade-off between image quality and variability. We propose an improved technique based on a local class-wise principal component analysis (PCA) and energy-based guidance for better class-specific initialization and adaptive stopping, enhancing sample diversity and generation quality. Considering that we do not explicitly train for generative modeling, we achieve a competitive Inception Score (IS) and Fr\'echet inception distance (FID) compared to hybrid discriminative-generative models.
>
---
#### [new 206] Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MM-UPT框架，通过GRPO算法实现多模态LLM的无监督后训练，解决依赖标注数据及方法复杂的问题。利用多数投票的自我奖励机制替代传统奖励，并结合自动生成的合成问题，提升模型推理能力，在多个数据集上显著优于无监督基线，接近监督方法效果。**

- **链接: [http://arxiv.org/pdf/2505.22453v1](http://arxiv.org/pdf/2505.22453v1)**

> **作者:** Lai Wei; Yuting Li; Chen Wang; Yue Wang; Linghe Kong; Weiran Huang; Lichao Sun
>
> **摘要:** Improving Multi-modal Large Language Models (MLLMs) in the post-training stage typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL). However, these supervised methods require expensive and manually annotated multi-modal data--an ultimately unsustainable resource. While recent efforts have explored unsupervised post-training, their methods are complex and difficult to iterate. In this work, we are the first to investigate the use of GRPO, a stable and scalable online RL algorithm, for enabling continual self-improvement without any external supervision. We propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs. MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that MM-UPT significantly improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3 %$\rightarrow$72.9 % on MathVista, 62.9 %$\rightarrow$68.7 % on We-Math), using standard dataset without ground truth labels. MM-UPT also outperforms prior unsupervised baselines and even approaches the results of supervised GRPO. Furthermore, we show that incorporating synthetic questions, generated solely by MLLM itself, can boost performance as well, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for continual, autonomous enhancement of MLLMs in the absence of external supervision. Our code is available at https://github.com/waltonfuture/MM-UPT.
>
---
## 更新

#### [replaced 001] UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control
- **分类: cs.CV; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.05749v4](http://arxiv.org/pdf/2502.05749v4)**

> **作者:** Kaizhen Zhu; Mokai Pan; Yuexin Ma; Yanwei Fu; Jingyi Yu; Jingya Wang; Ye Shi
>
> **摘要:** Recent advances in diffusion bridge models leverage Doob's $h$-transform to establish fixed endpoints between distributions, demonstrating promising results in image translation and restoration tasks. However, these approaches frequently produce blurred or excessively smoothed image details and lack a comprehensive theoretical foundation to explain these shortcomings. To address these limitations, we propose UniDB, a unified framework for diffusion bridges based on Stochastic Optimal Control (SOC). UniDB formulates the problem through an SOC-based optimization and derives a closed-form solution for the optimal controller, thereby unifying and generalizing existing diffusion bridge models. We demonstrate that existing diffusion bridges employing Doob's $h$-transform constitute a special case of our framework, emerging when the terminal penalty coefficient in the SOC cost function tends to infinity. By incorporating a tunable terminal penalty coefficient, UniDB achieves an optimal balance between control costs and terminal penalties, substantially improving detail preservation and output quality. Notably, UniDB seamlessly integrates with existing diffusion bridge models, requiring only minimal code modifications. Extensive experiments across diverse image restoration tasks validate the superiority and adaptability of the proposed framework. Our code is available at https://github.com/UniDB-SOC/UniDB/.
>
---
#### [replaced 002] Progressive Scaling Visual Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19990v2](http://arxiv.org/pdf/2505.19990v2)**

> **作者:** Jack Hong; Shilin Yan; Zehao Xiao; Jiayin Cai; Xiaolong Jiang; Yao Hu; Henghui Ding
>
> **摘要:** In this work, we propose a progressive scaling training strategy for visual object tracking, systematically analyzing the influence of training data volume, model size, and input resolution on tracking performance. Our empirical study reveals that while scaling each factor leads to significant improvements in tracking accuracy, naive training suffers from suboptimal optimization and limited iterative refinement. To address this issue, we introduce DT-Training, a progressive scaling framework that integrates small teacher transfer and dual-branch alignment to maximize model potential. The resulting scaled tracker consistently outperforms state-of-the-art methods across multiple benchmarks, demonstrating strong generalization and transferability of the proposed method. Furthermore, we validate the broader applicability of our approach to additional tasks, underscoring its versatility beyond tracking.
>
---
#### [replaced 003] Preference Adaptive and Sequential Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.10419v2](http://arxiv.org/pdf/2412.10419v2)**

> **作者:** Ofir Nabati; Guy Tennenholtz; ChihWei Hsu; Moonkyung Ryu; Deepak Ramachandran; Yinlam Chow; Xiang Li; Craig Boutilier
>
> **备注:** Accepted to ICML 2025 Link to PASTA dataset: https://www.kaggle.com/datasets/googleai/pasta-data
>
> **摘要:** We address the problem of interactive text-to-image (T2I) generation, designing a reinforcement learning (RL) agent which iteratively improves a set of generated images for a user through a sequence of prompt expansions. Using human raters, we create a novel dataset of sequential preferences, which we leverage, together with large-scale open-source (non-sequential) datasets. We construct user-preference and user-choice models using an EM strategy and identify varying user preference types. We then leverage a large multimodal language model (LMM) and a value-based RL approach to suggest an adaptive and diverse slate of prompt expansions to the user. Our Preference Adaptive and Sequential Text-to-image Agent (PASTA) extends T2I models with adaptive multi-turn capabilities, fostering collaborative co-creation and addressing uncertainty or underspecification in a user's intent. We evaluate PASTA using human raters, showing significant improvement compared to baseline methods. We also open-source our sequential rater dataset and simulated user-rater interactions to support future research in user-centric multi-turn T2I systems.
>
---
#### [replaced 004] DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20460v2](http://arxiv.org/pdf/2505.20460v2)**

> **作者:** Ruiqi Wu; Xinjie Wang; Liu Liu; Chunle Guo; Jiaxiong Qiu; Chongyi Li; Lichao Huang; Zhizhong Su; Ming-Ming Cheng
>
> **摘要:** We present DIPO, a novel framework for the controllable generation of articulated 3D objects from a pair of images: one depicting the object in a resting state and the other in an articulated state. Compared to the single-image approach, our dual-image input imposes only a modest overhead for data collection, but at the same time provides important motion information, which is a reliable guide for predicting kinematic relationships between parts. Specifically, we propose a dual-image diffusion model that captures relationships between the image pair to generate part layouts and joint parameters. In addition, we introduce a Chain-of-Thought (CoT) based graph reasoner that explicitly infers part connectivity relationships. To further improve robustness and generalization on complex articulated objects, we develop a fully automated dataset expansion pipeline, name LEGO-Art, that enriches the diversity and complexity of PartNet-Mobility dataset. We propose PM-X, a large-scale dataset of complex articulated 3D objects, accompanied by rendered images, URDF annotations, and textual descriptions. Extensive experiments demonstrate that DIPO significantly outperforms existing baselines in both the resting state and the articulated state, while the proposed PM-X dataset further enhances generalization to diverse and structurally complex articulated objects. Our code and dataset will be released to the community upon publication.
>
---
#### [replaced 005] AKRMap: Adaptive Kernel Regression for Trustworthy Visualization of Cross-Modal Embeddings
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14664v2](http://arxiv.org/pdf/2505.14664v2)**

> **作者:** Yilin Ye; Junchao Huang; Xingchen Zeng; Jiazhi Xia; Wei Zeng
>
> **摘要:** Cross-modal embeddings form the foundation for multi-modal models. However, visualization methods for interpreting cross-modal embeddings have been primarily confined to traditional dimensionality reduction (DR) techniques like PCA and t-SNE. These DR methods primarily focus on feature distributions within a single modality, whilst failing to incorporate metrics (e.g., CLIPScore) across multiple modalities. This paper introduces AKRMap, a new DR technique designed to visualize cross-modal embeddings metric with enhanced accuracy by learning kernel regression of the metric landscape in the projection space. Specifically, AKRMap constructs a supervised projection network guided by a post-projection kernel regression loss, and employs adaptive generalized kernels that can be jointly optimized with the projection. This approach enables AKRMap to efficiently generate visualizations that capture complex metric distributions, while also supporting interactive features such as zoom and overlay for deeper exploration. Quantitative experiments demonstrate that AKRMap outperforms existing DR methods in generating more accurate and trustworthy visualizations. We further showcase the effectiveness of AKRMap in visualizing and comparing cross-modal embeddings for text-to-image models. Code and demo are available at https://github.com/yilinye/AKRMap.
>
---
#### [replaced 006] X-GAN: A Generative AI-Powered Unsupervised Model for Main Vessel Segmentation of Glaucoma Screening
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06743v3](http://arxiv.org/pdf/2503.06743v3)**

> **作者:** Cheng Huang; Weizheng Xie; Tsengdar J. Lee; Jui-Kai Wang; Karanjit Kooner; Ning Zhang; Jia Zhang
>
> **摘要:** Structural changes in main retinal blood vessels serve as critical biomarkers for the onset and progression of glaucoma. Identifying these vessels is vital for vascular modeling yet highly challenging. This paper proposes X-GAN, a generative AI-powered unsupervised segmentation model designed for extracting main blood vessels from Optical Coherence Tomography Angiography (OCTA) images. The process begins with the Space Colonization Algorithm (SCA) to rapidly generate a skeleton of vessels, featuring their radii. By synergistically integrating the generative adversarial network (GAN) with biostatistical modeling of vessel radii, X-GAN enables a fast reconstruction of both 2D and 3D representations of the vessels. Based on this reconstruction, X-GAN achieves nearly 100% segmentation accuracy without relying on labeled data or high-performance computing resources. Experimental results confirm X-GAN's superiority in evaluating main vessel segmentation compared to existing deep learning models. Code is here: https://github.com/VikiXie/SatMar8.
>
---
#### [replaced 007] From Head to Tail: Towards Balanced Representation in Large Vision-Language Models through Adaptive Data Calibration
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12821v3](http://arxiv.org/pdf/2503.12821v3)**

> **作者:** Mingyang Song; Xiaoye Qu; Jiawei Zhou; Yu Cheng
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved significant progress in combining visual comprehension with language generation. Despite this success, the training data of LVLMs still suffers from Long-Tail (LT) problems, where the data distribution is highly imbalanced. Previous works have mainly focused on traditional VLM architectures, i.e., CLIP or ViT, and specific tasks such as recognition and classification. Nevertheless, the exploration of LVLM (e.g. LLaVA) and more general tasks (e.g. Visual Question Answering and Visual Reasoning) remains under-explored. In this paper, we first conduct an in-depth analysis of the LT issues in LVLMs and identify two core causes: the overrepresentation of head concepts and the underrepresentation of tail concepts. Based on the above observation, we propose an $\textbf{A}$daptive $\textbf{D}$ata $\textbf{R}$efinement Framework ($\textbf{ADR}$), which consists of two stages: $\textbf{D}$ata $\textbf{R}$ebalancing ($\textbf{DR}$) and $\textbf{D}$ata $\textbf{S}$ynthesis ($\textbf{DS}$). In the DR stage, we adaptively rebalance the redundant data based on entity distributions, while in the DS stage, we leverage Denoising Diffusion Probabilistic Models (DDPMs) and scarce images to supplement underrepresented portions. Through comprehensive evaluations across eleven benchmarks, our proposed ADR effectively mitigates the long-tail problem in the training data, improving the average performance of LLaVA 1.5 relatively by 4.36%, without increasing the training data volume.
>
---
#### [replaced 008] PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20429v2](http://arxiv.org/pdf/2505.20429v2)**

> **作者:** Shuhao Guan; Moule Lin; Cheng Xu; Xinyi Liu; Jinman Zhao; Jiexin Fan; Qi Xu; Derek Greene
>
> **备注:** ACL 2025 main
>
> **摘要:** This paper introduces PreP-OCR, a two-stage pipeline that combines document image restoration with semantic-aware post-OCR correction to enhance both visual clarity and textual consistency, thereby improving text extraction from degraded historical documents. First, we synthesize document-image pairs from plaintext, rendering them with diverse fonts and layouts and then applying a randomly ordered set of degradation operations. An image restoration model is trained on this synthetic data, using multi-directional patch extraction and fusion to process large images. Second, a ByT5 post-OCR model, fine-tuned on synthetic historical text pairs, addresses remaining OCR errors. Detailed experiments on 13,831 pages of real historical documents in English, French, and Spanish show that the PreP-OCR pipeline reduces character error rates by 63.9-70.3% compared to OCR on raw images. Our pipeline demonstrates the potential of integrating image restoration with linguistic error correction for digitizing historical archives.
>
---
#### [replaced 009] An Effective Training Framework for Light-Weight Automatic Speech Recognition Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16991v2](http://arxiv.org/pdf/2505.16991v2)**

> **作者:** Abdul Hannan; Alessio Brutti; Shah Nawaz; Mubashir Noman
>
> **备注:** Accepted at InterSpeech 2025
>
> **摘要:** Recent advancement in deep learning encouraged developing large automatic speech recognition (ASR) models that achieve promising results while ignoring computational and memory constraints. However, deploying such models on low resource devices is impractical despite of their favorable performance. Existing approaches (pruning, distillation, layer skip etc.) transform the large models into smaller ones at the cost of significant performance degradation or require prolonged training of smaller models for better performance. To address these issues, we introduce an efficacious two-step representation learning based approach capable of producing several small sized models from a single large model ensuring considerably better performance in limited number of epochs. Comprehensive experimentation on ASR benchmarks reveals the efficacy of our approach, achieving three-fold training speed-up and up to 12.54% word error rate improvement.
>
---
#### [replaced 010] Cross-Layer Feature Pyramid Transformer for Small Object Detection in Aerial Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19696v2](http://arxiv.org/pdf/2407.19696v2)**

> **作者:** Zewen Du; Zhenjiang Hu; Guiyu Zhao; Ying Jin; Hongbin Ma
>
> **摘要:** Object detection in aerial images has always been a challenging task due to the generally small size of the objects. Most current detectors prioritize the development of new detection frameworks, often overlooking research on fundamental components such as feature pyramid networks. In this paper, we introduce the Cross-Layer Feature Pyramid Transformer (CFPT), a novel upsampler-free feature pyramid network designed specifically for small object detection in aerial images. CFPT incorporates two meticulously designed attention blocks with linear computational complexity: Cross-Layer Channel-Wise Attention (CCA) and Cross-Layer Spatial-Wise Attention (CSA). CCA achieves cross-layer interaction by dividing channel-wise token groups to perceive cross-layer global information along the spatial dimension, while CSA enables cross-layer interaction by dividing spatial-wise token groups to perceive cross-layer global information along the channel dimension. By integrating these modules, CFPT enables efficient cross-layer interaction in a single step, thereby avoiding the semantic gap and information loss associated with element-wise summation and layer-by-layer transmission. In addition, CFPT incorporates global contextual information, which improves detection performance for small objects. To further enhance location awareness during cross-layer interaction, we propose the Cross-Layer Consistent Relative Positional Encoding (CCPE) based on inter-layer mutual receptive fields. We evaluate the effectiveness of CFPT on three challenging object detection datasets in aerial images: VisDrone2019-DET, TinyPerson, and xView. Extensive experiments demonstrate that CFPT outperforms state-of-the-art feature pyramid networks while incurring lower computational costs. The code is available at https://github.com/duzw9311/CFPT.
>
---
#### [replaced 011] Structurally Different Neural Network Blocks for the Segmentation of Atrial and Aortic Perivascular Adipose Tissue in Multi-centre CT Angiography Scans
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.03494v2](http://arxiv.org/pdf/2306.03494v2)**

> **作者:** Ikboljon Sobirov; Cheng Xie; Muhammad Siddique; Parijat Patel; Kenneth Chan; Thomas Halborg; Christos P. Kotanidis; Zarqaish Fatima; Henry West; Sheena Thomas; Maria Lyasheva; Donna Alexander; David Adlam; Praveen Rao; Das Indrajeet; Aparna Deshpande; Amrita Bajaj; Jonathan C L Rodrigues; Benjamin J Hudson; Vivek Srivastava; George Krasopoulos; Rana Sayeed; Qiang Zhang; Pete Tomlins; Cheerag Shirodaria; Keith M. Channon; Stefan Neubauer; Charalambos Antoniades; Mohammad Yaqub
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** Since the emergence of convolutional neural networks (CNNs) and, later, vision transformers (ViTs), deep learning architectures have predominantly relied on identical block types with varying hyperparameters. We propose a novel block alternation strategy to leverage the complementary strengths of different architectural designs, assembling structurally distinct components similar to Lego blocks. We introduce LegoNet, a deep learning framework that alternates CNN-based and SwinViT-based blocks to enhance feature learning for medical image segmentation. We investigate three variations of LegoNet and apply this concept to a previously unexplored clinical problem: the segmentation of the internal mammary artery (IMA), aorta, and perivascular adipose tissue (PVAT) from computed tomography angiography (CTA) scans. These PVAT regions have been shown to possess prognostic value in assessing cardiovascular risk and primary clinical outcomes. We evaluate LegoNet on large datasets, achieving superior performance to other leading architectures. Furthermore, we assess the model's generalizability on external testing cohorts, where an expert clinician corrects the model's segmentations, achieving DSC > 0.90 across various external, international, and public cohorts. To further validate the model's clinical reliability, we perform intra- and inter-observer variability analysis, demonstrating strong agreement with human annotations. The proposed methodology has significant implications for diagnostic cardiovascular management and early prognosis, offering a robust, automated solution for vascular and perivascular segmentation and risk assessment in clinical practice, paving the way for personalised medicine.
>
---
#### [replaced 012] MOSformer: Momentum encoder-based inter-slice fusion transformer for medical image segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.11856v2](http://arxiv.org/pdf/2401.11856v2)**

> **作者:** De-Xing Huang; Xiao-Hu Zhou; Mei-Jiang Gui; Xiao-Liang Xie; Shi-Qi Liu; Shuang-Yi Wang; Zhen-Qiu Feng; Zeng-Guang Hou
>
> **备注:** Under Review
>
> **摘要:** Medical image segmentation takes an important position in various clinical applications. 2.5D-based segmentation models bridge the computational efficiency of 2D-based models with the spatial perception capabilities of 3D-based models. However, existing 2.5D-based models primarily adopt a single encoder to extract features of target and neighborhood slices, failing to effectively fuse inter-slice information, resulting in suboptimal segmentation performance. In this study, a novel momentum encoder-based inter-slice fusion transformer (MOSformer) is proposed to overcome this issue by leveraging inter-slice information at multi-scale feature maps extracted by different encoders. Specifically, dual encoders are employed to enhance feature distinguishability among different slices. One of the encoders is moving-averaged to maintain consistent slice representations. Moreover, an inter-slice fusion transformer (IF-Trans) module is developed to fuse inter-slice multi-scale features. The MOSformer is evaluated on three benchmark datasets (Synapse, ACDC, and AMOS), achieving a new state-of-the-art with 85.63%, 92.19%, and 85.43% DSC, respectively. These results demonstrate MOSformer's competitiveness in medical image segmentation.
>
---
#### [replaced 013] End-to-End Breast Cancer Radiotherapy Planning via LMMs with Consistency Embedding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.15876v4](http://arxiv.org/pdf/2311.15876v4)**

> **作者:** Kwanyoung Kim; Yujin Oh; Sangjoon Park; Hwa Kyung Byun; Joongyo Lee; Jin Sung Kim; Yong Bae Kim; Jong Chul Ye
>
> **备注:** Accepted for Medical Image Analysis 2025
>
> **摘要:** Recent advances in AI foundation models have significant potential for lightening the clinical workload by mimicking the comprehensive and multi-faceted approaches used by medical professionals. In the field of radiation oncology, the integration of multiple modalities holds great importance, so the opportunity of foundational model is abundant. Inspired by this, here we present RO-LMM, a multi-purpose, comprehensive large multimodal model (LMM) tailored for the field of radiation oncology. This model effectively manages a series of tasks within the clinical workflow, including clinical context summarization, radiation treatment plan suggestion, and plan-guided target volume segmentation by leveraging the capabilities of LMM. In particular, to perform consecutive clinical tasks without error accumulation, we present a novel Consistency Embedding Fine-Tuning (CEFTune) technique, which boosts LMM's robustness to noisy inputs while preserving the consistency of handling clean inputs. We further extend this concept to LMM-driven segmentation framework, leading to a novel Consistency Embedding Segmentation (CESEG) techniques. Experimental results including multi-centre validation confirm that our RO-LMM with CEFTune and CESEG results in promising performance for multiple clinical tasks with generalization capabilities.
>
---
#### [replaced 014] CLIP-MoE: Towards Building Mixture of Experts for CLIP with Diversified Multiplet Upcycling
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.19291v3](http://arxiv.org/pdf/2409.19291v3)**

> **作者:** Jihai Zhang; Xiaoye Qu; Tong Zhu; Yu Cheng
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) has become a cornerstone in multimodal intelligence. However, recent studies discovered that CLIP can only encode one aspect of the feature space, leading to substantial information loss and indistinctive features. To mitigate this issue, this paper introduces a novel strategy that fine-tunes a series of complementary CLIP models and transforms them into a CLIP-MoE. Specifically, we propose a model-agnostic Diversified Multiplet Upcycling (DMU) framework for CLIP. Instead of training multiple CLIP models from scratch, DMU leverages a pre-trained CLIP and fine-tunes it into a diverse set with highly cost-effective multistage contrastive learning, thus capturing distinct feature subspaces efficiently. To fully exploit these fine-tuned models while minimizing computational overhead, we transform them into a CLIP-MoE, which dynamically activates a subset of CLIP experts, achieving an effective balance between model capacity and computational cost. Comprehensive experiments demonstrate the superior performance of CLIP-MoE across various zero-shot retrieval, zero-shot image classification tasks, and downstream Multimodal Large Language Model (MLLM) benchmarks when used as a vision encoder.
>
---
#### [replaced 015] Diffusion Adversarial Post-Training for One-Step Video Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.08316v2](http://arxiv.org/pdf/2501.08316v2)**

> **作者:** Shanchuan Lin; Xin Xia; Yuxi Ren; Ceyuan Yang; Xuefeng Xiao; Lu Jiang
>
> **摘要:** The diffusion models are widely used for image and video generation, but their iterative generation process is slow and expansive. While existing distillation approaches have demonstrated the potential for one-step generation in the image domain, they still suffer from significant quality degradation. In this work, we propose Adversarial Post-Training (APT) against real data following diffusion pre-training for one-step video generation. To improve the training stability and quality, we introduce several improvements to the model architecture and training procedures, along with an approximated R1 regularization objective. Empirically, our experiments show that our adversarial post-trained model, Seaweed-APT, can generate 2-second, 1280x720, 24fps videos in real time using a single forward evaluation step. Additionally, our model is capable of generating 1024px images in a single step, achieving quality comparable to state-of-the-art methods.
>
---
#### [replaced 016] PAEFF: Precise Alignment and Enhanced Gated Feature Fusion for Face-Voice Association
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17002v2](http://arxiv.org/pdf/2505.17002v2)**

> **作者:** Abdul Hannan; Muhammad Arslan Manzoor; Shah Nawaz; Muhammad Irzam Liaqat; Markus Schedl; Mubashir Noman
>
> **备注:** Accepted at InterSpeech 2025
>
> **摘要:** We study the task of learning association between faces and voices, which is gaining interest in the multimodal community lately. These methods suffer from the deliberate crafting of negative mining procedures as well as the reliance on the distant margin parameter. These issues are addressed by learning a joint embedding space in which orthogonality constraints are applied to the fused embeddings of faces and voices. However, embedding spaces of faces and voices possess different characteristics and require spaces to be aligned before fusing them. To this end, we propose a method that accurately aligns the embedding spaces and fuses them with an enhanced gated fusion thereby improving the performance of face-voice association. Extensive experiments on the VoxCeleb dataset reveals the merits of the proposed approach.
>
---
#### [replaced 017] Stereo Radargrammetry Using Deep Learning from Airborne SAR Images
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.20876v2](http://arxiv.org/pdf/2505.20876v2)**

> **作者:** Tatsuya Sasayama; Shintaro Ito; Koichi Ito; Takafumi Aoki
>
> **备注:** 5 pages, 5 figures, conference IGARSS2025
>
> **摘要:** In this paper, we propose a stereo radargrammetry method using deep learning from airborne Synthetic Aperture Radar (SAR) images. Deep learning-based methods are considered to suffer less from geometric image modulation, while there is no public SAR image dataset used to train such methods. We create a SAR image dataset and perform fine-tuning of a deep learning-based image correspondence method. The proposed method suppresses the degradation of image quality by pixel interpolation without ground projection of the SAR image and divides the SAR image into patches for processing, which makes it possible to apply deep learning. Through a set of experiments, we demonstrate that the proposed method exhibits a wider range and more accurate elevation measurements compared to conventional methods.
>
---
#### [replaced 018] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17982v3](http://arxiv.org/pdf/2505.17982v3)**

> **作者:** Bryan Wong; Jong Woo Kim; Huazhu Fu; Mun Yong Yi
>
> **摘要:** Vision-language models (VLMs) have recently been integrated into multiple instance learning (MIL) frameworks to address the challenge of few-shot, weakly supervised classification of whole slide images (WSIs). A key trend involves leveraging multi-scale information to better represent hierarchical tissue structures. However, existing methods often face two key limitations: (1) insufficient modeling of interactions within the same modalities across scales (e.g., 5x and 20x) and (2) inadequate alignment between visual and textual modalities on the same scale. To address these gaps, we propose HiVE-MIL, a hierarchical vision-language framework that constructs a unified graph consisting of (1) parent-child links between coarse (5x) and fine (20x) visual/textual nodes to capture hierarchical relationships, and (2) heterogeneous intra-scale edges linking visual and textual nodes on the same scale. To further enhance semantic consistency, HiVE-MIL incorporates a two-stage, text-guided dynamic filtering mechanism that removes weakly correlated patch-text pairs, and introduces a hierarchical contrastive loss to align textual semantics across scales. Extensive experiments on TCGA breast, lung, and kidney cancer datasets demonstrate that HiVE-MIL consistently outperforms both traditional MIL and recent VLM-based MIL approaches, achieving gains of up to 4.1% in macro F1 under 16-shot settings. Our results demonstrate the value of jointly modeling hierarchical structure and multimodal alignment for efficient and scalable learning from limited pathology data. The code is available at https://github.com/bryanwong17/HiVE-MIL
>
---
#### [replaced 019] Hunyuan-Game: Industrial-grade Intelligent Game Creation Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14135v2](http://arxiv.org/pdf/2505.14135v2)**

> **作者:** Ruihuang Li; Caijin Zhou; Shoujian Zheng; Jianxiang Lu; Jiabin Huang; Comi Chen; Junshu Tang; Guangzheng Xu; Jiale Tao; Hongmei Wang; Donghao Li; Wenqing Yu; Senbo Wang; Zhimin Li; Yetshuan Shi; Haoyu Yang; Yukun Wang; Wenxun Dai; Jiaqi Li; Linqing Wang; Qixun Wang; Zhiyong Xu; Yingfang Zhang; Jiangfeng Xiong; Weijie Kong; Chao Zhang; Hongxin Zhang; Qiaoling Zheng; Weiting Guo; Xinchi Deng; Yixuan Li; Renjia Wei; Yulin Jian; Duojun Huang; Xuhua Ren; Junkun Yuan; Zhengguang Zhou; Jiaxiang Cheng; Bing Ma; Shirui Huang; Jiawang Bai; Chao Li; Sihuan Lin; Yifu Sun; Yuan Zhou; Joey Wang; Qin Lin; Tianxiang Zheng; Jingmiao Yu; Jihong Zhang; Caesar Zhong; Di Wang; Yuhong Liu; Linus; Jie Jiang; Longhuang Wu; Shuai Shao; Qinglin Lu
>
> **摘要:** Intelligent game creation represents a transformative advancement in game development, utilizing generative artificial intelligence to dynamically generate and enhance game content. Despite notable progress in generative models, the comprehensive synthesis of high-quality game assets, including both images and videos, remains a challenging frontier. To create high-fidelity game content that simultaneously aligns with player preferences and significantly boosts designer efficiency, we present Hunyuan-Game, an innovative project designed to revolutionize intelligent game production. Hunyuan-Game encompasses two primary branches: image generation and video generation. The image generation component is built upon a vast dataset comprising billions of game images, leading to the development of a group of customized image generation models tailored for game scenarios: (1) General Text-to-Image Generation. (2) Game Visual Effects Generation, involving text-to-effect and reference image-based game visual effect generation. (3) Transparent Image Generation for characters, scenes, and game visual effects. (4) Game Character Generation based on sketches, black-and-white images, and white models. The video generation component is built upon a comprehensive dataset of millions of game and anime videos, leading to the development of five core algorithmic models, each targeting critical pain points in game development and having robust adaptation to diverse game video scenarios: (1) Image-to-Video Generation. (2) 360 A/T Pose Avatar Video Synthesis. (3) Dynamic Illustration Generation. (4) Generative Video Super-Resolution. (5) Interactive Game Video Generation. These image and video generation models not only exhibit high-level aesthetic expression but also deeply integrate domain-specific knowledge, establishing a systematic understanding of diverse game and anime art styles.
>
---
#### [replaced 020] ZooplanktonBench: A Geo-Aware Zooplankton Recognition and Classification Dataset from Marine Observations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18477v2](http://arxiv.org/pdf/2505.18477v2)**

> **作者:** Fukun Liu; Adam T. Greer; Gengchen Mai; Jin Sun
>
> **备注:** Accepted to KDD 2025 Datasets and Benchmarks Track
>
> **摘要:** Plankton are small drifting organisms found throughout the world's oceans and can be indicators of ocean health. One component of this plankton community is the zooplankton, which includes gelatinous animals and crustaceans (e.g. shrimp), as well as the early life stages (i.e., eggs and larvae) of many commercially important fishes. Being able to monitor zooplankton abundances accurately and understand how populations change in relation to ocean conditions is invaluable to marine science research, with important implications for future marine seafood productivity. While new imaging technologies generate massive amounts of video data of zooplankton, analyzing them using general-purpose computer vision tools turns out to be highly challenging due to the high similarity in appearance between the zooplankton and its background (e.g., marine snow). In this work, we present the ZooplanktonBench, a benchmark dataset containing images and videos of zooplankton associated with rich geospatial metadata (e.g., geographic coordinates, depth, etc.) in various water ecosystems. ZooplanktonBench defines a collection of tasks to detect, classify, and track zooplankton in challenging settings, including highly cluttered environments, living vs non-living classification, objects with similar shapes, and relatively small objects. Our dataset presents unique challenges and opportunities for state-of-the-art computer vision systems to evolve and improve visual understanding in dynamic environments characterized by significant variation and the need for geo-awareness. The code and settings described in this paper can be found on our website: https://lfk118.github.io/ZooplanktonBench_Webpage.
>
---
#### [replaced 021] Latent Beam Diffusion Models for Decoding Image Sequences
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20429v2](http://arxiv.org/pdf/2503.20429v2)**

> **作者:** Guilherme Fernandes; Vasco Ramos; Regev Cohen; Idan Szpektor; João Magalhães
>
> **摘要:** While diffusion models excel at generating high-quality images from text prompts, they struggle with visual consistency in image sequences. Existing methods generate each image independently, leading to disjointed narratives - a challenge further exacerbated in non-linear storytelling, where scenes must connect beyond adjacent frames. We introduce a novel beam search strategy for latent space exploration, enabling conditional generation of full image sequences with beam search decoding. Unlike prior approaches that use fixed latent priors, our method dynamically searches for an optimal sequence of latent representations, ensuring coherent visual transitions. As the latent denoising space is explored, the beam search graph is pruned with a cross-attention mechanism that efficiently scores search paths, prioritizing alignment with both textual prompts and visual context. Human and automatic evaluations confirm that BeamDiffusion outperforms other baseline methods, producing full sequences with superior coherence, visual continuity, and textual alignment.
>
---
#### [replaced 022] Eye-See-You: Reverse Pass-Through VR and Head Avatars
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18869v2](http://arxiv.org/pdf/2505.18869v2)**

> **作者:** Ankan Dash; Jingyi Gu; Guiling Wang; Chen Chen
>
> **备注:** 34th International Joint Conference on Artificial Intelligence, IJCAI 2025
>
> **摘要:** Virtual Reality (VR) headsets, while integral to the evolving digital ecosystem, present a critical challenge: the occlusion of users' eyes and portions of their faces, which hinders visual communication and may contribute to social isolation. To address this, we introduce RevAvatar, an innovative framework that leverages AI methodologies to enable reverse pass-through technology, fundamentally transforming VR headset design and interaction paradigms. RevAvatar integrates state-of-the-art generative models and multimodal AI techniques to reconstruct high-fidelity 2D facial images and generate accurate 3D head avatars from partially observed eye and lower-face regions. This framework represents a significant advancement in AI4Tech by enabling seamless interaction between virtual and physical environments, fostering immersive experiences such as VR meetings and social engagements. Additionally, we present VR-Face, a novel dataset comprising 200,000 samples designed to emulate diverse VR-specific conditions, including occlusions, lighting variations, and distortions. By addressing fundamental limitations in current VR systems, RevAvatar exemplifies the transformative synergy between AI and next-generation technologies, offering a robust platform for enhancing human connection and interaction in virtual environments.
>
---
#### [replaced 023] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.19385v4](http://arxiv.org/pdf/2503.19385v4)**

> **作者:** Jaihoon Kim; Taehoon Yoon; Jisung Hwang; Minhyuk Sung
>
> **备注:** Project page: https://flow-inference-time-scaling.github.io/
>
> **摘要:** We propose an inference-time scaling approach for pretrained flow models. Recently, inference-time scaling has gained significant attention in LLMs and diffusion models, improving sample quality or better aligning outputs with user preferences by leveraging additional computation. For diffusion models, particle sampling has allowed more efficient scaling due to the stochasticity at intermediate denoising steps. On the contrary, while flow models have gained popularity as an alternative to diffusion models--offering faster generation and high-quality outputs in state-of-the-art image and video generative models--efficient inference-time scaling methods used for diffusion models cannot be directly applied due to their deterministic generative process. To enable efficient inference-time scaling for flow models, we propose three key ideas: 1) SDE-based generation, enabling particle sampling in flow models, 2) Interpolant conversion, broadening the search space and enhancing sample diversity, and 3) Rollover Budget Forcing (RBF), an adaptive allocation of computational resources across timesteps to maximize budget utilization. Our experiments show that SDE-based generation, particularly variance-preserving (VP) interpolant-based generation, improves the performance of particle sampling methods for inference-time scaling in flow models. Additionally, we demonstrate that RBF with VP-SDE achieves the best performance, outperforming all previous inference-time scaling approaches.
>
---
#### [replaced 024] Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00619v2](http://arxiv.org/pdf/2502.00619v2)**

> **作者:** Yujin Oh; Pengfei Jin; Sangjoon Park; Sekeun Kim; Siyeop Yoon; Kyungsang Kim; Jin Sung Kim; Xiang Li; Quanzheng Li
>
> **备注:** ICML 2025 spotlight, see https://openreview.net/forum?id=BUONdewsBa
>
> **摘要:** Ensuring fairness in medical image segmentation is critical due to biases in imbalanced clinical data acquisition caused by demographic attributes (e.g., age, sex, race) and clinical factors (e.g., disease severity). To address these challenges, we introduce Distribution-aware Mixture of Experts (dMoE), inspired by optimal control theory. We provide a comprehensive analysis of its underlying mechanisms and clarify dMoE's role in adapting to heterogeneous distributions in medical image segmentation. Furthermore, we integrate dMoE into multiple network architectures, demonstrating its broad applicability across diverse medical image analysis tasks. By incorporating demographic and clinical factors, dMoE achieves state-of-the-art performance on two 2D benchmark datasets and a 3D in-house dataset. Our results highlight the effectiveness of dMoE in mitigating biases from imbalanced distributions, offering a promising approach to bridging control theory and medical image segmentation within fairness learning paradigms. The source code will be made available. The source code is available at https://github.com/tvseg/dMoE.
>
---
#### [replaced 025] Causality and "In-the-Wild" Video-Based Person Re-ID: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20540v2](http://arxiv.org/pdf/2505.20540v2)**

> **作者:** Md Rashidunnabi; Kailash Hambarde; Hugo Proença
>
> **备注:** 30 pages, 9 figures
>
> **摘要:** Video-based person re-identification (Re-ID) remains brittle in real-world deployments despite impressive benchmark performance. Most existing models rely on superficial correlations such as clothing, background, or lighting that fail to generalize across domains, viewpoints, and temporal variations. This survey examines the emerging role of causal reasoning as a principled alternative to traditional correlation-based approaches in video-based Re-ID. We provide a structured and critical analysis of methods that leverage structural causal models, interventions, and counterfactual reasoning to isolate identity-specific features from confounding factors. The survey is organized around a novel taxonomy of causal Re-ID methods that spans generative disentanglement, domain-invariant modeling, and causal transformers. We review current evaluation metrics and introduce causal-specific robustness measures. In addition, we assess practical challenges of scalability, fairness, interpretability, and privacy that must be addressed for real-world adoption. Finally, we identify open problems and outline future research directions that integrate causal modeling with efficient architectures and self-supervised learning. This survey aims to establish a coherent foundation for causal video-based person Re-ID and to catalyze the next phase of research in this rapidly evolving domain.
>
---
#### [replaced 026] SageAttention2++: A More Efficient Implementation of SageAttention2
- **分类: cs.LG; cs.AI; cs.AR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21136v2](http://arxiv.org/pdf/2505.21136v2)**

> **作者:** Jintao Zhang; Xiaoming Xu; Jia Wei; Haofeng Huang; Pengle Zhang; Chendong Xiang; Jun Zhu; Jianfei Chen
>
> **摘要:** The efficiency of attention is critical because its time complexity grows quadratically with sequence length. SageAttention2 addresses this by utilizing quantization to accelerate matrix multiplications (Matmul) in attention. To further accelerate SageAttention2, we propose to utilize the faster instruction of FP8 Matmul accumulated in FP16. The instruction is 2x faster than the FP8 Matmul used in SageAttention2. Our experiments show that SageAttention2++ achieves a 3.9x speedup over FlashAttention while maintaining the same attention accuracy as SageAttention2. This means SageAttention2++ effectively accelerates various models, including those for language, image, and video generation, with negligible end-to-end metrics loss. The code will be available at https://github.com/thu-ml/SageAttention.
>
---
#### [replaced 027] A Weak Supervision Learning Approach Towards an Equitable Mobility Estimation
- **分类: cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.04229v2](http://arxiv.org/pdf/2505.04229v2)**

> **作者:** Theophilus Aidoo; Till Koebe; Akansh Maurya; Hewan Shrestha; Ingmar Weber
>
> **备注:** To appear in the proceedings of the ICWSM'25 Workshop on Data for the Wellbeing of Most Vulnerable (DWMV). Please cite accordingly
>
> **摘要:** The scarcity and high cost of labeled high-resolution imagery have long challenged remote sensing applications, particularly in low-income regions where high-resolution data are scarce. In this study, we propose a weak supervision framework that estimates parking lot occupancy using 3m resolution satellite imagery. By leveraging coarse temporal labels -- based on the assumption that parking lots of major supermarkets and hardware stores in Germany are typically full on Saturdays and empty on Sundays -- we train a pairwise comparison model that achieves an AUC of 0.92 on large parking lots. The proposed approach minimizes the reliance on expensive high-resolution images and holds promise for scalable urban mobility analysis. Moreover, the method can be adapted to assess transit patterns and resource allocation in vulnerable communities, providing a data-driven basis to improve the well-being of those most in need.
>
---
#### [replaced 028] A Knowledge-guided Adversarial Defense for Resisting Malicious Visual Manipulation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08411v2](http://arxiv.org/pdf/2504.08411v2)**

> **作者:** Dawei Zhou; Suzhi Gang; Decheng Liu; Tongliang Liu; Nannan Wang; Xinbo Gao
>
> **摘要:** Malicious applications of visual manipulation have raised serious threats to the security and reputation of users in many fields. To alleviate these issues, adversarial noise-based defenses have been enthusiastically studied in recent years. However, ``data-only" methods tend to distort fake samples in the low-level feature space rather than the high-level semantic space, leading to limitations in resisting malicious manipulation. Frontier research has shown that integrating knowledge in deep learning can produce reliable and generalizable solutions. Inspired by these, we propose a knowledge-guided adversarial defense (KGAD) to actively force malicious manipulation models to output semantically confusing samples. Specifically, in the process of generating adversarial noise, we focus on constructing significant semantic confusions at the domain-specific knowledge level, and exploit a metric closely related to visual perception to replace the general pixel-wise metrics. The generated adversarial noise can actively interfere with the malicious manipulation model by triggering knowledge-guided and perception-related disruptions in the fake samples. To validate the effectiveness of the proposed method, we conduct qualitative and quantitative experiments on human perception and visual quality assessment. The results on two different tasks both show that our defense provides better protection compared to state-of-the-art methods and achieves great generalizability.
>
---
#### [replaced 029] Bridging Scales in Map Generation: A scale-aware cascaded generative mapping framework for seamless and consistent multi-scale cartographic representation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04991v3](http://arxiv.org/pdf/2502.04991v3)**

> **作者:** Chenxing Sun; Yongyang Xu; Xuwei Xu; Xixi Fan; Jing Bai; Xiechun Lu; Zhanlong Chen
>
> **摘要:** Multi-scale tile maps are essential for geographic information services, serving as fundamental outcomes of surveying and cartographic workflows. While existing image generation networks can produce map-like outputs from remote sensing imagery, their emphasis on replicating texture rather than preserving geospatial features limits cartographic validity. Current approaches face two fundamental challenges: inadequate integration of cartographic generalization principles with dynamic multi-scale generation and spatial discontinuities arising from tile-wise generation. To address these limitations, we propose a scale-aware cartographic generation framework (SCGM) that leverages conditional guided diffusion and a multi-scale cascade architecture. The framework introduces three key innovations: a scale modality encoding mechanism to formalize map generalization relationships, a scale-driven conditional encoder for robust feature fusion, and a cascade reference mechanism ensuring cross-scale visual consistency. By hierarchically constraining large-scale map synthesis with small-scale structural priors, SCGM effectively mitigates edge artifacts while maintaining geographic fidelity. Comprehensive evaluations on cartographic benchmarks confirm the framework's ability to generate seamless multi-scale tile maps with enhanced spatial coherence and generalization-aware representation, demonstrating significant potential for emergency mapping and automated cartography applications.
>
---
#### [replaced 030] Base and Exponent Prediction in Mathematical Expressions using Multi-Output CNN
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.14967v2](http://arxiv.org/pdf/2407.14967v2)**

> **作者:** Md Laraib Salam; Akash S Balsaraf; Gaurav Gupta; Ashish Rajeshwar Kulkarni
>
> **备注:** 4 pages, 9 figures
>
> **摘要:** The use of neural networks and deep learning techniques in image processing has significantly advanced the field, enabling highly accurate recognition results. However, achieving high recognition rates often necessitates complex network models, which can be challenging to train and require substantial computational resources. This research presents a simplified yet effective approach to predicting both the base and exponent from images of mathematical expressions using a multi-output Convolutional Neural Network (CNN). The model is trained on 10,900 synthetically generated images containing exponent expressions, incorporating random noise, font size variations, and blur intensity to simulate real-world conditions. The proposed CNN model demonstrates robust performance with efficient training time. The experimental results indicate that the model achieves high accuracy in predicting the base and exponent values, proving the efficacy of this approach in handling noisy and varied input images.
>
---
#### [replaced 031] Dual-Head Knowledge Distillation: Enhancing Logits Utilization with an Auxiliary Head
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.08937v2](http://arxiv.org/pdf/2411.08937v2)**

> **作者:** Penghui Yang; Chen-Chen Zong; Sheng-Jun Huang; Lei Feng; Bo An
>
> **备注:** Accepted by KDD 2025
>
> **摘要:** Traditional knowledge distillation focuses on aligning the student's predicted probabilities with both ground-truth labels and the teacher's predicted probabilities. However, the transition to predicted probabilities from logits would obscure certain indispensable information. To address this issue, it is intuitive to additionally introduce a logit-level loss function as a supplement to the widely used probability-level loss function, for exploiting the latent information of logits. Unfortunately, we empirically find that the amalgamation of the newly introduced logit-level loss and the previous probability-level loss will lead to performance degeneration, even trailing behind the performance of employing either loss in isolation. We attribute this phenomenon to the collapse of the classification head, which is verified by our theoretical analysis based on the neural collapse theory. Specifically, the gradients of the two loss functions exhibit contradictions in the linear classifier yet display no such conflict within the backbone. Drawing from the theoretical analysis, we propose a novel method called dual-head knowledge distillation, which partitions the linear classifier into two classification heads responsible for different losses, thereby preserving the beneficial effects of both losses on the backbone while eliminating adverse influences on the classification head. Extensive experiments validate that our method can effectively exploit the information inside the logits and achieve superior performance against state-of-the-art counterparts. Our code is available at: https://github.com/penghui-yang/DHKD.
>
---
#### [replaced 032] From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Reasoning-Driven Pedagogical Visualization
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16832v2](http://arxiv.org/pdf/2505.16832v2)**

> **作者:** Haonian Ji; Shi Qiu; Siyang Xin; Siwei Han; Zhaorun Chen; Dake Zhang; Hongyi Wang; Huaxiu Yao
>
> **备注:** 16 pages; 7 figures
>
> **摘要:** While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations. EduVisBench and EduVisAgent are available at https://github.com/aiming-lab/EduVisBench and https://github.com/aiming-lab/EduVisAgent.
>
---
#### [replaced 033] HoliTom: Holistic Token Merging for Fast Video Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21334v2](http://arxiv.org/pdf/2505.21334v2)**

> **作者:** Kele Shao; Keda Tao; Can Qin; Haoxuan You; Yang Sui; Huan Wang
>
> **备注:** version provides code link: https://github.com/cokeshao/HoliTom
>
> **摘要:** Video large language models (video LLMs) excel at video comprehension but face significant computational inefficiency due to redundant video tokens. Existing token pruning methods offer solutions. However, approaches operating within the LLM (inner-LLM pruning), such as FastV, incur intrinsic computational overhead in shallow layers. In contrast, methods performing token pruning before the LLM (outer-LLM pruning) primarily address spatial redundancy within individual frames or limited temporal windows, neglecting the crucial global temporal dynamics and correlations across longer video sequences. This leads to sub-optimal spatio-temporal reduction and does not leverage video compressibility fully. Crucially, the synergistic potential and mutual influence of combining these strategies remain unexplored. To further reduce redundancy, we introduce HoliTom, a novel training-free holistic token merging framework. HoliTom employs outer-LLM pruning through global redundancy-aware temporal segmentation, followed by spatial-temporal merging to reduce visual tokens by over 90%, significantly alleviating the LLM's computational burden. Complementing this, we introduce a robust inner-LLM token similarity-based merging approach, designed for superior performance and compatibility with outer-LLM pruning. Evaluations demonstrate our method's promising efficiency-performance trade-off on LLaVA-OneVision-7B, reducing computational costs to 6.9% of FLOPs while maintaining 99.1% of the original performance. Furthermore, we achieve a 2.28x reduction in Time-To-First-Token (TTFT) and a 1.32x acceleration in decoding throughput, highlighting the practical benefits of our integrated pruning approach for efficient video LLMs inference.
>
---
#### [replaced 034] Benchmarking performance, explainability, and evaluation strategies of vision-language models for surgery: Challenges and opportunities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10764v2](http://arxiv.org/pdf/2505.10764v2)**

> **作者:** Jiajun Cheng; Xianwu Zhao; Shan Lin
>
> **摘要:** Minimally invasive surgery (MIS) presents significant visual challenges, including a limited field of view, specular reflections, and inconsistent lighting conditions due to the small incision and the use of endoscopes. Over the past decade, many machine learning and deep learning models have been developed to identify and detect instruments and anatomical structures in surgical videos. However, these models are typically trained on manually labeled, procedure- and task-specific datasets that are relatively small, resulting in limited generalization to unseen data.In practice, hospitals generate a massive amount of raw surgical data every day, including videos captured during various procedures. Labeling this data is almost impractical, as it requires highly specialized expertise. The recent success of vision-language models (VLMs), which can be trained on large volumes of raw image-text pairs and exhibit strong adaptability, offers a promising alternative for leveraging unlabeled surgical data. While some existing work has explored applying VLMs to surgical tasks, their performance remains limited. To support future research in developing more effective VLMs for surgical applications, this paper aims to answer a key question: How well do existing VLMs, both general-purpose and surgery-specific perform on surgical data, and what types of scenes do they struggle with? To address this, we conduct a benchmarking study of several popular VLMs across diverse laparoscopic datasets. Specifically, we visualize the model's attention to identify which regions of the image it focuses on when making predictions for surgical tasks. We also propose a metric to evaluate whether the model attends to task-relevant regions. Our findings reveal a mismatch between prediction accuracy and visual grounding, indicating that models may make correct predictions while focusing on irrelevant areas of the image.
>
---
#### [replaced 035] Human-Object Interaction via Automatically Designed VLM-Guided Motion Policy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18349v2](http://arxiv.org/pdf/2503.18349v2)**

> **作者:** Zekai Deng; Ye Shi; Kaiyang Ji; Lan Xu; Shaoli Huang; Jingya Wang
>
> **备注:** change the style
>
> **摘要:** Human-object interaction (HOI) synthesis is crucial for applications in animation, simulation, and robotics. However, existing approaches either rely on expensive motion capture data or require manual reward engineering, limiting their scalability and generalizability. In this work, we introduce the first unified physics-based HOI framework that leverages Vision-Language Models (VLMs) to enable long-horizon interactions with diverse object types, including static, dynamic, and articulated objects. We introduce VLM-Guided Relative Movement Dynamics (RMD), a fine-grained spatio-temporal bipartite representation that automatically constructs goal states and reward functions for reinforcement learning. By encoding structured relationships between human and object parts, RMD enables VLMs to generate semantically grounded, interaction-aware motion guidance without manual reward tuning. To support our methodology, we present Interplay, a novel dataset with thousands of long-horizon static and dynamic interaction plans. Extensive experiments demonstrate that our framework outperforms existing methods in synthesizing natural, human-like motions across both simple single-task and complex multi-task scenarios. For more details, please refer to our project webpage: https://vlm-rmd.github.io/.
>
---
#### [replaced 036] DreamMask: Boosting Open-vocabulary Panoptic Segmentation with Synthetic Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.02048v2](http://arxiv.org/pdf/2501.02048v2)**

> **作者:** Yuanpeng Tu; Xi Chen; Ser-Nam Lim; Hengshuang Zhao
>
> **备注:** Accepted by SIGGRAPH2025 Project url: https://yuanpengtu.github.io/Dreammask-Page/
>
> **摘要:** Open-vocabulary panoptic segmentation has received significant attention due to its applicability in the real world. Despite claims of robust generalization, we find that the advancements of previous works are attributed mainly on trained categories, exposing a lack of generalization to novel classes. In this paper, we explore boosting existing models from a data-centric perspective. We propose DreamMask, which systematically explores how to generate training data in the open-vocabulary setting, and how to train the model with both real and synthetic data. For the first part, we propose an automatic data generation pipeline with off-the-shelf models. We propose crucial designs for vocabulary expansion, layout arrangement, data filtering, etc. Equipped with these techniques, our generated data could significantly outperform the manually collected web data. To train the model with generated data, a synthetic-real alignment loss is designed to bridge the representation gap, bringing noticeable improvements across multiple benchmarks. In general, DreamMask significantly simplifies the collection of large-scale training data, serving as a plug-and-play enhancement for existing methods. For instance, when trained on COCO and tested on ADE20K, the model equipped with DreamMask outperforms the previous state-of-the-art by a substantial margin of 2.1% mIoU.
>
---
#### [replaced 037] HTMNet: A Hybrid Network with Transformer-Mamba Bottleneck Multimodal Fusion for Transparent and Reflective Objects Depth Completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20904v2](http://arxiv.org/pdf/2505.20904v2)**

> **作者:** Guanghu Xie; Yonglong Zhang; Zhiduo Jiang; Yang Liu; Zongwu Xie; Baoshi Cao; Hong Liu
>
> **摘要:** Transparent and reflective objects pose significant challenges for depth sensors, resulting in incomplete depth information that adversely affects downstream robotic perception and manipulation tasks. To address this issue, we propose HTMNet, a novel hybrid model integrating Transformer, CNN, and Mamba architectures. The encoder is based on a dual-branch CNN-Transformer framework, the bottleneck fusion module adopts a Transformer-Mamba architecture, and the decoder is built upon a multi-scale fusion module. We introduce a novel multimodal fusion module grounded in self-attention mechanisms and state space models, marking the first application of the Mamba architecture in the field of transparent object depth completion and revealing its promising potential. Additionally, we design an innovative multi-scale fusion module for the decoder that combines channel attention, spatial attention, and multi-scale feature extraction techniques to effectively integrate multi-scale features through a down-fusion strategy. Extensive evaluations on multiple public datasets demonstrate that our model achieves state-of-the-art(SOTA) performance, validating the effectiveness of our approach.
>
---
#### [replaced 038] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12312v3](http://arxiv.org/pdf/2505.12312v3)**

> **作者:** Qi Feng
>
> **备注:** Author list corrected. In version 1, Hidetoshi Shimodaira was included as a co-author without their consent and has been removed from the author list
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [replaced 039] Diffusion Models as Cartoonists: The Curious Case of High Density Regions
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.01293v4](http://arxiv.org/pdf/2411.01293v4)**

> **作者:** Rafał Karczewski; Markus Heinonen; Vikas Garg
>
> **备注:** ICLR 2025
>
> **摘要:** We investigate what kind of images lie in the high-density regions of diffusion models. We introduce a theoretical mode-tracking process capable of pinpointing the exact mode of the denoising distribution, and we propose a practical high-density sampler that consistently generates images of higher likelihood than usual samplers. Our empirical findings reveal the existence of significantly higher likelihood samples that typical samplers do not produce, often manifesting as cartoon-like drawings or blurry images depending on the noise level. Curiously, these patterns emerge in datasets devoid of such examples. We also present a novel approach to track sample likelihoods in diffusion SDEs, which remarkably incurs no additional computational cost. Code is available at https://github.com/Aalto-QuML/high-density-diffusion.
>
---
#### [replaced 040] EventEgoHands: Event-based Egocentric 3D Hand Mesh Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19169v3](http://arxiv.org/pdf/2505.19169v3)**

> **作者:** Ryosei Hara; Wataru Ikeda; Masashi Hatano; Mariko Isogawa
>
> **备注:** IEEE International Conference on Image Processing 2025, Project Page: https://ryhara.github.io/EventEgoHands/
>
> **摘要:** Reconstructing 3D hand mesh is challenging but an important task for human-computer interaction and AR/VR applications. In particular, RGB and/or depth cameras have been widely used in this task. However, methods using these conventional cameras face challenges in low-light environments and during motion blur. Thus, to address these limitations, event cameras have been attracting attention in recent years for their high dynamic range and high temporal resolution. Despite their advantages, event cameras are sensitive to background noise or camera motion, which has limited existing studies to static backgrounds and fixed cameras. In this study, we propose EventEgoHands, a novel method for event-based 3D hand mesh reconstruction in an egocentric view. Our approach introduces a Hand Segmentation Module that extracts hand regions, effectively mitigating the influence of dynamic background events. We evaluated our approach and demonstrated its effectiveness on the N-HOT3D dataset, improving MPJPE by approximately more than 4.5 cm (43%).
>
---
#### [replaced 041] FocusChat: Text-guided Long Video Understanding via Spatiotemporal Information Filtering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.12833v2](http://arxiv.org/pdf/2412.12833v2)**

> **作者:** Zheng Cheng; Rendong Wang; Zhicheng Wang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Recently, multi-modal large language models have made significant progress. However, visual information lacking of guidance from the user's intention may lead to redundant computation and involve unnecessary visual noise, especially in long, untrimmed videos. To address this issue, we propose FocusChat, a text-guided multi-modal large language model (LLM) that emphasizes visual information correlated to the user's prompt. In detail, Our model first undergoes the semantic extraction module, which comprises a visual semantic branch and a text semantic branch to extract image and text semantics, respectively. The two branches are combined using the Spatial-Temporal Filtering Module (STFM). STFM enables explicit spatial-level information filtering and implicit temporal-level feature filtering, ensuring that the visual tokens are closely aligned with the user's query. It lowers the essential number of visual tokens inputted into the LLM. FocusChat significantly outperforms Video-LLaMA in zero-shot experiments, using an order of magnitude less training data with only 16 visual tokens occupied. It achieves results comparable to the state-of-the-art in few-shot experiments, with only 0.72M pre-training data.
>
---
#### [replaced 042] LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.02406v3](http://arxiv.org/pdf/2502.02406v3)**

> **作者:** Tzu-Tao Chang; Shivaram Venkataraman
>
> **摘要:** Cross-attention is commonly adopted in multimodal large language models (MLLMs) for integrating visual information into the language backbone. However, in applications with large visual inputs, such as video understanding, processing a large number of visual tokens in cross-attention layers leads to high memory demands and often necessitates distributed computation across multiple GPUs. Existing distributed attention mechanisms face significant communication overheads, making cross-attention layers a critical bottleneck for efficient training and inference of MLLMs. To address this, we propose LV-XAttn, a distributed, exact cross-attention mechanism with minimal communication overhead. We observe that in applications involving large visual inputs, the size of the query block is typically much smaller than that of the key-value blocks. Thus, in LV-XAttn we keep the large key-value blocks locally on each GPU and exchange smaller query blocks across GPUs. We also introduce an efficient activation recomputation technique to support longer visual context. We theoretically analyze the communication benefits of LV-XAttn and show that it can achieve speedups for a wide range of models. Our evaluations with Llama 3-V, mPLUG-Owl3 and OpenFlamingo models find that LV-XAttn achieves up to 10.62$\times$ end-to-end speedup compared to existing approaches.
>
---
#### [replaced 043] SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.03561v2](http://arxiv.org/pdf/2504.03561v2)**

> **作者:** Runnan Fang; Xiaobin Wang; Yuan Liang; Shuofei Qiao; Jialong Wu; Zekun Xi; Ningyu Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** In the interaction between agents and their environments, agents expand their capabilities by planning and executing actions. However, LLM-based agents face substantial challenges when deployed in novel environments or required to navigate unconventional action spaces. To empower agents to autonomously explore environments, optimize workflows, and enhance their understanding of actions, we propose SynWorld, a framework that allows agents to synthesize possible scenarios with multi-step action invocation within the action space and perform Monte Carlo Tree Search (MCTS) exploration to effectively refine their action knowledge in the current environment. Our experiments demonstrate that SynWorld is an effective and general approach to learning action knowledge in new environments. Code is available at https://github.com/zjunlp/SynWorld.
>
---
#### [replaced 044] Event-based Stereo Depth Estimation: A Survey
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.17680v2](http://arxiv.org/pdf/2409.17680v2)**

> **作者:** Suman Ghosh; Guillermo Gallego
>
> **备注:** 28 pages, 24 figures, 7 tables
>
> **摘要:** Stereopsis has widespread appeal in robotics as it is the predominant way by which living beings perceive depth to navigate our 3D world. Event cameras are novel bio-inspired sensors that detect per-pixel brightness changes asynchronously, with very high temporal resolution and high dynamic range, enabling machine perception in high-speed motion and broad illumination conditions. The high temporal precision also benefits stereo matching, making disparity (depth) estimation a popular research area for event cameras ever since its inception. Over the last 30 years, the field has evolved rapidly, from low-latency, low-power circuit design to current deep learning (DL) approaches driven by the computer vision community. The bibliography is vast and difficult to navigate for non-experts due its highly interdisciplinary nature. Past surveys have addressed distinct aspects of this topic, in the context of applications, or focusing only on a specific class of techniques, but have overlooked stereo datasets. This survey provides a comprehensive overview, covering both instantaneous stereo and long-term methods suitable for simultaneous localization and mapping (SLAM), along with theoretical and empirical comparisons. It is the first to extensively review DL methods as well as stereo datasets, even providing practical suggestions for creating new benchmarks to advance the field. The main advantages and challenges faced by event-based stereo depth estimation are also discussed. Despite significant progress, challenges remain in achieving optimal performance in not only accuracy but also efficiency, a cornerstone of event-based computing. We identify several gaps and propose future research directions. We hope this survey inspires future research in this area, by serving as an accessible entry point for newcomers, as well as a practical guide for seasoned researchers in the community.
>
---
#### [replaced 045] Variational Positive-incentive Noise: How Noise Benefits Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.07651v2](http://arxiv.org/pdf/2306.07651v2)**

> **作者:** Hongyuan Zhang; Sida Huang; Yubin Guo; Xuelong Li
>
> **备注:** Acceptted by IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** A large number of works aim to alleviate the impact of noise due to an underlying conventional assumption of the negative role of noise. However, some existing works show that the assumption does not always hold. In this paper, we investigate how to benefit the classical models by random noise under the framework of Positive-incentive Noise (Pi-Noise). Since the ideal objective of Pi-Noise is intractable, we propose to optimize its variational bound instead, namely variational Pi-Noise (VPN). With the variational inference, a VPN generator implemented by neural networks is designed for enhancing base models and simplifying the inference of base models, without changing the architecture of base models. Benefiting from the independent design of base models and VPN generators, the VPN generator can work with most existing models. From the experiments, it is shown that the proposed VPN generator can improve the base models. It is appealing that the trained variational VPN generator prefers to blur the irrelevant ingredients in complicated images, which meets our expectations.
>
---
#### [replaced 046] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12363v3](http://arxiv.org/pdf/2505.12363v3)**

> **作者:** Qi Feng
>
> **备注:** In version 1, Hidetoshi Shimodaira was included as a co-author without their consent and has been removed from the author list
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [replaced 047] MMIG-Bench: Towards Comprehensive and Explainable Evaluation of Multi-Modal Image Generation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19415v2](http://arxiv.org/pdf/2505.19415v2)**

> **作者:** Hang Hua; Ziyun Zeng; Yizhi Song; Yunlong Tang; Liu He; Daniel Aliaga; Wei Xiong; Jiebo Luo
>
> **摘要:** Recent multimodal image generators such as GPT-4o, Gemini 2.0 Flash, and Gemini 2.5 Pro excel at following complex instructions, editing images and maintaining concept consistency. However, they are still evaluated by disjoint toolkits: text-to-image (T2I) benchmarks that lacks multi-modal conditioning, and customized image generation benchmarks that overlook compositional semantics and common knowledge. We propose MMIG-Bench, a comprehensive Multi-Modal Image Generation Benchmark that unifies these tasks by pairing 4,850 richly annotated text prompts with 1,750 multi-view reference images across 380 subjects, spanning humans, animals, objects, and artistic styles. MMIG-Bench is equipped with a three-level evaluation framework: (1) low-level metrics for visual artifacts and identity preservation of objects; (2) novel Aspect Matching Score (AMS): a VQA-based mid-level metric that delivers fine-grained prompt-image alignment and shows strong correlation with human judgments; and (3) high-level metrics for aesthetics and human preference. Using MMIG-Bench, we benchmark 17 state-of-the-art models, including Gemini 2.5 Pro, FLUX, DreamBooth, and IP-Adapter, and validate our metrics with 32k human ratings, yielding in-depth insights into architecture and data design.
>
---
#### [replaced 048] VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20718v2](http://arxiv.org/pdf/2505.20718v2)**

> **作者:** Kui Wu; Shuhang Xu; Hao Chen; Churan Wang; Zhoujun Li; Yizhou Wang; Fangwei Zhong
>
> **摘要:** We introduce a novel self-improving framework that enhances Embodied Visual Tracking (EVT) with Vision-Language Models (VLMs) to address the limitations of current active visual tracking systems in recovering from tracking failure. Our approach combines the off-the-shelf active tracking methods with VLMs' reasoning capabilities, deploying a fast visual policy for normal tracking and activating VLM reasoning only upon failure detection. The framework features a memory-augmented self-reflection mechanism that enables the VLM to progressively improve by learning from past experiences, effectively addressing VLMs' limitations in 3D spatial reasoning. Experimental results demonstrate significant performance improvements, with our framework boosting success rates by $72\%$ with state-of-the-art RL-based approaches and $220\%$ with PID-based methods in challenging environments. This work represents the first integration of VLM-based reasoning to assist EVT agents in proactive failure recovery, offering substantial advances for real-world robotic applications that require continuous target monitoring in dynamic, unstructured environments. Project website: https://sites.google.com/view/evt-recovery-assistant.
>
---
#### [replaced 049] ORIGEN: Zero-Shot 3D Orientation Grounding in Text-to-Image Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22194v2](http://arxiv.org/pdf/2503.22194v2)**

> **作者:** Yunhong Min; Daehyeon Choi; Kyeongmin Yeo; Jihyun Lee; Minhyuk Sung
>
> **备注:** Project Page: https://origen2025.github.io
>
> **摘要:** We introduce ORIGEN, the first zero-shot method for 3D orientation grounding in text-to-image generation across multiple objects and diverse categories. While previous work on spatial grounding in image generation has mainly focused on 2D positioning, it lacks control over 3D orientation. To address this, we propose a reward-guided sampling approach using a pretrained discriminative model for 3D orientation estimation and a one-step text-to-image generative flow model. While gradient-ascent-based optimization is a natural choice for reward-based guidance, it struggles to maintain image realism. Instead, we adopt a sampling-based approach using Langevin dynamics, which extends gradient ascent by simply injecting random noise--requiring just a single additional line of code. Additionally, we introduce adaptive time rescaling based on the reward function to accelerate convergence. Our experiments show that ORIGEN outperforms both training-based and test-time guidance methods across quantitative metrics and user studies.
>
---
#### [replaced 050] Complex Wavelet Mutual Information Loss: A Multi-Scale Loss Function for Semantic Segmentation
- **分类: cs.CV; eess.IV; 68T07**

- **链接: [http://arxiv.org/pdf/2502.00563v2](http://arxiv.org/pdf/2502.00563v2)**

> **作者:** Renhao Lu
>
> **备注:** Accepted at ICML 2025. This version corresponds to the official camera-ready submission
>
> **摘要:** Recent advancements in deep neural networks have significantly enhanced the performance of semantic segmentation. However, class imbalance and instance imbalance remain persistent challenges, where smaller instances and thin boundaries are often overshadowed by larger structures. To address the multiscale nature of segmented objects, various models have incorporated mechanisms such as spatial attention and feature pyramid networks. Despite these advancements, most loss functions are still primarily pixel-wise, while regional and boundary-focused loss functions often incur high computational costs or are restricted to small-scale regions. To address this limitation, we propose the complex wavelet mutual information (CWMI) loss, a novel loss function that leverages mutual information from subband images decomposed by a complex steerable pyramid. The complex steerable pyramid captures features across multiple orientations and preserves structural similarity across scales. Meanwhile, mutual information is well-suited to capturing high-dimensional directional features and offers greater noise robustness. Extensive experiments on diverse segmentation datasets demonstrate that CWMI loss achieves significant improvements in both pixel-wise accuracy and topological metrics compared to state-of-the-art methods, while introducing minimal computational overhead. Our code is available at https://github.com/lurenhaothu/CWMI
>
---
#### [replaced 051] Functionality understanding and segmentation in 3D scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16310v5](http://arxiv.org/pdf/2411.16310v5)**

> **作者:** Jaime Corsetti; Francesco Giuliari; Alice Fasoli; Davide Boscaini; Fabio Poiesi
>
> **备注:** CVPR 2025 Highlight. Camera ready version. 20 pages, 12 figures, 7 tables. Fixed typo in Eq.2
>
> **摘要:** Understanding functionalities in 3D scenes involves interpreting natural language descriptions to locate functional interactive objects, such as handles and buttons, in a 3D environment. Functionality understanding is highly challenging, as it requires both world knowledge to interpret language and spatial perception to identify fine-grained objects. For example, given a task like 'turn on the ceiling light', an embodied AI agent must infer that it needs to locate the light switch, even though the switch is not explicitly mentioned in the task description. To date, no dedicated methods have been developed for this problem. In this paper, we introduce Fun3DU, the first approach designed for functionality understanding in 3D scenes. Fun3DU uses a language model to parse the task description through Chain-of-Thought reasoning in order to identify the object of interest. The identified object is segmented across multiple views of the captured scene by using a vision and language model. The segmentation results from each view are lifted in 3D and aggregated into the point cloud using geometric information. Fun3DU is training-free, relying entirely on pre-trained models. We evaluate Fun3DU on SceneFun3D, the most recent and only dataset to benchmark this task, which comprises over 3000 task descriptions on 230 scenes. Our method significantly outperforms state-of-the-art open-vocabulary 3D segmentation approaches. Project page: https://tev-fbk.github.io/fun3du/
>
---
#### [replaced 052] VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19684v2](http://arxiv.org/pdf/2505.19684v2)**

> **作者:** Bingrui Sima; Linhua Cong; Wenxuan Wang; Kun He
>
> **摘要:** The emergence of Multimodal Large Language Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA's significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs -- their visual reasoning -- can also serve as an attack vector, posing significant security risks.
>
---
#### [replaced 053] Meta Co-Training: Two Views are Better than One
- **分类: cs.CV; cs.LG; I.2.6; I.4.10**

- **链接: [http://arxiv.org/pdf/2311.18083v5](http://arxiv.org/pdf/2311.18083v5)**

> **作者:** Jay C. Rothenberger; Dimitrios I. Diochnos
>
> **备注:** 16 pages, 16 figures, 11 tables, for implementation see https://github.com/JayRothenberger/Meta-Co-Training
>
> **摘要:** In many critical computer vision scenarios unlabeled data is plentiful, but labels are scarce and difficult to obtain. As a result, semi-supervised learning which leverages unlabeled data to boost the performance of supervised classifiers have received significant attention in recent literature. One representative class of semi-supervised algorithms are co-training algorithms. Co-training algorithms leverage two different models which have access to different independent and sufficient representations or "views" of the data to jointly make better predictions. Each of these models creates pseudo-labels on unlabeled points which are used to improve the other model. We show that in the common case where independent views are not available, we can construct such views inexpensively using pre-trained models. Co-training on the constructed views yields a performance improvement over any of the individual views we construct and performance comparable with recent approaches in semi-supervised learning. We present Meta Co-Training, a novel semi-supervised learning algorithm, which has two advantages over co-training: (i) learning is more robust when there is large discrepancy between the information content of the different views, and (ii) does not require retraining from scratch on each iteration. Our method achieves new state-of-the-art performance on ImageNet-10% achieving a ~4.7% reduction in error rate over prior work. Our method also outperforms prior semi-supervised work on several other fine-grained image classification datasets.
>
---
#### [replaced 054] Sampling Strategies for Efficient Training of Deep Learning Object Detection Algorithms
- **分类: cs.CV; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.18302v2](http://arxiv.org/pdf/2505.18302v2)**

> **作者:** Gefei Shen; Yung-Hong Sun; Yu Hen Hu; Hongrui Jiang
>
> **摘要:** Two sampling strategies are investigated to enhance efficiency in training a deep learning object detection model. These sampling strategies are employed under the assumption of Lipschitz continuity of deep learning models. The first strategy is uniform sampling which seeks to obtain samples evenly yet randomly through the state space of the object dynamics. The second strategy of frame difference sampling is developed to explore the temporal redundancy among successive frames in a video. Experiment result indicates that these proposed sampling strategies provide a dataset that yields good training performance while requiring relatively few manually labelled samples.
>
---
#### [replaced 055] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.11985v4](http://arxiv.org/pdf/2405.11985v4)**

> **作者:** Jingqun Tang; Qi Liu; Yongjie Ye; Jinghui Lu; Shu Wei; Chunhui Lin; Wanqing Li; Mohamad Fitri Faiz Bin Mahmood; Hao Feng; Zhen Zhao; Yanjie Wang; Yuliang Liu; Hao Liu; Xiang Bai; Can Huang
>
> **备注:** Accepted by ACL 2025 findings
>
> **摘要:** Text-Centric Visual Question Answering (TEC-VQA) in its proper format not only facilitates human-machine interaction in text-centric visual environments but also serves as a de facto gold proxy to evaluate AI models in the domain of text-centric scene understanding. Nonetheless, most existing TEC-VQA benchmarks have focused on high-resource languages like English and Chinese. Despite pioneering works to expand multilingual QA pairs in non-text-centric VQA datasets through translation engines, the translation-based protocol encounters a substantial "visual-textual misalignment" problem when applied to TEC-VQA. Specifically, it prioritizes the text in question-answer pairs while disregarding the visual text present in images. Moreover, it fails to address complexities related to nuanced meaning, contextual distortion, language bias, and question-type diversity. In this work, we tackle multilingual TEC-VQA by introducing MTVQA, the first benchmark featuring high-quality human expert annotations across 9 diverse languages, consisting of 6,778 question-answer pairs across 2,116 images. Further, by comprehensively evaluating numerous state-of-the-art Multimodal Large Language Models~(MLLMs), including Qwen2-VL, GPT-4o, GPT-4V, Claude3, and Gemini, on the MTVQA benchmark, it is evident that there is still a large room for performance improvement (Qwen2-VL scoring 30.9 versus 79.7 for human performance), underscoring the value of MTVQA. Additionally, we supply multilingual training data within the MTVQA dataset, demonstrating that straightforward fine-tuning with this data can substantially enhance multilingual TEC-VQA performance. We aspire that MTVQA will offer the research community fresh insights and stimulate further exploration in multilingual visual text comprehension. The project homepage is available at https://bytedance.github.io/MTVQA/.
>
---
#### [replaced 056] VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01427v4](http://arxiv.org/pdf/2501.01427v4)**

> **作者:** Yuanpeng Tu; Hao Luo; Xi Chen; Sihui Ji; Xiang Bai; Hengshuang Zhao
>
> **备注:** Accepted by SIGGRAPH2025 Project page: https://videoanydoor.github.io/
>
> **摘要:** Despite significant advancements in video generation, inserting a given object into videos remains a challenging task. The difficulty lies in preserving the appearance details of the reference object and accurately modeling coherent motions at the same time. In this paper, we propose VideoAnydoor, a zero-shot video object insertion framework with high-fidelity detail preservation and precise motion control. Starting from a text-to-video model, we utilize an ID extractor to inject the global identity and leverage a box sequence to control the overall motion. To preserve the detailed appearance and meanwhile support fine-grained motion control, we design a pixel warper. It takes the reference image with arbitrary key-points and the corresponding key-point trajectories as inputs. It warps the pixel details according to the trajectories and fuses the warped features with the diffusion U-Net, thus improving detail preservation and supporting users in manipulating the motion trajectories. In addition, we propose a training strategy involving both videos and static images with a weighted loss to enhance insertion quality. VideoAnydoor demonstrates significant superiority over existing methods and naturally supports various downstream applications (e.g., talking head generation, video virtual try-on, multi-region editing) without task-specific fine-tuning.
>
---
#### [replaced 057] Coverage Biases in High-Resolution Satellite Imagery
- **分类: cs.CY; astro-ph.EP; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03842v2](http://arxiv.org/pdf/2505.03842v2)**

> **作者:** Vadim Musienko; Axel Jacquet; Ingmar Weber; Till Koebe
>
> **备注:** To appear in the proceedings of the ICWSM'25 Workshop on Data for the Wellbeing of Most Vulnerable (DWMV). Please cite accordingly
>
> **摘要:** Satellite imagery is increasingly used to complement traditional data collection approaches such as surveys and censuses across scientific disciplines. However, we ask: Do all places on earth benefit equally from this new wealth of information? In this study, we investigate coverage bias of major satellite constellations that provide optical satellite imagery with a ground sampling distance below 10 meters, evaluating both the future on-demand tasking opportunities as well as the availability of historic images across the globe. Specifically, forward-looking, we estimate how often different places are revisited during a window of 30 days based on the satellites' orbital paths, thus investigating potential coverage biases caused by physical factors. We find that locations farther away from the equator are generally revisited more frequently by the constellations under study. Backward-looking, we show that historic satellite image availability -- based on metadata collected from major satellite imagery providers -- is influenced by socio-economic factors on the ground: less developed, less populated places have less satellite images available. Furthermore, in three small case studies on recent conflict regions in this world, namely Gaza, Sudan and Ukraine, we show that also geopolitical events play an important role in satellite image availability, hinting at underlying business model decisions. These insights lay bare that the digital dividend yielded by satellite imagery is not equally distributed across our planet.
>
---
#### [replaced 058] Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20789v2](http://arxiv.org/pdf/2505.20789v2)**

> **作者:** Yang Zheng; Wen Li; Zhaoqiang Liu
>
> **备注:** ICML 2025
>
> **摘要:** Inverse problems (IPs) involve reconstructing signals from noisy observations. Recently, diffusion models (DMs) have emerged as a powerful framework for solving IPs, achieving remarkable reconstruction performance. However, existing DM-based methods frequently encounter issues such as heavy computational demands and suboptimal convergence. In this work, building upon the idea of the recent work DMPlug, we propose two novel methods, DMILO and DMILO-PGD, to address these challenges. Our first method, DMILO, employs intermediate layer optimization (ILO) to alleviate the memory burden inherent in DMPlug. Additionally, by introducing sparse deviations, we expand the range of DMs, enabling the exploration of underlying signals that may lie outside the range of the diffusion model. We further propose DMILO-PGD, which integrates ILO with projected gradient descent (PGD), thereby reducing the risk of suboptimal convergence. We provide an intuitive theoretical analysis of our approaches under appropriate conditions and validate their superiority through extensive experiments on diverse image datasets, encompassing both linear and nonlinear IPs. Our results demonstrate significant performance gains over state-of-the-art methods, highlighting the effectiveness of DMILO and DMILO-PGD in addressing common challenges in DM-based IP solvers.
>
---
#### [replaced 059] MagicTryOn: Harnessing Diffusion Transformer for Garment-Preserving Video Virtual Try-on
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21325v2](http://arxiv.org/pdf/2505.21325v2)**

> **作者:** Guangyuan Li; Siming Zheng; Hao Zhang; Jinwei Chen; Junsheng Luan; Binkai Ou; Lei Zhao; Bo Li; Peng-Tao Jiang
>
> **摘要:** Video Virtual Try-On (VVT) aims to simulate the natural appearance of garments across consecutive video frames, capturing their dynamic variations and interactions with human body motion. However, current VVT methods still face challenges in terms of spatiotemporal consistency and garment content preservation. First, they use diffusion models based on the U-Net, which are limited in their expressive capability and struggle to reconstruct complex details. Second, they adopt a separative modeling approach for spatial and temporal attention, which hinders the effective capture of structural relationships and dynamic consistency across frames. Third, their expression of garment details remains insufficient, affecting the realism and stability of the overall synthesized results, especially during human motion. To address the above challenges, we propose MagicTryOn, a video virtual try-on framework built upon the large-scale video diffusion Transformer. We replace the U-Net architecture with a diffusion Transformer and combine full self-attention to jointly model the spatiotemporal consistency of videos. We design a coarse-to-fine garment preservation strategy. The coarse strategy integrates garment tokens during the embedding stage, while the fine strategy incorporates multiple garment-based conditions, such as semantics, textures, and contour lines during the denoising stage. Moreover, we introduce a mask-aware loss to further optimize garment region fidelity. Extensive experiments on both image and video try-on datasets demonstrate that our method outperforms existing SOTA methods in comprehensive evaluations and generalizes to in-the-wild scenarios.
>
---
#### [replaced 060] See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20641v2](http://arxiv.org/pdf/2505.20641v2)**

> **作者:** Yuan Wu; Zhiqiang Yan; Yigong Zhang; Xiang Li; Jian Yang
>
> **摘要:** Occupancy prediction aims to estimate the 3D spatial distribution of occupied regions along with their corresponding semantic labels. Existing vision-based methods perform well on daytime benchmarks but struggle in nighttime scenarios due to limited visibility and challenging lighting conditions. To address these challenges, we propose \textbf{LIAR}, a novel framework that learns illumination-affined representations. LIAR first introduces Selective Low-light Image Enhancement (SLLIE), which leverages the illumination priors from daytime scenes to adaptively determine whether a nighttime image is genuinely dark or sufficiently well-lit, enabling more targeted global enhancement. Building on the illumination maps generated by SLLIE, LIAR further incorporates two illumination-aware components: 2D Illumination-guided Sampling (2D-IGS) and 3D Illumination-driven Projection (3D-IDP), to respectively tackle local underexposure and overexposure. Specifically, 2D-IGS modulates feature sampling positions according to illumination maps, assigning larger offsets to darker regions and smaller ones to brighter regions, thereby alleviating feature degradation in underexposed areas. Subsequently, 3D-IDP enhances semantic understanding in overexposed regions by constructing illumination intensity fields and supplying refined residual queries to the BEV context refinement process. Extensive experiments on both real and synthetic datasets demonstrate the superior performance of LIAR under challenging nighttime scenarios. The source code and pretrained models are available \href{https://github.com/yanzq95/LIAR}{here}.
>
---
#### [replaced 061] A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction based on Content/Style Modeling
- **分类: eess.IV; cs.CV; physics.med-ph; I.4.5**

- **链接: [http://arxiv.org/pdf/2409.13477v3](http://arxiv.org/pdf/2409.13477v3)**

> **作者:** Chinmay Rao; Matthias van Osch; Nicola Pezzotti; Jeroen de Bresser; Laurens Beljaards; Jakob Meineke; Elwin de Weerdt; Huangling Lu; Mariya Doneva; Marius Staring
>
> **摘要:** Since multiple MRI contrasts of the same anatomy contain redundant information, one contrast can guide the reconstruction of an undersampled subsequent contrast. To this end, several end-to-end learning-based guided reconstruction methods have been proposed. However, a key challenge is the requirement of large paired training datasets comprising raw data and aligned reference images. We propose a modular two-stage approach addressing this issue, additionally providing an explanatory framework for the multi-contrast problem based on the shared and non-shared generative factors underlying two given contrasts. A content/style model of two-contrast image data is learned from a largely unpaired image-domain dataset and is subsequently applied as a plug-and-play operator in iterative reconstruction. The disentanglement of content and style allows explicit representation of contrast-independent and contrast-specific factors. Consequently, incorporating prior information into the reconstruction reduces to a simple replacement of the aliased content of the reconstruction iterate with high-quality content derived from the reference scan. Combining this component with a data consistency step and introducing a general corrective process for the content yields an iterative scheme. We name this novel approach PnP-CoSMo. Various aspects like interpretability and convergence are explored via simulations. Furthermore, its practicality is demonstrated on the NYU fastMRI DICOM dataset, showing improved generalizability compared to end-to-end methods, and on two in-house multi-coil raw datasets, offering up to 32.6% more acceleration over learning-based non-guided reconstruction for a given SSIM. In a small radiological task, PnP-CoSMo allowed 33.3% more acceleration over clinical reconstruction at diagnostic quality.
>
---
#### [replaced 062] Enhancing Target-unspecific Tasks through a Features Matrix
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03414v4](http://arxiv.org/pdf/2505.03414v4)**

> **作者:** Fangming Cui; Yonggang Zhang; Xuan Wang; Xinmei Tian; Jun Yu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Recent developments in prompt learning of large Vision-Language Models (VLMs) have significantly improved performance in target-specific tasks. However, these prompting methods often struggle to tackle the target-unspecific or generalizable tasks effectively. It may be attributed to the fact that overfitting training causes the model to forget its general knowledge. The general knowledge has a strong promotion on target-unspecific tasks. To alleviate this issue, we propose a novel Features Matrix (FM) approach designed to enhance these models on target-unspecific tasks. Our method extracts and leverages general knowledge, shaping a Features Matrix (FM). Specifically, the FM captures the semantics of diverse inputs from a deep and fine perspective, preserving essential general knowledge, which mitigates the risk of overfitting. Representative evaluations demonstrate that: 1) the FM is compatible with existing frameworks as a generic and flexible module, and 2) the FM significantly showcases its effectiveness in enhancing target-unspecific tasks (base-to-novel generalization, domain generalization, and cross-dataset generalization), achieving state-of-the-art performance.
>
---
#### [replaced 063] Semantics-aware Test-time Adaptation for 3D Human Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.10724v2](http://arxiv.org/pdf/2502.10724v2)**

> **作者:** Qiuxia Lin; Rongyu Chen; Kerui Gu; Angela Yao
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** This work highlights a semantics misalignment in 3D human pose estimation. For the task of test-time adaptation, the misalignment manifests as overly smoothed and unguided predictions. The smoothing settles predictions towards some average pose. Furthermore, when there are occlusions or truncations, the adaptation becomes fully unguided. To this end, we pioneer the integration of a semantics-aware motion prior for the test-time adaptation of 3D pose estimation. We leverage video understanding and a well-structured motion-text space to adapt the model motion prediction to adhere to video semantics during test time. Additionally, we incorporate a missing 2D pose completion based on the motion-text similarity. The pose completion strengthens the motion prior's guidance for occlusions and truncations. Our method significantly improves state-of-the-art 3D human pose estimation TTA techniques, with more than 12% decrease in PA-MPJPE on 3DPW and 3DHP.
>
---
#### [replaced 064] Mouse Lockbox Dataset: Behavior Recognition for Mice Solving Lockboxes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15408v2](http://arxiv.org/pdf/2505.15408v2)**

> **作者:** Patrik Reiske; Marcus N. Boon; Niek Andresen; Sole Traverso; Katharina Hohlbaum; Lars Lewejohann; Christa Thöne-Reineke; Olaf Hellwich; Henning Sprekeler
>
> **摘要:** Machine learning and computer vision methods have a major impact on the study of natural animal behavior, as they enable the (semi-)automatic analysis of vast amounts of video data. Mice are the standard mammalian model system in most research fields, but the datasets available today to refine such methods focus either on simple or social behaviors. In this work, we present a video dataset of individual mice solving complex mechanical puzzles, so-called lockboxes. The more than 110 hours of total playtime show their behavior recorded from three different perspectives. As a benchmark for frame-level action classification methods, we provide human-annotated labels for all videos of two different mice, that equal 13% of our dataset. Our keypoint (pose) tracking-based action classification framework illustrates the challenges of automated labeling of fine-grained behaviors, such as the manipulation of objects. We hope that our work will help accelerate the advancement of automated action and behavior classification in the computational neuroscience community. Our dataset is publicly available at https://doi.org/10.14279/depositonce-23850
>
---
#### [replaced 065] HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.23907v2](http://arxiv.org/pdf/2503.23907v2)**

> **作者:** Zhichao Liao; Xiaokun Liu; Wenyu Qin; Qingyu Li; Qiulin Wang; Pengfei Wan; Di Zhang; Long Zeng; Pingfa Feng
>
> **摘要:** Image Aesthetic Assessment (IAA) is a long-standing and challenging research task. However, its subset, Human Image Aesthetic Assessment (HIAA), has been scarcely explored. To bridge this research gap, our work pioneers a holistic implementation framework tailored for HIAA. Specifically, we introduce HumanBeauty, the first dataset purpose-built for HIAA, which comprises 108k high-quality human images with manual annotations. To achieve comprehensive and fine-grained HIAA, 50K human images are manually collected through a rigorous curation process and annotated leveraging our trailblazing 12-dimensional aesthetic standard, while the remaining 58K with overall aesthetic labels are systematically filtered from public datasets. Based on the HumanBeauty database, we propose HumanAesExpert, a powerful Vision Language Model for aesthetic evaluation of human images. We innovatively design an Expert head to incorporate human knowledge of aesthetic sub-dimensions while jointly utilizing the Language Modeling (LM) and Regression heads. This approach empowers our model to achieve superior proficiency in both overall and fine-grained HIAA. Furthermore, we introduce a MetaVoter, which aggregates scores from all three heads, to effectively balance the capabilities of each head, thereby realizing improved assessment precision. Extensive experiments demonstrate that our HumanAesExpert models deliver significantly better performance in HIAA than other state-of-the-art models. Project webpage: https://humanaesexpert.github.io/HumanAesExpert/
>
---
#### [replaced 066] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18079v2](http://arxiv.org/pdf/2505.18079v2)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** V2 draft. Under review
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code will be released later.
>
---
#### [replaced 067] Flexible Sampling for Long-tailed Skin Lesion Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2204.03161v2](http://arxiv.org/pdf/2204.03161v2)**

> **作者:** Lie Ju; Yicheng Wu; Lin Wang; Zhen Yu; Xin Zhao; Xin Wang; Paul Bonnington; Zongyuan Ge
>
> **摘要:** Most of the medical tasks naturally exhibit a long-tailed distribution due to the complex patient-level conditions and the existence of rare diseases. Existing long-tailed learning methods usually treat each class equally to re-balance the long-tailed distribution. However, considering that some challenging classes may present diverse intra-class distributions, re-balancing all classes equally may lead to a significant performance drop. To address this, in this paper, we propose a curriculum learning-based framework called Flexible Sampling for the long-tailed skin lesion classification task. Specifically, we initially sample a subset of training data as anchor points based on the individual class prototypes. Then, these anchor points are used to pre-train an inference model to evaluate the per-class learning difficulty. Finally, we use a curriculum sampling module to dynamically query new samples from the rest training samples with the learning difficulty-aware sampling probability. We evaluated our model against several state-of-the-art methods on the ISIC dataset. The results with two long-tailed settings have demonstrated the superiority of our proposed training strategy, which achieves a new benchmark for long-tailed skin lesion classification.
>
---
#### [replaced 068] The Impact of the Single-Label Assumption in Image Recognition Benchmarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.18409v2](http://arxiv.org/pdf/2412.18409v2)**

> **作者:** Esla Timothy Anzaku; Seyed Amir Mousavi; Arnout Van Messem; Wesley De Neve
>
> **备注:** 34 pages, 7 figures
>
> **摘要:** Deep neural networks (DNNs) are typically evaluated under the assumption that each image has a single correct label. However, many images in benchmarks like ImageNet contain multiple valid labels, creating a mismatch between evaluation protocols and the actual complexity of visual data. This mismatch can penalize DNNs for predicting correct but unannotated labels, which may partly explain reported accuracy drops, such as the widely cited 11 to 14 percent top-1 accuracy decline on ImageNetV2, a replication test set for ImageNet. This raises the question: do such drops reflect genuine generalization failures or artifacts of restrictive evaluation metrics? We rigorously assess the impact of multi-label characteristics on reported accuracy gaps. To evaluate the multi-label prediction capability (MLPC) of single-label-trained models, we introduce a variable top-$k$ evaluation, where $k$ matches the number of valid labels per image. Applied to 315 ImageNet-trained models, our analyses demonstrate that conventional top-1 accuracy disproportionately penalizes valid but secondary predictions. We also propose Aggregate Subgroup Model Accuracy (ASMA) to better capture multi-label performance across model subgroups. Our results reveal wide variability in MLPC, with some models consistently ranking multiple correct labels higher. Under this evaluation, the perceived gap between ImageNet and ImageNetV2 narrows substantially. To further isolate multi-label recognition performance from contextual cues, we introduce PatchML, a synthetic dataset containing systematically combined object patches. PatchML demonstrates that many models trained with single-label supervision nonetheless recognize multiple objects. Altogether, these findings highlight limitations in single-label evaluation and reveal that modern DNNs have stronger multi-label capabilities than standard metrics suggest.
>
---
#### [replaced 069] ReLearn: Unlearning via Learning for Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.11190v3](http://arxiv.org/pdf/2502.11190v3)**

> **作者:** Haoming Xu; Ningyuan Zhao; Liming Yang; Sendong Zhao; Shumin Deng; Mengru Wang; Bryan Hooi; Nay Oo; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Current unlearning methods for large language models usually rely on reverse optimization to reduce target token probabilities. However, this paradigm disrupts the subsequent tokens prediction, degrading model performance and linguistic coherence. Moreover, existing evaluation metrics overemphasize contextual forgetting while inadequately assessing response fluency and relevance. To address these challenges, we propose ReLearn, a data augmentation and fine-tuning pipeline for effective unlearning, along with a comprehensive evaluation framework. This framework introduces Knowledge Forgetting Rate (KFR) and Knowledge Retention Rate (KRR) to measure knowledge-level preservation, and Linguistic Score (LS) to evaluate generation quality. Our experiments show that ReLearn successfully achieves targeted forgetting while preserving high-quality output. Through mechanistic analysis, we further demonstrate how reverse optimization disrupts coherent text generation, while ReLearn preserves this essential capability. Code is available at https://github.com/zjunlp/unlearn.
>
---
#### [replaced 070] SLoRD: Structural Low-Rank Descriptors for Shape Consistency in Vertebrae Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.08555v3](http://arxiv.org/pdf/2407.08555v3)**

> **作者:** Xin You; Yixin Lou; Minghui Zhang; Jie Yang; Yun Gu
>
> **备注:** JBHI accepted
>
> **摘要:** Automatic and precise multi-class vertebrae segmentation from CT images is crucial for various clinical applications. However, due to similar appearances between adjacent vertebrae and the existence of various pathologies, existing single-stage and multi-stage methods suffer from imprecise vertebrae segmentation. Essentially, these methods fail to explicitly impose both contour precision and intra-vertebrae voxel consistency constraints synchronously, resulting in the intra-vertebrae segmentation inconsistency, which refers to multiple label predictions inside a singular vertebra. In this work, we intend to label complete binary masks with sequential indices to address that challenge. Specifically, a contour generation network is proposed based on Structural Low-Rank Descriptors for shape consistency, termed SLoRD. For a structural representation of vertebral contours, we adopt the spherical coordinate system and devise the spherical centroid to calculate contour descriptors. Due to vertebrae's similar appearances, basic contour descriptors can be acquired offline to restore original contours. Therefore, SLoRD leverages these contour priors and explicit shape constraints to facilitate regressed contour points close to vertebral surfaces. Quantitative and qualitative evaluations on VerSe 2019 and 2020 demonstrate the superior performance of our framework over other single-stage and multi-stage state-of-the-art (SOTA) methods. Further, SLoRD is a plug-and-play framework to refine the segmentation inconsistency existing in coarse predictions from other approaches. Source codes are available.
>
---
#### [replaced 071] Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.06025v3](http://arxiv.org/pdf/2410.06025v3)**

> **作者:** Michael Kirchhof; James Thornton; Louis Béthune; Pierre Ablin; Eugene Ndiaye; Marco Cuturi
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** The adoption of text-to-image diffusion models raises concerns over reliability, drawing scrutiny under the lens of various metrics like calibration, fairness, or compute efficiency. We focus in this work on two issues that arise when deploying these models: a lack of diversity when prompting images, and a tendency to recreate images from the training set. To solve both problems, we propose a method that coaxes the sampled trajectories of pretrained diffusion models to land on images that fall outside of a reference set. We achieve this by adding repellency terms to the diffusion SDE throughout the generation trajectory, which are triggered whenever the path is expected to land too closely to an image in the shielded reference set. Our method is sparse in the sense that these repellency terms are zero and inactive most of the time, and even more so towards the end of the generation trajectory. Our method, named SPELL for sparse repellency, can be used either with a static reference set that contains protected images, or dynamically, by updating the set at each timestep with the expected images concurrently generated within a batch, and with the images of previously generated batches. We show that adding SPELL to popular diffusion models improves their diversity while impacting their FID only marginally, and performs comparatively better than other recent training-free diversity methods. We also demonstrate how SPELL can ensure a shielded generation away from a very large set of protected images by considering all 1.2M images from ImageNet as the protected set.
>
---
#### [replaced 072] Imitating Radiological Scrolling: A Global-Local Attention Model for 3D Chest CT Volumes Multi-Label Anomaly Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20652v3](http://arxiv.org/pdf/2503.20652v3)**

> **作者:** Theo Di Piazza; Carole Lazarus; Olivier Nempont; Loic Boussel
>
> **备注:** 13 pages, 4 figures. Accepted for MIDL 2025
>
> **摘要:** The rapid increase in the number of Computed Tomography (CT) scan examinations has created an urgent need for automated tools, such as organ segmentation, anomaly classification, and report generation, to assist radiologists with their growing workload. Multi-label classification of Three-Dimensional (3D) CT scans is a challenging task due to the volumetric nature of the data and the variety of anomalies to be detected. Existing deep learning methods based on Convolutional Neural Networks (CNNs) struggle to capture long-range dependencies effectively, while Vision Transformers require extensive pre-training, posing challenges for practical use. Additionally, these existing methods do not explicitly model the radiologist's navigational behavior while scrolling through CT scan slices, which requires both global context understanding and local detail awareness. In this study, we present CT-Scroll, a novel global-local attention model specifically designed to emulate the scrolling behavior of radiologists during the analysis of 3D CT scans. Our approach is evaluated on two public datasets, demonstrating its efficacy through comprehensive experiments and an ablation study that highlights the contribution of each model component.
>
---
#### [replaced 073] ControlTac: Force- and Position-Controlled Tactile Data Augmentation with a Single Reference Image
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20498v2](http://arxiv.org/pdf/2505.20498v2)**

> **作者:** Dongyu Luo; Kelin Yu; Amir-Hossein Shahidzadeh; Cornelia Fermüller; Yiannis Aloimonos; Ruohan Gao
>
> **备注:** 22 pages, 11 figures, 7 tables
>
> **摘要:** Vision-based tactile sensing has been widely used in perception, reconstruction, and robotic manipulation. However, collecting large-scale tactile data remains costly due to the localized nature of sensor-object interactions and inconsistencies across sensor instances. Existing approaches to scaling tactile data, such as simulation and free-form tactile generation, often suffer from unrealistic output and poor transferability to downstream tasks. To address this, we propose ControlTac, a two-stage controllable framework that generates realistic tactile images conditioned on a single reference tactile image, contact force, and contact position. With those physical priors as control input, ControlTac generates physically plausible and varied tactile images that can be used for effective data augmentation. Through experiments on three downstream tasks, we demonstrate that ControlTac can effectively augment tactile datasets and lead to consistent gains. Our three real-world experiments further validate the practical utility of our approach. Project page: https://dongyuluo.github.io/controltac.
>
---
#### [replaced 074] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; I.2.7; I.2.10; I.4.9**

- **链接: [http://arxiv.org/pdf/2503.07265v2](http://arxiv.org/pdf/2503.07265v2)**

> **作者:** Yuwei Niu; Munan Ning; Mengren Zheng; Weiyang Jin; Bin Lin; Peng Jin; Jiaqi Liao; Chaoran Feng; Kunpeng Ning; Bin Zhu; Li Yuan
>
> **备注:** Code, data and leaderboard: https://github.com/PKU-YuanGroup/WISE
>
> **摘要:** Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text to image generation. To address this challenge, we propose $\textbf{WISE}$, the first benchmark specifically designed for $\textbf{W}$orld Knowledge-$\textbf{I}$nformed $\textbf{S}$emantic $\textbf{E}$valuation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce $\textbf{WiScore}$, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at https://github.com/PKU-YuanGroup/WISE.
>
---
#### [replaced 075] CHATS: Combining Human-Aligned Optimization and Test-Time Sampling for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12579v2](http://arxiv.org/pdf/2502.12579v2)**

> **作者:** Minghao Fu; Guo-Hua Wang; Liangfu Cao; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** ICML 2025. The code is publicly available at https://github.com/AIDC-AI/CHATS
>
> **摘要:** Diffusion models have emerged as a dominant approach for text-to-image generation. Key components such as the human preference alignment and classifier-free guidance play a crucial role in ensuring generation quality. However, their independent application in current text-to-image models continues to face significant challenges in achieving strong text-image alignment, high generation quality, and consistency with human aesthetic standards. In this work, we for the first time, explore facilitating the collaboration of human performance alignment and test-time sampling to unlock the potential of text-to-image models. Consequently, we introduce CHATS (Combining Human-Aligned optimization and Test-time Sampling), a novel generative framework that separately models the preferred and dispreferred distributions and employs a proxy-prompt-based sampling strategy to utilize the useful information contained in both distributions. We observe that CHATS exhibits exceptional data efficiency, achieving strong performance with only a small, high-quality funetuning dataset. Extensive experiments demonstrate that CHATS surpasses traditional preference alignment methods, setting new state-of-the-art across various standard benchmarks.
>
---
#### [replaced 076] Progressive Language-guided Visual Learning for Multi-Task Visual Grounding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.16145v2](http://arxiv.org/pdf/2504.16145v2)**

> **作者:** Jingchao Wang; Hong Wang; Wenlong Zhang; Kunhua Ji; Dingjiang Huang; Yefeng Zheng
>
> **摘要:** Multi-task visual grounding (MTVG) includes two sub-tasks, i.e., Referring Expression Comprehension (REC) and Referring Expression Segmentation (RES). The existing representative approaches generally follow the research pipeline which mainly consists of three core procedures, including independent feature extraction for visual and linguistic modalities, respectively, cross-modal interaction module, and independent prediction heads for different sub-tasks. Albeit achieving remarkable performance, this research line has two limitations: 1) The linguistic content has not been fully injected into the entire visual backbone for boosting more effective visual feature extraction and it needs an extra cross-modal interaction module; 2) The relationship between REC and RES tasks is not effectively exploited to help the collaborative prediction for more accurate output. To deal with these problems, in this paper, we propose a Progressive Language-guided Visual Learning framework for multi-task visual grounding, called PLVL, which not only finely mine the inherent feature expression of the visual modality itself but also progressively inject the language information to help learn linguistic-related visual features. In this manner, our PLVL does not need additional cross-modal fusion module while fully introducing the language guidance. Furthermore, we analyze that the localization center for REC would help identify the to-be-segmented object region for RES to some extent. Inspired by this investigation, we design a multi-task head to accomplish collaborative predictions for these two sub-tasks. Extensive experiments conducted on several benchmark datasets comprehensively substantiate that our PLVL obviously outperforms the representative methods in both REC and RES tasks. https://github.com/jcwang0602/PLVL
>
---
#### [replaced 077] EVM-Fusion: An Explainable Vision Mamba Architecture with Neural Algorithmic Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17367v3](http://arxiv.org/pdf/2505.17367v3)**

> **作者:** Zichuan Yang
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Medical image classification is critical for clinical decision-making, yet demands for accuracy, interpretability, and generalizability remain challenging. This paper introduces EVM-Fusion, an Explainable Vision Mamba architecture featuring a novel Neural Algorithmic Fusion (NAF) mechanism for multi-organ medical image classification. EVM-Fusion leverages a multipath design, where DenseNet and U-Net based pathways, enhanced by Vision Mamba (Vim) modules, operate in parallel with a traditional feature pathway. These diverse features are dynamically integrated via a two-stage fusion process: cross-modal attention followed by the iterative NAF block, which learns an adaptive fusion algorithm. Intrinsic explainability is embedded through path-specific spatial attention, Vim {\Delta}-value maps, traditional feature SE-attention, and cross-modal attention weights. Experiments on a diverse 9-class multi-organ medical image dataset demonstrate EVM-Fusion's strong classification performance, achieving 99.75% test accuracy and provide multi-faceted insights into its decision-making process, highlighting its potential for trustworthy AI in medical diagnostics.
>
---
#### [replaced 078] Fast 3D point clouds retrieval for Large-scale 3D Place Recognition
- **分类: cs.CV; cs.IR; 68T10, 68T45; I.5.4; I.2.10**

- **链接: [http://arxiv.org/pdf/2502.21067v2](http://arxiv.org/pdf/2502.21067v2)**

> **作者:** Chahine-Nicolas Zede; Laurent Carrafa; Valérie Gouet-Brunet
>
> **备注:** 8 pages, 1 figures
>
> **摘要:** Retrieval in 3D point clouds is a challenging task that consists in retrieving the most similar point clouds to a given query within a reference of 3D points. Current methods focus on comparing descriptors of point clouds in order to identify similar ones. Due to the complexity of this latter step, here we focus on the acceleration of the retrieval by adapting the Differentiable Search Index (DSI), a transformer-based approach initially designed for text information retrieval, for 3D point clouds retrieval. Our approach generates 1D identifiers based on the point descriptors, enabling direct retrieval in constant time. To adapt DSI to 3D data, we integrate Vision Transformers to map descriptors to these identifiers while incorporating positional and semantic encoding. The approach is evaluated for place recognition on a public benchmark comparing its retrieval capabilities against state-of-the-art methods, in terms of quality and speed of returned point clouds.
>
---
#### [replaced 079] Sparse R-CNN OBB: Ship Target Detection in SAR Images Based on Oriented Sparse Proposals
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.07973v2](http://arxiv.org/pdf/2409.07973v2)**

> **作者:** Kamirul Kamirul; Odysseas Pappas; Alin Achim
>
> **备注:** Manuscript has been accepted for publication at the 2025 IEEE International Conference on Image Processing (ICIP)
>
> **摘要:** We present Sparse R-CNN OBB, a novel framework for the detection of oriented objects in SAR images leveraging sparse learnable proposals. The Sparse R-CNN OBB has streamlined architecture and ease of training as it utilizes a sparse set of 300 proposals instead of training a proposals generator on hundreds of thousands of anchors. To the best of our knowledge, Sparse R-CNN OBB is the first to adopt the concept of sparse learnable proposals for the detection of oriented objects, as well as for the detection of ships in Synthetic Aperture Radar (SAR) images. The detection head of the baseline model, Sparse R-CNN, is re-designed to enable the model to capture object orientation. We train the model on RSDD-SAR dataset and provide a performance comparison to state-of-the-art models. Experimental results show that Sparse R-CNN OBB achieves outstanding performance, surpassing most models on both inshore and offshore scenarios. The code is available at: www.github.com/ka-mirul/Sparse-R-CNN-OBB.
>
---
#### [replaced 080] Hypo3D: Exploring Hypothetical Reasoning in 3D
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00954v3](http://arxiv.org/pdf/2502.00954v3)**

> **作者:** Ye Mao; Weixun Luo; Junpeng Jing; Anlan Qiu; Krystian Mikolajczyk
>
> **备注:** 24 pages, 15 figures, 15 tables
>
> **摘要:** The rise of vision-language foundation models marks an advancement in bridging the gap between human and machine capabilities in 3D scene reasoning. Existing 3D reasoning benchmarks assume real-time scene accessibility, which is impractical due to the high cost of frequent scene updates. To this end, we introduce Hypothetical 3D Reasoning, namely Hypo3D, a benchmark designed to evaluate models' ability to reason without access to real-time scene data. Models need to imagine the scene state based on a provided change description before reasoning. Hypo3D is formulated as a 3D Visual Question Answering (VQA) benchmark, comprising 7,727 context changes across 700 indoor scenes, resulting in 14,885 question-answer pairs. An anchor-based world frame is established for all scenes, ensuring consistent reference to a global frame for directional terms in context changes and QAs. Extensive experiments show that state-of-the-art foundation models struggle to reason in hypothetically changed scenes. This reveals a substantial performance gap compared to humans, particularly in scenarios involving movement changes and directional reasoning. Even when the context change is irrelevant to the question, models often incorrectly adjust their answers. Project website: https://matchlab-imperial.github.io/Hypo3D/
>
---
#### [replaced 081] CityGo: Lightweight Urban Modeling and Rendering with Proxy Buildings and Residual Gaussians
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21041v2](http://arxiv.org/pdf/2505.21041v2)**

> **作者:** Weihang Liu; Yuhui Zhong; Yuke Li; Xi Chen; Jiadi Cui; Honglong Zhang; Lan Xu; Xin Lou; Yujiao Shi; Jingyi Yu; Yingliang Zhang
>
> **摘要:** Accurate and efficient modeling of large-scale urban scenes is critical for applications such as AR navigation, UAV based inspection, and smart city digital twins. While aerial imagery offers broad coverage and complements limitations of ground-based data, reconstructing city-scale environments from such views remains challenging due to occlusions, incomplete geometry, and high memory demands. Recent advances like 3D Gaussian Splatting (3DGS) improve scalability and visual quality but remain limited by dense primitive usage, long training times, and poor suit ability for edge devices. We propose CityGo, a hybrid framework that combines textured proxy geometry with residual and surrounding 3D Gaussians for lightweight, photorealistic rendering of urban scenes from aerial perspectives. Our approach first extracts compact building proxy meshes from MVS point clouds, then uses zero order SH Gaussians to generate occlusion-free textures via image-based rendering and back-projection. To capture high-frequency details, we introduce residual Gaussians placed based on proxy-photo discrepancies and guided by depth priors. Broader urban context is represented by surrounding Gaussians, with importance-aware downsampling applied to non-critical regions to reduce redundancy. A tailored optimization strategy jointly refines proxy textures and Gaussian parameters, enabling real-time rendering of complex urban scenes on mobile GPUs with significantly reduced training and memory requirements. Extensive experiments on real-world aerial datasets demonstrate that our hybrid representation significantly reduces training time, achieving on average 1.4x speedup, while delivering comparable visual fidelity to pure 3D Gaussian Splatting approaches. Furthermore, CityGo enables real-time rendering of large-scale urban scenes on mobile consumer GPUs, with substantially reduced memory usage and energy consumption.
>
---
#### [replaced 082] Beyond External Monitors: Enhancing Transparency of Large Language Models for Easier Monitoring
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05242v2](http://arxiv.org/pdf/2502.05242v2)**

> **作者:** Guanxu Chen; Dongrui Liu; Tao Luo; Lijie Hu; Jing Shao
>
> **备注:** 25 pages,6 figures,13 tables
>
> **摘要:** Large language models (LLMs) are becoming increasingly capable, but the mechanisms of their thinking and decision-making process remain unclear. Chain-of-thoughts (CoTs) have been commonly utilized to monitor LLMs, but this strategy fails to accurately reflect LLMs' thinking process. Techniques based on LLMs' hidden representations provide an inner perspective to monitor their latent thinking. However, previous methods only try to develop external monitors instead of making LLMs themselves easier to monitor. In this paper, we propose a novel method TELLME, improving the transparency of LLMs and helping monitors identify unsuitable and sensitive behaviors. Furthermore, we showcase the applications of TELLME on trustworthiness tasks (\eg, safety risks monitoring tasks and detoxification tasks), where LLMs achieve consistent improvement in transparency and task performance. More crucially, we theoretically analyze the improvement of TELLME on LLMs' generalization ability through optimal transport theory.
>
---
#### [replaced 083] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03558v2](http://arxiv.org/pdf/2412.03558v2)**

> **作者:** Zehuan Huang; Yuan-Chen Guo; Xingqiao An; Yunhan Yang; Yangguang Li; Zi-Xin Zou; Ding Liang; Xihui Liu; Yan-Pei Cao; Lu Sheng
>
> **备注:** Project page: https://huanngzh.github.io/MIDI-Page/
>
> **摘要:** This paper introduces MIDI, a novel paradigm for compositional 3D scene generation from a single image. Unlike existing methods that rely on reconstruction or retrieval techniques or recent approaches that employ multi-stage object-by-object generation, MIDI extends pre-trained image-to-3D object generation models to multi-instance diffusion models, enabling the simultaneous generation of multiple 3D instances with accurate spatial relationships and high generalizability. At its core, MIDI incorporates a novel multi-instance attention mechanism, that effectively captures inter-object interactions and spatial coherence directly within the generation process, without the need for complex multi-step processes. The method utilizes partial object images and global scene context as inputs, directly modeling object completion during 3D generation. During training, we effectively supervise the interactions between 3D instances using a limited amount of scene-level data, while incorporating single-object data for regularization, thereby maintaining the pre-trained generalization ability. MIDI demonstrates state-of-the-art performance in image-to-scene generation, validated through evaluations on synthetic data, real-world scene data, and stylized scene images generated by text-to-image diffusion models.
>
---
#### [replaced 084] SoPo: Text-to-Motion Generation Using Semi-Online Preference Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05095v2](http://arxiv.org/pdf/2412.05095v2)**

> **作者:** Xiaofeng Tan; Hongsong Wang; Xin Geng; Pan Zhou
>
> **摘要:** Text-to-motion generation is essential for advancing the creative industry but often presents challenges in producing consistent, realistic motions. To address this, we focus on fine-tuning text-to-motion models to consistently favor high-quality, human-preferred motions, a critical yet largely unexplored problem. In this work, we theoretically investigate the DPO under both online and offline settings, and reveal their respective limitation: overfitting in offline DPO, and biased sampling in online DPO. Building on our theoretical insights, we introduce Semi-online Preference Optimization (SoPo), a DPO-based method for training text-to-motion models using "semi-online" data pair, consisting of unpreferred motion from online distribution and preferred motion in offline datasets. This method leverages both online and offline DPO, allowing each to compensate for the other's limitations. Extensive experiments demonstrate that SoPo outperforms other preference alignment methods, with an MM-Dist of 3.25% (vs e.g. 0.76% of MoDiPO) on the MLD model, 2.91% (vs e.g. 0.66% of MoDiPO) on MDM model, respectively. Additionally, the MLD model fine-tuned by our SoPo surpasses the SoTA model in terms of R-precision and MM Dist. Visualization results also show the efficacy of our SoPo in preference alignment. Project page: https://xiaofeng-tan.github.io/projects/SoPo/ .
>
---
#### [replaced 085] OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20292v3](http://arxiv.org/pdf/2505.20292v3)**

> **作者:** Shenghai Yuan; Xianyi He; Yufan Deng; Yang Ye; Jinfa Huang; Bin Lin; Jiebo Luo; Li Yuan
>
> **备注:** Code and Dataset: https://github.com/PKU-YuanGroup/OpenS2V-Nexus
>
> **摘要:** Subject-to-Video (S2V) generation aims to create videos that faithfully incorporate reference content, providing enhanced flexibility in the production of videos. To establish the infrastructure for S2V generation, we propose OpenS2V-Nexus, consisting of (i) OpenS2V-Eval, a fine-grained benchmark, and (ii) OpenS2V-5M, a million-scale dataset. In contrast to existing S2V benchmarks inherited from VBench that focus on global and coarse-grained assessment of generated videos, OpenS2V-Eval focuses on the model's ability to generate subject-consistent videos with natural subject appearance and identity fidelity. For these purposes, OpenS2V-Eval introduces 180 prompts from seven major categories of S2V, which incorporate both real and synthetic test data. Furthermore, to accurately align human preferences with S2V benchmarks, we propose three automatic metrics, NexusScore, NaturalScore and GmeScore, to separately quantify subject consistency, naturalness, and text relevance in generated videos. Building on this, we conduct a comprehensive evaluation of 16 representative S2V models, highlighting their strengths and weaknesses across different content. Moreover, we create the first open-source large-scale S2V generation dataset OpenS2V-5M, which consists of five million high-quality 720P subject-text-video triples. Specifically, we ensure subject-information diversity in our dataset by (1) segmenting subjects and building pairing information via cross-video associations and (2) prompting GPT-Image-1 on raw frames to synthesize multi-view representations. Through OpenS2V-Nexus, we deliver a robust infrastructure to accelerate future S2V generation research.
>
---
#### [replaced 086] Advancing high-fidelity 3D and Texture Generation with 2.5D latents
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21050v2](http://arxiv.org/pdf/2505.21050v2)**

> **作者:** Xin Yang; Jiantao Lin; Yingjie Xu; Haodong Li; Yingcong Chen
>
> **摘要:** Despite the availability of large-scale 3D datasets and advancements in 3D generative models, the complexity and uneven quality of 3D geometry and texture data continue to hinder the performance of 3D generation techniques. In most existing approaches, 3D geometry and texture are generated in separate stages using different models and non-unified representations, frequently leading to unsatisfactory coherence between geometry and texture. To address these challenges, we propose a novel framework for joint generation of 3D geometry and texture. Specifically, we focus in generate a versatile 2.5D representations that can be seamlessly transformed between 2D and 3D. Our approach begins by integrating multiview RGB, normal, and coordinate images into a unified representation, termed as 2.5D latents. Next, we adapt pre-trained 2D foundation models for high-fidelity 2.5D generation, utilizing both text and image conditions. Finally, we introduce a lightweight 2.5D-to-3D refiner-decoder framework that efficiently generates detailed 3D representations from 2.5D images. Extensive experiments demonstrate that our model not only excels in generating high-quality 3D objects with coherent structure and color from text and image inputs but also significantly outperforms existing methods in geometry-conditioned texture generation.
>
---
#### [replaced 087] AgriFM: A Multi-source Temporal Remote Sensing Foundation Model for Crop Mapping
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21357v2](http://arxiv.org/pdf/2505.21357v2)**

> **作者:** Wenyuan Li; Shunlin Liang; Keyan Chen; Yongzhe Chen; Han Ma; Jianglei Xu; Yichuan Ma; Shikang Guan; Husheng Fang; Zhenwei Shi
>
> **摘要:** Accurate crop mapping fundamentally relies on modeling multi-scale spatiotemporal patterns, where spatial scales range from individual field textures to landscape-level context, and temporal scales capture both short-term phenological transitions and full growing-season dynamics. Transformer-based remote sensing foundation models (RSFMs) offer promising potential for crop mapping due to their innate ability for unified spatiotemporal processing. However, current RSFMs remain suboptimal for crop mapping: they either employ fixed spatiotemporal windows that ignore the multi-scale nature of crop systems or completely disregard temporal information by focusing solely on spatial patterns. To bridge these gaps, we present AgriFM, a multi-source remote sensing foundation model specifically designed for agricultural crop mapping. Our approach begins by establishing the necessity of simultaneous hierarchical spatiotemporal feature extraction, leading to the development of a modified Video Swin Transformer architecture where temporal down-sampling is synchronized with spatial scaling operations. This modified backbone enables efficient unified processing of long time-series satellite inputs. AgriFM leverages temporally rich data streams from three satellite sources including MODIS, Landsat-8/9 and Sentinel-2, and is pre-trained on a global representative dataset comprising over 25 million image samples supervised by land cover products. The resulting framework incorporates a versatile decoder architecture that dynamically fuses these learned spatiotemporal representations, supporting diverse downstream tasks. Comprehensive evaluations demonstrate AgriFM's superior performance over conventional deep learning approaches and state-of-the-art general-purpose RSFMs across all downstream tasks. Codes will be available at https://github.com/flyakon/AgriFM.
>
---
#### [replaced 088] Automating tumor-infiltrating lymphocyte assessment in breast cancer histopathology images using QuPath: a transparent and accessible machine learning pipeline
- **分类: q-bio.QM; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16979v2](http://arxiv.org/pdf/2504.16979v2)**

> **作者:** Masoud Tafavvoghi; Lars Ailo Bongo; André Berli Delgado; Nikita Shvetsov; Anders Sildnes; Line Moi; Lill-Tove Rasmussen Busund; Kajsa Møllersen
>
> **备注:** 16 Pages, 9 Figures, 3 tables
>
> **摘要:** In this study, we built an end-to-end tumor-infiltrating lymphocytes (TILs) assessment pipeline within QuPath, demonstrating the potential of easily accessible tools to perform complex tasks in a fully automatic fashion. First, we trained a pixel classifier to segment tumor, tumor-associated stroma, and other tissue compartments in breast cancer H&E-stained whole-slide images (WSI) to isolate tumor-associated stroma for subsequent analysis. Next, we applied a pre-trained StarDist deep learning model in QuPath for cell detection and used the extracted cell features to train a binary classifier distinguishing TILs from other cells. To evaluate our TILs assessment pipeline, we calculated the TIL density in each WSI and categorized them as low, medium, or high TIL levels. Our pipeline was evaluated against pathologist-assigned TIL scores, achieving a Cohen's kappa of 0.71 on the external test set, corroborating previous research findings. These results confirm that existing software can offer a practical solution for the assessment of TILs in H&E-stained WSIs of breast cancer.
>
---
#### [replaced 089] Interpreting CLIP with Hierarchical Sparse Autoencoders
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.20578v2](http://arxiv.org/pdf/2502.20578v2)**

> **作者:** Vladimir Zaigrajew; Hubert Baniecki; Przemyslaw Biecek
>
> **摘要:** Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing large-scale vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce Matryoshka SAE (MSAE), a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a new state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving 0.99 cosine similarity and less than 0.1 fraction of variance unexplained while maintaining ~80% sparsity. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA. We make the codebase available at https://github.com/WolodjaZ/MSAE.
>
---
#### [replaced 090] CreatiDesign: A Unified Multi-Conditional Diffusion Transformer for Creative Graphic Design
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19114v2](http://arxiv.org/pdf/2505.19114v2)**

> **作者:** Hui Zhang; Dexiang Hong; Maoke Yang; Yutao Cheng; Zhao Zhang; Jie Shao; Xinglong Wu; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Graphic design plays a vital role in visual communication across advertising, marketing, and multimedia entertainment. Prior work has explored automated graphic design generation using diffusion models, aiming to streamline creative workflows and democratize design capabilities. However, complex graphic design scenarios require accurately adhering to design intent specified by multiple heterogeneous user-provided elements (\eg images, layouts, and texts), which pose multi-condition control challenges for existing methods. Specifically, previous single-condition control models demonstrate effectiveness only within their specialized domains but fail to generalize to other conditions, while existing multi-condition methods often lack fine-grained control over each sub-condition and compromise overall compositional harmony. To address these limitations, we introduce CreatiDesign, a systematic solution for automated graphic design covering both model architecture and dataset construction. First, we design a unified multi-condition driven architecture that enables flexible and precise integration of heterogeneous design elements with minimal architectural modifications to the base diffusion model. Furthermore, to ensure that each condition precisely controls its designated image region and to avoid interference between conditions, we propose a multimodal attention mask mechanism. Additionally, we develop a fully automated pipeline for constructing graphic design datasets, and introduce a new dataset with 400K samples featuring multi-condition annotations, along with a comprehensive benchmark. Experimental results show that CreatiDesign outperforms existing models by a clear margin in faithfully adhering to user intent.
>
---
#### [replaced 091] Galileo: Learning Global & Local Features of Many Remote Sensing Modalities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09356v2](http://arxiv.org/pdf/2502.09356v2)**

> **作者:** Gabriel Tseng; Anthony Fuller; Marlena Reil; Henry Herzog; Patrick Beukema; Favyen Bastani; James R. Green; Evan Shelhamer; Hannah Kerner; David Rolnick
>
> **摘要:** We introduce a highly multimodal transformer to represent many remote sensing modalities - multispectral optical, synthetic aperture radar, elevation, weather, pseudo-labels, and more - across space and time. These inputs are useful for diverse remote sensing tasks, such as crop mapping and flood detection. However, learning shared representations of remote sensing data is challenging, given the diversity of relevant data modalities, and because objects of interest vary massively in scale, from small boats (1-2 pixels and transient) to glaciers (thousands of pixels and persistent). We present a novel self-supervised learning algorithm that extracts multi-scale features across a flexible set of input modalities through masked modeling. Our dual global and local contrastive losses differ in their targets (deep representations vs. shallow input projections) and masking strategies (structured vs. not). Our Galileo is a single generalist model that outperforms SoTA specialist models for satellite images and pixel time series across eleven benchmarks and multiple tasks.
>
---
