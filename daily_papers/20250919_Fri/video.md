# 计算机视觉 cs.CV

- **最新发布 82 篇**

- **更新 73 篇**

## 最新发布

#### [new 001] Sea-ing Through Scattered Rays: Revisiting the Image Formation Model for Realistic Underwater Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于合成水下图像生成任务，旨在解决传统模型忽略前向散射和非均匀介质的问题。提出改进的生成流程并构建BUCKET数据集，提升高浑浊环境下的图像真实性，获得82.5%的用户认可率。**

- **链接: [http://arxiv.org/pdf/2509.15011v1](http://arxiv.org/pdf/2509.15011v1)**

> **作者:** Vasiliki Ismiroglou; Malte Pedersen; Stefan H. Bengtson; Andreas Aakerberg; Thomas B. Moeslund
>
> **摘要:** In recent years, the underwater image formation model has found extensive use in the generation of synthetic underwater data. Although many approaches focus on scenes primarily affected by discoloration, they often overlook the model's ability to capture the complex, distance-dependent visibility loss present in highly turbid environments. In this work, we propose an improved synthetic data generation pipeline that includes the commonly omitted forward scattering term, while also considering a nonuniform medium. Additionally, we collected the BUCKET dataset under controlled turbidity conditions to acquire real turbid footage with the corresponding reference images. Our results demonstrate qualitative improvements over the reference model, particularly under increasing turbidity, with a selection rate of 82. 5\% by survey participants. Data and code can be accessed on the project page: vap.aau.dk/sea-ing-through-scattered-rays.
>
---
#### [new 002] Seeing 3D Through 2D Lenses: 3D Few-Shot Class-Incremental Learning via Cross-Modal Geometric Rectification
- **分类: cs.CV**

- **简介: 该论文提出CMGR框架，解决3D少样本类增量学习中几何错位与纹理偏差问题。通过跨模态几何校正与纹理增强模块，提升几何一致性与鲁棒性，适用于开放场景的3D识别任务。**

- **链接: [http://arxiv.org/pdf/2509.14958v1](http://arxiv.org/pdf/2509.14958v1)**

> **作者:** Xiang Tuo; Xu Xuemiao; Liu Bangzhen; Li Jinyi; Li Yong; He Shengfeng
>
> **备注:** ICCV2025
>
> **摘要:** The rapid growth of 3D digital content necessitates expandable recognition systems for open-world scenarios. However, existing 3D class-incremental learning methods struggle under extreme data scarcity due to geometric misalignment and texture bias. While recent approaches integrate 3D data with 2D foundation models (e.g., CLIP), they suffer from semantic blurring caused by texture-biased projections and indiscriminate fusion of geometric-textural cues, leading to unstable decision prototypes and catastrophic forgetting. To address these issues, we propose Cross-Modal Geometric Rectification (CMGR), a framework that enhances 3D geometric fidelity by leveraging CLIP's hierarchical spatial semantics. Specifically, we introduce a Structure-Aware Geometric Rectification module that hierarchically aligns 3D part structures with CLIP's intermediate spatial priors through attention-driven geometric fusion. Additionally, a Texture Amplification Module synthesizes minimal yet discriminative textures to suppress noise and reinforce cross-modal consistency. To further stabilize incremental prototypes, we employ a Base-Novel Discriminator that isolates geometric variations. Extensive experiments demonstrate that our method significantly improves 3D few-shot class-incremental learning, achieving superior geometric coherence and robustness to texture bias across cross-domain and within-domain settings.
>
---
#### [new 003] Out-of-Sight Trajectories: Tracking, Fusion, and Prediction
- **分类: cs.CV; cs.LG; cs.MA; cs.MM; cs.RO; 68T45, 68U10, 68T07, 68T40, 93C85, 93E11, 62M20, 62M10, 68U05, 94A12; F.2.2; I.2.9; I.2.10; I.4.1; I.4.8; I.4.9; I.5.4; I.3.7**

- **简介: 该论文属于轨迹预测任务，解决遮挡物体的噪声传感器数据轨迹预测问题。提出Out-of-Sight Trajectory方法，利用视觉定位去噪模块提升预测精度，并在多个数据集上取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.15219v1](http://arxiv.org/pdf/2509.15219v1)**

> **作者:** Haichao Zhang; Yi Xu; Yun Fu
>
> **摘要:** Trajectory prediction is a critical task in computer vision and autonomous systems, playing a key role in autonomous driving, robotics, surveillance, and virtual reality. Existing methods often rely on complete and noise-free observational data, overlooking the challenges associated with out-of-sight objects and the inherent noise in sensor data caused by limited camera coverage, obstructions, and the absence of ground truth for denoised trajectories. These limitations pose safety risks and hinder reliable prediction in real-world scenarios. In this extended work, we present advancements in Out-of-Sight Trajectory (OST), a novel task that predicts the noise-free visual trajectories of out-of-sight objects using noisy sensor data. Building on our previous research, we broaden the scope of Out-of-Sight Trajectory Prediction (OOSTraj) to include pedestrians and vehicles, extending its applicability to autonomous driving, robotics, surveillance, and virtual reality. Our enhanced Vision-Positioning Denoising Module leverages camera calibration to establish a vision-positioning mapping, addressing the lack of visual references, while effectively denoising noisy sensor data in an unsupervised manner. Through extensive evaluations on the Vi-Fi and JRDB datasets, our approach achieves state-of-the-art performance in both trajectory denoising and prediction, significantly surpassing previous baselines. Additionally, we introduce comparisons with traditional denoising methods, such as Kalman filtering, and adapt recent trajectory prediction models to our task, providing a comprehensive benchmark. This work represents the first initiative to integrate vision-positioning projection for denoising noisy sensor trajectories of out-of-sight agents, paving the way for future advances. The code and preprocessed datasets are available at github.com/Hai-chao-Zhang/OST
>
---
#### [new 004] Chain-of-Thought Re-ranking for Image Retrieval Tasks
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于图像检索任务，旨在解决现有方法未充分利用多模态大语言模型（MLLM）进行重排序的问题。提出Chain-of-Thought Re-Ranking（CoTRR）方法，通过设计列表级排序提示和查询分解提示，提升检索性能，实验表明其在多个任务中达到最优。**

- **链接: [http://arxiv.org/pdf/2509.14746v1](http://arxiv.org/pdf/2509.14746v1)**

> **作者:** Shangrong Wu; Yanghong Zhou; Yang Chen; Feng Zhang; P. Y. Mok
>
> **摘要:** Image retrieval remains a fundamental yet challenging problem in computer vision. While recent advances in Multimodal Large Language Models (MLLMs) have demonstrated strong reasoning capabilities, existing methods typically employ them only for evaluation, without involving them directly in the ranking process. As a result, their rich multimodal reasoning abilities remain underutilized, leading to suboptimal performance. In this paper, we propose a novel Chain-of-Thought Re-Ranking (CoTRR) method to address this issue. Specifically, we design a listwise ranking prompt that enables MLLM to directly participate in re-ranking candidate images. This ranking process is grounded in an image evaluation prompt, which assesses how well each candidate aligns with users query. By allowing MLLM to perform listwise reasoning, our method supports global comparison, consistent reasoning, and interpretable decision-making - all of which are essential for accurate image retrieval. To enable structured and fine-grained analysis, we further introduce a query deconstruction prompt, which breaks down the original query into multiple semantic components. Extensive experiments on five datasets demonstrate the effectiveness of our CoTRR method, which achieves state-of-the-art performance across three image retrieval tasks, including text-to-image retrieval (TIR), composed image retrieval (CIR) and chat-based image retrieval (Chat-IR). Our code is available at https://github.com/freshfish15/CoTRR .
>
---
#### [new 005] Fracture interactive geodesic active contours for bone segmentation
- **分类: cs.CV; cs.NA; math.NA; 68U10, 94A08**

- **简介: 该论文提出一种针对骨分割的骨折交互式测地线活动轮廓算法，解决边缘遮挡、泄漏及骨折问题。结合强度与梯度信息构建边缘检测函数，并引入距离信息引导轮廓演化，提升骨折区域分割准确性。属于医学图像分割任务。**

- **链接: [http://arxiv.org/pdf/2509.14817v1](http://arxiv.org/pdf/2509.14817v1)**

> **作者:** Liheng Wang; Licheng Zhang; Hailin Xu; Jingxin Zhao; Xiuyun Su; Jiantao Li; Miutian Tang; Weilu Gao; Chong Chen
>
> **备注:** 27 pages, 10 figures, 1 table
>
> **摘要:** For bone segmentation, the classical geodesic active contour model is usually limited by its indiscriminate feature extraction, and then struggles to handle the phenomena of edge obstruction, edge leakage and bone fracture. Thus, we propose a fracture interactive geodesic active contour algorithm tailored for bone segmentation, which can better capture bone features and perform robustly to the presence of bone fractures and soft tissues. Inspired by orthopedic knowledge, we construct a novel edge-detector function that combines the intensity and gradient norm, which guides the contour towards bone edges without being obstructed by other soft tissues and therefore reduces mis-segmentation. Furthermore, distance information, where fracture prompts can be embedded, is introduced into the contour evolution as an adaptive step size to stabilize the evolution and help the contour stop at bone edges and fractures. This embedding provides a way to interact with bone fractures and improves the accuracy in the fracture regions. Experiments in pelvic and ankle segmentation demonstrate the effectiveness on addressing the aforementioned problems and show an accurate, stable and consistent performance, indicating a broader application in other bone anatomies. Our algorithm also provides insights into combining the domain knowledge and deep neural networks.
>
---
#### [new 006] EchoVLM: Dynamic Mixture-of-Experts Vision-Language Model for Universal Ultrasound Intelligence
- **分类: cs.CV**

- **简介: 该论文提出EchoVLM，一种面向超声医学影像的视觉语言模型，旨在提升超声诊断的准确性和效率。通过混合专家架构，模型可完成报告生成、诊断和VQA任务，在多项指标上优于现有模型，具有重要临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.14977v1](http://arxiv.org/pdf/2509.14977v1)**

> **作者:** Chaoyin She; Ruifang Lu; Lida Chen; Wei Wang; Qinghua Huang
>
> **摘要:** Ultrasound imaging has become the preferred imaging modality for early cancer screening due to its advantages of non-ionizing radiation, low cost, and real-time imaging capabilities. However, conventional ultrasound diagnosis heavily relies on physician expertise, presenting challenges of high subjectivity and low diagnostic efficiency. Vision-language models (VLMs) offer promising solutions for this issue, but existing general-purpose models demonstrate limited knowledge in ultrasound medical tasks, with poor generalization in multi-organ lesion recognition and low efficiency across multi-task diagnostics. To address these limitations, we propose EchoVLM, a vision-language model specifically designed for ultrasound medical imaging. The model employs a Mixture of Experts (MoE) architecture trained on data spanning seven anatomical regions. This design enables the model to perform multiple tasks, including ultrasound report generation, diagnosis and visual question-answering (VQA). The experimental results demonstrated that EchoVLM achieved significant improvements of 10.15 and 4.77 points in BLEU-1 scores and ROUGE-1 scores respectively compared to Qwen2-VL on the ultrasound report generation task. These findings suggest that EchoVLM has substantial potential to enhance diagnostic accuracy in ultrasound imaging, thereby providing a viable technical solution for future clinical applications. Source code and model weights are available at https://github.com/Asunatan/EchoVLM.
>
---
#### [new 007] UCorr: Wire Detection and Depth Estimation for Autonomous Drones
- **分类: cs.CV**

- **简介: 该论文提出UCorr模型，用于无人机的电线检测与深度估计任务。针对电线细长难以检测的问题，采用单目端到端方法，结合时序相关层，提升检测与深度估计效果，提高无人机自主飞行安全性。**

- **链接: [http://arxiv.org/pdf/2509.14989v1](http://arxiv.org/pdf/2509.14989v1)**

> **作者:** Benedikt Kolbeinsson; Krystian Mikolajczyk
>
> **备注:** Published in Proceedings of the 4th International Conference on Robotics, Computer Vision and Intelligent Systems (ROBOVIS), 2024
>
> **摘要:** In the realm of fully autonomous drones, the accurate detection of obstacles is paramount to ensure safe navigation and prevent collisions. Among these challenges, the detection of wires stands out due to their slender profile, which poses a unique and intricate problem. To address this issue, we present an innovative solution in the form of a monocular end-to-end model for wire segmentation and depth estimation. Our approach leverages a temporal correlation layer trained on synthetic data, providing the model with the ability to effectively tackle the complex joint task of wire detection and depth estimation. We demonstrate the superiority of our proposed method over existing competitive approaches in the joint task of wire detection and depth estimation. Our results underscore the potential of our model to enhance the safety and precision of autonomous drones, shedding light on its promising applications in real-world scenarios.
>
---
#### [new 008] [Re] Improving Interpretation Faithfulness for Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer可解释性研究任务，旨在验证DDS方法是否提升模型在攻击下的鲁棒性，并评估其计算成本与环境影响。**

- **链接: [http://arxiv.org/pdf/2509.14846v1](http://arxiv.org/pdf/2509.14846v1)**

> **作者:** Izabela Kurek; Wojciech Trejter; Stipe Frkovic; Andro Erdelez
>
> **备注:** 13 pages article, 29 pdf pages, 19 figures, MLRC
>
> **摘要:** This work aims to reproduce the results of Faithful Vision Transformers (FViTs) proposed by arXiv:2311.17983 alongside interpretability methods for Vision Transformers from arXiv:2012.09838 and Xu (2022) et al. We investigate claims made by arXiv:2311.17983, namely that the usage of Diffusion Denoised Smoothing (DDS) improves interpretability robustness to (1) attacks in a segmentation task and (2) perturbation and attacks in a classification task. We also extend the original study by investigating the authors' claims that adding DDS to any interpretability method can improve its robustness under attack. This is tested on baseline methods and the recently proposed Attribution Rollout method. In addition, we measure the computational costs and environmental impact of obtaining an FViT through DDS. Our results broadly agree with the original study's findings, although minor discrepancies were found and discussed.
>
---
#### [new 009] Frame Sampling Strategies Matter: A Benchmark for small vision language models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频问答任务，旨在解决因帧采样策略不同导致的模型评估偏差问题。提出首个帧精确基准，验证了帧采样对小VLM性能的影响，并开源代码以推动标准化评估。**

- **链接: [http://arxiv.org/pdf/2509.14769v1](http://arxiv.org/pdf/2509.14769v1)**

> **作者:** Marija Brkic; Anas Filali Razzouki; Yannis Tevissen; Khalil Guetari; Mounim A. El Yacoubi
>
> **摘要:** Comparing vision language models on videos is particularly complex, as the performances is jointly determined by the model's visual representation capacity and the frame-sampling strategy used to construct the input. Current video benchmarks are suspected to suffer from substantial frame-sampling bias, as models are evaluated with different frame selection strategies. In this work, we propose the first frame-accurate benchmark of state-of-the-art small VLMs for video question-answering, evaluated under controlled frame-sampling strategies. Our results confirm the suspected bias and highlight both data-specific and task-specific behaviors of SVLMs under different frame-sampling techniques. By open-sourcing our benchmarking code, we provide the community with a reproducible and unbiased protocol for evaluating video VLMs and emphasize the need for standardized frame-sampling strategies tailored to each benchmarking dataset in future research.
>
---
#### [new 010] Attention Lattice Adapter: Visual Explanation Generation for Visual Foundation Model
- **分类: cs.CV**

- **简介: 该论文提出一种生成视觉解释的新方法，用于提升视觉基础模型的可解释性。针对现有方法适应性差的问题，设计了Attention Lattice Adapter和Alternating Epoch Architect机制，实现参数更新与解释生成，显著提升IoU等指标。**

- **链接: [http://arxiv.org/pdf/2509.14664v1](http://arxiv.org/pdf/2509.14664v1)**

> **作者:** Shinnosuke Hirano; Yuiga Wada; Tsumugi Iida; Komei Sugiura
>
> **备注:** Accepted for presentation at ICONIP2025
>
> **摘要:** In this study, we consider the problem of generating visual explanations in visual foundation models. Numerous methods have been proposed for this purpose; however, they often cannot be applied to complex models due to their lack of adaptability. To overcome these limitations, we propose a novel explanation generation method in visual foundation models that is aimed at both generating explanations and partially updating model parameters to enhance interpretability. Our approach introduces two novel mechanisms: Attention Lattice Adapter (ALA) and Alternating Epoch Architect (AEA). ALA mechanism simplifies the process by eliminating the need for manual layer selection, thus enhancing the model's adaptability and interpretability. Moreover, the AEA mechanism, which updates ALA's parameters every other epoch, effectively addresses the common issue of overly small attention regions. We evaluated our method on two benchmark datasets, CUB-200-2011 and ImageNet-S. Our results showed that our method outperformed the baseline methods in terms of mean intersection over union (IoU), insertion score, deletion score, and insertion-deletion score on both the CUB-200-2011 and ImageNet-S datasets. Notably, our best model achieved a 53.2-point improvement in mean IoU on the CUB-200-2011 dataset compared with the baselines.
>
---
#### [new 011] RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出RoboEye框架，用于提升机器人2D目标识别性能。针对电商仓库中因类别多、视角变化等问题导致的识别困难，结合3D几何关键点匹配增强2D特征，提高识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.14966v1](http://arxiv.org/pdf/2509.14966v1)**

> **作者:** Xingwu Zhang; Guanxuan Li; Zhuocheng Zhang; Zijun Long
>
> **摘要:** The rapidly growing number of product categories in large-scale e-commerce makes accurate object identification for automated packing in warehouses substantially more difficult. As the catalog grows, intra-class variability and a long tail of rare or visually similar items increase, and when combined with diverse packaging, cluttered containers, frequent occlusion, and large viewpoint changes-these factors amplify discrepancies between query and reference images, causing sharp performance drops for methods that rely solely on 2D appearance features. Thus, we propose RoboEye, a two-stage identification framework that dynamically augments 2D semantic features with domain-adapted 3D reasoning and lightweight adapters to bridge training deployment gaps. In the first stage, we train a large vision model to extract 2D features for generating candidate rankings. A lightweight 3D-feature-awareness module then estimates 3D feature quality and predicts whether 3D re-ranking is necessary, preventing performance degradation and avoiding unnecessary computation. When invoked, the second stage uses our robot 3D retrieval transformer, comprising a 3D feature extractor that produces geometry-aware dense features and a keypoint-based matcher that computes keypoint-correspondence confidences between query and reference images instead of conventional cosine-similarity scoring. Experiments show that RoboEye improves Recall@1 by 7.1% over the prior state of the art (RoboLLM). Moreover, RoboEye operates using only RGB images, avoiding reliance on explicit 3D inputs and reducing deployment costs. The code used in this paper is publicly available at: https://github.com/longkukuhi/RoboEye.
>
---
#### [new 012] Not All Degradations Are Equal: A Targeted Feature Denoising Framework for Generalizable Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像超分辨率任务，旨在解决模型在未知退化下泛化能力差的问题。提出针对性特征去噪框架，重点抑制对噪声的过拟合，提升模型在多种退化场景下的性能。**

- **链接: [http://arxiv.org/pdf/2509.14841v1](http://arxiv.org/pdf/2509.14841v1)**

> **作者:** Hongjun Wang; Jiyuan Chen; Zhengwei Yin; Xuan Song; Yinqiang Zheng
>
> **摘要:** Generalizable Image Super-Resolution aims to enhance model generalization capabilities under unknown degradations. To achieve this goal, the models are expected to focus only on image content-related features instead of overfitting degradations. Recently, numerous approaches such as Dropout and Feature Alignment have been proposed to suppress models' natural tendency to overfit degradations and yield promising results. Nevertheless, these works have assumed that models overfit to all degradation types (e.g., blur, noise, JPEG), while through careful investigations in this paper, we discover that models predominantly overfit to noise, largely attributable to its distinct degradation pattern compared to other degradation types. In this paper, we propose a targeted feature denoising framework, comprising noise detection and denoising modules. Our approach presents a general solution that can be seamlessly integrated with existing super-resolution models without requiring architectural modifications. Our framework demonstrates superior performance compared to previous regularization-based methods across five traditional benchmarks and datasets, encompassing both synthetic and real-world scenarios.
>
---
#### [new 013] PRISM: Product Retrieval In Shopping Carts using Hybrid Matching
- **分类: cs.CV**

- **简介: 该论文提出PRISM方法，解决零售场景中产品检索难题。通过结合视觉-语言模型与像素级匹配，分三阶段实现高效精准检索。实验表明其在ABV数据集上优于现有方法，提升4.21%的Top-1准确率，满足实时需求。**

- **链接: [http://arxiv.org/pdf/2509.14985v1](http://arxiv.org/pdf/2509.14985v1)**

> **作者:** Arda Kabadayi; Senem Velipasalar; Jiajing Chen
>
> **摘要:** Compared to traditional image retrieval tasks, product retrieval in retail settings is even more challenging. Products of the same type from different brands may have highly similar visual appearances, and the query image may be taken from an angle that differs significantly from view angles of the stored catalog images. Foundational models, such as CLIP and SigLIP, often struggle to distinguish these subtle but important local differences. Pixel-wise matching methods, on the other hand, are computationally expensive and incur prohibitively high matching times. In this paper, we propose a new, hybrid method, called PRISM, for product retrieval in retail settings by leveraging the advantages of both vision-language model-based and pixel-wise matching approaches. To provide both efficiency/speed and finegrained retrieval accuracy, PRISM consists of three stages: 1) A vision-language model (SigLIP) is employed first to retrieve the top 35 most semantically similar products from a fixed gallery, thereby narrowing the search space significantly; 2) a segmentation model (YOLO-E) is applied to eliminate background clutter; 3) fine-grained pixel-level matching is performed using LightGlue across the filtered candidates. This framework enables more accurate discrimination between products with high inter-class similarity by focusing on subtle visual cues often missed by global models. Experiments performed on the ABV dataset show that our proposed PRISM outperforms the state-of-the-art image retrieval methods by 4.21% in top-1 accuracy while still remaining within the bounds of real-time processing for practical retail deployments.
>
---
#### [new 014] NeRF-based Visualization of 3D Cues Supporting Data-Driven Spacecraft Pose Estimation
- **分类: cs.CV**

- **简介: 论文提出基于NeRF的可视化方法，用于揭示数据驱动航天器位姿估计模型依赖的3D视觉线索。该任务旨在提升位姿估计的可解释性，通过反向传播梯度训练图像生成器，提取并展示关键3D特征，增强对模型决策过程的理解。**

- **链接: [http://arxiv.org/pdf/2509.14890v1](http://arxiv.org/pdf/2509.14890v1)**

> **作者:** Antoine Legrand; Renaud Detry; Christophe De Vleeschouwer
>
> **备注:** under review (8 pages, 2 figures)
>
> **摘要:** On-orbit operations require the estimation of the relative 6D pose, i.e., position and orientation, between a chaser spacecraft and its target. While data-driven spacecraft pose estimation methods have been developed, their adoption in real missions is hampered by the lack of understanding of their decision process. This paper presents a method to visualize the 3D visual cues on which a given pose estimator relies. For this purpose, we train a NeRF-based image generator using the gradients back-propagated through the pose estimation network. This enforces the generator to render the main 3D features exploited by the spacecraft pose estimation network. Experiments demonstrate that our method recovers the relevant 3D cues. Furthermore, they offer additional insights on the relationship between the pose estimation network supervision and its implicit representation of the target spacecraft.
>
---
#### [new 015] Beyond Random Masking: A Dual-Stream Approach for Rotation-Invariant Point Cloud Masked Autoencoders
- **分类: cs.CV**

- **简介: 该论文提出一种双流掩码方法，用于改进旋转不变点云掩码自编码器。针对随机掩码忽略几何结构和语义一致性的缺陷，结合空间网格掩码与语义掩码，提升模型在不同旋转场景下的性能。**

- **链接: [http://arxiv.org/pdf/2509.14975v1](http://arxiv.org/pdf/2509.14975v1)**

> **作者:** Xuanhua Yin; Dingxin Zhang; Yu Feng; Shunqi Mao; Jianhui Yu; Weidong Cai
>
> **备注:** 8 pages, 4 figures, aceppted by DICTA 2025
>
> **摘要:** Existing rotation-invariant point cloud masked autoencoders (MAE) rely on random masking strategies that overlook geometric structure and semantic coherence. Random masking treats patches independently, failing to capture spatial relationships consistent across orientations and overlooking semantic object parts that maintain identity regardless of rotation. We propose a dual-stream masking approach combining 3D Spatial Grid Masking and Progressive Semantic Masking to address these fundamental limitations. Grid masking creates structured patterns through coordinate sorting to capture geometric relationships that persist across different orientations, while semantic masking uses attention-driven clustering to discover semantically meaningful parts and maintain their coherence during masking. These complementary streams are orchestrated via curriculum learning with dynamic weighting, progressing from geometric understanding to semantic discovery. Designed as plug-and-play components, our strategies integrate into existing rotation-invariant frameworks without architectural changes, ensuring broad compatibility across different approaches. Comprehensive experiments on ModelNet40, ScanObjectNN, and OmniObject3D demonstrate consistent improvements across various rotation scenarios, showing substantial performance gains over the baseline rotation-invariant methods.
>
---
#### [new 016] RynnVLA-001: Using Human Demonstrations to Improve Robot Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出RynnVLA-001，用于机器人操作任务。通过两阶段预训练方法，结合视觉-语言-动作模型与ActionVAE，提升动作表示与预测效果，解决机器人操作中视觉与动作联合建模的问题。**

- **链接: [http://arxiv.org/pdf/2509.15212v1](http://arxiv.org/pdf/2509.15212v1)**

> **作者:** Yuming Jiang; Siteng Huang; Shengke Xue; Yaxi Zhao; Jun Cen; Sicong Leng; Kehan Li; Jiayan Guo; Kexiang Wang; Mingxiu Chen; Fan Wang; Deli Zhao; Xin Li
>
> **备注:** GitHub Project: https://github.com/alibaba-damo-academy/RynnVLA-001
>
> **摘要:** This paper presents RynnVLA-001, a vision-language-action(VLA) model built upon large-scale video generative pretraining from human demonstrations. We propose a novel two-stage pretraining methodology. The first stage, Ego-Centric Video Generative Pretraining, trains an Image-to-Video model on 12M ego-centric manipulation videos to predict future frames conditioned on an initial frame and a language instruction. The second stage, Human-Centric Trajectory-Aware Modeling, extends this by jointly predicting future keypoint trajectories, thereby effectively bridging visual frame prediction with action prediction. Furthermore, to enhance action representation, we propose ActionVAE, a variational autoencoder that compresses sequences of actions into compact latent embeddings, reducing the complexity of the VLA output space. When finetuned on the same downstream robotics datasets, RynnVLA-001 achieves superior performance over state-of-the-art baselines, demonstrating that the proposed pretraining strategy provides a more effective initialization for VLA models.
>
---
#### [new 017] FMGS-Avatar: Mesh-Guided 2D Gaussian Splatting with Foundation Model Priors for 3D Monocular Avatar Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出FMGS-Avatar方法，解决单目视频中高保真3D人体Avatar重建问题。通过网格引导的2D高斯泼溅和基础模型先验知识融合，提升几何细节与外观真实性，实现更优重建效果。**

- **链接: [http://arxiv.org/pdf/2509.14739v1](http://arxiv.org/pdf/2509.14739v1)**

> **作者:** Jinlong Fan; Bingyu Hu; Xingguang Li; Yuxiang Yang; Jing Zhang
>
> **摘要:** Reconstructing high-fidelity animatable human avatars from monocular videos remains challenging due to insufficient geometric information in single-view observations. While recent 3D Gaussian Splatting methods have shown promise, they struggle with surface detail preservation due to the free-form nature of 3D Gaussian primitives. To address both the representation limitations and information scarcity, we propose a novel method, \textbf{FMGS-Avatar}, that integrates two key innovations. First, we introduce Mesh-Guided 2D Gaussian Splatting, where 2D Gaussian primitives are attached directly to template mesh faces with constrained position, rotation, and movement, enabling superior surface alignment and geometric detail preservation. Second, we leverage foundation models trained on large-scale datasets, such as Sapiens, to complement the limited visual cues from monocular videos. However, when distilling multi-modal prior knowledge from foundation models, conflicting optimization objectives can emerge as different modalities exhibit distinct parameter sensitivities. We address this through a coordinated training strategy with selective gradient isolation, enabling each loss component to optimize its relevant parameters without interference. Through this combination of enhanced representation and coordinated information distillation, our approach significantly advances 3D monocular human avatar reconstruction. Experimental evaluation demonstrates superior reconstruction quality compared to existing methods, with notable gains in geometric accuracy and appearance fidelity while providing rich semantic information. Additionally, the distilled prior knowledge within a shared canonical space naturally enables spatially and temporally consistent rendering under novel views and poses.
>
---
#### [new 018] DiffVL: Diffusion-Based Visual Localization on 2D Maps via BEV-Conditioned GPS Denoising
- **分类: cs.CV**

- **简介: 该论文提出DiffVL，将视觉定位转化为基于扩散模型的GPS去噪任务，利用BEV特征和SD地图实现无需HD地图的高精度定位，解决传统方法依赖HD地图和忽略GPS噪声的问题。**

- **链接: [http://arxiv.org/pdf/2509.14565v1](http://arxiv.org/pdf/2509.14565v1)**

> **作者:** Li Gao; Hongyang Sun; Liu Liu; Yunhao Li; Yang Cai
>
> **摘要:** Accurate visual localization is crucial for autonomous driving, yet existing methods face a fundamental dilemma: While high-definition (HD) maps provide high-precision localization references, their costly construction and maintenance hinder scalability, which drives research toward standard-definition (SD) maps like OpenStreetMap. Current SD-map-based approaches primarily focus on Bird's-Eye View (BEV) matching between images and maps, overlooking a ubiquitous signal-noisy GPS. Although GPS is readily available, it suffers from multipath errors in urban environments. We propose DiffVL, the first framework to reformulate visual localization as a GPS denoising task using diffusion models. Our key insight is that noisy GPS trajectory, when conditioned on visual BEV features and SD maps, implicitly encode the true pose distribution, which can be recovered through iterative diffusion refinement. DiffVL, unlike prior BEV-matching methods (e.g., OrienterNet) or transformer-based registration approaches, learns to reverse GPS noise perturbations by jointly modeling GPS, SD map, and visual signals, achieving sub-meter accuracy without relying on HD maps. Experiments on multiple datasets demonstrate that our method achieves state-of-the-art accuracy compared to BEV-matching baselines. Crucially, our work proves that diffusion models can enable scalable localization by treating noisy GPS as a generative prior-making a paradigm shift from traditional matching-based methods.
>
---
#### [new 019] Lost in Translation? Vocabulary Alignment for Source-Free Domain Adaptation in Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文提出VocAlign，用于开放词汇语义分割的源自由域适应任务。通过学生-教师框架与词汇对齐策略，结合LoRA和Top-K机制，提升伪标签生成与适应性能，在CityScapes数据集上取得显著提升。**

- **链接: [http://arxiv.org/pdf/2509.15225v1](http://arxiv.org/pdf/2509.15225v1)**

> **作者:** Silvio Mazzucco; Carl Persson; Mattia Segu; Pier Luigi Dovesi; Federico Tombari; Luc Van Gool; Matteo Poggi
>
> **备注:** BMVC 2025 - Project Page: https://thegoodailab.org/blog/vocalign - Code: https://github.com/Sisso16/VocAlign
>
> **摘要:** We introduce VocAlign, a novel source-free domain adaptation framework specifically designed for VLMs in open-vocabulary semantic segmentation. Our method adopts a student-teacher paradigm enhanced with a vocabulary alignment strategy, which improves pseudo-label generation by incorporating additional class concepts. To ensure efficiency, we use Low-Rank Adaptation (LoRA) to fine-tune the model, preserving its original capabilities while minimizing computational overhead. In addition, we propose a Top-K class selection mechanism for the student model, which significantly reduces memory requirements while further improving adaptation performance. Our approach achieves a notable 6.11 mIoU improvement on the CityScapes dataset and demonstrates superior performance on zero-shot segmentation benchmarks, setting a new standard for source-free adaptation in the open-vocabulary setting.
>
---
#### [new 020] ProtoMedX: Towards Explainable Multi-Modal Prototype Learning for Bone Health Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ProtoMedX，用于骨健康分类任务，解决AI模型可解释性不足的问题。通过融合DEXA扫描和患者记录，设计可解释的多模态原型学习模型，在NHS数据集上取得优于现有方法的性能。**

- **链接: [http://arxiv.org/pdf/2509.14830v1](http://arxiv.org/pdf/2509.14830v1)**

> **作者:** Alvaro Lopez Pellicer; Andre Mariucci; Plamen Angelov; Marwan Bukhari; Jemma G. Kerns
>
> **备注:** Accepted ICCV 2025. Adaptation, Fairness, Explainability in AI Medical Imaging (PHAROS-AFE-AIMI Workshop). 8 pages, 5 figures, 4 tables
>
> **摘要:** Bone health studies are crucial in medical practice for the early detection and treatment of Osteopenia and Osteoporosis. Clinicians usually make a diagnosis based on densitometry (DEXA scans) and patient history. The applications of AI in this field are ongoing research. Most successful methods rely on deep learning models that use vision alone (DEXA/X-ray imagery) and focus on prediction accuracy, while explainability is often disregarded and left to post hoc assessments of input contributions. We propose ProtoMedX, a multi-modal model that uses both DEXA scans of the lumbar spine and patient records. ProtoMedX's prototype-based architecture is explainable by design, which is crucial for medical applications, especially in the context of the upcoming EU AI Act, as it allows explicit analysis of model decisions, including incorrect ones. ProtoMedX demonstrates state-of-the-art performance in bone health classification while also providing explanations that can be visually understood by clinicians. Using a dataset of 4,160 real NHS patients, the proposed ProtoMedX achieves 87.58% accuracy in vision-only tasks and 89.8% in its multi-modal variant, both surpassing existing published methods.
>
---
#### [new 021] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文提出OmniSegmentor，用于多模态语义分割任务。旨在解决多模态预训练缺乏灵活性的问题，构建了包含五种视觉模态的ImageNeXt数据集，并设计高效预训练方法，实现跨模态场景下的性能提升。**

- **链接: [http://arxiv.org/pdf/2509.15096v1](http://arxiv.org/pdf/2509.15096v1)**

> **作者:** Bo-Wen Yin; Jiao-Long Cao; Xuying Zhang; Yuming Chen; Ming-Ming Cheng; Qibin Hou
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent research on representation learning has proved the merits of multi-modal clues for robust semantic segmentation. Nevertheless, a flexible pretrain-and-finetune pipeline for multiple visual modalities remains unexplored. In this paper, we propose a novel multi-modal learning framework, termed OmniSegmentor. It has two key innovations: 1) Based on ImageNet, we assemble a large-scale dataset for multi-modal pretraining, called ImageNeXt, which contains five popular visual modalities. 2) We provide an efficient pretraining manner to endow the model with the capacity to encode different modality information in the ImageNeXt. For the first time, we introduce a universal multi-modal pretraining framework that consistently amplifies the model's perceptual capabilities across various scenarios, regardless of the arbitrary combination of the involved modalities. Remarkably, our OmniSegmentor achieves new state-of-the-art records on a wide range of multi-modal semantic segmentation datasets, including NYU Depthv2, EventScape, MFNet, DeLiVER, SUNRGBD, and KITTI-360.
>
---
#### [new 022] ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data
- **分类: cs.CV**

- **简介: 该论文提出ScaleCUA，旨在解决开源计算机使用代理（CUA）缺乏大规模数据与模型的问题。通过构建跨平台数据集并训练模型，实现多操作系统与任务领域的通用CUA，显著提升性能并发布资源以推动研究。**

- **链接: [http://arxiv.org/pdf/2509.15221v1](http://arxiv.org/pdf/2509.15221v1)**

> **作者:** Zhaoyang Liu; JingJing Xie; Zichen Ding; Zehao Li; Bowen Yang; Zhenyu Wu; Xuehui Wang; Qiushi Sun; Shi Liu; Weiyun Wang; Shenglong Ye; Qingyun Li; Zeyue Tian; Gen Luo; Xiangyu Yue; Biqing Qi; Kai Chen; Bowen Zhou; Yu Qiao; Qifeng Chen; Wenhai Wang
>
> **摘要:** Vision-Language Models (VLMs) have enabled computer use agents (CUAs) that operate GUIs autonomously, showing great potential, yet progress is limited by the lack of large-scale, open-source computer use data and foundation models. In this work, we introduce ScaleCUA, a step toward scaling open-source CUAs. It offers a large-scale dataset spanning 6 operating systems and 3 task domains, built via a closed-loop pipeline uniting automated agents with human experts. Trained on this scaled-up data, ScaleCUA can operate seamlessly across platforms. Specifically, it delivers strong gains over baselines (+26.6 on WebArena-Lite-v2, +10.7 on ScreenSpot-Pro) and sets new state-of-the-art results (94.4% on MMBench-GUI L1-Hard, 60.6% on OSWorld-G, 47.4% on WebArena-Lite-v2). These findings underscore the power of data-driven scaling for general-purpose computer use agents. We will release data, models, and code to advance future research: https://github.com/OpenGVLab/ScaleCUA.
>
---
#### [new 023] Dataset Distillation for Super-Resolution without Class Labels and Pre-trained Models
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决数据效率低、依赖预训练模型的问题。提出无需类别标签和预训练模型的数据蒸馏方法，通过扩散模型生成训练数据，显著减少训练数据量和计算时间，取得先进性能。**

- **链接: [http://arxiv.org/pdf/2509.14777v1](http://arxiv.org/pdf/2509.14777v1)**

> **作者:** Sunwoo Cho; Yejin Jung; Nam Ik Cho; Jae Woong Soh
>
> **摘要:** Training deep neural networks has become increasingly demanding, requiring large datasets and significant computational resources, especially as model complexity advances. Data distillation methods, which aim to improve data efficiency, have emerged as promising solutions to this challenge. In the field of single image super-resolution (SISR), the reliance on large training datasets highlights the importance of these techniques. Recently, a generative adversarial network (GAN) inversion-based data distillation framework for SR was proposed, showing potential for better data utilization. However, the current method depends heavily on pre-trained SR networks and class-specific information, limiting its generalizability and applicability. To address these issues, we introduce a new data distillation approach for image SR that does not need class labels or pre-trained SR models. In particular, we first extract high-gradient patches and categorize images based on CLIP features, then fine-tune a diffusion model on the selected patches to learn their distribution and synthesize distilled training images. Experimental results show that our method achieves state-of-the-art performance while using significantly less training data and requiring less computational time. Specifically, when we train a baseline Transformer model for SR with only 0.68\% of the original dataset, the performance drop is just 0.3 dB. In this case, diffusion model fine-tuning takes 4 hours, and SR model training completes within 1 hour, much shorter than the 11-hour training time with the full dataset.
>
---
#### [new 024] A Race Bias Free Face Aging Model for Reliable Kinship Verification
- **分类: cs.CV**

- **简介: 该论文提出RA-GAN模型，解决人脸年龄转换中的种族偏见问题，提升亲属关系验证准确性。通过新模块RACEpSp和特征混合器生成无偏图像，实验表明其在多个数据集上优于现有方法。属于人脸识别与亲属验证任务。**

- **链接: [http://arxiv.org/pdf/2509.15177v1](http://arxiv.org/pdf/2509.15177v1)**

> **作者:** Ali Nazari; Bardiya Kariminia; Mohsen Ebrahimi Moghaddam
>
> **摘要:** The age gap in kinship verification addresses the time difference between the photos of the parent and the child. Moreover, their same-age photos are often unavailable, and face aging models are racially biased, which impacts the likeness of photos. Therefore, we propose a face aging GAN model, RA-GAN, consisting of two new modules, RACEpSp and a feature mixer, to produce racially unbiased images. The unbiased synthesized photos are used in kinship verification to investigate the results of verifying same-age parent-child images. The experiments demonstrate that our RA-GAN outperforms SAM-GAN on an average of 13.14\% across all age groups, and CUSP-GAN in the 60+ age group by 9.1\% in terms of racial accuracy. Moreover, RA-GAN can preserve subjects' identities better than SAM-GAN and CUSP-GAN across all age groups. Additionally, we demonstrate that transforming parent and child images from the KinFaceW-I and KinFaceW-II datasets to the same age can enhance the verification accuracy across all age groups. The accuracy increases with our RA-GAN for the kinship relationships of father-son and father-daughter, mother-son, and mother-daughter, which are 5.22, 5.12, 1.63, and 0.41, respectively, on KinFaceW-I. Additionally, the accuracy for the relationships of father-daughter, father-son, and mother-son is 2.9, 0.39, and 1.6 on KinFaceW-II, respectively. The code is available at~\href{https://github.com/bardiya2254kariminia/An-Age-Transformation-whitout-racial-bias-for-Kinship-verification}{Github}
>
---
#### [new 025] Synthetic-to-Real Object Detection using YOLOv11 and Domain Randomization Strategies
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，旨在解决合成数据到真实场景的域差距问题。研究使用YOLOv11和域随机化策略训练模型检测汤罐头，通过增加数据多样性与优化增强策略，最终在Kaggle竞赛中取得0.910 mAP@50成绩。**

- **链接: [http://arxiv.org/pdf/2509.15045v1](http://arxiv.org/pdf/2509.15045v1)**

> **作者:** Luisa Torquato Niño; Hamza A. A. Gardi
>
> **摘要:** This paper addresses the synthetic-to-real domain gap in object detection, focusing on training a YOLOv11 model to detect a specific object (a soup can) using only synthetic data and domain randomization strategies. The methodology involves extensive experimentation with data augmentation, dataset composition, and model scaling. While synthetic validation metrics were consistently high, they proved to be poor predictors of real-world performance. Consequently, models were also evaluated qualitatively, through visual inspection of predictions, and quantitatively, on a manually labeled real-world test set, to guide development. Final mAP@50 scores were provided by the official Kaggle competition. Key findings indicate that increasing synthetic dataset diversity, specifically by including varied perspectives and complex backgrounds, combined with carefully tuned data augmentation, were crucial in bridging the domain gap. The best performing configuration, a YOLOv11l model trained on an expanded and diverse dataset, achieved a final mAP@50 of 0.910 on the competition's hidden test set. This result demonstrates the potential of a synthetic-only training approach while also highlighting the remaining challenges in fully capturing real-world variability.
>
---
#### [new 026] Trade-offs in Cross-Domain Generalization of Foundation Model Fine-Tuned for Biometric Applications
- **分类: cs.CV**

- **简介: 论文研究基础模型在生物特征任务微调后的跨领域泛化能力下降问题，通过对比不同任务下的模型表现，揭示过拟合与任务复杂度的关系，并验证模型规模对泛化能力的影响。**

- **链接: [http://arxiv.org/pdf/2509.14921v1](http://arxiv.org/pdf/2509.14921v1)**

> **作者:** Tahar Chettaoui; Naser Damer; Fadi Boutros
>
> **备注:** Accepted at the IEEE International Joint Conference on Biometrics 2025 (IJCB 2025)
>
> **摘要:** Foundation models such as CLIP have demonstrated exceptional zero- and few-shot transfer capabilities across diverse vision tasks. However, when fine-tuned for highly specialized biometric tasks, face recognition (FR), morphing attack detection (MAD), and presentation attack detection (PAD), these models may suffer from over-specialization. Thus, they may lose one of their foundational strengths, cross-domain generalization. In this work, we systematically quantify these trade-offs by evaluating three instances of CLIP fine-tuned for FR, MAD, and PAD. We evaluate each adapted model as well as the original CLIP baseline on 14 general vision datasets under zero-shot and linear-probe protocols, alongside common FR, MAD, and PAD benchmarks. Our results indicate that fine-tuned models suffer from over-specialization, especially when fine-tuned for complex tasks of FR. Also, our results pointed out that task complexity and classification head design, multi-class (FR) vs. binary (MAD and PAD), correlate with the degree of catastrophic forgetting. The FRoundation model with the ViT-L backbone outperforms other approaches on the large-scale FR benchmark IJB-C, achieving an improvement of up to 58.52%. However, it experiences a substantial performance drop on ImageNetV2, reaching only 51.63% compared to 69.84% achieved by the baseline CLIP model. Moreover, the larger CLIP architecture consistently preserves more of the model's original generalization ability than the smaller variant, indicating that increased model capacity may help mitigate over-specialization.
>
---
#### [new 027] Geometric Image Synchronization with Deep Watermarking
- **分类: cs.CV**

- **简介: 论文提出SyncSeal方法，用于图像几何同步任务，解决现有水印方法对几何变换鲁棒性不足的问题。通过嵌入和提取网络，实现对图像旋转、裁剪等变换的准确估计与逆变换，提升水印抗几何攻击能力。**

- **链接: [http://arxiv.org/pdf/2509.15208v1](http://arxiv.org/pdf/2509.15208v1)**

> **作者:** Pierre Fernandez; Tomáš Souček; Nikola Jovanović; Hady Elsahar; Sylvestre-Alvise Rebuffi; Valeriu Lacatusu; Tuan Tran; Alexandre Mourachko
>
> **备注:** Pre-print. Code at: https://github.com/facebookresearch/wmar/tree/main/syncseal
>
> **摘要:** Synchronization is the task of estimating and inverting geometric transformations (e.g., crop, rotation) applied to an image. This work introduces SyncSeal, a bespoke watermarking method for robust image synchronization, which can be applied on top of existing watermarking methods to enhance their robustness against geometric transformations. It relies on an embedder network that imperceptibly alters images and an extractor network that predicts the geometric transformation to which the image was subjected. Both networks are end-to-end trained to minimize the error between the predicted and ground-truth parameters of the transformation, combined with a discriminator to maintain high perceptual quality. We experimentally validate our method on a wide variety of geometric and valuemetric transformations, demonstrating its effectiveness in accurately synchronizing images. We further show that our synchronization can effectively upgrade existing watermarking methods to withstand geometric transformations to which they were previously vulnerable.
>
---
#### [new 028] Adaptive and Iterative Point Cloud Denoising with Score-Based Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于点云去噪任务，旨在从含噪声的点云中恢复干净点云。提出基于分数扩散模型的自适应迭代去噪方法，通过估计噪声变化并设计自适应去噪计划，提升去噪效果，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14560v1](http://arxiv.org/pdf/2509.14560v1)**

> **作者:** Zhaonan Wang; Manyi Li; ShiQing Xin; Changhe Tu
>
> **摘要:** Point cloud denoising task aims to recover the clean point cloud from the scanned data coupled with different levels or patterns of noise. The recent state-of-the-art methods often train deep neural networks to update the point locations towards the clean point cloud, and empirically repeat the denoising process several times in order to obtain the denoised results. It is not clear how to efficiently arrange the iterative denoising processes to deal with different levels or patterns of noise. In this paper, we propose an adaptive and iterative point cloud denoising method based on the score-based diffusion model. For a given noisy point cloud, we first estimate the noise variation and determine an adaptive denoising schedule with appropriate step sizes, then invoke the trained network iteratively to update point clouds following the adaptive schedule. To facilitate this adaptive and iterative denoising process, we design the network architecture and a two-stage sampling strategy for the network training to enable feature fusion and gradient fusion for iterative denoising. Compared to the state-of-the-art point cloud denoising methods, our approach obtains clean and smooth denoised point clouds, while preserving the shape boundary and details better. Our results not only outperform the other methods both qualitatively and quantitatively, but also are preferable on the synthetic dataset with different patterns of noises, as well as the real-scanned dataset.
>
---
#### [new 029] Domain Adaptation for Ulcerative Colitis Severity Estimation Using Patient-Level Diagnoses
- **分类: cs.CV**

- **简介: 该论文属于医学图像领域中的域适应任务，旨在解决因设备和环境差异导致的溃疡性结肠炎严重程度估计模型性能下降问题。提出了一种弱监督域适应方法，利用患者级诊断结果作为弱监督信号，提升跨域场景下的估计效果。**

- **链接: [http://arxiv.org/pdf/2509.14573v1](http://arxiv.org/pdf/2509.14573v1)**

> **作者:** Takamasa Yamaguchi; Brian Kenji Iwana; Ryoma Bise; Shota Harada; Takumi Okuo; Kiyohito Tanaka; Kaito Shiku
>
> **备注:** Accepted to MICCAI workshop 2025 (International conference on machine learning in medical imaging)
>
> **摘要:** The development of methods to estimate the severity of Ulcerative Colitis (UC) is of significant importance. However, these methods often suffer from domain shifts caused by differences in imaging devices and clinical settings across hospitals. Although several domain adaptation methods have been proposed to address domain shift, they still struggle with the lack of supervision in the target domain or the high cost of annotation. To overcome these challenges, we propose a novel Weakly Supervised Domain Adaptation method that leverages patient-level diagnostic results, which are routinely recorded in UC diagnosis, as weak supervision in the target domain. The proposed method aligns class-wise distributions across domains using Shared Aggregation Tokens and a Max-Severity Triplet Loss, which leverages the characteristic that patient-level diagnoses are determined by the most severe region within each patient. Experimental results demonstrate that our method outperforms comparative DA approaches, improving UC severity estimation in a domain-shifted setting.
>
---
#### [new 030] No Modality Left Behind: Adapting to Missing Modalities via Knowledge Distillation for Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文针对脑肿瘤分割任务，解决多模态MRI中模态缺失导致的模型性能下降问题。提出AdaMM框架，通过知识蒸馏和三个协同模块提升模型适应性与鲁棒性，实验表明其在单模态和弱模态配置下表现优异。**

- **链接: [http://arxiv.org/pdf/2509.15017v1](http://arxiv.org/pdf/2509.15017v1)**

> **作者:** Shenghao Zhu; Yifei Chen; Weihong Chen; Shuo Jiang; Guanyu Zhou; Yuanhan Wang; Feiwei Qin; Changmiao Wang; Qiyuan Tian
>
> **备注:** 38 pages, 9 figures
>
> **摘要:** Accurate brain tumor segmentation is essential for preoperative evaluation and personalized treatment. Multi-modal MRI is widely used due to its ability to capture complementary tumor features across different sequences. However, in clinical practice, missing modalities are common, limiting the robustness and generalizability of existing deep learning methods that rely on complete inputs, especially under non-dominant modality combinations. To address this, we propose AdaMM, a multi-modal brain tumor segmentation framework tailored for missing-modality scenarios, centered on knowledge distillation and composed of three synergistic modules. The Graph-guided Adaptive Refinement Module explicitly models semantic associations between generalizable and modality-specific features, enhancing adaptability to modality absence. The Bi-Bottleneck Distillation Module transfers structural and textural knowledge from teacher to student models via global style matching and adversarial feature alignment. The Lesion-Presence-Guided Reliability Module predicts prior probabilities of lesion types through an auxiliary classification task, effectively suppressing false positives under incomplete inputs. Extensive experiments on the BraTS 2018 and 2024 datasets demonstrate that AdaMM consistently outperforms existing methods, exhibiting superior segmentation accuracy and robustness, particularly in single-modality and weak-modality configurations. In addition, we conduct a systematic evaluation of six categories of missing-modality strategies, confirming the superiority of knowledge distillation and offering practical guidance for method selection and future research. Our source code is available at https://github.com/Quanato607/AdaMM.
>
---
#### [new 031] DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection
- **分类: cs.CV**

- **简介: 该论文属于合成图像检测任务，旨在提升多模态大语言模型（MLLM）在检测合成图像时的准确性与可解释性。提出DF-LLaVA框架，通过提示引导知识注入，提升检测性能并保持可解释性。**

- **链接: [http://arxiv.org/pdf/2509.14957v1](http://arxiv.org/pdf/2509.14957v1)**

> **作者:** Zhuokang Shen; Kaisen Zhang; Bohan Jia; Yuan Fang; Zhou Yu; Shaohui Lin
>
> **备注:** Under review
>
> **摘要:** With the increasing prevalence of synthetic images, evaluating image authenticity and locating forgeries accurately while maintaining human interpretability remains a challenging task. Existing detection models primarily focus on simple authenticity classification, ultimately providing only a forgery probability or binary judgment, which offers limited explanatory insights into image authenticity. Moreover, while MLLM-based detection methods can provide more interpretable results, they still lag behind expert models in terms of pure authenticity classification accuracy. To address this, we propose DF-LLaVA, a simple yet effective framework that unlocks the intrinsic discrimination potential of MLLMs. Our approach first extracts latent knowledge from MLLMs and then injects it into training via prompts. This framework allows LLaVA to achieve outstanding detection accuracy exceeding expert models while still maintaining the interpretability offered by MLLMs. Extensive experiments confirm the superiority of our DF-LLaVA, achieving both high accuracy and explainability in synthetic image detection. Code is available online at: https://github.com/Eliot-Shen/DF-LLaVA.
>
---
#### [new 032] Transplant-Ready? Evaluating AI Lung Segmentation Models in Candidates with Severe Lung Disease
- **分类: cs.CV**

- **简介: 该论文评估了三种AI肺分割模型在严重肺病移植候选者中的性能，分析其在不同病情、病理类型和肺侧的表现，发现Unet-R231表现最佳，但严重病例性能下降明显，需进一步优化。任务为医学图像分割，解决肺移植术前规划中模型适用性问题。**

- **链接: [http://arxiv.org/pdf/2509.15083v1](http://arxiv.org/pdf/2509.15083v1)**

> **作者:** Jisoo Lee; Michael R. Harowicz; Yuwen Chen; Hanxue Gu; Isaac S. Alderete; Lin Li; Maciej A. Mazurowski; Matthew G. Hartwig
>
> **备注:** 24 pages
>
> **摘要:** This study evaluates publicly available deep-learning based lung segmentation models in transplant-eligible patients to determine their performance across disease severity levels, pathology categories, and lung sides, and to identify limitations impacting their use in preoperative planning in lung transplantation. This retrospective study included 32 patients who underwent chest CT scans at Duke University Health System between 2017 and 2019 (total of 3,645 2D axial slices). Patients with standard axial CT scans were selected based on the presence of two or more lung pathologies of varying severity. Lung segmentation was performed using three previously developed deep learning models: Unet-R231, TotalSegmentator, MedSAM. Performance was assessed using quantitative metrics (volumetric similarity, Dice similarity coefficient, Hausdorff distance) and a qualitative measure (four-point clinical acceptability scale). Unet-R231 consistently outperformed TotalSegmentator and MedSAM in general, for different severity levels, and pathology categories (p<0.05). All models showed significant performance declines from mild to moderate-to-severe cases, particularly in volumetric similarity (p<0.05), without significant differences among lung sides or pathology types. Unet-R231 provided the most accurate automated lung segmentation among evaluated models with TotalSegmentator being a close second, though their performance declined significantly in moderate-to-severe cases, emphasizing the need for specialized model fine-tuning in severe pathology contexts.
>
---
#### [new 033] MemEvo: Memory-Evolving Incremental Multi-view Clustering
- **分类: cs.CV**

- **简介: 该论文提出MemEvo方法，解决增量多视角聚类中的稳定性-可塑性困境。通过模拟海马体与前额叶皮层记忆机制，设计对齐模块与知识巩固模块，实现新旧知识平衡。属于多视角聚类任务。**

- **链接: [http://arxiv.org/pdf/2509.14544v1](http://arxiv.org/pdf/2509.14544v1)**

> **作者:** Zisen Kong; Bo Zhong; Pengyuan Li; Dongxia Chang; Yiming Wang
>
> **摘要:** Incremental multi-view clustering aims to achieve stable clustering results while addressing the stability-plasticity dilemma (SPD) in incremental views. At the core of SPD is the challenge that the model must have enough plasticity to quickly adapt to new data, while maintaining sufficient stability to consolidate long-term knowledge and prevent catastrophic forgetting. Inspired by the hippocampal-prefrontal cortex collaborative memory mechanism in neuroscience, we propose a Memory-Evolving Incremental Multi-view Clustering method (MemEvo) to achieve this balance. First, we propose a hippocampus-inspired view alignment module that captures the gain information of new views by aligning structures in continuous representations. Second, we introduce a cognitive forgetting mechanism that simulates the decay patterns of human memory to modulate the weights of historical knowledge. Additionally, we design a prefrontal cortex-inspired knowledge consolidation memory module that leverages temporal tensor stability to gradually consolidate historical knowledge. By integrating these modules, MemEvo achieves strong knowledge retention capabilities in scenarios with a growing number of views. Extensive experiments demonstrate that MemEvo exhibits remarkable advantages over existing state-of-the-art methods.
>
---
#### [new 034] Maize Seedling Detection Dataset (MSDD): A Curated High-Resolution RGB Dataset for Seedling Maize Detection and Benchmarking with YOLOv9, YOLO11, YOLOv12 and Faster-RCNN
- **分类: cs.CV**

- **简介: 该论文提出MSDD数据集，用于玉米幼苗检测任务，解决精准农业中幼苗计数难题。构建了高质量高分辨率数据集，并测试了YOLO系列和Faster-RCNN模型，为自动化农业监测提供基础。**

- **链接: [http://arxiv.org/pdf/2509.15181v1](http://arxiv.org/pdf/2509.15181v1)**

> **作者:** Dewi Endah Kharismawati; Toni Kazic
>
> **备注:** 18 pages, 10 figures, 8 tables. Submitted to IEEE Journal of Selected Topics in Signal Processing (JSTSP) Special Series on Artificial Intelligence for Smart Agriculture
>
> **摘要:** Accurate maize seedling detection is crucial for precision agriculture, yet curated datasets remain scarce. We introduce MSDD, a high-quality aerial image dataset for maize seedling stand counting, with applications in early-season crop monitoring, yield prediction, and in-field management. Stand counting determines how many plants germinated, guiding timely decisions such as replanting or adjusting inputs. Traditional methods are labor-intensive and error-prone, while computer vision enables efficient, accurate detection. MSDD contains three classes-single, double, and triple plants-capturing diverse growth stages, planting setups, soil types, lighting conditions, camera angles, and densities, ensuring robustness for real-world use. Benchmarking shows detection is most reliable during V4-V6 stages and under nadir views. Among tested models, YOLO11 is fastest, while YOLOv9 yields the highest accuracy for single plants. Single plant detection achieves precision up to 0.984 and recall up to 0.873, but detecting doubles and triples remains difficult due to rarity and irregular appearance, often from planting errors. Class imbalance further reduces accuracy in multi-plant detection. Despite these challenges, YOLO11 maintains efficient inference at 35 ms per image, with an additional 120 ms for saving outputs. MSDD establishes a strong foundation for developing models that enhance stand counting, optimize resource allocation, and support real-time decision-making. This dataset marks a step toward automating agricultural monitoring and advancing precision agriculture.
>
---
#### [new 035] MultiEdit: Advancing Instruction-based Image Editing on Diverse and Challenging Tasks
- **分类: cs.CV**

- **简介: 该论文提出MultiEdit数据集，用于改进指令驱动图像编辑任务。针对现有数据集样本少、类型单一及噪声多的问题，构建了包含107K高质量样本的多样化数据集，涵盖多种复杂编辑任务，提升模型在复杂场景下的编辑能力。**

- **链接: [http://arxiv.org/pdf/2509.14638v1](http://arxiv.org/pdf/2509.14638v1)**

> **作者:** Mingsong Li; Lin Liu; Hongjun Wang; Haoxing Chen; Xijun Gu; Shizhan Liu; Dong Gong; Junbo Zhao; Zhenzhong Lan; Jianguo Li
>
> **摘要:** Current instruction-based image editing (IBIE) methods struggle with challenging editing tasks, as both editing types and sample counts of existing datasets are limited. Moreover, traditional dataset construction often contains noisy image-caption pairs, which may introduce biases and limit model capabilities in complex editing scenarios. To address these limitations, we introduce MultiEdit, a comprehensive dataset featuring over 107K high-quality image editing samples. It encompasses 6 challenging editing tasks through a diverse collection of 18 non-style-transfer editing types and 38 style transfer operations, covering a spectrum from sophisticated style transfer to complex semantic operations like person reference editing and in-image text editing. We employ a novel dataset construction pipeline that utilizes two multi-modal large language models (MLLMs) to generate visual-adaptive editing instructions and produce high-fidelity edited images, respectively. Extensive experiments demonstrate that fine-tuning foundational open-source models with our MultiEdit-Train set substantially improves models' performance on sophisticated editing tasks in our proposed MultiEdit-Test benchmark, while effectively preserving their capabilities on the standard editing benchmark. We believe MultiEdit provides a valuable resource for advancing research into more diverse and challenging IBIE capabilities. Our dataset is available at https://huggingface.co/datasets/inclusionAI/MultiEdit.
>
---
#### [new 036] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes
- **分类: cs.CV**

- **简介: 该论文提出一种仅依赖RGB视频的动态场景相机参数优化方法，解决传统方法依赖GT信息且效率低的问题。方法包含三部分：块追踪滤波、异常值感知联合优化和两阶段优化策略，提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2509.15123v1](http://arxiv.org/pdf/2509.15123v1)**

> **作者:** Fang Li; Hao Zhang; Narendra Ahuja
>
> **备注:** NeurIPS 2025
>
> **摘要:** Although COLMAP has long remained the predominant method for camera parameter optimization in static scenes, it is constrained by its lengthy runtime and reliance on ground truth (GT) motion masks for application to dynamic scenes. Many efforts attempted to improve it by incorporating more priors as supervision such as GT focal length, motion masks, 3D point clouds, camera poses, and metric depth, which, however, are typically unavailable in casually captured RGB videos. In this paper, we propose a novel method for more accurate and efficient camera parameter optimization in dynamic scenes solely supervised by a single RGB video. Our method consists of three key components: (1) Patch-wise Tracking Filters, to establish robust and maximally sparse hinge-like relations across the RGB video. (2) Outlier-aware Joint Optimization, for efficient camera parameter optimization by adaptive down-weighting of moving outliers, without reliance on motion priors. (3) A Two-stage Optimization Strategy, to enhance stability and optimization speed by a trade-off between the Softplus limits and convex minima in losses. We visually and numerically evaluate our camera estimates. To further validate accuracy, we feed the camera estimates into a 4D reconstruction method and assess the resulting 3D scenes, and rendered 2D RGB and depth maps. We perform experiments on 4 real-world datasets (NeRF-DS, DAVIS, iPhone, and TUM-dynamics) and 1 synthetic dataset (MPI-Sintel), demonstrating that our method estimates camera parameters more efficiently and accurately with a single RGB video as the only supervision.
>
---
#### [new 037] Enhancing Feature Fusion of U-like Networks with Dynamic Skip Connections
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，解决传统U-like网络中特征融合的静态性和多尺度交互不足问题。提出动态跳跃连接块（DSC），包含测试时训练模块和动态多尺度核模块，提升跨层连接与特征融合能力，适用于多种U-like网络结构。**

- **链接: [http://arxiv.org/pdf/2509.14610v1](http://arxiv.org/pdf/2509.14610v1)**

> **作者:** Yue Cao; Quansong He; Kaishen Wang; Jianlong Xiong; Tao He
>
> **摘要:** U-like networks have become fundamental frameworks in medical image segmentation through skip connections that bridge high-level semantics and low-level spatial details. Despite their success, conventional skip connections exhibit two key limitations: inter-feature constraints and intra-feature constraints. The inter-feature constraint refers to the static nature of feature fusion in traditional skip connections, where information is transmitted along fixed pathways regardless of feature content. The intra-feature constraint arises from the insufficient modeling of multi-scale feature interactions, thereby hindering the effective aggregation of global contextual information. To overcome these limitations, we propose a novel Dynamic Skip Connection (DSC) block that fundamentally enhances cross-layer connectivity through adaptive mechanisms. The DSC block integrates two complementary components. (1) Test-Time Training (TTT) module. This module addresses the inter-feature constraint by enabling dynamic adaptation of hidden representations during inference, facilitating content-aware feature refinement. (2) Dynamic Multi-Scale Kernel (DMSK) module. To mitigate the intra-feature constraint, this module adaptively selects kernel sizes based on global contextual cues, enhancing the network capacity for multi-scale feature integration. The DSC block is architecture-agnostic and can be seamlessly incorporated into existing U-like network structures. Extensive experiments demonstrate the plug-and-play effectiveness of the proposed DSC block across CNN-based, Transformer-based, hybrid CNN-Transformer, and Mamba-based U-like networks.
>
---
#### [new 038] Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding
- **分类: cs.CV**

- **简介: 该论文研究零样本时空视频定位任务，旨在通过多模态大语言模型（MLLMs）实现文本查询到视频时空区域的定位。提出DSTH和TAS策略，提升模型对属性与动作线索的整合能力及时间一致性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.15178v1](http://arxiv.org/pdf/2509.15178v1)**

> **作者:** Zaiquan Yang; Yuhao Liu; Gerhard Hancke; Rynson W. H. Lau
>
> **摘要:** Spatio-temporal video grounding (STVG) aims at localizing the spatio-temporal tube of a video, as specified by the input text query. In this paper, we utilize multimodal large language models (MLLMs) to explore a zero-shot solution in STVG. We reveal two key insights about MLLMs: (1) MLLMs tend to dynamically assign special tokens, referred to as \textit{grounding tokens}, for grounding the text query; and (2) MLLMs often suffer from suboptimal grounding due to the inability to fully integrate the cues in the text query (\textit{e.g.}, attributes, actions) for inference. Based on these insights, we propose a MLLM-based zero-shot framework for STVG, which includes novel decomposed spatio-temporal highlighting (DSTH) and temporal-augmented assembling (TAS) strategies to unleash the reasoning ability of MLLMs. The DSTH strategy first decouples the original query into attribute and action sub-queries for inquiring the existence of the target both spatially and temporally. It then uses a novel logit-guided re-attention (LRA) module to learn latent variables as spatial and temporal prompts, by regularizing token predictions for each sub-query. These prompts highlight attribute and action cues, respectively, directing the model's attention to reliable spatial and temporal related visual regions. In addition, as the spatial grounding by the attribute sub-query should be temporally consistent, we introduce the TAS strategy to assemble the predictions using the original video frames and the temporal-augmented frames as inputs to help improve temporal consistency. We evaluate our method on various MLLMs, and show that it outperforms SOTA methods on three common STVG benchmarks. The code will be available at https://github.com/zaiquanyang/LLaVA_Next_STVG.
>
---
#### [new 039] Class-invariant Test-Time Augmentation for Domain Generalization
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种轻量级测试时增强方法CI-TTA，用于领域泛化任务，解决模型在分布偏移下的性能下降问题。通过生成同类图像变体并聚合预测结果，提升模型在未见领域的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.14420v1](http://arxiv.org/pdf/2509.14420v1)**

> **作者:** Zhicheng Lin; Xiaolin Wu; Xi Zhang
>
> **摘要:** Deep models often suffer significant performance degradation under distribution shifts. Domain generalization (DG) seeks to mitigate this challenge by enabling models to generalize to unseen domains. Most prior approaches rely on multi-domain training or computationally intensive test-time adaptation. In contrast, we propose a complementary strategy: lightweight test-time augmentation. Specifically, we develop a novel Class-Invariant Test-Time Augmentation (CI-TTA) technique. The idea is to generate multiple variants of each input image through elastic and grid deformations that nevertheless belong to the same class as the original input. Their predictions are aggregated through a confidence-guided filtering scheme that remove unreliable outputs, ensuring the final decision relies on consistent and trustworthy cues. Extensive Experiments on PACS and Office-Home datasets demonstrate consistent gains across different DG algorithms and backbones, highlighting the effectiveness and generality of our approach.
>
---
#### [new 040] Feature-aligned Motion Transformation for Efficient Dynamic Point Cloud Compression
- **分类: cs.CV**

- **简介: 该论文提出FMT框架，用于动态点云压缩任务，解决运动估计不准确和效率低的问题。通过时空对齐策略和随机访问参考机制，提升压缩效率与性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14591v1](http://arxiv.org/pdf/2509.14591v1)**

> **作者:** Xuan Deng; Xiandong Meng; Longguang Wang; Tiange Zhang; Xiaopeng Fan; Debin Zhao
>
> **备注:** 9 pages
>
> **摘要:** Dynamic point clouds are widely used in applications such as immersive reality, robotics, and autonomous driving. Efficient compression largely depends on accurate motion estimation and compensation, yet the irregular structure and significant local variations of point clouds make this task highly challenging. Current methods often rely on explicit motion estimation, whose encoded vectors struggle to capture intricate dynamics and fail to fully exploit temporal correlations. To overcome these limitations, we introduce a Feature-aligned Motion Transformation (FMT) framework for dynamic point cloud compression. FMT replaces explicit motion vectors with a spatiotemporal alignment strategy that implicitly models continuous temporal variations, using aligned features as temporal context within a latent-space conditional encoding framework. Furthermore, we design a random access (RA) reference strategy that enables bidirectional motion referencing and layered encoding, thereby supporting frame-level parallel compression. Extensive experiments demonstrate that our method surpasses D-DPCC and AdaDPCC in both encoding and decoding efficiency, while also achieving BD-Rate reductions of 20% and 9.4%, respectively. These results highlight the effectiveness of FMT in jointly improving compression efficiency and processing performance.
>
---
#### [new 041] SPATIALGEN: Layout-guided 3D Indoor Scene Generation
- **分类: cs.CV**

- **简介: 论文提出SPATIALGEN，解决室内场景高质量生成问题。通过构建大规模合成数据集，设计多模态扩散模型，实现基于布局和参考图像的3D场景生成，提升视觉质量与语义一致性。**

- **链接: [http://arxiv.org/pdf/2509.14981v1](http://arxiv.org/pdf/2509.14981v1)**

> **作者:** Chuan Fang; Heng Li; Yixun Liang; Jia Zheng; Yongsen Mao; Yuan Liu; Rui Tang; Zihan Zhou; Ping Tan
>
> **备注:** 3D scene ggeneration; diffusion model; Scene reconstruction and understanding
>
> **摘要:** Creating high-fidelity 3D models of indoor environments is essential for applications in design, virtual reality, and robotics. However, manual 3D modeling remains time-consuming and labor-intensive. While recent advances in generative AI have enabled automated scene synthesis, existing methods often face challenges in balancing visual quality, diversity, semantic consistency, and user control. A major bottleneck is the lack of a large-scale, high-quality dataset tailored to this task. To address this gap, we introduce a comprehensive synthetic dataset, featuring 12,328 structured annotated scenes with 57,440 rooms, and 4.7M photorealistic 2D renderings. Leveraging this dataset, we present SpatialGen, a novel multi-view multi-modal diffusion model that generates realistic and semantically consistent 3D indoor scenes. Given a 3D layout and a reference image (derived from a text prompt), our model synthesizes appearance (color image), geometry (scene coordinate map), and semantic (semantic segmentation map) from arbitrary viewpoints, while preserving spatial consistency across modalities. SpatialGen consistently generates superior results to previous methods in our experiments. We are open-sourcing our data and models to empower the community and advance the field of indoor scene understanding and generation.
>
---
#### [new 042] Edge-Aware Normalized Attention for Efficient and Detail-Preserving Single Image Super-Resolution
- **分类: cs.CV; 68T45, 68T07, 68U10**

- **简介: 该论文属于单图像超分辨率任务，旨在提升低分辨率图像的细节与结构。提出一种边缘感知的归一化注意力机制，通过自适应调制图增强结构显著区域，结合轻量残差设计与多目标损失函数，在保持模型复杂度的同时提升重建质量与感知真实感。**

- **链接: [http://arxiv.org/pdf/2509.14550v1](http://arxiv.org/pdf/2509.14550v1)**

> **作者:** Penghao Rao; Tieyong Zeng
>
> **备注:** 13 pages
>
> **摘要:** Single-image super-resolution (SISR) remains highly ill-posed because recovering structurally faithful high-frequency content from a single low-resolution observation is ambiguous. Existing edge-aware methods often attach edge priors or attention branches onto increasingly complex backbones, yet ad hoc fusion frequently introduces redundancy, unstable optimization, or limited structural gains. We address this gap with an edge-guided attention mechanism that derives an adaptive modulation map from jointly encoded edge features and intermediate feature activations, then applies it to normalize and reweight responses, selectively amplifying structurally salient regions while suppressing spurious textures. In parallel, we integrate this mechanism into a lightweight residual design trained under a composite objective combining pixel-wise, perceptual, and adversarial terms to balance fidelity, perceptual realism, and training stability. Extensive experiments on standard SISR benchmarks demonstrate consistent improvements in structural sharpness and perceptual quality over SRGAN, ESRGAN, and prior edge-attention baselines at comparable model complexity. The proposed formulation provides (i) a parameter-efficient path to inject edge priors, (ii) stabilized adversarial refinement through a tailored multiterm loss, and (iii) enhanced edge fidelity without resorting to deeper or heavily overparameterized architectures. These results highlight the effectiveness of principled edge-conditioned modulation for advancing perceptual super-resolution.
>
---
#### [new 043] LSTC-MDA: A Unified Framework for Long-Short Term Temporal Convolution and Mixed Data Augmentation in Skeleton-Based Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 论文提出LSTC-MDA框架，用于骨骼动作识别任务，解决样本稀缺与时空建模难题。引入LSTC模块处理长短时依赖，结合改进的JMDA增强数据多样性，实验表明其在多个数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.14619v1](http://arxiv.org/pdf/2509.14619v1)**

> **作者:** Feng Ding; Haisheng Fu; Soroush Oraki; Jie Liang
>
> **备注:** Submitted to ICASSP
>
> **摘要:** Skeleton-based action recognition faces two longstanding challenges: the scarcity of labeled training samples and difficulty modeling short- and long-range temporal dependencies. To address these issues, we propose a unified framework, LSTC-MDA, which simultaneously improves temporal modeling and data diversity. We introduce a novel Long-Short Term Temporal Convolution (LSTC) module with parallel short- and long-term branches, these two feature branches are then aligned and fused adaptively using learned similarity weights to preserve critical long-range cues lost by conventional stride-2 temporal convolutions. We also extend Joint Mixing Data Augmentation (JMDA) with an Additive Mixup at the input level, diversifying training samples and restricting mixup operations to the same camera view to avoid distribution shifts. Ablation studies confirm each component contributes. LSTC-MDA achieves state-of-the-art results: 94.1% and 97.5% on NTU 60 (X-Sub and X-View), 90.4% and 92.0% on NTU 120 (X-Sub and X-Set),97.2% on NW-UCLA. Code: https://github.com/xiaobaoxia/LSTC-MDA.
>
---
#### [new 044] AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt
- **分类: cs.CV; cs.CL**

- **简介: 论文提出AIP攻击方法，通过操纵指令提示来影响RAG系统的检索行为，实现隐蔽操控。属于安全攻防任务，解决RAG系统中因依赖外部检索而产生的新型攻击漏洞问题，设计了自然、实用且鲁棒的对抗性指令提示生成策略。**

- **链接: [http://arxiv.org/pdf/2509.15159v1](http://arxiv.org/pdf/2509.15159v1)**

> **作者:** Saket S. Chaturvedi; Gaurav Bagwe; Lan Zhang; Xiaoyong Yuan
>
> **备注:** Accepted at EMNLP 2025 Conference
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly. We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.
>
---
#### [new 045] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决事件相机缺乏密集深度标注数据的问题。提出跨模态蒸馏方法，利用视觉基础模型生成伪标签，并改进模型结构，实现无需昂贵标注的高精度深度估计。**

- **链接: [http://arxiv.org/pdf/2509.15224v1](http://arxiv.org/pdf/2509.15224v1)**

> **作者:** Luca Bartolomei; Enrico Mannocci; Fabio Tosi; Matteo Poggi; Stefano Mattoccia
>
> **备注:** ICCV 2025. Code: https://github.com/bartn8/depthanyevent/ Project Page: https://bartn8.github.io/depthanyevent/
>
> **摘要:** Event cameras capture sparse, high-temporal-resolution visual information, making them particularly suitable for challenging environments with high-speed motion and strongly varying lighting conditions. However, the lack of large datasets with dense ground-truth depth annotations hinders learning-based monocular depth estimation from event data. To address this limitation, we propose a cross-modal distillation paradigm to generate dense proxy labels leveraging a Vision Foundation Model (VFM). Our strategy requires an event stream spatially aligned with RGB frames, a simple setup even available off-the-shelf, and exploits the robustness of large-scale VFMs. Additionally, we propose to adapt VFMs, either a vanilla one like Depth Anything v2 (DAv2), or deriving from it a novel recurrent architecture to infer depth from monocular event cameras. We evaluate our approach with synthetic and real-world datasets, demonstrating that i) our cross-modal paradigm achieves competitive performance compared to fully supervised methods without requiring expensive depth annotations, and ii) our VFM-based models achieve state-of-the-art performance.
>
---
#### [new 046] Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决自回归模型在图像理解中的不足。提出ST-AR框架，通过引入自监督目标提升模型语义理解能力，显著改善生成质量。**

- **链接: [http://arxiv.org/pdf/2509.15185v1](http://arxiv.org/pdf/2509.15185v1)**

> **作者:** Xiaoyu Yue; Zidong Wang; Yuqing Wang; Wenlong Zhang; Xihui Liu; Wanli Ouyang; Lei Bai; Luping Zhou
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent studies have demonstrated the importance of high-quality visual representations in image generation and have highlighted the limitations of generative models in image understanding. As a generative paradigm originally designed for natural language, autoregressive models face similar challenges. In this work, we present the first systematic investigation into the mechanisms of applying the next-token prediction paradigm to the visual domain. We identify three key properties that hinder the learning of high-level visual semantics: local and conditional dependence, inter-step semantic inconsistency, and spatial invariance deficiency. We show that these issues can be effectively addressed by introducing self-supervised objectives during training, leading to a novel training framework, Self-guided Training for AutoRegressive models (ST-AR). Without relying on pre-trained representation models, ST-AR significantly enhances the image understanding ability of autoregressive models and leads to improved generation quality. Specifically, ST-AR brings approximately 42% FID improvement for LlamaGen-L and 49% FID improvement for LlamaGen-XL, while maintaining the same sampling strategy.
>
---
#### [new 047] Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出一种基于扩散模型的轻量且精确的多视角立体视觉方法，用于3D重建。通过条件扩散过程和置信度采样策略，提升深度估计效率与精度，在多个数据集上取得先进性能。**

- **链接: [http://arxiv.org/pdf/2509.15220v1](http://arxiv.org/pdf/2509.15220v1)**

> **作者:** Fangjinhua Wang; Qingshan Xu; Yew-Soon Ong; Marc Pollefeys
>
> **备注:** Accepted to IEEE T-PAMI 2025. Code: https://github.com/cvg/diffmvs
>
> **摘要:** To reconstruct the 3D geometry from calibrated images, learning-based multi-view stereo (MVS) methods typically perform multi-view depth estimation and then fuse depth maps into a mesh or point cloud. To improve the computational efficiency, many methods initialize a coarse depth map and then gradually refine it in higher resolutions. Recently, diffusion models achieve great success in generation tasks. Starting from a random noise, diffusion models gradually recover the sample with an iterative denoising process. In this paper, we propose a novel MVS framework, which introduces diffusion models in MVS. Specifically, we formulate depth refinement as a conditional diffusion process. Considering the discriminative characteristic of depth estimation, we design a condition encoder to guide the diffusion process. To improve efficiency, we propose a novel diffusion network combining lightweight 2D U-Net and convolutional GRU. Moreover, we propose a novel confidence-based sampling strategy to adaptively sample depth hypotheses based on the confidence estimated by diffusion model. Based on our novel MVS framework, we propose two novel MVS methods, DiffMVS and CasDiffMVS. DiffMVS achieves competitive performance with state-of-the-art efficiency in run-time and GPU memory. CasDiffMVS achieves state-of-the-art performance on DTU, Tanks & Temples and ETH3D. Code is available at: https://github.com/cvg/diffmvs.
>
---
#### [new 048] Pseudo-Label Enhanced Cascaded Framework: 2nd Technical Report for LSVOS 2025 VOS Track
- **分类: cs.CV**

- **简介: 论文提出一种伪标签增强的级联框架，用于复杂视频目标分割（VOS）任务，解决小目标、遮挡和快速运动等问题。基于SAM2框架，结合伪标签训练与多模型推理，提升分割精度，在MOSE测试集上取得第二名。**

- **链接: [http://arxiv.org/pdf/2509.14901v1](http://arxiv.org/pdf/2509.14901v1)**

> **作者:** An Yan; Leilei Cao; Feng Lu; Ran Hong; Youhai Jiang; Fengjie Zhu
>
> **摘要:** Complex Video Object Segmentation (VOS) presents significant challenges in accurately segmenting objects across frames, especially in the presence of small and similar targets, frequent occlusions, rapid motion, and complex interactions. In this report, we present our solution for the LSVOS 2025 VOS Track based on the SAM2 framework. We adopt a pseudo-labeling strategy during training: a trained SAM2 checkpoint is deployed within the SAM2Long framework to generate pseudo labels for the MOSE test set, which are then combined with existing data for further training. For inference, the SAM2Long framework is employed to obtain our primary segmentation results, while an open-source SeC model runs in parallel to produce complementary predictions. A cascaded decision mechanism dynamically integrates outputs from both models, exploiting the temporal stability of SAM2Long and the concept-level robustness of SeC. Benefiting from pseudo-label training and cascaded multi-model inference, our approach achieves a J\&F score of 0.8616 on the MOSE test set -- +1.4 points over our SAM2Long baseline -- securing the 2nd place in the LSVOS 2025 VOS Track, and demonstrating strong robustness and accuracy in long, complex video segmentation scenarios.
>
---
#### [new 049] Data Augmentation via Latent Diffusion Models for Detecting Smell-Related Objects in Historical Artworks
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决历史艺术品中气味相关物体识别的难题。通过引入潜在扩散模型生成合成数据，缓解标注稀疏与类别不平衡问题，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.14755v1](http://arxiv.org/pdf/2509.14755v1)**

> **作者:** Ahmed Sheta; Mathias Zinnen; Aline Sindel; Andreas Maier; Vincent Christlein
>
> **备注:** Appeared at the 4th International Workshop on Fine Art Pattern Extraction and Recognition (FAPER 2025), in conjunction with ICIAP 2025; proceedings forthcoming in ICIAP 2025 Workshops (LNCS, Springer)
>
> **摘要:** Finding smell references in historic artworks is a challenging problem. Beyond artwork-specific challenges such as stylistic variations, their recognition demands exceptionally detailed annotation classes, resulting in annotation sparsity and extreme class imbalance. In this work, we explore the potential of synthetic data generation to alleviate these issues and enable accurate detection of smell-related objects. We evaluate several diffusion-based augmentation strategies and demonstrate that incorporating synthetic data into model training can improve detection performance. Our findings suggest that leveraging the large-scale pretraining of diffusion models offers a promising approach for improving detection accuracy, particularly in niche applications where annotations are scarce and costly to obtain. Furthermore, the proposed approach proves to be effective even with relatively small amounts of data, and scaling it up provides high potential for further enhancements.
>
---
#### [new 050] Temporal Representation Learning of Phenotype Trajectories for pCR Prediction in Breast Cancer
- **分类: cs.CV**

- **简介: 论文提出一种学习表型轨迹时序表示的方法，用于预测乳腺癌患者新辅助化疗后的病理完全缓解（pCR）。通过MRI影像数据建模治疗反应动态，利用多任务模型提升预测性能。实验表明，使用多时间点数据可提高分类准确率。属于医疗影像分析中的时序预测任务。**

- **链接: [http://arxiv.org/pdf/2509.14872v1](http://arxiv.org/pdf/2509.14872v1)**

> **作者:** Ivana Janíčková; Yen Y. Tan; Thomas H. Helbich; Konstantin Miloserdov; Zsuzsanna Bago-Horvath; Ulrike Heber; Georg Langs
>
> **摘要:** Effective therapy decisions require models that predict the individual response to treatment. This is challenging since the progression of disease and response to treatment vary substantially across patients. Here, we propose to learn a representation of the early dynamics of treatment response from imaging data to predict pathological complete response (pCR) in breast cancer patients undergoing neoadjuvant chemotherapy (NACT). The longitudinal change in magnetic resonance imaging (MRI) data of the breast forms trajectories in the latent space, serving as basis for prediction of successful response. The multi-task model represents appearance, fosters temporal continuity and accounts for the comparably high heterogeneity in the non-responder cohort.In experiments on the publicly available ISPY-2 dataset, a linear classifier in the latent trajectory space achieves a balanced accuracy of 0.761 using only pre-treatment data (T0), 0.811 using early response (T0 + T1), and 0.861 using four imaging time points (T0 -> T3). The code will be made available upon paper acceptance.
>
---
#### [new 051] HybridMamba: A Dual-domain Mamba for 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 论文提出HybridMamba，用于3D医学图像分割。针对Mamba模型过度关注全局信息而忽略局部结构的问题，设计双域机制融合局部与全局特征，提升分割精度。实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14609v1](http://arxiv.org/pdf/2509.14609v1)**

> **作者:** Weitong Wu; Zhaohu Xing; Jing Gong; Qin Peng; Lei Zhu
>
> **摘要:** In the domain of 3D biomedical image segmentation, Mamba exhibits the superior performance for it addresses the limitations in modeling long-range dependencies inherent to CNNs and mitigates the abundant computational overhead associated with Transformer-based frameworks when processing high-resolution medical volumes. However, attaching undue importance to global context modeling may inadvertently compromise critical local structural information, thus leading to boundary ambiguity and regional distortion in segmentation outputs. Therefore, we propose the HybridMamba, an architecture employing dual complementary mechanisms: 1) a feature scanning strategy that progressively integrates representations both axial-traversal and local-adaptive pathways to harmonize the relationship between local and global representations, and 2) a gated module combining spatial-frequency analysis for comprehensive contextual modeling. Besides, we collect a multi-center CT dataset related to lung cancer. Experiments on MRI and CT datasets demonstrate that HybridMamba significantly outperforms the state-of-the-art methods in 3D medical image segmentation.
>
---
#### [new 052] Radiology Report Conditional 3D CT Generation with Multi Encoder Latent diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出Report2CT，基于多文本编码器的潜在扩散模型，从完整放射报告生成3D CT图像，解决现有方法语义对齐差的问题，提升临床真实性和图像质量，实现医学影像合成任务。**

- **链接: [http://arxiv.org/pdf/2509.14780v1](http://arxiv.org/pdf/2509.14780v1)**

> **作者:** Sina Amirrajab; Zohaib Salahuddin; Sheng Kuang; Henry C. Woodruff; Philippe Lambin
>
> **摘要:** Text to image latent diffusion models have recently advanced medical image synthesis, but applications to 3D CT generation remain limited. Existing approaches rely on simplified prompts, neglecting the rich semantic detail in full radiology reports, which reduces text image alignment and clinical fidelity. We propose Report2CT, a radiology report conditional latent diffusion framework for synthesizing 3D chest CT volumes directly from free text radiology reports, incorporating both findings and impression sections using multiple text encoder. Report2CT integrates three pretrained medical text encoders (BiomedVLP CXR BERT, MedEmbed, and ClinicalBERT) to capture nuanced clinical context. Radiology reports and voxel spacing information condition a 3D latent diffusion model trained on 20000 CT volumes from the CT RATE dataset. Model performance was evaluated using Frechet Inception Distance (FID) for real synthetic distributional similarity and CLIP based metrics for semantic alignment, with additional qualitative and quantitative comparisons against GenerateCT model. Report2CT generated anatomically consistent CT volumes with excellent visual quality and text image alignment. Multi encoder conditioning improved CLIP scores, indicating stronger preservation of fine grained clinical details in the free text radiology reports. Classifier free guidance further enhanced alignment with only a minor trade off in FID. We ranked first in the VLM3D Challenge at MICCAI 2025 on Text Conditional CT Generation and achieved state of the art performance across all evaluation metrics. By leveraging complete radiology reports and multi encoder text conditioning, Report2CT advances 3D CT synthesis, producing clinically faithful and high quality synthetic data.
>
---
#### [new 053] DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images
- **分类: cs.CV**

- **简介: 该论文提出DACoN框架，用于动漫线稿的自动上色任务。解决传统方法在遮挡、姿态变化等问题上的不足，通过融合基础模型与CNN特征，支持任意数量参考图，提升上色效果。**

- **链接: [http://arxiv.org/pdf/2509.14685v1](http://arxiv.org/pdf/2509.14685v1)**

> **作者:** Kazuma Nagata; Naoshi Kaneko
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Automatic colorization of line drawings has been widely studied to reduce the labor cost of hand-drawn anime production. Deep learning approaches, including image/video generation and feature-based correspondence, have improved accuracy but struggle with occlusions, pose variations, and viewpoint changes. To address these challenges, we propose DACoN, a framework that leverages foundation models to capture part-level semantics, even in line drawings. Our method fuses low-resolution semantic features from foundation models with high-resolution spatial features from CNNs for fine-grained yet robust feature extraction. In contrast to previous methods that rely on the Multiplex Transformer and support only one or two reference images, DACoN removes this constraint, allowing any number of references. Quantitative and qualitative evaluations demonstrate the benefits of using multiple reference images, achieving superior colorization performance. Our code and model are available at https://github.com/kzmngt/DACoN.
>
---
#### [new 054] MapAnything: Mapping Urban Assets using Single Street-View Images
- **分类: cs.CV**

- **简介: 论文提出MapAnything模块，通过单张街景图像自动获取城市物体的地理坐标，解决城市数据库更新效率低的问题。利用深度估计模型和几何原理计算距离，验证其在不同场景下的准确性，提升城市资产与事件的自动化测绘能力。**

- **链接: [http://arxiv.org/pdf/2509.14839v1](http://arxiv.org/pdf/2509.14839v1)**

> **作者:** Miriam Louise Carnot; Jonas Kunze; Erik Fastermann; Eric Peukert; André Ludwig; Bogdan Franczyk
>
> **摘要:** To maintain an overview of urban conditions, city administrations manage databases of objects like traffic signs and trees, complete with their geocoordinates. Incidents such as graffiti or road damage are also relevant. As digitization increases, so does the need for more data and up-to-date databases, requiring significant manual effort. This paper introduces MapAnything, a module that automatically determines the geocoordinates of objects using individual images. Utilizing advanced Metric Depth Estimation models, MapAnything calculates geocoordinates based on the object's distance from the camera, geometric principles, and camera specifications. We detail and validate the module, providing recommendations for automating urban object and incident mapping. Our evaluation measures the accuracy of estimated distances against LiDAR point clouds in urban environments, analyzing performance across distance intervals and semantic areas like roads and vegetation. The module's effectiveness is demonstrated through practical use cases involving traffic signs and road damage.
>
---
#### [new 055] Template-Based Cortical Surface Reconstruction with Minimal Energy Deformation
- **分类: cs.CV; cs.AI; cs.LG; q-bio.NC; stat.ML**

- **简介: 该论文属于脑皮层表面重建任务，旨在解决模板变形中的能量优化与训练一致性问题。提出MED损失函数，提升V2C-Flow模型的重建精度与可重复性。**

- **链接: [http://arxiv.org/pdf/2509.14827v1](http://arxiv.org/pdf/2509.14827v1)**

> **作者:** Patrick Madlindl; Fabian Bongratz; Christian Wachinger
>
> **摘要:** Cortical surface reconstruction (CSR) from magnetic resonance imaging (MRI) is fundamental to neuroimage analysis, enabling morphological studies of the cerebral cortex and functional brain mapping. Recent advances in learning-based CSR have dramatically accelerated processing, allowing for reconstructions through the deformation of anatomical templates within seconds. However, ensuring the learned deformations are optimal in terms of deformation energy and consistent across training runs remains a particular challenge. In this work, we design a Minimal Energy Deformation (MED) loss, acting as a regularizer on the deformation trajectories and complementing the widely used Chamfer distance in CSR. We incorporate it into the recent V2C-Flow model and demonstrate considerable improvements in previously neglected training consistency and reproducibility without harming reconstruction accuracy and topological correctness.
>
---
#### [new 056] A Real-Time Multi-Model Parametric Representation of Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种实时多模型点云参数化方法，用于高效表面检测与拟合。针对传统方法在精度与效率上的不足，结合高斯混合模型与曲面拟合，实现更优的鲁棒性与准确性，适用于低功耗实时场景。**

- **链接: [http://arxiv.org/pdf/2509.14773v1](http://arxiv.org/pdf/2509.14773v1)**

> **作者:** Yuan Gao; Wei Dong
>
> **摘要:** In recent years, parametric representations of point clouds have been widely applied in tasks such as memory-efficient mapping and multi-robot collaboration. Highly adaptive models, like spline surfaces or quadrics, are computationally expensive in detection or fitting. In contrast, real-time methods, such as Gaussian mixture models or planes, have low degrees of freedom, making high accuracy with few primitives difficult. To tackle this problem, a multi-model parametric representation with real-time surface detection and fitting is proposed. Specifically, the Gaussian mixture model is first employed to segment the point cloud into multiple clusters. Then, flat clusters are selected and merged into planes or curved surfaces. Planes can be easily fitted and delimited by a 2D voxel-based boundary description method. Surfaces with curvature are fitted by B-spline surfaces and the same boundary description method is employed. Through evaluations on multiple public datasets, the proposed surface detection exhibits greater robustness than the state-of-the-art approach, with 3.78 times improvement in efficiency. Meanwhile, this representation achieves a 2-fold gain in accuracy over Gaussian mixture models, operating at 36.4 fps on a low-power onboard computer.
>
---
#### [new 057] Calibration-Aware Prompt Learning for Medical Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于医学视觉-语言模型的校准任务，旨在解决模型置信度不准确的问题。提出CalibPrompt框架，在提示调优过程中引入校准目标，提升模型校准能力，同时保持准确率。实验表明其有效改进了多个Med-VLMs的校准效果。**

- **链接: [http://arxiv.org/pdf/2509.15226v1](http://arxiv.org/pdf/2509.15226v1)**

> **作者:** Abhishek Basu; Fahad Shamshad; Ashshak Sharifdeen; Karthik Nandakumar; Muhammad Haris Khan
>
> **备注:** Accepted in BMVC 2025
>
> **摘要:** Medical Vision-Language Models (Med-VLMs) have demonstrated remarkable performance across diverse medical imaging tasks by leveraging large-scale image-text pretraining. However, their confidence calibration is largely unexplored, and so remains a significant challenge. As such, miscalibrated predictions can lead to overconfident errors, undermining clinical trust and decision-making reliability. To address this, we introduce CalibPrompt, the first framework to calibrate Med-VLMs during prompt tuning. CalibPrompt optimizes a small set of learnable prompts with carefully designed calibration objectives under scarce labeled data regime. First, we study a regularizer that attempts to align the smoothed accuracy with the predicted model confidences. Second, we introduce an angular separation loss to maximize textual feature proximity toward improving the reliability in confidence estimates of multimodal Med-VLMs. Extensive experiments on four publicly available Med-VLMs and five diverse medical imaging datasets reveal that CalibPrompt consistently improves calibration without drastically affecting clean accuracy. Our code is available at https://github.com/iabh1shekbasu/CalibPrompt.
>
---
#### [new 058] DICE: Diffusion Consensus Equilibrium for Sparse-view CT Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出DICE框架，用于稀疏视角CT图像重建任务。针对传统方法难以捕捉复杂结构的问题，DICE结合扩散模型与数据一致性代理，提升重建质量。实验表明其在不同稀疏视角下均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14566v1](http://arxiv.org/pdf/2509.14566v1)**

> **作者:** Leon Suarez-Rodriguez; Roman Jacome; Romario Gualdron-Hurtado; Ana Mantilla-Dulcey; Henry Arguello
>
> **备注:** 8 pages, 4 figures, confenrence
>
> **摘要:** Sparse-view computed tomography (CT) reconstruction is fundamentally challenging due to undersampling, leading to an ill-posed inverse problem. Traditional iterative methods incorporate handcrafted or learned priors to regularize the solution but struggle to capture the complex structures present in medical images. In contrast, diffusion models (DMs) have recently emerged as powerful generative priors that can accurately model complex image distributions. In this work, we introduce Diffusion Consensus Equilibrium (DICE), a framework that integrates a two-agent consensus equilibrium into the sampling process of a DM. DICE alternates between: (i) a data-consistency agent, implemented through a proximal operator enforcing measurement consistency, and (ii) a prior agent, realized by a DM performing a clean image estimation at each sampling step. By balancing these two complementary agents iteratively, DICE effectively combines strong generative prior capabilities with measurement consistency. Experimental results show that DICE significantly outperforms state-of-the-art baselines in reconstructing high-quality CT images under uniform and non-uniform sparse-view settings of 15, 30, and 60 views (out of a total of 180), demonstrating both its effectiveness and robustness.
>
---
#### [new 059] AToken: A Unified Tokenizer for Vision
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文提出AToken，一种统一的视觉分词器，解决多模态视觉任务中重建与理解分离的问题。其通过4D共享空间和纯Transformer架构，实现图像、视频和3D资产的高质量重建与语义理解，支持生成与理解任务，推动下一代多模态AI系统发展。**

- **链接: [http://arxiv.org/pdf/2509.14476v1](http://arxiv.org/pdf/2509.14476v1)**

> **作者:** Jiasen Lu; Liangchen Song; Mingze Xu; Byeongjoo Ahn; Yanjun Wang; Chen Chen; Afshin Dehghan; Yinfei Yang
>
> **备注:** 30 pages, 14 figures
>
> **摘要:** We present AToken, the first unified visual tokenizer that achieves both high-fidelity reconstruction and semantic understanding across images, videos, and 3D assets. Unlike existing tokenizers that specialize in either reconstruction or understanding for single modalities, AToken encodes these diverse visual inputs into a shared 4D latent space, unifying both tasks and modalities in a single framework. Specifically, we introduce a pure transformer architecture with 4D rotary position embeddings to process visual inputs of arbitrary resolutions and temporal durations. To ensure stable training, we introduce an adversarial-free training objective that combines perceptual and Gram matrix losses, achieving state-of-the-art reconstruction quality. By employing a progressive training curriculum, AToken gradually expands from single images, videos, and 3D, and supports both continuous and discrete latent tokens. AToken achieves 0.21 rFID with 82.2% ImageNet accuracy for images, 3.01 rFVD with 32.6% MSRVTT retrieval for videos, and 28.19 PSNR with 90.9% classification accuracy for 3D. In downstream applications, AToken enables both visual generation tasks (e.g., image generation with continuous and discrete tokens, text-to-video generation, image-to-3D synthesis) and understanding tasks (e.g., multimodal LLMs), achieving competitive performance across all benchmarks. These results shed light on the next-generation multimodal AI systems built upon unified visual tokenization.
>
---
#### [new 060] Semi-Supervised 3D Medical Segmentation from 2D Natural Images Pretrained Model
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于3D医学图像分割任务，解决标注数据稀缺问题。提出M&N框架，通过2D预训练模型知识蒸馏和伪标签共训练，提升分割性能，适用于多种模型结构。**

- **链接: [http://arxiv.org/pdf/2509.15167v1](http://arxiv.org/pdf/2509.15167v1)**

> **作者:** Pak-Hei Yeung; Jayroop Ramesh; Pengfei Lyu; Ana Namburete; Jagath Rajapakse
>
> **备注:** Machine Learning in Medical Imaging (MLMI) 2025 Oral
>
> **摘要:** This paper explores the transfer of knowledge from general vision models pretrained on 2D natural images to improve 3D medical image segmentation. We focus on the semi-supervised setting, where only a few labeled 3D medical images are available, along with a large set of unlabeled images. To tackle this, we propose a model-agnostic framework that progressively distills knowledge from a 2D pretrained model to a 3D segmentation model trained from scratch. Our approach, M&N, involves iterative co-training of the two models using pseudo-masks generated by each other, along with our proposed learning rate guided sampling that adaptively adjusts the proportion of labeled and unlabeled data in each training batch to align with the models' prediction accuracy and stability, minimizing the adverse effect caused by inaccurate pseudo-masks. Extensive experiments on multiple publicly available datasets demonstrate that M&N achieves state-of-the-art performance, outperforming thirteen existing semi-supervised segmentation approaches under all different settings. Importantly, ablation studies show that M&N remains model-agnostic, allowing seamless integration with different architectures. This ensures its adaptability as more advanced models emerge. The code is available at https://github.com/pakheiyeung/M-N.
>
---
#### [new 061] Brain-HGCN: A Hyperbolic Graph Convolutional Network for Brain Functional Network Analysis
- **分类: cs.CV**

- **简介: 该论文提出Brain-HGCN，一种基于双曲几何的图卷积网络，用于脑功能网络分析。旨在解决欧氏GNN在建模大脑层级结构时的高失真问题，通过双曲空间建模提升精神疾病分类性能。**

- **链接: [http://arxiv.org/pdf/2509.14965v1](http://arxiv.org/pdf/2509.14965v1)**

> **作者:** Junhao Jia; Yunyou Liu; Cheng Yang; Yifei Sun; Feiwei Qin; Changmiao Wang; Yong Peng
>
> **摘要:** Functional magnetic resonance imaging (fMRI) provides a powerful non-invasive window into the brain's functional organization by generating complex functional networks, typically modeled as graphs. These brain networks exhibit a hierarchical topology that is crucial for cognitive processing. However, due to inherent spatial constraints, standard Euclidean GNNs struggle to represent these hierarchical structures without high distortion, limiting their clinical performance. To address this limitation, we propose Brain-HGCN, a geometric deep learning framework based on hyperbolic geometry, which leverages the intrinsic property of negatively curved space to model the brain's network hierarchy with high fidelity. Grounded in the Lorentz model, our model employs a novel hyperbolic graph attention layer with a signed aggregation mechanism to distinctly process excitatory and inhibitory connections, ultimately learning robust graph-level representations via a geometrically sound Fr\'echet mean for graph readout. Experiments on two large-scale fMRI datasets for psychiatric disorder classification demonstrate that our approach significantly outperforms a wide range of state-of-the-art Euclidean baselines. This work pioneers a new geometric deep learning paradigm for fMRI analysis, highlighting the immense potential of hyperbolic GNNs in the field of computational psychiatry.
>
---
#### [new 062] MedFact-R1: Towards Factual Medical Reasoning via Pseudo-Label Augmentation
- **分类: cs.CV**

- **简介: 该论文提出MEDFACT-R1框架，解决医学视觉-语言模型的事实推理问题。通过伪标签微调与强化学习结合，提升事实准确性，在三个基准上取得22.5%的绝对提升。**

- **链接: [http://arxiv.org/pdf/2509.15154v1](http://arxiv.org/pdf/2509.15154v1)**

> **作者:** Gengliang Li; Rongyu Chen; Bin Li; Linlin Yang; Guodong Ding
>
> **备注:** Tech report
>
> **摘要:** Ensuring factual consistency and reliable reasoning remains a critical challenge for medical vision-language models. We introduce MEDFACT-R1, a two-stage framework that integrates external knowledge grounding with reinforcement learning to improve the factual medical reasoning. The first stage uses pseudo-label supervised fine-tuning (SFT) to incorporate external factual expertise; while the second stage applies Group Relative Policy Optimization (GRPO) with four tailored factual reward signals to encourage self-consistent reasoning. Across three public medical QA benchmarks, MEDFACT-R1 delivers up to 22.5% absolute improvement in factual accuracy over previous state-of-the-art methods. Ablation studies highlight the necessity of pseudo-label SFT cold start and validate the contribution of each GRPO reward, underscoring the synergy between knowledge grounding and RL-driven reasoning for trustworthy medical AI. Codes are released at https://github.com/Garfieldgengliang/MEDFACT-R1.
>
---
#### [new 063] GenKOL: Modular Generative AI Framework For Scalable Virtual KOL Generation
- **分类: cs.CV**

- **简介: 该论文提出GenKOL，一种模块化生成AI框架，用于高效生成虚拟KOL图像，解决传统KOL合作成本高、流程复杂的问题。系统集成服装生成、化妆迁移等功能，支持本地或云端灵活部署，提升品牌内容生产效率。**

- **链接: [http://arxiv.org/pdf/2509.14927v1](http://arxiv.org/pdf/2509.14927v1)**

> **作者:** Tan-Hiep To; Duy-Khang Nguyen; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** Key Opinion Leader (KOL) play a crucial role in modern marketing by shaping consumer perceptions and enhancing brand credibility. However, collaborating with human KOLs often involves high costs and logistical challenges. To address this, we present GenKOL, an interactive system that empowers marketing professionals to efficiently generate high-quality virtual KOL images using generative AI. GenKOL enables users to dynamically compose promotional visuals through an intuitive interface that integrates multiple AI capabilities, including garment generation, makeup transfer, background synthesis, and hair editing. These capabilities are implemented as modular, interchangeable services that can be deployed flexibly on local machines or in the cloud. This modular architecture ensures adaptability across diverse use cases and computational environments. Our system can significantly streamline the production of branded content, lowering costs and accelerating marketing workflows through scalable virtual KOL creation.
>
---
#### [new 064] Controllable Localized Face Anonymization Via Diffusion Inpainting
- **分类: cs.CV**

- **简介: 该论文提出一种基于扩散修复的可控人脸匿名化方法，解决在保护个人身份的同时保持图像可用性的问题。通过设计自适应属性引导模块，实现对目标面部属性的精准控制与局部区域保留，无需额外训练即可优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14866v1](http://arxiv.org/pdf/2509.14866v1)**

> **作者:** Ali Salar; Qing Liu; Guoying Zhao
>
> **摘要:** The growing use of portrait images in computer vision highlights the need to protect personal identities. At the same time, anonymized images must remain useful for downstream computer vision tasks. In this work, we propose a unified framework that leverages the inpainting ability of latent diffusion models to generate realistic anonymized images. Unlike prior approaches, we have complete control over the anonymization process by designing an adaptive attribute-guidance module that applies gradient correction during the reverse denoising process, aligning the facial attributes of the generated image with those of the synthesized target image. Our framework also supports localized anonymization, allowing users to specify which facial regions are left unchanged. Extensive experiments conducted on the public CelebA-HQ and FFHQ datasets show that our method outperforms state-of-the-art approaches while requiring no additional model training. The source code is available on our page.
>
---
#### [new 065] MARIC: Multi-Agent Reasoning for Image Classification
- **分类: cs.CV; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出MARIC框架，用于图像分类任务。旨在解决传统模型依赖大量标注数据及单次表征不足的问题。通过多智能体协作，分解任务为全局分析、细粒度描述与合成推理，提升分类性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.14860v1](http://arxiv.org/pdf/2509.14860v1)**

> **作者:** Wonduk Seo; Minhyeong Yu; Hyunjin An; Seunghyun Lee
>
> **备注:** Preprint
>
> **摘要:** Image classification has traditionally relied on parameter-intensive model training, requiring large-scale annotated datasets and extensive fine tuning to achieve competitive performance. While recent vision language models (VLMs) alleviate some of these constraints, they remain limited by their reliance on single pass representations, often failing to capture complementary aspects of visual content. In this paper, we introduce Multi Agent based Reasoning for Image Classification (MARIC), a multi agent framework that reformulates image classification as a collaborative reasoning process. MARIC first utilizes an Outliner Agent to analyze the global theme of the image and generate targeted prompts. Based on these prompts, three Aspect Agents extract fine grained descriptions along distinct visual dimensions. Finally, a Reasoning Agent synthesizes these complementary outputs through integrated reflection step, producing a unified representation for classification. By explicitly decomposing the task into multiple perspectives and encouraging reflective synthesis, MARIC mitigates the shortcomings of both parameter-heavy training and monolithic VLM reasoning. Experiments on 4 diverse image classification benchmark datasets demonstrate that MARIC significantly outperforms baselines, highlighting the effectiveness of multi-agent visual reasoning for robust and interpretable image classification.
>
---
#### [new 066] Leveraging Geometric Visual Illusions as Perceptual Inductive Biases for Vision Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，旨在提升模型对复杂轮廓和纹理的感知能力。通过引入几何视觉错觉作为辅助监督，改进CNN和Transformer模型的结构敏感性，提升其泛化性能。**

- **链接: [http://arxiv.org/pdf/2509.15156v1](http://arxiv.org/pdf/2509.15156v1)**

> **作者:** Haobo Yang; Minghao Guo; Dequan Yang; Wenyu Wang
>
> **摘要:** Contemporary deep learning models have achieved impressive performance in image classification by primarily leveraging statistical regularities within large datasets, but they rarely incorporate structured insights drawn directly from perceptual psychology. To explore the potential of perceptually motivated inductive biases, we propose integrating classic geometric visual illusions well-studied phenomena from human perception into standard image-classification training pipelines. Specifically, we introduce a synthetic, parametric geometric-illusion dataset and evaluate three multi-source learning strategies that combine illusion recognition tasks with ImageNet classification objectives. Our experiments reveal two key conceptual insights: (i) incorporating geometric illusions as auxiliary supervision systematically improves generalization, especially in visually challenging cases involving intricate contours and fine textures; and (ii) perceptually driven inductive biases, even when derived from synthetic stimuli traditionally considered unrelated to natural image recognition, can enhance the structural sensitivity of both CNN and transformer-based architectures. These results demonstrate a novel integration of perceptual science and machine learning and suggest new directions for embedding perceptual priors into vision model design.
>
---
#### [new 067] AutoEdit: Automatic Hyperparameter Tuning for Image Editing
- **分类: cs.CV**

- **简介: 该论文提出AutoEdit，通过强化学习框架自动优化图像编辑中的超参数，解决传统方法依赖人工调参导致的高计算成本问题。属于图像编辑任务，旨在提升扩散模型编辑效率与实用性。**

- **链接: [http://arxiv.org/pdf/2509.15031v1](http://arxiv.org/pdf/2509.15031v1)**

> **作者:** Chau Pham; Quan Dao; Mahesh Bhosale; Yunjie Tian; Dimitris Metaxas; David Doermann
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Recent advances in diffusion models have revolutionized text-guided image editing, yet existing editing methods face critical challenges in hyperparameter identification. To get the reasonable editing performance, these methods often require the user to brute-force tune multiple interdependent hyperparameters, such as inversion timesteps and attention modification, \textit{etc.} This process incurs high computational costs due to the huge hyperparameter search space. We consider searching optimal editing's hyperparameters as a sequential decision-making task within the diffusion denoising process. Specifically, we propose a reinforcement learning framework, which establishes a Markov Decision Process that dynamically adjusts hyperparameters across denoising steps, integrating editing objectives into a reward function. The method achieves time efficiency through proximal policy optimization while maintaining optimal hyperparameter configurations. Experiments demonstrate significant reduction in search time and computational overhead compared to existing brute-force approaches, advancing the practical deployment of a diffusion-based image editing framework in the real world.
>
---
#### [new 068] Do Vision-Language Models See Urban Scenes as People Do? An Urban Perception Benchmark
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一个城市感知基准，评估视觉-语言模型对城市场景的理解。通过对比人类标注与模型表现，研究模型在客观属性与主观评价上的对齐程度，推动参与式城市分析的可重复评估。**

- **链接: [http://arxiv.org/pdf/2509.14574v1](http://arxiv.org/pdf/2509.14574v1)**

> **作者:** Rashid Mushkani
>
> **摘要:** Understanding how people read city scenes can inform design and planning. We introduce a small benchmark for testing vision-language models (VLMs) on urban perception using 100 Montreal street images, evenly split between photographs and photorealistic synthetic scenes. Twelve participants from seven community groups supplied 230 annotation forms across 30 dimensions mixing physical attributes and subjective impressions. French responses were normalized to English. We evaluated seven VLMs in a zero-shot setup with a structured prompt and deterministic parser. We use accuracy for single-choice items and Jaccard overlap for multi-label items; human agreement uses Krippendorff's alpha and pairwise Jaccard. Results suggest stronger model alignment on visible, objective properties than subjective appraisals. The top system (claude-sonnet) reaches macro 0.31 and mean Jaccard 0.48 on multi-label items. Higher human agreement coincides with better model scores. Synthetic images slightly lower scores. We release the benchmark, prompts, and harness for reproducible, uncertainty-aware evaluation in participatory urban analysis.
>
---
#### [new 069] M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出M4Diffuser框架，用于解决移动机械臂在非结构化环境中的鲁棒操作问题。通过多视角扩散策略生成任务目标，并结合ReM-QP控制器实现高效、可操控的移动操作，提升成功率与安全性。**

- **链接: [http://arxiv.org/pdf/2509.14980v1](http://arxiv.org/pdf/2509.14980v1)**

> **作者:** Ju Dong; Lei Zhang; Liding Zhang; Yao Ling; Yu Fu; Kaixin Bai; Zoltán-Csaba Márton; Zhenshan Bing; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** Project page: https://sites.google.com/view/m4diffuser, 10 pages, 9 figures
>
> **摘要:** Mobile manipulation requires the coordinated control of a mobile base and a robotic arm while simultaneously perceiving both global scene context and fine-grained object details. Existing single-view approaches often fail in unstructured environments due to limited fields of view, exploration, and generalization abilities. Moreover, classical controllers, although stable, struggle with efficiency and manipulability near singularities. To address these challenges, we propose M4Diffuser, a hybrid framework that integrates a Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP (ReM-QP) controller for mobile manipulation. The diffusion policy leverages proprioceptive states and complementary camera perspectives with both close-range object details and global scene context to generate task-relevant end-effector goals in the world frame. These high-level goals are then executed by the ReM-QP controller, which eliminates slack variables for computational efficiency and incorporates manipulability-aware preferences for robustness near singularities. Comprehensive experiments in simulation and real-world environments show that M4Diffuser achieves 7 to 56 percent higher success rates and reduces collisions by 3 to 31 percent over baselines. Our approach demonstrates robust performance for smooth whole-body coordination, and strong generalization to unseen tasks, paving the way for reliable mobile manipulation in unstructured environments. Details of the demo and supplemental material are available on our project website https://sites.google.com/view/m4diffuser.
>
---
#### [new 070] RLBind: Adversarial-Invariant Cross-Modal Alignment for Unified Robust Embeddings
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RLBind框架，解决多模态统一嵌入在对抗攻击下的鲁棒性问题。通过两阶段对齐策略，提升视觉编码器的抗干扰能力，并增强跨模态一致性，实现更安全可靠的机器人感知系统。**

- **链接: [http://arxiv.org/pdf/2509.14383v1](http://arxiv.org/pdf/2509.14383v1)**

> **作者:** Yuhong Lu
>
> **备注:** This paper is submitted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Unified multi-modal encoders that bind vision, audio, and other sensors into a shared embedding space are attractive building blocks for robot perception and decision-making. However, on-robot deployment exposes the vision branch to adversarial and natural corruptions, making robustness a prerequisite for safety. Prior defenses typically align clean and adversarial features within CLIP-style encoders and overlook broader cross-modal correspondence, yielding modest gains and often degrading zero-shot transfer. We introduce RLBind, a two-stage adversarial-invariant cross-modal alignment framework for robust unified embeddings. Stage 1 performs unsupervised fine-tuning on clean-adversarial pairs to harden the visual encoder. Stage 2 leverages cross-modal correspondence by minimizing the discrepancy between clean/adversarial features and a text anchor, while enforcing class-wise distributional alignment across modalities. Extensive experiments on Image, Audio, Thermal, and Video data show that RLBind consistently outperforms the LanguageBind backbone and standard fine-tuning baselines in both clean accuracy and norm-bounded adversarial robustness. By improving resilience without sacrificing generalization, RLBind provides a practical path toward safer multi-sensor perception stacks for embodied robots in navigation, manipulation, and other autonomy settings.
>
---
#### [new 071] Learning Mechanistic Subtypes of Neurodegeneration with a Physics-Informed Variational Autoencoder Mixture Model
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 论文提出一种结合物理方程的变分自编码器混合模型，用于从神经影像数据中学习神经退行性疾病的机制亚型。任务是识别不同亚型的潜在动态机制，解决传统方法仅考虑单一PDE导致的模型不准确问题。**

- **链接: [http://arxiv.org/pdf/2509.15124v1](http://arxiv.org/pdf/2509.15124v1)**

> **作者:** Sanduni Pinnawala; Annabelle Hartanto; Ivor J. A. Simpson; Peter A. Wijeratne
>
> **备注:** 13 pages, 5 figures, accepted at SASHIMI workshop, MICCAI 2025
>
> **摘要:** Modelling the underlying mechanisms of neurodegenerative diseases demands methods that capture heterogeneous and spatially varying dynamics from sparse, high-dimensional neuroimaging data. Integrating partial differential equation (PDE) based physics knowledge with machine learning provides enhanced interpretability and utility over classic numerical methods. However, current physics-integrated machine learning methods are limited to considering a single PDE, severely limiting their application to diseases where multiple mechanisms are responsible for different groups (i.e., subtypes) and aggravating problems with model misspecification and degeneracy. Here, we present a deep generative model for learning mixtures of latent dynamic models governed by physics-based PDEs, going beyond traditional approaches that assume a single PDE structure. Our method integrates reaction-diffusion PDEs within a variational autoencoder (VAE) mixture model framework, supporting inference of subtypes of interpretable latent variables (e.g. diffusivity and reaction rates) from neuroimaging data. We evaluate our method on synthetic benchmarks and demonstrate its potential for uncovering mechanistic subtypes of Alzheimer's disease progression from positron emission tomography (PET) data.
>
---
#### [new 072] Communication Efficient Split Learning of ViTs with Attention-based Double Compression
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文提出一种高效的Split Learning框架ADC，用于减少ViT模型训练中的通信开销。通过注意力机制合并相似激活和去除无意义token，实现通信压缩，提升效率且保持高精度。属于联邦学习中的通信优化任务。**

- **链接: [http://arxiv.org/pdf/2509.15058v1](http://arxiv.org/pdf/2509.15058v1)**

> **作者:** Federico Alvetreti; Jary Pomponi; Paolo Di Lorenzo; Simone Scardapane
>
> **摘要:** This paper proposes a novel communication-efficient Split Learning (SL) framework, named Attention-based Double Compression (ADC), which reduces the communication overhead required for transmitting intermediate Vision Transformers activations during the SL training process. ADC incorporates two parallel compression strategies. The first one merges samples' activations that are similar, based on the average attention score calculated in the last client layer; this strategy is class-agnostic, meaning that it can also merge samples having different classes, without losing generalization ability nor decreasing final results. The second strategy follows the first and discards the least meaningful tokens, further reducing the communication cost. Combining these strategies not only allows for sending less during the forward pass, but also the gradients are naturally compressed, allowing the whole model to be trained without additional tuning or approximations of the gradients. Simulation results demonstrate that Attention-based Double Compression outperforms state-of-the-art SL frameworks by significantly reducing communication overheads while maintaining high accuracy.
>
---
#### [new 073] Designing Latent Safety Filters using Pre-Trained Vision Models
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文研究如何利用预训练视觉模型设计视觉安全过滤器，以提升基于视觉的控制系统的安全性。论文探讨了不同训练策略的效果，并评估了模型在资源受限设备上的部署可行性。属于机器人控制安全领域任务。**

- **链接: [http://arxiv.org/pdf/2509.14758v1](http://arxiv.org/pdf/2509.14758v1)**

> **作者:** Ihab Tabbara; Yuxuan Yang; Ahmad Hamzeh; Maxwell Astafyev; Hussein Sibai
>
> **摘要:** Ensuring safety of vision-based control systems remains a major challenge hindering their deployment in critical settings. Safety filters have gained increased interest as effective tools for ensuring the safety of classical control systems, but their applications in vision-based control settings have so far been limited. Pre-trained vision models (PVRs) have been shown to be effective perception backbones for control in various robotics domains. In this paper, we are interested in examining their effectiveness when used for designing vision-based safety filters. We use them as backbones for classifiers defining failure sets, for Hamilton-Jacobi (HJ) reachability-based safety filters, and for latent world models. We discuss the trade-offs between training from scratch, fine-tuning, and freezing the PVRs when training the models they are backbones for. We also evaluate whether one of the PVRs is superior across all tasks, evaluate whether learned world models or Q-functions are better for switching decisions to safe policies, and discuss practical considerations for deploying these PVRs on resource-constrained devices.
>
---
#### [new 074] From Pixels to Urban Policy-Intelligence: Recovering Legacy Effects of Redlining with a Multimodal LLM
- **分类: cs.CY; cs.CV**

- **简介: 论文利用多模态大语言模型（MLLM）分析街景图像，推断社区贫困和树木覆盖率，评估1930年代红lining政策的遗留影响。属于政策评估任务，解决传统方法难以捕捉复杂环境效应的问题，验证了MLLM在城市测量中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.15132v1](http://arxiv.org/pdf/2509.15132v1)**

> **作者:** Anthony Howell; Nancy Wu; Sharmistha Bagchi; Yushim Kim; Chayn Sun
>
> **摘要:** This paper shows how a multimodal large language model (MLLM) can expand urban measurement capacity and support tracking of place-based policy interventions. Using a structured, reason-then-estimate pipeline on street-view imagery, GPT-4o infers neighborhood poverty and tree canopy, which we embed in a quasi-experimental design evaluating the legacy of 1930s redlining. GPT-4o recovers the expected adverse socio-environmental legacy effects of redlining, with estimates statistically indistinguishable from authoritative sources, and it outperforms a conventional pixel-based segmentation baseline-consistent with the idea that holistic scene reasoning extracts higher-order information beyond object counts alone. These results position MLLMs as policy-grade instruments for neighborhood measurement and motivate broader validation across policy-evaluation settings.
>
---
#### [new 075] Doppler Radiance Field-Guided Antenna Selection for Improved Generalization in Multi-Antenna Wi-Fi-based Human Activity Recognition
- **分类: eess.SP; cs.CV**

- **简介: 论文提出基于多天线Wi-Fi的DoRF引导天线选择方法，解决HAR中CSI噪声和异步时钟影响问题，提升模型泛化能力。属于Wi-Fi感知任务，通过优化天线选择提高手势识别性能。**

- **链接: [http://arxiv.org/pdf/2509.15129v1](http://arxiv.org/pdf/2509.15129v1)**

> **作者:** Navid Hasanzadeh; Shahrokh Valaee
>
> **摘要:** With the IEEE 802.11bf Task Group introducing amendments to the WLAN standard for advanced sensing, interest in using Wi-Fi Channel State Information (CSI) for remote sensing has surged. Recent findings indicate that learning a unified three-dimensional motion representation through Doppler Radiance Fields (DoRFs) derived from CSI significantly improves the generalization capabilities of Wi-Fi-based human activity recognition (HAR). Despite this progress, CSI signals remain affected by asynchronous access point (AP) clocks and additive noise from environmental and hardware sources. Consequently, even with existing preprocessing techniques, both the CSI data and Doppler velocity projections used in DoRFs are still susceptible to noise and outliers, limiting HAR performance. To address this challenge, we propose a novel framework for multi-antenna APs to suppress noise and identify the most informative antennas based on DoRF fitting errors, which capture inconsistencies among Doppler velocity projections. Experimental results on a challenging small-scale hand gesture recognition dataset demonstrate that the proposed DoRF-guided Wi-Fi-based HAR approach significantly improves generalization capability, paving the way for robust real-world sensing deployments.
>
---
#### [new 076] One-step Multi-view Clustering With Adaptive Low-rank Anchor-graph Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多视角聚类任务，旨在解决现有方法中冗余信息和噪声干扰问题。提出OMCAL方法，通过自适应低秩锚图学习和统一框架提升聚类效果与效率。**

- **链接: [http://arxiv.org/pdf/2509.14724v1](http://arxiv.org/pdf/2509.14724v1)**

> **作者:** Zhiyuan Xue; Ben Yang; Xuetao Zhang; Fei Wang; Zhiping Lin
>
> **备注:** 13 pages, 7 figures, journal article. Accepted by IEEE Transactions on Multimedia, not yet published online
>
> **摘要:** In light of their capability to capture structural information while reducing computing complexity, anchor graph-based multi-view clustering (AGMC) methods have attracted considerable attention in large-scale clustering problems. Nevertheless, existing AGMC methods still face the following two issues: 1) They directly embedded diverse anchor graphs into a consensus anchor graph (CAG), and hence ignore redundant information and numerous noises contained in these anchor graphs, leading to a decrease in clustering effectiveness; 2) They drop effectiveness and efficiency due to independent post-processing to acquire clustering indicators. To overcome the aforementioned issues, we deliver a novel one-step multi-view clustering method with adaptive low-rank anchor-graph learning (OMCAL). To construct a high-quality CAG, OMCAL provides a nuclear norm-based adaptive CAG learning model against information redundancy and noise interference. Then, to boost clustering effectiveness and efficiency substantially, we incorporate category indicator acquisition and CAG learning into a unified framework. Numerous studies conducted on ordinary and large-scale datasets indicate that OMCAL outperforms existing state-of-the-art methods in terms of clustering effectiveness and efficiency.
>
---
#### [new 077] Two Web Toolkits for Multimodal Piano Performance Dataset Acquisition and Fingering Annotation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 论文提出两个网络工具包PiaRec和ASDF，用于多模态钢琴表演数据的采集与指法标注，旨在解决大规模多模态数据获取困难的问题，提升数据集构建效率。**

- **链接: [http://arxiv.org/pdf/2509.15222v1](http://arxiv.org/pdf/2509.15222v1)**

> **作者:** Junhyung Park; Yonghyun Kim; Joonhyung Bae; Kirak Kim; Taegyun Kwon; Alexander Lerch; Juhan Nam
>
> **备注:** Accepted to the Late-Breaking Demo Session of the 26th International Society for Music Information Retrieval (ISMIR) Conference, 2025
>
> **摘要:** Piano performance is a multimodal activity that intrinsically combines physical actions with the acoustic rendition. Despite growing research interest in analyzing the multimodal nature of piano performance, the laborious process of acquiring large-scale multimodal data remains a significant bottleneck, hindering further progress in this field. To overcome this barrier, we present an integrated web toolkit comprising two graphical user interfaces (GUIs): (i) PiaRec, which supports the synchronized acquisition of audio, video, MIDI, and performance metadata. (ii) ASDF, which enables the efficient annotation of performer fingering from the visual data. Collectively, this system can streamline the acquisition of multimodal piano performance datasets.
>
---
#### [new 078] QuizRank: Picking Images by Quizzing VLMs
- **分类: cs.HC; cs.CV**

- **简介: 该论文提出QuizRank方法，利用VLMs通过生成和回答多选题来评估并排序图像，解决维基百科中图像选择不优的问题。通过对比目标与干扰概念提升区分度，验证了VLMs作为视觉评估工具的有效性。属于图像选择与评估任务。**

- **链接: [http://arxiv.org/pdf/2509.15059v1](http://arxiv.org/pdf/2509.15059v1)**

> **作者:** Tenghao Ji; Eytan Adar
>
> **摘要:** Images play a vital role in improving the readability and comprehension of Wikipedia articles by serving as `illustrative aids.' However, not all images are equally effective and not all Wikipedia editors are trained in their selection. We propose QuizRank, a novel method of image selection that leverages large language models (LLMs) and vision language models (VLMs) to rank images as learning interventions. Our approach transforms textual descriptions of the article's subject into multiple-choice questions about important visual characteristics of the concept. We utilize these questions to quiz the VLM: the better an image can help answer questions, the higher it is ranked. To further improve discrimination between visually similar items, we introduce a Contrastive QuizRank that leverages differences in the features of target (e.g., a Western Bluebird) and distractor concepts (e.g., Mountain Bluebird) to generate questions. We demonstrate the potential of VLMs as effective visual evaluators by showing a high congruence with human quiz-takers and an effective discriminative ranking of images.
>
---
#### [new 079] WorldForge: Unlocking Emergent 3D/4D Generation in Video Diffusion Model via Training-Free Guidance
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文提出WorldForge框架，解决视频扩散模型在3D/4D生成中的控制不足与几何不一致问题。通过三个无训练模块实现精确轨迹注入与运动控制，提升生成效果。属于可控视频生成任务。**

- **链接: [http://arxiv.org/pdf/2509.15130v1](http://arxiv.org/pdf/2509.15130v1)**

> **作者:** Chenxi Song; Yanming Yang; Tong Zhao; Ruibo Li; Chi Zhang
>
> **备注:** Project Webpage: https://worldforge-agi.github.io/
>
> **摘要:** Recent video diffusion models demonstrate strong potential in spatial intelligence tasks due to their rich latent world priors. However, this potential is hindered by their limited controllability and geometric inconsistency, creating a gap between their strong priors and their practical use in 3D/4D tasks. As a result, current approaches often rely on retraining or fine-tuning, which risks degrading pretrained knowledge and incurs high computational costs. To address this, we propose WorldForge, a training-free, inference-time framework composed of three tightly coupled modules. Intra-Step Recursive Refinement introduces a recursive refinement mechanism during inference, which repeatedly optimizes network predictions within each denoising step to enable precise trajectory injection. Flow-Gated Latent Fusion leverages optical flow similarity to decouple motion from appearance in the latent space and selectively inject trajectory guidance into motion-related channels. Dual-Path Self-Corrective Guidance compares guided and unguided denoising paths to adaptively correct trajectory drift caused by noisy or misaligned structural signals. Together, these components inject fine-grained, trajectory-aligned guidance without training, achieving both accurate motion control and photorealistic content generation. Extensive experiments across diverse benchmarks validate our method's superiority in realism, trajectory consistency, and visual fidelity. This work introduces a novel plug-and-play paradigm for controllable video synthesis, offering a new perspective on leveraging generative priors for spatial intelligence.
>
---
#### [new 080] Forecasting and Visualizing Air Quality from Sky Images with Vision-Language Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出一种基于视觉语言模型的AI系统，通过天空图像预测空气质量并生成污染场景可视化。任务为污染预测与可视化，解决传统监测系统覆盖有限的问题，结合纹理分析与生成模型实现可解释的污染模拟与用户交互。**

- **链接: [http://arxiv.org/pdf/2509.15076v1](http://arxiv.org/pdf/2509.15076v1)**

> **作者:** Mohammad Saleh Vahdatpour; Maryam Eyvazi; Yanqing Zhang
>
> **备注:** Published at ICCVW 2025
>
> **摘要:** Air pollution remains a critical threat to public health and environmental sustainability, yet conventional monitoring systems are often constrained by limited spatial coverage and accessibility. This paper proposes an AI-driven agent that predicts ambient air pollution levels from sky images and synthesizes realistic visualizations of pollution scenarios using generative modeling. Our approach combines statistical texture analysis with supervised learning for pollution classification, and leverages vision-language model (VLM)-guided image generation to produce interpretable representations of air quality conditions. The generated visuals simulate varying degrees of pollution, offering a foundation for user-facing interfaces that improve transparency and support informed environmental decision-making. These outputs can be seamlessly integrated into intelligent applications aimed at enhancing situational awareness and encouraging behavioral responses based on real-time forecasts. We validate our method using a dataset of urban sky images and demonstrate its effectiveness in both pollution level estimation and semantically consistent visual synthesis. The system design further incorporates human-centered user experience principles to ensure accessibility, clarity, and public engagement in air quality forecasting. To support scalable and energy-efficient deployment, future iterations will incorporate a green CNN architecture enhanced with FPGA-based incremental learning, enabling real-time inference on edge platforms.
>
---
#### [new 081] Generalizable Geometric Image Caption Synthesis
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于图像字幕生成任务，旨在解决几何图像理解与模型泛化能力不足的问题。通过引入RLVR方法，改进数据生成流程，提升模型在几何及非几何任务中的推理表现。**

- **链接: [http://arxiv.org/pdf/2509.15217v1](http://arxiv.org/pdf/2509.15217v1)**

> **作者:** Yue Xin; Wenyuan Wang; Rui Pan; Ruida Wang; Howard Meng; Renjie Pi; Shizhe Diao; Tong Zhang
>
> **摘要:** Multimodal large language models have various practical applications that demand strong reasoning abilities. Despite recent advancements, these models still struggle to solve complex geometric problems. A key challenge stems from the lack of high-quality image-text pair datasets for understanding geometric images. Furthermore, most template-based data synthesis pipelines typically fail to generalize to questions beyond their predefined templates. In this paper, we bridge this gap by introducing a complementary process of Reinforcement Learning with Verifiable Rewards (RLVR) into the data generation pipeline. By adopting RLVR to refine captions for geometric images synthesized from 50 basic geometric relations and using reward signals derived from mathematical problem-solving tasks, our pipeline successfully captures the key features of geometry problem-solving. This enables better task generalization and yields non-trivial improvements. Furthermore, even in out-of-distribution scenarios, the generated dataset enhances the general reasoning capabilities of multimodal large language models, yielding accuracy improvements of $2.8\%\text{-}4.8\%$ in statistics, arithmetic, algebraic, and numerical tasks with non-geometric input images of MathVista and MathVerse, along with $2.4\%\text{-}3.9\%$ improvements in Art, Design, Tech, and Engineering tasks in MMMU.
>
---
#### [new 082] A Knowledge-driven Adaptive Collaboration of LLMs for Enhancing Medical Decision-making
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出KAMAC框架，解决LLM在医疗决策中静态角色限制问题。通过知识驱动的动态协作，实现专家团队灵活组建与扩展，提升复杂临床场景下的决策效果。属于多智能体协作任务。**

- **链接: [http://arxiv.org/pdf/2509.14998v1](http://arxiv.org/pdf/2509.14998v1)**

> **作者:** Xiao Wu; Ting-Zhu Huang; Liang-Jian Deng; Yanyuan Qiao; Imran Razzak; Yutong Xie
>
> **备注:** The paper has been accepted to the EMNLP 2025 Main Conference
>
> **摘要:** Medical decision-making often involves integrating knowledge from multiple clinical specialties, typically achieved through multidisciplinary teams. Inspired by this collaborative process, recent work has leveraged large language models (LLMs) in multi-agent collaboration frameworks to emulate expert teamwork. While these approaches improve reasoning through agent interaction, they are limited by static, pre-assigned roles, which hinder adaptability and dynamic knowledge integration. To address these limitations, we propose KAMAC, a Knowledge-driven Adaptive Multi-Agent Collaboration framework that enables LLM agents to dynamically form and expand expert teams based on the evolving diagnostic context. KAMAC begins with one or more expert agents and then conducts a knowledge-driven discussion to identify and fill knowledge gaps by recruiting additional specialists as needed. This supports flexible, scalable collaboration in complex clinical scenarios, with decisions finalized through reviewing updated agent comments. Experiments on two real-world medical benchmarks demonstrate that KAMAC significantly outperforms both single-agent and advanced multi-agent methods, particularly in complex clinical scenarios (i.e., cancer prognosis) requiring dynamic, cross-specialty expertise. Our code is publicly available at: https://github.com/XiaoXiao-Woo/KAMAC.
>
---
## 更新

#### [replaced 001] Skeleton-based sign language recognition using a dual-stream spatio-temporal dynamic graph convolutional network
- **分类: cs.CV; cs.AI; I.2.m; I.2.0**

- **链接: [http://arxiv.org/pdf/2509.08661v2](http://arxiv.org/pdf/2509.08661v2)**

> **作者:** Liangjin Liu; Haoyang Zheng; Zhengzhong Zhu; Pei Zhou
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Isolated Sign Language Recognition (ISLR) is challenged by gestures that are morphologically similar yet semantically distinct, a problem rooted in the complex interplay between hand shape and motion trajectory. Existing methods, often relying on a single reference frame, struggle to resolve this geometric ambiguity. This paper introduces Dual-SignLanguageNet (DSLNet), a dual-reference, dual-stream architecture that decouples and models gesture morphology and trajectory in separate, complementary coordinate systems. The architecture processes these streams through specialized networks: a topology-aware graph convolution models the view-invariant shape from a wrist-centric frame, while a Finsler geometry-based encoder captures the context-aware trajectory from a facial-centric frame. These features are then integrated via a geometry-driven optimal transport fusion mechanism. DSLNet sets a new state-of-the-art, achieving 93.70%, 89.97%, and 99.79% accuracy on the challenging WLASL-100, WLASL-300, and LSA64 datasets, respectively, with significantly fewer parameters than competing models.
>
---
#### [replaced 002] A new dataset and comparison for multi-camera frame synthesis
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09068v2](http://arxiv.org/pdf/2508.09068v2)**

> **作者:** Conall Daly; Anil Kokaram
>
> **备注:** SPIE 2025 - Applications of Digital Image Processing XLVIII accepted manuscript, 13 pages
>
> **摘要:** Many methods exist for frame synthesis in image sequences but can be broadly categorised into frame interpolation and view synthesis techniques. Fundamentally, both frame interpolation and view synthesis tackle the same task, interpolating a frame given surrounding frames in time or space. However, most frame interpolation datasets focus on temporal aspects with single cameras moving through time and space, while view synthesis datasets are typically biased toward stereoscopic depth estimation use cases. This makes direct comparison between view synthesis and frame interpolation methods challenging. In this paper, we develop a novel multi-camera dataset using a custom-built dense linear camera array to enable fair comparison between these approaches. We evaluate classical and deep learning frame interpolators against a view synthesis method (3D Gaussian Splatting) for the task of view in-betweening. Our results reveal that deep learning methods do not significantly outperform classical methods on real image data, with 3D Gaussian Splatting actually underperforming frame interpolators by as much as 3.5 dB PSNR. However, in synthetic scenes, the situation reverses -- 3D Gaussian Splatting outperforms frame interpolation algorithms by almost 5 dB PSNR at a 95% confidence level.
>
---
#### [replaced 003] Multi-label Scene Classification for Autonomous Vehicles: Acquiring and Accumulating Knowledge from Diverse Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17101v3](http://arxiv.org/pdf/2506.17101v3)**

> **作者:** Ke Li; Chenyu Zhang; Yuxin Ding; Xianbiao Hu; Ruwen Qin
>
> **摘要:** Driving scenes are inherently heterogeneous and dynamic. Multi-attribute scene identification, as a high-level visual perception capability, provides autonomous vehicles (AVs) with essential contextual awareness to understand, reason through, and interact with complex driving environments. Although scene identification is best modeled as a multi-label classification problem via multitask learning, it faces two major challenges: the difficulty of acquiring balanced, comprehensively annotated datasets and the need to re-annotate all training data when new attributes emerge. To address these challenges, this paper introduces a novel deep learning method that integrates Knowledge Acquisition and Accumulation (KAA) with Consistency-based Active Learning (CAL). KAA leverages monotask learning on heterogeneous single-label datasets to build a knowledge foundation, while CAL bridges the gap between single- and multi-label data, adapting the foundation model for multi-label scene classification. An ablation study on the newly developed Driving Scene Identification (DSI) dataset demonstrates a 56.1% improvement over an ImageNet-pretrained baseline. Moreover, KAA-CAL outperforms state-of-the-art multi-label classification methods on the BDD100K and HSD datasets, achieving this with 85% less data and even recognizing attributes unseen during foundation model training. The DSI dataset and KAA-CAL implementation code are publicly available at https://github.com/KELISBU/KAA-CAL .
>
---
#### [replaced 004] Brought a Gun to a Knife Fight: Modern VFM Baselines Outgun Specialized Detectors on In-the-Wild AI Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12995v2](http://arxiv.org/pdf/2509.12995v2)**

> **作者:** Yue Zhou; Xinan He; Kaiqing Lin; Bing Fan; Feng Ding; Jinhua Zeng; Bin Li
>
> **摘要:** While specialized detectors for AI-generated images excel on curated benchmarks, they fail catastrophically in real-world scenarios, as evidenced by their critically high false-negative rates on `in-the-wild' benchmarks. Instead of crafting another specialized `knife' for this problem, we bring a `gun' to the fight: a simple linear classifier on a modern Vision Foundation Model (VFM). Trained on identical data, this baseline decisively `outguns' bespoke detectors, boosting in-the-wild accuracy by a striking margin of over 20\%. Our analysis pinpoints the source of the VFM's `firepower': First, by probing text-image similarities, we find that recent VLMs (e.g., Perception Encoder, Meta CLIP2) have learned to align synthetic images with forgery-related concepts (e.g., `AI-generated'), unlike previous versions. Second, we speculate that this is due to data exposure, as both this alignment and overall accuracy plummet on a novel dataset scraped after the VFM's pre-training cut-off date, ensuring it was unseen during pre-training. Our findings yield two critical conclusions: 1) For the real-world `gunfight' of AI-generated image detection, the raw `firepower' of an updated VFM is far more effective than the `craftsmanship' of a static detector. 2) True generalization evaluation requires test data to be independent of the model's entire training history, including pre-training.
>
---
#### [replaced 005] Human + AI for Accelerating Ad Localization Evaluation
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.12543v2](http://arxiv.org/pdf/2509.12543v2)**

> **作者:** Harshit Rajgarhia; Shivali Dalmia; Mengyang Zhao; Mukherji Abhishek; Kiran Ganesh
>
> **摘要:** Adapting advertisements for multilingual audiences requires more than simple text translation; it demands preservation of visual consistency, spatial alignment, and stylistic integrity across diverse languages and formats. We introduce a structured framework that combines automated components with human oversight to address the complexities of advertisement localization. To the best of our knowledge, this is the first work to integrate scene text detection, inpainting, machine translation (MT), and text reimposition specifically for accelerating ad localization evaluation workflows. Qualitative results across six locales demonstrate that our approach produces semantically accurate and visually coherent localized advertisements, suitable for deployment in real-world workflows.
>
---
#### [replaced 006] MovieCORE: COgnitive REasoning in Movies
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19026v3](http://arxiv.org/pdf/2508.19026v3)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Ying Cheng; Hung-Ting Su; Yung-Hao Tang; Shang-Hong Lai; Winston H. Hsu
>
> **备注:** Accepted for EMNLP'2025 Main Conference (Oral Presentation). Project Page: https://joslefaure.github.io/assets/html/moviecore.html
>
> **摘要:** This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by up to 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at https://joslefaure.github.io/assets/html/moviecore.html.
>
---
#### [replaced 007] Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08636v2](http://arxiv.org/pdf/2503.08636v2)**

> **作者:** Hubert Baniecki; Przemyslaw Biecek
>
> **备注:** Accepted by Machine Learning
>
> **摘要:** A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of part-prototype networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.
>
---
#### [replaced 008] GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.18309v2](http://arxiv.org/pdf/2502.18309v2)**

> **作者:** Xinran Liu; Xu Dong; Diptesh Kanojia; Wenwu Wang; Zhenhua Feng
>
> **摘要:** Generating high-quality full-body dance sequences from music is a challenging task as it requires strict adherence to genre-specific choreography. Moreover, the generated sequences must be both physically realistic and precisely synchronized with the beats and rhythm of the music. To overcome these challenges, we propose GCDance, a classifier-free diffusion framework for generating genre-specific dance motions conditioned on both music and textual prompts. Specifically, our approach extracts music features by combining high-level pre-trained music foundation model features with hand-crafted features for multi-granularity feature fusion. To achieve genre controllability, we leverage CLIP to efficiently embed genre-based textual prompt representations at each time step within our dance generation pipeline. Our GCDance framework can generate diverse dance styles from the same piece of music while ensuring coherence with the rhythm and melody of the music. Extensive experimental results obtained on the FineDance dataset demonstrate that GCDance significantly outperforms the existing state-of-the-art approaches, which also achieve competitive results on the AIST++ dataset. Our ablation and inference time analysis demonstrate that GCDance provides an effective solution for high-quality music-driven dance generation.
>
---
#### [replaced 009] Interactive Face Video Coding: A Generative Compression Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2302.09919v2](http://arxiv.org/pdf/2302.09919v2)**

> **作者:** Bolin Chen; Zhao Wang; Binzhe Li; Shurun Wang; Shiqi Wang; Yan Ye
>
> **摘要:** In this paper, we propose a novel framework for Interactive Face Video Coding (IFVC), which allows humans to interact with the intrinsic visual representations instead of the signals. The proposed solution enjoys several distinct advantages, including ultra-compact representation, low delay interaction, and vivid expression/headpose animation. In particular, we propose the Internal Dimension Increase (IDI) based representation, greatly enhancing the fidelity and flexibility in rendering the appearance while maintaining reasonable representation cost. By leveraging strong statistical regularities, the visual signals can be effectively projected into controllable semantics in the three dimensional space (e.g., mouth motion, eye blinking, head rotation, head translation and head location), which are compressed and transmitted. The editable bitstream, which naturally supports the interactivity at the semantic level, can synthesize the face frames via the strong inference ability of the deep generative model. Experimental results have demonstrated the performance superiority and application prospects of our proposed IFVC scheme. In particular, the proposed scheme not only outperforms the state-of-the-art video coding standard Versatile Video Coding (VVC) and the latest generative compression schemes in terms of rate-distortion performance for face videos, but also enables the interactive coding without introducing additional manipulation processes. Furthermore, the proposed framework is expected to shed lights on the future design of the digital human communication in the metaverse.
>
---
#### [replaced 010] Reconstruction Alignment Improves Unified Multimodal Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.07295v2](http://arxiv.org/pdf/2509.07295v2)**

> **作者:** Ji Xie; Trevor Darrell; Luke Zettlemoyer; XuDong Wang
>
> **备注:** 28 pages, 24 figures and 10 tables; Update related work and fix typos
>
> **摘要:** Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details--even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RecA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts," providing rich supervision without captions. Concretely, RecA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RecA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 27 GPU-hours, post-training with RecA substantially improves image generation performance on GenEval (0.73$\rightarrow$0.90) and DPGBench (80.93$\rightarrow$88.15), while also boosting editing benchmarks (ImgEdit 3.38$\rightarrow$3.75, GEdit 6.94$\rightarrow$7.25). Notably, RecA surpasses much larger open-source models and applies broadly across diverse UMM architectures, establishing it as an efficient and general post-training alignment strategy for UMMs
>
---
#### [replaced 011] Physics-Informed Representation Alignment for Sparse Radio-Map Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19160v3](http://arxiv.org/pdf/2501.19160v3)**

> **作者:** Haozhe Jia; Wenshuo Chen; Zhihui Huang; Lei Wang; Hongru Xiao; Nanqian Jia; Keming Wu; Songning Lai; Bowen Tian; Yutao Yue
>
> **摘要:** Radio map reconstruction is essential for enabling advanced applications, yet challenges such as complex signal propagation and sparse observational data hinder accurate reconstruction in practical scenarios. Existing methods often fail to align physical constraints with data-driven features, particularly under sparse measurement conditions. To address these issues, we propose **Phy**sics-Aligned **R**adio **M**ap **D**iffusion **M**odel (**PhyRMDM**), a novel framework that establishes cross-domain representation alignment between physical principles and neural network features through dual learning pathways. The proposed model integrates **Physics-Informed Neural Networks (PINNs)** with a **representation alignment mechanism** that explicitly enforces consistency between Helmholtz equation constraints and environmental propagation patterns. Experimental results demonstrate significant improvements over state-of-the-art methods, achieving **NMSE of 0.0031** under *Static Radio Map (SRM)* conditions, and **NMSE of 0.0047** with **Dynamic Radio Map (DRM)** scenarios. The proposed representation alignment paradigm provides **37.2%** accuracy enhancement in ultra-sparse cases (**1%** sampling rate), confirming its effectiveness in bridging physics-based modeling and deep learning for radio map reconstruction.
>
---
#### [replaced 012] Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09105v3](http://arxiv.org/pdf/2507.09105v3)**

> **作者:** Maoxiao Ye; Xinfeng Ye; Mano Manoharan
>
> **摘要:** Earlier Sign Language Production (SLP) models typically relied on autoregressive methods that generate output tokens one by one, which inherently provide temporal alignment. Although techniques like Teacher Forcing can prevent model collapse during training, they still cannot solve the problem of error accumulation during inference, since ground truth is unavailable at that stage. In contrast, more recent approaches based on diffusion models leverage step-by-step denoising to enable high-quality generation. However, the iterative nature of these models and the requirement to denoise entire sequences limit their applicability in real-time tasks like SLP. To address it, we explore a hybrid approach that combines autoregressive and diffusion models for SLP, leveraging the strengths of both models in sequential dependency modeling and output refinement. To capture fine-grained body movements, we design a Multi-Scale Pose Representation module that separately extracts detailed features from distinct articulators and integrates them via a Multi-Scale Fusion module. Furthermore, we introduce a Confidence-Aware Causal Attention mechanism that utilizes joint-level confidence scores to dynamically guide the pose generation process, improving accuracy and robustness. Extensive experiments on the PHOENIX14T and How2Sign datasets demonstrate the effectiveness of our method in both generation quality and real-time efficiency.
>
---
#### [replaced 013] Fovea Stacking: Imaging with Dynamic Localized Aberration Correction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00716v2](http://arxiv.org/pdf/2506.00716v2)**

> **作者:** Shi Mao; Yogeshwar Nath Mishra; Wolfgang Heidrich
>
> **摘要:** The desire for cameras with smaller form factors has recently lead to a push for exploring computational imaging systems with reduced optical complexity such as a smaller number of lens elements. Unfortunately such simplified optical systems usually suffer from severe aberrations, especially in off-axis regions, which can be difficult to correct purely in software. In this paper we introduce Fovea Stacking , a new type of imaging system that utilizes emerging dynamic optical components called deformable phase plates (DPPs) for localized aberration correction anywhere on the image sensor. By optimizing DPP deformations through a differentiable optical model, off-axis aberrations are corrected locally, producing a foveated image with enhanced sharpness at the fixation point - analogous to the eye's fovea. Stacking multiple such foveated images, each with a different fixation point, yields a composite image free from aberrations. To efficiently cover the entire field of view, we propose joint optimization of DPP deformations under imaging budget constraints. Due to the DPP device's non-linear behavior, we introduce a neural network-based control model for improved alignment between simulation-hardware performance. We further demonstrated that for extended depth-of-field imaging, fovea stacking outperforms traditional focus stacking in image quality. By integrating object detection or eye-tracking, the system can dynamically adjust the lens to track the object of interest-enabling real-time foveated video suitable for downstream applications such as surveillance or foveated virtual reality displays
>
---
#### [replaced 014] IV-tuning: Parameter-Efficient Transfer Learning for Infrared-Visible Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16654v4](http://arxiv.org/pdf/2412.16654v4)**

> **作者:** Yaming Zhang; Chenqiang Gao; Fangcen Liu; Junjie Guo; Lan Wang; Xinggan Peng; Deyu Meng
>
> **摘要:** Existing infrared and visible (IR-VIS) methods inherit the general representations of Pre-trained Visual Models (PVMs) to facilitate complementary learning. However, our analysis indicates that under the full fine-tuning paradigm, the feature space becomes highly constrained and low-ranked, which has been proven to seriously impair generalization. One solution is freezing parameters to preserve pre-trained knowledge and thus maintain diversity of the feature space. To this end, we propose IV-tuning, to parameter-efficiently harness PVMs for various IR-VIS downstream tasks, including salient object detection, semantic segmentation, and object detection. Compared with the full fine-tuning baselines and existing IR-VIS methods, IV-tuning facilitates the learning of complementary information between infrared and visible modalities with less than 3% of the backbone parameters, and effectively alleviates the overfitting problem. The code is available in https://github.com/Yummy198913/IV-tuning.
>
---
#### [replaced 015] Morph: A Motion-free Physics Optimization Framework for Human Motion Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.14951v3](http://arxiv.org/pdf/2411.14951v3)**

> **作者:** Zhuo Li; Mingshuang Luo; Ruibing Hou; Xin Zhao; Hao Liu; Hong Chang; Zimo Liu; Chen Li
>
> **备注:** Accepted by ICCV 2025, 15 pages, 6 figures
>
> **摘要:** Human motion generation has been widely studied due to its crucial role in areas such as digital humans and humanoid robot control. However, many current motion generation approaches disregard physics constraints, frequently resulting in physically implausible motions with pronounced artifacts such as floating and foot sliding. Meanwhile, training an effective motion physics optimizer with noisy motion data remains largely unexplored. In this paper, we propose \textbf{Morph}, a \textbf{Mo}tion-F\textbf{r}ee \textbf{ph}ysics optimization framework, consisting of a Motion Generator and a Motion Physics Refinement module, for enhancing physical plausibility without relying on expensive real-world motion data. Specifically, the motion generator is responsible for providing large-scale synthetic, noisy motion data, while the motion physics refinement module utilizes these synthetic data to learn a motion imitator within a physics simulator, enforcing physical constraints to project the noisy motions into a physically-plausible space. Additionally, we introduce a prior reward module to enhance the stability of the physics optimization process and generate smoother and more stable motions. These physically refined motions are then used to fine-tune the motion generator, further enhancing its capability. This collaborative training paradigm enables mutual enhancement between the motion generator and the motion physics refinement module, significantly improving practicality and robustness in real-world applications. Experiments on both text-to-motion and music-to-dance generation tasks demonstrate that our framework achieves state-of-the-art motion quality while improving physical plausibility drastically.
>
---
#### [replaced 016] MedFuncta: A Unified Framework for Learning Efficient Medical Neural Fields
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14401v3](http://arxiv.org/pdf/2502.14401v3)**

> **作者:** Paul Friedrich; Florentin Bieder; Julian McGinnis; Julia Wolleb; Daniel Rueckert; Philippe C. Cattin
>
> **备注:** Project page: https://pfriedri.github.io/medfuncta-io/ Code: https://github.com/pfriedri/medfuncta/ Dataset: https://doi.org/10.5281/zenodo.14898708
>
> **摘要:** Research in medical imaging primarily focuses on discrete data representations that poorly scale with grid resolution and fail to capture the often continuous nature of the underlying signal. Neural Fields (NFs) offer a powerful alternative by modeling data as continuous functions. While single-instance NFs have successfully been applied in medical contexts, extending them to large-scale medical datasets remains an open challenge. We therefore introduce MedFuncta, a unified framework for large-scale NF training on diverse medical signals. Building on Functa, our approach encodes data into a unified representation, namely a 1D latent vector, that modulates a shared, meta-learned NF, enabling generalization across a dataset. We revisit common design choices, introducing a non-constant frequency parameter $\omega$ in widely used SIREN activations, and establish a connection between this $\omega$-schedule and layer-wise learning rates, relating our findings to recent work in theoretical learning dynamics. We additionally introduce a scalable meta-learning strategy for shared network learning that employs sparse supervision during training, thereby reducing memory consumption and computational overhead while maintaining competitive performance. Finally, we evaluate MedFuncta across a diverse range of medical datasets and show how to solve relevant downstream tasks on our neural data representation. To promote further research in this direction, we release our code, model weights and the first large-scale dataset - MedNF - containing > 500 k latent vectors for multi-instance medical NFs.
>
---
#### [replaced 017] ReservoirTTA: Prolonged Test-time Adaptation for Evolving and Recurring Domains
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14511v3](http://arxiv.org/pdf/2505.14511v3)**

> **作者:** Guillaume Vray; Devavrat Tomar; Xufeng Gao; Jean-Philippe Thiran; Evan Shelhamer; Behzad Bozorgtabar
>
> **摘要:** This paper introduces ReservoirTTA, a novel plug-in framework designed for prolonged test-time adaptation (TTA) in scenarios where the test domain continuously shifts over time, including cases where domains recur or evolve gradually. At its core, ReservoirTTA maintains a reservoir of domain-specialized models -- an adaptive test-time model ensemble -- that both detects new domains via online clustering over style features of incoming samples and routes each sample to the appropriate specialized model, and thereby enables domain-specific adaptation. This multi-model strategy overcomes key limitations of single model adaptation, such as catastrophic forgetting, inter-domain interference, and error accumulation, ensuring robust and stable performance on sustained non-stationary test distributions. Our theoretical analysis reveals key components that bound parameter variance and prevent model collapse, while our plug-in TTA module mitigates catastrophic forgetting of previously encountered domains. Extensive experiments on scene-level corruption benchmarks (ImageNet-C, CIFAR-10/100-C), object-level style shifts (DomainNet-126, PACS), and semantic segmentation (Cityscapes->ACDC) covering recurring and continuously evolving domain shifts -- show that ReservoirTTA substantially improves adaptation accuracy and maintains stable performance across prolonged, recurring shifts, outperforming state-of-the-art methods. Our code is publicly available at https://github.com/LTS5/ReservoirTTA.
>
---
#### [replaced 018] Efficient Dual-domain Image Dehazing with Haze Prior Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11035v2](http://arxiv.org/pdf/2507.11035v2)**

> **作者:** Lirong Zheng; Yanshan Li; Rui Yu; Kaihao Zhang
>
> **备注:** 10 pages
>
> **摘要:** Transformer-based models exhibit strong global modeling capabilities in single-image dehazing, but their high computational cost limits real-time applicability. Existing methods predominantly rely on spatial-domain features to capture long-range dependencies, which are computationally expensive and often inadequate under complex haze conditions. While some approaches introduce frequency-domain cues, the weak coupling between spatial and frequency branches limits the overall performance. To overcome these limitations, we propose the Dark Channel Guided Frequency-aware Dehazing Network (DGFDNet), a novel dual-domain framework that performs physically guided degradation alignment across spatial and frequency domains. At its core, the DGFDBlock comprises two key modules: 1) the Haze-Aware Frequency Modulator (HAFM), which generates a pixel-level haze confidence map from dark channel priors to adaptively enhance haze-relevant frequency components, thereby achieving global degradation-aware spectral modulation; 2) the Multi-level Gating Aggregation Module (MGAM), which fuses multi-scale features through diverse convolutional kernels and hybrid gating mechanisms to recover fine structural details. Additionally, a Prior Correction Guidance Branch (PCGB) incorporates a closed-loop feedback mechanism, enabling iterative refinement of the prior by intermediate dehazed features and significantly improving haze localization accuracy, especially in challenging outdoor scenes. Extensive experiments on four benchmark haze datasets demonstrate that DGFDNet achieves state-of-the-art performance with superior robustness and real-time efficiency. Code is available at: https://github.com/Dilizlr/DGFDNet.
>
---
#### [replaced 019] Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.07027v3](http://arxiv.org/pdf/2509.07027v3)**

> **作者:** Jisung Hwang; Jaihoon Kim; Minhyuk Sung
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** We propose a novel regularization loss that enforces standard Gaussianity, encouraging samples to align with a standard Gaussian distribution. This facilitates a range of downstream tasks involving optimization in the latent space of text-to-image models. We treat elements of a high-dimensional sample as one-dimensional standard Gaussian variables and define a composite loss that combines moment-based regularization in the spatial domain with power spectrum-based regularization in the spectral domain. Since the expected values of moments and power spectrum distributions are analytically known, the loss promotes conformity to these properties. To ensure permutation invariance, the losses are applied to randomly permuted inputs. Notably, existing Gaussianity-based regularizations fall within our unified framework: some correspond to moment losses of specific orders, while the previous covariance-matching loss is equivalent to our spectral loss but incurs higher time complexity due to its spatial-domain computation. We showcase the application of our regularization in generative modeling for test-time reward alignment with a text-to-image model, specifically to enhance aesthetics and text alignment. Our regularization outperforms previous Gaussianity regularization, effectively prevents reward hacking and accelerates convergence.
>
---
#### [replaced 020] Dense Video Understanding with Gated Residual Tokenization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; 68T45, 68T07, 68T05, 68T10, 68T50, 68T09, 68U10, 68P20, 94A08,
  94A34, 62H30, 62H35; I.2.10; I.2.6; I.2.7; I.5.1; I.5.2; I.5.3; I.5.4; I.4.8; I.4.9;
  I.4.2; H.3.1; H.3.3; H.3.4; H.5.1; H.5.2; H.2.8**

- **链接: [http://arxiv.org/pdf/2509.14199v2](http://arxiv.org/pdf/2509.14199v2)**

> **作者:** Haichao Zhang; Wenhao Chai; Shwai He; Ang Li; Yun Fu
>
> **摘要:** High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and benchmarks mostly rely on low-frame-rate sampling, such as uniform sampling or keyframe selection, discarding dense temporal information. This compromise avoids the high cost of tokenizing every frame, which otherwise leads to redundant computation and linear token growth as video length increases. While this trade-off works for slowly changing content, it fails for tasks like lecture comprehension, where information appears in nearly every frame and requires precise temporal alignment. To address this gap, we introduce Dense Video Understanding (DVU), which enables high-FPS video comprehension by reducing both tokenization time and token overhead. Existing benchmarks are also limited, as their QA pairs focus on coarse content changes. We therefore propose DIVE (Dense Information Video Evaluation), the first benchmark designed for dense temporal reasoning. To make DVU practical, we present Gated Residual Tokenization (GRT), a two-stage framework: (1) Motion-Compensated Inter-Gated Tokenization uses pixel-level motion estimation to skip static regions during tokenization, achieving sub-linear growth in token count and compute. (2) Semantic-Scene Intra-Tokenization Merging fuses tokens across static regions within a scene, further reducing redundancy while preserving dynamic semantics. Experiments on DIVE show that GRT outperforms larger VLLM baselines and scales positively with FPS. These results highlight the importance of dense temporal information and demonstrate that GRT enables efficient, scalable high-FPS video understanding.
>
---
#### [replaced 021] MATTER: Multiscale Attention for Registration Error Regression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12924v2](http://arxiv.org/pdf/2509.12924v2)**

> **作者:** Shipeng Liu; Ziliang Xiong; Khac-Hoang Ngo; Per-Erik Forssén
>
> **摘要:** Point cloud registration (PCR) is crucial for many downstream tasks, such as simultaneous localization and mapping (SLAM) and object tracking. This makes detecting and quantifying registration misalignment, i.e., PCR quality validation, an important task. All existing methods treat validation as a classification task, aiming to assign the PCR quality to a few classes. In this work, we instead use regression for PCR validation, allowing for a more fine-grained quantification of the registration quality. We also extend previously used misalignment-related features by using multiscale extraction and attention-based aggregation. This leads to accurate and robust registration error estimation on diverse datasets, especially for point clouds with heterogeneous spatial densities. Furthermore, when used to guide a mapping downstream task, our method significantly improves the mapping quality for a given amount of re-registered frames, compared to the state-of-the-art classification-based method.
>
---
#### [replaced 022] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22159v3](http://arxiv.org/pdf/2505.22159v3)**

> **作者:** Jiawen Yu; Hairuo Liu; Qiaojun Yu; Jieji Ren; Ce Hao; Haitong Ding; Guangyu Huang; Guofan Huang; Yan Song; Panpan Cai; Cewu Lu; Wenqiang Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Vision-Language-Action (VLA) models have advanced general-purpose robotic manipulation by leveraging pretrained visual and linguistic representations. However, they struggle with contact-rich tasks that require fine-grained control involving force, especially under visual occlusion or dynamic uncertainty. To address these limitations, we propose ForceVLA, a novel end-to-end manipulation framework that treats external force sensing as a first-class modality within VLA systems. ForceVLA introduces FVLMoE, a force-aware Mixture-of-Experts fusion module that dynamically integrates pretrained visual-language embeddings with real-time 6-axis force feedback during action decoding. This enables context-aware routing across modality-specific experts, enhancing the robot's ability to adapt to subtle contact dynamics. We also introduce \textbf{ForceVLA-Data}, a new dataset comprising synchronized vision, proprioception, and force-torque signals across five contact-rich manipulation tasks. ForceVLA improves average task success by 23.2% over strong pi_0-based baselines, achieving up to 80% success in tasks such as plug insertion. Our approach highlights the importance of multimodal integration for dexterous manipulation and sets a new benchmark for physically intelligent robotic control. Code and data will be released at https://sites.google.com/view/forcevla2025.
>
---
#### [replaced 023] The Art of Saying "Maybe": A Conformal Lens for Uncertainty Benchmarking in VLMs
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13379v2](http://arxiv.org/pdf/2509.13379v2)**

> **作者:** Asif Azad; Mohammad Sadat Hossain; MD Sadik Hossain Shanto; M Saifur Rahman; Md Rizwan Parvez
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in complex visual understanding across scientific and reasoning tasks. While performance benchmarking has advanced our understanding of these capabilities, the critical dimension of uncertainty quantification has received insufficient attention. Therefore, unlike prior conformal prediction studies that focused on limited settings, we conduct a comprehensive uncertainty benchmarking study, evaluating 16 state-of-the-art VLMs (open and closed-source) across 6 multimodal datasets with 3 distinct scoring functions. Our findings demonstrate that larger models consistently exhibit better uncertainty quantification; models that know more also know better what they don't know. More certain models achieve higher accuracy, while mathematical and reasoning tasks elicit poorer uncertainty performance across all models compared to other domains. This work establishes a foundation for reliable uncertainty evaluation in multimodal systems.
>
---
#### [replaced 024] Fine-tuning Vision Language Models with Graph-based Knowledge for Explainable Medical Image Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.09808v2](http://arxiv.org/pdf/2503.09808v2)**

> **作者:** Chenjun Li; Laurin Lux; Alexander H. Berger; Martin J. Menten; Mert R. Sabuncu; Johannes C. Paetzold
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Accurate staging of Diabetic Retinopathy (DR) is essential for guiding timely interventions and preventing vision loss. However, current staging models are hardly interpretable, and most public datasets contain no clinical reasoning or interpretation beyond image-level labels. In this paper, we present a novel method that integrates graph representation learning with vision-language models (VLMs) to deliver explainable DR diagnosis. Our approach leverages optical coherence tomography angiography (OCTA) images by constructing biologically informed graphs that encode key retinal vascular features such as vessel morphology and spatial connectivity. A graph neural network (GNN) then performs DR staging while integrated gradients highlight critical nodes and edges and their individual features that drive the classification decisions. We collect this graph-based knowledge which attributes the model's prediction to physiological structures and their characteristics. We then transform it into textual descriptions for VLMs. We perform instruction-tuning with these textual descriptions and the corresponding image to train a student VLM. This final agent can classify the disease and explain its decision in a human interpretable way solely based on a single image input. Experimental evaluations on both proprietary and public datasets demonstrate that our method not only improves classification accuracy but also offers more clinically interpretable results. An expert study further demonstrates that our method provides more accurate diagnostic explanations and paves the way for precise localization of pathologies in OCTA images.
>
---
#### [replaced 025] Diffusion-Based Action Recognition Generalizes to Untrained Domains
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.08908v2](http://arxiv.org/pdf/2509.08908v2)**

> **作者:** Rogerio Guimaraes; Frank Xiao; Pietro Perona; Markus Marks
>
> **摘要:** Humans can recognize the same actions despite large context and viewpoint variations, such as differences between species (walking in spiders vs. horses), viewpoints (egocentric vs. third-person), and contexts (real life vs movies). Current deep learning models struggle with such generalization. We propose using features generated by a Vision Diffusion Model (VDM), aggregated via a transformer, to achieve human-like action recognition across these challenging conditions. We find that generalization is enhanced by the use of a model conditioned on earlier timesteps of the diffusion process to highlight semantic information over pixel level details in the extracted features. We experimentally explore the generalization properties of our approach in classifying actions across animal species, across different viewing angles, and different recording contexts. Our model sets a new state-of-the-art across all three generalization benchmarks, bringing machine action recognition closer to human-like robustness. Project page: $\href{https://www.vision.caltech.edu/actiondiff/}{\text{vision.caltech.edu/actiondiff}}$ Code: $\href{https://github.com/frankyaoxiao/ActionDiff}{\text{github.com/frankyaoxiao/ActionDiff}}$
>
---
#### [replaced 026] SWAT: Sliding Window Adversarial Training for Gradual Domain Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.19155v2](http://arxiv.org/pdf/2501.19155v2)**

> **作者:** Zixi Wang; Xiangxu Zhao; Tonglan Xie; Mengmeng Jing; Lin Zuo
>
> **备注:** submitted to NIPS 2025
>
> **摘要:** Domain shifts are critical issues that harm the performance of machine learning. Unsupervised Domain Adaptation (UDA) mitigates this issue but suffers when the domain shifts are steep and drastic. Gradual Domain Adaptation (GDA) alleviates this problem in a mild way by gradually adapting from the source to the target domain using multiple intermediate domains. In this paper, we propose Sliding Window Adversarial Training (SWAT) for GDA. SWAT first formulates adversarial streams to connect the feature spaces of the source and target domains. Then, a sliding window paradigm is designed that moves along the adversarial stream to gradually narrow the small gap between adjacent intermediate domains. When the window moves to the end of the stream, i.e., the target domain, the domain shift is explicitly reduced. Extensive experiments on six GDA benchmarks demonstrate the significant effectiveness of SWAT, especially 6.1% improvement on Rotated MNIST and 4.1% advantage on CIFAR-100C over the previous methods.
>
---
#### [replaced 027] Efficient Fine-Tuning of DINOv3 Pretrained on Natural Images for Atypical Mitotic Figure Classification in MIDOG 2025
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21041v2](http://arxiv.org/pdf/2508.21041v2)**

> **作者:** Guillaume Balezo; Hana Feki; Raphaël Bourgade; Lily Monnier; Alice Blondel; Albert Pla Planas; Thomas Walter
>
> **备注:** 4 pages. Challenge report for MIDOG 2025 (Task 2: Atypical Mitotic Figure Classification)
>
> **摘要:** Atypical mitotic figures (AMFs) represent abnormal cell division associated with poor prognosis. Yet their detection remains difficult due to low prevalence, subtle morphology, and inter-observer variability. The MIDOG 2025 challenge introduces a benchmark for AMF classification across multiple domains. In this work, we fine-tuned the recently published DINOv3-H+ vision transformer, pretrained on natural images, using low-rank adaptation (LoRA), training only ~1.3M parameters in combination with extensive augmentation and a domain-weighted Focal Loss to handle domain heterogeneity. Despite the domain gap, our fine-tuned DINOv3 transfers effectively to histopathology, reaching second place on the preliminary test set. These results highlight the advantages of DINOv3 pretraining and underline the efficiency and robustness of our fine-tuning strategy, yielding state-of-the-art results for the atypical mitosis classification challenge in MIDOG 2025.
>
---
#### [replaced 028] On the Role of Individual Differences in Current Approaches to Computational Image Aesthetics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20518v2](http://arxiv.org/pdf/2502.20518v2)**

> **作者:** Li-Wei Chen; Ombretta Strafforello; Anne-Sofie Maerten; Tinne Tuytelaars; Johan Wagemans
>
> **备注:** 14 pages
>
> **摘要:** Image aesthetic assessment (IAA) evaluates image aesthetics, a task complicated by image diversity and user subjectivity. Current approaches address this in two stages: Generic IAA (GIAA) models estimate mean aesthetic scores, while Personal IAA (PIAA) models adapt GIAA using transfer learning to incorporate user subjectivity. However, a theoretical understanding of transfer learning between GIAA and PIAA, particularly concerning the impact of group composition, group size, aesthetic differences between groups and individuals, and demographic correlations, is lacking. This work establishes a theoretical foundation for IAA, proposing a unified model that encodes individual characteristics in a distributional format for both individual and group assessments. We show that transferring from GIAA to PIAA involves extrapolation, while the reverse involves interpolation, which is generally more effective for machine learning. Extensive experiments with varying group compositions, including sub-sampling by group size and disjoint demographics, reveal substantial performance variation even for GIAA, challenging the assumption that averaging scores eliminates individual subjectivity. Score-distribution analysis using Earth Mover's Distance (EMD) and the Gini index identifies education, photography experience, and art experience as key factors in aesthetic differences, with greater subjectivity in artworks than in photographs. Code is available at https://github.com/lwchen6309/aesthetics_transfer_learning.
>
---
#### [replaced 029] Multimodal Knowledge Distillation for Egocentric Action Recognition Robust to Missing Modalities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.08578v2](http://arxiv.org/pdf/2504.08578v2)**

> **作者:** Maria Santos-Villafranca; Dustin Carrión-Ojeda; Alejandro Perez-Yus; Jesus Bermudez-Cameo; Jose J. Guerrero; Simone Schaub-Meyer
>
> **备注:** Project Page: https://visinf.github.io/KARMMA
>
> **摘要:** Existing methods for egocentric action recognition often rely solely on RGB videos, while additional modalities, e.g., audio, can improve accuracy in challenging scenarios. However, most prior multimodal approaches assume all modalities are available at inference, leading to significant accuracy drops, or even failure, when inputs are missing. To address this, we introduce KARMMA, a multimodal Knowledge distillation approach for egocentric Action Recognition robust to Missing ModAlities that requires no modality alignment across all samples during training or inference. KARMMA distills knowledge from a multimodal teacher into a multimodal student that benefits from all available modalities while remaining robust to missing ones, making it suitable for diverse multimodal scenarios without retraining. Our student uses approximately 50% fewer computational resources than our teacher, resulting in a lightweight and fast model. Experiments on Epic-Kitchens and Something-Something show that our student achieves competitive accuracy while significantly reducing accuracy drops under missing modality conditions.
>
---
#### [replaced 030] MINGLE: VLMs for Semantically Complex Region Detection in Urban Scenes
- **分类: cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.13484v2](http://arxiv.org/pdf/2509.13484v2)**

> **作者:** Liu Liu; Alexandra Kudaeva; Marco Cipriano; Fatimeh Al Ghannam; Freya Tan; Gerard de Melo; Andres Sevtsuk
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Understanding group-level social interactions in public spaces is crucial for urban planning, informing the design of socially vibrant and inclusive environments. Detecting such interactions from images involves interpreting subtle visual cues such as relations, proximity, and co-movement - semantically complex signals that go beyond traditional object detection. To address this challenge, we introduce a social group region detection task, which requires inferring and spatially grounding visual regions defined by abstract interpersonal relations. We propose MINGLE (Modeling INterpersonal Group-Level Engagement), a modular three-stage pipeline that integrates: (1) off-the-shelf human detection and depth estimation, (2) VLM-based reasoning to classify pairwise social affiliation, and (3) a lightweight spatial aggregation algorithm to localize socially connected groups. To support this task and encourage future research, we present a new dataset of 100K urban street-view images annotated with bounding boxes and labels for both individuals and socially interacting groups. The annotations combine human-created labels and outputs from the MINGLE pipeline, ensuring semantic richness and broad coverage of real-world scenarios.
>
---
#### [replaced 031] Gradient Distance Function
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22422v2](http://arxiv.org/pdf/2410.22422v2)**

> **作者:** Hieu Le; Federico Stella; Benoit Guillard; Pascal Fua
>
> **备注:** ICCV 2025 - Wild3D workshop
>
> **摘要:** Unsigned Distance Functions (UDFs) can be used to represent non-watertight surfaces in a deep learning framework. However, UDFs tend to be brittle and difficult to learn, in part because the surface is located exactly where the UDF is non-differentiable. In this work, we show that Gradient Distance Functions (GDFs) can remedy this by being differentiable at the surface while still being able to represent open surfaces. This is done by associating to each 3D point a 3D vector whose norm is taken to be the unsigned distance to the surface and whose orientation is taken to be the direction towards the closest surface point. We demonstrate the effectiveness of GDFs on ShapeNet Car, Multi-Garment, and 3D-Scene datasets with both single-shape reconstruction networks or categorical auto-decoders.
>
---
#### [replaced 032] Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2: Atypical Mitosis Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02591v3](http://arxiv.org/pdf/2509.02591v3)**

> **作者:** Mieko Ochi; Bae Yuan
>
> **摘要:** Mitotic figures are classified into typical and atypical variants, with atypical counts correlating strongly with tumor aggressiveness. Accurate differentiation is therefore essential for patient prognostication and resource allocation, yet remains challenging even for expert pathologists. Here, we leveraged Pathology Foundation Models (PFMs) pre-trained on large histopathology datasets and applied parameter-efficient fine-tuning via low-rank adaptation. In addition, we incorporated ConvNeXt V2, a state-of-the-art convolutional neural network architecture, to complement PFMs. During training, we employed a fisheye transform to emphasize mitoses and Fourier Domain Adaptation using ImageNet target images. Finally, we ensembled multiple PFMs to integrate complementary morphological insights, achieving competitive balanced accuracy on the Preliminary Evaluation Phase dataset.
>
---
#### [replaced 033] AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.14084v2](http://arxiv.org/pdf/2509.14084v2)**

> **作者:** Jingyi Yuan; Jianxiong Ye; Wenkang Chen; Chenqiang Gao
>
> **摘要:** Zero-Shot Anomaly Detection (ZSAD) seeks to identify anomalies from arbitrary novel categories, offering a scalable and annotation-efficient solution. Traditionally, most ZSAD works have been based on the CLIP model, which performs anomaly detection by calculating the similarity between visual and text embeddings. Recently, vision foundation models such as DINOv3 have demonstrated strong transferable representation capabilities. In this work, we are the first to adapt DINOv3 for ZSAD. However, this adaptation presents two key challenges: (i) the domain bias between large-scale pretraining data and anomaly detection tasks leads to feature misalignment; and (ii) the inherent bias toward global semantics in pretrained representations often leads to subtle anomalies being misinterpreted as part of the normal foreground objects, rather than being distinguished as abnormal regions. To overcome these challenges, we introduce AD-DINOv3, a novel vision-language multimodal framework designed for ZSAD. Specifically, we formulate anomaly detection as a multimodal contrastive learning problem, where DINOv3 is employed as the visual backbone to extract patch tokens and a CLS token, and the CLIP text encoder provides embeddings for both normal and abnormal prompts. To bridge the domain gap, lightweight adapters are introduced in both modalities, enabling their representations to be recalibrated for the anomaly detection task. Beyond this baseline alignment, we further design an Anomaly-Aware Calibration Module (AACM), which explicitly guides the CLS token to attend to anomalous regions rather than generic foreground semantics, thereby enhancing discriminability. Extensive experiments on eight industrial and medical benchmarks demonstrate that AD-DINOv3 consistently matches or surpasses state-of-the-art methods.The code will be available at https://github.com/Kaisor-Yuan/AD-DINOv3.
>
---
#### [replaced 034] A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13718v2](http://arxiv.org/pdf/2501.13718v2)**

> **作者:** Dario Serez; Marco Cristani; Alessio Del Bue; Vittorio Murino; Pietro Morerio
>
> **摘要:** In image generation, Multiple Latent Variable Generative Models (MLVGMs) employ multiple latent variables to gradually shape the final images, from global characteristics to finer and local details (e.g., StyleGAN, NVAE), emerging as powerful tools for diverse applications. Yet their generative dynamics remain only empirically observed, without a systematic understanding of each latent variable's impact. In this work, we propose a novel framework that quantifies the contribution of each latent variable using Mutual Information (MI) as a metric. Our analysis reveals that current MLVGMs often underutilize some latent variables, and provides actionable insights for their use in downstream applications. With this foundation, we introduce a method for generating synthetic data for Self-Supervised Contrastive Representation Learning (SSCRL). By leveraging the hierarchical and disentangled variables of MLVGMs, our approach produces diverse and semantically meaningful views without the need for real image data. Additionally, we introduce a Continuous Sampling (CS) strategy, where the generator dynamically creates new samples during SSCRL training, greatly increasing data variability. Our comprehensive experiments demonstrate the effectiveness of these contributions, showing that MLVGMs' generated views compete on par with or even surpass views generated from real data. This work establishes a principled approach to understanding and exploiting MLVGMs, advancing both generative modeling and self-supervised learning. Code and pre-trained models at: https://github.com/SerezD/mi_ml_gen.
>
---
#### [replaced 035] A-TDOM: Active TDOM via On-the-Fly 3DGS
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12759v2](http://arxiv.org/pdf/2509.12759v2)**

> **作者:** Yiwei Xu; Xiang Wang; Yifei Yu; Wentian Gan; Luca Morelli; Giulio Perda; Xiongwu Xiao; Zongqian Zhan; Xin Wang; Fabio Remondino
>
> **备注:** This is a short white paper for a coming Journal Paper
>
> **摘要:** True Digital Orthophoto Map (TDOM) serves as a crucial geospatial product in various fields such as urban management, city planning, land surveying, etc. However, traditional TDOM generation methods generally rely on a complex offline photogrammetric pipeline, resulting in delays that hinder real-time applications. Moreover, the quality of TDOM may degrade due to various challenges, such as inaccurate camera poses or Digital Surface Model (DSM) and scene occlusions. To address these challenges, this work introduces A-TDOM, a near real-time TDOM generation method based on On-the-Fly 3DGS optimization. As each image is acquired, its pose and sparse point cloud are computed via On-the-Fly SfM. Then new Gaussians are integrated and optimized into previously unseen or coarsely reconstructed regions. By integrating with orthogonal splatting, A-TDOM can render just after each update of a new 3DGS field. Initial experiments on multiple benchmarks show that the proposed A-TDOM is capable of actively rendering TDOM in near real-time, with 3DGS optimization for each new image in seconds while maintaining acceptable rendering quality and TDOM geometric accuracy.
>
---
#### [replaced 036] VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18042v2](http://arxiv.org/pdf/2502.18042v2)**

> **作者:** Pei Liu; Haipeng Liu; Haichao Liu; Xin Liu; Jinxin Ni; Jun Ma
>
> **摘要:** Human drivers adeptly navigate complex scenarios by utilizing rich attentional semantics, but the current autonomous systems struggle to replicate this ability, as they often lose critical semantic information when converting 2D observations into 3D space. In this sense, it hinders their effective deployment in dynamic and complex environments. Leveraging the superior scene understanding and reasoning abilities of Vision-Language Models (VLMs), we propose VLM-E2E, a novel framework that uses the VLMs to enhance training by providing attentional cues. Our method integrates textual representations into Bird's-Eye-View (BEV) features for semantic supervision, which enables the model to learn richer feature representations that explicitly capture the driver's attentional semantics. By focusing on attentional semantics, VLM-E2E better aligns with human-like driving behavior, which is critical for navigating dynamic and complex environments. Furthermore, we introduce a BEV-Text learnable weighted fusion strategy to address the issue of modality importance imbalance in fusing multimodal information. This approach dynamically balances the contributions of BEV and text features, ensuring that the complementary information from visual and textual modalities is effectively utilized. By explicitly addressing the imbalance in multimodal fusion, our method facilitates a more holistic and robust representation of driving environments. We evaluate VLM-E2E on the nuScenes dataset and achieve significant improvements in perception, prediction, and planning over the baseline end-to-end model, showcasing the effectiveness of our attention-enhanced BEV representation in enabling more accurate and reliable autonomous driving tasks.
>
---
#### [replaced 037] Image Super-Resolution Reconstruction Network based on Enhanced Swin Transformer via Alternating Aggregation of Local-Global Features
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.00241v5](http://arxiv.org/pdf/2401.00241v5)**

> **作者:** Yuming Huang; Yingpin Chen; Changhui Wu; Binhui Song; Hui Wang
>
> **摘要:** The Swin Transformer image super-resolution (SR) reconstruction network primarily depends on the long-range relationship of the window and shifted window attention to explore features. However, this approach focuses only on global features, ignoring local ones, and considers only spatial interactions, disregarding channel and spatial-channel feature interactions, limiting its nonlinear mapping capability. Therefore, this study proposes an enhanced Swin Transformer network (ESTN) that alternately aggregates local and global features. During local feature aggregation, shift convolution facilitates the interaction between local spatial and channel information. During global feature aggregation, a block sparse global perception module is introduced, wherein spatial information is reorganized and the recombined features are then processed by a dense layer to achieve global perception. Additionally, multiscale self-attention and low-parameter residual channel attention modules are introduced to aggregate information across different scales. Finally, the effectiveness of ESTN on five public datasets and a local attribution map (LAM) are analyzed. Experimental results demonstrate that the proposed ESTN achieves higher average PSNR, surpassing SRCNN, ELAN-light, SwinIR-light, and SMFANER+ models by 2.17dB, 0.13dB, 0.12dB, and 0.1dB, respectively, with LAM further confirming its larger receptive field. ESTN delivers improved quality of SR images. The source code can be found at https://github.com/huangyuming2021/ESTN.
>
---
#### [replaced 038] Structural-Spectral Graph Convolution with Evidential Edge Learning for Hyperspectral Image Clustering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09920v2](http://arxiv.org/pdf/2506.09920v2)**

> **作者:** Jianhan Qi; Yuheng Jia; Hui Liu; Junhui Hou
>
> **摘要:** Hyperspectral image (HSI) clustering assigns similar pixels to the same class without any annotations, which is an important yet challenging task. For large-scale HSIs, most methods rely on superpixel segmentation and perform superpixel-level clustering based on graph neural networks (GNNs). However, existing GNNs cannot fully exploit the spectral information of the input HSI, and the inaccurate superpixel topological graph may lead to the confusion of different class semantics during information aggregation. To address these challenges, we first propose a structural-spectral graph convolutional operator (SSGCO) tailored for graph-structured HSI superpixels to improve their representation quality through the co-extraction of spatial and spectral features. Second, we propose an evidence-guided adaptive edge learning (EGAEL) module that adaptively predicts and refines edge weights in the superpixel topological graph. We integrate the proposed method into a contrastive learning framework to achieve clustering, where representation learning and clustering are simultaneously conducted. Experiments demonstrate that the proposed method improves clustering accuracy by 2.61%, 6.06%, 4.96% and 3.15% over the best compared methods on four HSI datasets. Our code is available at https://github.com/jhqi/SSGCO-EGAEL.
>
---
#### [replaced 039] End4: End-to-end Denoising Diffusion for Diffusion-Based Inpainting Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13214v2](http://arxiv.org/pdf/2509.13214v2)**

> **作者:** Fei Wang; Xuecheng Wu; Zheng Zhang; Danlei Huang; Yuheng Huang; Bo Wang
>
> **摘要:** The powerful generative capabilities of diffusion models have significantly advanced the field of image synthesis, enhancing both full image generation and inpainting-based image editing. Despite their remarkable advancements, diffusion models also raise concerns about potential misuse for malicious purposes. However, existing approaches struggle to identify images generated by diffusion-based inpainting models, even when similar inpainted images are included in their training data. To address this challenge, we propose a novel detection method based on End-to-end denoising diffusion (End4). Specifically, End4 designs a denoising reconstruction model to improve the alignment degree between the latent spaces of the reconstruction and detection processes, thus reconstructing features that are more conducive to detection. Meanwhile, it leverages a Scale-aware Pyramid-like Fusion Module (SPFM) that refines local image features under the guidance of attention pyramid layers at different scales, enhancing feature discriminability. Additionally, to evaluate detection performance on inpainted images, we establish a comprehensive benchmark comprising images generated from five distinct masked regions. Extensive experiments demonstrate that our End4 effectively generalizes to unseen masking patterns and remains robust under various perturbations. Our code and dataset will be released soon.
>
---
#### [replaced 040] Efficient motion-based metrics for video frame interpolation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09078v2](http://arxiv.org/pdf/2508.09078v2)**

> **作者:** Conall Daly; Darren Ramsook; Anil Kokaram
>
> **备注:** SPIE 2025 - Applications of Digital Image Processing XLVIII accepted manuscript, 13 pages
>
> **摘要:** Video frame interpolation (VFI) offers a way to generate intermediate frames between consecutive frames of a video sequence. Although the development of advanced frame interpolation algorithms has received increased attention in recent years, assessing the perceptual quality of interpolated content remains an ongoing area of research. In this paper, we investigate simple ways to process motion fields, with the purposes of using them as video quality metric for evaluating frame interpolation algorithms. We evaluate these quality metrics using the BVI-VFI dataset which contains perceptual scores measured for interpolated sequences. From our investigation we propose a motion metric based on measuring the divergence of motion fields. This metric correlates reasonably with these perceptual scores (PLCC=0.51) and is more computationally efficient (x2.7 speedup) compared to FloLPIPS (a well known motion-based metric). We then use our new proposed metrics to evaluate a range of state of the art frame interpolation metrics and find our metrics tend to favour more perceptual pleasing interpolated frames that may not score highly in terms of PSNR or SSIM.
>
---
#### [replaced 041] Probing the Representational Power of Sparse Autoencoders in Vision Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.11277v2](http://arxiv.org/pdf/2508.11277v2)**

> **作者:** Matthew Lyle Olson; Musashi Hinck; Neale Ratzlaff; Changbai Li; Phillip Howard; Vasudev Lal; Shao-Yen Tseng
>
> **备注:** ICCV 2025 Findings
>
> **摘要:** Sparse Autoencoders (SAEs) have emerged as a popular tool for interpreting the hidden states of large language models (LLMs). By learning to reconstruct activations from a sparse bottleneck layer, SAEs discover interpretable features from the high-dimensional internal representations of LLMs. Despite their popularity with language models, SAEs remain understudied in the visual domain. In this work, we provide an extensive evaluation the representational power of SAEs for vision models using a broad range of image-based tasks. Our experimental results demonstrate that SAE features are semantically meaningful, improve out-of-distribution generalization, and enable controllable generation across three vision model architectures: vision embedding models, multi-modal LMMs and diffusion models. In vision embedding models, we find that learned SAE features can be used for OOD detection and provide evidence that they recover the ontological structure of the underlying model. For diffusion models, we demonstrate that SAEs enable semantic steering through text encoder manipulation and develop an automated pipeline for discovering human-interpretable attributes. Finally, we conduct exploratory experiments on multi-modal LLMs, finding evidence that SAE features reveal shared representations across vision and language modalities. Our study provides a foundation for SAE evaluation in vision models, highlighting their strong potential improving interpretability, generalization, and steerability in the visual domain.
>
---
#### [replaced 042] Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation
- **分类: cs.CV; cs.LG; 68T45 (Primary) 68T10, 68T07 (Secondary); F.1.2; F.1.4**

- **链接: [http://arxiv.org/pdf/2505.16360v2](http://arxiv.org/pdf/2505.16360v2)**

> **作者:** Estelle Chigot; Dennis G. Wilson; Meriem Ghrib; Thomas Oberlin
>
> **备注:** Published in Computer Vision and Image Understanding, September 2025 (CVIU 2025)
>
> **摘要:** Semantic segmentation models trained on synthetic data often perform poorly on real-world images due to domain gaps, particularly in adverse conditions where labeled data is scarce. Yet, recent foundation models enable to generate realistic images without any training. This paper proposes to leverage such diffusion models to improve the performance of vision models when learned on synthetic data. We introduce two novel techniques for semantically consistent style transfer using diffusion models: Class-wise Adaptive Instance Normalization and Cross-Attention (CACTI) and its extension with selective attention Filtering (CACTIF). CACTI applies statistical normalization selectively based on semantic classes, while CACTIF further filters cross-attention maps based on feature similarity, preventing artifacts in regions with weak cross-attention correspondences. Our methods transfer style characteristics while preserving semantic boundaries and structural coherence, unlike approaches that apply global transformations or generate content without constraints. Experiments using GTA5 as source and Cityscapes/ACDC as target domains show that our approach produces higher quality images with lower FID scores and better content preservation. Our work demonstrates that class-aware diffusion-based style transfer effectively bridges the synthetic-to-real domain gap even with minimal target domain data, advancing robust perception systems for challenging real-world applications. The source code is available at: https://github.com/echigot/cactif.
>
---
#### [replaced 043] Image-Text-Image Knowledge Transfer for Lifelong Person Re-Identification with Hybrid Clothing States
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.16600v2](http://arxiv.org/pdf/2405.16600v2)**

> **作者:** Qizao Wang; Xuelin Qian; Bin Li; Yanwei Fu; Xiangyang Xue
>
> **备注:** Accepted by TIP 2025
>
> **摘要:** With the continuous expansion of intelligent surveillance networks, lifelong person re-identification (LReID) has received widespread attention, pursuing the need of self-evolution across different domains. However, existing LReID studies accumulate knowledge with the assumption that people would not change their clothes. In this paper, we propose a more practical task, namely lifelong person re-identification with hybrid clothing states (LReID-Hybrid), which takes a series of cloth-changing and same-cloth domains into account during lifelong learning. To tackle the challenges of knowledge granularity mismatch and knowledge presentation mismatch in LReID-Hybrid, we take advantage of the consistency and generalization capabilities of the text space, and propose a novel framework, dubbed $Teata$, to effectively align, transfer, and accumulate knowledge in an "image-text-image" closed loop. Concretely, to achieve effective knowledge transfer, we design a Structured Semantic Prompt (SSP) learning to decompose the text prompt into several structured pairs to distill knowledge from the image space with a unified granularity of text description. Then, we introduce a Knowledge Adaptation and Projection (KAP) strategy, which tunes text knowledge via a slow-paced learner to adapt to different tasks without catastrophic forgetting. Extensive experiments demonstrate the superiority of our proposed $Teata$ for LReID-Hybrid as well as on conventional LReID benchmarks over advanced methods.
>
---
#### [replaced 044] Dual-Mode Deep Anomaly Detection for Medical Manufacturing: Structural Similarity and Feature Distance
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.05796v2](http://arxiv.org/pdf/2509.05796v2)**

> **作者:** Julio Zanon Diaz; Georgios Siogkas; Peter Corcoran
>
> **备注:** 20 pages, 6 figures, 14 tables
>
> **摘要:** Automated visual inspection in medical device manufacturing faces unique challenges, including small and imbalanced datasets, high-resolution imagery, and strict regulatory requirements. To address these, we propose two attention-guided autoencoder architectures for deep anomaly detection. The first employs a structural similarity-based scoring approach that enables lightweight, real-time defect detection with unsupervised thresholding and can be further enhanced through limited supervised tuning. The second applies a feature distance-based strategy using Mahalanobis scoring on reduced latent features, designed to monitor distributional shifts and support supervisory oversight. Evaluations on a representative sterile packaging dataset confirm that both approaches outperform baselines under hardware-constrained, regulated conditions. Cross-domain testing on the MVTec-Zipper benchmark further demonstrates that the structural similarity-based method generalises effectively and achieves performance comparable to state-of-the-art methods, while the feature distance-based method is less transferable but provides complementary monitoring capabilities. These results highlight a dual-pathway inspection strategy: structural similarity for robust inline detection and feature distance for supervisory monitoring. By combining operational performance with interpretability and lifecycle monitoring, the proposed methods also align with emerging regulatory expectations for high-risk AI systems.
>
---
#### [replaced 045] DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.02842v3](http://arxiv.org/pdf/2406.02842v3)**

> **作者:** Paul Couairon; Mustafa Shukor; Jean-Emmanuel Haugeard; Matthieu Cord; Nicolas Thome
>
> **备注:** NeurIPS 2024. Project page at https://diffcut-segmentation.github.io. Code at https://github.com/PaulCouairon/DiffCut
>
> **摘要:** Foundation models have emerged as powerful tools across various domains including language, vision, and multimodal tasks. While prior works have addressed unsupervised image segmentation, they significantly lag behind supervised models. In this paper, we use a diffusion UNet encoder as a foundation vision encoder and introduce DiffCut, an unsupervised zero-shot segmentation method that solely harnesses the output features from the final self-attention block. Through extensive experimentation, we demonstrate that the utilization of these diffusion features in a graph based segmentation algorithm, significantly outperforms previous state-of-the-art methods on zero-shot segmentation. Specifically, we leverage a recursive Normalized Cut algorithm that softly regulates the granularity of detected objects and produces well-defined segmentation maps that precisely capture intricate image details. Our work highlights the remarkably accurate semantic knowledge embedded within diffusion UNet encoders that could then serve as foundation vision encoders for downstream tasks. Project page at https://diffcut-segmentation.github.io
>
---
#### [replaced 046] T-SYNTH: A Knowledge-Based Dataset of Synthetic Breast Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04038v2](http://arxiv.org/pdf/2507.04038v2)**

> **作者:** Christopher Wiedeman; Anastasiia Sarmakeeva; Elena Sizikova; Daniil Filienko; Miguel Lago; Jana G. Delfino; Aldo Badano
>
> **备注:** International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) Open Data 2025
>
> **摘要:** One of the key impediments for developing and assessing robust medical imaging algorithms is limited access to large-scale datasets with suitable annotations. Synthetic data generated with plausible physical and biological constraints may address some of these data limitations. We propose the use of physics simulations to generate synthetic images with pixel-level segmentation annotations, which are notoriously difficult to obtain. Specifically, we apply this approach to breast imaging analysis and release T-SYNTH, a large-scale open-source dataset of paired 2D digital mammography (DM) and 3D digital breast tomosynthesis (DBT) images. Our initial experimental results indicate that T-SYNTH images show promise for augmenting limited real patient datasets for detection tasks in DM and DBT. Our data and code are publicly available at https://github.com/DIDSR/tsynth-release.
>
---
#### [replaced 047] Domain Generalization for In-Orbit 6D Pose Estimation
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2406.11743v2](http://arxiv.org/pdf/2406.11743v2)**

> **作者:** Antoine Legrand; Renaud Detry; Christophe De Vleeschouwer
>
> **备注:** accepted at AIAA Journal of Aerospace Information Systems (12 pages, 6 figures)
>
> **摘要:** We address the problem of estimating the relative 6D pose, i.e., position and orientation, of a target spacecraft, from a monocular image, a key capability for future autonomous Rendezvous and Proximity Operations. Due to the difficulty of acquiring large sets of real images, spacecraft pose estimation networks are exclusively trained on synthetic ones. However, because those images do not capture the illumination conditions encountered in orbit, pose estimation networks face a domain gap problem, i.e., they do not generalize to real images. Our work introduces a method that bridges this domain gap. It relies on a novel, end-to-end, neural-based architecture as well as a novel learning strategy. This strategy improves the domain generalization abilities of the network through multi-task learning and aggressive data augmentation policies, thereby enforcing the network to learn domain-invariant features. We demonstrate that our method effectively closes the domain gap, achieving state-of-the-art accuracy on the widespread SPEED+ dataset. Finally, ablation studies assess the impact of key components of our method on its generalization abilities.
>
---
#### [replaced 048] SCORPION: Addressing Scanner-Induced Variability in Histopathology
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.20907v2](http://arxiv.org/pdf/2507.20907v2)**

> **作者:** Jeongun Ryu; Heon Song; Seungeun Lee; Soo Ick Cho; Jiwon Shin; Kyunghyun Paeng; Sérgio Pereira
>
> **备注:** Accepted in UNSURE 2025 workshop in MICCAI
>
> **摘要:** Ensuring reliable model performance across diverse domains is a critical challenge in computational pathology. A particular source of variability in Whole-Slide Images is introduced by differences in digital scanners, thus calling for better scanner generalization. This is critical for the real-world adoption of computational pathology, where the scanning devices may differ per institution or hospital, and the model should not be dependent on scanner-induced details, which can ultimately affect the patient's diagnosis and treatment planning. However, past efforts have primarily focused on standard domain generalization settings, evaluating on unseen scanners during training, without directly evaluating consistency across scanners for the same tissue. To overcome this limitation, we introduce SCORPION, a new dataset explicitly designed to evaluate model reliability under scanner variability. SCORPION includes 480 tissue samples, each scanned with 5 scanners, yielding 2,400 spatially aligned patches. This scanner-paired design allows for the isolation of scanner-induced variability, enabling a rigorous evaluation of model consistency while controlling for differences in tissue composition. Furthermore, we propose SimCons, a flexible framework that combines augmentation-based domain generalization techniques with a consistency loss to explicitly address scanner generalization. We empirically show that SimCons improves model consistency on varying scanners without compromising task-specific performance. By releasing the SCORPION dataset and proposing SimCons, we provide the research community with a crucial resource for evaluating and improving model consistency across diverse scanners, setting a new standard for reliability testing.
>
---
#### [replaced 049] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.16815v2](http://arxiv.org/pdf/2507.16815v2)**

> **作者:** Chi-Pin Huang; Yueh-Hua Wu; Min-Hung Chen; Yu-Chiang Frank Wang; Fu-En Yang
>
> **备注:** NeurIPS 2025. Project page: https://jasper0314-huang.github.io/thinkact-vla/
>
> **摘要:** Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks.
>
---
#### [replaced 050] BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.13789v2](http://arxiv.org/pdf/2509.13789v2)**

> **作者:** Hanshuai Cui; Zhiqing Tang; Zhifei Xu; Zhi Yao; Wenyi Zeng; Weijia Jia
>
> **摘要:** Recent advancements in Diffusion Transformers (DiTs) have established them as the state-of-the-art method for video generation. However, their inherently sequential denoising process results in inevitable latency, limiting real-world applicability. Existing acceleration methods either compromise visual quality due to architectural modifications or fail to reuse intermediate features at proper granularity. Our analysis reveals that DiT blocks are the primary contributors to inference latency. Across diffusion timesteps, the feature variations of DiT blocks exhibit a U-shaped pattern with high similarity during intermediate timesteps, which suggests substantial computational redundancy. In this paper, we propose Block-Wise Caching (BWCache), a training-free method to accelerate DiT-based video generation. BWCache dynamically caches and reuses features from DiT blocks across diffusion timesteps. Furthermore, we introduce a similarity indicator that triggers feature reuse only when the differences between block features at adjacent timesteps fall below a threshold, thereby minimizing redundant computations while maintaining visual fidelity. Extensive experiments on several video diffusion models demonstrate that BWCache achieves up to 2.24$\times$ speedup with comparable visual quality.
>
---
#### [replaced 051] Direct Video-Based Spatiotemporal Deep Learning for Cattle Lameness Detection
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.16404v4](http://arxiv.org/pdf/2504.16404v4)**

> **作者:** Md Fahimuzzman Sohan; Raid Alzubi; Hadeel Alzoubi; Eid Albalawi; A. H. Abdul Hafez
>
> **摘要:** Cattle lameness is a prevalent health problem in livestock farming, often resulting from hoof injuries or infections, and severely impacts animal welfare and productivity. Early and accurate detection is critical for minimizing economic losses and ensuring proper treatment. This study proposes a spatiotemporal deep learning framework for automated cattle lameness detection using publicly available video data. We curate and publicly release a balanced set of 50 online video clips featuring 42 individual cattle, recorded from multiple viewpoints in both indoor and outdoor environments. The videos were categorized into lame and non-lame classes based on visual gait characteristics and metadata descriptions. After applying data augmentation techniques to enhance generalization, two deep learning architectures were trained and evaluated: 3D Convolutional Neural Networks (3D CNN) and Convolutional Long-Short-Term Memory (ConvLSTM2D). The 3D CNN achieved a video-level classification accuracy of 90%, with a precision, recall, and F1 score of 90.9% each, outperforming the ConvLSTM2D model, which achieved 85% accuracy. Unlike conventional approaches that rely on multistage pipelines involving object detection and pose estimation, this study demonstrates the effectiveness of a direct end-to-end video classification approach. Compared with the best end-to-end prior method (C3D-ConvLSTM, 90.3%), our model achieves comparable accuracy while eliminating pose estimation pre-processing.The results indicate that deep learning models can successfully extract and learn spatio-temporal features from various video sources, enabling scalable and efficient cattle lameness detection in real-world farm settings.
>
---
#### [replaced 052] EnCoBo: Energy-Guided Concept Bottlenecks for Interpretable Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08334v2](http://arxiv.org/pdf/2507.08334v2)**

> **作者:** Sangwon Kim; Kyoungoh Lee; Jeyoun Dong; Jung Hwan Ahn; Kwang-Ju Kim
>
> **备注:** The original version was accepted by ICCV2025 Workshops
>
> **摘要:** Concept Bottleneck Models (CBMs) provide interpretable decision-making through explicit, human-understandable concepts. However, existing generative CBMs often rely on auxiliary visual cues at the bottleneck, which undermines interpretability and intervention capabilities. We propose EnCoBo, a post-hoc concept bottleneck for generative models that eliminates auxiliary cues by constraining all representations to flow solely through explicit concepts. Unlike autoencoder-based approaches that inherently rely on black-box decoders, EnCoBo leverages a decoder-free, energy-based framework that directly guides generation in the latent space. Guided by diffusion-scheduled energy functions, EnCoBo supports robust post-hoc interventions-such as concept composition and negation-across arbitrary concepts. Experiments on CelebA-HQ and CUB datasets showed that EnCoBo improved concept-level human intervention and interpretability while maintaining competitive visual quality.
>
---
#### [replaced 053] Roll Your Eyes: Gaze Redirection via Explicit 3D Eyeball Rotation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06136v2](http://arxiv.org/pdf/2508.06136v2)**

> **作者:** YoungChan Choi; HengFei Wang; YiHua Cheng; Boeun Kim; Hyung Jin Chang; YoungGeun Choi; Sang-Il Choi
>
> **备注:** 9 pages, 5 figures, ACM Multimeida 2025 accepted
>
> **摘要:** We propose a novel 3D gaze redirection framework that leverages an explicit 3D eyeball structure. Existing gaze redirection methods are typically based on neural radiance fields, which employ implicit neural representations via volume rendering. Unlike these NeRF-based approaches, where the rotation and translation of 3D representations are not explicitly modeled, we introduce a dedicated 3D eyeball structure to represent the eyeballs with 3D Gaussian Splatting (3DGS). Our method generates photorealistic images that faithfully reproduce the desired gaze direction by explicitly rotating and translating the 3D eyeball structure. In addition, we propose an adaptive deformation module that enables the replication of subtle muscle movements around the eyes. Through experiments conducted on the ETH-XGaze dataset, we demonstrate that our framework is capable of generating diverse novel gaze images, achieving superior image quality and gaze estimation accuracy compared to previous state-of-the-art methods.
>
---
#### [replaced 054] Standardizing Generative Face Video Compression using Supplemental Enhancement Information
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15105v3](http://arxiv.org/pdf/2410.15105v3)**

> **作者:** Bolin Chen; Yan Ye; Jie Chen; Ru-Ling Liao; Shanzhi Yin; Shiqi Wang; Kaifa Yang; Yue Li; Yiling Xu; Ye-Kui Wang; Shiv Gehlot; Guan-Ming Su; Peng Yin; Sean McCarthy; Gary J. Sullivan
>
> **摘要:** This paper proposes a Generative Face Video Compression (GFVC) approach using Supplemental Enhancement Information (SEI), where a series of compact spatial and temporal representations of a face video signal (e.g., 2D/3D keypoints, facial semantics and compact features) can be coded using SEI messages and inserted into the coded video bitstream. At the time of writing, the proposed GFVC approach using SEI messages has been included into a draft amendment of the Versatile Supplemental Enhancement Information (VSEI) standard by the Joint Video Experts Team (JVET) of ISO/IEC JTC 1/SC 29 and ITU-T SG21, which will be standardized as a new version of ITU-T H.274 | ISO/IEC 23002-7. To the best of the authors' knowledge, the JVET work on the proposed SEI-based GFVC approach is the first standardization activity for generative video compression. The proposed SEI approach has not only advanced the reconstruction quality of early-day Model-Based Coding (MBC) via the state-of-the-art generative technique, but also established a new SEI definition for future GFVC applications and deployment. Experimental results illustrate that the proposed SEI-based GFVC approach can achieve remarkable rate-distortion performance compared with the latest Versatile Video Coding (VVC) standard, whilst also potentially enabling a wide variety of functionalities including user-specified animation/filtering and metaverse-related applications.
>
---
#### [replaced 055] BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.21085v3](http://arxiv.org/pdf/2502.21085v3)**

> **作者:** Jing-Yuan Chang
>
> **备注:** 8 pages main paper, 2 pages references, 8 pages supplementary material
>
> **摘要:** Badminton, known for having the fastest ball speeds among all sports, presents significant challenges to the field of computer vision, including player identification, court line detection, shuttlecock trajectory tracking, and player stroke-type classification. In this paper, we introduce a novel video clipping strategy to extract frames of each player's racket swing in a badminton broadcast match. These clipped frames are then processed by three existing models: one for Human Pose Estimation to obtain human skeletal joints, another for shuttlecock trajectory tracking, and the other for court line detection to determine player positions on the court. Leveraging these data as inputs, we propose Badminton Stroke-type Transformer (BST) to classify player stroke-types in singles. To the best of our knowledge, experimental results demonstrate that our method outperforms the previous state-of-the-art on the largest publicly available badminton video dataset (ShuttleSet), another badminton dataset (BadmintonDB), and a tennis dataset (TenniSet). These results suggest that effectively leveraging ball trajectory is a promising direction for action recognition in racket sports.
>
---
#### [replaced 056] Erased or Dormant? Rethinking Concept Erasure Through Reversibility
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16174v2](http://arxiv.org/pdf/2505.16174v2)**

> **作者:** Ping Liu; Chi Zhang
>
> **备注:** Dr. Chi Zhang is the corresponding author
>
> **摘要:** To what extent does concept erasure eliminate generative capacity in diffusion models? While prior evaluations have primarily focused on measuring concept suppression under specific textual prompts, we explore a complementary and fundamental question: do current concept erasure techniques genuinely remove the ability to generate targeted concepts, or do they merely achieve superficial, prompt-specific suppression? We systematically evaluate the robustness and reversibility of two representative concept erasure methods, Unified Concept Editing and Erased Stable Diffusion, by probing their ability to eliminate targeted generative behaviors in text-to-image models. These methods attempt to suppress undesired semantic concepts by modifying internal model parameters, either through targeted attention edits or model-level fine-tuning strategies. To rigorously assess whether these techniques truly erase generative capacity, we propose an instance-level evaluation strategy that employs lightweight fine-tuning to explicitly test the reactivation potential of erased concepts. Through quantitative metrics and qualitative analyses, we show that erased concepts often reemerge with substantial visual fidelity after minimal adaptation, indicating that current methods suppress latent generative representations without fully eliminating them. Our findings reveal critical limitations in existing concept erasure approaches and highlight the need for deeper, representation-level interventions and more rigorous evaluation standards to ensure genuine, irreversible removal of concepts from generative models.
>
---
#### [replaced 057] Debias your Large Multi-Modal Model at Test-Time via Non-Contrastive Visual Attribute Steering
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.12590v3](http://arxiv.org/pdf/2411.12590v3)**

> **作者:** Neale Ratzlaff; Matthew Lyle Olson; Musashi Hinck; Estelle Aflalo; Shao-Yen Tseng; Vasudev Lal; Phillip Howard
>
> **备注:** 10 pages, 6 Figures, 8 Tables. arXiv admin note: text overlap with arXiv:2410.13976
>
> **摘要:** Large Multi-Modal Models (LMMs) have demonstrated impressive capabilities as general-purpose chatbots able to engage in conversations about visual inputs. However, their responses are influenced by societal biases present in their training datasets, leading to undesirable differences in how the model responds when presented with images depicting people of different demographics. In this work, we propose a training-free debiasing framework for LMMs that intervenes on the model's representations during text generation by constructing a steering vector that reduces reference on protected attributes. Our framework introduces two complementary methods: (1) a dataset-based approach that constructs a steering vector by contrasting model activations on biased and neutral inputs, and (2) a novel optimization-based approach designed for low-resource settings, which constructs the steering vector using a single step of gradient-based perturbation without requiring additional data. Our experiments show that these interventions effectively reduce the propensity of LMMs to generate text related to protected attributes while maintaining sentiment and fluency. Furthermore, we demonstrate that debiased LMMs achieve comparable accuracy to their unmodified counterparts on downstream tasks, indicating that bias mitigation can be achieved without sacrificing model performance.
>
---
#### [replaced 058] FASL-Seg: Anatomy and Tool Segmentation of Surgical Scenes
- **分类: eess.IV; cs.AI; cs.CV; I.4.6; I.4.8; J.3**

- **链接: [http://arxiv.org/pdf/2509.06159v2](http://arxiv.org/pdf/2509.06159v2)**

> **作者:** Muraam Abdel-Ghani; Mahmoud Ali; Mohamed Ali; Fatmaelzahraa Ahmed; Muhammad Arsalan; Abdulaziz Al-Ali; Shidin Balakrishnan
>
> **备注:** 8 pages, 6 figures, Accepted at the European Conference on Artificial Intelligence (ECAI) 2025. To appear in the conference proceedings
>
> **摘要:** The growing popularity of robotic minimally invasive surgeries has made deep learning-based surgical training a key area of research. A thorough understanding of the surgical scene components is crucial, which semantic segmentation models can help achieve. However, most existing work focuses on surgical tools and overlooks anatomical objects. Additionally, current state-of-the-art (SOTA) models struggle to balance capturing high-level contextual features and low-level edge features. We propose a Feature-Adaptive Spatial Localization model (FASL-Seg), designed to capture features at multiple levels of detail through two distinct processing streams, namely a Low-Level Feature Projection (LLFP) and a High-Level Feature Projection (HLFP) stream, for varying feature resolutions - enabling precise segmentation of anatomy and surgical instruments. We evaluated FASL-Seg on surgical segmentation benchmark datasets EndoVis18 and EndoVis17 on three use cases. The FASL-Seg model achieves a mean Intersection over Union (mIoU) of 72.71% on parts and anatomy segmentation in EndoVis18, improving on SOTA by 5%. It further achieves a mIoU of 85.61% and 72.78% in EndoVis18 and EndoVis17 tool type segmentation, respectively, outperforming SOTA overall performance, with comparable per-class SOTA results in both datasets and consistent performance in various classes for anatomy and instruments, demonstrating the effectiveness of distinct processing streams for varying feature resolutions.
>
---
#### [replaced 059] Universal Gröbner Bases of (Universal) Multiview Ideals
- **分类: math.AC; cs.CV; math.AG**

- **链接: [http://arxiv.org/pdf/2509.12376v2](http://arxiv.org/pdf/2509.12376v2)**

> **作者:** Timothy Duff; Jack Kendrick; Rekha R. Thomas
>
> **摘要:** Multiview ideals arise from the geometry of image formation in pinhole cameras, and universal multiview ideals are their analogs for unknown cameras. We prove that a natural collection of polynomials form a universal Gr\"obner basis for both types of ideals using a criterion introduced by Huang and Larson, and include a proof of their criterion in our setting. Symmetry reduction and induction enable the method to be deployed on an infinite family of ideals. We also give an explicit description of the matroids on which the methodology depends, in the context of multiview ideals.
>
---
#### [replaced 060] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17240v3](http://arxiv.org/pdf/2411.17240v3)**

> **作者:** Junyuan Deng; Wei Yin; Xiaoyang Guo; Qian Zhang; Xiaotao Hu; Weiqiang Ren; Xiao-Xiao Long; Ping Tan
>
> **摘要:** In this paper, we present DM-Calib, a diffusion-based approach for estimating pinhole camera intrinsic parameters from a single input image. Monocular camera calibration is essential for many 3D vision tasks. However, most existing methods depend on handcrafted assumptions or are constrained by limited training data, resulting in poor generalization across diverse real-world images. Recent advancements in stable diffusion models, trained on massive data, have shown the ability to generate high-quality images with varied characteristics. Emerging evidence indicates that these models implicitly capture the relationship between camera focal length and image content. Building on this insight, we explore how to leverage the powerful priors of diffusion models for monocular pinhole camera calibration. Specifically, we introduce a new image-based representation, termed Camera Image, which losslessly encodes the numerical camera intrinsics and integrates seamlessly with the diffusion framework. Using this representation, we reformulate the problem of estimating camera intrinsics as the generation of a dense Camera Image conditioned on an input image. By fine-tuning a stable diffusion model to generate a Camera Image from a single RGB input, we can extract camera intrinsics via a RANSAC operation. We further demonstrate that our monocular calibration method enhances performance across various 3D tasks, including zero-shot metric depth estimation, 3D metrology, pose estimation and sparse-view reconstruction. Extensive experiments on multiple public datasets show that our approach significantly outperforms baselines and provides broad benefits to 3D vision tasks.
>
---
#### [replaced 061] Robust Shape Regularity Criteria for Superpixel Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1903.07146v3](http://arxiv.org/pdf/1903.07146v3)**

> **作者:** Rémi Giraud; Vinh-Thong Ta; Nicolas Papadakis
>
> **备注:** International Conference on Image Processing 2017
>
> **摘要:** Regular decompositions are necessary for most superpixel-based object recognition or tracking applications. So far in the literature, the regularity or compactness of a superpixel shape is mainly measured by its circularity. In this work, we first demonstrate that such measure is not adapted for superpixel evaluation, since it does not directly express regularity but circular appearance. Then, we propose a new metric that considers several shape regularity aspects: convexity, balanced repartition, and contour smoothness. Finally, we demonstrate that our measure is robust to scale and noise and enables to more relevantly compare superpixel methods.
>
---
#### [replaced 062] Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.13174v3](http://arxiv.org/pdf/2409.13174v3)**

> **作者:** Hao Cheng; Erjia Xiao; Yichi Wang; Chengyuan Yu; Mengshu Sun; Qiang Zhang; Yijie Guo; Kaidi Xu; Jize Zhang; Chao Shen; Philip Torr; Jindong Gu; Renjing Xu
>
> **摘要:** Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.
>
---
#### [replaced 063] TIDE: Achieving Balanced Subject-Driven Image Generation via Target-Instructed Diffusion Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06499v2](http://arxiv.org/pdf/2509.06499v2)**

> **作者:** Jibai Lin; Bo Ma; Yating Yang; Xi Zhou; Rong Ma; Turghun Osman; Ahtamjan Ahmat; Rui Dong; Lei Wang
>
> **摘要:** Subject-driven image generation (SDIG) aims to manipulate specific subjects within images while adhering to textual instructions, a task crucial for advancing text-to-image diffusion models. SDIG requires reconciling the tension between maintaining subject identity and complying with dynamic edit instructions, a challenge inadequately addressed by existing methods. In this paper, we introduce the Target-Instructed Diffusion Enhancing (TIDE) framework, which resolves this tension through target supervision and preference learning without test-time fine-tuning. TIDE pioneers target-supervised triplet alignment, modelling subject adaptation dynamics using a (reference image, instruction, target images) triplet. This approach leverages the Direct Subject Diffusion (DSD) objective, training the model with paired "winning" (balanced preservation-compliance) and "losing" (distorted) targets, systematically generated and evaluated via quantitative metrics. This enables implicit reward modelling for optimal preservation-compliance balance. Experimental results on standard benchmarks demonstrate TIDE's superior performance in generating subject-faithful outputs while maintaining instruction compliance, outperforming baseline methods across multiple quantitative metrics. TIDE's versatility is further evidenced by its successful application to diverse tasks, including structural-conditioned generation, image-to-image generation, and text-image interpolation. Our code is available at https://github.com/KomJay520/TIDE.
>
---
#### [replaced 064] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.14982v2](http://arxiv.org/pdf/2411.14982v2)**

> **作者:** Kaichen Zhang; Yifei Shen; Bo Li; Ziwei Liu
>
> **摘要:** Recent advances in Large Multimodal Models (LMMs) lead to significant breakthroughs in both academia and industry. One question that arises is how we, as humans, can understand their internal neural representations. This paper takes an initial step towards addressing this question by presenting a versatile framework to identify and interpret the semantics within LMMs. Specifically, 1) we first apply a Sparse Autoencoder(SAE) to disentangle the representations into human understandable features. 2) We then present an automatic interpretation framework to interpreted the open-semantic features learned in SAE by the LMMs themselves. We employ this framework to analyze the LLaVA-NeXT-8B model using the LLaVA-OV-72B model, demonstrating that these features can effectively steer the model's behavior. Our results contribute to a deeper understanding of why LMMs excel in specific tasks, including EQ tests, and illuminate the nature of their mistakes along with potential strategies for their rectification. These findings offer new insights into the internal mechanisms of LMMs and suggest parallels with the cognitive processes of the human brain.
>
---
#### [replaced 065] HPGN: Hybrid Priors-Guided Network for Compressed Low-Light Image Enhancement
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02373v2](http://arxiv.org/pdf/2504.02373v2)**

> **作者:** Hantang Li; Qiang Zhu; Xiandong Meng; Lei Xiong; Shuyuan Zhu; Xiaopeng Fan
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** In practical applications, low-light images are often compressed for efficient storage and transmission. Most existing methods disregard compression artifacts removal or hardly establish a unified framework for joint task enhancement of low-light images with varying compression qualities. To address this problem, we propose a hybrid priors-guided network (HPGN) that enhances compressed low-light images by integrating both compression and illumination priors. Our approach fully utilizes the JPEG quality factor (QF) and DCT quantization matrix to guide the design of efficient plug-and-play modules for joint tasks. Additionally, we employ a random QF generation strategy to guide model training, enabling a single model to enhance low-light images with different compression levels. Experimental results demonstrate the superiority of our proposed method..
>
---
#### [replaced 066] PVLM: Parsing-Aware Vision Language Model with Dynamic Contrastive Learning for Zero-Shot Deepfake Attribution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14129v2](http://arxiv.org/pdf/2504.14129v2)**

> **作者:** Yaning Zhang; Jiahe Zhang; Chunjie Ma; Weili Guan; Tian Gan; Zan Gao
>
> **摘要:** The challenge of tracing the source attribution of forged faces has gained significant attention due to the rapid advancement of generative models. However, existing deepfake attribution (DFA) works primarily focus on the interaction among various domains in vision modality, and other modalities such as texts and face parsing are not fully explored. Besides, they tend to fail to assess the generalization performance of deepfake attributors to unseen advanced generators like diffusion in a fine-grained manner. In this paper, we propose a novel parsing-aware vision language model with dynamic contrastive learning(PVLM) method for zero-shot deepfake attribution (ZS-DFA),which facilitates effective and fine-grained traceability to unseen advanced generators. Specifically, we conduct a novel and fine-grained ZS-DFA benchmark to evaluate the attribution performance of deepfake attributors to unseen advanced generators like diffusion. Besides, we propose an innovative parsing-guided vision language model with dynamic contrastive learning (PVLM) method to capture general and diverse attribution features. We are motivated by the observation that the preservation of source face attributes in facial images generated by GAN and diffusion models varies significantly. We employ the inherent face attributes preservation differences to capture face parsing-aware forgery representations. Therefore, we devise a novel parsing encoder to focus on global face attribute embeddings, enabling parsing-guided DFA representation learning via dynamic vision-parsing matching. Additionally, we present a novel deepfake attribution contrastive center loss to pull relevant generators closer and push irrelevant ones away, which can be introduced into DFA models to enhance traceability. Experimental results show that our model exceeds the state-of-the-art on the ZS-DFA benchmark via various protocol evaluations.
>
---
#### [replaced 067] Survivability of Backdoor Attacks on Unconstrained Face Recognition Systems
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01607v4](http://arxiv.org/pdf/2507.01607v4)**

> **作者:** Quentin Le Roux; Yannick Teglia; Teddy Furon; Philippe Loubet-Moundi; Eric Bourbao
>
> **摘要:** The widespread deployment of Deep Learning-based Face Recognition Systems raises multiple security concerns. While prior research has identified backdoor vulnerabilities on isolated components, Backdoor Attacks on real-world, unconstrained pipelines remain underexplored. This paper presents the first comprehensive system-level analysis of Backdoor Attacks targeting Face Recognition Systems and provides three contributions. We first show that face feature extractors trained with large margin metric learning losses are susceptible to Backdoor Attacks. By analyzing 20 pipeline configurations and 15 attack scenarios, we then reveal that a single backdoor can compromise an entire Face Recognition System. Finally, we propose effective best practices and countermeasures for stakeholders.
>
---
#### [replaced 068] GAF: Gaussian Action Field as a Dynamic World Model for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14135v3](http://arxiv.org/pdf/2506.14135v3)**

> **作者:** Ying Chai; Litao Deng; Ruizhi Shao; Jiajun Zhang; Liangjun Xing; Hongwen Zhang; Yebin Liu
>
> **备注:** http://chaiying1.github.io/GAF.github.io/project_page/
>
> **摘要:** Accurate scene perception is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we adopt a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing 4D modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF provides three interrelated outputs: reconstruction of the current scene, prediction of future frames, and estimation of init action via Gaussian motion. Furthermore, we employ an action-vision-aligned denoising framework, conditioned on a unified representation that combines the init action and the Gaussian perception, both generated by the GAF, to further obtain more precise actions. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR, +0.3864 SSIM and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average +7.3% success rate in robotic manipulation tasks over state-of-the-art methods.
>
---
#### [replaced 069] Mixture of Multicenter Experts in Multimodal AI for Debiased Radiotherapy Target Delineation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.00046v3](http://arxiv.org/pdf/2410.00046v3)**

> **作者:** Yujin Oh; Sangjoon Park; Xiang Li; Pengfei Jin; Yi Wang; Jonathan Paly; Jason Efstathiou; Annie Chan; Jun Won Kim; Hwa Kyung Byun; Ik Jae Lee; Jaeho Cho; Chan Woo Wee; Peng Shu; Peilong Wang; Nathan Yu; Jason Holmes; Jong Chul Ye; Quanzheng Li; Wei Liu; Woong Sub Koom; Jin Sung Kim; Kyungsang Kim
>
> **备注:** 12 pages, 5 figures, 4 tables, 1 supplementary material
>
> **摘要:** Clinical decision-making reflects diverse strategies shaped by regional patient populations and institutional protocols. However, most existing medical artificial intelligence (AI) models are trained on highly prevalent data patterns, which reinforces biases and fails to capture the breadth of clinical expertise. Inspired by the recent advances in Mixture of Experts (MoE), we propose a Mixture of Multicenter Experts (MoME) framework to address AI bias in the medical domain without requiring data sharing across institutions. MoME integrates specialized expertise from diverse clinical strategies to enhance model generalizability and adaptability across medical centers. We validate this framework using a multimodal target volume delineation model for prostate cancer radiotherapy. With few-shot training that combines imaging and clinical notes from each center, the model outperformed baselines, particularly in settings with high inter-center variability or limited data availability. Furthermore, MoME enables model customization to local clinical preferences without cross-institutional data exchange, making it especially suitable for resource-constrained settings while promoting broadly generalizable medical AI.
>
---
#### [replaced 070] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21448v2](http://arxiv.org/pdf/2505.21448v2)**

> **作者:** Ziqiao Peng; Jiwen Liu; Haoxian Zhang; Xiaoqiang Liu; Songlin Tang; Pengfei Wan; Di Zhang; Hongyan Liu; Jun He
>
> **备注:** Accepted as NeurIPS 2025 spotlight
>
> **摘要:** Lip synchronization is the task of aligning a speaker's lip movements in video with corresponding speech audio, and it is essential for creating realistic, expressive video content. However, existing methods often rely on reference frames and masked-frame inpainting, which limit their robustness to identity consistency, pose variations, facial occlusions, and stylized content. In addition, since audio signals provide weaker conditioning than visual cues, lip shape leakage from the original video will affect lip sync quality. In this paper, we present OmniSync, a universal lip synchronization framework for diverse visual scenarios. Our approach introduces a mask-free training paradigm using Diffusion Transformer models for direct frame editing without explicit masks, enabling unlimited-duration inference while maintaining natural facial dynamics and preserving character identity. During inference, we propose a flow-matching-based progressive noise initialization to ensure pose and identity consistency, while allowing precise mouth-region editing. To address the weak conditioning signal of audio, we develop a Dynamic Spatiotemporal Classifier-Free Guidance (DS-CFG) mechanism that adaptively adjusts guidance strength over time and space. We also establish the AIGC-LipSync Benchmark, the first evaluation suite for lip synchronization in diverse AI-generated videos. Extensive experiments demonstrate that OmniSync significantly outperforms prior methods in both visual quality and lip sync accuracy, achieving superior results in both real-world and AI-generated videos.
>
---
#### [replaced 071] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17651v4](http://arxiv.org/pdf/2502.17651v4)**

> **作者:** Bingxuan Li; Yiwei Wang; Jiuxiang Gu; Kai-Wei Chang; Nanyun Peng
>
> **备注:** ACL2025 Main
>
> **摘要:** Chart generation aims to generate code to produce charts satisfying the desired visual properties, e.g., texts, layout, color, and type. It has great potential to empower the automatic professional report generation in financial analysis, research presentation, education, and healthcare. In this work, we build a vision-language model (VLM) based multi-agent framework for effective automatic chart generation. Generating high-quality charts requires both strong visual design skills and precise coding capabilities that embed the desired visual properties into code. Such a complex multi-modal reasoning process is difficult for direct prompting of VLMs. To resolve these challenges, we propose METAL, a multi-agent framework that decomposes the task of chart generation into the iterative collaboration among specialized agents. METAL achieves 5.2% improvement over the current best result in the chart generation task. The METAL framework exhibits the phenomenon of test-time scaling: its performance increases monotonically as the logarithmic computational budget grows from 512 to 8192 tokens. In addition, we find that separating different modalities during the critique process of METAL boosts the self-correction capability of VLMs in the multimodal context.
>
---
#### [replaced 072] Preference Isolation Forest for Structure-based Anomaly Detection
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.10876v2](http://arxiv.org/pdf/2505.10876v2)**

> **作者:** Filippo Leveni; Luca Magri; Cesare Alippi; Giacomo Boracchi
>
> **备注:** Accepted at Pattern Recognition (2025)
>
> **摘要:** We address the problem of detecting anomalies as samples that do not conform to structured patterns represented by low-dimensional manifolds. To this end, we conceive a general anomaly detection framework called Preference Isolation Forest (PIF), that combines the benefits of adaptive isolation-based methods with the flexibility of preference embedding. The key intuition is to embed the data into a high-dimensional preference space by fitting low-dimensional manifolds, and to identify anomalies as isolated points. We propose three isolation approaches to identify anomalies: $i$) Voronoi-iForest, the most general solution, $ii$) RuzHash-iForest, that avoids explicit computation of distances via Local Sensitive Hashing, and $iii$) Sliding-PIF, that leverages a locality prior to improve efficiency and effectiveness.
>
---
#### [replaced 073] Deep Learning-Driven Multimodal Detection and Movement Analysis of Objects in Culinary
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00033v2](http://arxiv.org/pdf/2509.00033v2)**

> **作者:** Tahoshin Alam Ishat; Mohammad Abdul Qayum
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** This is a research exploring existing models and fine tuning them to combine a YOLOv8 segmentation model, a LSTM model trained on hand point motion sequence and a ASR (whisper-base) to extract enough data for a LLM (TinyLLaMa) to predict the recipe and generate text creating a step by step guide for the cooking procedure. All the data were gathered by the author for a robust task specific system to perform best in complex and challenging environments proving the extension and endless application of computer vision in daily activities such as kitchen work. This work extends the field for many more crucial task of our day to day life.
>
---
