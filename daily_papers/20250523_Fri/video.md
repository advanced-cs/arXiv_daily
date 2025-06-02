# 计算机视觉 cs.CV

- **最新发布 188 篇**

- **更新 86 篇**

## 最新发布

#### [new 001] REOBench: Benchmarking Robustness of Earth Observation Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于地球观测基础模型稳健性评估任务，旨在解决其在现实环境扰动下的鲁棒性不足问题。工作包括构建首个综合基准REOBench，覆盖6项任务和12类图像退化（含外观/几何扰动），系统评估多种预训练模型，揭示模型性能显著下降且差异显著，提出视觉语言模型更具鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.16793v1](http://arxiv.org/pdf/2505.16793v1)**

> **作者:** Xiang Li; Yong Tao; Siyuan Zhang; Siwei Liu; Zhitong Xiong; Chunbo Luo; Lu Liu; Mykola Pechenizkiy; Xiao Xiang Zhu; Tianjin Huang
>
> **备注:** 24 pages
>
> **摘要:** Earth observation foundation models have shown strong generalization across multiple Earth observation tasks, but their robustness under real-world perturbations remains underexplored. To bridge this gap, we introduce REOBench, the first comprehensive benchmark for evaluating the robustness of Earth observation foundation models across six tasks and twelve types of image corruptions, including both appearance-based and geometric perturbations. To ensure realistic and fine-grained evaluation, our benchmark focuses on high-resolution optical remote sensing images, which are widely used in critical applications such as urban planning and disaster response. We conduct a systematic evaluation of a broad range of models trained using masked image modeling, contrastive learning, and vision-language pre-training paradigms. Our results reveal that (1) existing Earth observation foundation models experience significant performance degradation when exposed to input corruptions. (2) The severity of degradation varies across tasks, model architectures, backbone sizes, and types of corruption, with performance drop varying from less than 1% to over 20%. (3) Vision-language models show enhanced robustness, particularly in multimodal tasks. REOBench underscores the vulnerability of current Earth observation foundation models to real-world corruptions and provides actionable insights for developing more robust and reliable models.
>
---
#### [new 002] Mesh-RFT: Enhancing Mesh Generation via Fine-grained Reinforcement Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文聚焦3D网格生成，解决预训练模型数据偏差与全局强化学习难以捕捉局部细节的问题。提出Mesh-RFT框架，采用M-DPO结合边界边比率（BER）和拓扑评分（TS）的细粒度评估，实现面级局部优化，提升几何与拓扑质量，达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.16761v1](http://arxiv.org/pdf/2505.16761v1)**

> **作者:** Jian Liu; Jing Xu; Song Guo; Jing Li; Jingfeng Guo; Jiaao Yu; Haohan Weng; Biwen Lei; Xianghui Yang; Zhuo Chen; Fangqi Zhu; Tao Han; Chunchao Guo
>
> **备注:** Under Review
>
> **摘要:** Existing pretrained models for 3D mesh generation often suffer from data biases and produce low-quality results, while global reinforcement learning (RL) methods rely on object-level rewards that struggle to capture local structure details. To address these challenges, we present \textbf{Mesh-RFT}, a novel fine-grained reinforcement fine-tuning framework that employs Masked Direct Preference Optimization (M-DPO) to enable localized refinement via quality-aware face masking. To facilitate efficient quality evaluation, we introduce an objective topology-aware scoring system to evaluate geometric integrity and topological regularity at both object and face levels through two metrics: Boundary Edge Ratio (BER) and Topology Score (TS). By integrating these metrics into a fine-grained RL strategy, Mesh-RFT becomes the first method to optimize mesh quality at the granularity of individual faces, resolving localized errors while preserving global coherence. Experiment results show that our M-DPO approach reduces Hausdorff Distance (HD) by 24.6\% and improves Topology Score (TS) by 3.8\% over pre-trained models, while outperforming global DPO methods with a 17.4\% HD reduction and 4.9\% TS gain. These results demonstrate Mesh-RFT's ability to improve geometric integrity and topological regularity, achieving new state-of-the-art performance in production-ready mesh generation. Project Page: \href{https://hitcslj.github.io/mesh-rft/}{this https URL}.
>
---
#### [new 003] Understanding Generative AI Capabilities in Everyday Image Editing Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文评估生成式AI在日常图像编辑中的能力。通过分析Reddit社区2013-2025年8.3万请求及30.5万次编辑，研究常见编辑需求及AI（如GPT-4o）的表现。发现AI在精准编辑（如保留人物特征）上逊于创意任务，仅满足33%请求，且常添加未要求修改。同时，VLM评估更倾向AI结果。旨在改进AI编辑器并明确其适用场景。**

- **链接: [http://arxiv.org/pdf/2505.16181v1](http://arxiv.org/pdf/2505.16181v1)**

> **作者:** Mohammad Reza Taesiri; Brandon Collins; Logan Bolton; Viet Dac Lai; Franck Dernoncourt; Trung Bui; Anh Totti Nguyen
>
> **备注:** Code and qualitative examples are available at: https://psrdataset.github.io
>
> **摘要:** Generative AI (GenAI) holds significant promise for automating everyday image editing tasks, especially following the recent release of GPT-4o on March 25, 2025. However, what subjects do people most often want edited? What kinds of editing actions do they want to perform (e.g., removing or stylizing the subject)? Do people prefer precise edits with predictable outcomes or highly creative ones? By understanding the characteristics of real-world requests and the corresponding edits made by freelance photo-editing wizards, can we draw lessons for improving AI-based editors and determine which types of requests can currently be handled successfully by AI editors? In this paper, we present a unique study addressing these questions by analyzing 83k requests from the past 12 years (2013-2025) on the Reddit community, which collected 305k PSR-wizard edits. According to human ratings, approximately only 33% of requests can be fulfilled by the best AI editors (including GPT-4o, Gemini-2.0-Flash, SeedEdit). Interestingly, AI editors perform worse on low-creativity requests that require precise editing than on more open-ended tasks. They often struggle to preserve the identity of people and animals, and frequently make non-requested touch-ups. On the other side of the table, VLM judges (e.g., o1) perform differently from human judges and may prefer AI edits more than human edits. Code and qualitative examples are available at: https://psrdataset.github.io
>
---
#### [new 004] Multilinear subspace learning for person re-identification based fusion of high order tensor features
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文针对行人重识别（PRe-ID）任务，提出融合CNN与LOMO特征的高阶张量方法（HDFF），通过张量融合和TXQDA多线性子空间学习降维，提升跨摄像头行人匹配精度，实验在三个数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15825v1](http://arxiv.org/pdf/2505.15825v1)**

> **作者:** Ammar Chouchane; Mohcene Bessaoudi; Hamza Kheddar; Abdelmalik Ouamane; Tiago Vieira; Mahmoud Hassaballah
>
> **摘要:** Video surveillance image analysis and processing is a challenging field in computer vision, with one of its most difficult tasks being Person Re-Identification (PRe-ID). PRe-ID aims to identify and track target individuals who have already been detected in a network of cameras, using a robust description of their pedestrian images. The success of recent research in person PRe-ID is largely due to effective feature extraction and representation, as well as the powerful learning of these features to reliably discriminate between pedestrian images. To this end, two powerful features, Convolutional Neural Networks (CNN) and Local Maximal Occurrence (LOMO), are modeled on multidimensional data using the proposed method, High-Dimensional Feature Fusion (HDFF). Specifically, a new tensor fusion scheme is introduced to leverage and combine these two types of features in a single tensor, even though their dimensions are not identical. To enhance the system's accuracy, we employ Tensor Cross-View Quadratic Analysis (TXQDA) for multilinear subspace learning, followed by cosine similarity for matching. TXQDA efficiently facilitates learning while reducing the high dimensionality inherent in high-order tensor data. The effectiveness of our approach is verified through experiments on three widely-used PRe-ID datasets: VIPeR, GRID, and PRID450S. Extensive experiments demonstrate that our approach outperforms recent state-of-the-art methods.
>
---
#### [new 005] SHaDe: Compact and Consistent Dynamic 3D Reconstruction via Tri-Plane Deformation and Latent Diffusion
- **分类: cs.CV**

- **简介: 该论文提出SHaDe框架，用于动态3D场景重建。针对传统方法效率低、时空表示复杂及时间一致性差的问题，采用三平面时变特征编码时空信息，通过显式变形场替代MLP运动建模，结合SH注意力渲染视图依赖颜色，并引入时序潜扩散模块优化特征，提升重建质量与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.16535v1](http://arxiv.org/pdf/2505.16535v1)**

> **作者:** Asrar Alruwayqi
>
> **摘要:** We present a novel framework for dynamic 3D scene reconstruction that integrates three key components: an explicit tri-plane deformation field, a view-conditioned canonical radiance field with spherical harmonics (SH) attention, and a temporally-aware latent diffusion prior. Our method encodes 4D scenes using three orthogonal 2D feature planes that evolve over time, enabling efficient and compact spatiotemporal representation. These features are explicitly warped into a canonical space via a deformation offset field, eliminating the need for MLP-based motion modeling. In canonical space, we replace traditional MLP decoders with a structured SH-based rendering head that synthesizes view-dependent color via attention over learned frequency bands improving both interpretability and rendering efficiency. To further enhance fidelity and temporal consistency, we introduce a transformer-guided latent diffusion module that refines the tri-plane and deformation features in a compressed latent space. This generative module denoises scene representations under ambiguous or out-of-distribution (OOD) motion, improving generalization. Our model is trained in two stages: the diffusion module is first pre-trained independently, and then fine-tuned jointly with the full pipeline using a combination of image reconstruction, diffusion denoising, and temporal consistency losses. We demonstrate state-of-the-art results on synthetic benchmarks, surpassing recent methods such as HexPlane and 4D Gaussian Splatting in visual quality, temporal coherence, and robustness to sparse-view dynamic inputs.
>
---
#### [new 006] RE-TRIP : Reflectivity Instance Augmented Triangle Descriptor for 3D Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D位置识别任务，旨在解决现有LiDAR方法忽视反射率信息导致的几何退化、相似结构及动态物体干扰下的识别鲁棒性不足问题。提出RE-TRIP描述子，融合几何与反射率数据，并设计关键点提取、实例分割、匹配及联合验证方法，实验证明其在公开数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16165v1](http://arxiv.org/pdf/2505.16165v1)**

> **作者:** Yechan Park; Gyuhyeon Pak; Euntai Kim
>
> **摘要:** While most people associate LiDAR primarily with its ability to measure distances and provide geometric information about the environment (via point clouds), LiDAR also captures additional data, including reflectivity or intensity values. Unfortunately, when LiDAR is applied to Place Recognition (PR) in mobile robotics, most previous works on LiDAR-based PR rely only on geometric measurements, neglecting the additional reflectivity information that LiDAR provides. In this paper, we propose a novel descriptor for 3D PR, named RE-TRIP (REflectivity-instance augmented TRIangle descriPtor). This new descriptor leverages both geometric measurements and reflectivity to enhance robustness in challenging scenarios such as geometric degeneracy, high geometric similarity, and the presence of dynamic objects. To implement RE-TRIP in real-world applications, we further propose (1) a keypoint extraction method, (2) a key instance segmentation method, (3) a RE-TRIP matching method, and (4) a reflectivity-combined loop verification method. Finally, we conduct a series of experiments to demonstrate the effectiveness of RE-TRIP. Applied to public datasets (i.e., HELIPR, FusionPortable) containing diverse scenarios such as long corridors, bridges, large-scale urban areas, and highly dynamic environments -- our experimental results show that the proposed method outperforms existing state-of-the-art methods in terms of Scan Context, Intensity Scan Context, and STD.
>
---
#### [new 007] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决视觉语言模型（VLM）与端到端（E2E）模型高效融合及实时决策问题。提出SOLVE框架，通过共享视觉编码器实现特征级交互，结合轨迹链式思考（T-CoT）渐进优化预测，并采用时间解耦策略平衡VLM输出与实时性，提升轨迹预测精度。**

- **链接: [http://arxiv.org/pdf/2505.16805v1](http://arxiv.org/pdf/2505.16805v1)**

> **作者:** Xuesong Chen; Linjiang Huang; Tao Ma; Rongyao Fang; Shaoshuai Shi; Hongsheng Li
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** The integration of Vision-Language Models (VLMs) into autonomous driving systems has shown promise in addressing key challenges such as learning complexity, interpretability, and common-sense reasoning. However, existing approaches often struggle with efficient integration and realtime decision-making due to computational demands. In this paper, we introduce SOLVE, an innovative framework that synergizes VLMs with end-to-end (E2E) models to enhance autonomous vehicle planning. Our approach emphasizes knowledge sharing at the feature level through a shared visual encoder, enabling comprehensive interaction between VLM and E2E components. We propose a Trajectory Chain-of-Thought (T-CoT) paradigm, which progressively refines trajectory predictions, reducing uncertainty and improving accuracy. By employing a temporal decoupling strategy, SOLVE achieves efficient cooperation by aligning high-quality VLM outputs with E2E real-time performance. Evaluated on the nuScenes dataset, our method demonstrates significant improvements in trajectory prediction accuracy, paving the way for more robust and reliable autonomous driving systems.
>
---
#### [new 008] Perceptual Quality Assessment for Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Embodied-IQA任务，解决传统图像质量评估无法评估机器人感知实用性的难题。基于感知-认知-决策管道构建框架，创建含3.6万图像对的数据库，并验证现有方法不足，推动开发更精准的具身AI质量评估指标。**

- **链接: [http://arxiv.org/pdf/2505.16815v1](http://arxiv.org/pdf/2505.16815v1)**

> **作者:** Chunyi Li; Jiaohao Xiao; Jianbo Zhang; Farong Wen; Zicheng Zhang; Yuan Tian; Xiangyang Zhu; Xiaohong Liu; Zhengxue Cheng; Weisi Lin; Guangtao Zhai
>
> **摘要:** Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 36k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world. Project page: https://github.com/lcysyzxdxc/EmbodiedIQA
>
---
#### [new 009] RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs
- **分类: cs.CV**

- **简介: 该论文提出RBench-V基准，评估视觉推理模型的多模态输出能力（如生成图像辅助推理），解决现有评测忽视此类输出的问题。通过803道需图像操作的题目测试模型，发现最佳模型准确率仅25.8%，远低于人类水平，凸显当前模型在多模态输出推理上的不足。**

- **链接: [http://arxiv.org/pdf/2505.16770v1](http://arxiv.org/pdf/2505.16770v1)**

> **作者:** Meng-Hao Guo; Xuanyu Chu; Qianrui Yang; Zhe-Han Mo; Yiqing Shen; Pei-lin Li; Xinjie Lin; Jinnian Zhang; Xin-Sheng Chen; Yi Zhang; Kiyohiro Nakayama; Zhengyang Geng; Houwen Peng; Han Hu; Shi-Nin Hu
>
> **备注:** 12 pages
>
> **摘要:** The rapid advancement of native multi-modal models and omni-models, exemplified by GPT-4o, Gemini, and o3, with their capability to process and generate content across modalities such as text and images, marks a significant milestone in the evolution of intelligence. Systematic evaluation of their multi-modal output capabilities in visual thinking processes (also known as multi-modal chain of thought, M-CoT) becomes critically important. However, existing benchmarks for evaluating multi-modal models primarily focus on assessing multi-modal inputs and text-only reasoning while neglecting the importance of reasoning through multi-modal outputs. In this paper, we present a benchmark, dubbed RBench-V, designed to assess models' vision-indispensable reasoning abilities. To construct RBench-V, we carefully hand-pick 803 questions covering math, physics, counting, and games. Unlike previous benchmarks that typically specify certain input modalities, RBench-V presents problems centered on multi-modal outputs, which require image manipulation such as generating novel images and constructing auxiliary lines to support the reasoning process. We evaluate numerous open- and closed-source models on RBench-V, including o3, Gemini 2.5 Pro, Qwen2.5-VL, etc. Even the best-performing model, o3, achieves only 25.8% accuracy on RBench-V, far below the human score of 82.3%, highlighting that current models struggle to leverage multi-modal reasoning. Data and code are available at https://evalmodels.github.io/rbenchv
>
---
#### [new 010] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于端到端自动驾驶任务，旨在解决多模态数据处理与复杂场景（如急转弯）的泛化问题。提出DriveMoE框架，结合视觉MoE（动态选择关键摄像头）和动作MoE（专精不同驾驶行为的专家模块），避免传统模型模式平均缺陷，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.16278v1](http://arxiv.org/pdf/2505.16278v1)**

> **作者:** Zhenjie Yang; Yilin Chai; Xiaosong Jia; Qifeng Li; Yuqian Shao; Xuekai Zhu; Haisheng Su; Junchi Yan
>
> **备注:** Project Page: https://thinklab-sjtu.github.io/DriveMoE/
>
> **摘要:** End-to-end autonomous driving (E2E-AD) demands effective processing of multi-view sensory data and robust handling of diverse and complex driving scenarios, particularly rare maneuvers such as aggressive turns. Recent success of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs) demonstrates that specialization of parameters enables strong scalability. In this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is built upon our $\pi_0$ Vision-Language-Action (VLA) baseline (originally from the embodied AI field), called Drive-$\pi_0$. Specifically, we add Vision MoE to Drive-$\pi_0$ by training a router to select relevant cameras according to the driving context dynamically. This design mirrors human driving cognition, where drivers selectively attend to crucial visual cues rather than exhaustively processing all visual information. In addition, we add Action MoE by training another router to activate specialized expert modules for different driving behaviors. Through explicit behavioral specialization, DriveMoE is able to handle diverse scenarios without suffering from modes averaging like existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of combining vision and action MoE in autonomous driving tasks. We will release our code and models of DriveMoE and Drive-$\pi_0$.
>
---
#### [new 011] NTIRE 2025 challenge on Text to Image Generation Model Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文描述了NTIRE 2025文本到图像（T2I）生成模型质量评估挑战赛，旨在通过图像-文本对齐和结构失真检测两个赛道，解决细粒度模型质量评估问题。使用EvalMuse-40K和EvalMuse-Structure数据集，吸引大量参赛团队，最终获胜方法在质量评估中表现优于基线。**

- **链接: [http://arxiv.org/pdf/2505.16314v1](http://arxiv.org/pdf/2505.16314v1)**

> **作者:** Shuhao Han; Haotian Fan; Fangyuan Kong; Wenjie Liao; Chunle Guo; Chongyi Li; Radu Timofte; Liang Li; Tao Li; Junhui Cui; Yunqiu Wang; Yang Tai; Jingwei Sun; Jianhui Sun; Xinli Yue; Tianyi Wang; Huan Hou; Junda Lu; Xinyang Huang; Zitang Zhou; Zijian Zhang; Xuhui Zheng; Xuecheng Wu; Chong Peng; Xuezhi Cao; Trong-Hieu Nguyen-Mau; Minh-Hoang Le; Minh-Khoa Le-Phan; Duy-Nam Ly; Hai-Dang Nguyen; Minh-Triet Tran; Yukang Lin; Yan Hong; Chuanbiao Song; Siyuan Li; Jun Lan; Zhichao Zhang; Xinyue Li; Wei Sun; Zicheng Zhang; Yunhao Li; Xiaohong Liu; Guangtao Zhai; Zitong Xu; Huiyu Duan; Jiarui Wang; Guangji Ma; Liu Yang; Lu Liu; Qiang Hu; Xiongkuo Min; Zichuan Wang; Zhenchen Tang; Bo Peng; Jing Dong; Fengbin Guan; Zihao Yu; Yiting Lu; Wei Luo; Xin Li; Minhao Lin; Haofeng Chen; Xuanxuan He; Kele Xu; Qisheng Xu; Zijian Gao; Tianjiao Wan; Bo-Cheng Qiu; Chih-Chung Hsu; Chia-ming Lee; Yu-Fan Lin; Bo Yu; Zehao Wang; Da Mu; Mingxiu Chen; Junkang Fang; Huamei Sun; Wending Zhao; Zhiyu Wang; Wang Liu; Weikang Yu; Puhong Duan; Bin Sun; Xudong Kang; Shutao Li; Shuai He; Lingzhi Fu; Heng Cong; Rongyu Zhang; Jiarong He; Zhishan Qiao; Yongqing Huang; Zewen Chen; Zhe Pang; Juan Wang; Jian Guo; Zhizhuo Shao; Ziyu Feng; Bing Li; Weiming Hu; Hesong Li; Dehua Liu; Zeming Liu; Qingsong Xie; Ruichen Wang; Zhihao Li; Yuqi Liang; Jianqi Bi; Jun Luo; Junfeng Yang; Can Li; Jing Fu; Hongwei Xu; Mingrui Long; Lulin Tang
>
> **摘要:** This paper reports on the NTIRE 2025 challenge on Text to Image (T2I) generation model quality assessment, which will be held in conjunction with the New Trends in Image Restoration and Enhancement Workshop (NTIRE) at CVPR 2025. The aim of this challenge is to address the fine-grained quality assessment of text-to-image generation models. This challenge evaluates text-to-image models from two aspects: image-text alignment and image structural distortion detection, and is divided into the alignment track and the structural track. The alignment track uses the EvalMuse-40K, which contains around 40K AI-Generated Images (AIGIs) generated by 20 popular generative models. The alignment track has a total of 371 registered participants. A total of 1,883 submissions are received in the development phase, and 507 submissions are received in the test phase. Finally, 12 participating teams submitted their models and fact sheets. The structure track uses the EvalMuse-Structure, which contains 10,000 AI-Generated Images (AIGIs) with corresponding structural distortion mask. A total of 211 participants have registered in the structure track. A total of 1155 submissions are received in the development phase, and 487 submissions are received in the test phase. Finally, 8 participating teams submitted their models and fact sheets. Almost all methods have achieved better results than baseline methods, and the winning methods in both tracks have demonstrated superior prediction performance on T2I model quality assessment.
>
---
#### [new 012] InspectionV3: Enhancing Tobacco Quality Assessment with Deep Convolutional Neural Networks for Automated Workshop Management
- **分类: cs.CV**

- **简介: 该论文提出InspectionV3系统，通过定制化卷积神经网络解决烟草加工中人工质检效率低、成本高的问题。基于21,113张图像的标注数据，模型融合颜色、成熟度等特征实现自动化分级，支持实时质检与数据驱动决策，准确率达97%，提升车间管理效能。**

- **链接: [http://arxiv.org/pdf/2505.16485v1](http://arxiv.org/pdf/2505.16485v1)**

> **作者:** Yao Wei; Muhammad Usman; Hazrat Bilal
>
> **备注:** 33 pages, 15 figures, 2 Tables
>
> **摘要:** The problems that tobacco workshops encounter include poor curing, inconsistencies in supplies, irregular scheduling, and a lack of oversight, all of which drive up expenses and worse quality. Large quantities make manual examination costly, sluggish, and unreliable. Deep convolutional neural networks have recently made strides in capabilities that transcend those of conventional methods. To effectively enhance them, nevertheless, extensive customization is needed to account for subtle variations in tobacco grade. This study introduces InspectionV3, an integrated solution for automated flue-cured tobacco grading that makes use of a customized deep convolutional neural network architecture. A scope that covers color, maturity, and curing subtleties is established via a labelled dataset consisting of 21,113 images spanning 20 quality classes. Expert annotators performed preprocessing on the tobacco leaf images, including cleaning, labelling, and augmentation. Multi-layer CNN factors use batch normalization to describe domain properties like as permeability and moisture spots, and so account for the subtleties of the workshop. Its expertise lies in converting visual patterns into useful information for enhancing workflow. Fast notifications are made possible by real-time, on-the-spot grading that matches human expertise. Images-powered analytics dashboards facilitate the tracking of yield projections, inventories, bottlenecks, and the optimization of data-driven choices. More labelled images are assimilated after further retraining, improving representational capacities and enabling adaptations for seasonal variability. Metrics demonstrate 97% accuracy, 95% precision and recall, 96% F1-score and AUC, 95% specificity; validating real-world viability.
>
---
#### [new 013] SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）推理任务，旨在解决现有方法缺乏对思维过程监督导致推理策略次优的问题。提出SophiaVL-R1，通过引入思维奖励模型、Trust-GRPO方法（基于可信度加权缓解奖励偏差）及退火训练策略（逐步降低思维奖励依赖），提升模型推理与泛化能力，在多项基准测试中超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.17018v1](http://arxiv.org/pdf/2505.17018v1)**

> **作者:** Kaixuan Fan; Kaituo Feng; Haoming Lyu; Dongzhan Zhou; Xiangyu Yue
>
> **备注:** Project page:https://github.com/kxfan2002/SophiaVL-R1
>
> **摘要:** Recent advances have shown success in eliciting strong reasoning abilities in multimodal large language models (MLLMs) through rule-based reinforcement learning (RL) with outcome rewards. However, this paradigm typically lacks supervision over the thinking process leading to the final outcome.As a result, the model may learn sub-optimal reasoning strategies, which can hinder its generalization ability. In light of this, we propose SophiaVL-R1, as an attempt to add reward signals for the thinking process in this paradigm. To achieve this, we first train a thinking reward model that evaluates the quality of the entire thinking process. Given that the thinking reward may be unreliable for certain samples due to reward hacking, we propose the Trust-GRPO method, which assigns a trustworthiness weight to the thinking reward during training. This weight is computed based on the thinking reward comparison of responses leading to correct answers versus incorrect answers, helping to mitigate the impact of potentially unreliable thinking rewards. Moreover, we design an annealing training strategy that gradually reduces the thinking reward over time, allowing the model to rely more on the accurate rule-based outcome reward in later training stages. Experiments show that our SophiaVL-R1 surpasses a series of reasoning MLLMs on various benchmarks (e.g., MathVisita, MMMU), demonstrating strong reasoning and generalization capabilities. Notably, our SophiaVL-R1-7B even outperforms LLaVA-OneVision-72B on most benchmarks, despite the latter having 10 times more parameters. All code, models, and datasets are made publicly available at https://github.com/kxfan2002/SophiaVL-R1.
>
---
#### [new 014] Decoupled Geometric Parameterization and its Application in Deep Homography Estimation
- **分类: cs.CV**

- **简介: 该论文属于深度单应估计任务。针对传统四角位置参数化缺乏几何可解释性且需解线性方程的问题，提出基于SKS分解的几何参数化方法，解耦相似与核变换参数，并推导核参数与角度偏移的线性关系，实现直接矩阵计算单应矩阵，性能与传统方法相当。**

- **链接: [http://arxiv.org/pdf/2505.16599v1](http://arxiv.org/pdf/2505.16599v1)**

> **作者:** Yao Huang; Si-Yuan Cao; Yaqing Ding; Hao Yin; Shibin Xie; Shuting Wang; Zhijun Fang; Jiachun Wang; Shen Cai; Junchi Yan; Shuhan Shen
>
> **摘要:** Planar homography, with eight degrees of freedom (DOFs), is fundamental in numerous computer vision tasks. While the positional offsets of four corners are widely adopted (especially in neural network predictions), this parameterization lacks geometric interpretability and typically requires solving a linear system to compute the homography matrix. This paper presents a novel geometric parameterization of homographies, leveraging the similarity-kernel-similarity (SKS) decomposition for projective transformations. Two independent sets of four geometric parameters are decoupled: one for a similarity transformation and the other for the kernel transformation. Additionally, the geometric interpretation linearly relating the four kernel transformation parameters to angular offsets is derived. Our proposed parameterization allows for direct homography estimation through matrix multiplication, eliminating the need for solving a linear system, and achieves performance comparable to the four-corner positional offsets in deep homography estimation.
>
---
#### [new 015] Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings
- **分类: cs.CV; cs.LG**

- **简介: 论文提出TEA方法，针对低查询黑盒场景加速定向对抗攻击。任务为通过扰动使图像被误判为目标类，解决现有方法依赖决策边界几何信息、忽视图像内容导致效率低的问题。TEA利用目标图像边缘信息生成更优初始扰动，减少70%查询，提升攻击效率。**

- **链接: [http://arxiv.org/pdf/2505.16313v1](http://arxiv.org/pdf/2505.16313v1)**

> **作者:** Arjhun Swaminathan; Mete Akgün
>
> **备注:** This paper contains 11 pages, 7 figures and 3 tables. For associated supplementary code, see https://github.com/mdppml/TEA
>
> **摘要:** Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70\% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.
>
---
#### [new 016] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于长视频时间定位任务，旨在解决现有方法处理长视频时计算成本过高的问题。提出DeCafNet，通过轻量级sidekick编码器生成显著图筛选关键片段，结合专家编码器及多尺度优化模块（DeCaf-Grounder），在减少47%计算量的同时达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.16376v1](http://arxiv.org/pdf/2505.16376v1)**

> **作者:** Zijia Lu; A S M Iftekhar; Gaurav Mittal; Tianjian Meng; Xiawei Wang; Cheng Zhao; Rohith Kukkala; Ehsan Elhamifar; Mei Chen
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Long Video Temporal Grounding (LVTG) aims at identifying specific moments within lengthy videos based on user-provided text queries for effective content retrieval. The approach taken by existing methods of dividing video into clips and processing each clip via a full-scale expert encoder is challenging to scale due to prohibitive computational costs of processing a large number of clips in long videos. To address this issue, we introduce DeCafNet, an approach employing ``delegate-and-conquer'' strategy to achieve computation efficiency without sacrificing grounding performance. DeCafNet introduces a sidekick encoder that performs dense feature extraction over all video clips in a resource-efficient manner, while generating a saliency map to identify the most relevant clips for full processing by the expert encoder. To effectively leverage features from sidekick and expert encoders that exist at different temporal resolutions, we introduce DeCaf-Grounder, which unifies and refines them via query-aware temporal aggregation and multi-scale temporal refinement for accurate grounding. Experiments on two LTVG benchmark datasets demonstrate that DeCafNet reduces computation by up to 47\% while still outperforming existing methods, establishing a new state-of-the-art for LTVG in terms of both efficiency and performance. Our code is available at https://github.com/ZijiaLewisLu/CVPR2025-DeCafNet.
>
---
#### [new 017] SCENIR: Visual Semantic Clarity through Unsupervised Scene Graph Retrieval
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像检索任务，旨在解决现有模型依赖低层次视觉特征及监督数据导致的语义理解不足问题。提出SCENIR，一种无监督的图自编码框架，通过场景图强调语义内容，并采用图编辑距离作为评估标准，提升检索性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.15867v1](http://arxiv.org/pdf/2505.15867v1)**

> **作者:** Nikolaos Chaidos; Angeliki Dimitriou; Maria Lymperaiou; Giorgos Stamou
>
> **摘要:** Despite the dominance of convolutional and transformer-based architectures in image-to-image retrieval, these models are prone to biases arising from low-level visual features, such as color. Recognizing the lack of semantic understanding as a key limitation, we propose a novel scene graph-based retrieval framework that emphasizes semantic content over superficial image characteristics. Prior approaches to scene graph retrieval predominantly rely on supervised Graph Neural Networks (GNNs), which require ground truth graph pairs driven from image captions. However, the inconsistency of caption-based supervision stemming from variable text encodings undermine retrieval reliability. To address these, we present SCENIR, a Graph Autoencoder-based unsupervised retrieval framework, which eliminates the dependence on labeled training data. Our model demonstrates superior performance across metrics and runtime efficiency, outperforming existing vision-based, multimodal, and supervised GNN approaches. We further advocate for Graph Edit Distance (GED) as a deterministic and robust ground truth measure for scene graph similarity, replacing the inconsistent caption-based alternatives for the first time in image-to-image retrieval evaluation. Finally, we validate the generalizability of our method by applying it to unannotated datasets via automated scene graph generation, while substantially contributing in advancing state-of-the-art in counterfactual image retrieval.
>
---
#### [new 018] Breaking Complexity Barriers: High-Resolution Image Restoration with Rank Enhanced Linear Attention
- **分类: cs.CV**

- **简介: 该论文针对高分辨率图像修复任务，解决Transformer模型中自注意力机制的二次复杂度与线性注意力性能不足问题。提出Rank Enhanced Linear Attention（RELA），通过深度卷积增强注意力特征，并构建LAformer模型，融合线性注意力、通道注意力及卷积前馈网络，去除低效操作。实验显示其性能与效率优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16157v1](http://arxiv.org/pdf/2505.16157v1)**

> **作者:** Yuang Ai; Huaibo Huang; Tao Wu; Qihang Fan; Ran He
>
> **备注:** 13 pages, 7 figures, 12 tables
>
> **摘要:** Transformer-based models have made remarkable progress in image restoration (IR) tasks. However, the quadratic complexity of self-attention in Transformer hinders its applicability to high-resolution images. Existing methods mitigate this issue with sparse or window-based attention, yet inherently limit global context modeling. Linear attention, a variant of softmax attention, demonstrates promise in global context modeling while maintaining linear complexity, offering a potential solution to the above challenge. Despite its efficiency benefits, vanilla linear attention suffers from a significant performance drop in IR, largely due to the low-rank nature of its attention map. To counter this, we propose Rank Enhanced Linear Attention (RELA), a simple yet effective method that enriches feature representations by integrating a lightweight depthwise convolution. Building upon RELA, we propose an efficient and effective image restoration Transformer, named LAformer. LAformer achieves effective global perception by integrating linear attention and channel attention, while also enhancing local fitting capabilities through a convolutional gated feed-forward network. Notably, LAformer eliminates hardware-inefficient operations such as softmax and window shifting, enabling efficient processing of high-resolution images. Extensive experiments across 7 IR tasks and 21 benchmarks demonstrate that LAformer outperforms SOTA methods and offers significant computational advantages.
>
---
#### [new 019] Consistent World Models via Foresight Diffusion
- **分类: cs.CV**

- **简介: 该论文属于世界模型任务，旨在解决扩散模型预测不一致的问题。针对扩散模型中条件理解与去噪过程纠缠导致预测能力不足的瓶颈，提出ForeDiff框架，通过分离预测流与去噪流，并引入预训练预测器指导生成，提升预测准确性和样本一致性。**

- **链接: [http://arxiv.org/pdf/2505.16474v1](http://arxiv.org/pdf/2505.16474v1)**

> **作者:** Yu Zhang; Xingzhuo Guo; Haoran Xu; Mingsheng Long
>
> **摘要:** Diffusion and flow-based models have enabled significant progress in generation tasks across various modalities and have recently found applications in world modeling. However, unlike typical generation tasks that encourage sample diversity, world models entail different sources of uncertainty and require consistent samples aligned with the ground-truth trajectory, which is a limitation we empirically observe in diffusion models. We argue that a key bottleneck in learning consistent diffusion-based world models lies in the suboptimal predictive ability, which we attribute to the entanglement of condition understanding and target denoising within shared architectures and co-training schemes. To address this, we propose Foresight Diffusion (ForeDiff), a diffusion-based world modeling framework that enhances consistency by decoupling condition understanding from target denoising. ForeDiff incorporates a separate deterministic predictive stream to process conditioning inputs independently of the denoising stream, and further leverages a pretrained predictor to extract informative representations that guide generation. Extensive experiments on robot video prediction and scientific spatiotemporal forecasting show that ForeDiff improves both predictive accuracy and sample consistency over strong baselines, offering a promising direction for diffusion-based world models.
>
---
#### [new 020] An Exploratory Approach Towards Investigating and Explaining Vision Transformer and Transfer Learning for Brain Disease Detection
- **分类: cs.CV**

- **简介: 该论文属于脑部疾病分类任务，旨在解决MRI图像分析复杂及数据不足问题。工作包括：对比Vision Transformer（ViT）与VGG16/19、ResNet50V2、MobileNetV2等迁移学习模型在孟加拉国MRI数据上的性能，并结合GradCAM等XAI方法提升可解释性。结果表明ViT准确率达94.39%，优于传统模型。**

- **链接: [http://arxiv.org/pdf/2505.16039v1](http://arxiv.org/pdf/2505.16039v1)**

> **作者:** Shuvashis Sarker; Shamim Rahim Refat; Faika Fairuj Preotee; Shifat Islam; Tashreef Muhammad; Mohammad Ashraful Hoque
>
> **备注:** Accepted for publication in 2024 27th International Conference on Computer and Information Technology (ICCIT)
>
> **摘要:** The brain is a highly complex organ that manages many important tasks, including movement, memory and thinking. Brain-related conditions, like tumors and degenerative disorders, can be hard to diagnose and treat. Magnetic Resonance Imaging (MRI) serves as a key tool for identifying these conditions, offering high-resolution images of brain structures. Despite this, interpreting MRI scans can be complicated. This study tackles this challenge by conducting a comparative analysis of Vision Transformer (ViT) and Transfer Learning (TL) models such as VGG16, VGG19, Resnet50V2, MobilenetV2 for classifying brain diseases using MRI data from Bangladesh based dataset. ViT, known for their ability to capture global relationships in images, are particularly effective for medical imaging tasks. Transfer learning helps to mitigate data constraints by fine-tuning pre-trained models. Furthermore, Explainable AI (XAI) methods such as GradCAM, GradCAM++, LayerCAM, ScoreCAM, and Faster-ScoreCAM are employed to interpret model predictions. The results demonstrate that ViT surpasses transfer learning models, achieving a classification accuracy of 94.39%. The integration of XAI methods enhances model transparency, offering crucial insights to aid medical professionals in diagnosing brain diseases with greater precision.
>
---
#### [new 021] Learning better representations for crowded pedestrians in offboard LiDAR-camera 3D tracking-by-detection
- **分类: cs.CV**

- **简介: 该论文属于3D多目标跟踪任务，旨在解决拥挤场景下行人感知与自动标注效率低的问题。工作包括：构建多视角LiDAR-相机数据集，开发自动标注系统，并提出密度与关系感知的高分辨率表示方法，提升小目标跟踪性能和标注效率。**

- **链接: [http://arxiv.org/pdf/2505.16029v1](http://arxiv.org/pdf/2505.16029v1)**

> **作者:** Shichao Li; Peiliang Li; Qing Lian; Peng Yun; Xiaozhi Chen
>
> **摘要:** Perceiving pedestrians in highly crowded urban environments is a difficult long-tail problem for learning-based autonomous perception. Speeding up 3D ground truth generation for such challenging scenes is performance-critical yet very challenging. The difficulties include the sparsity of the captured pedestrian point cloud and a lack of suitable benchmarks for a specific system design study. To tackle the challenges, we first collect a new multi-view LiDAR-camera 3D multiple-object-tracking benchmark of highly crowded pedestrians for in-depth analysis. We then build an offboard auto-labeling system that reconstructs pedestrian trajectories from LiDAR point cloud and multi-view images. To improve the generalization power for crowded scenes and the performance for small objects, we propose to learn high-resolution representations that are density-aware and relationship-aware. Extensive experiments validate that our approach significantly improves the 3D pedestrian tracking performance towards higher auto-labeling efficiency. The code will be publicly available at this HTTP URL.
>
---
#### [new 022] Creatively Upscaling Images with Global-Regional Priors
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出无微调图像超分辨率方法C-Upscale，解决高分辨率生成中全局结构一致性与区域细节创造力难以兼顾的问题。通过提取低频成分作为全局结构先验，并结合多模态LLM生成区域提示，利用区域注意力控制减少重复，提升细节创造性。实现4K/8K图像生成，兼具视觉保真度与区域创新。**

- **链接: [http://arxiv.org/pdf/2505.16976v1](http://arxiv.org/pdf/2505.16976v1)**

> **作者:** Yurui Qian; Qi Cai; Yingwei Pan; Ting Yao; Tao Mei
>
> **备注:** International Journal of Computer Vision (IJCV) 2025
>
> **摘要:** Contemporary diffusion models show remarkable capability in text-to-image generation, while still being limited to restricted resolutions (e.g., 1,024 X 1,024). Recent advances enable tuning-free higher-resolution image generation by recycling pre-trained diffusion models and extending them via regional denoising or dilated sampling/convolutions. However, these models struggle to simultaneously preserve global semantic structure and produce creative regional details in higher-resolution images. To address this, we present C-Upscale, a new recipe of tuning-free image upscaling that pivots on global-regional priors derived from given global prompt and estimated regional prompts via Multimodal LLM. Technically, the low-frequency component of low-resolution image is recognized as global structure prior to encourage global semantic consistency in high-resolution generation. Next, we perform regional attention control to screen cross-attention between global prompt and each region during regional denoising, leading to regional attention prior that alleviates object repetition issue. The estimated regional prompts containing rich descriptive details further act as regional semantic prior to fuel the creativity of regional detail generation. Both quantitative and qualitative evaluations demonstrate that our C-Upscale manages to generate ultra-high-resolution images (e.g., 4,096 X 4,096 and 8,192 X 8,192) with higher visual fidelity and more creative regional details.
>
---
#### [new 023] VideoGameQA-Bench: Evaluating Vision-Language Models for Video Game Quality Assurance
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VideoGameQA-Bench基准，评估视觉语言模型在游戏质量检测中的表现，解决现有评估标准无法满足游戏QA领域需求的问题。通过构建覆盖视觉单元测试、故障检测、缺陷报告生成等任务的综合评测体系，推动游戏开发自动化。**

- **链接: [http://arxiv.org/pdf/2505.15952v1](http://arxiv.org/pdf/2505.15952v1)**

> **作者:** Mohammad Reza Taesiri; Abhijay Ghildyal; Saman Zadtootaghaj; Nabajeet Barman; Cor-Paul Bezemer
>
> **备注:** Project website with code and data: https://asgaardlab.github.io/videogameqa-bench/
>
> **摘要:** With video games now generating the highest revenues in the entertainment industry, optimizing game development workflows has become essential for the sector's sustained growth. Recent advancements in Vision-Language Models (VLMs) offer considerable potential to automate and enhance various aspects of game development, particularly Quality Assurance (QA), which remains one of the industry's most labor-intensive processes with limited automation options. To accurately evaluate the performance of VLMs in video game QA tasks and determine their effectiveness in handling real-world scenarios, there is a clear need for standardized benchmarks, as existing benchmarks are insufficient to address the specific requirements of this domain. To bridge this gap, we introduce VideoGameQA-Bench, a comprehensive benchmark that covers a wide array of game QA activities, including visual unit testing, visual regression testing, needle-in-a-haystack tasks, glitch detection, and bug report generation for both images and videos of various games. Code and data are available at: https://asgaardlab.github.io/videogameqa-bench/
>
---
#### [new 024] AnchorFormer: Differentiable Anchor Attention for Efficient Vision Transformer
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉任务（分类/检测/分割），解决视觉Transformer计算复杂度高（O(n²)）和冗余计算问题。提出AnchorFormer，通过锚token学习关键信息，将复杂度降至O(mn)（m<n），利用可微分锚点注意力和马尔可夫过程近似全局自注意力，并验证其在效率与精度上的提升。**

- **链接: [http://arxiv.org/pdf/2505.16463v1](http://arxiv.org/pdf/2505.16463v1)**

> **作者:** Jiquan Shan; Junxiao Wang; Lifeng Zhao; Liang Cai; Hongyuan Zhang; Ioannis Liritzis
>
> **摘要:** Recently, vision transformers (ViTs) have achieved excellent performance on vision tasks by measuring the global self-attention among the image patches. Given $n$ patches, they will have quadratic complexity such as $\mathcal{O}(n^2)$ and the time cost is high when splitting the input image with a small granularity. Meanwhile, the pivotal information is often randomly gathered in a few regions of an input image, some tokens may not be helpful for the downstream tasks. To handle this problem, we introduce an anchor-based efficient vision transformer (AnchorFormer), which employs the anchor tokens to learn the pivotal information and accelerate the inference. Firstly, by estimating the bipartite attention between the anchors and tokens, the complexity will be reduced from $\mathcal{O}(n^2)$ to $\mathcal{O}(mn)$, where $m$ is an anchor number and $m < n$. Notably, by representing the anchors with the neurons in a neural layer, we can differentiable learn these distributions and approximate global self-attention through the Markov process. Moreover, we extend the proposed model to three downstream tasks including classification, detection, and segmentation. Extensive experiments show the effectiveness of our AnchorFormer, e.g., achieving up to a 9.0% higher accuracy or 46.7% FLOPs reduction on ImageNet classification, 81.3% higher mAP on COCO detection under comparable FLOPs, as compared to the current baselines.
>
---
#### [new 025] Seeing through Satellite Images at Street Views
- **分类: cs.CV**

- **简介: 该论文提出Sat2Density++方法，属于卫星到街景合成任务。针对卫星与街景视角差异大、街景特有元素（如天空/光照）缺失的挑战，通过神经辐射场建模街景要素，实现与卫星图像一致的逼真街景全景渲染，在城市场景中验证有效。**

- **链接: [http://arxiv.org/pdf/2505.17001v1](http://arxiv.org/pdf/2505.17001v1)**

> **作者:** Ming Qian; Bin Tan; Qiuyu Wang; Xianwei Zheng; Hanjiang Xiong; Gui-Song Xia; Yujun Shen; Nan Xue
>
> **备注:** Project page: https://qianmingduowan.github.io/sat2density-pp/, journal extension of ICCV 2023 conference paper 'Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs', submitted to TPAMI
>
> **摘要:** This paper studies the task of SatStreet-view synthesis, which aims to render photorealistic street-view panorama images and videos given any satellite image and specified camera positions or trajectories. We formulate to learn neural radiance field from paired images captured from satellite and street viewpoints, which comes to be a challenging learning problem due to the sparse-view natural and the extremely-large viewpoint changes between satellite and street-view images. We tackle the challenges based on a task-specific observation that street-view specific elements, including the sky and illumination effects are only visible in street-view panoramas, and present a novel approach Sat2Density++ to accomplish the goal of photo-realistic street-view panoramas rendering by modeling these street-view specific in neural networks. In the experiments, our method is testified on both urban and suburban scene datasets, demonstrating that Sat2Density++ is capable of rendering photorealistic street-view panoramas that are consistent across multiple views and faithful to the satellite image.
>
---
#### [new 026] An Approach Towards Identifying Bangladeshi Leaf Diseases through Transfer Learning and XAI
- **分类: cs.CV**

- **简介: 该论文提出基于迁移学习和XAI的孟加拉叶病识别方法。任务为分类6种植物的21类叶病，解决农民因专家不足难以有效管理的问题。采用CNN、VGG19、Xception等模型，结合GradCAM等XAI技术，实现高准确率（98.9%和98.66%），提升可解释性，帮助农民决策，促进农业生产力。**

- **链接: [http://arxiv.org/pdf/2505.16033v1](http://arxiv.org/pdf/2505.16033v1)**

> **作者:** Faika Fairuj Preotee; Shuvashis Sarker; Shamim Rahim Refat; Tashreef Muhammad; Shifat Islam
>
> **备注:** Accepted for publication in 2024 27th International Conference on Computer and Information Technology (ICCIT)
>
> **摘要:** Leaf diseases are harmful conditions that affect the health, appearance and productivity of plants, leading to significant plant loss and negatively impacting farmers' livelihoods. These diseases cause visible symptoms such as lesions, color changes, and texture variations, making it difficult for farmers to manage plant health, especially in large or remote farms where expert knowledge is limited. The main motivation of this study is to provide an efficient and accessible solution for identifying plant leaf diseases in Bangladesh, where agriculture plays a critical role in food security. The objective of our research is to classify 21 distinct leaf diseases across six plants using deep learning models, improving disease detection accuracy while reducing the need for expert involvement. Deep Learning (DL) techniques, including CNN and Transfer Learning (TL) models like VGG16, VGG19, MobileNetV2, InceptionV3, ResNet50V2 and Xception are used. VGG19 and Xception achieve the highest accuracies, with 98.90% and 98.66% respectively. Additionally, Explainable AI (XAI) techniques such as GradCAM, GradCAM++, LayerCAM, ScoreCAM and FasterScoreCAM are used to enhance transparency by highlighting the regions of the models focused on during disease classification. This transparency ensures that farmers can understand the model's predictions and take necessary action. This approach not only improves disease management but also supports farmers in making informed decisions, leading to better plant protection and increased agricultural productivity.
>
---
#### [new 027] Unlocking Smarter Device Control: Foresighted Planning with a World Model-Driven Code Execution Approach
- **分类: cs.CV**

- **简介: 该论文提出FPWC框架，针对移动设备自动控制中因环境信息有限导致的决策次优问题，通过构建任务导向的世界模型进行前瞻规划，生成代码执行动作。实验显示其成功率较现有方法提升44.4%。**

- **链接: [http://arxiv.org/pdf/2505.16422v1](http://arxiv.org/pdf/2505.16422v1)**

> **作者:** Xiaoran Yin; Xu Luo; Hao Wu; Lianli Gao; Jingkuan Song
>
> **摘要:** The automatic control of mobile devices is essential for efficiently performing complex tasks that involve multiple sequential steps. However, these tasks pose significant challenges due to the limited environmental information available at each step, primarily through visual observations. As a result, current approaches, which typically rely on reactive policies, focus solely on immediate observations and often lead to suboptimal decision-making. To address this problem, we propose \textbf{Foresighted Planning with World Model-Driven Code Execution (FPWC)},a framework that prioritizes natural language understanding and structured reasoning to enhance the agent's global understanding of the environment by developing a task-oriented, refinable \emph{world model} at the outset of the task. Foresighted actions are subsequently generated through iterative planning within this world model, executed in the form of executable code. Extensive experiments conducted in simulated environments and on real mobile devices demonstrate that our method outperforms previous approaches, particularly achieving a 44.4\% relative improvement in task success rate compared to the state-of-the-art in the simulated environment. Code and demo are provided in the supplementary material.
>
---
#### [new 028] Incorporating Visual Correspondence into Diffusion Model for Virtual Try-On
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对虚拟试衣（VTON）任务，提出通过引入视觉对应和3D感知线索改进扩散模型。为解决衣物细节丢失问题，将衣物纹理分解为语义点，通过局部流匹配到人体，并结合深度/法线图构建3D监督信号，设计点聚焦损失优化扩散过程，提升细节保留效果，在多个数据集上达最优性能。**

- **链接: [http://arxiv.org/pdf/2505.16977v1](http://arxiv.org/pdf/2505.16977v1)**

> **作者:** Siqi Wan; Jingwen Chen; Yingwei Pan; Ting Yao; Tao Mei
>
> **备注:** ICLR 2025. Code is publicly available at: https://github.com/HiDream-ai/SPM-Diff
>
> **摘要:** Diffusion models have shown preliminary success in virtual try-on (VTON) task. The typical dual-branch architecture comprises two UNets for implicit garment deformation and synthesized image generation respectively, and has emerged as the recipe for VTON task. Nevertheless, the problem remains challenging to preserve the shape and every detail of the given garment due to the intrinsic stochasticity of diffusion model. To alleviate this issue, we novelly propose to explicitly capitalize on visual correspondence as the prior to tame diffusion process instead of simply feeding the whole garment into UNet as the appearance reference. Specifically, we interpret the fine-grained appearance and texture details as a set of structured semantic points, and match the semantic points rooted in garment to the ones over target person through local flow warping. Such 2D points are then augmented into 3D-aware cues with depth/normal map of target person. The correspondence mimics the way of putting clothing on human body and the 3D-aware cues act as semantic point matching to supervise diffusion model training. A point-focused diffusion loss is further devised to fully take the advantage of semantic point matching. Extensive experiments demonstrate strong garment detail preservation of our approach, evidenced by state-of-the-art VTON performances on both VITON-HD and DressCode datasets. Code is publicly available at: https://github.com/HiDream-ai/SPM-Diff.
>
---
#### [new 029] ARPO:End-to-End Policy Optimization for GUI Agents with Experience Replay
- **分类: cs.CV**

- **简介: 该论文属于基于强化学习的GUI代理训练任务，旨在解决LLM在复杂GUI环境中因稀疏奖励、延迟反馈和高成本导致的长序列动作优化难题。提出ARPO方法，通过经验回放重用成功经验，并设计任务筛选策略稳定训练，实验显示其在OSWorld基准中建立新性能基线。**

- **链接: [http://arxiv.org/pdf/2505.16282v1](http://arxiv.org/pdf/2505.16282v1)**

> **作者:** Fanbin Lu; Zhisheng Zhong; Shu Liu; Chi-Wing Fu; Jiaya Jia
>
> **摘要:** Training large language models (LLMs) as interactive agents for controlling graphical user interfaces (GUIs) presents a unique challenge to optimize long-horizon action sequences with multimodal feedback from complex environments. While recent works have advanced multi-turn reinforcement learning (RL) for reasoning and tool-using capabilities in LLMs, their application to GUI-based agents remains relatively underexplored due to the difficulty of sparse rewards, delayed feedback, and high rollout costs. In this paper, we investigate end-to-end policy optimization for vision-language-based GUI agents with the aim of improving performance on complex, long-horizon computer tasks. We propose Agentic Replay Policy Optimization (ARPO), an end-to-end RL approach that augments Group Relative Policy Optimization (GRPO) with a replay buffer to reuse the successful experience across training iterations. To further stabilize the training process, we propose a task selection strategy that filters tasks based on baseline agent performance, allowing the agent to focus on learning from informative interactions. Additionally, we compare ARPO with offline preference optimization approaches, highlighting the advantages of policy-based methods in GUI environments. Experiments on the OSWorld benchmark demonstrate that ARPO achieves competitive results, establishing a new performance baseline for LLM-based GUI agents trained via reinforcement learning. Our findings underscore the effectiveness of reinforcement learning for training multi-turn, vision-language GUI agents capable of managing complex real-world UI interactions. Codes and models:https://github.com/dvlab-research/ARPO.git.
>
---
#### [new 030] Super-Resolution with Structured Motion
- **分类: cs.CV; I.4.1; I.4.3**

- **简介: 该论文属超分辨率任务，旨在突破传统方法分辨率提升有限及运动模糊干扰的限制。提出利用结构化运动、稀疏先验和凸优化技术，通过伪随机运动及单幅低分辨率图像实现大幅超分辨率重建，并证明运动模糊可助力提升，实验验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.15961v1](http://arxiv.org/pdf/2505.15961v1)**

> **作者:** Gabby Litterio; Juan-David Lizarazo-Ferro; Pedro Felzenszwalb; Rashid Zia
>
> **摘要:** We consider the limits of super-resolution using imaging constraints. Due to various theoretical and practical limitations, reconstruction-based methods have been largely restricted to small increases in resolution. In addition, motion-blur is usually seen as a nuisance that impedes super-resolution. We show that by using high-precision motion information, sparse image priors, and convex optimization, it is possible to increase resolution by large factors. A key operation in super-resolution is deconvolution with a box. In general, convolution with a box is not invertible. However, we obtain perfect reconstructions of sparse signals using convex optimization. We also show that motion blur can be helpful for super-resolution. We demonstrate that using pseudo-random motion it is possible to reconstruct a high-resolution target using a single low-resolution image. We present numerical experiments with simulated data and results with real data captured by a camera mounted on a computer controlled stage.
>
---
#### [new 031] Erased or Dormant? Rethinking Concept Erasure Through Reversibility
- **分类: cs.CV**

- **简介: 该论文属于扩散模型概念擦除任务，旨在探究现有方法是否真正消除目标概念的生成能力而非仅表面抑制。研究通过轻量微调测试两种擦除方法（Unified Concept Editing和Erased Stable Diffusion）的可逆性，发现擦除概念经简单调整后可高保真复现，指出需更深层的表征级干预及严格评估标准。**

- **链接: [http://arxiv.org/pdf/2505.16174v1](http://arxiv.org/pdf/2505.16174v1)**

> **作者:** Ping Liu; Chi Zhang
>
> **备注:** Dr. Chi Zhang is the corresponding author
>
> **摘要:** To what extent does concept erasure eliminate generative capacity in diffusion models? While prior evaluations have primarily focused on measuring concept suppression under specific textual prompts, we explore a complementary and fundamental question: do current concept erasure techniques genuinely remove the ability to generate targeted concepts, or do they merely achieve superficial, prompt-specific suppression? We systematically evaluate the robustness and reversibility of two representative concept erasure methods, Unified Concept Editing and Erased Stable Diffusion, by probing their ability to eliminate targeted generative behaviors in text-to-image models. These methods attempt to suppress undesired semantic concepts by modifying internal model parameters, either through targeted attention edits or model-level fine-tuning strategies. To rigorously assess whether these techniques truly erase generative capacity, we propose an instance-level evaluation strategy that employs lightweight fine-tuning to explicitly test the reactivation potential of erased concepts. Through quantitative metrics and qualitative analyses, we show that erased concepts often reemerge with substantial visual fidelity after minimal adaptation, indicating that current methods suppress latent generative representations without fully eliminating them. Our findings reveal critical limitations in existing concept erasure approaches and highlight the need for deeper, representation-level interventions and more rigorous evaluation standards to ensure genuine, irreversible removal of concepts from generative models.
>
---
#### [new 032] CrossLMM: Decoupling Long Video Sequences from LMMs via Dual Cross-Attention Mechanisms
- **分类: cs.CV**

- **简介: 该论文属于多模态处理任务，旨在解决长视频序列在LMMs中因token过多导致的计算成本激增问题。提出CrossLMM方法，通过视觉池化大幅减少token数量，并采用双交叉注意力机制：视觉间交叉注意力优化token利用，文视交叉注意力增强文本对视觉的理解，在降低计算量的同时保持性能。**

- **链接: [http://arxiv.org/pdf/2505.17020v1](http://arxiv.org/pdf/2505.17020v1)**

> **作者:** Shilin Yan; Jiaming Han; Joey Tsai; Hongwei Xue; Rongyao Fang; Lingyi Hong; Ziyu Guo; Ray Zhang
>
> **备注:** Project page: https://github.com/shilinyan99/CrossLMM
>
> **摘要:** The advent of Large Multimodal Models (LMMs) has significantly enhanced Large Language Models (LLMs) to process and interpret diverse data modalities (e.g., image and video). However, as input complexity increases, particularly with long video sequences, the number of required tokens has grown significantly, leading to quadratically computational costs. This has made the efficient compression of video tokens in LMMs, while maintaining performance integrity, a pressing research challenge. In this paper, we introduce CrossLMM, decoupling long video sequences from LMMs via a dual cross-attention mechanism, which substantially reduces visual token quantity with minimal performance degradation. Specifically, we first implement a significant token reduction from pretrained visual encoders through a pooling methodology. Then, within LLM layers, we employ a visual-to-visual cross-attention mechanism, wherein the pooled visual tokens function as queries against the original visual token set. This module enables more efficient token utilization while retaining fine-grained informational fidelity. In addition, we introduce a text-to-visual cross-attention mechanism, for which the text tokens are enhanced through interaction with the original visual tokens, enriching the visual comprehension of the text tokens. Comprehensive empirical evaluation demonstrates that our approach achieves comparable or superior performance across diverse video-based LMM benchmarks, despite utilizing substantially fewer computational resources.
>
---
#### [new 033] Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频虚假信息检测任务，针对数据不足、模型过拟合及推理能力弱的问题，提出FakeVV数据集（含10万+视频-文本对）与Fact-R1框架。Fact-R1通过三阶段强化学习（长链推理调优、偏好对齐、群体优化）实现多模态深度推理与可解释验证，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.16836v1](http://arxiv.org/pdf/2505.16836v1)**

> **作者:** Fanrui Zhang; Dian Li; Qiang Zhang; Chenjun; sinbadliu; Junxiong Lin; Jiahong Yan; Jiawei Liu; Zheng-Jun Zha
>
> **备注:** 28 pages, 27 figures
>
> **摘要:** The rapid spread of multimodal misinformation on social media has raised growing concerns, while research on video misinformation detection remains limited due to the lack of large-scale, diverse datasets. Existing methods often overfit to rigid templates and lack deep reasoning over deceptive content. To address these challenges, we introduce FakeVV, a large-scale benchmark comprising over 100,000 video-text pairs with fine-grained, interpretable annotations. In addition, we further propose Fact-R1, a novel framework that integrates deep reasoning with collaborative rule-based reinforcement learning. Fact-R1 is trained through a three-stage process: (1) misinformation long-Chain-of-Thought (CoT) instruction tuning, (2) preference alignment via Direct Preference Optimization (DPO), and (3) Group Relative Policy Optimization (GRPO) using a novel verifiable reward function. This enables Fact-R1 to exhibit emergent reasoning behaviors comparable to those observed in advanced text-based reinforcement learning systems, but in the more complex multimodal misinformation setting. Our work establishes a new paradigm for misinformation detection, bridging large-scale video understanding, reasoning-guided alignment, and interpretable verification.
>
---
#### [new 034] Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对脑肿瘤分割中缺失MRI模态及增量学习问题，提出ReHyDIL框架。通过域增量学习避免模型遗忘，结合超图网络捕捉患者间关联，并用Tversky-Aware对比损失平衡模态信息，提升动态模态下的分割精度。**

- **链接: [http://arxiv.org/pdf/2505.16809v1](http://arxiv.org/pdf/2505.16809v1)**

> **作者:** Junze Wang; Lei Fan; Weipeng Jing; Donglin Di; Yang Song; Sidong Liu; Cong Cong
>
> **摘要:** Existing methods for multimodal MRI segmentation with missing modalities typically assume that all MRI modalities are available during training. However, in clinical practice, some modalities may be missing due to the sequential nature of MRI acquisition, leading to performance degradation. Furthermore, retraining models to accommodate newly available modalities can be inefficient and may cause overfitting, potentially compromising previously learned knowledge. To address these challenges, we propose Replay-based Hypergraph Domain Incremental Learning (ReHyDIL) for brain tumor segmentation with missing modalities. ReHyDIL leverages Domain Incremental Learning (DIL) to enable the segmentation model to learn from newly acquired MRI modalities without forgetting previously learned information. To enhance segmentation performance across diverse patient scenarios, we introduce the Cross-Patient Hypergraph Segmentation Network (CHSNet), which utilizes hypergraphs to capture high-order associations between patients. Additionally, we incorporate Tversky-Aware Contrastive (TAC) loss to effectively mitigate information imbalance both across and within different modalities. Extensive experiments on the BraTS2019 dataset demonstrate that ReHyDIL outperforms state-of-the-art methods, achieving an improvement of over 2\% in the Dice Similarity Coefficient across various tumor regions. Our code is available at ReHyDIL.
>
---
#### [new 035] Deep mineralogical segmentation of thin section images based on QEMSCAN maps
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出基于U-Net的CNN模型，通过训练碳酸盐岩薄片图像与QEMSCAN矿物图，实现低成本自动矿物分割。旨在替代高成本、低效的QEMSCAN技术，解决人工分析主观性问题。工作包括：利用多视场图像配准处理分辨率差异，训练模型识别 Calcite、Dolomite等矿物及孔隙，验证其对已知/未知岩相的泛化能力（R²>0.97/0.88）。**

- **链接: [http://arxiv.org/pdf/2505.17008v1](http://arxiv.org/pdf/2505.17008v1)**

> **作者:** Jean Pablo Vieira de Mello; Matheus Augusto Alves Cuglieri; Leandro P. de Figueiredo; Fernando Bordignon; Marcelo Ramalho Albuquerque; Rodrigo Surmas; Bruno Cavalcanti de Paula
>
> **摘要:** Interpreting the mineralogical aspects of rock thin sections is an important task for oil and gas reservoirs evaluation. However, human analysis tend to be subjective and laborious. Technologies like QEMSCAN(R) are designed to automate the mineralogical mapping process, but also suffer from limitations like high monetary costs and time-consuming analysis. This work proposes a Convolutional Neural Network model for automatic mineralogical segmentation of thin section images of carbonate rocks. The model is able to mimic the QEMSCAN mapping itself in a low-cost, generalized and efficient manner. For this, the U-Net semantic segmentation architecture is trained on plane and cross polarized thin section images using the corresponding QEMSCAN maps as target, which is an approach not widely explored. The model was instructed to differentiate occurrences of Calcite, Dolomite, Mg-Clay Minerals, Quartz, Pores and the remaining mineral phases as an unique class named "Others", while it was validated on rock facies both seen and unseen during training, in order to address its generalization capability. Since the images and maps are provided in different resolutions, image registration was applied to align then spatially. The study reveals that the quality of the segmentation is very much dependent on these resolution differences and on the variety of learnable rock textures. However, it shows promising results, especially with regard to the proper delineation of minerals boundaries on solid textures and precise estimation of the minerals distributions, describing a nearly linear relationship between expected and predicted distributions, with coefficient of determination (R^2) superior to 0.97 for seen facies and 0.88 for unseen.
>
---
#### [new 036] KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models
- **分类: cs.CV**

- **简介: 该论文提出KRIS-Bench基准，评估图像编辑模型的知识推理能力。针对现有模型在知识型任务中的不足，基于教育理论分类三类知识（事实、概念、程序），设计22项任务及1267个标注样本，并提出知识合理性评估协议。实验显示模型存在显著差距，强调需知识驱动的基准推动智能图像编辑发展。**

- **链接: [http://arxiv.org/pdf/2505.16707v1](http://arxiv.org/pdf/2505.16707v1)**

> **作者:** Yongliang Wu; Zonghui Li; Xinting Hu; Xinyu Ye; Xianfang Zeng; Gang Yu; Wenbo Zhu; Bernt Schiele; Ming-Hsuan Yang; Xu Yang
>
> **备注:** 39 pages, 36 figures
>
> **摘要:** Recent advances in multi-modal generative models have enabled significant progress in instruction-based image editing. However, while these models produce visually plausible outputs, their capacity for knowledge-based reasoning editing tasks remains under-explored. In this paper, we introduce KRIS-Bench (Knowledge-based Reasoning in Image-editing Systems Benchmark), a diagnostic benchmark designed to assess models through a cognitively informed lens. Drawing from educational theory, KRIS-Bench categorizes editing tasks across three foundational knowledge types: Factual, Conceptual, and Procedural. Based on this taxonomy, we design 22 representative tasks spanning 7 reasoning dimensions and release 1,267 high-quality annotated editing instances. To support fine-grained evaluation, we propose a comprehensive protocol that incorporates a novel Knowledge Plausibility metric, enhanced by knowledge hints and calibrated through human studies. Empirical results on 10 state-of-the-art models reveal significant gaps in reasoning performance, highlighting the need for knowledge-centric benchmarks to advance the development of intelligent image editing systems.
>
---
#### [new 037] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（LVLM）中RoPE技术导致跨模态位置偏差的问题，提出Circle-RoPE方案。通过将图像位置编码映射至与文本线性轨迹正交的圆形路径，形成锥形结构，使文本token与所有图像token保持等距，减少位置偏置同时保留图像空间信息，并采用分层策略优化模型性能。任务为改进LVLM的位置编码，解决RoPE引起的错误跨模态对齐问题。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16416v1](http://arxiv.org/pdf/2505.16416v1)**

> **作者:** Chengcheng Wang; Jianyuan Guo; Hongguang Li; Yuchuan Tian; Ying Nie; Chang Xu; Kai Han
>
> **摘要:** Rotary Position Embedding (RoPE) is a widely adopted technique for encoding relative positional information in large language models (LLMs). However, when extended to large vision-language models (LVLMs), its variants introduce unintended cross-modal positional biases. Specifically, they enforce relative positional dependencies between text token indices and image tokens, causing spurious alignments. This issue arises because image tokens representing the same content but located at different spatial positions are assigned distinct positional biases, leading to inconsistent cross-modal associations. To address this, we propose Per-Token Distance (PTD) - a simple yet effective metric for quantifying the independence of positional encodings across modalities. Informed by this analysis, we introduce Circle-RoPE, a novel encoding scheme that maps image token indices onto a circular trajectory orthogonal to the linear path of text token indices, forming a cone-like structure. This configuration ensures that each text token maintains an equal distance to all image tokens, reducing artificial cross-modal biases while preserving intra-image spatial information. To further enhance performance, we propose a staggered layer strategy that applies different RoPE variants across layers. This design leverages the complementary strengths of each RoPE variant, thereby enhancing the model's overall performance. Our experimental results demonstrate that our method effectively preserves spatial information from images while reducing relative positional bias, offering a more robust and flexible positional encoding framework for LVLMs. The code is available at [https://github.com/lose4578/CircleRoPE](https://github.com/lose4578/CircleRoPE).
>
---
#### [new 038] Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对多模态场景下的Out-of-Distribution（OOD）检测与分割任务，提出Feature Mixing方法，通过合成异常样本解决未知数据缺乏监督导致的模型过自信问题。同时构建CARLA-OOD数据集，实验显示其在多个基准上达SOTA性能，速度提升10-370倍。**

- **链接: [http://arxiv.org/pdf/2505.16985v1](http://arxiv.org/pdf/2505.16985v1)**

> **作者:** Moru Liu; Hao Dong; Jessica Kelly; Olga Fink; Mario Trapp
>
> **摘要:** Out-of-distribution (OOD) detection and segmentation are crucial for deploying machine learning models in safety-critical applications such as autonomous driving and robot-assisted surgery. While prior research has primarily focused on unimodal image data, real-world applications are inherently multimodal, requiring the integration of multiple modalities for improved OOD detection. A key challenge is the lack of supervision signals from unknown data, leading to overconfident predictions on OOD samples. To address this challenge, we propose Feature Mixing, an extremely simple and fast method for multimodal outlier synthesis with theoretical support, which can be further optimized to help the model better distinguish between in-distribution (ID) and OOD data. Feature Mixing is modality-agnostic and applicable to various modality combinations. Additionally, we introduce CARLA-OOD, a novel multimodal dataset for OOD segmentation, featuring synthetic OOD objects across diverse scenes and weather conditions. Extensive experiments on SemanticKITTI, nuScenes, CARLA-OOD datasets, and the MultiOOD benchmark demonstrate that Feature Mixing achieves state-of-the-art performance with a $10 \times$ to $370 \times$ speedup. Our source code and dataset will be available at https://github.com/mona4399/FeatureMixing.
>
---
#### [new 039] Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决用户需专业词汇编写提示的问题。提出通过强化学习框架，利用大视觉语言模型（LVLM）自动生成优化提示并自我评估图像质量，减少对人工标注数据和预训练模型偏见的依赖。**

- **链接: [http://arxiv.org/pdf/2505.16763v1](http://arxiv.org/pdf/2505.16763v1)**

> **作者:** Hongji Yang; Yucheng Zhou; Wencheng Han; Jianbing Shen
>
> **摘要:** Text-to-image models are powerful for producing high-quality images based on given text prompts, but crafting these prompts often requires specialized vocabulary. To address this, existing methods train rewriting models with supervision from large amounts of manually annotated data and trained aesthetic assessment models. To alleviate the dependence on data scale for model training and the biases introduced by trained models, we propose a novel prompt optimization framework, designed to rephrase a simple user prompt into a sophisticated prompt to a text-to-image model. Specifically, we employ the large vision language models (LVLMs) as the solver to rewrite the user prompt, and concurrently, employ LVLMs as a reward model to score the aesthetics and alignment of the images generated by the optimized prompt. Instead of laborious human feedback, we exploit the prior knowledge of the LVLM to provide rewards, i.e., AI feedback. Simultaneously, the solver and the reward model are unified into one model and iterated in reinforcement learning to achieve self-improvement by giving a solution and judging itself. Results on two popular datasets demonstrate that our method outperforms other strong competitors.
>
---
#### [new 040] Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation
- **分类: cs.CV; cs.LG; 68T45 (Primary) 68T10, 68T07 (Secondary); F.1.2; F.1.4**

- **简介: 该论文属于合成到现实的领域适应任务，旨在解决合成数据训练的分割模型在真实场景表现差的问题。提出基于扩散模型的语义一致风格迁移方法CACTI及改进版CACTIF，通过类自适应归一化和注意力筛选保留语义结构，减少伪影。实验显示其生成图像质量更高，有效缩小领域差距。**

- **链接: [http://arxiv.org/pdf/2505.16360v1](http://arxiv.org/pdf/2505.16360v1)**

> **作者:** Estelle Chigot; Dennis G. Wilson; Meriem Ghrib; Thomas Oberlin
>
> **备注:** Under review
>
> **摘要:** Semantic segmentation models trained on synthetic data often perform poorly on real-world images due to domain gaps, particularly in adverse conditions where labeled data is scarce. Yet, recent foundation models enable to generate realistic images without any training. This paper proposes to leverage such diffusion models to improve the performance of vision models when learned on synthetic data. We introduce two novel techniques for semantically consistent style transfer using diffusion models: Class-wise Adaptive Instance Normalization and Cross-Attention (CACTI) and its extension with selective attention Filtering (CACTIF). CACTI applies statistical normalization selectively based on semantic classes, while CACTIF further filters cross-attention maps based on feature similarity, preventing artifacts in regions with weak cross-attention correspondences. Our methods transfer style characteristics while preserving semantic boundaries and structural coherence, unlike approaches that apply global transformations or generate content without constraints. Experiments using GTA5 as source and Cityscapes/ACDC as target domains show that our approach produces higher quality images with lower FID scores and better content preservation. Our work demonstrates that class-aware diffusion-based style transfer effectively bridges the synthetic-to-real domain gap even with minimal target domain data, advancing robust perception systems for challenging real-world applications. The source code is available at: https://github.com/echigot/cactif.
>
---
#### [new 041] VLM-R$^3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态推理任务，解决视觉语言模型在复杂任务中动态聚焦与迭代访问视觉区域以提升推理精度的问题。提出VLM-R³框架，通过区域识别、推理及优化，利用区域条件强化策略（R-GRPO）训练模型选择关键区域、调整视觉变换并整合到思维链中，并构建VLIR数据集提供监督，显著提升零/少样本场景下的推理性能。**

- **链接: [http://arxiv.org/pdf/2505.16192v1](http://arxiv.org/pdf/2505.16192v1)**

> **作者:** Chaoya Jiang; Yongrui Heng; Wei Ye; Han Yang; Haiyang Xu; Ming Yan; Ji Zhang; Fei Huang; Shikun Zhang
>
> **摘要:** Recently, reasoning-based MLLMs have achieved a degree of success in generating long-form textual reasoning chains. However, they still struggle with complex tasks that necessitate dynamic and iterative focusing on and revisiting of visual regions to achieve precise grounding of textual reasoning in visual evidence. We introduce \textbf{VLM-R$^3$} (\textbf{V}isual \textbf{L}anguage \textbf{M}odel with \textbf{R}egion \textbf{R}ecognition and \textbf{R}easoning), a framework that equips an MLLM with the ability to (i) decide \emph{when} additional visual evidence is needed, (ii) determine \emph{where} to ground within the image, and (iii) seamlessly weave the relevant sub-image content back into an interleaved chain-of-thought. The core of our method is \textbf{Region-Conditioned Reinforcement Policy Optimization (R-GRPO)}, a training paradigm that rewards the model for selecting informative regions, formulating appropriate transformations (e.g.\ crop, zoom), and integrating the resulting visual context into subsequent reasoning steps. To bootstrap this policy, we compile a modest but carefully curated Visuo-Lingual Interleaved Rationale (VLIR) corpus that provides step-level supervision on region selection and textual justification. Extensive experiments on MathVista, ScienceQA, and other benchmarks show that VLM-R$^3$ sets a new state of the art in zero-shot and few-shot settings, with the largest gains appearing on questions demanding subtle spatial reasoning or fine-grained visual cue extraction.
>
---
#### [new 042] LaViDa: A Large Diffusion Language Model for Multimodal Understanding
- **分类: cs.CV**

- **简介: 该论文提出LaViDa，基于扩散模型的多模态模型，解决自回归VLM（如LLaVA）在推理速度与可控生成上的局限。通过整合视觉编码器、联合微调及创新技术（互补掩码、前缀KV缓存、时间步偏移），其在多模态任务（如COCO标注、约束诗歌生成）中实现性能与效率提升，如CIDEr指标超LLaVA-Next 4.1分且提速1.92倍。**

- **链接: [http://arxiv.org/pdf/2505.16839v1](http://arxiv.org/pdf/2505.16839v1)**

> **作者:** Shufan Li; Konstantinos Kallidromitis; Hritik Bansal; Akash Gokul; Yusuke Kato; Kazuki Kozuka; Jason Kuen; Zhe Lin; Kai-Wei Chang; Aditya Grover
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Modern Vision-Language Models (VLMs) can solve a wide range of tasks requiring visual reasoning. In real-world scenarios, desirable properties for VLMs include fast inference and controllable generation (e.g., constraining outputs to adhere to a desired format). However, existing autoregressive (AR) VLMs like LLaVA struggle in these aspects. Discrete diffusion models (DMs) offer a promising alternative, enabling parallel decoding for faster inference and bidirectional context for controllable generation through text-infilling. While effective in language-only settings, DMs' potential for multimodal tasks is underexplored. We introduce LaViDa, a family of VLMs built on DMs. We build LaViDa by equipping DMs with a vision encoder and jointly fine-tune the combined parts for multimodal instruction following. To address challenges encountered, LaViDa incorporates novel techniques such as complementary masking for effective training, prefix KV cache for efficient inference, and timestep shifting for high-quality sampling. Experiments show that LaViDa achieves competitive or superior performance to AR VLMs on multi-modal benchmarks such as MMMU, while offering unique advantages of DMs, including flexible speed-quality tradeoff, controllability, and bidirectional reasoning. On COCO captioning, LaViDa surpasses Open-LLaVa-Next-8B by +4.1 CIDEr with 1.92x speedup. On bidirectional tasks, it achieves +59% improvement on Constrained Poem Completion. These results demonstrate LaViDa as a strong alternative to AR VLMs. Code and models will be released in the camera-ready version.
>
---
#### [new 043] When VLMs Meet Image Classification: Test Sets Renovation via Missing Label Identification
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出REVEAL框架，结合视觉语言模型（如LLaVA）与标注方法（如Cleanlab），解决图像分类测试集中的噪声标签和缺失标签问题。通过多模型预测聚合与共识过滤，改进数据集标签质量，经验证提升6个基准测试集准确性，助力公平模型评估。**

- **链接: [http://arxiv.org/pdf/2505.16149v1](http://arxiv.org/pdf/2505.16149v1)**

> **作者:** Zirui Pang; Haosheng Tan; Yuhan Pu; Zhijie Deng; Zhouan Shen; Keyu Hu; Jiaheng Wei
>
> **摘要:** Image classification benchmark datasets such as CIFAR, MNIST, and ImageNet serve as critical tools for model evaluation. However, despite the cleaning efforts, these datasets still suffer from pervasive noisy labels and often contain missing labels due to the co-existing image pattern where multiple classes appear in an image sample. This results in misleading model comparisons and unfair evaluations. Existing label cleaning methods focus primarily on noisy labels, but the issue of missing labels remains largely overlooked. Motivated by these challenges, we present a comprehensive framework named REVEAL, integrating state-of-the-art pre-trained vision-language models (e.g., LLaVA, BLIP, Janus, Qwen) with advanced machine/human label curation methods (e.g., Docta, Cleanlab, MTurk), to systematically address both noisy labels and missing label detection in widely-used image classification test sets. REVEAL detects potential noisy labels and omissions, aggregates predictions from various methods, and refines label accuracy through confidence-informed predictions and consensus-based filtering. Additionally, we provide a thorough analysis of state-of-the-art vision-language models and pre-trained image classifiers, highlighting their strengths and limitations within the context of dataset renovation by revealing 10 observations. Our method effectively reveals missing labels from public datasets and provides soft-labeled results with likelihoods. Through human verifications, REVEAL significantly improves the quality of 6 benchmark test sets, highly aligning to human judgments and enabling more accurate and meaningful comparisons in image classification.
>
---
#### [new 044] Four Eyes Are Better Than Two: Harnessing the Collaborative Potential of Large Models via Differentiated Thinking and Complementary Ensembles
- **分类: cs.CV**

- **简介: 该论文针对Ego4D EgoSchema挑战的视频理解任务，提出通过多模态大模型协作提升性能。旨在解决如何利用大模型的泛化能力优化视频分析。工作包括探索多样化提示策略与流程范式引导模型注意力，并设计互补集成方案，实现单模型超越此前SOTA，结合阶段性结果融合进一步提升效果。**

- **链接: [http://arxiv.org/pdf/2505.16784v1](http://arxiv.org/pdf/2505.16784v1)**

> **作者:** Jun Xie; Xiongjun Guan; Yingjian Zhu; Zhaoran Zhao; Xinming Wang; Feng Chen; Zhepeng Wang
>
> **摘要:** In this paper, we present the runner-up solution for the Ego4D EgoSchema Challenge at CVPR 2025 (Confirmed on May 20, 2025). Inspired by the success of large models, we evaluate and leverage leading accessible multimodal large models and adapt them to video understanding tasks via few-shot learning and model ensemble strategies. Specifically, diversified prompt styles and process paradigms are systematically explored and evaluated to effectively guide the attention of large models, fully unleashing their powerful generalization and adaptability abilities. Experimental results demonstrate that, with our carefully designed approach, directly utilizing an individual multimodal model already outperforms the previous state-of-the-art (SOTA) method which includes several additional processes. Besides, an additional stage is further introduced that facilitates the cooperation and ensemble of periodic results, which achieves impressive performance improvements. We hope this work serves as a valuable reference for the practical application of large models and inspires future research in the field.
>
---
#### [new 045] SuperPure: Efficient Purification of Localized and Distributed Adversarial Patches via Super-Resolution GAN Models
- **分类: cs.CV; cs.CR; eess.IV**

- **简介: 该论文提出SuperPure，针对对抗补丁攻击防御任务，解决现有方法对分布式攻击脆弱且效率低的问题。通过GAN超分辨率驱动的像素级遮罩方案，提升对局部/分布式补丁的鲁棒性（如分布式防御率58% vs 0%），并降低98%延迟，兼顾清洁准确率。**

- **链接: [http://arxiv.org/pdf/2505.16318v1](http://arxiv.org/pdf/2505.16318v1)**

> **作者:** Hossein Khalili; Seongbin Park; Venkat Bollapragada; Nader Sehatbakhsh
>
> **摘要:** As vision-based machine learning models are increasingly integrated into autonomous and cyber-physical systems, concerns about (physical) adversarial patch attacks are growing. While state-of-the-art defenses can achieve certified robustness with minimal impact on utility against highly-concentrated localized patch attacks, they fall short in two important areas: (i) State-of-the-art methods are vulnerable to low-noise distributed patches where perturbations are subtly dispersed to evade detection or masking, as shown recently by the DorPatch attack; (ii) Achieving high robustness with state-of-the-art methods is extremely time and resource-consuming, rendering them impractical for latency-sensitive applications in many cyber-physical systems. To address both robustness and latency issues, this paper proposes a new defense strategy for adversarial patch attacks called SuperPure. The key novelty is developing a pixel-wise masking scheme that is robust against both distributed and localized patches. The masking involves leveraging a GAN-based super-resolution scheme to gradually purify the image from adversarial patches. Our extensive evaluations using ImageNet and two standard classifiers, ResNet and EfficientNet, show that SuperPure advances the state-of-the-art in three major directions: (i) it improves the robustness against conventional localized patches by more than 20%, on average, while also improving top-1 clean accuracy by almost 10%; (ii) It achieves 58% robustness against distributed patch attacks (as opposed to 0% in state-of-the-art method, PatchCleanser); (iii) It decreases the defense end-to-end latency by over 98% compared to PatchCleanser. Our further analysis shows that SuperPure is robust against white-box attacks and different patch sizes. Our code is open-source.
>
---
#### [new 046] Robust Vision-Based Runway Detection through Conformal Prediction and Conformal mAP
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉跑道检测任务，旨在提升航空着陆系统中检测可靠性。通过微调YOLO模型并结合conformal预测，量化检测不确定性以满足用户风险阈值，提出C-mAP指标评估检测性能与置信度的结合，增强系统安全性并推动ML在航空领域的认证。**

- **链接: [http://arxiv.org/pdf/2505.16740v1](http://arxiv.org/pdf/2505.16740v1)**

> **作者:** Alya Zouzou; Léo andéol; Mélanie Ducoffe; Ryma Boumazouza
>
> **摘要:** We explore the use of conformal prediction to provide statistical uncertainty guarantees for runway detection in vision-based landing systems (VLS). Using fine-tuned YOLOv5 and YOLOv6 models on aerial imagery, we apply conformal prediction to quantify localization reliability under user-defined risk levels. We also introduce Conformal mean Average Precision (C-mAP), a novel metric aligning object detection performance with conformal guarantees. Our results show that conformal prediction can improve the reliability of runway detection by quantifying uncertainty in a statistically sound way, increasing safety on-board and paving the way for certification of ML system in the aerospace domain.
>
---
#### [new 047] MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MedFrameQA，首个评估多图像医疗视觉问答（VQA）的基准，解决临床诊断需对比多影像而现有方法侧重单图分析的问题。通过自动化提取视频帧构建逻辑连贯的VQA数据（2851题，覆盖43器官），并测试多种模型发现其推理能力不足，推动多图临床推理研究。**

- **链接: [http://arxiv.org/pdf/2505.16964v1](http://arxiv.org/pdf/2505.16964v1)**

> **作者:** Suhao Yu; Haojin Wang; Juncheng Wu; Cihang Xie; Yuyin Zhou
>
> **备注:** 9 pages, 4 Figures Benchmark data: https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA
>
> **摘要:** Existing medical VQA benchmarks mostly focus on single-image analysis, yet clinicians almost always compare a series of images before reaching a diagnosis. To better approximate this workflow, we introduce MedFrameQA -- the first benchmark that explicitly evaluates multi-image reasoning in medical VQA. To build MedFrameQA both at scale and in high-quality, we develop 1) an automated pipeline that extracts temporally coherent frames from medical videos and constructs VQA items whose content evolves logically across images, and 2) a multiple-stage filtering strategy, including model-based and manual review, to preserve data clarity, difficulty, and medical relevance. The resulting dataset comprises 2,851 VQA pairs (gathered from 9,237 high-quality frames in 3,420 videos), covering nine human body systems and 43 organs; every question is accompanied by two to five images. We comprehensively benchmark ten advanced Multimodal LLMs -- both proprietary and open source, with and without explicit reasoning modules -- on MedFrameQA. The evaluation challengingly reveals that all models perform poorly, with most accuracies below 50%, and accuracy fluctuates as the number of images per question increases. Error analysis further shows that models frequently ignore salient findings, mis-aggregate evidence across images, and propagate early mistakes through their reasoning chains; results also vary substantially across body systems, organs, and modalities. We hope this work can catalyze research on clinically grounded, multi-image reasoning and accelerate progress toward more capable diagnostic AI systems.
>
---
#### [new 048] SpatialScore: Towards Unified Evaluation for Multimodal Spatial Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态空间理解评估任务，解决现有模型3D空间感知能力不足及评测缺失问题。提出VGBench和综合基准SpatialScore（整合12数据集28K样本），开发SpatialAgent多智能体系统，并通过广泛评测揭示挑战及验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.17012v1](http://arxiv.org/pdf/2505.17012v1)**

> **作者:** Haoning Wu; Xiao Huang; Yaohui Chen; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **备注:** Technical Report; Project Page: https://haoningwu3639.github.io/SpatialScore
>
> **摘要:** Multimodal large language models (MLLMs) have achieved impressive success in question-answering tasks, yet their capabilities for spatial understanding are less explored. This work investigates a critical question: do existing MLLMs possess 3D spatial perception and understanding abilities? Concretely, we make the following contributions in this paper: (i) we introduce VGBench, a benchmark specifically designed to assess MLLMs for visual geometry perception, e.g., camera pose and motion estimation; (ii) we propose SpatialScore, the most comprehensive and diverse multimodal spatial understanding benchmark to date, integrating VGBench with relevant data from the other 11 existing datasets. This benchmark comprises 28K samples across various spatial understanding tasks, modalities, and QA formats, along with a carefully curated challenging subset, SpatialScore-Hard; (iii) we develop SpatialAgent, a novel multi-agent system incorporating 9 specialized tools for spatial understanding, supporting both Plan-Execute and ReAct reasoning paradigms; (iv) we conduct extensive evaluations to reveal persistent challenges in spatial reasoning while demonstrating the effectiveness of SpatialAgent. We believe SpatialScore will offer valuable insights and serve as a rigorous benchmark for the next evolution of MLLMs.
>
---
#### [new 049] Satellites Reveal Mobility: A Commuting Origin-destination Flow Generator for Global Cities
- **分类: cs.CV; cs.CY; eess.IV**

- **简介: 该论文提出GlODGen模型，利用卫星图像与人口数据生成全球城市通勤OD流，解决传统数据采集成本高和隐私问题。通过视觉语言地理模型提取城市语义特征，结合图扩散模型生成OD流，验证其跨大陆城市的通用性，工具已开源。**

- **链接: [http://arxiv.org/pdf/2505.15870v1](http://arxiv.org/pdf/2505.15870v1)**

> **作者:** Can Rong; Xin Zhang; Yanxin Xi; Hongjie Sui; Jingtao Ding; Yong Li
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Commuting Origin-destination~(OD) flows, capturing daily population mobility of citizens, are vital for sustainable development across cities around the world. However, it is challenging to obtain the data due to the high cost of travel surveys and privacy concerns. Surprisingly, we find that satellite imagery, publicly available across the globe, contains rich urban semantic signals to support high-quality OD flow generation, with over 98\% expressiveness of traditional multisource hard-to-collect urban sociodemographic, economics, land use, and point of interest data. This inspires us to design a novel data generator, GlODGen, which can generate OD flow data for any cities of interest around the world. Specifically, GlODGen first leverages Vision-Language Geo-Foundation Models to extract urban semantic signals related to human mobility from satellite imagery. These features are then combined with population data to form region-level representations, which are used to generate OD flows via graph diffusion models. Extensive experiments on 4 continents and 6 representative cities show that GlODGen has great generalizability across diverse urban environments on different continents and can generate OD flow data for global cities highly consistent with real-world mobility data. We implement GlODGen as an automated tool, seamlessly integrating data acquisition and curation, urban semantic feature extraction, and OD flow generation together. It has been released at https://github.com/tsinghua-fib-lab/generate-od-pubtools.
>
---
#### [new 050] Decouple and Orthogonalize: A Data-Free Framework for LoRA Merging
- **分类: cs.CV**

- **简介: 该论文提出DO-Merging方法，针对LoRA模型合并任务，解决现有方法因参数幅度差异大导致性能下降的问题。通过分解参数为幅度和方向分量独立合并，并引入正交约束减少干扰，实现高效数据无关合并，理论与实验验证其跨领域有效性。**

- **链接: [http://arxiv.org/pdf/2505.15875v1](http://arxiv.org/pdf/2505.15875v1)**

> **作者:** Shenghe Zheng; Hongzhi Wang; Chenyu Huang; Xiaohui Wang; Tao Chen; Jiayuan Fan; Shuyue Hu; Peng Ye
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** With more open-source models available for diverse tasks, model merging has gained attention by combining models into one, reducing training, storage, and inference costs. Current research mainly focuses on model merging for full fine-tuning, overlooking the popular LoRA. However, our empirical analysis reveals that: a) existing merging methods designed for full fine-tuning perform poorly on LoRA; b) LoRA modules show much larger parameter magnitude variance than full fine-tuned weights; c) greater parameter magnitude variance correlates with worse merging performance. Considering that large magnitude variances cause deviations in the distribution of the merged parameters, resulting in information loss and performance degradation, we propose a Decoupled and Orthogonal merging approach(DO-Merging). By separating parameters into magnitude and direction components and merging them independently, we reduce the impact of magnitude differences on the directional alignment of the merged models, thereby preserving task information. Furthermore, we introduce a data-free, layer-wise gradient descent method with orthogonal constraints to mitigate interference during the merging of direction components. We provide theoretical guarantees for both the decoupling and orthogonal components. And we validate through extensive experiments across vision, language, and multi-modal domains that our proposed DO-Merging can achieve significantly higher performance than existing merging methods at a minimal cost. Notably, each component can be flexibly integrated with existing methods, offering near free-lunch improvements across tasks.
>
---
#### [new 051] MEgoHand: Multimodal Egocentric Hand-Object Interaction Motion Generation
- **分类: cs.CV**

- **简介: 该论文属于第一人称手-物交互运动生成任务，解决现有方法依赖3D先验、泛化性差及多模态误差累积问题。提出MEgoHand框架，融合视觉语言模型与单目深度估计获取上下文先验，结合DiT流量匹配策略生成稳定轨迹，并构建大规模RGB-D数据集，显著提升了运动生成精度与场景泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.16602v1](http://arxiv.org/pdf/2505.16602v1)**

> **作者:** Bohan Zhou; Yi Zhan; Zhongbin Zhang; Zongqing Lu
>
> **摘要:** Egocentric hand-object motion generation is crucial for immersive AR/VR and robotic imitation but remains challenging due to unstable viewpoints, self-occlusions, perspective distortion, and noisy ego-motion. Existing methods rely on predefined 3D object priors, limiting generalization to novel objects, which restricts their generalizability to novel objects. Meanwhile, recent multimodal approaches suffer from ambiguous generation from abstract textual cues, intricate pipelines for modeling 3D hand-object correlation, and compounding errors in open-loop prediction. We propose MEgoHand, a multimodal framework that synthesizes physically plausible hand-object interactions from egocentric RGB, text, and initial hand pose. MEgoHand introduces a bi-level architecture: a high-level "cerebrum" leverages a vision language model (VLM) to infer motion priors from visual-textual context and a monocular depth estimator for object-agnostic spatial reasoning, while a low-level DiT-based flow-matching policy generates fine-grained trajectories with temporal orthogonal filtering to enhance stability. To address dataset inconsistency, we design a dataset curation paradigm with an Inverse MANO Retargeting Network and Virtual RGB-D Renderer, curating a unified dataset of 3.35M RGB-D frames, 24K interactions, and 1.2K objects. Extensive experiments across five in-domain and two cross-domain datasets demonstrate the effectiveness of MEgoHand, achieving substantial reductions in wrist translation error (86.9%) and joint rotation error (34.1%), highlighting its capacity to accurately model fine-grained hand joint structures and generalize robustly across diverse scenarios.
>
---
#### [new 052] PAEFF: Precise Alignment and Enhanced Gated Feature Fusion for Face-Voice Association
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对人脸-语音关联任务，提出PAEFF方法，通过精确对齐人脸与语音嵌入空间并采用增强门控融合，解决现有方法负样本挖掘复杂及依赖边际参数的问题，提升跨模态关联性能。**

- **链接: [http://arxiv.org/pdf/2505.17002v1](http://arxiv.org/pdf/2505.17002v1)**

> **作者:** Abdul Hannan; Muhammad Arslan Manzoor; Shah Nawaz; Muhammad Irzam Liaqat; Markus Schedl; Mubashir Noman
>
> **备注:** Accepted at InterSpeech 2025
>
> **摘要:** We study the task of learning association between faces and voices, which is gaining interest in the multimodal community lately. These methods suffer from the deliberate crafting of negative mining procedures as well as the reliance on the distant margin parameter. These issues are addressed by learning a joint embedding space in which orthogonality constraints are applied to the fused embeddings of faces and voices. However, embedding spaces of faces and voices possess different characteristics and require spaces to be aligned before fusing them. To this end, we propose a method that accurately aligns the embedding spaces and fuses them with an enhanced gated fusion thereby improving the performance of face-voice association. Extensive experiments on the VoxCeleb dataset reveals the merits of the proposed approach.
>
---
#### [new 053] Deep Learning-Driven Ultra-High-Definition Image Restoration: A Survey
- **分类: cs.CV**

- **简介: 该综述论文系统总结了深度学习驱动的超高清图像修复任务，针对超清图像退化问题（如超分辨率、去模糊等），梳理了降级模型、数据集及算法进展，提出基于网络架构与采样策略的分类框架，并指明未来方向。**

- **链接: [http://arxiv.org/pdf/2505.16161v1](http://arxiv.org/pdf/2505.16161v1)**

> **作者:** Liyan Wang; Weixiang Zhou; Cong Wang; Kin-Man Lam; Zhixun Su; Jinshan Pan
>
> **备注:** 20 papers, 12 figures
>
> **摘要:** Ultra-high-definition (UHD) image restoration aims to specifically solve the problem of quality degradation in ultra-high-resolution images. Recent advancements in this field are predominantly driven by deep learning-based innovations, including enhancements in dataset construction, network architecture, sampling strategies, prior knowledge integration, and loss functions. In this paper, we systematically review recent progress in UHD image restoration, covering various aspects ranging from dataset construction to algorithm design. This serves as a valuable resource for understanding state-of-the-art developments in the field. We begin by summarizing degradation models for various image restoration subproblems, such as super-resolution, low-light enhancement, deblurring, dehazing, deraining, and desnowing, and emphasizing the unique challenges of their application to UHD image restoration. We then highlight existing UHD benchmark datasets and organize the literature according to degradation types and dataset construction methods. Following this, we showcase major milestones in deep learning-driven UHD image restoration, reviewing the progression of restoration tasks, technological developments, and evaluations of existing methods. We further propose a classification framework based on network architectures and sampling strategies, helping to clearly organize existing methods. Finally, we share insights into the current research landscape and propose directions for further advancements. A related repository is available at https://github.com/wlydlut/UHD-Image-Restoration-Survey.
>
---
#### [new 054] Representation Discrepancy Bridging Method for Remote Sensing Image-Text Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **简介: 该论文针对遥感图像-文本检索任务，解决现有参数高效微调方法中图像与文本模态优化不平衡问题。提出表示差异桥接方法RDB，包含不对称适配器（CMAA，含视觉增强和文本语义适配模块）及双任务一致性损失（DTCL），提升跨模态对齐。实验显示优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16756v1](http://arxiv.org/pdf/2505.16756v1)**

> **作者:** Hailong Ning; Siying Wang; Tao Lei; Xiaopeng Cao; Huanmin Dou; Bin Zhao; Asoke K. Nandi; Petia Radeva
>
> **摘要:** Remote Sensing Image-Text Retrieval (RSITR) plays a critical role in geographic information interpretation, disaster monitoring, and urban planning by establishing semantic associations between image and textual descriptions. Existing Parameter-Efficient Fine-Tuning (PEFT) methods for Vision-and-Language Pre-training (VLP) models typically adopt symmetric adapter structures for exploring cross-modal correlations. However, the strong discriminative nature of text modality may dominate the optimization process and inhibits image representation learning. The nonnegligible imbalanced cross-modal optimization remains a bottleneck to enhancing the model performance. To address this issue, this study proposes a Representation Discrepancy Bridging (RDB) method for the RSITR task. On the one hand, a Cross-Modal Asymmetric Adapter (CMAA) is designed to enable modality-specific optimization and improve feature alignment. The CMAA comprises a Visual Enhancement Adapter (VEA) and a Text Semantic Adapter (TSA). VEA mines fine-grained image features by Differential Attention (DA) mechanism, while TSA identifies key textual semantics through Hierarchical Attention (HA) mechanism. On the other hand, this study extends the traditional single-task retrieval framework to a dual-task optimization framework and develops a Dual-Task Consistency Loss (DTCL). The DTCL improves cross-modal alignment robustness through an adaptive weighted combination of cross-modal, classification, and exponential moving average consistency constraints. Experiments on RSICD and RSITMD datasets show that the proposed RDB method achieves a 6%-11% improvement in mR metrics compared to state-of-the-art PEFT methods and a 1.15%-2% improvement over the full fine-tuned GeoRSCLIP model.
>
---
#### [new 055] Redemption Score: An Evaluation Framework to Rank Image Captions While Redeeming Image Semantics and Language Pragmatics
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Redemption Score框架，用于评估和排名图像标题，解决现有指标无法兼顾视觉语义与语言质量的问题。通过融合Mutual Information Divergence（全局图文对齐）、DINO-based图像生成相似性（视觉定位）和BERTScore（文本相似度），在Flickr8k实现56.43 Kendall-τ，超越12种方法，无需任务训练。**

- **链接: [http://arxiv.org/pdf/2505.16180v1](http://arxiv.org/pdf/2505.16180v1)**

> **作者:** Ashim Dahal; Ankit Ghimire; Saydul Akbar Murad; Nick Rahimi
>
> **摘要:** Evaluating image captions requires cohesive assessment of both visual semantics and language pragmatics, which is often not entirely captured by most metrics. We introduce Redemption Score, a novel hybrid framework that ranks image captions by triangulating three complementary signals: (1) Mutual Information Divergence (MID) for global image-text distributional alignment, (2) DINO-based perceptual similarity of cycle-generated images for visual grounding, and (3) BERTScore for contextual text similarity against human references. A calibrated fusion of these signals allows Redemption Score to offer a more holistic assessment. On the Flickr8k benchmark, Redemption Score achieves a Kendall-$\tau$ of 56.43, outperforming twelve prior methods and demonstrating superior correlation with human judgments without requiring task-specific training. Our framework provides a more robust and nuanced evaluation by effectively redeeming image semantics and linguistic interpretability indicated by strong transfer of knowledge in the Conceptual Captions and MS COCO datasets.
>
---
#### [new 056] ARB: A Comprehensive Arabic Multimodal Reasoning Benchmark
- **分类: cs.CV**

- **简介: 该论文提出ARB，首个阿拉伯语多模态推理基准，填补非英语语言评估空白。针对阿拉伯语丰富的文化与语言背景，构建涵盖11领域的1356个多模态样本及5119步人工推理步骤，评估12种模型，揭示其在连贯性、忠实性及文化适配性上的不足，推动包容性AI发展。**

- **链接: [http://arxiv.org/pdf/2505.17021v1](http://arxiv.org/pdf/2505.17021v1)**

> **作者:** Sara Ghaboura; Ketan More; Wafa Alghallabi; Omkar Thawakar; Jorma Laaksonen; Hisham Cholakkal; Salman Khan; Rao Muhammad Anwer
>
> **备注:** Github : https://github.com/mbzuai-oryx/ARB, Huggingface: https://huggingface.co/datasets/MBZUAI/ARB
>
> **摘要:** As Large Multimodal Models (LMMs) become more capable, there is growing interest in evaluating their reasoning processes alongside their final outputs. However, most benchmarks remain focused on English, overlooking languages with rich linguistic and cultural contexts, such as Arabic. To address this gap, we introduce the Comprehensive Arabic Multimodal Reasoning Benchmark (ARB), the first benchmark designed to evaluate step-by-step reasoning in Arabic across both textual and visual modalities. ARB spans 11 diverse domains, including visual reasoning, document understanding, OCR, scientific analysis, and cultural interpretation. It comprises 1,356 multimodal samples paired with 5,119 human-curated reasoning steps and corresponding actions. We evaluated 12 state-of-the-art open- and closed-source LMMs and found persistent challenges in coherence, faithfulness, and cultural grounding. ARB offers a structured framework for diagnosing multimodal reasoning in underrepresented languages and marks a critical step toward inclusive, transparent, and culturally aware AI systems. We release the benchmark, rubric, and evaluation suit to support future research and reproducibility. Code available at: https://github.com/mbzuai-oryx/ARB
>
---
#### [new 057] On the use of Graphs for Satellite Image Time Series
- **分类: cs.CV**

- **简介: 该论文属于时空遥感分析任务，旨在解决卫星图像时间序列（SITS）数据的复杂性和时空关系建模问题。提出基于图的流水线，构建时空图以捕捉对象间交互，应用于土地覆盖分类和水资源预测等任务，并通过案例研究验证方法潜力，探讨未来改进方向。**

- **链接: [http://arxiv.org/pdf/2505.16685v1](http://arxiv.org/pdf/2505.16685v1)**

> **作者:** Corentin Dufourg; Charlotte Pelletier; Stéphane May; Sébastien Lefèvre
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** The Earth's surface is subject to complex and dynamic processes, ranging from large-scale phenomena such as tectonic plate movements to localized changes associated with ecosystems, agriculture, or human activity. Satellite images enable global monitoring of these processes with extensive spatial and temporal coverage, offering advantages over in-situ methods. In particular, resulting satellite image time series (SITS) datasets contain valuable information. To handle their large volume and complexity, some recent works focus on the use of graph-based techniques that abandon the regular Euclidean structure of satellite data to work at an object level. Besides, graphs enable modelling spatial and temporal interactions between identified objects, which are crucial for pattern detection, classification and regression tasks. This paper is an effort to examine the integration of graph-based methods in spatio-temporal remote-sensing analysis. In particular, it aims to present a versatile graph-based pipeline to tackle SITS analysis. It focuses on the construction of spatio-temporal graphs from SITS and their application to downstream tasks. The paper includes a comprehensive review and two case studies, which highlight the potential of graph-based approaches for land cover mapping and water resource forecasting. It also discusses numerous perspectives to resolve current limitations and encourage future developments.
>
---
#### [new 058] Efficient Prototype Consistency Learning in Medical Image Segmentation via Joint Uncertainty and Data Augmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，旨在解决标注数据不足导致的原型表达不足问题。提出EPCL-JUDA方法，结合不确定性量化与数据增强，利用Mean-Teacher框架生成优化伪标签及可靠原型，融合标注与未标注数据的原型形成全局原型，并引入原型网络降低内存需求，提升分割效果。**

- **链接: [http://arxiv.org/pdf/2505.16283v1](http://arxiv.org/pdf/2505.16283v1)**

> **作者:** Lijian Li; Yuanpeng He; Chi-Man Pun
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2404.10717
>
> **摘要:** Recently, prototype learning has emerged in semi-supervised medical image segmentation and achieved remarkable performance. However, the scarcity of labeled data limits the expressiveness of prototypes in previous methods, potentially hindering the complete representation of prototypes for class embedding. To overcome this issue, we propose an efficient prototype consistency learning via joint uncertainty quantification and data augmentation (EPCL-JUDA) to enhance the semantic expression of prototypes based on the framework of Mean-Teacher. The concatenation of original and augmented labeled data is fed into student network to generate expressive prototypes. Then, a joint uncertainty quantification method is devised to optimize pseudo-labels and generate reliable prototypes for original and augmented unlabeled data separately. High-quality global prototypes for each class are formed by fusing labeled and unlabeled prototypes, which are utilized to generate prototype-to-features to conduct consistency learning. Notably, a prototype network is proposed to reduce high memory requirements brought by the introduction of augmented data. Extensive experiments on Left Atrium, Pancreas-NIH, Type B Aortic Dissection datasets demonstrate EPCL-JUDA's superiority over previous state-of-the-art approaches, confirming the effectiveness of our framework. The code will be released soon.
>
---
#### [new 059] QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对长视频实时理解任务，解决视频解码缓慢和LLM预填充内存占用过高的瓶颈。提出QuickVideo系统，包含并行视频解码器（QuickDecoder）、内存优化预填充（QuickPrefill）及CPU/GPU任务重叠方案，实现长视频推理加速，支持资源受限环境下的高效处理。**

- **链接: [http://arxiv.org/pdf/2505.16175v1](http://arxiv.org/pdf/2505.16175v1)**

> **作者:** Benjamin Schneider; Dongfu Jiang; Chao Du; Tianyu Pang; Wenhu Chen
>
> **备注:** 19 pages, 6 figures, 2 tables
>
> **摘要:** Long-video understanding has emerged as a crucial capability in real-world applications such as video surveillance, meeting summarization, educational lecture analysis, and sports broadcasting. However, it remains computationally prohibitive for VideoLLMs, primarily due to two bottlenecks: 1) sequential video decoding, the process of converting the raw bit stream to RGB frames can take up to a minute for hour-long video inputs, and 2) costly prefilling of up to several million tokens for LLM inference, resulting in high latency and memory use. To address these challenges, we propose QuickVideo, a system-algorithm co-design that substantially accelerates long-video understanding to support real-time downstream applications. It comprises three key innovations: QuickDecoder, a parallelized CPU-based video decoder that achieves 2-3 times speedup by splitting videos into keyframe-aligned intervals processed concurrently; QuickPrefill, a memory-efficient prefilling method using KV-cache pruning to support more frames with less GPU memory; and an overlapping scheme that overlaps CPU video decoding with GPU inference. Together, these components infernece time reduce by a minute on long video inputs, enabling scalable, high-quality video understanding even on limited hardware. Experiments show that QuickVideo generalizes across durations and sampling rates, making long video processing feasible in practice.
>
---
#### [new 060] A Shape-Aware Total Body Photography System for In-focus Surface Coverage Optimization
- **分类: cs.CV**

- **简介: 该论文提出基于形状感知的全身摄影系统，优化人体表面图像分辨率与清晰度。针对传统系统在复杂3D结构中成像质量不足的问题，系统采用360度旋转摄像头、3D体型估计及自适应对焦算法，选择最佳焦距提升覆盖区域对焦效果。评估显示其在模拟和实测中分别达85%和95%清晰度，优于传统自动对焦方法。**

- **链接: [http://arxiv.org/pdf/2505.16228v1](http://arxiv.org/pdf/2505.16228v1)**

> **作者:** Wei-Lun Huang; Joshua Liu; Davood Tashayyod; Jun Kang; Amir Gandjbakhche; Misha Kazhdan; Mehran Armand
>
> **备注:** Accepted to JBHI
>
> **摘要:** Total Body Photography (TBP) is becoming a useful screening tool for patients at high risk for skin cancer. While much progress has been made, existing TBP systems can be further improved for automatic detection and analysis of suspicious skin lesions, which is in part related to the resolution and sharpness of acquired images. This paper proposes a novel shape-aware TBP system automatically capturing full-body images while optimizing image quality in terms of resolution and sharpness over the body surface. The system uses depth and RGB cameras mounted on a 360-degree rotary beam, along with 3D body shape estimation and an in-focus surface optimization method to select the optimal focus distance for each camera pose. This allows for optimizing the focused coverage over the complex 3D geometry of the human body given the calibrated camera poses. We evaluate the effectiveness of the system in capturing high-fidelity body images. The proposed system achieves an average resolution of 0.068 mm/pixel and 0.0566 mm/pixel with approximately 85% and 95% of surface area in-focus, evaluated on simulation data of diverse body shapes and poses as well as a real scan of a mannequin respectively. Furthermore, the proposed shape-aware focus method outperforms existing focus protocols (e.g. auto-focus). We believe the high-fidelity imaging enabled by the proposed system will improve automated skin lesion analysis for skin cancer screening.
>
---
#### [new 061] R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLM）推理能力提升任务，旨在解决强化学习（RL）中的稀疏奖励和优势消失问题。提出Share-GRPO方法，通过扩展问题空间、共享多样化推理路径及分层优势估计，优化模型推理性能。**

- **链接: [http://arxiv.org/pdf/2505.16673v1](http://arxiv.org/pdf/2505.16673v1)**

> **作者:** Huanjin Yao; Qixiang Yin; Jingyi Zhang; Min Yang; Yibo Wang; Wenhao Wu; Fei Su; Li Shen; Minghui Qiu; Dacheng Tao; Jiaxing Huang
>
> **备注:** Technical report
>
> **摘要:** In this work, we aim to incentivize the reasoning ability of Multimodal Large Language Models (MLLMs) via reinforcement learning (RL) and develop an effective approach that mitigates the sparse reward and advantage vanishing issues during RL. To this end, we propose Share-GRPO, a novel RL approach that tackle these issues by exploring and sharing diverse reasoning trajectories over expanded question space. Specifically, Share-GRPO first expands the question space for a given question via data transformation techniques, and then encourages MLLM to effectively explore diverse reasoning trajectories over the expanded question space and shares the discovered reasoning trajectories across the expanded questions during RL. In addition, Share-GRPO also shares reward information during advantage computation, which estimates solution advantages hierarchically across and within question variants, allowing more accurate estimation of relative advantages and improving the stability of policy training. Extensive evaluations over six widely-used reasoning benchmarks showcase the superior performance of our method. Code will be available at https://github.com/HJYao00/R1-ShareVL.
>
---
#### [new 062] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching
- **分类: cs.CV**

- **简介: 该论文属于少样本计数任务，旨在解决领域迁移导致的模型泛化能力不足问题。提出URM模型，通过整合视觉语言预训练模型的通用表示提升原型泛化性，增强跨领域鲁棒性，实现领域内和跨领域场景的最优计数效果。**

- **链接: [http://arxiv.org/pdf/2505.16778v1](http://arxiv.org/pdf/2505.16778v1)**

> **作者:** Xianing Chen; Si Huo; Borui Jiang; Hailin Hu; Xinghao Chen
>
> **备注:** CVPR 2025
>
> **摘要:** Few-shot counting estimates the number of target objects in an image using only a few annotated exemplars. However, domain shift severely hinders existing methods to generalize to unseen scenarios. This falls into the realm of single domain generalization that remains unexplored in few-shot counting. To solve this problem, we begin by analyzing the main limitations of current methods, which typically follow a standard pipeline that extract the object prototypes from exemplars and then match them with image feature to construct the correlation map. We argue that existing methods overlook the significance of learning highly generalized prototypes. Building on this insight, we propose the first single domain generalization few-shot counting model, Universal Representation Matching, termed URM. Our primary contribution is the discovery that incorporating universal vision-language representations distilled from a large scale pretrained vision-language model into the correlation construction process substantially improves robustness to domain shifts without compromising in domain performance. As a result, URM achieves state-of-the-art performance on both in domain and the newly introduced domain generalization setting.
>
---
#### [new 063] Self-Classification Enhancement and Correction for Weakly Supervised Object Detection
- **分类: cs.CV**

- **简介: 该论文属于弱监督目标检测任务，针对现有方法在多类分类任务间存在分类歧义且未充分利用其优势的问题，提出自我分类增强模块（整合类内二分类弥合任务差距）与推理阶段的自我修正算法（融合两任务结果减少误分类），提升检测性能。**

- **链接: [http://arxiv.org/pdf/2505.16294v1](http://arxiv.org/pdf/2505.16294v1)**

> **作者:** Yufei Yin; Lechao Cheng; Wengang Zhou; Jiajun Deng; Zhou Yu; Houqiang Li
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** In recent years, weakly supervised object detection (WSOD) has attracted much attention due to its low labeling cost. The success of recent WSOD models is often ascribed to the two-stage multi-class classification (MCC) task, i.e., multiple instance learning and online classification refinement. Despite achieving non-trivial progresses, these methods overlook potential classification ambiguities between these two MCC tasks and fail to leverage their unique strengths. In this work, we introduce a novel WSOD framework to ameliorate these two issues. For one thing, we propose a self-classification enhancement module that integrates intra-class binary classification (ICBC) to bridge the gap between the two distinct MCC tasks. The ICBC task enhances the network's discrimination between positive and mis-located samples in a class-wise manner and forges a mutually reinforcing relationship with the MCC task. For another, we propose a self-classification correction algorithm during inference, which combines the results of both MCC tasks to effectively reduce the mis-classified predictions. Extensive experiments on the prevalent VOC 2007 & 2012 datasets demonstrate the superior performance of our framework.
>
---
#### [new 064] Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究自回归图像生成中的CoT推理任务，对比DPO与GRPO算法，解决文本-图像一致性、美学优化及奖励模型设计问题。通过评估两算法的领域内性能与泛化能力，分析奖励模型影响，并探索三种扩展策略以提升性能。**

- **链接: [http://arxiv.org/pdf/2505.17017v1](http://arxiv.org/pdf/2505.17017v1)**

> **作者:** Chengzhuo Tong; Ziyu Guo; Renrui Zhang; Wenyu Shan; Xinyu Wei; Zhenghao Xing; Hongsheng Li; Pheng-Ann Heng
>
> **备注:** Code is released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
> **摘要:** Recent advancements underscore the significant role of Reinforcement Learning (RL) in enhancing the Chain-of-Thought (CoT) reasoning capabilities of large language models (LLMs). Two prominent RL algorithms, Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO), are central to these developments, showcasing different pros and cons. Autoregressive image generation, also interpretable as a sequential CoT reasoning process, presents unique challenges distinct from LLM-based CoT reasoning. These encompass ensuring text-image consistency, improving image aesthetic quality, and designing sophisticated reward models, rather than relying on simpler rule-based rewards. While recent efforts have extended RL to this domain, these explorations typically lack an in-depth analysis of the domain-specific challenges and the characteristics of different RL strategies. To bridge this gap, we provide the first comprehensive investigation of the GRPO and DPO algorithms in autoregressive image generation, evaluating their in-domain performance and out-of-domain generalization, while scrutinizing the impact of different reward models on their respective capabilities. Our findings reveal that GRPO and DPO exhibit distinct advantages, and crucially, that reward models possessing stronger intrinsic generalization capabilities potentially enhance the generalization potential of the applied RL algorithms. Furthermore, we systematically explore three prevalent scaling strategies to enhance both their in-domain and out-of-domain proficiency, deriving unique insights into efficiently scaling performance for each paradigm. We hope our study paves a new path for inspiring future work on developing more effective RL algorithms to achieve robust CoT reasoning in the realm of autoregressive image generation. Code is released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
---
#### [new 065] Sketchy Bounding-box Supervision for 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督3D实例分割任务，旨在解决仅用不准确边界框（sketchy bounding box）训练时的性能问题。提出Sketchy-3DIS框架，包含自适应伪标签生成器（处理重叠区域点分配）和粗细渐进分割器，通过联合训练提升实例分割质量，在ScanNetV2和S3DIS benchmarks上达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.16399v1](http://arxiv.org/pdf/2505.16399v1)**

> **作者:** Qian Deng; Le Hui; Jin Xie; Jian Yang
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Bounding box supervision has gained considerable attention in weakly supervised 3D instance segmentation. While this approach alleviates the need for extensive point-level annotations, obtaining accurate bounding boxes in practical applications remains challenging. To this end, we explore the inaccurate bounding box, named sketchy bounding box, which is imitated through perturbing ground truth bounding box by adding scaling, translation, and rotation. In this paper, we propose Sketchy-3DIS, a novel weakly 3D instance segmentation framework, which jointly learns pseudo labeler and segmentator to improve the performance under the sketchy bounding-box supervisions. Specifically, we first propose an adaptive box-to-point pseudo labeler that adaptively learns to assign points located in the overlapped parts between two sketchy bounding boxes to the correct instance, resulting in compact and pure pseudo instance labels. Then, we present a coarse-to-fine instance segmentator that first predicts coarse instances from the entire point cloud and then learns fine instances based on the region of coarse instances. Finally, by using the pseudo instance labels to supervise the instance segmentator, we can gradually generate high-quality instances through joint training. Extensive experiments show that our method achieves state-of-the-art performance on both the ScanNetV2 and S3DIS benchmarks, and even outperforms several fully supervised methods using sketchy bounding boxes. Code is available at https://github.com/dengq7/Sketchy-3DIS.
>
---
#### [new 066] GRIT: Teaching MLLMs to Think with Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，解决现有模型缺乏视觉信息整合的问题。提出GRIT方法，通过结合文本与边界框坐标生成视觉 grounding 推理链，并采用改进的RL算法GRPO-GR，仅需少量标注数据即可高效训练多模态模型。**

- **链接: [http://arxiv.org/pdf/2505.15879v1](http://arxiv.org/pdf/2505.15879v1)**

> **作者:** Yue Fan; Xuehai He; Diji Yang; Kaizhi Zheng; Ching-Chen Kuo; Yuting Zheng; Sravana Jyothi Narayanaraju; Xinze Guan; Xin Eric Wang
>
> **摘要:** Recent studies have demonstrated the efficacy of using Reinforcement Learning (RL) in building reasoning models that articulate chains of thoughts prior to producing final answers. However, despite ongoing advances that aim at enabling reasoning for vision-language tasks, existing open-source visual reasoning models typically generate reasoning content with pure natural language, lacking explicit integration of visual information. This limits their ability to produce clearly articulated and visually grounded reasoning chains. To this end, we propose Grounded Reasoning with Images and Texts (GRIT), a novel method for training MLLMs to think with images. GRIT introduces a grounded reasoning paradigm, in which models generate reasoning chains that interleave natural language and explicit bounding box coordinates. These coordinates point to regions of the input image that the model consults during its reasoning process. Additionally, GRIT is equipped with a reinforcement learning approach, GRPO-GR, built upon the GRPO algorithm. GRPO-GR employs robust rewards focused on the final answer accuracy and format of the grounded reasoning output, which eliminates the need for data with reasoning chain annotations or explicit bounding box labels. As a result, GRIT achieves exceptional data efficiency, requiring as few as 20 image-question-answer triplets from existing datasets. Comprehensive evaluations demonstrate that GRIT effectively trains MLLMs to produce coherent and visually grounded reasoning chains, showing a successful unification of reasoning and grounding abilities.
>
---
#### [new 067] SAMba-UNet: Synergizing SAM2 and Mamba in UNet with Heterogeneous Aggregation for Cardiac MRI Segmentation
- **分类: cs.CV**

- **简介: 该论文提出SAMba-UNet模型，针对心脏MRI自动分割中复杂病灶特征提取与边界定位难题，融合SAM2视觉模型、Mamba状态空间模型及UNet，设计动态特征融合模块与异构注意力机制，提升小病灶及右心室异常等结构的分割精度，在ACDC数据集Dice达0.9103，HD95为1.0859mm。**

- **链接: [http://arxiv.org/pdf/2505.16304v1](http://arxiv.org/pdf/2505.16304v1)**

> **作者:** Guohao Huo; Ruiting Dai; Hao Tang
>
> **摘要:** To address the challenge of complex pathological feature extraction in automated cardiac MRI segmentation, this study proposes an innovative dual-encoder architecture named SAMba-UNet. The framework achieves cross-modal feature collaborative learning by integrating the vision foundation model SAM2, the state-space model Mamba, and the classical UNet. To mitigate domain discrepancies between medical and natural images, a Dynamic Feature Fusion Refiner is designed, which enhances small lesion feature extraction through multi-scale pooling and a dual-path calibration mechanism across channel and spatial dimensions. Furthermore, a Heterogeneous Omni-Attention Convergence Module (HOACM) is introduced, combining global contextual attention with branch-selective emphasis mechanisms to effectively fuse SAM2's local positional semantics and Mamba's long-range dependency modeling capabilities. Experiments on the ACDC cardiac MRI dataset demonstrate that the proposed model achieves a Dice coefficient of 0.9103 and an HD95 boundary error of 1.0859 mm, significantly outperforming existing methods, particularly in boundary localization for complex pathological structures such as right ventricular anomalies. This work provides an efficient and reliable solution for automated cardiac disease diagnosis, and the code will be open-sourced.
>
---
#### [new 068] SEDD-PCC: A Single Encoder-Dual Decoder Framework For End-To-End Learned Point Cloud Compression
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于点云压缩任务，针对现有方法分别处理几何与属性导致计算复杂度高、共享特征利用不足的问题，提出SEDD-PCC框架：采用单编码器提取几何与属性的共享特征至统一潜在空间，再通过双解码器依次重建几何与属性，并结合知识蒸馏优化特征学习，提升压缩效率。**

- **链接: [http://arxiv.org/pdf/2505.16709v1](http://arxiv.org/pdf/2505.16709v1)**

> **作者:** Kai Hsiang Hsieh; Monyneath Yim; Jui Chiu Chiang
>
> **摘要:** To encode point clouds containing both geometry and attributes, most learning-based compression schemes treat geometry and attribute coding separately, employing distinct encoders and decoders. This not only increases computational complexity but also fails to fully exploit shared features between geometry and attributes. To address this limitation, we propose SEDD-PCC, an end-to-end learning-based framework for lossy point cloud compression that jointly compresses geometry and attributes. SEDD-PCC employs a single encoder to extract shared geometric and attribute features into a unified latent space, followed by dual specialized decoders that sequentially reconstruct geometry and attributes. Additionally, we incorporate knowledge distillation to enhance feature representation learning from a teacher model, further improving coding efficiency. With its simple yet effective design, SEDD-PCC provides an efficient and practical solution for point cloud compression. Comparative evaluations against both rule-based and learning-based methods demonstrate its competitive performance, highlighting SEDD-PCC as a promising AI-driven compression approach.
>
---
#### [new 069] Zero-Shot Anomaly Detection in Battery Thermal Images Using Visual Question Answering with Prior Knowledge
- **分类: cs.CV**

- **简介: 该论文属于零样本电池热图像异常检测任务，旨在解决传统方法依赖大量标注数据的问题。通过VQA模型结合电池热行为先验知识设计提示，无需电池特定训练数据，评估三个模型，显示其与需训练数据的SOTA方法相当，证明零样本方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.16674v1](http://arxiv.org/pdf/2505.16674v1)**

> **作者:** Marcella Astrid; Abdelrahman Shabayek; Djamila Aouada
>
> **备注:** Accepted in EUSIPCO 2025
>
> **摘要:** Batteries are essential for various applications, including electric vehicles and renewable energy storage, making safety and efficiency critical concerns. Anomaly detection in battery thermal images helps identify failures early, but traditional deep learning methods require extensive labeled data, which is difficult to obtain, especially for anomalies due to safety risks and high data collection costs. To overcome this, we explore zero-shot anomaly detection using Visual Question Answering (VQA) models, which leverage pretrained knowledge and textbased prompts to generalize across vision tasks. By incorporating prior knowledge of normal battery thermal behavior, we design prompts to detect anomalies without battery-specific training data. We evaluate three VQA models (ChatGPT-4o, LLaVa-13b, and BLIP-2) analyzing their robustness to prompt variations, repeated trials, and qualitative outputs. Despite the lack of finetuning on battery data, our approach demonstrates competitive performance compared to state-of-the-art models that are trained with the battery data. Our findings highlight the potential of VQA-based zero-shot learning for battery anomaly detection and suggest future directions for improving its effectiveness.
>
---
#### [new 070] DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DetailMaster基准，评估文本到图像模型处理长细节提示的能力，解决其在专业场景中因长文本导致性能下降的问题。通过四个维度（角色属性、位置、场景及空间关系）测试12个模型，揭示其在结构理解与细节处理上的系统性缺陷，推动组合推理研究，并开源数据集与工具。**

- **链接: [http://arxiv.org/pdf/2505.16915v1](http://arxiv.org/pdf/2505.16915v1)**

> **作者:** Qirui Jiao; Daoyuan Chen; Yilun Huang; Xika Lin; Ying Shen; Yaliang Li
>
> **备注:** 22 pages, 8 figures, 10 tables
>
> **摘要:** While recent text-to-image (T2I) models show impressive capabilities in synthesizing images from brief descriptions, their performance significantly degrades when confronted with long, detail-intensive prompts required in professional applications. We present DetailMaster, the first comprehensive benchmark specifically designed to evaluate T2I models' systematical abilities to handle extended textual inputs that contain complex compositional requirements. Our benchmark introduces four critical evaluation dimensions: Character Attributes, Structured Character Locations, Multi-Dimensional Scene Attributes, and Explicit Spatial/Interactive Relationships. The benchmark comprises long and detail-rich prompts averaging 284.89 tokens, with high quality validated by expert annotators. Evaluation on 7 general-purpose and 5 long-prompt-optimized T2I models reveals critical performance limitations: state-of-the-art models achieve merely ~50% accuracy in key dimensions like attribute binding and spatial reasoning, while all models showing progressive performance degradation as prompt length increases. Our analysis highlights systemic failures in structural comprehension and detail overload handling, motivating future research into architectures with enhanced compositional reasoning. We open-source the dataset, data curation code, and evaluation tools to advance detail-rich T2I generation and enable broad applications that would otherwise be infeasible due to the lack of a dedicated benchmark.
>
---
#### [new 071] Fusion of Foundation and Vision Transformer Model Features for Dermatoscopic Image Classification
- **分类: cs.CV**

- **简介: 该论文聚焦皮肤镜图像分类任务，旨在提升皮肤癌诊断准确性。通过对比皮肤病专用基础模型PanDerm与两种Vision Transformer（ViT base和Swin V2）的性能，采用非线性探测（MLP/XGBoost/TabNet）分析PanDerm冻结特征，并对ViT进行全微调。实验表明PanDerm-MLP与Swin表现相当，融合两者预测进一步提升效果。**

- **链接: [http://arxiv.org/pdf/2505.16338v1](http://arxiv.org/pdf/2505.16338v1)**

> **作者:** Amirreza Mahbod; Rupert Ecker; Ramona Woitek
>
> **备注:** 6 pages
>
> **摘要:** Accurate classification of skin lesions from dermatoscopic images is essential for diagnosis and treatment of skin cancer. In this study, we investigate the utility of a dermatology-specific foundation model, PanDerm, in comparison with two Vision Transformer (ViT) architectures (ViT base and Swin Transformer V2 base) for the task of skin lesion classification. Using frozen features extracted from PanDerm, we apply non-linear probing with three different classifiers, namely, multi-layer perceptron (MLP), XGBoost, and TabNet. For the ViT-based models, we perform full fine-tuning to optimize classification performance. Our experiments on the HAM10000 and MSKCC datasets demonstrate that the PanDerm-based MLP model performs comparably to the fine-tuned Swin transformer model, while fusion of PanDerm and Swin Transformer predictions leads to further performance improvements. Future work will explore additional foundation models, fine-tuning strategies, and advanced fusion techniques.
>
---
#### [new 072] SD-MAD: Sign-Driven Few-shot Multi-Anomaly Detection in Medical Images
- **分类: cs.CV**

- **简介: 该论文属于少样本多异常检测任务，解决现有方法忽视多异常类别区分及数据不足的问题。提出SD-MAD框架：通过放射学征兆对齐异常类别、自动选择征兆缓解数据局限，并设计三协议评估性能。**

- **链接: [http://arxiv.org/pdf/2505.16659v1](http://arxiv.org/pdf/2505.16659v1)**

> **作者:** Kaiyu Guo; Tan Pan; Chen Jiang; Zijian Wang; Brian C. Lovell; Limei Han; Yuan Cheng; Mahsa Baktashmotlagh
>
> **摘要:** Medical anomaly detection (AD) is crucial for early clinical intervention, yet it faces challenges due to limited access to high-quality medical imaging data, caused by privacy concerns and data silos. Few-shot learning has emerged as a promising approach to alleviate these limitations by leveraging the large-scale prior knowledge embedded in vision-language models (VLMs). Recent advancements in few-shot medical AD have treated normal and abnormal cases as a one-class classification problem, often overlooking the distinction among multiple anomaly categories. Thus, in this paper, we propose a framework tailored for few-shot medical anomaly detection in the scenario where the identification of multiple anomaly categories is required. To capture the detailed radiological signs of medical anomaly categories, our framework incorporates diverse textual descriptions for each category generated by a Large-Language model, under the assumption that different anomalies in medical images may share common radiological signs in each category. Specifically, we introduce SD-MAD, a two-stage Sign-Driven few-shot Multi-Anomaly Detection framework: (i) Radiological signs are aligned with anomaly categories by amplifying inter-anomaly discrepancy; (ii) Aligned signs are selected further to mitigate the effect of the under-fitting and uncertain-sample issue caused by limited medical data, employing an automatic sign selection strategy at inference. Moreover, we propose three protocols to comprehensively quantify the performance of multi-anomaly detection. Extensive experiments illustrate the effectiveness of our method.
>
---
#### [new 073] MAFE R-CNN: Selecting More Samples to Learn Category-aware Features for Small Object Detection
- **分类: cs.CV**

- **简介: 该论文针对复杂场景中小目标检测的挑战，提出MAFE R-CNN。其解决模型难以学习小目标判别特征及训练样本选择困难的问题，通过多线索样本选择（MCSS）结合IoU、置信度和尺寸平衡采样，并采用类别感知特征增强机制（CFEM）提升特征交互，从而优化小目标检测效果。**

- **链接: [http://arxiv.org/pdf/2505.16442v1](http://arxiv.org/pdf/2505.16442v1)**

> **作者:** Yichen Li; Qiankun Liu; Zhenchao Jin; Jiuzhe Wei; Jing Nie; Ying Fu
>
> **摘要:** Small object detection in intricate environments has consistently represented a major challenge in the field of object detection. In this paper, we identify that this difficulty stems from the detectors' inability to effectively learn discriminative features for objects of small size, compounded by the complexity of selecting high-quality small object samples during training, which motivates the proposal of the Multi-Clue Assignment and Feature Enhancement R-CNN.Specifically, MAFE R-CNN integrates two pivotal components.The first is the Multi-Clue Sample Selection (MCSS) strategy, in which the Intersection over Union (IoU) distance, predicted category confidence, and ground truth region sizes are leveraged as informative clues in the sample selection process. This methodology facilitates the selection of diverse positive samples and ensures a balanced distribution of object sizes during training, thereby promoting effective model learning.The second is the Category-aware Feature Enhancement Mechanism (CFEM), where we propose a simple yet effective category-aware memory module to explore the relationships among object features. Subsequently, we enhance the object feature representation by facilitating the interaction between category-aware features and candidate box features.Comprehensive experiments conducted on the large-scale small object dataset SODA validate the effectiveness of the proposed method. The code will be made publicly available.
>
---
#### [new 074] Investigating Fine- and Coarse-grained Structural Correspondences Between Deep Neural Networks and Human Object Image Similarity Judgments Using Unsupervised Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文通过无监督对齐方法（Gromov-Wasserstein最优传输），比较深度网络与人类物体表征的细/粗粒度对应。研究发现CLIP模型在两层面均与人类高度匹配，而自监督模型仅捕获粗类别结构，揭示语言信息对精细表征的关键作用及自监督学习的局限性。**

- **链接: [http://arxiv.org/pdf/2505.16419v1](http://arxiv.org/pdf/2505.16419v1)**

> **作者:** Soh Takahashi; Masaru Sasaki; Ken Takeda; Masafumi Oizumi
>
> **备注:** 34 pages, 6 figures
>
> **摘要:** The learning mechanisms by which humans acquire internal representations of objects are not fully understood. Deep neural networks (DNNs) have emerged as a useful tool for investigating this question, as they have internal representations similar to those of humans as a byproduct of optimizing their objective functions. While previous studies have shown that models trained with various learning paradigms - such as supervised, self-supervised, and CLIP - acquire human-like representations, it remains unclear whether their similarity to human representations is primarily at a coarse category level or extends to finer details. Here, we employ an unsupervised alignment method based on Gromov-Wasserstein Optimal Transport to compare human and model object representations at both fine-grained and coarse-grained levels. The unique feature of this method compared to conventional representational similarity analysis is that it estimates optimal fine-grained mappings between the representation of each object in human and model representations. We used this unsupervised alignment method to assess the extent to which the representation of each object in humans is correctly mapped to the corresponding representation of the same object in models. Using human similarity judgments of 1,854 objects from the THINGS dataset, we find that models trained with CLIP consistently achieve strong fine- and coarse-grained matching with human object representations. In contrast, self-supervised models showed limited matching at both fine- and coarse-grained levels, but still formed object clusters that reflected human coarse category structure. Our results offer new insights into the role of linguistic information in acquiring precise object representations and the potential of self-supervised learning to capture coarse categorical structures.
>
---
#### [new 075] Native Segmentation Vision Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于语义分割任务，旨在解决传统均匀下采样在特征提取中的不足。提出Native Segmentation Vision Transformer，通过动态内容感知分组层替代传统下采样，根据图像边界和语义动态分配token，堆叠后形成原生层次化分割，无需额外分割头即可生成强分割mask，实现零样本分割与高效模型设计。**

- **链接: [http://arxiv.org/pdf/2505.16993v1](http://arxiv.org/pdf/2505.16993v1)**

> **作者:** Guillem Brasó; Aljoša Ošep; Laura Leal-Taixé
>
> **摘要:** Uniform downsampling remains the de facto standard for reducing spatial resolution in vision backbones. In this work, we propose an alternative design built around a content-aware spatial grouping layer, that dynamically assigns tokens to a reduced set based on image boundaries and their semantic content. Stacking our grouping layer across consecutive backbone stages results in hierarchical segmentation that arises natively in the feature extraction process, resulting in our coined Native Segmentation Vision Transformer. We show that a careful design of our architecture enables the emergence of strong segmentation masks solely from grouping layers, that is, without additional segmentation-specific heads. This sets the foundation for a new paradigm of native, backbone-level segmentation, which enables strong zero-shot results without mask supervision, as well as a minimal and efficient standalone model design for downstream segmentation tasks. Our project page is https://research.nvidia.com/labs/dvl/projects/native-segmentation.
>
---
#### [new 076] Generative AI for Autonomous Driving: A Review
- **分类: cs.CV; cs.RO**

- **简介: 该综述论文探讨生成AI在自动驾驶中的应用，分析其在静态地图创建、动态场景生成、轨迹预测及路径规划等任务中的作用，比较VAE、GAN、扩散模型等方法的优缺点，提出混合方法提升系统鲁棒性，讨论安全、可解释性及实时性挑战并提出改进建议。**

- **链接: [http://arxiv.org/pdf/2505.15863v1](http://arxiv.org/pdf/2505.15863v1)**

> **作者:** Katharina Winter; Abhishek Vivekanandan; Rupert Polley; Yinzhe Shen; Christian Schlauch; Mohamed-Khalil Bouzidi; Bojan Derajic; Natalie Grabowsky; Annajoyce Mariani; Dennis Rochau; Giovanni Lucente; Harsh Yadav; Firas Mualla; Adam Molin; Sebastian Bernhard; Christian Wirth; Ömer Şahin Taş; Nadja Klein; Fabian B. Flohr; Hanno Gottschalk
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Generative AI (GenAI) is rapidly advancing the field of Autonomous Driving (AD), extending beyond traditional applications in text, image, and video generation. We explore how generative models can enhance automotive tasks, such as static map creation, dynamic scenario generation, trajectory forecasting, and vehicle motion planning. By examining multiple generative approaches ranging from Variational Autoencoder (VAEs) over Generative Adversarial Networks (GANs) and Invertible Neural Networks (INNs) to Generative Transformers (GTs) and Diffusion Models (DMs), we highlight and compare their capabilities and limitations for AD-specific applications. Additionally, we discuss hybrid methods integrating conventional techniques with generative approaches, and emphasize their improved adaptability and robustness. We also identify relevant datasets and outline open research questions to guide future developments in GenAI. Finally, we discuss three core challenges: safety, interpretability, and realtime capabilities, and present recommendations for image generation, dynamic scenario generation, and planning.
>
---
#### [new 077] Challenger: Affordable Adversarial Driving Video Generation
- **分类: cs.CV**

- **简介: 论文提出Challenger框架，生成逼真对抗性驾驶视频，解决现有方法无法提供真实传感器数据测试自动驾驶的问题。通过物理感知轨迹优化和定制评分函数，生成cut-ins、急变道等场景，显著提升AD模型碰撞率，且对抗行为跨模型有效。**

- **链接: [http://arxiv.org/pdf/2505.15880v1](http://arxiv.org/pdf/2505.15880v1)**

> **作者:** Zhiyuan Xu; Bohan Li; Huan-ang Gao; Mingju Gao; Yong Chen; Ming Liu; Chenxu Yan; Hang Zhao; Shuo Feng; Hao Zhao
>
> **备注:** Project page: https://pixtella.github.io/Challenger/
>
> **摘要:** Generating photorealistic driving videos has seen significant progress recently, but current methods largely focus on ordinary, non-adversarial scenarios. Meanwhile, efforts to generate adversarial driving scenarios often operate on abstract trajectory or BEV representations, falling short of delivering realistic sensor data that can truly stress-test autonomous driving (AD) systems. In this work, we introduce Challenger, a framework that produces physically plausible yet photorealistic adversarial driving videos. Generating such videos poses a fundamental challenge: it requires jointly optimizing over the space of traffic interactions and high-fidelity sensor observations. Challenger makes this affordable through two techniques: (1) a physics-aware multi-round trajectory refinement process that narrows down candidate adversarial maneuvers, and (2) a tailored trajectory scoring function that encourages realistic yet adversarial behavior while maintaining compatibility with downstream video synthesis. As tested on the nuScenes dataset, Challenger generates a diverse range of aggressive driving scenarios-including cut-ins, sudden lane changes, tailgating, and blind spot intrusions-and renders them into multiview photorealistic videos. Extensive evaluations show that these scenarios significantly increase the collision rate of state-of-the-art end-to-end AD models (UniAD, VAD, SparseDrive, and DiffusionDrive), and importantly, adversarial behaviors discovered for one model often transfer to others.
>
---
#### [new 078] Temporal and Spatial Feature Fusion Framework for Dynamic Micro Expression Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动态微表情识别任务，旨在解决微表情短暂、局部导致的识别率低（仅50%）问题。提出TSFmicro框架，融合 retention网络与Transformer，通过创新的平行时空特征融合方法，捕捉语义级"时空"关联，提升模型语义表达，在三个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16372v1](http://arxiv.org/pdf/2505.16372v1)**

> **作者:** Feng Liu; Bingyu Nan; Xuezhong Qian; Xiaolan Fu
>
> **备注:** 17 pages
>
> **摘要:** When emotions are repressed, an individual's true feelings may be revealed through micro-expressions. Consequently, micro-expressions are regarded as a genuine source of insight into an individual's authentic emotions. However, the transient and highly localised nature of micro-expressions poses a significant challenge to their accurate recognition, with the accuracy rate of micro-expression recognition being as low as 50%, even for professionals. In order to address these challenges, it is necessary to explore the field of dynamic micro expression recognition (DMER) using multimodal fusion techniques, with special attention to the diverse fusion of temporal and spatial modal features. In this paper, we propose a novel Temporal and Spatial feature Fusion framework for DMER (TSFmicro). This framework integrates a Retention Network (RetNet) and a transformer-based DMER network, with the objective of efficient micro-expression recognition through the capture and fusion of temporal and spatial relations. Meanwhile, we propose a novel parallel time-space fusion method from the perspective of modal fusion, which fuses spatio-temporal information in high-dimensional feature space, resulting in complementary "where-how" relationships at the semantic level and providing richer semantic information for the model. The experimental results demonstrate the superior performance of the TSFmicro method in comparison to other contemporary state-of-the-art methods. This is evidenced by its effectiveness on three well-recognised micro-expression datasets.
>
---
#### [new 079] Position: Agentic Systems Constitute a Key Component of Next-Generation Intelligent Image Processing
- **分类: cs.CV**

- **简介: 该论文属于立场类研究，主张图像处理需从模型驱动转向代理系统设计以解决现有模型泛化差、适应性弱的问题。提出通过开发可动态组合工具的智能代理系统提升问题解决灵活性，并分析模型局限、制定代理设计原则及能力分级。**

- **链接: [http://arxiv.org/pdf/2505.16007v1](http://arxiv.org/pdf/2505.16007v1)**

> **作者:** Jinjin Gu
>
> **摘要:** This position paper argues that the image processing community should broaden its focus from purely model-centric development to include agentic system design as an essential complementary paradigm. While deep learning has significantly advanced capabilities for specific image processing tasks, current approaches face critical limitations in generalization, adaptability, and real-world problem-solving flexibility. We propose that developing intelligent agentic systems, capable of dynamically selecting, combining, and optimizing existing image processing tools, represents the next evolutionary step for the field. Such systems would emulate human experts' ability to strategically orchestrate different tools to solve complex problems, overcoming the brittleness of monolithic models. The paper analyzes key limitations of model-centric paradigms, establishes design principles for agentic image processing systems, and outlines different capability levels for such agents.
>
---
#### [new 080] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人运动学习任务，旨在解决现有离散潜在动作方法的信息损失与复杂动态建模不足问题。提出CoMo方法，通过早期时序特征差异机制和信息瓶颈原理学习连续运动表示，抑制噪声并提升零样本泛化能力，同时引入新评估指标，支持跨领域视频数据训练机器人策略。**

- **链接: [http://arxiv.org/pdf/2505.17006v1](http://arxiv.org/pdf/2505.17006v1)**

> **作者:** Jiange Yang; Yansong Shi; Haoyi Zhu; Mingyu Liu; Kaijing Ma; Yating Wang; Gangshan Wu; Tong He; Limin Wang
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Learning latent motion from Internet videos is crucial for building generalist robots. However, existing discrete latent action methods suffer from information loss and struggle with complex and fine-grained dynamics. We propose CoMo, which aims to learn more informative continuous motion representations from diverse, internet-scale videos. CoMo employs a early temporal feature difference mechanism to prevent model collapse and suppress static appearance noise, effectively discouraging shortcut learning problem. Furthermore, guided by the information bottleneck principle, we constrain the latent motion embedding dimensionality to achieve a better balance between retaining sufficient action-relevant information and minimizing the inclusion of action-irrelevant appearance noise. Additionally, we also introduce two new metrics for more robustly and affordably evaluating motion and guiding motion learning methods development: (i) the linear probing MSE of action prediction, and (ii) the cosine similarity between past-to-current and future-to-current motion embeddings. Critically, CoMo exhibits strong zero-shot generalization, enabling it to generate continuous pseudo actions for previously unseen video domains. This capability facilitates unified policy joint learning using pseudo actions derived from various action-less video datasets (such as cross-embodiment videos and, notably, human demonstration videos), potentially augmented with limited labeled robot data. Extensive experiments show that policies co-trained with CoMo pseudo actions achieve superior performance with both diffusion and autoregressive architectures in simulated and real-world settings.
>
---
#### [new 081] ALTo: Adaptive-Length Tokenizer for Autoregressive Mask Generation
- **分类: cs.CV**

- **简介: 该论文提出ALTo，解决多模态模型中固定token表示限制灵活性的问题。通过设计自适应token长度预测器、正则化项及可微分分块策略，集成到ALToLLM模型，并用GRPO平衡掩码质量与效率，在分割任务中实现自适应token成本的SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.16495v1](http://arxiv.org/pdf/2505.16495v1)**

> **作者:** Lingfeng Wang; Hualing Lin; Senda Chen; Tao Wang; Changxu Cheng; Yangyang Zhong; Dong Zheng; Wuyue Zhao
>
> **摘要:** While humans effortlessly draw visual objects and shapes by adaptively allocating attention based on their complexity, existing multimodal large language models (MLLMs) remain constrained by rigid token representations. Bridging this gap, we propose ALTo, an adaptive length tokenizer for autoregressive mask generation. To achieve this, a novel token length predictor is designed, along with a length regularization term and a differentiable token chunking strategy. We further build ALToLLM that seamlessly integrates ALTo into MLLM. Preferences on the trade-offs between mask quality and efficiency is implemented by group relative policy optimization (GRPO). Experiments demonstrate that ALToLLM achieves state-of-the-art performance with adaptive token cost on popular segmentation benchmarks. Code and models are released at https://github.com/yayafengzi/ALToLLM.
>
---
#### [new 082] Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于减少大型视觉-语言模型（LVLMs）幻觉问题的任务。针对现有方法计算成本高或干预效果不足的问题，提出SSL方法：通过稀疏自编码器识别与幻觉/事实相关的语义方向，精准调整模型表示，无需训练即能有效抑制幻觉，且跨模型适用性好、效率高。**

- **链接: [http://arxiv.org/pdf/2505.16146v1](http://arxiv.org/pdf/2505.16146v1)**

> **作者:** Zhenglin Hua; Jinghan He; Zijun Yao; Tianxu Han; Haiyun Guo; Yuheng Jia; Junfeng Fang
>
> **摘要:** Large vision-language models (LVLMs) have achieved remarkable performance on multimodal tasks such as visual question answering (VQA) and image captioning. However, they still suffer from hallucinations, generating text inconsistent with visual input, posing significant risks in real-world applications. Existing approaches to address this issue focus on incorporating external knowledge bases, alignment training, or decoding strategies, all of which require substantial computational cost and time. Recent works try to explore more efficient alternatives by adjusting LVLMs' internal representations. Although promising, these methods may cause hallucinations to be insufficiently suppressed or lead to excessive interventions that negatively affect normal semantics. In this work, we leverage sparse autoencoders (SAEs) to identify semantic directions closely associated with either hallucinations or actuality, realizing more precise and direct hallucination-related representations. Our analysis demonstrates that interventions along the faithful direction we identified can mitigate hallucinations, while those along the hallucinatory direction can exacerbate them. Building on these insights, we propose Steering LVLMs via SAE Latent Directions (SSL), a training-free method based on SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive experiments demonstrate that SSL significantly outperforms existing decoding approaches in mitigating hallucinations, while maintaining transferability across different model architectures with negligible additional time overhead.
>
---
#### [new 083] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频虚拟试穿任务，旨在解决时间不一致及人体-服装动态姿势交互不足的问题。提出DPIDM框架，利用扩散模型，通过骨架姿势适配器、层次注意力模块（建模空间与时序姿势交互）和时序正则化损失，显著提升生成效果，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16980v1](http://arxiv.org/pdf/2505.16980v1)**

> **作者:** Dong Li; Wenqi Zhong; Wei Yu; Yingwei Pan; Dingwen Zhang; Ting Yao; Junwei Han; Tao Mei
>
> **备注:** CVPR 2025
>
> **摘要:** Video virtual try-on aims to seamlessly dress a subject in a video with a specific garment. The primary challenge involves preserving the visual authenticity of the garment while dynamically adapting to the pose and physique of the subject. While existing methods have predominantly focused on image-based virtual try-on, extending these techniques directly to videos often results in temporal inconsistencies. Most current video virtual try-on approaches alleviate this challenge by incorporating temporal modules, yet still overlook the critical spatiotemporal pose interactions between human and garment. Effective pose interactions in videos should not only consider spatial alignment between human and garment poses in each frame but also account for the temporal dynamics of human poses throughout the entire video. With such motivation, we propose a new framework, namely Dynamic Pose Interaction Diffusion Models (DPIDM), to leverage diffusion models to delve into dynamic pose interactions for video virtual try-on. Technically, DPIDM introduces a skeleton-based pose adapter to integrate synchronized human and garment poses into the denoising network. A hierarchical attention module is then exquisitely designed to model intra-frame human-garment pose interactions and long-term human pose dynamics across frames through pose-aware spatial and temporal attention mechanisms. Moreover, DPIDM capitalizes on a temporal regularized attention loss between consecutive frames to enhance temporal consistency. Extensive experiments conducted on VITON-HD, VVT and ViViD datasets demonstrate the superiority of our DPIDM against the baseline methods. Notably, DPIDM achieves VFID score of 0.506 on VVT dataset, leading to 60.5% improvement over the state-of-the-art GPD-VVTO approach.
>
---
#### [new 084] Domain Adaptive Skin Lesion Classification via Conformal Ensemble of Vision Transformers
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于领域自适应的皮肤病变分类任务，旨在解决医学图像跨领域迁移中的模型性能下降和不确定性估计不足问题。提出CE-ViTs框架，通过集成视觉变换器并结合多数据集训练，利用符合性学习提升模型鲁棒性和领域适应性，实验显示覆盖率达90.38%，预测集规模显著增加。**

- **链接: [http://arxiv.org/pdf/2505.15997v1](http://arxiv.org/pdf/2505.15997v1)**

> **作者:** Mehran Zoravar; Shadi Alijani; Homayoun Najjaran
>
> **备注:** 5 pages, 4 figures, conference (ccece 2025)
>
> **摘要:** Exploring the trustworthiness of deep learning models is crucial, especially in critical domains such as medical imaging decision support systems. Conformal prediction has emerged as a rigorous means of providing deep learning models with reliable uncertainty estimates and safety guarantees. However, conformal prediction results face challenges due to the backbone model's struggles in domain-shifted scenarios, such as variations in different sources. To aim this challenge, this paper proposes a novel framework termed Conformal Ensemble of Vision Transformers (CE-ViTs) designed to enhance image classification performance by prioritizing domain adaptation and model robustness, while accounting for uncertainty. The proposed method leverages an ensemble of vision transformer models in the backbone, trained on diverse datasets including HAM10000, Dermofit, and Skin Cancer ISIC datasets. This ensemble learning approach, calibrated through the combined mentioned datasets, aims to enhance domain adaptation through conformal learning. Experimental results underscore that the framework achieves a high coverage rate of 90.38\%, representing an improvement of 9.95\% compared to the HAM10000 model. This indicates a strong likelihood that the prediction set includes the true label compared to singular models. Ensemble learning in CE-ViTs significantly improves conformal prediction performance, increasing the average prediction set size for challenging misclassified samples from 1.86 to 3.075.
>
---
#### [new 085] Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models
- **分类: cs.CV; cs.AI; 68T45, 68T07; I.2.10; I.4.8**

- **简介: 该论文研究多任务医疗图像分析，解决检测、定位和计数任务的协同优化问题。基于MedMultiPoints数据集，通过指令调优视觉语言模型（如Qwen2.5-VL-7B），采用LoRA微调策略，验证多任务训练在提升诊断精度（如降低计数误差）和鲁棒性方面的潜力，同时揭示边缘案例可靠性下降等权衡。**

- **链接: [http://arxiv.org/pdf/2505.16647v1](http://arxiv.org/pdf/2505.16647v1)**

> **作者:** Sushant Gautam; Michael A. Riegler; Pål Halvorsen
>
> **备注:** Accepted as a full paper at the 38th IEEE International Symposium on Computer-Based Medical Systems (CBMS) 2025
>
> **摘要:** We investigate fine-tuning Vision-Language Models (VLMs) for multi-task medical image understanding, focusing on detection, localization, and counting of findings in medical images. Our objective is to evaluate whether instruction-tuned VLMs can simultaneously improve these tasks, with the goal of enhancing diagnostic accuracy and efficiency. Using MedMultiPoints, a multimodal dataset with annotations from endoscopy (polyps and instruments) and microscopy (sperm cells), we reformulate each task into instruction-based prompts suitable for vision-language reasoning. We fine-tune Qwen2.5-VL-7B-Instruct using Low-Rank Adaptation (LoRA) across multiple task combinations. Results show that multi-task training improves robustness and accuracy. For example, it reduces the Count Mean Absolute Error (MAE) and increases Matching Accuracy in the Counting + Pointing task. However, trade-offs emerge, such as more zero-case point predictions, indicating reduced reliability in edge cases despite overall performance gains. Our study highlights the potential of adapting general-purpose VLMs to specialized medical tasks via prompt-driven fine-tuning. This approach mirrors clinical workflows, where radiologists simultaneously localize, count, and describe findings - demonstrating how VLMs can learn composite diagnostic reasoning patterns. The model produces interpretable, structured outputs, offering a promising step toward explainable and versatile medical AI. Code, model weights, and scripts will be released for reproducibility at https://github.com/simula/PointDetectCount.
>
---
#### [new 086] Joint Flow And Feature Refinement Using Attention For Video Restoration
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频修复任务，旨在解决利用时间信息时难以保持时间一致性的难题。提出JFFRA框架，通过注意力机制实现流对齐与特征修复的迭代协同优化，并采用遮挡感知损失减少闪烁，提升多任务修复性能。**

- **链接: [http://arxiv.org/pdf/2505.16434v1](http://arxiv.org/pdf/2505.16434v1)**

> **作者:** Ranjith Merugu; Mohammad Sameer Suhail; Akshay P Sarashetti; Venkata Bharath Reddy Reddem; Pankaj Kumar Bajpai; Amit Satish Unde
>
> **摘要:** Recent advancements in video restoration have focused on recovering high-quality video frames from low-quality inputs. Compared with static images, the performance of video restoration significantly depends on efficient exploitation of temporal correlations among successive video frames. The numerous techniques make use of temporal information via flow-based strategies or recurrent architectures. However, these methods often encounter difficulties in preserving temporal consistency as they utilize degraded input video frames. To resolve this issue, we propose a novel video restoration framework named Joint Flow and Feature Refinement using Attention (JFFRA). The proposed JFFRA is based on key philosophy of iteratively enhancing data through the synergistic collaboration of flow (alignment) and restoration. By leveraging previously enhanced features to refine flow and vice versa, JFFRA enables efficient feature enhancement using temporal information. This interplay between flow and restoration is executed at multiple scales, reducing the dependence on precise flow estimation. Moreover, we incorporate an occlusion-aware temporal loss function to enhance the network's capability in eliminating flickering artifacts. Comprehensive experiments validate the versatility of JFFRA across various restoration tasks such as denoising, deblurring, and super-resolution. Our method demonstrates a remarkable performance improvement of up to 1.62 dB compared to state-of-the-art approaches.
>
---
#### [new 087] GMatch: Geometry-Constrained Feature Matching for RGB-D Object Pose Estimation
- **分类: cs.CV**

- **简介: 该论文提出GMatch，一种无需训练的几何约束特征匹配方法，用于鲁棒的RGB-D物体6DoF姿态估计。针对传统方法在稀疏特征匹配中的局部模糊性问题，GMatch通过SE(3)几何一致性引导的增量搜索，结合完备几何特征确保全局匹配，与SIFT结合形成高效pipeline。实验显示其性能超越传统及学习方法，在纹理丰富场景表现优异。**

- **链接: [http://arxiv.org/pdf/2505.16144v1](http://arxiv.org/pdf/2505.16144v1)**

> **作者:** Ming Yang; Haoran Li
>
> **备注:** 9 pages + 3 pages references + 2 pages appendix; 6 figures; 1 table
>
> **摘要:** We present GMatch, a learning-free feature matcher designed for robust 6DoF object pose estimation, addressing common local ambiguities in sparse feature matching. Unlike traditional methods that rely solely on descriptor similarity, GMatch performs a guided, incremental search, enforcing SE(3)-invariant geometric consistency throughout the matching process. It leverages a provably complete set of geometric features that uniquely determine 3D keypoint configurations, ensuring globally consistent correspondences without the need for training or GPU support. When combined with classical descriptors such as SIFT, GMatch-SIFT forms a general-purpose pose estimation pipeline that offers strong interpretability and generalization across diverse objects and scenes. Experiments on the HOPE dataset show that GMatch outperforms both traditional and learning-based matchers, with GMatch-SIFT achieving or surpassing the performance of instance-level pose networks. On the YCB-Video dataset, GMatch-SIFT demonstrates high accuracy and low variance on texture-rich objects. These results not only validate the effectiveness of GMatch-SIFT for object pose estimation but also highlight the broader applicability of GMatch as a general-purpose feature matcher. Code will be released upon acceptance.
>
---
#### [new 088] An Effective Training Framework for Light-Weight Automatic Speech Recognition Models
- **分类: cs.CV**

- **简介: 该论文属于轻量级自动语音识别（ASR）模型训练任务，旨在解决大型ASR模型在低资源设备部署困难及压缩方法性能下降或训练耗时的问题。提出两步表示学习框架，通过单次大模型训练生成多个小型高效模型，在较少训练轮次下提升性能，实现三倍速度提升和12.54%词错误率优化。**

- **链接: [http://arxiv.org/pdf/2505.16991v1](http://arxiv.org/pdf/2505.16991v1)**

> **作者:** Abdul Hannan; Alessio Brutti; Shah Nawaz; Mubashir Noman
>
> **备注:** Accepted at InterSpeech 2025
>
> **摘要:** Recent advancement in deep learning encouraged developing large automatic speech recognition (ASR) models that achieve promising results while ignoring computational and memory constraints. However, deploying such models on low resource devices is impractical despite of their favorable performance. Existing approaches (pruning, distillation, layer skip etc.) transform the large models into smaller ones at the cost of significant performance degradation or require prolonged training of smaller models for better performance. To address these issues, we introduce an efficacious two-step representation learning based approach capable of producing several small sized models from a single large model ensuring considerably better performance in limited number of epochs. Comprehensive experimentation on ASR benchmarks reveals the efficacy of our approach, achieving three-fold training speed-up and up to 12.54% word error rate improvement.
>
---
#### [new 089] Efficient Motion Prompt Learning for Robust Visual Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉跟踪任务，针对现有跟踪器忽视视频时序关联性的问题，提出轻量级运动提示模块。通过运动编码器将长期轨迹嵌入视觉特征空间，并设计融合解码器与自适应权重机制动态结合视觉-运动特征，提升跟踪鲁棒性，且计算开销小。**

- **链接: [http://arxiv.org/pdf/2505.16321v1](http://arxiv.org/pdf/2505.16321v1)**

> **作者:** Jie Zhao; Xin Chen; Yongsheng Yuan; Michael Felsberg; Dong Wang; Huchuan Lu
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Due to the challenges of processing temporal information, most trackers depend solely on visual discriminability and overlook the unique temporal coherence of video data. In this paper, we propose a lightweight and plug-and-play motion prompt tracking method. It can be easily integrated into existing vision-based trackers to build a joint tracking framework leveraging both motion and vision cues, thereby achieving robust tracking through efficient prompt learning. A motion encoder with three different positional encodings is proposed to encode the long-term motion trajectory into the visual embedding space, while a fusion decoder and an adaptive weight mechanism are designed to dynamically fuse visual and motion features. We integrate our motion module into three different trackers with five models in total. Experiments on seven challenging tracking benchmarks demonstrate that the proposed motion module significantly improves the robustness of vision-based trackers, with minimal training costs and negligible speed sacrifice. Code is available at https://github.com/zj5559/Motion-Prompt-Tracking.
>
---
#### [new 090] AdvReal: Adversarial Patch Generation Framework with Application to Adversarial Safety Evaluation of Object Detection Systems
- **分类: cs.CV**

- **简介: 该论文属于对抗样本生成与模型安全评估任务，旨在解决自动驾驶中物体检测系统对现实对抗攻击的脆弱性问题。提出AdvReal框架，通过统一2D/3D联合训练和非刚性表面建模，生成高真实感对抗贴片，实验验证其在复杂环境下的强鲁棒性和跨模型迁移性。**

- **链接: [http://arxiv.org/pdf/2505.16402v1](http://arxiv.org/pdf/2505.16402v1)**

> **作者:** Yuanhao Huang; Yilong Ren; Jinlei Wang; Lujia Huo; Xuesong Bai; Jinchuan Zhang; Haiyan Yu
>
> **摘要:** Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in safety accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D samples to address the challenges of intra-class diversity and environmental variations in real-world scenarios. Building upon this framework, we introduce an adversarial sample reality enhancement approach that incorporates non-rigid surface modeling and a realistic 3D matching mechanism. We compare with 5 advanced adversarial patches and evaluate their attack performance on 8 object detecotrs, including single-stage, two-stage, and transformer-based models. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Moreover, proposed method demonstrates excellent robustness and transferability under multi-angle attacks, varying lighting conditions, and different distance in the physical world. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.
>
---
#### [new 091] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态大语言模型（MLLMs）的幻觉缓解任务，旨在解决视觉问答中初始和雪球式幻觉问题。提出FarSight方法，通过优化因果掩码动态分配注意力，抑制离群token干扰，增强上下文推理，并设计位置编码提升长序列建模，实验显示其有效降低图像和视频任务的幻觉。**

- **链接: [http://arxiv.org/pdf/2505.16652v1](http://arxiv.org/pdf/2505.16652v1)**

> **作者:** Feilong Tang; Chengzhi Liu; Zhongxing Xu; Ming Hu; Zelin Peng; Zhiwei Yang; Jionglong Su; Minquan Lin; Yifan Peng; Xuelian Cheng; Imran Razzak; Zongyuan Ge
>
> **备注:** Clarification note for the CVPR 2025 paper (FarSight). Prepared by a subset of the original authors; remaining co-authors are acknowledged in the text
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have significantly improved performance in visual question answering. However, they often suffer from hallucinations. In this work, hallucinations are categorized into two main types: initial hallucinations and snowball hallucinations. We argue that adequate contextual information can be extracted directly from the token interaction process. Inspired by causal inference in the decoding strategy, we propose to leverage causal masks to establish information propagation between multimodal tokens. The hypothesis is that insufficient interaction between those tokens may lead the model to rely on outlier tokens, overlooking dense and rich contextual cues. Therefore, we propose to intervene in the propagation process by tackling outlier tokens to enhance in-context inference. With this goal, we present FarSight, a versatile plug-and-play decoding strategy to reduce attention interference from outlier tokens merely by optimizing the causal mask. The heart of our method is effective token propagation. We design an attention register structure within the upper triangular matrix of the causal mask, dynamically allocating attention to capture attention diverted to outlier tokens. Moreover, a positional awareness encoding method with a diminishing masking rate is proposed, allowing the model to attend to further preceding tokens, especially for video sequence tasks. With extensive experiments, FarSight demonstrates significant hallucination-mitigating performance across different MLLMs on both image and video benchmarks, proving its effectiveness.
>
---
#### [new 092] Unsupervised Network Anomaly Detection with Autoencoders and Traffic Images
- **分类: cs.CV; cs.CR; eess.IV; eess.SP**

- **简介: 该论文属于无监督网络异常检测任务，旨在解决海量异构设备流量数据的高效安全检测问题。提出基于流量图像的紧凑表示方法（1秒时间窗），通过自编码器实现轻量化的无监督异常识别，减少复杂计算需求。**

- **链接: [http://arxiv.org/pdf/2505.16650v1](http://arxiv.org/pdf/2505.16650v1)**

> **作者:** Michael Neri; Sara Baldoni
>
> **备注:** Accepted for publication in EUSIPCO 2025
>
> **摘要:** Due to the recent increase in the number of connected devices, the need to promptly detect security issues is emerging. Moreover, the high number of communication flows creates the necessity of processing huge amounts of data. Furthermore, the connected devices are heterogeneous in nature, having different computational capacities. For this reason, in this work we propose an image-based representation of network traffic which allows to realize a compact summary of the current network conditions with 1-second time windows. The proposed representation highlights the presence of anomalies thus reducing the need for complex processing architectures. Finally, we present an unsupervised learning approach which effectively detects the presence of anomalies. The code and the dataset are available at https://github.com/michaelneri/image-based-network-traffic-anomaly-detection.
>
---
#### [new 093] Conditional Panoramic Image Generation via Masked Autoregressive Modeling
- **分类: cs.CV**

- **简介: 该论文提出统一框架PAR，解决全景图像生成中的两大问题：现有扩散模型因球面映射违反i.i.d假设而失效，及文本生成与全景补全任务分离。通过掩码自回归建模、循环填充和一致性对齐策略，实现跨任务生成，提升空间连贯性与生成质量。**

- **链接: [http://arxiv.org/pdf/2505.16862v1](http://arxiv.org/pdf/2505.16862v1)**

> **作者:** Chaoyang Wang; Xiangtai Li; Lu Qi; Xiaofan Lin; Jinbin Bai; Qianyu Zhou; Yunhai Tong
>
> **摘要:** Recent progress in panoramic image generation has underscored two critical limitations in existing approaches. First, most methods are built upon diffusion models, which are inherently ill-suited for equirectangular projection (ERP) panoramas due to the violation of the identically and independently distributed (i.i.d.) Gaussian noise assumption caused by their spherical mapping. Second, these methods often treat text-conditioned generation (text-to-panorama) and image-conditioned generation (panorama outpainting) as separate tasks, relying on distinct architectures and task-specific data. In this work, we propose a unified framework, Panoramic AutoRegressive model (PAR), which leverages masked autoregressive modeling to address these challenges. PAR avoids the i.i.d. assumption constraint and integrates text and image conditioning into a cohesive architecture, enabling seamless generation across tasks. To address the inherent discontinuity in existing generative models, we introduce circular padding to enhance spatial coherence and propose a consistency alignment strategy to improve generation quality. Extensive experiments demonstrate competitive performance in text-to-image generation and panorama outpainting tasks while showcasing promising scalability and generalization capabilities.
>
---
#### [new 094] Training-Free Reasoning and Reflection in MLLMs
- **分类: cs.CV**

- **简介: 该论文提出FRANK模型，属于无训练多模态推理任务。旨在无需重新训练或数据的情况下提升MLLMs的推理能力。通过分层融合视觉预训练MLLM与推理LLM（浅层保留视觉，深层强化文本推理），解决训练成本高和数据稀缺问题，实验显示其性能超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.16151v1](http://arxiv.org/pdf/2505.16151v1)**

> **作者:** Hongchen Wei; Zhenzhong Chen
>
> **摘要:** Recent advances in Reasoning LLMs (e.g., DeepSeek-R1 and OpenAI-o1) have showcased impressive reasoning capabilities via reinforcement learning. However, extending these capabilities to Multimodal LLMs (MLLMs) is hampered by the prohibitive costs of retraining and the scarcity of high-quality, verifiable multimodal reasoning datasets. This paper introduces FRANK Model, a training-FRee ANd r1-liKe MLLM that imbues off-the-shelf MLLMs with reasoning and reflection abilities, without any gradient updates or extra supervision. Our key insight is to decouple perception and reasoning across MLLM decoder layers. Specifically, we observe that compared to the deeper decoder layers, the shallow decoder layers allocate more attention to visual tokens, while the deeper decoder layers concentrate on textual semantics. This observation motivates a hierarchical weight merging approach that combines a visual-pretrained MLLM with a reasoning-specialized LLM. To this end, we propose a layer-wise, Taylor-derived closed-form fusion mechanism that integrates reasoning capacity into deep decoder layers while preserving visual grounding in shallow decoder layers. Extensive experiments on challenging multimodal reasoning benchmarks demonstrate the effectiveness of our approach. On the MMMU benchmark, our model FRANK-38B achieves an accuracy of 69.2, outperforming the strongest baseline InternVL2.5-38B by +5.3, and even surpasses the proprietary GPT-4o model. Our project homepage is at: http://iip.whu.edu.cn/frank/index.html
>
---
#### [new 095] Grounding Chest X-Ray Visual Question Answering with Generated Radiology Reports
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出结合生成放射报告的胸片VQA方法，处理单图异常检测与时序差异比较问题。通过两阶段模型（报告生成和答案生成），利用预测报告增强答案生成模块，统一处理两类问题，在Medical-Diff-VQA数据集达SOTA。任务为医学影像问答，解决如何有效融合报告提升模型性能并处理双模式问题。**

- **链接: [http://arxiv.org/pdf/2505.16624v1](http://arxiv.org/pdf/2505.16624v1)**

> **作者:** Francesco Dalla Serra; Patrick Schrempf; Chaoyang Wang; Zaiqiao Meng; Fani Deligianni; Alison Q. O'Neil
>
> **摘要:** We present a novel approach to Chest X-ray (CXR) Visual Question Answering (VQA), addressing both single-image image-difference questions. Single-image questions focus on abnormalities within a specific CXR ("What abnormalities are seen in image X?"), while image-difference questions compare two longitudinal CXRs acquired at different time points ("What are the differences between image X and Y?"). We further explore how the integration of radiology reports can enhance the performance of VQA models. While previous approaches have demonstrated the utility of radiology reports during the pre-training phase, we extend this idea by showing that the reports can also be leveraged as additional input to improve the VQA model's predicted answers. First, we propose a unified method that handles both types of questions and auto-regressively generates the answers. For single-image questions, the model is provided with a single CXR. For image-difference questions, the model is provided with two CXRs from the same patient, captured at different time points, enabling the model to detect and describe temporal changes. Taking inspiration from 'Chain-of-Thought reasoning', we demonstrate that performance on the CXR VQA task can be improved by grounding the answer generator module with a radiology report predicted for the same CXR. In our approach, the VQA model is divided into two steps: i) Report Generation (RG) and ii) Answer Generation (AG). Our results demonstrate that incorporating predicted radiology reports as evidence to the AG model enhances performance on both single-image and image-difference questions, achieving state-of-the-art results on the Medical-Diff-VQA dataset.
>
---
#### [new 096] CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering
- **分类: cs.CV**

- **简介: 该论文提出CT-Agent，针对3D CT医学影像问答任务。解决现有系统难以处理解剖结构复杂及跨切片空间关系的问题。通过解剖独立工具分解复杂度，并采用全局-局部 token 压缩策略捕捉空间关系，提升CT影像问答与报告生成的效率和准确性。**

- **链接: [http://arxiv.org/pdf/2505.16229v1](http://arxiv.org/pdf/2505.16229v1)**

> **作者:** Yuren Mao; Wenyi Xu; Yuyang Qin; Yunjun Gao
>
> **摘要:** Computed Tomography (CT) scan, which produces 3D volumetric medical data that can be viewed as hundreds of cross-sectional images (a.k.a. slices), provides detailed anatomical information for diagnosis. For radiologists, creating CT radiology reports is time-consuming and error-prone. A visual question answering (VQA) system that can answer radiologists' questions about some anatomical regions on the CT scan and even automatically generate a radiology report is urgently needed. However, existing VQA systems cannot adequately handle the CT radiology question answering (CTQA) task for: (1) anatomic complexity makes CT images difficult to understand; (2) spatial relationship across hundreds slices is difficult to capture. To address these issues, this paper proposes CT-Agent, a multimodal agentic framework for CTQA. CT-Agent adopts anatomically independent tools to break down the anatomic complexity; furthermore, it efficiently captures the across-slice spatial relationship with a global-local token compression strategy. Experimental results on two 3D chest CT datasets, CT-RATE and RadGenome-ChestCT, verify the superior performance of CT-Agent.
>
---
#### [new 097] RealEngine: Simulating Autonomous Driving in Realistic Context
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出RealEngine框架，解决现有驾驶模拟器在多模态感知、场景多样性、多智能体交互及效率上的不足。通过3D重建与视图合成技术分离场景背景与交通元素，实现真实动态场景组合，支持非反应模拟、安全测试及多智能体交互，形成可靠评估基准。**

- **链接: [http://arxiv.org/pdf/2505.16902v1](http://arxiv.org/pdf/2505.16902v1)**

> **作者:** Junzhe Jiang; Nan Song; Jingyu Li; Xiatian Zhu; Li Zhang
>
> **摘要:** Driving simulation plays a crucial role in developing reliable driving agents by providing controlled, evaluative environments. To enable meaningful assessments, a high-quality driving simulator must satisfy several key requirements: multi-modal sensing capabilities (e.g., camera and LiDAR) with realistic scene rendering to minimize observational discrepancies; closed-loop evaluation to support free-form trajectory behaviors; highly diverse traffic scenarios for thorough evaluation; multi-agent cooperation to capture interaction dynamics; and high computational efficiency to ensure affordability and scalability. However, existing simulators and benchmarks fail to comprehensively meet these fundamental criteria. To bridge this gap, this paper introduces RealEngine, a novel driving simulation framework that holistically integrates 3D scene reconstruction and novel view synthesis techniques to achieve realistic and flexible closed-loop simulation in the driving context. By leveraging real-world multi-modal sensor data, RealEngine reconstructs background scenes and foreground traffic participants separately, allowing for highly diverse and realistic traffic scenarios through flexible scene composition. This synergistic fusion of scene reconstruction and view synthesis enables photorealistic rendering across multiple sensor modalities, ensuring both perceptual fidelity and geometric accuracy. Building upon this environment, RealEngine supports three essential driving simulation categories: non-reactive simulation, safety testing, and multi-agent interaction, collectively forming a reliable and comprehensive benchmark for evaluating the real-world performance of driving agents.
>
---
#### [new 098] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像检索任务，旨在解决现有模型在属性聚焦查询中因全局嵌入忽略细节导致性能不足的问题。团队构建COCO-Facet基准测试，发现CLIP和MLLM模型表现不佳，提出通过可提示嵌入高亮关键属性，并设计加速策略提升实用性。**

- **链接: [http://arxiv.org/pdf/2505.15877v1](http://arxiv.org/pdf/2505.15877v1)**

> **作者:** Siting Li; Xiang Gao; Simon Shaolei Du
>
> **备注:** 25 pages, 5 figures
>
> **摘要:** While an image is worth more than a thousand words, only a few provide crucial information for a given task and thus should be focused on. In light of this, ideal text-to-image (T2I) retrievers should prioritize specific visual attributes relevant to queries. To evaluate current retrievers on handling attribute-focused queries, we build COCO-Facet, a COCO-based benchmark with 9,112 queries about diverse attributes of interest. We find that CLIP-like retrievers, which are widely adopted due to their efficiency and zero-shot ability, have poor and imbalanced performance, possibly because their image embeddings focus on global semantics and subjects while leaving out other details. Notably, we reveal that even recent Multimodal Large Language Model (MLLM)-based, stronger retrievers with a larger output dimension struggle with this limitation. Hence, we hypothesize that retrieving with general image embeddings is suboptimal for performing such queries. As a solution, we propose to use promptable image embeddings enabled by these multimodal retrievers, which boost performance by highlighting required attributes. Our pipeline for deriving such embeddings generalizes across query types, image pools, and base retriever architectures. To enhance real-world applicability, we offer two acceleration strategies: Pre-processing promptable embeddings and using linear approximations. We show that the former yields a 15% improvement in Recall@5 when prompts are predefined, while the latter achieves an 8% improvement when prompts are only available during inference.
>
---
#### [new 099] Temporal Object Captioning for Street Scene Videos from LiDAR Tracks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频时间对象标注任务，旨在解决现有模型在理解动态交通参与者时的时序语义不足及视觉静态偏差问题。提出基于LiDAR轨迹的规则化时序特征提取与模板式字幕生成方法，通过监督训练改进SwinBERT模型，实验证明其提升时序理解效果。**

- **链接: [http://arxiv.org/pdf/2505.16594v1](http://arxiv.org/pdf/2505.16594v1)**

> **作者:** Vignesh Gopinathan; Urs Zimmermann; Michael Arnold; Matthias Rottmann
>
> **摘要:** Video captioning models have seen notable advancements in recent years, especially with regard to their ability to capture temporal information. While many research efforts have focused on architectural advancements, such as temporal attention mechanisms, there remains a notable gap in understanding how models capture and utilize temporal semantics for effective temporal feature extraction, especially in the context of Advanced Driver Assistance Systems. We propose an automated LiDAR-based captioning procedure that focuses on the temporal dynamics of traffic participants. Our approach uses a rule-based system to extract essential details such as lane position and relative motion from object tracks, followed by a template-based caption generation. Our findings show that training SwinBERT, a video captioning model, using only front camera images and supervised with our template-based captions, specifically designed to encapsulate fine-grained temporal behavior, leads to improved temporal understanding consistently across three datasets. In conclusion, our results clearly demonstrate that integrating LiDAR-based caption supervision significantly enhances temporal understanding, effectively addressing and reducing the inherent visual/static biases prevalent in current state-of-the-art model architectures.
>
---
#### [new 100] Analyzing Hierarchical Structure in Vision Models with Sparse Autoencoders
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉模型表征分析任务，旨在探究模型是否隐含编码ImageNet层级结构。通过稀疏自编码器（SAEs）探测DINOv2各层激活，发现其通过逐层增强类别标记信息，形成与分类学结构一致的层级关系，建立系统化分析框架。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15970v1](http://arxiv.org/pdf/2505.15970v1)**

> **作者:** Matthew Lyle Olson; Musashi Hinck; Neale Ratzlaff; Changbai Li; Phillip Howard; Vasudev Lal; Shao-Yen Tseng
>
> **备注:** (Oral) CVPR 2025 Workshop on Mechanistic Interpretability for Vision. Authors 1 and 2 contributed equally
>
> **摘要:** The ImageNet hierarchy provides a structured taxonomy of object categories, offering a valuable lens through which to analyze the representations learned by deep vision models. In this work, we conduct a comprehensive analysis of how vision models encode the ImageNet hierarchy, leveraging Sparse Autoencoders (SAEs) to probe their internal representations. SAEs have been widely used as an explanation tool for large language models (LLMs), where they enable the discovery of semantically meaningful features. Here, we extend their use to vision models to investigate whether learned representations align with the ontological structure defined by the ImageNet taxonomy. Our results show that SAEs uncover hierarchical relationships in model activations, revealing an implicit encoding of taxonomic structure. We analyze the consistency of these representations across different layers of the popular vision foundation model DINOv2 and provide insights into how deep vision models internalize hierarchical category information by increasing information in the class token through each layer. Our study establishes a framework for systematic hierarchical analysis of vision model representations and highlights the potential of SAEs as a tool for probing semantic structure in deep networks.
>
---
#### [new 101] Panoptic Captioning: Seeking An Equivalency Bridge for Image and Text
- **分类: cs.CV**

- **简介: 该论文提出全景描述生成任务，旨在通过文本全面捕捉图像中所有实体、位置、属性及关系。针对现有大模型性能不足，提出数据引擎PancapEngine生成高质量数据，及分阶段生成方法PancapChain，并开发评估指标与测试集。实验显示其模型超越GPT-4o等SOTA模型。**

- **链接: [http://arxiv.org/pdf/2505.16334v1](http://arxiv.org/pdf/2505.16334v1)**

> **作者:** Kun-Yu Lin; Hongjun Wang; Weining Ren; Kai Han
>
> **备注:** Project page: https://visual-ai.github.io/pancap/
>
> **摘要:** This work introduces panoptic captioning, a novel task striving to seek the minimum text equivalence of images. We take the first step towards panoptic captioning by formulating it as a task of generating a comprehensive textual description for an image, which encapsulates all entities, their respective locations and attributes, relationships among entities, as well as global image state.Through an extensive evaluation, our work reveals that state-of-the-art Multi-modal Large Language Models (MLLMs) have limited performance in solving panoptic captioning. To address this, we propose an effective data engine named PancapEngine to produce high-quality data and a novel method named PancapChain to improve panoptic captioning. Specifically, our PancapEngine first detects diverse categories of entities in images by an elaborate detection suite, and then generates required panoptic captions using entity-aware prompts. Additionally, our PancapChain explicitly decouples the challenging panoptic captioning task into multiple stages and generates panoptic captions step by step. More importantly, we contribute a comprehensive metric named PancapScore and a human-curated test set for reliable model evaluation.Experiments show that our PancapChain-13B model can beat state-of-the-art open-source MLLMs like InternVL-2.5-78B and even surpass proprietary models like GPT-4o and Gemini-2.0-Pro, demonstrating the effectiveness of our data engine and method. Project page: https://visual-ai.github.io/pancap/
>
---
#### [new 102] OViP: Online Vision-Language Preference Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型幻觉抑制任务，旨在解决现有方法依赖低效负面样本导致的训练效果差问题。提出OViP框架，通过动态分析模型自身生成的幻觉输出，结合扩散模型合成负面图像，实时生成对比数据优化模型，同时改进评估协议。实验表明其有效降低幻觉并保持多模态能力。**

- **链接: [http://arxiv.org/pdf/2505.15963v1](http://arxiv.org/pdf/2505.15963v1)**

> **作者:** Shujun Liu; Siyuan Wang; Zejun Li; Jianxiang Wang; Cheng Zeng; Zhongyu Wei
>
> **备注:** 22 pages, 10 figures, 8 tables
>
> **摘要:** Large vision-language models (LVLMs) remain vulnerable to hallucination, often generating content misaligned with visual inputs. While recent approaches advance multi-modal Direct Preference Optimization (DPO) to mitigate hallucination, they typically rely on predefined or randomly edited negative samples that fail to reflect actual model errors, limiting training efficacy. In this work, we propose an Online Vision-language Preference Learning (OViP) framework that dynamically constructs contrastive training data based on the model's own hallucinated outputs. By identifying semantic differences between sampled response pairs and synthesizing negative images using a diffusion model, OViP generates more relevant supervision signals in real time. This failure-driven training enables adaptive alignment of both textual and visual preferences. Moreover, we refine existing evaluation protocols to better capture the trade-off between hallucination suppression and expressiveness. Experiments on hallucination and general benchmarks demonstrate that OViP effectively reduces hallucinations while preserving core multi-modal capabilities.
>
---
#### [new 103] Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
- **分类: cs.CV**

- **简介: 该论文提出Dimple——首个离散扩散多模态大语言模型，解决纯离散扩散训练的不稳定、性能不足及长度偏差问题。通过混合自回归与扩散训练范式提升性能（超LLaVA-NEXT 3.9%），并设计自信解码（迭代次数为响应长度的1/3）和预填充技术（加速1.5-7倍），实现高效推理及结构化响应控制，验证离散扩散模型的可行性。**

- **链接: [http://arxiv.org/pdf/2505.16990v1](http://arxiv.org/pdf/2505.16990v1)**

> **作者:** Runpeng Yu; Xinyin Ma; Xinchao Wang
>
> **摘要:** In this work, we propose Dimple, the first Discrete Diffusion Multimodal Large Language Model (DMLLM). We observe that training with a purely discrete diffusion approach leads to significant training instability, suboptimal performance, and severe length bias issues. To address these challenges, we design a novel training paradigm that combines an initial autoregressive phase with a subsequent diffusion phase. This approach yields the Dimple-7B model, trained on the same dataset and using a similar training pipeline as LLaVA-NEXT. Dimple-7B ultimately surpasses LLaVA-NEXT in performance by 3.9%, demonstrating that DMLLM can achieve performance comparable to that of autoregressive models. To improve inference efficiency, we propose a decoding strategy termed confident decoding, which dynamically adjusts the number of tokens generated at each step, significantly reducing the number of generation iterations. In autoregressive models, the number of forward iterations during generation equals the response length. With confident decoding, however, the number of iterations needed by Dimple is even only $\frac{\text{response length}}{3}$. We also re-implement the prefilling technique in autoregressive models and demonstrate that it does not significantly impact performance on most benchmark evaluations, while offering a speedup of 1.5x to 7x. Additionally, we explore Dimple's capability to precisely control its response using structure priors. These priors enable structured responses in a manner distinct from instruction-based or chain-of-thought prompting, and allow fine-grained control over response format and length, which is difficult to achieve in autoregressive models. Overall, this work validates the feasibility and advantages of DMLLM and enhances its inference efficiency and controllability. Code and models are available at https://github.com/yu-rp/Dimple.
>
---
#### [new 104] Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Suppression
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型（LVLM）幻觉缓解任务。针对现有方法增加推理延迟问题，提出SPIN方法：通过抑制对图像注意力低的头部，保留关键头，实验证明幻觉减少2.7倍，保持F1分数并提升1.8倍吞吐量。**

- **链接: [http://arxiv.org/pdf/2505.16411v1](http://arxiv.org/pdf/2505.16411v1)**

> **作者:** Sreetama Sarkar; Yue Che; Alex Gavin; Peter A. Beerel; Souvik Kundu
>
> **摘要:** Despite their remarkable progress in multimodal understanding tasks, large vision language models (LVLMs) often suffer from "hallucinations", generating texts misaligned with the visual context. Existing methods aimed at reducing hallucinations through inference time intervention incur a significant increase in latency. To mitigate this, we present SPIN, a task-agnostic attention-guided head suppression strategy that can be seamlessly integrated during inference, without incurring any significant compute or latency overhead. We investigate whether hallucination in LVLMs can be linked to specific model components. Our analysis suggests that hallucinations can be attributed to a dynamic subset of attention heads in each layer. Leveraging this insight, for each text query token, we selectively suppress attention heads that exhibit low attention to image tokens, keeping the top-K attention heads intact. Extensive evaluations on visual question answering and image description tasks demonstrate the efficacy of SPIN in reducing hallucination scores up to 2.7x while maintaining F1, and improving throughput by 1.8x compared to existing alternatives. Code is available at https://github.com/YUECHE77/SPIN.
>
---
#### [new 105] Efficient Correlation Volume Sampling for Ultra-High-Resolution Optical Flow Estimation
- **分类: cs.CV; cs.LG**

- **简介: 该论文聚焦光流估计任务，解决超高清图像因传统方法计算复杂度高、需降分辨率导致细节丢失的问题。提出高效相关体积采样方法，保持与RAFT相同的精度，速度提升达90%，内存降低95%，并优化SEA-RAFT模型，在8K数据集实现高精度、高效的最优结果。**

- **链接: [http://arxiv.org/pdf/2505.16942v1](http://arxiv.org/pdf/2505.16942v1)**

> **作者:** Karlis Martins Briedis; Markus Gross; Christopher Schroers
>
> **摘要:** Recent optical flow estimation methods often employ local cost sampling from a dense all-pairs correlation volume. This results in quadratic computational and memory complexity in the number of pixels. Although an alternative memory-efficient implementation with on-demand cost computation exists, this is slower in practice and therefore prior methods typically process images at reduced resolutions, missing fine-grained details. To address this, we propose a more efficient implementation of the all-pairs correlation volume sampling, still matching the exact mathematical operator as defined by RAFT. Our approach outperforms on-demand sampling by up to 90% while maintaining low memory usage, and performs on par with the default implementation with up to 95% lower memory usage. As cost sampling makes up a significant portion of the overall runtime, this can translate to up to 50% savings for the total end-to-end model inference in memory-constrained environments. Our evaluation of existing methods includes an 8K ultra-high-resolution dataset and an additional inference-time modification of the recent SEA-RAFT method. With this, we achieve state-of-the-art results at high resolutions both in accuracy and efficiency.
>
---
#### [new 106] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation
- **分类: cs.CV**

- **简介: 该论文提出UniPhy，一种统一神经构成模型，用于逆物理模拟任务。解决传统方法依赖材料类型信息或仅针对特定实例导致的泛化性差、精度低问题。通过跨材料共享训练学习物理属性，利用潜在优化匹配观测数据，实现未知材料属性推断及多样化场景重新模拟，提升逆推断的鲁棒性和准确性。**

- **链接: [http://arxiv.org/pdf/2505.16971v1](http://arxiv.org/pdf/2505.16971v1)**

> **作者:** Himangi Mittal; Peiye Zhuang; Hsin-Ying Lee; Shubham Tulsiani
>
> **备注:** CVPR 2025
>
> **摘要:** We propose UniPhy, a common latent-conditioned neural constitutive model that can encode the physical properties of diverse materials. At inference UniPhy allows `inverse simulation' i.e. inferring material properties by optimizing the scene-specific latent to match the available observations via differentiable simulation. In contrast to existing methods that treat such inference as system identification, UniPhy does not rely on user-specified material type information. Compared to prior neural constitutive modeling approaches which learn instance specific networks, the shared training across materials improves both, robustness and accuracy of the estimates. We train UniPhy using simulated trajectories across diverse geometries and materials -- elastic, plasticine, sand, and fluids (Newtonian & non-Newtonian). At inference, given an object with unknown material properties, UniPhy can infer the material properties via latent optimization to match the motion observations, and can then allow re-simulating the object under diverse scenarios. We compare UniPhy against prior inverse simulation methods, and show that the inference from UniPhy enables more accurate replay and re-simulation under novel conditions.
>
---
#### [new 107] CP-LLM: Context and Pixel Aware Large Language Model for Video Quality Assessment
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文属于视频质量评估（VQA）任务，旨在解决传统方法缺乏上下文理解及LLM对小失真敏感度不足的问题。提出CP-LLM模型，通过双视觉编码器分别捕捉视频上下文与像素失真，结合语言解码器联合推理，实现精准评分与可解释描述，多任务训练提升跨数据集性能与像素级失真敏感度。**

- **链接: [http://arxiv.org/pdf/2505.16025v1](http://arxiv.org/pdf/2505.16025v1)**

> **作者:** Wen Wen; Yaohong Wu; Yue Sheng; Neil Birkbeck; Balu Adsumilli; Yilin Wang
>
> **备注:** Under review
>
> **摘要:** Video quality assessment (VQA) is a challenging research topic with broad applications. Effective VQA necessitates sensitivity to pixel-level distortions and a comprehensive understanding of video context to accurately determine the perceptual impact of distortions. Traditional hand-crafted and learning-based VQA models mainly focus on pixel-level distortions and lack contextual understanding, while recent LLM-based models struggle with sensitivity to small distortions or handle quality scoring and description as separate tasks. To address these shortcomings, we introduce CP-LLM: a Context and Pixel aware Large Language Model. CP-LLM is a novel multimodal LLM architecture featuring dual vision encoders designed to independently analyze perceptual quality at both high-level (video context) and low-level (pixel distortion) granularity, along with a language decoder subsequently reasons about the interplay between these aspects. This design enables CP-LLM to simultaneously produce robust quality scores and interpretable quality descriptions, with enhanced sensitivity to pixel distortions (e.g. compression artifacts). The model is trained via a multi-task pipeline optimizing for score prediction, description generation, and pairwise comparisons. Experiment results demonstrate that CP-LLM achieves state-of-the-art cross-dataset performance on established VQA benchmarks and superior robustness to pixel distortions, confirming its efficacy for comprehensive and practical video quality assessment in real-world scenarios.
>
---
#### [new 108] Semantic Compression of 3D Objects for Open and Collaborative Virtual Worlds
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于语义的3D对象压缩方法，解决传统方法高压缩率下纹理失真和结构断裂问题。通过自然语言描述核心概念并用生成模型预测缺失结构，构建压缩管道，在Objaverse数据集实现105x压缩，优于传统方法在质量保留的关键区域。**

- **链接: [http://arxiv.org/pdf/2505.16679v1](http://arxiv.org/pdf/2505.16679v1)**

> **作者:** Jordan Dotzel; Tony Montes; Mohamed S. Abdelfattah; Zhiru Zhang
>
> **备注:** First two authors have equal contribution
>
> **摘要:** Traditional methods for 3D object compression operate only on structural information within the object vertices, polygons, and textures. These methods are effective at compression rates up to 10x for standard object sizes but quickly deteriorate at higher compression rates with texture artifacts, low-polygon counts, and mesh gaps. In contrast, semantic compression ignores structural information and operates directly on the core concepts to push to extreme levels of compression. In addition, it uses natural language as its storage format, which makes it natively human-readable and a natural fit for emerging applications built around large-scale, collaborative projects within augmented and virtual reality. It deprioritizes structural information like location, size, and orientation and predicts the missing information with state-of-the-art deep generative models. In this work, we construct a pipeline for 3D semantic compression from public generative models and explore the quality-compression frontier for 3D object compression. We apply this pipeline to achieve rates as high as 105x for 3D objects taken from the Objaverse dataset and show that semantic compression can outperform traditional methods in the important quality-preserving region around 100x compression.
>
---
#### [new 109] REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment Supercharges Diffusion Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于扩散模型训练优化任务，旨在解决Diffusion Transformers (DiTs) 训练缓慢及REPA方法后期失效的问题。提出HASTE方法，分两阶段训练：早期通过联合蒸馏教师模型的注意力图和特征投影加速收敛，后期终止对齐损失以释放模型生成能力，显著提升训练效率（28倍优化步骤减少），适用于多种扩散任务。**

- **链接: [http://arxiv.org/pdf/2505.16792v1](http://arxiv.org/pdf/2505.16792v1)**

> **作者:** Ziqiao Wang; Wangbo Zhao; Yuhao Zhou; Zekai Li; Zhiyuan Liang; Mingjia Shi; Xuanlei Zhao; Pengfei Zhou; Kaipeng Zhang; Zhangyang Wang; Kai Wang; Yang You
>
> **备注:** 24 pages
>
> **摘要:** Diffusion Transformers (DiTs) deliver state-of-the-art image quality, yet their training remains notoriously slow. A recent remedy -- representation alignment (REPA) that matches DiT hidden features to those of a non-generative teacher (e.g. DINO) -- dramatically accelerates the early epochs but plateaus or even degrades performance later. We trace this failure to a capacity mismatch: once the generative student begins modelling the joint data distribution, the teacher's lower-dimensional embeddings and attention patterns become a straitjacket rather than a guide. We then introduce HASTE (Holistic Alignment with Stage-wise Termination for Efficient training), a two-phase schedule that keeps the help and drops the hindrance. Phase I applies a holistic alignment loss that simultaneously distills attention maps (relational priors) and feature projections (semantic anchors) from the teacher into mid-level layers of the DiT, yielding rapid convergence. Phase II then performs one-shot termination that deactivates the alignment loss, once a simple trigger such as a fixed iteration is hit, freeing the DiT to focus on denoising and exploit its generative capacity. HASTE speeds up training of diverse DiTs without architecture changes. On ImageNet 256X256, it reaches the vanilla SiT-XL/2 baseline FID in 50 epochs and matches REPA's best FID in 500 epochs, amounting to a 28X reduction in optimization steps. HASTE also improves text-to-image DiTs on MS-COCO, demonstrating to be a simple yet principled recipe for efficient diffusion training across various tasks. Our code is available at https://github.com/NUS-HPC-AI-Lab/HASTE .
>
---
#### [new 110] Paired and Unpaired Image to Image Translation using Generative Adversarial Networks
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究图像到图像翻译任务，解决配对与非配对跨域转换问题。采用条件GAN处理配对数据，结合循环一致性损失应对非配对场景，实验不同损失函数、Patch-GAN结构，引入精度、召回率和FID等新指标定量分析，并进行定性验证。**

- **链接: [http://arxiv.org/pdf/2505.16310v1](http://arxiv.org/pdf/2505.16310v1)**

> **作者:** Gaurav Kumar; Soham Satyadharma; Harpreet Singh
>
> **备注:** 6 pages
>
> **摘要:** Image to image translation is an active area of research in the field of computer vision, enabling the generation of new images with different styles, textures, or resolutions while preserving their characteristic properties. Recent architectures leverage Generative Adversarial Networks (GANs) to transform input images from one domain to another. In this work, we focus on the study of both paired and unpaired image translation across multiple image domains. For the paired task, we used a conditional GAN model, and for the unpaired task, we trained it using cycle consistency loss. We experimented with different types of loss functions, multiple Patch-GAN sizes, and model architectures. New quantitative metrics - precision, recall, and FID score - were used for analysis. In addition, a qualitative study of the results of different experiments was conducted.
>
---
#### [new 111] CodeMerge: Codebook-Guided Model Merging for Robust Test-Time Adaptation in Autonomous Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对自动驾驶中测试时3D目标检测的不稳定适应问题，提出CodeMerge框架。通过低维特征指纹构建码本，利用岭杠杆评分高效合并模型，减少计算开销，提升nuScenes-C和KITTI数据集性能，同时优化下游任务。**

- **链接: [http://arxiv.org/pdf/2505.16524v1](http://arxiv.org/pdf/2505.16524v1)**

> **作者:** Huitong Yang; Zhuoxiao Chen; Fengyi Zhang; Zi Huang; Yadan Luo
>
> **摘要:** Maintaining robust 3D perception under dynamic and unpredictable test-time conditions remains a critical challenge for autonomous driving systems. Existing test-time adaptation (TTA) methods often fail in high-variance tasks like 3D object detection due to unstable optimization and sharp minima. While recent model merging strategies based on linear mode connectivity (LMC) offer improved stability by interpolating between fine-tuned checkpoints, they are computationally expensive, requiring repeated checkpoint access and multiple forward passes. In this paper, we introduce CodeMerge, a lightweight and scalable model merging framework that bypasses these limitations by operating in a compact latent space. Instead of loading full models, CodeMerge represents each checkpoint with a low-dimensional fingerprint derived from the source model's penultimate features and constructs a key-value codebook. We compute merging coefficients using ridge leverage scores on these fingerprints, enabling efficient model composition without compromising adaptation quality. Our method achieves strong performance across challenging benchmarks, improving end-to-end 3D detection 14.9% NDS on nuScenes-C and LiDAR-based detection by over 7.6% mAP on nuScenes-to-KITTI, while benefiting downstream tasks such as online mapping, motion prediction and planning even without training. Code and pretrained models are released in the supplementary material.
>
---
#### [new 112] GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于视觉生成任务，解决复杂文本提示（含多对象、精确空间关系）生成图像的困难。提出GoT-R1框架，结合强化学习与生成式思维链，设计双阶段多维奖励机制，通过MLLM评估推理过程与输出，提升语义、空间准确性及视觉质量，实验显示显著提升。**

- **链接: [http://arxiv.org/pdf/2505.17022v1](http://arxiv.org/pdf/2505.17022v1)**

> **作者:** Chengqi Duan; Rongyao Fang; Yuqing Wang; Kun Wang; Linjiang Huang; Xingyu Zeng; Hongsheng Li; Xihui Liu
>
> **备注:** Github page refer to: https://github.com/gogoduan/GoT-R1
>
> **摘要:** Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in visual generation. Building upon the Generation Chain-of-Thought approach, GoT-R1 enables models to autonomously discover effective reasoning strategies beyond predefined templates through carefully designed reinforcement learning. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in image generation by successfully transferring sophisticated reasoning capabilities to the visual generation domain. To facilitate future research, we make our code and pretrained models publicly available at https://github.com/gogoduan/GoT-R1.
>
---
#### [new 113] TAT-VPR: Ternary Adaptive Transformer for Dynamic and Efficient Visual Place Recognition
- **分类: cs.CV**

- **简介: 该论文针对视觉SLAM回环闭合任务，提出TAT-VPR模型：通过三值量化Transformer与动态激活稀疏门，实现运行时40%计算量削减且不降召回率；采用两阶段蒸馏保持描述子质量，使模型可在微无人机等嵌入式系统高效运行，同时达到SOTA定位精度。**

- **链接: [http://arxiv.org/pdf/2505.16447v1](http://arxiv.org/pdf/2505.16447v1)**

> **作者:** Oliver Grainge; Michael Milford; Indu Bodala; Sarvapali D. Ramchurn; Shoaib Ehsan
>
> **摘要:** TAT-VPR is a ternary-quantized transformer that brings dynamic accuracy-efficiency trade-offs to visual SLAM loop-closure. By fusing ternary weights with a learned activation-sparsity gate, the model can control computation by up to 40% at run-time without degrading performance (Recall@1). The proposed two-stage distillation pipeline preserves descriptor quality, letting it run on micro-UAV and embedded SLAM stacks while matching state-of-the-art localization accuracy.
>
---
#### [new 114] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration
- **分类: cs.CV**

- **简介: 该论文提出多天气夜间图像修复任务，解决复杂天气与光照共同退化问题。贡献包括AllWeatherNight数据集（光照感知合成退化图像）和ClearNight框架，通过Retinex双先验分离光照/纹理，并采用天气自适应协作机制，实现退化修复效果提升。**

- **链接: [http://arxiv.org/pdf/2505.16479v1](http://arxiv.org/pdf/2505.16479v1)**

> **作者:** Yuetong Liu; Yunqiu Xu; Yang Wei; Xiuli Bi; Bin Xiao
>
> **备注:** 17 pages, 20 figures
>
> **摘要:** Restoring nighttime images affected by multiple adverse weather conditions is a practical yet under-explored research problem, as multiple weather conditions often coexist in the real world alongside various lighting effects at night. This paper first explores the challenging multi-weather nighttime image restoration task, where various types of weather degradations are intertwined with flare effects. To support the research, we contribute the AllWeatherNight dataset, featuring large-scale high-quality nighttime images with diverse compositional degradations, synthesized using our introduced illumination-aware degradation generation. Moreover, we present ClearNight, a unified nighttime image restoration framework, which effectively removes complex degradations in one go. Specifically, ClearNight extracts Retinex-based dual priors and explicitly guides the network to focus on uneven illumination regions and intrinsic texture contents respectively, thereby enhancing restoration effectiveness in nighttime scenarios. In order to better represent the common and unique characters of multiple weather degradations, we introduce a weather-aware dynamic specific-commonality collaboration method, which identifies weather degradations and adaptively selects optimal candidate units associated with specific weather types. Our ClearNight achieves state-of-the-art performance on both synthetic and real-world images. Comprehensive ablation experiments validate the necessity of AllWeatherNight dataset as well as the effectiveness of ClearNight. Project page: https://henlyta.github.io/ClearNight/mainpage.html
>
---
#### [new 115] T2I-ConBench: Text-to-Image Benchmark for Continual Post-training
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出T2I-ConBench基准，解决文本到图像模型持续后训练缺乏标准化评估的问题。针对物品定制和领域增强场景，从知识保留、任务性能等四维度评估模型，测试十种方法，揭示无全能方案且跨任务泛化未解，公开工具加速研究。**

- **链接: [http://arxiv.org/pdf/2505.16875v1](http://arxiv.org/pdf/2505.16875v1)**

> **作者:** Zhehao Huang; Yuhang Liu; Yixin Lou; Zhengbao He; Mingzhen He; Wenxing Zhou; Tao Li; Kehan Li; Zeyi Huang; Xiaolin Huang
>
> **摘要:** Continual post-training adapts a single text-to-image diffusion model to learn new tasks without incurring the cost of separate models, but naive post-training causes forgetting of pretrained knowledge and undermines zero-shot compositionality. We observe that the absence of a standardized evaluation protocol hampers related research for continual post-training. To address this, we introduce T2I-ConBench, a unified benchmark for continual post-training of text-to-image models. T2I-ConBench focuses on two practical scenarios, item customization and domain enhancement, and analyzes four dimensions: (1) retention of generality, (2) target-task performance, (3) catastrophic forgetting, and (4) cross-task generalization. It combines automated metrics, human-preference modeling, and vision-language QA for comprehensive assessment. We benchmark ten representative methods across three realistic task sequences and find that no approach excels on all fronts. Even joint "oracle" training does not succeed for every task, and cross-task generalization remains unsolved. We release all datasets, code, and evaluation tools to accelerate research in continual post-training for text-to-image models.
>
---
#### [new 116] Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space
- **分类: cs.CV**

- **简介: 该论文属于视频重建与生成任务，旨在解决视频帧间动态令牌分配效率问题。提出AdapTok方法，通过块掩码策略、因果评分器及整数规划，在可控预算下实现自适应、内容感知的令牌分配，提升视频建模的效率与性能。**

- **链接: [http://arxiv.org/pdf/2505.17011v1](http://arxiv.org/pdf/2505.17011v1)**

> **作者:** Yan Li; Changyao Tian; Renqiu Xia; Ning Liao; Weiwei Guo; Junchi Yan; Hongsheng Li; Jifeng Dai; Hao Li; Xue Yang
>
> **备注:** Code: https://github.com/VisionXLab/AdapTok
>
> **摘要:** We propose AdapTok, an adaptive temporal causal video tokenizer that can flexibly allocate tokens for different frames based on video content. AdapTok is equipped with a block-wise masking strategy that randomly drops tail tokens of each block during training, and a block causal scorer to predict the reconstruction quality of video frames using different numbers of tokens. During inference, an adaptive token allocation strategy based on integer linear programming is further proposed to adjust token usage given predicted scores. Such design allows for sample-wise, content-aware, and temporally dynamic token allocation under a controllable overall budget. Extensive experiments for video reconstruction and generation on UCF-101 and Kinetics-600 demonstrate the effectiveness of our approach. Without additional image data, AdapTok consistently improves reconstruction quality and generation performance under different token budgets, allowing for more scalable and token-efficient generative video modeling.
>
---
#### [new 117] Image-to-Image Translation with Diffusion Transformers and CLIP-Based Image Conditioning
- **分类: cs.CV**

- **简介: 该论文提出基于扩散 Transformer（DiT）和 CLIP 图像条件的图像到图像转换方法，解决无文本/标签依赖下的高质量跨域映射问题。通过结合 CLIP 图像嵌入引导、语义相似度损失及感知损失，实现结构一致且视觉保真的转换，在人脸转漫画和边缘图转鞋类图像任务中验证效果，提供优于 GAN 的替代方案。**

- **链接: [http://arxiv.org/pdf/2505.16001v1](http://arxiv.org/pdf/2505.16001v1)**

> **作者:** Qiang Zhu; Kuan Lu; Menghao Huo; Yuxiao Li
>
> **摘要:** Image-to-image translation aims to learn a mapping between a source and a target domain, enabling tasks such as style transfer, appearance transformation, and domain adaptation. In this work, we explore a diffusion-based framework for image-to-image translation by adapting Diffusion Transformers (DiT), which combine the denoising capabilities of diffusion models with the global modeling power of transformers. To guide the translation process, we condition the model on image embeddings extracted from a pre-trained CLIP encoder, allowing for fine-grained and structurally consistent translations without relying on text or class labels. We incorporate both a CLIP similarity loss to enforce semantic consistency and an LPIPS perceptual loss to enhance visual fidelity during training. We validate our approach on two benchmark datasets: face2comics, which translates real human faces to comic-style illustrations, and edges2shoes, which translates edge maps to realistic shoe images. Experimental results demonstrate that DiT, combined with CLIP-based conditioning and perceptual similarity objectives, achieves high-quality, semantically faithful translations, offering a promising alternative to GAN-based models for paired image-to-image translation tasks.
>
---
#### [new 118] Towards Texture- And Shape-Independent 3D Keypoint Estimation in Birds
- **分类: cs.CV**

- **简介: 该论文属于鸟类三维姿态估计任务，旨在解决传统方法依赖纹理或形状的问题。基于3D-MuPPET框架，通过分割轮廓提取2D关键点，改进为不依赖纹理的3D姿态估计，并验证其在鸽子中的有效性及对其他鸟类的通用性。**

- **链接: [http://arxiv.org/pdf/2505.16633v1](http://arxiv.org/pdf/2505.16633v1)**

> **作者:** Valentin Schmuker; Alex Hoi Hang Chan; Bastian Goldluecke; Urs Waldmann
>
> **摘要:** In this paper, we present a texture-independent approach to estimate and track 3D joint positions of multiple pigeons. For this purpose, we build upon the existing 3D-MuPPET framework, which estimates and tracks the 3D poses of up to 10 pigeons using a multi-view camera setup. We extend this framework by using a segmentation method that generates silhouettes of the individuals, which are then used to estimate 2D keypoints. Following 3D-MuPPET, these 2D keypoints are triangulated to infer 3D poses, and identities are matched in the first frame and tracked in 2D across subsequent frames. Our proposed texture-independent approach achieves comparable accuracy to the original texture-dependent 3D-MuPPET framework. Additionally, we explore our approach's applicability to other bird species. To do that, we infer the 2D joint positions of four bird species without additional fine-tuning the model trained on pigeons and obtain preliminary promising results. Thus, we think that our approach serves as a solid foundation and inspires the development of more robust and accurate texture-independent pose estimation frameworks.
>
---
#### [new 119] LINEA: Fast and Accurate Line Detection Using Scalable Transformers
- **分类: cs.CV**

- **简介: 该论文提出LINEA，一种基于可变形线注意力（DLA）的线检测方法。针对现有Transformer模型速度慢且需预训练的缺陷，通过DLA消除预训练需求，提升推理速度与精度，在无分布数据集上表现更优。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16264v1](http://arxiv.org/pdf/2505.16264v1)**

> **作者:** Sebastian Janampa; Marios Pattichis
>
> **摘要:** Line detection is a basic digital image processing operation used by higher-level processing methods. Recently, transformer-based methods for line detection have proven to be more accurate than methods based on CNNs, at the expense of significantly lower inference speeds. As a result, video analysis methods that require low latencies cannot benefit from current transformer-based methods for line detection. In addition, current transformer-based models require pretraining attention mechanisms on large datasets (e.g., COCO or Object360). This paper develops a new transformer-based method that is significantly faster without requiring pretraining the attention mechanism on large datasets. We eliminate the need to pre-train the attention mechanism using a new mechanism, Deformable Line Attention (DLA). We use the term LINEA to refer to our new transformer-based method based on DLA. Extensive experiments show that LINEA is significantly faster and outperforms previous models on sAP in out-of-distribution dataset testing.
>
---
#### [new 120] TensorAR: Refinement is All You Need in Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文属于自回归图像生成任务，旨在解决AR模型无法迭代优化先前预测导致的生成质量不足问题。提出TensorAR方法，通过滑动窗口生成重叠图像块实现内容迭代优化，并设计离散张量噪声防止训练信息泄漏，实验验证其有效提升AR模型性能。**

- **链接: [http://arxiv.org/pdf/2505.16324v1](http://arxiv.org/pdf/2505.16324v1)**

> **作者:** Cheng Cheng; Lin Song; Yicheng Xiao; Yuxin Chen; Xuchong Zhang; Hongbin Sun; Ying Shan
>
> **摘要:** Autoregressive (AR) image generators offer a language-model-friendly approach to image generation by predicting discrete image tokens in a causal sequence. However, unlike diffusion models, AR models lack a mechanism to refine previous predictions, limiting their generation quality. In this paper, we introduce TensorAR, a new AR paradigm that reformulates image generation from next-token prediction to next-tensor prediction. By generating overlapping windows of image patches (tensors) in a sliding fashion, TensorAR enables iterative refinement of previously generated content. To prevent information leakage during training, we propose a discrete tensor noising scheme, which perturbs input tokens via codebook-indexed noise. TensorAR is implemented as a plug-and-play module compatible with existing AR models. Extensive experiments on LlamaGEN, Open-MAGVIT2, and RAR demonstrate that TensorAR significantly improves the generation performance of autoregressive models.
>
---
#### [new 121] Let Androids Dream of Electric Sheep: A Human-like Image Implication Understanding and Reasoning Framework
- **分类: cs.CV; cs.AI; cs.CY**

- **简介: 该论文提出LAD框架，解决AI在图像隐喻理解中的上下文缺口问题。通过感知（视觉转文本）、搜索（跨域知识整合）、推理（上下文化表达）三阶段，提升图像隐含意义理解能力，在中英文基准测试中超越15+多模态模型，开源实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.17019v1](http://arxiv.org/pdf/2505.17019v1)**

> **作者:** Chenhao Zhang; Yazhe Niu
>
> **备注:** 16 pages, 9 figures. Code & Dataset: https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep
>
> **摘要:** Metaphorical comprehension in images remains a critical challenge for AI systems, as existing models struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. While multimodal large language models (MLLMs) excel in basic Visual Question Answer (VQA) tasks, they struggle with a fundamental limitation on image implication tasks: contextual gaps that obscure the relationships between different visual elements and their abstract meanings. Inspired by the human cognitive process, we propose Let Androids Dream (LAD), a novel framework for image implication understanding and reasoning. LAD addresses contextual missing through the three-stage framework: (1) Perception: converting visual information into rich and multi-level textual representations, (2) Search: iteratively searching and integrating cross-domain knowledge to resolve ambiguity, and (3) Reasoning: generating context-alignment image implication via explicit reasoning. Our framework with the lightweight GPT-4o-mini model achieves SOTA performance compared to 15+ MLLMs on English image implication benchmark and a huge improvement on Chinese benchmark, performing comparable with the GPT-4o model on Multiple-Choice Question (MCQ) and outperforms 36.7% on Open-Style Question (OSQ). Additionally, our work provides new insights into how AI can more effectively interpret image implications, advancing the field of vision-language reasoning and human-AI interaction. Our project is publicly available at https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep.
>
---
#### [new 122] Swin Transformer for Robust CGI Images Detection: Intra- and Inter-Dataset Analysis across Multiple Color Spaces
- **分类: cs.CV**

- **简介: 该论文提出基于Swin Transformer的模型，解决跨RGB/YCbCr/HSV颜色空间区分CGI与真实图像的挑战。通过多数据集的intra-/inter-dataset测试、数据增强及t-SNE分析，验证模型在特征捕捉和领域泛化上的优势，结果优于VGG-19和ResNet-50，证明其在CGI检测中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.16253v1](http://arxiv.org/pdf/2505.16253v1)**

> **作者:** Preeti Mehta; Aman Sagar; Suchi Kumari
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2409.04734
>
> **摘要:** This study aims to address the growing challenge of distinguishing computer-generated imagery (CGI) from authentic digital images across three different color spaces; RGB, YCbCr, and HSV. Given the limitations of existing classification methods in handling the complexity and variability of CGI, this research proposes a Swin Transformer based model for accurate differentiation between natural and synthetic images. The proposed model leverages the Swin Transformer's hierarchical architecture to capture local and global features for distinguishing CGI from natural images. Its performance was assessed through intra- and inter-dataset testing across three datasets: CiFAKE, JSSSTU, and Columbia. The model was evaluated individually on each dataset (D1, D2, D3) and on the combined datasets (D1+D2+D3) to test its robustness and domain generalization. To address dataset imbalance, data augmentation techniques were applied. Additionally, t-SNE visualization was used to demonstrate the feature separability achieved by the Swin Transformer across the selected color spaces. The model's performance was tested across all color schemes, with the RGB color scheme yielding the highest accuracy for each dataset. As a result, RGB was selected for domain generalization analysis and compared with other CNN-based models, VGG-19 and ResNet-50. The comparative results demonstrate the proposed model's effectiveness in detecting CGI, highlighting its robustness and reliability in both intra-dataset and inter-dataset evaluations. The findings of this study highlight the Swin Transformer model's potential as an advanced tool for digital image forensics, particularly in distinguishing CGI from natural images. The model's strong performance indicates its capability for domain generalization, making it a valuable asset in scenarios requiring precise and reliable image classification.
>
---
#### [new 123] Auto-nnU-Net: Towards Automated Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Auto-nnU-Net改进医疗影像分割自动化，针对nnU-Net超参数固定与设计启发式问题，引入超参数优化、神经架构搜索及层次化NAS，并设计Regularized PriorBand平衡资源与精度。实验显示其在6/10数据集提升性能，其余持平，且资源需求合理。**

- **链接: [http://arxiv.org/pdf/2505.16561v1](http://arxiv.org/pdf/2505.16561v1)**

> **作者:** Jannis Becktepe; Leona Hennig; Steffen Oeltze-Jafra; Marius Lindauer
>
> **备注:** 31 pages, 19 figures. Accepted for publication at AutoML 2025
>
> **摘要:** Medical Image Segmentation (MIS) includes diverse tasks, from bone to organ segmentation, each with its own challenges in finding the best segmentation model. The state-of-the-art AutoML-related MIS-framework nnU-Net automates many aspects of model configuration but remains constrained by fixed hyperparameters and heuristic design choices. As a full-AutoML framework for MIS, we propose Auto-nnU-Net, a novel nnU-Net variant enabling hyperparameter optimization (HPO), neural architecture search (NAS), and hierarchical NAS (HNAS). Additionally, we propose Regularized PriorBand to balance model accuracy with the computational resources required for training, addressing the resource constraints often faced in real-world medical settings that limit the feasibility of extensive training procedures. We evaluate our approach across diverse MIS datasets from the well-established Medical Segmentation Decathlon, analyzing the impact of AutoML techniques on segmentation performance, computational efficiency, and model design choices. The results demonstrate that our AutoML approach substantially improves the segmentation performance of nnU-Net on 6 out of 10 datasets and is on par on the other datasets while maintaining practical resource requirements. Our code is available at https://github.com/LUH-AI/AutonnUNet.
>
---
#### [new 124] Detailed Evaluation of Modern Machine Learning Approaches for Optic Plastics Sorting
- **分类: cs.CV; 68T45; I.4.9; I.4.6**

- **简介: 论文评估现代机器学习方法在光学塑料分拣中的效果，旨在解决实际场景中因物理属性依赖导致的分拣准确性不足问题。研究构建2万余张图像数据集，对比两阶段与单阶段检测模型，结合Grad-CAM和混淆矩阵分析，发现现有方法受限于颜色和形状依赖，难以满足实际需求。**

- **链接: [http://arxiv.org/pdf/2505.16513v1](http://arxiv.org/pdf/2505.16513v1)**

> **作者:** Vaishali Maheshkar; Aadarsh Anantha Ramakrishnan; Charuvahan Adhivarahan; Karthik Dantu
>
> **备注:** Accepted at the 2024 REMADE Circular Economy Tech Summit and Conference, https://remadeinstitute.org/2024-conference/
>
> **摘要:** According to the EPA, only 25% of waste is recycled, and just 60% of U.S. municipalities offer curbside recycling. Plastics fare worse, with a recycling rate of only 8%; an additional 16% is incinerated, while the remaining 76% ends up in landfills. The low plastic recycling rate stems from contamination, poor economic incentives, and technical difficulties, making efficient recycling a challenge. To improve recovery, automated sorting plays a critical role. Companies like AMP Robotics and Greyparrot utilize optical systems for sorting, while Materials Recovery Facilities (MRFs) employ Near-Infrared (NIR) sensors to detect plastic types. Modern optical sorting uses advances in computer vision such as object recognition and instance segmentation, powered by machine learning. Two-stage detectors like Mask R-CNN use region proposals and classification with deep backbones like ResNet. Single-stage detectors like YOLO handle detection in one pass, trading some accuracy for speed. While such methods excel under ideal conditions with a large volume of labeled training data, challenges arise in realistic scenarios, emphasizing the need to further examine the efficacy of optic detection for automated sorting. In this study, we compiled novel datasets totaling 20,000+ images from varied sources. Using both public and custom machine learning pipelines, we assessed the capabilities and limitations of optical recognition for sorting. Grad-CAM, saliency maps, and confusion matrices were employed to interpret model behavior. We perform this analysis on our custom trained models from the compiled datasets. To conclude, our findings are that optic recognition methods have limited success in accurate sorting of real-world plastics at MRFs, primarily because they rely on physical properties such as color and shape.
>
---
#### [new 125] A Causal Approach to Mitigate Modality Preference Bias in Medical Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文针对医学视觉问答任务，解决模态偏好偏差问题（模型过度依赖问题文本忽视图像）。提出MedCFVQA模型，利用因果图消除推理偏差，并重构数据集改变问答先验依赖，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.16209v1](http://arxiv.org/pdf/2505.16209v1)**

> **作者:** Shuchang Ye; Usman Naseem; Mingyuan Meng; Dagan Feng; Jinman Kim
>
> **摘要:** Medical Visual Question Answering (MedVQA) is crucial for enhancing the efficiency of clinical diagnosis by providing accurate and timely responses to clinicians' inquiries regarding medical images. Existing MedVQA models suffered from modality preference bias, where predictions are heavily dominated by one modality while overlooking the other (in MedVQA, usually questions dominate the answer but images are overlooked), thereby failing to learn multimodal knowledge. To overcome the modality preference bias, we proposed a Medical CounterFactual VQA (MedCFVQA) model, which trains with bias and leverages causal graphs to eliminate the modality preference bias during inference. Existing MedVQA datasets exhibit substantial prior dependencies between questions and answers, which results in acceptable performance even if the model significantly suffers from the modality preference bias. To address this issue, we reconstructed new datasets by leveraging existing MedVQA datasets and Changed their P3rior dependencies (CP) between questions and their answers in the training and test set. Extensive experiments demonstrate that MedCFVQA significantly outperforms its non-causal counterpart on both SLAKE, RadVQA and SLAKE-CP, RadVQA-CP datasets.
>
---
#### [new 126] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于在线自由视点视频（FVV）重建任务，针对现有方法因点式建模导致存储需求过高的问题，提出ComGS框架。通过识别运动区域关键点、自适应运动传播及误差修正策略，利用运动一致性减少传输数据，实现159倍存储压缩，兼顾重建质量和效率。**

- **链接: [http://arxiv.org/pdf/2505.16533v1](http://arxiv.org/pdf/2505.16533v1)**

> **作者:** Jiacong Chen; Qingyu Mao; Youneng Bao; Xiandong Meng; Fanyang Meng; Ronggang Wang; Yongsheng Liang
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a high-fidelity and efficient paradigm for online free-viewpoint video (FVV) reconstruction, offering viewers rapid responsiveness and immersive experiences. However, existing online methods face challenge in prohibitive storage requirements primarily due to point-wise modeling that fails to exploit the motion properties. To address this limitation, we propose a novel Compact Gaussian Streaming (ComGS) framework, leveraging the locality and consistency of motion in dynamic scene, that models object-consistent Gaussian point motion through keypoint-driven motion representation. By transmitting only the keypoint attributes, this framework provides a more storage-efficient solution. Specifically, we first identify a sparse set of motion-sensitive keypoints localized within motion regions using a viewspace gradient difference strategy. Equipped with these keypoints, we propose an adaptive motion-driven mechanism that predicts a spatial influence field for propagating keypoint motion to neighboring Gaussian points with similar motion. Moreover, ComGS adopts an error-aware correction strategy for key frame reconstruction that selectively refines erroneous regions and mitigates error accumulation without unnecessary overhead. Overall, ComGS achieves a remarkable storage reduction of over 159 X compared to 3DGStream and 14 X compared to the SOTA method QUEEN, while maintaining competitive visual fidelity and rendering speed. Our code will be released.
>
---
#### [new 127] How Do Large Vision-Language Models See Text in Image? Unveiling the Distinctive Role of OCR Heads
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型（LVLMs）内部机制分析任务，旨在探究LVLMs如何定位和理解图像中的文本。通过识别并研究"OCR头"（负责文本识别的特定注意力头），发现其激活模式与特性不同于普通检索头，并验证了优化OCR头（如调整权重）可提升性能，推动模型可解释性研究。**

- **链接: [http://arxiv.org/pdf/2505.15865v1](http://arxiv.org/pdf/2505.15865v1)**

> **作者:** Ingeol Baek; Hwan Chang; Sunghyun Ryu; Hwanhee Lee
>
> **摘要:** Despite significant advancements in Large Vision Language Models (LVLMs), a gap remains, particularly regarding their interpretability and how they locate and interpret textual information within images. In this paper, we explore various LVLMs to identify the specific heads responsible for recognizing text from images, which we term the Optical Character Recognition Head (OCR Head). Our findings regarding these heads are as follows: (1) Less Sparse: Unlike previous retrieval heads, a large number of heads are activated to extract textual information from images. (2) Qualitatively Distinct: OCR heads possess properties that differ significantly from general retrieval heads, exhibiting low similarity in their characteristics. (3) Statically Activated: The frequency of activation for these heads closely aligns with their OCR scores. We validate our findings in downstream tasks by applying Chain-of-Thought (CoT) to both OCR and conventional retrieval heads and by masking these heads. We also demonstrate that redistributing sink-token values within the OCR heads improves performance. These insights provide a deeper understanding of the internal mechanisms LVLMs employ in processing embedded textual information in images.
>
---
#### [new 128] Training-Free Efficient Video Generation via Dynamic Token Carving
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散模型计算效率低的问题。提出Jenga方法，通过动态注意力裁剪（基于3D空间填充曲线选择token交互）和渐进式分辨率生成策略，在保持生成质量的同时加速推理（8.83倍提速），且无需模型重新训练。**

- **链接: [http://arxiv.org/pdf/2505.16864v1](http://arxiv.org/pdf/2505.16864v1)**

> **作者:** Yuechen Zhang; Jinbo Xing; Bin Xia; Shaoteng Liu; Bohao Peng; Xin Tao; Pengfei Wan; Eric Lo; Jiaya Jia
>
> **备注:** Project Page: https://julianjuaner.github.io/projects/jenga/ , 24 pages
>
> **摘要:** Despite the remarkable generation quality of video Diffusion Transformer (DiT) models, their practical deployment is severely hindered by extensive computational requirements. This inefficiency stems from two key challenges: the quadratic complexity of self-attention with respect to token length and the multi-step nature of diffusion models. To address these limitations, we present Jenga, a novel inference pipeline that combines dynamic attention carving with progressive resolution generation. Our approach leverages two key insights: (1) early denoising steps do not require high-resolution latents, and (2) later steps do not require dense attention. Jenga introduces a block-wise attention mechanism that dynamically selects relevant token interactions using 3D space-filling curves, alongside a progressive resolution strategy that gradually increases latent resolution during generation. Experimental results demonstrate that Jenga achieves substantial speedups across multiple state-of-the-art video diffusion models while maintaining comparable generation quality (8.83$\times$ speedup with 0.01\% performance drop on VBench). As a plug-and-play solution, Jenga enables practical, high-quality video generation on modern hardware by reducing inference time from minutes to seconds -- without requiring model retraining. Code: https://github.com/dvlab-research/Jenga
>
---
#### [new 129] M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video Conversion
- **分类: cs.CV**

- **简介: 该论文提出M2SVid模型，解决单目视频转立体视频任务。针对深度重投影生成的右视角图像中空洞区域（disocclusion）问题，通过扩展Stable Video Diffusion模型，利用左视频、变形右视频及空洞掩码作为条件输入，并改进注意力机制以融合邻近帧信息，实现端到端生成高质量右视角视频，精度与速度优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16565v1](http://arxiv.org/pdf/2505.16565v1)**

> **作者:** Nina Shvetsova; Goutam Bhat; Prune Truong; Hilde Kuehne; Federico Tombari
>
> **摘要:** We tackle the problem of monocular-to-stereo video conversion and propose a novel architecture for inpainting and refinement of the warped right view obtained by depth-based reprojection of the input left view. We extend the Stable Video Diffusion (SVD) model to utilize the input left video, the warped right video, and the disocclusion masks as conditioning input to generate a high-quality right camera view. In order to effectively exploit information from neighboring frames for inpainting, we modify the attention layers in SVD to compute full attention for discoccluded pixels. Our model is trained to generate the right view video in an end-to-end manner by minimizing image space losses to ensure high-quality generation. Our approach outperforms previous state-of-the-art methods, obtaining an average rank of 1.43 among the 4 compared methods in a user study, while being 6x faster than the second placed method.
>
---
#### [new 130] BadDepth: Backdoor Attacks Against Monocular Depth Estimation in the Physical World
- **分类: cs.CV**

- **简介: 该论文研究针对单目深度估计（MDE）模型的后门攻击任务。解决现有方法无法处理深度图标签的问题，提出BadDepth：通过分割目标对象并修改其深度、补全周围区域构建中毒数据集，并采用数字-物理域转换增强提升物理场景鲁棒性，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.16154v1](http://arxiv.org/pdf/2505.16154v1)**

> **作者:** Ji Guo; Long Zhou; Zhijin Wang; Jiaming He; Qiyang Song; Aiguo Chen; Wenbo Jiang
>
> **摘要:** In recent years, deep learning-based Monocular Depth Estimation (MDE) models have been widely applied in fields such as autonomous driving and robotics. However, their vulnerability to backdoor attacks remains unexplored. To fill the gap in this area, we conduct a comprehensive investigation of backdoor attacks against MDE models. Typically, existing backdoor attack methods can not be applied to MDE models. This is because the label used in MDE is in the form of a depth map. To address this, we propose BadDepth, the first backdoor attack targeting MDE models. BadDepth overcomes this limitation by selectively manipulating the target object's depth using an image segmentation model and restoring the surrounding areas via depth completion, thereby generating poisoned datasets for object-level backdoor attacks. To improve robustness in physical world scenarios, we further introduce digital-to-physical augmentation to adapt to the domain gap between the physical world and the digital domain. Extensive experiments on multiple models validate the effectiveness of BadDepth in both the digital domain and the physical world, without being affected by environmental factors.
>
---
#### [new 131] MAGIC: Motion-Aware Generative Inference via Confidence-Guided LLM
- **分类: cs.CV**

- **简介: 该论文提出MAGIC框架，解决视频生成中物理不一致问题。通过结合预训练扩散模型与LLM迭代推理，利用置信度反馈优化物理动态，并引入可微MPM模拟器，从单图生成物理合理、时序连贯的动态视频，无需训练或大量数据。**

- **链接: [http://arxiv.org/pdf/2505.16456v1](http://arxiv.org/pdf/2505.16456v1)**

> **作者:** Siwei Meng; Yawei Luo; Ping Liu
>
> **摘要:** Recent advances in static 3D generation have intensified the demand for physically consistent dynamic 3D content. However, existing video generation models, including diffusion-based methods, often prioritize visual realism while neglecting physical plausibility, resulting in implausible object dynamics. Prior approaches for physics-aware dynamic generation typically rely on large-scale annotated datasets or extensive model fine-tuning, which imposes significant computational and data collection burdens and limits scalability across scenarios. To address these challenges, we present MAGIC, a training-free framework for single-image physical property inference and dynamic generation, integrating pretrained image-to-video diffusion models with iterative LLM-based reasoning. Our framework generates motion-rich videos from a static image and closes the visual-to-physical gap through a confidence-driven LLM feedback loop that adaptively steers the diffusion model toward physics-relevant motion. To translate visual dynamics into controllable physical behavior, we further introduce a differentiable MPM simulator operating directly on 3D Gaussians reconstructed from the single image, enabling physically grounded, simulation-ready outputs without any supervision or model tuning. Experiments show that MAGIC outperforms existing physics-aware generative methods in inference accuracy and achieves greater temporal coherence than state-of-the-art video diffusion models.
>
---
#### [new 132] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出DOVE模型，解决视频超分辨率（VSR）中扩散模型推理速度慢的问题。通过微调预训练视频扩散模型，结合分阶段的latent-pixel训练策略和自建HQ-VSR数据集，实现单步高效推理，性能媲美多步方法，速度提升28倍。**

- **链接: [http://arxiv.org/pdf/2505.16239v1](http://arxiv.org/pdf/2505.16239v1)**

> **作者:** Zheng Chen; Zichen Zou; Kewei Zhang; Xiongfei Su; Xin Yuan; Yong Guo; Yulun Zhang
>
> **备注:** Code is available at: https://github.com/zhengchen1999/DOVE
>
> **摘要:** Diffusion models have demonstrated promising performance in real-world video super-resolution (VSR). However, the dozens of sampling steps they require, make inference extremely slow. Sampling acceleration techniques, particularly single-step, provide a potential solution. Nonetheless, achieving one step in VSR remains challenging, due to the high training overhead on video data and stringent fidelity demands. To tackle the above issues, we propose DOVE, an efficient one-step diffusion model for real-world VSR. DOVE is obtained by fine-tuning a pretrained video diffusion model (*i.e.*, CogVideoX). To effectively train DOVE, we introduce the latent-pixel training strategy. The strategy employs a two-stage scheme to gradually adapt the model to the video super-resolution task. Meanwhile, we design a video processing pipeline to construct a high-quality dataset tailored for VSR, termed HQ-VSR. Fine-tuning on this dataset further enhances the restoration capability of DOVE. Extensive experiments show that DOVE exhibits comparable or superior performance to multi-step diffusion-based VSR methods. It also offers outstanding inference efficiency, achieving up to a **28$\times$** speed-up over existing methods such as MGLD-VSR. Code is available at: https://github.com/zhengchen1999/DOVE.
>
---
#### [new 133] TRAIL: Transferable Robust Adversarial Images via Latent diffusion
- **分类: cs.CV**

- **简介: 论文提出TRAIL框架，解决对抗样本跨模型迁移性差的问题。通过测试时动态调整扩散模型参数，结合对抗目标与感知约束，生成分布对齐的对抗样本，提升黑盒攻击效果。任务为黑盒对抗攻击，方法为参数适应与分布对齐。**

- **链接: [http://arxiv.org/pdf/2505.16166v1](http://arxiv.org/pdf/2505.16166v1)**

> **作者:** Yuhao Xue; Zhifei Zhang; Xinyang Jiang; Yifei Shen; Junyao Gao; Wentao Gu; Jiale Zhao; Miaojing Shi; Cairong Zhao
>
> **摘要:** Adversarial attacks exploiting unrestricted natural perturbations present severe security risks to deep learning systems, yet their transferability across models remains limited due to distribution mismatches between generated adversarial features and real-world data. While recent works utilize pre-trained diffusion models as adversarial priors, they still encounter challenges due to the distribution shift between the distribution of ideal adversarial samples and the natural image distribution learned by the diffusion model. To address the challenge, we propose Transferable Robust Adversarial Images via Latent Diffusion (TRAIL), a test-time adaptation framework that enables the model to generate images from a distribution of images with adversarial features and closely resembles the target images. To mitigate the distribution shift, during attacks, TRAIL updates the diffusion U-Net's weights by combining adversarial objectives (to mislead victim models) and perceptual constraints (to preserve image realism). The adapted model then generates adversarial samples through iterative noise injection and denoising guided by these objectives. Experiments demonstrate that TRAIL significantly outperforms state-of-the-art methods in cross-model attack transferability, validating that distribution-aligned adversarial feature synthesis is critical for practical black-box attacks.
>
---
#### [new 134] From Evaluation to Defense: Advancing Safety in Video Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦视频大语言模型安全任务，针对其系统性风险问题，构建首个大规模多文化基准VSB-77k（7.7万视频-查询对），揭示视频模态使安全性能下降42.3%。提出VideoSafety-R1框架，通过报警标记微调与动态策略优化双阶段设计，使安全性能提升65.1%，有效增强多模态防御能力。**

- **链接: [http://arxiv.org/pdf/2505.16643v1](http://arxiv.org/pdf/2505.16643v1)**

> **作者:** Yiwei Sun; Peiqi Jiang; Chuanbin Liu; Luohao Lin; Zhiying Lu; Hongtao Xie
>
> **备注:** 49 pages, 12 figures, 17 tables
>
> **摘要:** While the safety risks of image-based large language models have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and spans 19 principal risk categories across 10 language communities. \textit{We reveal that integrating video modality degrades safety performance by an average of 42.3\%, exposing systemic risks in multimodal attack exploitation.} To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through two innovations: (1) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from passive harm recognition to active reasoning. The resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. \textit{Our codes are available in the supplementary materials.} \textcolor{red}{Warning: This paper contains examples of harmful language and videos, and reader discretion is recommended.}
>
---
#### [new 135] Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态视觉任务，解决MLLM在多帧空间理解不足的问题。提出Multi-SpatialMLLM框架，整合深度感知、视觉对应与动态感知，构建2700万样本的MultiSPA数据集及基准测试，提升多帧推理能力，支持机器人等应用。**

- **链接: [http://arxiv.org/pdf/2505.17015v1](http://arxiv.org/pdf/2505.17015v1)**

> **作者:** Runsen Xu; Weiyao Wang; Hao Tang; Xingyu Chen; Xiaodong Wang; Fu-Jen Chu; Dahua Lin; Matt Feiszli; Kevin J. Liang
>
> **备注:** 24 pages. An MLLM, dataset, and benchmark for multi-frame spatial understanding. Project page: https://runsenxu.com/projects/Multi-SpatialMLLM
>
> **摘要:** Multi-modal large language models (MLLMs) have rapidly advanced in visual tasks, yet their spatial understanding remains limited to single images, leaving them ill-suited for robotics and other real-world applications that require multi-frame reasoning. In this paper, we propose a framework to equip MLLMs with robust multi-frame spatial understanding by integrating depth perception, visual correspondence, and dynamic perception. Central to our approach is the MultiSPA dataset, a novel, large-scale collection of more than 27 million samples spanning diverse 3D and 4D scenes. Alongside MultiSPA, we introduce a comprehensive benchmark that tests a wide spectrum of spatial tasks under uniform metrics. Our resulting model, Multi-SpatialMLLM, achieves significant gains over baselines and proprietary systems, demonstrating scalable, generalizable multi-frame reasoning. We further observe multi-task benefits and early indications of emergent capabilities in challenging scenarios, and showcase how our model can serve as a multi-frame reward annotator for robotics.
>
---
#### [new 136] ViQAgent: Zero-Shot Video Question Answering via Agent with Open-Vocabulary Grounding Validation
- **分类: cs.CV; cs.CL; I.4.8**

- **简介: 该论文提出ViQAgent，用于零样本视频问答任务，解决长期物体追踪与推理决策对齐问题。结合思维链框架与YOLO-World增强视觉接地，通过时间帧交叉验证提升准确性，实现视频理解新SOTA，支持多领域应用。**

- **链接: [http://arxiv.org/pdf/2505.15928v1](http://arxiv.org/pdf/2505.15928v1)**

> **作者:** Tony Montes; Fernando Lozano
>
> **摘要:** Recent advancements in Video Question Answering (VideoQA) have introduced LLM-based agents, modular frameworks, and procedural solutions, yielding promising results. These systems use dynamic agents and memory-based mechanisms to break down complex tasks and refine answers. However, significant improvements remain in tracking objects for grounding over time and decision-making based on reasoning to better align object references with language model outputs, as newer models get better at both tasks. This work presents an LLM-brained agent for zero-shot Video Question Answering (VideoQA) that combines a Chain-of-Thought framework with grounding reasoning alongside YOLO-World to enhance object tracking and alignment. This approach establishes a new state-of-the-art in VideoQA and Video Understanding, showing enhanced performance on NExT-QA, iVQA, and ActivityNet-QA benchmarks. Our framework also enables cross-checking of grounding timeframes, improving accuracy and providing valuable support for verification and increased output reliability across multiple video domains. The code is available at https://github.com/t-montes/viqagent.
>
---
#### [new 137] DualComp: End-to-End Learning of a Unified Dual-Modality Lossless Compressor
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文提出DualComp，首个轻量级端到端双模态无损压缩器，解决单模态需多模型及多模态模型复杂度高的问题。通过统一tokenization、切换式上下文学习、路由专家混合及重参数化训练，实现高效压缩，参数少但性能媲美先进方法，在Kodak数据集上提升9%，模型大小仅1.2%。**

- **链接: [http://arxiv.org/pdf/2505.16256v1](http://arxiv.org/pdf/2505.16256v1)**

> **作者:** Yan Zhao; Zhengxue Cheng; Junxuan Zhang; Qunshan Gu; Qi Wang; Li Song
>
> **备注:** 18 pages, 11 figures, 7 tables
>
> **摘要:** Most learning-based lossless compressors are designed for a single modality, requiring separate models for multi-modal data and lacking flexibility. However, different modalities vary significantly in format and statistical properties, making it ineffective to use compressors that lack modality-specific adaptations. While multi-modal large language models (MLLMs) offer a potential solution for modality-unified compression, their excessive complexity hinders practical deployment. To address these challenges, we focus on the two most common modalities, image and text, and propose DualComp, the first unified and lightweight learning-based dual-modality lossless compressor. Built on a lightweight backbone, DualComp incorporates three key structural enhancements to handle modality heterogeneity: modality-unified tokenization, modality-switching contextual learning, and modality-routing mixture-of-experts. A reparameterization training strategy is also used to boost compression performance. DualComp integrates both modality-specific and shared parameters for efficient parameter utilization, enabling near real-time inference (200KB/s) on desktop CPUs. With much fewer parameters, DualComp achieves compression performance on par with the SOTA LLM-based methods for both text and image datasets. Its simplified single-modality variant surpasses the previous best image compressor on the Kodak dataset by about 9% using just 1.2% of the model size.
>
---
#### [new 138] Mitigating Overfitting in Medical Imaging: Self-Supervised Pretraining vs. ImageNet Transfer Learning for Dermatological Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文比较自监督预训练与ImageNet迁移学习在皮肤影像分类中的效果，旨在解决医学影像领域过拟合问题。提出基于VAE的无监督框架，从皮肤科数据集从头训练特征提取器，对比发现自监督模型过拟合更少、泛化性更强，凸显领域特异性特征提取的重要性。**

- **链接: [http://arxiv.org/pdf/2505.16773v1](http://arxiv.org/pdf/2505.16773v1)**

> **作者:** Iván Matas; Carmen Serrano; Miguel Nogales; David Moreno; Lara Ferrándiz; Teresa Ojeda; Begoña Acha
>
> **备注:** 6 pages, 2 tables, 2 figures
>
> **摘要:** Deep learning has transformed computer vision but relies heavily on large labeled datasets and computational resources. Transfer learning, particularly fine-tuning pretrained models, offers a practical alternative; however, models pretrained on natural image datasets such as ImageNet may fail to capture domain-specific characteristics in medical imaging. This study introduces an unsupervised learning framework that extracts high-value dermatological features instead of relying solely on ImageNet-based pretraining. We employ a Variational Autoencoder (VAE) trained from scratch on a proprietary dermatological dataset, allowing the model to learn a structured and clinically relevant latent space. This self-supervised feature extractor is then compared to an ImageNet-pretrained backbone under identical classification conditions, highlighting the trade-offs between general-purpose and domain-specific pretraining. Our results reveal distinct learning patterns. The self-supervised model achieves a final validation loss of 0.110 (-33.33%), while the ImageNet-pretrained model stagnates at 0.100 (-16.67%), indicating overfitting. Accuracy trends confirm this: the self-supervised model improves from 45% to 65% (+44.44%) with a near-zero overfitting gap, whereas the ImageNet-pretrained model reaches 87% (+50.00%) but plateaus at 75% (+19.05%), with its overfitting gap increasing to +0.060. These findings suggest that while ImageNet pretraining accelerates convergence, it also amplifies overfitting on non-clinically relevant features. In contrast, self-supervised learning achieves steady improvements, stronger generalization, and superior adaptability, underscoring the importance of domain-specific feature extraction in medical imaging.
>
---
#### [new 139] SoccerChat: Integrating Multimodal Data for Enhanced Soccer Game Understanding
- **分类: cs.CV; cs.AI; 68T45, 68T50; I.2.10; I.2.7; H.5.2**

- **简介: 该论文属于足球视频分析任务，旨在解决传统方法依赖单一数据源导致的语境理解不足问题。提出SoccerChat框架，整合视觉（含球衣颜色标注）与文本（语音转录）数据，通过多模态训练提升赛事解析与裁判决策准确性，验证于动作分类及判罚任务，强调多模态融合对体育AI分析的重要性。**

- **链接: [http://arxiv.org/pdf/2505.16630v1](http://arxiv.org/pdf/2505.16630v1)**

> **作者:** Sushant Gautam; Cise Midoglu; Vajira Thambawita; Michael A. Riegler; Pål Halvorsen; Mubarak Shah
>
> **摘要:** The integration of artificial intelligence in sports analytics has transformed soccer video understanding, enabling real-time, automated insights into complex game dynamics. Traditional approaches rely on isolated data streams, limiting their effectiveness in capturing the full context of a match. To address this, we introduce SoccerChat, a multimodal conversational AI framework that integrates visual and textual data for enhanced soccer video comprehension. Leveraging the extensive SoccerNet dataset, enriched with jersey color annotations and automatic speech recognition (ASR) transcripts, SoccerChat is fine-tuned on a structured video instruction dataset to facilitate accurate game understanding, event classification, and referee decision making. We benchmark SoccerChat on action classification and referee decision-making tasks, demonstrating its performance in general soccer event comprehension while maintaining competitive accuracy in referee decision making. Our findings highlight the importance of multimodal integration in advancing soccer analytics, paving the way for more interactive and explainable AI-driven sports analysis. https://github.com/simula/SoccerChat
>
---
#### [new 140] CoNav: Collaborative Cross-Modal Reasoning for Embodied Navigation
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于具身导航任务，旨在解决多模态融合中的数据稀缺与模态冲突问题。提出CoNav框架，通过3D-文本模型向图像-文本导航代理提供空间语义知识，实现跨模态信念对齐，经轻量微调提升导航精度与路径效率，在多个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.16663v1](http://arxiv.org/pdf/2505.16663v1)**

> **作者:** Haihong Hao; Mingfei Han; Changlin Li; Zhihui Li; Xiaojun Chang
>
> **摘要:** Embodied navigation demands comprehensive scene understanding and precise spatial reasoning. While image-text models excel at interpreting pixel-level color and lighting cues, 3D-text models capture volumetric structure and spatial relationships. However, unified fusion approaches that jointly fuse 2D images, 3D point clouds, and textual instructions face challenges in limited availability of triple-modality data and difficulty resolving conflicting beliefs among modalities. In this work, we introduce CoNav, a collaborative cross-modal reasoning framework where a pretrained 3D-text model explicitly guides an image-text navigation agent by providing structured spatial-semantic knowledge to resolve ambiguities during navigation. Specifically, we introduce Cross-Modal Belief Alignment, which operationalizes this cross-modal guidance by simply sharing textual hypotheses from the 3D-text model to the navigation agent. Through lightweight fine-tuning on a small 2D-3D-text corpus, the navigation agent learns to integrate visual cues with spatial-semantic knowledge derived from the 3D-text model, enabling effective reasoning in embodied navigation. CoNav achieves significant improvements on four standard embodied navigation benchmarks (R2R, CVDN, REVERIE, SOON) and two spatial reasoning benchmarks (ScanQA, SQA3D). Moreover, under close navigation Success Rate, CoNav often generates shorter paths compared to other methods (as measured by SPL), showcasing the potential and challenges of fusing data from different modalities in embodied navigation. Project Page: https://oceanhao.github.io/CoNav/
>
---
#### [new 141] OpenSeg-R: Improving Open-Vocabulary Segmentation via Step-by-Step Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文提出OpenSeg-R框架，针对开放词汇分割任务中现有方法缺乏推理导致相似类别区分困难的问题，通过多模态模型进行分步视觉推理生成结构化提示，提升分割精度和可解释性。**

- **链接: [http://arxiv.org/pdf/2505.16974v1](http://arxiv.org/pdf/2505.16974v1)**

> **作者:** Zongyan Han; Jiale Cao; Shuo Chen; Tong Wang; Jorma Laaksonen; Rao Muhammad Anwer
>
> **摘要:** Open-Vocabulary Segmentation (OVS) has drawn increasing attention for its capacity to generalize segmentation beyond predefined categories. However, existing methods typically predict segmentation masks with simple forward inference, lacking explicit reasoning and interpretability. This makes it challenging for OVS model to distinguish similar categories in open-world settings due to the lack of contextual understanding and discriminative visual cues. To address this limitation, we propose a step-by-step visual reasoning framework for open-vocabulary segmentation, named OpenSeg-R. The proposed OpenSeg-R leverages Large Multimodal Models (LMMs) to perform hierarchical visual reasoning before segmentation. Specifically, we generate both generic and image-specific reasoning for each image, forming structured triplets that explain the visual reason for objects in a coarse-to-fine manner. Based on these reasoning steps, we can compose detailed description prompts, and feed them to the segmentor to produce more accurate segmentation masks. To the best of our knowledge, OpenSeg-R is the first framework to introduce explicit step-by-step visual reasoning into OVS. Experimental results demonstrate that OpenSeg-R significantly outperforms state-of-the-art methods on open-vocabulary semantic segmentation across five benchmark datasets. Moreover, it achieves consistent gains across all metrics on open-vocabulary panoptic segmentation. Qualitative results further highlight the effectiveness of our reasoning-guided framework in improving both segmentation precision and interpretability. Our code is publicly available at https://github.com/Hanzy1996/OpenSeg-R.
>
---
#### [new 142] FPQVAR: Floating Point Quantization for Visual Autoregressive Model with FPGA Hardware Co-design
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉自回归模型（VAR）的轻量化部署任务，解决其参数量大、计算成本高导致的边缘设备部署难题。提出FPQVAR框架，结合算法（双格式量化、群组哈达玛变换）与FPGA硬件（低比特浮点运算单元及加速器）优化，在4/6比特量化下显著提升图像质量（FID从10.83降至3.58）与能效（较整数加速器提升3.6倍），实现1.1图片/秒的吞吐量。**

- **链接: [http://arxiv.org/pdf/2505.16335v1](http://arxiv.org/pdf/2505.16335v1)**

> **作者:** Renjie Wei; Songqiang Xu; Qingyu Guo; Meng Li
>
> **摘要:** Visual autoregressive (VAR) modeling has marked a paradigm shift in image generation from next-token prediction to next-scale prediction. VAR predicts a set of tokens at each step from coarse to fine scale, leading to better image quality and faster inference speed compared to existing diffusion models. However, the large parameter size and computation cost hinder its deployment on edge devices. To reduce the memory and computation cost, we propose FPQVAR, an efficient post-training floating-point (FP) quantization framework for VAR featuring algorithm and hardware co-design. At the algorithm level, we first identify the challenges of quantizing VAR. To address them, we propose Dual Format Quantization for the highly imbalanced input activation. We further propose Group-wise Hadamard Transformation and GHT-Aware Learnable Transformation to address the time-varying outlier channels. At the hardware level, we design the first low-bit FP quantizer and multiplier with lookup tables on FPGA and propose the first FPGA-based VAR accelerator featuring low-bit FP computation and an elaborate two-level pipeline. Extensive experiments show that compared to the state-of-the-art quantization method, our proposed FPQVAR significantly improves Fr\'echet Inception Distance (FID) from 10.83 to 3.58, Inception Score (IS) from 175.9 to 241.5 under 4-bit quantization. FPQVAR also significantly improves the performance of 6-bit quantized VAR, bringing it on par with the FP16 model. Our accelerator on AMD-Xilinx VCK190 FPGA achieves a throughput of 1.1 image/s, which is 3.1x higher than the integer-based accelerator. It also demonstrates 3.6x and 2.8x higher energy efficiency compared to the integer-based accelerator and GPU baseline, respectively.
>
---
#### [new 143] Action2Dialogue: Generating Character-Centric Narratives from Scene-Level Prompts
- **分类: cs.CV**

- **简介: 该论文提出Action2Dialogue框架，属于多模态叙事生成任务。解决场景级视频生成缺乏角色驱动对话及上下文一致性的难题。工作包括：通过视觉语言模型提取场景特征，结合结构化提示引导语言模型生成角色对话，并用递归叙事库维护对话历史，最终合成语音实现视听叙事。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16819v1](http://arxiv.org/pdf/2505.16819v1)**

> **作者:** Taewon Kang; Ming C. Lin
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Recent advances in scene-based video generation have enabled systems to synthesize coherent visual narratives from structured prompts. However, a crucial dimension of storytelling -- character-driven dialogue and speech -- remains underexplored. In this paper, we present a modular pipeline that transforms action-level prompts into visually and auditorily grounded narrative dialogue, enriching visual storytelling with natural voice and character expression. Our method takes as input a pair of prompts per scene, where the first defines the setting and the second specifies a character's behavior. While a story generation model such as Text2Story generates the corresponding visual scene, we focus on generating expressive character utterances from these prompts and the scene image. We apply a pretrained vision-language encoder to extract a high-level semantic feature from the representative frame, capturing salient visual context. This feature is then combined with the structured prompts and used to guide a large language model in synthesizing natural, character-consistent dialogue. To ensure contextual consistency across scenes, we introduce a Recursive Narrative Bank that conditions each dialogue generation on the accumulated dialogue history from prior scenes. This approach enables characters to speak in ways that reflect their evolving goals and interactions throughout a story. Finally, we render each utterance as expressive, character-consistent speech, resulting in fully-voiced video narratives. Our framework requires no additional training and demonstrates applicability across a variety of story settings, from fantasy adventures to slice-of-life episodes.
>
---
#### [new 144] TextureSAM: Towards a Texture Aware Foundation Model for Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出TextureSAM，解决Segment Anything Model（SAM）在纹理主导场景中因偏重形状导致分割不足的问题；通过纹理增强数据集与增量训练方法，提升模型对纹理特征的敏感性，在自然与合成数据上mIoU提升0.2/0.18。**

- **链接: [http://arxiv.org/pdf/2505.16540v1](http://arxiv.org/pdf/2505.16540v1)**

> **作者:** Inbal Cohen; Boaz Meivar; Peihan Tu; Shai Avidan; Gal Oren
>
> **摘要:** Segment Anything Models (SAM) have achieved remarkable success in object segmentation tasks across diverse datasets. However, these models are predominantly trained on large-scale semantic segmentation datasets, which introduce a bias toward object shape rather than texture cues in the image. This limitation is critical in domains such as medical imaging, material classification, and remote sensing, where texture changes define object boundaries. In this study, we investigate SAM's bias toward semantics over textures and introduce a new texture-aware foundation model, TextureSAM, which performs superior segmentation in texture-dominant scenarios. To achieve this, we employ a novel fine-tuning approach that incorporates texture augmentation techniques, incrementally modifying training images to emphasize texture features. By leveraging a novel texture-alternation of the ADE20K dataset, we guide TextureSAM to prioritize texture-defined regions, thereby mitigating the inherent shape bias present in the original SAM model. Our extensive experiments demonstrate that TextureSAM significantly outperforms SAM-2 on both natural (+0.2 mIoU) and synthetic (+0.18 mIoU) texture-based segmentation datasets. The code and texture-augmented dataset will be publicly available.
>
---
#### [new 145] Tracking the Flight: Exploring a Computational Framework for Analyzing Escape Responses in Plains Zebra (Equus quagga)
- **分类: cs.CV**

- **简介: 该论文提出计算框架分析斑马逃跑行为，解决无人机拍摄中动物与无人机运动区分难题。评估三种方法（生物成像注册、SfM及混合插值），应用于44只斑马的视频，提取轨迹并识别逃跑时的极化、间距变化等行为模式，验证方法有效性，助力群体行为研究。**

- **链接: [http://arxiv.org/pdf/2505.16882v1](http://arxiv.org/pdf/2505.16882v1)**

> **作者:** Isla Duporge; Sofia Minano; Nikoloz Sirmpilatze; Igor Tatarnikov; Scott Wolf; Adam L. Tyson; Daniel Rubenstein
>
> **备注:** Accepted to the CV4Animals workshop at CVPR 2025
>
> **摘要:** Ethological research increasingly benefits from the growing affordability and accessibility of drones, which enable the capture of high-resolution footage of animal movement at fine spatial and temporal scales. However, analyzing such footage presents the technical challenge of separating animal movement from drone motion. While non-trivial, computer vision techniques such as image registration and Structure-from-Motion (SfM) offer practical solutions. For conservationists, open-source tools that are user-friendly, require minimal setup, and deliver timely results are especially valuable for efficient data interpretation. This study evaluates three approaches: a bioimaging-based registration technique, an SfM pipeline, and a hybrid interpolation method. We apply these to a recorded escape event involving 44 plains zebras, captured in a single drone video. Using the best-performing method, we extract individual trajectories and identify key behavioral patterns: increased alignment (polarization) during escape, a brief widening of spacing just before stopping, and tighter coordination near the group's center. These insights highlight the method's effectiveness and its potential to scale to larger datasets, contributing to broader investigations of collective animal behavior.
>
---
#### [new 146] Semi-Supervised State-Space Model with Dynamic Stacking Filter for Real-World Video Deraining
- **分类: cs.CV**

- **简介: 该论文提出半监督时空状态空间模型，结合动态堆叠滤波器，解决真实视频去雨中合成与真实雨差异导致的泛化问题。通过空间-时间双分支提取特征、自适应滤波融合多帧，并设计中值损失生成伪标签；同时建立真实雨天数据集，提升下游任务效果。**

- **链接: [http://arxiv.org/pdf/2505.16811v1](http://arxiv.org/pdf/2505.16811v1)**

> **作者:** Shangquan Sun; Wenqi Ren; Juxiang Zhou; Shu Wang; Jianhou Gan; Xiaochun Cao
>
> **备注:** 11 Pages, 8 figures, CVPR 2025 Oral Presentation
>
> **摘要:** Significant progress has been made in video restoration under rainy conditions over the past decade, largely propelled by advancements in deep learning. Nevertheless, existing methods that depend on paired data struggle to generalize effectively to real-world scenarios, primarily due to the disparity between synthetic and authentic rain effects. To address these limitations, we propose a dual-branch spatio-temporal state-space model to enhance rain streak removal in video sequences. Specifically, we design spatial and temporal state-space model layers to extract spatial features and incorporate temporal dependencies across frames, respectively. To improve multi-frame feature fusion, we derive a dynamic stacking filter, which adaptively approximates statistical filters for superior pixel-wise feature refinement. Moreover, we develop a median stacking loss to enable semi-supervised learning by generating pseudo-clean patches based on the sparsity prior of rain. To further explore the capacity of deraining models in supporting other vision-based tasks in rainy environments, we introduce a novel real-world benchmark focused on object detection and tracking in rainy conditions. Our method is extensively evaluated across multiple benchmarks containing numerous synthetic and real-world rainy videos, consistently demonstrating its superiority in quantitative metrics, visual quality, efficiency, and its utility for downstream tasks.
>
---
#### [new 147] Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Pixel Reasoner框架，解决视觉语言模型（VLMs）在视觉密集任务中因依赖文本推理导致的局限。通过引入像素空间推理操作（如放大、选帧）及两阶段训练（指令调优+好奇心驱动强化学习），提升模型直接分析视觉信息的能力，在多个基准测试中达开源模型最佳精度。任务为视觉推理，核心是增强VLMs的像素级推理能力。**

- **链接: [http://arxiv.org/pdf/2505.15966v1](http://arxiv.org/pdf/2505.15966v1)**

> **作者:** Alex Su; Haozhe Wang; Weimin Ren; Fangzhen Lin; Wenhu Chen
>
> **备注:** Haozhe Wang and Alex Su contributed equally and listed alphabetically
>
> **摘要:** Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
>
---
#### [new 148] Ranked Entropy Minimization for Continual Test-Time Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对持续测试时适应中的稳定性问题，提出Ranked Entropy Minimization方法。通过渐进式掩码策略结构化预测难度，逐步对齐概率分布并保持熵排序，解决传统熵最小化导致的模型崩溃问题，实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2505.16441v1](http://arxiv.org/pdf/2505.16441v1)**

> **作者:** Jisu Han; Jaemin Na; Wonjun Hwang
>
> **备注:** ICML 2025
>
> **摘要:** Test-time adaptation aims to adapt to realistic environments in an online manner by learning during test time. Entropy minimization has emerged as a principal strategy for test-time adaptation due to its efficiency and adaptability. Nevertheless, it remains underexplored in continual test-time adaptation, where stability is more important. We observe that the entropy minimization method often suffers from model collapse, where the model converges to predicting a single class for all images due to a trivial solution. We propose ranked entropy minimization to mitigate the stability problem of the entropy minimization method and extend its applicability to continuous scenarios. Our approach explicitly structures the prediction difficulty through a progressive masking strategy. Specifically, it gradually aligns the model's probability distributions across different levels of prediction difficulty while preserving the rank order of entropy. The proposed method is extensively evaluated across various benchmarks, demonstrating its effectiveness through empirical results. Our code is available at https://github.com/pilsHan/rem
>
---
#### [new 149] Pose-invariant face recognition via feature-space pose frontalization
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于姿态不变的人脸识别任务，旨在解决侧脸与正面人脸匹配难题。提出特征空间姿态 frontalization 模块（FSPFM），将任意角度侧脸转化为正面表征，并设计预训练与注意力引导微调的训练范式，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16412v1](http://arxiv.org/pdf/2505.16412v1)**

> **作者:** Nikolay Stanishev; Yuhang Lu; Touradj Ebrahimi
>
> **摘要:** Pose-invariant face recognition has become a challenging problem for modern AI-based face recognition systems. It aims at matching a profile face captured in the wild with a frontal face registered in a database. Existing methods perform face frontalization via either generative models or learning a pose robust feature representation. In this paper, a new method is presented to perform face frontalization and recognition within the feature space. First, a novel feature space pose frontalization module (FSPFM) is proposed to transform profile images with arbitrary angles into frontal counterparts. Second, a new training paradigm is proposed to maximize the potential of FSPFM and boost its performance. The latter consists of a pre-training and an attention-guided fine-tuning stage. Moreover, extensive experiments have been conducted on five popular face recognition benchmarks. Results show that not only our method outperforms the state-of-the-art in the pose-invariant face recognition task but also maintains superior performance in other standard scenarios.
>
---
#### [new 150] Background Matters: A Cross-view Bidirectional Modeling Framework for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，解决现有方法忽视背景建模的问题。提出CVBM框架，通过背景辅助建模与双向一致性机制提升前景分割性能，在多数据集达SOTA，使用20%标注数据时胰腺分割DSC值超全监督方法。**

- **链接: [http://arxiv.org/pdf/2505.16625v1](http://arxiv.org/pdf/2505.16625v1)**

> **作者:** Luyang Cao; Jianwei Li; Yinghuan Shi
>
> **备注:** Accepted by IEEE Transactions on Image Processing
>
> **摘要:** Semi-supervised medical image segmentation (SSMIS) leverages unlabeled data to reduce reliance on manually annotated images. However, current SOTA approaches predominantly focus on foreground-oriented modeling (i.e., segmenting only the foreground region) and have largely overlooked the potential benefits of explicitly modeling the background region. Our study theoretically and empirically demonstrates that highly certain predictions in background modeling enhance the confidence of corresponding foreground modeling. Building on this insight, we propose the Cross-view Bidirectional Modeling (CVBM) framework, which introduces a novel perspective by incorporating background modeling to improve foreground modeling performance. Within CVBM, background modeling serves as an auxiliary perspective, providing complementary supervisory signals to enhance the confidence of the foreground model. Additionally, CVBM introduces an innovative bidirectional consistency mechanism, which ensures mutual alignment between foreground predictions and background-guided predictions. Extensive experiments demonstrate that our approach achieves SOTA performance on the LA, Pancreas, ACDC, and HRF datasets. Notably, on the Pancreas dataset, CVBM outperforms fully supervised methods (i.e., DSC: 84.57% vs. 83.89%) while utilizing only 20% of the labeled data. Our code is publicly available at https://github.com/caoluyang0830/CVBM.git.
>
---
#### [new 151] CMRINet: Joint Groupwise Registration and Segmentation for Cardiac Function Quantification from Cine-MRI
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出CMRINet模型，联合组间配准与分割任务，提升心脏功能定量（如LVEF和心肌应变）的准确性和效率。针对传统方法分离处理导致结果受限的问题，该模型通过端到端学习优化，基于374例四腔心MRI数据训练，性能优于传统elastix及现有深度学习方法，计算时间显著减少。**

- **链接: [http://arxiv.org/pdf/2505.16452v1](http://arxiv.org/pdf/2505.16452v1)**

> **作者:** Mohamed S. Elmahdy; Marius Staring; Patrick J. H. de Koning; Samer Alabed; Mahan Salehi; Faisal Alandejani; Michael Sharkey; Ziad Aldabbagh; Andrew J. Swift; Rob J. van der Geest
>
> **备注:** 15 pages, 7 figures, 1 appendix
>
> **摘要:** Accurate and efficient quantification of cardiac function is essential for the estimation of prognosis of cardiovascular diseases (CVDs). One of the most commonly used metrics for evaluating cardiac pumping performance is left ventricular ejection fraction (LVEF). However, LVEF can be affected by factors such as inter-observer variability and varying pre-load and after-load conditions, which can reduce its reproducibility. Additionally, cardiac dysfunction may not always manifest as alterations in LVEF, such as in heart failure and cardiotoxicity diseases. An alternative measure that can provide a relatively load-independent quantitative assessment of myocardial contractility is myocardial strain and strain rate. By using LVEF in combination with myocardial strain, it is possible to obtain a thorough description of cardiac function. Automated estimation of LVEF and other volumetric measures from cine-MRI sequences can be achieved through segmentation models, while strain calculation requires the estimation of tissue displacement between sequential frames, which can be accomplished using registration models. These tasks are often performed separately, potentially limiting the assessment of cardiac function. To address this issue, in this study we propose an end-to-end deep learning (DL) model that jointly estimates groupwise (GW) registration and segmentation for cardiac cine-MRI images. The proposed anatomically-guided Deep GW network was trained and validated on a large dataset of 4-chamber view cine-MRI image series of 374 subjects. A quantitative comparison with conventional GW registration using elastix and two DL-based methods showed that the proposed model improved performance and substantially reduced computation time.
>
---
#### [new 152] Beyond Face Swapping: A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦多模态深度伪造检测任务，针对扩散模型生成的逼真数字人类视频对现有检测方法的挑战，构建了首个大规模数据集DigiFakeAV（含6万视频），并提出DigiShield方法，通过融合视频时空特征与音频语义声学特征提升检测效果，解决新型深度伪造的识别难题。**

- **链接: [http://arxiv.org/pdf/2505.16512v1](http://arxiv.org/pdf/2505.16512v1)**

> **作者:** Jiaxin Liu; Jia Wang; Saihui Hou; Min Ren; Huijia Wu; Zhaofeng He
>
> **摘要:** In recent years, the rapid development of deepfake technology has given rise to an emerging and serious threat to public security: diffusion model-based digital human generation. Unlike traditional face manipulation methods, such models can generate highly realistic videos with consistency through multimodal control signals. Their flexibility and covertness pose severe challenges to existing detection strategies. To bridge this gap, we introduce DigiFakeAV, the first large-scale multimodal digital human forgery dataset based on diffusion models. Employing five latest digital human generation methods (Sonic, Hallo, etc.) and voice cloning method, we systematically produce a dataset comprising 60,000 videos (8.4 million frames), covering multiple nationalities, skin tones, genders, and real-world scenarios, significantly enhancing data diversity and realism. User studies show that the confusion rate between forged and real videos reaches 68%, and existing state-of-the-art (SOTA) detection models exhibit large drops in AUC values on DigiFakeAV, highlighting the challenge of the dataset. To address this problem, we further propose DigiShield, a detection baseline based on spatiotemporal and cross-modal fusion. By jointly modeling the 3D spatiotemporal features of videos and the semantic-acoustic features of audio, DigiShield achieves SOTA performance on both the DigiFakeAV and DF-TIMIT datasets. Experiments show that this method effectively identifies covert artifacts through fine-grained analysis of the temporal evolution of facial features in synthetic videos.
>
---
#### [new 153] One-Step Diffusion-Based Image Compression with Semantic Distillation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于生成式图像压缩任务，旨在解决扩散模型因多步采样导致的高延迟问题。提出OneDC方法：整合一步扩散生成器与潜码压缩模块，利用超先验作为语义信号替代文本提示，并通过语义蒸馏增强其表达能力，同时采用混合优化提升重建与感知质量，实现更快解码（20倍）和更低码率（40%）。**

- **链接: [http://arxiv.org/pdf/2505.16687v1](http://arxiv.org/pdf/2505.16687v1)**

> **作者:** Naifu Xue; Zhaoyang Jia; Jiahao Li; Bin Li; Yuan Zhang; Yan Lu
>
> **摘要:** While recent diffusion-based generative image codecs have shown impressive performance, their iterative sampling process introduces unpleasing latency. In this work, we revisit the design of a diffusion-based codec and argue that multi-step sampling is not necessary for generative compression. Based on this insight, we propose OneDC, a One-step Diffusion-based generative image Codec -- that integrates a latent compression module with a one-step diffusion generator. Recognizing the critical role of semantic guidance in one-step diffusion, we propose using the hyperprior as a semantic signal, overcoming the limitations of text prompts in representing complex visual content. To further enhance the semantic capability of the hyperprior, we introduce a semantic distillation mechanism that transfers knowledge from a pretrained generative tokenizer to the hyperprior codec. Additionally, we adopt a hybrid pixel- and latent-domain optimization to jointly enhance both reconstruction fidelity and perceptual realism. Extensive experiments demonstrate that OneDC achieves SOTA perceptual quality even with one-step generation, offering over 40% bitrate reduction and 20x faster decoding compared to prior multi-step diffusion-based codecs. Code will be released later.
>
---
#### [new 154] V2V: Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation
- **分类: cs.CV**

- **简介: 该论文属于事件视觉模型训练任务，旨在解决现有合成数据生成方法存储需求大、真实数据稀缺导致的数据集扩展难题。提出Video-to-Voxel（V2V）方法，直接将常规视频转为事件体素表示，存储减少150倍并支持实时参数随机化，利用52小时视频训练模型，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.16797v1](http://arxiv.org/pdf/2505.16797v1)**

> **作者:** Hanyue Lou; Jinxiu Liang; Minggui Teng; Yi Wang; Boxin Shi
>
> **摘要:** Event-based cameras offer unique advantages such as high temporal resolution, high dynamic range, and low power consumption. However, the massive storage requirements and I/O burdens of existing synthetic data generation pipelines and the scarcity of real data prevent event-based training datasets from scaling up, limiting the development and generalization capabilities of event vision models. To address this challenge, we introduce Video-to-Voxel (V2V), an approach that directly converts conventional video frames into event-based voxel grid representations, bypassing the storage-intensive event stream generation entirely. V2V enables a 150 times reduction in storage requirements while supporting on-the-fly parameter randomization for enhanced model robustness. Leveraging this efficiency, we train several video reconstruction and optical flow estimation model architectures on 10,000 diverse videos totaling 52 hours--an order of magnitude larger than existing event datasets, yielding substantial improvements.
>
---
#### [new 155] MAGE: A Multi-task Architecture for Gaze Estimation with an Efficient Calibration Module
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出MAGE，一种多任务眼动估计架构，解决现有方法无法提供3D六自由度（6DoF） gaze分析及个体差异导致的泛化问题。通过结合方向与位置特征的多解码器模型，并引入无需屏幕的Easy-Calibration模块进行个性化校准，实验证明其在公开数据集上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.16384v1](http://arxiv.org/pdf/2505.16384v1)**

> **作者:** Haoming Huang; Musen Zhang; Jianxin Yang; Zhen Li; Jinkai Li; Yao Guo
>
> **备注:** Under review
>
> **摘要:** Eye gaze can provide rich information on human psychological activities, and has garnered significant attention in the field of Human-Robot Interaction (HRI). However, existing gaze estimation methods merely predict either the gaze direction or the Point-of-Gaze (PoG) on the screen, failing to provide sufficient information for a comprehensive six Degree-of-Freedom (DoF) gaze analysis in 3D space. Moreover, the variations of eye shape and structure among individuals also impede the generalization capability of these methods. In this study, we propose MAGE, a Multi-task Architecture for Gaze Estimation with an efficient calibration module, to predict the 6-DoF gaze information that is applicable for the real-word HRI. Our basic model encodes both the directional and positional features from facial images, and predicts gaze results with dedicated information flow and multiple decoders. To reduce the impact of individual variations, we propose a novel calibration module, namely Easy-Calibration, to fine-tune the basic model with subject-specific data, which is efficient to implement without the need of a screen. Experimental results demonstrate that our method achieves state-of-the-art performance on the public MPIIFaceGaze, EYEDIAP, and our built IMRGaze datasets.
>
---
#### [new 156] Zero-Shot Hyperspectral Pansharpening Using Hysteresis-Based Tuning for Spectral Quality Control
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出一种基于迟滞调谐的零样本高光谱图像融合方法，针对传统方法光谱保真度不一致的问题，通过轻量级神经网络动态调整各波段权重，并重新定义空间损失函数以捕捉非线性依赖关系，实现全波段高质量融合。方法无需外部训练数据，实验显示其性能达前沿水平且一致性优异。**

- **链接: [http://arxiv.org/pdf/2505.16658v1](http://arxiv.org/pdf/2505.16658v1)**

> **作者:** Giuseppe Guarino; Matteo Ciotola; Gemine Vivone; Giovanni Poggi; Giuseppe Scarpa
>
> **摘要:** Hyperspectral pansharpening has received much attention in recent years due to technological and methodological advances that open the door to new application scenarios. However, research on this topic is only now gaining momentum. The most popular methods are still borrowed from the more mature field of multispectral pansharpening and often overlook the unique challenges posed by hyperspectral data fusion, such as i) the very large number of bands, ii) the overwhelming noise in selected spectral ranges, iii) the significant spectral mismatch between panchromatic and hyperspectral components, iv) a typically high resolution ratio. Imprecise data modeling especially affects spectral fidelity. Even state-of-the-art methods perform well in certain spectral ranges and much worse in others, failing to ensure consistent quality across all bands, with the risk of generating unreliable results. Here, we propose a hyperspectral pansharpening method that explicitly addresses this problem and ensures uniform spectral quality. To this end, a single lightweight neural network is used, with weights that adapt on the fly to each band. During fine-tuning, the spatial loss is turned on and off to ensure a fast convergence of the spectral loss to the desired level, according to a hysteresis-like dynamic. Furthermore, the spatial loss itself is appropriately redefined to account for nonlinear dependencies between panchromatic and spectral bands. Overall, the proposed method is fully unsupervised, with no prior training on external data, flexible, and low-complexity. Experiments on a recently published benchmarking toolbox show that it ensures excellent sharpening quality, competitive with the state-of-the-art, consistently across all bands. The software code and the full set of results are shared online on https://github.com/giu-guarino/rho-PNN.
>
---
#### [new 157] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出多模态扩散模型LLaDA-V，属于多模态任务，旨在改进传统自回归方法的不足，提升视觉语言对齐与数据扩展性。通过整合视觉编码器和MLP连接器，将视觉特征映射到语言空间，实验证明其在多模态任务中性能优于LLaMA3-V等模型，展现扩散模型在多模态领域的潜力。**

- **链接: [http://arxiv.org/pdf/2505.16933v1](http://arxiv.org/pdf/2505.16933v1)**

> **作者:** Zebin You; Shen Nie; Xiaolu Zhang; Jun Hu; Jun Zhou; Zhiwu Lu; Ji-Rong Wen; Chongxuan Li
>
> **摘要:** In this work, we introduce LLaDA-V, a purely diffusion-based Multimodal Large Language Model (MLLM) that integrates visual instruction tuning with masked diffusion models, representing a departure from the autoregressive paradigms dominant in current multimodal approaches. Built upon LLaDA, a representative large language diffusion model, LLaDA-V incorporates a vision encoder and MLP connector that projects visual features into the language embedding space, enabling effective multimodal alignment. Our empirical investigation reveals several intriguing results: First, LLaDA-V demonstrates promising multimodal performance despite its language model being weaker on purely textual tasks than counterparts like LLaMA3-8B and Qwen2-7B. When trained on the same instruction data, LLaDA-V is highly competitive to LLaMA3-V across multimodal tasks with better data scalability. It also narrows the performance gap to Qwen2-VL, suggesting the effectiveness of its architecture for multimodal tasks. Second, LLaDA-V achieves state-of-the-art performance in multimodal understanding compared to existing hybrid autoregressive-diffusion and purely diffusion-based MLLMs. Our findings suggest that large language diffusion models show promise in multimodal contexts and warrant further investigation in future research. Project page and codes: https://ml-gsai.github.io/LLaDA-V-demo/.
>
---
#### [new 158] VERDI: VLM-Embedded Reasoning for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶决策任务，旨在解决现有视觉语言模型（VLM）因参数量大、推理效率低导致部署困难的问题。提出VERDI框架，通过训练时将VLM的推理过程和常识知识蒸馏到自动驾驶模块（感知、预测、规划），使AD系统内化结构化推理能力，同时保持高效推理（NuScenes数据集上l2距离提升10%）。**

- **链接: [http://arxiv.org/pdf/2505.15925v1](http://arxiv.org/pdf/2505.15925v1)**

> **作者:** Bowen Feng; Zhiting Mei; Baiang Li; Julian Ost; Roger Girgis; Anirudha Majumdar; Felix Heide
>
> **摘要:** While autonomous driving (AD) stacks struggle with decision making under partial observability and real-world complexity, human drivers are capable of commonsense reasoning to make near-optimal decisions with limited information. Recent work has attempted to leverage finetuned Vision-Language Models (VLMs) for trajectory planning at inference time to emulate human behavior. Despite their success in benchmark evaluations, these methods are often impractical to deploy (a 70B parameter VLM inference at merely 8 tokens per second requires more than 160G of memory), and their monolithic network structure prohibits safety decomposition. To bridge this gap, we propose VLM-Embedded Reasoning for autonomous Driving (VERDI), a training-time framework that distills the reasoning process and commonsense knowledge of VLMs into the AD stack. VERDI augments modular differentiable end-to-end (e2e) AD models by aligning intermediate module outputs at the perception, prediction, and planning stages with text features explaining the driving reasoning process produced by VLMs. By encouraging alignment in latent space, \textsc{VERDI} enables the modular AD stack to internalize structured reasoning, without incurring the inference-time costs of large VLMs. We demonstrate the effectiveness of our method on the NuScenes dataset and find that VERDI outperforms existing e2e methods that do not embed reasoning by 10% in $\ell_{2}$ distance, while maintaining high inference speed.
>
---
#### [new 159] PCMamba: Physics-Informed Cross-Modal State Space Model for Dual-Camera Compressive Hyperspectral Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出PCMamba模型，针对双摄像头压缩高光谱成像（DCCHI）中因忽略物理因素导致的重建瓶颈，通过融合物理成像过程与跨模态交互，分解温度、发射率和纹理属性，设计CSMB模块提升重建质量，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16373v1](http://arxiv.org/pdf/2505.16373v1)**

> **作者:** Ge Meng; Zhongnan Cai; Jingyan Tu; Yingying Wang; Chenxin Li; Yue Huang; Xinghao Ding
>
> **摘要:** Panchromatic (PAN) -assisted Dual-Camera Compressive Hyperspectral Imaging (DCCHI) is a key technology in snapshot hyperspectral imaging. Existing research primarily focuses on exploring spectral information from 2D compressive measurements and spatial information from PAN images in an explicit manner, leading to a bottleneck in HSI reconstruction. Various physical factors, such as temperature, emissivity, and multiple reflections between objects, play a critical role in the process of a sensor acquiring hyperspectral thermal signals. Inspired by this, we attempt to investigate the interrelationships between physical properties to provide deeper theoretical insights for HSI reconstruction. In this paper, we propose a Physics-Informed Cross-Modal State Space Model Network (PCMamba) for DCCHI, which incorporates the forward physical imaging process of HSI into the linear complexity of Mamba to facilitate lightweight and high-quality HSI reconstruction. Specifically, we analyze the imaging process of hyperspectral thermal signals to enable the network to disentangle the three key physical properties-temperature, emissivity, and texture. By fully exploiting the potential information embedded in 2D measurements and PAN images, the HSIs are reconstructed through a physics-driven synthesis process. Furthermore, we design a Cross-Modal Scanning Mamba Block (CSMB) that introduces inter-modal pixel-wise interaction with positional inductive bias by cross-scanning the backbone features and PAN features. Extensive experiments conducted on both real and simulated datasets demonstrate that our method significantly outperforms SOTA methods in both quantitative and qualitative metrics.
>
---
#### [new 160] NovelSeek: When Agent Becomes the Scientist -- Building Closed-Loop System from Hypothesis to Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出NovelSeek框架，属于自主科学研究（ASR）任务，旨在通过闭环多智能体系统提升科研效率与跨领域适应性。解决传统科研中效率低、创新慢及人机协作不足的问题。工作包括构建可扩展（跨12领域）、交互式（支持专家反馈）和高效（快速提升模型性能）的系统，实验显示其在反应预测、活性预测等任务中显著缩短时间并提升精度。**

- **链接: [http://arxiv.org/pdf/2505.16938v1](http://arxiv.org/pdf/2505.16938v1)**

> **作者:** NovelSeek Team; Bo Zhang; Shiyang Feng; Xiangchao Yan; Jiakang Yuan; Zhiyin Yu; Xiaohan He; Songtao Huang; Shaowei Hou; Zheng Nie; Zhilong Wang; Jinyao Liu; Runmin Ma; Tianshuo Peng; Peng Ye; Dongzhan Zhou; Shufei Zhang; Xiaosong Wang; Yilan Zhang; Meng Li; Zhongying Tu; Xiangyu Yue; Wangli Ouyang; Bowen Zhou; Lei Bai
>
> **备注:** HomePage: https://alpha-innovator.github.io/NovelSeek-project-page
>
> **摘要:** Artificial Intelligence (AI) is accelerating the transformation of scientific research paradigms, not only enhancing research efficiency but also driving innovation. We introduce NovelSeek, a unified closed-loop multi-agent framework to conduct Autonomous Scientific Research (ASR) across various scientific research fields, enabling researchers to tackle complicated problems in these fields with unprecedented speed and precision. NovelSeek highlights three key advantages: 1) Scalability: NovelSeek has demonstrated its versatility across 12 scientific research tasks, capable of generating innovative ideas to enhance the performance of baseline code. 2) Interactivity: NovelSeek provides an interface for human expert feedback and multi-agent interaction in automated end-to-end processes, allowing for the seamless integration of domain expert knowledge. 3) Efficiency: NovelSeek has achieved promising performance gains in several scientific fields with significantly less time cost compared to human efforts. For instance, in reaction yield prediction, it increased from 27.6% to 35.4% in just 12 hours; in enhancer activity prediction, accuracy rose from 0.52 to 0.79 with only 4 hours of processing; and in 2D semantic segmentation, precision advanced from 78.8% to 81.0% in a mere 30 hours.
>
---
#### [new 161] SEM: Enhancing Spatial Understanding for Robust Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决现有方法在3D空间理解与机器人本体建模上的不足。提出SEM框架，通过空间增强模块（添加3D几何上下文）和图模型编码器（捕捉关节依赖关系），提升策略模型的空间推理能力，实现更鲁棒、通用的操作性能。**

- **链接: [http://arxiv.org/pdf/2505.16196v1](http://arxiv.org/pdf/2505.16196v1)**

> **作者:** Xuewu Lin; Tianwei Lin; Lichao Huang; Hongyu Xie; Yiwei Jin; Keyu Li; Zhizhong Su
>
> **摘要:** A key challenge in robot manipulation lies in developing policy models with strong spatial understanding, the ability to reason about 3D geometry, object relations, and robot embodiment. Existing methods often fall short: 3D point cloud models lack semantic abstraction, while 2D image encoders struggle with spatial reasoning. To address this, we propose SEM (Spatial Enhanced Manipulation model), a novel diffusion-based policy framework that explicitly enhances spatial understanding from two complementary perspectives. A spatial enhancer augments visual representations with 3D geometric context, while a robot state encoder captures embodiment-aware structure through graphbased modeling of joint dependencies. By integrating these modules, SEM significantly improves spatial understanding, leading to robust and generalizable manipulation across diverse tasks that outperform existing baselines.
>
---
#### [new 162] Generative Latent Coding for Ultra-Low Bitrate Image and Video Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出GLC模型，用于超低码率图像及视频压缩。针对传统像素空间编码与人类感知不匹配、难以兼顾高真实感与保真度的问题，其在生成VQ-VAE的潜在空间进行编码，改进超先验模块并引入语义一致损失函数。实验显示其在0.04bpp下图像压缩效果优于现有方法，视频压缩码率节省65.3%。**

- **链接: [http://arxiv.org/pdf/2505.16177v1](http://arxiv.org/pdf/2505.16177v1)**

> **作者:** Linfeng Qi; Zhaoyang Jia; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **摘要:** Most existing approaches for image and video compression perform transform coding in the pixel space to reduce redundancy. However, due to the misalignment between the pixel-space distortion and human perception, such schemes often face the difficulties in achieving both high-realism and high-fidelity at ultra-low bitrate. To solve this problem, we propose \textbf{G}enerative \textbf{L}atent \textbf{C}oding (\textbf{GLC}) models for image and video compression, termed GLC-image and GLC-Video. The transform coding of GLC is conducted in the latent space of a generative vector-quantized variational auto-encoder (VQ-VAE). Compared to the pixel-space, such a latent space offers greater sparsity, richer semantics and better alignment with human perception, and show its advantages in achieving high-realism and high-fidelity compression. To further enhance performance, we improve the hyper prior by introducing a spatial categorical hyper module in GLC-image and a spatio-temporal categorical hyper module in GLC-video. Additionally, the code-prediction-based loss function is proposed to enhance the semantic consistency. Experiments demonstrate that our scheme shows high visual quality at ultra-low bitrate for both image and video compression. For image compression, GLC-image achieves an impressive bitrate of less than $0.04$ bpp, achieving the same FID as previous SOTA model MS-ILLM while using $45\%$ fewer bitrate on the CLIC 2020 test set. For video compression, GLC-video achieves 65.3\% bitrate saving over PLVC in terms of DISTS.
>
---
#### [new 163] GradPCA: Leveraging NTK Alignment for Reliable Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出GradPCA方法，用于分布外检测（OOD）。针对现有方法性能不一致问题，利用神经切向核（NTK）对齐的梯度低秩结构，通过PCA分析梯度类均值提升检测可靠性。理论分析揭示特征质量（预训练 vs 非预训练）对检测效果的关键作用，实验验证GradPCA的优越性并为设计检测器提供指导。（98字）**

- **链接: [http://arxiv.org/pdf/2505.16017v1](http://arxiv.org/pdf/2505.16017v1)**

> **作者:** Mariia Seleznova; Hung-Hsu Chou; Claudio Mayrink Verdun; Gitta Kutyniok
>
> **摘要:** We introduce GradPCA, an Out-of-Distribution (OOD) detection method that exploits the low-rank structure of neural network gradients induced by Neural Tangent Kernel (NTK) alignment. GradPCA applies Principal Component Analysis (PCA) to gradient class-means, achieving more consistent performance than existing methods across standard image classification benchmarks. We provide a theoretical perspective on spectral OOD detection in neural networks to support GradPCA, highlighting feature-space properties that enable effective detection and naturally emerge from NTK alignment. Our analysis further reveals that feature quality -- particularly the use of pretrained versus non-pretrained representations -- plays a crucial role in determining which detectors will succeed. Extensive experiments validate the strong performance of GradPCA, and our theoretical framework offers guidance for designing more principled spectral OOD detectors.
>
---
#### [new 164] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **简介: 该论文提出MoRE-Brain框架，用于可解释且跨被试通用的fMRI视觉解码。针对现有方法忽视可解释性及泛化性问题，其采用基于脑网络的分层专家混合架构，通过动态路由机制结合CLIP编码与扩散模型，提升重建保真度与神经机制解释，并验证其有效性和信号利用效率。**

- **链接: [http://arxiv.org/pdf/2505.15946v1](http://arxiv.org/pdf/2505.15946v1)**

> **作者:** Yuxiang Wei; Yanteng Zhang; Xi Xiao; Tianyang Wang; Xiao Wang; Vince D. Calhoun
>
> **摘要:** Decoding visual experiences from fMRI offers a powerful avenue to understand human perception and develop advanced brain-computer interfaces. However, current progress often prioritizes maximizing reconstruction fidelity while overlooking interpretability, an essential aspect for deriving neuroscientific insight. To address this gap, we propose MoRE-Brain, a neuro-inspired framework designed for high-fidelity, adaptable, and interpretable visual reconstruction. MoRE-Brain uniquely employs a hierarchical Mixture-of-Experts architecture where distinct experts process fMRI signals from functionally related voxel groups, mimicking specialized brain networks. The experts are first trained to encode fMRI into the frozen CLIP space. A finetuned diffusion model then synthesizes images, guided by expert outputs through a novel dual-stage routing mechanism that dynamically weighs expert contributions across the diffusion process. MoRE-Brain offers three main advancements: First, it introduces a novel Mixture-of-Experts architecture grounded in brain network principles for neuro-decoding. Second, it achieves efficient cross-subject generalization by sharing core expert networks while adapting only subject-specific routers. Third, it provides enhanced mechanistic insight, as the explicit routing reveals precisely how different modeled brain regions shape the semantic and spatial attributes of the reconstructed image. Extensive experiments validate MoRE-Brain's high reconstruction fidelity, with bottleneck analyses further demonstrating its effective utilization of fMRI signals, distinguishing genuine neural decoding from over-reliance on generative priors. Consequently, MoRE-Brain marks a substantial advance towards more generalizable and interpretable fMRI-based visual decoding. Code will be publicly available soon: https://github.com/yuxiangwei0808/MoRE-Brain.
>
---
#### [new 165] Masked Conditioning for Deep Generative Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出masked conditioning方法，解决工程领域小规模、稀疏混合条件数据的生成问题。通过训练时掩码模拟稀疏条件，设计灵活嵌入处理数值/类别数据，整合至VAE和扩散模型，并结合预训练模型提升生成质量，实现高效可控的工程数据生成。**

- **链接: [http://arxiv.org/pdf/2505.16725v1](http://arxiv.org/pdf/2505.16725v1)**

> **作者:** Phillip Mueller; Jannik Wiese; Sebastian Mueller; Lars Mikelsons
>
> **摘要:** Datasets in engineering domains are often small, sparsely labeled, and contain numerical as well as categorical conditions. Additionally. computational resources are typically limited in practical applications which hinders the adoption of generative models for engineering tasks. We introduce a novel masked-conditioning approach, that enables generative models to work with sparse, mixed-type data. We mask conditions during training to simulate sparse conditions at inference time. For this purpose, we explore the use of various sparsity schedules that show different strengths and weaknesses. In addition, we introduce a flexible embedding that deals with categorical as well as numerical conditions. We integrate our method into an efficient variational autoencoder as well as a latent diffusion model and demonstrate the applicability of our approach on two engineering-related datasets of 2D point clouds and images. Finally, we show that small models trained on limited data can be coupled with large pretrained foundation models to improve generation quality while retaining the controllability induced by our conditioning scheme.
>
---
#### [new 166] TDFormer: A Top-Down Attention-Controlled Spiking Transformer
- **分类: cs.NE; cs.AI; cs.CV**

- **简介: 该论文提出TDFormer，改进传统脉冲神经网络（SNN）因膜电位隐式特性导致的时间信息表示不足及梯度消失问题。通过引入自顶向下反馈结构，利用高阶时序表征调节低阶信息处理，增强时间步间互信息并缓解梯度消失，在ImageNet获86.83%准确率，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.15840v1](http://arxiv.org/pdf/2505.15840v1)**

> **作者:** Zizheng Zhu; Yingchao Yu; Zeqi Zheng; Zhaofei Yu; Yaochu Jin
>
> **备注:** 28 pages
>
> **摘要:** Traditional spiking neural networks (SNNs) can be viewed as a combination of multiple subnetworks with each running for one time step, where the parameters are shared, and the membrane potential serves as the only information link between them. However, the implicit nature of the membrane potential limits its ability to effectively represent temporal information. As a result, each time step cannot fully leverage information from previous time steps, seriously limiting the model's performance. Inspired by the top-down mechanism in the brain, we introduce TDFormer, a novel model with a top-down feedback structure that functions hierarchically and leverages high-order representations from earlier time steps to modulate the processing of low-order information at later stages. The feedback structure plays a role from two perspectives: 1) During forward propagation, our model increases the mutual information across time steps, indicating that richer temporal information is being transmitted and integrated in different time steps. 2) During backward propagation, we theoretically prove that the feedback structure alleviates the problem of vanishing gradients along the time dimension. We find that these mechanisms together significantly and consistently improve the model performance on multiple datasets. In particular, our model achieves state-of-the-art performance on ImageNet with an accuracy of 86.83%.
>
---
#### [new 167] MambaStyle: Efficient StyleGAN Inversion for Real Image Editing with State-Space Models
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于StyleGAN逆过程与图像编辑任务，旨在解决现有方法在重建质量、编辑效果与计算效率间难以平衡的问题。提出MambaStyle，通过集成视觉状态空间模型（VSSM）的单阶段编码器架构，在减少参数与计算量的同时，实现高质量逆过程、灵活编辑及实时应用。**

- **链接: [http://arxiv.org/pdf/2505.15822v1](http://arxiv.org/pdf/2505.15822v1)**

> **作者:** Jhon Lopez; Carlos Hinojosa; Henry Arguello; Bernard Ghanem
>
> **摘要:** The task of inverting real images into StyleGAN's latent space to manipulate their attributes has been extensively studied. However, existing GAN inversion methods struggle to balance high reconstruction quality, effective editability, and computational efficiency. In this paper, we introduce MambaStyle, an efficient single-stage encoder-based approach for GAN inversion and editing that leverages vision state-space models (VSSMs) to address these challenges. Specifically, our approach integrates VSSMs within the proposed architecture, enabling high-quality image inversion and flexible editing with significantly fewer parameters and reduced computational complexity compared to state-of-the-art methods. Extensive experiments show that MambaStyle achieves a superior balance among inversion accuracy, editing quality, and computational efficiency. Notably, our method achieves superior inversion and editing results with reduced model complexity and faster inference, making it suitable for real-time applications.
>
---
#### [new 168] Bridging the Dynamic Perception Gap: Training-Free Draft Chain-of-Thought for Dynamic Multimodal Spatial Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文聚焦动态多模态空间推理任务，解决现有方法在动态环境中的推理局限。提出GRASSLAND迷宫导航基准及D2R框架，通过叠加动态视觉草图与文本推理链，无需训练即提升模型在动态场景中的推理能力。**

- **链接: [http://arxiv.org/pdf/2505.16579v1](http://arxiv.org/pdf/2505.16579v1)**

> **作者:** Siqu Ou; Hongcheng Liu; Pingjie Wang; Yusheng Liao; Chuan Xuan; Yanfeng Wang; Yu Wang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** While chains-of-thought (CoT) have advanced complex reasoning in multimodal large language models (MLLMs), existing methods remain confined to text or static visual domains, often faltering in dynamic spatial reasoning tasks. To bridge this gap, we present GRASSLAND, a novel maze navigation benchmark designed to evaluate dynamic spatial reasoning. Our experiments show that augmenting textual reasoning chains with dynamic visual drafts, overlaid on input images, significantly outperforms conventional approaches, offering new insights into spatial reasoning in evolving environments. To generalize this capability, we propose D2R (Dynamic Draft-Augmented Reasoning), a training-free framework that seamlessly integrates textual CoT with corresponding visual drafts into MLLMs. Extensive evaluations demonstrate that D2R consistently enhances performance across diverse tasks, establishing a robust baseline for dynamic spatial reasoning without requiring model fine-tuning. Project is open at https://github.com/Cratileo/D2R.
>
---
#### [new 169] Comprehensive Lung Disease Detection Using Deep Learning Models and Hybrid Chest X-ray Data with Explainable AI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于肺部疾病检测任务，旨在提升模型对混合数据的泛化能力和诊断准确性。通过整合多源胸片数据，测试了多种深度学习模型，发现VGG16等模型在混合数据上达99%准确率，并用LIME技术解释模型决策，解决黑箱问题，推动可信赖医疗AI发展。**

- **链接: [http://arxiv.org/pdf/2505.16028v1](http://arxiv.org/pdf/2505.16028v1)**

> **作者:** Shuvashis Sarker; Shamim Rahim Refat; Faika Fairuj Preotee; Tanvir Rouf Shawon; Raihan Tanvir
>
> **备注:** Accepted for publication in 2024 27th International Conference on Computer and Information Technology (ICCIT)
>
> **摘要:** Advanced diagnostic instruments are crucial for the accurate detection and treatment of lung diseases, which affect millions of individuals globally. This study examines the effectiveness of deep learning and transfer learning models using a hybrid dataset, created by merging four individual datasets from Bangladesh and global sources. The hybrid dataset significantly enhances model accuracy and generalizability, particularly in detecting COVID-19, pneumonia, lung opacity, and normal lung conditions from chest X-ray images. A range of models, including CNN, VGG16, VGG19, InceptionV3, Xception, ResNet50V2, InceptionResNetV2, MobileNetV2, and DenseNet121, were applied to both individual and hybrid datasets. The results showed superior performance on the hybrid dataset, with VGG16, Xception, ResNet50V2, and DenseNet121 each achieving an accuracy of 99%. This consistent performance across the hybrid dataset highlights the robustness of these models in handling diverse data while maintaining high accuracy. To understand the models implicit behavior, explainable AI techniques were employed to illuminate their black-box nature. Specifically, LIME was used to enhance the interpretability of model predictions, especially in cases of misclassification, contributing to the development of reliable and interpretable AI-driven solutions for medical imaging.
>
---
#### [new 170] Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于端到端自动驾驶任务，解决强化学习（RL）训练困难及模型基础RL依赖特权信息的问题。提出Raw2Drive，通过双流MBRL方法，先训练特权世界模型与规划器，再利用引导机制对齐原始传感器模型，并整合先验知识训练策略，成为CARLA 2.0首个RL方法且性能最优。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16394v1](http://arxiv.org/pdf/2505.16394v1)**

> **作者:** Zhenjie Yang; Xiaosong Jia; Qifeng Li; Xue Yang; Maoqing Yao; Junchi Yan
>
> **摘要:** Reinforcement Learning (RL) can mitigate the causal confusion and distribution shift inherent to imitation learning (IL). However, applying RL to end-to-end autonomous driving (E2E-AD) remains an open problem for its training difficulty, and IL is still the mainstream paradigm in both academia and industry. Recently Model-based Reinforcement Learning (MBRL) have demonstrated promising results in neural planning; however, these methods typically require privileged information as input rather than raw sensor data. We fill this gap by designing Raw2Drive, a dual-stream MBRL approach. Initially, we efficiently train an auxiliary privileged world model paired with a neural planner that uses privileged information as input. Subsequently, we introduce a raw sensor world model trained via our proposed Guidance Mechanism, which ensures consistency between the raw sensor world model and the privileged world model during rollouts. Finally, the raw sensor world model combines the prior knowledge embedded in the heads of the privileged world model to effectively guide the training of the raw sensor policy. Raw2Drive is so far the only RL based end-to-end method on CARLA Leaderboard 2.0, and Bench2Drive and it achieves state-of-the-art performance.
>
---
#### [new 171] IRONIC: Coherence-Aware Reasoning Chains for Multi-Modal Sarcasm Detection
- **分类: cs.CL; cs.AI; cs.CV; 68T50; I.2.7; I.2.10**

- **简介: 该论文属于多模态讽刺检测任务，旨在解决现有方法未能有效利用人类认知过程分析图文关联以识别讽刺的问题。提出IRONIC框架，通过多模态连贯关系（指代、类比、语用）建模图文联系，实现零样本场景下的讽刺检测，并达当前最优效果。**

- **链接: [http://arxiv.org/pdf/2505.16258v1](http://arxiv.org/pdf/2505.16258v1)**

> **作者:** Aashish Anantha Ramakrishnan; Aadarsh Anantha Ramakrishnan; Dongwon Lee
>
> **摘要:** Interpreting figurative language such as sarcasm across multi-modal inputs presents unique challenges, often requiring task-specific fine-tuning and extensive reasoning steps. However, current Chain-of-Thought approaches do not efficiently leverage the same cognitive processes that enable humans to identify sarcasm. We present IRONIC, an in-context learning framework that leverages Multi-modal Coherence Relations to analyze referential, analogical and pragmatic image-text linkages. Our experiments show that IRONIC achieves state-of-the-art performance on zero-shot Multi-modal Sarcasm Detection across different baselines. This demonstrates the need for incorporating linguistic and cognitive insights into the design of multi-modal reasoning strategies. Our code is available at: https://github.com/aashish2000/IRONIC
>
---
#### [new 172] ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ManipLVM-R1框架，针对具身机器人操作任务中依赖昂贵标注数据及泛化性差的问题，通过强化学习（RLVR）替代传统监督，设计两种规则奖励函数（Affordance Perception Reward提升交互区域定位，Trajectory Match Reward确保动作路径物理合理性），以增强模型在陌生场景中的推理与适应能力。**

- **链接: [http://arxiv.org/pdf/2505.16517v1](http://arxiv.org/pdf/2505.16517v1)**

> **作者:** Zirui Song; Guangxian Ouyang; Mingzhe Li; Yuheng Ji; Chenxi Wang; Zixiang Xu; Zeyu Zhang; Xiaoqing Zhang; Qian Jiang; Zhenhao Chen; Zhongzhi Li; Rui Yan; Xiuying Chen
>
> **备注:** 13pages
>
> **摘要:** Large Vision-Language Models (LVLMs) have recently advanced robotic manipulation by leveraging vision for scene perception and language for instruction following. However, existing methods rely heavily on costly human-annotated training datasets, which limits their generalization and causes them to struggle in out-of-domain (OOD) scenarios, reducing real-world adaptability. To address these challenges, we propose ManipLVM-R1, a novel reinforcement learning framework that replaces traditional supervision with Reinforcement Learning using Verifiable Rewards (RLVR). By directly optimizing for task-aligned outcomes, our method enhances generalization and physical reasoning while removing the dependence on costly annotations. Specifically, we design two rule-based reward functions targeting key robotic manipulation subtasks: an Affordance Perception Reward to enhance localization of interaction regions, and a Trajectory Match Reward to ensure the physical plausibility of action paths. These rewards provide immediate feedback and impose spatial-logical constraints, encouraging the model to go beyond shallow pattern matching and instead learn deeper, more systematic reasoning about physical interactions.
>
---
#### [new 173] Implicit Neural Shape Optimization for 3D High-Contrast Electrical Impedance Tomography
- **分类: math.NA; cs.CV; cs.NA**

- **简介: 该论文提出隐式神经形状优化框架，用于3D高对比度电阻抗层析成像，解决金属植入物等场景下传统方法因严重病态性导致的重建问题。通过结合形状导数优化与隐式神经表示，引入高对比度界面条件处理及高效潜空间降维，提升算法收敛与性能，适用于医疗和工业检测。**

- **链接: [http://arxiv.org/pdf/2505.16487v1](http://arxiv.org/pdf/2505.16487v1)**

> **作者:** Junqing Chen; Haibo Liu
>
> **摘要:** We present a novel implicit neural shape optimization framework for 3D high-contrast Electrical Impedance Tomography (EIT), addressing scenarios where conductivity exhibits sharp discontinuities across material interfaces. These high-contrast cases, prevalent in metallic implant monitoring and industrial defect detection, challenge traditional reconstruction methods due to severe ill-posedness. Our approach synergizes shape optimization with implicit neural representations, introducing key innovations including a shape derivative-based optimization scheme that explicitly incorporates high-contrast interface conditions and an efficient latent space representation that reduces variable dimensionality. Through rigorous theoretical analysis of algorithm convergence and extensive numerical experiments, we demonstrate substantial performance improvements, establishing our framework as promising for practical applications in medical imaging with metallic implants and industrial non-destructive testing.
>
---
#### [new 174] P3Net: Progressive and Periodic Perturbation for Semi-Supervised Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，旨在解决扰动机制过度使用及边界区域预测不准确问题。提出渐进周期性扰动机制（P3M）动态调整扰动，并设计边界聚焦损失增强细节敏感性，实验显示方法性能优越且可扩展。**

- **链接: [http://arxiv.org/pdf/2505.15861v1](http://arxiv.org/pdf/2505.15861v1)**

> **作者:** Zhenyan Yao; Miao Zhang; Lanhu Wu; Yongri Piao; Feng Tian; Weibing Sun; Huchuan Lu
>
> **摘要:** Perturbation with diverse unlabeled data has proven beneficial for semi-supervised medical image segmentation (SSMIS). While many works have successfully used various perturbation techniques, a deeper understanding of learning perturbations is needed. Excessive or inappropriate perturbation can have negative effects, so we aim to address two challenges: how to use perturbation mechanisms to guide the learning of unlabeled data through labeled data, and how to ensure accurate predictions in boundary regions. Inspired by human progressive and periodic learning, we propose a progressive and periodic perturbation mechanism (P3M) and a boundary-focused loss. P3M enables dynamic adjustment of perturbations, allowing the model to gradually learn them. Our boundary-focused loss encourages the model to concentrate on boundary regions, enhancing sensitivity to intricate details and ensuring accurate predictions. Experimental results demonstrate that our method achieves state-of-the-art performance on two 2D and 3D datasets. Moreover, P3M is extendable to other methods, and the proposed loss serves as a universal tool for improving existing methods, highlighting the scalability and applicability of our approach.
>
---
#### [new 175] Compressing Human Body Video with Interactive Semantics: A Generative Approach
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种基于生成模型的交互式人体视频压缩方法。通过3D人体模型将动态分解为可配置语义嵌入，实现高效压缩与可控编辑，解码时重建高质量视频。相比VVC及现有方案，在低码率下表现更优，支持实时交互且无需额外处理，适用于元宇宙数字人通信。**

- **链接: [http://arxiv.org/pdf/2505.16152v1](http://arxiv.org/pdf/2505.16152v1)**

> **作者:** Bolin Chen; Shanzhi Yin; Hanwei Zhu; Lingyu Zhu; Zihan Zhang; Jie Chen; Ru-Ling Liao; Shiqi Wang; Yan Ye
>
> **摘要:** In this paper, we propose to compress human body video with interactive semantics, which can facilitate video coding to be interactive and controllable by manipulating semantic-level representations embedded in the coded bitstream. In particular, the proposed encoder employs a 3D human model to disentangle nonlinear dynamics and complex motion of human body signal into a series of configurable embeddings, which are controllably edited, compactly compressed, and efficiently transmitted. Moreover, the proposed decoder can evolve the mesh-based motion fields from these decoded semantics to realize the high-quality human body video reconstruction. Experimental results illustrate that the proposed framework can achieve promising compression performance for human body videos at ultra-low bitrate ranges compared with the state-of-the-art video coding standard Versatile Video Coding (VVC) and the latest generative compression schemes. Furthermore, the proposed framework enables interactive human body video coding without any additional pre-/post-manipulation processes, which is expected to shed light on metaverse-related digital human communication in the future.
>
---
#### [new 176] UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UAV-Flow基准，专注语言引导的无人机精细轨迹模仿学习，解决短程反应式飞行控制问题。通过收集多环境数据、构建框架及模拟工具，使无人机模仿专家轨迹执行原子级指令，实验表明VLA模型优于VLN，凸显空间定位关键性。**

- **链接: [http://arxiv.org/pdf/2505.15725v1](http://arxiv.org/pdf/2505.15725v1)**

> **作者:** Xiangyu Wang; Donglin Yang; Yue Liao; Wenhao Zheng; wenjun wu; Bin Dai; Hongsheng Li; Si Liu
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are evolving into language-interactive platforms, enabling more intuitive forms of human-drone interaction. While prior works have primarily focused on high-level planning and long-horizon navigation, we shift attention to language-guided fine-grained trajectory control, where UAVs execute short-range, reactive flight behaviors in response to language instructions. We formalize this problem as the Flying-on-a-Word (Flow) task and introduce UAV imitation learning as an effective approach. In this framework, UAVs learn fine-grained control policies by mimicking expert pilot trajectories paired with atomic language instructions. To support this paradigm, we present UAV-Flow, the first real-world benchmark for language-conditioned, fine-grained UAV control. It includes a task formulation, a large-scale dataset collected in diverse environments, a deployable control framework, and a simulation suite for systematic evaluation. Our design enables UAVs to closely imitate the precise, expert-level flight trajectories of human pilots and supports direct deployment without sim-to-real gap. We conduct extensive experiments on UAV-Flow, benchmarking VLN and VLA paradigms. Results show that VLA models are superior to VLN baselines and highlight the critical role of spatial grounding in the fine-grained Flow setting.
>
---
#### [new 177] Backdoor Cleaning without External Guidance in MLLM Fine-tuning
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于多模态大模型微调中的后门防御任务。针对恶意微调植入后门的问题，提出BYE框架，通过分析注意力熵模式发现异常，分三阶段（提取注意力图、熵评分分层、无监督聚类）识别并过滤可疑样本，无需额外监督或模型修改，有效清除后门同时保持任务性能。**

- **链接: [http://arxiv.org/pdf/2505.16916v1](http://arxiv.org/pdf/2505.16916v1)**

> **作者:** Xuankun Rong; Wenke Huang; Jian Liang; Jinhe Bi; Xun Xiao; Yiming Li; Bo Du; Mang Ye
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly deployed in fine-tuning-as-a-service (FTaaS) settings, where user-submitted datasets adapt general-purpose models to downstream tasks. This flexibility, however, introduces serious security risks, as malicious fine-tuning can implant backdoors into MLLMs with minimal effort. In this paper, we observe that backdoor triggers systematically disrupt cross-modal processing by causing abnormal attention concentration on non-semantic regions--a phenomenon we term attention collapse. Based on this insight, we propose Believe Your Eyes (BYE), a data filtering framework that leverages attention entropy patterns as self-supervised signals to identify and filter backdoor samples. BYE operates via a three-stage pipeline: (1) extracting attention maps using the fine-tuned model, (2) computing entropy scores and profiling sensitive layers via bimodal separation, and (3) performing unsupervised clustering to remove suspicious samples. Unlike prior defenses, BYE equires no clean supervision, auxiliary labels, or model modifications. Extensive experiments across various datasets, models, and diverse trigger types validate BYE's effectiveness: it achieves near-zero attack success rates while maintaining clean-task performance, offering a robust and generalizable solution against backdoor threats in MLLMs.
>
---
#### [new 178] When Are Concepts Erased From Diffusion Models?
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于扩散模型概念擦除任务，旨在解决现有方法对目标概念删除彻底性不明确的问题。提出概念生成概率抑制和内部机制干扰两种擦除机制，并构建包含对抗攻击、探测技术和生成分析的评估框架，量化擦除效果与鲁棒性平衡。**

- **链接: [http://arxiv.org/pdf/2505.17013v1](http://arxiv.org/pdf/2505.17013v1)**

> **作者:** Kevin Lu; Nicky Kriplani; Rohit Gandikota; Minh Pham; David Bau; Chinmay Hegde; Niv Cohen
>
> **备注:** Project Page: https://nyu-dice-lab.github.io/when-are-concepts-erased/
>
> **摘要:** Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.
>
---
#### [new 179] Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于视觉语言模型（VLMs）推理优化任务，旨在解决现有强化学习方法（如GRPO）因强制生成完整推理轨迹导致的计算成本过高问题。提出TON方法，通过两阶段训练（监督微调阶段的"thought dropout"和GRPO探索阶段），使模型学会根据问题难度选择性推理，减少90%生成长度且保持性能，实现高效的人类-like推理模式。**

- **链接: [http://arxiv.org/pdf/2505.16854v1](http://arxiv.org/pdf/2505.16854v1)**

> **作者:** Jiaqi Wang; Kevin Qinghong Lin; James Cheng; Mike Zheng Shou
>
> **摘要:** Reinforcement Learning (RL) has proven to be an effective post-training strategy for enhancing reasoning in vision-language models (VLMs). Group Relative Policy Optimization (GRPO) is a recent prominent method that encourages models to generate complete reasoning traces before answering, leading to increased token usage and computational cost. Inspired by the human-like thinking process-where people skip reasoning for easy questions but think carefully when needed-we explore how to enable VLMs to first decide when reasoning is necessary. To realize this, we propose TON, a two-stage training strategy: (i) a supervised fine-tuning (SFT) stage with a simple yet effective 'thought dropout' operation, where reasoning traces are randomly replaced with empty thoughts. This introduces a think-or-not format that serves as a cold start for selective reasoning; (ii) a GRPO stage that enables the model to freely explore when to think or not, while maximizing task-aware outcome rewards. Experimental results show that TON can reduce the completion length by up to 90% compared to vanilla GRPO, without sacrificing performance or even improving it. Further evaluations across diverse vision-language tasks-covering a range of reasoning difficulties under both 3B and 7B models-consistently reveal that the model progressively learns to bypass unnecessary reasoning steps as training advances. These findings shed light on the path toward human-like reasoning patterns in reinforcement learning approaches. Our code is available at https://github.com/kokolerk/TON.
>
---
#### [new 180] Interactive Post-Training for Vision-Language-Action Models
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出基于强化学习的交互式后训练方法RIPT-VLA，用于优化视觉语言动作模型（VLA）。针对现有方法依赖专家数据、适应性差的问题，通过稀疏奖励和动态策略优化，在低数据下显著提升模型性能（如QueST提升21.2%，OpenVLA达97.5%），并实现跨任务泛化。属于VLA模型优化任务，解决新环境适应与数据效率问题。**

- **链接: [http://arxiv.org/pdf/2505.17016v1](http://arxiv.org/pdf/2505.17016v1)**

> **作者:** Shuhan Tan; Kairan Dou; Yue Zhao; Philipp Krähenbühl
>
> **备注:** Project page: https://ariostgx.github.io/ript_vla/
>
> **摘要:** We introduce RIPT-VLA, a simple and scalable reinforcement-learning-based interactive post-training paradigm that fine-tunes pretrained Vision-Language-Action (VLA) models using only sparse binary success rewards. Existing VLA training pipelines rely heavily on offline expert demonstration data and supervised imitation, limiting their ability to adapt to new tasks and environments under low-data regimes. RIPT-VLA addresses this by enabling interactive post-training with a stable policy optimization algorithm based on dynamic rollout sampling and leave-one-out advantage estimation. RIPT-VLA has the following characteristics. First, it applies to various VLA models, resulting in an improvement on the lightweight QueST model by 21.2%, and the 7B OpenVLA-OFT model to an unprecedented 97.5% success rate. Second, it is computationally efficient and data-efficient: with only one demonstration, RIPT-VLA enables an unworkable SFT model (4%) to succeed with a 97% success rate within 15 iterations. Furthermore, we demonstrate that the policy learned by RIPT-VLA generalizes across different tasks and scenarios and is robust to the initial state context. These results highlight RIPT-VLA as a practical and effective paradigm for post-training VLA models through minimal supervision.
>
---
#### [new 181] Benchmarking Chest X-ray Diagnosis Models Across Multinational Datasets
- **分类: eess.IV; cs.AI; cs.CV; I.2**

- **简介: 该研究对比视觉语言基础模型与CNN在多国胸部X光数据集的诊断性能，评估跨人群泛化能力。测试8种模型在6个公开及3个私有数据集，发现基础模型更优（MAVL表现最佳），但儿童病例准确率显著下降，强调结构化提示设计重要性，并建议扩展地域与集成模型提升临床应用。**

- **链接: [http://arxiv.org/pdf/2505.16027v1](http://arxiv.org/pdf/2505.16027v1)**

> **作者:** Qinmei Xu; Yiheng Li; Xianghao Zhan; Ahmet Gorkem Er; Brittany Dashevsky; Chuanjun Xu; Mohammed Alawad; Mengya Yang; Liu Ya; Changsheng Zhou; Xiao Li; Haruka Itakura; Olivier Gevaert
>
> **备注:** 78 pages, 7 figures, 2 tabeles
>
> **摘要:** Foundation models leveraging vision-language pretraining have shown promise in chest X-ray (CXR) interpretation, yet their real-world performance across diverse populations and diagnostic tasks remains insufficiently evaluated. This study benchmarks the diagnostic performance and generalizability of foundation models versus traditional convolutional neural networks (CNNs) on multinational CXR datasets. We evaluated eight CXR diagnostic models - five vision-language foundation models and three CNN-based architectures - across 37 standardized classification tasks using six public datasets from the USA, Spain, India, and Vietnam, and three private datasets from hospitals in China. Performance was assessed using AUROC, AUPRC, and other metrics across both shared and dataset-specific tasks. Foundation models outperformed CNNs in both accuracy and task coverage. MAVL, a model incorporating knowledge-enhanced prompts and structured supervision, achieved the highest performance on public (mean AUROC: 0.82; AUPRC: 0.32) and private (mean AUROC: 0.95; AUPRC: 0.89) datasets, ranking first in 14 of 37 public and 3 of 4 private tasks. All models showed reduced performance on pediatric cases, with average AUROC dropping from 0.88 +/- 0.18 in adults to 0.57 +/- 0.29 in children (p = 0.0202). These findings highlight the value of structured supervision and prompt design in radiologic AI and suggest future directions including geographic expansion and ensemble modeling for clinical deployment. Code for all evaluated models is available at https://drive.google.com/drive/folders/1B99yMQm7bB4h1sVMIBja0RfUu8gLktCE
>
---
#### [new 182] ATR-Bench: A Federated Learning Benchmark for Adaptation, Trust, and Reasoning
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出ATR-Bench框架，针对联邦学习中适应性、可信度和推理三大核心维度，解决其标准化评估缺失的问题。通过系统分析方法、基准测试异构环境下的模型表现，并开源工具促进联邦学习的系统化研究。**

- **链接: [http://arxiv.org/pdf/2505.16850v1](http://arxiv.org/pdf/2505.16850v1)**

> **作者:** Tajamul Ashraf; Mohammed Mohsen Peerzada; Moloud Abdar; Yutong Xie; Yuyin Zhou; Xiaofeng Liu; Iqra Altaf Gillani; Janibul Bashir
>
> **备注:** Federated Learning Benchmark for Domain Adaptation, Trustworthiness, and Reasoning
>
> **摘要:** Federated Learning (FL) has emerged as a promising paradigm for collaborative model training while preserving data privacy across decentralized participants. As FL adoption grows, numerous techniques have been proposed to tackle its practical challenges. However, the lack of standardized evaluation across key dimensions hampers systematic progress and fair comparison of FL methods. In this work, we introduce ATR-Bench, a unified framework for analyzing federated learning through three foundational dimensions: Adaptation, Trust, and Reasoning. We provide an in-depth examination of the conceptual foundations, task formulations, and open research challenges associated with each theme. We have extensively benchmarked representative methods and datasets for adaptation to heterogeneous clients and trustworthiness in adversarial or unreliable environments. Due to the lack of reliable metrics and models for reasoning in FL, we only provide literature-driven insights for this dimension. ATR-Bench lays the groundwork for a systematic and holistic evaluation of federated learning with real-world relevance. We will make our complete codebase publicly accessible and a curated repository that continuously tracks new developments and research in the FL literature.
>
---
#### [new 183] Benchmarking Retrieval-Augmented Multimomal Generation for Document Question Answering
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文属于文档视觉问答（DocVQA）任务，针对现有方法忽视视觉信息及缺乏多模态评估基准的问题，提出MMDocRAG基准，包含4,055个跨模态QA对与证据链，引入多模态评估指标，实验显示专有视觉模型表现更优，多模态输入提升效果显著，为开发多模态问答系统提供测试平台和优化方向。**

- **链接: [http://arxiv.org/pdf/2505.16470v1](http://arxiv.org/pdf/2505.16470v1)**

> **作者:** Kuicai Dong; Yujing Chang; Shijie Huang; Yasheng Wang; Ruiming Tang; Yong Liu
>
> **备注:** preprint. code available at \url{https://mmdocrag.github.io/MMDocRAG/}
>
> **摘要:** Document Visual Question Answering (DocVQA) faces dual challenges in processing lengthy multimodal documents (text, images, tables) and performing cross-modal reasoning. Current document retrieval-augmented generation (DocRAG) methods remain limited by their text-centric approaches, frequently missing critical visual information. The field also lacks robust benchmarks for assessing multimodal evidence selection and integration. We introduce MMDocRAG, a comprehensive benchmark featuring 4,055 expert-annotated QA pairs with multi-page, cross-modal evidence chains. Our framework introduces innovative metrics for evaluating multimodal quote selection and enables answers that interleave text with relevant visual elements. Through large-scale experiments with 60 VLM/LLM models and 14 retrieval systems, we identify persistent challenges in multimodal evidence retrieval, selection, and integration.Key findings reveal advanced proprietary LVMs show superior performance than open-sourced alternatives. Also, they show moderate advantages using multimodal inputs over text-only inputs, while open-source alternatives show significant performance degradation. Notably, fine-tuned LLMs achieve substantial improvements when using detailed image descriptions. MMDocRAG establishes a rigorous testing ground and provides actionable insights for developing more robust multimodal DocVQA systems. Our benchmark and code are available at https://mmdocrag.github.io/MMDocRAG/.
>
---
#### [new 184] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对剪枝大视觉语言模型安全性下降问题，提出分层安全校准（HSR）方法。通过评估注意力头的安全贡献，恢复关键神经元，提升剪枝后模型的安全性，首次实现轻量级安全恢复并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.16104v1](http://arxiv.org/pdf/2505.16104v1)**

> **作者:** Yue Li; Xin Yi; Dongsheng Shi; Gerard de Melo; Xiaoling Wang; Linlin Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** With the increasing size of Large Vision-Language Models (LVLMs), network pruning techniques aimed at compressing models for deployment in resource-constrained environments have garnered significant attention. However, we observe that pruning often leads to a degradation in safety performance. To address this issue, we present a novel and lightweight approach, termed Hierarchical Safety Realignment (HSR). HSR operates by first quantifying the contribution of each attention head to safety, identifying the most critical ones, and then selectively restoring neurons directly within these attention heads that play a pivotal role in maintaining safety. This process hierarchically realigns the safety of pruned LVLMs, progressing from the attention head level to the neuron level. We validate HSR across various models and pruning strategies, consistently achieving notable improvements in safety performance. To our knowledge, this is the first work explicitly focused on restoring safety in LVLMs post-pruning.
>
---
#### [new 185] From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Pedagogical Visualization
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出教育可视化评估基准EduVisBench及多智能体框架EduVisAgent，解决基础模型生成教学可视化效果差的问题。通过构建多领域问题集和协作式智能体系统，提升复杂推理的视觉表达能力，实验显示效果提升40.2%。**

- **链接: [http://arxiv.org/pdf/2505.16832v1](http://arxiv.org/pdf/2505.16832v1)**

> **作者:** Haonian Ji; Shi Qiu; Siyang Xin; Siwei Han; Zhaorun Chen; Hongyi Wang; Dake Zhang; Huaxiu Yao
>
> **备注:** 16 pages; 7 figures
>
> **摘要:** While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations. EduVisBench and EduVisAgent are available at https://github.com/aiming-lab/EduVisBench and https://github.com/aiming-lab/EduVisAgent.
>
---
#### [new 186] An Empirical Study on Configuring In-Context Learning Demonstrations for Unleashing MLLMs' Sentimental Perception Capability
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态情感分析（MSA）任务，旨在解决零样本学习下多模态语言模型（MLLMs）情感感知能力不足的问题。通过研究In-Context Learning（ICL）中演示样本的检索、呈现和分布三个关键因素，优化配置策略并纠正模型情感预测偏差，使准确率较零样本和随机ICL基线分别提升15.9%和11.2%。**

- **链接: [http://arxiv.org/pdf/2505.16193v1](http://arxiv.org/pdf/2505.16193v1)**

> **作者:** Daiqing Wu; Dongbao Yang; Sicheng Zhao; Can Ma; Yu Zhou
>
> **摘要:** The advancements in Multimodal Large Language Models (MLLMs) have enabled various multimodal tasks to be addressed under a zero-shot paradigm. This paradigm sidesteps the cost of model fine-tuning, emerging as a dominant trend in practical application. Nevertheless, Multimodal Sentiment Analysis (MSA), a pivotal challenge in the quest for general artificial intelligence, fails to accommodate this convenience. The zero-shot paradigm exhibits undesirable performance on MSA, casting doubt on whether MLLMs can perceive sentiments as competent as supervised models. By extending the zero-shot paradigm to In-Context Learning (ICL) and conducting an in-depth study on configuring demonstrations, we validate that MLLMs indeed possess such capability. Specifically, three key factors that cover demonstrations' retrieval, presentation, and distribution are comprehensively investigated and optimized. A sentimental predictive bias inherent in MLLMs is also discovered and later effectively counteracted. By complementing each other, the devised strategies for three factors result in average accuracy improvements of 15.9% on six MSA datasets against the zero-shot paradigm and 11.2% against the random ICL baseline.
>
---
#### [new 187] MM-MovieDubber: Towards Multi-Modal Learning for Multi-Modal Movie Dubbing
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于多模态电影配音生成任务，旨在解决现有技术在适配配音风格、处理多元对话类型及捕捉说话人年龄/性别等细节上的不足。提出MM-MovieDubber框架，通过多模态视觉语言模型分析视频识别配音类型与属性，并结合语音生成模型生成高质量配音，同时构建标注数据集提升性能。实验显示其在多个指标上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16279v1](http://arxiv.org/pdf/2505.16279v1)**

> **作者:** Junjie Zheng; Zihao Chen; Chaofan Ding; Yunming Liang; Yihan Fan; Huan Yang; Lei Xie; Xinhan Di
>
> **备注:** 5 pages, 4 figures, accepted by Interspeech 2025
>
> **摘要:** Current movie dubbing technology can produce the desired speech using a reference voice and input video, maintaining perfect synchronization with the visuals while effectively conveying the intended emotions. However, crucial aspects of movie dubbing, including adaptation to various dubbing styles, effective handling of dialogue, narration, and monologues, as well as consideration of subtle details such as speaker age and gender, remain insufficiently explored. To tackle these challenges, we introduce a multi-modal generative framework. First, it utilizes a multi-modal large vision-language model (VLM) to analyze visual inputs, enabling the recognition of dubbing types and fine-grained attributes. Second, it produces high-quality dubbing using large speech generation models, guided by multi-modal inputs. Additionally, a movie dubbing dataset with annotations for dubbing types and subtle details is constructed to enhance movie understanding and improve dubbing quality for the proposed multi-modal framework. Experimental results across multiple benchmark datasets show superior performance compared to state-of-the-art (SOTA) methods. In details, the LSE-D, SPK-SIM, EMO-SIM, and MCD exhibit improvements of up to 1.09%, 8.80%, 19.08%, and 18.74%, respectively.
>
---
#### [new 188] OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出OSCAR方法，解决扩散模型在图像压缩中多步采样效率低、多比特率需多模型的问题。通过将压缩潜码视为扩散过程的中间噪声状态，建立比特率到扩散时间步的映射，实现单模型一阶段高效解码，提升压缩效率与性能。**

- **链接: [http://arxiv.org/pdf/2505.16091v1](http://arxiv.org/pdf/2505.16091v1)**

> **作者:** Jinpei Guo; Yifei Ji; Zheng Chen; Kai Liu; Min Liu; Wang Rao; Wenbo Li; Yong Guo; Yulun Zhang
>
> **摘要:** Pretrained latent diffusion models have shown strong potential for lossy image compression, owing to their powerful generative priors. Most existing diffusion-based methods reconstruct images by iteratively denoising from random noise, guided by compressed latent representations. While these approaches have achieved high reconstruction quality, their multi-step sampling process incurs substantial computational overhead. Moreover, they typically require training separate models for different compression bit-rates, leading to significant training and storage costs. To address these challenges, we propose a one-step diffusion codec across multiple bit-rates. termed OSCAR. Specifically, our method views compressed latents as noisy variants of the original latents, where the level of distortion depends on the bit-rate. This perspective allows them to be modeled as intermediate states along a diffusion trajectory. By establishing a mapping from the compression bit-rate to a pseudo diffusion timestep, we condition a single generative model to support reconstructions at multiple bit-rates. Meanwhile, we argue that the compressed latents retain rich structural information, thereby making one-step denoising feasible. Thus, OSCAR replaces iterative sampling with a single denoising pass, significantly improving inference efficiency. Extensive experiments demonstrate that OSCAR achieves superior performance in both quantitative and visual quality metrics. The code and models will be released at https://github.com/jp-guo/OSCAR.
>
---
## 更新

#### [replaced 001] Motion by Queries: Identity-Motion Trade-offs in Text-to-Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07750v3](http://arxiv.org/pdf/2412.07750v3)**

> **作者:** Yuval Atzmon; Rinon Gal; Yoad Tewel; Yoni Kasten; Gal Chechik
>
> **备注:** (1) Project page: https://research.nvidia.com/labs/par/MotionByQueries/ (2) The methods and results in section 5, "Consistent multi-shot video generation", are based on the arXiv version 1 (v1) of this work. Starting version 2 (v2), we extend and further analyze those findings to efficient motion transfer (3) in v3 we added: results with WAN 2.1, baselines and more quality metrics
>
> **摘要:** Text-to-video diffusion models have shown remarkable progress in generating coherent video clips from textual descriptions. However, the interplay between motion, structure, and identity representations in these models remains under-explored. Here, we investigate how self-attention query (Q) features simultaneously govern motion, structure, and identity and examine the challenges arising when these representations interact. Our analysis reveals that Q affects not only layout, but that during denoising Q also has a strong effect on subject identity, making it hard to transfer motion without the side-effect of transferring identity. Understanding this dual role enabled us to control query feature injection (Q injection) and demonstrate two applications: (1) a zero-shot motion transfer method - implemented with VideoCrafter2 and WAN 2.1 - that is 10 times more efficient than existing approaches, and (2) a training-free technique for consistent multi-shot video generation, where characters maintain identity across multiple video shots while Q injection enhances motion fidelity.
>
---
#### [replaced 002] UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14671v2](http://arxiv.org/pdf/2505.14671v2)**

> **作者:** Ruichuan An; Sihan Yang; Renrui Zhang; Zijun Shen; Ming Lu; Gaole Dai; Hao Liang; Ziyu Guo; Shilin Yan; Yulin Luo; Bocheng Zou; Chaoqun Yang; Wentao Zhang
>
> **摘要:** Personalized models have demonstrated remarkable success in understanding and generating concepts provided by users. However, existing methods use separate concept tokens for understanding and generation, treating these tasks in isolation. This may result in limitations for generating images with complex prompts. For example, given the concept $\langle bo\rangle$, generating "$\langle bo\rangle$ wearing its hat" without additional textual descriptions of its hat. We call this kind of generation personalized knowledge-driven generation. To address the limitation, we present UniCTokens, a novel framework that effectively integrates personalized information into a unified vision language model (VLM) for understanding and generation. UniCTokens trains a set of unified concept tokens to leverage complementary semantics, boosting two personalized tasks. Moreover, we propose a progressive training strategy with three stages: understanding warm-up, bootstrapping generation from understanding, and deepening understanding from generation to enhance mutual benefits between both tasks. To quantitatively evaluate the unified VLM personalization, we present UnifyBench, the first benchmark for assessing concept understanding, concept generation, and knowledge-driven generation. Experimental results on UnifyBench indicate that UniCTokens shows competitive performance compared to leading methods in concept understanding, concept generation, and achieving state-of-the-art results in personalized knowledge-driven generation. Our research demonstrates that enhanced understanding improves generation, and the generation process can yield valuable insights into understanding. Our code and dataset will be released at: \href{https://github.com/arctanxarc/UniCTokens}{https://github.com/arctanxarc/UniCTokens}.
>
---
#### [replaced 003] Fast computation of the TGOSPA metric for multiple target tracking via unbalanced optimal transport
- **分类: math.OC; cs.CV; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.09449v2](http://arxiv.org/pdf/2503.09449v2)**

> **作者:** Viktor Nevelius Wernholm; Alfred Wärnsäter; Axel Ringh
>
> **备注:** 6 pages, 3 figures. Revision
>
> **摘要:** In multiple target tracking, it is important to be able to evaluate the performance of different tracking algorithms. The trajectory generalized optimal sub-pattern assignment metric (TGOSPA) is a recently proposed metric for such evaluations. The TGOSPA metric is computed as the solution to an optimization problem, but for large tracking scenarios, solving this problem becomes computationally demanding. In this paper, we present an approximation algorithm for evaluating the TGOSPA metric, based on casting the TGOSPA problem as an unbalanced multimarginal optimal transport problem. Following recent advances in computational optimal transport, we introduce an entropy regularization and derive an iterative scheme for solving the Lagrangian dual of the regularized problem. Numerical results suggest that our proposed algorithm is more computationally efficient than the alternative of computing the exact metric using a linear programming solver, while still providing an adequate approximation of the metric.
>
---
#### [replaced 004] Learning Joint ID-Textual Representation for ID-Preserving Image Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.14202v3](http://arxiv.org/pdf/2504.14202v3)**

> **作者:** Zichuan Liu; Liming Jiang; Qing Yan; Yumin Jia; Hao Kang; Xin Lu
>
> **摘要:** We propose a novel framework for ID-preserving generation using a multi-modal encoding strategy rather than injecting identity features via adapters into pre-trained models. Our method treats identity and text as a unified conditioning input. To achieve this, we introduce FaceCLIP, a multi-modal encoder that learns a joint embedding space for both identity and textual semantics. Given a reference face and a text prompt, FaceCLIP produces a unified representation that encodes both identity and text, which conditions a base diffusion model to generate images that are identity-consistent and text-aligned. We also present a multi-modal alignment algorithm to train FaceCLIP, using a loss that aligns its joint representation with face, text, and image embedding spaces. We then build FaceCLIP-SDXL, an ID-preserving image synthesis pipeline by integrating FaceCLIP with Stable Diffusion XL (SDXL). Compared to prior methods, FaceCLIP-SDXL enables photorealistic portrait generation with better identity preservation and textual relevance. Extensive experiments demonstrate its quantitative and qualitative superiority.
>
---
#### [replaced 005] EDM: Efficient Deep Feature Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05122v2](http://arxiv.org/pdf/2503.05122v2)**

> **作者:** Xi Li; Tong Rao; Cihui Pan
>
> **摘要:** Recent feature matching methods have achieved remarkable performance but lack efficiency consideration. In this paper, we revisit the mainstream detector-free matching pipeline and improve all its stages considering both accuracy and efficiency. We propose an Efficient Deep feature Matching network, EDM. We first adopt a deeper CNN with fewer dimensions to extract multi-level features. Then we present a Correlation Injection Module that conducts feature transformation on high-level deep features, and progressively injects feature correlations from global to local for efficient multi-scale feature aggregation, improving both speed and performance. In the refinement stage, a novel lightweight bidirectional axis-based regression head is designed to directly predict subpixel-level correspondences from latent features, avoiding the significant computational cost of explicitly locating keypoints on high-resolution local feature heatmaps. Moreover, effective selection strategies are introduced to enhance matching accuracy. Extensive experiments show that our EDM achieves competitive matching accuracy on various benchmarks and exhibits excellent efficiency, offering valuable best practices for real-world applications. The code is available at https://github.com/chicleee/EDM.
>
---
#### [replaced 006] Supervising 3D Talking Head Avatars with Analysis-by-Audio-Synthesis
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13386v2](http://arxiv.org/pdf/2504.13386v2)**

> **作者:** Radek Daněček; Carolin Schmitt; Senya Polikovsky; Michael J. Black
>
> **摘要:** In order to be widely applicable, speech-driven 3D head avatars must articulate their lips in accordance with speech, while also conveying the appropriate emotions with dynamically changing facial expressions. The key problem is that deterministic models produce high-quality lip-sync but without rich expressions, whereas stochastic models generate diverse expressions but with lower lip-sync quality. To get the best of both, we seek a stochastic model with accurate lip-sync. To that end, we develop a new approach based on the following observation: if a method generates realistic 3D lip motions, it should be possible to infer the spoken audio from the lip motion. The inferred speech should match the original input audio, and erroneous predictions create a novel supervision signal for training 3D talking head avatars with accurate lip-sync. To demonstrate this effect, we propose THUNDER (Talking Heads Under Neural Differentiable Elocution Reconstruction), a 3D talking head avatar framework that introduces a novel supervision mechanism via differentiable sound production. First, we train a novel mesh-to-speech model that regresses audio from facial animation. Then, we incorporate this model into a diffusion-based talking avatar framework. During training, the mesh-to-speech model takes the generated animation and produces a sound that is compared to the input speech, creating a differentiable analysis-by-audio-synthesis supervision loop. Our extensive qualitative and quantitative experiments demonstrate that THUNDER significantly improves the quality of the lip-sync of talking head avatars while still allowing for generation of diverse, high-quality, expressive facial animations. The code and models will be available at https://thunder.is.tue.mpg.de/
>
---
#### [replaced 007] Objective Bicycle Occlusion Level Classification using a Deformable Parts-Based Model
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15358v2](http://arxiv.org/pdf/2505.15358v2)**

> **作者:** Angelique Mangubat; Shane Gilroy
>
> **摘要:** Road safety is a critical challenge, particularly for cyclists, who are among the most vulnerable road users. This study aims to enhance road safety by proposing a novel benchmark for bicycle occlusion level classification using advanced computer vision techniques. Utilizing a parts-based detection model, images are annotated and processed through a custom image detection pipeline. A novel method of bicycle occlusion level is proposed to objectively quantify the visibility and occlusion level of bicycle semantic parts. The findings indicate that the model robustly quantifies the visibility and occlusion level of bicycles, a significant improvement over the subjective methods used by the current state of the art. Widespread use of the proposed methodology will facilitate accurate performance reporting of cyclist detection algorithms for occluded cyclists, informing the development of more robust vulnerable road user detection methods for autonomous vehicles.
>
---
#### [replaced 008] TWIG: Two-Step Image Generation using Segmentation Masks in Diffusion Models
- **分类: cs.CV; 68T07, 68U10, 68T45**

- **链接: [http://arxiv.org/pdf/2504.14933v2](http://arxiv.org/pdf/2504.14933v2)**

> **作者:** Mazharul Islam Rakib; Showrin Rahman; Joyanta Jyoti Mondal; Xi Xiao; David Lewis; Alessandra Mileo; Meem Arafat Manab
>
> **备注:** 16 pages, 9 figures, published to IFIP International Summer School on Privacy and Identity Management
>
> **摘要:** In today's age of social media and marketing, copyright issues can be a major roadblock to the free sharing of images. Generative AI models have made it possible to create high-quality images, but concerns about copyright infringement are a hindrance to their abundant use. As these models use data from training images to generate new ones, it is often a daunting task to ensure they do not violate intellectual property rights. Some AI models have even been noted to directly copy copyrighted images, a problem often referred to as source copying. Traditional copyright protection measures such as watermarks and metadata have also proven to be futile in this regard. To address this issue, we propose a novel two-step image generation model inspired by the conditional diffusion model. The first step involves creating an image segmentation mask for some prompt-based generated images. This mask embodies the shape of the image. Thereafter, the diffusion model is asked to generate the image anew while avoiding the shape in question. This approach shows a decrease in structural similarity from the training image, i.e. we are able to avoid the source copying problem using this approach without expensive retraining of the model or user-centered prompt generation techniques. This makes our approach the most computationally inexpensive approach to avoiding both copyright infringement and source copying for diffusion model-based image generation.
>
---
#### [replaced 009] Advances in Radiance Field for Dynamic Scene: From Neural Field to Gaussian Field
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10049v2](http://arxiv.org/pdf/2505.10049v2)**

> **作者:** Jinlong Fan; Xuepu Zeng; Jing Zhang; Mingming Gong; Yuxiang Yang; Dacheng Tao
>
> **摘要:** Dynamic scene representation and reconstruction have undergone transformative advances in recent years, catalyzed by breakthroughs in neural radiance fields and 3D Gaussian splatting techniques. While initially developed for static environments, these methodologies have rapidly evolved to address the complexities inherent in 4D dynamic scenes through an expansive body of research. Coupled with innovations in differentiable volumetric rendering, these approaches have significantly enhanced the quality of motion representation and dynamic scene reconstruction, thereby garnering substantial attention from the computer vision and graphics communities. This survey presents a systematic analysis of over 200 papers focused on dynamic scene representation using radiance field, spanning the spectrum from implicit neural representations to explicit Gaussian primitives. We categorize and evaluate these works through multiple critical lenses: motion representation paradigms, reconstruction techniques for varied scene dynamics, auxiliary information integration strategies, and regularization approaches that ensure temporal consistency and physical plausibility. We organize diverse methodological approaches under a unified representational framework, concluding with a critical examination of persistent challenges and promising research directions. By providing this comprehensive overview, we aim to establish a definitive reference for researchers entering this rapidly evolving field while offering experienced practitioners a systematic understanding of both conceptual principles and practical frontiers in dynamic scene reconstruction.
>
---
#### [replaced 010] Stronger ViTs With Octic Equivariance
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15441v2](http://arxiv.org/pdf/2505.15441v2)**

> **作者:** David Nordström; Johan Edstedt; Fredrik Kahl; Georg Bökman
>
> **摘要:** Recent efforts at scaling computer vision models have established Vision Transformers (ViTs) as the leading architecture. ViTs incorporate weight sharing over image patches as an important inductive bias. In this work, we show that ViTs benefit from incorporating equivariance under the octic group, i.e., reflections and 90-degree rotations, as a further inductive bias. We develop new architectures, octic ViTs, that use octic-equivariant layers and put them to the test on both supervised and self-supervised learning. Through extensive experiments on DeiT-III and DINOv2 training on ImageNet-1K, we show that octic ViTs yield more computationally efficient networks while also improving performance. In particular, we achieve approximately 40% reduction in FLOPs for ViT-H while simultaneously improving both classification and segmentation results.
>
---
#### [replaced 011] KAN-Mamba FusionNet: Redefining Medical Image Segmentation with Non-Linear Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.11926v2](http://arxiv.org/pdf/2411.11926v2)**

> **作者:** Akansh Agrawal; Akshan Agrawal; Shashwat Gupta; Priyanka Bagade
>
> **备注:** 11 pages, 2 figures, 4 tables
>
> **摘要:** Medical image segmentation is essential for applications like robotic surgeries, disease diagnosis, and treatment planning. Recently, various deep-learning models have been proposed to enhance medical image segmentation. One promising approach utilizes Kolmogorov-Arnold Networks (KANs), which better capture non-linearity in input data. However, they are unable to effectively capture long-range dependencies, which are required to accurately segment complex medical images and, by that, improve diagnostic accuracy in clinical settings. Neural networks such as Mamba can handle long-range dependencies. However, they have a limited ability to accurately capture non-linearities in the images as compared to KANs. Thus, we propose a novel architecture, the KAN-Mamba FusionNet, which improves segmentation accuracy by effectively capturing the non-linearities from input and handling long-range dependencies with the newly proposed KAMBA block. We evaluated the proposed KAN-Mamba FusionNet on three distinct medical image segmentation datasets: BUSI, Kvasir-Seg, and GlaS - and found it consistently outperforms state-of-the-art methods in IoU and F1 scores. Further, we examined the effects of various components and assessed their contributions to the overall model performance via ablation studies. The findings highlight the effectiveness of this methodology for reliable medical image segmentation, providing a unique approach to address intricate visual data issues in healthcare.
>
---
#### [replaced 012] Transferring Textual Preferences to Vision-Language Understanding through Model Merging
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13487v2](http://arxiv.org/pdf/2502.13487v2)**

> **作者:** Chen-An Li; Tzu-Han Lin; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Accepted to ACL 2025 main
>
> **摘要:** Large vision-language models (LVLMs) perform outstandingly across various multimodal tasks. However, their ability to evaluate generated content remains limited, and training vision-language reward models (VLRMs) with preference data is computationally expensive. This paper explores a training-free alternative by merging text-based reward models (RMs) with LVLMs to create VLRMs. Our approach shows that integrating these models leads to improved performance over LVLMs' scoring and text-based RMs, offering an efficient method for incorporating textual preferences into LVLMs.
>
---
#### [replaced 013] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.14716v2](http://arxiv.org/pdf/2411.14716v2)**

> **作者:** Haiming Zhang; Wending Zhou; Yiyao Zhu; Xu Yan; Jiantao Gao; Dongfeng Bai; Yingjie Cai; Bingbing Liu; Shuguang Cui; Zhen Li
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** This paper introduces VisionPAD, a novel self-supervised pre-training paradigm designed for vision-centric algorithms in autonomous driving. In contrast to previous approaches that employ neural rendering with explicit depth supervision, VisionPAD utilizes more efficient 3D Gaussian Splatting to reconstruct multi-view representations using only images as supervision. Specifically, we introduce a self-supervised method for voxel velocity estimation. By warping voxels to adjacent frames and supervising the rendered outputs, the model effectively learns motion cues in the sequential data. Furthermore, we adopt a multi-frame photometric consistency approach to enhance geometric perception. It projects adjacent frames to the current frame based on rendered depths and relative poses, boosting the 3D geometric representation through pure image supervision. Extensive experiments on autonomous driving datasets demonstrate that VisionPAD significantly improves performance in 3D object detection, occupancy prediction and map segmentation, surpassing state-of-the-art pre-training strategies by a considerable margin.
>
---
#### [replaced 014] Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01222v2](http://arxiv.org/pdf/2503.01222v2)**

> **作者:** Wenbin Wang; Yongcheng Jing; Liang Ding; Yingjie Wang; Li Shen; Yong Luo; Bo Du; Dacheng Tao
>
> **摘要:** High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To overcome the limitations of existing methods, this paper shifts away from prior dedicated heuristic approaches and revisits the most fundamental idea to HR perception by enhancing the long-context capability of MLLMs, driven by recent advances in long-context techniques like retrieval-augmented generation (RAG) for general LLMs. Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43% improvement on $V^*$ Bench and 19% on HR-Bench.
>
---
#### [replaced 015] Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14642v2](http://arxiv.org/pdf/2504.14642v2)**

> **作者:** Lin Li; Wei Chen; Jiahui Li; Kwang-Ting Cheng; Long Chen
>
> **备注:** Ongoing project
>
> **摘要:** Recent advances in multi-modal large language models (MLLMs) have significantly improved object-level grounding and region captioning. However, they remain limited in visual relation understanding, struggling even with binary relation detection, let alone \textit{N}-ary relations involving multiple semantic roles. The core reason is the lack of modeling for \textit{structural semantic dependencies} among multi-entities, leading to unreliable outputs, hallucinations, and over-reliance on language priors (\eg, defaulting to ``person drinks a milk'' if a person is merely holding it). To this end, we propose Relation-R1, the \textit{first unified} relation comprehension framework that explicitly integrates cognitive chain-of-thought (CoT)-guided supervised fine-tuning (SFT) and group relative policy optimization (GRPO) within a reinforcement learning (RL) paradigm. Specifically, we first establish foundational reasoning capabilities via SFT, enforcing structured outputs with thinking processes. Then, GRPO is utilized to refine these outputs via multi-rewards optimization, prioritizing visual-semantic grounding over language-induced biases, thereby improving generalization capability. Furthermore, we investigate the impact of various CoT strategies within this framework, demonstrating that a specific-to-general progressive approach in CoT guidance further improves generalization, especially in capturing synonymous \textit{N}-ary relations. Extensive experiments on widely-used PSG and SWiG datasets demonstrate that Relation-R1 achieves state-of-the-art performance in both binary and \textit{N}-ary relation understanding.
>
---
#### [replaced 016] Benchmarking Ophthalmology Foundation Models for Clinically Significant Age Macular Degeneration Detection
- **分类: eess.IV; cs.AI; cs.CV; q-bio.TO**

- **链接: [http://arxiv.org/pdf/2505.05291v2](http://arxiv.org/pdf/2505.05291v2)**

> **作者:** Benjamin A. Cohen; Jonathan Fhima; Meishar Meisel; Baskin Meital; Luis Filipe Nakayama; Eran Berkowitz; Joachim A. Behar
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Self-supervised learning (SSL) has enabled Vision Transformers (ViTs) to learn robust representations from large-scale natural image datasets, enhancing their generalization across domains. In retinal imaging, foundation models pretrained on either natural or ophthalmic data have shown promise, but the benefits of in-domain pretraining remain uncertain. To investigate this, we benchmark six SSL-pretrained ViTs on seven digital fundus image (DFI) datasets totaling 70,000 expert-annotated images for the task of moderate-to-late age-related macular degeneration (AMD) identification. Our results show that iBOT pretrained on natural images achieves the highest out-of-distribution generalization, with AUROCs of 0.80-0.97, outperforming domain-specific models, which achieved AUROCs of 0.78-0.96 and a baseline ViT-L with no pretraining, which achieved AUROCs of 0.68-0.91. These findings highlight the value of foundation models in improving AMD identification and challenge the assumption that in-domain pretraining is necessary. Furthermore, we release BRAMD, an open-access dataset (n=587) of DFIs with AMD labels from Brazil.
>
---
#### [replaced 017] MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09499v2](http://arxiv.org/pdf/2503.09499v2)**

> **作者:** Zhe Xu; Daoyuan Chen; Zhenqing Ling; Yaliang Li; Ying Shen
>
> **备注:** 22 pages, 7 tables
>
> **摘要:** Large foundation models face challenges in acquiring transferable, structured thinking abilities, especially when supervised with rigid templates or crowd-annotated instruction datasets. Unlike prior approaches, we focus on a thinking-centric data synthesis paradigm that enables models to evolve through self-generated, cognitively guided data. We propose MindGYM, a structured and scalable framework for question synthesis, composed of: (1) Cognitive Thinking Process Injection, which infuses high-level reasoning objectives to shape the model's synthesis behavior; (2) Seed Single-Hop Question Synthesis, generating atomic questions from diverse semantic types to encourage broader thinking; and (3) Challenging Multi-Hop QA Synthesis, composing more complex multi-hop questions based on QA seeds for deeper reasoning. Detailed analysis shows that synthetic data generated by our method achieves 16.7% higher average quality and 67.91% lower quality variance compared to baseline sources, highlighting that both high-quality and self-contained data are essential for effective, thinking-oriented fine-tuning. MindGYM improves performance on six reasoning benchmarks, achieving gains of up to 16% on MathVision using only 400 data samples, and generalizable improvements across different model sizes and architectures. MindGYM underscores the viability of self-challenging mechanisms in refining large model capabilities while minimizing human intervention and resource demands. Code and data are released to promote data-centric research into self-evolving foundation models driven by their internal reasoning capabilities.
>
---
#### [replaced 018] Strengthening Generative Robot Policies through Predictive World Modeling
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00622v2](http://arxiv.org/pdf/2502.00622v2)**

> **作者:** Han Qi; Haocheng Yin; Aris Zhu; Yilun Du; Heng Yang
>
> **备注:** Website: https://computationalrobotics.seas.harvard.edu/GPC
>
> **摘要:** We present generative predictive control (GPC), a learning control framework that (i) clones a generative diffusion-based policy from expert demonstrations, (ii) trains a predictive action-conditioned world model from both expert demonstrations and random explorations, and (iii) synthesizes an online planner that ranks and optimizes the action proposals from (i) by looking ahead into the future using the world model from (ii). Across a variety of robotic manipulation tasks, we demonstrate that GPC consistently outperforms behavior cloning in both state-based and vision-based settings, in simulation and in the real world.
>
---
#### [replaced 019] Persistence-based Hough Transform for Line Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16114v2](http://arxiv.org/pdf/2504.16114v2)**

> **作者:** Johannes Ferner; Stefan Huber; Saverio Messineo; Angel Pop; Martin Uray
>
> **备注:** Accepted at iDSC'25, Salzburg, Austria
>
> **摘要:** The Hough transform is a popular and classical technique in computer vision for the detection of lines (or more general objects). It maps a pixel into a dual space -- the Hough space: each pixel is mapped to the set of lines through this pixel, which forms a curve in Hough space. The detection of lines then becomes a voting process to find those lines that received many votes by pixels. However, this voting is done by thresholding, which is susceptible to noise and other artifacts. In this work, we present an alternative voting technique to detect peaks in the Hough space based on persistent homology, which very naturally addresses limitations of simple thresholding. Experiments on synthetic data show that our method significantly outperforms the original method, while also demonstrating enhanced robustness. This work seeks to inspire future research in two key directions. First, we highlight the untapped potential of Topological Data Analysis techniques and advocate for their broader integration into existing methods, including well-established ones. Secondly, we initiate a discussion on the mathematical stability of the Hough transform, encouraging exploration of mathematically grounded improvements to enhance its robustness.
>
---
#### [replaced 020] Generative Pre-trained Autoregressive Diffusion Transformer
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07344v4](http://arxiv.org/pdf/2505.07344v4)**

> **作者:** Yuan Zhang; Jiacheng Jiang; Guoqing Ma; Zhiying Lu; Haoyang Huang; Jianlong Yuan; Nan Duan
>
> **摘要:** In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space.
>
---
#### [replaced 021] GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15810v2](http://arxiv.org/pdf/2505.15810v2)**

> **作者:** Yuqi Zhou; Sunhao Dai; Shuai Wang; Kaiwen Zhou; Qinglin Jia; Jun Xu
>
> **摘要:** Recent Graphical User Interface (GUI) agents replicate the R1-Zero paradigm, coupling online Reinforcement Learning (RL) with explicit chain-of-thought reasoning prior to object grounding and thereby achieving substantial performance gains. In this paper, we first conduct extensive analysis experiments of three key components of that training pipeline: input design, output evaluation, and policy update-each revealing distinct challenges arising from blindly applying general-purpose RL without adapting to GUI grounding tasks. Input design: Current templates encourage the model to generate chain-of-thought reasoning, but longer chains unexpectedly lead to worse grounding performance. Output evaluation: Reward functions based on hit signals or box area allow models to exploit box size, leading to reward hacking and poor localization quality. Policy update: Online RL tends to overfit easy examples due to biases in length and sample difficulty, leading to under-optimization on harder cases. To address these issues, we propose three targeted solutions. First, we adopt a Fast Thinking Template that encourages direct answer generation, reducing excessive reasoning during training. Second, we incorporate a box size constraint into the reward function to mitigate reward hacking. Third, we revise the RL objective by adjusting length normalization and adding a difficulty-aware scaling factor, enabling better optimization on hard samples. Our GUI-G1-3B, trained on 17K public samples with Qwen2.5-VL-3B-Instruct, achieves 90.3% accuracy on ScreenSpot and 37.1% on ScreenSpot-Pro. This surpasses all prior models of similar size and even outperforms the larger UI-TARS-7B, establishing a new state-of-the-art in GUI agent grounding. The project repository is available at https://github.com/Yuqi-Zhou/GUI-G1.
>
---
#### [replaced 022] Leveraging Large Language Models For Scalable Vector Graphics Processing: A Review
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04983v2](http://arxiv.org/pdf/2503.04983v2)**

> **作者:** Boris Malashenko; Ivan Jarsky; Valeria Efimova
>
> **摘要:** In recent years, rapid advances in computer vision have significantly improved the processing and generation of raster images. However, vector graphics, which is essential in digital design, due to its scalability and ease of editing, have been relatively understudied. Traditional vectorization techniques, which are often used in vector generation, suffer from long processing times and excessive output complexity, limiting their usability in practical applications. The advent of large language models (LLMs) has opened new possibilities for the generation, editing, and analysis of vector graphics, particularly in the SVG format, which is inherently text-based and well-suited for integration with LLMs. This paper provides a systematic review of existing LLM-based approaches for SVG processing, categorizing them into three main tasks: generation, editing, and understanding. We observe notable models such as IconShop, StrokeNUWA, and StarVector, highlighting their strengths and limitations. Furthermore, we analyze benchmark datasets designed for assessing SVG-related tasks, including SVGEditBench, VGBench, and SGP-Bench, and conduct a series of experiments to evaluate various LLMs in these domains. Our results demonstrate that for vector graphics reasoning-enhanced models outperform standard LLMs, particularly in generation and understanding tasks. Furthermore, our findings underscore the need to develop more diverse and richly annotated datasets to further improve LLM capabilities in vector graphics tasks.
>
---
#### [replaced 023] L2RDaS: Synthesizing 4D Radar Tensors for Model Generalization via Dataset Expansion
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.03637v2](http://arxiv.org/pdf/2503.03637v2)**

> **作者:** Woo-Jin Jung; Dong-Hee Paek; Seung-Hyun Kong
>
> **备注:** 9 pages, 3 figures, Arxiv preprint
>
> **摘要:** 4-dimensional (4D) radar is increasingly adopted in autonomous driving for perception tasks, owing to its robustness under adverse weather conditions. To better utilize the spatial information inherent in 4D radar data, recent deep learning methods have transitioned from using sparse point cloud to 4D radar tensors. However, the scarcity of publicly available 4D radar tensor datasets limits model generalization across diverse driving scenarios. Previous methods addressed this by synthesizing radar data, but the outputs did not fully exploit the spatial information characteristic of 4D radar. To overcome these limitations, we propose LiDAR-to-4D radar data synthesis (L2RDaS), a framework that synthesizes spatially informative 4D radar tensors from LiDAR data available in existing autonomous driving datasets. L2RDaS integrates a modified U-Net architecture to effectively capture spatial information and an object information supplement (OBIS) module to enhance reflection fidelity. This framework enables the synthesis of radar tensors across diverse driving scenarios without additional sensor deployment or data collection. L2RDaS improves model generalization by expanding real datasets with synthetic radar tensors, achieving an average increase of 4.25\% in ${{AP}_{BEV}}$ and 2.87\% in ${{AP}_{3D}}$ across three detection models. Additionally, L2RDaS supports ground-truth augmentation (GT-Aug) by embedding annotated objects into LiDAR data and synthesizing them into radar tensors, resulting in further average increases of 3.75\% in ${{AP}_{BEV}}$ and 4.03\% in ${{AP}_{3D}}$. The implementation will be available at https://github.com/kaist-avelab/K-Radar.
>
---
#### [replaced 024] GOTPR: General Outdoor Text-based Place Recognition Using Scene Graph Retrieval with OpenStreetMap
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08575v2](http://arxiv.org/pdf/2501.08575v2)**

> **作者:** Donghwi Jung; Keonwoo Kim; Seong-Woo Kim
>
> **摘要:** We propose GOTPR, a robust place recognition method designed for outdoor environments where GPS signals are unavailable. Unlike existing approaches that use point cloud maps, which are large and difficult to store, GOTPR leverages scene graphs generated from text descriptions and maps for place recognition. This method improves scalability by replacing point clouds with compact data structures, allowing robots to efficiently store and utilize extensive map data. In addition, GOTPR eliminates the need for custom map creation by using publicly available OpenStreetMap data, which provides global spatial information. We evaluated its performance using the KITTI360Pose dataset with corresponding OpenStreetMap data, comparing it to existing point cloud-based place recognition methods. The results show that GOTPR achieves comparable accuracy while significantly reducing storage requirements. In city-scale tests, it completed processing within a few seconds, making it highly practical for real-world robotics applications. More information can be found at https://donghwijung.github.io/GOTPR_page/.
>
---
#### [replaced 025] Auto-Prompting SAM for Weakly Supervised Landslide Extraction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13426v2](http://arxiv.org/pdf/2501.13426v2)**

> **作者:** Jian Wang; Xiaokang Zhang; Xianping Ma; Weikang Yu; Pedram Ghamisi
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Weakly supervised landslide extraction aims to identify landslide regions from remote sensing data using models trained with weak labels, particularly image-level labels. However, it is often challenged by the imprecise boundaries of the extracted objects due to the lack of pixel-wise supervision and the properties of landslide objects. To tackle these issues, we propose a simple yet effective method by auto-prompting the Segment Anything Model (SAM), i.e., APSAM. Instead of depending on high-quality class activation maps (CAMs) for pseudo-labeling or fine-tuning SAM, our method directly yields fine-grained segmentation masks from SAM inference through prompt engineering. Specifically, it adaptively generates hybrid prompts from the CAMs obtained by an object localization network. To provide sufficient information for SAM prompting, an adaptive prompt generation (APG) algorithm is designed to fully leverage the visual patterns of CAMs, enabling the efficient generation of pseudo-masks for landslide extraction. These informative prompts are able to identify the extent of landslide areas (box prompts) and denote the centers of landslide objects (point prompts), guiding SAM in landslide segmentation. Experimental results on high-resolution aerial and satellite datasets demonstrate the effectiveness of our method, achieving improvements of at least 3.0\% in F1 score and 3.69\% in IoU compared to other state-of-the-art methods. The source codes and datasets will be available at https://github.com/zxk688.
>
---
#### [replaced 026] Progressive Local Alignment for Medical Multimodal Pre-training
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.18047v2](http://arxiv.org/pdf/2502.18047v2)**

> **作者:** Huimin Yan; Xian Yang; Liang Bai; Jiye Liang
>
> **备注:** We are currently revising the methodology described in the manuscript to improve its clarity. We have decided to withdraw the current version until a more robust and complete version is ready
>
> **摘要:** Local alignment between medical images and text is essential for accurate diagnosis, though it remains challenging due to the absence of natural local pairings and the limitations of rigid region recognition methods. Traditional approaches rely on hard boundaries, which introduce uncertainty, whereas medical imaging demands flexible soft region recognition to handle irregular structures. To overcome these challenges, we propose the Progressive Local Alignment Network (PLAN), which designs a novel contrastive learning-based approach for local alignment to establish meaningful word-pixel relationships and introduces a progressive learning strategy to iteratively refine these relationships, enhancing alignment precision and robustness. By combining these techniques, PLAN effectively improves soft region recognition while suppressing noise interference. Extensive experiments on multiple medical datasets demonstrate that PLAN surpasses state-of-the-art methods in phrase grounding, image-text retrieval, object detection, and zero-shot classification, setting a new benchmark for medical image-text alignment.
>
---
#### [replaced 027] VITAL: Interactive Few-Shot Imitation Learning via Visual Human-in-the-Loop Corrections
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.21244v2](http://arxiv.org/pdf/2407.21244v2)**

> **作者:** Hamidreza Kasaei; Mohammadreza Kasaei
>
> **摘要:** Imitation Learning (IL) has emerged as a powerful approach in robotics, allowing robots to acquire new skills by mimicking human actions. Despite its potential, the data collection process for IL remains a significant challenge due to the logistical difficulties and high costs associated with obtaining high-quality demonstrations. To address these issues, we propose a large-scale data generation from a handful of demonstrations through data augmentation in simulation. Our approach leverages affordable hardware and visual processing techniques to collect demonstrations, which are then augmented to create extensive training datasets for imitation learning. By utilizing both real and simulated environments, along with human-in-the-loop corrections, we enhance the generalizability and robustness of the learned policies. We evaluated our method through several rounds of experiments in both simulated and real-robot settings, focusing on tasks of varying complexity, including bottle collecting, stacking objects, and hammering. Our experimental results validate the effectiveness of our approach in learning robust robot policies from simulated data, significantly improved by human-in-the-loop corrections and real-world data integration. Additionally, we demonstrate the framework's capability to generalize to new tasks, such as setting a drink tray, showcasing its adaptability and potential for handling a wide range of real-world manipulation tasks. A video of the experiments can be found at: https://youtu.be/YeVAMRqRe64?si=R179xDlEGc7nPu8i
>
---
#### [replaced 028] Top-Down Compression: Revisit Efficient Vision Token Projection for Visual Instruction Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11945v2](http://arxiv.org/pdf/2505.11945v2)**

> **作者:** Bonan li; Zicheng Zhang; Songhua Liu; Weihao Yu; Xinchao Wang
>
> **备注:** Under Review
>
> **摘要:** Visual instruction tuning aims to enable large language models to comprehend the visual world, with a pivotal challenge lying in establishing an effective vision-to-language projection. However, existing methods often grapple with the intractable trade-off between accuracy and efficiency. In this paper, we present LLaVA-Meteor, a novel approach designed to break this deadlock, equipped with a novel Top-Down Compression paradigm that strategically compresses visual tokens without compromising core information. Specifically, we construct a trainable Flash Global Fusion module based on efficient selective state space operators, which aligns the feature space while enabling each token to perceive holistic visual context and instruction preference at low cost. Furthermore, a local-to-single scanning manner is employed to effectively capture local dependencies, thereby enhancing the model's capability in vision modeling. To alleviate computational overhead, we explore a Visual-Native Selection mechanism that independently assesses token significance by both the visual and native experts, followed by aggregation to retain the most critical subset. Extensive experiments show that our approach reduces visual tokens by 75--95% while achieving comparable or superior performance across 12 benchmarks, significantly improving efficiency.
>
---
#### [replaced 029] GeoBiked: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.17045v2](http://arxiv.org/pdf/2409.17045v2)**

> **作者:** Phillip Mueller; Sebastian Mueller; Lars Mikelsons
>
> **摘要:** We provide a dataset for enabling Deep Generative Models (DGMs) in engineering design and propose methods to automate data labeling by utilizing large-scale foundation models. GeoBiked is curated to contain 4 355 bicycle images, annotated with structural and technical features and is used to investigate two automated labeling techniques: The utilization of consolidated latent features (Hyperfeatures) from image-generation models to detect geometric correspondences (e.g. the position of the wheel center) in structural images and the generation of diverse text descriptions for structural images. GPT-4o, a vision-language-model (VLM), is instructed to analyze images and produce diverse descriptions aligned with the system-prompt. By representing technical images as Diffusion-Hyperfeatures, drawing geometric correspondences between them is possible. The detection accuracy of geometric points in unseen samples is improved by presenting multiple annotated source images. GPT-4o has sufficient capabilities to generate accurate descriptions of technical images. Grounding the generation only on images leads to diverse descriptions but causes hallucinations, while grounding it on categorical labels restricts the diversity. Using both as input balances creativity and accuracy. Successfully using Hyperfeatures for geometric correspondence suggests that this approach can be used for general point-detection and annotation tasks in technical images. Labeling such images with text descriptions using VLMs is possible, but dependent on the models detection capabilities, careful prompt-engineering and the selection of input information. Applying foundation models in engineering design is largely unexplored. We aim to bridge this gap with a dataset to explore training, finetuning and conditioning DGMs in this field and suggesting approaches to bootstrap foundation models to process technical images.
>
---
#### [replaced 030] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15298v2](http://arxiv.org/pdf/2505.15298v2)**

> **作者:** Kangan Qian; Sicong Jiang; Yang Zhong; Ziang Luo; Zilin Huang; Tianze Zhu; Kun Jiang; Mengmeng Yang; Zheng Fu; Jinyu Miao; Yining Shi; He Zhe Lim; Li Liu; Tianbao Zhou; Hongyi Wang; Huang Yu; Yifei Hu; Guang Li; Guang Chen; Hao Ye; Lijun Sun; Diange Yang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: \textbf{(i) Structured Data Generation}, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline}, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and \textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by \textbf{53.91\%} and enhances answer accuracy by \textbf{33.54\%}, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models.
>
---
#### [replaced 031] Multi-modal Collaborative Optimization and Expansion Network for Event-assisted Single-eye Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12007v3](http://arxiv.org/pdf/2505.12007v3)**

> **作者:** Runduo Han; Xiuping Liu; Shangxuan Yi; Yi Zhang; Hongchen Tan
>
> **摘要:** In this paper, we proposed a Multi-modal Collaborative Optimization and Expansion Network (MCO-E Net), to use event modalities to resist challenges such as low light, high exposure, and high dynamic range in single-eye expression recognition tasks. The MCO-E Net introduces two innovative designs: Multi-modal Collaborative Optimization Mamba (MCO-Mamba) and Heterogeneous Collaborative and Expansion Mixture-of-Experts (HCE-MoE). MCO-Mamba, building upon Mamba, leverages dual-modal information to jointly optimize the model, facilitating collaborative interaction and fusion of modal semantics. This approach encourages the model to balance the learning of both modalities and harness their respective strengths. HCE-MoE, on the other hand, employs a dynamic routing mechanism to distribute structurally varied experts (deep, attention, and focal), fostering collaborative learning of complementary semantics. This heterogeneous architecture systematically integrates diverse feature extraction paradigms to comprehensively capture expression semantics. Extensive experiments demonstrate that our proposed network achieves competitive performance in the task of single-eye expression recognition, especially under poor lighting conditions.
>
---
#### [replaced 032] Artificial intelligence in digital pathology: a systematic review and meta-analysis of diagnostic test accuracy
- **分类: physics.med-ph; cs.AI; cs.CV; eess.IV; q-bio.QM; I.2.1**

- **链接: [http://arxiv.org/pdf/2306.07999v3](http://arxiv.org/pdf/2306.07999v3)**

> **作者:** Clare McGenity; Emily L Clarke; Charlotte Jennings; Gillian Matthews; Caroline Cartlidge; Henschel Freduah-Agyemang; Deborah D Stocken; Darren Treanor
>
> **备注:** 26 pages, 5 figures, 8 tables + Supplementary materials Preprint is pre-peer review version. Please see link for updated, peer reviewed article to see latest version
>
> **摘要:** Ensuring diagnostic performance of AI models before clinical use is key to the safe and successful adoption of these technologies. Studies reporting AI applied to digital pathology images for diagnostic purposes have rapidly increased in number in recent years. The aim of this work is to provide an overview of the diagnostic accuracy of AI in digital pathology images from all areas of pathology. This systematic review and meta-analysis included diagnostic accuracy studies using any type of artificial intelligence applied to whole slide images (WSIs) in any disease type. The reference standard was diagnosis through histopathological assessment and / or immunohistochemistry. Searches were conducted in PubMed, EMBASE and CENTRAL in June 2022. We identified 2976 studies, of which 100 were included in the review and 48 in the full meta-analysis. Risk of bias and concerns of applicability were assessed using the QUADAS-2 tool. Data extraction was conducted by two investigators and meta-analysis was performed using a bivariate random effects model. 100 studies were identified for inclusion, equating to over 152,000 whole slide images (WSIs) and representing many disease types. Of these, 48 studies were included in the meta-analysis. These studies reported a mean sensitivity of 96.3% (CI 94.1-97.7) and mean specificity of 93.3% (CI 90.5-95.4) for AI. There was substantial heterogeneity in study design and all 100 studies identified for inclusion had at least one area at high or unclear risk of bias. This review provides a broad overview of AI performance across applications in whole slide imaging. However, there is huge variability in study design and available performance data, with details around the conduct of the study and make up of the datasets frequently missing. Overall, AI offers good accuracy when applied to WSIs but requires more rigorous evaluation of its performance.
>
---
#### [replaced 033] OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific Discovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.15415v2](http://arxiv.org/pdf/2501.15415v2)**

> **作者:** Siqi Fan; Yuguang Xie; Bowen Cai; Ailin Xie; Gaochao Liu; Mu Qiao; Jie Xing; Zaiqing Nie
>
> **摘要:** Understanding the chemical structure from a graphical representation of a molecule is a challenging image caption task that would greatly benefit molecule-centric scientific discovery. Variations in molecular images and caption subtasks pose a significant challenge in both image representation learning and task modeling. Yet, existing methods only focus on a specific caption task that translates a molecular image into its graph structure, i.e., OCSR. In this paper, we propose the Optical Chemical Structure Understanding (OCSU) task, which extends low-level recognition to multilevel understanding and aims to translate chemical structure diagrams into readable strings for both machine and chemist. To facilitate the development of OCSU technology, we explore both OCSR-based and OCSR-free paradigms. We propose DoubleCheck to enhance OCSR performance via attentive feature enhancement for local ambiguous atoms. It can be cascaded with existing SMILES-based molecule understanding methods to achieve OCSU. Meanwhile, Mol-VL is a vision-language model end-to-end optimized for OCSU. We also construct Vis-CheBI20, the first large-scale OCSU dataset. Through comprehensive experiments, we demonstrate the proposed approaches excel at providing chemist-readable caption for chemical structure diagrams, which provide solid baselines for further research. Our code, model, and data are open-sourced at https://github.com/PharMolix/OCSU.
>
---
#### [replaced 034] TRACE: Transformer-based Risk Assessment for Clinical Evaluation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.08701v2](http://arxiv.org/pdf/2411.08701v2)**

> **作者:** Dionysis Christopoulos; Sotiris Spanos; Valsamis Ntouskos; Konstantinos Karantzalos
>
> **摘要:** We present TRACE (Transformer-based Risk Assessment for Clinical Evaluation), a novel method for clinical risk assessment based on clinical data, leveraging the self-attention mechanism for enhanced feature interaction and result interpretation. Our approach is able to handle different data modalities, including continuous, categorical and multiple-choice (checkbox) attributes. The proposed architecture features a shared representation of the clinical data obtained by integrating specialized embeddings of each data modality, enabling the detection of high-risk individuals using Transformer encoder layers. To assess the effectiveness of the proposed method, a strong baseline based on non-negative multi-layer perceptrons (MLPs) is introduced. The proposed method outperforms various baselines widely used in the domain of clinical risk assessment, while effectively handling missing values. In terms of explainability, our Transformer-based method offers easily interpretable results via attention weights, further enhancing the clinicians' decision-making process.
>
---
#### [replaced 035] Refining CNN-based Heatmap Regression with Gradient-based Corner Points for Electrode Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.17105v3](http://arxiv.org/pdf/2412.17105v3)**

> **作者:** Lin Wu
>
> **摘要:** We propose a method for detecting the electrode positions in lithium-ion batteries. The process begins by identifying the region of interest (ROI) in the battery's X-ray image through corner point detection. A convolutional neural network is then used to regress the pole positions within this ROI. Finally, the regressed positions are optimized and corrected using corner point priors, significantly mitigating the loss of localization accuracy caused by operations such as feature map down-sampling and padding during network training. Our findings show that combining traditional pixel gradient analysis with CNN-based heatmap regression for keypoint extraction enhances both accuracy and efficiency, resulting in significant performance improvements.
>
---
#### [replaced 036] More Text, Less Point: Towards 3D Data-Efficient Point-Language Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.15966v3](http://arxiv.org/pdf/2408.15966v3)**

> **作者:** Yuan Tang; Xu Han; Xianzhi Li; Qiao Yu; Jinfeng Xu; Yixue Hao; Long Hu; Min Chen
>
> **摘要:** Enabling Large Language Models (LLMs) to comprehend the 3D physical world remains a significant challenge. Due to the lack of large-scale 3D-text pair datasets, the success of LLMs has yet to be replicated in 3D understanding. In this paper, we rethink this issue and propose a new task: 3D Data-Efficient Point-Language Understanding. The goal is to enable LLMs to achieve robust 3D object understanding with minimal 3D point cloud and text data pairs. To address this task, we introduce GreenPLM, which leverages more text data to compensate for the lack of 3D data. First, inspired by using CLIP to align images and text, we utilize a pre-trained point cloud-text encoder to map the 3D point cloud space to the text space. This mapping leaves us to seamlessly connect the text space with LLMs. Once the point-text-LLM connection is established, we further enhance text-LLM alignment by expanding the intermediate text space, thereby reducing the reliance on 3D point cloud data. Specifically, we generate 6M free-text descriptions of 3D objects, and design a three-stage training strategy to help LLMs better explore the intrinsic connections between different modalities. To achieve efficient modality alignment, we design a zero-parameter cross-attention module for token pooling. Extensive experimental results show that GreenPLM requires only 12% of the 3D training data used by existing state-of-the-art models to achieve superior 3D understanding. Remarkably, GreenPLM also achieves competitive performance using text-only data. The code and weights are available at: https://github.com/TangYuan96/GreenPLM.
>
---
#### [replaced 037] A Deep Unrolling Model with Hybrid Optimization Structure for Hyperspectral Image Deconvolution
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2306.06378v2](http://arxiv.org/pdf/2306.06378v2)**

> **作者:** Alexandros Gkillas; Dimitris Ampeliotis; Kostas Berberidis
>
> **摘要:** In recent literature there are plenty of works that combine handcrafted and learnable regularizers to solve inverse imaging problems. While this hybrid approach has demonstrated promising results, the motivation for combining handcrafted and learnable regularizers remains largely underexplored. This work aims to justify this combination, by demonstrating that the incorporation of proper handcrafted regularizers alongside learnable regularizers not only reduces the complexity of the learnable prior, but also the performance is notably enhanced. To analyze the impact of this synergy, we introduce the notion of residual structure, to refer to the structure of the solution that cannot be modeled by the handcrafted regularizers per se. Motivated by these, we propose a novel optimization framework for the hyperspectral deconvolution problem, called DeepMix. Based on the proposed optimization framework, an interpretable model is developed using the deep unrolling strategy, which consists of three distinct modules, namely, a data consistency module, a module that enforces the effect of the handcrafted regularizers, and a denoising module. Recognizing the collaborative nature of these modules, this work proposes a context aware denoising module designed to sustain the advancements achieved by the cooperative efforts of the other modules. This is facilitated through the incorporation of a proper skip connection, ensuring that essential details and structures identified by other modules are effectively retained and not lost during denoising. Extensive experimental results across simulated and real-world datasets demonstrate that DeepMix is notable for surpassing existing methodologies, offering marked improvements in both image quality and computational efficiency.
>
---
#### [replaced 038] VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12081v3](http://arxiv.org/pdf/2505.12081v3)**

> **作者:** Yuqi Liu; Tianyuan Qu; Zhisheng Zhong; Bohao Peng; Shu Liu; Bei Yu; Jiaya Jia
>
> **摘要:** Large vision-language models exhibit inherent capabilities to handle diverse visual perception tasks. In this paper, we introduce VisionReasoner, a unified framework capable of reasoning and solving multiple visual perception tasks within a shared model. Specifically, by designing novel multi-object cognitive learning strategies and systematic task reformulation, VisionReasoner enhances its reasoning capabilities to analyze visual inputs, and addresses diverse perception tasks in a unified framework. The model generates a structured reasoning process before delivering the desired outputs responding to user queries. To rigorously assess unified visual perception capabilities, we evaluate VisionReasoner on ten diverse tasks spanning three critical domains: detection, segmentation, and counting. Experimental results show that VisionReasoner achieves superior performance as a unified model, outperforming Qwen2.5VL by relative margins of 29.1% on COCO (detection), 22.1% on ReasonSeg (segmentation), and 15.3% on CountBench (counting).
>
---
#### [replaced 039] VAE-QWGAN: Addressing Mode Collapse in Quantum GANs via Autoencoding Priors
- **分类: quant-ph; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.10339v2](http://arxiv.org/pdf/2409.10339v2)**

> **作者:** Aaron Mark Thomas; Harry Youel; Sharu Theresa Jose
>
> **备注:** 30 pages, 13 figures
>
> **摘要:** Recent proposals for quantum generative adversarial networks (GANs) suffer from the issue of mode collapse, analogous to classical GANs, wherein the distribution learnt by the GAN fails to capture the high mode complexities of the target distribution. Mode collapse can arise due to the use of uninformed prior distributions in the generative learning task. To alleviate the issue of mode collapse for quantum GANs, this work presents a novel \textbf{hybrid quantum-classical generative model}, the VAE-QWGAN, which combines the strengths of a classical Variational AutoEncoder (VAE) with a hybrid Quantum Wasserstein GAN (QWGAN). The VAE-QWGAN fuses the VAE decoder and QWGAN generator into a single quantum model, and utilizes the VAE encoder for data-dependant latent vector sampling during training. This in turn, enhances the diversity and quality of generated images. To generate new data from the trained model at inference, we sample from a Gaussian mixture model (GMM) prior that is learnt on the latent vectors generated during training. We conduct extensive experiments for image generation QGANs on MNIST/Fashion-MNIST datasets and compute a range of metrics that measure the diversity and quality of generated samples. We show that VAE-QWGAN demonstrates significant improvement over existing QGAN approaches.
>
---
#### [replaced 040] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02567v3](http://arxiv.org/pdf/2505.02567v3)**

> **作者:** Xinjie Zhang; Jintao Guo; Shanshan Zhao; Minghao Fu; Lunhao Duan; Guo-Hua Wang; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** In this version, we incorporate new papers. This work is still in progress; Github project: https://github.com/AIDC-AI/Awesome-Unified-Multimodal-Models
>
> **摘要:** Recent years have seen remarkable progress in both multimodal understanding models and image generation models. Despite their respective successes, these two domains have evolved independently, leading to distinct architectural paradigms: While autoregressive-based architectures have dominated multimodal understanding, diffusion-based models have become the cornerstone of image generation. Recently, there has been growing interest in developing unified frameworks that integrate these tasks. The emergence of GPT-4o's new capabilities exemplifies this trend, highlighting the potential for unification. However, the architectural differences between the two domains pose significant challenges. To provide a clear overview of current efforts toward unification, we present a comprehensive survey aimed at guiding future research. First, we introduce the foundational concepts and recent advancements in multimodal understanding and text-to-image generation models. Next, we review existing unified models, categorizing them into three main architectural paradigms: diffusion-based, autoregressive-based, and hybrid approaches that fuse autoregressive and diffusion mechanisms. For each category, we analyze the structural designs and innovations introduced by related works. Additionally, we compile datasets and benchmarks tailored for unified models, offering resources for future exploration. Finally, we discuss the key challenges facing this nascent field, including tokenization strategy, cross-modal attention, and data. As this area is still in its early stages, we anticipate rapid advancements and will regularly update this survey. Our goal is to inspire further research and provide a valuable reference for the community. The references associated with this survey are available on GitHub (https://github.com/AIDC-AI/Awesome-Unified-Multimodal-Models).
>
---
#### [replaced 041] UniRestorer: Universal Image Restoration via Adaptively Estimating Image Degradation at Proper Granularity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20157v3](http://arxiv.org/pdf/2412.20157v3)**

> **作者:** Jingbo Lin; Zhilu Zhang; Wenbo Li; Renjing Pei; Hang Xu; Hongzhi Zhang; Wangmeng Zuo
>
> **摘要:** Recently, considerable progress has been made in all-in-one image restoration. Generally, existing methods can be degradation-agnostic or degradation-aware. However, the former are limited in leveraging degradation-specific restoration, and the latter suffer from the inevitable error in degradation estimation. Consequently, the performance of existing methods has a large gap compared to specific single-task models. In this work, we make a step forward in this topic, and present our UniRestorer with improved restoration performance. Specifically, we perform hierarchical clustering on degradation space, and train a multi-granularity mixture-of-experts (MoE) restoration model. Then, UniRestorer adopts both degradation and granularity estimation to adaptively select an appropriate expert for image restoration. In contrast to existing degradation-agnostic and -aware methods, UniRestorer can leverage degradation estimation to benefit degradation specific restoration, and use granularity estimation to make the model robust to degradation estimation error. Experimental results show that our UniRestorer outperforms state-of-the-art all-in-one methods by a large margin, and is promising in closing the performance gap to specific single task models.
>
---
#### [replaced 042] When LLMs Learn to be Students: The SOEI Framework for Modeling and Evaluating Virtual Student Agents in Educational Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15701v2](http://arxiv.org/pdf/2410.15701v2)**

> **作者:** Yiping Ma; Shiyu Hu; Xuchen Li; Yipei Wang; Yuqing Chen; Shiqing Liu; Kang Hao Cheong
>
> **摘要:** Recent advances in large language models (LLMs) have enabled intelligent tutoring systems, yet the development of LLM-based Virtual Student Agents (LVSAs) remains underexplored. Such agents are essential for teacher-facing applications, where simulating diverse learner traits can support adaptive instruction and pedagogical skill development. However, current methods lack principled personality modeling, scalable evaluation of behavioral consistency, and empirical validation in interactive teaching settings. We propose the SOEI framework, a structured pipeline comprising Scene, Object, Evaluation, and Interaction, for constructing and evaluating personality-aligned LVSAs in classroom scenarios. Leveraging Chinese language instruction as a cognitively and emotionally rich testbed, we generate five LVSAs based on Big Five traits through LoRA fine-tuning and expert-informed prompt design. Their behavioral realism and personality coherence are assessed using a hybrid human & GPT-4 evaluation and a multi-dimensional annotation protocol. Through controlled experiments with real pre-service teachers, we demonstrate that LVSAs can elicit adaptive teaching strategies and maintain trait-consistent behavior across multi-turn dialogues. Our results provide: (1) an educationally and psychologically grounded generation pipeline for LLM-based student agents; (2) a hybrid, scalable evaluation framework for behavioral realism; and (3) empirical insights into the pedagogical utility of LVSAs in shaping instructional adaptation. By embedding LVSAs into both generative modeling and human-in-the-loop teaching, SOEI bridges AI for Education (AI4Edu) and Education for AI (Edu4AI), positioning classroom interaction as a rigorous testbed for controllability, personality alignment, and human-likeness in large language models.
>
---
#### [replaced 043] Uncovering Cultural Representation Disparities in Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14729v2](http://arxiv.org/pdf/2505.14729v2)**

> **作者:** Ram Mohan Rao Kadiyala; Siddhant Gupta; Jebish Purbey; Srishti Yadav; Alejandro Salamanca; Desmond Elliott
>
> **备注:** 28 pages, 36 figures
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities across a range of tasks, yet concerns about their potential biases exist. This work investigates the extent to which prominent VLMs exhibit cultural biases by evaluating their performance on an image-based country identification task at a country level. Utilizing the geographically diverse Country211 dataset, we probe several large vision language models (VLMs) under various prompting strategies: open-ended questions, multiple-choice questions (MCQs) including challenging setups like multilingual and adversarial settings. Our analysis aims to uncover disparities in model accuracy across different countries and question formats, providing insights into how training data distribution and evaluation methodologies might influence cultural biases in VLMs. The findings highlight significant variations in performance, suggesting that while VLMs possess considerable visual understanding, they inherit biases from their pre-training data and scale that impact their ability to generalize uniformly across diverse global contexts.
>
---
#### [replaced 044] ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.17038v4](http://arxiv.org/pdf/2412.17038v4)**

> **作者:** Sipeng Shen; Yunming Zhang; Dengpan Ye; Xiuwen Shi; Long Tang; Haoran Duan; Yueyun Shang; Zhihong Tian
>
> **摘要:** While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.
>
---
#### [replaced 045] GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.19552v2](http://arxiv.org/pdf/2410.19552v2)**

> **作者:** Hosam Elgendy; Ahmed Sharshar; Ahmed Aboeitta; Yasser Ashraf; Mohsen Guizani
>
> **备注:** 14 pages, 5 figures, 3 tables
>
> **摘要:** Detecting temporal changes in geographical landscapes is critical for applications like environmental monitoring and urban planning. While remote sensing data is abundant, existing vision-language models (VLMs) often fail to capture temporal dynamics effectively. This paper addresses these limitations by introducing an annotated dataset of video frame pairs to track evolving geographical patterns over time. Using fine-tuning techniques like Low-Rank Adaptation (LoRA), quantized LoRA (QLoRA), and model pruning on models such as Video-LLaVA and LLaVA-NeXT-Video, we significantly enhance VLM performance in processing remote sensing temporal changes. Results show significant improvements, with the best performance achieving a BERT score of 0.864 and ROUGE-1 score of 0.576, demonstrating superior accuracy in describing land-use transformations.
>
---
#### [replaced 046] PRS-Med: Position Reasoning Segmentation with Vision-Language Model in Medical Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11872v2](http://arxiv.org/pdf/2505.11872v2)**

> **作者:** Quoc-Huy Trinh; Minh-Van Nguyen; Jung Peng; Ulas Bagci; Debesh Jha
>
> **摘要:** Recent advancements in prompt-based medical image segmentation have enabled clinicians to identify tumors using simple input like bounding boxes or text prompts. However, existing methods face challenges when doctors need to interact through natural language or when position reasoning is required - understanding spatial relationships between anatomical structures and pathologies. We present PRS-Med, a framework that integrates vision-language models with segmentation capabilities to generate both accurate segmentation masks and corresponding spatial reasoning outputs. Additionally, we introduce the MMRS dataset (Multimodal Medical in Positional Reasoning Segmentation), which provides diverse, spatially-grounded question-answer pairs to address the lack of position reasoning data in medical imaging. PRS-Med demonstrates superior performance across six imaging modalities (CT, MRI, X-ray, ultrasound, endoscopy, RGB), significantly outperforming state-of-the-art methods in both segmentation accuracy and position reasoning. Our approach enables intuitive doctor-system interaction through natural language, facilitating more efficient diagnoses. Our dataset pipeline, model, and codebase will be released to foster further research in spatially-aware multimodal reasoning for medical applications.
>
---
#### [replaced 047] Contrastive Learning-Enhanced Trajectory Matching for Small-Scale Dataset Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15267v2](http://arxiv.org/pdf/2505.15267v2)**

> **作者:** Wenmin Li; Shunsuke Sakai; Tatsuhito Hasegawa
>
> **备注:** Under review
>
> **摘要:** Deploying machine learning models in resource-constrained environments, such as edge devices or rapid prototyping scenarios, increasingly demands distillation of large datasets into significantly smaller yet informative synthetic datasets. Current dataset distillation techniques, particularly Trajectory Matching methods, optimize synthetic data so that the model's training trajectory on synthetic samples mirrors that on real data. While demonstrating efficacy on medium-scale synthetic datasets, these methods fail to adequately preserve semantic richness under extreme sample scarcity. To address this limitation, we propose a novel dataset distillation method integrating contrastive learning during image synthesis. By explicitly maximizing instance-level feature discrimination, our approach produces more informative and diverse synthetic samples, even when dataset sizes are significantly constrained. Experimental results demonstrate that incorporating contrastive learning substantially enhances the performance of models trained on very small-scale synthetic datasets. This integration not only guides more effective feature representation but also significantly improves the visual fidelity of the synthesized images. Experimental results demonstrate that our method achieves notable performance improvements over existing distillation techniques, especially in scenarios with extremely limited synthetic data.
>
---
#### [replaced 048] SegMatch: A semi-supervised learning method for surgical instrument segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2308.05232v2](http://arxiv.org/pdf/2308.05232v2)**

> **作者:** Meng Wei; Charlie Budd; Luis C. Garcia-Peraza-Herrera; Reuben Dorent; Miaojing Shi; Tom Vercauteren
>
> **备注:** Published, 19 pages, 8 figures
>
> **摘要:** Surgical instrument segmentation is recognised as a key enabler in providing advanced surgical assistance and improving computer-assisted interventions. In this work, we propose SegMatch, a semi-supervised learning method to reduce the need for expensive annotation for laparoscopic and robotic surgical images. SegMatch builds on FixMatch, a widespread semi supervised classification pipeline combining consistency regularization and pseudo-labelling, and adapts it for the purpose of segmentation. In our proposed SegMatch, the unlabelled images are first weakly augmented and fed to the segmentation model to generate pseudo-labels. In parallel, images are fed to a strong augmentation branch and consistency between the branches is used as an unsupervised loss. To increase the relevance of our strong augmentations, we depart from using only handcrafted augmentations and introduce a trainable adversarial augmentation strategy. Our FixMatch adaptation for segmentation tasks further includes carefully considering the equivariance and invariance properties of the augmentation functions we rely on. For binary segmentation tasks, our algorithm was evaluated on the MICCAI Instrument Segmentation Challenge datasets, Robust-MIS 2019 and EndoVis 2017. For multi-class segmentation tasks, we relied on the recent CholecInstanceSeg dataset. Our results show that SegMatch outperforms fully-supervised approaches by incorporating unlabelled data, and surpasses a range of state-of-the-art semi-supervised models across different labelled to unlabelled data ratios.
>
---
#### [replaced 049] Liver Cirrhosis Stage Estimation from MRI with Deep Learning
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.18225v3](http://arxiv.org/pdf/2502.18225v3)**

> **作者:** Jun Zeng; Debesh Jha; Ertugrul Aktas; Elif Keles; Alpay Medetalibeyoglu; Matthew Antalek; Federica Proietto Salanitri; Amir A. Borhani; Daniela P. Ladner; Gorkem Durak; Ulas Bagci
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** We present an end-to-end deep learning framework for automated liver cirrhosis stage estimation from multi-sequence MRI. Cirrhosis is the severe scarring (fibrosis) of the liver and a common endpoint of various chronic liver diseases. Early diagnosis is vital to prevent complications such as decompensation and cancer, which significantly decreases life expectancy. However, diagnosing cirrhosis in its early stages is challenging, and patients often present with life-threatening complications. Our approach integrates multi-scale feature learning with sequence-specific attention mechanisms to capture subtle tissue variations across cirrhosis progression stages. Using CirrMRI600+, a large-scale publicly available dataset of 628 high-resolution MRI scans from 339 patients, we demonstrate state-of-the-art performance in three-stage cirrhosis classification. Our best model achieves 72.8% accuracy on T1W and 63.8% on T2W sequences, significantly outperforming traditional radiomics-based approaches. Through extensive ablation studies, we show that our architecture effectively learns stage-specific imaging biomarkers. We establish new benchmarks for automated cirrhosis staging and provide insights for developing clinically applicable deep learning systems. The source code will be available at https://github.com/JunZengz/CirrhosisStage.
>
---
#### [replaced 050] Depth-Weighted Detection of Behaviours of Risk in People with Dementia using Cameras
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.15519v3](http://arxiv.org/pdf/2408.15519v3)**

> **作者:** Pratik K. Mishra; Irene Ballester; Andrea Iaboni; Bing Ye; Kristine Newman; Alex Mihailidis; Shehroz S. Khan
>
> **摘要:** The behavioural and psychological symptoms of dementia, such as agitation and aggression, present a significant health and safety risk in residential care settings. Many care facilities have video cameras in place for digital monitoring of public spaces, which can be leveraged to develop an automated behaviours of risk detection system that can alert the staff to enable timely intervention and prevent the situation from escalating. However, one of the challenges in our previous study was the presence of false alarms due to disparate importance of events based on distance. To address this issue, we proposed a novel depth-weighted loss to enforce equivalent importance to the events happening both near and far from the cameras; thus, helping to reduce false alarms. We further propose to utilize the training outliers to determine the anomaly threshold. The data from nine dementia participants across three cameras in a specialized dementia unit were used for training. The proposed approach obtained the best area under receiver operating characteristic curve performance of 0.852, 0.81 and 0.768, respectively, for the three cameras. Ablation analysis was conducted for the individual components of the proposed approach and effect of frame size and frame rate. The performance of the proposed approach was investigated for cross-camera, participant-specific and sex-specific behaviours of risk detection. The proposed approach performed reasonably well in reducing false alarms. This motivates further research to make the system more suitable for deployment in care facilities.
>
---
#### [replaced 051] How Well Can General Vision-Language Models Learn Medicine By Watching Public Educational Videos?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14391v2](http://arxiv.org/pdf/2504.14391v2)**

> **作者:** Rahul Thapa; Andrew Li; Qingyang Wu; Bryan He; Yuki Sahashi; Christina Binder; Angela Zhang; Ben Athiwaratkun; Shuaiwen Leon Song; David Ouyang; James Zou
>
> **摘要:** Publicly available biomedical videos, such as those on YouTube, serve as valuable educational resources for medical students. Unlike standard machine learning datasets, these videos are designed for human learners, often mixing medical imagery with narration, explanatory diagrams, and contextual framing. In this work, we investigate whether such pedagogically rich, yet non-standardized and heterogeneous videos can effectively teach general-domain vision-language models biomedical knowledge. To this end, we introduce OpenBiomedVi, a biomedical video instruction tuning dataset comprising 1031 hours of video-caption and Q/A pairs, curated through a multi-step human-in-the-loop pipeline. Diverse biomedical video datasets are rare, and OpenBiomedVid fills an important gap by providing instruction-style supervision grounded in real-world educational content. Surprisingly, despite the informal and heterogeneous nature of these videos, the fine-tuned Qwen-2-VL models exhibit substantial performance improvements across most benchmarks. The 2B model achieves gains of 98.7% on video tasks, 71.2% on image tasks, and 0.2% on text tasks. The 7B model shows improvements of 37.09% on video and 11.2% on image tasks, with a slight degradation of 2.7% on text tasks compared to their respective base models. To address the lack of standardized biomedical video evaluation datasets, we also introduce two new expert curated benchmarks, MIMICEchoQA and SurgeryVideoQA. On these benchmarks, the 2B model achieves gains of 99.1% and 98.1%, while the 7B model shows gains of 22.5% and 52.1%, respectively, demonstrating the models' ability to generalize and perform biomedical video understanding on cleaner and more standardized datasets than those seen during training. These results suggest that educational videos created for human learning offer a surprisingly effective training signal for biomedical VLMs.
>
---
#### [replaced 052] Looking Beyond Language Priors: Enhancing Visual Comprehension and Attention in Multimodal Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05626v2](http://arxiv.org/pdf/2505.05626v2)**

> **作者:** Aarti Ghatkesar; Uddeshya Upadhyay; Ganesh Venkatesh
>
> **摘要:** Achieving deep alignment between vision and language remains a central challenge for Multimodal Large Language Models (MLLMs). These models often fail to fully leverage visual input, defaulting to strong language priors. Our approach first provides insights into how MLLMs internally build visual understanding of image regions and then introduces techniques to amplify this capability. Specifically, we explore techniques designed both to deepen the model's understanding of visual content and to ensure that these visual insights actively guide language generation. We demonstrate the superior multimodal understanding of our resultant model through a detailed upstream analysis quantifying its ability to predict visually-dependent tokens as well as 10 pt boost on visually challenging tasks.
>
---
#### [replaced 053] seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03176v2](http://arxiv.org/pdf/2505.03176v2)**

> **作者:** Hafez Ghaemi; Eilif Muller; Shahab Bakhtiari
>
> **摘要:** Current self-supervised algorithms commonly rely on transformations such as data augmentation and masking to learn visual representations. This is achieved by enforcing invariance or equivariance with respect to these transformations after encoding two views of an image. This dominant two-view paradigm often limits the flexibility of learned representations for downstream adaptation by creating performance trade-offs between high-level invariance-demanding tasks such as image classification and more fine-grained equivariance-related tasks. In this work, we proposes \emph{seq-JEPA}, a world modeling framework that introduces architectural inductive biases into joint-embedding predictive architectures to resolve this trade-off. Without relying on dual equivariance predictors or loss terms, seq-JEPA simultaneously learns two architecturally segregated representations: one equivariant to specified transformations and another invariant to them. To do so, our model processes short sequences of different views (observations) of inputs. Each encoded view is concatenated with an embedding of the relative transformation (action) that produces the next observation in the sequence. These view-action pairs are passed through a transformer encoder that outputs an aggregate representation. A predictor head then conditions this aggregate representation on the upcoming action to predict the representation of the next observation. Empirically, seq-JEPA demonstrates strong performance on both equivariant and invariant benchmarks without sacrificing one for the other. Furthermore, it excels at tasks that inherently require aggregating a sequence of observations, such as path integration across actions and predictive learning across eye movements.
>
---
#### [replaced 054] Place Recognition: A Comprehensive Review, Current Challenges and Future Directions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14068v2](http://arxiv.org/pdf/2505.14068v2)**

> **作者:** Zhenyu Li; Tianyi Shang; Pengjie Xu; Zhaojun Deng
>
> **备注:** 35 pages
>
> **摘要:** Place recognition is a cornerstone of vehicle navigation and mapping, which is pivotal in enabling systems to determine whether a location has been previously visited. This capability is critical for tasks such as loop closure in Simultaneous Localization and Mapping (SLAM) and long-term navigation under varying environmental conditions. In this survey, we comprehensively review recent advancements in place recognition, emphasizing three representative methodological paradigms: Convolutional Neural Network (CNN)-based approaches, Transformer-based frameworks, and cross-modal strategies. We begin by elucidating the significance of place recognition within the broader context of autonomous systems. Subsequently, we trace the evolution of CNN-based methods, highlighting their contributions to robust visual descriptor learning and scalability in large-scale environments. We then examine the emerging class of Transformer-based models, which leverage self-attention mechanisms to capture global dependencies and offer improved generalization across diverse scenes. Furthermore, we discuss cross-modal approaches that integrate heterogeneous data sources such as Lidar, vision, and text description, thereby enhancing resilience to viewpoint, illumination, and seasonal variations. We also summarize standard datasets and evaluation metrics widely adopted in the literature. Finally, we identify current research challenges and outline prospective directions, including domain adaptation, real-time performance, and lifelong learning, to inspire future advancements in this domain. The unified framework of leading-edge place recognition methods, i.e., code library, and the results of their experimental evaluations are available at https://github.com/CV4RA/SOTA-Place-Recognitioner.
>
---
#### [replaced 055] Remote Sensing Spatio-Temporal Vision-Language Models: A Comprehensive Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02573v2](http://arxiv.org/pdf/2412.02573v2)**

> **作者:** Chenyang Liu; Jiafan Zhang; Keyan Chen; Man Wang; Zhengxia Zou; Zhenwei Shi
>
> **摘要:** The interpretation of multi-temporal remote sensing imagery is critical for monitoring Earth's dynamic processes-yet previous change detection methods, which produce binary or semantic masks, fall short of providing human-readable insights into changes. Recent advances in Vision-Language Models (VLMs) have opened a new frontier by fusing visual and linguistic modalities, enabling spatio-temporal vision-language understanding: models that not only capture spatial and temporal dependencies to recognize changes but also provide a richer interactive semantic analysis of temporal images (e.g., generate descriptive captions and answer natural-language queries). In this survey, we present the first comprehensive review of RS-STVLMs. The survey covers the evolution of models from early task-specific models to recent general foundation models that leverage powerful large language models. We discuss progress in representative tasks, such as change captioning, change question answering, and change grounding. Moreover, we systematically dissect the fundamental components and key technologies underlying these models, and review the datasets and evaluation metrics that have driven the field. By synthesizing task-level insights with a deep dive into shared architectural patterns, we aim to illuminate current achievements and chart promising directions for future research in spatio-temporal vision-language understanding for remote sensing. We will keep tracing related works at https://github.com/Chen-Yang-Liu/Awesome-RS-SpatioTemporal-VLMs
>
---
#### [replaced 056] Logit Scaling for Out-of-Distribution Detection
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.01175v2](http://arxiv.org/pdf/2409.01175v2)**

> **作者:** Andrija Djurisic; Rosanne Liu; Mladen Nikolic
>
> **摘要:** The safe deployment of machine learning and AI models in open-world settings hinges critically on the ability to detect out-of-distribution (OOD) data accurately, data samples that contrast vastly from what the model was trained with. Current approaches to OOD detection often require further training the model, and/or statistics about the training data which may no longer be accessible. Additionally, many existing OOD detection methods struggle to maintain performance when transferred across different architectures. Our research tackles these issues by proposing a simple, post-hoc method that does not require access to the training data distribution, keeps a trained network intact, and holds strong performance across a variety of architectures. Our method, Logit Scaling (LTS), as the name suggests, simply scales the logits in a manner that effectively distinguishes between in-distribution (ID) and OOD samples. We tested our method on benchmarks across various scales, including CIFAR-10, CIFAR-100, ImageNet and OpenOOD. The experiments cover 3 ID and 14 OOD datasets, as well as 9 model architectures. Overall, we demonstrate state-of-the-art performance, robustness and adaptability across different architectures, paving the way towards a universally applicable solution for advanced OOD detection.
>
---
#### [replaced 057] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10250v2](http://arxiv.org/pdf/2505.10250v2)**

> **作者:** Wenhao Shen; Wanqi Yin; Xiaofeng Yang; Cheng Chen; Chaoyue Song; Zhongang Cai; Lei Yang; Hao Wang; Guosheng Lin
>
> **备注:** Accepted to ICML 2025. Code: https://github.com/shenwenhao01/ADHMR
>
> **摘要:** Human mesh recovery (HMR) from a single image is inherently ill-posed due to depth ambiguity and occlusions. Probabilistic methods have tried to solve this by generating numerous plausible 3D human mesh predictions, but they often exhibit misalignment with 2D image observations and weak robustness to in-the-wild images. To address these issues, we propose ADHMR, a framework that Aligns a Diffusion-based HMR model in a preference optimization manner. First, we train a human mesh prediction assessment model, HMR-Scorer, capable of evaluating predictions even for in-the-wild images without 3D annotations. We then use HMR-Scorer to create a preference dataset, where each input image has a pair of winner and loser mesh predictions. This dataset is used to finetune the base model using direct preference optimization. Moreover, HMR-Scorer also helps improve existing HMR models by data cleaning, even with fewer training samples. Extensive experiments show that ADHMR outperforms current state-of-the-art methods. Code is available at: https://github.com/shenwenhao01/ADHMR.
>
---
#### [replaced 058] ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.11787v2](http://arxiv.org/pdf/2503.11787v2)**

> **作者:** Samuel W. Remedios; Shuwen Wei; Shuo Han; Jinwei Zhang; Aaron Carass; Kurt G. Schilling; Dzung L. Pham; Jerry L. Prince; Blake E. Dewey
>
> **摘要:** In clinical imaging, magnetic resonance (MR) image volumes are often acquired as stacks of 2D slices with decreased scan times, improved signal-to-noise ratio, and image contrasts unique to 2D MR pulse sequences. While this is sufficient for clinical evaluation, automated algorithms designed for 3D analysis perform poorly on multi-slice 2D MR volumes, especially those with thick slices and gaps between slices. Super-resolution (SR) methods aim to address this problem, but previous methods do not address all of the following: slice profile shape estimation, slice gap, domain shift, and non-integer or arbitrary upsampling factors. In this paper, we propose ECLARE (Efficient Cross-planar Learning for Anisotropic Resolution Enhancement), a self-SR method that addresses each of these factors. ECLARE uses a slice profile estimated from the multi-slice 2D MR volume, trains a network to learn the mapping from low-resolution to high-resolution in-plane patches from the same volume, and performs SR with anti-aliasing. We compared ECLARE to cubic B-spline interpolation, SMORE, and other contemporary SR methods. We used realistic and representative simulations so that quantitative performance against ground truth can be computed, and ECLARE outperformed all other methods in both signal recovery and downstream tasks. Importantly, as ECLARE does not use external training data it cannot suffer from domain shift between training and testing. Our code is open-source and available at https://www.github.com/sremedios/eclare.
>
---
#### [replaced 059] Goal-conditioned dual-action imitation learning for dexterous dual-arm robot manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2203.09749v3](http://arxiv.org/pdf/2203.09749v3)**

> **作者:** Heecheol Kim; Yoshiyuki Ohmura; Yasuo Kuniyoshi
>
> **备注:** 19 pages, published in Transactions on Robotics (T-RO)
>
> **摘要:** Long-horizon dexterous robot manipulation of deformable objects, such as banana peeling, is a problematic task because of the difficulties in object modeling and a lack of knowledge about stable and dexterous manipulation skills. This paper presents a goal-conditioned dual-action (GC-DA) deep imitation learning (DIL) approach that can learn dexterous manipulation skills using human demonstration data. Previous DIL methods map the current sensory input and reactive action, which often fails because of compounding errors in imitation learning caused by the recurrent computation of actions. The method predicts reactive action only when the precise manipulation of the target object is required (local action) and generates the entire trajectory when precise manipulation is not required (global action). This dual-action formulation effectively prevents compounding error in the imitation learning using the trajectory-based global action while responding to unexpected changes in the target object during the reactive local action. The proposed method was tested in a real dual-arm robot and successfully accomplished the banana-peeling task. Data from this and related works are available at: https://sites.google.com/view/multi-task-fine.
>
---
#### [replaced 060] Feature Map Similarity Reduction in Convolutional Neural Networks
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.03226v2](http://arxiv.org/pdf/2411.03226v2)**

> **作者:** Zakariae Belmekki; Jun Li; Patrick Reuter; David Antonio Gómez Jáuregui; Karl Jenkins
>
> **摘要:** It has been observed that Convolutional Neural Networks (CNNs) suffer from redundancy in feature maps, leading to inefficient capacity utilization. Efforts to address this issue have largely focused on kernel orthogonality method. In this work, we theoretically and empirically demonstrate that kernel orthogonality does not necessarily lead to a reduction in feature map redundancy. Based on this analysis, we propose the Convolutional Similarity method to reduce feature map similarity, independently of the CNN's input. The Convolutional Similarity can be minimized as either a regularization term or an iterative initialization method. Experimental results show that minimizing Convolutional Similarity not only improves classification accuracy but also accelerates convergence. Furthermore, our method enables the use of significantly smaller models to achieve the same level of performance, promoting a more efficient use of model capacity. Future work will focus on coupling the iterative initialization method with the optimization momentum term and examining the method's impact on generative frameworks.
>
---
#### [replaced 061] Efficient Feature Fusion for UAV Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.17983v3](http://arxiv.org/pdf/2501.17983v3)**

> **作者:** Xudong Wang; Yaxin Peng; Chaomin Shen
>
> **摘要:** Object detection in unmanned aerial vehicle (UAV) remote sensing images poses significant challenges due to unstable image quality, small object sizes, complex backgrounds, and environmental occlusions. Small objects, in particular, occupy small portions of images, making their accurate detection highly difficult. Existing multi-scale feature fusion methods address these challenges to some extent by aggregating features across different resolutions. However, they often fail to effectively balance the classification and localization performance for small objects, primarily due to insufficient feature representation and imbalanced network information flow. In this paper, we propose a novel feature fusion framework specifically designed for UAV object detection tasks to enhance both localization accuracy and classification performance. The proposed framework integrates hybrid upsampling and downsampling modules, enabling feature maps from different network depths to be flexibly adjusted to arbitrary resolutions. This design facilitates cross-layer connections and multi-scale feature fusion, ensuring improved representation of small objects. Our approach leverages hybrid downsampling to enhance fine-grained feature representation, improving spatial localization of small targets, even under complex conditions. Simultaneously, the upsampling module aggregates global contextual information, optimizing feature consistency across scales and enhancing classification robustness in cluttered scenes. Experimental results on two public UAV datasets demonstrate the effectiveness of the proposed framework. Integrated into the YOLO-v10 model, our method achieves a 2% improvement in average precision (AP) compared to the baseline YOLO-v10 model, while maintaining the same number of parameters. These results highlight the potential of our framework for accurate and efficient UAV object detection.
>
---
#### [replaced 062] Reconciling Privacy and Explainability in High-Stakes: A Systematic Inquiry
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20798v4](http://arxiv.org/pdf/2412.20798v4)**

> **作者:** Supriya Manna; Niladri Sett
>
> **备注:** Accepted at TMLR
>
> **摘要:** Deep learning's preponderance across scientific domains has reshaped high-stakes decision-making, making it essential to follow rigorous operational frameworks that include both Right-to-Privacy (RTP) and Right-to-Explanation (RTE). This paper examines the complexities of combining these two requirements. For RTP, we focus on `Differential privacy` (DP), which is considered the current gold standard for privacy-preserving machine learning due to its strong quantitative guarantee of privacy. For RTE, we focus on post-hoc explainers: they are the go-to option for model auditing as they operate independently of model training. We formally investigate DP models and various commonly-used post-hoc explainers: how to evaluate these explainers subject to RTP, and analyze the intrinsic interactions between DP models and these explainers. Furthermore, our work throws light on how RTP and RTE can be effectively combined in high-stakes applications. Our study concludes by outlining an industrial software pipeline, with the example of a wildly used use-case, that respects both RTP and RTE requirements.
>
---
#### [replaced 063] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.02813v3](http://arxiv.org/pdf/2409.02813v3)**

> **作者:** Xiang Yue; Tianyu Zheng; Yuansheng Ni; Yubo Wang; Kai Zhang; Shengbang Tong; Yuxuan Sun; Botao Yu; Ge Zhang; Huan Sun; Yu Su; Wenhu Chen; Graham Neubig
>
> **备注:** ACL 2025 Main
>
> **摘要:** This paper introduces MMMU-Pro, a robust version of the Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU) benchmark. MMMU-Pro rigorously assesses multimodal models' true understanding and reasoning capabilities through a three-step process based on MMMU: (1) filtering out questions answerable by text-only models, (2) augmenting candidate options, and (3) introducing a vision-only input setting where questions are embedded within images. This setting challenges AI to truly "see" and "read" simultaneously, testing a fundamental human cognitive skill of seamlessly integrating visual and textual information. Results show that model performance is substantially lower on MMMU-Pro than on MMMU, ranging from 16.8% to 26.9% across models. We explore the impact of OCR prompts and Chain of Thought (CoT) reasoning, finding that OCR prompts have minimal effect while CoT generally improves performance. MMMU-Pro provides a more rigorous evaluation tool, closely mimicking real-world scenarios and offering valuable directions for future research in multimodal AI.
>
---
#### [replaced 064] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16965v2](http://arxiv.org/pdf/2503.16965v2)**

> **作者:** Zhe Hu; Jing Li; Zhongzhu Pu; Hou Pong Chan; Yu Yin
>
> **摘要:** Vision Language Models exhibited immense potential for embodied AI, yet they often lack the sophisticated situational reasoning required for complex decision-making. This paper shows that VLMs can achieve surprisingly strong decision-making performance when visual scenes are represented merely as text-only descriptions, suggesting foundational reasoning can be effectively learned from language. Motivated by this insight, we propose Praxis-VLM, a reasoning VLM for vision-grounded decision-making. Praxis-VLM employs the GRPO algorithm on textual scenarios to instill robust reasoning capabilities, where models learn to evaluate actions and their consequences. These reasoning skills, acquired purely from text, successfully transfer to multimodal inference with visual inputs, significantly reducing reliance on scarce paired image-text training data. Experiments across diverse decision-making benchmarks demonstrate that Praxis-VLM substantially outperforms standard supervised fine-tuning, exhibiting superior performance and generalizability. Further analysis confirms that our models engage in explicit and effective reasoning, underpinning their enhanced performance and adaptability.
>
---
#### [replaced 065] Advanced Knowledge Transfer: Refined Feature Distillation for Zero-Shot Quantization in Edge Computing
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.19125v2](http://arxiv.org/pdf/2412.19125v2)**

> **作者:** Inpyo Hong; Youngwan Jo; Hyojeong Lee; Sunghyun Ahn; Sanghyun Park
>
> **备注:** Accepted at ACM SAC 2025
>
> **摘要:** We introduce AKT (Advanced Knowledge Transfer), a novel method to enhance the training ability of low-bit quantized (Q) models in the field of zero-shot quantization (ZSQ). Existing research in ZSQ has focused on generating high-quality data from full-precision (FP) models. However, these approaches struggle with reduced learning ability in low-bit quantization due to its limited information capacity. To overcome this limitation, we propose effective training strategy compared to data generation. Particularly, we analyzed that refining feature maps in the feature distillation process is an effective way to transfer knowledge to the Q model. Based on this analysis, AKT efficiently transfer core information from the FP model to the Q model. AKT is the first approach to utilize both spatial and channel attention information in feature distillation in ZSQ. Our method addresses the fundamental gradient exploding problem in low-bit Q models. Experiments on CIFAR-10 and CIFAR-100 datasets demonstrated the effectiveness of the AKT. Our method led to significant performance enhancement in existing generative models. Notably, AKT achieved significant accuracy improvements in low-bit Q models, achieving state-of-the-art in the 3,5bit scenarios on CIFAR-10. The code is available at https://github.com/Inpyo-Hong/AKT-Advanced-knowledge-Transfer.
>
---
#### [replaced 066] LiDAR MOT-DETR: A LiDAR-based Two-Stage Transformer for 3D Multiple Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12753v2](http://arxiv.org/pdf/2505.12753v2)**

> **作者:** Martha Teiko Teye; Ori Maoz; Matthias Rottmann
>
> **备注:** Template change
>
> **摘要:** Multi-object tracking from LiDAR point clouds presents unique challenges due to the sparse and irregular nature of the data, compounded by the need for temporal coherence across frames. Traditional tracking systems often rely on hand-crafted features and motion models, which can struggle to maintain consistent object identities in crowded or fast-moving scenes. We present a lidar-based two-staged DETR inspired transformer; a smoother and tracker. The smoother stage refines lidar object detections, from any off-the-shelf detector, across a moving temporal window. The tracker stage uses a DETR-based attention block to maintain tracks across time by associating tracked objects with the refined detections using the point cloud as context. The model is trained on the datasets nuScenes and KITTI in both online and offline (forward peeking) modes demonstrating strong performance across metrics such as ID-switch and multiple object tracking accuracy (MOTA). The numerical results indicate that the online mode outperforms the lidar-only baseline and SOTA models on the nuScenes dataset, with an aMOTA of 0.722 and an aMOTP of 0.475, while the offline mode provides an additional 3 pp aMOTP.
>
---
#### [replaced 067] Copy-Move Forgery Detection and Question Answering for Remote Sensing Image
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.02575v2](http://arxiv.org/pdf/2412.02575v2)**

> **作者:** Ze Zhang; Enyuan Zhao; Di Niu; Jie Nie; Xinyue Liang; Lei Huang
>
> **备注:** 11 figs, 7 tables
>
> **摘要:** Driven by practical demands in land resource monitoring and national defense security, this paper introduces the Remote Sensing Copy-Move Question Answering (RSCMQA) task. Unlike traditional Remote Sensing Visual Question Answering (RSVQA), RSCMQA focuses on interpreting complex tampering scenarios and inferring relationships between objects. We present a suite of global RSCMQA datasets, comprising images from 29 different regions across 14 countries. Specifically, we propose five distinct datasets, including the basic dataset RS-CMQA, the category-balanced dataset RS-CMQA-B, the high-authenticity dataset Real-RSCM, the extended dataset RS-TQA, and the extended category-balanced dataset RS-TQA-B. These datasets fill a critical gap in the field while ensuring comprehensiveness, balance, and challenge. Furthermore, we introduce a region-discrimination-guided multimodal copy-move forgery perception framework (CMFPF), which enhances the accuracy of answering questions about tampered images by leveraging prompt about the differences and connections between the source and tampered domains. Extensive experiments demonstrate that our method provides a stronger benchmark for RSCMQA compared to general VQA and RSVQA models. Our datasets and code are publicly available at https://github.com/shenyedepisa/RSCMQA.
>
---
#### [replaced 068] Robust 6DoF Pose Tracking Considering Contour and Interior Correspondence Uncertainty for AR Assembly Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11971v3](http://arxiv.org/pdf/2502.11971v3)**

> **作者:** Jixiang Chen; Jing Chen; Kai Liu; Haochen Chang; Shanfeng Fu; Jian Yang
>
> **备注:** Accepted by IEEE Transactions on Instrumentation and Measurement
>
> **摘要:** Augmented reality assembly guidance is essential for intelligent manufacturing and medical applications, requiring continuous measurement of the 6DoF poses of manipulated objects. Although current tracking methods have made significant advancements in accuracy and efficiency, they still face challenges in robustness when dealing with cluttered backgrounds, rotationally symmetric objects, and noisy sequences. In this paper, we first propose a robust contour-based pose tracking method that addresses error-prone contour correspondences and improves noise tolerance. It utilizes a fan-shaped search strategy to refine correspondences and models local contour shape and noise uncertainty as mixed probability distribution, resulting in a highly robust contour energy function. Secondly, we introduce a CPU-only strategy to better track rotationally symmetric objects and assist the contour-based method in overcoming local minima by exploring sparse interior correspondences. This is achieved by pre-sampling interior points from sparse viewpoint templates offline and using the DIS optical flow algorithm to compute their correspondences during tracking. Finally, we formulate a unified energy function to fuse contour and interior information, which is solvable using a re-weighted least squares algorithm. Experiments on public datasets and real scenarios demonstrate that our method significantly outperforms state-of-the-art monocular tracking methods and can achieve more than 100 FPS using only a CPU.
>
---
#### [replaced 069] Maximizing Discrimination Capability of Knowledge Distillation with Energy Function
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.14334v3](http://arxiv.org/pdf/2311.14334v3)**

> **作者:** Seonghak Kim; Gyeongdo Ham; Suin Lee; Donggon Jang; Daeshik Kim
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** To apply the latest computer vision techniques that require a large computational cost in real industrial applications, knowledge distillation methods (KDs) are essential. Existing logit-based KDs apply the constant temperature scaling to all samples in dataset, limiting the utilization of knowledge inherent in each sample individually. In our approach, we classify the dataset into two categories (i.e., low energy and high energy samples) based on their energy score. Through experiments, we have confirmed that low energy samples exhibit high confidence scores, indicating certain predictions, while high energy samples yield low confidence scores, meaning uncertain predictions. To distill optimal knowledge by adjusting non-target class predictions, we apply a higher temperature to low energy samples to create smoother distributions and a lower temperature to high energy samples to achieve sharper distributions. When compared to previous logit-based and feature-based methods, our energy-based KD (Energy KD) achieves better performance on various datasets. Especially, Energy KD shows significant improvements on CIFAR-100-LT and ImageNet datasets, which contain many challenging samples. Furthermore, we propose high energy-based data augmentation (HE-DA) for further improving the performance. We demonstrate that higher performance improvement could be achieved by augmenting only a portion of the dataset rather than the entire dataset, suggesting that it can be employed on resource-limited devices. To the best of our knowledge, this paper represents the first attempt to make use of energy function in knowledge distillation and data augmentation, and we believe it will greatly contribute to future research.
>
---
#### [replaced 070] Demystifying Variational Diffusion Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.06281v2](http://arxiv.org/pdf/2401.06281v2)**

> **作者:** Fabio De Sousa Ribeiro; Ben Glocker
>
> **摘要:** Despite the growing interest in diffusion models, gaining a deep understanding of the model class remains an elusive endeavour, particularly for the uninitiated in non-equilibrium statistical physics. Thanks to the rapid rate of progress in the field, most existing work on diffusion models focuses on either applications or theoretical contributions. Unfortunately, the theoretical material is often inaccessible to practitioners and new researchers, leading to a risk of superficial understanding in ongoing research. Given that diffusion models are now an indispensable tool, a clear and consolidating perspective on the model class is needed to properly contextualize recent advances in generative modelling and lower the barrier to entry for new researchers. To that end, we revisit predecessors to diffusion models like hierarchical latent variable models and synthesize a holistic perspective using only directed graphical modelling and variational inference principles. The resulting narrative is easier to follow as it imposes fewer prerequisites on the average reader relative to the view from non-equilibrium thermodynamics or stochastic differential equations.
>
---
#### [replaced 071] Evaluating Automated Radiology Report Quality through Fine-Grained Phrasal Grounding of Clinical Findings
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01031v3](http://arxiv.org/pdf/2412.01031v3)**

> **作者:** Razi Mahmood; Pingkun Yan; Diego Machado Reyes; Ge Wang; Mannudeep K. Kalra; Parisa Kaviani; Joy T. Wu; Tanveer Syeda-Mahmood
>
> **摘要:** Several evaluation metrics have been developed recently to automatically assess the quality of generative AI reports for chest radiographs based only on textual information using lexical, semantic, or clinical named entity recognition methods. In this paper, we develop a new method of report quality evaluation by first extracting fine-grained finding patterns capturing the location, laterality, and severity of a large number of clinical findings. We then performed phrasal grounding to localize their associated anatomical regions on chest radiograph images. The textual and visual measures are then combined to rate the quality of the generated reports. We present results that compare this evaluation metric with other textual metrics on a gold standard dataset derived from the MIMIC collection and show its robustness and sensitivity to factual errors.
>
---
#### [replaced 072] DongbaMIE: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03644v4](http://arxiv.org/pdf/2503.03644v4)**

> **作者:** Xiaojun Bi; Shuo Li; Junyao Xing; Ziyue Wang; Fuwen Luo; Weizheng Qiao; Lu Han; Ziwei Sun; Peng Li; Yang Liu
>
> **备注:** Our dataset can be obtained from: https://github.com/thinklis/DongbaMIE
>
> **摘要:** Dongba pictographic is the only pictographic script still in use in the world. Its pictorial ideographic features carry rich cultural and contextual information. However, due to the lack of relevant datasets, research on semantic understanding of Dongba hieroglyphs has progressed slowly. To this end, we constructed \textbf{DongbaMIE} - the first dataset focusing on multimodal information extraction of Dongba pictographs. The dataset consists of images of Dongba hieroglyphic characters and their corresponding semantic annotations in Chinese. It contains 23,530 sentence-level and 2,539 paragraph-level high-quality text-image pairs. The annotations cover four semantic dimensions: object, action, relation and attribute. Systematic evaluation of mainstream multimodal large language models shows that the models are difficult to perform information extraction of Dongba hieroglyphs efficiently under zero-shot and few-shot learning. Although supervised fine-tuning can improve the performance, accurate extraction of complex semantics is still a great challenge at present.
>
---
#### [replaced 073] Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination
- **分类: cs.LG; cs.CV; math.OC**

- **链接: [http://arxiv.org/pdf/2311.02960v3](http://arxiv.org/pdf/2311.02960v3)**

> **作者:** Peng Wang; Xiao Li; Can Yaras; Zhihui Zhu; Laura Balzano; Wei Hu; Qing Qu
>
> **备注:** 65 pages, 17 figures
>
> **摘要:** Over the past decade, deep learning has proven to be a highly effective tool for learning meaningful features from raw data. However, it remains an open question how deep networks perform hierarchical feature learning across layers. In this work, we attempt to unveil this mystery by investigating the structures of intermediate features. Motivated by our empirical findings that linear layers mimic the roles of deep layers in nonlinear networks for feature learning, we explore how deep linear networks transform input data into output by investigating the output (i.e., features) of each layer after training in the context of multi-class classification problems. Toward this goal, we first define metrics to measure within-class compression and between-class discrimination of intermediate features, respectively. Through theoretical analysis of these two metrics, we show that the evolution of features follows a simple and quantitative pattern from shallow to deep layers when the input data is nearly orthogonal and the network weights are minimum-norm, balanced, and approximate low-rank: Each layer of the linear network progressively compresses within-class features at a geometric rate and discriminates between-class features at a linear rate with respect to the number of layers that data have passed through. To the best of our knowledge, this is the first quantitative characterization of feature evolution in hierarchical representations of deep linear networks. Empirically, our extensive experiments not only validate our theoretical results numerically but also reveal a similar pattern in deep nonlinear networks which aligns well with recent empirical studies. Moreover, we demonstrate the practical implications of our results in transfer learning. Our code is available at https://github.com/Heimine/PNC_DLN.
>
---
#### [replaced 074] Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10804v2](http://arxiv.org/pdf/2504.10804v2)**

> **作者:** Jiani Liu; Zhiyuan Wang; Zeliang Zhang; Chao Huang; Susan Liang; Yunlong Tang; Chenliang Xu
>
> **备注:** 15 pages. 7 figures
>
> **摘要:** Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.
>
---
#### [replaced 075] Fast Sampling Through The Reuse Of Attention Maps In Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.01008v4](http://arxiv.org/pdf/2401.01008v4)**

> **作者:** Rosco Hunter; Łukasz Dudziak; Mohamed S. Abdelfattah; Abhinav Mehrotra; Sourav Bhattacharya; Hongkai Wen
>
> **摘要:** Text-to-image diffusion models have demonstrated unprecedented capabilities for flexible and realistic image synthesis. Nevertheless, these models rely on a time-consuming sampling procedure, which has motivated attempts to reduce their latency. When improving efficiency, researchers often use the original diffusion model to train an additional network designed specifically for fast image generation. In contrast, our approach seeks to reduce latency directly, without any retraining, fine-tuning, or knowledge distillation. In particular, we find the repeated calculation of attention maps to be costly yet redundant, and instead suggest reusing them during sampling. Our specific reuse strategies are based on ODE theory, which implies that the later a map is reused, the smaller the distortion in the final image. We empirically compare our reuse strategies with few-step sampling procedures of comparable latency, finding that reuse generates images that are closer to those produced by the original high-latency diffusion model.
>
---
#### [replaced 076] A Review of Pseudo-Labeling for Computer Vision
- **分类: cs.CV; cs.LG; I.2.0; I.5.4; I.4.0**

- **链接: [http://arxiv.org/pdf/2408.07221v2](http://arxiv.org/pdf/2408.07221v2)**

> **作者:** Patrick Kage; Jay C. Rothenberger; Pavlos Andreadis; Dimitrios I. Diochnos
>
> **备注:** 40 pages, 4 figures, 2 tables
>
> **摘要:** Deep neural models have achieved state of the art performance on a wide range of problems in computer science, especially in computer vision. However, deep neural networks often require large datasets of labeled samples to generalize effectively, and an important area of active research is semi-supervised learning, which attempts to instead utilize large quantities of (easily acquired) unlabeled samples. One family of methods in this space is pseudo-labeling, a class of algorithms that use model outputs to assign labels to unlabeled samples which are then used as labeled samples during training. Such assigned labels, called pseudo-labels, are most commonly associated with the field of semi-supervised learning. In this work we explore a broader interpretation of pseudo-labels within both self-supervised and unsupervised methods. By drawing the connection between these areas we identify new directions when advancements in one area would likely benefit others, such as curriculum learning and self-supervised regularization.
>
---
#### [replaced 077] Relative-Interior Solution for the (Incomplete) Linear Assignment Problem with Applications to the Quadratic Assignment Problem
- **分类: math.OC; cs.CV**

- **链接: [http://arxiv.org/pdf/2301.11201v4](http://arxiv.org/pdf/2301.11201v4)**

> **作者:** Tomáš Dlask; Bogdan Savchynskyy
>
> **摘要:** We study the set of optimal solutions of the dual linear programming formulation of the linear assignment problem (LAP) to propose a method for computing a solution from the relative interior of this set. Assuming that an arbitrary dual-optimal solution and an optimal assignment are available (for which many efficient algorithms already exist), our method computes a relative-interior solution in linear time. Since the LAP occurs as a subproblem in the linear programming (LP) relaxation of the quadratic assignment problem (QAP), we employ our method as a new component in the family of dual-ascent algorithms that provide bounds on the optimal value of the QAP. To make our results applicable to the incomplete QAP, which is of interest in practical use-cases, we also provide a linear-time reduction from the incomplete LAP to the complete LAP along with a mapping that preserves optimality and membership in the relative interior. Our experiments on publicly available benchmarks indicate that our approach with relative-interior solution can frequently provide bounds near the optimum of the LP relaxation and its runtime is much lower when compared to a commercial LP solver.
>
---
#### [replaced 078] Mask of truth: model sensitivity to unexpected regions of medical images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04030v3](http://arxiv.org/pdf/2412.04030v3)**

> **作者:** Théo Sourget; Michelle Hestbek-Møller; Amelia Jiménez-Sánchez; Jack Junchi Xu; Veronika Cheplygina
>
> **备注:** Updated after publication in the Journal of Imaging Informatics in Medicine
>
> **摘要:** The development of larger models for medical image analysis has led to increased performance. However, it also affected our ability to explain and validate model decisions. Models can use non-relevant parts of images, also called spurious correlations or shortcuts, to obtain high performance on benchmark datasets but fail in real-world scenarios. In this work, we challenge the capacity of convolutional neural networks (CNN) to classify chest X-rays and eye fundus images while masking out clinically relevant parts of the image. We show that all models trained on the PadChest dataset, irrespective of the masking strategy, are able to obtain an Area Under the Curve (AUC) above random. Moreover, the models trained on full images obtain good performance on images without the region of interest (ROI), even superior to the one obtained on images only containing the ROI. We also reveal a possible spurious correlation in the Chaksu dataset while the performances are more aligned with the expectation of an unbiased model. We go beyond the performance analysis with the usage of the explainability method SHAP and the analysis of embeddings. We asked a radiology resident to interpret chest X-rays under different masking to complement our findings with clinical knowledge. Our code is available at https://github.com/TheoSourget/MMC_Masking and https://github.com/TheoSourget/MMC_Masking_EyeFundus
>
---
#### [replaced 079] Leveraging Habitat Information for Fine-grained Bird Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.14999v3](http://arxiv.org/pdf/2312.14999v3)**

> **作者:** Tin Nguyen; Peijie Chen; Anh Totti Nguyen
>
> **摘要:** Traditional bird classifiers mostly rely on the visual characteristics of birds. Some prior works even train classifiers to be invariant to the background, completely discarding the living environment of birds. Instead, we are the first to explore integrating habitat information, one of the four major cues for identifying birds by ornithologists, into modern bird classifiers. We focus on two leading model types: (1) CNNs and ViTs trained on the downstream bird datasets; and (2) original, multi-modal CLIP. Training CNNs and ViTs with habitat-augmented data results in an improvement of up to +0.83 and +0.23 points on NABirds and CUB-200, respectively. Similarly, adding habitat descriptors to the prompts for CLIP yields a substantial accuracy boost of up to +0.99 and +1.1 points on NABirds and CUB-200, respectively. We find consistent accuracy improvement after integrating habitat features into the image augmentation process and into the textual descriptors of vision-language CLIP classifiers. Code is available at: https://anonymous.4open.science/r/reasoning-8B7E/.
>
---
#### [replaced 080] GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00063v2](http://arxiv.org/pdf/2505.00063v2)**

> **作者:** Siqi Li; Yufan Shen; Xiangnan Chen; Jiayi Chen; Hengwei Ju; Haodong Duan; Song Mao; Hongbin Zhou; Bo Zhang; Bin Fu; Pinlong Cai; Licheng Wen; Botian Shi; Yong Liu; Xinyu Cai; Yu Qiao
>
> **摘要:** The rapid advancement of multimodal large language models (MLLMs) has profoundly impacted the document domain, creating a wide array of application scenarios. This progress highlights the need for a comprehensive benchmark to evaluate these models' capabilities across various document-specific tasks. However, existing benchmarks often fail to locate specific model weaknesses or guide systematic improvements. To bridge this gap, we introduce a General Document Intelligence Benchmark (GDI-Bench), featuring 2.3k images across 9 key scenarios and 19 document-specific tasks. By decoupling visual complexity and reasoning complexity, the GDI-Bench structures graded tasks that allow performance assessment by difficulty, aiding in model weakness identification and optimization guidance. We evaluate various open-source and closed-source models on GDI-Bench, conducting decoupled analyses in the visual and reasoning domains, revealing their strengths and weaknesses. To address the diverse tasks and domains in the GDI-Bench, we propose a GDI-Model that mitigates catastrophic forgetting during the supervised fine-tuning (SFT) process through an intelligence-preserving training strategy, thereby reinforcing the inherent weaknesses of the base model. Our model achieves state-of-the-art performance on previous benchmarks and the GDI-Bench. Both our benchmark and models are or will be open-sourced on https://huggingface.co/GDIBench.
>
---
#### [replaced 081] Split Gibbs Discrete Diffusion Posterior Sampling
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01161v2](http://arxiv.org/pdf/2503.01161v2)**

> **作者:** Wenda Chu; Zihui Wu; Yifan Chen; Yang Song; Yisong Yue
>
> **摘要:** We study the problem of posterior sampling in discrete-state spaces using discrete diffusion models. While posterior sampling methods for continuous diffusion models have achieved remarkable progress, analogous methods for discrete diffusion models remain challenging. In this work, we introduce a principled plug-and-play discrete diffusion posterior sampling algorithm based on split Gibbs sampling, which we call SGDD. Our algorithm enables reward-guided generation and solving inverse problems in discrete-state spaces. We demonstrate the convergence of SGDD to the target posterior distribution and verify this through controlled experiments on synthetic benchmarks. Our method enjoys state-of-the-art posterior sampling performance on a range of benchmarks for discrete data, including DNA sequence design, discrete image inverse problems, and music infilling, achieving more than 30% improved performance compared to existing baselines.
>
---
#### [replaced 082] DC4CR: When Cloud Removal Meets Diffusion Control in Remote Sensing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14785v2](http://arxiv.org/pdf/2504.14785v2)**

> **作者:** Zhenyu Yu; Mohd Yamani Idna Idris; Pei Wang
>
> **摘要:** Cloud occlusion significantly hinders remote sensing applications by obstructing surface information and complicating analysis. To address this, we propose DC4CR (Diffusion Control for Cloud Removal), a novel multimodal diffusion-based framework for cloud removal in remote sensing imagery. Our method introduces prompt-driven control, allowing selective removal of thin and thick clouds without relying on pre-generated cloud masks, thereby enhancing preprocessing efficiency and model adaptability. Additionally, we integrate low-rank adaptation for computational efficiency, subject-driven generation for improved generalization, and grouped learning to enhance performance on small datasets. Designed as a plug-and-play module, DC4CR seamlessly integrates into existing cloud removal models, providing a scalable and robust solution. Extensive experiments on the RICE and CUHK-CR datasets demonstrate state-of-the-art performance, achieving superior cloud removal across diverse conditions. This work presents a practical and efficient approach for remote sensing image processing with broad real-world applications.
>
---
#### [replaced 083] Consistent Quantity-Quality Control across Scenes for Deployment-Aware Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10473v2](http://arxiv.org/pdf/2505.10473v2)**

> **作者:** Fengdi Zhang; Hongkun Cao; Ruqi Huang
>
> **备注:** 16 pages, 7 figures, 7 tables. Project page available at https://zhang-fengdi.github.io/ControlGS/
>
> **摘要:** To reduce storage and computational costs, 3D Gaussian splatting (3DGS) seeks to minimize the number of Gaussians used while preserving high rendering quality, introducing an inherent trade-off between Gaussian quantity and rendering quality. Existing methods strive for better quantity-quality performance, but lack the ability for users to intuitively adjust this trade-off to suit practical needs such as model deployment under diverse hardware and communication constraints. Here, we present ControlGS, a 3DGS optimization method that achieves semantically meaningful and cross-scene consistent quantity-quality control. Through a single training run using a fixed setup and a user-specified hyperparameter reflecting quantity-quality preference, ControlGS can automatically find desirable quantity-quality trade-off points across diverse scenes, from compact objects to large outdoor scenes. It also outperforms baselines by achieving higher rendering quality with fewer Gaussians, and supports a broad adjustment range with stepless control over the trade-off. Project page: https://zhang-fengdi.github.io/ControlGS/
>
---
#### [replaced 084] Playmate: Flexible Control of Portrait Animation via 3D-Implicit Space Guided Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07203v3](http://arxiv.org/pdf/2502.07203v3)**

> **作者:** Xingpei Ma; Jiaran Cai; Yuansheng Guan; Shenneng Huang; Qiang Zhang; Shunsi Zhang
>
> **摘要:** Recent diffusion-based talking face generation models have demonstrated impressive potential in synthesizing videos that accurately match a speech audio clip with a given reference identity. However, existing approaches still encounter significant challenges due to uncontrollable factors, such as inaccurate lip-sync, inappropriate head posture and the lack of fine-grained control over facial expressions. In order to introduce more face-guided conditions beyond speech audio clips, a novel two-stage training framework Playmate is proposed to generate more lifelike facial expressions and talking faces. In the first stage, we introduce a decoupled implicit 3D representation along with a meticulously designed motion-decoupled module to facilitate more accurate attribute disentanglement and generate expressive talking videos directly from audio cues. Then, in the second stage, we introduce an emotion-control module to encode emotion control information into the latent space, enabling fine-grained control over emotions and thereby achieving the ability to generate talking videos with desired emotion. Extensive experiments demonstrate that Playmate not only outperforms existing state-of-the-art methods in terms of video quality, but also exhibits strong competitiveness in lip synchronization while offering improved flexibility in controlling emotion and head pose. The code will be available at https://github.com/Playmate111/Playmate.
>
---
#### [replaced 085] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01218v2](http://arxiv.org/pdf/2502.01218v2)**

> **作者:** Zhizhen Zhang; Lei Zhu; Zhen Fang; Zi Huang; Yadan Luo
>
> **摘要:** Pre-training vision-language representations on human action videos has emerged as a promising approach to reduce reliance on large-scale expert demonstrations for training embodied agents. However, prior methods often employ time contrastive learning based on goal-reaching heuristics, progressively aligning language instructions from the initial to the final frame. This overemphasis on future frames can result in erroneous vision-language associations, as actions may terminate early or include irrelevant moments in the end. To address this issue, we propose Action Temporal Coherence Learning (AcTOL) to learn ordered and continuous vision-language representations without rigid goal-based constraint. AcTOL treats a video as a continuous trajectory where it (1) contrasts semantic differences between frames to reflect their natural ordering, and (2) imposes a local Brownian bridge constraint to ensure smooth transitions across intermediate frames. Extensive imitation learning experiments on both simulated and real robots show that the pretrained features significantly enhance downstream manipulation tasks with high robustness to different linguistic styles of instructions, offering a viable pathway toward generalized embodied agents.
>
---
#### [replaced 086] Continuous Representation Methods, Theories, and Applications: An Overview and Perspectives
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15222v2](http://arxiv.org/pdf/2505.15222v2)**

> **作者:** Yisi Luo; Xile Zhao; Deyu Meng
>
> **摘要:** Recently, continuous representation methods emerge as novel paradigms that characterize the intrinsic structures of real-world data through function representations that map positional coordinates to their corresponding values in the continuous space. As compared with the traditional discrete framework, the continuous framework demonstrates inherent superiority for data representation and reconstruction (e.g., image restoration, novel view synthesis, and waveform inversion) by offering inherent advantages including resolution flexibility, cross-modal adaptability, inherent smoothness, and parameter efficiency. In this review, we systematically examine recent advancements in continuous representation frameworks, focusing on three aspects: (i) Continuous representation method designs such as basis function representation, statistical modeling, tensor function decomposition, and implicit neural representation; (ii) Theoretical foundations of continuous representations such as approximation error analysis, convergence property, and implicit regularization; (iii) Real-world applications of continuous representations derived from computer vision, graphics, bioinformatics, and remote sensing. Furthermore, we outline future directions and perspectives to inspire exploration and deepen insights to facilitate continuous representation methods, theories, and applications. All referenced works are summarized in our open-source repository: https://github.com/YisiLuo/Continuous-Representation-Zoo
>
---
