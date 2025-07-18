# 计算机视觉 cs.CV

- **最新发布 126 篇**

- **更新 53 篇**

## 最新发布

#### [new 001] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决ViT在低比特量化下的精度下降问题。提出FIMA-Q方法，通过FIM近似提升量化效果。**

- **链接: [http://arxiv.org/pdf/2506.11543v1](http://arxiv.org/pdf/2506.11543v1)**

> **作者:** Zhuguanyu Wu; Shihe Wang; Jiayi Zhang; Jiaxin Chen; Yunhong Wang
>
> **备注:** CVPR 2025 Highlight
>
> **摘要:** Post-training quantization (PTQ) has stood out as a cost-effective and promising model compression paradigm in recent years, as it avoids computationally intensive model retraining. Nevertheless, current PTQ methods for Vision Transformers (ViTs) still suffer from significant accuracy degradation, especially under low-bit quantization. To address these shortcomings, we analyze the prevailing Hessian-guided quantization loss, and uncover certain limitations of conventional Hessian approximations. By following the block-wise reconstruction framework, we propose a novel PTQ method for ViTs, dubbed FIMA-Q. Specifically, we firstly establish the connection between KL divergence and FIM, which enables fast computation of the quantization loss during reconstruction. We further propose an efficient FIM approximation method, namely DPLR-FIM, by employing the diagonal plus low-rank principle, and formulate the ultimate quantization loss. Our extensive experiments, conducted across various vision tasks with representative ViT-based architectures on public datasets, demonstrate that our method substantially promotes the accuracy compared to the state-of-the-art approaches, especially in the case of low-bit quantization. The source code is available at https://github.com/ShiheWang/FIMA-Q.
>
---
#### [new 002] O2Former:Direction-Aware and Multi-Scale Query Enhancement for SAR Ship Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于SAR图像中船舶实例分割任务，解决尺度变化、密集目标和边界模糊问题。提出O2Former框架，包含多尺度查询生成和方向感知模块，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.11913v1](http://arxiv.org/pdf/2506.11913v1)**

> **作者:** F. Gao; Y Li; X He; J Sun; J Wang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Instance segmentation of ships in synthetic aperture radar (SAR) imagery is critical for applications such as maritime monitoring, environmental analysis, and national security. SAR ship images present challenges including scale variation, object density, and fuzzy target boundary, which are often overlooked in existing methods, leading to suboptimal performance. In this work, we propose O2Former, a tailored instance segmentation framework that extends Mask2Former by fully leveraging the structural characteristics of SAR imagery. We introduce two key components. The first is the Optimized Query Generator(OQG). It enables multi-scale feature interaction by jointly encoding shallow positional cues and high-level semantic information. This improves query quality and convergence efficiency. The second component is the Orientation-Aware Embedding Module(OAEM). It enhances directional sensitivity through direction-aware convolution and polar-coordinate encoding. This effectively addresses the challenge of uneven target orientations in SAR scenes. Together, these modules facilitate precise feature alignment from backbone to decoder and strengthen the model's capacity to capture fine-grained structural details. Extensive experiments demonstrate that O2Former outperforms state of the art instance segmentation baselines, validating its effectiveness and generalization on SAR ship datasets.
>
---
#### [new 003] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search
- **分类: cs.CV**

- **简介: 该论文属于视频描述任务，旨在解决现有评估基准不足的问题。提出AutoCaption框架，利用MCTS生成多样化的视频描述，构建了MCTS-VCB基准，并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2506.11155v1](http://arxiv.org/pdf/2506.11155v1)**

> **作者:** Linhao Yu; Xinguang Ji; Yahui Liu; Fanheng Kong; Chenxi Sun; Jingyuan Zhang; Hongzhi Zhang; V. W.; Fuzheng Zhang; Deyi Xiong
>
> **备注:** 28 pages; ACL 2025(main)
>
> **摘要:** Video captioning can be used to assess the video understanding capabilities of Multimodal Large Language Models (MLLMs). However, existing benchmarks and evaluation protocols suffer from crucial issues, such as inadequate or homogeneous creation of key points, exorbitant cost of data creation, and limited evaluation scopes. To address these issues, we propose an automatic framework, named AutoCaption, which leverages Monte Carlo Tree Search (MCTS) to construct numerous and diverse descriptive sentences (\textit{i.e.}, key points) that thoroughly represent video content in an iterative way. This iterative captioning strategy enables the continuous enhancement of video details such as actions, objects' attributes, environment details, etc. We apply AutoCaption to curate MCTS-VCB, a fine-grained video caption benchmark covering video details, thereby enabling a comprehensive evaluation of MLLMs on the video captioning task. We evaluate more than 20 open- and closed-source MLLMs of varying sizes on MCTS-VCB. Results show that MCTS-VCB can effectively and comprehensively evaluate the video captioning capability, with Gemini-1.5-Pro achieving the highest F1 score of 71.2. Interestingly, we fine-tune InternVL2.5-8B with the AutoCaption-generated data, which helps the model achieve an overall improvement of 25.0% on MCTS-VCB and 16.3% on DREAM-1K, further demonstrating the effectiveness of AutoCaption. The code and data are available at https://github.com/tjunlp-lab/MCTS-VCB.
>
---
#### [new 004] Teaching in adverse scenes: a statistically feedback-driven threshold and mask adjustment teacher-student framework for object detection in UAV images under adverse scenes
- **分类: cs.CV**

- **简介: 该论文属于无人机目标检测任务，解决恶劣场景下域适应问题。提出SF-TMAT框架，通过动态调整掩码和阈值提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.11175v1](http://arxiv.org/pdf/2506.11175v1)**

> **作者:** Hongyu Chen; Jiping Liu; Yong Wang; Jun Zhu; Dejun Feng; Yakun Xie
>
> **备注:** The manuscript has been accepted by ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Unsupervised Domain Adaptation (UDA) has shown promise in effectively alleviating the performance degradation caused by domain gaps between source and target domains, and it can potentially be generalized to UAV object detection in adverse scenes. However, existing UDA studies are based on natural images or clear UAV imagery, and research focused on UAV imagery in adverse conditions is still in its infancy. Moreover, due to the unique perspective of UAVs and the interference from adverse conditions, these methods often fail to accurately align features and are influenced by limited or noisy pseudo-labels. To address this, we propose the first benchmark for UAV object detection in adverse scenes, the Statistical Feedback-Driven Threshold and Mask Adjustment Teacher-Student Framework (SF-TMAT). Specifically, SF-TMAT introduces a design called Dynamic Step Feedback Mask Adjustment Autoencoder (DSFMA), which dynamically adjusts the mask ratio and reconstructs feature maps by integrating training progress and loss feedback. This approach dynamically adjusts the learning focus at different training stages to meet the model's needs for learning features at varying levels of granularity. Additionally, we propose a unique Variance Feedback Smoothing Threshold (VFST) strategy, which statistically computes the mean confidence of each class and dynamically adjusts the selection threshold by incorporating a variance penalty term. This strategy improves the quality of pseudo-labels and uncovers potentially valid labels, thus mitigating domain bias. Extensive experiments demonstrate the superiority and generalization capability of the proposed SF-TMAT in UAV object detection under adverse scene conditions. The Code is released at https://github.com/ChenHuyoo .
>
---
#### [new 005] Linearly Solving Robust Rotation Estimation
- **分类: cs.CV; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于旋转估计任务，解决非线性优化难题，通过线性方法和投票机制实现鲁棒、快速的旋转估计。**

- **链接: [http://arxiv.org/pdf/2506.11547v1](http://arxiv.org/pdf/2506.11547v1)**

> **作者:** Yinlong Liu; Tianyu Huang; Zhi-Xin Yang
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Rotation estimation plays a fundamental role in computer vision and robot tasks, and extremely robust rotation estimation is significantly useful for safety-critical applications. Typically, estimating a rotation is considered a non-linear and non-convex optimization problem that requires careful design. However, in this paper, we provide some new perspectives that solving a rotation estimation problem can be reformulated as solving a linear model fitting problem without dropping any constraints and without introducing any singularities. In addition, we explore the dual structure of a rotation motion, revealing that it can be represented as a great circle on a quaternion sphere surface. Accordingly, we propose an easily understandable voting-based method to solve rotation estimation. The proposed method exhibits exceptional robustness to noise and outliers and can be computed in parallel with graphics processing units (GPUs) effortlessly. Particularly, leveraging the power of GPUs, the proposed method can obtain a satisfactory rotation solution for large-scale($10^6$) and severely corrupted (99$\%$ outlier ratio) rotation estimation problems under 0.5 seconds. Furthermore, to validate our theoretical framework and demonstrate the superiority of our proposed method, we conduct controlled experiments and real-world dataset experiments. These experiments provide compelling evidence supporting the effectiveness and robustness of our approach in solving rotation estimation problems.
>
---
#### [new 006] AlignHuman: Improving Motion and Fidelity via Timestep-Segment Preference Optimization for Audio-Driven Human Animation
- **分类: cs.CV**

- **简介: 该论文属于音频驱动的人体动画任务，旨在解决运动自然性与视觉保真度之间的权衡问题。通过TPO和LoRAs优化，提升生成质量并加速推理。**

- **链接: [http://arxiv.org/pdf/2506.11144v1](http://arxiv.org/pdf/2506.11144v1)**

> **作者:** Chao Liang; Jianwen Jiang; Wang Liao; Jiaqi Yang; Zerong zheng; Weihong Zeng; Han Liang
>
> **备注:** Homepage: https://alignhuman.github.io/
>
> **摘要:** Recent advancements in human video generation and animation tasks, driven by diffusion models, have achieved significant progress. However, expressive and realistic human animation remains challenging due to the trade-off between motion naturalness and visual fidelity. To address this, we propose \textbf{AlignHuman}, a framework that combines Preference Optimization as a post-training technique with a divide-and-conquer training strategy to jointly optimize these competing objectives. Our key insight stems from an analysis of the denoising process across timesteps: (1) early denoising timesteps primarily control motion dynamics, while (2) fidelity and human structure can be effectively managed by later timesteps, even if early steps are skipped. Building on this observation, we propose timestep-segment preference optimization (TPO) and introduce two specialized LoRAs as expert alignment modules, each targeting a specific dimension in its corresponding timestep interval. The LoRAs are trained using their respective preference data and activated in the corresponding intervals during inference to enhance motion naturalness and fidelity. Extensive experiments demonstrate that AlignHuman improves strong baselines and reduces NFEs during inference, achieving a 3.3$\times$ speedup (from 100 NFEs to 30 NFEs) with minimal impact on generation quality. Homepage: \href{https://alignhuman.github.io/}{https://alignhuman.github.io/}
>
---
#### [new 007] DMAF-Net: An Effective Modality Rebalancing Framework for Incomplete Multi-Modal Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决多模态数据不完整时的模态不平衡问题。提出DMAF-Net模型，通过动态融合、关系蒸馏和训练监控提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.11691v1](http://arxiv.org/pdf/2506.11691v1)**

> **作者:** Libin Lan; Hongxing Li; Zunhui Xia; Yudong Zhang
>
> **备注:** 12 pages, 4 figures, 3 tables
>
> **摘要:** Incomplete multi-modal medical image segmentation faces critical challenges from modality imbalance, including imbalanced modality missing rates and heterogeneous modality contributions. Due to their reliance on idealized assumptions of complete modality availability, existing methods fail to dynamically balance contributions and neglect the structural relationships between modalities, resulting in suboptimal performance in real-world clinical scenarios. To address these limitations, we propose a novel model, named Dynamic Modality-Aware Fusion Network (DMAF-Net). The DMAF-Net adopts three key ideas. First, it introduces a Dynamic Modality-Aware Fusion (DMAF) module to suppress missing-modality interference by combining transformer attention with adaptive masking and weight modality contributions dynamically through attention maps. Second, it designs a synergistic Relation Distillation and Prototype Distillation framework to enforce global-local feature alignment via covariance consistency and masked graph attention, while ensuring semantic consistency through cross-modal class-specific prototype alignment. Third, it presents a Dynamic Training Monitoring (DTM) strategy to stabilize optimization under imbalanced missing rates by tracking distillation gaps in real-time, and to balance convergence speeds across modalities by adaptively reweighting losses and scaling gradients. Extensive experiments on BraTS2020 and MyoPS2020 demonstrate that DMAF-Net outperforms existing methods for incomplete multi-modal medical image segmentation. Extensive experiments on BraTS2020 and MyoPS2020 demonstrate that DMAF-Net outperforms existing methods for incomplete multi-modal medical image segmentation. Our code is available at https://github.com/violet-42/DMAF-Net.
>
---
#### [new 008] Technical Report for Argoverse2 Scenario Mining Challenges on Iterative Error Correction and Spatially-Aware Prompting
- **分类: cs.CV; cs.SE**

- **简介: 该论文属于自动驾驶场景挖掘任务，解决LLM生成代码中的错误和空间关系理解问题，提出迭代纠错和空间提示优化方法。**

- **链接: [http://arxiv.org/pdf/2506.11124v1](http://arxiv.org/pdf/2506.11124v1)**

> **作者:** Yifei Chen; Ross Greer
>
> **摘要:** Scenario mining from extensive autonomous driving datasets, such as Argoverse 2, is crucial for the development and validation of self-driving systems. The RefAV framework represents a promising approach by employing Large Language Models (LLMs) to translate natural-language queries into executable code for identifying relevant scenarios. However, this method faces challenges, including runtime errors stemming from LLM-generated code and inaccuracies in interpreting parameters for functions that describe complex multi-object spatial relationships. This technical report introduces two key enhancements to address these limitations: (1) a fault-tolerant iterative code-generation mechanism that refines code by re-prompting the LLM with error feedback, and (2) specialized prompt engineering that improves the LLM's comprehension and correct application of spatial-relationship functions. Experiments on the Argoverse 2 validation set with diverse LLMs-Qwen2.5-VL-7B, Gemini 2.5 Flash, and Gemini 2.5 Pro-show consistent gains across multiple metrics; most notably, the proposed system achieves a HOTA-Temporal score of 52.37 on the official test set using Gemini 2.5 Pro. These results underline the efficacy of the proposed techniques for reliable, high-precision scenario mining.
>
---
#### [new 009] Rethinking Multilingual Vision-Language Translation: Dataset, Evaluation, and Adaptation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言翻译任务，旨在解决多语言文本识别与翻译中的数据、模型和评估问题，提出新数据集和评估方法以提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11820v1](http://arxiv.org/pdf/2506.11820v1)**

> **作者:** Xintong Wang; Jingheng Pan; Yixiao Liu; Xiaohu Zhao; Chenyang Lyu; Minghao Wu; Chris Biemann; Longyue Wang; Linlong Xu; Weihua Luo; Kaifu Zhang
>
> **摘要:** Vision-Language Translation (VLT) is a challenging task that requires accurately recognizing multilingual text embedded in images and translating it into the target language with the support of visual context. While recent Large Vision-Language Models (LVLMs) have demonstrated strong multilingual and visual understanding capabilities, there is a lack of systematic evaluation and understanding of their performance on VLT. In this work, we present a comprehensive study of VLT from three key perspectives: data quality, model architecture, and evaluation metrics. (1) We identify critical limitations in existing datasets, particularly in semantic and cultural fidelity, and introduce AibTrans -- a multilingual, parallel, human-verified dataset with OCR-corrected annotations. (2) We benchmark 11 commercial LVLMs/LLMs and 6 state-of-the-art open-source models across end-to-end and cascaded architectures, revealing their OCR dependency and contrasting generation versus reasoning behaviors. (3) We propose Density-Aware Evaluation to address metric reliability issues under varying contextual complexity, introducing the DA Score as a more robust measure of translation quality. Building upon these findings, we establish a new evaluation benchmark for VLT. Notably, we observe that fine-tuning LVLMs on high-resource language pairs degrades cross-lingual performance, and we propose a balanced multilingual fine-tuning strategy that effectively adapts LVLMs to VLT without sacrificing their generalization ability.
>
---
#### [new 010] MTabVQA: Evaluating Multi-Tabular Reasoning of Language Models in Visual Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答任务，旨在解决多表格图像中的推理问题。提出MTabVQA基准，评估语言模型在多表格视觉空间中的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.11684v1](http://arxiv.org/pdf/2506.11684v1)**

> **作者:** Anshul Singh; Chris Biemann; Jan Strich
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable capabilities in interpreting visual layouts and text. However, a significant challenge remains in their ability to interpret robustly and reason over multi-tabular data presented as images, a common occurrence in real-world scenarios like web pages and digital documents. Existing benchmarks typically address single tables or non-visual data (text/structured). This leaves a critical gap: they don't assess the ability to parse diverse table images, correlate information across them, and perform multi-hop reasoning on the combined visual data. We introduce MTabVQA, a novel benchmark specifically designed for multi-tabular visual question answering to bridge that gap. MTabVQA comprises 3,745 complex question-answer pairs that necessitate multi-hop reasoning across several visually rendered table images. We provide extensive benchmark results for state-of-the-art VLMs on MTabVQA, revealing significant performance limitations. We further investigate post-training techniques to enhance these reasoning abilities and release MTabVQA-Instruct, a large-scale instruction-tuning dataset. Our experiments show that fine-tuning VLMs with MTabVQA-Instruct substantially improves their performance on visual multi-tabular reasoning. Code and dataset (https://huggingface.co/datasets/mtabvqa/MTabVQA-Eval) are available online (https://anonymous.4open.science/r/MTabVQA-EMNLP-B16E).
>
---
#### [new 011] VFaith: Do Large Multimodal Models Really Reason on Seen Images Rather than Previous Memories?
- **分类: cs.CV**

- **简介: 该论文属于多模态模型推理任务，旨在解决模型是否真正依赖视觉信息而非记忆进行推理的问题。通过构建VFaith-Bench基准和编辑工具，评估模型的视觉推理能力与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11571v1](http://arxiv.org/pdf/2506.11571v1)**

> **作者:** Jiachen Yu; Yufei Zhan; Ziheng Wu; Yousong Zhu; Jinqiao Wang; Minghui Qiu
>
> **摘要:** Recent extensive works have demonstrated that by introducing long CoT, the capabilities of MLLMs to solve complex problems can be effectively enhanced. However, the reasons for the effectiveness of such paradigms remain unclear. It is challenging to analysis with quantitative results how much the model's specific extraction of visual cues and its subsequent so-called reasoning during inference process contribute to the performance improvements. Therefore, evaluating the faithfulness of MLLMs' reasoning to visual information is crucial. To address this issue, we first present a cue-driven automatic and controllable editing pipeline with the help of GPT-Image-1. It enables the automatic and precise editing of specific visual cues based on the instruction. Furthermore, we introduce VFaith-Bench, the first benchmark to evaluate MLLMs' visual reasoning capabilities and analyze the source of such capabilities with an emphasis on the visual faithfulness. Using the designed pipeline, we constructed comparative question-answer pairs by altering the visual cues in images that are crucial for solving the original reasoning problem, thereby changing the question's answer. By testing similar questions with images that have different details, the average accuracy reflects the model's visual reasoning ability, while the difference in accuracy before and after editing the test set images effectively reveals the relationship between the model's reasoning ability and visual perception. We further designed specific metrics to expose this relationship. VFaith-Bench includes 755 entries divided into five distinct subsets, along with an additional human-labeled perception task. We conducted in-depth testing and analysis of existing mainstream flagship models and prominent open-source model series/reasoning models on VFaith-Bench, further investigating the underlying factors of their reasoning capabilities.
>
---
#### [new 012] CLIP Meets Diffusion: A Synergistic Approach to Anomaly Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于异常检测任务，旨在解决异常定义模糊、类型多样及数据稀缺的问题。通过融合CLIP和扩散模型，提取多尺度特征进行异常分割与分类。**

- **链接: [http://arxiv.org/pdf/2506.11772v1](http://arxiv.org/pdf/2506.11772v1)**

> **作者:** Byeongchan Lee; John Won; Seunghyun Lee; Jinwoo Shin
>
> **摘要:** Anomaly detection is a complex problem due to the ambiguity in defining anomalies, the diversity of anomaly types (e.g., local and global defect), and the scarcity of training data. As such, it necessitates a comprehensive model capable of capturing both low-level and high-level features, even with limited data. To address this, we propose CLIPFUSION, a method that leverages both discriminative and generative foundation models. Specifically, the CLIP-based discriminative model excels at capturing global features, while the diffusion-based generative model effectively captures local details, creating a synergistic and complementary approach. Notably, we introduce a methodology for utilizing cross-attention maps and feature maps extracted from diffusion models specifically for anomaly detection. Experimental results on benchmark datasets (MVTec-AD, VisA) demonstrate that CLIPFUSION consistently outperforms baseline methods, achieving outstanding performance in both anomaly segmentation and classification. We believe that our method underscores the effectiveness of multi-modal and multi-model fusion in tackling the multifaceted challenges of anomaly detection, providing a scalable solution for real-world applications.
>
---
#### [new 013] Manager: Aggregating Insights from Unimodal Experts in Two-Tower VLMs and MLLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型任务，旨在解决两塔架构中多模态对齐不足的问题。提出Manager模块，有效融合不同层次的单模态知识，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11515v1](http://arxiv.org/pdf/2506.11515v1)**

> **作者:** Xiao Xu; Libo Qin; Wanxiang Che; Min-Yen Kan
>
> **备注:** Accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). June 2025. DOI: https://doi.org/10.1109/TCSVT.2025.3578266
>
> **摘要:** Two-Tower Vision--Language Models (VLMs) have demonstrated strong performance across various downstream VL tasks. While BridgeTower further enhances performance by building bridges between encoders, it \textit{(i)} suffers from ineffective layer-by-layer utilization of unimodal representations, \textit{(ii)} restricts the flexible exploitation of different levels of unimodal semantic knowledge, and \textit{(iii)} is limited to the evaluation on traditional low-resolution datasets only with the Two-Tower VLM architecture. In this work, we propose Manager, a lightweight, efficient and effective plugin that adaptively aggregates insights from different levels of pre-trained unimodal experts to facilitate more comprehensive VL alignment and fusion. First, under the Two-Tower VLM architecture, we introduce ManagerTower, a novel VLM that introduces the manager in each cross-modal layer. Whether with or without VL pre-training, ManagerTower outperforms previous strong baselines and achieves superior performance on 4 downstream VL tasks. Moreover, we extend our exploration to the latest Multimodal Large Language Model (MLLM) architecture. We demonstrate that LLaVA-OV-Manager significantly boosts the zero-shot performance of LLaVA-OV across different categories of capabilities, images, and resolutions on 20 downstream datasets, whether the multi-grid algorithm is enabled or not. In-depth analysis reveals that both our manager and the multi-grid algorithm can be viewed as a plugin that improves the visual representation by capturing more diverse visual details from two orthogonal perspectives (depth and width). Their synergy can mitigate the semantic ambiguity caused by the multi-grid algorithm and further improve performance. Code and models are available at https://github.com/LooperXX/ManagerTower.
>
---
#### [new 014] AgentSense: Virtual Sensor Data Generation Using LLM Agent in Simulated Home Environments
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于智能家居中的行为识别任务，旨在解决真实数据不足的问题。通过生成虚拟传感器数据来提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11773v1](http://arxiv.org/pdf/2506.11773v1)**

> **作者:** Zikang Leng; Megha Thukral; Yaqi Liu; Hrudhai Rajasekhar; Shruthi K. Hiremath; Thomas Plötz
>
> **摘要:** A major obstacle in developing robust and generalizable smart home-based Human Activity Recognition (HAR) systems is the lack of large-scale, diverse labeled datasets. Variability in home layouts, sensor configurations, and user behavior adds further complexity, as individuals follow varied routines and perform activities in distinct ways. Building HAR systems that generalize well requires training data that captures the diversity across users and environments. To address these challenges, we introduce AgentSense, a virtual data generation pipeline where diverse personas are generated by leveraging Large Language Models. These personas are used to create daily routines, which are then decomposed into low-level action sequences. Subsequently, the actions are executed in a simulated home environment called VirtualHome that we extended with virtual ambient sensors capable of recording the agents activities as they unfold. Overall, AgentSense enables the generation of rich, virtual sensor datasets that represent a wide range of users and home settings. Across five benchmark HAR datasets, we show that leveraging our virtual sensor data substantially improves performance, particularly when real data are limited. Notably, models trained on a combination of virtual data and just a few days of real data achieve performance comparable to those trained on the entire real datasets. These results demonstrate and prove the potential of virtual data to address one of the most pressing challenges in ambient sensing, which is the distinct lack of large-scale, annotated datasets without requiring any manual data collection efforts.
>
---
#### [new 015] Methods for evaluating the resolution of 3D data derived from satellite images
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于3D数据质量评估任务，旨在解决卫星影像生成的3D数据分辨率评价问题，提出基于激光雷达的自动化评估方法与流程。**

- **链接: [http://arxiv.org/pdf/2506.11876v1](http://arxiv.org/pdf/2506.11876v1)**

> **作者:** Christina Selby; Holden Bindl; Tyler Feldman; Andrew Skow; Nicolas Norena Acosta; Shea Hagstrom; Myron Brown
>
> **备注:** 11 pages, 13 figures
>
> **摘要:** 3D data derived from satellite images is essential for scene modeling applications requiring large-scale coverage or involving locations not accessible by airborne lidar or cameras. Measuring the resolution of this data is important for determining mission utility and tracking improvements. In this work, we consider methods to evaluate the resolution of point clouds, digital surface models, and 3D mesh models. We describe 3D metric evaluation tools and workflows that enable automated evaluation based on high-resolution reference airborne lidar, and we present results of analyses with data of varying quality.
>
---
#### [new 016] OV-MAP : Open-Vocabulary Zero-Shot 3D Instance Segmentation Map for Robots
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D实例分割任务，解决机器人在开放世界中对象识别精度不足的问题。通过融合2D掩码和深度图像，实现零样本3D实例分割。**

- **链接: [http://arxiv.org/pdf/2506.11585v1](http://arxiv.org/pdf/2506.11585v1)**

> **作者:** Juno Kim; Yesol Park; Hye-Jung Yoon; Byoung-Tak Zhang
>
> **备注:** Accepted at IROS 2024
>
> **摘要:** We introduce OV-MAP, a novel approach to open-world 3D mapping for mobile robots by integrating open-features into 3D maps to enhance object recognition capabilities. A significant challenge arises when overlapping features from adjacent voxels reduce instance-level precision, as features spill over voxel boundaries, blending neighboring regions together. Our method overcomes this by employing a class-agnostic segmentation model to project 2D masks into 3D space, combined with a supplemented depth image created by merging raw and synthetic depth from point clouds. This approach, along with a 3D mask voting mechanism, enables accurate zero-shot 3D instance segmentation without relying on 3D supervised segmentation models. We assess the effectiveness of our method through comprehensive experiments on public datasets such as ScanNet200 and Replica, demonstrating superior zero-shot performance, robustness, and adaptability across diverse environments. Additionally, we conducted real-world experiments to demonstrate our method's adaptability and robustness when applied to diverse real-world environments.
>
---
#### [new 017] Environmental Change Detection: Toward a Practical Task of Scene Change Detection
- **分类: cs.CV**

- **简介: 该论文提出Environmental Change Detection（ECD）任务，解决实际中参考图像与查询场景视角不一致的问题。通过多参考候选和语义表示聚合，提升变化检测性能。**

- **链接: [http://arxiv.org/pdf/2506.11481v1](http://arxiv.org/pdf/2506.11481v1)**

> **作者:** Kyusik Cho; Suhan Woo; Hongje Seong; Euntai Kim
>
> **备注:** Preprint. Under review
>
> **摘要:** Humans do not memorize everything. Thus, humans recognize scene changes by exploring the past images. However, available past (i.e., reference) images typically represent nearby viewpoints of the present (i.e., query) scene, rather than the identical view. Despite this practical limitation, conventional Scene Change Detection (SCD) has been formalized under an idealized setting in which reference images with matching viewpoints are available for every query. In this paper, we push this problem toward a practical task and introduce Environmental Change Detection (ECD). A key aspect of ECD is to avoid unrealistically aligned query-reference pairs and rely solely on environmental cues. Inspired by real-world practices, we provide these cues through a large-scale database of uncurated images. To address this new task, we propose a novel framework that jointly understands spatial environments and detects changes. The main idea is that matching at the same spatial locations between a query and a reference may lead to a suboptimal solution due to viewpoint misalignment and limited field-of-view (FOV) coverage. We deal with this limitation by leveraging multiple reference candidates and aggregating semantically rich representations for change detection. We evaluate our framework on three standard benchmark sets reconstructed for ECD, and significantly outperform a naive combination of state-of-the-art methods while achieving comparable performance to the oracle setting. The code will be released upon acceptance.
>
---
#### [new 018] Digitization of Document and Information Extraction using OCR
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于文档信息提取任务，旨在解决从扫描件和数字文件中准确提取结构化信息的问题。通过结合OCR与大语言模型，提升提取的准确性和语义理解能力。**

- **链接: [http://arxiv.org/pdf/2506.11156v1](http://arxiv.org/pdf/2506.11156v1)**

> **作者:** Rasha Sinha; Rekha B S
>
> **摘要:** Retrieving accurate details from documents is a crucial task, especially when handling a combination of scanned images and native digital formats. This document presents a combined framework for text extraction that merges Optical Character Recognition (OCR) techniques with Large Language Models (LLMs) to deliver structured outputs enriched by contextual understanding and confidence indicators. Scanned files are processed using OCR engines, while digital files are interpreted through layout-aware libraries. The extracted raw text is subsequently analyzed by an LLM to identify key-value pairs and resolve ambiguities. A comparative analysis of different OCR tools is presented to evaluate their effectiveness concerning accuracy, layout recognition, and processing speed. The approach demonstrates significant improvements over traditional rule-based and template-based methods, offering enhanced flexibility and semantic precision across different document categories
>
---
#### [new 019] AgriPotential: A Novel Multi-Spectral and Multi-Temporal Remote Sensing Dataset for Agricultural Potentials
- **分类: cs.CV; eess.IV**

- **简介: 本文介绍AgriPotential数据集，用于农业潜力预测。该任务旨在通过遥感数据支持可持续土地利用规划，解决农业分类与建模问题。**

- **链接: [http://arxiv.org/pdf/2506.11740v1](http://arxiv.org/pdf/2506.11740v1)**

> **作者:** Mohammad El Sakka; Caroline De Pourtales; Lotfi Chaari; Josiane Mothe
>
> **摘要:** Remote sensing has emerged as a critical tool for large-scale Earth monitoring and land management. In this paper, we introduce AgriPotential, a novel benchmark dataset composed of Sentinel-2 satellite imagery spanning multiple months. The dataset provides pixel-level annotations of agricultural potentials for three major crop types - viticulture, market gardening, and field crops - across five ordinal classes. AgriPotential supports a broad range of machine learning tasks, including ordinal regression, multi-label classification, and spatio-temporal modeling. The data covers diverse areas in Southern France, offering rich spectral information. AgriPotential is the first public dataset designed specifically for agricultural potential prediction, aiming to improve data-driven approaches to sustainable land use planning. The dataset and the code are freely accessible at: https://zenodo.org/records/15556484
>
---
#### [new 020] Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于人类活动识别任务，比较BiLSTM与CNN+GRU模型在WiFi CSI数据上的表现，分析不同数据集对模型效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.11165v1](http://arxiv.org/pdf/2506.11165v1)**

> **作者:** Almustapha A. Wakili; Babajide J. Asaju; Woosub Jung
>
> **备注:** This Paper has been Accepted and will appear in the 23rd IEEE/ACIS International Conference on Software Engineering, Management and Applications (SERA 2025)
>
> **摘要:** This paper compares the performance of BiLSTM and CNN+GRU deep learning models for Human Activity Recognition (HAR) on two WiFi-based Channel State Information (CSI) datasets: UT-HAR and NTU-Fi HAR. The findings indicate that the CNN+GRU model has a higher accuracy on the UT-HAR dataset (95.20%) thanks to its ability to extract spatial features. In contrast, the BiLSTM model performs better on the high-resolution NTU-Fi HAR dataset (92.05%) by extracting long-term temporal dependencies more effectively. The findings strongly emphasize the critical role of dataset characteristics and preprocessing techniques in model performance improvement. We also show the real-world applicability of such models in applications like healthcare and intelligent home systems, highlighting their potential for unobtrusive activity recognition.
>
---
#### [new 021] Simple Radiology VLLM Test-time Scaling with Thought Graph Traversal
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在提升放射学VLLM的推理能力。通过引入TGT框架和推理预算策略，增强模型生成准确报告的能力。**

- **链接: [http://arxiv.org/pdf/2506.11989v1](http://arxiv.org/pdf/2506.11989v1)**

> **作者:** Yue Yao; Zelin Wen; Yan Tong; Xinyu Tian; Xuqing Li; Xiao Ma; Dongliang Xu; Tom Gedeon
>
> **备注:** arXiv admin note: text overlap with arXiv:2404.11209 by other authors
>
> **摘要:** Test-time scaling offers a promising way to improve the reasoning performance of vision-language large models (VLLMs) without additional training. In this paper, we explore a simple but effective approach for applying test-time scaling to radiology report generation. Specifically, we introduce a lightweight Thought Graph Traversal (TGT) framework that guides the model to reason through organ-specific findings in a medically coherent order. This framework integrates structured medical priors into the prompt, enabling deeper and more logical analysis with no changes to the underlying model. To further enhance reasoning depth, we apply a reasoning budget forcing strategy that adjusts the model's inference depth at test time by dynamically extending its generation process. This simple yet powerful combination allows a frozen radiology VLLM to self-correct and generate more accurate, consistent chest X-ray reports. Our method outperforms baseline prompting approaches on standard benchmarks, and also reveals dataset biases through traceable reasoning paths. Code and prompts are open-sourced for reproducibility at https://github.com/glerium/Thought-Graph-Traversal.
>
---
#### [new 022] HyBiomass: Global Hyperspectral Imagery Benchmark Dataset for Evaluating Geospatial Foundation Models in Forest Aboveground Biomass Estimation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于森林地上生物量估计任务，旨在解决Geo-FMs在不同地区和传感器下的评估问题。工作包括构建全球高光谱数据集并验证模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11314v1](http://arxiv.org/pdf/2506.11314v1)**

> **作者:** Aaron Banze; Timothée Stassin; Nassim Ait Ali Braham; Rıdvan Salih Kuzu; Simon Besnard; Michael Schmitt
>
> **摘要:** Comprehensive evaluation of geospatial foundation models (Geo-FMs) requires benchmarking across diverse tasks, sensors, and geographic regions. However, most existing benchmark datasets are limited to segmentation or classification tasks, and focus on specific geographic areas. To address this gap, we introduce a globally distributed dataset for forest aboveground biomass (AGB) estimation, a pixel-wise regression task. This benchmark dataset combines co-located hyperspectral imagery (HSI) from the Environmental Mapping and Analysis Program (EnMAP) satellite and predictions of AGB density estimates derived from the Global Ecosystem Dynamics Investigation lidars, covering seven continental regions. Our experimental results on this dataset demonstrate that the evaluated Geo-FMs can match or, in some cases, surpass the performance of a baseline U-Net, especially when fine-tuning the encoder. We also find that the performance difference between the U-Net and Geo-FMs depends on the dataset size for each region and highlight the importance of the token patch size in the Vision Transformer backbone for accurate predictions in pixel-wise regression tasks. By releasing this globally distributed hyperspectral benchmark dataset, we aim to facilitate the development and evaluation of Geo-FMs for HSI applications. Leveraging this dataset additionally enables research into geographic bias and generalization capacity of Geo-FMs. The dataset and source code will be made publicly available.
>
---
#### [new 023] Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本-图像计划生成任务，旨在解决多模态一致性与连贯性问题。通过分步生成与优化框架提升文本与图像的对齐和逻辑连贯性。**

- **链接: [http://arxiv.org/pdf/2506.11380v1](http://arxiv.org/pdf/2506.11380v1)**

> **作者:** Xiaoxin Lu; Ranran Haoran Zhang; Yusen Zhang; Rui Zhang
>
> **备注:** 18 pages, 10 figures; Accepted to ACL 2025 Findings
>
> **摘要:** People get informed of a daily task plan through diverse media involving both texts and images. However, most prior research only focuses on LLM's capability of textual plan generation. The potential of large-scale models in providing text-image plans remains understudied. Generating high-quality text-image plans faces two main challenges: ensuring consistent alignment between two modalities and keeping coherence among visual steps. To address these challenges, we propose a novel framework that generates and refines text-image plans step-by-step. At each iteration, our framework (1) drafts the next textual step based on the prediction history; (2) edits the last visual step to obtain the next one; (3) extracts PDDL-like visual information; and (4) refines the draft with the extracted visual information. The textual and visual step produced in stage (4) and (2) will then serve as inputs for the next iteration. Our approach offers a plug-and-play improvement to various backbone models, such as Mistral-7B, Gemini-1.5, and GPT-4o. To evaluate the effectiveness of our approach, we collect a new benchmark consisting of 1,100 tasks and their text-image pair solutions covering 11 daily topics. We also design and validate a new set of metrics to evaluate the multimodal consistency and coherence in text-image plans. Extensive experiment results show the effectiveness of our approach on a range of backbone models against competitive baselines. Our code and data are available at https://github.com/psunlpgroup/MPlanner.
>
---
#### [new 024] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning
- **分类: cs.CV**

- **简介: 该论文属于持续多模态指令调优任务，解决固定架构难以适应新任务的问题。提出D-MoLE方法，动态分配LoRA专家并调整模态更新比例，实现高效持续学习。**

- **链接: [http://arxiv.org/pdf/2506.11672v1](http://arxiv.org/pdf/2506.11672v1)**

> **作者:** Chendi Ge; Xin Wang; Zeyang Zhang; Hong Chen; Jiapei Fan; Longtao Huang; Hui Xue; Wenwu Zhu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Continual multimodal instruction tuning is crucial for adapting Multimodal Large Language Models (MLLMs) to evolving tasks. However, most existing methods adopt a fixed architecture, struggling with adapting to new tasks due to static model capacity. We propose to evolve the architecture under parameter budgets for dynamic task adaptation, which remains unexplored and imposes two challenges: 1) task architecture conflict, where different tasks require varying layer-wise adaptations, and 2) modality imbalance, where different tasks rely unevenly on modalities, leading to unbalanced updates. To address these challenges, we propose a novel Dynamic Mixture of Curriculum LoRA Experts (D-MoLE) method, which automatically evolves MLLM's architecture with controlled parameter budgets to continually adapt to new tasks while retaining previously learned knowledge. Specifically, we propose a dynamic layer-wise expert allocator, which automatically allocates LoRA experts across layers to resolve architecture conflicts, and routes instructions layer-wisely to facilitate knowledge sharing among experts. Then, we propose a gradient-based inter-modal continual curriculum, which adjusts the update ratio of each module in MLLM based on the difficulty of each modality within the task to alleviate the modality imbalance problem. Extensive experiments show that D-MoLE significantly outperforms state-of-the-art baselines, achieving a 15% average improvement over the best baseline. To the best of our knowledge, this is the first study of continual learning for MLLMs from an architectural perspective.
>
---
#### [new 025] Self-Calibrating BCIs: Ranking and Recovery of Mental Targets Without Labels
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于脑机接口任务，解决无标签情况下恢复用户心中目标图像的问题。提出CURSOR框架，无需标签即可预测相似度、排序和生成目标图像。**

- **链接: [http://arxiv.org/pdf/2506.11151v1](http://arxiv.org/pdf/2506.11151v1)**

> **作者:** Jonathan Grizou; Carlos de la Torre-Ortiz; Tuukka Ruotsalo
>
> **备注:** 10 pages, 4 figures, 11 appendix pages, 7 appendix figures
>
> **摘要:** We consider the problem of recovering a mental target (e.g., an image of a face) that a participant has in mind from paired EEG (i.e., brain responses) and image (i.e., perceived faces) data collected during interactive sessions without access to labeled information. The problem has been previously explored with labeled data but not via self-calibration, where labeled data is unavailable. Here, we present the first framework and an algorithm, CURSOR, that learns to recover unknown mental targets without access to labeled data or pre-trained decoders. Our experiments on naturalistic images of faces demonstrate that CURSOR can (1) predict image similarity scores that correlate with human perceptual judgments without any label information, (2) use these scores to rank stimuli against an unknown mental target, and (3) generate new stimuli indistinguishable from the unknown mental target (validated via a user study, N=53).
>
---
#### [new 026] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于医学视频表示学习任务，旨在解决超声心动图中因结构细微、动态复杂导致的预训练模型不足问题。提出DISCOVR框架，结合聚类与在线图像编码，提升视频表示质量。**

- **链接: [http://arxiv.org/pdf/2506.11777v1](http://arxiv.org/pdf/2506.11777v1)**

> **作者:** Divyanshu Mishra; Mohammadreza Salehi; Pramit Saha; Olga Patey; Aris T. Papageorghiou; Yuki M. Asano; J. Alison Noble
>
> **摘要:** Self-supervised learning (SSL) has achieved major advances in natural images and video understanding, but challenges remain in domains like echocardiography (heart ultrasound) due to subtle anatomical structures, complex temporal dynamics, and the current lack of domain-specific pre-trained models. Existing SSL approaches such as contrastive, masked modeling, and clustering-based methods struggle with high intersample similarity, sensitivity to low PSNR inputs common in ultrasound, or aggressive augmentations that distort clinically relevant features. We present DISCOVR (Distilled Image Supervision for Cross Modal Video Representation), a self-supervised dual branch framework for cardiac ultrasound video representation learning. DISCOVR combines a clustering-based video encoder that models temporal dynamics with an online image encoder that extracts fine-grained spatial semantics. These branches are connected through a semantic cluster distillation loss that transfers anatomical knowledge from the evolving image encoder to the video encoder, enabling temporally coherent representations enriched with fine-grained semantic understanding. Evaluated on six echocardiography datasets spanning fetal, pediatric, and adult populations, DISCOVR outperforms both specialized video anomaly detection methods and state-of-the-art video-SSL baselines in zero-shot and linear probing setups, and achieves superior segmentation transfer.
>
---
#### [new 027] EfficientQuant: An Efficient Post-Training Quantization for CNN-Transformer Hybrid Models on Edge Devices
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，解决混合模型在边缘设备部署的资源消耗问题。通过提出EfficientQuant方法，实现高效量化，降低延迟并保持精度。**

- **链接: [http://arxiv.org/pdf/2506.11093v1](http://arxiv.org/pdf/2506.11093v1)**

> **作者:** Shaibal Saha; Lanyu Xu
>
> **备注:** Accepted to the 4th Workshop on Transformers for Vision (T4V) at CVPR 2025
>
> **摘要:** Hybrid models that combine convolutional and transformer blocks offer strong performance in computer vision (CV) tasks but are resource-intensive for edge deployment. Although post-training quantization (PTQ) can help reduce resource demand, its application to hybrid models remains limited. We propose EfficientQuant, a novel structure-aware PTQ approach that applies uniform quantization to convolutional blocks and $log_2$ quantization to transformer blocks. EfficientQuant achieves $2.5 \times - 8.7 \times$ latency reduction with minimal accuracy loss on the ImageNet-1K dataset. It further demonstrates low latency and memory efficiency on edge devices, making it practical for real-world deployment.
>
---
#### [new 028] Leveraging Satellite Image Time Series for Accurate Extreme Event Detection
- **分类: cs.CV**

- **简介: 该论文属于极端事件检测任务，旨在通过卫星图像时间序列准确识别灾害事件。工作包括提出SITS-Extreme框架，融合多时相数据提升检测精度。**

- **链接: [http://arxiv.org/pdf/2506.11544v1](http://arxiv.org/pdf/2506.11544v1)**

> **作者:** Heng Fang; Hossein Azizpour
>
> **备注:** Accepted to the WACV 2025 Workshop on GeoCV. Code, datasets, and model checkpoints available at: https://github.com/hfangcat/SITS-ExtremeEvents
>
> **摘要:** Climate change is leading to an increase in extreme weather events, causing significant environmental damage and loss of life. Early detection of such events is essential for improving disaster response. In this work, we propose SITS-Extreme, a novel framework that leverages satellite image time series to detect extreme events by incorporating multiple pre-disaster observations. This approach effectively filters out irrelevant changes while isolating disaster-relevant signals, enabling more accurate detection. Extensive experiments on both real-world and synthetic datasets validate the effectiveness of SITS-Extreme, demonstrating substantial improvements over widely used strong bi-temporal baselines. Additionally, we examine the impact of incorporating more timesteps, analyze the contribution of key components in our framework, and evaluate its performance across different disaster types, offering valuable insights into its scalability and applicability for large-scale disaster monitoring.
>
---
#### [new 029] EyeSim-VQA: A Free-Energy-Guided Eye Simulation Framework for Video Quality Assessment
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视频质量评估任务，旨在解决VQA中时空复杂性和模型稳定性问题。提出EyeSimVQA框架，结合自由能引导的自修复机制和双分支结构，提升评估性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.11549v1](http://arxiv.org/pdf/2506.11549v1)**

> **作者:** Zhaoyang Wang; Wen Lu; Jie Li; Lihuo He; Maoguo Gong; Xinbo Gao
>
> **备注:** This work has been submitted to the IEEE TCSVT for possible publication
>
> **摘要:** Free-energy-guided self-repair mechanisms have shown promising results in image quality assessment (IQA), but remain under-explored in video quality assessment (VQA), where temporal dynamics and model constraints pose unique challenges. Unlike static images, video content exhibits richer spatiotemporal complexity, making perceptual restoration more difficult. Moreover, VQA systems often rely on pre-trained backbones, which limits the direct integration of enhancement modules without affecting model stability. To address these issues, we propose EyeSimVQA, a novel VQA framework that incorporates free-energy-based self-repair. It adopts a dual-branch architecture, with an aesthetic branch for global perceptual evaluation and a technical branch for fine-grained structural and semantic analysis. Each branch integrates specialized enhancement modules tailored to distinct visual inputs-resized full-frame images and patch-based fragments-to simulate adaptive repair behaviors. We also explore a principled strategy for incorporating high-level visual features without disrupting the original backbone. In addition, we design a biologically inspired prediction head that models sweeping gaze dynamics to better fuse global and local representations for quality prediction. Experiments on five public VQA benchmarks demonstrate that EyeSimVQA achieves competitive or superior performance compared to state-of-the-art methods, while offering improved interpretability through its biologically grounded design.
>
---
#### [new 030] Evaluating Sensitivity Parameters in Smartphone-Based Gaze Estimation: A Comparative Study of Appearance-Based and Infrared Eye Trackers
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于眼动追踪任务，旨在评估基于智能手机的视觉追踪算法性能，对比分析其在不同条件下的准确性与敏感性。**

- **链接: [http://arxiv.org/pdf/2506.11932v1](http://arxiv.org/pdf/2506.11932v1)**

> **作者:** Nishan Gunawardena; Gough Yumu Lui; Jeewani Anupama Ginige; Bahman Javadi
>
> **摘要:** This study evaluates a smartphone-based, deep-learning eye-tracking algorithm by comparing its performance against a commercial infrared-based eye tracker, the Tobii Pro Nano. The aim is to investigate the feasibility of appearance-based gaze estimation under realistic mobile usage conditions. Key sensitivity factors, including age, gender, vision correction, lighting conditions, device type, and head position, were systematically analysed. The appearance-based algorithm integrates a lightweight convolutional neural network (MobileNet-V3) with a recurrent structure (Long Short-Term Memory) to predict gaze coordinates from grayscale facial images. Gaze data were collected from 51 participants using dynamic visual stimuli, and accuracy was measured using Euclidean distance. The deep learning model produced a mean error of 17.76 mm, compared to 16.53 mm for the Tobii Pro Nano. While overall accuracy differences were small, the deep learning-based method was more sensitive to factors such as lighting, vision correction, and age, with higher failure rates observed under low-light conditions among participants using glasses and in older age groups. Device-specific and positional factors also influenced tracking performance. These results highlight the potential of appearance-based approaches for mobile eye tracking and offer a reference framework for evaluating gaze estimation systems across varied usage conditions.
>
---
#### [new 031] Lifting Data-Tracing Machine Unlearning to Knowledge-Tracing for Foundation Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于机器遗忘任务，旨在解决Foundation Models中数据撤销问题。提出将数据追踪转向知识追踪，以更符合人类记忆机制并满足多样化的遗忘需求。**

- **链接: [http://arxiv.org/pdf/2506.11253v1](http://arxiv.org/pdf/2506.11253v1)**

> **作者:** Yuwen Tan; Boqing Gong
>
> **备注:** 21 pages, 3 figures
>
> **摘要:** Machine unlearning removes certain training data points and their influence on AI models (e.g., when a data owner revokes their decision to allow models to learn from the data). In this position paper, we propose to lift data-tracing machine unlearning to knowledge-tracing for foundation models (FMs). We support this position based on practical needs and insights from cognitive studies. Practically, tracing data cannot meet the diverse unlearning requests for FMs, which may be from regulators, enterprise users, product teams, etc., having no access to FMs' massive training data. Instead, it is convenient for these parties to issue an unlearning request about the knowledge or capability FMs (should not) possess. Cognitively, knowledge-tracing unlearning aligns with how the human brain forgets more closely than tracing individual training data points. Finally, we provide a concrete case study about a vision-language FM to illustrate how an unlearner might instantiate the knowledge-tracing machine unlearning paradigm.
>
---
#### [new 032] SLRNet: A Real-Time LSTM-Based Sign Language Recognition System
- **分类: cs.CV; 68T07 (Artificial Intelligence), 68U10 (Image Processing)**

- **简介: 该论文属于手语识别任务，旨在解决听障人士与社会沟通的问题。通过结合MediaPipe和LSTM网络，实现实时ASL识别。**

- **链接: [http://arxiv.org/pdf/2506.11154v1](http://arxiv.org/pdf/2506.11154v1)**

> **作者:** Sharvari Kamble
>
> **备注:** 9 pages, 5 figures, includes experimental results. Code available at: https://github.com/Khushi-739/SLRNet
>
> **摘要:** Sign Language Recognition (SLR) plays a crucial role in bridging the communication gap between the hearing-impaired community and society. This paper introduces SLRNet, a real-time webcam-based ASL recognition system using MediaPipe Holistic and Long Short-Term Memory (LSTM) networks. The model processes video streams to recognize both ASL alphabet letters and functional words. With a validation accuracy of 86.7%, SLRNet demonstrates the feasibility of inclusive, hardware-independent gesture recognition.
>
---
#### [new 033] Predicting Patient Survival with Airway Biomarkers using nn-Unet/Radiomics
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于肺纤维化患者生存预测任务，旨在通过空气道生物标志物评估生存率。采用分割网络和SVM分类器进行特征提取与分类。**

- **链接: [http://arxiv.org/pdf/2506.11677v1](http://arxiv.org/pdf/2506.11677v1)**

> **作者:** Zacharia Mesbah; Dhruv Jain; Tsiry Mayet; Romain Modzelewski; Romain Herault; Simon Bernard; Sebastien Thureau; Clement Chatelain
>
> **备注:** 8 pages
>
> **摘要:** The primary objective of the AIIB 2023 competition is to evaluate the predictive significance of airway-related imaging biomarkers in determining the survival outcomes of patients with lung fibrosis.This study introduces a comprehensive three-stage approach. Initially, a segmentation network, namely nn-Unet, is employed to delineate the airway's structural boundaries. Subsequently, key features are extracted from the radiomic images centered around the trachea and an enclosing bounding box around the airway. This step is motivated by the potential presence of critical survival-related insights within the tracheal region as well as pertinent information encoded in the structure and dimensions of the airway. Lastly, radiomic features obtained from the segmented areas are integrated into an SVM classifier. We could obtain an overall-score of 0.8601 for the segmentation in Task 1 while 0.7346 for the classification in Task 2.
>
---
#### [new 034] VGR: Visual Grounded Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决传统方法依赖语言空间、忽视视觉细节的问题。提出VGR模型，结合视觉感知与语言推理，提升图像理解能力。**

- **链接: [http://arxiv.org/pdf/2506.11991v1](http://arxiv.org/pdf/2506.11991v1)**

> **作者:** Jiacong Wang; Zijiang Kang; Haochen Wang; Haiyong Jiang; Jiawen Li; Bohong Wu; Ya Wang; Jiao Ran; Xiao Liang; Chao Feng; Jun Xiao
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA.
>
---
#### [new 035] Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation
- **分类: cs.CV**

- **简介: 该论文属于图像与几何生成任务，解决新视角合成问题。通过扩散模型和跨模态注意力蒸馏，实现图像与几何的对齐生成。**

- **链接: [http://arxiv.org/pdf/2506.11924v1](http://arxiv.org/pdf/2506.11924v1)**

> **作者:** Min-Seop Kwak; Junho Kim; Sangdoo Yun; Dongyoon Han; Taekyoung Kim; Seungryong Kim; Jin-Hwa Kim
>
> **摘要:** We introduce a diffusion-based framework that performs aligned novel view image and geometry generation via a warping-and-inpainting methodology. Unlike prior methods that require dense posed images or pose-embedded generative models limited to in-domain views, our method leverages off-the-shelf geometry predictors to predict partial geometries viewed from reference images, and formulates novel-view synthesis as an inpainting task for both image and geometry. To ensure accurate alignment between generated images and geometry, we propose cross-modal attention distillation, where attention maps from the image diffusion branch are injected into a parallel geometry diffusion branch during both training and inference. This multi-task approach achieves synergistic effects, facilitating geometrically robust image synthesis as well as well-defined geometry prediction. We further introduce proximity-based mesh conditioning to integrate depth and normal cues, interpolating between point cloud and filtering erroneously predicted geometry from influencing the generation process. Empirically, our method achieves high-fidelity extrapolative view synthesis on both image and geometry across a range of unseen scenes, delivers competitive reconstruction quality under interpolation settings, and produces geometrically aligned colored point clouds for comprehensive 3D completion. Project page is available at https://cvlab-kaist.github.io/MoAI.
>
---
#### [new 036] Evaluating Fairness and Mitigating Bias in Machine Learning: A Novel Technique using Tensor Data and Bayesian Regression
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于机器学习公平性评估任务，旨在解决皮肤颜色分类中的偏见问题。通过张量数据和贝叶斯回归方法，提升模型对肤色的公平处理能力。**

- **链接: [http://arxiv.org/pdf/2506.11627v1](http://arxiv.org/pdf/2506.11627v1)**

> **作者:** Kuniko Paxton; Koorosh Aslansefat; Dhavalkumar Thakker; Yiannis Papadopoulos
>
> **摘要:** Fairness is a critical component of Trustworthy AI. In this paper, we focus on Machine Learning (ML) and the performance of model predictions when dealing with skin color. Unlike other sensitive attributes, the nature of skin color differs significantly. In computer vision, skin color is represented as tensor data rather than categorical values or single numerical points. However, much of the research on fairness across sensitive groups has focused on categorical features such as gender and race. This paper introduces a new technique for evaluating fairness in ML for image classification tasks, specifically without the use of annotation. To address the limitations of prior work, we handle tensor data, like skin color, without classifying it rigidly. Instead, we convert it into probability distributions and apply statistical distance measures. This novel approach allows us to capture fine-grained nuances in fairness both within and across what would traditionally be considered distinct groups. Additionally, we propose an innovative training method to mitigate the latent biases present in conventional skin tone categorization. This method leverages color distance estimates calculated through Bayesian regression with polynomial functions, ensuring a more nuanced and equitable treatment of skin color in ML models.
>
---
#### [new 037] Test-Time-Scaling for Zero-Shot Diagnosis with Visual-Language Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗图像诊断任务，旨在解决LLM在无监督情况下进行视觉-语言推理的难题。通过测试时缩放策略提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2506.11166v1](http://arxiv.org/pdf/2506.11166v1)**

> **作者:** Ji Young Byun; Young-Jin Park; Navid Azizan; Rama Chellappa
>
> **摘要:** As a cornerstone of patient care, clinical decision-making significantly influences patient outcomes and can be enhanced by large language models (LLMs). Although LLMs have demonstrated remarkable performance, their application to visual question answering in medical imaging, particularly for reasoning-based diagnosis, remains largely unexplored. Furthermore, supervised fine-tuning for reasoning tasks is largely impractical due to limited data availability and high annotation costs. In this work, we introduce a zero-shot framework for reliable medical image diagnosis that enhances the reasoning capabilities of LLMs in clinical settings through test-time scaling. Given a medical image and a textual prompt, a vision-language model processes a medical image along with a corresponding textual prompt to generate multiple descriptions or interpretations of visual features. These interpretations are then fed to an LLM, where a test-time scaling strategy consolidates multiple candidate outputs into a reliable final diagnosis. We evaluate our approach across various medical imaging modalities -- including radiology, ophthalmology, and histopathology -- and demonstrate that the proposed test-time scaling strategy enhances diagnostic accuracy for both our and baseline methods. Additionally, we provide an empirical analysis showing that the proposed approach, which allows unbiased prompting in the first stage, improves the reliability of LLM-generated diagnoses and enhances classification accuracy.
>
---
#### [new 038] MambaVSR: Content-Aware Scanning State Space Model for Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，旨在解决非局部依赖建模与计算效率问题。提出MambaVSR框架，结合内容感知扫描机制，提升性能并减少参数。**

- **链接: [http://arxiv.org/pdf/2506.11768v1](http://arxiv.org/pdf/2506.11768v1)**

> **作者:** Linfeng He; Meiqin Liu; Qi Tang; Chao Yao; Yao Zhao
>
> **摘要:** Video super-resolution (VSR) faces critical challenges in effectively modeling non-local dependencies across misaligned frames while preserving computational efficiency. Existing VSR methods typically rely on optical flow strategies or transformer architectures, which struggle with large motion displacements and long video sequences. To address this, we propose MambaVSR, the first state-space model framework for VSR that incorporates an innovative content-aware scanning mechanism. Unlike rigid 1D sequential processing in conventional vision Mamba methods, our MambaVSR enables dynamic spatiotemporal interactions through the Shared Compass Construction (SCC) and the Content-Aware Sequentialization (CAS). Specifically, the SCC module constructs intra-frame semantic connectivity graphs via efficient sparse attention and generates adaptive spatial scanning sequences through spectral clustering. Building upon SCC, the CAS module effectively aligns and aggregates non-local similar content across multiple frames by interleaving temporal features along the learned spatial order. To bridge global dependencies with local details, the Global-Local State Space Block (GLSSB) synergistically integrates window self-attention operations with SSM-based feature propagation, enabling high-frequency detail recovery under global dependency guidance. Extensive experiments validate MambaVSR's superiority, outperforming the Transformer-based method by 0.58 dB PSNR on the REDS dataset with 55% fewer parameters.
>
---
#### [new 039] A$^2$LC: Active and Automated Label Correction for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语义分割任务，解决手动标注成本高、易出错的问题。提出A$^2$LC框架，结合主动学习与自动标签校正，提升效率与效果。**

- **链接: [http://arxiv.org/pdf/2506.11599v1](http://arxiv.org/pdf/2506.11599v1)**

> **作者:** Youjin Jeon; Kyusik Cho; Suhan Woo; Euntai Kim
>
> **备注:** Preprint. Under review. 22 pages, 8 figures
>
> **摘要:** Active Label Correction (ALC) has emerged as a promising solution to the high cost and error-prone nature of manual pixel-wise annotation in semantic segmentation, by selectively identifying and correcting mislabeled data. Although recent work has improved correction efficiency by generating pseudo-labels using foundation models, substantial inefficiencies still remain. In this paper, we propose Active and Automated Label Correction for semantic segmentation (A$^2$LC), a novel and efficient ALC framework that integrates an automated correction stage into the conventional pipeline. Specifically, the automated correction stage leverages annotator feedback to perform label correction beyond the queried samples, thereby maximizing cost efficiency. In addition, we further introduce an adaptively balanced acquisition function that emphasizes underrepresented tail classes and complements the automated correction mechanism. Extensive experiments on Cityscapes and PASCAL VOC 2012 demonstrate that A$^2$LC significantly outperforms previous state-of-the-art methods. Notably, A$^2$LC achieves high efficiency by outperforming previous methods using only 20% of their budget, and demonstrates strong effectiveness by yielding a 27.23% performance improvement under an equivalent budget constraint on the Cityscapes dataset. The code will be released upon acceptance.
>
---
#### [new 040] Auditing Data Provenance in Real-world Text-to-Image Diffusion Models for Privacy and Copyright Protection
- **分类: cs.CV**

- **简介: 该论文属于数据溯源任务，旨在解决文本生成图像模型中的版权和隐私问题。提出FSCA框架，在无需内部信息的情况下实现高效审计。**

- **链接: [http://arxiv.org/pdf/2506.11434v1](http://arxiv.org/pdf/2506.11434v1)**

> **作者:** Jie Zhu; Leye Wang
>
> **备注:** Under Review; A user-level accuracy of 90% in a real-world auditing scenario
>
> **摘要:** Text-to-image diffusion model since its propose has significantly influenced the content creation due to its impressive generation capability. However, this capability depends on large-scale text-image datasets gathered from web platforms like social media, posing substantial challenges in copyright compliance and personal privacy leakage. Though there are some efforts devoted to explore approaches for auditing data provenance in text-to-image diffusion models, existing work has unrealistic assumptions that can obtain model internal knowledge, e.g., intermediate results, or the evaluation is not reliable. To fill this gap, we propose a completely black-box auditing framework called Feature Semantic Consistency-based Auditing (FSCA). It utilizes two types of semantic connections within the text-to-image diffusion model for auditing, eliminating the need for access to internal knowledge. To demonstrate the effectiveness of our FSCA framework, we perform extensive experiments on LAION-mi dataset and COCO dataset, and compare with eight state-of-the-art baseline approaches. The results show that FSCA surpasses previous baseline approaches across various metrics and different data distributions, showcasing the superiority of our FSCA. Moreover, we introduce a recall balance strategy and a threshold adjustment strategy, which collectively allows FSCA to reach up a user-level accuracy of 90% in a real-world auditing scenario with only 10 samples/user, highlighting its strong auditing potential in real-world applications. Our code is made available at https://github.com/JiePKU/FSCA.
>
---
#### [new 041] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于无监督域适应任务，旨在解决目标域视觉嵌入分布偏离预训练模型的问题。通过利用视觉与文本嵌入的几何结构，增强伪标签并提升目标域提示学习效果。**

- **链接: [http://arxiv.org/pdf/2506.11493v1](http://arxiv.org/pdf/2506.11493v1)**

> **作者:** Tung-Long Vuong; Hoang Phan; Vy Vo; Anh Bui; Thanh-Toan Do; Trung Le; Dinh Phung
>
> **摘要:** Recent approaches leveraging multi-modal pre-trained models like CLIP for Unsupervised Domain Adaptation (UDA) have shown significant promise in bridging domain gaps and improving generalization by utilizing rich semantic knowledge and robust visual representations learned through extensive pre-training on diverse image-text datasets. While these methods achieve state-of-the-art performance across benchmarks, much of the improvement stems from base pseudo-labels (CLIP zero-shot predictions) and self-training mechanisms. Thus, the training mechanism exhibits a key limitation wherein the visual embedding distribution in target domains can deviate from the visual embedding distribution in the pre-trained model, leading to misguided signals from class descriptions. This work introduces a fresh solution to reinforce these pseudo-labels and facilitate target-prompt learning, by exploiting the geometry of visual and text embeddings - an aspect that is overlooked by existing methods. We first propose to directly leverage the reference predictions (from source prompts) based on the relationship between source and target visual embeddings. We later show that there is a strong clustering behavior observed between visual and text embeddings in pre-trained multi-modal models. Building on optimal transport theory, we transform this insight into a novel strategy to enforce the clustering property in text embeddings, further enhancing the alignment in the target domain. Our experiments and ablation studies validate the effectiveness of the proposed approach, demonstrating superior performance and improved quality of target prompts in terms of representation.
>
---
#### [new 042] Vision-based Lifting of 2D Object Detections for Automated Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D目标检测任务，旨在用摄像头替代LiDAR进行自动驾驶中的物体检测。通过将2D检测结果提升为3D，实现低成本高效检测。**

- **链接: [http://arxiv.org/pdf/2506.11839v1](http://arxiv.org/pdf/2506.11839v1)**

> **作者:** Hendrik Königshof; Kun Li; Christoph Stiller
>
> **备注:** https://ieeexplore.ieee.org/document/9190325
>
> **摘要:** Image-based 3D object detection is an inevitable part of autonomous driving because cheap onboard cameras are already available in most modern cars. Because of the accurate depth information, currently, most state-of-the-art 3D object detectors heavily rely on LiDAR data. In this paper, we propose a pipeline which lifts the results of existing vision-based 2D algorithms to 3D detections using only cameras as a cost-effective alternative to LiDAR. In contrast to existing approaches, we focus not only on cars but on all types of road users. To the best of our knowledge, we are the first using a 2D CNN to process the point cloud for each 2D detection to keep the computational effort as low as possible. Our evaluation on the challenging KITTI 3D object detection benchmark shows results comparable to state-of-the-art image-based approaches while having a runtime of only a third.
>
---
#### [new 043] Camera-based method for the detection of lifted truck axles using convolutional neural networks
- **分类: cs.CV**

- **简介: 该论文属于车辆识别任务，旨在解决 lifted truck axles 的检测问题。通过 YOLOv8s 模型实现图像中轴体抬起的检测，提升交通监管效率。**

- **链接: [http://arxiv.org/pdf/2506.11574v1](http://arxiv.org/pdf/2506.11574v1)**

> **作者:** Bachir Tchana Tankeu; Mohamed Bouteldja; Nicolas Grignard; Bernard Jacob
>
> **摘要:** The identification and classification of vehicles play a crucial role in various aspects of the control-sanction system. Current technologies such as weigh-in-motion (WIM) systems can classify most vehicle categories but they struggle to accurately classify vehicles with lifted axles. Moreover, very few commercial and technical methods exist for detecting lifted axles. In this paper, as part of the European project SETO (Smart Enforcement of Transport Operations), a method based on a convolutional neural network (CNN), namely YOLOv8s, was proposed for the detection of lifted truck axles in images of trucks captured by cameras placed perpendicular to the direction of traffic. The performance of the proposed method was assessed and it was found that it had a precision of 87%, a recall of 91.7%, and an inference time of 1.4 ms, which makes it well-suited for real time implantations. These results suggest that further improvements could be made, potentially by increasing the size of the datasets and/or by using various image augmentation methods.
>
---
#### [new 044] GynSurg: A Comprehensive Gynecology Laparoscopic Surgery Dataset
- **分类: cs.CV**

- **简介: 该论文提出GynSurg数据集，解决妇科腹腔镜手术分析中的数据不足问题，支持动作识别、分割等多任务应用。**

- **链接: [http://arxiv.org/pdf/2506.11356v1](http://arxiv.org/pdf/2506.11356v1)**

> **作者:** Sahar Nasirihaghighi; Negin Ghamsarian; Leonie Peschek; Matteo Munari; Heinrich Husslein; Raphael Sznitman; Klaus Schoeffmann
>
> **摘要:** Recent advances in deep learning have transformed computer-assisted intervention and surgical video analysis, driving improvements not only in surgical training, intraoperative decision support, and patient outcomes, but also in postoperative documentation and surgical discovery. Central to these developments is the availability of large, high-quality annotated datasets. In gynecologic laparoscopy, surgical scene understanding and action recognition are fundamental for building intelligent systems that assist surgeons during operations and provide deeper analysis after surgery. However, existing datasets are often limited by small scale, narrow task focus, or insufficiently detailed annotations, limiting their utility for comprehensive, end-to-end workflow analysis. To address these limitations, we introduce GynSurg, the largest and most diverse multi-task dataset for gynecologic laparoscopic surgery to date. GynSurg provides rich annotations across multiple tasks, supporting applications in action recognition, semantic segmentation, surgical documentation, and discovery of novel procedural insights. We demonstrate the dataset quality and versatility by benchmarking state-of-the-art models under a standardized training protocol. To accelerate progress in the field, we publicly release the GynSurg dataset and its annotations
>
---
#### [new 045] Uncertainty Awareness Enables Efficient Labeling for Cancer Subtyping in Digital Pathology
- **分类: cs.CV**

- **简介: 该论文属于癌症亚型分类任务，旨在解决标注数据不足的问题。通过引入不确定性意识，优化标注过程，提升分类效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.11439v1](http://arxiv.org/pdf/2506.11439v1)**

> **作者:** Nirhoshan Sivaroopan; Chamuditha Jayanga Galappaththige; Chalani Ekanayake; Hasindri Watawana; Ranga Rodrigo; Chamira U. S. Edussooriya; Dushan N. Wadduwage
>
> **摘要:** Machine-learning-assisted cancer subtyping is a promising avenue in digital pathology. Cancer subtyping models, however, require careful training using expert annotations so that they can be inferred with a degree of known certainty (or uncertainty). To this end, we introduce the concept of uncertainty awareness into a self-supervised contrastive learning model. This is achieved by computing an evidence vector at every epoch, which assesses the model's confidence in its predictions. The derived uncertainty score is then utilized as a metric to selectively label the most crucial images that require further annotation, thus iteratively refining the training process. With just 1-10% of strategically selected annotations, we attain state-of-the-art performance in cancer subtyping on benchmark datasets. Our method not only strategically guides the annotation process to minimize the need for extensive labeled datasets, but also improves the precision and efficiency of classifications. This development is particularly beneficial in settings where the availability of labeled data is limited, offering a promising direction for future research and application in digital pathology.
>
---
#### [new 046] SignAligner: Harmonizing Complementary Pose Modalities for Coherent Sign Language Generation
- **分类: cs.CV**

- **简介: 该论文属于手语生成任务，旨在解决生成真实自然手语视频的问题。提出SignAligner方法，通过多模态协同生成与修正，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2506.11621v1](http://arxiv.org/pdf/2506.11621v1)**

> **作者:** Xu Wang; Shengeng Tang; Lechao Cheng; Feng Li; Shuo Wang; Richang Hong
>
> **摘要:** Sign language generation aims to produce diverse sign representations based on spoken language. However, achieving realistic and naturalistic generation remains a significant challenge due to the complexity of sign language, which encompasses intricate hand gestures, facial expressions, and body movements. In this work, we introduce PHOENIX14T+, an extended version of the widely-used RWTH-PHOENIX-Weather 2014T dataset, featuring three new sign representations: Pose, Hamer and Smplerx. We also propose a novel method, SignAligner, for realistic sign language generation, consisting of three stages: text-driven pose modalities co-generation, online collaborative correction of multimodality, and realistic sign video synthesis. First, by incorporating text semantics, we design a joint sign language generator to simultaneously produce posture coordinates, gesture actions, and body movements. The text encoder, based on a Transformer architecture, extracts semantic features, while a cross-modal attention mechanism integrates these features to generate diverse sign language representations, ensuring accurate mapping and controlling the diversity of modal features. Next, online collaborative correction is introduced to refine the generated pose modalities using a dynamic loss weighting strategy and cross-modal attention, facilitating the complementarity of information across modalities, eliminating spatiotemporal conflicts, and ensuring semantic coherence and action consistency. Finally, the corrected pose modalities are fed into a pre-trained video generation network to produce high-fidelity sign language videos. Extensive experiments demonstrate that SignAligner significantly improves both the accuracy and expressiveness of the generated sign videos.
>
---
#### [new 047] Scalable Context-Preserving Model-Aware Deep Clustering for Hyperspectral Images
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像聚类任务，解决传统方法计算复杂、结构约束不足的问题，提出一种联合局部与非局部结构的高效聚类方法。**

- **链接: [http://arxiv.org/pdf/2506.11377v1](http://arxiv.org/pdf/2506.11377v1)**

> **作者:** Xianlu Li; Nicolas Nadisic; Shaoguang Huang; Nikos Deligiannis; Aleksandra Pižurica
>
> **摘要:** Subspace clustering has become widely adopted for the unsupervised analysis of hyperspectral images (HSIs). Recent model-aware deep subspace clustering methods often use a two-stage framework, involving the calculation of a self-representation matrix with complexity of O(n^2), followed by spectral clustering. However, these methods are computationally intensive, generally incorporating solely either local or non-local spatial structure constraints, and their structural constraints fall short of effectively supervising the entire clustering process. We propose a scalable, context-preserving deep clustering method based on basis representation, which jointly captures local and non-local structures for efficient HSI clustering. To preserve local structure (i.e., spatial continuity within subspaces), we introduce a spatial smoothness constraint that aligns clustering predictions with their spatially filtered versions. For non-local structure (i.e., spectral continuity), we employ a mini-cluster-based scheme that refines predictions at the group level, encouraging spectrally similar pixels to belong to the same subspace. Notably, these two constraints are jointly optimized to reinforce each other. Specifically, our model is designed as an one-stage approach in which the structural constraints are applied to the entire clustering process. The time and space complexity of our method is O(n), making it applicable to large-scale HSI data. Experiments on real-world datasets show that our method outperforms state-of-the-art techniques. Our code is available at: https://github.com/lxlscut/SCDSC
>
---
#### [new 048] Pose Matters: Evaluating Vision Transformers and CNNs for Human Action Recognition on Small COCO Subsets
- **分类: cs.CV; cs.AI; I.2.0**

- **简介: 该论文属于动作识别任务，研究在小规模COCO数据集上比较ViT与CNN的性能，发现ViT表现更优，并分析其可解释性。**

- **链接: [http://arxiv.org/pdf/2506.11678v1](http://arxiv.org/pdf/2506.11678v1)**

> **作者:** MingZe Tang; Madiha Kazi
>
> **备注:** 7 pages, 9 figures
>
> **摘要:** This study explores human action recognition using a three-class subset of the COCO image corpus, benchmarking models from simple fully connected networks to transformer architectures. The binary Vision Transformer (ViT) achieved 90% mean test accuracy, significantly exceeding multiclass classifiers such as convolutional networks (approximately 35%) and CLIP-based models (approximately 62-64%). A one-way ANOVA (F = 61.37, p < 0.001) confirmed these differences are statistically significant. Qualitative analysis with SHAP explainer and LeGrad heatmaps indicated that the ViT localizes pose-specific regions (e.g., lower limbs for walking or running), while simpler feed-forward models often focus on background textures, explaining their errors. These findings emphasize the data efficiency of transformer representations and the importance of explainability techniques in diagnosing class-specific failures.
>
---
#### [new 049] FAME: A Lightweight Spatio-Temporal Network for Model Attribution of Face-Swap Deepfakes
- **分类: cs.CV**

- **简介: 该论文属于深度伪造视频的模型归属任务，旨在识别生成Deepfake的模型。提出FAME框架，结合时空注意力机制，提升归属准确性与效率。**

- **链接: [http://arxiv.org/pdf/2506.11477v1](http://arxiv.org/pdf/2506.11477v1)**

> **作者:** Wasim Ahmad; Yan-Tsung Peng; Yuan-Hao Chang
>
> **摘要:** The widespread emergence of face-swap Deepfake videos poses growing risks to digital security, privacy, and media integrity, necessitating effective forensic tools for identifying the source of such manipulations. Although most prior research has focused primarily on binary Deepfake detection, the task of model attribution -- determining which generative model produced a given Deepfake -- remains underexplored. In this paper, we introduce FAME (Fake Attribution via Multilevel Embeddings), a lightweight and efficient spatio-temporal framework designed to capture subtle generative artifacts specific to different face-swap models. FAME integrates spatial and temporal attention mechanisms to improve attribution accuracy while remaining computationally efficient. We evaluate our model on three challenging and diverse datasets: Deepfake Detection and Manipulation (DFDM), FaceForensics++, and FakeAVCeleb. Results show that FAME consistently outperforms existing methods in both accuracy and runtime, highlighting its potential for deployment in real-world forensic and information security applications.
>
---
#### [new 050] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks
- **分类: cs.CV**

- **简介: 该论文提出3D-RAD数据集，解决3D医学视觉问答任务中的多时态分析与多样化诊断问题，推动多模态医疗AI发展。**

- **链接: [http://arxiv.org/pdf/2506.11147v1](http://arxiv.org/pdf/2506.11147v1)**

> **作者:** Xiaotang Gai; Jiaxiang Liu; Yichen Li; Zijie Meng; Jian Wu; Zuozhu Liu
>
> **摘要:** Medical Visual Question Answering (Med-VQA) holds significant potential for clinical decision support, yet existing efforts primarily focus on 2D imaging with limited task diversity. This paper presents 3D-RAD, a large-scale dataset designed to advance 3D Med-VQA using radiology CT scans. The 3D-RAD dataset encompasses six diverse VQA tasks: anomaly detection, image observation, medical computation, existence detection, static temporal diagnosis, and longitudinal temporal diagnosis. It supports both open- and closed-ended questions while introducing complex reasoning challenges, including computational tasks and multi-stage temporal analysis, to enable comprehensive benchmarking. Extensive evaluations demonstrate that existing vision-language models (VLMs), especially medical VLMs exhibit limited generalization, particularly in multi-temporal tasks, underscoring the challenges of real-world 3D diagnostic reasoning. To drive future advancements, we release a high-quality training set 3D-RAD-T of 136,195 expert-aligned samples, showing that fine-tuning on this dataset could significantly enhance model performance. Our dataset and code, aiming to catalyze multimodal medical AI research and establish a robust foundation for 3D medical visual understanding, are publicly available at https://github.com/Tang-xiaoxiao/M3D-RAD.
>
---
#### [new 051] Auto-Connect: Connectivity-Preserving RigFormer with Direct Preference Optimization
- **分类: cs.CV**

- **简介: 该论文属于自动绑定任务，旨在解决骨骼连通性与皮肤变形质量问题。通过连通性保留的分词、奖励优化和测地线感知的骨骼选择提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11430v1](http://arxiv.org/pdf/2506.11430v1)**

> **作者:** Jingfeng Guo; Jian Liu; Jinnan Chen; Shiwei Mao; Changrong Hu; Puhua Jiang; Junlin Yu; Jing Xu; Qi Liu; Lixin Xu; Zhuo Chen; Chunchao Guo
>
> **摘要:** We introduce Auto-Connect, a novel approach for automatic rigging that explicitly preserves skeletal connectivity through a connectivity-preserving tokenization scheme. Unlike previous methods that predict bone positions represented as two joints or first predict points before determining connectivity, our method employs special tokens to define endpoints for each joint's children and for each hierarchical layer, effectively automating connectivity relationships. This approach significantly enhances topological accuracy by integrating connectivity information directly into the prediction framework. To further guarantee high-quality topology, we implement a topology-aware reward function that quantifies topological correctness, which is then utilized in a post-training phase through reward-guided Direct Preference Optimization. Additionally, we incorporate implicit geodesic features for latent top-k bone selection, which substantially improves skinning quality. By leveraging geodesic distance information within the model's latent space, our approach intelligently determines the most influential bones for each vertex, effectively mitigating common skinning artifacts. This combination of connectivity-preserving tokenization, reward-guided fine-tuning, and geodesic-aware bone selection enables our model to consistently generate more anatomically plausible skeletal structures with superior deformation properties.
>
---
#### [new 052] Stop learning it all to mitigate visual hallucination, Focus on the hallucination target
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言任务，旨在解决MLLMs中的幻觉问题。通过聚焦幻觉目标区域，提出一种偏好学习方法，有效减少幻觉并提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2506.11417v1](http://arxiv.org/pdf/2506.11417v1)**

> **作者:** Dokyoon Yoon; Youngsook Song; Woomyong Park
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) frequently suffer from hallucination issues, generating information about objects that are not present in input images during vision-language tasks. These hallucinations particularly undermine model reliability in practical applications requiring accurate object identification. To address this challenge, we propose \mymethod,\ a preference learning approach that mitigates hallucinations by focusing on targeted areas where they occur. To implement this, we build a dataset containing hallucinated responses, correct responses, and target information (i.e., objects present in the images and the corresponding chunk positions in responses affected by hallucinations). By applying a preference learning method restricted to these specific targets, the model can filter out irrelevant signals and focus on correcting hallucinations. This allows the model to produce more factual responses by concentrating solely on relevant information. Experimental results demonstrate that \mymethod\ effectively reduces hallucinations across multiple vision hallucination tasks, improving the reliability and performance of MLLMs without diminishing overall performance.
>
---
#### [new 053] WaveFormer: A Lightweight Transformer Model for sEMG-based Gesture Recognition
- **分类: cs.CV**

- **简介: 该论文属于sEMG手势识别任务，旨在解决相似手势分类准确率低和模型计算量大的问题。提出轻量级WaveFormer模型，融合时频特征，实现高效高精度识别。**

- **链接: [http://arxiv.org/pdf/2506.11168v1](http://arxiv.org/pdf/2506.11168v1)**

> **作者:** Yanlong Chen; Mattia Orlandi; Pierangelo Maria Rapa; Simone Benatti; Luca Benini; Yawei Li
>
> **备注:** 6 pages, 3 figures, submitted to IEEE EMBS Conference on Neural Engineering (NER)
>
> **摘要:** Human-machine interaction, particularly in prosthetic and robotic control, has seen progress with gesture recognition via surface electromyographic (sEMG) signals.However, classifying similar gestures that produce nearly identical muscle signals remains a challenge, often reducing classification accuracy. Traditional deep learning models for sEMG gesture recognition are large and computationally expensive, limiting their deployment on resource-constrained embedded systems. In this work, we propose WaveFormer, a lightweight transformer-based architecture tailored for sEMG gesture recognition. Our model integrates time-domain and frequency-domain features through a novel learnable wavelet transform, enhancing feature extraction. In particular, the WaveletConv module, a multi-level wavelet decomposition layer with depthwise separable convolution, ensures both efficiency and compactness. With just 3.1 million parameters, WaveFormer achieves 95% classification accuracy on the EPN612 dataset, outperforming larger models. Furthermore, when profiled on a laptop equipped with an Intel CPU, INT8 quantization achieves real-time deployment with a 6.75 ms inference latency.
>
---
#### [new 054] On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶感知任务，旨在提升视觉语言模型对视觉攻击的鲁棒性。通过引入V2LMs，无需对抗训练即可有效防御未知攻击，增强系统安全性。**

- **链接: [http://arxiv.org/pdf/2506.11472v1](http://arxiv.org/pdf/2506.11472v1)**

> **作者:** Pedram MohajerAnsari; Amir Salarpour; Michael Kühr; Siyu Huang; Mohammad Hamad; Sebastian Steinhorst; Habeeb Olufowobi; Mert D. Pesé
>
> **摘要:** Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.
>
---
#### [new 055] Synthetic Geology -- Structural Geology Meets Deep Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于地质建模任务，旨在解决地下结构可视化难题。通过深度学习与合成数据生成，从地表数据预测三维地下结构。**

- **链接: [http://arxiv.org/pdf/2506.11164v1](http://arxiv.org/pdf/2506.11164v1)**

> **作者:** Simon Ghyselincks; Valeriia Okhmak; Stefano Zampini; George Turkiyyah; David Keyes; Eldad Haber
>
> **备注:** 10 pages, 8 figures, submitted to "Communications Earth & Environment", geological simulation code at https://doi.org/10.5281/zenodo.15244035, generative AI code at https://github.com/chipnbits/flowtrain_stochastic_interpolation/releases/tag/v1.0.0
>
> **摘要:** Visualizing the first few kilometers of the Earth's subsurface, a long-standing challenge gating a virtually inexhaustible list of important applications, is coming within reach through deep learning. Building on techniques of generative artificial intelligence applied to voxelated images, we demonstrate a method that extends surface geological data supplemented by boreholes to a three-dimensional subsurface region by training a neural network. The Earth's land area having been extensively mapped for geological features, the bottleneck of this or any related technique is the availability of data below the surface. We close this data gap in the development of subsurface deep learning by designing a synthetic data-generator process that mimics eons of geological activity such as sediment compaction, volcanic intrusion, and tectonic dynamics to produce a virtually limitless number of samples of the near lithosphere. A foundation model trained on such synthetic data is able to generate a 3D image of the subsurface from a previously unseen map of surface topography and geology, showing increasing fidelity with increasing access to borehole data, depicting such structures as layers, faults, folds, dikes, and sills. We illustrate the early promise of the combination of a synthetic lithospheric generator with a trained neural network model using generative flow matching. Ultimately, such models will be fine-tuned on data from applicable campaigns, such as mineral prospecting in a given region. Though useful in itself, a regionally fine-tuned models may be employed not as an end but as a means: as an AI-based regularizer in a more traditional inverse problem application, in which the objective function represents the mismatch of additional data with physical models with applications in resource exploration, hazard assessment, and geotechnical engineering.
>
---
#### [new 056] DiffFuSR: Super-Resolution of all Sentinel-2 Multispectral Bands using Diffusion Models
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于遥感图像超分辨率任务，旨在提升Sentinel-2多光谱影像的空间分辨率。通过扩散模型和融合网络，实现所有波段的高质量上采样。**

- **链接: [http://arxiv.org/pdf/2506.11764v1](http://arxiv.org/pdf/2506.11764v1)**

> **作者:** Muhammad Sarmad; Arnt-Børre Salberg; Michael Kampffmeyer
>
> **备注:** preprint under review
>
> **摘要:** This paper presents DiffFuSR, a modular pipeline for super-resolving all 12 spectral bands of Sentinel-2 Level-2A imagery to a unified ground sampling distance (GSD) of 2.5 meters. The pipeline comprises two stages: (i) a diffusion-based super-resolution (SR) model trained on high-resolution RGB imagery from the NAIP and WorldStrat datasets, harmonized to simulate Sentinel-2 characteristics; and (ii) a learned fusion network that upscales the remaining multispectral bands using the super-resolved RGB image as a spatial prior. We introduce a robust degradation model and contrastive degradation encoder to support blind SR. Extensive evaluations of the proposed SR pipeline on the OpenSR benchmark demonstrate that the proposed method outperforms current SOTA baselines in terms of reflectance fidelity, spectral consistency, spatial alignment, and hallucination suppression. Furthermore, the fusion network significantly outperforms classical pansharpening approaches, enabling accurate enhancement of Sentinel-2's 20 m and 60 m bands. This study underscores the power of harmonized learning with generative priors and fusion strategies to create a modular framework for Sentinel-2 SR. Our code and models can be found at https://github.com/NorskRegnesentral/DiffFuSR.
>
---
#### [new 057] On the development of an AI performance and behavioural measures for teaching and classroom management
- **分类: cs.CV; H.5; J.4; I.2.7; I.2.10**

- **简介: 该论文属于教育技术领域，旨在通过AI分析课堂行为，辅助教师发展。工作包括构建数据集、开发行为指标及教学评估仪表盘，以客观分析课堂互动。**

- **链接: [http://arxiv.org/pdf/2506.11143v1](http://arxiv.org/pdf/2506.11143v1)**

> **作者:** Andreea I. Niculescu; Jochen Ehnen; Chen Yi; Du Jiawei; Tay Chiat Pin; Joey Tianyi Zhou; Vigneshwaran Subbaraju; Teh Kah Kuan; Tran Huy Dat; John Komar; Gi Soong Chee; Kenneth Kwok
>
> **备注:** 7 pages, 10 figures, A video demonstration of the teacher trainer dashboard can be accessed here: https://vimeo.com/1076482827
>
> **摘要:** This paper presents a two-year research project focused on developing AI-driven measures to analyze classroom dynamics, with particular emphasis on teacher actions captured through multimodal sensor data. We applied real-time data from classroom sensors and AI techniques to extract meaningful insights and support teacher development. Key outcomes include a curated audio-visual dataset, novel behavioral measures, and a proof-of-concept teaching review dashboard. An initial evaluation with eight researchers from the National Institute for Education (NIE) highlighted the system's clarity, usability, and its non-judgmental, automated analysis approach -- which reduces manual workloads and encourages constructive reflection. Although the current version does not assign performance ratings, it provides an objective snapshot of in-class interactions, helping teachers recognize and improve their instructional strategies. Designed and tested in an Asian educational context, this work also contributes a culturally grounded methodology to the growing field of AI-based educational analytics.
>
---
#### [new 058] Composite Data Augmentations for Synthetic Image Detection Against Real-World Perturbations
- **分类: cs.CV; cs.AI; I.2.m; I.4.0**

- **简介: 该论文属于合成图像检测任务，旨在解决真实世界干扰下合成图像识别困难的问题。通过数据增强组合与优化方法提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.11490v1](http://arxiv.org/pdf/2506.11490v1)**

> **作者:** Efthymia Amarantidou; Christos Koutlis; Symeon Papadopoulos; Panagiotis C. Petrantonakis
>
> **备注:** EUSIPCO 2025 (33rd European Signal Processing Conference)
>
> **摘要:** The advent of accessible Generative AI tools enables anyone to create and spread synthetic images on social media, often with the intention to mislead, thus posing a significant threat to online information integrity. Most existing Synthetic Image Detection (SID) solutions struggle on generated images sourced from the Internet, as these are often altered by compression and other operations. To address this, our research enhances SID by exploring data augmentation combinations, leveraging a genetic algorithm for optimal augmentation selection, and introducing a dual-criteria optimization approach. These methods significantly improve model performance under real-world perturbations. Our findings provide valuable insights for developing detection models capable of identifying synthetic images across varying qualities and transformations, with the best-performing model achieving a mean average precision increase of +22.53% compared to models without augmentations. The implementation is available at github.com/efthimia145/sid-composite-data-augmentation.
>
---
#### [new 059] BrainMAP: Multimodal Graph Learning For Efficient Brain Disease Localization
- **分类: cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于脑疾病定位任务，旨在解决现有方法无法高效定位病灶区域的问题。提出BrainMAP框架，通过图学习和多模态融合提升效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.11178v1](http://arxiv.org/pdf/2506.11178v1)**

> **作者:** Nguyen Linh Dan Le; Jing Ren; Ciyuan Peng; Chengyao Xie; Bowen Li; Feng Xia
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Recent years have seen a surge in research focused on leveraging graph learning techniques to detect neurodegenerative diseases. However, existing graph-based approaches typically lack the ability to localize and extract the specific brain regions driving neurodegenerative pathology within the full connectome. Additionally, recent works on multimodal brain graph models often suffer from high computational complexity, limiting their practical use in resource-constrained devices. In this study, we present BrainMAP, a novel multimodal graph learning framework designed for precise and computationally efficient identification of brain regions affected by neurodegenerative diseases. First, BrainMAP utilizes an atlas-driven filtering approach guided by the AAL atlas to pinpoint and extract critical brain subgraphs. Unlike recent state-of-the-art methods, which model the entire brain network, BrainMAP achieves more than 50% reduction in computational overhead by concentrating on disease-relevant subgraphs. Second, we employ an advanced multimodal fusion process comprising cross-node attention to align functional magnetic resonance imaging (fMRI) and diffusion tensor imaging (DTI) data, coupled with an adaptive gating mechanism to blend and integrate these modalities dynamically. Experimental results demonstrate that BrainMAP outperforms state-of-the-art methods in computational efficiency, without compromising predictive accuracy.
>
---
#### [new 060] Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出STT模型，用于高效图像分割任务。通过聚焦点提示区域的变分辨率分块策略，减少计算成本，提升效率。**

- **链接: [http://arxiv.org/pdf/2506.11131v1](http://arxiv.org/pdf/2506.11131v1)**

> **作者:** Tanner Schmidt; Richard Newcombe
>
> **摘要:** This paper presents Segment This Thing (STT), a new efficient image segmentation model designed to produce a single segment given a single point prompt. Instead of following prior work and increasing efficiency by decreasing model size, we gain efficiency by foveating input images. Given an image and a point prompt, we extract a crop centered on the prompt and apply a novel variable-resolution patch tokenization in which patches are downsampled at a rate that increases with increased distance from the prompt. This approach yields far fewer image tokens than uniform patch tokenization. As a result we can drastically reduce the computational cost of segmentation without reducing model size. Furthermore, the foveation focuses the model on the region of interest, a potentially useful inductive bias. We show that our Segment This Thing model is more efficient than prior work while remaining competitive on segmentation benchmarks. It can easily run at interactive frame rates on consumer hardware and is thus a promising tool for augmented reality or robotics applications.
>
---
#### [new 061] Affogato: Learning Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言理解任务，旨在解决开放词汇的可操作性定位问题。通过构建大规模数据集并训练模型，提升智能体对环境的理解与交互能力。**

- **链接: [http://arxiv.org/pdf/2506.12009v1](http://arxiv.org/pdf/2506.12009v1)**

> **作者:** Junha Lee; Eunha Park; Chunghyun Park; Dahyun Kang; Minsu Cho
>
> **摘要:** Affordance grounding-localizing object regions based on natural language descriptions of interactions-is a critical challenge for enabling intelligent agents to understand and interact with their environments. However, this task remains challenging due to the need for fine-grained part-level localization, the ambiguity arising from multiple valid interaction regions, and the scarcity of large-scale datasets. In this work, we introduce Affogato, a large-scale benchmark comprising 150K instances, annotated with open-vocabulary text descriptions and corresponding 3D affordance heatmaps across a diverse set of objects and interactions. Building on this benchmark, we develop simple yet effective vision-language models that leverage pretrained part-aware vision backbones and a text-conditional heatmap decoder. Our models trained with the Affogato dataset achieve promising performance on the existing 2D and 3D benchmarks, and notably, exhibit effectiveness in open-vocabulary cross-domain generalization. The Affogato dataset is shared in public: https://huggingface.co/datasets/project-affogato/affogato
>
---
#### [new 062] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于深度学习中的偏差缓解任务，旨在解决模型依赖无关信号的问题。通过引入SAM和DISCO方法，提升模型的因果推理能力。**

- **链接: [http://arxiv.org/pdf/2506.11653v1](http://arxiv.org/pdf/2506.11653v1)**

> **作者:** Emre Kavak; Tom Nuno Wolf; Christian Wachinger
>
> **摘要:** During prediction tasks, models can use any signal they receive to come up with the final answer - including signals that are causally irrelevant. When predicting objects from images, for example, the lighting conditions could be correlated to different targets through selection bias, and an oblivious model might use these signals as shortcuts to discern between various objects. A predictor that uses lighting conditions instead of real object-specific details is obviously undesirable. To address this challenge, we introduce a standard anti-causal prediction model (SAM) that creates a causal framework for analyzing the information pathways influencing our predictor in anti-causal settings. We demonstrate that a classifier satisfying a specific conditional independence criterion will focus solely on the direct causal path from label to image, being counterfactually invariant to the remaining variables. Finally, we propose DISCO, a novel regularization strategy that uses conditional distance correlation to optimize for conditional independence in regression tasks. We can show that DISCO achieves competitive results in different bias mitigation experiments, deeming it a valid alternative to classical kernel-based methods.
>
---
#### [new 063] Adaptive Object Detection with ESRGAN-Enhanced Resolution & Faster R-CNN
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决低分辨率图像检测效果差的问题。通过结合ESRGAN和Faster R-CNN，先增强图像质量再进行检测，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.11122v1](http://arxiv.org/pdf/2506.11122v1)**

> **作者:** Divya Swetha K; Ziaul Haque Choudhury; Hemanta Kumar Bhuyan; Biswajit Brahma; Nilayam Kumar Kamila
>
> **摘要:** In this study, proposes a method for improved object detection from the low-resolution images by integrating Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) and Faster Region-Convolutional Neural Network (Faster R-CNN). ESRGAN enhances low-quality images, restoring details and improving clarity, while Faster R-CNN performs accurate object detection on the enhanced images. The combination of these techniques ensures better detection performance, even with poor-quality inputs, offering an effective solution for applications where image resolution is in consistent. ESRGAN is employed as a pre-processing step to enhance the low-resolution input image, effectively restoring lost details and improving overall image quality. Subsequently, the enhanced image is fed into the Faster R-CNN model for accurate object detection and localization. Experimental results demonstrate that this integrated approach yields superior performance compared to traditional methods applied directly to low-resolution images. The proposed framework provides a promising solution for applications where image quality is variable or limited, enabling more robust and reliable object detection in challenging scenarios. It achieves a balance between improved image quality and efficient object detection
>
---
#### [new 064] TARDIS STRIDE: A Spatio-Temporal Road Image Dataset for Exploration and Autonomy
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出STRIDE数据集，用于研究时空环境下的自主导航问题。通过构建包含多视角图像的结构化数据，支持对空间和时间动态的建模与学习。**

- **链接: [http://arxiv.org/pdf/2506.11302v1](http://arxiv.org/pdf/2506.11302v1)**

> **作者:** Héctor Carrión; Yutong Bai; Víctor A. Hernández Castro; Kishan Panaganti; Ayush Zenith; Matthew Trang; Tony Zhang; Pietro Perona; Jitendra Malik
>
> **备注:** Computer Vision, Pattern Recognition, LLMs, Dataset, Data Augmentation
>
> **摘要:** World models aim to simulate environments and enable effective agent behavior. However, modeling real-world environments presents unique challenges as they dynamically change across both space and, crucially, time. To capture these composed dynamics, we introduce a Spatio-Temporal Road Image Dataset for Exploration (STRIDE) permuting 360-degree panoramic imagery into rich interconnected observation, state and action nodes. Leveraging this structure, we can simultaneously model the relationship between egocentric views, positional coordinates, and movement commands across both space and time. We benchmark this dataset via TARDIS, a transformer-based generative world model that integrates spatial and temporal dynamics through a unified autoregressive framework trained on STRIDE. We demonstrate robust performance across a range of agentic tasks such as controllable photorealistic image synthesis, instruction following, autonomous self-control, and state-of-the-art georeferencing. These results suggest a promising direction towards sophisticated generalist agents--capable of understanding and manipulating the spatial and temporal aspects of their material environments--with enhanced embodied reasoning capabilities. Training code, datasets, and model checkpoints are made available at https://huggingface.co/datasets/Tera-AI/STRIDE.
>
---
#### [new 065] Improving Surgical Risk Prediction Through Integrating Automated Body Composition Analysis: a Retrospective Trial on Colectomy Surgery
- **分类: cs.CV**

- **简介: 该论文属于医疗风险预测任务，旨在通过CT影像自动分析提升结肠切除术后的风险预测效果，结合临床变量进行模型构建与评估。**

- **链接: [http://arxiv.org/pdf/2506.11996v1](http://arxiv.org/pdf/2506.11996v1)**

> **作者:** Hanxue Gu; Yaqian Chen; isoo Lee; Diego Schaps; Regina Woody; Roy Colglazier; Maciej A. Mazurowski; Christopher Mantyh
>
> **备注:** 32 pages, 5 figures
>
> **摘要:** Objective: To evaluate whether preoperative body composition metrics automatically extracted from CT scans can predict postoperative outcomes after colectomy, either alone or combined with clinical variables or existing risk predictors. Main outcomes and measures: The primary outcome was the predictive performance for 1-year all-cause mortality following colectomy. A Cox proportional hazards model with 1-year follow-up was used, and performance was evaluated using the concordance index (C-index) and Integrated Brier Score (IBS). Secondary outcomes included postoperative complications, unplanned readmission, blood transfusion, and severe infection, assessed using AUC and Brier Score from logistic regression. Odds ratios (OR) described associations between individual CT-derived body composition metrics and outcomes. Over 300 features were extracted from preoperative CTs across multiple vertebral levels, including skeletal muscle area, density, fat areas, and inter-tissue metrics. NSQIP scores were available for all surgeries after 2012.
>
---
#### [new 066] FARCLUSS: Fuzzy Adaptive Rebalancing and Contrastive Uncertainty Learning for Semi-Supervised Semantic Segmentation
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于半监督语义分割任务，旨在解决伪标签利用不足、类别不平衡和预测不确定性问题。提出FARCLUSS框架，通过模糊伪标签、动态加权、自适应重平衡和对比正则化提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.11142v1](http://arxiv.org/pdf/2506.11142v1)**

> **作者:** Ebenezer Tarubinga; Jenifer Kalafatovich
>
> **备注:** Submitted to Pattern Recognition
>
> **摘要:** Semi-supervised semantic segmentation (SSSS) faces persistent challenges in effectively leveraging unlabeled data, such as ineffective utilization of pseudo-labels, exacerbation of class imbalance biases, and neglect of prediction uncertainty. Current approaches often discard uncertain regions through strict thresholding favouring dominant classes. To address these limitations, we introduce a holistic framework that transforms uncertainty into a learning asset through four principal components: (1) fuzzy pseudo-labeling, which preserves soft class distributions from top-K predictions to enrich supervision; (2) uncertainty-aware dynamic weighting, that modulate pixel-wise contributions via entropy-based reliability scores; (3) adaptive class rebalancing, which dynamically adjust losses to counteract long-tailed class distributions; and (4) lightweight contrastive regularization, that encourage compact and discriminative feature embeddings. Extensive experiments on benchmarks demonstrate that our method outperforms current state-of-the-art approaches, achieving significant improvements in the segmentation of under-represented classes and ambiguous regions.
>
---
#### [new 067] How Visual Representations Map to Language Feature Space in Multimodal LLMs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态学习任务，旨在解决视觉与语言表示对齐问题。通过冻结模型并训练线性适配器，分析视觉特征如何映射到语言空间，揭示对齐过程及层间差异。**

- **链接: [http://arxiv.org/pdf/2506.11976v1](http://arxiv.org/pdf/2506.11976v1)**

> **作者:** Constantin Venhoff; Ashkan Khakzar; Sonia Joseph; Philip Torr; Neel Nanda
>
> **摘要:** Effective multimodal reasoning depends on the alignment of visual and linguistic representations, yet the mechanisms by which vision-language models (VLMs) achieve this alignment remain poorly understood. We introduce a methodological framework that deliberately maintains a frozen large language model (LLM) and a frozen vision transformer (ViT), connected solely by training a linear adapter during visual instruction tuning. This design is fundamental to our approach: by keeping the language model frozen, we ensure it maintains its original language representations without adaptation to visual data. Consequently, the linear adapter must map visual features directly into the LLM's existing representational space rather than allowing the language model to develop specialized visual understanding through fine-tuning. Our experimental design uniquely enables the use of pre-trained sparse autoencoders (SAEs) of the LLM as analytical probes. These SAEs remain perfectly aligned with the unchanged language model and serve as a snapshot of the learned language feature-representations. Through systematic analysis of SAE reconstruction error, sparsity patterns, and feature SAE descriptions, we reveal the layer-wise progression through which visual representations gradually align with language feature representations, converging in middle-to-later layers. This suggests a fundamental misalignment between ViT outputs and early LLM layers, raising important questions about whether current adapter-based architectures optimally facilitate cross-modal representation learning.
>
---
#### [new 068] EasyARC: Evaluating Vision Language Models on True Visual Reasoning
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出EasyARC，一个用于评估视觉语言模型真实推理能力的基准。解决现有基准缺乏复杂视觉与语言交互的问题，通过多图像、多步骤推理任务进行测试。**

- **链接: [http://arxiv.org/pdf/2506.11595v1](http://arxiv.org/pdf/2506.11595v1)**

> **作者:** Mert Unsal; Aylin Akkus
>
> **备注:** CVPR2025 Workshop on Test-time Scaling for Computer Vision
>
> **摘要:** Building on recent advances in language-based reasoning models, we explore multimodal reasoning that integrates vision and text. Existing multimodal benchmarks primarily test visual extraction combined with text-based reasoning, lacking true visual reasoning with more complex interactions between vision and language. Inspired by the ARC challenge, we introduce EasyARC, a vision-language benchmark requiring multi-image, multi-step reasoning, and self-correction. EasyARC is procedurally generated, fully verifiable, and scalable, making it ideal for reinforcement learning (RL) pipelines. The generators incorporate progressive difficulty levels, enabling structured evaluation across task types and complexities. We benchmark state-of-the-art vision-language models and analyze their failure modes. We argue that EasyARC sets a new standard for evaluating true reasoning and test-time scaling capabilities in vision-language models. We open-source our benchmark dataset and evaluation code.
>
---
#### [new 069] DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频语言理解任务，旨在解决视频LLM在时间推理上的不足。提出DaMO模型，通过多模态融合和分阶段训练提升时间对齐与推理能力。**

- **链接: [http://arxiv.org/pdf/2506.11558v1](http://arxiv.org/pdf/2506.11558v1)**

> **作者:** Bo-Cheng Chiu; Jen-Jee Chen; Yu-Chee Tseng; Feng-Chi Chen
>
> **摘要:** Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
>
---
#### [new 070] A Watermark for Auto-Regressive Image Generation Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成模型的水印任务，旨在解决retokenization mismatch问题，提出C-reweight方法实现无损水印。**

- **链接: [http://arxiv.org/pdf/2506.11371v1](http://arxiv.org/pdf/2506.11371v1)**

> **作者:** Yihan Wu; Xuehao Cui; Ruibo Chen; Georgios Milis; Heng Huang
>
> **备注:** Technical report
>
> **摘要:** The rapid evolution of image generation models has revolutionized visual content creation, enabling the synthesis of highly realistic and contextually accurate images for diverse applications. However, the potential for misuse, such as deepfake generation, image based phishing attacks, and fabrication of misleading visual evidence, underscores the need for robust authenticity verification mechanisms. While traditional statistical watermarking techniques have proven effective for autoregressive language models, their direct adaptation to image generation models encounters significant challenges due to a phenomenon we term retokenization mismatch, a disparity between original and retokenized sequences during the image generation process. To overcome this limitation, we propose C-reweight, a novel, distortion-free watermarking method explicitly designed for image generation models. By leveraging a clustering-based strategy that treats tokens within the same cluster equivalently, C-reweight mitigates retokenization mismatch while preserving image fidelity. Extensive evaluations on leading image generation platforms reveal that C-reweight not only maintains the visual quality of generated images but also improves detectability over existing distortion-free watermarking techniques, setting a new standard for secure and trustworthy image synthesis.
>
---
#### [new 071] Autonomous Computer Vision Development with Agentic AI
- **分类: cs.CV; cs.AI; cs.MA**

- **简介: 该论文属于计算机视觉任务，旨在解决自主配置和规划问题。通过Agentic AI方法，自动完成肺、心、肋骨分割的模型配置与训练。**

- **链接: [http://arxiv.org/pdf/2506.11140v1](http://arxiv.org/pdf/2506.11140v1)**

> **作者:** Jin Kim; Muhammad Wahi-Anwa; Sangyun Park; Shawn Shin; John M. Hoffman; Matthew S. Brown
>
> **备注:** The paper is 13 pages long and contains 4 figures
>
> **摘要:** Agentic Artificial Intelligence (AI) systems leveraging Large Language Models (LLMs) exhibit significant potential for complex reasoning, planning, and tool utilization. We demonstrate that a specialized computer vision system can be built autonomously from a natural language prompt using Agentic AI methods. This involved extending SimpleMind (SM), an open-source Cognitive AI environment with configurable tools for medical image analysis, with an LLM-based agent, implemented using OpenManus, to automate the planning (tool configuration) for a particular computer vision task. We provide a proof-of-concept demonstration that an agentic system can interpret a computer vision task prompt, plan a corresponding SimpleMind workflow by decomposing the task and configuring appropriate tools. From the user input prompt, "provide sm (SimpleMind) config for lungs, heart, and ribs segmentation for cxr (chest x-ray)"), the agent LLM was able to generate the plan (tool configuration file in YAML format), and execute SM-Learn (training) and SM-Think (inference) scripts autonomously. The computer vision agent automatically configured, trained, and tested itself on 50 chest x-ray images, achieving mean dice scores of 0.96, 0.82, 0.83, for lungs, heart, and ribs, respectively. This work shows the potential for autonomous planning and tool configuration that has traditionally been performed by a data scientist in the development of computer vision applications.
>
---
#### [new 072] VIBE: Can a VLM Read the Room?
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于社会推理任务，旨在解决VLMs在理解非语言社交线索上的不足，提出新任务并构建数据集进行评估。**

- **链接: [http://arxiv.org/pdf/2506.11162v1](http://arxiv.org/pdf/2506.11162v1)**

> **作者:** Tania Chakraborty; Eylon Caplan; Dan Goldwasser
>
> **备注:** Pre-print, under review
>
> **摘要:** Understanding human social behavior such as recognizing emotions and the social dynamics causing them is an important and challenging problem. While LLMs have made remarkable advances, they are limited to the textual domain and cannot account for the major role that non-verbal cues play in understanding social situations. Vision Language Models (VLMs) can potentially account for this gap, however their ability to make correct inferences over such social cues has received little attention. In this paper, we explore the capabilities of VLMs at social reasoning. We identify a previously overlooked limitation in VLMs: the Visual Social-Pragmatic Inference gap. To target this gap, we propose a new task for VLMs: Visual Social-Pragmatic Inference. We construct a high quality dataset to test the abilities of a VLM for this task and benchmark the performance of several VLMs on it.
>
---
#### [new 073] Towards a general-purpose foundation model for fMRI analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于fMRI分析任务，旨在解决现有方法的可重复性和迁移性问题。提出NeuroSTORM模型，通过预训练和优化策略提升跨任务性能。**

- **链接: [http://arxiv.org/pdf/2506.11167v1](http://arxiv.org/pdf/2506.11167v1)**

> **作者:** Cheng Wang; Yu Jiang; Zhihao Peng; Chenxin Li; Changbae Bang; Lin Zhao; Jinglei Lv; Jorge Sepulcre; Carl Yang; Lifang He; Tianming Liu; Daniel Barron; Quanzheng Li; Randy Hirschtick; Byung-Hoon Kim; Xiang Li; Yixuan Yuan
>
> **摘要:** Functional Magnetic Resonance Imaging (fMRI) is essential for studying brain function and diagnosing neurological disorders, but current analysis methods face reproducibility and transferability issues due to complex pre-processing and task-specific models. We introduce NeuroSTORM (Neuroimaging Foundation Model with Spatial-Temporal Optimized Representation Modeling), a generalizable framework that directly learns from 4D fMRI volumes and enables efficient knowledge transfer across diverse applications. NeuroSTORM is pre-trained on 28.65 million fMRI frames (>9,000 hours) from over 50,000 subjects across multiple centers and ages 5 to 100. Using a Mamba backbone and a shifted scanning strategy, it efficiently processes full 4D volumes. We also propose a spatial-temporal optimized pre-training approach and task-specific prompt tuning to improve transferability. NeuroSTORM outperforms existing methods across five tasks: age/gender prediction, phenotype prediction, disease diagnosis, fMRI-to-image retrieval, and task-based fMRI classification. It demonstrates strong clinical utility on datasets from hospitals in the U.S., South Korea, and Australia, achieving top performance in disease diagnosis and cognitive phenotype prediction. NeuroSTORM provides a standardized, open-source foundation model to improve reproducibility and transferability in fMRI-based clinical research.
>
---
#### [new 074] JAFAR: Jack up Any Feature at Any Resolution
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出JAFAR，解决视觉特征上采样问题，提升基础视觉编码器输出的分辨率，以满足下游任务需求。**

- **链接: [http://arxiv.org/pdf/2506.11136v1](http://arxiv.org/pdf/2506.11136v1)**

> **作者:** Paul Couairon; Loick Chambon; Louis Serrano; Jean-Emmanuel Haugeard; Matthieu Cord; Nicolas Thome
>
> **备注:** Code available at https://github.com/PaulCouairon/JAFAR
>
> **摘要:** Foundation Vision Encoders have become essential for a wide range of dense vision tasks. However, their low-resolution spatial feature outputs necessitate feature upsampling to produce the high-resolution modalities required for downstream tasks. In this work, we introduce JAFAR, a lightweight and flexible feature upsampler that enhances the spatial resolution of visual features from any Foundation Vision Encoder to an arbitrary target resolution. JAFAR employs an attention-based module designed to promote semantic alignment between high-resolution queries, derived from low-level image features, and semantically enriched low-resolution keys, using Spatial Feature Transform (SFT) modulation. Notably, despite the absence of high-resolution supervision, we demonstrate that learning at low upsampling ratios and resolutions generalizes remarkably well to significantly higher output scales. Extensive experiments show that JAFAR effectively recovers fine-grained spatial details and consistently outperforms existing feature upsampling methods across a diverse set of downstream tasks. Project page at https://jafar-upsampler.github.io
>
---
#### [new 075] Enhanced Vehicle Speed Detection Considering Lane Recognition Using Drone Videos in California
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于车辆速度检测任务，旨在解决车道识别不足和分类不准确的问题。通过改进YOLOv11模型，提升检测精度并区分车辆类型。**

- **链接: [http://arxiv.org/pdf/2506.11239v1](http://arxiv.org/pdf/2506.11239v1)**

> **作者:** Amirali Ataee Naeini; Ashkan Teymouri; Ghazaleh Jafarsalehi; Michael Zhang
>
> **备注:** 7 pages
>
> **摘要:** The increase in vehicle numbers in California, driven by inadequate transportation systems and sparse speed cameras, necessitates effective vehicle speed detection. Detecting vehicle speeds per lane is critical for monitoring High-Occupancy Vehicle (HOV) lane speeds, distinguishing between cars and heavy vehicles with differing speed limits, and enforcing lane restrictions for heavy vehicles. While prior works utilized YOLO (You Only Look Once) for vehicle speed detection, they often lacked accuracy, failed to identify vehicle lanes, and offered limited or less practical classification categories. This study introduces a fine-tuned YOLOv11 model, trained on almost 800 bird's-eye view images, to enhance vehicle speed detection accuracy which is much higher compare to the previous works. The proposed system identifies the lane for each vehicle and classifies vehicles into two categories: cars and heavy vehicles. Designed to meet the specific requirements of traffic monitoring and regulation, the model also evaluates the effects of factors such as drone height, distance of Region of Interest (ROI), and vehicle speed on detection accuracy and speed measurement. Drone footage collected from Northern California was used to assess the proposed system. The fine-tuned YOLOv11 achieved its best performance with a mean absolute error (MAE) of 0.97 mph and mean squared error (MSE) of 0.94 $\text{mph}^2$, demonstrating its efficacy in addressing challenges in vehicle speed detection and classification.
>
---
#### [new 076] Wi-CBR: WiFi-based Cross-domain Behavior Recognition via Multimodal Collaborative Awareness
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于行为识别任务，旨在通过融合WiFi信号的相位和多普勒频移数据，提升跨领域行为识别的准确性。**

- **链接: [http://arxiv.org/pdf/2506.11616v1](http://arxiv.org/pdf/2506.11616v1)**

> **作者:** Ruobei Zhang; Shengeng Tang; Huan Yan; Xiang Zhang; Richang Hong
>
> **摘要:** WiFi-based human behavior recognition aims to recognize gestures and activities by analyzing wireless signal variations. However, existing methods typically focus on a single type of data, neglecting the interaction and fusion of multiple features. To this end, we propose a novel multimodal collaborative awareness method. By leveraging phase data reflecting changes in dynamic path length and Doppler Shift (DFS) data corresponding to frequency changes related to the speed of gesture movement, we enable efficient interaction and fusion of these features to improve recognition accuracy. Specifically, we first introduce a dual-branch self-attention module to capture spatial-temporal cues within each modality. Then, a group attention mechanism is applied to the concatenated phase and DFS features to mine key group features critical for behavior recognition. Through a gating mechanism, the combined features are further divided into PD-strengthen and PD-weaken branches, optimizing information entropy and promoting cross-modal collaborative awareness. Extensive in-domain and cross-domain experiments on two large publicly available datasets, Widar3.0 and XRF55, demonstrate the superior performance of our method.
>
---
#### [new 077] Gender Fairness of Machine Learning Algorithms for Pain Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗AI任务，研究ML/DL算法在疼痛检测中的性别公平性问题，分析模型偏差并探讨准确率与公平性的平衡。**

- **链接: [http://arxiv.org/pdf/2506.11132v1](http://arxiv.org/pdf/2506.11132v1)**

> **作者:** Dylan Green; Yuting Shang; Jiaee Cheong; Yang Liu; Hatice Gunes
>
> **备注:** To appear as part of the 2025 19th International Conference on Automatic Face and Gesture Recognition (FG) Workshop Proceedings
>
> **摘要:** Automated pain detection through machine learning (ML) and deep learning (DL) algorithms holds significant potential in healthcare, particularly for patients unable to self-report pain levels. However, the accuracy and fairness of these algorithms across different demographic groups (e.g., gender) remain under-researched. This paper investigates the gender fairness of ML and DL models trained on the UNBC-McMaster Shoulder Pain Expression Archive Database, evaluating the performance of various models in detecting pain based solely on the visual modality of participants' facial expressions. We compare traditional ML algorithms, Linear Support Vector Machine (L SVM) and Radial Basis Function SVM (RBF SVM), with DL methods, Convolutional Neural Network (CNN) and Vision Transformer (ViT), using a range of performance and fairness metrics. While ViT achieved the highest accuracy and a selection of fairness metrics, all models exhibited gender-based biases. These findings highlight the persistent trade-off between accuracy and fairness, emphasising the need for fairness-aware techniques to mitigate biases in automated healthcare systems.
>
---
#### [new 078] Real-Time Feedback and Benchmark Dataset for Isometric Pose Evaluation
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于动作评估任务，旨在解决等长运动姿势错误识别问题。通过构建数据集和提出新指标，实现实时反馈与精准诊断。**

- **链接: [http://arxiv.org/pdf/2506.11774v1](http://arxiv.org/pdf/2506.11774v1)**

> **作者:** Abhishek Jaiswal; Armeet Singh Luthra; Purav Jangir; Bhavya Garg; Nisheeth Srivastava
>
> **摘要:** Isometric exercises appeal to individuals seeking convenience, privacy, and minimal dependence on equipments. However, such fitness training is often overdependent on unreliable digital media content instead of expert supervision, introducing serious risks, including incorrect posture, injury, and disengagement due to lack of corrective feedback. To address these challenges, we present a real-time feedback system for assessing isometric poses. Our contributions include the release of the largest multiclass isometric exercise video dataset to date, comprising over 3,600 clips across six poses with correct and incorrect variations. To support robust evaluation, we benchmark state-of-the-art models-including graph-based networks-on this dataset and introduce a novel three-part metric that captures classification accuracy, mistake localization, and model confidence. Our results enhance the feasibility of intelligent and personalized exercise training systems for home workouts. This expert-level diagnosis, delivered directly to the users, also expands the potential applications of these systems to rehabilitation, physiotherapy, and various other fitness disciplines that involve physical motion.
>
---
#### [new 079] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉分割任务，解决跨模态对齐问题。通过结合多模态基础模型和分割模型，提出TAViS框架，提升分割精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.11436v1](http://arxiv.org/pdf/2506.11436v1)**

> **作者:** Ziyang Luo; Nian Liu; Xuguang Yang; Salman Khan; Rao Muhammad Anwer; Hisham Cholakkal; Fahad Shahbaz Khan; Junwei Han
>
> **摘要:** Audio-Visual Segmentation (AVS) faces a fundamental challenge of effectively aligning audio and visual modalities. While recent approaches leverage foundation models to address data scarcity, they often rely on single-modality knowledge or combine foundation models in an off-the-shelf manner, failing to address the cross-modal alignment challenge. In this paper, we present TAViS, a novel framework that \textbf{couples} the knowledge of multimodal foundation models (ImageBind) for cross-modal alignment and a segmentation foundation model (SAM2) for precise segmentation. However, effectively combining these models poses two key challenges: the difficulty in transferring the knowledge between SAM2 and ImageBind due to their different feature spaces, and the insufficiency of using only segmentation loss for supervision. To address these challenges, we introduce a text-bridged design with two key components: (1) a text-bridged hybrid prompting mechanism where pseudo text provides class prototype information while retaining modality-specific details from both audio and visual inputs, and (2) an alignment supervision strategy that leverages text as a bridge to align shared semantic concepts within audio-visual modalities. Our approach achieves superior performance on single-source, multi-source, semantic datasets, and excels in zero-shot settings.
>
---
#### [new 080] GPLQ: A General, Practical, and Lightning QAT Method for Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer的量化任务，旨在解决低比特量化中的精度下降和计算成本高的问题。提出GPLQ方法，通过分阶段量化实现高效且高精度的模型压缩。**

- **链接: [http://arxiv.org/pdf/2506.11784v1](http://arxiv.org/pdf/2506.11784v1)**

> **作者:** Guang Liang; Xinyao Liu; Jianxin Wu
>
> **摘要:** Vision Transformers (ViTs) are essential in computer vision but are computationally intensive, too. Model quantization, particularly to low bit-widths like 4-bit, aims to alleviate this difficulty, yet existing Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) methods exhibit significant limitations. PTQ often incurs substantial accuracy drop, while QAT achieves high accuracy but suffers from prohibitive computational costs, limited generalization to downstream tasks, training instability, and lacking of open-source codebase. To address these challenges, this paper introduces General, Practical, and Lightning Quantization (GPLQ), a novel framework designed for efficient and effective ViT quantization. GPLQ is founded on two key empirical insights: the paramount importance of activation quantization and the necessity of preserving the model's original optimization ``basin'' to maintain generalization. Consequently, GPLQ employs a sequential ``activation-first, weights-later'' strategy. Stage 1 keeps weights in FP32 while quantizing activations with a feature mimicking loss in only 1 epoch to keep it stay in the same ``basin'', thereby preserving generalization. Stage 2 quantizes weights using a PTQ method. As a result, GPLQ is 100x faster than existing QAT methods, lowers memory footprint to levels even below FP32 training, and achieves 4-bit model performance that is highly competitive with FP32 models in terms of both accuracy on ImageNet and generalization to diverse downstream tasks, including fine-grained visual classification and object detection. We will release an easy-to-use open-source toolkit supporting multiple vision tasks.
>
---
#### [new 081] Monocular 3D Hand Pose Estimation with Implicit Camera Alignment
- **分类: cs.CV; cs.GR; cs.LG; eess.IV**

- **简介: 该论文属于单目3D手部姿态估计任务，解决无深度信息和相机参数下的手部结构重建问题。通过优化流程和关键点对齐方法，提升估计精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.11133v1](http://arxiv.org/pdf/2506.11133v1)**

> **作者:** Christos Pantazopoulos; Spyridon Thermos; Gerasimos Potamianos
>
> **备注:** Code is available at https://github.com/cpantazop/HandRepo
>
> **摘要:** Estimating the 3D hand articulation from a single color image is a continuously investigated problem with applications in Augmented Reality (AR), Virtual Reality (VR), Human-Computer Interaction (HCI), and robotics. Apart from the absence of depth information, occlusions, articulation complexity, and the need for camera parameters knowledge pose additional challenges. In this work, we propose an optimization pipeline for estimating the 3D hand articulation from 2D keypoint input, which includes a keypoint alignment step and a fingertip loss to overcome the need to know or estimate the camera parameters. We evaluate our approach on the EgoDexter and Dexter+Object benchmarks to showcase that our approach performs competitively with the SotA, while also demonstrating its robustness when processing "in-the-wild" images without any prior camera knowledge. Our quantitative analysis highlights the sensitivity of the 2D keypoint estimation accuracy, despite the use of hand priors. Code is available at https://github.com/cpantazop/HandRepo
>
---
#### [new 082] Image-Based Method For Measuring And Classification Of Iron Ore Pellets Using Star-Convex Polygons
- **分类: cs.CV**

- **简介: 该论文属于图像分类与测量任务，旨在解决铁矿石球团质量检测问题，通过改进的StarDist算法实现更准确的尺寸测量与分类。**

- **链接: [http://arxiv.org/pdf/2506.11126v1](http://arxiv.org/pdf/2506.11126v1)**

> **作者:** Artem Solomko; Oleg Kartashev; Andrey Golov; Mikhail Deulin; Vadim Valynkin; Vasily Kharin
>
> **备注:** 15 pages, 41 figures
>
> **摘要:** We would like to present a comprehensive study on the classification of iron ore pellets, aimed at identifying quality violations in the final product, alongside the development of an innovative imagebased measurement method utilizing the StarDist algorithm, which is primarily employed in the medical field. This initiative is motivated by the necessity to accurately identify and analyze objects within densely packed and unstable environments. The process involves segmenting these objects, determining their contours, classifying them, and measuring their physical dimensions. This is crucial because the size distribution and classification of pellets such as distinguishing between nice (quality) and joint (caused by the presence of moisture or indicating a process of production failure) types are among the most significant characteristics that define the quality of the final product. Traditional algorithms, including image classification techniques using Vision Transformer (ViT), instance segmentation methods like Mask R-CNN, and various anomaly segmentation algorithms, have not yielded satisfactory results in this context. Consequently, we explored methodologies from related fields to enhance our approach. The outcome of our research is a novel method designed to detect objects with smoothed boundaries. This advancement significantly improves the accuracy of physical dimension measurements and facilitates a more precise analysis of size distribution among the iron ore pellets. By leveraging the strengths of the StarDist algorithm, we aim to provide a robust solution that addresses the challenges posed by the complex nature of pellet classification and measurement.
>
---
#### [new 083] GNSS-inertial state initialization by distance residuals
- **分类: cs.CV**

- **简介: 该论文属于导航定位任务，解决GNSS-inertial系统初始化问题。通过利用距离残差延迟使用全局GNSS数据，提升初始化精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.11534v1](http://arxiv.org/pdf/2506.11534v1)**

> **作者:** Samuel Cerezo; Javier Civera
>
> **备注:** 8 pages, 8 figures, RA-L submission
>
> **摘要:** Initializing the state of a sensorized platform can be challenging, as a limited set of initial measurements often carry limited information, leading to poor initial estimates that may converge to local minima during non-linear optimization. This paper proposes a novel GNSS-inertial initialization strategy that delays the use of global GNSS measurements until sufficient information is available to accurately estimate the transformation between the GNSS and inertial frames. Instead, the method initially relies on GNSS relative distance residuals. To determine the optimal moment for switching to global measurements, we introduce a criterion based on the evolution of the Hessian matrix singular values. Experiments on the EuRoC and GVINS datasets show that our approach consistently outperforms the naive strategy of using global GNSS data from the start, yielding more accurate and robust initializations.
>
---
#### [new 084] Dynamic Double Space Tower
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答任务，旨在解决复杂推理和空间关系理解问题。提出动态双层空间塔结构，增强模型对图像空间关系的感知与组织能力。**

- **链接: [http://arxiv.org/pdf/2506.11394v1](http://arxiv.org/pdf/2506.11394v1)**

> **作者:** Weikai Sun; Shijie Song; Han Wang
>
> **摘要:** The Visual Question Answering (VQA) task requires the simultaneous understanding of image content and question semantics. However, existing methods often have difficulty handling complex reasoning scenarios due to insufficient cross-modal interaction and capturing the entity spatial relationships in the image.\cite{huang2023adaptive}\cite{liu2021comparing}\cite{guibas2021adaptive}\cite{zhang2022vsa}We studied a brand-new approach to replace the attention mechanism in order to enhance the reasoning ability of the model and its understanding of spatial relationships.Specifically, we propose a dynamic bidirectional spatial tower, which is divided into four layers to observe the image according to the principle of human gestalt vision. This naturally provides a powerful structural prior for the spatial organization between entities, enabling the model to no longer blindly search for relationships between pixels but make judgments based on more meaningful perceptual units. Change from "seeing images" to "perceiving and organizing image content".A large number of experiments have shown that our module can be used in any other multimodal model and achieve advanced results, demonstrating its potential in spatial relationship processing.Meanwhile, the multimodal visual question-answering model July trained by our method has achieved state-of-the-art results with only 3B parameters, especially on the question-answering dataset of spatial relations.
>
---
#### [new 085] Quizzard@INOVA Challenge 2025 -- Track A: Plug-and-Play Technique in Interleaved Multi-Image Model
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多图像模型任务，旨在提升跨图像推理与交互能力。通过引入DCI连接器优化模型，对比实验验证了不同版本在不同数据集上的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.11737v1](http://arxiv.org/pdf/2506.11737v1)**

> **作者:** Dinh Viet Cuong; Hoang-Bao Le; An Pham Ngoc Nguyen; Liting Zhou; Cathal Gurrin
>
> **摘要:** This paper addresses two main objectives. Firstly, we demonstrate the impressive performance of the LLaVA-NeXT-interleave on 22 datasets across three different tasks: Multi-Image Reasoning, Documents and Knowledge-Based Understanding and Interactive Multi-Modal Communication. Secondly, we add the Dense Channel Integration (DCI) connector to the LLaVA-NeXT-Interleave and compare its performance against the standard model. We find that the standard model achieves the highest overall accuracy, excelling in vision-heavy tasks like VISION, NLVR2, and Fashion200K. Meanwhile, the DCI-enhanced version shows particular strength on datasets requiring deeper semantic coherence or structured change understanding such as MIT-States_PropertyCoherence and SlideVQA. Our results highlight the potential of combining powerful foundation models with plug-and-play techniques for Interleave tasks. The code is available at https://github.com/dinhvietcuong1996/icme25-inova.
>
---
#### [new 086] ContextLoss: Context Information for Topology-Preserving Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像分割任务，旨在解决结构拓扑保持问题。提出ContextLoss损失函数，提升分割结果的连通性与拓扑正确性。**

- **链接: [http://arxiv.org/pdf/2506.11134v1](http://arxiv.org/pdf/2506.11134v1)**

> **作者:** Benedict Schacht; Imke Greving; Simone Frintrop; Berit Zeller-Plumhoff; Christian Wilms
>
> **备注:** 13 pages, 7 figures, accepted to ICIP 2025
>
> **摘要:** In image segmentation, preserving the topology of segmented structures like vessels, membranes, or roads is crucial. For instance, topological errors on road networks can significantly impact navigation. Recently proposed solutions are loss functions based on critical pixel masks that consider the whole skeleton of the segmented structures in the critical pixel mask. We propose the novel loss function ContextLoss (CLoss) that improves topological correctness by considering topological errors with their whole context in the critical pixel mask. The additional context improves the network focus on the topological errors. Further, we propose two intuitive metrics to verify improved connectivity due to a closing of missed connections. We benchmark our proposed CLoss on three public datasets (2D & 3D) and our own 3D nano-imaging dataset of bone cement lines. Training with our proposed CLoss increases performance on topology-aware metrics and repairs up to 44% more missed connections than other state-of-the-art methods. We make the code publicly available.
>
---
#### [new 087] Teleoperated Driving: a New Challenge for 3D Object Detection in Compressed Point Clouds
- **分类: cs.CV; cs.NI; eess.IV**

- **简介: 该论文属于3D目标检测任务，旨在解决压缩点云数据中的车辆和行人检测问题，分析压缩算法与检测模型的性能及对V2X网络的影响。**

- **链接: [http://arxiv.org/pdf/2506.11804v1](http://arxiv.org/pdf/2506.11804v1)**

> **作者:** Filippo Bragato; Michael Neri; Paolo Testolina; Marco Giordani; Federica Battisti
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems
>
> **摘要:** In recent years, the development of interconnected devices has expanded in many fields, from infotainment to education and industrial applications. This trend has been accelerated by the increased number of sensors and accessibility to powerful hardware and software. One area that significantly benefits from these advancements is Teleoperated Driving (TD). In this scenario, a controller drives safely a vehicle from remote leveraging sensors data generated onboard the vehicle, and exchanged via Vehicle-to-Everything (V2X) communications. In this work, we tackle the problem of detecting the presence of cars and pedestrians from point cloud data to enable safe TD operations. More specifically, we exploit the SELMA dataset, a multimodal, open-source, synthetic dataset for autonomous driving, that we expanded by including the ground-truth bounding boxes of 3D objects to support object detection. We analyze the performance of state-of-the-art compression algorithms and object detectors under several metrics, including compression efficiency, (de)compression and inference time, and detection accuracy. Moreover, we measure the impact of compression and detection on the V2X network in terms of data rate and latency with respect to 3GPP requirements for TD applications.
>
---
#### [new 088] SphereDrag: Spherical Geometry-Aware Panoramic Image Editing
- **分类: cs.CV**

- **简介: 该论文属于全景图像编辑任务，解决边界不连续、轨迹变形和像素密度不均问题。提出SphereDrag框架，结合球面几何知识提升编辑精度与一致性。**

- **链接: [http://arxiv.org/pdf/2506.11863v1](http://arxiv.org/pdf/2506.11863v1)**

> **作者:** Zhiao Feng; Xuewei Li; Junjie Yang; Yuxin Peng; Xi Li
>
> **摘要:** Image editing has made great progress on planar images, but panoramic image editing remains underexplored. Due to their spherical geometry and projection distortions, panoramic images present three key challenges: boundary discontinuity, trajectory deformation, and uneven pixel density. To tackle these issues, we propose SphereDrag, a novel panoramic editing framework utilizing spherical geometry knowledge for accurate and controllable editing. Specifically, adaptive reprojection (AR) uses adaptive spherical rotation to deal with discontinuity; great-circle trajectory adjustment (GCTA) tracks the movement trajectory more accurate; spherical search region tracking (SSRT) adaptively scales the search range based on spherical location to address uneven pixel density. Also, we construct PanoBench, a panoramic editing benchmark, including complex editing tasks involving multiple objects and diverse styles, which provides a standardized evaluation framework. Experiments show that SphereDrag gains a considerable improvement compared with existing methods in geometric consistency and image quality, achieving up to 10.5% relative improvement.
>
---
#### [new 089] Cross-Modal Clustering-Guided Negative Sampling for Self-Supervised Joint Learning from Medical Images and Reports
- **分类: cs.CV**

- **简介: 该论文属于医学图像与报告的联合学习任务，旨在解决负样本不足、忽略局部细节等问题。提出CM-CGNS方法，通过跨模态聚类和掩码重建提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11674v1](http://arxiv.org/pdf/2506.11674v1)**

> **作者:** Libin Lan; Hongxing Li; Zunhui Xia; Juan Zhou; Xiaofei Zhu; Yongmei Li; Yudong Zhang; Xin Luo
>
> **备注:** This work has been submitted to the IEEE TMI for possible publication. Our code is available at https://github.com/violet-42/CM-CGNS
>
> **摘要:** Learning medical visual representations directly from paired images and reports through multimodal self-supervised learning has emerged as a novel and efficient approach to digital diagnosis in recent years. However, existing models suffer from several severe limitations. 1) neglecting the selection of negative samples, resulting in the scarcity of hard negatives and the inclusion of false negatives; 2) focusing on global feature extraction, but overlooking the fine-grained local details that are crucial for medical image recognition tasks; and 3) contrastive learning primarily targets high-level features but ignoring low-level details which are essential for accurate medical analysis. Motivated by these critical issues, this paper presents a Cross-Modal Cluster-Guided Negative Sampling (CM-CGNS) method with two-fold ideas. First, it extends the k-means clustering used for local text features in the single-modal domain to the multimodal domain through cross-modal attention. This improvement increases the number of negative samples and boosts the model representation capability. Second, it introduces a Cross-Modal Masked Image Reconstruction (CM-MIR) module that leverages local text-to-image features obtained via cross-modal attention to reconstruct masked local image regions. This module significantly strengthens the model's cross-modal information interaction capabilities and retains low-level image features essential for downstream tasks. By well handling the aforementioned limitations, the proposed CM-CGNS can learn effective and robust medical visual representations suitable for various recognition tasks. Extensive experimental results on classification, detection, and segmentation tasks across five downstream datasets show that our method outperforms state-of-the-art approaches on multiple metrics, verifying its superior performance.
>
---
#### [new 090] Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling
- **分类: cs.CV**

- **简介: 该论文属于安全X光图像中违禁物品的实例分割任务，旨在解决X光图像与自然物体的外观差异及物品重叠问题。通过引入SAM模型和设计遮挡感知的双层解码器模块实现有效分割。**

- **链接: [http://arxiv.org/pdf/2506.11661v1](http://arxiv.org/pdf/2506.11661v1)**

> **作者:** Yunhan Ren; Ruihuang Li; Lingbo Liu; Changwen Chen
>
> **备注:** Accepted by ICME 2025
>
> **摘要:** Instance segmentation of prohibited items in security X-ray images is a critical yet challenging task. This is mainly caused by the significant appearance gap between prohibited items in X-ray images and natural objects, as well as the severe overlapping among objects in X-ray images. To address these issues, we propose an occlusion-aware instance segmentation pipeline designed to identify prohibited items in X-ray images. Specifically, to bridge the representation gap, we integrate the Segment Anything Model (SAM) into our pipeline, taking advantage of its rich priors and zero-shot generalization capabilities. To address the overlap between prohibited items, we design an occlusion-aware bilayer mask decoder module that explicitly models the occlusion relationships. To supervise occlusion estimation, we manually annotated occlusion areas of prohibited items in two large-scale X-ray image segmentation datasets, PIDray and PIXray. We then reorganized these additional annotations together with the original information as two occlusion-annotated datasets, PIDray-A and PIXray-A. Extensive experimental results on these occlusion-annotated datasets demonstrate the effectiveness of our proposed method. The datasets and codes are available at: https://github.com/Ryh1218/Occ
>
---
#### [new 091] LLM-to-Phy3D: Physically Conform Online 3D Object Generation with LLMs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D生成任务，旨在解决LLM生成的3D对象缺乏物理合理性的问题。通过引入物理反馈机制，提升生成对象的物理可行性与几何新颖性。**

- **链接: [http://arxiv.org/pdf/2506.11148v1](http://arxiv.org/pdf/2506.11148v1)**

> **作者:** Melvin Wong; Yueming Lyu; Thiago Rios; Stefan Menzel; Yew-Soon Ong
>
> **摘要:** The emergence of generative artificial intelligence (GenAI) and large language models (LLMs) has revolutionized the landscape of digital content creation in different modalities. However, its potential use in Physical AI for engineering design, where the production of physically viable artifacts is paramount, remains vastly underexplored. The absence of physical knowledge in existing LLM-to-3D models often results in outputs detached from real-world physical constraints. To address this gap, we introduce LLM-to-Phy3D, a physically conform online 3D object generation that enables existing LLM-to-3D models to produce physically conforming 3D objects on the fly. LLM-to-Phy3D introduces a novel online black-box refinement loop that empowers large language models (LLMs) through synergistic visual and physics-based evaluations. By delivering directional feedback in an iterative refinement process, LLM-to-Phy3D actively drives the discovery of prompts that yield 3D artifacts with enhanced physical performance and greater geometric novelty relative to reference objects, marking a substantial contribution to AI-driven generative design. Systematic evaluations of LLM-to-Phy3D, supported by ablation studies in vehicle design optimization, reveal various LLM improvements gained by 4.5% to 106.7% in producing physically conform target domain 3D designs over conventional LLM-to-3D models. The encouraging results suggest the potential general use of LLM-to-Phy3D in Physical AI for scientific and engineering applications.
>
---
#### [new 092] CLAIM: Mitigating Multilingual Object Hallucination in Large Vision-Language Models with Cross-Lingual Attention Intervention
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多语言视觉-语言模型任务，旨在解决多语言物体幻觉问题。通过跨语言注意力干预（CLAIM）方法，提升模型在非英语查询下的视觉一致性。**

- **链接: [http://arxiv.org/pdf/2506.11073v1](http://arxiv.org/pdf/2506.11073v1)**

> **作者:** Zekai Ye; Qiming Li; Xiaocheng Feng; Libo Qin; Yichong Huang; Baohang Li; Kui Jiang; Yang Xiang; Zhirui Zhang; Yunfei Lu; Duyu Tang; Dandan Tu; Bing Qin
>
> **备注:** ACL2025 Main
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal abilities but remain prone to multilingual object hallucination, with a higher likelihood of generating responses inconsistent with the visual input when utilizing queries in non-English languages compared to English. Most existing approaches to address these rely on pretraining or fine-tuning, which are resource-intensive. In this paper, inspired by observing the disparities in cross-modal attention patterns across languages, we propose Cross-Lingual Attention Intervention for Mitigating multilingual object hallucination (CLAIM) in LVLMs, a novel near training-free method by aligning attention patterns. CLAIM first identifies language-specific cross-modal attention heads, then estimates language shift vectors from English to the target language, and finally intervenes in the attention outputs during inference to facilitate cross-lingual visual perception capability alignment. Extensive experiments demonstrate that CLAIM achieves an average improvement of 13.56% (up to 30% in Spanish) on the POPE and 21.75% on the hallucination subsets of the MME benchmark across various languages. Further analysis reveals that multilingual attention divergence is most prominent in intermediate layers, highlighting their critical role in multilingual scenarios.
>
---
#### [new 093] Structural Similarity-Inspired Unfolding for Lightweight Image Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决模型复杂度与性能之间的平衡问题。提出SSIU方法，结合数据驱动与模型驱动优势，提升效率并减少参数量。**

- **链接: [http://arxiv.org/pdf/2506.11823v1](http://arxiv.org/pdf/2506.11823v1)**

> **作者:** Zhangkai Ni; Yang Zhang; Wenhan Yang; Hanli Wang; Shiqi Wang; Sam Kwong
>
> **备注:** Accepted to IEEE Transactions on Image Processing
>
> **摘要:** Major efforts in data-driven image super-resolution (SR) primarily focus on expanding the receptive field of the model to better capture contextual information. However, these methods are typically implemented by stacking deeper networks or leveraging transformer-based attention mechanisms, which consequently increases model complexity. In contrast, model-driven methods based on the unfolding paradigm show promise in improving performance while effectively maintaining model compactness through sophisticated module design. Based on these insights, we propose a Structural Similarity-Inspired Unfolding (SSIU) method for efficient image SR. This method is designed through unfolding an SR optimization function constrained by structural similarity, aiming to combine the strengths of both data-driven and model-driven approaches. Our model operates progressively following the unfolding paradigm. Each iteration consists of multiple Mixed-Scale Gating Modules (MSGM) and an Efficient Sparse Attention Module (ESAM). The former implements comprehensive constraints on features, including a structural similarity constraint, while the latter aims to achieve sparse activation. In addition, we design a Mixture-of-Experts-based Feature Selector (MoE-FS) that fully utilizes multi-level feature information by combining features from different steps. Extensive experiments validate the efficacy and efficiency of our unfolding-inspired network. Our model outperforms current state-of-the-art models, boasting lower parameter counts and reduced memory consumption. Our code will be available at: https://github.com/eezkni/SSIU
>
---
#### [new 094] FCA2: Frame Compression-Aware Autoencoder for Modular and Fast Compressed Video Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频超分辨率任务，旨在解决压缩视频超分辨率中的计算复杂和推理慢问题。提出FCA2模型，通过压缩感知策略提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.11545v1](http://arxiv.org/pdf/2506.11545v1)**

> **作者:** Zhaoyang Wang; Jie Li; Wen Lu; Lihuo He; Maoguo Gong; Xinbo Gao
>
> **备注:** This work has been submitted to the IEEE TMM for possible publication
>
> **摘要:** State-of-the-art (SOTA) compressed video super-resolution (CVSR) models face persistent challenges, including prolonged inference time, complex training pipelines, and reliance on auxiliary information. As video frame rates continue to increase, the diminishing inter-frame differences further expose the limitations of traditional frame-to-frame information exploitation methods, which are inadequate for addressing current video super-resolution (VSR) demands. To overcome these challenges, we propose an efficient and scalable solution inspired by the structural and statistical similarities between hyperspectral images (HSI) and video data. Our approach introduces a compression-driven dimensionality reduction strategy that reduces computational complexity, accelerates inference, and enhances the extraction of temporal information across frames. The proposed modular architecture is designed for seamless integration with existing VSR frameworks, ensuring strong adaptability and transferability across diverse applications. Experimental results demonstrate that our method achieves performance on par with, or surpassing, the current SOTA models, while significantly reducing inference time. By addressing key bottlenecks in CVSR, our work offers a practical and efficient pathway for advancing VSR technology. Our code will be publicly available at https://github.com/handsomewzy/FCA2.
>
---
#### [new 095] CGVQM+D: Computer Graphics Video Quality Metric and Dataset
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于视频质量评估任务，解决合成内容和现代渲染失真评价问题。构建了新数据集并提出CGVQM度量方法，提升质量评估性能。**

- **链接: [http://arxiv.org/pdf/2506.11546v1](http://arxiv.org/pdf/2506.11546v1)**

> **作者:** Akshay Jindal; Nabil Sadaka; Manu Mathew Thomas; Anton Sochenov; Anton Kaplanyan
>
> **摘要:** While existing video and image quality datasets have extensively studied natural videos and traditional distortions, the perception of synthetic content and modern rendering artifacts remains underexplored. We present a novel video quality dataset focused on distortions introduced by advanced rendering techniques, including neural supersampling, novel-view synthesis, path tracing, neural denoising, frame interpolation, and variable rate shading. Our evaluations show that existing full-reference quality metrics perform sub-optimally on these distortions, with a maximum Pearson correlation of 0.78. Additionally, we find that the feature space of pre-trained 3D CNNs aligns strongly with human perception of visual quality. We propose CGVQM, a full-reference video quality metric that significantly outperforms existing metrics while generating both per-pixel error maps and global quality scores. Our dataset and metric implementation is available at https://github.com/IntelLabs/CGVQM.
>
---
#### [new 096] FAD-Net: Frequency-Domain Attention-Guided Diffusion Network for Coronary Artery Segmentation using Invasive Coronary Angiography
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决冠状动脉精准分割与狭窄检测问题。提出FAD-Net模型，结合频域注意力与扩散策略提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.11454v1](http://arxiv.org/pdf/2506.11454v1)**

> **作者:** Nan Mu; Ruiqi Song; Xiaoning Li; Zhihui Xu; Jingfeng Jiang; Chen Zhao
>
> **备注:** 35 pages, 12 figures
>
> **摘要:** Background: Coronary artery disease (CAD) remains one of the leading causes of mortality worldwide. Precise segmentation of coronary arteries from invasive coronary angiography (ICA) is critical for effective clinical decision-making. Objective: This study aims to propose a novel deep learning model based on frequency-domain analysis to enhance the accuracy of coronary artery segmentation and stenosis detection in ICA, thereby offering robust support for the stenosis detection and treatment of CAD. Methods: We propose the Frequency-Domain Attention-Guided Diffusion Network (FAD-Net), which integrates a frequency-domain-based attention mechanism and a cascading diffusion strategy to fully exploit frequency-domain information for improved segmentation accuracy. Specifically, FAD-Net employs a Multi-Level Self-Attention (MLSA) mechanism in the frequency domain, computing the similarity between queries and keys across high- and low-frequency components in ICAs. Furthermore, a Low-Frequency Diffusion Module (LFDM) is incorporated to decompose ICAs into low- and high-frequency components via multi-level wavelet transformation. Subsequently, it refines fine-grained arterial branches and edges by reintegrating high-frequency details via inverse fusion, enabling continuous enhancement of anatomical precision. Results and Conclusions: Extensive experiments demonstrate that FAD-Net achieves a mean Dice coefficient of 0.8717 in coronary artery segmentation, outperforming existing state-of-the-art methods. In addition, it attains a true positive rate of 0.6140 and a positive predictive value of 0.6398 in stenosis detection, underscoring its clinical applicability. These findings suggest that FAD-Net holds significant potential to assist in the accurate diagnosis and treatment planning of CAD.
>
---
#### [new 097] Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决语言指令下物体和环境泛化问题。提出Gondola模型，结合多视角图像与历史计划生成精准动作规划。**

- **链接: [http://arxiv.org/pdf/2506.11261v1](http://arxiv.org/pdf/2506.11261v1)**

> **作者:** Shizhe Chen; Ricardo Garcia; Paul Pacaud; Cordelia Schmid
>
> **摘要:** Robotic manipulation faces a significant challenge in generalizing across unseen objects, environments and tasks specified by diverse language instructions. To improve generalization capabilities, recent research has incorporated large language models (LLMs) for planning and action execution. While promising, these methods often fall short in generating grounded plans in visual environments. Although efforts have been made to perform visual instructional tuning on LLMs for robotic manipulation, existing methods are typically constrained by single-view image input and struggle with precise object grounding. In this work, we introduce Gondola, a novel grounded vision-language planning model based on LLMs for generalizable robotic manipulation. Gondola takes multi-view images and history plans to produce the next action plan with interleaved texts and segmentation masks of target objects and locations. To support the training of Gondola, we construct three types of datasets using the RLBench simulator, namely robot grounded planning, multi-view referring expression and pseudo long-horizon task datasets. Gondola outperforms the state-of-the-art LLM-based method across all four generalization levels of the GemBench dataset, including novel placements, rigid objects, articulated objects and long-horizon tasks.
>
---
#### [new 098] Joint Denoising of Cryo-EM Projection Images using Polar Transformers
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于图像去噪任务，旨在解决高噪声下冷冻电镜投影图像的去噪问题。通过结合聚类、对齐和去噪的Transformer架构，提升去噪效果。**

- **链接: [http://arxiv.org/pdf/2506.11283v1](http://arxiv.org/pdf/2506.11283v1)**

> **作者:** Joakim Andén; Justus Sagemüller
>
> **摘要:** Deep neural networks~(DNNs) have proven powerful for denoising, but they are ultimately of limited use in high-noise settings, such as for cryogenic electron microscopy~(cryo-EM) projection images. In this setting, however, datasets contain a large number of projections of the same molecule, each taken from a different viewing direction. This redundancy of information is useful in traditional denoising techniques known as class averaging methods, where images are clustered, aligned, and then averaged to reduce the noise level. We present a neural network architecture based on transformers that extends these class averaging methods by simultaneously clustering, aligning, and denoising cryo-EM images. Results on synthetic data show accurate denoising performance using this architecture, reducing the relative mean squared error (MSE) single-image DNNs by $45\%$ at a signal-to-noise (SNR) of $0.03$.
>
---
#### [new 099] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出EMLoC，解决大模型微调内存过高的问题，通过轻量级模拟器和LoRA修正实现高效微调。**

- **链接: [http://arxiv.org/pdf/2506.12015v1](http://arxiv.org/pdf/2506.12015v1)**

> **作者:** Hsi-Che Lin; Yu-Chu Yu; Kai-Po Chang; Yu-Chiang Frank Wang
>
> **备注:** Under review. Project page: https://hsi-che-lin.github.io/EMLoC/
>
> **摘要:** Open-source foundation models have seen rapid adoption and development, enabling powerful general-purpose capabilities across diverse domains. However, fine-tuning large foundation models for domain-specific or personalized tasks remains prohibitively expensive for most users due to the significant memory overhead beyond that of inference. We introduce EMLoC, an Emulator-based Memory-efficient fine-tuning framework with LoRA Correction, which enables model fine-tuning within the same memory budget required for inference. EMLoC constructs a task-specific light-weight emulator using activation-aware singular value decomposition (SVD) on a small downstream calibration set. Fine-tuning then is performed on this lightweight emulator via LoRA. To tackle the misalignment between the original model and the compressed emulator, we propose a novel compensation algorithm to correct the fine-tuned LoRA module, which thus can be merged into the original model for inference. EMLoC supports flexible compression ratios and standard training pipelines, making it adaptable to a wide range of applications. Extensive experiments demonstrate that EMLoC outperforms other baselines across multiple datasets and modalities. Moreover, without quantization, EMLoC enables fine-tuning of a 38B model on a single 24GB consumer GPU-bringing efficient and practical model adaptation to individual users.
>
---
#### [new 100] SIMSHIFT: A Benchmark for Adapting Neural Surrogates to Distribution Shifts
- **分类: cs.LG; cs.CV; physics.comp-ph**

- **简介: 该论文属于神经代理模型的领域适应任务，旨在解决模型在分布偏移下的性能下降问题。通过构建SIMSHIFT基准和扩展DA方法进行评估。**

- **链接: [http://arxiv.org/pdf/2506.12007v1](http://arxiv.org/pdf/2506.12007v1)**

> **作者:** Paul Setinek; Gianluca Galletti; Thomas Gross; Dominik Schnürer; Johannes Brandstetter; Werner Zellinger
>
> **摘要:** Neural surrogates for Partial Differential Equations (PDEs) often suffer significant performance degradation when evaluated on unseen problem configurations, such as novel material types or structural dimensions. Meanwhile, Domain Adaptation (DA) techniques have been widely used in vision and language processing to generalize from limited information about unseen configurations. In this work, we address this gap through two focused contributions. First, we introduce SIMSHIFT, a novel benchmark dataset and evaluation suite composed of four industrial simulation tasks: hot rolling, sheet metal forming, electric motor design and heatsink design. Second, we extend established domain adaptation methods to state of the art neural surrogates and systematically evaluate them. These approaches use parametric descriptions and ground truth simulations from multiple source configurations, together with only parametric descriptions from target configurations. The goal is to accurately predict target simulations without access to ground truth simulation data. Extensive experiments on SIMSHIFT highlight the challenges of out of distribution neural surrogate modeling, demonstrate the potential of DA in simulation, and reveal open problems in achieving robust neural surrogates under distribution shifts in industrially relevant scenarios. Our codebase is available at https://github.com/psetinek/simshift
>
---
#### [new 101] DiffPR: Diffusion-Based Phase Reconstruction via Frequency-Decoupled Learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于定量相位成像任务，解决深度学习中的过平滑问题。通过取消高阶跳跃连接并引入扩散模型，提升相位重建的细节和精度。**

- **链接: [http://arxiv.org/pdf/2506.11183v1](http://arxiv.org/pdf/2506.11183v1)**

> **作者:** Yi Zhang
>
> **摘要:** Oversmoothing remains a persistent problem when applying deep learning to off-axis quantitative phase imaging (QPI). End-to-end U-Nets favour low-frequency content and under-represent fine, diagnostic detail. We trace this issue to spectral bias and show that the bias is reinforced by high-level skip connections that feed high-frequency features directly into the decoder. Removing those deepest skips thus supervising the network only at a low resolution significantly improves generalisation and fidelity. Building on this insight, we introduce DiffPR, a two-stage frequency-decoupled framework. Stage 1: an asymmetric U-Net with cancelled high-frequency skips predicts a quarter-scale phase map from the interferogram, capturing reliable low-frequency structure while avoiding spectral bias. Stage 2: the upsampled prediction, lightly perturbed with Gaussian noise, is refined by an unconditional diffusion model that iteratively recovers the missing high-frequency residuals through reverse denoising. Experiments on four QPI datasets (B-Cell, WBC, HeLa, 3T3) show that DiffPR outperforms strong U-Net baselines, boosting PSNR by up to 1.1 dB and reducing MAE by 11 percent, while delivering markedly sharper membrane ridges and speckle patterns. The results demonstrate that cancelling high-level skips and delegating detail synthesis to a diffusion prior is an effective remedy for the spectral bias that limits conventional phase-retrieval networks.
>
---
#### [new 102] Sparse Autoencoders Bridge The Deep Learning Model and The Brain
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文属于模型与大脑对齐任务，旨在将深度学习视觉模型与fMRI响应直接关联。通过稀疏自编码器建立映射，揭示模型层与视觉皮层的层次结构对应关系。**

- **链接: [http://arxiv.org/pdf/2506.11123v1](http://arxiv.org/pdf/2506.11123v1)**

> **作者:** Ziming Mao; Jia Xu; Zeqi Zheng; Haofang Zheng; Dabing Sheng; Yaochu Jin; Guoyuan Yang
>
> **备注:** 54 pages, 41 figures
>
> **摘要:** We present SAE-BrainMap, a novel framework that directly aligns deep learning visual model representations with voxel-level fMRI responses using sparse autoencoders (SAEs). First, we train layer-wise SAEs on model activations and compute the correlations between SAE unit activations and cortical fMRI signals elicited by the same natural image stimuli with cosine similarity, revealing strong activation correspondence (maximum similarity up to 0.76). Depending on this alignment, we construct a voxel dictionary by optimally assigning the most similar SAE feature to each voxel, demonstrating that SAE units preserve the functional structure of predefined regions of interest (ROIs) and exhibit ROI-consistent selectivity. Finally, we establish fine-grained hierarchical mapping between model layers and the human ventral visual pathway, also by projecting voxel dictionary activations onto individual cortical surfaces, we visualize the dynamic transformation of the visual information in deep learning models. It is found that ViT-B/16$_{CLIP}$ tends to utilize low-level information to generate high-level semantic information in the early layers and reconstructs the low-dimension information later. Our results establish a direct, downstream-task-free bridge between deep neural networks and human visual cortex, offering new insights into model interpretability.
>
---
#### [new 103] ADAgent: LLM Agent for Alzheimer's Disease Analysis with Collaborative Coordinator
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出ADAgent，用于阿尔茨海默病分析的AI系统，解决多模态数据处理与任务整合问题，提升诊断与预后准确性。**

- **链接: [http://arxiv.org/pdf/2506.11150v1](http://arxiv.org/pdf/2506.11150v1)**

> **作者:** Wenlong Hou; Gangqian Yang; Ye Du; Yeung Lau; Lihao Liu; Junjun He; Ling Long; Shujun Wang
>
> **摘要:** Alzheimer's disease (AD) is a progressive and irreversible neurodegenerative disease. Early and precise diagnosis of AD is crucial for timely intervention and treatment planning to alleviate the progressive neurodegeneration. However, most existing methods rely on single-modality data, which contrasts with the multifaceted approach used by medical experts. While some deep learning approaches process multi-modal data, they are limited to specific tasks with a small set of input modalities and cannot handle arbitrary combinations. This highlights the need for a system that can address diverse AD-related tasks, process multi-modal or missing input, and integrate multiple advanced methods for improved performance. In this paper, we propose ADAgent, the first specialized AI agent for AD analysis, built on a large language model (LLM) to address user queries and support decision-making. ADAgent integrates a reasoning engine, specialized medical tools, and a collaborative outcome coordinator to facilitate multi-modal diagnosis and prognosis tasks in AD. Extensive experiments demonstrate that ADAgent outperforms SOTA methods, achieving significant improvements in accuracy, including a 2.7% increase in multi-modal diagnosis, a 0.7% improvement in multi-modal prognosis, and enhancements in MRI and PET diagnosis tasks.
>
---
#### [new 104] Grids Often Outperform Implicit Neural Representations
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于计算机视觉与信号处理领域，比较了网格与隐式神经表示（INR）的性能，旨在理解其容量与适用场景。研究发现，多数情况下网格表现更优，仅在特定低维结构任务中INR占优。**

- **链接: [http://arxiv.org/pdf/2506.11139v1](http://arxiv.org/pdf/2506.11139v1)**

> **作者:** Namhoon Kim; Sara Fridovich-Keil
>
> **摘要:** Implicit Neural Representations (INRs) have recently shown impressive results, but their fundamental capacity, implicit biases, and scaling behavior remain poorly understood. We investigate the performance of diverse INRs across a suite of 2D and 3D real and synthetic signals with varying effective bandwidth, as well as both overfitting and generalization tasks including tomography, super-resolution, and denoising. By stratifying performance according to model size as well as signal type and bandwidth, our results shed light on how different INR and grid representations allocate their capacity. We find that, for most tasks and signals, a simple regularized grid with interpolation trains faster and to higher quality than any INR with the same number of parameters. We also find limited settings where INRs outperform grids -- namely fitting signals with underlying lower-dimensional structure such as shape contours -- to guide future use of INRs towards the most advantageous applications. Code and synthetic signals used in our analysis are available at https://github.com/voilalab/INR-benchmark.
>
---
#### [new 105] AutoGen Driven Multi Agent Framework for Iterative Crime Data Analysis and Prediction
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文属于犯罪数据分析任务，旨在实现自主、可扩展的犯罪趋势预测。通过多智能体框架协同分析数据，减少人工干预，提升分析效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11475v1](http://arxiv.org/pdf/2506.11475v1)**

> **作者:** Syeda Kisaa Fatima; Tehreem Zubair; Noman Ahmed; Asifullah Khan
>
> **摘要:** This paper introduces LUCID-MA (Learning and Understanding Crime through Dialogue of Multiple Agents), an innovative AI powered framework where multiple AI agents collaboratively analyze and understand crime data. Our system that consists of three core components: an analysis assistant that highlights spatiotemporal crime patterns, a feedback component that reviews and refines analytical results and a prediction component that forecasts future crime trends. With a well-designed prompt and the LLaMA-2-13B-Chat-GPTQ model, it runs completely offline and allows the agents undergo self-improvement through 100 rounds of communication with less human interaction. A scoring function is incorporated to evaluate agent's performance, providing visual plots to track learning progress. This work demonstrates the potential of AutoGen-style agents for autonomous, scalable, and iterative analysis in social science domains maintaining data privacy through offline execution.
>
---
#### [new 106] Visual Pre-Training on Unlabeled Images using Reinforcement Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文将无标签图像预训练视为强化学习问题，通过价值函数学习特征，提升模型表示能力。**

- **链接: [http://arxiv.org/pdf/2506.11967v1](http://arxiv.org/pdf/2506.11967v1)**

> **作者:** Dibya Ghosh; Sergey Levine
>
> **摘要:** In reinforcement learning (RL), value-based algorithms learn to associate each observation with the states and rewards that are likely to be reached from it. We observe that many self-supervised image pre-training methods bear similarity to this formulation: learning features that associate crops of images with those of nearby views, e.g., by taking a different crop or color augmentation. In this paper, we complete this analogy and explore a method that directly casts pre-training on unlabeled image data like web crawls and video frames as an RL problem. We train a general value function in a dynamical system where an agent transforms an image by changing the view or adding image augmentations. Learning in this way resembles crop-consistency self-supervision, but through the reward function, offers a simple lever to shape feature learning using curated images or weakly labeled captions when they exist. Our experiments demonstrate improved representations when training on unlabeled images in the wild, including video data like EpicKitchens, scene data like COCO, and web-crawl data like CC12M.
>
---
#### [new 107] VLM@school -- Evaluation of AI image understanding on German middle school knowledge
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态理解任务，旨在评估AI在德语中学校知识上的图像理解能力。研究构建了包含2000余题的基准数据集，测试模型结合视觉与学科知识的能力，发现现有模型表现不佳，揭示了现有基准与实际应用的差距。**

- **链接: [http://arxiv.org/pdf/2506.11604v1](http://arxiv.org/pdf/2506.11604v1)**

> **作者:** René Peinl; Vincent Tischler
>
> **摘要:** This paper introduces a novel benchmark dataset designed to evaluate the capabilities of Vision Language Models (VLMs) on tasks that combine visual reasoning with subject-specific background knowledge in the German language. In contrast to widely used English-language benchmarks that often rely on artificially difficult or decontextualized problems, this dataset draws from real middle school curricula across nine domains including mathematics, history, biology, and religion. The benchmark includes over 2,000 open-ended questions grounded in 486 images, ensuring that models must integrate visual interpretation with factual reasoning rather than rely on superficial textual cues. We evaluate thirteen state-of-the-art open-weight VLMs across multiple dimensions, including domain-specific accuracy and performance on adversarial crafted questions. Our findings reveal that even the strongest models achieve less than 45% overall accuracy, with particularly poor performance in music, mathematics, and adversarial settings. Furthermore, the results indicate significant discrepancies between success on popular benchmarks and real-world multimodal understanding. We conclude that middle school-level tasks offer a meaningful and underutilized avenue for stress-testing VLMs, especially in non-English contexts. The dataset and evaluation protocol serve as a rigorous testbed to better understand and improve the visual and linguistic reasoning capabilities of future AI systems.
>
---
#### [new 108] Taming Stable Diffusion for Computed Tomography Blind Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于CT图像超分辨率任务，旨在解决高分辨率成像与辐射安全的矛盾。通过适配Stable Diffusion模型，结合文本描述实现高质量重建。**

- **链接: [http://arxiv.org/pdf/2506.11496v1](http://arxiv.org/pdf/2506.11496v1)**

> **作者:** Chunlei Li; Yilei Shi; Haoxi Hu; Jingliang Hu; Xiao Xiang Zhu; Lichao Mou
>
> **摘要:** High-resolution computed tomography (CT) imaging is essential for medical diagnosis but requires increased radiation exposure, creating a critical trade-off between image quality and patient safety. While deep learning methods have shown promise in CT super-resolution, they face challenges with complex degradations and limited medical training data. Meanwhile, large-scale pre-trained diffusion models, particularly Stable Diffusion, have demonstrated remarkable capabilities in synthesizing fine details across various vision tasks. Motivated by this, we propose a novel framework that adapts Stable Diffusion for CT blind super-resolution. We employ a practical degradation model to synthesize realistic low-quality images and leverage a pre-trained vision-language model to generate corresponding descriptions. Subsequently, we perform super-resolution using Stable Diffusion with a specialized controlling strategy, conditioned on both low-resolution inputs and the generated text descriptions. Extensive experiments show that our method outperforms existing approaches, demonstrating its potential for achieving high-quality CT imaging at reduced radiation doses. Our code will be made publicly available.
>
---
#### [new 109] Brain Network Analysis Based on Fine-tuned Self-supervised Model for Brain Disease Diagnosis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于脑疾病诊断任务，旨在解决脑网络分析中模型泛化能力不足的问题。通过改进预训练模型，提升其在多维度上的表现。**

- **链接: [http://arxiv.org/pdf/2506.11671v1](http://arxiv.org/pdf/2506.11671v1)**

> **作者:** Yifei Tang; Hongjie Jiang; Changhong Jing; Hieu Pham; Shuqiang Wang
>
> **备注:** 13 pages, 3 figures, International Conference on Neural Computing for Advanced Applications
>
> **摘要:** Functional brain network analysis has become an indispensable tool for brain disease analysis. It is profoundly impacted by deep learning methods, which can characterize complex connections between ROIs. However, the research on foundation models of brain network is limited and constrained to a single dimension, which restricts their extensive application in neuroscience. In this study, we propose a fine-tuned brain network model for brain disease diagnosis. It expands brain region representations across multiple dimensions based on the original brain network model, thereby enhancing its generalizability. Our model consists of two key modules: (1)an adapter module that expands brain region features across different dimensions. (2)a fine-tuned foundation brain network model, based on self-supervised learning and pre-trained on fMRI data from thousands of participants. Specifically, its transformer block is able to effectively extract brain region features and compute the inter-region associations. Moreover, we derive a compact latent representation of the brain network for brain disease diagnosis. Our downstream experiments in this study demonstrate that the proposed model achieves superior performance in brain disease diagnosis, which potentially offers a promising approach in brain network analysis research.
>
---
#### [new 110] Voxel-Level Brain States Prediction Using Swin Transformer
- **分类: q-bio.NC; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于脑状态预测任务，旨在通过fMRI数据预测未来脑活动。采用Swin Transformer模型处理时空信息，实现高精度预测。**

- **链接: [http://arxiv.org/pdf/2506.11455v1](http://arxiv.org/pdf/2506.11455v1)**

> **作者:** Yifei Sun; Daniel Chahine; Qinghao Wen; Tianming Liu; Xiang Li; Yixuan Yuan; Fernando Calamante; Jinglei Lv
>
> **摘要:** Understanding brain dynamics is important for neuroscience and mental health. Functional magnetic resonance imaging (fMRI) enables the measurement of neural activities through blood-oxygen-level-dependent (BOLD) signals, which represent brain states. In this study, we aim to predict future human resting brain states with fMRI. Due to the 3D voxel-wise spatial organization and temporal dependencies of the fMRI data, we propose a novel architecture which employs a 4D Shifted Window (Swin) Transformer as encoder to efficiently learn spatio-temporal information and a convolutional decoder to enable brain state prediction at the same spatial and temporal resolution as the input fMRI data. We used 100 unrelated subjects from the Human Connectome Project (HCP) for model training and testing. Our novel model has shown high accuracy when predicting 7.2s resting-state brain activities based on the prior 23.04s fMRI time series. The predicted brain states highly resemble BOLD contrast and dynamics. This work shows promising evidence that the spatiotemporal organization of the human brain can be learned by a Swin Transformer model, at high resolution, which provides a potential for reducing the fMRI scan time and the development of brain-computer interfaces in the future.
>
---
#### [new 111] When Algorithms Play Favorites: Lookism in the Generation and Perception of Faces
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于AI公平性研究，探讨算法在生成和识别面孔时的外貌偏见问题。通过实验发现生成模型将吸引力与特质关联，分类模型对非白人女性错误率高。**

- **链接: [http://arxiv.org/pdf/2506.11025v1](http://arxiv.org/pdf/2506.11025v1)**

> **作者:** Miriam Doh; Aditya Gulati; Matei Mancas; Nuria Oliver
>
> **备注:** Accepted as an extended abstract at the Fourth European Workshop on Algorithmic Fairness (EWAF) (URL: https://2025.ewaf.org/home)
>
> **摘要:** This paper examines how synthetically generated faces and machine learning-based gender classification algorithms are affected by algorithmic lookism, the preferential treatment based on appearance. In experiments with 13,200 synthetically generated faces, we find that: (1) text-to-image (T2I) systems tend to associate facial attractiveness to unrelated positive traits like intelligence and trustworthiness; and (2) gender classification models exhibit higher error rates on "less-attractive" faces, especially among non-White women. These result raise fairness concerns regarding digital identity systems.
>
---
#### [new 112] RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决动态融合中模态偏好问题。通过引入RollingQ方法，平衡注意力分配，恢复协作动态。**

- **链接: [http://arxiv.org/pdf/2506.11465v1](http://arxiv.org/pdf/2506.11465v1)**

> **作者:** Haotian Ni; Yake Wei; Hang Liu; Gong Chen; Chong Peng; Hao Lin; Di Hu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Multimodal learning faces challenges in effectively fusing information from diverse modalities, especially when modality quality varies across samples. Dynamic fusion strategies, such as attention mechanism in Transformers, aim to address such challenge by adaptively emphasizing modalities based on the characteristics of input data. However, through amounts of carefully designed experiments, we surprisingly observed that the dynamic adaptability of widely-used self-attention models diminishes. Model tends to prefer one modality regardless of data characteristics. This bias triggers a self-reinforcing cycle that progressively overemphasizes the favored modality, widening the distribution gap in attention keys across modalities and deactivating attention mechanism's dynamic properties. To revive adaptability, we propose a simple yet effective method Rolling Query (RollingQ), which balances attention allocation by rotating the query to break the self-reinforcing cycle and mitigate the key distribution gap. Extensive experiments on various multimodal scenarios validate the effectiveness of RollingQ and the restoration of cooperation dynamics is pivotal for enhancing the broader capabilities of widely deployed multimodal Transformers. The source code is available at https://github.com/GeWu-Lab/RollingQ_ICML2025.
>
---
#### [new 113] Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Poutine模型，用于端到端自动驾驶任务，解决长尾场景下的驾驶问题。通过视觉-语言-轨迹预训练和强化学习微调，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11234v1](http://arxiv.org/pdf/2506.11234v1)**

> **作者:** Luke Rowe; Rodrigue de Schaetzen; Roger Girgis; Christopher Pal; Liam Paull
>
> **摘要:** We present Poutine, a 3B-parameter vision-language model (VLM) tailored for end-to-end autonomous driving in long-tail driving scenarios. Poutine is trained in two stages. To obtain strong base driving capabilities, we train Poutine-Base in a self-supervised vision-language-trajectory (VLT) next-token prediction fashion on 83 hours of CoVLA nominal driving and 11 hours of Waymo long-tail driving. Accompanying language annotations are auto-generated with a 72B-parameter VLM. Poutine is obtained by fine-tuning Poutine-Base with Group Relative Policy Optimization (GRPO) using less than 500 preference-labeled frames from the Waymo validation set. We show that both VLT pretraining and RL fine-tuning are critical to attain strong driving performance in the long-tail. Poutine-Base achieves a rater-feedback score (RFS) of 8.12 on the validation set, nearly matching Waymo's expert ground-truth RFS. The final Poutine model achieves an RFS of 7.99 on the official Waymo test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. These results highlight the promise of scalable VLT pre-training and lightweight RL fine-tuning to enable robust and generalizable autonomy.
>
---
#### [new 114] Anti-Aliased 2D Gaussian Splatting
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于三维重建任务，解决2D高斯点云渲染中的混叠问题。通过引入频域约束和新滤波方法提升不同尺度下的渲染质量。**

- **链接: [http://arxiv.org/pdf/2506.11252v1](http://arxiv.org/pdf/2506.11252v1)**

> **作者:** Mae Younes; Adnane Boukhayma
>
> **备注:** Code will be available at https://github.com/maeyounes/AA-2DGS
>
> **摘要:** 2D Gaussian Splatting (2DGS) has recently emerged as a promising method for novel view synthesis and surface reconstruction, offering better view-consistency and geometric accuracy than volumetric 3DGS. However, 2DGS suffers from severe aliasing artifacts when rendering at different sampling rates than those used during training, limiting its practical applications in scenarios requiring camera zoom or varying fields of view. We identify that these artifacts stem from two key limitations: the lack of frequency constraints in the representation and an ineffective screen-space clamping approach. To address these issues, we present AA-2DGS, an antialiased formulation of 2D Gaussian Splatting that maintains its geometric benefits while significantly enhancing rendering quality across different scales. Our method introduces a world space flat smoothing kernel that constrains the frequency content of 2D Gaussian primitives based on the maximal sampling frequency from training views, effectively eliminating high-frequency artifacts when zooming in. Additionally, we derive a novel object space Mip filter by leveraging an affine approximation of the ray-splat intersection mapping, which allows us to efficiently apply proper anti-aliasing directly in the local space of each splat.
>
---
#### [new 115] Solving Inverse Problems in Stochastic Self-Organising Systems through Invariant Representations
- **分类: nlin.AO; cond-mat.dis-nn; cs.CV; cs.LG**

- **简介: 该论文属于逆问题求解任务，旨在从随机模式中恢复因果参数。通过构建不变表示空间，有效处理观测中的随机性，无需人工目标函数。**

- **链接: [http://arxiv.org/pdf/2506.11796v1](http://arxiv.org/pdf/2506.11796v1)**

> **作者:** Elias Najarro; Nicolas Bessone; Sebastian Risi
>
> **备注:** Preprint. Under review
>
> **摘要:** Self-organising systems demonstrate how simple local rules can generate complex stochastic patterns. Many natural systems rely on such dynamics, making self-organisation central to understanding natural complexity. A fundamental challenge in modelling such systems is solving the inverse problem: finding the unknown causal parameters from macroscopic observations. This task becomes particularly difficult when observations have a strong stochastic component, yielding diverse yet equivalent patterns. Traditional inverse methods fail in this setting, as pixel-wise metrics cannot capture feature similarities between variable outcomes. In this work, we introduce a novel inverse modelling method specifically designed to handle stochasticity in the observable space, leveraging the capacity of visual embeddings to produce robust representations that capture perceptual invariances. By mapping the pattern representations onto an invariant embedding space, we can effectively recover unknown causal parameters without the need for handcrafted objective functions or heuristics. We evaluate the method on two canonical models--a reaction-diffusion system and an agent-based model of social segregation--and show that it reliably recovers parameters despite stochasticity in the outcomes. We further apply the method to real biological patterns, highlighting its potential as a tool for both theorists and experimentalists to investigate the dynamics underlying complex stochastic pattern formation.
>
---
#### [new 116] Control Architecture and Design for a Multi-robotic Visual Servoing System in Automated Manufacturing Environment
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; 93C85 (Primary), 93B52 (Secondary)**

- **简介: 该论文属于多机器人视觉伺服控制任务，旨在解决制造环境中的不确定性问题，提出多机器人控制架构和相机位置优化算法以提高精度。**

- **链接: [http://arxiv.org/pdf/2506.11387v1](http://arxiv.org/pdf/2506.11387v1)**

> **作者:** Rongfei Li
>
> **备注:** 272 pages, 171 figures, PhD dissertation, University of California, Davis, 2025. To be published in ProQuest ETD
>
> **摘要:** The use of robotic technology has drastically increased in manufacturing in the 21st century. But by utilizing their sensory cues, humans still outperform machines, especially in micro scale manufacturing, which requires high-precision robot manipulators. These sensory cues naturally compensate for high levels of uncertainties that exist in the manufacturing environment. Uncertainties in performing manufacturing tasks may come from measurement noise, model inaccuracy, joint compliance (e.g., elasticity), etc. Although advanced metrology sensors and high precision microprocessors, which are utilized in modern robots, have compensated for many structural and dynamic errors in robot positioning, a well-designed control algorithm still works as a comparable and cheaper alternative to reduce uncertainties in automated manufacturing. Our work illustrates that a multi-robot control system that simulates the positioning process for fastening and unfastening applications can reduce various uncertainties, which may occur in this process, to a great extent. In addition, most research papers in visual servoing mainly focus on developing control and observation architectures in various scenarios, but few have discussed the importance of the camera's location in the configuration. In a manufacturing environment, the quality of camera estimations may vary significantly from one observation location to another, as the combined effects of environmental conditions result in different noise levels of a single image shot at different locations. Therefore, in this paper, we also propose a novel algorithm for the camera's moving policy so that it explores the camera workspace and searches for the optimal location where the image noise level is minimized.
>
---
#### [new 117] GaussMarker: Robust Dual-Domain Watermark for Diffusion Models
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于图像水印任务，旨在解决扩散模型生成图像的版权保护问题。提出GaussMarker，在空间和频域同时嵌入水印，并引入噪声恢复模块提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.11444v1](http://arxiv.org/pdf/2506.11444v1)**

> **作者:** Kecen Li; Zhicong Huang; Xinwen Hou; Cheng Hong
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** As Diffusion Models (DM) generate increasingly realistic images, related issues such as copyright and misuse have become a growing concern. Watermarking is one of the promising solutions. Existing methods inject the watermark into the single-domain of initial Gaussian noise for generation, which suffers from unsatisfactory robustness. This paper presents the first dual-domain DM watermarking approach using a pipelined injector to consistently embed watermarks in both the spatial and frequency domains. To further boost robustness against certain image manipulations and advanced attacks, we introduce a model-independent learnable Gaussian Noise Restorer (GNR) to refine Gaussian noise extracted from manipulated images and enhance detection robustness by integrating the detection scores of both watermarks. GaussMarker efficiently achieves state-of-the-art performance under eight image distortions and four advanced attacks across three versions of Stable Diffusion with better recall and lower false positive rates, as preferred in real applications.
>
---
#### [new 118] HQFNN: A Compact Quantum-Fuzzy Neural Network for Accurate Image Classification
- **分类: quant-ph; cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决噪声敏感和模型可解释性问题。提出HQFNN结合量子计算与模糊逻辑，实现高效、鲁棒的图像分类。**

- **链接: [http://arxiv.org/pdf/2506.11146v1](http://arxiv.org/pdf/2506.11146v1)**

> **作者:** Jianhong Yao; Yangming Guo
>
> **摘要:** Deep learning vision systems excel at pattern recognition yet falter when inputs are noisy or the model must explain its own confidence. Fuzzy inference, with its graded memberships and rule transparency, offers a remedy, while parameterized quantum circuits can embed features in richly entangled Hilbert spaces with striking parameter efficiency. Bridging these ideas, this study introduces a innovative Highly Quantized Fuzzy Neural Network (HQFNN) that realises the entire fuzzy pipeline inside a shallow quantum circuit and couples the resulting quantum signal to a lightweight CNN feature extractor. Each image feature is first mapped to a single qubit membership state through repeated angle reuploading. Then a compact rule layer refines these amplitudes, and a clustered CNOT defuzzifier collapses them into one crisp value that is fused with classical features before classification. Evaluated on standard image benchmarks, HQFNN consistently surpasses classical, fuzzy enhanced and quantum only baselines while using several orders of magnitude fewer trainable weights, and its accuracy degrades only marginally under simulated depolarizing and amplitude damping noise, evidence of intrinsic robustness. Gate count analysis further shows that circuit depth grows sublinearly with input dimension, confirming the model's practicality for larger images. These results position the model as a compact, interpretable and noise tolerant alternative to conventional vision backbones and provide a template for future quantum native fuzzy learning frameworks.
>
---
#### [new 119] Developing a Dyslexia Indicator Using Eye Tracking
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于诊断任务，旨在通过眼动追踪和机器学习早期检测阅读障碍。研究分析眼动模式，使用随机森林分类器实现高精度识别。**

- **链接: [http://arxiv.org/pdf/2506.11004v1](http://arxiv.org/pdf/2506.11004v1)**

> **作者:** Kevin Cogan; Vuong M. Ngo; Mark Roantree
>
> **备注:** The 23rd International Conference on Artificial Intelligence in Medicine (AIME 2025), LNAI, Springer, 11 pages
>
> **摘要:** Dyslexia, affecting an estimated 10% to 20% of the global population, significantly impairs learning capabilities, highlighting the need for innovative and accessible diagnostic methods. This paper investigates the effectiveness of eye-tracking technology combined with machine learning algorithms as a cost-effective alternative for early dyslexia detection. By analyzing general eye movement patterns, including prolonged fixation durations and erratic saccades, we proposed an enhanced solution for determining eye-tracking-based dyslexia features. A Random Forest Classifier was then employed to detect dyslexia, achieving an accuracy of 88.58\%. Additionally, hierarchical clustering methods were applied to identify varying severity levels of dyslexia. The analysis incorporates diverse methodologies across various populations and settings, demonstrating the potential of this technology to identify individuals with dyslexia, including those with borderline traits, through non-invasive means. Integrating eye-tracking with machine learning represents a significant advancement in the diagnostic process, offering a highly accurate and accessible method in clinical research.
>
---
#### [new 120] Framework of a multiscale data-driven digital twin of the muscle-skeletal system
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医疗健康领域，旨在解决肌肉骨骼疾病诊断与治疗问题。提出一种多尺度数字孪生框架，整合多种数据进行个性化建模与可视化分析。**

- **链接: [http://arxiv.org/pdf/2506.11821v1](http://arxiv.org/pdf/2506.11821v1)**

> **作者:** Martina Paccini; Simone Cammarasana; Giuseppe Patanè
>
> **摘要:** Musculoskeletal disorders (MSDs) are a leading cause of disability worldwide, requiring advanced diagnostic and therapeutic tools for personalised assessment and treatment. Effective management of MSDs involves the interaction of heterogeneous data sources, making the Digital Twin (DT) paradigm a valuable option. This paper introduces the Musculoskeletal Digital Twin (MS-DT), a novel framework that integrates multiscale biomechanical data with computational modelling to create a detailed, patient-specific representation of the musculoskeletal system. By combining motion capture, ultrasound imaging, electromyography, and medical imaging, the MS-DT enables the analysis of spinal kinematics, posture, and muscle function. An interactive visualisation platform provides clinicians and researchers with an intuitive interface for exploring biomechanical parameters and tracking patient-specific changes. Results demonstrate the effectiveness of MS-DT in extracting precise kinematic and dynamic tissue features, offering a comprehensive tool for monitoring spine biomechanics and rehabilitation. This framework provides high-fidelity modelling and real-time visualization to improve patient-specific diagnosis and intervention planning.
>
---
#### [new 121] crossMoDA Challenge: Evolution of Cross-Modality Domain Adaptation Techniques for Vestibular Schwannoma and Cochlea Segmentation from 2021 to 2023
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于跨模态域适应任务，旨在提升听神经瘤和耳蜗在T2 MRI上的自动分割。通过分析2021至2023年挑战赛进展，探索数据多样性对分割性能的影响。**

- **链接: [http://arxiv.org/pdf/2506.12006v1](http://arxiv.org/pdf/2506.12006v1)**

> **作者:** Navodini Wijethilake; Reuben Dorent; Marina Ivory; Aaron Kujawa; Stefan Cornelissen; Patrick Langenhuizen; Mohamed Okasha; Anna Oviedova; Hexin Dong; Bogyeong Kang; Guillaume Sallé; Luyi Han; Ziyuan Zhao; Han Liu; Tao Yang; Shahad Hardan; Hussain Alasmawi; Santosh Sanjeev; Yuzhou Zhuang; Satoshi Kondo; Maria Baldeon Calisto; Shaikh Muhammad Uzair Noman; Cancan Chen; Ipek Oguz; Rongguo Zhang; Mina Rezaei; Susana K. Lai-Yuen; Satoshi Kasai; Chih-Cheng Hung; Mohammad Yaqub; Lisheng Wang; Benoit M. Dawant; Cuntai Guan; Ritse Mann; Vincent Jaouen; Ji-Wung Han; Li Zhang; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** The cross-Modality Domain Adaptation (crossMoDA) challenge series, initiated in 2021 in conjunction with the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), focuses on unsupervised cross-modality segmentation, learning from contrast-enhanced T1 (ceT1) and transferring to T2 MRI. The task is an extreme example of domain shift chosen to serve as a meaningful and illustrative benchmark. From a clinical application perspective, it aims to automate Vestibular Schwannoma (VS) and cochlea segmentation on T2 scans for more cost-effective VS management. Over time, the challenge objectives have evolved to enhance its clinical relevance. The challenge evolved from using single-institutional data and basic segmentation in 2021 to incorporating multi-institutional data and Koos grading in 2022, and by 2023, it included heterogeneous routine data and sub-segmentation of intra- and extra-meatal tumour components. In this work, we report the findings of the 2022 and 2023 editions and perform a retrospective analysis of the challenge progression over the years. The observations from the successive challenge contributions indicate that the number of outliers decreases with an expanding dataset. This is notable since the diversity of scanning protocols of the datasets concurrently increased. The winning approach of the 2023 edition reduced the number of outliers on the 2021 and 2022 testing data, demonstrating how increased data heterogeneity can enhance segmentation performance even on homogeneous data. However, the cochlea Dice score declined in 2023, likely due to the added complexity from tumour sub-annotations affecting overall segmentation performance. While progress is still needed for clinically acceptable VS segmentation, the plateauing performance suggests that a more challenging cross-modal task may better serve future benchmarking.
>
---
#### [new 122] Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; 68; I.2.0; I.2.4; I.2.6; I.2.7; I.4.7; I.4.10; I.5.1; F.1.1**

- **简介: 该论文属于深度学习任务，旨在解决传统模型与人类心理相似性不一致的问题。通过引入可微分的Tversky相似性，设计新型神经网络层，提升模型性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.11035v1](http://arxiv.org/pdf/2506.11035v1)**

> **作者:** Moussa Koulako Bala Doumbouya; Dan Jurafsky; Christopher D. Manning
>
> **摘要:** Work in psychology has highlighted that the geometric model of similarity standard in deep learning is not psychologically plausible because its metric properties such as symmetry do not align with human perception. In contrast, Tversky (1977) proposed an axiomatic theory of similarity based on a representation of objects as sets of features, and their similarity as a function of common and distinctive features. However, this model has not been used in deep learning before, partly due to the challenge of incorporating discrete set operations. We develop a differentiable parameterization of Tversky's similarity that is learnable through gradient descent, and derive neural network building blocks such as the Tversky projection layer, which unlike the linear projection layer can model non-linear functions such as XOR. Through experiments with image recognition and language modeling, we show that the Tversky projection layer is a beneficial replacement for the linear projection layer, which employs geometric similarity. On the NABirds image classification task, a frozen ResNet-50 adapted with a Tversky projection layer achieves a 24.7% relative accuracy improvement over the linear layer adapter baseline. With Tversky projection layers, GPT-2's perplexity on PTB decreases by 7.5%, and its parameter count by 34.8%. Finally, we propose a unified interpretation of both projection layers as computing similarities of input stimuli to learned prototypes, for which we also propose a novel visualization technique highlighting the interpretability of Tversky projection layers. Our work offers a new paradigm for thinking about the similarity model implicit in deep learning, and designing networks that are interpretable under an established theory of psychological similarity.
>
---
#### [new 123] MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser
- **分类: eess.IV; cs.AI; cs.CV; cs.NE**

- **简介: 该论文提出MindGrab，用于快速准确的脑图像去骨处理，解决传统方法参数多、效率低的问题。**

- **链接: [http://arxiv.org/pdf/2506.11860v1](http://arxiv.org/pdf/2506.11860v1)**

> **作者:** Armina Fani; Mike Doan; Isabelle Le; Alex Fedorov; Malte Hoffmann; Chris Rorden; Sergey Plis
>
> **备注:** 12 pages, 1 table, 4 figures. 2 supplementary tables, 1 supplementary figure. Brainchop-cli: https://pypi.org/project/brainchop/ . Brainchop web: https://brainchop.org/
>
> **摘要:** We developed MindGrab, a parameter- and memory-efficient deep fully-convolutional model for volumetric skull-stripping in head images of any modality. Its architecture, informed by a spectral interpretation of dilated convolutions, was trained exclusively on modality-agnostic synthetic data. MindGrab was evaluated on a retrospective dataset of 606 multimodal adult-brain scans (T1, T2, DWI, MRA, PDw MRI, EPI, CT, PET) sourced from the SynthStrip dataset. Performance was benchmarked against SynthStrip, ROBEX, and BET using Dice scores, with Wilcoxon signed-rank significance tests. MindGrab achieved a mean Dice score of 95.9 with standard deviation (SD) 1.6 across modalities, significantly outperforming classical methods (ROBEX: 89.1 SD 7.7, P < 0.05; BET: 85.2 SD 14.4, P < 0.05). Compared to SynthStrip (96.5 SD 1.1, P=0.0352), MindGrab delivered equivalent or superior performance in nearly half of the tested scenarios, with minor differences (<3% Dice) in the others. MindGrab utilized 95% fewer parameters (146,237 vs. 2,566,561) than SynthStrip. This efficiency yielded at least 2x faster inference, 50% lower memory usage on GPUs, and enabled exceptional performance (e.g., 10-30x speedup, and up to 30x memory reduction) and accessibility on a wider range of hardware, including systems without high-end GPUs. MindGrab delivers state-of-the-art accuracy with dramatically lower resource demands, supported in brainchop-cli (https://pypi.org/project/brainchop/) and at brainchop.org.
>
---
#### [new 124] Exploring the Effectiveness of Deep Features from Domain-Specific Foundation Models in Retinal Image Synthesis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决医疗影像数据不足和隐私问题。研究比较了不同损失函数对视网膜图像生成的效果，发现传统边缘检测更有效。**

- **链接: [http://arxiv.org/pdf/2506.11753v1](http://arxiv.org/pdf/2506.11753v1)**

> **作者:** Zuzanna Skorniewska; Bartlomiej W. Papiez
>
> **备注:** To be published and presented at the MIUA 2025 conference
>
> **摘要:** The adoption of neural network models in medical imaging has been constrained by strict privacy regulations, limited data availability, high acquisition costs, and demographic biases. Deep generative models offer a promising solution by generating synthetic data that bypasses privacy concerns and addresses fairness by producing samples for under-represented groups. However, unlike natural images, medical imaging requires validation not only for fidelity (e.g., Fr\'echet Inception Score) but also for morphological and clinical accuracy. This is particularly true for colour fundus retinal imaging, which requires precise replication of the retinal vascular network, including vessel topology, continuity, and thickness. In this study, we in-vestigated whether a distance-based loss function based on deep activation layers of a large foundational model trained on large corpus of domain data, colour fundus imaging, offers advantages over a perceptual loss and edge-detection based loss functions. Our extensive validation pipeline, based on both domain-free and domain specific tasks, suggests that domain-specific deep features do not improve autoen-coder image generation. Conversely, our findings highlight the effectiveness of con-ventional edge detection filters in improving the sharpness of vascular structures in synthetic samples.
>
---
#### [new 125] Vector Representations of Vessel Trees
- **分类: eess.IV; cs.CV; cs.GR**

- **简介: 该论文属于医学图像中的树状结构建模任务，旨在解决血管网络的拓扑保持与高效表示问题。通过两阶段Transformer自编码器，学习精确的向量表示。**

- **链接: [http://arxiv.org/pdf/2506.11163v1](http://arxiv.org/pdf/2506.11163v1)**

> **作者:** James Batten; Michiel Schaap; Matthew Sinclair; Ying Bai; Ben Glocker
>
> **摘要:** We introduce a novel framework for learning vector representations of tree-structured geometric data focusing on 3D vascular networks. Our approach employs two sequentially trained Transformer-based autoencoders. In the first stage, the Vessel Autoencoder captures continuous geometric details of individual vessel segments by learning embeddings from sampled points along each curve. In the second stage, the Vessel Tree Autoencoder encodes the topology of the vascular network as a single vector representation, leveraging the segment-level embeddings from the first model. A recursive decoding process ensures that the reconstructed topology is a valid tree structure. Compared to 3D convolutional models, this proposed approach substantially lowers GPU memory requirements, facilitating large-scale training. Experimental results on a 2D synthetic tree dataset and a 3D coronary artery dataset demonstrate superior reconstruction fidelity, accurate topology preservation, and realistic interpolations in latent space. Our scalable framework, named VeTTA, offers precise, flexible, and topologically consistent modeling of anatomical tree structures in medical imaging.
>
---
#### [new 126] Real-World Deployment of a Lane Change Prediction Architecture Based on Knowledge Graph Embeddings and Bayesian Inference
- **分类: cs.AR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于车道变换预测任务，旨在解决算法与实际部署间的差距。通过结合知识图谱嵌入和贝叶斯推理，实现提前预测并采取安全制动措施。**

- **链接: [http://arxiv.org/pdf/2506.11925v1](http://arxiv.org/pdf/2506.11925v1)**

> **作者:** M. Manzour; Catherine M. Elias; Omar M. Shehata; R. Izquierdo; M. A. Sotelo
>
> **摘要:** Research on lane change prediction has gained a lot of momentum in the last couple of years. However, most research is confined to simulation or results obtained from datasets, leaving a gap between algorithmic advances and on-road deployment. This work closes that gap by demonstrating, on real hardware, a lane-change prediction system based on Knowledge Graph Embeddings (KGEs) and Bayesian inference. Moreover, the ego-vehicle employs a longitudinal braking action to ensure the safety of both itself and the surrounding vehicles. Our architecture consists of two modules: (i) a perception module that senses the environment, derives input numerical features, and converts them into linguistic categories; and communicates them to the prediction module; (ii) a pretrained prediction module that executes a KGE and Bayesian inference model to anticipate the target vehicle's maneuver and transforms the prediction into longitudinal braking action. Real-world hardware experimental validation demonstrates that our prediction system anticipates the target vehicle's lane change three to four seconds in advance, providing the ego vehicle sufficient time to react and allowing the target vehicle to make the lane change safely.
>
---
## 更新

#### [replaced 001] SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09565v2](http://arxiv.org/pdf/2506.09565v2)**

> **作者:** Qijing Li; Jingxiang Sun; Liang An; Zhaoqi Su; Hongwen Zhang; Yebin Liu
>
> **摘要:** Holistic 3D scene understanding, which jointly models geometry, appearance, and semantics, is crucial for applications like augmented reality and robotic interaction. Existing feed-forward 3D scene understanding methods (e.g., LSM) are limited to extracting language-based semantics from scenes, failing to achieve holistic scene comprehension. Additionally, they suffer from low-quality geometry reconstruction and noisy artifacts. In contrast, per-scene optimization methods rely on dense input views, which reduces practicality and increases complexity during deployment. In this paper, we propose SemanticSplat, a feed-forward semantic-aware 3D reconstruction method, which unifies 3D Gaussians with latent semantic attributes for joint geometry-appearance-semantics modeling. To predict the semantic anisotropic Gaussians, SemanticSplat fuses diverse feature fields (e.g., LSeg, SAM) with a cost volume representation that stores cross-view feature similarities, enhancing coherent and accurate scene comprehension. Leveraging a two-stage distillation framework, SemanticSplat reconstructs a holistic multi-modal semantic feature field from sparse-view images. Experiments demonstrate the effectiveness of our method for 3D scene understanding tasks like promptable and open-vocabulary segmentation. Video results are available at https://semanticsplat.github.io.
>
---
#### [replaced 002] IQE-CLIP: Instance-aware Query Embedding for Zero-/Few-shot Anomaly Detection in Medical Domain
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10730v2](http://arxiv.org/pdf/2506.10730v2)**

> **作者:** Hong Huang; Weixiang Sun; Zhijian Wu; Jingwen Niu; Donghuan Lu; Xian Wu; Yefeng Zheng
>
> **摘要:** Recently, the rapid advancements of vision-language models, such as CLIP, leads to significant progress in zero-/few-shot anomaly detection (ZFSAD) tasks. However, most existing CLIP-based ZFSAD methods commonly assume prior knowledge of categories and rely on carefully crafted prompts tailored to specific scenarios. While such meticulously designed text prompts effectively capture semantic information in the textual space, they fall short of distinguishing normal and anomalous instances within the joint embedding space. Moreover, these ZFSAD methods are predominantly explored in industrial scenarios, with few efforts conducted to medical tasks. To this end, we propose an innovative framework for ZFSAD tasks in medical domain, denoted as IQE-CLIP. We reveal that query embeddings, which incorporate both textual and instance-aware visual information, are better indicators for abnormalities. Specifically, we first introduce class-based prompting tokens and learnable prompting tokens for better adaptation of CLIP to the medical domain. Then, we design an instance-aware query module (IQM) to extract region-level contextual information from both text prompts and visual features, enabling the generation of query embeddings that are more sensitive to anomalies. Extensive experiments conducted on six medical datasets demonstrate that IQE-CLIP achieves state-of-the-art performance on both zero-shot and few-shot tasks. We release our code and data at https://github.com/hongh0/IQE-CLIP/.
>
---
#### [replaced 003] SG2VID: Scene Graphs Enable Fine-Grained Control for Video Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03082v2](http://arxiv.org/pdf/2506.03082v2)**

> **作者:** Ssharvien Kumar Sivakumar; Yannik Frisch; Ghazal Ghazaei; Anirban Mukhopadhyay
>
> **摘要:** Surgical simulation plays a pivotal role in training novice surgeons, accelerating their learning curve and reducing intra-operative errors. However, conventional simulation tools fall short in providing the necessary photorealism and the variability of human anatomy. In response, current methods are shifting towards generative model-based simulators. Yet, these approaches primarily focus on using increasingly complex conditioning for precise synthesis while neglecting the fine-grained human control aspect. To address this gap, we introduce SG2VID, the first diffusion-based video model that leverages Scene Graphs for both precise video synthesis and fine-grained human control. We demonstrate SG2VID's capabilities across three public datasets featuring cataract and cholecystectomy surgery. While SG2VID outperforms previous methods both qualitatively and quantitatively, it also enables precise synthesis, providing accurate control over tool and anatomy's size and movement, entrance of new tools, as well as the overall scene layout. We qualitatively motivate how SG2VID can be used for generative augmentation and present an experiment demonstrating its ability to improve a downstream phase detection task when the training set is extended with our synthetic videos. Finally, to showcase SG2VID's ability to retain human control, we interact with the Scene Graphs to generate new video samples depicting major yet rare intra-operative irregularities.
>
---
#### [replaced 004] Clustering is back: Reaching state-of-the-art LiDAR instance segmentation without training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13203v2](http://arxiv.org/pdf/2503.13203v2)**

> **作者:** Corentin Sautier; Gilles Puy; Alexandre Boulch; Renaud Marlet; Vincent Lepetit
>
> **备注:** Alpine ranks first in the leaderboard of SemanticKITTI's panoptic segmentation
>
> **摘要:** Panoptic segmentation of LiDAR point clouds is fundamental to outdoor scene understanding, with autonomous driving being a primary application. While state-of-the-art approaches typically rely on end-to-end deep learning architectures and extensive manual annotations of instances, the significant cost and time investment required for labeling large-scale point cloud datasets remains a major bottleneck in this field. In this work, we demonstrate that competitive panoptic segmentation can be achieved using only semantic labels, with instances predicted without any training or annotations. Our method outperforms state-of-the-art supervised methods on standard benchmarks including SemanticKITTI and nuScenes, and outperforms every publicly available method on SemanticKITTI as a drop-in instance head replacement, while running in real-time on a single-threaded CPU and requiring no instance labels. It is fully explainable, and requires no learning or parameter tuning. Alpine combined with state-of-the-art semantic segmentation ranks first on the official panoptic segmentation leaderboard of SemanticKITTI. Code is available at https://github.com/valeoai/Alpine/
>
---
#### [replaced 005] New Dataset and Methods for Fine-Grained Compositional Referring Expression Comprehension via Specialist-MLLM Collaboration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20104v3](http://arxiv.org/pdf/2502.20104v3)**

> **作者:** Xuzheng Yang; Junzhuo Liu; Peng Wang; Guoqing Wang; Yang Yang; Heng Tao Shen
>
> **备注:** Accepted by TPAMI 2025
>
> **摘要:** Referring Expression Comprehension (REC) is a foundational cross-modal task that evaluates the interplay of language understanding, image comprehension, and language-to-image grounding. It serves as an essential testing ground for Multimodal Large Language Models (MLLMs). To advance this field, we introduced a new REC dataset in our previous conference paper, characterized by two key features. First, it is designed with controllable difficulty levels, requiring multi-level fine-grained reasoning across object categories, attributes, and multi-hop relationships. Second, it incorporates negative text and images generated through fine-grained editing and augmentation, explicitly testing a model's ability to reject scenarios where the target object is absent, an often overlooked yet critical challenge in existing datasets. In this extended work, we propose two new methods to tackle the challenges of fine-grained REC by combining the strengths of Specialist Models and MLLMs. The first method adaptively assigns simple cases to faster, lightweight models and reserves complex ones for powerful MLLMs, balancing accuracy and efficiency. The second method lets a specialist generate a set of possible object regions, and the MLLM selects the most plausible one using its reasoning ability. These collaborative strategies lead to significant improvements on our dataset and other challenging benchmarks. Our results show that combining specialized and general-purpose models offers a practical path toward solving complex real-world vision-language tasks. Our dataset and code are available at https://github.com/sleepyshep/FineCops-Ref.
>
---
#### [replaced 006] TextCrafter: Accurately Rendering Multiple Texts in Complex Visual Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23461v3](http://arxiv.org/pdf/2503.23461v3)**

> **作者:** Nikai Du; Zhennan Chen; Zhizhou Chen; Shan Gao; Xi Chen; Zhengkai Jiang; Jian Yang; Ying Tai
>
> **摘要:** This paper explores the task of Complex Visual Text Generation (CVTG), which centers on generating intricate textual content distributed across diverse regions within visual images. In CVTG, image generation models often rendering distorted and blurred visual text or missing some visual text. To tackle these challenges, we propose TextCrafter, a novel multi-visual text rendering method. TextCrafter employs a progressive strategy to decompose complex visual text into distinct components while ensuring robust alignment between textual content and its visual carrier. Additionally, it incorporates a token focus enhancement mechanism to amplify the prominence of visual text during the generation process. TextCrafter effectively addresses key challenges in CVTG tasks, such as text confusion, omissions, and blurriness. Moreover, we present a new benchmark dataset, CVTG-2K, tailored to rigorously evaluate the performance of generative models on CVTG tasks. Extensive experiments demonstrate that our method surpasses state-of-the-art approaches.
>
---
#### [replaced 007] Fine-tune Smarter, Not Harder: Parameter-Efficient Fine-Tuning for Geospatial Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17397v2](http://arxiv.org/pdf/2504.17397v2)**

> **作者:** Francesc Marti-Escofet; Benedikt Blumenstiel; Linus Scheibenreif; Paolo Fraccaro; Konrad Schindler
>
> **备注:** Code available at https://github.com/IBM/peft-geofm
>
> **摘要:** Earth observation (EO) is crucial for monitoring environmental changes, responding to disasters, and managing natural resources. In this context, foundation models facilitate remote sensing image analysis to retrieve relevant geoinformation accurately and efficiently. However, as these models grow in size, fine-tuning becomes increasingly challenging due to the associated computational resources and costs, limiting their accessibility and scalability. Furthermore, full fine-tuning can lead to forgetting pre-trained features and even degrade model generalization. To address this, Parameter-Efficient Fine-Tuning (PEFT) techniques offer a promising solution. In this paper, we conduct extensive experiments with various foundation model architectures and PEFT techniques to evaluate their effectiveness on five different EO datasets. Our results provide a comprehensive comparison, offering insights into when and how PEFT methods support the adaptation of pre-trained geospatial models. We demonstrate that PEFT techniques match or even exceed full fine-tuning performance and enhance model generalisation to unseen geographic regions, while reducing training time and memory requirements. Additional experiments investigate the effect of architecture choices such as the decoder type or the use of metadata, suggesting UNet decoders and fine-tuning without metadata as the recommended configuration. We have integrated all evaluated foundation models and techniques into the open-source package TerraTorch to support quick, scalable, and cost-effective model adaptation.
>
---
#### [replaced 008] Self-supervised training of deep denoisers in multi-coil MRI considering noise correlations
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2208.07552v3](http://arxiv.org/pdf/2208.07552v3)**

> **作者:** Juhyung Park; Dongwon Park; Sooyeon Ji; Hyeong-Geol Shin; Se Young Chun; Jongho Lee
>
> **备注:** 9 pages, 5figures
>
> **摘要:** Deep learning-based denoising methods have shown powerful results for improving the signal-to-noise ratio of magnetic resonance (MR) images, mostly by leveraging supervised learning with clean ground truth. However, acquiring clean ground truth images is often expensive and time-consuming. Self supervised methods have been widely investigated to mitigate the dependency on clean images, but mostly rely on the suboptimal splitting of K-space measurements of an image to yield input and target images for ensuring statistical independence. In this study, we investigate an alternative self-supervised training method for deep denoisers in multi-coil MRI, dubbed Coil2Coil (C2C), that naturally split and combine the multi-coil data among phased array coils, generating two noise-corrupted images for training. This novel approach allows exploiting multi-coil redundancy, but the images are statistically correlated and may not have the same clean image. To mitigate these issues, we propose the methods to pproximately decorrelate the statistical dependence of these images and match the underlying clean images, thus enabling them to be used as the training pairs. For synthetic denoising experiments, C2C yielded the best performance against prior self-supervised methods, reporting outcome comparable even to supervised methods. For real-world denoising cases, C2C yielded consistent performance as synthetic cases, removing only noise structures.
>
---
#### [replaced 009] Manipulating Feature Visualizations with Gradient Slingshots
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.06122v3](http://arxiv.org/pdf/2401.06122v3)**

> **作者:** Dilyara Bareeva; Marina M. -C. Höhne; Alexander Warnecke; Lukas Pirch; Klaus-Robert Müller; Konrad Rieck; Sebastian Lapuschkin; Kirill Bykov
>
> **摘要:** Feature Visualization (FV) is a widely used technique for interpreting the concepts learned by Deep Neural Networks (DNNs), which synthesizes input patterns that maximally activate a given feature. Despite its popularity, the trustworthiness of FV explanations has received limited attention. In this paper, we introduce a novel method, Gradient Slingshots, that enables manipulation of FV without modifying the model architecture or significantly degrading its performance. By shaping new trajectories in the off-distribution regions of the activation landscape of a feature, we coerce the optimization process to converge in a predefined visualization. We evaluate our approach on several DNN architectures, demonstrating its ability to replace faithfuls FV with arbitrary targets. These results expose a critical vulnerability: auditors relying solely on FV may accept entirely fabricated explanations. To mitigate this risk, we propose a straightforward defense and quantitatively demonstrate its effectiveness.
>
---
#### [replaced 010] Consistent Video Editing as Flow-Driven Image-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07713v2](http://arxiv.org/pdf/2506.07713v2)**

> **作者:** Ge Wang; Songlin Fan; Hangxu Liu; Quanjian Song; Hewei Wang; Jinfeng Xu
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** With the prosper of video diffusion models, down-stream applications like video editing have been significantly promoted without consuming much computational cost. One particular challenge in this task lies at the motion transfer process from the source video to the edited one, where it requires the consideration of the shape deformation in between, meanwhile maintaining the temporal consistency in the generated video sequence. However, existing methods fail to model complicated motion patterns for video editing, and are fundamentally limited to object replacement, where tasks with non-rigid object motions like multi-object and portrait editing are largely neglected. In this paper, we observe that optical flows offer a promising alternative in complex motion modeling, and present FlowV2V to re-investigate video editing as a task of flow-driven Image-to-Video (I2V) generation. Specifically, FlowV2V decomposes the entire pipeline into first-frame editing and conditional I2V generation, and simulates pseudo flow sequence that aligns with the deformed shape, thus ensuring the consistency during editing. Experimental results on DAVIS-EDIT with improvements of 13.67% and 50.66% on DOVER and warping error illustrate the superior temporal consistency and sample quality of FlowV2V compared to existing state-of-the-art ones. Furthermore, we conduct comprehensive ablation studies to analyze the internal functionalities of the first-frame paradigm and flow alignment in the proposed method.
>
---
#### [replaced 011] Beyond the Visible: Multispectral Vision-Language Learning for Earth Observation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15969v2](http://arxiv.org/pdf/2503.15969v2)**

> **作者:** Clive Tinashe Marimo; Benedikt Blumenstiel; Maximilian Nitsche; Johannes Jakubik; Thomas Brunschwiler
>
> **摘要:** Vision-language models for Earth observation (EO) typically rely on the visual spectrum of data as the only model input, thus failing to leverage the rich spectral information available in the multispectral channels recorded by satellites. Therefore, we introduce Llama3-MS-CLIP, the first vision-language model pre-trained with contrastive learning on a large-scale multispectral dataset and report on the performance gains due to the extended spectral range. Furthermore, we present the largest-to-date image-caption dataset for multispectral data, consisting of one million Sentinel-2 samples and corresponding textual descriptions generated using Llama3-LLaVA-Next and Overture Maps data. We develop a scalable captioning pipeline, which is validated by domain experts. We evaluate Llama3-MS-CLIP on multispectral zero-shot image classification and retrieval using three datasets of varying complexity. Our results demonstrate that Llama3-MS-CLIP significantly outperforms other RGB-based approaches, improving classification accuracy by +6.77% on average and retrieval performance by +4.63% mAP compared to the second-best model. Our results emphasize the relevance of multispectral vision-language learning. The image-caption dataset, code, and model weights are available at https://github.com/IBM/MS-CLIP.
>
---
#### [replaced 012] MMMG: A Massive, Multidisciplinary, Multi-Tier Generation Benchmark for Text-to-Image Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10963v2](http://arxiv.org/pdf/2506.10963v2)**

> **作者:** Yuxuan Luo; Yuhui Yuan; Junwen Chen; Haonan Cai; Ziyi Yue; Yuwei Yang; Fatima Zohra Daha; Ji Li; Zhouhui Lian
>
> **备注:** 85 pages, 70 figures, code: https://github.com/MMMGBench/MMMG, project page: https://mmmgbench.github.io/
>
> **摘要:** In this paper, we introduce knowledge image generation as a new task, alongside the Massive Multi-Discipline Multi-Tier Knowledge-Image Generation Benchmark (MMMG) to probe the reasoning capability of image generation models. Knowledge images have been central to human civilization and to the mechanisms of human learning -- a fact underscored by dual-coding theory and the picture-superiority effect. Generating such images is challenging, demanding multimodal reasoning that fuses world knowledge with pixel-level grounding into clear explanatory visuals. To enable comprehensive evaluation, MMMG offers 4,456 expert-validated (knowledge) image-prompt pairs spanning 10 disciplines, 6 educational levels, and diverse knowledge formats such as charts, diagrams, and mind maps. To eliminate confounding complexity during evaluation, we adopt a unified Knowledge Graph (KG) representation. Each KG explicitly delineates a target image's core entities and their dependencies. We further introduce MMMG-Score to evaluate generated knowledge images. This metric combines factual fidelity, measured by graph-edit distance between KGs, with visual clarity assessment. Comprehensive evaluations of 16 state-of-the-art text-to-image generation models expose serious reasoning deficits -- low entity fidelity, weak relations, and clutter -- with GPT-4o achieving an MMMG-Score of only 50.20, underscoring the benchmark's difficulty. To spur further progress, we release FLUX-Reason (MMMG-Score of 34.45), an effective and open baseline that combines a reasoning LLM with diffusion models and is trained on 16,000 curated knowledge image-prompt pairs.
>
---
#### [replaced 013] Foundation Models in Medical Imaging -- A Review and Outlook
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09095v2](http://arxiv.org/pdf/2506.09095v2)**

> **作者:** Vivien van Veldhuizen; Vanessa Botha; Chunyao Lu; Melis Erdal Cesur; Kevin Groot Lipman; Edwin D. de Jong; Hugo Horlings; Clárisa I. Sanchez; Cees G. M. Snoek; Lodewyk Wessels; Ritse Mann; Eric Marcus; Jonas Teuwen
>
> **摘要:** Foundation models (FMs) are changing the way medical images are analyzed by learning from large collections of unlabeled data. Instead of relying on manually annotated examples, FMs are pre-trained to learn general-purpose visual features that can later be adapted to specific clinical tasks with little additional supervision. In this review, we examine how FMs are being developed and applied in pathology, radiology, and ophthalmology, drawing on evidence from over 150 studies. We explain the core components of FM pipelines, including model architectures, self-supervised learning methods, and strategies for downstream adaptation. We also review how FMs are being used in each imaging domain and compare design choices across applications. Finally, we discuss key challenges and open questions to guide future research.
>
---
#### [replaced 014] Scaling Prompt Instructed Zero Shot Composed Image Retrieval with Image-Only Data
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.00812v2](http://arxiv.org/pdf/2504.00812v2)**

> **作者:** Yiqun Duan; Sameera Ramasinghe; Stephen Gould; Ajanthan Thalaiyasingam
>
> **摘要:** Composed Image Retrieval (CIR) is the task of retrieving images matching a reference image augmented with a text, where the text describes changes to the reference image in natural language. Traditionally, models designed for CIR have relied on triplet data containing a reference image, reformulation text, and a target image. However, curating such triplet data often necessitates human intervention, leading to prohibitive costs. This challenge has hindered the scalability of CIR model training even with the availability of abundant unlabeled data. With the recent advances in foundational models, we advocate a shift in the CIR training paradigm where human annotations can be efficiently replaced by large language models (LLMs). Specifically, we demonstrate the capability of large captioning and language models in efficiently generating data for CIR only relying on unannotated image collections. Additionally, we introduce an embedding reformulation architecture that effectively combines image and text modalities. Our model, named InstructCIR, outperforms state-of-the-art methods in zero-shot composed image retrieval on CIRR and FashionIQ datasets. Furthermore, we demonstrate that by increasing the amount of generated data, our zero-shot model gets closer to the performance of supervised baselines.
>
---
#### [replaced 015] FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis without Learned Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16271v3](http://arxiv.org/pdf/2410.16271v3)**

> **作者:** Chin-Yang Lin; Chung-Ho Wu; Chang-Han Yeh; Shih-Han Yen; Cheng Sun; Yu-Lun Liu
>
> **备注:** Paper accepted to CVPR 2025. Project page: https://linjohnss.github.io/frugalnerf/
>
> **摘要:** Neural Radiance Fields (NeRF) face significant challenges in extreme few-shot scenarios, primarily due to overfitting and long training times. Existing methods, such as FreeNeRF and SparseNeRF, use frequency regularization or pre-trained priors but struggle with complex scheduling and bias. We introduce FrugalNeRF, a novel few-shot NeRF framework that leverages weight-sharing voxels across multiple scales to efficiently represent scene details. Our key contribution is a cross-scale geometric adaptation scheme that selects pseudo ground truth depth based on reprojection errors across scales. This guides training without relying on externally learned priors, enabling full utilization of the training data. It can also integrate pre-trained priors, enhancing quality without slowing convergence. Experiments on LLFF, DTU, and RealEstate-10K show that FrugalNeRF outperforms other few-shot NeRF methods while significantly reducing training time, making it a practical solution for efficient and accurate 3D scene reconstruction.
>
---
#### [replaced 016] LLaVA-c: Continual Improved Visual Instruction Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08666v2](http://arxiv.org/pdf/2506.08666v2)**

> **作者:** Wenzhuo Liu; Fei Zhu; Haiyang Guo; Longhui Wei; Cheng-Lin Liu
>
> **摘要:** Multimodal models like LLaVA-1.5 achieve state-of-the-art visual understanding through visual instruction tuning on multitask datasets, enabling strong instruction-following and multimodal performance. However, multitask learning faces challenges such as task balancing, requiring careful adjustment of data proportions, and expansion costs, where new tasks risk catastrophic forgetting and need costly retraining. Continual learning provides a promising alternative to acquiring new knowledge incrementally while preserving existing capabilities. However, current methods prioritize task-specific performance, neglecting base model degradation from overfitting to specific instructions, which undermines general capabilities. In this work, we propose a simple but effective method with two modifications on LLaVA-1.5: spectral-aware consolidation for improved task balance and unsupervised inquiry regularization to prevent base model degradation. We evaluate both general and task-specific performance across continual pretraining and fine-tuning. Experiments demonstrate that LLaVA-c consistently enhances standard benchmark performance and preserves general capabilities. For the first time, we show that task-by-task continual learning can achieve results that match or surpass multitask joint learning. The code will be publicly released.
>
---
#### [replaced 017] PATS: Proficiency-Aware Temporal Sampling for Multi-View Sports Skill Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04996v2](http://arxiv.org/pdf/2506.04996v2)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **摘要:** Automated sports skill assessment requires capturing fundamental movement patterns that distinguish expert from novice performance, yet current video sampling methods disrupt the temporal continuity essential for proficiency evaluation. To this end, we introduce Proficiency-Aware Temporal Sampling (PATS), a novel sampling strategy that preserves complete fundamental movements within continuous temporal segments for multi-view skill assessment. PATS adaptively segments videos to ensure each analyzed portion contains full execution of critical performance components, repeating this process across multiple segments to maximize information coverage while maintaining temporal coherence. Evaluated on the EgoExo4D benchmark with SkillFormer, PATS surpasses the state-of-the-art accuracy across all viewing configurations (+0.65% to +3.05%) and delivers substantial gains in challenging domains (+26.22% bouldering, +2.39% music, +1.13% basketball). Systematic analysis reveals that PATS successfully adapts to diverse activity characteristics-from high-frequency sampling for dynamic sports to fine-grained segmentation for sequential skills-demonstrating its effectiveness as an adaptive approach to temporal sampling that advances automated skill assessment for real-world applications.
>
---
#### [replaced 018] Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07044v4](http://arxiv.org/pdf/2506.07044v4)**

> **作者:** LASA Team; Weiwen Xu; Hou Pong Chan; Long Li; Mahani Aljunied; Ruifeng Yuan; Jianyu Wang; Chenghao Xiao; Guizhen Chen; Chaoqun Liu; Zhaodonghui Li; Yu Sun; Junao Shen; Chaojun Wang; Jie Tan; Deli Zhao; Tingyang Xu; Hao Zhang; Yu Rong
>
> **备注:** Technical Report, 53 pages, 25 tables, and 16 figures. Our webpage is https://alibaba-damo-academy.github.io/lingshu/
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ...
>
---
#### [replaced 019] ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10514v2](http://arxiv.org/pdf/2504.10514v2)**

> **作者:** Yijun Liang; Ming Li; Chenrui Fan; Ziyue Li; Dang Nguyen; Kwesi Cobbina; Shweta Bhardwaj; Jiuhai Chen; Fuxiao Liu; Tianyi Zhou
>
> **备注:** 36 pages, including references and appendix. Code is available at https://github.com/tianyi-lab/ColorBench
>
> **摘要:** Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI.
>
---
#### [replaced 020] Taming Rectified Flow for Inversion and Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.04746v3](http://arxiv.org/pdf/2411.04746v3)**

> **作者:** Jiangshan Wang; Junfu Pu; Zhongang Qi; Jiayi Guo; Yue Ma; Nisha Huang; Yuxin Chen; Xiu Li; Ying Shan
>
> **备注:** ICML 2025; GitHub: https://github.com/wangjiangshan0725/RF-Solver-Edit
>
> **摘要:** Rectified-flow-based diffusion transformers like FLUX and OpenSora have demonstrated outstanding performance in the field of image and video generation. Despite their robust generative capabilities, these models often struggle with inversion inaccuracies, which could further limit their effectiveness in downstream tasks such as image and video editing. To address this issue, we propose RF-Solver, a novel training-free sampler that effectively enhances inversion precision by mitigating the errors in the ODE-solving process of rectified flow. Specifically, we derive the exact formulation of the rectified flow ODE and apply the high-order Taylor expansion to estimate its nonlinear components, significantly enhancing the precision of ODE solutions at each timestep. Building upon RF-Solver, we further propose RF-Edit, a general feature-sharing-based framework for image and video editing. By incorporating self-attention features from the inversion process into the editing process, RF-Edit effectively preserves the structural information of the source image or video while achieving high-quality editing results. Our approach is compatible with any pre-trained rectified-flow-based models for image and video tasks, requiring no additional training or optimization. Extensive experiments across generation, inversion, and editing tasks in both image and video modalities demonstrate the superiority and versatility of our method. The source code is available at https://github.com/wangjiangshan0725/RF-Solver-Edit.
>
---
#### [replaced 021] Motion-R1: Chain-of-Thought Reasoning and Reinforcement Learning for Human Motion Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10353v2](http://arxiv.org/pdf/2506.10353v2)**

> **作者:** Runqi Ouyang; Haoyun Li; Zhenyuan Zhang; Xiaofeng Wang; Zheng Zhu; Guan Huang; Xingang Wang
>
> **摘要:** Recent advances in large language models, especially in natural language understanding and reasoning, have opened new possibilities for text-to-motion generation. Although existing approaches have made notable progress in semantic alignment and motion synthesis, they often rely on end-to-end mapping strategies that fail to capture deep linguistic structures and logical reasoning. Consequently, generated motions tend to lack controllability, consistency, and diversity. To address these limitations, we propose Motion-R1, a unified motion-language modeling framework that integrates a Chain-of-Thought mechanism. By explicitly decomposing complex textual instructions into logically structured action paths, Motion-R1 provides high-level semantic guidance for motion generation, significantly enhancing the model's ability to interpret and execute multi-step, long-horizon, and compositionally rich commands. To train our model, we adopt Group Relative Policy Optimization, a reinforcement learning algorithm designed for large models, which leverages motion quality feedback to optimize reasoning chains and motion synthesis jointly. Extensive experiments across multiple benchmark datasets demonstrate that Motion-R1 achieves competitive or superior performance compared to state-of-the-art methods, particularly in scenarios requiring nuanced semantic understanding and long-term temporal coherence. The code, model and data will be publicly available.
>
---
#### [replaced 022] HF-VTON: High-Fidelity Virtual Try-On via Consistent Geometric and Semantic Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19638v2](http://arxiv.org/pdf/2505.19638v2)**

> **作者:** Ming Meng; Qi Dong; Jiajie Li; Zhe Zhu; Xingyu Wang; Zhaoxin Fan; Wei Zhao; Wenjun Wu
>
> **备注:** After the publication of the paper, we discovered some significant errors/omissions that need to be corrected and improved
>
> **摘要:** Virtual try-on technology has become increasingly important in the fashion and retail industries, enabling the generation of high-fidelity garment images that adapt seamlessly to target human models. While existing methods have achieved notable progress, they still face significant challenges in maintaining consistency across different poses. Specifically, geometric distortions lead to a lack of spatial consistency, mismatches in garment structure and texture across poses result in semantic inconsistency, and the loss or distortion of fine-grained details diminishes visual fidelity. To address these challenges, we propose HF-VTON, a novel framework that ensures high-fidelity virtual try-on performance across diverse poses. HF-VTON consists of three key modules: (1) the Appearance-Preserving Warp Alignment Module (APWAM), which aligns garments to human poses, addressing geometric deformations and ensuring spatial consistency; (2) the Semantic Representation and Comprehension Module (SRCM), which captures fine-grained garment attributes and multi-pose data to enhance semantic representation, maintaining structural, textural, and pattern consistency; and (3) the Multimodal Prior-Guided Appearance Generation Module (MPAGM), which integrates multimodal features and prior knowledge from pre-trained models to optimize appearance generation, ensuring both semantic and geometric consistency. Additionally, to overcome data limitations in existing benchmarks, we introduce the SAMP-VTONS dataset, featuring multi-pose pairs and rich textual annotations for a more comprehensive evaluation. Experimental results demonstrate that HF-VTON outperforms state-of-the-art methods on both VITON-HD and SAMP-VTONS, excelling in visual fidelity, semantic consistency, and detail preservation.
>
---
#### [replaced 023] Capturing Temporal Dynamics in Large-Scale Canopy Tree Height Estimation
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19328v2](http://arxiv.org/pdf/2501.19328v2)**

> **作者:** Jan Pauls; Max Zimmer; Berkant Turan; Sassan Saatchi; Philippe Ciais; Sebastian Pokutta; Fabian Gieseke
>
> **备注:** ICML Camera-Ready, 9 pages main paper, 8 pages references and appendix, 9 figures, 8 tables
>
> **摘要:** With the rise in global greenhouse gas emissions, accurate large-scale tree canopy height maps are essential for understanding forest structure, estimating above-ground biomass, and monitoring ecological disruptions. To this end, we present a novel approach to generate large-scale, high-resolution canopy height maps over time. Our model accurately predicts canopy height over multiple years given Sentinel-1 composite and Sentinel~2 time series satellite data. Using GEDI LiDAR data as the ground truth for training the model, we present the first 10m resolution temporal canopy height map of the European continent for the period 2019-2022. As part of this product, we also offer a detailed canopy height map for 2020, providing more precise estimates than previous studies. Our pipeline and the resulting temporal height map are publicly available, enabling comprehensive large-scale monitoring of forests and, hence, facilitating future research and ecological analyses.
>
---
#### [replaced 024] DualX-VSR: Dual Axial Spatial$\times$Temporal Transformer for Real-World Video Super-Resolution without Motion Compensation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04830v2](http://arxiv.org/pdf/2506.04830v2)**

> **作者:** Shuo Cao; Yihao Liu; Xiaohui Li; Yuanting Gao; Yu Zhou; Chao Dong
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Transformer-based models like ViViT and TimeSformer have advanced video understanding by effectively modeling spatiotemporal dependencies. Recent video generation models, such as Sora and Vidu, further highlight the power of transformers in long-range feature extraction and holistic spatiotemporal modeling. However, directly applying these models to real-world video super-resolution (VSR) is challenging, as VSR demands pixel-level precision, which can be compromised by tokenization and sequential attention mechanisms. While recent transformer-based VSR models attempt to address these issues using smaller patches and local attention, they still face limitations such as restricted receptive fields and dependence on optical flow-based alignment, which can introduce inaccuracies in real-world settings. To overcome these issues, we propose Dual Axial Spatial$\times$Temporal Transformer for Real-World Video Super-Resolution (DualX-VSR), which introduces a novel dual axial spatial$\times$temporal attention mechanism that integrates spatial and temporal information along orthogonal directions. DualX-VSR eliminates the need for motion compensation, offering a simplified structure that provides a cohesive representation of spatiotemporal information. As a result, DualX-VSR achieves high fidelity and superior performance in real-world VSR task.
>
---
#### [replaced 025] Real-time Seafloor Segmentation and Mapping
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.10750v2](http://arxiv.org/pdf/2504.10750v2)**

> **作者:** Michele Grimaldi; Nouf Alkaabi; Francesco Ruscio; Sebastian Realpe Rua; Rafael Garcia; Nuno Gracias
>
> **摘要:** Posidonia oceanica meadows are a species of seagrass highly dependent on rocks for their survival and conservation. In recent years, there has been a concerning global decline in this species, emphasizing the critical need for efficient monitoring and assessment tools. While deep learning-based semantic segmentation and visual automated monitoring systems have shown promise in a variety of applications, their performance in underwater environments remains challenging due to complex water conditions and limited datasets. This paper introduces a framework that combines machine learning and computer vision techniques to enable an autonomous underwater vehicle (AUV) to inspect the boundaries of Posidonia oceanica meadows autonomously. The framework incorporates an image segmentation module using an existing Mask R-CNN model and a strategy for Posidonia oceanica meadow boundary tracking. Furthermore, a new class dedicated to rocks is introduced to enhance the existing model, aiming to contribute to a comprehensive monitoring approach and provide a deeper understanding of the intricate interactions between the meadow and its surrounding environment. The image segmentation model is validated using real underwater images, while the overall inspection framework is evaluated in a realistic simulation environment, replicating actual monitoring scenarios with real underwater images. The results demonstrate that the proposed framework enables the AUV to autonomously accomplish the main tasks of underwater inspection and segmentation of rocks. Consequently, this work holds significant potential for the conservation and protection of marine environments, providing valuable insights into the status of Posidonia oceanica meadows and supporting targeted preservation efforts
>
---
#### [replaced 026] Ming-Lite-Uni: Advancements in Unified Architecture for Natural Multimodal Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02471v3](http://arxiv.org/pdf/2505.02471v3)**

> **作者:** Inclusion AI; Biao Gong; Cheng Zou; Dandan Zheng; Hu Yu; Jingdong Chen; Jianxin Sun; Junbo Zhao; Jun Zhou; Kaixiang Ji; Lixiang Ru; Libin Wang; Qingpei Guo; Rui Liu; Weilong Chai; Xinyu Xiao; Ziyuan Huang
>
> **备注:** https://github.com/inclusionAI/Ming/tree/Ming-Lite-Omni-Preview/Ming-unify
>
> **摘要:** We introduce Ming-Lite-Uni, an open-source multimodal framework featuring a newly designed unified visual generator and a native multimodal autoregressive model tailored for unifying vision and language. Specifically, this project provides an open-source implementation of the integrated MetaQueries and M2-omni framework, while introducing the novel multi-scale learnable tokens and multi-scale representation alignment strategy. By leveraging a fixed MLLM and a learnable diffusion model, Ming-Lite-Uni enables native multimodal AR models to perform both text-to-image generation and instruction based image editing tasks, expanding their capabilities beyond pure visual understanding. Our experimental results demonstrate the strong performance of Ming-Lite-Uni and illustrate the impressive fluid nature of its interactive process. All code and model weights are open-sourced to foster further exploration within the community. Notably, this work aligns with concurrent multimodal AI milestones - such as ChatGPT-4o with native image generation updated in March 25, 2025 - underscoring the broader significance of unified models like Ming-Lite-Uni on the path toward AGI. Ming-Lite-Uni is in alpha stage and will soon be further refined.
>
---
#### [replaced 027] Efficient Visual State Space Model for Image Deblurring
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14343v2](http://arxiv.org/pdf/2405.14343v2)**

> **作者:** Lingshun Kong; Jiangxin Dong; Jinhui Tang; Ming-Hsuan Yang; Jinshan Pan
>
> **备注:** CVPR 2025
>
> **摘要:** Convolutional neural networks (CNNs) and Vision Transformers (ViTs) have achieved excellent performance in image restoration. While ViTs generally outperform CNNs by effectively capturing long-range dependencies and input-specific characteristics, their computational complexity increases quadratically with image resolution. This limitation hampers their practical application in high-resolution image restoration. In this paper, we propose a simple yet effective visual state space model (EVSSM) for image deblurring, leveraging the benefits of state space models (SSMs) for visual data. In contrast to existing methods that employ several fixed-direction scanning for feature extraction, which significantly increases the computational cost, we develop an efficient visual scan block that applies various geometric transformations before each SSM-based module, capturing useful non-local information and maintaining high efficiency. In addition, to more effectively capture and represent local information, we propose an efficient discriminative frequency domain-based feedforward network (EDFFN), which can effectively estimate useful frequency information for latent clear image restoration. Extensive experimental results show that the proposed EVSSM performs favorably against state-of-the-art methods on benchmark datasets and real-world images. The code is available at https://github.com/kkkls/EVSSM.
>
---
#### [replaced 028] Improving Acoustic Scene Classification with City Features
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.16862v2](http://arxiv.org/pdf/2503.16862v2)**

> **作者:** Yiqiang Cai; Yizhou Tan; Shengchen Li; Xi Shao; Mark D. Plumbley
>
> **摘要:** Acoustic scene recordings are often collected from a diverse range of cities. Most existing acoustic scene classification (ASC) approaches focus on identifying common acoustic scene patterns across cities to enhance generalization. However, the potential acoustic differences introduced by city-specific environmental and cultural factors are overlooked. In this paper, we hypothesize that the city-specific acoustic features are beneficial for the ASC task rather than being treated as noise or bias. To this end, we propose City2Scene, a novel framework that leverages city features to improve ASC. Unlike conventional approaches that may discard or suppress city information, City2Scene transfers the city-specific knowledge from pre-trained city classification models to scene classification model using knowledge distillation. We evaluate City2Scene on three datasets of DCASE Challenge Task 1, which include both scene and city labels. Experimental results demonstrate that city features provide valuable information for classifying scenes. By distilling city-specific knowledge, City2Scene effectively improves accuracy across a variety of lightweight CNN backbones, achieving competitive performance to the top-ranked solutions of DCASE Challenge in recent years.
>
---
#### [replaced 029] E2MPL:An Enduring and Efficient Meta Prompt Learning Framework for Few-shot Unsupervised Domain Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.04066v2](http://arxiv.org/pdf/2407.04066v2)**

> **作者:** Wanqi Yang; Haoran Wang; Lei Wang; Ge Song; Ming Yang; Yang Gao
>
> **摘要:** Few-shot unsupervised domain adaptation (FS-UDA) leverages a limited amount of labeled data from a source domain to enable accurate classification in an unlabeled target domain. Despite recent advancements, current approaches of FS-UDA continue to confront a major challenge: models often demonstrate instability when adapted to new FS-UDA tasks and necessitate considerable time investment. To address these challenges, we put forward a novel framework called Enduring and Efficient Meta-Prompt Learning (E2MPL) for FS-UDA. Within this framework, we utilize the pre-trained CLIP model as the backbone of feature learning. Firstly, we design domain-shared prompts, consisting of virtual tokens, which primarily capture meta-knowledge from a wide range of meta-tasks to mitigate the domain gaps. Secondly, we develop a task prompt learning network that adaptively learns task-specific specific prompts with the goal of achieving fast and stable task generalization. Thirdly, we formulate the meta-prompt learning process as a bilevel optimization problem, consisting of (outer) meta-prompt learner and (inner) task-specific classifier and domain adapter. Also, the inner objective of each meta-task has the closed-form solution, which enables efficient prompt learning and adaptation to new tasks in a single step. Extensive experimental studies demonstrate the promising performance of our framework in a domain adaptation benchmark dataset DomainNet. Compared with state-of-the-art methods, our method has improved accuracy by at least 15.4% and reduced the time by 68.5% on average in 5-way 1-shot tasks, and improved accuracy by 8.7% and reduced the time by 74.1% on average in 5-way 5-shot tasks. Moreover, our approach exhibits more enduring performance than the other methods, i.e., being more stable across 3600 test tasks.
>
---
#### [replaced 030] Geospatial Artificial Intelligence for Satellite-Based Flood Extent Mapping: Concepts, Advances, and Future Perspectives
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.02214v3](http://arxiv.org/pdf/2504.02214v3)**

> **作者:** Hyunho Lee; Wenwen Li
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Geospatial Artificial Intelligence (GeoAI) for satellite-based flood extent mapping systematically integrates artificial intelligence techniques with satellite data to identify flood events and assess their impacts, for disaster management and spatial decision-making. The primary output often includes flood extent maps, which delineate the affected areas, along with additional analytical outputs such as uncertainty estimation and change detection.
>
---
#### [replaced 031] Diversifying Human Pose in Synthetic Data for Aerial-view Human Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15939v2](http://arxiv.org/pdf/2405.15939v2)**

> **作者:** Yi-Ting Shen; Hyungtae Lee; Heesung Kwon; Shuvra S. Bhattacharyya
>
> **备注:** ICIP 2025
>
> **摘要:** Synthetic data generation has emerged as a promising solution to the data scarcity issue in aerial-view human detection. However, creating datasets that accurately reflect varying real-world human appearances, particularly diverse poses, remains challenging and labor-intensive. To address this, we propose SynPoseDiv, a novel framework that diversifies human poses within existing synthetic datasets. SynPoseDiv tackles two key challenges: generating realistic, diverse 3D human poses using a diffusion-based pose generator, and producing images of virtual characters in novel poses through a source-to-target image translator. The framework incrementally transitions characters into new poses using optimized pose sequences identified via Dijkstra's algorithm. Experiments demonstrate that SynPoseDiv significantly improves detection accuracy across multiple aerial-view human detection benchmarks, especially in low-shot scenarios, and remains effective regardless of the training approach or dataset size.
>
---
#### [replaced 032] Holstein-Friesian Re-Identification using Multiple Cameras and Self-Supervision on a Working Farm
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.12695v3](http://arxiv.org/pdf/2410.12695v3)**

> **作者:** Phoenix Yu; Tilo Burghardt; Andrew W Dowsey; Neill W Campbell
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** We present MultiCamCows2024, a farm-scale image dataset filmed across multiple cameras for the biometric identification of individual Holstein-Friesian cattle exploiting their unique black and white coat-patterns. Captured by three ceiling-mounted visual sensors covering adjacent barn areas over seven days on a working dairy farm, the dataset comprises 101,329 images of 90 cows, plus underlying original CCTV footage. The dataset is provided with full computer vision recognition baselines, that is both a supervised and self-supervised learning framework for individual cow identification trained on cattle tracklets. We report a performance above 96% single image identification accuracy from the dataset and demonstrate that combining data from multiple cameras during learning enhances self-supervised identification. We show that our framework enables automatic cattle identification, barring only the simple human verification of tracklet integrity during data collection. Crucially, our study highlights that multi-camera, supervised and self-supervised components in tandem not only deliver highly accurate individual cow identification, but also achieve this efficiently with no labelling of cattle identities by humans. We argue that this improvement in efficacy has practical implications for livestock management, behaviour analysis, and agricultural monitoring. For reproducibility and practical ease of use, we publish all key software and code including re-identification components and the species detector with this paper, available at https://tinyurl.com/MultiCamCows2024.
>
---
#### [replaced 033] One Diffusion to Generate Them All
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.16318v2](http://arxiv.org/pdf/2411.16318v2)**

> **作者:** Duong H. Le; Tuan Pham; Sangho Lee; Christopher Clark; Aniruddha Kembhavi; Stephan Mandt; Ranjay Krishna; Jiasen Lu
>
> **备注:** CVPR 2025; two first authors contribute equally
>
> **摘要:** We introduce OneDiffusion, a versatile, large-scale diffusion model that seamlessly supports bidirectional image synthesis and understanding across diverse tasks. It enables conditional generation from inputs such as text, depth, pose, layout, and semantic maps, while also handling tasks like image deblurring, upscaling, and reverse processes such as depth estimation and segmentation. Additionally, OneDiffusion allows for multi-view generation, camera pose estimation, and instant personalization using sequential image inputs. Our model takes a straightforward yet effective approach by treating all tasks as frame sequences with varying noise scales during training, allowing any frame to act as a conditioning image at inference time. Our unified training framework removes the need for specialized architectures, supports scalable multi-task training, and adapts smoothly to any resolution, enhancing both generalization and scalability. Experimental results demonstrate competitive performance across tasks in both generation and prediction such as text-to-image, multiview generation, ID preservation, depth estimation and camera pose estimation despite relatively small training dataset. Our code and checkpoint are freely available at https://github.com/lehduong/OneDiffusion
>
---
#### [replaced 034] PiPViT: Patch-based Visual Interpretable Prototypes for Retinal Image Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.10669v2](http://arxiv.org/pdf/2506.10669v2)**

> **作者:** Marzieh Oghbaie; Teresa Araújo; Hrvoje Bogunović
>
> **摘要:** Background and Objective: Prototype-based methods improve interpretability by learning fine-grained part-prototypes; however, their visualization in the input pixel space is not always consistent with human-understandable biomarkers. In addition, well-known prototype-based approaches typically learn extremely granular prototypes that are less interpretable in medical imaging, where both the presence and extent of biomarkers and lesions are critical. Methods: To address these challenges, we propose PiPViT (Patch-based Visual Interpretable Prototypes), an inherently interpretable prototypical model for image recognition. Leveraging a vision transformer (ViT), PiPViT captures long-range dependencies among patches to learn robust, human-interpretable prototypes that approximate lesion extent only using image-level labels. Additionally, PiPViT benefits from contrastive learning and multi-resolution input processing, which enables effective localization of biomarkers across scales. Results: We evaluated PiPViT on retinal OCT image classification across four datasets, where it achieved competitive quantitative performance compared to state-of-the-art methods while delivering more meaningful explanations. Moreover, quantitative evaluation on a hold-out test set confirms that the learned prototypes are semantically and clinically relevant. We believe PiPViT can transparently explain its decisions and assist clinicians in understanding diagnostic outcomes. Github page: https://github.com/marziehoghbaie/PiPViT
>
---
#### [replaced 035] SkillFormer: Unified Multi-View Video Understanding for Proficiency Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08665v2](http://arxiv.org/pdf/2505.08665v2)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **摘要:** Assessing human skill levels in complex activities is a challenging problem with applications in sports, rehabilitation, and training. In this work, we present SkillFormer, a parameter-efficient architecture for unified multi-view proficiency estimation from egocentric and exocentric videos. Building on the TimeSformer backbone, SkillFormer introduces a CrossViewFusion module that fuses view-specific features using multi-head cross-attention, learnable gating, and adaptive self-calibration. We leverage Low-Rank Adaptation to fine-tune only a small subset of parameters, significantly reducing training costs. In fact, when evaluated on the EgoExo4D dataset, SkillFormer achieves state-of-the-art accuracy in multi-view settings while demonstrating remarkable computational efficiency, using 4.5x fewer parameters and requiring 3.75x fewer training epochs than prior baselines. It excels in multiple structured tasks, confirming the value of multi-view integration for fine-grained skill assessment.
>
---
#### [replaced 036] Efficient Visual Representation Learning with Heat Conduction Equation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05901v3](http://arxiv.org/pdf/2408.05901v3)**

> **作者:** Zhemin Zhang; Xun Gong
>
> **备注:** Accepted by IJCAI2025
>
> **摘要:** Foundation models, such as CNNs and ViTs, have powered the development of image representation learning. However, general guidance to model architecture design is still missing. Inspired by the connection between image representation learning and heat conduction, we model images by the heat conduction equation, where the essential idea is to conceptualize image features as temperatures and model their information interaction as the diffusion of thermal energy. Based on this idea, we find that many modern model architectures, such as residual structures, SE block, and feed-forward networks, can be interpreted from the perspective of the heat conduction equation. Therefore, we leverage the heat equation to design new and more interpretable models. As an example, we propose the Heat Conduction Layer and the Refinement Approximation Layer inspired by solving the heat conduction equation using Finite Difference Method and Fourier series, respectively. The main goal of this paper is to integrate the overall architectural design of neural networks into the theoretical framework of heat conduction. Nevertheless, our Heat Conduction Network (HcNet) still shows competitive performance, e.g., HcNet-T achieves 83.0% top-1 accuracy on ImageNet-1K while only requiring 28M parameters and 4.1G MACs. The code is publicly available at: https://github.com/ZheminZhang1/HcNet.
>
---
#### [replaced 037] A Self-supervised Motion Representation for Portrait Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10096v2](http://arxiv.org/pdf/2503.10096v2)**

> **作者:** Qiyuan Zhang; Chenyu Wu; Wenzhang Sun; Huaize Liu; Donglin Di; Wei Chen; Changqing Zou
>
> **摘要:** Recent advancements in portrait video generation have been noteworthy. However, existing methods rely heavily on human priors and pre-trained generative models, Motion representations based on human priors may introduce unrealistic motion, while methods relying on pre-trained generative models often suffer from inefficient inference. To address these challenges, we propose Semantic Latent Motion (SeMo), a compact and expressive motion representation. Leveraging this representation, our approach achieve both high-quality visual results and efficient inference. SeMo follows an effective three-step framework: Abstraction, Reasoning, and Generation. First, in the Abstraction step, we use a carefully designed Masked Motion Encoder, which leverages a self-supervised learning paradigm to compress the subject's motion state into a compact and abstract latent motion (1D token). Second, in the Reasoning step, we efficiently generate motion sequences based on the driving audio signal. Finally, in the Generation step, the motion dynamics serve as conditional information to guide the motion decoder in synthesizing realistic transitions from reference frame to target video. Thanks to the compact and expressive nature of Semantic Latent Motion, our method achieves efficient motion representation and high-quality video generation. User studies demonstrate that our approach surpasses state-of-the-art models with an 81% win rate in realism. Extensive experiments further highlight its strong compression capability, reconstruction quality, and generative potential.
>
---
#### [replaced 038] We Care Each Pixel: Calibrating on Medical Segmentation Model
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05107v2](http://arxiv.org/pdf/2503.05107v2)**

> **作者:** Wenhao Liang; Wei Zhang; Lin Yue; Miao Xu; Olaf Maennel; Weitong Chen
>
> **备注:** Under Reviewing
>
> **摘要:** Medical image segmentation is fundamental for computer-aided diagnostics, providing accurate delineation of anatomical structures and pathological regions. While common metrics such as Accuracy, DSC, IoU, and HD primarily quantify spatial agreement between predictions and ground-truth labels, they do not assess the calibration quality of segmentation models, which is crucial for clinical reliability. To address this limitation, we propose pixel-wise Expected Calibration Error (pECE), a novel metric that explicitly measures miscalibration at the pixel level, thereby ensuring both spatial precision and confidence reliability. We further introduce a morphological adaptation strategy that applies morphological operations to ground-truth masks before computing calibration losses, particularly benefiting margin-based losses such as Margin SVLS and NACL. Additionally, we present the Signed Distance Calibration Loss (SDC), which aligns boundary geometry with calibration objectives by penalizing discrepancies between predicted and ground-truth signed distance functions (SDFs). Extensive experiments demonstrate that our method not only enhances segmentation performance but also improves calibration quality, yielding more trustworthy confidence estimates. Code is available at: https://github.com/EagleAdelaide/SDC-Loss.
>
---
#### [replaced 039] seg2med: a bridge from artificial anatomy to multimodal medical images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09182v2](http://arxiv.org/pdf/2504.09182v2)**

> **作者:** Zeyu Yang; Zhilin Chen; Yipeng Sun; Anika Strittmatter; Anish Raj; Ahmad Allababidi; Johann S. Rink; Frank G. Zöllner
>
> **备注:** 17 pages, 10 figures Web demo available at https://huggingface.co/spaces/Zeyu0601/frankenstein
>
> **摘要:** We present seg2med, a modular framework for anatomy-driven multimodal medical image synthesis. The system integrates three components to enable high-fidelity, cross-modality generation of CT and MR images based on structured anatomical priors. First, anatomical maps are independently derived from three sources: real patient data, XCAT digital phantoms, and synthetic anatomies created by combining organs from multiple patients. Second, we introduce PhysioSynth, a modality-specific simulator that converts anatomical masks into prior volumes using tissue-dependent parameters (e.g., HU, T1, T2, proton density) and modality-specific signal models. It supports simulation of CT and multiple MR sequences including GRE, SPACE, and VIBE. Third, the synthesized anatomical priors are used to train 2-channel conditional denoising diffusion models, which take the anatomical prior as structural condition alongside the noisy image, enabling generation of high-quality, structurally aligned images. The framework achieves SSIM of 0.94 for CT and 0.89 for MR compared to real data, and FSIM of 0.78 for simulated CT. The generative quality is further supported by a Frechet Inception Distance (FID) of 3.62 for CT synthesis. In modality conversion, seg2med achieves SSIM of 0.91 for MR to CT and 0.77 for CT to MR. Anatomical fidelity evaluation shows synthetic CT achieves mean Dice scores above 0.90 for 11 key abdominal organs, and above 0.80 for 34 of 59 total organs. These results underscore seg2med's utility in cross-modality synthesis, data augmentation, and anatomy-aware medical AI.
>
---
#### [replaced 040] Fish feeding behavior recognition and intensity quantification methods in aquaculture: From single modality analysis to multimodality fusion
- **分类: cs.CV; cs.ET; I.4.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2502.15311v2](http://arxiv.org/pdf/2502.15311v2)**

> **作者:** Shulong Zhang; Jiayin Zhao; Mingyuan Yao; Xiao Liu; Yukang Huo; Yingyi Chen; Haihua Wang
>
> **备注:** 24 pages, 4 figures,
>
> **摘要:** As a key part of aquaculture management, fish feeding behavior recognition and intensity quantification has been a hot area of great concern to researchers, and it plays a crucial role in monitoring fish health, guiding baiting work and improving aquaculture efficiency. In order to better carry out the related work in the future, this paper firstly analyzes and compares the existing reviews. Then reviews the research advances of fish feeding behavior recognition and intensity quantification methods based on computer vision, acoustics and sensors in a single modality. Meanwhile, the application of the current emerging multimodal fusion in fish feeding behavior recognition and intensity quantification methods is expounded. Finally, the advantages and disadvantages of various techniques are compared and analyzed, and the future research directions are envisioned.
>
---
#### [replaced 041] Scaling Human Activity Recognition: A Comparative Evaluation of Synthetic Data Generation and Augmentation Techniques
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07612v2](http://arxiv.org/pdf/2506.07612v2)**

> **作者:** Zikang Leng; Archith Iyer; Thomas Plötz
>
> **摘要:** Human activity recognition (HAR) is often limited by the scarcity of labeled datasets due to the high cost and complexity of real-world data collection. To mitigate this, recent work has explored generating virtual inertial measurement unit (IMU) data via cross-modality transfer. While video-based and language-based pipelines have each shown promise, they differ in assumptions and computational cost. Moreover, their effectiveness relative to traditional sensor-level data augmentation remains unclear. In this paper, we present a direct comparison between these two virtual IMU generation approaches against classical data augmentation techniques. We construct a large-scale virtual IMU dataset spanning 100 diverse activities from Kinetics-400 and simulate sensor signals at 22 body locations. The three data generation strategies are evaluated on benchmark HAR datasets (UTD-MHAD, PAMAP2, HAD-AW) using four popular models. Results show that virtual IMU data significantly improves performance over real or augmented data alone, particularly under limited-data conditions. We offer practical guidance on choosing data generation strategies and highlight the distinct advantages and disadvantages of each approach.
>
---
#### [replaced 042] MiniMaxAD: A Lightweight Autoencoder for Feature-Rich Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.09933v4](http://arxiv.org/pdf/2405.09933v4)**

> **作者:** Fengjie Wang; Chengming Liu; Lei Shi; Pang Haibo
>
> **备注:** Accept by Computers in Industry
>
> **摘要:** Previous industrial anomaly detection methods often struggle to handle the extensive diversity in training sets, particularly when they contain stylistically diverse and feature-rich samples, which we categorize as feature-rich anomaly detection datasets (FRADs). This challenge is evident in applications such as multi-view and multi-class scenarios. To address this challenge, we developed MiniMaxAD, a efficient autoencoder designed to efficiently compress and memorize extensive information from normal images. Our model employs a technique that enhances feature diversity, thereby increasing the effective capacity of the network. It also utilizes large kernel convolution to extract highly abstract patterns, which contribute to efficient and compact feature embedding. Moreover, we introduce an Adaptive Contraction Hard Mining Loss (ADCLoss), specifically tailored to FRADs. In our methodology, any dataset can be unified under the framework of feature-rich anomaly detection, in a way that the benefits far outweigh the drawbacks. Our approach has achieved state-of-the-art performance in multiple challenging benchmarks. Code is available at: \href{https://github.com/WangFengJiee/MiniMaxAD}{https://github.com/WangFengJiee/MiniMaxAD}
>
---
#### [replaced 043] 3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation for High-Fidelity 3D Shapes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19037v2](http://arxiv.org/pdf/2411.19037v2)**

> **作者:** Tejaswini Medi; Arianna Rampini; Pradyumna Reddy; Pradeep Kumar Jayaraman; Margret Keuper
>
> **摘要:** Autoregressive (AR) models have achieved remarkable success in natural language and image generation, but their application to 3D shape modeling remains largely unexplored. Unlike diffusion models, AR models enable more efficient and controllable generation with faster inference times, making them especially suitable for data-intensive domains. Traditional 3D generative models using AR approaches often rely on ``next-token" predictions at the voxel or point level. While effective for certain applications, these methods can be restrictive and computationally expensive when dealing with large-scale 3D data. To tackle these challenges, we introduce 3D-WAG, an AR model for 3D implicit distance fields that can perform unconditional shape generation, class-conditioned and also text-conditioned shape generation. Our key idea is to encode shapes as multi-scale wavelet token maps and use a Transformer to predict the ``next higher-resolution token map" in an autoregressive manner. By redefining 3D AR generation task as ``next-scale" prediction, we reduce the computational cost of generation compared to traditional ``next-token" prediction models, while preserving essential geometric details of 3D shapes in a more structured and hierarchical manner. We evaluate 3D-WAG to showcase its benefit by quantitative and qualitative comparisons with state-of-the-art methods on widely used benchmarks. Our results show 3D-WAG achieves superior performance in key metrics like Coverage and MMD, generating high-fidelity 3D shapes that closely match the real data distribution.
>
---
#### [replaced 044] SAP-Bench: Benchmarking Multimodal Large Language Models in Surgical Action Planning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07196v2](http://arxiv.org/pdf/2506.07196v2)**

> **作者:** Mengya Xu; Zhongzhen Huang; Dillan Imans; Yiru Ye; Xiaofan Zhang; Qi Dou
>
> **备注:** The authors could not reach a consensus on the final version of this paper, necessitating its withdrawal
>
> **摘要:** Effective evaluation is critical for driving advancements in MLLM research. The surgical action planning (SAP) task, which aims to generate future action sequences from visual inputs, demands precise and sophisticated analytical capabilities. Unlike mathematical reasoning, surgical decision-making operates in life-critical domains and requires meticulous, verifiable processes to ensure reliability and patient safety. This task demands the ability to distinguish between atomic visual actions and coordinate complex, long-horizon procedures, capabilities that are inadequately evaluated by current benchmarks. To address this gap, we introduce SAP-Bench, a large-scale, high-quality dataset designed to enable multimodal large language models (MLLMs) to perform interpretable surgical action planning. Our SAP-Bench benchmark, derived from the cholecystectomy procedures context with the mean duration of 1137.5s, and introduces temporally-grounded surgical action annotations, comprising the 1,226 clinically validated action clips (mean duration: 68.7s) capturing five fundamental surgical actions across 74 procedures. The dataset provides 1,152 strategically sampled current frames, each paired with the corresponding next action as multimodal analysis anchors. We propose the MLLM-SAP framework that leverages MLLMs to generate next action recommendations from the current surgical scene and natural language instructions, enhanced with injected surgical domain knowledge. To assess our dataset's effectiveness and the broader capabilities of current models, we evaluate seven state-of-the-art MLLMs (e.g., OpenAI-o1, GPT-4o, QwenVL2.5-72B, Claude-3.5-Sonnet, GeminiPro2.5, Step-1o, and GLM-4v) and reveal critical gaps in next action prediction performance.
>
---
#### [replaced 045] Sheet Music Benchmark: Standardized Optical Music Recognition Evaluation
- **分类: cs.CV; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.10488v2](http://arxiv.org/pdf/2506.10488v2)**

> **作者:** Juan C. Martinez-Sevilla; Joan Cerveto-Serrano; Noelia Luna; Greg Chapman; Craig Sapp; David Rizo; Jorge Calvo-Zaragoza
>
> **摘要:** In this work, we introduce the Sheet Music Benchmark (SMB), a dataset of six hundred and eighty-five pages specifically designed to benchmark Optical Music Recognition (OMR) research. SMB encompasses a diverse array of musical textures, including monophony, pianoform, quartet, and others, all encoded in Common Western Modern Notation using the Humdrum **kern format. Alongside SMB, we introduce the OMR Normalized Edit Distance (OMR-NED), a new metric tailored explicitly for evaluating OMR performance. OMR-NED builds upon the widely-used Symbol Error Rate (SER), offering a fine-grained and detailed error analysis that covers individual musical elements such as note heads, beams, pitches, accidentals, and other critical notation features. The resulting numeric score provided by OMR-NED facilitates clear comparisons, enabling researchers and end-users alike to identify optimal OMR approaches. Our work thus addresses a long-standing gap in OMR evaluation, and we support our contributions with baseline experiments using standardized SMB dataset splits for training and assessing state-of-the-art methods.
>
---
#### [replaced 046] PhysNav-DG: A Novel Adaptive Framework for Robust VLM-Sensor Fusion in Navigation Applications
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.01881v3](http://arxiv.org/pdf/2505.01881v3)**

> **作者:** Trisanth Srinivasan; Santosh Patapati
>
> **备注:** Accepted at IEEE/CVF Computer Society Conference on Computer Vision and Pattern Recognition Workshops 2025 (CVPRW)
>
> **摘要:** Robust navigation in diverse environments and domains requires both accurate state estimation and transparent decision making. We present PhysNav-DG, a novel framework that integrates classical sensor fusion with the semantic power of vision-language models. Our dual-branch architecture predicts navigation actions from multi-sensor inputs while simultaneously generating detailed chain-of-thought explanations. A modified Adaptive Kalman Filter dynamically adjusts its noise parameters based on environmental context. It leverages several streams of raw sensor data along with semantic insights from models such as LLaMA 3.2 11B and BLIP-2. To evaluate our approach, we introduce the MD-NEX Benchmark, a novel multi-domain dataset that unifies indoor navigation, autonomous driving, and social navigation tasks with ground-truth actions and human-validated explanations. Extensive experiments and ablations show that PhysNav-DG improves navigation success rates by over 20% and achieves high efficiency, with explanations that are both highly grounded and clear. This work connects high-level semantic reasoning and geometric planning for safer and more trustworthy autonomous systems.
>
---
#### [replaced 047] Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07903v2](http://arxiv.org/pdf/2506.07903v2)**

> **作者:** Kevin Rojas; Yuchen Zhu; Sichen Zhu; Felix X. -F. Ye; Molei Tao
>
> **备注:** Accepted to ICML 2025. Code available at https://github.com/KevinRojas1499/Diffuse-Everything
>
> **摘要:** Diffusion models have demonstrated remarkable performance in generating unimodal data across various tasks, including image, video, and text generation. On the contrary, the joint generation of multimodal data through diffusion models is still in the early stages of exploration. Existing approaches heavily rely on external preprocessing protocols, such as tokenizers and variational autoencoders, to harmonize varied data representations into a unified, unimodal format. This process heavily demands the high accuracy of encoders and decoders, which can be problematic for applications with limited data. To lift this restriction, we propose a novel framework for building multimodal diffusion models on arbitrary state spaces, enabling native generation of coupled data across different modalities. By introducing an innovative decoupled noise schedule for each modality, we enable both unconditional and modality-conditioned generation within a single model simultaneously. We empirically validate our approach for text-image generation and mixed-type tabular data synthesis, demonstrating that it achieves competitive performance.
>
---
#### [replaced 048] Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.05205v5](http://arxiv.org/pdf/2501.05205v5)**

> **作者:** Xueyi Ke; Satoshi Tsutsui; Yayun Zhang; Bihan Wen
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Infants develop complex visual understanding rapidly, even preceding the acquisition of linguistic skills. As computer vision seeks to replicate the human vision system, understanding infant visual development may offer valuable insights. In this paper, we present an interdisciplinary study exploring this question: can a computational model that imitates the infant learning process develop broader visual concepts that extend beyond the vocabulary it has heard, similar to how infants naturally learn? To investigate this, we analyze a recently published model in Science by Vong et al., which is trained on longitudinal, egocentric images of a single child paired with transcribed parental speech. We perform neuron labeling to identify visual concept neurons hidden in the model's internal representations. We then demonstrate that these neurons can recognize objects beyond the model's original vocabulary. Furthermore, we compare the differences in representation between infant models and those in modern computer vision models, such as CLIP and ImageNet pre-trained model. Ultimately, our work bridges cognitive science and computer vision by analyzing the internal representations of a computational model trained on an infant visual and linguistic inputs. Project page is available at https://kexueyi.github.io/webpage-discover-hidden-visual-concepts.
>
---
#### [replaced 049] Vision-Language Models for Edge Networks: A Comprehensive Survey
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07855v2](http://arxiv.org/pdf/2502.07855v2)**

> **作者:** Ahmed Sharshar; Latif U. Khan; Waseem Ullah; Mohsen Guizani
>
> **摘要:** Vision Large Language Models (VLMs) combine visual understanding with natural language processing, enabling tasks like image captioning, visual question answering, and video analysis. While VLMs show impressive capabilities across domains such as autonomous vehicles, smart surveillance, and healthcare, their deployment on resource-constrained edge devices remains challenging due to processing power, memory, and energy limitations. This survey explores recent advancements in optimizing VLMs for edge environments, focusing on model compression techniques, including pruning, quantization, knowledge distillation, and specialized hardware solutions that enhance efficiency. We provide a detailed discussion of efficient training and fine-tuning methods, edge deployment challenges, and privacy considerations. Additionally, we discuss the diverse applications of lightweight VLMs across healthcare, environmental monitoring, and autonomous systems, illustrating their growing impact. By highlighting key design strategies, current challenges, and offering recommendations for future directions, this survey aims to inspire further research into the practical deployment of VLMs, ultimately making advanced AI accessible in resource-limited settings.
>
---
#### [replaced 050] HandS3C: 3D Hand Mesh Reconstruction with State Space Spatial Channel Attention from RGB images
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2405.01066v4](http://arxiv.org/pdf/2405.01066v4)**

> **作者:** Zixun Jiao; Xihan Wang; Zhaoqiang Xia; Lianhe Shao; Quanli Gao
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Reconstructing the hand mesh from one single RGB image is a challenging task because hands are often occluded by other objects. Most previous works attempt to explore more additional information and adopt attention mechanisms for improving 3D reconstruction performance, while it would increase computational complexity simultaneously. To achieve a performance-reserving architecture with high computational efficiency, in this work, we propose a simple but effective 3D hand mesh reconstruction network (i.e., HandS3C), which is the first time to incorporate state space model into the task of hand mesh reconstruction. In the network, we design a novel state-space spatial-channel attention module that extends the effective receptive field, extracts hand features in the spatial dimension, and enhances regional features of hands in the channel dimension. This helps to reconstruct a complete and detailed hand mesh. Extensive experiments conducted on well-known datasets facing heavy occlusions (such as FREIHAND, DEXYCB, and HO3D) demonstrate that our proposed HandS3C achieves state-of-the-art performance while maintaining a minimal parameters.
>
---
#### [replaced 051] CheXGenBench: A Unified Benchmark For Fidelity, Privacy and Utility of Synthetic Chest Radiographs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10496v2](http://arxiv.org/pdf/2505.10496v2)**

> **作者:** Raman Dutt; Pedro Sanchez; Yongchen Yao; Steven McDonagh; Sotirios A. Tsaftaris; Timothy Hospedales
>
> **摘要:** We introduce CheXGenBench, a rigorous and multifaceted evaluation framework for synthetic chest radiograph generation that simultaneously assesses fidelity, privacy risks, and clinical utility across state-of-the-art text-to-image generative models. Despite rapid advancements in generative AI for real-world imagery, medical domain evaluations have been hindered by methodological inconsistencies, outdated architectural comparisons, and disconnected assessment criteria that rarely address the practical clinical value of synthetic samples. CheXGenBench overcomes these limitations through standardised data partitioning and a unified evaluation protocol comprising over 20 quantitative metrics that systematically analyse generation quality, potential privacy vulnerabilities, and downstream clinical applicability across 11 leading text-to-image architectures. Our results reveal critical inefficiencies in the existing evaluation protocols, particularly in assessing generative fidelity, leading to inconsistent and uninformative comparisons. Our framework establishes a standardised benchmark for the medical AI community, enabling objective and reproducible comparisons while facilitating seamless integration of both existing and future generative models. Additionally, we release a high-quality, synthetic dataset, SynthCheX-75K, comprising 75K radiographs generated by the top-performing model (Sana 0.6B) in our benchmark to support further research in this critical domain. Through CheXGenBench, we establish a new state-of-the-art and release our framework, models, and SynthCheX-75K dataset at https://raman1121.github.io/CheXGenBench/
>
---
#### [replaced 052] Real-Time AIoT for UAV Antenna Interference Detection via Edge-Cloud Collaboration
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03055v2](http://arxiv.org/pdf/2412.03055v2)**

> **作者:** Jun Dong; Jintao Cheng; Jin Wu; Chengxi Zhang; Shunyi Zhao; Xiaoyu Tang
>
> **摘要:** In the fifth-generation (5G) era, eliminating communication interference sources is crucial for maintaining network performance. Interference often originates from unauthorized or malfunctioning antennas, and radio monitoring agencies must address numerous sources of such antennas annually. Unmanned aerial vehicles (UAVs) can improve inspection efficiency. However, the data transmission delay in the existing cloud-only (CO) artificial intelligence (AI) mode fails to meet the low latency requirements for real-time performance. Therefore, we propose a computer vision-based AI of Things (AIoT) system to detect antenna interference sources for UAVs. The system adopts an optimized edge-cloud collaboration (ECC+) mode, combining a keyframe selection algorithm (KSA), focusing on reducing end-to-end latency (E2EL) and ensuring reliable data transmission, which aligns with the core principles of ultra-reliable low-latency communication (URLLC). At the core of our approach is an end-to-end antenna localization scheme based on the tracking-by-detection (TBD) paradigm, including a detector (EdgeAnt) and a tracker (AntSort). EdgeAnt achieves state-of-the-art (SOTA) performance with a mean average precision (mAP) of 42.1% on our custom antenna interference source dataset, requiring only 3 million parameters and 14.7 GFLOPs. On the COCO dataset, EdgeAnt achieves 38.9% mAP with 5.4 GFLOPs. We deployed EdgeAnt on Jetson Xavier NX (TRT) and Raspberry Pi 4B (NCNN), achieving real-time inference speeds of 21.1 (1088) and 4.8 (640) frames per second (FPS), respectively. Compared with CO mode, the ECC+ mode reduces E2EL by 88.9%, increases accuracy by 28.2%. Additionally, the system offers excellent scalability for coordinated multiple UAVs inspections. The detector code is publicly available at https://github.com/SCNU-RISLAB/EdgeAnt.
>
---
#### [replaced 053] Learning Class Prototypes for Unified Sparse Supervised 3D Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21099v2](http://arxiv.org/pdf/2503.21099v2)**

> **作者:** Yun Zhu; Le Hui; Hang Yang; Jianjun Qian; Jin Xie; Jian Yang
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Both indoor and outdoor scene perceptions are essential for embodied intelligence. However, current sparse supervised 3D object detection methods focus solely on outdoor scenes without considering indoor settings. To this end, we propose a unified sparse supervised 3D object detection method for both indoor and outdoor scenes through learning class prototypes to effectively utilize unlabeled objects. Specifically, we first propose a prototype-based object mining module that converts the unlabeled object mining into a matching problem between class prototypes and unlabeled features. By using optimal transport matching results, we assign prototype labels to high-confidence features, thereby achieving the mining of unlabeled objects. We then present a multi-label cooperative refinement module to effectively recover missed detections through pseudo label quality control and prototype label cooperation. Experiments show that our method achieves state-of-the-art performance under the one object per scene sparse supervised setting across indoor and outdoor datasets. With only one labeled object per scene, our method achieves about 78%, 90%, and 96% performance compared to the fully supervised detector on ScanNet V2, SUN RGB-D, and KITTI, respectively, highlighting the scalability of our method. Code is available at https://github.com/zyrant/CPDet3D.
>
---
